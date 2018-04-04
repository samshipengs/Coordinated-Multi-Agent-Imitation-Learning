from __future__ import print_function
#from __future__ import division

import time

import numpy as np
from math import sqrt
import random
import sys
import subprocess
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, BatchNormalization
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adagrad, Adam, SGD
#from keras.models import load_model
import keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines
import matplotlib.patches as patches

from multiprocessing import Pool
import multiprocessing

path = '/home/leh/Preprocessing/Progressive_Training/'

#activeRole = 'lcm'
activeRole = ['lcm','lcb', 'lb' , 'lw', 'lf', 'rcm', 'rcb', 'rb', 'rw', 'rf']
#activeRole = ['lcm','lcb']
#activeRole = ['lcb', 'lcm']
model_list = [path+'weights_progressive_Double_FullBatch_rollsteps_10_'+role+'.h5' for role in activeRole]

roleOrderDefense = ['gk','rb','rcb','lcb','lb','rw','rcm','lcm','lw','rf','lf']
roleOrderAttack = ['gk','rb','rcb','lcb','lb','rw','rcm','lcm','lw','rf','lf']

global roleOrderList
roleOrderList = [roleOrderDefense.index(role) for role in activeRole]

#global roleOrder
#roleOrder = roleOrderDefense.index(activeRole)


def chunks(X, length):
    return [X[0+i:length+i] for i in range(0, X.shape[0], length)]

def roll_out(params):
    goalPosition = [1.0, 0]

    prev_feature_vector, legacy_feature_vector, pos_prediction, roleOrder = params
    prev_feature_vector = np.concatenate((prev_feature_vector[:roleOrder*13], np.zeros(3),prev_feature_vector[roleOrder*13:]   ))
    legacy_feature_vector = np.concatenate((legacy_feature_vector[:roleOrder*13], np.zeros(3),legacy_feature_vector[roleOrder*13:]   ))

    legacy_current = legacy_feature_vector[0:390] # 308 = 28*11
    ball_current = legacy_feature_vector[390:399] # 317 = 28*11 +9

    legacy = legacy_current.reshape(30,13)
    ball = ball_current[0:2]
    new_matrix = np.zeros((22,13))

    role_long = legacy[roleOrder]
    teammateList = range(11)
    teammateList.remove(roleOrder)

    # fix role vector
    mainRoleIndex = roleOrderList.index(roleOrder)
    role_long[0:3] = np.zeros(3)
    role_long[3:5] = pos_prediction[2*mainRoleIndex:(2*mainRoleIndex+2)]
    role_long[5:7] = role_long[3:5] - prev_feature_vector[roleOrder*13+3:(roleOrder*13+5)] # velocity = current pos - prev pos

    role = role_long[3:5]
    role_long[7] = sqrt((role[0]-goalPosition[0])**2+(role[1]-goalPosition[1])**2 )
    if role_long[7] !=0:
        role_long[8] = (role[0]-goalPosition[0]) / role_long[7]
        role_long[9] = (role[1]-goalPosition[1]) / role_long[7]
    else:
        role_long[8] = 0.0
        role_long[9] = 0.0

    role_long[10] = sqrt((role[0]-ball[0])**2+(role[1]-ball[1])**2 )
    if role_long[10] != 0:
        role_long[11] = (role[0]-ball[0]) / role_long[10]
        role_long[12] = (role[1]-ball[1]) / role_long[10]
    else:
        role_long[11] = 0.0
        role_long[12] = 0.0
    new_matrix[roleOrder] = role_long

    # fix all teammates vector
    for teammate in teammateList:
        player = legacy[teammate]
        if teammate in roleOrderList: # if the teammate is one of the active players
            teammateRoleIndex = roleOrderList.index(teammate)
            player[3:5] = pos_prediction[2*teammateRoleIndex:(2*teammateRoleIndex+2)]
            currentPos = player[3:5]
            player[5:7] = currentPos - prev_feature_vector[teammate*13+3:(teammate*13+5)] # velocity = current pos - prev pos

            player[7] = sqrt((currentPos[0]-goalPosition[0])**2+(currentPos[1]-goalPosition[1])**2 )
            if player[7] !=0:
                player[8] = (currentPos[0]-goalPosition[0]) / player[7]
                player[9] = (currentPos[1]-goalPosition[1]) / player[7]
            else:
                player[8] = 0.0
                player[9] = 0.0

            player[10] = sqrt((currentPos[0]-ball[0])**2+(currentPos[1]-ball[1])**2 )
            if player[10] != 0:
                player[11] = (currentPos[0]-ball[0]) / player[10]
                player[12] = (currentPos[1]-ball[1]) / player[10]
            else:
                player[11] = 0.0
                player[12] = 0.0                    
        
        currentPos = player[3:5]

        player[0] = sqrt((currentPos[0]-role[0])**2+(currentPos[1]-role[1])**2 )
        if player[0] != 0:
            player[1] = (currentPos[0]-role[0]) / player[0]
            player[2] = (currentPos[1]-role[1]) / player[0]
        else:
            player[1] = prev_feature_vector[teammate*13+1]
            player[2] = prev_feature_vector[teammate*13+2]

        new_matrix[teammate] = player

    for opponent in range(11,22):
        player = legacy[opponent]
        currentPos = player[3:5]
        player[0] = sqrt((currentPos[0]-role[0])**2+(currentPos[1]-role[1])**2 )
        if player[0] != 0:
            player[1] = (currentPos[0]-role[0]) / player[0]
            player[2] = (currentPos[1]-role[1]) / player[0]
        else:
            player[1] = prev_feature_vector[opponent*13+1]
            player[2] = prev_feature_vector[opponent*13+2]
        new_matrix[opponent] = player

    teammates_distance = new_matrix[:11,0].copy()
    opponents_distance = new_matrix[11:22,0].copy()
    k = 4
    k_nearest_teammate = teammates_distance.argsort()[0:(k+1)]
    k_nearest_opponent = 11+opponents_distance.argsort()[0:k] # add 11 for offset
    # remove the role itself out of the nearest teammate list
    k_nearest_teammate = k_nearest_teammate[np.nonzero(k_nearest_teammate-roleOrder)]
    # Now look for the closest 3 teammates and duplicate the vector
    new_matrix = np.vstack((new_matrix, new_matrix[k_nearest_teammate], new_matrix[k_nearest_opponent])) ## Combine all ordered player_state with the nearest teammates and nearest opponents
    
    new_feature_vector = np.concatenate((new_matrix.flatten(), ball_current ))

    ## delete the 3 zeros
    new_feature_vector = np.concatenate((new_feature_vector[:roleOrder*13], new_feature_vector[roleOrder*13+3:] ))
    
    return new_feature_vector

if __name__ == '__main__':


    #####################################
    #### LOAD THE NORMAL DATA

    data = [np.load(path+role+'_data_CurrWithVel_training.npy') for role in activeRole]
    
    ## holder for the raw invert data
    data_invert = [np.load(path+role+'_data_CurrWithVel_training_invert.npy') for role in activeRole]

    totalTimeSteps = 50

    endOfSequenceMarker = np.nonzero(data[0][:,0])[0]
    numSequence = endOfSequenceMarker.shape[0]
    beginOfSequenceMarker = np.zeros(numSequence).astype(np.int64)
    for index in range(numSequence-1):
        beginOfSequenceMarker[index+1] = endOfSequenceMarker[index]+1

    sequenceLength = endOfSequenceMarker - beginOfSequenceMarker +1

    subsequenceLength = totalTimeSteps+1 # this will be the number of time steps later in the lstm model + 1
    overlapWindow = 26

    def chunking_window(subsequenceLength, overlapWindow):
        startSequence = []
        endSequence = []
        for index in range(numSequence):
            startSubSequence = []
            endSubSequence = []
            start = beginOfSequenceMarker[index]
            end = endOfSequenceMarker[index]
            while end >= (start+subsequenceLength-1):
                endSubSequence.append(end)
                startSubSequence.append(end-subsequenceLength+1)
                end = end - overlapWindow
            startSubSequence.reverse()
            endSubSequence.reverse()
            startSequence = startSequence + startSubSequence
            endSequence = endSequence + endSubSequence

        sequenceMarker = zip(startSequence, endSequence)

        return sequenceMarker

    sequenceMarker = chunking_window(subsequenceLength, overlapWindow)
    
    fullColumnIndex = np.arange(401) # 401 = 2+ 30*13+9 columns in the original training data file
    #excludedColumns = np.concatenate((np.array([0,1]),np.arange(2+roleOrder*13,2+roleOrder*13+3) )) # first two columns are time stamps
    #retainedColumns = np.delete(fullColumnIndex, excludedColumns)

    
    X_train_all = []
    Y_train_all = []

    for roleIndex in range(len(activeRole)):

        X_subset_data = []
        Y_subset_data = []
        excludedColumns = np.concatenate((np.array([0,1]),np.arange(2+roleOrderList[roleIndex]*13,2+roleOrderList[roleIndex]*13+3) )) # first two columns are time stamps
        retainedColumns = np.delete(fullColumnIndex, excludedColumns)    

        for index in sequenceMarker:
            dataSegment = data[roleIndex][index[0]:index[1]+1,:].copy()
            x_segment = dataSegment[:-1,retainedColumns] 
            y_segment = dataSegment[1:,(2+roleOrderList[roleIndex]*13+3):(2+roleOrderList[roleIndex]*13+5) ] # the role's position located in column 3 and 4, offset by the role order* number of feature per player

            X_subset_data.append(x_segment)
            Y_subset_data.append(y_segment)
        
        ## Load the invert data
        
        for index in sequenceMarker:
            dataSegment = data_invert[roleIndex][index[0]:index[1]+1,:].copy()
            x_segment = dataSegment[:-1,retainedColumns] 
            y_segment = dataSegment[1:,(2+roleOrderList[roleIndex]*13+3):(2+roleOrderList[roleIndex]*13+5) ] # the role's position located in column 3 and 4, offset by the role order* number of feature per player

            X_subset_data.append(x_segment)
            Y_subset_data.append(y_segment)
        
        ########################################################################
        #### COMBINE THE TWO BATCHES OF DATA

        X_train = np.vstack(X_subset_data)
        Y_train = np.vstack(Y_subset_data)

        X_train_all.append(X_train)
        Y_train_all.append(Y_train)

    featurelen = retainedColumns.shape[0]
    outputlen = 2 
    numOfPrevSteps = 1 # We are only looking at the most recent character each time. 
    #########################################################################
    print('Formatting Data')

    print('total training frame is ', X_train.shape[0])
    batches = chunks(X_train, totalTimeSteps)
    batchSize = 2200
    #batchSize = len(batches)
    batches = batches[0:len(batches)/batchSize*batchSize]
    offSet = len(batches) / batchSize
    #########################################################################
    ## Further processing. Clipping the maximum velocity norm here

    #velNorm = sqrt(Y[:,0]**2+Y[:,1]**2)
    #velLimit = np.percentile(velNorm, 95)  ## Clip the limit of velocity to 95 percentile of all velocities, to get rid of outliers # artificially remove the limit

    #X = np.zeros([batchSize, totalTimeSteps , featurelen]) 
    X_all = []
    Y_all = []
    for X_train in X_train_all:
        X = np.zeros([offSet*batchSize, totalTimeSteps , featurelen]) 
        for b in range(len(batches)):
            for r in range(totalTimeSteps):
                currentFeature = X_train[r + b*totalTimeSteps]
                X[b][r][:] = currentFeature
        X_all.append(X)
    
    for Y_train in Y_train_all:
        Y = np.zeros([offSet*batchSize, totalTimeSteps , outputlen])
        for b in range(len(batches)):
            for r in range(totalTimeSteps):
                currentPrediction = Y_train[r + b*totalTimeSteps]
                Y[b][r][:] = currentPrediction
        Y_all.append(Y)
    

    X_original = [X_all[index].copy() for index in range(len(activeRole))]

    ##################################
    #### Load the test data#####
    #data = np.load(activeRole+'_data_CurrWithVel_test1.npy')
    data = [np.load(path+role+'_data_CurrWithVel_test1.npy') for role in activeRole]

    endOfSequenceMarker = np.nonzero(data[0][:,0])[0]
    numSequence = endOfSequenceMarker.shape[0]
    beginOfSequenceMarker = np.zeros(numSequence).astype(np.int64)
    for index in range(numSequence-1):
        beginOfSequenceMarker[index+1] = endOfSequenceMarker[index]+1

    sequenceLength = endOfSequenceMarker - beginOfSequenceMarker +1

    includedSequence = np.where(sequenceLength>=50)[0] ## expect all sequences to have length of at least 50, since this is how the test set was formed

    maxlen = sequenceLength.max()
    totalTimeSteps_test = maxlen

    X_test_all = []
    for roleIndex in range(len(activeRole)):
        X_test_data = []
        Y_test_data = []

        excludedColumns = np.concatenate((np.array([0,1]),np.arange(2+roleOrderList[roleIndex]*13,2+roleOrderList[roleIndex]*13+3) )) # first two columns are time stamps
        retainedColumns = np.delete(fullColumnIndex, excludedColumns)    

        for index in includedSequence:
            dataSegment = data[roleIndex][beginOfSequenceMarker[index]:endOfSequenceMarker[index]+1,:].copy()
            x_segment = np.zeros((maxlen, retainedColumns.shape[0]))
            x_segment[:dataSegment.shape[0]] = dataSegment[:,retainedColumns] 

            X_test_data.append(x_segment)

        X_test = np.vstack(X_test_data)
        X_test_all.append(X_test)
    
    batches_test = chunks(X_test, totalTimeSteps_test)
    batchSize_test = 38
    
    X_test_test_all = []
    for X_test in X_test_all:
        X_test_test = np.zeros([batchSize_test, totalTimeSteps_test , featurelen]) 
        for b in range(len(batches_test)):
            for r in range(totalTimeSteps_test):
                currentFeature = X_test[r + b*totalTimeSteps_test]
                X_test_test[b][r][:] = currentFeature
        X_test_test_all.append(X_test_test)

    X_original_test = [X_test_test_all[index].copy() for index in range(len(activeRole))]
    
    ########################################################################
    #### FINISH LOADING THE DATA ####
    ########################################################################


    #############

    """
    init_model = Sequential()
    #init_model.add(BatchNormalization(batch_input_shape=(batchSize, numOfPrevSteps , featurelen) ))
    #init_model.add(LSTM(512 , batch_input_shape=(batchSize, numOfPrevSteps , featurelen), return_sequences=True,  stateful=True))
    init_model.add(LSTM(512 , batch_input_shape=(batchSize, numOfPrevSteps, featurelen), return_sequences=True,  stateful=True))
    init_model.add(LSTM(512 , return_sequences=False,stateful=True))
    init_model.add(Dense (2))
    init_model.add(Activation('linear'))
    init_model.compile(loss='mse', optimizer='rmsprop')
    init_model.reset_states()

    print('starting initializing')
    num_epochs = 5
    for e in range(num_epochs):
        print('epoch - ',e+1)
        #p = sampling_prob[e]

        startTime = time.time()
        training_loss = []
        #loss = init_model.fit(X,Y,nb_epoch = 5, batch_size = batchSize)
        
        for j in range(offSet):
            #for i in range(0,totalTimeSteps-1):
            for i in range(0,totalTimeSteps):
                #loss = init_model.train_on_batch(X[batchSize*j:batchSize*(j+1), numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], np.reshape(Y[batchSize*j:batchSize*(j+1), (i+1)*numOfPrevSteps, :], (batchSize, outputlen)) )
                loss = init_model.train_on_batch(X[batchSize*j:batchSize*(j+1), numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], Y[batchSize*j:batchSize*(j+1), i, :] )
                training_loss.append(loss)

            init_model.reset_states()
        print('Initializing loss: ',sum(training_loss)/len(training_loss))
        
        #print('Initializing loss: ',loss)
        totalTime = time.time() - startTime
        print('Completed epoch in ',totalTime,' seconds')
        print()
    print('Initializing complete')

    init_model.save_weights('init_weights_minibatch1024_10epochs.h5',overwrite = True)
    
    model = Sequential()
    #model.add(BatchNormalization(batch_input_shape=(batchSize, numOfPrevSteps , featurelen) ) )
    model.add(LSTM(512 ,batch_input_shape=(batchSize, numOfPrevSteps , featurelen), return_sequences=True,  stateful=True))
    model.add(LSTM(512 , return_sequences=False,stateful=True))
    model.add(Dense (2))
    model.add(Activation('linear'))
    adagrad = Adagrad(lr=0.005, epsilon=1e-08)
    model.compile(loss='mse', optimizer='adagrad')
    model.load_weights('init_weights_minibatch1024_10epochs.h5') # Load the pretrained model
    model.reset_states()
    """
    
    adagradOpt = Adagrad(lr=0.005, epsilon=1e-08)

    print('Load models...')
    policy = []
    #### Load the model
    for model_name in model_list:
        model = Sequential()
        model.add(LSTM(512 ,return_sequences=True, batch_input_shape=(batchSize, numOfPrevSteps , featurelen), stateful=True))
        model.add(LSTM(512 , return_sequences=False,stateful=True))
        model.add(Dense (2))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adagrad')
        #model.compile(loss='mse', optimizer=adagradOpt)
        model.load_weights(model_name)
        #model.load_weights('init_weights_minibatch1024_10epochs.h5') # Load the pretrained model
        model.reset_states()
        policy.append(model)    
    
    val_policy = []
    #### Load the model
    for index in range(len(activeRole)):
        val_model = Sequential()
        val_model.add(LSTM(512 ,batch_input_shape=(batchSize_test, numOfPrevSteps , featurelen) ,return_sequences=True,  stateful=True))
        val_model.add(LSTM(512 , return_sequences=False,stateful=True))
        val_model.add(Dense (2))
        val_model.add(Activation('linear'))
        val_model.compile(loss='mse', optimizer='adagrad')
        #val_model.compile(loss='mse', optimizer=adagradOpt)
        val_model.reset_states()
        val_policy.append(val_model)

    print('starting training')
    rollout_horizon = [10]
    num_epochs = 100

    #sampling_prob = np.linspace(0,1,num_epochs)[::-1]
    lr_schedule = np.zeros(num_epochs)

    # Set up parallel processing
    numProcess = multiprocessing.cpu_count()
    print('Number of processes ', numProcess)
    pool = Pool(processes = 16)

    for horizon in rollout_horizon:
        best_loss = [10 for index in range(len(activeRole))]
        bestValLoss = 10
        for e in range(num_epochs):
            text_file = open("Output_10Policy_double_2200batch_roll10_overlapWindow25.txt", "a")
            #lr_schedule[e] = model.optimizer.lr.get_value()
            #p = sampling_prob[e]
            print('epoch - ',e+1)
            text_file.write('epoch - %s \n' %(e+1) )
            print('training joint policies - Double rollout horizon 10 - Predict then train - adagrad- batch 2200 - Overlapping window')
            
            #print('learning rate before training ', lr_schedule[e])
            startTime = time.time()
            
            training_loss = [[] for index in range(len(activeRole))] # initialize empty list of list to store training loss
            for j in range(offSet):
                #for i in range(0,totalTimeSteps-1):
                for i in range(0,totalTimeSteps+1-horizon,horizon):                                                                               

                    # roll out horizon times
                    for k in range(horizon):
                        if i+k+1<50:
                            next_prediction_all = []
                            ## Roll out all next step predictions and gather them into one place
                            for index in range(len(activeRole)):
                                next_prediction = policy[index].predict_on_batch(X_all[index][batchSize*j:batchSize*(j+1), (i+k):(i+k+1), :])
                                next_prediction_all.append(next_prediction)
                            ## and then update all the feature vector for the next step, for each active role
                            next_prediction_all = np.hstack(next_prediction_all)
                            for index in range(len(activeRole)):
                                prev_feature = X_all[index][batchSize*j:batchSize*(j+1),i+k,:]
                                legacy_feature = X_all[index][batchSize*j:batchSize*(j+1),i+k+1,:]
                                order = np.empty(batchSize).astype(int)
                                order.fill(roleOrderList[index])
                                params = zip(prev_feature, legacy_feature, next_prediction_all, order)
                                result = pool.map(roll_out, params)
                                #result = map(roll_out, params)
                                X_all[index][batchSize*j:batchSize*(j+1),i+k+1,:] = np.array(result)
                                
                    for index in range(len(activeRole)):
                        ## train the model for the horizon steps
                        for k in range(horizon):
                            loss = policy[index].train_on_batch(X_all[index][batchSize*j:batchSize*(j+1), (i+k):(i+k+1), :], Y_all[index][batchSize*j:batchSize*(j+1), i+k, :] ) 
                            training_loss[index].append(loss)                                                                                                                     


                for index in range(len(activeRole)):
                    policy[index].reset_states()
            for index in range(len(activeRole)):
                print('training loss for role '+activeRole[index]+': ',sum(training_loss[index])/len(training_loss[index]))  
                rolledOutLoss = ((X_all[index][:,:,(roleOrderList[index]*13):(roleOrderList[index]*13+2)] - X_original[index][:,:,(roleOrderList[index]*13):(roleOrderList[index]*13+2)])**2).mean()
                print('rolled out loss for role '+ activeRole[index] +': ', rolledOutLoss)
            

            
            #### True roll out ####
            for index in range(len(activeRole)):
                for i in range(len(policy[index].layers)):
                    val_policy[index].layers[i].set_weights(policy[index].layers[i].get_weights())
                val_policy[index].reset_states()


            for i in range(0,totalTimeSteps_test-1):
                next_prediction_all = []
                ## Roll out all next step predictions and gather them into one place
                for index in range(len(activeRole)):
                    next_prediction = val_policy[index].predict_on_batch(X_test_test_all[index][0:batchSize_test, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :])
                    next_prediction_all.append(next_prediction)
                ## and then update all the feature vector for the next step, for each active role
                next_prediction_all = np.hstack(next_prediction_all)
                for index in range(len(activeRole)):
                    prev_feature = X_test_test_all[index][0:batchSize_test,i,:]
                    legacy_feature = X_test_test_all[index][0:batchSize_test,i+1,:]
                    order = np.empty(batchSize_test).astype(int)
                    order.fill(roleOrderList[index])
                    params = zip(prev_feature, legacy_feature, next_prediction_all, order)
                    result = pool.map(roll_out, params[0:len(batches_test)])
                    X_test_test_all[index][0:len(batches_test),i+1,:] = np.array(result)

            #model.reset_states()

            valLoss = 0
            for index in range(len(activeRole)):
                predPosition = []
                truePosition = []

                for i in includedSequence:
                    predPosition.append(X_test_test_all[index][i,:sequenceLength[i],(roleOrderList[index]*13):(roleOrderList[index]*13+2)])
                    truePosition.append(X_original_test[index][i,:sequenceLength[i],(roleOrderList[index]*13):(roleOrderList[index]*13+2)])
                #rolledOutLoss = ((X_test_test_all[index][:,:,(roleOrderList[index]*13):(roleOrderList[index]*13+2)] - X_original_test[index][:,:,(roleOrderList[index]*13):(roleOrderList[index]*13+2)])**2).mean()
                rolledOutLoss = ((np.vstack(predPosition) - np.vstack(truePosition))**2).mean()
                valLoss = valLoss + rolledOutLoss
                print('True validation loss for role '+ activeRole[index]+':', rolledOutLoss)
                text_file.write('True validation loss for role %s : %s \n' %(activeRole[index], rolledOutLoss))
                if rolledOutLoss < best_loss[index]:
                    best_loss[index] = rolledOutLoss
                policy[index].save_weights('weights_joint10Policy_DoubleOW25_batch2200_adagrad_rollsteps_'+str(horizon)+'_'+activeRole[index]+'_epoch'+str(e+1)+'.h5', overwrite = True)
                
                print('best validation loss so far with rollout '+str(horizon)+' for role '+ activeRole[index] + ': ', best_loss[index])
                text_file.write('best validation loss so far with rollout %s for role %s : %s \n' %(str(horizon), activeRole[index],best_loss[index] ) )
            print()
            print('Total validation loss this round: ', valLoss)
            text_file.write('Total validation loss this round: %s \n' %(valLoss))
            if valLoss < bestValLoss:
                bestValLoss = valLoss
            print('best total validation loss up to this round: ', bestValLoss)
            text_file.write('best total validation loss up to this round: %s \n' %bestValLoss)
            
            ### ENd of true roll out ####

            print()

            totalTime = time.time() - startTime
            print('Completed epoch in ',totalTime,' seconds')
            print()
            text_file.close()

    print('training complete')