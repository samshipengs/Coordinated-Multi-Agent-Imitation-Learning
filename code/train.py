import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('./train_logs/train.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
# ===================================================================================
logger.info('Start loading libraries and modules')
# load all the libraries
import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time

from datetime import datetime
import numpy as np
import pandas as pd
import glob, os, sys, math, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import copy, time, glob, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# customized ftns 
from helpers import *
from utilities import *
from model import rnn_horizon
logger.info('Done loading all libraries and modules')
# ---------------------------------------------------------


# Directories
main_dir = '../'
game_dir = main_dir+'data/'
Data = LoadData(main_dir, game_dir)

# Load raw data =================================================
logger.info('Load raw data')
# %%time
game_id = '0021500463'
game_data = Data.load_game(game_id)
events_df = pd.DataFrame(game_data['events'])
logger.info('raw events shape:', events_df.shape)

# Get some suplementary data =================================================
logger.info('Get suplementary data e.g. court defend and offend, homeid etc')
# its possible that F has similar role as G-f or F-G, we create empty slots to ensure meta order
# ddentify defending and offending runs (this is included in process_moments)
court_index = Data.load_csv('./meta_data/court_index.csv')
court_index = dict(zip(court_index.game_id, court_index.court_position))

# home and visitor ids
homeid = events_df.loc[0].home['teamid']
awayid = events_df.loc[0].visitor['teamid']

# Pre-process ===================================================
logger.info('Pre-processing - filter events, subsample frames, add velocity, '
            're-arrange team order shot clock, filter out event with short moments')
# filter out actions except 1: Make, 2: Miss, 4: Rebound, 6:Personal Foul, 7:Violation
use_event = [1, 2, 4, 6, 7]
discard_event = [3, 5, 8, 9, 10, 12, 13, 18]
events = filter_event_type(events_df, discard_event)
logger.info('After filtering events has shape:', events.shape)
# break up sequences at 24secs shot clock point (or irregular case, e.g. out of bound maybe),
# and obtain the game data
# subsample_factor = 0
# single_game = get_game_data(events, id_role, role_order, court_index, game_id, 
#                             event_threshold=10, subsample_factor=subsample_factor)
# print('Final number of events:', len(single_game))
subsample_factor = 0
single_game = get_game_data_ra(events, court_index, game_id, event_threshold=10, subsample_factor=subsample_factor)
logger.info('Final number of events:', len(single_game))

# get velocity
fs_base = 1./25 # 1/25 sec/frame   or  25 frames/sec
fs = fs_base * subsample_factor if subsample_factor != 0 else fs_base
single_game_velocity = [get_velocity(i, fs) for i in single_game]
# also drop the last row in positions due to the last row drop on velocity
single_game = [i[:-1, :] for i in single_game]
n_events = len(single_game)

# Role assignment and reorder moment =================================================
logger.info('Role assignment and reorder moment')
# first prepare data
n_defend = 5
n_offend = 5
# length for each moment
event_lengths = np.array([len(i) for i in single_game])
# repeat the event_lengths 5 times in order to match the unstack later on with moments
event_lengths_repeat = np.concatenate([event_lengths for _ in range(n_defend)], axis=0)
# all the moments
all_moments = np.concatenate(single_game, axis=0)
all_moments_vel = np.concatenate(single_game_velocity, axis=0) # vel
# we only need the first 5 players x,y coordindates
# defend
all_defend_moments = all_moments[:, :2*n_defend]
# all_defend_moments_vel = all_moments_vel[:, :2*n_defend]
# offend
all_offend_moments = all_moments[:, 2*n_offend:]
# all_offend_moments_vel = all_moments_vel[:, 2*n_offend:]

# flattened
all_defend_moments_ = np.concatenate([all_defend_moments[:, i:i+2] for i in range(0, 2*n_defend, 2)], axis=0)
all_offend_moments_ = np.concatenate([all_offend_moments[:, i:i+2] for i in range(0, 2*n_offend, 2)], axis=0)

# all_defend_moments_vel_ = np.concatenate([all_defend_moments_vel[:, i:i+2] for i in range(0, 2*n_defend, 2)], axis=0)
# all_offend_moments_vel_ = np.concatenate([all_offend_moments_vel[:, i:i+2] for i in range(0, 2*n_offend, 2)], axis=0)

# create hmm model
logger.info('training hmm model')
n_comp = 7
n_mix = None
RA = RoleAssignment()

# train
defend_state_sequence_, defend_means, defend_covs = RA.train_hmm(all_defend_moments_, event_lengths_repeat, n_comp, n_mix)
offend_state_sequence_, offend_means, offend_covs= RA.train_hmm(all_offend_moments_, event_lengths_repeat, n_comp, n_mix)
# get role orders
_, defend_roles = RA.assign_roles(all_defend_moments_, all_defend_moments, defend_means, event_lengths)
_, offend_roles = RA.assign_roles(all_offend_moments_, all_offend_moments, offend_means, event_lengths)

# reorder defend pos and vel
defend_pos = order_moment_ra([i[:, :10] for i in single_game], defend_roles)
defend_vel = order_moment_ra([i[:, :10] for i in single_game_velocity], defend_roles)

# reorder offend pos and vel
offend_pos = order_moment_ra([i[:, 10:] for i in single_game], offend_roles)
offend_vel = order_moment_ra([i[:, 10:] for i in single_game_velocity], offend_roles)

# concate above to create the data for model training
single_game = [np.concatenate([defend_pos[i], offend_pos[i], defend_vel[i], offend_vel[i]], axis=1) for i in range(n_events)]

# Create label, train and test set =========================================================================
logger.info('Create label, train and test set')
sequence_length = 30
overlap = 15
# pad short sequence and chunk long sequence with overlaps
train, target = get_sequences(single_game, sequence_length, overlap)

# create train and test set
p = 0.8 # train percentage
divider = int(len(train)*p)
train_game, test_game = train[:divider], train[divider:]
train_target, test_target = target[:divider], target[divider:]

# Build graph and starts training ==========================================================================
logger.info('Build graph and start training')
tf.reset_default_graph()
# use training start time as the unique naming
train_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logs_path = './train_logs/'

# hyper-parameters
# num_layers = 2
state_size = 128
batch_size = 32
dimx = 56
dimy = 2
learning_rate = 0.01
n_epoch = int(1e3)
true_seq_len = sequence_length-1

# lstm cells
lstm1 = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.)
# lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=0.8)
lstm2 = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.)
# lstm2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, output_keep_prob=0.8)
lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2])

# initial state
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# input placeholders
h = tf.placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32)
X = tf.placeholder(tf.float32, [batch_size, true_seq_len, dimx], name = 'train_input')
Y = tf.placeholder(tf.float32, [batch_size, true_seq_len, dimy], name = 'train_label')
# rnn structure
# output, last_states = rnn_horizon(cell = lstm_cell, 
#                                   initial_state = initial_state, 
#                                   input_ = X,
#                                   batch_size = batch_size,
#                                   seq_lengths = seq_len,
#                                   horizon = h,
#                                   output_dim = dimy)

output1, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                        inputs = X,
                                        sequence_length=seq_len,
                                        initial_state=initial_state)

output = tf.contrib.layers.fully_connected(inputs=output1, num_outputs=dimy)
# output as the prediction
pred = output

# tensorboard's graph visualization more convenient
with tf.name_scope('MSEloss'):
    # loss (also add regularization on params)
    tv = tf.trainable_variables()
    # l2 weight loss
#     regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    # l1 loss
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
    regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, tv)

    loss = tf.losses.mean_squared_error(Y, pred) + regularization_cost
    
    # no weight loss
#     loss = tf.losses.mean_squared_error(Y, pred)

with tf.name_scope('Adam'):
    # optimzier
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
# initialize variables
init = tf.global_variables_initializer()
# create a summary to monitor cost tensor
train_summary = tf.summary.scalar("TrainMSEloss", loss)
valid_summary = tf.summary.scalar("ValidMSEloss", loss)

# session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# initializing the variables
sess.run(init)
# op to write logs to Tensorboard
train_writer = tf.summary.FileWriter(logs_path+'/train'+train_time, graph=tf.get_default_graph())
valid_writer = tf.summary.FileWriter(logs_path+'/valid'+train_time, graph=tf.get_default_graph())

# ===============================================================================================
# start training
printn = int(1e2)    # how many epochs we print
# horizon = [2, 4, 8, 12, 16, 20, 25]
horizon = [1]
t_int = time.time()
train_step = 0
valid_step = 0
for k in horizon:
    # look-ahead horizon
#     seq_len = horizon[k]
#     seq_len =  # because we dropped one when creating targets
    logger.info('Horizon {0:} {1:}'.format(seq_len, '='*10))

    for epoch in range(n_epoch):
        epoch_loss =0.
        # number of train batches
        n_train_batch = len(train_game)//batch_size
        t1 = time.time()
        for batch in iterate_minibatches(train_game, train_target, batch_size, shuffle=False):
            train_xi, train_yi = batch
            p, l, _, train_sum = sess.run([output, loss, opt, train_summary], 
                                          feed_dict={X: train_xi, Y: train_yi, 
                                                     seq_len:true_seq_len,
                                                     h: 2})
            train_writer.add_summary(train_sum, train_step)
            epoch_loss += l/n_train_batch
            train_step += 1
        # print out info
        if epoch%printn ==0:
            # number of validation batches
            n_val_batch = len(test_game)//batch_size
            t2 = time.time()
            valid_loss = 0
            for test_batch in iterate_minibatches(test_game, test_target, batch_size, shuffle=False):
                val_xi, val_yi = test_batch
                val_l, valid_sum = sess.run([loss, valid_summary], 
                                            feed_dict={X: val_xi, Y: val_yi, 
                                                       seq_len:true_seq_len,
                                                       h: 2})

                valid_writer.add_summary(valid_sum, valid_step)
                valid_loss += val_l/n_val_batch
                valid_step += printn
            logger.info('Epoch {0:<4d} | loss: {1:<8.2f} | time took: {2:<.2f}s '
                  '| validation loss: {3:<8.2f}'.format(epoch, epoch_loss, (t2-t1), valid_loss))
                

t_end = time.time()
logger.info('Total time took: {0:<.2f}hrs'.format((time.time()-t_int)/60/60))

# Check model on train set
logger.info('Check model on train set')
# use while loop to make sure the 
train_batches = get_minibatches(train_game, train_target, batch_size, shuffle=False)

check_ind = np.random.randint(0, len(train_game)//batch_size)
logger.info('rand checking index: {0:} out of {1:}'.format(check_ind, len(train_game)//batch_size))

input_xi, output_yi = train_batches
y_pred = sess.run([output], feed_dict={X: input_xi[check_ind], seq_len:true_seq_len, h: 2})#, Y: train_yi, h:2})
y_true = output_yi[check_ind]
    
y_pred = y_pred[0][0].reshape(-1,2)
y_true = y_true[0].reshape(-1,2)

plt.figure(figsize=(15,8))
for k in range(0, len(y_pred)):
    plt.plot(y_pred[:, 0][k], y_pred[:, 1][k], linestyle="None", marker="o", markersize=k, color='g')
    plt.plot(y_true[:, 0][k], y_true[:, 1][k], linestyle="None", marker="o", markersize=k, color='b')

plt.plot(y_pred[:, 0], y_pred[:, 1],'g', y_true[:,0], y_true[:,1], 'b')#, pred_train[:, 0], pred_train[:, 1])
plt.grid(True)
plt.show()

# Check model on test set
logger.info('Check model on test set')
# use while loop to make sure the 
test_batches = get_minibatches(test_game, test_target, batch_size, shuffle=False)

check_ind = np.random.randint(0, len(test_game)//batch_size)
logger.info('rand checking index: {0:} out of {1:}'.format(check_ind, len(test_game)//batch_size))

input_xi, output_yi = test_batches
y_pred = sess.run([output], feed_dict={X: input_xi[check_ind], seq_len:true_seq_len, h: 2})#, Y: train_yi, h:2})
y_true = output_yi[check_ind]
    
y_pred = y_pred[0][0].reshape(-1,2)
y_true = y_true[0].reshape(-1,2)

plt.figure(figsize=(15,8))
for k in range(0, len(y_pred)):
    plt.plot(y_pred[:, 0][k], y_pred[:, 1][k], linestyle="None", marker="o", markersize=k, color='g')
    plt.plot(y_true[:, 0][k], y_true[:, 1][k], linestyle="None", marker="o", markersize=k, color='b')

plt.plot(y_pred[:, 0], y_pred[:, 1],'g', y_true[:,0], y_true[:,1], 'b')#, pred_train[:, 0], pred_train[:, 1])
plt.grid(True)
plt.show()
