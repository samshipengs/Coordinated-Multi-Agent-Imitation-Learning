# customized ftns 
from sequencing import get_sequences, get_minibatches, iterate_minibatches, subsample_sequence
import time, sys, logging

import numpy as np
from model import SinglePolicy


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

def train_all_single_policies(single_game, hyper_params, models_path):
    batch_size = hyper_params['batch_size']
    sequence_length = hyper_params['sequence_length']
    overlap = hyper_params['overlap']
    state_size = hyper_params['state_size']
    input_dim = hyper_params['input_dim']
    learning_rate = hyper_params['learning_rate']
    n_epoch = hyper_params['n_epoch']
    use_model = hyper_params['use_model'] 
    logging.info('Training with hyper parameters: \n{}'.format(hyper_params))

    # policies = range(7) # in total 7 different roles
    policies = [0]  # CHANGE
    for policy in policies:
        logging.info('Wroking on policy {}'.format(policy))
        # first get the right data
        # pad short sequence and chunk long sequence with overlaps
        train, target = get_sequences(single_game, policy, sequence_length, overlap)
        # create train and test set
        p = 0.8 # train percentage
        divider = int(len(train)*p)
        train_game, test_game = np.copy(train[:divider]), np.copy(train[divider:])
        train_target, test_target = np.copy(target[:divider]), np.copy(target[divider:])
        logging.info('train len: {} | test shape: {}'.format(len(train_game), len(test_game)))

        # create model
        model = SinglePolicy(policy_number=policy, use_model=use_model, state_size=state_size, 
                             batch_size=batch_size, input_dim=input_dim, output_dim=2,
                             learning_rate=learning_rate, seq_len=sequence_length-1, l1_weight_reg=False)
        # starts training
        printn = 10    # how many epochs we print
        # look-ahead horizon
        horizon = [0]       # CHANGE
        t_int = time.time() 
        train_step = 0
        valid_step = 0
        for k in horizon:
            logging.info('Horizon {0:} {1:}'.format(k, '='*10))

            for epoch in range(n_epoch):
                epoch_loss =0.
                # number of train batches
                n_train_batch = len(train_game)//batch_size
                t1 = time.time()
                for batch in iterate_minibatches(train_game, train_target, batch_size, shuffle=True):
                    train_xi, train_yi = batch
                    p, l, _, train_sum = model.train(train_xi, train_yi, k)
                    model.train_writer.add_summary(train_sum, train_step)
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
                        val_l, valid_sum = model.validate(val_xi, val_yi, k)

                        model.valid_writer.add_summary(valid_sum, valid_step)
                        valid_loss += val_l/n_val_batch
                        valid_step += printn
                    logging.info('Epoch {0:<4d} | loss: {1:<8.2f} | time took: {2:<.2f}s '
                        '| validation loss: {3:<8.2f}'.format(epoch, epoch_loss, (t2-t1), valid_loss))
                        
            logging.info('Total time took: {0:<.2f}hrs'.format((time.time()-t_int)/60/60))

        # save model
        model.save_model(models_path)
        logging.info('Done saving model for policy {}'.format(policy))