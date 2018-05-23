# train.py
import time, sys, os, logging, copy
from sklearn.model_selection import train_test_split
import numpy as np
# customized ftns 
from sequencing import get_sequences, get_minibatches, iterate_minibatches, subsample_sequence
from model import SinglePolicy
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
logging.getLogger("tensorflow").setLevel(logging.WARNING)

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

# ===============================================================================================
# train_all_single_policies =====================================================================
# ===============================================================================================
def train_all_single_policies(game_data, hyper_params, models_path):
    # get all the hyper-params and setup parameters
    train_p = hyper_params['train_percentage']
    batch_size = hyper_params['batch_size']
    sequence_length = hyper_params['sequence_length']
    overlap = hyper_params['overlap']
    state_size = hyper_params['state_size']
    input_dim = hyper_params['input_dim']
    learning_rate = hyper_params['learning_rate']
    n_epoch = hyper_params['n_epoch']
    printn = hyper_params['printn']
    use_model = hyper_params['use_model'] 
    use_peepholes = hyper_params['use_peepholes'] 
    rate = hyper_params['dropout_rate']
    policies = hyper_params['policies']
    horizons = hyper_params['horizons']

    logging.info('Training with setup: \n{}'.format(hyper_params))

    for policy in policies:
        logging.info('Wroking on policy {}'.format(policy))
        # first get the right data
        # pad short sequence and chunk long sequence with overlaps
        train, target = get_sequences(game_data, policy, sequence_length, overlap)
        # create random train and test split
        train_game, test_game, train_target, test_target = train_test_split(train, target, train_size=train_p, random_state=42)
        train_game, test_game, train_target, test_target = np.copy(train_game), np.copy(test_game), np.copy(train_target), np.copy(test_target) 
        logging.info('train len: {} | test shape: {}'.format(len(train_game), len(test_game)))

        # create model
        model = SinglePolicy(policy_number=policy, use_model=use_model, use_peepholes=use_peepholes, state_size=state_size, 
                             batch_size=batch_size, input_dim=input_dim, output_dim=2, rate=rate,
                             learning_rate=learning_rate, seq_len=sequence_length-1, l1_weight_reg=False)
        # starts training
        # look-ahead horizon
        t_int = time.time()                                             # overall initial time   
        train_step = 0                                                  # this is used to record number of steps for training
        valid_step = 0                                                  # steops for validation
        # gradually increase the training roll-out horizon, in hope to reduce drifting errors
        for k in horizons:
            logging.info('Horizon {0:} {1:}'.format(k, '='*10))
            # epochs
            for epoch in range(n_epoch):
                epoch_loss = 0.                                         # make a note of the epoch loss
                n_train_batch = len(train_game)//batch_size             # number of train batches
                t1 = time.time()                                        # beginning time of each epoch
                # iterate through all the batches
                for batch in iterate_minibatches(train_game, train_target, batch_size, shuffle=True):
                    train_xi_original, train_yi_original = batch
                    # if there's look-ahead horizon (k = 0 is just regular rnn trianing)
                    if k != 0:
                        # look-ahead
                        pred = model.train_forward_pass(train_xi_original)
                        train_xi_updated = copy.deepcopy(train_xi_original)
                        for h in range(1, k+1):                         # the steps within each horizon
                            # index selection: https://github.com/numpy/numpy/issues/5574
                            update_ind = np.ix_(range(len(train_xi_updated)), range(h, sequence_length-1, h+1), [policy*2, policy*2+1])
                            train_xi_updated[update_ind] = pred[0][:, range(h-1, sequence_length-2, h+1), :]
                            if h != k:                                  # no need to make a prediction if no more is needed
                                pred = model.train_forward_pass(train_xi_updated)
                        pred, loss, _, train_sum = model.train_backward_pass(train_xi_updated, train_yi_original)
                    # else k = 0 regular rnn training
                    else:
                        pred, loss, _, train_sum = model.train_backward_pass(train_xi_original, train_yi_original)
                    # write to tensorboard summary
                    model.train_writer.add_summary(train_sum, train_step)
                    epoch_loss += loss/n_train_batch
                    train_step += 1
                # print out info
                if epoch % printn == 0:
                    n_val_batch = len(test_game)//batch_size            # number of validation batches
                    t2 = time.time()
                    valid_loss = 0
                    for test_batch in iterate_minibatches(test_game, test_target, batch_size, shuffle=False):
                        val_xi, val_yi = test_batch
                        val_l, valid_sum = model.validate_forward_pass(val_xi, val_yi)

                        model.valid_writer.add_summary(valid_sum, valid_step)
                        valid_loss += val_l/n_val_batch
                        valid_step += printn
                    logging.info('Epoch {0:<4d} | loss: {1:<8.2f} | time took: {2:<.2f}s '
                        '| validation loss: {3:<8.2f}'.format(epoch, epoch_loss, (t2-t1), valid_loss))
                        
            logging.info('Total time took: {0:<.2f}hrs'.format((time.time()-t_int)/60/60))

        # save model
        model.save_model(models_path)
        logging.info('Done saving model for policy {}'.format(policy))