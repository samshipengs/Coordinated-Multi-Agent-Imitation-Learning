# customized ftns 
from helpers import *
from utilities import *
from model import *
import time

def train_all_single_policies(batch_size, sequence_length, train_game, train_target,
                              test_game, test_target, models_path):
    policies = range(7) # in total 7 different roles
    for policy in policies:
        print('Wroking on policy', policy)
        # create model
        model = SinglePolicy(policy_number=policy, state_size=128, batch_size=batch_size, input_dim=62, output_dim=2,
                            learning_rate=0.01, seq_len=sequence_length-1)
        # starts training
        printn = 100    # how many epochs we print
        n_epoch = int(1e2)
        # look-ahead horizon
        horizon = [0, 2]
        t_int = time.time()
        train_step = 0
        valid_step = 0
        for k in horizon:
            print('Horizon {0:} {1:}'.format(k, '='*10))

            for epoch in range(n_epoch):
                epoch_loss =0.
                # number of train batches
                n_train_batch = len(train_game)//batch_size
                t1 = time.time()
                for batch in iterate_minibatches(train_game, train_target, batch_size, shuffle=False):
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
                    print('Epoch {0:<4d} | loss: {1:<8.2f} | time took: {2:<.2f}s '
                        '| validation loss: {3:<8.2f}'.format(epoch, epoch_loss, (t2-t1), valid_loss))
                        
            print('Total time took: {0:<.2f}hrs'.format((time.time()-t_int)/60/60))

        # save model
        model.save_model(models_path)
        print('Done saving model for', policy, '\n')