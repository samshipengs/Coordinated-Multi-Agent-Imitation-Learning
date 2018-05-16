import unittest
import pandas as pd
import numpy as np

import sys, logging
sys.path.append("..")
import preprocessing 
import utilities
from sequencing import get_sequences, get_minibatches, iterate_minibatches, subsample_sequence


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


class TestPreProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # directories
        main_dir = '../../'
        game_dir = main_dir+'data/'
        Data = utilities.LoadData(main_dir, game_dir)
        # models_path = './models/' 

        cls.game_ids = ['0021500196', '0021500024']
        # cls.event_dfs = [pd.DataFrame(Data.load_game(g)['events']) for g in cls.game_ids]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # test_subsample ============================================================================
    def test_subsample(self):
        # first create a test sample, a list of events(array of moments)
        test_seq = [np.random.rand(int(np.random.randint(2, 10, size=1)), 3) for _ in range(10)]
        # subsample 2 
        test_result = [i[range(0, len(i), 2)] for i in test_seq]
        [self.assertEqual(test_result[i].tolist(), subsample_sequence(test_seq, 2)[i].tolist()) for i in range(len(test_seq))]
        # subsample 0
        [self.assertEqual(test_seq[i].tolist(), subsample_sequence(test_seq, 0)[i].tolist()) for i in range(len(test_seq))]

    # test_get_sequences ========================================================================
    def test_get_sequences(self):
        # first create a test sample, a list of events(array of moments)
        large, small = 20, 10
        n_seq = 500
        n_features = 4
        test_game = [np.random.rand(int(np.random.randint(small, large, size=1)), n_features) for _ in range(n_seq)]

        # get result
        test_policy = 0
        # test_sequence_length = 2
        # test_overlap = 1
        for test_sequence_length in [5, 7, 10, 15, 17, 20, 33]:
            for test_overlap in range(1, test_sequence_length):
                # logging.info('test_sequence_length {0:} | test_overlap {1:}'.format(test_sequence_length, test_overlap))
                test_train, test_target = get_sequences(test_game, test_policy, test_sequence_length, test_overlap)
                # print(test_game[0], '\n\n', test_train[:3], '\n\n', test_target[:3])
                # the length of each sequence should now be the required length (subtract 1)
                [self.assertEqual(len(i), test_sequence_length-1, msg='{0} out of {1}'.format(i, len(test_train)))  for i in test_train]
                # the number of features should not be changed
                [self.assertEqual(i.shape[1], n_features, msg=i)  for i in test_train]

    # 
    def test_iterate_minibatches(self):
        # first create a test sample, a list of events(array of moments)
        large, small = 10, 6
        n_seq = 500
        n_features = 4
        test_game = [np.random.rand(int(np.random.randint(small, large, size=1)), n_features) for _ in range(n_seq)]

        # get result
        batch_size = 3
        test_policy = 0
        # test_sequence_length = 2
        # test_overlap = 1
        for test_sequence_length in [5, 7, 10, 15, 17, 20, 30]:
            for test_overlap in range(1, test_sequence_length):
                # logging.info('test_sequence_length {0:} | test_overlap {1:}'.format(test_sequence_length, test_overlap))
                test_train, test_target = get_sequences(test_game, test_policy, test_sequence_length, test_overlap)

                p = 0.8 # train percentage
                divider = int(len(test_train)*p)
                train_game, _ = np.copy(test_train[:divider]), np.copy(test_train[divider:])
                train_target, _ = np.copy(test_target[:divider]), np.copy(test_target[divider:])
                # print(train_game, '\n\n', train_target)
                for batch in iterate_minibatches(train_game, train_target, batch_size, test_sequence_length, shuffle=False):
                    if batch != None:
                        train_xi, train_yi = batch
                        self.assertEqual(train_xi.shape[0], batch_size, msg='train_xi.shape: {}'.format(train_xi.shape))
                        self.assertEqual(train_xi.shape[1], n_features, msg='train_xi.shape: {}'.format(train_xi.shape))
                        self.assertListEqual(train_yi[0][0].tolist(), train_xi[0][1][:2].tolist())
                        self.assertListEqual(train_yi[-1][0].tolist(), train_xi[-1][1][:2].tolist())



if __name__ == '__main__':
    unittest.main()