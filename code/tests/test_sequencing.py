import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
import preprocessing 
import utilities


class TestPreProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # directories
        main_dir = '../../'
        game_dir = main_dir+'data/'
        Data = utilities.LoadData(main_dir, game_dir)
        # models_path = './models/' 

        cls.game_ids = ['0021500196', '0021500024']
        cls.event_dfs = [pd.DataFrame(Data.load_game(g)['events']) for g in cls.game_ids]

    def setUp(self):
        pass

    def tearDown(self):
        pass


    # def test_subsample(self):
    #     test_seq = [np.random.rand(int(np.random.randint(2, 10, size=1)), 3) for _ in range(10)]
    #     # the sequence should share the 

    















if __name__ == '__main__':
    unittest.main()