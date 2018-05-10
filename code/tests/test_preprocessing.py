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

    def test_remove_non_eleven(self):
        # check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            result_len = [len(i) for i in result]
            count = sum(np.array(result_len) >= event_length_th)
            # print(e[e.event_count < event_length_th]['event_count'])
            self.assertEqual(count, len(result))
            self.assertEqual(type(result[0]), list)

    def test_chunk_shotclock(self):
        # check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(df, event_length_th, verbose=False)
            result_len = [len(i) for i in result]
            count = sum(np.array(result_len) >= event_length_th)
            self.assertEqual(count, len(result))
            self.assertEqual(type(result[0]), list)

            for i in result:
                # check if None is in the shot clock from result
                self.assertNotIn(None, [j[3] for j in i])
                # check if the shot clock is in right order
                self.assertTrue(i[0][3] > i[-1][3])
                # check if the first two and last two sc are same
                self.assertTrue(i[0][3] != i[1][3])
                self.assertTrue(i[-1][3] != i[-2][3])

    def test_chunk_halfcourt(self):
        # check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_halfcourt(df, event_length_th, verbose=False)
            result_len = [len(i) for i in result]
            count = sum(np.array(result_len) >= event_length_th)
            self.assertEqual(count, len(result))
            self.assertEqual(type(result[0]), list)

            half_court = 94/2.
            for i in result:
                for j in i:
                    # the players must either be on the left court or the right
                    # print(sum(np.array(j[5])[1:, 2]>=half_court), sum(np.array(j[5])[1:, 2]<=half_court))
                    self.assertTrue(sum(np.array(j[5])[1:, 2]>=half_court) == 10 or sum(np.array(j[5])[1:, 2]<=half_court) == 10)


if __name__ == '__main__':
    unittest.main()