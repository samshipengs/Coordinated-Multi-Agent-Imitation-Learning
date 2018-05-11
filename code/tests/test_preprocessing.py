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
        # 1) check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            result_len = [len(i) for i in result]
            count = sum(np.array(result_len) >= event_length_th)
            self.assertEqual(count, len(result))
            # 2) check type is list
            self.assertEqual(type(result[0]), list)

    def test_chunk_shotclock(self):
        # 1) check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(df, event_length_th, verbose=False)
            result_len = [len(i) for i in result]
            count = sum(np.array(result_len) >= event_length_th)
            self.assertEqual(count, len(result))
            self.assertEqual(type(result[0]), list)

            for i in result:
                # 2) check if None is in the shot clock from result
                self.assertNotIn(None, [j[3] for j in i])
                # 3) check if the shot clock is in right order
                self.assertTrue(i[0][3] > i[-1][3])
                # 4) check if the first two and last two shotclock value are different
                self.assertTrue(i[0][3] != i[1][3])
                self.assertTrue(i[-1][3] != i[-2][3])

    def test_chunk_halfcourt(self):
        # 1) check if all event satisfies required length
        event_length_th = 30
        for e in self.event_dfs:
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
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
                    # 2) the players must either be on the left court or the right
                    self.assertTrue(sum(np.array(j[5])[1:, 2]>=half_court) == 10 or sum(np.array(j[5])[1:, 2]<=half_court) == 10)

    def test_reorder_teams(self):
        # now we want to reorder the team position based on meta data
        court_index = pd.read_csv('../meta_data/court_index.csv')
        court_index = dict(zip(court_index.game_id, court_index.court_position))
        # 1) 42 teams mapping
        self.assertEqual(len(court_index), 42)

        half_court = 94/2.
        event_length_th = 25
        for k, e in enumerate(self.event_dfs):
            # remove non eleven
            result, team_ids = preprocessing.remove_non_eleven(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})
            # chunk at 24s, None or stopped clocks
            result = preprocessing.chunk_shotclock(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})
            # chunk based on half court
            result = preprocessing.chunk_halfcourt(df, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})
            # 2) the homeid is always the first 5 players
            home_id, away_id = team_ids['home_id'], team_ids['away_id']
            for i in result:
                for j in i:
                    self.assertEqual(j[5][1][0], home_id)
                    self.assertEqual(j[5][-1][0], away_id)
            
            n_events = len(result)
            result = preprocessing.reorder_teams(df, self.game_ids[k])
            df = pd.DataFrame({'moments': result})
            
            # 3) the returned number of events should not change
            self.assertEqual(n_events, len(result))
            for i in result:
                for j in i:
                    # 4) all players should be already normalized to left half court
                    self.assertEqual(sum(np.array(j[5])[1:, 2]<=half_court), 10)
            # hardcode test cases:
            if k == '0021500196':
                correct = [[1610612761, 2449, 9.8067, 28.92548, 0.0],
                           [1610612761, 201960, 24.3734, 21.43063, 0.0],
                           [1610612761, 200768, 6.86594, 20.4174, 0.0],
                           [1610612761, 201942, 17.89619, 25.21197, 0.0],
                           [1610612761, 202687, 10.1885, 24.11858, 0.0],
                           [1610612746, 1718, 34.33032, 10.36747, 0.0],
                           [1610612746, 200755, 32.26942, 26.02228, 0.0],
                           [1610612746, 101108, 18.86048, 7.87816, 0.0],
                           [1610612746, 201599, 35.95785, 20.65878, 0.0],
                           [1610612746, 201933, 25.66964, 40.3966, 0.0]]
                self.assertListEqual(result[0][0][5][1:], correct)
                





















if __name__ == '__main__':
    unittest.main()