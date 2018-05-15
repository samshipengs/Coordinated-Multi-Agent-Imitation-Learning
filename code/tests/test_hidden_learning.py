import unittest
import pandas as pd
import numpy as np
import copy
import sys
sys.path.append("..")
import preprocessing 
import utilities
import hidden_role_learning

class TestHiddenLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # directories
        main_dir = '../../'
        game_dir = main_dir+'data/'
        Data = utilities.LoadData(main_dir, game_dir)
        # models_path = './models/' 

        cls.game_ids = ['0021500196', '0021500024']
        event_dfs = [pd.DataFrame(Data.load_game(g)['events']) for g in cls.game_ids]

        results = []
        hsls = []
        event_length_th = 30        
        for k, e in enumerate(event_dfs):
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_halfcourt(df, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.reorder_teams(df, cls.game_ids[k])
            df = pd.DataFrame({'moments': result})

            flattened, team_ids =  preprocessing.flatten_moments(df)
            df = pd.DataFrame({'moments': flattened})

            static_result = preprocessing.create_static_features(df)
            df = pd.DataFrame({'moments': copy.deepcopy(static_result)})
            fs = 1/25.
            dynamic_result = preprocessing.create_dynamic_features(df, fs)

            OHE = preprocessing.OneHotEncoding()
            result = OHE.add_ohs(dynamic_result, team_ids)
            df = pd.DataFrame({'moments': result})

            results.append(result)
            hsls.append(hidden_role_learning.HiddenStructureLearning(df))
        
        cls.results = results
        cls.hsls = hsls

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_hidden(self):
        half_court = 94/2.
        court_width = 50.
        tol = 5.7
        diag = np.sqrt(50**2 + 94**2)
        for k in range(len(self.game_ids)):
            result = self.results[k]
            HSL = self.hsls[k]
            # test for the feature index obtained
            for player in range(10):
                print('player', player, end='\r')
                player_features_ind, features_ind = HSL.find_features_ind(player)
                self.assertEqual(len(player_features_ind), 12)
                pxy_ind = player_features_ind[:2]
                polar_bball_ind = player_features_ind[2:6]
                polar_hoop_ind = player_features_ind[6:10]

                for ms in result:
                    nrows = len(ms)
                    # x,y
                    self.assertTrue(sum(ms[:, pxy_ind[0]] >= -tol) == nrows and sum(ms[:, pxy_ind[0]] <= half_court)==nrows,
                                    msg = '\033[91m player x coordinates: {}\033[00m'.format(ms[:, pxy_ind[0]][ms[:, pxy_ind[0]] < -tol or ms[:, pxy_ind[0]] > half_court]))
                    self.assertTrue(sum(ms[:, pxy_ind[1]] >= -tol) == nrows and sum(ms[:, pxy_ind[1]] <= court_width+tol)==nrows,
                                    msg = '\033[91m player y coordinates: {}\033[00m'.format(ms[:, pxy_ind[1]][ms[:, pxy_ind[1]] < -tol or ms[:, pxy_ind[1]] > half_court]))

                     # a1) the player displacements to the ball
                    self.assertTrue(sum(ms[:, polar_bball_ind[0]] > 0) == nrows and sum(ms[:, polar_bball_ind[0]] <= diag)==nrows,
                                    msg = '\033[91m player r ball displacements: {}\033[00m'.format(ms[:, polar_bball_ind[0]]))
                    # a2) the player displacements to the hoop
                    self.assertTrue(sum(ms[:, polar_hoop_ind[0]] > 0) == nrows and sum(ms[:, polar_hoop_ind[0]] <= diag)==nrows,
                                    msg = '\033[91m player r hoop displacements: {}\033[00m'.format(ms[:, polar_hoop_ind[0]]))

                    # b1) the player cos to the ball
                    self.assertTrue(sum(ms[:, polar_bball_ind[1]] >= -1) == nrows and sum(ms[:, polar_bball_ind[1]] <= 1)==nrows,
                                    msg = '\033[91m player r ball cos: {}\033[00m'.format(ms[:, polar_bball_ind[1]]))
                    # b2) the player cos to the hoop
                    self.assertTrue(sum(ms[:, polar_hoop_ind[1]] >= -1) == nrows and sum(ms[:, polar_hoop_ind[1]] <= 1)==nrows,
                                    msg = '\033[91m player r hoop cos: {}\033[00m'.format(ms[:, polar_hoop_ind[1]]))

                    # c1) the player sin to the ball
                    self.assertTrue(sum(ms[:, polar_bball_ind[2]] >= -1) == nrows and sum(ms[:, polar_bball_ind[2]] <= 1)==nrows,
                                    msg = '\033[91m player r ball sine: {}\033[00m'.format(ms[:, polar_bball_ind[2]]))
                    # c2) the player sin to the hoop
                    self.assertTrue(sum(ms[:, polar_hoop_ind[2]] >= -1) == nrows and sum(ms[:, polar_hoop_ind[2]] <= 1)==nrows,
                                    msg = '\033[91m player r hoop sine: {}\033[00m'.format(ms[:, polar_hoop_ind[2]]))

                    # d1) the player theta to the ball
                    self.assertTrue(sum(ms[:, polar_bball_ind[3]] >= 0) == nrows and sum(ms[:, polar_bball_ind[3]] <= np.pi)==nrows,
                                    msg = '\033[91m player r ball theta: {}\033[00m'.format(ms[:, polar_bball_ind[3]]))
                    # d2) the player sin to the ball
                    self.assertTrue(sum(ms[:, polar_hoop_ind[3]] >= 0) == nrows and sum(ms[:, polar_hoop_ind[3]] <= np.pi)==nrows,
                                    msg = '\033[91m player r hoop theta: {}\033[00m'.format(ms[:, polar_hoop_ind[3]]))

                    # e1) cos^2 + sin^2 = 1 ball
                    e_tol = 1e-5
                    self.assertTrue(sum((ms[:, polar_bball_ind[1]]**2 + ms[:, polar_bball_ind[2]]**2 - 1) < e_tol) == nrows,
                        msg = '\033[91m sin^2 + cos^2 =1 ball: {}\033[00m'.format(ms[:, polar_bball_ind[1]]**2 + ms[:, polar_bball_ind[2]]**2))
                    # e2) hoop
                    self.assertTrue(sum((ms[:, polar_hoop_ind[1]]**2 + ms[:, polar_hoop_ind[2]]**2 - 1) < e_tol) == nrows,
                        msg = '\033[91m sin^2 + cos^2 =1 hoop: {}\033[00m'.format(ms[:, polar_hoop_ind[1]]**2 + ms[:, polar_hoop_ind[2]]**2))



















if __name__ == '__main__':
    unittest.main()