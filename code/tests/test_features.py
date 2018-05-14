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

    def test_flatten(self):
        half_court = 94/2.
        court_width = 50.
        # corresponding features col index
        player_x_ind = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        player_y_ind = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        bball_x_ind = [20]
        bball_y_ind = [21]
        bball_z_ind = [22]
        qtr_ind = [23]
        time_left_ind = [24]
        sc_ind = [25]

        def check_features(f_moment):
            # print(type(f_moment))
            tol = 5.7
            # a) the player positions shall all be greater than zero
            # note: tol here was originally tested for 0, but there can be cases where the player runs off court
            self.assertTrue(sum(f_moment[player_x_ind] >= -tol) == 10 and sum(f_moment[player_x_ind] <= half_court)==10,
                            msg = '\033[91m player x coordinates: {}\033[00m'.format(f_moment[player_x_ind]))
            self.assertTrue(sum(f_moment[player_y_ind] >= -tol) == 10 and sum(f_moment[player_y_ind] <= court_width+tol)==10,
                            msg = '\033[91m player y coordinates: {}\033[00m'.format(f_moment[player_y_ind]))
            # b) bball positions
            self.assertTrue(f_moment[bball_x_ind] >= -tol and f_moment[bball_x_ind] <= half_court + tol, 
                            msg='\033[91m bball x: {}\033[00m'.format(f_moment[bball_x_ind]))
            self.assertTrue(f_moment[bball_y_ind] >= 0 and f_moment[bball_y_ind] <= court_width + tol, 
                            msg='\033[91m bball y: {}\033[00m'.format(f_moment[bball_y_ind]))
            self.assertTrue(f_moment[bball_z_ind] >= 0, msg='bball z: {}'.format(f_moment[bball_z_ind]))
            # c) quarter number
            self.assertTrue(f_moment[qtr_ind] >= 1 and f_moment[qtr_ind] <= 4.,
                            msg='\033[91m quarter number: {}\033[00m'.format(f_moment[qtr_ind]))
            # d) time left to the end of the period in seconds (12 mins per period)
            self.assertTrue(f_moment[time_left_ind] >= 0 and f_moment[sc_ind] <= 12.*60, 
                            msg='\033[91m time left: {}\033[00m'.format(f_moment[time_left_ind]))
            # e) shot clock
            self.assertTrue(f_moment[sc_ind] >= 0 and f_moment[sc_ind] <= 24., 
                            msg='\033[91m shot clock: {}\033[00m'.format(f_moment[sc_ind]))

        event_length_th = 30
        for k, e in enumerate(self.event_dfs):
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_halfcourt(df, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.reorder_teams(df, self.game_ids[k])
            df = pd.DataFrame({'moments': result})

            flattened, team_ids =  preprocessing.flatten_moments(df)
            # team_id
            # 1) the team_ids should not be the same but share the same several front digits
            [self.assertTrue(team_id[0] != team_id[1], msg='\033[91m team ids: {} | {}\033[00m'.format(team_id[0], team_id[1])) for team_id in team_ids]
            [self.assertTrue(str(team_id[0])[:8] == str(team_id[0])[:8] == '16106127',\
                             msg='\033[91m team ids: {} | {}\033[00m'.format(team_id[0], team_id[1])) for team_id in team_ids]
            # 2) check the values from the features 
            [check_features(j) for i in flattened for j in i]


    def test_static_features(self):
        # index related to polar displacement with bball
        r_ball_ind = list(range(26, 36))
        cos_ball_ind = list(range(36, 46))
        sin_ball_ind = list(range(46, 56))
        theta_ball_ind = list(range(56, 66))
        # hoop
        r_hoop_ind = list(range(66, 76))
        cos_hoop_ind = list(range(76, 86))
        sin_hoop_ind = list(range(86, 96))
        theta_hoop_ind = list(range(96, 106))
        diag = np.sqrt(50**2 + 94**2)

        def check_static(f_moment):
            # a1) the player displacements to the ball
            self.assertTrue(sum(f_moment[r_ball_ind] > 0) == 10 and sum(f_moment[r_ball_ind] <= diag)==10,
                            msg = '\033[91m player r ball displacements: {}\033[00m'.format(f_moment[r_ball_ind]))
            # a2) the player displacements to the hoop
            self.assertTrue(sum(f_moment[r_hoop_ind] > 0) == 10 and sum(f_moment[r_hoop_ind] <= diag)==10,
                            msg = '\033[91m player r hoop displacements: {}\033[00m'.format(f_moment[r_hoop_ind]))

            # b1) the player cos to the ball
            self.assertTrue(sum(f_moment[cos_ball_ind] >= -1) == 10 and sum(f_moment[cos_ball_ind] <= 1)==10,
                            msg = '\033[91m player r ball cos: {}\033[00m'.format(f_moment[cos_ball_ind]))
            # b2) the player cos to the hoop
            self.assertTrue(sum(f_moment[cos_hoop_ind] >= -1) == 10 and sum(f_moment[cos_hoop_ind] <= 1)==10,
                            msg = '\033[91m player r hoop cos: {}\033[00m'.format(f_moment[cos_hoop_ind]))

            # c1) the player sin to the ball
            self.assertTrue(sum(f_moment[sin_ball_ind] >= -1) == 10 and sum(f_moment[sin_ball_ind] <= 1)==10,
                            msg = '\033[91m player r ball sine: {}\033[00m'.format(f_moment[sin_ball_ind]))
            # c2) the player sin to the hoop
            self.assertTrue(sum(f_moment[sin_hoop_ind] >= -1) == 10 and sum(f_moment[sin_hoop_ind] <= 1)==10,
                            msg = '\033[91m player r hoop sine: {}\033[00m'.format(f_moment[sin_hoop_ind]))

            # d1) the player theta to the ball
            self.assertTrue(sum(f_moment[theta_ball_ind] >= 0) == 10 and sum(f_moment[theta_ball_ind] <= np.pi)==10,
                            msg = '\033[91m player r ball theta: {}\033[00m'.format(f_moment[theta_ball_ind]))
            # d2) the player sin to the ball
            self.assertTrue(sum(f_moment[theta_hoop_ind] >= 0) == 10 and sum(f_moment[theta_hoop_ind] <= np.pi)==10,
                            msg = '\033[91m player r hoop theta: {}\033[00m'.format(f_moment[theta_hoop_ind]))

            # e1) cos^2 + sin^2 = 1 ball
            e_tol = 1e-5
            self.assertTrue(sum((f_moment[cos_ball_ind]**2 + f_moment[sin_ball_ind]**2 - 1) < e_tol) == 10,
                msg = '\033[91m sin^2 + cos^2 =1 ball: {}\033[00m'.format(f_moment[cos_ball_ind]**2 + f_moment[sin_ball_ind]**2))
            # e2) hoop
            self.assertTrue(sum((f_moment[cos_hoop_ind]**2 + f_moment[sin_hoop_ind]**2 - 1) < e_tol) == 10,
                msg = '\033[91m sin^2 + cos^2 =1 hoop: {}\033[00m'.format(f_moment[cos_hoop_ind]**2 + f_moment[sin_hoop_ind]**2))


        event_length_th = 30
        for k, e in enumerate(self.event_dfs):
            result, _ = preprocessing.remove_non_eleven(e, event_length_th, verbose=True)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_shotclock(e, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.chunk_halfcourt(df, event_length_th, verbose=False)
            df = pd.DataFrame({'moments': result})

            result = preprocessing.reorder_teams(df, self.game_ids[k])
            df = pd.DataFrame({'moments': result})

            flattened, _ =  preprocessing.flatten_moments(df)
            df = pd.DataFrame({'moments': flattened})

            result = preprocessing.create_static_features(df)
            [check_static(i[j]) for i in result for j in range(i.shape[0])]














if __name__ == '__main__':
    unittest.main()