# features.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
import pandas as pd


# TESTED 
# ======
# =================================================================
# flatten_moments =================================================
# =================================================================
def flatten_moments(events_df):
    ''' This changes the nested list that represents single frame 
        to a 1-D array.
     '''
    df = events_df.copy()
    def flatten_moment(moment):
        m = np.array(moment[5])
        features = np.concatenate((m[1:11, 2:4].reshape(-1),    # x,y of all 10 players 
                                   m[0][2:5],                   # basketball x,y,z 
                                   np.array([moment[0]]),       # quarter number 
                                   np.array([moment[2]]),       # time in seconds left to the end of the period
                                   np.array([moment[3]])))      # shot clock 
        return features
    
    def get_team_ids(moment):
        m = np.array(moment[5])
        team_id1 = set(m[1:6, 0])
        team_id2 = set(m[6:11, 0])
        assert len(team_id1) == len(team_id2) == 1
        assert team_id1 != team_id2
        return [list(team_id1)[0], list(team_id2)[0]]
        
        
    df['flattened'] = df.moments.apply(lambda ms: [flatten_moment(m) for m in ms])
    df['team_ids'] = df.moments.apply(lambda ms: get_team_ids(ms[0])) # just use the first one to determine        
    
    return df['flattened'].values, df['team_ids'].values


# =================================================================
# create_static_features ==========================================
# =================================================================
def create_static_features(events_df):
    ''' Provide some static features:
            displacement, cos, sin and theta from each player to the ball, hoop 
    ''' 
    df = events_df.copy()
    def create_static_features_(moment):
        ''' moment: flatten moment i.e. (25=10*2+3+2,)'''
        # distance of each players to the ball
        player_xy = moment[:10*2]
        b_xy = moment[10*2:10*2+2]
        hoop_xy = np.array([3.917, 25])

        def disp_(pxy, target):
            # dispacement to bball
            disp = pxy.reshape(-1, 2) - np.tile(target, (10, 1))
            assert disp.shape[0] == 10
            r = np.sqrt(disp[:,0]**2 + disp[:, 1]**2)               # r 
            cos_theta = disp[:, 0]/r                                # costheta
            sin_theta = disp[:, 1]/r                                # sintheta
            theta = np.arccos(cos_theta)                            # theta
            assert sum(r>0) == 10
            return np.concatenate((r, cos_theta, sin_theta, theta))

        return np.concatenate((moment, disp_(player_xy, b_xy), disp_(player_xy, hoop_xy)))
    # vertical stack s.t. now each event i.e. a list of moments becomes an array
    # where each row is a frame (moment)
    df['enriched'] = df.moments.apply(lambda ms: np.vstack([create_static_features_(m) for m in ms]))
    return df['enriched'].values


# =================================================================
# create_dynamic_features =========================================
# =================================================================
def create_dynamic_features(events_df, fs):
    ''' Add velocity for players x, y direction and bball's x,y,z direction 
    '''
    df = events_df.copy()
    def create_dynamic_features_(moments, fs):
        ''' moments: (moments length, n existing features)'''
        pxy = moments[:, :23] # get the players x,y and basketball x,y,z coordinates
        next_pxy = np.roll(pxy, -1, axis=0) # get next frame value
        vel = ((next_pxy - pxy)/fs)[:-1, :] # the last velocity is not meaningful
        # when we combine this back to the original features, we shift one done,
        # i.e. [p1, p2, ..., pT] combine [_, p2-p1, ...., pT-pT_1]
        # the reason why we shift is that we don't want to leak next position info
        return np.column_stack([moments[1:, :], vel])
    df['enriched'] = df.moments.apply(lambda ms: create_dynamic_features_(ms, fs))
    return df['enriched'].values 


# =================================================================
# OneHotEncoding ==================================================
# =================================================================
class OneHotEncoding:
    '''
        Perform one hot encoding on the team id, use mapping 
        from the id_team.csv file (or you an pass your own)
    '''
    def __init__(self, cat=None):
        cat = pd.read_csv('./meta_data/id_team.csv')
        # binary encode
        # ensure uniqueness
        assert sum(cat.team_id.duplicated()) == 0
        self.mapping = dict(zip(cat.team_id, range(0, len(cat)))) # temporarily just one hot encode two teams
        # self.mapping = {1610612741:0, 1610612761:1}

    def encode(self, teams):
        nb_classes = len(self.mapping)
        targets = np.array([self.mapping[int(i)] for i in teams])
        one_hot_targets = np.eye(nb_classes)[targets]
        # print(one_hot_targets)
        return one_hot_targets.reshape(-1)

    def add_ohs(self, events, team_ids):
        return [np.column_stack((events[i], np.tile(self.encode(team_ids[i]), (len(events[i]), 1)))) for i in range(len(events))]