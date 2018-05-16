# features.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn import hmm


# NOT TESTED YET
# ===================================================================
class HiddenStructureLearning:
    def __init__(self, events_df):
        self.df = events_df.copy()
        self.defend_players = list(range(5))
        self.offend_players = list(range(5, 10))
        
    # =================================
    # find_features_ind ===============
    # =================================
    def find_features_ind(self, player):
        assert player < 10
        pxy_ind = [player*2, player*2+1]
        bball_xy_ind = [2*10, 2*10+1, 2*10+2]
        qtr_ind = [23]
        time_left_ind = [24]
        sc_ind = [25]
        polar_bball_ind = [26+player, 26+player+10, 26+player+20, 26+player+30]
        polar_hoop_ind = [66+player, 66+player+10, 66+player+20, 66+player+30]
        pvxy_ind = [106+player*2, 106+player*2+1]
        bball_vxy_ind = [126, 127, 128]

        player_features_ind = pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind
#         features_ind = np.array(pxy_ind + bball_xy_ind + qtr_ind + time_left_ind + sc_ind + polar_bball_ind \
#                      + polar_hoop_ind + pvxy_ind + bball_vxy_ind)
        # features_ind = np.array(pxy_ind + bball_xy_ind + sc_ind + polar_bball_ind \
        #                         + polar_hoop_ind + pvxy_ind + bball_vxy_ind)
        features_ind = np.array(pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind)
        return player_features_ind, features_ind
    
    # =================================
    # create_hmm_input ================
    # =================================
    def create_hmm_input(self, player_inds):
        event = self.df.moments.values
        # create X: array-like, shape (n_samples, n_features)
        X = np.concatenate([ms[:, self.find_features_ind(player)[1]] for player in player_inds for ms in event], axis=0)
        # create lengths : array-like of integers, shape (n_sequences, )
        lengths = [len(ms) for player in player_inds for ms in event]

        # lengths = np.concatenate([[len(ms) for ms in event] for _ in range(len(player_inds))],
        #                          axis=0)
        assert len(event[0]) == lengths[0]
        assert len(event[len(event)//2]) == lengths[len(lengths)//len(player_inds)//2], '{}'.format([len(event[len(event)//2]), lengths[len(lengths)//2]])
        assert len(event[-1]) == lengths[-1]
        assert X.shape[1] == 12 # the number of features used to determine hidden roles
        return X, lengths
    
    def train_hmm(self, player_inds, verbose=True, random_state=42):
        assert len(player_inds) == 5 # defend and offend players each are five
        X, lengths = self.create_hmm_input(player_inds=player_inds)
        model = hmm.GaussianHMM(n_components=5, 
                                covariance_type='diag', 
                                n_iter=50, 
                                random_state=random_state,
                                verbose=verbose)
        model.fit(X, lengths)
        state_sequence = model.predict(X, lengths)
        state_sequence_prob = model.predict_proba(X, lengths) # (n_samples, n_components)
        n_samples, _ = state_sequence_prob.shape
        cmeans = model.means_
        return {'X': X,
                'lengths': lengths,
                'state_sequence': state_sequence.reshape(5, -1),  # the shape here can be done because the original input is ordered by players chunk
                'state_sequence_prob': [state_sequence_prob[i:i+n_samples//5] for i in range(0, n_samples, n_samples//5)], 
                'cmeans': cmeans}
    
    def assign_roles(self, player_inds, mode='euclidean'):
        result = self.train_hmm(player_inds=player_inds)
        if mode == 'euclidean':
            ed = distance.cdist(result['X'], result['cmeans'], 'euclidean')
        elif mode == 'cosine':
            ed = distance.cdist(result['X'], result['cmeans'], 'cosine')

        assert len(player_inds) == 5
        n = len(ed)//5 # number of sequences for each players
        assert len(ed) % 5 == 0 # it should be divisibe by number of players
        
        role_assignments = np.array([self.assign_ind(ed[np.arange(5)*n + i]) for i in range(n)])
        return role_assignments, result

    def assign_ind(self, cost):
        row_ind, col_ind = linear_sum_assignment(cost)
        return col_ind
    
    def reorder_moment(self):
        defend_role_assignments, defend_result = self.assign_roles(player_inds=self.defend_players)
        offend_role_assignments, offend_result = self.assign_roles(player_inds=self.offend_players)
        
        original = copy.deepcopy(self.df.moments.values)
        reordered = copy.deepcopy(self.df.moments.values)
        # offset is to map the reordered index back to original range for offense players
        def reorder_moment_(players, original, reordered, role_assignments, offset):
            divider = 0
            lengths = [len(m) for m in original]
            # iteratve through each moments length
            for i in range(len(lengths)):
                # grab the corresponding moments' reordered roles
                ra_i = role_assignments[divider:divider+lengths[i]]
                # update the next starting index
                divider += lengths[i]
                # iterate through each moment in the current moments
                for j in range(lengths[i]):
                    # iterate through each players
                    for k, p in enumerate(players):
                        # get the current player feature index
                        p_ind = self.find_features_ind(p)[0]
                        # get the player feature index corresponding to the reordered role
                        re_p_ind = self.find_features_ind(ra_i[j][k]+offset)[0]
                        reordered[i][j][re_p_ind] = original[i][j][p_ind]
            return reordered
        reordered_defend = copy.deepcopy(reorder_moment_(self.defend_players, original, reordered, defend_role_assignments, 0))
        reordered_all = copy.deepcopy(reorder_moment_(self.offend_players, original, reordered_defend, offend_role_assignments, 5))
        return reordered_all
