# features.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import pandas as pd
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# NOT TESTED YET

def flatten_moments(events_df):
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
 

def create_static_features(events_df):
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
            r = np.sqrt(disp[:,0]**2 + disp[:, 1]**2)               # r 
            cos_theta = disp[:, 0]/r                                # costheta
            sin_theta = disp[:, 1]/r                                # sintheta
            theta = np.arccos(cos_theta)                            # theta
            return np.concatenate((r, cos_theta, sin_theta, theta))
        return np.concatenate((moment, disp_(player_xy, b_xy), disp_(player_xy, hoop_xy)))
    df['enriched'] = df.moments.apply(lambda ms: np.vstack([create_static_features_(m) for m in ms]))
    return df['enriched'].values
    

def create_dynamic_features(events_df, fs):
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


class HiddenStructureLearning:
    def __init__(self, events_df):
        self.df = events_df.copy()
        self.defend_players = list(range(5))
        self.offend_players = list(range(5, 10))
        
    def find_features_ind_(self, player):
        assert player < 10
        pxy_ind = [player*2, player*2+1]
        bball_xy_ind = [2*10, 2*10+1, 2*10+2]
        qtr_ind = [23]
        time_left_ind = [24]
        sc_ind = [25]
        polar_bball_ind = [26+player*4, 26+player*4+1, 26+player*4+2, 26+player*4+3]
        polar_hoop_ind = [66+player*4, 66+player*4+1, 66+player*4+2, 66+player*4+3]
        pvxy_ind = [106+player*2, 106+player*2+1]
        bball_vxy_ind = [126, 127, 128]
        player_features_ind = pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind
#         features_ind = np.array(pxy_ind + bball_xy_ind + qtr_ind + time_left_ind + sc_ind + polar_bball_ind \
#                      + polar_hoop_ind + pvxy_ind + bball_vxy_ind)
        # features_ind = np.array(pxy_ind + bball_xy_ind + sc_ind + polar_bball_ind \
        #                         + polar_hoop_ind + pvxy_ind + bball_vxy_ind)
        features_ind = np.array(pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind)
        return player_features_ind, features_ind
    
    def create_hmm_input_(self, player_inds):
        event = self.df.moments.values
        X = np.concatenate([np.concatenate([ms[:, self.find_features_ind_(player)[1]] for ms in event], axis=0) \
                            for player in player_inds], axis=0)
        lengths = np.concatenate([[len(ms) for ms in event] for _ in range(len(player_inds))],
                                 axis=0)
        assert len(event[0]) == lengths[0]
        assert len(event[-1]) == lengths[-1]
        return X, lengths
    
    def train_hmm_(self, player_inds, verbose=True, random_state=42):
        from hmmlearn import hmm
        assert len(player_inds) == 5 # defend and offend players each are five
        X, lengths = self.create_hmm_input_(player_inds=player_inds)
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
                'state_sequence': state_sequence.reshape(5, -1), 
                'state_sequence_prob': [state_sequence_prob[i:i+n_samples//5] for i in range(0, n_samples, n_samples//5)], 
                'cmeans': cmeans}
    
    def assign_roles(self, player_inds, mode='cosine'):
        result = self.train_hmm_(player_inds=player_inds)
        if mode == 'euclidean':
            ed = distance.cdist(result['X'], result['cmeans'], 'euclidean')
        elif mode == 'cosine':
            ed = distance.cdist(result['X'], result['cmeans'], 'cosine')

        assert len(player_inds) == 5
        n = len(ed)//5 # number of sequences for each players
        assert len(ed) % 5 == 0 # it should be divisibe by number of players
        
        def assign_ind_(cost):
            row_ind, col_ind = linear_sum_assignment(cost)
            return col_ind
        
        role_assignments = np.array([assign_ind_(ed[np.arange(5)*n + i]) for i in range(n)])
        return role_assignments, result
    
    def reorder_moment(self):
        defend_role_assignments, defend_result = self.assign_roles(player_inds=self.defend_players)
        offend_role_assignments, offend_result = self.assign_roles(player_inds=self.offend_players)
        
        original = copy.deepcopy(self.df.moments.values)
        reordered = copy.deepcopy(self.df.moments.values)
        def reorder_moment_(players, original, reordered, role_assignments, offset):# offset is to map the reordered index back to original range for offense players
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
                        p_ind = self.find_features_ind_(p)[0]
                        # get the player feature index corresponding to the reordered role
                        re_p_ind = self.find_features_ind_(ra_i[j][k]+offset)[0]
                        reordered[i][j][re_p_ind] = original[i][j][p_ind]
            return reordered
        reordered_defend = copy.deepcopy(reorder_moment_(self.defend_players, original, reordered, defend_role_assignments, 0))
        reordered_all = copy.deepcopy(reorder_moment_(self.offend_players, original, reordered_defend, offend_role_assignments, 5))
        return reordered_all


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


# def order_moment_ra(moments, role_assignments, components=5, n=5, n_ind=4):
#     '''
#         moments: list of momnets e.g. [(38, 20), (15, 20), ..., ()]
#         role_assignments: list of assignments
#         components: number of components 
#         n: numbner of players
#         n_ind: features for individual player
#     '''
#     # reorder moments based on HMM and la
#     def unstack_role(role):
#         '''map the given role to the 10 index i.e. 5 players times 2 x,y coordinates
#         '''
#         repeats = np.repeat(role*n_ind, [n_ind]*n, axis=1).copy() # 2 for x,y coordinates
#         for i in range(n-2):
#             repeats[:, range(i+1, n_ind*n, n_ind)] += i+1
#         return repeats
    

#     droles = [unstack_role(i) for i in role_assignments]
#     # reorder the moments
#     ro_single_game = []
#     for i in range(len(moments)):
#         ro_i = []
#         for j in range(len(moments[i])):
#             slots = np.zeros(n_ind*components)
#             for k, v in enumerate(droles[i][j]):
#                 slots[v] = moments[i][j][k]
#             ro_i.append(slots)
#         ro_single_game.append(np.array(ro_i))
#     return ro_single_game



# class EmissionVis:
#     ''' visualizing learned emission (single Gaussian) for each components
#         learned through HMM    
#     '''
#     def __init__(self, ncomp, cmeans, cvars):
#         self.ncomp = ncomp
#         self.cmeans = cmeans
#         self.cvars = cvars
    
#     def gaussian_superposition(self, means_arr, vars_arr, mix_p):
#         ''' means_arr: n_mix x feature_dim
#             vars_arr: n_mix x feature_dim
#         '''
#         #Create grid and multivariate normal
#         nx = 1000
#         ny = 500
#         x = np.linspace(0,94,nx)
#         y = np.linspace(0,50,ny)

#         X, Y = np.meshgrid(x,y)
#         pos = np.empty(X.shape + (2,))
#         pos[:, :, 0] = X; pos[:, :, 1] = Y
        
#         def single_gaussian_(mu, var):
#             #Parameters to set
#             mu_x, mu_y = mu[0], mu[1]
#             if mix_p != None:
#                 var_x, var_y = var[0], var[1]
#                 rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])
#             else:
#                 rv = multivariate_normal([mu_x, mu_y], var)
#             return rv.pdf(pos)
#         n_mix = means_arr.shape[0]
#         if mix_p != None:
#             sp = np.sum([single_gaussian_(means_arr[i], vars_arr[i])*mix_p[i] for i in range(n_mix)],axis=0)
#         else:
#             sp = single_gaussian_(means_arr, vars_arr)
#         return X, Y, sp, nx, ny

#     def plot_3d_gaussian(self, X,Y,Z):
#         #Make a 3D plot
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         ax.set_zlabel('Z axis')
#         plt.show()

#     def plot(self):
#         # plot all the 3ds
#         # gmms = model
#         nrows, ncols = 3,3
#         # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8))
#         # figsize=(10,8)
#         plt.figure(figsize=(10,8))
#         n_state = 0
#         for i in range(nrows):
#             for j in range(ncols):
#                 if n_state >= self.ncomp:
#                     break
#                 ax = plt.subplot(nrows, ncols, n_state+1)
#                 X, Y, Z, nx, ny = self.gaussian_superposition(self.cmeans[n_state], self.cvars[n_state], mix_p=None)# gmms.weights_[n_state])
#                 df = pd.DataFrame(Z, index=range(ny), columns=range(nx))
#                 sns.heatmap(df, ax=ax, xticklabels=False, yticklabels=False).set_title(str(n_state+1))
#                 n_state += 1


# class RoleAssignment:
#     ''' main class for assigning hiden rows based on HMM and linear assignment'''
#     def __init__(self, n_iter, verbose):
#         self.n_iter = n_iter
#         self.verbose = verbose

#     def train_hmm(self, X, lengths, n_comp, n_mix):
#         # model = hmm.GMMHMM(n_components=n_comp, n_mix=n_mix, verbose=True, n_iter=30)
#         model = hmm.GaussianHMM(n_components=n_comp, verbose=self.verbose, n_iter=self.n_iter, random_state=42)
#         model.fit(X, lengths)
        
#         # get the means 
#         cmeans = model.means_
#         covars = model.covars_
        
#         # predict the state
#         state_sequence = model.predict(X, lengths)
#         # unstack s.t. each row contains sequence for each of the players
#         state_sequence_ = state_sequence.reshape(5, -1).T
#         return state_sequence_, cmeans, covars, model

#     def assign_roles(self, all_moments_, all_moments, cmeans, event_lengths):
#         # compute the distance for each of the players (all_moments),
#         # to each of the component means (cmeans)
#         ed = distance.cdist(all_moments_, cmeans, 'euclidean')

#         n = len(ed)//5 # number of sequences
#         assert n == all_moments.shape[0] 

#         def assign_index_(cm):
#             ''' Find the roles/index the players from 0 to 4 belongs to.
#                 cm is the cost matrix: 
#                     e.g. 7x5 where 7 is the number of roles,
#                     and 5 is the number of players
#                 i is the loop index

#                 # A problem instance is described by a matrix C, where each C[i,j]
#                 # is the cost of matching vertex i of the first partite set (a “worker”) 
#                 # and vertex j of the second set (a “job”).
#                 # The goal is to find a complete assignment of workers to jobs of minimal cost.
#                 # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
#             '''
#             row_ind, col_ind = linear_sum_assignment(cm)
#             assignment = sorted(list(zip(row_ind, col_ind)), key=lambda x: x[1]) 
#             return [j[0] for j in assignment]

#         role_assignments = np.array([assign_index_(ed[np.arange(5)*n + i].T) for i in range(n)])
        
#         # stack back to role for each seq
#         role_assignments_seq = []
#         start_id = 0
#         for i in event_lengths:
#             role_assignments_seq.append(role_assignments[start_id:start_id+i])
#             start_id += i
            
#         return role_assignments, role_assignments_seq

    
