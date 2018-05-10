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
from preprocessing import *

class OneHotEncoding:
    '''
        Perform one hot encoding on the team id, use mapping 
        from the id_team.csv file (or you an pass your own)
    '''
    def __init__(self, cat=None):
        cat = pd.read_csv('./meta_data/id_team.csv')
        # binary encode
        # ensure uniqueness
        # assert sum(cat.team_id.duplicated()) == 0
        # self.mapping = dict(zip(cat.team_id, range(0, len(cat)))) # temporarily just one hot encode two teams
        self.mapping = {1610612741:0, 1610612761:1}

    def encode(self, teams):
        nb_classes = len(self.mapping)
        targets = np.array([self.mapping[int(i)] for i in teams])
        one_hot_targets = np.eye(nb_classes)[targets]
        # print(one_hot_targets)
        return one_hot_targets

def get_velocity(event, fs, mode=0):
    ''' 
        event: an array where each row is a moment/frame, columns are the input feature vectors
        fs: time lapse from frame to next frame

        this one appends velocity to the end of all the players positions 

        note: the last column is discarded because theres no velocity info for it
    '''
    pos = event.copy()
    next_pos = np.copy(np.roll(pos,-1, axis=0))
    vel = (next_pos - pos)/fs
    if mode == 0:
        return vel[:-1, :]
    elif mode == 1:
        vel = vel[:-1, :]
        pos = pos[:-1, :] # also drop the last one from postitions to match velocity
        d1, d2 = pos.shape
        combined = np.empty((d1, 2*d2))
        # add the position and velocity data
        start_ind = 0
        for i in range(0, 2*d2, 4): # 2 for 2d (x,y), 2 for (vx, vy)
            combined[:, i:i+2] = pos[:, start_ind:start_ind+2]
            combined[:, i+2:i+4] = vel[:, start_ind:start_ind+2]
            start_ind += 2
        return combined


def order_moment_ra(moments, role_assignments, components=5, n=5, n_ind=4):
    '''
        moments: list of momnets e.g. [(38, 20), (15, 20), ..., ()]
        role_assignments: list of assignments
        components: number of components 
        n: numbner of players
        n_ind: features for individual player
    '''
    # reorder moments based on HMM and la
    def unstack_role(role):
        '''map the given role to the 10 index i.e. 5 players times 2 x,y coordinates
        '''
        repeats = np.repeat(role*n_ind, [n_ind]*n, axis=1).copy() # 2 for x,y coordinates
        for i in range(n-2):
            repeats[:, range(i+1, n_ind*n, n_ind)] += i+1
        return repeats
    

    droles = [unstack_role(i) for i in role_assignments]
    # reorder the moments
    ro_single_game = []
    for i in range(len(moments)):
        ro_i = []
        for j in range(len(moments[i])):
            slots = np.zeros(n_ind*components)
            for k, v in enumerate(droles[i][j]):
                slots[v] = moments[i][j][k]
            ro_i.append(slots)
        ro_single_game.append(np.array(ro_i))
    return ro_single_game



class EmissionVis:
    ''' visualizing learned emission (single Gaussian) for each components
        learned through HMM    
    '''
    def __init__(self, ncomp, cmeans, cvars):
        self.ncomp = ncomp
        self.cmeans = cmeans
        self.cvars = cvars
    
    def gaussian_superposition(self, means_arr, vars_arr, mix_p):
        ''' means_arr: n_mix x feature_dim
            vars_arr: n_mix x feature_dim
        '''
        #Create grid and multivariate normal
        nx = 1000
        ny = 500
        x = np.linspace(0,94,nx)
        y = np.linspace(0,50,ny)

        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        
        def single_gaussian_(mu, var):
            #Parameters to set
            mu_x, mu_y = mu[0], mu[1]
            if mix_p != None:
                var_x, var_y = var[0], var[1]
                rv = multivariate_normal([mu_x, mu_y], [[var_x, 0], [0, var_y]])
            else:
                rv = multivariate_normal([mu_x, mu_y], var)
            return rv.pdf(pos)
        n_mix = means_arr.shape[0]
        if mix_p != None:
            sp = np.sum([single_gaussian_(means_arr[i], vars_arr[i])*mix_p[i] for i in range(n_mix)],axis=0)
        else:
            sp = single_gaussian_(means_arr, vars_arr)
        return X, Y, sp, nx, ny

    def plot_3d_gaussian(self, X,Y,Z):
        #Make a 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def plot(self):
        # plot all the 3ds
        # gmms = model
        nrows, ncols = 3,3
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8))
        # figsize=(10,8)
        plt.figure(figsize=(10,8))
        n_state = 0
        for i in range(nrows):
            for j in range(ncols):
                if n_state >= self.ncomp:
                    break
                ax = plt.subplot(nrows, ncols, n_state+1)
                X, Y, Z, nx, ny = self.gaussian_superposition(self.cmeans[n_state], self.cvars[n_state], mix_p=None)# gmms.weights_[n_state])
                df = pd.DataFrame(Z, index=range(ny), columns=range(nx))
                sns.heatmap(df, ax=ax, xticklabels=False, yticklabels=False).set_title(str(n_state+1))
                n_state += 1


class RoleAssignment:
    ''' main class for assigning hiden rows based on HMM and linear assignment'''
    def __init__(self, n_iter, verbose):
        self.n_iter = n_iter
        self.verbose = verbose

    def train_hmm(self, X, lengths, n_comp, n_mix):
        # model = hmm.GMMHMM(n_components=n_comp, n_mix=n_mix, verbose=True, n_iter=30)
        model = hmm.GaussianHMM(n_components=n_comp, verbose=self.verbose, n_iter=self.n_iter, random_state=42)
        model.fit(X, lengths)
        
        # get the means 
        cmeans = model.means_
        covars = model.covars_
        
        # predict the state
        state_sequence = model.predict(X, lengths)
        # unstack s.t. each row contains sequence for each of the players
        state_sequence_ = state_sequence.reshape(5, -1).T
        return state_sequence_, cmeans, covars, model

    def assign_roles(self, all_moments_, all_moments, cmeans, event_lengths):
        # compute the distance for each of the players (all_moments),
        # to each of the component means (cmeans)
        ed = distance.cdist(all_moments_, cmeans, 'euclidean')

        n = len(ed)//5 # number of sequences
        assert n == all_moments.shape[0]

        def assign_index_(cm):
            ''' Find the roles/index the players from 0 to 4 belongs to.
                cm is the cost matrix: 
                    e.g. 7x5 where 7 is the number of roles,
                    and 5 is the number of players
                i is the loop index

                # A problem instance is described by a matrix C, where each C[i,j]
                # is the cost of matching vertex i of the first partite set (a “worker”) 
                # and vertex j of the second set (a “job”).
                # The goal is to find a complete assignment of workers to jobs of minimal cost.
                # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
            '''
            row_ind, col_ind = linear_sum_assignment(cm)
            assignment = sorted(list(zip(row_ind, col_ind)), key=lambda x: x[1]) 
            return [j[0] for j in assignment]

        role_assignments = np.array([assign_index_(ed[np.arange(5)*n + i].T) for i in range(n)])
        
        # stack back to role for each seq
        role_assignments_seq = []
        start_id = 0
        for i in event_lengths:
            role_assignments_seq.append(role_assignments[start_id:start_id+i])
            start_id += i
            
        return role_assignments, role_assignments_seq

    
