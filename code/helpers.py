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


def get_sequences(single_game, policy, sequence_length, overlap, n_fts=4):
    ''' create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
        n_fts: individual player features e.g. n_fts = 4 => (x,y,vx,vy)
    '''
    train = []
    target = []
    for i in single_game:
        i_len = len(i)
        if i_len < sequence_length:
            sequences = np.pad(np.array(i), [(0, sequence_length-i_len), (0,0)], mode='constant')
            targets = [np.roll(sequences[:, policy*n_fts:policy*n_fts+2], -1, axis=0)[:-1, :]]
            sequences = [sequences[:-1, :]]
        else:
            # https://stackoverflow.com/questions/48381870/a-better-way-to-split-a-sequence-in-chunks-with-overlaps
            sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len-1 else np.array(i[j:j+sequence_length]) \
                for j in range(0, i_len-overlap, sequence_length-overlap)]
            targets = [np.roll(k[:, policy*n_fts:policy*n_fts+2], -1, axis=0)[:-1, :] for k in sequences] # drop the last row as the rolled-back is not real
            sequences = [l[:-1, :] for l in sequences] # since target has dropped one then sequence also drop one
        
        train += sequences
        target += targets
    return train, target


def get_minibatches(inputs, targets, batchsize, shuffle=False):
    '''
        inputs: A list of events where each event is a sequence (array) of moments
                with sequence_length
        targets: target created by shifting 1 from inputs
        batchsize: desired batch size
        shuffle: Shuffle input data

        return: array of data in batch, shape:  
    '''
    assert len(inputs) == len(targets)
    batches = []
    target_batches = []
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        batches.append(inputs[excerpt])
        target_batches.append(targets[excerpt])
    return np.array(batches), np.array(target_batches)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    '''
        same as get_minibatches, except returns a generator
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


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
        # for i in teams:
        #     print(int(i), i)

        targets = np.array([self.mapping[int(i)] for i in teams])
        one_hot_targets = np.eye(nb_classes)[targets]
        # print(one_hot_targets)
        return one_hot_targets


def subsample_sequence(moments, subsample_factor, random_sample=False):#random_state=42):
    ''' 
        moments: a list of moment 
        subsample_factor: number of folds less than orginal
        random_sample: if true then sample a random one from the window of subsample_factor size
    '''
    seqs = np.copy(moments)
    moments_len = seqs.shape[0]
    n_intervals = moments_len//subsample_factor # number of subsampling intervals
    left = moments_len % subsample_factor # reminder

    if random_sample:
        if left != 0:
            rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)] + [np.random.randint(0, left)]
        else:
            rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)]
        interval_ind = range(0, moments_len, subsample_factor)
        # the final random index relative to the input
        rs_ind = np.array([rs[i] + interval_ind[i] for i in range(len(rs))])
        return seqs[rs_ind, :]
    else:
        s_ind = np.arange(0, moments_len, subsample_factor)
        return seqs[s_ind, :]


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
            

# =============================================================================================
def remove_non11(moments, event_length_th=25):
    ''' Go through each moment, when encounters balls not present on court,
        or less than 10 players, discard these moments and then chunk the following moments 
        to as another event.
        
        Motivations: balls out of bound or throwing the ball at side line will
            probably create a lot noise for the defend trajectory learning model.
            We could add the case where players are less than 10 (it could happen),
            but this is not allowed in the model and it requres certain input dimension.
        
        moments: A list of moments
        event_length_th: The minimum length of an event
        
        segments: A list of events (or, list of moments) e.g. [ms1, ms2] where msi = [m1, m2]
    '''
    
    segments = []
    segment = []
    # looping through each moment
    for i in range(len(moments)):
        # get moment dimension
        moment_dim = len(moments[i][5])
        # 1 bball + 10 players
        if moment_dim == 11:
            segment.append(moments[i])
        # less than ten players or basketball is not on the court
        else:
            # only grab these satisfy the length threshold
            if len(segment) >= event_length_th:
                segments.append(segment)
            # reset the segment to empty list
            segment = []
    # grab the last one
    if len(segment) != 0:
        segments.append(segment)
    
    return segments


def chunk_shotclock(moments, event_length_th=25, verbose=False):
    ''' When encounters ~24secs or game stops, chunk the moment to another event.
        shot clock test:
        1) c = [20.1, 20, 19, None,18, 12, 9, 7, 23.59, 23.59, 24, 12, 10, None, None, 10]
          result = [[20.1, 20, 19], [18, 12, 9, 7], [23.59], [23.59], [24, 12, 10]]
        2) c = [20.1, 20, 19, None, None,18, 12, 9, 7, 7, 7, 23.59, 23.59, 24, 12, 10, None, None, 10]
          result = [[20.1, 20, 19], [18, 12, 9, 7], [7], [7], [23.59], [23.59], [24, 12, 10]]
       
        Motivations: game flow would make sharp change when there's 24s or 
        something happened on the court s.t. the shot clock is stopped, thus discard
        these special moments and remake the following valid moments to be next event.

        moments: A list of moments
        event_length_th: The minimum length of an event
        verbose: print out exceptions or not
        
        segments: A list of events (or, list of moments) e.g. [ms1, ms2] where msi = [m1, m2] 
    '''
    
    segments = []
    segment = []
    # naturally we won't get the last moment, but it should be okay
    for i in range(len(moments)-1):
        current_shot_clock_i = moments[i][3]
        next_shot_clock_i = moments[i+1][3]
        # sometimes the shot clock value is None, thus cannot compare
        try:
            # if the game is still going i.e. sc is decreasing
            if next_shot_clock_i < current_shot_clock_i:
                segment.append(moments[i])
            # for any reason the game is sstopped or reset
            else:
                # not forget the last moment before game reset or stopped
                if current_shot_clock_i < 24.:
                    segment.append(moments[i])
                # add length condition
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []
        # None value
        except Exception as e:
            if verbose: print(e)
            # not forget the last valid moment before None value
            if current_shot_clock_i != None:
                segment.append(moments[i])    
            if len(segment) >= event_length_th:
                segments.append(segment)
            # reset the segment to empty list
            segment = []
                
    # grab the last one
    if len(segment) != 0:
        segments.append(segment)            
    
    return segments


def chunk_halfcourt(moments, event_length_th=25):
    ''' Discard any plays that are not single sided. When the play switches 
        court withhin one event, we chunk it to be as another event
        
    '''
    
    # NBA court size 94 by 50 feet
    half_court = 94/2. # feet
    cleaned = []

    # remove any moments where two teams are not playing at either side of the court
    for i in moments:
        # the x coordinates is on the 3rd or 2 ind of the matrix,
        # the first and second is team_id and player_id
        team1x = np.array(i[5])[1:6, :][:, 2]    # player data starts from 1, 0 ind is bball
        team2x = np.array(i[5])[6:, :][:, 2]
        # if both team are on the left court:
        if sum(team1x <= half_court)==5 and sum(team2x <= half_court)==5:
            cleaned.append(i)
        elif sum(team1x >= half_court)==5 and sum(team2x >= half_court)==5:
            cleaned.append(i)

    # if teamns playing court changed during same list of moments,
    # chunk it to another event
    segments = []
    segment = []
    for i in range(len(cleaned)-1):
        current_mean = np.mean(np.array(cleaned[i][5])[:, 2], axis=0)
        current_pos = 'R' if current_mean >= half_court else 'L'
        next_mean = np.mean(np.array(cleaned[i+1][5])[:, 2], axis=0)
        next_pos = 'R' if next_mean >= half_court else 'L'

        # the next moment both team are still on same side as current
        if next_pos == current_pos:
            segment.append(cleaned[i])
        else:
            if len(segment) >= event_length_th:
                segments.append(segment)
            segment = []
    # grab the last one
    if len(segment) != 0:
        segments.append(segment)   

    return segments


def reorder_teams(input_moments, game_id):
    ''' 1) the matrix always lays as home top and away bot VERIFIED
        2) the court index indicate which side the top team (home team) defends VERIFIED
        
        Reorder the team position s.t. the defending team is always the first 
        
        input_moments: A list moments
        game_id: str of the game id
    '''
    # now we want to reorder the team position based on meta data
    court_index = pd.read_csv('./meta_data/court_index.csv')
    court_index = dict(zip(court_index.game_id, court_index.court_position))
    
    half_court = 94/2. # feet
    home_defense = court_index[int(game_id)]
    moments = copy.deepcopy(input_moments)
    for i in range(len(moments)):
        home_moment_x = np.array(moments[i][5])[1:6,2]
        away_moment_x = np.array(moments[i][5])[6:11,2]
        quarter = moments[i][0]
        # if the home team's basket is on the left
        if home_defense == 0:
            # first half game
            if quarter <= 2:
                # if the home team is over half court, this means they are doing offense
                # and the away team is defending, so switch the away team to top
                if sum(home_moment_x>=half_court) and sum(away_moment_x>=half_court):
                    moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
            # second half game      
            if quarter > 2: # second half game, 3,4 quarter
                # now the home actually gets switch to the other court
                if sum(home_moment_x<=half_court) and sum(away_moment_x<=half_court):
                    moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
        # if the home team's basket is on the right
        elif home_defense == 1:
            # first half game
            if quarter <= 2:
                # if the home team is over half court, this means they are doing offense
                # and the away team is defending, so switch the away team to top
                if sum(home_moment_x<=half_court) and sum(away_moment_x<=half_court):
                    moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
            # second half game      
            if quarter > 2: # second half game, 3,4 quarter
                # now the home actually gets switch to the other court
                if sum(home_moment_x>=half_court) and sum(away_moment_x>=half_court):
                    moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
    return moments


def process_game_data(game_id, events_df, event_threshold, subsample_factor):
    single_game = []
    for i in events_df.moments.values:
        single_game += remove_non11(i, event_length_th=event_threshold)

    single_game1 = []
    for i in single_game:
        single_game1 += chunk_shotclock(i, event_length_th=event_threshold)

    single_game2 = []
    for i in single_game1:
        single_game2 += chunk_halfcourt(i, event_length_th=event_threshold)

    single_game3 = [reorder_teams(i, game_id) for i in single_game2]
    single_game_ = []
    single_bball_ = []
    for i in single_game3:
        event_i = []
        ball_i = []
        for j in i:
            # player xy positions
            event_i.append(np.array(j[5])[1:11, 2:4].reshape(-1))
            ball_i.append(j[5][0][2:])
        single_game_.append(np.array(event_i))
        single_bball_.append(np.array(ball_i))

    if subsample_factor != 0: # do subsample
        print('subsample enabled with subsample factor', subsample_factor)
        return [subsample_sequence(m, subsample_factor) for m in single_game_],[subsample_sequence(m, subsample_factor) for m in single_bball_]
    else:
        return single_game_, single_bball_



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

    
