import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import pandas as pd
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm


def id_player(event_df):
    '''
        Map player id to player name.
    '''
    # get all the player_id and player_name mapping
    player_id_mapping = {}
    for i in range(event_df.shape[0]):
        home_players_i = event_df.iloc[i, :].home['players']
        away_players_i = event_df.iloc[i, :].visitor['players']
        for j in home_players_i:
            if j['playerid'] not in player_id_mapping.keys():
                player_id_mapping[j['playerid']] = j['firstname']+' '+j['lastname']
            elif j['firstname']+' '+j['lastname'] != player_id_mapping[j['playerid']]:
                print('Same id is being used for different players!')
        for j in away_players_i:
            if j['playerid'] not in player_id_mapping.keys():
                player_id_mapping[j['playerid']] = j['firstname']+' '+j['lastname']
            elif j['firstname']+' '+j['lastname'] != player_id_mapping[j['playerid']]:
                print('Same id is being used for different players!')
    return player_id_mapping


def check_game_roles_duplicates(id_role_mapping):
    '''
        input a dictionary contains id_role mapping for a single game events,
        check if there are role swaps.
    '''
    n_dup = 0
    for i in id_role_mapping.values():
        if len(i) > 1:
            n_dup += 1
    return n_dup 


def id_position(event_df):
    '''
        Map player id to a list of positions (in most case it's just one position/role)
    '''
    # get position mapping
    # get all the player_id and player_name mapping
    position_id_mapping = {}
    for i in range(event_df.shape[0]):
        home_players_i = event_df.iloc[i, :].home['players']
        away_players_i = event_df.iloc[i, :].visitor['players']
        for j in home_players_i:
            if j['playerid'] not in position_id_mapping.keys():
                position_id_mapping[j['playerid']] = [j['position']]
            else:
                if j['position'] not in position_id_mapping[j['playerid']]:
                    print('Same id is being used for different positions!')
                    position_id_mapping[j['playerid']].append(j['position'])
                
        for j in away_players_i:
            if j['playerid'] not in position_id_mapping.keys():
                position_id_mapping[j['playerid']] = [j['position']]
                # print(j['position'])
            else:
                if j['position'] not in position_id_mapping[j['playerid']]:
                    print('Same id is being used for different positions!')
                    position_id_mapping[j['playerid']].append(j['position'])
    return position_id_mapping


def id_teams(event_dfs):
    '''
        Map team id to team names
    '''
    def id_team_(event_df):
        one_row = event_df.loc[0] 
        home_id = one_row.home['teamid']
        home_team = one_row.home['name'].lower()

        away_id = one_row.visitor['teamid']
        away_team = one_row.visitor['name'].lower()
        return home_id, home_team, away_id, away_team
    result = {}
    for i in event_dfs:
        id1, name1, id2, name2 = id_team_(i)
        ks = result.keys()
        if id1 in ks:
            if result[id1] != name1:
                raise ValueError('team id is duplicated!')
        else:
            result[id1] = name1
        if id2 in ks:
            if result[id2] != name2:
                raise ValueError('team id is duplicated!')
        else:
            result[id2] = name2
    return result


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


def filter_event_type(events_df, discard_event):
    '''
        events_df: a single game events dataframe
        discard_event: a list of integers of event types to be discarded

        return: a event df with dicard_event filtered out
    '''
    def filter_events_(x, discard_event):
        etype = x['EVENTMSGTYPE'].values
        if len(set(etype).intersection(discard_event))!=0 or len(etype) ==0:
            # if the event contains discard events or if the event type is an empty list
            return False
        else:
            return True
            
    # def filter_events_(x, use_event):

    
    events = events_df[events_df.playbyplay.apply(lambda x: filter_events_(x, discard_event))].copy()
    events.reset_index(drop=True, inplace=True)
    return events

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
# ROLE ALIGNMENT
def process_moments_ra(moments, homeid, awayid, court_index, game_id):
    result = []
    shot_clock = []
    player_id = []
    ball_positions = []
    half_court = 47. # 94/2
    n_balls_missing = 0
    for i in range(len(moments)):
        # get quarter number
        quarter_number = moments[i][0]

        # ball position array
        dm = len(moments[i][5])
        player_ind = -1
        if dm == 11: # ball is present
            ball = np.array([moments[i][5][0][2:]])
            player_ind = 1
        elif dm == 10 and moments[i][5][0][:2] != [-1,-1]: # ball is not present
            # print('Ball is not present.')
            n_balls_missing += 1
            ball = np.array([[-1, -1, -1]])
            player_ind = 0
        else:
            print('Warning!: There are less than 10 players! (skip)')
            continue
        # get player position data
        pp = np.array(moments[i][5][player_ind:])
        
        # home (update: instead of using home/visitor, we will just follow court index)
        # ignore last null column, team_id
        hpp = pp[:5, 1:-1] # team1
        # visitor
        vpp = pp[5:, 1:-1] # team2
        
        # combine home and visit => update: in defend then offend order
        game_id = int(game_id)
        hv = []
        if quarter_number <= 2: # first half
            if court_index[game_id] == 0:
                # all the left players on the left side,
                # and the right court players also on the left side
                if sum(hpp[:, 1]<=half_court)==5 and sum(vpp[:, 1]<=half_court)==5:
                    hv = np.vstack((hpp,vpp))
                else:
                    continue
            else:
                # all the left players on the right side,
                # and the right court players also on the right side
                if sum(hpp[:, 1]>=half_court)==5 and sum(vpp[:, 1]>=half_court)==5:
                    vpp[:, 1] = 2*half_court - vpp[:, 1]
                    hpp[:, 1] = 2*half_court - hpp[:, 1]
                    hv = np.vstack((vpp,hpp))
                else:
                    continue
        elif quarter_number > 2: # second half the court position is switched
            if court_index[game_id] == 0:
                # all the left players on the left side,
                # and the right court players also on the left side
                if sum(hpp[:, 1]<=half_court)==5 and sum(vpp[:, 1]<=half_court)==5:
                    # now the defend team is the team2, v
                    hv = np.vstack((vpp,hpp))
                else:
                    continue
            else:
                # all the left players on the right side,
                # and the right court players also on the right side
                if sum(hpp[:, 1]>=half_court)==5 and sum(vpp[:, 1]>=half_court)==5:
                    hpp[:, 1] = 2*half_court - hpp[:, 1]
                    vpp[:, 1] = 2*half_court - vpp[:, 1]
                    hv = np.vstack((hpp,vpp))
                else:
                    continue
        if len(hv) == 0:
            continue

        # add the position of the ball
        ball_positions.append(ball.reshape(-1))

        # also record shot clocks for each of the moment/frame, this is used to
        # seperate a sequence into different frames (since when shot clock resets,
        # it usually implies a different state of game)
        shot_clock.append(moments[i][3])

        # just the player's position with player id
        result.append(np.array(hv)[:, 1:].reshape(-1))

        # resulti = list(np.array(resulti).reshape(-1)) + list(ball.reshape(-1)) + ohe_i
        # result.append(resulti)
    
    if len(result) == 0:
        return None
    else:
        return np.array(result), shot_clock, np.array(ball_positions)



def get_game_data_ra(events, court_index, game_id, event_threshold=10, subsample_factor=3):
    '''
        events: a single game's events dataframe
        
        return: a list of events, each event consists a sequence of moments 
    '''
    # 
    homeid = events.loc[0].home['teamid']
    awayid = events.loc[0].visitor['teamid']
    single_game = []
    single_game_balls = []
    # sc = 24. # init 24s shot clock

    # filter out seq length less than threshold, this has to be greater than 2
    # otherwise, there might be duplicates appearing
    len_th = event_threshold    # the number of moments in a single event need to satisfy
    n = 0    # record number that satisfies the threshold
    n_short = 0    # number that doesnt match
    for k, v in enumerate(events.moments.values):
        result_i = process_moments_ra(v, homeid, awayid, court_index, game_id)
        if result_i == None:
            continue
        else:
            s1 = 0 # index that divides the sequence, this usually happens for 24s shot clock
            pm, scs, ball_pos = result_i
            for i in range(len(scs)-1):
                # sometimes there are None shot clock value
                if scs[i] != None and scs[i+1] == None:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        single_game_balls.append(ball_pos[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
                elif scs[i] == None:
                    s1 += 1
                elif scs[i+1] >= scs[i]:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        single_game_balls.append(ball_pos[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
            # grab the end piece
            if s1 != len(scs)-2:
                if len(scs[s1:]) >= len_th:
                    single_game.append(pm[s1:])
                    single_game_balls.append(ball_pos[s1:])
                    n += 1
                else:
                    n_short += 1
    # dimensions extreme<3> x n_players<10> x (player_pos<2> + teamid_onehot<25> + ball<3>) = 900
    # dimensions = extreme<3> x n_players<10> x player_pos<2> + teamid_onehot<4> + ball<3> = 67

    if subsample_factor != 0: # do subsample
        print('subsample enabled with subsample factor', subsample_factor)
        return [subsample_sequence(m, subsample_factor) for m in single_game]
    else:
        return single_game, single_game_balls     #, (n, n_short)

def order_moment_ra(moments, role_assignments, components=7, n=5, n_ind=4):
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

    
