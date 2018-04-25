import numpy as np
import pandas as pd


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

# # chunk to sequences
# def get_sequences(X, length, dim):
#     ''' 
#     segment a given list of moments to list of chunks each with size length.
#     this is usually used before creating batches.
#     '''
#     n_segs = len(X)//length
#     return np.array([X[i*length:(i+1)*length] for i in range(0, n_segs)]).reshape(-1, length, dim)

def get_sequences(single_game, sequence_length, overlap):
    train = []
    target = []
    for i in single_game:
        i_len = len(i)
        if i_len < sequence_length:
            sequences = np.pad(np.array(i), [(0, sequence_length-i_len), (0,0)], mode='constant')
            targets = [np.roll(sequences[:, :2], -1, axis=0)[:-1, :]]
            sequences = [sequences[:-1, :]]
        else:
            # https://stackoverflow.com/questions/48381870/a-better-way-to-split-a-sequence-in-chunks-with-overlaps
            sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len-1 else np.array(i[j:j+sequence_length]) \
                for j in range(0, i_len-overlap, sequence_length-overlap)]
            targets = [np.roll(k[:, :2], -1, axis=0)[:-1, :] for k in sequences] # drop the last row as the rolled-back is not real
            sequences = [l[:-1, :] for l in sequences] # since target has dropped one then sequence also drop one
        
        train += sequences
        target += targets
    return train, target


# chunk to batch
# define ftn that generates batches
def get_minibatches(inputs, targets, batchsize, shuffle=False):
    '''
        inputs:
        targets:
        batchsize:
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

# define ftn that generates batches
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


def order_moment(m, rm, ro, extreme=3):
    '''
        m: moments, rm: role model, ro: role order
        for the case of multiple players sharing the smae roles,
        (this can happen to even with hiddlen structure learning, 
         although it might be allevaited by using lienar assignment)
        so for now, we come up with an extrem case where same role are occupied by say, 3 players,
        then we still follow the meta order but create paddings 3 times.
    '''
    # reorder moments by role based mapping, where first col is player id
    role = [rm[int(i)][0] for i in m[:,0]]
    u_role = list(set(role))
    assert len(u_role) >= 2, 'it goes over extreme case'
    
    d1,d2 = m.shape
    try:
        assert d1 == 5, 'd1,d2 = {0:}, {1:}'.format(d1, d2)
    except:
        print('Warning:', d1, d2, end='\r')
    # initialize slots (5 meta positions)
    slots = np.zeros((extreme*5, d2))
    counter = {}
    for i in range(len(role)):
        role_i = role[i]
        if role_i not in counter.keys():
            counter[role_i] = 0
        else:
            # note: this could possibly be better if add linear assignment
            counter[role_i] += 1
        # filling in the slots
        slots[ro[role_i]*extreme+counter[role_i], :] = m[i, :]
#     return slots[:, 1:] # [, 1:] slice 1 since we don't need the player id anymore
    return slots[:, 1:] 


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
    
def process_moments(moments, homeid, awayid, id_role, role_order, court_index, game_id, extreme=3):
    '''
        moments: a list of moment(frame)
        homeid: home team's team_id
        awayid: visit team's team_id
        id_role: a dictionary contains mapping from player id to a list of roles,
            e.g. {2200: ['C-F'], 2449: ['F'], ... }
        role_order: a dictionary contains mapping from role to order,
            e.g. {'F': 0, 'G':4, 'C-F':1, 'G-F':3, 'F-G':3, 'C':2, 'F-C':1}
        court_index: the index indicating which team from the array is the team
            on the left (i.e. close to the origin (0,0), this swaps in the second-half)
        game_id: id of the game

        return: a reordered moments array with defend team at first
    '''
    # init one hot encoding
    ohe = OneHotEncoding()
    result = []
    shot_clock = []
    # half court = 94/2
    half_court = 47.
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
        # ignore last null column
        hpp = pp[:5, :-1] # team1
        # visitor
        vpp = pp[5:, :-1] # team2
        # add one hot encoding for the teams
        h_team = hpp[:, 0]
        v_team = vpp[:, 0]
        # stack on the one hot encoding and leave the team id
        hpp = np.column_stack((hpp[:, 1:], ohe.encode(h_team)))
        vpp = np.column_stack((vpp[:, 1:], ohe.encode(v_team)))
        
        h = order_moment(hpp, id_role, role_order)
        v = order_moment(vpp, id_role, role_order)
        # note, the player_id is discarded after order_moment

        # combine home and visit => update: in defend then offend order
        game_id = int(game_id)
        hv = []
        if quarter_number <= 2: # first half
            if court_index[game_id] == 0:
                # all the left players on the left side,
                # and the right court players also on the left side
                if sum(hpp[:, 1]<=half_court)==5 and sum(vpp[:, 1]<=half_court)==5:
                    hv = np.vstack((h,v))
                else:
                    continue
            else:
                # all the left players on the right side,
                # and the right court players also on the right side
                if sum(hpp[:, 1]>=half_court)==5 and sum(vpp[:, 1]>=half_court)==5:
                    # we also normalize the court i.e. move back to the left court (mirror reflect)
                    v[:, 1] = 2*half_court - v[:, 1]
                    h[:, 1] = 2*half_court - h[:, 1]
                    hv = np.vstack((v,h))
                else:
                    continue
        elif quarter_number > 2: # second half the court position is switched
            if court_index[game_id] == 0:
                # all the left players on the left side,
                # and the right court players also on the left side
                if sum(hpp[:, 1]<=half_court)==5 and sum(vpp[:, 1]<=half_court)==5:
                    # now the defend team is the team2, v
                    hv = np.vstack((v,h))
                else:
                    continue
            else:
                # all the left players on the right side,
                # and the right court players also on the right side
                if sum(hpp[:, 1]>=half_court)==5 and sum(vpp[:, 1]>=half_court)==5:
                    h[:, 1] = 2*half_court - h[:, 1]
                    v[:, 1] = 2*half_court - v[:, 1]
                    hv = np.vstack((h,v))
                else:
                    continue
        if len(hv) == 0:
            continue

        # also record shot clocks for each of the moment/frame, this is used to
        # seperate a sequence into different frames (since when shot clock resets,
        # it usually implies a different state of game)
        shot_clock.append(moments[i][3])

        # only get the necessary parts
        ohe_i = list(hv[0][2:]) + list(hv[extreme*5][2:]) # first team's (defending) one hot concat with second 
        resulti = hv[:, :2] # just the player's position
        # stack on the ball position
        resulti = list(np.array(resulti).reshape(-1)) + list(ball.reshape(-1)) + ohe_i
        result.append(resulti)

    # if n_balls_missing!=0: print('n_balls_missing:', n_balls_missing)
    
    if len(result) == 0:
        return None
    else:
        return np.array(result), shot_clock         


def get_game_data(events, id_role, role_order, court_index, game_id, event_threshold=10, subsample_factor=3):
    '''
        events: a single game's events dataframe
        
        return: a list of events, each event consists a sequence of moments 
    '''
    # 
    homeid = events.loc[0].home['teamid']
    awayid = events.loc[0].visitor['teamid']
    single_game = []
    # sc = 24. # init 24s shot clock

    # filter out seq length less than threshold, this has to be greater than 2
    # otherwise, there might be duplicates appearing
    len_th = event_threshold    # the number of moments in a single event need to satisfy
    n = 0    # record number that satisfies the threshold
    n_short = 0    # number that doesnt match
    for k, v in enumerate(events.moments.values):
        result_i = process_moments(v, homeid, awayid, id_role, role_order, court_index, game_id)
        if result_i == None:
            continue
        else:
            s1 = 0 # index that divides the sequence, this usually happens for 24s shot clock
            pm, scs = result_i
            for i in range(len(scs)-1):
                # sometimes there are None shot clock value
                if scs[i] != None and scs[i+1] == None:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
                elif scs[i] == None:
                    s1 += 1
                elif scs[i+1] >= scs[i]:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
            # grab the end piece
            if s1 != len(scs)-2:
                if len(scs[s1:]) >= len_th:
                    single_game.append(pm[s1:])
                    n += 1
                else:
                    n_short += 1
    # dimensions extreme<3> x n_players<10> x (player_pos<2> + teamid_onehot<25> + ball<3>) = 900
    # dimensions = extreme<3> x n_players<10> x player_pos<2> + teamid_onehot<4> + ball<3> = 67

    if subsample_factor != 0: # do subsample
        print('subsample enabled with subsample factor', subsample_factor)
        return [subsample_sequence(m, subsample_factor) for m in single_game]
    else:
        return single_game#, (n, n_short)


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


def get_velocity(event, dim1, fs):
    ''' 
        event: an array where each row is a moment/frame, columns are the input feature vectors
        dim1: the position dimensions before adding velocities 
        fs: time lapse from frame to next frame

        note: the last column is discarded because theres no velocity info for it
    '''
    pos = event[:, :dim1]
    next_pos = np.copy(np.roll(pos,-1, axis=0)[:, :dim1])
    vel = (next_pos - pos)/fs
    return np.column_stack((event[:-1, :], vel[:-1, :]))



# =============================================================================================
# ROLE ALIGNMENT
def process_moments_ra(moments, homeid, awayid, court_index, game_id):
    result = []
    shot_clock = []
    player_id = []
    # half court = 94/2
    half_court = 47.
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
        return np.array(result), shot_clock



def get_game_data_ra(events, court_index, game_id, event_threshold=10, subsample_factor=3):
    '''
        events: a single game's events dataframe
        
        return: a list of events, each event consists a sequence of moments 
    '''
    # 
    homeid = events.loc[0].home['teamid']
    awayid = events.loc[0].visitor['teamid']
    single_game = []
    player_id = []
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
            pm, scs = result_i
            for i in range(len(scs)-1):
                # sometimes there are None shot clock value
                if scs[i] != None and scs[i+1] == None:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
                elif scs[i] == None:
                    s1 += 1
                elif scs[i+1] >= scs[i]:
                    if len(scs[s1:i+1]) >= len_th:
                        single_game.append(pm[s1:i+1])
                        n += 1
                    else:
                        n_short += 1
                    s1 = i+1
            # grab the end piece
            if s1 != len(scs)-2:
                if len(scs[s1:]) >= len_th:
                    single_game.append(pm[s1:])
                    n += 1
                else:
                    n_short += 1
    # dimensions extreme<3> x n_players<10> x (player_pos<2> + teamid_onehot<25> + ball<3>) = 900
    # dimensions = extreme<3> x n_players<10> x player_pos<2> + teamid_onehot<4> + ball<3> = 67

    if subsample_factor != 0: # do subsample
        print('subsample enabled with subsample factor', subsample_factor)
        return [subsample_sequence(m, subsample_factor) for m in single_game]
    else:
        return single_game#, (n, n_short)