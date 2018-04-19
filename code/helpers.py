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


# get player tracking
def get_player_trajectory(moments, player_id):
    '''
        return x,y position of player and x,y,z of the ball
    '''
    # i[5][0][2:] is the balls x,y,z position
    return [j[2:4] + i[5][0][2:] for i in moments for j in i[5][1:] if j[1] == player_id]


def segment(X, length, overlap=None):
    ''' 
        segment a given list of moments to list of chunks each with size length 
        to do: try to implement overlap option
    '''
    n_segs = len(X)//length
    return [X[i*length:(i+1)*length] for i in range(0, n_segs)]


# chunk to sequences
def get_sequences(X, length, dim):
    ''' 
    segment a given list of moments to list of chunks each with size length.
    this is usually used before creating batches.
    '''
    n_segs = len(X)//length
    return np.array([X[i*length:(i+1)*length] for i in range(0, n_segs)]).reshape(-1, length, dim)


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
        batches.append(inputs[excerpt, :, :])
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
        targets = np.array([self.mapping[int(i)] for i in teams])
        one_hot_targets = np.eye(nb_classes)[targets]
        
        return one_hot_targets
    
def process_moments(moments, homeid, awayid, id_role, role_order, court_index, game_id):
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
    for i in range(len(moments)):
        # get quarter number
        quarter_number = moments[i][0]
        # print(type(quarter_number))

        # print(moments[i][3], '====')
        
        # ball position array
        dm = len(moments[i][5])
        # ball_ind = -1
        player_ind = -1
        if dm == 11: # ball is present
            ball = np.array([moments[i][5][0][2:]])
            player_ind = 1
        elif dm == 10 and moments[i][5][0][:2] != [-1,-1]: # ball is not present
            ball = np.array([[-1, -1, -1]])
            player_ind = 0
        else:
            print('Warning!: There are less than 10 players! (skip)')
            continue
        # get player position data
        pp = np.array(moments[i][5][player_ind:])
        # # home
        # hpp = pp[pp[:, 0]==homeid, :]
        # # visitor
        # vpp = pp[pp[:, 0]==awayid, :]
        
        # home (update: instead of using home/visitor, we will just follow court index)
        hpp = pp[:5, :] # team1
        # visitor
        vpp = pp[5:, :] # team2
        # add one hot encoding for the teams
        h_team = hpp[:, 0]
        v_team = vpp[:, 0]
        # stack on the one hot encoding and leave the team id
        hpp = np.column_stack((hpp[:, 1:], ohe.encode(h_team)))
        vpp = np.column_stack((vpp[:, 1:], ohe.encode(v_team)))
        
        # reorder
        # [:,:-1] ignores the last null element
        h = order_moment(hpp[:, :-1], id_role, role_order)
        v = order_moment(vpp[:, :-1], id_role, role_order)
        # note, the player_id is discarded after order_moment

        # combine home and visit => update: in defend then offend order
        game_id = int(game_id)
        hv = []
        # print('>>>>>>>>>>', court_index[game_id])
        if quarter_number <= 2: # first half
            # print('first half')
            if court_index[game_id] == 0:
                # print(hpp.shape, sum(hpp[:, 1]<=half_court), sum(vpp[:, 1]<=half_court))
                # break
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
                    hv = np.vstack((v,h))
                else:
                    continue
        elif quarter_number > 2: # second half the court position is switched
            # print('second half')
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
                    hv = np.vstack((h,v))
                else:
                    continue
        if len(hv) == 0:
            continue
        # print(hv)



        # also record shot clocks for each of the moment/frame, this is used to
        # seperate a sequence into different frames (since when shot clock resets,
        # it usually implies a different state of game)
        shot_clock.append(moments[i][3])
        # stack on the ball position
        result.append(np.column_stack((hv, np.repeat(ball, hv.shape[0],0))))
    # print(len(result), len(moments), '??????????')
    if len(result) == 0:
        return None
    else:
        result = np.array(result) 
        # print(result.shape, '````````')
        return result.reshape(result.shape[0], -1), shot_clock