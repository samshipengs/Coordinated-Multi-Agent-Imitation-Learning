import numpy as np


def id_player(event_df):
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
    '''input a dictionary contains id_role mapping for a single game events,
        check if there are role swaps.'''
    n_dup = 0
    for i in id_role_mapping.values():
        if len(i) > 1:
            n_dup += 1
    return n_dup 

def id_position(event_df):
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
    '''return x,y position of player and x,y,z of the ball'''
    # i[5][0][2:] is the balls x,y,z position
    return [j[2:4] + i[5][0][2:] for i in moments for j in i[5][1:] if j[1] == player_id]

def segment(X, length, overlap=None):
    ''' 
    segment a given list of moments to list of chunks each with size length 
    to do: try to implement overlap option
    '''
    n_segs = len(X)//length
#     return [X[i+(i+1)*length-op:i+(i+2)*length-op] for i in range(0, n_segs)]
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