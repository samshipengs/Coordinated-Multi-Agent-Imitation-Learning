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