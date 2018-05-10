# preprocessing.py
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
from features import *

# ================================================================================================
# remove_non_eleven ==============================================================================
# ================================================================================================
def remove_non_eleven(events_df, event_length_th=25, verbose=False):
    df = events_df.copy()
    def remove_non_eleven_(moments, event_length_th=25, verbose=False):
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
    #             print('less than 11')
                # only grab these satisfy the length threshold
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []
        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)
        if len(segments) == 0:
            if verbose: print('Warning: Zero length event returned')
        return segments
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: remove_non_eleven_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values


# ================================================================================================
# chunk_shotclock ================================================================================
# ================================================================================================
def chunk_shotclock(events_df, event_length_th=25, verbose=False):
    df = events_df.copy()
    def chunk_shotclock_(moments, event_length_th, verbose):
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
        if len(segment) >= event_length_th:
            segments.append(segment)            
        if len(segments) == 0:
            if verbose: print('Warning: Zero length event returned')
        return segments
    
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: chunk_shotclock_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values


# ================================================================================================
# chunk_halfcourt ================================================================================
# ================================================================================================
def chunk_halfcourt(events_df, event_length_th=25, verbose=False):
    df = events_df.copy()
    def chunk_halfcourt_(moments, event_length_th, verbose):
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
            team2x = np.array(i[5])[6:11, :][:, 2]
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
        if len(segment) >= event_length_th:
            segments.append(segment)            
        if len(segments) == 0:
            if verbose: print('Warning: Zero length event returned')
        return segments
    
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: chunk_halfcourt_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values












# BELOW NOT TESTED




















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
    
    full_court = 94.
    half_court = full_court/2. # feet
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
                    for l in moments[i][5][1:6]:
                        l[2] = full_court - l[2]
                    for l in moments[i][5][6:11]:
                        l[2] = full_court - l[2]
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
                    for l in moments[i][5][1:6]:
                        l[2] = full_court - l[2]
                    for l in moments[i][5][6:11]:
                        l[2] = full_court - l[2]
    return moments


def process_game_data(game_id, events_df, event_threshold, subsample_factor):
    # single_game = []
    # for i in events_df.moments.values:
    #     single_game += remove_non11(i, event_length_th=event_threshold)

    result = remove_non_eleven(events_df, event_threshold)
    df = pd.DataFrame({'moments': result})

    result = chunk_shotclock(df, event_threshold)
    df = pd.DataFrame({'moments': result})

    result = chunk_halfcourt(df, event_threshold)
    # df = pd.DataFrame({'moments': result})

    # single_game1 = []
    # for i in single_game:
    #     single_game1 += chunk_shotclock(i, event_length_th=event_threshold)

    # single_game2 = []
    # for i in single_game1:
    #     single_game2 += chunk_halfcourt(i, event_length_th=event_threshold)

    single_game3 = [reorder_teams(i, game_id) for i in result]
    single_game_ = []
    single_bball_ = []
    # one-hot encode the game ids
    OHE = OneHotEncoding()
    team_ohes = []    
    for i in single_game3:
        event_i = []
        team_ids = [[k[0] for k in i[0][5][1:6]], [k[0] for k in i[0][5][6:11]]]
        assert len(set(team_ids[0])) == 1 and len(set(team_ids[1])) ==1, 'orders are wrong'
        team_id = [team_ids[0][0], team_ids[-1][0]]
        ball_i = []
        for j in i:
            assert [j[5][1][0], j[5][-1][0]] == team_id, 'Should equal'
            # player xy positions
            event_i.append(np.array(j[5])[1:11, 2:4].reshape(-1))
            ball_i.append(j[5][0][2:])
        single_game_.append(np.array(event_i))
        single_bball_.append(np.array(ball_i))
        team_ohes.append(OHE.encode(team_id).reshape(-1))

    if subsample_factor != 0: # do subsample
        print('subsample enabled with subsample factor', subsample_factor)
        return [subsample_sequence(m, subsample_factor) for m in single_game_],\
               [subsample_sequence(m, subsample_factor) for m in single_bball_],\
               team_ohes
    else:
        return single_game_, single_bball_, team_ohes



