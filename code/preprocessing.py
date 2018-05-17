# preprocessing.py
import glob, os, sys, math, warnings, copy, time, glob, logging
import numpy as np
import pandas as pd

from features import OneHotEncoding, flatten_moments, create_static_features, create_dynamic_features
from hidden_role_learning import HiddenStructureLearning
from sequencing import get_sequences, get_minibatches, iterate_minibatches, subsample_sequence

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

# ================================================================================================
# remove_non_eleven ==============================================================================
# ================================================================================================
def remove_non_eleven(events_df, event_length_th=25, verbose=False):
    df = events_df.copy()
    home_id = df.loc[0]['home']['teamid']
    away_id = df.loc[0]['visitor']['teamid']
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
    #             logging.debug('less than 11')
                # only grab these satisfy the length threshold
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []
        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)
        if len(segments) == 0:
            if verbose: logging.debug('Warning: Zero length event returned')
        return segments
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: remove_non_eleven_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values, {'home_id': home_id, 'away_id': away_id}


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
                if verbose: logging.debug(e)
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
            if verbose: logging.debug('Warning: Zero length event returned')
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
            if verbose: logging.debug('Warning: Zero length event returned')
        return segments
    
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: chunk_halfcourt_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values


# ================================================================================================
# reorder_teams ==================================================================================
# ================================================================================================
def reorder_teams(events_df, game_id):
    df = events_df.copy()
    def reorder_teams_(input_moments, game_id):
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
                    if sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        # also normalize the bball x location
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                # second half game      
                elif quarter > 2: # second half game, 3,4 quarter
                    # now the home actually gets switch to the other court
                    if sum(home_moment_x<=half_court)==5 and sum(away_moment_x<=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                    elif sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                else:
                    logging.debug('Should not be here, check quarter value')
            # if the home team's basket is on the right
            elif home_defense == 1:
                # first half game
                if quarter <= 2:
                    # if the home team is over half court, this means they are doing offense
                    # and the away team is defending, so switch the away team to top
                    if sum(home_moment_x<=half_court)==5 and sum(away_moment_x<=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                    elif sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                # second half game      
                elif quarter > 2: # second half game, 3,4 quarter
                    # now the home actually gets switch to the other court
                    if sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                else:
                    logging.debug('Should not be here, check quarter value')
        return moments
    return [reorder_teams_(m, game_id) for m in df.moments.values]










# BELOW NOT TESTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def process_game_data(Data, game_ids, event_threshold, subsample_factor):
    def process_game_data_(game_id, events_df, event_threshold, subsample_factor):
        # remove non elevens
        logging.debug('removing non eleven')
        result, _ = remove_non_eleven(events_df, event_threshold)
        df = pd.DataFrame({'moments': result})
        # chunk based on shot clock, Nones or stopped timer
        logging.debug('chunk shotclock')
        result = chunk_shotclock(df, event_threshold)
        df = pd.DataFrame({'moments': result})
        # chunk based on half court and normalize to all half court
        logging.debug('chunk half court')
        result = chunk_halfcourt(df, event_threshold)
        df = pd.DataFrame({'moments': result})
        # reorder team matrix s.t. the first five players are always defend side players
        logging.debug('reordering team')
        result = reorder_teams(df, game_id)
        df = pd.DataFrame({'moments': result})

        # features 
        # flatten data
        logging.debug('flatten moment')
        result, team_ids = flatten_moments(df)
        df = pd.DataFrame({'moments': result})  
        # static features
        logging.debug('add static features')
        result = create_static_features(df)
        df = pd.DataFrame({'moments': result})
        # dynamic features
        logging.debug('add velocities')
        fs = 1/25.
        result = create_dynamic_features(df, fs)
        # one hot encoding
        logging.debug('add one hot encoding')
        OHE = OneHotEncoding()
        result = OHE.add_ohs(result, team_ids)
        df = pd.DataFrame({'moments': result})
        return df

    game = []
    for i in range(len(game_ids)):
        logging.info('working on game {0:} | {1:} out of total {2:} games'.format(game_ids[i], i+1, len(game_ids)))
        game_data = Data.load_game(game_ids[i])
        events_df = pd.DataFrame(game_data['events'])
        game.append(process_game_data_(game_ids[i], events_df, event_threshold, subsample_factor))

    df = pd.concat(game, axis=0)
    # hidden role learning
    logging.debug('learning hidden roles')
    HSL = HiddenStructureLearning(df, defend_iter=120, offend_iter=140)
    result = HSL.reorder_moment()
    # subsample
    result = subsample_sequence(result, subsample_factor)
    return result