import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cPickle as pickle
import _pickle as pickle
import os


color_dict = {
    1610612737: ('#E13A3E', 'ATL'),
    1610612738: ('#008348', 'BOS'),
    1610612751: ('#061922', 'BKN'),
    1610612766: ('#1D1160', 'CHA'),
    1610612741: ('#CE1141', 'CHI'),
    1610612739: ('#860038', 'CLE'),
    1610612742: ('#007DC5', 'DAL'),
    1610612743: ('#4D90CD', 'DEN'),
    1610612765: ('#006BB6', 'DET'),
    1610612744: ('#FDB927', 'GSW'),
    1610612745: ('#CE1141', 'HOU'),
    1610612754: ('#00275D', 'IND'),
    1610612746: ('#ED174C', 'LAC'),
    1610612747: ('#552582', 'LAL'),
    1610612763: ('#0F586C', 'MEM'),
    1610612748: ('#98002E', 'MIA'),
    1610612749: ('#00471B', 'MIL'),
    1610612750: ('#005083', 'MIN'),
    1610612740: ('#002B5C', 'NOP'),
    1610612752: ('#006BB6', 'NYK'),
    1610612760: ('#007DC3', 'OKC'),
    1610612753: ('#007DC5', 'ORL'),
    1610612755: ('#006BB6', 'PHI'),
    1610612756: ('#1D1160', 'PHX'),
    1610612757: ('#E03A3E', 'POR'),
    1610612758: ('#724C9F', 'SAC'),
    1610612759: ('#BAC3C9', 'SAS'),
    1610612761: ('#000000', 'TOR'),
    1610612762: ('#00471B', 'UTA'),
    1610612764: ('#002B5C', 'WAS'),
}


#EVENTMSGTYPE
#1 - Make
#2 - Miss
#3 - Free Throw
#4 - Rebound
#5 - out of bounds / Turnover / Steal
#6 - Personal Foul
#7 - Violation
#8 - Substitution
#9 - Timeout
#10 - Jumpball
#12 - Start Q1?
#13 - Start Q2?
# there are empty playbyplay, what are those?

class LoadData:
    def __init__(self, main_dir, game_dir):
        # directories
        self.main_dir = main_dir
        self.game_dir = game_dir
    
    def load_game(self, gameid):
        '''return a dataframe from a game'''
        data = pd.read_pickle(self.game_dir+gameid+'.pkl')
        return data
    
    def load_csv(self, file_name):
        return pd.read_csv(file_name)


class PlotGame:
    ''' 
    see more for plotting: 
        https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/
    '''

    def __init__(self, gameid, main_dir, game_dir):
        # gameid='0021500463'
        self.gameid = gameid
        # directories
        self.main_dir = main_dir
        self.game_dir = game_dir
        self.court_path = main_dir + 'nba_court_T.png'

    def load_moment2img(self, data, event_number, moment_number, return_img=False):
        '''load_moment2img
        Given a game data, a certain event_number and a particular moment number,
        save the court plus players and ball info as an image to a directoy. 
        '''
        num_events = len(data['events'])

        player_fields = data['events'][0]['home']['players'][0].keys()

        # CHANGE THIS
        # specify an event number
        ii = event_number

        home_players = pd.DataFrame(data=[i for i in data['events'][0]['home']['players']], columns=player_fields)
        away_players = pd.DataFrame(data=[i for i in data['events'][0]['visitor']['players']], columns=player_fields)
        players = pd.merge(home_players, away_players, how='outer')
        jerseydict = dict(zip(players.playerid.values, players.jersey.values))

        # get the position of the players and the ball throughout the event
        ball_xy = np.array([x[5][0][2:5] for x in data['events'][ii]['moments']]) #create matrix of ball data
        player_xy = np.array([np.array(x[5][1:])[:,:4] for x in data['events'][ii]['moments']]) #create matrix of player data

        # get the play by play data for this clip
        playbyplay = data['events'][ii]['playbyplay']


        team_1_xy_mean = -np.ones((len(player_xy),2))
        team_2_xy_mean = -np.ones((len(player_xy),2))


        # CHANGE THIS
        # plot a certain frame:
        jj = moment_number


        print('event ' + str(ii) + '/' + str(num_events) + ", moment: "+ str(jj) + '/'+ str(len(player_xy)), end='\r')
        fig = plt.figure()
        ax = plt.gca() #create axis object

        img = mpimg.imread(self.court_path)  # read image. I got this image from gmf05's github.

        plt.imshow(img, extent=[0,94,0,50], zorder=0)  # show the image.

        # get player and ball data for the momenet
        ball = ball_xy[jj]
        player = player_xy[jj]


        # plot clock info
        clock_info = ax.annotate('', xy=[94.0/2 - 6.0/1.5 +0.1, 50 - 6.0/1.5 -0.35],
            color='black', horizontalalignment='center', verticalalignment='center')

        if not data['events'][ii]['moments'][jj][0] == None:
            quarter = data['events'][ii]['moments'][jj][0]
        else:
            quarter = 0

        if not data['events'][ii]['moments'][jj][2] == None:
            game_clock = data['events'][ii]['moments'][jj][2]
        else:
            game_clock = 0

        if not data['events'][ii]['moments'][jj][3] == None:
            game_shot = data['events'][ii]['moments'][jj][3]
        else:
            game_shot = 0

        clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
            quarter,
            int(game_clock) % 3600 // 60,
            int(game_clock) % 60,
            game_shot)
        clock_info.set_text(clock_test)

        # the event title
        temp = str(data['events'][ii]['home']['abbreviation'])+\
            ' vs. ' + str(data['events'][ii]['visitor']['abbreviation'])+\
            '\n'
        for idx, pp in playbyplay.iterrows():
            temp = temp + str(pp['HOMEDESCRIPTION'])+ " , " +\
                    str(pp['VISITORDESCRIPTION'])+ " , "+\
                    str(pp['PCTIMESTRING'])+ " , "+\
                    str(pp['event_str'])  + '\n'


        plt.title(temp)

        for kk in range(player.shape[0]): #create circle object and text object for each player

            #
            #kk = 1
            #
            team_id = player[kk,0]
            player_id = player[kk,1]
            xx = player[kk,2]
            yy  =player[kk, 3]

            # player circle
            player_circ = plt.Circle((xx,yy), 2.2,
                            facecolor=color_dict[team_id][0],edgecolor='k')
            ax.add_artist(player_circ)

            # player jersey # (text)
            ax.text(xx,yy,jerseydict[player_id],color='w',ha='center',va='center')

        # draw the ball
        ball_circ = plt.Circle((ball[0], ball[1]), ball[2]/7, color=[1, 0.4, 0])  # create circle object for bal
        ax.add_artist(ball_circ)

        # add the average position of each team tp the frame
        team_ids = np.unique(player[:,0])

        team_1_xy = player[player[:,0] == team_ids[0]]
        team_1_xy = team_1_xy[:,[2,3]]
        team_1_xy_mean[jj,:] = np.mean(team_1_xy,0)
        plt.plot(team_1_xy_mean[:jj+1,0],team_1_xy_mean[:jj+1,1],'o',
                color=color_dict[team_ids[0]][0],
                alpha=0.2)


        team_2_xy = player[player[:,0] == team_ids[1]]
        team_2_xy = team_2_xy[:,[2,3]]
        team_2_xy_mean[jj,:] = np.mean(team_2_xy,0)
        plt.plot(team_2_xy_mean[:jj+1,0],team_2_xy_mean[:jj+1,1],'o',
                color=color_dict[team_ids[1]][0],
                alpha=0.2)

        plt.xlim([0,94])
        plt.ylim([0,50])

        plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)

        if return_img:
            return ax
        else:
            # save image
            save_path = self.game_dir + 'game' + str(self.gameid) + '/' + 'event' + str(event_number) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + str(moment_number) + '.png')
            plt.cla()
            plt.close(fig)

    def load_pred_moment2img(self, data, event_number, moment_number):
        num_events = len(data['events'])

        player_fields = data['events'][0]['home']['players'][0].keys()

        # CHANGE THIS
        # specify an event number
        ii = event_number

        home_players = pd.DataFrame(data=[i for i in data['events'][0]['home']['players']], columns=player_fields)
        away_players = pd.DataFrame(data=[i for i in data['events'][0]['visitor']['players']], columns=player_fields)
        players = pd.merge(home_players, away_players, how='outer')
        jerseydict = dict(zip(players.playerid.values, players.jersey.values))

        # get the position of the players and the ball throughout the event
        ball_xy = np.array([x[5][0][2:5] for x in data['events'][ii]['moments']]) #create matrix of ball data
        player_xy = np.array([np.array(x[5][1:])[:,:4] for x in data['events'][ii]['moments']]) #create matrix of player data

        # get the play by play data for this clip
        playbyplay = data['events'][ii]['playbyplay']


        team_1_xy_mean = -np.ones((len(player_xy),2))
        team_2_xy_mean = -np.ones((len(player_xy),2))

        # CHANGE THIS
        # plot a certain frame:
        jj = moment_number

        print('event ' + str(ii) + '/' + str(num_events) + ", moment: "+ str(jj) + '/'+ str(len(player_xy)), end='\r')
        fig = plt.figure()
        ax = plt.gca() #create axis object

        img = mpimg.imread(self.court_path)  # read image. I got this image from gmf05's github.

        plt.imshow(img, extent=[0,94,0,50], zorder=0)  # show the image.

        # get player and ball data for the momenet
        ball = ball_xy[jj]
        player = player_xy[jj]


        # plot clock info
        clock_info = ax.annotate('', xy=[94.0/2 - 6.0/1.5 +0.1, 50 - 6.0/1.5 -0.35],
            color='black', horizontalalignment='center', verticalalignment='center')

        if not data['events'][ii]['moments'][jj][0] == None:
            quarter = data['events'][ii]['moments'][jj][0]
        else:
            quarter = 0

        if not data['events'][ii]['moments'][jj][2] == None:
            game_clock = data['events'][ii]['moments'][jj][2]
        else:
            game_clock = 0

        if not data['events'][ii]['moments'][jj][3] == None:
            game_shot = data['events'][ii]['moments'][jj][3]
        else:
            game_shot = 0

        clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
            quarter,
            int(game_clock) % 3600 // 60,
            int(game_clock) % 60,
            game_shot)
        clock_info.set_text(clock_test)

        # the event title
        temp = str(data['events'][ii]['home']['abbreviation'])+\
            ' vs. ' + str(data['events'][ii]['visitor']['abbreviation'])+\
            '\n'
        for idx, pp in playbyplay.iterrows():
            temp = temp + str(pp['HOMEDESCRIPTION'])+ " , " +\
                    str(pp['VISITORDESCRIPTION'])+ " , "+\
                    str(pp['PCTIMESTRING'])+ " , "+\
                    str(pp['event_str'])  + '\n'


        plt.title(temp)

        for kk in range(player.shape[0]): #create circle object and text object for each player

            #
            #kk = 1
            #
            team_id = player[kk,0]
            player_id = player[kk,1]
            xx = player[kk,2]
            yy  =player[kk, 3]

            # player circle
            player_circ = plt.Circle((xx,yy), 2.2,
                            facecolor=color_dict[team_id][0],edgecolor='k')
            ax.add_artist(player_circ)

            # player jersey # (text)
            ax.text(xx,yy,jerseydict[player_id],color='w',ha='center',va='center')

        # draw the ball
        ball_circ = plt.Circle((ball[0], ball[1]), ball[2]/7, color=[1, 0.4, 0])  # create circle object for bal
        ax.add_artist(ball_circ)

        # add the average position of each team tp the frame
        team_ids = np.unique(player[:,0])

        team_1_xy = player[player[:,0] == team_ids[0]]
        team_1_xy = team_1_xy[:,[2,3]]
        team_1_xy_mean[jj,:] = np.mean(team_1_xy,0)
        plt.plot(team_1_xy_mean[:jj+1,0],team_1_xy_mean[:jj+1,1],'o',
                color=color_dict[team_ids[0]][0],
                alpha=0.2)


        team_2_xy = player[player[:,0] == team_ids[1]]
        team_2_xy = team_2_xy[:,[2,3]]
        team_2_xy_mean[jj,:] = np.mean(team_2_xy,0)
        plt.plot(team_2_xy_mean[:jj+1,0],team_2_xy_mean[:jj+1,1],'o',
                color=color_dict[team_ids[1]][0],
                alpha=0.2)

        plt.xlim([0,94])
        plt.ylim([0,50])

        plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)

        # save image
        save_path = self.game_dir + 'game' + str(self.gameid) + '/' + 'predevent' + str(event_number) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'pred' + str(moment_number) + '.png')
        plt.cla()
        plt.close(fig)
        # return ax

def make_video(images, outvid, fps=20):
    ''' 
    Grabbed from here:
        http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    '''
    # Determine the width and height from the first image
    import cv2

    frame = cv2.imread(images[0])
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(outvid, fourcc, float(fps), (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame) # Write out frame to video
        # cv2.imshow('video',frame)
        # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        #     break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(outvid))


def plot_check(single_game, plt_ind=0, extreme=3):
    '''  
        Use plot to check if the game (list of events where each event is a list of moments) data
        is correct or not
    ''' 
    # cerate a simple plot shows the trajectory
    assert plt_ind < len(single_game), 'The plotting index is larger than the length of the game.'
    g = single_game[plt_ind]
    plt.figure(figsize=(5,7))
    extreme = int(extreme)
    # create color scheme
    c = ['b']*10*extreme + ['r']*10*extreme
    for i in range(0, extreme*10*2, 2): # extreme=3, palyers=10, x,y=2
        x_i, y_i = g[:, i], g[:, i+1]
        if sum(x_i) !=0 and sum(y_i) != 0:
            for k in range(0, len(x_i)):
                plt.plot(x_i[k], y_i[k], linestyle="None", marker="o", markersize=k/3, color=c[i])
    plt.grid(True)