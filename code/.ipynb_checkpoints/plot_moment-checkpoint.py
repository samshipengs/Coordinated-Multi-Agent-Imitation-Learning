import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cPickle as pickle
import _pickle as pickle

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





gameid='0021500463'

# directories
# CHANGE HERE
data_dir = '../'
game_dir = data_dir+'data/'
court_path = data_dir + 'nba_court_T.png'


#open the pickle file
# with open(game_dir+gameid+'.pkl', 'rb') as handle:
#     datta = pickle.load(handle)

datta = pd.read_pickle(game_dir+gameid+'.pkl')

num_events = len(datta['events'])

player_fields = datta['events'][0]['home']['players'][0].keys()

# CHANGE THIS
# specify an event number
ii = 100


home_players = pd.DataFrame(data=[i for i in datta['events'][0]['home']['players']], columns=player_fields)
away_players = pd.DataFrame(data=[i for i in datta['events'][0]['visitor']['players']], columns=player_fields)
players = pd.merge(home_players, away_players, how='outer')
jerseydict = dict(zip(players.playerid.values, players.jersey.values))

# get the position of the players and the ball throughout the event
ball_xy = np.array([x[5][0][2:5] for x in datta['events'][ii]['moments']]) #create matrix of ball data
player_xy = np.array([np.array(x[5][1:])[:,:4] for x in datta['events'][ii]['moments']]) #create matrix of player data

# get the play by play data for this clip
playbyplay = datta['events'][ii]['playbyplay']


team_1_xy_mean = -np.ones((len(player_xy),2))
team_2_xy_mean = -np.ones((len(player_xy),2))


# CHANGE THIS
# plot a certain frame:
jj = 200


print('event ' + str(ii) + '/' + str(num_events) + ", moment: "+ str(jj) + '/'+ str(len(player_xy)))
fig = plt.figure()
ax = plt.gca() #create axis object


img = mpimg.imread(court_path)  # read image. I got this image from gmf05's github.

plt.imshow(img, extent=[0,94,0,50], zorder=0)  # show the image.

# get player and ball data for the momenet
ball = ball_xy[jj]
player = player_xy[jj]


# plot clock info
clock_info = ax.annotate('', xy=[94.0/2 - 6.0/1.5 +0.1, 50 - 6.0/1.5 -0.35],
    color='black', horizontalalignment='center', verticalalignment='center')

if not datta['events'][ii]['moments'][jj][0] == None:
    quarter = datta['events'][ii]['moments'][jj][0]
else:
    quarter = 0

if not datta['events'][ii]['moments'][jj][2] == None:
    game_clock = datta['events'][ii]['moments'][jj][2]
else:
    game_clock = 0

if not datta['events'][ii]['moments'][jj][3] == None:
    game_shot = datta['events'][ii]['moments'][jj][3]
else:
    game_shot = 0

clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
    quarter,
    int(game_clock) % 3600 // 60,
    int(game_clock) % 60,
    game_shot)
clock_info.set_text(clock_test)

# the event title
temp = str(datta['events'][ii]['home']['abbreviation'])+\
    ' vs. ' + str(datta['events'][ii]['visitor']['abbreviation'])+\
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


plt.show()