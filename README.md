## Coordinated-Multi-Agent-Imitation-Learning

Toronto Raptors had created a ghosting system that would help coaching staff to analyze defend plays better. The game is recorded by camera system above the arena, staff would mark the position of they player where they thought the player should have been and this is the ghost of the player. However, this involves a lot of mannual annotations. In the coordinated multi-agent imitation learning, a data driven method was proposed. (For more details of the Raptors' ghosting system see [Lights, Cameras, Revolution](http://grantland.com/features/the-toronto-raptors-sportvu-cameras-nba-analytical-revolution/)). 

So in this repo we attempt to implement the paper [Coordinated-Multi-Agent-Imitation-Learning
](https://arxiv.org/pdf/1703.03121.pdf) (or [Sloan version](https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20170228130457/Data-Driven-Ghosting-using-Deep-Imitation-Learning-Paper1.pdf)) with Tensorflow. 

### Introduction
We are aiming to predict the movements or trajectories of defending players for a given team (in principle, we should also be able to create model that predicts offense trajectoy, but defending players were used for both the original ghosting work and also this paper. I assume the reason is that defending trajecotory is slightly easier to predict than offending).

In order to predict the trajectory, we need to roll out a sequence of prediction for the player's next action. The natural candidate to perform such task is Recurrent Neural Networks (LSTM more specifically), and the input data to the model will be a sequence of (x,y) coordinates of each players (both defendinging team and opponent).

The end result we would like to achieve is that, for a given game play suitation where team A is on defense, we can show what would another team B do, who presumably is the best defending team in the league. This is slightly different compaing to the original ghosting work done by Raptors. Instead of focusing on specifically what a player should do based on a coach experience, this work is modeling what another team would do in same suitation (again, in principle we could also model each specific player but that shall require a much larger data set and that is a more complicated task for the model to learn).

### Data
The update-to-date data is proprietary, but we found a tracking and play-by-play data for 42 Toronto Raptors games
played in Fall 2015 on this [link](http://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_2.pdf). We will use this data for our implementation. See the link for a detailed description of the data.

Below is a short preview of the data for game with id 0021500463:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>end_time_left</th>
      <th>home</th>
      <th>moments</th>
      <th>orig_events</th>
      <th>playbyplay</th>
      <th>quarter</th>
      <th>start_time_left</th>
      <th>visitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>702.31</td>
      <td>{'abbreviation': 'CHI', 'players': [{'playerid...</td>
      <td>[[1, 1451351428029, 708.28, 12.78, None, [[-1,...</td>
      <td>[0]</td>
      <td>GAME_ID  EVENTNUM  EVENTMSGTYPE  EVENTMS...</td>
      <td>1</td>
      <td>708.28</td>
      <td>{'abbreviation': 'TOR', 'players': [{'playerid...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>686.28</td>
      <td>{'abbreviation': 'CHI', 'players': [{'playerid...</td>
      <td>[[1, 1451351428029, 708.28, 12.78, None, [[-1,...</td>
      <td>[1]</td>
      <td>GAME_ID  EVENTNUM  EVENTMSGTYPE  EVENTMS...</td>
      <td>1</td>
      <td>708.28</td>
      <td>{'abbreviation': 'TOR', 'players': [{'playerid...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>668.42</td>
      <td>{'abbreviation': 'CHI', 'players': [{'playerid...</td>
      <td>[[1, 1451351444029, 692.25, 12.21, None, [[-1,...</td>
      <td>[2, 3]</td>
      <td>GAME_ID  EVENTNUM  EVENTMSGTYPE  EVENTMS...</td>
      <td>1</td>
      <td>692.25</td>
      <td>{'abbreviation': 'TOR', 'players': [{'playerid...</td>
    </tr>
  </tbody>
</table>
</div>

The main columns we use for building the model is *moments*, *quarter*, *home* and *visitor*. Moments contain the most information such as basketball location, all players locations and their team ID and player ID. Quarter is used in both input features and preprocessing. Home and visitor basically specifies the team name and ID which can be usful when validating the preprocessed data.

### Pre-processing
Not all the moments from the data set is used. Each event is supposed to describe a game play precisely but the given moments often contain frames that would not help the model. For examples, there are frames only ontain 8 or 9 players, or the basketball is out of bound, this is not allowed as the model expects a fixed input dimension. Many moments have frames that are not critical to decision making, e.g. dribbling before entering the half court, clocks being stopped etc. Shot clock sometimes has null value.   

ALso to make it easier for the model to learn, we perform some extra preprocessings. Such as, only model defending players and normalize the court to just half court, the reason is that the game swaps court after half-time which could confuse the model and game plays involvs whole court is more dynamic in nature so that it's harder to predict.

We list out each pre-processing details in the following:

1. Remove frames that do not contain 10 players and 1 basketball, and chunk the following frames as another event (same applies for any chunking in the subsequent processings).  
    You can find the function named `remove_non_eleven` does this in `preprocessing.py`.  
    This prevents players or basketball out of boundary. 
2. Chunk the moments from shotclock.  
    `chunk_halfcourt` does this in `preprocessing.py`  
    If the shotclock turns to 24 (shot clock reached) or 0 (resets, e.g. rebound or turnover), or shot clock is None or stopped, we remove them from the moments. Since the behavior of players differs dramatically at these time points.
3. Chunk moments to just half-court.  
    `chunk_halfcourt` in `preprocessing.py`   
    Remove all moments that are not contained within a half-court and change the x coordinates to be between 0 and 47 (NBA court is 50x94 feet).
4. Reorder data
    `reorder_teams` in `preprocessing.py`   
    Reorder the matrix in moments s.t. the first five players data are always from defending player.
    
## Features
1. Besides Cartesian coordiantes for basketball and all the players from the data, we also add Polar coodinates. 
2. The distance of each players to the ball and hoop in polar coordiantes.
3. Add velocities for both players and basketball (in Cartesian coordinates).




For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).


