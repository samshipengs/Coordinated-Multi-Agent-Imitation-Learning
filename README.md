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
    
Originally we would like to use the play-by-play data to do the data processing but it turns out the play-by-play data itself is not accurate. For example, In game 0021500196, event 2, `'time_left': [705, 704, 685, 684]}, 'event_str': ['miss', 'rebound', 'miss', 'rebound']`,
<p align="center">
  <img src="images/inaccurate.gif">  
</p>  
For 685.0 the shot clock is at 21.77, which at the time the shot was already missed for a while and the defending team got rebound and was already switching to offense. The event miss should have been marked right after 24s shot clock reset. This is resonable to human eyes but would certain affect the model learning.

### Features
1. Besides Cartesian coordiantes for basketball and all the players from the data, we also add Polar coodinates. 
2. The distance of each players to the ball and hoop in polar coordiantes.
3. Add velocities for both players and basketball (in Cartesian coordinates).

You can check out the details in `create_static_features` and `create_dynamic_features` functions form `features.py`.

Below is an example plot of a game event,
<p align="center">
  <img src="images/trajectory.png" width="50%">  
</p>  
<em>Blue is the defending team, red is the opponent and the green one is the basketball. The arrow indicates the velocity vector for each player. The black circle is the hoop. The smaller the dot is the earlier player is in the sequence</em>

### Hidden Structure Learning
Finally we will get into how we want to build the model. It may seem like how we want to approach this i.e. feed the input sequence of data into a LSTM where the label for each current time step is the input of the next time step. However, there are two major issues:

1. Since we are training on input data that contains multiple agents, we need to consider the order of the input. 
2. A standard one-to-one or many-to-one would not have practical use since in real game we would like to have predictions for next at least several time steps instead of just one prediction at a time.

In this section we mainly talk about the first issue. The input data point at each time step looks like,
<a href="https://www.codecogs.com/eqnedit.php?latex=(p_{1,x},&space;p_{1,y},&space;p_{1,vx},&space;p_{1,vy},&space;\cdots,&space;p_{2,x},&space;p_{2,y},&space;p_{2,vx},&space;p_{2,vy},&space;\cdots,&space;p_{10,x},&space;p_{10,y},&space;p_{10,vx},&space;p_{10,vy},&space;\cdots)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(p_{1,x},&space;p_{1,y},&space;p_{1,vx},&space;p_{1,vy},&space;\cdots,&space;p_{2,x},&space;p_{2,y},&space;p_{2,vx},&space;p_{2,vy},&space;\cdots,&space;p_{10,x},&space;p_{10,y},&space;p_{10,vx},&space;p_{10,vy},&space;\cdots)" title="(p_{1,x}, p_{1,y}, p_{1,vx}, p_{1,vy}, \cdots, p_{2,x}, p_{2,y}, p_{2,vx}, p_{2,vy}, \cdots, p_{10,x}, p_{10,y}, p_{10,vx}, p_{10,vy}, \cdots)" /></a>

we are supposed to feed into data that has consistent order to the model, otherwise the model is going to have a hard time to learn anything. This is known as "index free" multi-agent system. How do we define the order then? by their height, weight or their assigned roles e.g. Power-forward or Point-guard? Using the pre-defined roles sounds more reasonable but they may change during the actual game play. So instead of using fixed roles, the team of this paper suggested to learn the _hidden states/roles_ for each players.    

Here we will make use of the [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm) library ([pomegranate](https://pomegranate.readthedocs.io/en/latest/) looks like a good option too). We train a Hidden Markov model which would predict the hidden state for each time step, this is done by using [Baum–Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) from which we can know the emission probabilities for each hidden roles.  

Naturally we do not need to bother with the emission distribution, Viterbi algorithm would help us to find the most likely sequence of hidden roles. However since we are trying to assign hidden roles to each player then it is possible that different players get assigned the same hidden role (indeed it happened when I run Viterbi to get the sequence of assigned roles). More concretely, for each player at each time step we assign a hidden role:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\begin{bmatrix}&space;p^{(1)}_{1}&space;&&space;p^{(2)}_{1}&space;&&space;p^{(3)}_{1}&space;&&space;\dots&space;&&space;p^{(T)}_{1}&space;\\&space;p^{(1)}_{2}&space;&&space;p^{(2)}_{2}&space;&&space;p^{(3)}_{2}&space;&&space;\dots&space;&&space;p^{(T)}_{2}\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;p^{(1)}_{n}&space;&&space;p^{(2)}_{n}&space;&&space;p^{(3)}_{n}&space;&&space;\dots&space;&&space;p^{(T)}_{n}&space;\end{bmatrix}&space;\Rightarrow&space;\begin{bmatrix}&space;h^{(1)}_{1}&space;&&space;h^{(2)}_{4}&space;&&space;\cdots&space;&&space;h^{(t)}_{i}&space;\dots&space;\\&space;h^{(1)}_{1}&space;&&space;h^{(2)}_{3}&space;&&space;\cdots&space;&&space;h^{(t)}_{j}&space;\dots&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\\&space;h^{(1)}_{6}&space;&&space;h^{(2)}_{3}&space;&&space;\cdots&space;&&space;h^{(t)}_{k}&space;\dots&space;\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\begin{bmatrix}&space;p^{(1)}_{1}&space;&&space;p^{(2)}_{1}&space;&&space;p^{(3)}_{1}&space;&&space;\dots&space;&&space;p^{(T)}_{1}&space;\\&space;p^{(1)}_{2}&space;&&space;p^{(2)}_{2}&space;&&space;p^{(3)}_{2}&space;&&space;\dots&space;&&space;p^{(T)}_{2}\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;p^{(1)}_{n}&space;&&space;p^{(2)}_{n}&space;&&space;p^{(3)}_{n}&space;&&space;\dots&space;&&space;p^{(T)}_{n}&space;\end{bmatrix}&space;\Rightarrow&space;\begin{bmatrix}&space;h^{(1)}_{1}&space;&&space;h^{(2)}_{4}&space;&&space;\cdots&space;&&space;h^{(t)}_{i}&space;\dots&space;\\&space;h^{(1)}_{1}&space;&&space;h^{(2)}_{3}&space;&&space;\cdots&space;&&space;h^{(t)}_{j}&space;\dots&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\\&space;h^{(1)}_{6}&space;&&space;h^{(2)}_{3}&space;&&space;\cdots&space;&&space;h^{(t)}_{k}&space;\dots&space;\\&space;\end{bmatrix}" title="\begin{bmatrix} p^{(1)}_{1} & p^{(2)}_{1} & p^{(3)}_{1} & \dots & p^{(T)}_{1} \\ p^{(1)}_{2} & p^{(2)}_{2} & p^{(3)}_{2} & \dots & p^{(T)}_{2}\\ \vdots & \vdots & \vdots & \ddots & \vdots \\ p^{(1)}_{n} & p^{(2)}_{n} & p^{(3)}_{n} & \dots & p^{(T)}_{n} \end{bmatrix} \Rightarrow \begin{bmatrix} h^{(1)}_{1} & h^{(2)}_{4} & \cdots & h^{(t)}_{i} \dots \\ h^{(1)}_{1} & h^{(2)}_{3} & \cdots & h^{(t)}_{j} \dots \\ \vdots & \vdots & \vdots & \vdots & \\ h^{(1)}_{6} & h^{(2)}_{3} & \cdots & h^{(t)}_{k} \dots \\ \end{bmatrix}" /></a>
</p>

Notice that how player 1 and 2 both get assigned to hidden role 1 for initial time step, and player 2 and 5 get assigned to the same hidden role 3. We cannot have this assignment as we will need the hidden role to order the players, so instead of having the hard assignment for each player we employ [linear assignment](https://en.wikipedia.org/wiki/Assignment_problem) techniques, more specifically [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) to assign the hidden role. 

We do so by first compute the Euclidean distance (you can also try cosine similarity) from each player's data point at certain timestep to the center of each hidden roles distribution, which we assumed to be (mixture) multivariavte Gaussain. Then we use this as the cost matrix and apply Hungarian algorithm.

**try to create a vis for the hidden state**

### Imitation Learning
We are hoping the model can learn or mimic the trajectory by training on players tracking data. Naturally we make use of LSTM for this task. One common example of the LSTM architecture is to take a sequence of length T of state S and outputs the action for each next time step.
<p align="center">
  <img src="images/reg_lstm.png">  
</p> 
however, the first obvious issue is that in real game we do not have the sequence of player states (unlike in a machine translation problem where you have the complete sentence ready), which are exactly the values we are trying to predict for. If we have these values then we do not to predict them anyway. So simply we do not have the input for a sequence of inputs.

What one could do is to train the model based on available data, use the predicted output of current time step as the next time step input during run time, that is instead of using true value as next time step input we use the output from previous time step. This is doable and looks okay but in run time the model will get baffled by the _drifting or compound error_. As the prediction goes on for longer time steps, the prediction error gets larger and larger to the point where the prediction would be really far off from the realistic trajectories. This happens although the loss value is small in training time. 
<p align="center">
  <img src="images/drifting_lstm.png">  
</p> 
We demonstate this through a simple experiment. Below is a sine signla being added Gaussian noise with mean=2 and standard deviation=1. 
<p align="center">
  <img src="images/sineg.png">  
</p>
First we apply regular lstm that uses ground truth as the real input for every time step, the prediction result looks pretty _good_,
<p align="center">
  <img src="images/naive.png">  
</p>
but this is deceptive because in real settings we need to predict multiple steps ahead instead of relying ground truth. So if we take the trained model and simply make predictions based on previous result, the prediction quickly converges to the mean of the Gaussian noise,
<p align="center">
  <img src="images/drift_error.png">  
</p>

So the paper proposed to let the model see for longer time steps and experience this drifting error during train time. We first start training the regular lstm model where each time step input is ground truth. Then we extend the horizon where the input uses i.e. during training time we use the current time step output as the next step input. We increase the horzion by 1 and repeat. This gives model the experience of handling drifting error in train time, which leads to better performance in real run time setting.

for the sine wave example, the test result becomes much better when we gradually increase the horizon from 0 to 6,
<p align="center">
  <img src="images/horizon_fix.png">  
</p>

To illustrate this using network connections:
Step 1 | Step 2 | Step 3
:----------------------:|:----------------------:|:----------------------:|
![](images/step1.png)  |  ![](images/step2.png) | ![](images/step3.png)

