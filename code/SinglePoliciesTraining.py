# SinglePoliciesTraining.py
import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time

from datetime import datetime
import numpy as np
import pandas as pd
import glob, os, sys, math, warnings, copy, time, glob, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# customized ftns 
from preprocessing import process_game_data
from sequencing import get_sequences, get_minibatches, iterate_minibatches, subsample_sequence
from utilities import *
from model import *
from train import train_all_single_policies
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')
# ---------------------------------------------------------
# directories
main_dir = '../'
game_dir = main_dir+'data/'
Data = LoadData(main_dir, game_dir)
models_path = './models/' 

# Pre-process 
all_games_id = [i.split('/')[-1].split('.')[0] for i in glob.glob('../data/*.pkl')]

event_threshold = 100
subsample_factor = 2

game_files = './all_games_{0}_{1}_{2}.pkl'.format(len(all_games_id), event_threshold, subsample_factor)
if os.path.isfile(game_files):
    with open(game_files, 'rb') as f:
        game_data = pickle.load(f)
else:
    game_data = process_game_data(Data, all_games_id, event_threshold, subsample_factor)
    with open(game_files, 'wb') as f:
        pickle.dump(game_data, f)
print('Final number of events:', len(game_data))

# Build graph and starts training for all single policies
sequence_length = 50
overlap = 25
batch_size = 128

hyper_params = {'use_model': 'dynamic_rnn_layer_norm',
                'batch_size': batch_size,
                'sequence_length': sequence_length,
                'overlap': overlap,
                'state_size': [128, 128],
                'use_peepholes': None,
                'input_dim': 179,
                'dropout_rate':0.6,
                'learning_rate': 0.0001,
                'n_epoch': int(1e3)}

train_all_single_policies(game_data, hyper_params, models_path)