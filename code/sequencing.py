# sequencing.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
import pandas as pd







# BELOW NOT TESTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


def get_minibatches(inputs, targets, batchsize, shuffle=True):
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


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
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


def subsample_sequence(events, subsample_factor, random_sample=False):
    if subsample_factor == 0:
        return events
    
    def subsample_sequence_(moments, subsample_factor, random_sample=False):#random_state=42):
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

    return [subsample_sequence_(ms, subsample_factor) for ms in events]