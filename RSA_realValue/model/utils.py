import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from torch.utils.data import DataLoader
from dataset.ss_dataset import Sequence
import params.filePath as paramF
import params.hyperparams as paramH
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trim_padding_and_flat(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_trimmed_pred = np.array([])
    for i, seq in enumerate(sequences):
        # tmp_pred = pred[i][:len(seq)].cpu().detach().numpy()
        all_target = np.concatenate([all_target, seq.clean_target])
        # all_trimmed_pred = np.concatenate([all_trimmed_pred, tmp_pred])
    all_trimmed_pred = pred.cpu().detach().numpy()
    return all_target, all_trimmed_pred

def concat_target_and_output(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_pred = np.array([])
    for i, seq in enumerate(sequences):
        all_target = np.concatenate([all_target, seq.clean_target])
    pred = pred.squeeze(0)
    all_pred = pred.cpu().detach().numpy()
    return all_target, all_pred

def get_targetPred(sequences: List[Sequence], pred):    
    if paramH.padding:
        target, pred = trim_padding_and_flat(sequences, pred)
    else:
        target, pred = concat_target_and_output(sequences, pred)
    return target, pred


def batch_auc(target, pred):
    '''
    Given target&pred, calculate AUC score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        auc - float, auc score.
    '''
    # print(f'target: \n{target}')
    # print(f'pred: \n{pred}')
    #change wafa made
    target = np.array(target)
    pred = np.array(pred)
    #end my changes 
    print(target.shape)
    print(pred.shape)
    
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def batch_acc(target, pred):
    '''
    Given target&pred, calculate accuracy score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        acc - float, accuracy score.
    '''
    # Get the unique classes present in y_true
    acc = metrics.top_k_accuracy_score(target.astype(int), pred, k=1)
    # acc = metrics.accuracy_score(target, pred)
    return acc

def batch_pcc(target, pred):
    '''
    Given target&pred, calculate accuracy score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        acc - float, accuracy score.
    '''
     #res = stats.pearsonr(target, pred)
    pcc, _ = stats.pearsonr(target, pred)
     #print(res)
    #pcc = res.correlation
    return pcc

def get_batch_PreTargetList(pred, target, lens):
    '''
    if batch_size>1, ignore the padding regions and get the actural pred and target lists.

    params:
        pred - list, list of padded prediction values.
        target - list, list of padded target values.
        lens - list, true lengths of the sequences.
    return:
        pre_list - list, all predictions for multiple sequences
        target_list - list, all true values for multiple sequences
    '''
    pre_list = []
    target_list = []
    
    for p, t, l in zip(pred, target, lens):
        pre_list += p[:l].tolist()
        target_list += t[:l].tolist()
        
    return pre_list, target_list

# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        # seq_loss = criterion(output[i][:len(seq)], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        target = torch.tensor(seq.clean_target, device=device, dtype=torch.float)
        seq_loss = criterion(output[i].squeeze(-1), target)
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)

# save and load model
def save_checkpoint(net, optimizer, Loss, EPOCH, PATH):
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
                }, PATH)
    
def load_checkpoint(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(losslogger, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch, losslogger

def load_model(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("=> (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch

def count_modelParams(net):
    '''
    Given a mdoel, count the parameters inside this model.
    params:
        net - nn. Module

    return:
        int, number of params.
    '''
    return sum(p.numel() for p in net.parameters())
