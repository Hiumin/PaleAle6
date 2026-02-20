# Hieu Minh Truong modified
import os
from typing import List

import numpy as np
import pandas as pd
# import pandas as pd
import torch
import torch.nn as nn
from torch import optim


# from torch.utils.data import DataLoader

# from dataset.domainLinker_dataset import DomainLinkerDataset, Sequence, collate_fn
from dataset.utils import read_plm

from utils.common import dump_list2json, read_json2list
from params.filePath import paramF
import params.hyperparams_cbrcnn as paramH

import json

from model.utils import load_checkpoint

# import model
import importlib

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# HMT: Remove default model_folder value
def generate_models(list_modelInfo: list, model_folder):
    '''
    Given a list trained models information. Generate a
    params:
        list_modelInfo - list of dictionaries. [{}, {}, ...]
                        - for cnn model: {'model_name':, 'net_name':, 'net_type':, 'lr':, 'dropout':, 'featureType'}
                        - for rnn model: {'model_name':, 'net_name':, 'net_type':, 'n_features':, 'lr':,
                        'hidden_size':, 'num_layers':, 'bidirectional': }
    return:
        modes - list, list of models
    '''
    models = []
    for model_info in list_modelInfo:
        net_name = model_info['net_name']
        net_type = model_info['net_type']
        model_pth = model_info['model_pth']
        # import model
        model = importlib.import_module(net_name)

        # Instantiate the model
        '''
        if net_type == 'cnn':
            n_features, _ = get_numFeature(model_info['featureType'])
            net = model.Net(in_features=n_features, dropout=model_info['dropout']).to(device)
        elif net_type == 'rnn':
            net = model.Net(model_info['n_features'], hidden_size=model_info['hidden_size'], num_layers=model_info['num_layers'], bidirectional=model_info['bidirectional']).to(device)
        '''
        net = model.Net(paramH.n_features).to(device)

        optimizer = optim.Adam(net.parameters(), lr=model_info['lr'])
        # model_pth = os.path.join(paramF.path_output, f"trained_model/{model_info['featureType']}/{model_info['model_name']}.pth{model_info['fold']}")
        net, optimizer, start_epoch, losslogger = load_checkpoint(net, optimizer, model_pth)
        # Set all models to evaluation mode
        net.eval()
        models.append(net)
    return models

def remove_excluded(pred, label):

    true_label = []
    true_pred = []
    for i, l in enumerate(label):
        if l!=-1:
            true_label.append(l)
            true_pred.append(pred[i])
    return true_pred, true_label

# Prediction
# HMT: Change function parameters
# HMT: Move the section on path decoding to the wrapper function
def get_true_pred_and_label(model, path_embedded_decoded):
    '''
    Given the embeded file path, generate predictions.
    params:
        path_seq_embedded - path to embedded sequence file, .../entity_id.npy

    return:
        pred - the predictor output.
    '''

    seq_embedded = read_plm(path_embedded_decoded)
    data = seq_embedded.T.unsqueeze(0).float()
    data = data.to(device)
    pred = model(data).tolist()[0]

    return pred

# Define a function to perform ensemble prediction
def ensemble_predict(models, list_modelInfo, entity_id, label):
    '''
    Given the input data, and a list of model, generate the predictions from all of the models, average the predictions as the final output.

    params:
        models - list of models in evaluation mode.
        list_modelInfo - list of dictionaries. [{}, {}, ...]
        entity_id -
    '''
    predictions = []
    with torch.no_grad():
        for model, model_info in zip(models, list_modelInfo):
            feature_type = model_info['featureType']
            pred = get_true_pred_and_label(model, entity_id, feature_type)
            true_pred, true_label = remove_excluded(pred, label)
            predictions.append(true_pred)
        # Average the predictions (you can use other strategies like weighted averaging)
        ensemble_prediction = torch.tensor(predictions).mean(dim=0)
    return ensemble_prediction, true_label

# Define a function to perform ensemble prediction
def ensemble_predict_new(models, list_modelInfo, entity_id, feature_path, feature_type, injson_basename):
    '''
    Given the input data, and a list of model, generate the predictions from all of the models, average the predictions as the final output.

    params:
        models - list of models in evaluation mode.
        list_modelInfo - list of dictionaries. [{}, {}, ...]
        entity_id -
    '''
    print(entity_id)
    predictions = []
    with torch.no_grad():
        for model, model_info in zip(models, list_modelInfo):
            feature_type = model_info['featureType']

            # HMT: Use dynamic references to feature_path.
            # Also include injson_basename in the path since multi-sequence jsons are split into entities.
            if feature_type=='protTrans':
                path_embedded = os.path.join(feature_path, 'protTrans', injson_basename, '{}.npy'.format(entity_id))
            elif feature_type=='onehot':
                path_embedded = os.path.join(feature_path, 'onehot', injson_basename, '{}.npy'.format(entity_id))
            elif feature_type=='esm2':
                path_embedded = os.path.join(feature_path, 'esm2', injson_basename, '{}.npy'.format(entity_id))

            # HMT: Convert file name to Unicode to minimize issues with special characters.
            path_embedded_decoded = path_embedded.encode('utf-8')

            # HMT: adapt function call according to the modifications above
            true_pred = get_true_pred_and_label(model, path_embedded_decoded)
            predictions.append(true_pred)
        # Average the predictions (you can use other strategies like weighted averaging)
        ensemble_prediction = torch.tensor(predictions).mean(dim=0)
    return ensemble_prediction
