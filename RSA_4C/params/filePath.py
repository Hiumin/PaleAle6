import os
from params.hyperparams import *
#from params.hyperparams_cbrcnn import *


# Hieu Minh Truong modifications:
# Define a class to collect all the file paths. After importing the class
# into the parent script, paths can then be supplied to initialize an object
# of this class. This is much more adaptable than hard-coding paths into a module.

class paramF:
    def __init__(self, injson, preddir, outprefix, predmode):
        # Allowed values for prediction mode: RSA_2C, RSA_4C, RSA_realValue
        self.scriptdir = os.path.realpath('.')
        self.injson = injson
        self.preddir = preddir
        self.outprefix = outprefix
        self.predmode = predmode

        # Path to folder containing models for the selected prediction mode
        self.path_predictor = os.path.join(self.scriptdir, self.predmode)
        # Path to model training output (not prediction output for actual data)
        self.path_training = os.path.join(self.path_predictor, 'training')
        self.path_models = os.path.join(self.path_training, 'trained_model')
        # Path to training logs
        self.path_log = os.path.join(self.path_training, 'log', featureType)

        # Path to prediction output for the actual data
        # This directory structure is pre-generated in the PaleAle6.sh wrapper script
        self.path_dataset_test = os.path.join(self.preddir, 'json', injson)
        self.path_features = os.path.join(self.preddir, 'features')
        self.path_embedded_eval = os.path.join(self.path_features, 'evaluation')
        self.path_embedded_protTrans = os.path.join(self.path_features, 'protTrans')
        self.path_embedded_onehot = os.path.join(self.path_features, 'onehot')
        self.path_embedded_esm2 = os.path.join(self.path_features, 'esm2')

    def path_plots(self, featureType, model_name):
        self.plots_dir = os.path.join(self.path_training, 'plots', featureType, model_name)
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)
        return self.plots_dir
    def path_model_file(self, featureType, model_name):
        self.model_file = os.path.join(self.path_models, featureType, model_name+'.pth')
        return self.model_file
    def path_auc_loss_file(self, featureType, model_name):
        self.auc_loss_file = os.path.join(self.path_models, featureType, model_name+'.csv')
        return self.auc_loss_file


# HMT: Below is the old stuff from the original git repo, where paths were hard-coded.
"""
ROOT = os.path.realpath('..')

# Data
path_data = os.path.join(ROOT, 'data')

# 1. dataset
path_dataset = os.path.join(path_data, 'dataset')
path_dataset_train = os.path.join(path_dataset, 'train_dataset.json')
path_dataset_test = os.path.join(path_dataset, 'test_dataset.json')

# 2. features
path_features = os.path.join(path_data, 'features')
path_embedded_eval = os.path.join(path_features, 'evaluation')
path_embedded_protTrans = os.path.join(path_features, 'protTrans')
path_embedded_onehot = os.path.join(path_features, 'onehot')
# path_embedded_hmm = os.path.join(path_features, 'hmm')
path_embedded_esm2 = os.path.join(path_features, 'esm2')

# 3. model
path_predictor = os.path.join(ROOT, 'RSA_2C')
# HMT Note: this is the model training output, not final predictions.
path_output = os.path.join(path_predictor, 'output')

plots_dir = os.path.join(path_output, f'plots/{featureType}/{model_name}')
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

model_pth = os.path.join(path_output, f'trained_model/{featureType}/{model_name}.pth')
auc_loss_pth = os.path.join(path_output, f'auc_loss/{featureType}/{model_name}.csv')


# Log
path_log = os.path.join(path_output, f'log/{featureType}')

# 4. uniprot
path_uniprot = os.path.join(path_data, 'uniprot')

# 5. predictions
path_pred = os.path.join(path_output, 'pred')
path_pred_plot = os.path.join(path_pred, 'plots')
path_pred_files = os.path.join(path_pred, 'files')
"""
