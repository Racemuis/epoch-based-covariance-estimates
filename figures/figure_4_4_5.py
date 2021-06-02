# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:33:00 2021

Figure 4.4 and 4.5: Visualisation of the covariance matrices from the 3 pipelines.
"""

import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.spot_pilot import SpotPilotData
from moabb.paradigms import P300
from transformers import XdawnFilter
from utilities import shrinkage_regularization

# For sanity check
from pyriemann.estimation import XdawnCovariances

mne.set_log_level(False)

def covariance(x):
    # de-mean returns
    [n,t]=x.shape

    meanx=np.mean(x)
    x_s=x-meanx
    
    # Compute sample covariance matrix
    sample = 1/t * np.dot(x_s, x_s.T)
    return sample
        
# Extract configurations  
local_cfg_file = r'./configurations/local_config.yaml'
analysis_cfg_file = r'./configurations/analysis_config.yaml'
with open(local_cfg_file, 'r') as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)    
with open(analysis_cfg_file, 'r') as conf_f:
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)


prepro_cfg = ana_cfg['default']['data_preprocessing']
data_path = local_cfg['data_root']

prepro_cfg = ana_cfg['default']['data_preprocessing']

# Read data
dataset = SpotPilotData(load_single_trials=True, reject_non_iid=True)
dataset.path = data_path
subjects = [1]
paradigm = P300(resample=prepro_cfg['sampling_rate'], fmin=prepro_cfg['fmin'], fmax=prepro_cfg['fmax'],
                    reject_uv=prepro_cfg['reject_uv'], baseline=prepro_cfg['baseline'])

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

# Get data from a single session
ix = metadata.session == np.unique(metadata.session)[0]
print(np.unique(metadata.session)[0])
X = X[ix]
y = y[ix]
print(X.shape)
y_label = np.zeros(y.shape[0])
for i in range(y.shape[0]):
    if y[i] == 'NonTarget':
        y_label[i] = 0
    else:
        y_label[i] = 1

f =  XdawnFilter(nfilter=5, classes=[1])
X_p = f.fit_transform(X, y_label)

# For sanity checking, not included in the plot
c = XdawnCovariances(nfilter = 5, classes = [1])
X_c = c.fit_transform(X, y_label)

non_target = X_p[np.where(y_label==0)][0]
target = X_p[np.where(y_label==1)][0]

vmax = 0
vmin = 0

for response, i in zip([non_target, target], range(2)):
    cov = covariance(response)
    resp = shrinkage_regularization(response, "manifold", True, "lower right")[0]
    sup =  shrinkage_regularization(response, "manifold", True, "both")[0]
    
    vmax = np.ceil(np.max([np.max(cov), np.max(resp), np.max(sup), vmax]))
    vmin = np.floor(np.min([np.min(cov), np.min(resp), np.min(sup), vmax]))
    
    fig, axes = plt.subplots(1, 3, figsize =(30,10), dpi = 200)
    cbar_ax = fig.add_axes([.91, .2, .03, 0.6])
    
    axes[0].set_title("Tangent Space LDA", size = 20, color = 'tab:blue')
    axes[1].set_title("Response Shrinkage", size = 20, color = 'tab:orange')
    axes[2].set_title("Super-trial Shrinkage", size = 20, color = 'tab:green')
    
    fig.text(0.5, 0.1, 'Virtual channel', ha='center', size = 20)
    fig.text(0.1, 0.5, 'Virtual channel', va='center', rotation='vertical', size = 20)
    
    sns.heatmap(cov, vmin = vmin, vmax = vmax, annot=True, cbar = False, fmt=".4f", ax = axes[0], square = True, cmap = 'viridis')
    sns.heatmap(resp, vmin = vmin, vmax = vmax, annot=True, cbar = False, fmt=".4f", ax = axes[1], square = True, cmap = 'viridis')
    sns.heatmap(sup, vmin = vmin, vmax = vmax, annot=True, fmt=".4f", ax = axes[2], square = True, cbar_ax = cbar_ax, cmap = 'viridis')
    
    fig.text(0.5, .9, '{}'.format("Non-target" if i == 0 else "Target"), size=30)   
    plt.show()
    
    sns.heatmap(cov, vmin = vmin, vmax = vmax, annot=True, fmt=".4f",  square = True, cmap = 'viridis')
    plt.xlabel("Virtual channel")
    plt.ylabel("Virtual channel")
    plt.title("{} - Tangent Space LDA".format("Non-target" if i == 0 else "Target"))
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.set_dpi(100)
    plt.show()
    
    sns.heatmap(resp, vmin = vmin, vmax = vmax, annot=True, fmt=".4f",  square = True, cmap = 'viridis')
    plt.title("{} - Response Shrinkage".format("Non-target" if i == 0 else "Target"))
    plt.xlabel("Virtual channel")
    plt.ylabel("Virtual channel")
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.set_dpi(100)
    plt.show()
    
    sns.heatmap(sup, vmin = vmin, vmax = vmax, annot=True, fmt=".4f", square = True, cmap = 'viridis')
    plt.title("{} - Super-trial Shrinkage".format("Non-target" if i == 0 else "Target"))
    plt.xlabel("Virtual channel")
    plt.ylabel("Virtual channel")
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.set_dpi(100)
    plt.show()