# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:11:30 2021

Figure 3.1: The comparison of estimators.
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from utilities import MSE, estimate_covariance, generate_covariance

# Set hyperparameters
np.random.seed(42)
n_channels = 31
n_iterations = 100
n_samples = np.arange(10,500,10)

# Generate the population covariance
target = generate_covariance(n_channels)
mean = np.zeros(target.shape[0])

lw_shrinkage = np.zeros((n_samples.shape[0], n_iterations))
lw_mse = np.zeros((n_samples.shape[0], n_iterations))
scm_mse = np.zeros((n_samples.shape[0], n_iterations))

for i in range(n_samples.shape[0]):
    for j in range(n_iterations):
        # Generate data 
        x = np.random.multivariate_normal(mean, target, n_samples[i]) 
        
        # Collect observations
        estimator = LedoitWolf(assume_centered = True).fit(x)
        lw_shrinkage[i,j] = estimator.shrinkage_
        lw_mse[i,j] = MSE(estimator.covariance_, target)
        scm_mse[i,j] = MSE(estimate_covariance(x), target)
fig, axes = plt.subplots(2, 1, figsize =(6,4), dpi = 200)
axes[0].errorbar(n_samples, np.mean(lw_shrinkage, axis = 1), yerr = np.std(lw_shrinkage, axis = 1)/np.sqrt(n_iterations), color = 'tab:orange', label = 'Shrinkage parameter', capsize = 3 )
axes[0].legend()
axes[0].set_xlabel('Number of samples')
axes[0].set_ylabel('Mean shrinkage parameter', fontsize = 8)
axes[0].get_yaxis().set_label_coords(-0.11,0.5)
axes[1].errorbar(n_samples, np.mean(lw_mse, axis = 1), yerr = np.std(lw_mse, axis = 1)/np.sqrt(n_iterations), color = 'tab:orange', label = 'Ledoit & Wolf estimator', capsize = 3)
axes[1].errorbar(n_samples, np.mean(scm_mse, axis = 1), yerr = np.std(scm_mse, axis = 1)/np.sqrt(n_iterations), color = '#1f77b4', label = 'Sample covariance matrix', capsize = 3 )
axes[1].legend()
axes[1].set_xlabel('Number of samples')
axes[1].set_ylabel('Mean squared error', fontsize = 8)
axes[1].get_yaxis().set_label_coords(-0.11,0.5)

fig.suptitle('Comparison of the Ledoit & Wolf estimator to the sample covariance matrix')

    

