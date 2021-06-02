# -*- coding: utf-8 -*-
"""
Figure 2.1:
Generate eigenvalue spectra for different ratios of the number of samples
with respect to the number of features. 

Created on Sun Mar 28 21:08:19 2021
"""
from matplotlib import pyplot as plt
import numpy as np
from utilities import estimate_covariance, generate_covariance
    
# Set hyperparameters
np.random.seed(42)
n_channels = 100
n_samples = [10, 20, 50, 100, 200, 500]
target = generate_covariance(n_channels)


fig, axes = plt.subplots(2, 3, figsize =(6,4), dpi = 200)
for i, ax in zip(range(len(n_samples)), axes.flatten()):
    # Generate data
    x = np.random.multivariate_normal(np.zeros(n_channels), target, n_samples[i])
    
    # Plot the eigenvaluespectrum of the sample covariance matrix
    scm = estimate_covariance(x)
    eigvals, _ = np.linalg.eig(scm)
    ax.plot(range(len(eigvals)), sorted(np.real(eigvals), reverse=True), label = 'Sample covariance matrix')
    
    # Plot the eigenvaluespectrum of the population covariance matrix
    eigvals, _ = np.linalg.eig(target)
    ax.plot(range(len(eigvals)), sorted(np.real(eigvals), reverse=True), '--', label = 'True covariance matrix')
    ax.set_title("p/n = {}\ndet(SCM) = {:5.1e}".format(n_channels/n_samples[i], np.abs(np.linalg.det(scm))), fontsize=10)
    #ax.set_title("p/n = {}".format(n_channels/n_samples[i]), fontsize=10) Used in presentation

# Remove overlap
plt.tight_layout()

# Add common lables and legend
fig.text(0.5, 0.00, 'Eigenvalue index', ha='center')
fig.text(0.00, 0.5, 'Eigenvalue', va='center', rotation='vertical')
axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)

# Shift plots down for common title
fig.subplots_adjust(top=0.8)
fig.suptitle('The eigenvalue spectra for different ratios of p to n')

plt.show()
