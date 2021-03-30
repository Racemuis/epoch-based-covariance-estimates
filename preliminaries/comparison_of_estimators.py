# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:11:30 2021


The Riemannian distance cannot be used as the matrices for the low number of
samples are not positive semi-definite. This means that they have a negative
eigenvalue, which collides with the logarithm taken in the Riemannian distance.

https://www.quora.com/When-will-a-matrix-have-negative-eigenvalues-And-what-does-that-mean
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
from sklearn.covariance import LedoitWolf

# Metrics
def MSE(A, B, norm = 'fro'):
    """
    The Mean Squared Error (MSE) between two covariance matrices A and B.
    
    Parameters
    ----------
    A : ndarray of shape (n_channels, n_channels)
        A covariance matrix.
    B : ndarray of shape (n_channels, n_channels)
        A covariance matrix.
    norm :  an existing norm implemented by the numpy library.
            Default: Frobenius norm. 

    Returns
    -------
    int
        MSE of A and B, with resprect to the given norm.
    """
    return 1/A.shape[0] * np.linalg.norm(A - B, ord = norm)**2

def estimate_covariance(x):
    """
    Estimate the covariance matrix from the sample data. 
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Input array.

    Returns
    -------
    scm : ndarray of shape (n_channels, n_channels)
        The estimated sample covariance matrix.
    """
    scm = 1/(x.shape[0]-1)
    scm *= np.dot((x-np.mean(x, axis = 0)).transpose(), (x-np.mean(x, axis = 0)))
    return scm

def distance_riemann(A, B):
    """
    Riemannian distance between two covariance matrices A and B.
    Retreived from pyriemann.
    
    Parameters
    ----------
    A : ndarray of shape (n_channels, n_channels)
        A covariance matrix.
    B : ndarray of shape (n_channels, n_channels)
        A covariance matrix.

    Returns
    -------
    float
        Riemannian distance between A and B.
    """
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

def generate_covariance(n_channels):
    """
    Generate a symmetrical square matrix. (Based on SP assignment 2).

    Parameters
    ----------
    n_channels : int
        The dimension of the matrix.

    Returns
    -------
    covariance : ndarray of shape (n_channels, n_channels)
        The generated symmetrical square matrix.
    """
    temp = np.random.randn(n_channels, n_channels)
    covariance = np.dot(temp, temp.transpose()) 
    return covariance

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
fig, axes = plt.subplots(2, 1)
axes[0].errorbar(n_samples, np.mean(lw_shrinkage, axis = 1), yerr = np.std(lw_shrinkage, axis = 1)/np.sqrt(n_iterations), color = '#d62728', label = 'Shrinkage parameter', capsize = 3 )
axes[0].legend()
axes[0].set_xlabel('Number of samples')
axes[0].set_ylabel('Mean shrinkage parameter', fontsize = 8)
axes[0].get_yaxis().set_label_coords(-0.11,0.5)
axes[1].errorbar(n_samples, np.mean(lw_mse, axis = 1), yerr = np.std(lw_mse, axis = 1)/np.sqrt(n_iterations), color = '#d62728', label = 'Ledoit & Wolf estimator', capsize = 3)
axes[1].errorbar(n_samples, np.mean(scm_mse, axis = 1), yerr = np.std(scm_mse, axis = 1)/np.sqrt(n_iterations), color = '#1f77b4', label = 'Sample covariance matrix', capsize = 3 )
axes[1].legend()
axes[1].set_xlabel('Number of samples')
axes[1].set_ylabel('Mean squared error', fontsize = 8)
axes[1].get_yaxis().set_label_coords(-0.11,0.5)

fig.suptitle('Comparison of the Ledoit & Wolf estimator to the sample covariance matrix')

    

