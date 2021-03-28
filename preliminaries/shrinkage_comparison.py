# -*- coding: utf-8 -*-
"""
Generate plots that plot the distances between the following estimators:
    The Sample Covariance Matrix (SCM)
    The improved estimator given Ledoit-Wolf shrinkage
    The improved estimator given grid search shrinkage, minimizing the MSE

Created on Sat Mar 27 19:14:01 2021
"""

from matplotlib import pyplot as plt
from sklearn.datasets import make_spd_matrix
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

# Estimators
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

def ledoit_wolf_parameter(x):
    """
    Calculate the shrinkage parameter according 
    to the Ledoit-Wolf shrinkage technique. 
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Input array.
        
    Returns
    -------
    float
        The shrinkage parameter; the range of the return value is [0,1].
    """
    lw = LedoitWolf().fit(x)
    return lw.shrinkage_

def grid_search_parameter(x, target):
    """
    Wrapper function for grid search.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Input array.
    target : ndarray of shape (n_channels, n_channels) 
        The true covariance matrix (population).

    Returns
    -------
    float
        The parameter that minimizes the metric. 

    """
    return grid_search(MSE, 0.01, estimate_covariance(x), target)

def grid_search(metric, stepsize, estimator, target):
    """
    Find the parameter that minimizes the metric given the estimator
    and the target. 
    
    Parameters
    ----------
    metric : function
        The metric that is used for the grid search.
    stepsize : float
        The stepsize in the grid.
    estimator : ndarray of shape (n_channels, n_channels) 
        The estimator of the covariance matrix (sample).
    target : ndarray of shape (n_channels, n_channels) 
        The true covariance matrix (population).

    Returns
    -------
    float
        The parameter that minimizes the metric. 
    """
    grid = np.linspace(0,1,int(1/stepsize + 1))
    grid_parameter = np.argmin([metric(shrink(estimator, param), target) for param in grid])
    return grid_parameter*stepsize

def shrink(scm, rho):
    """
    Calculate the improved estimator of the population covariance matrix
    according to the Ledoit-Wolf shrinkage technique.
    
    Parameters
    ----------
    scm : ndarray of shape (n_channels, n_channels)
        The estimated sample covariance matrix.
    rho : int
        The shrinkage parameter.

    Returns
    -------
    ndarray of shape (n_channels, n_channels)
        The improved estimator.
    """
    d = scm.shape[0]
    v = np.trace(scm)/d
    return (1-rho)*scm + rho*v*np.identity(d)

# Data generation
def generate_epochs(n_epochs, n_samples, n_channels, target):
    """
    Samples |n_epochs| epochs according to a multivariate normal 
    distribution with a mean of 0. 
    
    Parameters
    ----------
    n_epochs : int
        The number of epochs that are sampled.
    n_samples : int
        The number of samples (timepoints) per epoch.
    n_channels : int
        The number of channels per epoch.
    target : ndarray of shape (n_channels, n_channels)
        The generated population covariance matrix.

    Returns
    -------
    iteration : [ndarray of shape (n_samples, n_channels)]
        A list with the simulated data.
    """
    mean = np.zeros(n_channels)
    iteration = [np.random.multivariate_normal(mean, target, n_samples) for i in range(n_epochs)]
    return iteration

# Observation generation
def get_summary_statistics(technique, iteration, **kwargs):
    """
    Calculate the mean and the Standard Error of the Mean (SEM) of the 
    given technique over a single iteration. 

    Parameters
    ----------
    technique : function
        The technique for which the summary statistics can be calculated.
    iteration : [ndarray of shape (n_samples, n_channels)]
        A list with the simulated data.     
    **kwargs : unknown
        The arguments of the technique, apart from the data. 
        
    Returns
    -------
    mean : float
        The mean value over the iteration given the technique.
    sem : float
        The SEM over the iteration given the technique. 
    """
    parameter_values = [technique(epoch, **kwargs) for epoch in iteration]
    return np.mean(parameter_values), np.std(parameter_values)/np.sqrt(len(iteration))

def get_summary_statistics_metric(metric, iterations):
    """
    Calculate the mean and the Standard Error of the Mean (SEM) given the 
    metric of the Ledoit-Wolf shrinkage technique, the original estimator 
    and the grid search estimator over a single iteration. 

    Parameters
    ----------
    metric : function
        The metric for which the summary statistics can be calculated.
    iteration : [ndarray of shape (n_samples, n_channels)]
        A list with the simulated data. 
        
    Returns
    -------
    mean_lw : float
        The mean over the iteration given the Ledoit_Wolf shrinkage technique
        and the metric.
    mean_grid : float
        The mean over the iteration given the grid search shrinkage
        and the metric.
    mean_scm : float
        The mean over the iteration given the sample covariance matrix
        and the metric.
    sem_lw : float
        The SEM over the iteration given the Ledoit_Wolf shrinkage technique
        and the metric.
    sem_grid : float
        The SEM over the iteration given the grid search shrinkage
        and the metric.
    sem_scm : float
       The SEM over the iteration given the sample covariance matrix
        and the metric.

    """
    mean_lw = np.zeros(len(iterations))
    mean_grid = np.zeros(len(iterations))
    mean_scm = np.zeros(len(iterations))
    sem_lw = np.zeros(len(iterations))
    sem_grid = np.zeros(len(iterations))
    sem_scm = np.zeros(len(iterations))
    
    for i in range(len(iterations)):
        scores = [metric(LedoitWolf().fit(epoch).covariance_, targets[i]) for epoch in iterations[i]]
        mean_lw[i] = np.mean(scores)
        sem_lw[i] = np.std(scores)/np.sqrt(len(iterations[i]))
        
        scores = [metric(shrink(estimate_covariance(epoch), grid_search_parameter(epoch, targets[i])), targets[i]) for epoch in iterations[i]]
        mean_grid[i] = np.mean(scores)
        sem_grid[i] = np.std(scores)/np.sqrt(len(iterations[i]))
        
        scores = [metric(estimate_covariance(epoch), targets[i]) for epoch in iterations[i]]
        mean_scm[i] = np.mean(scores)
        sem_scm[i] = np.std(scores)/np.sqrt(len(iterations[i]))
        
    return mean_lw, mean_grid, mean_scm, sem_lw, sem_grid, sem_scm

# Set hyperparameters
np.random.seed(42)
n_iterations = 50
n_epochs = 60
n_channels = 31
n_samples = 71

# Generate the data
targets = [make_spd_matrix(n_channels) for i in range(n_iterations)]
iterations = [generate_epochs(n_epochs, n_samples, n_channels, t) for t in targets]

# Calculate the summary statistics for the shrinkage parameter values
summary_statistics_lw = [get_summary_statistics(ledoit_wolf_parameter, i) for i in iterations]
summary_statistics_grid = [get_summary_statistics(grid_search_parameter, iterations[i], target = targets[i]) for i in range(len(iterations))]

# Unwrap the lists
mean_lw, sem_lw = zip(*summary_statistics_lw)
mean_grid, sem_grid = zip(*summary_statistics_grid)

# Plot the observations
plt.figure(figsize=(20, 4))
width = 0.4

plt.bar(np.arange(len(iterations)) - width/2, mean_lw, yerr = sem_lw, label = 'Ledoit-Wolf',  capsize = 3, width = width)
plt.bar(np.arange(len(iterations)) + width/2, mean_grid, yerr = sem_grid, label = 'shrinkage parameter with minimum MSE', capsize = 3, width = width)
plt.legend()
plt.ylabel('Mean shrinkage parameter value')
plt.xlabel('Target covariance index')
plt.title('The mean shrinkage parameter per simulated target covariance matrix')
plt.show()

# Calculate the summary statstics for the MSE
mean_lw, mean_grid, mean_scm , sem_lw, sem_grid, sem_scm = get_summary_statistics_metric(MSE, iterations)
    
# Plot the observations
plt.figure(figsize=(20, 4))
width = 0.3

plt.bar(np.arange(len(iterations)), mean_lw, yerr = sem_lw, label = 'Ledoit-Wolf',  capsize = 2, width = width, align='center')
plt.bar(np.arange(len(iterations))- width, mean_scm, yerr = sem_scm, label = 'Sample covariance matrix', capsize = 2, width = width, align='center', color='green')
plt.bar(np.arange(len(iterations))+ width, mean_grid, yerr = sem_grid, label = 'shrinkage parameter with minimum MSE', capsize = 2, width = width,align='center')
plt.legend()
plt.ylabel('Mean Squared Error')
plt.xlabel('Target covariance index')
plt.title('The mean MSE per simulated target covariance matrix')
plt.show()

# Calculate the summary statstics for the Riemannian distance
mean_lw, mean_grid, mean_scm , sem_lw, sem_grid, sem_scm = get_summary_statistics_metric(distance_riemann, iterations)
    
# Plot the observations
plt.figure(figsize=(20, 4))
width = 0.3

plt.bar(np.arange(len(iterations)), mean_lw, yerr = sem_lw, label = 'Ledoit-Wolf',  capsize = 2, width = width, align='center')
plt.bar(np.arange(len(iterations))- width, mean_scm, yerr = sem_scm, label = 'Sample covariance matrix', capsize = 2, width = width, align='center', color='green')
plt.bar(np.arange(len(iterations))+ width, mean_grid, yerr = sem_grid, label = 'shrinkage parameter with minimum MSE', capsize = 2, width = width,align='center')
plt.legend()
plt.ylabel('Riemannian distance')
plt.xlabel('Target covariance index')
plt.title('The mean Riemannian distance per simulated target covariance matrix')
plt.show()