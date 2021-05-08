# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:14:01 2021
"""

from matplotlib import pyplot as plt
import numpy as np
from pyriemann.utils.distance import distance_riemann
from utilities import estimate_covariance, generate_covariance, MSE, shrinkage_regularization


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
    _, shrinkage = shrinkage_regularization(x, location = 'tangent space', z_score = True)
    return shrinkage

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
    and the grid search estimator over a single iteration. Covariance
    matrix estimation includes z-scoring.

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
        scores_lw = np.zeros(len(iterations[i]))
        scores_grid = np.zeros(len(iterations[i]))
        scores_scm = np.zeros(len(iterations[i]))
        for epoch in range(len(iterations[i])):
            data = iterations[i][epoch]
            scm = estimate_covariance(data)
            
            lw_matrix, rho = shrinkage_regularization(data, location = 'tangent space', z_score = True)
            #lw_matrix  = shrink((scm-np.mean(scm))/np.std(scm),
                                #rho)*np.std(scm)+np.mean(scm)
            scores_lw[epoch] = metric(lw_matrix, targets[i])
            
            
            grid_matrix = shrink((scm-np.mean(scm))/np.std(scm),
                                 grid_search_parameter(data, targets[i]))*np.std(scm)+np.mean(scm)
            scores_grid[epoch] = metric(grid_matrix, targets[i])
            
            scores_scm[epoch] = metric(scm, targets[i])
            
        mean_lw[i] = np.mean(scores_lw)
        sem_lw[i] = np.std(scores_lw)/np.sqrt(len(iterations[i]))
        
        mean_grid[i] = np.mean(scores_grid)
        sem_grid[i] = np.std(scores_grid)/np.sqrt(len(iterations[i]))
        
        mean_scm[i] = np.mean(scores_scm)
        sem_scm[i] = np.std(scores_scm)/np.sqrt(len(iterations[i]))
        
    return mean_lw, mean_grid, mean_scm, sem_lw, sem_grid, sem_scm

# Set hyperparameters
np.random.seed(42)
n_iterations = 50
n_epochs = 60
n_channels = 31
n_samples = 71

# Generate the data
targets = [generate_covariance(n_channels) for i in range(n_iterations)]
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
