# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import LinAlgError

from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler

from moabb.paradigms import P300
from moabb.datasets import EPFLP300, bi2013a
from moabb.datasets import BNCI2014009 as bnci_1
from moabb.datasets import BNCI2014008 as bnci_als
from moabb.datasets import BNCI2015003 as bnci_2

from datasets.spot_pilot import SpotPilotData

def get_benchmark_config(dataset_name, cfg_prepro, subjects=None, sessions=None, data_path = None):
    """
    Retreived from: https://github.com/jsosulski/time-decoupled-lda
    Licensed under the MIT License.

    Parameters
    ----------
    dataset_name : string
        The name of the dataset.
    cfg_prepro : dict
        The preprocessing hyperparameters.
    subjects : list, optional
        The list of subjects that need to be included in the analyses.
        The default is None.
    sessions : list, optional
        The list of sessions that need to be included in the analyses.
        The default is None.
    data_path : string, optional
        The data path to the SPOT dataset. The default is None.

    Raises
    ------
    ValueError
        Raises a value-error if the given dataset name has not been specified.

    Returns
    -------
    benchmark_cfg : dict
        A dictionary containing the benchmark configurations.

    """
    
    benchmark_cfg = dict()
    paradigm = P300(resample=cfg_prepro['sampling_rate'], fmin=cfg_prepro['fmin'], fmax=cfg_prepro['fmax'],
                    reject_uv=cfg_prepro['reject_uv'], baseline=cfg_prepro['baseline'])
    load_ival = [0, 1]
    if dataset_name == 'spot_single':
        d = SpotPilotData(load_single_trials=True)
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        if data_path is not None:
            d.path = data_path
        n_channels = d.N_channels
    elif dataset_name == 'epfl':
        d = EPFLP300()
        d.interval = load_ival
        d.unit_factor = 1
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 32
    elif dataset_name == 'bnci_1':
        d = bnci_1()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    elif dataset_name == 'bnci_als':
        d = bnci_als()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'bnci_2':
        d = bnci_2()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'braininvaders':
        d = bi2013a()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized.')

    benchmark_cfg['dataset'] = d
    benchmark_cfg['N_channels'] = n_channels
    benchmark_cfg['paradigm'] = paradigm
    return benchmark_cfg


def verify_pd(X):
    """
    Verify whether the matrix is Definite & Positive.
    Based on the cholesky decomposition, which fails if a matrix is not
    Definite Positive. Covariance matrices are symmetric by nature, making the
    matrix SPD.

    Parameters
    ----------
    X : ndarray of shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trial.

    Returns
    -------
      : boolean
      true if the matrix is pd. 

    """
    counter = 0
    for trials in X:
        try:
            np.linalg.cholesky(trials)
        except LinAlgError:
            counter += 1
    if counter > 0:
        print(f'{X.shape[0]} matrices processed. There are {counter} non-SPD matrices.')
        return False
    else:
        return True
    
def verify_pd_single(X):
    """
    Verify whether the matrix is Definite & Positive.
    Based on the cholesky decomposition, which fails if a matrix is not
    Definite Positive. Covariance matrices are symmetric by nature, making the
    matrix SPD.

    Parameters
    ----------
    X : ndarray of shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trial.

    Returns
    -------
      : boolean
      true if the matrix is pd. 

    """
    try:
        np.linalg.cholesky(X)
    except LinAlgError:
        return False
    return True

def shrinkage_regularization(x, location, z_score, scope = None):
        """
        Apply shrinkage regularization.
    
        Parameters
        ----------
        x : ndarray of shape (n_features, n_samples)
            ndarray of spatially filtered prototype responses.
        location : string
            the shrinkage location.
            Possible locations:
                'manifold'
                'tangent space'
        z_score : bool, optional
            True if z-scoring should be applied. 
        scope : string, optional
                the part of the matrix to which shrinkage regularization
                needs to be applied. 
                Possible scopes:
                    'upper left'
                    'lower right'
                    'both'
                The default is None.
                    
        Returns
        -------
        sigma : ndarray of shape (n_c, n_c)
                ndarray of the enhanced estimator for one trial.
        shrinkage : float
            The shrinkage parameter.
            0 <= shrinkage <= 1.
    
        """
        # de-mean returns
        [n,t]=x.shape
    
        meanx=np.mean(x)
        x_s=x-meanx
        
        # Compute sample covariance matrix
        sample = 1/t * np.dot(x_s, x_s.T)
        
        if location == 'manifold' and scope == 'upper left':
            # Extract subdata
            sub_data = x[:n//2, :]
            
            # Transpose the data as ledoit_wolf takes n_samples x n_features
            sub_data = sub_data.T
            
            if z_score:
                # z-score data
                sc = StandardScaler()  
                X = sc.fit_transform(sub_data)
            
            # Apply shrinkage
            sigma, shrinkage = ledoit_wolf(X)
            
            if z_score:
                # Rescale
                sigma = sc.scale_[:, np.newaxis] * sigma * sc.scale_[np.newaxis, :]
            
            # Transpose is not needed because sigma is symmetric            
            
            # Replace submatrix
            sample[:n//2, :n//2] = sigma
            
        elif location == 'manifold' and scope == 'lower right':
            # Extract subdata
            sub_data = x[n//2:n, :]
            
            # Transpose the data as ledoit_wolf takes n_samples x n_features
            sub_data = sub_data.T
            
            if z_score:
                # z-score data
                sc = StandardScaler()  
                X = sc.fit_transform(sub_data)
            
            # Apply shrinkage
            sigma, shrinkage = ledoit_wolf(X)
            
            if z_score:
                # Rescale
                sigma = sc.scale_[:, np.newaxis] * sigma * sc.scale_[np.newaxis, :]
                        
            # Replace submatrix
            sample[n//2:n, n//2:n] = sigma
        
        elif location == 'manifold' and scope == 'both':
            # Extract subdata; bottom right
            sub_data = x[n//2:n, :]
            
            # Transpose the data as ledoit_wolf takes n_samples x n_features
            sub_data = sub_data.T
            
            if z_score:
                # z-score data
                sc = StandardScaler()  
                X = sc.fit_transform(sub_data)
            
            # Apply shrinkage
            sigma, shrinkage = ledoit_wolf(X)
            
            if z_score:
                # Rescale
                sigma = sc.scale_[:, np.newaxis] * sigma * sc.scale_[np.newaxis, :]
            
            # Replace submatrix
            sample[n//2:n, n//2:n] = sigma
            
            # Extract subdata; top left
            sub_data = x[:n//2, :]
            
            # Transpose the data as ledoit_wolf takes n_samples x n_features
            sub_data = sub_data.T
            
            if z_score:
                # z-score data
                sc = StandardScaler()  
                X = sc.fit_transform(sub_data)
            
            # Apply shrinkage
            sigma, shrinkage = ledoit_wolf(X)
            
            if z_score:
                # Rescale
                sigma = sc.scale_[:, np.newaxis] * sigma * sc.scale_[np.newaxis, :]
            
            # Replace submatrix
            sample[:n//2, :n//2] = sigma
            
            
        elif location == 'tangent space':
            sub_data = x
            
            # Transpose the data as ledoit_wolf takes n_samples x n_features
            sub_data = sub_data.T
            
            # z-score data
            sc = StandardScaler()  
            X = sc.fit_transform(sub_data)
            
            # Apply shrinkage
            sigma, shrinkage = ledoit_wolf(X)
            
            # Rescale
            sample = sc.scale_[:, np.newaxis] * sigma * sc.scale_[np.newaxis, :]
            
        else:
            raise(ValueError("Invalid location for shrinkage. Valid locations are 'manifold' with 'upper left', 'lower right' or 'both'', or 'tangent space'."))
        
        return sample, shrinkage

