# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import LinAlgError

from moabb.paradigms import P300
from moabb.datasets import EPFLP300, bi2013a
from moabb.datasets import BNCI2014009 as bnci_1
from moabb.datasets import BNCI2014008 as bnci_als
from moabb.datasets import BNCI2015003 as bnci_2

from datasets.spot_pilot import SpotPilotData

def get_benchmark_config(dataset_name, cfg_prepro, subjects=None, sessions=None, data_path = None):
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
    None.

    """
    counter = 0
    for trials in X:
        try:
            np.linalg.cholesky(X)
        except LinAlgError:
            counter += 1
    print(f'{X.shape[0]} matrices processed. There are {counter} non-SPD matrices.')
    
def shrinkage_regularization(x, technique, location, z_score):
        """
        Apply shrinkage regularization.
    
        Parameters
        ----------
        x : ndarray of shape (n_c, n_c)
            ndarray of covariance matrices for one trial.
        technique : string, optional
                the shrinkage regularization technique. 
                Possible techniques:
                    'ledoit-wolf'
                    'grid'
        location : string, optional
            the shrinkage location.
            Possible locations:
                'manifold'
                'tangent space'
        z_score : bool, optional
            True if z-scoring should be applied. 
                    
        Returns
        -------
        sigma : ndarray of shape (n_c, n_c)
                ndarray of the enhanced estimator for one trial.
        shrinkage : float
            The shrinkage parameter.
            0 <= shrinkage <= 1.
    
        """
        # de-mean returns
        [t,n]=x.shape
    
        meanx=np.mean(x);
        x=x-meanx
        
        # compute sample covariance matrix
        sample = 1/t * np.dot(x.T, x)
        
        # compute prior
        if location == 'manifold':
            eye = np.zeros((n,n))
            meanvar = 0
            for i in range(n//2):
                eye[i,i] = 1
                #meanvar += sample[i,i]
            #meanvar = meanvar/(n//2)
        elif location == 'tangent space':
            eye = np.identity(n)
            meanvar=np.mean(np.diagonal(sample))
        else:
            raise(ValueError("Invalid location for shrinkage. Valid locations are 'manifold' or 'tangent space'."))
        meanvar=np.mean(np.diagonal(sample))
        prior=meanvar*eye    	           
          
        # what we call p 
        y=np.power(x, 2)			                    
        phiMat=np.dot(y.T, y)/t-np.power(sample, 2)    
        phi=np.sum(np.sum(phiMat))
            
        # what we call r is not needed for this shrinkage target
        
        # what we call c
        gamma=np.linalg.norm(sample-prior,'fro')**2    
      
        # compute shrinkage constant
        kappa=phi/gamma
        shrinkage=max(0,min(1,kappa/t))
        
        # z-score covariance matrix
        if z_score:
            mean = np.mean(sample)
            std = np.std(sample)
            sample = (sample - mean)/std
            prior = (prior - eye*mean)/std
            
            # compute shrinkage estimator
            sigma=shrinkage*prior+(1-shrinkage)*sample
            sigma = sigma * std + mean
            
        else: 
            sigma=shrinkage*prior+(1-shrinkage)*sample

        return sigma, shrinkage
    
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function shrinkage_regularization is released under the BSD 2-clause license.

% Copyright (c) 2014, Olivier Ledoit and Michael Wolf 
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

