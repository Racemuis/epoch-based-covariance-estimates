# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utilities import shrinkage_regularization

class ShrinkageTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, technique = 'ledoit-wolf', location = 'manifold', z_score = True):
        """
        Init. 

        Parameters
        ----------
        technique : string, optional
            the shrinkage regularization technique. 
            Possible techniques:
                'ledoit-wolf'
                'grid'
            The default is 'ledoit-wolf'.
        location : string, optional
            the shrinkage location.
            Possible locations:
                'manifold'
                'tangent space'
            The default is 'manifold'.
        z_score : bool, optional
            True if z-scoring should be applied.
            The default is True.

        Returns
        -------
        None.

        """
        
        self.technique = technique
        self.location = location
        self.z_score = z_score
        
    def fit(self, X, y = None):
        """
        Fit. For compatibility purpose with sklearn.pipeline.Pipeline.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trial.
        y : ndarray of shape (n_trials, ), optional
            the lable of each trial. The default is None.

        Returns
        -------
        self : the ShrinkageTransform instance
            an instance of ShrinkageTransform.

        """
        return self
                        
    def transform(self, X):
        """
        Apply shrinkage regularization on the estimated covariance matrices.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_c, n_c)
            ndarray of covariance matrices for each trial.

        Returns
        -------
        shrink_mats : ndarray of shape (n_trials, n_c, n_c)
                    ndarray of covariance matrices for each trial.

        """
        [n_trials, n_c, n_c] = X.shape
        shrink_mats = np.zeros((n_trials, n_c, n_c))
        for i in range(n_trials):
            shrink_mats[i], _ = shrinkage_regularization(X[i], self.technique, self.location, self.z_score)
        return shrink_mats
    

class LDAShrinkageTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, technique = 'ledoit-wolf', location = 'manifold', z_score = True):
        """
        Init. 

        Parameters
        ----------
        technique : string, optional
            the shrinkage regularization technique. 
            Possible techniques:
                'ledoit-wolf'
                'grid'
            The default is 'ledoit-wolf'.
        location : string, optional
            the shrinkage location.
            Possible locations:
                'manifold'
                'tangent space'
            The default is 'manifold'.
        z_score : bool, optional
            True if z-scoring should be applied.
            The default is True.

        Returns
        -------
        None.

        """
        
        self.technique = technique
        self.location = location
        self.z_score = z_score
        
    def fit(self, X, y = None):
        """
        Fit. For compatibility purpose with sklearn.pipeline.Pipeline.

        Parameters
        ----------
        X : 
            
        y : ndarray of shape (n_trials, ), optional
            the lable of each trial. The default is None.

        Returns
        -------
        self : the LDAShrinkageTransform instance
            an instance of LDAShrinkageTransform.

        """
        return self
                        
    def transform(self, X):
       pass
    
    def score(self, X, y):
        pass



