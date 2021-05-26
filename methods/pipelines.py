# -*- coding: utf-8 -*-
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from transformers import ShrinkageTransform, XdawnFilter

def tangent_space_LDA(n_xdawn_components = 5):
    """
    The benchmark pipeline. No shrinkage is added, except for the shrinkage
    of the within and the total class covariance matrix of the LDA instance. 

    Parameters
    ----------
    n_xdawn_components : int, optional
        the number of spatial xDawn filters. The dimension of the resulting
        covariance matrix is equal to (n_xdawn_components*n_classes)*2, where
        the n_classes is either 1 (target or non-target) or 2 (both).
        The default is 5.

    Returns
    -------
    pipelines : dict
        the pipeline.

    """
    
    pipelines = dict()

    pipe = Pipeline([('Covariance Estimation', XdawnCovariances(nfilter=n_xdawn_components, classes=[1])),
                     ('Tangent Space Mapping', TangentSpace()),
                     ('Shrinkage + LDA', LDA(solver = 'lsqr', shrinkage='auto'))])
    pipelines['Tangent space LDA'] = pipe
    return pipelines

def response_shrinkage(n_xdawn_components = 5):
    """
    In the manifold shrinkage pipeline, shrinkage is applied to the bottom
    right the epoch based covariance matrices that are enhanced with the 
    prototypes. In the bottom right, the spatial covariance is stored.     

    Parameters
    ----------
    n_timepoints : int
        the number of timepoints in a single epoch
    n_xdawn_components : int, optional
        the number of spatial xDawn filters. The dimension of the resulting
        covariance matrix is equal to (n_xdawn_components*n_classes)*2, where
        the n_classes is either 1 (target or non-target) or 2 (both).
        The default is 5.

    Returns
    -------
    pipelines : dict
        the pipeline.

    """
    
    pipelines = dict()

    pipe = Pipeline([('Filtering', XdawnFilter(nfilter=n_xdawn_components, classes=[1])),
                     ('Shrinkage', ShrinkageTransform(location = 'manifold', scope = 'lower right')), 
                     ('Tangent Space Mapping', TangentSpace()),
                     ('LDA', LDA(solver = 'lsqr', shrinkage='auto'))])
    pipelines['Response shrinkage'] = pipe
    return pipelines

def super_trial_shrinkage(n_xdawn_components = 5):
    """
    In the prototype shrinkage pipeline, shrinkage is applied to the top left and the 
    bottom right the sub covariance matrices that are enhanced with the 
    prototypes. In the bottom right, the spatial covariance is stored.   
    In the top left, the prototype is located.

    Parameters
    ----------
    n_xdawn_components : int, optional
        the number of spatial xDawn filters. The dimension of the resulting
        covariance matrix is equal to (n_xdawn_components*n_classes)*2, where
        the n_classes is either 1 (target or non-target) or 2 (both).
        The default is 5.

    Returns
    -------
    pipelines : dict
        the pipeline.

    """
    
    pipelines = dict()

    pipe = Pipeline([('Filtering', XdawnFilter(nfilter=n_xdawn_components, classes=[1])),
                     ('Response & Prototype', ShrinkageTransform(location = 'manifold', scope = 'both')), 
                     ('Tangent Space Mapping', TangentSpace()),
                     ('LDA', LDA(solver = 'lsqr', shrinkage='auto'))])
    pipelines['Super-trial shrinkage'] = pipe
    return pipelines
