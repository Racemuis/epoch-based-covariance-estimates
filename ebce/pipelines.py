# -*- coding: utf-8 -*-
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from transformers import ShrinkageTransform, LDAShrinkageTransform

def benchmark_pipeline(n_xdawn_components = 5):
    pipelines = dict()

    pipe = Pipeline([('Covariance Estimation', XdawnCovariances(nfilter=n_xdawn_components)),
                     ('Tangent Space Mapping', TangentSpace()),
                     ('LDA', LDA())])
    pipelines['Benchmark'] = pipe
    return pipelines

def manifold_shrinkage_pipeline(n_xdawn_components = 5):
    pipelines = dict()

    pipe = Pipeline([('Covariance Estimation', XdawnCovariances(nfilter=n_xdawn_components)),
                     ('Shrinkage', ShrinkageTransform(location = 'manifold')), 
                     ('Tangent Space Mapping', TangentSpace()),
                     ('LDA', LDA())])
    pipelines['Manifold shrinkage'] = pipe
    return pipelines

def tangent_space_shrinkage_pipeline(n_xdawn_components = 5):
    pipelines = dict()

    pipe = Pipeline([('Covariance Estimation', XdawnCovariances(nfilter=n_xdawn_components)),
                     ('Tangent Space Mapping', TangentSpace()),
                     ('Shrinkage + LDA', LDAShrinkageTransform(location = 'tangent space'))])
    pipelines['Manifold shrinkage'] = pipe
    return pipelines
    pass

def tangent_space_tdlda_pipeline():
    pass

def procrastinate():
    pass
