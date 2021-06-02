# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utilities import shrinkage_regularization, verify_pd_single
from pyriemann.utils.covariance import _scm
from pyriemann.spatialfilters import Xdawn

class ShrinkageTransform(BaseEstimator, TransformerMixin):
    """
    Estimate the covariance matrix and apply shrinkage to a specific location
    in the aforementioned matrix.
    """
    def __init__(self, location = 'manifold', z_score = True, scope = 'lower right'):
        """
        Init. 

        Parameters
        ----------
        location : string, optional
            the shrinkage location.
            Possible locations:
                'manifold'
                'tangent space'
            The default is 'manifold'.
        n_timepoints : int, optional
            the number of timepoints in a single epoch.
        z_score : bool, optional
            True if z-scoring should be applied.
            The default is True.
        scope : string, optional
                the part of the matrix to which shrinkage regularization
                needs to be applied. 
                Possible parts:
                    'upper left'
                    'lower right'
                The default is None.

        Returns
        -------
        None.

        """
        self.location = location
        self.z_score = z_score
        self.scope = scope
        
    def fit(self, X, y = None):
        """
        Fit. For compatibility purpose with sklearn.pipeline.Pipeline.

        Parameters
        ----------
        X : ndarray of shape (n_trials, classes*2*n_channels, n_samples)
            ndarray of spatially filtered prototype responses.
        y : ndarray of shape (n_trials, ), optional
            the label of each trial. The default is None.

        Returns
        -------
        self : the ShrinkageTransform instance
            an instance of ShrinkageTransform.

        """
        return self
                        
    def transform(self, X):
        """
        Estimate the covariance matrices from the data and apply shrinkage
        regularization.

        Parameters
        ----------
        X : ndarray of shape (n_trials, classes*2*n_channels, n_samples)
            ndarray of spatially filtered prototype responses.

        Returns
        -------
        shrink_mats : ndarray of shape (n_trials, n_c, n_c)
                    ndarray of covariance matrices for each trial.

        """
        [n_trials, n_c, n_p] = X.shape
        shrink_mats = np.zeros((n_trials, n_c, n_c))
        for i in range(n_trials):
            candidate, _ = shrinkage_regularization(X[i],
                                                         location = self.location, 
                                                         z_score = self.z_score, 
                                                         scope = self.scope)
            if(verify_pd_single(candidate)):
                shrink_mats[i] = candidate
            else:
                shrink_mats[i] = _scm(X[i])
        return shrink_mats


class XdawnFilter(BaseEstimator, TransformerMixin):
    """This transformer is an adaption of the XdawnCovariances from pyRiemann. 
    For the latter function, the following copyright notice is included: 
    
        Copyright © 2015, authors of pyRiemann
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:
            * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
            * Redistributions in binary form must reproduce the above copyright
            notice, this list of conditions and the following disclaimer in the
            documentation and/or other materials provided with the distribution.
            * Neither the names of pyriemann authors nor the names of any
            contributors may be used to endorse or promote products derived from
            this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    Creation of the data matrix dedicated to ERP processing
    combined with Xdawn spatial filtering. A complete description of the
    method is available in [1].
    The advantage of this estimation is to reduce dimensionality of the
    covariance matrices efficiently.
    Parameters
    ----------
    nfilter: int (default 4)
        number of Xdawn filter per classes.
    applyfilters: bool (default True)
        if set to true, spatial filter are applied to the prototypes and the
        signals. When set to False, filters are applied only to the ERP prototypes
        allowing for a better generalization across subject and session at the
        expense of dimensionality increase. In that case, the estimation is
        similar to ERPCovariances with `svd=nfilter` but with more compact
        prototype reduction.
    classes : list of int | None (default None)
        list of classes to take into account for prototype estimation.
        If None (default), all classes will be accounted.
    xdawn_estimator : string (default: 'scm')
        covariance matrix estimator for xdawn spatial filtering.
    baseline_cov : baseline_cov : array, shape(n_chan, n_chan) | None (default)
        baseline_covariance for xdawn. see `Xdawn`.
    See Also
    --------
    Xdawn
    References
    ----------
    [1] Barachant, A. "MEG decoding using Riemannian Geometry and Unsupervised
        classification."
    """

    def __init__(self,
                 nfilter=4,
                 classes=None,
                 xdawn_estimator='scm',
                 baseline_cov=None):
        """Init."""
        self.xdawn_estimator = xdawn_estimator
        self.classes = classes
        self.nfilter = nfilter
        self.baseline_cov = baseline_cov

    def fit(self, X, y):
        """Fit.
        Estimate spatial filters and prototyped response for each classes.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.
        Returns
        -------
        self : XdawnFilter instance
            The XdawnFilter instance.
        """
        self.Xd_ = Xdawn(
            nfilter=self.nfilter,
            classes=self.classes,
            estimator=self.xdawn_estimator,
            baseline_cov=self.baseline_cov)
        self.Xd_.fit(X, y)
        self.P_ = self.Xd_.evokeds_
        return self

    def transform(self, X):
        """Estimate xdawn covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        Returns
        -------
        prototypes : ndarray, shape (n_trials, classes*2*n_channels, n_samples)
            ndarray of spatially filtered prototype responses.
        """
        X = self.Xd_.transform(X)
        Nt, Ne, Ns = X.shape
        Np, Ns = self.P_.shape
        prototypes = np.zeros((Nt, Ne + Np, Ns))
        for i in range(Nt):
            prototypes[i, :, :] = np.concatenate((self.P_, X[i, :, :]), axis=0)
        
        return prototypes