# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utilities import shrinkage_regularization, tangent_space, verify_pd
from pyriemann.utils.mean import mean_covariance

class ShrinkageTransform(BaseEstimator, TransformerMixin):
    
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
            shrink_mats[i], _ = shrinkage_regularization(X[i], self.location, self.z_score, self.scope)
        if(verify_pd(shrink_mats)):
            return shrink_mats
        else:
            return X

class TangentSpace(BaseEstimator, TransformerMixin):

    """Tangent space project TransformerMixin.
    Tangent space projection map a set of covariance matrices to their
    tangent space according to [1]. The Tangent space projection can be
    seen as a kernel operation, cf [2]. After projection, each matrix is
    represented as a vector of size :math:`N(N+1)/2` where N is the
    dimension of the covariance matrices.
    Tangent space projection is useful to convert covariance matrices in
    euclidean vectors while conserving the inner structure of the manifold.
    After projection, standard processing and vector-based classification can
    be applied.
    Tangent space projection is a local approximation of the manifold. it takes
    one parameter, the reference point, that is usually estimated using the
    geometric mean of the covariance matrices set you project. if the function
    `fit` is not called, the identity matrix will be used as reference point.
    This can lead to serious degradation of performances.
    The approximation will be bigger if the matrices in the set are scattered
    in the manifold, and lower if they are grouped in a small region of the
    manifold.
    After projection, it is possible to go back in the manifold using the
    inverse transform.
    Parameters
    ----------
    metric : string (default: 'riemann')
        The type of metric used for reference point mean estimation.
        see `mean_covariance` for the list of supported metric.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.
    See Also
    --------
    FgMDM
    FGDA
    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry,"" in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012
    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.
        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        # compute mean covariance
        self.reference_ = mean_covariance(X, metric=self.metric,
                                          sample_weight=sample_weight)
        return self

    def _check_data_dim(self, X):
        """Check data shape and return the size of cov mat."""
        shape_X = X.shape
        if len(X.shape) == 2:
            Ne = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if Ne != int(Ne):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(Ne)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        self._check_reference_points(X)
        if self.tsupdate:
            Cr = mean_covariance(X, metric=self.metric)
        else:
            Cr = self.reference_
        return tangent_space(X, Cr)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.
        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        # compute mean covariance
        self._check_reference_points(X)
        self.reference_ = mean_covariance(X, metric=self.metric,
                                          sample_weight=sample_weight)
        return tangent_space(X, self.reference_)