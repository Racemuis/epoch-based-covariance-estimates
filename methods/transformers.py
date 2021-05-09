# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utilities import shrinkage_regularization, tangent_space, verify_pd
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.covariance import covariances 
from pyriemann.spatialfilters import Xdawn

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
        Apply shrinkage regularization on the estimated covariance matrices.

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
            shrink_mats[i], _ = shrinkage_regularization(X[i],
                                                         location = self.location, 
                                                         z_score = self.z_score, 
                                                         scope = self.scope)
        if(verify_pd(shrink_mats)):
            return shrink_mats
        else:
            return covariances(X, estimator = 'scm') 

class XdawnFilter(BaseEstimator, TransformerMixin):
    """Creation of the data matrix dedicated to ERP processing
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