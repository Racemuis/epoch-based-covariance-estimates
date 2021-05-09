# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:12:20 2021

Figure utilities
"""
import numpy as np

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

def generate_covariance(n_channels):
    """
    Generate a symmetrical square matrix. (Based on SP assignment 2).

    Parameters
    ----------
    n_channels : int
        The dimension of the matrix.

    Returns
    -------
    covariance : ndarray of shape (n_channels, n_channels)
        The generated symmetrical square matrix.
    """
    temp = np.random.randn(n_channels, n_channels)
    covariance = np.dot(temp, temp.transpose()) 
    return covariance

# 3D utilities
def meshgrid_sphere(x_0 = 0, y_0 = 0, z_0 = 0, radius = 1):
    """
    Calculate the meshgrid of a sphere given the centre (x,y,z) and the radius.

    Parameters
    ----------
    x_0 : int, optional
        The x coordinate of the centre of the sphere. The default is 0.
    y_0 : int, optional
        The y coordinate of the centre of the sphere. The default is 0.
    z_0 : int, optional
        The z coordinate of the centre of the sphere. The default is 0.
    radius : int, optional
        The radius of the sphere. The default is 1.

    Returns
    -------
    x : ndarray of shape (20, 20)
        The meshgrid of the x coordinates.
    y : ndarray of shape (20, 20)
        The meshgrid of the y coordinates.
    z : ndarray of shape (20, 20)
        The meshgrid of the z coordinates.

    """
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 20)
    
    theta, phi = np.meshgrid(theta, phi)
        
    x = x_0 + radius * np.sin(theta)*np.cos(phi)
    y = y_0 + radius * np.sin(theta)*np.sin(phi)
    z = z_0 + radius * np.cos(theta)
    return x, y, z

def plot_tangent_line(ax, n_lines):
    """
    Plot n_lines tangent lines on the top of a unit sphere.

    Parameters
    ----------
    ax : axes.Axes 
        An array of axes.
    n_lines : int
        The number of tangent lines that are plotted on 1/2 circle.

    Returns
    -------
    None.

    """
    steps = int(np.ceil(n_lines/2))
    x = np.linspace(-1, 1, steps)
    y = np.sqrt(1-x**2)
    z = np.zeros(steps)
    if steps != 1:
        ax.quiver(z,z,z+1,x,y,z, color = 'black', linewidth = 3, arrow_length_ratio = 0.15)
    ax.quiver(z,z,z+1,x,-y,z, color = 'black', linewidth = 3, arrow_length_ratio = 0.15, label = 'Tangent vector')
    
def meshgrid_plane(x_0 = 0, y_0 = 0, length = 1):
    """
    Calculate the meshgrid of a square plane given the centriod (x,y).

    Parameters
    ----------
    x_0 : int, optional
        The x coordinate of the centroid of the plane. The default is 0.
    y_0 : int, optional
        The y coordinate of the centroid of the plane. The default is 0.
    length : int, optional
        The length of the sides of the plane. The default is 1.

    Returns
    -------
    x : ndarray of shape (20, 20)
        The meshgrid of the x coordinates.
    y : ndarray of shape (20, 20)
        The meshgrid of the y coordinates.
    z : ndarray of shape (20, 20)
        The meshgrid of the z coordinates.

    """
    # Create normal vector
    a = 0
    b = 0
    c = 1
    z_0 = 0
    
    d = (x_0 + y_0 + z_0)
    x = np.linspace(-length/2,length/2,10)

    x,y = np.meshgrid(x,x)
    
    # Calculate plane
    z = (d - a*(x-x_0) - b*(y-y_0)) / c + z_0
    return x, y, z

def ledoit_wolf(t, sub_matrix, sub_data, z_score):
    """
    Apply shrinkage regularization.

    Parameters
    ----------
    t : int
        the number of samples of sub_data.
    sub_matrix : ndarray
        the sub matrix in the covariance matrix that needs to be shrunk.
    sub_data : ndarray
        the data that corresponds to the sub matrix.
    z_score : bool
        True if z-scoring should be applied.

    Returns
    -------
    sigma : ndarray of shape (n_c, n_c)
        ndarray of the enhanced estimator for one trial.
    shrinkage : float
        The shrinkage parameter.
        0 <= shrinkage <= 1.

    """
    # Compute prior
    [sn,sn] = sub_matrix.shape
    eye = np.zeros((sn,sn))
    
    meanvar=np.mean(np.diagonal(sub_matrix))
    prior=meanvar*eye    	           
      
    y=np.power(sub_data, 2)			                    
    phiMat=np.dot(y, y.T)/t-np.power(sub_matrix, 2)    
    phi=np.sum(np.sum(phiMat))
                
    gamma=np.linalg.norm(sub_matrix-prior,'fro')**2    
  
    # Compute shrinkage constant
    kappa=phi/gamma
    shrinkage=max(0,min(1,kappa/t))
    
    # z-score covariance matrix
    if z_score:
        mean = np.mean(sub_matrix)
        std = np.std(sub_matrix)
        sub_matrix = (sub_matrix - mean)/std
        prior = (prior - eye*mean)/std
        
        # Compute shrinkage estimator
        sigma=shrinkage*prior+(1-shrinkage)*sub_matrix
        sigma = sigma * std + mean
        
    else: 
        sigma=shrinkage*prior+(1-shrinkage)*sub_matrix
    
    return sigma, shrinkage
        

def shrinkage_regularization(x, location, z_score, scope = None):
        """
        Apply shrinkage regularization.
    
        Parameters
        ----------
        x : ndarray of shape (classes*2*n_channels, n_samples)
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
        print("n: {} \nt: {}".format(n, t))
    
        meanx=np.mean(x)
        x=x-meanx
        
        # Compute sample covariance matrix
        sample = 1/t * np.dot(x, x.T)
        
        if location == 'manifold' and scope == 'upper left':
            # Extract submatrix
            sub_matrix = sample[:n//2, :n//2]
            sub_data = x[:n//2, :]
            sigma, shrinkage = ledoit_wolf(t, sub_matrix, sub_data, z_score)
            
            # Put back submatrix
            sample[:n//2, :n//2] = sigma
            
        elif location == 'manifold' and scope == 'lower right':
            # Extract submatrix
            sub_matrix = sample[n//2:n, n//2:n]
            sub_data = x[n//2:n, :]
            sigma, shrinkage = ledoit_wolf(t, sub_matrix, sub_data, z_score)
            
            # Put back submatrix
            sample[n//2:n, n//2:n] = sigma
        
        elif location == 'manifold' and scope == 'both':
            # Extract submatrix
            sub_matrix = sample[n//2:n, n//2:n]
            sub_data = x[n//2:n, :]
            sigma, shrinkage = ledoit_wolf(t, sub_matrix, sub_data, z_score)
            
            # Put back submatrix
            sample[n//2:n, n//2:n] = sigma
            
            # Extract submatrix
            sub_matrix = sample[:n//2, :n//2]
            sub_data = x[:n//2, :]
            sigma, shrinkage = ledoit_wolf(t, sub_matrix, sub_data, z_score)
            
            # Put back submatrix
            sample[:n//2, :n//2] = sigma
            
            
        elif location == 'tangent space':
            sub_matrix = sample
            sub_data = x
            sample, shrinkage = ledoit_wolf(t, sub_matrix, sub_data, z_score)
            
        else:
            raise(ValueError("Invalid location for shrinkage. Valid locations are 'manifold' with 'upper left', 'lower right' or 'both'', or 'tangent space'."))
        
        return sample, shrinkage
    
    
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

