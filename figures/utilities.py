# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:12:20 2021

Figure utilities
"""
import numpy as np

from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler

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

