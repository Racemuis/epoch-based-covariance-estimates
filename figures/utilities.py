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