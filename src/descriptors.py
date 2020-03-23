# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KDTree

def local_PCA(points):
    """
    Parameters
    ----------
    points : Nxd numpy array containing a neighbourhood

    Returns
    -------
    eigenvalues : 1d 3-numpy array containing 3 eigenvalues of the coveriance matrix of the points
                  sorted in descending order
    eigenvectors : 3x3 numpy array containing the 3 eigenvectors columnwise
                  sorted in the same order as the eigenvalues
    """
    
    q = points - points.mean(axis=0)
    H = 1 / q.shape[0] * q.T.dot(q)    
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Reverting order (ascending -> descending)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, axis=1)
    
    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):
    """
    Parameters
    ----------
    query_points : Nx3 numpy array: points for which we will be computing the normals
    cloud_points : Nx3 numpy array: the reference points of the cloud where the 
                   neighbourhoods will be extracted
    radius : float : radius of the lookup sphere when computing the neighbourhood 
                     of each point of query_points

    Returns
    -------
    all_eigenvalues : Nx3 numpy array containing eigenvalues for each query point, sorted
                      in descending order
    all_eigenvectors : N x 3 x 3 numpy array containing eigenvectors for each query
                       point, sorted in the same order as eigenvalues

    """
    
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    
    kdt = KDTree(cloud_points)
    
    # Looking up for neighbors indexes
    neighbor_indexes = kdt.query_radius(query_points, r=radius)
    
    for i in range(len(query_points)):
        neighbors = cloud_points[neighbor_indexes[i]]
        eigval, eigvect = local_PCA(neighbors)
        
        all_eigenvalues[i,:] = eigval
        all_eigenvectors[i,:] = eigvect
    
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    """
    Parameters
    ----------
    query_points : Nx3 numpy array: points for which we will be computing the features
    cloud_points : Nx3 numpy array: the reference points of the cloud where the 
                   neighbourhoods will be extracted
    radius : float : radius of the lookup sphere when computing the neighbourhood 
                     of each point of query_points

    Returns
    -------
    verticality : 1d N-numpy array
    linearity : 1d N-numpy array
    planarity : 1d N-numpy array
    sphericity : 1d N-numpy array

    """
    
    eigvals, eigvects = neighborhood_PCA(query_points, cloud_points, radius)

    verticality = np.squeeze(2/np.pi * np.arcsin(abs(eigvects[:,:,2].dot([[0],[0],[1]]))))
    linearity = 1 - eigvals[:,1] / (eigvals[:,0] + 1e-6)
    planarity = (eigvals[:,1] - eigvals[:,2]) / (eigvals[:,0] + 1e-6)
    sphericity = eigvals[:,2] / (eigvals[:,0] + 1e-6)

    return verticality, linearity, planarity, sphericity