# -*- coding: utf-8 -*-

import numpy as np

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

def features_from_PCA(eigvals, eigvects):
    """
    Parameters
    ----------
    eigenvalues : 1d 3-numpy array containing 3 eigenvalues of the coveriance matrix of some set of points
                  sorted in descending order
    eigenvectors : 3x3 numpy array containing the 3 eigenvectors columnwise
                  sorted in the same order as the eigenvalues

    Returns
    -------
    verticality : 1d N-numpy array
    linearity : 1d N-numpy array
    planarity : 1d N-numpy array
    sphericity : 1d N-numpy array

    """

    verticality = 2 / np.pi * np.arcsin(abs(eigvects[:, 2].dot([[0], [0], [1]])))[0]
    linearity = 1 - eigvals[1] / (eigvals[0] + 1e-6)
    planarity = (eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-6)
    sphericity = eigvals[2] / (eigvals[0] + 1e-6)

    return verticality, linearity, planarity, sphericity