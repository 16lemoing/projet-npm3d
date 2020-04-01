# -*- coding: utf-8 -*-

import numpy as np
from utils.ply import write_ply, read_ply
import time

def compute_plane(points):
    """
    Computing a plane passing through 3 input points

    Parameters
    ----------
    points : 3x3 numpy array (3 points stacked in a row-wise way)

    Returns
    -------
    point : a point in the found plane
    normal : a unit normal vector of the found plane
    """
    
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal = normal / np.sqrt(np.sum(normal**2))
    
    point = points[0]
    
    return point, normal


def in_plane(points, normals, ref_pt, ref_normal, threshold_in=0.1, threshold_normals=0.8):
    """
    Checks whether the given points belong to a given plane (with some threshold value)

    Parameters
    ----------
    points : Nx3 numpy array
    ref_pt : 3-numpy array (a point of the plane)
    ref_normal : 3-numpy array (unit vector of the plane)
    threshold_in : float: maximum distance to the plane for points to belong to it

    Returns
    -------
    indices : N-numpy array of booleans telling which points belong to the plane
    """
    dists = np.einsum("i,ji->j", ref_normal, points - ref_pt)
    indices = abs(dists) < threshold_in
    if normals is not None:
        normal_check = np.dot(ref_normal, normals.T) > threshold_normals
        return indices & normal_check
    return indices


def RANSAC(points, normals=None, NB_RANDOM_DRAWS=100, threshold_in=0.1, threshold_normals=0.8):
    """
    Applies the RANSAC algorithm to find a plane

    Parameters
    ----------
    points : Nx3 numpy array
    NB_RANDOM_DRAWS : number of tries: the biggest plane of all draws will be taken
    threshold_in : float : distance threshold telling whether a point belongs to a plane or not

    Returns
    -------
    best_ref_pt : 3-numpy array (point belonging to the best found plane)
    best_normal : 3-numpy array (unit normal vector of the best found plane)
    """
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_nb = -1
    
    for i in range(NB_RANDOM_DRAWS):
        # Drawing 3 different point indices at random
        rand_indices = np.zeros(3).astype(int)
        while len(np.unique(rand_indices)) != 3:
            rand_indices = np.random.randint(0, np.shape(points)[0], size=3)
            
        # Extracting the corresponding points
        pts = points[rand_indices]
        
        # Finding the associated plane
        ref_pt, ref_normal = compute_plane(pts)
        
        # Couting the number of points in this plane
        nb = np.sum(in_plane(points, normals, ref_pt, ref_normal, threshold_in))
        
        # Updating the best plane if needed
        if nb > best_nb:
            best_nb = nb
            best_ref_pt = ref_pt
            best_normal = ref_normal
                
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    """
    Applies the multi RANSAC algorithm to find several best planes

    Parameters
    ----------
    points : Nx3 numpy array
    NB_RANDOM_DRAWS : number of tries: the biggest plane of all draws will be taken
    threshold_in : float : distance threshold telling whether a point belongs to a plane or not
    NB_PLANES: int : the number of points we want to find

    Returns
    -------
    plane_inds : 1d numpy array: the indices of all points belonging the the best planes
    remaining_inds : 1d numpy array: all the other indices
    plane_labels : 1d numpy array of ints (same size as plane_inds) : the labels of the planes
                   of all points in plane_inds

    """
    
    # This is the list of indices for which no planes have been found at the moment
    # We will successively look inside it, remove the best plane and proces again
    # until finding NB_PLANES planes
    remaining_inds = np.arange(0, len(points), 1)
    
    plane_inds = np.array([]).astype(int)
    plane_labels = np.array([]).astype(int)
    
    for p in range(NB_PLANES):
        # Applying RANSAC on the current working remaining points
        best_ref_pt, best_normal = RANSAC(points[remaining_inds,:], NB_RANDOM_DRAWS, threshold_in)
        # Extracting the booleans telling whether remaining_inds are in the found plane
        points_in_plane = in_plane(points[remaining_inds,:], best_ref_pt, best_normal, threshold_in)
        # Converting to indices in the plane and outside it
        this_plane_inds = remaining_inds[np.where(points_in_plane)[0]]
        remaining_inds = remaining_inds[np.where(~points_in_plane)[0]] # Updating the working indices
        
        # Adding the plane to the found planes
        plane_inds = np.hstack((plane_inds, this_plane_inds))
        plane_labels = np.hstack((plane_labels, p * np.ones(len(this_plane_inds))))
        
    return plane_inds, remaining_inds, plane_labels
