#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import time
import numpy as np
from sklearn.neighbors import KDTree
from utils.ply import read_ply, write_ply
from plots import show_points
from descriptors import local_PCA



# The functions currently in this file for testing should be moved later into the source directory 

def determine_bounding_box(points):
    return np.min(points, axis=0), np.max(points, axis=0)
    
def voxelize_cloud(cloud, max_voxel_size = 0.3, seed = 42):
    """
    Transforms a cloud into a list of list of indices
    Each sublist of indices is a voxel, represented by the indices of its points
    """
    
    kdt = KDTree(cloud) # Can take a few seconds to build
    
    np.random.seed(seed) # For reproducibility
    
    all_idxs = np.array(range(len(cloud)))
    available_idxs = all_idxs.copy()
    available_idxs_mask = np.array(len(cloud) * [True])
    
    voxels = []
    voxels_mask = []
    
    while len(available_idxs) > 0:
        print(len(available_idxs))
        
        # Picking one available index at random
        picked_idx = available_idxs[np.random.randint(0, len(available_idxs))]
        picked_point = cloud[picked_idx, :]
        neighbours_idxs = kdt.query_radius(picked_point[None, :], r = max_voxel_size / 2)[0]
        
        # Filtering results to keep the points which have not already been selected in voxels and determining bounding box
        neighbours_idxs = neighbours_idxs[available_idxs_mask[neighbours_idxs]]
        min_voxel, max_voxel = determine_bounding_box(cloud[neighbours_idxs,:])
        
        # Extracting all points withing the bounding box and filtering  to keep admissible points
        voxel_idxs_mask = np.all(cloud >= min_voxel, axis = 1) & np.all(cloud <= max_voxel, axis = 1)
        voxel_idxs_mask = voxel_idxs_mask & available_idxs_mask
        voxel_idxs = np.where(voxel_idxs_mask)[0]
        
        # These indices are not available anymore
        available_idxs_mask = available_idxs_mask & ~voxel_idxs_mask
        available_idxs = all_idxs[available_idxs_mask]
        
        # Storing into voxel
        voxels.append(voxel_idxs)
        voxels_mask.append(voxel_idxs_mask)
    
    return voxels, np.array(voxels_mask)

def compute_voxel_features(cloud, dlaser, voxels):
    """
    Computes the s-voxels properties of a list of list of indices
    """
    nb_voxels = len(voxels)
    
    # TODO C'est quoi le "centre géométrique" ? Ici ma supposition
    voxel_geometric_center = np.zeros((nb_voxels, 3))
    voxel_mean_intensity = np.zeros(nb_voxels)
    voxel_var_intensity = np.zeros(nb_voxels)
    voxel_size = np.zeros((nb_voxels, 3))
    voxel_normal = np.zeros((nb_voxels, 3))
    
    for i in range(len(voxels)):
        vmin, vmax = determine_bounding_box(cloud[voxels[i], :])
        voxel_geometric_center[i] = (vmin + vmax) / 2
        voxel_mean_intensity[i] = dlaser[voxels[i]].mean()
        voxel_var_intensity[i] = dlaser[voxels[i]].var()
        voxel_size[i] = vmax - vmin
        eigval, eigvec = local_PCA(cloud[voxels[i], :])
        voxel_normal[i] = eigvec[:, 2]
        
    return voxel_geometric_center, voxel_mean_intensity, voxel_var_intensity, voxel_size, voxel_normal
        

def are_neighbours(gc1, gc2, mi1, mi2, vi1, vi2, s1, s2, c_D=0.25):
    """
    Tells whether two s-voxels are neighbours or not (using the conditions from the link-chain method, cf. article)
    """
    w_D = (s1 + s2) / 2
    w_I = max(vi1, vi2)
    return np.all(abs(gc1 - gc2) <= w_D + c_D) & (abs(mi1 - mi2) <= np.sqrt(w_I))
    

def explore_connected_components(A):
    """
    Builds the connected components of the link-chain method
    (used for segmentation of the cloud of s-voxels)
    """
    components = []
    indices = np.array(list(range(len(A))))
    indices_mask = np.ones(len(A), dtype=bool)
    
    while len(indices) > 0:
        print(len(indices))
        stack = [indices[0]]
        current_component = []
        
        # From now, indices is not up to date anymore : do not use it
        while len(stack) > 0:
            idx = stack.pop()
            if ~indices_mask[idx]:
                continue
            current_component.append(idx)
            indices_mask[idx] = False 
            next_idxs = np.where(A[:,idx])[0]
            stack.extend(list(next_idxs[indices_mask[next_idxs]]))
        components.append(current_component)
        
        # Updating indices
        indices = np.array(list(range(len(A))))[indices_mask]
        
    return components
        
        

def link_chain_voxels(voxels, geometric_centers, mean_intensity, var_intensity, size, normals):
    """
    Builds a list of connected components of voxels from a list of voxels
    """
    n_voxels = len(voxels)
    
    # Construction d'une matrice d'adjacence
    A = np.zeros((n_voxels, n_voxels))
    
    for i in range(n_voxels):
        print(i)
        for j in range(i):
            A[i,j] = are_neighbours(geometric_centers[i], geometric_centers[j], mean_intensity[i], mean_intensity[j],
                                    var_intensity[i], var_intensity[j], size[i], size[j])
    
    A = A + A.T
    A += np.eye(n_voxels)
    
    components = explore_connected_components(A)
    return A, components
    

# Generating a smaller dataset for tests
# data = read_ply("../data/GT_Madame1_2.ply")
# cloud = np.vstack((data['x'], data['y'], data['z'])).T
# dlaser = data['reflectance']
# dlabel = data["label"]
# dclass = data["class"]
# limit = 1000000
# decimation = 20
# write_ply('../data/Madame_extremelysmall2.ply', [cloud[:limit:decimation], dlaser[:limit:decimation].astype(int), dlabel[:limit:decimation].astype(int), dclass[:limit:decimation].astype(int)],['x', 'y', 'z', 'reflectance', 'label', 'class'])

data = read_ply("../data/Madame_extremelysmall2.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
dlaser = data['reflectance']
dlabel = data["label"]
dclass = data["class"]

voxels, voxels_mask = voxelize_cloud(cloud)
v_gc, v_mi, v_vi, v_s, v_n = compute_voxel_features(cloud, dlaser, voxels)
# TODO On pourrait transformer les voxels en objets pour éviter de se trimballer toutes les variables à chaque fois ?
show_points(cloud, voxels, v_mi)

A, components = link_chain_voxels(voxels, v_gc, v_mi, v_vi, v_s, v_n)
colors = np.zeros(len(voxels))
for i in range(len(components)):
    colors[np.array(components[i])] = np.random.random()
    
show_points(cloud, voxels, colors)
        
    