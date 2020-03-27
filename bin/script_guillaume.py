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

def compute_voxel_features(cloud, dlaser, voxels, rgb_colors=None):
    """
    Computes the s-voxels properties of a list of list of indices
    """
    nb_voxels = len(voxels)
    
    voxel_geometric_center = np.zeros((nb_voxels, 3))
    voxel_mean_intensity = np.zeros(nb_voxels)
    voxel_var_intensity = np.zeros(nb_voxels)
    voxel_size = np.zeros((nb_voxels, 3))
    voxel_normal = np.zeros((nb_voxels, 3))
    voxel_mean_color = np.zeros((nb_voxels, 3))
    voxel_var_color = np.zeros((nb_voxels, 3))
    
    for i in range(len(voxels)):
        vmin, vmax = determine_bounding_box(cloud[voxels[i], :])
        voxel_geometric_center[i] = (vmin + vmax) / 2
        voxel_mean_intensity[i] = dlaser[voxels[i]].mean()
        voxel_var_intensity[i] = dlaser[voxels[i]].var()
        voxel_size[i] = vmax - vmin
        eigval, eigvec = local_PCA(cloud[voxels[i], :])
        voxel_normal[i] = eigvec[:, 2]
        if rgb_colors is not None:
            voxel_mean_color[i] = rgb_colors[voxels[i]].mean(axis = 0)
            voxel_var_color[i] = rgb_colors[voxels[i]].var(axis = 0)
    
    return voxel_geometric_center, voxel_mean_intensity, voxel_var_intensity, voxel_size, voxel_normal, voxel_mean_color, voxel_var_color
        

def are_neighbours(gc1, gc2, mi1, mi2, vi1, vi2, s1, s2, mc1=None, mc2=None, vc1=None, vc2=None, c_D=0.25):
    """
    Tells whether two s-voxels are neighbours or not (using the conditions from the link-chain method, cf. article)
    """
    w_D = (s1 + s2) / 2
    cond_D = np.all(abs(gc1 - gc2) <= w_D + c_D)
    w_I = max(vi1, vi2)
    cond_I = (abs(mi1 - mi2) <= 3 * np.sqrt(w_I))
    if all(v is not None for v in [mc1, mc2, vc1, vc2]):
        w_C = np.max((vc1, vc2), axis=0)
        cond_C = np.all(abs(mc1 - mc2) <= 3 * np.sqrt(w_C))
    else: cond_C = True
    return cond_D & cond_I & cond_C
    
def are_neighbours_multi(gc, gc_multi, mi, mi_multi, vi, vi_multi, s, s_multi, mc=None, mc_multi=None, vc=None, vc_multi=None, c_D=0.25):
    """
    Generate a mask that tells whether one s-voxels and a group of s-voxels are neighbours or not (using the conditions from the link-chain method, cf. article)
    """
    w_D = (s + s_multi) / 2
    cond_D = np.all(abs(gc - gc_multi) <= w_D + c_D, axis=1)
    w_I = np.maximum(vi, vi_multi)
    cond_I = (abs(mi - mi_multi) <= 3 * np.sqrt(w_I))
    if all(v is not None for v in [mc, mc_multi, vc, vc_multi]):
        w_C = np.maximum(vc, vc_multi)
        cond_C = np.all(abs(mc - mc_multi) <= 3 * np.sqrt(w_C), axis=1)
    else: cond_C = True
    return cond_D & cond_I & cond_C
    

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


def link_chain_voxels(voxels, geometric_centers, mean_intensity, var_intensity, size, normals, mean_color=None, var_color=None):
    """
    Builds a list of connected components of voxels from a list of voxels
    """
    n_voxels = len(voxels)
    
    # Construction d'une matrice d'adjacence
    A = np.zeros((n_voxels, n_voxels))
    
    # If we have color information, use it to find neighborhoods
    if mean_color is not None and var_color is not None:
        for i in range(n_voxels):
            print(i)
            for j in range(i):
                A[i,j] = are_neighbours(geometric_centers[i], geometric_centers[j], mean_intensity[i], mean_intensity[j],
                                        var_intensity[i], var_intensity[j], size[i], size[j],
                                        mean_color[i], mean_color[j], var_color[i], var_color[j])
    else:
        for i in range(n_voxels):
            print(i)
            for j in range(i):
                A[i,j] = are_neighbours(geometric_centers[i], geometric_centers[j], mean_intensity[i], mean_intensity[j],
                                        var_intensity[i], var_intensity[j], size[i], size[j])
    
    A = A + A.T
    A += np.eye(n_voxels)
    
    components = explore_connected_components(A)
    return A, components
    
    
def link_chain_voxels_fast(voxels, geometric_centers, mean_intensity, var_intensity, size, normals, mean_color=None, var_color=None):
    """
    Builds a list of connected components of voxels from a list of voxels
    """
    
    n_voxels = len(voxels)
    kdt = KDTree(geometric_centers) # Can take a few seconds to build
    
    # Find potential neighboring voxels for all voxels
    c_D = 0.25
    max_size = np.max(size)
    potential_neighbours_idxs = kdt.query_radius(geometric_centers, r = max_size + c_D)
    
    # Check if they are truely neighbours
    voxel_neighbours = []
    for i in range(n_voxels):
        neighbours_mask = are_neighbours_multi(geometric_centers[i], geometric_centers[potential_neighbours_idxs[i]],
                                               mean_intensity[i], mean_intensity[potential_neighbours_idxs[i]],
                                               var_intensity[i], var_intensity[potential_neighbours_idxs[i]],
                                               size[i], size[potential_neighbours_idxs[i]],
                                               mean_color[i], mean_color[potential_neighbours_idxs[i]],
                                               var_color[i], var_color[potential_neighbours_idxs[i]])
        voxel_neighbours.append(potential_neighbours_idxs[i][neighbours_mask])
    
    # Explore connected components
    components = []
    indices = np.array(list(range(n_voxels)))
    indices_mask = np.ones(n_voxels, dtype=bool)
    
    while len(indices) > 0:
        print(len(indices))
        stack = [indices[0]]
        current_component = []
        
        # Run a depth first search to find all connected voxels
        while len(stack) > 0:
            idx = stack.pop()
            if ~indices_mask[idx]:
                continue
            current_component.append(idx)
            indices_mask[idx] = False 
            next_idxs = voxel_neighbours[idx]
            stack.extend(list(next_idxs[indices_mask[next_idxs]]))
        components.append(current_component)
        
        # Updating indices
        indices = np.array(list(range(n_voxels)))[indices_mask]
    
    return components
    
## Preparing data

data = read_ply("../data/bildstein_station5_xyz_intensity_rgb.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']
print(cloud.shape)
print(rgb_colors.shape)
print(dlaser.shape)

# Select portion of data
mask = ((cloud[:,0] > 9) & (cloud[:,0] < 17) & (cloud[:,1] > -51) & (cloud[:,1] < -31))
extracted_cloud = cloud[mask]
extracted_rgb_colors = rgb_colors[mask]
extracted_dlaser = dlaser[mask]
write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract.ply', [extracted_cloud, extracted_dlaser, extracted_rgb_colors],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])


# Reduce number of points
decimation = 10
limit = 1000000
write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply', [extracted_cloud[:limit:decimation], extracted_dlaser[:limit:decimation], extracted_rgb_colors[:limit:decimation]],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])

### Test script

## Retrieve data

data = read_ply("../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']

## Retrieve voxels

voxels, voxels_mask = voxelize_cloud(cloud)

## Compute features

v_gc, v_mi, v_vi, v_s, v_n, v_mc, v_vc = compute_voxel_features(cloud, dlaser, voxels, rgb_colors)

## Display voxels

normalized_v_mi = (v_mi - np.min(v_mi))/np.ptp(v_mi)
show_points(cloud, voxels, normalized_v_mi)

colors = np.random.random(len(voxels))
show_points(cloud, voxels[0:], colors)

## Link and show components

components = link_chain_voxels_fast(voxels, v_gc, v_mi, v_vi, v_s, v_n, v_mc, v_vc)
colors = np.zeros(len(voxels))
for i in range(len(components)):
    colors[np.array(components[i])] = np.random.random()
show_points(cloud, voxels, colors)

##
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
colors = np.random.random(len(voxels))
show_points(cloud, voxels[0:], colors)
# show_points(cloud, voxels[0:], v_mi)

A, components = link_chain_voxels(voxels, v_gc, v_mi, v_vi, v_s, v_n)
colors = np.zeros(len(voxels))
for i in range(len(components)):
    colors[np.array(components[i])] = np.random.random()
    
show_points(cloud, voxels, colors)
        
    