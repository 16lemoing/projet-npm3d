#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import time
import numpy as np
from sklearn.neighbors import KDTree
from utils.ply import read_ply, write_ply
from plots import show_points
from cloud import Cloud


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

# data = read_ply("../data/bildstein_station5_xyz_intensity_rgb.ply")
# cloud = np.vstack((data['x'], data['y'], data['z'])).T
# rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
# dlaser = data['reflectance']
# print(cloud.shape)
# print(rgb_colors.shape)
# print(dlaser.shape)

# # Select portion of data
# mask = ((cloud[:,0] > 9) & (cloud[:,0] < 17) & (cloud[:,1] > -51) & (cloud[:,1] < -31))
# extracted_cloud = cloud[mask]
# extracted_rgb_colors = rgb_colors[mask]
# extracted_dlaser = dlaser[mask]
# write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract.ply', [extracted_cloud, extracted_dlaser, extracted_rgb_colors],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])


# # Reduce number of points
# decimation = 10
# limit = 1000000
# write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply', [extracted_cloud[:limit:decimation], extracted_dlaser[:limit:decimation], extracted_rgb_colors[:limit:decimation]],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])

### Test script

## Retrieve data

data = read_ply("../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']

## Defining cloud and computing voxels and features
c = Cloud(cloud, dlaser, rgb_colors)
c.compute_voxels()
# access with c.voxels, c.voxels_mask
c.compute_voxels_features()
# access with c.s_voxels
c.are_neighbour_voxels(1, [1, 7])

## Display voxels
v_mi = c.s_voxels.loc[:, "mean_intensity"].values[:,0]
normalized_v_mi = (v_mi - np.min(v_mi)) / np.ptp(v_mi)
show_points(c, c.voxels, normalized_v_mi)

colors = np.random.random(len(c.voxels))
show_points(c, c.voxels, colors)

## Link and show components

components = link_chain_voxels_fast(voxels, v_gc, v_mi, v_vi, v_s, v_n, v_mc, v_vc)
colors = np.zeros(len(voxels))
for i in range(len(components)):
    colors[np.array(components[i])] = np.random.random()
show_points(cloud, voxels, colors)