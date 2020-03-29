# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:16:52 2020

@author: Hugues
"""

import numpy as np

class ComponentCloud:
    
    def __init__(self, voxelcloud):
        self.voxelcloud = voxelcloud
        
        self.components = [] # This is a list of list of indices (each index is the #id of a voxel in voxelcloud)
        self.compute_connected_components() # Fills self.components
        
        
        # Initializes and declares features
        self.barycenter = np.ones((len(self), 3))
        self.mean_intensity = np.nan * np.ones(len(self))
        self.mean_color = np.nan * np.ones((len(self), 3))
        self.compute_features() # Computes features
        
        
    def compute_connected_components(self):
        """
        Builds a list of connected components of voxels from a list of voxels
        """
        
        n_voxels = len(self.voxelcloud)
        voxel_neighbours = self.voxelcloud.find_neighbours(list(range(n_voxels)))
        
        # Explore connected components
        self.components = []
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
            self.components.append(current_component)
            
            # Updating indices
            indices = np.array(list(range(n_voxels)))[indices_mask]
            
    
    def compute_features(self):
        for i in range(len(self)):
            vx_barycenters = self.voxelcloud.barycenter[self.components[i]]
            vx_colors = self.voxelcloud.mean_color[self.components[i]]
            vx_intensities = self.voxelcloud.mean_intensity[self.components[i]]
            vx_nb_points = self.voxelcloud.nb_points[self.components[i]]
            
            self.barycenter[i, :] = np.sum(vx_barycenters * vx_nb_points[:, None], axis=0) / np.sum(vx_nb_points)
            self.mean_color[i, :] = np.sum(vx_colors * vx_nb_points[:, None], axis = 0) / np.sum(vx_nb_points)
            self.mean_intensity[i] = np.sum(vx_intensities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
        
    
    def get_all_3D_points_of_component(self, i):
        points = []
        for vx_id in self.components[i]:
            points.append(self.voxelcloud.get_all_3D_points_of_voxel(vx_id))
        return np.vstack(points)
        
    
    def has_color(self):
        """ Tells whether this cloud of connected components has color information """
        return self.voxelcloud.has_color()
    
    def has_laser_intensity(self):
        """ Tells whether this cloud of connected commonents has laser intensity information """
        return self.voxelcloud.has_laser_intensity()
    
    def __len__(self):
        return len(self.components)
        
        