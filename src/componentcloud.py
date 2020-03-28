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
            
    
    def get_all_3D_points_of_component(self, i):
        points = []
        for vx_id in self.components[i]:
            points.append(self.voxelcloud.get_all_3D_points_of_voxel(vx_id))
        return np.vstack(points)
        
    def __len__(self):
        return len(self.components)
        
        