# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:16:52 2020

@author: Hugues
"""

import numpy as np

class Component:
    
    def __init__(self, vc, idx):
        self.voxelcloud = vc
        self.idx = idx
        
    def get_voxels_idxs(self):
        return self.voxelcloud.components[self.idx]
    
    def get_voxels_3D_centers(self):
        return self.voxelcloud.s_geometric_center[self.voxelcloud.components[self.idx], :]
    
    def get_3D_points(self):
        points = []
        for vx in self.voxelcloud.components[self.idx]:
            points.append(self.voxelcloud.cloud.points[self.voxelcloud.cloud.voxels[vx], :])
        return np.vstack(points)
    
    def plot_component_points(self, ax, c = None):
        data = self.get_3D_points()
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': c} if c is not None else {}))
        
    def plot_component_voxel_centers(self, ax, c = None):
        data = self.get_voxels_3D_centers()
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': c} if c is not None else {}))
        
        
        