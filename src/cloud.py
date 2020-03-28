# -*- coding: utf-8 -*-

from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from descriptors import local_PCA

class Cloud:
    
    def __init__(self, points, laser_intensity = None, rgb_colors = None):
        self.points = points
        self.laser_intensity = laser_intensity
        self.rgb_colors = rgb_colors
        self.voxels = None
        self.kdt = KDTree(self.points) # Can take a few seconds to build
    
    @staticmethod
    def determine_bounding_box(points):
        return np.min(points, axis=0), np.max(points, axis=0)
    
    def compute_voxels(self, max_voxel_size = 0.3, seed = 42):
        """
        Transforms the cloud into a list of list of indices
        Each sublist of indices is a voxel, represented by the indices of its points
        /!\ max_voxel_size is a diameter
        """
    
        np.random.seed(seed) # For reproducibility
        
        all_idxs = np.array(range(len(self.points)))
        available_idxs = all_idxs.copy()
        available_idxs_mask = np.array(len(self.points) * [True])
        
        self.voxels = []
        self.voxels_mask = []
        
        while len(available_idxs) > 0:
            print(len(available_idxs))
            
            # Picking one available index at random
            picked_idx = available_idxs[np.random.randint(0, len(available_idxs))]
            picked_point = self.points[picked_idx, :]
            neighbours_idxs = self.kdt.query_radius(picked_point[None, :], r = max_voxel_size / 2)[0]
            
            # Filtering results to keep the points which have not already been selected in voxels and determining bounding box
            neighbours_idxs = neighbours_idxs[available_idxs_mask[neighbours_idxs]]
            min_voxel, max_voxel = self.determine_bounding_box(self.points[neighbours_idxs,:])
            
            # Extracting all points withing the bounding box and filtering  to keep admissible points
            voxel_idxs_mask = np.all(self.points >= min_voxel, axis = 1) & np.all(self.points <= max_voxel, axis = 1)
            voxel_idxs_mask = voxel_idxs_mask & available_idxs_mask
            voxel_idxs = np.where(voxel_idxs_mask)[0]
            
            # These indices are not available anymore
            available_idxs_mask = available_idxs_mask & ~voxel_idxs_mask
            available_idxs = all_idxs[available_idxs_mask]
            
            # Storing into voxel
            self.voxels.append(voxel_idxs)
            self.voxels_mask.append(voxel_idxs_mask)
        
    def compute_voxels_features(self):
        """
        Computes the s-voxels properties of a list of list of indices
        """
        if self.voxels is None:
            print("Voxels have not been computed yet. Computing them with default parameters.")
            self.compute_voxels()
            
        nb_voxels = len(self.voxels)
        
        nb_points = np.zeros(nb_voxels)
        geometric_center = np.zeros((nb_voxels, 3))
        size = np.zeros((nb_voxels, 3))
        normal = np.zeros((nb_voxels, 3))
        barycenter = np.zeros((nb_voxels, 3))
        mean_intensity = np.nan * np.ones(nb_voxels)
        var_intensity = np.nan * np.ones(nb_voxels)
        mean_color = np.nan * np.ones((nb_voxels, 3))
        var_color = np.nan * np.ones((nb_voxels, 3))
        
        for i in range(len(self.voxels)):
            nb_points[i] = len(self.voxels[i])
            vmin, vmax = self.determine_bounding_box(self.points[self.voxels[i], :])
            geometric_center[i] = (vmin + vmax) / 2
            size[i] = vmax - vmin
            eigval, eigvec = local_PCA(self.points[self.voxels[i], :])
            normal[i] = eigvec[:, 2]
            barycenter[i] = np.sum(self.points[self.voxels[i],:], axis=0) / len(self.voxels[i])
            if self.laser_intensity is not None:
                mean_intensity[i] = self.laser_intensity[self.voxels[i]].mean()
                var_intensity[i] = self.laser_intensity[self.voxels[i]].var()
            if self.rgb_colors is not None:
                mean_color[i] = self.rgb_colors[self.voxels[i]].mean(axis = 0)
                var_color[i] = self.rgb_colors[self.voxels[i]].var(axis = 0)
        
        self.s_voxels = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([
                ("number_points", "nb"),
                ("geometric_center", "x"),("geometric_center", "y"), ("geometric_center", "z"),
                ("size", "x"),("size", "y"), ("size", "z"),
                ("normal", "x"),("normal", "y"), ("normal", "z"),
                ("barycenter", "x"), ("barycenter", "y"), ("barycenter", "z"),
                ("mean_intensity", "i"), ("var_intensity", "i"),
                ("mean_color", "r"), ("mean_color", "g"), ("mean_color", "b"),
                ("var_color", "r"), ("var_color", "g"), ("var_color", "b")
            ]), 
            index = range(nb_voxels),
            data = np.hstack((
                nb_points[:,None], 
                geometric_center, 
                size, 
                normal, 
                barycenter, 
                mean_intensity[:,None], 
                var_intensity[:,None], 
                mean_color, 
                var_color
            ))
        )
        
    def __len__(self):
        return len(self.points)