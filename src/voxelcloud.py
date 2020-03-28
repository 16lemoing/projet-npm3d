# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KDTree
from descriptors import local_PCA


class VoxelCloud:
    def __init__(self, pointcloud, max_voxel_size = 0.3, c_D = 0.25):
        self.pointcloud = pointcloud
        self.compute_voxels(max_voxel_size)
        
        nb_voxels = len(self)
        self.nb_points = np.zeros(nb_voxels)
        self.geometric_center = np.zeros((nb_voxels, 3))
        self.size = np.zeros((nb_voxels, 3))
        self.normal = np.zeros((nb_voxels, 3))
        self.barycenter = np.zeros((nb_voxels, 3))
        self.mean_intensity = np.nan * np.ones(nb_voxels)
        self.var_intensity = np.nan * np.ones(nb_voxels)
        self.mean_color = np.nan * np.ones((nb_voxels, 3))
        self.var_color = np.nan * np.ones((nb_voxels, 3))
        self.compute_features()
        
        self.kdt = KDTree(self.geometric_center)
        self.c_D = c_D
    
    def has_laser_intensity(self):
        return self.pointcloud.has_laser_intensity()
    
    def has_color(self):
        return self.pointcloud.has_color()
    
    def compute_voxels(self, max_voxel_size, seed = 42):
        """
        Transforms the point cloud into a list of list of indices
        Each sublist of indices is a voxel, represented by the indices of its points
        /!\ max_voxel_size is a diameter
        """
    
        np.random.seed(seed) # For reproducibility
        
        all_idxs = np.array(range(len(self.pointcloud)))
        available_idxs = all_idxs.copy()
        available_idxs_mask = np.array(len(self.pointcloud) * [True])
        
        self.voxels = []
        self.voxels_mask = []
        
        while len(available_idxs) > 0:
            print(len(available_idxs))
            
            # Picking one available index at random
            picked_idx = available_idxs[np.random.randint(0, len(available_idxs))]
            neighbours_idxs = self.pointcloud.get_r_nn([picked_idx], r = max_voxel_size / 2)[0]
            #neighbours_idxs = self.kdt.query_radius(picked_point[None, :], r = max_voxel_size / 2)[0]
            
            # Filtering results to keep the points which have not already been selected in voxels and determining bounding box
            neighbours_idxs = neighbours_idxs[available_idxs_mask[neighbours_idxs]]
            neighbours_coordinates = self.pointcloud.get_coordinates(neighbours_idxs)
            min_voxel, max_voxel = np.min(neighbours_coordinates, axis = 0), np.max(neighbours_coordinates, axis = 0)
            
            # Extracting all points withing the bounding box and filtering  to keep admissible points
            voxel_idxs_mask = np.all(self.pointcloud.get_coordinates() >= min_voxel, axis = 1) & np.all(self.pointcloud.get_coordinates() <= max_voxel, axis = 1)
            voxel_idxs_mask = voxel_idxs_mask & available_idxs_mask
            voxel_idxs = np.where(voxel_idxs_mask)[0]
            
            # These indices are not available anymore
            available_idxs_mask = available_idxs_mask & ~voxel_idxs_mask
            available_idxs = all_idxs[available_idxs_mask]
            
            # Storing into voxel
            self.voxels.append(voxel_idxs)
            self.voxels_mask.append(voxel_idxs_mask)
        
    def compute_features(self):
        for i in range(len(self)):
            coordinates_voxel_i = self.pointcloud.get_coordinates(self.voxels[i])
            vmin, vmax = np.min(coordinates_voxel_i, axis=0), np.max(coordinates_voxel_i, axis=0)
            eigval, eigvec = local_PCA(coordinates_voxel_i)
            
            self.nb_points[i] = len(self.voxels[i])
            self.geometric_center[i] = (vmin + vmax) / 2
            self.size[i] = vmax - vmin
            self.normal[i] = eigvec[:, 2]
            self.barycenter[i] = np.sum(coordinates_voxel_i, axis=0) / len(self.voxels[i])
            if self.has_laser_intensity():
                self.mean_intensity[i] = self.pointcloud.get_laser_intensity(self.voxels[i]).mean()
                self.var_intensity[i] = self.pointcloud.get_laser_intensity(self.voxels[i]).var()
            if self.has_color():
                self.mean_color[i] = self.pointcloud.get_color(self.voxels[i]).mean(axis = 0)
                self.var_color[i] = self.pointcloud.get_color(self.voxels[i]).var(axis = 0)
    
    def are_neighbours(self, i, j):
        """
        Generate a mask that tells whether one s-voxels and a group of s-voxels are neighbours or not (using the conditions from the link-chain method, cf. article)
        i : index
        j : index or list of indices
        """
        
        isnum = False
        if type(j) is int:
            isnum = True
            j = [j]
        
        gc_target = self.geometric_center[i, :]
        gc_candidates = self.geometric_center[j, :]
        size_target = self.size[i, :]
        size_candidates = self.size[j, :]
        vi_target = self.var_intensity[i]
        vi_candidates = self.var_intensity[j]
        mi_target = self.mean_intensity[i]
        mi_candidates = self.mean_intensity[j]
        vc_target = self.var_color[i, :]
        vc_candidates = self.var_color[j, :]
        mc_target = self.mean_color[i, :]
        mc_candidates = self.mean_color[j, :]
        
        w_D = (size_target + size_candidates) / 2
        cond_D = np.all(abs(gc_target - gc_candidates) <= w_D + self.c_D, axis=1)
        
        cond_I = np.ones(len(j), dtype=bool)
        if self.has_laser_intensity() is not None:
            w_I = np.maximum(vi_target, vi_candidates)
            cond_I = abs(mi_target - mi_candidates) <= 3 * np.sqrt(w_I)
            
        cond_C = np.ones(len(j), dtype=bool)
        if self.has_color() is not None:
            w_C = np.maximum(vc_target, vc_candidates)
            cond_C = np.all(abs(mc_target - mc_candidates) <= 3 * np.sqrt(w_C), axis=1)
        
        cond = cond_D & cond_I & cond_C
        if isnum:
            return cond[0]
        
        return cond
    
    def find_spatial_neighbours(self, idxs):
        """ Returns a list of indices of potential neighbouring voxels """
        max_size = np.max(self.size)
        if type(idxs) is int:
            return self.kdt.query_radius(self.geometric_center[[idxs], :], r = max_size + self.c_D)[0]
        return self.kdt.query_radius(self.geometric_center[idxs, :], r = max_size + self.c_D)
        
    
    def find_neighbours(self, idxs):
        """ Returns a list of all indices who are truly neighbours of each index in idxs """
        
        isnum = False
        if type(idxs) is int:
            isnum = True
            idxs = [idxs]
        
        neighbours = []
        potential = self.find_spatial_neighbours(idxs)
        for i in range(len(idxs)):
            neighbours.append(potential[i][self.are_neighbours(idxs[i], potential[i])])
            
        if isnum:
            return neighbours[0]
            
        return neighbours    
    
    def get_all_3D_points_of_voxel(self, i):
        return self.pointcloud.get_coordinates(self.voxels[i])    
    
    def __len__(self):
        return len(self.voxels)