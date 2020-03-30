# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:16:52 2020

@author: Hugues
"""

import numpy as np
from descriptors import local_PCA

class ComponentCloud:
    
    def __init__(self, voxelcloud, c_D = 0.25):
        """
            Builds a cloud of connected component from a voxel cloud
    
            Parameters
            ----------
            voxelcloud : VoxelCloud object
            c_D : float, optional
                Constant used when computing the voxel neighbours, in order to correct
                the geometric distance. The default is 0.25.
        """
                
        self.voxelcloud = voxelcloud
        self.c_D = c_D
        
        self.components = [] # This is a list of list of indices (each index is the #id of a voxel in voxelcloud)
        self.compute_connected_components() # Fills self.components
        
        
        # Initializes and declares features
        self.nb_points = np.ones(len(self), dtype=int)
        self.nb_voxels = np.ones(len(self), dtype=int)
        self.barycenter = np.ones((len(self), 3))
        self.geometrical_center = np.ones((len(self), 3))
        self.mean_znormal = np.ones(len(self))
        self.var_znormal = np.ones(len(self))
        self.mean_xynormal = np.ones(len(self))
        self.var_xynormal = np.ones(len(self))
        self.mean_intensity = np.nan * np.ones(len(self))
        self.var_intensity = np.nan * np.ones(len(self))
        self.mean_color = np.nan * np.ones((len(self), 3))
        self.var_color = np.nan * np.ones((len(self), 3))
        self.pca_val = np.ones((len(self), 3))
        self.pca_vec = np.ones((len(self), 9))
        self.size = np.ones((len(self), 3))
        self.mean_verticality = np.ones(len(self))
        self.mean_linearity = np.ones(len(self))
        self.mean_planarity = np.ones(len(self))
        self.mean_sphericity = np.ones(len(self))
        self.computed_label = np.nan * np.ones(len(self), dtype=int)
        self.compute_features() # Computes features
        
        
    def compute_connected_components(self):
        """
            Builds a list of connected components of voxels from a voxelcloud object,
            by performing a depth first search by using the neighbourhood condition
            of voxels
            
            The list of connected components is then stored in 
            self.components
            
            Each item of self.component is a list of indices, which are the indices of
            the underlying VoxelCloud object
            eg. self.components[i] = [1, 2, 3]
                means that this component is made up of voxels 1, 2 and 3
        """
        
        n_voxels = len(self.voxelcloud)
        voxel_neighbours = self.voxelcloud.find_neighbours(list(range(n_voxels)), self.c_D)
        
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
        """
            Computes the individual features of each connected component
        """
        
        for i in range(len(self)):
            vx_nb_points = self.voxelcloud.nb_points[self.components[i]]
            vx_barycenters = self.voxelcloud.barycenter[self.components[i]]
            vx_colors = self.voxelcloud.mean_color[self.components[i]]
            vx_intensities = self.voxelcloud.mean_intensity[self.components[i]]
            vx_nb_points = self.voxelcloud.nb_points[self.components[i]]
            vx_geometric_centers = self.voxelcloud.geometric_center[self.components[i]]
            vx_sizes = self.voxelcloud.size[self.components[i]]
            vx_normals = self.voxelcloud.normal[self.components[i]]
            vx_verticalities = self.voxelcloud.verticality[self.components[i]]
            vx_linerities = self.voxelcloud.linearity[self.components[i]]
            vx_planarities = self.voxelcloud.planarity[self.components[i]]
            vx_sphericities = self.voxelcloud.sphericity[self.components[i]]
            
            self.nb_points[i] = np.sum(vx_nb_points)
            self.nb_voxels[i] = len(self.components[i])
            self.barycenter[i, :] = np.sum(vx_barycenters * vx_nb_points[:, None], axis=0) / np.sum(vx_nb_points)
            p_max = np.max(vx_geometric_centers + vx_sizes / 2, axis=0)
            p_min = np.min(vx_geometric_centers + vx_sizes / 2, axis=0)
            self.geometrical_center[i, :] = (p_max + p_min) / 2
            self.size[i, :] = p_max - p_min
            self.mean_color[i, :] = np.sum(vx_colors * vx_nb_points[:, None], axis = 0) / np.sum(vx_nb_points)
            self.var_color[i, :] = np.sum((vx_colors - self.mean_color[i, :]) ** 2 * vx_nb_points[:, None], axis = 0) / np.sum(vx_nb_points)
            self.mean_intensity[i] = np.sum(vx_intensities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            #self.var_intensity[i] = np.sum((vx_intensities - self.mean_intensity[i, :]) ** 2 * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.mean_znormal[i] = np.sum(vx_normals[:, 2] * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.var_znormal[i] = np.sum((vx_normals[:, 2] - self.mean_znormal[i]) ** 2 * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.mean_xynormal[i] = np.sum(np.linalg.norm(vx_normals[:, :2], axis = 1) * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.var_xynormal[i] = np.sum((np.linalg.norm(vx_normals[:, :2], axis = 1) - self.mean_xynormal[i]) ** 2 * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            eigval, eigvec = local_PCA(vx_geometric_centers)
            self.pca_val[i, :] = eigval
            self.pca_vec[i, :] = eigvec.flatten()
            self.mean_verticality[i] = np.sum(vx_verticalities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.mean_linearity[i] = np.sum(vx_linerities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.mean_planarity[i] = np.sum(vx_planarities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
            self.mean_sphericity[i] = np.sum(vx_sphericities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)


    def get_features(self):
        #TODO: stack all features
        pass
    
    def eval_classification_error(self, idx = None, ground_truth_type = "pointwise"):
        """
            Computes the classification error for one ore several components in the cloud
        """
        if type(idx) is int:
            idx = [idx]
        if idx is None:
            idx = list(range(len(self)))
            
        ground_truth = []
        computed = []
        for i in idx:
            if ground_truth_type == "pointwise":
                ground_truth.append(self.get_labels_of_all_3D_points_in_component(i))
            else:
                raise NotImplementedError()
            
            computed.append(self.nb_points[i] * [self.computed_label[i]])
        
        raise Exception()
            
        ground_truth, computed = np.hstack(ground_truth), np.hstack(computed)
        correct = np.sum(ground_truth == computed) / len(computed)
        
        print(f"{correct * 100:.2f}% correctly classified [{ground_truth_type}, {len(computed)} samples]")
            
    
    def get_all_3D_points_of_component(self, i):
        """
            Fetches all the individual 3D points of component i in a Nx3 numpy array
            i is a single index
        """
        return np.vstack(self.voxelcloud.get_all_3D_points(self.components[i]))
    
    def get_labels_of_all_3D_points_in_component(self, i):
        """ 
            Fetches all the labels of the individual points of the point cloud for component i
            and groups them in a 1-dimensional numpy array
            i is a single index
        """
        return np.hstack(self.voxelcloud.get_labels_of_3D_points(self.components[i]))
    
    def has_color(self):
        """ 
            Tells whether this cloud of connected components has color information
        """
        return self.voxelcloud.has_color()
    
    def has_laser_intensity(self):
        """
            Tells whether this cloud of connected commonents has laser intensity information
        """
        return self.voxelcloud.has_laser_intensity()
    
    def has_label(self):
        """
            Tells whether this cloud of connected components has labels
        """
        return self.voxelcloud.has_label()
    
    def __len__(self):
        return len(self.components)
        
        