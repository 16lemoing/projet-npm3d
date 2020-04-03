# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:16:52 2020

@author: Hugues
"""

import numpy as np
from descriptors import local_PCA
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix

class ComponentCloud:
    
    def __init__(self, voxelcloud, c_D = 0.25, method = "normal", K = 15, segment_out_ground = False, threshold_in = 1, threshold_normals = 0.8, min_component_length = 1):
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
        
        if method == "spectral":
            self.components = self.voxelcloud.find_connected_components_similarity(self.c_D, weights = [1,1,1], K = K)
            self.too_small_components = []
        else:
            self.components, self.too_small_components = self.voxelcloud.compute_connected_components(self.c_D, segment_out_ground, threshold_in, threshold_normals, min_component_length)
        
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
        self.compute_features() # Computes features
        
        # Initialize and declare labels
        self.majority_label = np.nan * np.ones(len(self), dtype=int)
        self.compute_labels() # Computes labels
        
        # Initialize predicted labels
        self.predicted_label = np.nan * np.ones(len(self), dtype=int)
            
    
    def compute_features(self):
        """
            Computes the individual features of each connected component
        """
        
        for i in range(len(self)):
            vx_nb_points = self.voxelcloud.features['nb_points'][self.components[i]]
            vx_barycenters = self.voxelcloud.features['barycenter'][self.components[i]]
            vx_colors = self.voxelcloud.features['mean_color'][self.components[i]]
            vx_intensities = self.voxelcloud.features['mean_intensity'][self.components[i]]
            vx_nb_points = self.voxelcloud.features['nb_points'][self.components[i]]
            vx_geometric_centers = self.voxelcloud.features['geometric_center'][self.components[i]]
            vx_sizes = self.voxelcloud.features['size'][self.components[i]]
            vx_normals = self.voxelcloud.features['normal'][self.components[i]]
            vx_verticalities = self.voxelcloud.features['verticality'][self.components[i]]
            vx_linerities = self.voxelcloud.features['linearity'][self.components[i]]
            vx_planarities = self.voxelcloud.features['planarity'][self.components[i]]
            vx_sphericities = self.voxelcloud.features['sphericity'][self.components[i]]
            
            self.nb_points[i] = np.sum(vx_nb_points)
            self.nb_voxels[i] = len(self.components[i])
            self.barycenter[i, :] = np.sum(vx_barycenters * vx_nb_points[:, None], axis=0) / np.sum(vx_nb_points)
            p_max = np.max(vx_geometric_centers + vx_sizes / 2, axis=0)
            p_min = np.min(vx_geometric_centers + vx_sizes / 2, axis=0)
            self.geometrical_center[i, :] = (p_max + p_min) / 2
            self.size[i, :] = p_max - p_min
            if self.has_color():
                self.mean_color[i, :] = np.sum(vx_colors * vx_nb_points[:, None], axis = 0) / np.sum(vx_nb_points)
                self.var_color[i, :] = np.sum((vx_colors - self.mean_color[i, :]) ** 2 * vx_nb_points[:, None], axis = 0) / np.sum(vx_nb_points)
            if self.has_laser_intensity():
                self.mean_intensity[i] = np.sum(vx_intensities * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
                self.var_intensity[i] = np.sum((vx_intensities - self.mean_intensity[i]) ** 2 * vx_nb_points, axis = 0) / np.sum(vx_nb_points)
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

    def compute_labels(self):
        """
            Computes the individual label of each connected component
        """
        
        if self.has_label():
            for i in range(len(self)):
                vx_labels = self.voxelcloud.features['majority_label'][self.components[i]]
                vx_nb_points = self.voxelcloud.features['nb_points'][self.components[i]]
                
                counts = np.bincount(vx_labels.astype(int), weights=vx_nb_points)
                self.majority_label[i] = np.argmax(counts)

    def get_features(self):
        """
            Computes the features necessary for training the component classifier
        """
        return np.hstack((self.nb_points[:, None],
                          self.nb_voxels[:, None],
                          self.barycenter,
                          self.geometrical_center,
                          self.mean_znormal[:, None],
                          self.var_znormal[:, None],
                          self.mean_xynormal[:, None],
                          self.var_xynormal[:, None],
                          (self.mean_intensity[:, None] if self.has_laser_intensity() else np.empty((len(self),0))),
                          (self.var_intensity[:, None] if self.has_laser_intensity() else np.empty((len(self),0))),
                          (self.mean_color if self.has_color() else np.empty((len(self),0))),
                          (self.var_color if self.has_color() else np.empty((len(self),0))),
                          self.pca_val,
                          self.pca_vec,
                          self.size,
                          self.mean_verticality[:, None],
                          self.mean_linearity[:, None],
                          self.mean_planarity[:, None],
                          self.mean_sphericity[:, None]))
        
    def get_labels(self):
        """
            Retrieves the labels necessary for training the component classifier
        """
        return self.majority_label
    
    def set_predicted_labels(self, predicted_label):
        """
            Assigns the component label values based on the trained classifier
        """
        self.predicted_label = predicted_label
    
    def eval_classification_error(self, idx = None, ground_truth_type = "pointwise", include_unassociated_points = False, classes = None):
        """
            Computes the classification error for one ore several components in the cloud
            and returns the confusion matrix if classes are provided
        """
        if type(idx) is int:
            idx = [idx]
        if idx is None:
            idx = list(range(len(self)))
            
        ground_truth = []
        predicted = []
        
        # Get ground truth and predicted labels for components
        for i in idx:
            if ground_truth_type == "pointwise":
                ground_truth.append(self.get_labels_of_all_3D_points_in_component(i))
            elif ground_truth_type == "componentwise":
                ground_truth.append(self.nb_points[i] * [self.majority_label[i]])
            else:
                raise NotImplementedError()
            
            predicted.append(self.nb_points[i] * [self.predicted_label[i]])
        
        # Get ground truth and predicted labels for all unassociated points
        if ground_truth_type == "pointwise" and include_unassociated_points:
            
            # Identify all voxels that are part of a component
            associated_voxels_mask = np.array(list(range(len(self.voxelcloud))))
            for component in self.too_small_components:
                for voxel in component:
                    associated_voxels_mask[voxel] = False
                    
            # Get predicted label for all associated voxels
            voxel_predicted_label = np.ones(len(self.voxelcloud))
            for i, voxels in enumerate(self.components):
                voxel_predicted_label[voxels] = self.predicted_label[i]
            voxel_predicted_label = voxel_predicted_label[associated_voxels_mask]
            
            # Build a KDTree from their geometric center
            gc = self.voxelcloud.features['geometric_center'][associated_voxels_mask]
            kdtc = KDTree(gc)
            
            # Retrieve all unassociated points (from isolated voxels and components)
            pts = self.get_all_unassociated_3D_points()
            
            # Find the nearest associated voxel
            _, ctr = kdtc.query(pts, k = 1)
            ctr = ctr[:, 0]
            
            # Transfer predicted labels to unassociated points
            predicted.append(voxel_predicted_label[ctr])
            
            # Retrieve ground truth labels for unassociated points
            unassociated_points_idxes = self.get_all_unassociated_3D_points_idxes()
            ground_truth.append(self.voxelcloud.pointcloud.get_label(unassociated_points_idxes))
            
        ground_truth, predicted = np.hstack(ground_truth), np.hstack(predicted)
        correct = np.sum(ground_truth == predicted) / len(predicted)
        
        print(f"{correct * 100:.2f}% correctly classified [{ground_truth_type}, {len(predicted)} samples {'(including unassociated points)' if include_unassociated_points else ''}]")
        
        if classes is not None:
            return confusion_matrix(ground_truth, predicted, classes)
            
    
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
    
    def get_all_unassociated_3D_points(self):
        """
            Fetches all the 3D points of voxels which were too small and that we weren't
            able to associate to any bigger voxel and also 3D points of voxels which
            were part of components too small
        """
        results = []
        for i in self.voxelcloud.too_small_voxels:
            results.append(self.voxelcloud.pointcloud.get_coordinates(i))
        for voxels in self.too_small_components:
            for i in voxels:
                results.append(self.voxelcloud.pointcloud.get_coordinates(i))
        return np.vstack(results) if len(results) > 0 else np.array([])

    def get_all_unassociated_3D_points_idxes(self):
        """
            Fetches all the 3D points idxes of voxels which were too small and that 
            we weren't able to associate to any bigger voxel and also 3D points of
            voxels which were part of components too small
        """
        results = []
        for i in self.voxelcloud.too_small_voxels:
            results.append(i)
        for voxels in self.too_small_components:
            for i in voxels:
                results.append(i)
        return np.hstack(results).astype(int) if len(results) > 0 else np.array([])
    
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
        
        