# -*- coding: utf-8 -*-

from sklearn.neighbors import KDTree
import numpy as np

class PointCloud:
    
    def __init__(self, points, laser_intensity = None, rgb_colors = None):
        self.points = points
        self.laser_intensity = laser_intensity
        self.rgb_colors = rgb_colors
        self.kdt = KDTree(self.points) # Can take a few seconds to build
    
    def get_coordinates(self, idxs = None):
        """
        Returns a Nx3 numpy array containing the 3D coordinates of the points given as input by their indices
        """
        if idxs is None:
            return self.points[:, :]
        else:
            return self.points[idxs, :]
    
    def has_laser_intensity(self):
        """
        Tells whether the point cloud has laser intensity information
        """
        return self.laser_intensity is not None
    
    def get_laser_intensity(self, idxs = None):
        """
        Returns a N-numpy array containing the laser intensities of the points given as input by their indices
        """
        if self.laser_intensity is None:
            return np.nan * (1 if type(idxs) is int else (np.ones(len(idxs) if idxs is not None else len(self))))
        if idxs is None:
            return self.laser_intensity
        else:
            return self.laser_intensity[idxs]
    
    def has_color(self):
        """
        Tells whether the point cloud has rgb color information
        """
        return self.rgb_colors is not None
    
    def get_color(self, idxs = None):
        """
        Returns a Nx3-numpy array containing the rgb colors of the points given as input by their indices
        """
        if self.rgb_colors is None:
            return np.nan * np.ones((1 if type(idxs) is int else (len(idxs) if idxs is not None else len(self)), 3))
        if idxs is None:
            return self.rgb_colors
        else:
            return self.rgb_colors[idxs, :]
        
    def get_r_nn(self, idxs, r):
        return self.kdt.query_radius(self.get_coordinates(idxs), r = r)
    
    def __len__(self):
        return len(self.points)