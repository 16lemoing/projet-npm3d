# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KDTree


class VoxelCloud:
    def __init__(self, cloud, c_D = 0.25):
        self.cloud = cloud
        self.voxels = cloud.s_voxels
        self.kdt = KDTree(self.voxels.loc[:, "geometric_center"])
        self.c_D = c_D
        
        # For faster access than accessing the pandas dataframe each time...
        self.s_geometric_center = self.voxels["geometric_center"].values
        self.s_size = self.voxels["size"].values
        self.s_var_intensity = self.voxels["var_intensity"].values
        self.s_mean_intensity = self.voxels["mean_intensity"].values
        self.s_var_color = self.voxels["var_color"].values
        self.s_mean_color = self.voxels["mean_color"].values
        
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
        
        gc_target = self.s_geometric_center[i, :]
        gc_candidates = self.s_geometric_center[j, :]
        size_target = self.s_size[i, :]
        size_candidates = self.s_size[j, :]
        vi_target = self.s_var_intensity[i, 0]
        vi_candidates = self.s_var_intensity[j, 0]
        mi_target = self.s_mean_intensity[i, 0]
        mi_candidates = self.s_mean_intensity[j, 0]
        vc_target = self.s_var_color[i, :]
        vc_candidates = self.s_var_color[j, :]
        mc_target = self.s_mean_color[i, :]
        mc_candidates = self.s_mean_color[j, :]
        
        w_D = (size_target + size_candidates) / 2
        cond_D = np.all(abs(gc_target - gc_candidates) <= w_D + self.c_D, axis=1)
        
        cond_I = np.ones(len(j), dtype=bool)
        if self.cloud.laser_intensity is not None:
            w_I = np.maximum(vi_target, vi_candidates)
            cond_I = abs(mi_target - mi_candidates) <= 3 * np.sqrt(w_I)
            
        cond_C = np.ones(len(j), dtype=bool)
        if self.cloud.rgb_colors is not None:
            w_C = np.maximum(vc_target, vc_candidates)
            cond_C = np.all(abs(mc_target - mc_candidates) <= 3 * np.sqrt(w_C), axis=1)
        
        cond = cond_D & cond_I & cond_C
        if isnum:
            return cond[0]
        
        return cond
    
    def find_spatial_neighbours(self, idxs):
        """ Returns a list of indices of potential neighbouring voxels """
        max_size = max(np.max(self.voxels.loc[:, "size"]))        
        return self.kdt.query_radius(self.voxels.loc[np.atleast_1d(idxs), "geometric_center"], r = max_size + self.c_D)
        
    
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
        
        
    def compute_connected_components(self):
        """
        Builds a list of connected components of voxels from a list of voxels
        """
        
        n_voxels = len(self.voxels)
        voxel_neighbours = self.find_neighbours(list(range(n_voxels)))
        
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
        
    
    def plot_voxel_points(self, ax, i, c = None):
        data = self.cloud.points[self.cloud.voxels[i], :]
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': c} if c is not None else {}))
        
    def plot_voxel_geometric_center(self, ax, i, c = None):
        data = self.s_geometric_center[[i], :]
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': c} if c is not None else {}))
        
    
    def __len__(self):
        return len(self.voxels)