# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
def plot_voxels(voxelcloud, idxs = None, colors = None, only_voxel_center = False):
    """
    Plots some voxels from a voxel cloud
    """
    
    if idxs is None:
        idxs = list(range(len(voxelcloud)))
    
    if type(idxs) is int:
        idxs = [idxs]
        
    if colors is not None:
        if len(colors) != len(idxs):
            raise Exception("Colors must have the same shape as idxs")
        colors = cm.jet(colors / np.max(colors))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(idxs)):
        if only_voxel_center:
            data = voxelcloud.geometric_center[[i], :]
        else:
            data = voxelcloud.pointcloud.get_coordinates(voxelcloud.voxels[i])
        
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': colors[i]} if colors is not None else {}))
    
    plt.show()
    
    
def plot_components(componentcloud, idxs = None, colors = None, only_voxel_center = True):
    """
    Plots some components from a component cloud
    """
    
    if idxs is None:
        idxs = list(range(len(componentcloud)))
    
    if type(idxs) is int:
        idxs = [idxs]
    
    if colors is not None:
        if len(colors) != len(idxs):
            raise Exception("Colors must have the same shape as idxs")
        colors = cm.jet(colors / np.max(colors))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(idxs)):
        if only_voxel_center:
            data = componentcloud.voxelcloud.geometric_center[componentcloud.components[i], :]
        else:
            data = componentcloud.get_all_3D_points_of_component(i)
            
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': colors[i]} if colors is not None else {}))
    
    plt.show()

