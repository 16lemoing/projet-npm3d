# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
def plot_voxels(voxelcloud, idxs = None, colors = None, only_center = False):
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
        if only_center:
            voxelcloud.plot_voxel_geometric_center(ax, idxs[i], **({'c': colors[i]} if colors is not None else {}))
        else:
            voxelcloud.plot_voxel_points(ax, idxs[i], **({'c': colors[i]} if colors is not None else {}))
    
    plt.show()
    
    
def plot_components(list_of_components, colors = None, only_voxel_center = True):
    """
    Plots some components from a list of components objects
    """
    
    if colors is not None:
        if len(colors) != len(list_of_components):
            raise Exception("Colors must have the same shape as the list of components")
        colors = cm.jet(colors / np.max(colors))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in list_of_components:
        if only_voxel_center:
            c.plot_component_voxel_centers(ax, **({'c': colors[i]} if colors is not None else {}))
        else:
            c.plot_component_points(ax, **({'c': colors[i]} if colors is not None else {}))
    plt.show()

