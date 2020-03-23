# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def show_points(cloud, idxs, color=None):
    """
    3D scatter plot of points, grouped by voxel 

    Parameters
    ----------
    cloud : Nx3 cloud of all points
    idxs : list of indexes, of size n_voxels. Each element is a list of the 
           indices of the points of 'clouds' which belong to the corresponding voxel
    color : numpy array of floats of size n_voxels, optional
        If provided, it is used to displauy each voxel with a specific color
    """
    
    if color is not None:
        color = cm.jet(color / np.max(color))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(idxs)):
        data = cloud[idxs[i], :]
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': color[i]} if color is not None else {}))
    
    plt.show()
    


