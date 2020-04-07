# -*- coding: utf-8 -*-

import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
# def plot_voxels(voxelcloud, idxs = None, colors = None, only_voxel_center = False, also_unassociated_points = False):
#     """
#     Plots some voxels from a voxel cloud
#     """
    
#     if idxs is None:
#         idxs = list(range(len(voxelcloud)))
    
#     if type(idxs) is int:
#         idxs = [idxs]
        
#     if colors is not None:
#         if len(colors) != len(idxs):
#             raise Exception("Colors must have the same shape as idxs")
#         if colors.shape[1] not in [3,4]:
#             colors = cm.jet(colors / np.max(colors))
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for i in range(len(idxs)):
#         if only_voxel_center:
#             data = voxelcloud.geometric_center[[i], :]
#         else:
#             data = voxelcloud.pointcloud.get_coordinates(voxelcloud.voxels[i])
        
#         ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': colors[i]} if colors is not None else {}))
    
#     if also_unassociated_points:
#         data = voxelcloud.get_all_unassociated_3D_points()
#         ax.plot(data[:,0], data[:,1], data[:,2], '+', c='black')
    
#     plt.show()
    
    
# def plot_components(componentcloud, idxs = None, colors = None, only_voxel_center = True, also_unassociated_points = False):
#     """
#     Plots some components from a component cloud
#     """
    
#     if idxs is None:
#         idxs = list(range(len(componentcloud)))
    
#     if type(idxs) is int:
#         idxs = [idxs]
    
#     if colors is not None:
#         if len(colors) != len(idxs):
#             raise Exception("Colors must have the same shape as idxs")
#         if colors.shape[1] not in [3,4]:
#             colors = cm.jet(colors / np.max(colors))
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for i in range(len(idxs)):
#         if only_voxel_center:
#             data = componentcloud.voxelcloud.geometric_center[componentcloud.components[i], :]
#         else:
#             data = componentcloud.get_all_3D_points_of_component(i)
            
#         ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': colors[i]} if colors is not None else {}))
    
#     if also_unassociated_points:
#         data = componentcloud.voxelcloud.get_all_unassociated_3D_points()
#         ax.plot(data[:,0], data[:,1], data[:,2], '+', c='black')
    
#     plt.show()


def plot(cloud, idxs = None, colors = None, only_voxel_center = True, also_unassociated_points = False, also_removed_points = False, figsize = None):
    
    if idxs is None:
        idxs = list(range(len(cloud)))
    
    if type(idxs) is int:
        idxs = [idxs]
    
    if colors is not None:
        if len(colors) != len(idxs):
            raise Exception("Colors must have the same shape as idxs")
        if len(colors.shape) == 1 or colors.shape[1] == 1:
            colors = cm.jet((colors - np.min(colors)) / (np.max(colors) - np.min(colors)))
        elif colors.shape[1] in [3, 4] and (np.max(colors) > 1 or np.min(colors) < 0):
            colors = colors / 255.0    
    
    if hasattr(cloud, "voxelcloud"):
        cloudtype = "component"
    else:
        cloudtype = "voxel"
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(idxs)):
        if only_voxel_center:
            if cloudtype == "component":
                data = cloud.voxelcloud.features['geometric_center'][cloud.components[i], :]
            else:
                data = cloud.features['geometric_center'][[i], :]
        else:
            if cloudtype == "component":
                data = cloud.get_all_3D_points_of_component(i)
            else:
                data = cloud.pointcloud.get_coordinates(cloud.voxels[i])
            
        ax.plot(data[:,0], data[:,1], data[:,2], '.', **({'c': colors[i]} if colors is not None else {}))
    
    if also_unassociated_points:
        if cloudtype == "component":
            data = cloud.voxelcloud.get_all_unassociated_3D_points()
        else:
            data = cloud.get_all_unassociated_3D_points()
        if len(data) > 0:
            ax.plot(data[:,0], data[:,1], data[:,2], '+', c='black')
        
    if also_removed_points:
        if cloudtype == "component":
            data = cloud.voxelcloud.get_all_removed_3D_points()
        else:
            data = cloud.get_all_removed_3D_points()
        if len(data) > 0:
            ax.plot(data[:,0], data[:,1], data[:,2], 'o', c='black')
    
    plt.show()

def plot_confusion_matrix(cm, classes, data_type, id, folder):
    
    tot_gt = np.sum(cm, axis=1)
    n_cm = (cm.T / tot_gt).T
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(n_cm)
    plt.title(f"Confusion matrix for {data_type}")
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.set_ticks_position('bottom')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{int(cm[i, j])}",
                horizontalalignment="center",
                color="white" if n_cm[i,j] < 0.5 else "black",
                fontsize=6)
    plt.tight_layout()
    plt.savefig(str(folder / f"{id}_{data_type}.png"))