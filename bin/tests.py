#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils.ply import read_ply, write_ply, make_ply
from plots import plot
from pointcloud import PointCloud
from voxelcloud import VoxelCloud
from componentcloud import ComponentCloud


# Parameters
name = "bildstein5_extract.ply"
ply_file = Path("..") / "data" / "relabeled_clouds" / name


c_D = 0.25
segment_out_ground = True

data = read_ply(ply_file)

# Creating point cloud
pc = PointCloud(
    points = np.vstack((data['x'], data['y'], data['z'])).T, 
    laser_intensity = data['reflectance'],
    rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T,
    label = data['label']
)


# Creating voxel cloud
vc = VoxelCloud(
    pointcloud = pc,
    max_voxel_size = 0.3,
    min_voxel_length = 4,
    threshold_grow = 1.5,
    method = "regular"
)
vc.remove_poorly_connected_voxels(c_D = 0.25, threshold_nb_voxels = 10)

# Plotting for inspection
c = vc.features['mean_color']
plot(vc, colors = c, also_unassociated_points = True, only_voxel_center = False)

# For spectral clustering : creating the graph and plotting it with the appropriate color
A, D = vc.build_similarity_graph(0.25, [1,1,1], sparse_matrix=False)
g = nx.from_numpy_matrix(A)
nx.draw(g, node_color = np.hstack((c/255, np.ones((len(c), 1)))))

# Component cloud with objects separated one from another
cc = ComponentCloud(
    voxelcloud = vc, 
    c_D = 0.25,
    segment_out_ground = True,
    method = "normal",
    min_component_length = 1,
    threshold_in = 0.2,
    threshold_normals = 0.8
)

# Plotting for inspection
plot(cc, only_voxel_center = False)

# For classification, please refer to the pipeline.py file, 
# as learning is not performed in this test file