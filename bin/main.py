#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
import pickle
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from utils.ply import read_ply, write_ply, make_ply
from plots import plot
from pointcloud import PointCloud
from voxelcloud import VoxelCloud
from componentcloud import ComponentCloud
from classifiers import ComponentClassifier

# Parameters
name = "bildstein5_extract.ply"
ply_file = Path("..") / "data" / "relabeled_clouds" / name
test_backup_folder = Path("..") / "data" / "backup" / "test"
test_backup_folder.mkdir(exist_ok = True)

c_D = 0.25
segment_out_ground = True
min_component_length = 5
method = "spectral"
K = 500

data = read_ply(ply_file)

pc = PointCloud(
    points = np.vstack((data['x'], data['y'], data['z'])).T, 
    laser_intensity = data['reflectance'],
    rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T,
    label = data['label']
)

vc = VoxelCloud(
    pointcloud = pc,
    max_voxel_size = 0.3,
    min_voxel_length = 4,
    threshold_grow = 1.5,
    method = "regular"
)

plot(vc)

cc = ComponentCloud(
    voxelcloud = vc, 
    c_D = 0.25,
    segment_out_ground = True,
    method = "normal",
    K = 1,
    min_component_length = 1
)

plot(cc, colors = np.array([0] + (len(cc) - 1) * [1]))

# Save component cloud
print("Saving component cloud for display")
ply_file = os.path.join(test_backup_folder, os.path.basename(pkl_file)).replace('pkl', 'ply')
cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
groundtruth_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
write_ply(ply_file, [cloud_point, component, groundtruth_label], ['x', 'y', 'z', 'predicted_component', 'groundtruth_label'])
