#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import time
import numpy as np
from utils.ply import read_ply, write_ply
from plots import plot_components, plot_voxels
from pointcloud import PointCloud
from voxelcloud import VoxelCloud
from componentcloud import ComponentCloud

# %%
## Preparing data

# data = read_ply("../data/bildstein_station5_xyz_intensity_rgb.ply")
# cloud = np.vstack((data['x'], data['y'], data['z'])).T
# rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
# dlaser = data['reflectance']
# print(cloud.shape)
# print(rgb_colors.shape)
# print(dlaser.shape)

# # Select portion of data
# mask = ((cloud[:,0] > 9) & (cloud[:,0] < 17) & (cloud[:,1] > -51) & (cloud[:,1] < -31))
# extracted_cloud = cloud[mask]
# extracted_rgb_colors = rgb_colors[mask]
# extracted_dlaser = dlaser[mask]
# write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract.ply', [extracted_cloud, extracted_dlaser, extracted_rgb_colors],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])


# # Reduce number of points
# decimation = 10
# limit = 1000000
# write_ply('../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply', [extracted_cloud[:limit:decimation], extracted_dlaser[:limit:decimation], extracted_rgb_colors[:limit:decimation]],['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue'])

### Test script

# cloud_data = read_ply("../data/bildstein_station5_xyz_intensity_rgb.ply")
# label_file = "../data/labels/bildstein_station5_xyz_intensity_rgb.labels"
# points_labels = np.loadtxt(label_file)
# cloud_data_2 = np.vstack((cloud_data['x'], cloud_data['y'], cloud_data['z'], cloud_data["red"], cloud_data["green"], cloud_data["blue"], cloud_data["reflectance"])).T
# mask = ((cloud_data_2[:,0] > 9) & (cloud_data_2[:,0] < 17) & (cloud_data_2[:,1] > -51) & (cloud_data_2[:,1] < -31)) & (points_labels > 0)

# write_ply('../data/bildstein_station5_xyz_intensity_rgb_test_extract.ply', [cloud_data_2[mask,:3], cloud_data_2[mask,-1], cloud_data_2[mask,3:6].astype(np.int32), points_labels[mask].astype(np.int32)], ['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue', 'label'])

# %%
## Retrieve data

data = read_ply("../data/bildstein_station5_xyz_intensity_rgb_test_extract.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']

# %%
## Defining cloud and computing voxels and features
pc = PointCloud(cloud, dlaser, rgb_colors)

vc = VoxelCloud(pc)
vc.are_neighbours(1, [1,3])
vc.find_neighbours([1, 4677, 2920])

# %%
## Display voxels
v_mi = vc.mean_intensity
normalized_v_mi = (v_mi - np.min(v_mi)) / np.ptp(v_mi)
plot_voxels(vc, colors = normalized_v_mi, only_voxel_center = False)

##
# %% Compute components
cc = ComponentCloud(vc)
plot_components(cc, only_voxel_center = False)
