#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import time
import numpy as np
from utils.ply import read_ply, write_ply
from plots import plot_components, plot_voxels
from cloud import Cloud
from voxelcloud import VoxelCloud
from component import Component

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

# %%
## Retrieve data

data = read_ply("../data/bildstein_station5_xyz_intensity_rgb_extract_small.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']

# %%
## Defining cloud and computing voxels and features
c = Cloud(cloud, dlaser, rgb_colors)
c.compute_voxels()
# access with c.voxels, c.voxels_mask
c.compute_voxels_features()
# access with c.s_voxels

vc = VoxelCloud(c)
vc.are_neighbours(1, 1)

# %%
## Display voxels
v_mi = c.s_voxels.loc[:, "mean_intensity"].values[:,0]
normalized_v_mi = (v_mi - np.min(v_mi)) / np.ptp(v_mi)
plot_voxels(vc, colors = normalized_v_mi)

##
# %% Compute components
vc.compute_connected_components()

cps = []
for i in range(len(vc.components)):
    cps.append(Component(vc, i))
    
plot_components(cps)
