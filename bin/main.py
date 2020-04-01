#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
import time
import matplotlib.pyplot as plt
import numpy as np
from utils.ply import read_ply, write_ply, make_ply
from plots import plot
from pointcloud import PointCloud
from voxelcloud import VoxelCloud
from componentcloud import ComponentCloud
from classifiers import Classifier

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

make_ply("../data/bildstein_station3_xyz_intensity_rgb.txt", "../data/labels/bildstein_station3_xyz_intensity_rgb.labels", "../data/bildstein_station3_xyz_intensity_rgb_labeled.ply", masked_label=0)


# %% Retrieve data

data = read_ply("../data/bildstein_station3_xyz_intensity_rgb_labeled.ply")
cloud = np.vstack((data['x'], data['y'], data['z'])).T
rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
dlaser = data['reflectance']
label = data['label'] if "label" in data.dtype.names else None

# %% Defining cloud and computing voxels and features
pc = PointCloud(cloud, dlaser, rgb_colors, label)
vc = VoxelCloud(pc, max_voxel_size = 0.3, threshold_grow = 2, min_voxel_length = 3, method = "regular")
print(f"Nombre de voxels trop petits non associés à des gros voxels : {len(vc.unassociated_too_small_voxels)}")
print(f"Nombre de gros voxels : {len(vc.voxels)}")

a, b = vc.remove_poorly_connected_voxels(0.25, 10)
print(f"Nombre de voxels supprimés après exploration du graphe car mal connectés : {a}")

#plot(vc, colors = vc.mean_color, only_voxel_center = True)
# %%
#vc.are_neighbours(1, [2,3])
#vc.find_neighbours([1, 4677, 2920])

# %% Saving voxel cloud for later use
import pickle
with open('../data/bildstein_station3_xyz_intensity_rgb_labeled_vc.pkl', 'wb') as handle:
    pickle.dump(vc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# %% Loading voxel cloud
import pickle
with open('../data/bildstein_station5_xyz_intensity_rgb_labeled_vc.pkl', 'rb') as handle:
    vc = pickle.load(handle)
with open('../data/bildstein_station3_xyz_intensity_rgb_labeled_vc.pkl', 'rb') as handle:
    vc2 = pickle.load(handle)

# %% Display voxels
#plot(vc, only_voxel_center = False)
#plot(vc, colors = vc.mean_intensity, only_voxel_center = False, also_unassociated_points = True)
plot(vc, colors = None, only_voxel_center = False, also_unassociated_points = True, also_removed_points = True)

# %% Compute components and display them
cc = ComponentCloud(vc, c_D = 0.25)
cc2 = ComponentCloud(vc2, c_D = 0.25)
#plot(cc, colors = None, only_voxel_center = True, also_unassociated_points = False)
#cc.eval_classification_error()


# %% Classify components
classifier = Classifier('random_forest', {'n_estimators': 20})
classifier.fit(cc)

cc2.set_predicted_labels(cc2.get_labels())
cc2.eval_classification_error(ground_truth_type = "pointwise")
cc2.eval_classification_error(ground_truth_type = "componentwise")

cc2.set_predicted_labels(classifier.predict(cc2))
cc2.eval_classification_error(ground_truth_type = "pointwise")
cc2.eval_classification_error(ground_truth_type = "componentwise")
