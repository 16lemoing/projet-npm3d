#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
from glob import glob
import pickle
import os
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

# %% Make ply files for clouds

make_ply("../data/other/untermaederbrunnen_station1_xyz_intensity_rgb.txt", "../data/labels/untermaederbrunnen_station1_xyz_intensity_rgb.labels", "../data/original_clouds/untermaederbrunnen1.ply", masked_label=0)

# %% Relabel clouds (merge 2 terrain classes)

original_clouds_folder = "../data/original_clouds"
relabeled_clouds_folder = "../data/relabeled_clouds"
overwrite = False

for ply_file in glob(os.path.join(original_clouds_folder, "*.ply")):
    relabeled_ply_file = os.path.join(relabeled_clouds_folder, os.path.basename(ply_file))
    if overwrite or not os.path.exists(relabeled_ply_file):
        print(f"Relabeling: {ply_file}")
        data = read_ply(ply_file)
        cloud = np.vstack((data['x'], data['y'], data['z'])).T
        rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        dlaser = data['reflectance']
        label = data['label']
        label[label == 1] = 2
        write_ply(relabeled_ply_file, [cloud, dlaser, rgb_colors.astype(np.int32), label.astype(np.int32)], ['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue', 'label'])
        print(f"Done relabeling: {ply_file}")

# %% Extract and backup voxel clouds

relabeled_clouds_folder = "../data/relabeled_clouds"
vc_backup_folder = "../data/backup/voxel_cloud"
overwrite = False

if not os.path.exists(vc_backup_folder):
    os.makedirs(vc_backup_folder)

for ply_file in glob(os.path.join(relabeled_clouds_folder, "*.ply")):
    backup_file = os.path.join(vc_backup_folder, os.path.basename(ply_file).replace("ply", "pkl"))
    if overwrite or not os.path.exists(backup_file):
        print(f"Processing: {ply_file}")
        
        # Retrieve data
        data = read_ply(ply_file)
        cloud = np.vstack((data['x'], data['y'], data['z'])).T
        rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        dlaser = data['reflectance']
        label = data['label'] if "label" in data.dtype.names else None
        
        # Define clouds
        pc = PointCloud(cloud, dlaser, rgb_colors, label)
        vc = VoxelCloud(pc, max_voxel_size = 0.3, threshold_grow = 2, min_voxel_length = 5, method = "regular")
        
        # Save voxel cloud
        with open(backup_file, 'wb') as handle:
            pickle.dump(vc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done processing: {ply_file}\n")
    
# %% Extract and backup component clouds

vc_backup_folder = "../data/backup/voxel_cloud"
cc_backup_folder = "../data/backup/component_cloud"
overwrite = False

if not os.path.exists(cc_backup_folder):
    os.makedirs(cc_backup_folder)

for pkl_file in glob(os.path.join(vc_backup_folder, "*.pkl")):
    backup_file = os.path.join(cc_backup_folder, os.path.basename(pkl_file))
    if overwrite or not os.path.exists(backup_file):
        print(f"Making component cloud for: {pkl_file}")
        
        # Retrieve voxel clouod
        with open(pkl_file, 'rb') as handle:
            vc = pickle.load(handle)
        
        # Compute component cloud
        cc = ComponentCloud(vc, c_D = 0.25, segment_out_ground=True, min_component_length=5)
        
        # Save voxel cloud
        with open(backup_file, 'wb') as handle:
            pickle.dump(cc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done making component cloud for: {pkl_file}\n")

# %% Classify components

cc_backup_folder = "../data/backup/component_cloud"
pc_backup_folder = "../data/backup/predicted_cloud"
train_cc_files = ["bildstein3.pkl", "domfountain2.pkl"]
test_cc_files = ["bildstein5.pkl", "neugasse.pkl", "untermaederbrunnen1.pkl"]

# Load train data
print("Loading train data")
train_cc = []
for filename in train_cc_files:
    pkl_file = os.path.join(cc_backup_folder, filename)
    with open(pkl_file, 'rb') as handle:
        cc = pickle.load(handle)
        train_cc.append(cc)

# Load test data
print("Loading test data")
test_cc = []
for filename in test_cc_files:
    pkl_file = os.path.join(cc_backup_folder, filename)
    with open(pkl_file, 'rb') as handle:
        cc = pickle.load(handle)
        test_cc.append(cc)

# Train classifier on train data
print("Training classifier")
cc_classifier = ComponentClassifier('random_forest', {'n_estimators': 20})
cc_classifier.fit(train_cc)

# Evaluate classifier on train data
for i, cc in enumerate(train_cc):
    print(f"Evaluation of [TRAIN DATA] {train_cc_files[i]}")
    cc.set_predicted_labels(cc_classifier.predict(cc))
    cc.eval_classification_error(ground_truth_type = "pointwise")
    cc.eval_classification_error(ground_truth_type = "pointwise", include_unassociated_points=True)
    cc.eval_classification_error(ground_truth_type = "componentwise")
    
# Evaluate classifier on test data
for i, cc in enumerate(test_cc):
    print(f"Evaluation of [TEST DATA] {test_cc_files[i]}")
    cc.set_predicted_labels(cc_classifier.predict(cc))
    cc.eval_classification_error(ground_truth_type = "pointwise")
    cc.eval_classification_error(ground_truth_type = "pointwise", include_unassociated_points=True)
    cc.eval_classification_error(ground_truth_type = "componentwise")

# Save train results (predicted components, predicted labels, groundtruth labels)
print("Saving train results")
for i, cc in enumerate(train_cc):
    ply_file = os.path.join(pc_backup_folder, 'train_' + train_cc_files[i]).replace('pkl', 'ply')
    cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
    component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
    predicted_label = np.hstack([cc.predicted_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    write_ply(ply_file, [cloud_point, component, predicted_label, groundtruth_label], ['x', 'y', 'z', 'predicted_component', 'predicted_label', 'groundtruth_label'])
    
# Save test results (predicted components, predicted labels, groundtruth labels)
print("Saving test results")
for i, cc in enumerate(test_cc):
    ply_file = os.path.join(pc_backup_folder, 'test_' + test_cc_files[i]).replace('pkl', 'ply')
    cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
    component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
    predicted_label = np.hstack([cc.predicted_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    write_ply(ply_file, [cloud_point, component, predicted_label, groundtruth_label], ['x', 'y', 'z', 'predicted_component', 'predicted_label', 'groundtruth_label'])
    
print("Done")

##################
#    Old main    #
##################

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

# a, b = vc.remove_poorly_connected_voxels(0.25, 10)
# print(f"Nombre de voxels supprimés après exploration du graphe car mal connectés : {a}")

#plot(vc, colors = vc.mean_color, only_voxel_center = True)
# %%
#vc.are_neighbours(1, [2,3])
#vc.find_neighbours([1, 4677, 2920])

# %% Saving voxel cloud for later use

with open('../data/bildstein_station3_xyz_intensity_rgb_labeled_vc.pkl', 'wb') as handle:
    pickle.dump(vc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# %% Loading voxel cloud
import pickle
with open('../data/bildstein_station3_xyz_intensity_rgb_labeled_vc.pkl', 'rb') as handle:
    vc = pickle.load(handle)
# with open('../data/bildstein_station3_xyz_intensity_rgb_labeled_vc.pkl', 'rb') as handle:
#     vc2 = pickle.load(handle)

# %% Display voxels
#plot(vc, only_voxel_center = False)
#plot(vc, colors = vc.mean_intensity, only_voxel_center = False, also_unassociated_points = True)
plot(vc, colors = None, only_voxel_center = False, also_unassociated_points = True, also_removed_points = True)

# %% Compute components and display them
cc = ComponentCloud(vc, c_D = 0.25, segment_out_ground=True, min_component_length=5)
# cc2 = ComponentCloud(vc2, c_D = 0.25)
#plot(cc, colors = None, only_voxel_center = True, also_unassociated_points = False)
#cc.eval_classification_error()

# %%

cc.set_predicted_labels(cc.get_labels())
cc.eval_classification_error(ground_truth_type = "pointwise")
cc.eval_classification_error(ground_truth_type = "componentwise")

# %%
import random 



# %% Classify components
classifier = Classifier('random_forest', {'n_estimators': 20})
classifier.fit(cc)

cc2.set_predicted_labels(cc2.get_labels())
cc2.eval_classification_error(ground_truth_type = "pointwise")
cc2.eval_classification_error(ground_truth_type = "componentwise")

cc2.set_predicted_labels(classifier.predict(cc2))
cc2.eval_classification_error(ground_truth_type = "pointwise")
cc2.eval_classification_error(ground_truth_type = "componentwise")
