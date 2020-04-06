#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
from pathlib import Path
import pickle
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from utils.ply import read_ply, write_ply, make_ply
from plots import plot, plot_confusion_matrix
from pointcloud import PointCloud
from voxelcloud import VoxelCloud
from componentcloud import ComponentCloud
from classifiers import NeighbourhoodClassifier, ComponentClassifier

# %% Make ply files for clouds

make_ply("../data/other/bildstein_station1_xyz_intensity_rgb.txt", "../data/labels/bildstein_station1_xyz_intensity_rgb.labels", "../data/original_clouds/domfountain1.ply", masked_label=0)

# %% Relabel clouds (merge 2 terrain classes)

# Parameters
original_clouds_folder = Path("..") / "data" / "original_clouds"
relabeled_clouds_folder = Path("..") / "data" / "relabeled_clouds"
overwrite = False

if not relabeled_clouds_folder.exists():
    relabeled_clouds_folder.mkdir()

for ply_file in original_clouds_folder.glob("*.ply"):
    relabeled_ply_file = relabeled_clouds_folder / ply_file.name
    if overwrite or not relabeled_ply_file.exists():
        print(f"Relabeling: {ply_file}")
        data = read_ply(str(ply_file))
        cloud = np.vstack((data['x'], data['y'], data['z'])).T
        rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        dlaser = data['reflectance']
        if "label" not in data.dtype.names:
            print(f"Cancelling relabeling because no label field was given")
            continue
        label = data['label']
        label[label == 1] = 2
        write_ply(str(relabeled_ply_file), [cloud, dlaser, rgb_colors.astype(np.int32), label.astype(np.int32)], ['x', 'y', 'z', 'reflectance', 'red', 'green', 'blue', 'label'])
        print(f"Done relabeling: {ply_file}")

# %% Extract and backup voxel clouds

# Parameters
relabeled_clouds_folder = "../data/relabeled_clouds"
vc_backup_folder = "../data/backup/voxel_cloud"
overwrite = False
max_voxel_size = 0.3
threshold_grow = 2
min_voxel_length = 5
method = "regular"
use_color = True
use_reflectance = True

if not vc_backup_folder.exists():
    vc_backup_folder.mkdir()

for ply_file in relabeled_clouds_folder.glob("*.ply"):
    backup_file = vc_backup_folder / ply_file.name.replace("ply", "pkl")
    if overwrite or not backup_file.exists():
        print(f"Processing: {ply_file}")
        
        # Retrieve data
        data = read_ply(str(ply_file))
        cloud = np.vstack((data['x'], data['y'], data['z'])).T
        rgb_colors = np.vstack((data['red'], data['green'], data['blue'])).T if use_color else None
        dlaser = data['reflectance'] if use_reflectance else None
        label = data['label'] if "label" in data.dtype.names else None
        
        # Define clouds
        pc = PointCloud(cloud, dlaser, rgb_colors, label)
        vc = VoxelCloud(pc, max_voxel_size = max_voxel_size, threshold_grow = threshold_grow, min_voxel_length = min_voxel_length, method = method)
        
        # Save voxel cloud
        with open(backup_file, 'wb') as handle:
            pickle.dump(vc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done processing: {ply_file}\n")
    
# %% Extract and backup component clouds

# Parameters
vc_backup_folder = Path("..") / "data" / "backup" / "voxel_cloud"
cc_backup_folder = Path("..") / "data" / "backup" / "component_cloud"
overwrite = False
c_D = 0.25
segment_out_ground = False
threshold_in = 1 # for ground detection
threshold_normals = 0.8 # for ground detection
min_component_length = 5
use_neighbourhood_classifier = False
train_vc_files_for_classifier = ["domfountain1.pkl", "untermaederbrunnen1.pkl"]
classifier_type = 'random_forest' #'SGD'
classifier_kwargs = {'n_estimators': 20} #{}
scale_data = True

if not cc_backup_folder.exists():
    cc_backup_folder.mkdir()

# Train neighbourhood classifier if needed
if use_neighbourhood_classifier:
    
    # Retrieve voxel clouds from training set
    print("Loading voxel clouds for training neighbourhood classifier")
    train_vc = []
    for filename in train_vc_files_for_classifier:
        pkl_file = vc_backup_folder / filename
        with open(pkl_file, 'rb') as handle:
            vc = pickle.load(handle)
            train_vc.append(vc)

    # Declare classifier
    print("Constructing neighbourhood classifier")
    neighbourhood_classifier = NeighbourhoodClassifier(classifier_type, classifier_kwargs, scale_data, c_D, segment_out_ground, threshold_in, threshold_normals)
    
    # Train classifier
    print("Training neighbourhood classifier")
    neighbourhood_classifier.fit(train_vc)
else:
    neighbourhood_classifier = None
        
# Create Component Cloud objects
for pkl_file in vc_backup_folder.glob("*.pkl"):
    backup_file = cc_backup_folder / pkl_file.name
    if overwrite or not backup_file.exists():
        print(f"Making component cloud for: {pkl_file}")
        
        # Retrieve voxel cloud
        with open(pkl_file, 'rb') as handle:
            vc = pickle.load(handle)
        
        # Assign trained neighbourhood classifier to voxel cloud
        # If 'neighbourhood_classifier' is None, standard neighbourhood criteria (from paper) will be applied
        vc.set_neighbourhood_classifier(neighbourhood_classifier)
        
        # Compute component cloud
        cc = ComponentCloud(vc, c_D = c_D, segment_out_ground = segment_out_ground, threshold_in = threshold_in, threshold_normals = threshold_normals, min_component_length = min_component_length)
        
        # Save component cloud
        with open(backup_file, 'wb') as handle:
            pickle.dump(cc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done making component cloud for: {pkl_file}\n")

# %% Classify components

# Parameters
cc_backup_folder = Path("..") / "data" / "backup" / "component_cloud"
pc_backup_folder = Path("..") / "data" / "backup" / "predicted_cloud"
plot_backup_folder = Path("..") / "data" / "backup" / "new_plot"
train_cc_files = ["domfountain1.pkl", "untermaederbrunnen1.pkl"]
test_cc_files = ["domfountain2.pkl", "domfountain3.pkl", "neugasse.pkl", "untermaederbrunnen3.pkl"]
extra_test_cc_files = ["bildstein3.pkl", "bildstein5.pkl"]
classes = {2: "terrain", 3: "high vegetation", 4: "low vegetation", 5: "buildings", 6:"hard scape", 7:"scanning artefacts", 8:"cars"}
classifier_type = 'random_forest'
classifier_kwargs = {'n_estimators': 20}
scale_data = False
id_experimentation = 3

if not pc_backup_folder.exists():
    pc_backup_folder.mkdir()
if not plot_backup_folder.exists():
    plot_backup_folder.mkdir()

# Load train data
print("Loading train data")
train_cc = []
for filename in train_cc_files:
    pkl_file = cc_backup_folder / filename
    with open(pkl_file, 'rb') as handle:
        cc = pickle.load(handle)
        train_cc.append(cc)

# Load test data
print("Loading test data")
test_cc = []
for filename in test_cc_files:
    pkl_file = cc_backup_folder / filename
    with open(pkl_file, 'rb') as handle:
        cc = pickle.load(handle)
        test_cc.append(cc)

# Load extra test data
print("Loading extra test data")
extra_test_cc = []
for filename in extra_test_cc_files:
    pkl_file = cc_backup_folder / filename
    with open(pkl_file, 'rb') as handle:
        cc = pickle.load(handle)
        extra_test_cc.append(cc)

# Train classifier on train data
print("Training classifier")
cc_classifier = ComponentClassifier(classifier_type, classifier_kwargs, scale_data)
cc_classifier.fit(train_cc)

# Evaluate classifier on train data
cm = np.zeros((len(classes), len(classes)))
for i, cc in enumerate(train_cc):
    print(f"Evaluation of [TRAIN DATA] {train_cc_files[i]}")
    cc.set_predicted_labels(cc_classifier.predict(cc))
    cc.eval_classification_error(ground_truth_type = "componentwise")
    cc.eval_classification_error(ground_truth_type = "pointwise")
    this_cm = cc.eval_classification_error(ground_truth_type = "pointwise", include_unassociated_points = True, classes = np.array(list(classes.keys())))
    plot_confusion_matrix(this_cm, list(classes.values()), data_type = 'train_' + train_cc_files[i].split('.')[0], id = id_experimentation, folder = plot_backup_folder)
    cm += this_cm
plot_confusion_matrix(cm, list(classes.values()), data_type = 'train', id = id_experimentation, folder = plot_backup_folder)

# Evaluate classifier on test data
cm = np.zeros((len(classes), len(classes)))
for i, cc in enumerate(test_cc):
    print(f"Evaluation of [TEST DATA] {test_cc_files[i]}")
    cc.set_predicted_labels(cc_classifier.predict(cc))
    cc.eval_classification_error(ground_truth_type = "componentwise")
    cc.eval_classification_error(ground_truth_type = "pointwise")
    this_cm = cc.eval_classification_error(ground_truth_type = "pointwise", include_unassociated_points=True, classes = np.array(list(classes.keys())))
    plot_confusion_matrix(this_cm, list(classes.values()), data_type = 'test_' + test_cc_files[i].split('.')[0], id = id_experimentation, folder = plot_backup_folder)
    cm += this_cm
plot_confusion_matrix(cm, list(classes.values()), data_type = 'test', id = id_experimentation, folder = plot_backup_folder)

# Evaluate classifier on extra test data
cm = np.zeros((len(classes), len(classes)))
for i, cc in enumerate(extra_test_cc):
    print(f"Evaluation of [EXTRA TEST DATA] {extra_test_cc_files[i]}")
    cc.set_predicted_labels(cc_classifier.predict(cc))
    cc.eval_classification_error(ground_truth_type = "componentwise")
    cc.eval_classification_error(ground_truth_type = "pointwise")
    this_cm = cc.eval_classification_error(ground_truth_type = "pointwise", include_unassociated_points=True, classes = np.array(list(classes.keys())))
    plot_confusion_matrix(this_cm, list(classes.values()), data_type = 'extra_test_' + extra_test_cc_files[i].split('.')[0], id = id_experimentation, folder = plot_backup_folder)
    cm += this_cm
plot_confusion_matrix(cm, list(classes.values()), data_type = 'extra_test', id = id_experimentation, folder = plot_backup_folder)

# Save train results (predicted components, predicted labels, groundtruth labels)
print("Saving train results")
for i, cc in enumerate(train_cc):
    ply_file = os.path.join(pc_backup_folder, 'train_' + train_cc_files[i]).replace('pkl', 'ply')
    cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
    component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
    predicted_label = np.hstack([cc.predicted_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_component_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_voxel_label = np.hstack([cc.voxelcloud.features['majority_label'][c] for c in cc.components])
    write_ply(str(ply_file), [cloud_point, component, predicted_label, groundtruth_component_label, groundtruth_voxel_label],
              ['x', 'y', 'z', 'predicted_component', 'predicted_label', 'groundtruth_component_label', 'groundtruth_voxel_label'])
    
# Save test results (predicted components, predicted labels, groundtruth labels)
print("Saving test results")
for i, cc in enumerate(test_cc):
    ply_file = os.path.join(pc_backup_folder, 'test_' + test_cc_files[i]).replace('pkl', 'ply')
    cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
    component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
    predicted_label = np.hstack([cc.predicted_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_component_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_voxel_label = np.hstack([cc.voxelcloud.features['majority_label'][c] for c in cc.components])
    write_ply(str(ply_file), [cloud_point, component, predicted_label, groundtruth_component_label, groundtruth_voxel_label],
              ['x', 'y', 'z', 'predicted_component', 'predicted_label', 'groundtruth_component_label', 'groundtruth_voxel_label'])

# Save extra test results (predicted components, predicted labels, groundtruth labels)
print("Saving extra test results")
for i, cc in enumerate(extra_test_cc):
    ply_file = os.path.join(pc_backup_folder, 'test_' + extra_test_cc_files[i]).replace('pkl', 'ply')
    cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
    component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
    predicted_label = np.hstack([cc.predicted_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_component_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
    groundtruth_voxel_label = np.hstack([cc.voxelcloud.features['majority_label'][c] for c in cc.components])
    write_ply(str(ply_file), [cloud_point, component, predicted_label, groundtruth_component_label, groundtruth_voxel_label],
              ['x', 'y', 'z', 'predicted_component', 'predicted_label', 'groundtruth_component_label', 'groundtruth_voxel_label'])
    
print("Done")