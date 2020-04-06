#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..\\src") # Windows
sys.path.append("../src") # Linux
import pickle
from pathlib import Path
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import sparse
from utils.ply import read_ply, write_ply, make_ply
from plots import plot, plot_confusion_matrix
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

vc.remove_poorly_connected_voxels(c_D = 0.25, threshold_nb_voxels = 10)

plot(vc, figsize = (8,8), also_unassociated_points = True, only_voxel_center = False)
plt.tight_layout()

i = vc.features['mean_intensity']
vari = vc.features['var_intensity']
i2 = cm.jet((i - np.min(i)) / (np.max(i) - np.min(i)))
vari2 = cm.jet((vari - np.min(vari)) / (np.max(vari) - np.min(vari)))
c = vc.features['mean_color']
varc = vc.features['var_color'] / np.max(vc.features['var_color'])

g = nx.from_numpy_matrix(A)
nx.draw(g, node_color = np.hstack((c/255, np.ones((len(c), 1)))))
nx.draw(g, node_color = i2)
nx.draw(g)

plot(vc, colors = c, only_voxel_center = False)
plot(vc, colors = vari, only_voxel_center = False)
plot(vc, colors = eigenvectors[:,1], only_voxel_center = False)

plot(vc, colors = lbl, only_voxel_center = False)

components_idx = np.unique(lbl)
components = []

for i in components_idx:
    components.append(np.where(lbl == i)[0])

return components

cc = ComponentCloud(
    voxelcloud = vc, 
    c_D = 0.25,
    segment_out_ground = True,
    method = "spectral",
    K = 14,
    min_component_length = 1,
    threshold_in = 0.2,
    threshold_normals = 0.8
)

A, D = vc.build_similarity_graph(0.25, [1,1,1])

import networkx as nx
g = nx.from_numpy_matrix(A)
nx.draw(g, node_color = np.hstack((c/255, np.ones((len(c), 1)))))
nx.draw(g, node_color = i2)
nx.draw(g)
c = vc.features['mean_color']

plot(cc, only_voxel_center = False, figsize=(8,8))

# Save component cloud
print("Saving component cloud for display")
ply_file = os.path.join(test_backup_folder, os.path.basename(pkl_file)).replace('pkl', 'ply')
cloud_point = np.vstack([cc.voxelcloud.features["geometric_center"][c] for c in cc.components])
component = np.hstack([random.random() * np.ones(len(c)) for c in cc.components])
groundtruth_label = np.hstack([cc.majority_label[i] * np.ones(len(c)) for i, c in enumerate(cc.components)])
write_ply(ply_file, [cloud_point, component, groundtruth_label], ['x', 'y', 'z', 'predicted_component', 'groundtruth_label'])



# %%
import numpy as np
A = np.array([[0,4,0],[4,0,1],[0,1,0]])
D = np.array([[4,0,0],[0,5,0],[0,0,1]])
L = D - A
e,v = np.linalg.eigh(L)
e1, e2, e3 = e
u1, u2, u3 = v[:,0][:,None], v[:,1][:,None], v[:,2][:,None]

Ls = 1/e2 * u2@u2.T + 1/e3 * u3@u3.T
sqrtK = np.sqrt(e2) * u2@u2.T + np.sqrt(e3) * u3@u3.T
np.linalg.eigh(Ls)
1/np.sqrt(e1) * u1
1/np.sqrt(e2) * u2
1/np.sqrt(e3) * u3
