# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_points(cloud, idxs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in idxs:
        data = cloud[i, :]
        ax.plot(data[:,0], data[:,1], data[:,2], '.')
    plt.show()


