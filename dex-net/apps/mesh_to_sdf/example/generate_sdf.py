#!/usr/bin/env python
# encoding: utf-8
"""
@version: 0.1
@author: Yaoxian
@software: PyCharm
@file: generate_sdf.py
@time: 2022/2/14 上午11:22
"""

from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np


# def compute_unit_sphere_transform(mesh):
#     """
#     returns translation and scale, which is applied to meshes before computing their SDF cloud
#     """
#     # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
#     translation = -mesh.bounding_box.centroid
#     scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
#     return translation, scale


mesh = trimesh.load('nontextured.obj')

points, sdf = sample_sdf_near_surface(mesh, number_of_points=500000)

# translation, scale = compute_unit_sphere_transform(mesh)
# points = (points / scale) - translation
# sdf /= scale

print(points.shape)
colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=1)
