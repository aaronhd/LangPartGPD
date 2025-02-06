#!/usr/bin/env python3
from array import array
import open3d as o3d
import numpy as np
import copy
import os
import glob


root_path = '/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/1a74b169a76e651ebc0909d98a1ff2b4'
print(root_path)

mesh = o3d.io.read_triangle_mesh(root_path + '/models/model_normalized.obj')
semantic_name = root_path.split('/')[-1]

# mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
pcd_1 = mesh.sample_points_uniformly(number_of_points=2500)
o3d.visualization.draw_geometries([pcd_1])

o3d.io.write_point_cloud(root_path + "/models/obj_to_pcd.pcd", pcd_1)
xyz_load = np.asarray(pcd_1.points)
np.savetxt(root_path +'/models/obj_to_pcd.txt', xyz_load)

normal_channel = False
pcd_path = root_path + "/" + semantic_name + ".txt"
data = np.loadtxt(pcd_path).astype(np.float32)
if not normal_channel:
    point_set = data[:, 0:3]
else:
    point_set = data[:, 0:6]
# seg = data[:, -1].astype(np.int32)
print(point_set.shape)
pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(point_set)
o3d.visualization.draw_geometries([pcd_2])
o3d.io.write_point_cloud(root_path + "/models/txt_to_pcd.pcd", pcd_2)
