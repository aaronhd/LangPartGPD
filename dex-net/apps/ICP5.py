#!/usr/bin/env python3
from array import array
import open3d as o3d
import numpy as np
import copy
import os
import glob

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # print('source_fpfh', source_fpfh.num, target_fpfh.num)
    #     print('source_fpfh',source_fpfh,np.asarray(source_fpfh.data))
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4, [
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    0.9),
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    distance_threshold)
    ], o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 500))
    return result


# DATA
#  source data
# obj_name = '035_power_drill'
# obj_name = '048_hammer'
obj_name = '002_master_chef_can'

# source_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply" % (index_src))
source_show = o3d.io.read_point_cloud("/media/aaronstc/hard_1/dataset/YCB/models/ycb/{object_name}/google_512k/nontextured.ply".format(object_name=obj_name))  #  source 为需要配准的点云

# target data
# target_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply" % (index_ref))
print('object_name: ', obj_name)
data_path = os.environ["PointNetGPD_FOLDER"] + "/PointNetGPD/data/ycb-tools/models/ycb"
path = data_path + "/{}/rgbd/clouds".format(obj_name)
pd_files = glob.glob(os.path.join(path, "pc_*_NP5_*.pcd"))
points = np.array([[0, 0, 0]])
# print(len(pd_files))
amount = 0
i = 0
# for i in range(10):
while amount < 2000000:
    pd_path = pd_files[i]
    # print(pd_path)
    file_name = pd_path.split("/")[-1]
    print(file_name)
    data_pd = o3d.io.read_point_cloud(pd_path)
    np_point = np.asarray(data_pd.points)
    points = np.vstack([points, np_point])
    amount = points.shape[0]
    i = i + 1
    if i >=len(pd_files):
        break
print(points.shape)
target_show = o3d.geometry.PointCloud()
target_show.points = o3d.utility.Vector3dVector(points)




# target_show, outlier_index = target_show.remove_radius_outlier(
#                                               nb_points=16,
#                                               radius=0.5)

# PROCESS
voxel_size = 0.03
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_show, target_show, voxel_size)
result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh, voxel_size)
# print(result_ransac.transformation)
result_icp = o3d.pipelines.registration.registration_icp(
source, target, 0.2, result_ransac.transformation,
o3d.pipelines.registration.TransformationEstimationPointToPoint())

# result_icp = o3d.pipelines.registration.registration_icp(
# source, target_down, 0.2, result_ransac.transformation,
# o3d.pipelines.registration.TransformationEstimationPointToPlane())

print(result_icp.transformation)
target_show, outlier_index = target_show.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
draw_registration_result(source_show, target_show, result_icp.transformation)

# trans_init = np.asarray([[ 6.82572007e-01, -7.30818987e-01,  1.30600005e-03,
#         -3.77200008e-03],
#        [-7.30821013e-01, -6.82570994e-01,  4.84999997e-04,
#         -4.56700008e-03],
#        [ 5.36000007e-04, -1.28600001e-03, -1.00000000e+00,
#          1.40433997e-01],
#        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          1.00000000e+00]])

# draw_registration_result(source, target, trans_init)
