#!/usr/bin/env python3
from array import array
import open3d as o3d
import numpy as np
import open3d
import copy
import os
import glob
import time
# from open3d.t.pipelines import registration as treg
# import open3d.pipelines.registration as treg

# if o3d.__DEVICE_API__ == 'cuda':
#      import open3d.cuda.pybind.t.pipelines.registration as treg
# else:
#      import open3d.cpu.pybind.t.pipelines.registration as treg

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


#读取电脑中的 ply 点云文件
source = o3d.io.read_point_cloud("/media/aaronstc/hard_1/dataset/YCB/models/ycb/035_power_drill/google_512k/nontextured.ply")  #source 为需要配准的点云
# target = o3d.io.read_point_cloud("/media/aaronstc/hard_1/dataset/YCB/models/ycb/003_cracker_box/rgbd/clouds/pc_NP3_NP5_117.ply")  #target 为目标点云

obj_name = '035_power_drill'
print(obj_name)
data_path = os.environ["PointNetGPD_FOLDER"] + "/PointNetGPD/data/ycb-tools/models/ycb"
path = data_path + "/{}/rgbd/clouds".format(obj_name)
pd_files = glob.glob(os.path.join(path, "pc_*_NP5_*.pcd"))
points = np.array([[0, 0, 0]])
# print(len(pd_files))
amount = 0
i = 0
# for i in range(10):
while amount < 10000:
    pd_path = pd_files[i]
    # print(pd_path)
    file_name = pd_path.split("/")[-1]
    print(file_name)
    data_pd = o3d.io.read_point_cloud(pd_path)
    np_point = np.asarray(data_pd.points)
    points = np.vstack([points, np_point])
    amount = points.shape[0]
    i = i + 1
print(points.shape)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(points)


# voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
voxel_size = 0.025

# List of Convergence-Criteria for Multi-Scale ICP:
# criteria_list = [
#     open3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
#                                 relative_rmse=0.0001,
#                                 max_iteration=20),
#     open3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
#     open3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
# ]

criteria = open3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                       relative_rmse=0.000001,
                                       max_iteration=50)

# `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
# max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
max_correspondence_distance = 0.07

# Initial alignment or source to target transform.
init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)

# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
estimation = open3d.t.pipelines.registration.TransformationEstimationPointToPlane()
# estimation = open3d.t.pipelines.registration.TransformationEstimationPointToPoint()

# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
save_loss_log = True

s = time.time()

# print(type(source))
# print(type(target))
# print(type(voxel_sizes))
# print(type(criteria_list))
# print(type(max_correspondence_distances))
# print(type(init_source_to_target))
# print(type(estimation))

# source_cuda = source.cuda(0)
# target_cuda = target.cuda(0)

# registration_ms_icp = open3d.t.pipelines.registration.multi_scale_icp(source_cuda, target_cuda, voxel_sizes,
#                                            criteria_list,
#                                            max_correspondence_distances,
#                                            init_source_to_target, estimation,
#                                            save_loss_log)


registration_icp = open3d.t.pipelines.registration.icp(source=source,
                            target=target, max_correspondence_distance=max_correspondence_distance,
                            init_source_to_target=init_source_to_target, estimation_method=estimation, 
                            criteria=criteria,
                            voxel_size=voxel_size, save_loss_log=save_loss_log)
icp_time = time.time() - s
print("Time taken by ICP: ", icp_time)
print("Inlier Fitness: ", registration_icp.fitness)
print("Inlier RMSE: ", registration_icp.inlier_rmse)
draw_registration_result(source, target, registration_icp.transformation)


# print("Time taken by Multi-Scale ICP: ", ms_icp_time)
# print("Inlier Fitness: ", registration_ms_icp.fitness)
# print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)

# draw_registration_result(source, target, registration_ms_icp.transformation)


# 不同的颜色
# source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
# target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal
# processed_source, outlier_index = source.remove_radius_outlier(
#                                               nb_points=16,
#                                               radius=0.5)

# processed_target, outlier_index = target.remove_radius_outlier(
#                                               nb_points=16,
#                                               radius=0.5)

processed_target, outlier_index = target.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
processed_source = source
# processed_target = target
threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])


# draw_registration_result(source, target, trans_init)

# #运行icp
reg_p2p = o3d.pipelines.registration.registration_icp(
        processed_source, processed_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)
    )
# 
#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p)
print(np.round(reg_p2p.transformation, 4))

# print(type(processed_target))
# print(type(processed_target.PointCloud))
# o3d.visualization.draw_geometries([processed_target])

tf = np.array([[-0.85618401, -0.020944  , -0.51624602, -0.011459  ],
       [-0.51666498,  0.029961  ,  0.85566598, -0.019795  ],
       [-0.002454  ,  0.999331  , -0.036473  ,  0.087675  ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
)

# draw_registration_result(processed_source, processed_target, tf)

draw_registration_result(processed_source, processed_target, reg_p2p.transformation)
