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
while amount < 100000:
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


# threshold = 0.02

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

# downpcd = processed_target.voxel_down_sample(voxel_size=0.05)
# downpcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# print(downpcd)
                                                
processed_source = source
# processed_target = target
threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])

# trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                                     [-0.139, 0.967, -0.215, 0.7],
#                                     [0.487, 0.255, 0.835, -1.4],
#                                     [0.0, 0.0, 0.0, 1.0]])


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


# processed_source.transform(reg_p2p.transformation)


# #创建一个 o3d.visualizer class
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# #将两个点云放入visualizer
# vis.add_geometry(processed_source)
# vis.add_geometry(processed_target)

# #让visualizer渲染点云
# vis.update_geometry()
# vis.poll_events()
# vis.update_renderer()

# vis.run()
