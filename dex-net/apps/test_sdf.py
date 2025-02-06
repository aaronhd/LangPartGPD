from meshpy.sdf_file import SdfFile
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np
import os

# sf = SdfFile('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/models/eff3a27a085e02e5146be45f8a3c1ff8/models/model_normalized.sdf')
# sf = SdfFile('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/1a64bf1e658652ddb11647ffa4306609/models/model_normalized.sdf')
# sf = SdfFile('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/2b28e2a5080101d245af43a64155c221/models/model_normalized.sdf')
# sf = SdfFile('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/2b28e2a5080101d245af43a64155c221/models/model_normalized_rescale.sdf')
# sf = SdfFile('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/1a74b169a76e651ebc0909d98a1ff2b4/models/model_normalized.sdf')
sf = SdfFile('/media/aaronstc/hard_1/dataset/shapenet_mesh/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/lang_data/model_normalized.sdf')
# sf = SdfFile('/media/aaronstc/hard_1/dataset/shapenet_mesh/03001627/5e338c489e940ab73aab636b8c7f0dd2/lang_data/model_normalized.sdf')

sdf_path = '/media/aaronstc/hard_1/dataset/shapenet_mesh/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/lang_data/model_normalized.sdf'
verify_sdf = open(sdf_path)
print(len(verify_sdf.readlines()))

sdf = sf.read()
points, sdf_value = sdf.surface_points(grid_basis=False)
print(sdf.resolution)  # 0.00208296
print(points.shape)  # (19478, 3)
print(points[:2])
# np.savetxt('/home/aaronstc/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/shapenet/1a64bf1e658652ddb11647ffa4306609/models/model_normalized_sdf.txt', points)
# np.savetxt('/media/aaronstc/hard_1/dataset/shapenet_mesh/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/lang_data/model_normalized_sdf.txt', points)
# np.savetxt('/media/aaronstc/hard_1/dataset/shapenet_mesh/03001627/1a38407b3036795d19fb4103277a6b93/lang_data/model_normalized_sdf.txt', points)

idx = np.where(sdf_value > 0)
print(points.shape)
points = points[idx]
print(points.shape)
sdf_value = sdf_value[idx]

colors = np.zeros(points.shape)
colors[sdf_value < 0, 2] = 1
colors[sdf_value > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=1)


# path = '/lang_data/sdf_grd/model_normalized_sdf_grd_list_17.txt'
# part_label = path.split('/')[-1][:-4].split('_')[-1]
# print(part_label)
#
#
# root_path = '/media/aaronstc/hard_1/dataset/shapenet_mesh'
#
# output_root = '/media/aaronstc/hard_1/dataset/shapenet_mesh_tmp'
# category_name = '02691156'
#
# new_folder = os.path.join(root_path, category_name, category_name + '_' + str(part_label))
# print(new_folder)
