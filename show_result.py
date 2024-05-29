
import open3d as o3d
import numpy as np
from PIL import Image
import  time

# txt_path = (r'C:\Users\ascen\Desktop\Pointnet_book_seam-main\save_results_ply\2024-05-26-17-22.pcd')
# pcd = np.genfromtxt(txt_path)
# pcd_vector = o3d.geometry.PointCloud()
#
# # 加载点坐标
# pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
# # colors = np.random.randint(255, size=(2, 3)) / 255  ####根据标签数量进行修改
# # pcd_vector.colors = o3d.utility.Vector3dVector(colors[list(map(int, pcd[:, 6])), :])
# pcd_vector.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])
# o3d.visualization.draw_geometries([pcd_vector])
# ###直接读点云数据




pcd = o3d.io.read_point_cloud(r"C:\Users\ascen\Desktop\Pointnet_book_seam-main\save_results_ply\2024-05-26-17-22.pcd")
pcd.paint_uniform_color([255, 0, 0])
# 可视化点云
o3d.visualization.draw_geometries([pcd])
# end_time = time.time()
# print("clos:",end_time-s_time)