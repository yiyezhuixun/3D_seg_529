import open3d as o3d
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

txt_path = ('0416_04_pc - Cloud.txt')
# txt_path ="data2/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1ea7a36e4f353416fe1f6e05091d5d9.txt"
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
# txt点云前三个数值一般对应x、y、z坐标，可以通过open3d.geometry.PointCloud().points加载
# 如果有法线或颜色，那么可以分别通过open3d.geometry.PointCloud().normals或open3d.geometry.PointCloud().colors加载
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])

# 获取点云三维坐标
points = np.asarray(pcd_vector.points)
# 获取点云XY坐标
point2d = np.c_[points[:, 0], points[:, 1]]
# 获取平面点云的凸多边形边界
ch2d = spatial.ConvexHull(point2d)
# 可视化凸多边形边界结果


plt.figure()
# 方法二：直接可视化
ax = plt.subplot(aspect="equal")
spatial.convex_hull_plot_2d(ch2d, ax=ax)
plt.show()