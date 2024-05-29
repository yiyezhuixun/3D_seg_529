import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import open3d as o3d
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

txt_path = ('data2/book_seam_dataset/00000001/0416_06_pc - Cloud.segmented.txt')
# txt_path ="data2/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1ea7a36e4f353416fe1f6e05091d5d9.txt"
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
# txt点云前三个数值一般对应x、y、z坐标，可以通过open3d.geometry.PointCloud().points加载
# 如果有法线或颜色，那么可以分别通过open3d.geometry.PointCloud().normals或open3d.geometry.PointCloud().colors加载
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
points =pcd_vector.points
xyz_load = np.asarray(points)
print(xyz_load)
# 生成三维圆点云
def generate_circle_points(radius, num_points):
    angles = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)

    noise = np.random.normal(0, 0.03, size=(num_points,))
    x += noise
    y += noise
    z += noise

    return np.column_stack((x, y, z))

# points = generate_circle_points(radius=1, num_points=50)

# 使用RANSAC算法拟合三维圆
def fit_3d_circle(points):
    polynomial_features = PolynomialFeatures(degree=2)
    X = polynomial_features.fit_transform(points[:, :2])

    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.1, max_trials=100)
    ransac.fit(X, points[:, 2])

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    inliers = points[inlier_mask]
    outliers = points[outlier_mask]

    return inliers, outliers


points =pcd_vector.points
xyz_load = np.asarray(points)
print(xyz_load)
inliers, outliers = fit_3d_circle(xyz_load)

# 绘制三维散点图和拟合圆的轮廓线
def plot_3d_scatter(points, ax):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

def plot_3d_circle(circle, ax):
    theta = np.linspace(0, 2*np.pi, 100)
    x = circle[0] + circle[3] * np.cos(theta)
    y = circle[1] + circle[3] * np.sin(theta)
    z = circle[2] * np.ones_like(theta)
    def f(c):

        x_c, y_c, r = c
        return (x * x_c + y * y_c + r) / np.sqrt(x ** 2 + y ** 2) - 1

        # 初始猜测

    center_guess = np.array([x.mean(), y.mean()])
    radius_guess = np.mean(np.sqrt(x ** 2 + y ** 2))
    c0 = np.array([center_guess[0], center_guess[1], radius_guess])

    # 使用最小二乘方法拟合
    result = least_squares(f, c0)

    # 返回拟合结果
    return result.x
    ax.plot(x, y, z, color='#FF0000', linewidth=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_3d_scatter(xyz_load, ax)

circle_params = np.append(inliers.mean(axis=0), np.sqrt(np.sum(np.square(inliers - inliers.mean(axis=0)), axis=1)).mean())
r =plot_3d_circle(circle_params, ax)
print("半径：",r[2])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.show()