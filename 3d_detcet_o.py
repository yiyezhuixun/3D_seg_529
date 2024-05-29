import numpy as np
from scipy.optimize import least_squares


def circle_fit(points):
    # 转换为二维数组
    points = np.asarray(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 设置拟合模型，参考 https://en.wikipedia.org/wiki/Circle#Parametric_equation
    def f(c):
        # c = (x_c, y_c, r)
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


# 示例点云数据
import open3d as o3d
import numpy as np
from PIL import Image

txt_path = ('results/label_txt/0_label.txt')
# txt_path ="data2/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1ea7a36e4f353416fe1f6e05091d5d9.txt"
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
# txt点云前三个数值一般对应x、y、z坐标，可以通过open3d.geometry.PointCloud().points加载
# 如果有法线或颜色，那么可以分别通过open3d.geometry.PointCloud().normals或open3d.geometry.PointCloud().colors加载
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])

# 拟合圆形
circle_params = circle_fit(pcd_vector.points)
print(f"圆的参数：中心点({circle_params[0]}, {circle_params[1]})，半径: {circle_params[2]}")