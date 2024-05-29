import open3d as o3d
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import copy




def get_center1(pcd):

    pcd.paint_uniform_color([0, 1, 0])

    # 输出最小圆的中心和半径

    points = np.asarray(pcd.points)
    print("points_len", len(points))

    # center2 = pcd.get_center()
    # print("orr:",center2)
    # 用户指定的点云数据

    x=-70.49092102050781
    y=-26.068490982055664
    z= 204.3885040283203
    xyz2 = np.array([[x, y,z]])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz2)  # 这里设置需要显示的点云
    pcd1.paint_uniform_color([0, 0, 255])
    center3= pcd1.get_center()
    print(center3)
    # 距离计算

    dists = pcd.compute_point_cloud_distance(pcd1)
    dists = np.asarray(dists)
    print("len_dus",len(dists))
    unique_array = np.unique(dists)
    np.savetxt('array.txt', unique_array,fmt='%d')
    plt.plot(unique_array)
    plt.show()
    ind = np.where((dists >= 264) & (dists <= 280))[0]
    print("len:",len(ind))
    pcd3 = pcd.select_by_index(ind)
    pcd3.paint_uniform_color([255, 0, 0])
    center4= pcd3.get_center()
    print(center4)


    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.paint_uniform_color([1, 0, 0])
    # pcd2 = pcd.select_by_index(ind)
    # pcd3.points = o3d.utility.Vector3dVector(pcd2.points)
    # 打印前三个点的坐标
    # for i in range(min(3, len(points))):
    #     print("Point", i+1, ":", points[i])
    # print(pcd3)

    o3d.visualization.draw_geometries([pcd3,pcd], window_name="bouding box")
    return pcd3


pcd1 = o3d.io.read_point_cloud('ply_data/PonintNet_0.pcd',)

pcd3 = get_center1(pcd1)




