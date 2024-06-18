import open3d as o3d
import numpy as np
import mayavi.mlab as mlab
import numpy as np
from plyfile import PlyData
import os
import time


def open3d_vector_compute():
    '''
     将点云的ply文件格式转为txt
    '''
    pcd_path = r'C:\Users\ascen\Desktop\Pointnet_book_seam-main\test\0530_12_pc(1).ply'
    pcd = o3d.io.read_point_cloud(pcd_path)
    ####计算点云法向量
    # pcd_radius = o3d.geometry.PointCloud()
    # pcd_radius.points = o3d.utility.Vector3dVector(pcd.points)
    # pcd_radius.colors = o3d.utility.Vector3dVector(pcd.colors)
    # pcd_radius.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=20))
    txt_file_path = r'C:\Users\ascen\Desktop\Pointnet_book_seam-main\test\count_nomals.ply'
    # txt_file_path = r'count_nomals.txt'
    ####保存为txt文件
    # np.savetxt(txt_file_path, np.array(pcd_radius.normals), fmt='%.8f')
    #       保存为ply文件
    o3d.io.write_point_cloud(txt_file_path, pcd, write_ascii=False,compressed=False)


def read_ply_cloud(pcd_path,save_path):
    '''
    将点云的ply格式转为txt格式
    '''
    pcd = o3d.io.read_point_cloud(pcd_path)
    normals = np.array(pcd.normals)    # 法向量结果与点云维度一致(N, 3)
    points = np.array(pcd.points)
    print(points.shape)
    cloud = np.empty([points.shape[0], 6])
    for i in range(points.shape[0]):
        # normals[i]
        p = np.append(points[i], normals[i])
        # na = np.array([0, 0, 0])  ###法向量为0
        # q = np.append(points[i], p)
        cloud[i] = p
    np.savetxt(save_path, cloud, fmt='%.8f')

if __name__ == '__main__':
    ss_time = time.time()
    ply_path = 'data/book_seam_dataset/weilai/0604_08_pc.ply'
    save_path = 'data/book_seam_dataset/weilai/0604_08_pc.txt'
    if os.path.exists(ply_path):
        read_ply_cloud(ply_path,save_path)
        print("okkk")
    print("txt:",time.time() - ss_time)

    # s_t = time.time()
    # open3d_vector_compute()
    # print("ply:", time.time() - s_t)


