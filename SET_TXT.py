# 打开文件读取所有行
import os
import numpy as np
import  time
import open3d as o3d
import datetime

# 获取当前时间



def loda_txt(before_file,after_file):
    '''

    将标签转为RGB
    '''
    a1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  ###红0， 绿1 ，蓝2
    a2 = np.array(a1)
    a = []
    points = np.loadtxt(before_file)
    for i in range(0, points.shape[0]):
        # 找到最后一列不为NaN的值的索引
        if points[i][6] == float(0.0):
            p = np.append(points[i][:6], a2[0])
            a.append(p)
        elif points[i][6] == float(1.0):
            p = np.append(points[i][:6], a2[1])
            a.append(p)
        elif points[i][6] == float(2.0):
            p = np.append(points[i][:6], a2[2])
            a.append(p)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(a[:, 3:6])
    ## 保存成ply数据格式
    pcd = pcd.uniform_down_sample(every_k_points=12)
    o3d.io.write_point_cloud(after_file, pcd, write_ascii=True)  # ascii编码
    #
    # np.savetxt(after_file, np.array(a), fmt='%.8f', delimiter='  ')

def load_label_2(input_path,out_path):
    import arrow
    points = np.loadtxt(input_path)
    a = []
    for i in range(0, points.shape[0]):
        if points[i][6] == float(1.0):
            a.append(points[i])
        elif points[i][6] == float(0.0):
            a.append(points[i])
    a = np.array(a)
    np.savetxt(out_path, a, fmt='%.8f')



if __name__ == '__main__':


    input_path = r"./result2/PonintNet_predict_txt/"
    out_path = "./result2/PonintNet_predict_txt/1111/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            load_label_2(input_path + filename, out_path + filename)
            print(out_path + filename + "  is ok")
