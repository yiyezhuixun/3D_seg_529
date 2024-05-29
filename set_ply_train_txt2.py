# 打开文件读取所有行
import os
import numpy as np
import open3d as o3d
##### 将txt 文件的行间距缩成训练需要的格式
def set_ply(file_path, save_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = [line.strip().split('\t') for line in lines]
    for row in new_lines:

        id_1 = row[3]
        id2 = row[4]

        row[3],row[4],row[5] = row[5],row[6],row[7]
        row[6],row[7]= id_1,id2

        if row[6] != "nan" and row[7] == "nan":
            row[6] = row[6]
        elif row[6] == "nan" and row[7] != "nan":
            row[6] = row[7]
        elif row[6] == "nan" and row[7] == "nan":
            row[6] = "nan"

    with open(save_path, 'w') as file:
        for row in new_lines:
            if row[6] != "nan" :
               row = list(map(str, row[:-1]))
               file.write('  '.join(row) + '\n')


def set_ply2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 处理每一行，分割并交换列
    new_lines = [line.strip().split(' ') for line in lines]
    with open(file_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row))
            file.write(' '.join(row) + '\n')


def set_ply3(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = [line.strip().split('\t') for line in lines]
    for row in new_lines:

        id_1 = row[3]

        row[3],row[4],row[5] = row[4],row[5],row[6]
        row[6]= id_1

    with open(file_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row[:7]))
            file.write('  '.join(row) + '\n')


def load_label_2():
    result_path = './result2/PonintNet_predict_txt/PonintNet_0.txt'
    points = np.loadtxt(result_path)

    a = []
    for i in range(0, points.shape[0]):
        #
        # if points[i][6] == float(0.0):
        #     a.append(points[i])

        if points[i][6] != float(1.0):
            a.append(points[i])

    a = np.array(a)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(a[:, 3:6])
    ## 保存成ply数据格式
    o3d.io.write_point_cloud('./525txt.ply', pcd, write_ascii=True)  # ascii编码

if __name__ == '__main__':
    #
    input_path = r"C:/Users/ascen/Desktop/2lable/1111/"
    out_path = r"C:/Users/ascen/Desktop/2lable/1111/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            save_path = out_path + filename
            set_ply(input_path + filename, save_path)
            print(save_path + "  is ok")

   # set_ply3('data/book_seam_dataset/12345678/0514_072.txt')