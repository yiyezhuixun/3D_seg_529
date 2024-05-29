# 打开文件读取所有行
import os

import numpy as np
def set_ply(file_path, save_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = [line.strip().split('\t') for line in lines]
    for row in new_lines:
        aa =np.array(row)
        if row[6] != "2.000000":
            row[6] = '1.000000'
            continue
        elif row[7] != "2.000000":
            row[6] = '3.000000'


    with open(save_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row))
            file.write('  '.join(row) + '\n')


def set_ply2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 处理每一行，分割并交换列
    new_lines = [line.strip().split('\t') for line in lines]
    with open(file_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row))
            file.write(' '.join(row) + '\n')


if __name__ == '__main__':

    input_path = "cloud_point/"
    out_path = "3d_edge_data/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            save_path = out_path + filename
            set_ply(input_path + filename, save_path)
            print(save_path + "  is ok")
