# 打开文件读取所有行
import os


#########预处理cloudcompare 生成的测试点云文件
def set_ply2(file_path,save_file):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 处理每一行，分割并交换列
    new_lines = [line.strip().split('\t') for line in lines]
    with open(save_file, 'w') as file:
        for row in new_lines:
            row = list(map(str, row))
            file.write(' '.join(row) + '\n')

if __name__ == '__main__':
    input_path = "./cloud_point/"
    out_path = "./"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            point_file = input_path + filename
            save_file = out_path + filename
            set_ply2(point_file,save_file)
            print(save_file + "  is ok")
