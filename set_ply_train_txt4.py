# 打开文件读取所有行
import os

##### 将txt 文件的行间距缩成训练需要的格式
def set_ply(file_path, save_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = [line.strip().split('\t') for line in lines]
    for row in new_lines:
        id_1 = row[3]
        id_2 = row[4]
        id_3 = row[5]
        id_4 = row[6]

        row[3], row[4], row[5] = row[7], row[8], row[9]
        row[6], row[7], row[8], row[9] = id_1, id_2, id_3, id_4
        if row[6] == "nan" and row[7] == "nan" and row[8] == "nan" and row[9] == "nan":
             del row
             continue
        elif row[6] != "nan" and row[7] == "nan" and row[8] == "nan" and row[9] == "nan":
            row[6] = '0.000000'
        elif row[6] == "nan" and row[7] != "nan" and row[8] == "nan" and row[9] == "nan":
            row[6] = '1.000000'
        elif row[6] == "nan" and row[7] != "nan" and row[8] == "nan" and row[9] != "nan":
            row[6] = '3.000000'
        elif row[6] == "nan" and row[7] == "nan" and row[8] != "nan" and row[9] == "nan" :
            row[6] = '2.000000'
        elif row[6] == "nan" and row[7] != "nan" and row[8] != "nan" and row[9] == "nan" :
            row[6] = '2.000000'

    with open(save_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row[:-3]))
            if row[6] != 'nan':
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


if __name__ == '__main__':

    input_path = r"C:/Users/ascen/Desktop/tire project/tag/0514/clound_points_4/"
    out_path =r"C:/Users/ascen/Desktop/tire project/tag/0514/result_4/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            save_path = out_path + filename
            set_ply(input_path + filename, save_path)
            print(save_path + "  is ok")
