# 打开文件读取所有行
import os

##### 将txt 文件的行间距缩成训练需要的格式
def set_ply(file_path, save_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 处理每一行，分割并交换列
    #####     6======0   top
    ###       7======1   wheel
    ##        8=====2    wheel_hub
    ###       9======3   hub
    ##        10 =====4  down
    new_lines = [line.strip().split('\t') for line in lines]
    for row in new_lines:
        id_1 = row[3]
        id_2 = row[4]
        id_3 = row[5]
        row[3], row[4], row[5] = row[6], row[7], row[8]
        row[6], row[7],row[8]= id_1, id_2, id_3

        if row[6] != "nan" and row[7] == "nan" and row[8] == "nan":
            row[6] = row[6]
        elif row[6] == "nan" and row[7] != "nan" and row[8] == "nan":
            row[6] = row[7]
        elif row[6] == "nan" and row[7] == "nan" and row[8] != "nan":
            row[6] =  row[8]
        # elif row[6] == "nan" and row[7] == "nan" and row[8] == "nan":
        #     row[6] = 'nan'
        #     continue

    with open(save_path, 'w') as file:
        for row in new_lines:
            row = list(map(str, row[:-2]))
            if row[6] != "nan":
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
    import time
    input_path = r"C:/Users/ascen/Desktop/txt/"
    out_path =r"C:/Users/ascen/Desktop/1111/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            save_path = out_path + filename
            s =time.time()
            set_ply(input_path + filename, save_path)
            n =time.time()
            print(save_path + "  is ok   time:",n-s)
