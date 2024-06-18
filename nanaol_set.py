import os
import numpy as np


def loda_txt(input_path, out_path):
    a = []
    points = np.loadtxt(input_path)
    for i in range(0, points.shape[0]):
        # 找到最后一列不为NaN的值的索引
        for t in range(3, 6):
            points[i][t] = float(0.0)
            a.append(points[i])

    np.savetxt(out_path, np.array(a), fmt='%.8f', delimiter='  ')


if __name__ == '__main__':


    input_path = "data/book_seam_dataset/05110/"
    out_path = "data/book_seam_dataset/05110_1/"
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            loda_txt(input_path + filename, out_path + filename)
            print(out_path + filename + "  is ok")