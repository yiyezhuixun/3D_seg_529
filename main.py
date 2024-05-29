import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import json
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import  shutil
import arrow
warnings.filterwarnings('ignore')
matplotlib.use("Agg")
import visualize_partseg_open3d_2 as vis3d

def get_latest_file(directory):
    latest_file = None
    latest_ctime = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and os.path.getctime(file_path) > latest_ctime:
            latest_file = file_path
            latest_ctime = os.path.getctime(file_path)
    return latest_file


if __name__ == '__main__':

    # 1）采集到的ply *1
    # 2) ply 转 txt 出两份（1份原始，1份无法向量）*2
    # 3) txt 进 整体分割模型（出1份轮胎的坐标合集补上1份原始的法向量）*1
    # 4）轮胎txt 出两份（1份原始，1份无法向量）*2 + 1
    # 5）无法向量版本分割模型
    # 6）使用模型（1份轮胎、1份轮毂、1份合集处，均补法向量，均有无法向量版本）*6

    S = time.time()
    img_root = r'./data/book_seam_dataset'
    target_root = r'./result1/'
    #=======================================
    path = 'data/book_seam_dataset/12345678/0528_03_pc_1 - Cloud.txt'
    #=======================================
    num_classes = 4
    classes = {'book': [0, 1, 2, 3]}
    "所有的模型以PointNet++为标准  输入两个参数 输出两个参数，如果模型仅输出一个，可以将其修改为多输出一个None！！！！"
    # # ==============================================
    from models.pointnet2_part_seg_msg import get_model as pointnet2

    model1 = pointnet2(num_classes=num_classes, normal_channel=True).eval()
    # ============================================
    "Dataset同理，都按ShapeNet格式输出三个变量 point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别"
    "不是这个格式的话就手动添加一个"
    TEST_DATASET = vis3d.PartNormalDataset(root=img_root, npoints=8000 * 100, normal_channel=True, path=path,
                                           seg_classes=classes)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=20,
                                                 drop_last=True)

    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
    print('第一次模型分割开始----------------')
    model_dict = {'PonintNet': [model1, r'././log/part_seg/lable_4/']}

    c = vis3d.Generate_txt_and_3d_img(img_root, target_root, num_classes, testDataLoader, model_dict, color_map, path=path,
                                      models=1)
    print('第一次模型分割完成----------------')
    # ===============================================
    # ===============================================
    # ===============================================
    print('第二次模型分割开始----------------')

    path2 = 'data/book_seam_dataset/12345678/PonintNet_0.txt'
    target_root2 = './result2/'
    num_classes2 = 3
    classes2 = {'book': [0, 1,2]}
    model2 = pointnet2(num_classes=num_classes2, normal_channel=True).eval()
    model_dict2 = {'PonintNet': [model2, r'./log/part_seg/pointnet_part_3/']}
    TEST_DATASET2 = vis3d.PartNormalDataset(root=img_root, npoints=6000 * 100, normal_channel=True, path=path2,
                                            seg_classes=classes2)
    testDataLoader2 = torch.utils.data.DataLoader(TEST_DATASET2, batch_size=1, shuffle=True, num_workers=20,
                                                  drop_last=True)
    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes2))}
    d = vis3d.Generate_txt_and_3d_img(img_root, target_root2, num_classes2, testDataLoader2, model_dict2, color_map,
                                      path=path2, models=2)
    print('第二次模型分割结束----------------')
    N = time.time()
    print("执行时间：", N-S)