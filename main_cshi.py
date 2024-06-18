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


def maim(rob_path):
    #
    txt_path = rob_path.replace(".ply", ".txt")
    # # save_path = r'./data/book_seam_dataset/save_ori_data/'
    # # if os.path.exists(txt_path):
    # #     shutil.move(rob_path,save_path)
    read_ply_cloud(rob_path,txt_path)
    shutil.move(rob_path, 'data/book_seam_dataset/save_ori_data')
    # os.remove(rob_path)
    S = time.time()
    img_root = r'./data/book_seam_dataset'
    target_root = r'./result1/'
    # =======================================
    path = txt_path
    # =======================================
    num_classes = 3
    classes = {'test1': [0, 1, 2]}
    "所有的模型以PointNet++为标准  输入两个参数 输出两个参数，如果模型仅输出一个，可以将其修改为多输出一个None！！！！"
    # # ==============================================
    from models.pointnet2_part_seg_msg import get_model as pointnet2

    model1 = pointnet2(num_classes=num_classes, normal_channel=True).eval()
    # ============================================
    "Dataset同理，都按ShapeNet格式输出三个变量 point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别"
    "不是这个格式的话就手动添加一个"
    TEST_DATASET = vis3d.PartNormalDataset(root=img_root, npoints=5000 * 100, normal_channel=True, path=path,
                                           seg_classes=classes)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=0,
                                                 drop_last=True)

    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
    print('第一次模型分割开始----------------')
    model_dict = {'PonintNet': [model1, r'././log/part_seg/wl_first/']}

    c = vis3d.Generate_txt_and_3d_img(img_root, target_root, num_classes, testDataLoader, model_dict, color_map,
                                      path=path,
                                      models=1)
    print('第一次模型分割完成----------------')
    # ===============================================
    # ===============================================
    # ===============================================
    print('第二次模型分割开始----------------')

    path2 = 'data/book_seam_dataset/ceshi/PonintNet_0.txt'
    target_root2 = './result2/'
    num_classes2 = 3
    classes2 = {'test2': [0, 1, 2]}
    model2 = pointnet2(num_classes=num_classes2, normal_channel=True).eval()
    model_dict2 = {'PonintNet': [model2, r'./log/part_seg/wl_second/']}
    TEST_DATASET2 = vis3d.PartNormalDataset(root=img_root, npoints=3000 * 100, normal_channel=True, path=path2,
                                            seg_classes=classes2)
    testDataLoader2 = torch.utils.data.DataLoader(TEST_DATASET2, batch_size=1, shuffle=True, num_workers=0,
                                                  drop_last=True)
    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes2))}
    d = vis3d.Generate_txt_and_3d_img(img_root, target_root2, num_classes2, testDataLoader2, model_dict2, color_map,
                                      path=path2, models=2)
    print('第二次模型分割结束----------------')
    N = time.time()
    print("执行时间：", N - S)
    cccc ='ok'
    return  cccc


# if __name__ == '__main__':
#     cccc =  maim('data/book_seam_dataset/ceshi/0604_08_pc.ply')