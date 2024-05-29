"传入模型权重文件，读取预测点，生成预测的txt文件"
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

from DLL.main import get_cloud_center
import open3d as o3d
import os
from ctypes import *

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/book_seam_dataset', npoints=4090, class_choice=None, normal_channel=True ,path='',seg_classes = ''):
        self.npoints = npoints  # 采样点数
        self.root = root  # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 类别和文件夹名字对应的路径
        self.cat = {}
        self.normal_channel = normal_channel  # 是否使用rgb信息

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in
                    self.cat.items()}  # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(
            len(self.cat))))  # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        if not class_choice is None:  # 选择一些类别进行训练  好像没有使用这个功能
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.datapath = [(list(self.cat.keys())[list(self.cat.values()).index(path.split('/')[-2])], path)]
        # 输出的是元组，('Airplane',123.txt)

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes  将类别的名称和索引对应起来  例如 飞机 <----> 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
        self.seg_classes 就是将大类和小类对应起来
        """
        # self.seg_classes = {'book': [0, 1]}
        self.seg_classes = seg_classes
        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 500000

    def __getitem__(self, index):

            fn = self.datapath[0] # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
            cat = self.datapath[0][0] # 拿到类名
            cls = self.classes[cat] # 将类名转换为索引
            cls = np.array([cls]).astype(np.int32)
            # 读取modelnet40
            data = np.loadtxt(fn[1],dtype=np.float32)

            # 读取shapenet
            # data2 = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
            if not self.normal_channel:  # 判断是否使用rgb信息
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32) # 拿到小类别的标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 做一个归一化
            choice = np.random.choice(len(seg), self.npoints, replace=True)  # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
            # resample
            point_set = point_set[choice, :]  # 根据索引采样
            seg = seg[choice]
            return point_set, choice, fn[1],cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)



class Generate_txt_and_3d_img:
    def __init__(self, img_root, target_root, num_classes, testDataLoader, model_dict, color_map=None, path='123.txt',models=''):
        self.img_root = img_root  # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.color_map = color_map
        self.heat_map = False  # 控制是否输出heatmap
        self.label_path_txt = os.path.join(self.target_root, 'label_txt')  # 存放label的txt文件，指标注
        self.path = path
        self.make_dir(self.label_path_txt)

        # 拿到模型 并加载权重
        self.model_name = []
        self.model = []
        self.model_weight_path = []

        for k, v in model_dict.items():
            self.model_name.append(k)
            self.model.append(v[0])
            self.model_weight_path.append(v[1])

        # 加载权重
        self.load_cheackpoint_for_models(self.model_name, self.model, self.model_weight_path)
        # 创建文件夹
        self.all_pred_image_path = []  # 所有预测结果的路径列表
        self.all_pred_txt_path = []  # 所有预测txt的路径列表
        for n in self.model_name:
            self.make_dir(os.path.join(self.target_root, n + '_predict_txt'))
            self.make_dir(os.path.join(self.target_root, n + '_predict_image'))
            self.all_pred_txt_path.append(os.path.join(self.target_root, n + '_predict_txt'))
            self.all_pred_image_path.append(os.path.join(self.target_root, n + '_predict_image'))
        "将模型对应的预测txt结果和img结果生成出来，对应几个模型就在列表中添加几个元素"


        self.generate_predict_to_txt()
        today = arrow.now().format("YYYY-MM-DD-HH-mm")
        today = str(today)
        pri_second_path = os.path.join('result1/pridit_second_mode/',  today +'_predict.txt')
        if models == 1:
           self.load_label_3(pri_second_path,today)
        elif models == 2:
           save_tow_cir = 'save_results_ply/' +today + '_two' + '.pcd'
           save_in_cir = 'save_cir_center/' +today+ '_in' +'.pcd'
           self.load_label_2(save_tow_cir,save_in_cir,today)

           # self.o3d_draw_3d_img(save_path)
           cenrt = get_cloud_center(save_in_cir)
           self.get_center1(save_tow_cir,cenrt)


    def get_center1(self,save_path,center):

        pcd = o3d.io.read_point_cloud(save_path)
        pcd.paint_uniform_color([0, 1, 0])

        # 输出最小圆的中心和半径

        points = np.asarray(pcd.points)
        print("points_len", len(points))

        # center2 = pcd.get_center()
        # print("orr:",center2)
        # 用户指定的点云数据
        x = center[0]
        y = center[1]
        z = center[2]
        xyz2 = np.array([[x, y, z]])
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyz2)  # 这里设置需要显示的点云
        pcd1.paint_uniform_color([0, 0, 255])
        center3 = pcd1.get_center()
        print(center3)
        # 距离计算

        dists = pcd.compute_point_cloud_distance(pcd1)
        dists = np.asarray(dists)
        print("len_dus", len(dists))
        unique_array = np.unique(dists)
        np.savetxt('array.txt', unique_array, fmt='%d')
        # plt.plot(unique_array)
        # plt.show()

        ind = np.where((dists >= 260) & (dists <= 270))[0]

        a = []
        for u in ind:
            a.append(dists[u])
        # a.sort()
        a = np.asarray(a)
        np.savetxt('1111.txt', a, fmt='%f.08')
        plt.plot(a)
        plt.show()
        print("len:", len(ind))
        pcd3 = pcd.select_by_index(ind)
        pcd3.paint_uniform_color([255, 0, 0])
        center4 = pcd3.get_center()
        print(center4)

        o3d.visualization.draw_geometries([pcd3, pcd], window_name="bouding box")
        # o3d.io.write_point_cloud(save_path, [pcd3, pcd], write_ascii=True)  # ascii编码



    def generate_predict_to_txt(self):

            for batch_id, (points,choice,fn,label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                          total=len(self.testDataLoader),smoothing=0.9):

                #点云数据、整个图像的标签、每个点的标签、  没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])

                points = points.transpose(2, 1)
                patt = str(fn[0])
                data = np.loadtxt(patt, dtype=np.float32)
                points_ori = data[:, 0:6]
                points_ori = points_ori[choice, :]
                points_ori = torch.from_numpy(points_ori)
                points_ori = points_ori.transpose(2, 1)
                xyz_feature_point = points[:, :6, :]
                xyz_ppp = points_ori[:, :6, :]
                # 将标签保存为txt文件
                point_set_without_normal = np.asarray(torch.cat([points.permute(0, 2, 1),target[:,:,None]],dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
                np.savetxt(os.path.join(self.label_path_txt,f'{batch_id}_label.txt'), point_set_without_normal, fmt='%.04f') # 将其存储为txt文件
                " points  torch.Size([16, 2048, 6])  label torch.Size([16, 1])  target torch.Size([16, 2048])"

                assert len(self.model) == len(self.all_pred_txt_path) , '路径与模型数量不匹配，请检查'

                for n,model,pred_path in zip(self.model_name,self.model,self.all_pred_txt_path):

                    seg_pred, trans_feat = model(points, self.to_categorical(label, 1))
                    # seg_pred = seg_pred.cpu().data.numpy()
                    seg_pred = seg_pred.cpu().data.numpy()
                    #=================================================
                    # seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                    if self.heat_map:
                        out =  np.asarray(np.sum(seg_pred,axis=2))
                        seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
                    else:
                        seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                    #=================================================
                    seg_pred = np.concatenate([np.asarray(xyz_ppp), seg_pred[:, None, :]],
                            axis=1).transpose((0, 2, 1)).squeeze(0)  # 将点云与预测结果进行拼接，准备生成txt文件
                    save_path = os.path.join(pred_path, f'{n}_{batch_id}.txt')
                    np.savetxt(save_path,seg_pred, fmt='%.04f')

    def o3d_draw_3d_img(self,save_path):


        pcd = o3d.io.read_point_cloud(save_path)

        pcd.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([pcd])

        # result_path = os.path.join(self.all_pred_txt_path[0], f'PonintNet_0.txt')
        # pcd = np.genfromtxt(save_path, delimiter=" ")
        # pcd_vector = o3d.geometry.PointCloud()
        # # 加载点坐标
        # pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
        # colors = np.random.randint(255, size=(1, 3)) / 255  ####根据标签数量进行修改
        # # print("colors",colors)
        # # a =[[1,0,0]]
        # # colors = np.array(a)
        # pcd_vector.colors = o3d.utility.Vector3dVector(colors[list(map(int, pcd[:, 6])), :])
        # pcd_vector.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])
        # o3d.visualization.draw_geometries([pcd_vector])

    def load_label_3(self, pri_second_path, today):
        result_path = os.path.join(self.all_pred_txt_path[0], f'PonintNet_0.txt')


        # today = arrow.now().format("YYYY-MM-DD-HH-mm")
        # save_path = os.path.join('result1/pridit_second_mode/', str(today))
        points = np.loadtxt(result_path)
        a = []
        for i in range(0, points.shape[0]):
            # 找到最后一列不为NaN的值的索引
            if points[i][6] == float(1.0):
                points[i][6] = 'nan'
                a.append(points[i])
            if points[i][6] == float(3.0):
                points[i][6] = 'nan'
                a.append(points[i])
        np.savetxt(pri_second_path, a, fmt='%.8f', delimiter='  ')
        save_path = os.path.join('data/book_seam_dataset/12345678/', f'PonintNet_0.txt')
        np.savetxt(save_path, a, fmt='%.8f', delimiter='  ')
        os.rename(result_path, self.all_pred_txt_path[0] + '/' + today + '.txt')


    def load_label_2(self,save_tow_cir,save_in_cir,today):

        result_path = os.path.join(self.all_pred_txt_path[0], f'PonintNet_0.txt')
        points = np.loadtxt(result_path)

        two_cir = []
        in_cir = []
        for i in range(0, points.shape[0]):
            if points[i][6] == float(0.0):
                two_cir.append(points[i])
            elif points[i][6] == float(2.0):
                two_cir.append(points[i])
                in_cir.append(points[i])
        in_cir = np.array(in_cir)
        two_cir = np.array(two_cir)
        pcd_two_cir = o3d.geometry.PointCloud()
        pcd_two_cir.points = o3d.utility.Vector3dVector(two_cir[:, :3])
        pcd_two_cir.normals = o3d.utility.Vector3dVector(two_cir[:, 3:6])
        # o3d.visualization.draw_geometries([pcd])
        ## 保存成ply数据格式
        pcd_in_cir =  o3d.geometry.PointCloud()
        pcd_in_cir.points = o3d.utility.Vector3dVector(in_cir[:, :3])
        pcd_in_cir.normals = o3d.utility.Vector3dVector(in_cir[:, 3:6])

        o3d.io.write_point_cloud(save_in_cir, pcd_in_cir,write_ascii=True)
        o3d.io.write_point_cloud(save_tow_cir, pcd_two_cir, write_ascii=True)
        os.rename(result_path, self.all_pred_txt_path[0] + '/' + today + '.txt')
        # ascii编码

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

    def load_cheackpoint_for_models(self, name, model, cheackpoints):

        assert cheackpoints is not None, '请填写权重文件'
        assert model is not None, '请实例化模型'

        for n, m, c in zip(name, model, cheackpoints):
            print(f'正在加载{n}的权重.....')
            weight_dict = torch.load(os.path.join(c, 'best_model.pth'))
            m.load_state_dict(weight_dict['model_state_dict'], strict=False)

            print(f'{n}权重加载完毕')


# if __name__ == '__main__':
#     import copy
#
#     img_root = r'./data/book_seam_dataset'
#     target_root = r'./results/'
#     path = 'data/book_seam_dataset/12345678/0514_07_pc - Cloud.remaining.txt'
#     num_classes = 4
#     classes = {'book': [0, 1, 2, 3]}
#     "所有的模型以PointNet++为标准  输入两个参数 输出两个参数，如果模型仅输出一个，可以将其修改为多输出一个None！！！！"
#     # # ==============================================
#     from models.pointnet2_part_seg_msg import get_model as pointnet2
#
#     model1 = pointnet2(num_classes=num_classes, normal_channel=True).eval()
#     # ============================================
#     "Dataset同理，都按ShapeNet格式输出三个变量 point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别"
#     "不是这个格式的话就手动添加一个"
#     TEST_DATASET = PartNormalDataset(root=img_root, npoints=4096 * 100, normal_channel=True, path=path,
#                                      seg_classes=classes)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=20,
#                                                  drop_last=True)
#
#     color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
#     print('第一次模型分割开始----------------')
#     model_dict = {'PonintNet': [model1, r'./log/part_seg/pointnet2_part_seg_msg/checkpoints/']}
#
#     c = Generate_txt_and_3d_img(img_root, target_root, num_classes, testDataLoader, model_dict, color_map, path=path,
#                                 models=1)
#     print('第一次模型分割完成----------------')
#     # ===============================================
#
#     print('第二次模型分割开始----------------')
#     path2 = 'data/book_seam_dataset/12345678/PonintNet_0.txt'
#     target_root2 = './result2/'
#     num_classes2 = 2
#     classes2 = {'book': [0, 1]}
#
#     model2 = pointnet2(num_classes=num_classes2, normal_channel=True).eval()
#
#     model_dict2 = {'PonintNet': [model2, r'./log/part_seg/2024-05-17_12-07/checkpoints/']}
#     TEST_DATASET2 = PartNormalDataset(root=img_root, npoints=5000 * 100, normal_channel=True, path=path2,
#                                       seg_classes=classes2)
#     testDataLoader2 = torch.utils.data.DataLoader(TEST_DATASET2, batch_size=1, shuffle=True, num_workers=20,
#                                                   drop_last=True)
#
#     color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes2))}
#
#     d = Generate_txt_and_3d_img(img_root, target_root2, num_classes2, testDataLoader2, model_dict2, color_map,
#                                 path=path2, models=2)
#
#     print('第二次模型分割结束----------------')


