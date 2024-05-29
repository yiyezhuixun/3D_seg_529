from DLL.c_to_python  import sACSegCircle3D
from DLL.c_to_python import sACSegCircle3D2
import open3d as o3d
import os
from ctypes import *

# 加载DLL
def get_cloud_center(path):
    posInfo = sACSegCircle3D2(path, 1, 500)
    return posInfo

if __name__ == '__main__':

    xodrPath = r"C:\Al\3D\3d_cloud_xuanzhuan\0528\PonintNet_2.pcd"

    # posInfo = sACSegCircle3D(xodrPath)
    # posInfo = sACSegCircle3D2(xodrPath,0.7,200)
    # for i in  range(1,1000):
    #     stemax = i*50
    posInfo = sACSegCircle3D2(xodrPath,1,500)

    a =[]
    for i in posInfo:
        a.append(i)
        print(a)


