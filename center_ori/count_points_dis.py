from c_to_python import sACSegCircle3D
import os
from ctypes import *

# 加载DLL

if __name__ == '__main__':

    xodrPath = r"C:/Users/ascen/Desktop/416/cloud_points/0416_02_hub.ply"
    posInfo = sACSegCircle3D(xodrPath)
    x = posInfo[0]
    y = posInfo[1]
    z = posInfo[2]
    radius = posInfo[3]




