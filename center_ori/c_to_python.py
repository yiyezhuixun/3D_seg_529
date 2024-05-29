from ctypes import *
from ctypes import cdll


# cdll.LoadLibrary("./advapi32.dll")
# cdll.LoadLibrary("./kernel32.dll")
# cdll.LoadLibrary("./msvcp140d.dll")
# cdll.LoadLibrary("./pcl_commond.dll")
#
#
# cdll.LoadLibrary("./ucrtbased.dll")
# cdll.LoadLibrary("./vcruntime140_1d.dll")
# cdll.LoadLibrary("./vcruntime140d.dll")
# cdll.LoadLibrary("./vtksys-9.3-gd.dll")
#
#
# cdll.LoadLibrary("./pcl_iod.dll")
# cdll.LoadLibrary("./vtkCommonCore-9.3-gd.dll")
# cdll.LoadLibrary("./pcl_segmentationd.dll")
def sACSegCircle3D(xodrPath):
    ddd = cdll.LoadLibrary("./Dll1.dll")

    ddd.sACSegCircle3D.restype = POINTER(c_float * 7)
    xodr = (c_char * 100)(*bytes(xodrPath, "utf-8"))
    cast(xodr, POINTER(c_char))
    posInfo = ddd.sACSegCircle3D(xodr)
    result = []
    for i in range(7):
        # 将函数返回结果存储到result数组中
        result.append(posInfo.contents[i])
    return result

def sACSegCircle3D2(xodrPath,thres_len,setmax):
    ddd = cdll.LoadLibrary(r"C:\Users\ascen\Desktop\Pointnet_book_seam-main\DLL\Dll1.dll")
    ddd.sACSegCircle3D2.restype = POINTER(c_float * 7)


    xodr = (c_char * 100)(*bytes(xodrPath, "utf-8"))
    th_len = c_double(thres_len)
    st_max = c_int(setmax)

    posInfo = ddd.sACSegCircle3D2(xodr,th_len,st_max)
    result = []
    for i in range(7):
        # 将函数返回结果存储到result数组中
        result.append(posInfo.contents[i])
    return result