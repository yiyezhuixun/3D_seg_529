import os


path = r'C:\Users\ascen\Desktop\tire_project\tag\0526_txl'
oldlist = os.listdir(path)  # 获得所有文件名列表
newlist = sorted(oldlist, key=lambda x: os.path.getmtime(os.path.join(path, x)))  # 按时间排序的文件名列表
a = 0
for i in newlist:
    os.rename(path + "/" + i, path + "/" + str(a) + ".bin")  # 重命名
    a += 1
