import os
import re
# 设置文件夹路径
folder_path = r'C:/Users/ascen/Desktop/tire_project/tag/0521/clound_points/'

# 获取文件夹中所有文件的名称
filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
a =[]
b ='shape_data/05110/'
# 将文件名称保存到txt文件
with open('filenames.txt', 'w') as file:
    for filename in filenames:
        filename = re.sub('.txt', "", filename)
        filename = b + filename
        a.append(filename)
        file.write('"'+filename +'"' +','+'\n')
print(a)