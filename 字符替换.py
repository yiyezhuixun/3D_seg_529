# 原字符
import os
import numpy as np

input_path = r"C:\Users\ascen\Desktop\Pointnet_book_seam-main\filenames.txt"

with open(input_path, 'r') as file:
    lines = file.readlines()
new_lines = [line.strip().split('\t') for line in lines]
for row in new_lines:
    row[0] = row[0].replace(".bin", " - Cloud.remaining")
with open(input_path, 'w') as file:
        for row in new_lines:
            file.write('"' + row[0] + '"'+ ','+ '\n')
