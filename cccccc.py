import  os
import shutil
from main_cshi import maim
or_path1 = r"rob_data/"
save_path = r"data/book_seam_dataset/ceshi/"

for filename in os.listdir(or_path1):
    if filename.endswith(".ply"):
        or_path = os.path.join(or_path1, filename)
        save_path = os.path.join(save_path, filename)
        shutil.move(or_path, save_path)
        cc = maim(save_path)
        print(or_path1)