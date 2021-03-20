import os
from PIL import Image


def getAllName(file_dir, tail_list=['.jpg', '.ppm']):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L


names = getAllName("pedestrians128x64")  # 这里是读取ppm文件的文件夹路径
save_dir = "pedestrians128x64"  # 保存jpg文件的文件夹路径

for name in names:
    base_name = os.path.basename(name)
    img = Image.open(name)

    base_name = base_name[:-3]+"jpg"
    save_name = os.path.join(save_dir, base_name)
    img.save(save_name)
