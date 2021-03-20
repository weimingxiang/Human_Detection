# 处理报错：iCCP: known incorrect sRGB profile
import os
# 你的ImageMagick位置
CMD = r'C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\magick.exe'
# 需要处理的文件夹的位置
SOURCE_PATH = r'INRIAPerson\Test'


def doStrip(path):
    data = {}
    print(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            name = file.lower()
            if name.find('.png') != -1:
                path = os.path.join(root, file)
                os.system('"{0}" {1} -strip {1}'.format(CMD, path, path))


doStrip(SOURCE_PATH)
