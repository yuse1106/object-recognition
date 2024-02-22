import cv2
import os
import random

#　画像ファイル名
file_name = './image/back.jpg'
# 切り出す画像のサイズ
frame_size = (84,84)
# 画像の読み込み
image = cv2.imread(file_name)
image = cv2.resize(image, (400, 400))
h, w, c = image.shape
# 生成画像保存
dir_name = 'train3/w_back'
os.makedirs(dir_name, exist_ok=True)
# 切り抜き処理
image_num = 720
for i in range(image_num):
    x = random.randint(0, w-frame_size[1])
    y = random.randint(0, h-frame_size[0])

    cut_img = image[y:y+frame_size[1], x:x+frame_size[0],:]
    #  画像保存
    cv2.imwrite(dir_name + f'/w_back_{i}.jpg', cut_img)