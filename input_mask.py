# 入力画像に異なる背景画像にする
import cv2
import numpy as np

number = 20
for i in range(number):
    # 画像の読み込み
    image = cv2.imread(f'test/input1/input_{i}.jpg')
    back_img = cv2.imread('image/back.jpg')
    back_img = cv2.resize(back_img, (400,400))

    # 白色の部分をマスクして
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thre, mask = cv2.threshold(gray_img, 245, 255, cv2.THRESH_BINARY)

    # マスク画像の反転
    mask_inv = cv2.bitwise_not(mask)

    # 画像と背景を合成
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    back_portion = cv2.bitwise_and(back_img, back_img, mask=mask)
    final_img = cv2.add(result, back_portion)

    # cv2.imshow('output', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f'test/input_back1/input_{i}.jpg', final_img)
