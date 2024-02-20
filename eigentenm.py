from __future__ import print_function
import os
import sys
import cv2
import glob
import numpy as np
import math
from PIL import Image
from hadamard import hadamard2
from skimage.exposure import rescale_intensity
from imutils import build_montages
import matplotlib.pyplot as plt
import re
import time

# 行列を生成する
def createMatrix(images):
    print("Creating data matrix", end = "...")
    images_num = len(images)
    size = images[0].shape
    #Matrix
    #data = np.zeros((images_num, size[0]*size[1]), dtype = np.float32)
    data = np.zeros((images_num, size[0]*size[1]*size[2]), dtype = np.float32)
    #data = np.zeros((images_num, size[0]*size[1]), dtype = np.float32)
    for i in range(images_num):
        #image = images[i].flatten()
        image = images[i].reshape(-1)
        data[i,:]= image

    return data

#テンプレートマッチング
def template(mean, eigenVectors):
    averageFace = mean.reshape(size)
    averageFace = cv2.resize(averageFace, (255,255))

    # Create a container to hold eigen faces.
    eigenFaces = []
    # Reshape eigen vectors to eigen faces.
    for i, eigenVector in enumerate(eigenVectors):
        eigenFace = eigenVector.reshape(size) 
        eigenFace1 = rescale_intensity(eigenFace, out_range=(0,255))   
        eigenFace1 = np.dstack([eigenFace1.astype("uint8")])    
        # cv2.imshow("com", eigenFace)
        # cv2.waitKey(0)
        #cv2.imread()
        # cv2.imwrite(f'eigenimage_python_512/eigen_{i}.jpg', eigenFace*2000+128)
        cv2.imwrite(f'eigenimage/eigenimage512/eigen_{i}.jpg', eigenFace1)
        eigenFaces.append(eigenFace)
    # montage = build_montages(eigenFaces, (32, 32), (3, 4))[0]
    # cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    # cv2.imshow("mean", averageFace)
    # cv2.imshow("component", montage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return eigenFaces

#画像データセット作成
def get_data(path):
    path_list = glob.glob(path+'/*')
    #path_list = ["apple", "lemon", "orange"]
    
    train_x = []
    train_y = []
    apple_rotate = []
    lemon_rotate = []
    orange_rotate = []
    for label, pic_path in enumerate(path_list):
        #pic_path = glob.glob(path + "/" + pic_path)
        train_list = glob.glob(pic_path+'/*.jpg')
        train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
        for i in train_list:
            x_train = cv2.imread(i)
            #画像サイズ変更
            x_train_c = cv2.resize(x_train, (84,84))
            if x_train_c is None:
                print("image:{} not read properly".format(i))
            else:
                #正規化
                x_train = np.float32(x_train) / 255
                #リストに追加
                train_x.append(x_train)
            # どのくらい回転しているかを保存
            if label == 0:
                apple_rotate.append(i)
            elif label == 1:
                lemon_rotate.append(i)
            elif label == 2:
                orange_rotate.append(i)
        train_y += [label]*len(train_list)

    return train_x,train_y,apple_rotate,lemon_rotate,orange_rotate


if __name__ == '__main__':
    start_time = time.time()
    #PCA後のテンプレート数
    NUM_EIGEN_FACES = 512
    #画像ファイルディレクトリ
    #dirname = "train1"
    path = "tra2"
    #images = load_face_dataset(dirname)
    x_train,y_train,apple_rotate,lemon_rotate,orange_rotate = get_data(path)
    np.save('npy/x_train.npy', x_train)
    np.save('npy/y_train.npy', y_train)
    np.save('npy/apple_rotate.npy', apple_rotate)
    np.save('npy/lemon_rotate.npy', lemon_rotate)
    np.save('npy/orange_rotate.npy', orange_rotate)

    #x = x_train[0]
    size = x_train[0].shape
    data = createMatrix(x_train[0:1079])
    # Compute the eigenvectors from the stack of images created.
    print("Calculating PCA", end = "...")
    mean, eigenVectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN_FACES)
    np.save('npy/eigenVectors.npy', eigenVectors)
    # eigenvalues = np.power(cv2.PCACompute(data, mean=None)[1],2)

    # #寄与率
    # ratio = eigenvalues / np.sum(eigenvalues)
    # sum_ratio = np.cumsum(ratio)

    # plt.plot(sum_ratio, marker='o')
    # plt.xlim(0,512)
    # plt.xlabel('num_eigen')
    # plt.ylabel('sum_ratio')
    # plt.show()

    eigenFaces = template(mean, eigenVectors)
    np.save('npy/eigen1_512.npy', eigenFaces)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")
    