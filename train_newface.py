import numpy as np
import cv2
import glob
from hadamard import hadamard2
import time
from skimage.exposure import rescale_intensity

#テンプレートマッチング
def template(eigenVectors):
    # Create a container to hold eigen faces.
    eigenFaces = []
    # Reshape eigen vectors to eigen faces.
    for i, eigenVector in enumerate(eigenVectors):
        eigenFace = eigenVector.reshape(size)  
        # eigenFace1 = rescale_intensity(eigenFace, out_range=(0,255))   
        # eigenFace1 = np.dstack([eigenFace1.astype("uint8")])
        # cv2.imshow("com", eigenFace1)
        # cv2.waitKey(0)
        # cv2.imread()
        # cv2.imwrite(f'new_eigen/new_eigen32/eigen_{i}.jpg', eigenFace1)
        eigenFaces.append(eigenFace)
    return eigenFaces

def add_data(path, train_x, train_y):
    path_list = glob.glob(path+'/*')

    for label, pic_path in enumerate(path_list):
        train_list = glob.glob(pic_path+'/*.jpg')
        train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
        for i in train_list:
            x_train = cv2.imread(i)
            # 正規化
            x_train = np.float32(x_train) / 255
            # リストに追加
            # train_x = train_x.tolist()
            # train_x.append(x_train)
            train_x = np.concatenate([train_x, np.expand_dims(x_train, axis=0)], axis=0)
        # train_y = train_y.tolist()
        train_y = np.concatenate([train_y, np.full(len(train_list), label)])
        # train_y += [label]*len(train_list)
    
    return train_x, train_y


if __name__ == '__main__':
    start_time = time.time()

    eigenVectors = np.load('npy/eigenVectors.npy')
    eigenFaces = np.load('npy/eigen1_512.npy')
    x_train = np.load('npy/x_train.npy')
    y_train = np.load('npy/y_train.npy')

    # indices = np.random.permutation(x_train.shape[0])
    # x_train = x_train[indices]
    # y_train = y_train[indices]

    # 訓練データにデータを加える
    path = "add_train1"
    x_train, y_train = add_data(path, x_train, y_train)
    print(y_train[1500])
    
    # 画像サイズ
    size = x_train[0].shape
    #事前学習データ
    train_x = np.ones((len(x_train), len(eigenFaces)))
    for i,img in enumerate(x_train):
        for j,eigenFace in enumerate(eigenFaces):
            result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCORR)
            train_x[i,j]=result
    
    # apple_train = train_x[0:359,:]
    # lemon_train = train_x[360:719,:]
    # orange_train = train_x[720:1079,:]
    # np.save('npy/apple_train.npy', apple_train)
    # np.save('npy/lemon_train.npy', lemon_train)
    # np.save('npy/orange_train.npy', orange_train)

    # 状態変換行列（ハッシュ関数生成）
    num_bit = 4
    Fmap = {'nu': 1e-5, 'lambda': 1e-2}
    trans = hadamard2(train_x, y_train, num_bit, Fmap)
    trans = trans.astype(np.float32)
    
    new_trans = np.transpose(trans)
    new_eigenVectors = np.dot(new_trans, eigenVectors)
    # # 新しいeigenface
    # eigen = np.transpose(eigenVectors)
    # # ハッシュ関数をかける
    # new_eigen = np.dot(eigen, trans)
    # # 元に戻す
    # new_eigenVectors = np.transpose(new_eigen)
    # 固有値画像生成
    new_eigenFaces = template(new_eigenVectors)

    #学習データ
    new_train_x = np.ones((len(x_train), len(new_eigenFaces)))
    for i,img in enumerate(x_train):
        for j,new_eigenFace in enumerate(new_eigenFaces):
            result = cv2.matchTemplate(img, new_eigenFace, cv2.TM_CCORR)
            new_train_x[i,j]=result


    # #カーネル
    # Ntrain = train_x.shape[0]
    # n_anchor = Ntrain
    # index = np.random.choice(Ntrain, n_anchor, replace=False)
    # X_anchor = train_x[index,:]
    # #ガウスカーネル
    # Vsigma = 0.3
    # train_X = np.exp(-sqdist(train_x, X_anchor) / (2*Vsigma*Vsigma))

    # 0以上を1，それ以外を-1にする
    # X_train = X_train_i >0
    # X_train = X_train.astype(int)
    # signによる識別関数
    X_train = np.sign(new_train_x)
    X_train = X_train.astype(int)

    # バイナリビットに変換しないときの最初の特徴ベクトルを取得
    X_train_seq = []
    # for j in range(0, 1081, 360):
    #     X_train_seq.append(X_train[j])
    # 変換したとき重複ないように
    X_train_seq, indice = np.unique(X_train, axis = 0, return_index = True)
    # X_train_seq = list(dict.fromkeys(map(tuple, X_train)))
    X_train_seq = np.array(X_train_seq)
    print(X_train_seq)
    y_train_seq = []
    # for j in range(0, 1080, 360):
    #     y_train_seq.append(y_train[j])
    y = y_train[indice]
    y_train_seq.append(y)
    #y_train_seq = np.unique(y_train, axis=0)
    print(y_train_seq)
    #X_train = X_train_i.astype(int)
    #np.savetxt('txt/my_data.txt', X_train, fmt='%d')

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")

    np.save('new_npy/new_eigen2_16', new_eigenFaces)
    np.save('new_npy/X_train_seq.npy', X_train_seq)
    np.save('new_npy/y_train_seq.npy', y_train_seq)