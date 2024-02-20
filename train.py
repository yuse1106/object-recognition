import numpy as np
import cv2
import glob
from hadamard import hadamard2
import time


#二乗距離
def sqdist(X, Y):
    dist = np.sum((X[:, np.newaxis]-Y) ** 2, axis=-1)
    return dist
#正規化相互相関
def norm(img1,img2):
    #img2 = np.float32(img2) / 255
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCORR)
    #result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF)
    return result

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

start_time = time.time()
x_train = np.load('npy/x_train.npy')
eigenFaces = np.load('npy/eigen1_512.npy')
y_train = np.load('npy/y_train.npy')

# 訓練データにデータを加える
path = "add_train"
x_train, y_train = add_data(path, x_train, y_train)
print(y_train[1500])

#学習データ
train_x = np.ones((len(x_train), len(eigenFaces)))
for i,img in enumerate(x_train):
    for j,eigenFace in enumerate(eigenFaces):
        result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCORR)
        train_x[i,j]=result

apple_train = train_x[0:359,:]
lemon_train = train_x[360:719,:]
orange_train = train_x[720:1079,:]
np.save('npy/apple_train.npy', apple_train)
np.save('npy/lemon_train.npy', lemon_train)
np.save('npy/orange_train.npy', orange_train)

# #カーネル
# Ntrain = train_x.shape[0]
# n_anchor = Ntrain
# index = np.random.choice(Ntrain, n_anchor, replace=False)
# X_anchor = train_x[index,:]
# #ガウスカーネル
# Vsigma = 0.3
# train_X = np.exp(-sqdist(train_x, X_anchor) / (2*Vsigma*Vsigma))

num_bit = 4
#状態変換行列生成
Fmap = {'nu': 1e-5, 'lambda': 1e-2}
trans = hadamard2(train_x, y_train, num_bit, Fmap)
trans = trans.astype(np.float32)
#学習画像データと状態変換行列
X_train_i = np.dot(train_x, trans) #>0

# apple_train = X_train_i[0:359,:]
# lemon_train = X_train_i[360:719,:]
# orange_train = X_train_i[720:1079,:]
# np.save('npy/apple_train.npy', apple_train)
# np.save('npy/lemon_train.npy', lemon_train)
# np.save('npy/orange_train.npy', orange_train)

# 0以上を1，それ以外を-1にする
# X_train = X_train_i >0
# X_train = X_train.astype(int)
# signによる識別関数
X_train = np.sign(X_train_i)
X_train = X_train.astype(int)

#X_train = int(X_train)
# バイナリビットに変換しないときの最初の特徴ベクトルを取得
X_train_seq = []
# for j in range(0, 1081, 360):
#     X_train_seq.append(X_train[j])
X_train_seq, indice = np.unique(X_train, axis = 0, return_index = True)
    # X_train_seq = list(dict.fromkeys(map(tuple, X_train)))
X_train_seq = np.array(X_train_seq)
print(X_train_seq)
# 変換したとき重複ないように
#X_train_seq = np.unique(X_train, sort=False, axis = 0)
# X_train_seq = list(dict.fromkeys(map(tuple, X_train)))
# X_train_seq = np.array(X_train_seq)
# print(X_train_seq)
y_train_seq = []
# for j in range(0, 1080, 360):
#     y_train_seq.append(y_train[j])
# print(y_train_seq)
y = y_train[indice]
y_train_seq.append(y)
print(y_train_seq)
# y_train_seq = np.unique(y_train, axis=0)
#X_train = X_train_i.astype(int)
#np.savetxt('txt/my_data.txt', X_train, fmt='%d')

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"実行時間:{minutes}分{seconds}秒")


np.save('npy/trans.npy', trans)
np.save('npy/X_train_seq.npy', X_train_seq)
np.save('npy/y_train_seq.npy', y_train_seq)
