import numpy as np
import cv2
import math
import time
from scipy.spatial import distance

def hammingdis(V, X):
    #w, h = V.shape
    #V = V.reshape(h,)
    #n1 = V.shape[0]
    n2, bits = X.shape

    Dh = np.zeros((1, n2))
    for n in range(n2):
        x = X[n, :]
        #x = x.reshape(1, -1)
        xor_result = np.bitwise_xor(V,x)
        dis = np.count_nonzero(xor_result)
        # dis = distance.hamming(V, x)
        # dis = dis * len(x)
        Dh[:,n] += dis
    return Dh

#ユークリッド距離
def euclidean(V,X):
    n2, bits = X.shape

    Dh = np.zeros((1, n2))
    for n in range(n2):
        x = X[n, :]
        dis = np.sqrt(np.sum((V-x) ** 2))
        Dh[:,n] += dis
    return Dh
    
def hamming_dis(V, x):
    # ビット列をNumPyの配列に変換
    xor_result = np.bitwise_xor(V,x)
    dis = np.count_nonzero(xor_result)
    return dis
    
def cos(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos

#二乗距離
def sqdist(X, Y):
    dist = np.sum((X[:, np.newaxis]-Y) ** 2, axis=-1)
    return dist

#　ハッシングで求めた領域で特徴ベクトルとの距離を求めて姿勢を求める
def rotate_a(features, apple_train):
    ham = euclidean(features, apple_train)
    ham_sort_index = np.argsort(ham, axis=None)
    #ham_sort = ham.flatten()[ham_sort_index]
    pos = ham_sort_index[0]
    return pos
def rotate_l(features, lemon_train):
    ham = euclidean(features, lemon_train)
    ham_sort_index = np.argsort(ham, axis=None)
    #ham_sort = ham.flatten()[ham_sort_index]
    pos = ham_sort_index[0]
    return pos
def rotate_o(features, orange_train):
    ham = euclidean(features, orange_train)
    ham_sort_index = np.argsort(ham, axis=None)
    #ham_sort = ham.flatten()[ham_sort_index]
    pos = ham_sort_index[0]
    return pos

# 量子化誤差
def quantization_error(feature, binary_feature):
    dis = np.sqrt(np.sum((feature-binary_feature) ** 2))
    return dis

# バウンディングボックス
def draw_boxes(img, boxes, i, location):
    dst = img.copy()
    for x1, y1, x2, y2 in boxes:
        location.append((i, x1, y1))
        if i == 0:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        elif i == 1:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,255,255), thickness=3)
        else:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,200,255), thickness=3)
    #print("number of boxes", len(boxes))
    return dst, location
# nms
def nms(boxes, scores, overlap_thresh):
    if len(boxes) <= 1:
        return boxes

    # float 型に変換する。
    #boxes = boxes.astype("float")

    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(-scores)  # スコアを降順にソートしたインデックス一覧
    selected = []  # NMS により選択されたインデックス一覧

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # indices は降順にソートされているので、一番最後の要素の値 (インデックス) が
        # 残っている中で最もスコアが高い。
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        #overlap = (i_w * i_h) / area[remaining_indices]
        overlap = (i_w * i_h) / (2 * (area[remaining_indices] - i_w * i_h))

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # 選択された短形の一覧を返す。
    return boxes[selected].astype("int")

# 物体の姿勢を求める
# def pose():
#     for x1, y1, x2, y2 in boxes:
#         location = y1 * wide + x1
#         feature = F[location,:]
#         if i == 0:

def draw_boxespose(img, boxes, i, loc):
    dst = img.copy()
    for x1, y1, x2, y2 in boxes:
        loc.append((i, x1, y1))
        location = y1 * h1 + x1
        feature = F[location,:]
        # 中心位置
        x_center, y_center = (x1+x2)//2, (y1+y2)//2
        line_length = 50
        if i == 0:
            pos = rotate_a(feature, apple_train)
            posi = np.radians(pos)
            x_end = int(x_center - line_length * np.sin(posi))
            y_end = int(y_center - line_length * np.cos(posi))
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.line(dst, (x_center, y_center), (x_end, y_end), color=(0,0,255), thickness=2)
        elif i == 1:
            pos = rotate_l(feature, lemon_train)
            posi = np.radians(pos)
            x_end = int(x_center - line_length * np.sin(posi))
            y_end = int(y_center - line_length * np.cos(posi))
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,255,255), thickness=2)
            cv2.line(dst, (x_center, y_center), (x_end, y_end), color = (50, 255, 255), thickness=2)
        else:
            pos = rotate_o(feature, orange_train)
            posi = np.radians(pos)
            x_end = int(x_center - line_length * np.sin(posi))
            y_end = int(y_center - line_length * np.cos(posi))
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,200,255), thickness=2)
            cv2.line(dst, (x_center, y_center), (x_end, y_end), color=(50,200,255), thickness=2)
    #print("number of boxes", len(boxes))
    return dst, loc

def cal_iou(true_location, prediction, w_e, h_e):
    dis = 1000
    index = 0
    x2, y2 = prediction
    for i in range(len(true_location)):
        x_1, y_1 = true_location[i]
        distance = abs(x_1-x2) + abs(y_1-y2)
        if distance <=dis:
            dis = distance
            index = i
    x1, y1 = true_location[index]
    w, h = w_e, h_e
    # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
    i_x1 = max(x1, x2)
    i_y1 = max(y1, y2)
    i_x2 = min(x1+w, x2+w)
    i_y2 = min(y1+h, y2+h)

    # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
    # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
    i_w = max(0, i_x2 - i_x1 + 1)
    i_h = max(0, i_y2 - i_y1 + 1)

    # 選択した短形と残りの短形の Overlap Ratio を計算する。
    #overlap = (i_w * i_h) / area[remaining_indices]
    overlap = (i_w * i_h) / (2 * (w * h) - i_w * i_h)
    return overlap, index

def pixel_dis(true_location, prediction):
    x1, y1 = true_location
    x2, y2 = prediction
    distance = abs(x1-x2) + abs(y1-y2)
    return distance

def accuracy(correct, prediction, thre_accu, distance_thre):
    correct_label = correct[:,0]
    correct_location = correct[:, 1:]
    prediction_label = prediction[:,0]
    prediction_location = prediction[:,1:]
    location_count = 0
    label_count = 0
    if prediction_label.size < correct_label.size:
        total = correct_label.size
    else:
        total = prediction_label.size
    loc_total = correct_label.size
    dis = []
    total_dis = 0
    # for i, (true, pre) in enumerate(zip(correct_location, prediction_location)):
    #     iou = cal_iou(true, pre, w_e, h_e)
    #     if iou >= thre_accu:
    #         distance = pixel_dis(true, pre)
    #         dis.append(distance)
    #         if distance <= distance_thre:
    #             location_count += 1
    #         if correct_label[i] == prediction_label[i]:
    #             label_count += 1
    for i, pre in enumerate(prediction_location):
        iou, index = cal_iou(correct_location, pre, w_e, h_e)
        true = correct_location[index]
        if iou >= thre_accu:
            # distance = pixel_dis(true, pre)
            # total_dis += distance
            # dis.append(distance)
            # if distance <= distance_thre:
            #     location_count += 1
            if correct_label[index] == prediction_label[i]:
                label_count += 1
                distance = pixel_dis(true, pre)
                total_dis += distance
                dis.append(distance)
                if distance <= distance_thre:
                    location_count += 1
    total_dis = total_dis / len(dis)
    # 位置の正確率
    accuracy_location = (location_count / loc_total) * 100
    # 分類の正解率
    accuracy_label = (label_count / total) * 100
    return accuracy_location, accuracy_label, dis, total_dis

eigenFaces = np.load('npy/eigen1_512.npy')
train_x = np.load('npy/train_x.npy')
y_train = np.load('npy/y_train.npy')
apple_train = np.load('npy/apple_train.npy')
lemon_train = np.load('npy/lemon_train.npy')
orange_train = np.load('npy/orange_train.npy')
X_train_seq = np.load('npy/X_train_seq.npy')
y_train_seq = np.load('npy/y_train_seq.npy')
trans = np.load('npy/trans.npy')

total_time = 0
total_loc_acc = 0
total_acc = 0
total_distance = 0
number = 20
for j in range(number):
    start_time = time.time()
    #実装
    # image = cv2.imread(f'test/input1/input_{j}.jpg')
    image = cv2.imread(f'test/input_back1/input_{j}.jpg')
    #image = image_c
    #image = np.float32(image_back) / 255
    image_c = np.float32(image) / 255
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('image.jpg', image_back)
    #img = image_back.copy()
    img = image.copy()
    results = []
    for eigenFace in eigenFaces:
        res = cv2.matchTemplate(image_c, eigenFace, cv2.TM_CCORR)
        #res = cv2.matchTemplate(image, eigenFace, cv2.TM_CCOEFF)
        result = res.flatten()
        results.append(result)
    w_e, h_e, c = eigenFace.shape
    results = np.array(results)
    w1, h1 = res.shape
    w, h = results.shape

    F = np.transpose(results)
    #  # #カーネル
    # Ntrain = train_x.shape[0]
    # n_anchor = Ntrain
    # index = np.random.choice(Ntrain, n_anchor, replace=False)
    # X_anchor = train_x[index,:]
    # #ガウスカーネル
    # Vsigma = 0.3
    # features_k = np.exp(-sqdist(F, X_anchor) / (2*Vsigma*Vsigma))
    #バイナリビット生成
    vH_i = np.dot(F, trans) #>0
    # vH = vH_i > 0
    # vH = vH.astype(int)

    # 符号関数
    vH = np.sign(vH_i)
    vH = vH.astype(int)
    # np.savetxt('txt/vH.txt', vH, fmt='%d')
    # #vH = vH.reshape(1, -1)

    classify = []
    apple = []
    orange = []
    lemon = []
    apple_posi = {}
    lemon_posi = {}
    orange_posi = {}
    posi = []
    count = 0
    #start_time = time.time()
    # for i in range(h):        #ピクセル分繰り返す
    i = 0
    while i < h:
        
        #1ピクセルごとにバイナリビット実行
        #features = np.array(features)
        fea = F[i, :]
        # バイナリビットに変更する前
        feature = vH_i[i,:]
        # バイナリビットに変更後
        features = vH[i,:]
        
        # #学習データとハミング距離を比較する
        # #ham = hammingdis(features, X_train_seq)
        # ham = euclidean(features, X_train_seq)
        # ham_sort_index = np.argsort(ham, axis=None)
        # #ham_sort_index = np.argsort(ham, axis=None)[::-1]
        # ham_sort = ham.flatten()[ham_sort_index]

        n2, bits = X_train_seq.shape

        for n in range(n2):
            x = X_train_seq[n, :]
            #x = X_train_seq[n][:]
            res = np.bitwise_xor(features,x)
            dis = np.count_nonzero(res)
            # dis = distance.hamming(features, x)
            # dis = dis * len(x)
            if dis == 0:
                true = 1
                cor_label = y_train_seq[0][n]
                break
            else:
                true = 0
                cor_label = 4


        threshold = 0.4
        # if i%18720 == 0:
        #     print(ham_sort[0])
        #if ham_sort[0] > threshold :
        if true == 0:
            classify.append(4)
            i += 1
        elif cor_label == 3:
            classify.append(3)
            i += 1
        else:
            #cor_label = y_train_seq[ham_sort_index[0]]
            classify.append(cor_label)
            if cor_label == 0:
                apple_error = quantization_error(feature, features)
                if apple_error <= threshold: 
                #rect, pos = rotate_a(fea, apple_train, thre)
                #
                #print(i)
                    count += 1
                    x1 = int(i % h1)
                    y1 = int(i / h1)
                    apple_posi[apple_error] = np.array([x1, y1, x1+w_e, y1+w_e])
                    #apple_posi[ham_sort[0]] = np.array([x1, y1, x1+w_e, y1+w_e])
                    image = cv2.rectangle(image, (x1, y1), (x1+w_e, y1+h_e), color = (0,0,255), thickness=4)
                i += 1
            elif cor_label == 1:
                lemon_error = quantization_error(feature, features)
                if lemon_error <= threshold:
                # rect, pos = rotate_l(fea, lemon_train, thre)
                # if rect == True:
                #print(i)
                    count += 1
                    x1 = int(i % h1)
                    y1 = int(i / h1)
                    lemon_posi[lemon_error] = np.array([x1, y1, x1+w_e, y1+w_e])
                    #lemon_posi[ham_sort[0]] = np.array([x1, y1, x1+w_e, y1+w_e])
                    image = cv2.rectangle(image, (x1, y1), (x1+w_e, y1+h_e), color = (50,255,255), thickness=4)
                i += 1     
            elif cor_label == 2:
                orange_error = quantization_error(feature, features)
                if orange_error <= threshold:
                # rect, pos = rotate_o(fea, orange_train, thre)
                # if rect == True:
                #print(i)
                    count += 1
                    x1 = int(i % h1)
                    y1 = int(i / h1)
                    orange_posi[orange_error] = np.array([x1, y1, x1+w_e, y1+w_e])
                    #orange_posi[ham_sort[0]] = np.array([x1, y1, x1+w_e, y1+w_e])
                    image = cv2.rectangle(image, (x1, y1), (x1+w_e, y1+h_e), color = (50,200,255), thickness=4)
                i += 1
    #nms スコア量子化誤差
    posi.append(apple_posi)
    posi.append(lemon_posi)
    posi.append(orange_posi)
    location = []
    overlap_thresh = 0.5
    for k, pos in enumerate(posi):
        scores = np.array(list(pos.keys()))
        boxes = np.array(list(pos.values()))
        box = nms(boxes, scores, overlap_thresh)
        img, location = draw_boxes(img, box, k, location)
        #img, location = draw_boxespose(img, box, i, location)

    # nms ユークリッド距離
    # posi.append(apple_posi)
    # posi.append(lemon_posi)
    # posi.append(orange_posi)
    # overlap_thresh = 0.5
    # for i, pos in enumerate(posi):
    #     scores = np.array(list(pos.keys()))
    #     boxes = np.array(list(pos.values()))
    #     box = nms(boxes, scores, overlap_thresh)
    #     img = draw_boxes(img, box, i)


    #実行時間
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    #print("実行時間:", elapsed_time)
    # print(count)
    #im = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('out.jpg', img)
    cv2.imwrite(f'result/output/output_{j}.jpg', image)
    cv2.imwrite(f'result/out/out_{j}.jpg', img)
    # cv2.imwrite(f'result/output_back/output_{j}.jpg', image)
    # cv2.imwrite(f'result/out_back/out_{j}.jpg', img)
    # cv2.imshow("Img",image_c)
    # cv2.waitKey()

    # 正解位置
    loca = np.load('npy/location.npy')
    correct = loca[j]
    #correct = np.load('npy/placed_objects.npy')
    # 位置
    location = np.array(location)
    sort_index = np.lexsort((location[:,1], location[:,2]))
    prediction = location[sort_index]
    # 閾値
    accu_thre = 0.5
    distance_thre = 5
    accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
    total_distance += total_dis
    # print(location)
    # print(correct)
    # print(prediction)
    print(dis)
    # print("位置の正解率:", accuracy_location)
    # print("正解率：", accuracy_label)
    total_loc_acc += accuracy_location
    total_acc += accuracy_label

average_time = total_time / number
average_pixel = total_distance / number
average_loc_acc = total_loc_acc / number
average_acc = total_acc / number
print('平均時間：', average_time)
print('平均ピクセル誤差：', average_pixel)
print('平均位置精度：', average_loc_acc)
print('平均精度：', average_acc)

    #np.set_printoptions(threshold=np.inf)
    #print(classify_x)
    # V = np.array(F)       #リストを行列に変換
    # V_ = V.transpose(1,0)  #軸を入れ替える
    # F_W = hadamard2()