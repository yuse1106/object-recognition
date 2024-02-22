import cv2
import os
import glob
import random 
import numpy as np

def check(new_x, new_y, object_x, object_y, placed_objects):
    # for(x, y, width, height, label) in placed_objects:
    #     if(new_x < x + width and new_x + object_x > x and
    #        new_y < y + height and new_y + object_y > y):
    #         return True
    #     return False
    count = 0
    for label, x, y in placed_objects:
        if(new_x < x + object_x and new_x + object_x > x and
           new_y < y + object_y and new_y + object_y > y):
            count += 1
    if count == 0:
        return True
    else:
        return False
    
if __name__ == '__main__':
    location = []
    for i in range(50):
        # 背景画像のサイズ
        back_height, back_width = 400, 400
        # 背景画像を真っ白に
        back_img = np.ones((back_height, back_width, 3), dtype=np.uint8) * 255

        path1 = 'tra1'
        path_list = glob.glob(path1+'/*')
        # 配置済みの位置
        placed_objects = []
        # placed_objects = {}
        for label, pic_path in enumerate(path_list):
            list = glob.glob(pic_path+'/*.jpg')
            for j in range(3):
                data = random.choice(list)
                select_img = cv2.imread(data)
                # cv2.imshow('img', select_img)
                # cv2.waitKey(0)
                # 物体画像のサイズ
                object_height, object_width, c = select_img.shape

                # 物体が重ならないように配置
                while True:
                    x_position = random.randint(0, back_width - object_width)
                    y_position = random.randint(0, back_height - object_height)

                    # 重なり
                    if check(x_position, y_position, object_width, object_height, placed_objects):
                        break
                # 物体の位置追加
                # placed_objects.append((x_position, y_position, object_width, object_height, label))
                placed_objects.append((label, x_position, y_position))
                # 背景画像に物体を配置する
                back_img[y_position:y_position + object_height, x_position:x_position + object_width] = select_img

        placed_objects = np.array(placed_objects)
        sort_index = np.lexsort((placed_objects[:,1], placed_objects[:,2]))
        placed_objects_sort = placed_objects[sort_index]
        location.append(placed_objects_sort)
        #print(location[0])
        np.save('npy/placed_objects.npy', placed_objects_sort)
        # cv2.imshow("out", back_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f'test/input1/input_{i}.jpg', back_img)
        #     # ディレクトリ内の物体画像のファイルリストを取得
        # object_images_list = [f for f in os.listdir(objects_directory) if os.path.isfile(os.path.join(objects_directory, f))]
    np.save('npy/location.npy', location)
    print(location)
    print('a',location[0])