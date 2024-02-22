import cv2

img = cv2.imread('image/apple.jpg', 1)


h, w, _ = img.shape[:3]

cw = w // 2
ch = h // 2
center = (cw,ch)
for i in range(0, 360):
    trans = cv2.getRotationMatrix2D(center, i*1.0, 1.0)
    img_ = cv2.warpAffine(img, trans,(w,h))
    img_trim = img_[19:103,18:102]
    cv2.imwrite('tra/apple/apple_'+ str(i) +'.jpg', img_trim)

img1 = cv2.imread('image/orange.jpg', 1)
h, w, _ = img1.shape[:3]

cw = w // 2
ch = h // 2
center = (cw,ch)
for i in range(0, 360):
    trans = cv2.getRotationMatrix2D(center, i*1.0, 1.0)
    img_1 = cv2.warpAffine(img1, trans,(w,h))
    img_trim1 = img_1[19:103,21:105]
    cv2.imwrite('tra/orange/orange_'+ str(i) +'.jpg', img_trim1)

# img2 = cv2.imread('train1/lemon.jpg', 1)
# h, w, _ = img2.shape[:3]

# cw = w // 2
# ch = h // 2
# center = (68,65)
# for i in range(0, 360):
#     trans = cv2.getRotationMatrix2D(center, i*1.0, 1.0)
#     img_2 = cv2.warpAffine(img2, trans,(w,h))
#     img_trim2 = img_2[25:109,25:109]
#     cv2.imwrite('tra/lemon/lemon_'+ str(i) +'.jpg', img_trim2)

