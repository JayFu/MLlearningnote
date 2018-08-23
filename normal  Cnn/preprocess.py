import cv2
from PIL import Image
from CnnN import binaryzation_denoise

# 这个模块用于图片预处理，日后会根据验证码类型扩展
# this part used for preprocess of img, to get the same size img.
# it will be expended later while type of catpatch increase
def img_preprocess(i_text, i_img):
    if i_text[-3:] == 'png':
        # 放大图片
        # resize
        i_img = cv2.copyMakeBorder(i_img, 5, 0, 18, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
        # 图片二值化
        # binarise img
        _, i_img = cv2.threshold(i_img, 30, 255, cv2.THRESH_BINARY)
    elif i_text[-3:] == 'jpg':
        i_img = cv2.copyMakeBorder(i_img, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
        i_img = binaryzation_denoise(i_img)

    return i_img