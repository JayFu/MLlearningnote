import cv2
import os

from CnnN import binaryzation_denoise
from cnn_constants import *

# 二值化并线降噪
def binaryzation_denoise(img):
    # 二值化
    _, img = cv2.threshold(img, 178, 255, cv2.THRESH_BINARY)

    # 降去盐粒噪声
    img_shape = img.shape
    for y in range(img_shape[0]):
        if y == 0 or y == 255:
            continue
        for x in range(img_shape[1]):
            if x == 0 or x == 255:
                continue
            count = 0
            if img[x, y - 1] > 245:
                count += 1
            if img[x, y + 1] > 245:
                count += 1 
            if img[x - 1, y] > 245:
                count += 1
            if img[x + 1, y] > 245:
                count += 1
            
            if count > 2:
                img[x, y] = 255

    return img

class img_preprocesser:
    def __init__(self):
        self.imgs = []
    
    def read_img_list(self):
        for _, _, files in os.walk(INPUT):
            for name in files:
                if name.endswith('Store') or name.endswith('ignore'):
                    continue
                self.imgs.append(name)

    def img_preprocess(self, img, text):
        if text[5:] == 'png':
            img = cv2.copyMakeBorder(img, 5, 0, 18, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
            _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
            img = binaryzation_denoise(img)

        return img

    def read_img_and_save(self):
        self.read_img_list()

        for name in self.imgs:
            read_path = os.path.join(INPUT, name)
            save_path = os.path.join(DATASET, name)
            img = cv2.imread(read_path)
            img = self.img_preprocess(img, name)
            cv2.imwrite(save_path, img)

            counter = self.imgs.index(name)
            if counter % 100 == 0:
                print('finished: {}/{}'.format(counter, len(self.imgs)))
