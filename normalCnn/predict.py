# coding:utf-8

import os, cv2, shutil
import numpy as np
import tensorflow as tf
from CnnN import cnn_graph, vec2text, binaryzation_denoise
from CnnN import CAPTCHA_HEIGHT, CAPTCHA_LEN, CAPTCHA_LIST, CAPTCHA_WIDTH
from preprocess import img_preprocess

# 把需要识别的验证码图片放在这个路径下
DATA_DIR = ''
# 把识别过后的验证码图片放在这个路径下
BIN_DIR = ''


def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    # 该函数主要负责识别验证码中的文本，先清理tf框架图
    # transform captcha img to text    
    # clean the tf graph
    tf.reset_default_graph()
    # 给x和丢失率设定占位符
    # set placeholder for x and dropout rate
    x = tf.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.placeholder(tf.float32)
    # 从前一份代码中导入卷积神经网络
    # import network
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 从前一份代码中导入存储器
    # import saver
    saver = tf.train.Saver()
    kkk = os.getcwd()
        
    # 开始会话
    # start session
    with tf.Session() as sess:
        # 导入已经训练好的模型
        # import modol
        print(kkk)
        saver.restore(sess, tf.train.latest_checkpoint(kkk))

        # 设置预测函数
        # set predict
        predict = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        # 将图片传入会话并运行，得到一个向量类型的预测结果
        # run session by predict and img feed, get a vector which includes the ending
        vector_list = sess.run(predict, feed_dict={x: image_list, keep_prob: 1})
        # 把向量类型结果转为验证码文本并返回
        # transform into list and catpatch text
        vector_list = vector_list.tolist()
        text_list = [vec2text(vector) for vector in vector_list]
        return text_list

def predict(data_d = DATA_DIR, bin_d = BIN_DIR):
    # 用于计算正确率
    # count accuracy
    rightnum = 0
    # 文件操作：传入的文件保存在/test中，预测之后的文件放在/trash_bin中
    # file operation, 
    # img stored in /test, after predict, put img in /trash_bin. 

    file_type_list = ['jpg', 'png', 'JPG', 'PNG']
    # pre_list = []
    data_dir = data_d
    bin_dir = bin_d
    # 读取需要预测的图片
    # read folder list to get imgs
    imgs = os.listdir(data_dir)
    for i in range(len(imgs)):
        # 在验证步骤中，文件名即是正确的验证码文本，因此这一步读取文件名主要用于对比预测结果并计算正确率
        # 实际操作中，没有必要计算正确率，但是读取文件名在文件操作同样需要，因此保留
        # read the file name for the next operation
        # however, in valication case, this step means read the label, which is used for counting accuracy 
        # while in predicting part, counting accuracy is needless, but it can be kept
        text = imgs[i]
        file_type = text[-3:]
        if file_type not in file_type_list:
            continue
        print(text)
        img_file = data_dir + text
        # 读取图片并转为灰度图
        # read image and transform into gray img
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) 
        # 图片预处理，需要随着验证码图片的增加更新而升级
        # img preprocess, waiting for upgrate
        img = img_preprocess(text, img)
        # 将图片转为array
        # transform into array
        img = np.array(img)
        # 获得标签，用于验证，实际操作并不需要
        # get the label, for validation
        tet = text[:4]
        # 将图片展开并归一化
        # flatten the img and normalization
        image = img.flatten() / 255
        # 调用预测函数预测图片上的验证码
        # predict the catpatch text in img
        pre_text = captcha2text([image])
        # 将预测结果打印出来。这一步将会改成返回预测结果
        # print predict ending, used conpairing predict ending and right answer.
        # this step will change into send frontend the predict ending
        print('Label:', tet, ' Predict:', pre_text)
        # 计算正确率·
        # count accuracy
        if tet == pre_text[0]: rightnum += 1
        # 判断垃圾桶中是否有重复文件，如果没有，将预测过的图片丢入垃圾桶
        # through the used img into trash bin only if there was no repeat file in the bin
        if os.path.exists(bin_dir + '/' + text):
            pass
        else:
            shutil.move(img_file, bin_dir)
        # pre_list.append(pre_text)
    # 打印正确率
    # print accuracy
    # print('accuracy:', rightnum/60)
    return pre_text

#******************************************************************************************
    # 这里是一个回收利用垃圾箱中数据的模块，即将这些图片放入训练集进行训练。因为每个批次输入的数据量为100，因此每次垃圾箱中数据达到100，就将数据回收
    # 但是目前预测正确率不能达到100%，这个时候贸然将数据放入训练集将会污染模型。
    # 在等待前端修正后放入垃圾箱，再回收至训练集。
    # this part is for recycling the trash----put them into training set
    # because every batch for traing has 100 imgs, so every time there are 100 in trash bin, recycle them
    # however, the accuracy for predict havenot touched 100%
    # so the trash bin cannot pour into training set immediately. after frontend finished, rewrite this part
    
    # tmp_files = os.listdir(bin_dir)
    # if (len(tmp_files) >= 100):
    #     for i in range(100):
    #         tmp_pwd = bin_dir + tmp_files 
    #         shutil.move(tmp_pwd, training_dir)
#******************************************************************************************

if __name__ == '__main__':
    predict()
