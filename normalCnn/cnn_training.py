# coding:utf-8

import random
import tensorflow as tf
from datetime import datetime
import numpy as np
import os
import cv2

from cnn_constants import * 
from preprocess import img_preprocesser

TIME_COUNTER = 0

def next_batch(imgs, batch_count=100, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    global TIME_COUNTER
    
    
    TIME_COUNTER += 1

    if batch_count * TIME_COUNTER > len(imgs): 
        TIME_COUNTER = 0

    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])

    # 通过计数器控制每个batch用不同的数据
    for i in range(batch_count * (TIME_COUNTER - 1), batch_count * TIME_COUNTER):
        text = imgs[i]
        img = cv2.imread(os.path.join(DATASET, text),cv2.IMREAD_GRAYSCALE)

        label = text[:4]
        i = i - batch_count * (TIME_COUNTER - 1)
        # 展平图像并归一化
        batch_x[i, :] = img.flatten() / 255
        batch_y[i, :] = text2vec(label)

    return batch_x, batch_y

def next_batch_validation(batch_count=350, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    imgs = os.listdir(VALIDPATH)
    for i in range(batch_count):
        text = imgs[i]
        
        img = cv2.imread(os.path.join(VALIDPATH, text),cv2.IMREAD_GRAYSCALE)

        label = text[:4]
        batch_x[i, :] = img.flatten() / 255
        batch_y[i, :] = text2vec(label)
    return batch_x, batch_y

def text2vec(text, captcha_len = CAPTCHA_LEN, captcha_list = CAPTCHA_LIST):
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError('')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len): 
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector

def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]

    return (''.join(text_list)).lower()

def weight_variable(shape, w_alpha=0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape, b_alpha=0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def cnn_graph(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # relu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # layer 3
    w_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    w_fc = weight_variable([4 * 12 * 64, 1024])
    b_fc = bias_variable([1024])
    h_drop5_re = tf.reshape(h_drop3, [-1, w_fc.get_shape().as_list()[0]])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop5_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # out layer
    w_out = weight_variable([1024, len(captcha_list)*captcha_len])
    b_out = bias_variable([len(captcha_list)*captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv

# 优化器：sigmoid交叉熵计算损失loss
def optimize_graph(y, y_conv):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    # 最小化loss优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
    return optimizer

# 偏差
def accuracy_graph(y, y_conv, width=len(CAPTCHA_LIST), height=CAPTCHA_LEN):
    # 预测值
    predict = tf.reshape(y_conv, [-1, height, width])
    max_predict_idx = tf.argmax(predict, 2)
    # 标签
    label = tf.reshape(y, [-1, height, width])
    max_label_idx = tf.argmax(label, 2)
    correct_p = tf.equal(max_predict_idx, max_label_idx)
    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
    return accuracy

def train(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行

    acc_rate = 0.85
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 优化器
    optimizer = optimize_graph(y, y_conv)
    # 准确率
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    imgs = os.listdir(DATASET)
    imgs = np.random.shuffle(imgs)
    while 1:
        batch_x, batch_y = next_batch(imgs, 100)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.9})
        # 每训练一百次验证一次
        if step % 100 == 0:
            batch_x_validation, batch_y_validation = next_batch_validation(350)
            # tf.Print(batch_y_validation)
            acc = sess.run(accuracy, feed_dict={x: batch_x_validation, y: batch_y_validation, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                saver.save(sess, MODEL_PATH, global_step=step)
                acc_rate += 0.01
                if acc_rate > 0.99: break
        step += 1
    sess.close()

if __name__ == '__main__':
    train()
