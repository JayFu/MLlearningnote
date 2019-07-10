# coding:utf-8
import numpy as np
import pickle
import os

CIFAR_PATH = './cifar-10-batches-py/'
CONFIG_PATH = CIFAR_PATH + 'batches.meta'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.
        
def read_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

if __name__ == "__main__":
    batch_list = []
    # batch_list = [f for _, _, f in os.walk(CIFAR_PATH)]
    for r, d, f in os.walk(CIFAR_PATH):
        print(f)
        for ff in f:
            if ff[:4]=='data':
                batch_list.append(os.path.join(r, ff))
    print(batch_list)

    Classfier = network(3)

    for batch in batch_list:
        X = read_data(batch)
        print("now the {} is being training".format(X['batch_label']))
        
        