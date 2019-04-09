import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter

K = 10

def E_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    distance = math.sqrt(sum(diffs_squared_distance))
    # print(distance)
    return distance
def K_NN(input, dataset, lable, k):
    # predict_label = '1'
    distance_list = []
    for i in range(len(dataset)):
        distance_list.append([E_distance(input, dataset[i]), lable[i]])
    
    distance_sorted = [i[1] for i in sorted(distance_list)]
    # print(distance_sorted)
    top_nearest = distance_sorted[:k]
    predict_label = Counter(top_nearest).most_common(1)[0][0]

    return predict_label


if __name__ == "__main__":

    # import iris data and devided into train and vali
    iris = load_iris()
    # datasets = irisdata.data
    # labels = irisdata.target
    # traindata = datasets[:120]
    # trainlabel = labels[:120]
    # validdata = datasets[121:]
    # validlabel = labels[121:]
    traindata, validdata, trainlabel, validlabel = train_test_split(iris.data, 
        iris.target, test_size=0.4, random_state=1)
    # train = np.array(zip(traindata, trainlabel))
    # valid = np.array(zip(validdata, validlabel))


    correct_counter = 0
    for i in range(len(validdata)):
        predict_label = K_NN(validdata[i], traindata, trainlabel, K)
        if str(validlabel[i]) == str(predict_label):
            correct_counter = correct_counter + 1
        print("correct answer: %d, prediction: %d"%(validlabel[i], predict_label))
    print("accuracy:", correct_counter/len(validdata))