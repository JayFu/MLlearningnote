# 根据此前R语言所完成的内容，使用sklearn对心脏疾病进行预测

from __future__ import print_function
import sklearn
import numpy
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
import csv
from _csv import reader

HeartDeseaseData = open(r'/Users/jay_fu/MLlearningnote/Heart.csv','rb')
reader = csv.reader(HeartDeseaseData)