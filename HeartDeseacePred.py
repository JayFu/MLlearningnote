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
headers=reader.next()  
# print(headers)  
  
featureList=[]  
labelList=[]  
  
for row in reader:  
    labelList.append(row[len(row)-1])  
    rowDict={}  
    for i in range(1,len(row)-1):  
        rowDict[headers[i]]=row[i]  
    featureList.append(rowDict)  
# print(featureList) 

vec=DictVectorizer()  
dummyX=vec.fit_transform(featureList).toarray()  
print("dummyX:"+str(dummyX))  
print(vec.get_feature_names())  
print("labelList:"+str(labelList))  