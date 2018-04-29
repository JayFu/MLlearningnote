# 根据此前R语言所完成的内容，使用sklearn对心脏疾病进行预测
from sklearn.feature_extraction import DictVectorizer  
import csv  
from sklearn import preprocessing  
from sklearn import tree  
from sklearn.externals.six import StringIO  
# from __builtin__ import str  
from _csv import reader  
from docutils.nodes import header

HeartDeseaseData = open(r'/Users/jay_fu/MLlearningnote/Heart.csv','rt')
reader = csv.reader(HeartDeseaseData)
headers=next(reader)
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
lb=preprocessing.LabelBinarizer()  
dummyY=lb.fit_transform(labelList)  
print("dummyY:"+str(dummyY))  
  
clf=tree.DecisionTreeClassifier(criterion='entropy')  
clf=clf.fit(dummyX, dummyY)  
print("clf:"+str(clf))  
  
with open("AllElectronics.dot",'w')as f:  
    f=tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)  
  
#predict  
oneRowX=dummyX[0,:]  
print("oneRowX:"+str(oneRowX))  
  
newRowX=oneRowX  
newRowX[0]=1  
newRowX[2]=0  
print("newRowX:"+str(newRowX))  
predictedY=clf.predict([newRowX])  
print("predictedY:"+str(predictedY))    

# 以上是决策树内容

# 随机森林判断

