import numpy
import operator
import matplotlib
import  matplotlib.pyplot as plt



def createDataSet():
    group = numpy.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sqDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        soredClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()  # 读取文件
    numberOfLines = len(arrayOfLines)   #计算函数
    returnMat = numpy.zeros((numberOfLines,3))   #将数据拆分成矩阵，m行3列
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)   #取出每一列的最小值
    maxVals=dataSet.max(0)      #取出每一列的最大值
    ranges=maxVals-minVals
    normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet=dataSet-numpy.tile(minVals,(m,1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

