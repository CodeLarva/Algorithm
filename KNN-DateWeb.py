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
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    soredClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return soredClassCount[0][0]   #返回类别最多的类别


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

def datingClassTest():
    hoRatio = 0.1  #取出10%的数据作为测试数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  #将数据进行分类
    normMat, ranges, minVals = autoNorm(datingDataMat) #归一化数据
    m=normMat.shape[0]            #算出数据的从行数
    numTestVecs = int(m*hoRatio)  #算出测试数据个数
    errorCount=0.0  #错误数量
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))


datingClassTest()