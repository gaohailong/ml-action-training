# 作者 ：高海龙
import numpy as np
from knn.knn import *
import os


# 读取文件并解析
def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # strip() 方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列。
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化（公式：newValue = (oldValue-min)/(max-min)）
def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试代码
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(os.getcwd() + 'test/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classfifier came back with :%d,the real answer is :%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is %f" % (errorCount / float(numTestVecs)))


# 正式测试程序
def classifyPerson():
    resulList = ['not at all', 'in samll doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per years?"))
    datingDataMat, datingLabels = file2matrix(os.getcwd() + 'test/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", resulList[classifierResult - 1])


if __name__ == '__main__':
    classifyPerson()
    # print(os.getcwd()+'test/datingTestSet2.txt')
