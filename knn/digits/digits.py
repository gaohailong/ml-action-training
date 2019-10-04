# 作者 ：高海龙
import numpy as np
import os
from knn.knn import *


# 初始化数据格式
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0 * 32 * i + j] = int(lineStr[j])
    return returnVect


def handwrtingClassTest():
    hwLabels = []
    traningFileList = os.listdir('训练数据集合')
    m = len(traningFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = traningFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('测试数据集')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d,the real answer is :%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is:%d" % errorCount)
    print("\nthe total error rate is :%f" % (errorCount / float(mTest)))
