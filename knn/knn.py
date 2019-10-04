# 作者 ：高海龙
import numpy as np
import operator  # 比较函数库


# knn 的基本算法
# inX 类似[0,0],分类 dataSet 是numpy的一个矩阵，labels 是特征标签（= 矩阵行数），k是要取的特征值个数
def classify0(inX, dataSet, labels, k):
    datasetSize = dataSet.shape[0]  # 0返回垂直方向的长度，1 返回水平方向的长度
    diffMat = np.tile(inX, (datasetSize, 1)) - dataSet  # tile  numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次
    print(diffMat)
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 将所有的值加起来,设axis=i ,则numpy沿着第i个下标变化的方向进行操作
    distances = sqDistances ** 0.5  # 权重
    sortDisIndicies = distances.argsort()  # argsort函数返回的是数组值从小到大的索引值
    classCount = {}  # 分类
    for i in range(k):  # 选择距离最小的k个点
        votelabel = labels[sortDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # items() 函数以列表返回可遍历的(键, 值) 元组数组。
    # itemgetter 用于获取对象的哪些位置的数据，参数即为代表位置的序号值
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def test():
    a = np.array([[1, 2, 3], [1, 2, 3]])
    print(a.shape[1])


if __name__ == '__main__':
    test()
