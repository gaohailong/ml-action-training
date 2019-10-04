# 作者 ：高海龙
import matplotlib.pyplot as plt
from knn.appointment.appoint import *
import os
import numpy as np


# 显示数据的图
def matShow():
    path = os.path.dirname(os.getcwd()) + '/appointment/test/datingTestSet2.txt'
    datingDataMat, datingLabels = file2matrix(path)
    flg = plt.figure()
    ax = flg.add_subplot(111)  # 就是在一张figure里面生成多张子图
    # scatter(x, y, 点的大小, 颜色，标记)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()


if __name__ == '__main__':
    matShow()
