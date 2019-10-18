#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def make_meshgrid(x, y, h=.001):

    x_min, x_max = x.min() - 0.3, x.max() + 0.3
    y_min, y_max = y.min() - 0.3, y.max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def svc_rbf(X, y):
    """利用网格搜索算法构建最优参数的高斯svm模型"""

    svm_rbf = SVC(kernel='rbf', probability=True)

    #列出要调整的参数和候选值
    para_grid = {
        'gamma': [2**x for x in range(-5, 15, 2)],
        "C": [2**x for x in range(-15, 3, 2)]
    }

    # 实例化一个GridSearchCV的类
    grid_search = GridSearchCV(svm_rbf, para_grid, cv=5)

    # 训练寻找最优参数
    grid_search.fit(X, y)

    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    return grid_search.best_estimator_ 

def scale(data):
    "数据进行归一化处理"

    # 标签取出，数据进行归一化
    fea_data = data.drop(columns=['label'])
    labels = data['label']
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(fea_data.values)
    clf = svc_rbf(norm_data, labels)
    norm_array = np.vstack((norm_data.T, labels.T)).T
    norm_frame = pd.DataFrame(data=norm_array, columns=['Dimension1', 'Dimension2', "label"])
    X0, X1 = norm_data[:, 0], norm_data[:, 1]
    return X0, X1, norm_frame, clf

def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)

def scatter_visual(pltdata):
    """对t-SNE降维的数据进行可视化"""

    p1 = pltdata[(pltdata.label == 1)]
    p2 = pltdata[(pltdata.label == 0)]
    x1 = p1.values[:, 0]
    y1 = p1.values[:, 1]
    x2 = p2.values[:, 0]
    y2 = p2.values[:, 1]

    # 绘制散点图
    plt.plot(x1, y1, 'o', color="#3dbde2", label='positive', markersize='3')
    plt.plot(x2, y2, 'o', color="#b41f87", label='negative', markersize='3')
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    # plt.margins(0.3)
    plt.legend(loc="upper right")


def main():
    parser = argparse.ArgumentParser(description="Draw decision boundary of various classifier")
    parser.add_argument('-i', '--infile', help="The input file should be csv format, and multiple file should be separated by commas", required=True)
    parser.add_argument('-o', '--outfile', help='The name of output picture', required=True)
    parser.add_argument('-l', '--layout', help='The layout of subplots, please input row,column number, default 1,1 ', default='1,1')
    parser.add_argument('-d', '--dpi', help='The dpi of output picture, default is 300dpi', default=300)
    parser.add_argument('-f', '--classifier', help='The classifier decides the decision boundary', default='svm')
    args = parser.parse_args()

    infile_list = args.infile.split(',') # 获得文件名列表
    row, column = [int(i) for i in args.layout.split(',')] 
    assert len(infile_list) <= row*column, "The number of inputfile to be drawn is not greater than the number of subplots of the canvas, please set '-l' parameter"

    plt.figure(figsize=(column * 4.7, row * 3))  #设置画布的尺寸

    # 遍历所有文件，进行降维，绘图处理，使用数字索引是为了确定子画布的位置
    for i in range(len(infile_list)):
        plt.subplot(row, column, i + 1)
        csvdata = pd.read_csv(infile_list[i])
        X0, X1, normdata, clf = scale(csvdata) # 数据归一化处理, 并建立svm模型
        xx, yy = make_meshgrid(X0, X1)      
        plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
        scatter_visual(normdata)
    
    # 调节子图之间的距离     
    plt.subplots_adjust(
        top=0.92, bottom=0.20, left=0.15, right=0.95, hspace=0.20,
        wspace=0.20)  
    plt.savefig(args.outfile, dpi=args.dpi)


if __name__ == "__main__":
    main()