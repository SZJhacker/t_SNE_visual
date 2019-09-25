#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def svm_hyperplane(X, y):
    # Standarize features(不明白标准化的意义是什么)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Create support vector classifier
    svc = LinearSVC(C=1.0)

    # Train model
    model = svc.fit(X_std, y)

    # 绘制数据点，不同类别不同颜色
    color = ['orange' if c == 0 else 'blue' for c in y]
    labels = ['negative' if c == 0 else 'positve' for c in y]
    plt.scatter(X_std[:,0], X_std[:,1], c=color)
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    left, right = plt.gca().get_xlim()
    top, bottom = plt.gca().get_ylim()
    plt.xlim((left*1.618, right*1.618))
    plt.ylim((top*1.3, bottom*1.3))
    # plt.title(title, fontsize=12)
    # plt.legend(loc="upper right")

    # 创建超平面(根据SVC的参数，求出直线的斜率与截距)
    w = svc.coef_[0]
    a = - w[0]/w[1]
    xx = np.linspace(-2.5, 2.5)
    yy = a * xx - (svc.intercept_[0])/w[1]

    # Plot the hyperplane
    plt.plot(xx, yy)

    # Plot the hyperplane
    # plt.plot(xx, yy)
    # plt.axis("off"), plt.show()


def tsne_visual(pltdata):
    
    # plt.figure(figsize=(column*6, row*3), dpi=300, facecolor=(1, 1, 1)) #平铺画布，设置dpi300,注意，发表文章dpi不能低于300
    p1 = pltdata[(pltdata.label==1)]
    p2 = pltdata[(pltdata.label==0)]
    x1 = p1.values[:, 0]
    y1 = p1.values[:, 1]
    x2 = p2.values[:, 0]
    y2 = p2.values[:, 1]

    #绘制散点图
    plt.plot(x1, y1, 'o', color="#3dbde2", label='positive', markersize='4')
    plt.plot(x2, y2, 'o', color="#b41f87", label='negative', markersize='4')
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    left, right = plt.gca().get_xlim()
    top, bottom = plt.gca().get_ylim()
    plt.xlim((left*1.618, right*1.618))
    plt.ylim((top*1.3, bottom*1.3))
    # plt.title(title, fontsize=12)
    plt.legend(loc="upper right")

def tsne_data(rawdata):
    """对输入数据进行t-SNE降维,并以矩阵的形式返回"""
    modle = TSNE(n_components=2, random_state=0)

    fea_data = rawdata.drop(columns=['class'])  # 取出所有特征向量用于降维
    redu_fea = modle.fit_transform(fea_data) # 将数据降到2维进行后期的可视化处理
    # labels = rawdata['class'].replace([0, 1], ['negative', 'positive']) # 将正反例数字替换成正反例标签用于收起展示
    labels = rawdata['class']
    redu_data = np.vstack((redu_fea.T, labels.T)).T # 将特征向量和正反例标签整合
    tsne_df = pd.DataFrame(data=redu_data, columns=['Dimension1', 'Dimension2', "label"])
    return  tsne_df # 返回用于可视化的数据


def main():
    parser = argparse.ArgumentParser(description="Visualzes the features based on t-sne in a 2D feature space")
    parser.add_argument('-i', '--infile', help="The input file should be csv format")
    parser.add_argument('-o', '--outfile', help='The name of output file, png ')
    parser.add_argument('-p', '--picturetype', help='The picture type, scatter or svm', default="scatter")
    parser.add_argument('-d', '--dpi', help='The dpi of the savefig, default is 300', default=300)
    parser.add_argument('-l', '--layout', help='The layout of subplots, please input row,column number, default 1,1 ', default='1,1')
    args = parser.parse_args()
    rawdata = args.infile
    csvdata = pd.read_csv(rawdata)
    redimen_data = tsne_data(csvdata)
    # redimen_data.to_csv(args.outfile)
    # tsne_visual(redimen_data)
    if args.picturetype == 'scatter':
        tsne_visual(redimen_data)
    else:
        svm_hyperplane(redimen_data.fro(columns=['class']), list(redimen_data.label))
    plt.savefig(args.outfile, args.dpi)    

if __name__ == "__main__":
    main()