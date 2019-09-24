#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def tsne_visual(pltdata):

    p1 = pltdata[(pltdata.label=="positive")]
    p2 = pltdata[(pltdata.label=="negative")]
    x1 = p1.values[:, 0]
    y1 = p1.values[:, 1]
    x2 = p2.values[:, 0]
    y2 = p2.values[:, 1]

    #绘制散点图
    plt.plot(x1, y1, 'o', color='#EE7621', label='positive', markersize='4')
    plt.plot(x2, y2, 'o', color='#36648B', label='negative', markersize='4')
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    # plt.title(title, fontsize=12)
    plt.legend(loc="upper right")

def tsne_data(rawdata):
    """对输入数据进行处理降维并返回降维结果"""
    modle = TSNE(n_components=2, random_state=0)

    fea_data = rawdata.drop(columns=['class'])  #取出所有特征向量用于降维
    redu_fea = modle.fit_transform(fea_data) #将数据降到2维进行后期的可视化处理
    labels = rawdata['class'].replace([0, 1], ['negative', 'positive']) #将正反例数字替换成正反例标签用于收起展示
    redu_data = np.vstack((redu_fea.T, labels.T)).T #将特征向量和正反例标签整合
    tsne_df = pd.DataFrame(data=redu_data, columns=['Dimension1', 'Dimension2', "label"])
    return  tsne_df #返回用于可视化的数据

def process(rawdata):
    csvdata = pd.read_csv(rawdata)
    redimen_data = tsne_data(csvdata)
    tsne_visual(redimen_data)

def main():
    parser = argparse.ArgumentParser(description="Visualzes the features based on t-sne in a 2D feature space")
    parser.add_argument('-i', '--infile', help="The input file should be csv format")
    parser.add_argument('-o', '--outfile', help='The name of output file, png ')
    parser.add_argument('-l', '--layout', help='The layout of subplots, please input row,column number, default 1,1 ', default='1,1')
    args = parser.parse_args()

    infile_list = args.infile.split(',')
    numb = len(infile_list)
    row, column = [int(i) for i in args.layout.split(',')] 
    # plt.figure(figsize=(column*6, row*3), dpi=300, facecolor=(1, 1, 1)) #平铺画布，设置dpi300,注意，发表文章dpi不能低于300

    if numb > 1:
        for i in range(numb):
            plt.subplot(row, column, i+1)
            plt.subplots_adjust(wspace =0.3, hspace =0.35) #调节子图之间的宽度 
            process(infile_list[i])
            # plt.figtext(0.1*(i+1), 0.9, chr(i+65), fontsize=13) #多张图需要表上A，B……
    else:
        process(infile_list[0])
            
    plt.savefig(args.outfile)    

if __name__ == "__main__":
    main()