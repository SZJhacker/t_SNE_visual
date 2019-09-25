## t-SNE 
该脚本利用t-SNE将特征提取结果的csv降至2维，并以散点图的形式进行可视化展示
### 背景介绍
T 分布随机近邻嵌入（ **T-Distribution Stochastic Neighbour Embedding** ）是一种用于降维的机器学习方法，它能帮我们识别相关联的模式。t-SNE 主要的优势就是保持局部结构的能力。这意味着高维数据空间中距离相近的点投影到低维中仍然相近。t-SNE 同样能生成漂亮的可视化。

### Usage
```shell
usage: tsnepro [-h] -i INFILE -o OUTFILE [-l LAYOUT] [-d DPI]

Visualzes the features based on t-sne in a 2D feature space

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        The input file should be csv format, and multiple file
                        should be separated by commas
  -o OUTFILE, --outfile OUTFILE
                        The name of output picture
  -l LAYOUT, --layout LAYOUT
                        The layout of subplots, please input row,column
                        number, default 1,1
  -d DPI, --dpi DPI     The dpi of output picture, default is 300dpi
```
### 参数详细解释说明
基本用法：
```shell
python3 tsne_pro.py -i test1.csv -o test.png
```

高级用法：
```shell
python3 tsne_pro.py -i test1.csv,test2.csv -o two.png -l 1,2
```
同时对多个文件进行降维处理并在一张图中展示，**-i**参数指定输入文件（绝对或相对路径），当有多个文件作为输入的时候，必须使用-l参数指定不同文件tsne图在画布中的分布，行列乘积应大于给定的文件数，示例中指定两个文件可视化图片为一行两列。