#!/usr/bin/env python
# -*- coding=utf-8 -*-

from tsne import tsne_visual
import pandas as pd
import matplotlib.pyplot as plt
import argparse 

def main():
    parser = argparse.ArgumentParser(description="Visualzes the features in a 2D feature space")
    parser.add_argument('-i', '--infile', help="The input file should be csv format")
    parser.add_argument('-o', '--outfile', help='The name of output file, png ')
    args = parser.parse_args()
    csvfile = pd.read_csv(args.infile, header=None, names=['Dimension1', 'Dimension2', 'label']) # tsne_visual识别label标签
    tsne_visual(csvfile)
    plt.savefig(args.outfile, dpi=400)

    
if __name__ == "__main__":
    main()