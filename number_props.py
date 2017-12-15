# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:32:27 2017

@author: Narita
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops
import function_props

def main():
    img = io.imread("numbers0.png",as_grey=True)
    
    height, width = img.shape
    size = height * width
    print("height : {}\nwidth : {}\nsize : {}".format(height, width, size) )
    
    idf = pd.read_csv('index0.csv') #numbers0.png の数字の位置座標と正解ラベル
    index = idf.values
    
    #入力画像と正解ラベルの確認
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    for idx in index:
        ax.annotate(str(int(idx[2])), xy=(idx[1], idx[0]), xytext=(idx[1]+10, idx[0]))
    plt.show()
     
    #二値化
    bimg = img < 0.7
     
    #ラベリング
    label_img = label(bimg)
    label_img[label_img==1] = 0 #枠を除外．ラスタスキャンなので１番が枠になる．
     
    data = function_props.props_img(label_img, height, width, index)
     
    df = pd.DataFrame(data,columns=['number',
                                    'area', 
                                    'hole area', 
                                    'dep area', 
                                    'euler_num', 
                                    'perimeter',
                                    'w', 'h',
                                    'dy', 'dx',
                                    'gy', 'gx'])
    df.to_csv('digit_features.csv', index=0)

if __name__=="__main__":
    main()