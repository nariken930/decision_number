# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:00:21 2017

@author: Narita
"""
import numpy as np
from skimage.measure import label, regionprops

def props_img(label_img, height, width, index = None):
    #数字特徴量抽出
    props = regionprops(label_img)
    
    norm = height*width / 10000
    
    #データ出力
    data = []
    for i, p in enumerate(props):
        vec = []
        if(not index is None):
            vec.append(index[i][2]) #正解ラベル
        vec.append(p.area / norm) #数字領域の面積
        vec.append(p.filled_area / norm - p.area / norm) #穴の面積
        vec.append(p.convex_area / norm -  p.filled_area / norm) #凹部面積
        vec.append(p.euler_number) #オイラー数
        vec.append(p.perimeter / norm) #周囲長
     
        uy, lx, ly, rx = p.bbox #外接枠
        w = rx - lx
        h = ly - uy
        vec.append(w / norm) #幅
        vec.append(h / norm) #高さ
     
        fprop = regionprops(label(p.filled_image)) #穴を塗りつぶした領域の特徴量
        vec.extend((np.array(fprop[0].centroid) - np.array(p.centroid) ) / norm ) #穴を塗りつぶしたときの重心と，数字だけの重心の差
        vec.extend(np.array(p.centroid)) #数字の重心座標
        data.append(vec)
        
    return data