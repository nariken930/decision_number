# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:33:41 2017

@author: Narita
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, tree
from skimage import io
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from number_props import props_img

#学習データの読込
df = pd.read_csv('digit_features.csv')
data = df.values

#機械学習
#classifier = svm.SVC()
classifier = tree.DecisionTreeClassifier()
train_X = data[:,1:-2] #data（最初の列は正解ラベル，最後の２列は数字座標のため除外）
train_y = data[:,0] #label（最初の列が正解ラベル）
classifier.fit(train_X, train_y)

#決定木の出力（出力されたtree.dotを　http://webgraphviz.com/　で可視化できる)
tree.export_graphviz(classifier, out_file="tree.dot", 
                     feature_names=df.columns[1:-2],
                     class_names=list(map(str,range(1,10))),
                     filled=True, rounded=True
                     )

#画像読み込み
img = io.imread('numbers0.png', as_grey=True) #入力画像を読み込み

h_img, w_img = img.shape
size_img = h_img * w_img
print("height : {}\nwidth : {}\nsize : {}".format(h_img, w_img, size_img) )

#二値化
bimg = img < 0.7

#ラベリング
label_img = label(bimg)
label_img[label_img==1] = 0 #枠を除外．ラスタスキャンなので１番が枠になる．

data = props_img(label_img=label_img, height=h_img, width=w_img, index=None)
data = np.array(data)
print(data)
pos = data[:, -2:]
print(pos)
data = data[:, :-2]
print(data)
##数字特徴量抽出
#props = regionprops(label_img)
#
#data = []
#pos = []
#for p in props:
#    vec = []
#    vec.append(p.area / size_img) #数字領域の面積
#    vec.append(p.filled_area / size_img - p.area / size_img) #穴の面積
#    vec.append(p.convex_area / size_img - p.filled_area / size_img) #凹部面積
#    vec.append(p.euler_number) #オイラー数
#    vec.append(p.perimeter / size_img) #周囲長
# 
#    uy, lx, ly, rx = p.bbox #外接枠
#    w = rx - lx
#    h = ly - uy
#    vec.append(w / size_img) #幅
#    vec.append(h / size_img) #高さ
# 
#    fprop = regionprops(label(p.filled_image)) #穴を塗りつぶした領域の特徴量
#    vec.extend((np.array(fprop[0].centroid) - np.array(p.centroid) ) / size_img) #穴を塗りつぶしたときの重心と，数字だけの重心の差
#    data.append(vec)
#
#    pos.append( (p.centroid[1], p.centroid[0]) ) #数字の重心座標

pred_y = classifier.predict(data)

#入力画像と判別ラベルの確認
fig = plt.figure(figsize=(12,8))
bx = fig.add_subplot(121)
bx.imshow(label2rgb(label_img,bg_label=0))

ax = fig.add_subplot(122)
ax.imshow(img, cmap='gray')
for i, num in enumerate(pred_y):
    ax.annotate(str(int(num)), xy=pos[i], xytext=np.array(pos[i])+np.array((10,0)))
plt.show()