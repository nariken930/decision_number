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
from skimage.filters import try_all_threshold, threshold_otsu, threshold_mean, threshold_li
import function_props

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
img = io.imread('numbers1s.png', as_grey=True) #入力画像を読み込み

h_img, w_img = img.shape
size_img = h_img * w_img
print("height : {}\nwidth : {}\nsize : {}".format(h_img, w_img, size_img) )

#二値化
thresh = threshold_otsu(img) #しきい値決定（大津）
bimg = img < thresh

#ラベリング
label_img = label(bimg)
label_img[label_img==1] = 0 #枠を除外．ラスタスキャンなので１番が枠になる．

data = function_props.props_img(label_img=label_img, height=h_img, width=w_img, index=None)
data = np.array(data)
pos = np.concatenate((np.c_[data[:, -1] ], np.c_[data[:, -2] ] ), axis=1)
pos = pos.tolist()
data = data[:, :-2]
data = data.tolist()

pred_y = classifier.predict(data)

#入力画像と判別ラベルの確認
fig = plt.figure(figsize=(12,8) )
bx = fig.add_subplot(121)
bx.imshow(label2rgb(label_img,bg_label=0) )

ax = fig.add_subplot(122)
ax.imshow(img, cmap='gray')
for i, num in enumerate(pred_y):
    ax.annotate(str(int(num)), xy=pos[i], xytext=np.array(pos[i]) + np.array((10,0) ) )
plt.savefig("result.png")
plt.show()