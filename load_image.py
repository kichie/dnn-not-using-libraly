# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image, ImageOps
import numpy as np

def load_image(normalize=True, to_gray_scale=True):
    train_table = pd.read_table('./signate_画像１０種類/train_master.tsv')
    
#    print(train_table.label_id)
#    print(train_table.file_name)
    x_train = []
    t_train = []
    x_test = []
    t_test = []
    counter = [0,0,0,0,0,0,0,0,0,0]
    print('loading...')
    for row in train_table.itertuples():
        img = Image.open('./signate_画像１０種類/train_images/'+str(row.file_name))
        if to_gray_scale:
            img = ImageOps.grayscale(img)
            #画像を保存
            
        np_img = np.array(img, dtype=np.float32)
        flatten_image = np_img.flatten()
#        flatten_image /= 255
        
        #標準化 詳しくはこちらhttps://deepage.net/features/numpy-normalize.html
        if normalize:
            flatten_image = normalizer(flatten_image)
        
        
        counter[row.label_id] += 1
        
        if counter[row.label_id]%5 == 0:
            x_test.append(flatten_image)
            t_test.append(row.label_id)
        else:
            x_train.append(flatten_image)
            t_train.append([row.label_id])
            
            
            
    print('done.')
    
    x_test = np.array(x_test, dtype=np.float32)
    t_test = np.array(t_test, dtype=np.int)
    x_train = np.array(x_train, dtype=np.float32)
    t_train = np.array(t_train, dtype=np.int)
        
    return x_train, t_train, x_test, t_test

#正規化
def normalizer(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

#labelと文字列
def load_corr_table():
    
    label_table = pd.read_table('./signate_画像１０種類/label_master.tsv')
    return label_table

if __name__ == '__main__':
    x,t,a,b = load_image(normalize=True, to_gray_scale=True)
