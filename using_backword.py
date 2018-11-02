# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict, Counter

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


from dataset.mnist import load_mnist
from load_image import load_image


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    
    if x.ndim == 2:
        
        x = x.T
        x = x - np.max(x, axis=0)
#        print(np.max(x, axis=0))
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x<=0)
        assert self.mask.any(), "勾配が消失しました。伝搬する値がありません。"
        out = x.copy()
        out[self.mask] = 0
        
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
#        https://deepage.net/features/numpy-axis.html
        self.db = np.sum(dout, axis=0)
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
#ネットワーク
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
#        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size) 
#        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
#        self.layers['Relu2'] = Relu()
#        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
#        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
#        grads['W3'] = self.layers['Affine3'].dW
#        grads['b3'] = self.layers['Affine3'].db

        return grads
    
    
#３ネットワーク
class ThreeLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads
    
    
if __name__ == "__main__":
    
    
    #データロード
#    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    x_train, t_train, x_test, t_test = load_image(normalize=True, to_gray_scale=True)
    
    print(type(x_train))
    print("画像データ",x_train.shape)
    print("データ数、行{},ピクセル数、列{}".format(x_train.shape[0],x_train.shape[1]))
    
    print(type(t_train))
    print("教師データ", t_train.shape)
    print("データ数、行{}".format(t_train.shape[0]))
    
#    x_test, t_test = x_train[4000:], t_train[4000:]
#    x_train, t_train = x_train[:4000], t_train[:4000]
    
    from pprint import pprint
    pprint(Counter(t_train.flatten()))
    pprint(Counter(t_test.flatten()))
    
    #教師データのタイプ数
    t_type_num = len(Counter(t_train.flatten()))
    
    print(t_type_num)
    TLN = TwoLayerNet(input_size=x_train.shape[1], hidden_size=1000, output_size=t_type_num)

    
    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    lr = 0.01
    
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    #train_sizeの数学習したら１epochとみなす 50000/100=500
    iter_per_epoch = max(train_size/batch_size, 1)
    
    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = TLN.gradient(x_batch,t_batch)
        
        for key in ["W1","W2","b1","b2"]:
            TLN.params[key] -= lr*grad[key]
            
        
        if i % iter_per_epoch == 0:
        
            train_loss = TLN.loss(x_batch, t_batch)
            train_loss_list.append(train_loss)
            test_loss = TLN.loss(x_test, t_test)
            test_loss_list.append(test_loss)
            
            train_acc = TLN.accuracy(x_train, t_train)
            test_acc = TLN.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("########epoch:", int(i/iter_per_epoch+1), "########")
            print("train_acc:",train_acc)
            print("test_acc:",test_acc)
            print("train_loss:",train_loss)
            print("test_loss:",test_loss)
            print()
            
    fp = FontProperties(fname='./IPAfont00303/ipamp.ttf', size=14)
    plt.plot(np.arange(len(train_loss_list)),train_loss_list, linestyle='-.', label="学習データに対する1epochごとの損失関数の推移")
    plt.plot(np.arange(len(test_loss_list)),test_loss_list,linestyle='--', label="テストデータに対する1epochごとの損失関数の推移")
    plt.legend(prop=fp, loc='upper left')
    plt.title("学習データとテストデータの損失関数の推移", fontproperties=fp)
    plt.show()
    
    plt.plot(np.arange(len(train_acc_list)),train_acc_list,linestyle='-.', label="学習データに対する1epochごとの正答率の推移")
    plt.plot(np.arange(len(test_acc_list)),test_acc_list, linestyle='--',label="テストデータに対する1epochごとの正答率の推移")
    plt.legend(prop=fp, loc='upper left')
    plt.title("学習データとテストデータの正解率の推移", fontproperties=fp)
    plt.show()
    
    
    
    
    
    
        
#    x_batch = x_train[:3]
#    t_batch = t_train[:3]
#    
#    print(x_batch.shape)
#    print(t_batch.shape)
#    
#    grad_numerical = TLN.numerical_gradient(x_batch, t_batch)
#    grad_backprop  = TLN.gradient(x_batch, t_batch)
#    
#    for key in grad_numerical.keys():
#        diff = np.average(abs(grad_backprop[key] - grad_numerical[key]))
#        print(key +":"+str(diff))
#        