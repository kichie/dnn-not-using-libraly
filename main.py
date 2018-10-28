# coding: utf-8
import numpy as np
from dataset.mnist import load_mnist

#TODO:どこで計算が0になっているかを探す

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_test, t_test

class DeepNet():
    params = {}
    grad = {}
    
    def __init__(self, input_data_shape=(1,784), output_dim=10):
        #->9216(96*96)
        
        mid1 = 25
        mid2 = 50
    #    データの画素値: data
        W1_shape = (input_data_shape[1], mid1)
        W2_shape = (mid1, mid2)
        W3_shape = (mid2, output_dim)
        #np.random.rand(dim)
        W1 = np.random.rand(W1_shape[0], W1_shape[1])
        W2 = np.random.rand(W2_shape[0], W2_shape[1])
        W3 = np.random.rand(W3_shape[0], W3_shape[1])
        
    #    入力したデータ数: input_data_shape
        
        b1_shape = (input_data_shape[0], mid1)
        b2_shape = (input_data_shape[0], mid2)
        b3_shape = (input_data_shape[0], output_dim)
        
        
        b1 = np.random.rand(b1_shape[0], b1_shape[1])
        b2 = np.random.rand(b2_shape[0], b2_shape[1])
        b3 = np.random.rand(b3_shape[0], b3_shape[1])
        
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1
        self.params['b2'] = b2
        self.params['b3'] = b3
    
    
    
    def loss(self, x, t):
        
        y = deepNet.predict(x[i])
        p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
        print(p)
        print(y[0][p])
        print(np.argmax(t[i]))
        
        #損失関数を求める
        return cross_entropy(y,t[i])
    
    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        print("x:",x.shape)
        print("W1:",W1.shape)
        a1 = np.dot(x, W1) + b1
        print("a1:",a1.shape)
        print()
        z1 = sigmoid(a1)
        
        print("z1:",z1.shape)
        print("W2:", W2.shape)
        a2 = np.dot(z1, W2) + b2
        print("a2:",a2.shape)
        print()
        z2 = sigmoid(a2)
        
        print("z2:",z2.shape)
        print("W3:", W3.shape)
        a3 = np.dot(z2, W3) + b3
        print("a3:",a3.shape)
        print()
        y = softmax(a3)
        print("y:",y.shape)
        return y
    
    def grad(self, x, t):
        loss_w = lambda W:self.loss(x, t)
        grads = {}
        
        grads["W1"] = numerical_gradient(loss_w, self.params["W1"])
        grads["W2"] = numerical_gradient(loss_w, self.params["W2"])
        grads["W3"] = numerical_gradient(loss_w, self.params["W3"])
        grads["b1"] = numerical_gradient(loss_w, self.params["b1"])
        grads["b2"] = numerical_gradient(loss_w, self.params["b2"])
        grads["b3"] = numerical_gradient(loss_w, self.params["b3"])
        
        return grads
        
    def learning():
        pass


def sigmoid(x):
    ips = 1e-06
    return 1 / (1 + np.exp(-x) + ips)

def softmax(x):
    ips = 1e-06
    return x / (np.sum(np.exp(x)) + ips)

#中間差分
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    #flattenして一つずつ取り出す
    it = np.nditer(x, flags=['multi_index'],  op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()   
        
    return grad


def cross_entropy(y, t):
    return -(np.sum(t*np.log(y)))
    
        

lr = 0.01

total_loss = []
x, t = get_data()
deepNet = DeepNet()
accuracy_cnt = 0
for i in range(len(x)):
    
    grad = deepNet.grad(x[i], t[i])
    print(grad)
    deepNet.params["W1"] = deepNet.params["W1"] - lr * grad["W1"]
    deepNet.params["W2"] = deepNet.params["W2"] - lr * grad["W2"]
    deepNet.params["W3"] = deepNet.params["W3"] - lr * grad["W3"]
    deepNet.params["b1"] = deepNet.params["b1"] - lr * grad["b1"]
    deepNet.params["b2"] = deepNet.params["b2"] - lr * grad["b2"]
    deepNet.params["b3"] = deepNet.params["b3"] - lr * grad["b3"]
    
    loss = deepNet.loss(x[i], t[i])
    total_loss.append(loss)
    print(loss)
for i in range(len(x)):
    y = deepNet.predict(x[i])
    p = np.argmax(y)
    if p == np.argmax(t[i]):
        accuracy_cnt += 1
    
    
    

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))