# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:18:53 2023

@author: user
"""

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import make_blobs,make_circles,make_moons
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_log_error,accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split


x,y=make_moons(n_samples=100,noise=0.1)
x=x.T


y_label=LabelBinarizer().fit_transform(y)
y_label=y_label.T

layers_dims=(x.shape[0],8,8,y_label.shape[1])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmo(x):
    return  x*(1-x)

def init(layers_dims):
    params={}
    L=len(layers_dims)
    
    for i in range(1,L):
        params["w"+str(i)]=np.random.randn(layers_dims[i],layers_dims[i-1])
        params["b"+str(i)]=np.random.randn(layers_dims[i],1)
        
        #for i in range(1,L+1):
            #z=params["w"+str(L)].dot(cache["a"+str(L-1)])+params["b"+str(L)]
            #cache["a"+str(L)]=softmax(z)
        
    return params

def forward(x,params):
    cache={"a0":x}
    L=len(params)//2
    for i in range(1,L+1):
        z=params["w"+str(i)].dot(cache["a"+str(i-1)])+params["b"+str(i)]
        cache["a"+str(i)]=sigmoid(z)

    return cache

def cross_entropy(y,cache,params):
    L=len(params)//2    
    m=y.shape[0]
    for i in range(1,L+1):
        log_loss=1/m*np.sum(y.dot(np.log(cache["a"+str(i)].T))+(1-y).dot(np.log(1-(cache["a"+str(i)].T))))
    
    return log_loss

def backward(y,cache,params):
    gradient={}
    L=len(params)//2
    m=y.shape[0]
    dz=cache["a"+str(L)]-y
    for i in reversed(range(1,L+1)):
        gradient["dw"+str(i)]=1/m*np.dot(dz,cache["a"+str(i-1)].T)
        gradient["db"+str(i)]=1/m*np.sum(dz,axis=1,keepdims=True)
        
        if L>1:
            dz=np.dot(params["w"+str(i)].T,dz)*deriv_sigmo(cache["a"+str(i-1)])
    return gradient 
    
def update(y,cache,params,learning_rate):
    gradient=backward(y,cache,params)
    L=len(params)//2
    for i in range(1,L+1):
        params["w"+str(i)]=params["w"+str(i)]-learning_rate*gradient["dw"+str(i)]
        params["b"+str(i)]=params["b"+str(i)]-learning_rate*gradient["db"+str(i)]
    return params

def predict(x,params):
    L=len(params)//2
    cache=forward(x, params)
    aL=cache["a"+str(L)]
    #return np.argmax(aL,axis=0)
    return aL>0.5

def visualisation(y_original,params):
    ax=plt.subplot()
    ax.scatter(x[0,:],x[1,:],c=y_original)
    x0_lim=ax.get_xlim()
    x1_lim=ax.get_ylim()
    
    x0=np.linspace(x0_lim[0],x0_lim[1],100)
    x1=np.linspace(x1_lim[0],x1_lim[1],100)
    
    w,v=np.meshgrid(x0,x1)
    tablo=np.vstack((w.ravel(),v.ravel()))
    
    z=predict(tablo,params)
    z=z.reshape((100,100))
    
    ax.pcolormesh(w,v,z,alpha=0.3,zorder=-1)
    ax.contour(w,v,z,colors="green")
    
def frontiere(x,y,y_original,layers_dims=(x.shape[0],8,8,y_label[0]),learning_rate=0.1,n_iter=5000):
    
    params=init(layers_dims)

    for i in tqdm(range(n_iter)):
        cache=forward(x,params)
        params=update(y, cache, params, learning_rate)

        
    y_pred=predict(x,params)
    visualisation(y_original, params)
    return y_pred

def apprentissage(x,y,y_original,layers_dims=(x.shape[0],8,8,y_label[0]),learning_rate=0.1,n_iter=5000):
    
    params=init(layers_dims)
    loss_hit=[]

    for i in tqdm(range(n_iter)):
        cache=forward(x,params)
        params=update(y, cache, params, learning_rate)
        
        loss=cross_entropy(y, cache, params)
        loss_hit.append(loss)

        
    y_pred=predict(x,params)
    plt.plot(loss_hit)
    return y_pred

plt.scatter(x[0,:],x[1,:],c=y)
plt.show()

def classification(x,y_label):
    reseaux=frontiere(x,y_label,y_original=y,layers_dims=(x.shape[0],8,8,y_label.shape[0]),learning_rate=0.1,n_iter=5000)
    return reseaux

def courbe(x,y_label,y_pred):
    params=init(layers_dims)
    y_pred=predict(x,params)
    reseaux=apprentissage(x,y_label,y_original=y,layers_dims=(x.shape[0],8,8,y_label.shape[0]),learning_rate=0.1,n_iter=5000)
    erreur=accuracy_score(y,y_pred)
    print("précision=",erreur)
    return reseaux