# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 09:07:30 2021

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import random

x=np.linspace(-1,1,100)
y=np.linspace(-1,1,100)

x,y=np.meshgrid(x,y)

def cartographie():

    f=lambda x,y: y**3-x**3
    z=f(x,y)

    fig=plt.figure()
    ax=fig.gca(projection="3d")
    surf= ax.plot_surface(x,y,z,cmap="viridis")
    nivx=plt.contour(x,y,z)

    bar=plt.colorbar(surf)
    bar.set_label("limites et zones de tolérance")

    plt.title("cartographie sonde lambda")
    plt.xlabel("valeur")
    plt.ylabel("détection")
    ax.set_zlabel("correction")
    ax.view_init(20,60)


def traitement():
    x=np.linspace(-1,1,100)
    y=x[::-1]
    
    while 1:
        r1=random.choice(x)
        r2=random.choice(y)
        z=r2**3-r1**3
        if z==-2 and z<-1.5:
            print("x=",r1,"y=",r2,"z=",z,"seuil critique min")
        elif z>=-1.5 and z<-1:
            print("x=",r1,"y=",r2,"z=",z,"limite min")
        elif z>=-1 and z<-0.5:
            print("x=",r1,"y=",r2,"z=",z,"tolérence min")
        elif z>=-0.5 and z<=0.5:
            print("x=",r1,"y=",r2,"z=",z,"zone de tolérence")
        elif z>0.5 and z<=1:
            print("x=",r1,"y=",r2,"z=",z,"tolérence max")
        elif z>1 and z<=1.5:
            print("x=",r1,"y=",r2,"z=",z,"limite max")
        else:
            z>1.5 and z==2
            print("x=",r1,"y=",r2,"z=",z,"seuil critique min")

def niveau():
    x=np.linspace(-1,1,100)
    y=x[::-1]

    x,y=np.meshgrid(x,y)

    z=y**3-x**3
    fig=plt.figure()
    nivx=plt.contourf(x,y,z)
    bar=fig.colorbar(nivx)

    plt.title("courbe de niveau lambda")
    bar.set_label("z")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid()
    bar.set_label=("z")
    plt.show()

