# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:26:54 2021

@author: user
"""
from math import cos,sin,atan2,pi,acos,sqrt

class localisation :
    
    def __init__(self,p1,p2,p3,p4):
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
    
 
    def get_longitude(self,u,v):
        self.u=u
        self.v=v
        x=cos(self.u)*sin(self.v)
        y=sin(self.u)*sin(self.v)
        z=cos(self.v)
        roh=sqrt((x**2)+(y**2)+(z**2))
        wu=acos(z/roh)
        return wu
    
    def get_latitude(self,u,v):
        self.u=u
        self.v=v
        x=cos(self.u)*sin(self.v)
        y=sin(self.u)*sin(self.v)
        wv=atan2(x,y)
        return wv
      
    def boucle(self,u,v):
       self.u=u
       self.v=v
       
       while self.u<6.28 and self.v<6.28:
            self.u+=0.01
            self.v+=0.01
            
            x=a.get_longitude(self.u,self.v)*180/pi
            y=a.get_latitude(self.u,self.v)*180/pi
            w=a.get_longitude(self.u,self.v)*180/pi
            z=a.get_latitude(self.u,self.v)*180/pi
            
            longitude1=round(x,3)
            latitude1=round(y,3)
            longitude2=round(w,3)
            latitude2=round(z,3)
    
        
            if longitude1==self.p1 and latitude1==self.p2:
                print("localisé")
            elif longitude2==self.p3 and latitude2==self.p4:
                print("localisé")
            else:
                print("M:longitude=",longitude1,"latitude=",latitude1)

a=localisation(1.901,-88.099,0.183,-89.817)
a.boucle(0,0)

        