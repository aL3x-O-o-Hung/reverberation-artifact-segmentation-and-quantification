from __future__ import print_function
from skimage import io
from confidence_map_test import confidence_map3d,confidence_map2d
import numpy as np
#import pydicom
from skimage.external.tifffile import imshow
import time
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.transform import resize
import skimage
import cv2
import csv
import os
#from EdgeHistogramSRAD import *
#import SimpleITK as sitk


def artifact_cluster(artifact,ver,hor):
    imm=np.zeros_like(artifact)
    pos=np.argwhere(artifact>0.005)
    lab=1
    st=[]
    dir=[]
    for i in range(-ver,ver+1):
        for j in range(-hor,hor+1):
            if (i**2)/(ver**2)+(j**2)/(hor**2)>1 or i**2+j**2==0:
                continue
            else:
                dir.append([i,j])
    #dir=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
    for p in pos:
        if imm[p[0],p[1]]!=0:
            continue
        else:
            st.append(p)
            imm[p[0],p[1]]=lab
            while len(st)!=0:
                temp=st[len(st)-1]
                st.pop()
                for d in dir:
                    if 0<=temp[0]+d[0]<imm.shape[0] and 0<=temp[1]+d[1]<imm.shape[1]:
                        if artifact[temp[0]+d[0],temp[1]+d[1]]>0.005 and imm[temp[0]+d[0],temp[1]+d[1]]==0:
                            st.append([temp[0]+d[0],temp[1]+d[1]])
                            imm[temp[0]+d[0],temp[1]+d[1]]=lab
            lab+=1
    return imm

def remove_false(lab,needle,dis):
    labb=lab.copy()
    m=np.max(lab)
    mark=np.ones((int(m)))
    for i in range(1,int(m)+1):
        xx=np.argwhere(lab==i)
        flag=True
        for x in xx:
            for j in range(1,dis+1):
                if x[0]-j>=0:
                    if needle[x[0]-j,x[1]]>0.5:
                        flag=False
                        break
            if not flag:
                break
        if flag:
            mark[i-1]=0
            for x in xx:
                labb[x[0],x[1]]=0
    return labb,mark








def pick_large(lab1,lab2):
    temp=(lab1>lab2).astype(np.float)
    return temp*lab1+(1-temp)*lab2


def new_map_generation(hard_label):
    new_map=hard_label
    g=(new_map>0).astype(np.float)
    z=g.copy()
    for j in range(hard_label.shape[1]):
        step=0
        for i in range(hard_label.shape[0]):
            if g[i,j]==1:
                step=0
                continue
            else:
                z[i,j]=1/(1+np.exp((step-64)/20))
                step+=1
    new_map=hard_label
    new_map*=z
    return new_map



def transform_image_step1(hard_label,needle):
    #compare with true label
    lab=artifact_cluster(hard_label,11,7)
    lab,mark=remove_false(lab,needle,10)
    #new_map=new_map_generation(hard_label)
    new_map=hard_label*(lab>0).astype(np.float)
    return new_map,lab,mark








def check_needle(i,j,needle,width):
    step=1
    f=False
    h=999999
    while i-step>=0:
        for jj in range(-width,width+1):
            if j+jj<0 or j+jj>=needle.shape[1]:
                continue
            if needle[i-step,j+jj]>0.5:
                if h>np.sqrt(width**2+step*2):
                    f=True
                    h=np.sqrt(width**2+step**2)
        step+=1
    if f:
        return h
    else:
        return None

def decay_function(x,h,height):
    #height=max(h,height)
    #t=np.sqrt(width**2+height**2)
    #print(np.log(-h+t+1)/np.log(t))
    #res=x*np.log(-h+t+1)/np.log(t)
    #res=x/(1+np.exp((h-height*3/4)/40))
    res=x*np.exp(-h/height*0.8)
    return res


def transform_image_step2(new_map,needle,lab,mark):
    #taking the position of needle into account
    for j in range(mark.shape[0]):
        if mark[j]==0:
            continue
        xx=np.argwhere(lab==j+1)
        max_min_h=-1
        h_=[]
        for x in xx:
            h=check_needle(x[0],x[1],needle,15)
            h_.append(h)
            if h is None:
                continue
            if h>max_min_h:
                max_min_h=h
        i=0
        for x in xx:
            if h_[i] is None:
                i+=1
                continue
            new_map[x[0],x[1]]=decay_function(new_map[x[0],x[1]],h_[i],max_min_h)
            i+=1
    return new_map


def transform_image_step2_(artifact,im,lab,mark):
    out=np.zeros_like(artifact)
    for i in range(mark.shape[0]):
        if mark[i]==0:
            continue
        mm=(lab==i+1).astype(np.float)
        temp=im*mm
        temp_=temp.copy()
        xx=np.argwhere(lab==i+1)
        #temp_=np.reshape(temp_,(temp_.shape[0]*temp_.shape[1]))
        tempp=np.sort(temp_,axis=None)[-xx.shape[0]//20:]
        m=np.mean(tempp)
        t=temp/(m+0.0001)
        xx=np.argwhere(t>1)
        for x in xx:
            t[x[0],x[1]]=1
        out+=artifact*t
    return out




def function(x,m):
    if m!=0:
        return 1/(1+np.exp(-(8/m*x-4)))
    if m==0:
        return 0

def transform_image_step3(im,new_map):
    #taking the original intensity into account
    win_x=2
    win_y=1
    xx=np.argwhere(new_map>0.01)
    temp=new_map.copy()
    for x in xx:
        m=-1
        for ii in range(-win_x,win_x+1):
            for jj in range(-win_y,win_y+1):
                if x[0]+ii<0 or x[0]+ii>=im.shape[0] or x[1]+jj<0 or x[1]+jj>=im.shape[1] or temp[x[0]+ii,x[1]+jj]<0.01:
                    continue
                if im[x[0]+ii,x[1]+jj]>m:
                    m=im[x[0]+ii,x[1]+jj]
        new_map[x[0],x[1]]*=function(im[x[0],x[1]],m)

    return new_map



i=0
#i=425
#i=403
dir='../data/needle/hpu_training_refined/'

dirr='../data/needle/training/'
crop=[89,624,178,678]
while os.path.exists(dir+str(i)+'_artifact.png'):
    artifact=cv2.imread(dir+str(i)+'_artifact.png')
    needle=cv2.imread(dir+str(i)+'_needle.png')
    artifact_std=cv2.imread(dir+str(i)+'_artifact_std.png')
    needle_std=cv2.imread(dir+str(i)+'_needle_std.png')
    im=cv2.imread(dirr+str(i)+'.png')
    artifact=cv2.cvtColor(artifact,cv2.COLOR_BGR2GRAY)/255
    needle=cv2.cvtColor(needle,cv2.COLOR_BGR2GRAY)/255
    artifact_std=cv2.cvtColor(artifact_std,cv2.COLOR_BGR2GRAY)/255
    needle_std=cv2.cvtColor(needle_std,cv2.COLOR_BGR2GRAY)/255
    im=cv2.cvtColor(im[crop[0]:crop[1],crop[2]:crop[3],:],cv2.COLOR_BGR2GRAY)/255
    im=cv2.resize(im,(256,256))
    p=artifact.copy()
    new1,lab,mark=transform_image_step1(artifact,needle)
    #plt.subplot(2,3,1)
    #plt.imshow(new1,'gray')
    new2=transform_image_step2(new1,needle,lab,mark)
    new2_=transform_image_step2_(new1,im,lab,mark)
    temp=np.zeros((new2.shape[0],new2.shape[1],2))
    temp[:,:,0]=new2
    temp[:,:,1]=new2_
    new2=np.max(temp,axis=-1)
    new2=np.reshape(new2,(new2.shape[0],new2.shape[1]))
    #plt.subplot(2,3,2)
    #plt.imshow(new2,'gray')
    new3=transform_image_step3(im,new2)
    ratio=new3/(p+0.0001)
    artifact_std*=ratio

    '''
    plt.subplot(2,3,3)
    plt.imshow(new3,'gray')
    plt.subplot(2,3,4)
    plt.imshow(p,'gray')
    plt.subplot(2,3,5)
    plt.imshow(im,'gray')
    plt.subplot(2,3,6)
    plt.imshow(needle,'gray')
    plt.show()
    '''

    #new2=transform_image_step2(new1,needle)

    cv2.imwrite('../data/needle/new_hpu_training_refined/'+str(i)+'artifact.png',new3*255)
    cv2.imwrite('../data/needle/new_hpu_training_refined/'+str(i)+'needle.png',needle*255)
    cv2.imwrite('../data/needle/new_hpu_training_refined/'+str(i)+'artifact_std.png',artifact_std*255)
    cv2.imwrite('../data/needle/new_hpu_training_refined/'+str(i)+'needle_std.png',needle_std*255)

    print(i)

    i+=1







