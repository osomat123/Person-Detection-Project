#!/usr/bin/env python
# coding: utf-8

# **This file compares the predictions made by all four algorithms on a picture picked from the datatset**

# In[1]:


import cv2
import os
import random


# In[2]:


def groundTruthBox(fileTxt,img,color=(0,255,0)):
    # Ground Truth Boxes
    path = 'PennFudanPed/GroundTruths/'+fileTxt

    # Read file with Ground Truth Bounding box
    f = open(path,'r')
    data = f.read()
    f.close()

    boxes = data.split('\n')

    for box in boxes:
    
        gt = box.split(" ")[1:]
        (x1,y1,x2,y2) = [int(i) for i in gt]
        # Creating Bounding box on image
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        
    return img


# In[3]:


def predictionBox(file,img,color=(0,0,255)):
    # Read file with Ground Truth Bounding box
    f = open(file,'r')
    data = f.read()
    f.close()

    boxes = data.split('\n')
    boxes.pop(0)

    if len(boxes) > 0:
        for box in boxes:
    
            pred = box.split(" ")[2:]
            try:
                (x1,y1,x2,y2) = [int(i) for i in pred]
                # Creating Bounding box on image
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            except Exception as e:
                continue
    return img


# In[4]:


# Uncomment the line below to pick a random image 
#file = random.choice(os.listdir('PennFudanPed/PNGImages'))
file = 'PennPed00052.png'


# In[5]:


image = cv2.imread('PennFudanPed/PNGImages/'+file)
fileTxt = file.replace('.png','.txt')


# In[6]:


gt = groundTruthBox(fileTxt,image.copy())

path = 'Predictions/Predictions_ViolaJones/'+fileTxt
violaJones = predictionBox(path,image.copy(),(255,0,0))

path = 'Predictions/Predictions_HOG/'+fileTxt
hog = predictionBox(path,image.copy(),(0,0,255))

path = 'Predictions/Predictions_MobileNetSSD/'+fileTxt
mobileNetSSD = predictionBox(path,image.copy(),(0,255,255))

path = 'Predictions/Predictions_YOLO/'+fileTxt
yolo = predictionBox(path,image.copy(),(255,255,0))


# In[7]:


cv2.imshow('Ground Truth',gt)
cv2.imshow('Viola Jones',violaJones)
cv2.imshow('HOG',hog)
cv2.imshow('YOLO',yolo)
cv2.imshow('MobileNet SSD',mobileNetSSD)
cv2.waitKey(0)


# In[ ]:




