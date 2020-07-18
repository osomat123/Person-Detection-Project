#!/usr/bin/env python
# coding: utf-8

# ## Viola Jones

# In[1]:


import cv2
import numpy as np
import os


# In[2]:


# Specifying Cascade File
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# In[3]:


for file in os.listdir('PennFudanPed/PNGImages'):
    
    # Reading Image and converting to grayscale
    png = 'PennFudanPed/PNGImages/'+file
    pic = cv2.imread(png)
    gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        
    rects = body_cascade.detectMultiScale(gray,1.03,7)
    
    path = 'Predictions/Predictions_ViolaJones/'+file.replace('.png','.txt')
    f = open(path,'w')
    flag = 1
    
    # Saving predictions in a file
    for (x,y,w,h) in rects:
        
        string = 'person 0.5 ' # Giving confidence of 0.5 to all predictions
        string += str(str(x)+' '+str(y)+' '+str(x+w)+' '+str(y+h))
        
        f.write(string)            
        if flag == len(rects):
            break
        f.write('\n')
        flag += 1
    f.close()

