#!/usr/bin/env python
# coding: utf-8

# ## Histograms of Oriented Gradients

# In[1]:


import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import os


# In[2]:


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# In[3]:


for file in os.listdir('PennFudanPed/PNGImages'):
    
    # Read image
    png = 'PennFudanPed/PNGImages/'+file
    img = cv2.imread(png)
    
    # Resize image
    img = imutils.resize(img,width=min(400,img.shape[1]))
    
    # Detect People
    (rects, weights) = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)
        
    # Applying non_max_suppression
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    path = 'Predictions/Predictions_HOG/'+file.replace('.png','.txt')
    f = open(path,'w')
    flag = 1
    
    # Saving predictions in a file
    for (x1,y1,x2,y2) in picks:
        
        string = 'person 0.5 ' # Giving confidence of 0.5 to all predictions
        string += str(str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2))
        
        f.write(string)            
        if flag == len(rects):
            break
        f.write('\n')
        flag += 1
    f.close()


# In[ ]:




