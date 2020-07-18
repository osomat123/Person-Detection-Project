#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import cv2
import numpy as np


# ## Get true bounding boxes from annotations

# In[2]:


annot = []

for file in os.listdir('Annotation'):
    d = {}
    d['name'] = file
    
    f = open('Annotation/'+file,'r')
    data = f.read()
    f.close()
    
    # Filtering Useless data from annotations
    exp = re.findall(r"\(([^)]+)\)",data)
    exp = exp[2:]
    
    exp = list(filter(lambda a: a != '"PASpersonWalking"',exp))
    exp = list(filter(lambda a: a != 'Xmin, Ymin', exp))
    exp = list(filter(lambda a: a != 'Xmax, Ymax', exp))
    exp = list(filter(lambda a: a != '"PASpersonStanding"',exp))
    
    # Converting String in to integers
    exp2 = []
    for i in exp:
        temp = [int(j) for j in i.split(', ')]
        exp2.append(temp[0])
        exp2.append(temp[1])
        
    num = int(len(exp2)/4)
    
    # Arranging data into a dictionary
    rect = []
    for i in range(num):
        rect.append(exp2[i*4:i*4+4])
    d['rect'] = rect    
    
    d['num'] = num
    
    annot.append(d)


# ## Create Ground Truth files in required format

# In[3]:


for file in annot:
    path = 'GroundTruths/'+file['name']
    f = open(path,'w')
    for i in range(len(file['rect'])):
        string = 'person '
        for j in file['rect'][i]:
            string+= str(j)+' '
        f.write(string.strip())
        if i == len(file['rect'])-1:
            break
        f.write('\n')
    f.close()


# ## Checking 

# In[72]:


for file in os.listdir('GroundTruths'):
    png = 'PNGImages/'+file.replace('.txt','.png')
    pic = cv2.imread(png)
    
    f = open('GroundTruths/'+file,'r')
    data = f.read()
    f.close()
    c = data.split('\n')
    
    for box in c:
        box = box[7:]
        (x1,y1,x2,y2) = [int(i) for i in box.split(' ')]
        cv2.rectangle(pic,(x1,y1),(x2,y2),(0,255,0),2)
        
    cv2.imshow('truth',pic)
    cv2.waitKey(0)

