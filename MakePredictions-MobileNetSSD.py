#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os


# In[2]:


#Getting Labels
LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]

# Setting path to weights and  config file
modelPath = 'MobileNetSSD_deploy.caffemodel'
configPath = 'MobileNetSSD_deploy.prototxt.txt'

# Loading YOLO trained on COCO Dataset
net = cv2.dnn.readNetFromCaffe(configPath,modelPath)


# In[4]:


for file in os.listdir('PennFudanPed/PNGImages'):
    
    # Loading Image
    img  = cv2.imread('PennFudanPed/PNGImages/'+file)
    (H, W) = img.shape[:2]
    
    # Creating blob from image to perform feature normalization
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007843,(300,300),127.5)

    # Setting Input and performing forward pass through YOLO Network
    net.setInput(blob)
    detections = net.forward()

    # Initialing required lists
    boxes = []
    confidences = []

    # Looping over detections
    for i in np.arange(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        idx = int(detections[0,0,i,1])
        
        if confidence >= 0.5 and LABELS[idx] == 'person':
        
            # scaling bounding box cordinates according to size of original image
            box = detections[0,0,i,3:7]*np.array([W,H,W,H])
            (x1,y1,x2,y2) = box.astype('int')
        
            boxes.append((x1,y1,x2,y2))
            confidences.append(float(confidence))
        
    path = 'Predictions/Predictions_MobileNetSSD/'+file.replace('.png','.txt')
    f = open(path,'w')
        
    flag = 1
    for i in range(len(confidences)):
        
        (x1, y1) = (boxes[i][0], boxes[i][1])
        (y2, y2) = (boxes[i][2], boxes[i][3])
            
        # Saving predi ctions in txt file
        string = 'person '
        string += str(confidences[i])+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)
            
        f.write(string)
        if flag == len(confidences):
            break
        f.write('\n')
        flag += 1
    f.close()


# In[ ]:




