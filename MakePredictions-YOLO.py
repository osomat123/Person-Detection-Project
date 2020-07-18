#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os


# In[2]:


#Getting Labels
labelsPath = 'coco.names'
LABELS = open(labelsPath).read().strip().split('\n')

# Setting path to weights and  config file
weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'

# Loading YOLO trained on COCO Dataset
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)


# In[3]:


for file in os.listdir('PennFudanPed/PNGImages'):
    
    # Loading Image
    img  = cv2.imread('PennFudanPed/PNGImages/'+file)
    (H, W) = img.shape[:2]
    
    # Getting Output layer names
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    # Creating blob from image to perform feature normalization
    blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)

    # Setting Input and performing forward pass through YOLO Network
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialing required lists
    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # If a person is detected with high confidence
            if (confidence>0.5) and (classID == 0):
            
                # scaling bounding box cordinates according to sise of original image
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX,centerY,width,height) = box.astype('int')
                
                # Converting into topleft coordinates
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
            
                # Appending to lists
                boxes.append((x,y,int(width),int(height)))
                confidences.append(float(confidence))

    # Applying Non-Maximum Suppression to remove duplicate boxes
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
    
    flag = 1
    if len(idxs)>0:
        
        path = 'Predictions/Predictions_YOLO/'+file.replace('.png','.txt')
        f = open(path,'w')
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # Saving predictions in txt file
            string = 'person '
            string += str(confidences[i])+' '+str(x)+' '+str(y)+' '+str(x+w)+' '+str(y+h)
            
            f.write(string)
            if flag == len(idxs.flatten()):
                break
            f.write('\n')
            flag += 1
        f.close()


# In[ ]:




