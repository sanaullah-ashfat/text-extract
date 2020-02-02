# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:25:54 2020

@author: Sanaullah
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import matplotlib.pyplot as plt



def decode_predictions(scores, geometry):
 	
 	(numRows, numCols) = scores.shape[2:4]
 	rects = []
 	confidences = []
 	for y in range(0, numRows):
         	
		 scoresData = scores[0, 0, y]
		 xData0 = geometry[0, 0, y]
		 xData1 = geometry[0, 1, y]
		 xData2 = geometry[0, 2, y]
		 xData3 = geometry[0, 3, y]
		 anglesData = geometry[0, 4, y]

		 for x in range(0, numCols):
 			 if scoresData[x] < .5:   
				  continue
 			 (offsetX, offsetY) = (x * 4.0, y * 4.0)
 			 angle = anglesData[x]
 			 cos = np.cos(angle)
 			 sin = np.sin(angle)
 			 h = xData0[x] + xData2[x]
 			 w = xData1[x] + xData3[x]
 			 endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
 			 endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
 			 startX = int(endX - w)
 			 startY = int(endY - h)

 			 rects.append((startX, startY, endX, endY))
 			 confidences.append(scoresData[x])

 	
 	return rects, confidences


def text(a,b,c,d,e):
    boxes = non_max_suppression(np.array(a), probs=b)
    results = []
    rW = c / float(320)
    rH = d / float(320)
  
    for (startX, startY, endX, endY) in boxes:
     	
     	startX = int(startX * rW)
     	startY = int(startY * rH)
     	endX = int(endX * rW)
     	endY = int(endY * rH)
    
     	dX = int((endX - startX) * 0.0)
     	dY = int((endY - startY) * 0.0)

     	startX = max(0, startX - dX)
     	startY = max(0, startY - dY)
     	endX = min(c, endX + (dX * 2))
     	endY = min(d, endY + (dY * 2))
    
     	roi = e[startY:endY, startX:endX]
    
     	config = ("-l eng --oem 1 --psm 7")
     	text = pytesseract.image_to_string(roi, config=config,lang='eng+ben')

     	results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key=lambda r:r[0][1])

    for ((startX, startY, endX, endY), text) in results:
    
     	print("{}".format(text))
         
         
def detect_text(a,b,c,d,e):
    boxes = non_max_suppression(np.array(a), probs=b)
    rW = c / float(320)
    rH = d / float(320)    
    for (startX, startY, endX, endY) in boxes:
    
     	startX = int(startX * rW)
     	startY = int(startY * rH)
     	endX = int(endX * rW)
     	endY = int(endY * rH) 
     	cv2.rectangle(e, (startX, startY), (endX, endY), (0, 255, 0), 2)
         
    plt.imshow(e)



image =cv2.imread('C:/Users/Sanaullah/Desktop/Tasks/Task3-vision/frames/g.jpg')
model='G:/opencv-text-detection/frozen_east_text_detection.pb'

orig = image.copy()
oH, oW = image.shape[0],image.shape[1]

image = cv2.resize(image, (320, 320))
H, W = image.shape[0],image.shape[1]

layerNames = [
 	"feature_fusion/Conv_7/Sigmoid",
 	"feature_fusion/concat_3"]

net = cv2.dnn.readNet(model)
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
 	(123.68, 116.78, 103.94), swapRB=True)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
rects, confidences = decode_predictions(scores, geometry)



text(rects, confidences,oW,oH,orig)

detect_text(rects, confidences,oW,oH,orig)








