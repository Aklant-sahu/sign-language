import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow 
# import seaborn as sns
import datetime
import os
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

#model = tensorflow.keras.models.load_model('model/saved-model')
#path="img/"
count=0
#model.summary()
inputImage = cv2.imread('unnamed (5).jpg')


# Convert RGB to grayscale:
#grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Convert the BGR image to HSV:
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

# Create the HSV range for the blue ink:
# [128, 255, 255], [90, 50, 70] [10,77,67] [107,145,86]
l_b = np.array([0,40,42])
u_b = np.array([255,158,97])

# Get binary mask of the blue ink:
mask=cv2.inRange(hsvImage,l_b,u_b)

    #print(1)
cv2.imshow('inputImage',inputImage)
    #cv2.imshow('frame',frame)
cv2.imshow('mask',mask)

# capture=np.zeros((480, 640, 3))
# capture = cv2.imread('img.jpg',0)
# capture = cv2.filter2D(src=capture, ddepth=-1, kernel=kernel)
# capture = cv2.filter2D(src=capture, ddepth=-1, kernel=kernel)
# # capture = cv2.filter2D(src=capture, ddepth=-1, kernel=kernel)
# cv2.imshow('hcb',capture)
# im11=np.zeros((capture.shape[0],capture.shape[1]),np.uint8)
# m,n= capture.shape #tuple unpacking
# min=100
# max=250
# for i in range(m):
#     for j in range(n):
#         if (capture[i,j] >min) and (capture[i,j]<max): 
#         # if img[i,j] > 0: # white blob
#             im11[i,j]= 255
#         else:
#             im11[i,j] = capture[i,j]
# cv2.imshow('lol',im11)
# cv2.imwrite('lol.jpg',im11)
cv2.waitKey(0)