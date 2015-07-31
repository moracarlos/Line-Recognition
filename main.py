import cv2
import numpy as np
from matplotlib import pyplot as plt


def showImage(name):
	cv2.imshow('image',name)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Invert->threshold(Mean)

#Load image
img = cv2.imread('formato2.jpg')
img = cv2.resize(img,(768,1024), interpolation = cv2.INTER_LINEAR)
img = cv2.GaussianBlur(img,(3,3),0)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,75,10) #11,2
#showImage(mean)
ret,thresh = cv2.threshold(mean,127,255,cv2.THRESH_BINARY_INV)
#showImage(thresh)

kernel = np.ones((4,4),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)
#showImage(erosion)
cv2.imwrite('erosion.png',erosion)

image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
indexMax = 0
x,y,w,h = cv2.boundingRect(contours[0])
maxArea = w*h

contoursSorted = sorted(contours, key=cv2.contourArea, reverse=True)

"""
for i in range(0,len(contours)):
	cnt = contours[i]
	x,y,w,h = cv2.boundingRect(cnt)
	if maxArea < w*h:
		maxArea=w*h
		indexMax = i
"""
cnt = contoursSorted[0]
x,y,w,h = cv2.boundingRect(cnt)
cnt2 = contoursSorted[1]
xn,yn,wn,hn = cv2.boundingRect(cnt2)

#showImage(erosion)
cv2.imwrite('erosion.png',erosion)

img = cv2.imread("formato2.jpg")
crop_img = img[y+12: y+(h+5), x+12:x+(w+5)] # Crop from x, y, w, h -> 100, 200, 300, 400
crop_img1 = img[yn+25: yn+(hn+12), xn+12:xn+(wn+5)]

showImage(crop_img)
showImage(crop_img1)
