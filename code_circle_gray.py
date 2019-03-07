import cv2
import numpy as np
import math
from math import hypot,sqrt
import base64
from PIL import Image
import io
from matplotlib import pyplot as plt
import time
from os.path import join

filename = "images/0.jpg"
image = cv2.imread(filename, cv2.IMREAD_COLOR)
print("\n\n",filename)
image = cv2.medianBlur(image,3)

# print(image[:1])
selectOnImage_copy = image.copy()
marker_image = np.zeros(image.shape[:2],dtype=np.int32)
segments = np.zeros(image.shape,dtype=np.uint8)

n_markers = 3
# from matplotlib import cm
# def create_rgb(i):
#     return tuple(np.array(cm.tab10(i)[:3])*255)
# colors = []
# for i in range(n_markers):
#     colors.append(create_rgb(i))
# print(colors[1])
# print(colors[2])
colors = [(0,0,0),(255,255,255),(56,56,56)]

current_marker = 1
marks_updated = False
def mouse_callback(event,x,y,flags,params):
    global marks_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker_image,(x,y),1,(current_marker),-1)
        cv2.circle(selectOnImage_copy,(x,y),1,colors[current_marker],-1)
        marks_updated = True
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
while True:
    cv2.imshow('Image',selectOnImage_copy)
    cv2.imshow('Watershed Segments', segments)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('c'):
        selectOnImage_copy = image.copy()
        marker_image = np.zeros(image.shape[:2],dtype=np.int32)
        segments = np.zeros(image.shape, dtype=np.uint8)
    elif k>0 and chr(k).isdigit():
        current_marker = int(chr(k))
    
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(image,marker_image_copy)
        segments = np.zeros(image.shape,dtype=np.uint8)
        for color_ind in range(n_markers):
            segments[marker_image_copy==(color_ind)] = colors[color_ind]
cv2.imwrite('images/seg.jpg',segments)
cv2.destroyAllWindows()

cimg = cv2.imread("images/seg.jpg", cv2.IMREAD_GRAYSCALE)
circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1.5,120,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
# print(len(circles[0,:]))

img = image.copy()
# for circle in circles[0,:]:
circle = circles[0,:][0]
# draw the outer circle
cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),2)
# draw the center of the circle
cv2.circle(img,(circle[0],circle[1]),2,(0,0,255),3)

cv2.imshow('detected circle (pupil)',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.equalizeHist(image)
imgR = image.copy()
cv2.imshow('histEq',imgR)
cv2.waitKey(0)
cv2.destroyAllWindows()
x, y, r = circles[0,:][0]
rn = r-5
rows, cols = imgR.shape

npixels = 0
intensitySum = 0
bgrList=[]
for i in range(cols):
    for j in range(rows):
        if hypot(i-x, j-y) < rn:
            npixels+=1
            intensitySum+=image[i][j]

intensity = intensitySum*1.0/npixels
print(intensity)