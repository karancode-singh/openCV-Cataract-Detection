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

filename = "1.jpg"
image = cv2.imread("images/"+filename, cv2.IMREAD_COLOR)
print("\n\n",filename)
image = cv2.medianBlur(image,3)

# selectOnImage_copy = image.copy()
# marker_image = np.zeros(image.shape[:2],dtype=np.int32)
# segments = np.zeros(image.shape,dtype=np.uint8)

# n_markers = 3
# # from matplotlib import cm
# # def create_rgb(i):
# #     return tuple(np.array(cm.tab10(i)[:3])*255)
# # colors = []
# # for i in range(n_markers):
# #     colors.append(create_rgb(i))
# # print(colors[1])
# # print(colors[2])
# colors = [(0,0,0),(255,255,255),(56,56,56)]

# current_marker = 1
# marks_updated = False
# def mouse_callback(event,x,y,flags,params):
#     global marks_updated
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(marker_image,(x,y),1,(current_marker),-1)
#         cv2.circle(selectOnImage_copy,(x,y),1,colors[current_marker],-1)
#         marks_updated = True
# cv2.namedWindow('Image')
# cv2.setMouseCallback('Image', mouse_callback)
# while True:
#     cv2.imshow('Image',selectOnImage_copy)
#     cv2.imshow('Watershed Segments', segments)
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#     elif k == ord('c'):
#         selectOnImage_copy = image.copy()
#         marker_image = np.zeros(image.shape[:2],dtype=np.int32)
#         segments = np.zeros(image.shape, dtype=np.uint8)
#     elif k>0 and chr(k).isdigit():
#         current_marker = int(chr(k))
    
#     if marks_updated:
#         marker_image_copy = marker_image.copy()
#         cv2.watershed(image,marker_image_copy)
#         segments = np.zeros(image.shape,dtype=np.uint8)
#         for color_ind in range(n_markers):
#             segments[marker_image_copy==(color_ind)] = colors[color_ind]
# cv2.imwrite("images/seg"+filename,segments)
# cv2.destroyAllWindows()

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.equalizeHist(image)
seg = cv2.imread("images/seg"+filename, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('detected circle (pupil)',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

npixels = 0
intensitySum = 0
histProbSum = 0
bgrList=[]
hist = {}
devSum = 0
for i in range(seg.shape[0]):
    for j in range(seg.shape[1]):
        if seg[i][j] == 255:
            npixels+=1
            if image[i][j] in hist:
                hist[image[i][j]] += 1
            else:
                hist[image[i][j]] = 1
for i in range(seg.shape[0]):
    for j in range(seg.shape[1]):
        if seg[i][j] == 255:
            p = image[i][j]
            intensitySum += image[i][j]
            histProbSum += (hist[image[i][j]]*1.0/npixels)
intensity = intensitySum*1.0/npixels
uniformity = (histProbSum*1.0/(npixels))**2
print('intensity', intensity)
print('uniformity', uniformity)
for i in range(seg.shape[0]):
    for j in range(seg.shape[1]):
        if seg[i][j] == 255:
            p = image[i][j]
            devSum+=(p-intensity)**2
sDev = sqrt(devSum*1.0/npixels)
print('standard deviation', sDev)
