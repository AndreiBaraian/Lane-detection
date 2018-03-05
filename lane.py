# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:43:55 2018

@author: bara_
"""

#importing some useful packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
    
def draw_lines(img, lines, color=[0,0,255], thickness=3):
    if lines is None:
        return
    
    img = np.copy(img)
    
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),color,thickness)
            
    img = cv2.addWeighted(img,0.8,line_img,1.0,0.0)
    return img

def resizeImage(originalImage,newWidth=512):
    r = newWidth / originalImage.shape[1]
    dim = (newWidth, int(originalImage.shape[0] * r))
    resized = cv2.resize(originalImage,dim,interpolation = cv2.INTER_AREA)
    return resized

def regionOfInterest(image,percentage=20):
    newY = int((image.shape[0] / 100) * percentage)
    newImage = image[newY:image.shape[0],0:image.shape[1]]
    return newImage

def apply_smoothing(image,kernel_size=15):
    return cv2.GaussianBlur(image,(kernel_size, kernel_size), 0)

def warp_image(img):
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]
    
    source_points = np.float32([
    [0.0 * x, y-100],
    [(0.5 * x) - (x*0.30), (1/3)*y],
    [(0.5 * x) + (x*0.15), (1/3)*y],
    [x - (0.0 * x), y-100]
    ])
    
#    print(source_points)
    
    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    
    warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)
    
    #print(source_points)
    #print(destination_points)
    
    return warped_img, inverse_perspective_transform

nameOfImage = 'test_images/imaj7.jpg'    
originalImage = cv2.imread(nameOfImage)
gray_image = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray image',gray_image)

resizedImage = resizeImage(gray_image)
cv2.imshow('resized image',resizedImage)

#croppedImage = regionOfInterest(resizedImage)
#cv2.imshow('cropped image', croppedImage)

#blurredImage = apply_smoothing(croppedImage)
#cv2.imshow('blurred image',blurredImage)

#print('The dimension of blurred image ',blurredImage.shape[0],' ',blurredImage.shape[1])

cannyed_image = cv2.Canny(resizedImage,100,200)
#cv2.imshow('canny image',cannyed_image)

birdseye_result, inverse_perp_trans = warp_image(cannyed_image)

image_size = (resizedImage.shape[1], resizedImage.shape[0])
x = resizedImage.shape[1]
y = resizedImage.shape[0]
print(y,x)
source_points = np.int32([
                    [0.0 * x, y-100],
                    [(0.5 * x) - (x*0.30), (1/3)*y],
                    [(0.5 * x) + (x*0.15), (1/3)*y],
                    [x - (0.0 * x), y-100]
                    ])

draw_poly = cv2.polylines(resizedImage,[source_points],True,(255,0,0), 5)
#
#cv2.imshow('orig',gray_image)
cv2.imshow('resized image',draw_poly)
#
cv2.imshow('bird',birdseye_result)

#canny2 = cv2.Canny(birdseye_result,100,200)
#cv2.imshow('canny bird',canny2)

#cannyed_image = cv2.Canny(birdseye_result,100,200)
#cv2.imshow('canny image',cannyed_image)

#cannyed_image = cv2.Canny(blurredImage,100,200)
#cv2.imshow('canny image',cannyed_image)

#cv2.imshow('originalImage',img)
#print('This image has dimensions: ',img.shape)
#height = img.shape[0]
#width = img.shape[1]
#
#r = 500.0 / img.shape[1]
#dim = (500, int(img.shape[0] * r))
#resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#
##cv2.imshow('resized',resized)
#height = resized.shape[0]
#width = resized.shape[1]
#
#print('The new dim are ',height,' ',width)
#
#
##region_of_interest_vertices=[(0,height),(width/2,height/2),(width,height)]
##cv2.imshow('croppedImage',cropped_img)
#
#
#gray_image = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
#cannyed_image = cv2.Canny(gray_image,100,200)
#
#cropped_image = resized[100:height,0:width]
#cropped_img = cannyed_image[100:height,0:width]

##cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices],np.int32))
#lines = cv2.HoughLinesP(cropped_img,rho=6,theta=np.pi/60,threshold=160,lines=np.array([]),minLineLength=40,maxLineGap=25)
#print(lines)
##
#line_image = draw_lines(cropped_image,lines)
#
#left_line_x = []
#left_line_y = []
#right_line_x = []
#right_line_y = []
#
#left_line_x = []
#left_line_y = []
#right_line_x = []
#right_line_y = []
#
#
#for line in lines:
#    for x1, y1, x2, y2 in line:
#        slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
#        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
#            continue
#        if slope <= 0: # <-- If the slope is negative, left group.
#            left_line_x.extend([x1, x2])
#            left_line_y.extend([y1, y2])
#        else: # <-- Otherwise, right group.
#            right_line_x.extend([x1, x2])
#            right_line_y.extend([y1, y2])
#            
##min_y = int(img.shape[0] * (3 / 5)) # <-- Just below the horizon
##max_y = img.shape[0] # <-- The bottom of the image
#
#poly_left = np.poly1d(np.polyfit(
#    left_line_y,
#    left_line_x,
#    deg=1
#))
#
#left_x_start = int(poly_left(max_y))
#left_x_end = int(poly_left(min_y))
#
#poly_right = np.poly1d(np.polyfit(
#    right_line_y,
#    right_line_x,
#    deg=1
#))
#
#right_x_start = int(poly_right(max_y))
#right_x_end = int(poly_right(min_y))
#
#line_image = draw_lines(
#    cropped_image,
#    [[
#        [left_x_start, max_y, left_x_end, min_y],
#        [right_x_start, max_y, right_x_end, min_y],
#    ]],
#    thickness=5,
#)
#
#cv2.imshow('grayImage',gray_image)
#cv2.imshow('cannyImage',cannyed_image)
#cv2.imshow('croppedImage',cropped_img)
#cv2.imshow('line_img',line_image)
#cv2.imshow('two_lines',line_img)
