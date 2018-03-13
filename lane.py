# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:43:55 2018

@author: Andrei Baraian
"""

#importing some useful packages
import matplotlib.pyplot as plt
import numpy as np
import cv2


## The ego point of the reference image, that will be used as the desired
## position of the car between the lanes
egoRefPoint = [345,257]

oneLane = False
max_right_error = -3 #dummy value, need to set a real value
max_left_error = 3 #dummy value, need to set a real value

    
def sobel_thresh(img, orient='x',thresh=(0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_threshold(img,sobel_kernel=3,thresh=(0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    mag = np.sqrt(x**2 + y**2)
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] = 1
    return binary_output

def dir_threshold(img,sobel_kernel=3,thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel))
    y = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel))
    direction = np.arctan2(y,x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output

def hls_select(img,sthresh=(0,255),lthresh=()):
    hls_img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    #cv2.imshow('hls',hls_img)
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                    & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output

def red_select(img,thresh=(0,255)):
    R=img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 255
    return binary_output

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
    [(0.5 * x) + (x*0.30), (1/3)*y],
    [x - (0.0 * x), y-100]
    ])
    
#    print(source_points)
    
#    destination_points = np.float32([
#    [0.05 * x, y],
#    [0.20 * x, 0],
#    [x - (0.20 * x), 0],
#    [x - (0.05 * x), y]
#    ])
    
    destination_points = np.float32([
    [50, y],
    [0, 0],
    [x, 0],
    [x -50, y]
    ])
    
    #print(destination_points)
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    
    warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)
    
    return warped_img, inverse_perspective_transform

def binary_pipeline(img):
    img_copy = cv2.GaussianBlur(img,(9,9),0)
    #cv2.imshow('gaussian',img_copy)
    
    #Color channel
    red_binary = red_select(img_copy,thresh=(200,255))
    #red_binary = cv2.GaussianBlur(red_binary,(9,9),0)
    cv2.imshow('filter white',red_binary)
    
    #Sobel 
    x_binary = sobel_thresh(img_copy,thresh=(50,150))
    y_binary = sobel_thresh(img_copy,orient='y',thresh=(50,150))
    xy = cv2.bitwise_and(x_binary,y_binary)
    
    #magnitude and direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3,thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3,thresh=(0.8,1.2))
    
    #stack each channnel
    gradient = np.zeros_like(red_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv2.bitwise_or(red_binary,gradient)
    
    return red_binary
    

##--------------------------------------------------------------------------
##  
## Find the center between the lanes and return it.
## !In case there is one lane, return the middle of that lane!
## The orientation of that lane will be calculated later


def find_ego(img):
    
    height = img.shape[0]
    width = img.shape[1]
    
    currentRow = int(0.9 * height)
    currentCol = 3
    
    ok = False
    while ok == False and currentCol < width - 1:
        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
        a = np.asarray(mat).reshape(-1)
        binary_mat = np.zeros(9,dtype='int')
        binary_mat[a > 150] = 1
        currentCol = currentCol + 1
        ok = np.bitwise_and.reduce(binary_mat)
        
    entry_line1 = currentCol
    #print('entry is ',entry_line1)
    ok = False
    while ok == False and currentCol < width - 1:
        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
        a = np.asarray(mat).reshape(-1)
        binary_mat = np.zeros(9,dtype='int')
        binary_mat[a < 150] = 1
        currentCol = currentCol + 1
        ok = np.bitwise_and.reduce(binary_mat)
        
    exit_line1 = currentCol
    #print('exit is ',exit_line1)
    #currentCol = currentCol + 30
    
    ok = False
    while ok == False and currentCol < width - 2:
        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
        a = np.asarray(mat).reshape(-1)
        #print(a,' ',currentCol,' width is ',width-2)
        binary_mat = np.zeros(9,dtype='int')
        binary_mat[a > 150] = 1
        currentCol = currentCol + 1
        ok = np.bitwise_and.reduce(binary_mat)
        
    entry_line2 = currentCol
     ## if the condition is true, it means that we have only one lane and we
     ## should further see if we have a right turning lane or a left one
    if entry_line2 == width - 2:
        global oneLane 
        oneLane = True
        return [currentRow, entry_line1 + ((exit_line1 - entry_line1) // 2)]
#    
#    ok = False
#    while ok == False:
#        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
#        a = np.asarray(mat).reshape(-1)
#        binary_mat = np.zeros(9,dtype='int')
#        binary_mat[a < 150] = 1
#        currentCol = currentCol + 1
#        ok = np.bitwise_and.reduce(binary_mat)
#        
#    exit_line2 = currentCol
    
    egoPoint = [currentRow,exit_line1 + (entry_line2 - exit_line1) // 2]
    
    #print('----',entry_line1,exit_line1,entry_line2,exit_line2)
    print('ego point is',egoPoint)
    
    return egoPoint


##--------------------------------------------------------------------------
##
## Decide if the lane is a right turning one or a left one.
##   Return 1 if we have a right turning lane
##   Return 0 if we have a right turning lane
##   Baseline is the middle point of that lane, situated in the lower part
##   of the picture
##
    
def findLaneOrientation(img, baseLine):
    
    height = img.shape[0]
    width = img.shape[1]
    
    currentRow = int(0.45 * height)
    currentCol = 3
    
    ok = False
    while ok == False and currentCol < width - 1:
        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
        a = np.asarray(mat).reshape(-1)
        binary_mat = np.zeros(9,dtype='int')
        binary_mat[a > 150] = 1
        currentCol = currentCol + 1
        ok = np.bitwise_and.reduce(binary_mat)
        
    entry_line1 = currentCol
    ok = False
    while ok == False and currentCol < width - 1:
        mat = img[currentRow-1:currentRow+2,currentCol-1:currentCol+2]
        a = np.asarray(mat).reshape(-1)
        binary_mat = np.zeros(9,dtype='int')
        binary_mat[a < 150] = 1
        currentCol = currentCol + 1
        ok = np.bitwise_and.reduce(binary_mat)
        
    exit_line1 = currentCol
    
    mx = entry_line1 + (exit_line1 - entry_line1) // 2
    if mx - baseLine[1] > 0:
        return 1  ## means we have a right turning lane
    else:
        return 0  ## means we have a left turning lane
    
    
##--------------------------------------------------------------------------
## 
## Calculate the error between the desired ego point and the actual one
## The error is calculated by substracting the x-axis points of the 
## desired ego point and the actual ego point
## 
## A positive results yields a right positioning and needs a left correction
## A negative results yields a left positioning and needs a right correction
##
        
def calculateError(img):
    global oneLane
    global max_right_error
    global max_left_error
    actualEgo = find_ego(img)
    if oneLane == True: ## the case of having just one lane
        oneLane = False
        laneTurning = findLaneOrientation(img,actualEgo)
        if laneTurning == 1:
            return max_right_error
        else:
            return max_left_error
    ## we have two lanes
    return egoRefPoint[1] - actualEgo[1]
    
    
    
    


nameOfImage = 'video/image25.png'    
originalImage = cv2.imread(nameOfImage)
gray_image = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
resizedImage = resizeImage(originalImage)
#cv2.imshow('gray image',gray_image)

pipeline_img = binary_pipeline(resizedImage)
cv2.imshow('pipeline image',pipeline_img)

cannyed_image = cv2.Canny(resizedImage,100,200);
#cv2.imshow('canny image',cannyed_image)
birdseye_result, inverse_perp_trans = warp_image(pipeline_img)

x = resizedImage.shape[1]
y = resizedImage.shape[0]

source_points = np.int32([
                    [0.0 * x, y-100],
                    [(0.5 * x) - (x*0.30), (1/3)*y],
                    [(0.5 * x) + (x*0.30), (1/3)*y],
                    [x - (0.0 * x), y-100]
                    ])

draw_poly = cv2.polylines(resizedImage,[source_points],True,(255,0,0), 5)

bx = birdseye_result.shape[1]
by = birdseye_result.shape[0]

scan_line1 = np.int32([
            [0,by - 0.1 * by],
            [bx,by - 0.1 * by]
            ])

#draw_poly = cv2.polylines(birdseye_result,[scan_line1],True,(100,100,0),5)
print('error is ',calculateError(birdseye_result))
#collision1 = find_ego(birdseye_result)
src = np.int32([[53,by - 0.1 * by],[257,by - 0.1 * by]])
draw_poly = cv2.polylines(birdseye_result,[src],True,(100,100,0),5)

#print(calculateError(birdseye_result))

cv2.imshow('bird-eye view',birdseye_result)
cv2.imshow('resized image',resizedImage)
##
##cv2.imshow('orig',gray_image)
#cv2.imshow('resized image',draw_poly)
##
#cv2.imshow('bird',birdseye_result)
##plt.imshow(birdseye_result)
#
#histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):,:],axis = 0)
#plt.figure();
#plt.plot(histogram);

#canny2 = cv2.Canny(birdseye_result,100,200)
#cv2.imshow('canny bird',canny2)
