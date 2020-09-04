# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:06:05 2020

@author: Sebastian
"""


import cv2
import numpy as np
from skimage.util import random_noise

def addNoiseGaus(img):
    dst = np.empty_like(img)
    noise = cv2.randn(dst, (80,80,80), (20,20,20))
    out = cv2.addWeighted(img, 0.5, noise, 0.5, 30)
    return out

def addNoiseSP(img):
    noise_img = random_noise(img, mode='s&p',amount=0.005)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

cap = cv2.VideoCapture('EditedVideos\secondvideo.mp4')

# 30 FPS
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# Set some params e.g. start of gaussian
frameNr = 0
gaus = 1
bilat = 0
sobelx = 1
sobely = 1
houghP1 = 50
houghP2 = 20
houghMin = 0
houghMax = 80
houghMinDist = 16
blue = 1
color = (255, 0, 0) 
template = cv2.imread('template2.png', 1)

w, h, c = template.shape

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameNr = frameNr + 1
    if(frameNr < 1136):
        continue
    
    if ret == True:
        # Switch to grayscale couple of times
        if (frameNr >= 0 and frameNr < 30) or (frameNr >= 60 and frameNr < 90):
            # Convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Apply bilateral filter
        if (frameNr >=121 and frameNr < 241):           
            if frameNr%40==0:
                print('Bilateral increased')
                bilat = bilat+5      
            if frameNr>160:
                # Set filter size to 5 by 5
                frame = cv2.bilateralFilter(frame,bilat, bilat*10, bilat*10, cv2.BORDER_DEFAULT)

        
        # Smooth the image with increasing gaussian filter
        if (frameNr >=241 and frameNr < 360):
            # Add some salt and pepper noise
            frame = addNoiseSP(frame)
            
            if frameNr%40==0:
                print('Gaussian increased')
                gaus = gaus+4      
            # Set filter size to 5 by 5
            frame = cv2.GaussianBlur(frame,(gaus,gaus),cv2.BORDER_DEFAULT)
                
        # Start the thresholding
        if(frameNr >= 390 and frameNr < 450):
            frame = cv2.inRange(frame,(120,120,120),(255,255,255))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Improve with gaussian blur to remove stars and visualise  
        if(frameNr >= 450 and frameNr < 601):
            improv = cv2.inRange(frame,(120,120,120),(255,255,255))
            improv = cv2.cvtColor(improv, cv2.COLOR_GRAY2BGR)
           
            frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
            frame = cv2.inRange(frame,(120,120,120),(255,255,255))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            diff = cv2.subtract(improv,frame)
            Conv_hsv_Gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
            frame[mask != 255] = [0, 0, 255]
            
        # Horizontal edge detection    
        if(frameNr >= 601 and frameNr < 660):
            if(frameNr%630==0):
                sobelx = sobelx + 2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=sobelx) 
            frame = np.absolute(frame)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Vertical edge detection    
        if(frameNr >= 660 and frameNr < 720):
            if(frameNr%690==0):
                sobely = sobely + 2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=sobely) 
            frame = np.absolute(frame)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Combination?    
        if(frameNr >= 720 and frameNr < 765):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Sobel(frame,cv2.CV_64F,1,1,ksize=5) 
            frame = np.absolute(frame)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # First a good result
        if(frameNr >=765 and frameNr < 810):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 19)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/16, param1=200, param2=20, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)    
        
        # Allow closer circles
        if(frameNr >=810 and frameNr < 870):
            if(frameNr%4==0):
                houghMinDist = houghMinDist * 2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 19)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/houghMinDist, param1=200, param2=20, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)    

        # Min radius
        if(frameNr >=870 and frameNr < 930):        
            if(frameNr%4==0):
                houghMin = houghMin+3
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 19)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/16, param1=200, param2=20, minRadius=houghMin, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)   

        if(frameNr >=930 and frameNr < 990):   
            if(frameNr%4==0):
                houghP2 = houghP2+1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 19)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/16, param1=200, param2=houghP2, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)           
        
        # Good result again
        if(frameNr >=990 and frameNr < 1076):   
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 19)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/16, param1=200, param2=20, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)     
        
        # Start tracking australia with rectangle
        if(frameNr >=1076 and frameNr < 1136):
            # Apply template Matching
            res = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Compute rectangle
            top_left = max_loc
            bottom_right = (top_left[0] + h, top_left[1] + w)
            
            if(frameNr%8==0):
                if(blue):
                    color = (0, 0, 255)
                    blue = 0
                else:
                    color = (255, 0, 0) 
                    blue = 1
            frame = cv2.rectangle(frame,top_left, bottom_right, color, 3)
        
        
        # Likelihood map
        if(frameNr >=1136 and frameNr < 1225):
            # Apply template Matching
            frame = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF)
            frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Compute rectangle
            top_left = max_loc
            bottom_right = (top_left[0] + h, top_left[1] + w)
            
            if(frameNr%8==0):
                if(blue):
                    color = (0, 0, 255)
                    blue = 0
                else:
                    color = (255, 0, 0) 
                    blue = 1
            frame = cv2.rectangle(frame,top_left, bottom_right, color, 3)
            frame = cv2.resize(frame,(1280,720), interpolation = cv2.INTER_AREA)
                
            
        out.write(frame)
            
            
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
            break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
