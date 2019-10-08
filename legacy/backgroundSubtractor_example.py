#!/usr/bin/env python

"""
Example script for running a background subtractor

Nick Walkden, May 2015
"""
from pyFastcamTools.create_log import create_log
logger = create_log(__name__)
logger.info('submodule test')

import cv2
import numpy as np
from pyFastcamTools.backgroundSubtractor import backgroundSubtractorMedian,backgroundSubtractorMean,backgroundSubtractorMin,backgroundSubtractorFFT
from pyAutoGit import pyAutoGit as git
vid = cv2.VideoCapture('../movies/29840.avi')

bgsub = cv2.createBackgroundSubtractorMOG2()
bgsub2 = backgroundSubtractorMin(5)
bgsub3 = backgroundSubtractorMean(20)


ROIx0,ROIx1 = 0,256
ROIy0,ROIy1 = 0,160


cv2.namedWindow('Video',cv2.WINDOW_NORMAL)

while(1):
	ret,frame = vid.read()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame = frame[ROIx0:ROIx1,ROIy0:ROIy1]
	bgmask2 = bgsub2.apply(frame)
	bgmask = bgsub.apply(frame,learningRate=0.1)	
	
	#Extract foreground from original image
	bgframe = cv2.bitwise_and(frame,frame, mask= bgmask)
	bgframe2 = bgmask2
	
	#Enhance contrast of foreground
	ret,bgframe = cv2.threshold(bgframe,10,255,cv2.THRESH_BINARY)
	ret,bgframe2 = cv2.threshold(bgframe2,10,255,cv2.THRESH_BINARY)

	cv2.imshow('cv2 Subtracted',cv2.resize(bgframe, (0,0), fx=2.0, fy=2.0))
	cv2.imshow('Original',cv2.resize(frame,(0,0),fx=2.0,fy=2.0))
	cv2.imshow('Manual',cv2.resize(bgframe2,(0,0),fx=2.0,fy=2.0))
	
	#Display at 25fps (40ms = 1/25)
	k = cv2.waitKey(40) & 0xff
	
	#Exit is esc key is pushed 
	if k == 27:
		break
	
vid.release()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.imshow(bgsub2.backgroundModel,cmap = cm.Greys_r)
plt.show()

git.init()
git.commit()


