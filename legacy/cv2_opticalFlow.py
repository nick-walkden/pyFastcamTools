#!/usr/bin/env python

"""
Example script to create a quiver plot showing the optical flow of information within a video using openCV

Nick Walkden, May 2015
"""

import cv2
import numpy as np
from backgroundSubtractor import backgroundSubtractorMin


vid = cv2.VideoCapture('../29840.avi')

bgsub2 = backgroundSubtractorMin(5)

ROIx0,ROIx1 = 0,256
ROIy0,ROIy1 = 0,160


ret, prvs = vid.read()
prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

prvs = prvs[ROIx0:ROIx1,ROIy0:ROIy1]
prvs = bgsub2.apply(prvs)

ret,prvs = cv2.threshold(prvs,10,255,cv2.THRESH_BINARY)

def draw_flow(img, flow, step=8):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	
	cv2.polylines(vis, lines, 0, (0, 0, 255))
	#for (x1, y1), (x2, y2) in lines:
	#    cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
	return vis
	
#Now try optical flow calculation

while(1):
	
	ret,nxt = vid.read()
	nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
	nxt = nxt[ROIx0:ROIx1,ROIy0:ROIy1]
	nxt = bgsub2.apply(nxt)
	ret,nxt = cv2.threshold(nxt,8,255,cv2.THRESH_BINARY)
	flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 5, 1, 3, 1.2, 0)
	
	prvs = nxt
	img = cv2.resize(draw_flow(prvs,flow),(0,0),fx=2.0,fy=2.0)
	cv2.imshow('Information Flow',img)
	
	
	k = cv2.waitKey(40) & 0xff
	
	#Exit is esc key is pushed 
	if k == 27:
		break
