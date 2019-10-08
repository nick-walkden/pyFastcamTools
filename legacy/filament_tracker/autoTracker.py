"""
Class to automatically track filaments in a given movie
"""

from pyFastcamTools.readMovie import readMovie
from pyFastcamTools.fieldlineTracer import getfieldlineTracer
from pyFastcamTools.backgroundSubtractor import *
from utils import *
import numpy as np
from copy import deepcopy as copy
import cv2

try:
	from CamTools import *
except:
	raise ImportError("ERROR: No Camtools module found. Needed for calibration.")


class autoTracker(object):
	
	def __init__(self,Nframes=None,startpos=0.0,gfile=None,moviefile=None,calibfile=None,tracertype='RK4'):
		
		
		if not gfile:
			gfile = getUserFile(type="equilibrium")
		self.tracer = getfieldlineTracer(type=tracertype,gfile=gfile)

		if not moviefile:
			moviefile = getUserFile(type="movie")		
		self.frames = readMovie(moviefile,Nframes=Nframes,startpos=startpos)
		self._currentframeNum = 0
		self._currentframe = self.frames[self._currentframeNum]	
		if not calibfile:
			calibfile = getUserFile(type="calibration")
		self.calibration = Fitting.CalibResults(calibfile)
		
		self.mask = None
		
		self.bgsub = backgroundSubtractorMin(10)
		self.bgsub.setHistory(self.frames)
		
		
	def setROI(self):
		self.frames.setROI()
		self.mask = self.frames.getROImask()
		self.frames[...] *= self.mask
		self.bgsub.setHistory(self.frames)
			
	def projectFieldline(self,fieldline):
		objpoints = np.array([[fieldline.X[i],fieldline.Y[i],fieldline.Z[i]] for i in np.arange(len(fieldline.X))])
		return self.calibration.ProjectPoints(objpoints)[:,0,:]		
	
	def sumIntensity(self,fieldline,frame):
		points = self.projectFieldline(fieldline)
		from scipy.interpolate import interp2d
		
		#Get frame intensity as a function of pixel coordinate
		#if self.mask != None:
		#	
		#	if len(frame.shape) > 2:
		#		#Convert to intensity
		#		newframe = np.sum(copy(frame*self.mask)**2.0,axis=2)**0.5
		#	else:
		#		newframe = copy(frame*self.mask)
		#else:
		if len(frame.shape) > 2:
			#Convert to intensity
			newframe = np.sum(copy(frame)**2.0,axis=2)**0.5
		else:
			newframe = copy(frame)
		
		intensity = np.vectorize(interp2d(np.arange(newframe.shape[0]),np.arange(newframe.shape[1]),newframe[:,:],kind='linear'))	
		total = np.sum(intensity(np.array(points[:,0]),np.array(points[:,1])))
		return total
	
	def getToroidalDist(self,R,Z,nphi=360):
		line = self.tracer.trace(R,Z,mxstep=1000,ds=0.1)
		
		intensity = []		
		dphi = 2.0*np.pi/nphi
		
		_,frame = cv2.threshold(self.bgsub.apply(self._currentframe*self.mask),5,255,cv2.THRESH_BINARY)
		
		for i in np.arange(nphi):
			print i
			intensity.append(self.sumIntensity(line,frame))
			line.rotateToroidal(dphi)
	
		return intensity
		
	def nextFrame(self):
		self._currentframeNum += 1
		self._currentframe = self.frames[self._currentframeNum]
		