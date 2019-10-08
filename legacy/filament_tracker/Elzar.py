"""
Class to manually track filaments in a given movie
"""

from pyFastcamTools.readMovie import readMovie
from fieldlineTracer import get_fieldline_tracer
from pyFastcamTools.backgroundSubtractor import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor,Button,Slider,CheckButtons
from matplotlib.text import Annotation as annotate
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colors as colors
from utils import *
import numpy as np
from copy import deepcopy as copy
import cv2
from scipy.interpolate import interp2d
from tkFileDialog import asksaveasfile

try:
	import Fitting,MachineGeometry,Render
except:
	print("WARNING: No Camtools module found. Needed for calibration.")


class Elzar(object):
	
	def __init__(self,Nframes=None,startpos=0.0,gfile=None,moviefile=None,calibfile=None,machine='MAST',shot=None,time=0.3):
		
		if shot is not None:
			self.frames = readMovie(str(shot),Nframes=Nframes,startpos=startpos)
			print("Finished reading movie file")
			
			try:
				self.calibration = Fitting.CalibResults(str(shot))
				self.DISABLE_PROJECT = False
			except:
				print("WARNING: Unable to read calibration file. Camera projection disabled.")
				self.DISABLE_PROJECT = True
			
			self.tracer = get_fieldline_tracer(type='RK4',machine='MAST',shot=shot,time=time,interp='linear')
			self.DISABLE_TRACER = False

		else:
			if gfile is None:
				gfile = raw_input("gfile: ")
			try:				
				self.tracer = get_fieldline_tracer(type='RK4',machine='MAST',shot=29840,time=time,interp='linear')
				self.DISABLE_TRACER = False
			except:
				print "WARNING: Unable to load gfile "+gfile+"! Field line tracing disabled."
				self.DISABLE_TRACER = True
			if moviefile is None:
				moviefile = raw_input("Movie file or shot number: ")	
			print("Reading Movie file, please be patient")				
			self.frames = readMovie(moviefile,Nframes=Nframes,startpos=startpos)
			print("Finished reading movie file")

			if calibfile is None:
				calibfile = raw_input("Calibration file: ")
			try:
				self.calibration = Fitting.CalibResults(calibfile)
				self.DISABLE_PROJECT = False
			except:
				print("WARNING: Unable to read calibration file. Camera projection disabled.")
				self.DISABLE_PROJECT = True


		self.flines = []
		self.projectLines = []
					
		self._currentframeNum = 0
		self._currentframe = self.frames[self._currentframeNum]	
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
				
		self.bgsub = backgroundSubtractorMin(20)
		#Initialize background model
		for frame in self.frames[0:19]:
			dummy = self.bgsub.apply(frame)
		
		self.widgets = {}
		self.dataCursor = None
	
		try:
			self.CADMod = MachineGeometry.MAST('high')
			self.wireframe = Render.MakeRender(self.CADMod,self.calibration,Verbose=False,Edges=True,EdgeWidth=1,EdgeMethod='simple')
			self.DISABLE_CCHECK = False
		except:
			print("WARNING: No CAD model found for MAST. Disabling calibration checking.")
			self.DISABLE_CCHECK = True
		
		self._currentZ = 0.0
		self._currentR = 1.45
		self._currentphi = 0.0

	def runUI(self):
		
		#Initialize some parameters for the UI
		self.gammaEnhance = False
		self.applySub = False
		self.threshold = False
		self.histEq = False
		self.edgeDet = False
		self.noiseRmv = False
		self.gamma = 1.0
		self.fieldlineArtists = []
		self.flineTxt = None
		self.selectedPixel = None
		self.pixelplot = None
		self.selectedLine = None
		self.linePlot = None
		axcolor = 'lightgoldenrodyellow'
		self.mask = copy(self._currentframe)
		self.mask[...]	= 1
		self.mask = np.uint8(self.mask)
		self.wireframeon = None
		
		#Set up UI window
		fig = plt.figure(figsize=(8,8),facecolor='w',edgecolor='k')
		#Set up axes for displaying images
		frameax = plt.axes([0.0,0.25,0.6,0.6])
		frame = self.enhanceFrame(copy(self._currentframe))
		self.img = frameax.imshow(frame)
		frameax.set_axis_off()
		frameax.set_xlim(0,frame.shape[1])
		frameax.set_ylim(frame.shape[0],0)
		text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
		self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
		frameax.add_artist(self.frametxt)
		
		#Set up axis for equilibrium plot
		eqax = plt.axes([0.7,0.3,0.25,0.5])
		eqax.set_xlim(0.0,2.0)
		eqax.set_ylim(-2.0,2.0)
		eqax.set_title('Poloidal Cross-section')
		eqax.set_ylabel('Z (m)')
		eqax.set_xlabel('R (m)')
		if not self.DISABLE_TRACER: wallplot = eqax.plot(self.tracer.eq.wall['R'],self.tracer.eq.wall['Z'],'-k')
		
		#Image enhancement selector
		enhancelabels = ('BG subtraction','Threshold','Gamma enhance','Detect edges','Equalise','Reduce Noise')
		enhanceCheck = CheckButtons(plt.axes([0.7,0.05,0.25,0.16]),enhancelabels,(False,False,False,False,False,False))
		gammaSlide = Slider(plt.axes([0.75,0.02,0.2,0.02],axisbg=axcolor), 'Gamma', 0.0, 3.0, valinit=1.0 )
		self._enhancedFrame = self._currentframe		
		def setEnhancement(label):
			if label == 'BG subtraction': self.applySub = not self.applySub
			elif label == 'Threshold' : self.threshold = not self.threshold
			elif label == 'Gamma enhance' : self.gammaEnhance = not self.gammaEnhance
			elif label == 'Detect edges'  : self.edgeDet = not self.edgeDet
			elif label == 'Equalise' : self.histEq = not self.histEq
			elif label == 'Reduce Noise' : self.noiseRmv = not self.noiseRmv
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)
		enhanceCheck.on_clicked(setEnhancement)
		
		def updateGamma(val):
			self.gamma = val
		gammaSlide.on_changed(updateGamma)
		
		#Field line launching area
		axR = plt.axes([0.2, 0.1, 0.35, 0.02], axisbg=axcolor)
		Rslide = Slider(axR, '$R$', 0.2, 2.0, valinit = 1.41 )
		self._currentR = 1.41	
		
		def updateR(val):
			self._currentR = val
		Rslide.on_changed(updateR)
		
		axZ = plt.axes([0.2, 0.06, 0.35, 0.02], axisbg=axcolor)
		Zslide = Slider(axZ, '$Z$', -2.0, 2.0, valinit = 0.0 )
		
		
		def updateZ(val):
			self._currentZ = val
		Zslide.on_changed(updateZ)
		
		axPhi = plt.axes([0.2,0.02,0.35,0.02],  axisbg=axcolor)
		PhiSlide = Slider(axPhi, '$\phi$', 0, 360, valinit = 180 )
		self._currentphi = 60.0
		
		def updatePhi(val):
			if not self.DISABLE_TRACER:
				dphi = val - self._currentphi
				if not self.flines:
					self._currentphi = val
					return
					
				ang = 2.0*np.pi*dphi/360.0
				self.flines[-1].rotateToroidal(ang)
				linepoints = self.projectFieldline(self.flines[-1])
				self.projectLines[-1] = linepoints
				self.fieldlineArtists[-1].set_data(linepoints[:,0],linepoints[:,1])
				fig.canvas.draw()
				self._currentphi = val 
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(linepoints)),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			
		PhiSlide.on_changed(updatePhi)
		
		launchButton = Button(plt.axes([0.01,0.08,0.16,0.05]),'Launch',hovercolor='r')
		def onpick(event,annotation):
			if not self.DISABLE_TRACER:
				thisline = event.artist
				thisind = self.fieldlineArtists.index(thisline)
				thisfline = self.flines[thisind]
				x,y = annotation.xy
				lineind = findNearest(self.projectLines[thisind],(x,y))
				R = thisfline.R[lineind]
				Z = thisfline.Z[lineind]
				psiN = self.tracer.eq.psiN(R,Z)
				phi = (thisfline.phi[lineind]*360.0/(2*np.pi)) % 360.0 
				annotation.set_text(self.dataCursor.template % (R,Z,phi,psiN))
				annotation.set_visible(True)
				event.canvas.draw()
			
		def launchFieldline(event):
			if not self.DISABLE_TRACER:
				self.flines.append(self.tracer.trace(self._currentR,self._currentZ,phistart = 2.0*np.pi*self._currentphi/360.0,mxstep=1000,ds=0.05))
				linepoints = self.projectFieldline(self.flines[-1])
				self.projectLines.append(linepoints)
				flineplot, = frameax.plot(linepoints[:,0],linepoints[:,1],picker=5,lw=1.5)
				self.fieldlineArtists.append(flineplot)
				self.eqlineplot, = eqax.plot(self.flines[-1].R,self.flines[-1].Z)
				if self.dataCursor:
					self.dataCursor.clear(fig)
					self.dataCursor.disconnect(fig)
				self.dataCursor = DataCursor(self.fieldlineArtists,func=onpick,template="R: %.2f\nZ: %.2f\nphi: %.2f\npsiN: %.2f")	
				self.dataCursor.connect(fig)	
				if self.flineTxt:
					self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(linepoints)),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			else:
				print("WARNING: Cannot launch field line, tracer is disabled!")
		launchButton.on_clicked(launchFieldline)
					
		def onRelease(event):
			if self.dataCursor:
				self.dataCursor.clear(fig)
		fig.canvas.mpl_connect('button_release_event', onRelease)	
		
		clearButton = Button(plt.axes([0.01,0.02,0.16,0.05]),'Clear',hovercolor='r')
		
		def clearFieldlines(event):
			frameax.clear()
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img = frameax.imshow(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.set_axis_off()
			frameax.set_xlim(0,frame.shape[1])
			frameax.set_ylim(frame.shape[0],0)
			eqax.clear()
			eqax.set_xlim(0.0,2.0)
			eqax.set_ylim(-2.0,2.0)
			eqax.set_title('Poloidal Cross-section')
			eqax.set_ylabel('Z (m)')
			eqax.set_xlabel('R (m)')
			if not self.DISABLE_TRACER: wallplot = eqax.plot(self.tracer.eq.wall['R'],self.tracer.eq.wall['Z'],'-k')
			self.flines = []
			self.fieldlineArtists = []
			self.projectLines = []
			self.dataCursor = None
			#self.flineTxt.set_visible(False)
			self.flineTxt = None
			self.selectedLine = None
		clearButton.on_clicked(clearFieldlines)
		
		#Frame selection section
		
		nextButton =  Button(plt.axes([0.01,0.9,0.13,0.05]),'Next',hovercolor='r')
		
		def plotNext(event):
			self.nextFrame()
			frame = self.mask*self.enhanceFrame(self._currentframe)
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
		nextButton.on_clicked(plotNext)
		
		prevButton =  Button(plt.axes([0.15,0.9,0.13,0.05]),'Previous',hovercolor='r')
		
		def plotPrev(event):
			self.previousFrame()
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
		prevButton.on_clicked(plotPrev)
			
		refreshButton = Button(plt.axes([0.29,0.9,0.13,0.05]),'Refresh',hovercolor='r')
		
		def refreshPlot(event):
			self.mask = copy(self._currentframe)
			self.mask[...] = 1
			self.mask = np.uint8(self.mask)
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)	
		refreshButton.on_clicked(refreshPlot)

		saveButton = Button(plt.axes([0.43,0.9,0.13,0.05]),'Save',hovercolor='r')
		
		def saveFrame(event):
			"""
			Save both current plot in frame axis, and save image data as pngs
			"""
			savefile = raw_input("Safe file: ")
			extent = frameax.get_position().get_points()
			inches = fig.get_size_inches()
			extent *= inches
			bbox = frameax.get_position()
			bbox.set_points(extent)
			if savefile != '':
				fig.savefig(savefile,bbox_inches = bbox)
				savefileparts = savefile.split('.')
				plt.imsave(savefileparts[0]+"DATA.png",self._enhancedFrame)
		saveButton.on_clicked(saveFrame)
		
		ROIButton = Button(plt.axes([0.72,0.84,0.22,0.05]),'Set ROI',hovercolor='r')
		
		self.selector = ROISelector(self.img)
		def setROI(event):
			self.ROI = np.asarray(self.selector.coords)
			self.mask[self.ROI[0,1]:self.ROI[1,1],self.ROI[0,0]:self.ROI[1,0]] = int(0)
			self.mask = int(1) - self.mask
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)
		ROIButton.on_clicked(setROI)
		
		calibButton = Button(plt.axes([0.72,0.9,0.22,0.05]),'Check Calibration',hovercolor='r')
		
		def checkCalibration(event):
			if not self.DISABLE_CCHECK:	
				if self.wireframeon == None:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					self.wireframeon = True
					return
			
			
				if self.wireframeon:
					#Already displaying so refresh screen
					frameax.clear()
					frame = self.mask*self.enhanceFrame(copy(self._currentframe))
					self.img = frameax.imshow(frame)
					text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
					self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
					frameax.set_axis_off()
					frameax.set_xlim(0,frame.shape[1])
					frameax.set_ylim(frame.shape[0],0)
					self.fieldlineArtists = []
					if self.projectLines:
						for line in self.projectLines:
							lplot, = frameax.plot(line[:,0],line[:,1],picker=5,lw=1.5)
							self.fieldlineArtists.append(lplot)
						self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
						xycoords='axes fraction',color='white',fontsize=8)
				else:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					
				self.wireframeon = not self.wireframeon
			else:
				print("WARNING: Calibration checking disabled!")
				
		calibButton.on_clicked(checkCalibration)
		
		#Some Image analysis options
		correlateButton = Button(plt.axes([0.47,0.18,0.11,0.05]),'Correlate',hovercolor='r')
		
		def getCorrelation(event):
			if not self.selectedPixel:
				return
			corr = self.frameCorrelation((self.selectedPixel[1],self.selectedPixel[0]))
			corrfig,corrax = plt.subplots()
			levels = np.linspace(-1.0,1.0,100)
			corrplot = corrax.imshow(corr,cmap = cm.coolwarm,norm = colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False))
			pixel = corrax.scatter(self.selectedPixel[0],self.selectedPixel[1],s=70,c='g',marker='+')
			corrax.set_title('Correlation of pixel %d,%d' % self.selectedPixel)
			corrax.set_xlim(0,frame.shape[1])
			corrax.set_ylim(frame.shape[0],0)
			cbar = corrfig.colorbar(corrplot)
			plt.show()
		correlateButton.on_clicked(getCorrelation)
			
		timeseriesButton = Button(plt.axes([0.24,0.18,0.22,0.05]),'Intensity Timeseries',hovercolor='r')
		
		def getTimeseries(event):
			if self.selectedLine is not None and len(self.selectedLine) == 2:
				timeseries = self.getIntensityTimeseries(self.selectedLine,frameax,fig)
				fig2 = plt.figure()
				levels = np.linspace(np.min(timeseries),np.max(timeseries),100)
				plt.contourf(timeseries,levels=levels)
				plt.ylabel('Time index')
				plt.xlabel('Index along line')
				plt.title('Intensity time series')
				plt.show()
		timeseriesButton.on_clicked(getTimeseries)
				
		distributionButton = Button(plt.axes([0.01,0.18,0.22,0.05]),'Toroidal Distribution',hovercolor = 'r')
		
		def getToroidalDist(event):
			if not self.DISABLE_TRACER:
				intensity = self.toroidalDistribution()
				fig3 = plt.figure()
				phi = (self._currentphi + 360*np.arange(3600)/3599) % 360
				mean = running_mean_periodic(intensity,41)
				plt.plot(phi,intensity,'b.')
				plt.plot(phi[np.argsort(phi)],mean[np.argsort(phi)],'r')
				plt.ylabel('Summed Intensity (arb)')
				plt.xlabel('Toroidal Angle (degrees)')
				plt.show()
			else:
				print("WARNING: Cannot get toroidal intensity distribution, field line tracing disabled!")
		distributionButton.on_clicked(getToroidalDist)
		
		#Handle Mouse click events
		def onClick(event):
			if event.button == 1:
				if event.dblclick:
					if event.inaxes is frameax:
						self.selectedPixel = (int(event.xdata),int(event.ydata))
						if self.pixelplot:
							self.pixelplot.set_visible(False)
						self.pixelplot = frameax.scatter(self.selectedPixel[0],self.selectedPixel[1],s=70,c='r',marker='+')
						fig.canvas.draw()	
			elif event.button == 3:
				if event.dblclick:
					if event.inaxes is frameax:
						if self.selectedLine is None or len(self.selectedLine) == 2:
							#Load in first pixel coordinate and draw
							if self.linePlot is not None:
								self.linePlot[0].set_visible(False)
								self.linePlot[1].set_visible(False)
								self.linePlot[2][0].remove()
							self.selectedLine = [[int(event.xdata),int(event.ydata)]]
							self.linePlot = [frameax.scatter(int(event.xdata),int(event.ydata),s=70,c='y',marker='x')]
							fig.canvas.draw()
						elif len(self.selectedLine) == 1:
							self.selectedLine.append([int(event.xdata),int(event.ydata)])
							self.linePlot.append(frameax.scatter(int(event.xdata),int(event.ydata),s=70,c='y',marker='x'))
							self.linePlot.append(frameax.plot([self.selectedLine[0][0],self.selectedLine[1][0]],
												[self.selectedLine[0][1],self.selectedLine[1][1]],'-y',lw=2))
							fig.canvas.draw()
												
		fig.canvas.mpl_connect('button_press_event', onClick)	
		
		#Display UI
		plt.show()
	
	
	def enhanceFrame(self,frame):
		frame = np.uint8(frame*255.0/np.max(frame))			
		if self.applySub:
			frame = self.bgsub.apply(frame)
		if self.noiseRmv:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.bilateralFilter(frame,5,75,75)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except:
				pass
	
		if self.threshold:
			_,frame = cv2.threshold(frame,5,255,cv2.THRESH_BINARY)
		if self.gammaEnhance:
			frame = np.uint8(np.float64(frame)**(self.gamma))
		if self.histEq:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.equalizeHist(frame)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except: 
				pass
		if self.edgeDet:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.Canny(frame,500,550,True)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except:
				pass
		
		#frame = np.uint8(frame*255.0/np.max(frame))	
		return frame
		
	def projectFieldline(self,fieldline):
		objpoints = np.array([[fieldline.X[i],fieldline.Y[i],fieldline.Z[i]] for i in np.arange(len(fieldline.X))])
		return self.calibration.ProjectPoints(objpoints)[:,0,:]	
		
		
	def nextFrame(self):
		self._currentframeNum += 1
		if self._currentframeNum >= self.frames[...].shape[0]:
			self._currentframeNum = self.frames[...].shape[0] - 1
		self._currentframe = self.frames[self._currentframeNum]
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
		
	def previousFrame(self):
		self._currentframeNum -= 1
		if self._currentframeNum < 0:
			self._currentframeNum = 0
		self._currentframe = self.frames[self._currentframeNum]
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
		
	def sumIntensity(self,line,frame=None):
		total = 0.0
		N = 0.0
		if frame is None:
			frame = self._enhancedFrame
		try:
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		except:
			pass
				
		for i in np.arange(line.shape[0]):
			yind,xind = int(line[i,0]),int(line[i,1])
			if xind > 0 and xind < frame.shape[0] and yind > 0 and yind < frame.shape[1]:
				total += copy(frame[xind,yind])
				N += 1.0
			else:	
				pass
		return total#/N
  
	def getIntensityTimeseries(self,lineCoords,axes,fig):
		#Need to calculate the indices along the line
		x0,y0 = lineCoords[0][0],lineCoords[0][1]
		x1,y1 = lineCoords[1][0],lineCoords[1][1]

		gradient = (float(y1) - float(y0))/(float(x1) - float(x0))
		intercept = y0 - gradient*x0
		dims = self.frames[:].shape
		timeseries = np.zeros((dims[0],abs(x1-x0)))
		
		
		for t in np.arange(dims[0]):
			i = 0
			currentFrame = cv2.cvtColor(self.enhanceFrame(self.frames[t]),cv2.COLOR_BGR2GRAY)
			for x in np.linspace(x0,x1,abs(x1-x0)):
				y = gradient*x + intercept
				timeseries[t,i] = currentFrame[y,x]
				i += 1
		
		return timeseries
  
	def frameCorrelation(self,coords=(0,0),delay=0):
		dims = self.frames[:].shape
		frames = np.empty((dims[0]-abs(delay),dims[1],dims[2]))
		
		for i in np.arange(dims[0]-abs(delay)):
			frames[i] = cv2.cvtColor(self.enhanceFrame(self.frames[i]),cv2.COLOR_BGR2GRAY)
			
		#Get pixel means and standard deviations
		frames -= frames.mean(axis=0)
		frames /= frames.std(axis=0)
		
		result = np.zeros((frames.shape[1],frames.shape[2]))
		if delay > 0:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[delay:,coords[0],coords[1]]*frames[0:-delay,x,y])
		elif delay < 0:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[0:delay,coords[0],coords[1]]*frames[-delay:,x,y])
		else:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[:,coords[0],coords[1]]*frames[:,x,y])
					
		return result
		
	def toroidalDistribution(self,nphi=3600,nR=0,Rstart=None,Rend=None,dR=None,verbose=False,frame=None):
		dphi = 2.0*np.pi/nphi
		if nR == 0:
			intensity = []
			if Rstart is None:
				Rstart = self._currentR
			fieldline = self.tracer.trace(Rstart,self._currentZ,phistart = 2.0*np.pi*self._currentphi/360.0,mxstep=1000,ds=0.05)
			for i in np.arange(nphi):
				if verbose:
					print("phi = %.2f" % (self._currentphi + i*dphi))
				linepoints = self.projectFieldline(fieldline)
				intensity.append(self.sumIntensity(linepoints,frame=frame))
				fieldline.rotateToroidal(dphi)
		else:
			if Rstart is None:
				Rstart = self._currentR
			if dR is None:
				if Rend is None:
					Rend = Rstart + 0.1
				dR = (Rend - Rstart)/nR
			fieldlines = [self.tracer.trace(Rstart + j*dR,self._currentZ,phistart = 0.0,mxstep=1000,ds=0.05) for j in np.arange(nR)]
			intensity = np.zeros([nR,nphi])
			j = 0
			for fieldline in fieldlines:
				if verbose:
					print("R = %.2f" % (Rstart + j*dR))
				for i in np.arange(nphi):
					linepoints = self.projectFieldline(fieldline)
					if verbose:
						print("phi = %.2f" % (float(i)*dphi))
					intensity[j,i] = self.sumIntensity(linepoints,frame=frame)
					fieldline.rotateToroidal(dphi)
				j += 1
				

		return intensity
		
	def projectFluxTube(self,Rmid,Zmid,phimid,dR,dphi,n=20):

		phimid = phimid*2.0*np.pi/360.0
		dphi = dphi*2.0*np.pi/360.0

		#Minimum number of points allowed
		if n < 4: n = 4
		
		#Get the points around the circumference of the flux tube
		dalpha = 2*np.pi/n

		Rpoints = []
		phipoints = []
		fieldlines = [self.tracer.trace(Rmid + dR*np.sin(float(i)*dalpha),Zmid,phimid + dphi*np.cos(float(i)*dalpha),mxstep=1000,ds=0.05) for i in np.arange(n)]
		
		for i in np.arange(n):
			Rpoints.append(Rmid + dR*np.sin(float(i)*dalpha))
			phipoints.append((Rmid + dR*np.sin(float(i)*dalpha))*(dphi*np.cos(float(i)*dalpha)))			
			
		imagepoints = [self.projectFieldline(fieldline) for fieldline in fieldlines]
		return fieldlines,imagepoints
		


	def run_fluxtube_launcher(self):
		self.gammaEnhance = False
		self.applySub = False
		self.threshold = False
		self.histEq = False
		self.edgeDet = False
		self.noiseRmv = False
		self.gamma = 1.0
		
		axcolor = 'lightgoldenrodyellow'
		self.mask = copy(self._currentframe)
		self.mask[...]	= 1
		self.mask = np.uint8(self.mask)
		self.wireframeon = None

		axcolor = 'lightgoldenrodyellow'
		#Set up UI window
		fig = plt.figure(figsize=(8,8),facecolor='w',edgecolor='k')
		#Set up axes for displaying images
		frameax = plt.axes([0.2,0.25,0.6,0.6])
		frame = self.enhanceFrame(copy(self._currentframe))
		self.img = frameax.imshow(frame)
		frameax.set_axis_off()
		frameax.set_xlim(0,frame.shape[1])
		frameax.set_ylim(frame.shape[0],0)
		text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
		self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
		frameax.add_artist(self.frametxt)
		
		#Image enhancement selector
		enhancelabels = ('BG subtraction','Threshold','Gamma enhance','Detect edges','Equalise','Reduce Noise')
		enhanceCheck = CheckButtons(plt.axes([0.7,0.05,0.25,0.16]),enhancelabels,(False,False,False,False,False,False))
		gammaSlide = Slider(plt.axes([0.75,0.02,0.2,0.02],axisbg=axcolor), 'Gamma', 0.0, 3.0, valinit=1.0 )
		self._enhancedFrame = self._currentframe		
		def setEnhancement(label):
			if label == 'BG subtraction': self.applySub = not self.applySub
			elif label == 'Threshold' : self.threshold = not self.threshold
			elif label == 'Gamma enhance' : self.gammaEnhance = not self.gammaEnhance
			elif label == 'Detect edges'  : self.edgeDet = not self.edgeDet
			elif label == 'Equalise' : self.histEq = not self.histEq
			elif label == 'Reduce Noise' : self.noiseRmv = not self.noiseRmv
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)
		enhanceCheck.on_clicked(setEnhancement)
		
		def updateGamma(val):
			self.gamma = val
		gammaSlide.on_changed(updateGamma)
		
		#Field line launching area
		axR = plt.axes([0.22, 0.14, 0.35, 0.02], axisbg=axcolor)
		Rslide = Slider(axR, '$R_{0}$', 0.2, 2.0, valinit = 1.50,valfmt=u'%.3f')
		self._currentR = 1.50	
		def updateR(val):
			self._currentR = val
		Rslide.on_changed(updateR)
		
		ax_dR = plt.axes([0.22, 0.10, 0.35, 0.02], axisbg=axcolor)
		dRslide = Slider(ax_dR, '$dR$', 0.0, 0.2, valinit = 0.03, valfmt=u'%.4f' )
		self._currentdR = 0.03
		def update_dR(val):
			self._currentdR = val
		dRslide.on_changed(update_dR)
		
		axPhi = plt.axes([0.22,0.06,0.35,0.02],  axisbg=axcolor)
		PhiSlide = Slider(axPhi, '$\phi$', 0, 360, valinit = 180 )
		self._currentphi = 180.0
		
		def updatePhi(val):
			self._currentphi = val
			
		PhiSlide.on_changed(updatePhi)
		
		axdPhi = plt.axes([0.22,0.02,0.35,0.02],  axisbg=axcolor)
		dPhiSlide = Slider(axdPhi, '$d\phi$', 0, 20, valinit = 10 )
		self._currentdPhi = 5.0
		
		def update_dPhi(val):
			self._currentdPhi = val
			
		dPhiSlide.on_changed(update_dPhi)

		launchButton = Button(plt.axes([0.01,0.08,0.16,0.05]),'Launch',hovercolor='r')
			
		def launchfluxtube(event):
			if not self.DISABLE_TRACER:
				fieldlines,imagepoints = self.projectFluxTube(self._currentR,0.0,self._currentphi,self._currentdR,self._currentdPhi)
				for points in imagepoints:
					frameax.plot(points[:,0],points[:,1],'r',lw=0.5,alpha=0.7)
				from mpl_toolkits.mplot3d import Axes3D
				fig2 = plt.figure()
		
				ax2 = fig2.add_subplot(111,projection='3d',aspect='equal')
				for fieldline in fieldlines:
					ax2.plot(fieldline.X,fieldline.Y,fieldline.Z,'k',lw=0.5)

				ax2.set_zlim3d(-2.0,2.0)
				ax2.set_xlim3d(-2.0,2.0)
				ax2.set_ylim3d(-2.0,2.0)
				plt.show()
			else:
				print("WARNING: Cannot launch field line, tracer is disabled!")
		launchButton.on_clicked(launchfluxtube)
					
		clearButton = Button(plt.axes([0.01,0.02,0.16,0.05]),'Clear',hovercolor='r')
		
		def clearFieldlines(event):
			frameax.clear()
			#frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img = frameax.imshow(self.mask*self._enhancedFrame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.set_axis_off()
			frameax.set_xlim(0,frame.shape[1])
			frameax.set_ylim(frame.shape[0],0)
			
		clearButton.on_clicked(clearFieldlines)
		
		#Frame selection section
		
		nextButton =  Button(plt.axes([0.01,0.9,0.13,0.05]),'Next',hovercolor='r')
		def plotNext(event):
			self.nextFrame()
			frame = self.mask*self.enhanceFrame(self._currentframe)
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			fig.canvas.draw()
		nextButton.on_clicked(plotNext)
		
		prevButton =  Button(plt.axes([0.15,0.9,0.13,0.05]),'Previous',hovercolor='r')
		
		def plotPrev(event):
			self.previousFrame()
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			fig.canvas.draw()
		prevButton.on_clicked(plotPrev)
			
		refreshButton = Button(plt.axes([0.29,0.9,0.13,0.05]),'Refresh',hovercolor='r')
		
		def refreshPlot(event):
			self.mask = copy(self._currentframe)
			self.mask[...] = 1
			self.mask = np.uint8(self.mask)
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)	
		refreshButton.on_clicked(refreshPlot)

		saveButton = Button(plt.axes([0.43,0.9,0.13,0.05]),'Save',hovercolor='r')
		
		def saveFrame(event):
			"""
			Save both current plot in frame axis, and save image data as pngs
			"""
			savefile = raw_input("Safe file: ")
			extent = frameax.get_position().get_points()
			inches = fig.get_size_inches()
			extent *= inches
			bbox = frameax.get_position()
			bbox.set_points(extent)
			if savefile != '':
				fig.savefig(savefile,bbox_inches = bbox)
				savefileparts = savefile.split('.')
				plt.imsave(savefileparts[0]+"DATA.png",self.enhanceFrame(self._currentframe))
		saveButton.on_clicked(saveFrame)
		
		ROIButton = Button(plt.axes([0.72,0.84,0.22,0.05]),'Set ROI',hovercolor='r')
		
		self.selector = ROISelector(self.img)
		def setROI(event):
			self.ROI = np.asarray(self.selector.coords)
			self.mask[self.ROI[0,1]:self.ROI[1,1],self.ROI[0,0]:self.ROI[1,0]] = int(0)
			self.mask = int(1) - self.mask
			frame = self.mask*self.enhanceFrame(copy(self._currentframe))
			self.img.set_data(frame)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframe)
		ROIButton.on_clicked(setROI)
		
		calibButton = Button(plt.axes([0.72,0.9,0.22,0.05]),'Check Calibration',hovercolor='r')
		
		def checkCalibration(event):
			if not self.DISABLE_CCHECK:	
				if self.wireframeon == None:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					self.wireframeon = True
					return
			
			
				if self.wireframeon:
					#Already displaying so refresh screen
					frameax.clear()
					frame = self.mask*self.enhanceFrame(copy(self._currentframe))
					self.img = frameax.imshow(frame)
					text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
					self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
					frameax.set_axis_off()
					frameax.set_xlim(0,frame.shape[1])
					frameax.set_ylim(frame.shape[0],0)
					self.fieldlineArtists = []
					if self.projectLines:
						for line in self.projectLines:
							lplot, = frameax.plot(line[:,0],line[:,1],picker=5,lw=1.5)
							self.fieldlineArtists.append(lplot)
						self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
						xycoords='axes fraction',color='white',fontsize=8)
				else:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					
				self.wireframeon = not self.wireframeon
			else:
				print("WARNING: Calibration checking disabled!")
				
		calibButton.on_clicked(checkCalibration)
		
		plt.show()

if __name__=='__main__':
	import sys
        try:
		shot = sys.argv[1]
	except:
		shot = None
		pass
		
	#Elzar = Elzar(Nframes=50,startpos=0.4,shot=shot)
	moviefile = '/net/edge1/scratch/jrh/SA1/rbf029840.ipx'
	Elzar = Elzar(Nframes=200,startpos=0.4,moviefile=moviefile,time=0.248,gfile='Dummy',calibfile='29840')
	Elzar.runUI()
	
	
	Elzar.run_fluxtube_launcher()
	#intensity = Elzar.toroidalDistribution(Rstart = 1.47,nR=20,Rend=1.60,nphi=1000,verbose=True,frame=Elzar.mask*Elzar.enhanceFrame(Elzar.frames[10]))
	#levels = np.linspace(np.min(intensity),np.max(intensity),100)
	#plt.contourf(np.linspace(0.0,360.0,1000),np.linspace(1.47,1.60,20),intensity,levels=levels)
	#plt.colorbar()	
	##plt.xlabel('Toroidal Angle (deg)')
	#plt.ylabel('Major radius (m)')
	#plt.show()


	
