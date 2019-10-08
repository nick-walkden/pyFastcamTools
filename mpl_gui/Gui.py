"""
Class to manually track filaments in a given movie
"""

from readMovie import readMovie
#from fieldlineTracer import get_fieldline_tracer
from backgroundSubtractor import *
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
from scipy.stats import moment,describe
from scipy.signal import argrelmax,argrelmin


class ROISelector(object):
    
    def __init__(self,artist):
            self.artist = artist
            self.selector = RectangleSelector(self.artist.axes,self.on_select,
                                       button=3, minspanx=5, minspany=5, spancoords='pixels',
                                       rectprops = dict(facecolor='red', edgecolor = 'red',
                                                        alpha=0.3, fill=True)) # drawtype='box'
            self.coords = []
            
    def on_select(self,click,release):
            x1,y1 = int(click.xdata),int(click.ydata)
            x2,y2 = int(release.xdata),int(release.ydata)
            self.coords =[(x1,y1),(x2,y2)]
            
    def activate(self):
        self.selector.set_active(True)
        
    def deactivate(self):
        self.selector.set_active(False)        
    


class Elzar(object):
	
	def __init__(self,Nframes=None,startpos=0.0,moviefile=None,transforms=[]):
		
		
		if moviefile is None:
			moviefile = raw_input("Movie file or shot number: ")	
		print("Reading Movie file, please be patient")				
		self.frames = readMovie(moviefile,Nframes=Nframes,startpos=startpos,transforms=transforms)
		print("Finished reading movie file")

								
		self._currentframeNum = 0
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)	
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
				
		self.bgsub = backgroundSubtractorMin(20)
		#Initialize background model
		for frame in self.frames[0:19]:
			dummy = self.bgsub.apply(frame)
		
		self.widgets = {}
		self.dataCursor = None
	
		try:
			#self.CADMod = MachineGeometry.MAST('high')
			#self.wireframe = Render.MakeRender(self.CADMod,self.calibration,Verbose=False,Edges=True,EdgeWidth=1,EdgeMethod='simple')
			self.DISABLE_CCHECK = True
		except:
			print("WARNING: No CAD model found for MAST. Disabling calibration checking.")
			self.DISABLE_CCHECK = True
		

                self.gammaEnhance = False
                self.applySub = False
                self.threshold = False
                self.histEq = False
                self.edgeDet = False
                self.noiseRmv = False
                self.negative = False
                self.gamma = 1.0


	def runUI(self):
		
		#Initialize some parameters for the UI
		self.gammaEnhance = False
		self.applySub = False
		self.threshold = False
		self.histEq = False
		self.edgeDet = False
		self.noiseRmv = False
		self.negative = False
		self.gamma = 1.0
		self.selectedPixel = None
		self.pixelplot = None
		self.selectedLine = None
		self.linePlot = None
		axcolor = 'lightgoldenrodyellow'
		self.mask = copy(self._currentframeDisplay)
		self.mask[...]	= 1
		self.mask = np.uint8(self.mask)

		
		#Set up UI window
		fig = plt.figure(figsize=(8,8),facecolor='w',edgecolor='k')
		#Set up axes for displaying images
		frameax = plt.axes([0.0,0.25,0.9,0.6])
		frame = self.enhanceFrame(copy(self._currentframeData))
		self.img = frameax.imshow(frame)
		frameax.set_axis_off()
		frameax.set_xlim(0,frame.shape[1])
		frameax.set_ylim(frame.shape[0],0)
		text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [s]'
		self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
		frameax.add_artist(self.frametxt)
		
		#Set up axis for equilibrium plot
		
		
		#Image enhancement selector
		enhancelabels = ('BG subtraction','Threshold','Gamma enhance','Detect edges','Equalise','Reduce Noise')
		enhanceCheck = CheckButtons(plt.axes([0.7,0.05,0.25,0.16]),enhancelabels,(False,False,False,False,False,False))
		gammaSlide = Slider(plt.axes([0.75,0.02,0.2,0.02],facecolor=axcolor), 'Gamma', 0.0, 3.0, valinit=1.0 )
		self._enhancedFrame = self._currentframeDisplay		
		def setEnhancement(label):
			if label == 'BG subtraction': self.applySub = not self.applySub
			elif label == 'Threshold' : self.threshold = not self.threshold
			elif label == 'Gamma enhance' : self.gammaEnhance = not self.gammaEnhance
			elif label == 'Detect edges'  : self.edgeDet = not self.edgeDet
			elif label == 'Equalise' : self.histEq = not self.histEq
			elif label == 'Reduce Noise' : self.noiseRmv = not self.noiseRmv
			elif label == 'Negative' : self.negative = not self.negative
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)
		enhanceCheck.on_clicked(setEnhancement)
		
		def updateGamma(val):
			self.gamma = val
		gammaSlide.on_changed(updateGamma)
		
		#Frame selection section
		
		nextButton =  Button(plt.axes([0.01,0.9,0.13,0.05]),'Next',hovercolor='r')
		
		def plotNext(event):
			self.nextFrame()
			frame = self.mask*self.enhanceFrame(self._currentframeData)
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [s]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			fig.canvas.draw()
		nextButton.on_clicked(plotNext)
		
		prevButton =  Button(plt.axes([0.15,0.9,0.13,0.05]),'Previous',hovercolor='r')
		
		def plotPrev(event):
			self.previousFrame()
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [s]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			fig.canvas.draw()
		prevButton.on_clicked(plotPrev)
			
		refreshButton = Button(plt.axes([0.29,0.9,0.13,0.05]),'Refresh',hovercolor='r')
		
		def refreshPlot(event):
			self.mask = copy(self._currentframeDisplay)
			self.mask[...] = 1
			self.mask = np.uint8(self.mask)
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)	
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
		
		ROIButton = Button(plt.axes([0.72,0.9,0.13,0.05]),'Set ROI',hovercolor='r')
		
		self.selector = ROISelector(self.img)
		def setROI(event):
			self.ROI = np.asarray(self.selector.coords)
			self.mask[self.ROI[0,1]:self.ROI[1,1],self.ROI[0,0]:self.ROI[1,0]] = int(0)
			self.mask = int(1) - self.mask
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)
		ROIButton.on_clicked(setROI)
		
		
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
				timeseries = self.getIntensityTimeseries(self.selectedLine)
				fig2 = plt.figure()
				#plt.subplot(121)
				levels = np.linspace(np.min(timeseries),np.max(timeseries),100)
				plt.contourf(timeseries,levels=levels)
				plt.ylabel('Time index')
				plt.xlabel('Index along line')
				plt.title('Intensity time series')
				#delta = self.get_approximate_rad_vel(self.selectedLine[0],self.selectedLine[1],intensity=timeseries,boxcar=10,threshold=2.5,plot=False)
				#plt.subplot(122)
				#plt.bar([0,1,2,3,4,5,6,7],np.histogram(delta,bins=[0,1,2,3,4,5,6,7,8])[0],edgecolor='red',width=1.0,align='center')
				#plt.xlim(0,8)
				#plt.title('Widths along line')
				#plt.xlabel('Line indices')
				#plt.ylabel('counts')
				plt.show()
				
		timeseriesButton.on_clicked(getTimeseries)
				
		
		
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
		#frame = np.uint8(frame*255.0/np.max(frame))			
		if self.applySub:
			frame = self.bgsub.apply(frame)
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)#np.uint8(frame*255.0/np.max(frame)) 
		frame = np.uint8(frame*255.0/np.max(frame))
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
			_,frame = cv2.threshold(frame,10,255,cv2.THRESH_BINARY)
		if self.gammaEnhance:
			gammaframe = np.float64(frame)**(self.gamma)
			frame = np.uint8(gammaframe*255.0/np.max(gammaframe))
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

		if self.negative:
			frame = 255 - frame		

		#frame = np.uint8(frame*255.0/np.max(frame))	
		return frame
		


	
	def nextFrame(self):
		self._currentframeNum += 1
		if self._currentframeNum >= self.frames[...].shape[0]:
			self._currentframeNum = self.frames[...].shape[0] - 1
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
		
	def previousFrame(self):
		self._currentframeNum -= 1
		if self._currentframeNum < 0:
			self._currentframeNum = 0
		#self._currentframe = self.frames[self._currentframeNum]
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]

  
	def getIntensityTimeseries(self,lineCoords):
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
			#print currentFrame.shape
			for x in np.linspace(x0,x1,abs(x1-x0)):
				y = gradient*x + intercept
				timeseries[int(t),i] = currentFrame[int(y),int(x)]
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
		

	def get_fft(self,subtract=False):
		self.fftfig = plt.figure(figsize=(8,8),facecolor='w')
		fftax1 = plt.axes([0.1,0.2,0.25,0.6])
		fftax2 = plt.axes([0.4,0.2,0.25,0.6])
		fftax3 = plt.axes([0.7,0.2,0.25,0.6])
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			if subtract:   frames[i] = cv2.cvtColor(cv2.cvtColor(self.bgsub.apply(self.frames[i]),cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
			else:	frames[i] = cv2.cvtColor(cv2.cvtColor(self.frames[i],cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
		fft = np.fft.rfft(frames,axis=0) 
		rate = 1.0/(self.frames.timestamps[1] - self.frames.timestamps[0])
		self.__power = np.abs(fft[1:])#**2.0
		self.__freqs = np.linspace(0,rate/2,self.__power.shape[0] + 1)[1:]
		#self.__power /= np.sum(self.__power,axis=0)
		self.fftimg1 = fftax1.imshow(self.__power[0],cmap=cm.coolwarm)#'afmhot')
		self.fftimg2 = fftax2.imshow((self.__power/np.sum(self.__power,axis=0))[0],cmap=cm.coolwarm)#'afmhot')
		self.fftimg3 = fftax3.imshow(((self.__power/np.sum(self.__power,axis=0))[0])/np.max((self.__power/np.sum(self.__power,axis=0))[0]),cmap=cm.coolwarm)#'afmhot')
		fftax1.set_axis_off()
		fftax1.set_title('Power')
		fftax2.set_axis_off()
		fftax2.set_title('Normalized')
		fftax3.set_axis_off()
		fftax3.set_title('Relative')
		freq_slider = Slider(plt.axes([0.3,0.08,0.5,0.03]),'Frequency (kHz)',self.__freqs[0]/1000.0,self.__freqs[-1]/1000.0,valinit=self.__freqs[0]/1000.0,valfmt=u'%.1f')

		def on_change(val):
			ind = np.abs(self.__freqs - val*1000.0).argmin()
			self.fftimg1.set_data(self.__power[ind])
			self.fftimg2.set_data((self.__power/np.sum(self.__power,axis=0))[ind])
			self.fftimg3.set_data(((self.__power/np.sum(self.__power,axis=0))[ind])/np.max((self.__power/np.sum(self.__power,axis=0))[ind]))
			self.fftfig.canvas.draw()

		freq_slider.on_changed(on_change)	
		
		plt.show()

	def get_moments(self,subtract=False):
		#momntfig = plt.figure(figsize=(8,8),facecolor='w')
		momntax1 = plt.axes([0.0,0.2,0.25,0.6])
		momntax1.set_axis_off()
		momntax1.set_title('Mean')
		momntax2 = plt.axes([0.25,0.2,0.25,0.6])
		momntax2.set_axis_off()
		momntax2.set_title('Variance')
		momntax3 = plt.axes([0.5,0.2,0.25,0.6])
		momntax3.set_axis_off()
		momntax3.set_title('Skewness')
		momntax4 = plt.axes([0.75,0.2,0.25,0.6])
		momntax4.set_axis_off()
		momntax4.set_title('Kurtosis')
	
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			
			if subtract: frames[i,0:,0:] = cv2.cvtColor(cv2.cvtColor(self.bgsub.apply(self.frames[i]),cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
			else: frames[i,0:,0:] = cv2.cvtColor(cv2.cvtColor(self.frames[i],cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
		#frames = self.frames[:]		
		frames /= np.max(frames)		
	
		#self.__mean = moment(frames,moment=0,axis=0)
		#self.__var  = moment(frames,moment=2,axis=0)
		#self.__skew = moment(frames,moment=3,axis=0)
		#self.__kurt = moment(frames,moment=4,axis=0)
		nobs,minmax,mean,var,skew,kurt = describe(frames,axis=0)

		im1 = momntax1.imshow(mean,cmap='afmhot')
		im2 = momntax2.imshow(var,cmap='afmhot')
		im3 = momntax3.imshow(skew,cmap='bwr',vmin=-8.0,vmax=8.0)
		im4 = momntax4.imshow(kurt,cmap='afmhot',vmax=64.0)
		cax1 = plt.axes([0.01,0.1,0.23,0.04])
		cax2 = plt.axes([0.26,0.1,0.23,0.04])
		cax3 = plt.axes([0.51,0.1,0.23,0.04])
		cax4 = plt.axes([0.76,0.1,0.23,0.04])
		plt.colorbar(im1,cax=cax1,orientation='horizontal')
		plt.colorbar(im2,cax=cax2,orientation='horizontal')
		plt.colorbar(im3,cax=cax3,orientation='horizontal')
		plt.colorbar(im4,cax=cax4,orientation='horizontal')
		plt.show()

		return mean,var,skew,kurt

	def get_svd(self,nmom=None,bgsub=True):

		if bgsub: frame = np.float64(self.bgsub.apply(self._currentframeData))
		else: frame = np.float64(self._currentframeData)
		S,u,v = cv2.SVDecomp(frame)

		size = frame.shape
		print S.shape
		print u.shape
		print v.shape
		
		nx = size[0]
		ny = size[1]
		reconst = np.zeros(size)
		if nmom is None:
			nmom = nx
		print reconst.shape
		for i in np.arange(nx):
			for j in np.arange(nmom-1):
				reconst[i,:] = reconst[i,:] +  u[i,j]*S[j,0]*v[j,:]

		print reconst.shape

		plt.imshow(np.concatenate([frame,reconst],axis=1),cmap='gray')
		plt.title(str(nmom)+" Moments")
		plt.show()
			

	def get_autocorrelation(self,subtract=False,nt=100):
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			if subtract: frames[i] = self.bgsub.apply(self.frames[i])
			else: frames[i] = self.frames[i]
		frames -= frames.mean(axis=0)
		frames /= frames.std(axis=0)
		corr = np.zeros((nt,frames.shape[1],frames.shape[2]))
		for delay in np.arange(nt):
			corr[delay] = (frames*np.roll(frames,-delay,axis=0)).mean(axis=0)
		auto_corr = np.zeros((frames.shape[1],frames.shape[2]))
		#auto_corr = 1.0 + 2.0*np.sum(corr,axis=0)
		#auto_corr *= (self.frames.timestamps[1]-self.frames.timestamps[0])
		for x in np.arange(frames.shape[1]):
			for y in np.arange(frames.shape[2]):
				for i in np.arange(nt):
					if corr[i,x,y] < 1.0/2.718:
						auto_corr[x,y] = i
						break
		auto_corr *= (self.frames.timestamps[1]-self.frames.timestamps[0])
		return auto_corr


	
	