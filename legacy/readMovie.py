#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
from frameHistory import frameHistory
#try:
from pyIpx.movieReader import ipxReader,mrawReader,imstackReader
from pyFastcamTools.Frames import ExperimentalFrame
import logging
# from pyIpx import ipxReader,mrawReader,imstackReader
#except:
#	print("WARNING: No movieReader module found. Cannot read .ipx or .mraw files.")

logger = logging.getLogger(__name__)

class movieFrame(np.ndarray):
	"""
	numpy ndarray with additional attributes for frame number and timestamp
	"""

	def __new__(cls,frame,timestamp=None, frameNumber=None):

		obj = np.asarray(frame).view(cls)

		obj.timestamp = timestamp
		obj.frameNumber = frameNumber

		return obj

	def __array_finalize__(self,obj):

		if obj is None: return

		self.timestamp = getattr(obj,'timestamp',None)
		self.frameNumber = getattr(obj,'frameNumber',None)

def readMovie(filename,Nframes=None,stride=1,startpos=None,endpos=1.0,verbose=True,startframe=0,endframe=-1,
              transforms=[], history=frameHistory, frame_type=ExperimentalFrame, **kwargs):

    """
    Function to read in a movie file using openCV and store as a frameHistory

    Arguments:
        filename	-> 	name of the movie file to read from
                    OR
                    MAST shot number to read

    keywords:
        Nframes		-> 	Number of frames to read	Default: None, read entire movie
        stride		->	Read frames with a stride	Default: 1, read every frame
        startpos	-> 	Relative position to start	Default: 0, start from beginning of movie
                    reading from, 0 = start,
                    1 = end, 0.xx is xx% through
                    the movie
        endpos		->	Relative position to end 	Default: 1, read until end of movie
                    reading
        transforms	->	Apply any or all of the following transforms to the image data
                        'reverse_x' : reverse the x dimension
                        'reverse_y' : reverse the y dimension
                        'transpose' : transpose the image (flip x and y)

    Example:

        frames = readMove('myMovie.mov',Nframes=100,stride=2,startpos=0.3,endpos=0.5)

        This will read every 2nd frame of a movie from the file 'myMovie.mov' starting from 30% into the movie and
        ending after 100 frames, or when it reaches 50% through, whichever comes first

    """
    if issubclass(history.__class__, frameHistory):  # Existing instance of frameHistory supplied so use that
        frames = history
    else:  # Create a new frameHistory instance of the supplied type
        frames = history(Nframes, info='readMovie')
    if '.' not in filename or filename.split('.')[-1] == 'ipx' or filename.split('.')[-1] == 'mraw':
        if filename.split('.')[-1] == 'ipx':
            #Assume it is a shot number
            if '.' not in filename:
                vid = ipxReader(shot=filename)
            else:
                vid = ipxReader(filename=filename)
            if startpos is not None and startframe is None:
                startframe = int(startpos*vid.file_header['numFrames'])
            if endpos is not None and endframe is None:
                endframe = int(endpos*vid.file_header['numFrames'])
            elif endframe is -1:
                endframe = int(1.0*vid.file_header['numFrames'])
        else:
            if filename.split('.')[-1] == 'mraw':
                vid = mrawReader(filename=filename)
            else:
                vid = imstackReader(directory=filename)
            if startpos is not None and startframe is None:
                startframe = int(startpos*int(vid.file_header['TotalFrame']))
            if endpos is not None and endframe is None:
                endframe = int(endpos*int(vid.file_header['TotalFrame']))
            elif endpos is -1:
                endframe = int(1.0*int(vid.file_header['TotalFrame']))

        if Nframes is None:
            Nframes = endframe - startframe + 1
        frames.N = Nframes
        #print(vid.file_list)
        vid.set_frame_number(startframe)
       
        N = 0
        for i in np.arange(Nframes*stride):
            ret,frame,header = vid.read(transforms=transforms)
            # frame = ExperimentalFrame(frame=frame, n=startframe + i, t=header['time_stamp'])
            if ret and not N + startframe > endframe:
                if i % stride == 0:
                    if verbose:
                        print("Reading movie frame {} at time {}".format(startframe + i,header['time_stamp']))
                    # frames.add(frame,header['time_stamp'],startframe + i)
                    
                    frames.add_array(frame,frame_type=frame_type, t=header['time_stamp'],n=startframe+i, **kwargs)
                    logger.debug('Added frame {} to {}'.format(startframe + i, str(frames)))
                    N += 1
            else:
                break
        pass
    else:
        #Reading a non-ipx file with openCV

        vid = cv2.VideoCapture(filename)
        #Set the starting point of the video
        vid.set(2,startpos)

        if Nframes==None:
        #Set number of frames in history to number of frames in movie (accounting for stride)
            Nframes=int(vid.get(7)/stride)



        times = []
        frameNos = []

        for i in np.arange(Nframes*stride):
            ret,frame = vid.read()
            if ret and not vid.get(2)>endpos:
                #Succesful read
                if i%stride==0:
                    #Only take every stride frames
                    frames.add(frame,vid.get(0),vid.get(1))
                    if verbose:
                        print("Reading movie frame "+str(i), end='\r')
            else:
                break
    vid.release()
    return frames


if __name__=='__main__':

	import matplotlib.pyplot as plt

	# frames = readMovie('/home/nwalkden/python_tools/pySynthCam/29852_movie.mpg',Nframes=100)
	# frames = readMovie('Test_mraw.mraw',Nframes=500,startpos=0.2,endpos=0.7)
	frames = readMovie('/net/edge1/scratch/jrh/SA1/rbf029852.ipx',Nframes=3,startpos=0.2,endpos=0.7)
	#frames.setROI()

	from share_memory import share_memory

	print(share_memory(frames.frames, frames[0].frame))

	frames[-1].plot()
	frames[0][:] = -1
	plt.show()

