#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
FrameHistory object for working with sets of Frames
"""

import os
import sys
import numpy as np
import inspect
import cv2
from pyFastcamTools.create_log import create_log
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import logging

# from pyIpx.movieReader import movieReader
from pyFastcamTools.Frames import Frame, SyntheticFrame, ExperimentalFrame

try:
    import pandas as pd
except ImportError:
    sys.path.append(os.path.expanduser('/home/tfarley/.local/lib/python2.7/site-packages/'))
    import pandas as pd
    print('Using tfarley .local version of pandas')

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def debug_trace_QT():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook

    # Or for Qt5
    # from PyQt5.QtCore import pyqtRemoveInputHook

    from pdb import set_trace

    pyqtRemoveInputHook()
    set_trace()


class frameHistory(object):
    """
    Simple class to store a movie as a history of frames

    Frames within a frame history can be accessed by indexing, ie

    frames = frameHistory()

    frame = frames[1,:,:,:]
    """
    frame_types_all = {'Frame': Frame, 'ExperimentalFrame': ExperimentalFrame, 'SyntheticFrame': SyntheticFrame}
    frame_types = {'Frame': Frame, 'Experimental': ExperimentalFrame, 'Synthetic': SyntheticFrame}
    frame_type = frame_types_all['Frame']

    def __init__(self, N=100, info=None, loop_frames=False, frame_type=None, verbose=True):
        self.N = N if N is not None else 100 # Store maximum of 100 frames in history by default
        self.info = info  # String describing the frameHistory instance
        self.verbose = verbose
        self.loop_frames = loop_frames

        self.frames = None  # 3D np.ndarray containing all frames in history
        self.frames_meta = None  # pandas DataFrame containing metta data about frames

        self.frame_objs = []  # list of Frame objects
        self.timestamps = []  # view of times column in self.frames_meta
        self.frameNumbers = []  # view of frame numbers column in self.frames_meta
        self.shape = [self.N, None, None]  # shape of self.frames
        self.enhancements = []  # ordered list of enhancements applied to frame
        self._enhanced = []  # Record of currently applied enhancements
        self.enhance_str = ''  # string detailing enhancements for file output
        self.bgsub = None  # background subtractor object
        self.bgsub_settings = None  # settings defining which frames are in bgsub._history
        self.next_frame = 0  # Index of next frame to be added (self.next_frame == self.N when fully loaded)
        self.ROI = None  # NOT IMPLEMENTED

        if frame_type is not None:
            self.set_frame_type(frame_type)
        logger.debug('Initialised frame history object: {}'.format(str(self)))

    def clear(self):

        self.frames = None  # 3D np.ndarray containing all frames in history
        self.frames_meta = None  # pandas DataFrame containing metta data about frames

        self.N = None  # Number of frames in history
        self.frame_objs = []  # list of Frame objects
        self.timestamps = []  # view of times column in self.frames_meta
        self.frameNumbers = []  # view of frame numbers column in self.frames_meta
        self.shape = [self.N, None, None]  # shape of self.frames
        self.enhancements = []  # ordered list of enhancements applied to frame
        self._enhanced = []  # Record of currently applied enhancements
        self.enhance_str = ''  # string detailing enhancements for file output
        self.bgsub = None  # background subtractor object
        self.bgsub_settings = None  # settings defining which frames are in bgsub._history
        self.next_frame = 0  # Index of next frame to be added (self.next_frame == self.N when fully loaded)
        self.ROI = None  # NOT IMPLEMENTED

    def allocate_frame_memory(self, example_frame, N=None):
        """ Allocate memory to be filled by frames, given format of example frame
        """
        assert (isinstance(example_frame, (np.ndarray, Frame))), 'example_frame type: {}'.format(type(example_frame))

        if N is not None:
            self.N = N

        if self.N is None:
            self.N = 10
            print('Warning: No frameHistory length supplied. Defaulting to {}'.format(self.N))

        self.shape = [self.N, example_frame[:].shape[0], example_frame[:].shape[1]]
        logger.info('Allocating memory for frame history: {}'.format(str(self)))
        self.frames = np.zeros(self.shape, dtype=example_frame[:].dtype)

        self.frames_meta = pd.DataFrame(-1, index=np.arange(self.N), columns=['n', 't'])
        self.frames_meta['n'] = self.frames_meta['n'].astype(np.int32)
        self.frames_meta.set_index('n')

        logger.info('Allocated memory for frame history: {}'.format(str(self)))

        self.timestamps = self.frames_meta.loc[:, 't'].values  # views of meta data
        self.frameNumbers = self.frames_meta.loc[:, 'n'].values

        if isinstance(self.frames.dtype, float):
            self.frames.fill(np.nan)
        elif isinstance(self.frames.dtype, (int, long)):  # Need to fix!
            self.frames.fill(-1)

        self.next_frame = 0

    def reallocate_frame_memory(self, extra=0.10):
        """ Append empty (nan) rows to self.fil dataframe so further filaments can be added
        """
        assert self.frames is not None  # Memory must already be allocated

        # Allow memory for eg 10% more frames that previously expected
        self.N = int(np.ceil((1.0+extra) * self.next_frame))
        self.shape = [self.N, self.shape[1], self.shape[2]]

        if self.N > len(self.frames_meta):  # append space to end of dataframe
            n_add = self.N-len(self.frames_meta)
            add_frames = np.zeros((n_add, self.shape[1], self.shape[2]), dtype=self.frames[:].dtype)
            add = pd.DataFrame(np.nan, index=np.arange(n_add), columns=self.frames_meta.columns)
            self.frames = np.concatenate((self.frames, add_frames))
            self.frames_meta = self.frames_meta.append(add, ignore_index=True)  # append empty rows to dataframe
            print('Reallocated memory for self.frames, N: {} -> {})'.format(
                self.N-len(add), self.N))
        elif self.N == self.next_frame:
            pass  # Don't need to change memory
        else:
            self.N = self.next_frame  # avoid rounding up with ceil as know exact number of frames
            print('Trimming empty frame history memory: {} -> {}'.format(len(self.frames), self.N))
            self.frames_meta = self.frames_meta.iloc[0:self.N]  # Remove excess rows
            self.frames = self.frames[0:self.N]  # Remove excess blank frames

    def set_frame_type(self, frame_type):
        """ Set type of Frame object:
        Frame, ExperimentalFrame, SyntheticFrame, ExperimentalElzarFrame, SyntheticElzarFrame
        """
        if issubclass(frame_type, Frame):
            self.frame_type = frame_type
        elif isinstance(frame_type, basestring):
            assert frame_type in self.frame_types_all.keys()
            self.frame_type = self.frame_types_all[frame_type]

    def set(self, frames, frame_type=Frame):
        """
        Set the frame history to an existing frame history instance or build from a numpy array of frames
        """

        if type(frames) == type(self):
            for key, value in frames.__dict__.iteritems():
                self.__dict__[key] = value
        else:
            if isinstance(frames, np.ndarray):
                # Passed an array containing stack of frames
                self.N = frames.shape[0]
                self.frames = copy(frames)
                self.frame_objs = []
                for frame in self.frames:
                    self.frame_objs.append(frame_type(frame=frame, frame_history=self))

    def add_frame(self, frame, n=None, no_repeat=True, loop_frames=None):
        """ Add existing Frame object
        """
        assert (issubclass(frame.__class__, Frame)), (type(frame), Frame)
        logger.debug('Adding frame {} to history {}, id: {}'.format(repr(frame), repr(self), id(self)))
        # print('Adding frame {} to history {}, id: {}'.format(repr(frame), repr(self), id(self)))

        if loop_frames is not None:
            self.loop_frames = loop_frames
        # If reached max number of frames, loop to begining and replace early frames
        if n is None and self.loop_frames:
            if self.next_frame == self.N:
                import pdb; pdb.set_trace()
                logger.debug('Looping to begining of {}'.format(str(self)))
                print('Looping to begining of {}'.format(str(self)))
            self.next_frame = self.next_frame % self.N

        i = self.next_frame if n is None else n
        frame.i = i

        if self.frames is None:
            self.allocate_frame_memory(frame.raw)
        if i >= self.N:
            if self.loop_frames:
                i = 0
                self.next_frame = i
            else:
                self.reallocate_frame_memory(extra=0.10)
            # raise RuntimeError("Can't add frame to frameHistory - outside range of allocated memory i={} > N={}".format(
            #     i, self.N))

        # Avoid duplicate frames
        if no_repeat and frame in self.frame_objs:
            logger.info(
                '********************************** Frame already in history **********************************')
            # import pdb; pdb.set_trace()
            return

        # Add frame to frame history
        frame.frame_history = self

        self.frames[i] = frame.raw  # Raw unenhanced data, does not affect enhancements if alredy applied
        frame.raw = self.frames[i]  # Make sure raw is a view of the allocated memory - save space
        self.frame_objs.append(frame)
        self.frames_meta.loc[i, 'i'] = copy(frame.i)
        self.frames_meta.loc[i, 'n'] = copy(frame.n)
        self.frames_meta.loc[i, 't'] = copy(frame.t)

        # Not sure why views don't remain from during memory assignement...
        self.timestamps = self.frames_meta.loc[:, 't'].values  # views of meta data
        self.frameNumbers = self.frames_meta.loc[:, 'n'].values

        # print('Incrementing self.next_frame')
        self.next_frame += 1

    def add_array(self, frame, frame_type=None, no_repeat=True, **kwargs):
        """ Add numpy array (convert to Frame object)
        """
        # import pdb; pdb.set_trace()
        logger.debug('Entering add_array method of {}'.format(str(self)))
        if frame_type is None:
            frame_type = self.frame_type
        else:
            self.frame_type = frame_type
        kws = {k: v for k, v in kwargs.iteritems() if
               k in inspect.getargspec(frame_type.__init__)[0] or k in inspect.getargspec(Frame.__init__)[0]}

        frame = frame_type(frame=frame, frame_history=self, no_repeat=no_repeat, i=self.next_frame,
                           **kws)  # Supplying frame history automatically handles adding frame object to history in Frame.__init__
        # self.add_frame(frame, no_repeat=no_repeat)
        logger.debug('Leaving add_array method of {}'.format(str(self)))

    # def append(self, frame):  # Needs updating!
    #     """
    #     Used to add a frame and increasing framecount in the history
    #     """
    #
    #     if self.frames is None:
    #         self.frames = frame[np.newaxis, ...]
    #         self.frame_objs.append(frame)
    #         self._updates += 1
    #     else:
    #         self.frames = np.append(self.frames, frame[np.newaxis, ...], axis=0)
    #         self.frame_objs.append(frame)
    #         self._updates += 1
    #         self.N += 1

    def read_movie(self, filename, verbose=True, **kwargs):
        """ Read a movie file
        """
        from pyFastcamTools.readMovie import readMovie
        # Call parent Frame object's __init__
        self.next_frame = 0
        _func = readMovie
        kws = {k: v for k, v in kwargs.iteritems() if k in inspect.getargspec(_func)[0]}
        # Change default values for object types

        kws.setdefault('history', self)
        kws.setdefault('frame_type', self.frame_types['Experimental'])
        frame_history = _func(filename, **kws)
        assert frame_history is self
        # for key, value in frame_history.__dict__.iteritems():
        #     self.__dict__[key] = value
        if verbose:
            print('Read frames {:d} - {:d}'.format(np.amin(self.frameNumbers), np.amax(self.frameNumbers)))
            # print('self.frameNumbers', self.frameNumbers)
            # print('self.frameNumbers', )
    
    def get_frame_list(self, n_current, Nbackwards=10, Nforwards=0, stepBackwards=1, stepForwards=1, skipBackwards=0,
                        skipForwards=0, unique=True, verbose=True):
        """ Return list of frame numbers (frame marks) given input
        """
        frame_list_settings = {'Nbackwards': Nbackwards, 'Nforwards': Nforwards, 'skipBackwards': skipBackwards,
                               'skipForwards': skipForwards,
                               'stepBackwards': stepBackwards, 'stepForwards': stepForwards}
        # import pdb; pdb.set_trace()

        ## Get list of frames equal to length of frame history that bracket the current frame and do not go outside
        ##  the range of frame numbers
        frameNumStart = n_current - frame_list_settings['skipBackwards'] - frame_list_settings['stepBackwards'] * (
        frame_list_settings['Nbackwards'] - 1) - 1
        frameNumEnd = n_current + frame_list_settings['skipForwards'] + frame_list_settings['stepForwards'] * (
        frame_list_settings['Nforwards'] - 1) + 1

        frame_nos = (np.linspace(frameNumStart,
                                n_current - frame_list_settings['skipBackwards'] - 1,
                                num=frame_list_settings['Nbackwards']),
                    # np.array([frameNum0]),
                    np.linspace(n_current + frame_list_settings['skipForwards'] + 1,
                                frameNumEnd,
                                num=frame_list_settings['Nforwards']))
        frame_nos = np.round(np.hstack(frame_nos)).astype(int)
        logger.debug('Frames in frame_list:  {}'.format(str(frame_nos)))

        # Make sure frames are in frame range
        frame_nos = frame_nos.clip(np.min(self.frameNumbers), np.max(self.frameNumbers))
        # frameMarks = frameNos + self.frameNumbers[0]
        if unique:  # remove duplicates
            frame_nos = list(set(frame_nos))
        return frame_nos

    def set_enhancements(self, enhancements, _apply=True):
        """ Set enhancements to be applied to frames: BG subtraction etc
        """
        self.enhancements = enhancements
        logger.info('Enhancements in {} set to: {}'.format(repre(self), self.enhancements))
        if _apply:
            logger.debug('Applying enhancements: {}'.format(self.enhancements))
            for frame in self.frame_objs:
                frame.enhance(enhancements=enhancements)
            self.enhance_str = frame.enhance_str
            self._enhanced += enhancements

    def reset_frames(self):
        """ Reset frames to their original raw date before enhancements etc were applied
        """
        for frame in self.frame_objs:
            frame.reset_frame()

    def setBgsubHistory(self, frame, Nbackwards=10, Nforwards=0, stepBackwards=1, stepForwards=1, skipBackwards=0,
                        skipForwards=0, verbose=True):
        """ Create a frame history centred around the supplied frame """
        if verbose: logger.debug('Setting frame history')
        # Initialize background model

        # for i, frame in enumerate(self.frames[0:self.Nbgsub]):
        #     dummy = self.bgsub.apply(frame)
        logger.debug('In setBgsubHistory')
        self.bgsub_settings = {'Nbackwards': Nbackwards, 'Nforwards': Nforwards, 'skipBackwards': skipBackwards,
                               'skipForwards': skipForwards,
                               'stepBackwards': stepBackwards, 'stepForwards': stepForwards}

        Ntot = self.bgsub_settings['Nbackwards'] + self.bgsub_settings['Nforwards']

        if Ntot > 0:
            frameNum0 = self.frame_objs.index(frame)
            frameMark0 = frame.n

            frame_nos = self.get_frame_list(frameMark0)
            logger.debug('Frames in bgsub history:  {}'.format(str(frame_nos)))
            logger.debug('Unique frames in bgsub history:  {}'.format(str(frame_nos)))

            from pyFastcamTools.backgroundSubtractor import backgroundSubtractorMin

            # Set new frame history
            # Only apply enhancements before 'BG subtraction' in enhancements list
            enhancements = self.enhancements if 'BG subtraction' not in self.enhancements else self.enhancements[
                                                                                            0: self.enhancements.index(
                                                                                                   'BG subtraction')]
            if self.bgsub is None:  # Initialise bg subtractor
                self.bgsub = backgroundSubtractorMin(len(frame_nos))
            elif len(frame_nos) != self.bgsub._history.N:  # Reallocate memory for new frame history length
                logger.info('Reallocating memory for bgsub, length {} -> {}'.format(
                                                                    self.bgsub._history.N, len(frame_nos)))
                self.bgsub._history.allocate_frame_memory(frame, N=len(frame_nos))
            else:
                self.bgsub._history.next_frame = 0  # loop over existing frames
                logger.debug('Set bgsub next_frame to 0 as still n_unique == self.N == {}'.format(len(frame_nos)))
            self.bgsub._STATIC_HISTORY = False  # When next apply in loop below add frames to _history
            # print('self.bgsub._history.next_frame', self.bgsub._history.next_frame, frame_nos)
            for n in frame_nos:  # TODO: make this clever, so it only adds frames that haven't already been computed!
                # make sure copy else changes frameHistory object for current frame --> very confusing!!!
                # import pdb; pdb.set_trace()
                # print('self.bgsub._history.next_frame, n', self.bgsub._history.next_frame, n)
                self.bgsub.apply(copy(self(n=n)).enhance(enhancements), get_foreground=False)

            if self.bgsub._history.next_frame != len(frame_nos):
                print('WARNING: self.bgsub._history.next_frame != len(frame_nos): {} != {}'.format(
                    self.bgsub._history.next_frame, len(frame_nos)))
                print('frame_nos', frame_nos, len(frame_nos))
                # import pdb; pdb.set_trace()
            self.bgsub._STATIC_HISTORY = True  # When next apply, do not change frame history by adding current frame
            # self.bgsub.apply(self.frames[i], )
            # import pdb; pdb.set_trace()
        else:  # TODO: fix bug when frame history length is zero
            print('WARNING: Ntot = {} <= 0'.format(Ntot))
            ## Set new frame history
            self.bgsub = backgroundSubtractorMin(1)
            blank = np.zeros(frame.shape)
            self.bgsub.apply(blank)

        # def setROI(self,ROI=[]):
        # 	"""
        # 	sets the ROI of the frames in the history
        #
        # 	"""
        #
        # 	if not ROI:
        # 		"""
        # 		User selection of the ROI
        #
        # 		"""
        # 		ROI = []
        # 		def onClick(event):
        # 			ROI.append([int(event.ydata),int(event.xdata)])
        # 			plt.scatter([event.xdata],[event.ydata],c='r')
        # 			plt.draw()
        #
        # 		fig = plt.figure()
        # 		cid = fig.canvas.mpl_connect('button_press_event', onClick)
        # 		plt.imshow(self[0])
        # 		plt.title("Please select ROI coordinates")
        # 		plt.xlim(0,self[0].shape[1])
        # 		plt.ylim(self[0].shape[0],0)
        # 		plt.show()
        #
        # 		print "\n ROI Coordinates: \n",ROI
        #
        # 	self.ROI = np.array(ROI)
        #
        #
        # def getROImask(self,ROI=None):
        # 	"""
        # 	return a boolean mask determining which points in the frames are within the ROI
        # 	"""
        # 	if ROI is None:
        # 		ROI = self.ROI
        #
        # 	def point_inside_ROI(point,ROI):
        #

    # 		n = ROI.shape[0]
    #
    # 		inside = 0
    # 		x,y = point
    # 		p1x = ROI[0,0]
    # 		p1y = ROI[0,1]
    # 		for i in range(n+1):
    # 			p2x = ROI[i % n,0]
    # 			p2y = ROI[i % n,1]
    # 			if y > min(p1y,p2y) and y <= max(p1y,p2y) and x <= max(p1x,p2x):
    # 				if p1y != p2y:
    # 					xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
    # 				if p1x == p2x or x <= xinters:
    # 					inside = (inside + 1) % 2
    #
    # 			p1x,p1y = p2x,p2y
    #
    # 		return inside
    #
    # 	nx = np.max(ROI[:,0]) - np.min(ROI[:,0]) + 1
    # 	ny = np.max(ROI[-1,1]) - np.min(ROI[0,1]) + 1
    #
    #
    # 	xpoints = np.arange(nx) + np.min(ROI[:,0])
    # 	ypoints = np.arange(ny) + np.min(ROI[:,1])
    #
    # 	pointsinROI = np.zeros(self[0].shape,dtype=np.uint8)
    # 	pointsinROI[...] = False
    # 	for x in xpoints:
    # 		for y in ypoints:
    # 			pointsinROI[x,y] = point_inside_ROI((x,y),ROI)
    #
    # 	return np.uint8(pointsinROI)


    def __iter__(self):
        """
        Iterate frames using a call like

        for frame in frameHistory:

        """
        for N in np.arange(self.frames.shape[0]):
            yield self.frames[N]

    def __getitem__(self, index, mask=False):
        """
        Access frames in the frameHistory using
        frameHistory()[i,j,k,...]
        """
        if self.frames is None:
            logger.warning('Cannot index frame history when no frames are loaded')
            return
        if not mask:
            if isinstance(index, (int, long)):
                # Return individual frame object
                return self.frame_objs[index]
            else:
                # Return slice/individual pixel intensities
                return self.frames[index]

    def __setitem__(self, index, setvalue):
        """
        Set individual frames using index
        """
        self.frames[index] = setvalue

    def __str__(self):
        info = self.info + ':' if self.info is not None else ''
        return '<FrameHistory({0}{1[0]}x{1[1]}x{1[2]}):{2}>'.format(info, self.shape, self.next_frame)

    def __call__(self, **kwargs):
        """ Return frame_obj object for given condition
        """
        # import pdb; pdb.set_trace()
        i = np.array(self.lookup('i', **kwargs)).astype(int)  # find index of frame
        i = i[i >= 0].tolist()  # integer nan = -9223372036854775808
        out = [self.frame_objs[j] for j in i]  # get list of frame objects
        if len(out) == 1:
            return out[0]
        else:
            return out

    def lookup(self, var, **kwargs):  # Need to extend to multiple kwargs
        """ Return value of var in self.frames_meta for which each keyword value is satisfied
        Example call signature
        self.lookup('n', t=0.2615)  # Return frame number of frame with time stamp 't'=0.2615 s
        """
        if len(kwargs) == 0:
            return None
        assert len(kwargs) == 1
        assert var in self.frames_meta.columns

        for key, value in kwargs.iteritems():
            assert key in self.frames_meta.columns.tolist()
            series = self.frames_meta[self.frames_meta[key] == value][var]
            # idx1.intersection(idx2)

        if len(series.values) == 1:
            return series.values[0]
        else:
            return series.values

    def get_index(self, **kwargs):
        assert len(kwargs) == 1

        for key, value in kwargs.iteritems():
            index = self.frames_meta[self.frames_meta[key] == value].index  # assumes only one match
            # idx1.intersection(idx2)

        if len(index):
            return index[0]
        else:
            return index

    def get_frame(self, **kwargs):
        assert len(kwargs) == 1

        i = self.get_index(**kwargs)
        return self.frame_objs[i]