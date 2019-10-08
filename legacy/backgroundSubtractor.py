#!usr/bin/env python
"""
Classes and methods used to create a background subtraction of a given image

Nick Walkden, May 2015
"""

import numpy as np
import cv2
from pyFastcamTools.create_log import create_log
from pyFastcamTools.frameHistory import frameHistory
from pyFastcamTools.Frames import Frame


class backgroundSubtractor:
    """
    Class mimicing opencv background subtractors

    Develop a background model from a frame history

    Assume that the input is a greyscale image

    """

    def __init__(self, history=None):
        logger = create_log('backgroundSubtractor')
        logger.info('backgroundSubtractor instance created')

        self._history = frameHistory(info=str(self.__class__).replace(str(self.__class__.__module__) + '.', ''),
                                     loop_frames=True)
        self.backgroundModel = None
        self._STATIC_HISTORY = False
        self._history.N = 10  # Use 10 frames by default
        if history is not None:
            self.setHistory(history)

    def apply(self, frame, get_foreground=True):

        # First add the current frame to the history
        if not self._STATIC_HISTORY:
            # print('Adding frame to bgsub history')
            if issubclass(frame.__class__, Frame):
                self._history.add_frame(frame, no_repeat=True)
            else:
                self._history.add_array(frame, no_repeat=True)

        if get_foreground:
            self.getBackground()
            # self.backgroundModel = np.uint8(self.backgroundModel)
            # Convert to uint8 for opencv and zero out points below the background
            foreground = frame[:] - self.backgroundModel

            foreground[np.where(foreground < 0.0)] = 0.0
            # foreground = np.uint8(foreground)
            return foreground

    def setHistory(self, history):

        if isinstance(history, (int, long, float)):
            self._history.N = history  # Number of frames to store in history
        else:
            self._history.set(history)  # Set the history to the given frames and do not reset
            self._STATIC_HISTORY = True


class backgroundSubtractorMedian(backgroundSubtractor):
    """
    Take the median of each pixel in the frame history
    """

    def __init__(self, history=None):
        backgroundSubtractor.__init__(self)
        logger = create_log(__name__)
        logger.info('backgroundSubtractorMedian instance created')

    def getBackground(self):
        self.backgroundModel = np.median(self._history.frames, axis=0)


class backgroundSubtractorMin(backgroundSubtractor):
    """
    Take the median of each pixel in the frame history
    """

    def __init__(self, history=None):
        backgroundSubtractor.__init__(self, history)
        logger = create_log(__name__)
        logger.info('backgroundSubtractorMin instance created')

    def getBackground(self, learningRate=None):
        self.backgroundModel = np.min(self._history.frames, axis=0)


class backgroundSubtractorMean(backgroundSubtractor):
    """
    Take the mean of each pixel in the frame history
    """

    def __init__(self, history=None):
        backgroundSubtractor.__init__(self)
        logger = create_log(__name__)
        logger.info('backgroundSubtractorMean instance created')

    def getBackground(self, learningRate=None):
        self.backgroundModel = np.mean(self._history.frames, axis=0)


class backgroundSubtractorFFT(backgroundSubtractor):
    def __init__(self, history=None):
        backgroundSubtractor.__init__(self)
        logger = create_log(__name__)
        logger.info('backgroundSubtractorFFT instance created')

    def getBackground(self, learningRate=None):
        if self._history._updates < 2:
            self.backgroundModel = 0.0
            return
        Rfft = np.fft.rfft(self._history.frames, axis=0)
        # zero out all but DC and Nyquist component
        Rfft[2:-2, ...] = 0.0

        result = np.fft.irfft(Rfft, axis=0)
        self.backgroundModel = result[-1, ...]

# class backgroundSubtractorSVD(backgroundSubtractor):
