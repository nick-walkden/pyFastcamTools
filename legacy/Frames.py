#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
""" Classes for analysing filaments in synthetic fast camera data
"""
__author__ = 'tfarley'

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
# import matplotlib
import logging
import os
import pickle
import re
import inspect
import logging
import itertools
import cv2
import gc
from copy import copy, deepcopy

import psutil
proc = psutil.Process(os.getpid())

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

class Frame(object):
    EQUALITY_KEYS = ['n', 't', 'frame', 'raw',  'enhancements', 'frame_history']

    # ['info', 'enhancements', 'frame', 'pulse', 'n', 'raw', 'camera', 't', 'store_enhanced', 'frame_history', 'enhance_str', 'resolution']
    def __init__(self, frame=None, n=None, t=None, i=None, info=None, frame_history=None, **kwargs):
        logger.debug('Entered Frame.__init__')
        self.n = n  # Frame number in movie (frame 'mark')
        self.t = t  # Frame time
        self.i = i  # Frame index (in set of frames being considered ie relative to start frame)
        self.info = info
        self.frame_history = frame_history

        self.raw = frame  # Raw data - always point to data in frame history if there is one
        self.frame = self.raw  # Enhanced data - local to this Frame object
        self.enhancements = []
        self._enhanced = []  # Record of currently applied enhancements
        self.enhance_str = ''
        self.abs_gauss_noise_level = None  # Level of gaussian noise applied to frames
        self.store_enhanced = True  # if true, enhancements will modify self.frame (not just be returned in output from enhancements method)

        self.refresh()  # set resolution etc

        if frame_history is not None and self.frame is not None:
            from pyFastcamTools.frameHistory import frameHistory
            assert issubclass(frame_history.__class__, frameHistory)

            _func = frame_history.add_frame
            kws = {k: v for k, v in kwargs.iteritems() if k in inspect.getargspec(_func)[0]}
            _func(self, **kws)

        self.refresh()

        for arg in ['frame', 'pulse', 'n', 't', 'info', 'camera', 'fig', 'ax']:
            self.__dict__.setdefault(arg, None)
        pass
        logger.debug('Leaving Frame.__init__')


    def __getitem__(self, item):
        return self.frame[item]  # Enhanced data - local to this Frame object

    def __setitem__(self, key, value):
        self.frame[key] = value

    def __str__(self):
        return '\n'+repr(self)+'\n'+self.frame.__str__()

    def __eq__(self, other):
        # return self is other
        if isinstance(other, np.ndarray):
            return self.frame == other
        elif issubclass(other.__class__, Frame):
            try:
                _dict = {key: self.__dict__[key] for key in self.__class__.EQUALITY_KEYS}
                result = np.all([np.all(other.__dict__[key] == value) for key, value in _dict.iteritems()])
                return result
            except KeyError:
                return False

    def __ne__(self, other):
        return not self == other

    def __contains__(self, item):
        return self.frame.__contains__(item)

    def __contains__(self, item):
        return self.frame.__contains__(item)

    def __add__(self, item):
        return self.frame.__add__(item)

    def __sub__(self, item):
        return self.frame.__sub__(item)

    def __mul__(self, item):
        return self.frame.__mul__(item)

    def __div__(self, item):
        return self.frame.__div__(item)

    # def add_to_frame_history(self, frame_history):
    #     pass

    def refresh(self):
        if self.frame is not None:
            logger.debug('Refreshed frame resolution')
            self.resolution = self.frame.shape

    def stats(self, print_=True):
        stats = scipy.stats.describe(self.frame.flatten())
        if print_:
            print(stats)
        return stats

    def plot(self, show=True, save=False, annotate=True):
        self.clColor = (242/255,241/255,239/255)  # Qt widget color

        self.fig = plt.figure(repr(self), facecolor=self.clColor, edgecolor=self.clColor)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(0, 0, 1, 1)  # maximise figure margins so image fills full canvas
        self.ax.set_xlim(-0.5, self.frame.shape[1]-0.5)
        self.ax.set_ylim(self.frame.shape[0]-0.5, -0.5)
        img = self.ax.imshow(self.frame, cmap='gray', interpolation='none')

        if annotate:
            n_str = 'Frame: {:d}'.format(self.n) if self.n is not None else ''
            t_str = '  Time: {:0.5f} [s]'.format(self.t) if self.t is not None else ''
            text = n_str + t_str
            frametxt = self.ax.annotate(text, xy=(0.05,0.95), xycoords='axes fraction', color='white', fontsize=8)
            frametxt.set_bbox(dict(color='k', alpha=0.5, edgecolor=None))
        plt.tight_layout()
        if save:
            if not isinstance(save, basestring):
                if self.frame_history is not None:
                    save = self.frame_history.out_name(n=self.n, prefix='frame', extension='.png',
                                                        dtype=True, resolution=False, ident=False, enhancements=True)
                    save = os.path.join(os.path.expanduser('~/elzar'), 'images', 'frames', save)
                assert isinstance(save, basestring)
                # fn = 'frame-'+self.name_out+'.png'
                # save = os.path.join(self.image_path, 'frames', fn)
            self.fig.savefig(save, bbox_inches='tight', transparent=True, dpi=90)
            logger.info('Frame image saved to: '+save)
        if show:
            plt.show()
        return self.fig

    def hist(self, nbins=None, show=True, save=False, log=True):
        self.clColor = (242/255,241/255,239/255)  # Qt widget color

        if self.fig is not None:
            self.fig.clear()
            self.fig = None
        self.fig = plt.figure('Histogram: '+repr(self), facecolor=self.clColor, edgecolor=self.clColor)
        self.ax = self.fig.add_subplot(111)
        # the histogram of the data
        if nbins is None:
            nbins = np.amax(self.frame) - np.amin(self.frame)
            nbins = nbins / 10**(np.round(np.log10(0.5))-1)
        n, bins, patches = self.ax.hist(self.frame.flatten(), nbins,
                                        normed=False, facecolor='green', alpha=0.75, log=log)
        self.ax.set_xlabel('Pixel intensity [arb]')
        self.ax.set_ylabel('Frequency')
        self.ax.grid()
        if save:
            if not isinstance(save, basestring):
                fn = 'hist-'+self.name_out+'.png'
                save = os.path.join(self.image_path, 'frames', fn)
            self.fig.savefig(save, bbox_inches='tight', transparent=True, dpi=90)
            logger.info('Histogram of frame pixel intenisties saved to: '+save)
        if show:
            plt.show()
        return self.fig


    def subclass(self, cls):
        """ Convert a base frame instance to an instance of one of its subclasses eg SyntheticFrame
        """
        assert issubclass(cls, self.__class__)
        # self.__class__ = cls
        # self.__init__()
        child = cls()
        for key, value in self.__dict__.iteritems():
            child.__dict__[key] = value
        return child

    def add_abs_gauss_noise(self, width=0.05, return_noise=False):
        """ Add noise to frame to emulate experimental radom noise. A positive definite gaussian distribution is used
        so as to best model the noise in background subtracted frame data
        """
        self.abs_gauss_noise_level = width
        scale = width * np.ptp(self.frame)
        noise = np.abs(np.random.normal(loc=0.0, scale=scale, size=self.frame.shape))
        if not return_noise:
            self.frame += noise
        else:
            return noise
        # return self.frame

    def enhance(self, enhancements, store_enhanced=None, reset=True):
        """ Process image applying background subtraction and openCV routines etc
        """
        # If same enhancements have already been applied and stored, return the stored enhanced frame
        if self._enhanced == enhancements and store_enhanced:
            return self.frame
        else:
            self.enhancements = enhancements
            logger.debug('Applying enhancements {} to frame {}'.format(self.enhancements, repr(self)))

        if reset:
            self.reset_frame()  # Reset the frame to the original raw data

        if store_enhanced is not None:
            self.store_enhanced = store_enhanced

        enhance_strs = {'BG subtraction': 'bg', 'Threshold': 'th', 'Detect edges': 'eD','Gamma enhance': 'gm',
                        'Equalise (global)': 'eqG', 'Equalise (adaptive)': 'eqA', 'Reduce noise': 'rN',
                        'Negative': 'ne', 'Sharpen': 'sh', 'Add noise': 'aN'}
        logger.debug('Applying enhancements', enhancements)
        max_I = float(np.max(self.raw))

        frame = (self.frame*255.0/max_I).astype(np.uint8) # convert frame data to 8 bit bitmap for cv2

        try:
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)#np.uint8(frame*255.0/np.max(frame))
        except:
            print('Failed to convert COLOR_GRAY2BGR')
            pass

        ## Background subtraction
        def apply_sub(frame):
            if self.frame_history is not None:
                try:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#np.uint8(frame*255.0/np.max(frame))
                except:
                    pass
                frame = frame*max_I/255.0  # convert frame data from 8 bit bitmap to original format
                # print('Memory before bgsub_history: {:0.4g}'.format(proc.get_memory_info().rss))
                self.frame_history.setBgsubHistory(self)
                frame = self.frame_history.bgsub.apply(frame)
                # print('Memory after bgsub_history:  {:0.4g}'.format(proc.get_memory_info().rss))
                # del self.frame_history.bgsub  # free memory
                # print('Memory after del bgsub:      {:0.4g}'.format(proc.get_memory_info().rss))
                gc.collect()
                # print('Memory after gc.collect:     {:0.4g}'.format(proc.get_memory_info().rss))

                max_I2 = float(np.max(frame))
                frame = (frame*255.0/max_I2).astype(np.uint8) # convert frame data to 8 bit bitmap for cv2
                try:
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)#np.uint8(frame*255.0/np.max(frame))
                except:
                    pass
                return frame
            else:
                logger.warning('Cannot apply background subtraction to frame without frame_history')
                return frame

        ## Noise reduction
        def apply_noiseRmv(frame):
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            except:
               pass
            # frame = cv2.guidedFilter(frame, frame, 3, 9)  # guide, src (in), radius, eps  -- requires OpenCV3
            frame = cv2.bilateralFilter(frame,5,75,75)  # strong but slow noise filter
            #frame = cv2.fastNlMeansDenoising(frame,None,7,21)
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            except:
                pass
            return frame

        ## Threshold
        def apply_threshold(frame):
            _,frame = cv2.threshold(frame,10,255,cv2.THRESH_BINARY) # image, threshold value, maxVal, threshold type
            # _,frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THREESH_BINARY, 11, 2) # image, threshold value, maxVal, threshold type
            return frame

        ## Gamma enahnce
        def apply_gammaEnhance(frame):
            gammaframe = np.float64(frame)**(self.gamma)
            frame = np.uint8(gammaframe*255.0/np.max(gammaframe))
            return frame

        ## Equalise
        def apply_histEq(frame):
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            except:
                pass
            frame = cv2.equalizeHist(frame)
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            except:
                pass
            return frame
                ## Equalise
        ## Adaptive equalise CLAHE (Contrast Limited Adaptive Histogram Equalization)
        def apply_histEqClahe(frame):
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            except:
                pass
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            frame = clahe.apply(frame)
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            except:
                pass
            return frame
        ## Edge detection
        def apply_edgeDet(frame):
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            except:
                pass
            frame = cv2.Canny(frame,50,250,True) # 500, 550
            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            except:
                pass
            return frame
        ## Invert image
        def apply_negative(frame):
            frame = 255 - frame
            return frame
        ## Sharpen image
        def apply_sharpen(frame):
            ## Simple unsharp masking - gaussian blur
            frame_blur = cv2.GaussianBlur(frame,(15,15),16)
            ## Subtract gaussian blur from image - sharpens small features
            frame = cv2.addWeighted(frame,1.5,frame_blur,-0.5,0.0)
            return frame
            # try:
            #     ## Convert back to single colour 2D frame - clashes with mask
            #     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # except:
            #     pass
        def add_noise(frame):
            noise = self.add_abs_gauss_noise(return_noise=True)
            # import pdb; pdb.set_trace()
            noise = (noise*255.0/max_I).astype(np.uint8)
            noise = cv2.cvtColor(noise,cv2.COLOR_GRAY2BGR)
            frame = cv2.add(frame, noise)  # Avoid integer overflow
            return frame

        enhance_funcs = {'BG subtraction': apply_sub, 'Threshold': apply_threshold, 'Detect edges': apply_edgeDet,
                         'Gamma enhance': apply_gammaEnhance, 'Equalise (global)': apply_histEq,
                         'Equalise (adaptive)': apply_histEqClahe, 'Reduce noise': apply_noiseRmv,
                         'Negative': apply_negative, 'Sharpen': apply_sharpen, 'Add noise': add_noise}
        self.enhance_str = []
        for enhancement in enhancements:
            func = enhance_funcs[enhancement]
            frame = func(frame)
            self.enhance_str.append(enhance_strs[enhancement])
        self.enhance_str = '_'.join(self.enhance_str)
        self._enhanced += self.enhancements

        # convert enhanced array back to original 2d grey format
        try:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#np.uint8(frame*255.0/np.max(frame))
        except:
            pass
        frame = frame*max_I/255.0  # convert frame data to 8 bit bitmap for cv2

        if self.store_enhanced:  # store enhanced frame - requires more memory if set for whole frame history
            self.frame = frame
            return self
        else:  # return a copy of the current frame with the data set to the enhanced values
            frame_ob = copy(self)
            frame_ob.frame = frame
            return frame_ob

    def reset_frame(self):
        """ Reset frame to original data before enhancements and noise addition
        """
        self.frame = self.raw.copy()
        # self.enhancements = []  # ordered list of enhancements applied to frame
        self.enhance_str = ''  # string detailing enhancements for file output
        self._enhanced = []
        self.abs_gauss_noise_level = None
        # return self.frame


class SyntheticFrame(Frame):
    """ Object to hold synthetic frame data along with meta data about the synthetic filaments in the image and the
    fit results of the filament identification algorithm
    """
    # EQUALITY_KEYS = [itertools.chain.from_iterable([cls.EQUALITY_KEYS for cls in SyntheticFrame.__subclasses__()])]
    def __init__(self, path=None, **kwargs):
        # Call parent Frame object's __init__
        # _func = super(SyntheticFrame, self).__init__
        _func = Frame.__init__
        kws = {k: v for k, v in kwargs.iteritems() if k in inspect.getargspec(_func)[0]}
        _func(self, **kws)

        self.path = path

    def load_from_objects(self):
        pass

class ExperimentalFrame(Frame):
    def __init__(self, pulse=None, camera=None, **kwargs):
        logger.debug('Entered ExperimentalFrame.__init__')
        self.camera = camera
        self.pulse = pulse
        # Call parent Frame object's __init__
        _func = Frame.__init__
        # _func = super(ExperimentalFrame, self).__init__
        kws = {k: v for k, v in kwargs.iteritems() if k in inspect.getargspec(_func)[0]}
        # print('kws sent to Frame.__init__:', kws)
        _func(self, **kws)

        self.name_out = 'p{}_n{}'.format(self.pulse, self.n)

        logger.debug('Leaving ExperimentalFrame.__init__')

    def __repr__(self):
        res_str = '({0[0]}x{0[1]})'.format(self.resolution) if self.resolution is not None else ''
        # import re
        # m = re.search("<class '(.*)\.?(.*)'>", str(self.__class__))
        # class_str = m.group(2) if m.group(2) != '' else m.group(1)
        return '{}({}, cam={}, pulse={}, t={}, n={})'.format(self.__class__.__name__, res_str, self.camera, self.pulse,
                                                             self.t, self.n)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    path = '/home/nwalkden/python_tools/cySynthCam/error_analysis/full_frames/Nfil_41/'
    fn = 'Frame_27.p'
    frame = SyntheticFrame(path+fn, fn_log='info.log')
    # frame.plot(save=True, show=False)
    # frame.identify_filaments(pickle_settings='111')
    print(frame[2:4, 1:3])
    a = frame[np.newaxis, ...]
    a = np.append(a, frame[np.newaxis, ...], axis=0)
    pass

    frame2 = Frame()

    frame2 = frame2.subclass(SyntheticFrame)

    frame3 = ExperimentalFrame(frame=frame.frame, n=3, t=0.265)
    print(issubclass(frame3.__class__, ExperimentalFrame))
    frame3.stats()
    frame3.hist(log=True)
    pass
