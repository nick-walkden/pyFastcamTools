#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
__author__ = 'tfarley'

import unittest
import os
import pickle
import inspect
import numpy as np
from pyFastcamTools.Frames import Frame, ExperimentalFrame, SyntheticFrame
from pyFastcamTools.readMovie import readMovie
import elzar
from elzar.filaments import RphiGrid, FrameStore

pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

class TestFrame(unittest.TestCase):

    def setUp(self):
        self.fn = os.path.join(pwd, 'Frame_7_width_2.0.p')
        self.frame_data = pickle.load(open(self.fn, 'rb'))

    def test_create_without_frame_history(self):
        print('- Running test_create_without_frame_history')

        self.frame = Frame(self.frame_data)

    def test_enhance(self):
        print('- Running test_enhance')

        self.frame = Frame(self.frame_data)
        enhancements = ['BG subtraction','Reduce noise', 'Sharpen', 'Equalise (adaptive)', 'Negative']
        self.frame.enhance(enhancements, store_enhanced=True)
        # self.frame.plot()

    def test_bgsub(self):
        print('- Running test_bgsub')

        frames = readMovie('/net/edge1/scratch/jrh/SA1/rbf029852.ipx', Nframes=40, startframe=2500, endpos=0.7)
        # print('frames_meta:\n', frames.frames_meta)
        # print('t', frames.timestamps)
        self.frame = frames[20]
        enhancements = ['BG subtraction']
        self.frame.enhance(enhancements, store_enhanced=True)
        # self.frame.plot()

    def tearDown(self):
        pass

class TestSyntheticFrame(unittest.TestCase):

    def setUp(self):
        self.fn = os.path.join(pwd, 'Frame_7_width_2.0.p')
        self.frame = SyntheticFrame(self.fn, fn_log='info_width_2.0.log')
        pass

    def test_load_from_pickle(self):
        print('Running test_load_from_pickle')

        self.frame = SyntheticFrame(self.fn, fn_log='info_width_2.0.log')
        self.assertEqual(self.frame.n, 7)
        self.assertTrue(isinstance(self.frame.data, dict), 'Filament information dict associated with frame not loaded')
        with self.assertRaises(AssertionError):
            frame = SyntheticFrame('This is not a path!', fn_log='info_width_2.0.log')

    def test_plot(self):
        print('Running test_plot')

        # frame = SyntheticFrame(self.fn, fn_log='info_width_2.0.log')
        self.frame.plot(show=False)

    def tearDown(self):
        pass


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestFrame('test_bgsub'))
    suite.addTest(TestFrame('test_enhance'))
    # suite.addTest(FooTestCase('test_eleven'))
    # suite.addTest(BarTestCase('test_twelve'))
    return suite 

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())