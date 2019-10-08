#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
__author__ = 'tfarley'

import unittest
import os
import pickle
import inspect
import numpy as np
from pyFastcamTools.frameHistory import frameHistory
from pyFastcamTools.Frames import Frame, ExperimentalFrame, SyntheticFrame
from pyFastcamTools.readMovie import readMovie
import elzar
from elzar.filaments import RphiGrid, FrameStore

pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

class TestFrameHistory(unittest.TestCase):

    def setUp(self):
        self.frames = frameHistory(info='TestFrameHistory')
        self.frames.read_movie('/net/edge1/scratch/jrh/SA1/rbf029852.ipx', startframe=2500, endframe=2550)
        pass

    def test_lookup(self):
        print('- Running test_lookup')
        self.assertTrue(self.frames.lookup('t', n=2510) == 0.20511)
        self.assertTrue(self.frames.lookup('n', t=0.20511) == 2510)

    def test_get_frame(self):
        print('- Running test_get_frame')
        n = 2510
        self.assertTrue(self.frames.get_frame(n=n).n == n)




def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestFrame('test_bgsub'))
    suite.addTest(TestElzarFrames('test_create_without_frame_history'))
    # suite.addTest(FooTestCase('test_eleven'))
    # suite.addTest(BarTestCase('test_twelve'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())