import numpy as np
from read_movie import read_movie as read
from background_subtractor import *
#from pyFieldlineTracer.fieldlineTracer import get_fieldline_tracer

import matplotlib.pyplot as plt
#from scipy.stats import describe

import matplotlib as mpl
import os
import pickle
from scipy.interpolate import interp1d
#mpl.rcParams['font.family'] = 'serif'
fontsize = 15
params = {'backend':'ps',
        #'text.latex.preamble':['\usepackage{gensymb}'],
        'axes.labelsize':fontsize,
        'axes.titlesize':fontsize,
        'font.size':fontsize,
        'legend.fontsize':fontsize-2,
        'xtick.labelsize':fontsize-2,
        'ytick.labelsize':fontsize-2,
#        'test.usetex':True,
        'font.family':'serif'}
mpl.rcParams.update(params)
from matplotlib.gridspec import GridSpec as grid
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
A quick script to read a movie and produce background subtracted movie frames


Nick Walkden, UKAEA
21st Nov 2016

"""

with_orig = True
time_stamp = True
bgsub = True

skip = 1

############## User Settings ##############


#For a stack of images, need to give the directory, not the file

moviefile = '/Volumes/SAMSUNG/SA1/29646/C001H001S0001/C001H001S0001-03.mraw'
#moviefile = '/net/fuslsa/data/MAST_IMAGES/029/29768/rbb029768.ipx' 
Nframes = 200
startframe = 3100

Nbg = 20  
bgsub = background_subtractor_min(Nbg)

gamma = 0.25

shot = 29768 
time = 0.24

save_path='29576/'



frames = read(moviefile,Nframes,stride=skip,startframe=startframe,endframe=startframe+Nframes+1,verbose=True,transforms=['reverse_x','transpose'])[:]
#print frames.shape
if bgsub:
    for frame in frames[0:Nbg]:
        bgsub.apply(frame)[:]
bg = bgsub.background_model
frames_sub = np.zeros((frames.shape[0]-Nbg,frames.shape[1],frames.shape[2]))
frames=frames[Nbg:]
if bgsub:
    for i in np.arange(frames.shape[0]):
        frames_sub[i] = bgsub.apply(frames[i])**gamma
    else:
        frames_sub[i] = frames[i]**gamma
#plt.ion()

sub_max = np.max(frames_sub)

frames_max = np.max(frames)
for i in np.arange(frames_sub.shape[0]):
    if with_orig:
        out_frame = np.concatenate([sub_max*frames[i]/frames_max,frames_sub[i]],axis=1)
    else:
        out_frame = frames_sub[i]

    
        
    plt.imshow(out_frame,cmap='gray',vmin=0,vmax=sub_max,interpolation='none')
    plt.axis('off')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(save_path+'frame_'+str(i)+'.png',bbox_inches='tight')
    plt.clf()
    
    
    
    
