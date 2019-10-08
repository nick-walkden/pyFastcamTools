from pyFastcamTools.utilities.read_movie import read_movie
from pyFastcamTools.utilities.filters import *
from pyFastcamTools.utilities.background_subtractor import *
import matplotlib.pyplot as plt
import numpy as np


frames = read_movie('/Users/nwalkden/Documents/SOLTurb/Movies/29724/C001H001S0001/C001H001S0001-04.mraw',Nframes=500,startframe=1000,transforms=['transpose','reverse_x'])[:]
DC = lowpass_filter(frames,0.1)[-200:]
low = bandpass_filter(frames,0.1,0.3)[-200:]
mid1 = bandpass_filter(frames,0.3,0.5)[-200:]
mid2 = bandpass_filter(frames,0.5,0.7)[-200:]
mid3 = bandpass_filter(frames,0.7,0.9)[-200:]
high = highpass_filter(frames,0.9)[-200:]

bgsub = background_subtractor_min(10)
for frame in frames[-200:-100]:
    bgsub.apply(frame)

#bgsubframes = []
#for frame in frames[-100:]:
#    bgsubframes.append(bgsub.apply(frame))
#
#bgsubframes=np.array(bgsubframes)  
#
#bgsubframes -= bgsubframes[-1].min()
#bgsubframes /= bgsubframes[-1].max() - bgsubframes[-1].min()  

im = np.concatenate([(DC-DC.min())/(DC.max()-DC.min()),(low-low.min())/(low.max()-low.min()),(mid1-mid1.min())/(mid1.max()-mid1.min()),(mid2-mid2.min())/(mid2.max()-mid2.min()),(mid3-mid3.min())/(mid3.max()-mid3.min()),(high-high.min())/(high.max()-high.min())],axis=2)

plt.imshow(DC.mean(axis=0),interpolation='none')
plt.show()
plt.imshow(low.mean(axis=0),interpolation='none')
plt.show()
plt.imshow(mid1.mean(axis=0),interpolation='none')
plt.show()
plt.imshow(mid2.mean(axis=0),interpolation='none')
plt.show()
plt.imshow(mid3.mean(axis=0),interpolation='none')
plt.show()
plt.imshow(high.mean(axis=0),interpolation='none')
plt.show()
dqsdsa

for i in np.arange(200):
    
    
    j = i
    
    
    plt.imshow(im[j],interpolation='none')
    plt.gcf().set_size_inches(15,105)
    plt.savefig('frames/29724_freq_decomp_'+str(i)+'.eps',bbox_inches='tight')
    plt.clf()