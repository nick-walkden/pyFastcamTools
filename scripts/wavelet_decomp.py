from pyFastcamTools.utilities.read_movie import read_movie
from pyFastcamTools.utilities.filters import *
from pyFastcamTools.utilities.background_subtractor import *
import matplotlib.pyplot as plt
import numpy as np
import pywt


frames = read_movie('/Volumes/SAMSUNG/SA1/29611/C001H001S0001/C001H001S0001-03.mraw',Nframes=200,startframe=1000,transforms=['transpose','reverse_x'])[:]
#DC = lowpass_filter(frames,0.1)[-200:]
#low = bandpass_filter(frames,0.1,0.95)[-200:]
#mid1 = bandpass_filter(frames,0.3,0.5)[-200:]
#mid2 = bandpass_filter(frames,0.5,0.7)[-200:]
#mid3 = bandpass_filter(frames,0.7,0.9)[-200:]
#high = highpass_filter(frames,0.9)[-200:]

frames_sub = run_bgsub_min(frames,20)

plt.imshow(frames_sub[-1])
plt.show()
print(pywt.wavelist())
coeffs = pywt.dwt2(frames_sub, 'db2',axes=(1,2))

cA, (cH, cV, cD) = coeffs

fig,ax = plt.subplots(1,5,sharex=True,sharey=True)
for i,a in enumerate([cA,cH,cV,cD]):
    print(np.min(a),np.max(a))
    #a -= a[:,0,0].mean()
    
    ax[i+1].imshow(a[-1],cmap='gray')
#cH[:] = 0
#cV[:] = 0
#cD[:] = 0
coeffs = cA, (cH,cV,cD)    
ax[0].imshow(pywt.idwt2(coeffs,'db2',axes=(1,2))[-1,::2,::2],cmap='gray')
plt.show()

