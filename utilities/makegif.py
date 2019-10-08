from subprocess import call,Popen,PIPE
import numpy as np


nframes = 790

prepend = 'disrupt/'
extra = 'frame_'
filetype = '.png'

quality = 100
delay = 2

args = ''

for i in np.arange(nframes):
	args += ' '+prepend+extra+str(i)+filetype

call('convert -delay '+str(delay)+' -quality '+str(quality)+' '+args+' '+prepend+'disruption.gif',shell=True)

 
