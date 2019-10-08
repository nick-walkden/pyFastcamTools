from readMovie import readMovie
import matplotlib.pyplot as plt
from backgroundSubtractor import backgroundSubtractorMin
moviefile = '/home/nwalkden/Movies/29656/C001H001S0001-03.mraw'


frames = readMovie(moviefile,Nframes=100,startpos=0.2,transforms=['transpose','reverse_x'])

plt.imshow(frames[0])
plt.show()


bgsub = backgroundSubtractorMin(20)

bgframes = []
for frame in frames:
    bgframes.append(bgsub.apply(frame))
    
bgframes = bgframes[20:]

plt.imshow(bgframes[0])
plt.show()    



from Elzar import Elzar

UI = Elzar(Nframes=500,startpos=0.2,moviefile=moviefile,transforms=['transpose','reverse_x'])


UI.runUI()	