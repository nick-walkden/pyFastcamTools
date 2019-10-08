from Gui import Elzar

moviefile = '/Users/nwalkden/Documents/SOLTurb/Movies/29693/C001H001S0001-04.mraw'

Nframes = 100
startpos = 0.2


E = Elzar(moviefile=moviefile,startpos=startpos,Nframes=Nframes,transforms=['transpose','reverse_x'])

E.runUI()