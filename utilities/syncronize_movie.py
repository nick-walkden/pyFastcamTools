import numpy as np
from readMovie import readMovie as read
import matplotlib.pyplot as plt
from backgroundSubtractor import backgroundSubtractorMin

def syncronize_movie(movie,times,t_window=1e-3,N_interval=3,t_start=1e-4,bgsub=0):
    """
    This function splits the input movie into data chunks sycronized to the ELM times given. 
    
    ARGUMENTS
        movie:  The movie object from a call to readMovie that the ELMs are contained in
        ELM_times:   A list of times that ELMs occur and that the movie will be syncronized around
    
    KEYWORDS
        t_window:  The time in seconds to window after an ELM has occured
        N_inter:   The number of intervals to split the window after an ELM into
        t_start: The time in seconds, relative to t_window, to start from
                   If t_window = 1e-3, Ninter = 3 and t_start = 1e-4, the analysis will produce frames
                   in the range 0.1 - 0.4ms, 0.4 - 0.7ms and 0.7 - 1ms after each time
    
                   if t_window = 1e-3, Ninter = 2 and t_start = -0.5e-3, the analysis will produce 
                   frames in the range -0.5ms - 0ms, 0ms - 0.5ms around each t_window 

        bgsub:  if > 0, perform background subtraction with a framehistory of bgsub number of frames
                if <= 0 dont perform background subtraction
    
    RETURNS
        synced_frames:  A list of the ELM_synced frames for each interval within the analysis window
        synced_times:   A list of the times at which the ELM_synced frame data is taken
    
    
    
    EXAMPLE (requires readMovie)
    
        moviefile = '/home/selmore/Movies/29795/C001H001S0001-04.mraw'
        Nframes = 2499
        startframe = 2500

        ##### Times of each ELM to be included in analysis #####
        ELM_times = [0.259]

        ##### Total window after an ELM to be analyses #####
        t_window = 1.5e-3

        ##### Number of intervals in ELM window to be used #####
        N_inter = 3

        ##### Time to exclude immediately after the ELM (ie due to data contamination) #####
        t_exclude = 1.5e-4

        movie = readMovie(moviefile,Nframes,startframe=startframe,endframe=startframe+Nframes+1,transforms=['transpose','reverse_y'],verbose=True)
    
        synced_frames,synced_times = get_elm_syned_movie(movie,ELM_times,t_window=t_window,N_inter=N_inter,t_exclude=t_exclude,bgsub=0)
    """ 
     
    frames = movie[:]
    times = movie.timestamps
    
    frames_sub = np.zeros(frames.shape)
    
    if bgsub > 0:
        bgsub = backgroundSubtractorMin(bgsub)
    
        for i in np.arange(frames.shape[0]):
            frames_sub[i] = bgsub.apply(frames[i])
        frames = frames_sub
        
    
    synced_frames = []
    synced_times = []
    
    for i in np.arange(N_inter):
        
        synced_frames.append([])
        synced_times.append([])
    
    for time in times:
      
        try:
            start_ind = np.where(times >= time + t_exclude)[0][0]
        except:
            raise Error("Error: time outside movie time window")
        try:
            end_ind = np.where(times >= time + t_window)[0][0]
        except:
            if time + t_window > times[-1]:
                end_ind = frames.shape[0] - 1
            else:
                raise Error("Error: end_ind calculation failed")
        
        chunked_frames = np.array_split(frames[start_ind:end_ind],N_inter,axis=0)
        chunked_times = np.array_split(times[start_ind:end_ind],N_inter,axis=0)
        
        for i in np.arange(N_inter):
            synced_frames[i].append(chunked_frames[i])
            synced_times[i].append(chunked_times[i])
    
    for i in np.arange(N_inter):
        
        synced_frames[i] = np.concatenate(synced_frames[i],axis=0)
        synced_times[i] = np.concatenate(synced_times[i],axis=0)
        
    return synced_frames,synced_times



def test_elm_sync():
    moviefile = '/home/selmore/Movies/29795/C001H001S0001-04.mraw'
    Nframes = 2499
    startframe = 2500

    ##### Times of each ELM to be included in analysis #####
    ELM_times = [0.259]

    ##### Total window after an ELM to be analyses #####
    t_window = 1.5e-3

    ##### Number of intervals in ELM window to be used #####
    N_inter = 3

    ##### Time to exclude immediately after the ELM (ie due to data contamination) #####
    t_exclude = 1.5e-4

    movie = read(moviefile,Nframes,startframe=startframe,endframe=startframe+Nframes+1,transforms=['transpose','reverse_y'],verbose=True)
    
    synced_frames,synced_times = get_elm_syned_movie(movie,ELM_times,t_window=t_window,N_inter=N_inter,t_exclude=t_exclude,bgsub=0)
    
    
    plt.plot(movie.timestamps,np.mean(movie[:],axis=(1,2)),'k',lw=0.5)
    for i in np.arange(N_inter):
        plt.plot(synced_times[i],np.mean(synced_frames[i],axis=(1,2)),'-x',lw=2)
        
    plt.show()


    fig,axs = plt.subplots(1,N_inter,sharey=True)
    for i in np.arange(N_inter):
        axs[i].imshow(np.std(synced_frames[i],axis=0))
    plt.show()
        
        
        
if __name__=='__main__':
    test_elm_sync()