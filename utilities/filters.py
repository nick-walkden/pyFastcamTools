def box_average(frames,boxsize):
    
  
    Ns = frames.shape[0]

    Ns_trim = Ns - (Ns % boxsize)

    newframes = frames[:Ns_trim]

    newframes.reshape((Ns_trim/boxsize,boxsize,frames.shape[1],frames.shape[2]))

    return newframes.mean(axis=1)
    
    
def lowpass_filter(frames,freq_norm):
    
    from scipy import signal
    
    #Set up a butterworth filter
    b,a = signal.butter(3, freq_norm)    
    
    #Apply the filter
    
    filtered_frames = signal.lfilter(b,a,frames,axis=0)
    
    return filtered_frames
    

def highpass_filter(frames,freq_norm):
    
    from scipy import signal
    
    #Set up a butterworth filter
    b,a = signal.butter(3, freq_norm,btype='high')    
    
    #Apply the filter
    
    filtered_frames = signal.lfilter(b,a,frames,axis=0)
    
    return filtered_frames
    
def bandpass_filter(frames,low,high):
    
    from scipy import signal
    
    #Set up a butterworth filter
    b,a = signal.butter(3, [low,high] ,btype='band')    
    
    #Apply the filter
    
    filtered_frames = signal.lfilter(b,a,frames,axis=0)
    
    return filtered_frames