import numpy as np

def get_camera_tangency_angle(calibration,R,Z):
    angs = np.linspace(0.0,360.0,361)

    def refine_angle(angles,start_angle=0.0,R=0.6,Z=-1.25):
        xind = 0
        current_ang = start_angle 
        #R = 0.6
        #Z = -1.2  
        for ang in angs: 
            X = R*np.cos(ang*2.0*np.pi/360.0)
            Y = R*np.sin(ang*2.0*np.pi/360.0)
            inds = calibration.ProjectPoints([X,Y,Z])
            #print(inds[0,0,0])
            if inds[0,0,0] > xind:
                xind = inds[0,0,0]
                current_ang = ang
        
        return current_ang
    ang = refine_angle(angs,R=R,Z=Z)
    #Second refinement
    angs = np.linspace(ang-10.0,ang+10.0,50)
    ang = refine_angle(angs,ang,R=R,Z=Z)   
    #Third refinement
    angs = np.linspace(ang - 0.5,ang + 0.5,50)
    #plt.clf()
    return refine_angle(angs,ang,R=R,Z=Z)*2.0*np.pi/360.0
    

def project_flux_surface(calibration,tracer,psiN,Zrange=None,Rrange=None):
    
    tan_ang = get_camera_tangency_angle(calibration,0.6,-1.25)
    print(tan_ang)
    
    fline = tracer.trace(1.4,0.0,ds=1e-2,mxstep=10000,psiN=psiN)
    
    Z = np.asarray(fline.Z).squeeze()
    X = np.asarray(fline.R).squeeze()*np.cos(tan_ang)
    Y = np.asarray(fline.R).squeeze()*np.sin(tan_ang)
    print(psiN,fline.R[0],fline.R[-1])
    inds = None
    r = np.asarray(fline.R).squeeze()
    if Zrange is not None:
        zmin = Zrange[0]
        zmax = Zrange[1]
        inds = np.where(Z > zmin)[0]
        #print(inds)	
        inds = inds[np.where(Z[inds] < zmax)[0]]
        #print(inds)
        Z = Z[inds]
        X = X[inds]
        Y = Y[inds]
        r = r[inds]
    if Rrange is not None:
        rmax = Rrange[0]
        rmin = Rrange[0]
        inds = np.where(r > rmin)[0]
        #print(inds)
        inds = inds[np.where(r[inds] < rmax)[0]]
        #print(inds)
        X = X[inds]
        Y = Y[inds]
        Z = Z[inds]
		
    objpoints = np.array([[X[i],Y[i],Z[i]] for i in np.arange(len(X))])
        
    return np.array(calibration.ProjectPoints(objpoints)[:,0,:]).squeeze(), r,Z
    
    
def cross_correlation(frames,coords=(0,0),delay=0):
    dims = frames[:].shape
    #frames = np.empty((dims[0]-abs(delay),dims[1],dims[2]))
    #Get pixel means and standard deviations

    frames_cor = frames - frames.mean(axis=0)
    frames_cor /= frames.std(axis=0)

    result = np.zeros((frames.shape[1],frames.shape[2]))
    if delay > 0:
        for x in np.arange(dims[1]):
            for y in np.arange(dims[2]):
                result[x,y] = np.mean(frames_cor[delay:,coords[0],coords[1]]*frames_cor[0:-delay,x,y])
    elif delay < 0:
        for x in np.arange(dims[1]):
            for y in np.arange(dims[2]):
                result[x,y] = np.mean(frames_cor[0:delay,coords[0],coords[1]]*frames_cor[-delay:,x,y])
    else:
        for x in np.arange(dims[1]):
            for y in np.arange(dims[2]):
                result[x,y] = np.mean(frames_cor[:,coords[0],coords[1]]*frames_cor[:,x,y])

    return result
    
    
def correlation_flow_map(frames,xstride=5,ystride=5,tstride=1,xref=None,yref=None):
    
    if xref is None:
        xref = np.arange(frames.shape[1])[:,np.newaxis]*np.ones(frames.shape[2])
    if yref is None:
        yref = np.arange(frames.shape[2])[np.newaxis,:]*np.ones(frames.shape[1])
    
    
    inds_x = np.arange(frames.shape[1])[::xstride]
    inds_y = np.arange(frames.shape[2])[::ystride]
    
    vx = np.zeros((inds_x.shape[0],inds_y.shape[0]))
    vy = np.zeros((inds_x.shape[0],inds_y.shape[0]))
    
    for i,x in enumerate(inds_x):
        for j,y in enumerate(inds_y):
            C_pp = cross_correlation(frames,(x,y),delay=tstride)
            try:
                inds_max = np.where(C_pp == C_pp.max())
            
                vx[i,j] = xref[x,y] - xref[inds_max[0],inds_max[1]]  
                vy[i,j] = yref[x,y] - yref[inds_max[0],inds_max[1]]  
            except:
                pass
            
            
    return vx, vy
            
            
            
            
            
            
            
            
            
    
    