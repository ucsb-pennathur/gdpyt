import numpy as np
import imageio
import os 
import sys
#import time as tm

from tkinter import filedialog


#import warnings
#warnings.filterwarnings("ignore")

def generate_synthetic_images(settings_file, txt_folder, destination_folder):

    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    mic = {}
    with open(settings_file) as f:
        for line in f:
            words = line.split('=')
            mic[words[0].strip()] = eval(words[1].strip())
        mic['pixel_dim_x'] = int(mic['pixel_dim_x'])
        mic['pixel_dim_y'] = int(mic['pixel_dim_y'])
        mic['n_rays'] = int(mic['n_rays'])

    data_files = [os.path.join(txt_folder, fname) for fname in os.listdir(txt_folder) if fname.endswith('.txt')]
    #%%
    #ii = 0
    #ii_tot = len(data_files)

    for data in data_files:
        #ii = ii+1
        # print('creating image {0} of {1} ...'.format(ii,ii_tot))
        P = np.genfromtxt(data)
        if len(P.shape)==1:
            P = np.array([P])
       
        head, tail = os.path.split(data)
        I = take_image(mic,P)
        imageio.imwrite(os.path.join(destination_folder,(tail[:-3] + 'tif')),
               np.uint16(I))
        
    print('done!')

#%%

def take_image(mic,P):
    
    # NOTE: x and xp represent here light fields and should not be confused$
    # with particle image coordinates which are represented by P
    

    I = np.zeros((mic['pixel_dim_y'],mic['pixel_dim_x']));
       
    dp_s = np.unique(P[:,3])
    if P.shape[1]==5 or P.shape[1]==8:
        k_id = P[:,-1]
    else:
        k_id = np.ones(P.shape[0])
        
    
    if P.shape[1]<=5 and dp_s.size==1:
    
        n_points = int(np.round(mic['points_per_pixel']*2*np.pi*
                (dp_s*mic['magnification']/ mic['pixel_size'])**2))             
        xp = create_particle(dp_s,n_points,mic['n_rays'])  
        
        for ii in range(0,P.shape[0]):
            Id = image_spherical(mic,xp,P[ii,0:3])
            I = I + Id*k_id[ii]
            
    elif P.shape[1]<=5 and dp_s.size!=1:
        
        for ii in range(0,P.shape[0]):
             n_points = int(np.round(mic['points_per_pixel']*2*np.pi*
                (P[ii,3]*mic['magnification']/ mic['pixel_size'])**2))             
             xp = create_particle(P[ii,3],n_points,mic['n_rays'])  
        
             Id = image_spherical(mic,xp,P[ii,0:3])
             I = I + Id*k_id[ii]
        
      
    elif P.shape[1]>=7:
        
        for ii in range(0,P.shape[0]):      
            n_points = int(np.round(mic['points_per_pixel']*2*np.pi*
                (dp_s*mic['magnification']/ mic['pixel_size'])**2)) 
        
            ecc = P[ii,4]
            if ecc>1:
                # area elipsoid/area sphere
                fact = 1/2*(1+ecc/np.sqrt(1-1/ecc**2)
                        *np.arcsin(np.sqrt(1-1/ecc**2)))
                n_points = int(np.round(fact*n_points))
            elif ecc<1:
                # area elipsoid/area sphere
                fact = 1/2*(1+ecc^2/np.sqrt(1-ecc**2)
                        *np.arctan(np.sqrt(1-ecc**2))) 
                n_points = int(np.round(fact*n_points))
        
        
            xp = create_ellipsoid(P[ii,3:7],n_points,mic['n_rays'])       
            Id = image_spherical(mic,xp,P[ii,0:3]);
            I = I + Id*k_id[ii]
            
    I = I*mic['gain']

    if mic['background_mean']!=0:
        I = I+mic['background_mean'] 

    if mic['background_noise']!=0:
        Irand = np.random.normal(0,mic['background_noise'],
                (mic['pixel_dim_y'],mic['pixel_dim_x']))
        I = I+np.round(Irand)
#        I = np.round(I+random('norm',0,mic.background_noise,...
#            mic.pixel_dim_y,mic.pixel_dim_x));

    return I


#%%
def image_spherical(mic,xp,P1):
                 
    # take image of a particle with a spherical lens
    # NOTE: x and xp represent here light fields and should not be confused$
    # with particle image coordinates which are represented by P1

    lens_radius = (np.tan(np.arcsin(mic['numerical_aperture']))
                   *(1+1/mic['magnification'])*mic['focal_length'])
    # distance lens-ccd
    dCCD = -mic['focal_length']*(mic['magnification']+1);
    # distance particle-lens
    dPART = P1[2]+mic['focal_length']*(1/mic['magnification']+1);
    
    # linear transformation from the object plane to the lens plane
    T2 = np.array([[1,  0,  dPART, 0    ],
                    [0,  1,  0,     dPART],
                    [0,  0,  1,     0    ],
                    [0,  0,  0,     1    ]])
      
    # light field right before the lens
    x = np.linalg.inv(T2)@xp
    
    # remove rays outside of the lens aperture
    ind = x[0,:]**2+x[1,:]**2<=lens_radius**2
    x = x[:,ind]
    
    # transformation of the light field with spherical lens
    a = x[0,:];  b = x[1,:]
    c = x[2,:];  d = x[3,:]
    # radius of curvature of the lens
    rk = mic['focal_length']*(mic['ri_lens']/mic['ri_medium']-1)*2
    dum = a*0
    # refraction medium-lens
    # ray-vector befor lens
    Vr = np.vstack((1+dum, c, d))
    Vr = (Vr/np.tile(np.sqrt(sum(Vr**2)),(3,1)))
    # normal-vector to the lens surface
    Vl = np.vstack((rk+dum, a, b))
    Vl = (Vl/np.tile(np.sqrt(sum(Vl**2)),(3,1)))
    # tangent-vector to the lens surface
    Vrot = np.cross(Vr,Vl,axisa=0, axisb=0)
    Vrot = np.cross(Vrot,Vl,axisa=1, axisb=0).transpose()
    Vrot = Vrot/np.tile(np.sqrt(sum(Vrot**2)),(3,1))
    # angle after snell-law correction
    vx = np.sum(Vr*Vl,axis=0) # dot product!
    vy = np.sum(Vr*Vrot,axis=0) # dot product!
    th11 = np.arcsin(mic['ri_medium']/mic['ri_lens']*
                     np.sin(np.arctan(vy/vx)))
    # new ray-vector inside the lens
    Vr11 = (Vl*np.tile(np.cos(th11),(3,1))+
            Vrot*np.tile(np.sin(th11),(3,1)))
    Vr = Vr11/np.tile(Vr11[0,:],(3,1))
    # refraction lens-medium
    # normal-vector to the lens surface
    Vl2 = np.vstack((Vl[0,:], -Vl[1:,:]))
    # tangent-vector to the lens surface
    Vrot = np.cross(Vr,Vl2,axisa=0, axisb=0)
    Vrot = np.cross(Vrot,Vl2,axisa=1, axisb=0).transpose()
    Vrot = Vrot/np.tile(np.sqrt(sum(Vrot**2)),(3,1))
    # angle after snell-law correction
    vx = np.sum(Vr*Vl2,axis=0) # dot product!
    vy = np.sum(Vr*Vrot,axis=0) # dot product!
       
    th11 = np.arcsin(mic['ri_lens']/mic['ri_medium']*
                     np.sin(np.arctan(vy/vx)))
    # new ray-vector outside the lens
    Vr11 = (Vl2*np.tile(np.cos(th11),(3,1))+
            Vrot*np.tile(np.sin(th11),(3,1)))
    Vr = Vr11/np.tile(Vr11[0,:],(3,1))
    # light field after the spherical lens
    x[2,:] = Vr[1,:]
    x[3,:] = Vr[2,:]
    
    if mic['cyl_focal_length']==0:
    #     linear transformation from the lens plane to the ccd plane
        T1 = np.array([[1,  0,  -dCCD,  0   ],
                        [0,  1,  0,    -dCCD],
                        [0,  0,  1,     0   ],
                        [0,  0,  0,     1   ]])
     #     light field at the ccd plane
        xs = np.linalg.inv(T1)@x
    else:
#    #     linear transformation from the lens plane to the cyl_lens plane
        T1c = np.array([[1, 0, -dCCD*1/3, 0        ], 
                        [0, 1, 0,         -dCCD*1/3],
                        [0, 0, 1,         0],
                        [0, 0, 0,         1]])
#    #     light field at the cylindrical lens plane
        xc = np.linalg.inv(T1c)@x
#    #     light field after the cylindrical lens plane
        Tc = np.array([[1,                          0, 0, 0], 
                       [0,                          1, 0, 0],
                       [-1/mic['cyl_focal_length'], 0, 1, 0], 
                       [0,                          0, 0, 1]])
        xc_a = np.linalg.inv(Tc)@xc
#    #     light field at the ccd plane
        T1 = np.array([[1, 0, -dCCD*2/3, 0        ],
                       [0, 1, 0,         -dCCD*2/3],
                       [0, 0, 1,         0        ],
                       [0, 0, 0,         1        ]]);
#    #     light field at the ccd plane
        xs = np.linalg.inv(T1)@xc_a


    
    # transform the position in pixel units
    X = np.round(xs[0,:]/mic['pixel_size']+P1[0])
    Y = np.round(xs[1,:]/mic['pixel_size']+P1[1])
    
    # remove rays outside the CCD
    ind = np.all([X>0,X<=mic['pixel_dim_x'],Y>0,Y<=mic['pixel_dim_y'],
                  X.imag==0,Y.imag==0], axis=0)
    
    # count number of rays in each pixel
    countXY = np.sort(Y[ind] + (X[ind]-1)*mic['pixel_dim_y'])
    indi, ia = np.unique(countXY, return_index=True)
    nCounts = np.hstack((ia[1:],countXY.size+1))-ia
    
    # prepare image
    I = np.zeros((mic['pixel_dim_y'],mic['pixel_dim_x']))
    Ifr= I.flatten('F')
    Ifr[indi.astype(int)-1] = nCounts
    I = Ifr.reshape(mic['pixel_dim_y'],mic['pixel_dim_x'], order='F')

    return I

#%%
def create_particle(D,Ns,Nr):

    R = D/2
    
    V = spiral_sphere(Ns)
    V[0:2,V[0,:]>0] = -V[0:2,V[0,:]>0]
    x = R*V[0,:]
    y = R*V[1,:]
    z = R*V[2,:]
    
    V0 = spiral_sphere(Nr+2)
    V0 = V0[:,1:-1]
    u = np.tile(x,(Nr,1))
    v = np.tile(y,(Nr,1))
    s = u*0
    t = u*0
    
    phs = np.random.uniform(-np.pi,np.pi,z.size)
    cs = np.cos(phs)
    sn = np.sin(phs)
    for k in range(0,Ns):
        Rot = np.array([[cs[k],-sn[k],0],
                      [sn[k],cs[k],0],[0,0,1]]) 
        Vr = Rot@V0
        Vr[0,:] = -abs(Vr[0,:])
        s[:,k] = Vr[1,:]/Vr[0,:]
        t[:,k] = Vr[2,:]/Vr[0,:]
        u[:,k] = y[k]-s[:,k]*x[k]
        v[:,k] = z[k]-t[:,k]*x[k]

    xp = np.vstack((u.flatten('F'), v.flatten('F'), 
                s.flatten('F'), t.flatten('F')))
    
          
    return xp


#%%
def create_ellipsoid(Deab,Ns,Nr):

    D = Deab[0]; ecc = Deab[1] 
    alpha = Deab[2]; beta = Deab[3]
        
    R = D/2        
        
    V = spiral_sphere(Ns)
    V = R*V
    V[2,:] = V[2,:]*ecc 
    
    R_beta = np.array([[np.cos(beta),  0, np.sin(beta)],
                       [0,             1, 0           ],
                       [-np.sin(beta), 0, np.cos(beta)]])
    R_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha),  0],
                        [0,             0,              1]])
    
    Vf = R_alpha@(R_beta@V)
    
    ii1 = (Vf[1,:]==np.min(Vf[1,:])).nonzero()[0][0]
    ii2 = (Vf[1,:]==np.max(Vf[1,:])).nonzero()[0][0]
    ii3 = (Vf[2,:]==np.min(Vf[2,:])).nonzero()[0][0]
    ii4 = (Vf[2,:]==np.max(Vf[2,:])).nonzero()[0][0]
    Vdum = Vf[:,[ii1,ii2,ii3,ii4]]
    
    A = np.c_[Vdum[1,:], Vdum[2,:], np.ones(Vdum.shape[1])]
    C,_,_,_ = np.linalg.lstsq(A, Vdum[0,:], rcond=None) 
    V1dum =  C[0]*Vf[1,:] + C[1]*Vf[2,:] + C[2]
    ind = (Vf[0,:]-V1dum)<0
    x = Vf[0,ind]
    y = Vf[1,ind]
    z = Vf[2,ind]
    Ns = z.size
    
    V0 = spiral_sphere(Nr+2)
    V0 = V0[:,1:-1]
    u = np.tile(x,(Nr,1))
    v = np.tile(y,(Nr,1))
    s = u*0
    t = u*0
    
    phs = np.random.uniform(-np.pi,np.pi,z.size)
    cs = np.cos(phs)
    sn = np.sin(phs)
    for k in range(0,Ns):
        Rot = np.array([[cs[k],-sn[k],0],
                      [sn[k],cs[k],0],[0,0,1]]) 
        Vr = Rot@V0
        Vr[0,:] = -abs(Vr[0,:])
        s[:,k] = Vr[1,:]/Vr[0,:]
        t[:,k] = Vr[2,:]/Vr[0,:]
        u[:,k] = y[k]-s[:,k]*x[k]
        v[:,k] = z[k]-t[:,k]*x[k]

    xp = np.vstack((u.flatten('F'), v.flatten('F'), 
                s.flatten('F'), t.flatten('F')))
    
    return xp

#%%
def spiral_sphere(N):

    gr = (1+np.sqrt(5))/2       # golden ratio
    ga = 2*np.pi*(1-1/gr)       # golden angle
    
    ind_p = np.arange(0,N)              # particle (i.e., point sample) index
    lat = np.arccos(1-2*ind_p/(N-1))  # latitude is defined so that particle index is proportional to surface area between 0 and lat
    lon = ind_p*ga               # position particles at even intervals along longitude
    
    # Convert from spherical to Cartesian co-ordinates
    x = np.sin(lat)*np.cos(lon)
    y = np.sin(lat)*np.sin(lon)
    z = np.cos(lat)
    V = np.vstack((x, y, z))
    
    return V

#%%