import sys
import os
import numpy as np
from scipy.optimize import minimize, basinhopping, fmin_l_bfgs_b
import jscatter as js
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import finufft
import time
import hickle as hkl
from hyperalg.utils import export_structure, omp_get_thread_count
import skimage.transform as sktf
#from SkPotential import Sk_potential
#import multiprocessing
#from hull_utils import origin_in_hull_2d
from hyperalg.potentials import WCAPotential, InversePower, ExplicitSkPotential#CCPotential, HertzianPotential, InversePower, WCAPotential, ExplicitCCPotential, ExplicitSkPotential

def log_phi(x):
    """
    callback function
    """
    global counter
    #traj_x.append(x)
    phi_values_wca.append(phi_wca)
    grad_values_wca.append(np.linalg.norm(grad_wca))
    phi_values_sk.append(phi_sk)
    grad_values_sk.append(np.linalg.norm(grad_sk))

    if counter%100 == 0:
        print(counter)
    #    #print(phi_sk+phi_gr)
        print(phi_sk)
        print(phi_wca)
    #f len(phi_values_cc) == 400:
    #    np.savetxt('mixed_profile_cpp.dat',np.vstack([phi_values_cc,phi_values_hz]))
    counter += 1
def main(jam, N=100, K=256, ndim=2, n=1,size_ratio = 1.0, use_points=False):
    '''Generates n Hyperuniform point patterns given a box size and number density'''

    global phi_values_sk, phi_values_wca, grad_values_sk, grad_values_wca
    global traj_x, traj_grad
    global counter
    counter = 0
    traj_x = []
    traj_grad = []
    # Assume unit equilateral box
    L = np.ones(ndim)


    times = 0

    # Choose particle radius at jamming transition
    # Assume bidisperse in 1:1 number ratio
    #rad = np.sqrt(jam*np.prod(L)/(N*np.pi))
    if ndim == 2:
        rad_min = np.sqrt(jam*np.prod(L)/(0.5*N*np.pi*(1+size_ratio**2)))
        radii = rad_min*np.ones(N)
        radii[int(N/2):] *= size_ratio
    elif ndim == 3:
        rad_min = (jam*np.prod(L)/(2/3*N*np.pi*(1+size_ratio**3)))**(1/3)
        radii = rad_min*np.ones(N)
        radii[int(N/2):] *= size_ratio
    print('rmin:' + str(radii[0]))
    print('rmax:' + str(radii[-1]))
    np.random.shuffle(radii)
    a = float(sys.argv[1])
    i = int(sys.argv[2])
    Kint = int(K)
    Kvec = np.arange(-Kint,Kint+1).reshape((1,-1))
    Nk = Kvec.shape[-1]

    if ndim == 2:
        Kmag = np.sqrt(Kvec**2 + (Kvec**2).T)
    elif ndim == 3:
        Kmag = np.sqrt(Kvec.reshape((1,1,-1))**2 + Kvec.reshape((1,-1,1))**2 + Kvec.reshape((-1,1,1))**2)
    Kmask = np.array(Kmag <= K,dtype=np.int32)
    err_mode = 1
    if a == 0:
        Sk = np.zeros(Kmag.shape)
        #Kmask = np.ones(Kmag.shape)
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Stealthy Structure Factor')
        err_mode = 0
    elif a ==-1:
        Sk = 0.5*(1-np.cos(Kmag*5*np.pi/K))
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Cosine Structure Factor')
    elif a == -2:
        Sk = np.zeros(Kmag.shape)
        ellipse = (Kvec/K)**2 + ((2*Kvec/K)**2).T
        Kmask = np.array(ellipse <= 1,dtype=np.int32)
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Ellipse Stealthy Structure Factor')
        err_mode = 0
    elif a == -3:
        Sk = np.zeros(Kmag.shape)
        center = int(Kmask.shape[0]/2)
        for x in range(Kmask.shape[0]):
            for y in range(Kmask.shape[1]):
                if Kmask[x,y] == 1 and Kmag[x,y]>K*(np.sin(5*np.arctan2(y-center,x-center))+1)/2:
                    #Kmask[x,y] = 0
                    Sk[x,y] = 1
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Rose Stealthy Structure Factor')
        err_mode = 0
    elif a == -4:
        Sk = np.zeros(Kmag.shape)
        Kmask = np.ones(Kmag.shape)
        box = int(Kmask.shape[0]/7)
        for x in range(Kmask.shape[0]):
            for y in range(Kmask.shape[1]):
                if (int(x/box)+int(y/box))%2==1:
                    #Kmask[x,y] = 0
                    Sk[x,y] = 1
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Checkerboard Structure Factor')
        err_mode = 0
    elif a == -5:
        Sk = np.zeros(Kmag.shape)
        Kmask*=0
        center = int(Kmask.shape[0]/2)
        for x in range(Kmask.shape[0]):
            for y in range(Kmask.shape[1]):
                theta = np.arctan2(y-center,x-center)
                for t in range(4):
                    if Kmask[x,y] == 0 and np.abs(Kmag[x,y]-12*(theta+2*np.pi*t))<=10:
                        Kmask[x,y] = 1
                        #Sk[x,y] = 1
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Spiral Stealthy Structure Factor')
        err_mode = 0
    elif int(a) == -6:
        Nq = int((a+6)*-10)
        print(Nq)
        tau = (1+np.sqrt(Nq))/2
        gwidth = K/18
        gaussian = np.exp(-(Kmag**2/(2*gwidth**2)))#+0j
        smart = mplimg.imread('Martiniani_tiny.jpg')
        smart = np.mean(smart,axis=-1)/255
        padlength = Kmag.shape[0]-smart.shape[0]
        padlength = int(padlength/2)
        #smart = np.pad(smart,((padlength,padlength),(padlength,padlength)),mode='constant',constant_values=0)
        #gaussian = np.fft.fftshift(np.fft.fft2((smart*gaussian),Kmag.shape))
        Sk = np.zeros(Kmag.shape)
        center = int(Kmask.shape[0]/2)
        pspan = 1
        p = np.arange(-pspan,pspan+1).reshape(1,-1)
        kfactors = 2*np.pi/np.sqrt(Nq)*(p+p.T/tau)
        X = np.pi*2j/np.sqrt(Nq)*(p.T*tau - p)
        X = X.reshape(-1,1)
        if ndim == 2:
            espan = np.arange(Nq)
            evecs = np.array([np.cos(espan*2*np.pi/Nq),np.sin(espan*2*np.pi/Nq)]).T
            kvecs = None
            peaks = None
            for x in espan:
                for y in espan[(x+1):]:
                    ei = evecs[x]
                    ej = evecs[y]
 
                    kvec =  kfactors.reshape(1,-1,1)*ei.reshape(1,1,-1) + kfactors.reshape(-1,1,1)*ej.reshape(1,1,-1)
                    kvec = kvec.reshape(-1,2)
                    vol = np.dot(ei,np.flip(ej)*np.array([-1,1]))
                    f1ij = np.sinc(X/2)*np.exp(1j*np.dot(kvec,np.flip(ej)*np.array([-1,1]))/vol)
                    f1ji = np.sinc(X/2)*np.exp(1j*np.dot(kvec,np.flip(ei)*np.array([-1,1]))/vol)
                    peak = np.sum(np.absolute(f1ij*f1ji)**2,axis=0)
                    if kvecs is None:
                        kvecs = kvec
                        peaks = peak
                    else:
                        kvecs = np.vstack([kvecs,kvec])
                        peaks = np.hstack([peaks,peak])
            kvecs *= Kmag.shape[0]/4
            kmags = np.linalg.norm(kvecs,axis=-1)
            cut = np.nonzero(kmags<K)
            kvecs = kvecs[cut]
            peaks = peaks[cut]
            for kvec,peak in zip(kvecs,peaks):
                pwidth = 65
                theta = np.arctan2(kvec[1],kvec[0])*180/np.pi+180
                ref = center-int(pwidth/2)
                x,y = np.round(kvec).astype(int) + ref
                img = sktf.rotate(smart,theta)
                Sk[x:x+pwidth,y:y+pwidth] += img*peak*gaussian[ref:ref+pwidth,ref:ref+pwidth]
                
                
                
            #peaks = np.where(peaks>np.mean(peaks),peaks,0)
            '''kvecs *= 2*np.pi/Kmag.shape[0]
            real = finufft.nufft2d1(kvecs[:,0],kvecs[:,1],peaks+0j,n_modes=Kmag.shape)
            Sk = np.fft.ifft2(((real/np.amax(real))*gaussian),Kmag.shape)
            Sk *= N
            Sk = np.absolute(Sk)'''
            Sk /= np.amax(Sk)
            #Kmask = Kmask*np.where(Sk>1,0, 1)
        elif ndim == 3:
            ei = np.array([0,tau,1])/np.sqrt(1+tau*tau)
            ej = np.array([tau,1,0])/np.sqrt(1+tau*tau)
            ek = np.array([1,0,tau,1])/np.sqrt(1+tau*tau)
            vol = np.dot(ei,np.cross(ej,ek))
            kvecs =  kfactors.reshape(-1,1,1,1)*ei.reshape(1,1,1,-1) + kfactors.reshape(1,-1,1,1)*ej.reshape(1,1,1,-1) + kfactors.reshape(1,1,-1,1)*ek.reshape(1,1,1,-1)
            kvecs = kvecs.reshape(-1,3)
            f1ijk = np.sinc(X/2)*np.exp(1j*np.dot(kvecs,np.cross(ej,ek))/vol)
            f1jki = np.sinc(X/2)*np.exp(1j*np.dot(kvecs,np.cross(ek,ei))/vol)
            f1kij = np.sinc(X/2)*np.exp(1j*np.dot(kvecs,np.cross(ei,ej))/vol)
            peaks = np.sum(np.absolute(f1ijk*f1jki*f1kij)**2,axis=0)
            kvecs *= 2*np.pi/K
            real = finufft.nufft2d1(kvecs[:,0],kvecs[:,1],kvecs[:,2],peaks+0j,n_modes=Kmag.shape)
            Sk = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(real/np.amax(real)*gaussian),Kmag.shape))
        for x in range(Kmask.shape[0]):
            for y in range(Kmask.shape[1]):
                theta = np.arctan2(y-center,x-center)+np.pi/2
                if Kmask[x,y] == 1 and y<center:#np.floor(10*theta/(2*np.pi))%2==0:
                    Kmask[x,y] = 0
                    #Sk[x,y] = 1
        
        Vk = Kmask*Sk#/(1+np.exp(-(K/2-Kmag)))
        #Sk*=0
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask*Sk
        hkl.dump([Sk,kvecs],'init_SK.hkl')
        fig = plt.figure()
        ax = plt.gca()
        ax.imshow(Sk,vmin=0,vmax=1)
        plt.savefig('test.png')
        print('Quasicrystal Structure Factor')
        err_mode = 0
    elif a == -7:
        img = mplimg.imread('Martiniani_small.jpg')
        img = np.mean(img,axis=-1)/256
        img2 = mplimg.imread('Kasiulis_small.jpg')
        img2 = np.mean(img2,axis=-1)/256
        Sk = np.zeros(Kmag.shape)
        Kmask = np.zeros(Kmag.shape)
        center = int(Kmask.shape[0]/2)
        Kmask[center+1:,:] = 1
        Sk[center+1:,center+1:] = img
        Sk[center+1:,0:center] = img2
        print('Martiniani Structure Factor')
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        err_mode = 0
    elif int(a) == -8:
        Nq = int((a+8)*-100)
        print(Nq)
        gwidth = K/16
        gaussian = np.exp(-(Kmag**2/(2*gwidth**2)))#+0j
        #gaussian = np.fft.fftshift(np.fft.fft2((smart*gaussian),Kmag.shape))
        Sk = np.zeros(Kmag.shape)
        center = int(Kmask.shape[0]/2)
        kvecs = np.random.random((Nq,ndim))*np.pi*np.array([2,1])
        if ndim == 2:
            real = finufft.nufft2d1(kvecs[:,0],kvecs[:,1],np.ones(Nq)+0j,n_modes=Kmag.shape)
            Sk = np.fft.ifft2(((real/np.amax(real))*gaussian),Kmag.shape)
            Sk *= Nq
            Sk = np.absolute(Sk)
            Sk /= np.amax(Sk)
            #Kmask = Kmask*np.where(Sk>1,0, 1)
        elif ndim == 3:
            real = finufft.nufft2d1(kvecs[:,0],kvecs[:,1],kvecs[:,2],peaks+0j,n_modes=Kmag.shape)
            Sk = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(real/np.amax(real)*gaussian),Kmag.shape))
        for y in range(Kmask.shape[1]):
            if Kmask[center,y] == 1 and y<center:
                Kmask[:,y] = 0
        
        Vk = Kmask*Sk#/(1+np.exp(-(K/2-Kmag)))
        #Sk*=0
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask*Sk
        hkl.dump([Sk,kvecs],'init_SK.hkl')
        fig = plt.figure()
        ax = plt.gca()
        ax.imshow(Sk*Kmask*Kmask*Kmask*Kmask*Kmask*Kmask*Kmask*Kmask,vmin=0,vmax=1)
        plt.savefig('test.png')
        print('Anticrystal Structure Factor')
        err_mode = 0
    elif a ==-9:
        Sk = (Kmag <=0.75*K)*(Kmag >= 0.5*K)*1
        Vk = Kmask*(Kmag>=0.25*K)#/(1+np.exp(-(K/2-Kmag)))
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Bandgap Structure Factor')
    elif a == -10:
        Sk = np.zeros(Kmag.shape)
        Kmask*=0
        center = int(Kmask.shape[0]/2)
        for x in range(Kmask.shape[0]):
            for y in range(Kmask.shape[1]):
                theta = np.arctan2(y-center,x-center)
                if Kmask[x,y] == 0 and Kmag[x,y] > 1430 and Kmag[x,y] < 2500 and theta >=0 and theta < np.pi and np.abs((Kmag[x,y]-1430)*np.pi/(2500-1430)-theta)<=np.pi/12:
                    Kmask[x,y] = 1
                    #Sk[x,y] = 1
        Vk = Kmask#/(1+np.exp(-(K/2-Kmag)))
        #Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask
        print('Rainbow Spiral Structure Factor')
        print(np.sum(Kmask))
        err_mode = 0
    else:
        Sk = generate_HPY(jam,a,radii[0],Kmag)
        Vk = np.where(Kmag.ravel()>0,Kmag.ravel()**-ndim,0).reshape(Kmag.shape)*Kmask+Kmask
        print('Power Law Structure Factor')
    del Kmask, Kint, Kmag
    if not use_points:
        IP = InversePower(2.5, 1, radii, np.array(L), use_cell_lists=(N>=262144))
        #sig = 2*radii[0]/(2**(1/6))
        #eps = 1.0
        #WCA = WCAPotential(sig,eps, radii, np.array(L), use_cell_lists=True)
    center = int(Nk/2)
    if ndim == 2:
        Vk[center,center] = 0
    elif ndim ==3:
        Vk[center, center, center] = 0
    #SK = Sk_potential(Kvec,Vk,L,Sk)
    SK = ExplicitSkPotential(radii,K,Sk,Vk,L,err_mode = err_mode)
    jammed = 0
    phi_values_wca = []
    grad_values_wca = []
    phi_values_sk = []
    grad_values_sk = []
    np.random.seed(123+i)
    init = 'random'
    print('Loading initial condition')
    file_name = 'HPY'+str(ndim)+'D_phi'+str(jam)+'_a' + str(a) + '_N'+str(N)+'_K'+str(K)
    if use_points:
        file_name = file_name+'_points'
    elif size_ratio != 1.0:
        file_name = file_name+'_bidisperse'+str(size_ratio)


    if init == 'random':
        points = np.random.random(N*ndim)
    elif init == 'restart':
        data = hkl.load('/home/ahshih/Desktop/HPY'+str(ndim)+'D/phi'+str(jam)+'/a'+str(a)+'/'+file_name+'_'+str(i-1)+'.hkl')
        np.savetxt('temp.txt',data)
        data = np.loadtxt('temp.txt')
        points = data[:,0:ndim].ravel()
    elif init == 'rsa':
        points = rsa(N,radii[0],L)
    elif init == 'square':
        Nx = int(np.sqrt(N))
        x = np.linspace(0,1,Nx,endpoint=False)
        grid = np.zeros((Nx,Nx,2))
        grid[:,:,0]=x.reshape(-1,1)
        grid[:,:,1]=x.reshape(1,-1)
        points = grid.ravel()
    elif init == 'duplicate':
        factor = int(np.round((N/262144)**(1/ndim)))
        print(factor)
        data = np.loadtxt('/home/ahshih/Desktop/init_configs/hard'+str(int(ndim))+'d/N'+str(int(N/(factor**ndim)))+'/Phi'+f'{jam:.4f}'+'/'+str(np.char.zfill(str(i+1),3))+'/dump.txt')
        points = data[:,0:ndim]
        for y in range(ndim):
            dupes = np.array(points)
            for x in range(factor-1):
                vec = np.zeros(ndim).reshape(1,ndim)
                vec[0,y] = x+1
                dupes = np.vstack([dupes,points+vec])
            points = dupes
        points = points.ravel()/factor
        assert len(points)/ndim == N
    else:
        data = np.loadtxt('/home/ahshih/Desktop/init_configs/hard'+str(int(ndim))+'d/N'+str(int(N))+'/Phi'+f'{jam:.4f}'+'/'+str(np.char.zfill(str(i+1),3))+'/dump.txt')
        points = data[:,0:ndim].ravel()
        hkl.dump(np.hstack([points.reshape(-1,ndim), radii.reshape(-1,1)]),'init_HSL/init_HSL'+str(ndim)+'D_phi'+str(jam)+'_N'+str(N)+'_' + str(i) + '.hkl',mode='w')
    #file_name = 'HPY'+str(ndim)+'D_phi'+str(jam)+'_a' + str(a) + '_N'+str(N)+'_K256'
    #data = hkl.load('/home/ahshih/Desktop/HPY2D/phi'+str(jam)+'/a'+str(a)+'/'+file_name+'_'+str(i)+'.hkl')
    #points = data[0,:].ravel()
    assert points.shape[0] == N*ndim
    #print('Exporting Structure')
    #q,S,SK,rspan,g,GR = export_structure(points,L,Nk=400)
    #test_gradient(SK,points)
    if use_points:
        #res = minimize(Sk_potential, points, args=(Kvec, Vk, L, Sk,Kmask), method='L-BFGS-B', jac=True, options={'ftol': 1e-20,'gtol': 1e-20} ,callback=log_phi )
        res = minimize(mixed_potential, points, args=([SK,SK]), method='L-BFGS-B', jac=True, options={'ftol': 1e-20,'gtol': 1e-20} ,callback=log_phi )
    else:    
        res = minimize(mixed_potential, points, args=([SK,IP]), method='L-BFGS-B', jac=True, options={'ftol': 1e-20,'gtol': 1e-20} ,callback=log_phi )
        #points, fmin, d = fmin_l_bfgs_b(Sk_potential, points, args=(Kvec, Vk, L, Sk, Ux, gr0),factr=1,pgtol=1e-20, callback=log_phi )
        #res = basinhopping(Sk_potential, points,minimizer_kwargs={'method':'L-BFGS-B', 'args':(Kvec, Vk, L, Sk,Kmask),'jac':True},T=300,disp=True)
        #res = basinhopping(cc_potential, points, T=1.0, minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "args":(kspan, V0, L, C0, pair_dists), "callback":log_phi})
    points = res.x
    points %= L[0]
    '''traj_sk = []
    for r in traj_x:
        phi, grad = SK.get_energy_gradient(r)
        traj_grad.append(grad)
        q,S,_Sk,rspan,g,GR = export_structure(r, L,Nk)
        traj_sk.append(S)'''
    phi, grad = SK.get_energy_gradient(points)
    os.chdir('/home/ahshih/Desktop/HPY'+str(ndim)+'D/phi'+str(jam)+'/a'+str(a))
    hkl.dump(np.hstack([points.reshape(-1, ndim),radii.reshape(-1,1)]),file_name+'_'+str(i)+'.hkl',mode='w')
    #hkl.dump([traj_x,traj_grad],file_name+'_'+str(i)+'_traj.hkl',mode='w')
    q,S,SK,rspan,g,GR = export_structure(points, L,Nk)
    hkl.dump(np.array([q,S]),file_name+'_structure_' + str(i) + '.hkl',mode='w')
    #hkl.dump([q,traj_sk],file_name+'_structure_' + str(i) + '_traj.hkl',mode='w')
    hkl.dump(np.array([rspan,g]),file_name+'_rdf_' + str(i) + '.hkl',mode='w')
    hkl.dump(SK,file_name+'_SK_' + str(i) + '.hkl',mode='w')
    hkl.dump(GR,file_name+'_GR_' + str(i) + '.hkl',mode='w')
    print(phi) 
    print("phi: "+str(jam))
    print("N: "+str(N))
    print("a: "+str(a))
    print("K: "+str(K))
    print("Rad ratio: "+str(size_ratio))
    print("Points: "+str(use_points))

    print("Initial: "+str(phi_values_sk[0]+phi_values_wca[0]))
    print("Final:   "+str(phi_values_sk[-1]+phi_values_wca[-1]))
    print("Sk: "+str(phi_values_sk[-1]))
    print("Hz: "+str(phi_values_wca[-1]))
      
    #np.savetxt('mixed_profile.dat', np.vstack([phi_values_cc,phi_values_hz]))
    #np.savetxt('mixed_Hz_a' + str(a) + '_phi_' + str(i) + '.dat', np.vstack([phi_values_cc,phi_values_hz]))
    if i == 0:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.array(phi_values_sk)+np.array(phi_values_wca), label = 'Total')
        ax.plot(phi_values_sk, label = 'Sk')
        ax.plot(phi_values_wca, label= 'WCA')
        ax.legend()
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Energy')
        plt.savefig('HPY'+str(ndim)+'D_a' + str(a) + '_N'+str(N)+'_K'+str(K)+'_eval_' + str(i) + '.png')
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.array(grad_values_sk)+np.array(grad_values_wca), label = 'Total')
        ax.plot(grad_values_sk, label = 'Sk')
        ax.plot(grad_values_wca, label= 'WCA')
        ax.legend()
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient')
        #ax.set_xlim(1200,1400)
        #ax.set_ylim(0,3e-3)
        plt.savefig('HPY'+str(ndim)+'D_a' + str(a) + '_N'+str(N)+'_K'+str(K)+'_grad_' + str(i) + '.png')
        hkl.dump(np.array([phi_values_sk,phi_values_wca,grad_values_sk,grad_values_wca]),'HPY'+str(ndim)+'D_a' + str(a) + '_N'+str(N)+'_K'+str(K)+'_eval_' + str(i) + '.hkl')
        
    #print(str(jammed) + ' out of '+str(n)+' configurations are jammed')
    return times/n

def generate_HPY(phi, a, radius, Kmag):
    ndim = len(Kmag.shape)
    if ndim == 2:
        PY = js.sf.PercusYevick2D(Kmag.ravel()*2*np.pi,radius,eta=phi)[1]
    elif ndim == 3:
        PY = js.sf.PercusYevick(Kmag.ravel()*2*np.pi,radius,eta=phi)[1]
    idx = np.argsort(Kmag.ravel())
    slope = 0
    PYlog = np.log10(PY)
    x = 1
    while slope < a:
        front = idx[x+1]
        back  = idx[x-1]
        if PYlog[front]-PYlog[back] > 1e-5:
            slope = (PYlog[front]-PYlog[back])/(np.log10(Kmag.ravel()[front])-np.log10(Kmag.ravel()[back]))
        x += 1
    #print(Kmag.ravel()[idx[x]])
    D = PY[idx[x]]/(Kmag.ravel()[idx[x]]**a)
    Sk = np.where(Kmag.ravel() < Kmag.ravel()[idx[x]], D*(Kmag.ravel()**a), PY)
    return Sk.reshape(Kmag.shape)

def mixed_potential(r, potentials, weights=[1,10]):
    global phi_sk, phi_wca, grad_sk, grad_wca
    phi_step=0
    phi_grad = np.zeros(r.shape)
    phi_sk, grad_sk = potentials[0].get_energy_gradient(r)
    phi_wca, grad_wca = potentials[1].get_energy_gradient(r)
    grad_sk = np.asarray(grad_sk)
    grad_wca = np.asarray(grad_wca)
    #grad_sk /= np.linalg.norm(grad_sk)
    #grad_wca /= np.linalg.norm(grad_wca)
    #phi_grad = np.zeros(r.shape)
    '''for pot,wt in zip(potentials,weights):
        phi, grad = pot.get_energy_gradient(r)
        p)hi_step += phi
        phi_sk = phi
        phi_grad += np.array(grad)*wt'''
    #if len(phi_values_cc) < 100:
    #    np.savetxt('mixed_config_cpp_'+str(len(phi_values_cc))+'.dat',r)
    #    np.savetxt('mixed_grad_cpp_'+str(len(phi_values_cc))+'.dat',np.vstack([grad_cc,grad_hz]))
    phi_step = weights[0]*phi_sk+weights[1]*phi_wca
    phi_grad = weights[0]*grad_sk + weights[1]*grad_wca
    return phi_step,phi_grad

def test_gradient(pot, points):
    e, g = pot.get_energy_gradient(np.array(points))
    eps = 1e-6
    error = np.zeros(10)
    print("Calculating numerical gradient")
    for i in range(10):
        rdisp = np.array(points)
        rdisp[i] += eps
        plus = pot.get_energy(rdisp)
        rdisp[i] -= eps*2
        minus = pot.get_energy(rdisp)
        print(i)
        disp = (plus-minus)/(2*eps*2*np.pi)
        print('Analytical grad: '+ str(g[i]))
        #print((disp-e)/(eps*2*np.pi))
        print('Numerical grad: ' + str(disp))
        err=(g[i] - disp)/g[i]
        print('Error: '+str(err))
        error[i] += err**2
    print('Mean: '+str(np.mean(error)))
    print('Std Dev: '+str(np.std(error)))
    print('Max: '+str(np.amax(error)))
    print('Min: '+str(np.amin(error)))
def tri(Nx,Ny,L):
    x = []
    y = []
    dx = L[0]/Nx
    dy = L[1]/Ny
    dy = dx*np.sqrt(3)/2
    row = np.arange(Nx)*dx
    col = np.zeros(Nx)
    for i in range(Ny):
        x += [row+0.5*dx*(i%2)]
        y += [col+i*dy]
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    points = (np.stack([x,y]).T.flatten()+0.01)%1.
    #hkl.dump(points,'triangular.hkl')
    return points

def rsa(N, radius, L):
    ndim =len(L)
    r2 = 4*radius**2
    points = []
    while len(points) < N:
        point = np.random.rand(ndim)*np.array(L)
        add = True
        for p in points:
            dr = (p-point)
            dr = np.where(np.abs(dr)<L[0]/2,dr, dr-np.sign(dr)*L[0])
            if np.sum(dr**2) < r2:
                add = False
                break
        if add:
            points += [point]
    return np.array(points).flatten()

if __name__ == '__main__':
    print(omp_get_thread_count())
    phirange = [0.25,0.5,0.6,0.68,0.7,0.72]
    #phirange = [0.25,0.4, 0.48,0.6] 
    phirange = [0.6]
    #threads.map(main,phirange)
    krange = [3000]
    #krange = [24]
    cwd = os.getcwd()
    for phi in phirange:
        for k in krange:
            os.chdir(cwd)
            main(phi,N=10000000,K=k,ndim=2,size_ratio=1.0, use_points=True)
    sys.exit()
