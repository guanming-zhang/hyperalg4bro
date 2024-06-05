import os
import json
import numpy as np
import finufft
import sys
from numba import jit
import Utils
sys.path.append('../../hyperalg')
from hyperalg import potentials
from hyperalg.distances import Distance

def calculate_activity(dl:Utils.DataLoader):
    '''
    calculate the activity = the number of active particles / the totoal number of particles
    '''
    _,na = dl.get_data_list('n_active')
    if len(na) > 10:
        na_last = np.mean(na[-10:])
        fa = na_last/dl.info["N"]
    else:
        na_last = np.array(na[-1])
        fa = na_last/dl.info["N"]
    return fa

def calculate_energy(dl:Utils.DataLoader):
    _,e = dl.get_data_list('total_energy')
    if len(e) > 10: 
        e_last = np.array(e[-10:0])
    else:
        e_last = np.array(e[-1])
    return e_last


def calculate_number_of_nbrs(dl:Utils.DataLoader):
    '''
    calculate the number of neighbour for each particle
    rtype: numpy array of length N
    '''
    _,x = dl.get_data_list('x')
    x_last = np.array(x[-1])
    pot = potentials.InversePower(a=1.0,eps=1.0,boxv=np.array(dl.info['box_size']),
                                  radii = np.full(dl.info['N'],dl.info["r"]),ncellx_scale = None, 
                                  enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    inbr,r_d = pot.getNeighbors(x_last)
    n_nbrs = np.zeros(dl.info['N'],dtype=np.int32)
    for iatom in range(dl.info['N']):
        n_nbrs[iatom] = len(inbr[iatom])
    return n_nbrs


def _pos_perturb(pot,x,sigma,n_repeat=1000):
    e_ref = pot.get_energy(x)
    energies = []
    for _ in range(n_repeat):
        noise = np.random.normal(scale=sigma,size = x.shape)
        y = x + noise
        energies.append(pot.get_energy(y) - e_ref)
    dE = np.mean(energies)
    std_dE = np.std(energies)
    return dE,std_dE   

def calculate_dE(dl:Utils.DataLoader,sigma=0.01,n_repeat=2000):
    '''
    perturb the flattened position vector of size d*N
    and calculate the average energy change
    '''
    _,x = dl.get_data_list('x')
    x_last = np.array(x[-1])
    if dl.info['potential_type'] == "linear":
        pot = potentials.InversePower(a=1.0,eps=1.0,boxv=np.array(dl.info['box_size']),
                                  radii = np.full(dl.info['N'],dl.info["r"]),ncellx_scale = None, 
                                  enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    else:
        raise NotImplementedError("potentail not implemented")
    sigma *= dl.info["r"]
    dE,std_dE = _pos_perturb(pot,x_last,sigma=sigma,n_repeat=n_repeat)
    # make dE independent of particle size
    dE /= sigma*sigma
    std_dE /= sigma*sigma
    return dE,std_dE

def calculate_relaxation_time(data_dir):
    '''
    input the data_dir
    return tau
    '''
    f = os.path.join(data_dir,'data_time.json')
    if not os.path.isfile(f):
        raise FileNotFoundError("data_time.json file not found in \n" + f)
    with open(f,'r') as fs:
        data = json.loads(fs.read())
    phi = data['phi']
    t = np.array(data['t'])
    fa = np.array(data['f_active'])
    if fa[-1] < 1e-6:
        f_inf = 0.0
    else:
        f_inf = np.mean(fa[-20:])
    tau = -1
    for i in range(len(t)):
        if f_inf < 1e-6 and fa[i] < 1e-6:
            tau = t[i]
            break
        elif f_inf > 1e-6 and abs(fa[i]/f_inf - 1.0) < 0.10:# 001:
            tau = t[i]
            break
    if tau < 0:
        raise ValueError("the correct relaxiation time is not fount for file:\n" + data_dir)
    return tau


def calculate_strucutre_factor(dl:Utils.DataLoader,N_res=512):
    # x: numpy array for particle positions
    # N: resolvsion in k-space
    #    the output is a N^dim array
    _,x = dl.get_data_list('x')
    x_last = np.array(x[-1])
    x = np.reshape(x_last,(-1,dl.info["dim"]))
    if dl.info["dim"] == 2:
        x1 = (x[:,0]/dl.info["box_size"][0] - 0.5)*2.0*np.pi 
        x2 = (x[:,1]/dl.info["box_size"][1] - 0.5)*2.0*np.pi
        c = np.ones(dl.info["N"]).astype(np.complex128)
        f = finufft.nufft2d1(x1, x2, c, (N_res, N_res),modeord=0)
        s = np.abs(f*np.conjugate(f))/N_res
        kx = np.fft.fftshift(np.fft.fftfreq(N_res)*N_res*2*np.pi/dl.info["box_size"][0])
        ky = np.fft.fftshift(np.fft.fftfreq(N_res)*N_res*2*np.pi/dl.info["box_size"][1])
        return (kx,ky,s)
    elif dl.info["dim"] == 3:
        x1 = (x[:,0]/dl.info["box_size"][0] - 0.5)*2.0*np.pi 
        x2 = (x[:,1]/dl.info["box_size"][1] - 0.5)*2.0*np.pi
        x3 = (x[:,2]/dl.info["box_size"][2] - 0.5)*2.0*np.pi
        c = np.ones(dl.info["N"]).astype(np.complex128)
        f = finufft.nufft3d1(x1, x2, x3, c, (N_res, N_res, N_res),modeord=0)
        s = np.abs(f*np.conjugate(f))/N_res
        kx = np.fft.fftshift(np.fft.fftfreq(N_res)*N_res*2*np.pi/dl.info["box_size"][0])
        ky = np.fft.fftshift(np.fft.fftfreq(N_res)*N_res*2*np.pi/dl.info["box_size"][1])
        kz = np.fft.fftshift(np.fft.fftfreq(N_res)*N_res*2*np.pi/dl.info["box_size"][2])
        return (kx,ky,kz,s)

@jit(nopython=True)
def calculate_radial_structure_factor2d(kx,ky,s,n_bins=800,cut_off = 0.6):
    k_max = np.sqrt(kx[-1]**2 + ky[-1]**2)*cut_off 
    s_k = np.zeros(n_bins)
    dA = (kx[2]-kx[1])*(ky[2]-ky[1])
    delta_k = k_max/n_bins
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j])
            bin_num = int(k/delta_k) 
            if bin_num < n_bins:
                s_k[bin_num] += s[i,j]*dA
    k = np.arange(n_bins)*delta_k + 0.5*delta_k
    s_k /= 2.0*np.pi*k*delta_k
    return k,s_k

@jit(nopython=True)
def calculate_radial_structure_factor3d(kx,ky,kz,s,n_bins=100,cut_off = 0.6):
    k_max = np.sqrt(kx[-1]**2 + ky[-1]**2 + kz[-1]**2)*cut_off 
    s_k = np.zeros(n_bins)
    dV = (kx[2]-kx[1])*(ky[2]-ky[1])*(kz[2]-kz[1])
    delta_k = k_max/n_bins
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for l in range(s.shape[2]):
                k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l])
                bin_num = int(k/delta_k)
                if bin_num < n_bins:
                    s_k[bin_num] += s[i,j,l]*dV
    k = np.arange(n_bins)*delta_k + 0.5*delta_k
    s_k /= 4.0*np.pi*k*k*delta_k
    return k[1:],s_k[1:]/s_k[-1]

        
