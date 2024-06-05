import sys
import os
sys.path.append('../hyperalg')
sys.path.append('./utils')
import numpy as np
#import torch as np
import matplotlib.pyplot as plt
import random
import subprocess
import json
from hyperalg import potentials
from hyperalg.distances import Distance
from scipy.optimize import minimize
import zipfile
import numba
from numba import prange
from math import sqrt
import Utils

loc = sys.argv[1]
input_file = os.path.join(loc,'info.json')
print(loc)
if os.path.isfile(input_file):
    with open(input_file) as fs:
        input_data = json.loads(fs.read())
    # particle configurations
    N = input_data['N'] # number of particles
    box_size = input_data['box_size']  # size of the box 
    boxv = np.array(box_size)
    phi = input_data['phi'] # packing fraction
    dim = input_data['dim']  # dimension
    if "potential_type" in input_data.keys():
        potential_type = input_data["potential_type"]
    else:
        potential_type = "n-def"
        a0 = 1.0
        mpow = 2.0
    if potential_type =="inv_pow_logi":
        k = input_data['k']
        a0 = input_data['a0'] # for inverse_power potential
        mpow = input_data['mpow'] # for inverse_power potential
    elif potential_type == "inverse_power":
        a0 = input_data['a0'] # for inverse_power potential
        mpow = input_data['mpow'] # for inverse_power potential
    if dim == 3:
        if len(box_size)!= 3:
            raise Exception('dim is not equal to len(box_size)')
        r = np.power(3.0/4.0*box_size[0]*box_size[1]*box_size[2]*phi/np.pi/N,1.0/3.0)
    elif dim == 2:
        if len(box_size)!= 2:
            raise Exception('dim is not equal to len(box_size)')
        r = np.power(box_size[0]*box_size[1]*phi/N/np.pi,1.0/2.0)
    if 'batch_lr' in input_data:
        batch_lr = input_data['batch_lr']*2.0*r # learning rate
    elif 'lr' in input_data:
        lr = input_data['lr']*2.0*r
        sqrt_lr = np.sqrt(lr)
    if 'x' in input_data:
        x = np.array(input_data['x'],dtype=float)
    if input_data['init_config'] == 'random':
        x = np.zeros((N,dim))
        for d in range(dim):
            x[:,d] = np.random.uniform(0.0,box_size[d],N)
        x = x.reshape(N*dim)
    elif input_data['init_config'] == 'fixed':
        st0 = np.random.get_state() #store the random seed
        np.random.seed(314) # set random seed to a lucky number
        x = np.zeros((N,dim))
        np.random.seed(None)
        for d in range(dim):
            x[:,d] = np.random.uniform(0.0,box_size[d],N)
        x = x.reshape(N*dim)
        np.random.set_state(st0) #recover the old random seed
    elif input_data['init_config'] == 'appending':
        dl = Utils.DataLoader(loc,mode='last_existed')
        iters,x_list = dl.get_data_list('x')
        t0 = iters[-1]
        x = np.array(x_list[-1])

    # calculation configurations
    if "optimization_method" in input_data.keys():
        optimization_method = input_data["optimization_method"]
    else:
        optimization_method = "pair_sgd"
        input_data['optimization_method'] = "pair_sgd"

    if "batch_size" in input_data.keys():
        batch_size = input_data['batch_size'] # batch size(# of pairs)
        batch_mode = "by_size"
    if "fraction" in input_data.keys():
        fraction = input_data['fraction'] # batch size(=fraction* total # of pairs)
        batch_mode = "by_fraction"
    if "batch_size" in input_data.keys() and "fraction" in input_data.keys():
        raise Exception('only one of the two parameters, batch_size or fraction, can be set')

    if 'bro' in optimization_method:
        eps = input_data['eps']*2.0*r
    if optimization_method == 'ro' or optimization_method == 'ro_nbr':
        delta = input_data['delta']*2.0*r # in units of 2r
        sqrt_delta = np.sqrt(delta)
    elif optimization_method == 'sme_ro':
        delta = input_data['delta']*2.0*r
        dt = input_data['relative_dt']*delta
        sqrt_dt,sqrt_delta = np.sqrt(dt),np.sqrt(delta)
    elif optimization_method == 'sme_bro' or optimization_method == 'sme_bro_interaction':
        eps = input_data['eps']*2.0*r
        dt = input_data['relative_dt']*eps
        sqrt_dt,sqrt_eps = np.sqrt(dt),np.sqrt(eps)
    elif optimization_method == 'sme_bro_noise':
        eps = input_data['eps']*2.0*r
        M_grad = input_data['M_grad']
        D_noise = input_data['D_noise']
        D_brown = input_data['D_brown']
        dt = input_data['relative_dt']*eps
        sqrt_dt,sqrt_eps_noise,sqrt_eps_brown = np.sqrt(dt),np.sqrt(eps*D_noise),np.sqrt(eps*D_brown)
    elif optimization_method == 'sme_particle_sgd':
        eps = input_data['eps']*2.0*r
        dt = input_data['relative_dt']*eps
        sqrt_dt,sqrt_eps = np.sqrt(dt),np.sqrt(eps)
    if 'alpha' in input_data.keys():
        alpha = input_data['alpha']*2.0*r

    if 'normalized_n_iter' in input_data:
        n_iter = int(input_data['normalized_n_iter']/input_data['fraction'] +0.5)
    else:
        n_iter = input_data['n_iter']
        
    run_id = input_data['runID'] # run id
    n_save = input_data['n_save'] # save data every n_save iterations 
    if "n_record" in input_data.keys():
        n_record = input_data['n_record']
    else:
        n_record = -1
    iter_zoom = [-1,-1]
    if 'iter_zoom' in input_data:
        iter_zoom = input_data['iter_zoom']
        if 'zoom_rate' in input_data:
            zoom_rate = input_data['zoom_rate']
        else:
            exp_nsave = input_data['exp_nsave']
            force_save_at = 1
        
    if 'save_mode' in input_data:
        save_mode = input_data['save_mode']
    else:
        save_mode = 'full'
    
    # update the info.json
    input_data['last_iteration'] = n_iter #calcualtion stops at
    input_data['r'] = r
    
else:
    raise Exception('No input file, need info.json')

print(input_data)
# Inverse Potential
# E(xi,xj) = a0/a*(1-dij/(ri+rj))^a, if(dij<ri+rj) where dij = sqrt((xi-xj)^2) 
#          = 0 ,if dij>=ri+rj
# xi and xj are the coordinates of atom i and j
# the input for get_energy is x=[x1,y1,z1,x2,y2,z2 ...]
# x1,y1,z1 is the coordinate for the first atom in 3D
# return value e is the energy
# return value g is the jacobian 
# where dE/dxi_alpha = g[i*dim + alpha] i=0..N-1, alpha = 0..dim-1
#pot = potentials.InversePower(a=mpow,eps=a0,boxv=boxv,radii = np.full(N,r),
#                        ncellx_scale = None,enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
if potential_type == "inverse_power":
    pot = potentials.InversePower(a=mpow,eps=a0,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif potential_type == "inv_pow_logi":
    pot = potentials.InvPowLogiPotential(a=mpow,eps=a0,k=k,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif potential_type == "linear":
    pot = potentials.InversePower(a=1.0,eps=r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif potential_type == "quadratic":
    pot = potentials.InversePower(a=2.0,eps=2.0*r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif potential_type == "1.5th":
    pot = potentials.InversePower(a=1.5,eps=1.5*r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif potential_type == "2.5th":
    pot = potentials.InversePower(a=2.5,eps=2.5*r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
else:
    potential_type = "inverse_power"
    input_data['potential_type'] = "inverse_power"
    pot = potentials.InversePower(a=2.0,eps=1.0,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
# special potential type
if optimization_method == 'bro_parallel_scsm':
    pot = potentials.Bro(mean=0.5*eps,sigma=eps/sqrt(12.0),boxv=boxv,noise_type=0,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif optimization_method == 'bro_parallel_smsc':
    pot = potentials.Bro(mean=1.0*eps,sigma=0.0,boxv=boxv,noise_type=0,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif optimization_method == 'bro_parallel_reciprocal':
    pot = potentials.Bro(mean=0.5*eps,sigma=eps/sqrt(12.0),boxv=boxv,noise_type=1,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif "bro" in optimization_method:
    pot = potentials.InversePower(a=1.0,eps=r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
elif optimization_method =="sme_bro_interaction":
    pot_noise = potentials.Bro(mean=0.0,sigma=sqrt(eps/3.0),boxv=boxv,noise_type=1,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    pot = potentials.InversePower(a=1.0,eps=r,boxv=boxv,radii = np.full(N,r),ncellx_scale = None, 
          enable_batch = True,method=Distance.PERIODIC,balance_omp=True)

# set to set the number of cores for NUMBA using environment variables 
# export NUMBA_NUM_THREADS=2

# for adam optimization   
numerical_eps = 1e-30
beta1,beta2 = 0.9,0.999 
m = np.zeros(dim*N,dtype=float) 
v = np.zeros(dim*N,dtype=float)

# normal distribution with size is not availalbe in numba
# size is a tuple, (dim1,dim2)
@numba.jit(nopython=True)
def normal_mat(size):
    arr = np.zeros(size)
    for i in prange(size[0]):
        for j in range(size[1]):
            arr[i,j] = np.random.normal()
    return arr

@numba.jit(nopython=True)
def normal_vec(n):
    arr = np.zeros((n,))
    for i in prange(n):
        arr[i] = np.random.normal()
    return arr

# uniform distribution with size is not availalbe in numba
# size is a tuple, (dim1,dim2)
@numba.jit(nopython=True)
def uniform_mat(size):
    arr = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            arr[i,j] = np.random.uniform(0.0,1.0)
    return arr


# The full broadcast is not supported by the current numba
# I have to write lots of loops
@numba.jit(nopython=True,parallel=True, fastmath=True)
def ro_step(x,n_nbr):
    dx = np.zeros((N*dim,)) # the size must be a tuple for numba to work
    for i in prange(N):
        if n_nbr[i] > 0:
            # generate vectors from a uniform,spherical distribution
            # uniform spherical distribution:Werner Krauth Page42
            gaussian = normal_vec(dim)
            norm = np.linalg.norm(gaussian)
            u = np.random.uniform(0.0,1.0)
            gamma = np.power(u,1.0/dim)
            kick = delta*gamma*gaussian/norm
            for d in range(dim):
                dx[i*dim + d] = kick[d]
    x += dx

@numba.jit(nopython=True,parallel=True, fastmath=True)
def ro_step_nbr(x,n_nbr):
    dx = np.zeros((N*dim,)) # the size must be a tuple for numba to work
    for i in prange(N):
        if n_nbr[i] > 0:
            kick = np.zeros((dim,))
            for n in range(n_nbr[i]):
                # generate vectors from a uniform,spherical distribution
                # uniform spherical distribution:Werner Krauth Page42
                gaussian = normal_vec(dim)
                norm = np.linalg.norm(gaussian)
                u = np.random.uniform(0.0,1.0)
                gamma = np.power(u,1.0/dim)
                kick += delta*gamma*gaussian/norm
            for pos in range(dim):
                dx[i*dim + pos] = kick[pos]
    x += dx

def save_data(loc,itern,data,rm=True,compress=True,mode="full"):
    if not os.path.isdir(loc):
        subprocess.run(["mkdir", loc])
    if mode == "concise":
        data.pop('total_gradient')
        data.pop('neighbor')
    elif mode == "super_concise":
        data.pop('total_gradient')
        data.pop('neighbor')
        data.pop('x')
        
    file_name = "data_iter{}.json".format(itern)
    f = os.path.join(loc,file_name)
    if os.path.exists(f) and rm:
        subprocess.run(["rm", "-r",f])
    fzip = f.replace(".json",".zip")
    # save the data as json a file
    if compress:
        fzip = f.replace(".json",".zip")
        with zipfile.ZipFile(fzip, 'w') as myzip:
            myzip.writestr("data.json",json.dumps(data),compress_type=zipfile.ZIP_DEFLATED)
    else:
        with open(f,'w') as fs:
            fs.write(json.dumps(data))

# this learning rate guarantee that each move is smaller than r
# where a0/2r is the maximum gradient between a pair of atoms

def get_cluster(N,inbr):
    visited = np.full(N,False)
    n_active = 0
    def bfs(iatom_start):
        #breadth first search to access neighbouring atoms
        queue = [iatom_start]
        qpos = 0
        visited[iatom_start] = True
        while qpos < len(queue):
            iatom = queue[qpos]
            for new_iatom in inbr[iatom]:
                if not visited[new_iatom]:
                    visited[new_iatom] = True
                    queue.append(new_iatom)
            qpos += 1
        return queue
    clusters = []
    for iatom in range(N):
        if len(inbr[iatom])>0:
            n_active += 1
        if visited[iatom]:
            continue
        else:
            c = bfs(iatom)
            clusters.append(c)
    return n_active,clusters

with open(input_file,'w') as fs:
    fs.write(json.dumps(input_data,indent=4))

record_data = {'t':[],'f_active':[],'total_energy':[],'phi':phi}
if 'eps' in input_data:
    record_data['eps'] = input_data['eps']
if 'lr' in input_data:
    record_data['lr'] = input_data['lr']
if 'batch_lr' in input_data:
    record_data['batch_lr'] = input_data['batch_lr']
    
    
import time
s1 = time.time()

if input_data['init_config'] == 'appending':
    t0 = iters[-1] + 1
else:
    t0 = 0
for itern in range(t0,n_iter):        
    if optimization_method == 'particle_sgd' or optimization_method == 'particle_sgd_match_bro':
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        rg= np.reshape(g_tot,(N,dim))
        rg_norm = np.linalg.norm(rg,axis=1).reshape(N,1)
        #com = np.mean(rx,axis=0)
        active_particles = np.array([i for i in range(N) if rg_norm[i] > numerical_eps])
        n_active = len(active_particles)
        if batch_mode == "by_fraction":
            batch_size = int(active_particles.size *fraction) + 1 
        if optimization_method == 'particle_sgd_match_bro':
            batch_size = int(active_particles.size *fraction) + 1
            lr = eps*active_particles.size/batch_size
        if 'alpha' in input_data:
            lr = alpha
        np.random.shuffle(active_particles)
        unselected_particles = active_particles[batch_size:]
        if unselected_particles.size > 0: 
            rg[unselected_particles] = 0.0
        dx = -lr*rg
        rx += dx
    elif optimization_method == 'pair_sgd' or optimization_method == 'pair_sgd_match_bro':
        if batch_mode == "by_size":
            n_pairs = pot.initialize_batch(x,batch_size,mode = "by_size",reset_clist=True)
        else:
            n_pairs = pot.initialize_batch(x,fraction,mode = "by_fraction",reset_clist=True)
            batch_size = int(n_pairs*fraction) + 1
        if optimization_method == 'pair_sgd_match_bro':
            lr = eps*n_pairs/batch_size
            batch_size = int(n_pairs*fraction) + 1
        elif 'alpha' in input_data:
            lr = alpha
        e_batch,g_batch = pot.get_batch_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        rg= np.reshape(g_batch,(N,dim))
        dx = -lr*rg
        #com = np.mean(rx,axis=0)
        rx += dx
    elif optimization_method == 'ro':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(N)])
        ro_step(x,n_nbr)
    elif optimization_method == 'ro_nbr':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(N)])
        ro_step_nbr(x,n_nbr)
    elif optimization_method == 'bro_parallel_smsc': # this is the way in the BRO paper
        e_tot,g_tot = pot.get_energy_gradient(x)
        dx= -np.reshape(g_tot,(N,dim))
        rx = np.reshape(x,(N,dim))
        #com = np.mean(rx,axis=0)
        dx_norm = np.linalg.norm(dx,axis=1) + numerical_eps # avoid dividing by 0
        dx = dx/dx_norm.reshape(N,1)*np.random.uniform(0.0,eps,(N,1))
        rx += dx 
    elif optimization_method == 'bro_parallel_scsm': # this is weakly approximated to bro_parallel_smsc 
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        #com = np.mean(rx,axis=0)
        x -= g_tot
    elif optimization_method == 'bro_parallel_reciprocal': # this is weakly approximated to bro_parallel_smsc 
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        #com = np.mean(rx,axis=0)
        x -= g_tot
    elif optimization_method == 'bfgs':
        res = minimize(pot.get_energy_gradient, x, method='L-BFGS-B', jac=True,options={'disp': True,'maxiter':n_iter,'gtol':1e-20})
        x = res.x
    elif optimization_method == 'sme_ro':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(N)]).reshape(N,1)
        n_nbr[np.where(n_nbr > 0)] = 1
        rx = np.reshape(x,(N,dim))
        dW = sqrt_dt*np.random.normal(size=(N,dim)) 
        if (dim == 2):
            rx += np.sqrt(delta*n_nbr/4.0)*dW
        elif (dim==3):
            rx += np.sqrt(delta*n_nbr/5.0)*dW
    elif optimization_method == 'sme_ro_nbr':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(N)]).reshape(N,1)
        rx = np.reshape(x,(N,dim))
        dW = sqrt_dt*np.random.normal(size=(N,dim)) 
        if (dim == 2):
            rx += np.sqrt(delta*n_nbr/4.0)*dW
        elif (dim==3):
            rx += np.sqrt(delta*n_nbr/5.0)*dW
    elif optimization_method == 'sme_bro':
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        dW = sqrt_dt*np.random.normal(size = (N,1))
        rg= np.reshape(g_tot,(N,dim))
        rg_norm = np.linalg.norm(rg,axis=1) + numerical_eps # avoid dividing by 0
        rx += -rg*dt + sqrt_eps*np.sqrt(1.0/3.0)*dW*rg # used to be sqrt(1/12)
    elif optimization_method == 'sme_bro_reciprocal':
        pass
    elif optimization_method == 'sme_bro_noise':
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        dW1 = sqrt_dt*np.random.normal(size = (N,1))
        dW2 = sqrt_dt*np.random.normal(size = (N,1))
        rg= np.reshape(g_tot,(N,dim))
        rx += -M_grad*rg*dt + sqrt_eps_noise*dW1*rg + 0.5*sqrt_eps_brown*dW2
    elif optimization_method == 'sme_bro_interaction':
        e_tot,g_tot = pot.get_energy_gradient(x)
        e,g_noise = pot_noise.get_energy_gradient(x)
        g_tot = np.array(g_tot)
        g_noise = np.array(g_noise)
        x-= (g_tot*dt + g_noise*sqrt_dt)
    elif optimization_method == 'sme_particle_sgd':
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(N,dim))
        rg= np.reshape(g_tot,(N,dim))
        rg_norm = np.linalg.norm(rg,axis=1)
        active_particles = np.array([i for i in range(N) if rg_norm[i] > numerical_eps])
        dW = sqrt_dt*np.random.normal(size = (N,1))
        n_active = active_particles.size
        if n_active > 0:
            B = min(batch_size,n_active)
            rx += -rg*lr/n_active*dt + sqrt_lr*np.sqrt(1.0/(B*n_active) - 1.0/(n_active*n_active))*dW*rg
          
    # check if absoring state every 2000 steps
    is_absorb = False
    if itern % 1000 == 0:
        if potential_type == "quadratic" or potential_type == "1.5th" or potential_type == "2.5th":
            inbr,r_d = pot.getNeighbors(x,cutoff_factor=1.0 - 1e-8)
        else:
            inbr,r_d = pot.getNeighbors(x)
        n_active = 0
        for iatom in range(N):
            if len(inbr[iatom])>0:
                n_active += 1
        if n_active ==0: # reaching the absorbing state
            is_absorb = True
    
    if itern >= iter_zoom[0] and itern<=iter_zoom[1]:
        if 'exp_nsave' in input_data: 
            if itern == force_save_at:
                n_save = 1
                force_save_at = int(exp_nsave*force_save_at + 1.0)
            else:
                n_save = 10000 # a very large number
        elif 'zoom_rate' in input_data:
            n_save = input_data['n_save']//zoom_rate
    else:
        n_save = input_data['n_save']
    
    if "n_record" in input_data.keys() and itern % n_record == 0:
        if potential_type == "quadratic" or potential_type == "1.5th" or potential_type == "2.5th":
            inbr,r_d = pot.getNeighbors(x,cutoff_factor=1.0 - 1e-8)
        else:
            inbr,r_d = pot.getNeighbors(x)
        n_active = 0
        for iatom in range(N):
            if len(inbr[iatom])>0:
                n_active += 1
        e_tot,g_tot =  pot.get_energy_gradient(x)
        record_data['t'].append(itern)
        record_data['f_active'].append(n_active*1.0/N)
        record_data['total_energy'].append(e_tot)
    if itern % n_save == 0 or itern==n_iter-1 or is_absorb:
        #print("Center of mass:" + str(com))
        e_tot,g_tot =  pot.get_energy_gradient(x)
        if potential_type == "quadratic" or potential_type == "1.5th" or potential_type == "2.5th":
            inbr,r_d = pot.getNeighbors(x,cutoff_factor=1.0 - 1e-8)
        else:
            inbr,r_d = pot.getNeighbors(x)
        n_active = 0
        for iatom in range(N):
            if len(inbr[iatom])>0:
                n_active += 1
        print("iteration={}, energy={}, n_active={}".format(itern,e_tot,n_active))
        # construct data dictionary to for output
        data = dict()
        data['iter'] = itern
        data['x'] = x.tolist()
        data['total_energy'] = e_tot
        data['total_gradient'] = g_tot
        data['neighbor'] = inbr
        data['n_active'] = n_active
        # additional informations
        if optimization_method == 'particle_sgd':
            data['batch_particles'] = active_particles[0:batch_size].tolist()
        elif optimization_method == 'pair_sgd':
            batch_pairs = pot.get_batch_pairs()
            data['batch_pairs'] = batch_pairs 
        elif optimization_method == 'bfgs':     
            data['iter'] = 1  
        save_data(loc,itern,data,mode = save_mode)
        if is_absorb or optimization_method == 'bfgs':
            break
        

#update the information file
with open(input_file,'w') as fs:
    fs.write(json.dumps(input_data,indent=4))

#save the time and f_active file
record_file = os.path.join(loc,'data_time.json')
with open(record_file,'w') as fs:
    fs.write(json.dumps(record_data,indent=4))

e1 = time.time()
print(e1-s1)
