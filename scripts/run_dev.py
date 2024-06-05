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
import time


class Info:
    def __init__(self,input_dir):
        input_file= os.path.join(input_dir,'info.json')
        if not os.path.isfile(input_file):
            raise FileNotFoundError("The input file" + input_file + "does not exist")
        with open(input_file) as fs:
            input_dict = json.loads(fs.read())
        self.input_dict = input_dict
        self.loc = input_dir
        compulsory_keys = ["N","dim","box_size","phi","optimization_method","init_config"
                       "potential_type"]
        
        print("--------Input info:--------")
        print(json.dumps(input_dict,indent=4))
        
        for k in compulsory_keys:
            if not k in input_dict:
                raise ValueError(k + " missing in the info.json")
            else:
                setattr(self,k,input_dict[k])
        if self.dim == 3:
            self.r = np.power(3.0/4.0*self.box_size[0]*self.box_size[1]*self.box_size[2]*self.phi/np.pi/self.N,1.0/3.0)
            input_dict["r"] = self.r
        elif self.dim == 2:
            self.r = np.power(self.box_size[0]*self.box_size[1]*self.phi/self.N/np.pi,1.0/2.0)
            input_dict["r"] = self.r
        if "sgd" in input_dict["optimization_method"]:
            if not "alpha" in input_dict:
                raise ValueError("alpha(learning rate) for sgd is missing in the info.json")
            else:
                self.alpha *= 2.0*self.r
            if "fraction" in input_dict:
                self.batch_mode = "by_fraction"
            elif "batch_size" in input_dict:
                self.batch_mode = "by_size"
            else:
                raise ValueError("For sgd methods, fraction or batch_size must be given")            
        if input_dict["potential_type"] == "inverse_power":
            if "a0" not in input_dict:
                raise ValueError("a0 is missing in the info.json")
            if "mpow" not in input_dict:
                raise ValueError("mpow is missing in the info.json")
        elif "bro" in input_dict["potential_type"]:
            if not "eps" in input_dict:
                raise ValueError("eps(kick size) for bro is missing in the info.json")
            else:
                self.eps *= 2.0*self.r
        
        if 'normalized_n_iter' in input_dict:
            self.int(input_dict['normalized_n_iter']/input_dict['fraction'] +0.5)

        if not "cut_off" in input_dict:
            self.cut_off = 0.0 # the default cut_off value
        # update the input information
        with open(input_file,'w') as fs:
            fs.write(json.dumps(input_dict,indent=4))

def initialize_position(input:Info):
    x = np.zeros((input.N,input.dim))
    current_t = 0
    if input.init_config == "random":
        for d in range(input.dim):
            x[:,d] = np.random.uniform(0.0,input.box_size[d],input.N)
        x = x.reshape(input.N*input.dim)
    elif input.init_config == 'fixed':
        st0 = np.random.get_state() #store the random seed
        np.random.seed(314) # set random seed to a lucky number
        x = np.zeros((input.N,input.dim))
        np.random.seed(None)
        for d in range(input.dim):
            x[:,d] = np.random.uniform(0.0,input.box_size[d],input.N)
            x = x.reshape(input.N*input.dim)
        np.random.set_state(st0) #recover the old random seed
    elif input.init_config == 'appending':
        dl = Utils.DataLoader(input.loc,mode='last_existed')
        iters,x_list = dl.get_data_list('x')
        current_t = iters[-1]
        x = np.array(x_list[-1])
    return current_t,x

def construct_potential(input:Info):
    boxv = np.array(input.box_size)
    if input.potential_type == "inverse_power":
        pot = potentials.InversePower(a=input.mpow,eps=input.a0,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type == "inv_pow_logi": # this is the Pokemen(a cute potential) I created but never used
        pot = potentials.InvPowLogiPotential(a=input.mpow,eps=input.a0,k=k,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type == "linear":
        pot = potentials.InversePower(a=1.0,eps=input.r,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type == "quadratic":
        pot = potentials.InversePower(a=2.0,eps=2.0*input.r,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type == "1.5th":
        pot = potentials.InversePower(a=1.5,eps=1.5*input.r,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type == "2.5th":
        pot = potentials.InversePower(a=2.5,eps=2.5*input.r,boxv=boxv,radii = np.full(input.N,input.r),ncellx_scale = None, 
                enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
    elif input.potential_type  == 'bro':
        if input.optimization_method == 'bro_parallel_scsm':
            pot = potentials.Bro(mean=0.5*input.eps,sigma=input.eps/sqrt(12.0),boxv=boxv,noise_type=0,radii = np.full(input.N,input.r),ncellx_scale = None, 
                    enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
        elif input.optimization_method == 'bro_parallel_smsc':
            pot = potentials.Bro(mean=1.0*input.eps,sigma=0.0,boxv=boxv,noise_type=0,radii = np.full(input.N,input.r),ncellx_scale = None, 
                    enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
        elif input.optimization_method == 'bro_parallel_reciprocal':
            pot = potentials.Bro(mean=0.5*input.eps,sigma=input.eps/sqrt(12.0),boxv=boxv,noise_type=1,radii = np.full(input.N,input.r),ncellx_scale = None, 
                    enable_batch = True,method=Distance.PERIODIC,balance_omp=True)
        else:
            raise NotImplementedError("BRO method, " + input.optimization_method + " not implemented, no potential constructed")
    else:
        raise NotImplementedError("Potential," + input.potential_type + "not implemented")
    return pot

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

# set to set the number of cores for NUMBA using environment variables 
# export NUMBA_NUM_THREADS=2



#----------------------------functions using numba for RO------------------------------------------------
# 1)without numba, it is slow,
# 2)numba does not support broadcasting 

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
    dx = np.zeros((info.N*info.dim,)) # the size must be a tuple for numba to work
    for i in prange(N):
        if n_nbr[i] > 0:
            # generate vectors from a uniform,spherical distribution
            # uniform spherical distribution:Werner Krauth Page42
            gaussian = normal_vec(info.dim)
            norm = np.linalg.norm(gaussian)
            u = np.random.uniform(0.0,1.0)
            gamma = np.power(u,1.0/info.dim)
            kick = info.delta*gamma*gaussian/norm
            for d in range(info.dim):
                dx[i*info.dim + d] = kick[d]
    x += dx

@numba.jit(nopython=True,parallel=True, fastmath=True)
def ro_step_nbr(x,n_nbr):
    dx = np.zeros((info.N*info.dim,)) # the size must be a tuple for numba to work
    for i in prange(N):
        if n_nbr[i] > 0:
            kick = np.zeros((info.dim,))
            for n in range(n_nbr[i]):
                # generate vectors from a uniform,spherical distribution
                # uniform spherical distribution:Werner Krauth's book Page42
                gaussian = normal_vec(info.dim)
                norm = np.linalg.norm(gaussian)
                u = np.random.uniform(0.0,1.0)
                gamma = np.power(u,1.0/info.dim)
                kick += info.delta*gamma*gaussian/norm
            for pos in range(info.dim):
                dx[i*info.dim + pos] = kick[pos]
    x += dx

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

#----------------------------the main code starts------------------------------------------------
loc = sys.argv[1]     
# get the input information
info = Info(loc)
# initialization
t_0,x = initialize_position(info)
# construct the potential
pot = construct_potential(info)

#---------------------------- define the auxiliary variables-------------------------------------
data_time = {"step":[],"f_active":[],"total_energy":[]} 
# the smallest number in this simulation   
numerical_eps = 1e-15

#---------------------------the simulation loop starts------------------------------------------
print("--------Simulation starts--------")
# start the timer
s_t = time.time()
for itern in range(t_0,info.n_iter):        
    if info.optimization_method == 'particle_sgd':
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(info.N,info.dim))
        rg= np.reshape(g_tot,(info.N,info.dim))
        rg_norm = np.linalg.norm(rg,axis=1).reshape(info.N,1)
        #com = np.mean(rx,axis=0)
        active_particles = np.array([i for i in range(info.N) if rg_norm[i] > numerical_eps])
        n_active = len(active_particles)
        if info.batch_mode == "by_fraction":
            batch_size = int(active_particles.size *info.fraction) + 1 
        np.random.shuffle(active_particles)
        unselected_particles = active_particles[batch_size:]
        if unselected_particles.size > 0: 
            rg[unselected_particles] = 0.0
        dx = -info.alpha*rg
        rx += dx
    elif info.optimization_method == 'pair_sgd':
        if info.batch_mode == "by_size":
            n_pairs = pot.initialize_batch(x,batch_size,mode = "by_size",reset_clist=True)
        else:
            n_pairs = pot.initialize_batch(x,info.fraction,mode = "by_fraction",reset_clist=True)
            batch_size = int(n_pairs*info.fraction) + 1
        e_batch,g_batch = pot.get_batch_energy_gradient(x)
        rx = np.reshape(x,(info.N,info.dim))
        rg= np.reshape(g_batch,(info.N,info.dim))
        dx = -info.alpha*rg
        rx += dx
    elif info.optimization_method == 'ro':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(info.N)])
        ro_step(x,n_nbr)
    elif info.optimization_method == 'ro_nbr':
        inbr,dist = pot.getNeighbors(x)
        n_nbr = np.array([len(inbr[i]) for i in range(info.N)])
        ro_step_nbr(x,n_nbr)
    elif info.optimization_method == 'bro_parallel_smsc': # this is the way in Sam's BRO paper
        e_tot,g_tot = pot.get_energy_gradient(x)
        dx= -np.reshape(g_tot,(info.N,info.dim))
        rx = np.reshape(x,(info.N,info.dim))
        #com = np.mean(rx,axis=0)
        dx_norm = np.linalg.norm(dx,axis=1) + numerical_eps # avoid dividing by 0
        dx = dx/dx_norm.reshape(info.N,1)*np.random.uniform(0.0,info.eps,(info.N,1))
        rx += dx 
    elif info.optimization_method == 'bro_parallel_scsm': # this is weakly approximated to bro_parallel_smsc 
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(info.N,info.dim))
        #com = np.mean(rx,axis=0)
        x -= g_tot
    elif info.optimization_method == 'bro_parallel_reciprocal': # this is weakly approximated to bro_parallel_smsc 
        e_tot,g_tot = pot.get_energy_gradient(x)
        rx = np.reshape(x,(info.N,info.dim))
        #com = np.mean(rx,axis=0)
        x -= g_tot
    elif info.optimization_method == 'bfgs':
        res = minimize(pot.get_energy_gradient, x, method='L-BFGS-B', jac=True,options={'disp': True,'maxiter':info.n_iter,'gtol':1e-20})
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
          
    # check if absoring state every 1000 steps
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
        

record_file = os.path.join(loc,'data_time.json')
with open(record_file,'w') as fs:
    fs.write(json.dumps(data_time,indent=4))

e_t = time.time()
print("--------Simulation ends--------")
print("This simulation took " + str(e_t-s_t) + " secs")

