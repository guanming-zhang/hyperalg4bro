import sys
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
from scipy.signal import convolve2d
from scipy.optimize import fmin_l_bfgs_b, minimize, basinhopping
from numba import njit, jit
import itertools
import matplotlib
import matplotlib.pyplot as plt
import time
#import pyrtools as pt

def log_phi(x):
    """
    callback function
    """
    phi_values_cc.append(phi_cc)
    phi_values_hz.append(phi_hz)

def main(N,n):
    '''Generates n Hyperuniform point patterns given a box size and number density'''

    global phi_values_cc, phi_values_hz
    # Assume unit square in 2D
    L = (1, 1)

    # Pick a number of particles appropriate for the triangular lattice limit
    #N = 168

    # Pick a cutoff K/pi, step potential V(k)=V_0 for |k| < K
    K = 10  # units of 2pi
    V0 = 1

    times = 0

    kspan = find_k(K * 2*np.pi, L)
    print('Number of k-vectors: ' + str(kspan.shape))

    # Structure factor power law form S(k) ~ |k|^a
    a = 1
    C0 = -N / 2
    if a > 0:
        D = 75 / (K * 2*np.pi)**a
        knorm = np.linalg.norm(kspan,axis=1)
        #C0 = C0 + D*2*((knorm-10*np.pi)*(knorm>10*np.pi))**a
        C0 = C0 + D*(knorm)**a
        #C0 = (-1-np.cos(knorm*3/(K*2)))*N/4
    N_pairs = int(N * (N - 1) / 2)
    pair_dists = np.zeros([N_pairs, 2])

    jam = 0.84
    rad = np.sqrt(jam*np.prod(L)/(N*np.pi))

    for i in range(n):
        # Initialize points
        points = np.random.rand(N * len(L))
        #points = np.loadtxt('Stealthy/CCD_N_'+str(N)+'_K_'+str(K)+'pi_'+str(i)+'.dat')
        phi_values_cc = []
        phi_values_hz = []

        tic = time.process_time()
        res = minimize(mixed_potential, points, args=(kspan, V0, L, C0, pair_dists, 2.5, 1.0, rad),
                       method='L-BFGS-B', jac=True, callback=log_phi)#, options={'ftol': 1e-20,'gtol': 1e-20})  # , )
        #res = basinhopping(cc_potential, points, T=1.0, minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "args":(kspan, V0, L, C0, pair_dists), "callback":log_phi})
        toc = time.process_time()
        print('Elapsed time: ' + str(toc - tic))
        times += toc-tic

        points = res.x
        points -= np.floor(points)
        print(phi_values_cc[-1])
        print(phi_values_hz[-1])
        #plot_points('Points_generated_N_'+str(N)+'_K_'+str(K)+'pi_',points, L)
        #plot_voro('Points_generated_N_'+str(N)+'_K_'+str(K)+'pi_',points, L)
        np.savetxt('mixed_Hz_a' + str(a) + '_' + str(i) + '.dat', points)
        #print(phi_values[-1])
        np.savetxt('mixed_Hz_a' + str(a) + '_phi_' + str(i) +'.dat',np.asarray([phi_values_cc,phi_values_hz]))
        if i == 0:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(phi_values_cc, label = 'CC')
            ax.plot(phi_values_hz, label = 'Hertzian')
            ax.plot(np.array(phi_values_cc)+np.array(phi_values_hz), label='Total')
            ax.legend()
            x.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy')
            plt.savefig('mixed_Hz_a' + str(a) + '_eval_' + str(i) + '.png')
    # points = np.array([0.92,0.14,0.33,0.45,0.82,0.27,0.67,0.55])
    # #points = np.array([0.0,0.0,0.0,0.5,0.5,0.0,0.5,0.5])
    # #points = np.array([0,0,0.5,0.5])
    # #points=np.loadtxt('CCD_a1_0.dat')
    # phi = cc_potential(points, kspan, V0, L, C0, pair_dists)
    # print(phi[0])
    # print(phi[1])
    # print(num_grad(cc_potential,points, kspan, V0, L, C0, pair_dists))

    # phi = hertzian_potential(points, 2.5,1, 0.04, L, pair_dists)
    # print(phi[0])
    #print(phi[1])
    #print(num_grad(hertzian_potential,points, 2.5,1, 0.5, L,pair_dists))
    return times/n

def find_k(K, L):
    '''Returns the list of all k vectors for a 2D square box of length L such that |k| < K'''
    k = (0, 0)
    for nx in range(int(K * L[0] / (2 * np.pi) + 1)):
        for ny in range(int(K * L[1] / (2 * np.pi) + 1)):
            if 4 * np.pi**2 * (nx**2 / L[0]**2 + ny**2 / L[1]**2) < K**2:
                k = np.vstack((k, (2 * np.pi * nx, 2 * np.pi * ny), (-2 * np.pi * nx, 2 * np.pi * ny),
                               (2 * np.pi * nx, -2 * np.pi * ny), (-2 * np.pi * nx, -2 * np.pi * ny)))
    k = np.unique(k, axis=0)
    k = np.delete(k, int(k.shape[0] / 2), axis=0)
    return k

def cc_potential(r, *args):
    k = args[0]
    V0 = args[1]
    L = args[2]
    C0 = args[3]
    pair_dists = args[4]
    global phi_step
    phi_step, phi_grad = _cc_potential(r, k, V0, L, C0, pair_dists)
    return phi_step, phi_grad

def _cc_potential(r, k, V0, L, C0, pair_dists):
    rr = np.reshape(r, (-1, len(L))) % L[0]
    N = rr.shape[0]
    get_pair_distances(rr, pair_dists)
    CC = C(k, pair_dists.T) - C0
    phi_step = V0 * np.sum(CC**2)
    dC = dCdx(k, rr)
    #print(dC.shape)
    #print(np.sum(dC,axis=1))
    phi_grad = V0 * np.sum(2 * CC * dC, axis=-1)
    return phi_step, phi_grad

def hertzian_potential(r, *args):
    a = args[0]
    eps = args[1]
    rad = args[2]
    L = args[3]
    pair_dists = args[4]
    global phi_step
    phi_step, phi_grad = _hertzian_potential(r, a, eps, rad, L, pair_dists)
    return phi_step, phi_grad

def _hertzian_potential(r, a, eps, rad, L, pair_dists):
    rr = np.reshape(r, (-1, len(L))) % L[0]
    N = rr.shape[0]
    get_pair_distances(rr, pair_dists)
    dists = np.linalg.norm(pair_dists, axis=1)
    phi_step = 0.0
    for dist in dists:
        if dist < rad*2:
            phi_step += ((1-dist/(rad*2))**a)*eps/a
    phi_grad = np.zeros(rr.shape)
    for i in range(N):
        dr = np.delete(rr[i] - rr,i,0)
        dists = np.linalg.norm(dr,axis=1)
        dr = dr/np.vstack([dists,dists]).T
        gradfactor = 0.0
        for dist in dists:
            if dist < rad*2:
                gradfactor += ((1-dist/(rad*2))**(a-1))*eps/(-rad*2)
        phi_grad[i] = np.sum(dr*gradfactor,axis=0)
    return phi_step, phi_grad.ravel()

def mixed_potential(r, *args):
    k = args[0]
    V0 = args[1]
    L = args[2]
    C0 = args[3]
    pair_dists = args[4]
    a = args[5]
    eps = args[6]
    rad = args[7]
    global phi_cc, phi_hz
    phi_cc, grad_cc = _cc_potential(r, k, V0, L, C0, pair_dists)
    phi_hz, grad_hz = _hertzian_potential(r, a, eps, rad, L, pair_dists)
    phi_step = phi_cc + phi_hz
    return phi_step, grad_cc+grad_hz

def num_grad(func, r, *args):
    eps = 1e-6
    grad = np.zeros(r.shape)
    rr = np.array(r)
    for i in range(len(grad)):
        rr[i] -= eps
        eminus = func(rr, *args)[0]
        rr[i] += 2*eps
        eplus = func(rr, *args)[0]
        grad[i] = (eplus-eminus)/(2*eps)
        rr[i] = r[i]
    return grad

@njit
def C(k, dr):
    '''
    Real Collective Coordinate function: S(k) = N+2*C(k)
    sums over the pair distances
    '''
    return np.sum(np.cos(np.dot(k, dr)), axis=1)


def dCdx(k, rr):
    dC = np.zeros((rr.shape[0], 2, k.shape[0]), dtype='d')
    _dCdx(k, rr, dC)
    return dC.reshape(2*rr.shape[0], k.shape[0])

@njit
def _dCdx(k, rr, dC):
    for i in range(rr.shape[0]):
        dri = rr - rr[i]
        dC[i] = k.T * np.sum(np.sin(np.dot(k, dri.T)), axis=1)

@njit
def get_pair_distances(r, pairs):
    '''Updates pairs to hold the pair distances between each pair of r elements'''
    x = 0
    N = r.shape[0]
    for i in range(N - 1): # i from 0 to N-2
        for j in range(i + 1, N): # j from i+1 to N-1
            pairs[x] = (r[i] - r[j] + 0.5) % 1 - 0.5
            x += 1
def plottime():
    fig = plt.figure()
    ax = fig.gca()
    
    data = np.loadtxt('times')
    ax.scatter([20,40,60,80,100,200,300], data)#, label='Initial Points: '+str(n))
    #data = hkl.load(file_name+'_generated_structure.hkl')
    #print(data.shape)
    #ax.scatter(data[0], data[1], label='Generated')
    #ax.plot([10,10],[0,2],c='k',linestyle='dashed')
    #ax.legend()
    ax.set_title('Runtime of Collective Coordinates')
    #ax.set_xlim(0,25)
    #ax.set_ylim(0,2.5)
    #ax.set_xlabel(r'$|k|/(2\pi$)')
    ax.set_xlabel('N')
    ax.set_ylabel('Time (s)')
    plt.savefig('time.png', bbox_inches = 'tight', pad_inches = 0)



if __name__ == '__main__':
    #times = []
    #for N in [20,40,60,80,100,200,300]:
    #    times.append(main(N,10))
    #np.savetxt('times',np.array(times))
    #plottime()
    main(168,1)
    sys.exit()