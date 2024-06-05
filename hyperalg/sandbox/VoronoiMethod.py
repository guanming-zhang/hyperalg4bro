import sys
import imageio
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.spatial import Voronoi, voronoi_plot_2d
import itertools
import matplotlib
import matplotlib.pyplot as plt
import time
from PIL import Image
#import pyrtools as pt

def main(n):
    '''Generates a Hyperuniform point pattern given a box size and number density'''

    # Assume unit square in 2D
    L = (1,1)

    # Pick a number of particles appropriate for the triangular lattice limit
    N = 31*31#113*113#418

    points = np.random.rand(N,len(L))
    #plot_points('Points_random',points, L)
    #plot_voro('Points_random',points, L)
    vor = Voronoi(points)

    #fig=voronoi_plot_2d(vor, show_vertices=False)
    #ims=[]
    tic = time.process_time()
    for h in range(10000):
        periodic = np.vstack((points, points + np.array((L[0],0))))
        periodic = np.vstack((periodic, periodic + np.array((0,L[1]))))
        periodic += 0.5
        periodic %= 2
        periodic -= 0.5
        vor = Voronoi(periodic)
        '''
        if h%10 == 0:
            plot_points('Points_generated_'+str(h),periodic, L)
        plot_voro('_Points_generated_'+str(h),periodic, L)
        ims.append(imageio.imread('_Points_generated_'+str(h)+'_voro.png'))
        plt.close()'''
        for i in range(N):
            points[i] = find_center(vor.regions[vor.point_region[i]],vor.vertices)
    toc = time.process_time()
    #print(toc-tic)
    #plot_voro('Points_generated_'+str(h),points, L)
    #imageio.mimsave('VoronoiMethod.gif', ims, fps=10)

    np.savetxt('VM_'+str(n)+'.dat',points)
    #print(str(L)+'\n'+str(points.flatten()))


def find_center(region, vertices):
    '''Returns the center point of a convex polygon'''
    center = np.asarray([0.0,0.0])
    for i in region:
        center += vertices[i]
    return center/len(region)

if __name__ == '__main__':
    for i in range(8):
        main(i)
    sys.exit()
