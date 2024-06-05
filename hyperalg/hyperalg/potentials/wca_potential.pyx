"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings
from hyperalg.distances.distance_enum import Distance
from hyperalg.distances.distance_utils import get_ncellsx_scale

cdef class _Cdef_BasePotential(object):
    """
    """
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_energy(self, x):
        cdef double energy = self.baseptr.get().get_energy(x)
        return energy

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_energy_gradient(self, x):
        cdef vector[double] grad = np.zeros(self.natoms*self.ndim)
        cdef double energy = self.baseptr.get().get_energy_gradient(x, grad)
        return energy, grad

    def getNeighbors(self, np.ndarray[double, ndim=1] coords not None,
                      include_atoms=None, cutoff_factor=1.0):
        cdef vector[vector[size_t]] c_neighbor_indss
        cdef vector[vector[vector[double]]] c_neighbor_distss
        cdef vector[short] c_include_atoms

        if include_atoms is None:
            (<cppBasePotential*>self.baseptr.get()).get_neighbors(
                coords, c_neighbor_indss, c_neighbor_distss,
                cutoff_factor)
        else:
            c_include_atoms = vector[short](len(include_atoms))
            for i in xrange(len(include_atoms)):
                c_include_atoms[i] = include_atoms[i]
            (<cppBasePotential*>self.baseptr.get()).get_neighbors_picky(
                coords, c_neighbor_indss, c_neighbor_distss,
                c_include_atoms, cutoff_factor)

        neighbor_indss = []
        for i in xrange(c_neighbor_indss.size()):
            neighbor_indss.append([])
            for c_neighbor_ind in c_neighbor_indss[i]:
                neighbor_indss[-1].append(c_neighbor_ind)

        neighbor_distss = []
        for i in xrange(c_neighbor_distss.size()):
            neighbor_distss.append([])
            for c_nneighbor_dist in c_neighbor_distss[i]:
                neighbor_distss[-1].append([])
                for dist_comp in c_nneighbor_dist:
                    neighbor_distss[-1][-1].append(dist_comp)

        return neighbor_indss, neighbor_distss


cdef class _Cdef_WCAPotential(_Cdef_BasePotential):

    def __cinit__(self, double sigma, double eps, np.ndarray radii, np.ndarray boxv, use_cell_lists = False, ncellx_scale=None, balance_omp=True, method=Distance.PERIODIC):
        self.boxv = np.asarray(boxv, dtype='d')
        self.radii = np.asarray(radii, dtype='d')
        self.use_cell_lists = use_cell_lists
        if ncellx_scale is None:
            self.ncellx_scale = get_ncellsx_scale(radii, boxv)
            print("WCA setting ncellx_scale to value: ", self.ncellx_scale)
        else:
            self.ncellx_scale = ncellx_scale
        self.balance_omp = balance_omp
        self.method = method
        self.natoms = self.radii.size()
        self.ndim = self.boxv.size()
        self.sigma = sigma
        self.eps = eps

        if method == Distance.CARTESIAN:
            if self.ndim == 2:
                self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                    cppWCA[INT2](self.sigma, self.eps, self.radii))
            elif self.ndim == 3:
                self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                    cppWCA[INT3](self.sigma, self.eps, self.radii))
            else:
                raise NotImplementedError
        elif method == Distance.PERIODIC:
            if use_cell_lists:
                if self.ndim == 2:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppWCAPeriodicCellLists[INT2](self.sigma, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppWCAPeriodicCellLists[INT3](self.sigma, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                else:
                    raise NotImplementedError
            else:
                if self.ndim == 2:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppWCAPeriodic[INT2](self.sigma, self.eps, self.radii, self.boxv))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppWCAPeriodic[INT3](self.sigma, self.eps, self.radii, self.boxv))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
    
    def __reduce__(self):
        d = {}
        return (self.__class__, (self.sigma, self.eps, self.radii, self.boxv, self.use_cell_lists, self.ncellx_scale, self.balance_omp, self.method,), d)

    def __setstate__(self, d):
        pass

class WCAPotential(_Cdef_WCAPotential):
    """
    python wrapper to WCA
    """
