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
    
    #batch energy and gradient
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_batch_energy(self, x):
        cdef double energy = self.baseptr.get().get_batch_energy(x)
        return energy

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_batch_energy_gradient(self, x):
        cdef vector[double] grad = np.zeros(self.natoms*self.ndim)
        cdef double energy = self.baseptr.get().get_batch_energy_gradient(x, grad)
        return energy, grad
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.    
    def get_batch_pairs(self):
        cdef cset[vector[unsigned long]] c_batch_pairs
        self.baseptr.get().get_batch_pairs(c_batch_pairs)
        #print(c_batch_pairs)
        batch_pairs = []
        for pair in c_batch_pairs:
            batch_pairs.append((pair[0],pair[1]))
        return batch_pairs
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def initialize_batch(self,np.ndarray[double, ndim=1] coords not None,size_or_frac,reset_clist = True,mode="by_size"):
        cdef unsigned long npairs
        if mode == "by_fraction":
            fraction = size_or_frac
            npairs = self.baseptr.get().initialize_batch_by_fraction(coords,fraction,reset_clist)
        elif mode == "by_size":
            batch_size = size_or_frac
            npairs = self.baseptr.get().initialize_batch_by_size(coords,batch_size,reset_clist)
        else:
            raise NotImplementedError("Batch initialization method not implemented")
        return npairs


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



cdef class _Cdef_Bro(_Cdef_BasePotential):

    def __cinit__(self, double mean, double sigma, np.ndarray radii, np.ndarray boxv, int noise_type = 0, 
                  use_cell_lists = False, enable_batch = False,ncellx_scale=None, balance_omp=True, method=Distance.PERIODIC):
        self.boxv = np.asarray(boxv, dtype='d')
        self.radii = np.asarray(radii, dtype='d')
        self.use_cell_lists = use_cell_lists
        self.enable_batch = enable_batch
        if ncellx_scale is None:
            self.ncellx_scale = get_ncellsx_scale(radii, boxv)
            print("Bro setting ncellx_scale to value: ", self.ncellx_scale)
        else:
            self.ncellx_scale = ncellx_scale
        self.balance_omp = balance_omp
        self.method = method
        self.ndim = self.boxv.size()
        self.natoms = self.radii.size()
        self.mean = mean 
        self.sigma = sigma
        self.noise_type = noise_type

        if method == Distance.CARTESIAN:
            raise NotImplementedError
        elif method == Distance.PERIODIC:
            if enable_batch:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicBatchCellLists[INT1](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicBatchCellLists[INT2](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicBatchCellLists[INT3](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicBatchCellLists[INT4](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                else:
                    raise NotImplementedError
            elif use_cell_lists:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicCellLists[INT1](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicCellLists[INT2](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicCellLists[INT3](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppBasePotential](<cppBasePotential*> new \
                        cppBroPeriodicCellLists[INT4](self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.ncellx_scale, self.balance_omp))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def __reduce__(self):
        d = {}
        return (self.__class__, (self.mean, self.sigma, self.radii, self.boxv, self.noise_type, self.use_cell_lists, self.ncellx_scale, self.balance_omp, self.method,), d)

    def __setstate__(self, d):
        pass

class Bro(_Cdef_Bro):
    """
    python wrapper to Bro
    """
