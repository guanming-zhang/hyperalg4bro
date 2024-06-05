"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_HertzianPotential:

    def __cinit__(self, size_t N, double pow, double eps, vector[double]& radii, double L):
        self.N = N
        self.ndim = 3
        self.pow = pow
        self.eps = eps
        self.radii = radii
        self.L = L
        self.grad = np.zeros(self.N*self.ndim)
        self.baseptr = shared_ptr[cppHertzianPotential](new cppHertzianPotential(self.N, self.pow, self.eps, self.radii, self.L))

    def get_energy(self, x):
        cdef double energy = self.baseptr.get().get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = self.baseptr.get().get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.N, self.pow, self.eps, self.radii, self.L,), d)

    def __setstate__(self, d):
        pass


class HertzianPotential(_Cdef_HertzianPotential):
    """
    python wrapper to HertzianPotential
    """
