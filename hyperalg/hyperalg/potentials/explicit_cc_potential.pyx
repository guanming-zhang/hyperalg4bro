"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_ExplicitCCPotential:

    def __cinit__(self, size_t N, vector[vector[double]] K, vector[double] Sk, vector[double] V, double L):
        self.N = N
        self.ndim = 2
        self.K = K
        self.Sk = Sk
        self.V = V
        self.L = L
        self.grad = np.zeros(self.N*self.ndim)
        self.baseptr = shared_ptr[cppExplicitCCPotential](new cppExplicitCCPotential(self.N, self.K, self.Sk, self.V, self.L))

    def get_energy(self, x):
        cdef double energy = self.baseptr.get().get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = self.baseptr.get().get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.N, self.K, self.Sk, self.V, self.L,), d)

    def __setstate__(self, d):
        pass


class ExplicitCCPotential(_Cdef_ExplicitCCPotential):
    """
    python wrapper to ExplicitCCPotential
    """
