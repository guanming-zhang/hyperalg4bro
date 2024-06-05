"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_CCPotential:

    def __cinit__(self, size_t N, double K, double a, double DKa, double L):
        self.N = N
        self.ndim = 2
        self.K = K
        self.a = a
        self.DKa = DKa
        self.L = L
        self.grad = np.zeros(self.N*self.ndim)
        self.baseptr = shared_ptr[cppCCPotential](new cppCCPotential(self.N, self.K, self.a, self.DKa, self.L))

    def get_energy(self, x):
        cdef double energy = self.baseptr.get().get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = self.baseptr.get().get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.N, self.K, self.a, self.DKa, self.L,), d)

    def __setstate__(self, d):
        pass


class CCPotential(_Cdef_CCPotential):
    """
    python wrapper to CCPotential
    """
