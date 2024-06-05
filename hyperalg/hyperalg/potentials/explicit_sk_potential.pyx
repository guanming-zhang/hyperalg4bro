"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_ExplicitSkPotential:

    def __cinit__(self, np.ndarray radii, double K, np.ndarray Sk, np.ndarray V, np.ndarray L, double beta=100, double gamma=0.1, int err_mode = 1):
        self.radii = radii
        self.N = len(radii)
        self.K = K
        self.Sk = Sk.ravel()
        self.V = V.ravel()
        assert len(self.Sk) == len(self.V)
        self.L = L
        self.ndim = len(L)
        self.beta = beta
        self.gamma = gamma
        self.err_mode = err_mode
        self.grad = np.zeros(self.N*self.ndim)
        self.baseptr = shared_ptr[cppExplicitSkPotential](new cppExplicitSkPotential(self.radii, self.K, self.Sk, self.V, self.L, self.beta, self.gamma, self.err_mode))

    def get_energy(self, x):
        cdef double energy = self.baseptr.get().get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = self.baseptr.get().get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.radii, self.K, self.Sk, self.V, self.L,self.beta, self.gamma, self.err_mode), d)

    def __setstate__(self, d):
        pass


class ExplicitSkPotential(_Cdef_ExplicitSkPotential):
    """
    python wrapper to ExplicitSkPotential
    """
