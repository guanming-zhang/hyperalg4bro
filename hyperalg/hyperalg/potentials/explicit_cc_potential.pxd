from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

cdef extern from "hyperalg/explicit_cc_potential.hpp" namespace "ha":
    cdef cppclass cppExplicitCCPotential "ha::explicit_CC_potential":
        cppExplicitCCPotential(size_t N, vector[vector[double]]& K, vector[double]& Sk, vector[double]& V, double L) except +
        double get_energy(vector[double]& x) except +
        double get_energy_gradient(vector[double]& x, vector[double]& grad) except +

cdef class _Cdef_ExplicitCCPotential:
    cdef shared_ptr[cppExplicitCCPotential] baseptr
    cdef public size_t N, ndim
    cdef public double L
    cdef vector[double] grad
    cdef vector[double] Sk, V
    cdef vector[vector[double]] K
