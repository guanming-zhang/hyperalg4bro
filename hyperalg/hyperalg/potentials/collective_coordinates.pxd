from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

cdef extern from "hyperalg/cc_potential.hpp" namespace "ha":
    cdef cppclass cppCCPotential "ha::CC_potential":
        cppCCPotential(size_t N, double K, double a, double DKa, double L) except +
        double get_energy(vector[double]& x) except +
        double get_energy_gradient(vector[double]& x, vector[double]& grad) except +

cdef class _Cdef_CCPotential:
    cdef shared_ptr[cppCCPotential] baseptr
    cdef public size_t N, ndim
    cdef public double K, a, DKa, L
    cdef vector[double] grad
