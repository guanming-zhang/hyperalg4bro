from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

cdef extern from "hyperalg/explicit_sk_potential.hpp" namespace "ha":
    cdef cppclass cppExplicitSkPotential "ha::explicit_Sk_potential":
        cppExplicitSkPotential(vector[double]&, double, vector[double]&, vector[double]&, vector[double]&, double, double, int) except +
        double get_energy(vector[double]&) except +
        double get_energy_gradient(vector[double]&, vector[double]&) except +

cdef class _Cdef_ExplicitSkPotential:
    cdef shared_ptr[cppExplicitSkPotential] baseptr
    cdef public size_t ndim, N
    cdef public double K, beta, gamma
    cdef int err_mode
    cdef vector[double] radii, Sk, V, grad, L
