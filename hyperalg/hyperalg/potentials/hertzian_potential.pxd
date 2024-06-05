from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

cdef extern from "hyperalg/hertzian_potential.hpp" namespace "ha":
    cdef cppclass cppHertzianPotential "ha::Hertzian_potential":
        cppHertzianPotential(size_t N, double pow, double eps, vector[double]& radii, double L) except +
        double get_energy(vector[double]& x) except +
        double get_energy_gradient(vector[double]& x, vector[double]& grad) except +

cdef class _Cdef_HertzianPotential:
    cdef shared_ptr[cppHertzianPotential] baseptr
    cdef public size_t N, ndim
    cdef public double pow, eps, L
    cdef vector[double] grad, radii
