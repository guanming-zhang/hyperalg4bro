from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libc.stdlib cimport malloc, free
cimport numpy as np

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type
    ctypedef int INT4 "4"    # a fake type


# use external c++ classes
cdef extern from "hyperalg/base_potential.hpp" namespace "ha":
    cdef cppclass cppBasePotential "ha::BasePotential":
        double get_energy(vector[double]&) except +
        double get_energy_gradient(vector[double]&, vector[double]&) except +

        double get_batch_energy(vector[double]&) except +
        double get_batch_energy_gradient(vector[double]&, vector[double]&) except +
        void get_batch_pairs(cset[vector[unsigned long] ] & batch_pairs) except +
        unsigned long initialize_batch_by_size(vector[double]&,unsigned long,bint) except+
        unsigned long initialize_batch_by_fraction(vector[double]&,double,bint) except+

        void get_neighbors(vector[double]&, vector[vector[size_t]]&, vector[vector[vector[double]]]&, double) except +
        void get_neighbors_picky(vector[double]&, vector[vector[size_t]]&, vector[vector[vector[double]]]&, vector[short]&, double) except +
cdef extern from "hyperalg/inversepower_potential.hpp" namespace "ha":
    cdef cppclass cppInversePowerCartesian "ha::InversePowerCartesian"[ndim]:
        cppInversePowerCartesian(double, double, vector[double]&) except +
    cdef cppclass cppInversePowerPeriodic "ha::InversePowerPeriodic"[ndim]:
        cppInversePowerPeriodic(double, double, vector[double]&, vector[double]&) except +
    cdef cppclass cppInversePowerPeriodicCellLists "ha::InversePowerPeriodicCellLists"[ndim]:
        cppInversePowerPeriodicCellLists(double, double, vector[double]&, vector[double]&, double, bint) except +
    cdef cppclass cppInversePowerPeriodicBatchCellLists "ha::InversePowerPeriodicBatchCellLists"[ndim]:
        cppInversePowerPeriodicBatchCellLists(double, double, vector[double]&, vector[double]&, double, bint) except +

cdef class _Cdef_BasePotential:
    cdef shared_ptr[cppBasePotential] baseptr
    cdef public size_t natoms, ndim
    cdef vector[double] boxv, radii
    cdef double a, eps, ncellx_scale
    cdef bint balance_omp

# https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
