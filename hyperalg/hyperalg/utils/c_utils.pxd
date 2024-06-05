from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
# from pele.potentials._pele cimport shared_ptr
from libcpp.memory cimport shared_ptr
cimport numpy as np

cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type
    ctypedef int INT4 "4"    # a fake type
    ctypedef int INT5 "5"    # a fake type

cdef extern from "hyperalg/omp_utils.hpp":
    cpdef size_t omp_get_thread_count() except +


cdef extern from "hyperalg/point_pattern_statistics.hpp" namespace "ha":
    cdef cppclass cppPointPatternStatisticsInterface "ha::PointPatternStatisticsInterface":
        size_t get_seed() except +
        void set_generator_seed(size_t) except +
        double get_mean() except +
        double get_variance() except +
        size_t get_count() except +
        void run(size_t) except +
    cdef cppclass cppPointPatternStatistics "ha::PointPatternStatistics"[ndim]:
        cppPointPatternStatistics(vector[double], vector[double], double, size_t) except +

cdef class _Cdef_PointPatternStatisticsInterface:
    cdef shared_ptr[cppPointPatternStatisticsInterface] baseptr
    cdef public size_t rseed, natoms, ndim
    cdef double radius
