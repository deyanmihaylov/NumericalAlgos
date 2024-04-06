cdef class CubicSpline:
    cdef:
        public double[:] x0
        public double[:] a
        public double[:] b
        public double[:] c
        public double[:] d
