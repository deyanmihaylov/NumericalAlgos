cdef class CubicSpline:
    cdef:
        public double[:] x0
        public int n
        public double[:] a
        public double[:] b
        public double[:] c
        public double[:] d
        readonly int type_start
        readonly int type_end
        readonly double val_start
        readonly double val_end
