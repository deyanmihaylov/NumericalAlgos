from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="CubicSpline",
    ext_modules=cythonize([
        "cinterpolate.pyx", "cinterpolate.pxd",
    ]),
    include_dirs=[np.get_include()],
)
