from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="CubicSpline",
    ext_modules=cythonize([
        "cinterpolate.pyx",
        "cinterpolate.pxd",
    ],
    compiler_directives={'language_level' : "3"},
    ),
    include_dirs=[np.get_include()],
)
