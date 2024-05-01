from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="NumericalAlgos",
    ext_modules=cythonize([
        "src/numalgos/interpolate/cython/splines.pyx",
        "src/numalgos/interpolate/cython/splines.pxd",
    ],
    compiler_directives={'language_level' : "3"},
    ),
    include_dirs=[np.get_include()],
)
