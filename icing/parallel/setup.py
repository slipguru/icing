"""setup.py for parallel module.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "d_matrix_omp",
    sources=["d_matrix_omp.pyx", "alignment.c"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
    # language='c++'
)

setup(
    name='d_matrix_omp',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
