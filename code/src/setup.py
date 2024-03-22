from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize([Extension("preprocess_cython", ["preprocess_cython.pyx"])]),
    include_dirs=[np.get_include()]
)
