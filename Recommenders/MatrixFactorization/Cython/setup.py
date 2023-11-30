from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "MatrixFactorization_Cython_Epoch",
        ["MatrixFactorization_Cython_Epoch.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
)

