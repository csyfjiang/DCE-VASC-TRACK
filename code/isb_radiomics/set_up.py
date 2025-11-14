"""Setup for Cython builds."""
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "isb_radiomics.FeatureClasses.glcm",
        ["isb_radiomics/FeatureClasses/glcm.pyx"],
        extra_compile_args=["/openmp"],
        include_dirs=[numpy.get_include()],
    ),
    # ... add all your .pyx extensions similarly
]

setup(
    name="isb_radiomics",
    ext_modules=cythonize(ext_modules, gdb_debug=True),
    packages=["isb_radiomics", "isb_radiomics.FeatureClasses"],
    include_package_data=True,
)