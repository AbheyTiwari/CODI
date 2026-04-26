"""
Setup script for building turboQuant C++ extension with pybind11.
Enables quantized embeddings for 4x compression and faster search.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
from pathlib import Path

# Get pybind11 include path from the installed package
try:
    import pybind11
    PYBIND11_INCLUDE = pybind11.get_include()
except ImportError:
    PYBIND11_INCLUDE = None

# Define the C++ extension
ext_modules = [
    Extension(
        "turboquant_pybind",
        ["turboQuant/turboquant_pybind.cpp"],
        include_dirs=[
            PYBIND11_INCLUDE,
            "turboQuant/",
        ] if PYBIND11_INCLUDE else ["turboQuant/"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"] if sys.platform != "win32" else ["/O2", "/std:c++14"],
    ),
]

setup(
    name="turboquant-pybind",
    version="0.1.0",
    description="TurboQuant C++ extension for fast vector quantization",
    author="CODI Team",
    ext_modules=ext_modules,
    install_requires=[
        "pybind11>=2.6.0",
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    zip_safe=False,
)
