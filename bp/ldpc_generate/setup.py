from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
ldpc_generate = Extension("pyldpc_generate",
                         ["ldpc_generate.pyx", "ldpc_generate1.c", "ldpc_h2g1.c"],
                         include_dirs=[numpy.get_include()])

demo = Extension("demo",
                 ["demo.pyx", "demolibrary.c"],
                 include_dirs=[numpy.get_include()])

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [demo, ldpc_generate
        ]
)

