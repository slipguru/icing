"""setup.py for alignment module.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from distutils.core import setup, Extension

module1 = Extension('align',
                    sources=['alignment.c'])

setup(name='ignet_alignment',
      version='1.0',
      description='This is a demo package',
      ext_modules=[module1])
