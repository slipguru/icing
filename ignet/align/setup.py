"""setup.py for alignment module."""

from distutils.core import setup, Extension

module1 = Extension('align',
                    sources=['alignment.c'])

setup(name='ignet_alignment',
      version='1.0',
      description='This is a demo package',
      ext_modules=[module1])
