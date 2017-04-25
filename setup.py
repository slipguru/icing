#!/usr/bin/python
"""icing setup script.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

from setuptools import setup, Extension
import numpy as np

# Package Version
from icing import __version__ as version
# alignment_module = Extension('icing.align.align',
#                              sources=['icing/align/alignment.c'])
ssk_module = Extension(
    'icing.kernel.stringkernel',
    sources=['icing/kernel/sum_string_kernel.cpp'],
    include_dirs=[np.get_include()])
setup(
    name='icing',
    version=version,

    description=('A package to clonal relate immunoglobulins'),
    long_description=open('README.md').read(),
    author='Federico Tomasi',
    author_email='federico.tomasi@dibris.unige.it',
    maintainer='Federico Tomasi',
    maintainer_email='federico.tomasi@dibris.unige.it',
    url='https://github.com/slipguru/icing',
    download_url='https://github.com/slipguru/icing/archive/'+version+'.tar.gz',
    keywords=['IG', 'immunoglobulins', 'clonotypes'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',
    packages=['icing', 'icing.core', 'icing.utils', 'icing.align',
              'icing.externals', 'icing.models', 'icing.parallel',
              'icing.plotting', 'icing.kernel', 'icing.validation'],
    requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'sklearn (>=0.17)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)'],
    scripts=['scripts/ici_run.py', 'scripts/ici_analysis.py'],
    ext_modules=[ssk_module],
    include_dirs=[np.get_include()]
)
