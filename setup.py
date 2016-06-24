#!/usr/bin/python
"""ignet setup script."""

from distutils.core import setup

# Package Version
from ignet import __version__ as version

setup(
    name='ignet',
    version=version,

    description=('A package to clonal relate immunoglobulins'),
    long_description=open('README.md').read(),
    author='Federico Tomasi',
    author_email='federico.tomasi@dibris.unige.it',
    maintainer='Federico Tomasi',
    maintainer_email='federico.tomasi@dibris.unige.it',
    url='https://github.com/slipguru/ignet',
    download_url='https://github.com/slipguru/ignet/tarball/'+version,
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
    packages=['ignet', 'ignet.core', 'ignet.utils'],
    requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'sklearn (>=0.17)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)'],
    scripts=['scripts/ig_run.py', 'scripts/ig_analysis.py'],
)
