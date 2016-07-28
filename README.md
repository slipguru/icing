===========================================
ICING - Infer clones of immunoglobulin data
===========================================
A Python package to clonal relate immunoglobulins.

**ICING** is an implementation of an unsupervised learning technique
used to identify `clones`, which are groups of immunoglobulins which share
a common ancestor. In other words, immunoglobulins in the same group descend
from the same germline.
Also, the method is designed to be used also in contexts where
the number of samples is very high. For this reason, the framework can be used
in combination with NGS technologies.

## Dependencies
ICING is developed using Python 2.7 and inherits its main functionalities from:

* numpy
* scipy
* scikit-learn
* matplotlib
* seaborn

## Authors and Contributors
Current developer: Federico Tomasi ([@fdtomasi](https://github.com/fdtomasi)).

## Support or Contact
Check out our documentation or contact us:

* federico [dot] tomasi [at] dibris [dot] unige [dot] it
