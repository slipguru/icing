===========================================
ICING - Infer clones of immunoglobulin data
===========================================

:Release: |release|
:Homepage: http://www.slipguru.unige.it/Software/icing
:Repository: https://github.com/slipguru/icing

**ICING** is an implementation of an unsupervised learning technique
used to identify `clones`, which are groups of immunoglobulins which share
a common ancestor. In other words, immunoglobulins in the same group descend
from the same germline.
An overview of the problem may be found in the :ref:`overview` section
illustrating the intrisic difficulty of the problem, due to the type of data
at hand. Also, the method is designed to be used also in contexts where
the number of samples is very high. For this reason, the framework can be used
in combination with NGS technologies.

This library is composed by a set of Python scripts (described
in the :ref:`tutorial`) and a set of useful functions (described in
the :ref:`api` section) that could be used to manually read and/or analyze
high-throughput data extending/integrating the proposed pipeline.


User documentation
==================
.. toctree::
   :maxdepth: 2

   description.rst
   tutorial.rst


.. _api:

Public API
==========
.. toctree::
   :maxdepth: 1

   api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
