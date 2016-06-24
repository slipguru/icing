.. _tutorial:

Quick start tutorial
====================
IGNET may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install ignet

    or

    $ easy_install ignet

Installation from sources
-------------------------
If you like to manually install IGNET, download the .zip or .tar.gz archive
from `<http://slipguru.github.io/ignet/>`_. Then extract it and move into the root directory::

    $ unzip slipguru-ignet-|release|.zip
    $ cd ignet-|release|/

or::

    $ tar xvf slipguru-ignet-|release|.tar.gz
    $ cd ignet-|release|/

Otherwise you can clone our `GitHub repository <https://github.com/slipguru/ignet>`_::

   $ git clone https://github.com/slipguru/ignet.git

From here, you can follow the standard Python installation step::

    $ python setup.py install

This tutorial assumes that you downloaded and extracted IGNET
source package which contains a ``examples/data`` directory with some data files (``.npy`` or ``.csv``) which will be used to show IGNET functionalities.

IGNET needs only an ingredient:

* ``input table`` of immunoglobulins

The path of ``input table`` is specified inside a ``configuration`` file, which must be passed as the only argument of the ``ig_run.py`` script.


Input data format
-----------------
Input data are assumed to be:

* tabular data stored in tab-separated ``.tab`` files (or comma-separated ``.csv`` files; see the Configuration File section on how to specify your preferred format for loading the table) presenting the variables header on the first row and the sample indexes on the first column.
The file must contain at least the following required columns:

* SEQUENCE_ID
* SUBSET (optional)
* MUT (optional)
* V_CALL
* J_CALL
* JUNCTION
* JUNCTION_LENGTH (optional)

The format of the file is the same which is returned from `HighV-QUEST`. See ``http://www.imgt.org/HighV-QUEST/`` for further information.


.. _configuration:

Configuration File
------------------
IGNET configuration file is a standard Python script. It is
imported as a module, then all the code is executed. In this file the user can define all the option needed to read the data.
A ``db_file`` filename is required. Note: to avoid problems, the path must be absolute.
Other options can be specified, regarding the file loading configuration:

* ``dialect`` is a string which is used to specify how values are separated in the database file. Only ``excel-tab`` and ``excel`` are supported by the standard library. If your database is a tab-delimited file (``.tab``), use ``excel-tab``. If the database is a comma-separated file (``.csv``), use ``excel``.
* ``subsets`` is a tuple or list of allowed immunoglobulin subsets to load. The database should contain a SUBSET column, and this variable is used to load from the database only those immunoglobulins which have a subset compatible. Names in ``subsets`` should be lowercase.
* ``mutation`` is a tuple or list composed by two float values. Only immunoglobulins which have a mutation level inside this range are loaded.
* ``apply_filter`` allows to specify the final filter which is used to load records from the database. The default filter in the configuration file allows to load records according to the previous statement. IGNET, however, allows a full customisation of the analysis. Expert users, therefore, can modify this function in order to load arbitrary records following their rules.

Optionally, a ``force_silhouette`` variable can be defined and set to True if, independently from the dimension of the distance matrix that will be produced, the user wants to perform a silhouette analysis on the data.

.. literalinclude:: ../../ignet/config.py
   :language: python

.. _experiment:

Experiment runner
-----------------
The ``ig_run.py`` script executes the IGNET main features, that is the definition of immunoglobulin clones. The prototype is the following:

    $ ig_run.py config.py

When launched, the script reads the record database from the filename specified in the ``config.py`` file , then it perform the analysis saving the results in a tree-like structure which has the current folder as root.

.. _analysis:

Results analysis
----------------
If the number of records analysed is acceptable, or if the user specified  script provides useful summaries and graphs from the results of the experiment. This script accepts as only parameter a result directory
already created::

    $ ig_analysis.py result-dir

The script produces a set of textual and graphical results. An output example obtained by one of the implemented pipelines is represented below.

.. image:: pca.png
   :scale: 80 %
   :alt: broken link

.. image:: kpca.png
   :scale: 80 %
   :alt: broken link

You can reproduce the example above specifying ``data_source.load('circles')`` in the configuration file.

Example dataset
----------------
An example dataset can be dowloaded :download:`here <TCGA-PANCAN-HiSeq-801x20531.tar.gz>`. The dataset is a random extraction of 801 samples (with dimension 20531) measuring RNA-Seq gene expression of patients affected by 5 different types of tumor: breast invasive carcinoma (BRCA), kidney renal clear cell carcinoma (KIRC), colon  (COAD), lung  (LUAD) and prostate adenocarcinoma (PRAD). The full dataset is maintained by The Cancer Genome Atlas Pan-Cancer Project [1] and we refer to the `original repository <https://www.synapse.org/#!Synapse:syn4301332>`_ for furher details.

Reference
----------------
[1] Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.
