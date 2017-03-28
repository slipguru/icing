# ICING - Infer clones of immunoglobulin data

A Python package to clonal relate immunoglobulins.

**ICING** is an implementation of an unsupervised learning technique
used to identify `clones`, which are groups of immunoglobulins which share
a common ancestor. In other words, immunoglobulins in the same group descend
from the same germline.
Also, the method is designed to be used also in contexts where
the number of samples is very high. For this reason, the framework can be used
in combination with NGS technologies.

## Quickstart
First of all, download ICING from the Python Package Index by using package manager pip
with
```bash
$ pip install icing
```
or clone it from our Github repository with
```bash
$ git clone https://github.com/slipguru/icing
```
and then install it with
```bash
$ python setup.py build_ext --inplace install
```
If you cloned the repository, navigate under `icing/examples` directory. There, run an example with
```bash
$ ici_run.py config_example.py
```
The output should be in the form of
```bash
CRITICAL (2017-02-28 12:52:28,564): Start analysis for clones_95.tab
CRITICAL (2017-02-28 12:52:44,282): Number of clones: 104
```

If you want to produce some plots for the analysis, run
```bash
$ ici_analysis.py icing_example_result/icing_clones_95.tab_<today date> .
```
and that's it.

### "Ok, now I want to use it for real."
Great!

First of all, create a new configuration file with the `-c` command.
```bash
$ ici_run.py -c config.py
```
Modify the configuration file to specify, for example, the path of your input files and non-standard ICING settings.

Run the icing core with
```bash
$ ici_run.py config.py
```
which will produce an output folder in `<current_path>/results/icing_<date_of_today>`.

Now analyse the result
```bash
$ ici_analysis.py results/output_folder/
```
to produce plots related to your analysis.
Done!

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
