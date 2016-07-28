#!/usr/bin/env python
"""Perform the analysis on the results of `ig_run.py` script.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

from __future__ import print_function

import imp
import sys
import os
import time
import logging
import argparse
import cPickle as pkl
import gzip

from ignet.core import analyse_results
from ignet.utils import extra


def main(dumpfile):
    """Run ignet analysis."""
    # Load the configuration file
    config_path = os.path.dirname(dumpfile)
    config_path = os.path.join(os.path.abspath(config_path), 'config.py')
    config = imp.load_source('config', config_path)
    extra.set_module_defaults(config, {'file_format': 'pdf',
                                       'plotting_context': 'paper',
                                       'force_silhouette': False,
                                       'threshold': 0.0536})

    # Initialize the log file
    filename = 'results_' + os.path.basename(dumpfile)[0:-7]
    logfile = os.path.join(os.path.dirname(dumpfile), filename+'.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    # Load the results
    tic = time.time()
    print("\nUnpickling similarity matrix and clusters ...", end=' ')
    with gzip.open(dumpfile, 'r') as f:
        sm = pkl.load(f)
    with gzip.open(os.path.join(os.path.dirname(dumpfile),
                   os.path.basename(dumpfile).replace('similarity_matrix',
                                                      'clusters')), 'r') as f:
        clusters = pkl.load(f)
    print("done: {} s".format(extra.get_time_from_seconds(time.time() - tic)))

    # Analyze the pipelines
    analyse_results.analyse(sm=sm, labels=clusters,
                            root=os.path.dirname(dumpfile),
                            plotting_context=config.plotting_context,
                            file_format=config.file_format,
                            force_silhouette=config.force_silhouette,
                            threshold=config.threshold)

# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':
    from ignet import __version__
    parser = argparse.ArgumentParser(  # usage="%(prog)s RESULTS_DIR",
                                     description='ignet script for analysing '
                                                 'clustering.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("result_folder", help="specify results directory")
    args = parser.parse_args()
    root_folder = args.result_folder
    filename = [f for f in os.listdir(root_folder)
                if os.path.isfile(os.path.join(root_folder, f)) and
                f.endswith('.pkl.tz') and not f.endswith('_clusters.pkl.tz')]
    if not filename:
        sys.stderr.write("No .pkl file found in {}. Aborting...\n"
                         .format(root_folder))
        sys.exit(-1)

    # Run analysis
    # print("Starting the analysis of {}".format(filename))
    main(os.path.join(os.path.abspath(root_folder), filename[0]))
