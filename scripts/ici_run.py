#!/usr/bin/env python
"""Assign Ig sequences into clones.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

import os
import imp
import shutil
import argparse
import logging
import time
import numpy as np
import pandas as pd

import icing
from icing import __version__
from icing.utils import extra
from icing.utils import io

from icing.inference import DefineClones
from icing.learner import LearningFunction

__author__ = 'Federico Tomasi'


def init_logger(filename, root, verbose):
    """Initialise logger."""
    logfile = os.path.join(root, filename + '.log')
    logging.shutdown()
    root_logger = logging.getLogger()
    for _ in list(root_logger.handlers):
        root_logger.removeHandler(_)
        _.flush()
        _.close()
    for _ in list(root_logger.filters):
        root_logger.removeFilter(_)
        _.flush()
        _.close()

    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(asctime)-15s): %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if verbose else logging.ERROR)
    stream_handler.setFormatter(
        logging.Formatter('%(levelname)s (%(asctime)-15s): %(message)s'))

    root_logger.addHandler(stream_handler)
    return logfile


def main(config_file):
    """Run icing main features."""
    # Load the configuration file
    config_file = os.path.abspath(config_file)
    config = imp.load_source('config', config_file)

    # Load input file
    extra.set_module_defaults(config, {
        'subsets': (), 'mutation': (0, 0), 'apply_filter': None,
        'max_records': None, 'dialect': 'excel-tab', 'exp_tag': 'debug',
        'output_root_folder': 'results', 'force_silhouette': False,
        'sim_func_args': {}, 'threshold': None, 'verbose': False,
        'learning_function_quantity': 0.3,
        'learning_function_order': 3,
        'igsimilarity': None,
        'clustering': 'ap', 'clustering_method': None,
        'compute_similarity': True,
        'correct_by': None})

    # Define logging file
    root = config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    for db_file, exp_tag in zip(
            [config.db_file] if not isinstance(config.db_file, list)
            else config.db_file, [config.exp_tag] if not
            isinstance(config.exp_tag, list) else config.exp_tag):
        filename = '_'.join(('icing', exp_tag, extra.get_time(True)))
        logfile = init_logger(filename, root, config.verbose)
        logging.critical("Start analysis for %s", exp_tag)
        tic = time.time()
        db_iter = list(io.read_db(db_file,
                                  filt=config.apply_filter,
                                  dialect=config.dialect,
                                  max_records=config.max_records))
        logging.info("Database loaded (%i records)", len(db_iter))

        import copy
        igsimilarity_local = copy.deepcopy(config.igsimilarity)
        # local_sim_func_args = config.sim_func_args.copy()
        alpha_plot, threshold = None, None
        if igsimilarity_local.correct and igsimilarity_local.correct_by is None:
            record_quantity = np.clip(config.learning_function_quantity, 0, 1)
            logging.info("Generate correction function with %.2f%% of records",
                         record_quantity * 100)
            # func_args_copy = local_sim_func_args.copy()
            # func_args_copy.pop('clustering', 'ap')
            # (local_sim_func_args['correction_function'], threshold,
            #  alpha_plot) = generate_correction_function(
            #      db_file, quantity=record_quantity,
            #      sim_func_args=func_args_copy,
            #      order=config.learning_function_order, root=root)
            records = pd.read_csv(db_file, dialect='excel-tab')
            learner = LearningFunction(
                db_file, quantity=record_quantity,
                igsimilarity=igsimilarity_local,
                order=config.learning_function_order, root=root).fit(records)
            learning_function = learner.learning_function
            igsimilarity_local.correct_by = learning_function
            threshold = learner.threshold_naive

        if config.threshold is None and threshold is None:
            # no correction function and no threshold specified in config
            threshold = .05
        elif config.threshold is not None:
            threshold = config.threshold
        logging.info("Start define_clones function ...")
        # clustering = local_sim_func_args.pop('clustering', 'ap')
        # outfolder, clone_dict = define_clones(
        #     db_iter, exp_tag=filename, root=root,
        #     method=clustering,
        #     sim_func_args=local_sim_func_args,
        #     threshold=threshold, db_file=db_file)
        clones = DefineClones(
            tag=filename, root=root, cluster=config.clustering,
            igsimilarity=igsimilarity_local, threshold=threshold,
            compute_similarity=config.compute_similarity,
            clustering=config.clustering_method).fit(
                db_iter, db_name=db_file)
        outfolder, clone_dict = clones.output_folder_, clones.clone_dict_

        try:
            # Copy the config just used in the output folder
            shutil.copy(config_file, os.path.join(outfolder, 'config.py'))
            # Move the logging file into the outFolder
            shutil.move(logfile, outfolder)
            if alpha_plot is not None:
                shutil.move(alpha_plot, outfolder)
        except BaseException as msg:
            logging.critical(msg)

        # Save clusters in a copy of the original database with a new column
        result_db = os.path.join(
            outfolder, 'db_file_clusters.' + db_file.split(".")[-1])
        io.write_clusters_db(db_file, result_db, clone_dict, config.dialect)
        # config.sim_func_args["correction_function"] = None  # bugfix
        logging.info("Clusters correctly created and written on file. "
                     "Now run ici_analysis.py on the results folder.")
        logging.info("Run completed in %s",
                     extra.get_time_from_seconds(time.time() - tic))


def init_run():
    """Parse commands and start icing core."""
    parser = argparse.ArgumentParser(description='icing script for running '
                                                 'analysis.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='config.py')
    args = parser.parse_args()

    if args.create:
        std_config_path = os.path.join(icing.__path__[0], 'config.py')
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]

        if os.path.exists(args.configuration_file):
            parser.error("icing configuration file already exists")

        # Copy the config file
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(args.configuration_file)


if __name__ == '__main__':
    init_run()
