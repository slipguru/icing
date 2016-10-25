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

from icing.core.cloning import define_clones
from icing.core.learning_function import generate_correction_function
from icing.utils import extra
from icing.utils import io

__author__ = 'Federico Tomasi'


def main(config_file):
    """Run icing main features."""
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('config', config_path)

    # Load input file
    extra.set_module_defaults(config, {'subsets': (),
                                       'mutation': (0, 0),
                                       'apply_filter': None,
                                       'max_records': None,
                                       'dialect': 'excel-tab',
                                       'exp_tag': 'debug',
                                       'output_root_folder': 'results',
                                       'force_silhouette': False,
                                       'sim_func_args': {},
                                       'threshold': 0.0536,
                                       'verbose': False,
                                       'learning_function_quantity': 0.15})
    # Define logging file
    root = config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    filename = '_'.join(('icing', config.exp_tag, extra.get_time()))
    logfile = os.path.join(root, filename + '.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    if config.verbose:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    logging.info("Start analysis ...")
    db_iter = list(io.read_db(config.db_file,
                              filt=config.apply_filter,
                              dialect=config.dialect,
                              max_records=config.max_records))
    logging.info("Database loaded ({} records)".format(len(db_iter)))

    if config.sim_func_args.pop("correction_function", None) is None:
        logging.info("Generate correction function with {}% of records"
                     .format(config.learning_function_quantity))
        (config.sim_func_args['correction_function'],
         config.threshold) = \
            generate_correction_function(
                config.db_file, quantity=config.learning_function_quantity,
                sim_func_args=config.sim_func_args)

    outfolder, clone_dict = define_clones(
        db_iter, exp_tag=filename, root=root,
        force_silhouette=config.force_silhouette,
        sim_func_args=config.sim_func_args,
        threshold=config.threshold)

    # Copy the ade_config just used into the outFolder
    shutil.copy(config_path, os.path.join(outfolder, 'config.py'))
    # Move the logging file into the outFolder
    shutil.move(logfile, outfolder)

    # Save clusters in a copy of the original database with a new column
    result_db = os.path.join(outfolder, 'db_file_clusters' + config.db_file[-4:])
    # shutil.copy(config.db_file, result_db)

    io.write_clusters_db(config.db_file, result_db, clone_dict, config.dialect)
    logging.info("Clusters correctly created and written on file. "
                 "Now run ici_analysis.py on the results folder.")


if __name__ == '__main__':
    from icing import __version__
    parser = argparse.ArgumentParser(description='icing script for running '
                                                 'analysis.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v'+__version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='config.py')
    args = parser.parse_args()

    if args.create:
        import icing
        std_config_path = os.path.join(icing.__path__[0], 'config.py')
        # Check for .pyc
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]
        # Check if the file already exists
        if os.path.exists(args.configuration_file):
            parser.error("icing configuration file already exists")
        # Copy the config file
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(args.configuration_file)
