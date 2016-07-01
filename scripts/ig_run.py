#!/usr/bin/env python
"""Assign Ig sequences into clones."""

import os
import imp
import shutil
import argparse
import logging

from ignet.core.cloning import define_clones
from ignet.utils import extra
from ignet.utils import io

__author__ = 'Federico Tomasi'


def main(config_file):
    """Run ignet main features."""
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('config', config_path)

    # Load input file
    extra.set_module_defaults(config, {'subsets': (),
                                       'mutation': (0, 0),
                                       'apply_filter': None,
                                       'dialect': 'excel-tab',
                                       'exp_tag': 'debug',
                                       'output_root_folder': 'results',
                                       'force_silhouette': False,
                                       'sim_func_args': {}})

    db_iter = list(io.read_db(config.db_file, filt=config.apply_filter,
                              dialect=config.dialect))

    # Define logging file
    root = config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    filename = '_'.join(('ignet', config.exp_tag, extra.get_time()))
    logfile = os.path.join(root, filename+'.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    outfolder, clone_dict = define_clones(db_iter, exp_tag=filename, root=root,
                                          force_silhouette=config.force_silhouette,
                                          sim_func_args=config.sim_func_args)

    # Copy the ade_config just used into the outFolder
    shutil.copy(config_path, os.path.join(outfolder, 'config.py'))
    # Move the logging file into the outFolder
    shutil.move(logfile, outfolder)

    # Save clusters in a copy of the original database with a new column
    result_db = os.path.join(outfolder, 'db_file_clusters' + config.db_file[-4:])
    # shutil.copy(config.db_file, result_db)

    io.write_clusters_db(config.db_file, result_db, clone_dict, config.dialect)


if __name__ == '__main__':
    from ignet import __version__
    parser = argparse.ArgumentParser(description='ignet script for running '
                                                 'analysis.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v'+__version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='config.py')
    args = parser.parse_args()

    if args.create:
        import ignet
        std_config_path = os.path.join(ignet.__path__[0], 'config.py')
        # Check for .pyc
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]
        # Check if the file already exists
        if os.path.exists(args.configuration_file):
            parser.error("ignet configuration file already exists")
        # Copy the config file
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(args.configuration_file)
