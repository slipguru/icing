#!/usr/bin/env python
"""Utilities for input/output operations.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

import csv
import sys

from itertools import ifilter, islice
from icing.externals import IgRecord


def read_db(db_file, filt=None, ig=True, dialect='excel-tab',
            max_records=None):
    """Read a database file.

    Parameters
    ----------
    db_file : str
        A database file. Delimited according to `dialect`.
    filt : function, optional, default None
        Filter sequences in the database.
    ig : boolean, default True
        If True, convert fields to a IgRecord.
    dialect : ('excel-tab', 'excel')
        Dialect for csv.DictReader.

    Returns
    -------
    it : iterator
        Database record iterator.
    """
    try:
        f = open(db_file, 'rb')
        db_reader = csv.DictReader(f, dialect=dialect)
        db_reader.fieldnames = \
            [n.strip().upper() for n in db_reader.fieldnames]
        db_iter = islice(ifilter(filt, (IgRecord({k.upper(): v.upper()
                                                  for k, v in r.iteritems()})
                                        for r in db_reader)),
                         None, max_records) if ig else db_reader
    except IOError:
        sys.exit('ERROR: File %s cannot be read' % db_file)
    except Exception as e:
        sys.exit('ERROR: {}'.format(e))
    return db_iter


def get_max_mut(db_file, dialect='excel-tab'):
    """Get the maximum amount of mutations in a database file.

    Parameters
    ----------
    db_file : str
        A database file. Delimited according to `dialect`.
    dialect : ('excel-tab', 'excel')
        Dialect for csv.DictReader.

    Returns
    -------
    max_mutation : float
        The maximum mutation level between all the records.
    """
    try:
        f = open(db_file, 'rb')
        db_reader = csv.DictReader(f, dialect=dialect)
        return max(float(row['MUT']) for row in db_reader)
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except Exception:
        sys.exit('ERROR:  File %s is invalid' % db_file)


def get_num_records(db_file, dialect='excel-tab'):
    """Get the number of records of a database file.

    Parameters
    ----------
    db_file : str
        A database file. Delimited according to `dialect`.
    dialect : ('excel-tab', 'excel')
        Dialect for csv.DictReader.

    Returns
    -------
    num_records : int
        The number of records.
    """
    # Count records and check file
    try:
        with open(db_file) as f:
            for i, __ in enumerate(csv.reader(f, dialect=dialect)):
                pass
            db_count = i
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except Exception:
        sys.exit('ERROR:  File %s is invalid' % db_file)
    return db_count


def write_clusters_db(db_file, result_db, clones, dialect='excel-tab'):
    """Write in a database file a new column, `clones`.

    Parameters
    ----------
    db_file : str
        A database file. Delimited according to `dialect`.
    result_db : str
        Output filename for the new database.
    clones : dict
        For each IgRecord id, its associated clone.
    dialect : ('excel-tab', 'excel')
        Dialect for csv.DictReader.

    """
    with open(db_file, 'r') as csvinput, open(result_db, 'w') as csvoutput:
        reader = csv.reader(csvinput, dialect=dialect)
        writer = csv.writer(csvoutput, dialect=dialect, lineterminator='\n')

        all_list = []
        row = next(reader)
        row.append('CLONE')
        all_list.append(row)
        indexid = [n.strip().upper() for n in row].index("SEQUENCE_ID")

        for row in reader:
            row.append(clones.get(row[indexid].upper(), ''))
            all_list.append(row)

        writer.writerows(all_list)


def load_dm_from_file(filename, index_col=0, header='infer',
                      ensure_symmetry=False):
    """Load a distance matrix."""
    ext = filename[-3:].lower()
    if ext == 'csv':
        import pandas as pd
        dm = pd.io.parsers.read_csv(filename, index_col=index_col,
                                    header=header).as_matrix()
    elif ext == 'npy':
        import numpy as np
        dm = np.load(filename)
    elif filename[-7:] == '.pkl.tz':
        import cPickle as pkl
        import gzip
        with gzip.open(filename, 'r') as f:
            dm = pkl.load(f)

    if ensure_symmetry:
        from icing.utils.extra import ensure_symmetry
        dm = ensure_symmetry(dm)

    return dm
