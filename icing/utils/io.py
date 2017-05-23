#!/usr/bin/env python
"""Utilities for input/output operations.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

import csv
import pandas as pd
import sys

from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from itertools import ifilter, islice

from icing.externals.DbCore import parseAllele, gene_regex, junction_re
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


def get_max_mut(db_file, dialect='excel-tab', return_num_records=True):
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
        max_mut = -1
        num_records = 0
        for row in db_reader:
            num_records += 1
            aux = float(row['MUT'])
            if max_mut < aux:
                max_mut = aux

        if return_num_records:
            return max_mut, num_records
        return max_mut
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except Exception:
        sys.exit('ERROR:  File %s is invalid' % db_file)


def get_num_records(db_file, filt=None, dialect='excel-tab'):
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
    db = read_db(db_file, filt=filt, ig=False, dialect=dialect)
    for i, _ in enumerate(db):
        pass
    return i + 1


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
        row.append('CLONE_ICING')
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


def load_dataframe(db_file, dialect='excel-tab'):
    df = pd.read_csv(db_file, dialect=dialect)
    df = df.rename(columns=dict(zip(df.columns, df.columns.str.lower())))

    # df = df[df['functional'] == 'F']

    # parse v and j genes to speed up computation later
    df['v_gene_set'] = [set(
        parseAllele(x, gene_regex, 'set')) for x in df.v_call]
    df['v_gene_set_str'] = [str(set(
        parseAllele(x, gene_regex, 'set'))) for x in df.v_call]
    df['j_gene_set'] = [set(
        parseAllele(x, gene_regex, 'set')) for x in df.j_call]
    df['junc'] = [junction_re(x) for x in df.junction]
    df['aa_junc'] = [str(Seq(x, generic_dna).translate()) for x in df.junc]
    df['aa_junction_length'] = [len(x) for x in df.aa_junc]

    return df
