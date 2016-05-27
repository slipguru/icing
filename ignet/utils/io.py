import numpy as np
import csv
import itertools
import sys

from changeo.DbCore import IgRecord

def read_db(db_file, filt=None, ig=True):
    """
    Reads database files

    Arguments:
    db_file = a tab delimited database file
    ig = if True convert fields to an IgRecord

    Returns:
    a database record iterator
    """
    try:
        f = open(db_file, 'rb')
        db_reader = csv.DictReader(f, dialect='excel-tab')
        db_reader.fieldnames = [n.strip().upper() for n in db_reader.fieldnames]
        db_iter = itertools.ifilter(filt, (IgRecord(r) for r in db_reader)) if ig else db_reader
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except Exception as e:
        sys.exit('ERROR: {}'.format(e))
    return db_iter

def get_max_mut(db_file):
    """
    Get the maximum amount of mutations in a database file

    Arguments:
    db_file = a tab delimited database file

    Returns:
    the maximum mutation level
    """
    try:
        f = open(db_file, 'rb')
        db_reader = csv.DictReader(f, dialect='excel-tab')
        return max(float(row['MUT']) for row in db_reader)
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except:
        sys.exit('ERROR:  File %s is invalid' % db_file)
