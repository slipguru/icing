import sys
import csv
import itertools

from ..externals.DbCore import IgRecord


def read_db(db_file, filt=None, ig=True, dialect='excel-tab'):
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
        db_reader.fieldnames = [n.strip().upper() for n in db_reader.fieldnames]
        db_iter = itertools.ifilter(filt, (IgRecord({k.upper(): v.upper() for k, v in r.iteritems()}) for r in db_reader)) if ig else db_reader
    except IOError:
        sys.exit('ERROR: File %s cannot be read' % db_file)
    except Exception as e:
        sys.exit('ERROR: {}'.format(e))
    return db_iter


def get_max_mut(db_file):
    """Get the maximum amount of mutations in a database file.

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

    Returns
    -------
    None
    """
    with open(db_file, 'r') as csvinput:
        with open(result_db, 'w') as csvoutput:
            reader = csv.reader(csvinput, dialect=dialect)
            writer = csv.writer(csvoutput, dialect=dialect, lineterminator='\n')

            all_list = []
            row = next(reader)
            row.append('CLONE')
            all_list.append(row)
            print([n.strip().upper() for n in row])
            indexid = [n.strip().upper() for n in row].index("SEQUENCE_ID")

            for row in reader:
                row.append(clones.get(row[indexid], ''))
                all_list.append(row)

            writer.writerows(all_list)
