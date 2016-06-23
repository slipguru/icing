#!/usr/bin/env python
"""Record file for immunoglobulin data type.

This file is adapted from Change-O modules.
See changeo.DbCore for the original version.
Reference: http://changeo.readthedocs.io/en/latest/
"""
import csv
import os
import re
import sys
from itertools import product
from collections import OrderedDict
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC


__author__ = 'Federico Tomasi'
# __copyright__ = 'Copyright 2014 Kleinstein Lab, Yale University. All rights reserved.'
# __license__ = 'Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported'
# __version__ = '0.2.4'
# __date__ = '2015.08.18'


# Regular expression globals
allele_regex = re.compile(r'((IG[HLK]|TR[ABGD])([VDJ][A-Z0-9]+[-/\w]*[-\*][\.\w]+))')
gene_regex = re.compile(r'((IG[HLK]|TR[ABGD])([VDJ][A-Z0-9]+[-/\w]*))')
family_regex = re.compile(r'((IG[HLK]|TR[ABGD])([VDJ][A-Z0-9]+))')

v_allele_regex = re.compile(r'((IG[HLK]|TR[ABGD])V[A-Z0-9]+[-/\w]*[-\*][\.\w]+)')
d_allele_regex = re.compile(r'((IG[HLK]|TR[ABGD])D[A-Z0-9]+[-/\w]*[-\*][\.\w]+)')
j_allele_regex = re.compile(r'((IG[HLK]|TR[ABGD])J[A-Z0-9]+[-/\w]*[-\*][\.\w]+)')


# TODO:  might be better to just use the lower case column name as the member variable name. can use getattr and setattr.
class IgRecord:
    """
    A class defining a V(D)J germline sequence alignment
    """
    # Mapping of member variables to column names
    _key_map = {'id': 'SEQUENCE_ID',
                'v_call': 'V_CALL',
                'v_call_geno': 'V_CALL_GENOTYPED',
                'd_call': 'D_CALL',
                'j_call': 'J_CALL',
                'seq_input': 'SEQUENCE_INPUT',
                'seq_vdj': 'SEQUENCE_VDJ',
                'seq_imgt': 'SEQUENCE_IMGT',
                'junction': 'JUNCTION',
                'junctionaa': 'JUNCTIONAA',
                'functional': 'FUNCTIONAL',
                'in_frame': 'IN_FRAME',
                'stop': 'STOP',
                'mutated_invariant': 'MUTATED_INVARIANT',
                'indels': 'INDELS',
                'v_seq_start': 'V_SEQ_START',
                'v_seq_length': 'V_SEQ_LENGTH',
                'v_germ_start': 'V_GERM_START',
                'v_germ_length': 'V_GERM_LENGTH',
                'n1_length': 'N1_LENGTH',
                'd_seq_start': 'D_SEQ_START',
                'd_seq_length': 'D_SEQ_LENGTH',
                'd_germ_start': 'D_GERM_START',
                'd_germ_length': 'D_GERM_LENGTH',
                'n2_length': 'N2_LENGTH',
                'j_seq_start': 'J_SEQ_START',
                'j_seq_length': 'J_SEQ_LENGTH',
                'j_germ_start': 'J_GERM_START',
                'j_germ_length': 'J_GERM_LENGTH',
                'junction_length': 'JUNCTION_LENGTH',
                'v_score': 'V_SCORE',
                'v_identity': 'V_IDENTITY',
                'v_evalue': 'V_EVALUE',
                'j_score': 'J_SCORE',
                'j_identity': 'J_IDENTITY',
                'j_evalue': 'J_EVALUE',
                'mut': 'MUT',
                'subset': 'SUBSET'}

    # Mapping of column names to member variables
    _field_map = {v: k for k, v in _key_map.iteritems()}

    # Mapping of member variables to parsing functions
    _parse_map = {'id': '_identity',
                  'v_call': '_identity',
                  'v_call_geno': '_identity',
                  'd_call': '_identity',
                  'j_call': '_identity',
                  'seq_input': '_sequence',
                  'seq_vdj': '_sequence',
                  'seq_imgt': '_sequence',
                  'junction': '_sequence',
                  'junctionaa': '_sequence',
                  'functional': '_logical',
                  'in_frame': '_logical',
                  'stop': '_logical',
                  'mutated_invariant': '_logical',
                  'indels': '_logical',
                  'v_seq_start': '_integer',
                  'v_seq_length': '_integer',
                  'v_germ_start': '_integer',
                  'v_germ_length': '_integer',
                  'n1_length': '_integer',
                  'd_seq_start': '_integer',
                  'd_seq_length': '_integer',
                  'd_germ_start': '_integer',
                  'd_germ_length': '_integer',
                  'n2_length': '_integer',
                  'j_seq_start': '_integer',
                  'j_seq_length': '_integer',
                  'j_germ_start': '_integer',
                  'j_germ_length': '_integer',
                  'junction_length': '_integer',
                  'v_score': '_float',
                  'v_identity': '_float',
                  'v_evalue': '_float',
                  'j_score': '_float',
                  'j_identity': '_float',
                  'j_evalue': '_float',
                  'mut': '_float',
                  'subset': '_identity'} # added field MUT in IgRecord. --toma

    _logical_parse = {'F':False, 'T':True, 'TRUE':True, 'FALSE':False, 'NA':None, 'None':None}
    _logical_deparse = {False:'F', True:'T', None:'None'}

    # Private methods
    @staticmethod
    def _identity(v, deparse=False):
        return v

    @staticmethod
    def _logical(v, deparse=False):
        if not deparse:
            try:  return IgRecord._logical_parse[v]
            except:  return None
        else:
            try:  return IgRecord._logical_deparse[v]
            except:  return ''

    @staticmethod
    def _integer(v, deparse=False):
        if not deparse:
            try:  return int(v)
            except:  return None
        else:
            try:  return str(v)
            except:  return ''

    @staticmethod
    def _float(v, deparse=False):
        if not deparse:
            try:  return float(v)
            except:  return None
        else:
            try:  return str(v)
            except:  return ''

    @staticmethod
    def _sequence(v, deparse=False):
        if not deparse:
            try:  return Seq(v, IUPAC.ambiguous_dna)
            except:  return None
        else:
            try:  return str(v)
            except:  return ''

    # Initializer
    #
    # Arguments:  row = dictionary of {field:value} data
    #             genotyped = if True assign v_call from genotyped field
    # Returns:    IgRecord
    def __init__(self, row, genotyped=True):
        required_keys = ('id',)
        optional_keys = (x for x in IgRecord._parse_map if x not in required_keys)

        # Not ideal. Will place V_CALL_GENOTYPED in annotations
        if not genotyped and 'v_call_geno' in optional_keys:
            del optional_keys['v_call_geno']

        try:
            for k in required_keys:
                f = getattr(IgRecord, IgRecord._parse_map[k])
                setattr(self, k, f(row.pop(IgRecord._key_map[k])))
        except:
            sys.exit('ERROR:  Input must contain valid %s values' \
                     % ','.join([IgRecord._key_map[k] for k in required_keys]))

        # Defined optional logical values
        for k in optional_keys:
            f = getattr(IgRecord, IgRecord._parse_map[k])
            setattr(self, k, f(row.pop(IgRecord._key_map[k], None)))

        # Add remaining elements as annotations dictionary
        self.annotations = row

    # Get a field value by column name and return it as a string
    #
    # Arguments:  field = column name
    # Returns:    value in the field as a string
    def getField(self, field):
        if field in IgRecord._field_map:
            v = getattr(self, IgRecord._field_map[field])
        elif field in self.annotations:
            v = self.annotations[field]
        else:
            return None

        if isinstance(v, str):
            return v
        else:
            return str(v)

    def getVSeq(self):
        # _seq_vdj = self.getField('SEQUENCE_VDJ')
        _seq_vdj = self.getField('SEQUENCE_IMGT')
        # _idx_v = int(self.getField('V_SEQ_LENGTH'))
        return _seq_vdj[:312] # 312: V length without cdr3

    # Get a field value converted to a Seq object by column name
    #
    # Arguments:  field = column name
    # Returns:    value in the field as a Seq object
    def getSeqField(self, field):
        if field in IgRecord._field_map:
            v = getattr(self, IgRecord._field_map[field])
        elif field in self.annotations:
            v = self.annotations[field]
        else:
            return None

        if isinstance(v, Seq):
            return v
        elif isinstance(v, str):
            return Seq(v, IUPAC.ambiguous_dna)
        else:
            return None

    # Returns: dictionary of the namespace
    def toDict(self):
        d = {}
        n = self.__dict__
        for k, v in n.iteritems():
            if k == 'annotations':
                d.update({i.upper():j for i, j in n['annotations'].iteritems()})
            else:
                f = getattr(IgRecord, IgRecord._parse_map[k])
                d[IgRecord._key_map[k]] = f(v, deparse=True)
        return d

    # Methods to get multiple allele, gene and family calls
    #
    # Arguments:  calls = iterable of calls to get; one or more of ('v','d','j')
    #             actions = one of ('first','set')
    # Returns:    list of requested calls in order
    def getAlleleCalls(self, calls, action='first'):
        vdj = {'v': self.getVAllele(action),
               'd': self.getDAllele(action),
               'j': self.getJAllele(action)}
        return [vdj[k] for k in calls]

    def getGeneCalls(self, calls, action='first'):
        vdj = {'v':self.getVGene(action),
               'd':self.getDGene(action),
               'j':self.getJGene(action)}
        return [vdj[k] for k in calls]

    def getFamilyCalls(self, calls, action='first'):
        vdj = {'v':self.getVFamily(action),
               'd':self.getDFamily(action),
               'j':self.getJFamily(action)}
        return [vdj[k] for k in calls]

    # Individual allele, gene and family getter methods
    #
    # Arguments:  actions = one of ('first','set')
    # Returns:    call as a string
    def getVAllele(self, action='first'):
        # TODO: this can't distinguish empty value ("") from missing field (no column)
        x = self.v_call_geno if self.v_call_geno is not None else self.v_call
        return parseAllele(x, allele_regex, action)

    def getDAllele(self, action='first'):
        return parseAllele(self.d_call, allele_regex, action)

    def getJAllele(self, action='first'):
        return parseAllele(self.j_call, allele_regex, action)

    def getVGene(self, action='first'):
        return parseAllele(self.v_call, gene_regex, action)

    def getDGene(self, action='first'):
        return parseAllele(self.d_call, gene_regex, action)

    def getJGene(self, action='first'):
        return parseAllele(self.j_call, gene_regex, action)

    def getVFamily(self, action='first'):
        return parseAllele(self.v_call, family_regex, action)

    def getDFamily(self, action='first'):
        return parseAllele(self.d_call, family_regex, action)

    def getJFamily(self, action='first'):
        return parseAllele(self.j_call, family_regex, action)


class DbData:
    """
    A class defining IgRecord data objects for worker processes
    """
    # Instantiation
    def __init__(self, key, records):
        self.id = key
        self.data = records
        self.valid = (key is not None and records is not None)

    # Boolean evaluation
    def __nonzero__(self):
        return self.valid

    # Length evaluation
    def __len__(self):
        if isinstance(self.data, IgRecord):
            return 1
        elif self.data is None:
            return 0
        else:
            return len(self.data)


class DbResult:
    """
    A class defining IgRecord result objects for collector processes
    """
    # Instantiation
    def __init__(self, key, records):
        self.id = key
        self.data = records
        self.results = None
        self.valid = False
        self.log = OrderedDict([('ID', key)])
        #if isinstance(values, list):
        #    for v in values:  setattr(self, v, None)
        #else:
        #    setattr(self, values, None)

    # Boolean evaluation
    def __nonzero__(self):
        return self.valid

    # Length evaluation
    def __len__(self):
        if isinstance(self.results, IgRecord):
            return 1
        elif self.results is None:
            return 0
        else:
            return len(self.results)

    # Set data_count to number of data records
    @property
    def data_count(self):
        if isinstance(self.data, IgRecord):
            return 1
        elif self.data is None:
            return 0
        else:
            return len(self.data)


# TODO:  might be cleaner as getAllele(), getGene(), getFamily()
def parseAllele(alleles, regex, action='first'):
    """
    Extract alleles from strings

    Arguments:  alleles = string with allele calls
                regex = compiled regular expression for allele match
                action = action to perform for multiple alleles;
                         one of ('first', 'set', 'list').
    Returns:    string of the allele for action='first';
                tuple of allele calls for 'set' or 'list' actions.
    """
    try:
        match = [x.group(0) for x in regex.finditer(alleles)]
    except:
        match = None

    if action == 'first':
        return match[0] if match else None
    elif action == 'set':
        return tuple(sorted(set(match))) if match else None
    elif action == 'list':
        return tuple(sorted(match)) if match else None
    else:
        return None


# TODO:  change to require output fields rather than in_file? probably better that way.
def getDbWriter(out_handle, in_file=None, add_fields=None, exclude_fields=None):
    """
    Opens a writer object for an output database file

    Arguments:
    out_handle = the file handle to write to
    in_file = the input filename to determine output fields from;
              if None do not define output fields from input file
    add_fields = a list of fields added to the writer not present in the in_file;
                 if None do not add fields
    exclude_fields = a list of fields in the in_file excluded from the writer;
                     if None do not exclude fields

    Returns:
    a writer object
    """
    # Get output field names from input file
    if in_file is not None:
        fields = (readDbFile(in_file, ig=False)).fieldnames
    else:
        fields = []
    # Add extra fields
    if add_fields is not None:
        if not isinstance(add_fields, list):  add_fields = [add_fields]
        fields.extend([f for f in add_fields if f not in fields])
    # Remove unwanted fields
    if exclude_fields is not None:
        if not isinstance(exclude_fields, list):  exclude_fields = [exclude_fields]
        fields = [f for f in fields if f not in exclude_fields]

    # Create writer
    try:
        # TODO:  THIS NEEDS TO BE FIXED, extrasaction='ignore' IS A WORKAROUND FOR ADDITIONS TO IgRecord
        db_writer = csv.DictWriter(out_handle, fieldnames=fields, dialect='excel-tab', extrasaction='ignore')
        db_writer.writeheader()
    except:
        sys.exit('ERROR:  File %s cannot be written' % out_handle.name)

    return db_writer


# TODO:  Need to close db_handle?
def readDbFile(db_file, ig=True):
    """
    Reads database files

    Arguments:
    db_file = a tab delimited database file
    ig = if True convert fields to an IgRecord

    Returns:
    a database record iterator
    """
    # Read and check file
    try:
        db_handle = open(db_file, 'rb')
        db_reader = csv.DictReader(db_handle, dialect='excel-tab')
        db_reader.fieldnames = [n.strip().upper() for n in db_reader.fieldnames]
        if ig:
            db_iter = (IgRecord(r) for r in db_reader)
        else:
            db_iter = db_reader
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except:
        sys.exit('ERROR:  File %s is invalid' % db_file)

    return db_iter


def countDbFile(db_file):
    """
    Counts the records in database files

    Arguments:
    db_file = a tab delimited database file

    Returns:
    the count of records in the database file
    """
    # Count records and check file
    try:
        with open(db_file) as db_handle:
            db_records = csv.reader(db_handle, dialect='excel-tab')
            for i, __ in enumerate(db_records):  pass
        db_count = i
    except IOError:
        sys.exit('ERROR:  File %s cannot be read' % db_file)
    except:
        sys.exit('ERROR:  File %s is invalid' % db_file)
    else:
        if db_count == 0:  sys.exit('ERROR:  File %s is empty' % db_file)

    return db_count




def formClusters(dists, link, distance):
    """
    Form clusters based on hierarchical clustering of input distance matrix
    with linkage type and cutoff distance
    :param dists: numpy matrix of distances
    :param link: linkage type for hierarchical clustering
    :param distance: distance at which to cut into clusters
    :return: list of cluster assignments
    """
    # Make distance matrix square
    dists = squareform(dists)
    # Compute linkage
    links = linkage(dists, link)

    # import matplotlib.pyplot as plt
    # from scipy.cluster import hierarchy
    # plt.figure(figsize=(15,5))
    # p = hierarchy.dendrogram(links)

    # Break into clusters based on cutoff
    clusters = fcluster(links, distance, criterion='distance')
    return clusters


def scoreDNA(a, b, n_score=None, gap_score=None):
    """
    Returns the score for a pair of IUPAC Ambiguous Nucleotide characters

    Arguments:
    a = first characters
    b = second character
    n_score = score for all matches against an N character;
              if None score according to IUPAC character identity
    gap_score = score for all matches against a [-, .] character;
                if None score according to IUPAC character identity

    Returns:
    score for the character pair
    """
    # Define ambiguous character translations
    IUPAC_trans = {'AGWSKMBDHV':'R', 'CTSWKMBDHV':'Y', 'CGKMBDHV':'S', 'ATKMBDHV':'W', 'GTBDHV':'K',
                   'ACBDHV':'M', 'CGTDHV':'B', 'AGTHV':'D', 'ACTV':'H', 'ACG':'V', 'ABCDGHKMRSTVWY':'N',
                   '-.':'.'}
    # Create list of tuples of synonymous character pairs
    IUPAC_matches = [p for k, v in IUPAC_trans.iteritems() for p in list(product(k, v))]

    # Check gap condition
    if gap_score is not None and (a in '-.' or b in '-.'):
        return gap_score

    # Check N-value condition
    if n_score is not None and (a == 'N' or b == 'N'):
        return n_score

    # Determine and return score for IUPAC match conditions
    # Symmetric and reflexive
    if a == b:
        return 1
    elif (a, b) in IUPAC_matches:
        return 1
    elif (b, a) in IUPAC_matches:
        return 1
    else:
        return 0


def scoreAA(a, b, n_score=None, gap_score=None):
    """
    Returns the score for a pair of IUPAC Extended Protein characters

    Arguments:
    a = first character
    b = second character
    n_score = score for all matches against an X character;
              if None score according to IUPAC character identity
    gap_score = score for all matches against a [-, .] character;
                if None score according to IUPAC character identity

    Returns:
    score for the character pair
    """
    # Define ambiguous character translations
    IUPAC_trans = {'RN':'B', 'EQ':'Z', 'LI':'J', 'ABCDEFGHIJKLMNOPQRSTUVWYZ':'X',
                   '-.':'.'}
    # Create list of tuples of synonymous character pairs
    IUPAC_matches = [p for k, v in IUPAC_trans.iteritems() for p in list(product(k, v))]

    # Check gap condition
    if gap_score is not None and (a in '-.' or b in '-.'):
        return gap_score

    # Check X-value condition
    if n_score is not None and (a == 'X' or b == 'X'):
        return n_score

    # Determine and return score for IUPAC match conditions
    # Symmetric and reflexive
    if a == b:
        return 1
    elif (a, b) in IUPAC_matches:
        return 1
    elif (b, a) in IUPAC_matches:
        return 1
    else:
        return 0



if __name__ == '__main__':
    """Print module information."""
    print('Version: %s %s %s' % (os.path.basename(__file__), __version__, __date__))
    print('Location: %s' % os.path.dirname(os.path.realpath(__file__)))
