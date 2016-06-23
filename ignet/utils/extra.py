#!/usr/bin/env python
"""Utility functions for ignet."""

import sys
import time
import datetime
import fcntl
import termios
import struct
import re
import os
import numpy as np

# class Counter(object):
#     """Counter which contains the lock. Atomic update"""
#
#     def __init__(self, initval=0):
#         self.val = mp.RawValue('i', initval)
#         self.lock = mp.Lock()
#
#     def increment(self, n=1):
#         with self.lock:
#             self.val.value += n
#
#     def value(self):
#         with self.lock:
#             return self.val.value
# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i+n]


def split_list(l, n):
    """Split list in n chunks."""
    for i in xrange(n):
        yield l[i*n:(i+1)*n]


def junction_re(x, n='N', filt='[\.-/]'):
    """Convert filt char in n in x."""
    return re.sub(filt, n, str(x))


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] if type(x) in (list, np.ndarray) else [x]


def _terminate(ps, e=''):
    """Terminate processes in ps and exit the program."""
    sys.stderr.write(e)
    sys.stderr.write('Terminating processes ...')
    for p in ps:
        p.terminate()
        p.join()
    sys.stderr.write('... done.\n')
    sys.exit()


# def get_time_from_seconds(seconds):
#     """Transform seconds into formatted time string"""
#     m, s = divmod(seconds, 60)
#     h, m = divmod(m, 60)
#     return "{02d}:{02d}:{02d}".format(h, m, s)


def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')


def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


# progress bar
try:
    TERMINAL_COLS = struct.unpack('hh',  fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, '1234'))[1]
except:
    TERMINAL_COLS = 50


def _bold(msg):
    return u'\033[1m{}\033[0m'.format(msg)


def _progress(current, total):
    prefix = '%d / %d' % (current, total)
    bar_start = ' ['
    bar_end = '] '

    bar_size = TERMINAL_COLS - len(prefix + bar_start + bar_end)
    try:
        amount = int(current / (total / float(bar_size)))
    except (ZeroDivisionError):
        amount = 0
    remain = bar_size - amount

    bar = '=' * amount + ' ' * remain
    return _bold(prefix) + bar_start + bar + bar_end


def progressbar(i, max_i):
    """Print a progressbar at time i, where i goes from 0 to max_i."""
    sys.stdout.flush()
    sys.stdout.write('\r' + _progress(i, max_i))
    if i >= max_i:
        sys.stdout.write('\n')
    sys.stdout.flush()
