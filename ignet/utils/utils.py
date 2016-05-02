import sys
import time
import datetime
import fcntl
import termios
import struct
import re, os

def junction_re(x, n='N'):
    return re.sub('[\.-]', n, str(x))

def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list or type(x) == np.ndarray else [x]

def _terminate(ps, e=''):
    '''Terminate processes in ps and exit the program. '''
    sys.stderr.write(e)
    sys.stderr.write('Terminating processes ...')
    for p in ps:
        p.terminate()
        p.join()
    sys.stderr.write('... done.\n')
    sys.exit()

### Time utilities ###
def get_time_from_seconds(seconds):
    """Transform seconds into formatted time string"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{02d}:{02d}:{02d}".format(h, m, s)

def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

### progress bar ###
try:
    TERMINAL_COLS = struct.unpack('hh',  fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, '1234'))[1]
except:
    TERMINAL_COLS = 50

def bold(msg):
    return u'\033[1m{}\033[0m'.format(msg)

def progress(current, total):
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
    return bold(prefix) + bar_start + bar + bar_end

def progressbar(i, max_i):
    sys.stdout.flush()
    sys.stdout.write('\r' + progress(i, max_i))
    if i >= max_i: sys.stdout.write('\n')
    sys.stdout.flush()
