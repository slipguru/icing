#!/usr/bin/env python
"""Assign Ig sequences into clones."""

import sys

from ignet.core.cloning import run

__author__ = 'Federico Tomasi'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("USAGE: ig_run.py <CONFIG_FILE>")
        sys.exit(-1)
    run(sys.argv[1])
