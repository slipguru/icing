"""Example of an ICING configuration file.

Author: Federico Tomasi
Copyright (c) 2017, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import os
current_folder = os.path.dirname(os.path.abspath(__file__))

exp_tag = 'example'
output_root_folder = 'icing_example_result'

# this can be a list or a single input file
db_file = [
    'data/clones_95.tab',
    # 'data/clones_100.1.tab',
    # 'data/clones_100.2.tab'
]

db_file = [os.path.join(current_folder, x) for x in db_file]
exp_tag = [x.split('/')[-1] for x in db_file]

# type of input, if in excel format or excel-tab
dialect = "excel-tab"

# Percentual of IG sequences on which to calculate the correction function
learning_function_quantity = 1

# Parameter of the function for IG similarity
sim_func_args = {
    'vj_weight': 0,
    'sk_weight': 1,
    # turn the following on to avoid correction function
    # 'correction_function': lambda x: 1,

    # string kernel parameters.
    'model': 'sk',
    'ssk_params': {
        'min_kn': 3, 'max_kn': 9,
        'lamda': .25, 'check_min_length': 1
    },
    # turn this on to use hamming similarity
    # 'model': 'ham',

    # clustering: ap or hc
    'clustering': 'ap',

    # tolerance on HCDR3 length
    'tol': 6
}
# This is ignored with AP clustering
# threshold = .025
