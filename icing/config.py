"""Default configuration file for icing.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

exp_tag = 'debug'
output_root_folder = 'results'

db_file = "/path/to/database_file.tab"
dialect = "excel-tab"

learning_function_quantity = 1
learning_function_order = 3
sim_func_args = {'method': 'jaccard', 'v_weight': 1, 'j_weight': 1}

# Analysis options
file_format = 'png'
plotting_context = 'notebook'
force_silhouette = False
