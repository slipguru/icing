"""Default configuration file for ignet.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

exp_tag = 'debug'
output_root_folder = 'results'

db_file = "/path/to/database_file.tab"
dialect = "excel-tab"
subsets = ('n', 'naive')
mutation = (0, 0)

apply_filter = lambda x: x.subset.lower() in subsets and mutation[0] <= x.mut <= mutation[1]
sim_func_args = {'method': 'jaccard'}

# Analysis options
file_format = 'png'
plotting_context = 'notebook'
force_silhouette = False
