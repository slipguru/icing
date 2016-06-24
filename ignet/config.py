"""Default configuration file for ignet."""

exp_tag = 'debug'
output_root_folder = 'results'

db_file = "/path/to/database_file.tab"
dialect = "excel-tab"
subsets = ('n', 'naive')
mutation = (0, 0)

apply_filter = lambda x: x.subset.lower() in subsets and mutation[0] <= x.mut <= mutation[1]
