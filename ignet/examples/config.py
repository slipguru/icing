# Configuration file for define_clones_network.

# db_file = '/home/fede/Dropbox/projects/davide/new_seqs/B4_db-pass.tab_CON-FUN-N_new.tab'
# db_file = '/home/fede/Dropbox/projects/ig_davide/new_seqs/b8000.tab'

db_file = "/home/fede/Dropbox/projects/davide" +\
          "/new_seqs/B4_db-pass.tab_CON-FUN-N_new_ord-MUT-Naive.tab"

subsets = ('n', 'naive')
mutation = (0, 0)

apply_filter = lambda x: x.subset.lower() in subsets and mutation[0] <= x.mut <= mutation[1]
