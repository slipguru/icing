import cloning as cl

# file loading, return an iterator
in_file = cl.readDbFile('../new_seqs/B4_db-pass.tab_CON-FUN-N_new_ord-MUT.tab')

# group sequences with the same V J and junction length
ig_dic = cl.indexJunctions(in_file, action='set')

l = []
for i in ig_dic:
    l.append(cl.distanceClones(ig_dic.get(i)))
