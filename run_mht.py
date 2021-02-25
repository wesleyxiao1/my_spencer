import pandas
import os

in_dir = 'data/det_formatted/'
out_dir = 'data/tracking_results/'
if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
for f in os.listdir(in_dir):
        if 'group-01' in f:
                infile = in_dir + f
                outfile = out_dir + f
                cmd = "python3 -m openmht " + infile + " " + outfile + " openmht/params.txt"
                os.system(cmd)