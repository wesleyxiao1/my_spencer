import pandas
import time
import os

def format_data(data, format_file):
    header = "frame,u,v"
    with open(format_file, "w") as file:
        file.write(header + "\n")
    
    with open(format_file, "a+") as file:
        for _, row in data.iterrows():
            x = row.x
            y = row.y
            frame = int(row.frameID)
            out = ','.join([str(frame), str(x), str(y)])
            file.write(out + "\n")

def time_tracker(infile, outfile):
    cmd = "python3 -m openmht " + infile + " " + outfile + " openmht/params.txt"
    start_time = time.time()
    # run openmht
    os.system(cmd)
    end_time = time.time()
    elapsed = end_time - start_time
    return elapsed

def get_data(n):
    df = pandas.read_csv('data/group_track_data.csv')
    df = df.loc[((df.dataset == 'test') & (df.segment_num == 1))]

    df = df.head(n)
    return df

results_file = "benchmarking/results.csv"
with open(results_file, "w") as file:
    file.write("n,time\n")

dets = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
for n in dets:
    data = get_data(n)
    
    ifile = "benchmarking/det_" + str(n) + "_in.csv"
    ofile = "benchmarking/det_" + str(n) + "_out.csv"
    format_data(data, ifile)
    elapsed = time_tracker(ifile, ofile)

    with open(results_file, "a") as file:
        file.write(str(n)+","+str(elapsed)+"\n")

    print("n = ", n)
    print("elapsed = ", elapsed)
    print("\n")