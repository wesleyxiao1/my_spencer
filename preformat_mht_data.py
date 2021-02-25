import pandas
import argparse
import os

# Takes in the group detections and formats it for MHT
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")

    return parser.parse_args()

def format_normal(df, output_file, input_file):
    df = pandas.read_csv(input_file)
    header = "frame,u,v"
    with open(output_file, "w") as file:
        file.write(header + "\n")
    
    with open(output_file, "a+") as file:
        for _, row in df.loc[df.frameID > 18002].iterrows():
            x = row.x
            y = row.y
            frame = int(row.frameID)
            out = ','.join([str(frame), str(x), str(y)])
            file.write(out + "\n")

def format_detections():
    group = 1
    seg = 0
    s = 1
    n = 50
    out_dir = "data/det_formatted/"
    in_dir = "data/eval/test/dets/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    header = "frame,u,v"
    for f in os.listdir(in_dir):
        df = pandas.read_csv(in_dir + f)
        print(f)
        if len(df) % n == 0:
            l = len(df)
        else:
            l = int(len(df) / n) + 1
        
        for i in range(l):
            output_file = out_dir + "group-0" + str(group) + "-"+str(s)+".csv"
            with open(output_file, "w") as file:
                file.write(header + "\n")
            with open(output_file, "a+") as file:
                #import pdb; pdb.set_trace()

                for _, row in df.iloc[seg:seg+n-1].iterrows():
                    x = row.x
                    y = row.y
                    frame = int(row.frameID)
                    out = ','.join([str(frame), str(x), str(y)])
                    file.write(out + "\n")
            seg += n
            s += 1
        
        group += 1
        seg = 0
        s = 0

def format_detections1():
    group = 1
    seg = 0
    s = 1
    out_dir = "data/det_formatted1/"
    in_dir = "data/eval/test/dets/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    header = "frame,u,v"
    for f in os.listdir(in_dir):
        df = pandas.read_csv(in_dir + f)
        print(f)
        
        output_file = out_dir + f[:-3] + 'csv'
        with open(output_file, "w") as file:
            file.write(header + "\n")
        with open(output_file, "a+") as file:
            #import pdb; pdb.set_trace()

            for _, row in df.iterrows():
                x = row.x
                y = row.y
                frame = int(row.frameID)
                out = ','.join([str(frame), str(x), str(y)])
                file.write(out + "\n")


def main():
    #args = parseArguments()
    #format_detections()
    format_detections1()

if __name__ == "__main__":
    main()