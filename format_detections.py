import pandas
import argparse
import os
import shutil

def main():
    dets = pandas.read_csv("data/detected_groups_test1.csv")
    df = pandas.read_csv("data/group_track_data.csv")
    
    f = []
    for _, row in dets.iterrows():
        d = df.loc[df.frameID == row.frameID]
        dataset = d.iloc[0].dataset
        group_num = d.iloc[0].segment_num

        if dataset != "test":
            continue

        outfile = "data/eval/"+dataset+"/group-{:02d}/det/det.txt".format(group_num)
        if outfile in f:
            aw = 'a'
        else:
            aw = 'w'
            f.append(outfile)
        with open(outfile, aw) as file:
            print(row.values.tolist())
            out = row.values.tolist()

            out[0] = int(out[0])
            out[0] = int(out[0] % 18000)

            out[1] = int(out[1])
            out += [-1, -1, -1, -1]
            out = ','.join([str(x) for x in out])
            print(out + "\n")
            file.write(out + "\n")

    eval_dir = "data/eval/test/"
    for f in os.listdir(eval_dir):
        print(f)
        if f == "dets":
            continue
        src = eval_dir + f + "/det/det.txt"
        dst = eval_dir + "dets/" + f + ".txt"
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()