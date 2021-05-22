import pandas
import argparse
import os
import shutil

def main():
    #dets = pandas.read_csv("data/detected_groups_test2.csv")
    dets = pandas.read_csv("data/detected_groups_robogem2.csv")
    df = pandas.read_csv("data/group_track_data.csv")
    
    f = []
    files = ['data/eval/test/group-0'+str(x)+'/det/det.txt' for x in range(1, 7)]
    for x in files:
        with open(x, "w") as fp:
            pass
    
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
            max_frame = round(df.loc[((df.dataset == dataset) & (df.segment_num == group_num))].frameID.min(), -3)
            out = row.values.tolist()

            out[0] = int(out[0])
            out[0] = int(out[0] % max_frame)

            out[1] = int(out[1])
            out += [-1, -1, -1, -1]
            out = ','.join([str(x) for x in out])
            file.write(out + "\n")
        
        if row.frameID % 1000 == 0:
            print(row.frameID)
    eval_dir = "data/eval/test/"
    for f in os.listdir(eval_dir):
        #print(f)
        if "dets" in f:
            continue
        src = eval_dir + f + "/det/det.txt"
        dst = eval_dir + "dets/" + f + ".txt"
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()