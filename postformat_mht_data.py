import pandas
import argparse
import math

# Takes in the group detections and formats it for MHT
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")

    return parser.parse_args()

def main():
    #args = parseArguments()

    in_dir = "data/tracking_results_current/"
    out_dir = "data/tracking_results_formatted/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for ifile in os.listdir():
        df_mht = pandas.read_csv(in_dir + ifile)
        ofile = out_dir + ifile[:8] + '.csv'

        if not os.path.exists(ofile):
            header = "frameID,groupID,x,y,w,h"
            with open(args.output_file, "w") as file:
                file.write(header)

        # mht output frames --> actual frame id
        frame_map = {}
        

        for _, row in df_mht.iterrows():
            groupID = row.track
            frame = row.frame
            x = row.u
            y = row.v

            if x == 'None' or y == 'None':
                continue
            else:
                df = pandas.read_csv('data/eval/test/dets/group-01.csv')
                dets = df.loc[df.frameID == frame]
                minDist = math.inf
                closest = None
                for _, d in dets.iterrows():
                    #import pdb; pdb.set_trace()
                    dist = (float(x) - float(d.x))**2 + (float(y) - float(d.y))**2
                    if dist < minDist:
                        minDist = dist
                        closest = d

                if closest == None:
                    continue
                with open(args.output_file, "w") as file:
                    out = ','.join([str(closest.frameID), 
                                    str(closest.groupID), 
                                    str(closest.x),
                                    str(closest.y),
                                    str(closest.w),
                                    str(closest.h)])
                    file.write(out)

if __name__ == "__main__":
    main()