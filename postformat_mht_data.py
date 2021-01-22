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
    args = parseArguments()

    df = pandas.read_csv(args.input_file)

    header = "frameID,groupID,x,y,w,h"
    with open(args.output_file, "w") as file:
        file.write(header)
    
    for _, row in df.iterrows():
        groupID = row.track
        frame = row.frame
        x = row.u
        y = row.v

        if x == None or y == None:
            continue
        else:
            dets = df.loc[df.frameID == frame]
            minDist = math.inf
            closest = None
            for _, d in dets.interrows(x):
                dist = math.dist([x, y], [d.x, d.y])
                if dist < minDist:
                    minDist = dist
                    closest = d

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