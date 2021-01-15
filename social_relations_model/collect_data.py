import pandas
import math
import argparse
import numpy as np

from collections import defaultdict
from itertools import combinations

def distance(det1, det2):
    return math.hypot(det1.x.iloc[0] - det2.x.iloc[0], det1.y.iloc[0] - det2.y.iloc[0])

def speed(det, df, n):
    # go back n frames to calc speed
    curr_frameID = det.frameID.iloc[0]
    curr_pedID = det.pedID.iloc[0]

    # check previous frames and that they exist
    prev_dets = df.loc[(df.pedID == curr_pedID) & (df.frameID < curr_frameID)]
    if len(prev_dets) == 0:
        return 0
    
    # go back n frames (or as far back as possible) to calc distance
    prev_frame = min(n, len(prev_dets))
    prev_det = None
    while prev_frame >= 0:
        prev_det = df.loc[(df.pedID == curr_pedID) & (df.frameID == curr_frameID - prev_frame)]
        if len(prev_det) == 0:
            prev_frame -= 1
        else:
            break
    if len(prev_det) == 0:
        return 0

    dist = distance(det, prev_det)
    return dist / n

def delta_speed(det1, det2, df, n):
    speed1 = speed(det1, df, n)
    speed2 = speed(det2, df, n)
    return speed1 - speed2

def delta_angle(det1, det2):
    vector1 = [det1.x.iloc[0], det1.y.iloc[0]]
    vector2 = [det2.x.iloc[0], det2.y.iloc[0]]

    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle

def output_data(filename, data):
    with open(filename, "w") as outfile:
        outfile.write("frameID,pedID1,pedID2,dist,speed,angle,group_label\n")
        for d in data:
            line = ','.join([str(x) for x in d])
            outfile.write(line + "\n")

# Output format: 
# Diff b/w two pedestrians in same frame: (distance, speed, angle) -> (1/0)
# 1 = in same group
# 0 = not in same group
def collect_data(infile, outfile, n):
    df = pandas.read_csv(infile)
    df = df.loc[df.dataset == "train"]
    
    ped_pairs = defaultdict(list)
    for frameID in df.frameID.unique():
        p = list(combinations(df.loc[df.frameID == frameID].pedID, 2))
        ped_pairs[frameID] += p

    data = []
    count = 0
    for frameID in ped_pairs:
        for pair in ped_pairs[frameID]:
            pedID1, pedID2 = pair
            det1 = df.loc[(df.frameID == frameID) & (df.pedID == pedID1)]
            det2 = df.loc[(df.frameID == frameID) & (df.pedID == pedID2)]

            # calc dist
            dist = distance(det1, det2)
            # calc speed
            speed = delta_speed(det1, det2, df, n)
            # calc angle
            angle = delta_angle(det1, det2)
            # in same group
            if det1.groupID.iloc[0] == det2.groupID.iloc[0] and \
                det1.groupID.iloc[0] != -1 and det2.groupID.iloc[0] != -1:
                group_label = 1
            else:
                group_label = -1
            
            # frameID,pedID1,pedID2,dist,speed,angle,group_label
            data.append([frameID, det1.pedID.iloc[0], det2.pedID.iloc[0], dist, speed, angle, group_label])
            count += 1
            if count % 5000 == 0:
                print(count)
        

    output_data(outfile, data)

def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument("--input_file", default=None, required=True)
    parser.add_argument("--output_file", default=None, required=True)
    parser.add_argument("--n", default=1, required=True, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.input_file, args.output_file, args.n)
    collect_data(args.input_file, args.output_file, args.n)


