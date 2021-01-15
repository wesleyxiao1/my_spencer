import pandas
from collections import defaultdict
from itertools import combinations

df = pandas.read_csv('data/group_track_data.csv')
df = df.loc[df.dataset == "train"]

ped_pairs = defaultdict(list)
total = 0
for frameID in df.frameID.unique():
    p = list(combinations(df.loc[df.frameID == frameID].pedID, 2))
    ped_pairs[frameID] += p
    total += len(p)
print("total")
print(total)