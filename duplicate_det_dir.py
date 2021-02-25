import shutil
import os

src = "data/eval/test/dets"
for i in range(1, 11):
    dst = "data/eval/test/dets_"+str(i)
    shutil.copytree(src, dst)