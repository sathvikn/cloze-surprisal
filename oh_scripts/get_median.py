"""
Code for taking the median over 5 runs
"""

import numpy as np
import sys

def read_raw_loocv(fn):
    ll = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            ll.append(float(line.strip()))
    return ll

def read_raw_loocv_twocol(fn):
    ll = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            ll.append(float(line.strip().split(" ")[-1]))
    return ll

all_loocv = []
for i in sys.argv[1:]:
    all_loocv.append(read_raw_loocv(i))

# print(all_loocv)
assert len(all_loocv) == 5

# print(np.array(all_loocv))
for i in np.median(np.array(all_loocv), axis=0).tolist():
    print(i)