import numpy as np
import pandas as pd
import scipy
import sys


def read_raw_loocv(fn):
    ll = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            ll.append(float(line.strip()))
    return ll

df1 = np.array(read_raw_loocv(sys.argv[1]))
df2 = np.array(read_raw_loocv(sys.argv[2]))
print(len(np.isfinite(df1)))

# print(df1)
# print(np.corrcoef(df1[np.isfinite(df1)], df2[np.isfinite(df1)]))
res = scipy.stats.pearsonr(df1[np.isfinite(df1)], df2[np.isfinite(df1)], method=scipy.stats.PermutationMethod())
print(res)
print(res.confidence_interval(confidence_level=0.95))
