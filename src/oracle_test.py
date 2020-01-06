import pickle
from os import path
import numpy as np

files = [
    "../raw_data/origin_oracle_40_final/train_src_scores_f1",
    "../raw_data/origin_oracle_40_final/train_src_scores_f2",
    "../raw_data/origin_oracle_40_final/train_src_scores_fl"
]
for file in files:
    data = pickle.load(open(file, "rb"))
    v = ['p', 'r', 'f']
    for i,n in enumerate(v):
        a = list(map(lambda x: x[i], data))
        print(path.basename(file),":", n,": ", np.median(a))
