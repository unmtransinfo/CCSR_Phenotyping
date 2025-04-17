'''
# generate smaller files

import os
import pickle
import argparse
import yaml
import numpy as np

# load IO filenames from the YAML file
parser = argparse.ArgumentParser()
parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
p_args = parser.parse_args()
with open(p_args.iofiles, 'r') as fi:
    iodata = yaml.safe_load(fi)

print("load person condition dictionary")
with open(iodata['output_files']['person_cond_file'], 'rb') as f:
    person_conds_dict = pickle.load(f)

pids = list(person_conds_dict.keys())
print(f"number of unique patients: {len(pids)}")

print(f"select 50k records")
np.random.seed(7)
indx = np.random.permutation(len(pids))

pid_conds = {}
with open("IOData/pklFiles/person_cond_dict_small.pkl", 'wb') as f:
    for v in indx[:50000]:
        pid_conds[pids[v]] = person_conds_dict[pids[v]]
    pickle.dump(pid_conds, f, protocol=5)


# select all self-harm codes
import json

self_harm_codes = set()
with open(fname, 'r') as f:
    data = json.load(f)
    for i in range(len(data['items'])):
        self_harm_codes.add(data['items'][i]['concept']['CONCEPT_ID'])

print(self_harm_codes)
print(f"total number of codes: {len(self_harm_codes)}")
'''

import json
self_harm_codes = {}
fname = "/home/praveen/Downloads/records.json"
with open(fname, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)
    for i in range(len(data)):
        self_harm_codes[data[i]['r.SAB']] = self_harm_codes.setdefault(data[i]['r.SAB'], 0) + 1
for k,v in self_harm_codes.items():
    print(str(k) + "\t" + str(v))
print(sum(list(self_harm_codes.values())))



