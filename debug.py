import math
import torch
from pathlib import Path
import json
import torch
import sys
import os
import glob
import matplotlib.pyplot as plt
#sys.path.insert(0, '/users/Etu6/28718016/.conda/pkgs/pytorch-1.12.0-py3.10_cuda11.3_cudnn8.3.2_0')

# import torch
# from torch_geometric.nn import GCNConv

# print(os.path.abspath(torch.__file__))
# print(torch.version.cuda)

# print("torch version: ", torch.__version__)

fpath= "/users/Etu6/28718016/Data/BinaryClassification/test/json/"

files= glob.glob(os.path.join(fpath,"*.json"))
print(len(files))
newlist=[]
for f in files:
    with open (f) as json_file:
        json_data= json.load(json_file)
    if len(json_data['nuc'])==1:
        newlist.append(os.path.basename(f))
        Path.unlink(Path(f))
print(len(newlist))
sys.exit()
"""newlist= ['SOB_M_DC-14-16875-400-005.json', 'SOB_M_MC-14-19979-400-019.json', 'SOB_B_A-14-22549AB-400-017.json', 'SOB_M_MC-14-19979-400-013.json']
for i,name in enumerate(newlist):
    name=newlist[0][:-4] +'png'
    im = plt.imread(os.path.join('/users/Etu6/28718016/prat/Output400all/overlay',name))
    plt.imsave(f"test{i}.png",im)"""


fpath= "/users/Etu6/28718016/prat/GNNs/Output/json/SOB_M_DC-14-10926-40-005.json"
with open (fpath) as json_file:
    json_data= json.load(json_file)
num_cells = len(json_data["nuc"])
print(num_cells)
for node_id, node_data in json_data['nuc'].items():
    print(f"node_id {node_id} node_data {node_data}")
    break
if (1,2) == (2,1):
    print("true")