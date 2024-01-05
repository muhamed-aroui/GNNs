import math
import torch
from pathlib import Path
import json
fpath= "/users/Etu6/28718016/prat/GNNs/Output/json/SOB_M_DC-14-10926-40-005.json"
with open (fpath) as json_file:
    json_data= json.load(json_file)
num_cells = len(json_data["nuc"])
print(num_cells)
for node_id, node_data in json_data['nuc'].items():
    print(f"node_id {node_id} node_data {node_data}")
    break