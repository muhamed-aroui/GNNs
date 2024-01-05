import math
import torch
from pathlib import Path
import json
from configuration import config
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import normalize
from utils import *


class GraphDataset(Dataset):
    def __init__(self, root,class_to_idx, transform=None,config = False):
        super().__init__(root, transform,)
        self.root = Path(root)
        self.transform = transform
        self.config = config
        self.class_to_idx = class_to_idx
    @property
    def raw_files_paths(self):
        return [s/"data.json" for s in self.root.iterdir()]

    def _process_data(self, idx):

        with open (self.raw_files_paths[idx]) as json_file:
            json_data= json.load(json_file)
        
        graph_features = list()

        num_cells = len(json_data["nuc"])
        if num_cells == 0:
            raise ValueError("no cells in the graph")
        
        for node_id, node_data in json_data['nuc'].items():
            fused_embedding = [node_data['type_prob'],*node_data['centroid']]

            contour_feats = get_contour_feats(node_data['contour'])
            values_list = list(contour_feats.values())
            fused_embedding += values_list
            graph_features.append(fused_embedding)

        graph_features = np.array(graph_features, dtype=np.float32)
        elif self.config["data_config"]["normalize"]["l1"]:
            graph_features = normalize(graph_features, norm="l1")
        elif self.config["data_config"]["normalize"]["l2"]:
            graph_features = normalize(graph_features, norm="l2")


            
    