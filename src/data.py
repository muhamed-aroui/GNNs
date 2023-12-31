import math
import torch
from pathlib import Path
import json
import numpy as np
from configuration import config
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import normalize
from utils import get_contour_feats, directory_samples
import logging
from configuration import config


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
        edge_features = list()
        edge_indices = list()

        num_cells = len(json_data["nuc"])
        if num_cells == 0:
            raise ValueError("no cells in the graph")
        
        for node_id, node_data in json_data['nuc'].items():
            fused_embedding = [node_data['type_prob'],*node_data['centroid']]

            contour_feats = get_contour_feats(node_data['contour'])
            values_list = list(contour_feats.values())
            fused_embedding += values_list
            graph_features.append(fused_embedding)
            # Extracting edge information (if any)
            for neighbor_node_id, neighbor_node_data in json_data['nuc'].items():
                if node_id != neighbor_node_id:
                    if [int(node_id) - 1, int(neighbor_node_id) - 1] in edge_indices:
                        continue

                    edge_indices.append([int(node_id) - 1, int(neighbor_node_id) - 1])  # Assuming nodes are 1-indexed
                    edge_indices.append([int(neighbor_node_id) - 1, int(node_id) - 1])
                    
                    _feat = self._edge_feature_gen(node_data["centroid"],
                                                     neighbor_node_data["centroid"])
                    edge_features.append(_feat)
                    edge_features.append(_feat)

        graph_features = np.array(graph_features, dtype=np.float32)
        if self.config["data_config"]["normalize"]["l1"]:
            graph_features = normalize(graph_features, norm="l1")
        elif self.config["data_config"]["normalize"]["l2"]:
            graph_features = normalize(graph_features, norm="l2")
        graph_features = torch.tensor(graph_features, dtype=torch.float)

        edge_index = []

        graph_data = Data(x=graph_features, 
                          edge_index=edge_index.t().contiguous(),
                          edge_attr=torch.tensor(edge_features, dtype=torch.float),
                          y = torch.tensor(self.class_to_idx.get(json_data["gt"], -1), dtype=torch.long))
        return graph_data
    
    @staticmethod
    def _edge_feature_gen(center1, center2):
        x1, y1 = center1
        x2, y2 = center2
        myradians = math.atan2(y2 - y1, x2 - x1)

        return [math.hypot(x2 - x1, y2 - y1), myradians]
    def len(self):
        return len(self.raw_files_paths)

    def get(self, idx):
        data = self._process_data(idx)
        if self.transform:
            data = self.transform(data)
        return data

def get_data(config, logger):
    train_samples = directory_samples(config["data_config"]["db_root"])
    test_samples = directory_samples(config["data_config"]["test_root"])
    validation_samples = directory_samples(config["data_config"]["validation_root"])
    
    logger.info(f"{len(train_samples)} graphs")
    logger.info(f"{len(validation_samples)} graphs")
    logger.info(f"{len(test_samples)} graphs")

if __name__ == "__main__":
    print("here")
    format = "%(asctime)s:     %(message)s"
    logging.basicConfig(filename="training.log",
                        filemode="w",
                        format=format, 
                        level=logging.DEBUG,
                        datefmt="%H:%M:%S")
    get_data(config, logging)
            
    