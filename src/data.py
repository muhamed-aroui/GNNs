import math
import torch
from pathlib import Path
import json
import numpy as np
from configuration import config
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import normalize
from torch_geometric.loader import DataLoader
from utils import get_contour_feats, directory_samples
import logging
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision.models import ResNet18_Weights



class GraphDataset(Dataset):
    def __init__(self, root, transform=None,config = False,tf=None):
        super().__init__(root, transform,)
        self.root = Path(root)
        self.transform = transform
        self.tf= tf
        self.config = config
        self.imroot = "/users/Etu6/28718016/mag400/train/"
    @property
    def raw_files_paths(self):
        return [s for s in self.root.iterdir()]

    def _process_data(self, idx):
        filename=os.path.basename(self.raw_files_paths[idx])
        filename = os.path.splitext(filename)[0] + ".png"
        image_file = os.path.join(self.imroot, filename)
        #image_data = read_image(image_file)
        image_data = Image.open(image_file)
        with open (self.raw_files_paths[idx]) as json_file:
            json_data= json.load(json_file)
        
        graph_features = list()
        edge_features = list()
        edge_index = list()

        num_cells = len(json_data["nuc"])
        if num_cells == 0:
            raise ValueError("no cells in the graph")
        
        for node_id, node_data in json_data['nuc'].items():
            fused_embedding = [node_data['type_prob'],node_data['type'],*node_data['centroid']]

            contour_feats = get_contour_feats(node_data['contour'])
            values_list = list(contour_feats.values())
            fused_embedding += values_list
            graph_features.append(fused_embedding)
            # Extracting edge information (if any)
            for neighbor_node_id, neighbor_node_data in json_data['nuc'].items():
                dist=self._edge_feature_gen(node_data["centroid"],
                                                     neighbor_node_data["centroid"])
                if dist[0] > 150:
                    continue
                if node_id != neighbor_node_id:
                    if [int(node_id) - 1, int(neighbor_node_id) - 1] in edge_index:
                        continue

                    edge_index.append([int(node_id) - 1, int(neighbor_node_id) - 1])  # Assuming nodes are 1-indexed
                    edge_index.append([int(neighbor_node_id) - 1, int(node_id) - 1])
                    
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

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        graph_data = Data(x=graph_features, 
                          edge_index=edge_index.t().contiguous(),
                          edge_attr=torch.tensor(edge_features, dtype=torch.float),
                          y = torch.tensor(json_data["label_int"], dtype=torch.float))
        return graph_data, image_data
    
    @staticmethod
    def _edge_feature_gen(center1, center2):
        x1, y1 = center1
        x2, y2 = center2
        myradians = math.atan2(y2 - y1, x2 - x1)

        return [math.hypot(x2 - x1, y2 - y1), myradians]
    def len(self):
        return len(self.raw_files_paths)

    def get(self, idx):
        graph_data, image_data = self._process_data(idx)
        image_data = self.tf(image_data)
        if self.transform:
            image_data = self.transform(image_data)
            #graph_data = self.transform(graph_data)
            
        return graph_data,image_data

def get_data(config, logger):
    train_samples = directory_samples(config["data_config"]["db_root"])
    test_samples = directory_samples(config["data_config"]["test_root"])
    validation_samples = directory_samples(config["data_config"]["validation_root"])
    
    logger.info(f"{train_samples} graphs")
    logger.info(f"{validation_samples} graphs")
    logger.info(f"{test_samples} graphs")
    # classes = {
    #     "train":train_classes,
    #     "test": test_classes,
    #     "validation": validation_classes
    # }
    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    train_dataset = GraphDataset(root=config["data_config"]["db_root"], config=config,tf=transforms)
    temp_data = next(iter(train_dataset))
    test_dataset = GraphDataset(root=config["data_config"]["test_root"], config=config,tf=transforms)
    validation_dataset = GraphDataset(root=config["data_config"]["validation_root"], config=config,tf=transforms)
    dataloaders = {
        x:DataLoader(dataset, 
               batch_size=config["training_config"]["batch_size"], 
               num_workers=config["training_config"]["num_workers"],
               pin_memory=True)
        for x, dataset in [("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)]
    }

    temp_data = next(iter(validation_dataset))
    
    return dataloaders , temp_data[0].x.shape[1]

if __name__ == "__main__":
    print("here")
    format = "%(asctime)s:     %(message)s"
    logging.basicConfig(filename="training.log",
                        filemode="w",
                        format=format, 
                        level=logging.DEBUG,
                        datefmt="%H:%M:%S")
    get_data(config, logging)
            
    