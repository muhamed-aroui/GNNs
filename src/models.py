import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, node_features, embedding_size, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.lin = nn.Linear(embedding_size, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)
        
        return x
def get_model(config, logger):
    if config["model_config"]["layer_type"] == "GCN":
        return GCN(node_features  = config["data_config"]["node_features"], 
                   embedding_size = config["model_config"]["embedding_size"],
                   num_classes    = config["data_config"]["num_classes"],
                   dropout        = config["model_config"]["dropout"])