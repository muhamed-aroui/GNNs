import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torchvision import P

class GCN(nn.Module):
    def __init__(self, node_features, embedding_size, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        #self.resnet = 
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

class GATv2(nn.Module):
    def __init__(self, node_features, embedding_size, num_classes, heads, concatenate, dropout):
        super(GATv2, self).__init__()
        if concatenate:
            embedding_size = int(embedding_size/heads)
        self.conv1 = GATv2Conv(node_features, embedding_size, heads=heads, dropout=dropout, concat=concatenate)
        self.conv2 = GATv2Conv(embedding_size*heads, embedding_size*heads, heads=1, dropout=dropout, concat=False)
        self.lin3 = nn.Linear(embedding_size*heads, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return x

def get_model(config, logger):
    if config["model_config"]["layer_type"] == "GCN":
        return GCN(node_features  = config["data_config"]["node_features"], 
                   embedding_size = config["model_config"]["embedding_size"],
                   num_classes    = config["data_config"]["num_classes"],
                   dropout        = config["model_config"]["dropout"])
                   
    elif config["model_config"]["layer_type"] == "GATv2":
        return GATv2(node_features = config["data_config"]["node_features"], 
                   embedding_size = config["model_config"]["embedding_size"],
                   num_classes=config["data_config"]["num_classes"],
                   heads = config["model_config"]["heads"],
                   concatenate=config["model_config"]["concatenate"],
                   dropout = config["model_config"]["dropout"])