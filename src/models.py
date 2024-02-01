import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torchvision import models
import torch
from torchvision.models import resnet18, ResNet18_Weights

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
    
class ResNetModel(nn.Module):
    def __init__(self, node_features, embedding_size, num_classes, dropout):
        super(ResNetModel, self).__init__()
        # Pre-trained ResNet
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights, progress=False).eval()
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        self.resnet_fc_features = resnet.fc.in_features
        # Readout layer
        self.lin = nn.Linear(self.resnet_fc_features, num_classes)

    def forward(self, x, edge_index, x_image, batch):
        # ResNet
        with torch.no_grad():
            x_image = self.resnet(x_image)
        x_image = x_image.view(-1, self.resnet_fc_features)
        # 3. Apply a final classifier
        x_image = self.lin(x_image)
        return x_image
    
class GCNResNetConcatModel(nn.Module):
    def __init__(self, node_features, embedding_size, num_classes, dropout):
        super(GCNResNetConcatModel, self).__init__()
        self.conv1 = GCNConv(node_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        # Pre-trained ResNet
        #resnet = models.resnet18(pretrained=True)
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights, progress=False).eval()
        #self.transforms = weights.transforms()
        #resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        self.resnet_fc_features = resnet.fc.in_features
        # Readout layer
        self.lin = nn.Linear(embedding_size + self.resnet_fc_features, num_classes)

    def forward(self, x, edge_index, x_image, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # ResNet
        with torch.no_grad():
            #x_image = self.transforms(x_image)
            x_image = self.resnet(x_image)
        x_image = x_image.view(-1, self.resnet_fc_features)

        # Concatenate GCN and ResNet embeddings
        x_combined = torch.cat([x, x_image], dim=1)

        # 3. Apply a final classifier
        x = self.lin(x_combined)
        
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
    
    elif config["model_config"]["layer_type"] == "ResnetConcate":
        return GCNResNetConcatModel(node_features  = config["data_config"]["node_features"], 
                   embedding_size = config["model_config"]["embedding_size"],
                   num_classes    = config["data_config"]["num_classes"],
                   dropout        = config["model_config"]["dropout"])
    elif config["model_config"]["layer_type"] == "Resnet":
        return ResNetModel(node_features  = config["data_config"]["node_features"], 
                   embedding_size = config["model_config"]["embedding_size"],
                   num_classes    = config["data_config"]["num_classes"],
                   dropout        = config["model_config"]["dropout"])