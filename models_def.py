# Architectures must match the notebook exactly — mismatches break state_dict loading.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

# from_smiles produces 9 node features
NUM_NODE_FEATURES = 9
HIDDEN_CHANNELS = 64


# 1) SimpleGCN — 3 GCNConv layers, mean pooling
class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features=NUM_NODE_FEATURES, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        out = self.lin(x)
        return out


# 2) MajorityGCN — 2 GCNConv layers, mean+max pooling concatenated; 27 copies, majority vote
class MajorityGCN(torch.nn.Module):
    def __init__(self, num_node_features=NUM_NODE_FEATURES, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x_1 = global_mean_pool(x, batch)
        x_2 = global_max_pool(x, batch)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.lin(x)
        x = x.relu()
        out = self.lin2(x)
        return out


# 3) ChemBERTaClassifier — frozen backbone + classification head
class ChemBERTaClassifier(nn.Module):
    def __init__(self, model_name=CHEMBERTA_MODEL_NAME, freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel  # lazy to avoid loading transformers at startup
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # token [CLS]
        return self.head(cls)
