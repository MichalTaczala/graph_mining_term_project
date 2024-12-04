import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def gcn_collate_fn(batch):
    edge_indices = []
    features_list = []
    labels = []
    batch_indices = []
    current_index = 0

    for i, (source, destination, label) in enumerate(batch):
        all_metabolites = list(set(source + destination))
        all_metabolites.sort()
        features = torch.tensor(all_metabolites, dtype=torch.long)
        src_indices = [all_metabolites.index(m) for m in source]
        dst_indices = [all_metabolites.index(m) for m in destination]
        edge_index = torch.tensor([[s, d] for s in src_indices for d in dst_indices], dtype=torch.long).t()
        edge_indices.append(edge_index + current_index)
        features_list.append(features)
        labels.append(label)
        batch_indices.append(torch.full((features.size(0),), i, dtype=torch.long))
        current_index += features.size(0)

    return torch.cat(edge_indices, dim=1), torch.cat(features_list, dim=0), torch.cat(batch_indices, dim=0), torch.tensor(labels, dtype=torch.float)



class GCNReactionDirectionPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_feats, out_feats):
        super(GCNReactionDirectionPredictor, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)
        self.fc = nn.Linear(out_feats, 1)

    def forward(self, edge_index, features, batch):
        x = self.embedding(features).squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()