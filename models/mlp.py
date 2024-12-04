import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class ImprovedMLPWithEmbeddings(nn.Module):
    def __init__(self, max_metabolite_id, embedding_dim, hidden_dim, output_dim):
        super(ImprovedMLPWithEmbeddings, self).__init__()
        self.embedding = nn.Embedding(max_metabolite_id, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, destination):
        source_emb = self.embedding(source).mean(dim=1)
        destination_emb = self.embedding(destination).mean(dim=1)
        x = torch.cat([source_emb, destination_emb], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x)).squeeze()

def mlp_collate_fn(batch):
    sources, destinations, labels = zip(*batch)
    padded_sources = pad_sequence([torch.tensor(src) for src in sources], batch_first=True, padding_value=0)
    padded_destinations = pad_sequence([torch.tensor(dst) for dst in destinations], batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_sources, padded_destinations, labels