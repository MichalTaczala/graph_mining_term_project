import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import numpy as np

# Load dataset
class ReactionDataset(Dataset):
    def __init__(self, data_file, answers_file=None, is_training=True):
        self.data = pd.read_csv(data_file)
        self.is_training = is_training

        if not is_training and answers_file:
            self.answers = pd.read_csv(answers_file)
            self.data_x = self.data
            self.data_y = self.answers['direction']
        else:
            self.data_x = self.data.drop('direction', axis=1)
            self.data_y = self.data['direction']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data_x.iloc[idx]
        source_metabolites = list(eval(row['source']))
        destination_metabolites = list(eval(row['destination']))        
        label = 1 if self.data_y.iloc[idx] else 0
        
        return source_metabolites, destination_metabolites, label

# Hypergraph Neural Network Model
class HypergraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(HypergraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(0.1)

    def forward(self, adjacency_matrix, features):
        aggregated_features = adjacency_matrix @ features
        aggregated_features = self.dropout(aggregated_features)
        return F.leaky_relu(self.fc(aggregated_features))
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, adjacency_matrix, features):
        h = self.fc(features)
        N = h.size(0)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = self.leakyrelu(torch.matmul(a_input, self.attn).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacency_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class ReactionDirectionPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(ReactionDirectionPredictor, self).__init__()
        
        self.conv1 = HypergraphConvLayer(in_feats, hidden_feats)
        self.ga = GraphAttentionLayer(hidden_feats, hidden_feats)
        self.conv2 = HypergraphConvLayer(hidden_feats, hidden_feats)
        self.conv25 = HypergraphConvLayer(hidden_feats, hidden_feats)
        self.conv3 = HypergraphConvLayer(hidden_feats, out_feats)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_feats, 1)

    def forward(self, adjacency_matrix, features):
        h = self.conv1(adjacency_matrix, features)
        h = self.relu(h)
        h = self.ga(adjacency_matrix, h)
        h = self.relu(h)
        h = self.conv2(adjacency_matrix, h)
        h = self.relu(h)
        h = self.conv3(adjacency_matrix, h)
        hg = torch.mean(h, dim=0)
        return self.fc(hg)

# Create directed hypergraph from source and destination metabolites
def create_directed_hypergraph(source_metabolites, destination_metabolites):
    all_metabolites = list(set(map(int, source_metabolites + destination_metabolites)))
    num_nodes = len(all_metabolites)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    src_indices = [all_metabolites.index(m) for m in source_metabolites]
    dst_indices = [all_metabolites.index(m) for m in destination_metabolites]
    
    # Adding directed hyperedges from source metabolites to destination metabolites
    for src_idx in src_indices:
        for dst_idx in dst_indices:
            adjacency_matrix[src_idx, dst_idx] = 1
    
    return torch.tensor(adjacency_matrix, dtype=torch.float32), torch.tensor(all_metabolites, dtype=torch.long)

# Training the model
def train(model, data_loader, val_loader, epochs=40, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for source, destination, label in data_loader:
            source = [int(m) for m in source]
            destination = [int(m) for m in destination]
            adjacency_matrix, nodes = create_directed_hypergraph(source, destination)
            features = torch.randn(adjacency_matrix.size(0), 1)  # Random features for diversity
            label = torch.tensor([label], dtype=torch.float32)

            prediction = model(adjacency_matrix, features)
            loss = loss_fn(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for source, destination, label in val_loader:
                source = [int(m) for m in source]
                destination = [int(m) for m in destination]
                adjacency_matrix, nodes = create_directed_hypergraph(source, destination)
                features = torch.randn(adjacency_matrix.size(0), 1)  # Random features for diversity
                label = torch.tensor([label], dtype=torch.float32)

                prediction = model(adjacency_matrix, features)
                predicted_label = torch.round(torch.sigmoid(prediction))
                correct += (predicted_label == label).sum().item()
                total += 1
        train_losses.append(total_loss / len(data_loader))
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy * 100:.2f}%")
    return val_accuracies, train_losses

# Main Script
data_file = "dataset/Classification_training.csv"
train_dataset = ReactionDataset(data_file)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = ReactionDataset('dataset/Classification_valid_query.csv', answers_file='dataset/Classification_valid_answer.csv', is_training=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

in_feats = 1
hidden_feats = 32
out_feats = 16

model = ReactionDirectionPredictor(in_feats, hidden_feats, out_feats)
val, loss = train(model, train_loader, val_loader)
import matplotlib.pyplot as plt
# plt.plot(val)
# plt.xlabel('Epochs')
# plt.ylabel('Validation Accuracy')
# plt.title('Validation Accuracy vs. Epochs')
# plt.show()

plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epochs')
plt.show()