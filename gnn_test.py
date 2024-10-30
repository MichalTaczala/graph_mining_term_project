import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

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

# GCN-based Model
class GCNReactionDirectionPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_feats, out_feats):
        super(GCNReactionDirectionPredictor, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim)  # Embedding layer for metabolites (assuming 10,000 unique metabolites)
        self.conv1 = GCNConv(embedding_dim, hidden_feats, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_feats, hidden_feats, add_self_loops=True, normalize=True)
        self.conv3 = GCNConv(hidden_feats, out_feats, add_self_loops=True, normalize=True)
        self.fc = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, edge_index, features, batch):
        x = self.embedding(features).squeeze(1)  # Apply embedding
        x = self.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Add dropout for regularization
        x = self.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Add dropout for regularization
        x = self.leaky_relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # Pooling over the entire graph
        return self.fc(x).view(-1)  # Removed sigmoid to use BCEWithLogitsLoss directly

# Create Graph Data from source and destination metabolites
# Note: This function needs to be adapted for PyTorch Geometric
def create_graph_data(source_metabolites, destination_metabolites):
    # Combine metabolites and create node features
    all_metabolites = list(set(source_metabolites + destination_metabolites))
    all_metabolites = sorted(all_metabolites)  # Ensure consistent ordering
    num_nodes = len(all_metabolites)
    features = torch.tensor(all_metabolites, dtype=torch.long)  # Use indices for embedding lookup

    # Create edge index for PyTorch Geometric (2 x num_edges)
    edge_index = []
    src_indices = [all_metabolites.index(m) for m in source_metabolites]
    dst_indices = [all_metabolites.index(m) for m in destination_metabolites]
    for src in src_indices:
        for dst in dst_indices:
            edge_index.append([src, dst])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index, features

# Custom collate function to handle varying sizes in batch
def custom_collate(batch):
    edge_indices = []
    features_list = []
    labels = []
    batch_indices = []
    current_index = 0

    for i, (source, destination, label) in enumerate(batch):
        edge_index, features = create_graph_data(source, destination)
        edge_indices.append(edge_index + current_index)
        features_list.append(features)
        labels.append(label)
        batch_indices.append(torch.full((features.size(0),), i, dtype=torch.long))
        current_index += features.size(0)

    edge_index = torch.cat(edge_indices, dim=1)
    features = torch.cat(features_list, dim=0)
    batch = torch.cat(batch_indices, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)

    return edge_index, features, batch, labels

# Training the model
def train_gcn(model, data_loader, val_loader, epochs=40, lr=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add weight decay for regularization
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for edge_index, features, batch, label in data_loader:
            prediction = model(edge_index, features, batch)
            loss = loss_fn(prediction, label) + 1e-4 * sum(p.pow(2.0).sum() for p in model.parameters())

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
            for edge_index, features, batch, label in val_loader:
                prediction = model(edge_index, features, batch)
                predicted_label = torch.round(torch.sigmoid(prediction))
                correct += (predicted_label == label).sum().item()
                total += len(label)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        train_losses.append(total_loss / len(data_loader))
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy * 100:.2f}%")
    return val_accuracies, train_losses

# Main Script
data_file = "dataset/Classification_training.csv"
train_dataset = ReactionDataset(data_file)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

val_dataset = ReactionDataset('dataset/Classification_valid_query.csv', answers_file='dataset/Classification_valid_answer.csv', is_training=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

embedding_dim = 64  # Set embedding dimension
hidden_feats = 64
out_feats = 32

model = GCNReactionDirectionPredictor(embedding_dim, hidden_feats, out_feats)
val, loss = train_gcn(model, train_loader, val_loader)

# Plot Training Loss
import matplotlib.pyplot as plt
plt.plot(val)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epochs')
plt.show()
