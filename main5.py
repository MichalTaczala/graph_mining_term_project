import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd

# Load dataset
import numpy as np

class ReactionDataset(Dataset):
    def __init__(self, data_file, max_metabolite_id, answers_file=None, is_training=True):
        self.data = pd.read_csv(data_file)
        self.is_training = is_training
        self.max_metabolite_id = max_metabolite_id

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

        # One-hot encoding
        source_vector = np.zeros(self.max_metabolite_id)
        destination_vector = np.zeros(self.max_metabolite_id)
        source_vector[source_metabolites] = 1
        destination_vector[destination_metabolites] = 1
        feature = np.concatenate([source_vector, destination_vector])

        return torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).squeeze()
        return torch.sigmoid(x)
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.bn1(x)
        x = self.fc2(x).relu()
        x = self.bn2(x)
        x = self.fc3(x).relu()
        x = self.fc4(x).squeeze()
        return torch.sigmoid(x)

import pandas as pd

def get_max_metabolite_id(data_file):
    data = pd.read_csv(data_file)
    max_id = 0
    for _, row in data.iterrows():
        source_metabolites = list(eval(row['source']))
        destination_metabolites = list(eval(row['destination']))
        max_id = max(max_id, max(source_metabolites + destination_metabolites))
    return max_id + 1  # Add 1 for zero-based indexing
class GCNModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Additional GCN layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Additional fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()  # Additional GCN layer
        x = global_mean_pool(x, batch)
        x = self.fc1(x).relu()
        x = self.fc2(x).squeeze()
        return torch.sigmoid(x)





# Train function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for features, labels in loader:  # Unpack features and labels from DataLoader
        optimizer.zero_grad()
        out = model(features)  # Pass only the features to the model
        loss = criterion(out, labels)  # Compute the loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluate function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:  # Unpack features and labels
            out = model(features)  # Pass only features to the model
            predicted_labels = (out > 0.5).float()  # Apply threshold
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Load datasets
max_metabolite_id = get_max_metabolite_id("dataset/Classification_training.csv")
train_dataset = ReactionDataset("dataset/Classification_training.csv", max_metabolite_id)
val_dataset = ReactionDataset("dataset/Classification_valid_query.csv", max_metabolite_id=max_metabolite_id, answers_file="dataset/Classification_valid_answer.csv", is_training=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model configuration
input_dim = max(max(len(eval(row['source'])) + len(eval(row['destination'])) for _, row in train_dataset.data.iterrows()),
                max(len(eval(row['source'])) + len(eval(row['destination'])) for _, row in val_dataset.data.iterrows()))
# model = GCNModel(num_nodes=num_nodes, embedding_dim=128, hidden_dim=128, output_dim=1)
# Model parameters
input_dim = 3  # Feature dimension
hidden_dim = 64  # Size of hidden layers
output_dim = 1  # Binary classification output

# Initialize the model
# Initialize the model with the correct input_dim
input_dim = 4  # Number of features in the dataset (updated from 3 to 4)
hidden_dim = 64  # Size of hidden layers
output_dim = 1  # Binary classification output

# Determine input_dim based on one-hot encoding
input_dim = 2 * max_metabolite_id  # Source and destination metabolites
hidden_dim = 64  # Hidden layer size
output_dim = 1  # Binary classification output

# Initialize the model
model = ImprovedMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



# Training loop
epochs = 20
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "gcn_model.pth")
