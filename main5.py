import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
def get_max_metabolite_id(data_file):
    data = pd.read_csv(data_file)
    max_id = 0
    for _, row in data.iterrows():
        source_metabolites = list(eval(row['source']))
        destination_metabolites = list(eval(row['destination']))
        max_id = max(max_id, max(source_metabolites + destination_metabolites))
    return max_id + 1  # Add 1 for zero-based indexing


# Load dataset with metabolite-specific embeddings
class ReactionDataset(Dataset):
    def __init__(self, data_file, max_metabolite_id, embedding_dim, answers_file=None, is_training=True):
        self.data = pd.read_csv(data_file)
        self.is_training = is_training
        self.max_metabolite_id = max_metabolite_id
        self.embedding_dim = embedding_dim

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

        # Debugging: Validate sequence lengths
        if len(source_metabolites) == 0 or len(destination_metabolites) == 0:
            raise ValueError(f"Empty sequence at index {idx}: Source={source_metabolites}, Destination={destination_metabolites}")

        return (
            torch.tensor(source_metabolites, dtype=torch.long),
            torch.tensor(destination_metabolites, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )






from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Unpack batch into separate lists
    sources, destinations, labels = zip(*batch)

    # Debugging: Print raw sequence lengths
    

    # Pad source and destination tensors
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)  # Shape: (batch_size, max_source_len)
    padded_destinations = pad_sequence(destinations, batch_first=True, padding_value=0)  # Shape: (batch_size, max_dest_len)

    # Stack labels into a tensor (no padding needed for labels)
    labels = torch.stack(labels)  # Shape: (batch_size,)

    return padded_sources, padded_destinations, labels




# Define the Improved MLP with additional layers and embeddings
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
        # Embed source and destination metabolites
        source_emb = self.embedding(source)  # Shape: (batch_size, max_source_len, embedding_dim)
        destination_emb = self.embedding(destination)  # Shape: (batch_size, max_dest_len, embedding_dim)

        # Create masks to ignore padding (padding value = 0)
        source_mask = (source != 0).unsqueeze(-1)  # Shape: (batch_size, max_source_len, 1)
        destination_mask = (destination != 0).unsqueeze(-1)  # Shape: (batch_size, max_dest_len, 1)

        # Compute mean embeddings, ignoring padding
        source_emb = (source_emb * source_mask).sum(dim=1) / source_mask.sum(dim=1).clamp(min=1e-9)
        destination_emb = (destination_emb * destination_mask).sum(dim=1) / destination_mask.sum(dim=1).clamp(min=1e-9)

        # Concatenate source and destination embeddings
        x = torch.cat([source_emb, destination_emb], dim=1)

        # Pass through fully connected layers
        x = self.fc1(x).relu()
        x = self.bn1(x)
        x = self.fc2(x).relu()
        x = self.bn2(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x).squeeze()

        return torch.sigmoid(x)



# Train function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for i, (source, destination, labels) in enumerate(loader):

        optimizer.zero_grad()
        out = model(source, destination)
        loss = criterion(out, labels)
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
        for source, destination, labels in loader:
            out = model(source, destination)
            predicted_labels = (out > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    return correct / total


# Load datasets with embeddings
embedding_dim = 64  # Dimension of learned metabolite embeddings
max_metabolite_id = get_max_metabolite_id("dataset/Classification_training.csv")
train_dataset = ReactionDataset("dataset/Classification_training.csv", max_metabolite_id, embedding_dim)
val_dataset = ReactionDataset(
    "dataset/Classification_valid_query.csv", max_metabolite_id, embedding_dim, answers_file="dataset/Classification_valid_answer.csv", is_training=False
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)



# Model configuration
hidden_dim = 256  # Increased hidden layer size
output_dim = 1

# Initialize the model with metabolite embeddings
model = ImprovedMLPWithEmbeddings(max_metabolite_id=max_metabolite_id, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Optimizer, loss, and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training loop with scheduler
epochs = 1
best_accuracy = 0

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_accuracy = evaluate(model, val_loader)
    scheduler.step(val_accuracy)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

print(f"Best Validation Accuracy: {best_accuracy:.4f}")



# Define ranges for hyperparameter tuning
learning_rates = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
weight_decays = [1e-4, 3e-4, 5e-4, 7e-4]

# Variables to track the best configuration
best_lr = None
best_wd = None
best_accuracy = 0

# Loop through learning rates and weight decays
for lr in learning_rates:
    for wd in weight_decays:
        print(f"\nTesting with Learning Rate = {lr}, Weight Decay = {wd}")

        # Initialize the model with metabolite embeddings
        model = ImprovedMLPWithEmbeddings(max_metabolite_id=max_metabolite_id, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        # Optimizer, loss, and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Training loop
        epochs = 30  # Adjust based on time/resource constraints
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_accuracy = evaluate(model, val_loader)
            scheduler.step(val_accuracy)
            print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

            # Save the best model configuration
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_lr = lr
                best_wd = wd
                torch.save(model.state_dict(), "best_model.pth")

print(f"\nBest Validation Accuracy: {best_accuracy:.4f}")
print(f"Best Learning Rate: {best_lr}, Best Weight Decay: {best_wd}")
