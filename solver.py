import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from classification_model_enum import ClassificationModelEnum
from models.gnn import GCNReactionDirectionPredictor, gcn_collate_fn
from models.mlp import ImprovedMLPWithEmbeddings, mlp_collate_fn
from models.tree_classifier import DecisionTreeSolver, preprocess_for_decision_tree


class Solver:
    def __init__(self, model_type, train_dataset, val_dataset, test_dataset, max_metabolite_id=20000, embedding_dim=64, epochs=10, output_file="Classification_test_answer.csv"):
        self.model_type = model_type
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.output_file = output_file
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.max_metabolite_id = max_metabolite_id
        self.embedding_dim = embedding_dim

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model, optimizer, loss function, and data loaders."""
        if self.model_type == ClassificationModelEnum.MLP:
            self.model = ImprovedMLPWithEmbeddings(self.max_metabolite_id, self.embedding_dim, 256, 1)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
            self.criterion = nn.BCELoss()
            self.train_loader = DataLoader(self.train_dataset, batch_size=32, collate_fn=mlp_collate_fn)
            self.val_loader = DataLoader(self.val_dataset, batch_size=32, collate_fn=mlp_collate_fn)
            self.test_loader = DataLoader(self.test_dataset, batch_size=32, collate_fn=mlp_collate_fn)

        elif self.model_type == ClassificationModelEnum.GCN:
            self.model = GCNReactionDirectionPredictor(self.embedding_dim, 64, 32)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            self.criterion = nn.BCEWithLogitsLoss()
            self.train_loader = DataLoader(self.train_dataset, batch_size=16, collate_fn=gcn_collate_fn)
            self.val_loader = DataLoader(self.val_dataset, batch_size=16, collate_fn=gcn_collate_fn)
            self.test_loader = DataLoader(self.test_dataset, batch_size=16, collate_fn=gcn_collate_fn)

        elif self.model_type == ClassificationModelEnum.DECISION_TREE:
            self.model = DecisionTreeSolver()

    def train(self):
        """Train the model."""
        if self.model_type == ClassificationModelEnum.DECISION_TREE:
            X_train, y_train = preprocess_for_decision_tree(self.train_dataset)
            self.model.train(X_train, y_train)
            return

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                if self.model_type == ClassificationModelEnum.MLP:
                    source, destination, labels = batch
                    outputs = self.model(source, destination)
                elif self.model_type == ClassificationModelEnum.GCN:
                    edge_index, features, batch_indices, labels = batch
                    outputs = self.model(edge_index, features, batch_indices)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

    def evaluate(self):
        """Evaluate the model on the validation dataset."""
        if self.model_type == ClassificationModelEnum.DECISION_TREE:
            X_val, y_val = preprocess_for_decision_tree(self.val_dataset)
            predictions = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")
            return

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_type == ClassificationModelEnum.MLP:
                    source, destination, labels = batch
                    outputs = self.model(source, destination)
                elif self.model_type == ClassificationModelEnum.GCN:
                    edge_index, features, batch_indices, labels = batch
                    outputs = self.model(edge_index, features, batch_indices)
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    def predict_and_save(self):
        """Generate predictions for the test dataset and save to a CSV file."""
        if self.model_type == ClassificationModelEnum.DECISION_TREE:
            X_test, _ = preprocess_for_decision_tree(self.test_dataset)
            predictions = self.model.predict(X_test)
        else:
            self.model.eval()
            predictions = []
            reaction_ids = []
            with torch.no_grad():
                for batch in self.test_loader:
                    if self.model_type == ClassificationModelEnum.MLP:
                        source, destination, _ = batch
                        outputs = self.model(source, destination)
                    elif self.model_type == ClassificationModelEnum.GCN:
                        edge_index, features, batch_indices, _ = batch
                        outputs = self.model(edge_index, features, batch_indices)
                    predicted = (outputs > 0.5).float().numpy()
                    predictions.extend(predicted)

        reaction_ids = self.test_dataset.data["reaction id"].tolist()

        df = pd.DataFrame({"reaction id": reaction_ids, "direction": predictions})
        df["direction"] = df["direction"].map({1: True, 0: False})
        df.to_csv(self.output_file, index=False)
        print(f"Predictions saved to {self.output_file}")
