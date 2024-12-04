
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeSolver:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def preprocess_for_decision_tree(dataset):
    features = []
    labels = []
    for source, destination, label in dataset:
        source_set = set(source)
        destination_set = set(destination)
        source_size = len(source_set)
        destination_size = len(destination_set)
        intersection = len(source_set.intersection(destination_set))
        union = len(source_set.union(destination_set))
        jaccard_similarity = intersection / union if union > 0 else 0
        size_difference = abs(source_size - destination_size)
        features.append([source_size, destination_size, jaccard_similarity, size_difference])
        labels.append(label)
    return np.array(features), np.array(labels)