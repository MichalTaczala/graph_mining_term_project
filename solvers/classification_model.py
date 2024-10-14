from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import hypernetx as hnx
from models.models_constants import ModelEnum


class ClassificationSolver():
    def __init__(self, frames: Dict[str, pd.DataFrame], model: ModelEnum = ModelEnum.LINEAR_REGRESSION) -> None:
        self.training = self._preprocess_training_data(frames["Classification_training"])
        self.valid_query = self._preprocess_query(frames["Classification_valid_query"])
        self.test_query = self._preprocess_query(frames["Classification_test_query"])
        self.valid_answer = self._preprocess_valid_answer(frames["Classification_valid_answer"])
        self.model = model
       
    def _preprocess_training_data(self, tr):
        tr["source"] = tr["source"].apply(lambda x: set(x[1:-1].split(", ")))
        tr["destination"] = tr["destination"].apply(lambda x: set(x[1:-1].split(", ")))
        tr["direction"] = tr["direction"].map({True: 1, False: 0})
        return tr

    def _preprocess_query(self, test_query):
        test_query["source"] = test_query["source"].apply(lambda x: set(x[1:-1].split(", ")))
        test_query["destination"] = test_query["destination"].apply(lambda x: set(x[1:-1].split(", ")))
        return test_query

    def _preprocess_valid_answer(self, valid_answer):
        valid_answer["direction"] = valid_answer["direction"].map({True: 1, False: 0})
        return valid_answer

    def solve(self):        
        if self.model == ModelEnum.TREE_CLASSIFIER:
            return self.solve_tree_classifier()
        else:
            hyperedges = {}

            for idx, row in self.training.iterrows():
                print(row.head())
                # Use reaction ID to label the hyperedges
                hyperedges[f'{row["reaction id"]}_source'] = row['source']  # Source metabolites
                hyperedges[f'{row["reaction id"]}_dest'] = row['destination']  # Destination metabolites

            # Create the hypergraph
            H = hnx.Hypergraph(hyperedges)


    def solve_tree_classifier(self):
        X_train = np.array([self._extract_features(row) for row in self.training.itertuples()])
        y_train = self.training["direction"].values
        X_valid = np.array([self._extract_features(row) for row in self.valid_query.itertuples()])
        y_valid = self.valid_answer["direction"].values
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)

    def _extract_features(self, row):
        source_set = row.source
        destination_set = row.destination
        source_size = len(source_set)
        destination_size = len(destination_set)
        intersection = len(source_set.intersection(destination_set))
        union = len(source_set.union(destination_set))
        jaccard_similarity = intersection / union if union > 0 else 0
        size_difference = abs(source_size - destination_size)
        return [source_size, destination_size, jaccard_similarity, size_difference]

