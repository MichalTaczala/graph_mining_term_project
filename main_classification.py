import argparse

from classification_dataset import ReactionDataset
from classification_model_enum import ClassificationModelEnum
from solver import Solver


parser = argparse.ArgumentParser(description="Set model type for classification.")
parser.add_argument(
    "--model_type", 
    type=str, 
    required=True, 
    choices=["GCN", "MLP", "DECISION_TREE"], 
    help="Type of model to use for classification. Options: GCN, MLP, DECISION_TREE."
)
args = parser.parse_args()

# Map string argument to ClassificationModelEnum
model_type_map = {
    "GCN": ClassificationModelEnum.GCN,
    "MLP": ClassificationModelEnum.MLP,
    "DECISION_TREE": ClassificationModelEnum.DECISION_TREE
}
model_type = model_type_map[args.model_type]
train_dataset = ReactionDataset(data_file="dataset/Classification_training.csv")
val_dataset = ReactionDataset(
    data_file="dataset/Classification_valid_query.csv", 
    answers_file="dataset/Classification_valid_answer.csv", 
    is_training=False
)
max_metabolite_id = train_dataset.get_max_metabolite_id()
test_dataset = ReactionDataset(data_file="dataset/Classification_test_query.csv", is_training=False)




solver = Solver(model_type, train_dataset, val_dataset, max_metabolite_id=max_metabolite_id, embedding_dim=64, epochs=30,output_file="Classification_test_answer.csv", test_dataset=test_dataset)
solver.train()
solver.evaluate()
solver.predict_and_save()
