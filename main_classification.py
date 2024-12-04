from classification_dataset import ReactionDataset
from classification_model_enum import ClassificationModelEnum
from solver import Solver



train_dataset = ReactionDataset(data_file="dataset/Classification_training.csv")
val_dataset = ReactionDataset(
    data_file="dataset/Classification_valid_query.csv", 
    answers_file="dataset/Classification_valid_answer.csv", 
    is_training=False
)
max_metabolite_id = train_dataset.get_max_metabolite_id()
test_dataset = ReactionDataset(data_file="dataset/Classification_test_query.csv", is_training=False)


#CHANGE MODEL TYPE HERE

# model_type = ClassificationModelEnum.GCN
# model_type = ClassificationModelEnum.MLP
model_type = ClassificationModelEnum.DECISION_TREE

#CHANGE MODEL TYPE HERE


solver = Solver(model_type, train_dataset, val_dataset, max_metabolite_id=max_metabolite_id, embedding_dim=64, epochs=10,output_file="Classification_test_answer.csv", test_dataset=test_dataset)
solver.train()
solver.evaluate()
solver.predict_and_save()
