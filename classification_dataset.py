import pandas as pd
from torch.utils.data import Dataset

class ReactionDataset(Dataset):
    def __init__(self, data_file, answers_file=None, is_training=True):
        self.data = pd.read_csv(data_file)
        self.is_training = is_training
        self.data_y = None
        if not is_training and answers_file:
            self.answers = pd.read_csv(answers_file)
            self.data_x = self.data
            self.data_y = self.answers['direction']
        if is_training:
            self.data_x = self.data.drop('direction', axis=1)
            self.data_y = self.data['direction']
        else:
            self.data_x = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data_x.iloc[idx]
        source_metabolites = list(eval(row['source']))
        destination_metabolites = list(eval(row['destination']))
        if self.data_y is not None:
            label = self.data_y.iloc[idx]
        else:
            label = 0

        # label = 1 if self.data_y.iloc[idx] else 0
        return source_metabolites, destination_metabolites, label
    def get_max_metabolite_id(self):
        if not self.is_training:
            return 10000
        max_id = 0
        for idx in range(len(self.data_x)):
            row = self.data_x.iloc[idx]
            source_metabolites = list(eval(row['source']))
            destination_metabolites = list(eval(row['destination']))
            max_id = max(max_id, max(source_metabolites + destination_metabolites))
        return max_id + 1
