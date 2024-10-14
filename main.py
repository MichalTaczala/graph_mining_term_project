
import glob
import os
import pandas as pd
from solvers.classification_model import ClassificationSolver

DATASETS_FOLDER_PATH = 'dataset/'

if __name__ == "__main__":


    csv_files = glob.glob(os.path.join(DATASETS_FOLDER_PATH, "*.csv"))

    frames_classification = {}
    frames_prediction = {}
    frames_completion = {}
    current_frame = None
    for file in csv_files:
        file_name = os.path.basename(file)
        file_name_no_extension, _ = os.path.splitext(file_name)
        if file_name_no_extension.startswith("Classification"):
            current_frame = frames_classification
        elif file_name_no_extension.startswith("Completion"):
            current_frame = frames_completion
        else:
            current_frame = frames_prediction
        current_frame[file_name_no_extension] = pd.read_csv(file)

    classification_solver = ClassificationSolver(frames=frames_classification)
    classification_accuracy = classification_solver.solve()
    print(classification_accuracy)
