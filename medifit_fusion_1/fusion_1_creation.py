# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:41:57 2023

@author: anton
"""

from joblib import load, dump
import numpy as np


class Fusion1Algorithm:
    def __init__(self, training_file1, training_file2, validation_file1, validation_file2):
        self.data_training1 = load(training_file1)
        self.data_training2 = load(training_file2)
        self.data_validation1 = load(validation_file1)
        self.data_validation2 = load(validation_file2)

    def extract_labels_features(self, data):
        labels = data[:, 0]
        features = data[:, 1:6]
        return labels, features

    def combine_features(self, features1, features2):
        return np.hstack((features1, features2))

    def create_combined_data(self, labels, combined_features):
        return np.hstack((labels.reshape(-1, 1), combined_features))

    def process(self):
        labels_training1, features_training1 = self.extract_labels_features(self.data_training1)
        labels_training2, features_training2 = self.extract_labels_features(self.data_training2)
        labels_validation1, features_validation1 = self.extract_labels_features(self.data_validation1)
        labels_validation2, features_validation2 = self.extract_labels_features(self.data_validation2)

        combined_features_training = self.combine_features(features_training1, features_training2)
        combined_features_validation = self.combine_features(features_validation1, features_validation2)

        self.combined_data_training = self.create_combined_data(labels_training1, combined_features_training)
        self.combined_data_validation = self.create_combined_data(labels_validation1, combined_features_validation)

        return self.combined_data_training, self.combined_data_validation

    def save_combined_data(self, training_file, validation_file):
        dump(self.combined_data_training, training_file)
        dump(self.combined_data_validation, validation_file)


class Fusion1AlgorithmExternal:
    
    def __init__(self, training_data1, training_data2, from_files=True):
        if from_files:
            self.data_training1 = load(training_data1)
            self.data_training2 = load(training_data2)
        else:
            self.data_training1 = training_data1
            self.data_training2 = training_data2    
    
    # def __init__(self, training_file1, training_file2):
    #     self.data_training1 = load(training_file1)
    #     self.data_training2 = load(training_file2)

    def combine_features(self, features1, features2):
        return np.hstack((features1, features2))

    # def process(self):
    #     combined_features_training = self.combine_features(self.data_training1, self.data_training2)
    #     return combined_features_training
    
    def process(self):
        self.combined_features_training = self.combine_features(self.data_training1, self.data_training2)
        return self.combined_features_training

    def save_combined_data(self, training_file):
        dump(self.combined_features_training, training_file)
        
        
# USAGE of 1st CLASS:
# training_file1 = load(r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTNIR\data_training.joblib")
# training_file2 = load(r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTIR_ATR\data_training2.joblib")
# validation_file1 = load(r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTNIR\data_validation.joblib")
# validation_file2 = load(r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTIR_ATR\data_validation2.joblib")

# fusion_algorithm = Fusion1Algorithm(r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTNIR\data_training.joblib", r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTIR_ATR\data_training2.joblib", r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTNIR\data_validation.joblib", r"D:\Desktop\Folders\MEDIFIT_More\newer\Original Data\original_processed\FTIR_ATR\data_validation2.joblib")
# combined_data_training, combined_data_validation = fusion_algorithm.process()
# fusion_algorithm.save_combined_data('combined_data_training.joblib', 'combined_data_validation.joblib')


# USAGE of 2nd CLASS:
# fusion_algorithm = Fusion1AlgorithmExternal(training_file1, training_file2)

# # Step 2: Process the data
# combined_features_training = fusion_algorithm.process()

# # Step 3: Save the combined data to a new joblib file
# output_training_file_path = f'{output_folder_path}/fusion_1_data.joblib'
# fusion_algorithm.save_combined_data(output_training_file_path)

# print(f"Combined data saved to {output_training_file_path}")