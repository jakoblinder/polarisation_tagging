import sys
import numpy as np
import os
import pickle
import copy
import argparse
import datetime
import json5
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# model_name = 'single_model20230413_164724_100kE_100B_lr1e-4'
# model_name = 'single_model20230413_183207_1kE_500B'
# model_name = 'single_model_20230506_105057'
model_name = 'single_model_20230502_091942'
saved_model_filename = f"/data/mincudin/IBMpresent_pol/Code/models_fulltraining_new/{model_name}.model"
saved_dataset_path = f"/data/mincudin/IBMpresent_pol/Code/datasets_fulltraining_new/{model_name}/"
Path(saved_dataset_path).mkdir(exist_ok=True)
# model_configuration = json5.load(open(saved_model_filename + ".json"))
# number_of_neurons = model_configuration['number_of_neurons']
model_configuration = None
number_of_neurons = 1000

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim):
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.head = nn.Linear(self.in_dim, self.emb_dim[0])
        self.linear1 = nn.Linear(self.emb_dim[0], self.emb_dim[1])
        self.linear2 = nn.Linear(self.emb_dim[1], self.emb_dim[2])
        self.linear3 = nn.Linear(self.emb_dim[2], self.emb_dim[1])
        self.linear4 = nn.Linear(self.emb_dim[1], self.emb_dim[0])
        self.out = nn.Linear(self.emb_dim[0], self.out_dim)

    def forward(self, x):
        x1 = F.relu(self.head(x))
        x2 = F.relu(self.linear1(x1))
        x3 = F.relu(self.linear2(x2))
        x4 = F.relu(self.linear3(x3))
        x5 = F.relu(self.linear4(x2))
        out = self.out(x5)

        return out

def evaluate(the_model, the_X):

    predictions = []
    n_elements = the_X.shape[0]

    for idx in range(n_elements):
        # print("Processing element: ", the_X[idx])
        prediction = the_model(the_X[idx])
        predictions.append(prediction.detach().numpy())
        if idx % 1000 == 0:
            print(".", end="", flush=True)

    return np.array(predictions)


def save_data(features, labels, filename):
    matrix = np.concatenate([features, labels], axis = 1)
    np.savetxt(filename, matrix)

# =============== DATA LOADING ===============
print("Data loading", flush=True)
file_txt = '/data/mincudin/restrictedPhaseSpace/pp2zj_unpol_pythia_withQED_cut1_forTest.txt'
dataset = np.loadtxt(file_txt)
features = dataset # np.delete(dataset,[-1], axis=1)
label = dataset[:,-1]
print("shape of the dataset", dataset.shape)

# =============== DATA PREPROCESSING ===============
MAX_ELEMENTS = len(features)
X = features[:MAX_ELEMENTS]
# y = label[:MAX_ELEMENTS].reshape((-1, 1))

input_dim = X.shape[1]
output_dim = 1

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X = (X - means) / stds

np.random.seed(1234567)
# all_indexes = np.random.choice([2], MAX_ELEMENTS, p=[1.0])

# X_train = X[all_indexes == 1]
# X_val = X[all_indexes == 2]
# X_test = X[all_indexes == 3]
# y_train = y[all_indexes == 1]
# y_val = y[all_indexes == 2]
# y_test = y[all_indexes == 3]
# assert len(np.unique(y_test)) > 4, "All labels are the same, error!"

# =============== DATA PREPROCESSING (pytorch) ===============
# train_features = torch.from_numpy(X_train)
# train_labels   = torch.from_numpy(y_train)
# val_features   = torch.from_numpy(X_val)
# val_labels     = torch.from_numpy(y_val)
# test_features  = torch.from_numpy(X_test)
# test_labels    = torch.from_numpy(y_test)

# train_features = train_features.float()
# train_labels   = train_labels.float()
# val_features   = val_features.float()
# val_labels     = val_labels.float()
# test_features  = test_features.float()
# test_labels    = test_labels.float()

pyt_features = torch.from_numpy(X)
pyt_features = pyt_features.float()

# =============== MODEL CREATION ===============
print("Model creation", flush=True)
input_dim = pyt_features.shape[1]
output_dim = 1

trained_model = Net(input_dim, output_dim, emb_dim=[number_of_neurons] * 3)
trained_model.load_state_dict(torch.load(saved_model_filename), strict = True)

# =============== PREDICTIONS ===============
print("Predictions", flush=True)
test_labels_predicted = evaluate(trained_model, pyt_features)

# =============== OUTPUT ===============
print("Output", flush=True)
# test_features_numpy = pyt_features.detach().numpy()
numpy_features = pyt_features.detach().numpy()
numpy_labels = test_labels_predicted
# denormalized_test_features_numpy = stds * test_features_numpy + means

print("Final shape", numpy_labels.shape)

# # print("Checking: ")
# # checking_index = 2
# # print("Original feature: ", features[test_indexes[checking_index]])
# # print("Recreated feature: ", denormalized_test_features_numpy[checking_index])
# save_data(denormalized_test_features_numpy, test_labels, saved_dataset_path + "events_testing_groundtruth.dat")
save_data(features, numpy_labels, saved_dataset_path + "events_testing_predicted_2_no_normalization.dat")

