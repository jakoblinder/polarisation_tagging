import sys
import numpy as np
import os
import pickle
import copy
import argparse
import datetime
# import matplotlib.pyplot as plt
import json5
from joblib import Parallel, delayed
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pathlib import Path


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


#np.random.seed(123456)
#NUMPY_SEED_TRAINING_SPLIT = 124553
#NUMPY_SEED_TESTING_SPLIT = 354038
device = torch.device('cuda')

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
N_EPOCHS = 1000
N_BATCH = 500
print(f"Setting {N_EPOCHS=} {N_BATCH=}")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")

print("Loading the input file", flush=True)
# file_txt = '/data/mincudin/events_fs_momenta_and_rl.dat'
file_txt = '/data/mincudin/restrictedPhaseSpace/pp2zj_unpol_pythia_withQED_cut1_forTrain.txt'
dataset = np.loadtxt(file_txt)
print("Dataset has shape ", dataset.shape, flush=True)

# features = ['Emu+' , 'pxmu+', 'pymu+','pzmu+','Emu-','pxmu-','pymu-', 'pzmu-', 'Ej', 'pxj', 'pyj', 'pzj', 'RL']
features = np.delete(dataset,[-1], axis=1)
label = dataset[:,-1]

MAX_ELEMENTS = len(features)
X = features[:MAX_ELEMENTS]
y = label[:MAX_ELEMENTS].reshape((-1, 1))

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X = (X - means) / stds

np.random.seed(1234567)

all_indexes = np.random.choice([1, 2, 3], MAX_ELEMENTS, p=[0.5, 0.25, 0.25])
train_indexes = all_indexes[all_indexes == 1]
val_indexes = all_indexes[all_indexes == 2]
test_indexes = all_indexes[all_indexes == 3]

X_train = X[all_indexes == 1]
X_val = X[all_indexes == 2]
X_test = X[all_indexes == 3]
y_train = y[all_indexes == 1]
y_val = y[all_indexes == 2]
y_test = y[all_indexes == 3]
assert len(np.unique(y_train)) > 4, "All elements in y_test are equal, error during data loading"

input_dim = X_train.shape[1]
output_dim = 1

print("Loading the data on torch", flush=True)
train_features = torch.from_numpy(X_train).to(device)
train_labels   = torch.from_numpy(y_train).to(device)
val_features   = torch.from_numpy(X_val).to(device)
val_labels     = torch.from_numpy(y_val).to(device)
test_features  = torch.from_numpy(X_test).to(device)
test_labels    = torch.from_numpy(y_test).to(device)

train_features = train_features.float()
train_labels   = train_labels.float()
val_features   = val_features.float()
val_labels     = val_labels.float()
test_features  = test_features.float()
test_labels    = test_labels.float()

def run_training(the_model, the_X, the_y, n_epochs, n_batch, lr=0.001):
    """
    :param the_model: instance of 'Net' object
    :param the_X: training features
    :param the_y: training labels
    :param n_epochs: number of epochs of training
    :param n_batch: number of elements for each batch
    :return the history of losses, but the model is trained afterward as a side effect
    """

    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.RMSprop(the_model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

    n_elements = the_X.shape[0]

    history_losses = []

    start_time = datetime.datetime.now()
    last_epoch_time = start_time

    for epoch in range(n_epochs):

        # get a random batch from the dataset
        batch_index = np.random.choice(np.arange(n_elements), n_batch, replace=False)
        batch_X = the_X[batch_index]
        batch_y = the_y[batch_index]
        current_epoch_loss = 0

	# reset the optimizer object
        optimizer.zero_grad()

        # process the current batch
        for idx in range(n_batch):  

            # calculate the label of the current element 
            prediction = the_model(batch_X[idx])

            # calculate the loss with respect to the true value
            loss = mse_loss(prediction, batch_y[idx])
            current_epoch_loss += loss.cpu()

            # calculate the gradient of all the parameters in the_model.parameters having has_gradient=True
            loss.backward()

        # modify the parameters according to the gradient calculated
        optimizer.step()

        # normalize loss
        current_epoch_loss /= n_batch

        # save and print current loss value
        history_losses.append(current_epoch_loss)
        if epoch % 20 == 0:
            current_time = datetime.datetime.now()
            delta_time = current_time - last_epoch_time
            last_epoch_time = current_time
            print(f"Epoch {epoch: 4d} has loss {current_epoch_loss: 15.10f} in {delta_time.total_seconds() / 60: 5.3f} minutes", flush=True)

    return history_losses


def run_evaluation(the_model, the_X, the_y):

    mse_loss = torch.nn.MSELoss()
    n_elements = the_X.shape[0]
    current_loss = 0

    for idx in range(n_elements):
        prediction = the_model(the_X[idx])
        current_loss += mse_loss(prediction, the_y[idx])

    avg_loss = current_loss / n_elements
    return avg_loss


def get_opt_dimensions():
    return [
#        Integer(3, 10), # number of layers
        Integer(500, 1200), # number of neurons
        Real(1e-4, 1e-3), # learning rate
    ]


def run_hyperparameter_optimization(train_features, train_labels, val_features, val_labels, n_iterations):
    
    bayesian_optimizer = Optimizer(
        dimensions=get_opt_dimensions(),
        random_state=1,
        base_estimator='gp',
        acq_func="PI",
        acq_optimizer="sampling",
        acq_func_kwargs={"xi": 10000.0, "kappa": 10000.0}
    )
    
    input_dim = X_train.shape[1]
    output_dim = 1
    
    def hyperparams_cost_function(params):
        # be careful! I cannot pass further variables to the cost function so
        # you either use global variables or closures
        # global input_dim, output_dim 
        # global train_features, train_labels
        # global val_features, val_labels
        print(f"Calling hyperparams_cost_function function with params {params}", flush=True)

        # unpack the parameters
        number_of_layers = 3
        number_of_neurons = int(params[0])
        learning_rate = float(params[1])
        
        # hardcoded parameters (we cannot really optimize EVERYTHING)
        n_epochs = N_EPOCHS
        n_batch = N_BATCH

        # create model
        candidate_model = Net(input_dim, output_dim, emb_dim=[number_of_neurons] * number_of_layers).to(device)
        # train model
        history_loss = run_training(candidate_model, train_features, train_labels, n_epochs=n_epochs, n_batch=n_batch, lr=learning_rate)
        # calculate loss with respect to the validation (NOT testing) set
        print(f"Starting evaluation", flush=True)
        validation_mse = run_evaluation(candidate_model, val_features, val_labels)
        # logging
        print(f"\nWith parameters {params} ({number_of_layers=} {number_of_neurons=} {learning_rate=}) the validation MSE is {validation_mse}", flush=True)
        # serialization
        date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        saved_model_filename = 'models_2/trained_model_'+ date_time_now + '.model'
        saved_model = torch.save(candidate_model.state_dict(), saved_model_filename)
        with open('models_2/trained_model_'+ date_time_now + '.json', 'w') as outfile:
            json5.dump({'number_of_layers': number_of_layers, 
                       'number_of_neurons': number_of_neurons,
                       'learning_rate': learning_rate, 
                       'n_epochs': n_epochs,
                       'n_batch': n_batch,
                       'validation_mse': float(validation_mse)
                      }, outfile)
        return float(validation_mse.detach().cpu().numpy())

    # run bayesian optimization
    for i in range(n_iterations):
        print(f"Epoch of training {i=}", flush=True)
        # sample some initial point for the optimization epoch
        x = bayesian_optimizer.ask(n_points=8) 
        # calculate the cost of sampled points
        #y = Parallel(n_jobs=8)(delayed()# evaluate in parallel
        y = [hyperparams_cost_function(v) for v in x]  # evaluate points
        # update optimizer
        # print(y, type(y))
        bayesian_optimizer.tell(x, y)
    
    # get the solution having the lowest cost
    min_index = np.argmin(bayesian_optimizer.yi)
    min_cost = bayesian_optimizer.yi[min_index]
    min_solution = bayesian_optimizer.Xi[min_index]
    
    return {
        'bayesian_opt_solution': min_solution, 
        'number_of_layers': 3,
        'number_of_neurons': int(min_solution[0]),
        'learning_rate': float(min_solution[1])
    }

# create models directory if not exists
Path("models_fulltraining_new").mkdir(exist_ok=True)

# print("Starting hyperparam optimization...", flush=True)
# solution = run_hyperparameter_optimization(train_features, train_labels, val_features, val_labels, 20)
# print(f"Best solution is {solution}", flush=True)

# print("Finalizing the best model", flush=True)
# good_model = Net(input_dim, output_dim, emb_dim=[solution['number_of_neurons']] * solution['number_of_layers'])
# run_training(good_model, train_features, train_labels, n_epochs=N_EPOCHS, n_batch=N_BATCH, lr=solution['learning_rate'])

# print("Saving best model to file", flush=True)
# date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
# saved_model_filename = 'models/good_model_'+ date_time_now + '.model'
# saved_model = torch.save(good_model.state_dict(), saved_model_filename)
# json5.dump({'number_of_layers': solution['number_of_layers'], 
#             'number_of_neurons': solution['number_of_neurons'],
#             'learning_rate': solution['learning_rate'], 
#             'n_epochs': N_EPOCHS,
#             'n_batch': N_BATCH
#             }, open('models_2/good_model_'+ date_time_now + '.json', 'w'))

# create model
candidate_model = Net(input_dim, output_dim, emb_dim=[1000, 1000, 1000]).to(device)
# candidate_model.load_state_dict(torch.load("models_2/single_model20230413_233134_2kE_500B.model"), strict = True)
# train model
history_loss = run_training(candidate_model, train_features, train_labels, n_epochs=N_EPOCHS, n_batch=N_BATCH, lr=1e-4)
# calculate loss with respect to the validation (NOT testing) set
validation_mse = run_evaluation(candidate_model, val_features, val_labels)
# serialization
date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
saved_model_filename = 'models_fulltraining_new/single_model_'+ date_time_now + '.model'
saved_model = torch.save(candidate_model.state_dict(), saved_model_filename)
