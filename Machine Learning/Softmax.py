#######################################################################################################################
# MULTINOMIAL LOGISTIC REGRESSION (SOFTMAX)
#######################################################################################################################

#Work inspired from Sebastian Raschka https://github.com/rasbt/deeplearning-models 
# This is model of multinomial logistic regression in order to predict the particles' magnetic confinement.
#The datasets must be contained in the same folder as the code itself.
#The model was built with the help of Pytorch librairies. 
# This code is run with one CPU.

#######################################################################################################################
# Import Libraries
#######################################################################################################################
import os
import numpy as np
import pandas as pd
import glob
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#######################################################################################################################
### SETTINGS
#######################################################################################################################

# Hyperparameters
num_features = 6
num_classes = 2

#test/valid dataset size
test_size = 0.3
valid_size = 0.15

# Hyperparameters
random_seed = 123
learning_rate = 1e-4
num_epochs = 100
batch_size = 50000

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################################################################################################
##############################DECLARATION OF CLASSES###################################################################
#######################################################################################################################

#######################################################################################################################
### Class Data Loaders
#######################################################################################################################

class MyDataset(Dataset):

    def __init__(self, X, y):
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        training_example, training_label = self.X[index], self.y[index]
        return training_example, training_label

    def __len__(self):
        return self.y.shape[0]

#######################################################################################################################
### MODEL
#######################################################################################################################

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
            
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
            
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
model = SoftmaxRegression(num_features=num_features,
                            num_classes=num_classes)
model.to(device)

#######################################################################################################################
##############################IMPORT DATASET###########################################################################
#######################################################################################################################

with h5py.File("/home/gabrielv/simulation2.h5", "r") as hdf:
    ls = list(hdf.keys())
    print("List of species of particles: \n", ls)
    data1 = hdf.get("Xe+")
    data2 = hdf.get("Xe++")
    print("List of datasets: \n", list(data1.keys()))
    x1 = data1.get("features")
    x1 = np.array(x1)
    y1 = data1.get("target")
    y1 = np.array(y1)
    print("Size of features:", x1.shape, "Size of target:", y1.shape)
    print("List of datasets: \n", list(data2.keys()))
    x2 = data2.get("features")
    x2 = np.array(x2)
    y2 = data2.get("target")
    y2 = np.array(y2)
    print("Size of features:",x2.shape,"Size of target:", y2.shape)

#######################################################################################################################
##############################SIMULATION FOR Xe+#######################################################################
#######################################################################################################################

#######################################################################################################################
### DATASET for Xe+
#######################################################################################################################
X, y = x1, y1
X = X.astype(np.float)
y = y.astype(np.int)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)
print('length X', len(X))

# Shuffling & train/test split
split_tt = int((1-test_size)*len(x1))
split_vt = int((1-valid_size)*len(x1))
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:split_tt]], X[shuffle_idx[split_tt:]]
y_train, y_test = y[shuffle_idx[:split_tt]], y[shuffle_idx[split_tt:]]
X_valid, y_valid =  X[shuffle_idx[split_vt:]], y[shuffle_idx[split_vt:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
X_valid = (X_valid - mu) / sigma

print(len(X_test)+len(X_train))

#######################################################################################################################
#SPLIT THE DATASET FOR THE TESTING SET AND VALIDATION SET
#######################################################################################################################
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
valid_dataset = MyDataset(X_valid, y_valid)


train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True # want to shuffle the dataset
                          ) # number processes/GPUs to use)

test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)

valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False)

#######################################################################################################################
### COST AND OPTIMIZER
#######################################################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Commented out IPython magic to ensure Python compatibility.
# Manual seed for deterministic data loader
torch.manual_seed(random_seed)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples, acc = 0, 0, 0
    cross_entropy, loss = 0., 0.
    for features, targets in data_loader:
        features = features.cuda()
        targets = targets.cuda()
        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits,targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        acc = correct_pred / num_examples * 100
        if (acc == 0):
            pass
        loss = cross_entropy/num_examples * 100
            
    return acc, loss

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

for epoch in range(num_epochs):
        
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
            
        features = features.cuda()
        targets = targets.cuda()
                
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
            
        # note that the PyTorch implementation of
        # CrossEntropyLoss works with logits, not
        # probabilities
        cost = F.cross_entropy(logits,targets).cuda()
        optimizer.zero_grad()
        cost.backward()
            
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
            
        ### LOGGING
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy(model, train_loader)
        valid_acc, valid_loss = compute_accuracy(model, valid_loader)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
    if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%% valid accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              train_acc, valid_acc))
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

#######################################################################################################################
#Plot Traning and Validation Loss for Xe+
#######################################################################################################################

fig1, (ax1,ax2) = plt.subplots(2, figsize=(8, 8))
fig1.subplots_adjust(hspace=0.3)
ax1.plot(range(1, num_epochs+1), train_loss_lst, label='Training loss')
ax1.plot(range(1, num_epochs+1), valid_loss_lst, label='Validation loss')
ax1.legend(loc='upper right')
ax1.set(ylabel='Loss')
ax1.set(xlabel='Epoch')

ax2.plot(range(1, num_epochs+1), train_acc_lst, label='Training accuracy')
ax2.plot(range(1, num_epochs+1), valid_acc_lst, label='Validation accuracy')
ax2.legend(loc='lower right')
ax2.set(ylabel='Accuracy')
ax2.set(xlabel='Epoch')
fig1.savefig('/home/gabrielv/Figures/Xe+.png')


#######################################################################################################################
#Test accuracy
#######################################################################################################################

model.eval()
with torch.set_grad_enabled(False): # save memory during inference
        test_acc, test_loss = compute_accuracy(model, test_loader)
        print(f'Test accuracy: {test_acc:.2f}%')