#######################################################################################################################
# MULTINOMIAL LOGISTIC REGRESSION (SOFTMAX) with Data Distributed Parallel method
#######################################################################################################################

#Work inspired from Sebastian Raschka https://github.com/rasbt/deeplearning-models 
# This is model of multinomial logistic regression in order to predict the particles' magnetic confinement.
#The datasets must be contained in the same folder as the code itself.
#The model was built with the help of Pytorch librairies. 
# This code has been executed with the Data Distributed Parallal method,
# a multi-GPU processing splitting the dataset in multiple batches which they will be analyzed simultaneously.
# a shell script will be presented

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch.distributed as dist
import torch.utils.data.distributed

import argparse



#######################################################################################################################
# Parser Arguement
#######################################################################################################################
parser = argparse.ArgumentParser(description='Confined Particles softmax regression, distributed data parallel test')
parser.add_argument('--lr', default=1e-4, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')

parser.add_argument('--init_method', default='tcp://10.70.35.4:8103', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


print("Starting...")
args = parser.parse_args()
ngpus_per_node = torch.cuda.device_count()
print(ngpus_per_node)

rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + int(os.environ.get("SLURM_LOCALID")) 
print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
#init the process group
dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
print("process group ready!")

print('From Rank: {}, ==> Making model..'.format(rank))



#######################################################################################################################
### SETTINGS
#######################################################################################################################

# Hyperparameters
num_features = 6
num_classes = 2

#test/valid dataset size
test_size = 0.3
valid_size = 0.15

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
model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model)

print('From Rank: {}, ==> Preparing data..'.format(rank))

#######################################################################################################################
##############################IMPORT DATASET###########################################################################
#######################################################################################################################

with h5py.File("/home/gabrielv/simulation.h5", "r") as hdf:
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
    x1 = np.array(x1)
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
split_tt = int((1-test_size)*len(x2))
split_vt = int((1-valid_size)*len(x2))
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

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=(train_sampler is None), # want to shuffle the dataset
                            num_workers=args.num_workers, # number processes/GPUs to use
                            sampler = train_sampler)

test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=(test_sampler is None),
                            num_workers=args.num_workers,
                            sampler = test_sampler)

valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=(valid_sampler is None),
                            num_workers=args.num_workers,
                            sampler = valid_sampler)

#######################################################################################################################
### COST AND OPTIMIZER
#######################################################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# Commented out IPython magic to ensure Python compatibility.
# Manual seed for deterministic data loader

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

for epoch in range(args.max_epochs):
        
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)
        
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

elapsed = (time.time() - start_time)/60
print("From Rank: {}, Training time {}".format(rank, elapsed))
print(f'Total Training Time: {elapsed:.2f} min')

#######################################################################################################################
#Plot Traning and Validation Loss for Xe+
#######################################################################################################################

fig1, (ax1,ax2) = plt.subplots(2, figsize=(8, 8))
fig1.subplots_adjust(hspace=0.3)
ax1.plot(range(1, args.max_epochs+1), train_loss_lst, label='Training loss')
ax1.plot(range(1, args.max_epochs+1), valid_loss_lst, label='Validation loss')
ax1.legend(loc='upper right')
ax1.set(ylabel='Loss')
ax1.set(xlabel='Epoch')

ax2.plot(range(1, args.max_epochs+1), train_acc_lst, label='Training accuracy')
ax2.plot(range(1, args.max_epochs+1), valid_acc_lst, label='Validation accuracy')
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

