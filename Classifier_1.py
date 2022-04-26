
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from torchvision.datasets import MNIST


# input the training data the ttar.gz file
# transform the input images input pytorch tensor

# get the test data from the user images from the .zip files

# create a trainging loop with dataloader, model, loss_fn, optimizer

# test funciton used after 10 training examples


# creates neural nerwork class
# define the constructor
# defines foward method


# define loss function
# define optimizer

# loop and call train method

# stochastic gradient decent optomizer drives the learning, paramters are what it adjusts

# loss function
# optomiser

# check overfitting 20;00 in video (run model in data it hasnt seen - test data)


# step 1 - data
target_directory = "mnist"


train_data = MNIST(target_directory, train=True,
                   download=True, transform=ToTensor())

train_data, validate_data = data.random_split(train_data, (48000, 12000))
len(train_data), len(validate_data)  # split the training data into two parts

test_data = MNIST(target_directory, train=True,
                  download=True, transform=ToTensor())

batch_size = 64
# loader for the training data
loader_train = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
# loader for the validation data
loader_validate = data.DataLoader(
    validate_data, batch_size=batch_size, shuffle=False)
loader_test = data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False)  # loader for the test data

###############################


# step 2 - training loop


def train(model, loss_fn, optimizer):
    print("Training")
    for batch, (input, output) in enumerate(train_data):
        print(batch)
###############################


# step 3 - testing loop

def test(model):
    print("Testing")


##############################

# step 4 - model

image_size = 28*28
hidden = 512
classes = 10

model = nn.Sequential(nn.Linear(image_size, hidden),
                      nn.ReLU(), nn.Linear(hidden, classes))
opt = optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


n_epochs = 5

print(model)
for epoch in range(n_epochs):
    print("Epoch", epoch + 1)
    train(model, loss_func, opt)
    test(model)
print("Done")
