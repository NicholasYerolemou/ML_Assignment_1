
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from torchvision.datasets import MNIST

import time
# step 1 - data
start = time.time()

# from Tutorial
target_directory = ""


def flatten(inp):
    return inp.reshape(-1)


transform = transforms.Compose([transforms.ToTensor(), flatten])

train_data = MNIST(target_directory, train=True,
                   download=True, transform=transform)

print("The shape is", train_data[0][0].shape)

train_data, validate_data = data.random_split(train_data, (48000, 12000))
len(train_data), len(validate_data)  # split the training data into two parts

test_data = MNIST(target_directory, train=True,
                  download=True, transform=transform)

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

# from example
def train(model, loss_fn, optimizer):
    print("Training")
    for batch, (input, target_output) in enumerate(loader_train):
        #input, target_output = input.to(device), target_output.to(device)

        # the predicted values when model is passed the input
        predicted = model(input)
        # difference between out value and target value
        loss = loss_fn(predicted, target_output)

        # backpropogation
        optimizer.zero_grad()  # why do we zero the gradient of the optomiser
        loss.backward()
        optimizer.step()

        size = len(loader_train.dataset)
        part = size/10

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

###############################

# from example
# step 3 - testing loop


def test(model, loss_func):
    size_test = len(loader_test.dataset)
    size_validation = len(loader_validate.dataset)
    model.eval()  # puts it into evalutaion mode to test rather than train
    test_loss, test_correct = 0, 0
    val_loss, val_correct = 0, 0

    with torch.no_grad():  # disables gradient calculation
        for input, target_output in loader_validate:  # loop through dntire validation data set
            pred = model(input)  # what the model predicts
            # sums the loss for each element in the data set
            val_loss += loss_func(pred, target_output).item()
            val_correct += (pred.argmax(1) ==
                            target_output).type(torch.float).sum().item()  # the sum of every element thte ANN correctly classifies

    test_loss /= size_test  # calculates the average loss over the test data set
    # calculates percentage of elements the ANN correctly classifies
    test_correct /= size_test

    with torch.no_grad():  # disables gradient calculation
        for input, target_output in loader_test:  # loop through dntire test data set
            pred = model(input)  # what the model predicts
            # sums the loss for each element in the data set
            test_loss += loss_func(pred, target_output).item()
            test_correct += (pred.argmax(1) ==
                             target_output).type(torch.float).sum().item()  # the sum of every element thte ANN correctly classifies
    # pred.argmax(1) gets the digit that has the highest probability. i.e. what digit the ANN classifies the image as

    test_loss /= size_test  # calculates the average loss over the test data set
    # calculates percentage of elements the ANN correctly classifies
    test_correct /= size_test

    val_loss /= size_validation
    val_correct /= size_validation

    print()
    print("val size", size_validation)
    print("Validation data set")
    print("validation loss", val_loss)
    print("Correct", val_correct)
    print(
        f"Validation error: \n Accuracy: {(100*val_correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    print("Test data set")
    print("test size", size_test)
    print("test loss", test_loss)
    print("Correct", test_correct)
    print(
        f"Test Error: \n Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################

# step 4 - model


# from tutorial
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 28*28
hidden = 512
classes = 10

model = nn.Sequential(nn.Linear(image_size, hidden),
                      nn.ReLU(),
                      nn.Linear(hidden, hidden),
                      nn.ReLU(),
                      nn.Linear(hidden, classes),
                      nn.ReLU())
opt = optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


n_epochs = 5

print(model)
for epoch in range(n_epochs):
    print("Epoch", epoch + 1)
    train(model, loss_func, opt)
    test(model, loss_func)
end = time.time()
print("Time:", end-start)
print("Done")


"""
path = input("Please enter a file path:")


while(path != "exit" or path != "EXIT"):
    image = data.DataLoader(path, batch_size=1, shuffle=False)
    print(model(image))
"""
