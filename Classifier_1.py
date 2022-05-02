
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


# step 1 - data
target_directory = ""


def flatten(inp):
    return inp.reshape(-1)


transform = transforms.Compose([transforms.ToTensor(), flatten])

train_data = MNIST(target_directory, train=True,
                   download=True, transform=transform)

print("the shape is", train_data[0][0].shape)

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


# step 3 - testing loop

def test(model, loss_func):
    size = len(loader_test)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, target_output in loader_test:
            pred = model(input)
            test_loss += loss_func(pred, target_output).item()
            correct += (pred.argmax(1) ==
                        target_output).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("test loss", test_loss)
    print("Size:", size)
    print("Correct", correct)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################

# step 4 - model


device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 28*28
hidden = 512
classes = 10

model = nn.Sequential(nn.Linear(image_size, hidden),
                      nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, classes))
opt = optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


n_epochs = 5

print(model)
for epoch in range(n_epochs):
    print("Epoch", epoch + 1)
    train(model, loss_func, opt)
    test(model, loss_func)
print("Done")
