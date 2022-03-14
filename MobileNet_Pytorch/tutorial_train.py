import torch.optim as optim
import torch.nn as nn
from model import *
from dataset import *
import matplotlib.pyplot as plt
import numpy as np

'''
trainloader = train_dl
testloader = val_dl
그냥 데이터를 여기서 불러오자.
'''




CEE = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Do I need tqdm?  ->  what is tqdm.set_postfix()
# for epoch in range(2):  # loop over the dataset multiple times

Mean_loss = []

for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = CEE(outputs, labels)
    Mean_loss.append(loss.item())
    loss.backward()
    optimizer.step()
        

print(f'Mean loss was {sum(Mean_loss) / len(Mean_loss)}')