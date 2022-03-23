from unittest import TestLoader
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import ssl
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('cifar10')

path2data = 'cifar10/'
if not os.path.exists(path2data):
    os.mkdir(path2data)


train_ds = datasets.CIFAR10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.CIFAR10(path2data, split='test', download=True, transform=transforms.ToTensor())

transformation = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224)])

# apply transformation to dataset
train_ds.transform = transformation
val_ds.transform = transformation



# make dataloade
trainloader= DataLoader(train_ds, batch_size=32, shuffle=True)
testloader = DataLoader(val_ds, batch_size=32, shuffle=True)

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# img_grid = torchvision.utils.make_grid(images)
# writer.add_image('four_fashion_mnist_images', img_grid)
