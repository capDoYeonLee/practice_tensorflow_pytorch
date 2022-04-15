import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet_model import *
from test import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


    
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        # train(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD)
        
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


####################load data and processing##################

data_dir  = '../Garbage classification/Garbage classification'

classes = os.listdir(data_dir)
transform = transforms.Compose([transforms.Resize((256,256)),
                               transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform = transform)

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 0, pin_memory = True)
test_dl = DataLoader(test_ds, batch_size*2, num_workers = 0, pin_memory = True)


###############################################################
  
model = ResNet().to(device)
# model = to_device(ResNet(), device)

num_epochs = 8
opt_func = torch.optim.Adam
lr = 5.5e-5
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
torch.save(model, 'model.pth')







#######################predict##########################

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


img, label = test_ds[17]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))