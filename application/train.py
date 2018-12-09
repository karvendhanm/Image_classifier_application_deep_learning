import argparse
import time
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("data_directory", metavar = "", help = "Enter training data filepath")
parser.add_argument("-sd","--save_dir", metavar = "", required = True, help = "Set directory to save checkpoints")
parser.add_argument("-ah", "--arch", metavar = "",  required = True, help = "Choose architecture", choices = ["vgg11","vgg13","vgg16","vgg19"])
parser.add_argument("-lr", "--learning_rate", type = float, metavar = "",  required = True, help = "Enter learning rate")
parser.add_argument("-hu", "--hidden_units", type = int,  required = True, metavar = "", help = "Enter number of hidden units")
parser.add_argument("-ep", "--epochs", type = int,   required = True, metavar = "", help = "Enter number of epochs")

group = parser.add_mutually_exclusive_group()
group.add_argument("-G","--gpu", help = "Use gpu for model training(recommended)", action = "store_true")
group.add_argument("-C","--cpu", help = "Use cpu for model training(not recommended)", action = "store_true")
args = parser.parse_args()

data_directory = str(args.data_directory)
save_dir = str(args.save_dir)


def build_classifier(input_size, output_size, hidden_layers, drop = [0.5]):
    classifier = nn.Sequential(OrderedDict([
        ("fc1",nn.Linear(input_size, hidden_layers[0])),
        ("relu1",nn.ReLU()),
        ("drop1",nn.Dropout(drop[0])),
        ("fc2",nn.Linear(hidden_layers[0],output_size)),
        ("output",nn.LogSoftmax(dim = 1)),
        ]))
    return classifier

def validation(model, testloader, criterion,device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim = 1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return test_loss, accuracy
    
def train_model(model, trainloader, epochs, print_every, criterion, optimizer,device):
    model.to(device)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if(steps % print_every == 0):
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validationloader, criterion,device)
                print("epochs {}/{}...".format(e+1, epochs), 
                      "Training loss: {:.4f}".format(running_loss/print_every),
                      "Validation loss: {:.3f}".format(test_loss/len(validationloader)),
                      "Validation accuracy: {:.3f}".format(accuracy/len(validationloader))
                      )
                running_loss = 0
                model.train()


train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'
test_dir = data_directory + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, shuffle = True, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32)
validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size = 32)

# import json
# with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)
    
# TODO: Build and train your network
if str(args.arch) == "vgg11":
    model = models.vgg11(pretrained = True)
elif str(args.arch) == "vgg13":
    model = models.vgg13(pretrained = True)
elif str(args.arch) == "vgg16":
    model = models.vgg16(pretrained = True)
elif str(args.arch) == "vgg19":
    model = models.vgg19(pretrained = True)
    

for param in model.parameters():
    param.requires_grad = False
    
input_size = 25088
output_size = 102
hidden_layers = [args.hidden_units]
drop = [0.5]
classifier = build_classifier(input_size, output_size, hidden_layers,drop)
model.classifier = classifier

learning_rate = args.learning_rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

print_every = 40
epochs = args.epochs
if args.cpu:
    device = "cpu"
elif args.gpu:
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        print ("gpu not found")
        
        
        
train_model(model, trainloader, epochs, print_every, criterion, optimizer,device)

# TODO: Save the checkpoint 
checkpoint_dict = {
    "input_size":input_size,
    "output_size":output_size,
    "hidden_layers":hidden_layers,
    "drop":drop,
    "learning_rate":learning_rate,
    "epochs":epochs,
    "state_dict":model.state_dict(),
    "class_to_idx":train_datasets.class_to_idx,
    "optimizer_dict":optimizer.state_dict(),
    "classifier": model.classifier,
    "model_used": args.arch,
    "optimizer":optimizer # this line
}
checkpoint_filename = args.save_dir + "/checkpoint.pth" 
torch.save(checkpoint_dict, checkpoint_filename)
