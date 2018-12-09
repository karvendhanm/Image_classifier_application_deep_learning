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
import json

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("imagepath",metavar = "", help = "Enter imagepath")
parser.add_argument("checkpoint", metavar = "", help = "Enter checkpoint path")
parser.add_argument("-t", "--topk", metavar = "",type = int, required = True, help = "Enter the number of top classes required")
parser.add_argument("-cn", "--category_names", metavar = "", required = True, help = "Enter the category filepath")

group = parser.add_mutually_exclusive_group()
group.add_argument("-G","--gpu", help = "Use gpu for prediction(recommended)", action = "store_true")
group.add_argument("-C","--cpu", help = "Use cpu for prediction(not recommended)", action = "store_true")
args = parser.parse_args()

if args.cpu:
    device = "cpu"
elif args.gpu:
    if torch.cuda.is_available():
        device = "cpu"
    else:
        print ("gpu not found")    

imagepath = str(args.imagepath)
checkpath = str(args.checkpoint)
category_names = str(args.category_names)

def load_saved_model(filename):
    checkpoint_dict = torch.load(filename)
    if checkpoint_dict["model_used"] == "vgg11":
        model_check = models.vgg11(pretrained = True)
    elif checkpoint_dict["model_used"] == "vgg13":
        model_check = models.vgg13(pretrained = True)
    elif checkpoint_dict["model_used"] == "vgg16":
        model_check = models.vgg16(pretrained = True)
    elif checkpoint_dict["model_used"] == "vgg19":
        model_check = models.vgg19(pretrained = True)
    
    for param in model_check.parameters():
        param.requires_grad = False
    
    optimizer = checkpoint_dict["optimizer"]
    model_check.classifier =  checkpoint_dict["classifier"]
    model_check.load_state_dict(checkpoint_dict["state_dict"]) 
    optimizer.load_state_dict(checkpoint_dict["optimizer_dict"])
    model_check.class_to_idx = checkpoint_dict["class_to_idx"]
    return model_check

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop(box = (16,16,240,240))
    arr = np.array(img)
    arr = np.divide(arr,255)
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    arr = (arr - mean)/std
    arr = arr.transpose((2,0,1))
    
    return arr 

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, filename, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model = load_saved_model(filename)
    model.eval()
    model.to(device)
    model.double()
       
    processed_image = process_image(image_path)
    processed_image = torch.from_numpy(processed_image)
    processed_image = processed_image.unsqueeze_(0)
    processed_image.to(device)
    
        
    with torch.no_grad():
        outputs = model(processed_image)
             
    ps = torch.exp(outputs)
    probs, classes = torch.topk(ps, topk)
    return(probs, classes)
    


with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
probs, classes = predict(imagepath,checkpath,args.topk)
model = load_saved_model(checkpath)
inv_dict = {values : keys for keys, values in model.class_to_idx.items()}
classes_list = [inv_dict[idx] for idx in classes.numpy()[0]]
probs_list = probs.numpy()[0].tolist()
cat_list = [cat_to_name[lbl] for lbl in classes_list]
print(cat_list)
print(classes_list)
print(probs_list)
