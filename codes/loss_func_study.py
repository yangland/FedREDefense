import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import *
from torch.utils.data import DataLoader
from models import *
import random
from collections import Counter
from collections import OrderedDict
import seaborn as sns
import copy

cos = nn.CosineSimilarity(dim=0, eps=1e-9)
device = "cuda"

# study 1 model's training with the new loss function with dist limits

# adjustable parameters
alpha_d = 100 # IID
local_ep = 20 
n_clients = 30 # dataset size for one client
mali_local_ep = 10
global attack 
attack = "untargeted" #"backdoor", "tlp", "ut"
model_name = "ConvNet" # "resnet8", "ConvNet"
num_classes = 10
dataset ="fmnist"

def cos_dist(w1, w2):
    """Compute cosine similarity between two flattened weight tensors"""
    w1_flat, w2_flat = torch.cat([p.view(-1) for p in w1]), torch.cat([p.view(-1) for p in w2])
    return 1 - torch.dot(w1_flat, w2_flat) / (torch.norm(w1_flat) * torch.norm(w2_flat))

def get_delta_cos(model1, model2, model0_sd):
    flat_model0 = flat_dict(model0_sd)
    flat_model1 = flat_dict(model1.state_dict())
    flat_model2 = flat_dict(model2.state_dict())
    
    delta = torch.abs(flat_model1 - flat_model2)
    org_cos = cos((flat_model1 - flat_model0), (flat_model2 - flat_model0))
    return delta, 1-org_cos.item()

def model_eval(model, test_loader, attack):
    acc = eval_op_ensemble([model], test_loader)
    if attack == "tlp":
        asr = eval_op_ensemble_tr_lf_attack([model], test_loader)
    elif attack == "backdoor":
        asr = eval_op_ensemble_attack([model], test_loader)
    elif attack == "untargeted":
        asr = None
    return list(acc.values())[0], list(asr.values())[0]

def reverse_train_w_cos(model, loader, optimizer, epochs, model0_sd, model1_sd, beta, budget):    
    model.train()

    grad_ben = (flat_dict(model1_sd) - flat_dict(model0_sd)).to(device)
    
    losses = []
    # import pdb; pdb.set_trace()
    running_loss, samples = 0.0, 0
    for ep in range(epochs):
        for it, (x, y) in enumerate(loader):
            if it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_ce = nn.CrossEntropyLoss()(model(x), y)
            # in the untraining reverse the sign of loss
            loss_ce = - loss_ce
            running_loss += loss_ce.item() * y.shape[0]
            samples += y.shape[0]
            
            # add cos loss 
            w = torch.cat([p.clone().detach().view(-1) for p in model.parameters()]).to(device)
            grad_mail = w - flat_dict(model0_sd)
            target = torch.ones(len(w)).to(device)
            loss_cos = nn.CosineEmbeddingLoss()(grad_ben, grad_mail, target)
            loss_obj = (1-beta) * loss_ce + beta * loss_cos
            loss_obj.backward()
            optimizer.step()
            print(f"ep{ep}, loss_cs: {loss_ce}, loss_cos: {loss_cos}, loss_obj: {loss_obj}")
        
        # break
        cos_d = cos_dist(grad_ben, grad_mail)
        print("eval losses", losses)
        
        if cos_d <= budget:
            break
        

    return {"loss": running_loss / samples}
    


def main():
    # Define transformation (convert images to tensors and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image with mean and std
    ])

    # Load the training dataset
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Load the test dataset
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader for batch processing
    client_loaders, test_loader, client_data_subsets =\
        data.get_loaders(train_data, test_data, n_clients,
                        alpha=alpha_d, batch_size=32, n_data=None, num_workers=4, seed=4)
        
    model_fn = partial(models.get_model(model_name)[
                            0], num_classes=num_classes, dataset=dataset)

    client_loader = client_loaders[0]

    # created models 
    model0 = model_fn().to(device) # orginal model
    model1 = model_fn().to(device) # train with clean data
    model2 = model_fn().to(device) # train with new loss function
    model3 = model_fn().to(device)

    model0_sd = {k: v.clone().detach() for k, v in model1.state_dict().items()}

    optimizer0 = optim.SGD(model0.parameters(), lr=0.001)
    optimizer1 = optim.SGD(model1.parameters(), lr=0.001)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.001)
    optimizer3 = optim.SGD(model3.parameters(), lr=0.001)
    
    # model1 train benign
    train_op(model1, client_loader, optimizer1, epochs=local_ep, print_train_loss=True)

    model1_sd = {key: value.clone() for key, value in model1.state_dict().items()}
    
    # model2 train with new loss function
    model2.load(model1_sd)
    model2 = reverse_train_w_cos(model2, client_loader, optimizer2, epochs=1, 
                                 model0_sd = model0_sd, 
                                 model1_sd = model1_sd, 
                                 beta = 0.5, 
                                 budget = 0.4)
    
    model1_result = eval_op_ensemble(model1, test_loader)
    print("model1_result", model1_result)
    
    cos_d_model1_2 = cos_dist(model1, model2)
    print("model1_2 cos dist", cos_d_model1_2)
    
    model2_result = eval_op_ensemble(model2, test_loader)
    print("model2_result", model2_result)