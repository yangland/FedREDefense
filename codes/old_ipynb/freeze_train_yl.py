import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
cos = nn.CosineSimilarity(dim=0, eps=1e-9)
from torchvision import transforms
# from codes.models import ConvNet, resnet8, MLP
from utils import *
from torch.utils.data import DataLoader
from models import *
import random
device = "cuda"

# adjustable parameters
alpha_d = 1.0
local_ep = 1
points = 41
attack ="tlp" #"bd"
model_name = "ConvNet" # "resnet8"
num_classes = 10
dataset ="fmnist"

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
    data.get_loaders(train_data, test_data, n_clients=100,
                    alpha=alpha_d, batch_size=32, n_data=None, num_workers=4, seed=4)
    
model_fn = partial(models.get_model(model_name)[
                        0], num_classes=num_classes, dataset=dataset)

client_loader = client_loaders[0]

# created models
model1 = model_fn().to(device)
model2 = model_fn().to(device)
model3 = model_fn().to(device)

model0_sd = {k: v.clone().detach() for k, v in model1.state_dict().items()}

optimizer1 = optim.SGD(model1.parameters(), lr=0.001)
optimizer2 = optim.SGD(model2.parameters(), lr=0.001)
optimizer3 = optim.SGD(model3.parameters(), lr=0.001)

def freeze_random_weights(model, freeze_ratio=0.5):
    """
    Freezes a random fraction of individual weights across all parameters.
    
    Args:
        model (torch.nn.Module): The model whose parameters should be partially frozen.
        freeze_ratio (float): The fraction of weights to freeze (e.g., 0.5 for 50%).
    """
    for name, param in model.named_parameters():
        if param.requires_grad:  # Skip already frozen params
            # Create a mask of the same shape as param with ones
            mask = torch.ones_like(param, dtype=torch.bool)

            # Compute number of elements to freeze
            num_elements = param.numel()
            num_to_freeze = int(num_elements * freeze_ratio)

            # Randomly select indices to freeze
            indices = random.sample(range(num_elements), num_to_freeze)

            # Set selected indices to 0 (frozen)
            flat_mask = mask.view(-1)
            flat_mask[indices] = 0
            mask = flat_mask.view(param.shape)

            # Apply the mask using detach
            param.data = param.data * mask  # Keep values, but zero-out frozen ones
            param.requires_grad = True  # Re-enable grad for the masked tensor

            # Store mask as a buffer (optional, for later reference)
            model.register_buffer(f"{name.replace('.', '_')}_freeze_mask", mask)

def get_delta_cos(model1, model2, model0_sd):
    flat_model0 = flat_dict(model0_sd)
    flat_model1 = flat_dict(model1.state_dict())
    flat_model2 = flat_dict(model2.state_dict())
    
    delta = torch.abs(flat_model1 - flat_model2)
    org_cos = cos((flat_model1 - flat_model0), (flat_model2 - flat_model0))
    return delta, 1-org_cos.item()

def model_eval(model, test_loader):
    acc = eval_op_ensemble([model], test_loader)
    asr = eval_op_ensemble_tr_lf_attack([model], test_loader)
    return list(acc.values())[0], list(asr.values())[0]

def defreeze_model(model):
    """
    Unfreezes all parameters in the model, making them trainable again.
    
    Args:
        model (torch.nn.Module): The model to defreeze.
    """
    for param in model.parameters():
        param.requires_grad = True  # Enable gradient updates

    # Remove any freeze masks stored as buffers (optional)
    buffer_keys = [key for key in model.state_dict().keys() if key.endswith("_freeze_mask")]
    for key in buffer_keys:
        del model._buffers[key]  # Remove stored masks


            
train_op(model1, client_loader, optimizer1, epochs=local_ep, print_train_loss=True)

# model2 normal malicious training
train_op_tr_flip(model2, client_loader, optimizer2, epochs=local_ep, class_num=10, print_train_loss=True)

acc1, asr1 = model_eval(model1, test_loader)
acc2, asr2 = model_eval(model2, test_loader)

# delta, org_cos
delta, org_cos2 = get_delta_cos(model1, model2, model0_sd)
print(f"model1 acc:{acc1}, asr:{asr1}, cos dist:{0}")
print(f"model2 acc:{acc2}, asr:{asr2}, cos dist:{org_cos2}")



results = {}
for k in np.linspace(start=0, stop=1, num=points):
    # reverse back
    defreeze_model(model2)
    # model2.load_state_dict(model0_sd)
    model2.load_state_dict(model1.state_dict())
    
    train_op_tr_flip(model2, client_loader, optimizer2, epochs=local_ep, class_num=10, print_train_loss=True)
    freeze_random_weights(model2, freeze_ratio=k)
    acc_, asr_ = model_eval(model2, test_loader)
    defreeze_model(model2)
    delta, cos_dist = get_delta_cos(model1, model2, model0_sd)
    
    
    results[k] = (acc_, asr_, cos_dist)
    
print("results:\n", results)