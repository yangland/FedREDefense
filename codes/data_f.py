import torch, torchvision
import numpy as np
import os, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized
import torch
import json
import numpy as np
import os
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()
    
    # print("labels", labels) # all 50,000 targets
    
    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    
    # print("label_distribution1", label_distribution)

    if double_stochstic:
        label_distribution = make_double_stochstic(label_distribution)
      
    # print("label_distribution2", label_distribution) # all 

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    
    # old distribution code
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    
    # new distribution code
    # for c, fracs in zip(class_idcs, label_distribution):
    #     fracs = fracs / fracs.sum()  # Normalize to ensure sum is exactly 1
    #     split_points = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
    #     split_points = np.clip(split_points, 0, len(c))  # Ensure valid indices
    #     class_splits = np.split(c, split_points)

    #     for i, idcs in enumerate(class_splits):
    #         client_idcs[i].extend(idcs)  # Use extend instead of +=

    # client_idcs = [np.array(idcs) for idcs in client_idcs]  # Convert to NumPy arrays
    

    print_split(client_idcs, labels)
  
    return client_idcs

def get_loan(path):
    loan_directory = '../data/loan'
    train_data = torchvision.datasets.ImageFolder(loan_directory + '/train', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    test_data = torchvision.datasets.ImageFolder(loan_directory + '/test', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    return train_data, test_data


def get_cinic10(path):
    cinic_directory = 'data/cinic10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    train_data = torchvision.datasets.ImageFolder(cinic_directory + '/train', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    test_data = torchvision.datasets.ImageFolder(cinic_directory + '/test', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    return train_data, test_data

def get_cifar10(path):
  transforms = torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=False, download=True, transform=transforms)

  return train_data, test_data

def get_fmnist(path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = torchvision.datasets.FashionMNIST(root=path+"FMNIST", train=True, download=True, transform=transforms)
    test_data = torchvision.datasets.FashionMNIST(root=path+"FMNIST", train=False, download=True, transform=transforms)

    return train_data, test_data

def get_mnist(path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(root=path+"MNIST", train=True, download=True, transform=transforms)
    test_data = torchvision.datasets.MNIST(root=path+"MNIST", train=False, download=True, transform=transforms)

    return train_data, test_data

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    """Parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

def get_data(dataset, path):
  return {"fmnist": get_fmnist, 
          "cifar10" : get_cifar10,
          "cinic10" : get_cinic10, 
          "mnist": get_mnist}[dataset](path)


def get_loaders(train_data, test_data, n_clients=10, alpha=0, batch_size=128, n_data=None, num_workers=0, seed=0):
    #   import pdb; pdb.set_trace()
    subset_idcs = split_dirichlet(train_data.targets, n_clients, n_data, alpha, seed=seed)
    client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]
    # print("subset_idcs0", subset_idcs[0])
    # print("subset_idcs7", subset_idcs[7])

    client_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers) for subset in client_data]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, num_workers=num_workers)
    
    label_counts = []
    for subset in client_data:
        labels = [train_data.targets[i] for i in subset.indices]
        label_counts.append(Counter(sorted(Counter(labels).items())))
    
    # print("label_counts", label_counts)
    
    return client_loaders, test_loader, client_data

def get_loaders_classes(train_data, test_data, n_clients=10, alpha=0, batch_size=128, n_data=None, num_workers=0, seed=0, classes =  [0,2,4], total_num = 1500, indices=None):
    print(f"number of clients {n_clients}")
    if indices is None:
        num_per_class= int(total_num/len(classes))
        n_clients = len(classes)
        classwise_indices = [[i for i in range(len(train_data)) if train_data.targets[i] == j] for j in classes]
        for i, class_ind in enumerate(classwise_indices):
            for j in class_ind:
                train_data.targets[j] = i
        classwise_indices_sampled = [np.random.choice(indices, num_per_class, replace=False) for indices in classwise_indices]
    else:
        classwise_indices_sampled = indices
        for i, class_ind in enumerate(classwise_indices_sampled):
            for j in class_ind:
                train_data.targets[j] = i
    client_data = [torch.utils.data.Subset(train_data, classwise_indices_sampled[i]) for i in range(n_clients)]
    # client_data = [torch.utils.data.Subset(train_data, np.concatenate(classwise_indices_sampled)) for i in range(n_clients)]
    classwise_indices_test = [i for i in range(len(test_data)) if test_data.targets[i] in classes]
    for i in classwise_indices_test:
        test_data.targets[i] = classes.index(test_data.targets[i])
    test_data = torch.utils.data.Subset(test_data, classwise_indices_test)
    client_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers) for subset in client_data]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, num_workers=num_workers)
    print(f"number of data per class: {[len(x) for x in classwise_indices_sampled]}:")
    print("train class sampled indices:")
    print(classwise_indices_sampled)
    print("test class sampled indices:")
    print(classwise_indices_test)
    return client_loaders, test_loader, classwise_indices_sampled



from torch.utils.data import Dataset
class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        # import pdb; pdb.set_trace()
        labels_hold[self.indices] = torch.LongTensor(labels )
        self.labels = labels_hold
        self.targets = torch.LongTensor(labels )
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)


# def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
#     '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

#     np.random.seed(seed)

#     if isinstance(labels, torch.Tensor):
#       labels = labels.numpy()
    
#     n_classes = np.max(labels)+1
#     # import pdb; pdb.set_trace()
#     label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

#     if double_stochstic:
#       label_distribution = make_double_stochstic(label_distribution)

#     class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
#            for y in range(n_classes)]

#     client_idcs = [[] for _ in range(n_clients)]
#     for c, fracs in zip(class_idcs, label_distribution):
#         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
#             client_idcs[i] += [idcs]

#     client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

#     print_split(client_idcs, labels)
  
#     return client_idcs

def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x



def print_split(idcs, labels):
  n_labels = np.max(labels) + 1 
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()




class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices, return_index):
        self.dataset = dataset
        self.indices = indices
        self.return_index = return_index

    def __getitem__(self, idx):
        if self.return_index:
          return self.dataset[self.indices[idx]], idx
        else:
          return self.dataset[self.indices[idx]]#, idx

    def __len__(self):
        return len(self.indices)



