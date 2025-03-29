import torch
import numpy as np
from functools import partial
import models as model_utils
from utils import restore_dict_grad_dict, cos_pairs_and_mean, \
 flat_dict, filter_trainable_state_dict,  restore_dict_grad_flat, \
     cos_dist_w, eval_epoch, cos_dist_w, weighted_avg_budget_cos, \
         get_model_update, parameters_dict_to_vector, craft_tensor, state_dict_to_w
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from typing import Tuple
import numpy as np
import torch
from typing import Tuple

class FLTrustAttack:
    def __init__(self):
        self.previous_centroids = None
        self.last_attack = None

    def kmeans(self, data: torch.Tensor, k: int, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize centroids randomly
        indices = torch.randperm(data.size(0))[:k]
        centroids = data[indices]
        
        for _ in range(num_iters):
            # Compute distances and assign labels
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
            
            if torch.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        
        return labels, centroids

    def untargeted_fltrust_attack(self, ben_grad_all: torch.Tensor) -> torch.Tensor:
        labels, centroids = self.kmeans(ben_grad_all, k=2, num_iters=100)
        
        if self.previous_centroids is None:
            # Randomly select one centroid for the initial attack
            attack_index = torch.randint(0, len(centroids), (1,)).item()
        else:
            # Find the centroid that is farthest from the last attack
            distances = torch.tensor([torch.norm(self.last_attack - centroid).item() for centroid in centroids])
            attack_index = torch.argmax(distances).item()
        
        # Store the current attack as last_attack for the next iteration
        current_attack = centroids[attack_index]
        self.last_attack = current_attack
        
        # Update the previous centroids
        self.previous_centroids = centroids
        
        # Return the current attack
        return current_attack
