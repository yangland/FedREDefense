import torch
import numpy as np
from functools import partial
import models as model_utils
from utils import restore_dict_grad_dict, cos_pairs_and_mean, \
 flat_dict, filter_trainable_state_dict,  restore_dict_grad_flat, \
     cos_dist_w, eval_epoch, cos_dist_w, weighted_avg_budget_cos, \
         get_model_update, parameters_dict_to_vector, craft_tensor
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def untargeted_cos_budget_attack(malicc, mali_clients, server, ben_grad_all, mal_user_grad_ben_mean, 
                                 model_name, num_classes, xp, hp, K, beta, lambda_):
    """Performs an untargeted cosine budget attack by optimizing malicious updates."""
    adhoc_model_fn = partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset=hp['dataset'])
    
    # Synchronize malicious client with the server
    malicc.synchronize_with_server(server)
    model0 = adhoc_model_fn().to(device)
    model0.load_state_dict(malicc.server_state)
    
    # Test accuracy before attack
    acc_results0 = malicc.feedback_on_attack(class_num=10).items()
    
    # Compute cosine distances and log statistics
    all_w = torch.stack([flat_dict(client.W) for client in mali_clients])
    print(f"all_w shape {all_w.shape}")
    
    # Compute the median norm of benign clients
    norm_list = np.array([torch.norm(torch.tensor(grad), p=2).item() for grad in ben_grad_all])
    benign_norm = np.median(norm_list)
    
    print(f"benign norm {benign_norm}")
    
    # Initialize benign mean model
    ben_mean_model = adhoc_model_fn().to(device)
    # Restore benign mean weights and load into model
    benign_mean_w = restore_dict_grad_dict(mal_user_grad_ben_mean, malicc.server_state, malicc.model.state_dict())
    ben_mean_model.load_state_dict(benign_mean_w)
    cos_mean, cos_med, cos_std, cos_to_mean = cos_pairs_and_mean(all_w, benign_mean_w)
    xp.log({"cos_mean": cos_mean, "cos_med": cos_med, "cos_std": cos_std, "mean_cos_to_mean": cos_to_mean})
    
    # Prepare malicious client for attack
    # malicc.sub_loader = malicc.get_sub_dataloader(mult=min(2, malicc.data_multiplier))
    malicc.reset_lr(new_lr=0.005)
    
    # Compute attack budget
    budget = max(1e-5, (1 - cos_mean))
    
    # malicc model load benign mean weights
    malicc.model.load_state_dict(benign_mean_w)
    
    acc_benign_mean = malicc.feedback_on_attack(class_num=10).items()
    
    # Update malicious weights
    train_rev_w_cos(malicc.model, malicc.sub_loader, malicc.optimizer, malicc.scheduler, epochs=K, 
                    model0=model0, model1=ben_mean_model, beta=0.5, budget=budget)

    # Evaluate attack progress
    acc_results1 = malicc.feedback_on_attack(class_num=10).items()
    
    # Compute and normalize malicious gradient update
    mali_grad = get_model_update(malicc.model.state_dict(), malicc.server_state)
    mali_grad_norm = torch.norm(parameters_dict_to_vector(mali_grad), p=2)
    print(f"benign norm {benign_norm}, mali norm {mali_grad_norm}")
    norm_mali_flat = flat_dict(mali_grad) / torch.norm(flat_dict(mali_grad), p=2) * benign_norm 
    
    if torch.isnan(norm_mali_flat).any():
        print("crafted normalized_mali_flat has NA values!")
        
    # Scale with lambda and update model
    mali_w2 = restore_dict_grad_flat(norm_mali_flat * lambda_, malicc.server_state, malicc.model.state_dict())

    model_has_nan = torch.stack([torch.isnan(p).any() for p in mali_w2.values()]).any().item()
    if model_has_nan:
        print("crafted model weight has NA values!")
    cos_d = nn.CosineSimilarity(dim=0, eps=1e-9)
    final_cos = 1 - cos_d(flat_dict(benign_mean_w),flat_dict(mali_w2)).item()
    
    malicc.model.load_state_dict(mali_w2, strict=False)
    
    # Evaluate final attack results
    acc_results2 = malicc.feedback_on_attack(class_num=10).items()
    
    return budget, acc_results0, acc_results1, acc_results2, acc_benign_mean, mali_w2, float(final_cos)


def train_rev_w_cos(model, loader, optimizer, scheduler, epochs, model0, model1, beta, budget):    
    model.train()
    # model.parameters need to use 
    flat_model0 = flat_dict(filter_trainable_state_dict(model0))
    flat_model1 = flat_dict(filter_trainable_state_dict(model1))
    
    losses = []
    running_loss, samples = 0.0, 0
    print(f"data length {len(loader) * loader.batch_size}: batches {len(loader)}, batch_size {loader.batch_size}")
    
    last_mail_w = flat_model0.clone().detach()
    for ep in range(epochs):
        for it, (x, y) in enumerate(loader):
            if it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 1 negative CE loss
            loss_ce = nn.CrossEntropyLoss(reduction="mean")(model(x), y)
            loss_oppo_ce = - loss_ce

            running_loss += loss_oppo_ce.item() * y.shape[0]
            samples += y.shape[0]
            
            # add cos loss 
            w = torch.cat([p.view(-1) for p in model.parameters()]).to(device)
            target = torch.ones(len(w)).to(device)
            loss_cos = nn.CosineEmbeddingLoss()(flat_model0.unsqueeze(0), w.unsqueeze(0), target)
            
            # combindation loss
            loss_obj = (1-beta) * loss_oppo_ce + beta * loss_cos
            # only negative loss
            # loss_obj = loss_oppo_ce 
            
            loss_obj.backward()
            optimizer.step()
            scheduler.step()
            if it % 5 == 0:
                print(f"ep{ep}, loss_ce: {loss_oppo_ce:.2f}, loss_cos: {loss_cos:.6f}, loss_obj: {loss_obj:.2f}, lr: {optimizer.param_groups[0]['lr']}")
        
        # break
        crafted_cos_d = cos_dist_w(flat_model0, w)
        print(f"cos_d: {crafted_cos_d}, budget: {budget}")

        if crafted_cos_d > budget:
            print(f"budget exceeded, finish training early, ep = {ep}")
            break
        
        last_mail_w = w
        
    # craft_g = combine_tensors(B=grad_ben, M=grad_mail, budget=budget)
    craft_w = craft_tensor(B=flat_model0, M1=last_mail_w, M2=w, k=budget)
    
    # restored_crafted = restore_dict_grad_flat(craft_w, model0.state_dict(), model.state_dict())
    restored_crafted = restore_dict_w_flat(craft_w, model.state_dict())
    model.load_state_dict(restored_crafted)
    crafted_cos_d = cos_dist_w(flat_model0, craft_w)
        
    print(f"crafted cos_d: {crafted_cos_d}")        

    return {"loss": running_loss / samples}


def restore_dict_w_flat(param_flat, model_dict):
    restored_w = {}
    start = 0
    for name, param in model_dict.items():
            num_elements = param.numel()
            restored_w[name] = param_flat[start:start + num_elements].view(param.shape)                          
            start += num_elements
    return restored_w


def train_rev_w_cos_grad(model, loader, optimizer, scheduler, epochs, model0, model1, beta, budget):  
    print("loader length", len(loader))
    model.train()
    # model.parameters need to use 
    flat_model0 = flat_dict(filter_trainable_state_dict(model0))
    flat_model1 = flat_dict(filter_trainable_state_dict(model1))
    grad_ben = (flat_model1 - flat_model0).to(device)
    
    losses = []
    running_loss, samples = 0.0, 0
    print(f"data length {len(loader) * loader.batch_size}: batches {len(loader)}, batch_size {loader.batch_size}")
    
    last_grad_mail = grad_ben
    for ep in range(epochs):
        for it, (x, y) in enumerate(loader):
            if it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 1 negative CE loss
            loss_ce = nn.CrossEntropyLoss(reduction="mean")(model(x), y)
            loss_oppo_ce = - loss_ce

            
            # 2 add sigmod on CE loss
            # # Step 1: Apply sigmoid to logits
            # logits = model(x)
            
            # probs = torch.sigmoid(logits)

            # # Step 2: Normalize probabilities so they sum to 1 (mimic softmax)
            # probs = probs / probs.sum(dim=1, keepdim=True)

            # # Step 3: Compute cross-entropy loss manually
            # loss_log = F.nll_loss(torch.log(probs), y)
            
            # loss_oppo_ce = - loss_log

            # 
            running_loss += loss_oppo_ce.item() * y.shape[0]
            samples += y.shape[0]
            
            # add cos loss 
            w = torch.cat([p.view(-1) for p in model.parameters()]).to(device)
            grad_mail = w - flat_model0
            target = torch.ones(len(w)).to(device)
            loss_cos = nn.CosineEmbeddingLoss()(grad_ben.unsqueeze(0), grad_mail.unsqueeze(0), target)
            
            # combindation loss
            loss_obj = (1-beta) * loss_oppo_ce + beta * loss_cos
            # only negative loss
            # loss_obj = loss_oppo_ce 
            
            loss_obj.backward()
            optimizer.step()
            scheduler.step()
            if it % 10 == 0:
                print(f"ep{ep}, loss_ce: {loss_oppo_ce:.0f}, loss_cos: {loss_cos:.4f}, loss_obj: {loss_obj:.0f}, lr: {optimizer.param_groups[0]['lr']}")
        
        # break
        crafted_cos_d = cos_dist_w(grad_ben, grad_mail)
        # print("eval losses", losses)
        print(f"cos_d: {crafted_cos_d}, budget: {budget}")

        if crafted_cos_d > budget:
            print(f"budget exceeded, finish training early, ep = {ep}")
            break
        
        last_grad_mail = grad_mail
    
    #TODO debugging
    # grad_ben_flat = torch.cat([p.view(-1) for p in grad_ben]).to(device)
    # grad_mail_flat = torch.cat([p.view(-1) for p in grad_mail]).to(device)
    # # grad_mail_flat_norm = grad_mail_flat / torch.norm(grad_mail_flat, p=2) * torch.norm(grad_ben_flat, p=2)
    
    # craft_g, best_t, ca_cos_d = weighted_avg_budget_cos(a=grad_ben_flat, b=grad_mail_flat, budget=budget)
    # print("best_t", best_t)
    # print("ca_cos_d", ca_cos_d)
    
    # #TODO debugging
    craft_g = craft_tensor(B=grad_ben, M1=last_grad_mail, M2=grad_mail, k=budget)
    
    restored_crafted = restore_dict_grad_flat(craft_g, model0.state_dict(), model.state_dict())
    model.load_state_dict(restored_crafted)
    crafted_cos_d = cos_dist_w(grad_ben, craft_g)
        
    print(f"crafted cos_d: {crafted_cos_d}")        

    return {"loss": running_loss / samples}