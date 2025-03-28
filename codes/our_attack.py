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

def check_state_dict(state_dict):
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in {key}")
            if torch.isinf(value).any():
                raise ValueError(f"Inf detected in {key}")
    print("State dictionary is valid.")


def untargeted_cos_budget_attack(
    malicc, mali_clients, server, ben_grad_all, mal_user_grad_ben_mean,
    model_name, num_classes, xp, hp, K, beta_, lambda_, adv_lr, percentile,
    if_PGD=True, lambda_searcher=None, search_lambda=False
):
    """Performs an untargeted cosine budget attack by optimizing malicious updates."""

    adhoc_model_fn = partial(
        model_utils.get_model(model_name)[0], num_classes=num_classes, dataset=hp['dataset']
    )

    # Synchronize malicious client with the server
    malicc.synchronize_with_server(server)
    model0 = adhoc_model_fn().to(device)
    model0.load_state_dict(malicc.server_state)

    # Test accuracy before attack
    server_vali_acc = malicc.feedback_on_attack(class_num=10).items()

    # Compute cosine distances and log statistics
    all_weights = torch.stack([
        flat_dict(filter_trainable_state_dict(client.model)) for client in mali_clients
    ])
    print(f"All state_dict shape: {all_weights.shape}")

    # Compute the median norm of benign clients
    norm_list = np.array([
        torch.norm(torch.tensor(grad), p=2).item() for grad in ben_grad_all
    ])
    benign_grad_norm = np.median(norm_list)

    # Initialize and restore benign mean model
    ben_mean_model = adhoc_model_fn().to(device)
    benign_mean_sd = restore_dict_grad_dict(
        mal_user_grad_ben_mean, malicc.server_state, malicc.model.state_dict()
    )
    ben_mean_model.load_state_dict(benign_mean_sd)
    benign_mean_w = filter_trainable_state_dict(ben_mean_model)

    server_to_benign_cos = cos_dist_w(
        flat_dict(benign_mean_sd), flat_dict(malicc.server_state)
    ).detach().item()

    cos_metrics = cos_pairs_and_mean(
        all_weights, benign_mean_w, percentile=percentile
    )
    cos_mean, cos_med, cos_std, cos_percentile, cos_to_mean = cos_metrics

    xp.log({
        "cos_mean": cos_mean,
        "cos_med": cos_med,
        "cos_percentile": cos_percentile,
        "cos_std": cos_std,
        "server_to_benign_cos": server_to_benign_cos,
    })

    # Prepare malicious client for attack
    malicc.reset_lr(new_lr=adv_lr if if_PGD else 0.01)
    budget = max(1e-8, cos_percentile)
    malicc.model.load_state_dict(malicc.server_state)
    acc_benign_mean = malicc.feedback_on_attack(class_num=10).items()
    mali_benign_grads_cos = None
    
    # Train malicious model
    train_fn = train_rev_w_cos if if_PGD else train_rev_w_cos_no_budget
    loss_dict, trained_w_cos = train_fn(
        model=malicc.model,
        loader=malicc.sub_loader,
        optimizer=malicc.optimizer,
        scheduler=malicc.scheduler,
        epochs=K if if_PGD else int(K*0.5),
        model0=model0,
        model1=ben_mean_model,
        beta_=beta_,
        budget=budget,
    )

    # Evaluate attack progress
    mali_trained_vali_acc = malicc.feedback_on_attack(class_num=10).items()

    # Check for NaN or Inf in model parameters
    for param in malicc.model.parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {param.shape}")
        if torch.isinf(param).any():
            print(f"Inf detected in {param.shape}")

    if if_PGD:
        mali_grad = get_model_update(malicc.sd, malicc.server_state)
        mali_grad_norm = torch.norm(flat_dict(mali_grad), p=2)
        xp.log({"benign_grad_norm": float(benign_grad_norm), "mali_grad_norm": float(mali_grad_norm)})
        mali_div_benign_norm_rate = float(mali_grad_norm/(benign_grad_norm + 1e-9))
        xp.log({"mali_div_benign_norm_rate": mali_div_benign_norm_rate})

        mali_grad_norm_flat = flat_dict(mali_grad) / (mali_grad_norm + 1e-9) * benign_grad_norm
        if torch.isnan(mali_grad_norm_flat).any():
            print("Crafted normalized_mali_flat has NA values!")

        # check the normized acc
        normalized_sd = restore_dict_grad_flat(
            mali_grad_norm_flat, malicc.server_state, malicc.model.state_dict()
        )
        malicc.model.load_state_dict(normalized_sd, strict=False)
        mali_normalized_vali_acc = malicc.feedback_on_attack(class_num=10).items()

        normalized_cos = cos_dist_w(
            flat_dict(ben_mean_model.state_dict()), flat_dict(normalized_sd), eps=1e-9
        )

        if search_lambda:
            if not hasattr(untargeted_cos_budget_attack, '_initialized'):
                untargeted_cos_budget_attack._initialized = True
                print("First call detected - initializing lambda searcher")
            lambda_ = lambda_searcher.choose_lambda()
            lambda_searcher.update(next(iter(server_vali_acc))[1])
        else:
            mali_grad_norm_flat *= lambda_

        xp.log({"lambda_": float(lambda_)})
        if mali_div_benign_norm_rate < lambda_:
            print("Scaling factor lambda is larger then mali_div_benign_norm_rate, causing d_cos overbudget")
        
        benign_grad = get_model_update(benign_mean_sd, malicc.server_state)
        mali_benign_grads_cos = cos_dist_w(mali_grad_norm_flat, flat_dict(benign_grad)).detach().item()
                
        mali_sd = restore_dict_grad_flat(
            mali_grad_norm_flat, malicc.server_state, malicc.model.state_dict()
        )
    else:
        mali_sd = malicc.sd

    # Compute final cosine similarity
    scaled_cos = cos_dist_w(
        flat_dict(ben_mean_model.state_dict()), flat_dict(mali_sd), eps=1e-9
    )

    # Validate before loading
    try:
        check_state_dict(mali_sd)
    except ValueError as e:
        print(f"Warning mali_sd: {e}")

    malicc.model.load_state_dict(mali_sd, strict=False)
    mali_scaled_vali_acc = malicc.feedback_on_attack(class_num=10).items()

    return budget, server_vali_acc, mali_trained_vali_acc, mali_scaled_vali_acc, mali_normalized_vali_acc, \
        acc_benign_mean, mali_sd, float(scaled_cos), float(trained_w_cos), float(mali_benign_grads_cos), float(normalized_cos)



def stable_log_cosh_cross_entropy_loss(preds, targets):
    """Computes a stable log-cosh version of cross-entropy loss."""
    ce_loss = F.cross_entropy(preds, targets, reduction='none')  # Standard CE loss
    stable_loss = torch.abs(ce_loss) + torch.log1p(torch.exp(-2 * torch.abs(ce_loss))) - torch.log(torch.tensor(2.0))
    return torch.mean(stable_loss)

def safe_cross_entropy(logits, labels): 
    """Compute cross-entropy safely, avoiding NaN issues."""
    
    # Check for NaNs or Infs in logits
    if torch.isnan(logits).any() :
        print("Warning: NaN detected in logits! Clamping values...")
        logits = torch.clamp(logits, min=-1e6, max=1e6)

    if torch.isinf(logits).any():
        print("Inf detected in logits! Clamping values...")
        logits = torch.clamp(logits, min=-1e6, max=1e6)

    # Check for NaNs or invalid values in labels
    if torch.isnan(labels).any() or torch.isinf(labels).any():
        raise ValueError("Error: NaN or Inf detected in labels!")

    # Ensure labels are long and within the expected range
    labels = labels.long()
    num_classes = logits.shape[1]
    
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"Error: Labels out of range! Expected between 0 and {num_classes-1}, but got {labels.min()} to {labels.max()}.")

    # Stability shift
    logits = logits - logits.max(dim=1, keepdim=True)[0]

    # Compute the stable log-cosh cross-entropy loss
    loss = stable_log_cosh_cross_entropy_loss(logits, labels)

    # Check for NaNs in final loss
    if not torch.isfinite(loss):
        print("Warning: Non-finite loss detected! Returning fallback value.")
        return torch.tensor(0.0, requires_grad=True)

    return loss




def train_rev_w_cos(model, loader, optimizer, scheduler, epochs, model0, model1, beta_, budget):    
    model.train()
    # model.parameters need to use, no state_dict, the trainable parameters
    flat_w0 = flat_dict(filter_trainable_state_dict(model0))
    flat_w1 = flat_dict(filter_trainable_state_dict(model1))
    
    losses = []
    running_loss, samples = 0.0, 0
    print(f"data length {len(loader) * loader.batch_size}: batches {len(loader)}, batch_size {loader.batch_size}")
    
    # initial as the server model
    latest_w = flat_w0.clone().detach()
    
    # Initialize GradScaler only once
    scaler = torch.amp.GradScaler()
    
    for ep in range(epochs):
        running_loss = 0  # Reset running loss each epoch
        samples = 0  # Reset sample count each epoch

        for it, (x, y) in enumerate(loader):
            if it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device):
                # Compute losses
                loss_ce = safe_cross_entropy(model(x), y)
                loss_oppo_ce = -loss_ce
                running_loss += loss_oppo_ce.item() * y.shape[0]
                samples += y.shape[0]

                # Add cosine loss
                w = torch.cat([p.view(-1) for p in model.parameters()]).to(device)
                target = torch.ones(len(w)).to(device)
                loss_cos = nn.CosineEmbeddingLoss()(flat_w1.unsqueeze(0), w.unsqueeze(0), target)

                # Combine losses
                loss_obj = (1 - beta_) * loss_oppo_ce + beta_ * loss_cos

            # Scale loss and backpropagate
            scaler.scale(loss_obj).backward()

            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Step optimizer and scheduler
            scaler.step(optimizer)
            scaler.update()  # This is important for adjusting the scaling factor
            scheduler.step()
            
            if it % 5 == 0:
                print(f"ep{ep}, loss_ce: {loss_oppo_ce:.2f}, loss_cos: {loss_cos:.6f}, loss_obj: {loss_obj:.2f}, lr: {optimizer.param_groups[0]['lr']}")
        
        crafted_cos_d = 1 - F.cosine_similarity(flat_w1, w, dim=0, eps=1e-12)
        print(f"cos_d: {crafted_cos_d}, budget: {budget}")

        if crafted_cos_d > budget:
            print(f"budget exceeded, finish training early, ep = {ep}")
            break
        
        latest_w = w
        
    # craft between last_mail_w and the current iteration w
    craft_w = craft_tensor(B=flat_w1, M1=latest_w, M2=w, k=budget)
    restored_crafted = restore_dict_w_flat(craft_w, model1)
    model.load_state_dict(restored_crafted)
    
    crafted_cos_d = 1 - F.cosine_similarity(flat_w1, craft_w, dim=0, eps=1e-12)     

    return {"loss": running_loss / samples}, crafted_cos_d


def train_rev_w_cos_no_budget(model, loader, optimizer, scheduler, epochs, model0, model1, beta_, budget):    
    # for ablation study, no need to use budget and beta
    budget = 1
    
    model.train()
    # model.parameters need to use, no state_dict, the trainable parameters
    flat_w0 = flat_dict(filter_trainable_state_dict(model0))
    flat_w1 = flat_dict(filter_trainable_state_dict(model1))
    
    losses = []
    running_loss, samples = 0.0, 0
    print(f"data length {len(loader) * loader.batch_size}: batches {len(loader)}, batch_size {loader.batch_size}")
    
    # initial as the server model
    latest_w = flat_w0.clone().detach()
    
    # Initialize GradScaler only once
    scaler = torch.amp.GradScaler()
    
    for ep in range(epochs):
        running_loss = 0  # Reset running loss each epoch
        samples = 0  # Reset sample count each epoch

        for it, (x, y) in enumerate(loader):
            if it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device):
                # Compute losses
                loss_ce = safe_cross_entropy(model(x), y)
                loss_oppo_ce = -loss_ce
                running_loss += loss_oppo_ce.item() * y.shape[0]
                samples += y.shape[0]
                w = torch.cat([p.view(-1) for p in model.parameters()]).to(device)
                
                loss_obj =  loss_oppo_ce 

            # Scale loss and backpropagate
            scaler.scale(loss_obj).backward()

            # Step optimizer and scheduler
            scaler.step(optimizer)
            scaler.update()  # This is important for adjusting the scaling factor
            scheduler.step()
            
            if it % 5 == 0:
                print(f"ep{ep}, loss_ce: {loss_oppo_ce:.2f}, lr: {optimizer.param_groups[0]['lr']}")
        
        crafted_cos_d = 1 - F.cosine_similarity(flat_w1, w, dim=0, eps=1e-12)
        print(f"cos_d: {crafted_cos_d}, budget: {budget}")

        if crafted_cos_d > budget:
            print(f"budget exceeded, finish training early, ep = {ep}")
            break
        
        # latest_w = w
        
    # craft between last_mail_w and the current iteration w
    # craft_w = craft_tensor(B=flat_w1, M1=latest_w, M2=w, k=budget)
    # restored_crafted = restore_dict_w_flat(craft_w, model1)
    # model.load_state_dict(restored_crafted)
    
    crafted_cos_d = 1 - F.cosine_similarity(flat_w1, w, dim=0, eps=1e-12)     

    return {"loss": running_loss / samples}, crafted_cos_d


def restore_dict_w_flat(param_flat, model):
    restored_w = {}
    start = 0
    param_names = {name for name, _ in model.named_parameters()}
    # print("param_names", param_names)
    for name, param in model.state_dict().items():
        if name in param_names:
            num_elements = param.numel()
            restored_w[name] = param_flat[start:start + num_elements].view(param.shape)                          
            start += num_elements
        else:
            restored_w[name] = param
    return restored_w

