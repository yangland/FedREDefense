import random
import models as model_utils
from utils import *
from client import Device
import hdbscan
from utils import kd_loss, DiffAugment
import sklearn.metrics.pairwise as smp
from torch.utils.data import DataLoader, SubsetRandomSampler
from MADS import MADS
from our_attack import train_rev_w_cos
from copy import deepcopy
import numpy as np
from collections import defaultdict
import data_f

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def cos_sim_nd(tensor1, tensor2):
    # return 1 - (p * q / (p.norm() * q.norm())).sum()
    dot_product = torch.sum(tensor1 * tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    similarity = (dot_product+1e-8) / (norm1 * norm2 + 1e-8)
    return 1-similarity


def cos(a, b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9)
                         * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr


def cosScoreAndClipValue(net1, net2):
    '''net1 -> centre, net2 -> local, net3 -> early model'''
    vector1 = model2vector(net1)
    vector2 = model2vector(net2)

    return cos(vector1, vector2), norm_clip(vector1, vector2)


def norm_clip(nparr1, nparr2):
    '''v -> nparr1, v_clipped -> nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    # import pdb; pdb.set_trace()
    return vnum / (np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9)


def get_model_update(model1, model0, multi=1):
    '''get the update weight'''
    output = OrderedDict()
    for key, var in model1.items():
        output[key] = (model1[key].detach() - model0[key].detach()) * multi
    return output

def get_model_merged(model1, model2):
    '''get the update weight'''
    output = OrderedDict()
    for key, var in model1.items():
        output[key] = model1[key].detach() + model2[key].detach()
    return output


def epoch(mode, dataloader, net, optimizer, criterion, aug=True, args=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().cuda()
        lab = datum[1].cuda()
        if aug and mode == "train":
            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        if mode == 'train':
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(
            ), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))
        else:
            acc = np.sum(np.equal(
                np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


class Server(Device):
    def __init__(self, model_names, loader, num_classes=10, dataset='cifar10', val_loader=None):
        super().__init__(loader)
        # import pdb; pdb.set_trace()
        print(f"dataset server {dataset}")

        self.model_dict = {model_name: partial(model_utils.get_model(model_name)[
                                               0], num_classes=num_classes, dataset=dataset)().to(device) for model_name in model_names}
        self.parameter_dict = {model_name: {key: value for key, value in model.named_parameters(
        )} for model_name, model in self.model_dict.items()}
        
        self.sd_dict = {model_name: {key: value for key, value in model.state_dict(
        ).items()} for model_name, model in self.model_dict.items()}
        
        self.val_loader = val_loader
        self.my_client = {model_name: partial(model_utils.get_model(model_name)[
                                              0], num_classes=num_classes, dataset=dataset)().to(device) for model_name in model_names}

        self.models = list(self.model_dict.values())
        self.model = self.models[0]
        self.fltrust_rootds = None

    def evaluate_ensemble(self, loader=None):
        return eval_op_ensemble(self.models, self.loader if not loader else loader)

    def evaluate_ensemble_with_preds(self):
        return eval_op_ensemble_with_preds(self.models, self.loader)

    def evaluate_backdoor_attack(self, loader=None):
        return eval_op_ensemble_attack(self.models, self.loader if not loader else loader)
    
    def evaluate_lp_attack(self, loader=None, class_num=None):
        return  eval_op_ensemble_lp_attack(self.models, self.loader if not loader else loader, class_num)

    def evaluate_tr_lf_attack(self, loader=None):
        return eval_op_ensemble_tr_lf_attack(self.models, self.loader if not loader else loader)

    def evaluate_attack_with_preds(self, loader=None):
        return eval_op_ensemble_attack_with_preds(self.models, self.loader if not loader else loader)

    def centralized_training(self, syn_data, syn_label, args):
        # import pdb; pdb.set_trace()
        syn_data = torch.cat(syn_data[0:72], dim=0)
        syn_label = torch.cat(syn_label[0:72], dim=0)
        for model_name in self.my_client:
            evaluate_synset(
                0, self.my_client[model_name], 0.1, syn_data, syn_label, self.loader, args)
        exit()

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac))

    def select_clients_masked(self, clients, frac=1.0, mask=None):
        # return [clients[0]]
        available_clients = [item for i, item in enumerate(clients) if mask[i]]
        k = int(len(clients)*frac)
        if k > len(available_clients):
            return available_clients
            raise ValueError(
                "Sample larger than population or not enough masked values.")
        return random.sample(available_clients, k)


    def apply_krum_aggregation(self, clients, mali_ratio, multi_k, unique_client_model_names, mali_ids_all, sd1):
        selected_clients_ids = self.krum(clients, mali_ratio, multi_k=multi_k)
        krum_candidates = [clients[i] for i in list(set(selected_clients_ids))]
        
        reduce_average(target=sd1, sources=[client.sd for client in krum_candidates])
        
        malicious_count = sum(1 for client in selected_clients_ids if client in mali_ids_all)
        mali_select_p = (malicious_count / len(selected_clients_ids))
        
        return mali_select_p, selected_clients_ids
        
    def server_aggregation(self, aggregation_mode, clients, server_lr, mali_ratio, 
                           mali_ids_all, if_two_steps, v_layers_indices, layer_num):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        
        mali_select_p = []
        selected_clients_ids = []
        
        for model_name in unique_client_model_names:
            sd0 = deepcopy(self.sd_dict[model_name])
            sd1 = deepcopy(self.sd_dict[model_name])            
            
            if not if_two_steps:
                if aggregation_mode=="FedAVG":
                    reduce_average(target=sd1, sources=[
                            client.sd for client in clients if client.model_name == model_name])
                elif aggregation_mode=="median":
                    reduce_median(target=sd1, sources=[
                            client.sd for client in clients if client.model_name == model_name])
                elif aggregation_mode=="NormBound":    
                    reduce_normbound(target=sd1, 
                                    server_sd=self.sd_dict[model_name], 
                                    clients=clients, 
                                    mali_ratio=mali_ratio)
                elif aggregation_mode == "krum":
                    mali_select_p, selected_clients_ids = self.apply_krum_aggregation(clients, 
                                                mali_ratio, 
                                                multi_k=False, 
                                                unique_client_model_names=unique_client_model_names, 
                                                mali_ids_all = mali_ids_all,
                                                sd1=sd1)
                elif aggregation_mode == "multi-krum":
                    mali_select_p, selected_clients_ids = self.apply_krum_aggregation(clients, 
                                                mali_ratio, 
                                                multi_k=True, 
                                                unique_client_model_names=unique_client_model_names, 
                                                mali_ids_all = mali_ids_all,
                                                sd1=sd1)
                elif aggregation_mode == "flame":
                    mali_select_p, selected_clients_ids = reduce_flame(target=sd1, 
                                            sources=[client.sd for client in clients if client.model_name == model_name],
                                                malicious_rate=mali_ratio,
                                                wrong_mal=0,
                                                right_ben=0,
                                                noise=0.001,
                                                turn=0)
                elif aggregation_mode == "rfa":
                    reduce_rfa(target=sd1,
                                sources=[client.sd for client in clients if client.model_name == model_name])
            
            else:
                sources1 = [filter_state_dict(client.sd, v_layers_indices) for client in clients if client.model_name == model_name]
                non_v_layers_indices = [item for item in list(range(layer_num)) if item not in v_layers_indices]  
                sources2 = [filter_state_dict(client.sd, non_v_layers_indices) for client in clients if client.model_name == model_name]
                
                if aggregation_mode == "flame":
                    mali_select_p1, selected_clients_ids1 = reduce_flame(target=filter_state_dict(sd1, v_layers_indices), 
                                                                        sources=sources1,
                                                                        malicious_rate=mali_ratio,
                                                                        wrong_mal=0,
                                                                        right_ben=0,
                                                                        noise=0.001,
                                                                        turn=0)     
                    
                    mali_select_p2, selected_clients_ids2 = reduce_flame(target=filter_state_dict(sd1, non_v_layers_indices), 
                                                                        sources=sources2,
                                                                        malicious_rate=mali_ratio,
                                                                        wrong_mal=0,
                                                                        right_ben=0,
                                                                        noise=0.001,
                                                                        turn=0)         
                elif aggregation_mode == "multi-krum":
                    selected_clients_ids1 = reduce_krum(target=filter_state_dict(sd1, v_layers_indices), 
                                                        sources=sources1, 
                                                        mali_ratio=mali_ratio,
                                                        multi_k=True)                  
                    
                    selected_clients_ids2 = reduce_krum(target=filter_state_dict(sd1, non_v_layers_indices), 
                                                        sources=sources2, 
                                                        mali_ratio=mali_ratio,
                                                        multi_k=True)   
                    
                        
                if list(set(selected_clients_ids1) & set(selected_clients_ids2)) != []:
                    join_selected_clients_ids = list(set(selected_clients_ids1) & set(selected_clients_ids2))
                else:
                    join_selected_clients_ids = list(set(selected_clients_ids1) | set(selected_clients_ids2))
                    
                
                print("selected_clients_ids1", selected_clients_ids1)
                print("selected_clients_ids2", selected_clients_ids2)
                print("join_selected_clients_ids", join_selected_clients_ids)
                selected_clients_ids = join_selected_clients_ids
                

                # averaging the join selectioned clients 
                reduce_average(target=sd1, sources=[
                        client.sd for client in clients if client.model_name == model_name and 
                        client.id in join_selected_clients_ids])
                    
            sd_final = get_model_merged(sd0, get_model_update(sd1, sd0, multi = server_lr))
            self.model_dict[model_name].load_state_dict(sd_final)
            
            # Calculate the percentage
            intersection = set(selected_clients_ids) & set(mali_ids_all)
            mali_select_p = (len(intersection) / len(selected_clients_ids))
            return mali_select_p, selected_clients_ids

    def fedavg(self, clients):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        # print("fedavg unique_client_model_names", unique_client_model_names) # ['ConvNet']
        self.weights = torch.Tensor([1. / len(clients)] * len(clients))
        for model_name in unique_client_model_names:
            reduce_average(target=self.sd_dict[model_name], sources=[
                           client.sd for client in clients if client.model_name == model_name])

    def median(self, clients):
        # import pdb; pdb.set_trace()
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_median(target=self.sd_dict[model_name], sources=[
                          client.sd for client in clients if client.model_name == model_name])

    def TrimmedMean(self, clients, mali_ratio):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_trimmed_mean(target=self.sd_dict[model_name], sources=[
                                client.sd for client in clients if client.model_name == model_name], mali_ratio=mali_ratio)

    def krum(self, clients, mali_ratio, multi_k=False):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        
        if not multi_k:
            # run as single Krum
            for model_name in unique_client_model_names:
                krum_candidate_indices = reduce_krum(target=self.sd_dict[model_name], 
                                                     sources=[client.sd for client in clients if client.model_name == model_name], 
                                                     mali_ratio=mali_ratio,
                                                     multi_k=False)
                print("krum_candidate_indice", krum_candidate_indices)
        else:
            for model_name in unique_client_model_names:

                krum_candidate_indices = reduce_krum(target=self.sd_dict[model_name], 
                                                     sources=[client.sd for client in clients if client.model_name == model_name], 
                                                     mali_ratio=mali_ratio,
                                                     multi_k=True)
                
                print("krum_candidate_indices", krum_candidate_indices)

        return krum_candidate_indices

            
    def normbound(self, clients, mali_ratio):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        self.weights = torch.Tensor([1. / len(clients)] * len(clients))
        user_num = len(clients)
        weight = []
        for name in self.parameter_dict[unique_client_model_names[0]]:
            weight.append(torch.flatten(
                self.parameter_dict[unique_client_model_names[0]][name].detach()))
        weight = torch.cat(weight)
        new_model = []
        updates = []
        for client in clients:
            source = client.sd
            new_model_i = []
            for name in client.sd:
                new_model_i.append(torch.flatten(source[name].detach()))
            new_model_i = torch.cat(new_model_i)
            updates_i = new_model_i - weight
            new_model.append(new_model_i)
            updates.append(updates_i)
        new_model = torch.stack(new_model)
        # updates = torch.stack(updates)
        norm_list = [update.norm().unsqueeze(dim=0) for update in updates]
        # import pdb; pdb.set_trace()
        benign_norm_list = []
        for client, norm in zip(clients, norm_list):
            if client.id < (1 - mali_ratio) * user_num:
                benign_norm_list.append(norm)
        if len(benign_norm_list) != 0:
            median_tensor = sum(benign_norm_list)/len(benign_norm_list)
        else:
            median_tensor = sum(norm_list)/len(norm_list)
        # import pdb; pdb.set_trace()
        clipped_models = [
            update * min(1, (median_tensor+1e-8) / (update.norm()+1e-8)) for update in updates]
        clipped_model = torch.mean(torch.stack(clipped_models), dim=0)
        
        for model_name in unique_client_model_names:
            idx = 0
            for name in self.sd_dict[model_name]:
                self.sd_dict[model_name][name].data = self.sd_dict[model_name][name].data + clipped_model[idx:(
                    idx+self.sd_dict[model_name][name].data.numel())].reshape(self.sd_dict[model_name][name].data.shape)
                idx += self.sd_dict[model_name][name].data.numel()

    # adding RLR aggregation rule from FLAME (https://github.com/zhmzm/FLAME)
    def RLR(self, clients, robustLR_threshold):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_RLR(target=self.sd_dict[model_name],
                       sources=[
                           client.sd for client in clients if client.model_name == model_name],
                       robustLR_threshold=robustLR_threshold)

    def flame(self, clients, malicious_rate, wrong_mal, right_ben, noise, turn):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            mali_select_p=reduce_flame(target=self.sd_dict[model_name], sources=[client.sd for client in clients if client.model_name == model_name],
                                        malicious_rate=malicious_rate,
                                        wrong_mal=wrong_mal,
                                        right_ben=right_ben,
                                        noise=noise,
                                        turn=turn)
            return mali_select_p

    def foolsgold(self, clients):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_foolsgold(target=self.sd_dict[model_name],
                             sources=[client.sd for client in clients if client.model_name == model_name])




    def fltrust(self, clients, root_loader, epochs):
        sd_before = copy.deepcopy(self.model.state_dict())
        server_train_stats = train_op(self.model, root_loader, self.optimizer, epochs)
        server_update = get_model_update(self.model.state_dict(), sd_before)
        
        unique_client_model_names = np.unique([client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_fltrust(  target=self.sd_dict[model_name],
                             sources=[client.sd for client in clients if client.model_name == model_name],
                             server_update= server_update)

    def rfa(self, clients):
        unique_client_model_names = np.unique(
            [client.model_name for client in clients])
        for model_name in unique_client_model_names:
            reduce_rfa(target=self.sd_dict[model_name],
                             sources=[client.sd for client in clients if client.model_name == model_name])        



    def pre_assessment(self, model_name, num_classes, optimizer_fn, dataset, args, initial_model_state):
        if dataset == "fmnist":
            anal_dataset = "mnist"
        elif dataset == "cifar10":
            anal_dataset = "SVHM"
        
        print("anal_dataset", anal_dataset)
        # output the sensitivity score 
        train_data, test_data = data_f.get_data(dataset=anal_dataset, path=args.DATA_PATH)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, num_workers=4)
        
        # train the model to 60% acc
        subloader = get_sub_dataloader(size=608, dataset=train_data)
        optimizer = optimizer_fn(self.model.parameters())
        
        eval_result =0
        while eval_result<0.55:
            train_op(model=self.model,
                    loader=subloader,
                    optimizer=optimizer,
                    epochs=1)
            eval_result = self.evaluate_ensemble(loader=test_loader)['test_accuracy']
            print("pre_assessment model0 eval_result", eval_result)
        
        model0_sd = deepcopy(self.model.state_dict())
        train_op(model=self.model,
                loader=subloader,
                optimizer=optimizer,
                epochs=1)
        eval_result = self.evaluate_ensemble(loader=test_loader)['test_accuracy']
        print("pre_assessment model1 eval_result", eval_result)
        model1_sd = deepcopy(self.model.state_dict())
        
        print("model_name", model_name)
        adhoc_model_fn = partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset=dataset)
        model0 = adhoc_model_fn().to(device)
        model1 = adhoc_model_fn().to(device)
        model0.load_state_dict(model0_sd)
        model1.load_state_dict(model1_sd)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
        
        # perform the attack
        loss_dict, crafted_w_cos = train_rev_w_cos(model = self.model, 
                                        loader = subloader, 
                                        optimizer = optimizer, 
                                        scheduler = scheduler, 
                                        epochs = 12, 
                                        model0 = model0, 
                                        model1 = model1, 
                                        beta_ = 0.5, 
                                        budget = 0.001)
        
        print("pre_assessment crafted_w_cos", crafted_w_cos)
        benign_grad = get_model_update(model1_sd, model0_sd)
        mali_grad = get_model_update(self.model.state_dict(), model0_sd)
        benign_norm = torch.norm(flat_dict(benign_grad), p=2)
        mali_norm = torch.norm(flat_dict(mali_grad), p=2)
        norm_mali_flat = flat_dict(mali_grad) / (mali_norm + 1e-9) * benign_norm 
        mali_sd = restore_dict_grad_flat(norm_mali_flat, model0_sd, self.model.state_dict())
        
        # get the sensitivity scores, between mali_sd and model1
        layer_names, num_parameters, l2_distances, cos_dissimilarities, layer_num = compare_model_weights(mali_sd, model1_sd)
        sensitivity_scores = get_sensitivity_scores(layer_names, num_parameters, l2_distances, cos_dissimilarities)
        
        
        # return the server model to the initial model state
        self.models[0].load_state_dict(initial_model_state)

        return sensitivity_scores, layer_names, layer_num
    
    
def get_sensitivity_scores(layer_names, num_parameters, l2_distances, cos_dissimilarities):
    scores = []
    for i in range(len(layer_names)):
        scores.append(cos_dissimilarities[i] / num_parameters[i])
    return scores

def get_sub_dataloader(size, dataset):
    dataset_size = len(dataset)
    assert size <= dataset_size, "size should be less than dataset size"
    
    # Group indices by class labels
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):  # Assuming dataset returns (data, label)
        class_indices[label].append(idx)
    
    # Determine per-class sample size
    num_classes = len(class_indices)
    samples_per_class = size // num_classes
    
    selected_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])

    # Shuffle final selected indices
    np.random.shuffle(selected_indices)
    
    # Use SubsetRandomSampler
    train_sampler = SubsetRandomSampler(selected_indices)
    sub_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    return sub_loader


def compare_model_weights(model1_sd, model2_sd):
    """
    Compare the weights of two PyTorch models layer by layer using bar charts.
    
    Args:
        model1 (torch.nn.Module): The first model to compare.
        model2 (torch.nn.Module): The second model to compare.
    
    Returns:
        None (displays a bar chart of L2 distance and cosine dissimilarity).
    """
    layer_names = []
    l2_distances = []
    cos_dissimilarities = []
    num_parameters = []
    layer_num = 0
    
    for (name1, param1), (name2, param2) in zip(model1_sd.items(), model2_sd.items()):
        if name1 != name2:
            raise ValueError(f"Layer mismatch: {name1} vs {name2}")
        
        layer_names.append(name1)
        layer_num+=1
        param1_flat = param1.view(-1)
        param2_flat = param2.view(-1)
        num_parameters.append(param1.numel()) 
        
        l2_distance = torch.norm(param1_flat - param2_flat, p=2).item()
        l2_distances.append(l2_distance)
        
        cos_sim = torch.nn.functional.cosine_similarity(param1_flat, param2_flat, dim=0).item()
        cos_dissimilarities.append(1 - cos_sim)

    return layer_names, num_parameters, l2_distances, cos_dissimilarities, layer_num


# add a special Server class malicious command center
class MaliCC(Device):
    def __init__(self, model_name, loader, optimizer_fn, num_classes=10, dataset='cifar10',
                 data=None, mali_ids=None, search_algo="MADS", obj=None):
        super().__init__(loader)
        print(f"Malicous command center {dataset}")
        # self.parameter_dict = {model_name : {key : value for key, value in model.named_parameters()} for model_name, model in self.model_dict.items()}
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.sd = {key: value for key, value in self.model.state_dict().items()}
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1)
        self.obj = obj
        self.num_classes = num_classes

        self.mali_ids = mali_ids
        self.data_multiplier = len(mali_ids)
        self.x = None
        self.history = []
        self.search_algo = search_algo
        self.dsm = None
        self.sub_loader = None
        self.data = data
        self.server_state = None

    def reset_lr(self, new_lr):
        self.scheduler._step_count = 0
        self.scheduler.last_epoch = -1  
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr  # Set to your desired value


    def get_sub_dataloader(self, size):
        dataset_size = len(self.data)
        assert size <= dataset_size, "size should be less than dataset size"
        
        # Group indices by class labels
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.data):  # Assuming dataset returns (data, label)
            class_indices[label].append(idx)
        
        # Determine per-class sample size
        num_classes = len(class_indices)
        samples_per_class = size // num_classes
        
        selected_indices = []
        for label, indices in class_indices.items():
            np.random.shuffle(indices)
            selected_indices.extend(indices[:samples_per_class])

        # Shuffle final selected indices
        np.random.shuffle(selected_indices)
        
        # Use SubsetRandomSampler
        train_sampler = SubsetRandomSampler(selected_indices)
        sub_loader = DataLoader(self.data, batch_size=32, sampler=train_sampler)
        return sub_loader

    
    
    def search_initial(self, x0, bounds=[[0, 1], [0, 180], [0, 3]], detlta0=0.1, delta_min=1e-5):
        self.x = x0
        if self.search_algo == "MADS":
            self.dsm = MADS(x0, initial_value=1, bounds=np.array(
                bounds), delta0=detlta0, delta_min=delta_min)
            print("search_initial search_algo")


    def get_server_feedback(self, server_models, loader=None):
        feedback = None
        if self.obj == "targeted_label_flip":
            feedback = eval_op_ensemble_tr_lf_attack(server_models, self.loader if not loader else loader)
        elif self.obj == ["label_flip", "rev_cos"]:
            feedback = eval_op_ensemble(server_models, self.loader if not loader else loader)
        elif self.obj == "Backdoor":
            feedback = eval_op_ensemble_attack(server_models, self.loader if not loader else loader)
        return feedback

    def feedback_on_attack(self, loader=None, class_num=10):
        feedback = None
        if self.obj == "targeted_label_flip":
            feedback = eval_op_ensemble_tr_lf_attack([self.model], self.loader if not loader else loader)
        elif self.obj == "label_flip":
            feedback = eval_op_ensemble_lp_attack([self.model], self.loader if not loader else loader, class_num)
        elif self.obj == "Backdoor":
            feedback = eval_op_ensemble_attack([self.model], self.loader if not loader else loader)
        elif self.obj == "rev_cos":
            # print("rev_cos model check:", self.model.state_dict()["features.0.weight"])
            feedback = eval_op_ensemble([self.model], self.loader if not loader else loader)
        else:
            print("objective unknown")
        return feedback
            
        
    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.server_state = server_state
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def compute_weight_mali_update(self, model0, model1, epochs=1, loader=None, beta=0.5, budget=0.1):
        if self.obj == "label_flip":
            train_stats = train_op_flip(
                self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
        elif self.obj == "targeted_label_flip":
            train_stats = train_op_tr_flip(
                self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
        elif self.obj == "Backdoor":
            train_stats = train_op_backdoor(
                self.model, self.loader if not loader else loader, self.optimizer, epochs)
        elif self.obj == "rev_cos":
            train_stats = train_rev_w_cos(
                self.model, loader, self.optimizer, self.scheduler, epochs, model0, model1, beta, budget)
        else:
            raise Exception("Unknown mali objetive")
        return train_stats