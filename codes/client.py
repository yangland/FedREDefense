import random
from tqdm import tqdm
from functools import partial
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import *
import models as model_utils
from sklearn.linear_model import LogisticRegression
import os
import math
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Device(object):
    def __init__(self, loader):

        self.loader = loader

    def evaluate(self, loader=None):
        return eval_op(self.model, self.loader if not loader else loader)

    def save_model(self, path=None, name=None, verbose=True):
        if name:
            torch.save(self.model.state_dict(), path+name)
            if verbose:
                print("Saved model to", path+name)

    def load_model(self, path=None, name=None, verbose=True):
        if name:
            self.model.load_state_dict(torch.load(path+name))
            if verbose:
                print("Loaded model from", path+name)


class Client(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.benign_grad = None

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None, print_train_loss=False,  hp=None):
        clip_bound, privacy_sigma = None, None
        train_stats = train_op(self.model, self.loader if not loader else loader,
                               self.optimizer, epochs, print_train_loss=print_train_loss)
        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_flip(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.num_classes = num_classes

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        train_stats = train_op_flip(self.model, self.loader if not loader else loader,
                                    self.optimizer, epochs, class_num=self.num_classes)
        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_tr_flip(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.num_classes = num_classes

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        train_stats = train_op_tr_flip(
            self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_MinMax(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum

        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 1
        self.mal_user_grad_mean2 = None
        self.mal_user_grad_std2 = None
        self.all_grads = None
        self.benign_grad = None

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.server_state = server_state
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def compute_weight_update(self, epochs=1, loader=None, dev_type='std', threshold=30):
        all_updates = torch.Tensor(np.array(self.all_grads)).cuda()
        model_re = torch.mean(all_updates, dim=0)
        if dev_type == 'unit_vec':
            # unit vector, dir opp to good dir
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        lamda = torch.Tensor([threshold]).float().cuda()

        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        if len(all_updates) != 1:
            distances = []
            for update in all_updates:
                distance = torch.norm((all_updates - update), dim=1) ** 2
                distances = distance[None, :] if not len(
                    distances) else torch.cat((distances, distance[None, :]), 0)
            max_distance = torch.max(distances)
            del distances

            while torch.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = (model_re - lamda * deviation)
                distance = torch.norm((all_updates - mal_update), dim=1) ** 2
                max_d = torch.max(distance)

                if max_d <= max_distance:
                    # print('successful lamda is ', lamda)
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2
            mal_update = (model_re - lamda_succ * deviation)
        else:
            mal_update = (model_re - model_re)

        idx = 0
        user_grad = OrderedDict()
        for name in self.W:
            user_grad[name] = mal_update[idx:(
                idx+self.W[name].numel())].reshape(self.W[name].shape)
            self.W[name].data = self.server_state[name] + user_grad[name]
            idx += self.W[name].numel()

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        print(self.W['classification_layer.bias'])
        print(self.model.state_dict()['classification_layer.bias'])
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_MinSum(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum

        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 1
        self.mal_user_grad_mean2 = None
        self.mal_user_grad_std2 = None
        self.all_grads = None
        self.benign_grad = None

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.server_state = server_state
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def compute_weight_update(self, epochs=1, loader=None, dev_type='std', threshold=30):
        # import pdb; pdb.set_trace()
        all_updates = torch.Tensor(np.array(self.all_grads)).cuda()
        model_re = torch.mean(all_updates, dim=0)
        if dev_type == 'unit_vec':
            # unit vector, dir opp to good dir
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        if len(all_updates) != 1:
            distances = []
            for update in all_updates:
                distance = torch.norm((all_updates - update), dim=1) ** 2
                distances = distance[None, :] if not len(
                    distances) else torch.cat((distances, distance[None, :]), 0)

            scores = torch.sum(distances, dim=1)
            min_score = torch.min(scores)
            del distances

            while torch.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = (model_re - lamda * deviation)
                distance = torch.norm((all_updates - mal_update), dim=1) ** 2
                score = torch.sum(distance)

                if score <= min_score:
                    # print('successful lamda is ', lamda)
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2
            mal_update = (model_re - lamda_succ * deviation)
        # print(lamda_succ)
        else:
            mal_update = (model_re - model_re)

        idx = 0
        user_grad = OrderedDict()
        for name in self.W:
            user_grad[name] = mal_update[idx:(
                idx+self.W[name].numel())].reshape(self.W[name].shape)
            self.W[name].data = self.server_state[name] + user_grad[name]
            idx += self.W[name].numel()
        # import pdb; pdb.set_trace()
        # return train_stats
        # print(self.W['classification_layer.bias'])

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        # import pdb; pdb.set_trace()
        print(self.W['classification_layer.bias'])
        print(self.model.state_dict()['classification_layer.bias'])
        with torch.no_grad():
            y_ = self.model(x)

        return y_


def compute_lambda(all_updates, model_re, n_attackers):
    # import pdb; pdb.set_trace()
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = nd.norm(all_updates - update, axis=1)
        distances.append(distance)
    distances = nd.stack(*distances)

    distances = nd.sort(distances, axis=1)
    scores = nd.sum(distances[:, :n_benign - 1 - n_attackers], axis=1)
    min_score = nd.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1)
                          * nd.sqrt(nd.array([d]))[0])
    max_wre_dist = nd.max(nd.norm(all_updates - model_re,
                          axis=1)) / (nd.sqrt(nd.array([d]))[0])

    return (term_1 + max_wre_dist)


def score(gradient, v, nbyz):
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance = torch.sort(torch.sum((v - gradient) ** 2, axis=1))[0]
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).item()


def multi_krum(all_updates, n_attackers, multi_k=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates.clone()
    all_indices = torch.arange(len(all_updates))
    candidates = None

    while len(remaining_updates) > 2 * n_attackers + 2:
        scores = torch.tensor(
            [score(gradient, remaining_updates, n_attackers) for gradient in remaining_updates])
        min_idx = int(scores.argmin(axis=0).item())
        candidate_indices.append(min_idx)
        candidates = torch.reshape(remaining_updates[min_idx].clone(), shape=(1, -1)) if not isinstance(
            candidates, torch.Tensor) else torch.cat((candidates, torch.reshape(remaining_updates[min_idx].clone(), shape=(1, -1))), dim=0)
        if min_idx == remaining_updates.shape[0] - 1:
            remaining_updates = remaining_updates[:min_idx, :]
        elif min_idx == 0:
            remaining_updates = remaining_updates[min_idx + 1:, :]
        else:
            remaining_updates = torch.cat(
                (remaining_updates[:min_idx, :], remaining_updates[min_idx + 1:, :]), dim=0)
        if not multi_k:
            break
    aggregate = torch.mean(candidates, axis=0)
    if multi_k == False:
        return aggregate, candidate_indices[0]
    else:
        return aggregate, candidate_indices


class Client_Krum(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum

        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 1
        self.mal_user_grad_mean2 = None
        self.mal_user_grad_std2 = None
        self.all_grads = None
        self.benign_grad = None

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.server_state = server_state
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        all_updates = torch.Tensor(np.array(self.all_grads)).cuda()
        model_re = torch.mean(all_updates, dim=0)
        if len(all_updates) != 1:
            user_grad = OrderedDict()

            deviation = torch.sign(model_re)/torch.norm(torch.sign(model_re))
            lamda = compute_lambda(all_updates, model_re, len(all_updates))
            threshold = 1e-5
            mal_update = []
            while lamda > threshold:
                mal_update = (-lamda * deviation)
                agg_grads, krum_candidate = multi_krum(
                    all_updates, len(all_updates), multi_k=False)
                if krum_candidate < len(all_updates):
                    break
                else:
                    mal_update = []
                lamda *= 0.5

            mal_update = (model_re - lamda * deviation)
        else:
            mal_update = model_re - model_re
        idx = 0
        user_grad = OrderedDict()
        for name in self.W:
            user_grad[name] = mal_update[idx:(
                idx+self.W[name].numel())].reshape(self.W[name].shape)
            self.W[name].data = self.server_state[name] + user_grad[name]
            idx += self.W[name].numel()

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        # import pdb; pdb.set_trace()
        print(self.W['classification_layer.bias'])
        print(self.model.state_dict()['classification_layer.bias'])
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_Fang(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum

        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 1
        self.mal_user_grad_mean2 = None
        self.mal_user_grad_std2 = None
        self.all_grads = None
        self.benign_grad = None

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def synchronize_with_server(self, server):
        server_state = server.model_dict[self.model_name].state_dict()
        self.server_state = server_state
        self.model.load_state_dict(server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        all_updates = torch.Tensor(np.array(self.all_grads)).cuda()
        model_re = torch.mean(all_updates, dim=0)
        if len(all_updates) != 1:
            model_std = torch.std(all_updates, 0)
            user_grad = OrderedDict()

            deviation = torch.sign(model_re)

            max_vector_low = model_re + 3 * model_std
            max_vector_hig = model_re + 4 * model_std
            min_vector_low = model_re - 4 * model_std
            min_vector_hig = model_re - 3 * model_std
            max_range = torch.cat(
                (max_vector_low[:, None], max_vector_hig[:, None]), dim=1)
            min_range = torch.cat(
                (min_vector_low[:, None], min_vector_hig[:, None]), dim=1)

            rand = torch.from_numpy(np.random.uniform(
                0, 1, [len(deviation)])).type(torch.FloatTensor).cuda()
            max_rand = max_range[:, 0].T + rand * \
                (max_range[:, 1] - max_range[:, 0]).T
            min_rand = min_range[:, 0].T + rand * \
                (min_range[:, 1] - min_range[:, 0]).T

            mal_update = (deviation < 0) * max_rand.T + \
                (deviation > 0) * min_rand.T
        else:
            mal_update = model_re - model_re
        # import pdb; pdb.set_trace()
        idx = 0
        user_grad = OrderedDict()
        for name in self.W:
            user_grad[name] = mal_update[idx:(
                idx+self.W[name].numel())].reshape(self.W[name].shape)
            self.W[name].data = self.server_state[name] + user_grad[name]
            idx += self.W[name].numel()
        # print(self.W['classification_layer.bias'])

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        # import pdb; pdb.set_trace()
        print(self.W['classification_layer.bias'])
        print(self.model.state_dict()['classification_layer.bias'])
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_MPAF(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 3

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        # import pdb; pdb.set_trace()
        user_grad = OrderedDict()
        # import pdb; pdb.set_trace()
        for name in self.W:
            user_grad[name] = self.init_model[name] - self.W[name].detach()
            self.W[name].data = self.server_state[name] + \
                self.scale*user_grad[name]

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_Scaling(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 3

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        # print(self.scale)
        train_stats = train_op_backdoor(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)

        user_grad = OrderedDict()
        # import pdb; pdb.set_trace()
        for name in self.W:
            user_grad[name] = self.W[name].detach() - self.server_state[name]
            self.W[name].data = self.server_state[name] + \
                self.scale*user_grad[name]

        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_DBA(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        print(f"dataset client {dataset}")
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 3

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_update(self, epochs=1, loader=None):
        train_stats = train_op_dba(
            self.model, self.loader, self.optimizer, epochs, cid=self.id)

        user_grad = OrderedDict()
        # import pdb; pdb.set_trace()
        for name in self.W:
            user_grad[name] = self.W[name].detach() - self.server_state[name]
            self.W[name].data = self.server_state[name] + \
                self.scale*user_grad[name]

        return train_stats

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_


class Client_AOP(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10', obj="targeted_label_flip"):
        super().__init__(loader)
        self.id = idnum
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)
        self.num_classes = num_classes
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.benign_grad = dict()
        self.mali_grad = dict()
        self.obj = obj
        self.mal_user_grad_mean2 = None
        self.mal_user_grad_std2 = None
        self.gamma = 1
        # gradients from all malicious clients
        self.all_grads = None
        self.min_idx_map = None
        self.ben_cos_mean = None
        self.ben_cos_med = None
        self.ben_cos_std = None
        self.pool_mali_grad = None
        self.mali_mean = None
        self.server_w = None
        self.critical_layer = None

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.server_w = {key: value for key, value in server.model_dict[self.model_name].named_parameters()}
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def compute_weight_mali_update(self, epochs=1, loader=None):
        if self.obj == "label_flip":
            train_stats = train_op_flip(
                self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
        elif self.obj == "targeted_label_flip":
            train_stats = train_op_tr_flip(
                self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
        elif self.obj == "Backdoor":
            train_stats = train_op_backdoor(
                self.model, self.loader if not loader else loader, self.optimizer, epochs)
        else:
            raise Exception("Unknown mali objetive")
        return train_stats

    def compute_weight_update(self, epochs=1, loader=None):
        # closest_benign = torch.tensor(self.all_grads[self.min_idx_map[self.id]])
        # mali grad selection
        # 1. self.mali_grad - individual grad
        # 2. self.pool_mali_grad
        # 3. self.mal_user_grad_mean2 - benign mean
        
        # delta between mali-benign mean and mali-mali mean
        abs_delta = torch.abs(flat_dict(self.mal_user_grad_mean2) - flat_dict(self.mali_mean)).to(device)
        # print("abs_delta", abs_delta, torch.count_nonzero(abs_delta))

        # benign_g = self.benign_grad
        # mali_g = self.mali_grad
        benign_g = self.mal_user_grad_mean2
        mali_g = self.mali_mean
        
        budget_1sigma = (1 - self.ben_cos_med) + self.ben_cos_std * 1
        budget_med = 1 - self.ben_cos_med
        budget_mean = 1 - self.ben_cos_mean
        budget_ben_to_mean = 1 - self.ben_cos_to_mean
        
        # craft_g, k = craft_mali_weighted_avg(ben_g=benign_g, 
        #                                     mali_g=mali_g,
        #                                     budget=budget_ben_to_mean, 
        #                                     measure="cos")

        # print("len craft_g", len(craft_g)) # 4902090
        # print("self.server_w", len(flat_dict(self.server_w))) # 4902090
        
        craft_g, k = craft_critical_layer(ben_g=benign_g, 
                                            mali_g=mali_g,
                                            budget=budget_ben_to_mean, 
                                            measure="cos",
                                            critical_layer=self.critical_layer)
        
        
        restored_crafted = restore_dict_grad_flat(flat_dict(craft_g), self.server_w, self.server_state)
        
        
        self.model.load_state_dict(restored_crafted)
        logger.info(f"client ID: {self.id}, k tops: {k}")
        return k


    def get_cloest_benign(self):
        cos_sim, closest_tensor = closest_tensor_cosine_similarity(flat_dict(self.mali_grad),
                                                                   torch.tensor(self.all_grads).to(device))
        return cos_sim, closest_tensor


class Client_UAM(Device):
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10'):
        super().__init__(loader)
        self.id = idnum
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[
                                0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scale = 3
        self.benign_grad = None
        self.mali_grad = None
        

    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def compute_weight_benign_update(self, epochs=1, loader=None):
        train_stats = train_op(
            self.model, self.loader if not loader else loader, self.optimizer, epochs)
        return train_stats

    def compute_cos_simility_to_mean(self, per_layer=False):
        # Calculate the cos similiaty (in degrees) between the mean of benign_updates of malicous client and
        # each malicoius client
        cos = nn.CosineSimilarity(dim=0, eps=1e-9)
        cos_simility_per_layer = None
        cos_simility_flat = math.degrees(cos(flat_dict(self.mal_user_grad_mean2),
                                             flat_dict(self.benign_grad)).item())

        if per_layer:
            cos_simility_per_layer = dict()
            for name in self.W:
                cos_simility_per_layer[name] = math.degrees(cos(torch.flatten(self.mal_user_grad_mean2[name].detach()),
                                                                torch.flatten(self.benign_grad[name])).item())
        # print("cos_simility_per_layer", cos_simility_per_layer)
        # print("cos_simility_flat", cos_simility_flat)
        return cos_simility_flat, cos_simility_per_layer

        # construct malicious model weight based command from search_algo
    def compute_weight_update(self, epochs=1, loader=None):
        for name in self.W:
            self.W[name].data = self.server_state[name] + self.W[name].data

    def predict_logit(self, x):
        """Softmax prediction on input"""
        self.model.train()

        with torch.no_grad():
            y_ = self.model(x)

        return y_

    def predict_logit_eval(self, x):
        """Softmax prediction on input"""
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)

        return y_
