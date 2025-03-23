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

def pre_assessment(model_name, optimizer_fn, num_classes, dataset):
    if dataset == "fmnist":
        anal_dataset = "mnist"
    