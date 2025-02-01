import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
import random
import logging
import json
from hashlib import md5
import copy
import torch.nn.functional as F
import os
import sys
from collections import defaultdict
from tensorboardX import SummaryWriter
import pickle
import math
import logging
from networkx.algorithms.community import louvain_communities
import networkx as nx

import warnings

warnings.filterwarnings("ignore")

# Directory where the json file of arguments will be present
directory = './Parse_Files'
from options import args_parser

# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model
# from utility.options import args_parser
from models.Update import test_client, finetune_client, finetune_client1, train_client1, train_client2, \
    train_client0, test_client1
from models.Fed import FedAvg, FedAvg1, FedAvg2, FedAvg3, FedAvg4, FedAvg0
from models.test import test_img



def calculate_euclidean_distance(model1, model2):
    sum = 0.0
    for parameter_tem, parameter_avg in (zip(model1.parameters(), model2.parameters())):
        sum += EuclideanDistances(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1))
    return sum



def calculate_euclidean_model(args, model1, model2):
    """ Returns the global model based euclidean distance.
        """
    dis = []

    euc_distance = calculate_euclidean_distance(model1, model2)
    if math.isnan(euc_distance):
        dis = torch.tensor([1e-8])
    else:
        dis = euc_distance * 10
    dis = torch.tensor(dis)
    dis = F.softmax(dis)

    return dis



def cos_similiar(args, model1, model2):
    sum = 0.0

    for parameter_tem, parameter_avg in list((zip(model1.parameters(), model2.parameters()))):
        # print(parameter_tem)[args.base_layers:]:
        sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
    return sum


def cos_similiar1(args, model1, model2):
    sum = 0.0
    for parameter_tem, parameter_avg in list((zip(model1.parameters(), model2.parameters())))[
                                        0:args.base_layers]:
        sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
    return sum


def cos_similiar2(args, model1, model2):
    sum = 0.0
    for parameter_tem, parameter_avg in list((zip(model1.parameters(), model2.parameters())))[
                                        args.base_layers:]:
        sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
    return sum




def cos_similiar_model(args, model1, model2):
    """ Returns the global model based euclidean distance.
        """
    dis = []

    euc_distance2 = cos_similiar(model1, model2)

    dis = torch.tensor(dis)
    dis = F.softmax(dis)

    return dis



