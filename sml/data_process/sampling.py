import numpy as np
from torchvision import datasets, transforms
import random
from sklearn.cluster import KMeans
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import louvain_communities
import csv
import pandas as pd
import os


# two functions for each type of dataset - one to divide data in iid manner and one in non-iid manner

def mnist_iid(args, dataset, num_users):
    """
    Sample I.I.D. individual data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(args, dataset, num_users):
    """
    Sample non-I.I.D individual data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(args, dataset, num_users):
    """
    Sample I.I.D. individual data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(args, dataset, num_users):
    num_shards, num_imgs = 500, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    train_dict_cilent = {i: np.array([]) for i in range(num_users)}
    test_dict_cilent = {i: np.array([]) for i in range(num_users)}
    dict_users_per = [[0] * num_users for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    dict_user_class = defaultdict(set)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels1 = idxs_labels[1, :]
    # divide and assign
    idx_shard1 = idx_shard[0:180]
    idx_shard2 = idx_shard[180:360]
    idx_shard3 = idx_shard[360:500]

    for i in range(num_users):

        rand_set = set(np.random.choice(idx_shard, 15, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        j = 0
        for rand in rand_set:
            temp_dict = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            train_dict_cilent[i] = np.concatenate(
                (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
            test_dict_cilent[i] = np.concatenate(
                (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)

            for it in labels1[rand * num_imgs: (rand + 1) * num_imgs]:
                # print(it)
                dict_user_class[i].add(it)

    dictuserdictuser = dict(dict_users)
    dictusertrain = dict(train_dict_cilent)
    dictusertest = dict(test_dict_cilent)
    dictuserclass = dict(dict_user_class)
    df = pd.DataFrame.from_dict(dictuserdictuser, orient='index')
    df = df.transpose()
    df.to_csv('/root/data/dictuser1001.csv', index=False)
    df = pd.DataFrame.from_dict(dictusertrain, orient='index')
    df = df.transpose()
    df.to_csv('/root/data/dictuser100train1.csv', index=False)
    df = pd.DataFrame.from_dict(dictusertest, orient='index')
    df = df.transpose()
    df.to_csv('/root/data/dictuser100test1.csv', index=False)
    df = pd.DataFrame.from_dict(dictuserclass, orient='index')
    df = df.transpose()
    df.to_csv('/root/data/dictuserclass1001.csv', index=False)
    return dict_users, train_dict_cilent, test_dict_cilent, dict_user_class


def office31_noniid(args, dataset, num_users):
    num_shards, num_imgs = 50, 10
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    train_dict_cilent = {i: np.array([]) for i in range(num_users)}
    test_dict_cilent = {i: np.array([]) for i in range(num_users)}
    dict_users_per = [[0] * num_users for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    dict_user_class = defaultdict(set)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels1 = idxs_labels[1, :]
    # divide and assign

    for i in range(num_users):

        rand_set = set(np.random.choice(idx_shard, 5, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        j = 0
        for rand in rand_set:
            temp_dict = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            train_dict_cilent[i] = np.concatenate(
                (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
            test_dict_cilent[i] = np.concatenate(
                (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)

            for it in labels1[rand * num_imgs: (rand + 1) * num_imgs]:
                # print(it)
                dict_user_class[i].add(it)

    return dict_users, train_dict_cilent, test_dict_cilent, dict_user_class


def officehome_noniid(args, dataset, num_users):
    num_shards, num_imgs = 100, 23
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    train_dict_cilent = {i: np.array([]) for i in range(num_users)}
    test_dict_cilent = {i: np.array([]) for i in range(num_users)}
    dict_users_per = [[0] * num_users for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    dict_user_class = defaultdict(set)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels1 = idxs_labels[1, :]
    # divide and assign

    for i in range(num_users):

        rand_set = set(np.random.choice(idx_shard, 10, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        j = 0
        for rand in rand_set:
            temp_dict = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            train_dict_cilent[i] = np.concatenate(
                (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
            test_dict_cilent[i] = np.concatenate(
                (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)

            for it in labels1[rand * num_imgs: (rand + 1) * num_imgs]:
                # print(it)
                dict_user_class[i].add(it)

    return dict_users, train_dict_cilent, test_dict_cilent, dict_user_class


def ina_noniid(args, daset, num_users):
    num_shards, num_imgs = 1500, 100
    idx_shard = [i for i in range(num_shards)]
    idx = {i: [] for i in range(3)}
    dict_users = {i: np.array([]) for i in range(num_users)}
    train_dict_cilent = {i: np.array([]) for i in range(num_users)}
    test_dict_cilent = {i: np.array([]) for i in range(num_users)}
    dict_user_class = defaultdict(set)

    datalist = {i: np.array([]) for i in range(13)}

    addatalist = np.genfromtxt('/data/output.csv', dtype='int', delimiter=',')

    for i in range(len(addatalist[0])):
        ddd = []
        for j in range(1, len(addatalist)):
            if (addatalist[j][i] == -1):
                break
            else:
                ddd.append(addatalist[j][i])
        datalist[i] = ddd

    pla = datalist[0]  # 158407 Plants
    fun = datalist[5]  # 5826 Fungi
    chro = datalist[11]  # 398 Algae
    pro = datalist[12]  # 308 Protozoa
    # 164939
    
    ins = datalist[1]  # 100479 Insects
    amp = datalist[6]  # 15318 Amphibians
    rep = datalist[3]  # 35201 Reptiles
    mam = datalist[4]  # 29333 Mammals
    ara = datalist[9]  # 4873 Arachnids (spiders)
    mol = datalist[7]  # 7536 Mollusks
    # 192740
    
    ani = datalist[8]  # 5228 Animals not belonging to other categories
    act = datalist[10]  # 1982 Fish
    ave = datalist[2]  # 214295 Birds
    # 221505


    idx[0] = pla + fun + chro + pro
    idx[1] = ins + amp + rep + mam + ara + mol
    idx[2] = ani + act + ave

    idx1_shard11, idx1_shard22, idx1_shard33 = [idx_shard.copy() for _ in range(3)]

    for i in range(num_users):
        # print(i)
        if (i >= 0 and i < 4):

            rand_set1 = set(np.random.choice(idx1_shard11, 50, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 4 and i < 6):

            rand_set1 = set(np.random.choice(idx1_shard11, 45, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard22, 5, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 6 and i < 8):

            rand_set1 = set(np.random.choice(idx1_shard11, 45, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard33, 5, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 8 and i < 10):

            rand_set1 = set(np.random.choice(idx1_shard11, 40, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard22, 5, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set2)
            rand_set3 = set(np.random.choice(idx1_shard33, 5, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set3)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set3:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 10 and i < 14):

            rand_set1 = set(np.random.choice(idx1_shard22, 50, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set1)

            for rand in rand_set1:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 14 and i < 16):

            rand_set1 = set(np.random.choice(idx1_shard11, 5, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard22, 45, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 16 and i < 18):

            rand_set1 = set(np.random.choice(idx1_shard22, 45, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard33, 5, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 18 and i < 20):

            rand_set1 = set(np.random.choice(idx1_shard11, 5, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard22, 40, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set2)
            rand_set3 = set(np.random.choice(idx1_shard33, 5, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set3)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set3:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 20 and i < 24):

            rand_set1 = set(np.random.choice(idx1_shard33, 50, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set1)

            for rand in rand_set1:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 24 and i < 26):

            rand_set1 = set(np.random.choice(idx1_shard11, 5, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard33, 45, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 26 and i < 28):

            rand_set1 = set(np.random.choice(idx1_shard22, 5, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard33, 45, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set2)

            for rand in rand_set1:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        elif (i >= 28 and i < 30):

            rand_set1 = set(np.random.choice(idx1_shard11, 5, replace=False))
            idx1_shard11 = list(set(idx1_shard11) - rand_set1)
            rand_set2 = set(np.random.choice(idx1_shard22, 5, replace=False))
            idx1_shard22 = list(set(idx1_shard22) - rand_set2)
            rand_set3 = set(np.random.choice(idx1_shard33, 40, replace=False))
            idx1_shard33 = list(set(idx1_shard33) - rand_set3)

            for rand in rand_set1:
                temp_dict = idx[0][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[0][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set2:
                temp_dict = idx[1][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[1][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

            for rand in rand_set3:
                temp_dict = idx[2][rand * num_imgs: (rand + 1) * num_imgs]
                train_dict_cilent[i] = np.concatenate(
                    (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
                test_dict_cilent[i] = np.concatenate(
                    (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
                dict_users[i] = np.concatenate((dict_users[i], idx[2][rand * num_imgs: (rand + 1) * num_imgs]),
                                               axis=0)

        for it in dict_users[i]:
            dict_user_class[i].add(daset[int(it)][1])

    dictuserdictuser = dict(dict_users)
    dictusertrain = dict(train_dict_cilent)
    dictusertest = dict(test_dict_cilent)
    dictuserclass = dict(dict_user_class)

    """
    df = pd.DataFrame.from_dict(dictuserdictuser, orient='index')
    df = df.transpose()
    df.to_csv('/data/dictuserina.csv', index=False)
    df = pd.DataFrame.from_dict(dictusertrain, orient='index')
    df = df.transpose()
    df.to_csv('/data/dictusertrainina.csv', index=False)
    df = pd.DataFrame.from_dict(dictusertest, orient='index')
    df = df.transpose()
    df.to_csv('/data/dictusertestina.csv', index=False)
    df = pd.DataFrame.from_dict(dictuserclass, orient='index')
    df = df.transpose()
    df.to_csv('/data/dictuserclassina.csv', index=False)
    """
    return train_dict_cilent, test_dict_cilent, dict_user_class


def imagenet_noniid(args, dataset, num_users):
    num_shards, num_imgs = 12811, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    train_dict_cilent = {i: np.array([]) for i in range(num_users)}
    test_dict_cilent = {i: np.array([]) for i in range(num_users)}
    dict_users_per = [[0] * num_users for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    dict_user_class = defaultdict(set)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels1 = idxs_labels[1, :]
    # divide and assign

    for i in range(num_users):

        rand_set = set(np.random.choice(idx_shard, 100, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        j = 0
        for rand in rand_set:
            temp_dict = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            train_dict_cilent[i] = np.concatenate(
                (train_dict_cilent[i], (random.sample(list(temp_dict), int(0.6 * len(temp_dict))))), axis=0)
            test_dict_cilent[i] = np.concatenate(
                (test_dict_cilent[i], list(set(temp_dict) - set(train_dict_cilent[i]))), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)

            for it in labels1[rand * num_imgs: (rand + 1) * num_imgs]:
                # print(it)
                dict_user_class[i].add(it)
    return train_dict_cilent, test_dict_cilent, dict_user_class


def process_data(data):
    """
    Process the data and extract each user's data into a list until encountering -1.
    This function assumes that the data is in the form of a 2D list where each column represents a user and each row represents a data point.
    
    Parameters:
    - data: A 2D list or numpy array where each column represents user data and each row corresponds to a data point for that user.
    
    Returns:
    - A dictionary where the keys are user indices (columns), and the values are lists of user data (excluding -1).
    """
    data_dict = {i: [] for i in range(len(data[0]))}  # Initialize dictionary to hold user data lists
    
    # Iterate over each user (each column in the data)
    for i in range(len(data[0])):
        user_data = []
        
        # Iterate over rows (data points) for each user
        for j in range(1, len(data)):
            if data[j][i] == -1:  # Stop when -1 is encountered
                break
            else:
                user_data.append(data[j][i])  # Add data point to user's data list
        data_dict[i] = user_data  # Store the user's data in the dictionary
    
    return data_dict


def data_reading():
    """
    Read data from CSV files, process the data, and return dictionaries for training data, test data, and class data.


    Returns:
    - datadicttrain: A dictionary with the training data for each user.
    - datadicttest: A dictionary with the test data for each user.
    - dataclass: A dictionary with the class data for each user.
    """
    # Load data from CSV files
    ad = np.genfromtxt('/root/sml/utility/dictuserina_new30.csv', dtype='int', delimiter=',')
    adtrain = np.genfromtxt('/root/sml/utility/dictusertrainina_new30.csv', dtype='int', delimiter=',')
    adtest = np.genfromtxt('/root/sml/utility/dictusertestina_new30.csv', dtype='int', delimiter=',')
    adclass = np.genfromtxt('/root/sml/utility/dictuserclassina_new30.csv', dtype='int', delimiter=',')

    # Process data to convert to dictionaries for each user
    datadicttrain = process_data(adtrain)
    datadicttest = process_data(adtest)
    dataclass = process_data(adclass)

    # Return processed data dictionaries
    return datadicttrain, datadicttest, dataclass


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
