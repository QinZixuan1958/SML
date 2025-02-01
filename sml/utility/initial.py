import random
import numpy as np
import copy


def initialize_data(args, dict_users_train, dict_users_test):
    """
    初始化每个用户的训练数据、测试数据，创建公共数据集和边缘数据集。

    参数:
        args: 配置参数
        dict_users_train: 划分后的训练数据
        dict_users_test: 划分后的测试数据

    返回:
        train_data_users: 每个用户的训练数据
        test_data_users: 每个用户的测试数据
        pubdataset: 各集群的公共数据
        train_edge: 包含边缘数据集的字典
    """
    # 初始化训练数据和测试数据

    train_data_users = {}
    test_data_users = {}
    for i in range(args.num_users):
        train_data_users[i] = list(dict_users_train[i])
        test_data_users[i] = list(dict_users_test[i])

    # 创建公共数据集
    pub_dataset = {i: np.array([]) for i in range(args.num_group)}

    for j in range(args.num_users):
        if 0 <= j < 10:
            pub_dataset[0] = np.concatenate((pub_dataset[0], list(random.sample(train_data_users[j], 20))), axis=0)
        elif 10 <= j < 20:
            pub_dataset[1] = np.concatenate((pub_dataset[1], list(random.sample(train_data_users[j], 20))), axis=0)
        elif 20 <= j < 30:
            pub_dataset[2] = np.concatenate((pub_dataset[2], list(random.sample(train_data_users[j], 20))), axis=0)

    return train_data_users, test_data_users, pub_dataset


def initialize_models(args, net_glob):
    """
    初始化模型，加载全局模型的权重，并为每个客户端创建本地模型。

    参数:
        args: 配置参数
        num_users: 客户端数量
        net_glob: 全局模型

    返回:
        local_nets: 每个客户端的本地模型
        old_model: 保存历史模型的字典
        now_model: 当前模型的字典
        model_weights: 初始化后的权重字典
    """
    # 加载全局模型的权重
    w_glob = net_glob.state_dict()

    # 初始化模型字典
    local_net = {}
    edge_model = {}
    now_model = {}
    w_locals = {}
    w_globs = {}

    # 初始化每个客户端的模型，并加载全局模型的权重
    for i in range(args.num_users):
        local_net[i] = net_glob
        edge_model[i] = net_glob
        now_model[i] = net_glob

        # 设置模型为训练模式
        local_net[i].train()
        edge_model[i].train()
        now_model[i].train()

        # 加载全局模型的权重
        local_net[i].load_state_dict(w_glob)
        edge_model[i].load_state_dict(w_glob)
        now_model[i].load_state_dict(w_glob)
        w_globs[i] = w_glob
        w_locals[i] = w_glob

    return local_net, edge_model, now_model, w_globs, w_locals


def initialize_train(args, dataset, train_data_users, test_data_users,
                     local_net, edge_model, now_model):
    """
    对所有客户端进行训练，并评估其性能。

    参数:
        args: 配置参数
        datasett: 数据集
        train_data_users: 每个用户的训练数据
        test_data_users: 每个用户的测试数据
        local_nets: 本地模型字典
        oldmodel: 每个客户端的旧模型
        now_model: 每个客户端的当前模型
        annstep: 每个客户端的步骤数

    返回:
        w_locals1: 每个客户端的权重
    """

    # 遍历每个客户端进行训练与测试
    for idx in range(0, args.num_users):
        # 训练客户端
        w, loss, accc = train_client2(args, dataset, train_data_users[idx],
                                      net=local_net[idx], oldmodel=edge_model[idx],
                                      net_glob=now_model[idx], user_id=idx)

        # 更新本地模型的权重、损失等信息
        edge_model[idx].load_state_dict(w)
        local_net[idx].load_state_dict(w)
        now_model[idx].load_state_dict(w)

    return local_net, edge_model, now_model
