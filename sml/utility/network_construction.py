import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
import networkx as nx
import community as community_louvain
import numpy as np

def community_detection(args, dict_user_class):
    """
    Use the Louvain community detection algorithm to divide clients into multiple groups based on their class lists.
    :param args: Parameters containing num_users
    :param dict_user_class: A dictionary where the key is the user ID and the value is the list of classes for that user
    :return: A dictionary of groups, where each group contains a list of user IDs belonging to that group
    """

    # Step 1: Build a graph where edge weights represent the number of common classes between users
    G = nx.Graph()

    # Add users as nodes to the graph
    for user_id in range(args.num_users):
        G.add_node(user_id)

    # Calculate common classes between users and create edges with weights
    for i in range(args.num_users):
        for j in range(i + 1, args.num_users):
            common_classes = set(dict_user_class[i]) & set(dict_user_class[j])
            if common_classes:
                # The weight of the edge is the number of common classes
                G.add_edge(i, j, weight=len(common_classes))

    # Step 2: Use the Louvain method for community detection
    partition = community_louvain.best_partition(G, weight='weight')

    # Step 3: Divide users into different groups based on the community partition
    group = {}
    for user_id, group_id in partition.items():
        if group_id not in group:
            group[group_id] = []
        group[group_id].append(user_id)

    return group


def fun(args, arr):
    ret = [1 if i in arr else 0 for i in range((args.num_classes-1) + 1)]
    return ret


def net_construct_cluster(args, dict_user_class):
    """
    Sample non-I.I.D client data from CIFAR dataset and perform clustering based on class distribution
    :param args: Arguments containing clustering parameters
    :param dataset: CIFAR dataset
    :param num_users: Number of users (clients)
    :return: Data dictionary, clusters, subgroups, class distribution, network graph
    """

    dataclass = {}
    for i in range(args.num_users):
        dataclass[i] = list(dict_user_class[i])

    # Create a similarity matrix based on common classes between users
    matrix_ij = np.zeros((args.num_users, args.num_users))
    for i in range(args.num_users):
        for j in range(args.num_users):
            set_c = list(set(dataclass[i]) & set(dataclass[j]))

            matrix_ij[i][j] = len(set_c)

    # Build Network Graph
    G = nx.Graph()
    H = nx.path_graph(len(matrix_ij))
    G.add_nodes_from(H)
    for i in range(len(matrix_ij)):
        for j in range(len(matrix_ij)):
            if (matrix_ij[i][j] != 0 and i != j):
                G.add_edge(i, j)

    X = []  # Ensure you have the data `X` for clustering
    for i in range(args.num_users):
        X.append(fun(args, dataclass[i]))
    X = np.array(X)
    
    # Initial Clustering with KMeans
    num_cluster = args.num_group
    kmeans = KMeans(n_clusters=num_cluster).fit(X)
    user_label = kmeans.labels_

    # Step 5: Second-Level Clustering within each cluster
    group = {i: [] for i in range(num_cluster)}
    for i in range(args.num_users):
        group[user_label[i]].append(i)

    # Create subgroups within each cluster
    subgroup = {i: [] for i in range(num_cluster)}

    # First, compute the size of each cluster
    cluster_sizes = {i: len(group[i]) for i in range(num_cluster)}
    print(group)
    # Now, within each cluster, we select the individuals to form subgroups
    for cluster_idx in range(num_cluster):
        cluster_users = group[cluster_idx]
        cluster_vectors = X[cluster_users]

        # Perform KMeans again within each cluster
        kmeans_sub = KMeans(n_clusters=2).fit(cluster_vectors)
        sub_labels = kmeans_sub.labels_

        # Compute the number of individuals in each subcluster
        subcluster_sizes = {0: 0, 1: 0}
        for label in sub_labels:
            subcluster_sizes[label] += 1

        # Choose the subgroup with fewer individuals (the one with the smallest size)
        smallest_subcluster_label = 0 if subcluster_sizes[0] < subcluster_sizes[1] else 1

        # Append the users from the smaller subgroup to the main subgroup of the cluster
        for i, user_idx in enumerate(cluster_users):
            if sub_labels[i] == smallest_subcluster_label:
                subgroup[cluster_idx].append(user_idx)

    # Step 6: Visualization of Clusters and Subgroups
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))

    # Draw the network graph with different colors for each cluster
    for cluster_idx in range(num_cluster):
        nx.draw_networkx_nodes(G, pos, nodelist=group[cluster_idx], node_color=f"C{cluster_idx}",
                               label=f"Cluster {cluster_idx}")
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.axis('off')
    plt.title("Network Graph based on Class Distribution Similarity")
    plt.show()

    return group


def construct_graph_based_on_classes(dict_user_class, num_users):
    """
    构建一个个体之间类别联系图。
    :param dict_user_class: 用户与其类别的字典
    :param num_users: 用户数量
    :return: 类别联系图 G
    """
    G = nx.Graph()

    # 建立图，节点是用户，边连接有共同类别的用户
    for i in range(num_users):
        for j in range(i + 1, num_users):
            # 如果两个用户有共同的类别，则建立边
            if len(list(set(dict_user_class[i]) & set(dict_user_class[j]))) > 0:
                G.add_edge(i, j)

    return G


def get_edge_users_by_degree(group_dict, G):
    """
    根据度数为每个组选择边缘个体。
    :param group_dict: 每个组的用户集合
    :param G: 类别联系图
    :return: 边缘个体集合
    """
    edge_users = {i: [] for i in group_dict.keys()}

    # 计算每个用户的度数
    user_degrees = dict(G.degree())

    for group_idx, users in group_dict.items():
        # 计算每个组内用户的度数
        sorted_users = sorted(users, key=lambda x: user_degrees[x], reverse=True)

        # 选择度数最大的几个用户作为边缘个体
        if len(sorted_users) > 1:
            edge_users[group_idx].append(sorted_users[0])  # 选度数最大的用户
            edge_users[group_idx].append(sorted_users[1])  # 也可以选第二大的用户
        else:
            edge_users[group_idx].append(sorted_users[0])

    return edge_users


def modify_graph_based_on_edge_users(G, edge_users, group_dict):
    """
    删除图中非边缘个体之间的边，只保留边缘个体与其他小组的连接。
    :param G: 原始图
    :param edge_users: 每个组的边缘个体集合
    :param group_dict: 每个组的用户集合
    :return: 修改后的图
    """
    # 创建一个新的图副本
    new_G = G.copy()

    # 遍历每条边，删除不符合要求的边
    for edge in list(new_G.edges()):
        u, v = edge

        # 如果两个节点不属于同一个组，并且其中至少一个节点不是边缘个体，则删除该边
        u_group = None
        v_group = None
        for group_idx, users in group_dict.items():
            if u in users:
                u_group = group_idx
            if v in users:
                v_group = group_idx

        if u_group != v_group:  # 不同组
            if u not in edge_users[u_group] and v not in edge_users[v_group]:
                new_G.remove_edge(u, v)

    return new_G


def visualize_graph(G):
    """
    可视化图 G
    :param G: 图对象
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.axis('off')
    plt.title("Modified Network Graph with Edge Users")
    plt.show()


def net_construct(group, dict_user_class):
    """
    基于现有的组信息生成并修改网络图。
    :param group: 包含每个组的个体 ID 列表
    :param dict_user_class: 用户与类别的字典
    :return: 最终的图 G
    """
    # Step 1: 基于类别交集构建原始图
    num_users = sum(len(g) for g in group.values())
    
    G = construct_graph_based_on_classes(dict_user_class, num_users)

    # Step 2: 根据给定的 group 字典生成用户的分组
    group_dict = {i: group[i] for i in range(len(group))}

    # Step 3: 根据度数选择边缘个体
    edge_users = get_edge_users_by_degree(group_dict, G)

    # Step 4: 修改图，删除不符合条件的边
    new_G = modify_graph_based_on_edge_users(G, edge_users, group_dict)

    # Step 5: 可视化最终的图
    visualize_graph(new_G)

    return new_G, edge_users


