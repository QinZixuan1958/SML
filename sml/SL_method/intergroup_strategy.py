import numpy as np
import copy
import torch.nn.functional as F

def cos_similiar(model1, model2):
    """
    Calculate the cosine similarity between two models.
    :param model1: Model 1
    :param model2: Model 2
    :return: Cosine similarity
    """
    sum_sim = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        sum_sim += F.cosine_similarity(param1.view(-1, 1), param2.view(-1, 1), dim=0, eps=1e-8)
    return sum_sim


class InterGroupSL:
    def __init__(self, args, local_net, group, outstanding, ordinary, all_edge_users, edge_users, now_model, edge_model,
                 aggregation):
        self.args = args
        self.local_net = local_net
        self.group = group
        self.outstanding = outstanding
        self.ordinary = ordinary
        self.all_edge_users = all_edge_users
        self.edge_users = edge_users
        self.now_model = now_model
        self.edge_model = edge_model
        self.aggregation = aggregation

    def exchange_individual(self):
        """
        Aggregate individuals into edge and non-edge groups.
        """
        # Aggregate edge models
        self._aggregate_edge_individuals()

        # Perform inter-group exchange
        self._exchange_intergroup()

    def _aggregate_edge_individuals(self):
        """
        Aggregate edge individuals: Select ordinary and outstanding individuals from each group for aggregation.
        """
        ordinary_individuals = {i: copy.deepcopy(self.ordinary[i]) for i in range(self.args.num_group)}  # Record ordinary individuals for each group
        outstanding_individuals = {i: copy.deepcopy(self.outstanding[i]) for i in range(self.args.num_group)}  # Copy outstanding individuals for each group

        # Select ordinary individuals from each group
        for group_idx in range(self.args.num_group):

            # Control the number of individuals to aggregate per group
            l_outstanding = int(0.4 * len(self.outstanding[group_idx]))
            l_ordinary = int(0.2 * len(self.ordinary[group_idx]))

            for idx in self.edge_users[group_idx]:
                w_edge = self._select_individuals_for_aggregation(outstanding_individuals[group_idx],
                                                                   ordinary_individuals[group_idx], l_outstanding,
                                                                   l_ordinary)
                self._update_edge_models(idx, w_edge)

    def _select_individuals_for_aggregation(self, outstanding_individuals, ordinary_individuals, l_outstanding, l_ordinary):
        """
        Aggregate a group of outstanding and ordinary individuals.
        :param outstanding_individuals: Outstanding individuals of the current group
        :param ordinary_individuals: Ordinary individuals of the current group
        :param l_outstanding: Number of outstanding individuals to select
        :param l_ordinary: Number of ordinary individuals to select
        :return: Aggregated model
        """
        w = []

        # Select outstanding individuals
        for idx in np.random.choice(outstanding_individuals, l_outstanding, replace=False):
            w.append(self.local_net[idx].state_dict())
            outstanding_individuals.remove(idx)

        # Select ordinary individuals
        for idx in np.random.choice(ordinary_individuals, l_ordinary, replace=False):
            w.append(self.local_net[idx].state_dict())
            ordinary_individuals.remove(idx)

        return w

    def _update_edge_models(self, idx, w_edge):
        """
        Update the model of individuals in the group.
        :param idx: Index of the current edge model
        :param w_good: Aggregated weights for the group
        """
        w_glob = self.aggregation(w_edge)
        self.edge_model[idx].load_state_dict(w_glob)

    def _update_individual_models(self, idx, w):
        """
        Update the model of individuals in the group.
        :param idx: Index of the current individual model
        :param w_good: Aggregated weights for the group
        """
        w_glob = self.aggregation(w)
        self.now_model[idx].load_state_dict(w_glob)

    def _exchange_intergroup(self):
        """
        Aggregate non-edge individuals, selecting the most similar edge individuals for aggregation.
        """
        for group_idx in range(self.args.num_group):
            for idx in self.group[group_idx]:
                w = self._strategies_similar_individuals(idx, group_idx)
                self._update_individual_models(idx, w)

    def _strategies_similar_individuals(self, idx, group_idx):
        """
        Select the most similar edge individuals that do not belong to the current individual’s group.
        :param idx: ID of the current individual
        :return: Most similar edge individual models
        """
        similar = {}
        w = []

        # Iterate over all edge individuals, calculate similarity, and exclude edge individuals from the current group
        for i, edge_group in self.edge_users.items():
            # Exclude the current individual's group
            if i != group_idx:
                for edge_idx in edge_group:
                    similar[edge_idx] = cos_similiar(self.local_net[idx], self.edge_model[edge_idx])

        # Sort by similarity
        sorted_diss = sorted(similar.items(), key=lambda d: d[1], reverse=True)

        if idx in self.edge_users[group_idx]:
            top_k = self.args.top_k_inter + 1
        else:
            top_k = self.args.top_k_inter

        # Select the top 'top_k' most similar edge individuals
        for j, _ in sorted_diss[:top_k]:
            w.append(self.edge_model[j].state_dict())

        return w

    def intra_group_exchange_all(self):
        """
        Perform intra-group exchange: aggregate all models.
        """
        selected_models = []
        for idx in range(self.args.num_users):
            selected_models.append(self.local_net[idx].state_dict())

        for idx in range(self.args.num_users):
            # Update local models
            self._update_individual_models(idx, selected_models)
