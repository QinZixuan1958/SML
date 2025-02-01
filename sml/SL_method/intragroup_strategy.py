import copy
import numpy as np
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


class IntraGroupSL:
    def __init__(self, args, local_net, group, outstanding, ordinary, now_model, aggregation_sl_strategies, global_ep):
        self.args = args
        self.local_net = local_net
        self.group = group
        self.outstanding = outstanding
        self.ordinary = ordinary
        self.now_model = now_model
        self.aggregation_sl_strategies = aggregation_sl_strategies
        self.global_ep = global_ep

    def intra_group_exchange(self):
        """
        Perform intra-group communication: Each individual in the group calculates similarity with others in the group, 
        selects some outstanding individuals and some similar individuals for model aggregation.
        """
        for group_idx, group_members in self.group.items():
            for idx in group_members:
                # Exclude the current individual's model
                outstandings = [individual for individual in self.outstanding[group_idx] if individual != idx]
                ordinarys = [individual for individual in self.ordinary[group_idx] if individual != idx]

                # Select outstanding individuals for aggregation
                w_outstanding = self._strategies_outstanding_individuals(outstandings, self.args.frac_outstand)

                # Select similar individuals for aggregation
                w_similar = self._strategies_similar_individuals(idx, ordinarys, self.args.frac_similar)

                # Update local model
                self._update_individual_models(idx, w_similar, w_outstanding, self.global_ep)
    
    def intra_group_exchange_all(self):
        """
        Perform intra-group communication: Aggregate all models.
        """
        selected_models = []
        for idx in range(self.args.num_users):
            selected_models.append(self.local_net[idx].state_dict())

        for idx in range(self.args.num_users):
            # Update local model
            self._update_individual_models_all(idx, selected_models)

    def _strategies_outstanding_individuals(self, outstanding_list, fraction):
        """
        Randomly select a fraction of outstanding individuals for aggregation.
        :param outstanding_list: List of individuals
        :param fraction: Fraction of outstanding individuals to select
        :return: Selected individual models
        Can randomly select or select the most similar top-k outstanding individuals based on similarity
        """
        selected_models = []
        num_selected = int(fraction * len(outstanding_list))

        # Randomly select outstanding individuals
        selected_individuals = np.random.choice(outstanding_list, num_selected, replace=False)

        for individual in selected_individuals:
            selected_models.append(self.local_net[individual].state_dict())

        return selected_models

    def _strategies_similar_individuals(self, idx, ordinary_list, fraction):
        """
        Select the most similar `top_k` individuals to the current individual for aggregation.
        :param idx: The ID of the current individual
        :param ordinary_list: List of individuals
        :param top_k: Number of most similar individuals to select
        :return: Selected individual models
        """
        # Calculate similarity with the current individual
        similar_individuals = self._compute_similarity(idx, ordinary_list)

        # Sort by similarity and select the most similar individuals
        sorted_similar_individuals = sorted(similar_individuals.items(), key=lambda item: item[1], reverse=True)

        top_k = int(fraction * len(ordinary_list))
        selected_models = []
        for individual_idx, _ in sorted_similar_individuals[:top_k]:
            selected_models.append(self.local_net[individual_idx].state_dict())

        return selected_models

    def _compute_similarity(self, idx, individual_list):
        """
        Compute the similarity between the current individual and other individuals using cosine similarity.
        :param idx: The ID of the current individual
        :param individual_list: List of individuals
        :return: A dictionary of similarities between the individual and others
        """
        similarities = {}
        for individual_idx in individual_list:
            similarities[individual_idx] = cos_similiar(self.local_net[idx], self.local_net[individual_idx])

        return similarities

    def _update_individual_models(self, idx, w_similar, w_outstanding, global_ep):
        """
        Update the model of the individual in the group.
        :param idx: The index of the current individual model
        :param w_good: The aggregated weights for the group
        """
        w_glob = self.aggregation_sl_strategies(self.args.gamma, w_similar, w_outstanding, global_ep)
        self.now_model[idx].load_state_dict(w_glob)
        
    def _update_individual_models_all(self, w, idx):
        """
        Update the model of the individual in the group.
        :param idx: The index of the current individual model
        :param w_good: The aggregated weights for the group
        """
        w_glob = self.aggregation(w)
        self.now_model[idx].load_state_dict(w_glob)

    def aggregation(self, w):
        pass
