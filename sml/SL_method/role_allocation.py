import torch
import torch.nn.functional as F
import math
import warnings

warnings.filterwarnings("ignore")


class RoleAssignment:
    def __init__(self, args, group, dataset, pub_dataset, local_net, test_individual, edge_users, performance):
        """
        Initialize the RoleAssignment class.

        :param args: Contains parameters such as num_users, num_group, etc.
        :param group: Dictionary, where the key is the group ID and the value is a list of user IDs in that group.
        :param dataset: The test dataset.
        :param pub_dataset: The public dataset for each group.
        :param local_net: Local neural networks corresponding to each individual.
        :param test_individual: Function to test individual performance, returns accuracy and loss.
        :param edge_users: Dictionary of edge users for each group, where the key is group ID and the value is a list of edge user IDs.
        """
        self.args = args
        self.group = group
        self.dataset = dataset
        self.pub_dataset = pub_dataset
        self.local_net = local_net
        self.test_individual = test_individual
        self.edge_users = edge_users  # Records the edge users for each group
        self.performance = performance
        self.outstanding = {i: [] for i in range(args.num_group)}  # Initialize list of "outstanding" individuals
        self.ordinary = {i: [] for i in range(args.num_group)}  # Initialize list of "ordinary" individuals

    def cos_similiar(self, model1, model2):
        """
        Calculate the cosine similarity between two models, considering all layers.

        :param model1: The first model.
        :param model2: The second model.
        :return: The total cosine similarity between the two models.
        """
        sum = 0.0
        for parameter_tem, parameter_avg in zip(model1.parameters(), model2.parameters()):
            sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
        return sum

    def cos_similiar1(self, model1, model2):
        """
        Calculate the cosine similarity between two models, considering the first `args.base_layers` layers.

        :param model1: The first model.
        :param model2: The second model.
        :return: The total cosine similarity between the first `args.base_layers` layers of the two models.
        """
        sum = 0.0
        for parameter_tem, parameter_avg in zip(model1.parameters(), model2.parameters())[:self.args.base_layers]:
            sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
        return sum

    def cos_similiar2(self, model1, model2):
        """
        Calculate the cosine similarity between two models, considering layers starting from `args.base_layers`.

        :param model1: The first model.
        :param model2: The second model.
        :return: The total cosine similarity between layers starting from `args.base_layers` of the two models.
        """
        sum = 0.0
        for parameter_tem, parameter_avg in zip(model1.parameters(), model2.parameters())[self.args.base_layers:]:
            sum += F.cosine_similarity(parameter_tem.view(-1, 1), parameter_avg.view(-1, 1), dim=0, eps=1e-8)
        return sum

    def role_assignment(self):
        """
        Perform role assignment based on individual performance and categories, 
        determining "outstanding" and "ordinary" individuals.
        :return: Updated `outstanding` and `ordinary` dictionaries.
        """
        # Clear the "outstanding" list for each group
        for i in range(self.args.num_group):
            self.outstanding[i].clear()

        # Calculate the accuracy for each group and update "outstanding" individuals
        for idx in range(self.args.num_group):
            acc_avg = 0  # Average accuracy for the group
            test_results = {}

            # Calculate the accuracy for all users in the group, excluding edge users
            for j in self.group[idx]:
                if j not in self.edge_users[idx]:  # Exclude edge users
                    acc_test, _ = self.test_individual(self.args, self.dataset, self.pub_dataset[idx],
                                                    self.local_net[j])
                    test_results[j] = acc_test.cpu().detach().numpy()
                    acc_avg += test_results[j]

            # Calculate the average accuracy
            acc_avg /= (len(self.group[idx]) - len(self.edge_users[idx]))
            #print((len(self.group[idx]) - len(self.edge_users[idx])))
            print(acc_avg)
            # Sort the results based on accuracy
            sorted_results = sorted(test_results.items(), key=lambda d: d[1], reverse=True)
            print(sorted_results)
            # Update the "outstanding" individuals list
            self.outstanding[idx] = [user_id for user_id, acc in sorted_results if acc > acc_avg]
        print(self.outstanding)

        # 3. Update the "ordinary" individuals list and ensure edge users are always ordinary
        for i in range(self.args.num_group):
            # Set edge users as ordinary individuals
            #self.ordinary[i] = [user_id for user_id in self.group[i] if user_id not in self.outstanding[i]] + \
            #                   self.edge_users[i]
            self.ordinary[i] = list(set(self.group[i]) - set(self.outstanding[i]) | set(self.edge_users[i]))
        # Ensure edge users do not appear in the "outstanding" list
        for i in range(self.args.num_group):
            self.outstanding[i] = [user_id for user_id in self.outstanding[i] if user_id not in self.edge_users[i]]

        # Update performance scores for each individual based on their role
        for i in range(self.args.num_group):
            # Set performance to 1 for "outstanding" individuals
            for user_id in self.outstanding[i]:
                self.performance[user_id] = 1
            # Set performance to 0 for "ordinary" individuals
            for user_id in self.ordinary[i]:
                self.performance[user_id] = 0

        return self.outstanding, self.ordinary, self.performance

    def assign_similar_individuals(self, k, a):
        """
        Assign ordinary individuals into similar and dissimilar groups based on cosine similarity 
        to the k-th individual.

        :param k: The index of the k-th individual.
        :param a: The number of most similar individuals to select.
        :return: A tuple of the most similar individuals and the dissimilar ones.
        """
        similar_individuals = []
        dissimilar_individuals = []

        # Get the model of the k-th individual
        model_k = self.local_net[k]

        similarities = []
        # Calculate cosine similarity for all ordinary individuals with the k-th individual
        for i in self.ordinary[k]:
            model_i = self.local_net[i]
            similarity = self.cos_similiar(model_k, model_i)  # Use cos_similiar1 or cos_similiar2 as needed
            similarities.append((i, similarity))  # Pair user ID with similarity

        # Sort similarities from high to low
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Select the top 'a' most similar individuals
        for i, similarity in similarities[:a]:
            similar_individuals.append(i)

        # The remaining individuals are considered dissimilar
        dissimilar_individuals = [i for i, similarity in similarities[a:]]

        return similar_individuals, dissimilar_individuals
