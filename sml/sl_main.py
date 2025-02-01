import numpy as np
import torch
import time
import copy
import logging
import warnings
from SL_method.aggregation_method import aggregation, aggregation_sl_strategies
from SL_method.intergroup_strategy import InterGroupSL
from SL_method.intragroup_strategy import IntraGroupSL
from SL_method.role_allocation import RoleAssignment
from models.test import test_client
from models.test_individual import IndividualTester
from models.train_individual import IndividualTrainer
from utility.initial import initialize_data, initialize_models, initialize_train
from utility.network_construction import net_construct_cluster, net_construct, community_detection
from utility.LoadSplit import Load_Dataset, Load_Model
import csv
warnings.filterwarnings("ignore")

# Directory where the json file of arguments will be present
directory = './Parse_Files'
from options import args_parser

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    save_dir = '/root/user/'
    # Initialize argument dictionary
    args = args_parser()
    init_train = False
    # Setting the device - GPU or CPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    net_initial = Load_Model(args=args)
    local_net, edge_model, now_model, w_globs, w_locals = initialize_models(args, net_initial)

    # Set up log file

    dataset, dict_users_train, dict_users_test, dict_user_class = Load_Dataset(args=args)

    train_data_users, test_data_users, pub_dataset = initialize_data(args, dict_users_train, dict_users_test)

    if init_train:
        local_net, edge_model, now_model = initialize_train(args, dataset, train_data_users, test_data_users,
                                                            local_net, edge_model, now_model)
    group = community_detection(args, dict_user_class)
    print(group)
    new_G, edge_users = net_construct(group, dict_user_class)

    for group_idx in group.keys():
        print(f"Group {group_idx}:")
        print(f"  Group members: {group[group_idx]}")
        print(f"  Edge users: {edge_users[group_idx]}")

    all_edge_users = []
    # add in all_edge_users列表中
    for group_edge_users in edge_users.values():
        all_edge_users.extend(group_edge_users)

    # Initialize individual performance, good performance is 1, bad performance is 0
    performance = [0] * args.num_users
    client_good = {i: [] for i in range(args.num_group)}
    ordinary = {i: [] for i in range(args.num_group)}

    for i in range(args.num_group):
        ordinary[i] = group[i]

    individual_test = {}


    print("---------Initialization training starts---------")
    for idx in range(0, args.num_users):
        
        individual_trainer = IndividualTrainer(args, dataset, train_data_users[idx], local_net[idx], now_model[idx],
                                               iter, performance[idx])
        loss, train_accuracy, w_individual = individual_trainer.init_train()
        w_locals[idx] = w_individual
        local_net[idx].load_state_dict(w_individual)

        print('-------------------')
        print("Client {}:".format(idx))
        print("Training loss: {:.3f}".format(loss))
        print("Training accuracy: {:.3f}".format(train_accuracy * 100))
        print('-------------------')

    print("---------Initialization training ends---------")


    # Start training
    start = time.time()
    print("Start of Training")
    for iter in range(args.epochs):
        # individual local training
        control_lambda =  iter % 3
        print('Round {}'.format(iter))
        print("---------Round {}---------".format(iter))

        all_test = []
        for idx in range(0, args.num_users):
            individual_trainer = IndividualTrainer(args, dataset, train_data_users[idx], local_net[idx], now_model[idx],
                                                   iter, performance[idx])
            loss, train_accuracy, w_individual = individual_trainer.train()
            w_locals[idx] = w_individual
            local_net[idx].load_state_dict(w_individual)

            individual_tester = IndividualTester(args, dataset, local_net[idx], test_data_users[idx])
            test_accuracy, test_loss, f1 = individual_tester.test_individual()
            all_test.append(test_accuracy.cpu().detach().numpy())
            individual_test[idx] = test_accuracy.cpu().detach().numpy()

            print('-------------------')
            print("Client {}:".format(idx))
            print("Training loss: {:.3f}".format(loss))
            print("Training accuracy: {:.3f}".format(train_accuracy * 100))
            print("Test accuracy: {:.3f}".format(test_accuracy))
            print("Test loss: {:.3f}".format(test_loss))
            print('-------------------')

        all_test_avg = sum(all_test) / len(all_test)
        print("Average Client accuracy on their test data {}:".format(all_test_avg))
        
        # all_test_avg 
        with open('all_test_avg.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if iter == 0:  
                writer.writerow(["Round", "Average Test Accuracy"])
            writer.writerow([iter, all_test_avg])

        # Instantiate the Role Assignment class
        role_assigner = RoleAssignment(args, group, dataset, pub_dataset, local_net, test_client, edge_users,
                                       performance)
        
        # Get outstanding and ordinary individuals
        if iter % 3 == 0:
            outstanding, ordinary, performance = role_assigner.role_assignment()
            print("Outstanding individuals:", outstanding)
            print("Ordinary individuals:", ordinary)
        
        # Perform inter-group and intra-group cross-learning with different frequencies
        if control_lambda == 0:
            # Create an InterGroupSL instance
            inter_sl = InterGroupSL(args, local_net, group, outstanding, ordinary, all_edge_users, edge_users,
                                    now_model,
                                    edge_model, aggregation)
        
            # Perform inter-group communication
            print("=================inter-group communication===============")
            inter_sl.exchange_individual()
        
        else:
            # Create an IntraGroupSL instance
            intra_sl = IntraGroupSL(args, local_net, group, outstanding, ordinary, now_model, aggregation_sl_strategies,
                                    iter)
        
            # Perform intra-group communication
            print("=================intra-group communication===============")
            intra_sl.intra_group_exchange()


    end = time.time()

    print("Training Time: {}s".format(end - start))
    print("End of Training")
