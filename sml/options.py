#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=41, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: n")
    parser.add_argument('--num_group', type=int, default=3, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--split_ratio', type=float, default=0.6, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=30, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--overlapping_classes', type=int, default=5, help="maximum number of samples/user to use for fine-tuning")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")    
    parser.add_argument('--alpha', type=float, default=0.1, help='hyper parameter for SL')
    parser.add_argument('--gamma', type=float, default=0.2, help='hyper parameter for SL strategy')
    parser.add_argument('--top_k_inter', type=int, default=3, help='select the top_k most similar edge individuals')
    parser.add_argument("--lamda", type=float, default=0.5, help="Regularization term")
    parser.add_argument('--frac_outstand', type=float, default=0.8, help="the fraction of outstanding: C")
    parser.add_argument('--frac_similar', type=float, default=0.6, help="the fraction of similar: C")



    # model arguments
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--base_layers', type=int, default=45, help='number layers to keep')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')


    # other arguments
    parser.add_argument('--dataset', type=str, default='ina17', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--num_classes', type=int, default=5089, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', default=5, type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')

    args = parser.parse_args()
    return args
