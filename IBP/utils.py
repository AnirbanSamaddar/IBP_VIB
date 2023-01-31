from imghdr import tests
import torch
import torch.nn
import torch.nn.init
import torchvision#, torchtext
import argparse
import sklearn.datasets
import numpy as np
import random
import os
from typing import Any

class Dataset(torch.utils.data.Dataset):

    def __init__(self,X,Y):
        self.data = X
        self.targets = Y
    
    def __getitem__(self,index):
        data, target = torch.from_numpy(self.data[index,:]).type('torch.FloatTensor')\
                        , torch.from_numpy(self.targets[index]).squeeze()
        return data, target
    
    def __len__(self):
        return len(self.targets)


def get_mnist(d=0):
    ''' 
    This function returns the MNIST dataset in training, validation, test splits.
    '''

    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))

    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,)), torchvision.transforms.RandomRotation(degrees=(d,d))]))
    
    return trainset, testset


def get_CIFAR10(d=0):
    data_path_C10 = "../data/CIFAR10/"
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        torchvision.transforms.RandomRotation(degrees=(d,d))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_path_C10, train=True,
                    download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path_C10, train=False,
                    download=True, transform=transform_test)
    return trainset,testset


def get_manifold(dataset_name='dollar'):
    inputs = np.load('/mnt/home/samadda1/Intrinsic_SparC_IB/data/{}/input.npy'.format(dataset_name))
    targets = np.load('/mnt/home/samadda1/Intrinsic_SparC_IB/data/{}/target.npy'.format(dataset_name))
    trainset = Dataset(inputs[0:8000,],targets[0:8000])
    testset = Dataset(inputs[8000:10000,],targets[8000:10000])
    return trainset,testset

def get_data(dataset='mnist',d=0,noisy=False):
    '''
    This function returns the training and validation set from MNIST
    '''

    if dataset == 'mnist':
        return get_mnist(d)
    elif dataset == 'CIFAR10':
        #return get_CIFAR10T(noisy)
        return get_CIFAR10(d)
    else:
        return get_manifold(dataset_name=dataset)

def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Run convex IB Lagrangian on MNIST dataset (with Pytorch)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--logs_dir', default = '../results/logs/',
        help = 'folder to output the logs')
    parser.add_argument('--figs_dir', default = '../results/figures/',
        help = 'folder to output the images')
    parser.add_argument('--models_dir', default = '../results/models/',
        help = 'folder to save the models')
    parser.add_argument('--repl_n', type = int, default = 1,
        help = 'replication number')
    parser.add_argument('--rotation_deg', type = int, default = 0,
        help = 'rotation degree of test data')
    parser.add_argument('--n_epochs', type = int, default = 100,
        help = 'number of training epochs')
    parser.add_argument('--beta', type = float, default = 0.0,
        help = 'Lagrange multiplier (only for train_model)')
    parser.add_argument('--dim_pen', type = float, default = 1.0,
        help = 'Penalty on the dimension encoder KL')
    parser.add_argument('--n_betas', type = int, default = 50,
        help = 'Number of Lagrange multipliers (only for study behavior)')
    parser.add_argument('--beta_lim_min', type = float, default = 0.0,
        help = 'minimum value of beta for the study of the behavior')
    parser.add_argument('--beta_lim_max', type = float, default = 1.0,
        help = 'maximum value of beta for the study of the behavior')  
    parser.add_argument('--u_func_name', choices = ['pow', 'exp','shifted-exp','none'], default = 'exp',
        help = 'monotonically increasing, strictly convex function')
    parser.add_argument('--hyperparameter', type = float, default = 1.0,
        help = 'hyper-parameter of the h function (e.g., alpha in the power and eta in the exponential case)')
    parser.add_argument('--compression', type = float, default = 1.0,
        help = 'desired compression level (in bits). Only for the shifted exponential.')
    parser.add_argument('--example_clusters', action = 'store_true', default = False,
        help = 'plot example of the clusters obtained (only for study behavior of power with alpha 1, otherwise change the number of clusters to show)')
    parser.add_argument('--K', type = int, default = 2,
        help = 'Dimensionality of the bottleneck varaible')
    parser.add_argument('--K_array', type = int, default = np.geomspace(2,16,4).round(),
        help = 'Dimensionality of the bottleneck varaible')    
    parser.add_argument('--logvar_kde', type = float, default = -1.0,
        help = 'initial log variance of the KDE estimator')
    parser.add_argument('--a', type = float, default = 1.0,
        help = 'prior beta parameter 1')
    parser.add_argument('--b', type = float, default = 1.0,
        help = 'prior beta parameter 1')
    parser.add_argument('--prior', choices = ['compound', 'cat'], default = 'compound',
        help = 'prior type')
    parser.add_argument('--logvar_t', type = float, default = -1.0,
        help = 'initial log varaince of the bottleneck variable')
    parser.add_argument('--sgd_batch_size', type = int, default = 128,
        help = 'mini-batch size for the SGD on the error')
    parser.add_argument('--mi_batch_size', type = int, default = 1000,
        help = 'mini-batch size for the I(X;T) estimation')
    parser.add_argument('--same_batch', action = 'store_true', default = True,
        help = 'use the same mini-batch for the SGD on the error and I(X;T) estimation')
    parser.add_argument('--dataset', choices = ['mnist','CIFAR10','dollar','s_in_a_box','s-and-plane'], default = 'mnist',
        help = 'dataset where to run the experiments. Classification: MNIST or CIFAR10.')
    parser.add_argument('--optimizer_name', choices = ['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'], default = 'adam',
        help = 'optimizer')
    parser.add_argument('--method', choices = [ 'IBP'], default =  'variational_IB',
        help = 'information bottleneck computation method')
    parser.add_argument('--learning_rate', type = float, default = 0.0001,
        help = 'initial learning rate')
    parser.add_argument('--learning_rate_drop', type = float, default = 0.6,
        help = 'learning rate decay rate (step LR every learning_rate_steps)')
    parser.add_argument('--learning_rate_steps', type = int, default = 10,
        help = 'number of steps (epochs) before decaying the learning rate')
    parser.add_argument('--train_logvar_t', action = 'store_true', default = False,
        help = 'train the log(variance) of the bottleneck variable')
    parser.add_argument('--eval_rate', type = int, default = 20,
        help = 'evaluate I(X;T), I(T;Y) and accuracies every eval_rate epochs')
    parser.add_argument('--visualize', action = 'store_true', default = False,
        help = 'visualize the results every eval_rate epochs')
    parser.add_argument('--verbose', action = 'store_true', default = False,
        help = 'report the results every eval_rate epochs')

    return parser.parse_args()
