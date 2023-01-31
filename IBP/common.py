import pdb
import math
import sys
import os
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from derivatives import Beta, Digamma
SMALL = 1e-7
EULER_GAMMA = np.euler_gamma


def kl_divergence(a, b, prior_alpha = 5., prior_beta = 1., log_beta_prior = np.log(1./5.), num_terms=10):
    """
    KL divergence between Kumaraswamy(a, b) and Beta(prior_alpha, prior_beta)
    as in Nalisnick & Smyth (2017) (12)
    - we require you to calculate the log of beta function, since that's a fixed quantity
    """
    digamma = Digamma.apply
    # digamma = b.log() - 1/(2. * b) - 1./(12 * b.pow(2)) # this doesn't seem to work
    first_term = ((a - prior_alpha)/(a+SMALL)) * (-1 * EULER_GAMMA - digamma(b.view(-1, 1)).view(b.size()) - 1./(b+SMALL))
    second_term = (a+SMALL).log() + (b+SMALL).log() + log_beta_prior
    third_term = -(b - 1)/(b+SMALL)

    sum_term = Variable(torch.cuda.DoubleTensor(a.size()).zero_())


    # we should figure out if this is enough
    for i in range(1, num_terms+1):
        beta_ = Beta.apply
        sum_term += beta_(float(i)/(a.view(-1, 1) + SMALL), b.view(-1, 1)).view(a.size())/(i + a * b)

    return (first_term + second_term + third_term + (prior_beta - 1) * b * sum_term)



def log_density_expconcrete(logalphas, logsample, temp):
    """
    log-density of the ExpConcrete distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalphas + logsample.mul(-temp)
    log_prob = exp_term + np.log(temp) - 2. * F.softplus(exp_term)
    return log_prob

# here, logsample is an instance of the ExpConcrete distribution, i.e. a y in the paper
def kl_discrete(logit_posterior, logit_prior, logsample, temp, temp_prior):
    """
    KL divergence between the prior and posterior
    inputs are in logit-space
    """
    logprior = log_density_expconcrete(logit_prior, logsample, temp_prior)
    logposterior = log_density_expconcrete(logit_posterior, logsample, temp)
    kl = logposterior - logprior
    #print(logposterior)
    #print(logprior)
    return kl


def kumaraswamy_sample(a, b):
    u = a.data.clone().uniform_(0.001, 0.999)
    u = Variable(u, requires_grad=False)
    # return (1. - u.pow(1./b)).pow(1./a)
    return (1. - u.log().div(b+SMALL).exp() + SMALL).log().div(a+SMALL).exp()


def reparametrize(a, b, ibp=False, log=False):
    v = kumaraswamy_sample(a, b)
    batch_size = a.size()[0]
    cuda = v.is_cuda
    if cuda:
        newTensor = torch.cuda.DoubleTensor
    else:
        newTensor = torch.DoubleTensor

    if ibp:
        # IBP: no need to sum to 1
        v_term = (v+SMALL).log()
        logpis = torch.cumsum(v_term, dim=1)
    else:
        # offset the vs
        v_term = torch.cat([(v+SMALL).log(), Variable(newTensor(batch_size).view(-1, 1).zero_(), requires_grad=False)], 1)

        # offset the 1 - vs
        inv_term = torch.cumsum(torch.cat([Variable(newTensor(batch_size).view(-1, 1).zero_(), requires_grad=False), (1. - v + SMALL).log()], 1), dim=1)
        logpis = v_term + inv_term

    if log:
        return logpis
    else:
        return logpis.exp()


def reparametrize_discrete(logalphas, temp):
    """
    input:  logit, output: logit
    """
    uniform = Variable(logalphas.data.clone().uniform_(1e-4, 1. - 1e-4),  requires_grad = False)
    logistic = torch.log(uniform) - torch.log(1. - uniform)
    logsample = (logalphas + logistic) / temp
    return logsample