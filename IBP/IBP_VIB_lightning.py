import torch
import math
import autograd
import scipy.optimize
import scipy.special as sc
from scipy.stats import pearsonr
from progressbar import progressbar
import numpy as np
import random
import torchvision
from network_4_large_decoder import nlIB_network
from derivatives import Beta
from derivatives import Digamma
from common import SMALL, kl_divergence, kl_discrete
import gc
import os
import psutil
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import pandas as pd
import glob
from utils import get_args, get_data

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_ITY(logits_y,y):
    if problem_type == 'classification':
        tmp = logits_y.clone()           
        tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
        y = y.repeat(logits_y.shape[0])            
        logits_y = tmp
        HY_given_T = torch.nn.CrossEntropyLoss()(logits_y,y)
        ITY = (HY - HY_given_T) / np.log(2) # in bits
        return ITY
    else:
        tmp = logits_y.clone()           
        tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
        y = y.repeat(logits_y.shape[0],1)            
        logits_y = tmp
        MSE = torch.nn.MSELoss()(logits_y,y)
        ITY = 0.5 * torch.log(varY / MSE) / np.log(2) # in bits
        return ITY , (HY - MSE) / np.log(2) # in bits

def evaluate(logits_y,y):
    with torch.no_grad():
        if problem_type == 'classification':
            tmp = logits_y.clone()
            tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
            y = y.repeat(logits_y.shape[0])                
            logits_y = tmp
            y_hat = y.eq(torch.max(logits_y,dim=1)[1])
            accuracy = torch.mean(y_hat.float())
            return accuracy
        else: 
            tmp = logits_y.clone()
            tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
            y = y.repeat(logits_y.shape[0],1)                
            logits_y = tmp
            mse = torch.nn.MSELoss()(logits_y,y) 
            return mse 

def pi_prior(K,a,b):
    if method == "variational_IB":
        def fun(i):
            i = i.item()
            comb_term = sc.loggamma((K-1+1)) - (sc.loggamma(i+1) + sc.loggamma((K-(i+1)+1)))
            tmp = comb_term + np.log(sc.beta((a+i),(b+K-(i+1)))) - np.log(sc.beta(a,b)) ## i = k-1
            return np.exp(tmp)
            
        A = np.arange(K,dtype=int)
        A = np.reshape(A,(1,len(A)))
        B = np.apply_along_axis(fun,0,A)
        B = B/np.sum(B)
        B[B==0] = 1/1e7
        prior = torch.from_numpy(B)
    else:
        prior = torch.tensor(1)
    return prior.to(dev)


def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)



class CP_IB(pl.LightningModule):
    def __init__(self,repl_n,HY,n_x,n_y,a,b,compression_level,u_func,method,K,beta,dim_pen,network_type,dataset_name,prior,problem_type,learning_rate,learning_rate_steps,learning_rate_drop,datasetsize):
        super().__init__()
        self.repl_n = repl_n
        torch.manual_seed(self.repl_n)
        self.HY = HY # in natts
        #self.maxIXY = self.HY # in natts
        self.varY = 0 # to be updated with the training dataset
        self.IXT = 0 # to be updated
        self.ITY = 0 # to be
        self.n_x = n_x
        self.a = a
        self.b = b
        self.compression_level = compression_level
        self.u_func = u_func
        self.method = method        
        self.K = K
        self.beta = beta
        self.dim_pen = dim_pen         
        self.network = nlIB_network(K,n_x,n_y,a,b,network_type=network_type,method = self.method).cuda()
        if dataset_name == 'ImageNet':
            xavier_init(self.network)
        self.dataset_name = dataset_name
        self.prior = prior
        self.problem_type = problem_type
        self.automatic_optimization = False
        self.learning_rate=learning_rate
        self.learning_rate_drop = learning_rate_drop 
        self.learning_rate_steps = learning_rate_steps
        self.datasetsize = datasetsize
    def forward(self, x):
        logits_y = self.network(x)
        return logits_y

    def logit(self,p):
        return (p/(1-p+SMALL)+SMALL).log()

    def kl(self, x, a, b, logsample, z_discrete, logit_post, log_prior, mu, logvar, dataset_size, test=False):
        batch_size = x.size()[0]
        KL_zreal = -0.5 * (1. + logvar - mu**2 - logvar.exp())
        KL_beta = kl_divergence(a, b, prior_alpha=self.a, prior_beta= self.b, log_beta_prior=np.log(1./self.a)).repeat(batch_size, 1) * (1. / dataset_size)
        # in test mode, our samples are essentially coming from a Bernoulli
        if not test:
            KL_discrete = kl_discrete(logit_post, self.logit(log_prior.exp()), logsample, 0.01, 0.01)
        else:
            pi_prior = torch.exp(log_prior)
            pi_posterior = torch.sigmoid(logit_post)
            kl_1 = z_discrete * (pi_posterior + SMALL).log() + (1 - z_discrete) * (1 - pi_posterior + SMALL).log()
            kl_2 = z_discrete * (pi_prior + SMALL).log() + (1 - z_discrete) * (1 - pi_prior + SMALL).log()
            KL_discrete = kl_1 - kl_2
        return  KL_zreal, KL_beta, KL_discrete


    def get_IXT(self,t,mean_t,sigma_t,gamma,pi_s=1.,a_x=1.,b_x=1.,pi=1.,pi_prior = 1.,dim_pen=1.0,datasetsize=10000,Test=True):
        if method == 'IBP':
            logsample = self.logit(gamma)
            z_discrete = gamma
            logit_post = pi
            log_prior = pi_s
            IXT1, IXT2, IXT3 = self.kl(t, a_x, b_x, logsample, z_discrete, logit_post, log_prior, mean_t, 2*(sigma_t+SMALL).log(), dataset_size=datasetsize, test=Test)
            IXT_1 = IXT1.sum(1).mean().div(math.log(2))
            IXT_2 = IXT2.sum(1).mean().div(math.log(2)) + IXT3.sum(1).mean().div(math.log(2))
            IXT = IXT_1 + IXT_2
            #print(IXT_1)
            #print(IXT2.sum(1).mean().div(math.log(2)))
            #print(IXT3.sum(1).mean().div(math.log(2)))
        # NaNs and exploding gradients control
        with torch.no_grad():
            if u_func_name == 'shifted-exp':
                if IXT > compression_level:
                    IXT -= (IXT - compression_level - 0.01)
            if u_func(torch.Tensor([IXT])) == float('inf'):
                IXT = torch.Tensor([1e5])
        return IXT_1.to(dev),IXT_2.to(dev), IXT.to(dev)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()
        sgd_train_logits_y = self.network(x)
        if self.problem_type =='classification':
            sgd_train_ITY = get_ITY(sgd_train_logits_y,y)
        else:
            sgd_train_ITY, sgd_train_ITY_lower = get_ITY(sgd_train_logits_y,y)
        mi_train_t,mi_train_gamma,mi_train_pi,mi_sigma_t,mi_train_a,mi_train_b,mi_train_mean_t,mi_train_pi_s = self.network.encode(x,random=False)
        #print('input: {}'.format(x))
        #print('a: {}'.format(mi_train_a))
        #print('pi: {}'.format(mi_train_pi))
        #print('b: {}'.format(mi_train_b))
        mi_train_IXT_1,mi_train_IXT_2,mi_train_IXT = self.get_IXT(mi_train_t,mi_train_mean_t,mi_sigma_t,mi_train_gamma,mi_train_pi_s,mi_train_a,mi_train_b,mi_train_pi,self.prior,dim_pen=self.dim_pen,datasetsize=self.datasetsize)
        #print('IXT: {}'.format(mi_train_IXT))
        #print('IXT_1: {}'.format(mi_train_IXT_1))
        #print('IXT_2: {}'.format(mi_train_IXT_2))
        if self.problem_type == 'classification':
            loss = - 1.0 * (sgd_train_ITY - self.beta * self.u_func(mi_train_IXT))
        else: 
            loss = - 1.0 * (sgd_train_ITY_lower - self.beta * self.u_func(mi_train_IXT))
        #print("parameter is nan: {}".format([list(self.network.encoder.parameters())[i].isnan().sum().item() for i in range(len(list(self.network.encoder.parameters())))]))
        self.manual_backward(loss)
        #print(['####### id {}: {}'.format(id,p.grad) for id,p in enumerate(list(self.network.encoder.parameters()))])
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        opt.step()
        #print(self.trainer.is_last_batch)
        #print((self.trainer.current_epoch + 1) % 2)
        #print('lr : {}'.format(sch.get_lr()))        
        self.log("train_loss", loss,on_epoch=True)
        self.log("train_accuracy",evaluate(sgd_train_logits_y,y),on_epoch=True,on_step=False)
        self.log("train_IXT",(mi_train_IXT_1+mi_train_IXT_2),on_epoch=True,on_step=False)
        self.log("train_IXT_1",(mi_train_IXT_1),on_epoch=True,on_step=False)
        self.log("train_IXT_2",(mi_train_IXT_2),on_epoch=True,on_step=False)
        self.log("train_ITY",(sgd_train_ITY),on_epoch=True,on_step=False)
        #self.log("train_a",mi_train_a,on_epoch=True,on_step=True)
        #self.log("train_b",mi_train_b,on_epoch=True,on_step=True)
        return loss
    def training_epoch_end(self,outputs):
        sch = self.lr_schedulers()
        if (self.trainer.current_epoch + 1) % self.learning_rate_steps == 0:
            print("###########################")
            print('lr : {}'.format(sch.get_last_lr()[0]))
            sch.step()
            print('lr : {}'.format(sch.get_last_lr()[0]))

    def validation_step(self,val_batch,batch_idx):
        x, y = val_batch
        sgd_valid_logits_y = self.network(x)
        if self.problem_type =='classification':
            sgd_valid_ITY = get_ITY(sgd_valid_logits_y,y)
        else:
            sgd_valid_ITY, sgd_valid_ITY_lower = get_ITY(sgd_valid_logits_y,y)
        valid_t,valid_gamma,valid_pi,sigma_t,valid_a,valid_b,valid_mean_t,valid_pi_s = self.network.encode(x,random=False)
        valid_IXT_1,valid_IXT_2,valid_IXT = self.get_IXT(valid_t,valid_mean_t,sigma_t,valid_gamma,valid_pi_s,valid_a,valid_b,valid_pi,self.prior,dim_pen=self.dim_pen,datasetsize=self.datasetsize)
        if self.problem_type == 'classification':
            loss = - 1.0 * (sgd_valid_ITY - self.beta * self.u_func(valid_IXT))
        else: 
            loss = - 1.0 * (sgd_valid_ITY_lower - self.beta * self.u_func(valid_IXT))
        tmp = psutil.virtual_memory()
        print(f"Available:  {round(tmp.available/1024**3,2)}   Used:  {round(tmp.used/1024**3,2)}   Total  {round(tmp.total/1024**3,2)}   Shared:  {round(tmp.shared/1024**3,2)}",flush = True)
        # Logging to TensorBoard by default
        self.log("validation_ITY", sgd_valid_ITY,on_epoch=True,on_step=False)
        self.log("validation_IXT", (valid_IXT_1+valid_IXT_2),on_epoch=True,on_step=False)
        self.log("validation_IXT_1", (valid_IXT_1),on_epoch=True,on_step=False)
        self.log("validation_IXT_2", (valid_IXT_2),on_epoch=True,on_step=False)
        self.log("validation_loss", loss,on_epoch=True)
        tmp = torch.nn.functional.gumbel_softmax(torch.log(valid_pi.repeat(10,1,1)),tau=0.1,hard=True)
        mask = tmp.cumsum(dim=2)
        gamma = ((1 - mask) + tmp)
        mean_t_hat = (valid_mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
        sgd_valid_logits_y = self.network.decode(mean_t_hat)
        self.log("validation_accuracy",evaluate(sgd_valid_logits_y[None,:,:],y),on_epoch=True,on_step=False)
    def configure_optimizers(self):     
        if self.dataset_name == 'CIFAR10':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate,weight_decay=5e-4)
        elif self.dataset_name == 'ImageNet':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,betas=(0.5,0.999))
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
            step_size=1,gamma=self.learning_rate_drop)
        
        return {"optimizer":optimizer,"lr_scheduler":{"scheduler": learning_rate_scheduler,
                                                        "interval": "step",
                                                        "frequency": 1,
                                                        "strict": True,
                                                        "name": None    
                                                    }}

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.beta = args.repl_n%5 if args.repl_n%5 != 0 else 5
    args.repl_n = int((args.repl_n + 4.5)//5)
    print(args.beta,flush=True)
    print(args.repl_n,flush=True)
    epochs = args.n_epochs
    beta = np.linspace(0,1,50)[int(args.beta)]#[0.02,0.04,0.06,0.08,0.1][int(args.beta)-1]
    #beta = args.beta
    dim_pen = args.dim_pen
    u_func_name = args.u_func_name
    hyperparameter = args.hyperparameter
    compression_level = args.compression
    K =args.K
    a = args.a
    b = args.b
    batch_size = args.sgd_batch_size
    dataset_name = args.dataset
    if dataset_name == 'mnist':
        trainset, validationset = get_data(dataset_name)
        n_x = 784
        n_y = 10
        network_type = 'mlp_mnist'
        maxIXY = np.log2(10)
        HY = np.log(n_y)
        problem_type = 'classification'
        TEXT = None
        deterministic = True
    elif dataset_name == 'CIFAR10':
        trainset, validationset = get_data(dataset_name)
        n_x = (3,32,32)
        n_y = 10
        network_type = 'mlp_CIFAR10'
        maxIXY = np.log2(n_y)
        HY = np.log(n_y)
        problem_type = 'classification'
        TEXT = None
        deterministic = True
    elif dataset_name == 'dollar' or dataset_name == 's_in_a_box' or dataset_name == 's-and-plane':
        trainset, validationset = get_data(dataset_name)
        n_x = 100
        n_y = 2
        network_type = 'mlp_manifold'
        maxIXY = np.log2(n_y)
        HY = np.log(n_y)
        problem_type = 'classification'
        TEXT = None
        deterministic = True
    optimizer_name = args.optimizer_name
    method = args.method
    learning_rate = args.learning_rate
    learning_rate_drop = args.learning_rate_drop
    learning_rate_steps = args.learning_rate_steps
    if u_func_name == 'pow':
        u_func = lambda r: r ** (1+hyperparameter)
    elif u_func_name == 'exp':
        u_func = lambda r: torch.exp(hyperparameter*r)
    elif u_func_name == 'shifted-exp':
        u_func = lambda r: torch.exp((r-compression_level)*hyperparameter)*hyperparameter
    else:
        u_func = lambda r: r
    torch.manual_seed(args.repl_n)
    np.random.seed(args.repl_n)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    g = torch.Generator()
    g.manual_seed(args.repl_n)
    datasetsize = len(trainset)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,num_workers=25,worker_init_fn=seed_worker,generator=g, \
                                            shuffle=False, multiprocessing_context='fork')
    test_loader = torch.utils.data.DataLoader(validationset,batch_size=len(validationset),num_workers=5,shuffle=False, \
                                            multiprocessing_context='fork') 
    prior = pi_prior(K,a,b)

    if method == 'variational_IB':
        logs_dir = "../output/python/logs/" + dataset_name + "/" + u_func_name + '_' + str(round(hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_compound_train_8_no_dim_encoder" + '/'
        models_dir = "../output/python/models/" + dataset_name + "/" + u_func_name + '_' + str(round(hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_compound_train_8_no_dim_encoder" +'/'
    else:
        logs_dir = "../output/python/logs/" + dataset_name + "/" + u_func_name + '_' + str(round(hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_" + args.method + "_train_8" +  '/'
        models_dir = "../output/python/models/" + dataset_name + "/" + u_func_name + '_' + str(round(hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_" + args.method + "_train_8" + '/'

    os.makedirs(logs_dir) if not os.path.exists(logs_dir) else None
    os.makedirs(models_dir) if not os.path.exists(models_dir) else None
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dir+"lightning_logs_K_{}_B_{}_ep{}/".format(K,round(beta,3),epochs),name="model_{}".format(args.repl_n))
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir+"lightning_logs_K_{}_B_{}_ep{}/".format(K,round(beta,3),epochs), save_top_k=1, monitor="validation_loss",save_last=True)
    CP_IB_model = CP_IB(args.repl_n,HY,n_x,n_y,a,b,compression_level,u_func,method,K,beta,dim_pen,network_type,dataset_name,prior,problem_type,learning_rate,learning_rate_steps,learning_rate_drop,datasetsize)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[TQDMProgressBar(refresh_rate=20),checkpoint_callback],logger=tb_logger,track_grad_norm='inf',detect_anomaly=True)
    trainer.fit(model=CP_IB_model, train_dataloaders=train_loader,val_dataloaders=test_loader)

'''Usage:
nohup python3 IBP_VIB_lightning.py --n_epoch 100 --repl_n 1 --K 100 --a 2.0 --b 2.0 --u_func_name 'none' --dataset 's-and-plane' --method 'IBP' &> nohup1.out &
'''
