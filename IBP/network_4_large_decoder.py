from statistics import mean
import torch
import torchvision
import numpy as np
import scipy.special as sc
import math
from derivatives import Beta,Log_gamma
from common import SMALL, reparametrize_discrete, reparametrize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IBP_encoder(torch.nn.Module):
    '''
    Probabilistic encoder of the network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
    '''

    def __init__(self,K,n_x,network_type):
        super(IBP_encoder,self).__init__()

        self.K = K
        self.n_x = n_x
        self.network_type = network_type

        if self.network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,2*self.K))
            self.f_theta = torch.nn.Sequential(*layers)       
        elif self.network_type == 'mlp_manifold':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,(2*self.K)))
            self.f_theta = torch.nn.Sequential(*layers)       

        elif self.network_type == 'mlp_CIFAR10':
            #layers = []
            #layers.append(torch.nn.Linear(self.n_x,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,2*self.K))
            #layers.append(torch.nn.Conv2d(self.n_x[0],64,3,1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(64,128,3,1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(128,256,3,1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(256,256,3,1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(256,512,3,1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(512,512,3,1))
            
            model = torchvision.models.vgg16()
            self.f_theta_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            #layers = []
            #layers.append(torch.nn.Linear(204800,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,512))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(512,self.K*2))
            #self.f_theta_lin = torch.nn.Sequential(*layers)       
            self.f_theta_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 2*self.K)
                #torch.nn.ReLU(True),
                #torch.nn.Dropout(p=dropout),
                #torch.nn.Linear(4096, 4096),
                #torch.nn.ReLU(True),
                #torch.nn.Dropout(p=dropout),
                #torch.nn.Linear(4096, 2*self.K),
            )

        elif self.network_type == 'mlp_ImageNet':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,1024))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(1024,1024))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(1024,(2*self.K)))
            self.f_theta = torch.nn.Sequential(*layers)       
        elif self.network_type == 'conv_net_fashion_mnist':
            layers = []
            layers.append(torch.nn.ReflectionPad2d(1))
            layers.append(torch.nn.Conv2d(1,32,5,2))
            layers.append(torch.nn.ReLU6())
            layers.append(torch.nn.Conv2d(32,128,5,2))
            self.f_theta_conv = torch.nn.Sequential(*layers)
            
            layers = []
            layers.append(torch.nn.Linear(128*5*5,128))
            layers.append(torch.nn.ReLU6())
            layers.append(torch.nn.Linear(128,2*self.K))
            self.f_theta_lin = torch.nn.Sequential(*layers)

        elif self.network_type == 'mlp_california_housing':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,128))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128,128))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128,self.K))
            self.f_theta = torch.nn.Sequential(*layers)
        
        elif self.network_type == 'conv_net_trec':
            '''
            Network type largely inspired on 'https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb'
            '''
            self.embedding = torch.nn.Embedding(n_x,100)
            self.convolutions = torch.nn.ModuleList([
                torch.nn.Conv2d(1,100,(fs,100)) for fs in [2,3,4]
            ])
            self.f_theta_lin = torch.nn.Linear(3*100,self.K)

    def forward(self,x):

        if self.network_type == 'mlp_mnist' or self.network_type == 'mlp_california_housing' or self.network_type == 'mlp_ImageNet' or self.network_type == 'mlp_manifold':
            x = x.view(-1,self.n_x)
            mean_t = self.f_theta(x)
        elif self.network_type == 'mlp_CIFAR10':
            mean_t_conv = self.f_theta_conv(x)
            mean_t_conv = mean_t_conv.flatten(1)
            mean_t = self.f_theta_lin(mean_t_conv)
        elif self.network_type == 'conv_net_fashion_mnist':
            mean_t_conv = self.f_theta_conv(x) 
            mean_t_conv = mean_t_conv.view(-1,5*5*128)
            mean_t = self.f_theta_lin(mean_t_conv)
        elif self.network_type == 'conv_net_trec':
            x = x.permute(1,0)
            z = self.embedding(x)
            z = z.unsqueeze(1)
            z_convs = [torch.nn.functional.relu6(convolution(z)).squeeze(3) for convolution in self.convolutions]
            z_pooled = [torch.nn.functional.max_pool1d(z_conv, z_conv.shape[2]).squeeze(2) for z_conv in z_convs]
            z_cat = torch.cat(z_pooled,dim=1)
            mean_t = self.f_theta_lin(z_cat)

        return mean_t


class Deterministic_decoder(torch.nn.Module):
    '''
    Deterministic decoder of the network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_y (int) : dimensionality of the output variable (number of classes)
    '''

    def __init__(self,K,n_y,network_type):
        super(Deterministic_decoder,self).__init__()

        self.K = K
        self.network_type = network_type

        if network_type == 'mlp_mnist' or network_type == 'conv_net_fashion_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.K,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'mlp_manifold':
            layers = []
            layers.append(torch.nn.Linear(self.K,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'mlp_CIFAR10' or network_type =='mlp_ImageNet':
            #layers = []
            #layers.append(torch.nn.Linear(self.K,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,n_y))
            #layers.append(torch.nn.Linear(self.K,512))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(512,1024))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(1024,512))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(512,n_y))
            #self.g_theta = torch.nn.Sequential(*layers)
            self.g_theta = torch.nn.Sequential(torch.nn.Linear(self.K,n_y))
        elif network_type == 'mlp_california_housing':
            layers = []
            layers.append(torch.nn.Linear(self.K,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'conv_net_trec':
            layers = []
            layers.append(torch.nn.Linear(self.K,n_y))
            self.g_theta = torch.nn.Sequential(*layers)


    def forward(self,t,gamma=1.0):

        logits_y =  self.g_theta(t*gamma)
        return logits_y

class nlIB_network(torch.nn.Module):
    '''
    Nonlinear Information Bottleneck network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
        · n_y (int) : dimensionality of the output variable (number of classes)
        · train_logvar_t (bool) : if true, logvar_t is trained
    '''

    def __init__(self,K,n_x,n_y,a_prior,b_prior,logvar_t=-1.0,train_logvar_t=False,network_type='mlp_mnist',method='nonlinear_IB',TEXT=None):
        super(nlIB_network,self).__init__()

        self.network_type = network_type
        self.method = method
        self.K = K
        def logit(p):
            return (p/(1-p+SMALL)+SMALL).log()
        
        self.logit = logit
        
        if self.method == 'IBP':
            self.encoder = IBP_encoder(K,n_x,self.network_type)
            #self.pi_encoder = Pi_encoder(K,n_x,self.network_type)
            a_val = np.log(np.exp(a_prior) - 1)
            b_val = np.log(np.exp(b_prior) - 1)
            self.prior_a = torch.tensor([a_val],requires_grad=False).to(dev)
            self.prior_b = torch.tensor([b_val],requires_grad=False).to(dev)
            self.a = torch.nn.Parameter(data=(torch.zeros(1,K) + a_val).to(dev),requires_grad=True)
            self.b = torch.nn.Parameter(data=(torch.zeros(1,K) + b_val).to(dev),requires_grad=True)
        self.decoder = Deterministic_decoder(K,n_y,self.network_type)

    def encode(self,x,random=True):

        if self.method == 'IBP':
            m = torch.nn.Softplus()
            s = torch.nn.Sigmoid()
            tmp1 = self.encoder(x) 
            mean_t = tmp1[:,0:self.K]
            sigma_t = m(tmp1[:,self.K:(tmp1.shape[1])])
            #print(mean_t.shape)
            #print(sigma_t.shape)
            #logit_x = tmp1[:,(tmp1.shape[1]-self.K):tmp1.shape[1]]
            a = m(self.a) + 0.01
            b = m(self.b) + 0.01
            #pi = self.pi_encoder(x)
            #pi1 = pi.clone()
            #pi1[pi1 == 0] = 1e-7
            #pi1[(1-pi1) == 0] = 1e-7 
            #pi = pi1
            #print(torch.isnan(pi).sum())
            #print((pi<0).sum())
            #print((pi==0).sum())
            #print((pi>1).sum())
            #u = torch.rand_like(a,requires_grad = False).to(dev)
            #v_s = (1-(1-u)**(1/b))**(1/a) + 1e-7   ### Kumaraswamy inverse CDF transformation
            #pi_s = (v_s.log().cumsum(dim=1).exp()).repeat(logit_x.shape[0],1) + 1e-7
            log_prior = reparametrize(self.prior_a.expand(mean_t.shape[0],self.K)
                                    ,self.prior_b.expand(mean_t.shape[0],self.K),ibp=True,log=True)
            #logit_post = logit_x #+ self.logit(log_prior.exp())
            #cat_pi = torch.cat((pi[:,:,None],(1-pi)[:,:,None]),dim = -1)
            #a_new = a*0 + 10.0
            #b_new = b*0 + 1.0
            log_post = reparametrize(a.view(1,self.K).expand(mean_t.shape[0],self.K)
                                   ,b.view(1,self.K).expand(mean_t.shape[0],self.K),ibp=True,log=True)
            #log_post = reparametrize(a_new.view(1,self.K).expand(mean_t.shape[0],self.K)
            #                       ,b_new.view(1,self.K).expand(mean_t.shape[0],self.K),ibp=True,log=True)
            #log_post = log_prior
            pi = self.logit(log_post.exp())
            pi_s = log_prior


        if random:
            if self.method == 'IBP':
                t = mean_t.repeat(10,1,1) + sigma_t.repeat(10,1,1) * torch.randn_like(mean_t.repeat(10,1,1)).to(dev)
                #tmp = torch.nn.functional.gumbel_softmax(torch.log(cat_pi.repeat(10,1,1,1)),tau=0.1,hard=False)
                #tmp1 = tmp[:,:,:,0]
                #gamma = tmp1
                logsample = reparametrize_discrete(pi.repeat(10,1,1),0.1)
                gamma = s(logsample)

        else:
            if self.method == 'IBP':
                t = mean_t
                #tmp = torch.nn.functional.gumbel_softmax(torch.log(cat_pi),tau=0.1,hard=False)
                #tmp1 = tmp[:,:,0]
                #gamma = tmp1
                logsample = reparametrize_discrete(pi,0.1)
                gamma = s(logsample)

        return t,gamma,pi,sigma_t,a,b,mean_t,pi_s

    def decode(self,t,gamma=1):

        logits_y = self.decoder(t,gamma)
        return logits_y

    def forward(self,x):

        t,gamma,_,_,_,_,_,_ = self.encode(x)
        logits_y = self.decode(t,gamma)
        return logits_y
