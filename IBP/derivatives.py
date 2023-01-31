import torch

class Beta(torch.autograd.Function):

    @staticmethod
    def forward(self,a,b):
        tmp = (a.lgamma()+b.lgamma() - (a+b).lgamma()).exp()
        self.save_for_backward(a,b,tmp)
        return tmp

    @staticmethod
    def backward(self, grad_output):
        a,b,beta_ab = self.saved_tensors
        tmp_a = grad_output*beta_ab*(torch.polygamma(0,a) - torch.polygamma(0,(a+b)))
        tmp_b = grad_output*beta_ab*(torch.polygamma(0,b) - torch.polygamma(0,(a+b)))
        return tmp_a,tmp_b

class Digamma(torch.autograd.Function):

    @staticmethod
    def forward(self,input):
        self.save_for_backward(input)
        return torch.digamma(input)
    
    @staticmethod
    def backward(self,grad_output):
        input, = self.saved_tensors
        return grad_output*torch.polygamma(1,input)


class Log_gamma(torch.autograd.Function):
    @staticmethod
    def forward(self,x):
        tmp = x.lgamma()
        self.save_for_backward(x)
        return tmp
    @staticmethod
    def backward(self, grad_output):
        x, = self.saved_tensors
        tmp = grad_output*torch.digamma(x)
        return tmp
