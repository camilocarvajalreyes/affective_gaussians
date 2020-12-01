""" Pytorch implementation of BSG Model """
import torch

"""
input_dim = 100
h_dim = 100  # the number of components in the first hidden layers
z_dim = 100  # the number of dimensions of the latent vectors

subsampling_threshold = None
nr_neg_samples = 10
margin = 5.0  # margin in the hinge loss"""

class BSG(torch.nn.Module):
    """ Encoder from https://github.com/abrazinskas/BSG/blob/80089f9ec4302096ca6c81e79145ec5685c8d26e/models/bsg.py#L89"""
    def __init__(self, ):
        super(BSG, self).__init__()
