""" Pytorch implementation of BSG Model """
import torch
from encoder import Encoder
from torch.distributions import LowRankMultivariateNormal
from torch.distributions.kl import kl_divergence


class BSG(torch.nn.Module):
    """ Encoder from https://github.com/abrazinskas/BSG/blob/80089f9ec4302096ca6c81e79145ec5685c8d26e/models/bsg.py#L89"""
    def __init__(self, vocab, non_linearity, latent_dim, h_dim, cov_mat='diagonal', margin=1., subsampling_threshold=None):
        """
        # Arguments:
             h_dim: the number of components in the first hidden layers
             z_dim: the number of dimensions of the latent vectors
             margin: margin in the hinge loss
             vocab_size: amount of words in the vocabulary
        """
        super(BSG, self).__init__()
        self.h_dim = h_dim
        self.z_dim = latent_dim
        self.var_dim = self.variance_dimension(cov_mat,self.z_dim)

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.non_linearity = non_linearity
        self.initialise_layers()

        self.margin = margin
        self.subsampling_t = subsampling_threshold

    def initialise_layers(self):
        self.embeddings = torch.nn.Embedding(self.vocab_size,self.z_dim)
        self.input_dim = self.h_dim*2+self.var_dim
        self.encoder = Encoder(self.input_dim,self.h_dim,self.z_dim,self.var_dim,self.non_linearity)

    @staticmethod
    def variance_dimension(cov_mat,latent_dim):
        """Takes string "diagonal, "spherical or "full" and returns the number of dimensions"""
        assert isinstance(cov_mat, str)
        if cov_mat == 'diagonal':
            var_dim = 1
        elif cov_mat == 'spherical':
            var_dim = latent_dim
        elif cov_mat == 'full':
            #var_dim == latent_dim*latent_dim
            raise NotImplementedError("For using full cov matrices, KL div definition would need to be updated")
        else:
            raise ValueError('"cov_mat" must be "diagonal, "spherical or "full ')
        return var_dim

    def max_margin(self, mu_q, sigma_q, pos_context_words, neg_context_words):
        """ Computes a sum over context words margin(hinge loss).
        # Arguments:
            param pos_context_words:  a tensor with true context words ids [batch_size x window_size]
            param neg_context_words: a tensor with negative context words ids [batch_size x window_size]
        # Returns: 
            tensor [batch_size x 1]
        """
        raise NotImplementedError

    def load_pretrained(self, file_path):
        """Loads params from .pt file"""
        self.load_state_dict(torch.load(file_path))


def kl_div(mu_q, sigma_q, mu_p, sigma_p):
    """
    Kullback Leibler divergence between two Gaussians
    See: https://pytorch.org/docs/stable/distributions.html
    :return: tensor [batch_size x 1]  #CHECK
    """
    q = LowRankMultivariateNormal(mu_q,torch.tensor([[0.],[0.]]),sigma_q*torch.ones(2))
    p = LowRankMultivariateNormal(mu_p,torch.tensor([[0.],[0.]]),sigma_p*torch.ones(2))

    return kl_divergence(q,p)
