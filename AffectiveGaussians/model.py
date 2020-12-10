""" Pytorch implementation of BSG Model """
import torch
from AffectiveGaussians.encoder import Encoder
from torch.distributions import LowRankMultivariateNormal
from torch.distributions.kl import kl_divergence


class BSG(torch.nn.Module):
    """ Encoder from https://github.com/abrazinskas/BSG/blob/80089f9ec4302096ca6c81e79145ec5685c8d26e/models/bsg.py#L89"""
    def __init__(self, vocab, non_linearity, latent_dim, h_dim, window_size, cov_mat='spherical', margin=5., subsampling_threshold=None):
        """
        # Arguments:
            vocab: vocabulary object
            non_linearity: torch non-linearity
            latent_dim: the number of dimensions of the latent vectors
            h_dim: the number of components in the first hidden layers
            window_size: number of words for context windows
            cov_mat: str with type of covariance matrix ('spherical' or 'diagonal')
            margin: margin in the hinge loss
            subsampling_threshold: to handle frequent words
        """
        super(BSG, self).__init__()
        self.h_dim = h_dim
        self.z_dim = latent_dim
        self.var_dim = self.variance_dimension(cov_mat,self.z_dim)

        self.vocab = vocab
        self.vocab_size = 100 #len(vocab) changeeee

        self.non_linearity = non_linearity
        self.initialise_layers()

        self.margin = margin
        self.subsampling_t = subsampling_threshold

    def initialise_layers(self):
        self.mu_embeddings = torch.nn.Embedding(self.vocab_size,self.z_dim)
        self.log_sigma_embeddings = torch.nn.Embedding(self.vocab_size,self.var_dim)

        self.input_dim = self.h_dim*2+self.var_dim

        self.encoder = Encoder(self.input_dim,self.h_dim,self.z_dim,self.var_dim,self.non_linearity)

    def compute_prior(self, word_ids):
        """ Takes a tensor of shape batch_size x 1 with the word indices and returns tuples with mu and sigma """
        mu = self.mu_embeddings(word_ids)
        sigma = torch.exp(self.log_sigma_embeddings(word_ids))
        return mu, sigma

    @staticmethod
    def variance_dimension(cov_mat,latent_dim):
        """Takes string "diagonal, "spherical or "full" and returns the number of dimensions"""
        assert isinstance(cov_mat, str)
        if cov_mat == 'diagonal':
            var_dim = latent_dim
        elif cov_mat == 'spherical':
            var_dim = 1
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
            tensor [batch_size x 1] with hinge loss
        """
        b, window_size = pos_context_words.shape
        mu_q = mu_q.reshape(b,1,-1).repeat(1,window_size,1)
        sigma_q = sigma_q.reshape(b,1,-1).repeat(1,window_size,1)

        pos_mu_p, pos_sigma_p = self.compute_prior(pos_context_words)
        neg_mu_p, neg_sigma_p = self.compute_prior(neg_context_words)

        pos_kl = kl_div(mu_q,sigma_q,pos_mu_p,pos_sigma_p).reshape(b, window_size,1)
        neg_kl = kl_div(mu_q,sigma_q,neg_mu_p,neg_sigma_p).reshape(b,window_size,1)
        zero = torch.zeros(b, window_size,1)
        return torch.sum(torch.max(zero, self.margin + pos_kl-neg_kl),dim=1) 

    def load_pretrained(self, file_path):
        """Loads params from .pt file"""
        self.load_state_dict(torch.load(file_path))


def kl_div(mu_q, sigma_q, mu_p, sigma_p):
    """
    Kullback Leibler divergence between two Gaussians
    See: https://pytorch.org/docs/stable/distributions.html
    # Return: 
        tensor [batch_size x 1]  #CHECK
    """
    dim = mu_q.shape[-1]
    zero = torch.zeros(dim, 1)
    q = LowRankMultivariateNormal(mu_q, zero, sigma_q*torch.ones(dim))
    p = LowRankMultivariateNormal(mu_p, zero, sigma_p*torch.ones(dim))

    return kl_divergence(q,p)
