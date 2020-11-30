""" BSG encoder, based on Variational auto-encoder """
import torch

class Encoder(torch.nn.Module):
    """
    Encoder that is specific to the original BSG version from https://github.com/abrazinskas/BSG/blob/80089f9ec4/layers/custom/bsg_encoder.py. 
    It uses one input representation of words, and performs transformation of context and center word representations.
    """
    def __init__(self, input_dim, hidden_dim ,latent_dim, var_dim, non_linearity):
        super(Encoder, self).__init__()
        #self.embeddings line deleted, we'll use the same embeddings for the encoder instead of training a new set of them
        #input dimension will be 2*(mu_dim+sigma_dim), that way we use the uncertainty information from centre and context words
        #input dimension can be set to 
        self.linear_hidden = torch.nn.Linear(2*input_dim,hidden_dim)
        self.non_linearity = non_linearity
        self.linear_mu = torch.nn.Linear(hidden_dim,latent_dim)
        self.linear_sigma = torch.nn.Linear(hidden_dim,var_dim)

    def forward(self, emb_mu, emb_sigma):
        emb = torch.cat((emb_mu,emb_sigma), dim=0) #see: https://pytorch.org/docs/stable/generated/torch.cat.html
        h = self.non_linearity(self.linear_hidden(emb))
        mu_q = self.linear_mu(h)
        sigma_q = self.linear_sigma(h) # this really corresponds to log sigma squared
        return mu_q, sigma_q

