"""  BSG encoder, based on Variational auto-encoder """
import torch


class Encoder(torch.nn.Module):
    """
    Encoder that is specific to the original BSG version from https://github.com/abrazinskas/BSG/blob/80089f9ec4/layers/custom/bsg_encoder.py. 
    It uses one input representation of words, and performs transformation of context and center word representations.
    """
    def __init__(self, input_dim, hidden_dim ,latent_dim, var_dim, non_linearity):
        super(Encoder, self).__init__()
        #self.embeddings line deleted, we'll use the same embeddings for the encoder instead of training a new set of them
        #input dimension will be (2*mu_dim)+sigma_dim, that way we use the uncertainty information from centre and context words
        #input dimension can be set to 
        self.input_dim = input_dim
        self.linear_hidden = torch.nn.Linear(input_dim,hidden_dim)
        self.non_linearity = non_linearity
        self.linear_mu = torch.nn.Linear(hidden_dim,latent_dim)
        self.linear_sigma = torch.nn.Linear(hidden_dim,var_dim)

    def combine_centre_context(self, centre_emb, context_embs):
        """
        :centre_emb: tensor with representation of centre word
        :context_embs: tensor of shape (# of contexts,latent_dim,1) - mus of context words stacked
        :return: sum of contatenations between centre word and each of the context words
        """
        contexts = torch.unbind(context_embs,dim=1)
        all_contexts_centre = torch.empty(size=(context_embs.shape[1],1,self.input_dim))
        for i, context in enumerate(contexts):
            joint = torch.cat([centre_emb,context],dim=1)
            all_contexts_centre[i] = joint
        return all_contexts_centre

    def forward(self, centre_mu, centre_sigma, context_embs):
        """
        :centre_mu: tensor of shape (latent_dim,1) - mu of centre word
        :centre_sigma: tensor of shape (sigma_dim,1) - sigma of centre word
        :context_embs: tensor of shape (# of contexts,latent_dim,1) - mus of context words stacked
        """
        centre = torch.cat([centre_mu,centre_sigma],dim=1) #see: https://pytorch.org/docs/stable/generated/torch.cat.html
        all_centre_context = self.combine_centre_context(centre, context_embs)
        all_h = self.linear_hidden(all_centre_context)
        all_h = self.non_linearity(all_h)
        h = torch.sum(all_h, dim=0)
        mu_q = self.linear_mu(h)
        sigma_q = self.linear_sigma(h) # this really corresponds to log sigma squared
        return mu_q, torch.exp(sigma_q) 


#tests
"""
mu_centre = torch.tensor([[ 0.5,-0.3]]) 
sigma = torch.tensor([[0.1]])
centre = torch.cat([mu_centre,sigma],dim=1)
context = torch.tensor([[ -0.4,0.25]])
context2 = torch.tensor([[ 0.7,0.7]])
contexts = torch.stack([context,context2],dim=1)
separated_contexts = torch.unbind(contexts,dim=1)

dim = 2*mu_centre.shape[1]+sigma.shape[1]
latent_dim = mu_centre.shape[1]
print('input dim = '+str(dim))
print('latent and hidden dim = '+str(latent_dim))
nonLinearity = torch.nn.ReLU()
###########
encoder = Encoder(dim,latent_dim,latent_dim,sigma.shape[1],nonLinearity)
combined = encoder.combine_centre_context(centre,contexts)
print('combined')
print(combined)
print(combined.shape)
print('layer')
print(encoder.linear_hidden.weight)
print(encoder.linear_hidden.bias)
print(encoder.linear_hidden(combined))
print(encoder.linear_hidden(combined).shape)
print('summed')
print(torch.sum(combined,dim=0))
print(torch.sum(combined,dim=0).shape)
mu_q, sigma_q = encoder.forward(mu_centre,sigma,contexts)
print(mu_q)
print(sigma_q)
"""
