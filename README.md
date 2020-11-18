# Affective Gaussians
Emotions visualisation using Gaussian word embeddings. The goal is to load some existing models involving gaussian representation of words in the literature.

---
## Gaussian mixture model
This is an adaptation of [word2gm](https://github.com/benathi/word2gm). All details including the original paper with technical details and BibTex enrty for citations can be found there. The original code allows to train on an arbitrary set. However, it was built for old versions of python and tensorflow. Here we only focus on their pretrained distributions, which can be found [here](https://bens-embeddings.s3-us-west-2.amazonaws.com/embeddings_project/word2gm_d50/w2gm-k2-d50.tar.gz). I expect to adapt the code so that the show_nearest_neighbors function works using the notebook, but that has not been done yet.

The instructions bellow allow for visualisation using TensorBoard. It requires a python virtual environment (version 3.7.9):
1. Install tensorflow version 1.15, latest version didn't work because of some deprecated calls.

`pip install tensorflow==1.15.4`

2. [The trained model](https://bens-embeddings.s3-us-west-2.amazonaws.com/embeddings_project/word2gm_d50/w2gm-k2-d50.tar.gz) has to be downloaded to: **word2gm/modelfiles/w2gm-k2-d50**

3. Run the first five cells in Analyze Model.ipynb. The rest of the cells show results and examples from the author, but running them will raise an error.

After a few seconds, the embeddings should appear on the corresponding port and displayed within the notebook.

---
## Bayesian Skip-gram
Similarly as with word2gm, the code is adapted from the [BSG repository](https://github.com/abrazinskas/BSG), which also contains the link to the original paper and the citation information. For the moment, we only focus on the execution of [pre-trained representations](https://drive.google.com/file/d/1YQQHFV215YjKLlvxpxsKWLm__TlQMw1Q/view) provided by the author. 

The notebook allows for comparing words using Kullback-Leibler divergence, which is not possible when using euclidean word embeddings. The asymmetric behaviour of the KL-divergence allows for the entailement prediction (see examples in the notebook). 

The only requirement is numpy. All the functions are defined in eval_support.py which is directly taken from the original repository. The [pre-trained parameters](https://drive.google.com/file/d/1YQQHFV215YjKLlvxpxsKWLm__TlQMw1Q/view) of the gaussian vectors must be downloaded to the folder **BSG/pretrained_vectors**

---
## Visualisation of Gaussians
In `Visualising Gaussian Embeddings.ipynb` we plot different emotional words using the information of their variance. Pre-trained Gaussian representations are taken from the [BSG repository](https://github.com/abrazinskas/BSG) and different dimensionality reductions are compared.

---
## Algorithm
Under construction