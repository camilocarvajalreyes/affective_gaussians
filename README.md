# Affective Gaussians
Emotions visualisation using Gaussian word embeddings. The goal is to load some existing models involving gaussian representation of words in the literature.

---
## Gaussian mixture model
This is an adaptation of [word2gm](https://github.com/benathi/word2gm). All details including the original paper with technical details and BibTex enrty for citations can be found there. The original code allows to train on an arbitrary set. However, it was built for old versions of python and tensorflow. Here we only focus on their pretrained, which can be found [here](https://bens-embeddings.s3-us-west-2.amazonaws.com/embeddings_project/word2gm_d50/w2gm-k2-d50.tar.gz). I expect to adapt the code so that the show_nearest_neighbors function works using the notebook, but that has not been done yet.

The instructions bellow allow for visualisation using TensorBoard. It requires a python virtual environment (version 3.7.9):
1. Install tensorflow version 1.15, latest version didn't work because of some deprecated calls.

`pip install tensorflow==1.15.4`

2. [The trained model](https://bens-embeddings.s3-us-west-2.amazonaws.com/embeddings_project/word2gm_d50/w2gm-k2-d50.tar.gz) has to be downloaded to: **modelfiles/w2gm-k2-d50**

3. Run the first four cells in Analyze Model.ipynb. The rest of the cells show results and examples from the author, but running them will raise an error.

4. Finally, the following command has to be executed from the command line:

`tensorboard --logdir=word2gm/modelfiles/w2gm-k2-d50_emb --port=8889`

After a few seconds, the embeddings should appear on the corresponding port.
