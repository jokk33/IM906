#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np

import numpy.random as npr

from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

import matplotlib.cm as cm

from functools import reduce


# In[ ]:


class sample_GMM():
    #create modes with different labels
    def __init__(self, num_examples, means=None, variances=None, priors=None,**kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', 0)
            rng = np.random.RandomState(seed)
        gaussian_mixture = GMM_distribution(means=means,variances=variances,priors=priors,rng=rng)
        self.means = gaussian_mixture.means
        self.variances = gaussian_mixture.variances
        self.priors = gaussian_mixture.priors
        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)
        data ={'samples': features, 'label': labels, 'density': densities}
        self.data = data


# In[ ]:


class GMM_distribution(object):
    def __init__(self, means=None, variances=None, priors=None, rng=None, seed=None):
        if means is None:
            means = map(lambda x:  10.0 * np.array(x), [[0, 0],
                                                        [1, 1],
                                                        [-1, -1],
                                                        [1, -1],
                                                        [-1, 1]])
        # Number of components
        self.ncomponents = len(means)
        self.dim = means[0].shape[0]
        self.means = means
        # If prior is not specified let prior be flat.
        if priors is None:
            priors = [1.0/self.ncomponents for _ in range(self.ncomponents)]
        self.priors = priors
        # If variances are not specified let variances be identity
        if variances is None:
            variances = [np.eye(self.dim) for _ in range(self.ncomponents)]
        self.variances = variances
        assert len(means) == len(variances), "Mean variances mismatch"
        assert len(variances) == len(priors), "prior mismatch"
        if rng is None:
            rng = npr.RandomState(seed=seed)
        self.rng = rng
    def _sample_prior(self, nsamples):
        return self.rng.choice(a=self.ncomponents,
                               size=(nsamples, ),
                               replace=True,
                               p=self.priors)
    def sample(self, nsamples):
        # Sampling priors
        samples = []
        fathers = self._sample_prior(nsamples=nsamples).tolist()
        for father in fathers:
            samples.append(self._sample_gaussian(self.means[father],
                                                 self.variances[father]))
        return np.array(samples), np.array(fathers)
    def _sample_gaussian(self, mean, variance):
        # sampling unit gaussians
        epsilons = self.rng.normal(size=(self.dim, ))
        return mean + np.linalg.cholesky(variance).dot(epsilons)
    def _gaussian_pdf(self, x, mean, variance):
        return multivariate_normal.pdf(x, mean=mean, cov=variance)
    def pdf(self, x):
        "Evaluates the the probability density function at the given point x"
        pdfs = map(lambda m, v, p: p * self._gaussian_pdf(x, m, v),
                   self.means, self.variances, self.priors)
        return reduce(lambda x, y: x + y, pdfs, 0.0)


# In[ ]:


def plot_GMM(dataset, save_path):
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    ax = axes
    ax.set_aspect('equal')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.axis('on')
    ax.set_title('$\mathbf{x} \sim $GMM$(\mathbf{x})$')
    x = dataset.data['samples']
    targets = dataset.data['label']
    axes.scatter(x[:, 0], x[:, 1], marker='.', c=cm.Set1(targets.astype(float)/2.0/2.0) , alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, transparent=True, bbox_inches='tight')


# In[ ]:


from numpy.random import RandomState
from random import Random
seed = 42
py_rng = Random(seed)
np_rng = RandomState(seed)
def set_seed(n):
    global seed, py_rng, np_rng
    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)


# In[ ]:


import numpy as np
from sklearn import utils as skutils
def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]
def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)
def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = int(n / size)
    if n % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tqdm import tqdm
tf.reset_default_graph()
slim = tf.contrib.slim
ds = tf.contrib.distributions
graph_replace = tf.contrib.graph_editor.graph_replace
""" parameters """
n_epoch = 1000
batch_size  = 64
dataset_size = 512
input_dim = 2
latent_dim = 2
eps_dim = 2
n_layer_disc = 2
n_hidden_disc = 256
n_layer_gen = 2
n_hidden_gen= 256
n_layer_inf = 2
n_hidden_inf= 256
""" Create directory for results """
result_dir = 'results/DiscoGAN/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)


# In[ ]:


means = map(lambda x:  np.array(x), [[0, 0],
                                     [2, 2],
                                     [-3, -1],
                                     [1, -4],
                                     [-1, 4]])
means = list(means)
std = 0.1
variances = [np.eye(2) * std for _ in means]
priors = [1.0/len(means) for _ in means]
gaussian_mixture = GMM_distribution(means=means, variances=variances, priors=priors)
dataset = sample_GMM(dataset_size, means, variances, priors, sources=('features', ))
save_path = result_dir + 'X_gmm_data.pdf'
plot_GMM(dataset, save_path)
X_np_data= dataset.data['samples']
X_labels = dataset.data['label']


# In[ ]:


means = map(lambda x:  np.array(x), [[-1, -1],[1, 1],[-1,2]])
means = list(means)
std = 0.1
variances = [np.eye(2) * std for _ in means]
priors = [1.0/len(means) for _ in means]
gaussian_mixture = GMM_distribution(means=means,
                                               variances=variances,
                                               priors=priors)
dataset = sample_GMM(dataset_size, means, variances, priors, sources=('features', ))
save_path = result_dir + 'Z_gmm_data.pdf'
plot_GMM(dataset, save_path)
Z_np_data= dataset.data['samples']
Z_labels = dataset.data['label']


# In[ ]:


X_dataset = X_np_data
Z_dataset = Z_np_data
""" Networks """
def generative_network(z, input_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generative",reuse=tf.AUTO_REUSE):
        h = z
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, input_dim, activation_fn=None, scope="p_x")
    return x
def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("inference",reuse=tf.AUTO_REUSE ):
        h = x
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
    return z
def data_network_x(x, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate x log data density."""
    h = tf.concat(x, 1)
    with tf.variable_scope('discriminator_x',reuse=tf.AUTO_REUSE ):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])
def data_network_z(z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate z log data density."""
    h = tf.concat(z, 1)
    with tf.variable_scope('discriminator_z',reuse=tf.AUTO_REUSE):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])


# In[ ]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))
p_x = generative_network(z, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
q_z = inference_network(x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
decoder_logit_x = data_network_x(p_x, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_x = graph_replace(decoder_logit_x, {p_x: x})
decoder_logit_z = data_network_z(q_z, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_z = graph_replace(decoder_logit_z, {q_z: z})

encoder_sigmoid_x = tf.nn.softplus(encoder_logit_x)
decoder_sigmoid_x = tf.nn.softplus(decoder_logit_x)
encoder_sigmoid_z = tf.nn.softplus(encoder_logit_z)
decoder_sigmoid_z = tf.nn.softplus(decoder_logit_z)

decoder_loss = decoder_sigmoid_x + decoder_sigmoid_z
encoder_loss = encoder_sigmoid_x + encoder_sigmoid_z
# decoder_loss = decoder_logit_x + decoder_logit_z
# encoder_loss = encoder_logit_x + encoder_logit_z
disc_loss = tf.reduce_mean(  encoder_loss ) - tf.reduce_mean( decoder_loss)
rec_z = inference_network(p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim )
cost_z = tf.reduce_mean(tf.pow(rec_z - z, 2))
rec_x = generative_network(q_z, input_dim , n_layer_gen, n_hidden_gen,  eps_dim )
cost_x = tf.reduce_mean(tf.pow(rec_x - x, 2))
adv_loss = tf.reduce_mean(  decoder_loss ) # + tf.reduce_mean( encoder_loss )
gen_loss = 1*adv_loss + 1.*cost_x  + 1.*cost_z
qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")
dvars_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_z")
opt = tf.train.AdamOptimizer(1e-4, beta1=0.5)
train_gen_op =  opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars_x + dvars_z)


# In[ ]:


tf.InteractiveSession.close


# In[ ]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
FG = []
FD = []
for epoch in tqdm( range(n_epoch), total=n_epoch):
    X_dataset, Z_dataset= shuffle(X_dataset, Z_dataset)
    for xmb, zmb in iter_data(X_dataset, Z_dataset, size=batch_size):
        for _ in range(1):
            f_d, _ = sess.run([disc_loss, train_disc_op], feed_dict={x: xmb, z:zmb})
        for _ in range(5):
            f_g, _ = sess.run([[adv_loss, cost_x, cost_z], train_gen_op], feed_dict={x: xmb, z:zmb})
        FG.append(f_g)
        FD.append(f_d)


# In[ ]:


n_viz = 1
imz = np.array([]); rmz = np.array([]); imx = np.array([]); rmx = np.array([]);
for _ in range(n_viz):
    for xmb, zmb in iter_data(X_np_data, Z_np_data, size=batch_size):
        temp_imz = sess.run(q_z, feed_dict={x: xmb, z:zmb})
        imz = np.vstack([imz, temp_imz]) if imz.size else temp_imz
        temp_rmz = sess.run(rec_z, feed_dict={x: xmb, z:zmb})
        rmz = np.vstack([rmz, temp_rmz]) if rmz.size else temp_rmz
        temp_imx = sess.run(p_x, feed_dict={x: xmb, z:zmb})
        imx = np.vstack([imx, temp_imx]) if imx.size else temp_imx
        temp_rmx = sess.run(rec_x, feed_dict={x: xmb, z:zmb})
        rmx = np.vstack([rmx, temp_rmx]) if rmx.size else temp_rmx
## inferred marginal z
fig_mz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(X_labels, (n_viz))
ax.scatter(imz[:, 0], imz[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
        edgecolor='none', alpha=0.5)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_xlabel('$B_1$'); ax.set_ylabel('$B_2$')
ax.axis('on')
plt.savefig(result_dir + 'inferred_mz.pdf', transparent=True, bbox_inches='tight')
##  reconstruced z
fig_pz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(Z_labels, (n_viz))
ax.scatter(rmz[:, 0], rmz[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_xlabel('$B_1$'); ax.set_ylabel('$B_2$')
ax.axis('on')
plt.savefig(result_dir + 'reconstruct_mz.pdf', transparent=True, bbox_inches='tight')
## inferred marginal x
fig_pz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(Z_labels, (n_viz))
ax.scatter(imx[:, 0], imx[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
        edgecolor='none', alpha=0.5)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_xlabel('$A_1$'); ax.set_ylabel('$A_2$')
ax.axis('on')
plt.savefig(result_dir + 'inferred_mx.pdf', transparent=True, bbox_inches='tight')
##  reconstruced x
fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(X_labels, (n_viz))
ax.scatter(rmx[:, 0], rmx[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_xlabel('$A_1$'); ax.set_ylabel('$A_2$')
ax.axis('on')
plt.savefig(result_dir + 'reconstruct_mx.pdf', transparent=True, bbox_inches='tight')
## learning curves
fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.plot(FD, label="Discriminator")
ax.plot(np.array(FG)[:,0], label="Generator")
ax.plot(np.array(FG)[:,1], label="Reconstruction A")
ax.plot(np.array(FG)[:,2], label="Reconstruction B")
plt.xlabel('Iteration')
plt.ylabel('Loss')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.axis('on')
plt.savefig(result_dir + 'learning_curves.pdf', bbox_inches='tight')

