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


from __future__ import print_function, division
import numpy as np 
import pandas as pd 
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

from keras.layers.advanced_activations import LeakyReLU

import sys
sys.setrecursionlimit(10000)
get_ipython().system('pip install git+https://www.github.com/keras-team/keras-contrib.git')
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import sys
import os
from imageio import imread


# In[ ]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[ ]:


# load data
def load_data(dataset_name,batch_size=1,is_val=False):
    data_type = "train" if not is_val else "val"
    path = glob('../input/%s/%s/%s/*'%(dataset_name,dataset_name,data_type))
    batch_images = np.random.choice(path,size=batch_size)
    img_res = (128,128)
    imgs_A = []
    imgs_B = []
    for img_path in batch_images:
        img = imread(img_path)
        h,w,_ =img.shape
        half_w = int(w/2)
        img_A, img_B = img[:,half_w:,:], img[:,:half_w,:]
        img_A = transform.resize(img_A, img_res)
        img_B = transform.resize(img_B, img_res)
        if not is_val and np.random.random()>0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
        imgs_A.append(img_A)
        imgs_B.append(img_B)
    
    imgs_A = np.array(imgs_A)/127.5-1.
    imgs_B = np.array(imgs_B)/127.5-1.
    return imgs_A,imgs_B
    


# In[ ]:


import imageio
#load batch 
def load_batch(dataset_name,batch_size=1, is_val=False):
        data_type = "train" if not is_val else "val"
        path = glob('../input/%s/%s/%s/*' % (dataset_name,dataset_name, data_type))
        global n_batches
        n_batches=int(len(path)/batch_size)
        img_res=(128,128)
        for i in range(n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                # because in the edges2shoes and maps dataset the input image comes before the ground truth.
                if (dataset_name=="edges2shoes"or dataset_name=="maps"):
                      img_A, img_B = img[:, half_w:, :],img[:, :half_w, :] 
                else:  
                      img_A, img_B = img[:, :half_w, :], img[:, half_w:, :]
                img_A = transform.resize(img_A, img_res)#Ground truth image
                img_B = transform.resize(img_B, img_res)# input image
 # when training => do random flip , this is a trick to avoid overfitting 
                if not is_val and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            # normalizing the images 
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

def imread(path):
        return imageio.imread(path).astype(np.float)


# In[ ]:


def build_generator():
    #Taking advantage of U-net shape--good for sizes convolution 
    def conv2d(layer_input,filters,f_size=4,bn=True):
        #layers for downsampling
        d = Conv2D(filters,kernel_size=f_size,strides=2,padding="same")(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        #use leakyrelu as activation funcion
        if bn:
            d = InstanceNormalization()(d)
        return d
    def deconv2d(layer_input,skip_input,filters,f_size=4,dropout_rate=0):
        #layers for upsampling
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters,kernel_size=f_size,strides=1,padding='same',activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u,skip_input])
        #to skip connect
        return u
    
    d0 = Input(shape=img_shape)
    #downsampling layers
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1,gf*2)
    d3 = conv2d(d2,gf*4)
    d4 = conv2d(d3,gf*8)
    d5 = conv2d(d4,gf*8)
    d6 = conv2d(d5,gf*8)
    d7 = conv2d(d6,gf*8)
    #upsampling deconvolution layers
    u1 = deconv2d(d7,d6,gf*8)
    u2 = deconv2d(u1,d5,gf*8)
    u3 = deconv2d(u2,d4,gf*8)
    u4 = deconv2d(u3,d3,gf*4)
    u5 = deconv2d(u4,d2,gf*2)
    u6 = deconv2d(u5,d1,gf)
    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels,kernel_size=4,strides=1,padding='same',activation='tanh')(u7)
    return Model(d0,output_img)
        


# In[ ]:


def build_discriminator():
    def d_layer(layer_input,filters,f_size=4,bn=True):
        d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = InstanceNormalization()(d)
        return d
    
    img = Input(shape=img_shape)
    d1 = d_layer(img,df,bn=False)
    d2 = d_layer(d1,df*2)
    d3 = d_layer(d2,df*4)
    d4 = d_layer(d3,df*8)
    validity = Conv2D(1,kernel_size=4,strides=1,padding='same')(d4)
    return Model(img,validity)

        
        
    


# In[ ]:


def train(dataset_name,epochs, batch_size=128, sample_interval=50):
        start_time = datetime.datetime.now()
        global n_batches
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + disc_patch)
        fake = np.zeros((batch_size,) + disc_patch)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(dataset_name,batch_size)):
                #  Train Discriminators
                # Translate images to opposite domain
                fake_B = g_AB.predict(imgs_A)
                fake_A = g_BA.predict(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss_real = d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                # Train the generators
                g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, valid,                                                                          imgs_B, imgs_A,                                                                          imgs_A, imgs_B])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                if batch_i%200 == 0:
                    print ("[%d] [%d/%d] time: %s, [d_loss: %f, g_loss: %f]" % (epoch, batch_i,
                                                                            n_batches,
                                                                            elapsed_time,
                                                                            d_loss[0], g_loss[0]))
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    sample_images(dataset_name,epoch, batch_i)


# In[ ]:


def sample_images(dataset_name,epoch, batch_i):
    os.makedirs('images/%s' % dataset_name, exist_ok=True)
    r, c = 2, 3
    imgs_A, imgs_B = load_data(dataset_name,batch_size=1, is_val=True)
    # Translate images to the other domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)
    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.show()
    plt.close()


# In[ ]:


img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows,img_cols,channels)
dataset_name = 'edges2shoes'

patch = int(img_rows/ 2**4)
disc_patch = (patch,patch,1)
gf = 64
df = 64
optimizer = Adam (0.0002,0.5)
#discriminator
d_A = build_discriminator()
d_B = build_discriminator()
d_A.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
d_B.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
#two generators : input from both domains
g_AB = build_generator()
g_BA = build_generator()
img_A = Input(shape =img_shape)
img_B = Input(shape =img_shape)
#translate one to other domain
fake_B = g_AB(img_A)
fake_A = g_BA(img_B)
#translate back from fake AB
reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)
#set discriminators untrainable
d_A.trainable = False
d_B.trainable = False
#determine validity
valid_A = d_A(fake_A)
valid_B = d_B(fake_B)
combined = Model(inputs=[img_A,img_B],
                outputs=[valid_A,valid_B,fake_B,fake_A,reconstr_A,reconstr_B])
combined.compile(loss=['mse','mse','mae','mae','mae','mae'],optimizer=optimizer)


# In[ ]:


train("edges2shoes",epochs=20, batch_size=1, sample_interval=200)

