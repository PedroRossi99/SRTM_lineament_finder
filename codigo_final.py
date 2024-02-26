#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lib
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage


# # Imagem SRTM

# In[2]:


srtm = skimage.io.imread('srtm.tif')
print(f'{srtm.shape=}')
fig, ax = plt.subplots(1, 2, width_ratios=(2,1))
_ = ax[0].imshow(srtm), ax[1].hist(srtm.flatten(), bins=256)


# In[3]:


srtm = (srtm - srtm.min()) / (srtm.max() - srtm.min())


# # Lineamentos anotados

# In[4]:


lines = skimage.io.imread('lineamentos.tif')
lines = 1 - lines # 0: fundo, 1: linhas
plt.imshow(lines)
lines.shape, type(lines)


# Adicionando canal único aos shapes das imagens por conformidade com pytorch:

# In[5]:


srtm = srtm[:,:,np.newaxis]
lines = lines[:,:,np.newaxis]

print(f'{srtm.shape=}, {lines.shape=}') 


# # Visualização de exemplos

# In[6]:


lib.view_random_sample(srtm, lines, size=128)


# In[7]:


batch_size = 32
tile_size = 128
learning_rate = 0.0004
weight_decay = 1e-12
num_epochs = 50
first_hidden_channels = 16
depth=5
model_name= "modelo_30_srtm"


# # Dados
# 
# Crie os conjuntos de treinamento e validação:
# 
# ```train_dataloader = lib.load_examples(imagem[:,:], mascara[:,:], tile_size=128, stride=32, batch_size=batch_size, shuffle=True)```
# 
# ```val_dataloader = lib.load_examples(srtm[:,16:], lines[:,16:], tile_size=128, stride=64, batch_size=batch_size, shuffle=True)```

# In[8]:


#train_dataloader = lib.load_examples(srtm[:,750], lines[:,750], tile_size=128, stride=8, batch_size=batch_size, shuffle=True)


# In[9]:


#val_dataloader = lib.load_examples(srtm[:,8:], lines[:,8:], tile_size=128, stride=11, batch_size=batch_size, shuffle=True)


# In[10]:


#len(train_dataloader.dataset), len(val_dataloader.dataset)


# In[11]:


#Treinando por faixas 


# In[12]:


train_dataloader = lib.load_examples(srtm[1125:,:], lines[1125:,:], tile_size=128, stride=15, batch_size=batch_size, shuffle=True)


# In[13]:


val_dataloader = lib.load_examples(srtm[1125:,:], lines[1125:,:], tile_size=128, stride=23, batch_size=batch_size, shuffle=True)


# In[14]:


len(train_dataloader.dataset), len(val_dataloader.dataset)


# # Modelo
# 
# Arquitetura U-Net:

# In[7]:


model = lib.UNet(in_channels=1, out_channels=1, depth=depth, first_hidden_channels=first_hidden_channels).to('cuda')
model


# Função de custo:

# In[16]:


criterion = lib.DiceLoss()
criterion


# Otimizador:

# In[17]:


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer


# # Treinamento

# In[8]:


model = torch.load ("modelo_6")


# In[10]:


metrics = lib.train_and_validate(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer)


# Visualizando métricas: 

# In[9]:


lib.view_metrics(*metrics)


# # Inferência

# In[6]:


from torchvision import transforms


# In[11]:


model = torch.load("modelo_9")


# In[12]:


pred_jp = lib.predict(srtm, model, tile_size=128, stride=32)

fig, ax = plt.subplots(1,2,sharey=True,figsize=(10,10))
_ = ax[0].imshow(pred_jp), ax[1].imshow(lines)


# In[35]:


skimage.io.imsave ("pred_jpcuvat30.tif", pred_jp)


# # Teste Cego - Área Pedro

# In[13]:


srtm_pedro = skimage.io.imread('SRTM_pedro.tif')
srtm_pedro.shape


# In[14]:


print(f'{srtm_pedro.shape=}')
fig, ax = plt.subplots(1, 2, width_ratios=(2,1))
_ = ax[0].imshow(srtm_pedro), ax[1].hist(srtm_pedro.flatten(), bins=256)


# Normalização min-max:

# In[15]:


srtm_pedro = (srtm_pedro - srtm_pedro.min()) / (srtm_pedro.max() - srtm_pedro.min())


# Canal adicional por compatibilidade com pytorch

# In[16]:


srtm_pedro = srtm_pedro[:,:,np.newaxis]


# Predição

# In[17]:


pred = lib.predict(srtm_pedro, model, tile_size=128, stride=32)


# In[34]:


pred2 = pred**3
threshold = np.percentile(pred2, 70)
pred2_bin = np.where(pred2 > threshold, 1.0, 0.0)


# In[18]:


fig, ax = plt.subplots(1,3,sharey=True,figsize=(10,10))
_ = ax[0].imshow(pred)
_ = ax[1].imshow(pred2)
_ = ax[2].imshow(pred2_bin)


# In[32]:


torch.save(model,model_name)


# In[19]:


skimage.io.imsave ("pred_pr_9.tif", pred)

