# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:01:51 2017

@author: avarfolomeev
"""

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.gridspec as gridspec

#display example of each class and show number of samples
cols = 6
figsize = (10, 20)

gs = gridspec.GridSpec(n_classes // cols + 1, cols)

fig1 = plt.figure(num=1, figsize=figsize)
ax = []
for i in range(n_classes):
    row = (i // cols)
    col = i % cols
    ax.append(fig1.add_subplot(gs[row, col]))
    ax[-1].set_title('class %d, N=%d' % (i ,  classCounts[i]))
    #example
    img = X_train[classIndicies[i][40]]
    #rescale to make dark images visible
    cf = np.int(255/np.max(img)) 
    ax[-1].imshow(img*cf)
    ax[-1].axis('off')
    
#%%

image = X_train[classIndicies[4][40]] #sl70    
#%%
figsize = (20, 30)
gs = gridspec.GridSpec(1, 3)

fig1 = plt.figure(num=2, figsize=figsize)
ax = []
ax.append(fig1.add_subplot(gs[0, 0]))
ax[-1].set_title('Original image')
ax[-1].imshow(img)
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 1]))
ax[-1].set_title('Motion blur 3')
ax[-1].imshow(img_m3)
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 2]))
ax[-1].set_title('Motion blur 5')
ax[-1].imshow(img_m5)
ax[-1].axis('off')
    
#%%
#rotations
figsize = (20, 10)
gs = gridspec.GridSpec(2, 3)

fig1 = plt.figure(num=2, figsize=figsize)
ax = []
ax.append(fig1.add_subplot(gs[0, 0]))
ax[-1].set_title('Z -15')
ax[-1].imshow(transformImg(img,0,0,-15))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[1, 0]))
ax[-1].set_title('Z +15')
ax[-1].imshow(transformImg(img,0,0,15))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 1]))
ax[-1].set_title('Y -20')
ax[-1].imshow(transformImg(img,0,-20,0))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[1, 1]))
ax[-1].set_title('Y +20')
ax[-1].imshow(transformImg(img,0,20,0))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 2]))
ax[-1].set_title('X +20')
ax[-1].imshow(transformImg(img,20,0,0))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[1, 2]))
ax[-1].set_title('X +20, Y +15, Z +15')
ax[-1].imshow(transformImg(img,20,15,15))
ax[-1].axis('off')

#%%
#scale
figsize = (10, 10)
gs = gridspec.GridSpec(1, 3)

fig1 = plt.figure(num=2, figsize=figsize)
ax = []
ax.append(fig1.add_subplot(gs[0, 0]))
ax[-1].set_title('Original image')
ax[-1].imshow(img)
ax[-1].axis('off')


ax.append(fig1.add_subplot(gs[0, 1]))
ax[-1].set_title('scale 0.8')
ax[-1].imshow(transformImg(img,0,0,0,0.8))
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 2]))
ax[-1].set_title('scale 1.2')
ax[-1].imshow(transformImg(img,0,0,0,1.2))
ax[-1].axis('off')
#%%
#light
figsize = (10, 10)
gs = gridspec.GridSpec(1, 3)

fig1 = plt.figure(num=2, figsize=figsize)
ax = []
ax.append(fig1.add_subplot(gs[0, 0]))
ax[-1].set_title('Original image')
ax[-1].imshow(img)
ax[-1].axis('off')

tmp=normalizeImageG(img)

ax.append(fig1.add_subplot(gs[0, 1]))
ax[-1].set_title('lightened')
ax[-1].imshow(np.clip(tmp+0.3,-0.5,0.5)+0.5)
ax[-1].axis('off')

ax.append(fig1.add_subplot(gs[0, 2]))
ax[-1].set_title('darkened')
ax[-1].imshow(np.clip(tmp-0.3,-0.5,0.5)+0.5)
ax[-1].axis('off')

#%%
#errors
figsize = (20, 10)
gs = gridspec.GridSpec(1, 7)

fig1 = plt.figure(num=3, figsize=figsize)
ax = []


for i in range(7):
    ax.append(fig1.add_subplot(gs[0, i]))
    ax[-1].set_title('')
    ax[-1].imshow(X_valid[e[i+7]])
    ax[-1].axis('off')

