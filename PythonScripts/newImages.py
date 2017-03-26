# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:46:21 2017

@author: diz
"""

import os

newImages = []
for entry in os.scandir('../Data/Signs/hard'):
    if entry.is_file():
        print(entry.name)
        img = cv2.imread(entry.path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(32,32))
        newImages.append(img)
        
newImages = np.array(newImages)
        
cols = 5

figsize = (10, 5)
nImages = newImages.shape[0]

gs = gridspec.GridSpec(nImages // cols + 1, cols)

fig1 = plt.figure(num=1, figsize=figsize)
ax = []
for i in range(nImages):
    row = (i // cols)
    col = i % cols
    ax.append(fig1.add_subplot(gs[row, col]))
    #example
    img = newImages[i]
    ax[-1].imshow(img)
    ax[-1].axis('off')

newImagesN = normalizeImageList(newImages)        
