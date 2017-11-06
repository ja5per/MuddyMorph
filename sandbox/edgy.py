#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silhouette & Contour Detection Prototype.
"""

############
# Settings #
############

home      = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
imagefile = home + r'/frogfront1.jpg'

threshold = 0.2
channel   = 'luminosity'
invert    = False
blur      = 2
backcolor = None #(49, 219, 8)
linecolor = (0, 0, 0)
dolines   = True


################
# Dependencies #
################

# Open source
import numpy as np
from os import path
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.color import label2rgb

# Home grown
import muddymorph_algo as algo


########
# Edgy #
########

# Make an announcement
print("")
print("MuddyMorph Edgy Proto")
print("=====================")
print("")

# Load that image
B = algo.load_rgba(imagefile)

# Auto background color is median of edge pixels
if backcolor is None:
    W = np.zeros(B.shape[:2], dtype=bool)
    
    W[: , 0 ] = 1
    W[: , -1] = 1
    W[0 , : ] = 1
    W[-1, : ] = 1
    
    backcolor = []
    for c in range(3):
        Bsub = np.squeeze(B[:, :, c])[W]
        backcolor.append(int(255 * np.median(Bsub)))
    backcolor = tuple(backcolor)

# Difference with the background
D = B.copy()
if channel != 'alpha':
    for c in range(3):
        D[:, :, c] = abs(D[:, :, c] - backcolor[c] / 255.)
D = algo.desaturate(D, channel)

# Remove lines from silhouette
if dolines:
    channel2 = channel if channel != 'alpha' else 'rms'
    D2 = B.copy()
    for c in range(3):
        D2[:, :, c] = abs(D2[:, :, c] - linecolor[c] / 255.)
    D2     = algo.desaturate(D2, channel2)
    low    = D2 < D
    D[low] = D2[low]

if invert: D = 1 - D

if blur > 0: D = ndimage.gaussian_filter(D, sigma=blur)

S = D >= threshold

if S.any():
    S2 = ndimage.binary_dilation(S)
    E  = S != S2
else:
    E = S.copy()


##########
# Review #
##########

h, w    = B.shape[:2]
name, _ = path.splitext(path.basename(imagefile))
n_tot   = B.shape[0] * B.shape[1]
n_e     = E.sum()
edgy    = 100. * n_e / n_tot
Gx, Gy  = algo.grid(E)
xe, ye  = Gx[E].flatten(), Gy[E].flatten()

print("Image       \t{0:s}"   .format(name ))
print("Pixels total\t{0:d}"   .format(n_tot))
print("Pixels edge \t{0:d}"   .format(n_e  ))
print("Edgyness    \t{0:.2f}%".format(edgy ))
print("")

plt.close('all')

if True:
    # All-In-One chart
    clr_in  = (0., 1., 0.)
    clr_out = (1., 1., 1.)
    plt.figure(figsize=(10, 10. * h / w))
    L = label2rgb(S.astype(int), D, alpha=0.3,
                  colors=[clr_out, clr_in])
    plt.imshow(L)
    plt.plot(xe, ye, '.', color=clr_in, markersize=2)
    plt.axis('image')
    plt.title('Silhouette Chart', fontweight='bold')
    plt.show()

else:
    # Subplot heaven
    fig = algo.big_figure('MuddyMorph Proto - Edge', w * 2, h * 2)
    
    plt.subplot(2, 2, 1)
    plt.imshow(B)
    plt.axis('image')
    plt.title('Image', fontweight='bold')
    
    plt.subplot(2, 2, 2)
    plt.imshow(D, cmap=plt.cm.gray)
    plt.axis('image')
    plt.title('Delta', fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 3)
    plt.imshow(S, cmap=plt.cm.gray)
    plt.axis('image')
    plt.title('Silhouette', fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 4)
    plt.imshow(E, cmap=plt.cm.gray)
    plt.plot(xe, ye, 'w.', markersize=2, alpha=0.2)
    plt.axis('image')
    plt.title('Edge', fontweight='bold')
    plt.xticks([])
    plt.yticks([])

    fig.tight_layout()
