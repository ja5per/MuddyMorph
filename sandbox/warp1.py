#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My first warp!

Using scikit-image piecewise affine transformation,
based on manual node assignment with stable corners.
A midpoint morph (halfway between the key frames) is generated.

http://scikit-image.org/docs/dev/auto_examples/plot_piecewise_affine.html
"""

############
# Settings #
############

home     = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a    = home + r'/ball1.jpg'
key_b    = home + r'/ball2.jpg'
nodefile = home + r'/ball_nodeclick.csv'


################
# Dependencies #
################

# Open source
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

# Home grown
import muddymorph_algo as algo


########
# Warp #
########

# Make an announcement
print("")
print("MuddyMorph Warp Proto 1")
print("=======================")
print("")

# Load data
print("Loading images and coordinates ... ", end="")
Ka    = algo.load_rgba(key_a)
Kb    = algo.load_rgba(key_b)
h, w  = Ka.shape[:2]
nodes = np.loadtxt(nodefile, delimiter=',').astype(int)
print("done")

# Add edges to node paths
for x in [0, w - 1]:
    for y in [0, h - 1]:
        nodes = np.row_stack((nodes, [x, y, x, y]))

# Source and destination coordinates  
print("Warping like crazy ... ", end="")      
pa = nodes[:, 0:2]
pb = nodes[:, 2:4]
pi = pa + 0.5 * (pb - pa) 

# Transform A
Ta = PiecewiseAffineTransform()
Ta.estimate(pi, pa)
Wa = warp(Ka, Ta)

# Transform B
dst_b = pb + 0.5 * (pa - pb) 
Tb = PiecewiseAffineTransform()
Tb.estimate(pi, pb)
Wb = warp(Kb, Tb)
print("done")


##########
# Review #
##########

# Show plain images
print("Plotting input ... ", end="")
plt.close('all')
fig = algo.big_figure('MuddyMorph Proto - Warp 1', w * 3, h * 2)
plt.subplot(2, 3, 1)
plt.imshow(Ka)
plt.axis('image')
plt.plot(nodes[:, 0], nodes[:, 1], 'r+')
plt.title('A plain', fontweight='bold')
plt.subplot(2, 3, 2)
plt.imshow(Kb)
plt.axis('image')
plt.plot(nodes[:, 2], nodes[:, 3], 'r+')
plt.title('B plain', fontweight='bold')
plt.subplot(2, 3, 3)
plt.imshow(0.5 * Ka + 0.5 * Kb)
plt.axis('image')
plt.title('A&B plain', fontweight='bold')
print("done")

# Show warped images
print("Plotting result ... ", end="")
plt.subplot(2, 3, 4)
plt.imshow(Wa)
plt.axis('image')
plt.plot(pi[:, 0], pi[:, 1], 'r+')
plt.title('A warp', fontweight='bold')
plt.subplot(2, 3, 5)
plt.imshow(Wb)
plt.axis('image')
plt.plot(pi[:, 0], pi[:, 1], 'r+')
plt.title('B warp', fontweight='bold')
plt.subplot(2, 3, 6)
plt.imshow(0.5 * Wa + 0.5 * Wb)
plt.axis('image')
plt.title('A&B plain', fontweight='bold')
print("done")
