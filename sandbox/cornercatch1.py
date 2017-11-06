#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test corner detection based keypoint extraction.
Try out three juicy of-the-shelf algorithms; BRIEF, ORB, CENSURE.
"""

# The image to process
home      = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
imagefile = home + r'/shake1.png'

# Convert to greyscale thusly
blur      = 0
channel   = 'lightness'
backcolor = (0, 0, 0)
dolines   = False

# Enforce silhouette?
threshold = 0.2
silly     = False

# Keypoint detection
target        = 2000
cutoff        = 0.01
algorithm     = 'orb' # ORB / CENSURE
orb_threshold = 0.08
censure_mode  = 'STAR' # DoB / Octagon / STAR

# Dependencies
from time import time
import matplotlib.pyplot as plt
import muddymorph_algo as algo

# Load images
print("Loading images ... ", end="")
K = algo.load_rgba(imagefile)
print("done")

# Keypoint extraction
# Tweak these parameters for a few test cases,
# and decide which ones should become knobs in the user interface.
print("Keypoint extraction ... ", end="")
stopwatch = -time()
kp, descriptors = algo.cornercatch(K, target=target, algorithm=algorithm, 
                                   channel=channel, blur=blur, cutoff=cutoff,
                                   orb_threshold=orb_threshold, 
                                   censure_mode=censure_mode)
stopwatch += time()
print("{0} points found in {1:.2f}s".format(len(kp), stopwatch))

# Review the result
plt.close('all')
G = algo.desaturate(K, channel)
h, w = G.shape
fig = algo.big_figure('MuddyMorph - CornerCatch Proto', w * 1.5, h)
plt.subplot(1, 2, 1)
plt.imshow(K)
plt.title('Image', fontweight='bold')
plt.subplot(1, 2, 2)
plt.imshow(G, cmap=plt.cm.gray)
plt.plot(kp[:,0], kp[:, 1], '.', color='orange')
plt.axis('image')
plt.title('Keypoints', fontweight='bold')
fig.tight_layout()

