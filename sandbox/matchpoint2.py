#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test BRIEF based matching of silhouette keypoints.

Conclusion: Does not work on line art at all.
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/flower1.png'
key_b = home + r'/flower2.png'

# Edge detection
threshold = 0.2
channel   = 'alpha'
blur      = 2
dolines   = False

# Keypoint detection
keyfill   = 0.2
detail    = 10

# Keypoint matching
cutoff     = 0.02
patch_size = 81


################
# Dependencies #
################

# Open source
import numpy as np
from os import path
import matplotlib.pyplot as plt
from time import time
from skimage.feature import BRIEF, match_descriptors

# Home grown
import muddymorph_algo as algo


##############
# Just Do It #
##############

# Make an announcement
print("")
print("MuddyMorph ProxiMatch Test")
print("==========================")
print("")
print("Image A\t{}".format(path.basename(key_a)))
print("Image B\t{}".format(path.basename(key_b)))
print("")

# Load images
print("Loading images ... ", end="")
Ka = algo.load_rgba(key_a)
Kb = algo.load_rgba(key_b)
print("done")

# Get that silhouette
print("Edge detection ... ", end="")
Da, Sa, Ea = algo.edgy(Ka, channel=channel,
                       threshold=threshold,
                       blur=blur, dolines=dolines)
Db, Sb, Eb = algo.edgy(Kb, channel=channel,
                       threshold=threshold,
                       blur=blur, dolines=dolines)
print("done")

# Equalize that histogram
print("Equalizing ... ", end="")
if channel.lower().startswith('a'):
    Ga = algo.desaturate(Ka, blur=blur, cutoff=cutoff)
    Gb = algo.desaturate(Kb, blur=blur, cutoff=cutoff)
else:
    Ga = algo.desaturate(Da, cutoff=cutoff)
    Gb = algo.desaturate(Db, cutoff=cutoff)
print("done")

# Center of mass determines translation
print("A ", end="")
com_a = algo.commie(Sa)
print("B ", end="")
com_b = algo.commie(Sb)
base = np.array([[com_a['x'], com_a['y'],
                  com_b['x'], com_b['y']]])

# Detect key points
target = int(np.floor(min([Ea.sum(), Eb.sum()]) * keyfill))
print("Picking key points ... ", end="")
kp_a = algo.spawn(Ea, base[:, [0, 1]], target, r_min=detail)
kp_b = algo.spawn(Eb, base[:, [2, 3]], target, r_min=detail)
print("done")

# Match key points
print("Matching key points ... ", end="")
stopwatch = -time()
extractor = BRIEF(patch_size=patch_size, sigma=0)

extractor.extract(Ga, kp_a[:, [1, 0]])
dc_a = extractor.descriptors

extractor.extract(Gb, kp_b[:, [1, 0]])
dc_b = extractor.descriptors

matches = match_descriptors(dc_a, dc_b)
nodes = np.column_stack((kp_a[matches[:, 0]],
                         kp_b[matches[:, 1]]))

stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))


##########
# Review #
##########

fig = plt.figure(figsize=(12, 8))

# Greyscale backdrop
G = 0.25 * (1 - Ga) + 0.25 * (1 - Gb) + 0.5
plt.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)

# Plot paths as green lines
plt.plot([com_a['x'], com_b['x']],
         [com_a['y'], com_b['y']], 'g-')
for i in range(len(nodes)):
    plt.plot(nodes[i, [0, 2]],
             nodes[i, [1, 3]], 'g-')

# Show start points as red dots, and stop points as blue dots
plt.plot(kp_a [:, 0], kp_a [:, 1], 'r.')
plt.plot(kp_b [:, 0], kp_b [:, 1], 'b.')
plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
plt.plot(nodes[:, 2], nodes[:, 3], 'bo')
plt.plot(com_a['x' ], com_a['y' ], 'rs')
plt.plot(com_b['x' ], com_b['y' ], 'bs')

# Bring it on
plt.axis('image')
plt.title('Node paths - From red to blue', fontweight='bold')
fig.tight_layout()

