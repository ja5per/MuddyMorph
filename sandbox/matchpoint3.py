#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Put corner keypoint extraction and matching (ORB / CENSURE) to the test.
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/head1.jpg' #r'/church1.jpg' #r'/clay1.jpg'
key_b = home + r'/head2.jpg' #r'/church2.jpg' #r'/clay2.jpg'

# Keypoint extraction
algorithm     = 'orb'
channel       = 'lightness'
blur          = 2
target        = 2000
cutoff        = 0.01
orb_threshold = 0.08
censure_patch = 49

# Keep only decent matches (not too far, and similar)
simisize = 20
similim  = 0.4
maxmove  = 150


################
# Dependencies #
################

# Open source
import numpy as np
from os import path
import matplotlib.pyplot as plt
from time import time
from skimage.feature import match_descriptors

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

#%% Load images
print("Loading images ... ", end="")
Ka = algo.load_rgba(key_a)
Kb = algo.load_rgba(key_b)
print("done")

#%% Key point extraction
print("Extracting key points ... ", end="")
stopwatch = -time()
kp_a, dc_a = algo.cornercatch(Ka, algorithm=algorithm, target=target,
                              channel=channel, blur=blur,
                              cutoff=cutoff, orb_threshold=orb_threshold)
kp_b, dc_b = algo.cornercatch(Kb, algorithm=algorithm, target=target,
                              channel=channel, blur=blur,
                              cutoff=cutoff, orb_threshold=orb_threshold)
stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))

# Match key points
print("Matching key points ... ", end="")
stopwatch  = -time()
matches    = match_descriptors(dc_a, dc_b)
nodes      = np.column_stack((kp_a[matches[:, 0]],
                              kp_b[matches[:, 1]]))
nodes      = np.round(nodes).astype(int)
stopwatch += time()
print("{0} candidates in {1:.3f}s".format(len(nodes), stopwatch))

#%% Refine selection, looking at the similarity score and distance traveled
print("Refining match points ... ", end="")
stopwatch = -time()
simi = []
for i in range(len(nodes)):
    simi.append(algo.similarity(Ka, Kb, 
                                xa=nodes[i, 0], ya=nodes[i, 1], 
                                xb=nodes[i, 2], yb=nodes[i, 3], 
                                simisize=simisize))
simi = np.array(simi)
move = np.sqrt((nodes[:, 2] - nodes[:, 0]) ** 2 + \
               (nodes[:, 3] - nodes[:, 1]) ** 2)
stopwatch += time()
keep = (simi >= similim) & (move <= maxmove)
nodez = nodes[keep, :]
print("{0} nodes remain after {1:.3f}s".format(len(nodez), stopwatch))

#%% Group together nodes that are very close to each other
print("Grouping together nearby nodes ... ", end="")
stopwatch = -time()
noodle    = []
free      = np.ones(len(nodez), dtype=bool)
i         = 0
while i < len(free) and np.any(free):
    while not(free[i]): i += 1
    
    da = algo.distance(nodez[i, 0], nodez[i, 1], nodez[:, 0], nodez[:, 1])
    db = algo.distance(nodez[i, 2], nodez[i, 3], nodez[:, 2], nodez[:, 3])
    j  = algo.find((free) & (da <= simisize) & (db <= simisize))
    mu = np.mean(nodez[j, :], axis=0).astype(int)
    
    noodle.append(mu)
    free[j] = False

noodle = np.array(noodle)
stopwatch += time()
print("{0} nodes remain after {1:.3f}s".format(len(noodle), stopwatch))

#%% In case of crossing nodes discard the longest path
print("Discarding crossing paths ... ", end="")
stopwatch = -time()
move = np.sqrt((noodle[:, 2] - noodle[:, 0]) ** 2 + \
               (noodle[:, 3] - noodle[:, 1]) ** 2)
free = np.ones(len(noodle), dtype=bool)
keep = np.ones_like(free)
i    = 0
while i < len(free) and np.any(free):
    while not(free[i]): i += 1
    free[i] = False
    
    iscross = np.zeros_like(keep)
    for j in algo.find(keep):
        if j != i:
            iscross[j] = algo.crisscross(*noodle[i, :],
                                         *noodle[j, :])
    
    if np.any(iscross):
        iscross[i]  = True
        scrap       = np.argmax(move * iscross)
        keep[scrap] = False
        free[scrap] = False

nono = noodle[keep, :]
stopwatch += time()
print("{0} nodes remain after {1:.3f}s".format(len(nono), stopwatch))


##########
# Review #
##########

#%% Trajectory chart
plt.close('all')
fig = plt.figure(figsize=(12, 8))

# Greyscale backdrop
Ga = algo.desaturate(Ka, blur=blur, cutoff=cutoff)
Gb = algo.desaturate(Kb, blur=blur, cutoff=cutoff)
G  = 0.25 * (1 - Ga) + 0.25 * (1 - Gb) + 0.5
plt.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)

# Plot paths as straight lines
for i in range(len(nono)):
    plt.plot(nono[i, [0, 2]],
             nono[i, [1, 3]], 'k-')

# Show start points as red dots, and stop points as blue dots
plt.plot(nono[:, 0], nono[:, 1], 'r.')
plt.plot(nono[:, 2], nono[:, 3], 'b.')

# Bring it on
plt.axis('image')
plt.title('Node paths - From red to blue', fontweight='bold')
fig.tight_layout()

#%% Halfway inbetween

corner  = algo.seed(*Ka.shape[:2])
traject = np.row_stack((corner, nono))
T       = algo.tween(Ka, Kb, traject)
fig     = plt.figure(figsize=(12, 8))
plt.imshow(T)
plt.axis('image')
plt.title('Halfway inbetween', fontweight='bold')
fig.tight_layout()
