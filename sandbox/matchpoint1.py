#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test proximity based matching of silhouette keypoints.
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/flower1.png'
key_b = home + r'/flower2.png'

# Edge detection
threshold  = 0.2
channel    = 'alpha'
blur       = 0
dolines    = True
doscharr   = False

# Keypoint detection and matching
simisize   = 14
similim    = 0.5
keyfill    = 0.4
target     = 1000
neighbours = 10
dorotate   = True
maxmove    = 200


################
# Dependencies #
################

# Open source
import numpy as np
from os import path
import matplotlib.pyplot as plt
from time import time

# Home grown
import muddymorph_algo as algo


##############
# Just Do It #
##############

#%% Make an announcement
print("")
print("MuddyMorph ProxiMatch Test")
print("==========================")
print("")
print("Image A\t{}".format(path.basename(key_a)))
print("Image B\t{}".format(path.basename(key_b)))

# Load images
print("Loading images ...", end="")
Ka = algo.load_rgba(key_a)
Kb = algo.load_rgba(key_b)
print("done")

# Get that silhouette
print("Edge detection ...", end="")
Da, Sa, Ea = algo.edgy(Ka, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines, doscharr=doscharr)
Db, Sb, Eb = algo.edgy(Kb, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines, doscharr=doscharr)
print("done")

#%% Compute CoM
print("A ", end="")
com_a = algo.commie(Sa)
print("B ", end="")
com_b = algo.commie(Sb)
if not dorotate:
    com_a['a'] = 0
    com_b['a'] = 0
base = np.array([[com_a['x'], com_a['y'],
                  com_b['x'], com_b['y']]])

#%% Detect key points
target = int(np.floor(min([Ea.sum(), Eb.sum()]) * keyfill))
print("Selecting up to {} key points ... ".format(target), end="")

kp_a  = algo.spawn(Ea, base[:, [0, 1]], target, r_min=simisize)
kp_b  = algo.spawn(Eb, base[:, [2, 3]], target, r_min=simisize)
kp_at = algo.swirl(kp_a, com_a, com_b)
print("done")

#%% Match key points
print("Matching key points ... ", end="")
stopwatch = -time()
nodes_ab, simi_ab = algo.proximatch(Ka, Kb,
                                    kp_a, kp_b, com_a, com_b,
                                    neighbours=neighbours   ,
                                    simisize=simisize       ,
                                    n=int(target / 2)       )
nodes_ba, simi_ba = algo.proximatch(Kb, Ka,
                                    kp_b, kp_a, com_b, com_a,
                                    neighbours=neighbours   ,
                                    simisize=simisize       ,
                                    n=int(target / 2)       )
stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))

#%% Refine key points
print("Refining matches ... ")
stopwatch = -time()
nodes     = np.row_stack((nodes_ab, nodes_ba[:, [2, 3, 0, 1]]))
simi      = np.append(simi_ab, simi_ba)
nodez     = nodes[simi >= similim]

print("\tDiscarding excessive moves ... ")
keep = algo.notsofast(nodez, maxmove, com_a, com_b)
nodez = nodez[keep]

keep = np.zeros_like(nodez, dtype=bool)
while np.any(~keep):
    print("\tStraightening up ... ")
    keep = algo.straightenup(nodez)
    nodez = nodez[keep]

stopwatch += time()
print("\tdone in {0:.3f}s".format(stopwatch))


##########
# Review #
##########

#%% Trajectory plot
plt.close('all')
plt.figure(figsize=(12, 8))

base    = algo.seed(*Ka.shape[:2], com_a, com_b)
traject = np.row_stack((base, nodez))

if True:
    algo.movemap(Da, Db, traject, com_a, com_b)
    plt.axis('image')
    plt.title('Key points - From A (blue) to B (orange)', fontweight='bold')
    
else:
    # Use silhouettes as greyscale backdrop
    G = 0.25 * (1 - Da) + 0.25 * (1 - Db) + 0.5
    plt.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)
    
    # Plot paths as green lines
    plt.plot([com_a['x'], com_b['x']],
             [com_a['y'], com_b['y']], 'g-')
    for i in range(len(nodez)):
        plt.plot(nodez[i, [0, 2]],
                 nodez[i, [1, 3]], 'g-')
    
    # Show start points as red dots, and stop points as blue dots
    plt.plot(kp_a [:, 0], kp_a [:, 1], 'r.')
    plt.plot(kp_at[:, 0], kp_at[:, 1], 'g.')
    plt.plot(kp_b [:, 0], kp_b [:, 1], 'b.')
    plt.plot(nodez[:, 0], nodez[:, 1], 'ro')
    plt.plot(nodez[:, 2], nodez[:, 3], 'bo')
    plt.plot(com_a['x' ], com_a['y' ], 'rs')
    plt.plot(com_b['x' ], com_b['y' ], 'bs')

    # Bring it on
    plt.axis('image')
    plt.title('Key points - From A red to B blue', fontweight='bold')

#%% Doublecheck if the swirl feature works as planned

if False and dorotate:
    clr_a = np.array([0, 0.5, 1])
    clr_b = np.array([1, 0.5, 0])
    plt.figure()
    
    for d in np.linspace(0, 1, 100):
        pot = algo.swirl(traject[:, :2], com_a, com_b, d)[4:]
        clr = clr_a + d * (clr_b - clr_a)
        plt.plot(pot[:,0], pot[:,1], '.', color=clr)
    
    plt.title('Swirl', fontweight='bold')
    plt.axis('image')


#%% Inbetween plot

Ta  = algo.tween(Ka, Kb, traject, d=0.5, f=0.0)
Tb  = algo.tween(Ka, Kb, traject, d=0.5, f=1.0)
T   = algo.tween(Ka, Kb, traject, d=0.5, f=0.5)
fig = plt.figure(figsize=(14, 8))

plt.subplot(1, 3, 1)
plt.imshow(Ta)
plt.axis('image')
plt.title('Midpoint warp of A', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(Tb)
plt.axis('image')
plt.title('Midpoint warp of B', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(T)
plt.axis('image')
plt.title('Fade', fontweight='bold')
plt.xticks([])
plt.yticks([])

fig.tight_layout()
