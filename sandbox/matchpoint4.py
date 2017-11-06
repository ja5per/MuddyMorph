#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine silhouette & corner based key point detection & matching.
Does this yield the powerful and flexible solution we seek?
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/face1.jpg'
key_b = home + r'/face2.jpg'

# Edge detection
threshold = 0.2
channel   = 'lightness'
blur      = 0
dolines   = False
doscharr  = True

# Keypoint detection and matching
docorners     = True
dosilhouette  = False
spin          = False
detail        = 0.01
similim       = 0.4
maxmove       = 0.2
maxpoints     = 1000
cornercatcher = 'CENSURE'

# Probably not useful enough to be user settings
spawnpoints   = min(1000, maxpoints)
neighbours    = 10

# Try out edge alpha vignette as a bonus
vignette = 100


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
print("MuddyMorph PathFinder Proto")
print("===========================")
print("")
print("Image A\t{}".format(path.basename(key_a)))
print("Image B\t{}".format(path.basename(key_b)))
print("")

#%% Load images
print("Loading images ...")

stopwatch = -time()
Ka = algo.load_rgba(key_a)
Kb = algo.load_rgba(key_b)
simisize = min(int(np.ceil(max(Ka.shape[:2]) * detail)) + 1, 4)
stopwatch += time()
V = algo.vignette(*Ka.shape[:2], vignette)

print("\tResolution {} x {}".format(Ka.shape[1], Ka.shape[0]))
print("\tDone in {0:.3f}s\n".format(stopwatch))

#%% Get that silhouette
print("Edge detection ...")

stopwatch  = -time()
Da, Sa, Ea = algo.edgy(Ka, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines, doscharr=doscharr)
Db, Sb, Eb = algo.edgy(Kb, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines, doscharr=doscharr)
stopwatch += time()

print("\t{} edge points in A".format(Ea.sum()))
print("\t{} edge points in B".format(Eb.sum()))
print("\tDone in {0:.3f}s\n" .format(stopwatch))

#%% Compute CoM
if dosilhouette or spin:
    print("Center of mass computation ...")
    
    stopwatch = -time()
    print("\tA ", end="")
    com_a = algo.commie(Sa)
    print("\tB ", end="")
    com_b = algo.commie(Sb)
    stopwatch += time()
    print("\tDone in {0:.3f}s\n".format(stopwatch))
    
    if not spin:
        com_a['a'] = 0
        com_b['a'] = 0
    
    nodes0 = algo.seed(*Ea.shape, com_a, com_b)
else:
    nodes0 = algo.seed(*Ka.shape[:2])
    com_a  = dict(x=0, y=0, r=0, a=0.0)
    com_b  = dict(x=0, y=0, r=0, a=0.0)

#%% Detect corners
if docorners:
    print("Extracting corner key points ...")
    
    channel2   = 'lightness' if channel.lower().startswith('a') else channel
    stopwatch  = -time()
    kp_a, dc_a = algo.cornercatch(Ka, target=spawnpoints,
                                  algorithm=cornercatcher,
                                  channel=channel2, blur=blur)
    kp_b, dc_b = algo.cornercatch(Kb, target=spawnpoints,
                                  algorithm=cornercatcher,
                                  channel=channel2, blur=blur)
    stopwatch += time()
    
    print("\t{} corners in A".format(len(kp_a)))
    print("\t{} corners in B".format(len(kp_b)))
    print("\tDone in {0:.3f}s\n".format(stopwatch))

#%% Match corners
if docorners:
    print("Matching corner key points ...")
    
    stopwatch = -time()
    nodes1, simi1 = algo.matchpoint(Ka, Kb, kp_a, kp_b, dc_a, dc_b,
                                    simisize=simisize, similim=similim)
    stopwatch += time()
    
    print("\t{} matches made".format(len(nodes1)))
    print("\tDone in {0:.3f}s\n".format(stopwatch))

#%% Detect silhouette key points
if dosilhouette:
    print("Detecting silhouette points ...")
    
    stopwatch  = -time()
    base       = np.row_stack((nodes0[4:], nodes1)) if docorners else nodes0[4:]
    sp_a       = algo.spawn(Ea, base[:, [0, 1]], spawnpoints, r_min=simisize)
    sp_b       = algo.spawn(Eb, base[:, [2, 3]], spawnpoints, r_min=simisize)
    sp_at      = algo.swirl(sp_a, com_a, com_b)
    stopwatch += time()
    
    print("\t{} silhouette points in A".format(len(sp_a)))
    print("\t{} silhouette points in B".format(len(sp_b)))
    print("\tDone in {0:.3f}s\n"       .format(stopwatch))

#%% Match key points
if dosilhouette:
    print("Matching silhouette points ...")
    
    n_spawn = int(spawnpoints / 2)
    stopwatch = -time()
    
    nodes2, simi2 = algo.proximatch(Ka, Kb, sp_a, sp_b, com_a, com_b,
                                    neighbours=neighbours, n=n_spawn,
                                    simisize=simisize, similim=similim)
    
    nodes3, simi3 = algo.proximatch(Kb, Ka, sp_b, sp_a, com_b, com_a,
                                    neighbours=neighbours, n=n_spawn,
                                    simisize=simisize, similim=similim)
    
    nodes4 = np.row_stack((nodes2, nodes3[:, [2, 3, 0, 1]]))
    simi4 = np.append(simi2, simi3)
    stopwatch += time()
    
    print("\t{} matches made".format(len(nodes4)))
    print("\tDone in {0:.3f}s\n".format(stopwatch))

#%% Keep only kosher matches
print("Refining matches ... ")

if dosilhouette and docorners:
    nodez = np.row_stack((nodes1, nodes4))
    simiz = np.append(simi1, simi4)
elif dosilhouette:
    nodez, simiz = nodes4, simi4
elif docorners:
    nodez, simiz = nodes1, simi1
else:
    raise ValueError('Nothing to refine')
print("\tStarting point is {} matches".format(len(nodez)))

print("\tCombining effective duplicates ... ", end="")
stopwatch = -time()
nodez, simiz = algo.gettogether(nodez, simiz, simisize)
print("{} matches remain".format(len(nodez)))

print("\tDiscarding excessive moves ... ", end="")
diago = np.ceil(np.sqrt(Ka.shape[0] ** 2 + Ka.shape[1] ** 2))
m     = int(maxmove * diago)
keep  = algo.notsofast(nodez, m, com_a, com_b)
nodez = nodez[keep]
simiz = simiz[keep]
print("{} matches remain".format(len(nodez)))

keep = np.zeros_like(nodez, dtype=bool)
repeat = 1
while np.any(~keep) and repeat <= 10:
    print("\tStraightening up pass {} ... ".format(repeat), end="")
    keep    = algo.straightenup(nodez)
    nodez   = nodez[keep]
    simiz   = simiz[keep]
    repeat += 1
    print("{} matches remain".format(len(nodez)))

if len(nodez) > maxpoints:
    print("\tCherry picking the top ... ", end="")
    seq   = np.argsort(simiz)[::-1]
    nodez = nodez[seq][:maxpoints]
    simiz = simiz[seq][:maxpoints]
    print("{} matches remain".format(len(nodez)))

traject = np.row_stack((nodes0, nodez))
stopwatch += time()
print("\tDone in {0:.3f}s\n".format(stopwatch))

#%% Inbetween
print("Inbetweening ...")

print("\tWarp A to midpoint ... ", end="")
stopwatch = -time()
Ta = algo.tween(Ka, Kb, traject, d=0.5, f=0.0)
stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))

print("\tWarp B to midpoint ... ", end="")
stopwatch = -time()
Tb = algo.tween(Ka, Kb, traject, d=0.5, f=1.0)
stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))

print("\tWarp and fade A & B ... ", end="")
stopwatch = -time()
Tc = algo.tween(Ka, Kb, traject, d=0.5)
stopwatch += time()
print("done in {0:.3f}s".format(stopwatch))

print("\tBlobbify ... ", end="")
radii = algo.inflate_blobs(traject, scale=1., r_max=int(diago),
                           com_a=com_a, com_b=com_b)
stopwatch = -time()
Td = algo.tween(Ka, Kb, traject, d=0.5, radii=radii, V=V)
stopwatch += time()
print("done in {0:.3f}s\n".format(stopwatch))


#%% Plot trajectories, deformations, and inbetweens
plt.close('all')
plt.figure(figsize=(12, 8))
algo.movemap(Da, Db, traject, com_a, com_b)
plt.title('Key points - From A (blue) to B (orange)', fontweight='bold')

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
plt.imshow(Tc)
plt.axis('image')
plt.title('Fade', fontweight='bold')
plt.xticks([])
plt.yticks([])

fig.tight_layout()

plt.figure(figsize=(9, 6))
plt.imshow(Td)
plt.axis('image')
plt.title('Blob morph', fontweight='bold')
plt.xticks([])
plt.yticks([])
