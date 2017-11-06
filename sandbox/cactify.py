#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cactification try-out. Find perpendicular angles to local contour lines.

If succesfull, the silhouette node matching protocol can be upgraded;
favor plain expansion or contraction of the shape, 
thus preventing crossing paths and other nasty entanglements.
"""


############
# Settings #
############

# Process these files
home = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
imagefile = home + r'/chameleon1.png'

# Edge detection
threshold   = 0.2
channel     = 'alpha'
blur        = 1
dolines     = False
doscharr    = False
spawnpoints = 100

# Angle evaluation box size
detail = 0.03


################
# Dependencies #
################

# Open source
from time import time
import numpy as np
from os import path
import matplotlib.pyplot as plt

# Home grown
import muddymorph_algo as algo


###########
# Prepare #
###########


#%% Make an announcement
print("")
print("MuddyMorph Cactify Proto")
print("========================")
print("")
print("Image\t{}".format(path.basename(imagefile)))
print("")


#%% Load images
print("Loading images ...")

stopwatch = -time()
K = algo.load_rgba(imagefile)
simisize = max(int(np.ceil(max(K.shape[:2]) * detail)) + 1, 4)
stopwatch += time()

print("\tResolution {} x {}".format(K.shape[1], K.shape[0]))
print("\tDone in {0:.3f}s\n".format(stopwatch))


#%% Get that silhouette
print("Edge detection ...")

stopwatch  = -time()
D, S, E = algo.edgy(K, channel=channel, threshold=threshold,
                    blur=blur, dolines=dolines, doscharr=doscharr)
stopwatch += time()

print("\t{} edge points".format(E.sum()))
print("\tDone in {0:.3f}s\n" .format(stopwatch))


#%% Center of mass
print("Center of mass computation ...\n\t", end="")

stopwatch = -time()
com = algo.commie(S)
stopwatch += time()
print("\tDone in {0:.3f}s\n".format(stopwatch))


#%% Detect silhouette key points
print("Detecting silhouette points ...", end="")

stopwatch  = -time()
nodes0     = algo.seed(*E.shape, com, com)
base       = nodes0[4:]
sp         = algo.spawn(E, base[:, [0, 1]], spawnpoints, r_min=simisize)
stopwatch += time()

print("\t{} silhouette points".format(len(sp)))
print("\tDone in {0:.3f}s\n"  .format(stopwatch))


###########
# Cactify #
###########

#%% Mini center of mass evaluation for each evaluation point
print("Cactification ... ", end="")

stopwatch  = -time()
h, w  = K.shape[:2]
vv = np.zeros(sp.shape)
for i in range(len(sp)):
    x, y = sp[i, 0], sp[i, 1]
    dxm  = min(simisize, x)
    dym  = min(simisize, y)
    dxp  = min(simisize, w - x - 1)
    dyp  = min(simisize, h - y - 1)
    Q    = E[(y - dym):(y + dyp + 1), (x - dxm):(x + dxp + 1)]
    ting = algo.commie(Q, verbose=False)
    
    vv[i, 0] = ting['vx']
    vv[i, 1] = ting['vy']

stopwatch += time()
print("done in {0:.3f}s\n".format(stopwatch))

#%% Measure angle to CoM
print("Measuring angles ... ", end="")

stopwatch = -time()
zeta = np.zeros(len(sp))
for i in range(len(vv)):
    phi = algo.measure_angle(com['x'], com['y'],
                             sp[i, 0], sp[i, 1],
                             sp[i, 0] + vv[i, 0], sp[i, 1] + vv[i, 1])
    zeta[i] = phi * 2 / np.pi

stopwatch += time()
print("done in {0:.3f}s\n".format(stopwatch))


##########
# Review #
##########

plt.close('all')
G    = algo.desaturate(K, channel)
fig  = plt.figure('MuddyMorph - CornerCatch Proto', figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(K)
plt.title('Image', fontweight='bold')

plt.subplot(1, 2, 2)
plt.imshow(G, cmap=plt.cm.gray)
plt.plot(sp[:,0], sp[:, 1], '.', color='orange')
for i in range(len(sp)):
    x1 = sp[i, 0] + simisize * zeta[i] * vv[i, 1]
    x2 = sp[i, 0] - simisize * zeta[i] * vv[i, 1]
    y1 = sp[i, 1] - simisize * zeta[i] * vv[i, 0]
    y2 = sp[i, 1] + simisize * zeta[i] * vv[i, 0]
    plt.plot([x1, x2], [y1, y2], '-', color='orange')
plt.plot(com['x'], com['y'], 'r+')
plt.axis('image')
plt.title('Cactus', fontweight='bold')
fig.tight_layout()

