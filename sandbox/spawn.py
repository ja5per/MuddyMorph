#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spawn prototype. Pick evenly spaced points along the silhouette edge.
"""

from time import time
import matplotlib.pyplot as plt
import muddymorph_algo as algo
import muddymorph_go as gogo


############
# Settings #
############

settings = gogo.default_settings()

home = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
settings['keyframes'] = [home + r'/face2.jpg']

n = 30

se = settings['edge']
#se['channel'  ] = 'luminosity'
#se['backcolor'] = (255, 255, 255)
se['blur'] = 4
se['threshold'] = 0.3
#se['dolines'  ] = False


##################
# Test test test #
##################


# Make an announcement
print("")
print("MuddyMorph Spawn Test")
print("=====================")
print("")

# Load data
print("Loading image ... ", end="")
K = algo.load_rgba(settings['keyframes'][0])
h, w = K.shape[:2]
print("done")

# Edge detection
print("Edge detection ... ", end="")
D, S, E = algo.edgy(K,
                    channel   = se['channel'  ],
                    threshold = se['threshold'],
                    blur      = se['blur'     ],
                    dolines   = se['dolines'  ])
print("done")

# CoM detection
com  = algo.commie(S)
base = algo.seed(com, com, *S.shape)[:, :2]

# Pick edge points
print("Picking {} edge points ... ".format(n), end="")
stopwatch  = - time()
kp         = algo.spawn(E, base, n)
stopwatch += time()
print("done in {0:.0f} ms".format(stopwatch * 1000.))


##########
# Review #
##########

print("Plotting result ... ", end="")
plt.close('all')
fig = plt.figure(figsize=(14, 9))

# Show silhouette
plt.subplot(1, 2, 1)
algo.edgeplot(D, S, E)
plt.title('Edge detection', fontweight='bold')

# Show image and points
plt.subplot(1, 2, 2)
plt.imshow(K)
plt.plot(com['x'], com['y'], 'rs', label='com')
plt.plot(kp[:, 0], kp[:, 1], 'r.', label='edge points')
plt.axis('image')
plt.title('Edge node candidates', fontweight='bold')

fig.tight_layout()
