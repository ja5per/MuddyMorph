#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp prototype. A function has been written. Will it fly?
Try out a simple CoM translation ...
"""

from numpy import array
import matplotlib.pyplot as plt
import muddymorph_algo as algo
import muddymorph_go as gogo

############
# Settings #
############

settings = gogo.default_settings()

home = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
settings['keyframes'] = [home + r'/ball1.jpg',
                         home + r'/ball2.jpg']

se = settings['edge']


##################
# Test test test #
##################


# Make an announcement
print("")
print("MuddyMorph Warp Proto 2")
print("=======================")
print("")

# Load data
print("Loading images ... ", end="")
Ka   = algo.load_rgba(settings['keyframes'][0])
Kb   = algo.load_rgba(settings['keyframes'][1])
h, w = Ka.shape[:2]
print("done")

# Edge detection
print("Edge detection ... ", end="")
Da, Sa, Ea = algo.edgy(Ka,
                       channel   = se['channel'  ],
                       threshold = se['threshold'],
                       blur      = se['blur'     ],
                       dolines   = se['dolines'  ])
Db, Sb, Eb = algo.edgy(Kb,
                       channel   = se['channel'  ],
                       threshold = se['threshold'],
                       blur      = se['blur'     ],
                       dolines   = se['dolines'  ])
print("done")

# CoM detection
print("A ", end="")
com_a = algo.commie(Sa)
print("B ", end="")
com_b = algo.commie(Sb)

# Basic trajectory
nodes_ab = algo.seed(com_a, com_b, h, w)
nodes_ba = nodes_ab[:, [2, 3, 0, 1]]
nodes_i  = 0.5 * (nodes_ab[:, [0, 1]] + nodes_ab[:, [2, 3]])

# Warp
print("Warping ...", end="")
Wa, posi_a = algo.deform(Ka, nodes_ab, 0.5, com_a, com_b)
Wb, posi_b = algo.deform(Kb, nodes_ba, 0.5, com_b, com_a)
print("done")

# Apply solid background color
backcolor = (0.5 * (array(algo.bordercolor(Ka)) + \
                    array(algo.bordercolor(Kb)))).astype(int)
Waf = algo.flatten_bitmap(Wa, backcolor)
Wbf = algo.flatten_bitmap(Wb, backcolor)


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
plt.plot(nodes_ab[:, 0], nodes_ab[:, 1], 'r+')
plt.title('A plain', fontweight='bold')
plt.subplot(2, 3, 2)
plt.imshow(Kb)
plt.axis('image')
plt.plot(nodes_ab[:, 2], nodes_ab[:, 3], 'r+')
plt.title('B plain', fontweight='bold')
plt.subplot(2, 3, 3)
plt.imshow(0.5 * Ka + 0.5 * Kb)
plt.axis('image')
plt.title('A&B plain', fontweight='bold')
print("done")

# Show warped images
print("Plotting result ... ", end="")
plt.subplot(2, 3, 4)
plt.imshow(Waf)
plt.plot(posi_a[:, 0], posi_a[:, 1], 'r+')
#plt.plot(nodes_i[:, 0], nodes_i[:, 1], 'r+')
plt.axis('image')
plt.title('A warp', fontweight='bold')
plt.subplot(2, 3, 5)
plt.imshow(Wbf)
plt.plot(posi_b[:, 0], posi_b[:, 1], 'r+')
#plt.plot(nodes_i[:, 0], nodes_i[:, 1], 'r+')
plt.axis('image')
plt.title('B warp', fontweight='bold')
plt.subplot(2, 3, 6)
plt.imshow(0.5 * Waf + 0.5 * Wbf)
plt.axis('image')
plt.title('A&B plain', fontweight='bold')
print("done")
