#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoM prototype. Estimate center of mass properties for two key frames.
Does it succeed in decently capturing the main movement?
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/dog1.png'
key_b = home + r'/dog2.png'

# Edge detection
threshold = 0.2
channel   = 'alpha'
blur      = 2
dolines   = True


################
# Dependencies #
################

# Open source
from os import path
import matplotlib.pyplot as plt

# Home grown
import muddymorph_algo as algo
  

##############
# Just Do It #
##############

# Make an announcement
print("")
print("MuddyMorph CoM Proto")
print("====================")
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
Da, Sa, Ea = algo.edgy(Ka, channel=channel,
                       threshold=threshold,
                       blur=blur, dolines=dolines)
Db, Sb, Eb = algo.edgy(Kb, channel=channel,
                       threshold=threshold,
                       blur=blur, dolines=dolines)
print("done\n")

# Compute CoM
print("A ", end="")
com_a = algo.commie(Sa)
print("B ", end="")
com_b = algo.commie(Sb)


##########
# Review #
##########


fig = plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.imshow(Da, cmap=plt.cm.gray)
algo.complot(com_a)
plt.axis('image')
plt.title('A', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(Db, cmap=plt.cm.gray)
algo.complot(com_b)
plt.axis('image')
plt.title('B', fontweight='bold')
plt.xticks([])
plt.yticks([])

fig.tight_layout()
