#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manually assign motion paths for two key frames by clicking on an overlay.

The coordinates will be saved to a headerless CSV file
with columns [x_src, y_src, x_dst, y_dst],
thus enabling morph and warp experiments elsewhere.
"""


############
# Settings #
############

# Key frames
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
key_a = home + r'/ball1.jpg'
key_b = home + r'/ball2.jpg'

# Save node coordinates to this file (CSV)
nodefile = home + r'/ball_nodeclick.csv'


################
# Dependencies #
################

# Open source
import csv
import warnings
from os import path
import matplotlib.pyplot as plt

# Home grown
import muddymorph_algo as algo


###########
# Prepare #
###########

# Make an announcement
print("")
print("MuddyMorph Node Click")
print("=====================")
print("")

# Quick key frame sanity check
print("Sanity check ... ", end="")
if not path.isfile(key_a): raise ValueError('Unable to locate ' + key_a)
if not path.isfile(key_b): raise ValueError('Unable to locate ' + key_b)
print("ok")

# Load bitmaps
print("Loading bitmaps ... ", end="")
Ka   = algo.load_rgba(key_a)
Kb   = algo.load_rgba(key_b)
h, w = Ka.shape[:2]
print("done")

# Show overlay of both keys in a big figure window
print("Plotting overlay ... ", end="")
plt.close('all')
M   = 0.5 * Ka + 0.5 * Kb
fig = algo.big_figure('MuddyMorph Proto - Point & Click', w, h)
plt.imshow(M)
plt.axis('off')
print("done")


#################
# Point & Click #
#################

fid = open(nodefile, 'w')
doc = csv.writer(fid)

print("\nPlease click pairs of node start and stop points " + \
      "(press Enter key when done) ...")

nodes = []
coordfrmt = "({0:04d}, {1:04d}) -> ({2:04d}, {3:04d})"

while True:
    print("#{} ".format(len(nodes) + 1), end="")

    # Ignore annoying event loop deprecation warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = plt.ginput(n=2, timeout=0)
        
    if len(response) < 2:
        print("Done")
        break
    
    pa, pb = response
    xa, ya = int(pa[0]), int(pa[1])
    xb, yb = int(pb[0]), int(pb[1])
    row    = [xa, ya, xb, yb]
    
    nodes.append(row)
    doc.writerow(row)
    print(coordfrmt.format(*row))

fid.close()
