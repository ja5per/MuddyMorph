#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Widen up narrow images to full frame aspect ratio,
by adding mirrored copies on the sides.

These images are gradient-wise faded out towards the edge,
either to the image border color or a specific hand picked color.
This creates a vignette-ish effect,
guiding the eye towards the original image in the image center.

Quick-and-dirty script with hard coded parameters.
"""

############
# Settings #
############

# Process these files
folder_in = '/Users/jasper/Movies/Project/GRR/Sketsj/cells'
filepattern = '*.jpg'

# Crop this many pixels from the left and right of the input images
sidecrop = (2, 2)

# Full frame size (width, height) in pixels
size_out = (1920, 1080)

# Export quality in case of JPEG
quality = 90

# Fade to this color
# (Set to None to fade out to the image edge color)
fadecolor = None

# Fade out with this amount
# (From 0 for no fading to 1 for fulll monty)
fade = 1

# Save the result in this folder, using the same file names
# (Take care, existing files will be overwritten)
folder_out = '/Users/jasper/Movies/Project/GRR/Sketsj/mirror'


############
# Protocol #
############

# Dependencies
import numpy as np
from os import path
from glob import glob
from time import time
import muddymorph_algo as algo
from skimage.transform import rescale

# Look for images
print('Looking for images ...', end='')
files_in = glob(path.join(folder_in, filepattern))
print('found {}\n'.format(len(files_in)))

# Process them pretty pictures, yee-haw!
progress = 1
stopwatch = -time()
for file_in in files_in:
    
    # Spread the word
    name = path.basename(file_in)
    msg = 'Processing [{progress}/{n}] {name} ... '
    print(msg.format(progress=progress, n=len(files_in), name=name), end='')
    
    # Load that image
    Mi = algo.load_rgba(file_in)
    
    # Crop away stripy artefacts
    if sum(sidecrop):
        Mi = Mi[:, +sidecrop[0]:-sidecrop[1]]
    
    # Border color
    boco = algo.bordercolor(Mi)
    boco = np.append(boco, 255)
    faco = boco if fadecolor is None else fadecolor
    faco = [f / 255. for f in faco]
    
    # Rescale to the specified height
    w, h = size_out
    if Mi.shape[0] == h:
        Mm = Mi
    else:
        scale = 1. * h / Mi.shape[0]
        Mm = rescale(Mi, scale)
    
    # Flip horizontally
    Ml = Mm[:, ::-1].copy()
    Mr = Mm[:, ::-1].copy()
    
    # Crop the left and right sides
    margin = w - Mm.shape[1]
    wl     = int(np.floor(margin / 2.))
    wr     = int(np.ceil (margin / 2.))
    Ml     = Ml[:, -wl:]
    Mr     = Mr[:, :+wr]
    
    # Make mesh grid for x-coordinates
    Xl, _ = algo.grid(Ml)
    Xr, _ = algo.grid(Mr)
    
    # Fade factors slash alpha masks
    Fl = +1. * Xl / Xl[0, -1] + 0
    Fr = -1. * Xr / Xr[0, -1] + 1
    
    # Faaaaaaaade out, again!
    # (Yes, that was a very subtle Radiohead reference)
    Bl = np.ones((h, Ml.shape[1])) * (1 - Fl)
    Br = np.ones((h, Mr.shape[1])) * (1 - Fr)
    for c in range(Mm.shape[2]):
        Ml[:, :, c] = Ml[:, :, c] * Fl + Bl * faco[c]
        Mr[:, :, c] = Mr[:, :, c] * Fr + Br * faco[c]
    
    # Put it all together dude
    Mo = np.zeros((size_out[1], size_out[0], Mm.shape[2]))
    for c in range(Mm.shape[2]):
        Mo[:, :, c] = np.column_stack((Ml[:, :, c],
                                       Mm[:, :, c],
                                       Mr[:, :, c]))
    
    # Save the harvest
    file_out = path.join(folder_out, name)
    algo.save_rgba(Mo, file_out, quality=quality)
    
    # On to the next round
    progress += 1
    print("done")

# The deed has been done
stopwatch += time()
if len(files_in):
    msg = 'Finished! {n} images processed in {t:.1f}s\n'
    print(msg.format(n=len(files_in), t=stopwatch))
