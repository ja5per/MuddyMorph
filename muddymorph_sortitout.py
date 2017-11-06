#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sort a series of images in order to make the sequence morph-friendly.

Such a re-arrangement can be beneficial for a long series of images without a
clear chronological sequence, like a sketchbook tour or a photo album with
assorted snapshot, where the purpose is to create a slideshow-like movie clip.

Settings come from the parameter file *muddymorph_sortitout.json*.

This is a quick & dirty script, operating on stochastic brute force. Many 
random sequences are generated, and the one that yields the most subtle overall
frame-to-frame changes wins the race. A future version may see a more sensible
and resource-friendly optimization strategy.
"""

###########
# Prepare #
###########

# Dependencies: Open source
import numpy as np
from os import path
from glob import glob
from PIL import Image
from time import time
from shutil import copyfile

# Dependencies: Home grown
import muddymorph_go as gogo
import muddymorph_algo as algo

# Load the settings from file
settings = gogo.loadit('muddymorph_sortitout.json')
for prop in ['folder_in', 'folder_out', 'file_filter', 
             'thumbnail_size', 'maxit_n', 'maxit_t', 'gamma']:
    if not prop in settings:
        raise ValueError('Missing parameter ' + prop)

# Verify both input and output folders exist
if not path.isdir(settings['folder_in']):
    raise ValueError('Unable to locate input folder ' + settings['folder_in'])
if not path.isdir(settings['folder_out']):
    raise ValueError('Unable to locate output folder ' + settings['folder_out'])

# Look for input files
print(gogo.timestamp() + 'Looking for image files ...', end='')
files_src = glob(path.join(settings['folder_in'  ],
                           settings['file_filter']))
n = len(files_src)
print('found {}'.format(n))
if not len(files_src):
    raise ValueError('No files to process')

# All images must have the same dimensions
if not gogo.size_consistency_check(files_src):
    raise ValueError('Inconsistent image sizes')


#############
# Correlate #
#############

# Recycle existing correlation matrix if available
file_corr = path.join(settings['folder_out'], 'correlation.npy')
if path.isfile(file_corr):
    print(gogo.timestamp() + 'Loading correlation matrix from previous run')
    R = np.load(file_corr)
    
else:
    R = np.eye(n)
    
    # Load images as thumbnails for faster processing
    Thumbs = []
    thumbsize = (settings['thumbnail_size'], settings['thumbnail_size'])
    for a in range(n):
        print(gogo.timestamp() + 'Thumbnailing image {}/{}'.format(a + 1, n))
        
        # Fetch greyscale thumbnail
        im = Image.open(files_src[a]).convert('L')
        im.thumbnail(thumbsize, Image.ANTIALIAS)
        
        # Convert to array and stretch contrast
        w, h = im.size
        L    = np.array(im.getdata())
        G    = L.reshape((h, w))
        G    = algo.flotate_bitmap(G)
        G    = G / G.max()
        
        # Add to the collection
        Thumbs.append(G)
        
    # Compute correlation coefficients for all combinations
    for a in range(n):
        print(gogo.timestamp() + 'Correlating image {}/{}'.format(a + 1, n))
        for b in range(a + 1, n):
            La      = Thumbs[a].flatten()
            Lb      = Thumbs[b].flatten()
            simi    = np.corrcoef(La, Lb)[0, 1] ** 2
            R[a, b] = simi
            R[b, a] = simi

    # Save the precious matrix
    print(gogo.timestamp() + 'Saving correlation coefficients')
    np.save(file_corr, R)


###########
# Shuffle #
###########

# Get ready for a long trip
it         = 0
t_start    = time()
t_report   = time()
stopwatch  = 0.
best_seq   = None
best_score = 0.
ori        = np.arange(n)

# Fire up the loop
print(gogo.timestamp() + 'Firing up the iteration loop')
while stopwatch < settings['maxit_t'] and it < settings['maxit_n']:

    if best_seq is None:
        # Kick off with the original sequence
        seq = ori
    
    else:
        # Pick a completely random start point
        pick = algo.pick_int((0, n - 1))
        free = np.ones(n, dtype=bool)
        seq  = [pick]
        
        free[pick] = False
        
        # Continue on a semi-random foot
        # Use correlation coefficients as selection weights
        while len(seq) < n and np.any(free):
            candi = ori[free]
            if len(candi) == 1:
                pick = candi[0]
            else:
                prob    = R[seq[-1], candi].flatten()
                ranking = np.cumsum(prob)
                dice    = np.random.rand() * ranking[-1]
                place   = algo.find(ranking >= dice)[0]
                pick    = candi[place]
            
            seq.append(pick)
            free[pick] = False
        
        seq = np.array(seq)          
    
    # Evaluate sequence overall smoothness
    score = 0.
    for i in range(n - 1):
        score += R[seq[i], seq[i + 1]] ** settings['gamma']
    score /= n - 1
    
    # Is this the best result yet?
    better = score > best_score
    if better:
        best_seq = seq
        best_score = score
    
    # Give a status update at least every ten seconds or so
    if it == 0 or better or time() - t_report > 10:
        msg = 'Best smoothness score is {0:.6f} after {1} iterations'
        print(gogo.timestamp() + msg.format(best_score, it))
        t_report = time()
    
    # On to the next round
    it += 1
    stopwatch = time() - t_start


########
# Copy #
########

print(gogo.timestamp() + 'Copying files in a suitable sequence ...', end='')

# Give output files the same base name as the input files
name = path.basename(path.commonprefix(files_src).strip('0-_ '))
if not name: name = 'img'

# Log the sequence to file
file_seq = path.join(settings['folder_out'], 'sequence.txt')
doc = open(file_seq, 'w')

# Copy the files to the output folder with new names
for b in range(n):
    a        = best_seq[b]
    file_src = files_src[a]
    _, ext   = path.splitext(file_src)
    base     = '{0}_{1:03d}{2}'.format(name, b + 1, ext)
    file_dst = path.join(settings['folder_out'], base)
    
    doc.write(file_src + '\n')
    copyfile(file_src, file_dst)

doc.close()
print('done!')
