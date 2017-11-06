#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local contrast prototype.

Local contrast evaluation is supposed to yield a map that approximates the
intensity difference between nearby light and dark zones for all points of a
supplied bitmap, given a measure for the evaluation radius. Points with high
local contrast are eligible for serving as morph nodes.

The easiest approach that comes to mind is:
    
    1. Convert the bitmap to greyscale,
       using the same channel as for silhouette detection
       (or luminosity in case of alpha).
       
    2. Normalize contrast through adaptive histogram equalization, see
       https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
       
    3. Subtract a Gaussian blurred copy (use the evaluation radius as sigma).
    
    4. Take the absolute value of this difference. Tah dah! Finished!
       Zero indicates a total absence of local intensity variation,
       and one indicates ultimate contrast.
"""


############
# Settings #
############

# The file to process
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
image = home + r'/clay1.jpg'

# Silhouette detection
channel = 'greenscreen'

# Intensity equalization threshold (0.1 is already quite agressive)
cutoff = 0.04

# The algorithm to use (muddy/scharr/sobel/prewitt)
algorithm = 'muddy'

# Convolution radius in pixels (in case of muddy approach)
detail = 16


################
# Dependencies #
################

# Open source
from os import path
from time import time
import matplotlib.pyplot as plt

# Home grown
import muddymorph_algo as algo


###########
# Try-out #
###########

# Make an announcement
print("")
print("MuddyMorph Local Contrast Proto")
print("===============================")
print("")

B = algo.load_rgba(image)
D, _, _ = algo.edgy(B, channel=channel)

stopwatch = -time()
C = algo.loco(D,
              algorithm=algorithm,
              cutoff=cutoff, detail=detail)
stopwatch += time()


##########
# Review #
##########

print("Image  \t{}"      .format(path.basename(image)))
print("Channel\t{}"      .format(channel ))
print("Cutoff \t{}"      .format(cutoff  ))
print("Detail \t{}"      .format(detail  ))
print("Min    \t{0:.3f}" .format(C.min ()))
print("Mean   \t{0:.3f}" .format(C.mean()))
print("Max    \t{0:.3f}" .format(C.max ()))
print("Time   \t{0:.3f}s".format(C.max ()))

fig = plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(B)
plt.axis('image')
plt.title('Image', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(D, cmap=plt.cm.gray)
plt.axis('image')
plt.title('Greyscale', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(C, cmap=plt.cm.gray)
plt.axis('image')
plt.title('Local Contrast', fontweight='bold')
plt.xticks([])
plt.yticks([])

fig.tight_layout()
