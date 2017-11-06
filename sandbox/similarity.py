#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity prototype. How similar are two spots in two bitmaps?

Given two similar bitmaps, try out many arbitrary edge point pairs.
Measure the time needed, and show the best and worst spots.
"""

############
# Settings #
############

# Process these files
home  = r'/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
#key_a = home + r'/clay1.jpg'
#key_b = home + r'/clay2.jpg'
key_a = home + r'/chameleon1.png'
key_b = home + r'/chameleon2.png'

# Edge detection
threshold = 0.2
channel   = 'alpha'
blur      = 2
dolines   = True

# Similarity evaluation
repeats  = 1000
simisize = 20


################
# Dependencies #
################

# Open source
from time import time
from os import path
import matplotlib.pyplot as plt

# Home grown
import muddymorph_algo as algo


########
# Test #
########

# Make an announcement
print("")
print("MuddyMorph Similarity Proto")
print("===========================")
print("")

# Load images
print("Loading images ...", end="")
Ka = algo.load_rgba(key_a)
Kb = algo.load_rgba(key_b)
print("done")

print("Edge detection ...", end="")
Da, Sa, Ea = algo.edgy(Ka, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines)
Db, Sb, Eb = algo.edgy(Kb, channel=channel, threshold=threshold,
                       blur=blur, dolines=dolines)
Gx, Gy   = algo.grid(Ea)
xea, yea = Gx[Ea].flatten(), Gy[Ea].flatten()
xeb, yeb = Gx[Eb].flatten(), Gy[Eb].flatten()
print("done")

# Serial random similarity evaluation
msg = "Random similarity evaluation ({} edge points) ..."
print(msg.format(repeats), end="")
best, worst = None, None
stopwatch = -time()
for repeat in range(repeats):
    
    ia           = algo.pick_int((0, len(xea) - 1))
    ib           = algo.pick_int((0, len(xeb) - 1))
    xa, ya       = xea[ia], yea[ia]
    xb, yb       = xeb[ib], yeb[ib]
    simi, Qa, Qb = algo.similarity(Ka, Kb, xa, ya, xb, yb,
                                   simisize, returnboxes=True)
    
    if best is None or simi > best['simi']:
        best = dict(simi=simi, xa=xa, ya=ya, xb=xb, yb=yb, Qa=Qa, Qb=Qb)
    
    if worst is None or simi < worst['simi']:
        worst = dict(simi=simi, xa=xa, ya=ya, xb=xb, yb=yb, Qa=Qa, Qb=Qb)

stopwatch += time()
print("done")


##########
# Review #
##########

h, w = Ka.shape[:2]
name_a, _ = path.splitext(path.basename(key_a))
name_b, _ = path.splitext(path.basename(key_b))
print("")
print("File  A          \t{}"       .format(name_a))
print("File  B          \t{}"       .format(name_b))
print("Resolution       \t{h}x{w}"  .format(w=w, h=h))
print("Edge points A    \t{}"       .format(len(xea)))
print("Edge points B    \t{}"       .format(len(xeb)))
print("Edge fill A      \t{0:.2f}%" .format(100. * len(xea) / (w * h)))
print("Edge fill B      \t{0:.2f}%" .format(100. * len(xeb) / (w * h)))
print("Passes           \t{}"       .format(repeats))
print("Time total       \t{0:.1f}s" .format(stopwatch))
print("Time per simi    \t{0:.0f}ms".format(stopwatch / repeats * 1e6))
print("Best  match score\t{0:.0f}%" .format(100. * best ['simi']))
print("Worst match score\t{0:.0f}%" .format(100. * worst['simi']))
print("Best  match A    \t({},{})"  .format(best['xa'], best['ya']))
print("Worst match B    \t({},{})"  .format(best['xb'], best['yb']))
print("")

plt.close('all')
h, w = Ka.shape[:2]
fig = algo.big_figure('MuddyMorph Proto - Edge', w * 3, h * 2)

plt.subplot(2, 4, 1)
plt.imshow(Ka)
plt.axis('image')
plt.title('A', fontweight='bold')

plt.subplot(2, 4, 5)
plt.imshow(Kb)
plt.axis('image')
plt.title('B', fontweight='bold')

plt.subplot(2, 4, 2)
plt.imshow(Ea, cmap=plt.cm.gray)
plt.axis('image')
plt.plot(xea, yea, 'w.', markersize=2, alpha=.2)
plt.plot(best ['xa'], best ['ya'], 'go')
plt.plot(worst['xa'], worst['ya'], 'ro')
plt.title('A - Edge', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 6)
plt.imshow(Eb, cmap=plt.cm.gray)
plt.axis('image')
plt.plot(xeb, yeb, 'w.', markersize=2, alpha=.2)
plt.plot(best ['xb'], best ['yb'], 'go')
plt.plot(worst['xb'], worst['yb'], 'ro')
plt.title('B - Edge', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 3)
plt.imshow(best['Qa'], interpolation='none')
plt.title('A - Best', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 7)
plt.imshow(best['Qb'], interpolation='none')
plt.title('B - Best', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 4)
plt.imshow(worst['Qa'], interpolation='none')
plt.title('A - Worst', fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 8)
plt.imshow(worst['Qb'], interpolation='none')
plt.title('B - Worst', fontweight='bold')
plt.xticks([])
plt.yticks([])

fig.tight_layout()
