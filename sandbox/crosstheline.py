#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do two line segments intersect?
"""

import numpy as np
import matplotlib.pyplot as plt

def crosstheline(xa, ya, xb, yb, xc, yc, xd, yd):
    """
    Check whether line segments AB and CD intersect, and return True if they do.
    Naturally this is done by validating a bunch of determinants.
    Touching start (A & C) or end points (B & D) also count as an intersection.
    
    Coordinates must be integers;
    the equality check will fail in case of floats.
    
    http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    
    def ccw(xa, ya, xb, yb, xc, yc):
        # Check whether point C is counter-clockwise with respect to A-B.
        return (yc - ya) * (xb - xa) > (yb - ya) * (xc - xa)
    
    # Same start point?
    if xa == xc and ya == yc: return True
    if xb == xd and yb == yd: return True
    
    # Are A & B at different sides of C-D?
    ok1 = ccw(xa, ya, xc, yc, xd, yd) != ccw(xb, yb, xc, yc, xd, yd)
    
    # Are C & D at different sides of A - B?
    ok2 = ccw(xa, ya, xb, yb, xc, yc) != ccw(xa, ya, xb, yb, xd, yd)
    
    return ok1 and ok2


# Run a bunch of elementary test cases
print("Test 1\tIntersect\t", end="")
print(crosstheline(0, 0, 1, 1, 0, 1, 1, 0))

print("Test 2\tParallel\t", end="")
print(crosstheline(0, 0, 1, 0, 0, 1, 1, 1))

print("Test 3\tSame start\t", end="")
print(crosstheline(0, 0, 1, 0, 0, 0, 0, 1))

print("Test 4\tSame end\t", end="")
print(crosstheline(0, 0, 1, 1, 1, 0, 1, 1))

print("Test 5\tSeparate\t", end="")
print(crosstheline(0, 0, 1, 1, 0, 1, 2, 4))


# Run 100 random tests and plot the result
print("Running 100 random tests ... ")
fig = plt.figure(figsize=(12, 12))
for i in range(100):
    coords = np.round(np.random.rand(8) * 10).astype(int)
    cross  = crosstheline(*coords)
    clr    = 'orange' if cross else 'blue'
    print("{}\t".format(i + 1) + str(coords) + " --> " + str(cross))
    
    plt.subplot(10, 10, i + 1)
    plt.title(i + 1, fontweight='bold')
    plt.plot([coords[0], coords[2]], [coords[1], coords[3]], '-', color=clr)
    plt.plot([coords[4], coords[6]], [coords[5], coords[7]], '-', color=clr)
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])

fig.tight_layout()

    
