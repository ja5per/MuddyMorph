#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A jolly collection of assist functions & algorithms for image analysis and manipulation;
loading and saving bitmaps, identifying key points, planning motion trajectories,
compositing and deforming key frames, and so forth.

Spatially warped cross-dissolving for the masses!

Usage
-----
Recommended import convention
>>> import muddymorph_algo as algo

See also
--------
This module offers the fundamental building blocks for all required steps.
For convenience functions and logistics see *muddymorph_go*.

Notes
-----
https://en.wikipedia.org/wiki/Morphing
"""

# Project meta info
# (For documentation, logging, splash screens, and so forth)
__author__   = 'Jasper Menger'
__version__  = '1.0 Amazing Ant'
__revision__ = '2017-Nov'

# Dependencies
import warnings
import numpy as np
import bottleneck as bn
from os import path
from PIL import Image, ImageFile
from skimage import exposure
from skimage.filters import scharr
from skimage.color import label2rgb
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.feature import ORB, CENSURE, BRIEF, match_descriptors
from scipy.ndimage import binary_dilation, gaussian_filter
import matplotlib.pyplot as plt

# One-liners
# - Find indices of non-zero elements of a vector.
# - Is a variable an array (bitmap or vector)?
# - Pick a random number between two limits.
# - Euclidian distance between two 2D points.
find       = lambda x: np.ravel(np.nonzero(x))
isarray    = lambda x: type(x) == type(np.array([[0]]))
pick_float = lambda lim: lim[0] + (lim[1] - lim[0]) * np.random.rand()
pick_int   = lambda lim: int(round(pick_float(lim)))
distance   = lambda xa, ya, xb, yb: np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)

# Boost blob render speed by storing clone brush tips in this container
brushes = dict()


def motion_profile(t, flavor='linear'):
    """
    Relative distance travelled as a function of relative time.

    Usage
    -----
    >>> d = motion_profile(t, flavor)

    Parameters
    ----------
    t : float or array
        Relative time that has passed between two key frames,
        subject to 0 <= t <= 1.
    flavor : str, optional
        The kind of motion the blobs are making between these frames. Options:
            - linear. Travel with constant speed. d = t.
            - doublestop. During the first half-time accelerate from
              standstill, and during the second half-time decelerate to
              standstill. d = 3t**2 - 2t**3.
            - accelerate. Start from rest and accelerate to constant speed.
              d = 2t**2 - t**3.
            - decelerate. Start from constant speed and decelerate to rest.
              d = t + t**2 - t**3.

    Returns
    -------
    Relative distance travelled, somewhere between 0 (start) and 1 (stop).

    Notes
    -----
    Alternative functions that were considered but not implemented:
        - Double stop. d = 1/2 * sin(pi * (t - 1/2)) + 1/2
        - Accelerate.  d = t ** (2 - t)
        - Decelerate.  d = t ** (1 + t)
    """
    if np.any(t < 0) or np.any(t > 1):
        raise ValueError('Relative time must be within bounds 0 <= t <= 1')
    
    flavor = flavor.lower().strip()

    if flavor == 'linear':
        d = t
    
    elif flavor == 'doublestop':
        d = 3 * t ** 2 - 2 * t ** 3
    
    elif flavor == 'accelerate':
        d = 2 * t ** 2 - t ** 3
    
    elif flavor == 'decelerate':
        d = t + t ** 2 - t ** 3
    
    else:
       raise ValueError('Unsupported motion profile "%s"' % flavor)

    return d


def fade_profile(t, d, tau=1.):
    """
    Fade factor as a function of relative time.
    
    Usage
    -----
    >>> f = fade_profile(t, d, tau)
    
    Parameters
    ----------
    t : array
        Relative time that has passed between two key frames,
        subject to 0 <= t <= 1.
    d : array
        Output of *motion_profile*
    tau : float, optional
        Crossfade factor, subject to 0 <= tau <= 1.
    
    Notes
    -----
    By default the fade factor will be equal to the relative distance travelled.
    At f=0 only the (possibly warped) start frame A will be shown,
    and at f=1 only the (possibly deformed) stop frame B is visible.
    
    However if the fade period tau is set lower than one,
    then a talud profile will be applied with the following function:
        
        - f(t) = 0 for t < (1 - tau) / 2.
        - f(t) = 1 for t > (1 + tau) / 2.
        - f(t) = (t - 1/2) / tau + 1/2 in between.
    
    And for the special case of tau = 0:
        
        - f(t) = 0 for t <= 1/2.
        - f(t) = 1 for t >  1/2.
    
    In case of a messy looking midpoint but decent trajectories tau=0
    is recommended, as this will prevent edge smearing.
    """
    if tau >= 1:
        f = d
    
    elif tau <= 0:
        f = t * 0
        f[t > 0.5] = 1
    
    else:
        f = (t - 0.5) / tau + 0.5
        f = np.clip(f, 0, 1)
    
    return f


def crisscross(xa, ya, xb, yb, xc, yc, xd, yd):
    """
    Check whether two line segments A-->B and C-->D intersect,
    and return True if they do.

    Crossing node paths are an eyesore, and have to be avoided at all cost.

    Usage
    -----
    >>> guilty = crisscross(xa, ya, xb, yb, xc, yc, xd, yd)

    Notes
    -----
    Naturally this check is performed by validating a bunch of determinants.
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


def flotate_bitmap(M):
    """
    Convert integers (0-255) to floats (0-1).
    This way subsequent edit operations won't cause discrete artefacts,
    and things like scaling and applying thresholds become more intuitive.
    """
    if M.dtype == np.dtype('uint8'):
        return M.astype(float) / 255.
    else:
        return M


def read_image_dimensions(filename):
    """
    Obtain image dimensions in pixels from the file header
    (typically first KB).
    
    Usage
    -----
    >>> height, width = read_image_dimensions(filename)
    """
    fid = open(filename, 'rb')
    p = ImageFile.Parser()
    
    while True:
        data = fid.read(1024)
        if not data:
            break
        p.feed(data)
        if p.image:
            return p.image.size[1], p.image.size[0]

    fid.close()
    return None


def load_rgba(imagefile):
    """
    Load a bitmap and store it as a floating point RGBa array,
    regardless of file format and content (colour/greyscale/b&w).

    Usage
    -----
    >>> M = load_rgba(imagefile)

    Returns
    -------
    The array will have dimensions M[y, x, c], with channel:

    ===   ===   =======
    #     id    content
    ===   ===   =======
    0     R     Red
    1     G     Green
    2     B     Blue
    3     a     Alpha
    ===   ===   =======

    Data type will be floating point (values between 0 - 1).
    """
    Mi = plt.imread(imagefile)
    Mf = flotate_bitmap(Mi)

    if Mf.ndim == 2:
        # From greyscale or black-n-white to RGBa
        Ma = np.ones((Mf.shape[0], Mf.shape[1], 4))
        for c in range(3):
            Ma[:, :, c] = Mf

    elif Mf.shape[2] == 3:
        # From RGB to RGBa
        Ma = np.ones((Mf.shape[0], Mf.shape[1], 4))
        Ma[:, :, :3] = Mf

    else:
        # No need to do anything
        Ma = Mf

    return Ma


def is_opaque(M, eps=1e-3):
    """
    Detect whether the image is fully opaque
    (alpha channel practically one for all pixels).
    """
    return np.all(M[:, :, 3] >= 1 - eps)

    
def most_frequent_value(x):
    """
    Returns the most frequent value in given list or array.
    
    Usage
    -----
    >>> v = most_frequent_value(x)
    
    Notes
    -----    
    Special cases:
        - If all values are equally common, return the first value.
        - If multiple (but not all) values are equally common,
          return the lowest value of these.
        - In case of non-array-ish input (float or string or whatever),
          just return the input as-is.
        - In order to stay JSON compatible,
          data type boolean / integer / string / float is enforced.
    """
    xa = np.array([x]).flatten()
    xu = np.unique(xa)
    n  = np.array([sum(xa == xi) for xi in xu])
    
    if len(np.unique(n)) > 1:
        j = bn.nanargmax(n)
        v = xu[j]
    else:
        v = xa[0]
    
    t = str(type(v))
    
    if   'bool' in t : v = bool (v)
    elif 'int'  in t : v = int  (v)
    elif 'str'  in t : v = str  (v)
    else             : v = float(v)
    
    return v


def save_rgba(M, imagefile, quality=90):
    """
    Save RGBa bitmap float array to bitmap file (PNG or JPG or whatever).

    Usage
    -----
    >>> save_rgba(M, imagefile, quality)
    """
    ext = path.splitext(imagefile)[-1][1:].lower()
    if ext == 'jpg' or ext == 'jpeg':
        
        # Make it as flat as a pancake
        Mi = flatten_bitmap(M)
        Mi = np.round(Mi * 255)
        Mi = np.clip(Mi, 0, 255)
        Mi = Mi.astype(np.uint8)
        
        # Diss the alpha channel
        # (Penalty would be an OSError)
        if Mi.ndim >= 3: Mi = Mi[:, :, :3]
        
        img = Image.fromarray(Mi)
        img.save(imagefile, quality=quality)
        
    else:
        plt.imsave(imagefile, M)


def save_grey(M, imagefile):
    """
    Save bitmap array to image file (PNG or whatever).
    Type of array can be either boolean (b&w) or float (greyscale).
    Useful for saving silhouettes, edge contours, and keypoint descriptors.

    Usage
    -----
    >>> save_grey(M, imagefile)
    """

    # Convert to unsigned integer greyscale
    if M.dtype == bool or (M.max() == 1 and len(np.unique(M)) == 2):
        # Input is binary black and white
        grey = M * 255
    else:
        # Input is greyscale
        grey  = M.astype(float)
        grey *= 255. / np.max(M)
    
    # Ensure we have only valid intensities in this joint
    grey = np.clip(grey, 0, 255)
    grey = grey.astype(np.uint8)

    # Convert to image and save
    img = Image.fromarray(grey)
    img.save(imagefile)


def composite_bitmaps(A, B=None, backcolor=None):
    """
    Make a composite of bitmap A over bitmap B and/or solid background color.
    Both A & B should be RGBa arrays.
    """
    if not backcolor is None:
        G = np.ones_like(A)
        for c in range(3):
            G[:, :, c] = backcolor[c] / 255.
        if B is None:
            B = G
        else:
            B = composite_bitmaps(B, G)

    M = np.zeros_like(A)

    t_a = A[:, :, 3]
    t_b = B[:, :, 3]

    for c in range(3):
        M[:, :, c] = A[:, :, c] * t_a + B[:, :, c] * t_b * (1 - t_a)
    
    M[:, :, 3] = t_a + t_b * (1 - t_a)

    return M


def flatten_bitmap(A, backcolor=(0, 0, 0)):
    """
    Flatten a bitmap; apply a solid background color.
    """
    return composite_bitmaps(A, backcolor=backcolor)


def make_background(h, w, backcolor=(0, 0, 0), transparent=True, backdrop=''):
    """
    Generate a solid background RGBa array.
    This can be either transparent, a solid color, or a backdrop image.

    Usage
    -----
    >>> G = make_background(h, w, backcolor, transparent, backdrop)
    """

    G = np.zeros((h, w, 4))

    if not backcolor is None:
        for c in range(len(backcolor)):
            G[:, :, c] = backcolor[c] / 255.

    if not transparent: G[:, :, 3] = 1.

    if isarray(backdrop):
        G = composite_bitmaps(backdrop, G)
    elif not backdrop is None and len(backdrop) > 0 and path.isfile(backdrop):
        B = load_rgba(backdrop)
        G = composite_bitmaps(B, G)

    return G


def big_figure(figname, w, h, maxsize=(20., 10.), facecolor='white'):
    """
    Set up a large figure window,
    with canvas size aspect matching the given bitmap dimensions,
    and with axes spanning the full canvas.

    Usage
    -----
    >>> fig = big_figure(figname, w, h)

    Notes
    -----
    Because it is quite a hassle to set up a matplotlib widget in a stylish
    and appealing way (at least for nitwits like me), it is more appealing to
    generate a plot in the background, save it as a bitmap, and then show that
    bitmap in the GUI in a label widget or such.
    """
    aspect = 1. * w / h
    fig_w  = maxsize[0]
    fig_h  = fig_w / aspect

    if fig_h > maxsize[1]:
        fig_h = maxsize[1]
        fig_w = fig_h * aspect

    fig = plt.figure(figname, facecolor=facecolor, figsize=(fig_w, fig_h))
    plt.axes((0, 0, 1, 1))

    return fig


def desaturate(M, channel='average', blur=0, cutoff=0):
    """
    Desaturate an RGB or RGBa bitmap (3D array) to flat grayscale (2D array).

    These channel options are endorsed:

    ===========  =================================
    Channel      Formula
    ===========  =================================
    average      mean(R, G, B)
    luminosity   0.2126*R + 0.7152*G + 0.0722*B
    lightness    (max(R, G, B) + min(R, G, B)) / 2
    maximum      max(R, G, B)
    red          R
    green        G
    blue         B
    alpha        a
    greenscreen  0.5*R + 0.5*B
    bluescreen   0.5*R + 0.5*G
    ===========  =================================

    As a bonus, Gaussian blurring and adaptive histogram equalization
    can be added to the mix if so desired.
    """

    # Already greyscale?
    if M.ndim < 3:
        G = M

    else:
        # Make sure floating point operations are legit
        F = flotate_bitmap(M)

        channel = channel.lower().strip()

        if not channel in ('alpha', 'a'):
            F = flatten_bitmap(F)

        if channel in ('average', 'avg', 'mean'):
            G = bn.nanmean(F[:, :, :3], axis=2)

        elif channel in ('luminosity', 'lum', 'lumi'):
            G = np.squeeze(0.2126 * F[:, :, 0] + \
                           0.7152 * F[:, :, 1] + \
                           0.0722 * F[:, :, 2])

        elif channel in ('lightness', 'light'):
            G = 0.5 * np.min(F[:, :, :3], axis=2) + \
                0.5 * np.max(F[:, :, :3], axis=2)

        elif channel in ('maximum', 'max', 'maxi'):
            G = np.max(F[:, :, :3], axis=2)

        elif channel in ('rootmeansquare', 'rms'):
            G = np.squeeze(np.sqrt((F[:, :, 0] ** 2 + \
                                    F[:, :, 1] ** 2 + \
                                    F[:, :, 2] ** 2) / 3.))

        elif channel in ('red', 'r'):
            G = np.squeeze(F[:, :, 0])

        elif channel in ('green', 'g'):
            G = np.squeeze(F[:, :, 1])

        elif channel in ('blue', 'b'):
            G = np.squeeze(F[:, :, 2])

        elif channel in ('greenscreen', 'rb'):
            G = 0.5 * np.squeeze(F[:, :, 0]) + \
                0.5 * np.squeeze(F[:, :, 2])

        elif channel in ('bluescreen', 'rg'):
            G = 0.5 * np.squeeze(F[:, :, 0]) + \
                0.5 * np.squeeze(F[:, :, 1])

        elif channel in ('alpha', 'a'):
            if F.shape[2] < 4:
                G = np.ones((F.shape[0], F.shape[1]))
            else:
                G = np.squeeze(F[:, :, 3])

        else:
            raise ValueError('Invalid channel "{}"'.format(channel))

    if blur > 0: G = gaussian_filter(G, sigma=blur)
    
    G = np.clip(G, 0, 1)

    if cutoff > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = exposure.equalize_adapthist(G, clip_limit=cutoff)

    return G


def grid(M, x=None, y=None):
    """
    Set up a mesh grid covering a bitmap.

    Usage
    -----
    For all coordinates
    >>> X, Y = grid(M)

    For selected x & y levels only
    >>> X, Y = grid(M, x, y)

    Notes
    -----
    Primary use is to be able to quickly extract coordinates
    from a masked array, e.g. a probability map with zero zones excluded.
    """
    if x is None: x = np.arange(M.shape[1])
    if y is None: y = np.arange(M.shape[0])

    X, Y = np.meshgrid(x, y)

    return X, Y


def commie(S, X=None, Y=None, verbose=True):
    """
    Compute the center of mass (CoM) position and related properties for the
    given binary silhouette bitmap. These form a decent initial starting point
    for node trajectory assignment.

    Usage
    -----
    >>> com = commie(S)

    Parameters
    ----------
    Optionally supply mesh grid arrays X & Y, see *grid*.

    Returns
    -------
    Out comes a juicy dictionary:

    ======  =====  ===========================================================
    Field   Type   Content
    ======  =====  ===========================================================
    x       int    Center of mass X position.
    y       int    Center of mass Y position.
    r       float  Radius. Average distance of in-silhouette pixels to center.
    a       float  Angle of primary eigenvector in radians.
    relvar  float  Relative variance of primary eigenvector.
    vx      float  Primary eigenvector x component
    vy      float  Primary eigenvector y component
    ======  =====  ===========================================================

    Notes
    -----
    The relative variance is a confidence indicator for the angle. One may want
    to apply a threshold based on this (e.g. relvar >= 0.8), in order to prevent
    rotation being applied for (nearly) symmetric shapes, as these yield
    eigenvectors with arbitrary orientations.
    """

    # Start from scratch
    com = dict(x=0, y=0, r=1, a=0.0, relvar=0.0, vx=0.0, vy=0.0)

    # Find center of mass position
    x   = np.arange(S.shape[1])
    y   = np.arange(S.shape[0])
    sx  = np.sum(S, axis=0)
    sy  = np.sum(S, axis=1)
    ssx = np.sum(sx)
    ssy = np.sum(sy)

    # No silhouette shape at all?
    if ssx == 0 or ssy == 0:
        # Let the center of the frame be a symbolical CoM
        com['x'] = int(np.round(S.shape[1] / 2))
        com['y'] = int(np.round(S.shape[0] / 2))
    else:
        # Good to go
        xc = np.sum(x * sx) / ssx
        yc = np.sum(y * sy) / ssy
        com['x'] = int(np.round(xc))
        com['y'] = int(np.round(yc))
    
        # Find average distance to center of mass
        if X is None or Y is None: X, Y = grid(S, x, y)
        dx       = (X[S].flatten() - com['x'])
        dy       = (Y[S].flatten() - com['y'])
        r        = np.sqrt(dx ** 2 + dy ** 2)
        com['r'] = bn.nanmean(r)
    
        # Compute and sort eigenvectors (V) & eigenvalues (l)
        M        = np.column_stack((dx, dy))
        Mn       = M - np.array([com['x'], com['y']])
        S        = np.cov(Mn, rowvar=False)
        l, V     = np.linalg.eig(S)
        i        = np.argsort(l)[::-1]
        l, V     = l[i], V[:, i]
        relvar   = l ** 2
        relvar  /= relvar.sum()
    
        # What's your angle in this affair?
        com['vx'    ] = V[0,0]
        com['vy'    ] = V[1,0]
        com['a'     ] = np.arctan2(V[1,0], V[0,0])
        com['relvar'] = relvar[0]

    # Summarize the results
    if verbose:
        msg = "CoM x={x:d}, y={y:d}, r={r:.0f}, a={a:.3f}, relvar={relvar:.3f}"
        print(msg.format(**com))

    return com


def swirl(kp_a, com_a, com_b, d=1.0):
    """
    Apply translation, magnification, and rotation to a series of coordinates.
    Move from one center of mass frame (A) to the next one (B).

    Usage
    -----
    >>> kp_b = swirl(kp_a, com_a, com_b, d)

    Parameters
    ----------
    kp_a : int or float
        Coordinates [x, y]
    com_a : dict
        Source center of mass properties
    com_b : dict
        Destination center of mass properties
    d : float, optional
        Relative distance to travel, between 0 and 1.
        Use this for nice and swirly interpolation.

    Notes
    -----
    https://en.wikipedia.org/wiki/Transformation_matrix
    """

    # Do we need to do anything at all?
    if com_a is None or com_b is None or \
       d == 0 or com_a['r'] == 0 or com_b['r'] == 0: return kp_a

    # Interpolate if so required
    if d == 1:
        com_i = com_b
    else:
        com_i = dict()
        for item in com_b:
            com_i[item] = com_a[item] + d * (com_b[item] - com_a[item])

    # Determine relative magnification and rotation
    # Go for the most subtle rotation possible
    mag = com_b['r'] / com_a['r']
    rot = com_b['a'] - com_a['a']
    d90 = 0.5 * np.pi
    if abs(rot) > d90:
        for i in [-4, +4, -3, +3, -2, +2, -1, +1]:
            roti = rot - i * d90
            if abs(roti) < abs(rot):
                rot = roti
                break

    # Interpolate
    tx   = d * (com_b['x'] - com_a['x'])
    ty   = d * (com_b['y'] - com_a['y'])
    magi = d * (mag - 1) + 1
    roti = d * rot 
    
    # Totally transform those coordinates
    xm = magi * (kp_a[:, 0] - com_a['x'])
    ym = magi * (kp_a[:, 1] - com_a['y'])
    xr = xm * np.cos(roti) - ym * np.sin(roti)
    yr = xm * np.sin(roti) + ym * np.cos(roti)
    xb = xr + com_a['x'] + tx
    yb = yr + com_a['y'] + ty

    return np.column_stack((xb, yb))


def bordercolor(B, thickness=1):
    """
    Automatic background color detection;
    median value of the outermost pixels.
    """

    # Set up a mask for the outermost edge
    W = np.zeros(B.shape[:2], dtype=bool)
    for t in range(thickness):
        W[:, t     ] = 1
        W[:, -1 - t] = 1
        W[t     , :] = 1
        W[-1 - t, :] = 1

    # Fetch the median value for red, green, blue
    clr = []
    for c in range(3):
        E = np.squeeze(B[:, :, c])[W]
        clr.append(int(255 * bn.nanmedian(E)))

    # Done deal
    clr = tuple(clr)
    return clr


def edgy(B, channel='lightness', threshold=0.2,
         backcolor=None, linecolor=(0, 0, 0),
         dolines=False, doscharr=False, blur=2, invert=False):
    """
    Identify silhouette shape and edge pixels for the given bitmap.

    Usage
    -----
    >>> D, S, E = edgy(B, channel, threshold, ...)

    Parameters
    ----------
    B : image array
        The bitmap that is to be analysed.
    channel : str, optional
        Greytone channel that is to be applied to the difference with the
        background color, see *desaturate* for the various options.
    threshold : float, optional
        Intensity threshold for silhouette.
    backcolor : RGB tuple or None, optional
        Background color. Defaults to auto-pick, see *color_pick_border*.
    linecolor : RGB tuple, optional
        Color of the line art, if any.
    dolines : bool, optional
        Whether or not to exclude lines from the silhouette shape.
        When the overall shape is detected through alpha then for lines
        (opaque by definition) the lightness channel is used.
    docharr : bool, optional
        Whether or no to exclude contrast-rich edges as detected by the Scharr
        algorithm. Use the same channel as the *dolines* option.
    blur : int or float, optional
        Gaussian blur to soften the silhouette, set to zero to disable.
    invert : bool, optional
        Whether or not to invert grey tones prior to applying the threshold.

    Returns
    -------
    D : greyscale image array
        Difference with background
    S : boolean image array
        Silhouette shape
    E : boolean image array
        Edge pixel mask
    """

    # Automatic background color detection
    channel = channel.lower().strip()
    if backcolor is None or len(backcolor) == 0:
        if 'screen' in channel or channel == 'alpha':
            backcolor = (0, 0, 0)
        else:
            backcolor = bordercolor(B)

    # Difference with the background
    D = B.copy()
    if not channel.startswith('a'):
        for c in range(3):
            D[:, :, c] = abs(D[:, :, c] - backcolor[c] / 255.)

    D = desaturate(D, channel)

    # A bit of blur works wonders to suppress noise
    if blur > 0: D = gaussian_filter(D, sigma=blur)
    
    # Remove lines from silhouette
    channel2 = 'lightness' if channel.startswith('a') else channel
    if dolines:        
        D2 = B.copy()
        for c in range(3):
            D2[:, :, c] = abs(D2[:, :, c] - linecolor[c] / 255.)

        D2     = desaturate(D2, channel2)
        low    = D2 < D
        D[low] = D2[low]
    
    # Remove contrasty edges from silhouette
    if doscharr:
        D2     = gaussian_filter(B, sigma=blur)
        D2     = scharr(desaturate(D2, channel2))
        D2     = 1 - D2 / D2.max()
        low    = D2 < D
        D[low] = D2[low]

    # Invert is so desired, and apply the threshold
    if invert: D = 1 - D
    S = D >= threshold

    if S.any():
        S2 = binary_dilation(S)
        E  = S != S2
    else:
        E = S.copy()

    return D, S, E


def similarity(A, B, xa, ya, xb, yb,
               simisize=10, flatline=1e-2, returnboxes=False):
    """
    Asses the similarity between spot (xa, ya) in bitmap A
    and location (xb, yb) in bitmap B.

    Usage
    -----
    >>> simi = similarity(A, B, xa, ya, xb, yb, hr)

    Parameters
    ----------
    A, B : bitmap arrays
        Key frame images.
    xa, ya, xb, yb : int
        Index coordinates of both bitmaps; A[ya, xa] en B[yb, xb].
    simisize : int, optional
        Half range dimension of the square evaluation zone.
    flatline : float, optional
        Minimal standard deviation for both evaluation patches.
        Return a zero similarity score if this condition is not met.
        This is to prevent empty patches with high similarity scores.
    returnboxes : bool, optional
        Whether to return the bounding boxes in addition to the score.

    Returns
    -------
    Squared correlation coefficient (R2), a number between 0 and 1.

    Notes
    -----
    The higher the similarity score, the more eligible the given point pair is
    to serve as a morph node. Local similarity is defined as the Pearson's
    correlation coefficient across all RGBa channels for a small square zone
    around the given coordinates. As such it is sensitive to local shape
    likeness, but less so to constant differences in color or lightness.
    """

    # Get a grip
    h , w  = A.shape[:2]
    xa, ya = int(xa), int(ya)
    xb, yb = int(xb), int(yb)

    # The bounding box has to shrink in vicinity of image edges
    dxm = min(simisize, xa, xb)
    dym = min(simisize, ya, yb)
    dxp = min(simisize, w - xa - 1, w - xb - 1)
    dyp = min(simisize, h - ya - 1, h - yb - 1)

    # Cut out the evaluation boxes
    Qa = A[(ya - dym):(ya + dyp + 1), (xa - dxm):(xa + dxp + 1)]
    Qb = B[(yb - dym):(yb + dyp + 1), (xb - dxm):(xb + dxp + 1)]
    
    # Verify both patches contain more than just a flat color
    if np.std(Qa) < flatline or np.std(Qb) < flatline:
        simi = 0.
    else:
        # Correlate R & G & B in one fell swoop
        simi = np.corrcoef(Qa.flatten(), Qb.flatten())[0, 1] ** 2

    if returnboxes:
        return simi, Qa, Qb
    else:
        return simi


def seed(h, w, com_a=None, com_b=None):
    """
    Initialize the node trajectory table;
    always start with the four corners and center of mass.

    Usage
    -----
    >>> nodes = seed(h, w, com_a, com_b)

    Returns
    -------
    Trajectory array with rows [xa, ya, xb, yb].
    """
    base = np.array([[0    , 0    , 0    , 0    ],
                     [w - 1, 0    , w - 1, 0    ],
                     [w - 1, h - 1, w - 1, h - 1],
                     [0    , h - 1, 0    , h - 1]])

    if not com_a is None and not com_b is None:
        base = np.row_stack((base, [com_a['x'], com_a['y'],
                                    com_b['x'], com_b['y']]))

    return base


def notsofast(nodes, maxmove=1000, com_a=None, com_b=None):
    """
    Discard long paths with a simple distance threshold filter.
    Path distance is evaluated both in the absolute sense,
    and relative to the CoM swirl.

    Usage
    -----
    >>> keep = notsofast(nodes, maxmove, com_a, com_b)
    """
    
    move = distance(nodes[:, 0], nodes[:, 1],
                    nodes[:, 2], nodes[:, 3])

    keep = move <= maxmove
    
    if not com_a is None and not com_b is None:
        posi   = swirl(nodes, com_a, com_b)
        swoosh = distance(posi [:, 0], posi [:, 1],
                          nodes[:, 2], nodes[:, 3])
        keep   = (keep) & (swoosh <= maxmove)

    return keep


def gettogether(nodes, simi=None, mindist=10):
    """
    Merge nodes that are a bit too close together (effectively duplicates).
    Return the average position and similarity score for distinct locations.
    Get together, right now, over me!

    Usage
    -----
    >>> nodes_clean, simi_clean = gettogether(nodes, simi, mindist)

    Parameters
    ----------
    Node trajectories, similarity scores, and minimal distance threshold.
    """

    noodle = []
    sumi   = []
    free   = np.ones(len(nodes), dtype=bool)
    i      = 0
    
    if simi is None: simi = np.ones(len(nodes)) * np.nan

    while i < len(free) and np.any(free):
        while not(free[i]): i += 1

        da = distance(nodes[i, 0], nodes[i, 1], nodes[:, 0], nodes[:, 1])
        db = distance(nodes[i, 2], nodes[i, 3], nodes[:, 2], nodes[:, 3])
        j  = find((free) & (da < mindist) & (db < mindist))
        mu = bn.nanmean(nodes[j, :], axis=0).astype(int)

        noodle.append(mu)
        sumi.append(bn.nanmean(simi[j]))
        free[j] = False

    noodle = np.array(noodle)
    sumi = np.array(sumi)
    
    return noodle, sumi
 
    
def spawn(E, kp0, n=None, r_min=1, X=None, Y=None):
    """
    Evenly spread out points over the silhouette edge.

    Usage
    -----
    >>> kp = spawn(E, kp0, n)

    Parameters
    ----------
    E : boolean bitmap array
        Silhouette edge points, see *edgy*.
    kp0 : array
        Predefined keypoint coordinates [x, y], for example from *seed*.
    n : int, optional
        Goal for the number of points to pick.
        Defaults to 10% of the edge points.
    r_min : int, optional
        Minimal distance between points.
    X, Y : arrays, optional
        Mesh grid coordinates, see *grid*.

    Returns
    -------
    Key point coordinates [x, y],
    being bitmap row and column indices respectively.
    """

    # Default target
    if n is None: n = int(np.floor(E.sum() * 0.1))

    # List all pickable positions
    if X is None or Y is None: X, Y = grid(E)
    xe, ye = X[E], Y[E]
    if len(xe) <= n:
        return np.column_stack((xe, ye))

    # Initial force field
    # Prevent flippy results by a miniscule bias towards the origin
    h, w  = E.shape
    force = 1e-6 * (xe - 0.5 * w) / w ** 2 + \
            1e-6 * (ye - 0.5 * h) / h ** 2
    for x, y in kp0:
        r2 = (xe - x) ** 2 + (ye - y) ** 2
        force += 1. / (r2 + 1.)
        force[r2 <= r_min ** 2] = np.inf

    # Add key points as long as it makes sense
    kp = []
    while len(kp) < n and np.any(np.isfinite(force)):

        # Find furthest free location
        pick = bn.nanargmin(force)
        x, y = xe[pick], ye[pick]
        kp.append([x, y])

        # Update force field
        r2 = (xe - x) ** 2 + (ye - y) ** 2
        force += 1. / (r2 + 1.)
        force[r2 <= r_min ** 2] = np.inf

    return np.array(kp)


def measure_angle(xa, ya, xb, yb, xc, yc, normalize=True):
    """
    Compute the angle of the connection through points A, B, C.

    Usage
    -----
    >>> phi = angle(xa, ya, xb, yb, xc, yc)

    Returns
    -------
    The angle phi between lines AB & BC in radians.
    If the points are not unique not-a-number is returned.
    
    By default a normalized number is returned;
    the smallest positive angle of the
    continuous intersecting lines.

    Notes
    -----
    Can come in handy for selecting suitable locations for new nodes,
    and for computing the smoothness of a drawing.

    See also
    --------
    Formula follows from vector notation of the cosine law.
    http://en.wikipedia.org/wiki/Law_of_cosines
    """

    x_ba, y_ba = (xa - xb, ya - yb)
    x_bc, y_bc = (xc - xb, yc - yb)

    if ((x_ba, y_ba) == (0, 0)) or ((x_bc, y_bc) == (0, 0)): return np.nan

    p1  = x_ba * x_bc + y_ba * y_bc
    p2  = (x_ba ** 2 + y_ba ** 2) * (x_bc ** 2 + y_bc ** 2)
    phi = np.arccos(1. * p1 / np.sqrt(p2))
    
    if normalize:
        phi = abs(phi)
        phi = min(phi, np.pi - phi)

    return phi


def proximatch(Ka, Kb, Ea, kp_a, kp_b, com_a=None, com_b=None,
               neighbours=3, simisize=20, similim=0, n=100):
    """
    Find matching key points based primarily on proximity.
    Amongst the nearest neighbours favor
    high similarities (good content correlation)
    and perpendicular angles (low risk of crossing path entanglement).

    Usage
    -----
    >>> nodes, simi = proximatch(Ka, Kb, kp_a, kp_b, com_a, com_b)

    Parameters
    ----------
    Ka, Kb : bitmap arrays
        Key frames A & B.
    Ea: boolean bitmap array
        Silhouette edges for A.
    kp_a, kp_b : coordinate arrays
        Key points for A & B.
    com_a, com_b : dictionaries, optional
        Center of mass properties for A & B.
        If supplied, translation & rotation & zoom are taken into account.
    neighbours : int, optional
        The number of closest neighbours to consider,
        both directly and after translation & rotation & magnification.
    simisize : int, optional
        Half width of bounding box for similarity evaluation.
    similim : float, optional
        Minimal similarity score for a valid match.
    n : int, optional
        Limit on the number of nodes.

    Returns
    -------
    Node coordinates [xa, ya, xb, yb] and associated similarity scores,
    ranked from highest to lowest score.
    """

    # Prepare for battle
    nodes   = []
    simi    = []
    xy_a    = kp_a.copy()
    xy_b    = kp_b.copy()
    n_max   = max([len(kp_a), len(kp_b)])
    xy_at   = swirl(kp_a, com_a, com_b)
    dotrans = np.any(xy_a - xy_at)
    h, w    = Ka.shape[:2]

    # Find close neighbours that look alike
    while len(xy_a) > 0 and len(xy_b) > 0 and len(nodes) < n_max:
        
        # Locate neighbours
        d1 = distance(xy_a[0, 0], xy_a[0, 1], xy_b[:, 0], xy_b[:, 1])
        if dotrans:
            d2 = distance(xy_at[0, 0], xy_at[0, 1], xy_b[:, 0], xy_b[:, 1])
            j  = np.unique(np.append(np.argsort(d1)[:neighbours],
                                     np.argsort(d2)[:neighbours]))
        else:
            j = np.argsort(d1)[:neighbours]

        # Evaluate similarity
        ss = []
        for k in j:
            ss.append(similarity(Ka, Kb,
                                 xy_a[0, 0], xy_a[0, 1],
                                 xy_b[k, 0], xy_b[k, 1],
                                 simisize=simisize))
        
        # Measure local silhouette angle (of A)
        dxm = min(simisize, xy_a[0, 0])
        dym = min(simisize, xy_a[0, 1])
        dxp = min(simisize, w - xy_a[0, 0] - 1)
        dyp = min(simisize, h - xy_a[0, 1] - 1)
        Q   = Ea[(xy_a[0, 1] - dym):(xy_a[0, 1] + dyp + 1),
                 (xy_a[0, 0] - dxm):(xy_a[0, 0] + dxp + 1)]
        if Q.sum() >= 3:
            ting   = commie(Q, verbose=False)
            vx, vy = ting['vx'], ting['vy']
        else:
            vx, vy = 0., 0.
        
        # Assess perpendicularity of close neighbours (in B)
        pp = []
        for k in j:
            phi = measure_angle(xy_b[k, 0]     , xy_b[k, 1]     ,
                                xy_a[0, 0]     , xy_a[0, 1]     ,
                                xy_a[0, 0] + vx, xy_a[0, 1] + vy)
            perp = phi * 2 / np.pi if np.isfinite(phi) else 0.
            pp.append(perp)

        # Pick our favourite neighbour
        best = 0
        for k in range(1, len(ss)):
            if ss[k] > ss[best] and pp[k] > pp[best]:
                best = k
        k = j[best]

        # Add result to the collection
        nodes.append([xy_a[0, 0], xy_a[0, 1],
                      xy_b[k, 0], xy_b[k, 1]])
        simi.append(np.max(ss))
        
        # Remove chosen points from candidate lists
        xy_a  = xy_a [1:, :]
        xy_at = xy_at[1:, :]
        xy_b  = np.delete(xy_b, (k), axis=0)

    # Convert to array for easy processing
    nodes = np.array(nodes)
    simi  = np.array(simi)

    # The best of the best (apply hard similarity threshold)
    keep = simi >= similim
    nodes, simi  = nodes[keep], simi[keep]
    if not n is None and n < len(simi):
        nodes, simi = nodes[:n], simi[:n]

    return nodes, simi
    
    
def cornercatch(K, target=500, algorithm='orb',
                channel='lightness', blur=2, cutoff=0.02,
                orb_threshold=0.08, censure_mode='STAR', censure_patch=49):
    """
    Keypoint extraction by means of corner detection.

    Usage
    -----
    >>> kp, dc = cornercatch(K, target, algorithm)

    Parameters
    ----------
    K : bitmap array
        Key frame image.
    channel : str, optional
        Greyscale mode.
    blur : int or float, optional
        A bit of Gaussian blur for noise suppression.
    cutoff : float, optional
        Intensity cutoff threshold for adaptive histogram equalization.
    algorithm : str, optional
        Choose between ORB and CENSURE.
    target : int, optional
        The number of desired keypoints.
    orb_threshold : float, optional
        Fast threshold for ORB, usually in range 0 - 0.2.
    censure_mode : str, optional
        Choose between DoB, Octagon, or STAR.
    censure_patch : int, optional
        Patch size of BRIEF descriptors of CENSURE keypoints.

    Returns
    -------
    Key point coordinates [x, y] and associated binary descriptor array.
    """

    # Convert to grayscale and enhance contrast
    G = desaturate(K, channel, blur, cutoff)

    # Detect those pesky points
    method = algorithm.lower().strip()
    if method == 'orb':
        detector = ORB(n_keypoints=target,
                       fast_threshold=orb_threshold)
        detector.detect_and_extract(G)

        descriptors = detector.descriptors
        keypoints = detector.keypoints

    elif method == 'censure':
        detector = CENSURE(mode=censure_mode)
        detector.detect(G)
        keypoints = detector.keypoints[:target, :]

        extractor = BRIEF(patch_size=censure_patch, sigma=0)
        extractor.extract(G, keypoints)
        descriptors = extractor.descriptors

    else:
        raise ValueError('Unfamiliar with algorithm "{}"'.format(algorithm))

    # Watch out! Scikit keypoint column convention is [y, x]
    kp = keypoints[:, [1, 0]]
    kp = np.round(kp).astype(int)

    return kp, descriptors

    
def matchpoint(Ka, Kb, kp_a, kp_b, dc_a, dc_b, simisize=20, similim=0.3):
    """
    Find matching key points primarily based on their binary descriptors.
    Keep only matches with a satisfying similarity score.

    Usage
    -----
    >>> nodes, simi = matchpoint(Ka, Kb, kp_a, kp_b, dc_a, dc_b)
    
    Parameters
    ----------
    The similarity evaluation zone setting
    double times as a minimal distance criterion.
    """

    # Match key points
    matches = match_descriptors(dc_a, dc_b)
    nodes   = np.column_stack((kp_a[matches[:, 0]],
                               kp_b[matches[:, 1]]))
    
    # Keep a respectable distance, no cluttering please
    nodes, _ = gettogether(nodes, mindist=simisize)

    # Evaluate similarity for all distinct matches
    simi = []
    for i in range(len(nodes)):
        simi.append(similarity(Ka, Kb,
                               xa=nodes[i, 0], ya=nodes[i, 1],
                               xb=nodes[i, 2], yb=nodes[i, 3],
                               simisize=simisize))
    simi = np.array(simi)

    # Apply similarity threshold
    if similim > 0:
        keep  = simi >= similim
        nodes = nodes[keep, :]
        simi  = simi[keep]

    return nodes, simi


def straightenup(nodes):
    """
    Discard rogue nodes that cross paths with their law abiding fellows.
    For all trajectories that overlap, keep only the shortest one.

    Usage
    -----
    >>> keep = straightenup(nodes)
    
    Returns
    -------
    Mask vector, True only for the straight non-doublecrossing nodes.
    """

    move = distance(nodes[:, 0], nodes[:, 1],
                    nodes[:, 2], nodes[:, 3])

    free = np.ones(len(nodes), dtype=bool)
    keep = np.ones_like(free)
    i    = 0

    while i < len(free) and np.any(free):
        while not(free[i]): i += 1
        free[i] = False

        iscross = np.zeros_like(keep)
        for j in find(keep):
            if j != i: iscross[j] = crisscross(*nodes[i, :],
                                               *nodes[j, :])

        if np.any(iscross):
            iscross[i]  = True
            scrap       = np.argmax(move * iscross)
            keep[scrap] = False
            free[scrap] = False

    return keep


def complot(com, color=(0, .5, 1.), segments=360, ax=None):
    """
    Visualize center of mass properties as a steering wheel symbol.
    """
    
    if ax is None: ax = plt.gca()

    # Start with unit coordinates in this reference frame
    unit = dict(x=0, y=0, a=0, r=1)

    # Circle coordinates
    a    = np.linspace(0, 2 * np.pi, segments)
    xyoo = np.column_stack((np.cos(a), np.sin(a)))
    xyo  = swirl(xyoo, unit, com)

    # Primary and secondary spoke coordinates
    xy1 = swirl(np.array([[-1,  0], [+1,  0]]), unit, com)
    xy2 = swirl(np.array([[ 0, -1], [ 0, +1]]), unit, com)

    # Draw that very symbolic symbol
    ax.plot(xy1[:,0], xy1[:,1], '-', color=color, linewidth=0.5)
    ax.plot(xy2[:,0], xy2[:,1], ':', color=color, linewidth=0.5)
    ax.plot(xyo[:,0], xyo[:,1], '-', color=color, linewidth=2  )
    ax.plot(com['x'], com['y'], '.', color=color, markersize=9 )


def edgeplot(D, S, E,
             com=None, X=None, Y=None, ax=None,
             color_in=(0.,0.5,1.), color_out=(1., 1., 1.),
             color_com=(0, 1., 0.), alpha=0.3):
    """
    Show the result of edge detection. Produce an overlay plot of:

        1. Difference map (intensity)
        2. Silhouette shape (color)
        3. Edges (highlight)
        4. CoM (symbol)

    Usage
    -----
    >>> edgeplot(D, S, E)
    """
    if ax is None: ax = plt.gca()
    if X is None or Y is None: X, Y = grid(E)

    if not S is None and np.any(S):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = label2rgb(S.astype(int), D, alpha=alpha,
                          colors=[color_out, color_in])
        ax.imshow(L)
    else:
        ax.imshow(D, cmap=plt.cm.gray, vmin=0, vmax=1)
    
    if not E is None and np.any(E):
        xe, ye = X[E].flatten(), Y[E].flatten()
        ax.plot(xe, ye, '.', color=color_in,
                markersize=2, label='edge')
    
    if not com is None:
        complot(com, color=color_com, ax=ax)

    ax.axis((0, X[-1,-1], Y[-1,-1], 0))


def movemap(Da, Db, nodes,
            com_a=None, com_b=None, tweens=12,
            clr_a=(0.,.5,1.), clr_b=(1.,0.5,0.), ax=None):
    """
    Plot node trajectories on top of an onion skin background.
    The first four nodes are ignored, as those are supposed to be frame corners.
    """

    # Start with an overlay of both difference bitmaps
    if ax is None: ax = plt.gca()
    G = 0.25 * (1 - Da) + 0.25 * (1 - Db) + 0.5
    ax.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)
    
    # Plot frame to frame trajectories as line segments
    # Do not mess with fancy motion profiles here. Distance equals time.
    d = np.linspace(0, 1, 2 + tweens)
    for i in range(len(d) - 1):
        pos1 = interpos(nodes, d[i    ], com_a, com_b)
        pos2 = interpos(nodes, d[i + 1], com_a, com_b)
        f    = 0.5 * d[i] + 0.5 * d[i + 1]
        clr  = np.array(clr_a) * (1 - f) + np.array(clr_b) * f
        
        for j in range(4, len(pos1)):
            x = [pos1[j, 0], pos2[j, 0]]
            y = [pos1[j, 1], pos2[j, 1]]
            ax.plot(x, y, '-', color=clr, linewidth=2)
    
    # Emphasize begin and end points
    ax.plot(nodes[4:, 0], nodes[4:, 1], '.', color=clr_a, markersize=3)
    ax.plot(nodes[4:, 2], nodes[4:, 3], '.', color=clr_b, markersize=3)
    
    # Tight fit around bitmap
    ax.set_xlim(0, Da.shape[1])
    ax.set_ylim(Da.shape[0], 0)


def vignette(h, w, edge, doblur=True):
    """
    Generate a linear vignette alpha mask. Apply this to prevent nasty cut-off
    effects whenever blobs go near or over the edge of the canvas.
    
    Usage
    -----
    >>> V = vignette(h, w, edge, doblur)
    
    Parameters
    ----------
    h, w : int
        Width and height of the canvas in pixels.
    edge : int
        Vignette border size in pixels (same for all borders).
    doblur : bool, optional
        Whether or not to soften up the corners for that smooth & sensual look.
    
    Notes
    -----
    Because the same vignette will be applied to all morphs and inbetweens
    in a project, it pays off to generate one upfront and recycle it.
    """
    
    # Start transparent
    V = np.ones((h, w))
    if edge == 0: return V

    # Linear gradient from edges to center
    for i in range(edge):
        v = float(i) / edge
        V[i:(h - i), i        ] = v
        V[i:(h - i), w - 1 - i] = v
        V[i        , i:(w - i)] = v
        V[h - i - 1, i:(w - i)] = v
    
    # Blur out the disturbing 45 degree angle at the canvas corners
    if doblur:
        V = gaussian_filter(V, sigma=edge/3.)
        
        # Maintain a full opacity range over all edges
        iy    = int(V.shape[0] / 2)
        ix    = int(V.shape[1] / 2)
        v_min = min([min(V[iy, :]), min(V[:, ix])])
        v_max = max([max(V[iy, :]), max(V[:, ix])])
        eps   = 1. / edge
        V     = (V - v_min) / (v_max - v_min) * (1 - eps) + eps
        V     = np.clip(V, 0, 1)
    
    return V


def brush_tip(r, hardness=100, recycle=True):
    """
    Round brush alpha mask of given radius and hardness.
    Useful for making blobs by means of clone brushing.
    
    Usage
    -----
    >>> B = brush_tip(r, hardness)
    
    Parameters
    ----------
    r : integer
        Radius in pixels,
        measured from center to outer edge (including soft blur).
    hardness : int, optional
        Hardness parameter, from 0 supersoft to 1 superhard.
    
    Returns
    -------
    A square grayscale bitmap array,
    with values between 0 - 1 (transparent - opaque).
    """
    if r <= 0:
        raise ValueError('Brush size must be positive')
    if not 0 <= hardness <= 100:
        raise ValueError('Brush hardness must be between 0 and 100')
    
    r = int(np.ceil(r))
    hardness = int(round(hardness))
    if recycle:
        name = "h{h}r{r}".format(h=hardness, r=r)
        if name in brushes: return brushes[name]
    
    # Measure distance to center for each pixel
    xy   = np.linspace(-1, +1, 1 + 2 * r)
    xy   = xy[1:-1]
    X, Y = np.meshgrid(xy, xy)
    R    = np.sqrt(X ** 2 + Y ** 2)
    
    # Apply at least one pixel worth of softness for the sake of anti-aliasing
    # (As a bonus this prevents division by zero, a win-win situation I say)
    resolution = xy[1] - xy[0]
    h = np.clip(hardness / 100., 0, 1 - resolution)
    
    # Apply linear decay beyond the hard zone
    T = 1. +  (h - R) / (1. - h)
    T = np.clip(T, 0, 1)
    
    # Smoothen it up
    # (use the same polynomial as the double stop motion profile)
    F = 3 * T ** 2 - 2 * T ** 3
    
    # Remember for next time (store up to one thousand brush tips)
    if recycle and len(brushes) < 1000: brushes[name] = F
    
    return F


def clone_brush(K, x1, y1, x2, y2, r, hardness=50):
    """
    Draw a clone brush dot on a dedicated RGBa array a.k.a. layer.
    
    Usage
    -----
    >>> B = clone_brush(K, x1, y1, x2, y2, r, hardness)
    
    Parameters
    ----------
    Given are source bitmap, source and destination coordinates, brush size,
    and brush hardness. All coordinates must be integers.
    """

    # Start with empty destination layer
    B = np.zeros_like(K)
    w, h = K.shape[1], K.shape[0]

    # Brush alpha mask
    T = brush_tip(r, hardness)
    
    # Clip square copy box to frame boundaries if needed
    ri  = int(np.ceil(r))
    rxm = min((ri - 1, x1, x2))
    rym = min((ri - 1, y1, y2))
    rxp = min((ri - 1, w - 1 - x1, w - 1 - x2))
    ryp = min((ri - 1, h - 1 - y1, h - 1 - y2))
    
    # Copy square region
    for c in range(4):
        Kc = K[y1-rym:y1+ryp+1, x1-rxm:x1+rxp+1, c]
        B[y2-rym:y2+ryp+1, x2-rxm:x2+rxp+1, c] = Kc
    
    # Apply brush tip alpha mask
    Tc = T[ri-rym-1:ri+ryp, ri-rxm-1:ri+rxp]
    B[y2-rym:y2+ryp+1, x2-rxm:x2+rxp+1, 3] *= Tc
    
    # And that's really all there is to it
    return B


def inflate_blobs(nodes, scale=2., minsize=10, neighbours=3,
                  r_max=1000, com_a=None, com_b=None):
    """
    Set blob sizes based on near neighbour distances.
    
    Usage
    -----
    >>> ra, rb = inflate_blobs(nodes, scale, minsize, neighbours, ...)
    
    Parameters
    ----------
    nodes : array
        Blob center coordinates for key fames A & B.
        The first four nodes should be frame corners, and are ignored.
        The fith node is usually the center of mass.
    scale : float, optional
        Multiply the median nearest neighbour distance with this factor.
    minsize : int, optional
        Minimal radius in pixels.
    neighbours : int, optional
        The number of nearest neighbours to take into account.
    r_max : int, optional
        Maximal permissible radius.
    com_a, com_b : dict, optional
        Center of mass properties.
        If supplied, the CoM radius is applied to the fifth node.
    
    Returns
    -------
    Blub radii for key frames A & B respectively, as integer array [ra, rb].
    
    Notes
    -----
    Think of this as inflating a bundle of balloons, where each balloon is
    free to expand until it bumps into its brethren. The number of neighbours
    to account for is thus a measure of elasticity, if you know what I mean.
    """

    n_blob = len(nodes)
    radii  = np.ones((n_blob, 2), dtype=int) * minsize
    
    # Life is easy if we only have one blob
    if n_blob < 5:
        return radii
    if n_blob < 6:
        radii[4, :] = round(np.clip(r_max * scale, minsize, r_max))
        return radii
    
    # Measure distances between all nodes
    D = np.zeros((n_blob, n_blob, 2))
    for k in [0, 1]:
        if k == 0:
            x, y = nodes[:, 0], nodes[:, 1]
        else:
            x, y = nodes[:, 2], nodes[:, 3]
        for i in range(n_blob):
            for j in range(i + 1, n_blob):
                D[i, j, k] = distance(x[i], y[i], x[j], y[j])
                D[j, i, k] = D[i, j, k]
    
    # CoM blob: apply given size (based on drawing radius)
    i_start = 4
    if n_blob > 4 and \
            not com_a is None and not com_b is None and \
            com_a['r'] > 0 and com_b['r'] > 0:
        radii[i_start, :] = round(com_a['r'] * scale)
        i_start += 1
    
    # Plain blobs: assign size based on proximity of nearest neighbours
    for i in range(i_start, n_blob):
        da = np.squeeze(D[i, :, 0])
        db = np.squeeze(D[i, :, 1])
        da = np.sort(da[da > 0])
        db = np.sort(db[db > 0])
        da = bn.median(da[:neighbours]) * scale
        db = bn.median(db[:neighbours]) * scale
        da = round(np.clip(da, minsize, r_max))
        db = round(np.clip(db, minsize, r_max))
        
        radii[i, :] = [da, db]
    
    return radii


def draw_blob(K, nodes, radii, i, d=0.5, hardness=50, roundup=True):
    """
    Draw a single blob on a dedicated clone brush layer.
    
    Usage
    -----
    >>> B = draw_blob(K, blob, i, d, hardness)
    
    Parameters
    ----------   
    K : RGBa array
         Key frame to sample from
    nodes : array
         Node center positions
    i : int
         Index of the to-be-drawn blob
    d : float, optional
         Relative distance travelled by the blob.
         Value must be between 0 (start) and 1 (stop).
    hardness : int, optional
         Softness of brush tip in percent.
    forward : bool, optional
         Whether to move forwards (from start to stop position)
         or backwards (from stop to start position).
    
    Returns
    -------
    Clone brush dots from A & B respectively.
    Use these to create the inbetween frame by means of compositing.
    """

    # Blob start and stop coordinates
    xa, ya, ra = nodes[i, 0], nodes[i, 1], radii[i, 0]
    xb, yb, rb = nodes[i, 2], nodes[i, 3], radii[i, 1]
    
    # Blob current coordinates
    x = xa * (1 - d) + xb * d
    y = ya * (1 - d) + yb * d
    r = ra * (1 - d) + rb * d

    # Round coordinates to integers to enable indexing.
    # Do this slightly different for forwards and backwards;
    # poor man's anti-aliasing.
    if roundup:
        xi, yi = int(np.ceil(x)), int(np.ceil(y))
    else:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        
    
    # Round radii too, for the sake of speed (recycle brush tips)
    r = int(np.round(r))
    
    # Create clone brush dot
    B = clone_brush(K, xa, ya, xi, yi, r, hardness)

    return B


def interpos(nodes, d=0.5, com1=None, com2=None):
    """
    Interpolate node positions for a given intermediate frame.
    
    Usage
    -----
    >>> posi = interpos(nodes, d, com1, com2)
    
    Parameters
    ----------
    See *deform*.
    """
    
    # Start and stop positions
    pos1 = nodes[:, 0:2]
    pos2 = nodes[:, 2:4]
    if d == 0: return pos1
    if d == 1: return pos2
 
    # Swirl around the center of mass, but keep the corners in place
    if com1 is None or com2 is None:
        pot1 = pos1
    else:
        pot1 = swirl(pos1[:, :2], com1, com2, d)
        pot1[:5] = pos1[:5]

    # Plain linear interpolation
    posi = pot1 + d * (pos2 - pot1)
    
    return posi


def deform(M, nodes, d=0.5, com_a=None, com_b=None,
           radii=None, hardness=50, roundup=True):
    """
    Deform an image. Apply either warping or blobbing. Warping boils down to
    piecewise affine transformation on an image, by moving the nodes towards
    the corresponding positions on the next frame. In case of blobbing this
    deformation is accomplished by means of clone brushing.

    When morphing a fresh inbetween, this function is to be called twice;
    key A is warped forwards to B, and also key B is warped backwards to A.

    Usage
    -----
    >>> W, posi = deform(M, nodes, d, com_a, com_b, radii, hardness)

    Parameters
    ----------
    M : bitmap array
        The image to be distorted
    nodes : array
        Coordinates of corresponding points on current (1)
        and destination frame (2), with rows [x1, y1, x2, y2].
    d : float, optional
        Relative distance travelled, between 0 and 1.
    com_a, com_b : dicts, optional
        Center of gravity properties for both frames.
        If not supplied, straight line interpolation will be applied.
    forward : bool, optional
        Move forwards (from A to B) or rather backwards (from B to A).
    radii : array, optional
        Blob sizes for current and destination frame, with rows [r1, r2].
        If not supplied, warp deformation will be applied.
    hardness : int, optional
        Clone brush hardness in percent.
    roundup : bool, optional
        Whether to round blob positions up (True) or down (False) to integers.

    Returns
    -------
    Bitmap array of the warped image, and interpolated positions.
    """
    
    if not 0 <= d <= 1:
        raise ValueError('Relative distance must be between 0 and 1')
    
    posi = interpos(nodes, d, com_a, com_b)

    if radii is None:
        trafo = PiecewiseAffineTransform()
        trafo.estimate(posi, nodes[:, [0, 1]])
        W = warp(M, trafo)
        
    else:
        W = M.copy()
        for i in range(4, len(nodes)):
            if np.any(abs(nodes[i, [2, 3]] - nodes[i, [0, 1]])):
                B = draw_blob(M, nodes, radii, i, d, hardness, roundup)
                W = composite_bitmaps(B, W)

    return W, posi


def tween(Ka, Kb, nodes, d=0.5, f=None, V=None, G=None,
          com_a=None, com_b=None, radii=None, hardness=50, backfade=False):
    """
    Make an inbetween frame. The prescribed deformation can be achieved by:
    
        - warp. Piecewise linear transformation, the default.
        - blob. Clone brushing. Gives a rather bubbly look, hence the name.

    Usage
    -----
    >>> T = tween(Ka, Kb, nodes, d, f, G, V, com_a, com_b)

    Parameters
    ----------
    Basically the same story as for *deform*.
    Only here supply two key frames (A & B),
    and optionally a background image (G) and vignette mask (V).
    
    In case of background fading (backfade=True) the fixed image (G) is ignored,
    and a crossfade of the undeformed key frames serves as the backdrop.
    This option is particularly useful for edge vignetting opaque keys.
    
    By default the fade factor *f* is the same as the distance travelled *d*.
    Blob deformation is applied if and only if *radii* is supplied.
    """
    
    if f is None: f = d

    # Only deform things if we really need to
    Wa, Wb = Ka, Kb
    
    # Forwards deformation from A to B
    if d > 0 and f < 1:
        Wa, posa = deform(Ka, nodes, d, com_a, com_b,
                          radii, hardness, roundup=False)
    
    # Backwards deformation from B to A
    if d < 1 and f > 0:
        nodes_r  = nodes[:, [2, 3, 0, 1]]
        radii_r  = radii[:, [1, 0]] if not radii is None else None
        Wb, posb = deform(Kb, nodes_r, 1 - d, com_b, com_a,
                          radii_r, hardness, roundup=True)

    # Fade
    T = Wa * (1 - f) + Wb * f
    
    # Apply vignette
    if not V is None: T[:, :, 3] *= V
    
    # Apply background
    if backfade:
        G = Ka * (1 - f) + Kb * f
    if not G is None:
        T = composite_bitmaps(T, G)

    return T

