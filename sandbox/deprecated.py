#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graveyard for deprecated algorithms. May they rest in peace.
"""


from muddymorph_algo import desaturate
from scipy.ndimage import gaussian_filter
from skimage.filters import roberts, sobel, scharr, prewitt

def loco(D, algorithm='muddy', detail=12, cutoff=0.04):
    """
    Estimate local contrast for the given bitmap. Zero indicates a total
    absence of local intensity variation, and one indicates ultimate contrast.
    With other words, this sounds like a job for a traditional edge detector.

    Parameters
    ----------
    D : greyscale image array
        Desaturated difference with the background.
    algorithm : str, optional
        The method to use. Choose between muddy (home grown, see notes below)
        or one of Scharr / Sobel / Prewitt / Roberts (classics from scikits).
    cutoff : float, optional
        Eualization clipping limit, somewhere in the range 0 - 1.
        Note that a value of 0.1 is already quite agressive.
    detail : int or float, optional
        A measure for the evaluation radius, in pixels.
        Set this to the typical line width or edge gradient width.
        Only relevant for the muddy algorithm.

    Notes
    -----
    This is supposed to yield a map that approximates the intensity difference
    between nearby light and dark zones for all points. Points with high local
    contrast are eligible for serving as morph nodes. Our way of working here:

        1. Normalize contrast through adaptive histogram equalization
        2. Apply the selected algorithm. In case of muddy:
            a) Subtract a Gaussian blur
            b) Take the absolute value of this difference.
    """

    G = desaturate(D, cutoff=cutoff)

    algorithm = algorithm.lower().strip()

    if algorithm == 'muddy':
        F = gaussian_filter(G, sigma=detail)
        C = abs(G - F)

    elif algorithm == 'scharr' : C = scharr (G)
    elif algorithm == 'sobel'  : C = sobel  (G)
    elif algorithm == 'prewitt': C = prewitt(G)
    elif algorithm == 'roberts': C = roberts(G)

    return C
