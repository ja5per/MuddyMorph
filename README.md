# MuddyMorph

A Fuzzy Inbetween Generator For Lazy Animators.


## What?
This wizard-style application gladly accepts a series of bitmap images (key frames),
and will bravely and semi-automatically attempt to generate appropriate inbetween frames.
The aim is to swiftly create a smooth and fluent motion with minimal user input.

The output consists of a series of bitmaps images; originals interlaced with tweens.
These can then be easily imported into the users favourite video editing software.


## Why?
Good question. This tool can be used for making traditional morphs, such as blending 
different faces into each other, or daftly metamorphosing people into animals or objects.
But the main purpose is to facilitate traditional animation; think of hand drawn paper
based animation, or claymation and other forms of stop motion. In such cases key frames 
usually come in the shape of bitmaps (photographs or scans) rather than vectors.

The process of drawing or photoshopping countless intermediate images can be long and
tedious. If you are a lazy animator, craving fast results and not minding a few muddy
quirks here and there, this tool may be of service to you. If however you are a
perfectionist, or allergic to unexpected and often psychedelic results, then better stay
at a safe distance.


## How?
Animation cells usually contain a single character or subject with a clear silhouette,
well defined against a blank background. MuddyMorph exploits this trait, and identifies
for each pair of key frames matching points on the edges of both silhouettes;
a match is made when locations are relatively close to each other and look alike.
These point pairs are dubbed nodes, and define movement start and stop locations.

The wizard guides the user through these steps, one tab at a time:

1. **Images**. Select key frames.
2. **Preferences**. Set the typical number of tweens and such.
3. **Silhouette**. Identify key shapes by means of a simple threshold filter.
4. **Trajectories**. Find corresponding points for each pair of key frames.
5. **Motion**. Define the duration, amount of fading, and accelleration.
6. **Render**. Export the morph.


## Implementation
This project is written in python, and relies heavily on the image analysis and
manipulation capabilities of *PIL*, *skimage*, and *scipy*. The main modules are:

File               | Contents
-------------------|-------------------------------------
MuddyMorph.py      | GUI (PyQT5)
muddymorph_algo.py | Algorithms and assist functions
muddymorph_gogo.py | All the logistics for making a morph
muddymorph.ui      | User interface design (Qt designer)

