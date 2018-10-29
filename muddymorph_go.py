#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a Morph, one step at a time. Go go go!

This is a collection of methods that handle the various intricate logistical
processes; set up a temporary path, call the various algorithms, and organize
settings and intermediate result files. With other words, everything that may
be of interest for eithr API or GUI based operation.

Usage
-----
Recommended import convention
>>> import muddymorph_go as gogo
"""

# Dependencies: Home grown
import muddymorph_algo as algo

# Dependencies: Open source
import json
import numpy as np
from glob import glob
from copy import deepcopy
from time import asctime, strftime, time
from os import path, mkdir, remove
from shutil import rmtree, copyfile
import matplotlib.pyplot as plt

# Project meta info
__author__   = algo.__author__
__version__  = algo.__version__
__revision__ = algo.__revision__

# Options for saving diagrams
chartopts = dict(dpi=150, transparent=True, bbox_inches='tight')


def temppath():
    """
    Find a good spot for storing semi-temporary files.
    
    On Linux or Windows:
        ~/.muddymorph/temp
    On Mac:
        ~/Library/MuddyMorph/temp
    
    (Where ~ is an alias for the user home directory).
    """
    home = path.expanduser('~')
    
    if home.startswith('/Users'):
        folder = path.join(home, 'Library', 'MuddyMorph', 'temp')
    else:
        folder = path.join(home, '.muddymorph', 'temp')
    
    return folder


def timestamp():
    """
    Return a timestamp string for a new log message.
    Formatted example (2011-12-30 21:30:46).
    """
    return '(%s) ' % strftime('%Y-%m-%d %H:%M:%S')


def duration(t):
    """
    Give a nice short and readable string representation of a given time
    duration in seconds, e.g. how long it took to render the whole project.
    
    Usage
    -----
    >>> timetag = duration(t)

    Parameters
    ----------
    The time that has passed in seconds (int or float).
    
    Returns
    -------
    A nice and short readable time tag, such as '3h:15m:02s'.
    """
    if t < 10:
        return '%.2fs' % t
    elif t < 60:
        return '%.0fs' % t
    else:
        h = int(t // 3600)
        m = int((t - h * 3600) // 60)
        s = t - h * 3600 - m * 60
        if h < 1:
            return '%02d:%02d' % (m, s)
        else:
            return '%02dh:%02dm:%02ds' % (h, m, s)


def default_settings():
    """
    Factory default settings for algorithm parameters.
    Returns a dictionary of mostly dictionaries, if you know what I mean.
    
    The names of most variables roughly correspond to those on GUI controls.
    
    Usage
    -----
    >>> settings = default_settings()
    
    Notes
    -----
    Default location of temporary folder 'temp':
    - On Mac in the directory of this module
    - On Linux/Windows/Other under user home '~/.blobmorph/temp'
    """
    
    print(timestamp() + 'Digging up default settings')
    
    # Store application version and revision,
    # for the sake of enabling backwards compatibility in future versions
    v = 'MuddyMorph {} {}'.format(__version__, __revision__)
    
    # This will be a list of the selected key frames
    # These will be stored along with the settings mostly for remembering the
    # users favorite image storage folder. Obviously this is empty by default.
    keyframes = []
    
    # Main settings for rendering and previewing the project
    s_render = {'name'           : 'morph'    ,
                'ext'            : 'png'      ,
                'folder'         : ''         ,
                'backdrop'       : ''         ,
                'loop'           : False      ,
                'reverse'        : False      ,
                'reversepreview' : True       ,
                'quality'        : 90         , # 10 - 100 percent
                'transparent'    : False      ,
                'framerate'      : 6          , # 1 - 30 fps
                'autoback'       : True       ,
                'backcolor'      : (0, 255, 0),
                'linecolor'      : (0, 0, 0)  ,
                'lineart'        : False      ,
                'equalize'       : 2          , # >= 0 percent
                'vignette'       : 10         , # pixels
                'blobscale'      : 100        , # percent
                'blobhardness'   : 50         } # percent
    
    # Silhouette and edge detection
    # (All can vary per key frame)
    s_edge = {'invert'        : False      ,
              'scharr'        : False      ,
              'channel'       : 'lightness',
              'threshold'     : 20         , # 1 - 100 percent
              'blur'          : 0          , # 0 - 100 pixels
              'cornercatcher' : 'ORB'      }
    
    # Keypoint trajectories and node trajectories
    # (All can vary per morph sequence)
    s_traject = {'corners'    : False,
                 'silhouette' : True ,
                 'spin'       : True ,
                 'arc'        : True ,
                 'detail'     : 20   , # >=1 promille
                 'similim'    : 30   , # 0 - 100 percent
                 'maxmove'    : 20   , # 1 - 100 percent
                 'neighbours' : 10   , # >= 1 points
                 'maxpoints'  : 1000 } # >= 1 points
    
    # Motion control
    # (All can vary per morph sequence)
    s_motion = {'blob'       : False   ,
                'fade'       : 0       , # 0 - 100 percent
                'inbetweens' : 6       , # >= 1 frames
                'profile'    : 'linear'}
    
    # Pack all settings into one convenient bundle
    settings = {'version'   : v         ,
                'step'      : 1         ,
                'temppath'  : temppath(),
                'keyframes' : keyframes ,
                'edge'      : s_edge    ,
                'traject'   : s_traject ,
                'motion'    : s_motion  ,
                'render'    : s_render  }
    
    return settings


def count_morphs(settings):
    """
    Count the number of morph sequences,
    given the number of key frames and loop switch setting.
    """    
    return len(settings['keyframes']) - 1 + settings['render']['loop']


def fetch_dimensions(settings, K=None):
    """
    Fetch key frame dimensions (height, width) in pixels.
    """
    if K is None:
        h, w = algo.read_image_dimensions(settings['keyframes'][0])
    else:
        h, w = K.shape[:2]
    
    return h, w

    
def morph_key_indices(settings, m):
    """
    Return indices of key frames for the given morph sequence.
    
    Usage
    -----
    >>> a, b = morph_key_indices(settings, m)
    """
    n = len(settings['keyframes'])
    if settings['render']['loop'] and m == n - 1:
        a, b = m, 0
    else:
        a, b = m, m + 1
    if a < 0 or a >= n or b < 0 or b >= n:
        raise ValueError('Invalid morph sequence index {}'.format(m))
    
    return a, b


def can_copy_key(settings, k):
    """
    Can the given key frame file simply be copied to the export folder?
    This is the case if the key and export file extensions match,
    and in case of PNG files if no solid background color is to be applied.
    """    
    filename = settings['keyframes'][k]
    ext_key  = path.splitext(filename)[-1][1:].lower()
    ext_out  = settings['render']['ext']
    trans    = settings['render']['transparent']
    cancopy  = ext_key == ext_out
    
    if ext_out == 'png' and not trans: cancopy = False

    return cancopy


def size_consistency_check(imagefiles):
    """
    Check whether a series of image files has the exact same dimensions.
    """
    print(timestamp() + 'Image size consistency check ... ', end='')
    
    ok, dim_prev = True, ''
    
    for f in imagefiles:
        try:
            s = algo.read_image_dimensions(f)
            dim_this = '{} x {}'.format(s[1], s[0])
            
            if not dim_prev:
                dim_prev = dim_this
            else:
                ok = dim_this == dim_prev
                if not ok: break
        
        except TypeError:
            msg = 'Unable to read image dimensions for ' + path.basename(f)
            raise ValueError(msg)
    
    if ok:
        print('Okay')
    else:
        print('Nok! Mismatch for ' + path.basename(f))
    
    return ok


def expand_settings(settings):
    """
    Expand (vectorize) settings to match the number of key frames and morphs.
    This is best done directly after choosing the key frames and loop mode.
    
    Usage
    -----
    >>> expand_settings(settings)
    """
    
    # Expand key frame specific settings    
    n = len(settings['keyframes'])
    for prop in settings['edge']:
        v = settings['edge'][prop]
        f = algo.most_frequent_value(v)
        settings['edge'][prop] = [f] * n

    # Expand morph sequence specific settings
    n = count_morphs(settings)
    for branch in ['traject', 'motion']:
        for prop in settings[branch]:
            v = settings[branch][prop]
            f = algo.most_frequent_value(v)
            settings[branch][prop] = [f] * n


def init_temp(settings):
    """
    Set up temporary file folder for a new project:
        
        1) Eradicate any lingering content from the previous round.        
        2) Create folder.
        3) Create subdirectories.
        4) Write table of contents to a short readme file.
        
    Usage
    -----
    >>> init_temp(settings)
    
    Notes
    -----
    If the parent path does not exist, it will be created too.
    This is to facilitate the default folder ~/.muddymorph/temp
    """
    
    legend = """
    Numbering convention for files and folders:
       - k001, k002, k003, ... for key frames.
       - m001, m002, m003, ... for morph sequences.
       - f001, f002, f003, ... for frames (keys + inbetweens).
    
    Files in the root directory:
        - readme.txt. Table of contents, this very file.
        - bg.png. Background (color and/or image).
        - bgc.png. Background composite with representative key.
    
    Subdirectories:
        - k??? for key frame analyses, containing files;
             - silly.png      silhouette shape binary map.
             - edgy.png       silhouette edges binary map.
             - com.json       silhouette center of mass properties.
             - shape.png      silhouette detection diagram.
             - orb.npy        corner keypoint coordinates from ORB.
             - orb.png        corner keypoint binary descriptors from ORB.
             - orb_p.png      corner keypoint preview (dots on grayscale).
             - censure.npy    corner keypoint coordinates from CENSURE.
             - censure.png    corner keypoint binary descriptors from CENSURE.
             - censure_p.png  corner keypoint preview (dots on grayscale).
        - m??? for morph sequences, containing files;
             - nodes.csv      node trajectory coordinates.
             - radii.csv      blob start and stop sizes.
             - move.png       node trajectory chart.
             - mid.{jpg/png}  midpoint frame.
             - f???.{jpg/png} individual frames.
    """
    
    print(timestamp() + 'Setting up temporary path')
    
    temp       = settings['temppath']
    keyframes  = settings['keyframes']
    parentpath = path.abspath(path.join(temp, path.pardir))
    
    if path.exists(temp):
        rmtree(temp)
    elif not path.exists(parentpath):
        mkdir(parentpath)
    
    mkdir(temp)
    
    for i in range(len(keyframes)):
        mkdir(path.join(temp, 'k%03d' % (i + 1)))
    
    for i in range(count_morphs(settings)):
        mkdir(path.join(temp, 'm%03d' % (i + 1)))
    
    with open(path.join(temp, 'readme.txt'), 'w') as doc:
        doc.write('MuddyMorph Temporary Files Folder\n')
        doc.write('Generated on {}\n\n'.format(asctime()))
        doc.write(legend)


def clear_temp_key(settings, k=None):
    """
    Delete all temporary key frame analysis files.
    For either all key frames (default),
    or a particular selection (list of indices).
    
    Cleaning up files is necessary in interactive mode when applying new
    settings across multiple frames, or when going back one or more steps.
    """
    if k is None: k = range(len(settings['keyframes']))
    
    for kk in k:
        search = path.join(settings['temppath'],
                           'k{0:03d}'.format(kk + 1), '*.*')
        hitlist = glob(search)
        if len(hitlist):
            msg = 'Clearing temp files for key {}'.format(kk + 1)
            print(timestamp() + msg)
        
        for hit in hitlist: remove(hit)


def clear_temp_traject(settings, m=None, k=None):
    """
    Delete all temporary keypoint and trajectory analysis files.
    For either all morphs and keys (default),
    or a particular selection (list of indices).
    
    Cleaning up files is necessary in interactive mode when applying new
    settings across, or when going back one or several steps.
    """
    if m is None: m = range(count_morphs(settings))
    if k is None: k = range(len(settings['keyframes']))
    
    for mm in m:
        search = path.join(settings['temppath'],
                           'm{0:03d}'.format(mm + 1), 'm*.*')
        hitlist = glob(search)
        if len(hitlist):
            msg = 'Clearing temp files for trajectory {}'.format(mm + 1)
            print(timestamp() + msg)
        
        for hit in hitlist: remove(hit)
    
    for kk in k:
        folder = path.join(settings['temppath'], 'k{0:03d}'.format(kk + 1))
        hitlist = glob(path.join(folder, 'orb*.*'    )) + \
                  glob(path.join(folder, 'censure*.*'))
        if len(hitlist):
            msg = 'Clearing temp files for keypoints {}'.format(kk + 1)
            print(timestamp() + msg)
        
        for hit in hitlist: remove(hit)


def clear_temp_motion(settings, m=None):
    """
    Delete all temporary inbetween frame files.
    For either all morphs (default),
    or a particular selection (list of indices).
    
    Cleaning up files is necessary in interactive mode when applying new
    settings across multiple morphs, or when going back one or more steps.
    """
    if m is None: m = range(count_morphs(settings))
    
    for mm in m:
        search = path.join(settings['temppath'],
                           'm{0:03d}'.format(mm + 1), 'f*.*')
        hitlist = glob(search)
        if len(hitlist) > 0:
            msg = 'Clearing temp files for morph {}'.format(mm + 1)
            print(timestamp() + msg)
        
        for hit in hitlist: remove(hit)


def load_settings(settingsfile=None):
    """
    Load user preferences from file.
    """
    
    # Start with default settings (facilitate backwards compatibility)
    settings = default_settings()
    deftemppath = settings['temppath']
    
    # Load the file
    if not settingsfile:
        settingsfile = path.join(deftemppath, '..', 'muddymorph.json')
    if not path.isfile(settingsfile):
        raise ValueError('Unable to locate settings file ' + settingsfile)
    settingz = loadit(settingsfile)
    
    # Is this really a settings file? Perform a few basic checks
    for prop in ['version', 'step'   , 'temppath', 'keyframes',
                 'edge'   , 'traject', 'motion'  , 'render'   ]:
        if not prop in settingz:
            raise ValueError('Missing field "{}" in settings'.format(prop))
    if not type(settingz['version']) is str \
            or not settingz['version'].startswith('MuddyMorph'):
        raise ValueError('Settings file does not appear to be from MuddyMorph')
    
    # Apply those sizzling settings
    for prop in ['version', 'step', 'keyframes']:
        settings[prop] = settingz[prop]
    for branch in ['edge', 'traject', 'motion', 'render']:
        settings[branch].update(settingz[branch])

    # Make sure all settings are decently expanded per morph
    if settings['step'] > 2:
        for branch in ['edge', 'traject', 'motion']:
            if branch == 'edge':
                n = len(settings['keyframes'])
            else:
                n = count_morphs(settings)
            for prop in settings[branch]:
                v = settings[branch][prop]
                if not type(v) is list or len(v) != n:
                    msg  = 'Setting {}/{} is not fully specified. '
                    msg  = msg.format(branch, prop)
                    msg += 'Limiting tabs to step 2.'
                    print(timestamp() + msg)
                    
                    settings['step'] = 2
                    break
    
    # Make sure the temp path is valid
    # (These settings may have came come from a different computer)
    if not path.isdir(settings['temppath']):
        if not path.isdir(deftemppath):
            raise ValueError('Unable to locate the temporary file path')
        settings['temppath'] = deftemppath
    
    return settings


def save_settings(settings, settingsfile=None):
    """
    Save user preferences to file.
    """
    if not settingsfile:
        settingsfile = path.join(settings['temppath'], '..', 'muddymorph.json')
    saveit(settings, settingsfile)


def semi_short_file_name(filename, settings=None):
    """
    Given a potentially long file, take only the last bit;
    being the parent folder and basename.
    """
    parts = filename.replace('\\', '/').split('/')
    showname = '/'.join(parts[-2:])
    return showname


def loadit(filename, settings=None):
    """
    Load analysis data from JSON/CSV/PNG file,
    presumably somewhere in the temporary folder.
    CSV files must be integer-only and headerless arrays,
    and PNG bitmaps must be boolean masks.
    """
    showname = semi_short_file_name(filename)
    print(timestamp() + 'Loading ' + showname)
    
    if not settings is None:
        filename = path.join(settings['temppath'], filename)
    
    ext = path.splitext(filename)[1].lower()
    if ext == '.json':
        with open(filename, 'r') as doc:
            data = json.load(doc)
    elif ext == '.csv':
        data = np.loadtxt(filename, delimiter=',').astype(int)
    elif ext == '.png':
        data = plt.imread(filename).astype(bool)
    else:
        raise ValueError('Unsupported file extension ' + ext)
    
    return data


def saveit(data, filename, settings=None):
    """
    Save analysis data to JSON/CSV/PNG file,
    presumably somewhere in the temporary folder.
    """
    showname = semi_short_file_name(filename)
    print(timestamp() + 'Saving ' + showname)
    
    if not settings is None:
        filename = path.join(settings['temppath'], filename)    
    
    ext = path.splitext(filename)[1].lower()
    if ext == '.json':
        with open(filename, 'w', encoding='utf-8') as doc:
            json.dump(data, doc, indent=4, sort_keys=True)
    elif ext == '.csv':
        np.savetxt(filename, data, fmt='%d', delimiter=',')
    elif ext == '.png':
        algo.save_grey(data, filename)
    else:
        raise ValueError('Unsupported file extension ' + ext)


def background(settings, K=None, B=None):
    """
    Create and save the background image for all morph sequences.
    Typically this will just be an empty transparent canvas,
    but a solid color or fixed image may also be applied.
    
    Usage
    -----
    >>> G = background(settings, K, B)
    """
    print(timestamp() + 'Setting up backdrop')

    if B is None: B = settings['render']['backdrop']
    
    h, w  = fetch_dimensions(settings, K)
    clr   = settings['render']['backcolor'  ]
    trans = settings['render']['transparent']
    G     = algo.make_background(h, w, clr, trans, B)
    f     = path.join(settings['temppath'], 'bg.png')
    
    algo.save_rgba(G, f)
    
    return G


def vinny(settings, K=None):
    """
    Create edge vignette mask.
    """
    print(timestamp() + 'Making vignette mask')
    
    h, w  = fetch_dimensions(settings, K)
    V = algo.vignette(h, w, settings['render']['vignette'])
    
    return V


def silhouette(settings, k, recycle=False, K=None, X=None, Y=None,
               showsil=True, showedge=True, showcom=True):
    """
    Perform silhouette extraction and contour detection for frame *k*.
    These analysis files are generated:

         - silly.png  silhouette shape binary map.
         - edgy.png   silhouette edges binary map.
         - com.json   silhouette center of mass properties.
         - shape.png  silhouette detection diagram (returned for preview).
    
    Usage
    -----
    >>> f, K, E, com = silhouette(settings, k)
    
    Returns
    -------
    1. Filename of diagnostics chart
    2. Key frame bitmap
    3. Edge map
    4. Center of mass properties
    """
    
    # This is where the action is
    folder = path.join(settings['temppath'], 'k{0:03d}'.format(k + 1))  
    f = path.join(folder, 'shape.png')

    # Fetch ingredients
    se = settings['edge']
    sr = settings['render']
    bc = None if sr['autoback'] else sr['backcolor']
    
    if K is None: K = algo.load_rgba(settings['keyframes'][k])
    
    # Do we need to do anything at all?
    if recycle and path.isfile(f) and \
                   path.isfile(path.join(folder, 'com.json' )) and \
                   path.isfile(path.join(folder, 'silly.png')) and \
                   path.isfile(path.join(folder, 'edgy.png' )):
        return f, K, None, None

    # Make mesh grid
    if X is None or Y is None: X, Y = algo.grid(K)
    
    # Extract silhouette
    D, S, E = algo.edgy(K, 
                        backcolor = bc,
                        linecolor = sr['linecolor'],
                        dolines   = sr['lineart'  ],
                        threshold = se['threshold'][k] * 0.01,
                        channel   = se['channel'  ][k],
                        doscharr  = se['scharr'   ][k],
                        blur      = se['blur'     ][k],
                        invert    = se['invert'   ][k])
    
    # Center of mass measurement
    com = algo.commie(S, X=X, Y=Y, verbose=False)
    
    # Save the harvest
    saveit(com, path.join(folder, 'com.json' ))
    saveit(S  , path.join(folder, 'silly.png'))
    saveit(E  , path.join(folder, 'edgy.png' ))
    
    # Combine all results into one classy chart
    Sp   = S   if showsil  else None
    Ep   = E   if showedge else None
    comp = com if showcom  else None
    fig  = algo.big_figure('MuddyMorph - Silhouette Chart', *E.shape)
    
    algo.edgeplot(D, Sp, Ep, comp, X=X, Y=Y)
    plt.axis('off')
    plt.savefig(f, **chartopts)
    plt.close(fig)
    
    return f, K, E, com


def abs_detail(settings, m, h=None, w=None):
    """
    Denormalize detail setting for a given morph, from promille to pixels.
    This is used both as a similarity evaluation zone and minimal blob size.
    
    Usage
    -----
    >>> r = abs_detail(settings, m, h, w)
    """
    
    if not h or not w: h, w = fetch_dimensions(settings)
    
    detail = settings['traject']['detail'][m] * 1e-3

    # FIXME: Remove this after testing new traject detail parameter
    #a, b   = morph_key_indices(settings, m)
    #se     = settings['edge']
    #detail = 0.5 * se['detail'][a] * 1e-3 + \
    #         0.5 * se['detail'][b] * 1e-3
    
    r = min(int(np.ceil(max(h, w) * detail)) + 1, 4)
    
    return r


def cornercatch(settings, k, recycle=True, K=None):
    """
    Detect corner key points for key frame *k*.
    These analysis files are generated:
         - In case of ORB; orb.csv, orb.png, orb_p.png.
         - In case of CENSURE; censure.csv, censure.png, censure_p.png.
    
    The full file name of the diagnostics diagram is returned for previewing.
    
    Usage
    -----
    >>> f, kp, dc = cornercatch(settings, k)
    
    Returns
    -------
    1. Filename of diagnostics chart
    2. Key point coordinates (None in case of file recycle).
    3. Key point binary descriptors (None in case of file recycle).
    """
    print(timestamp() + 'Collecting corners for key {}'.format(k))
    
    # Algorithm flavour and save file base
    catcher = settings['edge']['cornercatcher'][k]
    folder  = path.join(settings['temppath'], 'k{0:03d}'.format(k + 1))
    base    = path.join(folder, catcher.lower())
    f1      = base + '_p.png'
    f2      = base + '.csv'
    f3      = base + '.png'
    
    # Do we need to do anything at all?
    if recycle and path.isfile(f1) \
               and path.isfile(f2) \
               and path.isfile(f3): return f1, None, None
    
    # Collect the other parameters
    blur        = settings['edge']['blur'][k]
    spawnpoints = min(1000, *settings['traject']['maxpoints'])
    channel     = settings['edge']['channel'][k]
    if channel.lower().startswith('a'): channel = 'lightness'
    
    # Say it like it is
    msg = '{} corner extraction for key {}'
    print(timestamp() + msg.format(catcher, k + 1))

    # Load bitmap
    if K is None: K = algo.load_rgba(settings['keyframes'][k])
    
    # Do dat ting
    kp, dc = algo.cornercatch(K, channel=channel, algorithm=catcher,
                              target=spawnpoints, blur=blur)
    
    # Save the harvest
    saveit(kp, base + '.csv')
    saveit(dc, base + '.png')
    
    # Produce a simple diagnostics chart (just a bunch of orange dots)
    G   = algo.desaturate(K, channel, blur=blur)
    fig = algo.big_figure('MuddyMorph - Corner key points', *G.shape)
    
    plt.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.plot(kp[:, 0], kp[:, 1], '.', markersize=7, color=(1., .5, 0.))
    plt.axis('image')
    plt.axis('off')
    plt.savefig(f1, **chartopts)
    plt.close(fig)
    
    return f1, kp, dc

    
def shoutout(msg='', img=None, thread=None):
    """
    Give an intermediate status report
    (during trajectory planning or morph generation).
    """
    if msg:
        print(timestamp() + msg)
    if thread:
        if msg: thread.report.emit(msg + ' ... ')
        if img: thread.report.emit(img)

    
def trajectory(settings, m, recycle=False, thread=None, X=None, Y=None,
               Ka=None, Kb=None, Ea=None, Eb=None, com_a=None, com_b=None,
               kp_a=None, kp_b=None, dc_a=None, dc_b=None):
    """
    Figure out node trajectories for the given morph sequence *m*,
    based on silhouette map and corner descriptors.
    
    Two temporary analysis files are generated:
         - nodes.csv node trajectory coordinates.
         - move.png  node trajectory chart.
    
    Usage
    -----
    >>> nodes, Ka, Kb, com_a, com_b = trajectory(settings, m, recycle, thread)
    
    Parameters
    ----------
    recycle : bool, optional
        If set to True and if output exists from a previous run,
        then that will be recycled.
    thread : object, optional
        Send status reports back through this channel,
        presumably a PyQt Qthread activated by the grapical user interface.
        This can be any object though, as long as it contains:
            - *abort*. A boolean status flag (True/False) that signals whether
              the user has had enough, and pressed a cancel button or such.
            - *report*. A progress report signal.
              Must have a method *emit* that accepts strings
              (which will either an image file name or one-line status report).
    
    Returns
    -------
    None in case of user abort,
    node trajectory coordinate array otherwise.
    
    Notes
    -----
    If silhouette map and corner keypoint coordinates are not available,
    then *silhouette* and *cornercatch* will be called to create these.
    """
    
    # Tell everyone about the fantastic voyage we are about to embark upon
    if count_morphs(settings) > 1:
        label = ' for morph {}'.format(m + 1)
    else:
        label = ''
    shoutout(msg='Detecting trajectories' + label, thread=thread)
    
    # Start the timer
    stopwatch = -time()

    # Key frame indices and output files
    a, b     = morph_key_indices(settings, m)
    folder_m = path.join(settings['temppath'], 'm{0:03d}'.format(m + 1))
    folder_a = path.join(settings['temppath'], 'k{0:03d}'.format(a + 1))
    folder_b = path.join(settings['temppath'], 'k{0:03d}'.format(b + 1))
    f1       = path.join(folder_m, 'move.png')
    f2       = path.join(folder_m, 'nodes.csv')
    
    # Assemble the settings
    s = settings['traject']
    docorners    = s['corners'   ][m]
    dosilhouette = s['silhouette'][m]
    arc          = s['arc'       ][m]
    spin         = s['spin'      ][m] and arc
    similim      = s['similim'   ][m] * 1e-2
    maxmove      = s['maxmove'   ][m] * 1e-2
    maxpoints    = s['maxpoints' ][m]
    neighbours   = s['neighbours'][m]
    
    # Detect silhouette of key frame A
    if Ka is None or Ea is None or com_a is None:
        msg = 'Extracting silhouette for key {}'.format(a + 1)
        shoutout(msg=msg, thread=thread)
        result = silhouette(settings, a, K=Ka, X=X, Y=Y, recycle=True)
        fsa, Ka, Ea, com_a = result
        shoutout(img=fsa, thread=thread)
        if thread and thread.abort: return
    
    # Detect silhouette of key frame B
    if Kb is None or Eb is None or com_b is None:
        msg = 'Extracting silhouette for key {}'.format(b + 1)
        shoutout(msg, thread=thread)
        result = silhouette(settings, b, K=Kb, X=X, Y=Y, recycle=True)
        fsb, Kb, Eb, com_b = result
        shoutout(img=fsb, thread=thread)
        if thread and thread.abort: return
    
    # Catch corners
    if docorners:
        shoutout('Catching corners for key {}'.format(a + 1), thread=thread)
        fca, kp_a, dc_a = cornercatch(settings, a, K=Ka, recycle=True)
        shoutout(img=fca, thread=thread)
        if thread and thread.abort: return

        shoutout('Catching corners for key {}'.format(b + 1), thread=thread)
        fcb, kp_b, dc_b = cornercatch(settings, b, K=Kb, recycle=True)
        shoutout(img=fcb, thread=thread)
        if thread and thread.abort: return
    
    # Nothing can beat the need for shear speed
    if recycle and path.isfile(f1) \
               and path.isfile(f2):
        shoutout(img=f1, thread=thread)
        nodes = loadit(f2)
        return nodes, Ka, Kb, com_a, com_b
    
    # Convert detail zone units from promille to pixels
    # FIXME: Remove this after testing new traject detail setting
    #se       = settings['edge'  ]
    #detail   = 0.5 * se['detail'][a] * 1e-3 + \
    #           0.5 * se['detail'][b] * 1e-3
    detail = settings['traject']['detail'][m] * 1e-3
    simisize = max(int(np.ceil(max(Ka.shape[:2]) * detail)) + 1, 4)
    
    # Show the nitty gritty details
    print(timestamp() + 'Similim  = {} %'   .format(s['similim'][m]))
    print(timestamp() + 'Detail   = {0:.3f}'.format(detail))
    print(timestamp() + 'Simisize = {} px'  .format(simisize))

    # Start with the foundation;
    # The four screen corners and center of mass
    if dosilhouette:
        if Ea is None: Ea = loadit(path.join(folder_a, 'edgy.png'))
        if Eb is None: Eb = loadit(path.join(folder_b, 'edgy.png'))
        
        if com_a is None: com_a = loadit(path.join(folder_a, 'com.json'))
        if com_b is None: com_b = loadit(path.join(folder_b, 'com.json'))
        
        nodes0 = algo.seed(*Ka.shape[:2], com_a, com_b)
        if not spin: com_a['a'], com_b['a'] = 0, 0
        
    else:
        nodes0 = algo.seed(*Ka.shape[:2])
        com_a  = dict(x=0, y=0, r=0, a=0.0)
        com_b  = dict(x=0, y=0, r=0, a=0.0)

    # Use CoM as repellant for edge nodes
    base = nodes0[4:]
    if thread and thread.abort: return
    
    # Match corners
    if docorners:
        shoutout('Matching corners' + label, thread=thread)
        if Ka is None: Ka = algo.load_rgba(settings['keyframes'][a])
        if Kb is None: Kb = algo.load_rgba(settings['keyframes'][b])
        
        catcher = settings['edge']['cornercatcher']
        catch_a = path.join(folder_a, catcher[a].lower())
        catch_b = path.join(folder_b, catcher[b].lower())
        
        if kp_a is None: kp_a = loadit(catch_a + '.csv')
        if kp_b is None: kp_b = loadit(catch_b + '.csv')
        if dc_a is None: dc_a = loadit(catch_a + '.png')
        if dc_b is None: dc_b = loadit(catch_b + '.png')
        
        nodes1, simi1 = algo.matchpoint(Ka, Kb, kp_a, kp_b, dc_a, dc_b,
                                        simisize=simisize, similim=similim)
        
        base = np.row_stack((base, nodes1))
        if thread and thread.abort: return
    
    # Extract and match silhouette key points
    if dosilhouette:
        shoutout('Matching silhouettes' + label, thread=thread)
        spawnpoints = min(1000, *settings['traject']['maxpoints'])
        
        sp_a = algo.spawn(Ea, base[:, [0, 1]], spawnpoints, r_min=simisize)
        sp_b = algo.spawn(Eb, base[:, [2, 3]], spawnpoints, r_min=simisize)
        n_half = int(spawnpoints / 2)
        
        nodes2, simi2 = algo.proximatch(Ka, Kb, Ea, sp_a, sp_b, com_a, com_b,
                                        neighbours=neighbours, n=n_half,
                                        simisize=simisize, similim=similim)
        
        nodes3, simi3 = algo.proximatch(Kb, Ka, Eb, sp_b, sp_a, com_b, com_a,
                                        neighbours=neighbours, n=n_half,
                                        simisize=simisize, similim=similim)
        
        try:
            nodes4 = np.row_stack((nodes2, nodes3[:, [2, 3, 0, 1]]))
            simi4 = np.append(simi2, simi3)
        except IndexError:
            nodes4, simi4 = nodes2, simi2
        if thread and thread.abort: return
    
    # Combine the results. One big happy family!
    if dosilhouette and docorners:
        nodez = np.row_stack((nodes1, nodes4))
        simiz = np.append(simi1, simi4)
    elif dosilhouette:
        nodez, simiz = nodes4, simi4
    elif docorners:
        nodez, simiz = nodes1, simi1
    else:
        nodez = []
    
    # Combine duplicates
    if len(nodez):
        shoutout('Combining duplicate trajectories' + label, thread=thread)
        nodez, simiz = algo.gettogether(nodez, simiz, simisize)
    
    # Discard excessive moves
    if len(nodez):
        shoutout('Discarding excessive moves' + label, thread=thread)
        diago = np.ceil(np.sqrt(Ka.shape[0] ** 2 + \
                                Ka.shape[1] ** 2))
        
        lim   = int(maxmove * diago)        
        keep  = algo.notsofast(nodez, lim, com_a, com_b)
        nodez = nodez[keep]
        simiz = simiz[keep]
        
        # Are we doing sensible things in this joint?
        print(timestamp() + 'Max move = {} px'.format(lim))
        if thread and thread.abort: return
    
    # In case of crossing paths discard the longest trajectory
    if len(nodez):
        shoutout('Discarding crossing paths' + label, thread=thread)
        keep = np.zeros_like(nodez, dtype=bool)
        repeat = 1
        while np.any(~keep) and repeat <= 10:
            if thread and thread.abort: return
            keep    = algo.straightenup(nodez)
            nodez   = nodez[keep]
            simiz   = simiz[keep]
            repeat += 1
    
    # Cherry pick nodes with the highest similarity score
    if len(nodez) > maxpoints:
        shoutout('Cherry picking' + label, thread=thread)
        seq   = np.argsort(simiz)[::-1]
        nodez = nodez[seq][:maxpoints]
        simiz = simiz[seq][:maxpoints]
    
    # Pack it all together into one cozy bundle
    if len(nodes0) and len(nodez):
        nodes = np.row_stack((nodes0, nodez))
    elif len(nodes0):
        nodes = nodes0
    else:
        nodes = nodez
    
    # Save the harvest
    saveit(nodes, f2)
    if thread and thread.abort: return
    
    # Fade to gray baby
    shoutout('Making trajectory chart' + label, thread=thread)
    channel_a = settings['edge']['channel'][a]
    channel_b = settings['edge']['channel'][b]
    if channel_a.lower().startswith('a'): channel_a = 'lightness'
    if channel_b.lower().startswith('a'): channel_b = 'lightness'
    Ga = algo.desaturate(Ka, channel_a)
    Gb = algo.desaturate(Kb, channel_b)
    
    # Produce a tingly trajectory chart       
    fig = algo.big_figure('MuddyMorph - Trajectories', *Ga.shape)
    if arc:
        comp_a, comp_b = com_a, com_b
    else:
        comp_a, comp_b = None, None
    try:
        tweens = settings['motion']['inbetweens'][m]
    except IndexError:
        tweens = algo.most_frequent_value(settings['motion']['inbetweens'])
    algo.movemap(Ga, Gb, nodes, comp_a, comp_b, tweens=tweens)
    plt.axis('off')
    plt.savefig(f1, **chartopts)
    plt.close(fig)
    
    # Our work here is done
    stopwatch += time()
    msg = 'Trajectory extraction took ' + duration(stopwatch)
    shoutout(msg, f1, thread)
    return nodes, Ka, Kb, com_a, com_b


def blobbify(settings, nodes, com_a, com_b, m, h=None, w=None):
    """
    Compute and save blob sizes for a given morph sequence,
    provided that blobs have been enabled.
    
    Usage
    -----
    >>> radii = blobbify(settings, nodes, com_a, com_b, m, h, w)
    """
    if not settings['motion']['blob'][m]: return
    
    print(timestamp() + 'Inflating blobs for morph {}'.format(m + 1))
    
    if not h or not w: h, w  = fetch_dimensions(settings)
    
    diago      = np.ceil(np.sqrt(h ** 2 + w ** 2))
    r_max      = int(diago)
    minsize    = abs_detail(settings, m, h, w)
    scale      = settings['render' ]['blobscale' ] / 100.
    neighbours = settings['traject']['neighbours'][m]
    
    radii = algo.inflate_blobs(nodes, scale, minsize,
                               neighbours, r_max, com_a, com_b)
    
    folder = path.join(settings['temppath'], 'm{0:03d}'.format(m + 1))
    
    saveit(radii, path.join(folder, 'radii.csv'))
    
    return radii


def inbetween(settings, m, savefile, d=0.5, f=0.5,
              Ka=None, Kb=None, nodes=None, radii=None,
              com_a=None, com_b=None, V=None, G=None, backfade=False):
    """
    Produce an inbetween frame, either by warping or blobbing.
    
    Usage
    -----
    >>> inbetween(settings, m, savefile, d, f)
    
    Returns
    -------
    File name of midpoint warp image in temporary folder.
    
    Notes
    -----
    Node trajectories and blob sizes (when not warping)
    must be precomputed. See *trajectory*.
    """
    
    # Key frame indices and output files
    a, b     = morph_key_indices(settings, m)
    folder_m = path.join(settings['temppath'], 'm{0:03d}'.format(m + 1))
    folder_a = path.join(settings['temppath'], 'k{0:03d}'.format(a + 1))
    folder_b = path.join(settings['temppath'], 'k{0:03d}'.format(b + 1))
    
    # Load images and trajectories
    if Ka    is None: Ka    = algo.load_rgba(settings['keyframes'][a])
    if Kb    is None: Kb    = algo.load_rgba(settings['keyframes'][b])
    if nodes is None: nodes = loadit(path.join(folder_m, 'nodes.csv'))
    
    # Center of mass is needed when we want arcs
    if settings['traject']['arc'][m]:
        if com_a is None: com_a = loadit(path.join(folder_a, 'com.json'))
        if com_b is None: com_b = loadit(path.join(folder_b, 'com.json'))
        if not settings['traject']['spin'][m]:
            com_a['a'], com_b['a'] = 0, 0
    else:
        com_a, com_b = None, None
    
    # To blob or not to blob?
    hardness = settings['render']['blobhardness']
    if settings['motion']['blob'][m] and radii is None:
        radii = loadit(path.join(folder_m, 'radii.csv'))
    
    # Time to do the deed
    T = algo.tween(Ka, Kb, nodes,
                   d=d, f=f, V=V, G=G,
                   com_a=com_a, com_b=com_b,
                   radii=radii, hardness=hardness, backfade=backfade)
    
    # Take the money and run
    fullfile = path.join(folder_m, savefile)
    qual = settings['render']['quality']
    algo.save_rgba(T, fullfile, qual)


def midpoint(settings, m, recycle=False,
             Ka=None, Kb=None, nodes=None,
             com_a=None, com_b=None, G=None):
    """
    Make a midpoint warp, for the purpose of reviewing trajectory quality;
    if the midpoint looks fine, then all inbetweens should be kosher.
    
    Regardless of the settings warping will be applied (no blobs),
    no edge vignetting is applied.
    
    Usage
    -----
    >>> f = midpoint(settings, m, recycle)
    """
    print(timestamp() + 'Making midpoint warp for morph {}'.format(m + 1))
    
    settingz = deepcopy(settings)
    settingz['motion']['blob'][m] = False
    
    savefile = path.join(settings['temppath'],
                         'm{0:03d}'.format(m + 1),
                         'mid.' + settings['render']['ext'])
    
    if recycle and path.isfile(savefile): return savefile
    
    inbetween(settingz, m, savefile,
              Ka=Ka, Kb=Kb, nodes=nodes,
              com_a=com_a, com_b=com_b, G=G)
    
    return savefile


def motion(settings, m, recycle_nodes=True, recycle_frames=True, thread=None,
           Ka=None, Kb=None, nodes=None, radii=None,
           com_a=None, com_b=None, X=None, Y=None, V=None, G=None):
    """
    Generate a series of bitmaps that together form a morph sequence.
    
    Usage
    -----
    >>> movie = motion(settings, m)
    
    Parameters
    ----------
    See *default_settings* for info on settings,
    and *trajectory* for details regarding thread.
    
    Notes
    -----
    - Start and stop key frames are included.
      This way it is easy to generate a preview per sequence.
    - Missing analysis data will be generated by invoking *trajectory*.
    - For the generation of single inbetween frames see *inbetween*.
    
    Returns
    -------
    A list of saved bitmaps.
    """
    
    # And so it begins
    movie = []
    n_m   = count_morphs(settings)
    msg   = 'Making motion for morph {}'.format(m + 1)
    shoutout(msg=msg, thread=thread)
    stopwatch = -time()
    
    # Prepare for battle
    frames   = np.arange(settings['motion']['inbetweens'][m] + 2)
    t        = 1. * frames / max(frames)
    d        = algo.motion_profile(t , settings['motion']['profile'][m])
    f        = algo.fade_profile(t, d, settings['motion']['fade'][m] * 0.01)
    a, b     = morph_key_indices(settings, m)
    comfiles = path.join(settings['temppath'], 'k{0:03d}', 'com.json')
    folder   = path.join(settings['temppath'], 'm{0:03d}'.format(m + 1))
    
    # Fetch the basic ingredients for inbetweening
    if Ka is None: Ka   = algo.load_rgba(settings['keyframes'][a])
    if Kb is None: Kb   = algo.load_rgba(settings['keyframes'][b])
    if X  is None: X, Y = algo.grid(Ka)
    if G  is None: G    = background(settings, Ka)
    if V  is None: V    = vinny(settings, Ka)
    if thread and thread.abort: return
    
    # Load precomputed goodies if we are allowed to
    if recycle_nodes:
        file_nodes = path.join(folder, 'nodes.csv')
        file_com_a = comfiles.format(a + 1)
        file_com_b = comfiles.format(b + 1)
        if nodes is None and path.isfile(file_nodes): nodes = loadit(file_nodes)
        if com_a is None and path.isfile(file_com_a): com_a = loadit(file_com_a)
        if com_b is None and path.isfile(file_com_b): com_b = loadit(file_com_b)
        if thread and thread.abort: return
    
    # Is stuff still missing? Then go and compute
    if nodes is None or com_a is None or com_b is None:
        result = trajectory(settings, m, recycle_nodes, thread, X=X, Y=Y,
                            Ka=Ka, Kb=Kb, com_a=com_a, com_b=com_b)
        
        if result is None or (thread and thread.abort): return
        nodes, _, _, com_a, com_b = result
    
    # Blob sizes
    if radii is None:
        radii = blobbify(settings, nodes,
                         com_a, com_b, m, *Ka.shape[:2])
    
    # If both key frames are opaque, then so should be all inbetweens
    backfade = algo.is_opaque(Ka) and algo.is_opaque(Kb)

    # Hop through the frames
    for i in frames:
        if thread and thread.abort: return
        
        basename = 'f{0:03d}.{1}'.format(i + 1, settings['render']['ext'])
        savefile = path.join(folder, basename)
        
        # Remember frame for playback or export
        movie.append(savefile)
        
        # Can we recycle existing material?
        if recycle_frames and path.isfile(savefile): continue
    
        # Time for a short newsflash
        if n_m > 1:
            msg = 'Generating morph {} frame {}'.format(m + 1, i + 1)
        else:
            msg = 'Generating frame {}'.format(i + 1)
        shoutout(msg, thread=thread)

        # Copy or generate the file we need
        if t[i] == 0 and can_copy_key(settings, a):
            msg  = 'Copying ' + semi_short_file_name(settings['keyframes'][a])
            msg += ' to '     + semi_short_file_name(savefile)
            print(timestamp() + msg)
            copyfile(settings['keyframes'][a], savefile)

        elif t[i] == 1 and can_copy_key(settings, b):
            msg  = 'Copying ' + semi_short_file_name(settings['keyframes'][b])
            msg += ' to '     + semi_short_file_name(savefile)
            print(timestamp() + msg)
            copyfile(settings['keyframes'][b], savefile)

        else:
            msg = 'Generating {0} with t={1:.0f}%, d={2:.0f}%, f={3:.0f}%'
            msg = msg.format(semi_short_file_name(savefile),
                             t[i]*100, d[i]*100, f[i]*100)
            print(timestamp() + msg)
            
            inbetween(settings, m, savefile, d=d[i], f=f[i],
                      Ka=Ka, Kb=Kb, nodes=nodes, radii=radii,
                      com_a=com_a, com_b=com_b, V=V, G=G, backfade=backfade)

        # Show the frame
        shoutout(img=savefile, thread=thread)

    # Peace out
    stopwatch += time()
    msg = 'Inbetweening took ' + duration(stopwatch)
    shoutout(msg, thread=thread)
    return movie


def render(settings, recycle=True, thread=None, savesettings=True):
    """
    Generate inbetween frames for the full project.
    
    This is in essence just a matter of invoking *make_motion* to generate the
    frames, and then copying these from the temporary path to export folder.
    
    By default existing frames will be recycled.
    
    Returns
    -------
    List of exported image files.
    
    Parameters
    ----------
    See *default_settings* for info on settings,
    and *trajectory* for details regarding thread.
    
    Notes
    -----
    Suppose the project name is 'morph', the export folder is
    '/Users/Me/Pictures/Muddy', and extension is jpg,
    then these files will be generated:
        
        - /Users/Me/Pictures/Muddy/morph.json (settings)
        - /Users/Me/Pictures/Muddy/morph001.jpg (first frame)
        - /Users/Me/Pictures/Muddy/morph002.jpg (second frame)
        - /Users/Me/Pictures/Muddy/morph003.jpg (third frame)
        - ... (and so forth)
    """
    
    # Shorthand name for render settings
    s = settings['render']
    
    # Enough fooling around, now is the time for serious business
    shoutout('Rendering ' + s['name'], thread=thread)
    stopwatch = -time()
    
    # First of all save the settings
    if savesettings:
        projectfile = path.join(s['folder'], s['name'] + '.json')
        save_settings(settings, projectfile)
    
    # Keep track of the exported files
    stash = []
    
    # Mesh grid and background are the same for all inbetweens
    G = background(settings)
    X, Y = algo.grid(G)
    if thread and thread.abort: return

    # Generate missing frames for all morphs
    # and fetch all file names while at it
    n_m = count_morphs(settings)
    morphs = []
    for m in range(n_m):
        tempfiles = motion(settings, m,
                           recycle, recycle,
                           thread, X=X, Y=Y, G=G)
        morphs.append(tempfiles)
        if thread and thread.abort: return
    
    # Export file name pattern
    exportfiles = path.join(s['folder'], s['name'] + '{0:03d}.' + s['ext'])

    # Copy files to export folder in chronological sequence
    print(timestamp() + 'Copying frames to export folder')
    f = 0
    for m in range(n_m):
        tempfiles = morphs[m]
        
        # Prevent duplicate frames
        # (The end of one morph is also the beginning of the next one)
        if m < n_m - 1 or s['loop'] or (m == n_m - 1 and s['reverse']):
            tempfiles = tempfiles[:-1]
            
        for tempfile in tempfiles:
            exportfile = exportfiles.format(f + 1)
            
            msg  = 'Copying ' + semi_short_file_name(tempfile)
            msg += ' to '     + semi_short_file_name(exportfile)
            shoutout(msg, thread=thread)
            if thread and thread.abort: return

            copyfile(tempfile, exportfile)
            stash.append(exportfile)
            shoutout(img=exportfile, thread=thread)
            f += 1

    # I put my thing down, flip it, and reverse it
    # (Same same, but different)
    if s['reverse']:
        print(timestamp() + 'Copying frames to export folder in reverse')
        for m in range(n_m - 1, -1, -1):
            tempfiles = morphs[m]
            if m > 0:
                tempfiles = tempfiles[1:]
            for tempfile in tempfiles[::-1]:
                exportfile = exportfiles.format(f + 1)
                
                msg  = 'Copying' + semi_short_file_name(tempfile)
                msg += ' to '    + semi_short_file_name(exportfile)
                shoutout(msg, thread=thread)
                if thread and thread.abort: return

                copyfile(tempfile, exportfile)
                stash.append(exportfile)
                shoutout(img=exportfile, thread=thread)
                f += 1
    
    # Celebrate good times, come on!
    stopwatch += time()
    msg = 'Finished. Rendering took ' + duration(stopwatch)
    shoutout(msg, thread=thread)
    return stash
    