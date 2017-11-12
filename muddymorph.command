#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start the MuddyMorph app on a Mac.

Make this file executable by opening a terminal and typing:
chmod -x muddymorph.command

After that double-clicking this file in Finder should be enough to get things going.
"""

import os
import MuddyMorph

scriptfile = os.path.abspath(__file__)
folder = os.path.dirname(scriptfile)
os.chdir(folder)

try:
    MuddyMorph.main()
except:
    # Prevent annoying 'python 3 has crashed' alerts
    # after each and every program close event.
    # In case of a real error, the crash report 'muddymorph_lasterror.log'
    # ought to be created, providing details on what exactly went wrong.
    print('Game Over!\n')
