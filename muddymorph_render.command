#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Re)render the current MuddyMorph project on a Mac.

Re-rendering can be useful for continuing a job after program crash or
accidental abort, without the need to plough through all wizard screens again.
"""

import os
import muddymorph_go as gogo

scriptfile = os.path.abspath(__file__)
folder = os.path.dirname(scriptfile)
os.chdir(folder)

settings = gogo.load_settings()
gogo.render(settings)

