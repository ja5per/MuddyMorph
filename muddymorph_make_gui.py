#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile QT Designer XML user interface definition to python code.

For compiling the project resources (icons and images),
open a terminal window and run this command:

pyrcc5 muddymorph.qrc > muddymorph_rc.py
"""

from glob import glob
from os import path
from PyQt5 import uic

for f_in in glob('*.ui'):
    base, _  = path.splitext(path.basename(f_in))
    f_out    = base.replace(' ', '_') + '_ui.py'
    
    print("Generating {} ...".format(f_out), end="")
    with open(f_out, 'w', encoding='utf-8') as doc:
        uic.compileUi(f_in, doc)
    print("done")
