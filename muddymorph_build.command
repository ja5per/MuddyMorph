#!/bin/bash

#################
# Build Mac App #
#################

# Go to where the action is
cd /Users/jasper/Documents/PythonSpul/MuddyMorph 

## For full blown build:
## Use tailored options from spec file
pyinstaller muddymorph_make.spec

## For testing and debugging:
## Use default options (produces new spec file)
# pyinstaller MuddyMorph.py
# pyinstaller --hiddenimport _sysconfigdata_m_darwin_ MuddyMorph.py

## Notes to self:
# - Previously hidden import pywt._extensions._cwt was required and the build worked,
#   but after an anaconda update on 2017-11-5 everything broke down.
#   Yet to be fixed, probably will need to ask for support.
# - The onefile option does not work; PyQt5 gets very confused.
