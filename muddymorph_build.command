#!/bin/bash
# Build Mac executable slash application using PyInstaller

cd /Users/jasper/Documents/PythonSpul/muddymorph 

# Use options from spec file
pyinstaller MuddyMorph.spec

# 
# pyinstaller --hiddenimport _sysconfigdata_m_darwin_ MuddyMorph.py

# Notes to self:
# - Previously hidden import pywt._extensions._cwt was required and the build worked,
#   but after an anaconda update on 2017-11-5 everything broke down.
#   Yet to be fixed, probably will need to ask for support.
# - The onefile option does not work; PyQt5 gets very confused.
