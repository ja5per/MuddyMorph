#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try out basic logistical manoeuvres:
    
    1) Get default settings
    2) Clean up temporary folder 
    3) Save settings to file
"""

import muddymorph_go as gogo

settings = gogo.default_settings()

keyfolder = '/Users/jasper/Documents/PythonSpul/muddymorph/testcases'
settings['keyframes'] = [keyfolder + '/rabbitguard1.png',
                         keyfolder + '/rabbitguard2.png']

gogo.init_temp(settings)

gogo.save_json(settings, 'muddymorph.json')

