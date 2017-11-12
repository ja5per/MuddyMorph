# -*- mode: python -*-

##################
# Build settings #
##################

block_cipher = None

a = Analysis(['MuddyMorph.py'],
             pathex=[SPECPATH],
             binaries=[],
             datas=[],
             hiddenimports=['scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries = True,
          name             = 'MuddyMorph',
          debug            = False,
          strip            = False,
          upx              = False,
          console          = True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip = False,
               upx   = False,
               name  = 'MuddyMorph')

#app = BUNDLE(coll,
#             name='MuddyMorph.app',
#             icon='guicandy/muddymorph.icns',
#             bundle_identifier=None,
#             info_plist={'NSHighResolutionCapable': 'True'})
