# -*- mode: python -*-

block_cipher = None


a = Analysis(['MuddyMorph.py'],
             pathex=['/Users/jasper/Documents/PythonSpul/muddymorph'],
             binaries=[],
             datas=[],
             hiddenimports=['_sysconfigdata_m_darwin_'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='MuddyMorph',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MuddyMorph')
