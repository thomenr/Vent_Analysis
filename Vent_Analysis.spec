# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Vent_Analysis.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['nibabel==5.1.0', 'numpy==1.24.4', 'numpy-stl==2.13.0', 'scipy==1.10.1', 'SimpleITK==2.3.1', 'pydicom==2.4.3', 'pyMapVBVD==0.5.4', 'matplotlib==3.7.2', 'matplotlib-inline==0.1.6', 'Pillow==10.0.0'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Vent_Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['pirl.ico'],
)
