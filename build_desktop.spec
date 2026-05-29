# -*- mode: python ; coding: utf-8 -*-
"""
钢材缺陷检测系统 - 桌面版打包配置
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH)

block_cipher = None

a = Analysis(
    [str(PROJECT_ROOT / 'app.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / 'runs' / 'train' / 'screw_defect-11' / 'weights' / 'best.pt'), '.'),
        (str(PROJECT_ROOT / 'inference.py'), '.'),
        (str(PROJECT_ROOT / 'constants.py'), '.'),
        (str(PROJECT_ROOT / 'feedback.py'), '.'),
        (str(PROJECT_ROOT / 'visualization.py'), '.'),
        (str(PROJECT_ROOT / 'app_icon.ico'), '.'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'ultralytics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'seaborn',
        'pandas',
        'scipy',
        'flask',
        'flask_socketio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='钢材缺陷检测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'app_icon.ico'),
)
