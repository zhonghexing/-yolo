# -*- mode: python ; coding: utf-8 -*-
"""
钢材缺陷检测系统 - 桌面版打包配置（可部署到任意电脑）
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
        (str(PROJECT_ROOT / 'best.pt'), '.'),
        (str(PROJECT_ROOT / 'yolov8n.pt'), '.'),
        (str(PROJECT_ROOT / 'inference.py'), '.'),
        (str(PROJECT_ROOT / 'constants.py'), '.'),
        (str(PROJECT_ROOT / 'feedback.py'), '.'),
        (str(PROJECT_ROOT / 'db.py'), '.'),
        (str(PROJECT_ROOT / 'web_dashboard.py'), '.'),
        (str(PROJECT_ROOT / 'visualization.py'), '.'),
        (str(PROJECT_ROOT / 'app_icon.ico'), '.'),
    ],
    hiddenimports=[
        # PyQt5
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',
        'PyQt5.sip',
        # OpenCV
        'cv2',
        # NumPy
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        # PyTorch
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.nn.modules',
        'torch.optim',
        'torch.utils',
        'torch.utils.data',
        'torchvision',
        'torchvision.transforms',
        # Ultralytics
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.utils',
        'ultralytics.utils.checks',
        'ultralytics.utils.torch_utils',
        'ultralytics.cfg',
        'ultralytics.engine',
        'ultralytics.engine.model',
        'ultralytics.engine.predictor',
        'ultralytics.engine.results',
        # Flask
        'flask',
        'flask.json',
        'flask_socketio',
        'engineio',
        'engineio.async_drivers.threading',
        'socketio',
        # 语音
        'pyttsx3',
        'pyttsx3.drivers',
        'pyttsx3.drivers.sapi5',
        # 其他
        'sqlite3',
        'PIL',
        'PIL.Image',
        'requests',
        'yaml',
        'matplotlib',
        'scipy',
        'psutil',
        'tqdm',
        'pandas',
        'seaborn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'unittest',
        'xmlrpc',
        'pydoc',
        'doctest',
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
