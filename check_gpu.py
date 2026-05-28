# -*- coding: utf-8 -*-
import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')

print('='*50)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    mem = torch.cuda.get_device_properties(0).total_memory
    print('显存:', round(mem / 1024**3, 1), 'GB')
    print('Compute Capability:', torch.cuda.get_device_capability(0))
print('='*50)
