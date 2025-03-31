import atexit
import os
import sys
import torch

root = os.path.dirname(os.getcwd())
sys.path.append(root)

@atexit.register
def _():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清空 CUDA GPU 显存。")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("已清空 MPS 显存。")