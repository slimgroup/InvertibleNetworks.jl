# Test memory usage of normflows as network depth increases

#salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=30G srun --pty python 

import torch
import nvidia_smi

def _get_gpu_mem(synchronize=True, empty_cache=True):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return mem.used

def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if hook_type == 'pre':
            return
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1
        mem_all = _get_gpu_mem()
        torch.cuda.synchronize()
        lname = type(self).__name__
        lname = 'conv' if 'conv' in lname.lower() else lname
        lname = 'ReLU' if 'relu' in lname.lower() else lname
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': f"{lname}_{hook_type}",
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
        })
    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)
    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)
    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)


def log_mem(model, inp, mem_log=None, exp=None):
    nvidia_smi.nvmlInit()
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)
    try:
        out = model(inp)
        loss = out.sum()
        loss.backward()
    except Exception as e:
        print(f"Errored with error {e}")
    finally:
        [h.remove() for h in hr]
    return mem_log

# Used this in InvertibleNetowrks.jl so should use here. Though didnt see much difference in performance
import os
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"

import pandas as pd
import torch
import torchvision as tv
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm 

torch.manual_seed(0)

L = 3
nx = 256
max_mems = []

for depth in [4,8,16,32,48,64]:
    print(depth)
    K = depth
    input_shape = (3, nx, nx)
    n_dims = np.prod(input_shape)
    channels = 3
    hidden_channels = 256
    split_mode = 'channel'
    scale = True

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                         split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.DiagGaussian(latent_shape,trainable=False)]
    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)
    batch_size = 8
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.),tv.transforms.Resize(size=input_shape[1])])
    train_data = tv.datasets.CIFAR10('datasets/', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
    train_iter = iter(train_loader)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x, y = next(train_iter)

    mem_log = []
    try:
        mem_log.extend(log_mem(model, x.to(device), exp='std'))
    except Exception as e:
        print(f'log_mem failed because of {e}')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    df = pd.DataFrame(mem_log)
    max_mem = df['mem_all'].max()/(1024**3)
    print("Peak memory usage: %.2f Gb" % (max_mem,))
    max_mems.append(max_mem)

# >>> max_mems
# [5.0897216796875, 7.8826904296875, 13.5487060546875, 24.8709716796875, 36.2010498046875, 39.9959716796875]