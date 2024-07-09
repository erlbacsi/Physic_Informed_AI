import torch
import os, sys

import matplotlib.pyplot as plt

def save_checkpoint(path:str, model, epoch:int):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)

def load_checkpoint(path:str):
    assert os.path.isfile(path)
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint

def visualize_result(x, y, c,cmap='Reds', min_val=None, max_val=None, name=None, label='K'):
    if not min_val:
        min_val = torch.min(c)
    if not max_val:
        max_val = torch.max(c)
    if not name:
        name = 'output.png'
    plt.scatter(x, y, c=c, cmap=cmap, vmin=min_val, vmax=max_val)
    plt.colorbar(label=f'[{label}]', pad=0.2)
    plt.savefig(name)
    plt.clf()

def visualize_stress_diff(x, y, c,cmap='Reds', name=None, label='MPa'):
    min_val = torch.min(c)
    max_val = torch.max(c)
    if not name:
        name = 'output.png'
    plt.scatter(x, y, c=c, cmap=cmap, vmin=min_val, vmax=max_val)
    plt.colorbar(label=f'[{label}]', pad=0.2)
    plt.savefig(name)
    plt.clf()