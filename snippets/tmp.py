from cyccon.extension.dist_chamfer_idx import chamferDist
import torch

chamfer = chamferDist()

dev = torch.device('cuda')

B = 32
N = 2500

a = torch.empty((B, N, 3), dtype=torch.float32).uniform_(0., 1.).to(dev)
b = torch.empty((B, N, 3), dtype=torch.float32).uniform_(0., 1.).to(dev)

while True:
    chamfer(a, b)
