import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("Exercise 02 2")

if rank==0:
    t = torch.randn(10, device=dev)
else:
    t = torch.empty(10, device=dev)

dist.broadcast(t, src=0)

outs = [torch.empty_like(t) for _ in range(world)]
dist.all_gather(outs, t)

if rank == 0:
    all_same = all(torch.equal(outs[0], outs[i]) for i in range(world))
    if all_same:
        print(f"Verified! Content:{outs[0]}")
    else:
        print(f"Expected:{outs[0]} but FAILED!")