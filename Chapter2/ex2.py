import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("Exercise 02")

if rank==0:
    t = torch.randn(10, device=dev)
    expect = t.clone()
else:
    t = torch.empty(10, device=dev)
dist.broadcast(t, src=0)


if rank == 0:
    is_same = torch.allclose(t, expect)
else:
    print(f"Rank {rank} received:{t}")

if rank==0:
    if torch.allclose(t, expect):
        print(f"Verified! Content:{t}")
    else:
        print(f"Expected:{expect} but FAILED!")


