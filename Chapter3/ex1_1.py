import time
import torch
import torch.distributed as dist
from mini_tp import lab_init
dev, rank, world = lab_init("exp3")

t = torch.ones(32, 32, device=dev)  # 4KB

for i in range(6):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(t)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    if rank==0:
        print(f"第{i}次 all_reduce: {(time.perf_counter()-t0)*1e3:.3f}ms")

dist.destroy_process_group()