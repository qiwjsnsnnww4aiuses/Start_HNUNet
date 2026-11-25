import torch
import torch.distributed as dist
from mini_tp import lab_init
dev, rank, world = lab_init("exp4")

t = torch.ones(1024, 1024, device=dev)

for _ in range(3):
    dist.all_reduce(t)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.CUDA]) as prof:
    dis.all_reduce(t)
    torch.cuda.synchronize()

if rank==0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
dist.destroy_process_group()
