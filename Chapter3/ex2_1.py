import torch
import torch.distributed as dist
from mini_tp import lab_init
dev, rank, world = lab_init("exp4")

t = torch.ones(1024*4, 1024*4, device=dev)  # 64MB

for _ in range(3):
    dist.all_reduce(t)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.CUDA]) as prof:
    dist.all_reduce(t)
    torch.cuda.synchronize()

if rank==0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
dist.destroy_process_group()

#   NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P torchrun --nproc_per_node=2 ex2_1.py 2>&1 | grep -E "Ring|via|comm" | head -10