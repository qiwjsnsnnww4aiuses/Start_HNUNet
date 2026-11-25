import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：子组")
assert world == 4, "请用 torchrun --nproc_per_node=4运行"
g01 = dist.new_group([0, 1])
g23 = dist.new_group([2, 3])

t = torch.tensot([rank + 1.0], device=dev)
if rank in (0, 1):
    dist.all_reduce(t, group=g01)
elif rank in (2, 3):
    dist.all_reduce(t, group=g23)

print(f"rank {rank}:组内 all_reduce = {t.item()}", flush=True)
dist.destroy_process_group()