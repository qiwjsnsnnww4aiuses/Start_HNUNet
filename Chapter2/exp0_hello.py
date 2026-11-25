import torch
import torch.distributed as dist
from mini_tp import lab_init
dev, rank, world = lab_init("Environ Check")
if dev.type == "cuda":
    print(f"rank {rank}/{world}: {torch.cuda.get_device_name(dev)} @ {dev}", flush=True)
else:
    print(f"rank {rank}/{world}:CPU(gloo后端)", flush = True)

t = torch.tensor([rank + 1.0], device = dev)
dist.all_reduce(t)
expect = world*(world+1)/2
status = "OK" if t.item()==expect else "FAIL"
print(f"ran {rank}: all_reduce result = {t.item()}(expected {expect})[{status}]")
dist.destroy_process_group()