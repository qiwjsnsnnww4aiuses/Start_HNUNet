# all_reduce = reduce+broadcast
import torch
import torch.distributed as dist
import time
from mini_tp import lab_init
dev, rank, world = lab_init("Exercise 5")

t = torch.randn(10, device=dev)
t_native = t.clone()

dist.barrier()
torch.cuda.synchronize()
start_time = time.time()
dist.reduce(t, dst=0)
dist.broadcast(t, src=0)
torch.cuda.synchronize()
end_time = time.time()

dist.barrier()
torch.cuda.synchronize()
start_native_time = time.time()
dist.all_reduce(t_native)
torch.cuda.synchronize()
end_native_time = time.time()

if torch.allclose(t, t_native):
    print(f"Verified!: Rank{rank}")
    print(f"Reduce+Broadcast:{t}\nAll_Reduce:{t_native}")
else:
    print(f"Fail!: Rank{rank}")
    print(f"Reduce+Broadcast:{t}\nAll_Reduce:{t_native}")

dist.barrier()
if rank==0:
    print(f"Reduce+Broadcast  time:{(end_time-start_time)*1000:.4f}ms")
    print(f"All_Reduce        time:{(end_native_time-start_native_time)*1000:.4f}ms")