import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：集合通信原语")
me = float(rank+1)

def report(name, t):
    "把各个rank手里的数据t汇总到rank 0,打印全局视角"
    outs = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(outs, t.reshape(1))
    if rank == 0:
        cells = " ".join(f"r{i}={o.item():.0f}" for i, o in enumerate(outs))
        print(f"{name:<15} {cells}", flush=True)
    
report("初始", torch.tensor([me], device=dev))

# broadcast
t = torch.tensor([99.0 if rank==0 else 0.0], device=dev)
dist.broadcast(t, src=0)
report("broadcast", t)

# all_reduce
t = torch.tensor([me], device=dev)
dist.all_reduce(t)
report("all_reduce", t)

#reduce
t = torch.tensor([me], device=dev)
dist.reduce(t, dst=0)
if rank == 0:
    print(f"reduce    只有r0拿到{t.item():.0f}, 其他rank手里的值不作数", flush=True)

# all_gather
t = torch.tensor([me], device=dev)
outs = [torch.empty_like(t) for _ in range(world)]
dist.all_gather(outs, t)
if rank==0:
    print(f"all_gather 每个人手里都是完整列表 {[int(o.item()) for o in outs]}", flush=True)

t = torch.tensor([me*(10**i) for i in range(world)], device=dev)
out = torch.empty(1, device=dev)
dist.reduce_scatter_tensor(out, t)
report("reduce_scatter", out)

# send/recv
if rank==0:
    t = torch.tensor([77.0], device=dev)
    dist.send(t, dst=1)
elif rank==1:
    t = torch.empty(1, device=dev)
    dist.recv(t, src=0)
    print(f"senf/recv   r1收到r0寄来的{t.item():.0f}", flush=True)

dist.destroy_process_group()