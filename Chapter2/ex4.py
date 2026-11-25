import torch
import torch.distributed as dist
import time
from mini_tp import lab_init
dev, rank, world = lab_init("Exercise 04")


t = torch.randn(10, device=dev)
t_native = t.clone()

if rank==0:
    G = [torch.empty_like(t) for _ in range(world)]
    G[0] = t.clone()
    for i in range(1, world):
        dist.recv(G[i], src=i)
    total = sum(G)
    t = total
    for i in range(1, world):
        dist.send(t, dst=i)
else:
    dist.send(t, dst=0)
    dist.recv(t, src=0)
torch.cuda.synchronize()
dist.barrier()




t = torch.randn(10, device=dev)
t_native = t.clone()

dist.barrier()
torch.cuda.synchronize()
start_time = time.time()
if rank==0:
    G = [torch.empty_like(t) for _ in range(world)]
    G[0] = t.clone()
    
    for i in range(1, world):
        dist.recv(G[i], src=i)
    print(f"Data all Received!")

    total = sum(G)
    t = total

    for i in range(1, world):
        dist.send(t, dst=i)
else:
    dist.send(t, dst=0)
    dist.recv(t, src=0)
torch.cuda.synchronize()
end_time = time.time()
print("Senc/Recv Finished!")

dist.barrier()
torch.cuda.synchronize()
start_native_time = time.time()
dist.all_reduce(t_native)
torch.cuda.synchronize()
end_native_time = time.time()
print("All_Reduce Finished!")

if torch.allclose(t, t_native):
    print(f"Verified!: Rank{rank}\n")
else:
    print(f"Fail!: Rank{rank}\n")

if rank==0:
    print(f"Send/Recv  time:{(end_time-start_time)*1000:.4f}ms\n")
    print(f"All_Reduce time:{(end_native_time-start_native_time)*1000:.4f}ms\n")