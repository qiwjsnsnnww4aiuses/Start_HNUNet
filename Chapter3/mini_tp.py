import argparse
import os
import torch
import torch.distributed as dist

def lab_init(desc):
    """实验统一入口：解析--device, 初始化进程组, 返回(dev,rank,world)"""
    """GPU机器上默认cuda/nccl  笔记本退回cpu/gloo"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    backend = "nccl" if args.device == "cuda" else "gloo"

    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")
    return dev, rank, world