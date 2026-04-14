import argparse
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skiptest", "-s", action="store_true")
    parser.add_argument("--resultsname", "-i", type=str, default="results_sendrecv.npz")
    return parser.parse_args()

def send_recv(send_minus, send_plus):
    recv_minus = torch.zeros_like(send_minus)
    recv_plus = torch.zeros_like(send_plus)
    plus_rank = rank + 1
    minus_rank = rank - 1

    ops = []
    if rank > 0:
        ops += [
            dist.P2POp(dist.irecv, recv_minus, minus_rank),
            dist.P2POp(
                dist.isend,
                send_minus.contiguous(),
                minus_rank,
            ),
        ]
    if rank < (world_size - 1):
        ops += [
            dist.P2POp(
                dist.isend,
                send_plus.contiguous(),
                plus_rank,
            ),
            dist.P2POp(dist.irecv, recv_plus, plus_rank),
        ]
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()



if __name__ == "__main__":
    args = get_inputs()

    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        os.environ["NCCL_IB_HCA"] = f"mlx5_{rank}"

        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    except:
        rank = 1
        world_size = 1
        args.skiptest = True

    if not args.skiptest:
        stats_list = {
            "sizes":[],
            "median":[],
            "all":[],
        }
        tensor_size = 2
        for test_id in range(28):
            send_minus = torch.randn(tensor_size).to(device)
            send_plus = torch.randn(tensor_size).to(device)
            time_list = []
            for _ in range(200):
                t0 = time.time()
                send_recv(send_minus, send_plus)
                torch.cuda.synchronize()
                if dist.get_rank() == 0:
                    time_list.append(time.time()-t0)
            if dist.get_rank() == 0:
                print(f"test: {test_id} tensor size {tensor_size} (~ {tensor_size/512**3}), median {np.median(time_list)} s")
                stats_list["sizes"].append(tensor_size)
                stats_list["median"].append(np.median(time_list))
                stats_list["all"].append(np.array(time_list))
            tensor_size *= 2

        stats_list["world_size"] = dist.get_world_size()
        stats_list["torch_version"] = torch.__version__
        args.resultsname = f"results_sendrecv_world{dist.get_world_size()}.npz"
        if dist.get_rank() == 0:
            np.savez(args.resultsname, **stats_list)



    if rank == 0:
        hostname = os.environ["HOSTNAME"]
        data = np.load(args.resultsname)
        x = data["sizes"]
        y = data["median"]
        ngpus = data["world_size"]
        version = data["torch_version"]
        fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))
        ax.plot(x,y)
        # ax.set_xscale("log", base=2)
        # ax.set_yscale("log", base=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(f"Median time [s]")
        ax.set_xlabel(f"Tensor size [-]")
        ax.set_title(f"Pytorch SendRecv test. \nHost: {hostname}. " rf"$N_{{GPU}}$ = {ngpus}" + f"\nVersion {version}")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.savefig("plot_sendrecv.png", dpi=300)

    try:
        dist.destroy_process_group()
    except:
        pass