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
    parser.add_argument("--resultsname", "-i", type=str, default="results_allreduce.npz")
    return parser.parse_args()

def all_reduce(send_tensor):
    handle = dist.all_reduce(send_tensor)
    # handle.wait()



if __name__ == "__main__":
    args = get_inputs()

    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # os.environ["NCCL_IB_HCA"] = f"mlx5_{rank}"
        if torch.cuda.is_available():
            device = torch.device('cuda', rank)
            torch.cuda.set_device(device)
            dist.init_process_group("nccl")
            print("gpu found, using cuda/nccl")
        else:
            device = torch.device('cpu', rank)
            dist.init_process_group("gloo")
            print("Using gloo")
            

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
            send_tensor = torch.randn(tensor_size).to(device)
            time_list = []
            for _ in range(200):
                t0 = time.time()
                all_reduce(send_tensor)
                if torch.cuda.is_available():
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
        stats_list["device_type"] = device.type
        stats_list["torch_version"] = torch.__version__
        args.resultsname = f"results_allreduce_world{dist.get_world_size()}_{device.type}.npz"
        if dist.get_rank() == 0:
            np.savez(args.resultsname, **stats_list)



    if rank == 0:
        hostname = os.environ["HOSTNAME"]
        data = np.load(args.resultsname)
        x = data["sizes"]
        y = data["median"]
        ngpus = data["world_size"]
        version = data["torch_version"]
        device_type = data["device_type"]
        fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))
        ax.plot(x,y)
        # ax.set_xscale("log", base=2)
        # ax.set_yscale("log", base=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(f"Median time [s]")
        ax.set_xlabel(f"Tensor size [-]")
        ax.set_title(f"Pytorch AllReduce test. \nHost: {hostname}. " rf"$N_{{device}}$ = {ngpus}" + f"\nVersion {version}, Device {device_type}")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.savefig("plot_allreduce.png", dpi=300)

    try:
        dist.destroy_process_group()
    except:
        pass