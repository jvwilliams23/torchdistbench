# torchdistbench

A set of benchmarks for `torch.distributed` library. Besides a pytorch installation, `numpy` and `matplotlib` are needed (see `requirements.txt`).

Launch as:

```bash
export NCCL_DEBUG=INFO # optional
torchrun --nproc-per-node=2 allreduce.py
```
