import time
import torch

from spd_ops import *
from spd_ops_naive import *


def sync(device):
    # sinon les temps sont faux sur gpu / mps
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_it(fn, *args, iters=20, warmup=5, device=None):
    # petit benchmark 
    for _ in range(warmup):
        fn(*args)
    if device:
        sync(device)

    t0 = time.time()
    for _ in range(iters):
        fn(*args)
    if device:
        sync(device)

    return (time.time() - t0) / iters

def main():
    mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu = torch.device("cpu")

    batch = 256
    n = 32

    print(f"device = {mps}, batch = {batch}, n = {n}\n")
    print("benchmark des fonctions :\n")

    # --- vectorize / devectorize : ok sur mps
    M_mps = random_spd_batch(batch, n, device=mps)
    v_mps = vectorize_lower(M_mps)

    print("vectorize:",
          time_it(vectorize_lower_naive, M_mps, iters=3, device=mps),
          time_it(vectorize_lower, M_mps, iters=30, device=mps))

    print("devectorize:",
          time_it(devectorize_lower_naive, v_mps, n, iters=3, device=mps),
          time_it(devectorize_lower, v_mps, n, iters=30, device=mps))

    # --- sqrt / log : eigh pas supporté sur mps -> on force cpu
    print("\n(sqrt/log sur cpu car eigh n'est pas dispo sur mps)\n")

    M_cpu = random_spd_batch(batch, n, device=cpu)

    print("sqrt:",
          time_it(spd_matrix_sqrt_naive, M_cpu, iters=3, device=cpu),
          time_it(spd_matrix_sqrt, M_cpu, iters=20, device=cpu))

    print("log:",
          time_it(spd_matrix_log_naive, M_cpu, iters=3, device=cpu),
          time_it(spd_matrix_log, M_cpu, iters=20, device=cpu))


if __name__ == "__main__":
    main()
