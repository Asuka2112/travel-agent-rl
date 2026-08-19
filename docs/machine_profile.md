# Machine profile

Captured 2026-08-19 before any project job was launched.

| Item | Observed |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-83 |
| CPU | 2 x AMD EPYC 7763, 256 logical CPUs |
| RAM | 503 GiB total, 422 GiB available |
| Data filesystem | 15 TiB total, 8.4 TiB free |
| GPU | 8 x NVIDIA RTX 4090, 24,564 MiB each |
| Driver / CUDA | 550.144.03 / 12.4 |
| Toolkit | nvcc 12.4.131 |
| Python / Git | 3.10.12 / 2.34.1 |

GPU 0, 3, 4, and 7 had compute workloads from multiple users at audit time.
GPUs 1, 2, 5, and 6 were nearly empty. This snapshot is not a reservation;
every GPU command must repeat the check and must not terminate unknown jobs.

