# LLM Docker Compose

Machine-scoped Docker Compose deployments for local LLM services.

## Layout

- `spark/`: NVIDIA DGX Spark host at `192.168.0.33`.
- `hyper03/`: NVIDIA RTX PRO 6000 Blackwell host for GLM-4.5-Air 4-bit/NVFP4.

Each machine directory should contain the Compose files used on that host.
