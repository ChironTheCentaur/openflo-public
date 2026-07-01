# Docker — Linux CI parity + GPU (RAPIDS/CuPy) test bed

Two images. Build from the **repo root** (the build context must be `.`).

## CPU — local Linux CI mirror (`Dockerfile`)
A faithful mirror of the `ubuntu-latest` CI job: same pinned tools, Xvfb for the
Tk GUI tests, `requirements.txt` runtime surface. Use it to confirm the suite —
**including the golden baseline** — passes on Linux exactly as on Windows
(i.e. results are *synonymous* across OSes).

```bash
docker build -f docker/Dockerfile -t openflo-ci .
docker run --rm openflo-ci      # ruff -> pyright(Linux) -> pytest -> golden
```

> Note: CI already runs this on every push (ubuntu **and** windows, both green
> with the golden baseline), so Linux↔Windows parity is verified automatically.
> This image is for reproducing/iterating on it locally.

## GPU — CuPy (`Dockerfile.gpu`)
Self-contained CuPy image (same stack as the CPU image + CuPy + the CUDA lib
wheels). Exercises OpenFlo's CuPy GPU paths — compensation, logicle/hyperlog,
event-density, KDE — on a real GPU; `tests/test_gpu_accel.py`'s GPU-parity tests
(skipped on CPU CI) run here and assert the GPU results match the CPU /
flowutils / scipy reference.

```bash
docker build -f docker/Dockerfile.gpu -t openflo-gpu .
docker run --rm --gpus all openflo-gpu          # parity tests + suite + golden
```

**Verified** here (WSL2, RTX 3080 Ti): `gpu_available=True` in-container, all 11
GPU-parity tests pass, golden HEALTHY.

It is deliberately **not** based on `rapidsai/base` — RAPIDS pins numpy/pandas
versions that conflict with OpenFlo's pinned stack. So the RAPIDS-only paths
(cuML UMAP, cuGraph clustering in `pipeline._probe_gpu`) are NOT covered here; a
rapidsai-based image with its own dependency reconciliation is a separate task.

### GPU prerequisites — nvidia-container-toolkit (one-time, in the engine)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && service docker restart
```

## Caveats / prerequisites
- **Disk:** images are large (~10 GB CPU, more for GPU). Keep the engine's
  data-root off a full C: — on this box the WSL2 `docker` distro lives on D:
  (Docker root `/var/lib/docker`), so images land there.
- Both build from the **repo root** as context; `.dockerignore` keeps it small
  (excludes `.venv`, `.git`, outputs).
