# RAPIDS shim: how close can the GPU image get to OpenFlo's pins?

OpenFlo pins **numpy 2.4.6** and **pandas 3.0.3** (`pyproject.toml`, `requirements.txt`).
RAPIDS (cuDF/cuML/cuGraph) ships its own numpy/pandas. This note records the
empirical answer to "can we shim RAPIDS onto our exact pins?" so the decision in
`docker/Dockerfile.rapids` is reproducible.

## Method

Probe script `scripts`-style one-shot (see commit history): in a clean RAPIDS
container, import cuDF/cuML/cuGraph at the stock versions, then force `numpy==2.4.6`,
then force `pandas==3.0.3`, re-importing after each and capturing the exact failure.
Run against two bases:

```
docker run --rm --gpus all rapidsai/base:24.12-cuda12.5-py3.12 bash probe.sh
docker run --rm --gpus all rapidsai/base:26.06-cuda12-py3.12   bash probe.sh
```

## Results

| Base | stock numpy / pandas | force numpy 2.4.6 | force pandas 3.0.3 |
|---|---|---|---|
| `24.12-cuda12.5-py3.12` | 2.0.2 / 2.2.3 — cuDF/cuML/cuGraph OK | **FAIL** `Numba needs NumPy 2.0 or less. Got NumPy 2.4` | (n/a — numpy already broke import) |
| `26.06-cuda12-py3.12` | **2.4.6** / 2.3.3 — cuDF/cuML/cuGraph OK | OK (already 2.4.6) | **FAIL** `AttributeError: module 'pandas.api.types' has no attribute 'is_interval'` |

## Conclusions

- **numpy parity — ACHIEVED (exact).** The 24.12 blocker was *Numba ↔ NumPy*: the
  Numba in RAPIDS 24.12 hard-caps NumPy at 2.0, so our 2.4.6 could not load. RAPIDS
  **26.06** ships a newer Numba and stock **numpy 2.4.6 — our exact pin, for free.**

- **pandas parity — UPSTREAM WALL.** Even the newest cuDF (26.6) requires
  `pandas <2.4.0` (`numpy <3.0,>=1.26` for numpy). Forcing our `pandas==3.0.3`
  breaks cuDF/cuML/cuGraph at import: pandas 3.0 removed `pandas.api.types.is_interval`,
  which cuDF's pandas-compat layer calls. An environment holds exactly one pandas,
  so cuDF and our 3.0.3 pin are **mutually exclusive** until RAPIDS ships pandas-3
  support. `is_interval` is only the *first* such call; monkeypatching cuDF's
  pandas internals would be a fragile whack-a-mole that risks **silent correctness
  drift** in a tool whose entire value is golden parity — so we do not do it.

## Decision (encoded in `docker/Dockerfile.rapids`)

Bump the base **24.12 → 26.06**. This buys exact numpy parity (2.0.2 → 2.4.6) and
moves pandas one minor closer (2.2.3 → 2.3.3), keeping the full RAPIDS GPU stack.
Behavioral parity is still validated by the golden self-test (the prior round
confirmed 7/7 on 24.12 with `leidenalg==0.11.0`; 26.06 is closer to our pins).

**Full exact-pin parity (pandas 3.0.3 + cuDF) is not achievable today** and is
blocked upstream in RAPIDS, not in OpenFlo. Re-check when a RAPIDS release advertises
pandas-3 support.

### The two GPU images, restated

- **`Dockerfile.gpu` (CuPy)** — OpenFlo's *exact* pins (numpy 2.4.6 / pandas 3.0.3),
  full suite + golden green, accelerates compensation/arcsinh/density. **No GPU
  clustering** (no cuGraph). The dependency-identical image.
- **`Dockerfile.rapids`** — adds GPU clustering (cuML/cuGraph); numpy exact, pandas
  one minor below ours (2.3.3). Behaviorally synonymous (golden), not pin-identical.
