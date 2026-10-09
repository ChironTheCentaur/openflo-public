# CPU CI-parity image — a local Linux mirror of the `ubuntu-latest` CI job.
# Build + run it to confirm the suite (incl. the golden baseline) passes on
# Linux exactly as on Windows, i.e. that results are synonymous across OSes.
#
#   docker build -f docker/Dockerfile -t openflo-ci .
#   docker run --rm openflo-ci                 # ruff + pyright + pytest (+golden)
#
# Mirrors .github/workflows/ci.yml: same pinned tools, Xvfb for the Tk tests,
# requirements.txt for the runtime surface, `pip install -e . --no-deps`.
FROM python:3.11-slim

ENV MPLBACKEND=Agg \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git: setuptools-scm / version; xvfb + xauth: virtual display for the Tk GUI
# tests (xvfb-run needs xauth); tk: the Tcl/Tk runtime libs tkinter links
# against (the slim image lacks them); build-essential: compile any sdist.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git xvfb xauth tk build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Deps first (cache layer), then the source.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "pyright[nodejs]==1.1.410" pytest pytest-xdist "ruff==0.15.19"
COPY . .
RUN pip install -e . --no-deps

# Default: the full CI gate (ruff -> pyright both platforms -> suite + golden).
CMD ["bash", "-lc", "ruff check . && pyright --pythonplatform Linux src && xvfb-run -a pytest -n auto -q && python -m openflo.diagnostics --quick"]
