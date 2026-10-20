#!/usr/bin/env bash
# ============================================================
#  OpenFlo one-time setup (Linux / macOS).
#  Creates a local .venv and installs OpenFlo + every
#  dependency, so ./openflo-gui.sh can launch. Safe to re-run
#  (idempotent: re-uses the venv, re-installs/updates deps).
#
#  No Python? No problem: when no Python 3.11+ is found, setup
#  uses uv (https://github.com/astral-sh/uv), which downloads
#  its own Python. A uv already on PATH is used as is;
#  otherwise a pinned uv release is downloaded into .tools/uv
#  and its SHA-256 is checked against the hash below before it
#  is unpacked. A mismatch stops setup.
#
#  Run:  ./setup.sh        (chmod +x once if needed)
# ============================================================
set -euo pipefail
# Let failures inside $(...) stop setup too (bash 4.4+; harmless elsewhere).
shopt -s inherit_errexit 2>/dev/null || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { echo "[OpenFlo setup] $*"; }
die() { echo "[OpenFlo setup] $*" >&2; exit 1; }

# ---- Pinned uv release (used only when no Python 3.11+ exists) ----------
# Hashes copied from the release's <asset>.sha256 files. setup.bat pins the
# same version and the same Windows hashes; tests check the two agree.
UV_VERSION="0.12.18"
uv_sha256_for() {
    case "$1" in
        uv-x86_64-apple-darwin.tar.gz)      echo "2e4108f5395397c8bc5d43bf83d3bdbb2d0e92b90d0efa607756be704905fa33" ;;
        uv-aarch64-apple-darwin.tar.gz)     echo "cf40e0c6a202190ccd9e0406dcfdd5b2d6668a9a5c779b17948963df32aafe5b" ;;
        uv-x86_64-unknown-linux-gnu.tar.gz) echo "89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16" ;;
        uv-aarch64-unknown-linux-gnu.tar.gz) echo "afb6291f3f0a6b4521fc67b947822506c41dde5b60d2189dd8f3695b2ac8c9e7" ;;
        uv-x86_64-pc-windows-msvc.zip)      echo "cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff" ;;
        uv-aarch64-pc-windows-msvc.zip)     echo "17f27b1c64eacc757ae603579f116a014881e486c5e79ae81877980d4699e943" ;;
        *) return 1 ;;
    esac
}
# Where release assets are fetched from. Overridable (mirrors, tests); the
# hash above is checked whatever the source, so this cannot weaken it.
UV_BASE_URL="${OPENFLO_UV_BASE_URL:-https://github.com/astral-sh/uv/releases/download}"
UV_PYTHON_VERSION="3.12"
TOOLS_DIR="$ROOT/.tools"

# ---- Helpers -------------------------------------------------------------
python_ok() {
    # True when $1 runs and is Python 3.11 or newer.
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        return 1
    fi
}

download() {  # download URL DEST
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 2 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        die "Neither curl nor wget is available to download uv. Install one, or install Python 3.11+, and re-run."
    fi
}

uv_target() {
    local os arch
    case "$(uname -s)" in
        Darwin) os="apple-darwin" ;;
        Linux)  os="unknown-linux-gnu" ;;
        *) return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64)  arch="x86_64" ;;
        arm64|aarch64) arch="aarch64" ;;
        *) return 1 ;;
    esac
    echo "$arch-$os"
}

fetch_uv() {
    # Download the pinned uv into .tools/uv, verify it, print the binary path.
    local target asset expected tmp actual
    target="$(uv_target)" || die "No pinned uv build for $(uname -s) $(uname -m). Install Python 3.11+ and re-run."
    asset="uv-$target.tar.gz"
    expected="$(uv_sha256_for "$asset")" || die "No pinned hash for $asset."
    if [ -x "$TOOLS_DIR/uv/uv" ] && [ -f "$TOOLS_DIR/uv/VERSION" ] \
        && [ "$(cat "$TOOLS_DIR/uv/VERSION")" = "$UV_VERSION" ]; then
        echo "$TOOLS_DIR/uv/uv"
        return 0
    fi
    say "Downloading uv $UV_VERSION ($asset) ..." >&2
    mkdir -p "$TOOLS_DIR"
    tmp="$(mktemp -d "$TOOLS_DIR/dl.XXXXXX")"
    download "$UV_BASE_URL/$UV_VERSION/$asset" "$tmp/$asset" \
        || { rm -rf "$tmp"; die "Download of $asset failed."; }
    actual="$(sha256_of "$tmp/$asset")" \
        || { rm -rf "$tmp"; die "No sha256sum or shasum available to verify the download."; }
    if [ "$actual" != "$expected" ]; then
        rm -rf "$tmp"
        die "SHA-256 mismatch for $asset (expected $expected, got $actual). Refusing to continue."
    fi
    say "Checksum OK." >&2
    tar -xzf "$tmp/$asset" -C "$tmp" || { rm -rf "$tmp"; die "Could not unpack $asset."; }
    rm -rf "$TOOLS_DIR/uv"
    mkdir -p "$TOOLS_DIR/uv"
    mv "$tmp/uv-$target/uv" "$TOOLS_DIR/uv/uv"
    [ -f "$tmp/uv-$target/uvx" ] && mv "$tmp/uv-$target/uvx" "$TOOLS_DIR/uv/uvx"
    chmod +x "$TOOLS_DIR/uv/uv"
    echo "$UV_VERSION" >"$TOOLS_DIR/uv/VERSION"
    rm -rf "$tmp"
    echo "$TOOLS_DIR/uv/uv"
}

# ---- Locate a Python 3.11+, else fall back to uv ---------------------------
VPY="$ROOT/.venv/bin/python"
if [ ! -x "$VPY" ]; then
    PY=""
    for cand in python3 python; do
        p="$(command -v "$cand" || true)"
        if [ -n "$p" ]; then
            if python_ok "$p"; then
                PY="$p"
                break
            fi
            say "Ignoring $p ($("$p" --version 2>&1 || echo 'unknown version')): OpenFlo needs Python 3.11+."
        fi
    done

    if [ -n "$PY" ]; then
        say "Using Python: $PY ($("$PY" --version 2>&1))"
        say "Creating virtual environment in .venv ..."
        "$PY" -m venv "$ROOT/.venv"
    else
        say "No Python 3.11+ found; using uv to provide Python $UV_PYTHON_VERSION."
        UV="$(command -v uv || true)"
        if [ -n "$UV" ]; then
            say "Using uv on PATH: $UV"
        else
            UV="$(fetch_uv)"
        fi
        "$UV" python install "$UV_PYTHON_VERSION" || die "uv could not install Python $UV_PYTHON_VERSION."
        # --seed puts pip in the venv so the install below is the same as ever.
        "$UV" venv --seed --python "$UV_PYTHON_VERSION" "$ROOT/.venv" \
            || die "uv could not create the virtual environment."
    fi
fi

[ -x "$VPY" ] || die "The virtual environment has no Python at .venv/bin/python."
say "Upgrading pip ..."
"$VPY" -m pip install --upgrade pip
say "Installing OpenFlo + dependencies (this can take a few minutes) ..."
"$VPY" -m pip install -e ".[gui]"

echo
say "Done. Launch OpenFlo with:  ./openflo-gui.sh"
