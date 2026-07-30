#!/bin/bash
#SBATCH -J czech_embed_eval
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=gpu_cc8.6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/tomchikr/czech_embedding_benchmark/jobs/logs/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=roman.tomchik953@student.cuni.cz

# Runs the retrieval evaluation for a single (small) model on the UFAL AIC cluster.
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/per_model/intfloat_multilingual-e5-small.toml jobs/scripts/run_aic.sh
#
# Notes:
# - Default constraint requests a 24G VRAM GPU (`gpuram24G`) to avoid
#   older incompatible cards and reduce OOM risk for larger models.
# - You can override at submit time, e.g.:
#   sbatch --constraint=gpu_cc8.6 --export=ALL,CONFIG=... jobs/scripts/run_aic.sh
#
# CONFIG may be absolute or relative to the repository root. Results are
# written by evaluation/evaluate.py to results/mteb/... exactly as usual.

set -euo pipefail

nvidia-smi

REPO="/home/tomchikr/czech_embedding_benchmark"
LOG_DIR="$REPO/jobs/logs"

# Mirror all output to a shared logfile so progress is visible while the job runs.
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Validate input -----------------------------------------------------
if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG not set. Submit with: sbatch --export=ALL,CONFIG=<path/to/config.toml> $0" >&2
    exit 1
fi

CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" != /* ]] && CONFIG_PATH="$REPO/$CONFIG_PATH"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

# --- Repo-local cache/temp (uses project storage quota) -------------------
CACHE_ROOT="$REPO/.cache"
export TMPDIR="$CACHE_ROOT/tmp"
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# --- Environment ---------------------------------------------------------
# VENV may override the default virtualenv (e.g. VENV=venvs/.venv-nemotron for
# the nvidia/llama-embed-nemotron-8b model, which needs transformers==4.51.0 +
# flash-attn and cannot share the main .venv). Relative to $REPO. All venvs
# live under venvs/ (for `uv sync`/`uv add` on the main env, export
# UV_PROJECT_ENVIRONMENT=venvs/.venv first - uv has no config-file setting for this).
cd "$REPO"
VENV="${VENV:-venvs/.venv}"
source "$VENV/bin/activate"

echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Config:   $CONFIG_PATH"
echo "Started:  $(date)"

# --- Run evaluation (writes results to results/ in the repo) -------------
python evaluation/evaluate.py --config "$CONFIG_PATH"

echo "Finished: $(date)"
