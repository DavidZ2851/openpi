#!/bin/bash
# MolmoBot-Data depth mixture (local/mbdepth_mix_rope4d): LoRA, full fine-tune, full fine-tune + 4D RoPE,
# batch 32 x 30k steps each (config.py _mbdepth). One job per model:
#
#   sbatch --gres=gpu:1 scripts/grogu/mbdepth_train.sbatch lora
#   sbatch --gres=gpu:1 scripts/grogu/mbdepth_train.sbatch full
#   sbatch --gres=gpu:1 scripts/grogu/mbdepth_train.sbatch rope4d
#
# lora / full train with JAX (scripts/train.py, FSDP over the job's GPUs); rope4d trains with PyTorch
# (scripts/train_pytorch.py, DDP over the job's GPUs; the global batch 32 is split across them). One >=80 GB GPU
# fits each run; more GPUs only add speed (full FT on 48 GB cards needs 2 for FSDP).
# Paths below default to the grogu layout; override with the environment (OPENPI, VENV, HF_LEROBOT_HOME, ...).
#SBATCH --job-name=mbdepth
#SBATCH --partition=dheld
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
MODE=${1:?lora | full | rope4d}
OPENPI=${OPENPI:-$HOME/openpi}
VENV=${VENV:-$OPENPI/.venv}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/scratch/haotian/lerobot}
export PI05_JOINTPOS_DIR=${PI05_JOINTPOS_DIR:-$OPENPI/checkpoints/pi05_droid_jointpos}
export PI05_JOINTPOS_PYTORCH_DIR=${PI05_JOINTPOS_PYTORCH_DIR:-${PI05_JOINTPOS_DIR}_pytorch}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
NGPU=$(nvidia-smi -L | wc -l)
cd "$OPENPI"
echo "[$(date)] $MODE on $(hostname), $NGPU GPU(s): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
case $MODE in
  lora | full)
    exec "$VENV"/bin/python scripts/train.py pi05_droid_jointpos_mbdepth_$MODE --exp-name mbdepth_$MODE \
      --fsdp-devices "$NGPU" --resume ;;
  rope4d)
    # train_pytorch.py --resume fails without an existing run, unlike train.py's.
    RESUME=(); [ -d checkpoints/pi05_droid_jointpos_mbdepth_rope4d_full/mbdepth_rope4d_full ] && RESUME=(--resume)
    exec "$VENV"/bin/torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" scripts/train_pytorch.py \
      pi05_droid_jointpos_mbdepth_rope4d_full --exp_name mbdepth_rope4d_full "${RESUME[@]}" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
