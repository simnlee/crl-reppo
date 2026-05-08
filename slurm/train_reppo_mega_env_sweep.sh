#!/bin/bash
#SBATCH --constraint="a40|l40|l40s|a6000|b6000"
#SBATCH --job-name=reppo-mega-env-sweep
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=60G
#SBATCH --time=0-10:00:00
#SBATCH --requeue
#SBATCH --array=0-20

# cd /path/to/reppo
# source .venv/bin/activate

# Fix SSL certs (system bundle missing on this cluster)
# export SSL_CERT_FILE=./cacert.pem

# =============================================================================
# OMEGA env sweep: 7 envs x 3 seeds = 21 tasks
# Runs train_reppo_mega.py (REPPO + OMEGA goal sampling, v7).
# =============================================================================

ENVS=(ant_big_maze ant_hardest_maze arm_push_hard ant_u5_maze ant_u4_maze humanoid arm_push_easy)

SEEDS=(0 1 2)

NUM_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# W&B group-name
GROUP="0505_omega_v1"

# W&B logging (overridable via --export)
WANDB_PROJECT="${WANDB_PROJECT:-crl-reppo}"
WANDB_ENTITY="${WANDB_ENTITY:-simlee-upenn}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${GROUP}}"
WANDB_DIR="${WANDB_DIR:-.}"

mkdir -p "${WANDB_DIR}"

echo "[mega-env-sweep] task=${TASK_ID} env=${ENV_ID} seed=${SEED}"

uv run train_reppo_mega.py \
	--env-id      "${ENV_ID}" \
	--eval-env-id "${ENV_ID}" \
	--seed        "${SEED}" \
	--use-omega \
	--wandb-project-name "${WANDB_PROJECT}" \
	--wandb-entity       "${WANDB_ENTITY}" \
	--wandb-mode         "${WANDB_MODE}" \
	--wandb-group        "${WANDB_GROUP}" \
	--wandb-dir          "${WANDB_DIR}"
