#!/bin/bash
#SBATCH --constraint="a40|l40|l40s|a6000|b6000|3090"
#SBATCH --job-name=reppo-jaxgcrl
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=60G
#SBATCH --time=0-3:00:00
#SBATCH --requeue  

cd /home/simlee/reppo/crl-reppo
source .venv/bin/activate

# Fix SSL certs (system bundle missing on this cluster)
export SSL_CERT_FILE=./cacert.pem

# Weights & Biases logging spec
WANDB_PROJECT="scaling-crl-baseline"
WANDB_ENTITY="simlee-upenn"
WANDB_MODE="online"    # use "offline" on restricted clusters
WANDB_GROUP="crl-baseline"
WANDB_DIR="."

mkdir -p "${WANDB_DIR}"

uv run train.py \
	--env_id "ant" \
	--eval_env_id "ant" \
	--wandb-project-name "${WANDB_PROJECT}" \
	--wandb-entity "${WANDB_ENTITY}" \
	--wandb-mode "${WANDB_MODE}" \
	--wandb-group "${WANDB_GROUP}" \
	--wandb-dir "${WANDB_DIR}" \
	--num_epochs 1 \
	--total_env_steps 5000000 \
	--critic_depth 16 \
	--actor_depth 16 \
	--actor_skip_connections 4 \
	--critic_skip_connections 4 \
	--batch_size 512 \
	--vis_length 1000 \
	--save_buffer 0




