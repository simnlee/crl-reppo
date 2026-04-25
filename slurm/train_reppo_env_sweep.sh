#!/bin/bash
#SBATCH --constraint="a40|l40|l40s|a6000|b6000"
#SBATCH --job-name=reppo-gcrl-env-sweep
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-34

cd /home/simlee/reppo/crl-reppo
source .venv/bin/activate

# Fix SSL certs (system bundle missing on this cluster)
export SSL_CERT_FILE=./cacert.pem

# =============================================================================
# Env sweep: 7 envs x 5 seeds = 35 tasks, fixed HER variant = crit_act_tdL, k=1
# TASK_ID layout:
#   env_idx  = TASK_ID / 5
#   seed_idx = TASK_ID % 5
# Submit the full sweep with (datetime is shared across array tasks):
#   DATETIME=$(date +%Y%m%d_%H%M) sbatch --export=ALL,DATETIME slurm/train_reppo_env_sweep.sh
# Override W&B env vars via:
#   sbatch --export=ALL,WANDB_PROJECT=foo,... slurm/train_reppo_env_sweep.sh
# =============================================================================

ENVS=(ant_big_maze ant_u4_maze ant_u5_maze ant_hardest_maze \
      arm_push_hard arm_binpick_hard arm_push_easy)

SEEDS=(0 1 2 3 4)

NUM_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Fixed HER variant: crit_act_tdL
HER_CRITIC=true
HER_ACTOR=true
HER_TD_LAMBDA=true
HER_PER_EPOCH=true

# Non-variation flags (mirror train_reppo.sh so the two sweeps are comparable)
HER_K=1
DEPTH=4
STAGGER_ENVS=false
STAGGER_DEBUG=false
NORMALIZE_HINDSIGHT_LOSS=false
SAMPLE_NEW_ACTION_FOR_TDL=true

# Shared across all array tasks — set at submission time, see header comment
# DATETIME="${DATETIME:-$(date +%Y%m%d_%H%M)}"

# W&B group-name scheme mirrors train_reppo.sh, with an env_sweep suffix
GROUP="k${HER_K}"
[[ "$STAGGER_ENVS" == "true" ]]              && GROUP+="_stagger_envs"
[[ "$NORMALIZE_HINDSIGHT_LOSS" == "true" ]]   && GROUP+="_normalize_hindsight_loss"
[[ "$SAMPLE_NEW_ACTION_FOR_TDL" == "true" ]]  && GROUP+="_sample_new_action_for_tdL"
GROUP+="_v7_r1000_herresample_tempsigns"

# W&B logging (overridable via --export)
WANDB_PROJECT="${WANDB_PROJECT:-crl-reppo}"
WANDB_ENTITY="${WANDB_ENTITY:-simlee-upenn}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${GROUP}}"
WANDB_DIR="${WANDB_DIR:-.}"

mkdir -p "${WANDB_DIR}"

# Translate true/false into tyro's --flag / --no-flag pair
bool_flag() {
	if [ "$2" = "true" ]; then echo "--$1"; else echo "--no-$1"; fi
}

echo "[env-sweep] task=${TASK_ID} env=${ENV_ID} seed=${SEED}"
echo "[env-sweep] HER_CRITIC=${HER_CRITIC} HER_ACTOR=${HER_ACTOR} HER_TD_LAMBDA=${HER_TD_LAMBDA} HER_PER_EPOCH=${HER_PER_EPOCH}"

uv run train_reppo.py \
	--env-id      "${ENV_ID}" \
	--eval-env-id "${ENV_ID}" \
	--seed        "${SEED}" \
	$(bool_flag use-her-critic            "${HER_CRITIC}") \
	$(bool_flag use-her-actor             "${HER_ACTOR}") \
	$(bool_flag use-her-td-lambda         "${HER_TD_LAMBDA}") \
	$(bool_flag her-per-epoch             "${HER_PER_EPOCH}") \
	--her-k "${HER_K}" \
	--actor-depth  "${DEPTH}" \
	--critic-depth "${DEPTH}" \
	$(bool_flag stagger-envs              "${STAGGER_ENVS}") \
	$(bool_flag stagger-debug             "${STAGGER_DEBUG}") \
	$(bool_flag normalize-hindsight-loss  "${NORMALIZE_HINDSIGHT_LOSS}") \
	$(bool_flag sample-new-action-for-tdL "${SAMPLE_NEW_ACTION_FOR_TDL}") \
	--wandb-project-name "${WANDB_PROJECT}" \
	--wandb-entity       "${WANDB_ENTITY}" \
	--wandb-mode         "${WANDB_MODE}" \
	--wandb-group        "${WANDB_GROUP}" \
	--wandb-dir          "${WANDB_DIR}"
