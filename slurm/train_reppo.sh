#!/bin/bash
#SBATCH --constraint="a40|l40|l40s|a6000|b6000"
#SBATCH --job-name=reppo-gcrl-sweep
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=60G
#SBATCH --time=0-8:00:00
#SBATCH --requeue
#SBATCH --array=0-139

cd /home/simlee/reppo/crl-reppo
source .venv/bin/activate

# Fix SSL certs (system bundle missing on this cluster)
export SSL_CERT_FILE=./cacert.pem

# =============================================================================
# Sweep grid: 7 envs x 4 HER variations x 5 seeds = 140 tasks
# TASK_ID layout:
#   env_idx  = TASK_ID / 20
#   var_idx  = (TASK_ID % 20) / 5
#   seed_idx = TASK_ID % 5
# Submit the full sweep with:  sbatch slurm/train_reppo.sh
# Override W&B env vars via:   sbatch --export=WANDB_PROJECT=foo,... slurm/train_reppo.sh
# =============================================================================

ENVS=(ant_big_maze ant_u4_maze ant_u5_maze ant_hardest_maze \
      arm_push_easy arm_push_hard arm_binpick_hard)

SEEDS=(0 1 2 3 4)

# NAME:HER_CRITIC:HER_ACTOR:HER_TD_LAMBDA
VARIATIONS=(
	"crit_td0:true:false:false"
	"crit_act_td0:true:true:false"
	"crit_tdL:true:false:true"
	"crit_act_tdL:true:true:true"
)

NUM_SEEDS=${#SEEDS[@]}
NUM_VARS=${#VARIATIONS[@]}
STRIDE_ENV=$((NUM_VARS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_IDX=$((TASK_ID / STRIDE_ENV))
REM=$((TASK_ID % STRIDE_ENV))
VAR_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
IFS=':' read -r VAR_NAME HER_CRITIC HER_ACTOR HER_TD_LAMBDA \
	<<< "${VARIATIONS[$VAR_IDX]}"

# Non-variation flags (drive both the tyro invocation and the W&B group name)
HER_K=1
STAGGER_ENVS=true
STAGGER_DEBUG=true
NORMALIZE_HINDSIGHT_LOSS=false
SAMPLE_NEW_ACTION_FOR_TDL=true

# W&B group-name scheme mirrors reppo-gcrl/slurm/train_reppo_her_sweep.bash
GROUP="baseline_k${HER_K}"
[[ "$STAGGER_ENVS" == "true" ]]              && GROUP+="_stagger_envs"
[[ "$NORMALIZE_HINDSIGHT_LOSS" == "true" ]]   && GROUP+="_normalize_hindsight_loss"
[[ "$SAMPLE_NEW_ACTION_FOR_TDL" == "true" ]]  && GROUP+="_sample_new_action_for_tdL"

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

echo "[sweep] task=${TASK_ID} env=${ENV_ID} var=${VAR_NAME} seed=${SEED}"
echo "[sweep] HER_CRITIC=${HER_CRITIC} HER_ACTOR=${HER_ACTOR} HER_TD_LAMBDA=${HER_TD_LAMBDA}"

uv run train_reppo.py \
	--env-id      "${ENV_ID}" \
	--eval-env-id "${ENV_ID}" \
	--seed        "${SEED}" \
	$(bool_flag use-her-critic            "${HER_CRITIC}") \
	$(bool_flag use-her-actor             "${HER_ACTOR}") \
	$(bool_flag use-her-td-lambda         "${HER_TD_LAMBDA}") \
	--her-k "${HER_K}" \
	$(bool_flag stagger-envs              "${STAGGER_ENVS}") \
	$(bool_flag stagger-debug             "${STAGGER_DEBUG}") \
	$(bool_flag normalize-hindsight-loss  "${NORMALIZE_HINDSIGHT_LOSS}") \
	$(bool_flag sample-new-action-for-tdL "${SAMPLE_NEW_ACTION_FOR_TDL}") \
	--wandb-project-name "${WANDB_PROJECT}" \
	--wandb-entity       "${WANDB_ENTITY}" \
	--wandb-mode         "${WANDB_MODE}" \
	--wandb-group        "${WANDB_GROUP}" \
	--wandb-dir          "${WANDB_DIR}"
