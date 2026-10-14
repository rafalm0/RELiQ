#!/encs/bin/tcsh
#SBATCH -J "RELiq train best"   # job name
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=4  # number of CPU cores per task
#SBATCH --gpus=1   # gpu devices per node
#SBATCH --partition=pg   # partition
#SBATCH --mem=30G   # memory
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00   # job time limit
#SBATCH --output=lastlog.out

module load python/3.12.0/default
mkdir -p /speed-scratch/$USER/tmp
setenv TMP /speed-scratch/$USER/tmp
setenv TMPDIR /speed-scratch/$USER/tmp
python -m venv $TMPDIR/reliqvenv
module load cuda/12.8/default
setenv PIP_CACHE_DIR /speed-scratch/$USER/tmp/reliqvenv/cache
source /speed-scratch/$USER/QRL/RELiQ/Reliqvenv/bin/activate.csh
pip install -r /speed-scratch/r_albuq/QRL/replica/RELiQ/requirements.txt

echo "Starting Script"


# Train, select and evaluate the best model of each method, then plot them.
#
#   reliq        RELiQ as published: original reward, automatic purification  (main baseline)
#   rt-reliq     RT-RELiQ: agent picks purification rounds per hop, ratio reward
#   reliq-ratio  RELiQ trained on the ratio reward (ablation: separates the effect of
#                the reward from the effect of letting the agent purify)
#
# Phases (each one skips work that is already done, so the script can be re-run):
#   1. train     every method x seed, with the training recipe from the RELiQ README
#   2. validate  model_best.pt and model_last.pt of every run on held-out validation
#                topologies (EVAL_SEEDS[VAL_OFFSET:], never used for training or testing)
#   3. select    per run the better checkpoint; per method the best run -> best_models/
#   4. test      the selected checkpoint of every run on the standard test topologies
#                (EVAL_SEEDS[0:TEST_EPISODES]) with the evaluation flags from the README
#   5. report    figure + table: best model of each method, with every seed shown
#
# Usage (from anywhere):
#     bash experiments/train_best.sh
#     PHASES="report" bash experiments/train_best.sh        # just redo the plot
#
# Settings (environment variables):
#     RESULTS_DIR     output root                                  (results/best)
#     METHODS         methods to run                               ("reliq rt-reliq reliq-ratio")
#     SEEDS           training seeds per method                    ("0 1 2")
#     DEVICE          cuda or cpu                                  (cuda)
#     JOBS            processes run concurrently                   (3)
#     TOTAL_STEPS     training steps per run                       (15000000, as in the README)
#     RT_ROUNDS       max purification rounds for RT-RELiQ         (3)
#     VAL_OFFSET      first validation topology index              (500)
#     VAL_EPISODES    validation episodes per checkpoint           (50)
#     TEST_EPISODES   test episodes                                (100)
#     SELECT_METRIC   validation metric used to pick models        (average_episode_packets_mean)
#     SELECT_MINIMIZE 1 if lower SELECT_METRIC is better            (0)
#     TRAIN_EXTRA     extra flags appended to every training run (override defaults)
#     EVAL_EXTRA      extra flags appended to every evaluation run
#     PHASES          subset of "train validate select test report"
#     PYTHON          interpreter                                  (python)
#
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:src"

RESULTS_DIR="${RESULTS_DIR:-results/best}"
METHODS="${METHODS:-reliq rt-reliq reliq-ratio}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda}"
JOBS="${JOBS:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-15000000}"
RT_ROUNDS="${RT_ROUNDS:-3}"
VAL_OFFSET="${VAL_OFFSET:-500}"
VAL_EPISODES="${VAL_EPISODES:-50}"
TEST_EPISODES="${TEST_EPISODES:-100}"
SELECT_METRIC="${SELECT_METRIC:-average_episode_packets_mean}"
SELECT_MINIMIZE="${SELECT_MINIMIZE:-0}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
EVAL_EXTRA="${EVAL_EXTRA:-}"
PHASES="${PHASES:-train validate select test report}"
PYTHON="${PYTHON:-python}"

if (( VAL_OFFSET < TEST_EPISODES )); then
  echo "VAL_OFFSET (${VAL_OFFSET}) must be >= TEST_EPISODES (${TEST_EPISODES}) so validation and test topologies do not overlap" >&2
  exit 1
fi

# Training recipe from the RELiQ README (identical for every method).
TRAIN_COMMON=(
  --use-future-rewards
  --netmon-agg-type=sage
  --no-idle-action
  --request-based-observation
  --fixed-requests
  --action-mask
  --disable-progressbar
  --total-steps="${TOTAL_STEPS}"
  --step-between-train=200
  --step-before-train=100000
  --netmon
  --model=dqn
  --device="${DEVICE}"
  --capacity=100000
  --min-path-length=1
  # short sanity evaluation at the end of training; selection and test use phases 2-4
  --eval-episodes=10
)

# Evaluation flags from the RELiQ README (model architecture and method are restored
# from the checkpoint).
EVAL_COMMON=(
  --request-based-observation
  --no-idle-action
  --use-realistic-decay
  --action-mask
  --fixed-requests
  --eval
  --disable-progressbar
  --device="${DEVICE}"
)

method_flags() {
  case "$1" in
    reliq)       echo "" ;;
    rt-reliq)    echo "--method rt-reliq --max-purification-rounds ${RT_ROUNDS} --reward-type ratio" ;;
    reliq-ratio) echo "--reward-type ratio" ;;
    *) echo "unknown method $1" >&2; exit 1 ;;
  esac
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    wait -n 2>/dev/null || sleep 5
  done
}

run_dir() { echo "${RESULTS_DIR}/runs/$1/seed$2"; }

# --- 1. train -------------------------------------------------------------------------
train_one() {
  local method="$1" seed="$2" out
  out="$(run_dir "$method" "$seed")"
  if [[ -f "${out}/model_last.pt" ]]; then log "[train] skip ${method} seed=${seed} (done)"; return 0; fi
  mkdir -p "${out}"
  log "[train] start ${method} seed=${seed}"
  # shellcheck disable=SC2046
  if "${PYTHON}" -u src/main.py "${TRAIN_COMMON[@]}" --seed "${seed}" --output-dir "${out}" \
       $(method_flags "$method") ${TRAIN_EXTRA} > "${out}/train.log" 2>&1 \
     && [[ -f "${out}/model_last.pt" ]]; then
    log "[train] done  ${method} seed=${seed}"
  else
    log "[train] FAIL  ${method} seed=${seed} -> ${out}/train.log"
  fi
}

# --- 2./4. evaluate one checkpoint -----------------------------------------------------
eval_one() {
  local ckpt="$1" out="$2" offset="$3" episodes="$4"; shift 4
  if [[ -f "${out}/metrics.json" ]]; then return 0; fi
  if [[ ! -f "${ckpt}" ]]; then log "[eval ] missing ${ckpt}"; return 0; fi
  mkdir -p "${out}"
  if "${PYTHON}" -u src/main.py "${EVAL_COMMON[@]}" --model-load-path="${ckpt}" \
       --eval-output-dir="${out}" --eval-seed-offset="${offset}" --eval-episodes="${episodes}" \
       "$@" ${EVAL_EXTRA} > "${out}/eval.log" 2>&1 && [[ -f "${out}/metrics.json" ]]; then
    log "[eval ] done  ${out}"
  else
    log "[eval ] FAIL  ${out} -> ${out}/eval.log"
  fi
}

has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

mkdir -p "${RESULTS_DIR}"
log "results: ${RESULTS_DIR}  methods: ${METHODS}  seeds: ${SEEDS}  device: ${DEVICE}  jobs: ${JOBS}"

if has_phase train; then
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      throttle; train_one "$method" "$seed" &
    done
  done
  wait
fi

if has_phase validate; then
  for method in ${METHODS}; do
    for seed in ${SEEDS}; do
      dir="$(run_dir "$method" "$seed")"
      for ckpt in model_best model_last; do
        throttle
        eval_one "${dir}/${ckpt}.pt" "${dir}/val_${ckpt}" "${VAL_OFFSET}" "${VAL_EPISODES}" &
      done
    done
  done
  wait
fi

minimize_flag=""
[[ "${SELECT_MINIMIZE}" == "1" ]] && minimize_flag="--minimize"

if has_phase select; then
  "${PYTHON}" experiments/best_models.py select "${RESULTS_DIR}" \
      --metric "${SELECT_METRIC}" ${minimize_flag} || exit 1
fi

if has_phase test; then
  selection="${RESULTS_DIR}/selection.json"
  if [[ ! -f "${selection}" ]]; then echo "no ${selection}; run the select phase first" >&2; exit 1; fi
  # one line per run: <run dir> <selected checkpoint path>
  while read -r dir ckpt; do
    throttle
    eval_one "${ckpt}" "${dir}/test" 0 "${TEST_EPISODES}" --detailed-eval-logs &
  done < <("${PYTHON}" experiments/best_models.py list-selected "${RESULTS_DIR}")
  wait
fi

if has_phase report; then
  "${PYTHON}" experiments/best_models.py report "${RESULTS_DIR}" || exit 1
fi

log "finished. Best models: ${RESULTS_DIR}/best_models/  report: ${RESULTS_DIR}/report/"
