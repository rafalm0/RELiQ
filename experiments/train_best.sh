#!/usr/bin/env bash
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
# Usage (from anywhere, e.g. inside your own sbatch job):
#     bash experiments/train_best.sh
#     PHASES="report" bash experiments/train_best.sh        # just redo the plot
#
# Sequential by default (JOBS=1): one training at a time, one GPU. A full run takes
# well over a day, so the 9 runs need more than one 7-day job. Just submit the same job
# again: finished runs are skipped (they have model_last.pt), so it continues where the
# previous job stopped. A run cut off by the time limit restarts from step 0 (no resume),
# which is why the script avoids starting a training it cannot finish (see below).
#
# Resources per process (measured, 100 nodes, README recipe):
#     training    ~10.2 GiB host RAM (100k replay buffer, allocated at start) + CUDA,
#                 budgeted as TRAIN_MEM_GB=13. JOBS is lowered automatically if
#                 JOBS x TRAIN_MEM_GB does not fit the job's memory (SLURM_MEM_PER_NODE).
#     evaluation  no replay buffer; small.
#
# Time limit: before each training the script compares the time left in the job
# (TIME_BUDGET_HOURS if set, else Slurm's remaining time via squeue) with the duration of
# already finished runs (or RUN_HOURS). If the next training would not finish and still
# leave EVAL_RESERVE_HOURS for evaluation, it stops training and evaluates what is done.
#
# Settings (environment variables):
#     RESULTS_DIR     output root                                  (results/best)
#     METHODS         methods to run                               ("reliq rt-reliq reliq-ratio")
#     SEEDS           training seeds per method                    ("0 1 2")
#     DEVICE          cuda or cpu                                  (cuda)
#     JOBS            trainings run at the same time (one GPU)     (1)
#     EVAL_JOBS       evaluations run at the same time             (JOBS)
#     TRAIN_MEM_GB    RAM budget per training process, GB          (13)
#     TIME_BUDGET_HOURS  wall time available to this invocation    (unset: ask Slurm)
#     RUN_HOURS       expected hours per training run              (unset: learned from finished runs)
#     EVAL_RESERVE_HOURS hours kept free for the evaluation phases (12)
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
JOBS="${JOBS:-1}"
EVAL_JOBS="${EVAL_JOBS:-${JOBS}}"
TRAIN_MEM_GB="${TRAIN_MEM_GB:-13}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-}"
RUN_HOURS="${RUN_HOURS:-}"
EVAL_RESERVE_HOURS="${EVAL_RESERVE_HOURS:-12}"
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

# --- memory guard ---------------------------------------------------------------------
# Memory available to this job in GB: Slurm's --mem if set, else MemAvailable.
mem_limit_gb() {
  if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then echo $(( SLURM_MEM_PER_NODE / 1024 )); return; fi
  awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo
}
MEM_GB="$(mem_limit_gb)"
max_jobs=$(( MEM_GB / TRAIN_MEM_GB ))
if (( max_jobs < 1 )); then
  echo "WARNING: ${MEM_GB} GB available but one training needs ~${TRAIN_MEM_GB} GB; it will likely be OOM-killed." >&2
  max_jobs=1
fi
if (( JOBS > max_jobs )); then
  echo "WARNING: JOBS=${JOBS} trainings need ~$(( JOBS * TRAIN_MEM_GB )) GB but only ${MEM_GB} GB are available; using JOBS=${max_jobs}." >&2
  JOBS="${max_jobs}"
fi

# Split the CPUs of the job between the concurrent processes. Without this every torch
# process starts one thread per visible core and they oversubscribe the allocation.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  ncpu="${SLURM_CPUS_PER_TASK:-$(nproc)}"
  export OMP_NUM_THREADS=$(( ncpu / JOBS > 0 ? ncpu / JOBS : 1 ))
fi

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

# --- time guard -----------------------------------------------------------------------
# Seconds left in the current Slurm job, from `squeue -o %L` ([d-]hh:mm:ss, mm:ss, ...).
slurm_seconds_left() {
  [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue > /dev/null || return 1
  local t d=0 h=0 m=0 sec=0 rest
  t="$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2> /dev/null | head -n 1)"
  [[ -z "$t" || "$t" == "UNLIMITED" || "$t" == "NOT_SET" || "$t" == "INVALID" ]] && return 1
  rest="$t"
  if [[ "$rest" == *-* ]]; then d="${rest%%-*}"; rest="${rest#*-}"; fi
  IFS=: read -r -a parts <<< "$rest"
  case ${#parts[@]} in
    3) h="${parts[0]}"; m="${parts[1]}"; sec="${parts[2]}" ;;
    2) m="${parts[0]}"; sec="${parts[1]}" ;;
    1) sec="${parts[0]}" ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$sec ))
}

DEADLINE=""
if [[ -n "${TIME_BUDGET_HOURS}" ]]; then
  DEADLINE=$(( $(date +%s) + TIME_BUDGET_HOURS * 3600 ))
elif left="$(slurm_seconds_left)"; then
  DEADLINE=$(( $(date +%s) + left ))
fi

# Expected seconds per training: RUN_HOURS, else the longest finished run, else unknown.
expected_run_seconds() {
  if [[ -n "${RUN_HOURS}" ]]; then echo $(( RUN_HOURS * 3600 )); return; fi
  local longest=0 f v
  for f in "${RESULTS_DIR}"/runs/*/seed*/train_seconds; do
    [[ -f "$f" ]] || continue
    v="$(cat "$f")"; (( v > longest )) && longest="$v"
  done
  (( longest > 0 )) && echo "$longest"
}

# 0 if a new training may start now, 1 (with a message) if it would not finish in time.
training_fits() {
  [[ -n "${DEADLINE}" ]] || return 0
  local need now
  need="$(expected_run_seconds)" || return 0
  [[ -n "$need" ]] || return 0
  now="$(date +%s)"
  if (( now + need * 11 / 10 + EVAL_RESERVE_HOURS * 3600 > DEADLINE )); then
    log "[train] not starting more trainings: one takes ~$(( need / 3600 ))h, $(( (DEADLINE - now) / 3600 ))h left in this job (keeping ${EVAL_RESERVE_HOURS}h for evaluation). Resubmit to continue."
    return 1
  fi
  return 0
}

# --- 1. train -------------------------------------------------------------------------
train_one() {
  local method="$1" seed="$2" out
  out="$(run_dir "$method" "$seed")"
  if [[ -f "${out}/model_last.pt" ]]; then log "[train] skip ${method} seed=${seed} (done)"; return 0; fi
  mkdir -p "${out}"
  log "[train] start ${method} seed=${seed}"
  local t0; t0="$(date +%s)"
  # shellcheck disable=SC2046
  if "${PYTHON}" -u src/main.py "${TRAIN_COMMON[@]}" --seed "${seed}" --output-dir "${out}" \
       $(method_flags "$method") ${TRAIN_EXTRA} > "${out}/train.log" 2>&1 \
     && [[ -f "${out}/model_last.pt" ]]; then
    echo $(( $(date +%s) - t0 )) > "${out}/train_seconds"
    log "[train] done  ${method} seed=${seed} ($(( ($(date +%s) - t0) / 3600 ))h)"
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
log "results: ${RESULTS_DIR}  methods: ${METHODS}  seeds: ${SEEDS}  device: ${DEVICE}"
log "trainings at once: ${JOBS}  evaluations at once: ${EVAL_JOBS}  threads/process: ${OMP_NUM_THREADS}  memory: ${MEM_GB} GB"
if [[ -n "${DEADLINE}" ]]; then
  log "time left in this job: $(( (DEADLINE - $(date +%s)) / 3600 ))h"
fi
done_runs=(); todo_runs=()
for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    if [[ -f "$(run_dir "$method" "$seed")/model_last.pt" ]]; then done_runs+=("${method}:${seed}"); else todo_runs+=("${method}:${seed}"); fi
  done
done
log "trained: ${#done_runs[@]} [${done_runs[*]:-}]  to train: ${#todo_runs[@]} [${todo_runs[*]:-}]"

if has_phase train; then
  # seed-major order: every method gets seed 0 before any method gets seed 1, so
  # partial results are balanced across methods
  stop_training=0
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      (( stop_training )) && break 2
      [[ -f "$(run_dir "$method" "$seed")/model_last.pt" ]] && continue
      throttle
      if ! training_fits; then stop_training=1; break 2; fi
      train_one "$method" "$seed" &
    done
  done
  wait
fi

if has_phase validate; then
  for method in ${METHODS}; do
    for seed in ${SEEDS}; do
      dir="$(run_dir "$method" "$seed")"
      for ckpt in model_best model_last; do
        JOBS="${EVAL_JOBS}" throttle
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
    JOBS="${EVAL_JOBS}" throttle
    eval_one "${ckpt}" "${dir}/test" 0 "${TEST_EPISODES}" --detailed-eval-logs &
  done < <("${PYTHON}" experiments/best_models.py list-selected "${RESULTS_DIR}")
  wait
fi

if has_phase report; then
  "${PYTHON}" experiments/best_models.py report "${RESULTS_DIR}" || exit 1
fi

log "finished. Best models: ${RESULTS_DIR}/best_models/  report: ${RESULTS_DIR}/report/"
