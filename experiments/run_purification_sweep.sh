#!/usr/bin/env bash
#
# Purification-control sweep.
#
# Arms:
#
#   reliq             RELiQ as published - default flags only, nothing added. Main baseline.
#   reliq_nopurif     RELiQ with automatic purification off (isolates what it buys)
#   rt-reliq_P*_c*    RT-RELiQ: the agent picks the purification rounds per hop jointly
#                     with the next hop (P = max rounds, c = purification cost weight)
#
# Run from the repository root:
#     bash experiments/run_purification_sweep.sh
#
# Knobs (environment variables):
#     RESULTS_DIR   where runs are written            (default: results/purification)
#     SEEDS         seeds to repeat each arm with     (default: "0 1 2")
#     ROUNDS        values of --max-purification-rounds for RT-RELiQ (default: "1 2 3")
#     COSTS         values of --purification-cost-weight for RT-RELiQ (default: "0 0.005 0.02")
#     TOTAL_STEPS   training steps per run            (default: 1000000)
#     JOBS          runs executed concurrently        (default: 4)
#     DRY_RUN       set to 1 to print commands only
#     EXTRA_ARGS    appended to every run; later flags win, so this overrides COMMON
#                   (e.g. EXTRA_ARGS="--n-router 20 --episode-steps 50" for a dry run)
#
set -uo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/purification}"
SEEDS="${SEEDS:-0 1 2}"
ROUNDS="${ROUNDS:-1 2 3}"
COSTS="${COSTS:-0 0.005 0.02}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
JOBS="${JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"

# Shared environment/training configuration. Every arm must use exactly these values,
# otherwise the comparison is confounded. Edit here, not per arm.
COMMON=(
  --policy trained
  --env-type quantum
  --total-steps "${TOTAL_STEPS}"
  --n-router 100
  --neighbors 5
  --n-quantum 20
  --episode-steps 500
  --ttl 20
  --action-mask
  --netmon
  --eval-episodes 100
  --eval-episode-steps 1000
  --disable-progressbar
)

# The fidelity table is loaded relative to the working directory, so always run from
# the repository root with src on the path.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p "${RESULTS_DIR}"

run_one() {
  local name="$1"; shift
  local seed="$1"; shift
  local out="${RESULTS_DIR}/${name}/seed${seed}"

  if [[ -f "${out}/eval/metrics.json" ]]; then
    echo "[skip] ${name} seed=${seed} (already finished)"
    return 0
  fi

  mkdir -p "${out}"
  # EXTRA_ARGS last so it overrides COMMON (argparse keeps the final occurrence)
  local cmd=(python src/main.py "${COMMON[@]}" --seed "${seed}" --output-dir "${out}" \
             "$@" ${EXTRA_ARGS:-})

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\n' "${cmd[*]}"
    return 0
  fi

  echo "[start] ${name} seed=${seed}"
  if "${cmd[@]}" > "${out}/run.log" 2>&1; then
    echo "[done ] ${name} seed=${seed}"
  else
    echo "[FAIL ] ${name} seed=${seed} -> ${out}/run.log"
  fi
}

# throttle to JOBS concurrent runs
throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    wait -n 2>/dev/null || sleep 5
  done
}

for seed in ${SEEDS}; do
  # --- baseline: RELiQ exactly as published (no added flags) --------------------
  throttle
  run_one "reliq" "${seed}" &

  # --- ablation: RELiQ without automatic purification ----------------------------
  throttle
  run_one "reliq_nopurif" "${seed}" \
      --auto-distillation-threshold 0 &

  # --- RT-RELiQ: agent decides rounds jointly with routing -----------------------
  for p in ${ROUNDS}; do
    for c in ${COSTS}; do
      throttle
      run_one "rt-reliq_P${p}_c${c}" "${seed}" \
          --method rt-reliq \
          --max-purification-rounds "${p}" \
          --purification-cost-weight "${c}" &
    done
  done
done

wait
echo
echo "Sweep finished. Collect results with:"
echo "    python experiments/collect_metrics.py ${RESULTS_DIR} -o ${RESULTS_DIR}/summary.csv"
