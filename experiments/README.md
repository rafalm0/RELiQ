# RELiQ vs RT-RELiQ

| method | flag | who decides purification |
| --- | --- | --- |
| **RELiQ** (default) | none, or `--method reliq` | the network, automatically (`--auto-distillation-threshold`, default 0.98); the agent only routes |
| **RT-RELiQ** | `--method rt-reliq` | the agent, per hop, in the same action as the routing choice; automatic purification off |

With no new flags the code runs the upstream RELiQ path unchanged. This was checked
against unmodified `HEAD` for the trained policy and for `qpath`: 211/211 evaluation
metrics matched, timing fields excluded.

## RT-RELiQ options

| flag | default | meaning |
| --- | --- | --- |
| `--max-purification-rounds P` | 3 | rounds the agent may pick per hop; action space becomes `1 + neighbours·(P+1)` |
| `--purification-cost-weight λ` | 0 | terminal reward penalty per extra link burned purifying |
| `--keep-auto-distillation` | off | ablation: automatic purification stays on as well |

The action layout is `0 = idle` and `1 + e·(P+1) + r = take edge e with r rounds`. Only
`env.encode_action` and `env.decode_action` know this layout. RT-only flags are rejected
under `--method reliq`, and `--method rt-reliq` requires `--policy trained`.

Reward and success definition are RELiQ's unchanged (`max(fidelity, 0.5)` on reaching
the target, 0 if dropped, −0.75 for idling when a move exists). At `λ = 0` RT-RELiQ's
reward is identical to RELiQ's. Links burned purifying then cost the agent nothing
directly; they only cost *other* requests through link scarcity. *Hypothesis, not
measured:* this pushes the agent towards heavy purification. λ adds a direct
cost.

## Metrics

All upstream metrics keep their definitions. In particular, `resources_per_success`
still counts one link per hop and does **not** include purification. New metrics:

| metric | meaning |
| --- | --- |
| `purification_links_agent(_per_success)` | links burned by agent-chosen purification (RT-RELiQ; also qpath/qleap) |
| `purification_links_auto(_per_success)` | links burned by the network's automatic purification (RELiQ) |
| `purification_rounds_requested` | rounds the agent asked for |
| `total_links_per_success` | routing + agent + auto per success. **Use this to compare methods** |

## Training the best model of each method (`train_best.sh`)

```bash
bash experiments/train_best.sh            # on the server; DEVICE=cuda by default
```

The script trains, selects and evaluates three methods with the RELiQ README recipe
(15M steps, DQN + NetMon/SAGE, future rewards):

| key | what | flags |
| --- | --- | --- |
| `reliq` | main baseline, original | none |
| `rt-reliq` | agent-chosen purification + ratio reward | `--method rt-reliq --max-purification-rounds 3 --reward-type ratio` |
| `reliq-ratio` | ablation: RELiQ on the ratio reward | `--reward-type ratio` |

The phases are `train`, `validate`, `select`, `test` and `report`:

1. Every run's `model_best.pt` and `model_last.pt` are scored on validation topologies
   `EVAL_SEEDS[500:550]`.
2. For each method, the best run is copied to `best_models/<method>.pt`.
3. Every run's selected checkpoint is tested on `EVAL_SEEDS[0:100]`, the standard test
   set, using the README evaluation flags (`--use-realistic-decay`).

The test set is never used for selection. Outputs go to `results/best/`:
`best_models/`, `selection.json`, and `report/{best_models.png, results.md,
test_metrics_best.json}`. Finished work is skipped, so the script can be re-run;
`PHASES="report"` redoes only the plot. The full list of settings is at the top of the
script (`SEEDS`, `JOBS`, `DEVICE`, `TOTAL_STEPS`, `SELECT_METRIC`, ...).
`best_models.py` needs only numpy and matplotlib, so the repo `.venv` is enough.

### Ratio reward (`--reward-type ratio`)

On reaching the target, the reward is `max(F, 0.5) · h_min / max(links, hops, 1)`. Here
`links` are the links the request consumed (routing plus its own purification), and
`h_min` is its shortest hop count.

- `h_min` is fixed per request, so within a request this is exactly fidelity per link.
  It keeps the reward on RELiQ's 0.5–1 scale, and requests of different lengths count
  equally.
- Because the denominator is at least the hops travelled, breaking a swap chain early
  never pays more than a full delivery.
- For RELiQ, automatic purification is network-side and not attributable to a request,
  so the ratio only sees routing links.

## Parameter sweep (`run_purification_sweep.sh`)

```bash
bash experiments/run_purification_sweep.sh     # from the repo root
```

Arms: `reliq` (default flags only), `reliq_nopurif` (automatic purification off), and
`rt-reliq_P{1,2,3}_c{0,0.005,0.02}`, each with 3 seeds. Override settings with the
environment variables `RESULTS_DIR`, `SEEDS`, `ROUNDS`, `COSTS`, `TOTAL_STEPS`, `JOBS`,
`EXTRA_ARGS` and `DRY_RUN=1`. Finished runs are skipped.

```bash
python experiments/collect_metrics.py results/purification -o summary.csv --per-episode
python experiments/plot_purification.py summary.csv -o figures/
```

These two scripts need pandas and matplotlib, which the repo `.venv` does not include.
Run them from any environment that has both.

- `fig1_resource_efficiency.png`: links per success, stacked as routing, agent purification and automatic purification
- `fig2_cost_quality.png`: cost plotted against fidelity and against success rate. A saving only counts if the point moves left without moving down.
- `fig3_cost_weight_sweep.png`: effect of λ, one line per P

## Tests

```bash
PYTHONPATH=src python tests/test_purification_action_space.py
```
