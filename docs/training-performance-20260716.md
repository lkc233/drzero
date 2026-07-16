# Training performance diagnosis — Iteration 1

## Technical summary

The first 10 completed production steps average **290.6 seconds per step**. Challenger generation is the primary bottleneck at **185.6 seconds (63.9%)**, followed by reward computation at **75.9 seconds (26.1%)**. Actor optimization is only **23.1 seconds (8.0%)**, so optimizing backward/update kernels first would have little end-to-end impact.

The reward stage becomes slower as the policy produces more valid candidates: average format score and reward latency have a Pearson correlation of **0.83** in this window. This is expected because valid candidates trigger five Solver samples and Rubric evaluation. Mean response length and generation latency have a correlation of **0.94**, making generated-token volume the strongest observed generation driver.

## Module timing

| Module | Mean | Median | Min–max | Share of step |
|---|---:|---:|---:|---:|
| Challenger generation, including tool use | 185.6 s | 171.3 s | 157.1–246.0 s | 63.9% |
| Reward: Solver + Rubric + scoring | 75.9 s | 88.0 s | 23.3–126.1 s | 26.1% |
| Actor update | 23.1 s | 22.5 s | 22.1–27.3 s | 8.0% |
| Old log-probability | 6.0 s | 5.5 s | 5.2–10.7 s | 2.1% |
| Advantage calculation | <0.1 s | <0.1 s | <0.1 s | <0.1% |
| **Complete step** | **290.6 s** | **290.1 s** | **258.3–315.2 s** | **100%** |

`reshard` averages 2.6 seconds but is already included inside VERL's `gen` timer, so it is not counted again in the additive shares.

## What the current evidence supports

1. **Generated-token volume is the first optimization target.** Responses average roughly 1.4k–2.0k tokens, and response length explains most observed variation in generation latency. However, 20–31% of responses hit the 2,560-token cap, so simply lowering the cap risks truncating more samples and degrading training quality.
2. **Reward work scales with successful policy output.** As format-valid output rises, more samples qualify for five Solver rollouts and Rubric evaluation. Reward latency therefore grows during learning rather than remaining constant.
3. **Increasing training-kernel speed has limited leverage.** Even eliminating the entire actor-update phase would save only about 8%.
4. **The current top-level timer cannot separate Retriever, Solver, and Rubric.** A detailed reward timer has now been added for format parsing, Solver preparation, Solver rollout, Rubric preparation, Rubric evaluation, final scoring, and total reward time. It takes effect when a new training process starts.

## Recommended optimization order

1. Measure one full new-process window with the new Solver/Rubric timers before changing reward semantics.
2. Reduce unnecessary Challenger verbosity through prompting or stopping criteria while preserving the 2,560 hard cap until truncation quality is evaluated.
3. Run Solver and Rubric stages concurrently because they use separate model services and Rubric construction does not depend on Solver outputs. Validate GPU contention and output equivalence with a one-step smoke test before production use.
4. After separate timings are available, tune Solver concurrency from 64 using a small sweep. Choose throughput from measured latency and failure rate rather than increasing it blindly.
5. Keep `reward_rollout_n=5` unless an accuracy study shows that 3–4 samples preserve the difficulty estimate; reducing it is high leverage but changes reward variance.

## Scope and methodology

Source window: production Iteration 1 log `iter1_challenger_ratio4321_20260716_034703.log`, completed steps 1–10. Values come from VERL's per-step `timing_s/*` fields. Shares use summed component wall time divided by summed step wall time. Correlations are descriptive Pearson correlations over ten observations and do not establish causality.

The reusable command is:

```bash
.venv/bin/python scripts/analyze_training_timing.py \
  logs/iter1_challenger_ratio4321_20260716_034703.log
```

## Limitations and next questions

- Retriever latency is included in multi-turn generation and Solver rollout but is not yet emitted as a separate aggregate.
- Solver and Rubric are internally concurrent across requests, but in the current reward function the two stages execute sequentially relative to each other.
- Ten steps are enough to identify the dominant modules, but not enough to select a stable concurrency setting.
- The next decision should use at least five steps containing the new detailed reward timing line.
