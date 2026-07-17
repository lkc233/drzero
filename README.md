# Dr. Zero: Self-Evolving Search Agents without Training Data

[![Paper](https://img.shields.io/badge/Paper-arXiv:2601.07055-b31b1b.svg)](https://arxiv.org/abs/2601.07055)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for [**Dr. Zero: Self-Evolving Search Agents without Training Data**](https://arxiv.org/abs/2601.07055). In this work, we introduce Dr. Zero, a framework enabling search agents to effectively self-evolve without any training data. In particular, we design a self-evolution feedback loop where a proposer generates diverse questions to train a solver initialized from the same base model. As the solver evolves, it incentivizes the proposer to produce increasingly difficult yet solvable tasks, thus establishing an automated curriculum to refine both agents. To enhance training efficiency, we also introduce hop-grouped relative policy optimization (HRPO). This method clusters structurally similar questions to construct group-level baselines, effectively minimizing the sampling overhead in evaluating each query's individual difficulty and solvability. Consequently, HRPO significantly reduces the compute requirements for solver training without compromising performance or stability. Extensive experiment results demonstrate that the data-free Dr. Zero matches or surpasses fully supervised search agents, proving that complex reasoning and search capabilities can emerge solely through self-evolution.

## 🚀 Overview

The core idea is to bootstrap a search agent from a base model (e.g., Qwen or Llama) via iterative self-evolution: the agent synthesizes tasks and then learns to solve them in a multi-turn, tool-using environment.

*   **Proposer:** A question generation agent that aims to create hard yet solvable questions and thereby driving the solver improvement.
*   **Solver:** The primary search agent that is trained with synthetic data from the proposer to answer challenging questions using the search tool.
*   **Zero-Data Initialization:** The process starts with zero training data and relies solely on an external search engine (e.g., Wikipedia passage retriever).

<img src=verl/intro.png width=1000>

## 🛠️ Setup & Installation

### 1. Environment

Create the Python 3.10 environment and install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

The default environment includes the CUDA 12 builds of PyTorch, SGLang, and FAISS used by the training and retrieval scripts. FlashAttention requires a local CUDA toolkit (`nvcc`); after making it available, install the optional training kernels with:

```bash
uv sync --extra training-kernels
```

### 2. Search Engine

This framework relies on a local server with a retriever model. Prepare the corpus and build the index before training.

The recommended one-click setup uses this repository's `search/` implementation and an isolated uv environment. All data, environments, and caches remain inside the project directory:

```bash
bash setup_retriever.sh
```

By default, the retriever environment is `.venv-retriever`, the index and corpus
are stored under `data/retriever`, and the server listens on port 8020. Training
and keepout evaluation use `http://127.0.0.1:8020/retrieve` by default. Set
`RETRIEVER_PORT` when launching and `DRZERO_RETRIEVER_URL` for training if the
endpoint is different. Run `bash setup_retriever.sh --no-launch` to prepare
without launching.

`FAISS_USE_GPU=auto` is the default. On Hopper/H100 (`sm_90`), setup compiles FAISS GPU from a pinned source commit because the prebuilt wheel lacks H100 kernels. The first build can take several minutes; subsequent runs reuse the installed build. Building and running require a system CUDA 12.x Toolkit with `nvcc`, system BLAS/LAPACK libraries, and a compatible NVIDIA driver. Set `FAISS_USE_GPU=1` or `FAISS_USE_GPU=0` to force GPU or CPU index mode.

**Manual Download & Index Alternative:**
Execute the following commands to download the Wikipedia English dump and build the faiss index for the retriever (default: `intfloat/e5-base-v2`). More details can be found under the search folder and the [Search-R1 repository](https://github.com/PeterGriffinJin/Search-R1).

```bash
save_path=./corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

## 🏃 Iterative Self-Evolution Workflow

The training process proceeds in iterations (Iter 1, Iter 2, Iter 3...). Each iteration typically consists of three phases:

### Phase 0: Initial Data Preparation

Before the first iteration, prepare the initial prompts for training and the benchmarks for evaluation.

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
python scripts/filter_validation_set.py
```

Solver validation uses `data/test_musique.parquet`, containing all 2,417 MuSiQue
examples from `data/test.parquet`. No sampling is applied. The filtering command
fails if the source does not contain exactly 2,417 MuSiQue rows. All iteration
solver scripts use this same validation set.

### Iteration 1

For a stronger local Judge/Updater, prepare the isolated Qwen3.6 serving
environment without changing the training environment:

```bash
bash setup_qwen36_judge.sh --no-launch
bash setup_qwen36_judge.sh
```

The defaults serve `Qwen/Qwen3.6-35B-A3B` on port 8000 with two GPUs. Both the
judge (`meta_model`) and skills/rubrics update (`updater_model`) use this local
OpenAI-compatible endpoint. Override `GPU_DEVICES`, `JUDGE_TP_SIZE`,
`JUDGE_PORT`, or `JUDGE_CONTEXT_LENGTH` when launching it. If the endpoint or
served model changes, set `DRZERO_META_BASE_URL` / `DRZERO_META_MODEL` and
`DRZERO_UPDATER_BASE_URL` / `DRZERO_UPDATER_MODEL` before running the pipeline.
The current scripts colocate this service and the retriever on GPUs 0-1. The
Challenger and generation stages use GPUs 2-7, while Solver training defaults
to all eight GPUs because it does not call Qwen3.6. Stop the local Qwen3.6
service before starting a Solver stage so GPUs 0-1 are available. Verification
runs only after the generation workers exit, then reuses GPU 2; GPU 7 is not reserved. Qwen3.6
defaults to tensor parallel size 2 and an 80% total-device static-memory target;
with the retriever already resident this leaves Qwen enough Mamba/KV cache while retaining
headroom. Override
`TRAIN_GPU_DEVICES`, `SOLVER_GPU_DEVICES`, or `GENERATION_GPU_DEVICES` if your
placement differs. Each
judge-dependent stage checks `/v1/models` before starting.

Start all three persistent services/jobs with the checked-in tmux launcher:

```bash
bash start_training_tmux.sh
```

To resume the complete pipeline at a particular iteration's Solver stage, set
`START_ITERATION` and `START_STAGE`. The first iteration skips its already
completed Challenger and data-generation stages; later iterations run in full:

```bash
ROUNDS=3 START_ITERATION=1 START_STAGE=solver bash run_multiround_training.sh
```

The pipeline prints each stage's start time, finish time, elapsed seconds, and
exit status to the main log. It also appends tab-separated timing records to
`logs/training_timing.tsv`; override the path with `TRAINING_TIMING_LOG`.

### Local and remote service topologies

The training code uses the same Retriever and OpenAI-compatible judge contracts
whether those services run locally or on another server. Copy one deployment
profile and edit only the machine-local copy:

```bash
# All services run on this host.
cp deploy/local-all.env.example deploy/current.env

# Or: this host trains against externally managed services.
cp deploy/remote-services.env.example deploy/current.env
```

`deploy/current.env` is ignored by Git. `MANAGE_RETRIEVER` and `MANAGE_JUDGE`
control service ownership; `DRZERO_RETRIEVER_URL`, `DRZERO_META_BASE_URL`, and
`DRZERO_UPDATER_BASE_URL` control where the algorithm sends requests. A remote
profile therefore skips local service tmux sessions but still validates each
service contract before training.

To select a profile outside the repository:

```bash
DRZERO_DEPLOY_CONFIG=/etc/drzero/server.env bash start_training_tmux.sh
```

The same deployment profile is loaded when an individual iteration script is
run directly. API-key settings contain the *name* of a secret environment
variable (`DRZERO_META_API_KEY_ENV` / `DRZERO_UPDATER_API_KEY_ENV`), never the
secret itself.

It creates (or preserves) the locally managed service sessions and the
`training` session.

**1. Train Proposer:**
Train the proposer agent to generate challenging yet manageable questions for the base solver.

```bash
bash iter1_challenger.sh
```

**2. Synthesize Data:**
Generate training data using the learnt proposer model. Parameters such as model path and sample size can be specified in the script.

For iteration 1, the script first merges the 8-way FSDP proposer checkpoint into
`global_step_50/merged_hf`, then initializes the seven generation workers from that
world-size-independent model. The merge is cached using `.merge_complete`, which
records a source-shard fingerprint so a replaced checkpoint is merged again.

Candidate verification uses an append-only progress journal instead of rewriting the
entire candidate snapshot after every document. A rerun resumes completed verification
groups only when the saved generation manifest still matches the iteration state,
selected source documents, merged model files, rollout configuration, and tool config.
Set `data.resume_candidates=false` as an extra Hydra argument to force regeneration.
`data.candidate_manifest_path` and `data.candidate_progress_path` can override the
default sidecar paths next to the candidate JSONL.

To repair a candidate snapshot whose persisted trajectories are valid but whose
derived format/evidence fields are stale, run:

```bash
python -m verl.iteration.repair_candidates \
  data/<run>.candidates.jsonl \
  --backup-path data/<run>.candidates.before_repair.jsonl
```

The command refuses to overwrite an existing backup or repair a snapshot with a
non-empty verification progress journal. It streams the snapshot, re-extracts
evidence, recomputes format scores with the generation scorer, clears downstream
rubric/verify state, creates a hard-link backup, and atomically replaces the original.
After repair, rerunning the generation command resumes verification from the repaired
snapshot instead of repeating proposer generation.

```bash
bash iter1_gen_data.sh
```

**3. Train Solver:**
Train the solver agent on the generated synthetic data using GRPO. This optimizes the solver's ability to search and reason over challenging questions.

```bash
bash iter1_solver.sh
```

**4. Convert Solver to HF Format:**
Specify the trained model path and convert the FSDP checkpoint to the HF format. This allows the proposer to load the latest solver for reward estimation in the next training iteration.

```bash
bash convert.sh
```

### Subsequent Iterations (Iter 2, Iter 3...)

Repeat the process using the scripts for the respective iteration. The model checkpoints from the previous iteration are used as the starting point for the next. You may need to modify the iteration number and model paths in the scripts.

*   `iter2_challenger.sh` -> `iter2_gen_data.sh` -> `iter2_solver.sh` -> `convert.sh`
*   `iter3_challenger.sh` -> `iter3_gen_data.sh` -> `iter3_solver.sh` -> `convert.sh`

## Citation
If you find Dr. Zero interesting, please consider citing our paper :)
```
@article{yue2026dr,
  title={Dr. Zero: Self-Evolving Search Agents without Training Data},
  author={Yue, Zhenrui and Upasani, Kartikeya and Yang, Xianjun and Ge, Suyu and Nie, Shaoliang and Mao, Yuning and Liu, Zhe and Wang, Dong},
  journal={arXiv preprint arXiv:2601.07055},
  year={2026}
}
```

## License
The code is released under a non-commercial license. See [LICENSE](LICENSE.md) for more details.

## Acknowledgements
During the implementation we base our code mostly on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [VeRL](https://github.com/volcengine/verl). Many thanks to these awesome authors for their great work!
