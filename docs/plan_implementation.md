# Implementation Plan: Problem Verify + Dynamic Skills/Rubrics

## Context

The drzero proposer-solver co-evolution system suffers from a core failure mode: the proposer generates unanswerable or ambiguous questions that poison solver training. This plan adds three mechanisms to fix it:

1. **Problem verify** — a pre-filter that checks whether generated questions are answerable before feeding them to the solver
2. **Dynamic skills** — an evolving instruction set injected into the proposer prompt, updated each iteration based on evidence
3. **Dynamic rubrics** — evaluation dimensions that score question quality, feeding into the skills updater

These integrate into a new four-phase iteration: `challenger` → `gen_data` → `solver` → `update_state`.

## Design Decisions

| Decision | Choice |
|---|---|
| Verifier model | Fixed external model, same endpoint as reward (port 8001) |
| Verifier input | Document + question, no tools |
| Correctness judge | Existing `em_check`/`normalize_answer` |
| Verify timing | Only during `gen_data`, not RL training |
| Verify scope | After candidate selection (best only, not all n) |
| Skills/rubrics updater | LLM-based, separately configurable endpoint |
| Rubric evaluator | LLM-based, same endpoint as verifier, single call per question |
| Skills injection | From iteration 0, including RL training |
| Rubrics visibility | Evaluation-only, never shown to proposer |
| Rubric evaluation phase | In `update_state` (no solver dependency) |
| Trajectory capture | Minimal hook in ray_trainer.py |
| Trajectory format | Condensed (think steps + tool queries, no tool_response content) |
| Binary correctness | Existing reward score (1.0 = correct) |
| Question identifier | Parquet row index |
| State persistence | Iteration-level directory (`iterations/iter_t/`) |
| Failure handling | Retry 3x on network/JSON/schema/empty errors, then fallback |
| Skills prompt injection | Patch existing baked prompt by string insertion before anchor |

## File Changes Overview

### New files
- `verl/trainer/main_update_state.py` — entry point for the `update_state` phase
- `verl/experimental/dynamic_state/` — module for skills, rubrics, verify logic
  - `__init__.py`
  - `verifier.py` — problem verify implementation
  - `rubric_evaluator.py` — rubric evaluation LLM calls
  - `skills_updater.py` — skills update LLM calls
  - `rubrics_updater.py` — rubrics update LLM calls
  - `state_io.py` — iteration state loading/saving
  - `prompts.py` — prompts for verifier, evaluator, updaters
  - `schemas.py` — JSON schemas for validation
  - `trajectory_utils.py` — trajectory reading, condensing, sampling
- `tests/test_verify.py`
- `tests/test_skills.py`
- `tests/test_rubrics.py`
- `tests/test_state_io.py`
- `tests/test_trajectory_utils.py`
- `tests/test_iteration_flow.py`
- `iter1_update_state.sh` — shell wrapper for update_state phase

### Modified files
- `verl/prompts.py` — add `CURRENT_SKILLS` block template and injection anchor
- `verl/utils/dataset/rl_dataset.py` — runtime skills injection into proposer prompt
- `verl/trainer/main_generation.py` — add verify step, write two outputs
- `verl/trainer/ppo/ray_trainer.py` — add question_index to JSONL dumps
- `config/search_multiturn_grpo.yaml` — new config fields
- `verl/trainer/config/ppo_trainer.yaml` — new config fields
- `iter1_challenger.sh` — pass skills path
- `iter1_gen_data.sh` — pass iteration dir
- `iter1_solver.sh` — set `trainer.rollout_data_dir`

---

## Implementation Steps

### Step 1: State data structures and I/O

**Module:** `verl/experimental/dynamic_state/state_io.py`

Define the canonical data structures and implement load/save for iteration state.

**Skills structure:**
```json
[{"id": "skill-N", "instruction": "string", "evidence": "string"}]
```

**Default skills (when no prior state):**
```json
[{"id": "skill-1", "instruction": "Generate questions that require evidence from the provided document and have one unambiguous answer.", "evidence": "default"}]
```

**Rubrics structure:**
```json
[{"id": "rubric-N", "name": "string", "description": "string", "score_min": 1, "score_max": 5}]
```

**Default rubrics (5 items):** difficulty, answerability, document_dependency, solver_discrimination, ambiguity. Each with `score_min=1`, `score_max=5`.

**Iteration state directory layout:**
```
iterations/
  iter_0/
    state.json          # skills_t, rubrics_t (input state for this iteration)
    gen_metadata.json    # verify results, questions, documents, answers
    verify_stats.json    # generated_total, verify_passed, verify_failed, pass_rate
    rubric_evaluations.json
    skills_update.json   # before, after, diff, raw_output, success, error
    rubrics_update.json  # before, after, diff, raw_output, success, error
  iter_1/
    state.json          # skills_{t+1}, rubrics_{t+1} from iter_0's update
    ...
```

**Key functions:**
- `load_iteration_state(base_dir, iteration) -> (skills, rubrics)` — reads `iterations/iter_{t}/state.json`, falls back to defaults if missing
- `save_iteration_state(base_dir, iteration, skills, rubrics)`
- `save_gen_metadata(base_dir, iteration, metadata)` — called by gen_data
- `save_verify_stats(base_dir, iteration, stats)` — called by gen_data
- `save_update_result(base_dir, iteration, update_type, result)` — called by update_state
- `compute_diff(before, after, key_field="id")` — computes added/removed/modified diff by id

**Schema validation module:** `verl/experimental/dynamic_state/schemas.py`

---

### Step 2: Skills injection into proposer prompt

**Problem:** The proposer prompt is baked into parquet at `process_train.py:68`. Skills must be injected at runtime because they change between iterations.

**Approach:** Patch existing baked prompt by string insertion.

**2a. Add to `verl/prompts.py`:**

```python
SKILLS_BLOCK = """
### Current Skills
Follow these skill instructions when generating questions:
{skills_json}
"""

SKILLS_INJECTION_ANCHOR = "Now, generate a question"
```

Skills are inserted immediately before this anchor in the existing prompt.

**2b. Add injection utility to `verl/experimental/dynamic_state/state_io.py`:**

```python
def inject_skills_into_prompt(prompt_content: str, skills: list[dict]) -> str:
    skills_json = json.dumps(skills, indent=2)
    skills_block = SKILLS_BLOCK.format(skills_json=skills_json)
    return prompt_content.replace(
        SKILLS_INJECTION_ANCHOR,
        skills_block + SKILLS_INJECTION_ANCHOR
    )
```

**2c. Modify `verl/utils/dataset/rl_dataset.py`:**

In `__getitem__()` (around line 211), after building messages:
- Check if `self.config.get("skills_path")` is set
- If so, load skills (cached at init) and call `inject_skills_into_prompt()` on the user message content
- If not, use the prompt as-is (backward compatible)

**2d. For `main_generation.py`:** Same injection — read skills from iteration state, inject into each prompt before generation.

**Key file paths:**
- `verl/prompts.py:18-75` — `DEFAULT_CHALLENGER_PREFIX`
- `verl/utils/dataset/rl_dataset.py:211-328` — `__getitem__()`
- `process_train.py:56-102` — `process_single_row()`

---

### Step 3: Problem verify mechanism

**Module:** `verl/experimental/dynamic_state/verifier.py`

**Verify flow (called from `main_generation.py`):**
1. Format prompt with document + question
2. Call external model (single-turn, no tools) → `model_answer`
3. Extract answer from `<answer>` tags using existing `extract_solution()` (`verl/utils/reward_score/search_r1_like_qa_em.py:66`)
4. Compare with `reference_answer` using `em_check()` (`verl/custom_reward/reward_function.py:44`)
5. Return `{"model_answer": str, "passed": bool, "reason": str}`

**Verifier prompt (`verl/experimental/dynamic_state/prompts.py`):**
```
Given the following document and question, provide your answer.

Document: {document}

Question: {question}

Provide your answer inside <answer> and </answer> tags.
```

**Model call:** Reuse the HTTP client pattern from `reward_rollout.py:603-653`. POST to `{base_url}/v1/completions`. Retry 3 times with 1s and 2s backoff.

**Integration into `main_generation.py`:**

After line 199 (`sample_idx = np.argmax(format_scores)`):
```python
if config.get("enable_problem_verify", True):
    verify_result = await verify_question(
        document=raw_doc,
        question=raw_qs[sample_idx],
        reference_answer=raw_ans[sample_idx],
        base_url=..., model_name=...,
    )
    if not verify_result["passed"]:
        failed_samples.append({...})
        continue
```

After the batch loop, write two outputs:
1. **Filtered parquet** (existing format, minus failed questions) — for solver
2. **Gen metadata JSON** — all questions with verify results, documents, skills_t/rubrics_t snapshot — for update_state

**Config (`config/search_multiturn_grpo.yaml`):**
```yaml
problem_verify:
  enable: true
  base_url: "http://127.0.0.1:8001"
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  max_retries: 3
```

---

### Step 4: Solver trajectory capture

**Goal:** JSONL dumps include `question_index` for per-question `rollout_correctness`.

**4a. Add `question_index` to solver parquet** in `main_generation.py`:
```python
dataset["question_index"] = range(len(dataset))
```

Field flows through: Parquet → `RLHFDataset.__getitem__()` → `collate_fn()` → `DataProto.from_single_dict()` → `batch.non_tensor_batch["question_index"]`.

**4b. Modify `ray_trainer.py`** at lines 1345-1362, before `_dump_generations`:
```python
if "question_index" in batch.non_tensor_batch:
    reward_extra_infos_dict["question_index"] = batch.non_tensor_batch["question_index"].tolist()
```

~3 lines of change. Existing `_dump_generations` includes all `reward_extra_infos_dict` entries.

**4c. Enable rollout data dumping** in `iter1_solver.sh`:
```bash
trainer.rollout_data_dir="./iterations/iter_${iter}/solver_trajectories"
```

**Binary correctness:** Already captured — `score` in JSONL is 1.0 for correct (`compute_score` in `search_r1_like_qa_em.py:96`).

---

### Step 5: Trajectory utilities

**Module:** `verl/experimental/dynamic_state/trajectory_utils.py`

**Functions:**
- `load_solver_trajectories(trajectory_dir) -> List[dict]`
- `condense_trajectory(raw_trajectory) -> dict` — condensed format:
  ```json
  {
    "question_index": 42,
    "question": "...",
    "reference_answer": "...",
    "solver_answer": "...",
    "correct": false,
    "global_step": 150,
    "reasoning_steps": ["<think>...</think>", "<tool_call>...</tool_call>", ...]
  }
  ```
- `sample_trajectories(trajectories, sample_size, seed) -> List[dict]` — deterministic, no replacement. If len < sample_size, return all.
- `compute_rollout_stats(trajectories) -> dict` — per-question `rollout_correctness` + overall `training_rollout_success_rate`

**Parsing:** Regex to extract `<think>`, `<tool_call>` blocks; strip `<tool_response>` content; extract `<answer>` for solver_answer.

---

### Step 6: Rubric evaluator

**Module:** `verl/experimental/dynamic_state/rubric_evaluator.py`

Single LLM call per question scoring all rubrics at once.

**Prompt (`verl/experimental/dynamic_state/prompts.py`):**
```
Evaluate the following question on each rubric. For each rubric, provide a score (1-5) and a brief reason.

Document: {document}
Question: {question}
Reference Answer: {reference_answer}

Rubrics:
{rubrics_json}

Respond in JSON format:
[{"rubric_id": "...", "score": N, "reason": "..."},  ...]
```

**Functions:**
- `async evaluate_question(...) -> List[dict]` — single question
- `async evaluate_all_questions(...) -> List[List[dict]]` — batch with async concurrency

On failure after 3 retries: all scores = 1, reason = error message.

---

### Step 7: Skills updater

**Module:** `verl/experimental/dynamic_state/skills_updater.py`

**Prompt inputs:** skills_t, rollout stats (with cross-policy-version caveat), condensed trajectories, rubric evaluations, verify stats.

**Function:**
```python
async update_skills(skills_t, rollout_stats, sampled_trajectories,
                    rubric_evaluations, verify_stats,
                    base_url, model_name) -> (skills_new, update_record)
```

- Validates schema: each skill has id, instruction, evidence; ids unique; non-empty list
- On 3rd failure: return `skills_t` unchanged, record error
- Returns (new_skills, {before, after, diff, raw_output, success, error})

**Config:**
```yaml
state_update:
  base_url: "http://127.0.0.1:8001"  # separately configurable
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  max_retries: 3
  skill_update_trajectory_sample_size: 8
```

---

### Step 8: Rubrics updater

**Module:** `verl/experimental/dynamic_state/rubrics_updater.py`

Same pattern as skills updater. Key difference: receives `skills_{t+1}` (just updated), not `skills_t`.

**Function:**
```python
async update_rubrics(rubrics_t, skills_new, rubric_evaluations,
                     rollout_stats, verify_stats,
                     base_url, model_name) -> (rubrics_new, update_record)
```

Schema: id, name, description, score_min=1, score_max=5; ids unique; non-empty list.

---

### Step 9: Update state entry point

**File:** `verl/trainer/main_update_state.py`

```python
def main(config):
    iteration = config.iteration
    base_dir = config.iterations_dir

    # 1. Load state
    skills_t, rubrics_t = load_iteration_state(base_dir, iteration)

    # 2. Load gen_data metadata
    gen_metadata = load_gen_metadata(base_dir, iteration)
    verify_stats = load_verify_stats(base_dir, iteration)
    passed_questions = [q for q in gen_metadata if q["verify_passed"]]

    # 3. Load and process solver trajectories
    raw_trajectories = load_solver_trajectories(config.solver_trajectory_dir)
    rollout_stats = compute_rollout_stats(raw_trajectories)
    condensed = [condense_trajectory(t) for t in raw_trajectories]
    sampled = sample_trajectories(condensed, config.skill_update_trajectory_sample_size, config.seed)

    # 4. Rubric evaluation
    rubric_evaluations = await evaluate_all_questions(passed_questions, rubrics_t, ...)
    save_rubric_evaluations(base_dir, iteration, rubric_evaluations)

    # 5. Update skills
    skills_new, skills_record = await update_skills(
        skills_t, rollout_stats, sampled, rubric_evaluations, verify_stats, ...
    )
    save_update_result(base_dir, iteration, "skills", skills_record)

    # 6. Update rubrics (uses skills_new, NOT skills_t)
    rubrics_new, rubrics_record = await update_rubrics(
        rubrics_t, skills_new, rubric_evaluations, rollout_stats, verify_stats, ...
    )
    save_update_result(base_dir, iteration, "rubrics", rubrics_record)

    # 7. Save state for next iteration
    save_iteration_state(base_dir, iteration + 1, skills_new, rubrics_new)
```

**Shell wrapper (`iter1_update_state.sh`):**
```bash
python -m verl.trainer.main_update_state \
    --iteration=0 \
    --iterations_dir="./iterations" \
    --solver_trajectory_dir="./iterations/iter_0/solver_trajectories" \
    --seed=42 \
    --skill_update_trajectory_sample_size=8 \
    --state_update.base_url="http://127.0.0.1:8001" \
    --state_update.model_name="Qwen/Qwen2.5-3B-Instruct"
```

---

### Step 10: Configuration changes

**`config/search_multiturn_grpo.yaml`:**
```yaml
problem_verify:
  enable: true
  base_url: "http://127.0.0.1:8001"
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  max_retries: 3

state_update:
  base_url: "http://127.0.0.1:8001"
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  max_retries: 3
  skill_update_trajectory_sample_size: 8

iterations_dir: "./iterations"
```

**Backward compatibility:** `load_iteration_state` falls back to defaults when no state exists. Old parquets without `question_index` still load. Old checkpoints unaffected — skills/rubrics live in separate iteration dirs.

---

### Step 11: Shell script updates

Full iteration sequence:
```bash
bash iter1_challenger.sh    # Phase 1: Train proposer with skills_t
bash iter1_gen_data.sh      # Phase 2: Generate + verify + filter
bash iter1_solver.sh        # Phase 3: Train solver, dump trajectories
bash iter1_update_state.sh  # Phase 4: Evaluate rubrics, update skills/rubrics
```

Changes per script:
- `iter1_challenger.sh` — pass `data.skills_path` to inject skills during RL training
- `iter1_gen_data.sh` — pass `--iterations_dir` and `--iteration` for state loading and metadata output
- `iter1_solver.sh` — set `trainer.rollout_data_dir` for trajectory capture

---

### Step 12: Tests

| Test file | Cases |
|---|---|
| `tests/test_verify.py` | correct answer passes; incorrect fails; model call failure → false; parse failure → false; `enable=false` → all pass; stats correct incl. total=0 |
| `tests/test_skills.py` | prompt contains serialized CURRENT_SKILLS in order; update input includes all evidence; success → new state + diff; failure → retains old |
| `tests/test_rubrics.py` | evaluation produces all scores per question; eval failure → scores=1; update uses skills_{t+1}; success/failure handling |
| `tests/test_state_io.py` | save/restore round-trip; missing state → defaults |
| `tests/test_trajectory_utils.py` | condensing strips tool_response; same seed → same sample; insufficient → all; rollout_correctness and success_rate correct; global_step + correctness present |
| `tests/test_iteration_flow.py` | full single-iteration ordering matches spec |

**Run:** `pytest tests/ -v`

---

## Verification

1. **Unit tests:** `pytest tests/ -v` — all pass
2. **Existing tests:** no regressions
3. **Integration smoke test:** one full iteration on small dataset — skills injected, verify filters, trajectories captured, rubrics evaluated, state updated, diffs logged
4. **Backward compatibility:** old parquet/checkpoint without new fields → defaults used, no errors
