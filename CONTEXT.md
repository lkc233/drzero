# Dr. Zero Co-Evolution

A proposer-solver co-evolution system where a question-generating model (proposer) and a search-augmented answering model (solver) train iteratively to improve each other.

## Language

### Models and Roles

**Proposer**:
The model that generates multi-hop reasoning questions from source documents. Also called "challenger" in the codebase.
_Avoid_: Challenger (in documentation), question generator

**Solver**:
The model that learns to answer questions using a search tool. Trained via GRPO on proposer-generated questions.
_Avoid_: Agent, answerer, search agent

**Verifier**:
A fixed external model that checks whether a generated question is answerable given its source document. Does not evolve between iterations.
_Avoid_: Validator, checker

### Iteration Structure

**Iteration**:
One complete cycle of the co-evolution loop. Consists of four phases executed in strict order: challenger, gen_data, solver, update_state. All state (skills, rubrics) is frozen within an iteration and only updated at the end.
_Avoid_: Round, epoch, cycle

**Phase**:
One of the four sequential steps within an iteration: `challenger` (proposer RL training), `gen_data` (question generation + verify + filter), `solver` (solver GRPO training), `update_state` (rubric evaluation + skills/rubrics update).
_Avoid_: Step (ambiguous with training step), stage

### Skills and Rubrics

**Skill**:
A single instruction to the proposer about how to generate questions, represented as `{id, instruction, evidence}`. Skills are serialized as JSON and injected into the proposer prompt. They evolve between iterations based on evidence from solver performance.
_Avoid_: Rule, guideline, strategy

**Rubric**:
An evaluation dimension used to score question quality, represented as `{id, name, description, score_min, score_max}`. Rubrics are never shown to the proposer; they influence it indirectly through the skills updater.
_Avoid_: Metric, criterion, evaluation dimension

**Skills Updater**:
An LLM-based component that analyzes iteration evidence (trajectories, rubric scores, verify stats) and produces an updated skills list. Uses a separately configurable model endpoint.
_Avoid_: Skills optimizer, meta-learner

**Rubrics Updater**:
An LLM-based component that updates evaluation dimensions based on iteration evidence. Receives the already-updated skills (not the previous iteration's skills).
_Avoid_: Rubrics optimizer

### Data Flow

**Verify**:
A pre-filter that tests whether a generated question can be answered correctly from its source document by a fixed external model. Questions that fail verify are excluded from solver training.
_Avoid_: Validate, check, filter (too generic)

**Trajectory**:
A complete solver interaction trace for one question: the chain of reasoning steps, tool calls, and the final answer. Each trajectory carries the global_step at which it was generated and a binary correctness label.
_Avoid_: Trace, rollout (overloaded — rollout refers to the generation process, trajectory to the recorded result)

**Condensed Trajectory**:
A trajectory stripped of tool_response content (search results), retaining only think steps, tool call queries, and the final answer. Used as input to the skills updater to keep context manageable.
_Avoid_: Summary, compressed trajectory

**Rollout Correctness**:
Per-question mean of binary correctness across all training rollouts for that question. This is a training-time statistic collected across multiple policy versions, not a fixed-model evaluation.
_Avoid_: Accuracy, success rate (reserved for the aggregate)

**Training Rollout Success Rate**:
The fraction of all training rollouts in an iteration that were correct. An aggregate across the full iteration, not attributable to any single solver version.
_Avoid_: Solver accuracy, overall correctness

### State

**Iteration State**:
The skills and rubrics that are active at the start of an iteration, persisted in `iterations/iter_t/state.json`. Frozen for the duration of the iteration; only updated in the update_state phase.
_Avoid_: Checkpoint (reserved for model weights), config
