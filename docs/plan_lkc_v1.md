# Dr.Zero：问题验证与动态 Skills/Rubrics 实施计划

## 1. 目标

在现有 proposer（代码中称 `challenger`）与 solver 协同进化流程上增加：

1. 问题 verify：确认合成问题和标准答案被 proposer 使用的完整证据支持。
2. 动态 skills：根据新 solver 在 keepout 集上的表现，逐轮更新 proposer 的问题生成指导。
3. 动态 rubrics：逐轮更新问题质量评价标准，并将 rubric 评分加入 proposer 训练 reward。
4. 版本化 iteration state：由统一 orchestrator 管理跨阶段状态、产物、恢复和审计。

不改变当前多跳问题定义、搜索工具协议和 solver 的 GRPO 训练目标；不做无关重构。

## 2. 术语

- **Proposer**：生成问题和标准答案的模型；现有代码与脚本中称 `challenger`。
- **Solver**：学习回答 proposer 问题的模型。
- **Verifier**：轮初 solver 在“给定完整证据、禁止搜索”的模式下承担的验证角色。
- **Meta/Judge Model**：独立、可配置的模型服务，负责语义等价判断、rubric 评分、trajectory 分析和 skills/rubrics 更新。
- **Evidence Bundle**：seed 文档及 proposer 搜索过程中实际获得的全部证据的结构化集合。
- **Keepout Set**：本轮 verify 通过、但未用于 solver 训练的数据；用于训练后评估新 solver。
- **Iteration State**：某轮的版本化状态，包含模型引用、skills、rubrics、阶段状态、配置快照和产物路径。

## 3. 当前实现基线

当前流程由多组 shell 脚本手工串联：

1. `iterN_challenger.sh`：训练 proposer。
2. `iterN_gen_data.sh` → `verl/trainer/main_generation.py`：每个 seed 生成多个候选，仅按 format score 选一个。
3. `iterN_solver.sh`：用生成数据训练 solver。
4. `convert.sh`：将 solver checkpoint 转为下一轮使用的 HF 模型。

关键现状：

- `verl/custom_reward/reward_function.py` 中 proposer reward 为：
  `0.5 × format_score + difficulty_score`。
- `verl/custom_reward/reward_rollout.py` 已支持异步多轮 solver rollout、搜索工具调用和重试。
- `main_generation.py` 只保存 solver prompt、标准答案和 `metadata.raw_context`，没有结构化 evidence、verify 或 keepout。
- 训练 checkpoint 只覆盖模型、优化器和 dataloader 等 VeRL 状态，没有跨 proposer/solver 的 iteration state。

## 4. 固定单轮流程

设第 `t` 轮开始时状态为：

- proposer 模型 `proposer_t`
- solver 模型 `solver_t`
- `skills_t`
- `rubrics_t`

一轮严格执行：

1. orchestrator 加载并校验 `state_t`。
2. 将完整 `skills_t` 注入 proposer prompt。
3. 用 `rubrics_t` 计算 rubric reward，训练得到本轮 proposer。
4. 每个 seed 文档生成 5 个候选问题。
5. 使用 `rubrics_t` 评价候选并与 format score 组合排序。
6. 按排序依次 verify，首个通过的候选进入本轮有效数据集。
7. 按稳定 `doc_id` 分组，将有效数据切分为 90% solver train 和 10% keepout。
8. 只使用 solver train 训练 `solver_t`，得到 `solver_{t+1}`。
9. 用 `solver_{t+1}` 在 keepout 上执行 question-only、允许搜索的固定评估。
10. 对每条 keepout trajectory 做结构化分析，再汇总成全局报告。
11. 基于本轮证据先更新 `skills_t → skills_{t+1}`。
12. 使用新 `skills_{t+1}` 再更新 `rubrics_t → rubrics_{t+1}`。
13. 原子保存全部产物和 `state_{t+1}`，并在 proposer checkpoint 中写入状态快照或内容寻址引用。

`skills_t` 和 `rubrics_t` 在本轮内不可变。轮末产生的新状态只允许下一轮使用，不得反向影响本轮已经完成的训练、生成、verify 或评估。

## 5. Proposer 候选生成与排序

### 5.1 候选生成

保持当前每个 seed 生成 5 个候选的行为，但不得再只保存拼接后的 `raw_context`。每个候选至少结构化保存：

```json
{
  "candidate_id": "string",
  "iteration": 0,
  "doc_id": "string",
  "hop_count": 0,
  "source_document": "string",
  "proposer_trajectory": [],
  "evidence_bundle": [],
  "question": "string",
  "reference_answer": "string",
  "format_score": 0.0,
  "rubric_evaluation": [],
  "rank_score": 0.0
}
```

`doc_id` 由源数据中的稳定 ID 生成；源数据没有 ID 时，对规范化后的 seed 文档计算稳定哈希。禁止使用行号或本次运行生成的 UUID 作为切分键。

### 5.2 Evidence Bundle

Evidence Bundle 必须包含：

1. seed 文档；
2. proposer 每次 search 的 query；
3. 对应 tool response 中实际返回的文档/片段；
4. 证据顺序和来源；
5. 可回溯到 proposer trajectory 中原始消息的位置。

推荐结构：

```json
{
  "evidence_id": "evidence-N",
  "kind": "seed_document | search_result",
  "query": "string",
  "content": "string",
  "source": "string",
  "trajectory_index": 0
}
```

解析失败、缺失 tool response 或证据无法与 trajectory 对齐时视为阶段失败，不允许静默丢失证据后继续。

### 5.3 Rubric 排序

对 5 个候选全部使用当前 `rubrics_t` 评分：

```text
normalized_rubric_mean = mean((score - 1) / 4)
rank_score = 0.5 × format_score + 0.5 × normalized_rubric_mean
```

按 `rank_score` 降序执行 verify；分数相同时使用候选生成顺序作为稳定 tie-breaker。第一个通过者进入有效数据集，其余候选仍保存评分和状态，但不进入 solver 数据。

## 6. 问题 Verify

### 6.1 目的

verify 检查的是：

1. 轮初 solver 读取完整 Evidence Bundle 和 question 后，3 次回答中是否至少 1 次正确；
2. 同一 solver 只读取 question 后，3 次回答是否全部错误。

verify 不是 keepout 能力评估，也不允许通过外部搜索补足 Evidence Bundle。

### 6.2 Verifier 执行

- 模型：轮初 `solver_t`。
- 输入分两组：Evidence Bundle + question，以及仅 question。
- 搜索工具：禁用。
- 采样数：每组固定 `K=3`，共 6 次。
- 正确性：使用与 solver reward 一致的归一化 exact match。
- 输出：保留两组三次完整响应、提取答案和逐次 correctness。

### 6.3 两条件判定

不再调用独立 Meta/Judge。程序直接汇总两组 correctness：

- `with_evidence_succeeded = any(with_evidence_samples[].correct)`；
- `question_only_all_incorrect = not any(question_only_samples[].correct)`；
- `passed = with_evidence_succeeded and question_only_all_incorrect`。

结构化输出：

```json
{
  "verification_mode": "two_condition_em",
  "with_evidence_samples": [{"sample_index": 0, "correct": true}],
  "question_only_samples": [{"sample_index": 0, "correct": false}],
  "passed": true,
  "reason": "string"
}
```

每个候选保存 solver 原始输出、提取答案、correctness、汇总判定、耗时和失败原因。

## 7. 动态 Skills

### 7.1 Schema

```json
{
  "id": "skill-N",
  "instruction": "string",
  "evidence": "string"
}
```

约束：

- `id` 唯一；
- updater 返回完整列表，不返回增量补丁；
- 可新增、改写和删除；
- 最多 12 项；
- 对单项和总 prompt 设置明确长度上限；
- 每轮保存 added、removed、modified diff。

### 7.2 初始 Skills

第一轮固定包含三项：

1. 构造从 seed 文档到最终答案的完整、可验证证据链。
2. 确保每一跳对最终答案都是必要的，不能跳过中间关系。
3. 消除问题、证据关系和标准答案中的歧义。

### 7.3 Prompt 注入

在运行时把按列表顺序序列化的完整 `skills_t` 注入 proposer prompt。训练和 gen data 必须走同一注入函数。

不得只修改 `process_train.py` 后重新生成静态 parquet，否则 resume 或下一轮更新可能继续使用旧 skills。

## 8. 动态 Rubrics 与 Proposer Reward

### 8.1 Schema

```json
{
  "id": "rubric-N",
  "name": "string",
  "description": "string",
  "score_1_anchor": "string",
  "score_3_anchor": "string",
  "score_5_anchor": "string"
}
```

约束：

- 每项评分为 1–5 的整数；
- 所有 rubrics 等权；
- `id` 唯一；
- 可新增、改写和删除；
- 最多 12 项；
- updater 返回完整列表；
- 每轮保存 added、removed、modified diff。

### 8.2 初始 Rubrics

第一轮固定包含：

1. **证据支持**：问题和答案是否由完整证据链支持。
2. **答案唯一性**：证据是否导向一个明确、规范的答案。
3. **多跳必要性**：每一跳是否不可跳过且参与最终推理。
4. **检索可解性**：问题在正常搜索设置下是否可被 solver 解出。
5. **能力区分度**：问题是否能区分不同水平的 solver，而非过易或无解。

每项必须在初始状态中给出明确的 1/3/5 分锚点。

### 8.3 Rubric 评价输入

训练 proposer 时，Meta/Judge Model 对每条 rollout 查看：

- seed 文档；
- 完整 proposer 搜索 trajectory；
- question；
- reference answer；
- 当前全部 `rubrics_t`。

输出每项 score 和 reason，并计算：

```text
normalized_rubric_mean = mean((score - 1) / 4)
```

### 8.4 Reward 组合

保留现有 reward，新增 rubric 项：

```text
proposer_reward =
    0.5 × format_score
    + difficulty_score
    + 0.5 × normalized_rubric_mean
```

日志必须分别记录三个 reward 分量和最终值，避免只记录总分后无法定位 reward 漂移。

## 9. Solver Train/Keepout 切分

对 verify 通过的数据按 `doc_id` 做稳定分组切分：

- 90%：solver train；
- 10%：keepout。

要求：

- 同一 `doc_id` 的所有候选只能进入同一侧；
- 使用项目全局 seed 和稳定哈希，保证重跑结果一致；
- 切分 manifest 独立保存；
- keepout 数据不得被 dataloader、采样器或 resume 状态送入 solver 训练；
- 空 train 或空 keepout 必须在训练前报错并中止本轮。

## 10. 新 Solver 的 Keepout 评估

训练结束后使用 `solver_{t+1}` 评估 keepout：

- 输入只包含 question；
- 允许使用现有 search tool；
- 每题生成 1 条完整 trajectory；
- Meta/Judge Model 判断最终答案与 reference answer 是否语义等价；
- 每题保存二值 `correct`，全局计算 keepout accuracy。

逐题记录：

```json
{
  "candidate_id": "string",
  "doc_id": "string",
  "question": "string",
  "reference_answer": "string",
  "trajectory": [],
  "model_answer": "string",
  "judge_result": {},
  "correct": true
}
```

这里不计算“逐题准确率”：每题只有一个 trajectory，因此只有二值 correctness。总体 accuracy 是正确题数除以 keepout 总题数。

## 11. Trajectory 分层分析

### 11.1 逐条 Summary

对 keepout 中每条 trajectory 生成结构化 summary：

```json
{
  "candidate_id": "string",
  "correct": true,
  "outcome_stage": "string",
  "root_causes": ["string"],
  "related_rubric_ids": ["string"],
  "evidence_quotes": ["string"],
  "actionable_improvements": ["string"]
}
```

分析既覆盖失败样本，也覆盖成功样本，不随机抽样。

### 11.2 全局报告

若全部 summaries 超出上下文窗口，使用确定性的分块 map-reduce：

1. 固定顺序分块；
2. 每块生成局部报告；
3. 汇总局部报告为全局报告；
4. 保留问题频次、成功/失败模式、关联 rubric、代表案例和改进建议；
5. 保存每层输入、原始输出和结构化结果。

原始 trajectory、逐条 summary、局部报告和全局报告均不得相互覆盖。

## 12. Skills 与 Rubrics 更新

### 12.1 Skills Updater

输入：

- `skills_t`；
- `rubrics_t` 及本轮 rubric evaluations；
- verify 结果与统计；
- keepout accuracy；
- 全部逐条 summaries 和全局报告。

输出完整 `skills_{t+1}`。更新器必须引用输入证据解释每项保留、改写、删除或新增的原因。

### 12.2 Rubrics Updater

Skills 更新成功后再调用。输入：

- `rubrics_t`；
- 新的 `skills_{t+1}`；
- 本轮 rubric evaluations；
- verify 结果与统计；
- keepout accuracy；
- 全部逐条 summaries 和全局报告。

输出完整 `rubrics_{t+1}`。禁止使用 `skills_t` 替代 `skills_{t+1}`。

## 13. Meta/Judge Model

语义 judge、rubric 评分、trajectory 分析和两个 updater 共用一个可配置模型服务，但使用独立 prompt 和独立 schema。

至少配置：

```yaml
meta_model:
  model_name: string
  base_url: string
  api_key_env: string
  timeout_seconds: 120
  max_retries: 3
  max_concurrency: 32
```

不得在配置、日志或 iteration state 中写入 API key 明文。

所有结构化输出必须通过 schema 校验。模型调用、超时、解析或 schema 校验在重试耗尽后均中止本轮。

## 14. Orchestrator 与 Iteration State

### 14.1 Orchestrator

新增 iteration-level Python CLI，负责：

- 读取配置和 `state_t`；
- 启动或调用现有 proposer、generation、solver、convert 阶段；
- 执行 verify、切分、keepout eval、分析和状态更新；
- 检查阶段前置条件；
- 保存阶段 manifest；
- 支持从最后一个成功阶段恢复；
- 防止同一 iteration 并发写入。

现有训练入口继续承担单阶段计算，不把跨阶段状态机塞入 `RayPPOTrainer.fit()`。

### 14.2 State Schema

```json
{
  "schema_version": 1,
  "iteration": 0,
  "status": "running | failed | completed",
  "models": {
    "proposer": "string",
    "solver_before": "string",
    "solver_after": "string"
  },
  "skills": [],
  "rubrics": [],
  "config_snapshot": {},
  "stages": {},
  "artifacts": {},
  "created_at": "string",
  "updated_at": "string"
}
```

State 使用临时文件加原子 rename 写入。每个阶段只有在产物存在、校验通过且 manifest 完整时才能标记 completed。

### 14.3 Checkpoint 关联

每个 proposer checkpoint 保存：

- iteration state 快照，或
- state 内容哈希和规范路径引用。

恢复时必须校验：

- checkpoint iteration 与 state 一致；
- skills/rubrics 内容哈希一致；
- proposer/solver 模型引用一致；
- 配置中的关键语义参数未发生未声明变化。

## 15. 失败与恢复语义

失败策略：

- 单个候选的 verifier 调用在重试耗尽后标记 `verify_error`，保存失败详情并继续后续候选；
- Meta/Judge、rubric、分析或 updater 调用在重试耗尽后，中止本轮；
- 任一结构化输出解析或 schema 校验最终失败，中止本轮；
- verifier 故障与两条件不成立的 `verify_failed` 必须区分；
- 不使用旧 skills/rubrics 假装本轮更新成功；
- 已落盘的原始输出和失败信息必须保留；
- 修复外部服务或配置后，可从失败阶段幂等恢复。

## 16. 主要配置与默认值

```yaml
iteration:
  candidate_count_per_document: 5
  solver_train_ratio: 0.9
  split_seed: ${global_seed}

verify:
  enabled: true
  solver_samples: 3
  allow_search: false

proposer_reward:
  format_weight: 0.5
  difficulty_weight: 1.0
  rubric_weight: 0.5

dynamic_state:
  max_skills: 12
  max_rubrics: 12
  max_retries: 3
```

所有默认值必须由一个主配置来源定义，shell 脚本只覆盖，不复制另一套默认值。

## 17. 预计代码落点

### 修改

- `process_train.py`
  - 增加稳定 `doc_id` 和结构化 source document 字段。
- `verl/prompts.py`
  - 增加 skills 注入、verifier、rubric、judge、trajectory analysis 和 updater prompts。
- `verl/custom_reward/reward_function.py`
  - 接入 rubric reward，拆分并记录 reward 分量。
- `verl/custom_reward/reward_rollout.py`
  - 抽取可复用的批量 rollout client，支持禁用工具的 verifier 模式和完整 trajectory 返回。
- `verl/trainer/main_generation.py`
  - 保存 5 个结构化候选、evidence、rubric 评价、排序、顺序 verify 和有效数据。
- `config/search_multiturn_grpo.yaml`
  - 接入新增配置。
- checkpoint 相关模块
  - 保存和校验 iteration state 快照/引用。

### 新增

具体目录遵循现有模块布局，实施前以最小改动确定最终文件名：

- iteration orchestrator/CLI；
- Meta/Judge OpenAI-compatible client；
- Pydantic 或等价 schema；
- evidence extractor；
- problem verifier；
- stable group splitter；
- keepout evaluator；
- trajectory analyzer；
- skills/rubrics updater；
- iteration state store；
- 对应单元和最小集成测试。

## 18. 测试与验收

### 18.1 单元测试

1. proposer trajectory 可正确提取 seed 文档、query 和 tool responses。
2. 缺失或损坏的 tool response 不会产生不完整 Evidence Bundle。
3. rubric 1–5 分归一化和 rank score 正确。
4. 候选按稳定顺序依次 verify，并在首个通过后停止。
5. verify 的两个条件缺任一项均失败。
6. verifier 严格禁用 search tool，且两组各固定生成 3 个样本。
7. stable `doc_id` 在重跑和数据重排后不变。
8. group split 不会让同一 `doc_id` 同时进入 train/keepout。
9. skills 注入顺序稳定，训练和 gen data 使用同一实现。
10. proposer reward 三个分量及总值计算正确。
11. skills/rubrics schema、唯一 ID、12 项上限和长度限制生效。
12. skills diff 与 rubrics diff 正确。
13. 每条 keepout trajectory 只产生一个二值 correctness。
14. trajectory summary 和分块全局报告可回溯到原记录。
15. skills 更新先于 rubrics，rubrics updater 收到 `skills_{t+1}`。
16. 原子 state 写入和内容哈希校验正确。

### 18.2 最小集成测试

1. 使用 mock solver/meta model 跑通完整单轮。
2. 5 个候选中前两个失败、第三个通过时只保留第三个。
3. verify 通过数据按 doc_id 形成互斥的 90/10 train/keepout。
4. solver 训练只读取 train manifest。
5. 新 solver 以 question-only + search 模式产生 keepout trajectory。
6. 全部 trajectory 被分析并汇总，随后依序更新 skills/rubrics。
7. 单题 verifier 调用重试耗尽时继续后续题；其他模型调用重试耗尽时 iteration 标记 failed。
8. 从 verify、solver 训练、分析和 updater 等失败点分别恢复时，不重复已完成且校验通过的阶段。

### 18.3 回归测试

- 原有 proposer format/difficulty reward 在 rubric weight 设为 0 时保持一致。
- verify 关闭时，仅用于开发兼容测试；正式默认值必须为开启。
- 原有 solver GRPO reward 和 search tool 协议不变。
- 旧数据缺少结构化 evidence 时给出明确迁移错误，不默默从不可靠文本继续训练。

## 19. 日志与观测

每轮至少记录：

- 生成候选数、每个排序位置被验证次数、最终通过率；
- verify 两种条件失败原因和 invocation error 分布；
- format、difficulty、rubric 和总 proposer reward 分布；
- train/keepout 的文档数、问题数和 hop 分布；
- keepout 总体 accuracy；
- trajectory 根因类别频次；
- skills/rubrics 前后状态和 diff；
- Meta/Judge 调用次数、token、延迟、重试和失败；
- 各阶段耗时、状态和 artifact 路径。

## 20. 完成标准

以下条件全部满足才视为完成：

1. 单轮严格遵循第四节时序。
2. verify 对“完整 Evidence Bundle + question”和“仅 question”各采样 3 次，并按两个 EM 条件直接判定。
3. solver train 与 keepout 按 `doc_id` 完全隔离。
4. 新 solver 的 keepout 结果可逐题追踪到完整 trajectory、judge 和 summary。
5. 所有 trajectories 均参与分层分析。
6. proposer 训练实际使用当前 skills 和 rubrics reward。
7. skills 先更新，rubrics 使用新 skills 后更新。
8. iteration state、checkpoint 引用和全部产物可恢复、可审计。
9. 所有新增测试通过，现有相关测试无回归。
10. verifier 单题故障被隔离并记录；其他模型调用最终失败时，本轮明确失败，不产生伪完成状态。
