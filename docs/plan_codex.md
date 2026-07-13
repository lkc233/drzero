请修改这个仓库中 proposer 和 solver 协同进化的实现，新增“问题 verify 机制”以及 proposer 侧动态 skills/rubrics 机制。以下流程、数据结构、默认值、失败行为和验收标准均为固定要求。

开始编码前必须完成以下工作：
1. 阅读 README、全部训练配置、训练入口脚本和核心 iteration 代码，画出当前单轮流程。
2. 定位 generate data 的实现，确认文档、合成问题、标准答案、proposer metadata 和 solver 训练样本的数据结构与流转路径。
3. 定位 proposer/solver 的 prompt、模型调用、状态保存、训练和更新逻辑。
4. 定位 evaluation、trajectory、logging、checkpoint 和重试机制。
5. 输出一次实现计划，明确修改文件、数据结构和各机制的接入位置；等待确认后再编码。

## 一、固定 iteration 流程

设第 `t` 轮开始时的 proposer 状态为 `skills_t` 和 `rubrics_t`，solver 训练开始时
加载的模型为 `solver_t`。本轮末更新得到的 `skills_{t+1}`、`rubrics_{t+1}` 和
`solver_{t+1}` 只供下一轮使用，不得反过来改变本轮已经完成的生成、评价或 rollout
结果。

每个 iteration 必须严格按以下数据流执行，不得交换步骤：

1. **加载轮初状态**：从 checkpoint 读取 `skills_t` 和 `rubrics_t`。旧 checkpoint
   缺少对应字段时，分别使用第三节和第四节规定的初始值。
2. **生成候选问题**：将完整的 `skills_t` 注入 proposer prompt，生成本轮候选集合
   `generated_t`。此时不得使用或预先生成 `skills_{t+1}`。
3. **逐题 verify**：对 `generated_t` 中每个候选问题执行 problem verify，保存每题
   verify 输出，并将集合拆分为 `passed_t` 和 `failed_t`。
4. **过滤数据**：从本轮 generated data 中删除 `failed_t`。此后本轮的 rubrics
   评价、solver 训练和 solver 评估只能接收 `passed_t`，不得接收未验证或验证失败的
   样本。
5. **使用轮初 rubrics 评价问题**：用完整的 `rubrics_t` 评价 `passed_t` 中的每个
   问题，得到 `rubric_evaluations_t`。本步骤使用的是更新前的 rubrics。
6. **训练 solver 并收集 rollout 反馈**：从 `solver_t` 开始，使用 `passed_t`
   执行仓库现有的 solver 训练流程。仓库的 GRPO 训练在每个 batch 中先用当前 actor
   生成 rollout，再计算 correctness，最后更新 actor；因此一轮训练中的 trajectories
   来自不同 global step 的 solver，不代表同一个固定模型。必须为每条 trajectory
   保存问题标识、生成时的 global step 或 policy version、完整 trajectory 和二值
   correctness。训练结束后保存得到的模型为 `solver_{t+1}`。
   
   对训练期间收集到的反馈只计算以下统计：
   - 每题 `rollout_correctness`：该问题所有训练 rollouts 的二值 correctness 均值；
   - `training_rollout_success_rate`：本轮正确训练 rollouts 数除以本轮训练
     rollouts 总数；总数为 0 时记为 `0.0`。
   
   以上两项是跨多个 solver 版本的训练 rollout 统计，不得命名或解释为
   `solver_t`、`solver_{t+1}` 的“总体正确率”或泛化准确率。本轮结果记为
   `solver_rollout_results_t` 和 `trajectories_t`。若仓库现有流程在固定 validation
   set 上评估 `solver_{t+1}`，其结果单独保存为 `solver_validation_metrics_{t+1}`，
   不得与训练 rollout 统计混合。
7. **确定 skills 更新样本**：按第三节规定的固定抽样规则，从 `trajectories_t` 中
   得到 `sampled_trajectories_t`。抽样只影响 skills updater 的输入，不得删除或覆盖
   已保存的完整 trajectories。
8. **先更新 skills**：以 `skills_t`、`solver_rollout_results_t`、
   `sampled_trajectories_t`、`rubric_evaluations_t` 和本轮完整 verify 统计为输入，
   调用 skill updater，得到 `skills_{t+1}`。更新失败时令
   `skills_{t+1} = skills_t`。
9. **再更新 rubrics**：skills 更新结束后，以 `rubrics_t`、刚得到的
   `skills_{t+1}`、`rubric_evaluations_t`、`solver_rollout_results_t` 和本轮完整
   verify 统计为输入，调用 rubric updater，得到 `rubrics_{t+1}`。更新失败时令
   `rubrics_{t+1} = rubrics_t`。禁止使用 `skills_t` 代替 `skills_{t+1}`。
10. **持久化并记录**：将 `skills_{t+1}`、`rubrics_{t+1}`、每题 verify 输出、
    verify 统计、`rubric_evaluations_t`、完整 solver trajectories、两次 updater
    的原始输出、成功状态、错误和 diff 一并写入现有 checkpoint/state；随后写入本轮
    日志和 metrics。

若 `passed_t` 为空，仍须记录 verify 结果与统计；不得把失败样本送入 solver。
solver、抽样、skills 更新和 rubrics 更新如何处理空输入，应复用仓库现有空批次行为；
若仓库没有对应行为，则跳过这些模型调用，保持 `skills_{t+1} = skills_t`、
`rubrics_{t+1} = rubrics_t`，记录跳过原因，并正常保存 checkpoint。

## 二、问题 verify 机制

### 输入与输出

每个 verify 输入固定包含：
- `document`：合成问题使用的原始文档或上下文；
- `question`：合成问题；
- `reference_answer`：标准答案；
- `proposer_reasoning`：proposer reasoning；没有内容时保存为空字符串；
- `proposer_metadata`：proposer metadata；没有内容时保存为空对象。

每个 verify 输出固定为：
```json
{
  "model_answer": "string",
  "passed": true,
  "reason": "string"
}
```

### 执行规则

1. 使用仓库现有 model client 调用一次 verifier 模型，输入原始文档和问题，生成 `model_answer`。
2. 仓库存在 judge/evaluator 时，使用该实现比较 `model_answer` 与 `reference_answer`；仓库不存在 judge/evaluator 时，实现统一的规范化精确匹配：Unicode NFKC 归一化、转小写、去除首尾空白、连续空白压缩为单个空格后比较。
3. judge 判定正确时设置 `passed=true`；判定错误、模型调用最终失败、输出解析最终失败时均设置 `passed=false`。
4. `passed=true` 的样本保留；`passed=false` 的样本从本轮 generated data 中删除。
5. verifier 复用现有并发、批处理、缓存、重试和模型调用封装。仓库没有重试配置时固定最多调用 3 次，等待间隔依次为 1 秒和 2 秒。
6. 新增配置 `enable_problem_verify: true`。默认值固定为 `true`；仅当用户显式设为 `false` 时跳过 verify，并将全部候选问题视为通过。

### 统计

每轮固定记录：
- `generated_total`；
- `verify_passed`；
- `verify_failed`；
- `verify_pass_rate`，定义为 `verify_passed / generated_total`，当总数为 0 时记为 `0.0`；
- 每个失败样本的 `question`、`model_answer` 和 `reason`。

## 三、proposer 动态 skills 机制

### 数据结构与初始值

skills 使用 JSON 列表保存，每项结构固定为：
```json
{
  "id": "skill-N",
  "instruction": "string",
  "evidence": "string"
}
```

旧配置和旧 checkpoint 不含 skills 时，初始值固定为：
```json
[
  {
    "id": "skill-1",
    "instruction": "Generate questions that require evidence from the provided document and have one unambiguous answer.",
    "evidence": "default"
  }
]
```

### Prompt 注入

每次 proposer 生成问题时，将 skills 按列表顺序序列化为 JSON，注入固定的 `CURRENT_SKILLS` prompt 区块。禁止在未注入 skills 的情况下调用 proposer。

### 更新输入与规则

每轮 rubrics 评价、solver 训练和 rollout 反馈收集完成后更新一次 skills。更新输入固定包含：
1. 更新前的完整 skills；
2. 本轮每题 `rollout_correctness` 和 `training_rollout_success_rate`，并明确标注它们是
   训练期间跨 policy version 收集的 rollout 统计；
3. 固定抽样的 solver trajectories；
4. 每个问题的 rubrics 评价结果；
5. 本轮完整 verify 统计和失败样本。

trajectory 抽样数量由 `skill_update_trajectory_sample_size` 控制，默认固定为 `8`。使用项目全局随机种子，从本轮 trajectories 中无放回抽样；不足 8 条时全部使用。相同输入和随机种子必须得到相同样本。

skill updater 必须返回完整的新 skills 列表，而不是增量文本。updater 根据输入证据决定保留、改写、删除和新增的 skill；返回结果必须满足上述 schema，`id` 必须唯一。解析或模型调用在重试 3 次后仍失败时，保留更新前的 skills，并记录错误；iteration 继续执行。

每次更新固定保存：
- `skills_before`；
- `skills_after`；
- 以 `id` 为键计算的 added、removed、modified diff；
- updater 原始输出；
- 更新成功状态和错误信息。

## 四、proposer 动态 rubrics 机制

### 数据结构与初始值

rubrics 使用 JSON 列表保存，每项结构固定为：
```json
{
  "id": "rubric-N",
  "name": "string",
  "description": "string",
  "score_min": 1,
  "score_max": 5
}
```

旧配置和旧 checkpoint 不含 rubrics 时，固定初始化五项 rubric：
1. `difficulty`：问题是否需要非平凡推理；
2. `answerability`：文档是否提供充分证据且存在唯一答案；
3. `document_dependency`：问题是否必须依赖给定文档回答；
4. `solver_discrimination`：问题是否能区分 solver 能力；
5. `ambiguity`：问题、证据和答案是否无歧义，其中 5 分表示无歧义。

每项 `score_min=1`、`score_max=5`。

### 评价

对每个 verify 通过的问题，使用当前全部 rubrics 评价一次。每项评价固定输出：
```json
{
  "rubric_id": "string",
  "score": 1,
  "reason": "string"
}
```

score 必须是 1 到 5 的整数。解析或模型调用在重试 3 次后仍失败时，该问题所有 rubric score 固定记为 1，`reason` 写入错误信息，iteration 继续执行。

### 更新输入与规则

skills 更新完成后，rubrics 每轮更新一次。更新输入固定包含：
1. 更新前的完整 rubrics；
2. 更新后的完整 skills；
3. 本轮所有 rubrics 评价结果；
4. 本轮每题 `rollout_correctness` 和 `training_rollout_success_rate`，并明确标注它们是
   训练期间跨 policy version 收集的 rollout 统计；
5. 本轮完整 verify 统计和失败样本。

rubrics updater 必须返回完整的新 rubrics 列表。返回结果必须满足上述 schema，`id` 必须唯一，score 范围固定为 1 到 5。解析或模型调用在重试 3 次后仍失败时，保留更新前的 rubrics，并记录错误；iteration 继续执行。

每次更新固定保存：
- `rubrics_before`；
- `rubrics_after`；
- 以 `id` 为键计算的 added、removed、modified diff；
- updater 原始输出；
- 更新成功状态和错误信息。

## 五、状态、配置与兼容性

新增配置及默认值固定为：
```yaml
enable_problem_verify: true
skill_update_trajectory_sample_size: 8
dynamic_state_update_max_retries: 3
```

skills、rubrics、评价结果、verify 统计和更新记录写入项目现有 checkpoint/state。恢复训练时以 checkpoint 中的 skills/rubrics 为准；旧 checkpoint 缺少字段时使用本文初始值。新增字段不得改变旧字段含义，也不得阻止旧 checkpoint 加载。

## 六、工程约束

1. 必须复用仓库现有 model client、prompt 模板、配置、日志、checkpoint/state、并发、缓存和重试机制；对应机制不存在时，仅实现本文明确规定的最小逻辑。
2. 不做无关重构，不改变与本需求无关的行为。
3. 保持现有代码风格、命名和目录结构；新增文件放入对应 proposer、evaluation 或 state 模块所在目录。
4. 所有模型结构化输出均执行 schema 校验。
5. verify 单样本失败只过滤该样本；skills/rubrics 评价或更新失败按本文规则降级，均不终止 iteration。
6. 日志不得省略失败原因、更新前后状态和 diff。

## 七、测试与验收

必须新增并通过以下单元测试或最小集成测试：
1. verify 正确答案通过、错误答案失败。
2. verifier 模型调用失败和解析失败均过滤样本。
3. `enable_problem_verify=false` 时所有候选问题直接通过。
4. verify 统计和 `generated_total=0` 时的通过率正确。
5. proposer prompt 包含按顺序序列化的 `CURRENT_SKILLS`。
6. trajectory 在相同 seed 下抽样结果一致，不足样本数时使用全部 trajectories。
7. 每条 solver trajectory 保存生成时的 global step 或 policy version 及二值
   correctness；每题 `rollout_correctness` 和全轮 `training_rollout_success_rate`
   计算正确，且不被记录为固定 solver 的总体正确率。
8. skills 更新输入包含 rollout correctness 统计、trajectories、rubrics 评价和
   verify 统计。
9. skills 更新成功时保存新状态与 diff，失败时保留旧状态。
10. rubrics 对每个通过 verify 的问题输出五项初始评分。
11. rubrics 更新使用更新后的 skills，成功时保存新状态与 diff，失败时保留旧状态。
12. 新 checkpoint 能保存并恢复 skills/rubrics。
13. 旧 checkpoint 缺少新增字段时加载固定初始值。
14. 完整单轮流程严格遵循本文第一节规定的顺序。

运行仓库现有完整测试命令并修复本次改动造成的失败。测试因确定的外部依赖缺失而无法运行时，记录原始命令、完整错误和已执行的替代测试，不得将“未运行”表述为“通过”。

## 八、完成后输出

实现完成后固定输出：
1. 修改文件列表；
2. 新增配置项及默认值；
3. 新机制在 iteration 中的接入顺序；
4. 实际运行的测试命令及结果；
5. 已知限制，若无则明确写“无已知限制”。