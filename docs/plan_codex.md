请修改这个仓库中 proposer 和 solver 协同进化的实现，新增“问题 verify 机制”以及 proposer 侧动态 skills/rubrics 机制。以下流程、数据结构、默认值、失败行为和验收标准均为固定要求。

开始编码前必须完成以下工作：
1. 阅读 README、全部训练配置、训练入口脚本和核心 iteration 代码，画出当前单轮流程。
2. 定位 generate data 的实现，确认文档、合成问题、标准答案、proposer metadata 和 solver 训练样本的数据结构与流转路径。
3. 定位 proposer/solver 的 prompt、模型调用、状态保存、训练和更新逻辑。
4. 定位 evaluation、trajectory、logging、checkpoint 和重试机制。
5. 输出一次实现计划，明确修改文件、数据结构和各机制的接入位置；等待确认后再编码。

## 一、固定 iteration 流程

每个 iteration 严格按以下顺序执行：
1. 从 checkpoint 加载 proposer skills 和 rubrics；无对应字段时加载本文规定的初始值。
2. 将当前 skills 注入 proposer prompt，生成候选问题。
3. 对每个候选问题执行 problem verify。
4. 丢弃 verify 未通过的问题，仅将通过的问题送入 solver 训练和评估。
5. 使用当前 rubrics 评价所有 verify 通过的问题。
6. 运行 solver，保存 correctness 和完整 trajectories。
7. 使用固定抽样规则选取 solver trajectories。
8. 先更新 skills，再使用更新后的 skills 更新 rubrics。
9. 保存 skills、rubrics、评价结果、更新记录和 verify 统计到 checkpoint/state。
10. 写入本轮日志和 metrics。

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

每轮 rubrics 评价和 solver 评估完成后更新一次 skills。更新输入固定包含：
1. 更新前的完整 skills；
2. 本轮 solver correctness 和总体正确率；
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
4. 本轮 solver correctness 和总体正确率；
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
7. skills 更新输入包含 correctness、trajectories、rubrics 评价和 verify 统计。
8. skills 更新成功时保存新状态与 diff，失败时保留旧状态。
9. rubrics 对每个通过 verify 的问题输出五项初始评分。
10. rubrics 更新使用更新后的 skills，成功时保存新状态与 diff，失败时保留旧状态。
11. 新 checkpoint 能保存并恢复 skills/rubrics。
12. 旧 checkpoint 缺少新增字段时加载固定初始值。
13. 完整单轮流程严格遵循本文第一节规定的顺序。

运行仓库现有完整测试命令并修复本次改动造成的失败。测试因确定的外部依赖缺失而无法运行时，记录原始命令、完整错误和已执行的替代测试，不得将“未运行”表述为“通过”。

## 八、完成后输出

实现完成后固定输出：
1. 修改文件列表；
2. 新增配置项及默认值；
3. 新机制在 iteration 中的接入顺序；
4. 实际运行的测试命令及结果；
5. 已知限制，若无则明确写“无已知限制”。