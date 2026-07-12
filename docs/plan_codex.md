请修改这个仓库中 proposer 和 solver 协同进化的实现，新增“问题 verify 机制”以及 proposer 侧动态 skills/rubrics 机制。

请先不要急着改代码。先完成以下阅读和定位：
1. 阅读 README、配置文件、入口脚本和核心训练/迭代代码，理解当前每个 iteration 的流程。
2. 找到 generate data 的实现位置，确认合成问题、文档、答案/标签、solver 训练数据之间的数据结构和流转路径。
3. 找到 proposer 和 solver 当前的 prompt、状态保存、训练/更新逻辑。
4. 找到现有的 evaluation、trajectory、logging/checkpoint 机制，如果没有，请说明缺口并采用最小可行实现。
5. 先给出一个简短实现计划，说明准备改哪些文件、增加哪些数据结构、每个机制接入到哪个阶段。确认计划后再实现。

需要实现的功能：

一、问题 verify 机制

目标：
在每个 iteration 的 generate data 阶段，对 proposer 合成出来的问题做自动验证。只有能被模型基于原始文档正确回答的问题，才保留进入后续训练/评估；验证失败的问题要丢弃，并记录原因。

具体要求：
1. verify 输入应至少包含：
   - 合成问题所使用的文档/上下文；
   - 合成问题；
   - 标准答案或可判定答案；
   - 必要时包含 proposer 生成的 reasoning/metadata。
2. verify 过程调用模型回答该问题，并用现有 judge/evaluator 或新增最小判定逻辑判断回答是否正确。
3. 如果 verify 通过，保留该问题；否则从本轮 generated data 中过滤掉。
4. 记录每轮 verify 统计信息，包括：
   - 生成问题总数；
   - verify 通过数；
   - verify 失败数；
   - 失败样例或失败原因；
   - 通过率。
5. verify 逻辑应可配置开关，例如 `enable_problem_verify`，默认开启或按项目现有配置风格决定。
6. 如果项目已有并发、批处理、缓存、重试、模型调用封装，请复用现有机制，不要另起一套。

二、proposer 动态 skills 机制

目标：
为 proposer 增加一组动态维护的 skills，用于指导后续问题合成。skills 会随着训练迭代更新。

skills 的作用：
1. 在 proposer 生成问题的 prompt 中注入当前 skills。
2. skills 应描述“应该如何合成更有价值/更难/更适合训练 solver 的问题”。
3. skills 要能跨 iteration 保存和加载，加入 checkpoint 或项目已有状态管理。

skills 的更新依据：
每轮 iteration 后，根据以下信息更新 skills：
1. solver 对本轮问题的正确率；
2. sampled solver trajectories，即抽样若干 solver 解题过程；
3. 当前 rubrics 对本轮问题的评价；
4. 可选：verify 通过/失败统计。

更新要求：
1. 新增一个清晰的 skill update prompt 或 updater 逻辑。
2. 更新时保留有用旧 skills，删除或改写效果差的 skills，必要时新增 skills。
3. skills 最好采用结构化格式保存，例如 list/dict/json/yaml，避免只存一整段不可解析文本。
4. 记录每次 skills 更新前后的 diff 或摘要。

三、proposer 动态 rubrics 机制

目标：
为 proposer 增加一组动态 rubrics，用于评价合成问题的质量，并参与后续 skills 更新和问题筛选/分析。

rubrics 的作用：
1. 对本轮合成问题进行评价，例如难度、可回答性、文档依赖程度、区分 solver 能力的价值、是否有歧义等。
2. rubrics 的评价结果要能被 skills 更新过程使用。
3. rubrics 本身也要随着训练迭代动态更新。

rubrics 的更新依据：
每轮 iteration 后，根据以下信息更新 rubrics：
1. 当前 skills；
2. 本轮 rubrics 对合成问题的评价结果；
3. solver 对本轮问题的正确率；
4. 可选：verify 统计和失败问题样例。

更新要求：
1. 新增 rubrics update prompt 或 updater 逻辑。
2. 更新后的 rubrics 应更好地区分“对 solver 训练有价值的问题”和“低质量问题”。
3. rubrics 应结构化保存，并能跨 iteration 加载。
4. 记录每次 rubrics 更新前后的 diff 或摘要。

工程约束：
1. 尽量复用仓库已有的 model client、prompt 模板、配置系统、日志系统、checkpoint/state 机制。
2. 不要做无关重构，不要改变与本需求无关的行为。
3. 保持现有代码风格、命名习惯和目录结构。
4. 如果必须新增文件，请放在与现有 proposer/solver/training 代码一致的位置。
5. 所有新增功能都应有合理默认配置，并兼容旧 checkpoint/旧配置。
6. 对模型输出做基本鲁棒解析，避免一次格式错误导致整个 iteration 崩溃。
7. 对 verify、skills update、rubrics update 的失败要有降级策略：记录错误并尽量不中断整个训练流程，除非项目现有风格是 fail-fast。

验收标准：
1. 每个 iteration 的 generate data 阶段会执行问题 verify，并过滤掉未通过的问题。
2. 日志或 metrics 中能看到 verify 通过率和失败统计。
3. proposer 生成问题时会使用当前 skills。
4. 每轮后会基于 solver correctness、sampled trajectories 和 rubrics evaluation 更新 skills。
5. 每轮后会基于当前 skills、rubrics evaluation 和 solver correctness 更新 rubrics。
6. skills 和 rubrics 能保存到 checkpoint/state，并在下一轮或恢复训练时加载。
7. 相关单元测试或最小集成测试覆盖：
   - verify 通过/失败过滤；
   - skills 注入 proposer prompt；
   - skills 更新；
   - rubrics 评价与更新；
   - checkpoint 兼容。
8. 运行项目现有测试命令，并修复失败。如果无法运行，请说明原因和已做的替代验证。

完成后请输出：
1. 修改了哪些文件；
2. 新增了哪些配置项；
3. 每个新机制接入到训练流程的哪个阶段；
4. 运行了哪些测试/验证；
5. 还有哪些已知限制或后续可改进点。