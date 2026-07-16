# 方法

这个仓库实现了proposer和solver的协同进化，请你阅读并理解它，现在我希望对它做如下的修改：

1. 增加对于问题的 verify 机制：在每个 iteration 的 generate data 过程中，让 solver 对“合成问题所使用的全部文档 + 问题”和“仅问题”各回答 3 次；前者至少答对 1 次且后者 3 次全错时才保留问题，否则抛弃
2. 对于proposer，增加skills和rubrics，两者在训练的过程中都会动态变化：
    1. skills：用于指导问题的合成，模型会根据solver做出问题的正确率、sample一些solver解决问题的trajectory、现有rubrics对于问题的评价来更新skills；
    2. rubrics：用于评价问题，模型根据当前skills、对于本轮合成问题的评价、solver解决问题的正确率来更新rubrics

请你完成这些修改
