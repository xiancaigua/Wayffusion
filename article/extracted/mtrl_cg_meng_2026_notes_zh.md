# MTRL-CG 论文整理

题目：MTRL-CG: Multi-Task Reinforcement Learning Method with Spectral Clustering-Based Task Grouping

作者：Wenjia Meng, Teng Zhang, Haoliang Sun, Yilong Yin

核心问题：多任务强化学习中，不相关或冲突任务共享同一策略/价值网络时会产生 negative interference。某个任务有利的更新可能损害其他任务，导致整体成功率下降。

核心方法：MTRL-CG 先估计任务间亲和关系，再用谱聚类把相关任务分到同组，把冲突任务分开，最后对每个任务组训练独立的 SAC learner。

算法步骤：

1. 在所有任务上训练一个共享 SAC 模型。
2. 对任务 `Ti` 做一次 critic 的单任务梯度更新，观察这个更新对任务 `Tj` 的 Q 值影响，得到有向亲和 `Wij`。
3. 累积多个估计时刻的亲和矩阵，并与转置平均得到对称矩阵。
4. 将矩阵拆成正亲和 `W+` 和负亲和 `W-`。
5. 构造 signed Laplacian，并求解广义特征值问题。
6. 使用最小特征值对应的前 `K` 个特征向量作为任务谱嵌入。
7. 对任务嵌入运行 k-means++，得到任务组。
8. 每个任务组单独训练一个 SAC policy/Q-function。

实验：论文在 Meta-World MT10 和 MT50 上评估，包含 fixed 与 mixed 两种设置。对比方法包括 MT-SAC、CARE、CMTA、PaCo，以及这些方法加 MTRL-CG 后的结果。指标为 success rate，三随机种子平均。

主要结论：

- MTRL-CG 通常提升 max/final success rate。
- 在 MT10-Fixed 上，MT-SAC+MTRL-CG 比 MT-SAC 更快达到相近成功率。
- CARE+MTRL-CG 在 MT10-Fixed 上提升明显。
- 分组数消融显示 `K=3` 在 MT10 上整体较好。
- affinity matrix 和 t-SNE 可视化显示分组与任务关系有一定一致性。

复现风险：

- PDF 没有完整给出所有训练预算和底层超参数。
- 官方代码是多个 backbone 的修改版集合，工程复杂度较高。
- 完全复现实验表格需要 Meta-World 长时间训练和多 seed。
- 方法级复现可先实现 affinity estimation + signed spectral clustering + group-wise SAC。

