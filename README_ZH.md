# Cupid：不确定性感知流场预测与因果在线校准

**NeurIPS 2026 RealPDE Competition · Track 2（LTTTA）**

Yaoyu Chang · Wenhao Liu · Hao Shi

[English](README.md) · [官方最终榜](https://realpdecompetition.github.io/results.html) · [完整方法](submission/METHOD.md) · [复现说明](docs/reproduction.md)

![Cupid 方法概览](docs/assets/cupid-overview.svg)

真实流场预测同时面临仿真到实测的差异、未来预测的不确定性，以及部署过程中的分布变化。
Cupid 将离线训练的流场预测网络与轻量在线校准结合：**神经网络保持冻结，已经揭示的历史目标用于更新统计量和区域线性读出。**

本仓库展示 Cupid 队的 Track2 最终提交 **948801（L33）**，并保留完整的从头训练链路。

## 已验证结果

| 证据类型 | 结果 |
|---|---|
| 官方最终私有测试集 | **并列第 9，总分 82.6104，标准差 0.0653** |
| 历史开发榜提交 | 948801，总分 82.186978 |
| 本地 Docker Engine + A800 复训 | 31 阶段全部完成，4 小时 7 分 24 秒 |
| 本地公开数据配对回放 | 716 尾窗质量均值：复训 83.271780，封存版 83.277454，差 −0.005674 |

最终排名来自[官方结果页](https://realpdecompetition.github.io/results.html)。官方结果汇总了三次从头训练、每次五次评测。
本地公开回放与训练数据重叠，其质量均值不含官方 Time 分数，不能替代或直接对比官方隐藏集总分。

## 方法结构

1. **完整离线训练**：三个 Sim 教师、学生模型、Real 点预测、形状分支、均值校准及不确定性学习，依次构成固定的 31 阶段链路。
2. **因果在线适应**：只使用已经揭示的历史目标，更新区域均值、波动幅度、残差分布以及 32 维特征的正则线性读出；每条轨迹重置状态。
3. **高效推理**：权重混合 INT8/FP16 存储、FP16 推理、Torch 编译、复用缓冲区和 CUDA Graph。在线阶段没有神经网络梯度更新。

输入为 20 帧，输出为未来 20 帧及上下界，评测网格为 32×64。Real 数据的压力通道为零。

## 从哪里开始

- [方法说明与因果边界](docs/method.md)
- [公开数据来源与目录要求](docs/data.md)
- [Docker 构建、从头训练与 Final API](docs/reproduction.md)
- [结果、证据及局限](docs/results.md)
- [核心代码导航](docs/code-map.md)

`submission/` 的 1,835 个文件逐字节保留原 Final 源码包。运行 `python scripts/verify_source.py` 可检查完整性；完整训练在 Linux + CUDA 环境中执行。
从源码构建即可复现，不依赖访问历史私有 Docker Hub 仓库。

原包 README 是提交时点的记录；10 月 10 日完成的 Docker Engine 全流程证据在外层文档中补充说明。
仓库中的历史依赖目录不表示另一个受支持的 Track1 发布，当前训练入口只支持 Track2。

## 团队与来源

Cupid：Yaoyu Chang（队长）、Wenhao Liu、Hao Shi。
这是比赛方案发布；不将比赛结果或本地诊断包装为已发表论文。
感谢 RealPDE 与 RealPDEBench 团队提供数据、接口与评测环境。
团队原创代码与文档采用 [MIT 许可](LICENSE)，第三方许可保留在原包中，复用条件见[许可说明](docs/licensing.md)。
