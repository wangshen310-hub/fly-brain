# 第二阶段：可选连续双窗训练与严格选模（2026-10-26）

## 本次完成什么

基于第一阶段提交 `2b9d0fe3944121efad5025be52066c8f77260152`，在同一独立恢复分支新增
**显式 opt-in** 协议 `continuous-two-window-v1`。默认仍为 `legacy-cold-v1`。
这是训练协议实现和小型 CPU 验证，**不是一次完整脑训练，也没有新学习成绩**。

第一阶段的 Windows CI 为 282 项通过；本阶段新增 22 项回归，本地共 132 项相关单元测试
及独立 6 项导入器测试通过。详见 `unit.xml`、`importer.xml`。远程 CI 必须另按新提交核验。

## 为什么需要这一改动

第一阶段虽然能报告连续两窗，但训练损失和选模仍只看冷窗口。这样一个冷窗完美、热窗全部
右转的候选仍可能按冷分数被选中。历史存档已显示冷/热偏置不能混为一谈。
这里修正的是**显式启用的新协议**的目标与选模不一致，不宣称解决生物动力学或优化瓶颈。

| 规则 | 默认 `legacy-cold-v1` | 可选 `continuous-two-window-v1` |
|---|---|---|
| 每个训练 batch | 原单窗 | 同一 RGB 编码的连续两个 100 ms |
| 窗口边界 | 无第二训练窗 | 原样传递完整状态和延迟队列，不 detach/reset |
| 监督目标 | 原 CE + 无输出项 | 两窗原 CE 与无输出项分别平均 |
| 参数正则 | 原正则一次 | 相同正则一次，不重复加倍 |
| 开发验证选模 | 原冷窗 BA | 两窗各 BA ≥80%，全部六项召回 ≥75%，否则无选模资格 |
| 合格候选之间 | 冷 BA 更高 | 依次最大化最差窗口 BA、最低类别召回、两窗平均 BA |

指标必须有限、三类都有支持、BA 与召回均值一致；不相信报告中缓存的 `gate.passed`。
仅有一个窗口、错位时钟、缺类、NaN/Inf 或任一不达标类别均不能获得资格。
数值门使用 `1e-12` 浮点表示容差，避免数学上 80%（例如 75%/75%/90%）被浮点误差拒绝；
不是按百分比粗略四舍五入，低于门槛的真实样本仍被拒绝。

## 状态与梯度的验证

新增测试直接运行原 `SpikingConnectome` 与非零增益的 `PlasticAfferentBrain` 7 神经元夹具：

- 仅从第二窗 CE 反传，第一窗最终电压 **和延迟队列** 都得到有限、非零梯度；现有可训练参数也有梯度。
- 第二次积分收到同一输入对象、第一次返回的完整状态对象；不经过 `act`、无梯度评估器或固定外围重放。
- 故意 detach 的负对照切断跨窗梯度；故意重置状态的负对照被时钟检查拒绝。
- 默认单窗损失和所有可训练参数梯度与原公式逐值完全一致。
- 两窗损失取均值，原正则仍只在训练循环中加一次。

原 `_Spike` surrogate derivative 和其既有离散 reset 处理未修改。没有用代理分类器代替脑，
也没有把五细胞固定边界回放接入训练器。

**这是相同静态画面的 200 ms 训练。** 各 batch 之间仍重置；没有实现连续移动游戏序列、
跨 batch 梯度或奖励学习。开发验证参与选模，不能当作未使用的封存泛化测试。

## 没有合格候选时

训练仍保存最后已提交的参数、Adam 状态、RNG、协议/源码身份和诊断指标。
但 `best_checkpoint=null`、`selection_state=no_eligible_checkpoint`，并在访问任何
`test_rgb` / `test_labels` 前退出，不评估封存测试、不把失败候选称为最佳已学会模型。
恢复测试实际模拟了提交后进程中断，验证可从相同协议恢复。

若初始模型本身满足门槛，且没有更优合格更新，则明确记录
`selected_checkpoint_origin=unchanged_initial` 和 `qualified_updated_checkpoint_selected=false`。
为兼容旧读取器保留 `trained` 字段，但明确它只是 `selected_checkpoint_evaluation` 的历史别名；
解释文本不会把未改动的初始模型称为训练成功。

协议名称进入存档配置；恢复前检查协议、配置、源码和数据身份，**包括已完成任务的快捷返回路径**。
跨协议、跨源码续训被拒绝。额外恢复测试检查了合格选模 tuple、初始选择身份、Adam 和 RNG。
这些恢复测试发生在最后一次更新已经提交之后；它们验证存档状态恢复，
**没有证明恢复后继续执行多次新更新与不中断训练逐值等价**。

## 使用与成本边界

本次只修改 `scripts/train_teacher_spiking.py` 及测试/报告。
该 CLI 仍使用原来的 `SpikingConnectome` 起始路径；没有把第 60 步突触模型或未恢复的第 112 步
存档接入此训练器。虽然共享目标函数在两个类的小型夹具上测试，
`probe_afferent_teacher.py` 和 `evaluate_afferent_frozen.py` 的协议没有被改写。
神经核心、光学输入、已有连接、固定解码器、默认模型和旧存档保持不变。

在未来**已核验完整图、兼容身份与资源预算**的独立实验中，可在原训练命令末尾选择：

```text
--protocol legacy-cold-v1
# 或，显式启用本轮新增协议：
--protocol continuous-two-window-v1 --window-ms 100
```

不要在当前 CPU 环境据此启动完整脑训练。200 ms 未截断计算图约需双倍积分工作，
自动微分峰值内存可能超过原来的两倍；尚未实测完整图成本。本次没有 GPU、付费算力、
模型下载或完整脑训练/推理，也没有放宽原 BA/召回门槛。

本地复现（使用第一阶段独立环境及依赖，线程限制 2）：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest tests/test_teacher_continuous_protocol.py \
  tests/test_teacher_training_safety.py tests/test_teacher_spiking.py \
  tests/test_teacher_afferent.py tests/test_teacher_game.py tests/test_teacher_data.py \
  tests/test_optical_aperture.py tests/test_game.py tests/test_decoder.py -q
python -m pytest experiments/continuation-2026-10-18/test_import_artifacts.py -q
```

测试中的图、RGB、标签、临时 NPZ 和检查点均为显式微型合成夹具；不上传这些临时数据。
只提交代码、测试和结果报告。无全图效果对比，无“已经学会”的结论。

下一步仍需在获准且身份兼容的全图环境中量测真实资源成本、连续训练与冻结评估，
再做独立种子、匹配乱标签/未训练对照、冻结方案后的封存图像与普通闭环游戏验证。
