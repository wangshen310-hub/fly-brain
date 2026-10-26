# 2026-10-26：已发布源码恢复与 CPU 评估安全修复

## 结果先行

从 GitHub 提交 `e582e7405f3f4afc172539162060240221d18bd9` 恢复源码，在独立分支
`recovery/cpu-diagnosis-20261026` 完成小型 CPU 检查与修复。

**这不是第 112 步续训，不是新模型，也没有证明完整脑已经学会贪吃蛇。**
恢复的是公开的第 60 步模型所配套源码及 10 月 18 日独立实验代码；没有下载/加载模型权重、
完整图、训练数据或原始重放数据。仅使用源码、已有 JSON 报告及显式小型测试夹具。

| 本次实际执行 | 结果 | 证据 |
|---|---|---|
| 修改前 7 个相关测试模块 | 89 通过，0 失败/跳过 | `baseline-unit.xml` |
| 10 月 18 日合成导入器安全测试 | 6 通过，0 失败/跳过 | `baseline-importer.xml` |
| 修改后同 89 项 + 21 项新回归 | 110 通过，0 失败/跳过 | `final-unit.xml` |
| 对已有历史 JSON 重新评分 | 双窗 BA 66.67% / 33.33%，学习数值门不通过 | `diagnosis.json` |

共 116 项不同的相关测试通过；不是运行完整仓库测试套件。历史 JSON 重新评分不是新的脑推理。

## 已复现并修复的缺口

1. **缺失类别虚高分。** 原 `train_teacher_spiking.metrics` 在一个全部为“前进”的样本上返回
   BA=100%，尽管左/右类别支持数为零。现在缺失类别保留 `recall=null`，三类 BA 为 `null`，
   并显式标记覆盖不完整、拒绝通过门槛。验证集缺类不能用于选检查点。
2. **冷启动评分隐藏热窗口坍缩。** `train_teacher_spiking.evaluate` 原来只计算冷窗口。
   现在对每个独立 batch 连续运行相同输入的两窗，完整返回状态原样传入第二窗，不丢电压、
   电导、适应、不应期、延迟队列及时间。单独记录两窗的三类召回率、BA、无输出和动作分布。
   必须每窗 100 ms、每窗 BA ≥80%、全部六项召回率 ≥75% 才能通过**必要数值门**。
   无输出仍计为错误；不合并两窗掩盖失败；固定解码器及其原配置 margin 不变。
3. **后续非有限梯度可污染优化器。** 原训练器只检查首次梯度，后续默认裁剪可让 Inf 梯度
   经 Adam 把参数变成 NaN。现在每一步在优化器更新前拒绝非有限总范数，并记录裁剪前范数。
   回归验证在先做一次正常 Adam 更新后，Inf、NaN 及 float32 范数溢出均不改变参数或 Adam 记忆。
4. 指标输入不再将小数脉冲静默取整；拒绝错位、空集、非法标签、负数及非有限计数。

`baseline-defects.json` 是修改前合成复现；不是历史训练曾发生 NaN 的证据。
这些修复解决评估与更新防护漏洞，**没有解决神经学习瓶颈**。

## 关于真实学习瓶颈的证据边界

仓库已有 `docs/evidence/timecourse/summary.json` 表明，同一历史训练姿态的左/前/右图像：

| 窗口 | 左/前/右召回率 | BA | 预测分布（左/前/右） |
|---|---|---|---|
| 0–100 ms | 0% / 100% / 100% | 66.67% | 0 / 2 / 1 |
| 100–200 ms | 0% / 0% / 100% | 33.33% | 0 / 0 / 3 |

新脚本只校验既有动作与原脉冲解码一致并重新评分；来源 SHA-256 已写入 `diagnosis.json`。
这支持继续研究持续状态下的偏置，不能外推为所有姿态或所有参数下的结论。
此前另一路未恢复的第 112 步“其他分区梯度能量 >99.999%”没有在本次复现，
不得拿它作为当前分支的测量。分区裁剪、低学习率、C+.01A+.01B 仍只是待验证假设。

## 未改变的部分与兼容性

- `flysnake/research/teacher_spiking.py`、`teacher_afferent.py`、光学采样、动作解码、
  完整图定义、`models/current`、10 月 18 日快照均未修改。
- 本次修复入口是 `scripts/train_teacher_spiking.py`；README 默认的
  `scripts/evaluate_afferent_frozen.py` 未修改。不能宣称所有评估入口已改成双窗门控。
- **训练损失和检查点选择依旧是冷窗口协议。** 新双窗结果为附加冻结评估，
  顶层 `balanced_accuracy` 仍表示冷窗口；未把这一改动说成连续 200 ms 梯度训练。
- 修改后的训练脚本源码身份不同，旧训练进程的源码锁应继续阻止直接续训。
  今后使用此版本必须建立独立实验，不覆盖旧存档，不绕过 CUDA/源码身份校验。
- 每次该入口的冻结评估神经积分量约为原来的两倍；本次仅在 7 神经元夹具上执行，
  不能据此估算完整脑 GPU 时间或证明全图数值恢复。
- 真实微型神经回归覆盖基础模型及**非零既有突触增益**模型；两段 100 ms 与连续
  200 ms 的脉冲和全部返回状态完全相同。它是实现正确性证据，不是学习证据。

## 环境与资源

Python 3.12.14、Torch 2.11.0+cpu、NumPy 2.3.3；全部依赖安装在独立虚拟环境，
来源为官方 PyTorch CPU 索引及 PyPI。完整版本见 `environment-freeze.txt`。
OMP/MKL/OpenBLAS 线程均限制为 2。当前容器可见 CPU affinity 0–8，内存约 9.7 GiB，
未暴露 cgroup 配额文件，因此不把 affinity 当成保证的算力配额。无 GPU 节点，CUDA 不可用。

本次最终 110 项相关测试约 3.5 秒；不是全图训练速度。保留 PyTorch 两条 CSR 警告；
它们已在修改前存在，未通过修改原核心来隐藏。无付费算力、无完整脑训练/推理、无外置分类器。

## 复现命令

从本分支仓库根目录，创建新的独立环境（不要替换原 GPU 环境）：

```bash
python -m venv .venv-recovery
.venv-recovery/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv-recovery/bin/python -m pip install --index-url https://pypi.org/simple numpy==2.3.3 Pillow==12.3.0 pytest==9.1.1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

.venv-recovery/bin/python -m pytest \
  tests/test_teacher_training_safety.py tests/test_teacher_spiking.py \
  tests/test_teacher_afferent.py tests/test_teacher_game.py tests/test_teacher_data.py \
  tests/test_optical_aperture.py tests/test_game.py tests/test_decoder.py -q

# 独立进程，避免主模块与实验快照混用。
.venv-recovery/bin/python -m pytest experiments/continuation-2026-10-18/test_import_artifacts.py -q
.venv-recovery/bin/python scripts/diagnose_teacher_training_safety.py
```

需要文件时给诊断脚本加 `--output NEW-FILE.json`；已有文件会被拒绝覆盖。
复现修改前 89 项时使用基准提交，去掉新增 `test_teacher_training_safety.py`。
`baseline-defects.json` 的缺类案例是 `metrics([[5,5,0,0]],[1],...)`；非有限案例是
先给单参数梯度赋 Inf，再运行原来的默认 `clip_grad_norm_(...,1.)` 与 Adam。

## 尚未验证／下一步

1. 10 月 18 日原始 34,266,757 字节 ZIP 不在 Git 中；7 项数据依赖测试未运行。
   必须取得 SHA-256 为 `942cc762b84bf1361f078d27a07471b011e3eb28bb5c43a6e48d39aebf00ef4b`
   的原包并通过现有安全导入器，不能制造代替数据或以合成测试冒充复现。
2. 未恢复第 112 步模型，也未取得完整图/CUDA 身份兼容环境。完整脑连续 200 ms 训练、
   分区梯度测量、候选回灌和新冻结评估均未执行。
3. 将来在独立、身份核验通过的全图实验中先报告六项召回、两窗 BA 和连续队列，
   再验证正确/乱标签对照、多训练种子、封存图像及普通 12×12 闭环游戏成绩。
4. 远程 CI 是另外的验证阶段；本报告记录本地 CPU 结果，不预称 GitHub CI 成功。

独立只读审阅已检查门槛与状态传递语义，未发现阻断问题。
所有原始提交仍保留；本分支只新增修复代码、测试与报告，不替换默认模型。
