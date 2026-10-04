# fly brain

果蝇完整连接组控制贪吃蛇的研究项目。仓库名为 `fly-brain`，保存当前模型、训练权重、恢复记忆、代码、测试和失败结果。完整连接图与训练权重一起构成可运行模型。

**当前模型尚未学会普通贪吃蛇。** 三个训练画面中有两个动作正确；72个新画面中33个正确；24局普通12×12贪吃蛇累计吃到0个食物。有限的条件右转依赖视觉和训练后的已有突触，但持续运行仍有明显右转偏置。完整结果见[模型审计](docs/current-model-audit.md)。

## 完整模型在哪里

| 内容 | 位置 |
|---|---|
| 166,700个神经元、25,582,938条解剖边及完整推理数组 | [Release v0.1.0-audit](https://github.com/wangshen310-hub/fly-brain/releases/tag/v0.1.0-audit) 的 `graph.npz`，739,883,106字节 |
| 完整图的来源、归一化、各数组及文件校验 | Release 的 `manifest.json`；仓库内[相同清单](models/current/graph-manifest.json) |
| 当前训练参数 | [models/current/brain.pt](models/current/brain.pt) |
| 第60步参数与优化器记忆 | [models/current/training-memory.pt](models/current/training-memory.pt) |
| 非零神经状态和继续运行的参考结果 | `models/current/replay.pt`、`replay-frames.npz`、`replay-states.npz`、`replay.json` |
| 初始模型、上游训练记忆、错配示范对照 | [models/controls](models/controls) |
| 固定训练/验证/测试RGB资料 | [models/current/dataset](models/current/dataset) |
| 模型身份、所有包内文件和源码SHA-256 | [models/current/model.json](models/current/model.json) |
| 实际冻结游戏、因果干预和时序记录 | [docs/evidence](docs/evidence) |

完整图超过Git单文件100MB限制，因此保存在**同一仓库的Release附件**中。它不是Git LFS指针、删减图或随机替代模型。单独下载源代码ZIP不会包含Release附件，请使用下方命令一起下载。仓库为私有，下载需要有访问权限的GitHub登录。

完整图SHA-256：`8b6a70780c1cb042e87a839b6beed36ca8c4681b1ec5643be14d6b26df18989f`。

## 运行机制与边界

```mermaid
flowchart LR
    RGB[固定整棋盘RGB] --> Eye[固定光学采样和光感受器]
    Eye --> Brain[完整连接组神经动力学]
    Brain --> Spikes[真实运动神经元整数脉冲]
    Spikes --> Action[固定左右脉冲差解码]
    Action --> Game[移动与碰撞结算]
    Game --> RGB
```

推理仅接收180×320的uint8 RGB；无嗅觉、食物坐标、专家标签、寻路接管或随机动作兜底。左/右DNa02细胞ID为523769/10360，前进读出为10162/10059。左右差不超过较强一侧脉冲的20%则直行；否则按脉冲差转向；四个读出全部沉默则无动作，游戏记录惩罚后继续神经计算。

专家示范只用于训练。近似梯度修改原图已有的2,319条运动输入突触增益，连接端点、兴奋/抑制符号和动作解码固定。当前工程模型使用1ms积分、2ms传递延迟和有界电导动力学；它不是旧C++模型的数值等价GPU移植。435,541条纯调制解剖边保留在图中，其快速传递效力为零，没有实现完整天然多巴胺学习。

现有普通游戏结果来自**冻结评估**：食物+1、撞墙/撞身−1、无输出−0.1被记录，参数不在线更新。不得把保存成功、突触改变或长期绕圈生存说成学会了贪吃蛇。

## 安装与恢复当前模型

已验证环境：Windows、Python3.12.13、Torch2.11.0+cu128、NumPy2.3.3、CUDA12.8，NVIDIA RTX4060 Laptop GPU。保存的模型锁定Torch/CUDA与源码身份；CPU单元测试与恢复这个CUDA检查点是不同的检查。

```powershell
gh repo clone wangshen310-hub/fly-brain
cd fly-brain
uv venv --python 3.12 .venv-supervised
uv pip install --python .venv-supervised/Scripts/python.exe -r requirements-supervised.txt

# gh auth login 用于尚未登录且有本私有仓库权限的账户。
.venv-supervised/Scripts/python.exe scripts/download_current_graph.py
.venv-supervised/Scripts/python.exe scripts/check_current_model.py --graph runtime/supervised-graph-20261003-v4 --output runtime/model-check.json
```

下载器校验完整图的大小、SHA-256和配套清单。验证器检查源码、包内文件、训练参数、优化器记忆、三个实际全图神经输出，以及从200ms非零状态继续运行的两个窗口。脉冲、动作、图像身份和时序要求一致；浮点状态容差为1e−6，不声称GPU状态逐字节一致。每次验证需选择一个尚不存在的输出文件。

### 运行冻结普通游戏评估

```powershell
.venv-supervised/Scripts/python.exe scripts/evaluate_afferent_frozen.py --graph runtime/supervised-graph-20261003-v4 --dataset models/current/dataset --normal-run models/current --mismatch-run models/controls/mismatch --output runtime/new-frozen-evaluation --episodes 24 --max-decisions 100
```

这是有限的真实神经评估，会实际运行正常、错配和重置突触等对照并保存收据；不会后台无限训练。新的输出目录必须尚不存在。

### 复现当前有限训练协议

```powershell
.venv-supervised/Scripts/python.exe scripts/probe_afferent_teacher.py --graph runtime/supervised-graph-20261003-v4 --dataset models/current/dataset --warm-run models/controls/warm-upstream --run runtime/new-normal-probe --steps 60 --lr 0.03
# 相同命令追加 --mismatch 并改用另一个 --run 目录，即错配示范对照。
```

该协议只训练三个画面，是条件动作检查，不是完整游戏学习方案。重新训练会建立新存档；仓库所附原始记忆不会被覆盖。

## 测试

```powershell
.venv-supervised/Scripts/python.exe -m pytest tests -q --junitxml=runtime/local-tests.xml
```

测试覆盖神经积分、真实脉冲与反向导数、已有突触塑性、动作收据、固定视觉、碰撞、奖惩事务和检查点恢复。小型CPU测试使用显式fixture；完整脑模型的GPU验证见[实际验证报告](reports/model-verification-2026-10-04.json)。[完整测试报告](reports/tests-complete-2026-10-04.xml)与[Windows CPU持续集成](.github/workflows/tests.yml)随代码提供。持续集成不会恢复CUDA训练模型。

## 来源与历史

MaleCNS v1.0数据来源、CC-BY-4.0归属及vendored MIT许可见[NOTICE.md](NOTICE.md)。源数据下载锁与图重建脚本保留；Release提供已经校验的完整导出，避免每次重新准备约1.1GB原始资料。

旧研究代码、审计文件和`Start-*`启动器保留供回顾。它们的服务端口、嗅觉方案与运行说明属于历史版本，不能代表当前示范训练模型。原始说明在[历史README](docs/LEGACY_README.md)，当前版本以本页、模型清单和[当前审计](docs/current-model-audit.md)为准。旧本地存档保持原状，虚拟环境、下载缓存及全部历史运行目录不纳入Git。
