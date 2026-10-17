# 连续双窗口条件重放实验 — 2026-10-18

**这是研究代码的合入，不是将成功学习的模型升级为默认模型。** 本目录保存上一轮交付的条件训练、41点插值搜索、离线视觉信息诊断和测试。原仓库的 `models/current`、完整脑核心代码、固定动作解码及 CUDA 身份校验保持不变。

## 范围与证据

本实验从旧 V5 代码快照 `3fdac042eb8ba18d04365b9124608d7441b8345e` 恢复所需的四个模块，重放边界来自 `learning-feedback-anchor-s1101-v1`。它不是完整 V5 分支的合并，也不是 F82 检查点的替换。

五个细胞内部仍计算连续两个100ms窗口及延迟队列，但外围神经释放序列固定为父模型记录。修改参数后的结果必须在完整脑重新验证。**三个候选均未通过完整脑学习验收；离线 Ridge 分类器不会接管游戏。**

| 条件候选 | 开发验证第一窗 | 开发验证第二窗 |
|---|---:|---:|
| A：12次 Adam 更新 | 87.50% | 29.17% |
| B：16次更新，时序一致性约束 | 54.17% | 66.67% |
| C：41点 TRAIN 插值选择 | 54.17% | 70.83% |

开发验证24张图已多次用于研究，不是封存最终测试；与48张训练图没有相同RGB哈希，但存在 pose_id 重叠。见 [原始实验报告](REPORT.md)、[结构化结果](RESULTS.json) 和 [本次整合测试记录](INTEGRATION-VALIDATION.json)。

`REPORT.md` 与 `SOURCE-PROVENANCE.json` 保留交付时的原文与校验信息。报告末尾“GitHub未修改”和清单中的 `uploaded_to_github: false` 描述的是**原ZIP生成时**的历史状态，不是本目录合入后的状态。清单中的路径相对于原ZIP；原ZIP的 `README.md` 在这里保存为 `REPORT.md`。

## 数据没有嵌入 Git

源代码、报告、校验清单入库。较大的边界 NPZ、参数 NPZ、原始历史日志和元数据仍在对话中交付的 **`FlyBrain-continuation-kit-2026-10-18.zip`** 中；本次没有为它创建 GitHub Release 附件或公开下载地址。克隆仓库后仍需取得这份原始包，不能只克隆就声称完成复现。

原ZIP：34,266,757字节，SHA-256：

```text
942cc762b84bf1361f078d27a07471b011e3eb28bb5c43a6e48d39aebf00ef4b
```

`import_artifacts.py` 使用固定的整包与清单SHA-256，并验证18个数据/结果文件的大小及哈希。只导入本目录的 `runtime/`、`results/` 和 `reproduction.log`，不提取或覆盖源代码及原模型。错误哈希、路径穿越、符号链接和不同内容的同名文件会被拒绝；已有相同文件可安全重复导入。

## 安装与运行

使用独立 Python 3.13 环境，不要改变原有完整脑的 GPU 环境。本次实际整合测试环境：Python3.13.5、Torch2.10.0+cpu、NumPy2.3.5、pytest9.0.2；运行下述命令前先进入此目录并激活单独的虚拟环境。

```bash
cd experiments/continuation-2026-10-18
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# 将下一行路径替换为原始ZIP的实际位置
python import_artifacts.py /path/to/FlyBrain-continuation-kit-2026-10-18.zip
python import_artifacts.py --check
python -m pytest test_continuation.py test_import_artifacts.py -q

python continue_conditional_experiment.py --source . --output results/repeat-A --steps 12 --lr 0.005 --temporal-weight 0
python continue_conditional_experiment.py --source . --output results/repeat-B --steps 16 --lr 0.002 --temporal-weight 0.4
python sweep_continuation.py --source . --proposal results/experiment-01/conditional-selected-parameters.npz --output results/repeat-sweep
python diagnose_visual_separability.py --source . --output results/repeat-separability
```

每个训练输出目录必须此前不存在。不要把本目录的 `files/` 覆盖到主仓库；其中的模块快照仅供独立实验进程使用。实验测试应与主仓库测试分开运行，避免同一个Python进程混用两套 `flysnake` 模块。

合入前已在独立目录验证原包导入，并通过 **7项条件重放测试 + 6项导入器安全测试 = 13项测试，0失败、0跳过**。6项导入器测试使用显式合成夹具，不是神经学习证据。主仓库原有Windows CI与本目录的实际数据测试不是同一套检查。

完整脑 `graph.npz` 继续使用主仓库已有Release提供的版本，本实验不包含它，也没有在本次合并中完成新的全脑训练、真实游戏闭环或封存测试。
