# APL骨架与接触位置：数据准备已完成

2026-10-02。本阶段获取官方 MaleCNS v1.0 空间数据，验证它与当前模拟连接图的对应关系。**两个APL细胞的骨架与接触位置已实际下载、提取并核验；局部非脉冲动力学尚未接入，不作为学会贪吃蛇的证据。**

官方出处：[MaleCNS数据说明](https://male-cns.janelia.org/download)。使用匿名公共存储读取，没有使用neuPrint令牌。数据位于独立目录，旧神经源码和训练记忆保留。

## 已得到的数据

| 细胞 | 骨架采样点 | 独立连通分量 | 原始输入接触 | 原始输出接触 |
|---|---:|---:|---:|---:|
| APL_R，10540 | 43,737 | 3 | 124,656 | 123,836 |
| APL_L，10977 | 45,168 | 2 | 118,348 | 133,530 |

骨架采样点是同一个细胞的形态节点，不能当作额外神经元。右APL主分量43,724点，另外两个分量4点、9点；左APL主分量45,160点，另一个分量8点。全部原样保留，没有把小片段人工连接到主树。

完整伙伴表扫描了 **311,833,243 行**，提取出涉及这两个APL的 **500,348 条唯一伙伴记录**。同一突触前位置可连接多个突触后位置，伙伴记录与突触前释放位置的数量不同：右APL有15,966个不同的突触前位置，左APL有16,911个。

其中 **443,915 条接触**的双方都属于当前模型保留的节点，对应 **12,027 条已有聚合连接**。余下 **56,433 条**涉及当前模型没有保留的对象；空间表仍保留它们，并通过标志和−1索引明确区分。方向统计可能重复计入同一条APL到APL的记录，不能直接相加当作去重总数。

## 实际核验结果

准备程序已完成，随后另一个核验程序独立读取文件，并使用不同的分组方法检查原始官方权重表。

| 核验 | 结果 |
|---|---|
| 6,777,179,098字节伙伴表的官方CRC32C | 一致，独立复读再次通过 |
| 整份缓存的SHA256 | 与准备记录一致 |
| 两个原始SWC的官方CRC32C与本地SHA256 | 一致 |
| SWC到NPZ的节点、父节点、坐标、单位和半径转换 | 独立核对一致 |
| 接触表的原始全局行号 | 唯一、有序，来自完整扫描 |
| 原始细胞ID到当前166,700节点顺序的映射 | 一致 |
| 当前模型12,027对APL相关连接的接触数 | 全部一致，0对不匹配 |
| 包括未保留对象的62,413对原始APL连接 | 与官方原始权重表全部一致 |

独立核验读取的是原始 `body_pre/body_post/weight` 表，不只是重复准备程序对规范化索引的检查。检查范围是数据完整性、形态转换与连接对应；它不验证膜参数、局部传播速度、突触释放机制或学习效果。

准备和恢复的11项测试通过：形态单位、父节点与不连通分量保留，非法骨架拒绝，原始ID映射与连接计数，以及有效下载块复用、损坏块拒绝、源版本改变拒绝。没有在本阶段运行游戏或新的神经训练。

## 文件与数据接口

主要数据目录：[APL空间数据](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1)。

| 文件 | 内容 |
|---|---|
| [10540.swc](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/10540.swc)、[10977.swc](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/10977.swc) | 官方原始骨架 |
| [右APL骨架NPZ](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/apl-10540-skeleton.npz)、[左APL骨架NPZ](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/apl-10977-skeleton.npz) | 节点、父节点索引、连通分量、原始和微米坐标、边长 |
| [APL伙伴接触表](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/apl-partners.feather) | 500,348条接触记录，约16.7MB |
| [准备记录](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/runtime/apl-spatial-20261002-v1/status.json) | 官方对象版本、校验值、来源快照、下载块、统计及核对结果 |
| [独立核验结果](C:/Users/wangs/Documents/Codex/2026-09-12/new-chat/fly-brain-snake/diagnostics/apl-spatial-data-audit-2026-10-02.json) | 复读校验和原始连接逐对比较 |
| `cache/syn-partners.feather` | 约6.8GB完整官方表缓存，保留用于复核 |

接触表保留官方字段：

`x_pre/y_pre/z_pre/body_pre/conf_pre/x_post/y_post/z_post/body_post/conf_post/primary_post`

另附 `global_partner_row`、`pre_native_index`、`post_native_index` 和 `retained_native_contact`。

- 接触坐标及原始SWC坐标均处于MaleCNS EM体素空间；**1体素=8nm=0.008µm**。未镜像、未做跨模板变换。
- NPZ的 `xyz_voxel` 保留原坐标，`xyz_um` 已乘0.008。`parent_index` 从0开始，−1表示分量根。
- 索引−1表示当前模拟中没有该对象，不表示一个补造的神经元。
- `primary_post` 是官方突触后脑区标签，未重新解释为APL的突触前树分区。

读取示例：

```python
from pathlib import Path
import numpy as np
import pyarrow.feather as feather

root = Path('runtime/apl-spatial-20261002-v1')
contacts = feather.read_table(root / 'apl-partners.feather')
with np.load(root / 'apl-10540-skeleton.npz', allow_pickle=False) as right_apl:
    xyz_um = right_apl['xyz_um'].copy()
    parent_index = right_apl['parent_index'].copy()
```

## 这一阶段解决了什么

之前只有细胞间接触数量，不能知道输入和输出发生在APL树的哪里。现在实际伙伴坐标和骨架已经补齐，而且能精确回到当前已有连接，不需要拿胞体位置猜接触位置。

下一阶段需要将接触点可靠地附着到骨架分支，构建有局部传播和电压依赖释放的APL模型，并用无奖惩的感觉传递实验校准。原始SWC标注为 **coarse**；它的几何半径、骨架密度和小片段不能直接当作已经测量好的电缆电学参数。空间准备成功不保证新的动力学或记忆实验会成功。

## 复现与下载恢复

新增数据依赖位于独立文件 `requirements-apl-spatial.txt`，没有改动旧神经依赖清单。

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-apl-spatial.txt
.\.venv\Scripts\python.exe scripts/prepare_apl_spatial_data.py --output runtime/apl-spatial-new
.\.venv\Scripts\python.exe scripts/audit_apl_spatial_data.py --run runtime/apl-spatial-new --output diagnostics/apl-spatial-new-audit.json
```

对中断的下载，在同一目录和参数下使用 `--resume`。程序先验证来源版本和已提交数据块的哈希再继续；存在大文件或状态显示`running`不能代替这些验证。当前这一份准备和独立核验均已完成，不需重下载。
