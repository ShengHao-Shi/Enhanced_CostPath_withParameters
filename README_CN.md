# 增强型最低成本路径工具（带参数控制）

> **English documentation** → [README.md](README.md)

本工具在标准最低成本路径（LCP）算法的基础上，增加了**弯曲度控制**、**距离因子**、
**代价感知路径拉直**和 **Chaikin 平滑**四项功能，针对 ESRI 内置 LCP 工具的根本性局限进行了改进。

支持**独立 Python 库**（无需 ArcGIS 许可证）运行，同时提供 **ArcGIS Pro Python 工具箱**
（`.pyt`），供偏好地理处理 UI 的用户使用。

---

## 目录

1. [为什么需要本工具？](#1-为什么需要本工具)
2. [几何原理](#2-几何原理)
3. [可视化效果](#3-可视化效果)
4. [参数说明](#4-参数说明)
5. [快速入门](#5-快速入门)
6. [下载](#6-下载)
7. [仓库结构](#7-仓库结构)
8. [运行测试](#8-运行测试)
9. [许可证](#9-许可证)

---

## 1. 为什么需要本工具？

ArcGIS 内置的*最低成本路径*工具仅提供三个输入项：成本栅格、起点、终点。
在实际应用中，通常还需要以下几项控制：

| ArcGIS 原生 LCP 的局限 | 造成的问题 |
|---|---|
| **无弯曲度控制** | 生成的路径常出现突兀的急转弯，对道路、管道等线性基础设施而言不切实际。 |
| **无距离加权** | 只考虑成本表面，可能为了穿越低成本区域而绕很大的弯，忽视了更短但略贵的路线。 |
| **栅格对齐的锯齿伪影** | 八方向栅格路径产生阶梯状折线，即使整体走向正确，视觉上也不自然。 |
| **无平滑曲线** | 即使有转弯约束，格网对齐路径的转角依然尖锐；基础设施设计需要平滑弧线。 |

本工具通过**弯曲度惩罚**、**距离惩罚**、**代价感知后处理拉直**和 **Chaikin 平滑**
四种互补机制解决上述问题，并对任意成本栅格保持完全兼容。

---

## 2. 几何原理

### 2.1 八方向格网搜索

算法在**八连通栅格格网**上执行改进型 Dijkstra 搜索。
从每个像元出发，可向所有八个相邻像元移动（正方向与对角线方向）。

```
西北 北 东北        (行-1,列-1) (行-1,列) (行-1,列+1)
西   ·  东   →     (行,  列-1)     ·     (行,  列+1)
西南 南 东南        (行+1,列-1) (行+1,列) (行+1,列+1)
```

步长欧几里得距离：

- **正方向步**（N、S、E、W）：`d = 像元尺寸`
- **对角线步**（NE、NW、SE、SW）：`d = √2 × 像元尺寸`

这确保路径长度度量在几何上是准确的。

启用弯曲度控制时，搜索**状态**从 `(行, 列)` 扩展为 `(行, 列, 来向)`，
从而可以在每一步计算转弯角度并将其纳入代价函数，代价是状态空间增大约 8 倍。

### 2.2 扩展代价函数

对于从像元 *A* → 像元 *B* 的每一步（从方向 *d_入* 到 *d_出*），每步代价为：

```
step_cost = base_cost（基础代价）
          + curvature_penalty（弯曲度惩罚）
          + straightness_penalty（直线惩罚）
          + distance_penalty（距离惩罚）
```

| 组成部分 | 公式 | 生效条件 |
|---|---|---|
| **base_cost（基础代价）** | `cost_raster[B] × 步长` | 始终生效 |
| **curvature_penalty（弯曲度惩罚）** | `curvature_factor × 5 × cost_scale × (θ / 180°) × 步长` | `curvature_factor > 0` |
| **straightness_penalty（直线惩罚）** | `similarity × 0.3 × cost_scale × 步长` | 方向感知模式下发生方向变化时 |
| **distance_penalty（距离惩罚）** | `distance_factor × cost_scale × 步长` | `distance_factor > 0` |

**关键变量说明：**

- `θ` — *d_入* 与 *d_出* 之间的转弯角度（0° = 直行，45° = 一步对角线，90° = 直角转弯，180° = 掉头）。
- `cost_scale` — 所有有限非负栅格值的均值；使惩罚项与基础代价保持相同数量级。
- `similarity` — `max(0, 1 − |cost_B − cost_直行| / cost_scale)`。
  当目标像元与直行像元代价相近时值高；在异质区域抑制防锯齿惩罚。

**硬转弯约束：** 当 `max_turning_angle < 180°` 时，转弯角度超过阈值的步骤
将从搜索中完全排除（而非仅加惩罚）。

### 2.3 代价感知路径拉直

原始 Dijkstra 路径沿栅格格网行进，因此含有阶梯状锯齿。
后处理步骤尝试将每段多像元子路径替换为直线捷径，从而减少不必要的航点。

算法使用**超覆盖线**（格网遍历光栅化）枚举直线经过的所有像元。
对每条候选捷径 *A → E*：

1. **障碍检查** — 若直线上有任何 NODATA 或越界像元，则拒绝该捷径。
2. **代价检查** — 计算直线上的累积代价（平均像元代价 × 欧几里得距离），
   并与同一段原始栅格路径代价比较：

```
接受捷径 A → E  ⟺  cost(捷径) ≤ cost(A→…→E) × cost_tolerance
```

这防止了拉直路径悄悄穿越 Dijkstra 搜索正确避开的高成本区域。
`cost_tolerance` 参数（默认 1.05，即允许 5% 额外代价）提供了对
视觉直线度与成本保真度之间权衡的明确控制。

### 2.4 Chaikin 平滑

拉直后，**Chaikin 角切割算法**通过迭代细分将每个尖角替换为平滑弧线。
对于航点序列 `[P₀, P₁, P₂, …]`，每次迭代在每个内部顶点附近插入两个新点：

```
Q = 0.75 × Pᵢ + 0.25 × Pᵢ₊₁
R = 0.25 × Pᵢ + 0.75 × Pᵢ₊₁
```

经过 3 次迭代后，结果收敛为原始折线的 B 样条近似（圆角转弯）。
最后进行 NODATA 安全检查；若任何平滑线段穿越障碍像元，则回退到未平滑的拉直路径。

---

## 3. 可视化效果

### 算法流水线

三个后处理阶段将原始格网路径转化为平滑整洁的结果：

![算法流水线：格网路径 → 拉直 → 平滑](docs/images/pipeline_steps.png)

| 阶段 | 典型航点数（每 100 像元路径） | 视觉质量 |
|---|---|---|
| 原始八方向格网路径 | ~100–200 | 阶梯锯齿 |
| 代价感知拉直后 | 减少 70–90% | 整洁直线段 |
| Chaikin 平滑后 | （点密度增加） | 转角平滑弧线 |

---

### 参数效果对比

每个参数独立控制路径形态的不同方面：

![弯曲度因子、距离因子及组合参数的效果](docs/images/comparison_parameters.png)

| 面板 | 参数设置 | 可见效果 |
|---|---|---|
| 标准 LCP | 全部默认（0） | 锯齿格网路径；可能绕行较远 |
| + 弯曲度 | `curvature_factor=0.7` | 转弯更平缓；路径避免急转弯 |
| + 距离 | `distance_factor=0.4` | 路径向更短路线靠拢 |
| 全参数组合 | 弯曲度 + 距离 + 拉直 | 平滑、直接、贴近实际的路径 |

---

### 代价容差效果

`cost_tolerance` 控制拉直步骤在格网路径上绕过的激进程度：

![代价容差从 1.0 到 2.0 的效果](docs/images/cost_tolerance_effect.png)

- **1.0** — 只接受代价不超过原始路径的捷径；非均匀表面上拉直效果很少。
- **1.05** *（默认）* — 允许 5% 额外代价；良好的平衡点。
- **1.2** — 更激进的拉直；可能经过略贵的捷径。
- **2.0** — 基本上只检查 NODATA；最大化视觉直线度。

---

## 4. 参数说明

### 完整参数参考

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|---|---|---|---|---|
| `cost_raster` | 二维 NumPy 数组 | — | *必填* | 遍历成本表面。`NaN`/`Inf`/负值像元为不可通行障碍。 |
| `start` | `(行, 列)` | — | *必填* | 起点（零起始行/列索引）。 |
| `end` | `(行, 列)` | — | *必填* | 终点（零起始行/列索引）。 |
| `curvature_factor` | float | 0.0 – 1.0 | 0.0 | 急转弯的软惩罚权重。0 = 标准 LCP；1 = 最大平滑。 |
| `max_turning_angle` | float | 0 – 180 | 180.0 | 转弯角度硬上限（度）。180 = 不限制。 |
| `distance_factor` | float | 0.0 – 1.0 | 0.0 | 路径长度的权重。越高 ⇒ 越偏向更短路径。 |
| `straighten_factor` | float | 0.0 – 0.5 | 0.3 | 控制拉直步骤向前搜索捷径的距离。 |
| `cost_tolerance` | float | ≥ 1.0 | 1.05 | 捷径代价与原始路径代价之比的最大允许值。 |
| `cell_size` | `(y, x)` | — | `(1, 1)` | 每个栅格像元的物理尺寸（地图单位）。 |

### 输出字典

| 键 | 类型 | 说明 |
|---|---|---|
| `path` | `list[(int, int)]` | 从起点到终点的原始八连通格网路径。 |
| `straightened_path` | `list[(float, float)]` | 代价感知视线拉直后的路径。 |
| `smoothed_path` | `list[(float, float)]` | 最终 Chaikin 平滑路径（转角为圆弧）。 |
| `total_cost` | `float` | 沿最优格网路径的累积代价。 |
| `path_length` | `float` | 格网路径的物理长度（地图单位）。 |
| `directions` | `list[int]` | 每步的方向索引（0–7）。 |
| `turning_angles` | `list[float]` | 格网路径每个内部顶点的转弯角度（度）。 |

---

## 5. 快速入门

### 5.1 独立 Python 使用

**安装依赖：**

```bash
pip install numpy rasterio        # 最低依赖
pip install numba                  # 可选——大栅格上速度提升 20–50 倍
```

**基本使用示例：**

```python
import numpy as np
from pure_python.cost_aware_straighten_lcp import cost_aware_least_cost_path

# 示例：200×200 随机成本栅格
raster = np.random.default_rng(0).uniform(1, 10, (200, 200)).astype("float32")

result = cost_aware_least_cost_path(
    raster,
    start=(0, 0),
    end=(199, 199),
    curvature_factor=0.5,       # 平滑转弯
    max_turning_angle=135.0,    # 禁止接近掉头的转弯
    distance_factor=0.3,        # 轻度偏向更短路径
    straighten_factor=0.3,      # 适度后处理拉直
    cost_tolerance=1.05,        # 允许捷径有 5% 额外代价
)

print(f"格网路径像元数    : {len(result['path'])}")
print(f"拉直后航点数      : {len(result['straightened_path'])}")
print(f"平滑后点数        : {len(result['smoothed_path'])}")
print(f"总代价            : {result['total_cost']:.2f}")
print(f"路径长度          : {result['path_length']:.2f} 地图单位")
print(f"最大转弯角度      : {max(result['turning_angles']):.0f}°")
```

**Numba 加速版本**（即插即用替换，需安装 `numba`）：

```python
from numba_accelerated.cost_aware_straighten_lcp import cost_aware_least_cost_path

result = cost_aware_least_cost_path(raster, (0, 0), (199, 199),
                                    curvature_factor=0.5)
```

**加载 GeoTIFF 成本栅格：**

```python
import rasterio
import numpy as np
from pure_python.cost_aware_straighten_lcp import cost_aware_least_cost_path

with rasterio.open("cost_surface.tif") as src:
    data = src.read(1).astype("float64")
    data[data == src.nodata] = np.nan          # 将 NODATA 标记为障碍
    cell_y = abs(src.transform.e)              # 像元高度（地图单位）
    cell_x = abs(src.transform.a)              # 像元宽度（地图单位）

result = cost_aware_least_cost_path(
    data,
    start=(row_start, col_start),
    end=(row_end, col_end),
    cell_size=(cell_y, cell_x),
)
```

### 5.2 ArcGIS Pro 使用

1. 从 [Releases 页面](../../releases/latest) **下载** `EnhancedCostPath_ArcGIS.zip`
   并解压到本地文件夹。
2. 在 **ArcGIS Pro** → **目录**面板 → **工具箱** → 右键 → **添加工具箱**
   → 选择 `arcgis_toolbox_with_progress.pyt`。
3. 展开工具箱，可看到两个工具：
   - **Cost-Aware LCP (Pure Python)** — 无需额外依赖即可运行。
   - **Cost-Aware LCP (Numba Accelerated)** — 需要在 ArcGIS Pro Python
     环境中安装 `numba`（`conda install -c conda-forge numba`）。
4. 双击所需工具，填入参数，点击**运行**。
   地理处理面板中将逐步显示进度信息。

---

## 6. 下载

| 包名 | 内容 | 适用场景 |
|---|---|---|
| [EnhancedCostPath_ArcGIS.zip](../../releases/latest) | 工具箱 + 算法包 | ArcGIS Pro 用户 |
| [EnhancedCostPath_Standalone.zip](../../releases/latest) | 仅算法包 | 独立 Python / 无 ArcGIS |

如需自行构建安装包，请在仓库根目录执行 `bash release/build_release.sh`。
详见 [release/README.md](release/README.md)。

---

## 7. 仓库结构

```
Enhanced_CostPath_withParameters/
├── arcgis_toolbox_with_progress.pyt   ← ArcGIS Python 工具箱（最新版）
├── pure_python/
│   └── cost_aware_straighten_lcp.py   ← 主算法（纯 Python）
├── numba_accelerated/
│   └── cost_aware_straighten_lcp.py   ← Numba JIT 加速算法
├── tests/                             ← Pytest 测试套件
├── docs/
│   ├── generate_readme_figures.py     ← 重新生成 README 图像的脚本
│   ├── images/                        ← README 中使用的图像
│   ├── COST_AWARE_LCP_EN.md           ← 详细英文算法文档
│   ├── COST_AWARE_LCP_CN.md           ← 详细中文算法文档
│   ├── TOOL_REPORT_CN.md              ← 开发报告（中文）
│   └── PERFORMANCE_ANALYSIS_CN.md    ← 性能分析报告（中文）
├── release/
│   ├── README.md                      ← 打包发布说明
│   └── build_release.sh               ← 构建可分发 zip 包的脚本
├── archive/                           ← 早期算法变体（方案A和B，仅供参考）
├── requirements.txt
├── README.md                          ← 英文文档
└── README_CN.md                       ← 本文件（中文）
```

---

## 8. 运行测试

```bash
# 安装测试依赖
pip install pytest numpy

# 仅运行纯 Python 测试（无需 numba）
python -m pytest tests/test_cost_aware_straighten_lcp.py tests/test_progress_callback.py -v

# 运行全部测试（包含 Numba 加速变体）
pip install numba
python -m pytest tests/ -v
```

---

## 9. 许可证

MIT

```
特此免费授予任何获得本软件及相关文档文件（"软件"）副本的人不受限制地处置
本软件的权利，包括但不限于使用、复制、修改、合并、发布、分发、再授权和/或
出售本软件副本的权利，以及允许获得本软件的人员这样做，但须符合以下条件：
上述版权声明和本许可声明应包含在本软件的所有副本或大部分内容中。
本软件按"原样"提供，不附带任何明示或暗示的保证。
```
