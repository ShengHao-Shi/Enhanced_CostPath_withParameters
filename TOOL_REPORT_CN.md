# 增强型最低成本路径工具——运行逻辑汇总报告

> 本文档使用中文，对仓库内所有脚本与组件的运行逻辑、参数含义进行详细拆分说明。

---

## 目录

1. [整体架构](#1-整体架构)
2. [文件结构](#2-文件结构)
3. [核心算法模块 enhanced_lcp.py](#3-核心算法模块-enhanced_lcppy)
   - 3.1 [八方向连通定义](#31-八方向连通定义)
   - 3.2 [转弯角度计算](#32-转弯角度计算)
   - 3.3 [内部常量](#33-内部常量)
   - 3.4 [算法入口函数](#34-算法入口函数-enhanced_least_cost_path)
   - 3.5 [代价函数详解](#35-代价函数详解)
   - 3.6 [标准 Dijkstra 搜索](#36-标准-dijkstra-搜索-_dijkstra_standard)
   - 3.7 [方向感知 Dijkstra 搜索](#37-方向感知-dijkstra-搜索-_dijkstra_with_direction)
   - 3.8 [路径回溯重建](#38-路径回溯重建)
   - 3.9 [路径平滑——Chaikin 角切割](#39-路径平滑chaikin-角切割)
   - 3.10 [栅格文件读写辅助函数](#310-栅格文件读写辅助函数)
4. [ArcGIS 工具箱 arcgis_toolbox.pyt](#4-arcgis-工具箱-arcgis_toolboxpyt)
5. [实验脚本 enhanced_lcp_high_straightness.py](#5-实验脚本-enhanced_lcp_high_straightnesspy)
6. [用户参数速查表](#6-用户参数速查表)
7. [输出结果字典](#7-输出结果字典)
8. [运行流程总览图](#8-运行流程总览图)

---

## 1. 整体架构

本工具的目标是对标准 ArcGIS **最低成本路径 (Least Cost Path, LCP)** 工具进行增强。标准 ESRI LCP 工具仅接受三个输入（成本栅格、起点、终点），无法控制路径的弯曲度或长度偏好。本工具在标准 LCP 基础上增加了：

| 增强功能 | 解决的问题 |
|---|---|
| **弯曲度控制（Curvature Control）** | 标准 LCP 经常产生不切实际的急转弯，不适用于道路、管道等线性设施规划 |
| **距离因子（Distance Factor）** | 标准 LCP 可能为了经过低成本区域而产生过长的绕行路径 |
| **防锯齿/防闪电弯折（Anti-zigzag）** | 在成本变化较小的栅格上，路径经常出现闪电状弯折 |
| **路径平滑（Path Smoothing）** | 栅格路径本身只能沿 8 个方向移动，需要后处理产生平滑曲线 |

架构采用 **"核心算法 + 界面包装"** 策略：
- **核心算法**（`enhanced_lcp.py`）仅依赖 NumPy，可独立于 ArcGIS 运行
- **ArcGIS 工具箱**（`arcgis_toolbox.pyt`）是一层薄包装，负责读取 ArcGIS 数据并调用核心算法

---

## 2. 文件结构

```
CostPath_withParameters/
├── enhanced_lcp.py                    # 核心算法（仅依赖 NumPy）
├── enhanced_lcp_high_straightness.py  # 实验脚本（大幅增加直线鼓励因子）
├── arcgis_toolbox.pyt                 # ArcGIS Python 工具箱（需要 arcpy）
├── requirements.txt                   # Python 依赖（numpy, rasterio）
├── tests/
│   ├── __init__.py
│   └── test_enhanced_lcp.py           # 单元测试（37 个测试用例）
└── README.md                          # 英文说明文档
```

---

## 3. 核心算法模块 `enhanced_lcp.py`

### 3.1 八方向连通定义

算法在栅格网格上运行，每个栅格单元与其周围 8 个邻居相连（上、下、左、右及四个对角方向）。

```
方向编号:  0=北(N)  1=东北(NE)  2=东(E)  3=东南(SE)
           4=南(S)  5=西南(SW)  6=西(W)  7=西北(NW)
```

每个方向用 `(行偏移, 列偏移)` 表示：

| 编号 | 方向 | (dr, dc) | 说明 |
|------|------|----------|------|
| 0 | 北 N | (-1, 0) | 行号减1，列号不变 |
| 1 | 东北 NE | (-1, 1) | 行号减1，列号加1 |
| 2 | 东 E | (0, 1) | 行号不变，列号加1 |
| 3 | 东南 SE | (1, 1) | 行号加1，列号加1 |
| 4 | 南 S | (1, 0) | 行号加1，列号不变 |
| 5 | 西南 SW | (1, -1) | 行号加1，列号减1 |
| 6 | 西 W | (0, -1) | 行号不变，列号减1 |
| 7 | 西北 NW | (-1, -1) | 行号减1，列号减1 |

**实际意义**：栅格坐标系中，行号向下增大（北→南），列号向右增大（西→东）。

### 3.2 转弯角度计算

`turning_angle(dir_from, dir_to)` 函数计算从一个方向转到另一个方向的偏转角度。

**计算原理**：在 8 个方向组成的圆上，两个方向之间最短弧所跨过的步数乘以 45° 即为转弯角度。

```
转弯角度 = min(|dir_from - dir_to|, 8 - |dir_from - dir_to|) × 45°
```

| 转弯情况 | 角度 | 示例 |
|----------|------|------|
| 直行（同方向） | 0° | 北→北 |
| 微偏转 | 45° | 北→东北 |
| 直角转弯 | 90° | 北→东 |
| 大幅转弯 | 135° | 北→东南 |
| 掉头 | 180° | 北→南 |

**性能优化**：算法预先计算了一个 8×8 的查找表 `_TURN_ANGLE_LUT`，在内循环中通过索引直接获取转弯角度，避免重复计算。

### 3.3 内部常量

| 常量名 | 值 | 含义 |
|--------|------|------|
| `_CURVATURE_AMPLIFIER` | 5.0 | 弯曲度放大器。因为用户输入的 `curvature_factor` 范围是 0.0–1.0，直接作为惩罚权重太小，需要乘以此放大器以产生可感知的效果 |
| `_STRAIGHTNESS_PENALTY` | 0.3 | 直线前进鼓励因子（防锯齿惩罚）。当路径改变方向而前方直行的成本相近时，额外施加此惩罚以鼓励直线前进 |
| `NO_DIR` | -1 | 哨兵值，表示起始点没有"来时方向" |

### 3.4 算法入口函数 `enhanced_least_cost_path`

这是用户调用的主函数，完整签名：

```python
enhanced_least_cost_path(
    cost_raster,          # 成本栅格（二维 NumPy 数组）
    start,                # 起点 (行, 列)
    end,                  # 终点 (行, 列)
    curvature_factor=0.0, # 弯曲度因子
    max_turning_angle=180.0,  # 最大允许转弯角度
    distance_factor=0.0,  # 距离因子
    cell_size=(1.0, 1.0), # 栅格单元尺寸 (y方向, x方向)
)
```

**运行逻辑**：

1. **参数校验** → 调用 `_validate_params()` 检查所有参数合法性
2. **选择算法分支**：
   - 若 `curvature_factor > 0` 或 `max_turning_angle < 180`，使用 **方向感知 Dijkstra**（状态空间更大但支持弯曲度控制）
   - 否则使用 **标准 Dijkstra**（更快、更省内存）
3. 两个分支最终都返回包含路径、平滑路径、总代价等信息的字典

**设计意图**：当用户不需要弯曲度控制时，自动使用更高效的标准 Dijkstra，避免不必要的内存和计算开销。

### 3.5 代价函数详解

每一步从当前单元 A 移动到邻居单元 B 时，总代价 = 四个组件之和：

```
总代价 = 基础代价 + 弯曲度惩罚 + 直线偏好惩罚 + 距离惩罚
```

#### 3.5.1 基础代价 (base_cost)

```
base_cost = cost_raster[B] × step_distance
```

- `cost_raster[B]`：目标单元 B 上的成本值（来自用户输入的成本栅格）
- `step_distance`：从 A 到 B 的欧氏距离
  - 上下左右移动 = `cell_size`（正交方向的单元尺寸）
  - 对角线移动 = `√(cell_y² + cell_x²)`

**实际意义**：这是最基本的代价，即"穿过某片区域的难度"。成本栅格值越高，穿越该单元的代价越大。例如在地形分析中，陡坡的成本值比平地高。

#### 3.5.2 弯曲度惩罚 (curvature_penalty)

```
curvature_penalty = curvature_factor × _CURVATURE_AMPLIFIER × (angle / 180) × step_distance × cost_scale
```

- `curvature_factor`：用户设定的弯曲度控制参数 (0.0–1.0)
- `_CURVATURE_AMPLIFIER`：内部放大器 (= 5.0)
- `angle`：本次转弯的角度（0°/45°/90°/135°/180°）
- `cost_scale`：成本栅格所有有效值的平均值（用于使惩罚与基础代价在同一数量级）

**实际意义**：转弯角度越大、弯曲度因子越高，惩罚越重。这迫使算法寻找更平滑的路径。`cost_scale` 的作用是自适应归一化——无论成本栅格值是 1–10 还是 100–1000，惩罚强度都与基础代价成比例。

#### 3.5.3 直线偏好惩罚 / 防锯齿惩罚 (straightness_penalty)

```
straightness_penalty = similarity × _STRAIGHTNESS_PENALTY × cost_scale × step_distance
```

仅在以下条件同时满足时触发：
- 已有来时方向（不是起始点）
- 当前准备改变方向（`d_out ≠ d_in`）

其中 `similarity`（相似度）的计算：

```
similarity = max(0.0, 1.0 - |目标单元成本 - 直行单元成本| / cost_scale)
```

- 如果目标单元 B 和直行方向单元的成本 **相近**（similarity → 1.0），惩罚 **最重**——既然走直线也差不多，为什么要拐弯？
- 如果目标单元 B 比直行方向 **便宜很多**（similarity → 0.0），惩罚 **很轻**——拐弯是因为确实更便宜

**实际意义**：这是解决"闪电弯折"问题的关键机制。在成本变化很小的区域（如均匀平原），标准 Dijkstra 可能在多个成本几乎相同的路线之间反复跳跃，产生锯齿形路径。此惩罚使算法在"成本差不多"的情况下更倾向于保持直线。

#### 3.5.4 距离惩罚 (distance_penalty)

```
distance_penalty = distance_factor × cost_scale × step_distance
```

- `distance_factor`：用户设定的距离控制参数 (0.0–1.0)

**实际意义**：每走一步都会产生与距离成正比的额外代价。这鼓励算法寻找更短的路径，即使需要穿越稍高成本的区域。例如，设定较高的 `distance_factor` 可以避免为了走低成本区域而产生的大幅绕行。

#### 3.5.5 代价函数总结

| 组件 | 作用 | 何时激活 |
|------|------|----------|
| 基础代价 | 反映地表成本 | 始终 |
| 弯曲度惩罚 | 惩罚急转弯 | `curvature_factor > 0` |
| 直线偏好惩罚 | 防止闪电弯折 | 方向改变且前方成本相近时 |
| 距离惩罚 | 鼓励更短路径 | `distance_factor > 0` |

### 3.6 标准 Dijkstra 搜索 (`_dijkstra_standard`)

当 `curvature_factor == 0` 且 `max_turning_angle == 180` 时使用此分支。

**状态空间**：`(行, 列)` —— 每个栅格单元一个状态
**数据结构**：
- `best[rows, cols]`：记录到达每个单元的最低代价（float32，节省内存）
- `parent_dir[rows, cols]`：记录到达每个单元的来时方向（int8），用于路径回溯

**搜索过程**（A* 优化的 Dijkstra）：

1. 初始化：起点代价 = 0，加入优先队列
2. 从优先队列中取出代价最低的节点
3. 如果是终点，搜索结束
4. 遍历 8 个邻居：
   - 检查边界和障碍物（NaN/Inf）
   - 计算 `base_cost + distance_penalty + straightness_penalty`
   - 如果新代价优于已知最优，更新并加入优先队列
5. 重复步骤 2-4 直到找到终点或队列为空

**A* 启发式**：使用当前点到终点的欧氏距离乘以最小单元成本作为下界估计，加速搜索。

**虽然此分支不使用弯曲度惩罚，但仍包含防锯齿机制**（`_STRAIGHTNESS_PENALTY`），以减少闪电弯折。

### 3.7 方向感知 Dijkstra 搜索 (`_dijkstra_with_direction`)

当 `curvature_factor > 0` 或 `max_turning_angle < 180` 时使用此分支。

**状态空间**：`(行, 列, 来时方向)` —— 每个栅格单元有最多 9 个状态（8 个方向 + 1 个"无方向"用于起点）
**数据结构**：
- `best[rows, cols, 9]`：三维数组，记录以不同方向到达每个单元的最低代价
- `parent_d[rows, cols, 9]`：记录父节点的来时方向，用于路径回溯

**搜索过程**（与标准 Dijkstra 类似，但增加了以下逻辑）：

1. **硬转弯约束**：如果 `转弯角度 > max_turning_angle`，直接跳过该邻居（不允许过大的转弯）
2. **弯曲度惩罚**：转弯角度越大，惩罚越重
3. **直线偏好惩罚**：与标准分支相同

**为什么需要方向感知？**
标准 Dijkstra 中，到达某单元的"最优路径"只有一条。但当引入弯曲度惩罚后，"从北方到达 (3,3)" 和 "从东方到达 (3,3)" 的最优后续路径可能完全不同。因此必须将来时方向纳入状态空间。

**内存代价**：状态空间扩大 9 倍。对于 6908×4750 的大型栅格，使用 float32 而非 float64 可节省约 1.18 GB 内存。

### 3.8 路径回溯重建

搜索完成后，从终点沿父节点指针逆向回溯至起点，重建完整路径。

#### 标准 Dijkstra 的回溯 (`_build_result`)

```
当前单元 → 查看 parent_dir[r, c] 得到来时方向 d
→ 父单元 = (r - DIRECTIONS[d][0], c - DIRECTIONS[d][1])
→ 重复直到来时方向为 -1（起点）
```

#### 方向感知 Dijkstra 的回溯 (`_build_result_directed`)

```
当前状态 (r, c, d_idx) → 查看 parent_d[r, c, d_idx] 得到父节点的来时方向
→ 父单元 = (r - DIRECTIONS[d_idx][0], c - DIRECTIONS[d_idx][1])
→ 重复直到来时方向为 NO_DIR（起点）
```

回溯后还会计算：
- **路径物理长度**：各段欧氏距离之和
- **各步方向编号**
- **各顶点处的转弯角度**

### 3.9 路径平滑——Chaikin 角切割

`smooth_path(path, iterations=3)` 对离散栅格路径进行后处理平滑。

**Chaikin 角切割算法原理**：

对路径中每一对相邻点 P₀ 和 P₁，生成两个新点：
- Q = 0.75 × P₀ + 0.25 × P₁（距 P₀ 的 1/4 处）
- R = 0.25 × P₀ + 0.75 × P₁（距 P₀ 的 3/4 处）

用 Q 和 R 替换原来的线段，重复迭代。每次迭代使路径点数约翻倍，角落逐渐变圆。

**关键特性**：
- 起点和终点始终被保留
- 默认 3 次迭代
- 如果路径只有 2 个点（直线），不做平滑

**实际意义**：栅格路径只能沿 8 个方向移动，必然会在方向变化处产生 45°/90° 的锐角。Chaikin 平滑将这些锐角替换为平滑曲线，使输出更适合实际工程应用（如道路设计）。

### 3.10 栅格文件读写辅助函数

#### `load_cost_raster(filepath)`

使用 `rasterio` 库读取 GeoTIFF 等格式的成本栅格。

**返回值**：
- `data`：二维 NumPy 数组（float64），nodata 区域设为 NaN
- `metadata`：包含坐标变换、坐标参考系、单元尺寸等信息的字典

#### `save_path_raster(filepath, path, reference_metadata)`

将路径保存为二值栅格（1 = 路径上, 0 = 路径外），格式为 GeoTIFF。

---

## 4. ArcGIS 工具箱 `arcgis_toolbox.pyt`

这是一个 ArcGIS Python Toolbox，为 ArcGIS Pro 用户提供图形化界面。

### 4.1 工具箱结构

- **`Toolbox` 类**：工具箱容器，注册工具列表
- **`EnhancedLeastCostPathTool` 类**：实际的地理处理工具

### 4.2 参数定义 (`getParameterInfo`)

工具箱定义了 7 个参数，对应 ArcGIS Pro 界面上的输入控件：

| 序号 | 参数名 | 类型 | 输入方式 | 说明 |
|------|--------|------|----------|------|
| 0 | Cost Raster | GPRasterLayer | 下拉菜单（当前地图中的栅格图层） | 成本栅格 |
| 1 | Start Point | GPFeatureLayer (Point) | 下拉菜单（当前地图中的点要素图层） | 起点 |
| 2 | End Point | GPFeatureLayer (Point) | 下拉菜单（当前地图中的点要素图层） | 终点 |
| 3 | Curvature Factor | GPDouble (0.0–1.0) | 数值输入（可选，默认 0.0） | 弯曲度因子 |
| 4 | Max Turning Angle | GPDouble (0–180) | 数值输入（可选，默认 180.0） | 最大转弯角度 |
| 5 | Distance Factor | GPDouble (0.0–1.0) | 数值输入（可选，默认 0.0） | 距离因子 |
| 6 | Output Path | DEFeatureClass | 文件路径 | 输出路径要素类 |

### 4.3 执行流程 (`execute`)

```
1. 重新加载 enhanced_lcp 模块（方便开发调试，无需重启 ArcGIS Pro）
2. 读取用户输入参数
3. 通过 arcpy 读取成本栅格：
   - arcpy.Raster() → 读取栅格对象
   - arcpy.RasterToNumPyArray() → 转为 NumPy 数组
   - 获取单元尺寸 (meanCellWidth, meanCellHeight)
   - 获取范围 (extent) 和空间参考 (spatialReference)
4. 提取起点/终点要素的坐标 → 转换为栅格 (行, 列)
5. 调用 enhanced_least_cost_path() 核心算法
6. 将结果的平滑路径转换为折线要素类并写入输出
```

### 4.4 坐标转换辅助函数

#### `_fc_to_point(fc_path)`
从要素类中提取第一个点的 (X, Y) 地图坐标。

#### `_xy_to_rowcol(xy, extent, cell_x, cell_y, shape)`
将地图坐标 (X, Y) 转换为栅格坐标 (行, 列)：
```
列 = (X - 范围左边界) / 单元宽度
行 = (范围上边界 - Y) / 单元高度
```
结果会被裁剪到合法范围内。

#### `_write_polyline(path, extent, cell_x, cell_y, sr, output_fc)`
将路径 (行, 列) 转回地图坐标 (X, Y)，创建折线要素类：
```
X = 范围左边界 + (列 + 0.5) × 单元宽度
Y = 范围上边界 - (行 + 0.5) × 单元高度
```
加 0.5 是为了定位到单元中心。

---

## 5. 实验脚本 `enhanced_lcp_high_straightness.py`

这是 `enhanced_lcp.py` 的实验性副本，**唯一修改是大幅增加了防锯齿参数**：

| 常量 | 原始值 | 实验值 | 倍率 |
|------|--------|--------|------|
| `_STRAIGHTNESS_PENALTY` | 0.3 | 5.0 | ≈16× |
| `_CURVATURE_AMPLIFIER` | 5.0 | 20.0 | 4× |

**实验目的**：验证"极大增加直线前进鼓励因子是否能消除闪电形状弯折路径"。

**预期效果**：
- 路径应该更加平直，减少方向改变次数
- 代价可能略有上升（为了保持直线而放弃了一些低成本路线）
- 在成本变化较小的区域效果最为明显

**使用方法**：在 ArcGIS 工具箱中将 `import enhanced_lcp` 改为 `import enhanced_lcp_high_straightness as enhanced_lcp`，或在独立脚本中直接导入此模块。

---

## 6. 用户参数速查表

### 6.1 `cost_raster`（成本栅格）

- **类型**：二维 NumPy 数组 / ArcGIS 栅格图层
- **含义**：每个栅格单元的穿越成本。值越高，穿越该单元越"困难"或"昂贵"
- **特殊值**：`NaN` 或 `Inf` 表示不可通行的障碍物
- **应用示例**：
  - 地形坡度栅格（坡度越大成本越高）
  - 土地利用成本栅格（建设用地成本低、自然保护区成本高）
  - 综合加权成本表面

### 6.2 `start` / `end`（起点/终点）

- **类型**：`(行, 列)` 元组 / ArcGIS 点要素图层
- **含义**：路径的起始位置和目标位置
- **限制**：必须在栅格范围内，且不能位于 NaN/Inf 单元上

### 6.3 `curvature_factor`（弯曲度因子）

- **范围**：0.0 – 1.0
- **默认值**：0.0（不施加弯曲度惩罚）
- **效果**：
  - `0.0`：无弯曲度控制，等同于标准 LCP（但保留防锯齿）
  - `0.1–0.3`：轻微平滑，允许大部分转弯但稍微偏好直线
  - `0.3–0.6`：中等平滑，明显减少急转弯
  - `0.6–1.0`：强力平滑，路径非常平缓，可能为了避免转弯而显著偏离最低成本路线
- **内部计算**：实际惩罚权重 = `curvature_factor × 5.0 × cost_scale`

### 6.4 `max_turning_angle`（最大转弯角度）

- **范围**：0° – 180°
- **默认值**：180°（不限制转弯角度）
- **效果**：这是一个 **硬约束**，任何超过此角度的转弯都被 **完全禁止**
  - `180°`：允许所有转弯（包括掉头）
  - `135°`：禁止掉头
  - `90°`：只允许直行、45° 偏转和 90° 转弯
  - `45°`：只允许直行和 45° 微调
- **注意**：设置过低可能导致找不到路径（特别是在有障碍物的地形中）

### 6.5 `distance_factor`（距离因子）

- **范围**：0.0 – 1.0
- **默认值**：0.0（不考虑路径长度）
- **效果**：
  - `0.0`：纯粹按成本最低原则寻路
  - `0.3`：轻微偏好较短路径
  - `0.5–0.7`：中等偏好，会放弃一些低成本绕行
  - `1.0`：强烈偏好短路径，可能穿越较高成本区域
- **内部计算**：每步额外代价 = `distance_factor × cost_scale × step_distance`

### 6.6 `cell_size`（单元尺寸）

- **类型**：`(y方向尺寸, x方向尺寸)` 元组
- **默认值**：`(1.0, 1.0)`
- **含义**：每个栅格单元在地图单位中的物理尺寸
- **ArcGIS 中**：自动从栅格的 `meanCellWidth` 和 `meanCellHeight` 获取
- **影响**：
  - 影响步进距离计算（正交步 vs 对角步）
  - 影响路径物理长度（`path_length`）的计算
  - 影响 A* 启发式距离估计

---

## 7. 输出结果字典

核心算法返回一个包含以下键的字典：

| 键名 | 类型 | 说明 |
|------|------|------|
| `path` | `list[(int, int)]` | 从起点到终点的栅格单元序列，每个元素为 `(行, 列)` |
| `smoothed_path` | `list[(float, float)]` | 经过 Chaikin 平滑后的路径，坐标为分数值 `(行, 列)` |
| `total_cost` | `float` | 沿最优路径的累计总代价 |
| `path_length` | `float` | 路径的物理长度（地图单位） |
| `directions` | `list[int]` | 每一步的方向编号（0–7），长度 = 路径点数 - 1 |
| `turning_angles` | `list[float]` | 每个内部顶点处的转弯角度（度），长度 = 路径点数 - 2 |

---

## 8. 运行流程总览图

```
用户输入
  │
  ├─ 成本栅格 (cost_raster)
  ├─ 起点 (start)
  ├─ 终点 (end)
  ├─ 弯曲度因子 (curvature_factor)  [可选]
  ├─ 最大转弯角度 (max_turning_angle)  [可选]
  ├─ 距离因子 (distance_factor)  [可选]
  └─ 单元尺寸 (cell_size)  [自动获取或手动设定]
          │
          ▼
  ┌─────────────────────────┐
  │   参数校验              │
  │   _validate_params()    │
  └──────────┬──────────────┘
             │
             ▼
   curvature_factor > 0         curvature_factor == 0
   或 max_turning_angle < 180?   且 max_turning_angle == 180?
         │                              │
         ▼                              ▼
  ┌──────────────────┐     ┌────────────────────────┐
  │ 方向感知 Dijkstra│     │ 标准 Dijkstra          │
  │ (含弯曲度惩罚)   │     │ (仅含防锯齿惩罚)       │
  │ 状态: (r,c,dir)  │     │ 状态: (r,c)            │
  │ A* 启发式加速    │     │ A* 启发式加速          │
  └────────┬─────────┘     └──────────┬─────────────┘
           │                          │
           ▼                          ▼
  ┌─────────────────────────────────────┐
  │        路径回溯重建                 │
  │   从终点逆向追踪 parent 至起点      │
  │   计算路径长度、方向、转弯角度      │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │   Chaikin 路径平滑                  │
  │   3 次角切割迭代                    │
  │   保留起点和终点                    │
  └──────────────┬──────────────────────┘
                 │
                 ▼
          输出结果字典
            ├─ path (原始栅格路径)
            ├─ smoothed_path (平滑路径)
            ├─ total_cost (总代价)
            ├─ path_length (物理长度)
            ├─ directions (方向序列)
            └─ turning_angles (转弯角度序列)
```

---

## 附录：参数调优建议

| 场景 | 建议参数 |
|------|----------|
| 标准最低成本路径（无额外控制） | `curvature_factor=0, max_turning_angle=180, distance_factor=0` |
| 道路规划（需要平滑弯道） | `curvature_factor=0.3–0.5, max_turning_angle=90, distance_factor=0.2` |
| 管道铺设（弯道成本高） | `curvature_factor=0.6–0.8, max_turning_angle=45–90, distance_factor=0.3` |
| 消除闪电弯折（实验） | 使用 `enhanced_lcp_high_straightness.py`，或手动增大 `_STRAIGHTNESS_PENALTY` |
| 最短路径偏好 | `distance_factor=0.8–1.0` |
