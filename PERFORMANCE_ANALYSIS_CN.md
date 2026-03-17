# `cost_aware_straighten_lcp.py` 性能分析报告

## 1. 概述

本报告针对 `cost_aware_straighten_lcp.py` 在较大栅格（约 7000×7000 cell）上运行缓慢（约 30 分钟甚至数小时）的问题，进行了代码审阅与性能瓶颈分析，并提出了分层次的优化方案。

### 测试环境与基准数据

使用 `cProfile` 在随机成本栅格上测量的基准性能：

| 栅格大小     | 标准 Dijkstra（无曲率） | 方向感知 Dijkstra（有曲率） |
|:------------|:----------------------|:-------------------------|
| 200 × 200   | ~1.0 s                | ~3.7 s                   |
| 500 × 500   | ~12.0 s               | ~44 s（推算）              |
| 7000 × 7000 | ~30 min（推算）         | ~数小时（推算）             |

> **关键发现**：当启用曲率控制（`curvature_factor > 0` 或 `max_turning_angle < 180`）时，运行时间激增约 3–4 倍。这解释了用户报告中"有时运行极慢达数小时"的情况——恰好对应使用曲率参数的场景。

---

## 2. 瓶颈分析

以 200×200 随机栅格的 cProfile 数据为基础，分析各模块耗时占比：

### 2.1 整体耗时分布（200×200, 标准 Dijkstra）

```
总耗时: 2.242 s

_dijkstra_standard       0.530 s  (23.6%)   ← Dijkstra 搜索本身
cost_aware_straighten_path 1.542 s (68.8%)  ← 路径拉直（后处理）
  ├── _line_cost          0.778 s  (34.7%)
  ├── _is_line_clear      0.742 s  (33.1%)
  └── _supercover_line    0.597 s  (26.6%)  ← 被上面两个重复调用
_smooth_path_nodata_safe  0.010 s  (0.4%)   ← 平滑（可忽略）
```

### 2.2 瓶颈一：纯 Python Dijkstra 循环（~24% 耗时，大栅格时主导）

**位置**：`_dijkstra_standard()` 第 248–283 行，`_dijkstra_with_direction()` 第 348–396 行

**问题**：整个 Dijkstra 搜索的内层循环完全用纯 Python 实现，包括：
- `heapq.heappop()` / `heapq.heappush()`：每个节点的入队和出队
- Python 级别的元组拆包、数组索引、浮点运算
- 每次迭代需要访问 `cost_data[nr, nc]`（单元素 NumPy 索引 → Python float）

**规模分析**：
- 7000×7000 栅格 = ~49,000,000 个 cell
- 每个 cell 需评估最多 8 个邻居
- Python 单次操作开销约 ~100 ns（vs C 语言 ~1 ns），即纯 Python 开销放大 ~100 倍

**方向感知版本更严重**：
- 状态空间从 `(row, col)` 扩展为 `(row, col, direction)`
- `best` 数组从 2D (`rows × cols`) 变为 3D (`rows × cols × 9`)
- 对 7000×7000 栅格，状态空间达 ~441,000,000
- 这是"数小时运行"的直接原因

### 2.3 瓶颈二：路径拉直中的重复计算（~69% 耗时）

**位置**：`cost_aware_straighten_path()` 第 563–661 行

**问题**：对每个候选快捷路径，**先后调用** `_is_line_clear()` 和 `_line_cost()`，两者内部都独立调用 `_supercover_line()` 计算完全相同的像元序列。

```python
# 当前代码（第 639–646 行）：
if not _is_line_clear(path[i], path[j], cost_data, rows, cols):
    continue   # ← 第一次计算 supercover line

shortcut_cost = _line_cost(
    path[i], path[j], cost_data, rows, cols, cell_size
)                # ← 第二次计算完全相同的 supercover line
```

在 200×200 测试中：
- `_supercover_line` 被调用 25,989 次
- 其中 `_is_line_clear` 调用 12,291 次，`_line_cost` 调用 12,291 次
- 即约一半的 `_supercover_line` 调用是 **完全冗余的**

### 2.4 瓶颈三：`_supercover_line` 纯 Python 实现

**位置**：第 69–103 行

**问题**：使用纯 Python `list.append()` 逐一构建像元列表。每条线段的像元数量与距离成正比，在大栅格上，跨越数千像元的线段会产生大量 Python 对象开销。

### 2.5 瓶颈四：`h_map` 全栅格预计算

**位置**：第 228–232 行 / 第 320–324 行

**问题**：预计算整个栅格的 A* 启发式值（`h_map`），需要创建 `rows × cols` 大小的 float32 数组。对 7000×7000 栅格：
- 内存占用：~196 MB
- 计算时间：包含广播乘法和 `sqrt`

**影响**：这是一次性操作，不是循环热点，但在内存紧张时可能导致额外的页面交换开销。

### 2.6 瓶颈五：`np.ascontiguousarray` 数据复制

**位置**：第 234 行 / 第 326 行（Dijkstra）和第 612 行（拉直）

**问题**：`cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)` 在每个阶段都复制一次 7000×7000 的 float64 数组（~392 MB）。在整个流程中可能被创建 2–3 次。

---

## 3. 优化方案

按实施难度和预期收益分为三个层次：

### 3.1 低风险优化（立即可实施，不引入新依赖）

#### 方案 A：合并 `_is_line_clear` 和 `_line_cost`

**原理**：将两个函数合并为一个 `_line_cost_or_inf()` 函数，只调用一次 `_supercover_line()`，同时完成障碍检查和成本计算。

**预期收益**：路径拉直阶段 `_supercover_line` 调用量减少 ~50%。

**已实施**：✅（见代码中新增的 `_line_cost_or_inf()` 函数）

#### 方案 B：避免重复 `np.ascontiguousarray` 复制

**原理**：在函数入口处检查数组是否已经是 contiguous float64，如果是则避免复制。同时在 Dijkstra 和拉直之间复用同一份 `cost_data`。

**预期收益**：减少 ~392 MB 内存分配和复制时间。

**已实施**：✅（Dijkstra 函数中已将 `cost_data` 传递给后续函数）

#### 方案 C：`_supercover_line` 使用预分配数组

**原理**：不使用 Python list 动态追加，改为预分配 NumPy int32 数组（最大长度 = 2×(|dr|+|dc|)+1），返回数组切片。

**预期收益**：减少大量 Python 对象创建和 GC 压力。

### 3.2 中等风险优化（需引入 Numba 依赖）

#### 方案 D：Numba JIT 编译 Dijkstra 内层循环 ⭐ 推荐

**原理**：使用 `@numba.njit` 装饰器将 Dijkstra 的核心循环编译为机器码。Numba 支持 heapq 操作和 NumPy 数组访问的 JIT 编译。

```python
import numba

@numba.njit
def _dijkstra_core(cost_data, sr, sc, er, ec, step_dists, ...):
    # 内层循环在编译后接近 C 速度
    ...
```

**预期收益**：Dijkstra 部分提速 **20–100 倍**，对 7000×7000 栅格可从 ~30 分钟降至 ~1 分钟以内。

**注意事项**：
- 首次调用有 JIT 编译开销（~2–5 秒），后续调用无开销
- 需要 `pip install numba`（依赖 LLVM）
- Numba 不支持 Python 的 `heapq` 模块，需手写堆操作或使用 Numba 兼容的实现
- 可考虑使用 Numba 的 typed list 或直接用数组实现二叉堆

#### 方案 E：Numba JIT 编译 `_supercover_line` 和拉直循环

**原理**：将 `_supercover_line` 和 `cost_aware_straighten_path` 的内层循环也用 Numba 编译。

**预期收益**：路径拉直部分提速 **10–50 倍**。

### 3.3 架构级优化（较大改动）

#### 方案 F：使用 SciPy 稀疏图 Dijkstra

**原理**：将栅格转换为稀疏图（`scipy.sparse.csr_matrix`），使用 `scipy.sparse.csgraph.shortest_path()` 或 `dijkstra()`。SciPy 的实现是 C + Cython 编写，速度远快于纯 Python。

**局限性**：
- 标准 Dijkstra 可直接使用，但 **方向感知 Dijkstra** 需要扩展状态空间（节点数 ×9），构建稀疏图本身可能消耗大量内存
- 无法直接表达曲率惩罚和反锯齿偏好等自定义逻辑

#### 方案 G：双向 Dijkstra

**原理**：同时从起点和终点开始搜索，在中间汇合时停止。

**预期收益**：理论上可减少搜索空间 ~50%，即提速约 2 倍。

**局限性**：方向感知版本较难正确实现双向搜索。

#### 方案 H：使用 C 扩展或 Cython 重写核心循环

**原理**：将 Dijkstra 核心循环和 `_supercover_line` 用 Cython 或 C 扩展重写。

**预期收益**：与 Numba 类似（20–100 倍），但无 JIT 编译开销。

**缺点**：需要编译环境，部署复杂度增加。

---

## 4. 推荐实施路线

### 第一阶段：立即实施（本次 PR）
1. ✅ **合并 `_is_line_clear` + `_line_cost`**（方案 A）
2. ✅ **避免重复 `np.ascontiguousarray` 复制**（方案 B）

### 第二阶段：中期优化（推荐下一步）
3. ⭐ **Numba JIT 编译 Dijkstra 核心循环**（方案 D）——这是收益最大的单一优化
4. **Numba JIT 编译拉直循环**（方案 E）

### 第三阶段：长期架构（如需进一步提速）
5. 考虑 SciPy 稀疏图替换标准 Dijkstra（方案 F）
6. 考虑 Cython 重写（方案 H）

---

## 5. 本次已实施的优化

### 5.1 合并 `_is_line_clear` 和 `_line_cost` 为 `_line_cost_or_inf()`

**变更前**：
```python
# cost_aware_straighten_path 中每个候选快捷路径：
if not _is_line_clear(path[i], path[j], ...):   # 调用 _supercover_line
    continue
shortcut_cost = _line_cost(path[i], path[j], ...)  # 再次调用 _supercover_line
```

**变更后**：
```python
# 只调用一次 _supercover_line，同时完成障碍检查和成本计算：
shortcut_cost = _line_cost_or_inf(path[i], path[j], ...)
if math.isinf(shortcut_cost):
    continue
```

### 5.2 避免重复数组复制

- Dijkstra 内部创建的 `cost_data` 通过 `_build_result` / `_build_result_directed` 传递给拉直函数
- 拉直函数不再重复调用 `np.ascontiguousarray`

### 5.3 预期综合收益

- 路径拉直阶段：`_supercover_line` 调用量减少 ~50%
- 内存使用：减少 ~392 MB 的重复数组复制
- 总体预估提速：在标准 Dijkstra 模式下约 **30–40%**（主要来自拉直阶段加速）
- Dijkstra 搜索本身的速度未改变——需要 Numba/Cython（方案 D/H）才能获得数量级提升
