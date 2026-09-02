---
title: "TorchMorph-CUDA-accelerated-Morphological-Transforms"
source: https://arxiv.org/pdf/2608.24738v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:18:12"
field: "GPU-accelerated image processing"
keywords: ["Mathematical Morphology", "Distance Transform", "Optimal Transport", "CUDA", "PyTorch Extension", "GPU Acceleration"]
innovations: ["Unified PyTorch-native batched CUDA kernels for 22 morphological/distance/transport operators with scipy.ndimage-compatible API", "Fused morphology kernels with host-resolved geometry and interior fast path eliminating per-voxel boundary tests", "Batch-tiled Sinkhorn solver with CUDA-graph replay for fixed-point iterations"]
benchmarks: ["scipy.ndimage throughput", "POT Sinkhorn solver", "EDT/Chamfer/BFDT accuracy"]
---

# 论文速读：TorchMorph-CUDA-accelerated-Morphological-Transforms

## 一句话总结
TorchMorph 是一个轻量级 PyTorch 扩展库，将 22 个形态学变换算子（二值/灰度形态学、精确/近似距离变换、熵正则化最优传输）实现为融合 CUDA 内核，原生支持 `(B, C, Spatial...)` 张量批量处理与最高 8 维空间维度，API 与 scipy.ndimage 逐参数对齐，使这些算子可直接嵌入 GPU 训练循环。

## 研究问题与动机
1. **CPU-only 参考实现无法嵌入 GPU 训练流程**：生态默认实现 scipy.ndimage 仅支持 CPU、单数组处理，每次调用需经历设备→主机拷贝、单线程 CPU 计算、主机→设备拷贝，造成昂贵延迟。
2. **现有 GPU 视觉库覆盖不足**：Kornia 仅支持 2D 可微形态学；cuCIM 通过 CuPy 间接提供 GPU 形态学而非原生 torch.Tensor；MONAI 部分后处理仍委托回 SciPy/CuPy；OpenCV 和 scikit-image 快速路径仅限 2D。
3. **现代 AI 成像管线三大特征未被满足**：数据以 PyTorch 张量形式驻留 GPU；训练步骤批量处理数十个体积；三维/四维数据（体积时间序列、多通道层析）的 CPU 成本随空间 extents 乘积增长。
4. **熵正则化最优传输分散在其他栈中**： practitioners 需组合三个库（形态学、距离变换、最优传输）才能同时获得批量 GPU 形态学、精确距离变换与可微运输距离。

## 核心贡献（创新点）
1. **统一 GPU 原生实现 22 个形态学/距离/传输算子**：一次性填补批量 GPU 形态学、精确欧氏距离变换与可微 Sinkhorn 求解器的生态空白，而现有方案需拼凑三个独立库。
2. **API 与 scipy.ndimage 逐参数对齐**：包括边界模式、结构元素原点、预分配输出缓冲等，现有管线仅需更改 import 即可迁移，无需重写代码。
3. **融合 CUDA 内核设计支持 (B, C, Spatial...) 高维批量张量**：内核直接作用于最高 8 维空间 CUDA 张量，结构元素几何在主机端解析，内部快速路径消除逐轴边界测试。
4. **分层架构实现职责分离**：Python 层负责参数归一化与 SciPy 兼容性，绑定层通过 pybind11 暴露 8 个内核入口，内核层实现 CUDA 逻辑，派生算子无需额外设备代码。
5. **差分测试保障数值一致性**：78 个测试函数覆盖所有导出算子，逐元素对比 scipy.ndimage（形态学/距离变换）与 POT（运输），并验证不变量（如运输计划重 marginals、梯度与有限差分一致）。

## 方法详解
### 2.1 算子分类（4 类 + 1 工具组）
- **二值与灰度形态学**：腐蚀/膨胀为核心，派生出开/闭运算、梯度、拉普拉斯、顶帽/底帽共 11 个算子。
- **距离变换**：精确欧氏距离变换（EDT）、Chamfer 变换（棋盘/Taxicab 度量）、暴力变换（作为正确性 oracle）。
- **熵正则化最优传输**：批量 Sinkhorn 求解器，支持 scaling 与 log 域两种形式，对两个 marginals 可微。
- **结构元素工具**：主机端解析 `structure > footprint > size` 优先级链。

### 2.2 三层架构
- **Python 层**：参数归一化与验证（展开 origin、映射边界模式字符串、组合派生算子），唯一知晓 SciPy 约定。
- **绑定层**：单一 pybind11 模块暴露 8 个内核入口（6 个 CUDA 翻译单元编译）。
- **内核层**：所有几何在主机端解析，内核针对运行时空间秩编写（编译时上限），坐标暂存于寄存器。

### 2.3 内核设计
#### 融合形态学内核
- **主机端预处理**：结构元素展平为活跃条目列表，每条记录 per-axis 偏移与预计算平坦偏移，非活跃位置不抵达设备。
- **设备端快速路径**：线程先测试输出坐标是否为内部点（通过主机计算的 per-axis 偏移极值判断），内部线程直接加预计算平坦偏移，无逐轴算术与边界测试；仅边缘线程走通用路径处理 5 种边界模式（灰度）或单边界值（二值）。
- **模板化实例**：腐蚀/膨胀为同一模板化内核的不同 functor 实例化；二值内核添加 done 谓词，一旦结果确定即提前退出（稀疏掩码时显著节省）。

#### 距离变换内核
- **EDT（可分离下包络算法）**：每维单次扫描计算抛物线族下包络；每个线程块处理一条扫描线，共享内存构建包络，并行二分搜索交点数组。
  - 2D 特化（extent ≤ 2048）：行扫描完全连续，最终平方根融合进列扫描。
  - 通用路径：转置活跃轴至最内层，使每扫描见连续内存。
  - 回退路径：扫描线超过共享内存预算时溢出到懒分配全局缓冲区。
- **Chamfer 变换**：维度可分离的前向/后向扫描，棋盘度量额外对角线扫描。
- **暴力变换**：背景坐标以 256 .tile 分块通过共享内存，模板化度量与空间秩，坐标循环完全展开。

#### 批量分块 Sinkhorn 求解器
- **共享成本矩阵**：每个线程块处理 `(row, batch-tile)`，tile 大小 8，流式读取矩阵行并应用于寄存器中 8 个缩放向量，矩阵流量降至 1/8。
- **单次 pass log 域更新**：维护运行最大值与重缩放运行和，空状态定义合并算子避免 NaN。
- **CUDA 图回放**：≥100 次迭代时捕获 25 次迭代到 CUDA 图并回放；固定点迭代在 ping-pong 缓冲区上执行，额外迭代安全；图捕获不可用时优雅降级为普通 launch。
- **自定义 autograd**：返回中心化对偶势，由包络定理给出相对于 marginals 的精确梯度；势在前向中中心化并暂存，反向仅为单次广播乘法，不穿透迭代。

## 实验与结果
### 实验环境
- **硬件**：NVIDIA GeForce RTX 4090 D（Ada Lovelace, 48 GB, driver 580.173.02），2× Intel Xeon Gold 6330（56 cores/112 threads, 2.00 GHz），128 GB 系统内存。
- **软件**：Ubuntu 24.04.3 LTS, CUDA 12.4, Python 3.12.13, PyTorch 2.6.0+cu124, NumPy 2.5.1, SciPy 1.18.0, POT 0.9.6.post1。
- **基准**：单线程 SciPy/POT（占用单核），GPU 结果使用单设备；计时使用 `torch.utils.benchmark`，`torch.cuda.synchronize()` 内采集中位数。

### 数值一致性（Table 2）
| 算子族 | 参考 | 最大绝对误差 | 相对 $\ell_2$ 误差 |
|--------|------|-------------|-------------------|
| 二值形态学 | ndimage | 0 | 0 |
| 灰度形态学 | ndimage | $4.77 \times 10^{-7}$ | $2.24 \times 10^{-8}$ |
| 欧氏距离变换 | ndimage | $2.06 \times 10^{-7}$ | $6.77 \times 10^{-9}$ |
| Chamfer DT（棋盘） | ndimage | 0 | 0 |
| Chamfer DT（Taxicab） | ndimage | 0 | 0 |
| 暴力 DT ($\ell_2$) | ndimage | $2.06 \times 10^{-7}$ | $6.71 \times 10^{-9}$ |
| Sinkhorn 距离 | POT | $1.75 \times 10^{-6}$ | $8.82 \times 10^{-8}$ |

- 二值形态学与 Chamfer 距离变换在测试中**精确匹配** SciPy。
- 所有浮点算子最大绝对误差 $< 1.8 \times 10^{-6}$，相对误差 $< 9 \times 10^{-8}$。
- NaN 传播是与 SciPy 的唯一已知行为差异。

### 吞吐量（Table 3， inputs/ms）
**灰度形态学**（最大加速）：
- Grey dilation on $256^2$: SciPy 0.756 → TM (B=8) **111.1**（≈147× 加速）
- Grey erosion on $256^2$: SciPy 0.740 → TM (B=8) **83.3**（≈112× 加速）

**二值形态学**（B=8）：
- Binary erosion/dilation on $256^2$: ≈**55 inputs/ms**（vs 0.865，≈64×）

**距离变换**：
- EDT 2-D $256^2$: SciPy 0.160 → TM (B=8) **40.0**（≈250×）
- EDT 3-D $128^3$: SciPy 0.003 → TM (B=8) **10.8**（≈3600×，但绝对吞吐量低）
- Chamfer 2-D $1024^2$: SciPy 0.037 → TM (B=8) **5.9**（≈159×）
- 暴力 BFDT $256^2$: <0.001 → **1.2 inputs/ms**（作为正确性 oracle，非性能主张）

**最优传输（Table 4，d=32² 网格）**：
- Scaling, 100 it.: POT 23.0 ms → TM **1.2 ms**（19×）
- Scaling, 1000 it.: POT 229.7 ms → TM **10.1 ms**（23×）
- Log-domain, 200 it.: POT 35.4 ms → TM **3.0 ms**（12×）
- **Batch n=16, 100 it.**: POT 367.3 ms → TM **8.7 ms**（**42.4×**，最强结果）
- 运输计划相对误差：$1.54 \times 10^{-3}$（batch 情形）。

### 结论
- **批量加速显著**：小输入（$256^2$）随 batch 增大吞吐量线性提升；大输入（$1024^2$ EDT）单个输入已接近设备饱和，批量增益 diminish。
- **速度上限**：灰度形态学达 **1.1×10³×** 加速，精确 EDT 达 **350×**，Sinkhorn 达 **42×**（vs 单核 CPU 参考）。

## 相关工作脉络
1. **scipy.ndimage**：CPU-only 参考实现，定义 border mode、structuring-element origin 与 iteration 语义的事实标准；TorchMorph 逐参数镜像其 API 以确保迁移兼容。
2. **Kornia**：提供 2D 可微形态学但缺乏 N 维算子、完整边界模式矩阵与精确 EDT；TorchMorph 填补 N-D 与精确距离变换空白。
3. **cuCIM**：通过 CuPy 提供 GPU 形态学，属互操作层而非原生 `torch.Tensor` 算子；TorchMorph 直接操作 PyTorch 张量，无缝嵌入训练循环。
4. **MONAI**：部分形态学后处理步骤委托回 SciPy 或 CuPy；TorchMorph 消除该回退，实现端到端 GPU 管道。
5. **POT (Python Optimal Transport)**：熵正则化最优传输的 CPU/reference 实现；TorchMorph 提供批量 GPU Sinkhorn 求解器，快 42× 且可微。
6. **OpenCV / scikit-image**：主机侧且快速路径仅限 2D；TorchMorph 支持高达 8 维空间张量的批量处理。

## 局限性与未来方向
**自述局限**：
1. **形态学与距离变换内核仅支持前向传播**，不可微（仅运输模块可微）；腐蚀/膨胀虽有 subgradient 路由到 arg-min/arg-max 位置，但未实现完整 autograd。
2. **需 CUDA**（意图回退为 SciPy），而运输模块可在 CPU 上回退到纯 PyTorch。
3. **使用 `--use fast math` 在 float32 下计算**，NaN 传播不一定与参考匹配。
4. **缺少连接组件与重建算子**（如 binary fill holes 虽实现但其他高级算子未覆盖）。

**未来方向（论文提及）**：
- 为形态学算子添加 autograd。
- 扩展更多 connected-component 与重建算子。

## 研究启发与可借鉴点
1. **API 镜像策略降低迁移成本**：与领域事实标准（scipy.ndimage）逐参数对齐，使现有管线仅需 `import torchmorph as tm` 替换，而非重写逻辑——可作为 GPU 化传统科学计算库的通用策略。
2. **分层架构分离职责**：Python 层（参数验证/SciPy 约定）、绑定层（pybind11 入口）、内核层（CUDA 实现）的职责分离，派生算子无需额外设备代码——适合其他需兼容多参考实现的 GPU 库。
3. **融合内核 + 主机预处理消除设备端重复计算**：结构元素展平、预计算平坦偏移、内部快速路径消除边界测试——可复用于其他邻域聚合算子（如卷积替代）。
4. **差分测试与 invariant 验证结合**：不仅逐元素对比参考实现，还验证不变量（运输计划重 marginals、梯度与有限差分一致）——可作为 GPU 数值库的正确性验证范式。
5. **CUDA 图回放用于迭代算法**：对固定点迭代（Sinkhorn）捕获 chunk 到图并回放，额外迭代安全——可推广至其他迭代优化（如 Power iteration、EM）。

## 关键术语表
- **Morphological Transforms**：数学形态学变换，包括腐蚀、膨胀、开/闭运算等，用于形状与掩码处理。
- **Structuring Element**：结构元素，定义形态学操作的邻域几何，可为 footprint、size 或 explicit structure。
- **Distance Transform**：距离变换，将每个前景像素标记到最近背景的距离，用于骨架化、分水岭种子与边界损失。
- **Exact Euclidean Distance Transform (EDT)**：精确欧氏距离变换，使用可分离下包络算法线性时间计算。
- **Chamfer Transform**：Chamfer 变换，使用可分离扫描近似 L∞（棋盘）与 L₁（Taxicab）度量距离。
- **Sinkhorn Iteration**：Sinkhorn 迭代，求解熵正则化最优传输的对偶缩放因子的迭代算法。
- **Entropy-Regularised Optimal Transport**：熵正则化最优传输，引入熵项使运输计划更平滑、计算更高效。
- **CUDA Graph**：CUDA 图，将多次 kernel launch 捕获为单次 replay，减少 launch 开销，适用于固定迭代模式。

## 可复现要素
- **数据集**：自定义基准测试（2-D/3-D/更高维输入，各尺寸与 batch 组合）；无公开数据集。
- **代码开源**：是，MIT 许可证，源码与文档：https://intcomp.github.io/tm
- **权重**：不适用（库而非模型）。
- **关键超参**：
  - 空间秩上限：8（Python 层硬编码）。
  - Sinkhorn batch tile 大小：8。
  - CUDA 图捕获 chunk 大小：25 次迭代。
  - 编译标志：`-O3 --use fast math`。
- **依赖**：PyTorch ≥2.6.0, CUDA toolchain；无其他第三方运行时依赖。
