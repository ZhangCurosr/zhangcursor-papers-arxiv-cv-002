---
title: "Predicting-Functions-Not-Features-KANs-with-Function-Space-J"
source: https://arxiv.org/pdf/2608.12050v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:21:15"
field: "医学图像分割与可微分算子学习"
keywords: ["Kolmogorov-Arnold Networks", "Joint-Embedding Predictive Learning", "Medical Image Segmentation", "Function-space Learning", "Self-supervised Learning", "Multi-radius Signature"]
innovations: ["将JEPA预测学习扩展至KAN预聚合函数空间", "多半径函数签名表征边缘函数局部行为", "训练期预测推理期移除的自蒸馏范式"]
benchmarks: ["BUSI", "DDTI", "TN3K", "CVC-ClinicDB", "GlaS"]
---

# 论文速读：Predicting-Functions-Not-Features-KANs-with-Function-Space-J

## 一句话总结
本文提出了 FS-JEPA（Function-Space Joint-Embedding Predictive Learning），将联合嵌入预测学习从特征嵌入扩展到 KAN 的预聚合函数空间，通过预测多半径边缘函数签名实现结构化的 KAN 边缘函数学习，在五个医学图像分割基准上达到了最高的平均 Dice 分数。

## 研究问题与动机
1. **KAN 边缘函数缺乏显式学习目标**：现有 KAN 分割模型仅通过后聚合任务损失优化边缘函数，不同边缘函数配置可能产生相同的节点输出，导致单个函数的行为未被显式监督
2. **单点评估无法表征函数局部行为**：KAM 边缘函数在前向传播中仅在输入依赖的锚点处求值，相同响应的函数可能在附近区域行为不同
3. **传统 JEPA 方法局限于特征层**：现有 JEPA 方法主要在图像、token 或特征级嵌入上操作，未将可学习神经网络函数作为预测目标
4. **函数空间预测的对齐挑战**：不同 KAN 边代表不同的输入-输出连接，在线分支和目标分支采样的边必须保持显式对应关系

## 核心贡献（创新点）
1. **在 KAN 原生预聚合函数空间中 formulated 联合嵌入预测学习**：将预测目标从特征嵌入转移到结构化边缘函数签名，本质区别在于监督信号定义在聚合之前而非之后
2. **引入多半径函数签名表示**：通过锚点周围的多点函数求值捕获 KAN 边缘函数的局部行为变化，与单响应预测相比提供了更丰富的预测目标
3. **开发仅训练期的预测框架**：通过共享边索引保持在线-目标边对应关系，EMA 目标分支提供稳定目标而推理时完全移除预测组件
4. **端到端从零开始联合优化**：无需预训练检查点或外部教师模型，函数空间预测目标与分割任务损失从零开始联合优化

## 方法详解

### KAN 边缘函数与采样
KAN 层中边缘 $(i,j)$ 对标量输入 $u$ 的响应为：
$$\phi_{j,i}(u) = w_{j,i}^b \mathrm{SiLU}(u) + s_{j,i} \sum_{\ell=1}^{L} w_{j,i,\ell}^s B_\ell^p(u)$$

直接计算所有边缘的响应复杂度为 $O(BNd_{in}d_{out})$，因此随机采样 $K$ 个输入-输出边缘对 $\mathcal{S} = \{(i_k, j_k)\}_{k=1}^K$，通过选择算子 $\Phi_\mathcal{S}$ 将辅助激活成本降至 $O(BNK)$。

### 多半径边缘函数签名
对采样边 $e=(i,j)$ 与锚点输入 $u$，定义局部偏移集合 $\mathcal{D} = \{-0.10, -0.05, -0.025, 0.025, 0.05, 0.10\}$，签名为：
$$s_e = [\phi_e(u-0.10), \phi_e(u-0.05), \phi_e(u-0.025), \phi_e(u+0.025), \phi_e(u+0.05), \phi_e(u+0.10)]$$

### 确定性边对应
通过确定性坐标编码边身份：
$$p_i = 2\frac{i-1}{d_{in}-1} - 1, \quad p_j = 2\frac{j-1}{d_{out}-1} - 1$$
坐标对 $(p_i, p_j)$ 与多半径签名拼接作为预测器的条件，确保在线和目标分支的签名对应同一输入-输出边。

### 函数签名预测目标
在线预测器 $q_\psi$ 接收掩码上下文表示 $z_e$（拼接掩码边缘响应、输入锚点和确定性坐标），生成预测签名 $\hat{s}_e = q_\psi(z_e)$；EMA 目标分支提供稳定目标 $s_e^+ = \mathrm{sg}(\hat{s}_e^{EMA})$。预测损失为：
$$\mathcal{L}_{sig} = \mathrm{SmoothL1}(\hat{s}_e, s_e^+)$$

### 联合优化
总损失为：
$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{deep}\mathcal{L}_{deep} + \frac{\lambda_{sig}}{4}\sum_{b=1}^{4}\mathcal{L}_{sig}^{(b)}$$
预测组件仅在训练期使用，推理时完全移除。

## 实验与结果

### 数据集
五个医学图像分割基准：
- **BUSI**：乳腺超声图像分割
- **DDTI**：甲状腺超声结节分割
- **TN3K**：甲状腺区域与结节分割
- **CVC-ClinicDB**：息肉分割
- **GlaS**：组织病理学图像腺体分割

### 主要结果（Dice 分数）
| 方法 | BUSI | DDTI | TN3K | CVC | GlaS | **Average** |
|------|------|------|------|-----|------|-------------|
| UUEKAN（最强 KAN 基线） | 0.7569 | 0.7447 | 0.8206 | 0.8346 | 0.8992 | 0.8112 |
| **FS-JEPA（本文）** | **0.7926** | **0.7432** | **0.8258** | **0.8852** | **0.9215** | **0.8337** |
| Matched Scaffold | 0.7808 | 0.7289 | 0.8217 | 0.8798 | 0.9149 | 0.8252 |

- **平均 Dice 提升**：较 Matched Scaffold 提升 +0.85pp，较最强 KAN 方法 UUEKAN 提升 +2.25pp
- **参数量**：7.74M，远低于 UUEKAN 的 35.02M
- **跨骨干泛化**：在 U-KAN、UUEKAN、KMUNet 上均有提升，平均增益 0.47pp

### 消融实验关键发现
1. **预测目标空间对比**（BUSI）：
   - 无预测：0.7808 Dice
   - Token-JEPA：0.7808（无益）
   - Node-JEPA：0.7845
   - Edge-JEPA（单响应）：0.7771（反而下降）
   - **多半径签名**：0.7926（最优）

2. **签名构造质量**：Raw multi-point 签名在相对 MAE（0.9175 vs 0.9910）、NRMSE（0.8999 vs 0.9893）、Pearson 相关系数（0.4622 vs 0.3139）上均最优

## 相关工作脉络
1. **U-KAN (Li et al., 2025)**：首个将 KAN 引入医学图像分割的工作，但仅通过后聚合任务损失优化边缘函数
2. **KAT (Yang & Wang, 2025)**：将 KAN 应用于视觉 token 处理，同样缺乏预聚合函数级监督
3. **JEPA (Assran et al., 2023)**：联合嵌入预测架构，从输入重建转向潜在表示预测，但限于通用特征嵌入
4. **BYOL/VICReg**：无负样本的预测式自监督学习方法，为本工作的预测学习基础
5. **UUEKAN (Chen et al., 2026)**：当前最强 KAN 分割方法，使用不确定性引导注意力，参数量 35M

## 局限性与未来方向
1. **随机采样的覆盖率限制**：仅采样 $K$ 个边缘而非全部，可能遗漏部分重要函数结构
2. **固定偏移集合的普适性**：$\mathcal{D}$ 中的偏移值固定为 ±0.025/0.05/0.10，对不同尺度输入可能需要自适应
3. ** EMA 更新率的敏感性**：目标分支的指数移动平均更新率未系统研究其影响
4. **仅验证于医学图像分割**：方法在通用视觉任务上的泛化能力待验证
5. **推理时无函数空间信息**：预测组件在推理时完全移除，可能丢失部分有益的函数表示先验

## 研究启发与可借鉴点
1. **函数空间预训练的迁移价值**：FS-JEPA 证明可以将预测学习扩展到可学习算子空间，这一思路可迁移到 Neural ODE、Operator Network 等函数表示模型
2. **多半径签名的表征设计**：通过锚点邻域多点求值捕获局部行为变化的思路，可推广到其他函数逼近场景（如 SIREN、Fourier Feature Networks）
3. **训练期预测组件、推理期移除**：这种"自蒸馏"范式无需增加推理成本，适合资源受限的临床部署场景
4. **确定性坐标对齐机制**：用坐标对替代可学习嵌入来保持边对应的思路，避免了额外参数引入，具有简洁性优势
5. **与现有 KAN 架构的兼容性**：FS-JEPA 作为插件式模块可适配 U-KAN、KAT 等多种 backbone，证明函数空间学习的通用性

## 关键术语表
- **Kolmogorov-Arnold Network (KAN)**：将网络边参数化为可学习单变量函数的网络架构，节点输出为入边函数求和
- **Joint-Embedding Predictive Architecture (JEPA)**：预测潜在表示而非重建输入的自监督学习框架
- **Exponential Moving Average (EMA) 目标分支**：通过滑动平均维护稳定目标表示的辅助网络，不接收梯度更新
- **Multi-radius Function Signature**：在锚点周围多个偏移位置求值构成的函数局部行为表征向量
- **Pre-aggregation**：KAN 边缘函数求和后聚合前的状态，此处是预测学习的目标空间
- **SmoothL1 Loss**：结合 L1 和 L2 特性的损失函数，对离群点比 L2 更鲁棒
- **Matched Scaffold**：与 FS-JEPA 相同分割架构但无预测分支的对照模型
- **Stop-gradient (sg)**：阻止梯度回传的运算，用于目标分支保持稳定

## 可复现要素
- **数据集**：五个公开医学图像分割数据集（BUSI、DDTI、TN3K、CVC-ClinicDB、GlaS），论文未提供代码仓库链接
- **训练配置**：300 epochs，8×NVIDIA RTX 4090 GPU，随机初始化，无外部预训练
- **关键超参**：
  - 偏移集合 $\mathcal{D} = \{-0.10, -0.05, -0.025, 0.025, 0.05, 0.10\}$
  - 4 个 KAN 块配备预测分支
  - 损失权重 $\lambda_{deep}$、$\lambda_{sig}$ 论文未明确给出数值
- **开源状态**：论文未提及代码或权重开源计划
