---
title: "Measuring-Browser-Webcam-Gaze-Honestly-A-Capture-Clock-Metho"
source: https://arxiv.org/pdf/2608.11566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:21:03"
field: "浏览器眼动追踪与交互延迟评估"
keywords: ["Browser WebGaze", "Latency Measurement", "Weakly-Supervised Segmentation", "rVFC API", "Reproducibility", "Webcam Eye Tracking", "Kernel Ridge Regression"]
innovations: ["基于rVFC API的逐帧捕获时钟方法论，修复浏览器眼动延迟≈0ms的测量缺陷", "开源TypeScript参考实现与双引擎可互换基准测试套件", "揭示消费级webcam在临床弱监督分割任务中无法替代专家红外眼动仪的上限"]
benchmarks: ["Kvasir-SEG", "WebGazer baseline", "FaceMesh+KRR pipeline"]
---

# 论文速读：Measuring-Browser-Webcam-Gaze-Honestly-A-Capture-Clock-Metho

## 一句话总结
本文提出基于浏览器 `requestVideoFrameCallback` (rVFC) API 的逐帧捕获时钟方法论，修复了浏览器眼动追踪中常见的“延迟≈0 ms”测量缺陷，并开源了双引擎可互换的 TypeScript 基准测试套件；实证表明消费级 webcam 在临床弱监督分割任务中无法替代专家级红外眼动仪。

## 研究问题与动机
- **测量实践存在系统性缺陷**：浏览器眼动流水线包含视频解码、推理循环、渲染管线等多个异步边界，主流做法仅在 gaze 样本发射时记录 `performance.now()`，将发射时间误当作帧捕获时间，导致报告的推理延迟恒为 ≈0 ms。
- **无法评估交互延迟预算**：0 ms 假象掩盖了真实引擎开销，难以判断系统是否满足 50 ms 等交互式延迟目标。
- **webcam 眼动是否具备下游临床可用性不明**：现有 gaze-prompted 分割工作均依赖实验室红外眼动仪，消费级 webcam 眼动信号的质量是否足以支撑弱监督训练缺乏实证。
- **精度评估维度单一**：既往工作常以聚合误差（mean/p95）概括性能，忽略了空间散布与时间抖动的解耦差异。

## 核心贡献（创新点）
1. **捕获时钟方法论**：利用 rVFC API 恢复逐帧捕获时间戳（优先 `captureTime`，回退至 `presentationTime` 时明确声明为下界），为透明引擎提供精确 FIFO 配对，为黑盒引擎提供可验证的下界近似。
2. **开源参考实现与基准套件**：发布完整的 TypeScript 单页应用与基准 harness，内置 WebGazer 基线与新型 FaceMesh+KRR 引擎，支持 Sweep/Drift 双协议与多维度分析脚本。
3. **实证发现延迟鸿沟与精度解耦**：纠正测量后报告真实中位延迟 22–34 ms（p95 达 50–52 ms），揭示空间散布相近但时间抖动相差 1.6–3.5× 的精度分裂现象，并为临床弱监督管道提供上限级实证。

## 方法详解
- **流水线建模**：将浏览器眼动管道抽象为 `视频解码 → 推理循环 → 渲染提交` 三级异步边界。核心指标为推理延迟 $\ell_I = t_e - t_c$ 与管线延迟 $\ell_P = t_r - t_c$，其中 $t_c$ 为源帧捕获时钟。
- **rVFC 时钟恢复**：`requestVideoFrameCallback` 每解码一帧触发，附带 `captureTime`（相机捕获时刻）与 `presentationTime`（合成提交时刻）。实现优先读取 `captureTime`，若不可用则回退 `presentationTime`，并在导出 CSV 头部显式记录 `capture_clock_source`。
- **精确配对（§3.2）**：对暴露推理流水线的引擎（如 FaceMesh+KRR），每收到一帧则将当前帧时钟推入 FIFO 队列；gaze 样本发射时出队首元素作为 $t_c^{(k)}$，实现源帧与估计值的严格一一对应。
- **下界配对（§3.3）**：对黑盒引擎（如 WebGazer），维护标量 $\tilde{t}_c = \max\{\tau_i : i \leq \text{now}\}$，每次发射时以最新观测帧时钟标记。由于真实源帧必然早于或等于 $\tilde{t}_c$，报告值 $\tilde{\ell}_I$ 构成真实延迟的可验证下界，其松弛量为有效队列深度 × 帧间隔。
- **控制层与校准**：One-Euro 低通滤波（β=0.007）平滑输入；I-VT 在线分类器（阈值 1200 px/s，连续2帧超限判定为扫视，≥3样本后计算凝视质心）提取 fixation；平滑追踪校准（Lissajous 轨迹，18 s）补偿 100 ms 生理延迟。
- **FaceMesh+KRR 引擎**：提取 13 维 MediaPipe 虹膜/眼角特征向量（含相对位移、眼距代理、左右非对称项、归一化原始坐标），经标准差缩放（底限 2×10⁻² 防病态）后输入 RBF 核岭回归（λ=10⁻³，γ 由校准特征中位数两两距离启发式设定）。

## 实验与结果
- **协议设置**：单用户（N=1），14-inch MacBook Pro M4，集成 1080p 摄像头 30 Hz，视距 60 cm，固定光照与坐姿。运行 WebGazer/FaceMesh 各 Sweep(16×8) 与 Drift(随机10点) 共四轮。
- **延迟鸿沟**：朴素测量报告 ≈0 ms；捕获时钟校正后 FaceMesh 中位 22.0–22.8 ms（p95 26.8–27.0 ms），WebGazer 中位 32.8–34.0 ms（p95 50.6–52.0 ms）。20–50 ms 差距直接决定 50 ms 交互预算是否达标。
- **精度与空间结构**：FaceMesh 平均误差 8.05°（Sweep）/ 6.50°（Drift），WebGazer 11.11° / 11.09°，差异落在约 4.6° 的跨轮变异带内，不予排名。两引擎径向 p95 散布相近（≈6.1°），但时间抖动 $v_{p99}$ 相差 1.6–3.5×；错误分布呈质量性差异（FaceMesh 径向扩散，WebGazer 对角偏差）。
- **网格分辨率缩放**：误差随单元格俯角降低基本持平（L1–L5 斜率可忽略），命中率先降；对应当前用户误差预算，可用单元格下限约为 7° 视角。
- **下游临床弱监督验证（GazeMedSeg on Kvasir-SEG）**：保持分割管道完全不变，仅替换 gaze CSV 源。专家 EyeLink 1000 标注训练得到 test Dice 0.679；消费级 webcam 标注 train Dice ≈ 0.000。webcam 凝视落在息肉内的比例仅 17%（EyeLink 90%），伪掩码 Dice 0.12–0.17 vs 0.75–0.78。该差距为纯硬件惩罚的上界（标注者专业度与浏览指令同步改变）。

## 相关工作脉络
- **WebGazer [7]**：基于眼补丁像素的脊回归基线，十年仍为标准参考；其宣传准确率 4–5° 与实际部署 8–11° 存在显著落差。
- **TurkerGaze [13]**：众包眼动显著性图，同样受限于浏览器测量缺陷未加修正。
- **MediaPipe FaceMesh [4]**：提供逐帧虹膜关键点，本文首次将其与核岭回归及公开 benchmark 组合。
- **rVFC API [12]**：W3C 规范提供逐帧元数据，本文首次将其用于浏览器眼动推理延迟的真实测量。
- **GazeMedSeg [14]**：MICCAI 2024 强监督工作，利用专家 EyeLink 注视生成高斯热图训练 U-Net；本文在其上游验证 webcam 信号能否胜任同类弱监督任务。
- **GazeSAM [11] / Gaze2Segment [5]**：均依赖实验室眼动仪，未解决浏览器端可部署性问题。

## 局限性与未来方向
- **样本量受限**：N=1 单用户单会话设计，组内引擎对比稳健，但空间误差结构等结论需多用户重复实验验证（已在筹备中）。
- **校准混淆**：跨引擎精度差异与校准方式（九点网格 vs 平滑追踪）耦合，L6 固定网格对照实验显示 WebGazer 对校准方式敏感。
- **下游任务多重变量混杂**：硬件更换伴随标注者专业度与浏览指令变化，当前结果仅给出硬件惩罚上界，需协议对齐的纯硬件隔离实验。
- **未来方向**：多用户复现实验（N=5）、手术视频端到端临床试点、校准-引擎解耦的标准化采集协议。

## 研究启发与可借鉴点
- **测量范式修正**：rVFC 捕获时钟可作为浏览器实时感知管道（语音、手势、AR 追踪）延迟测量的通用修复方案，避免“零延迟”假象误导工程决策。
- **精度多维刻画**：将空间散布与时间抖动解耦评估，比单一聚合误差更能揭示引擎特性（如本工做 FaceMesh 抖动高但空间集中，WebGazer 相反），值得后续眼动/触控工作借鉴。
- **变异带保守声明**：主动量化并报告跨轮变异带（~4.6°），将小于该带的差异定性为“观察值而非排名”，提升实证严谨性。
- **端到端管道交换验证**：通过固定下游 GazeMedSeg 管道仅切换 gaze 源，直观呈现信号质量阈值，为弱监督医疗 AI 提供可复用的硬件能力评估范式。
- **开源与可复现设计**：完整 TypeScript SPA、原始 CSV 日志、分析脚本与网格缩放协议一并公开，符合 FAIR 原则。

## 关键术语表
- **rVFC API**：`requestVideoFrameCallback`，浏览器每解码一帧触发回调并提供 `captureTime`/`presentationTime` 的 Web API。
- **捕获时钟**：以相机实际帧捕获时刻为基准的时间戳，用于真实计算推理与管线延迟。
- **精确配对**：对暴露逐帧推理流水线的引擎，通过 FIFO 队列将 gaze 样本与其源帧严格对应。
- **下界配对**：对黑盒引擎，用最新观测帧时钟标记发射样本，报告延迟为真实值的可验证下界。
- **One-Euro 滤波**：基于输入速度动态调整截止频率的低通滤波器，常用于平滑 noisy 交互信号。
- **I-VT 分类器**：速度-阈值法在线凝视/扫视分割算法，连续超限判定为扫视，稳定样本数达标后输出质心。
- **GazeMedSeg**：基于专家红外眼动标注的医疗图像弱监督分割方法，将注视点卷积为高斯热图生成伪掩码。
- **Kernel Ridge Regression (KRR)**：核岭回归，FaceMesh+KRR 引擎将 13 维面部特征映射至屏幕坐标的核心回归器。

## 可复现要素
- **数据集**：Kvasir-SEG（公开）；自定义采集 CSV 日志与逐帧时间戳随源码发布。
- **代码/权重**：开源 TypeScript 单页应用、benchmark harness、`clock_probe.html`、分析脚本与附录热图（随论文附源仓库）。
- **关键超参**：WebGazer 3.4.0；MediaPipe FaceMesh 0.4.1633559619；One-Euro β=0.007；KRR λ=10⁻³；γ 由校准特征中位数两两距离启发式设定；I-VT 阈值 1200 px/s；平滑追踪校准延迟补偿 100 ms。
- **硬件环境**：14-inch MacBook Pro (M4, 2024)，集成 1080p 摄像头 30 Hz，Chrome arm64 / macOS Sequoia 15.6，视距 60 cm，80% 浏览器缩放。
