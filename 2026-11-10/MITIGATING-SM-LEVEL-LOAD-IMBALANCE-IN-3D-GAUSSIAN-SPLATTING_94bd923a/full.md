# MITIGATING SM-LEVEL LOAD IMBALANCE IN 3D GAUSSIAN SPLATTING ON GPUS

Junha Ryu <sup>1</sup> Hoi-Jun Yoo <sup>1</sup>

## ABSTRACT

3D Gaussian Splatting (3DGS) enables photorealistic, real-time rendering on commodity GPUs and is increasingly adopted in generative world models and autonomous driving simulators. However, existing optimizations rely on algorithmic changes that often require retraining, or on specialized accelerators that lack flexibility across evolving 3DGS variants. In this paper, we identify a fundamental inefficiency in GPU-based 3DGS rendering: severe SM-level load imbalance caused by uneven workloads across screen tiles. This imbalance creates a long execution tail, leaving most SMs idle while a few process computationally intensive tiles, particularly in large-scale urban scenes. To address this, we present a software-defined GPU scheduling approach that mitigates the execution tail. Our approach uses a persistent kernel to schedule heavy tiles first and selectively chunks outlier tiles across multiple CTAs. A lightweight per-frame policy determines when and how much to chunk using tile statistics already available in the rendering pipeline. Our approach requires neither retraining nor specialized hardware and can be applied orthogonally to existing optimizations across diverse 3DGS variants. Averaged per scene, it accelerates alpha blending by up to 2.93× on a large-scale urban scene, and by up to 5.6× on its most imbalanced frames, while retaining the original alpha blending formulation.

## 1 INTRODUCTION

3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has emerged as a widely adopted 3D scene representation, enabling photorealistic rendering at real-time frame rates. Its ability to represent both reconstructed and generated 3D scenes has driven its adoption across diverse applications, including generative world models (Shen et al., 2026; World Labs, 2025) and large-scale autonomous driving simulations (Chen et al., 2025). For instance, NVIDIA Omniverse NuRec (NVIDIA Corporation, 2025) reconstructs real-world driving scenes as 3DGS digital twins, which AlpaSim (Cao et al., 2025) uses to evaluate vision-languageaction (VLA) driving models such as Alpamayo (Wang et al., 2025).

As these applications grow in scale and complexity, efficient 3DGS rendering becomes critical. However, current optimization efforts concentrate at the two ends of the stack: algorithm-level optimizations and specialized hardware acceleration. On the algorithmic front, research has primarily focused on reducing the overall Gaussian count through pruning and budget-constrained densification (Fang & Wang, 2024; Hanson et al., 2025; Mallick et al., 2024). However, these approaches are typically integrated into the training process, making them difficult to apply to already reconstructed 3D scenes without costly retraining. Moreover, these reductions do not inherently resolve the underlying architectural inefficiencies on general-purpose GPUs. On the hardware front, specialized accelerators (Lee et al., 2024; Feng et al., 2025b) offer high energy efficiency but lack the flexibility to support evolving 3DGS variants, such as 2DGS (Huang et al., 2024) for better surface modeling and 3DGUT (Wu et al., 2025) for diverse camera-model support. Furthermore, modern simulation environments such as AlpaSim necessitate the seamless integration of 3DGS with heterogeneous DNN workloads (e.g., VLA models). In these scenarios, the GPU remains the only viable platform as it provides both programmable cores for flexible rendering and tensor cores for high-throughput neural inference.

Despite the central role of GPUs, system-level software optimizations for maximizing GPU utilization remain relatively underexplored. Existing tile-based rasterizers assign each screen tile to a GPU Cooperative Thread Array (CTA), even though workloads vary substantially across tiles. Consequently, a few computationally intensive tiles can continue executing long after most others have finished, leaving many Streaming Multiprocessors (SMs) idle. This long execution tail is particularly pronounced in large-scale urban and driving scenes, hindering real-time rendering performance.

In this paper, we present a software-defined GPU scheduling approach that mitigates the execution tail in 3DGS rasterization. Our approach employs a persistent kernel to dispatch tiles in longest-processing-time-first (LPT) order based on their Gaussian counts, allowing heavy tiles to start early. Since scheduling alone cannot shorten exceptionally heavy tiles, it chunks these outliers into smaller units processed in parallel across multiple CTAs. By exploiting the composability of alpha blending, it preserves the original rendering formulation with negligible numerical differences, while saturation signaling restores early termination. To limit chunking overhead, it selectively chunks heavy tiles and coexecutes them with LPT-scheduled tiles on separate CUDA streams. Finally, a lightweight online cost model determines how many tiles to chunk and how deeply for each frame, using Gaussian counts already available in the rendering pipeline.

Specifically, our contributions are as follows:

• Quantitative Analysis of GPU Load Imbalance: We characterize tile-level workload distributions and SM activity across diverse 3DGS variants and scenes, ranging from object-level benchmarks to city-scale environments. We demonstrate that a small number of computationally intensive tiles can dominate execution time, leaving most SMs idle and exposing a fundamental limitation of conventional GPU rasterization.

• LPT Scheduling via Persistent Kernel: We introduce workload-aware tile scheduling through a persistent kernel. By dispatching tiles in descending order of their projected Gaussian counts, our scheduler prioritizes heavy tiles and reduces the execution tail caused by late CTA dispatch.

• Depth-wise Chunking with Saturation Signaling: LPT scheduling changes when a tile executes, not how long it takes, leaving exceptionally heavy tiles on the critical path. We exploit the composability of alpha blending to partition these tiles into depth-contiguous chunks that execute in parallel across multiple CTAs. To avoid redundant computation, we introduce saturation signaling that restores early termination across chunks while retaining the original alpha blending formulation.

• Selective Chunking and Stream Co-execution: Chunking every tile incurs unnecessary overhead, particularly on relatively balanced workloads. We therefore chunk only the heaviest tiles and co-execute their chunks with LPT-scheduled tiles on separate CUDA streams. This allows the GPU hardware scheduler to dynamically allocate SM resources between the two workloads.

![](images/35cce9775463cf7f7e062a56f317343f1230493e343a6efe32a0a0114b71ebf7.jpg)  
Figure 1. Common rendering pipeline of 3DGS and its variants.

• Online Per-Frame Chunking Policy: We develop a lightweight cost model that balances execution-tail reduction against chunking overhead. Using per-tile Gaussian counts, the policy selects both the number of tiles to chunk and the chunk depth for each frame, including the option to disable chunking. This enables online adaptation without exhaustive configuration searches or scene-specific tuning.

• Performance Validation: We evaluate our approach across diverse 3DGS workloads, from object-level benchmarks to large-scale urban and driving scenes, achieving up to 2.93× speedup in alpha blending execution. Our approach supports multiple 3DGS variants, including 2DGS, 3DGUT, Hier-3DGS (Kerbl et al., 2024), and CityGaussianV2 (Liu et al., 2025). It can also be applied orthogonally to existing GPU rasterization optimizations, including SwiftGS (Gao et al., 2026) and HiGS (Paj ˛ak et al., 2026), without modifying scene representations or requiring retraining.

## 2 BACKGROUND

## 2.1 3D Gaussian Splatting and Its Variants

3DGS represents a 3D scene as a collection of anisotropic 3D Gaussians, each defined by a center $\mu ~ \in ~ \mathbb { R } ^ { 3 }$ , a covariance Σ parameterized by scale and rotation, an opacity $o \in [ 0 , 1 ]$ , and a color. As illustrated in Figure 1, the rendering pipeline consists of three stages: preprocessing, sorting, and alpha blending. During preprocessing, Gaussians outside the camera’s field of view are culled, and the remaining Gaussians are projected onto the image plane. Given a viewing transformation W and projection Jacobian J, the projected 2D covariance $\Sigma ^ { \prime }$ is:

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T } .\tag{1}
$$

The screen is partitioned into tiles, and each projected Gaussian with center $\mu ^ { \prime } \in \mathbb { R } ^ { 2 }$ is assigned to all overlapping tiles. During sorting, Gaussians within each tile are ordered front to back by camera-space depth. During alpha blending, each tile accumulates Gaussian contributions to its pixels. For a pixel coordinate $\boldsymbol { x } \in \mathbb { R } ^ { 2 }$ , the alpha value of Gaussian i is:

![](images/6ddba1877ed6aa4ff012099ff5e540530417b039f97e8795c7d2f5bbffc9fde6.jpg)  
Figure 2. Rendering-time breakdown across 3DGS, 2DGS, and 3DGUT.

$$
\alpha _ { i } = o _ { i } \cdot \exp \left( - \frac { 1 } { 2 } ( x - \mu ^ { \prime } ) ^ { T } \Sigma ^ { \prime - 1 } ( x - \mu ^ { \prime } ) \right) .\tag{2}
$$

Then, the final pixel color C is accumulated as:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } T _ { i } , \qquad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{3}
$$

Here, $c _ { i }$ denotes the color of Gaussian $i ,$ and $T _ { i }$ is the remaining transmittance before processing it. Since transmittance decreases monotonically, a pixel stops blending once $T _ { i }$ falls below $1 0 ^ { - 4 }$ , where subsequent contributions become negligible. This early termination allows a tile to finish before processing all its assigned Gaussians.

Beyond conventional 3DGS, 2DGS (Huang et al., 2024) uses oriented planar disks and computes $\alpha _ { i }$ via ray–splat intersection for accurate surface reconstruction. 3DGUT (Wu et al., 2025) replaces the linearized projection of Eq. (1) with the unscented transform to support distorted cameras and secondary rays. Although these variants differ in how $\alpha _ { i }$ is computed, they retain the same tile-based alpha blending of Eq. (3). As shown in Figure 2, alpha blending accounts for a substantial fraction of rendering time across these variants, making it a common target for optimization.

## 2.2 GPU Architecture and CTA Scheduling

Figure 3 illustrates the GPU execution model relevant to 3DGS rasterization. Modern GPUs consist of multiple SMs, each equipped with CUDA cores, registers, and shared memory. When a kernel is launched, its threads are organized into CTAs, each consisting of one or more 32-thread warps. The hardware CTA scheduler (GigaThread Engine in NVIDIA GPUs) assigns CTAs to available SMs based on resource requirements. Each CTA executes on a single SM, while multiple CTAs can reside on the same SM concurrently. Kernels launched in separate CUDA streams may also execute concurrently when GPU resources permit.

![](images/63962216329959d072e1a0b2dea7633afcf755216e4b1f4013ae0448f950e46d.jpg)  
Figure 3. GPU execution model and CTA scheduling across CUDA streams and SMs.

In conventional tile-based 3DGS rasterization, each $1 6 \times$ 16-pixel tile is mapped to one 256-thread CTA, with each thread computing the final color of one pixel. Since the hardware scheduler operates at CTA granularity, it cannot redistribute a tile’s workload across multiple SMs once its CTA is scheduled. Consequently, large variations in tile workloads can leave a few long-running CTAs executing while other SMs become idle. This rigid mapping between screen tiles and CTAs limits GPU utilization, motivating more flexible workload scheduling and parallelization.

## 3 CHALLENGES AND OPPORTUNITIES

## 3.1 Workload Imbalance in Tile-Based Rendering

Prior work accelerates 3DGS rendering by reducing Gaussian processing through pruning or tighter bounding volumes (Hanson et al., 2025; Feng et al., 2025a). While these optimizations reduce overall computation, they do not directly address how the remaining workload is distributed across GPU CTAs. To characterize this distribution, we measure per-tile Gaussian counts across 19 scenes spanning standard benchmarks (Tanks & Temples, Deep Blending, and Mip-NeRF 360) and large-scale urban and driving scenes. Figure 4 reveals highly skewed workload distributions. Standard benchmarks exhibit maximum-to-minimum count ratios of 55–605×, while large-scale scenes reach over 6,100×. However, these counts are only a proxy for blending work, since early termination can skip Gaussians. More importantly, a large skew alone does not imply low GPU utilization.

## 3.2 SM Utilization Across 3DGS Variants

We now examine when this skew translates into GPU underutilization. Even with highly uneven tiles, the GPU remains busy as long as enough CTAs are available; a long execution tail arises only when the heaviest tiles are large relative to the GPU’s parallel capacity and keep executing after most other CTAs have completed. To quantify this effect, let $N _ { t }$ denote the projected Gaussian count of tile $\begin{array} { r } { t , N _ { t o t a l } = \sum _ { t } N _ { t } . } \end{array}$ , and K the number of CTAs the GPU can execute concurrently. Under an ideal distribution, each CTA would process a share of $N _ { t o t a l } / K$ Gaussians. Since conventional rasterization assigns each tile to a single CTA, a tile much heavier than this share can dominate the critical path, as illustrated in Figure 5(a). We use the ratio of the heaviest tile to this share as a proxy for the execution tail.

![](images/79d47b24ee4659b75d56373210ea00e808eeb1974f76dd7f3839599c9289dd9d.jpg)  
Figure 4. Distribution of per-tile Gaussian counts across small-scale and large-scale scenes, the latter shaded blue. Each raincloud plot shows the density (violin), interquartile range (box), and per-tile samples (dots). Driving scenes use the front camera.

![](images/906494e92166361a296a82d7595f9734d543a6ef0232289217485093898c42c2.jpg)  
(b)  
Figure 5. (a) Execution tail caused by tile imbalance. (b) Pertile Gaussian counts and corresponding SM activity timelines for representative frames, sampled with Nsight Compute (NVIDIA Corporation, 2026).

Figure 5(b) compares this ratio with measured SM activity. In Street Gaussians (Yan et al., 2024), the heaviest tile is only 1.1× the per-CTA share, and average SM activity remains high at 88.2%. In contrast, the 3DGS train scene exhibits a ratio of 1.7×, with average SM activity of 49.2%. The imbalance is even more pronounced in Hier-3DGS, where the heaviest tile reaches 6.8× the share and average SM activity drops to 24.9%. In the latter two, only a few SMs remain active near the end of the frame.

![](images/f6732e96b130ac2b7ac3a0e42834139178ba028143cfa83797688cb4bcdb2058.jpg)  
Figure 6. CDF of per-frame SM activity across 3DGS variants and datasets (front camera for driving scenes), measured with Nsight Compute.

Figure 6 extends this analysis to six dataset categories. Here, SM activity is the fraction of elapsed cycles during which an SM has at least one resident warp; it reflects SM idleness rather than instruction throughput or occupancy. Standard benchmarks mostly sustain high utilization, with some underutilized frames. Among large-scale scenes, Hier-3DGS exhibits low utilization across many frames, while the nuScenes front camera falls below 80% in nearly half of its frames. In contrast, Street Gaussians stays highly utilized in most frames despite substantial tile skew. Underutilization thus varies across scenes and even across frames of the same scene, so load balancing must account for both the heaviest tiles and the available GPU parallelism rather than applying one strategy to every frame.

## 3.3 Opportunities

CTA Scheduling via Persistent Kernel. The hardware CTA scheduler is unaware of tile workloads, so a heavy tile may start late and finish after all others. However, the per-tile Gaussian count available after sorting can serve as a workload estimate. A persistent kernel (Gupta et al., 2012) allows a fixed pool of CTAs to repeatedly fetch tiles from a software-managed work queue, enabling workload-aware execution ordering. Persistent-kernel approaches have also been explored in LLM workloads, including FlashAttention-3 (Shah et al., 2024) and FlashInfer (Ye et al., 2025b), while Mirage Persistent Kernel (MPK) (Cheng et al., 2026) extends persistent execution to an entire inference pass.

Alpha Blending Composability. Scheduling changes when a tile executes, but cannot shorten the workload of a tile assigned to a single CTA. Alpha blending’s composability removes this limitation. Consider partitioning a tile’s depth-ordered Gaussian list before Gaussian $m .$ . The front segment (Gaussians $1 , \ldots , m - 1 )$ produces a partial color $C _ { A }$ and remaining transmittance $T _ { m }$ as defined in Eq. (3), while the back segment independently computes $C _ { B }$ assuming unit initial transmittance. The results can be combined as

$$
C = C _ { A } + T _ { m } \cdot C _ { B } .\tag{4}
$$

This identity allows a tile’s depth-ordered workload to be divided into contiguous chunks that execute independently across multiple CTAs, while retaining the original alpha blending formulation. Since 2DGS, 3DGUT, and large-scale variants such as Hier-3DGS and CityGaussianV2 (Liu et al., 2025) share this compositing structure, the same opportunity extends across diverse 3DGS renderers.

## 4 METHOD

Our approach mitigates GPU execution tails through workload-aware tile scheduling and selective intra-tile parallelism. We first use a persistent kernel to dispatch tiles in longest-processing-time-first (LPT) order based on their Gaussian counts (§4.1). For outlier tiles that remain bottlenecks under LPT, we introduce depth-wise chunking with saturation signaling to parallelize their execution across multiple CTAs while reducing redundant computation (§4.2). We co-execute chunked and LPT-scheduled tiles on separate CUDA streams (§4.3), while a lightweight per-frame cost model determines when and how much to chunk (§4.4). These optimizations preserve the original alpha blending formulation without modifying scene representations or requiring specialized hardware.

## 4.1 LPT Scheduling via Persistent Kernel

Conventional 3DGS rasterizers launch one CTA per tile and leave the dispatch order to the hardware scheduler, which is unaware of tile workloads. As a result, a heavy tile may start late and become the execution tail. To control the execution order, we instead launch a persistent kernel with K CTAs, the number the GPU can run concurrently. Each CTA then repeatedly takes the next tile from a software-managed work queue Q until the queue is empty, with one thread per CTA advancing the queue through a single atomicAdd.

![](images/eaecc1ac6c5da3472ff8ab80f34ade9daccc28887bb6aec6d2e6c81b58a7e3f6.jpg)

(a)  
![](images/5a1cfe9ba6391721861f8647462821d07eda74b616452ef45251527efbd003f0.jpg)

(b)  
![](images/17e920fad60aca3031fb11014b5c5a645657451c0061964a7f88e13c5f3f9f16.jpg)  
(c)  
Figure 7. (a) LPT scheduling via a persistent kernel. (b) SM activity before and after LPT scheduling for the frames in Figure 5(b). (c) LPT speedup over Vanilla versus the heaviest tile relative to the per-CTA share.

LPT ordering. We order Q so that the heaviest tiles are dispatched first. To estimate the workload of tile t, we use its projected Gaussian count $N _ { t } ,$ , which is directly available from the tile’s range in the sorted Gaussian list. We then radix-sort the tile indices in descending order of $N _ { t }$ without moving the Gaussian data. This counting and sorting takes less than 1% of the alpha blending kernel time on average across the three workloads in Figure 7(b). As illustrated in Figure 7(a), the heavy tile thus starts first while the other CTAs process lighter tiles, so the kernel finishes at $T _ { L P T }$ instead of $T _ { V a n i l l a }$

Effectiveness and limitation. Figure 7(b) shows that LPT shortens the alpha blending kernel of the profiled frame by 25% on 3DGS train and 14% on Street Gaussians, but by only 7% on Hier-3DGS. This is because the heaviest tile of Hier-3DGS is 6.8× the per-CTA share and still runs long after the other tiles finish, even when dispatched first. Figure 7(c) shows the same trend across frames. LPT achieves

![](images/e6b3f7fd71ab150892bc852bddfce32b2c27ebc3554b8d7e4d2a5eec2de85002.jpg)

![](images/a40cd0a608ca646cb1513a39ce1a541ef4cc19efaae06f182cb6ea4636c906a5.jpg)

![](images/6a60dcde42f2748b176da38b989bae9d2dcf6f2a5609a63e790eaa927dd577ef.jpg)

(a)  
![](images/78007d1d606004724bbc686be5b1b05a7930d5e56d0ec08c872d894eaba9a029.jpg)

![](images/31383d0841fc19ea6e1b3eb95c25582c22616f6f85258c4e3e6af3e238a3e173.jpg)  
(b)  
Figure 8. (a) Depth-wise chunking and saturation signaling. (b) Per-frame speedup of Chunking-only over LPT-only versus the number of chunks.

1.1–1.6× speedup on many standard benchmark frames. In contrast, Street Gaussians frames, whose heaviest tile mostly stays below the per-CTA share, have little tail to remove and gain only 1.06× on average. At the other extreme, most Hier-3DGS frames, whose heaviest tile far exceeds the share, gain only 1.10×. In other words, LPT changes when a tile executes but not how long it takes, so shortening such outlier tiles requires parallelism within a tile, which we introduce next.

## 4.2 Depth-wise Chunking with Saturation Signaling

To shorten an exceptionally heavy tile, we partition its depthordered Gaussian list into k contiguous chunks, each processed by a separate CTA. Each chunk independently performs alpha blending with unit initial transmittance and produces a partial color $C _ { p }$ and transmittance $T _ { p }$ for each pixel. A reduction kernel then combines these partial results in depth order:

$$
C = \sum _ { p = 1 } ^ { k } C _ { p } \prod _ { q = 1 } ^ { p - 1 } T _ { q } .\tag{5}
$$

This extends Eq. (4) to k chunks, allowing their blending computations to execute independently across multiple SMs (Figure 8(a)). Similar to Flash-Decoding (Dao et al., 2023), this approach parallelizes a long sequential computation across CTAs and combines partial results, although 3DGS requires per-pixel color and transmittance for composition.

Saturation signaling. Independent chunk execution disrupts early termination because each chunk starts with unit transmittance, unaware of the opacity accumulated by preceding chunks. Consequently, later chunks may process Gaussians that the original sequential renderer would skip. We address this through saturation signaling, which communicates early-termination information across independently executing CTAs (Figure 8(a)). Each chunked tile maintains a global-memory flag recording the earliest chunk whose pixels have all reached the transmittance threshold. Upon saturation, a chunk updates the flag using atomicMin. Later chunks check it once per 256-Gaussian batch and stop when an earlier chunk has saturated. This is safe because transmittance only decreases in depth order. If an earlier chunk alone has driven every pixel below the threshold, the true transmittance before any later chunk is even lower, so the sequential renderer would also have stopped before the skipped Gaussians. On 3DGS train, saturation signaling accelerates Chunking-only execution by 1.21× on average and up to 1.66× per frame.

Benefits and overheads. Depth-wise chunking shortens outlier tiles but incurs additional global-memory traffic for partial results and a reduction pass. Figure 8(b) evaluates this trade-off by chunking every tile and comparing against LPT-only execution. On Hier-3DGS, Chunkingonly achieves an average 1.9× speedup over LPT at k = 16, as distributing outlier workloads across CTAs reduces the execution tail. Conversely, on the relatively balanced 3DGS bicycle scene, it achieves only 0.68–0.79× the performance of LPT on average across chunk counts. Since bicycle has no extremely heavy tile, chunking barely shortens the frame, while every chunked tile still pays the extra memory traffic and reduction. Chunking therefore helps only for the heavy tiles that bound the frame, which motivates chunking them selectively rather than every tile.

## 4.3 Selective Chunking and Co-execution

LPT avoids late dispatch of heavy tiles, whereas chunking shortens individual outliers. Since neither performs best across all workload distributions, we combine both within a frame, chunking the heaviest tiles while scheduling the remainder in LPT order.

Stream co-execution. Given a heavy-tile ratio $\rho$ and chunk count k, we select the heaviest fraction $\rho$ of tiles from the existing LPT order and partitions each into k chunks. The remaining tiles are assigned to the persistent LPT kernel without additional sorting. As shown in Figure 9(a), the chunk kernel assigns one CTA per chunk, while the persistent kernel dispatches the remaining tiles in LPT order.

![](images/d0bef802755e62bc52c79f3f1b468f80d72ce04d5dc09b3edc4a29a5f3f09d39.jpg)

(a)  
![](images/c0235610654118658f595e2673268b3a6bc98336b79eb970d1ad1f148eaef276.jpg)

![](images/9f91328ce9cec563b62a9f4b1e1fd697b7089fd52670a79d0e841e06d1488d08.jpg)  
(b)  
Figure 9. (a) Stream co-execution of chunk and LPT kernels. (b) Per-frame speedup of co-execution over LPT-only (left) and Chunking-only (right).

The kernels execute on separate CUDA streams, allowing their CTAs to share SM resources as hardware scheduling and resource availability permit. After both kernels complete, a reduction kernel combines the partial results, with dependencies enforced through CUDA events.

Effectiveness of co-execution. Figure 9(b) compares coexecution with LPT-only and Chunking-only using the best fixed $( \rho , k )$ configuration for each workload. On Hier-3DGS small city, co-execution outperforms Chunking-only by 12– 75% across frames. On highly imbalanced 3DGS train frames, it is up to 34% faster than LPT-only. However, chunking even the heaviest 1% of tiles slows relatively balanced 3DGS train frames by up to 6%. The best fixed configuration also differs between 3DGS train (1%, 2) and Hier-3DGS (2%, 8). These results motivate adapting both the number of chunked tiles and the chunk count to each frame.

## 4.4 Per-Frame Chunking Policy

Since tile imbalance varies from frame to frame, the best $( h , k )$ must be chosen at runtime without searching over configurations. Chunking too few tiles leaves outliers on the critical path, whereas chunking too many adds unnecessary overhead. We therefore use a lightweight cost model to select two parameters on the fly for each frame: the number of heaviest tiles to chunk, h, and the number of chunks per tile, k. The policy uses only the Gaussian counts already available from $\mathrm { L P T }$ sorting.

Cost estimation model. Let $N _ { t }$ denote the projected Gaussian count of tile $t ,$ indexed so that $N _ { 1 } \geq N _ { 2 } \geq \cdots$ , and let $\begin{array} { r } { N _ { t o t a l } = \sum _ { t } N _ { t } } \end{array}$ . The model balances two competing terms. First, chunking h tiles into k chunks each adds an overhead of $\lambda h k .$ , where λ is a constant modeling the per-chunk cost of storing and reducing partial results. Distributed across K concurrent CTA workers, this yields an average-workload term of $( N _ { t o t a l } + \lambda h k ) / K$ . Second, execution time is constrained by the longest remaining work unit. After chunking the h heaviest tiles, we estimate its size as

![](images/8be4a6dba77219003bcecb42c640bfdae7f7b95f31df0a0d4d60197ee1816423.jpg)

![](images/fa2c0d8b2953ee1555d6004fa0cb1de1dc8cbb90c0aca1013986d79257a8ac5e.jpg)  
Figure 10. Per-frame chunking cost estimation on a 3DGS train frame. Left: Workload distribution after chunking. Right: Estimated cost versus the number of chunked tiles.

$$
\ell ( h , k ) = \operatorname* { m a x } \left( \frac { N _ { 1 } } { k } , N _ { h + 1 } \right) ,
$$

where the two terms represent the largest chunked workload and the heaviest unchunked tile. Adding a penalty βℓ for residual scheduling imbalance gives

$$
\hat { T } ( h , k ) = \operatorname* { m a x } \left( \frac { N _ { t o t a l } + \lambda h k } { K } , \ell \right) + \beta \ell .\tag{6}
$$

$\hat { T }$ is a heuristic cost expressed in Gaussian-equivalent work units, rather than a direct prediction of measured latency.

Choosing h and k. For each candidate $k ,$ the policy evaluates increasing h in descending workload order. Each additional chunked tile can reduce ℓ but raises the averageworkload term by $\lambda k / K$ . Figure 10 illustrates this trade-off for $k \ = \ 4$ on a 3DGS train frame. Chunking the first 24 tiles reduces the critical-path term; beyond this point, further chunking lowers the $\beta \ell$ penalty until the estimated cost reaches its minimum at $h = 7 9$ . The policy searches $k \in \{ 2 , 4 , 8 , 1 6 \}$ and selects the $( h ^ { * } , k ^ { * } )$ minimizing $\hat { T } _ { : }$ including the no-chunking option $( h \ : = \ : 0 )$ . To account for workload overestimation caused by early termination, chunking is enabled only when the estimated cost is at least $\epsilon = 1 0 \%$ lower than LPT-only.

Effectiveness of the online policy. Figure 11 compares our policy with two offline baselines that are chosen in hindsight after measuring a set of candidate $( \rho , k )$ configurations on every frame. The best fixed baseline applies the single configuration with the highest average speedup to all frames of a scene, and the per-frame oracle picks the fastest configuration for each frame individually. Neither is available at runtime. Even so, our policy outperforms the best fixed baseline by 2.0% on 3DGS train and 2.7% on Hier-3DGS. On 3DGS room, it avoids chunking and remains within 0.6% of LPT-only. On Hier-3DGS, it performs within 2.0% of the per-frame oracle on average without any offline configuration search.

![](images/ebc013c991708b5efc6ab4e4e69309e89b167e5aacb64fd898974c34399aa3ca.jpg)

![](images/398644a148f931784f50d756968b7cae4cb515d80916d942b723d5e7f6a80f95.jpg)  
Figure 11. Effectiveness of the per-frame chunking policy.

## 5 EVALUATION

## 5.1 Experimental Setup

We evaluate our approach on an NVIDIA GB10 (48 SMs) and an NVIDIA RTX A6000 (84 SMs). We render all thirteen scenes of Mip-NeRF 360, Tanks & Temples and Deep Blending with 3DGS, four of them with 2DGS and 3DGUT, train and truck with SwiftGS (Gao et al., 2026) and HiGS (Paj ˛ak et al., 2026), small city with Hier-3DGS, and the MatrixCity aerial and street scenes with the 2DGS-based CityGaussianV2 (Liu et al., 2025). Through gsplat (Ye et al., 2025a), we render the front camera of 15 driving sequences from nuScenes, Argoverse 2, PandaSet and Waymo reconstructed by OmniRe (Chen et al., 2025), eight camera views of NVIDIA NuRec (NVIDIA Corporation, 2025) scenes with 3DGUT, and three scenes generated by Lyra 2.0. We compare Vanilla, the original rasterizer; LPT-only (§4.1); Chunking-only, which chunks every tile (§4.2); and Ours, which combines LPT dispatch with selective chunking under the per-frame policy (§4.3). We report the alpha blending speedup over Vanilla and image quality in PSNR, SSIM and LPIPS.

## 5.2 Overall Performance

Rendering Performance. As shown in Figure 12, our approach achieves an average speedup of 1.23× on the driving scenes and 1.12× on the standard benchmarks, and up to 3.20× on individual frames. On the standard benchmarks, most of the gain comes from LPT scheduling. On the most imbalanced driving scenes, selective chunking is essential. For example, on nuScenes scene 005, our approach reaches 1.77×, whereas LPT-only achieves 1.06×. Chunking-only, in contrast, adds overhead on relatively balanced scenes, which the per-frame policy avoids.

Table 1. Quality difference of Chunking-only from Vanilla.
<table><tr><td>Renderer</td><td>Dataset</td><td>#Scene</td><td>∆PSNR</td><td>∆SSIM</td><td>∆LPIPS</td></tr><tr><td rowspan="3">3DGS</td><td>M360</td><td>9</td><td>+0.0001</td><td>0.0000</td><td>0.0000</td></tr><tr><td>T&amp;T</td><td>2</td><td>-0.0001</td><td>0.0000</td><td>0.0000</td></tr><tr><td>DB</td><td>2</td><td>-0.0004</td><td>0.0000</td><td>0.0000</td></tr><tr><td rowspan="2">2DGS</td><td>M360</td><td>3</td><td>+0.0002</td><td>0.0000</td><td>0.0000</td></tr><tr><td>T&amp;T</td><td>1</td><td>-0.0001</td><td>0.0000</td><td>0.0000</td></tr><tr><td rowspan="2">3DGUT</td><td>M360</td><td>3</td><td>+0.0002</td><td>0.0000</td><td>0.0000</td></tr><tr><td>T&amp;T</td><td>1</td><td>+0.0001</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Hier-3DGS</td><td>City</td><td>1</td><td>-0.0015</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Street GS</td><td>Waymo</td><td>2</td><td>-0.0002</td><td>0.0000</td><td>0.0000</td></tr></table>

Rendering Quality. LPT scheduling changes only the dispatch order of tiles, whereas depth-wise chunking reorders the floating-point accumulation within a tile. Table 1 therefore compares Vanilla with Chunking-only, which chunks every tile and thus changes the accumulation order the most. Across 24 scenes, the differences are at most 0.0015 dB in PSNR, $2 . 4 \times 1 0 ^ { - 5 }$ in SSIM and $7 \times 1 0 ^ { - 5 }$ in LPIPS. On every frame of the driving and NuRec workloads, our output also stays above 76 dB PSNR relative to the original renderer.

## 5.3 Generality and Applicability

3DGS Variants. Our approach applies to any renderer built on tile-based alpha blending, without modifying the scene representation or the trained model. Beyond 3DGS (Figure 12) and Hier-3DGS (2.93× on average and up to 5.64× per frame, Figure 11), it achieves 1.22–1.31× on the NuRec views rendered with 3DGUT and 1.10× on street with the 2DGS-based CityGaussianV2, while staying on par with Vanilla (1.01×) on aerial (Figure 13).

Combining with Optimized Rasterizers. SwiftGS (Gao et al., 2026) reduces the per-Gaussian blending cost, and HiGS (Paj ˛ak et al., 2026) renders fixed-size batches of each macro-tile’s Gaussian list on separate CTAs and merges them in a post-blending pass. Neither orders work by cost, and HiGS blends every batch even after its macro-tile saturates. We apply our approach unchanged to SwiftGS. For HiGS, whose lists are already chunked, we add saturation signaling across batches, dispatch earlier batches first, and render single-batch macro-tiles without the post-blending pass. As shown in Figure 14, our approach achieves 1.24× over SwiftGS (up to 2.09×) and 1.23× over HiGS (1.08– 1.43× on every frame) on train, and 1.08× and 1.01× on the less skewed truck.

GPU Architectures. The impact of the execution tail depends on the number of SMs: a GPU with more SMs finishes the light tiles sooner, while a heavy tile still takes just as long. We therefore examine how effective our approach is on GPUs with different SM counts, comparing an RTX A6000 with 84 SMs and the GB10 with 48 SMs. As shown in Figure 15, on ten driving scenes, it achieves 1.13– 2.13× (1.45× on average) on the A6000, compared with 1.10–1.77× (1.32×) on the GB10. Our approach therefore becomes more beneficial as GPUs scale to more SMs.

![](images/f12cc121ba9a8e534ae358d6992e71f0273d47bd08cf49bc6231bae8268b2231.jpg)  
Figure 12. Speedup over Vanilla on driving scenes (top) and standard 3DGS benchmarks (bottom), per frame versus tile imbalance (left) and per scene (right).

![](images/debeefce977c4504cb14274418789c14fa8e44023d10c8d15f56cabb08f30df3.jpg)  
Figure 13. Speedup of our approach over Vanilla with CityGaussianV2 on MatrixCity and 3DGUT on NuRec views (CL: cross-left, FW: front-wide).

![](images/bf227400a3c9ff5f48590f09cbaefd4a1caa811f57fd98751390562d7d1d9a7e.jpg)

![](images/cf268e021113f7bebab5410fd7946656aad3c22cd04b01b010cfcfb263a50bb3.jpg)  
Figure 14. Per-frame speedup of our approach over SwiftGS and HiGS.

World-Model-Generated Scenes. Generative world models such as Lyra 2.0 (Shen et al., 2026) synthesize 3D scenes directly as Gaussians rather than reconstructing them from captured images, and their Gaussian density varies widely across viewpoints. As shown in Figure 16, our approach achieves 1.07–1.17× on average over the three scenes and up to 2.86× on the most imbalanced frames, whereas LPTonly achieves only 1.01–1.06×. The per-frame speedup closely follows the tile imbalance, indicating that its benefit depends on the workload distribution rather than on how the scene is constructed.

![](images/e6b0faad10041a25d68f42ce75aabe8c85ed01e1595898f29832df280cf806b0.jpg)  
Figure 15. Speedup of our approach over Vanilla on the GB10 (48 SMs) and RTX A6000 (84 SMs).

## 5.4 Design Alternatives

Gaussian Count as a Workload Proxy. LPT orders tiles by their projected Gaussian count, which ignores early termination. An offline oracle that orders tiles by the number of Gaussians actually processed, measured from a prior execution of the same frame, is only 1.3% faster, whereas reversing the order increases execution time by 37% (Figure 17). Since early termination only reduces work, projected Gaussian counts provide conservative workload estimates, enabling near-oracle LPT scheduling without runtime profiling.

Stream Co-execution versus a Fused Kernel. Alternatively, the chunk and LPT workloads can share one persistent kernel whose CTAs are partitioned between them, with work stealing when either queue empties. As shown in Figure 18, its latency is sensitive to the fraction of CTAs assigned to chunks, whose best value varies across workloads. Even with workload-specific tuning, fusion reduces latency by less than 1%, whereas a suboptimal partition increases it by up to 29%. Stream co-execution thus achieves comparable performance without partition tuning by leaving SM allocation to the hardware scheduler.

![](images/8529e7aa191b3ae2ce30825a08f1f13229c6ea09f203dce9eb70e9c85cb6821e.jpg)  
Figure 16. Per-frame speedup of our approach over Vanilla on Lyra 2.0-generated scenes.

![](images/a156be615dfa84665dd5799f683b911bf6f9512c3800c0ff1d9efc20c2086d7f.jpg)

![](images/ed47c76d6a16426776e0508de24cc7c8fbeb966d1a5fd9c8b9a65edcdd36b0e8.jpg)  
Figure 17. Gaussian count as an LPT workload proxy on 3DGS train. Left: kernel time by tile order. Right: processed versus projected Gaussians per tile.

Comparison with Mean-Threshold Tile Splitting. CaT-GS (Zhang et al., 2026) splits every tile exceeding twice the mean Gaussian count into mean-sized chunks and dispatches them after the regular tiles. This threshold depends only on the workload distribution, not on how many CTAs the GPU executes concurrently, so it can split far more tiles than needed. Each chunk adds partial-result traffic and reduction, and without saturation signaling, chunks keep blending Gaussians that sequential rendering would skip. Our approach instead chunks only when its cost model predicts a shorter critical path. As a result, it is 40.5% and 16.0% faster on average on 3DGS train and Hier-3DGS, respectively, with a maximum slowdown of 3.4% on any frame (Figure 19).

## 6 LIMITATIONS AND CONCLUSION

Limitations. Our approach targets the alpha blending stage of inference-time rendering, and its benefit scales with the tile imbalance of each frame. On balanced workloads, such as the Waymo and two of the PandaSet sequences, the execution tail is short and the room for improvement is limited (1.03–1.10× in Figure 12); there, the per-frame policy rarely chunks, and our approach performs close to LPT-only without slowing down Vanilla. The policy uses a single set of cost-model constants and projected Gaussian counts across all evaluated workloads. Calibrating these constants per GPU or learning them online could further narrow the gap to the per-frame oracle, which we leave to future work.

![](images/1f30a91f43da470c96c6c3f843a48ab245e4f1e40dd7462d8d3db15a42c913dd.jpg)

![](images/6629c6ceb345b9db51339e7f808defdc92b96731b13cf7a3648cf2d027711877.jpg)

Figure 18. Fused-kernel latency normalized to stream co-execution versus chunk-CTA ratio.  
![](images/83cad92eb1fbdf07d60fc805fcb621dffc3d824bc6dc2c2e73c5a535ce2a2153.jpg)

![](images/9db5c0706251e1fd3c7dab5671ff5df6b36a7eac6c9e4e34e930106d554067f1.jpg)  
Figure 19. Per-frame speedup of our approach over CaT-GS.

Conclusion. This paper identified SM-level load imbalance across screen tiles as a fundamental source of inefficiency in GPU-based 3DGS rendering. We presented a software-defined GPU scheduling approach that dispatches heavy tiles first through a persistent kernel, chunks outlier tiles across multiple CTAs with saturation signaling, and co-executes the chunks with LPT-scheduled tiles under a lightweight per-frame policy. Across 3DGS variants, optimized rasterizers, and GPU architectures, our approach accelerates alpha blending by up to 2.93× on a large-scale urban scene and by up to 5.6× on its most imbalanced frames, without retraining, modifying scene representations, or changing the alpha blending formulation. As 3DGS rendering scales to city-scale driving simulation and generative world models, our approach provides a practical path to efficient rendering on general-purpose GPUs.

## REFERENCES

Cao, Y., de Lutio, R., Fidler, S., Garcia Cobo, G., Gojcic, Z., Igl, M., Ivanovic, B., Karkus, P., Martinez Esturo, J., Pavone, M., Smith, A., Tanimura, E., Tyszkiewicz,

M., Watson, M., Wu, Q., and Zhang, L. AlpaSim: A modular, lightweight, and data-driven research simulator for autonomous driving. https://github.com/ NVlabs/alpasim, October 2025.

Chen, Z., Yang, J., Huang, J., de Lutio, R., Martinez Esturo, J., Ivanovic, B., Litany, O., Gojcic, Z., Fidler, S., Pavone, M., Song, L., and Wang, Y. OmniRe: Omni urban scene reconstruction. In International Conference on Learning Representations (ICLR), 2025.

Cheng, X., Zhang, Z., Zhou, Y., Ji, J., Jiang, J., Zhao, Z., Xiao, Z., Ye, Z., Huang, Y., Lai, R., Jin, H., Hou, B., Wu, M., Dong, Y., Yip, A., Ye, Z., Wang, S., Yang, W., Miao, X., Chen, T., and Jia, Z. MPK: A compiler and runtime for mega-kernelizing tensor programs. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26), pp. 1909–1926, 2026.

Dao, T., Haziza, D., Massa, F., and Sizov, G. Flash-Decoding for long-context inference. Stanford CRFM Blog, 2023. URL https://crfm.stanford.edu/ 2023/10/12/flashdecoding.html.

Fang, G. and Wang, B. Mini-Splatting: Representing scenes with a constrained number of Gaussians. In European Conference on Computer Vision (ECCV), pp. 165– 181. Springer Nature Switzerland, 2024. doi: 10.1007/ 978-3-031-72980-5 10.

Feng, G., Chen, S., Fu, R., Liao, Z., Wang, Y., Liu, T., Hu, B., Xu, L., Pei, Z., Li, H., Li, X., Sun, N., Zhang, X., and Dai, B. FlashGS: Efficient 3D Gaussian splatting for large-scale and high-resolution rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26652–26662, June 2025a.

Feng, X., Wang, H., Tang, C., Wu, T., Yang, H., and Liu, Y. 1.78mJ/frame 373fps 3D GS processor based on shapeaware hybrid architecture using earlier computation skipping and Gaussian cache scheduler. In 2025 IEEE International Solid-State Circuits Conference (ISSCC), pp. 1–3, 2025b. doi: 10.1109/ISSCC49661.2025.10904813.

Gao, L., Wang, Z., Mo, Z., and Fan, H. SwiftGS: Algorithm and system co-optimization for fast 3D Gaussian splatting on GPUs. In Proceedings of Machine Learning and Systems (MLSys), volume 8, 2026.

Gupta, K., Stuart, J. A., and Owens, J. D. A study of Persistent Threads style GPU programming for GPGPU workloads. In 2012 Innovative Parallel Computing (InPar), pp. 1–14, 2012. doi: 10.1109/InPar.2012.6339596.

Hanson, A., Tu, A., Lin, G., Singla, V., Zwicker, M., and Goldstein, T. Speedy-Splat: Fast 3D Gaussian splatting

with sparse pixels and sparse primitives. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 21537–21546, June 2025. doi: 10.1109/CVPR52734.2025.02006.

Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S. 2D Gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers, pp. 1– 11. Association for Computing Machinery, 2024. doi: 10.1145/3641519.3657428.

Kerbl, B., Kopanas, G., Leimkuhler, T., and Drettakis, G. 3D¨ Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, July 2023. doi: 10.1145/3592433.

Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., and Drettakis, G. A hierarchical 3D Gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4):1–15, July 2024. doi: 10.1145/3658160.

Lee, J., Lee, S., Lee, J., Park, J., and Sim, J. GSCore: Efficient radiance field rendering via architectural support for 3D Gaussian splatting. In Proceedings ofthe 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS), pp. 497–511. Association for Computing Machinery, 2024. doi: 10.1145/3620666.3651385.

Liu, Y., Luo, C., Mao, Z., Peng, J., and Zhang, Z. CityGaussianV2: Efficient and geometrically accurate reconstruction for large-scale scenes. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Mallick, S. S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F. V., and De La Torre, F. Taming 3DGS: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers. Association for Computing Machinery, 2024. doi: 10.1145/3680528.3687694.

NVIDIA Corporation. NVIDIA Omniverse NuRec. https://docs.nvidia.com/nurec/, 2025. Accessed: April 2026.

NVIDIA Corporation. NVIDIA Nsight Compute. https: //developer.nvidia.com/nsight-compute, 2026.

Paj ˛ak, D., Bisson, M., and Lima, R. HiGS: A hierarchical rendering architecture for real-time 3D Gaussian splatting. arXiv preprint arXiv:2606.00352, 2026.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 68658–68685, 2024.

Shen, T., Bahmani, S., He, K., Srinivasan, S. G., Cao, T., Ren, J., Li, R., Wang, Z., Sharp, N., Gojcic, Z., Fidler, S., Huang, J., Ling, H., Gao, J., and Ren, X. Lyra 2.0: Explorable generative 3D worlds. arXiv preprint arXiv:2604.13036, 2026.

Wang, Y., Luo, W., Bai, J., Cao, Y., Che, T., Chen, K., Chen, Y., Diamond, J., Ding, Y., Ding, W., Feng, L., Heinrich, G., Huang, J., Karkus, P., Li, B., Li, P., Lin, T.-Y., Liu, D., Liu, M.-Y., Liu, L., Liu, Z., Lu, J., Mao, Y., Molchanov, P., Pavao, L., Peng, Z., Ranzinger, M., Schmerling, E., Shen, S., Shi, Y., Tariq, S., Tian, R., Wekel, T., Weng, X., Xiao, T., Yang, E., Yang, X., You, Y., Zeng, X., Zhang, W., Ivanovic, B., and Pavone, M. Alpamayo-R1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

World Labs. Marble: A multimodal world model. https://www.worldlabs.ai/blog/ marble-world-model, November 2025. Accessed: April 2026.

Wu, Q., Martinez Esturo, J., Mirzaei, A., Moenne-Loccoz, N., and Gojcic, Z. 3DGUT: Enabling distorted cameras and secondary rays in Gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26036–26046, 2025. doi: 10.1109/CVPR52734.2025.02425.

Yan, Y., Lin, H., Zhou, C., Wang, W., Sun, H., Zhan, K., Lang, X., Zhou, X., and Peng, S. Street Gaussians: Modeling dynamic urban scenes with Gaussian splatting. In European Conference on Computer Vision (ECCV), pp. 156–173, 2024. doi: 10.1007/978-3-031-73464-9 10.

Ye, V., Li, R., Kerr, J., Turkulainen, M., Yi, B., Pan, Z., Seiskari, O., Ye, J., Hu, J., Tancik, M., and Kanazawa, A. gsplat: An open-source library for Gaussian splatting. Journal of Machine Learning Research, 26(34): 1–17, 2025a.

Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., and Ceze, L. FlashInfer: Efficient and customizable attention engine for LLM inference serving. In Proceedings of Machine Learning and Systems (MLSys), volume 7, 2025b.

Zhang, T., Chen, B., Liu, S., Wu, F., and Chen, G. CaT-GS: Efficient 3DGS rendering for large-scale scenes with inter-frame caching and tile scheduling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 37391–37400, June 2026.