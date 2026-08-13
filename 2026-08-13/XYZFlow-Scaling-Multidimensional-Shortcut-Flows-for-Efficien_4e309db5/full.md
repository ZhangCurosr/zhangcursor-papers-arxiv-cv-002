# XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling

Jinxiu Liu <sup>1</sup> <sup>\*</sup> Xuanming Liu <sup>2</sup> <sup>\*</sup> Kangfu Mei <sup>3</sup> Yandong Wen <sup>2</sup> Weiyang Liu <sup>1</sup>

spherelab.ai/xyzflow

## Abstract

High-fidelity image generation faces a trade-off between speed and quality. Diffusion models produce strong visuals but require costly iterative sampling. Existing efficient methods mainly distill pretrained models into few-step samplers, a challenging process that depends heavily on teacher-model quality. In this paper, we introduce XYZFlow, a framework that rethinks efficient generation through multidimensional scaling of flow matching. Unlike single-step mappings, XYZFlow enhances expressivity by making probability paths more identifiable and learnable through structured multidimensional conditioning. We view autoregressive modeling as implicit flow straightening, where richer context reduces trajectory ambiguity. XYZFlow realizes this idea through two orthogonal dimensions: temporal scaling, which uses non-Markovian conditioning on the full denoising history; and spatial scaling, enabled by Next Shortcut Prediction, which sequentially generates patches using preceding patches’ denoising trajectories as priors. Experiments show that XYZFlow achieves stateof-the-art performance, with 7.2-8.5× teacher speedups and competitive FID, while Next Shortcut Prediction delivers superior quality-latency trade-offs over model scaling or step reduction.

## 1. Introduction

Generative models, particularly diffusion probabilistic models, have revolutionized synthetic data generation across various modalities (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Rombach et al., 2022; Ho et al., 2020). The dominant paradigm involves a gradual forward process that incrementally corrupts data with noise, followed by a learned reverse process for iterative data reconstruction. While models like DDPM (Ho et al., 2020) and Score-SDE (Song et al., 2021b) achieve remarkable quality, this performance comes at a substantial computational cost (Lu et al., 2022b;a; Karras et al., 2022), often requiring hundreds of neural function evaluations per sample. Such a cost prevents these models from real-time applications.

The pursuit of efficiency has centered on a key insight: few-step generation quality fundamentally depends on the uniqueness of the noise-to-data trajectory mapping (Lipman et al., 2023; Frans et al., 2025; Boffi et al., 2024; Salimans & Ho, 2022). This uniqueness enables effective distillation by reducing the problem from learning complex distributions to fitting deterministic functions. Pioneering methods like Rectified Flows (Lipman et al., 2023), Consistency Models (Song et al., 2023) and Shortcut Models (Frans et al., 2025) address this by constructing straight, deterministic probability flows through novel training objectives. However, these approaches primarily focus on improvements to distillation algorithms themselves, which is a challenging and model-dependent endeavor (Salimans & Ho, 2022; Geng et al., 2024a; Sauer et al., 2024).

Despite recent progresses, we identify another fundamental challenge that remains largely unexplored: how can we scale the expressive power of generative models under strict sampling step constraints, without relying solely on distillation strategies? More profoundly, can we design probability flows that are intrinsically more unique and learnable through model architecture? As conceptually visualized in Figure 1(a), the ambiguous trajectories in conventional denoising stem from a lack of focused constraints.

In this paper, we introduce a new scaling paradigm. Instead of extensive scaling through added parameters or steps, we scale intensively by enhancing probability flow expressivity via structured, multidimensional conditionalization. We reinterpret autoregressive modeling not merely as a generative strategy, but as an implicit mechanism for flow enhancement and uniqueness enforcement (Li et al., 2024). The expanding autoregressive context imposes progressively specific constraints, reducing flow variance and straightening trajectories. This perspective inherits the insight from flow straightening methods (Lipman et al., 2023; Frans et al., 2025; Albergo & Vanden-Eijnden, 2023) where deterministic paths are crucial for efficient distillation, demonstrating that structured conditioning enforces such uniqueness.

![](images/a7bd0e5fe4d417e3c3f61b31ce02d129636e5f4b95a00aaf8de667514a9d260f.jpg)  
(b) Our Next Shortcut Prediction paradigm  
Figure 1. (a) Conventional one-shot denoising suffers from overlapping and ambiguous probability paths (blurred results) as the model attempts to denoise the entire image at once. (b) Our Next Shortcut Prediction paradigm: Denoising proceeds sequentially patch-by-patch (e.g., for patches I, C, M, L). The rightward small arrows trace the denoising trajectory of each patch over time. Crucially, the downward blue arrows transfer the complete denoising trajectory of the preceding patch as a powerful prior. This allows subsequent patches to leverage the established context, straightening their paths and requiring fewer denoising steps (longer horizontal sequences) to achieve high fidelity.

Guided by this insight, we propose XYZFlow, a framework that scales flow matching along two orthogonal dimensions for high-fidelity few-step generation, complementary to the prevailing path of distillation-based step compression. (1) Temporal Scaling: We condition each flow step on the complete history of previous states, creating a temporal coordinate system that straightens trajectories. This transforms denoising from Markovian to non-Markovian, where the KV cache of past states acts as a conditioning anchor, inspired by recent advances in recurrent diffusion processes (Hang et al., 2025). (2) Spatial Scaling: We propose Next Shortcut Prediction based on next-patch prediction. By dividing images into grids $( e . g . , 2 \times 2 )$ , this mechanism sequentially generates patches. Unlike standard patch-wise generation that treats patches independently, our method explicitly transfers the denoising trajectory (not just the final output) as an effective prior for subsequent patches. As illustrated in Figure 1(b), the full denoising trajectory of the first patch serves as a conditional guidance, enabling faster generation without any quality loss.

In principle, we demonstrate that multidimensional scaling equips probability paths with high-dimensional coordinate systems. While flow straightening methods seek unique points in one-dimensional flows, our approach enhances uniqueness through orthogonal dimensional coordinates. We ground our method in two crucial principles: (1) strengthening conditions can drive reverse process variance toward zero, ensuring mapping uniqueness (Song et al., 2023; Song & Dhariwal, 2024; Geng et al., 2024b; Lu & Song, 2025; Yang et al., 2024); (2) autoregressive trajectories of preceding patches can effectively guide subsequent predictions, straightening paths in few-step regimes (Hang et al., 2025; Yan et al., 2025; Wang et al., 2024; Ren et al., 2024). Our contributions are summarized below:

• Novel Scaling Paradigm. We propose to scale generative models by enhancing probability flow expressivity through structured multidimensional conditionalization, advocating that scaling constraint dimensionality provides a principled path to mapping uniqueness.

• We introduce XYZFlow, a practical framework that implements both temporal and spatial flow scaling, featuring Next Shortcut Prediction for efficient inference-time cross-patch implicit trajectory guidance.

• Strong Theoretical Justification. We establish a theoretical framework that formalizes autoregressive modeling as explicit flow enhancement, and empirically demonstrate competitive few-step generation on ImageNet 256×256, highlighting dimensional scaling as a promising alternative to conventional approaches.

## 2. Related Work

Few-step Diffusion and Flow Matching. Diffusion models (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song et al., 2021b;a) and their flow matching extensions (Lipman et al., 2023; Albergo & Vanden-Eijnden, 2023; Liu et al., 2022b) have established a powerful framework for generative modeling. Current research on efficient sampling primarily follows a path of extensive scaling, focusing on refining distillation algorithms or training objectives. Distillation-based methods (Salimans & Ho, 2022; Geng et al., 2024a; Sauer et al., 2024; Luo et al., 2024; Yin et al., 2024) aim to compress pre-trained models, while consistency-type approaches (Song et al., 2023; Song & Dhariwal, 2024; Geng et al., 2024b; Lu & Song, 2025; Yang et al., 2024) enforce self-consistency constraints along trajectories. Recent works like Flow Map Matching (Boffi et al., 2024) and Shortcut Models (Frans et al., 2025) further explore self-consistency principles to straighten probability paths, with Inductive Moment Matching (Zhou et al., 2025) modeling the self-consistency of stochastic interpolants at different time steps. While these methods have advanced the state-of-the-art performance, their reliance on distillation algorithm improvements represents a form of extensive scaling that faces fundamental challenges in model dependency and optimization complexity. In contrast, our work addresses the core challenge of trajectory uniqueness with intensive scaling, which enhances flow expressivity through multidimensional conditionalization rather than pursuing better distillation strategies for existing flows.

Autoregressive Models for Visual Generation. Autoregressive image generation has evolved from discrete tokenization (Razavi et al., 2019; Esser et al., 2021; Lee et al., 2022) to continuous representations that avoid quantization errors (Li et al., 2024; Ren et al., 2024; 2025; Wu et al., 2025). Methods like MAR (Li et al., 2024) and DISA (Zhao et al., 2025) combine autoregressive modeling with diffusion processes, while acceleration techniques focus on caching (Yan et al., 2025) or speculative decoding (Wang et al., 2024). FAR (Hang et al., 2025) replaces the diffusion head of MAR (Li et al., 2024) with a shortcut model, accelerating through architectural changes. In a concurrent work, DART (Gu et al., 2025) and ARD (Kim et al., 2025) employ an autoregressive model that sequences 2D token maps from the diffusion process. However, unlike DART which is a standard multi-step diffusion model and only apply autoregression in denoising dimension, we reinterpret autoregressive modeling as an implicit mechanism for flow enhancement and uniqueness enforcement not only in denoising dimension but also in spatial dimension, and propose a distillation method XYZFlow that operates in the low-step regime. Specifically, XYZFlow reduces inference steps to 3-4 by following the backward ODE trajectories of a pre-trained diffusion model, while DART uses the forward noising process on ground truth data.

## 3. The Proposed XYZFlow Framework

We introduce XYZFlow, a framework that enhances the expressivity of flow models through multidimensional conditioning. Motivated by the view that autoregressive modeling provides an effective mechanism for flow straightening by incrementally imposing constraints, we propose a new training objective, Next Shortcut Prediction, which enables efficient generation via multidimensional conditioning. We interpret the expanding autoregressive context as a sequence of increasingly specific and gradually stronger constraints that reduce variance in the probability flow and straighten trajectories. This perspective extends existing insights from flow straightening methods by showing how structured conditional information can better promote path uniqueness. Within this framework, Next Shortcut Prediction operationalizes the principle of intensive scaling, as it leverages spatial constraints to construct a high-dimensional coordinate system that effectively enforces flow uniqueness and yields straighter trajectories.

## 3.1. Autoregressive Modeling as Flow Enhancement

Traditional autoregressive approaches frame image generation as a sequence of conditional predictions. Given an image divided into patches $\langle \mathbf { x } ^ { 1 } , \mathbf { x } ^ { 2 } , \ldots , \mathbf { x } ^ { P } \rangle$ , the autoregressive generation process is formulated as the chain rule:

$$
p ( \mathbf { x } ^ { 1 } , \mathbf { x } ^ { 2 } , \ldots , \mathbf { x } ^ { P } ) = \prod _ { p = 1 } ^ { P } p ( \mathbf { x } ^ { p } \mid \mathbf { x } ^ { 1 } , \ldots , \mathbf { x } ^ { p - 1 } ) .\tag{1}
$$

While this formulation is mathematically sound, we reconceptualize it through the lens of flow enhancement. The growing context $\mathbf { x } ^ { 1 } , \ldots , \mathbf { x } ^ { p - 1 }$ acts as a set of progressively stronger constraints, which can gradually reduce the variance of the conditional distribution $p ( \mathbf { x } ^ { p } | \cdots )$ and straightens the probability flow path from noise to data. This conceptual shift allows us to leverage autoregressive structure not just for sequential prediction, but for intrinsically making the flow more unique and deterministic. This insight has been used for ill-posed 3D reconstruction (Liu et al., 2022a; Wen et al., 2021) and video generation (Liu et al., 2026).

Formally, we define flow enhancement as the process where conditional information $C$ transforms a base probability flow $p ( \mathbf { x } )$ into a conditioned flow $p ( \mathbf { x } | C )$ with reduced path variance: $\mathbb { V } [ \mathbf { x } _ { t } | C ] < \mathbb { V } [ \mathbf { x } _ { t } ] .$ , leading to straighter and more deterministic trajectories. This variance reduction directly contributes to mapping uniqueness. For further theoretical justification, please refer to the Appendix.

## 3.2. A Motivating Observation from Progressive Constraint Strengthening

Our approach is motivated by the empirical observation that as more patches are generated, the conditional distribution becomes more constrained, making subsequent patches easier to sample. As illustrated in Figure 2, this observation manifests in three important phenomena:

Next patches can be better predicted at later generation stages. When we probe the conditional strength by predicting $\mathbf { x } ^ { p }$ based on the representation of previously generated patches, predictions for early positions are blurry and lack detail, while predictions for later positions become increasingly precise. This demonstrates that accumulated context provides stronger conditional guidance, as shown in the panda image sequence (top-right of Figure 2).

The variance of latent patch distributions decreases for later patches. When sampling multiple possible patches for each position during generation, the variance among sampled patches is high for early positions but decreases significantly for later positions, indicating a more concentrated and simpler distribution under stronger conditioning. This is visually validated by the progression from noisy to clean patches in the bottom rows of Figure 2.

![](images/5631b655cac3013cae6375a7583153b063e61ea1775476c68b3694e20674ec12.jpg)  
Figure 2. Next Shortcut Prediction in XYZFlow. (Top-Left) Flow diagram showing the generation sequence, where a blue curve represents progressively strengthening constraints. (Top-Right) Visualization of a non-uniform patch-based denoising process: the first image patch undergoes the most denoising steps, while subsequent patches are generated with fewer steps (“shortcuts”). This forms a long autoregressive sequence where the denoising flow from prior patches (green and blue arrows) guides the denoising of subsequent ones.

Denoising paths become straighter for later patches. Following Rectified Flow, we measure path straightness using:

$$
S ( \{ \mathbf { x } _ { t } \} _ { t = 0 } ^ { 1 } , \mathbf { z } ) = \mathbb { E } _ { t \sim [ 0 , 1 ] } \left[ \| ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) - \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } \mid t , \mathbf { z } ) \| ^ { 2 } \right] .\tag{2}
$$

Our experiments have demonstrated that S decreases for patches generated later in the sequence, validating that strong contextual conditioning can effectively straighten the flow. The blue curve in Figure 2 (on the left) illustrates this path straightening effect.

## 3.3. Multidimensional Conditioning

Building on these observations, XYZFlow implements a dual-path conditioning architecture that enhances the probability flow along both temporal and spatial dimensions through Next Shortcut Prediction.

Temporal Conditioning: Intra-patch Trajectory Conditioning. To strengthen the conditioning along the temporal dimension, we propose to enhance the flow matching process for each patch by conditioning it on its own generation history. Specifically, for a patch $\mathbf { x } ^ { p } .$ , the conditioning signal at time t is its entire state trajectory from the beginning of generation up to, but not including, the current time t. We denote this generation history as $\mathcal { H } _ { t } ^ { p } = \{ \mathbf { x } _ { \tau } ^ { p } \} _ { \tau = 0 } ^ { t - \Delta t }$ . The temporal conditioning loss for the patch $\mathbf { x } ^ { p }$ is defined as the deviation of the predicted flow from the true conditional flow, given this patch’s historical context $\mathcal { H } _ { t } ^ { p }$ . Therefore, the loss can be formulated as follows:

$$
\mathcal { L } _ { \mathrm { t e m p } } ^ { p } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } ^ { p } , \mathbf { x } _ { 1 } ^ { p } } \Vert v _ { \theta } ( \mathbf { x } _ { t } ^ { p } \vert t , \mathcal { H } _ { t } ^ { p } ) - ( \mathbf { x } _ { 1 } ^ { p } - \mathbf { x } _ { 0 } ^ { p } ) \Vert ^ { 2 } ,\tag{3}
$$

where this self-conditioning with self attention can act as a strong generative prior, stabilizing the generation path by providing a temporal coordinate system for the flow.

Spatial Conditioning: Inter-patch Trajectory Conditioning. In the spatial dimension, the generation of each patch is conditioned on the complete trajectories of all previously generated patches. Unlike intra-patch temporal conditioning, which operates within the history of a single patch, spatial conditioning captures dependencies across patches. As illustrated in Figure 3, the key innovation is that each patch depends not only on the final content of preceding patches, but also on their full generation trajectories. This provides a substantially richer contextual signal, thereby enhancing flow expressivity across the spatial domain:

$$
p ( \mathbf { x } ^ { p } | \mathbf { x } ^ { 1 } , \ldots , \mathbf { x } ^ { p - 1 } ) = p ( \mathbf { x } ^ { p } | \mathcal { T } _ { < p } )\tag{4}
$$

where $\mathcal { T } _ { < p } = \{ \tau ^ { 1 } , \tau ^ { 2 } , \dots , \tau ^ { p - 1 } \}$ and $\tau ^ { i } = \{ \mathbf { x } _ { t } ^ { i } \} _ { t = 0 } ^ { 1 }$ represents the complete generation trajectory of patch i. Conditioning on full trajectories $\mathcal { T } _ { < p }$ rather than just final states provides significantly stronger constraints: each trajectory $\tau ^ { i }$ adds multiple temporal anchor points that collectively reduce the solution space for generating $\mathbf { x } ^ { p }$ , making the reverse process more deterministic. The attention patterns shown in Figure 3 demonstrate how different attention mechanisms can effectively integrate trajectory condition.

## 3.4. Next Shortcut Prediction

The core contribution of XYZFlow is Next Shortcut Prediction, which shifts the scaling paradigm from resourceintensive scaling to constraint-intensive scaling. Rather than increasing model size or the number of training steps, we scale the dimensionality of the conditioning constraints by training the model to generate effectively under progressively stronger constraints from $\mathcal { T } _ { < p }$ . As illustrated in Figure 2, Next Shortcut Prediction trains the model to exploit rich contextual information for accelerating the generation process. Specifically, we define a progressive denoising schedule where each patch p is assigned fewer denoising steps as it is generated later in the sequence:

![](images/ff62ab2ac6dcdd9dbbe8b9f49ad6053bc043a1f36c1e1010db986f7a7e7f9803.jpg)

![](images/05b40a198626341ba6255df4b94c178b356d672b7c38b5f46399662527134282.jpg)  
(d)  
Figure 3. Illustration of attention mechanisms for image generation. (a) Vanilla Image Generation: Standard full-image denoising with independent patch processing. (b) Autoregressive in Denoising Dimension: Sequential denoising across patches over time. (c) Next Patch Prediction: Complete denoising of one patch before starting the next. (d) Next Shortcut Prediction: Early patches undergo more denoising steps, with full denoising trajectories of previous patches conditioning subsequent ones.

$$
T ( p ) = T _ { \mathrm { f u l l } } - \Delta T \cdot ( p - 1 ) \quad \mathrm { f o r } \quad p = 1 , \ldots , P ,\tag{5}
$$

with the constraint $T ( p ) \geq T _ { \mathrm { m i n } } > 0$ . Our training objective is formulated as teacher model trajectory distillation. The key insight is to enhance the uniqueness and straighten the probability flow by imposing powerful, structured constraints. We achieve this with an autoregressive formulation:

$$
p ( \mathbf { x } _ { 0 } ^ { p } \mid \mathcal { T } _ { < p } ) = p _ { \mathrm { p r i o r } } ( \mathbf { x } _ { T ( p ) } ^ { p } ) \times \prod _ { t = 1 } ^ { T ( p ) } p ( \mathbf { x } _ { t - 1 } ^ { p } \mid \mathbf { x } _ { T ( p ) : t } ^ { p } , \mathcal { T } _ { < p } ) ,\tag{6}
$$

where $\mathbf { x } _ { T ( p ) : t } ^ { p } = [ \mathbf { x } _ { T ( p ) } ^ { p } , \mathbf { x } _ { T ( p ) - 1 } ^ { p } , \ldots , \mathbf { x } _ { t } ^ { p } ]$ denotes the historical denoising trajectory. This formulation offers two fundamental advantages that embody our principle of intensive scaling. First, it equips each denoising step with a higher-dimensional coordinate system for conditioning. Specifically, the combination of spatial context from previously generated patches, $\scriptstyle { \mathcal { T } } _ { < p } ,$ , and temporal context from the historical trajectory, $\mathbf { x } _ { T ( p ) : t } ^ { p }$ , imposes a highly specific constraint. This drastically reduces the variance of the reverse process, transforming the mapping from $\mathbf { x } _ { t } ^ { p } \operatorname { t o } \mathbf { x } _ { t - 1 } ^ { p }$ from an ambiguous, one-to-many problem into a nearly deterministic, one-to-one function, thereby straightening the probability flow path. Second, it enables synergistic information fusion. To predict $\mathbf { x } _ { t - 1 } ^ { p }$ at every step, the model learns to integrate both coarse-grained and fine-grained information. The recent denoised sample $\mathbf { x } _ { t } ^ { p }$ is the best source for fine-grained details, while the historical trajectory closer to $\mathbf { x } _ { T ( p ) } ^ { p }$ provides better coarse-grained structural information. We aim to estimate $p ( \mathbf { x } _ { t - 1 } ^ { p } \mid \mathbf { x } _ { T ( p ) : t } ^ { p } , \mathcal { T } _ { < p } )$ , which, under our strong conditioning, approximates a Dirac delta distribution. This is achieved within the Flow Matching framework by defining the mapping function:

$$
\begin{array} { r l } & { \mathbf { x } _ { t - 1 } ^ { p } = G ( \mathbf { x } _ { T ( p ) : t } ^ { p } , \mathcal { T } _ { < p } , t ) } \\ & { \qquad : = \mathbf { x } _ { t } ^ { p } + ( \gamma ( t - 1 ) - \gamma ( t ) ) \cdot v _ { \theta } ( \mathbf { x } _ { T ( p ) : t } ^ { p } , t , \mathcal { T } _ { < p } ) , } \end{array}\tag{7}
$$

which is approximated by our student neural network v using an Euler step. Here, γ is the noise schedule. The complete training objective integrates multidimensional conditioning with this progressive schedule. It distills the teacher’s trajectory by regressing the target sample:

$$
\mathcal { L } _ { \mathrm { N e x t S h o r t c u t } } = \mathbb { E } _ { p \sim [ 1 , P ] } \left[ \sum _ { t = 1 } ^ { T ( p ) } \left. G _ { \theta } ( \mathbf { x } _ { T ( p ) : t } ^ { p } , t , \mathcal { T } _ { < p } ) - \mathbf { x } _ { t - 1 } ^ { p } \right. _ { 2 } ^ { 2 } \right] .\tag{8}
$$

Here we have that $G _ { \theta } ( \mathbf { x } _ { T ( p ) : t } ^ { p } , t , \mathcal { T } _ { < p } ) = \mathbf { x } _ { t } ^ { p } + ( \gamma ( t - 1 ) -$ $\gamma ( t ) ) \cdot v _ { \theta } ( \mathbf { x } _ { T ( p ) : t } ^ { p } , t , \mathcal { T } _ { < p } )$ represents the student’s one-step prediction. The transformer architecture allows computing $G _ { \theta }$ for all t simultaneously by using an attention mask. We design the attention mask to be block-wise causal, allowing the model to use the entire trajectory history $\mathbf { x } _ { T ( p ) : t } ^ { p }$ as context, which is the most flexible and effective option. This objective directly embodies our intensive scaling principle: it trains the student network to predict the optimal denoising path using both temporal (historical trajectory) and spatial (previous patches) conditioning. The green arrows in Figure 2 illustrate this accelerated generation path. Our framework can also benefit from an additional discriminator loss applied to the final generated patch $\hat { \mathbf { x } } _ { 0 } ^ { p }$ . This adversarial training, which uses real data as supervision, further enhances the high-frequency details in the generated outputs. During inference, the model generates the first patch with the full step budget $T ( 1 ) = T _ { \mathrm { f u l l } }$ to establish a robust and reliable anchor. Each subsequent patch $p \geq 2$ is generated in an autoregressive manner. Starting from $\mathbf { x } _ { T ( p ) } ^ { p } \sim p _ { \mathrm { p r i o r } } ,$ , at each step t, the model predicts $\hat { \mathbf { x } } _ { t - 1 } ^ { p } = G _ { \theta } ( \hat { \mathbf { x } } _ { T ( p ) : t } ^ { p } , t , \mathcal { T } _ { < p } )$ based on the entire history of predictions $\hat { \mathbf { x } } _ { T ( p ) : t } ^ { p } = [ \mathbf { x } _ { T ( p ) } ^ { p } , \hat { \mathbf { x } } _ { T ( p ) - 1 } ^ { p } , \ldots , \hat { \mathbf { x } } _ { t } ^ { p } ]$ ]. The information of the historical predictions is efficiently managed using a key-value cache. This process leverages the accumulated context $\mathcal { T } _ { < p }$ and the learned ability to exploit constraints, achieving significant speedup while maintaining generation quality through constraint exploitation.

XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling
<table><tr><td>Model</td><td>#params</td><td>AR steps</td><td>Diff steps</td><td>FID↓</td><td>IS↑</td><td>Pre.↑</td><td>Rec.↑</td><td>Time (s)↓</td><td>Speed-Up↑</td></tr><tr><td colspan="10">Base Models (170M-208M parameters)</td></tr><tr><td>MAR-B (Li et al., 2024)</td><td>208M</td><td>256</td><td>100</td><td>2.31</td><td>281.7</td><td>0.82</td><td>0.57</td><td>0.650</td><td>1.0×</td></tr><tr><td></td><td></td><td>64</td><td>50</td><td>2.39↑0.08</td><td>281.0↓0.7</td><td>0.82</td><td>0.57</td><td>0.134↓0.516</td><td>4.9×↑3.9</td></tr><tr><td>FlowAR-S (Ren et al., 2024)</td><td>170M</td><td>5</td><td>25</td><td>3.70↑1.39</td><td>235.1↓46.6</td><td>0.81↓0.01</td><td>0.51↓0.06</td><td>0.024↓0.626</td><td>27.1×↑26.1</td></tr><tr><td>xAR-B (Ren et al., 2025)</td><td>172M</td><td>4</td><td>50</td><td>1.67↓0.64</td><td>265.2↓16.5</td><td>0.80↓0.02</td><td>0.62↑0.05</td><td>0.130↑0.520</td><td>5.0×↑4.0</td></tr><tr><td>XYZFlow-B (w/o GAN)</td><td>172M</td><td>4</td><td>5→2</td><td>2.02↓0.29</td><td>261.1↓20.6</td><td>0.80↓0.02</td><td>0.58↑0.01</td><td>0.018↓0.632</td><td>36.1×↑35.1</td></tr><tr><td>XYZFlow-B (w/ GAN)</td><td>172M</td><td>4</td><td>5→2</td><td>1.63↓0.68</td><td>268.5↓13.2</td><td>0.81↓0.01</td><td>0.62↑0.05</td><td>0.018↓0.632</td><td>36.1×↑35.1</td></tr><tr><td colspan="10">Large Models (479M-820M parameters)</td></tr><tr><td>DiT/XL-2 (Peebles &amp; Xie, 2023)</td><td>676M</td><td></td><td>25</td><td>2.89</td><td>230.2</td><td>0.80</td><td>0.57</td><td>0.494</td><td>1.0×</td></tr><tr><td>DART</td><td>812M</td><td></td><td>16</td><td>5.62↑2.73</td><td>231.7↑1.5</td><td>0.78↓0.02</td><td>0.55↓0.02</td><td>0.075↓0.419</td><td>6.6×↑5.6</td></tr><tr><td>DART-AR (Gu et al., 2025)</td><td>812M</td><td>4096</td><td>-</td><td>3.98↑1.09</td><td>256.8↑26.6</td><td>0.80</td><td>0.58↑0.01</td><td>7.44↑6.946</td><td>0.07×↓0.93</td></tr><tr><td>DART-FM</td><td>820M</td><td></td><td>16</td><td>3.82↑0.93</td><td>263.8↑33.6</td><td>0.81↑0.01</td><td>0.60↑0.03</td><td>0.32↑0.174</td><td>1.5×↑0.5</td></tr><tr><td>MeanFlow-XL/2 (w/o CFG)</td><td>676M</td><td></td><td>1</td><td>3.43↑0.54</td><td></td><td></td><td></td><td>0.009↓0.485</td><td>54.9×↑53.9</td></tr><tr><td>MeanFlow-XL/2</td><td>676M</td><td></td><td>1</td><td>2.93↑0.04</td><td></td><td></td><td></td><td>0.018↓0.476</td><td>27.4×↑26.4</td></tr><tr><td>MeanFlow-XL/2+</td><td>676M</td><td></td><td>1</td><td>2.20↓0.69</td><td></td><td></td><td></td><td>0.018↓0.476</td><td>27.4×↑26.4</td></tr><tr><td>DART Distill</td><td>676M</td><td></td><td>2</td><td>10.92↑8.03</td><td>167.0↓63.12</td><td>0.68↓0.12</td><td>0.52↓0.05</td><td>0.033↓0.461</td><td>15.0×↑14.0</td></tr><tr><td>ARD (w/o GAN)</td><td>676M</td><td></td><td>2</td><td>6.29↑3.40</td><td>188.0↓42.15</td><td>0.74↓0.06</td><td>0.56↓0.01</td><td>0.034↓0.460</td><td>14.5×↑13.5</td></tr><tr><td>DART Distill (w/o GAN)</td><td>676M</td><td></td><td>4</td><td>10.25↑7.36</td><td>181.6↓48.62</td><td>0.70↓0.10</td><td>0.47↓0.10</td><td>0.065↓0.429</td><td>7.6×↑6.6</td></tr><tr><td>ARD (w/o GAN)</td><td>676M</td><td></td><td>4</td><td>4.32↑1.43</td><td>209.0↓21.2</td><td>0.77↓0.03</td><td>0.57</td><td>0.066↓0.428</td><td>7.5×↑6.5</td></tr><tr><td>DART Distill (w/ GAN)</td><td>676M</td><td></td><td>4</td><td>3.84↑0.95</td><td>221.1↓9.1</td><td>0.78↓0.02</td><td>0.55↓0.02</td><td>0.065↓0.429</td><td>7.6×↑6.6</td></tr><tr><td>ARD (w/ GAN)</td><td>676M</td><td></td><td>4 100</td><td>1.84↓1.05 1.78↓1.11</td><td>235.8↓5.6</td><td>0.80</td><td>0.62↑0.05</td><td>0.066↓0.428</td><td>7.5×↑6.5</td></tr><tr><td>MAR-L (Li et al., 2024)</td><td>479M</td><td>256</td><td>50</td><td>1.86↓1.03</td><td>296.0↑65.8</td><td>0.81↑0.01</td><td>0.60↑0.03</td><td>1.102↑0.608</td><td>0.4×↓0.6</td></tr><tr><td>FlowAR-L (Ren et al., 2024)</td><td></td><td>64</td><td>25</td><td>1.87↓1.02</td><td>294.0↑63.8</td><td>0.80</td><td>0.61↑0.04</td><td>0.250↓0.244</td><td>2.0×↑1.0</td></tr><tr><td></td><td>589M</td><td>5</td><td></td><td></td><td>273.1↑42.9</td><td>0.80</td><td>0.62↑0.05</td><td>0.124↓0.370</td><td>4.0×↑3.0</td></tr><tr><td>xAR-L (Ren et al., 2025)</td><td>608M</td><td>4</td><td>50</td><td>1.28↓1.61</td><td>292.5↑62.3</td><td>0.82↑0.02</td><td>0.62↑0.05</td><td>0.394↑0.100</td><td>1.3×↑0.3</td></tr><tr><td>XYZFlow-L (w/o GAN)</td><td>608M</td><td>4</td><td>5→2</td><td>1.79↓1.10</td><td>265.2↓35.0</td><td>0.81↑0.01</td><td>0.61↑0.04</td><td>0.050↓0.444</td><td>9.9×↑8.9</td></tr><tr><td>XYZFlow-L (w/ GAN)</td><td>608M</td><td>4</td><td>5→2</td><td>1.25↓1.64</td><td>295.8↑65.6</td><td>0.83↑0.03</td><td>0.63↑0.06</td><td>0.050↓0.444</td><td>9.9×↑8.9</td></tr><tr><td colspan="10">Huge Models (943M-2.0B parameters)</td></tr><tr><td>FlowAR-H (Ren et al., 2024)</td><td>1.9B</td><td>5</td><td>50</td><td>1.67</td><td>276.3</td><td>0.80</td><td>0.62</td><td>0.423</td><td>1.0×</td></tr><tr><td>VAR-d30 (Tian et al., 2024)</td><td>2.0B</td><td>10</td><td></td><td>1.92↑0.25</td><td>323.1↑46.8</td><td>0.82↑0.02</td><td>0.59↓0.03</td><td>0.039↓0.384</td><td>10.8×↑9.8</td></tr><tr><td>MAR-H (Li et al., 2024)</td><td>943M</td><td>256</td><td>100</td><td>1.55↓0.12</td><td>303.7↑27.4</td><td>0.81↑0.01</td><td>0.62</td><td>1.957↑1.534</td><td>0.2×↓0.8</td></tr><tr><td></td><td></td><td>64</td><td>50</td><td>1.65↓0.02</td><td>299.8↑23.5</td><td>0.80</td><td>0.62</td><td>0.462↑0.039</td><td>0.9×↓0.1</td></tr><tr><td>xAR-H (Ren et al., 2025)</td><td>1.1B</td><td>4</td><td>50</td><td>1.24↓0.43</td><td>301.6↑25.3</td><td>0.83↑0.03</td><td>0.64↑0.02</td><td>0.896↑0.473</td><td>0.5×↓0.5</td></tr><tr><td>XYZFlow-H (w/o GAN)</td><td>1.1B</td><td>4</td><td>5→2</td><td>1.73↓0.06</td><td>271.5↓4.8</td><td>0.82↑0.02</td><td>0.62</td><td>0.105↓0.318</td><td>4.0×↑3.0</td></tr><tr><td>XYZFlow-H (w/ GAN)</td><td>1.1B</td><td>4</td><td>5→2</td><td>1.22↓0.45</td><td>304.2↑27.9</td><td>0.84↑0.04</td><td>0.64↑0.02</td><td>0.105↓0.318</td><td>4.0×↑3.0</td></tr></table>

Table 1. Main results of different methods on ImageNet 256×256. Models are organized by parameter count from small to large. Colored numbers indicate performance change relative to baseline models.

## 4. Experiments and Results

We empirically validate the efficacy of XYZFlow, focusing on the theoretical claims in Section 3. Our experiments demonstrate that: (1) Multidimensional conditioning straightens the probability flow for subsequent patches, enabling better generation quality; (2) The Next Shortcut Prediction objective effectively trains models to utilize accumulated context for accelerated generation by decreasing total denoising steps; (3) XYZFlow achieves promising efficiency-quality trade-offs in image synthesis.

## 4.1. Experimental Setup

We train and evaluate XYZFlow on the ImageNet 256×256 class-conditional generation benchmark (Deng et al., 2009). Training is conducted on 8 NVIDIA H100 GPUs for 300K steps with a batch size of 128 and a learning rate of 0.0001.

To better evaluate scalability, we employ three autoregressive teacher models of varying sizes, including Base (172M), Large (608M), and Huge (1.1B) (Ren et al., 2025). ODE trajectory data is generated by running each teacher for 50 steps with a classifier-free guidance scale of 2.3, pre-computing 2.5M trajectories for distillation. We comprehensively evaluate sample quality using four established metrics: Frechet´ Inception Distance (FID) (Heusel et al., 2017), Inception Score (IS) (Salimans et al., 2016), and Precision/Recall (Dhariwal & Nichol, 2021) to quantify fidelity and diversity. Inference time (shown in seconds) and speed-up relative to baseline models are reported to measure efficiency.

## 4.2. Main Results and Analysis

Table 1 presents a comprehensive system-level comparison on ImageNet 256×256. XYZFlow achieves superior efficiency-quality trade-offs across all model scales, demonstrating the advantage of our spatio-temporal autoregressive framework. Our key innovation lies in a unified framework for spatio-temporal autoregression: (1) Temporally, the method models the denoising trajectory, making it straighter and more predictable; (2) Spatially, it decomposes the image into a sequence of patches and predicts each subsequent patch conditioned on previous patches and the learned temporal trajectory. The linearization of the temporal trajectory provides stable and simplified global context, which significantly eases the generation task for each individua patch. This enables the use of a lightweight network for fast inference. The results validate this design. Our base model, XYZFlow-B, with only 172M parameters, matches the inference speed of the 676M-parameter one-step mode MeanFlow-XL/2+ (Geng et al., 2025) (both at 0.018s per image) while achieving superior FID (1.63 vs. 2.20). Furthermore, XYZFlow-B significantly outperforms the 820Mparameter DART-FM (Gu et al., 2025) in both FID (1.63 vs. 3.82) and inference speed (0.018s vs. 0.32s). The consistent improvements across scales—XYZFlow-L (608M) and XYZFlow-H (1.1B) achieve FIDs of 1.25 and 1.22, respectively—demonstrate the scalability and effectiveness of our approach. Compared to teacher models, XYZFlow provides substantial speed-up while maintaining quality. It also outperforms other accelerated iterative methods. For instance, XYZFlow-B provides a 36.1× speed-up over its teacher with an FID of 1.63, whereas FlowAR-S provides a 27.1× speed-up with a higher FID of 3.70. This demonstrates that intensive modeling of the generative probability flow (”depth scaling”) offers a viable and efficient alternative to simply enlarging model scale (”width scaling”) or adding distillation steps. In summary, XYZFlow success fully straightens the generative trajectory through tempora conditioning, simplifies patch-wise generation via spatial decomposition, and leverages their synergy for efficient, high-fidelity image synthesis. While ARD and DART Distill variants demonstrate competitive performance at larger scales, they primarily serve to highlight the efficiency of our architectural design. For instance, ARD (w/ GAN, 676M) achieves a strong FID of 1.84 at 0.066s, and DART Distil (w/ GAN, 676M) attains an FID of 3.84 at 0.065s. However, their inference speeds are 3.6× slower than our 172Mparameter XYZFlow-B (0.018s) which also achieves a better FID of 1.63. This indicates that while these methods leverage distillation and GAN augmentation for quality improvements, their underlying autoregressive or iterative structures do not fundamentally accelerate inference. In contrast, our spatio-temporal autoregressive framework fundamentally rethinks the generative process, achieving superior speed and quality with a significantly more compact model.

To further evaluate robustness under a weaker teacher, we additionally replace the xAR teacher with DiT/XL-2 (25- step) and perform patch-wise trajectory distillation under the same 256×256 setting. The results in Table 2 show that standard 5-step distillation degrades sharply with this teacher, yielding FID 8.97 without GAN and 3.37 with GAN. In contrast, XYZFlow-B maintains substantially stronger performance: the progressive 5→4→3→2 schedule achieves FID 3.85 without GAN, and further improves to 1.74 with GAN. Notably, the progressive schedule matches the constant 5→5→5→5 variant (3.85 vs. 3.83) while requiring fewer total denoising steps, confirming that Next Shortcut Prediction preserves fidelity while improving efficiency. These results show that the gains of XYZFlow arise from the proposed spatio-temporal architecture itself rather than depending on a particularly strong teacher.

<table><tr><td>Method</td><td>Params</td><td>Steps</td><td>FID↓</td><td>Total</td></tr><tr><td>DiT/XL-2 (Teacher)</td><td>676M</td><td>25</td><td>2.89</td><td>25</td></tr><tr><td>DiT/XL-2 Distilled</td><td>676M</td><td>5</td><td>8.97</td><td>5</td></tr><tr><td>DiT/XL-2 Distilled (GAN)</td><td>676M</td><td>5</td><td> $3 . 3 7 \downarrow 5 . 6 0 $ </td><td>5</td></tr><tr><td>XYZFlow-B (Constant)</td><td>172M</td><td> $5 \to 5 \to 5 \to 5$ </td><td> $3 . 8 3 \downarrow 5 . 1 4$ </td><td>20</td></tr><tr><td>XYZFlow-B</td><td>172M</td><td> $5 {  } 4 {  } 3 {  } 2$ </td><td> $3 . 8 5 \downarrow 5 . 1 2$ </td><td>14</td></tr><tr><td>XYZFlow-B (GAN)</td><td>172M</td><td> $5 {  } 4 {  } 3 {  } 2$ </td><td> $\mathbf { 1 . 7 4 . } 1 . 6 3$ </td><td>14</td></tr></table>

Table 2. Weak-teacher distillation results on ImageNet 256×256 using DiT/XL-2 (25-step) as the teacher.

<table><tr><td>Teacher</td><td>Teacher FID↓</td><td>XYZFlow-L↓</td><td>XYZFlow-L (GAN)↓</td></tr><tr><td>xAR-B</td><td>1.61</td><td>1.91↑0.30</td><td>1.32↓0.59</td></tr><tr><td>XAR-L</td><td>1.28</td><td>1.79↑0.51</td><td>1.25↓0.54</td></tr><tr><td>xAR-H</td><td>1.24</td><td>1.77↑0.53</td><td>1.25↓0.52</td></tr></table>

Table 3. Comparison across xAR teachers on ImageNet 256×256.

This trend is consistent across teachers of different strengths. As summarized in Table 3, XYZFlow-L without GAN remains stable when distilling from xAR-B, xAR-L, and xAR-H, varying only from 1.91 to 1.77 FID. After GAN finetuning, the gap narrows further, reaching 1.32, 1.25, and 1.25, respectively. The largest gain appears for the weakest teacher, where GAN improves XYZFlow-L by 0.59 FID, while the stronger teachers show smaller but still consistent gains of 0.54 and 0.52. This result indicates that stronger teachers are helpful, but the proposed architecture preserves robust performance even when the teacher quality degrades.

## 4.3. Ablation Study

Component Importance Analysis. Table 4 presents a systematic evaluation of XYZFlow’s core components. Our analysis reveals the distinct contributions of each component through controlled ablations: (1) Full History Guidance emerges as the most critical component. Removing full history guidance $( { \mathcal { T } } _ { < p } )$ causes FID to degrade by approximately 0.5 across all model sizes (e.g., 2.02→3.51 for Base). These results show that inter-patch trajectory conditioning is essential for maintaining generation quality, and also validate our hypothesis that complete trajectory information can provide richer contextual signals than final patch content alone. (2) Shortcut Prediction brings an interesting advantage in denoising efficiency. While having minimal impact on final quality (e.g., FID differences < 0.03), it provides substantial acceleration benefits. We can observe that the “- Shortcut” variant maintains similar FID scores but requires 20 total steps compared to XYZFlow’s 14 steps. This result confirms that Shortcut Prediction primarily improves efficiency rather than quality, consistent with its design goal of exploiting straightened trajectories for faster convergence. (3) Local History Conditioning $( \mathcal { H } _ { t } ^ { p } )$ contributes moderately to performance, with its removal causing FID degradation of approximately 0.2-0.3. This result suggests that while intra-patch temporal conditioning provides useful stabilization, the inter-patch spatial conditioning plays a more significant role in the overall framework. (4) Adversarial loss consistently improves all metrics across different model sizes, which demonstrates that XYZFlow’s straightened paths provide a favorable foundation for adversarial training on real data, enabling the student model to potentially exceed the teacher’s capabilities.

XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling
<table><tr><td>Method</td><td>Params</td><td>Steps</td><td>FID↓</td><td>IS↑</td><td>Pre↑</td><td>Rec↑</td><td>Total</td></tr><tr><td>Teacher-Base</td><td>172M</td><td>50</td><td>1.72</td><td>280.4</td><td>0.82</td><td>0.59</td><td>200</td></tr><tr><td>Distilled-Base</td><td>172M</td><td>5</td><td>3.03</td><td>225.3</td><td>0.78</td><td>0.55</td><td>20</td></tr><tr><td>+ Local History</td><td>172M</td><td>5→2</td><td>2.25↓0.78</td><td>249.8↑24.5</td><td>0.78</td><td>0.54↓0.01</td><td>14</td></tr><tr><td>- Full History</td><td>172M</td><td> $5 \to 2$ </td><td>3.51↑0.48</td><td>219.9↓5.4</td><td>0.77↓0.01</td><td>0.52↓0.03</td><td>14</td></tr><tr><td>- Shortcut</td><td>172M</td><td>5</td><td>2.05↓0.98</td><td>258.5↑33.2</td><td>0.80↑0.02</td><td>0.58↑0.03</td><td>20</td></tr><tr><td>XYZFlow-B</td><td>172M</td><td> $5 \to 2$ </td><td> $2 . 0 2 \downarrow 1 . 0 1$ </td><td>261.1↑35.8</td><td>0.80↑0.02</td><td>0.58↑0.03</td><td>14</td></tr><tr><td>XYZFlow-B (GAN)</td><td>172M</td><td>5→2</td><td>1.63↓1.40</td><td>268.5↑43.2</td><td>0.81↑0.03</td><td>0.62↑0.07</td><td>14</td></tr><tr><td>Teacher-Large</td><td>608M</td><td>50</td><td>1.28</td><td>292.5</td><td>0.82</td><td>0.62</td><td>200</td></tr><tr><td>Distilled-Large</td><td>608M</td><td>5</td><td>2.85</td><td>235.1</td><td>0.79</td><td>0.57</td><td>20</td></tr><tr><td>+ Local History</td><td>608M</td><td>5→2</td><td>2.02↓0.83</td><td>254.3↑19.2</td><td>0.79</td><td>0.57</td><td>14</td></tr><tr><td>- Full History</td><td>608M</td><td>5→2</td><td> $3 . 3 5 \uparrow 0 . 5 0$ </td><td>229.3↓5.8</td><td>0.78↓0.01</td><td>0.54↓0.03</td><td>14</td></tr><tr><td>- Shortcut</td><td>608M</td><td>5</td><td> $1 . 8 2 \downarrow 1 . 0 3$ </td><td>263.8↑28.7</td><td>0.81↑0.02</td><td>0.61↑0.04</td><td>20</td></tr><tr><td>XYZFlow-L</td><td>608M</td><td>5→2</td><td> $1 . 7 9 \downarrow 1 . 0 6$ </td><td>265.2↑30.1</td><td>0.81↑0.02</td><td>0.61↑0.04</td><td>14</td></tr><tr><td>XYZFlow-L (GAN)</td><td>608M</td><td>5→2</td><td>1.25↓1.60</td><td>295.8↑60.7</td><td>0.83↑0.04</td><td>0.63↑0.06</td><td>14</td></tr><tr><td>Teacher-Huge</td><td>1.1B</td><td>50</td><td>1.24</td><td>301.6</td><td>0.83</td><td>0.64</td><td>200</td></tr><tr><td>Distilled-Huge</td><td>1.1B</td><td>5</td><td>2.75</td><td>240.8</td><td>0.80</td><td>0.59</td><td>20</td></tr><tr><td>+ Local History</td><td>1.1B</td><td>5→2</td><td>1.96↓0.79</td><td>259.1↑18.3</td><td>0.80</td><td>0.57↓0.02</td><td>14</td></tr><tr><td>- Full History</td><td>1.1B</td><td> $5 \to 2$ </td><td>3.25↑0.50</td><td>234.6↓6.2</td><td>0.79↓0.01</td><td>0.56↓0.03</td><td>14</td></tr><tr><td>- Shortcut</td><td>1.1B</td><td>5</td><td> $1 . 7 6 \downarrow 0 . 9 9 $ </td><td>268.2↑27.4</td><td>0.82↑0.02</td><td>0.61↑0.02</td><td>20</td></tr><tr><td>XYZFlow-H</td><td>1.1B</td><td>5→2</td><td> $1 . 7 3 \downarrow 1 . 0 2 $ </td><td>271.5↑30.7</td><td>0.82↑0.02</td><td>0.62↑0.03</td><td>14</td></tr><tr><td>XYZFlow-H (GAN)</td><td>1.1B</td><td> $5 \to 2$ </td><td> $\pmb { 1 . 2 2 \downarrow 1 . 5 3 }$ </td><td>304.2↑63.4</td><td>0.84↑0.04</td><td>0.64↑0.05</td><td>14</td></tr></table>

Table 4. Ablation study of XYZFlow components. FID↓ is lower-better; IS↑, Pre↑, Rec↑ are higher-better. Total Steps↓ represents the cumulative inference steps. Colored numbers indicate performance change relative to baseline (Distilled) models.In the table, ’+’ and ’- denote the baseline model with and without the corresponding component, respectively.

<table><tr><td>Cell Size</td><td>Grid Size</td><td>FID↓</td><td>IS↑</td></tr><tr><td>1 × 1</td><td> $1 6 \times 1 6$ </td><td>2.85</td><td>280.7</td></tr><tr><td>2 × 2</td><td> $8 \times 8$ </td><td>2.46</td><td>283.9</td></tr><tr><td> $4 \times 4$ </td><td> $4 \times 4$ </td><td>2.11</td><td>289.2</td></tr><tr><td> $\mathbf { 8 \times 8 }$ </td><td> $\mathbf { 2 } \times \mathbf { 2 }$ </td><td>2.02</td><td>301.5</td></tr><tr><td> $1 6 \times 1 6$ </td><td> $1 \times 1$ </td><td>2.55</td><td>283.8</td></tr></table>

Table 5. Ablation on the cell size $( k \times$ k tokens).

Ablation Study on Cell Size. A prediction cell is formed by grouping a cluster of k × k spatially adjacent tokens. Using a larger cell size incorporates more tokens in a single prediction step, thereby capturing broader contextual information. For a $2 5 6 \times 2 5 6$ input image, the encoded continuous latent representation has a spatial resolution of $1 6 \times 1 6 .$ . Consequently, the image can be partitioned into an $m \times m$ grid, where each cell consists of k × k neighboring tokens. Under the XYZFlow framework, we employ a unified denoising schedule of 5 steps per patch and use the Base Model configuration. As shown in Table 5, we evaluate different cell sizes with $k \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ . Here, k = 1 corresponds to a single token, while k = 16 represents the entire image as a single entity. Performance improves as k increases, reaching its peak with an FID of 2.02 at a cell size of $8 \times 8 ( i . e .$ $k = 8 )$ . Beyond this point, performance declines, yielding an FID of 2.55 when the whole image is treated as one entity. These results indicate that using cells as the prediction unit, rather than the entire image, allows the model to explicitly condition on previously generated context. This conditioning enhances prediction confidence while preserving both rich semantic information and fine-grained local details.

Analysis of Next Shortcut Prediction. Our ablation study of shortcut prediction strategies, summarized in Table 6, yields four key insights that support our design choices. (1) Progressive denoising step reduction achieves optimal efficiency-quality trade-off. Our proposed $5 {  } 4 {  } 3 {  } 2$ strategy achieves nearly identical quality to the constant 5-step approach (FID 1.63 vs. 1.63) but with 30% fewer total steps (14 vs. 20), demonstrating that sampling can be accelerated more aggressively in later stages without compromising quality. (2) Initial step configuration is critical. The comparable performance of constant strategies (5→5→5→5 vs. 4→4→4→4) highlights that an initial step of T(1)=5 (a divisor of the teacher’s 50-step trajectory) provides the optimal starting point for effective distillation. (3) Aggressive reduction can harm generation diversity. The 5→2→2→2 strategy shows degraded recall (0.58 vs. 0.62), confirming that overly aggressive step reduction compromises sample diversity, while our gradual approach better preserves solution space coverage. (4) Computational cost must be balanced. Although the 8→8→8→8 strategy achieves the lowest FID (1.62), it still requires 32 steps in total, which is over twice of our method’s cost. This validates our focus on optimal efficiency-quality trade-offs rather than purely maximizing generation quality.

![](images/fc37a86c91c7198c468d808ab4c221ed8e1f35630ceccd7bad9298bc602fff06.jpg)  
Figure 4. Visual comparison demonstrating the efficiency of XYZFlow. Our 1.1B-parameter model achieves an 8.5× faster generation time than the teacher model and an additional 1.5× speedup over the base student distillation, with no perceptible loss in quality.

<table><tr><td>Schedule T(p)</td><td>FID↓</td><td>IS↑</td><td>Pre↑</td><td>Rec↑</td><td>Total Steps↓</td></tr><tr><td>Teacher (50 steps)</td><td>1.72</td><td>280.4</td><td>0.82</td><td>0.59</td><td>200</td></tr><tr><td>5→4→3→2 (Ours)</td><td>1.63↓0.09</td><td>268.5↓11.9</td><td>0.81↓0.01</td><td>0.62↑0.03</td><td>14↓186</td></tr><tr><td>8→4→2→1 (Uniform)</td><td>1.75↑0.03</td><td>255.2↓25.2</td><td>0.77↓0.05</td><td>0.57↓0.02</td><td>15↓185</td></tr><tr><td>4→4→4→4</td><td>1.84↑0.12</td><td>248.9↓31.5</td><td>0.74↓0.08</td><td>0.55↓0.04</td><td>16↓184</td></tr><tr><td>4→3→2→1</td><td>1.88↑0.16</td><td>245.3↓35.1</td><td>0.73↓0.09</td><td>0.54↓0.05</td><td>10↓190</td></tr><tr><td>8→8→8→8</td><td>1.61↓0.11</td><td>269.0↓11.4</td><td>0.81↓0.01</td><td>0.62↑0.03</td><td>32↓168</td></tr><tr><td>5→5→5→5 (Constant)</td><td>1.63↓0.09</td><td>267.9↓12.5</td><td>0.81↓0.01</td><td>0.61↑0.02</td><td>20↓180</td></tr><tr><td>5→4→4→2</td><td>1.64↓0.08</td><td>266.2↓14.2</td><td>0.80↓0.02</td><td>0.60↑0.01</td><td>15↓185</td></tr><tr><td>5→2→2→2 (Aggressive)</td><td>1.70↓0.02</td><td>258.6↓21.8</td><td>0.78↓0.04</td><td>0.58↓0.01</td><td>11↓189</td></tr></table>

Table 6. Ablation study of Next Shortcut Prediction strategies on Base Model (172M). FID↓ is lower-better; IS↑, Pre↑, Rec↑ are higher-better. Total Steps↓ represents the cumulative inference steps. Colored numbers indicate performance change relative to baseline (50 steps). Bold indicates best performance, underline indicates second best.

## 5. Concluding Remarks

In this work, we propose XYZFlow, a spatio-temporal generative framework for improving few-step generation through historical state conditioning and Next Shortcut Prediction. By coupling temporal trajectory conditioning with patch-wise autoregressive generation, XYZFLow imposes richer structure on the denoising process, making its evolution more predictable and easier to approximate with a small number of steps. This design yields a strong efficiencyquality trade-off that remains robust even under weaker teachers and transfers consistently across teachers of varying strengths. More broadly, our results suggest that scaling the dimensionality of constraints, rather than relying solely on larger models or more distillation steps, offers a promising direction for efficient, high-fidelity generative modeling.

## Impact Statement

This work introduces XYZFlow, a framework for improving generative modeling efficiency by scaling the expressivity of probability flows through constraint dimensionality rather than model size. The main impact is methodological, as it offers a principled direction for high-fidelity generation with lower computational cost, advancing the understanding and practical design of efficient generative models.

Broader societal impacts are not the primary focus of this work. As with generative modeling methods generally, downstream risks depend on application context, including potential misuse in synthetic media or privacy-sensitive data generation. Responsible evaluation and deployment should therefore be considered for specific applications.

## References

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In ICLR, 2023. 2

Boffi, N. M., Albergo, M. S., and Vanden-Eijnden, E. Flow map matching. arXiv preprint arXiv:2406.07507, 2024. 1, 2

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 6

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 6

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In CVPR, 2021. 3

Frans, K., Hafner, D., Levine, S., and Abbeel, P. One step diffusion via shortcut models. In ICLR, 2025. 1, 2, 3

Geng, Z., Pokle, A., and Kolter, J. Z. One-step diffusion distillation via deep equilibrium models. In NeurIPS, 2024a. 1, 2

Geng, Z., Pokle, A., Luo, W., Lin, J., and Kolter, J. Z. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024b. 2

Geng, Z., Deng, M., Bai, X., Kolter, J. Z., and He, K. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 7

Gu, J., Wang, Y., Zhang, Y., Zhang, Q., Zhang, D., Jaitly, N., Susskind, J., and Zhai, S. Dart: Denoising autoregressive transformer for scalable text-to-image generation. In ICLR, 2025. 3, 6, 7

Hang, T., Bao, J., Wei, F., and Chen, D. Fast autoregressive models for continuous latent generation. arXiv preprint arXiv:2504.18391, 2025. 2, 3

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 6

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 1

Kim, Y., Anagnostidis, S., Du, Y., Schonfeld, E., Kohler,¨ J., Georgopoulos, M., Pumarola, A., Thabet, A., and Sanakoyeu, A. Autoregressive distillation of diffusion transformers. In CVPR, 2025. 3

Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. Autoregressive image generation using residual quantization. In CVPR, 2022. 3

Li, T., Tian, Y., Li, H., Deng, M., and He, K. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 2, 3, 6

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In ICLR, 2023. 1, 2

Liu, J., Liu, X., Mei, K., Wen, Y., Yang, M.-H., and Liu, W. Streaming autoregressive video generation via diagonal distillation. In ICLR, 2026. 3

Liu, W., Liu, Z., Paull, L., Weller, A., and Scholkopf, B.¨ Structural causal 3d reconstruction. In ECCV, 2022a. 3

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022b. 2

Lu, C. and Song, Y. Simplifying, stabilizing and scaling continuous-time consistency models. In ICLR, 2025. 2

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022a. 1

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022b. 1

Luo, W., Hu, T., Zhang, S., Sun, J., Li, Z., and Zhang, Z. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In NeurIPS, 2024. 2

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In ICCV, 2023. 6

Razavi, A., Van Den Oord, A., and Vinyals, O. Generating diverse high-fidelity images with vq-vae-2. NeurIPS, 2019. 3

Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., and Chen, L.- C. Flowar: Scale-wise autoregressive image generation meets flow matching. arXiv preprint arXiv:2412.15205, 2024. 2, 3, 6

Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., and Chen, L.-C. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025. 3, 6

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 1, 2

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training gans. In NeurIPS, 2016. 6

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. In ECCV, 2024. 1, 2

Shaul, N., Singer, U., Gat, I., and Lipman, Y. Transition matching: Scalable and flexible generative modeling. arXiv preprint arXiv:2506.23589, 2025. 12, 16

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 1, 2

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In ICLR, 2021a. 2

Song, Y. and Dhariwal, P. Improved techniques for training consistency models. In ICLR, 2024. 2

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 1, 2

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In ICLR, 2021b. 1, 2

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In ICML, 2023. 1, 2

Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 6

Wang, Z., Zhang, R., Ding, K., Yang, Q., Li, F., and Xiang, S. Continuous speculative decoding for autoregressive image generation. arXiv preprint arXiv:2411.11925, 2024. 2, 3

Wen, Y., Liu, W., Raj, B., and Singh, R. Self-supervised 3d face reconstruction via conditional estimation. In ICCV, 2021. 3

Wu, S., Zhang, W., Xu, L., Jin, S., Wu, Z., Tao, Q., Liu, W., Li, W., and Loy, C. C. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025. 3

Yan, F., Wei, Q., Tang, J., Li, J., Wang, Y., Hu, X., Li, H., and Zhang, L. Lazymar: Accelerating masked autoregressive models via feature caching. arXiv preprint arXiv:2503.12450, 2025. 2, 3

Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang, W., Meng, C., Ermon, S., and Cui, B. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024. 2

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In CVPR, 2024. 2

Zhao, Q., Singh, J., Xu, M., Asthana, A., Gould, S., and Zheng, L. Disa: Diffusion step annealing in autoregressive image generation. arXiv preprint arXiv:2505.20297, 2025. 3

Zhou, L., Ermon, S., and Song, J. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025. 3

## Appendix

A Experiment Results 13   
B Generated Samples 13   
C Theoretical Proofs of Multi-Dimensional Conditional Enhancement 13   
C.1 Information-Theoretic Foundation of Conditional Modeling 13   
C.2 Theoretical Proof of Denoising-Dimension Conditional Enhancement 14   
C.3 Theoretical Proof of Spatial-Dimension Conditional Enhancement 15   
C.4 Theoretical Proof of Next-Shortcut Prediction 15   
C.5 Unified Perspective: Path Straightening through Conditional Enhancement 16   
C.6 Corollaries and Quantitative Implications 16   
C.7 Comparison with Transition Matching Framework (Shaul et al., 2025) 16

## A. Experiment Results

Student Model Training Configuration We adhere to the teacher configuration for student training, with the exception of gradient clipping and batch size. The training configuration for the 5-step student model using regression loss is as follows: a learning rate of 10<sup>−4</sup>, weight decay of 0.0, gradient clipping of 1.0, batch size of 64, 300k training iterations, and an EMA decay rate of 0.9999. The student model is initialized with the teacher’s weights. Training is performed on 8 NVIDIA H100 GPUs and requires approximately 2 days to complete 300k iterations. According to our convergence analysis, the FID metric exhibits stable convergence within the first 100k iterations (equivalent to roughly 16 hours of training). The same configuration is applied to the baseline (step distillation) as well.

Discriminator Loss Configuration When incorporating an additional discriminator loss, we use the teacher network as a feature extractor and train only the discriminator heads attached to the features extracted from each transformer block. The discriminator heads predict logits on a per-token basis. We employ hinge loss and adopt the discriminator head architecture proposed in the same work. The discriminator is trained using the student model’s final prediction and real data. It is trained with a learning rate of $1 \times 1 0 ^ { - 3 }$ and no weight decay. Adaptive balancing is applied between the regression loss and the discriminator loss. A batch size of 48 is used for both the student model and the discriminator.

Adversarial Fine-tuning Procedure By adding the discriminator loss and further fine-tuning a student model that was pre-trained with regression loss, we observe significant performance gains. The adversarial training component consistently improves all metrics across different model sizes. The fine-tuning process is conducted for 40k iterations, during which both the student generator and the discriminator are jointly optimized with adaptive loss balancing.

Performance Improvement Results As shown in our ablation studies (Table 2), the adversarial fine-tuning yields substantial improvements: the Base model’s FID improves from 2.02 to 1.63, the Large model from 1.79 to 1.25, and the Huge model from 1.73 to 1.22. These results demonstrate the effectiveness of incorporating adversarial training into the distillation framework, with consistent enhancements observed across all model scales.

## B. Generated Samples

Figure 5 presents samples generated by xAR (trained on ImageNet 256×256). These results collectively validate XYZFlow’s multidimensional conditioning approach in maintaining straighter trajectories and delivering consistent speed-up advantages while preserving sample quality across all model scales and highlight XYZFlow’s ability to generate images with exceptional visual quality.

## C. Theoretical Proofs of Multi-Dimensional Conditional Enhancement

## C.1. Information-Theoretic Foundation of Conditional Modeling

Definition C.1 (Conditional Entropy Reduction). Let target distribution be $p ( \mathbf { x } )$ where $\mathbf { x } \in \mathbb { R } ^ { d }$ is a high-dimensional random variable, and conditioning variable be c. The conditional distribution $p ( \mathbf { x } | \mathbf { c } )$ has lower entropy than the unconditional distribution $p ( \mathbf { x } )$ under the following conditions:

1. x and c are not independent: $I ( { \bf x } ; { \bf c } ) > 0$

2. The conditional distribution is well-defined and has finite second moments

3. The dimensionality d is sufficiently large for concentration effects

Theorem C.2 (Quantified Conditional Entropy Inequality). For any distribution $p ( \mathbf { x } )$ with finite covariance matrix $\Sigma _ { \mathbf { x } } ,$ , the conditional entropy satisfies:

$$
H ( \mathbf { x } | \mathbf { c } ) \leq H ( \mathbf { x } ) - \frac { 1 } { 2 } \log \left( 1 + \frac { I ( \mathbf { x } ; \mathbf { c } ) } { \lambda _ { \operatorname* { m i n } } ( \Sigma _ { \mathbf { x } } ) } \right)\tag{9}
$$

where $\lambda _ { \operatorname* { m i n } } ( \Sigma _ { \mathbf { x } } )$ is the minimum eigenvalue of the covariance matrix, and the bound holds for distributions beyond exponential families.

![](images/a6777566a10d291e24833f4e680d37707bba56319749c802181f370a1ca74040.jpg)  
Figure 5. Randomly selected examples of generated images from XYZFlow. XYZFlow shows high-quality generative modeling abilities.

Proof. By the entropy power inequality and the De Bruijn identity:

$$
H ( \mathbf { x } | \mathbf { c } ) = H ( \mathbf { x } ) - I ( \mathbf { x } ; \mathbf { c } )\tag{10}
$$

$$
\leq H ( \mathbf { x } ) - { \frac { 1 } { 2 } } \log \left( 1 + { \frac { I ( \mathbf { x } ; \mathbf { c } ) } { \operatorname { V a r } [ \mathbf { x } ] } } \right) \quad { \mathrm { ( E P I ~ f o r ~ g e n e r a l ~ d i s t r i b u t i o n s ) } }\tag{11}
$$

For high-dimensional cases, we use the covariance matrix characterization. The mutual information lower bound comes from the Cramer-Rao bound applied to the estimation of´ x given c. □

## C.2. Theoretical Proof of Denoising-Dimension Conditional Enhancement

C.2.1. AUTOREGRESSIVE TRAJECTORY AS CONDITION

Assumption C.3 (Diffusion Process Regularity). The diffusion process satisfies:

1. Smoothness: The score function $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } )$ is L-Lipschitz continuous

2. Optimality: The model is optimally trained: $v _ { \theta } ( \mathbf { x } _ { t } , t ) = \mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \vert \mathbf { x } _ { t } ]$

3. Discretization Error: Time discretization error is bounded by $O ( \Delta t ^ { 2 } )$

Theorem C.4 (Trajectory Straightening with Quantitative Bounds). Under the above assumptions, when using complete historical trajectory conditioning, the path straightness metric improvement satisfies:

$$
\Delta S = S ^ { u n c o n d i t i o n a l } - S ^ { c o n d i t i o n a l } \geq \frac { \mathbb { E } [ V a r [ v _ { \theta } | \mathcal { H } _ { t } ] ] } { L ^ { 2 } T ^ { 2 } }\tag{12}
$$

where T is the total time horizon and L is the Lipschitz constant.

Proof. Starting from the straightness metric decomposition:

$$
\begin{array} { r l } & { S = \mathbb { E } _ { t } \left[ \Vert ( { \mathbf { x } } _ { 1 } - { \mathbf { x } } _ { 0 } ) - v _ { \theta } ( { \mathbf { x } } _ { t } , t ) \Vert ^ { 2 } \right] } \\ & { \quad = { \mathrm { V a r } } [ v _ { \theta } ] + \Vert \mathbb { E } [ v _ { \theta } ] - ( { \mathbf { x } } _ { 1 } - { \mathbf { x } } _ { 0 } ) \Vert ^ { 2 } + 2 \mathbb { E } [ \langle v _ { \theta } - \mathbb { E } [ v _ { \theta } ] , \mathbb { E } [ v _ { \theta } ] - ( { \mathbf { x } } _ { 1 } - { \mathbf { x } } _ { 0 } ) \rangle ] } \end{array}\tag{13}
$$

(14)

The cross term vanishes due to orthogonality. For the conditional case, by the law of total variance:

$$
\mathrm { V a r } [ v _ { \theta } ^ { \mathrm { c o n d i t i o n a l } } ] = \mathbb { E } [ \mathrm { V a r } [ v _ { \theta } ^ { \mathrm { c o n d i t i o n a l } } | \mathcal { H } _ { t } ] ]\tag{15}
$$

$$
\begin{array} { r l } { \leq \mathbb { E } [ \mathrm { V a r } [ v _ { \theta } ^ { \mathrm { u n c o n d i t i o n a l } } | \mathcal { H } _ { t } ] ] } & { { } \mathrm { ( b y ~ c o n d i t i o n i n g ) } } \end{array}\tag{16}
$$

The bias term change is bounded by the Lipschitz continuity:

$$
\| \mathbb { E } [ v _ { \theta } ^ { \mathrm { c o n d i t i o n a l } } ] - \mathbb { E } [ v _ { \theta } ^ { \mathrm { u n c o n d i t i o n a l } } ] \| \leq L \cdot \mathbb { E } [ \| \mathcal { H } _ { t } \| ]\tag{17}
$$

Combining these bounds gives the quantitative improvement.

## C.3. Theoretical Proof of Spatial-Dimension Conditional Enhancement

Theorem C.5 (High-Dimensional Spatial Variance Reduction). For high-dimensional patch-based generation with ddimensional patches, the spatial conditional variance reduction satisfies:

$$
\frac { \sigma _ { c o n d } ^ { 2 } } { \sigma _ { u n c o n d } ^ { 2 } } \leq 1 - \frac { \rho ^ { 2 } } { d } + O \left( \frac { 1 } { d ^ { 3 / 2 } } \right)\tag{18}
$$

where $\rho$ is the average correlation between patches, and the bound holdsfor non-Gaussian distributions via Berry-Esseen type arguments.

Proof. Using the covariance decomposition for high-dimensional systems:

$$
\Sigma _ { \mathrm { t o t a l } } = \Sigma _ { \mathrm { w i t h i n } } + \Sigma _ { \mathrm { b e t w e e n } }\tag{19}
$$

$$
\mathrm { V a r } [ \mathbf { x } ^ { i } ] = \mathbb { E } [ \mathrm { V a r } [ \mathbf { x } ^ { i } | \mathbf { x } ^ { 1 : i - 1 } ] ] + \mathrm { V a r } [ \mathbb { E } [ \mathbf { x } ^ { i } | \mathbf { x } ^ { 1 : i - 1 } ] ]\tag{20}
$$

For high dimensions, the variance ratio converges to:

$$
{ \frac { \sigma _ { \mathrm { c o n d } } ^ { 2 } } { \sigma _ { \mathrm { u n c o n d } } ^ { 2 } } } \to 1 - { \frac { 1 } { d } } \sum _ { j = 1 } ^ { i - 1 } \rho _ { j } ^ { 2 } \quad { \mathrm { a s ~ } } d \to \infty\tag{21}
$$

where $\rho _ { j }$ is the correlation between patch i and patch $j .$

## C.4. Theoretical Proof of Next-Shortcut Prediction

Theorem C.6 (Trajectory Alignment with Exponential Convergence). Under next-shortcut prediction, the trajectory alignment error decreases exponentially with the number of conditioning patches:

$$
\mathbb { E } [ \| \mathbf { v } _ { t } ^ { i } - \mathbf { v } _ { t } ^ { j } \| ^ { 2 } ] \leq C \cdot e ^ { - \lambda ( i - j ) } + \epsilon _ { a p p r o x }\tag{22}
$$

where $\lambda > 0$ is the alignment rate constant, C depends on the smoothness, and $\epsilon _ { a p p r o x }$ is the function approximation error.

Proof. Consider the trajectory dynamics as a dynamical system. The alignment process can be viewed as contractive mapping:

$$
\mathbf { v } _ { t } ^ { i } = f ( \mathbf { v } _ { t } ^ { i - 1 } , \mathcal { H } _ { t } ) + w _ { t }\tag{23}
$$

$$
\| f ( \mathbf { v } ) - f ( \mathbf { v } ^ { \prime } ) \| \leq \alpha \| \mathbf { v } - \mathbf { v } ^ { \prime } \| \quad \mathrm { w i t h } \alpha < 1\tag{24}
$$

By contraction mapping principle, the distance between consecutive trajectories decreases geometrically. The exponential rate comes from solving the recurrence relation. □

## C.5. Unified Perspective: Path Straightening through Conditional Enhancement

Theorem C.7 (Multi-Scale Path Straightening). The multidimensional conditioning in XYZFlow achieves path straightening at multiple scales:

1. Micro-scale (temporal): Variance reduction within each patch trajectory

2. Meso-scale (spatial): Alignment between adjacent patches

3. Macro-scale (global): Overall distribution matching

The combined effect is multiplicative rather than additive.

Proof. The proof uses multi-scale analysis. Define straightness metrics at different scales:

$$
S _ { \mathrm { m i c r o } } ^ { ( i ) } = \mathbb { E } _ { t } [ \| ( \mathbf { x } _ { 1 } ^ { i } - \mathbf { x } _ { 0 } ^ { i } ) - \mathbf { v } _ { t } ^ { i } \| ^ { 2 } ]
$$

$$
S _ { \mathrm { m e s o } } ^ { ( i , j ) } = \mathbb { E } _ { t } [ \| \mathbf { v } _ { t } ^ { i } - \mathbf { v } _ { t } ^ { j } \| ^ { 2 } ]\tag{25}
$$

(26)

$$
S _ { \mathrm { m a c r o } } = \mathrm { K L } ( p _ { \mathrm { m o d e l } } \| p _ { \mathrm { d a t a } } )\tag{27}
$$

The scaling relationship follows from the tensor product structure of the condition space and the orthogonality of different conditioning dimensions. □

## C.6. Corollaries and Quantitative Implications

Corollary C.8 (Sampling Complexity Reduction). With multidimensional conditioning, the number of sampling steps required to achieve ϵ-accuracy scales as:

$$
N ( \epsilon ) = O \left( \frac { \log ( 1 / \epsilon ) } { \lambda _ { m i n } \cdot \prod _ { k = 1 } ^ { K } ( 1 - \alpha _ { k } ) } \right)\tag{28}
$$

where $\alpha _ { k } < 1$ are the contraction factors for each conditioning dimension, and $\lambda _ { m i n }$ is the smallest time scale.

Corollary C.9 (Dimension-Free Convergence). For high-dimensional systems, the convergence rate becomes dimension-free under sufficient conditioning:

$$
\operatorname * { l i m } _ { d  \infty } \frac { S ^ { c o n d i t i o n a l } } { S ^ { u n c o n d i t i o n a l } } = \prod _ { k = 1 } ^ { K } ( 1 - \alpha _ { k } ) + o ( 1 )\tag{29}
$$

where the o(1) term vanishes as the conditioning dimensions K increase.

## C.7. Comparison with Transition Matching Framework (Shaul et al., 2025)

Theorem C.10 (Theoretical Distinction from Transition Matching). XYZFlow provides the following theoretical advantages over Transition Matching:

1. Dimensionality: TM uses single temporal dimension; XYZFlow uses spatio-temporal dimensions

2. Conditioning: TM conditions on current state; XYZFlow conditions on full history

3. Convergence: XYZFlow achieves exponential convergence vs polynomial in TM

Proof. The distinction follows from comparing the condition spaces:

$$
\begin{array} { r l } { \mathcal { C } _ { \mathrm { T M } } = \left\{ \mathbf { x } _ { t } \right\} } & { { } { \mathrm { ( s i n g l e ~ t i m e ~ p o i n t ) } } } \end{array}\tag{30}
$$

$$
\begin{array} { r l } { \mathcal { C } _ { \mathrm { X Y Z } } = \{ \mathbf { x } _ { \tau } \} _ { \tau = 0 } ^ { t } \cup \{ \mathbf { x } _ { s } ^ { j } \} _ { j = 1 , s = 0 } ^ { i - 1 , T } } & { { } \mathrm { ( f u l l ~ h i s t o r y + s p a t i a l ) } } \end{array}\tag{31}
$$

The richer condition space provides stronger constraints, leading to improved convergence rates via the contraction mapping analysis. □