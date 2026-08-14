# P²Fusion: Prompt-based Progressive Infrared-Visible Image Fusion via Dual-Prior Distillation

Yi Shi<sup>1†</sup>, Huichao Xie<sup>1†</sup>, Yuqing Wang<sup>1</sup>, Mingyu Wang<sup>1</sup>, Kaihui Yang<sup>1</sup>, Yu Liu<sup>2</sup>, Ruitao Lu<sup>3</sup>, Lizhe Li<sup>1</sup>, Junwei Han<sup>1,4\*</sup>, and Dingwen Zhang<sup>1</sup> 1\*

<sup>1</sup> Northwestern Polytechnical University, Xi’an, China <sup>2</sup> Hefei University of Technology, Hefei, China

3 Rocket Force University of Engineering, Xi’an, China

4 Chongqing University of Posts and Telecommunications, Chongqing, China

Abstract. Infrared-visible image fusion (IVIF) is pivotal for multimodal perception, yet reconciling the inherent information disparity between thermal and textural features remains a fundamental challenge. Existing prior-guided methods often rely on static constraints that induce optimization conflicts or utilize extrinsic semantic priors from largescale foundation models (e.g., CLIP/DINO), which frequently fail to exploit the intrinsic modality characteristics essential for high-fidelity fusion. To address these issues, we propose P²Fusion, a prior-guided distillation-based framework that reformulates IVIF via dual intrinsic prompts. Instead of imposing hard-coded penalties, we distill imageintrinsic priors, thermal saliency and spatial quality—into learnable, dynamic regulators. Specifically, a Teach-to-Fuse mechanism provides dual-granularity progressive guidance, coupled with a Gated Dynamic Expert Recalibration (GDER) module for decoupled feature refinement. This design enables the network to adaptively mediate modal competition through expert specialization. Extensive experiments demonstrate that P²Fusion achieves state-of-the-art performance across five mainstream datasets. Notably, our framework demonstrates consistent performance advantages in fusion quality, achieving state-of-the-art results in 14 out of 20 key evaluation metrics across 5 benchmarks. Furthermore, it efectively contributes to the robustness of downstream perception, such as +3.2% mAP on MSRS, +0.5% mAP on M3FD and +0.9% mAP on DroneVehicle for object detection. Our code will be available at https://github.com/YiShi99/P2Fusion.

Keywords: Infrared-visible image fusion · Prompt-based learning · Knowledge distillation · Multimodal perception

![](images/1c5e216d2995837d41fcc06bb2da5094f3703fcd24730dd45615ac38130420b9.jpg)  
Fig. 1: Motivation and Performance Overview. (a) Prior integration paradigm (Left): Existing methods inject priors as static constraints, often causing optimization conflicts. Our P<sup>2</sup>Fusion provides dynamic, prompt-based guidance via image-intrinsic priors to mitigate gradient interference. (b) Fusion quality (Middle): Radar charts across four datasets (MSRS, M3FD, FMB, RoadScene) demonstrate our superior visual fidelity. (c) Downstream perception (Right): P<sup>2</sup>Fusion consistently enhances object detection and semantic segmentation performance on FMB and M3FD benchmarks.

## 1 Introduction

Multi-modal image fusion aims to integrate complementary information from diverse sensors to produce a comprehensive representation with rich details [22,56]. This paradigm is indispensable in critical applications such as autonomous driving [4, 6, 8], target re-identification [3, 10], and medical imaging [62]. Among these, Infrared–Visible Image Fusion (IVIF) has emerged as a predominant research direction, leveraging the inherent synergy between thermal radiation and textural details [18, 30]. Since IVIF is essentially an unsupervised generative task due to the lack of paired ground-truth images, its performance heavily relies on the heuristic design of network architectures and loss functions [60, 65]. Without explicit labels, existing methods often struggle to balance detail fidelity and structural integrity, leading to the emergence of prior-guided fusion methods [24, 31, 43, 49]. By incorporating modality-specific cues, these methods provide indirect supervision to endow networks with scene-adaptive capabilities.

However, we observe that existing prior-guided paradigms face fundamental limitations across three mainstream technical routes. Firstly, methods based on explicit modality-specific hard-constraints [21,31,43] often treat segmentation masks or saliency maps as static regularization terms. This forces the network to prioritize prior distribution fitting over pixel-level feature integration, compelling the model to sacrifice fine-grained textures to satisfy rigid prior boundaries. Secondly, the route of joint optimization with downstream tasks [12, 18, 24, 49] integrates object detection or segmentation as auxiliary tasks. Nevertheless, these frameworks frequently introduce multi-objective optimization conflicts, where the drive for task-level accuracy compromises background detail preservation, creating a persistent trade-of between visual aesthetics and task performance. Thirdly, the recent trend of extrinsic semantic guidance [50, 56] leverages foundation models (e.g., CLIP or DINO) as soft guidance. While mitigating the "hard-constraint" issue, it introduces extrinsic information that deviates from the modal essence. Since image fusion is fundamentally a pixel-level task, the abstract high-level representations lead to a granularity mismatch between semantics and pixels, preventing precise guidance for low-level textural reconstruction.

Based on these insights, we propose a novel Teach-to-Fuse (T2F) paradigm (Fig. 1) that returns to the essence of image data. Following this paradigm, we develop P<sup>2</sup>Fusion, a network built on the core philosophy that priors should be endogenous and dynamic. Rather than introducing extrinsic or rigid constraints, we focus on maximizing the mining of intrinsic attributes from the infrared and visible images themselves. We move away from seeking a "perfect" static prior and instead build an adaptive prompt learning framework. In this work, we select infrared thermal saliency and visible spatial quality as two complementary intrinsic priors. These two priors are chosen because they inherently capture distinctive modality traits and intuitively align with human visual perception, as supported by established literature [21] [43]. Crucially, our framework serves to instantiate a prior-guided prompt-based philosophy rather than chasing the "best" theoretical teachers. It is prior-agnostic and can seamlessly extend to other task-related auxiliary cues (e.g., depth, semantics), thereby releasing optimization freedom and achieving dynamic scene adaptation.

To further optimize fusion performance and strengthen cross-modal interactions, we design a fusion architecture featuring a Gated Dynamic Expert Recalibration (GDER) module, which functions as a self-correcting feedback mechanism. Unlike traditional Mixture-of-Experts (MoE) used for capacity expansion, our GDER module is tasked with feature decoupling and correction. Within this module, we synergize a prior-guided Modality Specialist with a global-aware Attention Specialist to perform modality-specific feedback refinement. By leveraging a reliability-based gating mechanism to adaptively recalibrate the fusion strategy, we efectively decouple conflicting features and rectify biased signals, thereby ensuring a robust and perceptually consistent fusion representation.

The main contributions of this work are summarized as follows:

We propose the "Teach-to-Fuse" paradigm for multi-modal image fusion. By systematically addressing the pitfalls of hard-constrained priors, multiobjective conflicts, and granularity mismatch, we transform static penalties into process-oriented guidance that maximizes the mining of intrinsic modality attributes via dynamic prompts.

– We develop the Gated Dynamic Expert Recalibration (GDER) module to optimize feature refinement and cross-modal interactions. This MoEbased self-correction mechanism synergizes a prior-guided Modality Specialist with a global-aware Attention Specialist, enabling the network to adaptively decouple conflicting features and rectify biased signals through modality-specific feedback.

– Extensive Benchmarking and Perceptual Analysis. We conduct a comprehensive evaluation of the proposed framework across five public datasets and over ten benchmark tasks. The results consistently demonstrate the superiority of our Teach-to-Fuse paradigm in terms of high-fidelity visual fusion, robust cross-domain generalization, and significant performance gains in downstream perception tasks.

## 2 Related Work

## 2.1 Deep Learning Paradigms in IVIF

Deep learning has become the prevailing paradigm in IVIF, outperforming conventional methodologies such as sparse representation and multiscale decomposition [2,34,35,54]. Early deep fusion frameworks primarily explored diverse architectures to improve feature representation, including Auto-Encoders (AEs) [13, 14, 25], CNNs [19, 55, 63], Generative Adversarial Networks (GANs) [17, 32, 53], and more recently, Transformers [30, 44, 62] and Difusion Models [40, 64].

These advanced architectures have demonstrated notable advantages in synthesizing complementary information through attention mechanisms [45,61], sophisticated feature interactions [15,37,65], and disentangled representation learning [5, 60]. However, most of these purely data-driven approaches rely on manually designed loss functions and heuristic architectural constraints to achieve fusion objectives [28, 29]. More comprehensive reviews and comparisons of IVIF methods can be found in [23].

## 2.2 Prior Knowledge Integration in Image Fusion

Prior-guided fusion has emerged as a pivotal paradigm in IVIF, substantially advancing both perceptual quality and downstream task performance [1,20,24,56]. A dominant trend exploits high-level semantics from large-scale foundation models. Representative works include SAGE [49], which harnesses Segment Anything Model (SAM) [11] priors via semantic-persistent attention and dual-level distillation, and CMFS [7, 50], which employs CLIP [36] guidance for cross-modal interaction and frequency-spatial collaboration to suppress noise.

Complementary to foundation models, another line of research integrates priors from specialized vision tasks. STDFusionNet [31], for example, leverages saliency detection masks for spatial guidance and selective feature extraction. Others utilize semantic guidance from pre-trained segmentation models [22, 39, 42, 50] or employ object detection meta-features to mitigate feature misalignment [18,61]. Beyond task-level semantics, alternative prior formulations are embedded within the network architecture. These include incorporating lowrank representation priors to guide feature decomposition [15], exploiting visible image quality priors [43], and exploring modality-specific priors within hybrid CNN-Transformer frameworks [58].

In summary, existing prior-guided IVIF methods often rely on rigid or extrinsic priors, which may cause optimization conflicts or granularity mismatch, leaving room for more adaptive, modality-intrinsic prior modeling.

![](images/73e6c3788f5f32e16e06b87c0ec40de3a9876291c2ed9304052cf60b40119cff.jpg)  
Fig. 2: Illustration of the proposed framework. (a) Overall pipeline: prior-guided visible and infrared features are fused via cross-attention, where the proposed GDER module adaptively refines modality-specific and interaction features for reconstruction. (b) Spatial Quality Distillation: a quality assessor provides global quality guidance to supervise spatial-detail learning. (c) Thermal Saliency Distillation: a saliency teacher ofers thermal saliency priors to enhance target-aware fusion.

## 3 Methodology

In this section, we elucidate the proposed $\mathrm { P ^ { 2 } } ^ { \cdot }$ Fusion framework, which instantiates the "Teach-to-Fuse" paradigm. The architecture, as illustrated in Fig. 2, consists of two core stages: (1) Dual-Teacher Prompt Distillation, which transforms static priors into learnable, process-oriented guidance signals; and (2) Adaptive Fusion with Gated Dynamic Expert Recalibration (GDER), which achieves robust feature integration and self-correcting refinement via decoupled specialized experts.

## 3.1 Dual-Teacher Prompt Distillation

In conventional prior-guided methods, knowledge (e.g., saliency maps) is typically injected as rigid spatial constraints, which limits the optimization flexibility of the network. We argue that prior knowledge should participate in the fusion process as soft, dynamic prompts. We select two complementary intrinsic priors as a validation instance:

– Infrared Saliency Prior $\left( T _ { i r } \right)$ : Derived from a SegFormer-B2 [51] pre-trained on the representative MSRS [41] and FMB [22] datasets, this prior serves as the Infrared-Saliency Teacher (IST) to localize high-thermal-radiation targets (e.g., pedestrians and vehicles) in the infrared modality.

– Visible Quality Prior $( T _ { v i s } )$ : Generated by the no-reference Visible-Quality Teacher (VQT) based on the BRISQUE [33] evaluator, this prior assesses textural reliability spatially, such as underexposed or overexposed areas.

For the visible branch, the VQT generates a global quality score $T _ { v i s } \in \mathbb { R }$ We project the high-dimensional prompt $P _ { v i s }$ into a scalar $P _ { v i s } ^ { \prime }$ to match the teacher’s output. For the infrared branch, the IST provides a saliency map $T _ { i r } \in$ $\mathbb { R } ^ { H \times W }$ . The final enriched features $F = \{ F _ { v i s } , F _ { i r } \}$ are obtained via element-wise summation: $F _ { m } = f _ { m } + \mathrm { E m b e d } ( P _ { m } )$ , where m $\in \{ v i s , i r \}$

![](images/e1d4a541b8b6f57eb4b693a444c062d8964da9d63f464e92770dd133d7a9a09e.jpg)  
Fig. 3: Illustration of the diference between the original dataset and that used for training our infrared thermal saliency teacher: pedestrians and vehicles are defined as foreground, while all others are treated as background.

Specifically, for the infrared branch,semantic predictions are post-processed into binary saliency masks to incorporate domain priors, where pedestrians and vehicles are treated as foreground and all remaining categories as background as shown in Fig. 3. The resulting saliency map is then downsampled via bilinear interpolation to match the spatial resolution of the infrared prompt $P _ { i r }$ . For the visible branch, given an image-level score $s \in [ 0 , 1 0 0 ]$ , we normalize it into a continuous quality prior by $T _ { v i s } = ( 1 0 0 - s ) / 1 0 0$ , where higher values correspond to stronger textural reliability.

## 3.2 Adaptive Fusion with GDER Refinement

The feature spaces of infrared and visible modalities are inherently heterogeneous—one being sensitive to thermal contours and the other rich in textural details. To mitigate modality confusion, we maintain dual-branch processing and employ cross-attention (CA) [27, 46] mechanism for feature modulation:

$$
F _ { i r } ^ { \prime } = \mathrm { C A } ( F _ { i r } , F _ { v i s } , F _ { v i s } ) , \quad F _ { v i s } ^ { \prime } = \mathrm { C A } ( F _ { v i s } , F _ { i r } , F _ { i r } )\tag{1}
$$

where $\operatorname { C A } ( Q , K , V ) =$ softmax $\left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V$ . This stage yields cross-modal interactions $F ^ { \prime } = \{ F _ { v i s } ^ { \prime } , F _ { i r } ^ { \prime } \}$ . To further refine the fused representations and capture modal-specific nuances, we propose the Gated Dynamic Expert Recalibration (GDER) module. Unlike Mixture-of-Experts (MoE) architectures designed primarily for model capacity expansion, our GDER module functions as a decoupled feature recalibrator with explicit functional specialization, aiming to adaptively aggregate the most informative cues from heterogeneous feature spaces:

1. Prior-Responsive Modality Expert $( E _ { m o d } )$ : This expert explicitly consumes the dynamic prompts to reinforce modality-specific salient information $( \mathrm { e . g . }$ , thermal targets or high-contrast textures). It ensures that the network maintains a high-fidelity focus on regions indicated by the learned priors, facilitating efective knowledge injection.

2. Prior-Agnostic Attention Expert $( E _ { a t t } )$ : Operating independently of the prompts, this expert utilizes a global attention mechanism (CBAM) to mine intrinsic long-range dependencies and fine-grained structural details. It acts as a contextual compensator to ensure the integrity of textures that may fall outside the prior’s primary focus.

The GDER module acts as an adaptive integration arbiter. A gating network $G ( \cdot )$ computes dynamic weights $w = \mathrm { s o f t m a x } ( G ( F ^ { \prime } , P ) )$ based on the mutual agreement between local feature characteristics and global prompt signals. The refined feature is formulated as:

$$
F ^ { \prime \prime } = F ^ { \prime } + w _ { 1 } \cdot E _ { m o d } ( F ^ { \prime } ) + w _ { 2 } \cdot E _ { a t t } ( F ^ { \prime } )\tag{2}
$$

By synergizing the outputs of these decoupled specialists, the GDER module adaptively recalibrates bimodal information flow. The Prior-Responsive Expert anchors features on saliency-guided regions, while the Prior-Agnostic Expert captures the underlying structural dependencies that might be overlooked by prompts. This dual-expert design allows the network to transcend the limitations of any single representation, achieving robust feature recalibration that balances salient foreground targets with rich background textures.

In this configuration, the gating network acts as a dynamic weight allocator that optimizes the integration of modality-specific and context-aware features. When both specialists provide congruent information, they work in synergy to enhance feature representational power; when their focuses diverge, the gating mechanism adaptively balances their contributions to ensure that the fused feature remains both semantically meaningful and structurally complete. This mechanism achieves adaptive feature refinement through multi-perspective complementarity, rather than rigid constraint. After N iterative refinements, the refined dual-modality features are concatenated and fed into the reconstruction decoder to generate the final fused image $I _ { f u s e d }$

## 3.3 Optimization Objectives

We replace traditional hard-coded prior losses with learnable prompt guidance.   
The total loss $\mathcal { L } _ { t o t a l }$ comprises perceptual constraints and distillation terms.

Perceptual Constraints: The Gradient Loss ensures edge preservation: $\mathcal { L } _ { g r a d } = \| \nabla I _ { f u s e d } - \operatorname* { m a x } ( | \nabla I _ { i r } | , | \nabla I _ { v i s } | ) \| _ { 1 }$ . The Intensity Loss balances thermal

and texture information: $\mathcal { L } _ { i n t } = \Vert I _ { f u s e d } - \operatorname* { m a x } ( I _ { i r } , I _ { v i s } ) \Vert _ { 1 }$ . We further introduce the SSIM Loss to maintain structural consistency:

$$
\mathcal { L } _ { s s i m } = 1 - ( \omega _ { i r } \mathrm { S S I M } ( I _ { i r } , I _ { f u s e d } ) + \omega _ { v i s } \mathrm { S S I M } ( I _ { v i s } , I _ { f u s e d } ) )\tag{3}
$$

where weights $\omega$ are adaptively computed based on the mean gradient magnitude of each modality.

Prompt Distillation Loss: The visible prompts are optimized via MSE: $\mathcal { L } _ { v i s } ^ { d i s t i l l } = \mathcal { \bar { \| } } P _ { v i s } ^ { \prime } - T _ { v i s } \| _ { 2 } ^ { 2 }$ . The infrared prompts are optimized via Binary Cross-Entropy (BCE) to ensure spatial saliency alignment:

$$
\mathcal { L } _ { i r } ^ { d i s t i l l } = - ( T _ { i r } \odot \log P _ { i r } ^ { m a p } + ( 1 - T _ { i r } ) \odot \log ( 1 - P _ { i r } ^ { m a p } ) )\tag{4}
$$

The overall optimization objective is defined as:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { 1 } \mathcal { L } _ { i n t } + \lambda _ { 2 } \mathcal { L } _ { s s i m } + \lambda _ { 3 } \mathcal { L } _ { g r a d } + \lambda _ { 4 } \mathcal { L } _ { i r } ^ { d i s t i l l } + \lambda _ { 5 } \mathcal { L } _ { v i s } ^ { d i s t i l l }\tag{5}
$$

## 4 Experiments

## 4.1 Experiments Settings

Datasets and metrics. We evaluate $\mathrm { P ^ { 2 } }$ Fusion on five public benchmarks: MSRS [41], M3FD [18], FMB [22], RoadScene [52] and DroneVehicle [38]. Generalization is assessed on their oficial test splits, covering fusion quality (all), object detection (MSRS, M3FD, DroneVehicle), and semantic segmentation (MSRS, FMB). Fusion quality is measured using four standard metrics, while downstream evaluations follow the respective standard protocols.

Implementations. Our framework is implemented in $\mathrm { P y }$ Torch on four NVIDIA RTX 3090 GPUs. Adam optimizes parameters for 100 epochs (batch size 12, initial lr $1 \times 1 0 ^ { - 4 } )$ , decayed by 0.5 at iterations [3000, 6000, 9000, 12000, 15000]. The optimal weights for the total loss (λ1–λ5) are (8,25,20,0.5,5).

Comparisons. We compare with 12 SOTA fusion networks (SAGE [49], Dit-Fuse [16], MRFS [59], Freefusion [60], FreqGAN [48], LutFuse [57], LRRNet [15], DDFM [64], TarDal [18], ReCoNet [9], TIM [26]) to validate our method’s efectiveness and generalization ability.

## 4.2 Infrared and visible image fusion

We conduct IVIF experiments on MSRS (361 pairs), M3FD (300), FMB (280), RoadScene (221) and DroneVehicle (8980). Training uses combined MSRS (1083 pairs) and FMB (1220) training sets, whose segmentation labels train the infrared saliency teacher as mentioned in 3.1.

Qualitative Comparisons. Qualitative results for selected pairs are in Fig. 4 (first row) and Fig. 5. On the MSRS dataset, SAGE, DiTFuse, FreqGAN, LRRNet, FreeFusion, and TIM exhibit an over-reliance on visible textures at the expense of infrared saliency, resulting in attenuated structural details in low-light regions. Conversely, MRFS yields over-smoothed textures with suboptimal local contrast, while LUT-Fuse introduces noticeable artifacts around moving targets despite better bimodal preservation. In contrast, our method efectively recovers background details (e.g., tree branches and shadowed building textures) while maintaining sharp pedestrian contours and high fidelity.

![](images/e2c3fd8e36e4afb9e8d673f424ea04d56035344b6c8a9a5d14bb1740eb3c50d8.jpg)  
Fig. 4: Comprehensive qualitative comparison on the MSRS dataset. Row 1 shows fused images; Row 2 and Row 3 show the corresponding semantic segmentation and object detection results, respectively. Our method consistently achieves superior performance across image fusion, semantic segmentation, and object detection tasks.

In the high-contrast M3FD scene (Fig. 5, first row), competing methods suffer from modality collapse, making the background text indistinct. Our method maintains a stable equilibrium, yielding clearer structures and superior text legibility.

In the smoke-occluded FMB scene (Fig. 5, second row), most competitors (SAGE, MRFS, FreqGAN, LUT-Fuse, LRRNet, TIM) under-exploit infrared cues, leaving details corrupted by smoke. Meanwhile, DiTFuse and FreeFusion over-rely on the infrared modality, inducing an unnatural night-like shift and losing visible textures like the left tree. Our method ensures an optimal balance, simultaneously preserving infrared contours and fine visible details.

In the over-exposed RoadScene scene (Fig. 5, third row), SAGE, MRFS, FreeFusion, FreqGAN, and LUT-Fuse under-utilize infrared cues, resulting in over-exposed outputs that miss critical contours (e.g., the pedestrian near the tree trunk). In contrast, DiTFuse, LRRNet, and TIM fail to maintain modality balance, producing unnaturally dark skies and losing details on billboards. Our method achieves a more stable dynamic balance, recovering the global appearance while preserving rich textures and fine details.

![](images/c86c840d8c6dd5805295ed98fb7993f31c27465e98f5cfca9bdfbdd4c99344c3.jpg)  
Fig. 5: Qualitative fusion comparison. Representative high-contrast scenes from M3FD, smoke-occluded scenes from FMB, and over-exposed scenes from RoadScene are selected. Our method consistently preserves infrared salient targets and visible textures across degradations, achieving the best modality balance.

Table 1: Quantitative comparison on the MSRS, M3FD, FMB, and RoadScene datasets using four key evaluation metrics. (1st: Red, 2nd: Light Blue)
<table><tr><td>Datasets</td><td colspan="3">MSRS</td><td colspan="3"></td><td colspan="2">M3FD</td><td colspan="3"></td><td colspan="5">RoadScene</td></tr><tr><td>Methods</td><td>|FQIE↑ VIF↑ Qabf↑</td><td></td><td></td><td>MI↑</td><td>|FQIE↑ VIF↑ Qabf↑</td><td></td><td></td><td></td><td>MI↑ |FQIE↑</td><td>VIF↑</td><td>Qabf↑</td><td>MI↑</td><td>|FQIE↑ VIF↑</td><td></td><td>Qabf↑</td><td>MI↑</td></tr><tr><td>DiTFuse(TPAMI&#x27;25)</td><td>0.642</td><td>0.281</td><td>0.363</td><td>2.146</td><td>0.389</td><td>0.188</td><td>0.274</td><td>2.514</td><td>0.536</td><td>0.215</td><td>0.363</td><td>2.620</td><td>0.445</td><td>0.320</td><td>0.426</td><td>2.938</td></tr><tr><td>FreqGAN(TCSVT&#x27;25)</td><td>0.679</td><td>0.285</td><td>0.492</td><td>2.871</td><td>0.445</td><td>0.282</td><td>0.399</td><td>3.272</td><td>0.518</td><td>0.293</td><td>0.437</td><td>3.434</td><td>0.442</td><td>0.279</td><td>0.444</td><td>2.967</td></tr><tr><td>Freefusion(TPAMI&#x27;25)</td><td>0.590</td><td>0.281</td><td>0.430</td><td>1.984</td><td>0.621</td><td>0.473</td><td>0.539</td><td>2.712</td><td>0.696</td><td>0.501</td><td>0.587</td><td>3.004</td><td>0.403</td><td>0.362</td><td>0.403</td><td>2.692</td></tr><tr><td>LutFuse(ICCV&#x27;25)</td><td>0.783</td><td>0.424</td><td>0.610</td><td>3.625</td><td>0.466</td><td>0.319</td><td>0.480</td><td>3.987</td><td>0.575</td><td>0.319</td><td>0.527</td><td>3.897</td><td>0.328</td><td>0.265</td><td>0.381</td><td>3.644</td></tr><tr><td>SAGE(CVPR&#x27;25)</td><td>0.763</td><td>0.335</td><td>0.535</td><td>3.217</td><td>0.659</td><td>0.363</td><td>0.575</td><td>3.115</td><td>0.763</td><td>0.384</td><td>0.634</td><td>3.429</td><td>0.379</td><td>0.237</td><td>0.334</td><td>2.919</td></tr><tr><td>TIM(TPAMI&#x27;24)</td><td>0.655</td><td>0.277</td><td>0.469</td><td>3.025</td><td>0.533</td><td>0.325</td><td>0.524</td><td>3.533</td><td>0.679</td><td>0.125</td><td>0.581</td><td>3.534</td><td>0.297</td><td>0.181</td><td>0.354</td><td>3.611</td></tr><tr><td>MRFS(CVPR&#x27;24)</td><td>0.625</td><td>0.340</td><td>0.484</td><td>3.068</td><td>0.287</td><td>0.194</td><td>0.249</td><td>2.965</td><td>0.477</td><td>0.161</td><td>0.388</td><td>3.087</td><td>0.268</td><td>0.159</td><td>0.271</td><td>2.788</td></tr><tr><td>LRRNet(TPAMI&#x27;23)</td><td>0.599</td><td>0.243</td><td>0.420</td><td>2.938</td><td>0.530</td><td>0.283</td><td>0.483</td><td>2.823</td><td>0.649</td><td>0.112</td><td>0.539</td><td>3.024</td><td>0.277</td><td>0.133</td><td>0.326</td><td>2.791</td></tr><tr><td>DDFM(ICCV&#x27;23)</td><td>0.569</td><td>0.363</td><td>0.469</td><td>2.666</td><td>0.522</td><td>0.311</td><td>0.456</td><td>2.851</td><td>0.566</td><td>0.214</td><td>0.504</td><td>3.144</td><td>0.485</td><td>0.372</td><td>0.471</td><td>2.940</td></tr><tr><td>TarDal(CVPR&#x27;22)</td><td>0.495</td><td>0.357</td><td>0.428</td><td>2.646</td><td>0.425</td><td>0.317</td><td>0.402</td><td>3.182</td><td>0.443</td><td>0.304</td><td>0.406</td><td>3.419</td><td>0.395</td><td>0.328</td><td>0.392</td><td>3.404</td></tr><tr><td>ReCoNet(ECCV&#x27;22)</td><td>0.720</td><td>0.325</td><td>0.500</td><td>3.115</td><td>0.570</td><td>0.291</td><td>0.481</td><td>3.095</td><td>0.692</td><td>0.168</td><td>0.546</td><td>3.288</td><td>0.422</td><td>0.245</td><td>0.334</td><td>3.139</td></tr><tr><td>Ours</td><td colspan="10">0.850 0.445 0.682 3.905| 0.743</td><td colspan="7"></td></tr></table>

Quantitative Comparisons. Tables 1 report the quantitative results. Across all datasets, our method achieves SOTA on 12 out of 16 metrics and ranks second on 3 others, highlighting its overall superiority and strong cross-dataset generalization.

![](images/f2617962d31a536ce55aa228fd2c5ef46b8e6802c4bf3e20af908192c305a9fe.jpg)  
Fig. 6: Qualitative evaluation on downstream tasks. We visualize object detection results on M3FD (Row 1) and semantic segmentation results on FMB (Row 2). Our framework demonstrates superior performance in downstream tasks.

Table 2: Quantitative results of downstream tasks: Object Detection on M3FD and Semantic Segmentation on FMB. (1st: Red, 2nd: Light Blue)
<table><tr><td>Datasets</td><td colspan="6">M3FD(Object Detection)</td><td colspan="8"></td></tr><tr><td>Methods</td><td>[Person</td><td>Car</td><td>Bus</td><td>Lamp</td><td>Moto.</td><td>Truck</td><td>mAP↑| T.Sign</td><td>Person</td><td>Car</td><td>Truck</td><td></td><td>Moto. Pole</td><td></td><td>mPA↑ mIoU↑</td></tr><tr><td>DiTFuse(TPAMI&#x27;25)</td><td>0.514</td><td>0.684</td><td>0.742</td><td>0.497</td><td>0.525</td><td>0.663 60.46</td><td>73.146</td><td>63.568</td><td>81.763</td><td>33.809</td><td>38.908</td><td>43.214</td><td>63.340</td><td>56.808</td></tr><tr><td>FreqGAN(TCSVT&#x27;25)</td><td>0.544</td><td>0.700</td><td>0.766 0.510</td><td></td><td>0.519</td><td>0.673 61.85</td><td>73.558</td><td>66.301</td><td>82.225</td><td>31.691</td><td>38.425</td><td>44.730</td><td>64.471</td><td>57.284</td></tr><tr><td>Freefusion(TPAMI&#x27;25)</td><td>0.527</td><td>0.702</td><td>0.770 0.526</td><td>0.512</td><td></td><td>0.674 61.87</td><td>71.751</td><td>64.320</td><td>80.160</td><td>36.174</td><td>33.359</td><td>41.968</td><td>63.469</td><td>56.421</td></tr><tr><td>LutFuse(ICCV&#x27;25)</td><td>0.530</td><td>0.691 0.763</td><td>0.500</td><td>0.501</td><td>0.670</td><td>60.91</td><td>72.434</td><td>65.192</td><td>80.893</td><td>16.908</td><td>38.130</td><td>42.354</td><td>62.221</td><td>55.113</td></tr><tr><td>SAGE(CVPR&#x27;25)</td><td>0.531</td><td>0.707</td><td>0.735 0.483</td><td>0.539</td><td>0.658</td><td>60.86</td><td>72.390</td><td>65.286</td><td>81.387</td><td>31.706</td><td>35.223</td><td>41.864</td><td>63.798</td><td>56.479</td></tr><tr><td>TIM(TPAMI&#x27;24)</td><td>0.496</td><td>0.686</td><td>0.743 0.503</td><td></td><td>0.506</td><td>0.665 59.97</td><td>72.216</td><td>62.701</td><td>81.081</td><td>26.209</td><td>30.750</td><td>41.752</td><td>61.942</td><td>54.776</td></tr><tr><td>MRFS(CVPR&#x27;24)</td><td>0.508</td><td>0.687</td><td>0.760 0.482</td><td></td><td>0.483</td><td>0.657 59.68</td><td>67.617</td><td>63.632</td><td>79.955</td><td>28.056</td><td>33.916</td><td>40.510</td><td>60.926</td><td>53.959</td></tr><tr><td>LRRNet(TPAMI&#x27;23)</td><td>0.519</td><td>0.693 0.756</td><td>0.514</td><td>0.499</td><td>0.659</td><td>60.67</td><td>70.693</td><td>64.877</td><td>81.654</td><td>34.215</td><td>31.684</td><td>41.634</td><td>63.853</td><td>56.382</td></tr><tr><td>DDFM(ICCV&#x27;23)</td><td>0.526</td><td>0.706 0.749</td><td>0.483</td><td>0.558</td><td>0.662</td><td>61.44</td><td>71.478</td><td>65.847</td><td>81.797</td><td>32.143</td><td>33.463</td><td>42.865</td><td>64.466</td><td>57.001</td></tr><tr><td>TarDal(CVPR&#x27;22)</td><td>0.544</td><td>0.693 0.755</td><td>0.512</td><td>0.495</td><td>0.672</td><td>61.22</td><td>72.249</td><td>66.804</td><td>80.383</td><td>15.760</td><td>34.881</td><td>41.201</td><td>62.462</td><td>55.193</td></tr><tr><td>ReCoNet(ECCV&#x27;22)</td><td>0.525</td><td>0.704 0.755</td><td>0.529</td><td>0.500</td><td></td><td>0.679 61.53</td><td>71.570</td><td>64.709</td><td>81.360</td><td>28.587</td><td>32.394</td><td>41.458</td><td>63.631</td><td>56.113</td></tr><tr><td>Ours</td><td>0.534 0.704</td><td>0.773</td><td></td><td>0.521</td><td>0.521 0.690</td><td>62.37</td><td>74.153</td><td>65.670</td><td>82.270 38.270</td><td></td><td>37.024</td><td>43.913</td><td>64.672 57.456</td><td></td></tr></table>

## 4.3 IVIF for Downstream Tasks

For object detection, we train YOLOv7s [47] on MSRS-detection (80 pairs) and M3FD (4200 pairs) with an 8:2 split. Tables 2 (left) and 3 (left) show that our method achieves the highest mAP<sub>50–95</sub>, outperforming the previous SOTA by +3.2% on MSRS and +0.5% on M3FD. Visualizations in Figs. 4 (third row) and 6 confirm that our fused images successfully detect all scene targets, whereas other methods sufer from false positives or missed detections.

For semantic segmentation, SegFormer-B5 [51] is adopted on MSRS (1083/361 pairs) and FMB (1220/280). As reported in Tables 2 (right) and 3 (right), our framework delivers the best performance with consistent gains. Qualitative results (Figs. 4, 6, second rows) demonstrate more coherent masks with better structural preservation and reduced fragmentation.

Table 3: Quantitative comparison on the MSRS datasets for object detection and segmentation. (1st: Red, 2nd: Light Blue)
<table><tr><td>Tasks</td><td colspan="5">Object Detection (mAP50 | | mAP50→95)|</td><td colspan="6">Semantic Segmentation</td></tr><tr><td>Methods</td><td>Person</td><td>Car</td><td></td><td>All</td><td>Car</td><td>Person</td><td>Curve</td><td>Stop</td><td>Cone</td><td></td><td>mPA↑ mIoU↑</td></tr><tr><td>DiTFuse(TPAMI&#x27;25)</td><td>0.993 0.717</td><td>0.838 0.715</td><td>0.915</td><td>0.716</td><td>88.073</td><td>68.929</td><td>40.988</td><td>53.430</td><td>66.992</td><td>70.555</td><td>62.390</td></tr><tr><td>FreqGAN(TCSVT&#x27;25)</td><td>0.995 0.731</td><td>0.906 0.645</td><td>0.951</td><td>0.688</td><td>87.681</td><td>69.427</td><td>33.235</td><td>62.014</td><td>66.369</td><td>70.149</td><td>62.100</td></tr><tr><td>Freefusion(TPAMI&#x27;25)</td><td>0.991 0.723</td><td>0.763 0.385</td><td>0.877</td><td>0.554</td><td>84.702</td><td>66.407</td><td>36.520</td><td>64.899</td><td>64.334</td><td>68.776</td><td>61.120</td></tr><tr><td>LutFuse(ICCV&#x27;25)</td><td>0.995 0.727</td><td>0.937 0.628</td><td>0.966</td><td>0.677</td><td>87.848</td><td>70.602</td><td>44.679</td><td>59.176</td><td>67.835</td><td>71.558</td><td>63.471</td></tr><tr><td>SAGE(CVPR&#x27;25)</td><td>0.991 0.700</td><td>0.609 0.476</td><td>0.800</td><td>0.588</td><td>91.265</td><td>66.107</td><td>27.288</td><td>29.242</td><td>56.066</td><td>63.055</td><td>56.228</td></tr><tr><td>TIM(TPAMI&#x27;24)</td><td>0.9860.603</td><td>0.920 0.643</td><td>0.953</td><td>0.623</td><td>89.800</td><td>62.801</td><td>31.877</td><td>32.128</td><td>56.872</td><td>61.471</td><td>55.267</td></tr><tr><td>MRFS(CVPR&#x27;24)</td><td>0.991 0.738</td><td>0.920 0.614</td><td>0.955</td><td>0.676</td><td>91.222</td><td>65.075</td><td>26.845</td><td>29.521</td><td>56.422</td><td>62.333</td><td>55.073</td></tr><tr><td>LRRNet(TPAMI&#x27;23)</td><td>0.994 0.757</td><td>0.962 0.606</td><td>0.978</td><td>0.681</td><td>91.705</td><td>67.741</td><td>30.996</td><td>31.368</td><td>57.681</td><td>62.450</td><td>56.567</td></tr><tr><td>DDFM(ICCV&#x27;23)</td><td>0.986 0.697</td><td>0.747 0.585</td><td>0.866</td><td>0.641</td><td>91.911</td><td>68.686</td><td>31.325</td><td>28.690</td><td>56.075</td><td>63.932</td><td>56.875</td></tr><tr><td>TarDal(CVPR&#x27;22)</td><td>0.974 0.716</td><td>0.858 0.669</td><td>0.916</td><td>0.692</td><td>91.304</td><td>68.907</td><td>28.039</td><td>28.831</td><td>55.925</td><td>62.784</td><td>56.534</td></tr><tr><td>ReCoNet(ECCV&#x27;22)</td><td>0.994 0.712</td><td>0.866 0.573</td><td>0.931</td><td>0.643</td><td>91.750</td><td>66.247</td><td>30.113</td><td>32.249</td><td>56.779</td><td>62.844</td><td>56.928</td></tr><tr><td>VIS</td><td>0.851 0.477</td><td>0.862 0.626</td><td>0.856</td><td>0.552</td><td>91.667</td><td>57.770</td><td>25.817</td><td>29.342</td><td>59.742</td><td>60.604</td><td>54.559</td></tr><tr><td>IR</td><td>0.979 0.613</td><td>0.962 0.672</td><td>0.971</td><td>0.643</td><td>87.155</td><td>68.749</td><td>25.281</td><td>22.444</td><td>45.700</td><td>54.172</td><td>48.778</td></tr><tr><td>Ours</td><td>0.995 0.776|0.995 0.720 0.995</td><td></td><td></td><td>0.748</td><td>88.520</td><td>70.030</td><td>40.903</td><td>59.800</td><td>69.494</td><td>71.480</td><td>63.793</td></tr></table>

## 4.4 Ablation Study

Study on Prior Knowledge Integration. Fig. 7(a) further demonstrates that removing either prior consistently degrades performance across all metrics. Specifically, excluding the visible prior (W/O vis) causes a sharp drop in VIF, highlighting its importance in preserving fine-grained textures and structural fidelity. In contrast, the full ablation of both infrared and visible priors (W/O ir+vis) yields the worst performance, indicating that prior guidance is essential for balanced fusion quality.

Fig. 8 demonstrates the efectiveness of bimodal priors in orthogonal gating. The full model (a) achieves structured, spatially selective gating. Removing $P _ { v i s }$ (b) causes the gating to collapse into a near-uniform distribution, confirming its role in providing localized guidance. Conversely, removing $P _ { i r }$ (c) results in over-smoothed IR responses and a loss of cross-modal enhancement, highlighting its importance for discriminative performance.

Study on Gated Dynamic Expert Recalibration. Fig. 7(b) compares our GDER module against intuitive alternatives, such as applying patch-level rather than global-level loss to $P _ { \mathrm { v i s } }$ ("Distribution"). Empirical results confirm the superiority of our approach. Fig. 7(c) validates the GDER teacher framework, where the $" \mathrm { W / O }$ both" configuration shows the largest performance drop, underscoring the architecture’s importance. Additionally, a Pearson correlation of r = 0.46 between the Modality and Attention experts demonstrates efective functional decoupling. Our full model achieves the optimal score across all metrics, confirming that our dual-teacher strategy provides comprehensive, non-redundant guidance.

Fig. 9 visualizes expert activation maps and prior-guided prompts. Complementarity within one modality and distinct responses across modalities validate our decoupled specialization over naive fusion. In overexposed scenes, the attention expert captures structural details while the modality expert isolates thermal targets, proving they are synergistically coupled yet representationally decoupled. Guided by high-fidelity spatial anchors, our framework ensures robust cross-modal integration.

![](images/20b7688e77ccc3ab84e9d2903a352787261c131ac49955b61522fe7d5e61d166.jpg)  
(a)

![](images/9783d37d0fa4cf350e05ff7931db60a5cf47ece6662f85ea416ae19b4a50a5cd.jpg)  
(b)

![](images/00f3facaeaa2f6c467c7b9eb73af58ceba721aed622cfd8ec6666941750ff6e8.jpg)

![](images/36f1a6408c648d12107350e1e8040fca3379450b39b7849bbfbb3fb804c72c5e.jpg)  
(c)

Fig. 7: Ablation study of the proposed components. All metrics are normalized relative to the full model (Ours). (a) Impact of image priors: Dual priors show clear synergy in maintaining visual fidelity. (b) Structural alternatives to GDER: Comparison with simpler architectural designs. (c) Analysis of GDER teachers: (Left) Performance gains from Modality and Attention teachers. (Right) Correlation analysis verifying the functional decoupling and non-redundancy between experts.  
![](images/0ba744ed77637f373db9c2a4091e0609859b871a7a725ad937dc32077bbb248f.jpg)  
（a) Ours

![](images/0ddce777ba1c36c45d4890b2955682bacfd0e24ab35e81d553c03a6f77633459.jpg)  
（b) w/o Pvis

![](images/d330db2edd9ccfc68d6e06b534561370abb95bdc27ff056f45c7afeee7839fa4.jpg)  
（C) w/o Pir  
Fig. 8: Visual analysis of the gating mechanism. Visualization of cross-modal gating allocations, where ablating $P _ { v i s }$ or $P _ { i r }$ disrupts spatial discrimination and leads to uniform or blurred feature distributions.

Table 4: Quantitative results of OOD Verification on the DroneVehicle, including image fusion and downstream object detection results. (1st: Red, 2nd: Light Blue)
<table><tr><td>Tasks</td><td colspan="4">Image Fusion Metrics</td><td colspan="5">Object Detection</td></tr><tr><td>Methods</td><td> $| F Q I _ { E } \uparrow$ </td><td>VIF↑</td><td> $Q _ { a b f } \uparrow$ </td><td>MI↑</td><td>Car</td><td>Truck</td><td>Bus</td><td>Van</td><td> $m A P _ { 5 0  9 5 } \uparrow$ </td></tr><tr><td>DiTFuse (TPAMI&#x27;25)</td><td>0.3929</td><td>0.1556</td><td>0.3134</td><td>2.1892</td><td>0.382</td><td>0.354</td><td>0.578</td><td>0.234</td><td>0.387</td></tr><tr><td>FreqGAN (TCSVT’25)</td><td>0.5340</td><td>0.2775</td><td>0.4643</td><td>2.6642</td><td>0.601</td><td>0.490</td><td>0.710</td><td>0.360</td><td>0.540</td></tr><tr><td>FreeFusion (TPAMI&#x27;25)</td><td>0.4893</td><td>0.2690</td><td>0.4339</td><td>2.4806</td><td>0.558</td><td>0.377</td><td>0.659</td><td>0.260</td><td>0.463</td></tr><tr><td>LutFuse (ICCV&#x27;25)</td><td>0.5203</td><td>0.2955</td><td>0.4984</td><td>3.8038</td><td>0.571</td><td>0.411</td><td>0.681</td><td>0.292</td><td>0.489</td></tr><tr><td>SAGE (CVPR’25)</td><td>0.4443</td><td>0.1791</td><td>0.3636</td><td>2.4162</td><td>0.582</td><td>0.447</td><td>0.694</td><td>0.324</td><td>0.512</td></tr><tr><td>TIM (TPAMI&#x27;24)</td><td>0.2728</td><td>0.2211</td><td>0.3182</td><td>2.6823</td><td>0.622</td><td>0.567</td><td>0.729</td><td>0.463</td><td>0.595</td></tr><tr><td>MRFS (CVPR’24)</td><td>0.3832</td><td>0.1790</td><td>0.3268</td><td>2.4254</td><td>0.562</td><td>0.383</td><td>0.676</td><td>0.263</td><td>0.471</td></tr><tr><td>LRRNet (TPAMI&#x27;23)</td><td>0.2490</td><td>0.1937</td><td>0.2848</td><td>2.1306</td><td>0.568</td><td>0.400</td><td>0.666</td><td>0.245</td><td>0.470</td></tr><tr><td>DDFM (ICCV&#x27;23)</td><td>0.4549</td><td>0.2515</td><td>0.3829</td><td>2.5912</td><td>0.597</td><td>0.487</td><td>0.705</td><td>0.363</td><td>0.538</td></tr><tr><td>TarDal (CVPR&#x27;22)</td><td>0.3992</td><td>0.2160</td><td>0.3576</td><td>2.7220</td><td>0.578</td><td>0.451</td><td>0.694</td><td>0.329</td><td>0.513</td></tr><tr><td>ReCoNet (ECCV&#x27;22)</td><td>0.4129</td><td>0.1982</td><td>0.3395</td><td>2.3772</td><td>0.581</td><td>0.435</td><td>0.701</td><td>0.289</td><td>0.501</td></tr><tr><td>Ours</td><td>0.6282</td><td>0.2655</td><td>0.5269</td><td>2.9998</td><td>0.632 0.576 0.740 0.469</td><td></td><td></td><td></td><td>0.604</td></tr></table>

![](images/aa6349c823d2aa3ad3df7cc2b3d0bd1514953186cb346fd4c9a5428b3e8934d1.jpg)  
Fig. 9: Qualitative analysis of expert decoupling and feature visualization. We present the activation maps for the Modality and Attention experts across infrared and visible branches. The distinct focus of each expert—targeting either thermal saliency or structural textures—validates the functional complementarity of our MoE-based decoupling strategy.

## 4.5 Robustness and Out-of-Distribution (OOD) Verification

Prior-guided IVIF frameworks are often sensitive to prior quality, risking performance degradation in degraded scenes. To verify the resilience of P<sup>2</sup>Fusion, we evaluate it under challenging scenarios like extreme overexposure and smoke occlusion (Figs. 5, 9). Our framework mitigates prior dependency via the Teach-to-Fuse paradigm and its error-correction capability, as evidenced by SOTA results on M3FD and RoadScene without domain-specific fine-tuning (Table 1).

To test adaptability, we performed rigorous out-of-distribution (OOD) verification on the DroneVehicle dataset. This benchmark introduces a severe domain shift with aerial perspectives and severe visibility degradations. As summarized in Table 4, P²Fusion consistently maintains superiority without any prior domain exposure during training. Specifically, in image fusion, our method achieves SOTA on two metrics and second place on another among five primary indicators. More importantly, in downstream object detection, P²Fusion achieves a comprehensive SOTA across all categories, outperforming the secondbest competitor by 0.9% in $\mathrm { m A P _ { 5 0 - 9 5 } }$ . By dynamically mediating between intrinsic prompts and raw features, P²Fusion exhibits exceptional self-adaptability in unseen, hostile environments.

## 5 Conclusion

In this paper, we rethink the role of prior knowledge in infrared-visible image fusion and propose P²Fusion, a novel framework that shifts the fusion paradigm from static prior constraints to dynamic intrinsic prompting. $\mathrm { B y }$ introducing a Teach-to-Fuse mechanism, we successfully distill task-specific physical priors—thermal saliency and spatial quality—into learnable regulators that procedurally govern the fusion process. Central to our architecture is the Gated Dynamic Expert Recalibration (GDER) module, which adaptively mediates modal competition and rectifies feature conflicts through expert specialization.

Extensive evaluations across five benchmarks and various challenging scenarios—including overexposure, smoke occlusion, and rigorous out-of-distribution (OOD) verification demonstrate that P<sup>2</sup>Fusion not only achieves superior visual fidelity but also significantly enhances downstream perception. The remarkable generalization capability in unseen domains further underscores that our method efectively mitigates the inherent prior-dependency of traditional frameworks. Ultimately, this work provides a robust and versatile solution for multi-modal information integration, bridging the gap between high-level physical guidance and adaptive feature representation.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 62293543 and Grant 62322605.

## References

1. Chen, J., Yang, L., Yu, W., Gong, W., Cai, Z., Ma, J.: Sdsfusion: A semantic-aware infrared and visible image fusion network for degraded scenes. IEEE Transactions on Image Processing 34, 3139–3153 (2025). https://doi.org/10.1109/TIP.2025. 3571339

2. Cvejic, N., Bull, D., Canagarajah, N.: Region-based multimodal image fusion using ica bases. IEEE Sensors Journal 7(5), 743–751 (2007). https://doi.org/10.1109/ JSEN.2007.894926

3. Feng, J., Wu, A., Zheng, W.S.: Shape-erased feature learning for visible-infrared person re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22752–22761 (2023). https://doi. org/10.1109/CVPR52729.2023.02179

4. Gao, H., Cheng, B., Wang, J., Li, K., Zhao, J., Li, D.: Object classification using cnn-based fusion of vision and lidar in autonomous vehicle environment. IEEE Transactions on Industrial Informatics 14(9), 4224–4231 (2018). https://doi. org/10.1109/TII.2018.2822828

5. Gao, Y., Ma, S., Liu, J.: Dcdr-gan: A densely connected disentangled representation generative adversarial network for infrared and visible image fusion. IEEE Transactions on Circuits and Systems for Video Technology 33(2), 549–561 (2023). https://doi.org/10.1109/TCSVT.2022.3206807

6. Gao, Y., Dai, Y., Li, H., Ye, W., Chen, J., Chen, D., Zhang, D., He, T., Zhang, G., Han, J.: Cosurfgs: 3d surface gaussian splatting with collaborative distributed learning for large-scale scene reconstruction. International Journal of Computer Vision 134, 195 (2026). https://doi.org/10.1007/s11263-025-02627-9

7. Guilin Su, Yuqing Huang, C.Y., He, Z.: Cmfs: Clip-guided modality interaction for mitigating noise in multi-modal image fusion and segmentation. In: Proceedings of the Thirty-four International Joint Conference on Artificial Intelligence (2025)

8. Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T.: Mfnet: Towards realtime semantic segmentation for autonomous vehicles with multi-spectral scenes. In: Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 5108–5115 (2017). https://doi.org/10.1109/IROS.2017.8206396

9. Huang, Z., Liu, J., Fan, X., Liu, R., Zhong, W., Luo, Z.: Reconet: Recurrent correction network for fast and eficient multi-modality image fusion. In: Computer Vision – ECCV 2022. pp. 539–555. Springer Nature Switzerland, Cham (2022)

10. Ji, D., Wang, H., Hu, H., Gan, W., Wu, W., Yan, J.: Context-aware graph convolution network for target re-identification. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 1646–1654 (2021). https://doi.org/10.1609/aaai. v35i2.16257

11. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. arXiv:2304.02643 (2023)

12. Li, H., Qin, R., Zou, Z., He, D., Li, B., Dai, B., Zhang, D., Han, J.: Langsurf: Language-embedded surface gaussians for 3d scene understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence (01), 1–13 (2026). https://doi.org/10.1109/TPAMI.2026.3701287

13. Li, H., Wu, X.J.: Densefuse: A fusion approach to infrared and visible images. IEEE Transactions on Image Processing 28(5), 2614–2623 (2019). https://doi. org/10.1109/TIP.2018.2887342

14. Li, H., Wu, X.J., Kittler, J.: Rfn-nest: An end-to-end residual fusion network for infrared and visible images. Information Fusion 73, 72–86 (2021). https: //doi.org/10.1016/j.inffus.2021.02.023, https://www.sciencedirect.com/ science/article/pii/S1566253521000440

15. Li, H., Xu, T., Wu, X.J., Lu, J., Kittler, J.: Lrrnet: A novel representation learning guided fusion network for infrared and visible images. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(9), 11040–11052 (2023). https://doi.org/10.1109/TPAMI.2023.3268209

16. Li, J., Jiang, C., Jiang, J., Liang, P., Ma, J., Nie, L.: Towards unified semantic and controllable image fusion: A difusion transformer approach. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–18 (2025). https://doi.org/10. 1109/TPAMI.2025.3642842

17. Li, Q., Lu, L., Li, Z., Wu, W., Liu, Z., Jeon, G., Yang, X.: Coupled gan with relativistic discriminators for infrared and visible images fusion. IEEE Sensors Journal 21(6), 7458–7467 (2021). https://doi.org/10.1109/JSEN.2019.2921803

18. Liu, J., Fan, X., Huang, Z., Wu, G., Liu, R., Zhong, W., Luo, Z.: Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5792–5801 (2022). https://doi.org/10.1109/CVPR52688.2022.00571

19. Liu, J., Fan, X., Jiang, J., Liu, R., Luo, Z.: Learning a deep multi-scale feature ensemble and an edge-attention guidance for image fusion. IEEE Transactions on Circuits and Systems for Video Technology 32(1), 105–119 (2022). https://doi. org/10.1109/TCSVT.2021.3056725

20. Liu, J., Li, X., Wang, Z., Jiang, Z., Zhong, W., Fan, W., Xu, B.: Promptfusion: Harmonized semantic prompt learning for infrared and visible image fusion. IEEE/CAA Journal of Automatica Sinica 12(3), 502–515 (2025). https: //doi.org/10.1109/JAS.2024.124878

21. Liu, J., Lin, R., Wu, G., Liu, R., Luo, Z., Fan, X.: Coconet: Coupled contrastive learning network with multi-level feature ensemble for multi-modality image fusion. International Journal of Computer Vision 132(5), 1748–1775 (2023). https://doi. org/10.1007/s11263-023-01952-1, https://doi.org/10.1007/s11263-023- 01952-1

22. Liu, J., Liu, Z., Wu, G., Ma, L., Liu, R., Zhong, W., Luo, Z., Fan, X.: Multiinteractive feature learning and a full-time multi-modality benchmark for image fusion and segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8081–8090 (2023). https://doi.org/10.1109/ ICCV51070.2023.00745

23. Liu, J., Wu, G., Liu, Z., Wang, D., Jiang, Z., Ma, L., Zhong, W., Fan, X., Liu, R.: Infrared and visible image fusion: From data compatibility to task adaption. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(4), 2349–2369 (2025). https://doi.org/10.1109/TPAMI.2024.3521416

24. Liu, J., Zhang, B., Mei, Q., Li, X., Zou, Y., Jiang, Z., Ma, L., Liu, R., Fan, X.: Dcevo: Discriminative cross-dimensional evolutionary learning for infrared and visible image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2226–2235 (2025). https: //doi.org/10.1109/CVPR52734.2025.00213

25. Liu, R., Liu, Z., Liu, J., Fan, X.: Searching a hierarchically aggregated fusion architecture for fast multi-modality image fusion. In: Proceedings of the 29th ACM International Conference on Multimedia. p. 1600–1608. MM ’21, Association for

Computing Machinery, New York, NY, USA (2021). https://doi.org/10.1145/ 3474085.3475299, https://doi.org/10.1145/3474085.3475299

26. Liu, R., Liu, Z., Liu, J., Fan, X., Luo, Z.: A task-guided, implicitly-searched and meta-initialized deep model for image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(10), 6594–6609 (2024). https://doi.org/10. 1109/TPAMI.2024.3382308

27. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. Proceedings of the IEEE/CVF International Conference on Computer Vision pp. 9992–10002 (2021), https://api.semanticscholar.org/CorpusID:232352874

28. Long, Y., Jia, H., Zhong, Y., Jiang, Y., Jia, Y.: Rxdnfuse: A aggregated residual dense network for infrared and visible image fusion. Information Fusion 69, 128– 141 (2021). https://doi.org/10.1016/j.inffus.2020.11.009, https://www. sciencedirect.com/science/article/pii/S1566253520304152

29. Ma, J., Liang, P., Yu, W., Chen, C., Guo, X., Wu, J., Jiang, J.: Infrared and visible image fusion via detail preserving adversarial learning. Information Fusion 54, 85–98 (2020). https://doi.org/10.1016/j.inffus.2019.07.005, https: //www.sciencedirect.com/science/article/pii/S1566253519300314

30. Ma, J., Tang, L., Fan, F., Huang, J., Mei, X., Ma, Y.: Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal of Automatica Sinica 9(7), 1200–1217 (2022). https://doi.org/10.1109/ JAS.2022.105686

31. Ma, J., Tang, L., Xu, M., Zhang, H., Xiao, G.: Stdfusionnet: An infrared and visible image fusion network based on salient target detection. IEEE Transactions on Instrumentation and Measurement 70, 1–13 (2021). https://doi.org/10. 1109/TIM.2021.3075747

32. Ma, J., Zhang, H., Shao, Z., Liang, P., Xu, H.: Ganmcc: A generative adversarial network with multiclassification constraints for infrared and visible image fusion. IEEE Transactions on Instrumentation and Measurement 70, 1–14 (2021). https: //doi.org/10.1109/TIM.2020.3038013

33. Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing 21(12), 4695–4708 (2012). https://doi.org/10.1109/TIP.2012.2214050

34. Petrovic, V., Xydeas, C.: Gradient-based multiresolution image fusion. IEEE Transactions on Image Processing 13(2), 228–237 (2004). https://doi.org/10. 1109/TIP.2004.823821

35. Piella, G.: A general framework for multiresolution image fusion: from pixels to regions. Information Fusion 4(4), 259–280 (2003). https://doi.org/10.1016/ S1566-2535(03)00046-0, https://www.sciencedirect.com/science/article/ pii/S1566253503000460

36. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning (2021)

37. Rao, D., Xu, T., Wu, X.J.: Tgfuse: An infrared and visible image fusion approach based on transformer and generative adversarial network. IEEE Transactions on Image Processing pp. 1–1 (2023). https://doi.org/10.1109/TIP.2023.3273451

38. Sun, Y., Cao, B., Zhu, P., Hu, Q.: Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning. IEEE Transactions on Circuits and Systems for Video Technology pp. 1–1 (2022). https://doi.org/10.1109/TCSVT. 2022.3168279

39. Tang, L., Deng, Y., Ma, Y., Huang, J., Ma, J.: Superfusion: A versatile image registration and fusion network with semantic awareness. IEEE/CAA Journal of Automatica Sinica 9(12), 2121–2137 (2022). https://doi.org/10.1109/JAS.2022. 106082

40. Tang, L., Li, C., Ma, J.: Mask-difuser: A masked difusion model for unified unsupervised image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–18 (2025). https://doi.org/10.1109/TPAMI.2025.3609323

41. Tang, L., Li, C., Ma, J.: Mask-difuser: A masked difusion model for unified unsupervised image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–18 (2025)

42. Tang, L., Yuan, J., Ma, J.: Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network. Information Fusion 82, 28–42 (2022). https://doi.org/10.1016/j.inffus.2021.12.004, https://www.sciencedirect.com/science/article/pii/S1566253521002542

43. Tang, L., Yuan, J., Zhang, H., Jiang, X., Ma, J.: Piafusion: A progressive infrared and visible image fusion network based on illumination aware. Information Fusion 83-84, 79–92 (2022). https://doi.org/10.1016/j.inffus.2022.03.007

44. Tang, W., He, F., Liu, Y.: Ydtr: Infrared and visible image fusion via y-shape dynamic transformer. IEEE Transactions on Multimedia 25, 5413–5428 (2023). https://doi.org/10.1109/TMM.2022.3192661

45. Tang, W., He, F., Liu, Y., Duan, Y., Si, T.: Datfuse: Infrared and visible image fusion via dual attention transformer. IEEE Transactions on Circuits and Systems for Video Technology 33(7), 3159–3172 (2023). https://doi.org/10.1109/TCSVT. 2023.3234340

46. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 6000–6010 (2017)

47. Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M.: Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In: Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7464– 7475 (2023). https://doi.org/10.1109/CVPR52729.2023.00721

48. Wang, Z., Zhang, Z., Qi, W., Yang, F., Xu, J.: Freqgan: Infrared and visible image fusion via unified frequency adversarial learning. IEEE Transactions on Circuits and Systems for Video Technology 35(1), 728–740 (2025). https://doi.org/10. 1109/TCSVT.2024.3460172

49. Wu, G., Liu, H., Fu, H., Peng, Y., Liu, J., Fan, X., Liu, R.: Every sam drop counts: Embracing semantic priors for multi-modality image fusion and beyond. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17882–17891 (2025). https://doi.org/10.1109/CVPR52734.2025.01666

50. Xiao, Q., Jin, H., Su, H., Zhang, Y., Xiao, Z., Wang, B.: Spdfusion: A semantic prior knowledge-driven method for infrared and visible image fusion. IEEE Transactions on Multimedia 27(1), 1691–1705 (2025). https://doi.org/10.1109/TMM.2024. 3521848

51. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers. In: Neural Information Processing Systems (NeurIPS) (2021)

52. Xu, H., Ma, J., Jiang, J., Guo, X., Ling, H.: U2fusion: A unified unsupervised image fusion network. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(1), 502–518 (2022). https://doi.org/10.1109/TPAMI.2020.3012548

53. Xu, H., Ma, J., Zhang, X.P.: Mef-gan: Multi-exposure image fusion via generative adversarial networks. IEEE Transactions on Image Processing 29, 7203–7216 (2020). https://doi.org/10.1109/TIP.2020.2999855

54. Yang, B., Li, S.: Multifocus image fusion and restoration with sparse representation. IEEE Transactions on Instrumentation and Measurement 59(4), 884–892 (2010). https://doi.org/10.1109/TIM.2009.2026612

55. Yang, K., Xiang, W., Chen, Z., Zhang, J., Liu, Y.: A review on infrared and visible image fusion algorithms based on neural networks. Journal of Visual Communication and Image Representation 101, 104179 (2024). https://doi.org/10.1016/j. jvcir.2024.104179, https://www.sciencedirect.com/science/article/pii/ S1047320324001342

56. Yi, X., Xu, H., Zhang, H., Tang, L., Ma, J.: Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 27016– 27025 (2024). https://doi.org/10.1109/CVPR52733.2024.02552

57. Yi, X., Zhang, Y., Xiang, X., Yan, Q., Xu, H., Ma, J.: Lut-fuse: Towards extremely fast infrared and visible image fusion via distillation to learnable look-up tables. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025)

58. Yuan, B., Sun, H., Guo, Y., Liu, Q., Zhan, X.: Explainable analysis of infrared and visible light image fusion based on deep learning. Scientific Reports 15(1), 2223 (2025). https://doi.org/10.1038/s41598-024-79684-6, https://doi.org/10. 1038/s41598-024-79684-6

59. Zhang, H., Zuo, X., Jiang, J., Guo, C., Ma, J.: Mrfs: Mutually reinforcing image fusion and segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26964–26973 (2024). https://doi.org/ 10.1109/CVPR52733.2024.02547

60. Zhao, W., Cui, H., Wang, H., He, Y., Lu, H.: Freefusion: Infrared and visible image fusion via cross reconstruction learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(9), 8040–8056 (2025). https://doi.org/10.1109/ TPAMI.2025.3572599

61. Zhao, W., Xie, S., Zhao, F., He, Y., Lu, H.: Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13955– 13965 (2023). https://doi.org/10.1109/CVPR52729.2023.01341

62. Zhao, Z., Bai, H., Zhang, J., Zhang, Y., Xu, S., Lin, Z., Timofte, R., Van Gool, L.: Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5906–5916 (2023). https://doi.org/10.1109/ CVPR52729.2023.00572

63. Zhao, Z., Bai, H., Zhang, J., Zhang, Y., Zhang, K., Xu, S., Chen, D., Timofte, R., Van Gool, L.: Equivariant multi-modality image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25912– 25921 (2024). https://doi.org/10.1109/CVPR52733.2024.02448

64. Zhao, Z., Bai, H., Zhu, Y., Zhang, J., Xu, S., Zhang, Y., Zhang, K., Meng, D., Timofte, R., Van Gool, L.: Ddfm: Denoising difusion model for multi-modality image fusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8082–8093 (October 2023)

65. Zheng, N., Zhou, M., Huang, J., Hou, J., Li, H., Xu, Y., Zhao, F.: Probing synergistic high-order interaction in infrared and visible image fusion. In: Proceedings

of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26374–26385 (2024). https://doi.org/10.1109/CVPR52733.2024.02492