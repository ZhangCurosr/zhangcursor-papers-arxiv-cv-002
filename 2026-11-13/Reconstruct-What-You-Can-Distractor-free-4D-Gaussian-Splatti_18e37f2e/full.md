# Reconstruct What You Can: Distractor-free 4D Gaussian Splatting

Joel Julin<sup>1</sup> Liuyue Xie<sup>1</sup> Ananya Bal<sup>1</sup> Mose Sakashita<sup>2</sup> Koichiro Niinuma<sup>2</sup> Laszlo A. Jeni<sup>1</sup> <sup>1</sup>Carnegie Mellon University <sup>2</sup>Fujitsu Research of America

https://recan-gs.github.io

![](images/a355c17db79e77cf2a2874c60d7f61b8b7d88588beb62e2715df865b66dd51d8.jpg)  
Figure 1. From monocular frames of noisy, dynamic scenes (a), our method decomposes reconstructable (green) and unreconstructable motion (red) (b), enabling distractor-free dynamic reconstruction (c) while preserving fine-scale details.

## Abstract

Dynamic reconstruction from casual video must preserve valid subject motion while suppressing transient occluders, shadows, and volumetric effects. Existing methods either suppress valid motion through static-scene assumptions or model all observations jointly, producing ghosting and unstable geometry. We introduce ReCan-GS, a 4D Gaussian Splatting framework that separates coherent scene content from inconsistent observations. It derives a Jacobian-based deformation-sensitivity score from the learned deformation field, using it as an empirical proxyfor reconstruction ambiguity. The score guides separation into dynamic Core Gaussians and per-view Distractor Gaussians, or post-hoc pruning of sensitive Core Gaussians. The same criterion supports removal of spatial occluders, photometric artifacts, and evolving smoke or fog without masks, pretrained segmenters, or medium-specific supervision. Experiments on synthetic and real dynamic scenes and synthetic smoke/fog scenes demonstrate its usefulness across the evaluated distractor regimes.

## 1. Introduction

Reconstructing dynamic 3D scenes from unconstrained monocular video remains challenging because real-world sequences often contain transient occluders, reflections, shadows, and flickers. These effects can introduce conflicting constraints on a shared scene representation. Standard dynamic reconstruction methods attempt to model them jointly with the subject, which can produce ghosting and unstable geometry.

Static distractor-removal methods avoid this failure mode by treating dynamic content as noise, but therefore fail when the moving object is the reconstruction target. We instead define distractors by deformation-space instability rather than object identity, speed, or appearance. This lets reconstructable motion remain in the core scene while inconsistent content is routed away.

The same criterion also handles slow-moving volumetric distractors such as fog and smoke: although they may appear photometrically stable, their evolving appearance can induce inconsistent deformation constraints and high scores, enabling removal by post-hoc pruning without fogspecific design.

To this end, we present ReCan-GS, a 4D Gaussian Splatting framework for distractor removal in dynamic scenes. ReCan-GS derives a motion-stability uncertainty directly from the Jacobian of the learned deformation field. This signal is intended to highlight observations associated with sensitive or inconsistent deformation, allowing the model to route stable content into Core Gaussians and inconsistent content into a separate Per-View Distractor layer. ReCan-GS is primarily designed for dynamic scenes with transient distractors and for volumetric fog and smoke removal; its performance is more limited on standard static distractor benchmarks.

Our main contributions are:

• A Jacobian-derived uncertainty metric that quantifies deformation-space instability in a continuous 4D Gaussian deformation field, providing a routing cue without explicit appearance models or speed thresholds.

• A unified distractor-removal framework for 4D Gaussian Splatting that separates geometrically consistent content from inconsistent observations, covering dynamic-scene occluders, photometric artifacts, and dynamic volumetric distractors such as fog and smoke.

• Empirical validation across synthetic and real dynamic scenes with distractors, plus fog/smoke scenes with clean ground truth.

## 2. Related Works

## 2.1. Handling Distractors

Recent research (Table 1) addresses the challenge of transient distractors within predominantly static scene assumptions. Approaches generally fall into three categories based on their distractor-handling strategies:

Photometric and Uncertainty-Based Methods identify distractors through reconstruction error or predictive uncertainty. NeRF-W [19] first separated static and transient components. BayesRays [6] extended this with probabilistic uncertainty, while FisherRF [9] used Fisher information. RobustNeRF [28] uses robust statistics to mask high-error regions. Related work addresses reflections and inconsistent visibility through appearance and regularization strategies [1, 17, 22, 34]. Concurrent work [7] estimates per-Gaussian reliability from recurring observations for motion propagation under occlusion; our score instead measures deformation-field sensitivity to separate transient content.

Semantic Feature-Based Methods replace photometric cues with semantic information from pretrained vision models. In the NeRF domain, NeRF-HuGS [3] uses semantic segmentation, and NeRF On-the-go [27] leverages DINOv2-derived [23] features. This trend extends to 3DGS, where SpotLessSplats [29] clusters text-to-image embeddings and Robust 3D Gaussian Splatting [32] incorporates SAM-derived segmentations [11].

Decompositional Methods explicitly separate static and transient content into distinct representational layers. Several methods model appearance variation using learned embeddings or fields (SWAG [4], GS-W [46], Wild-GS [42]). Others use optimization strategies, such as WildGaussians [12], which uses uncertainty masks for gradient scaling.

A more explicit decomposition is used by DeSplat [36], which models transients with per-view splats, and HybridGS [16], which combines persistent 3D Gaussians with view-specific 2D Gaussians. Splatfacto-W [41] also uses a hybrid approach with MLPs for background modeling and robust masks for transients.

While these methods are effective in their respective contexts, they share a common limitation: all assume a static scene and ultimately aim to recover a single clean reconstruction, with uncertainty or transient handling largely heuristic or learned. In contrast, we address fully dynamic scenes corrupted by severe spatial and photometric inconsistencies and, importantly, show that uncertainty can be derived analytically, offering a principled and explainable view of the optimization process that prior work lacks.

<table><tr><td>Property</td><td>S ]</td><td>GGS] -466]</td><td>2] -]</td><td>SPl] a-1]</td><td>il] is2]</td><td>[] pesats</td><td>Deε] l36]</td><td>Ours</td></tr><tr><td>No pre-trained model Fast 3DGS rendering</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>√</td><td></td><td>vvvvv&gt;</td></tr><tr><td>Explicit scene separation</td><td>x x</td><td>x x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>vv√</td><td></td></tr><tr><td></td><td></td><td></td><td>x</td><td>x</td><td>x</td><td>√</td><td></td><td></td></tr><tr><td>Dynamic scene reconstruction</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td></td></tr><tr><td>Uncertainty Metric</td><td>x</td><td>x</td><td>x x</td><td>x x</td><td>x</td><td>x x</td><td>x</td><td></td></tr><tr><td>Fog/Smoke Removal</td><td>x</td><td>x</td><td></td><td></td><td>x</td><td></td><td>x</td><td></td></tr></table>

Table 1. Feature comparison. Our method uniquely supports distractor removal in fully dynamic scenes without pretraining.

## 2.2. Fog and Volumetric Distractor Removal

Volumetric distractors such as fog and smoke further challenge appearance-based removal, since they may appear photometrically stable while remaining geometrically inconsistent across views. Our method instead uses deformation sensitivity as a cue for removal without fog-specific supervision. The experiments evaluate this mechanism against general distractor-removal baselines; they do not establish competitiveness with dedicated smoke-removal methods.

## 2.3. Dynamic Scene Reconstruction

Reconstructing general dynamic scenes is a distinct but related challenge. Methods such as DeformableGS [43], 4DGS [39], GSLK [40], DynamicGS [18], STGS [15], Realtime4DGS [44], and dynamic NeRF variants [5, 24–26] model time-varying scenes by learning motion jointly with appearance. However, they lack an explicit mechanism to separate transient distractors from core scene motion. As a result, they tend to average all visual phenomena, leading to severe artifacts such as ghosting when occluders are present.

More recent approaches, such as Shape of Motion [35] and MoSca [13], leverage strong externally-derived priors from vision foundation models (e.g., point tracking or depth estimation) to regularize the motion field. While effective, these methods introduce additional complexity and depend heavily on the quality of the pretrained models. In contrast, our approach is orthogonal: we propose an analytical, priorfree method that explicitly separates geometrically consistent motion from temporal multi-view contradictions using an uncertainty signal derived directly from the deformation field, improving both robustness and interpretability.

![](images/2c53a4d264dac01039867280b9d70a13536a36870c0fb36f855f3eb416cc95db.jpg)  
Figure 2. An overview of our ReCan-GS pipeline. The scene is modeled with two components: dynamic Core Gaussians (G) and per-view Distractor Gaussians (G<sub>dist</sub>). An MLP deforms the core Gaussians over time (t). A Jacobian-based uncertainty signal, derived from the deformation field, guides the optimization to penalize unstable motion in the core layer, effectively forcing transient effects into the distractor layer.

## 3. Preliminary

3DGS [10] represents a scene as a set of 3D Gaussians, each defined by a position $\mu ,$ covariance Σ, opacity $\alpha ,$ and view-dependent color via spherical harmonics. Gaussians are initialized from a sparse SfM point cloud [20] and adaptively split or pruned during training. Pixel colors are computed by alpha-compositing depth-sorted 2D Gaussian projections:

$$
\mathbf { C } = \sum _ { i = 1 } ^ { N } T _ { i } \big ( 1 - \exp ( - \sigma _ { i } \delta _ { i } ) \big ) \mathbf { c } _ { i } , \quad T _ { i } = \exp \Bigl ( - \sum _ { j = 1 } ^ { i - 1 } \sigma _ { j } \delta _ { j } \Bigr ) .\tag{1}
$$

For dynamic scenes, a common approach is to learn a deformation field mapping a canonical representation to its configuration at a given time [18, 24, 26, 45]. We adopt this strategy for Gaussian Splatting following [43], using an MLP that maps each canonical position $\mu _ { i }$ and time t to position, rotation, and scale offsets.

## 4. Method

To robustly reconstruct dynamic scenes with transient distractors, ReCan-GS decomposes the scene into two geometric layers: dynamic Core Gaussians modeling stable, temporally consistent content, and per-view Distractor Gaussians capturing transient or view-specific phenomena. The separation is driven by a motion-derived uncertainty signal from the learned deformation field, reinforced by competition between the two layers during optimization, and requires no masks or annotations. Fig. 2 shows the pipeline.

Definition of Distractors. We operationally identify candidate distractors as observations that induce inconsistent deformation constraints in the shared canonical space. Routing uses deformation sensitivity rather than an explicit threshold on speed or object identity. This criterion does not determine the intended subject: a well-observed distractor can admit stable deformation, while poorly observed target content can receive a high score. The separation therefore depends on observation coverage and the learned representation.

Sec. 5.3 evaluates coverage and speed (Fig. 5), pruning sensitivity (Fig. 7), and distractor localization; smoke results provide a further test of the same score (Fig. 4).

## 4.1. Decomposed 4D Gaussian Splatting

Our method represents the scene with two distinct sets of 3D Gaussians, drawing inspiration from decompositional ideas explored in static settings such as DeSplat [36].

Core Gaussians. The first component, the Core Gaussians $G _ { c o r e }$ , models the stable scene content, including the static background and the primary dynamic subject. These Gaussians are defined in a canonical space and made dynamic via a learned deformation field N, implemented as an 8-layer, 256-wide MLP. For a given canonical position $\mu _ { i }$ and time t, the MLP predicts deformation offsets:

$$
N ( \mu _ { i } , t ) = ( \Delta \mu _ { i } , \Delta q _ { i } , \Delta s _ { i } )\tag{2}
$$

These offsets update the position, rotation, and scale of each Gaussian to reflect its configuration at time t, enabling complex, non-rigid motion. This MLP architecture is identical to [43].

Distractor Gaussians. We adopt the per-view distractor formulation of DeSplat [36], which provides a simple mechanism for modeling view-dependent transients. For completeness, we summarize the components here, noting that our primary contribution lies in the uncertainty-guided routing described in Sec. 4.2.

Initialization. Each training view n maintains its own distractor set $G _ { d i s t , n }$ . We initialize K Gaussians on a shallow camera-aligned plane via:

$$
{ \pmb \mu } _ { d } = { \bf t } _ { n } - R _ { n } ^ { \top } { \bf u } , \qquad { \bf u } = \left[ \begin{array} { l } { \rho u } \\ { \rho v } \\ { \rho } \end{array} \right] , \quad u , v \sim \mathcal { U } ( 0 , 1 ) ,\tag{3}
$$

where $\rho$ controls the plane depth. Colors are parameterized as RGB vectors to reflect their view-dependent nature.

Densification. Distractor Gaussians undergo per-view Adaptive Density Control based on view-space gradients. Updates occur only when a per-view counter $s _ { n }$ exceeds threshold S, allowing each view to refine its own distractor set independently.

Compositing. The core and distractor layers are rendered separately and then combined via standard alpha compositing:

$$
c _ { \mathrm { c o m p } } = c _ { \mathrm { d i s t } } + \left( 1 - \alpha _ { \mathrm { d i s t } } \right) c _ { \mathrm { c o r e } } .\tag{4}
$$

This enables distractors to occlude the core where appropriate.

## 4.2. Guiding Decomposition with Scene Dynamics

A standard photometric loss is insufficient to enforce a meaningful separation, as it encourages the powerful Core model $G _ { \mathrm { c o r e } }$ to overfit to all visual phenomena. To resolve this, we leverage a Jacobian-derived uncertainty score to actively guide the decomposition, identifying geometrically inconsistent motion for routing to $G _ { \mathrm { d i s t } }$ or removal.

Jacobian Derived Motion Uncertainty. Our guiding signal is derived from the core deformation field N and measures variation among its output gradients. We use this deformation-sensitivity score as an empirical proxy for reconstruction ambiguity, referring to it as Jacobian uncertainty. It is not a calibrated probabilistic uncertainty estimate: complex but correctly reconstructed deformation may also yield a high score.

In the controlled experiment of Sec. 5.3, increasing camera coverage reduces the score at both tested motion speeds. This supports its usefulness beyond raw velocity, but does not guarantee that valid motion will always receive low uncertainty.

For each Core Gaussian $g _ { i } ,$ , we compute the Jacobian of the predicted offsets $( \Delta \mu _ { i } , \Delta q _ { i } , \Delta s _ { i } )$ with respect to the inputs (canonical position $\mu _ { i }$ and time t). For example, for the position offset $\Delta \mu .$ , the Jacobian $J _ { \Delta \mu }$ is the matrix of partial derivatives:

$$
J _ { \Delta \mu } = \frac { \partial \Delta \mu } { \partial ( \mu , t ) }\tag{5}
$$

Our uncertainty score is derived from the sample covariance of the Jacobian’s rows, which represent the gradients of each output dimension with respect to the inputs. High variance among these gradients indicates heterogeneous local sensitivity, which we use as a routing cue. The uncertainty contribution from a single parameter type (e.g., position) is the trace of this covariance matrix $U _ { \Delta \mu } ( g _ { i } ) =$ Tr $\left( \mathrm { C o v } ( J _ { \Delta \mu } ) \right)$

The trace represents the sum of the variances of its constituent variables, providing a unified measure of deformation sensitivity. In practice, we compute this Jacobian using a stochastically perturbed time input $t ^ { \prime } = t + \epsilon$ for a robust local estimate. The total uncertainty score for a Gaussian $g _ { i }$ is the sum of the scores for each parameter type:

$$
U ( g _ { i } ) = U _ { \Delta \mu } ( g _ { i } ) + U _ { \Delta q } ( g _ { i } ) + U _ { \Delta s } ( g _ { i } )\tag{6}
$$

Uncertainty-Guided Methods Once high-uncertainty regions are detected, we use one of two methods to suppress their influence in the core model and encourage a handover to the distractor model.

Variant 1: Loss-Based Regularization. Our primary approach integrates the uncertainty score directly into the optimization process. We adopt the alpha-based regularization from DeSplat [36], which encourages a sparse distractor layer and an opaque core layer $( \mathcal { L } _ { \mathrm { { a l p h a } } } = \lambda _ { s } | 1 - \alpha _ { \mathrm { { c o r e } } } | +$ $\lambda _ { d } | \alpha _ { \mathrm { d i s t } } | )$ , we introduce an uncertainty-aware regularizer:

$$
{ \mathcal { L } } _ { \mathrm { u n c } } = \sum _ { g _ { i } \in G _ { \mathrm { c o r e } } } U ( g _ { i } ) \cdot \alpha _ { \mathrm { c o r e } , i }\tag{7}
$$

This term penalizes core Gaussians that are both uncertain and opaque, encouraging uncertain regions to transfer to the distractor layer. The final loss function combines these terms:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { p h o t o } + \mathcal { L } _ { a l p h a } + \lambda _ { u n c } \mathcal { L } _ { u n c }\tag{8}
$$

Variant 2: Post-Hoc Uncertainty Pruning. For severe clutter, we also consider a hard post-training separation. In this approach, we omit the uncertainty regularizer ${ \mathcal { L } } _ { \mathrm { u n c } }$ during training and instead track the stability of each core Gaussian throughout optimization. We accumulate uncertainty using an exponential moving average (EMA):

$$
U _ { t } ( g _ { i } ) = \beta U _ { t - 1 } ( g _ { i } ) + ( 1 - \beta ) U _ { \mathrm { i n s t } } ( g _ { i } ) ,\tag{9}
$$

where $U _ { \mathrm { i n s t } } ( g _ { i } )$ is the instantaneous uncertainty at iteration t and $\beta$ is a momentum parameter. After training converges, any Core Gaussian whose accumulated uncertainty exceeds a threshold is removed:

$$
g _ { i } \in \mathrm { t o p } \substack { - r \% \{ U _ { t } ( g _ { j } ) \} _ { j } . }\tag{10}
$$

This pruning step eliminates residual unstable geometry that may remain when relying solely on soft regularization.

Input DeformableGS [43] 4DGS [39] SpotLessSplats [29] DeSplat [36] Ours (Distractor) Ours (Core)  
![](images/e8b6c87d4149032ea58f558a9e89aa3e9c0b98fc401ac5052d33e9836209289f.jpg)  
Figure 3. Qualitative results on dynamic scenes with distractors. Columns 2–3 are dynamic reconstruction methods that jointly model all motion, often producing ghosting from transient occluders. Columns 4–5 are standard distractor-removal methods that assume a static scene and therefore struggle with valid dynamic content. ReCan-GS explicitly decomposes inconsistent observations into a distractor laye and recovers a clean, stable core reconstruction.

As demonstrated in Fig. 7, adjusting the pruning threshold hierarchically isolates topological artifacts, ranging from severe spatial occluders to subtle photometric inconsistencies (evaluated in Sec. 5.3).

## 5. Experiments

To validate ReCan-GS, we evaluate across three distractor regimes that stress qualitatively different aspects of the method.

## 5.1. Experimental Setup

Datasets. Dynamic scenes test robustness to spatial and photometric occlusions; smoke scenes isolate deformation instability independent of apparent motion.

Dynamic scenes with distractors. We utilize four scenes featuring transient occluders and lighting artifacts:

• Synthetic: Bouncing Balls (three balls occluded by a passing runner; Blender [2]) and Squatting (a subject partially occluded by a running figure; Unity [33]). We apply our loss-based regularization (Variant 1) to these scenes.

• Real-world: Two iPhone 15 Pro captures, Jeep (stopmotion with heavy hand occlusions and moving cast shadows) and Bottle (frequently occluded by a walking person). We apply our post-hoc pruning (Variant 2) to these captures.

Smoke scenes. To explicitly test our method’s response to complex topology changes with minimal rigid motion, we evaluate on four synthetic scenes from SmokeSeer [8]: Muscle Car, Chair, Hotdog, and Lego. Generated with Blender’s Mantaflow simulator over standard NeRF objects [21], they pair unstable smoke observations with smoke-free ground truth.

Static distractor benchmark. To test generalization to standard static distractor removal, we evaluate on the NeRF On-the-go benchmark [27], treating all dynamic in-the-wild content as transient.

Baselines and Metrics. We compare against dynamic scene methods (DeformableGS [43], 4DGS [39]) and specialized static-scene distractor-removal techniques (De-Splat [36], SpotLessSplats [29], WildGaussians [12], NeRF On-the-go [27], AsymGS [14]). Performance is quantified via PSNR(↑) [38], SSIM(↑) [37], and LPIPS(↓) [47].

Evaluation Protocol and Implementation. For synthetic scenes, metrics are computed against clean full-sequence re-renders. For real scenes, we use distractor-free holdout subsets of 30–58 frames per scene, selected manually by one binary criterion (the distractor is fully absent) and shared identically across all methods; frame lists are released with the code. For the distractor-retention evaluation, we query three VLMs (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro) with the input frame and the Core render, without indicating which method produced it, and ask for a 1–5 severity rating. Three human annotators independently rate 50 of these frames per method using the same rubric and image pairs. Distractor-present frames are determined from SAM masks independently of any method. The prompt is on the project website. Camera poses are obtained with COLMAP [30]. We train 40k iterations on synthetic scenes and 60k on real scenes on a single NVIDIA H200, with K = 1000 (synthetic) and $K = 2 0 0 0$ (real) distractor Gaussians per view, per-view densification after S = 10 visits, and post-hoc pruning of the top 5 − 10% most uncertain Core Gaussians. We begin deformation optimization at 3k and distractor optimization at 7k iterations, use $\lambda _ { s } = 0 . 0 1$ and $\lambda _ { \mathrm { u n c } } = 0 . 3$ , warm $\lambda _ { d }$ from 0 to 0.01 over the next 10k iterations, and set $\beta = 0 . 9$ for post-hoc pruning. Across three random seeds, PSNR standard deviation is 0.53 dB on Bottle and 0.09 dB on Bouncing Balls.

![](images/0056c4deaf20b0466b76f42ea2eff00a9f38c4268da549effcc6b8baec3a4935.jpg)

Figure 4. Smoke removal results. ReCan-GS separates slow-moving volumetric distractors from the core scene by detecting deformation space instability, enabling clean smoke-free reconstructions where appearance-based methods leave residual haze.
<table><tr><td rowspan="2">Method</td><td colspan="3">Squatting</td><td colspan="3">Bouncing Balls</td><td colspan="3">Jeep</td><td colspan="3">Bottle</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td colspan="10">Static distractor-removal methods</td><td colspan="3"></td></tr><tr><td>WildGaussians [12]</td><td>24.3</td><td>0.88</td><td>0.15</td><td></td><td></td><td></td><td>25.4</td><td>0.90</td><td>0.18</td><td>18.2</td><td>0.67</td><td>0.33</td></tr><tr><td>SLS-MLP [29]</td><td>22.9</td><td>0.87</td><td>0.15</td><td>15.3</td><td>0.84</td><td>0.25</td><td>25.8</td><td>0.90</td><td>0.17</td><td>17.8</td><td>0.65</td><td>0.35</td></tr><tr><td>Robust3DGS [32]</td><td>24.4</td><td>0.90</td><td>0.11</td><td>24.6</td><td>0.97</td><td>0.07</td><td>26.6</td><td>0.92</td><td>0.07</td><td>17.3</td><td>0.67</td><td>0.30</td></tr><tr><td>DeSplat [36]</td><td>25.0</td><td>0.89</td><td>0.12</td><td>14.3</td><td>0.87</td><td>0.16</td><td>26.5</td><td>0.89</td><td>0.17</td><td>17.1</td><td>0.67</td><td>0.30</td></tr><tr><td>AsymGS [14]</td><td>24.7</td><td>0.88</td><td>0.12</td><td>21.7</td><td>0.92</td><td>0.13</td><td>26.3</td><td>0.91</td><td>0.09</td><td>17.7</td><td>0.69</td><td>0.25</td></tr><tr><td colspan="10">Dynamic reconstruction methods</td><td colspan="3"></td></tr><tr><td>DeformableGS [43]</td><td>31.5</td><td>.95</td><td>.07</td><td>27.8</td><td>0.96</td><td>0.05</td><td>30.3</td><td>.94</td><td>.05</td><td>21.6</td><td>0.75</td><td>0.26</td></tr><tr><td>4DGS [39]</td><td>27.2</td><td>0.79</td><td>0.27</td><td>22.3</td><td>0.93</td><td>0.09</td><td>29.2</td><td>0.91</td><td>0.16</td><td>16.7</td><td>0.50</td><td>0.75</td></tr><tr><td>ReCan-GS (Ours)</td><td>29.3</td><td>0.92</td><td>0.10</td><td>29.4</td><td>.98</td><td>.03</td><td>28.4</td><td>0.93</td><td>0.07</td><td>23.0</td><td>.78</td><td>.22</td></tr></table>

Table 2. Quantitative results on dynamic scenes with distractors, grouped into static distractor-removal and dynamic reconstruction methods. Higher is better for PSNR/SSIM and lower for LPIPS; best results per scene are in bold. Real-scene holdout frames are distractor-free, so these metrics do not fully measure retained distractor geometry; see Table 3.

## 5.2. Results

Qualitative Results. Fig. 3 compares against general dynamic methods (DeformableGS [43], 4DGS [39]) and static distractor-removal techniques (DeSplat [36], SpotLessSplats [29]). Dynamic methods jointly model all motion and produce blurred, semi-transparent ghosts where transient hands occlude the scene, while static methods damage the fidelity of the primary moving subject. ReCan-GS isolates transient elements into the distractor layer, leaving the core reconstruction clean and stable.

Quantitative Results. We evaluate ReCan-GS on its pri mary task: reconstructing clean dynamics amidst distractors. Table 2 compares against both static distractorremoval methods and general dynamic reconstruction meth ods. ReCan-GS substantially outperforms the static base lines across the dynamic-distractor scenes, remains competitive with DeformableGS in overall reconstruction fidelity, and outperforms 4DGS on all but one scene. On Bouncing Balls, ReCan-GS leads all methods at 29.4 dB, while on Bottle it achieves the best dynamic-method result at 23.0 dB. DeformableGS leads on Squatting and Jeep, and Bottle remains our hardest scene in absolute terms (SSIM 0.78, LPIPS 0.22). Holdout frames are selected precisely because the distractor is absent, so a method that absorbs the occluder into its deformation field may not be penalized for that geometry when it is absent from the evaluated renders. To assess retained distractors, we ask three human annotators and three vision-language judges, blind to method identity, to rate per frame how strongly the distrac tor is reproduced (Table 3). On Jeep, where DeformableGS leads on PSNR, they detect the occluder in 73–83% of its frames and 77–88% of 4DGS frames, versus 0–11% for ReCan-GS; no human annotator rated any ReCan-GS frame as showing the occluder. On Bottle, the corresponding rates are 57–77% for DeformableGS versus 6–26% for ReCan-GS. 4DGS leaks heavily despite the weakest reconstruction quality, indicating that retention and reconstruction fidelity capture different aspects of performance. Relative to DeformableGS, ReCan-GS loses 2.2 dB on Squatting and 1.9 dB on Jeep, while gaining 1.6 dB on Bouncing Balls and 1.4 dB on Bottle. Lower distractor retention therefore does not imply uniformly better fidelity. Real-scene retention ratings do not verify target completeness during occlusion; removing valid target content can also reduce apparent distractors.

ReCan-GS also performs strongly on volumetric distractor removal (Table 4, Fig. 4). While appearance-based methods often retain semi-transparent smoke as part of the scene, ReCan-GS exploits geometric multi-view instability to remove it. This yields gains of 5.7 dB and 6.2 dB over the strongest baselines on the Muscle Car and Hotdog scenes, respectively. These comparisons demonstrate benefits over the evaluated general distractor-removal methods; dedicated smoke-removal baselines are not included.

Computational Cost. ReCan-GS increases training cost relative to DeformableGS (16.23 vs. 9.22 min; 45.9 vs. 72.3 train FPS; 14.0 vs. 2.8 GB peak memory; 9.70M vs. 3.62M params), primarily due to the distractor layer and trainingtime Jacobian computation. It adds no inference overhead (rendering FPS is identical), and 14.0 GB fits on consumer

<table><tr><td></td><td></td><td></td><td colspan="3">VLM judges</td></tr><tr><td>Scene</td><td>Method</td><td>Human</td><td>GPT-5</td><td>Claude</td><td>Gemini</td></tr><tr><td rowspan="3">Bottle</td><td>4DGS [39]</td><td>3.41 (82.0)</td><td>2.09 (35.5)</td><td>3.15 (70.0)</td><td>3.12 (75.1)</td></tr><tr><td>DeformableGS [43]</td><td>3.37 (76.7)</td><td>2.60 (56.7)</td><td>3.23 (73.8)</td><td>3.10 (70.4)</td></tr><tr><td>ReCan-GS (Ours)</td><td>1.33 (9.3)</td><td>1.21 (5.6)</td><td>1.81 (26.0)</td><td>1.71 (24.7)</td></tr><tr><td rowspan="3"> $J e e p$ </td><td>4DGS [39]</td><td>3.43 (76.7)</td><td>3.42 (82.8)</td><td>3.68 (83.2)</td><td>3.63 (87.8)</td></tr><tr><td>DeformableGS [43]</td><td>3.29 (72.7)</td><td>3.38 (78.6)</td><td>3.87 (78.8)</td><td>3.46 (82.6)</td></tr><tr><td>ReCan-GS (Ours)</td><td>1.09 (0.0)</td><td>1.02 (0.9)</td><td>1.06 (1.7)</td><td>1.29 (11.0)</td></tr></table>

Table 3. Retained distractor geometry. Three human annotators and three VLM judges rate per frame how strongly the distractor is reproduced (1 = absent, 5 = fully reproduced); mean severity and (% of frames rated $\geq 3 ) .$ . VLMs score all distractor-present frames (Bottle: 534, Jeep: 345); the Human column averages three annotators over the same 50 frames per method, who agree closely (mean pairwise quadratic-weighted κ = 0.85, Spearman ρ = 0.87 over 900 paired ratings). On Jeep, no annotator rated any ReCan-GS frame ≥ 3. Per-annotator scores are on the project website.

![](images/5fd76e6ddd51be6d2720b2ff9e3a33f352c9cbf090c4a74a27a68733dd7a7482.jpg)

Figure 5. Uncertainty vs. Motion and Coverage. Uncertainty decreases as the number of camera views increases, indicating that the metric captures reconstructability rather than raw motion magnitude.  
![](images/dda9c5aa7972108b00b93097d0411584b8a9f82c9aed09ea2f736cdb80aceb56.jpg)  
Figure 6. Jacobian uncertainty vs. motion magnitude. Motion magnitude responds strongly to valid subject motion, whereas the Jacobian score more selectively localizes the transient distractor.

GPUs, so the H200 is not a requirement.

Spatial Localization of Transient Distractors. We evaluate the projected spatial support of Jacobian uncertainty on Bottle against SAM-derived masks, used only for evaluation and playing no role in the method. Across 534 distractor-containing frames, Jacobian uncertainty achieves 0.532 aggregate IoU and 0.483 median IoU, versus 0.284 and 0.071 for motion magnitude. As shown in Fig. 6, motion magnitude also responds to valid subject motion, whereas Jacobian uncertainty more selectively localizes the transient distractor. This comparison isolates the localization signal on one scene, but does not isolate the end-to-end benefit of Jacobian routing within the same Core–Distractor architecture.

<table><tr><td>Method</td><td colspan="3">Muscle Car</td><td colspan="3">Chair</td><td colspan="3">Hotdog</td><td colspan="3">Lego</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>SLS-MLP [29]</td><td>5.4</td><td>0.01</td><td>0.71</td><td>6.3</td><td>0.05</td><td>0.72</td><td>5.6</td><td>0.14</td><td>0.76</td><td>4.3</td><td>0.06</td><td>0.80</td></tr><tr><td>DeSplat [36]</td><td>11.2</td><td>0.19</td><td>0.46</td><td>8.6</td><td>0.05</td><td>0.63</td><td>8.6</td><td>0.19</td><td>0.60</td><td>8.8</td><td>0.14</td><td>0.59</td></tr><tr><td>Robust3DGS [32]</td><td>10.9</td><td>0.17</td><td>0.52</td><td>8.6</td><td>0.05</td><td>0.67</td><td>8.6</td><td>0.14</td><td>0.65</td><td>8.2</td><td>0.09</td><td>0.67</td></tr><tr><td>AsymGS [14]</td><td>11.1</td><td>0.18</td><td>0.48</td><td>8.7</td><td>0.06</td><td>0.65</td><td>8.6</td><td>0.15</td><td>0.65</td><td>8.5</td><td>0.09</td><td>0.66</td></tr><tr><td>Ours</td><td>16.8</td><td>.69</td><td>.26</td><td>10.9</td><td>.33</td><td>.50</td><td>14.8</td><td>.69</td><td>.30</td><td>12.5</td><td>.58</td><td>.39</td></tr></table>

Table 4. Fog/smoke removal on synthetic scenes with fog-free ground truth. Appearance-based methods misidentify volumetric fog as stable background; ReCan-GS removes it without any fog-specific design. Best results in bold.

## 5.3. Ablation Studies and Analysis

Generalization to Static Distractor Settings. Although ReCan-GS targets dynamic scenes, we verify it remains functional on standard static distractor benchmarks. Table 5 reports results on NeRF On-the-go [27], averaged over four scenes (Mountain, Fountain, Corner, Patio) with all motion treated as transient. ReCan-GS trails SLS-MLP by 2.6 dB PSNR, 0.15 SSIM, and 0.18 LPIPS. Its PSNR exceeds Splatfacto-W-A, but its SSIM and LPIPS are the weakest in this comparison. These results limit its applicability to static distractor removal; the present experiments do not isolate the cause of this gap.

<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>NeRF On-the-go [27]</td><td>21.32</td><td>0.70</td><td>0.25</td></tr><tr><td>Splatfacto [31]</td><td>21.28</td><td>0.76</td><td>0.15</td></tr><tr><td>Splatfacto-W-A [41]</td><td>20.54</td><td>0.72</td><td>0.18</td></tr><tr><td>Splatfacto-W-T [41]</td><td>21.22</td><td>0.74</td><td>0.16</td></tr><tr><td>SLS-MLP [29]</td><td>23.50</td><td>0.83</td><td>0.13</td></tr><tr><td>WildGaussians [12]</td><td>21.71</td><td>0.73</td><td>0.21</td></tr><tr><td>DeSplat [36]</td><td>21.70</td><td>0.77</td><td>0.14</td></tr><tr><td>Ours</td><td>20.92</td><td>0.68</td><td>0.31</td></tr></table>

Table 5. Static distractor generalization on NeRF On-the-go [27].

Validation of the Jacobian Uncertainty Metric. To verify our metric captures geometric ambiguity rather than raw velocity, we ablate camera coverage versus motion speed (Fig. 5). Across fixed speeds (“Slow” and “Fast”), increasing multi-view coverage from 1 to 3 views consistently drops the uncertainty score. This supports the score as a coverage-sensitive cue in this setting, rather than establishing a general separation of valid motion and distractors.

![](images/1a28f41f69cef3e30ef82704c0aea4048e48be93597de351e91c17e9e2aca2ed.jpg)  
Figure 7. Hierarchical Sensitivity. Post-hoc pruning removes the top 0%, 5%, 7%, and 15% most uncertain Core Gaussians, ranked by accumulated deformation sensitivity. Without pruning, transient hands and moving cast shadows remain. Removing the top 5% primarily suppresses the hands, while pruning the top 7% and 15% additionally suppresses moving cast shadows.

Hierarchical Sensitivity of the Uncertainty Metric. Sweeping the post-hoc pruning fraction across the Jeep sequence (Fig. 7) illustrates how deformation sensitivity ranks different artifacts. Removing the highest-scoring 5% of Core Gaussians primarily suppresses transient hands. Increasing the pruning fraction to 7% and 15% additionally suppresses moving cast shadows.

## 6. Conclusion

We introduced ReCan-GS, a 4D Gaussian Splatting framework that uses a Jacobian-derived deformation-sensitivity score to guide distractor separation. Across the evaluated dynamic and synthetic smoke scenes, the same score supports routing to a per-view Distractor layer or posthoc pruning without pretrained segmentation models or medium-specific supervision.

Limitations and Future Work. Deformation sensitivity is not calibrated uncertainty or distractor identity: stable transients may remain, while poorly observed targets or valid complex motion may be suppressed. Retention ratings do not verify target completeness during real occlusion. Evaluation covers four dynamic scenes (two real) and four synthetic smoke scenes, without dedicated smoke-removal baselines or a controlled routing ablation. Static-benchmark performance is limited (Table 5). Motion or semantic priors could help preserve intended targets.

## Acknowledgments

This research was supported by Fujitsu.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. ICCV, 2021. 2

[2] Blender Foundation. Blender. https://www.blender. org/, 2024. 5

[3] Jiahao Chen, Yipeng Qin, Lingjie Liu, Jiangbo Lu, and Guanbin Li. Nerf-hugs: Improved neural radiance fields in non-static scenes using heuristics-guided segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19436–19446, 2024. 2

[4] Hiba Dahmani, Moussab Bennehar, Nathan Piasco, Luis Roldao, and Dzmitry Tsishkou. Swag: Splatting in the wild images with appearance-conditioned gaussians, 2024. 2

[5] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In CVPR, 2023. 2

[6] Lily Goli, Cody Reading, Silvia Sellan, Alec Jacobson, and´ Andrea Tagliasacchi. Bayes’ Rays: Uncertainty quantification for neural radiance fields. CVPR, 2024. 2

[7] Fengzhi Guo, Chih-Chuan Hsu, Sihao Ding, and Cheng Zhang. Uncertainty matters in dynamic gaussian splatting for monocular 4d reconstruction. In The Fourteenth International Conference on Learning Representations, 2026. 2

[8] Neham Jain, Andrew Jong, Sebastian Scherer, and Ioannis Gkioulekas. SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction. In International Conference on 3D Vision (3DV). IEEE, IEEE, 2026. 5

[9] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fisherrf: Active view selection and uncertainty quantification for radiance fields using fisher information. arXiv, 2023. 2

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 3

[11] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and´ Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 2

[12] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sattler. WildGaussians: 3D gaussian splatting in the wild. NeurIPS, 2024. 2, 5, 6, 8

[13] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024. 2

[14] Chengqi Li, Zhihao Shi, Yangdi Lu, Wenbo He, and Xiangyu Xu. Robust neural rendering in the wild with asymmetric dual 3d gaussian splatting. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 5, $^ { 6 , }$ 8

[15] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition (CVPR), pages 8508–8520, 2024. 2

[16] Jingyu Lin, Jiaqi Gu, Lubin Fan, Bojian Wu, Yujing Lou, Renjie Chen, Ligang Liu, and Jieping Ye. Hybridgs: Decoupling transients and statics with 2d and 3d gaussian splatting. In CVPR, 2025. 2

[17] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 2

[18] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024. 2, 3

[19] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duck worth. NeRF in the Wild: Neural Radiance Fields for Un constrained Photo Collections. In CVPR, 2021. 2

[20] Natan Micheletti, Jim H Chandler, and Stuart N Lane. Structure from Motion (SfM) Photogrammetry. British Society for Geomorphology Geomorphological Techniques, 2(2):1– 12, 2015. 3

[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view syn thesis. In ECCV, 2020. 5

[22] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 2

[23] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2

[24] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. ICCV, 2021. 2, 3

[25] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M. Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. ACM Trans. Graph., 40(6), 2021.

[26] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 2, 3

[27] Weining Ren, Zihan Zhu, Boyang Sun, Jiaqi Chen, Marc Pollefeys, and Songyou Peng. Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 5, 8

[28] Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin, David J. Fleet, and Andrea Tagliasacchi. Robustnerf: Ignoring distractors with robust losses. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20626–20636, 2023. 2

[29] Sara Sabour, Lily Goli, George Kopanas, Mark Matthews, Dmitry Lagun, Leonidas Guibas, Alec Jacobson, David J. Fleet, and Andrea Tagliasacchi. SpotLessSplats: Ignoring distractors in 3d gaussian splatting. arXiv:2406.20055, 2024. 2, 5, 6, 8

[30] Johannes Lutz Schonberger and Jan-Michael Frahm.¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 6

[31] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, 2023. 8

[32] Paul Ungermann, Armin Ettenhofer, Matthias Nießner, and Barbara Roessle. Robust 3d gaussian splatting for novel view synthesis in presence of distractors. In DAGM German Conference on Pattern Recognition, pages 153–167. Springer, 2024. 2, 6, 8

[33] Unity Technologies. Unity game engine. https:// unity.com/, 2024. 5

[34] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF: Structured view-dependent appearance for neural radiance fields. CVPR, 2022. 2

[35] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In International Conference on Computer Vision (ICCV), 2025. 2

[36] Yihao Wang, Marcus Klasson, Matias Turkulainen, Shuzhe Wang, Juho Kannala, and Arno Solin. DeSplat: Decomposed Gaussian splatting for distractor-free rendering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 4, 5, 6, 8

[37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Trans. Image Process., 13(4): 600–612, 2004. 5

[38] Stephen Welstead. Fractal and Wavelet Image Compression Techniques. 2009. 5

[39] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310– 20320, 2024. 2, 5, 6, 7

[40] Liuyue Xie, Joel Julin, Koichiro Niinuma, and Laszlo A. Jeni. Gaussian splatting lucas-kanade, 2025. 2

[41] Congrong Xu, Justin Kerr, and Angjoo Kanazawa. Splatfacto-w: A nerfstudio implementation of gaussian splatting for unconstrained photo collections, 2024. 2, 8

[42] Jiacong Xu, Yiqun Mei, and Vishal M. Patel. Wild-gs: Realtime novel view synthesis from unconstrained photo collections, 2024. 2

[43] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023. 2, 3, 5, 6, 7

[44] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations (ICLR), 2024. 2

[45] Heng Yu, Joel Julin, Zoltan A Milacski, Koichiro Niinuma, and Laszlo A Jeni. Cogs: Controllable gaussian splatting. arXiv, 2023. 3

[46] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341–359. Springer, 2024. 2

[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proc. IEEE/CVF CVPR, pages 586–595, 2018. 5