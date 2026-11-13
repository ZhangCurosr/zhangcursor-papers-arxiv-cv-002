# Selective Generative Restoration for Video Deraining via Uncertainty-Aware Diffusion

Yi‘ang Chen, Han Li, Laibin Chang, Hui Liu,

Kui Jiang, Member, IEEE, Junjun Jiang, Senior Member, IEEE

Abstract—Video deraining is a challenging restoration problem due to complex degradations caused by rain streaks, occlusions, and atmospheric effects. Existing methods mainly follow a deterministic restoration paradigm, which assumes that the underlying clean scene can be uniquely recovered from rainy observations. However, rain degradation is spatially heterogeneous: reliable regions require faithful reconstruction, whereas severely occluded regions contain ambiguous contents that require stronger priordriven inference. Therefore, conventional deterministic restoration often produces over-smoothed results, while global generative restoration may introduce unnecessary hallucinations. In this paper, we introduce a new Selective Generative Restoration paradigm for video deraining, where generative priors are adaptively activated according to restoration uncertainty. Based on this principle, we propose Selective Generative Video Deraining (SGVD), an uncertainty-aware diffusion framework that enables adaptive fidelity preservation and generative recovery. Specifically, SGVD exploits stochastic diversity between dual one-step diffusion predictions to estimate region-level restoration uncertainty, which determines where observation fidelity should be preserved and where generative inference should be activated. Furthermore, an anchor-guided trajectory consistency mechanism is proposed to regularize stochastic diffusion evolution across frames, reducing temporal generation drift and improving video coherence. To achieve practical efficiency, SGVD adopts one-step diffusion inference while maintaining uncertainty-aware generative capability. Extensive experiments demonstrate that SGVD achieves superior restoration performance, perceptual quality, and temporal consistency compared with state-of-the-art video deraining methods, validating the effectiveness of selective generative restoration for complex rainy video restoration. The source code is available at https://github.com/BrewieCya/SGVD.

Index Terms—Video Deraining, Diffusion Models, Selective Restoration, Temporal Consistency.

## I. INTRODUCTION

O <sup>UTDOOR</sup> <sup>videos</sup> <sup>captured</sup> <sup>under</sup> <sup>rainy</sup> <sup>conditions</sup> <sup>often</sup>suffer from severe degradation caused by rain streaks, suffer from severe degradation caused by rain streaks, raindrop occlusions, rain accumulation, and atmospheric veiling effects. These degradations destroy scene structures and temporal consistency, significantly affecting downstream vision tasks such as detection [1], segmentation [2], tracking [3], and autonomous driving [4]. Therefore, robust video deraining is essential for reliable perception under adverse weather conditions.

![](images/ecd2b596021e86274d4c19fce7ab20dc9b224eaa587085eab1fad7b1d6d0a4e3.jpg)  
Fig. 1: Motivation of Selective Generative Restoration. Deterministic restoration tends to produce over-smoothed details in severely degraded regions, while global generative restoration may introduce content hallucination in reliable regions. In contrast, the proposed paradigm performs region-adaptive restoration by preserving fidelity in low-uncertainty regions and selectively activating generative recovery in high-uncertainty regions.

Existing video deraining methods have achieved remarkable progress through convolutional networks [5], [6], recurrent architectures [7], [8], Transformers [9], [10], and state space models [11], [12], mainly by improving spatial representation and temporal dependency modeling. However, most methods follow a deterministic restoration paradigm that maps each rainy observation to a single clean output. Such a formulation assumes that the underlying scene can be uniquely recovered from degraded observations, which becomes problematic under severe rain occlusions where background information is partially missing and multiple plausible restorations may exist.

Unlike uniformly degraded restoration problems, rain degradation is inherently spatially heterogeneous. Reliable regions preserve sufficient observations and mainly require fidelityoriented reconstruction, whereas heavily occluded regions contain uncertain contents and require stronger prior-driven inference. Therefore, video deraining is not merely a reconstruction problem, but a spatially varying inverse problem with different levels of restoration uncertainty. This observation reveals a fundamental limitation of existing restoration paradigms. Global deterministic restoration [13]–[15] tends to over-smooth ambiguous regions, while global generative restoration may introduce unnecessary hallucinations into reliable regions. Consequently, effective video deraining requires an adaptive strategy that determines where observation fidelity should dominate and where generative priors should be activated.

Recent diffusion models provide a promising direction by modeling conditional distributions rather than predicting a single deterministic solution. Their powerful generative priors make them suitable for recovering missing contents from ambiguous observations. However, directly applying diffusion models to video deraining remains challenging because rain degradation exhibits strong spatial uncertainty variation. As illustrated in Fig. 1, uniformly applying diffusion priors may improve uncertain regions but simultaneously alter trustworthy contents. Thus, the key problem is not whether diffusion should be introduced, but how to selectively activate generation according to restoration uncertainty.

To address this issue, we introduce a new restoration paradigm, termed Selective Generative Restoration. The core idea is that diffusion priors should be adaptively activated according to region-level uncertainty: low-uncertainty regions preserve observation fidelity, while high-uncertainty regions exploit generative inference for plausible recovery.

However, realizing Selective Generative Restoration requires addressing three challenges. First, an explicit uncertainty estimation mechanism is required to determine where generative restoration is necessary. Second, stochastic diffusion prediction may introduce inconsistent restoration trajectories across frames, causing temporal flickering. Third, conventional iterative diffusion sampling is computationally expensive and limits practical video restoration applications.

To address these challenges, we propose Selective Generative Video Deraining (SGVD), an uncertainty-aware diffusion framework supporting one-step restoration, built upon the principle of Selective Generative Restoration. SGVD uses stochastic prediction disagreement as a practical signal of restoration uncertainty. Specifically, given the same rainy observation, SGVD performs two one-step diffusion predictions with different noise perturbations and measures their latent discrepancy as an approximation of posterior uncertainty. The resulting uncertainty map explicitly identifies regions with different restoration difficulty: low-uncertainty regions contain reliable observations and are reconstructed with stronger fidelity constraints, whereas high-uncertainty regions contain missing or ambiguous contents and are enhanced through selective generative inference.

Furthermore, to address temporal instability caused by stochastic diffusion prediction, SGVD introduces an anchorguided trajectory consistency mechanism. The proposed strategy regularizes relative trajectory distances to an anchor obtained by averaging predicted latent representations within a five-frame subclip. This local temporal constraint is applied during training with 49-frame inputs. This design reduces frame-wise generation drift and encourages temporally coherent restoration across neighboring frames. Finally, to achieve efficient diffusion-based video deraining, SGVD adopts a onestep inference strategy that directly predicts clean latent representations from rainy inputs. Different from conventional iterative diffusion restoration, the proposed formulation preserves the uncertainty-aware generative capability while substantially reducing sampling complexity, making diffusion-based video deraining more practical for real-world applications.

Overall, SGVD establishes a new perspective that diffusion restoration should move beyond global generation toward uncertainty-aware selective generation. By adaptively balancing observation fidelity, generative inference, and temporal consistency, SGVD provides an effective solution for restoring rainy videos with heterogeneous degradation uncertainty.

Overall, the main contributions of this work are summarized as follows.

• We introduce a new Selective Generative Restoration paradigm for video deraining, revealing that rain degradation is a spatially heterogeneous inverse problem where different regions require different balances between observation fidelity and generative inference.

• We propose SGVD, a one-step uncertainty-aware diffusion framework that selectively activates generative restoration according to region-level posterior uncertainty. By exploiting stochastic diversity between dual diffusion predictions, SGVD constructs an explicit uncertainty representation and achieves adaptive restoration between faithful reconstruction and plausible content recovery.

• We develop an anchor-guided trajectory consistency mechanism to regularize stochastic diffusion evolution across video frames. Together with the proposed one-step inference, SGVD effectively suppresses temporal generation drift while maintaining high restoration efficiency, enabling practical diffusion-based video deraining.

## II. RELATED WORK

## A. Video Deraining

Video deraining has evolved from prior-based rain– background decomposition to learned spatio-temporal restoration. Early methods exploited rain sparsity and background low-rankness [16], [17], while subsequent CNN and recurrent models learned spatial rain representations and temporal dependencies directly from data [6], [8], [18]. Transformer-based methods further improved long-range interaction and motionaware aggregation [10], [19], and recent state-space models enhanced long-sequence modeling efficiency [11], [12]. Despite these advances, most methods formulate video deraining as deterministic point estimation. This formulation is effective when sufficient background evidence remains observable, but often produces over-smoothed results in severely occluded regions where multiple plausible restorations may exist. Recent works have therefore explored pretrained diffusion priors for video deraining, including zero-shot restoration [20] and nighttime rain removal [21]. However, existing generative approaches generally apply diffusion priors globally and do not explicitly distinguish reliable observations from ambiguous rain-occluded regions.

## B. Diffusion Models for Low-Level Vision

Diffusion models extend low-level vision from deterministic prediction [22], [23] to conditional distribution modeling, enabling plausible recovery when observations are incomplete. SR3 [24] introduced conditional diffusion for super-resolution, while StableSR [25] leveraged pretrained latent diffusion priors for realistic texture reconstruction. Subsequent studies have applied diffusion models to various image and video restoration tasks [26]–[29]. Nevertheless, strong generative priors introduce a fidelity–perception trade-off. FiDeSR [30] regulates this trade-off for one-step super-resolution, and WeatherDiffusion [31] supports weather-guided forward and inverse rendering. These methods mainly control generation globally, whereas rain degradation is spatially heterogeneous. In contrast, SGVD estimates region-level posterior uncertainty from stochastic prediction diversity and uses it to selectively preserve reliable observations or activate generative recovery.

![](images/a6c68af416544cccbab48811e11ae9a0e163471445dba9c0fa3988201f40f76b.jpg)  
Fig. 2: Overall architecture of the proposed SGVD for video deraining. The rainy video is first encoded into the latent space by a frozen VAE encoder. The Uncertainty-Guided Selective Generation (USG) module estimates a spatial uncertainty map from the discrepancy between dual-noise clean latent predictions, and adaptively combines the generative latent with the fidelity-preserving LQ latent. The resulting conditional latent is injected into the frozen DiT backbone through a lightweight trainable adapter. Meanwhile, Anchor-Guided Trajectory Consistency (ATC) regularizes each frame's distance to a shared five-frame anchor across trajectory states, promoting temporal coherence without evaluating this training loss at inference.

## C. Efficient Generative Models

The iterative sampling process of diffusion models limits their applicability to video restoration. DDIM [32] and DPM-Solver [33] accelerate inference through shortened or improved numerical sampling. Flow Matching [34] and Rectified Flow [35] instead learn continuous transport trajectories and have supported scalable video generators such as Wan [36]. Consistency Models [37] and Latent Consistency Models [38] further compress multi-step trajectories into fewstep or one-step prediction, with SinSR [39] and OSEDiff [40] demonstrating efficient one-step image restoration. However, sampling acceleration alone does not ensure temporally stable video restoration, since stochastic predictions of neighboring frames may follow inconsistent generative trajectories. SGVD combines one-step generative inference with Anchor-Guided Trajectory Consistency, which regularizes relative inter-frame evolution during training without additional inference cost.

III. SELECTIVE GENERATIVE RESTORATION FRAMEWORK A. Overview of SGVD

The proposed SGVD is built upon the principle of Selective Generative Restoration, which aims to adaptively balance observation fidelity and generative inference according to restoration uncertainty. Different from conventional restoration frameworks that apply a uniform reconstruction strategy to all regions, SGVD considers rainy video restoration as a spatially heterogeneous inverse problem. Specifically, regions with reliable observations should preserve the original visual evidence, while regions with severe information loss should exploit generative priors to recover plausible contents.

Given a rainy video sequence $\mathbf { X } ~ = ~ \{ x _ { i } \} _ { i = 1 } ^ { T }$ , SGVD targets the recovery of a clean and temporally coherent video $\hat { \textbf { Y } } = \{ \hat { y } _ { i } \} _ { i = 1 } ^ { T }$ . As illustrated in Fig. 2, SGVD consists of three key components: (1) a One-Step Generative Restoration Backbone that provides efficient latent-space generative reconstruction; (2) an Uncertainty-Guided Selective Generation (USG) module that estimates region-level restoration uncertainty and adaptively controls the activation of generative priors; and (3) an Anchor-Guided Trajectory Consistency (ATC) mechanism that regularizes stochastic generation trajectories to maintain temporal coherence.

Specifically, the rainy video is first encoded into the latent space using a frozen VAE encoder. The latent representation is then processed by the USG module, where stochastic generative predictions are exploited to estimate restoration uncertainty. Based on the estimated uncertainty, SGVD selectively integrates fidelity-preserving rainy latent features and generative clean latent predictions to construct an adaptive restoration condition.

The resulting condition and uncertainty map are injected into the frozen generative backbone through a trainable adapter. We use the fixed prompt Remove rain and restore clean background during training and standard testing. The frozen text encoder supplies the task instruction, while the rainy video and uncertainty-guided latent condition provide scene-specific structure and appearance. During training, ATC regularizes the relative evolution of neighboring frames along the generative trajectory, suppressing stochastic temporal drift without additional inference cost. Together with uncertaintyguided selective generation, SGVD adaptively balances fidelity preservation and generative recovery for complex rainy videos.

## B. One-Step Generative Restoration Backbone

Selective Generative Restoration requires a generative model that can provide strong prior knowledge while maintaining practical restoration efficiency. However, conventional diffusion models typically initialize generation from pure Gaussian noise and rely on iterative denoising, which may introduce excessive generation freedom and substantial computational overhead for video restoration. SGVD uses the pretrained Wan2.1-T2V-1.3B backbone [36] with an LQ-aware restoration condition. Flow matching and consistency learning provide the training formulation for predicting clean latent endpoints.

Given a GT latent representation $z _ { 0 } .$ , an LQ latent representation $z _ { 1 } ^ { l q }$ and a Gaussian noise latent $z _ { 1 } ^ { n o i s e } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , we first construct an LQ-aware noisy latent:

$$
z _ { 1 } = \alpha z _ { 1 } ^ { l q } + \sigma z _ { 1 } ^ { n o i s e } ,\tag{1}
$$

where $\alpha$ and σ control the contributions of the LQ observation and stochastic perturbation, respectively. Then flow matching defines a continuous transport trajectory between the data distribution and noise distribution:

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t z _ { 1 } , \qquad t \in [ 0 , 1 ] .\tag{2}
$$

The corresponding velocity field is $\begin{array} { r } { u _ { t } = \frac { d z _ { t } } { d t } = z _ { 1 } - z _ { 0 } } \end{array}$ . A conditional velocity network $v _ { \theta , \phi }$ approximates this transport direction, where θ denotes the frozen pretrained parameters and $\phi$ denotes the trainable adapter parameters.

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , z _ { 0 } , z _ { 1 } } \left[ \left\| v _ { \theta , \phi } ( z _ { t } , t , c ) - u _ { t } \right\| _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where c denotes the condition extracted from the rainy video and text embedding.

Although flow matching provides an efficient formulation for generative modeling, directly solving the continuous trajectory still requires numerical integration. Therefore, SGVD further adopts consistency learning following Consistency Models [37] to compress the generative trajectory into a direct endpoint prediction.

The consistency function $F _ { \theta , \phi }$ is trained to produce consistent clean latent predictions from different states along the same trajectory.

$$
F _ { \theta , \phi } ( z _ { t } , t , c ) = F _ { \theta , \phi } ( z _ { s } , s , c ) = z _ { 0 } ,\tag{4}
$$

where $\textit { s } < \textit { t } .$ . The corresponding consistency objective is formulated as

$$
\mathcal { L } _ { \mathrm { C M } } = \left. F _ { \theta , \phi } ( z _ { t } , t , c ) - \mathrm { s g } \left[ F _ { \theta , \bar { \phi } } ( z _ { s } , s , c ) \right] \right. _ { 2 } ^ { 2 } ,\tag{5}
$$

Here $\bar { \phi }$ denotes the exponential moving average of the adapter parameters, while the pretrained weights θ remain fixed. The operator $\operatorname { s g } ( \cdot )$ stops gradients through the consistency target.

## C. Diffusion Posterior Uncertainty Estimation

Selective Generative Restoration requires identifying which regions should preserve observed evidence and which regions require generative inference. Therefore, estimating regionlevel restoration uncertainty is the fundamental step for adaptive generation. Given a rainy video sequence X, the frozen VAE encoder first maps the input into the latent space: $z _ { 1 } ^ { l q } \ = \ \mathcal { E } ( \mathbf { X } )$ , where $\dot { z } _ { 1 } ^ { l q }$ represents the latent observation containing both reliable scene information and rain-induced degradation. Unlike deterministic restoration models that estimate a single solution, generative models characterize a conditional distribution: $p \big ( z _ { 0 } \big | z _ { 1 } ^ { l q } \big )$ , where $z _ { \mathrm { 0 } }$ denotes the latent representation of possible clean videos. The uncertainty of this posterior reflects the ambiguity of restoration: regions with sufficient observations exhibit concentrated predictions, whereas severely occluded regions produce diverse plausible solutions.

To approximate this posterior uncertainty, SGVD exploits the stochasticity of the generative trajectory. Specifically, two independent noise perturbations are injected into the same rainy latent representation:

$$
z _ { t } ^ { a } = \alpha _ { t } z _ { 1 } ^ { l q } + \sigma _ { t } \epsilon ^ { a } , ~ z _ { t } ^ { b } = \alpha _ { t } z _ { 1 } ^ { l q } + \sigma _ { t } \epsilon ^ { b } ,\tag{6}
$$

where $\epsilon ^ { a } , \epsilon ^ { b } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . The two perturbed states are processed by the frozen generative backbone without the adapter.

$$
\hat { v } ^ { a } = \mathcal { F } ^ { \mathrm { f r z } } ( z _ { t } ^ { a } , t , z _ { 1 } ^ { l q } ) , \hat { v } ^ { b } = \mathcal { F } ^ { \mathrm { f r z } } ( z _ { t } ^ { b } , t , z _ { 1 } ^ { l q } ) .\tag{7}
$$

Based on the predicted velocity fields, the corresponding clean latent estimations are recovered as:

$$
\hat { z } _ { 0 } ^ { a } = z _ { t } ^ { a } - t \hat { v } ^ { a } , \hat { z } _ { 0 } ^ { b } = z _ { t } ^ { b } - t \hat { v } ^ { b } .\tag{8}
$$

Although both predictions are generated from the same rainy observation, their divergence reflects the uncertainty of the conditional restoration distribution. Specifically, lowuncertainty regions with reliable visual evidence produce consistent clean predictions, while ambiguous regions with missing background information exhibit larger stochastic variations. Accordingly, SGVD estimates the diffusion posterior uncertainty map as:

$$
M = \mathcal { N } \left( \mathrm { M e a n } _ { C } ( | \hat { z } _ { 0 } ^ { a } - \hat { z } _ { 0 } ^ { b } | ) \right) .\tag{9}
$$

Here Mean<sub>C</sub> averages channels, and $\mathcal { N }$ performs $3 \times 3$ 3×3 spatial mean filtering without temporal smoothing, followed by clipwise min–max normalization.

The obtained uncertainty map $M ~ \in ~ [ 0 , 1 ]$ provides an explicit spatial measurement of restoration ambiguity: larger values indicate regions with higher posterior uncertainty and stronger demand for generative recovery, while smaller values correspond to reliable regions where observation fidelity should be maintained.

## D. Uncertainty-Guided Selective Generation

The estimated posterior uncertainty provides a spatial criterion for selectively activating generative restoration. Instead of applying generative priors uniformly, SGVD dynamically balances observation fidelity and generative inference according to the uncertainty map. First, the two stochastic clean latent predictions are aggregated to obtain a generative restoration proposal: $\begin{array} { r } { \hat { z } _ { 0 } = \frac { 1 } { 2 } ( \hat { z } _ { 0 } ^ { a } + \hat { z } _ { 0 } ^ { b } ) } \end{array}$ . To alleviate the unnecessary hallucinations in reliable regions, SGVD thus performs uncertaintyguided selective fusion between the generative prediction and the observed rainy latent representation:

$$
z _ { \mathrm { s e l } } = M \odot \hat { z } _ { 0 } + ( 1 - M ) \odot z _ { 1 } ^ { l q } ,\tag{10}
$$

where $\odot$ denotes element-wise multiplication. This formulation explicitly realizes the principle of Selective Generative Restoration. For high-uncertainty regions, M increases and activates the generative prior to recover ambiguous contents. For low-uncertainty regions, $( 1 - M )$ dominates and preserves trustworthy observations.

To adapt the selected latent representation to the frozen generative backbone, a lightweight trainable adapter is introduced: $z _ { \mathrm { c o n d } } = \mathcal { A } _ { \phi } ( z _ { \mathrm { s e l } } , M )$ , where $\mathcal { A } _ { \phi }$ denotes the trainable adapter. It uses a convolutional embedding module to map the concatenated ${ \cal z } _ { \mathrm { s e l } }$ and M into tokens aligned with the DiT grid, followed by block-specific linear projections. The resulting condition is injected into the generative backbone by scaling the projected features with pooled uncertainty and adding them to each DiT block’s output to produce the restored clean latent representation. The injection projections are initialized to zero to preserve the backbone’s initial behavior. Through uncertainty-guided selective activation, SGVD avoids the two opposite failure modes of previous restoration paradigms: over-smoothing caused by excessive fidelity constraints and hallucination caused by indiscriminate generation.

Therefore, USG transforms diffusion generation from a global restoration process into an uncertainty-aware selective restoration process, enabling adaptive recovery for spatially heterogeneous rainy videos.

## E. Anchor-Guided Trajectory Consistency

Although uncertainty-guided selective generation enables adaptive activation of generative priors, stochastic generation may still introduce temporal trajectory divergence among consecutive frames. Different from deterministic restoration, where each frame is mapped to a fixed output, generative restoration produces a distribution of possible restoration trajectories. Therefore, maintaining temporal consistency requires regularizing the evolution of generative trajectories rather than only constraining the final restored frames.

To address this issue, we introduce an Anchor-Guided Trajectory Consistency (ATC) mechanism. The key idea is to construct a shared subclip-level temporal anchor and enforce the relative trajectory evolution of neighboring frames to remain consistent during one-step generative restoration. The shared anchor provides a common reference that incorporates all frames within a subclip, offering broader temporal context than an immediate-neighbor constraint. Given a fiveframe subclip from a 49-frame training input $\textbf { z } = \{ z _ { i } \} _ { i = 1 } ^ { K }$ with $K = 5 ,$ , the generative backbone predicts clean latent representations at different trajectory states:

$$
\hat { z } _ { i } ^ { t } = \mathcal { F } _ { \theta , \phi } ( z _ { i } ^ { t } , t , z _ { \mathrm { c o n d } } ) ,\tag{11}
$$

where t denotes the flow trajectory state and $z _ { \mathrm { c o n d } }$ represents the uncertainty-guided selective condition.

To provide a stable temporal reference, we construct a subclip-level anchor representation:

$$
z _ { \mathrm { a n c } } ^ { t } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \Psi ( \hat { z } _ { i } ^ { t } ) ,\tag{12}
$$

where $\Psi ( \cdot )$ denotes a latent projection function. We use the parameter-free identity mapping $\Psi ( z ) = z$ for all frames and trajectory states.

The anchor captures the shared semantic structure of the subclip while preserving frame-specific variations. Based on this anchor, we measure the relative distance between each frame trajectory and the shared subclip-level reference:

$$
d _ { i } ^ { t } = \left. \Psi ( \hat { z } _ { i } ^ { t } ) - z _ { \mathrm { a n c } } ^ { t } \right. _ { 1 } .\tag{13}
$$

For stochastic generative trajectories at two different states $t _ { 1 }$ and $t _ { 2 }$ , temporally coherent restoration should maintain consistent relative distances: $d _ { i } ^ { t _ { 1 } }$ ≈ $d _ { i } ^ { t _ { 2 } }$ . Accordingly, the proposed ATC objective is formulated as:

$$
\mathcal { L } _ { \mathrm { A T C } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left| d _ { i } ^ { t _ { 1 } } - d _ { i } ^ { t _ { 2 } } \right| .\tag{14}
$$

ATC regularizes changes in each frame’s distance to the shared anchor across trajectory states, allowing frame-specific motion and appearance while discouraging stochastic temporal drift. By introducing anchor-guided trajectory regularization, SGVD extends selective generation from spatial adaptation to temporally consistent generative restoration, enabling stable video recovery under stochastic inference.

## F. Optimization Objective

During training, SGVD optimizes the selective generative restoration process while keeping the pretrained generative prior fixed. Only the adapter parameters $\phi$ are trainable. The pretrained DiT backbone, VAE encoder and decoder, and text encoder remain frozen. The overall training objective consists of three complementary terms: (1) a flow matching objective for learning the generative transport trajectory, (2) a consistency objective for compressing the continuous trajectory into one-step restoration, and (3) the proposed anchorguided trajectory consistency objective for temporally coherent generation.

The flow matching objective $\mathcal { L } _ { \mathrm { F M } }$ learns the transport direction from LQ-aware noisy latents toward the clean latent distribution, while the consistency objective ${ \mathcal { L } } _ { \mathrm { C M } }$ encourages different states along the same trajectory to predict a consistent clean endpoint for efficient one-step restoration. In addition, the proposed $\mathcal { L } _ { \mathrm { A T C } }$ regularizes frame-wise latent trajectories relative to the shared anchor of each five-frame subclip, suppressing stochastic temporal drift while preserving natural motion and frame-specific variations.

TABLE I: Quantitative comparisons with representative video restoration and task-specific video deraining methods on synthetic and real-world video deraining datasets. The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Dataset</td><td>Metric</td><td>|DiffIR2VR [41] Diff-Plugin [42] T3-Diff [43] WeatherDiff [31] DiffBIR [44]</td><td></td><td></td><td></td><td></td><td></td><td>ZSVD [20] DOVE [45] RainMamba [11] DeLiVR [10]</td><td></td><td></td><td>SGVD</td></tr><tr><td rowspan="7">NTURain-Syn</td><td>PSNR↑</td><td>23.7304</td><td>24.1138</td><td>25.3339</td><td>25.4756</td><td>25.9601</td><td>25.9677</td><td>26.2488</td><td>36.3430</td><td>36.7115</td><td>26.5174</td></tr><tr><td>SSIM↑</td><td>0.6850</td><td>0.7161</td><td>0.8633</td><td>0.8110</td><td>0.7802</td><td>0.8027</td><td>0.8534</td><td>0.9705</td><td>0.9684</td><td>0.8810</td></tr><tr><td>LPIPS↓</td><td>0.2755</td><td>0.1137</td><td>0.1350</td><td>0.1471</td><td>0.1221</td><td>0.1937</td><td>0.1333</td><td>0.0257</td><td>0.0281</td><td>0.1310</td></tr><tr><td>NIQE↓</td><td>4.2206</td><td>3.8413</td><td>3.4850</td><td>3.4752</td><td>3.6846</td><td>3.7548</td><td>4.8571</td><td>3.6070</td><td>3.5866</td><td>3.4224</td></tr><tr><td>CLIP-IQA↑</td><td>0.3601</td><td>0.3273</td><td>0.2877</td><td>0.2918</td><td>0.2583</td><td>0.2452</td><td>0.2537</td><td>0.2766</td><td>0.2666</td><td>0.3565</td></tr><tr><td>MUSIQ↑</td><td>57.3385</td><td>59.4466</td><td>57.1062</td><td>58.2231</td><td>53.3840</td><td>53.1633</td><td>52.7594</td><td>57.2592</td><td>56.4605</td><td>58.3749</td></tr><tr><td>WE↓</td><td>0.0286</td><td>0.0351</td><td>0.0210</td><td>0.0286</td><td>0.0249</td><td>0.0226</td><td>0.0241</td><td>0.0214</td><td>0.0221</td><td>0.0207</td></tr><tr><td rowspan="7">RainSynLight25</td><td>PSNR↑</td><td>22.7688</td><td>21.9237</td><td>24.8369</td><td>24.3363</td><td>23.7407</td><td>21.7876</td><td>25.5674</td><td>24.2891</td><td>24.3912</td><td>25.9165</td></tr><tr><td>SSIM↑</td><td>0.6314</td><td>0.6366</td><td>0.8439</td><td>0.8387</td><td>0.7068</td><td>0.6393</td><td>0.7734</td><td>0.8364</td><td>0.8382</td><td>0.8492</td></tr><tr><td>LPIPS↓</td><td>0.3657</td><td>0.1962</td><td>0.2547</td><td>0.2430</td><td>0.2908</td><td>0.3620</td><td>0.2664</td><td>0.3590</td><td>0.3722</td><td>0.3013</td></tr><tr><td>NIQE↓</td><td>5.1659</td><td>3.7525</td><td>4.5336</td><td>4.5294</td><td>4.5301</td><td>4.4077</td><td>5.1229</td><td>4.4971</td><td>4.2732</td><td>4.1671</td></tr><tr><td>CLIP-IQA↑</td><td>0.4660</td><td>0.4837</td><td>0.4106</td><td>0.3943</td><td>0.3814</td><td>0.3257</td><td>0.3513</td><td>0.4003</td><td>0.4249</td><td>0.4971</td></tr><tr><td>MUSIQ↑</td><td>58.0105</td><td>57.5581</td><td>58.1555</td><td>57.5304</td><td>52.6834</td><td>50.7805</td><td>54.4247</td><td>58.2881</td><td>58.1040</td><td>58.4274</td></tr><tr><td>WE↓</td><td>0.0503</td><td>0.0444</td><td>0.0489</td><td>0.0527</td><td>0.0470</td><td>0.0446</td><td>0.0255</td><td>0.0444</td><td>0.0367</td><td>0.0347</td></tr><tr><td rowspan="8"></td><td>PSNR↑</td><td>15.2853</td><td>16.7380</td><td>17.0794</td><td>16.0397</td><td>17.0434</td><td>15.2853</td><td>18.5361</td><td>17.1749</td><td>18.7076</td><td>18.5545</td></tr><tr><td>SSIM↑</td><td>0.4172</td><td>0.4647</td><td>0.5772</td><td>0.5267</td><td>0.5186</td><td>0.3862</td><td>0.5810</td><td>0.5505</td><td>0.5432</td><td>0.5812</td></tr><tr><td>LPIPS↓</td><td>0.5225</td><td>0.3437</td><td>0.4135</td><td>0.4583</td><td>0.4244</td><td>0.5434</td><td>0.4036</td><td>0.5212</td><td>0.4991</td><td>0.4922</td></tr><tr><td>RainSynComplex25 NIQE↓</td><td>8.3184</td><td>3.8672</td><td>5.6912</td><td>8.0030</td><td>5.2094</td><td>6.6324</td><td>5.6132</td><td>5.9325</td><td>5.0987</td><td>4.2315</td></tr><tr><td>CLIP-IQA↑</td><td>0.3905</td><td>0.4257</td><td>0.3168</td><td>0.3060</td><td>0.3211</td><td>0.2465</td><td>0.2934</td><td>0.3029</td><td>0.3069</td><td>0.3521</td></tr><tr><td>MUSIQ↑</td><td>56.6605</td><td>58.1845</td><td>56.6112</td><td>57.8192</td><td>50.8225</td><td>43.5738</td><td>48.5194</td><td>54.1892</td><td>53.8462</td><td>58.8422</td></tr><tr><td>WE↓</td><td>0.1180</td><td>0.0783</td><td>0.1239</td><td>0.1438</td><td>0.1117</td><td>0.1304</td><td>0.0446</td><td>0.1221</td><td>0.1001</td><td>0.1014</td></tr><tr><td>NIQE↓</td><td>4.2686</td><td>3.7203</td><td>3.3842</td><td>3.3784</td><td>3.9342</td><td>3.9149</td><td>4.9289</td><td>3.3609</td><td>3.3340</td><td>3.2927</td></tr><tr><td rowspan="4">NTURain-Real</td><td>CLIP-IQA↑</td><td>0.3385</td><td>0.4077</td><td>0.3786</td><td>0.3947</td><td>0.3401</td><td>0.2755</td><td>0.2360</td><td>0.3618</td><td>0.3440</td><td>0.3930</td></tr><tr><td>MUSIQ↑</td><td>60.2405</td><td>62.0713</td><td>62.5504</td><td>64.3716</td><td>57.2874</td><td>54.7643</td><td>53.7124</td><td>62.2050</td><td>62.3364</td><td></td></tr><tr><td>WE↓</td><td>0.0276</td><td>0.0395</td><td>0.0335</td><td>0.0327</td><td>0.0326</td><td>0.0258</td><td>0.0205</td><td>0.0325</td><td>0.0315</td><td>62.6503</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0204</td></tr></table>

Combining the above objectives, the final training objective is formulated as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { F M } } \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { C M } } \mathcal { L } _ { \mathrm { C M } } + \lambda _ { \mathrm { A T C } } \mathcal { L } _ { \mathrm { A T C } } ,\tag{15}
$$

where λ<sub>FM</sub>, λ<sub>CM</sub>, and λ<sub>ATC</sub> are the balancing coefficients for flow matching, one-step consistency, and anchor-guided trajectory consistency, respectively.

## IV. EXPERIMENTAL RESULTS

## A. Experimental Settings

Datasets. We evaluate SGVD on NTURain-Syn [6], RainSyn-Light25, RainSynComplex25 [18], and real-world NTURain videos without clean references. The NTURain-Syn training set contains 25 rainy sequences synthesized from eight clean videos with varying rain severity. All methods are retrained on the same NTURain-Syn split and evaluated on common test splits, with direct cross-dataset evaluation on RainSynLight25 and RainSynComplex25 without fine-tuning.

Evaluation Metrics. We report PSNR and SSIM for reconstruction fidelity; LPIPS, NIQE, CLIP-IQA, and MUSIQ for perceptual quality; and optical-flow-based warping error (WE) for temporal consistency, using RAFT [46] for flow estimation. All methods share the metric computation protocol. For real-world videos without clean references, we report NIQE, MUSIQ, CLIP-IQA, and WE together with qualitative results.

Experimental Setup. SGVD is initialized from the official Wan-AI/Wan2.1-T2V-1.3B release [36], with the pretrained DiT, VAE, and UMT5-XXL text encoder frozen. The complete model contains 7,298.47 M parameters, of which only the 71.67 M adapter parameters (0.98%) are trainable. The shared training protocol uses 49-frame inputs, random 256 × 256 crops, AdamW with a learning rate of $1 0 ^ { - 3 }$ and weight decay of $1 0 ^ { - 2 } .$ , four gradient accumulation steps, and 100,000 iterations. ATC constructs each anchor by averaging predicted latent representations within a five-frame subclip. The text prompt Remove rain and restore clean background is fixed throughout training and standard testing.

## B. Comparison with State-of-the-Art Methods

Quantitative Comparisons. We compare SGVD with representative state-of-the-art restoration methods, including both diffusion-based and non-diffusion approaches. The diffusionbased baselines include DiffIR2VR [41], Diff-Plugin [42], DiffBIR [44], ZSVD [20], DOVE [45], WeatherDiffusion (WeatherDiff) [31], and T3-Diff [43]. The non-diffusion baselines include the Transformer-based DeLiVR [10] and the state-space-model-based RainMamba [11].

As shown in Table I, SGVD achieves consistent improvements over the compared diffusion-based methods in reconstruction fidelity across different synthetic rain scenarios. In particular, SGVD obtains superior PSNR and SSIM performance with competitive NIQE scores on challenging rain conditions with severe occlusions, demonstrating its capability to recover missing contents while preserving reliable observations. SGVD also achieves better NIQE, CLIP-IQA, MUSIQ, and WE than DeLiVR and RainMamba on both NTURain-Syn and NTURain-Real. These advantages extend to cross-dataset evaluation, where SGVD leads the two non-diffusion baselines on all seven metrics on RainSynLight25 and five metrics on RainSynComplex25. Different from existing diffusion-based methods that apply generative priors globally, SGVD selectively activates generation according to restoration uncertainty. This uncertainty-aware strategy balances fidelity preservation and perceptual recovery. Reliable regions maintain observation consistency, while ambiguous regions benefit from generative reconstruction. Retaining reliable observations can limit texture regeneration, reflecting the trade-off between pixellevel fidelity and perceptual similarity. SGVD balances these objectives to achieve strong reconstruction fidelity and competitive perceptual quality. Moreover, SGVD achieves highly competitive temporal consistency across the evaluated datasets and obtains the lowest warping error on NTURain-Syn and NTURain-Real. This improvement verifies that the proposed anchor-guided trajectory consistency mechanism effectively suppresses stochastic generation drift and maintains temporally coherent restoration.

![](images/be79f33bf0f66539b7b3a5a3c278f238f83839f130f82d1cfd7f10d3c513a352.jpg)  
Fig. 4: Visual comparisons on real-world rainy videos from the NTURain-Real dataset [6].

Qualitative Comparisons. Figure 3 presents visual comparisons under different rain degradation levels. Existing restoration methods generally remove rain artifacts but tend to generate over-smoothed results in severely corrupted regions due to insufficient prior information. Meanwhile, global generative restoration methods can recover plausible structures but may introduce unrealistic details into reliable areas. In contrast, SGVD performs uncertainty-aware selective generation: lowuncertainty regions preserve faithful observations, whereas high-uncertainty regions activate generative recovery. As a result, SGVD produces cleaner backgrounds, sharper structures, and more realistic details under complex rain degradations.

Real-World Evaluation. To further evaluate practical robustness, we conduct experiments on real-world rainy videos without clean references. As shown in Fig. 4, SGVD generalizes well to diverse real-world rain conditions, including complex illumination changes and non-uniform rain patterns. Compared with existing methods, SGVD achieves better visual quality while avoiding excessive hallucination and preserving scene structures. These results demonstrate that uncertainty-aware selective generation provides a robust restoration strategy beyond synthetic benchmarks.

## C. Downstream Task Evaluation

To further evaluate the practical utility of SGVD, we conduct downstream object detection and tracking experiments on the RVDT [12] benchmark. The restored videos are directly fed into the downstream models without additional finetuning. For object detection, we employ YOLO-v3 [47] and

TABLE II: Object detection and tracking results on the RVDT benchmark. mAP and mAP denote the detection results obtained by YOLO-v3 and MEGA, respectively. The tracking results are obtained using GTR.
<table><tr><td rowspan="2">Method</td><td colspan="2">Object Detection</td><td colspan="2">Object Tracking</td></tr><tr><td> $\mathrm { m A P _ { 1 } } \uparrow$ </td><td>mAP2↑</td><td>IDF1↑ MOTA↑</td><td>MOTP↑</td></tr><tr><td>Rainy Input</td><td>39.73</td><td>44.69</td><td>64.5 28.0</td><td>19.9</td></tr><tr><td>ZSVD</td><td>38.86</td><td>43.32</td><td>64.8 26.4</td><td>20.3</td></tr><tr><td>DOVE</td><td>41.35</td><td>45.87</td><td>65.7 30.1</td><td>20.6</td></tr><tr><td>WeatherDiff</td><td>37.82</td><td>42.12</td><td>63.5 29.5</td><td>18.8</td></tr><tr><td>DiffBIR</td><td>38.06</td><td>44.56</td><td>65.3 27.9</td><td>20.4</td></tr><tr><td>DiffIR2VR</td><td>41.08</td><td>43.35</td><td>65.9 28.1</td><td>21.0</td></tr><tr><td>SGVD</td><td>42.02</td><td>44.86</td><td>66.4 30.2</td><td>20.8</td></tr></table>

MEGA [48], while the tracking performance is evaluated using GTR [49]. As shown in Table II, SGVD achieves improved detection and tracking performance over rainy inputs. Specifically, SGVD obtains the highest $\mathrm { \ m A P _ { 1 } }$ of 42.02%. Compared with the rainy input, SGVD improves $\mathrm { \ m A P _ { 1 } }$ and $\mathrm { \ m A P _ { 2 } }$ by 2.29 and 0.17 percentage points, respectively, demonstrating that the proposed selective generation strategy effectively restores discriminative object details. Furthermore, SGVD achieves the highest IDF1 of 66.4 and MOTA of 30.2, outperforming existing diffusion-based methods. This improvement verifies that the proposed ATC effectively reduces temporal generation drift and enhances cross-frame object association, leading to more reliable downstream video analysis.

TABLE III: Spearman correlation between the estimated uncertainty map and restoration error on synthetic video deraining datasets. Higher values indicate that regions with larger estimated uncertainty tend to exhibit larger restoration errors.
<table><tr><td>Metric</td><td>NTURain-Syn</td><td>RainSynLight25</td><td>RainSynComplex25</td></tr><tr><td>Spearman Correlation (ρs)↑ |</td><td>0.912</td><td>0.910</td><td>0.933</td></tr></table>

## D. Analysis of Diffusion Posterior Uncertainty

We further analyze whether the estimated uncertainty map accurately reflects restoration ambiguity. Figure 5 presents uncertainty maps generated by the proposed diffusion posterior uncertainty estimation strategy. As observed, high uncertainty responses are mainly concentrated in regions with severe rain streaks, dense occlusions, and missing background structures. In contrast, reliable background regions exhibit relatively low uncertainty. This observation indicates that stochastic prediction diversity effectively captures spatial variations in restoration difficulty.

![](images/0a8829609e63206a5157d1ed83793df58548de4e78d8689b1aab8a481b899827.jpg)  
Fig. 5: Visualization of diffusion posterior uncertainty in SGVD. The uncertainty map is estimated from the discrepancy between two stochastic one-step diffusion predictions. Severely degraded regions exhibit stronger uncertainty responses, while reliable observations remain relatively low-uncertainty, indicating that the estimated uncertainty effectively reflects spatial variations in restoration ambiguity.

To quantitatively verify whether the estimated uncertainty corresponds to actual restoration ambiguity, we analyze the spatial correlation between uncertainty maps and reconstruction errors. Specifically, given the uncertainty map M and the pixel-level restoration error $E _ { i } ( p ) \ = \ \| \hat { y } _ { i } ( p ) - y _ { i } ( p ) \| _ { 1 }$ we employ Spearman’s rank correlation coefficient $\rho _ { s }$ to measure their monotonic spatial correspondence. Here yˆ<sub>i</sub>(p) and $y _ { i } ( p )$ denote the restored and ground-truth pixel values of frame i, respectively. A larger $\rho _ { s }$ indicates that regions with higher estimated uncertainty tend to exhibit larger restoration errors. As shown in Table III, consistently high correlations are observed across different synthetic datasets, showing that stochastic prediction disagreement tracks spatial variations in reconstruction error.

TABLE IV: Ablation study of the core components of SGVD on the NTURain-Syn test set. USG mainly improves fidelity and perceptual recovery, whereas ATC primarily improves temporal stability.
<table><tr><td>Model</td><td>USG</td><td>ATC</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>MUSIQ↑</td><td>WE↓</td></tr><tr><td>D1</td><td></td><td></td><td>23.0013 0.7819</td><td>0.1899</td><td>3.5131</td><td>0.3221</td><td>57.8211</td><td>0.0287</td></tr><tr><td>D2</td><td> $\checkmark$ </td><td></td><td>26.2416 0.8732</td><td>0.1362</td><td>3.5114</td><td>0.3487</td><td>57.9213</td><td>0.0309</td></tr><tr><td>D3</td><td></td><td>√</td><td>25.9347 0.8638</td><td>0.1449</td><td>3.7052</td><td>0.3316</td><td>57.2845</td><td>0.0225</td></tr><tr><td>SGVD</td><td> $\checkmark$ </td><td>√</td><td>26.5174 0.8810</td><td>0.1310</td><td>3.4224</td><td>0.3565</td><td>58.3749</td><td>0.0207</td></tr></table>

## E. Temporal Trajectory Consistency Analysis

We evaluate whether the proposed Anchor-Guided Trajectory Consistency (ATC) effectively stabilizes the temporal evolution of one-step generative restoration.

Quantitative Temporal Stability. We first isolate the contribution of ATC using the component ablation results in

![](images/76552945cd95db3028182894474ba67aa1700917f947becdbd0b3abc40e9c56c.jpg)  
Fig. 6: Temporal slice comparison of rainy input, SGVD without ATC, full SGVD, and ground truth (top to bottom). ATC improves structural consistency across frames and reduces temporal artifacts, yielding a lower warping error (WE).

Table IV. The one-step restoration baseline without USG or ATC obtains a temporal warping error of 0.0287. Introducing USG alone substantially improves spatial restoration metrics, but only slightly changes the warping error from 0.0287 to 0.0309. This result indicates that uncertainty-guided spatial selection alone cannot fully resolve stochastic temporal drift.

In contrast, introducing ATC alone reduces the warping error from 0.0287 to 0.0225, while also improving PSNR and SSIM. When USG and ATC are jointly incorporated, the complete SGVD further reduces the warping error to 0.0207. These results show that ATC improves temporal stability alongside reconstruction fidelity. They also reveal the complementary relation: USG controls the spatial activation of generative priors, whereas ATC regularizes their temporal evolution.

Temporal-Slice Visualization. To qualitatively examine crossframe stability, we construct temporal slices by extracting the same spatial line from consecutive frames and stacking them along the temporal dimension. As shown in Fig. 6, the result without ATC exhibits irregular fluctuations and discontinuous structural traces, particularly around object boundaries and texture-rich regions. These artifacts indicate that independently restored frames follow inconsistent stochastic trajectories.

With ATC, the temporal slices become substantially smoother and more continuous, while the underlying motioninduced variations remain visible. This observation confirms that ATC does not force neighboring frames to become identical. Instead, it regularizes relative distances to the shared anchor within each five-frame subclip, allowing motiondependent variations while discouraging incoherent temporal evolution. Since ATC is used only during training, this temporal improvement introduces no additional inference overhead.

Temporal Regularization Comparison. Table V compares ATC with a neighboring-frame relative-trajectory loss while holding the backbone, USG, training budget, and loss-weight selection protocol fixed. The five-frame shared anchor reduces WE from 0.0323 to 0.0207 (35.9%) and improves PSNR from 26.3309 to 26.5174 dB and SSIM from 0.8763 to 0.8810. NIQE and MUSIQ also improve to 3.4224 and 58.3749. The neighboring-frame objective obtains lower LPIPS and higher CLIP-IQA, while ATC offers stronger temporal consistency and fidelity with competitive perceptual quality. Relating each frame to the full five-frame subclip provides a common temporal reference beyond an immediate-neighbor pair.

TABLE V: Temporal regularization comparison on NTURain-Syn. Best results are bold.
<table><tr><td>Metric</td><td>Neighboring-frame</td><td>Shared-anchor ATC</td></tr><tr><td>PSNR↑</td><td>26.3309</td><td>26.5174</td></tr><tr><td>SSIM↑</td><td>0.8763</td><td>0.8810</td></tr><tr><td>LPIPS↓</td><td>0.1257</td><td>0.1310</td></tr><tr><td>NIQE↓</td><td>3.5280</td><td>3.4224</td></tr><tr><td>CLIP-IQA↑</td><td>0.3671</td><td>0.3565</td></tr><tr><td>MUSIQ↑</td><td>57.9113</td><td>58.3749</td></tr><tr><td>WE↓</td><td>0.0323</td><td>0.0207</td></tr></table>

TABLE VI: End-to-end SGVD efficiency with and without uncertainty probing on 49- frame inputs at 640 × 480 resolution.
<table><tr><td>Method</td><td>Steps</td><td>Time (ms/frame)</td><td>FPS</td><td>FLOPs (G/frame)</td><td>Total (M)</td><td>Trainable (M)</td></tr><tr><td>SGVD</td><td>1</td><td>296</td><td>3.37</td><td>21548</td><td>7298.47</td><td>71.67</td></tr><tr><td>SGVD</td><td>30</td><td>2078</td><td>0.48</td><td>226611</td><td>7298.47</td><td>71.67</td></tr><tr><td>SGVD w/o probes</td><td>1</td><td>133</td><td>7.52</td><td>11372</td><td>7298.47</td><td>71.67</td></tr></table>

## F. Computational Efficiency

We measure end-to-end inference on a single 48GB NVIDIA RTX 4090 GPU using 49-frame inputs at 640 × 480 resolution (width × height), batch size 1, and BF16 precision. After one warm-up run, latency is averaged over three timed runs of the same clip with CUDA synchronization. Model loading and disk I/O are excluded. Latency and FLOPs are normalized by the 49 output frames, and FPS is computed from the unrounded clip latency. FLOPs include matrix multiplications, convolutions, and attention, counting two operations per multiply-accumulate. The 30-step cost is calculated from the profiled fixed pipeline cost plus 30 repetitions of the per-step computation.

As shown in Table VI, one-step SGVD requires 296 ms/frame and 21,548 G/frame, compared with 2,078 ms/frame and 226,611 G/frame for 30-step inference. This corresponds to approximately 7.0× faster inference, an 86% reduction in latency, and a 90.5% reduction in FLOPs. By replacing iterative denoising with direct latent restoration, SGVD substantially reduces repeated backbone evaluations and improves video-processing throughput. The end-to-end speedup is smaller than the sampling-step ratio because operations such as VAE encoding and decoding incur costs independent of the number of restoration steps. The two uncertainty probes increase latency from 133 to 296 ms/frame and FLOPs from 11,372 to 21,548 G/frame, reducing throughput from approximately 7.52 to 3.37 FPS. These additional backbone evaluations provide spatial guidance for selective restoration, while reusing the frozen model without introducing trainable parameters. In addition, only 71.67 M of SGVD’s 7,298.47 M parameters are trainable (0.98%). Restricting optimization to the adapter reduces the number of parameters requiring stored gradients and optimizer states by approximately 99.02%, enabling task-specific adaptation of the large pretrained model with a small parameter-update burden.

## G. Ablation Studies

After validating the Selective Generative Restoration paradigm and the estimated diffusion posterior uncertainty, we further conduct controlled experiments to isolate the contributions of individual components and analyze the sensitivity of the proposed framework.

Effect of Core Components. Table IV evaluates the individual and joint effects of Uncertainty-Guided Selective Generation (USG) and Anchor-Guided Trajectory Consistency (ATC). The baseline D1 denotes the one-step generative restoration backbone without either component. Adding USG to the baseline improves PSNR from 23.0013 to 26.2416 and reduces LPIPS from 0.1899 to 0.1362. It also improves the no-reference perceptual metrics, confirming that uncertaintyguided activation achieves a better balance between fidelity preservation and generative recovery. However, its temporal warping error remains close to that of the baseline, indicating that spatial uncertainty modeling alone does not explicitly constrain stochastic temporal evolution. Adding ATC alone improves PSNR and SSIM and reduces the warping error from 0.0287 to 0.0225. These gains support the contribution of ATC to both spatial reconstruction and temporal stability. The complete SGVD achieves the best overall performance, reaching a PSNR of 26.5174, an LPIPS of 0.1310, and a warping error of 0.0207. Compared with the baseline, these correspond to a PSNR improvement of 3.5161 dB, an LPIPS reduction of 0.0589, and a temporal warping error reduction of 27.9%. The consistent improvements across fidelity, perception, and temporal stability demonstrate that USG and ATC are complementary. USG determines where generation should be activated, while ATC determines how the activated generation should evolve consistently across frames.

TABLE VII: Effect of LQ latent utilization strategies on the NTURain [6] synthetic dataset.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>MUSIQ↑ WE↓</td></tr><tr><td>Start Only</td><td>19.8903</td><td>0.5467</td><td>0.2796 3.3830</td><td>0.3528</td><td>57.8211</td><td>0.0256</td></tr><tr><td>Condition Only</td><td>17.0639</td><td>0.6489</td><td>0.3435</td><td>3.7909</td><td>0.1627</td><td>42.4441 0.0221</td></tr><tr><td>Both (SGVD)</td><td>26.5174</td><td>0.8810</td><td>0.1310 3.4224</td><td>0.3565</td><td>58.3749</td><td>0.0207</td></tr></table>

TABLE VIII: Effect of the number of stochastic posterior probes used for uncertainty estimation. The two-probe configuration achieves the best overall accuracy–quality– efficiency trade-off.
<table><tr><td>Number of Probes</td><td>PSNR↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>MUSIQ↑</td><td>WE↓</td></tr><tr><td>w/o Uncertainty</td><td>25.6821</td><td>0.1478</td><td>3.8126</td><td>56.9821</td><td>0.0318</td></tr><tr><td>Single Probe</td><td>26.0712</td><td>0.1404</td><td>3.6468</td><td>57.5316</td><td>0.0284</td></tr><tr><td>Two Probes</td><td>26.5174</td><td>0.1310</td><td>3.4224</td><td>58.3749</td><td>0.0207</td></tr><tr><td>Four Probes</td><td>26.4938</td><td>0.1332</td><td>3.4517</td><td>58.1263</td><td>0.0205</td></tr></table>

Effect of LQ Latent Utilization. We investigate the influence of different strategies for incorporating low-quality (LQ) latent information into the generative restoration process. Specifically, we compare three variants: using the LQ latent only as the diffusion trajectory starting point, using it only as conditional guidance, and our full design that exploits the LQ latent for both trajectory initialization and conditioning. As shown in Table VII, using the LQ latent only as the starting point retains more information from the degraded observation, but provides insufficient guidance for recovering heavily corrupted contents, leading to limited fidelity and perceptual quality. Using the LQ latent only as conditional guidance cannot adequately constrain the generative trajectory and results in substantial structural deviation from the target content. By jointly using the LQ latent as both the trajectory starting

(a) Complex textures

state and the conditional signal, SGVD achieves substantially better reconstruction fidelity and perceptual quality, while also improving temporal consistency. In particular, the full design achieves the best PSNR, SSIM, LPIPS, CLIP-IQA, MUSIQ, and WE, while maintaining competitive NIQE, demonstrating that the two roles of the LQ latent are complementary for balancing observation preservation and generative recovery.

Number of Stochastic Posterior Probes. We next investigate the number of stochastic predictions used to estimate diffusion posterior uncertainty. As shown in Table VIII, removing uncertainty guidance degrades restoration performance, while a single prediction cannot explicitly characterize cross-sample posterior dispersion. Using two independent predictions introduces an effective disagreement signal and achieves the best overall performance in PSNR, LPIPS, NIQE, and MUSIQ, indicating that pairwise stochastic diversity is sufficient for region-level uncertainty estimation. Increasing the number of predictions from two to four only slightly improves WE from 0.0207 to 0.0205, while slightly degrading the other metrics and doubling the uncertainty-probing cost. Therefore, we adopt two stochastic predictions as the default configuration for the best accuracy–quality–efficiency trade-off.

TABLE IX: Sensitivity analysis of the weighting coefficients for flow matching, one-step consistency, and anchor-guided trajectory consistency.
<table><tr><td>Variant</td><td>λFM</td><td>λCM</td><td>λATC</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>WE↓</td></tr><tr><td>Group1</td><td>1.0</td><td>1.0</td><td>0.05</td><td>26.2847</td><td>0.8746</td><td>0.1328</td><td>57.9482</td><td>0.0231</td></tr><tr><td>Group2</td><td>1.0</td><td>1.0</td><td>0.10</td><td>26.5174</td><td>0.8810</td><td>0.1310</td><td>58.3749</td><td>0.0207</td></tr><tr><td>Group3</td><td>1.0</td><td>1.5</td><td>0.15</td><td>26.4263</td><td>0.8789</td><td>0.1358</td><td>58.1637</td><td>0.0220</td></tr><tr><td>Group4</td><td>1.0</td><td>0.5</td><td>0.50</td><td>26.3518</td><td>0.8767</td><td>0.1341</td><td>58.0126</td><td>0.0214</td></tr></table>

Sensitivity to Loss Weights. We further analyze the sensitivity of SGVD to the weighting coefficients of the flow matching, one-step consistency, and anchor-guided trajectory consistency objectives. The corresponding results are presented in Table IX. When the ATC loss weights are insufficient, the model cannot adequately balance generative trajectory learning, onestep endpoint consistency, and temporal trajectory regularization. The default configuration Group2 achieves the best overall performance across PSNR, SSIM, LPIPS, MUSIQ, and temporal warping error, indicating a suitable balance among the three complementary objectives. Changing the weight combination in Group3 and Group4 does not yield further improvements, indicating sensitivity to the balance among the training objectives.

Text Conditioning. We compare the fixed training prompt Remove rain and restore clean background, the shorter prompt Remove rain, and an empty prompt at inference. The checkpoint, input clips, random seeds, sampling settings, guidance scale, and negative prompt are identical across the three settings. In Table X, the original prompt leads on all seven metrics. Relative to the empty prompt, it improves PSNR by 0.1494 dB and SSIM by 0.0067, reduces LPIPS from 0.1368 to 0.1310, and lowers WE by approximately 1.9%. The short and empty prompts give similar reconstruction quality, suggesting that visual conditioning supplies strong scene guidance, while the full task instruction provides a further improvement.

Input-Length Generalization. We evaluate the same model trained on 49-frame inputs using 25-, 49-, and 81-frame inputs, without retraining or fine-tuning. Source sequences and scored frames are matched, and the resolution, prompt, and sampling settings are held fixed. In Table XI, the 81-frame configuration improves PSNR by 0.1365 dB and SSIM by 0.0087 over 49 frames and leads on five of the seven metrics. Its LPIPS of 0.1316 remains close to 0.1310 at 49 frames. The 25- frame setting retains comparable quality, with PSNR lower by 0.2217 dB. WE is 0.0214, 0.0207, and 0.0209 at 25, 49, and 81 frames, respectively, indicating stable temporal behavior on longer inputs. These results support length generalization over the evaluated range. The five-frame extent of the training regularizer does not fix the inference input length.

TABLE X: Text-conditioning comparison on NTURain-Syn. Best results are bold.
<table><tr><td>Prompt</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>MUSIQ↑</td><td>WE↓</td></tr><tr><td>Original</td><td>26.5174</td><td>0.8810</td><td>0.1310</td><td>3.4224</td><td>0.3565</td><td>58.3749</td><td>0.0207</td></tr><tr><td>Short</td><td>26.3683</td><td>0.8740</td><td>0.1359</td><td>3.4231</td><td>0.3433</td><td>57.9668</td><td>0.0211</td></tr><tr><td>Empty</td><td>26.3680</td><td>0.8743</td><td>0.1368</td><td>3.4251</td><td>0.3467</td><td>58.2029</td><td>0.0211</td></tr></table>

TABLE XI: Input-length evaluation using a model trained on 49-frame clips. Best results are bold.
<table><tr><td>Metric</td><td>25 frames</td><td>49 frames</td><td>81 frames</td></tr><tr><td>PSNR↑</td><td>26.2957</td><td>26.5174</td><td>26.6539</td></tr><tr><td>SSIM↑</td><td>0.8784</td><td>0.8810</td><td>0.8897</td></tr><tr><td>LPIPS↓</td><td>0.1339</td><td>0.1310</td><td>0.1316</td></tr><tr><td>NIQE↓</td><td>3.4480</td><td>3.4224</td><td>3.3489</td></tr><tr><td>CLIP-IQA↑</td><td>0.3497</td><td>0.3565</td><td>0.3609</td></tr><tr><td>MUSIQ↑</td><td>58.1333</td><td>58.3749</td><td>58.6926</td></tr><tr><td>WE↓</td><td>0.0214</td><td>0.0207</td><td>0.0209</td></tr></table>

![](images/c7888e0c03d574f47aa87ac0e5f67d11bbac9b510f39fc2daab80affa754d079.jpg)  
Fig. 7: Restoration examples in challenging scenes. From left to right: rain streak map, uncertainty overlay, and reconstruction error.

## H. Limitations

SGVD largely removes rain streaks in the challenging examples shown in Fig. 7, while remaining reconstruction errors are concentrated around complex textures, motion boundaries, and regions with varying illumination. Generative priors can recover plausible local details that differ from the clean reference. In addition, the two stochastic predictions used for uncertainty estimation add computational cost. Improving detail fidelity and reducing this computational cost are directions for future work.

## V. CONCLUSION

In this paper, we revisit video deraining from the perspective of uncertainty-aware generative restoration and reveal that rain degradation is a spatially heterogeneous inverse problem. Based on this insight, we propose Selective Generative Video Deraining (SGVD), a new restoration framework that adaptively balances observation fidelity and generative inference. SGVD exploits stochastic diversity of one-step generative predictions to estimate diffusion posterior uncertainty, enabling selective activation of generative priors in ambiguous regions while preserving reliable observations. Furthermore, an Anchor-Guided Trajectory Consistency mechanism is introduced to regularize stochastic restoration trajectories and improve temporal coherence without additional inference cost. Extensive experiments demonstrate that SGVD achieves strong reconstruction fidelity, competitive perceptual quality, and stable temporal behavior compared with existing video deraining methods. This work provides a new perspective for diffusionbased restoration by shifting from global generation toward uncertainty-aware selective generation, which can be further extended to other restoration tasks with spatially varying degradation.

## REFERENCES

[1] J. Li, R. Xu, X. Liu, J. Ma, B. Li, Q. Zou, J. Ma, and H. Yu, “Domain adaptation based object detection for autonomous driving in foggy and rainy weather,” IEEE Transactions on Intelligent Vehicles, vol. 10, no. 2, pp. 900–911, 2025.

[2] Y. Pan, R. Sun, W. Li, and T. Zhang, “Exploring weather-aware aggregation and adaptation for semantic segmentation under adverse conditions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 13 952–13 962.

[3] J. Liu, “RainyTrack: Enhancing object tracking in adverse weather conditions with siamese networks,” in Proceedings of the 7th International Conference on Image and Graphics Processing. ACM, 2024, pp. 308– 315.

[4] X. Zhong, S. Tu, X. Ma, K. Jiang, W. Huang, and Z. Wang, “Rainy WCity: A real rainfall dataset with diverse conditions for semantic driving scene understanding,” in International Joint Conference on Artificial Intelligence, 2022, pp. 1743–1749.

[5] M. Li, Q. Xie, Q. Zhao, W. Wei, S. Gu, J. Tao, and D. Meng, “Video rain streak removal by multiscale convolutional sparse coding,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6644–6653.

[6] J. Chen, C.-H. Tan, J. Hou, L.-P. Chau, and H. Li, “Robust video content alignment and compensation for rain removal in a CNN framework,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6286–6295.

[7] W. Yang, J. Liu, and J. Feng, “Frame-consistent recurrent video deraining with dual-level flow,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 1661–1670.

[8] W. Yang, R. T. Tan, J. Feng, S. Wang, B. Cheng, and J. Liu, “Recurrent multi-frame deraining: Combining physics guidance and adversarial learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8569–8586, 2022.

[9] H. Wu, Y. Yang, H. Chen, J. Ren, and L. Zhu, “Mask-guided progressive network for joint raindrop and rain streak removal in videos,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7216–7225.

[10] S. Sun, J. Lu, X. Chen, J. Wang, D. Lu, G. Zhang, G. Gao, and Z. Zheng, “DeLiVR: Differential spatiotemporal Lie bias for efficient video deraining,” in International Conference on Learning Representations, 2026.

[11] H. Wu, Y. Yang, H. Xu, W. Wang, J. Zhou, and L. Zhu, “RainMamba: Enhanced locality learning with state space models for video deraining,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 7881–7890.

[12] S. Sun, W. Ren, J. Zhou, S. Wang, J. Gan, and X. Cao, “Semisupervised state-space model with dynamic stacking filter for real-world video deraining,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 114–26 124.

[13] X. Liao, T. Zheng, J. Zhong, P. Zhang, and C. Ren, “Asymmetric mask scheme for self-supervised real image denoising,” in European Conference on Computer Vision, 2024, pp. 199–215.

[14] W. Liu, P. Zhang, H. Qin, X. Huang, J. Yang, and M. Ng, “Fast global image smoothing via quasi weighted least squares,” International Journal of Computer Vision, vol. 132, no. 12, pp. 6039–6068, 2024.

[15] W. Liu, P. Zhang, Y. Lei, X. Huang, J. Yang, and M. Ng, “A generalized framework for edge-preserving and structure-preserving image smoothing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6631–6648, 2022.

[16] J.-H. Kim, J.-Y. Sim, and C.-S. Kim, “Video deraining and desnowing using temporal correlation and low-rank matrix completion,” IEEE Transactions on Image Processing, vol. 24, no. 9, pp. 2658–2670, 2015.

[17] W. Ren, J. Tian, Z. Han, A. Chan, and Y. Tang, “Video desnowing and deraining based on matrix decomposition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 4210–4219.

[18] J. Liu, W. Yang, S. Yang, and Z. Guo, “D3R-Net: Dynamic routing residue recurrent network for video rain removal,” IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 699–712, 2019.

[19] Q. Wen, Y. Wu, and Q. Chen, “Video waterdrop removal via spatiotemporal fusion in driving scenes,” in 2023 IEEE International Conference on Robotics and Automation, 2023, pp. 10 003–10 009.

[20] T. Varanka, J. L. Gonzalez, H. Kim, P. Garrido, and X. Yao, “Zeroshot video deraining with video diffusion models,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 677–687.

[21] P. Yang, H. Ci, B. Lin, Y. Song, and M. Z. Shou, “UENR-600K: A large-scale physically grounded dataset for nighttime video deraining,” arXiv preprint arXiv:2604.04402, 2026.

[22] Y. Zhu, L. Lv, P. Zhang, X. Liu, T. Tang, F. Tian, W. Sun, and H. Lu, “Interactive spatial-frequency fusion Mamba for multi-modal image fusion,” IEEE Transactions on Image Processing, vol. 35, pp. 2380– 2392, 2026.

[23] H. Sun, L. Lv, P. Zhang, T. Tang, F. Tian, W. Sun, and H. Lu, “Spatial-frequency enhanced Mamba for multi-modal image fusion,” IEEE Transactions on Image Processing, vol. 34, pp. 7684–7696, 2025.

[24] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4713– 4726, 2023.

[25] J. Wang, Z. Yue, S. Zhou, K. C. K. Chan, and C. C. Loy, “Exploiting diffusion prior for real-world image super-resolution,” International Journal of Computer Vision, vol. 132, no. 12, pp. 5929–5949, 2024.

[26] J. Cai, Y. Lei, K. Wang, T. Zhong, F. Zhou, and J. Song, “Conditionenhanced guided diffusion for image super-resolution,” IEEE Transactions on Multimedia, vol. 28, pp. 13 079–13 094, 2026.

[27] Y. Liu, J. Pan, Y. Li, Q. Dong, C. Zhu, Y. Guo, and F. Wang, “UltraVSR: Achieving ultra-realistic video super-resolution with efficient one-step diffusion space,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 7785–7794.

[28] J. Gu, P.-T. Jiang, H. Zhang, M. Zhou, J. Chen, W. Yang, and B. Li, “Improving consistency in diffusion models for image super-resolution,” IEEE Transactions on Multimedia, vol. 28, pp. 12 864–12 875, 2026.

[29] J. Han, G. Sim, G. Kim, H.-S. Lee, K. Choi, Y. Han, and S. Cho, “DC-VSR: Spatially and temporally consistent video super-resolution with video diffusion prior,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference, 2025, pp. 1–11.

[30] A. Kim, M. Jang, C. Moon, Y. Shin, J. Jeong, and S.-h. Park, “FiDeSR: High-fidelity and detail-preserving one-step diffusion super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 38 270–38 280.

[31] Y. Zhu, Z. Zhu, M. Hašan, J. Yang, J. Xie, and B. Wang, “WeatherDiffusion: Weather-guided diffusion model for forward and inverse rendering,” arXiv preprint arXiv:2508.06982, 2025.

[32] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.

[33] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 5775–5787.

[34] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations, 2023.

[35] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in International Conference on Learning Representations, 2023.

[36] Team Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang, X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan, Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z.-F. Wu, and Z. Liu, “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[37] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning, 2023, pp. 32 211– 32 252.

[38] S. Luo, Y. Tan, L. Huang, J. Li, and H. Zhao, “Latent consistency models: Synthesizing high-resolution images with few-step inference,” in International Conference on Learning Representations, 2024.

[39] Y. Wang, W. Yang, X. Chen, Y. Wang, L. Guo, L.-P. Chau, Z. Liu, Y. Qiao, A. C. Kot, and B. Wen, “SinSR: Diffusion-based image superresolution in a single step,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 796–25 805.

[40] R. Wu, L. Sun, Z. Ma, and L. Zhang, “One-step effective diffusion network for real-world image super-resolution,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 92 529–92 553.

[41] C.-H. Yeh, H.-S. Shiu, C.-Y. Lin, Z. Wang, C.-W. Hsiao, T.-H. Chen, and Y.-L. Liu, “DiffIR2VR-Zero: Zero-shot video restoration with diffusionbased image restoration models,” arXiv preprint arXiv:2407.01519, 2024.

[42] Y. Liu, Z. Ke, F. Liu, N. Zhao, and R. W. H. Lau, “Diff-Plugin: Revitalizing details for diffusion-based low-level tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4197–4208.

[43] S. Chen, T. Ye, K. Zhang, Z. Xing, Y. Lin, and L. Zhu, “Teaching tailored to talent: Adverse weather restoration via prompt pool and depth-anything constraint,” in European Conference on Computer Vision, 2024, pp. 95–115.

[44] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang, and C. Dong, “DiffBIR: Toward blind image restoration with generative diffusion prior,” in European Conference on Computer Vision, 2024, pp. 430–448.

[45] Z. Chen, Z. Zou, K. Zhang, X. Su, X. Yuan, Y. Guo, and Y. Zhang, “DOVE: Efficient one-step diffusion model for real-world video superresolution,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 94 905–94 924.

[46] Z. Teed and J. Deng, “RAFT: Recurrent all-pairs field transforms for optical flow,” in European Conference on Computer Vision, 2020, pp. 402–419.

[47] J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” arXiv preprint arXiv:1804.02767, 2018.

[48] Y. Chen, Y. Cao, H. Hu, and L. Wang, “Memory enhanced globallocal aggregation for video object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 10 337–10 346.

[49] X. Zhou, V. Koltun, and P. Krähenbühl, “Global tracking transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8771–8780.