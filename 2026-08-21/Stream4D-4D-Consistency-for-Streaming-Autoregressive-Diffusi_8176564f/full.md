# Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models

Yuanhao Ban<sup>1</sup>, Jiaqi Feng<sup>2</sup>, Hengguang Zhou<sup>1</sup>, Xiaohuan Pei<sup>1</sup>, Justin Cui<sup>1</sup>, and Cho-Jui Hsieh<sup>1</sup>

<sup>1</sup>UCLA <sup>2</sup>Tsinghua University

August 21, 2026

## Abstract

Streaming autoregressive diffusion models enable real-time, long-horizon video generation, but their training objectives optimize local frame prediction rather than the geometry and dynamics of a coherent world: long rollouts accumulate geometric drift and degrade into static or unnatural motion. Recent bidirectional approaches address this problem using rewards signals built upon 3D Gaussian Splatting reconstruction. However, a single rigid 3d reconstruction cannot model a dynamic scene, so this critic penalizes genuine object motion as reconstruction error and is maximized by freezing the video. This shortcut is especially detrimental in the AR setting, where each chunk can propagate an already-static configuration. In this work, we propose Stream4D, which replaces the static critic with a feed-forward 4D reconstruction reward that explicitly models scene dynamics, allowing coherent motion to receive high consistency rewards. To further guide motion magnitude and quality, we add a motion prior that rewards natural scene-flow magnitude while penalizing jitter and non-rigid artifacts. Our final recipe combines these two terms with a lightweight perceptual anchor. Across various autoregressive video backbones and various generation horizons, Stream4D improves 4D reconstruction quality, preserves motion more effectively, and achieves higher human-aligned preference. Project page: https://banyuanhao.github.io/Stream4D

## 1 Introduction

Autoregressive (AR) diffusion video models (Huang et al., 2025; Yang et al., 2025; Cui et al., 2025; Zhu et al., 2026; Yesiltepe et al., 2026) generate videos chunk by chunk, enabling extendable temporal horizons. Combined with recent few-step backbones, such as Wan2.1 (Wan et al., 2025), they offer a practical path toward real-time streaming video generation for embodied agents, simulators, and interactive applications. However, existing objectives do not sufficiently enforce consistent world dynamics over long rollouts. A successful rollout should maintain both 4D consistency geometry, depth, object identity, and camera motion, and a natural motion prior with realistic magnitude and temporal smoothness. In practice, AR models often accumulate errors over time, causing scale drift, inconsistent depth relationships, and implausible scene or camera motion.

Recent reinforcement-learning approaches, including VideoGPA (Du et al., 2026) and World-R1 (Wang et al., 2026a), attempt to improve geometric consistency using rewards derived from static 3D Gaussian Splatting (3D-GS) reconstruction. However, the rigid 3d reconstruction model prior cannot represent a dynamic scene. Any genuine object motion appears as reconstruction error and is punished by the reward, which results in frozen scene and permitting only camera movement, as illustrated in Figure 1. Moreover, these rewards were originally developed for bidirectional video-generation settings, whereas their limitations become more pronounced in autoregressive generation. An AR model can attend only to previously generated frames and therefore has less temporal context with which to maintain coherent scene dynamics. Under this restricted information, collapsing to a static scene becomes an even easier shortcut: once motion is suppressed in early frames, subsequent chunks can continue propagating the same rigid configuration while still receiving a high reconstruction-based reward.

![](images/1056f781ad6a5abb3ffd23c066a83864f2284182985cccf4fcc766602b8892a9.jpg)  
Figure 1: Static-3D rewards freeze the scene; a 4D reward keeps it moving. Five uniformly-spaced frames from a 10.3 s LongLive rollout (“a cat running away with a fish while people chase behind”). The distilled base contains large motion but the cat and fish drift across frames; the static-3DGS rewards World-R1 and VideoGPA reduce the scene to a more rigid, low-motion configuration. Stream4D keeps the cat running with clear forward motion while preserving a coherent subject and scene.

To address the issues mentioned above, we introduce Stream4D, a reinforcement-learning framework that distills 4D-consistency and natural-motion priors into streaming AR video models. Our key insight is that recent feed-forward 4D Gaussian Splatting (4D-GS) models, such as MoVieS (Lin et al., 2026), provide a useful learned prior over how dynamic scenes evolve in space and time. Given a rollout sampled from the student policy, we reconstruct it using a pretrained 4D-GS model, re-render the resulting dynamic scene from its estimated camera trajectory, and measure perceptual agreement between the rendered frames and the original rollout. This reconstruction score rewards rollouts that can be explained by coherent geometry, motion, and camera dynamics, serving as a good 4d-consistency score.

To further imporve the motion quality and guide the tensity, Stream4D adds a Gaussian motion reward peaking at the natural motion tensity that penalizes both insufficient and excessive motion. We further modulate it using temporal-smoothness and rigidity-quality factors, preventing the policy from reaching the target motion magnitude through jitter, blur, or geometrically implausible shortcuts. Finally, we include a lightweight image reward model as a perceptual anchor to preserve the visual fidelity of the original model. The three reward components are standardized independently using z-normalization and then combined additively, producing a lightweight recipe that transfers across multiple distilled AR backbones. We evaluate Stream4D on three streaming AR architectures, covering both 5s and 10s video generation. Across Self-Forcing (Huang et al., 2025), Causal-Forcing (Zhu et al., 2026), and LongLive (Yang et al., 2025) backbones, Stream4D improves 4D-PSNR from 16.88, 15,44 and 17.44 to 20.34, 20.97 24.20, respectively, while preserving motion and maintaining strong performance under general-purpose video reward models. Our contributions are threefold:

• We identify accumulated 4D inconsistency as a central failure mode of distilled streaming AR video models and formulate its mitigation as a reinforcement-learning problem.

• We introduce Stream4D, a transferable training objective that combines structured 4D reconstruction consistency, a target-centered natural-motion prior, and perceptual anchoring.

• We demonstrate consistent improvements across three distilled AR backbones, including gains of up to 6.76 dB in 4D-PSNR, while preserving motion and general video quality over both short and long generation horizons.

## 2 Related Work

## 2.1 Autoregressive Diffusion Video Models

Diffusion-based video generation has progressed from joint-frame DiTs (Yang et al., 2024; Bao et al., 2024; Kong et al., 2024; Wan et al., 2025) toward streaming autoregressive variants that emit clips chunk-by-chunk. Early teacher-forced AR approaches (Hu et al., 2024; Gao et al., 2024) suffered from error accumulation; Diffusion Forcing (Chen et al., 2024), CausVid (Yin et al., 2025) with distribution-matching distillation (Yin et al., 2024), Self-Forcing (Huang et al., 2025), LongLive (Yang et al., 2025), Infinity-RoPE (Yesiltepe et al., 2026), and Causal-Forcing (Zhu et al., 2026) progressively closed the train–test gap via per-frame noise schedules, block-causal attention, self-rollout, and AR teachers. The resulting distilled streaming mod els (Sun et al., 2025) support real-time generation and double as video world models, but their supervision provides no incentive for the 3D scene to remain coherent across the rollout explicitly.

## 2.2 Reinforcement Learning for Image and Video Generation

Dance-GRPO (Xue et al., 2025) and Flow-GRPO (Liu et al., 2025a) run on-policy GRPO over reverseprocess trajectories, requiring log-probability estimation along the sampling chain and full trajectory storage. DiffusionNFT (Zheng et al., 2025) reformulates RL on the forward process via negative-aware fine-tuning; WorldCompass (Wang et al., 2026b) adapts it to autoregressive world models, and Astrolabe (Zhang et al., 2026) brings it to distilled streaming AR backbones with a rolling KV cache and a multi-reward objective.

## 2.3 Geometry-Aware Rewards for Video Generation

A complementary line uses 3D reconstruction as a critic. World-R1 (Wang et al., 2026a) combines a feed forward depth/camera estimator (Lin et al., 2025), a per-clip 3D Gaussian-Splat reconstruction (Ye et al., 2025), and a VLM critic (Bai et al., 2025) into a 3D-aware reward for bidirectional text-to-video under Flow-GRPO; VideoGPA (Du et al., 2026) extends this direction with preference alignment and a static 3D evaluation suite (Kupyn et al., 2025; Liu et al., 2025b). Both reconstruct the scene as a single rigid 3D Gaussian-Splat, which cannot represent a dynamic scene; consequently inter-frame object motion is structurally penalized as 3D inconsistency, and the reward is maximized by suppressing motion rather than merely being indifferent to it. We recast this 3D-consistency framing as 4D-consistency by replacing the reconstructor with MoVieS (Lin et al., 2026), a feed-forward 4D-GS model that factors the scene into a canonical point cloud plus per-frame attribute overrides and scene-flow offsets.

## 3 Method

Stream4D is a reinforcement-learning recipe for distilled autoregressive (AR) video models. Each rollout is graded against a feed-forward 4D Gaussian-Splatting reconstruction of itself, through 4D reconstruction consistency, a gated motion term, and a lightweight perceptual anchor. Then the rewards are summed under per-axis z-normalization and optimized with the forward-process DiffusionNFT loss.

## 3.1 Preliminaries

Forward-process streaming-AR RL. We build on Astrolabe (Zhang et al., 2026), which combines rolling-KV-cache generation with the DiffusionNFT (Zheng et al., 2025) update for distilled AR video models. From a shared context, we sample a group of G candidate rollouts $\{ \bar { W } ^ { ( i ) } \} _ { i = 1 } ^ { G }$ for group-wise reward normalization. For each rollout and denoising step, let $v _ { \theta }$ and $v _ { \theta _ { \mathrm { o l d } } }$ denote the current and reference-model velocity predictions in the latent-video space. We write $v _ { \theta } , v _ { \theta _ { \mathrm { o l d } } } , v ^ { + } , v ^ { - } , v _ { \mathrm { t a r g e t } } \in \mathbb { R } ^ { d }$ after flattening the latent tensor. DiffusionNFT forms positive and negative interpolated velocity predictions

$$
v ^ { + } = ( 1 - \beta ) v _ { \theta _ { \mathrm { o l d } } } + \beta v _ { \theta } , \qquad v ^ { - } = ( 1 + \beta ) v _ { \theta _ { \mathrm { o l d } } } - \beta v _ { \theta } ,\tag{1}
$$

and optimizes

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = \tilde { r } \left. v ^ { + } - v _ { \mathrm { t a r g e t } } \right. _ { 2 } ^ { 2 } + ( 1 - \tilde { r } ) \left. v ^ { - } - v _ { \mathrm { t a r g e t } } \right. _ { 2 } ^ { 2 } ,\tag{2}
$$

where $v _ { \mathrm { t a r g e t } }$ is the diffusion velocity target and $\tilde { r } \in [ 0 , 1 ]$ is the normalized rollout reward. Training details are provided in Sec. 4.1.

MoVieS 4D-GS reconstruction. Given a candidate rollout $W ^ { ( i ) } = \{ W _ { t } ^ { ( i ) } \} _ { t = 1 } ^ { T }$ , we first estimate per-frame cameras with StreamVGGT (Zhuo et al., 2026). We then run MoVieS (Lin et al., 2026), a feed-forward 4D Gaussian-Splatting reconstructor, conditioned on the sampled frames and estimated cameras. MoVieS represents the video as a dynamic Gaussian scene: a canonical set of 3D Gaussians together with timedependent deformation and appearance parameters that allow the scene to move and change across frames. In our pipeline, we use two outputs from this reconstruction. First, MoVieS re-renders reconstructed frames $\tilde { W } _ { t } ^ { ( i ) }$ from the estimated camera at each time t, which we use for the reconstruction reward. Second, MoVieS provides a per-pixel 3D motion field $P \in \mathbb { R } ^ { T \times H \times W \times 3 }$ and confidence map conf $\in [ 0 , 1 ] ^ { T \times H \times W }$ , which we use to compute motion magnitude, smoothness, and rigidity.

## 3.2 Stream4D Reward design

This section first defines the three reward components and then explains how they are combined into the final advantage.

4D-GS reconstruction $R _ { \mathbf { r e c o n } } .$ . For each candidate rollout, we subsample 26 frames and use the MoVieS reconstruction described above to render the corresponding reconstructed frames from the estimated cameras. We compare each rendered frame with the original generated frame and define the reconstruction reward as the clipped average perceptual agreement:

$$
R _ { \mathrm { r e c o n } } = \mathrm { c l i p } \left( 1 - \frac { 1 } { T } \sum _ { t } \mathrm { L P I P S } ( \tilde { W } _ { t } ^ { ( i ) } , W _ { t } ^ { ( i ) } ) , 0 , 1 \right) .\tag{3}
$$

Here $\boldsymbol { W } _ { t } ^ { ( i ) }$ is the t-th frame of candidate rollout $i , \tilde { W } _ { t } ^ { ( i ) }$ is the corresponding frame re-rendered from the MoVieS 4D-GS reconstruction using the estimated camera, and LPIPS (Zhang et al., 2018) denotes the

Learned Perceptual Image Patch Similarity distance. LPIPS measures perceptual discrepancy in a pretrained deep feature space, making it less sensitive than pixel-wise losses to small low-level misalignments while still penalizing visual reconstruction errors. Thus, the reward is high when the rollout can be explained by a coherent 4D reconstruction and low when the video contains inconsistent geometry, drifting object identity, or motion that cannot be organized into a stable dynamic scene.

Gated motion conjunction $R _ { \mathbf { m o t } } .$ . Unlike the static 3D reward, the 4D reconstructor does not punish motion. However, lower-motion clips are still slightly easier to reconstruct. As shown in Fig. 2, there is a weak Spearman $\rho = - 0 . 2 7$ between per-prompt 4D-PSNR and the motion-gate input m computed on the Self-Forcing base model over the held-out prompts. To address this, we add an explicit, quality-weighted motion reward that guides both motion intensity and motion quality:

$$
R _ { \mathrm { m o t } } = g ( m ) \cdot \mathrm { s m o o t h \cdot r i g i d } ,\tag{4}
$$

where $m$ is the motion intensity, $g ( \cdot )$ is a Gaussian gate, and smooth and rigid are quality factors that penalize jitter and spatially erratic flow. We compute these terms from the MoVieS motion field $P$ and confidence map conf defined in Sec. 3.1. Let $\begin{array} { r c l } { v _ { t } } & { = } & { P _ { t + 1 } \mathrm { ~ - ~ } } \end{array}$ $P _ { t }$ be the per-pixel 3D scene-flow velocity and $\| v _ { t } \|$ its magnitude. The dynamic mask $\mathcal { D }$ contains the top 20% fastest pixel-time entries in the clip. We write $\begin{array} { r } { \mathrm { m e a n } _ { \mathcal { D } } ^ { \mathrm { c o n f } } [ f ] = \left( \sum _ { ( p , t ) \in \mathcal { D } } \mathrm { c o n f } _ { p , t } f _ { p , t } \right) / \left( \sum _ { ( p , t ) \in \mathcal { D } } \mathrm { c o n f } _ { p , t } \right) } \end{array}$ for the confidence-weighted mask mean and mean<sub>D</sub>[f] for the uniform one. The clip-level motion magnitude is $m =$ mean $\mathbf { \mathbf { \mathbf { \mathbf { \phi } } } } _ { \mathcal { D } } ^ { \mathrm { c o n f } } \| \mathbf { \mathbf { \mathbf { \mathit { v } } } } _ { t } \|$

![](images/c4ff4d783f21928741fbc02ff97c499fe8b9e4ecc3ba9040a2c5711daf7edd34.jpg)  
Figure 2: Unlike a static 3D reward, which punishes motion outright, our dynamic 4D-GS backbone does not penalize motion. Perprompt 4D-PSNR only drifts weakly downward with motion magnitude m.

The motion gate is a peaked Gaussian penalizing both static collapse and runaway blur:

$$
\begin{array} { r } { g ( m ) \ = \ \exp \left( - \frac { ( m - m _ { \mathrm { n a t } } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , } \end{array}\tag{5}
$$

where the target $m _ { \mathrm { n a t } }$ is the median per-clip m of base rollouts. We choose the Gaussian kernel because it penalizes both overly aggressive and nearly static motion, thereby encouraging motion intensity to remain within a natural range. We also evaluate alternative formulations, such as a linear motion function, in the ablation study in Sec. 4.

Smoothness penalizes excessive aggregate motion and deformation and is formulated as follows:

$$
\mathrm { s m o o t h } \ = \ \exp \biggl ( - \frac { \mathrm { m e a n } _ { \mathscr D } ^ { \mathrm { c o n f } } \left\| v _ { t + 1 } - v _ { t } \right\| } { m + c } \biggr ) .\tag{6}
$$

where $c = 1 0 ^ { - 4 }$ is a small constant for numerical stability.

Rigidity penalizes spatially erratic scene flow. Although real objects can deform or articulate, nearby points on the same moving object usually have locally coherent 3D velocities. We therefore compute the spatial finite differences of the per-pixel scene-flow velocity $v _ { t }$ within the dynamic mask $\mathcal { D } .$ . Large $\| \nabla _ { \boldsymbol { x } } \boldsymbol { v } _ { t } \|$ or $\| \nabla _ { y } v _ { t } \|$ indicates neighboring pixels moving with sharply different 3D velocities, which often corresponds to tearing, jitter, or melting artifacts. We map this roughness to a bounded quality factor:

$$
\begin{array} { r } { \mathrm { r i g i d ~ } = \mathrm { \exp } \left( - \frac { k _ { \mathrm { r o u g h } } } { 2 } \left( \mathrm { m e a n } _ { \mathcal D } \| \nabla _ { x } v _ { t } \| + \mathrm { m e a n } _ { \mathcal D } \| \nabla _ { y } v _ { t } \| \right) \right) , \quad k _ { \mathrm { r o u g h } } = 4 0 0 . } \end{array}\tag{7}
$$

This encourages locally coherent motion while still allowing smooth non-rigid or articulated dynamics.

Perceptual anchor $R _ { \mathbf { h p s v } 2 }$ . We use HPSv2 (Wu et al., 2023) to score the aesthetics of the generated frames directly. The reconstruction and motion terms constrain geometry and dynamics but leave per-frame appearance unconstrained. The perceptual anchor addresses this by tying the reward to human aesthetic preference, keeping generations close to the visual fidelity of the base model.

Z-norm reward ensembling. Following (Liu et al., 2026), our final reward sums three z-normalized axes evaluated on each candidate rollout $W ^ { ( i ) }$ :

$$
R ( W ^ { ( i ) } ) ~ = ~ w _ { \mathrm { r e c o n } } ~ \tilde { z } \big [ R _ { \mathrm { r e c o n } } \big ] ~ + ~ w _ { \mathrm { m o t } } ~ \tilde { z } \big [ R _ { \mathrm { m o t } } \big ] ~ + ~ w _ { \mathrm { h p s v 2 } } ~ \tilde { z } \big [ R _ { \mathrm { h p s v 2 } } \big ] ,\tag{8}
$$

where $\tilde { z } [ \cdot ]$ is the per-axis z-score over group. Following the DiffusionNFT (Zheng et al., 2025), we compute the group-centered advantage $\begin{array} { r } { A ^ { ( i ) } = \hat { R } ( \hat W ^ { ( i ) } ) - \frac { 1 } { G } \hat { \sum _ { j } } R ( W ^ { ( j ) } ) } \end{array}$ with clipped affine normalization,

$$
\tilde { r } ^ { ( i ) } = \mathrm { c l i p } \Big ( A ^ { ( i ) } / A _ { \mathrm { m a x } } , - 1 , 1 \Big ) / 2 + 1 / 2 ,\tag{9}
$$

where $A _ { \mathrm { m a x } } { = } 5$ is a fixed constant.

## 4 Experiments

## 4.1 Setup

Backbones and training. We evaluate Stream4D on three distilled autoregressive video backbones: Self-Forcing (Huang et al., 2025), Causal-Forcing (Zhu et al., 2026), and LongLive (Yang et al., 2025), using the Stream4D reward formulation in Eq. 8. Self-Forcing and Causal-Forcing use their native 81-frame setting, corresponding to approximately 5s at 16fps, while LongLive uses a 10.3,s window of 165 frames. A LoRA adapter is trained over the frozen base with the forward-process NFT loss of Sec. 3.1. We trained every method for 150 epochs and reported the results on final checkpoints. We randomly sampled training prompts from VidProM (Wang and Yang, 2024). Full hyperparameters, the reward-stack implementation, and compute details are in Appendices A and B; the full procedure is Algorithm 1.

Evaluation protocol. To stress-test the motion-preservation capabilities of our method, we construct a 500-prompt motion-prominent subset of VidProM using a motion filter and a keyword-based filter targeting multi-agent interactions and object motion. We also report results on 500 randomly selected prompts in Appendix G. We evaluate with three groups of metrics:

• MoVieS Recon (PSNR ↑ / SSIM ↑ / LPIPS ↓): re-render error of the MoVieS 4D-GS reconstruction. Full computation pipeline and the definition of PSNR, SSIM and LPIPS could be found in Appendix D.

• 4DGT Recon To check a gain is not MoVieS-specific, we also report it under 4DGT (Xu et al., 2025), a reconstructor with disjoint architecture, weights, and data.

• LLM judge (Motion ↑, Consist. win% ↑): a reward-blind Gemini-3.5-Flash judge, order-debiased, scoring object-motion preservation and a head-to-head consistency verdict that requires kept motion , so freezing cannot win solely. Judge details like prompts and debias mechanism can be found in Appendix E.

• VideoReward (Liu et al., 2025b) (VQ / MQ / TA / Overall ↑): paired win-rate vs. the distilled base under a learned video-quality model.

## 4.2 Main results

Table 1 reports Stream4D at ckpt-150 on the 500 motion-prominent testing set against the distilled base, World-R1 (Wang et al., 2026a) and VideoGPA (Du et al., 2026) on each of the three bacbones.

Table 1: Main results on the 500 object-prominent testing set. Metrics as defined in Sec. 4.1: MoVieS 4D reconstruction score; the 4DGT reconstruction (Xu et al., 2025) score (a reconstructor-independent crosscheck, Appendix D); the order-debiased LLM judge evaluating Motion preservation and Consistency); and VideoReward (Liu et al., 2025b) paired win% vs. base.
<table><tr><td rowspan="2">Method</td><td colspan="3">MoVieS Recon.</td><td colspan="3">4DGT Recon.</td><td colspan="2">LLM judge</td><td colspan="3">VideoReward win%</td></tr><tr><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td>Motion↑</td><td>Consist.↑ VQ↑</td><td>MQ↑</td><td></td><td>TA↑ Ovr↑</td></tr><tr><td colspan="10">Reference: full-diffusion T2V baselines (non-distilled, no RL)</td></tr><tr><td>Wan2.1-T2V-1.3B (Wan et al., 2025)</td><td>19.15 0.828</td><td>0.258</td><td>17.14</td><td>0.614</td><td></td><td>0.399</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CogVideoX-1.5-5B (Yang et al., 2024)</td><td>17.02 0.768</td><td>0.279</td><td></td><td>15.06</td><td>0.518</td><td>0.454</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Distilled AR backbone: Self-Forcing (Huang et al., 2025) (5 s)</td></tr><tr><td>Base (B0)</td><td>0.785</td><td></td><td></td><td>0.559</td><td>0.419</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>16.88 18.52</td><td>0.865</td><td>0.282 0.197</td><td>16.28 16.56 0.566</td><td>0.396</td><td>0.832</td><td>75.9</td><td>66.0</td><td>50.6</td><td>54.0</td><td>61.8</td></tr><tr><td>VideoGPA (Du et al., 2026)</td><td>17.75</td><td>0.868</td><td>0.202</td><td>16.00</td><td>0.529</td><td>0.401</td><td>0.737 60.9</td><td>51.8</td><td>39.5</td><td>61.9</td><td>48.0</td></tr><tr><td>+ Stream4D (Ours)</td><td>20.34 0.874</td><td></td><td>0.195</td><td>17.24</td><td>0.650</td><td>0.417</td><td>0.833 82.2</td><td>62.4</td><td>62.8</td><td>56.2</td><td>66.2</td></tr><tr><td colspan="10">Distilled AR backbone: Causal-Forcing (Zhu et al., 2026) (5 s)</td></tr><tr><td>Base (B0)</td><td></td><td></td><td></td><td></td><td>0.450</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>15.44 19.18</td><td>0.767 0.884</td><td>0.279 0.167</td><td>15.03 16.13</td><td>0.515</td><td></td><td>69.1</td><td>76.8</td><td>66.8</td><td>63.0</td><td>74.8</td></tr><tr><td>VideoGPA (Du et al., 2026)</td><td>18.04</td><td>0.878</td><td>0.195</td><td>14.86</td><td>0.565 0.503</td><td>0.424 0.495</td><td>0.676 0.706</td><td>60.4</td><td>72.0</td><td>58.3 64.9</td><td>66.3</td></tr><tr><td>+ Stream4D (Ours)</td><td>20.97</td><td>0.893</td><td>0.150</td><td>17.21</td><td>0.643</td><td>0.421</td><td>0.765</td><td>73.9</td><td>77.8</td><td>73.2 59.0</td><td>76.0</td></tr><tr><td colspan="10">Distilled AR backbone: LongLive (Yang et al., 2025) (10.3 s)</td></tr><tr><td>Base (B0)</td><td>17.44</td><td>0.844</td><td>0.231 15.45</td><td>0.521</td><td>0.428</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>22.64</td><td>0.941</td><td>0.135</td><td>17.55</td><td>0.641 0.420</td><td>0.498</td><td>54.0</td><td>81.4</td><td>67.6</td><td>66.8</td><td>78.2</td></tr><tr><td>VideoGPA (Du et al., 2026)</td><td>20.50</td><td>0.933</td><td>0.147</td><td>17.19</td><td>0.608</td><td>0.374</td><td>0.306</td><td>32.9</td><td>59.3 47.6</td><td>63.5</td><td>57.9</td></tr><tr><td>+ Stream4D (Ours)</td><td>24.20</td><td>0.905</td><td>0.146</td><td>20.03</td><td>0.663</td><td>0.392</td><td>0.706</td><td>74.2</td><td>81.6 79.0</td><td>74.0</td><td>84.4</td></tr></table>

4D-consistency gains transfer across all three backbones. Stream4D lifts 4D-PSNR by +3.46 dB on Self-Forcing , +5.53 dB on Causal-Forcing, and +6.76 dB on LongLive , with SSIM and LPIPS improving on all three. Under 4DGT (Xu et al., 2025), a reconstructor with disjoint architecture, weights, and training data from MoVieS, Stream4D again posts the best PSNR and SSIM in every backbone block, leading World-R1 by +0.7 / +1.1 / +2.5 dB. This indicates the 4D-reconstruction gain is not specific to MoVieS model; it does not by itself certify metric-accurate geometry, a caveat we make precise in Appendix D. Furthermore, the vision-LLM judge gives Stream4D the best consistency score on every backbone, beating base on 82.2 % / 73.9 % / 74.2 % of prompts against World-R1’s 75.9 % / 69.1 % / 54.0 %, confirming the improved consistency.

Object motion is preserved; 3D method collapses it. The vision-LLM judge scores Stream4D’s motion preservation at 0.83 / 0.77 / 0.71 on SF / CF / LL highest across the three backnones compared to the baselines. One can see qualitative filmstrips for all three backbones in Appendix Fig. S1, showing the failure mode directly: the static-reward baselines lock the subject in place while Stream4D keeps it moving. Stream4D beats World-R1 on VideoReward-Overall on every backbone. Overall win-rates against base are 66.2 % / 76.0 % / 84.4 %, versus World-R1’s 61.8 % / 74.8 % / 78.2 %. The margin is largest on the motion-quality head (+12.2 / +6.4 / +11.4 pp): on motion-prominent prompts, a reward that keeps motion alive wins preference over one that freezes it.

Human evaluation. We conduct a blinded human study on the 10.3s LongLive backbone, where accumulated drift and the static-collapse shortcut are most severe. We sample 50 high-motion prompts and form the three pairings (Stream4D vs. base, World-R1, VideoGPA), giving 150 two-alternative forced-choice comparisons, each judged once and distributed across 5 raters. Each trial asks which clip shows more natural motion and which is better consistency jointly weighing motion amount, motion quality, and object consistency. As shown in Table 2, humans prefer Stream4D over both geometry-reward baselines by a wide

Table 2: Human study on LongLive. We focus the human evaluation on the long-horizon LongLive backbone, where accumulated drift and the static-collapse shortcut matter most. For each pair raters pick which video shows more natural motion and which is better overall (jointly weighing motion amount, motion quality, and object consistency); win% is Stream4D’s rate (ties count <sup>1</sup><sub>2</sub> , >50% prefers ours). For reference we list the Gemini judge’s Motion and Consistency win% on the same 50 prompts: the judge agrees with humans on every ranking (Stream4D preferred over both reward baselines, and below 50% on raw motion vs. base), corroborating that the automatic scores are not an artifact, though the judge is somewhat more generous to ours in absolute terms.
<table><tr><td rowspan="2">Stream4D vs.</td><td colspan="2">Human win%↑</td><td colspan="2">Gemini judge win%↑</td></tr><tr><td>Motion</td><td>Overall</td><td>Motion</td><td>Consist.</td></tr><tr><td>Base</td><td>43</td><td>60</td><td>63</td><td>70</td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>72</td><td>76</td><td>93</td><td>68</td></tr><tr><td>VideoGPA (Du et al., 2026)</td><td>87</td><td>80</td><td>95</td><td>82</td></tr></table>

Table 3: Reward-axis ablation: dropping one of the three reward axes, on all three backbones. Each block lists the distilled base, the deployed Stream4D recipe, then the three single-axis drops. Metric conventions match Table 1.
<table><tr><td rowspan="2">Reward configuration</td><td colspan="3">MoVieS Recon.</td><td colspan="3">4DGT Recon.</td><td colspan="2">LLM judge</td><td colspan="3">VideoReward win%</td></tr><tr><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>Motion↑</td><td>Consist.↑</td><td>VQ↑ MQ↑ TA↑ Ovr↑</td><td></td><td></td></tr><tr><td colspan="10">Backbone: Self-Forcing (5 s)</td></tr><tr><td>Base (no RL)</td><td>16.88</td><td>0.785</td><td>0.282</td><td>16.28</td><td>0.559</td><td>0.419</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stream4D (Ours)</td><td>20.34</td><td>0.874</td><td>0.195</td><td>17.24</td><td>0.650</td><td>0.417</td><td>0.833</td><td>82.2 62.4</td><td>62.8</td><td>56.2</td><td>66.2</td></tr><tr><td>- perceptual anchor (a)</td><td>22.39</td><td>0.905</td><td>0.163</td><td>18.53</td><td>0.723</td><td>0.371</td><td>0.830 81.7</td><td>66.4</td><td>64.8</td><td>52.8</td><td>66.6</td></tr><tr><td>– motion term (b)</td><td>24.37</td><td>0.952</td><td>0.096</td><td>19.36</td><td>0.738</td><td>0.328</td><td>0.341</td><td>31.1 63.0</td><td>56.8</td><td>44.0</td><td>59.2</td></tr><tr><td>— reconstruction (c)</td><td>13.98</td><td>0.692</td><td>0.342</td><td>14.17</td><td>0.466</td><td>0.476</td><td>0.983 24.2</td><td>31.6</td><td>26.4</td><td>34.6</td><td>28.2</td></tr><tr><td colspan="10">Backbone: Causal-Forcing (5 s)</td></tr><tr><td>Base (no RL)</td><td>15.44</td><td>0.767</td><td>0.279</td><td>15.03</td><td>0.515</td><td>0.450</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stream4D (Ours)</td><td>20.97</td><td>0.893</td><td>0.150</td><td>17.21</td><td>0.643</td><td>0.421</td><td>0.765</td><td>73.9</td><td>77.8</td><td>73.2 259.0</td><td>76.0</td></tr><tr><td>— perceptual anchor (a)</td><td>23.93</td><td>0.901</td><td>0.106</td><td>18.48</td><td>0.792</td><td>0.341</td><td>0.591</td><td>47.3</td><td>67.0 57.8</td><td>48.8</td><td>64.4</td></tr><tr><td>– motion term (b)†</td><td>24.19</td><td>0.862</td><td>0.094</td><td>18.89</td><td>0.776</td><td>0.329</td><td>0.453</td><td>22.4 71.6</td><td>56.2</td><td>51.0</td><td>63.4</td></tr><tr><td>— reconstruction (c)</td><td>14.88</td><td>0.706</td><td>0.297</td><td>14.41 0.414</td><td></td><td>0.495</td><td>0.984</td><td>45.1 45.2</td><td>50.4</td><td>50.0</td><td>49.2</td></tr><tr><td colspan="10">Backbone: LongLive (10.3 s)</td></tr><tr><td>Base (no RL)</td><td>17.44</td><td>0.844</td><td>0.231</td><td>15.45</td><td>0.521</td><td>0.428</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stream4D (Ours)</td><td>24.20</td><td>0.905</td><td>0.146</td><td>20.03</td><td>0.663</td><td>0.392</td><td>0.706</td><td>74.2 81.6</td><td></td><td>79.0 74.0</td><td>84.4</td></tr><tr><td>— perceptual anchor (a)</td><td>20.57</td><td>0.936</td><td>0.143</td><td>16.84</td><td>0.639</td><td>0.379</td><td>0.745</td><td>74.2</td><td>75.0 60.0</td><td>57.0</td><td>69.4</td></tr><tr><td>– motion term (b)</td><td>22.84</td><td>0.931</td><td>0.133</td><td>17.69</td><td>0.611</td><td>0.438</td><td>0.416</td><td>44.6</td><td>78.4 66.6 54.2</td><td>65.4</td><td>74.4</td></tr><tr><td>— reconstruction (c)</td><td>18.29</td><td>0.844</td><td>0.217</td><td>16.13</td><td>0.520</td><td>0.427</td><td>0.956</td><td>80.2</td><td>66.8</td><td>59.8</td><td>63.4</td></tr></table>

margin—76 % overall vs. World-R1 and 80 % vs. VideoGPA. Full protocol and screening are in Appendix F.

## 4.3 Reward ablation study

We study the reward in two parts: dropping each of the three axes in Table 3, and varying the shape of the motion term in Table 4.

Each axis is necessary on at least one backbone; none can be dropped safely on all three. In Table 3, dropping the motion term (b) collapses motion on every backbone and loses the vlm consistency score to

Table 4: Motion-term formulation ablation: varying the shape of the motion term $R _ { \mathrm { m o t } }$ while keeping the reconstruction and perceptual axes fixed, on all three backbones.
<table><tr><td rowspan="2">Motion-term  $R _ { \mathbf { m o t } }$ </td><td colspan="3">MoVieS Recon.</td><td colspan="3">4DGT Recon.</td><td colspan="2">LLM judge</td><td colspan="3">VideoReward win%</td></tr><tr><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>Motion↑ Consist.↑</td><td></td><td></td><td>VQ↑ MQ↑ TA↑ Ovr↑</td><td></td></tr><tr><td colspan="10">Backbone: Self-Forcing (5 s)</td></tr><tr><td>Base (no RL)</td><td>16.88</td><td>0.785</td><td>0.282</td><td>16.28 0.559</td><td></td><td>0.419</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>g(m)·smooth·rigid</td><td>20.34</td><td>0.874</td><td>0.195</td><td>17.24 0.650</td><td>0.417</td><td>0.833</td><td>82.2</td><td>62.4</td><td></td><td>62.8 56.2</td><td>66.2</td></tr><tr><td>smooth·rigid,</td><td>23.43</td><td>0.947</td><td>0.119</td><td>18.02 0.671</td><td>0.405</td><td>0.406</td><td>36.4</td><td>78.4</td><td>60.4</td><td>53.8</td><td>74.6</td></tr><tr><td>m·smooth·rigid (e)</td><td>18.46</td><td>0.833</td><td>0.234</td><td>16.85 0.632</td><td>0.392</td><td>0.923</td><td>71.2</td><td>46.4</td><td>47.4</td><td>41.8</td><td>46.0</td></tr><tr><td>g(m) only</td><td>23.07</td><td>0.907</td><td>0.155</td><td>18.99 0.745</td><td>0.356</td><td>0.795</td><td>75.8</td><td>64.4</td><td>65.0</td><td>52.2</td><td>64.0</td></tr><tr><td>g(m)+smooth+rigid</td><td>21.45</td><td>0.907</td><td>0.165</td><td>17.37 0.647</td><td>0.410</td><td>0.750</td><td>75.9</td><td>65.8</td><td>57.0</td><td>56.0</td><td>65.6</td></tr><tr><td colspan="10">Backbone: Causal-Forcing (5 s)</td></tr><tr><td>Base (no RL)</td><td>15.44</td><td>0.767</td><td>0.279</td><td>15.03</td><td>0.515</td><td>0.450</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>g(m)·smooth·rigid</td><td>20.97</td><td>0.893</td><td>0.150</td><td>17.21</td><td>0.643 0.421</td><td>0.765</td><td></td><td>73.9 77.8</td><td></td><td>73.2 59.0</td><td>76.0</td></tr><tr><td>smooth·rigid,</td><td>23.14</td><td>0.922</td><td>0.116</td><td>18.37</td><td>0.741</td><td>0.338 0.396</td><td></td><td>31.2</td><td>75.0</td><td>65.8 51.0</td><td>71.0</td></tr><tr><td>m·smooth·rigid (e)</td><td>22.76</td><td>0.918</td><td>0.120</td><td>17.74</td><td>0.728 0.400</td><td>0.578</td><td></td><td>61.8</td><td>69.6 68.6</td><td>49.6</td><td>68.8</td></tr><tr><td>g(m) only</td><td>21.68</td><td>0.898</td><td>0.130</td><td>17.85 0.710</td><td>0.355</td><td>0.619</td><td>58.4</td><td>72.2</td><td>64.0</td><td>50.4</td><td>68.2</td></tr><tr><td>g(m)+smooth+rigid</td><td>21.66</td><td>0.886</td><td>0.131</td><td>18.16 0.685</td><td>0.343</td><td>0.525</td><td>44.4</td><td>74.2</td><td>61.8</td><td>51.8</td><td>68.0</td></tr><tr><td colspan="10">Backbone: LongLive (10.3 s)</td></tr><tr><td>Base (no RL)</td><td>17.44</td><td>0.844</td><td>0.231</td><td>15.45</td><td>0.521</td><td>0.428</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>g(m)·smooth·rigid</td><td>24.20</td><td>0.905</td><td>0.146</td><td>20.03</td><td>0.663 0.392</td><td>0.706</td><td>74.2</td><td>81.6</td><td></td><td>79.0 74.0</td><td>84.4</td></tr><tr><td>smooth·rigid,</td><td>23.44</td><td>0.924</td><td>0.142</td><td>18.58 0.652</td><td>0.417</td><td>0.699</td><td>74.1</td><td>74.6 58.0</td><td>72.2</td><td>72.6</td><td>79.2</td></tr><tr><td>m·smooth·rigid (e)</td><td>18.85</td><td>0.889</td><td>0.193</td><td>15.91</td><td>0.578</td><td>0.459</td><td>0.884</td><td>58.6</td><td>56.8</td><td>60.4</td><td>60.6</td></tr><tr><td>g(m) only</td><td>20.02</td><td>0.847</td><td>0.232</td><td>18.92</td><td>0.766</td><td>0.287</td><td>0.978</td><td>64.0 37.4</td><td>64.0</td><td>50.6</td><td>50.2</td></tr><tr><td>g(m)+smooth+rigid</td><td>16.70</td><td>0.763</td><td>0.327</td><td>16.40</td><td>0.700</td><td>0.346</td><td>0.995</td><td>30.6</td><td>30.6 48.2</td><td>41.4</td><td>38.0</td></tr></table>

base on all three despite near-best PSNR. Dropping reconstruction (c) can achieve the most motion, but coherence collapses. Dropping the perceptual anchor makes CF lose to base on vlm consistency judge and on LL it costs 3.6 dB of 4D-PSNR and 15 pp of VideoReward-Overall.

## The motion term needs its peaked, multiplicative

shape. In Table 4, removing the gate (d) freezes SF and CF as smooth and rigid are trivially maximized by a near-static rollout. A linear term (e) over-drives motion on SF/LL at a reconstruction and preference cost. The gate alone (f) can preserve the motion and improve the reconstruction, but it will cost the VideoReward 2.2 7.8, and 34.2 pp and is less preferred by the vlm consistency judge, indicating the low quality motion appears without these two terms. The additive split (g) is benign on SF but a net loss on CF and catastrophic on LL.

![](images/ee6bf649354979b6ac92e4d8933ccff29a796990a5d80feb09ce8ab5ea77a96c.jpg)

Figure 3: Reconstruction-motion trade-off. DeTrade-off between motion and reconstruction. ployed recipes (stars) sit on the upper-right frontier.Figure 3 plots every trained variant in the reconstruction-motion plane. We can observe two failure modes: high-recon/low-motion frozen runs (upper left, e.g. rows (b),(d)) and high-motion/low-recon incoherent runs (lower right, e.g. row (c)). Note that the deployed recipes (stars) sit on the upper-right frontier.

Sensitivity to the gate target $m _ { \mathbf { n } \mathbf { a } \mathbf { t } }$ . The one constant the recipe calibrates from data is the gate target $m _ { \mathrm { n { a t } } } =$ 0.020. Table 5 ablates it at 0.010 and 0.030 on Self-Forcing and shows the gate behaves as designed: motion rises monotonically from $0 . 5 2  0 . 8 3  0 . 9 2$ and trades off MoVieS-PSNR from $2 1 . 1 0  2 0 . 3 4 $ 18.87. Setting it as 0.020 just hits a balance between motion and consistency, gaining the best LLM judge Consist score and Video reward results.

Table 5: Sensitivity to the gate target $m _ { \mathbf { n } \mathbf { a } \mathbf { t } }$ on Self-Forcing. The 0.020 row is the deployed recipe, landing on the knee of the motion-fidelity trade-off, taking the best Consistency and VideoReward-Overall.
<table><tr><td rowspan="2"> $m _ { \mathbf { n } \mathbf { a } \mathbf { t } }$ </td><td colspan="3">MoVieS Recon.</td><td colspan="3">4DGT Recon.</td><td colspan="2">LLM judge</td><td colspan="4">VideoReward win% vs base</td></tr><tr><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>Motion↑ Consist.↑</td><td></td><td>VQ↑ MQ↑ TA↑</td><td></td><td></td><td>Ovr↑</td></tr><tr><td>Self-Forcing base</td><td>16.88</td><td>0.785</td><td>0.282</td><td>16.28</td><td>0.559</td><td>0.419</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.010 (0.5×)</td><td>21.10</td><td>0.924</td><td>0.135</td><td>17.86</td><td>0.672</td><td>0.345</td><td>0.522</td><td>44.7</td><td>62.8</td><td>59.2</td><td>65.6</td><td>63.8</td></tr><tr><td>0.020 (deployed)</td><td>20.34</td><td>0.874</td><td>0.195</td><td>17.24</td><td>0.650</td><td>0.417</td><td>0.833</td><td>82.2</td><td>62.4</td><td>62.8</td><td>56.2</td><td>66.2</td></tr><tr><td>0.030 (1.5×)</td><td>18.87</td><td>0.836</td><td>0.228</td><td>17.50</td><td>0.645</td><td>0.374</td><td>0.916</td><td>74.4</td><td>54.0</td><td>51.0</td><td>45.6</td><td>53.2</td></tr></table>

## 5 Conclusion

We presented Stream4D, a reinforcement-learning recipe that moves geometric reward design for streaming AR video models from 3D- to 4D-consistency. A rigid 3D reconstruction cannot model a dynamic scene, so it penalizes motion and lets the policy earn reward by freezing the scene; a feed-forward 4D reconstruction does not penalize motion, and paired with a peaked motion gate and a light perceptual anchor it instead rewards coherent moving content. The single recipe transfers across three distilled backbones, improving 4D reconstruction, motion preservation, and preference win-rate over both the distilled base and a faithful World-R1 reproduction, with a reconstructor-independent cross-check and a reward-blind judge corroborating each gain. A natural next step is a streaming 4D reconstructor to match LongLive’s native horizon, and grading rollouts against explicit action or camera inputs for embodied control.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann.´ Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh.

Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

Hongyang Du, Junjie Ye, Xiaoyan Cong, Runhao Li, Jingcheng Ni, Aman Agarwal, Zeqi Zhou, Zekun Li, Randall Balestriero, and Yue Wang. Videogpa: Distilling geometry priors for 3d-consistent video generation. In International Conference on Machine Learning (ICML), 2026. arXiv:2601.23286.

Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024.

Haoran He, Yuxiao Ye, Jie Liu, Jiajun Liang, Zhiyong Wang, Ziyang Yuan, Xintao Wang, Hangyu Mao, Pengfei Wan, and Ling Pan. Gardo: Reinforcing diffusion models without reward hacking. arXiv preprint arXiv:2512.24138, 2025.

Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Orest Kupyn, Fabian Manhardt, Federico Tombari, and Christian Rupprecht. Epipolar geometry improves video generation models. arXiv preprint arXiv:2510.21615, 2025.

Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Tao Hu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. MoVieS: Motion-aware 4d dynamic view synthesis in one second. In CVPR, 2026. arXiv:2507.10065.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, and Wanli Ouyang. Improving video generation with human feedback. In Advances in Neural Information Processing Systems, 2025b.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y Chen, Zhiyuan He, et al. World-r1: Reinforcing 3d constraints for text-to-video generation. In International Conference on Machine Learning (ICML), 2026a. arXiv:2604.24764.

Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024.

Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, et al. Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022, 2026b.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4DGT: Learning a 4D Gaussian transformer using real-world monocular videos. In NeurIPS, 2025. arXiv:2506.08015.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025.

Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout. In CVPR, 2026. arXiv:2511.20649.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and¨ Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586–595. IEEE, 2018.

Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Yue Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051, 2026. ECCV 2026 submission.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregres sive diffusion distillation done right for high-quality real-time interactive video generation. In Interna tional Conference on Machine Learning (ICML), 2026. arXiv:2602.02214.

Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. In ICLR, 2026. arXiv:2507.11539.

This appendix is organized as follows: Appendix A details the deployed reward; Appendix B lists training hyperparameters and compute; Appendix C gives the full training loop; Appendix D details the evaluation protocol and metrics; Appendix E documents the LLM judge (prompts, position debiasing, reproducibility); Appendix F details the human-study protocol;Appendix G reports the random-subset robustness check; Appendix H collects qualitative comparisons; Appendix I states limitations; and Appendix ?? discusses the scope of the reward backbone.

## A Reward Implementation Details

This appendix gives the deployed-recipe details that the main paper Sec. 3.2 compresses. The reward $R ( W ^ { ( i ) } ) = w _ { \mathrm { r e c o n } } { \tilde { z } } [ R _ { \mathrm { r e c o n } } ] + w _ { \mathrm { m o t } } { \tilde { z } } [ R _ { \mathrm { m o t } } ] + w _ { \mathrm { h p s v } 2 } { \tilde { z } } [ R _ { \mathrm { h p s v } 2 } ]$ (Eq. 8) is computed as follows; the per-backbone axis weights $( w _ { \mathrm { r e c o n } } , w _ { \mathrm { m o t } } , w _ { \mathrm { h p s v 2 } } )$ are $( 1 . 0 , 1 . 0 , 0 . 3 )$ on Self-Forcing, (1.0, 0.5, 0.6) on Causal-Forcing, and $( 0 . 8 , 1 . 0 , 0 . 6 )$ on LongLive, and all other hyperparameters are shared.

4D-GS reconstruction backbone. StreamVGGT (Zhuo et al., 2026) predicts per-frame camera extrinsics $\{ C _ { t } \} _ { t = 1 } ^ { T }$ and intrinsics from a 26-frame linearly-subsampled version of the rollout at 294×518. MoVieS (Lin et al., 2026) consumes the same subsampled frames and the predicted cameras and returns: (i) a canonical Gaussian point cloud $G _ { \mathrm { c a n } }$ encoding frame-invariant geometry, (ii) per-frame Gaussian-attribute overrides (opacity, scale, color) $\{ A _ { t } \} _ { t = 1 } ^ { T }$ , and (iii) per-frame scene-flow offsets $\{ \Delta _ { t } \} _ { t = 1 } ^ { T }$ . We re-render each frame with GaussianRenderer.render using $G _ { \mathrm { c a n } } \oplus A _ { t } \oplus \Delta _ { t }$ at $C _ { t }$ , then compute frame-wise LPIPS-AlexNet against the input frame and combine as in Eq. 3. A single MoVieS forward per candidate rollout is shared between $R _ { \mathrm { r e c o n } }$ and $R _ { \mathrm { m o t } }$ via a per-batch cache.

Motion-quality conjunction. $R _ { \mathrm { m o t } } = g ( m )$ · smooth · rigid exactly as in Eqs. $4 { - } 7 \colon g ( m ) = \exp ( { - } ( m -$ $m _ { \mathrm { n a t } } ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) ,$ ) with $m _ { \mathrm { n { a t } } } = 0 . 0 2 0 , \sigma = 0 . 0 1 0$ (MoVieS canonical-scene units); smoot $\begin{array} { r }  \mathbf { \tau } _ { 1 } = \exp ( - \operatorname { m e a n } _ { \mathcal { D } } ^ { \mathrm { c o n f } } \lVert v _ { t + 1 } - \ \end{array}$ $v _ { t } \| / ( m + \epsilon ) )$ (Eq. 6); and rigid $\begin{array} { r l } { } & { { } = \exp \bigl ( { - \frac { k _ { \mathrm { r o u g h } } } { 2 } \bigl ( \mathrm { m e a n } _ { \mathscr D _ { x } } \| \nabla _ { x } v _ { t } \| + \mathrm { m e a n } _ { \mathscr D _ { y } } \| \nabla _ { y } v _ { t } \| \bigr ) } \bigr ) } \end{array}$ with $k _ { \mathrm { r o u g h } } = 4 0 0$ (Eq. 7). The scene-flow magnitude m is the within-clip confidence-weighted mean over the top-20% velocity mask; it is used only inside the reward at training time and is not one of the reported evaluation metrics.

Per-axis z-normalization. Each axis $i \in \{ \mathrm { r e c o n } , \mathrm { m o t } , \mathrm { h p s v } 2 \}$ contributes $\tilde { z } [ R _ { i } ] = ( R _ { i } \mathrm { - m e a n } _ { \mathrm { b a t c h } } ( R _ { i } ) ) / \mathrm { s t d } _ { \mathrm { b a t c h } } ( R _ { i } )$ to the GRPO advantage, independently per minibatch. Per-axis z-norm decouples axis scales: e.g. HPSv2 sits at ∼0.20 raw while the motion conjunction sits at ∼0.80, but after z-norm both contribute on a common scale with weights 0.3 and 1.0.

## B Training Details and Hyperparameters

Table S1 lists the full training configuration.

Training details. We fine-tune a LoRA adapter $\scriptstyle ( r = \alpha = 2 5 6 )$ over the frozen base with AdamW $( \eta = 1 0 ^ { - 5 } )$ mixed-precision ${ \mathrm { b f 1 6 } } ,$ , 4-step distilled timesteps, group size G=24 per prompt, rolling window $L { = } 2 1$ frame-sink $S { = } 3 .$ , and NFT trust-region $\beta { = } 0 . 1$ . Group advantages over Eq. 8 are mapped to the NFT reward $\tilde { r } \in [ 0 , 1 ]$ exactly as in Sec. 3.2 (Eq. 9). We inherit Astrolabe’s stabilization recipe: an EMA-updated old policy $\theta _ { \mathrm { o l d } }$ for the negative reference, a selective KL penalty (He et al., 2025), and a conditional referencepolicy reset on KL drift. At training time, StreamVGGT (Zhuo et al., 2026) predicts cameras on a 26-frame linearly-subsampled version of each candidate rollout at $2 9 4 \times 5 1 8 ;$ the LPIPS-AlexNet comparison of Eq. 3 runs between the MoVieS re-render and the input video at $4 8 0 \times 8 3 2$ ; HPSv2 (Wu et al., 2023) runs on the generated frames directly.

Compute. The reward stack is a single StreamVGGT+MoVieS forward per candidate rollout (26 frames at $2 9 4 \times 5 1 8 )$ , measured at ≈7 s/rollout; because that one forward is shared between the reconstruction and motion axes via a per-batch cache, the motion-quality conjunction adds negligible marginal cost over a reconstruction-only reward. Each backbone trains for 150 RL steps at effective batch 384 (16 prompts × G=24 rollouts), i.e. ≈57,600 scored rollouts. End-to-end LoRA fine-tuning to the reported ckpt-150 costs ≈690 / 770 / 635 GPU-hours on Self-Forcing / Causal-Forcing / LongLive (≈20Rollout sampling dominates<sup>˙</sup> per-step wall-clock. This is a deliberately lightweight recipe: a LoRA adapter over a frozen 4-step distilled base, one shared reconstruction forward per candidate, and no per-backbone tuning beyond the three rewardaxis weights.

Table S1: Stream4D training hyperparameters. A single training-hyperparameter set is used across Self-Forcing, Causal-Forcing, and LongLive; only the three reward-axis weights vary per backbone.
<table><tr><td>Module</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone</td><td>Architecture Video resolution (H × W)</td><td>Causal Wan 2.1 (Self-Forcing / Causal-Forcing / LongLive) 480 × 832</td></tr><tr><td>LoRA fine-tuning</td><td>Rank r / scaling α Dropout Gradient checkpointing Hardware</td><td>256 / 256 0.0 True 4 nodes × 8×H200, DDP</td></tr><tr><td>Optimization</td><td>Precision Optimizer Learning rate / weight decay Max gradient norm Distillation timesteps T</td><td>bf16 AdamW (β1=0.9, β2=0.999) 10−5 / 10−4 1.0</td></tr><tr><td>NFT-GRPO</td><td>Trust-region β Per-axis z-normalization Dynamic-phase schedule Group size G per prompt</td><td>4, sampled from {1000, 750, 500, 250} 0.1 enabled disabled</td></tr><tr><td>Rollout</td><td>Rolling window L / frame-sink S Window selection Camera estimator</td><td>24 21 /3 Random StreamVGGT (Zhuo et al., 2026)</td></tr><tr><td>Reward backbone</td><td>4D Gaussian-Splatting reconstructor Recon comparison Perceptual anchor</td><td>MoVieS (Lin et al., 2026) (26-frame subsample, 294 × 518) LPIPS-AlexNet between re-render and input HPSv2 (Wu et al., 2023) on input rollout frames { Rrecon, Rmot, Rhpsv2} (Eq. 8)</td></tr><tr><td>Reward stack</td><td>Axes Per-axis weights (wrecon, wmot, whpsv2) Gauss-gate  $( m _ { \mathrm { n a t } } , \sigma )$  Rigidity constant  $k _ { \mathrm { r o u g h } }$ </td><td>SF (1.0, 1.0, 0.3); CF (1.0, 0.5, 0.6); LL (0.8, 1.0, 0.6) (0.020, 0.010) 400</td></tr></table>

## C Algorithm

Algorithm 1 gives the full Stream4D training loop with the deployed three-axis reward (Eq. 8). The reward backbone is a single MoVieS forward per candidate rollout, shared between the reconstruction and motionquality axes via a per-batch reward cache.

## D Evaluation Protocol Details

Evaluation sets. The main-paper results (Tables 1, 3, and 4) use a 500-prompt motion-prominent subset of VidProM (Wang and Yang, 2024), disjoint from the training prompts. We choose VidProM rather than the World-R1 evaluation set because VidProM’s prompts include substantial moving-subject content (people walking, vehicles, non-rigid foliage) where our reward is designed to operate; the World-R1 set is dominated by camera-tour-through-static-scenes prompts that bias toward rigid-scene methods. Model versions. All learned components are public releases used unmodified, with identical weights at training and evaluation time: MoVieS (Lin et al., 2026) official release checkpoint (movies ckpt.safetensors);

Algorithm 1 Stream4D: Streaming Forward-Process RL with a 4D-Consistency Reward   
Require: Distilled AR policy $\pi _ { \theta } .$ , behavior $\pi _ { \theta _ { \mathrm { o l d } } } .$ , KL reference $\pi _ { \theta _ { \mathrm { r e f } } } ;$ camera estimator (StreamVGGT); 4D-  
GS reconstructor (MoVieS); prompts $\mathcal { D } ;$ rolling window $L ,$ frame sink S, total windows $N ;$ reward   
weights $( w _ { \mathrm { r e c o n } } ,$ w<sub>mot</sub>, w<sub>hpsv2</sub>)   
Ensure: Optimized policy π<sub>θ</sub>   
1: Initialize $\theta _ { \mathrm { o l d } }  \theta , \theta _ { \mathrm { r e f } }  \theta ,$ risk buffer $B  \emptyset , \rho  \rho _ { 0 }$   
2: for each epoch k do   
3: Sample prompts $\{ c _ { i } \} _ { i = 1 } ^ { B } \sim \mathcal { D }$   
4: // Phase 1: Full N-window streaming rollout, G candidates each   
5: for $n = 1$ to N do   
6: Form context ${ \mathcal { C } } _ { n }$ from S-frame sink and L rolling frames   
7: Decode G candidates $\{ W _ { n } ^ { ( i , j ) } \} _ { j = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \mathcal { C } _ { n } , c _ { i } )$ (shared prefix)   
8: Update KV cache with chosen $\dot { W } _ { n }$ (one j per rollout)   
9: end for   
10: // Phase 2: 4D-consistency reward over the rollout   
11: for each rollout $( i , j )$ do   
12: $C _ { 1 : T } \gets \mathrm { S t r e a m V G G T } ( W _ { 1 : N } ^ { ( i , j ) } )$ //per-frame cameras, 26-frame subsample   
13: $( G _ { \mathrm { c a n } } , A _ { 1 : T } , \Delta _ { 1 : T } , P , \mathrm { c o n f } ) \gets \mathrm { M o V i e S } ( W _ { 1 : N } ^ { ( i , j ) } , C _ { 1 : T } )$ // canonical 4D-GS + sceneflow   
14: Re-render $\tilde { W } _ { 1 : T } ^ { ( i , j ) }$ from $G _ { \mathrm { c a n } } \oplus A _ { t } \oplus \Delta _ { t } \mathrm { a t } C _ { t }$   
15: $R _ { \mathrm { r e c o n } } ^ { ( i , j ) }  \mathrm { L P I P S }$ score on $( \tilde { W } , W )$ // Eq. 3   
16: $R _ { \mathrm { m o t . } } ^ { ( i , j ) }  g ( m )$ · smooth · rigid from (P, conf) // Eq. 4   
17: $R ^ { ( i , j ) } \gets w _ { \mathrm { r e c o n } } \tilde { z } [ R _ { \mathrm { r e c o n } } ] + w _ { \mathrm { m o t } } \tilde { z } [ R _ { \mathrm { m o t } } ] + w _ { \mathrm { h p s v } 2 } \tilde { z } [ R _ { \mathrm { h p s v } 2 } ]$ // Eq. 8   
18: end for   
19: Compute group advantages $\begin{array} { r } { A ^ { ( i , j ) } \gets R ^ { ( i , j ) } - \frac { 1 } { G } \sum _ { h } R ^ { ( i , h ) } } \end{array}$   
20: // Reward uncertainty via rank disagreement (selective KL)   
21: $\mathcal { M } ^ { ( i , j ) }$ ← high-uncertainty mask from rank disagreement (He et al., 2025)   
22: // Phase 3: Window-local NFT update with rollout-level advantage   
23: for each training window $n \in \{ 1 , \overline { { \ldots } } , N \}$ do   
24: Re-roll forward to step n; detach KV cache of $W _ { < n }$ as constant   
25: for each mini-batch over $( W _ { n } ^ { ( i , j ) } , A ^ { ( i , j ) } , \mathcal { M } ^ { ( i , j ) } )$ do   
26: Sample $t \sim \mathcal { U } ( \mathcal { T } _ { \mathrm { d i s t i l } } ) , ~ W _ { n } ^ { t } \gets ( 1 - t ) W _ { n } + t \epsilon , ~ v _ { \mathrm { t a r g e t } } \gets \epsilon - W _ { n }$   
27: $v _ { \theta } , v _ { \theta _ { \mathrm { o l d } } } , v _ { \theta _ { \mathrm { r e f } } }  \pi . ( W _ { n } ^ { t } , t , \mathcal { C } _ { n } , c )$   
28: $v ^ { + } \gets \beta v _ { \theta } + ( 1 - \beta ) v _ { \theta _ { \mathrm { o l d } } } , ~ v ^ { - } \gets ( 1 + \beta ) v _ { \theta _ { \mathrm { o l d } } } - \beta v _ { \theta }$   
29: $\tilde { r } ^ { ( i , j ) } \gets \mathrm { c l i p } ( \dot { A } ^ { ( i , j ) } / A _ { \operatorname* { m a x } } ^ { \cdots } , - 1 , 1 ) / 2 + 1 / 2$   
30: $\mathcal { L } _ { \mathrm { p o l i c y } }  \tilde { r } \Vert \dot { v } ^ { + } - \dot { v } _ { \mathrm { t a r g e t } } \Vert ^ { 2 } + ( 1 - \tilde { r } ) \Vert v ^ { - } - v _ { \mathrm { t a r g e t } } \Vert ^ { 2 }$   
31: $\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } }  \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { ( i , j ) : \mathcal { M } ^ { ( i , j ) } = 1 } \Vert v _ { \theta } ^ { ( i , j ) } - v _ { \theta _ { \mathrm { r e f } } } ^ { ( i , j ) } \Vert ^ { 2 } } \end{array}$   
32: $\theta \gets \theta - \eta \dot { \nabla } _ { \theta } ( \mathcal { L } _ { \mathrm { p o l i c y } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } )$   
33: end for   
34: end for   
35: if $\mathcal { L } _ { \mathrm { K L } } > \tau _ { \mathrm { K L } }$ or $k - k _ { \mathrm { l a s t } } > K _ { \mathrm { m a x } }$ then   
36: $\theta _ { \mathrm { r e f } }  \theta , \ k _ { \mathrm { l a s t } }  k$   
37: end if   
38: $\theta _ { \mathrm { o l d } }  \gamma \theta _ { \mathrm { o l d } } + ( 1 - \gamma ) \theta$   
39: end for

StreamVGGT (Zhuo et al., 2026) public checkpoint (HuggingFace lch01/StreamVGGT); 4DGT (Xu et al., 2025) released full model (4dgt full.pth, level-of-detail config tlod-l3); VideoReward (Liu et al., 2025b) released Qwen2-VL-7B checkpoint; HPSv2 (Wu et al., 2023) checkpoint HPS v2.1 compressed.pt (v2.1). The LLM judge’s model string and run dates are given in Appendix E.

4D-PSNR / SSIM / LPIPS. We re-run MoVieS 4D Gaussian-Splatting on each generated rollout at evaluation time, using StreamVGGT for camera estimation followed by MoVieS reconstruction over 26 frames at a resolution of $2 9 4 \times 5 1 8$ . We then re-render the reconstructed 4D scene from the estimated camera corresponding to each input frame and compare the resulting rendering $\hat { \mathbf { I } } _ { t }$ against the original generated frame $\mathbf { I } _ { t } .$ We report standard image-space PSNR, SSIM, and LPIPS-AlexNet for each frame and average the resulting scores over the full rollout.

For PSNR, we first compute the mean-squared error over all RGB pixels:

$$
\mathrm { M S E } _ { t } = \frac { 1 } { 3 H W } \left\| \mathbf { I } _ { t } - \hat { \mathbf { I } } _ { t } \right\| _ { 2 } ^ { 2 } .\tag{10}
$$

Assuming RGB values are normalized to [0, 1], the per-frame PSNR is

$$
\mathrm { P S N R } _ { t } = 1 0 \log _ { 1 0 } \left( \frac { 1 } { \mathrm { M S E } _ { t } } \right) .\tag{11}
$$

We report 4D-PSNR by averaging the per-frame PSNR values:

$$
4 \mathrm { D - P S N R } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { P S N R } _ { t } .\tag{12}
$$

Thus, $\mathrm { \Sigma ^ { \bullet \bullet } 4 D \mathrm { - P S N R } ^ { \bullet \bullet } }$ does not denote a new PSNR formulation; rather, it refers to standard PSNR computed after fitting and re-rendering the rollout with the dynamic 4D Gaussian-Splat representation.

SSIM measures local structural similarity between $\mathbf { I } _ { t }$ and $\hat { \mathbf { I } } _ { t }$ . For corresponding local image windows x and $y ,$ SSIM is computed as

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{13}
$$

where $\mu _ { x }$ and $\mu _ { y }$ denote the local means, $\sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ denote the local variances, $\sigma _ { x y }$ denotes the local covariance, and $C _ { 1 }$ and $C _ { 2 }$ are numerical-stability constants. We spatially average the local SSIM values to obtain a per-frame score and then average across frames:

$$
4 \mathrm { D - S S I M } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { S S I M } \left( \mathbf { I } _ { t } , \hat { \mathbf { I } } _ { t } \right) .\tag{14}
$$

Higher PSNR and SSIM indicate better reconstruction fidelity.

For LPIPS, we use the AlexNet-based perceptual distance. Each image pair is passed through a pretrained AlexNet feature extractor, and normalized deep features from multiple layers are compared using the learned LPIPS channel weights. Denoting the normalized feature representation at layer l by $\tilde { \phi } _ { l } ( \cdot )$ and its learned channel-wise weighting by $\mathbf { w } _ { l } .$ , the per-frame LPIPS distance can be written as

$$
\mathrm { L P I P S } _ { t } = \sum _ { l } \frac { 1 } { H _ { l } W _ { l } } \sum _ { h , w } \left\| \mathbf { w } _ { l } \odot \left( \tilde { \phi } _ { l } ( \mathbf { I } _ { t } ) _ { h , w } - \tilde { \phi } _ { l } ( \hat { \mathbf { I } } _ { t } ) _ { h , w } \right) \right\| _ { 2 } ^ { 2 } ,\tag{15}
$$

where $H _ { l }$ and $W _ { l }$ denote the spatial dimensions of the feature map at layer l. We report the average LPIPS distance across frames:

$$
\mathrm { 4 D - L P I P S } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { L P I P S } _ { t } .\tag{16}
$$

Unlike PSNR and SSIM, lower LPIPS indicates better perceptual agreement.

This evaluation uses the same reconstruction backbone as the training reward, so it directly measures the quantity optimized by the policy: how faithfully a generated rollout can be explained by a temporally dynamic 3D representation. We refer to it as “4D” reconstruction because MoVieS uses time-varying perframe Gaussian attributes and scene flow, yielding a representation over both space and time. On the same generated rollouts, 4D reconstruction achieves approximately 5–7 dB higher PSNR than its static-3DGS counterpart, since moving content is in-distribution for the dynamic reconstructor.

What 4DGT does and does not control for. We call this metric reconstructor-independent, not fully independent, and state its scope precisely. It swaps the reconstruction model with 4DGT, so a gain that shows up under both is not an artifact of MoVieS’ particular inductive biases. Yet, they both obtained cameras from the same StreamVGGT estimator.

VideoReward. We use the Qwen2-VL-7B VideoReward (Liu et al., 2025b) model, which scores video along three heads: Visual Quality (VQ, appearance and visual fidelity), Motion Quality (MQ, motion naturalness and temporal consistency), and Text Alignment (TA, adherence to the prompt), plus an Overall score. In the main tables we report the per-prompt paired win-rate against the same backbone’s distilled base: the fraction of prompts on which the method’s video scores higher than the base’s under each head.

Vision-LLM judge. The two LLM-judge columns in the main tables come from a reward-blind vision-LLM (Gemini-3.5-Flash). Motion is a three-way verdict on object/subject motion (preserved= 1 / reduced= <sup>1</sup><sub>2</sub> / mostly lost= 0), averaged over prompts to a score in [0, 1]. Consistency win% is a head-to-head verdict (win= 1 / tie= <sup>1</sup><sub>2</sub> / loss= 0 vs. base) requiring both kept motion and better consistency, averaged and centered at 50 %; a frozen rollout is always a loss, so consistency bought by suppressing motion earns no credit. All reported numbers are position-debiased by averaging the two presentation orders. Full judge prompts, the debiasing protocol, and reproducibility checks are given in Appendix E.

Baseline reproductions. World-R1 and VideoGPA target bidirectional T2V models, so no published checkpoints exist for the distilled-AR setting; both baselines are our reproductions on the same bases, trainer, and prompt data as our method. Reproduction configs are released with the code.

## E LLM-Judge Details

This appendix documents the vision-LLM judge behind the Motion and Consistency win% columns of Tables 1, 3, and 4: the exact prompts, the decoding configuration, the position-debiasing protocol, and reproducibility checks.

Setup. The judge is Gemini-3.5-Flash, called via the REST API with temperature 0 and a constrained JSON response schema with a three-way verdict enum plus a one-line free-text reason. Each call attaches two videos inline as mp4, sampled at 5 fps. The judge never sees the reward, the scene-flow metric, the method names, or which video is ours; the prompts label the two clips only as “BASE (reference)” and “EVALUATED.”

Motion judge prompt. The three-way motion verdict uses the following prompt, verbatim; {prompt} is the text-to-video prompt and {base label} / {eval label} are filled with “Video 1” / “Video 2” according to the current presentation order. The verdict labels are neutral comparisons rather than loaded terms, so the judge is not primed toward a conclusion.

You are judging two AI-generated videos made from the SAME text prompt.   
Text prompt:   
"""{prompt}"""   
Two videos are attached: Video 1 (first) and Video 2 (second).

- The BASE (reference) video is {base\_label}.   
- The video being EVALUATED is {eval\_label}.   
Compare the amount of MOTION in the EVALUATED video ({eval\_label}) against the   
BASE video ({base\_label}). "Motion" means genuine object/subject motion (people   
walking, cars driving, animals moving, crowds, explosions...) - NOT camera   
pans/zooms over a frozen scene, and NOT flickering/texture noise.   
Answer with exactly one verdict (about the EVALUATED video relative to the BASE):   
"motion\_preserved" = the EVALUATED video shows a similar amount of (or more)   
object motion than the BASE   
"motion\_reduced" = the EVALUATED video shows noticeably less object motion   
than the BASE, but subjects still clearly move   
"motion\_mostly\_lost" = the EVALUATED video is near-static or frozen while the   
BASE clearly moves - most of the object motion is gone   
Judge motion only - ignore visual quality/aesthetics differences.

The reported motion score is (#preserved + <sup>1</sup> #reduced)/N ∈ [0, 1].

Joint motion+consistency (Consistency) judge prompt. The head-to-head verdict folds motion retention and consistency into a single call; the key design choice is that a frozen evaluated video is a loss even if it looks perfectly stable, which removes the static bias that a pure consistency rating would leave behind:

You are comparing two AI-generated videos made from the SAME text prompt.   
Text prompt:   
"""{prompt}"""   
Two videos are attached: Video 1 (first) and Video 2 (second).   
- The BASE model’s output (the reference) is {base\_label}.   
- The video being EVALUATED is {eval\_label}.   
Decide whether the EVALUATED video ({eval\_label}) is BETTER than the BASE   
({base\_label}). The EVALUATED video is better ONLY if it does BOTH of these,   
relative to the BASE:   
(A) KEEPS THE MOTION - it preserves roughly as much genuine object/subject motion   
as the BASE (people, animals, vehicles, crowds, effects). "Motion" is real   
object/subject movement, NOT camera pans over a frozen scene and NOT texture   
flicker.   
(B) IS MORE CONSISTENT THAN THE BASE - on that moving content, objects hold their   
identity, shape, and structure BETTER than in the BASE: less morphing, warping,   
popping in/out, dissolving, or flicker than the BASE shows.   
Being more consistent only counts if the motion is still there. A video that looks   
cleaner only because it stopped moving has NOT kept the motion, so it is NOT better -   
it is worse.   
Answer with exactly one verdict (about the EVALUATED video relative to the BASE):   
"win" = the EVALUATED video is BETTER than the BASE: it keeps about as much   
(or more) object motion AND is clearly more consistent than the BASE   
on that motion.   
"partial" = roughly a TIE: the EVALUATED video keeps the motion but is about as   
consistent as the BASE (no clear improvement either way) - neither   
clearly better nor clearly worse.   
"loss" = the EVALUATED video is WORSE than the BASE: its motion is far behind

the BASE (largely frozen / near-static), OR it is less consistent than the BASE (more morphing, identity changes, structural breakdown). Judge motion and consistency only - ignore overall visual quality / aesthetics and prompt wording. Consistency is a COMPARISON against the BASE, not an absolute; a frozen or near-static EVALUATED video is always WORSE because it did not keep the motion.

The reported win-rate is $( \# \mathrm { w i n } + \textstyle \frac { 1 } { 2 } \# \mathrm { p a r t i a l } ) / N$ , centered at $5 0 \% ; > 5 0 \%$ means the method beats its own base.

Position debiasing. Both prompts refer to the videos through the role labels (BASE / EVALUATED) rather than the attachment slots, so the same question can be asked with the two videos in either physical order. Every pair is judged twice: once with the base attached first and once with the variant attached first, with {base label} / {eval label} swapped accordingly, so both verdicts mean “the variant relative to the base.” Each of the two verdicts is mapped to the three-point numeric scale (Motion: mostly-lost= 0, reduced= <sup>1</sup> , preserved= 1; Consistency: loss= 0, tie= <sup>1</sup> , win= 1) and the pair’s reported score is the average of its two order scores, so a video that wins in one order and ties in the other is credited 0.75 rather than being forced to one discrete verdict. Averaging the two symmetric presentations cancels the additive position bias exactly. Reported column values are the mean of these per-pair averages over the prompt set. Reproducibility. The contested SF main-table pair (ours vs. World-R1) was run as three independent debiased passes; preserved% agrees within 1 pt across passes (ours 61.2–61.4, World-R1 63.4–64.4), and we report the final pass. Independent duplicate passes on two deployed rows differ by $\leq 0 . 0 0 5$ motion score and $\leq 1 . 2$ pp win-rate. Per-pair verdicts with per-order records and one-line reasons are archived as JSONL alongside the aggregates.

## F Human Study Protocol

Design. A two-alternative forced-choice (2AFC) study on the LongLive backbone, which we prioritize because its 10.3 s horizon is where accumulated drift and the static-collapse shortcut matter most. We sample 50 prompts from the high-motion held-out subset and form three pairings per prompt and compare Stream4D against the distilled base, the World-R1 reproduction, and VideoGPA. Each comparison is judged exactly once; the 150 comparisons are partitioned across the 5 raters. For each pair the rater answers two questions: (1) which video shows more natural motion: enough real movement, not frozen; smooth, not jittery, and (2) which video is better overall: jointly weighing motion amount, motion quality, and object consistency.

Quality control. Each rater additionally sees two interleaved attention-check trials with an obviously frozen clip vs. a moving one; a rater is excluded if either check is failed. All 5 raters passed both checks, so all 5 are retained and none are excluded; the reported numbers therefore use the full 50 judgments per comparison.

## G Random-Subset Robustness Check

The main evaluation (Table 1) uses a motion-prominent filtered subset, chosen because it is the regime where the static-collapse shortcut matters. Table S2 repeats the comparison on a uniformly random 500-prompt subset of Vidprom, using the identical protocol. Because the pool is dominated by lower-motion prompts, the frozen VideoGPA and World-R1 sweep reconstruction and even VideoReward, yet under the motionaware judge they lose to their own bases. Stream4D is the only method that beats its base under the joint verdict on all three backbones, and its judge scores are nearly unchanged from the motion-prominent subset (SF motion 0.816 vs 0.833; LL Consistency 68.5 vs 74.2).

Table S2: Robustness check: main comparison on a uniformly random 500-prompt subset of the same held-out VidProM set (seed-0 sample, no motion filter; 122 prompts overlap the high-motion subset of Table 1). Metrics and conventions as in Table 1 (4DGT: reconstructor-independent dynamic-reconstruction cross-check, Appendix D); all RL’d rows at ckpt-150, judged against the same backbone’s base. <sup>‡</sup>The VideoGPA row uses the standard VideoGPA recipe (the best-weight variant of Table 1 was generated only on the high-motion subset); on this unfiltered, low-motion-dominated subset its clean freeze sweeps both reconstructors. <sup>†</sup>marks methods whose Consistency win-rate falls below the 50 % break-even against their own base: on this low-motion-dominated subset, their reconstruction and VideoReward advantages are bought by suppressing motion (motion scores 0.14–0.48), not by generating better video. Bold = best RL’d row per backbone block; shaded = ours.
<table><tr><td rowspan="2">Method</td><td colspan="2">MoVieS Recon.</td><td colspan="2">4DGT Recon.</td><td colspan="2">LLM judge</td><td colspan="3">VideoReward win%</td></tr><tr><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td>Motion↑</td><td>Consist.↑</td><td>VQ↑ MQ↑</td><td></td><td>TA↑ Ovr↑</td></tr><tr><td colspan="9">Distilled AR backbone: Self-Forcing (Huang et al., 2025) (5 s)</td></tr><tr><td>Base (B0)</td><td>18.19 0.831</td><td>0.239</td><td>16.23 0.546</td><td>0.439</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>19.35 0.885</td><td>0.182</td><td>16.60 0.560</td><td>0.412</td><td>0.773</td><td>68.2</td><td>61.4</td><td>49.4 54.2</td><td>56.8</td></tr><tr><td>VideoGPA (Du et al., 2026)‡†</td><td>24.88 0.908</td><td>0.143</td><td>19.87 0.621</td><td>0.396</td><td>0.477</td><td>38.3</td><td>63.0</td><td>63.4 65.0</td><td>67.4</td></tr><tr><td>+ Stream4D (Ours)</td><td>21.23 0.895</td><td>0.177 17.70</td><td>0.657</td><td>0.411</td><td>0.816</td><td>75.1</td><td>59.8</td><td>56.6 54.4</td><td>59.8</td></tr><tr><td colspan="10">Distilled AR backbone: Causal-Forcing (Zhu et al., 2026) (5 s)</td></tr><tr><td>Base (B0)</td><td>16.30 0.802</td><td>0.252 14.94</td><td>0.512</td><td>0.455</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)</td><td>19.67 0.894</td><td>0.162</td><td>16.34 0.575</td><td>0.417</td><td>0.637</td><td>61.1</td><td>72.6</td><td>60.6 57.0</td><td>65.8</td></tr><tr><td>VideoGPA (Du et al., 2026)‡†</td><td>22.94 0.843</td><td>0.147</td><td>19.33 0.605</td><td>0.377</td><td>0.484</td><td>36.5</td><td>67.6</td><td>54.8</td><td>53.4 62.2</td></tr><tr><td>+ Stream4D (Ours)</td><td>20.99 0.890</td><td>0.156</td><td>17.49 0.650</td><td>0.404</td><td>0.716</td><td>64.9</td><td>71.4</td><td>65.8</td><td>52.8 68.6</td></tr><tr><td colspan="10">Distilled AR backbone: LongLive (Yang et al., 2025) (10.3 s)</td></tr><tr><td>Base (B0)</td><td>18.11 0.863</td><td>0.217</td><td>15.45 0.509</td><td>0.447</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-R1 (Wang et al., 2026a)†</td><td>23.01 0.940</td><td>0.137</td><td>17.53</td><td>0.630 0.438</td><td>0.457</td><td>46.8</td><td>74.8</td><td>68.6 63.4</td><td>75.8</td></tr><tr><td>VideoGPA (Du et al., 2026)‡†</td><td>22.99 0.943</td><td>0.130</td><td>19.24</td><td>0.668 0.339</td><td>0.136</td><td>9.8</td><td>66.0</td><td>63.6</td><td>54.6 67.0</td></tr><tr><td>+ Stream4D (Ours)</td><td>24.60 0.910</td><td>0.147</td><td>19.59</td><td>0.646 0.417</td><td>0.701</td><td>68.5</td><td>74.8</td><td>76.6</td><td>65.4 78.2</td></tr></table>

On this unfiltered subset thefrozen specialists sweep the reconstruction metrics under both reconstructors—SF VideoGPA<sup>‡</sup> takes the best MoVieS and 4DGT PSNR (and the best VideoReward-Overall)—because a cleanfreeze ofa near-static scene reconstructs well under any reconstructor. The judge is what exposes the shortcut: VideoGPA’s motion score is 0.477 and it loses to its own base on Consistency (38.3 %), as do CF VideoGPA<sup>‡</sup> (36.5 %), LL VideoGPA<sup>‡</sup> (9.8 %), and LL World-R1 (46.8 %). Stream4D is the only method that beats its base under the joint verdict on all three backbones. Joint verdicts cover 499/500 prompts per row (one APIfailure each).

## H Additional Qualitative Results

Figure S1 shows motion-preservation filmstrips for one prompt per backbone (uniformly-spaced frames). Figures S2–S4 show the complementary failure, identical timestamps for every method: rollouts that move but deform, rollouts that buy stability by freezing, and Stream4D avoiding both.

## I Limitations

(i) The judge metrics come from a single vision-LLM; they are corroborated by VideoReward, the 4D metrics. (ii) 4DGT score rules out dependence on MoVieS’ training bias, but they still share the VGGT component.

![](images/eba718f2347a7e1f4d61d49cee0d5c5f2e6cad0eb4745f115d00f87fb1f7c215.jpg)

![](images/beecabe7646e1b5a030d37c860930db90c9150e2d01d87146fc00a1092b3e90b.jpg)  
Figure S2: Object-identity under motion: the base breaks the object apart, the static-reward baselines freeze the scene, and only ours keeps a moving object intact.

![](images/fb8d578acc45f8abf4ac93a0164f494fc44e2165d23c05db7a41acd4ecfa852b.jpg)  
Figure S3: Consistency companion on Self-Forcing: an ancient-battlefield beast Red outlines mark failure frames and the coloured badge names each row’s behaviour. The failures are complementary: the base moves but morphs; the static-reward baselines are either frozen. Stream4D stays coherent: one beast, one identity, approaching the camera.

![](images/f06063eb5b319c43bf9355c6483ab1c887d6439dc86e208dd2dcb5d1d445dbc3.jpg)  
Figure S4: Consistency companion on LongLive: a planet descending over a city The base deforms into spreading debris/fire; VideoGPA’s explosion flickers in and out over an otherwise static scene; World-R1 is frozen. Stream4D keeps a coherent descending planet over a stable city.