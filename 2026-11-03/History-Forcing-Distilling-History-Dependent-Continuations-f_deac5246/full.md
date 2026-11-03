# History Forcing: Distilling History-Dependent Continuations for Long-Horizon Video World Models

Hangyi Kuang<sup>1,\*</sup> Chengzhi Wang<sup>1,\*</sup> Yefei He<sup>1</sup> Weijie Wang<sup>1</sup> Yuke Zhao<sup>1,2</sup> Zhuokun Chen<sup>3</sup> Zisu Huang<sup>2</sup> Yifan Yang<sup>2</sup> Xiaoxuan He<sup>1,†</sup> Feng Chen<sup>4,†</sup> Bohan Zhuang<sup>1</sup> <sup>1</sup> Zhejiang University <sup>2</sup> Microsoft Research <sup>3</sup> Monash University <sup>4</sup> Adelaide University

![](images/2974099881e7f8abda3e5d3a52eb96cd8c0a06f55889ab87a6423784e45f525d.jpg)  
Figure 1 History Forcing for long-horizon video world models. Left: Long-video generation. Our method maintains appearance consistency and scene continuity, while the baseline exhibits appearance drift and changes in scene layout. Right: Out-of-sight world modeling. Our method captures plausible state evolution and maintains object permanence upon revisit, while the baseline exhibits inconsistent scene changes and object disappearance.

Abstract. Few-step autoregressive video difusion enables eficient generation, yet maintaining consistency over long rollouts remains challenging, especially when subjects or scenes reappear after going out of sight. Despite progress in fidelity and rollout robustness, existing distillation methods do not fully exploit richer historical information to constrain the student’s current prediction. To address this gap, we introduce History Forcing, a unified on-policy distillation framework that strengthens historical supervision for few-step autoregressive video generators. The framework uses the residual between teacher predictions with and without informative historical context to decouple this supervision. Such a historical context can be obtained using earlier generated chunks with less accumulated drift or a fixed-view teacher rollout that keeps the scene visible while the student looks away. The residual is then weighted and added to the score or velocity prediction used for distillation, forming a corrected target. By aligning the student with this corrected target through distillation, History Forcing incorporates long-range historical constraints into autoregressive generation. Our method achieves state-of-the-art performance among 1.3B models, with a VBench quality score of 85.07 for 15s video generation and a VBench-Long aggregate score of 79.82 for 60s generation. For out-of-sight world modeling, evaluations further demonstrate more plausible scene evolution and more consistent states when the camera returns, with Evolution Quality improving from 2.35 to 2.70 on PlayWorld and Object Consistency from 58.43 to 62.18 on MemoBench.

Project Page: https://ziplab.github.io/History-Forcing-Project-Page/ Correspondence: xiaoxuanhe@zju.edu.cn chenfeng1271@gmail.com Code: https://github.com/ziplab/History-Forcing Keywords: Video World Models, On-Policy Distillation, Long Video Generation

![](images/912685e9b3fde4accff693dd4faf681618161390a5152335fa260b3aa3b3e019.jpg)  
Figure 2 Comparison of distillation strategies. (a) DMD uses a bidirectional teacher’s prediction as the reference for the current chunk. (b) AR distillation directly uses the history-conditioned prediction of an AR teacher as the full target. (c) History Forcing uses the context residual between the AR teacher’s predictions with and without history to correct the reference prediction.

## 1 Introduction

Few-step autoregressive (AR) video difusion models have enabled eficient long-horizon video generation [39, 57, 63]. However, maintaining consistency over extended autoregressive rollouts remains challenging. Coherent continuation often relies on earlier observations of persistent subjects, scenes, and dynamics, yet such information may be insuficiently incorporated into later predictions, leading to gradual subject and scene drift. This reliance on long-range history becomes even more pronounced in video world models. When an entity leaves the field of view, it may reappear only after a long interval, by which point the recent frames carry no information about it [16, 54]. These challenges make efective use of long-range history critical for consistent autoregressive generation.

Existing training paradigms have progressively increased the role of historical information in autoregressive supervision. Teacher forcing conditions predictions on ground-truth prefixes [18, 31], whereas Self Forcing applies distribution matching distillation (DMD) to student-generated rollouts using a bidirectional difusion teacher [28]. Subsequent methods further improve long-horizon training by extending supervision to longer generated sequences, preserving information from earlier frames, or propagating learning signals from later predictions to earlier generated representations [39, 57, 73]. More recent approaches condition teacher predictions on longer contexts, causally matched prefixes, or privileged observations, allowing the distillation target to better reflect historical dependencies relevant to autoregressive generation [4, 11, 38].

Despite these advances, existing distillation targets do not provide an explicit mechanism to strengthen historical constraints independently of the teacher’s overall generative supervision. As illustrated in Fig. 2, DMD with a bidirectional difusion teacher provides strong supervision for video quality, but does not explicitly incorporate distant autoregressive history into teacher predictions. History-conditioned methods instead distill from an AR teacher with access to richer historical context [4, 11]. However, matching its full conditional distribution couples supervision for temporal consistency and visual quality. The AR teacher’s weaknesses in visual quality may therefore constrain the student [63]. We refer to this limitation as history–quality coupling.

To address this limitation, we introduce History Forcing, a unified on-policy distillation framework that isolates and strengthens historical supervision. We compare teacher predictions with and without informative historical context to compute a residual that captures the influence of history on the current prediction. This residual is then weighted and added to the score or velocity prediction used for distillation, guiding the student to generate continuations consistent with the historical observations.

We apply History Forcing to diferent tasks and develop corresponding history construction strategies. In long-video generation, we use earlier chunks of the student’s rollout to provide subject and scene information less afected by accumulated drift. The resulting residual corrects the DMD target while retaining the bidirectional teacher’s supervision for visual quality. For out-of-sight world modeling, we obtain history from a teacher rollout that keeps the camera fixed on the region to model its evolution while the region is outside the student’s view. This history is used to compute the residual that guides the student’s continuation upon revisit. As illustrated in Fig. 1, History Forcing better preserves subject appearance and scene structure over long rollouts and produces more consistent scene states upon revisit, demonstrating the benefits of strengthening historical supervision.

Our contributions are summarized as follows:

• We introduce History Forcing, a unified distillation framework that decouples historical supervision from generative supervision. By comparing teacher predictions with and without informative history, we compute a context residual that is weighted and added to the original distillation target, allowing historical constraints to be strengthened independently. This formulation can be readily incorporated into DMD and on-policy self-distillation and is compatible with existing long-video training strategies.

• We develop history construction strategies for long-video generation and out-of-sight world modeling, using earlier generated chunks to constrain temporal drift and fixed-camera teacher rollouts to supervise scene evolution.

• History Forcing achieves a VBench quality score of 85.07 for 15s generation and directly extrapolates to 60s generation with a VBench-Long aggregate score of 79.82, using student training rollouts of at most 15s. With only 1.3B teachers, it outperforms the compared 1.3B models, including those distilled from 14B teachers, and approaches the 5B LongLive-2.0 model on VBench-Long. For out-of-sight world modeling, it further improves scene evolution and revisit consistency on PlayWorld and MemoBench.

## 2 Related Work

Long-horizon autoregressive video difusion. Existing video difusion models provide strong visual and motion priors for short video synthesis [23, 33, 50, 58], while extending generation to long temporal horizons requires mitigating error accumulation and maintaining useful historical context. Training-free approaches extend pretrained models through diagonal denoising, temporal feature manipulation, and cache compression [32, 40, 60]. Autoregressive approaches instead improve the rollout process through extended self-rollouts, bounded contexts, rolling denoising, or error-aware training [8, 13, 20, 22, 25, 36, 39, 57, 59]. Context packing, learned memory representations, and history-aware supervision further improve access to earlier observations [5, 15, 43, 55, 65–67, 73]. History-Guided Video Difusion combines predictions under diferent historical conditions during sampling [45]. We instead design distillation targets that explicitly incorporate historical supervision into on-policy training.

Few-step causal video difusion distillation. Few-step causal video difusion builds on difusion and flow distillation to reduce autoregressive denoising costs [7, 10, 17, 21, 41, 42]. DMD and DMD2 establish distribution matching as an efective objective for few-step generation [61, 62]. CausVid uses DMD to distill a few-step AR generator from a bidirectional difusion teacher, while Self Forcing further performs distillation on student-generated rollouts to reduce exposure bias [28, 63]. Subsequent work improves causal initialization and teacher–student alignment [69, 70]: Causal Forcing uses an AR teacher for ODE initialization, while Context-Matched Distillation aligns teacher scoring with the causal context available to the student [4, 72]. Context Forcing incorporates longer history into an AR teacher’s conditional distribution, while OPSD-V uses real-video history to provide EMA velocity targets along student trajectories [11, 38]. These approaches improve sampling eficiency and teacher–student alignment, but the contribution of historical context remains implicit in the target.

History mechanisms in video world models. Historical conditioning is also central to video world models [52], where camera motion can separate a revisit from its last informative observation. Video generation provides a basis for interactive world simulation [2, 3, 6, 24, 34, 49, 71]. Real-time interactive systems combine controllable autoregressive generation with long-rollout training and scene memory [19, 48, 68]. WorldMem combines spatial and temporal cues for historical conditioning, while VMem uses geometry-indexed view retrieval to maintain cross-view consistency [35, 53, 64]. Memory Forcing trains models to balance temporal and spatial memory during exploration and revisits, while WorldPlay reconstitutes historical context and aligns teacher and student memory during distillation [27, 47]. LiveWorld uses historical references and fixed-view virtual monitors to preserve scene identity and continue unseen events, while ReMind trains models to recall reliable historical observations across occlusions and viewpoint changes [16, 54]. These methods improve scene consistency through memory mechanisms and history-aware training, but coherent continuation across long observation gaps remains challenging.

## 3 Preliminaries

Difusion models. Difusion models learn to reverse a Gaussian perturbation process. Given a clean latent x<sub>0</sub>, its noisy state at difusion time t is $x _ { t } = \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $\left( \alpha _ { t } , \sigma _ { t } \right)$ specifies the noise schedule. Under the linear interpolation commonly used in flow matching, the denoising network predicts the velocity $v _ { t } = \epsilon - x _ { 0 } \ [ 3 7 ]$ . A predicted velocity $\hat { v } _ { t }$ gives the clean estimate $\hat { x } _ { 0 } = x _ { t } - \sigma _ { t } \hat { v } _ { t }$ . Let $p _ { t }$ denote the distribution of $x _ { t }$ . Its score is $s ( x _ { t } , t ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } )$ and can be estimated from the denoising prediction [46, 62]:

![](images/13e05e1fbdbf5c0f8159a4a99a67ec5c7863923e82e987ccc7515b2014761940.jpg)  
Figure 3 Overview of History Forcing. An AR teacher predicts the same noisy student chunk with and without additional history. The resulting history contrast modifies the reference score in DMD and the velocity target in on-policy self-distillation.

$$
s ( x _ { t } , t ) \approx \frac { \alpha _ { t } \hat { x } _ { 0 } - x _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{1}
$$

Sampling starts from Gaussian noise and iteratively updates the noisy latent using the denoising predictions.

Autoregressive video difusion. A pretrained 3D variational autoencoder (VAE) encodes RGB video frames into a spatially and temporally compressed sequence of latent feature maps, referred to as latent frames [58]. The difusion model operates in this latent space, and the denoised latents are decoded into video frames using the VAE decoder [33]. Autoregressive video difusion models generate videos sequentially in chunks, each comprising consecutive latent frames [18]. Let $x _ { 0 } ^ { i }$ denote the i-th clean latent chunk in a sequence of N chunks. The video distribution is factorized as:

$$
p _ { \theta } ( x _ { 0 } ^ { 1 : N } ) = \prod _ { i = 1 } ^ { N } p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { 0 } ^ { < i } ) .\tag{2}
$$

At inference, each chunk is generated by iteratively denoising its latent frames, conditioned on previously generated chunks.

Distribution matching distillation. Distribution Matching Distillation (DMD) trains a few-step generator to match a reference distribution defined by a pretrained difusion model [61, 62]. Let q<sub>θ,t</sub> and $p _ { \mathrm { r e f } , t }$ denote the difused student and reference distributions. DMD minimizes $\mathbb { E } _ { t } [ D _ { \mathrm { K L } } ( q _ { \theta , t } \Vert p _ { \mathrm { r e f } , t } ) ]$ ]. Its distribution-matching gradient is approximated by

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D M D } } \approx \mathbb { E } \left[ w _ { t } \alpha _ { t } J _ { \boldsymbol { \theta } } ^ { \top } \left( s _ { \mathrm { f a k e } } - s _ { \mathrm { r e f } } \right) \right] ,\tag{3}
$$

where $J _ { \theta } ~ = ~ \partial x _ { 0 } ^ { i } / \partial \theta$ and $w _ { t }$ is the DMD weighting factor. Here, $s _ { \mathrm { r e f } } = \nabla _ { x _ { t } ^ { i } } \log p _ { \mathrm { r e f } , t } ( x _ { t } ^ { i } )$ and $s _ { \mathrm { f a k e } } ~ =$ $\nabla _ { x _ { t } ^ { i } } \log { q _ { \theta , t } ( x _ { t } ^ { i } ) }$ denote the reference and fake scores. The reference score is estimated by the pretrained difusion model, while the fake score is estimated by an online denoiser trained on student-generated samples. Both scores are obtained from denoising predictions using Eq. (1).

## 4 Method

History Forcing derives a history-dependent correction by comparing AR teacher predictions for the same noisy student sample under two contexts that difer only in additional history. As shown in Fig. 3, this correction is applied to the reference score in DMD for long-horizon consistency and to the teacher velocity target in on-policy self-distillation for out-of-sight evolution.

![](images/e666fa3618e537f7b6356d5ba33c56575112c90d2cc2cd68b4a7238a0cf4f6ec.jpg)  
Figure 4 Comparison of distillation targets, where + and − denote distributions with and without additional history. (a) DMD uses a reference close to the data distribution without history. (b) AR Distillation uses the history-conditioned AR distribution, with larger modeling error. (c) History Forcing transfers the history-induced shift from the AR teacher to the reference, producing a target closer to the data distribution with history.

## 4.1 History-Reweighted Distribution

For the current chunk $x _ { 0 } ^ { i }$ , let $h ^ { i }$ denote the additional historical observations used for supervision and c denote the shared condition, including the recent context and other inputs held fixed across the two AR teacher predictions. The AR difusion teacher predicts the same noisy chunk with and without $h ^ { i }$ , keeping c fixed. We denote the corresponding difused distributions by $p _ { \mathrm { A R } , t } ^ { + } = p _ { \mathrm { A R } , t } ( \cdot \mid h ^ { i } , c )$ and $p _ { \mathrm { A R } , t } ^ { - } = p _ { \mathrm { A R } , t } ( \cdot \mid c )$ with scores $s _ { \mathrm { A R } } ^ { + }$ and $s _ { \mathrm { A R } } ^ { - }$ , respectively. Let $p _ { \mathrm { r e f } , t }$ denote a reference distribution, such as that defined by a pretrained bidirectional difusion teacher, with score $s _ { \mathrm { r e f } }$ . Motivated by classifier-free guidance (CFG) [26, 45], we express the history-dependent correction in score form as:

$$
s _ { \mathrm { H F } } = s _ { \mathrm { r e f } } + \lambda \left( s _ { \mathrm { A R } } ^ { + } - s _ { \mathrm { A R } } ^ { - } \right) ,\tag{4}
$$

where $\lambda \geq 0$ controls the strength of historical supervision. All scores are estimated at the same noisy chunk and difusion time from the corresponding denoising predictions using Eq. (1). Using the definition of the score, Eq. (4) can be written as:

$$
\begin{array} { r l } & { s _ { \mathrm { H F } } = \nabla _ { x _ { t } ^ { i } } \log p _ { \mathrm { H F } , t } ( x _ { t } ^ { i } ) } \\ & { \qquad = \nabla _ { x _ { t } ^ { i } } \log \left[ p _ { \mathrm { r e f } , t } ( x _ { t } ^ { i } ) \left( \frac { p _ { \mathrm { A R } , t } ^ { + } ( x _ { t } ^ { i } ) } { p _ { \mathrm { A R } , t } ^ { - } ( x _ { t } ^ { i } ) } \right) ^ { \lambda } \right] . } \end{array}\tag{5}
$$

This gives the normalized target distribution:

$$
p _ { \mathrm { H F } , t } ( x _ { t } ^ { i } ) = \frac { 1 } { Z _ { t } } p _ { \mathrm { r e f } , t } ( x _ { t } ^ { i } ) \left( \frac { p _ { \mathrm { A R } , t } ^ { + } ( x _ { t } ^ { i } ) } { p _ { \mathrm { A R } , t } ^ { - } ( x _ { t } ^ { i } ) } \right) ^ { \lambda } ,\tag{6}
$$

where $Z _ { t }$ is the partition function, which is assumed to be finite and positive. The ratio reweights the reference distribution according to how $h ^ { i }$ changes the likelihood of the noisy chunk under the AR teacher. Treating the history-free AR distribution as the marginal over additional history, Bayes’ rule gives:

$$
\begin{array} { r l } & { \frac { p _ { \mathrm { A R } , t } ^ { + } \left( x _ { t } ^ { i } \right) } { p _ { \mathrm { A R } , t } ^ { - } \left( x _ { t } ^ { i } \right) } = \frac { p _ { \mathrm { A R } , t } \left( h ^ { i } \mid x _ { t } ^ { i } , c \right) } { p _ { \mathrm { A R } } \left( h ^ { i } \mid c \right) } , } \\ & { s _ { \mathrm { H F } } = s _ { \mathrm { r e f } } + \lambda \nabla _ { x _ { t } ^ { i } } \log p _ { \mathrm { A R } , t } ( h ^ { i } \mid x _ { t } ^ { i } , c ) . } \end{array}\tag{7}
$$

The log-posterior gradient defines a local direction that increases the AR teacher’s posterior density of the additional history, providing guidance for the reference score.

## 4.2 Distillation for Long-Horizon Consistency

To improve long-horizon consistency, we use the history-reweighted distribution in Eq. (6) as the target for DMD. A pretrained bidirectional difusion model defines the reference distribution, while a frozen AR teacher provides the history-dependent correction.

During long autoregressive rollouts, subject appearance and scene structure may gradually drift as generation errors accumulate. Earlier chunks can provide subject and scene information that is less afected by accumulated drift. To incorporate this information into distillation, we construct the additional history $h ^ { i }$ from consecutive latent frames immediately preceding the shared recent context in the same student rollout. One AR teacher context combines this additional history with the recent latent frames in chronological order, whereas the other retains only the recent frames. Sharing the recent context helps preserve autoregressive continuity and avoid an excessive gap between the two conditional distributions.

The residual between the two teacher scores provides a history-dependent correction to the bidirectional reference score. As illustrated in ${ \mathrm { F i g . } }$ 4, the reference distribution closely approximates the data distribution but does not incorporate the additional history. Direct AR distillation uses the history-conditioned AR distribution as the full target, tying historical supervision to the AR teacher’s generative accuracy. History Forcing instead retains the bidirectional reference score $s _ { \mathrm { r e f } }$ for visual-quality supervision and uses the posterior-gradient term in Eq. (7) to encourage consistency with the additional history, thereby decoupling historical supervision from matching the full AR distribution.

To match the history-reweighted target, the student minimizes $\mathcal { L } _ { \mathrm { H F } } = \mathbb { E } _ { t } [ D _ { \mathrm { K L } } ( q _ { \theta , t } \Vert p _ { \mathrm { H F } , t } ) ]$ . The history and shared condition are held fixed during each student update. Using Eq. (7), minimizing the reverse KL at timestep t is equivalent to maximizing

$$
\mathcal { I } _ { t } ( \theta ) = - D _ { \mathrm { K L } } ( q _ { \theta , t } \| p _ { \mathrm { r e f } , t } ) + \lambda \mathbb { E } _ { x _ { t } ^ { i } \sim q _ { \theta , t } } \left[ \log p _ { \mathrm { A R } , t } ( h ^ { i } \mid x _ { t } ^ { i } , c ) \right] .\tag{8}
$$

The KL term matches the student to the bidirectional reference distribution, while the posterior term encourages consistency with the additional history.

This objective is optimized using autoregressive rollouts generated with the student’s inference-time sampling procedure. Score estimates are obtained from denoising predictions on sampled chunks perturbed at random difusion timesteps. Replacing the reference score in Eq. (3) with s<sub>HF</sub> yields the DMD update:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { H F } } \approx \mathbb { E } \left[ w _ { t } \alpha _ { t } J _ { \boldsymbol { \theta } } ^ { \top } \left( s _ { \mathrm { f a k e } } - s _ { \mathrm { H F } } \right) \right] .\tag{9}
$$

The expectation is taken over student samples, difusion timesteps, and perturbation noise. The DMD weighting and fake-score training remain unchanged.

## 4.3 Refinement for Out-of-Sight Evolution

In camera-controlled video world models, scene evolution during out-of-sight intervals poses an additional challenge to long-horizon consistency. Earlier chunks of the student’s rollout provide historical observations that constrain subject appearance and scene structure upon revisit, but do not capture changes occurring outside the field of view. Historical supervision derived from these observations therefore provides limited guidance for out-of-sight evolution.

To provide supervision for out-of-sight evolution, we use a teacher model to generate a video from the same initial scene image as the student. The student follows its prescribed camera trajectory, whereas the teacher keeps the camera at its initial pose to model how the scene evolves while outside the student’s view. As illustrated in Fig. 3, we construct $h ^ { i }$ for each supervised student chunk using latent frames from the teacher’s fixed-camera rollout at timestamps preceding the start of that chunk. To extract supervision from this history, we compare teacher predictions with and without $h ^ { i }$ . The teacher’s inputs share the latent frames immediately preceding the current chunk in the student’s rollout to preserve continuity along the student’s camera trajectory.

We refine the pretrained world model through on-policy self-distillation to improve out-of-sight evolution while preserving its visual quality. Because a distilled few-step generator is not trained to provide accurate denoising predictions across arbitrary noise levels, directly using it as a score teacher can be unreliable. Following OPSD-V [38], we instead supervise the student’s velocity predictions along its sampling trajectory using history-guided targets constructed from an exponential moving average (EMA) teacher. This formulation requires no auxiliary fake-score model.

Let $v _ { \mathrm { E M A } } ^ { + }$ and $v _ { \mathrm { E M A } } ^ { - }$ denote the teacher’s velocity predictions with and without $h ^ { i } .$ , using the same noisy student chunk, timestep, prompt, and camera controls. History Forcing constructs the corrected target as:

$$
v _ { \mathrm { H F } } = v _ { \mathrm { E M A } } ^ { - } + \lambda \left( v _ { \mathrm { E M A } } ^ { + } - v _ { \mathrm { E M A } } ^ { - } \right) .\tag{10}
$$

The student is optimized through

$$
\mathcal { L } _ { \mathrm { H F } } = \mathbb { E } [ \mathrm { M S E } ( v _ { \theta } , \mathrm { s g } [ v _ { \mathrm { H F } } ] ) ] ,\tag{11}
$$

where $v _ { \theta }$ denotes the student’s velocity prediction, and sg denotes stop-gradient. Supervision is applied to chunks generated during revisits.

Table 1 Comparison on VBench for 15s video generation. Causal Forcing’s released AR difusion checkpoint [72], trained on 5s videos, serves as the 5s teacher. We further fine-tune this checkpoint on 10s videos. We evaluate the released checkpoints of all competing methods. Best and second-best scores are bold and underlined.
<table><tr><td>Method</td><td>Quality Score ↑</td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td><td>Temporal Flickering ↑</td><td>Motion Smoothness ↑</td><td>Dynamic Degree ↑</td><td>Aesthetic Quality ↑</td><td>Imaging Quality ↑</td></tr><tr><td>CausVid</td><td>82.68</td><td>92.16</td><td>92.43</td><td>99.08</td><td>98.23</td><td>60.56</td><td>66.30</td><td>67.94</td></tr><tr><td>Self-Forcing</td><td>83.70</td><td>92.46</td><td>93.97</td><td>98.84</td><td>98.61</td><td>61.11</td><td>66.54</td><td>70.97</td></tr><tr><td>LongLive</td><td>84.20</td><td>94.15</td><td>95.75</td><td>99.11</td><td>98.77</td><td>53.33</td><td>68.06</td><td>70.94</td></tr><tr><td>Rolling Forcing</td><td>84.20</td><td>95.61</td><td>95.80</td><td>98.85</td><td>98.62</td><td>51.11</td><td>66.89</td><td>72.65</td></tr><tr><td>Infinity-RoPE</td><td>84.54</td><td>92.64</td><td>95.52</td><td>99.25</td><td>98.49</td><td>68.89</td><td>66.47</td><td>69.65</td></tr><tr><td>Context Forcing</td><td>83.76</td><td>94.20</td><td>95.33</td><td>98.77</td><td>98.28</td><td>60.28</td><td>66.47</td><td>69.32</td></tr><tr><td>Baseline</td><td>84.44</td><td>91.84</td><td>93.58</td><td>98.99</td><td>98.46</td><td>76.67</td><td>66.10</td><td>69.83</td></tr><tr><td>History Forcing (5s teacher)</td><td>84.82</td><td>92.58</td><td>93.91</td><td>99.05</td><td>98.42</td><td>78.06</td><td>66.70</td><td>69.64</td></tr><tr><td>History Forcing (10s teacher)</td><td>85.07</td><td>93.21</td><td>95.03</td><td>99.34</td><td>98.56</td><td>70.00</td><td>67.52</td><td>70.98</td></tr></table>

Table 2 Comparison on VBench-Long for 60s video generation. Results for competing methods are taken from LongLive-2.0 [12]. Aggregate Score uses VBench normalization and weights over the six metrics. Teacher Size refers to the DMD teacher. History Forcing uses the AR teacher fine-tuned on 10s videos. Best and second-best results among 1.3B models are bold and underlined.
<table><tr><td>Method</td><td>Student Size</td><td>Teacher Size</td><td>Aggregate Score ↑</td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td><td>Motion Smoothness ↑</td><td>Dynamic Degree ↑</td><td>Aesthetic Quality ↑</td><td>Imaging Quality ↑</td></tr><tr><td>Other model sizes</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NOVA</td><td>0.6B</td><td></td><td>64.22</td><td>77.50</td><td>88.06</td><td>98.94</td><td>12.00</td><td>47.53</td><td>44.97</td></tr><tr><td>MAGI-1</td><td>4.5B</td><td></td><td>71.33</td><td>79.46</td><td>87.76</td><td>99.26</td><td>56.00</td><td>52.10</td><td>54.54</td></tr><tr><td>LongLive-2.0</td><td>5B</td><td>5B</td><td>79.90</td><td>97.48</td><td>97.00</td><td>98.86</td><td>60.62</td><td>53.68</td><td>65.51</td></tr><tr><td>1.3B models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Causal Forcing</td><td>1.3B</td><td>14B</td><td>76.43</td><td>93.52</td><td>94.12</td><td>95.74</td><td>72.32</td><td>51.24</td><td>62.30</td></tr><tr><td>SkyReels-V2</td><td>1.3B</td><td></td><td>74.80</td><td>84.99</td><td>89.95</td><td>98.67</td><td>44.00</td><td>57.64</td><td>66.67</td></tr><tr><td>Self-Forcing</td><td>1.3B</td><td>14B</td><td>77.73</td><td>95.84</td><td>95.27</td><td>98.20</td><td>51.72</td><td>56.05</td><td>62.22</td></tr><tr><td>CausVid</td><td>1.3B</td><td>1.3B</td><td>76.85</td><td>86.75</td><td>89.85</td><td>98.47</td><td>52.00</td><td>62.88</td><td>67.47</td></tr><tr><td>Rolling Forcing</td><td>1.3B 1.3B</td><td>14B 14B</td><td>79.22 79.12</td><td>94.09 97.13</td><td>94.47 95.89</td><td>98.65</td><td>36.00</td><td>63.50</td><td>72.42</td></tr><tr><td>LongLive</td><td></td><td></td><td></td><td></td><td></td><td>98.61</td><td>44.56</td><td>58.17</td><td>67.56</td></tr><tr><td>History Forcing</td><td>1.3B</td><td>1.3B</td><td>79.82</td><td>97.32</td><td>96.19</td><td>98.46</td><td>49.06</td><td>58.06</td><td>69.17</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Long-video generation. We initialize the bidirectional DMD teacher and the trainable fake-score model from pretrained Wan2.1-T2V-1.3B [50] weights. The AR student is initialized from the released causal ODE checkpoint of Causal Forcing [72]. The AR teacher is initialized from Causal Forcing’s released multi-step AR difusion checkpoint, which was trained on 5s video clips. To assess the gains in student performance from improving the AR teacher’s ability to model long-range dependencies, we further fine-tune this teacher on 10s clips from UltraVideo-Long [56]. We train the student for 300 iterations on filtered and expanded VidProM [51] prompts.

Student training follows the long-rollout training strategy of Self-Forcing++ [13]. Rollout durations are uniformly sampled between 5s and 15s, with DMD supervision applied to the final 5s window. Both AR teacher inputs share the latest 6 student-generated latent frames. The history-conditioned input additionally includes the 12 latent frames immediately preceding this shared context. For eficiency, history supervision is restricted to 1 randomly sampled eligible chunk. Unless otherwise stated, historical supervision is applied with λ = 1 at difusion timesteps t < 750. Our baseline follows the Self-Forcing++ training strategy, using the same student initialization and training setup as History Forcing but without historical supervision. We evaluate 15s generation using VBench [29]. Following LongLive-2.0 [12], we use VBench-Long [30] to evaluate 60s videos generated from MovieGenBench prompts [44]. See Section A.1 for details.

Out-of-sight world modeling. We initialize the world model from the released DreamX-World-5B [48] checkpoint, which also serves as the baseline. The training set contains 200 initial scene images from Pexels [1], each used with 4 camera trajectories that leave and return to the initial viewpoint, yielding 800 training samples. Our adapted OPSD-V [38] baseline uses fixed-view teacher rollouts in place of real-video teacher history and corresponds to the λ = 1 case of History Forcing. We fine-tune the model for 100 steps using LoRA, with supervision at all 4 denoising steps of the revisit chunks. Unless otherwise stated,

Table 3 World-model comparison on the PlayWorld Out-of-sight subset and MemoBench Real subset. OPSD-V is adapted using fixed-view rollout history and corresponds to λ = 1. PlayWorld metrics use a 1–5 rating scale. Bold indicates the best result.
<table><tr><td>Metric</td><td>Baseline</td><td>OPSD-V</td><td>History Forcing</td></tr><tr><td>PlayWorld Out-of-sight</td><td></td><td></td><td></td></tr><tr><td>Raw Mean ↑</td><td>2.38</td><td>2.44</td><td>2.69</td></tr><tr><td>Gated Mean ↑</td><td>2.29</td><td>2.37</td><td>2.49</td></tr><tr><td>Evolution Quality ↑</td><td>2.35</td><td>2.42</td><td>2.70</td></tr><tr><td>Subject Identity ↑</td><td>2.70</td><td>2.72</td><td>2.98</td></tr><tr><td>MemoBench Real</td><td></td><td></td><td></td></tr><tr><td>VisQual ↑</td><td>50.41</td><td>50.62</td><td>50.73</td></tr><tr><td>MotSmooth ↑</td><td>64.03</td><td>65.71</td><td>66.56</td></tr><tr><td>ObjConsist ↑</td><td>58.43</td><td>59.73</td><td>62.18</td></tr><tr><td>3DConsist ↑</td><td>84.35</td><td>84.51</td><td>86.22</td></tr><tr><td>ORS ↑</td><td>62.57</td><td>62.83</td><td>62.89</td></tr><tr><td>PSNR (dB) ↑</td><td>14.2971</td><td>14.3888</td><td>14.4671</td></tr><tr><td>SSIM ↑</td><td>0.5363</td><td>0.5487</td><td>0.5463</td></tr><tr><td>LPIPS ↓</td><td>0.5703</td><td>0.5674</td><td>0.5558</td></tr><tr><td>ImgReward ↑</td><td>52.27</td><td>52.91</td><td>53.43</td></tr></table>

Table 4 Ablation study on historical supervision strength.
<table><tr><td>Setting</td><td>Quality</td><td>Subject</td><td>Background Dynamic Score ↑ Consistency ↑ Consistency ↑ Degree ↑</td><td></td></tr><tr><td colspan="5">Timestep threshold</td></tr><tr><td>τ = 1000</td><td>84.62</td><td>92.35</td><td>93.45</td><td>80.28</td></tr><tr><td>τ = 750</td><td>84.82</td><td>92.58</td><td>93.91</td><td>78.06</td></tr><tr><td>τ = 500</td><td>85.08</td><td>92.52</td><td>94.01</td><td>80.00</td></tr><tr><td colspan="5">History weight</td></tr><tr><td>λ = 0 (Baseline)</td><td>84.44</td><td>91.84</td><td>93.58</td><td>76.67</td></tr><tr><td>λ = 0.5</td><td>84.68</td><td>91.17</td><td>93.16</td><td>89.72</td></tr><tr><td>λ = 1</td><td>84.82</td><td>92.58</td><td>93.91</td><td>78.06</td></tr><tr><td>λ = 2</td><td>84.50</td><td>93.29</td><td>93.97</td><td>73.61</td></tr></table>

Table 5 Ablation study on history construction.
<table><tr><td>History (+)</td><td>(−)</td><td>History Quality</td><td>Subject</td><td>Background Dynamic Score ↑ Consistency ↑ Consistency ↑ Degree ↑</td><td></td></tr><tr><td>18 recent</td><td>6 recent</td><td>84.82</td><td>92.58</td><td>93.91</td><td>78.06</td></tr><tr><td>18 recent</td><td>None</td><td>81.91</td><td>90.68</td><td>93.50</td><td>65.28</td></tr><tr><td>3 sink, 15 recent 6 recent</td><td></td><td>83.81</td><td>89.78</td><td>92.00</td><td>95.83</td></tr></table>

History Forcing uses λ = 3. We evaluate the models on the PlayWorld Out-of-sight subset (43 cases) [14] and MemoBench Real subset (164 clips) [9] using preset camera trajectories. See Section A.2 for details.

## 5.2 Quantitative Comparison

For 15s generation, we compare History Forcing with CausVid [63], Self-Forcing [28], LongLive [57], Rolling Forcing [39], Infinity-RoPE [59], and Context Forcing [11]. For 60s generation, we compare against the methods reported in LongLive-2.0 [12], covering student model sizes from 0.6B to 5B. For out-of-sight world modeling, the baselines are DreamX-World-5B [48] and its adapted OPSD-V variant [38].

15-second generation. As shown in Table 1, History Forcing achieves the highest quality score among the compared methods and improves temporal consistency over the DMD baseline. Adding historical supervision from the AR teacher raises the quality score from 84.44 to 84.82 and improves subject consistency, background consistency, and dynamic degree. Fine-tuning this teacher on 10s videos further raises the quality score to 85.07, with additional gains in both consistency measures. These additional gains suggest that improving the AR teacher’s ability to capture long-range dependencies strengthens historical supervision during distillation.

60-second generation. In Table 2, History Forcing achieves the highest aggregate score of 79.82 among the compared 1.3B students on VBench-Long, exceeding Rolling Forcing by 0.60 points. It also achieves the best subject and background consistency in this group, indicating stronger preservation of subject identity and scene appearance over minute-long rollouts. These results are obtained with student training rollouts of at most 15s, compared with the 60s rollouts used in LongLive’s streaming long tuning. Using a 1.3B DMD teacher, our method achieves a higher aggregate score than Rolling Forcing and LongLive, which use 14B DMD teachers. Its aggregate score also approaches that of the 5B LongLive-2.0 model (79.90).

Out-of-sight evaluation. Table 3 shows that History Forcing improves camera-controlled revisits on both PlayWorld Out-of-sight and MemoBench Real. On PlayWorld, it achieves the best results across all 4 metrics, with Evolution Quality and Subject Identity exceeding OPSD-V by 0.28 and 0.26 points, respectively. These gains indicate more plausible scene progression during unobserved intervals while preserving subject identity upon revisit. On MemoBench Real, History Forcing improves all 9 reported metrics over the baseline and outperforms OPSD-V on 8 of them. The improvements in visual quality are accompanied by better object preservation and higher fidelity to the reference videos, suggesting that historical supervision benefits both the realism of generated frames and the consistency of scene content across viewpoint changes.

## 5.3 Ablation Studies

We ablate the timestep threshold, history weight, teacher history construction, and the normalization scheme for long-video generation, as well as the number of supervised denoising steps and history weight for world-model self-distillation. The long-video ablations use Causal Forcing’s released multistep AR difusion checkpoint for historical supervision and report VBench results for 15s generation.

Table 6 Ablation study on historical supervision strength for video world modeling.
<table><tr><td rowspan="2">Setting</td><td colspan="4">PlayWorld Out-of-sight</td><td colspan="10">MemoBench Real</td></tr><tr><td>Raw</td><td>Gated</td><td>Evo.</td><td>Subject Mean ↑ Mean ↑ Qual. ↑ Identity ↑</td><td>VisQual ↑</td><td>Mot.</td><td>Obj. Smooth ↑ Consist ↑ Consist ↑</td><td>3D</td><td>ORS ↑</td><td>PSNR (dB) ↑</td><td>↑</td><td>SSIM LPIPS ↓</td><td>Img. Reward ↑</td></tr><tr><td colspan="10">Supervised denoising steps</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All 4 steps</td><td>2.60</td><td>2.51</td><td>2.60</td><td>2.86</td><td>50.64</td><td>66.38</td><td>61.82</td><td>85.28</td><td></td><td>61.66 14.4463 0.5480 0.5595</td><td></td><td></td><td>53.36</td></tr><tr><td>Last 3 steps</td><td>2.28</td><td>2.07</td><td>2.23</td><td>2.51</td><td>50.69</td><td>65.49</td><td>59.82</td><td>84.97</td><td></td><td>61.2614.37450.54770.5653</td><td></td><td></td><td>52.88</td></tr><tr><td>Last 2 steps</td><td>2.45</td><td>2.36</td><td>2.44</td><td>2.65</td><td>50.61</td><td>64.86</td><td>59.26</td><td>84.24</td><td></td><td>61.1914.3600 0.54760.5667</td><td></td><td></td><td>53.02</td></tr><tr><td colspan="10">History weight</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>λ = 1</td><td>2.44</td><td>2.37</td><td>2.42</td><td>2.72</td><td>50.62</td><td>65.71</td><td>59.73</td><td>84.51</td><td></td><td>62.8314.38880.54870.5674</td><td></td><td></td><td>52.91</td></tr><tr><td>λ = 2</td><td>2.60</td><td>2.51</td><td>2.60</td><td>2.86</td><td>50.64</td><td>66.38</td><td>61.82</td><td>85.28</td><td></td><td>61.66 14.4463 0.54800.5595</td><td></td><td></td><td>53.36</td></tr><tr><td>λ = 3</td><td>2.69</td><td>2.49</td><td>2.70</td><td>2.98</td><td>50.73</td><td>66.56</td><td>62.18</td><td>86.22</td><td></td><td>62.8914.4671 0.54630.5558</td><td></td><td></td><td>53.43</td></tr><tr><td>λ = 5</td><td>2.63</td><td>2.41</td><td>2.65</td><td>2.95</td><td>50.62</td><td>66.43</td><td>62.78</td><td>86.96</td><td></td><td>62.2614.4680 0.5407 0.5534</td><td></td><td></td><td>53.73</td></tr></table>

Table 7 Ablation study on gradient normalization.
<table><tr><td>Setting</td><td>Quality</td><td>Subject</td><td>Background Dynamic Score ↑ Consistency ↑ Consistency ↑ Degree ↑</td><td></td></tr><tr><td>Without joint norm.</td><td>84.09</td><td>93.78</td><td>94.46</td><td>59.72</td></tr><tr><td>Joint norm.</td><td>84.82</td><td>92.58</td><td>93.91</td><td>78.06</td></tr></table>

Table 8 Training eficiency for long-video generation.
<table><tr><td></td><td>Method Training Time ↓ Peak Allocated Memory ↓ (s/step)</td><td>(GiB/GPU)</td></tr><tr><td>DMD</td><td>171.30</td><td>33.10</td></tr><tr><td>Ours</td><td>175.26 (+2.31%)</td><td>33.76 (+1.99%)</td></tr></table>

Historical supervision strength. The timestep threshold τ restricts historical supervision to $t < \tau ,$ while λ controls its contribution to the DMD target. Table 4 varies each factor separately, keeping the other at its default value of $\tau = 7 5 0$ or λ = 1. Reducing τ from 1000 to 500 raises the quality score from 84.62 to 85.08, whereas $\tau = 7 5 0$ achieves the highest subject consistency (92.58). These results favor concentrating historical supervision at lower noise levels. In the weight ablation, increasing λ from 0.5 to 2 improves subject and background consistency but reduces dynamic degree. The highest quality score is achieved at λ = 1, indicating that a moderate history weight balances historical constraints with the bidirectional teacher’s supervision for visual quality.

In world-model self-distillation, Table 6 studies the number of supervised denoising steps and the history weight. With λ = 2 fixed, supervising all 4 steps yields better overall results across both benchmarks than supervising only the last 3 or 2 steps, supporting historical supervision throughout denoising. At λ = 1, the distillation target is the teacher’s velocity prediction conditioned on additional history. For out-of-sight revisits, recent student observations provide little information about the revisited region, making this history essential for constraining its identity and evolving state. Using λ > 1 amplifies the history-induced prediction change, placing greater emphasis on these constraints during self-distillation. The history-weight ablation favors λ = 3 across both benchmarks, indicating that stronger historical supervision benefits scene consistency and evolution across viewpoint changes.

History construction. Table 5 studies the roles of shared recent context and initial sink frames. In the default setting, both AR teacher inputs share the latest 6 student-generated latent frames, while the + input additionally includes the 12 latent frames immediately preceding this shared context. Removing the 6 recent frames from the − input reduces the quality score from 84.82 to 81.91 and lowers both consistency measures. Sharing recent context keeps the immediate history fixed, allowing the residual to capture the contribution of earlier observations. At the same history budget, replacing the oldest 3 frames in the + input with initial sink frames also lowers the quality score to 83.81 and reduces subject and background consistency. These results favor retaining consecutive earlier chunks for historical supervision, together with a shared recent context that preserves autoregressive continuity.

Gradient normalization. Table 7 examines whether the history term is jointly normalized with the DMD gradient. Standard DMD divides its distillation gradient by the mean absolute diference between the student-generated sample and the bidirectional teacher’s denoising prediction. Without joint normalization, the history term is added after this division. Joint normalization instead computes the normalization factor using the denoising target that includes the history term, then divides the combined DMD and history update by this factor. This increases the quality score from 84.09 to 84.82 and dynamic degree from 59.72 to 78.06, while the alternative yields higher subject and background consistency. These results favor joint normalization for balancing temporal consistency with motion dynamics.

(a) Yoga in a park

![](images/65342749852c334fa4796e9283623e6c27ee1dd4a28a30c03ce71f46f077df0b.jpg)  
Figure 5 Qualitative comparisons of consistency. (a) Temporal consistency in 60s autoregressive video generation with Self-Forcing, LongLive, Rolling Forcing, and History Forcing. (b,c) Revisit consistency in video world models, with each pair showing the baseline and History Forcing.

Training eficiency. Table 8 reports the training overhead relative to DMD. By restricting historical supervision to one sampled chunk, we keep the increases in training time per step and peak allocated memory to only 2.31% and 1.99%, respectively. History Forcing thus achieves its quality and consistency gains with modest additional training cost.

## 5.4 Qualitative Comparison

Fig. 5 presents qualitative comparisons of temporal consistency in long-video generation and revisit consistency in out-of-sight world modeling. In the 60s yoga example (a), Self-Forcing develops pronounced color artifacts as generation progresses. LongLive exhibits limb duplication, while Rolling Forcing shows leg deformations during pose transitions. History Forcing maintains more consistent body structure and clothing appearance while preserving the park setting throughout the rollout. For camera-controlled revisits (b,c), the baseline alters the drinking glass geometry and the colored-band layout. In contrast, History Forcing retains the original glass shape and preserves the surrounding pattern as paint covers the pink region. Additional qualitative examples are provided in Section B.

## 6 Conclusion

We presented History Forcing, which decouples historical supervision for few-step autoregressive video generators by comparing teacher predictions with and without informative history. The method augments DMD with long-range historical constraints while retaining the bidirectional teacher’s supervision for visual quality. Fixed-view teacher history further guides out-of-sight evolution through the OPSD velocity target. Experiments show that History Forcing reduces appearance and layout drift in long videos and better preserves object-state changes across out-of-sight intervals, producing more consistent states when the camera returns.

Limitations and Future Work. Our experiments are limited to 1.3B teachers and students for long-video generation and a 5B backbone for world modeling. Across both settings, historical supervision remains constrained by the underlying models’ ability to capture temporal dynamics. In world modeling, for example, limited scene evolution in fixed-view teacher rollouts can weaken supervision for out-of-sight state changes. Future work will explore larger teachers and students, longer-context training, and improvements to the world model’s ability to simulate physically plausible dynamics.

## References

[1] Pexels. https://www.pexels.com/. Accessed: 2026-10-19. 7

[2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tifany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 3

[3] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Difusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024. 3

[4] Hmrishav Bandyopadhyay, Xuanchi Ren, Zijian Huang, Jay Zhangjie Wu, Tianshi Cao, Ruilong Li, Bryan Chu, Sanja Fidler, Yi-Zhe Song, and Zian Wang. Context-matched distillation: Teacher causality for autoregressive video distillation. arXiv preprint arXiv:2608.13391, 2026. 2, 3

[5] Yuxuan Bian, Zeyue Xue, Songchun Zhang, Shiyi Zhang, Weiyang Jin, Yaowei Li, Junhao Zhuang, Haoran Li, Jie Huang, Haoyang Huang, et al. Echo-infinity: Learning evolving memory for real-time infinite video generation. arXiv preprint arXiv:2606.04527, 2026. 3

[6] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024. 3

[7] Shengqu Cai, Weili Nie, Chao Liu, Julius Berner, Lvmin Zhang, Nanye Ma, Hansheng Chen, Maneesh Agrawala, Leonidas Guibas, Gordon Wetzstein, et al. Mode seeking meets mean seeking for fast long video generation. arXiv preprint arXiv:2602.24289, 2026. 3

[8] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3

[9] Haoyu Chen, Kaichen Zhou, Hang Hua, Kaile Zhang, Jingwen Qian, Wufei Ma, Haonan Chen, Chunjiang Liu, Yizhou Zhao, Xiaoyuan Wang, et al. Memobench: Benchmarking world modeling in dynamically changing environments. In European Conference on Computer Vision, pages 658–676. Springer, 2026. 8, 17

[10] Siyi Chen, Shaowei Liu, Yixuan Jia, Zian Wang, Huan Ling, Qing Qu, and Jun Gao. Data-forcing distillation: Restoring diversity and fidelity in few-step video generation. arXiv preprint arXiv:2606.18478, 2026. 3

[11] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026. 2, 3, 8

[12] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, et al. Longlive-2.0: An nvfp4 parallel infrastructure for long video generation. arXiv preprint arXiv:2605.18739, 2026. 7, 8

[13] Jiaxing Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. In International Conference on Learning Representations, pages 85802–85822, 2026. 3, 7

[14] Kaixin Ding, Xi Chen, Minghong Cai, Zhiyuan Xu, Yiyang Wang, Yuxiang Lu, Junyi Li, Shuyang Chen, Yuan Gao, Xin Tao, et al. Playworld: Benchmarking world models with agent players over long-horizon objectives. arXiv preprint arXiv:2608.13552, 2026. 8, 16

[15] Yixuan Ding, Jiahao Kong, Wei Huang, Ruijie Quan, and Yi Yang. Layerrecall: A state-conditioned memory router for long-horizon consistency in video generation. arXiv preprint arXiv:2608.28460, 2026. 3

[16] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145, 2026. 2, 3

[17] Jiaqi Feng, Justin Cui, Yuanhao Ban, and Cho-Jui Hsieh. One-forcing: Towards stable one-step autoregressive video generation. arXiv preprint arXiv:2605.23458, 2026. 3

[18] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Eficient autoregressive video difusion model with causal generation and cache sharing. In International Conference on Machine Learning, pages 18550–18565. PMLR, 2025. 2, 4

[19] Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026. 3

[20] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 3

[21] Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video difusion model with on-policy flow map distillation. In European Conference on Computer Vision, pages 163–181. Springer, 2026. 3

[22] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-end training for autoregressive video difusion via self-resampling. In European Conference on Computer Vision, pages 324–344. Springer, 2026. 3

[23] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103, 2024. 3

[24] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 3

[25] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2568–2577. IEEE, 2025. 3

[26] Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5

[27] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025. 3

[28] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026. 2, 3, 8

[29] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818. IEEE, 2024. 7

[30] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 7

[31] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for eficient video generative modeling. In International Conference on Learning Representations, pages 23378–23402, 2025. 2

[32] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-difusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024. 3

[33] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3, 4

[34] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2(3):6, 2025. 3

[35] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 25690–25699. IEEE, 2025. 3

[36] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinitelength video generation with error recycling. In International Conference on Learning Representations, pages 23406–23432, 2026. 3

[37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3

[38] Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. Opsd-v: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026. 2, 3, 6, 7, 8

[39] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video difusion in real time. In International Conference on Learning Representations, pages 91177–91196, 2026. 2, 3, 8

[40] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2024. 3

[41] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Eficient streaming video generation with rewarded distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 34385–34397, 2026. 3

[42] Yanzuo Lu, Ronglai Zuo, and Jiankang Deng. Raven: Real-time autoregressive video extrapolation with consistency-model grpo. arXiv preprint arXiv:2605.15190, 2026. 3

[43] Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi, and Kaipeng Zhang. Packforcing: Short video training sufices for long video sampling and long context inference. In European Conference on Computer Vision, pages 576–593. Springer, 2026. 3

[44] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 7

[45] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video difusion. arXiv preprint arXiv:2502.06764, 2025. 3, 5

[46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456, 2020. 4

[47] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025. 3

[48] DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, et al. Dreamx-world 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026. 3, 7, 8

[49] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Difusion models are real-time game engines. In International Conference on Learning Representations, pages 73754–73776, 2025. 3

[50] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 7

[51] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video difusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024. 7

[52] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393, 2026. 3

[53] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. Advances in Neural Information Processing Systems, 38: 49632–49652, 2026. 3

[54] Tianshuo Xu, Yichen Xie, Depu Meng, Chensheng Peng, Quentin Herau, Bo Jiang, Yihan Hu, and Wei Zhan. Teaching video generators to remember: Eliciting dynamic memory for out-of-sight state evolution. arXiv preprint arXiv:2605.25333, 2026. 2, 3

[55] Bowen Xue, Brandon Y Feng, Chenguo Lin, Yuchen Lin, Yujia Zeng, Lvmin Zhang, Maneesh Agrawala, Honglei Yan, and Panwang Pan. Ring forcing: Towards precise long-term memory for autoregressive video difusion. In European Conference on Computer Vision, pages 225–242. Springer, 2026. 3

[56] Zhucun Xue, Jiangning Zhang, Teng Hu, Haoyang He, Yinan Chen, Yuxuan Cai, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, et al. Ultravideo: High-quality uhd video dataset with comprehensive captions. arXiv preprint arXiv:2506.13691, 2025. 7

[57] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025. 2, 3, 8

[58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations, pages 83048–83077, 2025. 3, 4

[59] Hidir Yesiltepe, Tuna Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40256–40265, 2026. 3, 8

[60] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Trainingfree long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025. 3

[61] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 3, 4

[62] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623. IEEE, 2024. 3, 4

[63] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22963–22974. IEEE, 2025. 2, 3, 8

[64] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025. 3

[65] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, et al. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025. 3

[66] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv e-prints, pages arXiv–2504, 2025.

[67] Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Beijia Lu, Anyi Rao, Song Han, Gordon Wetzstein, and Maneesh Agrawala. Pretraining frame preservation in autoregressive video memory compression. arXiv e-prints, pages arXiv–2512, 2025. 3

[68] Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, et al. minwm: A full-stack open-source framework for real-time interactive video world models. arXiv preprint arXiv:2605.30263, 2026. 3

[69] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive difusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026. 3

[70] Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rcm: A unified teacher-forcing and self-forcing open recipe for autoregressive difusion distillation in streaming video generation and interactive world models. arXiv preprint arXiv:2606.25473, 2026. 3

[71] Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. Sana-wm: Eficient minute-scale world modeling with hybrid linear difusion transformer. arXiv preprint arXiv:2605.15178, 2026. 3

[72] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026. 3, 7

[73] Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, et al. Self gradient forcing: Native long video extrapolation. arXiv preprint arXiv:2607.20368, 2026. 2, 3

## Appendix

In the Appendix, we provide the following:

• additional training, inference, and evaluation details in Appendix A.

• additional qualitative comparisons and examples of long-video generation and out-of-sight world modeling in Appendix B.

## A Additional Implementation Details

Both applications use AdamW with betas (0, 0.999), weight decay 0.01, and a maximum gradient norm of 10. Training uses 8 NVIDIA A100 GPUs with 1 sample per GPU and random seed 42. Student sampling uses 4 SDE steps with unshifted timesteps (1000, 750, 500, 250) and a timestep shift of 5. Each autoregressive chunk contains 3 latent frames, and videos are generated at 16 FPS. All context lengths below are measured in latent frames.

## A.1 Long-Video Generation

Training. The student learning rate is $2 \times 1 0 ^ { - 6 }$ , and the fake-score model learning rate is $4 \times 1 0 ^ { - 7 }$ . The bidirectional and AR teachers remain frozen during student distillation. The bidirectional teacher uses a classifier-free guidance scale of 3.0. The fake-score model is updated at every iteration and the student once every 5 iterations. Gradient accumulation over 8 steps gives an efective batch size of 64. The student EMA uses a decay of 0.99 and is initialized at iteration 200. The chunk for historical supervision is sampled from the DMD supervision window. AR teacher fine-tuning uses teacher forcing with a learning rate of $2 \times 1 0 ^ { - 6 }$ and an efective batch size of 8. For the 10s AR teacher, we use the fine-tuning checkpoint at iteration 2500. The AR teachers are configured with attention windows of 21 and 42 latent frames for the 5s and 10s variants, respectively. The two predictions use separate KV caches, with matched temporal positions for the current chunk. The threshold is applied to the shifted DMD scoring timestep. Joint normalization is used by default.

Inference. Inference uses student EMA weights at 832 × 480 resolution. All student variants use a 21-frame KV window with 3 sink frames in both training and inference.

Evaluation. The 15s VBench evaluation uses Wan2.1-augmented prompts, with 5 videos generated per prompt. The 60s VBench-Long evaluation uses all 1,003 entries in the original MovieGenBench prompt list without prompt expansion. We generate 1 video per entry, with each rollout containing 1,005 frames.

## A.2 Out-of-sight World Modeling

Training. Training camera trajectories use left/right rotations or translations followed by a return to the initial viewpoint. The learning rate is $1 \times 1 0 ^ { - 5 }$ , with LoRA rank 64, scaling parameter α = 128, and 0 dropout. No gradient accumulation is used, giving an efective batch size of 8. The EMA teacher is initialized from the student and updated after each optimization step with a decay of 0.9999. For each training sample, we generate the student and fixed-view teacher rollouts online with shared initial noise and SDE transition noise. The fixed-view teacher rollout uses a 12-frame KV window with 3 sink frames. Since the world model performs image-to-video generation, both inputs used for historical supervision retain 1 sink frame corresponding to the initial image. To reserve capacity for additional history within the model’s native attention window of 12 latent frames, the shared recent context is reduced to 4 student-generated latent frames. The history-conditioned input additionally includes up to 4 earlier latent frames from the fixed-view teacher rollout.

Inference. Evaluation uses the fine-tuned student weights. The student uses a 12-frame KV window with 3 sink frames at $1 2 8 0 \times 7 0 4$ resolution, generating 1 video per input.

Evaluation. All models are evaluated using the same initial images, prompts, preset camera trajectories, and random seeds. Generation prompts describe the initial scene and its ongoing process, while task objectives are used only for evaluation. For the PlayWorld Out-of-sight subset [14], VLM-based scoring uses Doubao Seed 2.0 Lite. Raw Mean averages ratings before instruction gating, and Gated Mean assigns a score of 1 to videos that fail the instruction check. Evolution Quality measures the extent and causal plausibility of state evolution. Subject Identity measures subject and appearance consistency across revisits. Both metrics report mean 1–5 ratings before instruction gating. For the MemoBench Real subset [9], automated metrics are computed using the oficial MemoBench implementation.

## B Additional Qualitative Analysis

Out-of-sight world modeling. Fig. 6 shows improved object preservation and scene evolution upon revisit. In the sand-filling example, the baseline accumulates sand outside the jar, while History Forcing keeps the sand predominantly inside it. History Forcing retains the tomatoes in the bowl, preserves the drinking glass as the cofee level rises, and keeps the purple solution inside the beaker. Some appearance and scene drift nevertheless remains. The fixed-camera teacher rollouts used for historical supervision can themselves contain temporal errors and exhibit limited scene evolution, as discussed in Section 6. These limitations reduce the accuracy of historical supervision and can contribute to inconsistencies upon revisit.

15-second video generation. Fig. 7 compares the DMD baseline and History Forcing on six 15s examples: a violin performance in a candle-lit chapel, a desert carnival, bread baking, a walk through a Tokyo street, a glacial cave, and a Hong Kong tram. In the chapel example (a), the baseline background changes abruptly from the chapel interior to outdoor greenery between the displayed frames at 5.44s and 5.5s. In the carnival example (b), the baseline also changes the subject’s appearance across frames. Across these examples, History Forcing better preserves the subjects, clothing, and surrounding scene at the sampled timestamps. 60-second video generation. Fig. 8 compares History Forcing with Self-Forcing, LongLive, and Rolling Forcing on a 60s toothbrushing example. Self-Forcing develops severe color saturation and high-contrast artifacts as the rollout progresses. LongLive produces inconsistent toothbrush geometry, with the brush appearing as an elongated object grasped at both ends in several frames. Rolling Forcing distorts the mirror geometry: the reflected face extends beyond the mirror boundary, and the rim intersects the mouth and cheek. History Forcing maintains natural colors and more consistent toothbrush appearance and mirror geometry through 60s. Fig. 9 further shows that History Forcing maintains consistent clothing, object appearance, and scene layout across the 60s rollouts.

![](images/54167ee03abaf7227ec9c83ac8e37ac80f2472e4e8fdf02ecf54b20c003fc3d7.jpg)  
Figure 6 Additional qualitative comparisons of out-of-sight world modeling with the baseline and History Forcing: (a) sand filling, (b) tomatoes into a bowl, (c) cofee pouring, and (d) purple solution pouring.

(c) Bread baking in an autumn kitchen  
(d) Walking through a neon-lit Tokyo street  
(a) A violin performance in a candle-lit chapel  
![](images/df262da08bc2138ed19bb3de70fed2dffd40763dc91c29228b2b56918d4937a8.jpg)

(b) A celebration at a desert carnival  
![](images/44b49e41a934e4ff8643e3b9b6c6b0570f03b71c5a6e9a49f34095f7873ab93e.jpg)

![](images/99d40d8ab8a13f42311f5626fde5767065692296380c3f115eb5786415506017.jpg)

![](images/b27e82be4fe6b52406a2aa45a8d14136b2a33015f09e9584c797673bf356f8fb.jpg)

![](images/0779a22de098a5728f22827f89f377b445e38705ffb7a072e32d5060696ce03c.jpg)

![](images/1105d52c0cb6fdd6334ea522f61e2e73bde5dabb684cef4a334a4fa14d998813.jpg)  
Figure 7 Additional qualitative comparisons of 15s video generation: (a) a violin performance in a candle-lit chapel, (b) a desert carnival, (c) bread baking, (d) a Tokyo street, (e) a glacial cave, and (f) a Hong Kong tram.

Brushing teeth in front of a mirror

![](images/af345f831c5a6438eb2a07503c3964a9be6e295ccb9767b4d7b6556d96eb92b0.jpg)  
Figure 8 Additional qualitative comparisons of 60s video generation: brushing teeth in front of a mirror.

![](images/d99507130212e13ed21fef373f7565033735fa9c7372bef3fe3db3952f6d34e3.jpg)  
Figure 9 Additional qualitative examples of 60s video generation with History Forcing: (a) making dumplings, (b) playing an acoustic guitar, and (c) a chef preparing a dish in a kitchen.