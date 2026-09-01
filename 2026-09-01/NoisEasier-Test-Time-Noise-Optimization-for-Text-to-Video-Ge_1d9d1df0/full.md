# NoisEasier: Test-Time Noise Optimization for Text-to-Video Generation

Yujiang Pu<sup>1</sup> and Yu Kong<sup>1</sup>

Michigan State University, East Lansing, MI 48824, USA {puyujian,yukong}@msu.edu https://yujiangpu20.github.io/noiseasier/

Abstract. Difusion models have recently advanced text-to-video (T2V) generation, yet they still struggle with fine-grained compositional alignment, such as attribute binding, spatial relations, and object interactions. While reward-based fine-tuning improves alignment, it is susceptible to reward hacking and adapts poorly to new prompt distributions. In this work, we propose NoisEasier, a test-time scaling framework that improves T2V generation through diferentiable reward-guided noise optimization without modifying the underlying model. By combining eficient short-step generators with a multi-objective reward formulation, NoisEasier enables stable and practical test-time optimization under realistic inference budgets. Our key insight is that jointly optimizing the entire stochastic trajectory accelerates reward convergence and improves compositional alignment over optimizing only the initial latent, with negligible additional computational and time cost. Experiments on VBench and T2V-CompBench demonstrate consistent improvements across multiple backbones, achieving over 10% average gains on challenging dimensions such as attribute binding, object interaction, and numeracy. Overall, NoisEasier serves as both a flexible alternative and a complementary enhancement to reward-based fine-tuning, establishing test-time scaling as an efective paradigm for controllable text-to-video generation.

Keywords: Video generation · Test-time scaling · Noise optimization

## 1 Introduction

Text-to-video (T2V) difusion models [3, 5, 16, 18, 41, 52, 55, 69] have recently achieved remarkable progress, producing videos with increasingly realistic appearance, diverse content, and coherent motion. Despite these advances, today’s systems still struggle with fine-grained compositional control that faithfully reflects user intent [45, 48]. Subtle details such as attribute bindings, object interactions, and motion dynamics often become unreliable when prompts require precise coordination among multiple semantic concepts. These limitations reveal a broader challenge: how can we align video generation more closely with user-specified semantics while preserving visual fidelity?

Dog firefighter rescues kittens from a tree.  
![](images/07414d73977a720a7c6ee52e864446e23fb094c6eed63a9f839efbb6d9f7891f.jpg)

Purple balloon floating near a yellow car.  
![](images/e0adbc8150e3efa91ae13632af6faaefc9b26a7c3e17b2fc7d837c6d0ce313ef.jpg)  
Six robots dance rhythmically in the lab.

![](images/a1dd48aa316707cb923e6fda60ec3458a499142d8c641846bb342831468071a1.jpg)

A football rolling from the right to the left on the grass.  
![](images/d131c5bb8f498343cc8bd5843175c4b4f8635f54ccb54561b454cc6e0935d46a.jpg)  
Fig. 1: NoisEasier significantly boosts semantic alignment while maintaining the visual fidelity of generated videos. The results are obtained with 25-step noise optimization.

Most existing works draw inspiration from Reinforcement Learning from Human Feedback (RLHF) [9, 36] and adapt similar paradigms to T2V generation. Early methods [39, 65] optimize difusion models using diferentiable rewards [23, 58, 61], while subsequent works rely on human-preference datasets for direct preference optimization [8, 11, 33] or reward-based fine-tuning [32, 59]. While efective in improving global alignment, these methods inherit two intrinsic limitations. First, the learned preferences are baked into the model parameters, making continual adaptation to new reward functions or prompt distributions expensive. Second, these methods remain susceptible to reward hacking, where optimizing imperfect reward functions can degrade visual quality or motion realism. This motivates a natural question: can we improve compositional alignment at test time without modifying the underlying video generator?

Recent advances in test-time scaling have shown that increasing inferencetime computation can substantially improve model capability without retraining. For difusion models, latent noise has emerged as an efective optimization variable [13,30,40,62,70], leading to successful applications in image [6,17], motion [22], and music generation [34]. However, extending noise optimization to T2V remains largely unexplored due to two fundamental challenges. First, modern T2V models [24,52] require long denoising chains and high memory consumption, making iterative gradient-based refinement prohibitively expensive. Second, video reward signals are considerably more fragile than image rewards [39, 65], making optimization prone to instability and reward hacking.

In this work, we propose NoisEasier, a practical test-time scaling framework for T2V generation. To overcome the prohibitive cost of iterative optimization, we build upon recent Video Consistency Models (VCMs) [27, 28, 53], which distill long difusion sampling into only a few denoising steps, making end-to-end gradient-based refinement practical. While VCMs address the computational bottleneck, stable optimization still requires reliable reward supervision. We therefore formulate test-time refinement as a robust multi-objective optimization problem that combines complementary semantic, aesthetic, and motion-aware rewards to mitigate reward hacking. To further strengthen compositional supervision, we introduce a lightweight negative-aware reward calibration strategy that calibrates absolute reward scores using semantically hard negative prompts, yielding more discriminative feedback.

Interestingly, the intermediate stochastic perturbations exposed by VCMs provide additional optimization variables beyond the initial latent noise. Rather than optimizing only the initialization [13,17,46,62], we find that jointly refining all available stochastic variables consistently yields faster reward convergence and stronger compositional alignment, while introducing negligible additional runtime or memory overhead. This observation suggests that optimizing the entire denoising process provides finer control over video generation than optimizing only its initialization, ofering a simple yet efective test-time scaling strategy for modern T2V models.

Extensive experiments on VBench [21] and T2V-CompBench [44] demonstrate that NoisEasier significantly improves the performance of VCM baselines [27, 53] with only modest additional inference overhead. Compared with optimizing the initial latent alone, jointly optimizing the intermediate perturbations yields more consistent improvements on compositional dimensions, including attribute binding, spatial relations, and object interactions. Furthermore, NoisEasier continues to improve reward-finetuned models, suggesting that offline fine-tuning does not fully amortize reward signals into the model parameters. Consequently, NoisEasier serves not only as a flexible alternative to reward fine-tuning, but also as a complementary test-time scaling mechanism that unlocks additional gains beyond ofline alignment.

In summary, our main contributions are as follows:

– We propose NoisEasier, a test-time scaling framework for text-to-video generation that enables eficient and robust reward-guided optimization without model fine-tuning.

– We develop an eficient optimization pipeline based on short-step Video Consistency Models together with a robust reward formulation for stable compositional optimization.

– We show that jointly optimizing the intermediate perturbations consistently accelerates reward convergence and improves compositional alignment over initial-noise optimization with negligible additional computational cost.

## 2 Related Work

Video Generation with Human Feedback. Reinforcement Learning from Human Feedback (RLHF) [9, 36] has shown great success in aligning large language models (LLMs) with human intent, motivating its extension to image and video generation. Existing work typically follows two paradigms. One is preference-driven optimization, where models are trained directly on large-scale human preference datasets. This includes policy gradient methods like DDPO [2] and DPOK [15], as well as direct preference optimization methods [31,33,50,63] that bypass explicit reward models. While efective, these approaches require large-scale preference annotations and cannot easily incorporate structured human knowledge. The other is reward-driven optimization, where pretrained reward models provide diferentiable feedback for fine-tuning generative models [39,65] or guide distillation for faster inference [26–28]. Although these methods reduce training cost and improve stability, they are generally one-shot systems that lack mechanisms for iterative refinement. In contrast, we directly optimize latent noise via diferentiable rewards, enabling preference-controllable text-to-video generation with eficient test-time refinement.

Test-time Noise Optimization. Recent studies show that the initial latent noise strongly influences difusion generation quality, with certain golden noise consistently yielding outputs that are more faithful and aesthetically pleasing. Building on this observation, test-time methods that operate on noise can be broadly categorized into two directions: selection-based noise search and optimization-based noise refinement. Selection-based methods [30, 62, 70] treat sampling as a search problem over initial noises or trajectories and select the best sample under a fixed backbone. In contrast, optimization-based methods directly refine the noisy latent via gradient-based updates during sampling. DOODL [51] pioneered end-to-end diferentiable latent optimization, and subsequent work extended this paradigm to images [6, 17, 46, 70] and other modalities, including music [35] and motion [22]. While efective, optimization-based approaches can be prohibitively slow at inference time; for example, DOODL [51] and D-Flow [1] may take tens of minutes per sample. ReNO [13] partially mitigates this bottleneck by applying optimization to one-step difusion models, reducing runtime to tens of seconds. In this work, we revisit noise optimization for text-to-video generation, which remains underexplored due to the high cost of video sampling and the scarcity of suitable spatiotemporal reward signals.

## 3 Method

## 3.1 Problem Formulation

Let $\mathcal { G } _ { \theta }$ denote a pretrained text-to-video (T2V) difusion model. Given a text prompt $c ,$ the generation process starts from an initial latent noise $\mathbf { z } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and progressively denoises it into a video sequence. Existing test-time optimization methods formulate noise refinement by directly optimizing the initial latent:

$$
\mathbf { z } _ { T } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { z } _ { T } } \mathcal { R } ( \mathcal { G } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { T } , c ) ) ,\tag{1}
$$

where $\mathcal { R } ( \cdot )$ denotes a diferentiable reward measuring semantic alignment or perceptual quality. While conceptually simple, directly optimizing $\mathbf { z } _ { T }$ for T2V generation is computationally expensive, as gradients must propagate through dozens of denoising steps. Previous works [10, 38, 39] alleviate this cost through gradient truncation or partial backpropagation, sacrificing end-to-end optimization while remaining computationally demanding. In contrast, we build upon recent Video Consistency Models (VCMs) [27, 53], which distill long difusion sampling into only a few denoising steps.

Specifically, a consistency model learns a mapping that directly predicts the clean sample along the probability-flow ODE trajectory: $f : ( \mathbf { z } _ { t } , t , c ) \mapsto \mathbf { z } _ { \kappa } , t \in$ $[ \kappa , T ]$ , where κ is a small positive constant. The model is trained with a selfconsistency objective that enforces identical predictions for diferent noisy states originating from the same trajectory:

$$
f ( \mathbf { z } _ { t } , t , c ) = f ( \mathbf { z } _ { t ^ { \prime } } , t ^ { \prime } , c ) , \quad \forall t , t ^ { \prime } \in [ \kappa , T ] ,\tag{2}
$$

allowing the original difusion trajectory to be approximated using only a small number of denoising steps. This substantially reduces the cost of gradient propagation and enables practical end-to-end optimization for video generation. During inference, VCM alternates between predicting the clean sample $\hat { \mathbf { z } } _ { 0 } = f ( \mathbf { z } _ { t } , t , c )$ and injecting Gaussian perturbations $\varepsilon _ { t } \sim \mathcal { N } ( 0 , \mathbf { I } )$ to obtain the next latent

$$
\mathbf { z } _ { t - 1 } = \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { z } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \varepsilon _ { t } .\tag{3}
$$

Unlike conventional difusion sampling, the stochastic formulation of VCM naturally exposes the injected perturbations as additional optimization variables. Rather than optimizing only the initial latent as in Eq. 1, we jointly optimize all available stochastic variables throughout the denoising process:

$$
( \mathbf { z } _ { T } ^ { * } , \{ \varepsilon _ { t } ^ { * } \} ) = \arg \operatorname* { m a x } _ { \mathbf { z } _ { T } , \{ \varepsilon _ { t } \} } \mathcal { R } \big ( \mathcal { G } _ { \theta } ( \mathbf { z } _ { T } , c ; \{ \varepsilon _ { t } \} ) \big ) .\tag{4}
$$

This formulation provides finer control over the denoising trajectory and consistently accelerates reward convergence compared with optimizing only the initial latent, while introducing negligible additional computational overhead.

## 3.2 Reward Models for Video Generation

While human-preference reward models are well studied for image generation [23, 57, 61], reward modeling for video remains underexplored. Recent works such as VideoRM [59], VideoScore [20], and VisionReward [60] leverage multimodal large language models (MLLMs) to assess generated videos. However, directly using large MLLM-based reward models inside gradient-based noise refinement can be computationally prohibitive, since the reward must be evaluated repeatedly during optimization. Therefore, we focus on lightweight, diferentiable rewards built on pretrained vision-language models (VLMs), which provide strong semantic alignment signals while remaining eficient for test-time optimization. Specifically, we consider three types of reward objectives:

Image-Text Alignment Reward. We employ HPSv2 [57] and ImageReward [61], both trained to model human preferences for image-text relevance. HPSv2 uses a

![](images/978f6bdaf1b43e12768eb6e57ebf5ab17bdbaab350298c01cd94547319d076ab.jpg)  
Fig. 2: Overview of the proposed NoisEasier framework. Given a text prompt, both the initial latent noise and intermediate perturbations are iteratively optimized using gradients from multiple reward models. Negative prompts generated by an LLM further calibrate the reward signal for fine-grained compositional feedback.

CLIP-based [42] architecture with a fine-tuned reward head, while ImageReward adopts BLIP [29] as its backbone, fine-tuned with supervised human feedback to improve perceptual alignment. Both models produce scalar scores that reflect the semantic alignment between the generated image and the text prompt. In practice, we compute the reward by evaluating each frame-text pair and take the mean score across all frames as the final image-text alignment reward.

Video-Text Alignment Reward. Image-level reward models provide strong semantic cues but lack temporal perception, making them inefective at capturing motion consistency in video generation. To address this, we incorporate Vi-CLIP [56], a video-language model trained on the large-scale InternVid dataset. ViCLIP jointly encodes short video clips and text descriptions to assess temporal and semantic consistency. In practice, we sample two temporally ofset clips from the generated video, each consisting of 8 frames, and compute their similarity to the text prompt. The final reward is the average of the two clip-text similarity scores, which provides video-level alignment feedback.

Motion-aware Reward. Although VLM-based rewards help improve semantic alignment with the prompt, they often push the model to favor static content to maximize alignment scores. As a result, the generated videos may lack meaningful motion or introduce temporal artifacts. To mitigate this, we introduce a motion-aware reward that explicitly encourages localized dynamics while regularizing temporal coherence. Specifically, we use RAFT [47] to estimate optical flow $\{ \check { \mathbf { F } } _ { t } \} _ { t = 1 } ^ { T - \mathrm { \bar { 1 } } }$ , where $\mathbf { F } _ { t } \in \mathbb { R } ^ { 2 \times H \times W }$ captures motion from frame t to t + 1. We define the spatial mean flow vector $\begin{array} { r } { \bar { \mathbf { f } } _ { t } = \frac { 1 } { H W } \sum _ { h , w } \mathbf { F } _ { t } [ : , h , w ] } \end{array}$ and subtract it to suppress global camera motion. The motion reward is computed as:

$$
\mathcal { R } _ { \mathrm { m o } } = \mathrm { t a n h } \Big ( \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \lVert \mathbf { F } _ { t } - \bar { \mathbf { f } } _ { t } \rVert _ { 1 } \Big ) + \lambda \Big ( 1 - \mathrm { t a n h } \Big ( \frac { 1 } { T - 2 } \sum _ { t = 1 } ^ { T - 2 } \lVert \mathbf { F } _ { t + 1 } - \mathbf { F } _ { t } \rVert _ { 1 } \Big ) \Big ) ,\tag{5}
$$

where the first term encourages localized motion beyond global translation, and the second term penalizes abrupt changes in flow magnitude to promote temporally coherent dynamics. In practice, this objective efectively mitigates the static bias induced by image-based rewards, leading to more expressive and dynamic video generation.

## 3.3 Negative-Aware Reward Calibration

Most existing reward models [23, 57, 58, 61] are trained on human preferences using pairwise ranking, providing only weak supervision for fine-grained compositional reasoning. Prior studies [12, 49, 66] further show that VLM-based evaluators often behave like bag-of-words classifiers. Consequently, reward signals can be overly coarse and insensitive to subtle semantic errors. To address this limitation, we introduce negative-aware reward calibration, which contrasts the target prompt against semantically perturbed distractors rather than relying solely on absolute reward scores.

Given a prompt $c ^ { + }$ , we utilize a large language model (LLM) to construct a set of hard negative prompts $\{ c _ { j } ^ { - } \} _ { j = 1 } ^ { N }$ that are syntactically similar but semantically incorrect, as illustrated in Figure 2. These negatives encourage the reward to capture fine-grained semantic distinctions rather than superficial lexical similarity. Given a generated video x and a reward model $\mathcal { R } ( \cdot )$ for a video–text pair, the calibrated reward is defined as

$$
\tilde { \mathcal { R } } = \mathcal { R } ( x , c ^ { + } ) - \tau \log \sum _ { j = 1 } ^ { N } \exp \left( \frac { \mathcal { R } ( x , c _ { j } ^ { - } ) - \mathcal { R } ( x , c ^ { + } ) + m } { \tau } \right) ,\tag{6}
$$

where m is a margin enforcing separation between the target prompt and its negatives, and \tau is a temperature parameter controlling the sharpness of the contrastive penalty. Intuitively, the calibrated reward encourages the generated video to achieve a high score for the target prompt while suppressing semantically similar distractors, making the optimization process more sensitive to compositional errors such as attribute binding, spatial relations, and numeracy.

Finally, we optimize both the initial latent noise $\mathbf { z } _ { T }$ and the stochastic perturbations $\{ \varepsilon _ { t } \} _ { t = 1 } ^ { \bar { T } }$ through gradient ascent on a weighted combination of the calibrated rewards. For brevity, we denote all optimization variables by $\epsilon =$ $\{ \mathbf { z } _ { T } , \varepsilon _ { 1 } , \dots , \varepsilon _ { T } \}$ , leading to the update rule

$$
\epsilon ^ { k + 1 } = \epsilon ^ { k } + \eta \cdot \nabla _ { \epsilon ^ { k } } \left[ \sum _ { i } \alpha _ { i } \tilde { \mathcal { R } } _ { i } ( \mathcal { G } _ { \theta } ( \epsilon ^ { k } , c ) , c ) \right] ,\tag{7}
$$

where $\eta$ denotes the learning rate and $\alpha _ { i }$ are the weights of individual reward components. Together, the proposed reward formulation and joint stochastic optimization enable stable and efective test-time refinement.

Table 1: Quantitative evaluation on VBench (%). The best results among open-source models are in bold and the second best is underlined. \* denotes our reproduction.
<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>(Closed-source)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gen-3 [43]</td><td>26.69</td><td>53.64</td><td>80.90</td><td>65.09</td><td>99.23</td><td>60.14</td></tr><tr><td>Sora [3]</td><td>26.26</td><td>70.85</td><td>80.11</td><td>74.29</td><td>98.74</td><td>79.91</td></tr><tr><td>Veo 3 [16]</td><td>27.88</td><td>82.20</td><td>82.48</td><td>84.26</td><td>99.16</td><td>72.43</td></tr><tr><td>(Open-source)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ModelScope [54]</td><td>25.67</td><td>38.98</td><td>81.72</td><td>33.68</td><td>95.79</td><td>66.39</td></tr><tr><td>VideoCrafter2 [5]</td><td>28.23</td><td>40.66</td><td>92.92</td><td>35.86</td><td>97.73</td><td>42.50</td></tr><tr><td>CogVideoX-2B [64]</td><td>27.33</td><td>66.59</td><td>83.01</td><td>74.27</td><td>97.51</td><td>64.79</td></tr><tr><td>Vchitect-2.0 [14]</td><td>27.57</td><td>68.84</td><td>87.04</td><td>57.55</td><td>98.98</td><td>63.89</td></tr><tr><td>AnimateLCM* [53]</td><td>25.74</td><td>30.78</td><td>81.84</td><td>39.94</td><td>98.19</td><td>29.17</td></tr><tr><td>+ NoisEasier (Ours)</td><td>28.79 ↑3.05</td><td>44.04 ↑13.26</td><td>88.22 ↑6.38</td><td>46.70 ↑6.76</td><td>98.57 ↑0.38</td><td>30.56 ↑1.39</td></tr><tr><td>T2V-Turbo (MS) [27]</td><td>27.51</td><td>58.63</td><td>89.67</td><td>45.74</td><td>95.64</td><td>66.39</td></tr><tr><td>T2V-Turbo (MS) *</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>+ NoisEasier (Ours)</td><td>32.05 ↑4.65</td><td>76.62 ↑22.12</td><td>94.38 ↑4.48</td><td>48.93 ↑2.31</td><td>95.29 ↓0.22</td><td>71.39 ↑0.56</td></tr><tr><td>T2V-Turbo (VC2) [27]</td><td>28.16</td><td>54.65</td><td>89.90</td><td>38.67</td><td>97.34</td><td>49.17</td></tr><tr><td>T2V-Turbo (VC2)* *</td><td>28.12</td><td>54.33</td><td>90.59</td><td>39.95</td><td>96.29</td><td>51.67</td></tr><tr><td>+ NoisEasier (Ours)</td><td>31.95 ↑3.83</td><td>79.18 ↑24.85</td><td>93.63 ↑3.04</td><td>52.61 ↑12.66</td><td>96.10 ↓0.19</td><td>59.17 ↑7.50</td></tr></table>

## 4 Experiment

Experimental Setup. We evaluate NoisEasier on two VCM backbones: AnimateLCM [53] and T2V-Turbo [27]. For T2V-Turbo, we consider two distilled variants: T2V-Turbo (MS) and T2V-Turbo (VC2), which are distilled from ModelScope [54] and VideoCrafter2 [5], respectively. T2V-Turbo (MS) generates videos at 256 × 256 resolution, while T2V-Turbo (VC2) produces 512 × 320 videos. AnimateLCM generates videos at 512 × 512 resolution. All three backbones support fast sampling with 4–8 denoising steps. To balance generation quality and eficiency, we adopt a 4-step sampler for fast inference. For noise refinement, we optimize the latent variables for 25 iterations using AdamW with gradient clipping and a learning rate of 0.01. Unless otherwise specified, all experiments use the same optimization configuration across backbones. To make noise optimization feasible, we employ gradient checkpointing [7] together with mixed-precision inference to reduce memory overhead. All the experiments are conducted on two RTX 6000 Ada GPUs. More implementation details can be found in the supplementary material.

Benchmarks and Metrics. We evaluate NoisEasier on two widely used benchmarks: VBench [21] and T2V-CompBench [45]. VBench provides a comprehensive evaluation across 16 dimensions covering visual quality and semantic alignment. Following common practice, we report results on four semantic dimensions (overall consistency, multiple objects, color, and spatial relations) and two motion-related dimensions, i.e., motion smoothness and dynamic degree. T2V-

Table 2: Quantitative evaluation on T2V-CompBench. The best results among opensource models are in bold, and the second best are underlined. \* denotes reproduction.
<table><tr><td>Models</td><td>Consist Attr.</td><td>Dynamic Attr.</td><td>Spatial</td><td>Motion</td><td>Action</td><td>Interaction</td><td>Numeracy</td></tr><tr><td>(Closed-source)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gen-3 [43]</td><td>0.5980</td><td>0.0687</td><td>0.5194</td><td>0.2754</td><td>0.5233</td><td>0.5906</td><td>0.2306</td></tr><tr><td>Dreamina [4]</td><td>0.6913</td><td>0.0051</td><td>0.5773</td><td>0.2361</td><td>0.5924</td><td>0.6824</td><td>0.4380</td></tr><tr><td>PixVerse [37]</td><td>0.7060</td><td>0.0624</td><td>0.5979</td><td>0.2867</td><td>0.8722</td><td>0.8309</td><td>0.6066</td></tr><tr><td>Kling [25]</td><td>0.6931</td><td>0.0098</td><td>0.5690</td><td>0.2562</td><td>0.5787</td><td>0.7128</td><td>0.4413</td></tr><tr><td>(Open-source)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ModelScope [54]</td><td>0.5148</td><td>0.0161</td><td>0.4118</td><td>0.2408</td><td>0.3639</td><td>0.4613</td><td>0.1986</td></tr><tr><td>Show-1 [67]</td><td>0.5670</td><td>0.0115</td><td>0.4544</td><td>0.2291</td><td>0.3881</td><td>0.6244</td><td>0.3086</td></tr><tr><td>VideoTetris [48]</td><td>0.6211</td><td>0.0104</td><td>0.4832</td><td>0.2249</td><td>0.4939</td><td>0.6578</td><td>0.3467</td></tr><tr><td>VideoCrafter2 [5]</td><td>0.6182</td><td>0.0103</td><td>0.4838</td><td>0.2259</td><td>0.5030</td><td>0.6365</td><td>0.3330</td></tr><tr><td>Open-Sora 1.2 [68]</td><td>0.5639</td><td>0.0189</td><td>0.5063</td><td>0.2468</td><td>0.4833</td><td>0.5039</td><td>0.3719</td></tr><tr><td>CogVideoX-5B [64]</td><td>0.6164</td><td>0.0219</td><td>0.5172</td><td>0.2658</td><td>0.5333</td><td>0.6069</td><td>0.3706</td></tr><tr><td>AnimateLCM* [53]</td><td>0.6746</td><td>0.0096</td><td>0.4651</td><td>0.2208</td><td>0.3425</td><td>0.4504</td><td>0.2494</td></tr><tr><td>+ NoisEasier (Ours)</td><td>0.7807 ↑0.1061</td><td>0.0183 ↑0.0087</td><td>0.5113 ↑0.0462</td><td>0.2281 ↑0.0073</td><td>0.4362 ↑0.0937</td><td>0.5365 ↑0.0861</td><td>0.2567 ↑0.0073</td></tr><tr><td>T2V-Turbo (MS)* [27]</td><td>0.7313</td><td>0.0124</td><td>0.4620</td><td>0.2306</td><td>0.4923</td><td>0.5822</td><td>0.3030</td></tr><tr><td>+ NoisEasier (Ours)</td><td>0.8375 ↑0.1062</td><td>0.0147 ↑0.0023</td><td>0.5259 ↑0.0639</td><td>0.2502 ↑0.0196</td><td>0.6477 ↑0.1554</td><td>0.6743 ↑0.0921</td><td>0.3983 ↑0.0953</td></tr><tr><td>T2V-Turbo (VC2)* [27]</td><td>0.7852</td><td>0.0113</td><td>0.5079</td><td>0.2222</td><td>0.6241</td><td>0.7017</td><td>0.2900</td></tr><tr><td>+ NoisEasier (Ours)</td><td>0.8737 ↑0.0885</td><td>0.0097 ↓0.0016</td><td>0.5660</td><td>↑0.0581 0.2343 ↑0.0121</td><td>0.7524 ↑0.1283</td><td>0.7915 ↑0.0898</td><td>0.3753 ↑0.0853</td></tr></table>

CompBench focuses on compositional reasoning in video generation and evaluates seven categories, including consistent/dynamic attribute binding, spatial relationships, action/motion binding object interactions, and numeracy.

## 4.1 Main Results

We first report quantitative results on VBench, comparing against both closedsource models and open-source baselines. As shown in Table 1, NoisEasier consistently improves VCM-based generators across diferent backbones. On AnimateLCM, NoisEasier increases Overall Consistency from 25.74% to 28.79%, with notable gains in compositional dimensions such as Multiple Objects and Spatial Relation. Similar trends are observed on T2V-Turbo, where the improvements are even more pronounced. For example, Multiple Objects increases from 54.50% to 76.62% on T2V-Turbo (MS) and from 54.33% to 79.18% on T2V-Turbo (VC2), while Spatial Relation improves by 12.66% on the VC2 variant. Although Motion Smoothness decreases slightly for the T2V-Turbo variants, the consistent increase in Dynamic Degree suggests that NoisEasier produces richer motion dynamics while largely preserving temporal coherence.

NoisEasier also yields consistent gains on T2V-CompBench across diverse compositional dimensions, as shown in Table 2. On AnimateLCM, it substantially improves Attribute/Action Binding while also boosting Spatial relationship and Interaction. Similar trends hold for T2V-Turbo, where Consist. Attribute Binding increases to 0.8375 (MS) and 0.8737 (VC2), and Action binding improves notably on both variants. Importantly, Numeracy also sees sizable gains (e.g., 0.0953 on MS), indicating improved robustness in reasoning about discrete object counts. These results further support the efectiveness of our negativeaware reward calibration in providing more precise compositional feedback.

Table 3: The efect of reward models for NoisEasier on VBench. The best score among individual reward models is marked in green while the second-best is in yellow .  
![](images/af29e0abb51aff240752d62cb1424cc15eda55c7490b97befff38d5af9762ab2.jpg)

![](images/8635ba6831b21be2883d0c3c0ff6d69f0503b4650f12551fe9b1251a93426c41.jpg)

![](images/c33dc612b70def10649cc72ec50ee2909acc91a23769989af1658478050ec66d.jpg)

![](images/df7fc565e72cdc77dc3fdec7e066121de7e58100bd95b9640849b479e76751b0.jpg)  
Fig. 3: Reward curve during noise optimization. The red solid line represents full noise optimization, while the blue dotted line denotes initial noise optimization. All scores are scale-normalized.

## 4.2 Ablation Study

Contribution of each reward model. As shown in Table 3, dif ferent reward models emphasize diferent aspects of generation. Im ageReward and ViCLIP contribute most to semantic alignment, but optimizing semantic rewards alone tends to yield overly static videos, reflected by reduced Dynamic Degree. In contrast, the motion reward alone increases temporal dy-

<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>T2V-Turbo (MS)*</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>+ HPSv2</td><td>27.85</td><td>64.66</td><td>90.20</td><td>46.21</td><td>95.19</td><td>54.72</td></tr><tr><td>+ ImageReward</td><td>28.24</td><td>73.78</td><td>90.27</td><td>47.16</td><td>95.36</td><td>61.11</td></tr><tr><td>+ ViCLIP</td><td>36.21</td><td>68.05</td><td>91.19</td><td>41.92</td><td>94.98</td><td>63.89</td></tr><tr><td>+ Motion</td><td>26.05</td><td>47.68</td><td>87.32</td><td>42.07</td><td>95.59</td><td>85.56</td></tr><tr><td>All w/o negatives</td><td>31.74</td><td>75.82</td><td>93.29</td><td>47.95</td><td>95.38</td><td>73.33</td></tr><tr><td>All (NoisEasier)</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr></table>

namics and motion coherence, yet degrades semantic metrics, highlighting the need to balance complementary objectives rather than relying on a single signal. Figure 4a further explains the dynamic term in our motion reward: by subtracting global motion in Eq. 5, the resulting localized motion suppresses camera shifts while emphasizing subject movement. Combining all rewards yields complementary gains across semantic and motion dimensions, and negative-aware calibration further strengthens compositional alignment, with only a slight decrease in motion-related scores since it primarily reshapes VLM-based objectives.

Initial noise vs. Full trajectory optimization. To quantify the benefit of full noise optimization, we compare against a variant that updates only the initial latent noise. As shown in Table 4, full

Table 4: Initial vs. full noise optimization.
<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>AnimateLCM+Init.</td><td>26.76</td><td>43.28</td><td>85.61</td><td>44.52</td><td>98.36</td><td>30.83</td></tr><tr><td>AnimateLCM+Full.</td><td>28.79</td><td>44.04</td><td>88.22</td><td>46.70</td><td>98.57</td><td>30.56</td></tr><tr><td>T2V-Turbo (MS)+Init.</td><td>29.67</td><td>72.58</td><td>90.03</td><td>44.10</td><td>95.58</td><td>66.94</td></tr><tr><td>T2V-Turbo (MS)+Full.</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr><tr><td>T2V-Turbo (VC2)+Init.</td><td>29.97</td><td>75.41</td><td>92.98</td><td>45.15</td><td>96.16</td><td>60.83</td></tr><tr><td>T2V-Turbo (VC2)+Full.</td><td>31.95</td><td>79.18</td><td>93.63</td><td>52.61</td><td>96.10</td><td>59.17</td></tr></table>

trajectory optimization provides additional control beyond initial-noise-only refinement, with the largest gains on compositional metrics such as color and spatial relations. Interestingly, motion-related dimensions show smaller gains or

Black swan gliding past a green lily pad

![](images/7ce697c54cdbc9f628ff3a89d9dc15c3977b4a2b2a2a6fa3b34dbd8abc3fcf44.jpg)  
(a) Global vs. localized motion.

A yellow teddy bear has breakfast from a blue bowl, sitting on the dining table  
![](images/4a158aae3b70b85e219ad83cb87e1a9edc5c1e0cdcae35321c53b3f20876bf98.jpg)  
(b) Initial noise optimization vs. Full noise optimization.  
Fig. 4: The efect of (a) motion reward; and (b) noise optimization formulation.

occasional advantages for initial noise optimization, suggesting lower sensitivity to intermediate-noise refinement. More results on T2V-CompBench can be found in Table 11. Figure 3 further demonstrates that jointly optimizing the full noise trajectory accelerates reward convergence. We also provide a qualitative example in Figure 4b: the initial noise mainly controls global appearance and coarse layout, whereas intermediate perturbations refine finer structures and stylistic details, resulting in sharper contrast and more visually pleasing outputs. Overall, these results validate the efectiveness of full trajectory optimization for improving both quantitative performance and visual quality.

## Comparison of diferent nega-

tive prompts. We compare LLMgenerated negatives against randomly sampled negatives based on part-of-speech (POS) perturbations. Specifically, we use $\mathrm { \ s p a C y ^ { 1 } }$ to identify key primitives (verbs,

Table 5: Efect of diferent negative prompt construction.
<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>NoisEaiser w/o neg.</td><td>31.74</td><td>75.82</td><td>93.29</td><td>47.95</td><td>95.38</td><td>73.33</td></tr><tr><td>Random sample</td><td>31.83</td><td>75.41</td><td>93.14</td><td>46.39</td><td>95.46</td><td>71.11</td></tr><tr><td>Gemini-2.5 Flash</td><td>32.00</td><td>76.55</td><td>93.81</td><td>48.72</td><td>95.53</td><td>74.44</td></tr><tr><td>GPT-4o (Ours)</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr></table>

nouns, adjectives, adpositions, and numerals) and randomly replace them with alternatives drawn from the test prompt suite. As shown in Table 5, incorporating negative prompts consistently improves performance over optimizing with the positive prompt alone. Overall Consistency increases slightly across all strategies, with the strongest gains achieved by LLM-constructed negatives. Specifically, GPT-4o negatives yield the best Color and Spatial Relation scores, suggesting that semantically targeted negatives provide more informative contrast than random perturbations. By contrast, POS-based random negatives ofer little improvement and even reduce Spatial Relation, indicating that arbitrary replacements are insuficient for efective calibration. Motion Smoothness remains essentially unchanged across variants, while Dynamic Degree changes only modestly, consistent with our design that negative-aware calibration primarily sharpens semantic and attribute alignment without degrading motion.

Comparison of test-time scaling methods. We further compare NoisEasier with two canonical test-time scaling (TTS) methods: Best-of-n sampling (BoS) and prompt enhancement (PE). Under the same wall-clock time budget,

Table 6: Performance of diferent test-time scaling methods.
<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>T2V-Turbo (MS)</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>+ NoisEasier</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr><tr><td>+ Best-of-n sampling</td><td>29.39</td><td>80.06</td><td>91.43</td><td>48.54</td><td>95.52</td><td>79.44</td></tr><tr><td>+ PE (GPT-4o)</td><td>28.70</td><td>71.95</td><td>87.47</td><td>47.58</td><td>95.89</td><td>70.56</td></tr><tr><td>+ PE (Gemini-2.5 Flash)</td><td>28.54</td><td>72.13</td><td>86.54</td><td>51.13</td><td>95.80</td><td>70.83</td></tr></table>

BoS samples multiple seeds and selects the highest-reward video, while PE uses an LLM to iteratively rewrite the prompt (with fixed noise) and chooses the best result. As shown in Table 6, NoisEasier achieves the highest Overall Consistency and Color scores, indicating that gradient-based noise optimization is efective for semantic alignment and attribute binding. In contrast, BoS attains the best Dynamic Degree and Multiple Objects, suggesting that strong motion patterns and complex multi-object layouts are largely driven by the global structure induced by the initial noise. PE slightly improves Motion Smoothness but substantially reduces Color binding, implying that enriched prompts may distract the model from precise attribute alignment while stabilizing cross-frame predictions. Interestingly, Gemini-based PE achieves the best Spatial Relation score, suggesting that some LLMs rewrite prompts with clearer spatial expressions. Overall, diferent TTS strategies favor diferent aspects of quality, and NoisEasier ofers a complementary path for improving semantic and attribute-level alignment.

Comparison with reward finetuning. Since NoisEasier optimizes rewards at inference time, we also examine whether incorporating the same rewards directly

Table 7: Comparison with reward fine-tuning.
<table><tr><td>Models</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>T2V-Turbo (MS)</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>+ reward fine-tuning (RFT)</td><td>28.58</td><td>73.49</td><td>90.95</td><td>51.16</td><td>98.65</td><td>74.72</td></tr><tr><td>+ NoisEasier</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr><tr><td>+ RFT + NoisEasier</td><td>31.91</td><td>84.56</td><td>94.72</td><td>54.71</td><td>98.65</td><td>77.50</td></tr></table>

into the model parameters yields comparable benefits. To this end, we perform reward fine-tuning (RFT) using LoRA on the T2V-Turbo backbone under the same mixed reward objectives. We follow the default LoRA configuration of T2V-Turbo and fine-tune on the full VBench prompt suite (∼1k prompts) for 10k steps, after which further training begins to exhibit overfitting. As shown in Table 7, both approaches improve over the baseline but with diferent strengths. RFT mainly improves motion-related metrics such as Motion Smoothness and Dynamic Degree, whereas NoisEasier yields larger gains on compositional dimensions including Multiple Objects and Color. Notably, applying NoisEasier on top of the RFT model further improves most metrics, indicating that reward signals are not fully amortized into the model parameters during ofline tuning. This suggests that noise optimization provides a complementary test-time scaling mechanism even when the backbone has already been reward-tuned.

## 4.3 Eficiency Analysis

Inference Eficiency. Table 8 reports the peak memory and runtime of 25-step optimization under identical settings on a single RTX 6000 Ada GPU. Compared with optimizing only the initial latent, full trajectory optimization incurs negligible additional overhead, increasing runtime by less than 1% while leaving peak memory virtually unchanged. This is because both methods backpropagate through the same denoising computation graph; our method only introduces additional optimized noise variables without requiring extra forward or backward passes. The dominant computational cost instead comes from multi-step backpropagation through the generator and diferentiable reward models. On a single H100 GPU, the runtime can be further reduced to 31.25 s (MS), 72.92 s (VC2), and 105.37 s (AnimateLCM).

Table 8: Inference Eficiency of 25-step test-time noise optimization.
<table><tr><td></td><td>Model Peak VRAM</td><td>Time</td><td></td><td>Model Peak VRAM</td><td>Time</td><td>|Model</td><td>Peak VRAM</td><td>Time</td></tr><tr><td>MS</td><td>8.23 GB</td><td>1.04s</td><td>VC2</td><td>8.76 GB</td><td>1.41s</td><td>Ani-LCM</td><td>8.78 GB</td><td>2.46s</td></tr><tr><td>+Init.</td><td>22.93 GB</td><td>44.63s</td><td>+Init.</td><td>33.24 GB</td><td>115.19s</td><td>+Init.</td><td>40.53 GB</td><td>176.71s</td></tr><tr><td>+Full.</td><td>22.95 GB</td><td>45.28s</td><td>+Full.</td><td>33.27 GB</td><td>115.63s</td><td>+Full.</td><td>40.53 GB</td><td>177.86s</td></tr></table>

Quality–Cost Trade-of. Figure 5 compares the quality–cost tradeof between initial-noise and fulltrajectory optimization on the T2V-Turbo (MS) backbone using an RTX 6000 Ada GPU, with the Overall Consistency dimension of VBench as the evaluation metric. As the optimization budget increases, both methods improve semantic alignment, but fulltrajectory optimization consistently traces a superior Pareto frontier. No-

![](images/afdfd61230c86fdfdb4220569a738a736e816d06ed93fa390c89bdad02813790.jpg)  
Fig. 5: Quality-Cost Pareto on VBench.

tably, the advantage becomes more pronounced with additional optimization steps, indicating that jointly refining all intermediate perturbations utilizes the additional computation more efectively than optimizing only the initial latent. We adopt 25 optimization steps as the default setting, providing a favorable balance between quality and inference cost.

## 4.4 Human Evaluation.

We conduct a user study to further assess the practical benefits of NoisEasier. We uniformly sample 10 prompts from each of the seven T2V-CompBench dimensions, resulting in 70 test prompts in total. We compare NoisEasier against three baselines: T2V-Turbo (VC2),

<table><tr><td rowspan=1 colspan=1>61.4%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38.6%</td><td rowspan=1 colspan=1>60.3%</td><td rowspan=1 colspan=1>39.7%T2</td><td rowspan=1 colspan=1>V-Turbo (VC2)</td></tr><tr><td rowspan=2 colspan=1>60.9%</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>39.1%</td><td rowspan=2 colspan=1>54.6%</td><td rowspan=2 colspan=1>45.4%</td><td></td></tr><tr><td rowspan=1 colspan=1>CogVideoX-2B</td></tr><tr><td rowspan=1 colspan=1>58.6%</td><td rowspan=1 colspan=2>41.4%</td><td rowspan=1 colspan=1>52.0%</td><td rowspan=1 colspan=1>48.0%</td><td rowspan=1 colspan=1>Wan2.1-1.3B</td></tr></table>

Fig. 6: User study on T2V-CompBench.

CogVideoX-2B [64], and Wan2.1-1.3B [52]. For each prompt, we use the same random seed across models. Each comparison is judged by five Amazon Mechanical Turk workers, who select the candidate with better semantic alignment and visual quality. As shown in Figure 6, NoisEasier achieves a 61.4% win rate over its backbone, indicating that test-time noise optimization improves semantic faithfulness. Moreover, NoisEasier remains competitive against stronger T2V baselines, receiving around 61% preference on alignment and 52% on visual quality compared to CogVideoX-2B and Wan2.1-1.3B, despite operating as a lightweight add-on rather than retraining a larger T2V model.

![](images/4efbc3e960a3f2d6778105cd381df473e98f8bc3247a6ccc5cc4eb7131ac6a51.jpg)  
Fig. 7: Efect of negative-aware reward calibration at diferent noise optimization steps.

## 4.5 Qualitative Analysis

We present qualitative comparisons in Figure 7 to illustrate the efect of our optimization framework. The baseline captures the rough semantics of the prompt but exhibits compositional errors such as incorrect color binding and inconsistent object motion. Noise optimization progressively improves both spatial and temporal coherence: the taxi becomes a coherent object with stable right-toleft motion, and attribute separation becomes clearer, although minor artifacts remain. With negative-aware reward calibration, semantic disentanglement is further enhanced, i.e., the facade becomes consistently yellow while the green taxi is preserved with minimal leakage. Additional examples are provided in Figure 1 and the supplementary material.

While efective, we also observe two common failure modes, as shown in Figure 8. First, although NoisEasier improves visual fidelity and semantic alignment, it still struggles with temporal state transitions. In the melting-ice example, the generated video preserves the appearance of the ice cube but fails to capture its gradual transition into water. We attribute this limitation to existing reward models, which primarily assess frame-level semantics and provide only weak supervision for long-term temporal evolution. Second, NoisEasier remains challenged by fine-grained object counting. In the example on the right, the model fails to satisfy the exact numeracy constraint despite improving overall semantic consistency. This suggests that current VLM-based rewards are insuficiently sensitive to precise object counting and complex compositional reasoning.

“Clear ice cube melts into shapeless water”  
![](images/fb8ae8e6be9ed4e7e3203a1863f516f8a3f8090a893824bcaf517f10133f8f12.jpg)  
“Three apples hang heavy on the tree branch, and three birds perch nearby.”

![](images/3143eea65c2a6093ccee0b52085316509756f804798b2fd494d7d4cf9f1334a9.jpg)  
Fig. 8: Two failure cases of NoisEasier: temporal state transition (left) and object counting (right).

These observations indicate that the performance of test-time optimization is fundamentally bounded by the quality of the underlying reward models. Future work may benefit from incorporating structured reward signals, such as object detection, visual grounding, or temporal-state verification models, to provide more reliable supervision for compositional video generation.

## 5 Conclusion

In this work, we propose NoisEasier, a test-time scaling framework that improves text-to-video generation through diferentiable reward-guided noise optimization without model fine-tuning. By combining eficient optimization with robust reward supervision, NoisEasier consistently enhances compositional alignment while preserving visual fidelity. We further show that jointly optimizing intermediate noises provides more efective control over the denoising trajectory than optimizing only the initial latent, yielding faster reward convergence and stronger compositional alignment with negligible additional computational cost. Moreover, NoisEasier complements ofline reward fine-tuning, demonstrating that test-time optimization remains efective even after model alignment. Overall, our findings establish noise optimization as a practical test-time scaling paradigm for improving alignment in video generative models.

## Acknowledgements

This work was partially supported by the Ofice of Naval Research (ONR) grant (N00014-23-1-2417), and the Army Research Ofice (ARO) grant (W911NF-24- 1-0385). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of ONR or ARO.

## References

1. Ben-Hamu, H., Puny, O., Gat, I., Karrer, B., Singer, U., Lipman, Y.: D-flow: Diferentiating through flows for controlled generation. arXiv preprint arXiv:2402.14017 (2024)

2. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training difusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)

3. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators. Tech. rep., OpenAI (2024), https://openai. com/research/video-generation-models-as-world-simulators, technical Report, OpenAI

4. Capcut: Dreamina. https://dreamina.capcut.com/ai-tool/home (2024)

5. Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7310–7320 (2024)

6. Chen, S.X., Vaxman, Y., Ben Baruch, E., Asulin, D., Moreshet, A., Lien, K.C., Sra, M., Sen, P.: Tino-edit: Timestep and noise optimization for robust difusion-based image editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6337–6346 (2024)

7. Chen, T., Xu, B., Zhang, C., Guestrin, C.: Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174 (2016)

8. Cheng, H., Dong, Q., Peng, L., Sha, Z., Feng, W., Xie, J., Song, Z., Wen, S., He, X., Wu, B.: Discriminator-free direct preference optimization for video difusion. arXiv preprint arXiv:2504.08542 (2025)

9. Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017)

10. Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning difusion models on diferentiable rewards. arXiv preprint arXiv:2309.17400 (2023)

11. Ding, X., Zhang, K., Han, J., Hong, L., Xu, H., Li, X.: Pami-vdpo: Mitigating video hallucinations by prompt-aware multi-instance video preference learning. arXiv preprint arXiv:2504.05810 (2025)

12. Doveh, S., Arbelle, A., Harary, S., Herzig, R., Kim, D., Cascante-Bonilla, P., Alfassy, A., Panda, R., Giryes, R., Feris, R., et al.: Dense and aligned captions (dac) promote compositional reasoning in vl models. Advances in Neural Information Processing Systems 36, 76137–76150 (2023)

13. Eyring, L., Karthik, S., Roth, K., Dosovitskiy, A., Akata, Z.: Reno: Enhancing one-step text-to-image models through reward-based noise optimization. Advances in Neural Information Processing Systems 37, 125487–125519 (2024)

14. Fan, W., Si, C., Song, J., Yang, Z., He, Y., Zhuo, L., Huang, Z., Dong, Z., He, J., Pan, D., et al.: Vchitect-2.0: Parallel transformer for scaling up video difusion models. arXiv preprint arXiv:2501.08453 (2025)

15. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Dpok: Reinforcement learning for fine-tuning text-to-image difusion models. Advances in Neural Information Processing Systems 36, 79858–79885 (2023)

16. Google: Veo-3 technical report. Tech. rep., Google DeepMind (May 2025)

17. Guo, X., Liu, J., Cui, M., Li, J., Yang, H., Huang, D.: Initno: Boosting textto-image difusion models via initial noise optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9380– 9389 (2024)

18. Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., Dai, B.: Animatedif: Animate your personalized text-to-image difusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023)

19. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al.: Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103 (2024)

20. He, X., Jiang, D., Zhang, G., Ku, M., Soni, A., Siu, S., Chen, H., Chandra, A., Jiang, Z., Arulraj, A., et al.: Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252 (2024)

21. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024)

22. Karunratanakul, K., Preechakul, K., Aksan, E., Beeler, T., Suwajanakorn, S., Tang, S.: Optimizing difusion noise can serve as universal motion priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1334–1345 (2024)

23. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 36652–36663 (2023)

24. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)

25. Kuaishou: Kling. https://kling.kuaishou.com/ (2024)

26. Li, J., Feng, W., Chen, W., Wang, W.Y.: Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027 (2024)

27. Li, J., Feng, W., Fu, T.J., Wang, X., Basu, S., Chen, W., Wang, W.Y.: T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. arXiv preprint arXiv:2405.18750 (2024)

28. Li, J., Long, Q., Zheng, J., Gao, X., Piramuthu, R., Chen, W., Wang, W.Y.: T2vturbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. arXiv preprint arXiv:2410.05677 (2024)

29. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888–12900. PMLR (2022)

30. Li, S., Le, H., Xu, J., Salzmann, M.: Enhancing compositional text-to-image generation with reliable random seeds. In: The Thirteenth International Conference on Learning Representations (2024)

31. Liang, Y., He, J., Li, G., Li, P., Klimovskiy, A., Carolan, N., Sun, J., Pont-Tuset, J., Young, S., Yang, F., et al.: Rich human feedback for text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19401–19411 (2024)

32. Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Qin, W., Xia, M., et al.: Improving video generation with human feedback. arXiv preprint arXiv:2501.13918 (2025)

33. Liu, R., Wu, H., Ziqiang, Z., Wei, C., He, Y., Pi, R., Chen, Q.: Videodpo: Omni-preference alignment for video difusion generation. arXiv preprint arXiv:2412.14167 (2024)

34. Novack, Z., McAuley, J., Berg-Kirkpatrick, T., Bryan, N.: Ditto-2: Distilled difusion inference-time t-optimization for music generation. arXiv preprint arXiv:2405.20289 (2024)

35. Novack, Z., McAuley, J., Berg-Kirkpatrick, T., Bryan, N.J.: Ditto: Diffusion inference-time t-optimization for music generation. arXiv preprint arXiv:2401.12179 (2024)

36. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)

37. PixVerse: Pixverse. https://app.pixverse.ai (2024)

38. Prabhudesai, M., Goyal, A., Pathak, D., Fragkiadaki, K.: Aligning text-to-image difusion models with reward backpropagation (2023)

39. Prabhudesai, M., Mendonca, R., Qin, Z., Fragkiadaki, K., Pathak, D.: Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737 (2024)

40. Qi, Z., Bai, L., Xiong, H., Xie, Z.: Not all noises are created equally: Difusion noise selection and optimization. arXiv preprint arXiv:2407.14041 (2024)

41. Qing, Z., Zhang, S., Wang, J., Wang, X., Wei, Y., Zhang, Y., Gao, C., Sang, N.: Hierarchical spatio-temporal decoupling for text-to-video generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6635–6645 (2024)

42. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

43. Runway: Introducing gen-3 alpha: A new frontier for video generation. https: //runwayml.com/blog/introducing-gen-3-alpha/ (2024)

44. Sun, K., Huang, K., Liu, X., Wu, Y., Xu, Z., Li, Z., Liu, X.: T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. arXiv preprint arXiv:2407.14505 (2024)

45. Sun, K., Huang, K., Liu, X., Wu, Y., Xu, Z., Li, Z., Liu, X.: T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 8406–8416 (2025)

46. Tang, Z., Peng, J., Tang, J., Hong, M., Wang, F., Chang, T.H.: Inferencetime alignment of difusion models with direct noise optimization. arXiv preprint arXiv:2405.18881 (2024)

47. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419. Springer (2020)

48. Tian, Y., Yang, L., Yang, H., Gao, Y., Deng, Y., Wang, X., Yu, Z., Tao, X., Wan, P., ZHANG, D., et al.: Videotetris: Towards compositional text-to-video generation. Advances in Neural Information Processing Systems 37, 29489–29513 (2024)

49. Trager, M., Perera, P., Zancato, L., Achille, A., Bhatia, P., Soatto, S.: Linear spaces of meanings: compositional structures in vision-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15395–15404 (2023)

50. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Difusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8228–8238 (2024)

51. Wallace, B., Gokul, A., Ermon, S., Naik, N.: End-to-end difusion latent optimization improves classifier guidance. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7280–7290 (2023)

52. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

53. Wang, F.Y., Huang, Z., Bian, W., Shi, X., Sun, K., Song, G., Liu, Y., Li, H.: Animatelcm: Computation-eficient personalized style video generation without personalized video data. In: SIGGRAPH Asia 2024 Technical Communications, pp. 1–5 (2024)

54. Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope textto-video technical report. arXiv preprint arXiv:2308.06571 (2023)

55. Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang, Y., Yang, C., He, Y., Yu, J., Yang, P., et al.: Lavie: High-quality video generation with cascaded latent difusion models. International Journal of Computer Vision 133(5), 3059–3078 (2025)

56. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942 (2023)

57. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)

58. Wu, X., Sun, K., Zhu, F., Zhao, R., Li, H.: Human preference score: Better aligning text-to-image models with human preference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2096–2105 (2023)

59. Wu, X., Huang, S., Wang, G., Xiong, J., Wei, F.: Boosting text-to-video generative model with mllms feedback. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)

60. Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., et al.: Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059 (2024)

61. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)

62. Xu, K., Zhang, L., Shi, J.: Good seed makes a good crop: Discovering secret seeds in text-to-image difusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 3024–3034. IEEE (2025)

63. Yang, K., Tao, J., Lyu, J., Ge, C., Chen, J., Shen, W., Zhu, X., Li, X.: Using human feedback to fine-tune difusion models without any reward model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8941–8951 (2024)

64. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)

65. Yuan, H., Zhang, S., Wang, X., Wei, Y., Feng, T., Pan, Y., Zhang, Y., Liu, Z., Albanie, S., Ni, D.: Instructvideo: Instructing video difusion models with human feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6463–6474 (2024)

66. Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., Zou, J.: When and why vision-language models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936 (2022)

67. Zhang, D.J., Wu, J.Z., Liu, J.W., Zhao, R., Ran, L., Gu, Y., Gao, D., Shou, M.Z.: Show-1: Marrying pixel and latent difusion models for text-to-video generation. International Journal of Computer Vision pp. 1–15 (2024)

68. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404 (2024)

69. Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., Feng, J.: Magicvideo: Eficient video generation with latent difusion models. arXiv preprint arXiv:2211.11018 (2022)

70. Zhou, Z., Shao, S., Bai, L., Zhang, S., Xu, Z., Han, B., Xie, Z.: Golden noise for difusion models: A learning framework. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17688–17697 (2025)

# NoisEasier: Test-Time Noise Optimization for Text-to-Video Generation Supplementary Material

## A More Implementation Details

Considering that diferent reward models produce outputs on diferent scales, we assign them separate weights α in Eq. 7. Specifically, HPSv2 and ViCLIP measure cosine similarity between text and video, with most values falling in the range of [0.2, 0.4], so their weights are set to 2. The raw scores of ImageReward lie within [−2, +2]; we normalize them to [0, 1] using min-max scaling to ensure non-negative rewards, and assign a weight of 1. For the motion reward, both the dynamic and smoothness terms are in [0, 1], with $\lambda = 0 . 5$ and an overall weight of 1. We use GPT-4o to generate 8 negative samples per prompt. For negative-aware reward calibration, we set $m = 0 . 0 2$ and $\tau = 0 . 0 1$ for $\mathrm { H P S v 2 }$ and $m = \tau = 0 . 0 5$ for ViCLIP. We keep the original ImageReward since it is not based on video-text contrastive pretraining, and this also helps save memory and maintain inference speed. On both benchmarks, we empirically apply the above default settings for the two baseline models, which already demonstrate strong performance. Our code will be open-source upon acceptance.

To construct hard negatives for reward calibration, we designed an automated pipeline using the OpenAI API. Figure 9 illustrates the prompt template along with several generated examples. Given an input prompt from a specific dimension, the pipeline iterates through each prompt and queries GPT-4o with explicit instructions to generate syntactically similar but semantically diferent variants. Negatives are produced by modifying only one category at a time $( e . g .$ subject, object, or action), ensuring that changes reflect clear semantic shifts rather than trivial rephrasings or synonyms. For each prompt, the model returns a fixed number of negatives (eight by default) in strict JSON format, which are automatically parsed, validated, and saved into a consolidated JSON file. This process yields a large pool of structured negatives that are consistent in format and directly usable for contrastive learning and evaluation.

## B More Ablation Studies

Diferent Combination of Reward Models. In this section, we present all the combinations of the reward models in Table 9. Among single-reward settings, ViCLIP delivers the strongest gains in overall consistency and color fidelity, highlighting its strength in semantic alignment and global appearance, whereas ImageReward performs best on multiple objects and spatial relations, reflecting its training on fine-grained visual-text cues. The motion reward is more distinctive: although it trails others on appearance-related metrics, it substantially boosts dynamic degree and smoothness, underscoring its importance in preventing static generations. When combining rewards, synergies are clear: pairing the motion reward with ViCLIP or ImageReward strikes a strong balance between temporal dynamics and spatial fidelity, while conflicts also appear; for instance, HPSv2 with other rewards can suppress motion expressiveness even as it improves color quality. Notably, the best three-reward combinations (e.g., ImageReward + ViCLIP + Motion) nearly match the four-reward setting. Overall, the results show that diferent reward models specialize in complementary dimensions, and that careful combinations are crucial to balancing fidelity, compositionality, and temporal quality.

Table 9: All Combinations of reward models applied to NoisEasier (MS) on VBench. The best score for each number of reward models is highlighted in green , and the second-best in yellow . \* indicates integrating negative-aware reward calibration.
<table><tr><td>HPSv2</td><td>ImgReward</td><td>ViCLIP</td><td>Motion</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>X</td><td>x</td><td>x</td><td>X</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>27.85</td><td>64.66</td><td>90.20</td><td>46.21</td><td>95.19</td><td>54.72</td></tr><tr><td>X</td><td>V</td><td>x</td><td>x</td><td>28.24</td><td>73.78</td><td>90.27</td><td>47.16</td><td>95.36</td><td>61.11</td></tr><tr><td>x</td><td>x</td><td>√</td><td>x</td><td>36.21</td><td>68.05</td><td>91.19</td><td>41.92</td><td>94.98</td><td>63.89</td></tr><tr><td>X</td><td>x</td><td>x</td><td>√</td><td>26.05</td><td>47.68</td><td>87.32</td><td>42.07</td><td>95.59</td><td>85.56</td></tr><tr><td>√</td><td>V</td><td>x</td><td>x</td><td>28.40</td><td>74.02</td><td>89.35</td><td>47.18</td><td>95.53</td><td>55.28</td></tr><tr><td>√</td><td>x</td><td>√</td><td>x</td><td>34.57</td><td>70.78</td><td>92.63</td><td>43.87</td><td>94.77</td><td>58.89</td></tr><tr><td></td><td>x</td><td>x</td><td>√</td><td>27.28</td><td>58.28</td><td>89.31</td><td>44.09</td><td>95.52</td><td>76.67</td></tr><tr><td>X</td><td>V</td><td>√</td><td>x</td><td>32.31</td><td>71.75</td><td>90.69</td><td>46.89</td><td>95.31</td><td>62.78</td></tr><tr><td>x</td><td></td><td>x</td><td>√</td><td>28.03</td><td>73.26</td><td>89.34</td><td>44.43</td><td>95.43</td><td>74.72</td></tr><tr><td>X</td><td>x</td><td>√</td><td>√</td><td>34.61</td><td>64.04</td><td>89.28</td><td>43.53</td><td>94.95</td><td>80.28</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>31.95</td><td>74.09</td><td>92.01</td><td>46.78</td><td>95.30</td><td>59.44</td></tr><tr><td>√</td><td>V</td><td>x</td><td>V</td><td>28.20</td><td>73.16</td><td>91.50</td><td>46.36</td><td>95.38</td><td>75.28</td></tr><tr><td>V</td><td>x</td><td>√</td><td></td><td>33.46</td><td>71.39</td><td>91.54</td><td>47.26</td><td>95.19</td><td>76.11</td></tr><tr><td>X</td><td>√</td><td>√</td><td>√</td><td>31.96</td><td>74.82</td><td>92.22</td><td>47.49</td><td>95.35</td><td>75.56</td></tr><tr><td></td><td>V</td><td>√</td><td>√</td><td>31.74</td><td>75.82</td><td>93.29</td><td>47.95</td><td>95.38</td><td>73.33</td></tr><tr><td>√*</td><td>V</td><td>√*</td><td></td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr></table>

Weight Sensitivity of Reward Models. As discussed in Sec.,A, diferent reward models operate on diferent numerical scales, so we assign separate weights α to HPSv2, ImageReward, ViCLIP, and the motion reward. Among the evaluated dimensions, Overall Consistency mainly reflects video–text semantic alignment, whereas the remaining dimensions capture object- and motion-level properties. We observe that semantic alignment varies only moderately across settings, and most metrics remain within a narrow range. One exception is Dynamic Degree, which shows higher variance because VBench uses a RAFT-based binary test (above or below a motion threshold), so small motion changes can flip the score. In contrast, Motion Smoothness exhibits very low variance, since it is computed from interpolation error and most samples already have reasonably smooth motion. As a result, changing the reward weights has limited efect on this metric. In terms of trade-ofs, placing greater weight on HPSv2 and Vi-CLIP slightly reduces Dynamic Degree, whereas emphasizing ImageReward and motion improves Dynamic Degree at some cost to Spatial Relation. Our final configuration strikes a balanced compromise, achieving competitive semantic alignment and strong object- and motion-related scores, while overall variation across settings remains modest.

Table 10: Performance of NoisEasier (MS) on VBench under diferent reward weight combinations. The best performance across all combinations is shown in bold, while the worst performance is shaded in gray.
<table><tr><td>HPSv2</td><td>ImgReward</td><td>ViCLIP</td><td>Motion</td><td>Overall Consist.</td><td>Multiple Objects</td><td>Color</td><td>Spatial Relation.</td><td>Motion Smooth.</td><td>Dynamic Degree</td></tr><tr><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>27.40</td><td>54.50</td><td>89.90</td><td>46.62</td><td>95.51</td><td>70.83</td></tr><tr><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>31.16</td><td>75.08</td><td>91.16</td><td>46.27</td><td>95.55</td><td>70.56</td></tr><tr><td>1.0</td><td>2.0</td><td>1.0</td><td>2.0</td><td>30.10</td><td>73.11</td><td>91.53</td><td>48.95</td><td>95.43</td><td>70.83</td></tr><tr><td>5.0</td><td>1.0</td><td>5.0</td><td>1.0</td><td>32.76</td><td>75.43</td><td>92.88</td><td>47.64</td><td>95.35</td><td>65.00</td></tr><tr><td>2.0</td><td>2.0</td><td>1.0</td><td>1.0</td><td>29.91</td><td>75.06</td><td>93.01</td><td>49.95</td><td>95.48</td><td>70.00</td></tr><tr><td>2.0</td><td>1.0</td><td>1.0</td><td>2.0</td><td>30.55</td><td>71.07</td><td>93.56</td><td>47.44</td><td>95.45</td><td>74.72</td></tr><tr><td>1.0</td><td>2.0</td><td>2.0</td><td>1.0</td><td>31.45</td><td>76.25</td><td>93.06</td><td>46.01</td><td>95.25</td><td>70.28</td></tr><tr><td>1.0</td><td>1.0</td><td>2.0</td><td>2.0</td><td>31.90</td><td>72.01</td><td>93.15</td><td>44.86</td><td>95.33</td><td>77.50</td></tr><tr><td>2.0</td><td>1.0</td><td>2.0</td><td>1.0</td><td>32.05</td><td>76.62</td><td>94.38</td><td>48.93</td><td>95.29</td><td>71.39</td></tr></table>

Table 11: Initial noise vs. Full trajectory optimization on T2V-CompBench.
<table><tr><td>Models</td><td>Attr.</td><td>Consist Dynamic  $\mathrm { A t t r . }$ </td><td>Spatial</td><td>Motion</td><td></td><td></td><td>Action Interaction Numeracy</td></tr><tr><td>AnimateLCM+Init.</td><td>0.7326</td><td>0.0121</td><td>0.4899</td><td>0.2291</td><td>0.4027</td><td>0.4469</td><td>0.2503</td></tr><tr><td>AnimateLCM+Full.</td><td>0.7807</td><td>0.0183</td><td>0.5113</td><td>0.2281</td><td>0.4362</td><td>0.5365</td><td>0.2567</td></tr><tr><td>T2V-Turbo (MS)+Init.</td><td>0.8343</td><td>0.0094</td><td>0.5262</td><td>0.2370</td><td>0.5989</td><td>0.6243</td><td>0.3641</td></tr><tr><td>T2V-Turbo (MS)+Full.</td><td>0.8375</td><td>0.0147</td><td>0.5259</td><td>0.2502</td><td>0.6477</td><td>0.6743</td><td>0.3983</td></tr><tr><td>T2V-Turbo (VC2)+Init.</td><td>0.8524</td><td>0.0092</td><td>0.5434</td><td>0.2207</td><td>0.6961</td><td>0.7765</td><td>0.3781</td></tr><tr><td>T2V-Turbo (VC2)+Full.</td><td>0.8737</td><td>0.0097</td><td>0.5660</td><td>0.2343</td><td>0.7524</td><td>0.7915</td><td>0.3753</td></tr></table>

## C Additional Results on Modern Video Generators

Since NoisEasier performs iterative gradient-based optimization, directly applying it to large difusion backbones with long denoising chains, e.g., Wan2.1- 14B [52] or HunyuanVideo [24], remains computationally prohibitive. For example, generating a single 5-second video with Wan2.1-14B requires approximately 56 minutes and 69 GB peak VRAM on a single A100 GPU, making repeated forward/backward optimization impractical. To further evaluate the generality of NoisEasier, we additionally apply our framework to two recent distilled video generators, FastWan 2.1-1.3B<sup>2</sup> (81 frames, 3-step sampling) and LTX-Video-2Bdistilled [19] (121 frames, 10-step sampling).

Table 12: VBench Evaluation.
<table><tr><td>Model</td><td>Consist. Object Color</td><td></td><td></td></tr><tr><td>FastWan 2.1-1.3B</td><td>21.49</td><td>58.43</td><td>87.77</td></tr><tr><td>+ NoisEasier</td><td>23.71</td><td>68.09</td><td>90.01</td></tr></table>

Table 13: T2V-CompBench.
<table><tr><td>Model</td><td>Consist.</td><td>Action</td><td>Interaction</td></tr><tr><td>LTXV-2B-distilled</td><td>0.7702</td><td>0.3631</td><td>0.4135</td></tr><tr><td>+ NoisEasier</td><td>0.8081</td><td>0.4230</td><td>0.4931</td></tr></table>

“A cute fluffy panda eating Chinese food in a restaurant.”

![](images/0bdfd47722f57a61e8a625427ee8546a82cee4511584a530f5934dc78c4ecd6d.jpg)

“A blue car drives past a white picket fence on a sunny day”  
![](images/33c487baca7020720d161f212e195c6dc982b692382b17c13088c60496c43688.jpg)  
Fig. 10: Qualitative results on two modern fast generators.

Tables 12 and 13 show that NoisEasier consistently improves both FastWan and LTX-Video-2B-distilled under the default optimization setting. On Fast-Wan, NoisEasier improves Overall Consistency by 2.22% while also yielding substantial gains in object recognition and color consistency. Similarly, on LTX-Video-2B-distilled, NoisEasier consistently improves overall consistency, action understanding, and object interaction. Figure 10 further presents representative qualitative examples, demonstrating improved semantic alignment across diverse prompts. These preliminary results suggest that NoisEasier generalizes beyond the VCM backbones studied in the main paper and remains efective on modern distilled video generators.

## D More Visualization Results

We provide more qualitative examples from Figure 11 to Figure 14 to illustrate the efectiveness of our method.

## E Limitations and Future Work

As a test-time optimization framework, NoisEasier inevitably introduces additional inference overhead compared with vanilla sampling, making it less suitable for latency-sensitive applications. Moreover, it primarily improves semantic alignment rather than the intrinsic visual quality of the backbone, which remains bounded by the underlying generator and imperfect reward proxies. Our study also focuses on short, single-scene video generation; extending noise refinement to longer videos, more complex narratives, or higher resolutions remains challenging due to weaker reward signals and increased optimization cost. Future work will explore stronger video-level rewards and more eficient optimization strategies for long-horizon video generation.

![](images/01916b06035aa7ae26c32170762d12c38049b5ea55a88c7f45ab40330203a714.jpg)  
Fig. 9: Prompt template and generated negative samples of GPT-4o.

AnimateLCMNoisEasier

AnimateLCMNoisEasier

an orange vase.  
![](images/4aaf89970fce37ffa06c4017b122288328ffd9bf3e72bd57bc4e5253e80b81b5.jpg)  
Origami dancers in white paper, 3D render, on white background, studio shot, dancing modern dance.  
AnimateLCMNoisEasier

![](images/2f48ca47f6d774e9dc8f1c268d874ad178af62cd6edf64a33f630b034cfbe3eb.jpg)

a bottle on the left of a wine glass, front view  
![](images/e7a0496f534c304888521dd49b6af9ef70855e1c123aa9e7d5ade670ad943fce.jpg)

a car and a motorcycle  
![](images/12b0afde8810c1befb323f6a698cd0bbb35e1b590c7302ef43869b4e67d8cd13.jpg)  
Fig. 11: Qualitative comparison between AnimateLCM and NoisEasier on VBench.

A blue car drives past a white picket fence on a sunny day  
![](images/df8be334774722679c88b5961ab84ae6294aa50bc96ddc31d7e79583c7c6e30a.jpg)

Velvet ribbon tied on an iron fence  
![](images/09df74842948b610b257819055b4d6576effa6143cb23cf48813485afd8c1a55.jpg)

An elephant standing on the left of a rowboat in a small pond  
![](images/81f3e6d8bc80f53174613f064c6deba11aabaaa1e694ef4d49c311c5ea4539b3.jpg)

Five cups sit on a table, steaming with fresh coffee.  
![](images/d741414e4d9dae310a142c8f81b6a883b08068ae8601b0d3b17058fce1d910fd.jpg)  
Fig. 12: Qualitative comparison between AnimateLCM and NoisEasier on T2V-CompBench.

![](images/e7835035a6464200f7062f312dfcb593e3921358efa2e94efc74ccacc9bdef35.jpg)  
Fig. 13: Qualitative comparison between baseline results and NoisEasier on VBench. Left: T2V-Turbo (VC2). Right: T2V-Turbo (MS).

![](images/d3db04b4813d548b50051d99800f9b8c94b90a3e18d0bb65d63ef8e90c215c15.jpg)  
Fig. 14: Qualitative comparison between baseline results and NoisEasier on T2V-CompBench. Left: T2V-Turbo (VC2). Right: T2V-Turbo (MS).