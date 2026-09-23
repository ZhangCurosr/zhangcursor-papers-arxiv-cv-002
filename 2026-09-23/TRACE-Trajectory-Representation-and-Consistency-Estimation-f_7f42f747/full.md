# TRACE: Trajectory Representation and Consistency Estimation for AI-Generated Video Detection

Huangsen Cao<sup>1</sup>, Hongkang Chu<sup>2</sup>, Siyao Yu<sup>1</sup>, Xin Ding<sup>3</sup> Jianfeng Dong<sup>4</sup>, Yongwei Wang<sup>1</sup>

<sup>1</sup>Zhejiang University

<sup>2</sup>University of the Chinese Academy of Sciences

<sup>3</sup>Nanjing University of Information Science and Technology

<sup>4</sup>AZhejiang Gongshang University huangsen cao@zju.edu.cn, yongwei.wang@zju.edu.cn

## Abst<sub>r</sub>act

Recent advances in generative video models have enabled the synthesis of visually realistic content, posing significant challenges to synthetic video detection. Existing detectors often rely on appearance artifacts, semantic inconsistencies, and temporal patterns that may be generator-specific, limitating generalization to unseen synthesis models. We investigate whether responses to a pretrained generative model provide more transferable forensic cues. Our key observation is that real and AI-generated videos exhibit distinct velocity responses under a pretrained Flow Matching video model. This distinction persists when diferent pretrained video-generation back bones are used as probes, suggesting that velocity responses ofer transferable forensic signals beyond visual artificts. Motivated by this observation, we propose TRACE (Trajectory Representation and Consistency Estimation), a generation-process-aware framework for AI-generated video detection. TRACE leverages a pretrained video DiT as a velocity-field probe to extract representations at multiple flow time points, and models cross-frame consistency through velocity diferences between adjacent frames. We further introduce a Real-Centered Trajectory Optimization ob jective that encourages generator-invariant representation learning. Extensive experiments on AIGVDBench demonstrate that TRACE generalizes efectively across diverse generators, substantially outperforming prior state-of-the-art methods on unseen open- and closed-source video generation models.

## 1 Introduction

Recent advances in generative visual intelligence have enabled increasingly realistic and diverse visual synthesis. Difusion models (Esser et al., 2024; Gu et al., 2022) and Flow-Matching based models (Lipman et al., 2022) have demonstrated strong capabilities for generating videos from text, images, and existing videos (Ho et al., 2022; Peebles & Xie, 2023; Lipman et al., 2023; Wan Team, 2025). As synthesized quality improves, detecting AI-generated content becomes more challenging. Meanwhile, evolving of generative architectures continuously alters the artifacts present in synthesized videos, making it dificult to learn forensic cues that can generalize to unseen generators. These challenges motivate exploring forensic cues tied to the generative process than only to the rendered appearance.

Most existing detectors analyze directly on rendered video content. Image-level methods exploit local pixel dependencies, frequency patterns, or transferable semantic representations (Wang et al., 2020; Ojha et al., 2023; Tan et al., 2024; Chen et al., 2025), while video-specific approaches further model cross-frame consistency, fine-grained spatiotemporal artifacts, or feature trajectories (Ma et al., 2025a; Chen et al., 2026a; Corvi et al., 2025;

Intern\`o et al., 2025; Cui et al., 2026a). Despite their efectiveness, these methods primarily characterize the final outputs rather than the responses induced by an underlying generative model. As a result, supervised detectors may learn appearance- or motion-based shortcuts that are correlated with the generators observed during training, limiting their ability to generalize to unseen video sources. The substantial performance variation across generation sources reported by AIGVDBench further highlights this challenge (Ma et al., 2026).

A complementary direction is to exploit pretrained generative models as forensic probes. Existing image-based approaches investigate reconstruction errors or predicted-noise responses (Wang et al., 2023; Luo et al., 2024; Zhang & Xu, 2023), while recent methods further explore denoising trajectories, probability-flow velocity, and curvature (Vasilcoiu et al., 2025; Liang et al., 2025; Jin et al., 2026; Li et al., 2025). For video, related methods analyze difusion reconstruction errors or scorederived spatiotemporal gradients (Liu et al., 2024; Chen et al., 2026b; Zhang et al.,

![](images/98af328c4a11256cb1171b243e74b1d933464e26ef5fc1d4766d9998f1a61703.jpg)  
Figure 1: Real videos exhibit smaller velocity responses than AI-generated videos, providing a discriminative cue for detection.

2025). However, it remains underexplored how to efectively represent videos through their responses to a pretrained Flow-Matching video DiT. In particular, the velocity field defines a time-dependent vector field that describes the transport between data and noise distributions (Lipman et al., 2023). Its responses to the same video at diferent flow times may therefore reveal generation-aware forensic cues that are complementary to those contained in the rendered content.

Motivated by this perspective, we investigate whether the generative dynamics encoded by a pretrained video generator can provide more transferable forensic evidence than rendered artifacts. Specifically, we probe the same video at multiple points along the generator’s noising trajectory and examine its responses under the pretrained velocity field. As illustrated in Figure 1, real and AI-generated videos exhibit distinct velocity responses not only at individual flow times, but also in the variation of these responses along the generative trajectory. This motivates two complementary sources of evidence: flow-time-conditioned velocity responses and their local changes across neighboring flow times. Importantly, these changes occur along the generator’s flow-time dimension rather than the physical temporal dimension of the video, providing a distinct perspective from conventional inter-frame temporal-consistency analysis.

Based on this insight, we propose TRACE, which repurposes a pretrained video DiT as a generative probe to extract trajectory-aware representations from multiple flow times. TRACE further estimates the consistency of these representations across neighboring video frames by modeling their velocity diferences. In addition, we observe that real videos generally exhibit smaller velocity responses than AI-generated videos, motivating a Real-Centered Trajectory Optimization objective. This objective organizes the learned feature space around a real-video reference center by compacting real representations while pushing generated representations away from it. Such an asymmetric structure encourages the detector to characterize deviations from the real-video distribution rather than relying on generator-specific artifacts. By shifting forensic analysis from what is rendered to how a pretrained generative prior responds, TRACE aims to learn generation-aware representations with improved transferability across unseen synthesis sources.

Using Open-Sora as the sole synthetic training source, TRACE achieves macro-average AUCs of 97.95% and 95.11% on the open- and closed-source subsets of AIGVDBench, respectively. These results outperform the strongest baselines respectively by 12.87% and 17.01%, demonstrating the remarkable efectiveness of generative trajectory responses as transferable forensic signals for AI-generated video detection. TRACE further achieves the best average performance across six additional video-generation datasets, demonstrating strong cross-dataset generalization (Section 4.3).

Our main contributions are threefold:

• We introduce a generation-aware forensic perspective for AI-generated video detection by probing a pretrained Flow-Matching video DiT at multiple flow times, revealing novel forensic cues beyond conventional rendered-content features.

• We propose TRACE, which encodes multi-flow-time velocity responses, estimates cross-frame trajectory consistency through velocity diferences, and proposes a real-centered optimization objective to facilitate generator-invariant representation learning.

• We conduct extensive cross-generator evaluation on AIGVDBench, covering 31 generation settings spanning open- and closed-source models. TRACE achieves macroaverage AUCs of 97.95% and 95.11%, outperforming the prior strongest baselines by 12.87% and 17.01%, respectively.

## 2 Related Work

## 2.1 Video Generation Models

Within difusion-based (Esser et al., 2024; Gu et al., 2022; Ho et al., 2020) video synthesis, early models used spatiotemporal U-Nets (Ho et al., 2022), while recent systems scale latent video difusion transformers (Peebles & Xie, 2023) for spatiotemporal modeling and text conditioning. Representative open models include Open-Sora (Zheng et al., 2024b) and its spatial–temporal DiT, e.g. CogVideoX (Yang et al., 2025c) with a 3D autoencoder and expert transformer, HunyuanVideo (Kong et al., 2024), and the Wan family (Wan Team, 2025). In parallel, Flow Matching (Lipman et al., 2023) learns continuous normalizing flows by regressing a time-dependent vector field along a prescribed probability path. The resulting velocity field describes how samples evolve between the data and noise distributions, rather than only the endpoint produced by a generator. Prior analysis has also shown that internal representations of video difusion models contain motion-aware information (Xiao et al., 2024). These findings suggest that a pretrained video generator can provide process-level representations for forensics, complementing cues measured only in rendered pixels.

## 2.2 AI-Generated Visual Content Detection

Image-level detection. Early detectors learned transferable artifacts from CNN-generated images, with augmentation improving robustness to post-processing and unseen architectures (Wang et al., 2020). UnivFD (Ojha et al., 2023) leverages frozen CLIP (Radford et al., 2021) features for cross-generator transfer, while NPR (Tan et al., 2024), D<sup>3</sup> (Yang et al., 2025b), and DDA (Chen et al., 2025) target local dependencies and generator-invariant pixel-frequency cues. Recent image-forensics studies further explore adaptive feature modeling and explainable or interactive evidence analysis, including HyperDet (Cao et al., 2027), REVEAL (Cao et al., 2025), Veritas (Tan et al., 2026), and ClueAegis (Cao et al., 2026). Beyond rendered-content analysis, another line of work exploits pretrained generative models as forensic probes through reconstruction errors, noise responses, and denoising trajectories, including DIRE (Wang et al., 2023), LaRE<sup>2</sup> (Luo et al., 2024), DNF (Zhang & Xu, 2023), TSG (Zeng et al., 2024), timestep ensembling (Wu et al., 2025), LATTE (Vasilcoiu et al., 2025), Denoising Trajectory Biases (Liang et al., 2025), and DySy-Det (Jin et al., 2026). Recent studies further show that velocity and curvature can provide informative forensic cues (Li et al., 2025). However, frame-wise image detection cannot fully exploit the temporal structure of videos.

Video-level detection. Recent video detectors extend content-based analysis by modeling cross-frame consistency, spatiotemporal inconsistencies, low-level cues, or feature dynamics, including DeCoF (Ma et al., 2025a), DeMamba (Chen et al., 2026a), WaveRep (Corvi et al., 2025), D3 (Zheng et al., 2025), ReStraV (Intern\`o et al., 2025), and V-PVP (Cui et al., 2026a). STALL (Ben Hayun et al., 2026b) and Skyra (Li et al., 2026b) further exploit real-video statistics and multimodal artifact reasoning, respectively. Meanwhile, process-based approaches extend generative-model probing to videos through difusion reconstruction errors and score-derived spatiotemporal cues, as explored by DIVID (Liu et al., 2024), ReConFuse (Chen et al., 2026b), and NSG-VD (Zhang et al., 2025). Real-centered methods, such as Beyond Generation (Zhong et al., 2025), RCDN (McCurdy et al., 2026), LRD-Net (Zhang & Chaudhary, 2026), and SphereVideo (Li et al., 2026a), instead characterize authentic video distributions to improve detection robustness. Despite these advances, large-scale evaluation reveals substantial performance variation across detectors and generation sources (Ma et al., 2026), while existing process-based video methods focus primarily on difusion models. The forensic potential of multi-flow-time velocity responses from Flow-Matching video generators therefore remains largely unexplored. TRACE addresses this gap by modeling multi-step flow velocities and their cross-frame diferences from a pretrained Flow-Matching video DiT.

![](images/771c2494f1cdbdca0fcacc5d5767f5269aeb15755b8a33e731250d37ed249fdb.jpg)  
Figure 2: Overview of TRACE. The proposed framework learns generative trajectory representations, estimates adjacent-frame consistency through velocity diferences, and constrains the resulting feature space around a real-video center.

## 3 Methodology

TRACE is motivated by a key observation that real and AI-generated videos exhibit distinct velocity responses in pretrained Flow Matching models. Based on this finding, TRACE models video generation dynamics beyond appearance-level artifacts. As illustrated in Figure 1, controlled noise perturbations reveal diferences in these responses. The overall framework, shown in Figure 2, consists of three components: Generative Trajectory Representation Learning, Trajectory Consistency Estimation, and Real-Centered Trajectory Optimization. Together, they learn generation-aware representations to improve generalization to unseen video generators.

## 3.1 Generative Trajectory Representation Learning

Recent Flow Matching-based video generation models formulate generation as a transport process between data and noise distributions, parameterized by a velocity field. TRACE exploits the responses of a pretrained velocity field to capture generation-aware cues beyond appearance-level artifacts.

Let $\mathbf { x } _ { \mathrm { 0 } }$ denote the clean latent of a video frame and $\epsilon \sim p _ { 0 }$ a noise sample. In the continuous Flow Matching formulation,

$$
{ \mathbf x } _ { t } = ( 1 - t ) { \mathbf x } _ { 0 } + t { \boldsymbol \epsilon } , \qquad t \in [ 0 , 1 ] .\tag{1}
$$

Rather than performing the complete generation or denoising process, TRACE treats the pretrained Wan2.2-A14B DiT as a fixed velocity field and probes it at a small set of scheduler timesteps. Let

$$
{ \cal S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { K } \} .\tag{2}
$$

Each $s _ { k }$ specifies the noise level applied to the clean latent before the DiT forward pass, and thus represents a probing location rather than a denoising step. We use a 1000-level

scheduler with $K = 5 \mathrm { : }$

$$
\begin{array} { r } { S = \{ 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \} . } \end{array}\tag{3}
$$

These probe points are fixed throughout all experiments. As analyzed in Appendix D, $s =$ 100 already provides strong performance, while larger noise levels yield limited additional gains.

To capture temporal information, we uniformly sample N frames from each video. $\operatorname { L e t } \mathbf { x } _ { 0 } ^ { ( n ) }$ denote the clean latent of the n-th frame. A single noise realization $\epsilon \sim p _ { 0 }$ is shared across all sampled frames. For each $s _ { k } \in \mathcal S$ , the perturbed latent is obtained using the Wan2.2 Flow Matching scheduler:

$$
\begin{array} { r } { \mathbf { x } _ { s _ { k } } ^ { ( n ) } = \mathcal { P } \left( \mathbf { x } _ { 0 } ^ { ( n ) } , \epsilon , s _ { k } \right) , \quad \quad n = 1 , \hdots , N , \quad k = 1 , \hdots , K , } \end{array}\tag{4}
$$

where $\mathcal { P } ( \cdot )$ denotes the scheduler-defined perturbation operation. At $s _ { k } = 0$ , the clean latent is directly used.

The perturbed latent is then fed into the frozen DiT to obtain the velocity representation:

$$
\mathbf { v } _ { s _ { k } } ^ { ( n ) } = f _ { \theta } ^ { \mathrm { D i T } } \left( \mathbf { x } _ { s _ { k } } ^ { ( n ) } , s _ { k } \right) .\tag{5}
$$

For each frame, the velocity responses across probe points form a generation-aware trajectory:

$$
\mathbf { \boldsymbol { \mathcal { V } } } ^ { ( n ) } = \left[ \mathbf { v } _ { s _ { 1 } } ^ { ( n ) } , \mathbf { v } _ { s _ { 2 } } ^ { ( n ) } , \ldots , \mathbf { v } _ { s _ { K } } ^ { ( n ) } \right] .\tag{6}
$$

Stacking the trajectories of all sampled frames yields the frame–noise representation:

$$
\mathbf { V } = \left[ \begin{array} { c c c c } { \mathbf { v } _ { s _ { 1 } } ^ { ( 1 ) } } & { \mathbf { v } _ { s _ { 1 } } ^ { ( 2 ) } } & { \ldots } & { \mathbf { v } _ { s _ { 1 } } ^ { ( N ) } } \\ { \mathbf { v } _ { s _ { 2 } } ^ { ( 1 ) } } & { \mathbf { v } _ { s _ { 2 } } ^ { ( 2 ) } } & { \ldots } & { \mathbf { v } _ { s _ { 2 } } ^ { ( N ) } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { v } _ { s _ { K } } ^ { ( 1 ) } } & { \mathbf { v } _ { s _ { K } } ^ { ( 2 ) } } & { \ldots } & { \mathbf { v } _ { s _ { K } } ^ { ( N ) } } \end{array} \right] \in \mathbb { R } ^ { K \times N \times D } ,\tag{7}
$$

where D denotes the velocity feature dimension. The horizontal and vertical axes correspond to temporal progression and noise-level variation, respectively. Thus, V jointly preserves temporal and noise-conditioned responses of the pretrained generative model, providing generation-aware representations for subsequent trajectory consistency modeling and classification.

## 3.2 Trajectory Consistency Estimation

The velocity representation V characterizes the responses of individual video frames to the pretrained Flow Matching model at diferent scheduler timesteps. While these frame-level responses provide generation-aware cues, they do not explicitly capture temporal relationships between neighboring frames. We therefore introduce Trajectory Consistency Estimation to model temporal consistency in the velocity space.

Neighboring frames in real videos typically describe the same underlying scene and evolve smoothly, leading to relatively consistent velocity responses. In contrast, temporal inconsistencies in AI-generated videos may induce irregular variations between adjacent responses. We thus measure frame-to-frame velocity changes while keeping the scheduler timestep and noise realization fixed. For the k-th probe timestep $s _ { k } \in \mathcal S$ , we compute

$$
\Delta \mathbf { v } _ { s _ { k } } ^ { ( n ) } = \mathbf { v } _ { s _ { k } } ^ { ( n + 1 ) } - \mathbf { v } _ { s _ { k } } ^ { ( n ) } , \qquad n = 1 , \dots , N - 1 .\tag{8}
$$

Since the two frames share the same scheduler timestep and noise realization $\epsilon ,$ the resulting diference primarily captures the change in model response induced by temporal variations in the video content. Repeating this operation across multiple probe timesteps yields

$$
\Delta \mathbf { v } _ { s _ { k } } ^ { ( 1 ) } , \Delta \mathbf { v } _ { s _ { k } } ^ { ( 2 ) } , . . . , \Delta \mathbf { v } _ { s _ { k } } ^ { ( N - 1 ) } ,\tag{9}
$$

which characterizes temporal consistency under diferent noise conditions.

The original velocity representation V and temporal velocity diferences provide complementary cues: V preserves the responses of individual frames, while $\Delta \mathbf { v } _ { s _ { k } } ^ { ( n ) }$ captures their temporal variations. We therefore fuse them into the final trajectory-aware representation:

$$
\mathbf { E } = \mathrm { F u s e } \left( \mathbf { V } , \left\{ \Delta \mathbf { v } _ { s _ { k } } ^ { ( n ) } \right\} \right) ,\tag{10}
$$

where Fuse(·) denotes the feature fusion operation and E represents the final trajectoryaware video feature. This representation jointly captures generation-aware velocity responses and temporal consistency cues, and is subsequently used for Real-Centered Trajectory Optimization.

## 3.3 Real-Centered Trajectory Optimization

Although trajectory-aware representations capture generation-process and temporal cues, binary supervision may still encourage the detector to exploit generator-specific characteristics that are correlated with the training sources, limiting generalization to unseen generators. To address this issue, TRACE introduces a Real-Centered Trajectory Optimization objective. Our observation that real videos generally exhibit smaller velocity responses than AI-generated videos motivates us to use real videos as the reference distribution. Specifically, TRACE encourages real-video representations to form a compact region around a real-feature center while pushing AI-generated representations away from this center. This asymmetric structure encourages the detector to learn deviations from the real-video distribution rather than relying on generator-specific artifacts.

Let z denote the fused trajectory representation and let $y \in \{ 0 , 1 \}$ denote the ground-truth label, where $y = 0$ represents a real video and $y = 1$ represents an AI-generated video. The classification head produces a logit

$$
\boldsymbol \ell = h ( \mathbf { z } ) ,\tag{11}
$$

and the standard binary classification loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B C E } } = \mathrm { B C E W i t h L o g i t s } ( \ell , y ) . } \end{array}\tag{12}
$$

To characterize the real-video distribution, we maintain a real-feature center µ using an exponential moving average (EMA). Given the mean feature $\bar { \mathbf { z } } ^ { r }$ of a real batch, the center is updated by

$$
\pmb { \mu }  \alpha \pmb { \mu } + ( 1 - \alpha ) \bar { \bf z } ^ { r } ,\tag{13}
$$

where $\alpha = 0 . 9 9$ . We then measure the distance between an individual feature and the real center as

$$
d = \| \mathbf { z } - { \pmb { \mu } } \| _ { 2 } .\tag{14}
$$

For real videos, we introduce a Pull objective that encourages their trajectory-aware representations to concentrate around the real center:

$$
\mathcal { L } _ { \mathrm { p u l l } } = \lambda _ { \mathrm { p u l l } } \mathbb { 1 } [ y = 0 ] d ^ { 2 } ,\tag{15}
$$

where $\lambda _ { \mathrm { p u l l } }$ controls the strength of the real-centered constraint.

In contrast, AI-generated videos are encouraged to move away from the real center. We therefore introduce a Push objective:

$$
\mathcal { L } _ { \mathrm { p u s h } } = \lambda _ { \mathrm { p u s h } } \mathbb { 1 } [ y = 1 ] \mathrm { s o f t p l u s } \left( \frac { m - d } { m } \right) ,\tag{16}
$$

where m denotes the separation margin. This objective penalizes generated samples when their representations are close to the real center and gradually reduces the penalty as their distance exceeds the margin. Therefore, it does not impose an unnecessary constraint on already well-separated generated samples.

The overall training objective is

$$
\mathcal { L } = w _ { \mathrm { B C E } } \mathcal { L } _ { \mathrm { B C E } } + \lambda _ { \mathrm { p u l l } } \mathcal { L } _ { \mathrm { p u l l } } + \lambda _ { \mathrm { p u s h } } \mathcal { L } _ { \mathrm { p u s h } } ,\tag{17}
$$

where w<sub>BCE</sub>, $\lambda _ { \mathrm { p u l l } }$ , and $\lambda _ { \mathrm { p u s h } }$ are set to 1 by default, and the separation margin is set to $m = 4 0$ . When the real-feature center has not yet been initialized, only $\mathcal { L } _ { \mathrm { B C E } }$ is applied.

The proposed objective explicitly imposes an asymmetric structure on the trajectory-aware feature space: real videos are pulled toward a compact real-centered region, whereas AIgenerated videos are pushed away from it. This formulation encourages TRACE to focus on deviations from the real-video response pattern and reduces its reliance on generator-specific characteristics. As a result, the learned representation is better suited to generalize across unseen video generators. During inference, TRACE uses the classification logit ℓ as the detection score. The distance d is used only during training to construct the real-centered optimization objective.

## 4 Experiments

## 4.1 Experimental Setup

To evaluate TRACE, we adopt Wan2.2-A14B (Wan Team, 2025) as the pretrained backbone and attach an MLP classifier to its final-layer DiT features. Experiments are conducted on AIGVDBench (Ma et al., 2026), the largest and most comprehensive benchmark for AI-generated video detection to date. Following a generator-disjoint evaluation protocol, TRACE is trained on a balanced subset of AIGVDBench comprising 14K real videos and 14K videos generated by Open-Sora, which serves as the sole synthetic source during training. For a fair comparison, all competing methods are trained on the same data. We evaluate the resulting detectors across 31 distinct generator–task configurations, covering 20 open-source and 11 closed-source settings, to assess their ability to generalize beyond the training generator. Detailed training and implementation configurations are provided in Appendix A, detailed dataset information is provided in Appendix B, and descriptions of the competing methods are given in Appendix C.

Evaluation Metrics. We report AUC as the primary evaluation metric on AIGVD Bench, measuring discrimination performance across decision thresholds. For completeness, AIGVDBench results for ACC@5% FPR and AP are reported in Appendix E and Ap pendix F, respectively. We also use ACC@5% FPR for the additional datasets in Section 4.3. Visualizations of the experimental results are provided in Appendices G and H.

## 4.2 Comparison to State-of-the-Art Detectors Evaluation on AIGVDbench

We compare TRACE with state-of-the-art visual detectors, including both image- and videobased methods, on AIGVDBench. We report results separately on the open-source and closed-source generation settings to evaluate the cross-generator generalization of diferent detection paradigms.

## 4.2.1 Results on Open-Source Data

Table 1 reports the results on the open-source generator subset of AIGVDBench. Existing methods perform well primarily on in-domain Open-Sora videos but drop substantially on unseen generators. In contrast, TRACE achieves an AUC of 97.95%, outperforming the strongest baseline by 12.87%. This improvement indicates that modeling the generative trajectory enables TRACE to capture generation-aware cues beyond generator-specific appearance artifacts, resulting in substantially stronger generalization to unseen generators.

## 4.2.2 Results on Closed-Source Data

Table 2 reports the results on the closed-source generator subset of AIGVDBench. Compared with open-source generators, closed-source models present a more challenging setting, where existing methods sufer substantial performance degradation on unseen generators. In contrast, TRACE achieves an AUC of 95.11%, outperforming the strongest baseline by 17.01%. Notably, TRACE maintains strong performance across unseen closed-source generators, demonstrating its superior capability in cross-generator generalization.

Table 1: 12.87% AUC improvement over existing detectors across open-source video generators, with the best and second-best highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="6">I2V</td><td colspan="6"></td><td colspan="6">T2V</td><td colspan="2">V2V</td><td colspan="2">AVG</td></tr><tr><td>Easy Animate</td><td>LTX</td><td>Pyramid Flow</td><td>SEINE</td><td>SVD</td><td>Video Crafter</td><td>Acc Video</td><td>Animate Diff</td><td>Cogvideo x1.5</td><td>Easy Animate</td><td>Hunyuan</td><td>IPOC</td><td>LTX</td><td>Open Sora</td><td>Pyramid Flow</td><td></td><td>Rep Video</td><td>Video Crafter</td><td>Wan 2.1</td><td>Cogvideo x1.5</td><td>LTX</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Image Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNNSpot</td><td>76.71</td><td>75.41</td><td>89.59</td><td>91.79</td><td>97.72</td><td>53.92</td><td>58.54</td><td>39.90</td><td>78.08</td><td>64.15</td><td>68.82</td><td>85.43</td><td>95.72</td><td>100.0</td><td>96.37</td><td>75.95</td><td>72.78</td><td></td><td>57.16</td><td>71.42</td><td>71.01</td><td>76.02</td></tr><tr><td>UnivFD</td><td>76.39</td><td>80.11</td><td>87.65</td><td>97.70</td><td>95.79</td><td>71.21</td><td>61.32</td><td>77.83</td><td>87.53</td><td>68.99</td><td>60.68</td><td>86.78</td><td>89.47</td><td>99.99</td><td>94.24</td><td></td><td>79.78</td><td>87.59</td><td>54.28</td><td>81.94</td><td>73.14</td><td>80.62</td></tr><tr><td>NPR</td><td>77.27</td><td>88.64</td><td>95.04</td><td>93.58</td><td>98.07</td><td>56.26</td><td>42.96</td><td>47.70</td><td>78.34</td><td>44.83</td><td>57.74</td><td>81.03</td><td>96.69</td><td>100.0</td><td>93.63</td><td>69.02</td><td>72.69</td><td></td><td>45.70</td><td>87.58</td><td>84.71</td><td>75.57</td></tr><tr><td>DDA D3</td><td>55.44 77.85</td><td>52.00 82.16</td><td>64.36</td><td>62.55</td><td>72.19</td><td>40.60</td><td>64.03</td><td>67.84</td><td>67.15</td><td>73.67</td><td>66.97</td><td>58.09</td><td>61.06</td><td>100.0</td><td>83.45</td><td>52.04</td><td>62.72</td><td></td><td>62.63</td><td>52.92</td><td>50.81</td><td>63.53</td></tr><tr><td></td><td></td><td></td><td>89.58</td><td>98.18</td><td>97.12</td><td>82.20</td><td>63.67</td><td>72.67</td><td>86.83</td><td>73.80</td><td>60.04</td><td>90.09</td><td>90.86</td><td>100.0</td><td>96.25</td><td>81.24</td><td>90.51</td><td></td><td>58.14</td><td>81.09</td><td>77.07</td><td>82.47</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Video Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeMamba</td><td>70.08</td><td>69.61</td><td>80.82</td><td>87.35</td><td>94.50</td><td>58.75</td><td>62.80</td><td>73.81</td><td>79.05</td><td>76.53</td><td>71.46</td><td>79.33</td><td>83.77</td><td>100.0</td><td>91.21</td><td>66.99</td><td>92.18</td><td>62.01</td><td></td><td>70.99</td><td>66.06</td><td>76.87</td></tr><tr><td>DeCoF</td><td>77.04</td><td>81.92</td><td>86.83</td><td>93.66</td><td>97.26</td><td>60.24</td><td>78.46</td><td>73.99</td><td>91.23</td><td>81.74</td><td>79.96</td><td>95.29</td><td>93.14</td><td>100.0</td><td>98.51</td><td>90.38</td><td>91.91</td><td></td><td>68.01</td><td>81.42</td><td>80.68</td><td>85.08</td></tr><tr><td>WaveRep</td><td>49.74</td><td>48.66</td><td>49.40</td><td>49.60</td><td>49.03</td><td>49.44</td><td>48.89</td><td>49.03</td><td>50.04</td><td>51.69</td><td>46.37</td><td>49.85</td><td>50.66</td><td>50.27</td><td>51.28</td><td>49.31</td><td>52.63</td><td>49.42</td><td></td><td>46.86</td><td>49.00</td><td>49.56</td></tr><tr><td>ReStraV</td><td>97.08</td><td>45.42</td><td>49.48</td><td>46.96</td><td>60.89</td><td>32.46</td><td>42.23</td><td>15.69</td><td>73.32</td><td>91.25</td><td>46.01</td><td>84.31</td><td>47.22</td><td>98.79</td><td>44.32</td><td>91.36</td><td>15.28</td><td>76.89</td><td></td><td>54.08</td><td>50.18</td><td>58.16</td></tr><tr><td>Qwen2.5-ViT</td><td>51.14</td><td>48.75 46.47</td><td>53.85</td><td>51.78</td><td>50.69</td><td>50.05</td><td>56.67</td><td>50.46</td><td>60.53</td><td>52.27</td><td>56.84</td><td>58.04</td><td>52.30</td><td>60.70</td><td>55.85</td><td>54.85</td><td>55.25</td><td>51.88</td><td></td><td>57.07</td><td>50.20</td><td>53.96 54.77</td></tr><tr><td>STALL V-PVP</td><td>50.29 72.16</td><td>75.20</td><td>47.41</td><td>45.12</td><td>50.05</td><td>54.02</td><td>56.89</td><td>67.25</td><td>55.52</td><td>62.23</td><td>59.58</td><td>62.68</td><td>46.17</td><td>60.21</td><td>56.15</td><td>54.88</td><td>64.06</td><td>57.93</td><td></td><td>49.95</td><td>48.52</td><td>81.82</td></tr><tr><td>TRACE</td><td>98.41</td><td>99.06</td><td>84.17 99.71</td><td>89.40</td><td>93.16</td><td>83.54 100.0</td><td>75.23</td><td>85.64</td><td>76.56</td><td>78.73</td><td>76.30</td><td>84.92</td><td>81.75</td><td>99.90</td><td>93.07</td><td>77.61</td><td>94.61</td><td>70.63</td><td></td><td>72.47</td><td>71.40</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>100.0</td><td>100.0</td><td></td><td>96.00</td><td>99.95</td><td>96.79</td><td>99.23</td><td>96.96</td><td>98.77</td><td>99.57</td><td>100.0</td><td>99.94</td><td>95.91</td><td>100.0</td><td></td><td>84.11</td><td>97.51</td><td>97.12</td><td>97.95</td></tr></table>

Table 2: 17.01% AUC improvement over existing methods on closed-source video generators, with the best and second-best highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="10">Closed-Source Approaches</td><td rowspan="2">AVG</td></tr><tr><td>Gen2</td><td>Gen3</td><td>Jimeng</td><td>Luma</td><td>Open Sora</td><td>Sora</td><td>Causvid 24fps</td><td>Kling</td><td>Pika</td><td>Vidu</td><td>Wan</td></tr><tr><td></td><td></td><td></td><td>AI-Generated Image</td><td></td><td></td><td></td><td>e Detection Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNNSpot</td><td>37.23</td><td>60.95</td><td>15.72</td><td>57.38</td><td>61.64</td><td>59.71</td><td>69.60</td><td>52.88</td><td>63.08</td><td>75.54</td><td>54.47</td><td>55.29</td></tr><tr><td>UnivFD</td><td>41.04</td><td>46.20</td><td>23.89</td><td>51.35</td><td>75.17</td><td>52.65</td><td>82.24</td><td>61.47</td><td>54.27</td><td>72.59</td><td>54.52</td><td>55.94</td></tr><tr><td>NPR</td><td>24.70</td><td>47.85</td><td>3.19</td><td>50.51</td><td>61.95</td><td>50.37</td><td>38.31</td><td>50.70</td><td>63.16</td><td>70.34</td><td>43.99</td><td>45.92</td></tr><tr><td>DDA</td><td>65.97</td><td>72.90</td><td>64.86</td><td>69.54</td><td>83.44</td><td>72.88</td><td>69.89</td><td>71.22</td><td>65.97</td><td>73.80</td><td>69.41</td><td>70.90</td></tr><tr><td>D3</td><td>36.59</td><td>41.77</td><td>31.97</td><td>42.34</td><td>67.67</td><td>42.72</td><td>82.57</td><td>60.92</td><td>51.82</td><td>66.09</td><td>50.27</td><td>52.25</td></tr><tr><td colspan="14">AI-Generated Video Detection Models</td></tr><tr><td>DeMamba</td><td>68.57</td><td>64.46</td><td>54.10</td><td>61.67</td><td>78.56</td><td>66.44</td><td>68.63</td><td>54.98</td><td>76.26</td><td>72.54</td><td>57.22</td><td>65.77</td></tr><tr><td>DeCoF</td><td>54.31</td><td>69.45</td><td>29.02</td><td>73.18</td><td>90.69</td><td>74.85</td><td>95.48</td><td>74.54</td><td>83.10</td><td>84.41</td><td>70.82</td><td>72.71</td></tr><tr><td>WaveRep</td><td>56.27</td><td>53.20</td><td>52.56</td><td>51.44</td><td>55.20</td><td>57.26</td><td>54.78</td><td>48.16</td><td>53.04</td><td>55.04</td><td>52.12</td><td>53.55</td></tr><tr><td>ReStraV</td><td>20.84</td><td>41.24</td><td>16.78</td><td>37.34</td><td>41.55</td><td>57.55</td><td>42.21</td><td>43.14</td><td>39.20</td><td>45.32</td><td>54.90</td><td>40.01</td></tr><tr><td>Qwen2.5-ViT</td><td>71.92</td><td>57.40</td><td>71.19</td><td>55.52</td><td>63.76</td><td>53.06</td><td>57.36</td><td>50.42</td><td>58.53</td><td>54.64</td><td>53.06</td><td>58.80</td></tr><tr><td>STALL</td><td>85.48</td><td>79.22</td><td>77.89</td><td>81.43</td><td>82.27</td><td>74.98</td><td>80.56</td><td>72.10</td><td>79.80</td><td>70.74</td><td>74.68</td><td>78.10</td></tr><tr><td>V-PVP</td><td>74.35</td><td>76.28</td><td>63.22</td><td>69.80</td><td>87.74</td><td>70.55</td><td>71.99</td><td>72.92</td><td>90.23</td><td>77.55</td><td>70.66</td><td>75.03</td></tr><tr><td>TRACE</td><td>99.48</td><td>93.11</td><td>99.50</td><td>97.20</td><td>99.38</td><td>81.52</td><td>97.94</td><td>97.47</td><td>99.86</td><td>92.98</td><td>87.80</td><td>95.11</td></tr></table>

## 4.3 Evaluation on Additional Datasets

To assess generalization beyond AIGVD-Bench, we evaluate TRACE on six additional datasets: VideoFeedback (He et al., 2024), GenVideo (Chen et al., 2026a), ComGenVid (Ben Hayun et al., 2026a), Magic Videos (Li et al., 2026c), GVD (Bai et al., 2024), and GVF (Ma et al., 2025b). Figure 3 compares TRACE with DDA, De-Mamba, DeCoF, STALL, and V-PVP using ACC@5% FPR. TRACE ranks first on four of the six datasets, with an unweighted mean of 85.48%, compared with 76.48% for V-PVP, the strongest baseline on average. The largest gain over the best baseline is 15.60% on ComGen-Vid. STALL performs best on GenVideo and Magic Videos, indicating that the gains vary across datasets. While TRACE does

![](images/79b07947d9ae2c1adfe770370b3fd2e2b6336f962533220d448f1de751e0335a.jpg)  
Figure 3: ACC@5% FPR (%) on six datasets.

not consistently outperform all baselines on every dataset, its performance remains comparable to that of existing state-of-the-art methods, demonstrating strong generalization across diverse video generation scenarios.

Table 3: Ablation study of TRACE components. “Open” and “Closed” denote openand closed-source test sets, respectively.
<table><tr><td>Components</td><td>Noise Level</td><td>AUC (%)</td><td></td><td>ACC@5%FPR (%)</td></tr><tr><td>Noise RCTO</td><td>t</td><td>Open</td><td>Closed </td><td> $\mathrm { O p e n }$  Closed</td></tr><tr><td>x x</td><td></td><td>96.29</td><td>91.70</td><td>90.98 87.04</td></tr><tr><td>√ x</td><td>100</td><td>97.39</td><td>93.95 93.09</td><td>90.69</td></tr><tr><td>x √</td><td></td><td>96.71</td><td>92.38</td><td>91.54 87.78</td></tr><tr><td>√ √</td><td>200</td><td>97.21</td><td>93.86</td><td>92.74 90.32</td></tr><tr><td>√</td><td>100</td><td>97.95</td><td>95.11</td><td>94.27 92.03</td></tr></table>

![](images/ac40c6efa01e573ec7d9f3a80ed1c9cd934e2294fe00a6958c4b9e1c167310be.jpg)  
Figure 4: Comparisons of diferent backbones on AIGVDBench. Reported results are averaged across all 31 settings.

## 4.4 Ablation Studies

## 4.4.1 Ablation of Individual Components

We conduct ablation studies on the main components of TRACE to evaluate their individual contributions to the overall detection performance. As shown in Table 3, we investigate the efects of noise perturbation and Real-Centered Trajectory Optimization (RCTO), while keeping the remaining settings unchanged. The results demonstrate that each component contributes to the final performance, and their combination yields the most efective detection capability.

## 4.4.2 Backbone Comparison

We investigate the efectiveness of diferent pretrained video generation backbones for extracting generation-aware representations. Specifically, we replace the Wan2.2-A14B DiT backbone with HunyuanVideo-I2V(Kong et al., 2024), Wan2.1-1.3B, Wan2.1-14B, and Open-Sora(Zheng et al., 2024a), while keeping all other components of TRACE unchanged. As shown in Figure 4, the detection performance generally improves with the capability and scale of the pretrained video generation backbone, with the more advanced and larger models yielding stronger generation-aware representations. Meanwhile, all evaluated backbones provide efective velocity representations for distinguishing real and AI-generated videos, demonstrating that the velocity responses of pretrained generative models contain useful generation-related cues for video detection.

![](images/5edcdf3757ec9b0e45ef5bacc01f58e48b996c76713d79aa6dbeb247558be032.jpg)  
Figure 5: Robustness evaluation of TRACE in terms of AUC under diferent video distortions, including JPEG compression, Gaussian blur, and resizing.

## 4.5 Robustness Evaluation

We further evaluate the robustness of TRACE on AIGVDBench under JPEG compression $( q \in \{ 1 0 0 , 9 0 , 8 0 , 7 0 , 6 0 \} )$ , Gaussian blur $( \sigma \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \} )$ , and resolution scaling $( s \in \{ 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \}$ , with restoration to the original size). Perturbations are applied to raw frames prior to model-specific preprocessing. We report AUC across all 14 settings, as shown in Figure 5. TRACE consistently achieves the best performance across the evaluated robustness conditions, demonstrating its strong robustness to common video distortions. This robustness can be attributed to its reliance on generation-aware velocity representations and trajectory-level consistency, which provide complementary forensic cues beyond appearance-based artifacts and help preserve the discriminative information under various input distortions.

## 5 Conclusion

This work presented TRACE, a trajectory-based framework for AI-generated video detection that extracts generation-aware cues from pretrained Flow Matching models. By modeling velocity responses across multiple flow time points and their cross-frame consistency, TRACE captures transferable forensic signals beyond conventional appearance-based artifacts. Extensive experiments demonstrate that TRACE significantly outperforms existing detectors in generalizing to unseen open- and closed-source video generators.

## References

Jianfa Bai, Man Lin, Gang Cao, and Zijie Lou. Ai-generated video detection via spatial temporal anomaly learning. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pp. 460–470. Springer, 2024.

Omer Ben Hayun, Roy Betser, Meir Yossef Levi, Levi Kassel, and Guy Gilboa. Trainingfree detection of generated videos via spatial-temporal likelihoods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16299–16310, 2026a.

Omer Ben Hayun, Roy Betser, Meir Yossef Levi, Levi Kassel, and Guy Gilboa. Trainingfree detection of generated videos via spatial-temporal likelihoods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16299– 16310, 2026b. URL https://openaccess.thecvf.com/content/CVPR2026/html/ Hayun\_Training-free\_Detection\_of\_Generated\_Videos\_via\_Spatial-Temporal\_ Likelihoods\_CVPR\_2026\_paper.html.

Huangsen Cao, Qin Mei, Zhiheng Li, Yuxi Li, Zhan Meng, Ying Zhang, Chen Li, Zhimeng Zhang, Xin Ding, Yongwei Wang, et al. Reveal: Reasoning-enhanced forensic evidence analysis for explainable ai-generated image detection. arXiv preprint arXiv:2511.23158, 2025.

Huangsen Cao, Hongkang Chu, Yuxi Li, Ying Zhang, Chen Li, Jing Lyu, Yongwei Wang, Yu Zhao, and Fei Wu. Clueaegis: Heuristic-to-reasoning cognitive-skill learning for unified evidence-based synthetic image detection. arXiv preprint arXiv:2605.25009, 2026.

Huangsen Cao, Yongwei Wang, Yu Zhao, Kangtao Lv, Xin Ding, and Fei Wu. Towards generalizable detection of ai-generated images via adaptively fusing multi-view hyper experts. Information Processing & Management, 64(1):104992, 2027.

Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, et al. DeMamba: AI-generated video detection on million-scale GenVideo benchmark. Science China Information Sciences, 69 (6):162103, 2026a.

Ruoxin Chen, Junwei Xi, Zhiyuan Yan, Ke-Yue Zhang, Shuang Wu, Jingyi Xie, Xu Chen, Lei Xu, Isabel Guan, Taiping Yao, and Shouhong Ding. Dual data alignment makes AI generated image detector easier generalizable. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://openreview.net/forum?id=C39ShJwtD5.

Xiaojing Chen, Xinyu Lu, Changtao Miao, and Yunfeng Diao. ReConFuse: Reconstructionerror guided semantic fusion for AI-generated video detection. arXiv preprint arXiv:2606.04706, 2026b. URL https://arxiv.org/abs/2606.04706.

Riccardo Corvi, Davide Cozzolino, Ekta Prashnani, Shalini De Mello, Koki Nagano, and Luisa Verdoliva. Seeing what matters: Generalizable AI-generated video detection with forensic-oriented augmentation. In Advances in Neural Information Processing Systems, volume 38, pp. 26418–26446, 2025. URL https://openreview.net/forum?id= dOGXKBL7IE.

Riccardo Corvi, Davide Cozzolino, Ekta Prashnani, Shalini De Mello, Koki Nagano, and Luisa Verdoliva. Seeing what matters: Generalizable ai-generated video detection with forensic-oriented augmentation. Advances in neural information processing systems, 38: 26418–26446, 2026.

Manni Cui, Ziheng Qin, ZiAn Wang, Ruiqi Liu, Dianyuan Zou, Jianglan Wei, Han Zhou, Yu Liu, Jingrui Xu, Wenhao Wang, and Zhenyu Zhang. Rethinking the readout: Unlocking video backbones for AI-generated video detection. arXiv preprint arXiv:2607.15321, 2026a. URL https://arxiv.org/abs/2607.15321.

Manni Cui, Ziheng Qin, ZiAn Wang, Ruiqi Liu, Dianyuan Zou, Jianglan Wei, Han Zhou, Yu Liu, Jingrui Xu, Wenhao Wang, et al. Rethinking the readout: Unlocking video backbones for ai-generated video detection. arXiv preprint arXiv:2607.15321, 2026b.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized difusion model for text-to-image synthesis. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10686– 10696. IEEE, 2022.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2105– 2123, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video difusion models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/ paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html.

Christian Intern\`o, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, and David Klindt. AI-generated video detection via perceptual straightening. In Advances in Neural Information Processing Systems, volume 38, pp. 20672–20705, 2025. URL https://openreview.net/forum?id=LsmUgStXby.

Christian Intern\`o, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, and David Klindt. Ai-generated video detection via perceptual straightening. Advances in neural information processing systems, 38:20672–20705, 2026.

Fanli Jin, Feng Lin, Gaojian Wang, Tong Wu, and Zhisheng Yan. DySy-Det: A synergistic framework with dynamic reconstruction-path consistency for AI-generated image detection. Proceedings of the AAAI Conference on Artificial Intelligence, 40(42):35571–35579, 2026. doi: 10.1609/aaai.v40i42.40868.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. URL https://arxiv. org/abs/2412.03603.

Fei Li, Yue Yu, Yuran Wang, Xinghan Li, Jingjing Chen, and Yu-Gang Jiang. SphereVideo: Prototype-anchored hyperspherical boundary for continual AI-generated video detection. arXiv preprint arXiv:2608.01334, 2026a. URL https://arxiv.org/abs/2608.01334.

Jiaming Li, Weihua Ou, Ni Li, and Meilin Zheng. Synthetic image detection via curvature of difusion probability flows. OpenReview, ICLR 2026 submission, 2025. URL https: //openreview.net/forum?id=BK5iPNVUdl.

Yifei Li, Wenzhao Zheng, Yanran Zhang, Runze Sun, Yu Zheng, Lei Chen, Jie Zhou, and Jiwen Lu. Skyra: AI-generated video detection via grounded artifact reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4482–4493, 2026b. URL https: //openaccess.thecvf.com/content/CVPR2026/html/Li\_Skyra\_AI-Generated\_ Video\_Detection\_via\_Grounded\_Artifact\_Reasoning\_CVPR\_2026\_paper.html.

Zhengcen Li, Chenyang Jiang, Hang Zhao, Shiyang Zhou, Yunyang Mo, Feng Gao, Fan Yang, Qiben Shan, Shaocong Wu, and Jingyong Su. Preserving forgery artifacts: Ai-generated video detection at native scale. arXiv preprint arXiv:2604.04634, 2026c.

Yachao Liang, Min Yu, Gang Li, Jianguo Jiang, Fuqiang Du, Jingyuan Li, Lanchi Xie, Zhen Xu, and Weiqing Huang. Denoising trajectory biases for zero-shot AIgenerated image detection. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://papers.nips.cc/paper\_files/paper/2025/hash/ dfd12fe50b18505e3c912c4426707cc7-Abstract-Conference.html.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Qingyuan Liu, Pengyuan Shi, Yun-Yun Tsai, Chengzhi Mao, and Junfeng Yang. Turns out i’m not real: Towards robust detection of AI-generated videos. arXiv preprint arXiv:2406.09601, 2024. URL https://arxiv.org/abs/2406.09601.

Yunpeng Luo, Junlong Du, Ke Yan, and Shouhong Ding. LaRE<sup>2</sup>: Latent reconstruction error based method for difusion-generated image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17006–17015, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Luo\_LaRE2\_ Latent\_Reconstruction\_Error\_Based\_Method\_for\_Diffusion-Generated\_Image\_ Detection\_CVPR\_2024\_paper.html.

Long Ma, Zhiyuan Yan, Qinglang Guo, Yong Liao, Haiyang Yu, and Pengyuan Zhou. Detecting AI-generated video via frame consistency. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pp. 2173–2178, 2025a. doi: 10.1109/ICME59968.2025. 11210049.

Long Ma, Zhiyuan Yan, Qinglang Guo, Yong Liao, Haiyang Yu, and Pengyuan Zhou. Detecting ai-generated video via frame consistency. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6. IEEE, 2025b.

Long Ma, Zihao Xue, Yan Wang, Zhiyuan Yan, Jin Xu, Xiaorui Jiang, Haiyang Yu, Yong Liao, and Zhen Bi. Your one-stop solution for AI-generated video detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4458–4470, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/ html/Ma\_Your\_One-Stop\_Solution\_for\_AI-Generated\_Video\_Detection\_CVPR\_ 2026\_paper.html.

Wyatt McCurdy, Xin Zhang, Yuqi Song, and Min Gao. RCDN: Real-centered detection network for robust face forgery identification. arXiv preprint arXiv:2601.12111, 2026. URL https://arxiv.org/abs/2601.12111.

Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards universal fake image detectors that generalize across generative models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24480–24489. IEEE, 2023.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Peebles\_Scalable\_ Diffusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 28130–28139, 2024.

Hao Tan, Zichang Tan, Senyuan Shi, Ajian Liu, Chuanbiao Song, Huijia Zhu, Weiqiang Wang, Jun Wan, Zhen Lei, et al. Veritas: Generalizable deepfake detection via patternaware reasoning. In International Conference on Learning Representations, volume 2026, pp. 66420–66477, 2026.

Ana Vasilcoiu, Ivona Najdenkoska, Zeno Geradts, and Marcel Worring. LATTE: Latent trajectory embedding for difusion-generated image detection. arXiv preprint arXiv:2507.03054, 2025. URL https://arxiv.org/abs/2507.03054.

Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnngenerated images are surprisingly easy to spot. . . for now. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8692–8701. IEEE, 2020.

Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. DIRE for difusion-generated image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22445–22455, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Wang\_DIRE\_ for\_Diffusion-Generated\_Image\_Detection\_ICCV\_2023\_paper.html.

Yixin Wu, Feiran Zhang, Tianyuan Shi, Ruicheng Yin, Zhenghua Wang, Zhenliang Gan, Xiaohua Wang, Changze Lv, Xiaoqing Zheng, and Xuanjing Huang. Explainable synthetic image detection through difusion timestep ensembling. arXiv preprint arXiv:2503.06201, 2025. URL https://arxiv.org/abs/2503.06201.

Zeqi Xiao, Yifan Zhou, Shuai Yang, and Xingang Pan. Video difusion models are trainingfree motion interpreter and controller. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/8b2fc235787852ead92da2268cd9e90c-Abstract-Conference.html.

Yongqi Yang, Zhihao Qian, Ye Zhu, Olga Russakovsky, and Yu Wu. D 3: Scaling up deepfake detection by learning from discrepancy. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23850–23859. IEEE, 2025a.

Yongqi Yang, Zhihao Qian, Ye Zhu, Olga Russakovsky, and Yu Wu. D<sup>3</sup>: Scaling up deepfake detection by learning from discrepancy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23850–23859, 2025b. URL https://openaccess.thecvf.com/content/CVPR2025/html/Yang\_D3\_Scaling\_ Up\_Deepfake\_Detection\_by\_Learning\_from\_Discrepancy\_CVPR\_2025\_paper.html.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025c.

Ziyue Zeng, Haoyuan Liu, Dingjie Peng, Luoxu Jing, and Hiroshi Watanabe. Time step generating: A universal synthesized deepfake image detector. arXiv preprint arXiv:2411.11016, 2024. URL https://arxiv.org/abs/2411.11016.

Shuhai Zhang, Zihao Lian, Jiahao Yang, Daiyuan Li, Guoxuan Pang, Feng Liu, Bo Han, Shutao Li, and Mingkui Tan. Physics-driven spatiotemporal modeling for AI-generated video detection. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://openreview.net/forum?id=HiBoJLCyEo.

Xuecen Zhang and Vipin Chaudhary. LRD-Net: A lightweight real-centered detection network for cross-domain face forgery detection. arXiv preprint arXiv:2604.10862, 2026. URL https://arxiv.org/abs/2604.10862.

Yichi Zhang and Xiaogang Xu. Difusion noise feature: Accurate and fast generated image detection. arXiv preprint arXiv:2312.02625, 2023. URL https://arxiv.org/abs/2312. 02625.

Chende Zheng, Ruiqi Suo, Chenhao Lin, Zhengyu Zhao, Le Yang, Shuai Liu, Minghui Yang, Cong Wang, and Chao Shen. D3: Training-free AI-generated video detection using second-order features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12852–12862, 2025. URL https://openaccess. thecvf.com/content/ICCV2025/html/Zheng\_D3\_Training-Free\_AI-Generated\_ Video\_Detection\_Using\_Second-Order\_Features\_ICCV\_2025\_paper.html.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing eficient video produc tion for all. arXiv preprint arXiv:2412.20404, 2024a.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404, 2024b. URL https://arxiv.org/abs/ 2412.20404.

Nan Zhong, Haoyu Chen, Yiran Xu, Zhenxing Qian, and Xinpeng Zhang. Beyond generation: A difusion-based low-level feature extractor for detecting AI-generated images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8258–8268, 2025. URL https://openaccess.thecvf.com/content/ CVPR2025/html/Zhong\_Beyond\_Generation\_A\_Diffusion-based\_Low-level\_ Feature\_Extractor\_for\_Detecting\_AI-generated\_CVPR\_2025\_paper.html.

## A Implementation and Training Details

Training Configuration. Unless otherwise specified, we freeze the VAE, T5 text encoder, and DiT backbone of Wan2.2-A14B, and optimize only the trajectory classification head. We use the low-noise DiT expert to probe each video at five flow-time points, $\mathcal { T } = \{ 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ The resulting features are fused by concatenating the absolute probe features, adjacent flow-time diferences along the noise path, and two temporal differences between clean-frame representations, corresponding to the middle–first and last– middle frame pairs. The resulting representation is fed into a lightweight classification head consisting of LayerNorm, Dropout with a rate of 0.1, and a linear layer with one output unit. Following the default text-conditioning interface of Wan2.2, all video clips are associated with the fixed prompt “this is a video.”

Preprocessing and Augmentation. For each video, we uniformly sample 9 frames and process them at a spatial resolution of 480×832. We employ an adaptive crop-resize strategy: videos smaller than the target resolution are resized using bilinear interpolation, whereas larger videos are center-cropped without downscaling. During training, spatial appearance augmentation is applied with probability 0.5, where a random subset of Gaussian blur, additive noise, and color jitter is sampled. Otherwise, the original frames are retained. In addition, a random temporal crop is independently applied with probability 0.5 when the source video contains suficient frames. All augmentations are disabled during evaluation, while the same spatial preprocessing is retained.

Objective and Optimization. The trajectory classifier is optimized with the following objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { p u l l } } + \mathcal { L } _ { \mathrm { p u s h } } ,\tag{18}
$$

where all three terms are assigned unit weights. $\mathcal { L } _ { \mathrm { B C E } }$ denotes the binary cross-entropy loss. $\mathcal { L } _ { \mathrm { p u l l } }$ encourages real-video features to remain close to an EMA-updated real-feature center with a decay rate of 0.99, while ${ \mathcal L } _ { \mathrm { p u s h } }$ encourages fake-video features to remain beyond a margin of $m = 4 0$ from the real-feature center. We optimize the classifier using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of 0.01. Training is performed with a per-GPU batch size of 4 on 4 GPUs, resulting in an efective batch size of 16, for a total of 5 epochs (8,750 optimization steps). Unless otherwise stated, all reported results are obtained from the checkpoint at step 8,750. The frozen Wan2.2-A14B backbone is executed in bfloat16 precision.

Metric Computation. We report AUC as the primary threshold-independent metric. To avoid relying on an arbitrarily selected classification threshold when computing accuracy, we additionally report ACC@5% FPR. Specifically, for each evaluation split, the decision threshold is determined as the 95th percentile of the real-video scores, fixing the falsepositive rate on real videos to approximately 5%. This provides a consistent operating point for comparing diferent detectors without tuning the classification threshold for accuracy. Average Precision (AP) is computed as the standard area under the precision-recall curve, using the predicted probability of the fake class for ranking.

## B Dataset Details

We conduct experiments on AIGVDBench (Ma et al., 2026), which contains 31 generator– task configurations, including 20 open-source and 11 closed-source settings. Each opensource setting provides 20,000 generated videos and 20,000 corresponding real videos, where each real video is paired with a generated video under the same prompt.

Training and Evaluation. Following the cross-generator evaluation protocol, we select Open-Sora as the sole source of synthetic videos for training. Specifically, we use 14,000 Open-Sora-generated videos and 14,000 matched real videos for training. The remaining evaluation data are not used for model optimization. For the 20 open-source generator– task configurations, we use 3,000 generated videos and 3,000 matched real videos from each configuration for testing. For the 11 closed-source configurations, each setting contains 2,000 generated videos and 2,000 matched real videos for evaluation. Therefore, all generators other than Open-Sora are unseen during training, enabling a direct evaluation of TRACE’s cross-generator generalization.

## C Details of Baselines

CNNSpot(Wang et al., 2020). CNNSpot is a widely used image-level detector for identifying AI-generated content, which trains a binary classifier directly on RGB images. We adapt CNNSpot to the video detection setting and follow its original architecture and training strategy, while using the same training data and evaluation protocol as other baselines for a fair comparison.

UnivFD(Ojha et al., 2023). UnivFD is a universal synthetic image detector that leverages pretrained visual representations(CLIP(Radford et al., 2021)) to generalize across different generative models. We use the oficial implementation and pretrained model released by the authors and apply it to sampled video frames, aggregating frame-level predictions to obtain video-level scores.

NPR(Tan et al., 2024). NPR is a synthetic image detector that exploits artifacts introduced by upsampling operations in CNN-based generative networks to capture generationrelated forensic traces. We use the oficial implementation and follow the recommended preprocessing and evaluation settings.

DDA(Chen et al., 2025). DDA is a synthetic image detector that improves generalization through dual data alignment. We follow the oficial implementation and train DDA using the same training data and evaluation protocol as other baselines.

D3(Yang et al., 2025a). D3 is a deepfake detection method that learns discrepancy representations to distinguish real and AI-generated images. We follow its oficial implementation and train the model under the same data and evaluation protocol as other baselines for a fair comparison.

DeMamba(Chen et al., 2026a). DeMamba is a synthetic video detector that incorporates the Mamba architecture to learn discriminative forensic representations. We follow the oficial implementation and train DeMamba using the same training data and evaluation protocol as other baselines.

DeCoF(Ma et al., 2025b). DeCoF is a synthetic video detection method that exploits frame-level consistency to distinguish real and AI-generated videos. We follow the oficial implementation and train DeCoF using the same training data and evaluation protocol as other baselines.

WaveRep(Corvi et al., 2026). WaveRep is a synthetic video detection method that exploits wavelet-domain representations to capture forensic cues in both the spatial and frequency domains. We follow the oficial implementation and train WaveRep using the same training data and evaluation protocol as other baselines.

ReStraV(Intern\`o et al., 2026). ReStraV is an AI-generated video detection method that learns perceptually straightened representations to improve detection generalization across unseen generators. We follow the oficial implementation and train ReStraV using the same training data and evaluation protocol as other baselines.

Qwen2.5-ViT(Li et al., 2026c). Qwen2.5-ViT is a vision-based baseline built upon the visual encoder of Qwen2.5-VL. We use the pretrained visual encoder as a frozen feature extractor and train a lightweight binary classification head using the same training data and evaluation protocol as TRACE.

STALL(Ben Hayun et al., 2026a). STALL is a training-free AI-generated video detection method that exploits spatial-temporal likelihoods to distinguish real and generated videos. We follow the oficial implementation and apply STALL using the same preprocessing and evaluation protocol as other baselines.

V-PVP(Cui et al., 2026b). V-PVP is an AI-generated video detection method that leverages pretrained video backbones for discriminative forensic representation learning. We follow the oficial implementation and train V-PVP using the same training data, video sampling strategy, and evaluation protocol as other baselines.

## D Velocity Representation Analysis

To investigate whether pretrained generative models inherently encode generation-aware information, we analyze the velocity representations extracted from an untrained Wan2.2- A14B model. Specifically, we compare the velocity responses of real and AI-generated videos at diferent noise levels without updating any parameters of the generative model. As shown in Fig. 6, the velocity representations already exhibit a strong discriminative capability between real and AI-generated videos, even when the Wan2.2-A14B backbone is completely frozen and has never been optimized for synthetic video detection. This observation suggests that the velocity field of a pretrained generative model implicitly captures characteristics associated with the underlying video generation process, providing a generation-aware forensic cue beyond conventional appearance-based features.

We further examine how noise perturbation afects the discriminability of the velocity representations. As the noise level increases, the separation between real and AIgenerated videos becomes progressively more pronounced, indicating that noise perturbation exposes additional generation-related differences in the velocity space. Notably, the discriminative performance reaches a peak around timestep t = 100, after which further increasing the noise level yields only marginal improvements. This saturation suggests that a moderate amount of noise is suficient to reveal the generation-dependent characteristics encoded in the velocity field, while excessive perturbation provides limited additional forensic information. Based on this observation, we adopt $\mathcal { T } = \{ 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ as the probe timesteps in TRACE.

![](images/6628e8755e6f5739f2e240f332e0914793b6f82a7135d2f985970d1a728b96dd.jpg)  
Figure 6: Detection performance based on velocity representations under diferent noise levels. A relatively low noise level of 100 already yields strong performance, with further noise perturbation providing limited improvement.

## E ACC@5% FPR Evaluation Results

## E.1 ACC@5% FPR on Open-Source Generators

Table 4: 22.59% Improvement in ACC@5%FPR for Open-Source Video Generators.
<table><tr><td rowspan="2">Method</td><td colspan="6">I2V</td><td colspan="10"></td><td colspan="3">V2V</td><td colspan="2"></td></tr><tr><td>Easy Animate</td><td>LTX</td><td>Pyramid Flow</td><td>SEINE</td><td>SVD</td><td>Video Crafter</td><td>Acc Video</td><td>Animate Diff</td><td>Cogvideo x1.5</td><td>Easy Animate</td><td>Hunyuan</td><td>IPOC</td><td>LTX</td><td>Open Sora</td><td>Pyramid Flow</td><td>Rep Video</td><td>Video Crafter</td><td>Wan 2.1</td><td>Cogvideo x1.5</td><td>LTX</td><td>AVG</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Image Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNNSpot</td><td>61.65</td><td>61.43</td><td>76.35</td><td>78.65</td><td>91.62</td><td>52.00</td><td>54.42</td><td>49.38</td><td>65.93</td><td>55.78</td><td>57.33</td><td>73.57</td><td>86.85</td><td>97.48</td><td>89.25</td><td>64.30</td><td>61.25</td><td>53.47</td><td>59.20</td><td>57.42</td><td>67.37</td></tr><tr><td>UnivFD</td><td>61.70</td><td>67.80</td><td>73.82</td><td>92.10</td><td>87.57</td><td>54.23</td><td>52.45</td><td>64.25</td><td>73.33</td><td>54.80</td><td>52.67</td><td>72.63</td><td>76.12</td><td>97.48</td><td>82.90</td><td>65.25</td><td>67.53</td><td>50.63</td><td>67.87</td><td>61.88</td><td>68.85</td></tr><tr><td>NPR</td><td>62.75</td><td>76.82</td><td>87.33</td><td>83.27</td><td>92.90</td><td>51.58</td><td>51.62</td><td>51.28</td><td>70.42</td><td>51.80</td><td>56.12</td><td>70.08</td><td>90.15</td><td>97.50</td><td>84.77</td><td>62.48</td><td>57.30</td><td>50.85</td><td>74.67</td><td>70.38</td><td>69.70</td></tr><tr><td>DDA</td><td>51.10</td><td>50.63</td><td>56.08</td><td>57.60</td><td>62.02</td><td>50.05</td><td>55.30</td><td>52.67</td><td>56.80</td><td>58.45</td><td>55.87</td><td>53.08</td><td>55.27</td><td>97.50</td><td>69.23</td><td>52.43</td><td>53.05</td><td>52.93</td><td>50.95</td><td>50.83</td><td>57.09</td></tr><tr><td>D3</td><td>63.45</td><td>68.10</td><td>75.88</td><td>93.22</td><td>91.92</td><td>63.45</td><td>53.17</td><td>59.67</td><td>72.52</td><td>58.52</td><td>53.43</td><td>76.37</td><td>77.57</td><td>97.50</td><td>88.05</td><td>67.30</td><td>72.22</td><td>52.63</td><td>65.78</td><td>62.53</td><td>70.66</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Video Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeMamba</td><td></td><td>57.33</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>70.35</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.23</td></tr><tr><td>DeCoF</td><td>57.62</td><td>64.70</td><td>65.28 70.02</td><td>70.52 80.62</td><td>83.35</td><td>50.57 52.32</td><td>53.35</td><td>56.77</td><td>65.05</td><td>59.80</td><td>56.87</td><td>65.13</td><td>81.25</td><td>97.50</td><td>77.30</td><td>57.17</td><td>76.93</td><td>51.62</td><td>57.37</td><td>54.68 62.85</td><td>71.68</td></tr><tr><td>WaveRep</td><td>60.02 49.93</td><td>49.85</td><td>49.97</td><td>50.22</td><td>90.82</td><td></td><td>60.48</td><td>59.50</td><td>77.63</td><td>63.38</td><td>61.28</td><td>84.98</td><td></td><td>97.50</td><td>93.65</td><td>75.95</td><td>76.75</td><td>55.32</td><td>64.67</td><td></td><td>49.87</td></tr><tr><td>ReStraV</td><td>91.87</td><td>49.02</td><td>49.17</td><td>47.62</td><td>50.20 48.12</td><td>49.77 47.70</td><td>49.65</td><td>49.90 47.52</td><td>50.05</td><td>49.95</td><td>48.98</td><td>49.65</td><td>50.27 49.70</td><td></td><td>50.13 48.83</td><td>49.85</td><td>50.57</td><td>49.72</td><td>49.33 51.27</td><td>49.78 49.57</td><td>59.83</td></tr><tr><td>Qwen2.5-ViT</td><td>49.70</td><td>50.20</td><td>51.05</td><td>50.27</td><td>50.08</td><td>50.93</td><td>48.73 52.03</td><td>51.88</td><td>59.55 53.92</td><td>85.12 50.50</td><td>48.38</td><td>80.80</td><td>51.02 51.28</td><td>94.60 53.92</td><td>51.70</td><td>86.37 51.55</td><td>47.52 51.70</td><td>63.75 50.47</td><td>53.28</td><td>50.43</td><td>51.51</td></tr><tr><td>STALL</td><td>50.65</td><td>49.77</td><td>49.65</td><td>49.30</td><td>51.57</td><td>52.22</td><td>53.62</td><td>55.93</td><td>53.00</td><td>55.67</td><td>52.02 53.78</td><td>53.28 56.97</td><td>49.48</td><td>53.18</td><td>52.92</td><td>51.67</td><td>50.77</td><td>53.17</td><td>50.60</td><td>49.83</td><td>52.19</td></tr><tr><td>V-PVP</td><td>57.55</td><td>61.05</td><td>68.17</td><td>74.60</td><td>81.32</td><td>68.85</td><td>62.43</td><td>71.75</td><td>63.42</td><td>62.85</td><td>61.08</td><td>70.30</td><td>66.68</td><td>97.35</td><td>81.33</td><td>64.92</td><td>85.70</td><td>57.43</td><td>58.90</td><td>57.75</td><td>68.67</td></tr><tr><td>TRACE</td><td>93.98</td><td>95.58</td><td>97.08</td><td>97.47</td><td>97.47</td><td>97.47</td><td>91.13</td><td>97.47</td><td>92.38</td><td>95.85</td><td>92.12</td><td>94.92</td><td>96.82</td><td>97.47</td><td>97.42</td><td>90.85</td><td>97.47</td><td>78.08</td><td>92.67</td><td>91.77</td><td>94.27</td></tr></table>

Table 4 reports ACC@5% FPR on the open-source generator subset of AIGVDBench. TRACE achieves an average accuracy of 94.27%, substantially exceeding the strongest baseline, DeCoF (71.68%), by 22.59 percentage points. TRACE obtains the highest accuracy in 19 of the 20 generator–task configurations; the only exception is the in-domain

Open-Sora setting, where TRACE achieves 97.47%, closely matching the best baseline result of 97.50%. More importantly, TRACE maintains consistently high accuracy across diverse generator–task configurations despite being trained exclusively on Open-Sora. This consistent performance across unseen generation sources highlights the strong cross-generator generalization of TRACE and suggests that its trajectory-based representation is less dependent on generator-specific artifacts.

## E.2 ACC@5% FPR on Closed-Source Generators

Table 5: 23.27% Improvement in ACC@5%FPR over Existing Methods on Closed-Source Video Generators.
<table><tr><td rowspan=2 colspan=13>Closed-Source ApproachesMethod                                                                                        AVGGen2                                                              Vidu   Wan</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=13>AI-Generated Image Detection Models</td></tr><tr><td rowspan=5 colspan=1>CNNSpotUnivFDNPRDDAD3</td><td rowspan=2 colspan=1>58.7257.66</td><td rowspan=2 colspan=1>65.3261.40</td><td rowspan=1 colspan=1>57.08</td><td rowspan=1 colspan=1>62.08</td><td rowspan=1 colspan=1>64.32</td><td rowspan=2 colspan=1>62.4259.62</td><td rowspan=2 colspan=1>63.7069.70</td><td rowspan=1 colspan=1>65.42</td><td rowspan=1 colspan=1>66.54</td><td rowspan=1 colspan=1>70.56</td><td rowspan=1 colspan=1>62.90</td><td rowspan=1 colspan=1>63.55</td></tr><tr><td rowspan=1 colspan=1>57.76</td><td rowspan=1 colspan=1>59.58</td><td rowspan=1 colspan=1>66.56</td><td rowspan=1 colspan=1>64.92</td><td rowspan=1 colspan=1>60.66</td><td rowspan=1 colspan=1>67.62</td><td rowspan=1 colspan=1>60.48</td><td rowspan=4 colspan=1>62.3663.1565.8961.91</td></tr><tr><td rowspan=1 colspan=1>58.68</td><td rowspan=1 colspan=1>65.78</td><td rowspan=1 colspan=1>57.04</td><td rowspan=1 colspan=1>62.58</td><td rowspan=1 colspan=1>65.32</td><td rowspan=1 colspan=1>63.48</td><td rowspan=1 colspan=1>59.40</td><td rowspan=1 colspan=1>66.98</td><td rowspan=1 colspan=1>66.80</td><td rowspan=1 colspan=1>66.52</td><td rowspan=1 colspan=1>62.04</td></tr><tr><td rowspan=2 colspan=1>62.3457.82</td><td rowspan=2 colspan=1>67.7061.30</td><td rowspan=2 colspan=1>64.4057.80</td><td rowspan=2 colspan=1>65.6859.08</td><td rowspan=2 colspan=1>74.0463.12</td><td rowspan=2 colspan=1>66.0059.20</td><td rowspan=2 colspan=1>63.8672.82</td><td rowspan=1 colspan=1>65.60</td><td rowspan=2 colspan=1>63.6059.78</td><td rowspan=2 colspan=1>65.6465.12</td><td rowspan=2 colspan=1>65.9060.64</td></tr><tr><td rowspan=1 colspan=1>64.28</td></tr><tr><td rowspan=1 colspan=13>AI-Generated Video Detection Models</td></tr><tr><td rowspan=2 colspan=1>DeMambaDeCoF</td><td rowspan=1 colspan=1>62.46</td><td rowspan=1 colspan=1>66.28</td><td rowspan=1 colspan=1>59.84</td><td rowspan=1 colspan=1>63.22</td><td rowspan=1 colspan=1>73.76</td><td rowspan=1 colspan=1>63.84</td><td rowspan=1 colspan=1>64.82</td><td rowspan=1 colspan=1>63.72</td><td rowspan=1 colspan=1>68.16</td><td rowspan=1 colspan=1>67.78</td><td rowspan=1 colspan=1>61.82</td><td rowspan=1 colspan=1>65.06</td></tr><tr><td rowspan=1 colspan=1>58.60</td><td rowspan=1 colspan=1>64.24</td><td rowspan=1 colspan=1>57.12</td><td rowspan=1 colspan=1>65.48</td><td rowspan=1 colspan=1>79.54</td><td rowspan=1 colspan=1>65.50</td><td rowspan=1 colspan=1>87.38</td><td rowspan=1 colspan=1>69.22</td><td rowspan=1 colspan=1>71.68</td><td rowspan=1 colspan=1>72.30</td><td rowspan=1 colspan=1>65.32</td><td rowspan=1 colspan=1>68.76</td></tr><tr><td rowspan=3 colspan=1>WaveRepReStraVQwen2.5-ViT</td><td rowspan=1 colspan=1>60.24</td><td rowspan=1 colspan=1>59.60</td><td rowspan=1 colspan=1>59.28</td><td rowspan=1 colspan=1>59.10</td><td rowspan=1 colspan=1>59.76</td><td rowspan=1 colspan=1>60.32</td><td rowspan=1 colspan=1>59.48</td><td rowspan=1 colspan=1>58.84</td><td rowspan=1 colspan=1>59.46</td><td rowspan=1 colspan=1>59.90</td><td rowspan=1 colspan=1>59.26</td><td rowspan=1 colspan=1>59.57</td></tr><tr><td rowspan=1 colspan=1>57.06</td><td rowspan=1 colspan=1>57.50</td><td rowspan=1 colspan=1>57.04</td><td rowspan=1 colspan=1>57.46</td><td rowspan=1 colspan=1>58.86</td><td rowspan=1 colspan=1>59.16</td><td rowspan=1 colspan=1>57.06</td><td rowspan=1 colspan=1>58.63</td><td rowspan=1 colspan=1>57.34</td><td rowspan=1 colspan=1>57.96</td><td rowspan=1 colspan=1>60.64</td><td rowspan=1 colspan=1>58.06</td></tr><tr><td rowspan=1 colspan=1>69.32</td><td rowspan=1 colspan=1>61.28</td><td rowspan=1 colspan=1>68.24</td><td rowspan=1 colspan=1>60.28</td><td rowspan=1 colspan=1>63.46</td><td rowspan=1 colspan=1>59.84</td><td rowspan=1 colspan=1>60.38</td><td rowspan=1 colspan=1>59.42</td><td rowspan=1 colspan=1>61.16</td><td rowspan=1 colspan=1>59.64</td><td rowspan=1 colspan=1>60.24</td><td rowspan=1 colspan=1>62.11</td></tr><tr><td rowspan=1 colspan=1>STALL</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>68.28</td><td rowspan=1 colspan=1>69.86</td><td rowspan=1 colspan=1>71.28</td><td rowspan=1 colspan=1>72.54</td><td rowspan=1 colspan=1>66.30</td><td rowspan=1 colspan=1>67.52</td><td rowspan=1 colspan=1>64.40</td><td rowspan=1 colspan=1>69.08</td><td rowspan=1 colspan=1>63.70</td><td rowspan=1 colspan=1>68.08</td><td rowspan=1 colspan=1>68.73</td></tr><tr><td rowspan=2 colspan=1>V-PVPTRACE</td><td rowspan=1 colspan=1>67.98</td><td rowspan=1 colspan=1>69.10</td><td rowspan=1 colspan=1>61.84</td><td rowspan=1 colspan=1>64.62</td><td rowspan=1 colspan=1>78.86</td><td rowspan=1 colspan=1>64.42</td><td rowspan=1 colspan=1>62.82</td><td rowspan=1 colspan=1>67.48</td><td rowspan=1 colspan=1>82.18</td><td rowspan=1 colspan=1>67.92</td><td rowspan=1 colspan=1>66.54</td><td rowspan=1 colspan=1>68.52</td></tr><tr><td rowspan=1 colspan=1>96.06</td><td rowspan=1 colspan=1>90.10</td><td rowspan=1 colspan=1>96.08</td><td rowspan=1 colspan=1>92.92</td><td rowspan=1 colspan=1>96.18</td><td rowspan=1 colspan=1>83.14</td><td rowspan=1 colspan=1>93.26</td><td rowspan=1 colspan=1>92.98</td><td rowspan=1 colspan=1>96.82</td><td rowspan=1 colspan=1>88.16</td><td rowspan=1 colspan=1>86.68</td><td rowspan=1 colspan=1>92.03</td></tr></table>

Table 5 reports ACC@5% FPR on the closed-source generator subset. TRACE achieves an average accuracy of 92.03%, substantially outperforming DeCoF (68.76%), the strongest baseline by average accuracy, by 23.27 percentage points. Notably, TRACE consistently outperforms all compared methods across all 11 generator–task configurations, with accuracy ranging from 83.14% on Sora to 96.82% on Pika. Such consistent improvements across diverse closed-source generators demonstrate that TRACE generalizes efectively beyond the Open-Sora training distribution, despite having no access to these generators during training. The results further indicate that trajectory-based representations provide a robust forensic signal for distinguishing AI-generated videos under a strict 5% false-positive-rate constraint.

## E.3 ACC@5% FPR under Robustness Tests

We further evaluate the robustness of TRACE on AIGVDBench under JPEG compression $( q \in \{ 1 0 0 , 9 0 , 8 0 , 7 0 , 6 0 \} )$ ), Gaussian blur $( \sigma \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \} )$ , and resolution scaling $( s \in \{ 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \}$ , with restoration to the original size). Perturbations are applied to raw video frames prior to model-specific preprocessing. We report ACC@5% FPR across all 14 settings, as shown in Figure 7. TRACE consistently achieves the best performance across all evaluated distortion conditions, maintaining a clear advantage over competing methods under varying levels of compression, blur, and resolution changes. These results demonstrate that TRACE is not only efective across diverse generation sources, but also robust to common distortions that substantially alter the visual appearance of video inputs. The consistent performance under such perturbations further supports the generalizability of trajectory-based forensic representations beyond clean, unaltered video content.

## F AP Evaluation Results

## F.1 AP on Open-Source Generators

Table 6 reports AP on the open-source generator subset. TRACE achieves an average AP of 98.29%, substantially surpassing the strongest baseline, DeCoF (84.25%), by 14.04 percentage points. TRACE ranks first in 19 of the 20 generator–task configurations and ties for first on the in-domain Open-Sora setting. Moreover, TRACE achieves an AP above 96% in all configurations except Wan2.1 text-to-video, where it still attains 87.56%. The consistently high AP across diverse generation sources demonstrates that TRACE maintains reliable ranking capability beyond its Open-Sora training distribution, indicating that its trajectory-based representation provides strong and generalizable discrimination between real and AI-generated videos.

![](images/d8a5568eb42d16d0274d69adfd8228d2214f32bf06e1b2126789616bcd591a79.jpg)  
Figure 7: Robustness evaluation of TRACE in terms of ACC@5% FPR under various video distortions, including JPEG compression, Gaussian blur, and resizing.

Table 6: 14.04% Improvement in AP for Open-Source Video Generators.
<table><tr><td rowspan="2">Method</td><td colspan="5">I2V</td><td rowspan="2"></td><td colspan="7">T2V</td><td colspan="7"></td><td rowspan="2">V2V AVG</td></tr><tr><td>Easy Animate</td><td>Pyramid LTX Flow</td><td></td><td>SEINE</td><td>SVD</td><td>Video Crafter</td><td>Acc Animate Video Diff</td><td>Cogvideo x1.5</td><td>Easy Animate</td><td>Hunyuan</td><td>IPOC</td><td>LTX</td><td>Open Sora</td><td>Pyramid Flow</td><td>Rep Video</td><td>Video Crafter</td><td>Wan 2.1</td><td>Cogvideo LTX x1.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Image Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNNSpot</td><td>76.81</td><td>75.89</td><td>89.96</td><td>91.94</td><td>97.94</td><td>56.33</td><td>60.80</td><td>44.13</td><td>80.19 64.95</td><td>69.37</td><td>87.14</td><td>95.76</td><td>100.0</td><td>96.69</td><td>78.23</td><td>74.66</td><td>59.14</td><td>72.33</td><td>70.71</td><td>77.15</td></tr><tr><td>UnivFD</td><td>76.02</td><td>81.60</td><td>87.90</td><td>97.33</td><td>95.62</td><td>66.40</td><td>59.31</td><td>78.20</td><td>87.76</td><td>66.26 59.08</td><td>87.07</td><td>89.72</td><td>99.99</td><td>93.81</td><td>80.11</td><td>83.62</td><td>53.36</td><td>82.74</td><td>74.47</td><td>80.02</td></tr><tr><td>NPR</td><td>77.23</td><td>89.54</td><td>95.48</td><td>93.66</td><td>98.11</td><td>55.97</td><td>49.52</td><td>51.28</td><td>82.30 50.56</td><td>62.32</td><td>83.59</td><td>97.03</td><td>100.0</td><td>94.35</td><td>73.41</td><td>70.75</td><td>49.68</td><td>88.44</td><td>85.31</td><td>77.43</td></tr><tr><td>DDA</td><td>53.99</td><td>51.82</td><td>65.55</td><td>66.08</td><td>74.58</td><td>45.96</td><td>64.30</td><td>63.08</td><td>67.66 72.05</td><td>66.42</td><td>58.64</td><td>62.33</td><td>100.0</td><td>84.32</td><td>54.82</td><td>61.28</td><td>60.66</td><td>53.13</td><td>51.23</td><td>63.90</td></tr><tr><td>D3</td><td>78.10 83.20</td><td></td><td>89.81</td><td>98.01</td><td>97.41</td><td>79.88</td><td>61.91</td><td>73.15 87.14</td><td>72.61</td><td>60.18</td><td>90.01</td><td>90.97</td><td>100.0</td><td>96.07</td><td>82.17</td><td>88.17</td><td>58.33</td><td>81.56</td><td>77.40</td><td>82.30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AI-Generated Video Detection Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeMamba</td><td>70.27</td><td>69.98</td><td>81.21</td><td>87.11</td><td>94.42</td><td>55.97</td><td>61.80</td><td>71.17</td><td>80.15</td><td>75.52</td><td>70.51 80.30</td><td>85.09</td><td>100.0</td><td></td><td>68.23</td><td>91.58</td><td>59.10</td><td></td><td>65.44</td><td>76.50</td></tr><tr><td>DeCoF</td><td>75.37</td><td>80.99</td><td>86.25</td><td>92.98</td><td>97.17</td><td>58.85</td><td>76.68</td><td>73.47</td><td>91.03</td><td>80.09 77.99</td><td>94.97</td><td>93.05</td><td>100.0</td><td>91.52 98.39</td><td>90.29</td><td>90.84</td><td></td><td>70.70 80.61</td><td>79.31</td><td>84.25</td></tr><tr><td>WaveRep</td><td>49.91</td><td>49.17</td><td>49.74</td><td>50.18</td><td>49.76</td><td>49.59</td><td>48.94</td><td>49.32</td><td>50.41</td><td>51.00 46.62</td><td>49.54</td><td>50.94</td><td>49.88</td><td>50.90</td><td>49.68</td><td>52.60</td><td>66.73 49.38</td><td>47.83</td><td>49.29</td><td>49.73</td></tr><tr><td>ReStraV</td><td>97.52</td><td>46.12</td><td>48.40</td><td>45.03</td><td>55.46</td><td>38.47</td><td>45.11</td><td>33.18</td><td>72.13</td><td>93.27 46.11</td><td>89.05</td><td>50.30</td><td>98.69</td><td>45.02</td><td>93.71</td><td>33.06</td><td>76.96</td><td>53.15</td><td>49.10</td><td>60.49</td></tr><tr><td>Qwen2.5-ViT</td><td>50.74</td><td>49.96</td><td>53.82</td><td>51.51</td><td>51.08</td><td>51.88</td><td>56.92</td><td>53.88</td><td>61.74</td><td>52.41 57.27</td><td>59.24</td><td>53.66</td><td>61.36</td><td>55.83</td><td>55.66</td><td>56.20</td><td>52.60</td><td>58.65</td><td>50.90</td><td>54.77</td></tr><tr><td>STALL</td><td>51.46</td><td>47.92 75.19</td><td>48.22</td><td>46.95</td><td>52.66</td><td>55.07</td><td>59.16</td><td>66.87</td><td>57.53</td><td>64.13</td><td>60.52 65.86</td><td>47.65</td><td>60.23</td><td>57.94</td><td>54.84</td><td>57.99</td><td>58.76</td><td>51.28</td><td>48.93</td><td>55.70</td></tr><tr><td>V-PVP</td><td>70.76 98.39</td><td>99.04</td><td>84.03 99.69</td><td>89.02</td><td>92.97</td><td>83.86</td><td>76.34</td><td>86.38</td><td>77.56</td><td>78.24</td><td>75.89 85.23</td><td>81.99</td><td>99.90</td><td>93.17</td><td>78.91</td><td>94.88</td><td>70.04</td><td>72.30</td><td>70.68</td><td>81.87</td></tr><tr><td>TRACE</td><td></td><td></td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>96.75</td><td>99.95</td><td>97.51</td><td>99.25</td><td>97.39 98.88</td><td>99.54</td><td>100.0</td><td>99.94</td><td>96.79</td><td>100.0</td><td>87.56</td><td>97.71</td><td>97.38</td><td>98.29</td></tr></table>

## F.2 AP on Closed-Source Generators

Table 7: 25.06% Improvement in AP over Existing Methods on Closed-Source Video Generators.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=11>Closed-Source Approaches</td><td rowspan=2 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=1>Gen2</td><td rowspan=1 colspan=1>Gen3</td><td rowspan=1 colspan=1>Jimeng</td><td rowspan=1 colspan=1>Luma</td><td rowspan=1 colspan=1>OpenSora</td><td rowspan=1 colspan=1>Sora</td><td rowspan=1 colspan=1>Causvid24fps</td><td rowspan=1 colspan=1>Kling</td><td rowspan=1 colspan=1>Pika</td><td rowspan=1 colspan=1>Vidu</td><td rowspan=1 colspan=1>Wan</td></tr><tr><td rowspan=1 colspan=13>AI-Generated ImageDetection Models</td></tr><tr><td rowspan=2 colspan=1>CNNSpotUnivFD</td><td rowspan=1 colspan=1>34.59</td><td rowspan=1 colspan=1>57.87</td><td rowspan=1 colspan=1>25.27</td><td rowspan=1 colspan=1>50.38</td><td rowspan=1 colspan=1>56.53</td><td rowspan=1 colspan=1>52.12</td><td rowspan=1 colspan=1>59.84</td><td rowspan=1 colspan=1>54.12</td><td rowspan=1 colspan=1>60.46</td><td rowspan=1 colspan=1>71.40</td><td rowspan=2 colspan=1>50.4145.75</td><td rowspan=2 colspan=1>52.0948.99</td></tr><tr><td rowspan=1 colspan=1>33.66</td><td rowspan=1 colspan=1>43.14</td><td rowspan=1 colspan=1>27.79</td><td rowspan=1 colspan=1>41.46</td><td rowspan=1 colspan=1>64.83</td><td rowspan=1 colspan=1>42.55</td><td rowspan=1 colspan=1>71.76</td><td rowspan=1 colspan=1>55.33</td><td rowspan=1 colspan=1>46.37</td><td rowspan=1 colspan=1>66.24</td></tr><tr><td rowspan=3 colspan=1>NPRDDAD3</td><td rowspan=1 colspan=1>30.31</td><td rowspan=1 colspan=1>52.14</td><td rowspan=1 colspan=1>28.97</td><td rowspan=1 colspan=1>48.46</td><td rowspan=1 colspan=1>57.79</td><td rowspan=1 colspan=1>49.64</td><td rowspan=1 colspan=1>35.55</td><td rowspan=1 colspan=1>55.81</td><td rowspan=1 colspan=1>61.10</td><td rowspan=1 colspan=1>64.36</td><td rowspan=1 colspan=1>44.63</td><td rowspan=1 colspan=1>48.07</td></tr><tr><td rowspan=2 colspan=1>54.5831.79</td><td rowspan=2 colspan=1>67.4642.31</td><td rowspan=2 colspan=1>56.8030.32</td><td rowspan=2 colspan=1>61.1736.73</td><td rowspan=1 colspan=1>78.01</td><td rowspan=1 colspan=1>64.11</td><td rowspan=1 colspan=1>58.89</td><td rowspan=2 colspan=1>62.2954.76</td><td rowspan=2 colspan=1>57.2943.70</td><td rowspan=2 colspan=1>63.7059.13</td><td rowspan=2 colspan=1>62.3143.77</td><td rowspan=2 colspan=1>62.4246.65</td></tr><tr><td rowspan=1 colspan=1>57.56</td><td rowspan=1 colspan=1>37.42</td><td rowspan=1 colspan=1>75.67</td></tr><tr><td rowspan=1 colspan=13>AI-GeneratedVideo Detection Models</td></tr><tr><td rowspan=3 colspan=1>DeMambaDeCoFWaveRep</td><td rowspan=1 colspan=1>56.91</td><td rowspan=1 colspan=1>61.08</td><td rowspan=1 colspan=1>43.76</td><td rowspan=1 colspan=1>53.94</td><td rowspan=1 colspan=1>76.28</td><td rowspan=1 colspan=1>57.60</td><td rowspan=1 colspan=1>60.66</td><td rowspan=1 colspan=1>51.19</td><td rowspan=1 colspan=1>69.54</td><td rowspan=1 colspan=1>66.63</td><td rowspan=1 colspan=1>50.33</td><td rowspan=1 colspan=1>58.90</td></tr><tr><td rowspan=1 colspan=1>41.54</td><td rowspan=1 colspan=1>60.29</td><td rowspan=1 colspan=1>28.56</td><td rowspan=1 colspan=1>62.99</td><td rowspan=1 colspan=1>86.18</td><td rowspan=1 colspan=1>64.26</td><td rowspan=1 colspan=1>92.76</td><td rowspan=1 colspan=1>69.18</td><td rowspan=1 colspan=1>76.47</td><td rowspan=1 colspan=1>77.43</td><td rowspan=1 colspan=1>62.21</td><td rowspan=1 colspan=1>65.63</td></tr><tr><td rowspan=1 colspan=1>45.68</td><td rowspan=1 colspan=1>42.92</td><td rowspan=1 colspan=1>42.41</td><td rowspan=1 colspan=1>41.21</td><td rowspan=1 colspan=1>44.24</td><td rowspan=1 colspan=1>46.42</td><td rowspan=1 colspan=1>44.09</td><td rowspan=1 colspan=1>38.84</td><td rowspan=1 colspan=1>42.97</td><td rowspan=1 colspan=1>45.01</td><td rowspan=1 colspan=1>41.91</td><td rowspan=1 colspan=1>43.25</td></tr><tr><td rowspan=2 colspan=1>ReStraVQwen2.5-ViT</td><td rowspan=1 colspan=1>26.42</td><td rowspan=1 colspan=1>32.81</td><td rowspan=1 colspan=1>25.62</td><td rowspan=1 colspan=1>31.40</td><td rowspan=1 colspan=1>35.66</td><td rowspan=1 colspan=1>44.26</td><td rowspan=1 colspan=1>32.75</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>32.27</td><td rowspan=1 colspan=1>35.71</td><td rowspan=1 colspan=1>45.70</td><td rowspan=1 colspan=1>34.28</td></tr><tr><td rowspan=1 colspan=1>68.06</td><td rowspan=1 colspan=1>48.56</td><td rowspan=1 colspan=1>66.41</td><td rowspan=1 colspan=1>45.97</td><td rowspan=1 colspan=1>56.16</td><td rowspan=1 colspan=1>43.18</td><td rowspan=1 colspan=1>46.80</td><td rowspan=1 colspan=1>41.13</td><td rowspan=1 colspan=1>49.05</td><td rowspan=1 colspan=1>44.35</td><td rowspan=1 colspan=1>44.27</td><td rowspan=1 colspan=1>50.36</td></tr><tr><td rowspan=1 colspan=1>STALL</td><td rowspan=1 colspan=1>80.45</td><td rowspan=1 colspan=1>70.48</td><td rowspan=1 colspan=1>71.19</td><td rowspan=1 colspan=1>74.80</td><td rowspan=1 colspan=1>76.58</td><td rowspan=1 colspan=1>66.22</td><td rowspan=1 colspan=1>70.92</td><td rowspan=1 colspan=1>61.18</td><td rowspan=1 colspan=1>71.60</td><td rowspan=1 colspan=1>60.45</td><td rowspan=1 colspan=1>68.25</td><td rowspan=1 colspan=1>70.19</td></tr><tr><td rowspan=2 colspan=1>V-PVPTRACE</td><td rowspan=1 colspan=1>67.67</td><td rowspan=1 colspan=1>69.55</td><td rowspan=1 colspan=1>52.73</td><td rowspan=1 colspan=1>60.48</td><td rowspan=1 colspan=1>84.66</td><td rowspan=1 colspan=1>60.69</td><td rowspan=1 colspan=1>59.17</td><td rowspan=1 colspan=1>65.71</td><td rowspan=1 colspan=1>87.68</td><td rowspan=1 colspan=1>68.52</td><td rowspan=1 colspan=1>63.40</td><td rowspan=1 colspan=1>67.30</td></tr><tr><td rowspan=1 colspan=1>99.22</td><td rowspan=1 colspan=1>93.62</td><td rowspan=1 colspan=1>99.25</td><td rowspan=1 colspan=1>96.78</td><td rowspan=1 colspan=1>99.23</td><td rowspan=1 colspan=1>84.22</td><td rowspan=1 colspan=1>96.99</td><td rowspan=1 colspan=1>96.98</td><td rowspan=1 colspan=1>99.83</td><td rowspan=1 colspan=1>92.47</td><td rowspan=1 colspan=1>89.18</td><td rowspan=1 colspan=1>95.25</td></tr></table>

Table 7 reports AP on the closed-source generator subset. TRACE achieves an average AP of 95.25%, substantially exceeding the strongest baseline, STALL (70.19%), by 25.06 percentage points. TRACE leads in all 11 generator–task configurations, reaching APs of 99.83% on Pika, 99.25% on Jimeng, and 99.22% on Gen2. Although Sora remains the most challenging setting, TRACE still achieves an AP of 84.22%, substantially higher than the best baseline at 66.22%. Notably, these closed-source generators are unseen during training, yet TRACE maintains consistently strong ranking performance across all evaluated settings. This result further highlights the strong cross-generator generalization of trajectory-based forensic representations, which remain efective even when the evaluated generators are entirely outside the training distribution.

![](images/3b91485a4198c579d89f148b87acf2613a1dd97154d7392806f7269046a65343.jpg)  
Figure 8: Robustness evaluation of TRACE in terms of AP under diferent video distortions, including JPEG compression, Gaussian blur, and resizing.

## F.3 AP under Robustness Tests

We further evaluate the robustness of TRACE on AIGVDBench under JPEG compression $( q \in \{ 1 0 0 , 9 0 , 8 0 , 7 0 , 6 0 \} )$ ), Gaussian blur $( \sigma \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \} )$ , and resolution scaling $( s \in \{ 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \}$ , with restoration to the original size). Perturbations are applied to raw video frames prior to model-specific preprocessing. We report AP across all 14 settings, as shown in Figure 8. TRACE consistently achieves the best AP across all evaluated distortion conditions and maintains a clear advantage over competing methods under diferent perturbation strengths. This consistent ranking performance indicates that TRACE is robust to substantial changes in the visual appearance of video inputs and does not rely heavily on fragile, distortion-sensitive artifacts. More importantly, the stability of its performance across diverse distortions further demonstrates the generalizability of trajectory-based forensic representations beyond clean video inputs.

## G Visualization Analysis

To better understand the forensic evidence exploited by TRACE, we conduct a gradientbased saliency analysis to investigate the regions contributing to its predictions. This visualization provides qualitative insights into whether TRACE relies primarily on global semantic content or localized, generation-related forensic cues.

## G.1 Gradient-based Saliency Analysis

We first visualize the gradient-based saliency maps of TRACE to examine the spatial regions that contribute most to its forensic predictions. Given the predicted fake score $f ( \mathbf { x } )$ , we compute the gradient of the score with respect to the input representation and aggregate the gradient magnitude over the feature channels to obtain a saliency map:

$$
S ( \mathbf { x } ) = \left\| \frac { \partial f ( \mathbf { x } ) } { \partial \mathbf { x } } \right\| .\tag{19}
$$

![](images/67f667daad49ccf82de91c7f8a04895531bc27f6d89bbae27f2de03c928529e0.jpg)  
Figure 9: Saliency analysis. Grad-CAM visualizations of TRACE at the probe timestep t=50 on AI-generated videos. The top row shows the original frames, while the bottom row presents the corresponding saliency overlays.

The resulting saliency maps are normalized and projected onto the corresponding video frames for visualization. As shown in Figure 9, TRACE primarily focuses on localized regions rather than uniformly attending to the entire frame. These highlighted regions may contain fine-grained generation-related artifacts or inconsistencies that are dificult to identify from high-level semantic information alone.

Furthermore, the highlighted regions exhibit a degree of spatial continuity across neighboring frames. This observation suggests that TRACE captures structured forensic cues associated with the evolution of video content over time, rather than relying solely on isolated spatial artifacts. Such behavior is consistent with the design of Trajectory Consistency Estimation, which models frame-to-frame diferences in velocity representations under shared probe conditions. Overall, the visualization provides qualitative evidence that TRACE exploits localized and temporally structured cues for AI-generated video detection.

## G.2 Occlusion Analysis

To further assess the functional importance of the regions highlighted by the saliency maps, we conduct an occlusion-based analysis. Unlike gradient-based visualization, which measures local prediction sensitivity, occlusion analysis directly evaluates the efect of removing specific input regions on the model’s prediction. For an input x and its occluded version $\bar { \mathbf { x } ^ { ( r ) } }$ , where region r is replaced with a neutral value, we define the occlusion importance as

$$
\Delta _ { r } = f ( \mathbf { x } ) - f ( \mathbf { x } ^ { ( r ) } ) .\tag{20}
$$

A larger ∆<sub>r</sub> indicates that masking region r causes a greater decrease in the predicted fake score, suggesting that the region makes a stronger contribution to the model’s decision. Figures 10, 11 and 12 present occlusion sensitivity maps for three representative video examples, comparing TRACE with several competing methods. Across these examples, TRACE exhibits stronger and more spatially concentrated responses to specific patches, indicating that its predictions are particularly sensitive to localized regions containing informative forensic cues. Notably, the highlighted regions frequently correspond to motion-related structures and locally varying content across neighboring frames. This observation suggests that TRACE is sensitive to temporal changes and motion-related inconsistencies, rather than relying exclusively on static appearance artifacts.

![](images/958ce2f17915a06740b6bbd0628c644f3623b2335f55264b1cedb4eb63884a3f.jpg)  
Figure 10: Occlusion sensitivity analysis. Patch-occlusion sensitivity maps of TRACE, DDA, V-PVP, and STALL on AI-generated videos. The left column shows the original frames, while the right column presents the corresponding occlusion overlays, where $\Delta =$ $\ell _ { \mathrm { o r i g } } - \ell _ { \mathrm { m a s k e d } }$ denotes the change in the predicted fake logit after patch masking. All methods share the same color scale.

![](images/b9b5241facf858622b2766d3b75dfb02945a9a339b5866c438a1d3896f1fd5f6.jpg)  
Figure 11: Occlusion sensitivity analysis. Patch-occlusion sensitivity maps of TRACE, DDA, V-PVP, and STALL on AI-generated videos. The left column shows the original frames, while the right column presents the corresponding occlusion overlays, where $\Delta =$ $\ell _ { \mathrm { o r i g } } - \ell _ { \mathrm { m a s k e d } }$ denotes the change in the predicted fake logit after patch masking. All methods share the same color scale.

Moreover, occluding the highlighted regions produces a more pronounced change in TRACE’s prediction, whereas masking less relevant regions generally leads to smaller variations in the fake score. These results provide intervention-based evidence that the regions identified by TRACE are functionally involved in its forensic decisions, rather than merely exhibiting high gradient sensitivity.

Taken together, the saliency and occlusion analyses ofer complementary evidence for the learned forensic representations. The former identifies regions that are sensitive to the model’s prediction, while the latter verifies their functional contribution through direct input perturbation. Their agreement suggests that TRACE captures localized and structured forensic cues, including cues related to temporal content variations, beyond broad semantic information alone.

## H t-SNE Visualization

Figure 13 visualizes the fused trajectory features of TRACE and competing methods across four datasets, resulting in 16 t-SNE projections. TRACE exhibits more consistent real– generated separation across datasets, indicating its ability to learn generalizable generationaware representations. In contrast, competing methods show clear separation only on a subset of datasets or those closely related to their training distributions, while exhibiting greater overlap on other datasets. These visualizations provide qualitative evidence of the stronger cross-dataset generalization of TRACE’s learned representations.

![](images/402e025a1a1bab03e624001cdedf417e5251d830e393c6523115908b21ebc347.jpg)  
Figure 12: Occlusion sensitivity analysis. Patch-occlusion sensitivity maps of TRACE, DDA, V-PVP, and STALL on AI-generated videos. The left column shows the original frames, while the right column presents the corresponding occlusion overlays, where ∆ = $\ell _ { \mathrm { o r i g } } - \ell _ { \mathrm { m a s k e d } }$ denotes the change in the predicted fake logit after patch masking. All methods share the same color scale.

![](images/30ae0c5390ea3248ae113720d3425e26a677a9c9349f1ff7e705c3910f2ae264.jpg)  
Figure 13: t-SNE comparison of detector representations on AIGVDBench. Rows correspond to TRACE, DDA, V-PVP, and STALL, while columns represent Open-Sora (T2V), SVD (I2V), CogVideoX1.5 (V2V), and Jimeng (closed-source). Each panel visualizes the representations of 2,000 real test videos (blue) and 2,000 generated test videos (colored) from the corresponding source. TRACE consistently achieves clearer real–fake separation across diferent generator types, whereas prior methods exhibit better separation mainly on sources aligned with their training distributions and substantial overlap on unseen sources.