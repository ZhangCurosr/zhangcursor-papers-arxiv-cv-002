# From Legacy to Contemporary: Scalable Color Modernization for Long Videos

Dongyun Kim<sup>∗</sup>, Jiwoo Choi<sup>∗</sup>, Jaeseong Lee, Seon Joo Kim

Yonsei University

kdyun01@yonsei.ac.kr jiw55@yonsei.ac.kr

## Abstract

Bringing legacy video back to contemporary screens does not necessarily make it look contemporary. Earlier imaging pipelines often leave color casts, muted chroma, weak contrast, and tonal imbalance, while correcting these characteristics still relies heavily on expert colorists and scales poorly to long, multi-shot videos. We formulate long video color modernization, a task that updates the chromatic and tonal appearance of legacy color footage while preserving scene content and coherence across frames and shots. We introduce a scalable, training-free framework that jointly modernizes grids of temporally adjacent keyframes using a pretrained text-guided image editing model, producing coherent contemporary references without paired supervision. Compact shot-specific color transforms then propagate these adjustments throughout each shot, preserving spatial structure and temporal stability. We conduct a multidimensional evaluation on diverse legacy videos spanning five decades. We combine a human evaluation of perceived modernness with quantitative measures of content fidelity, temporal stability, perceptual quality, and eficiency, and introduce Source-Weighted Inter-shot Grade Consistency (SW-IGC) to assess grade coherence across shots. On our evaluation set, our framework achieves the highest perceived modernness while remaining competitive across complementary measures, opening a path toward the systematic study of long video color modernization. Code is available at https://github.com/jiwoomood/Video\_Color\_Modernization.git

## 1 Introduction

Legacy videos continue to return to contemporary screens through restoration, redistribution, and archival release. Yet their return alone does not ensure that they meet the visual expectations of today’s viewers. The issue is often not what they depict, but the color appearance inherited from earlier imaging pipelines. Limited capture devices, analog or early digital processing, and broadcast transfer can leave persistent color casts, desaturation, reduced contrast, and tonal imbalance. These color and tonal degradations make otherwise valuable footage feel visually dated, motivating color modernization methods that preserve scene content and structure.

Growing interest in AI-assisted remastering, automatic color correction, and reference-based look matching reflects the practical demand for such modernization. However, high quality results still typically rely on expert colorists to adjust exposure, contrast, and chroma while maintaining consistency across frames and shots. Since this process depends on aesthetic judgment and technical expertise, it scales poorly to long videos with diverse scenes and lighting. These limitations call for automated color modernization that can scale without compromising video coherence.

We formalize this problem as long video color modernization: adjusting the color and tone of legacy footage toward a broadly contemporary appearance while preserving scene content and structure. Unlike restoration (Chan et al. 2022; Lin and Simo-Serra 2024; Mao et al. 2025; Wan et al. 2022), which aims to recover degraded visual signals, and colorization (Lei and Chen 2019; Zhang et al. 2019; Yang et al. 2024; Chen et al. 2024), which infers missing chroma for grayscale footage, our goal is to modernize footage that already contains color yet appears visually dated.

This formulation raises two technical challenges and one evaluation challenge. First, image-level modernization is underdetermined. Modern color is not a unique target: multiple color grades may be perceived as contemporary, and paired old-to-modern supervision is generally unavailable. Second, long videos require color transformations that are both shot-adaptive and video-coherent. Frame-dependent transformations can fluctuate with content and introduce flicker, whereas a single global transformation can fail to account for shot-specific lighting and color statistics. Efective modernization must therefore remain stable within each shot while avoiding arbitrary grade drift across visually related shots. Third, to our knowledge, evaluating long video color modernization remains underexplored. Existing assessments largely focus on individual images (Gunawan et al. 2023) or short clips (Chen et al. 2017; Deng et al. 2021; Wen, Gao, and Zou 2023; Chen et al. 2023; Kar and Su 2025), leaving the broader requirements of long multi-shot videos less systematically examined. A meaningful evaluation therefore requires a multidimensional assessment encompassing perceived modernness, content preservation, temporal stability, cross-shot grade coherence, perceptual quality, and eficiency.

To address these technical challenges, we propose a scalable framework that combines keyframe modernization with shot-level propagation. We first divide the input video into shots and select one representative keyframe per shot. Temporally adjacent keyframes are then packed into a grid and jointly modernized using a pretrained text-guided image editing model (Black Forest Labs et al. 2025). The model’s learned generative prior provides plausible contemporary references without paired supervision, while joint processing promotes coherent grading across neighboring shots. Finally, we fit a compact, shot-specific transform to each source–modernized keyframe pair and propagate its chromatic and tonal adjustments throughout the corresponding shot. Per-shot estimation accommodates variations in lighting and color statistics, while applying the same pixelwise transform throughout each shot maintains stable grading without altering spatial structure. By applying generative editing only to representative keyframes, the framework scales eficiently to long multi-shot videos.

To systematically assess long video color modernization, we curate a diverse evaluation set of multi-shot legacy videos from the Condensed Movies Dataset (Bain et al. 2020), spanning five decades of footage. We evaluate perceived modernness through a user study and complement it with quantitative measures of content preservation, temporal stability, perceptual quality, and eficiency. We further introduce Source-Weighted Inter-shot Grade Consistency (SW-IGC), which evaluates inter-shot grade coherence by weighting shot-pair comparisons according to source color similarity. Among the eight evaluated versions, our framework achieves the best mean rank for perceived modernness, whereas several baselines do not surpass the unedited original. It also remains competitive across the complementary quantitative criteria.

Our contributions are threefold: (i) we formulate long video color modernization as a distinct task for updating the chromatic and tonal appearance of legacy color footage; (ii) we propose a scalable framework that jointly modernizes temporally adjacent keyframes and propagates their adjustments through compact, shot-specific color transforms; and (iii) we provide a multidimensional evaluation of this task, combining human judgments of perceived modernness with complementary quantitative measures and introducing SW-IGC for inter-shot grade coherence.

## 2 Related Work

## 2.1 Reference-based Image Color Style Transfer

Modernizing the color appearance of legacy footage is closely related to color grading, in which colorists adjust color and tone to achieve a desired visual look rather than recover a single objectively correct appearance. Because such an appearance is subjective and lacks a unique ground truth, it can be specified through a reference image and approached through color style transfer.

Previous methods transfer color from source–reference image pairs either by aligning feature statistics or by predicting color operators. Feature-statistics-based methods, such as WCT variants (Li et al. 2017, 2018; An et al. 2020; Chiu and Gurari 2022; Wen, Gao, and Zou 2023), whiten source features and recolor them using the statistics of the reference features. In contrast, color operators are commonly represented as locally afine bilateral grids (Xia et al. 2020) or parametric color mappings (Ho and Zhou 2021; Ke et al. 2023; Larchenko et al. 2025; Li et al. 2025; Gong et al. 2025). Since these methods alter image appearance without explicitly modifying spatial structure, they generally preserve image layout and can be applied at high resolution.

Our scalable framework supports various colorpropagation methods, but prioritizes deterministic color mapping over inference-time style transfer for eficient longvideo processing. Our evaluation shows that this training-free approach provides suficient visual quality.

## 2.2 Video Color Grading

Extending reference-based color style transfer to video requires keeping the appearance consistent across frames. Existing studies (Li et al. 2019; Deng et al. 2021; Wu et al. 2022; Wen, Gao, and Zou 2023; Kar and Su 2025) estimate a transformation operator for each frame or map an operator to the video without any temporal component such as optical flow or a temporal loss. These methods mainly address a single shot or a short clip, and do not explicitly consider coherent grading across the many shots of a long-form video.

Even studies that explicitly target video color grading (Shin et al. 2025; Guo et al. 2026) treat a sequence as one continuous clip, and do not address the color inconsistency that arises within a shot or between shots. Shin et al. (2025) generate a LUT learned from a source–reference pair, but apply that single LUT to the entire video, irrespective of shot boundaries. Guo et al. (2026) use LLM-based RAG and a tree-of-thoughts procedure to determine the parameters of a single 3D LUT, which is likewise applied uniformly across the whole log video. Recent video generation models ofer flexible end-to-end manipulation (Jiang et al. 2025; Yang et al. 2025), but are developed primarily for synthesizing or editing short, motion-rich clips rather than preserving the source content exactly. Their generative formulation can introduce unintended content changes and temporal inconsistencies, while the high cost of dense generation limits their applicability to long-form video modernization.

## 3 Method

## 3.1 Framework Overview

Given a long-form legacy color video $V ~ = ~ ( I _ { t } ) _ { t = 1 } ^ { T }$ , our goal is to produce a modernized video $\widetilde { V } = ( \widetilde { I _ { t } } ) _ { t = 1 } ^ { T }$ with a contemporary chromatic and tonal appearance while preserving the original content and maintaining temporal consistency. As illustrated in Fig. 1, our framework comprises three stages: keyframe extraction, image-level modernization, and color propagation. First, we partition the input video into shots and select a representative keyframe from each shot (Sec. 3.2). We then pack temporally adjacent keyframes and modernize them jointly using a pretrained text-guided image editing model to promote inter-shot consistency (Sec. 3.3). Finally, we estimate a shot-specific color mapping from each original–modernized keyframe pair and apply it throughout the corresponding shot (Sec. 3.4).

![](images/dd4a5b369eaff7772b42f096f04d9242f7a9056935d6a93c6a12d21e21601bc6.jpg)  
Figure 1: Overview of our long video color modernization framework. We first partition the input video into shots and select a representative keyframe from each shot. We then jointly modernize temporally adjacent keyframes. Finally, we estimate a shot-specific color mapping from each original–modernized keyframe pair and apply it to all frames in the corresponding shot.

## 3.2 Keyframe Extraction

To support eficient shot-level color propagation, we partition each video into shots and select one representative keyframe per shot through boundary detection, segment merging, and keyframe selection.

Shot detection & merging. We detect candidate boundaries using TransNet V2 (Souček and Lokoč 2024) with a recall-oriented low threshold, preventing a single color mapping from spanning distinct shots. To correct false splits caused by flashes or rapid motion, we greedily merge adjacent segments when the DINOv2 (Oquab et al. 2024) cosine similarity between the frames immediately surrounding their boundary exceeds a threshold. This semantic criterion helps distinguish transient within-shot appearance changes from true boundaries between distinct shots.

Keyframe selection. For each shot, we sample candidate frames and use a vision–language model (Bai et al. 2025) to select the clearest and most representative one, since visual clarity and shot representativeness are dificult to capture with a single heuristic. This keyframe serves both as the modernization input and as the reference for shot-level color propagation.

## 3.3 Image-Level Modernization

Image modernization is inherently ill-posed: there is no single ground-truth modernized result, and paired legacy and modern versions of the same visual content are generally unavailable. We therefore do not learn a direct old-tomodern mapping from paired supervision. Instead, we employ a pretrained text-guided image editing model to modernize the selected keyframes. Specifically, we use FLUX.1- Kontext (Black Forest Labs et al. 2025) with a fixed instruction that requests natural, contemporary color and tonal adjustments while preserving the original visual content and composition.

Processing each keyframe independently may produce inconsistent color rendering across neighboring shots. To improve inter-shot consistency, we adopt a three-step joint modernization procedure. First, during keyframe packing, we pack up to four temporally adjacent keyframes into a 2 × 2 grid. We then perform joint keyframe modernization by editing the packed grid in a single model invocation. By exposing the model to neighboring shots simultaneously, this joint processing encourages consistent chromatic and tonal adjustments across them. Finally, during keyframe unpacking, we split the edited grid into individual modernized keyframes. Each modernized keyframe is subsequently paired with its original counterpart to estimate the color mapping for the corresponding shot.

## 3.4 Color Propagation

This stage consists of two components. We first estimate a shot-specific color transform from each legacy–modernized keyframe pair, and then apply it consistently to all frames in the corresponding shot with highlight roll-of.

Shot-specific color mapping. For each shot s, we compute a global afine mapping that transports the RGB color distribution of its selected legacy keyframe to that of its modernized counterpart. Since the modernization stage targets primarily chromatic and tonal adjustments while preserving scene content, a single global mapping provides a compact approximation that can be applied pixelwise without altering spatial structure.

Let $( \pmb { \mu } _ { s } , \pmb { \Sigma } _ { s } )$ and $( { \pmb { \mu } } _ { s } ^ { \mathrm { m } } , { \pmb { \Sigma } } _ { s } ^ { \mathrm { m } } )$ denote the RGB means and covariance matrices of the legacy and modernized keyframes, respectively. We compute the shot-specific transform using the Monge–Kantorovitch linear (MKL) color mapping (Pitié and Kokaram 2007). Given an RGB color vector x ∈ R<sup>3</sup>,

$$
\begin{array} { r l } & { \mathcal { M } _ { s } ( { \bf x } ) = \mathrm { \bf T } _ { s } \left( { \bf x } - { \pmb \mu } _ { s } \right) + { \pmb \mu } _ { s } ^ { \mathrm { m } } , } \\ & { \qquad \mathrm { \bf T } _ { s } = \Sigma _ { s } ^ { - 1 / 2 } \left( \Sigma _ { s } ^ { 1 / 2 } \Sigma _ { s } ^ { \mathrm { m } } \Sigma _ { s } ^ { 1 / 2 } \right) ^ { 1 / 2 } \Sigma _ { s } ^ { - 1 / 2 } . } \end{array}\tag{1}
$$

Here, $( \cdot ) ^ { 1 / 2 }$ denotes the principal matrix square root. Under Gaussian approximations of the legacy and modernized color distributions, $\mathcal { M } _ { s }$ matches the target mean and covariance while minimizing the expected squared transport cost. Unlike independent per-channel mappings, the full $3 \times 3$ matrix $\mathbf { T } _ { s }$ captures cross-channel correlations and enables coupled adjustments across RGB channels.

Shot-level color propagation. The afine mapping can produce RGB values outside the valid range, causing hard clipping, particularly in bright regions. We therefore apply a luma-based highlight roll-of that smoothly compresses luma values above a fixed knee using a tanh-based curve and uniformly rescales the RGB channels to preserve hue. For every frame $I _ { t }$ in shot s, we apply $\mathcal { M } _ { s }$ pixelwise, followed by the highlight roll-of, to obtain $\widetilde { I _ { t } }$ . Using the same mapping and roll-of throughout the shot yields a temporally consistent color transformation across its frames.

## 4 Experiments

We evaluate whether our framework produces perceptually convincing modern color appearances while preserving source fidelity and temporal consistency. We assess modern color preference through a user study, complement it with quantitative and qualitative analyses, and examine key design choices across the framework through ablation studies.

## 4.1 Experimental Setup

Datasets. We evaluate on a curated subset of the Condensed Movies Dataset (Bain et al. 2020), a large-scale collection of key scenes from thousands of films. To capture diverse color characteristics across eras, we sample 20 videos per decade from the 1960s through the 2000s, yielding 100 videos, 3.9 hours of footage, and 3,390 shots. Each video is 2–4 minutes long; these short excerpts increase content diversity, while our framework is not limited to such durations.

Baselines. We consider two families of baselines, categorized by which stages of our framework they replace.

End-to-end video editing. We compare against two end-toend video editors, CogVideoX-5B (Yang et al. 2025) and Wan2.1-VACE-1.3B (Jiang et al. 2025), which directly modernize the input video without explicit keyframe processing or color propagation. This comparison evaluates whether dense video editing can preserve source content and temporal coherence over long videos.

Modular color propagation. The second family replaces only the propagation stage of our framework. We retain the same shot boundaries and modernized reference keyframes, and use CAP-VSTNet (Wen, Gao, and Zou 2023), Neural Preset (Ke et al. 2023), Video Color Grading (VCG) (Shin et al. 2025), and SA-LUT (Gong et al. 2025) to transfer the reference color grade to all frames within each shot. This controlled comparison isolates the efect of the propagation mechanism.

Evaluation metrics. Video modernization is inherently a one-to-many problem, as multiple color grades can provide valid modern renditions of the same source video, precluding evaluation against a single ground-truth grade. We therefore assess perceived modern color appearance through human judgments as the primary evaluation objective, complemented by measures of structural preservation, intra-shot temporal color consistency, perceptual and aesthetic quality, inter-shot grade consistency, and computational eficiency.

Modern color preference. We conduct a blinded user study with 65 participants recruited via Prolific to evaluate perceived modern color appearance. Each participant evaluates the same 15 video sets in randomized order. For each video, participants simultaneously view eight anonymized versions in a grid—the output of our method, the source video, and six baseline outputs—with their positions randomized. Including the source video allows us to assess whether the edited outputs are perceived as more contemporary than the original, beyond their relative preference among competing methods. Participants rank all eight versions from 1 (best) to 8 (worst) by modern color appearance (judged on color alone) and temporal consistency of colors and content. The 975 rankings are summarized by Mean Rank, Rank-1, Top-3, and the full rank distribution.

Structural and temporal consistency. We evaluate structural preservation using Edge-SSIM (Wang et al. 2004; Soria, Pomboza-Junez, and Sappa 2022) and intra-shot temporal color consistency using CDC (Liu et al. 2024), which measures temporal variation in framewise color distributions.

Perceptual and aesthetic quality. We assess perceptual quality using DeQA-Score (You et al. 2025) and the quality score of Q-Align (Wu et al. 2024) (Q-Align<sub>Q</sub>), and aesthetic quality using its aesthetic score (Q-Align<sub>A</sub>).

Eficiency. We measure the propagation throughput in frames per second.

Inter-shot grade consistency. We introduce Source-Weighted Inter-shot Grade Consistency (SW-IGC) to measure the consistency of shot-level grading transformations while weighting shot pairs by source color similarity. For each shot i, we fit an afine mapping $\mathcal { F } _ { i } ( \mathbf { x } ) = \mathbf { A } _ { i } \mathbf { x } ^ { \star } + \mathbf { b } _ { i }$ in linear RGB using aligned source–output pixel pairs. We then apply each fitted mapping to a fixed set of K=24 ColorChecker reference colors, enabling a content-independent comparison of the resulting color transformations. We define the discrepancy between the fitted mappings for shots i and $j$ as

$$
d _ { i j } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Delta E _ { 0 0 } ( \operatorname { L a b } ( \mathcal { F } _ { i } ( \mathbf { c } _ { k } ) ) , \operatorname { L a b } ( \mathcal { F } _ { j } ( \mathbf { c } _ { k } ) ) ) ,\tag{2}
$$

where $\mathbf { c } _ { k }$ denotes the k-th ColorChecker reference color. Let $\mathbf { m } _ { i } ^ { \mathrm { s r c } }$ be the channel-wise median CIELAB color of the sampled source pixels in shot i. The source similarity weight

![](images/8140118bc9f9defba725aa1641936a2d17df0f13a7b23031bc17b84b91013379.jpg)  
Figure 2: Distribution of contemporary color preference rankings over 975 complete rankings. Each bar shows the proportions of Rank 1, Ranks 2–3, and Ranks 4–8, with the mean rank reported on the right.

is defined as

$$
w _ { i j } = \exp \left( - \frac { \Delta E _ { 0 0 } \left( \mathbf { m } _ { i } ^ { \mathrm { s r c } } , \mathbf { m } _ { j } ^ { \mathrm { s r c } } \right) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{3}
$$

We set σ to the median pairwise source color distance within each video, adapting the weighting scale to the source color diferences across its shots. SW-IGC is then computed as

$$
\mathrm { S W - I G C } = \frac { \sum _ { i < j } w _ { i j } d _ { i j } } { \sum _ { i < j } w _ { i j } } .\tag{4}
$$

Lower SW-IGC indicates greater consistency among grading transformations for shots with similar source colors.

## 4.2 Comparisons

Modern color preference. Fig. 2 shows a clear overall preference for our method. Ours achieves the lowest mean rank of 2.31, which is 0.71 lower than that of CAP-VSTNet, the strongest baseline, and 1.31 lower than that of the source video. It receives the top rank in 37.9% of rankings, more than twice the 17.6% achieved by CAP-VSTNet, and places within the top three in 81.7%, compared with 66.3% for CAP-VSTNet. The two end-to-end editors receive the worst mean ranks, highlighting source preservation as essential to successful modernization. Neural Preset and SA-LUT rank below the source despite using the same modernized keyframes, showing that keyframe quality alone does not ensure efective video-level results.

Complementary quantitative results. Tab. 1 complements the human evaluation with quantitative measures of structural fidelity, temporal consistency, visual quality, intershot grade consistency, and computational eficiency. The end-to-end video editing methods substantially alter the source content, yielding markedly lower Edge-SSIM scores of 0.439–0.535, compared with 0.745–0.886 for the modular methods. They also obtain lower perceptual and aesthetic quality scores and generate only about 0.6 frames per second, indicating the dificulty of preserving source fidelity and eficiency through dense video regeneration. Within the modular family, our method remains competitive in structural fidelity and visual quality. It achieves the best SW-IGC of 5.90, indicating more consistent grading transformations across shots with similar source colors. Its training-free propagation runs at 816 FPS, on par with Neural Preset, about 10× faster than CAP-VSTNet, and over 35× faster than Video Color Grading and SA-LUT. Overall, our method combines competitive source fidelity with improved inter-shot grade consistency and high computational eficiency, enabling practical modernization of long-form videos.

Qualitative results. Fig. 3 shows qualitative comparisons among the evaluated methods. End-to-end video editing methods often reduce source fidelity. CogVideoX-5B alters facial identity and object appearance, while Wan2.1-VACE-1.3B removes fine details and produces blurry, desaturated outputs. Modular color-propagation methods better preserve structure but introduce distinct grading biases: Neural Preset lowers contrast, Video Color Grading oversaturates colors, SA-LUT produces dark outputs with a red cast, and CAP-VSTNet applies inconsistent grading across shots. In contrast, our method preserves source fidelity, reduces dominant color casts, and maintains natural skin tones with balanced contrast and saturation across shots.

## 4.3 Ablation Studies

Inter-shot consistency in keyframe modernization. We ablate design choices for improving inter-shot color consistency in keyframe modernization. Independent modernizes each keyframe in a separate model invocation, while Independent +post-hoc smoothing temporally smooths the estimated shot-specific color mappings before propagation. For joint keyframe modernization, we arrange up to four keyframes into a 2 × 2 grid and compare two grouping strategies. Packing (clustered) groups keyframes by source color similarity, whereas Packing (consecutive) groups temporally adjacent keyframes.

As shown in Tab. 2, independent modernization produces the highest SW-IGC, indicating large variation in inter-shot color mappings. Post-hoc smoothing reduces SW-IGC most efectively, but retains full inference cost and cannot coordinate keyframe appearances during generation. Fig. 4 visualizes the resulting inconsistencies and smoothing artifacts. The two packing strategies achieve comparable SW-IGC, indicating little benefit from explicit source color clustering for overall inter-shot consistency. Moreover, clustering requires comparing keyframes across the entire video and introduces a separate grouping stage and associated hyperparameters, whereas consecutive packing follows the temporal order directly. We therefore adopt Packing (consecutive) as the simpler and more eficient design, without sacrificing inter-shot consistency.

<table><tr><td>Method</td><td>Edge-SSIM ↑</td><td>CDC↓</td><td>SW-IGC↓</td><td></td><td>DeQA ↑ Q-AlignQ ↑ Q-AlignA ↑ FPS ↑</td><td></td><td></td></tr><tr><td>Original</td><td>0.966</td><td>0.56</td><td>0.19</td><td>2.73</td><td>3.16</td><td>2.32</td><td></td></tr><tr><td colspan="8">End-to-End Video Editing</td></tr><tr><td>CogVideoX-5B</td><td>0.439</td><td>0.77</td><td></td><td>2.32</td><td>2.55</td><td>1.92</td><td>一</td></tr><tr><td>Wan2.1-VACE-1.3B</td><td>0.535</td><td>1.37</td><td></td><td>1.92</td><td>1.96</td><td>1.63</td><td>一</td></tr><tr><td colspan="8">Modular Color Propagation</td></tr><tr><td>Neural Preset</td><td>0.886</td><td>0.56</td><td>7.72</td><td>2.73</td><td>3.11</td><td>2.34</td><td>820</td></tr><tr><td>SA-LUT</td><td>0.745</td><td>0.77</td><td>7.23</td><td>2.49</td><td>2.91</td><td>2.23</td><td>15</td></tr><tr><td>Video Color Grading</td><td>0.843</td><td>0.54</td><td>6.45</td><td>2.66</td><td>3.03</td><td>2.24</td><td>23</td></tr><tr><td>CAP-VSTNet</td><td>0.812</td><td>0.68</td><td>7.50</td><td>2.80</td><td>3.13</td><td>2.32</td><td>82</td></tr><tr><td>Ours</td><td>0.865</td><td>0.56</td><td>5.90</td><td>2.78</td><td>3.13</td><td>2.33</td><td>816</td></tr></table>

Table 1: Complementary quantitative comparison on 100 videos from the Condensed Movies Dataset. The unedited source video (Original) is shown in gray for reference and excluded from ranking. CDC is reported in units of $1 0 ^ { - 2 }$ . SW-IGC is omitted for end-to-end methods since reliable source–output correspondence is unavailable. FPS denotes the propagation throughput on an NVIDIA H200 GPU (see Appendix B.4).

Source  
CogVideoX  
Wan2.1-VACE  
Neural Preset  
SA-LUT  
VCG  
CAP-VSTNet  
Ours  
![](images/cfe53d66a1c045cc7ed1fc4ec640f367699b14b7e16561cb9b8509f682bc886e.jpg)  
Figure 3: Qualitative comparison of video modernization methods on representative frames.

<table><tr><td>Strategy</td><td>Invocations</td><td>SW-IGC↓</td></tr><tr><td>Independent</td><td>N</td><td>7.23</td></tr><tr><td>Independent + post-hoc smoothing</td><td>N</td><td>4.46†</td></tr><tr><td>Packing (clustered)</td><td>[N/4]</td><td>5.87</td></tr><tr><td>Packing (consecutive)</td><td>[N/4]</td><td>5.90</td></tr></table>

Table 2: Ablation of keyframe modernization strategies. N denotes the number of shots in each video, and the packing strategies process up to four keyframes per model invocation. <sup>†</sup>Post-hoc smoothing directly regularizes the shot-specific mappings evaluated by SW-IGC.

Operator design in color propagation. We examine the design of the shot-specific color propagation operator. Diagonal independently matches the mean and standard deviation of each channel, MKL matches the full covariance to model cross-channel correlations, and Histogram performs independent per-channel CDF matching. We further evaluate

<table><tr><td>Color mapping operator</td><td>SWD↓</td><td>Edge-SSIM ↑</td></tr><tr><td>Diagonal</td><td>0.821</td><td>0.884</td></tr><tr><td>Histogram</td><td>0.490</td><td>0.856</td></tr><tr><td>MKL</td><td>0.742</td><td>0.883</td></tr><tr><td>MKL + roll-off</td><td>0.748</td><td>0.885</td></tr></table>

Table 3: Ablation of shot-specific color mapping operators. Each operator is fitted and evaluated on the same source keyframe and its modernized counterpart.

MKL + roll-of, which augments the MKL mapping with a luma-based highlight roll-of before final clipping to the valid RGB range. For the quantitative evaluation, each mapping is applied to the source keyframe used for fitting. We compare the resulting keyframe with its modernized counterpart using the sliced-Wasserstein distance (SWD) in CIELAB for reference fidelity, and with the original source keyframe using Edge-SSIM for structural preservation.

![](images/714071a0918bc8ade32ab432b090b11ef395a1f3bb473f26b3ea0190f6dbfeeb.jpg)  
Figure 4: Qualitative comparison of inter-shot consistency strategies for keyframe modernization. Independent modernization exhibits inter-shot color inconsistency, while post-hoc smoothing produces unnatural color distortions. The two joint strategies produce more coherent keyframe appearances across shots.

Source  
Target  
Diagonal  
Histogram  
MKL  
![](images/11cf7dad05136818ef425fa1b3cc9c629ce75df1c78439aa49a9d4611c94c54d.jpg)  
MKL + roll-off  
Figure 5: Qualitative comparison of shot-specific color propagation operators. (A) Results obtained by applying each fitted mapping to its source keyframe. Top: Diagonal fails to reproduce the target grade, whereas Histogram introduces local color artifacts. Bottom: The per-channel mappings fail to reproduce the cross-channel target grade and produce strong color distortions, whereas MKL closely matches it. (B) Results after propagating the fitted mapping to a brighter frame: hard clipping saturates highlights, while roll-of preserves their tonal variation.

Tab.3 shows a trade-of between reference fidelity and structural preservation. Diagonal retains high Edge-SSIM but yields the largest SWD, indicating that independent channel-wise statistics have limited expressiveness for reproducing the target color distribution. In contrast, Histogram achieves the smallest SWD by independently matching the CDF of each source channel to that of the corresponding channel in the modernized keyframe, but yields the lowest Edge-SSIM. Fig. 5(A) qualitatively illustrates this trade-of. By modeling cross-channel correlations, MKL reduces SWD by 9.6% relative to Diagonal while maintaining nearly identical Edge-SSIM, providing a more favorable balance between the two criteria. MKL + roll-of performs comparably to MKL in the fitting-keyframe evaluation while reducing clipping-induced highlight saturation on propagated frames, as shown in Fig. 5(B). We therefore adopt it as our final propagation operator.

## 5 Conclusion

In this paper, we propose a scalable framework for long video color modernization. Our framework jointly modernizes packed keyframes from neighboring shots using the generative prior of a pretrained text-guided image editing model. It then propagates the modernized colors throughout each shot using a deterministic, shot-specific color mapping. Extensive experiments show that our results are perceived as having the most contemporary color appearance. Our framework also achieves the best inter-shot color consistency among the compared methods using lightweight, training-free color propagation, while remaining competitive on the other quantitative metrics. These results demonstrate a practical approach to modernizing long-form video archives without compromising their original content or coherence.

An, J.; Xiong, H.; Huan, J.; and Luo, J. 2020. Ultrafast Photorealistic Style Transfer via Neural Architecture Search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 10443–10450.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bain, M.; Nagrani, A.; Brown, A.; and Zisserman, A. 2020. Condensed Movies: Story Based Retrieval with Contextual Embeddings. In Proceedings of the Asian Conference on Computer Vision (ACCV).

Black Forest Labs; Batifol, S.; Blattmann, A.; Boesel, F.; Consul, S.; Diagne, C.; Dockhorn, T.; English, J.; English, Z.; Esser, P.; et al. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742.

Chan, K. C. K.; Zhou, S.; Xu, X.; and Loy, C. C. 2022. BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5972–5981.

Chen, D.; Liao, J.; Yuan, L.; Yu, N.; and Hua, G. 2017. Coherent Online Video Style Transfer. In Proceedings ofthe IEEE International Conference on Computer Vision (ICCV), 1105–1114.

Chen, S.; Li, X.; Zhang, X.; Wang, M.; Zhang, Y.; Han, J.; and Zhang, Y. 2024. Exemplar-Based Video Colorization with Long-Term Spatiotemporal Dependency. Knowledge-Based Systems, 284: 111240.

Chen, Y.; Yang, H.; Yang, Y.; Liu, Y.; Wang, W.; Wen, X.; and Xie, C. 2023. NLUT: Neural-Based 3D Lookup Tables for Video Photorealistic Style Transfer. arXiv preprint arXiv:2303.09170.

Chiu, T.-Y.; and Gurari, D. 2022. PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2868–2877.

Deng, Y.; Tang, F.; Dong, W.; Huang, H.; Ma, C.; and Xu, C. 2021. Arbitrary Video Style Transfer via Multi-Channel Correlation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 1210–1217.

Gong, Z.; Wu, Z.; Tao, Q.; Li, Q.; and Loy, C. C. 2025. SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 18294– 18303.

Gunawan, A.; Kim, S. Y.; Sim, H.; Lee, J.-H.; and Kim, M. 2023. Modernizing Old Photos Using Multiple References via Photorealistic Style Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 12460–12469.

Guo, Y.; Gong, J.; Cai, H.; Cheung, Y.-m.; and Su, W. 2026. LumiVideo: An Intelligent Agentic System for Video Color Grading. arXiv preprint arXiv:2604.02409.

Ho, M. M.; and Zhou, J. 2021. Deep Preset: Blending and Retouching Photos with Color Style Transfer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2113–2121.

Jiang, Z.; Han, Z.; Mao, C.; Zhang, J.; Pan, Y.; and Liu, Y. 2025. VACE: All-in-One Video Creation and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 17191–17202.

Kar, A.; and Su, G.-M. 2025. Temporal Consistent Semantic Video Color Transfer from Multiple References. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 6253–6261.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. MUSIQ: Multi-Scale Image Quality Transformer. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision (ICCV), 5148–5157.

Ke, Z.; Liu, Y.; Zhu, L.; Zhao, N.; and Lau, R. W. H. 2023. Neural Preset for Color Style Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14173–14182.

Larchenko, M.; Lobashev, A.; Guskov, D.; and Palyulin, V. V. 2025. Color Transfer with Modulated Flows. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 4464–4472.

Lei, C.; and Chen, Q. 2019. Fully Automatic Video Colorization with Self-Regularization and Diversity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3748–3756.

Li, M.; Wang, G.; Zhang, X.; Liao, Q.; and Xiao, C. 2025. D-LUT: Photorealistic Style Transfer via Difusion Process. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 9206–9214.

Li, X.; Liu, S.; Kautz, J.; and Yang, M.-H. 2019. Learning Linear Transformations for Fast Image and Video Style Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3809– 3817.

Li, Y.; Fang, C.; Yang, J.; Wang, Z.; Lu, X.; and Yang, M.- H. 2017. Universal Style Transfer via Feature Transforms. In Advances in Neural Information Processing Systems, volume 30.

Li, Y.; Liu, M.-Y.; Li, X.; Yang, M.-H.; and Kautz, J. 2018. A Closed-Form Solution to Photorealistic Image Stylization. In Proceedings of the European Conference on Computer Vision (ECCV), 468–483.

Lin, S.; and Simo-Serra, E. 2024. Restoring Degraded Old Films with Recursive Recurrent Transformer Networks. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 6704–6714.

Liu, Y.; Zhao, H.; Chan, K. C. K.; Wang, X.; Loy, C. C.; Qiao, Y.; and Dong, C. 2024. Temporally Consistent Video Colorization with Deep Feature Propagation and Self-Regularization Learning. Computational Visual Media, 10(2): 375–395.

Mao, Y.; Luo, H.; Zhong, Z.; Chen, P.; Zhang, Z.; and Wang, S. 2025. Making Old Film Great Again: Degradation-Aware

State Space Model for Old Film Restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 28039–28049.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.

Pitié, F.; and Kokaram, A. 2007. The Linear Monge– Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer. In Proceedings of the 4th European Conference on Visual Media Production, 1–9. IET.

Shin, S.; Shin, D.; Shin, J.; Jeon, H.-G.; and Lee, J.-Y. 2025. Video Color Grading via Look-Up Table Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 19141–19152.

Soria, X.; Pomboza-Junez, G.; and Sappa, A. D. 2022. LDC: Lightweight Dense CNN for Edge Detection. IEEE Access, 10: 68281–68290.

Souček, T.; and Lokoč, J. 2024. TransNet V2: An Efective Deep Network Architecture for Fast Shot Transition Detection. In Proceedings ofthe 32nd ACM International Conference on Multimedia, 11218–11221.

Wan, Z.; Zhang, B.; Chen, D.; and Liao, J. 2022. Bringing Old Films Back to Life. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17694–17703.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4): 600–612.

Wen, L.; Gao, C.; and Zou, C. 2023. CAP-VSTNet: Content Afinity Preserved Versatile Style Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18300–18309.

Wu, H.; Zhang, Z.; Zhang, W.; Chen, C.; Liao, L.; Li, C.; Gao, Y.; Wang, A.; Zhang, E.; Sun, W.; Yan, Q.; Min, X.; Zhai, G.; and Lin, W. 2024. Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels. In Proceedings of the 41st International Conference on Machine Learning (ICML), 54015–54029.

Wu, Z.; Zhu, Z.; Du, J.; and Bai, X. 2022. CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer. In Proceedings of the European Conference on Computer Vision (ECCV), 189–206.

Xia, X.; Zhang, M.; Xue, T.; Sun, Z.; Fang, H.; Kulis, B.; and Chen, J. 2020. Joint Bilateral Learning for Real-Time Universal Photorealistic Style Transfer. In Proceedings of the European Conference on Computer Vision (ECCV), 327– 342.

Yang, Y.; Pan, J.; Peng, Z.; Du, X.; Tao, Z.; and Tang, J. 2024. BiSTNet: Semantic Image Prior Guided Bidirectional Temporal Feature Fusion for Deep Exemplar-Based Video Colorization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8): 5612–5624.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2025. CogVideoX: Text-to-Video Difusion Models with an Expert Transformer. In International Conference on Learning Representations (ICLR), 83048–83077.

You, Z.; Cai, X.; Gu, J.; Xue, T.; and Dong, C. 2025. Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14483–14494.

Zhang, B.; He, M.; Liao, J.; Sander, P. V.; Yuan, L.; Bermak, A.; and Chen, D. 2019. Deep Exemplar-Based Video Colorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8052– 8061.

# From Legacy to Contemporary: Scalable Color Modernization for Long Videos

## Appendix

## A Overview

This appendix provides additional details and analyses, organized as follows:

• Implementation details of our framework in Appendix B.

• Detailed experimental settings in Appendix C.

• Additional quantitative results in Appendix D.

• Additional qualitative results in Appendix E.

• Limitations and future directions in Appendix F.

## B Implementation Details

This section provides implementation details for the proposed framework, following the stage structure introduced in the main paper. Tab. 4 summarizes the pipeline configurations used throughout all experiments.

## B.1 Keyframe Extraction

Shot detection. We adopt TransNet V2 (Souček and Lokoč 2024) to detect candidate shot boundaries. TransNet V2 is an eficient shot transition detection network that processes a sequence of neighboring frames and predicts a per-frame boundary probability, enabling fast shot segmentation of long videos. We threshold this probability at 0.1, a deliberately recall-oriented choice: a missed boundary would cause a single color mapping to span distinct shots, whereas false positives merely over-segment the video and are corrected in the subsequent merging stage. Consecutive boundary predictions are consolidated into a single cut at the probability peak, yielding a contiguous, non-overlapping partition of the entire video.

Shot merging. The recall-oriented threshold may oversegment a continuous shot in the presence of flashes or rapid motion. To correct these false splits, we compare the frames immediately before and after each candidate boundary using the cosine similarity of their pooled DINOv2-S embeddings (Oquab et al. 2024), and greedily merge adjacent segments whose similarity exceeds a merge threshold of 0.8, so that a shot split into multiple fragments is fully recombined. This semantic criterion helps distinguish transient within-shot appearance changes from true boundaries between distinct shots.

Keyframe selection. The selected keyframe serves as both the modernization input and the color reference for its shot, so it must satisfy three requirements at once: it should be sharply focused, representative of the shot’s content and color, and an actual film frame rather than a title card, poster, or other graphic that legacy clips frequently contain. These joint requirements are dificult to capture with a single hand-crafted criterion, so we cast selection as a holistic judgment by a vision–language model. From each shot, we uniformly sample 16 candidate frames and query Qwen3-VL (Bai et al.

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Keyframe Extraction</td><td></td></tr><tr><td>Shot detection detector</td><td>TransNet V2</td></tr><tr><td>boundary threshold</td><td>0.1</td></tr><tr><td></td><td></td></tr><tr><td>Shot merging</td><td>DINOv2-S</td></tr><tr><td>boundary encoder merge threshold</td><td>0.8</td></tr><tr><td></td><td></td></tr><tr><td>Keyframe selection selector</td><td>Qwen3-VL-8B-Instruct</td></tr><tr><td>candidates per shot</td><td>16</td></tr><tr><td>Image-Level Modernization</td><td></td></tr><tr><td>Keyframe packing &amp; unpacking</td><td></td></tr><tr><td>strategy</td><td>2×2, consecutive</td></tr><tr><td>padding</td><td>replicate</td></tr><tr><td></td><td></td></tr><tr><td>Joint keyframe modernization</td><td>FLUX.1-Kontext-dev</td></tr><tr><td>editor</td><td></td></tr><tr><td>guidance scale</td><td>2.5</td></tr><tr><td>inference steps</td><td>30</td></tr><tr><td>Color Propagation</td><td></td></tr><tr><td>Shot-specific color mapping</td><td></td></tr><tr><td>mapping operator</td><td>MKL (RGB)</td></tr><tr><td>Shot-level color propagation</td><td></td></tr><tr><td></td><td></td></tr><tr><td>roll-off knee k</td><td>0.75</td></tr></table>

Table 4: Pipeline configurations used for all experiments.

2025) once using the fixed instruction shown below; the returned index determines the shot’s keyframe. Here, {N} is replaced by the number of candidate frames at query time.

Keyframe Selection Prompt   
["These are {N} frames sampled in order   
from one continuous shot of an old movie,   
with indices ranging from 0 to {N-1}. Pick   
the ONE best frame to modernize and use as   
the color reference for the whole shot: (1)   
sharply in focus, NO motion blur; (2)   
clearly shows the main subject,   
representative of the shot; (3) a real film   
frame, NOT an end-card / movie poster /   
thumbnail grid / logo / text graphic.   
Answer with ONLY the index number   
(0-{N-1}), nothing else."]

## B.2 Image-Level Modernization

Keyframe packing & unpacking. We pack up to four temporally adjacent keyframes into a 2×2 image grid. When fewer than four keyframes are available, we fill the unused grid cells by repeating the available keyframes rather than using constant-valued padding. This avoids introducing large uniform regions that are atypical of natural images and may interfere with image editing. The packed grid is modernized in a single model invocation and subsequently unpacked into individual modernized keyframes, retaining only the outputs corresponding to the original keyframes.

Joint keyframe modernization. We use FLUX.1- Kontext (Black Forest Labs et al. 2025), a 12B-parameter flow-matching model for in-context image generation and editing, as the keyframe editor. Each packed grid is resized to the nearest supported resolution and edited in a single model invocation using the fixed instruction shown below, with a guidance scale of 2.5 and 30 sampling steps. Although the model is not specifically trained for grid-structured inputs, it generally applies coherent color and tonal adjustments across cells while preserving the content of each keyframe. The editor may slightly shift the grid boundaries, leading to minor cropping when the edited grid is unpacked. Such cropping has limited downstream impact because the propagation stage relies on global color statistics rather than pixelwise alignment.

Joint Keyframe Modernization Prompt   
["Modernize the color of each old film   
frame in this image grid to a natural,   
contemporary look. Apply clean, balanced   
colors and natural skin tones consistently   
across all grid cells, removing dated or   
faded color casts. Preserve the content,   
identity, composition, framing, spatial   
layout, and original lighting and time of   
day of every frame. Keep the grid structure   
and cell boundaries unchanged. Do not add,   
remove, or hallucinate any content."]

## B.3 Color Propagation

Shot-specific color mapping. For each shot, we fit the afine mapping in Eq. (1) of the main paper using MKL (Pitié and Kokaram 2007) from the RGB distributions of the legacy and modernized keyframes. Means and covariances are computed over all pixels, with each covariance regularized as $\mathbf { \bar { \Sigma } } \mathbf { \Sigma } \mathbf { E }  \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { + } 1 0 ^ { - \mathbf { \bar { 6 } } } \mathbf { I } ,$ . Since MKL matches global color distributions rather than pixel correspondences, fitting does not require spatial alignment between the two keyframes. The detailed formulation and closed-form solution are provided in the original work (Pitié and Kokaram 2007).

Shot-level color propagation. The fitted afine mapping is applied uniformly to all frames in each shot. The mapping may produce RGB values above the valid range, for which direct clipping can flatten highlight detail and introduce hue shifts through uneven channel saturation. We therefore apply a luma-based highlight roll-of to each mapped pixel $\mathbf { \dot { x } } ^ { \prime } = ( R ^ { \prime } , G ^ { \prime } , B ^ { \prime } ) ^ { \top }$ . Its Rec. 709 luma is computed from the gamma-encoded RGB values as

$$
Y ^ { \prime } = 0 . 2 1 2 6 R ^ { \prime } + 0 . 7 1 5 2 G ^ { \prime } + 0 . 0 7 2 2 B ^ { \prime } .\tag{5}
$$

Luma values above a fixed knee $k = 0 . 7 5$ are smoothly compressed using

$$
\rho ( Y ^ { \prime } ) = \left\{ { { Y ^ { \prime } } , \atop { k + ( 1 - k ) \operatorname { t a n h } \left( { \frac { Y ^ { \prime } - k } { 1 - k } } \right) , } } \right. \ { { Y ^ { \prime } } \leq k } ,\tag{6}
$$

<table><tr><td>Stage</td><td>Device</td><td>Cost</td></tr><tr><td>Shot detection</td><td>CPU</td><td>~24 s / min of video</td></tr><tr><td>Shot merging</td><td>GPU</td><td>~0.9 s / shot</td></tr><tr><td>Keyframe selection</td><td>GPU</td><td>~13 s / shot</td></tr><tr><td>Image-level modernization</td><td>GPU</td><td>~14 s / grid</td></tr><tr><td>Color propagation</td><td>CPU</td><td>~80 FPS</td></tr></table>

Table 5: Per-stage processing cost, pooled over ten videos from our dataset.

For $Y ^ { \prime } > k ,$ , the RGB channels are uniformly rescaled by $\rho ( Y ^ { \prime } ) / Y ^ { \prime }$ , followed by clipping to the valid range. This shared scaling smoothly compresses highlight overshoots while preserving the relative channel ratios.

## B.4 Runtime

We measure per-stage cost on ten videos from our dataset stratified by shot count, with one NVIDIA H200 GPU, and report each stage normalized by its natural cost unit in Tab. 5; one-time model loading is excluded, as it amortizes over batch processing.

On this sample, end-to-end processing averages 10.9 minutes per video, dominated by the per-shot VLM queries. Color propagation runs faster than real time even on the CPU, measured as wall-clock throughput including video I/O.

Throughput protocol. Tab. 1 reports the propagation throughput alone, excluding the shared shot detection, keyframe selection, and keyframe modernization stages. All propagation methods are timed on the same GPU with a common protocol on 40 length-stratified shots (10,334 frames), with models preloaded, host–device transfers included, and video I/O excluded. All methods use a common batch of four frames, chosen so that SA-LUT fits in GPU memory, except Video Color Grading, which processes each shot as a whole. Ours and Neural Preset are within measurement uncertainty, with overlapping 95% bootstrap intervals over shots. The end-to-end editors, timed on a representative clip, generate about 0.6 frames per second.

## C Experiment Settings

## C.1 Dataset Curation

We curate a subset of the Condensed Movies Dataset (Bain et al. 2020) through a two-stage filtering process. First, we retain videos produced between the 1960s and the 2000s to capture diverse color characteristics across eras. Second, since our task focuses on modernizing existing color rather than colorizing black-and-white footage, we exclude videos consisting predominantly of black-and-white content. From the remaining videos, we randomly sample 20 per decade, resulting in an evaluation set of 100 videos.

## C.2 Baseline Details

End-to-end video editing. For Wan2.1-VACE-1.3B (Jiang et al. 2025) and CogVideoX-5B (Yang et al. 2025), we use the default inference configurations. We split each evaluation video into clips that match the frame length recommended for each model and concatenate the edited clips to reconstruct the full video. We use the same prompt for both video models:

<table><tr><td>Method</td><td>Mean Rank↓</td><td>Rank-1 ↑</td><td>Top-3 ↑</td></tr><tr><td>Ours</td><td>2.313</td><td>37.9%</td><td>81.7%</td></tr><tr><td>CAP-VSTNet</td><td>3.015</td><td>17.6%</td><td>66.3%</td></tr><tr><td>Video Color Grading</td><td>3.559</td><td>12.9%</td><td>49.0%</td></tr><tr><td>Original</td><td>3.619</td><td>13.3%</td><td>45.1%</td></tr><tr><td>Neural Preset</td><td>4.028</td><td>11.0%</td><td>38.6%</td></tr><tr><td>SA-LUT</td><td>5.461</td><td>5.8%</td><td>12.8%</td></tr><tr><td>CogVideoX-5B</td><td>6.859</td><td>0.4%</td><td>4.8%</td></tr><tr><td>Wan2.1-VACE-1.3B</td><td>7.146</td><td>0.9%</td><td>1.6%</td></tr></table>

Table 6: Human evaluation of modern color preference over 975 complete rankings (65 participants × 15 videos). Mean Rank uses all placements from 1 to 8; Rank-1 and Top-3 denote the percentage of rankings in the corresponding range.

Modernization Prompt for Video Models   
["Modernize the color of this old film to a   
natural, contemporary look: clean,   
balanced, natural colors and skin tones,   
without dated or faded color casts. Keep   
the content, identity, framing, and the   
original lighting and time of day   
unchanged; do not add, remove, or   
hallucinate anything."]   
Negative Prompt for Video Models   
["overexposed, oversaturated, blurry, low   
quality, jpeg artifacts, warped face,   
changed identity, changed pose, changed   
clothing, changed framing, hallucinated   
objects, extra limbs, deformed hands,   
subtitles, watermark, text"]

Modular color propagation. For CAP-VSTNet (Wen, Gao, and Zou 2023), SA-LUT (Gong et al. 2025), and Video Color Grading (Shin et al. 2025), we use the default inference configurations. For Neural Preset (Ke et al. 2023), as neither pretrained weights nor oficial code are publicly available, we reproduce the method based on the paper. Since the LUTs used to train Neural Preset were not publicly released, exact reproduction is not possible. We therefore assemble a separate LUT collection for our implementation.

## C.3 User Study Details

We recruit 65 participants through the Prolific website for human evaluation. Fig. 8 presents the instructions provided to the participants. We prepare 15 video sets, each displayed as a grid of eight videos playing simultaneously, comprising the original video and the outputs of the evaluated methods. We randomize both the order of the video sets and the positions of all videos within each grid, including the original. After viewing each grid, as illustrated in Fig. 9, participants rank the videos according to two criteria: modern color appearance, judged on color alone, and temporal consistency of colors and content.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Videos</td><td>15 8</td></tr><tr><td colspan="2">Methods per video</td></tr><tr><td colspan="2">Unordered method pairs per video ((2))</td></tr><tr><td colspan="2">2AFC queries per pair (forward + reversed order)</td></tr><tr><td>Total VLM queries</td><td>2 840</td></tr><tr><td>Complete swapped-order comparisons</td><td>420</td></tr><tr><td>Order-consistent comparisons</td><td>262 / 420 (62.4%)</td></tr><tr><td>Order-inconsistent comparisons</td><td>158 / 420 (37.6%)</td></tr><tr><td>Candidate A selection rate</td><td>35.0%</td></tr><tr><td>Candidate B selection rate</td><td>65.0%</td></tr></table>

Table 7: Order-swap stability of the VLM-based twoalternative forced-choice (2AFC) color judge. For each unordered method pair, we query the judge twice: once in the original A/B order and once with the two candidates swapped. A comparison is consistent only when both queries select the same underlying method after mapping the A/B labels back to method identities.

## D Additional Quantitative Results

## D.1 VLMs Cannot Reliably Assess Modern Color

Our primary evaluation is a modern color preference study. Determining whether the dated colors of legacy footage have been suficiently transformed into a contemporary appearance is inherently subjective, suggesting two possible evaluation strategies: VLM-based evaluation and human evaluation. However, we find that VLM-based judgments are inconsistent and therefore unreliable. Based on Fig. 6 and Tab. 7, we exclude VLM-based evaluation from our evaluation and instead recruit human participants, highlighting that assessing modern color appearance requires direct human judgment rather than relying solely on existing automated evaluators.

We assess the reliability of VLM-based evaluation using two commonly adopted protocols: direct scoring and pairwise preference comparison under a two-alternative forcedchoice (2AFC) setting. We use Qwen3-VL (Bai et al. 2025) as the judge and evaluate it on 15 video sets sampled from the evaluation set. Each set contains the source video and the outputs of all compared methods, from which we uniformly sample 24 frames per video.

For direct scoring, we divide the 24 sampled frames for each method into six chunks of four frames and ask the VLM to assign a modern color score from 1 to 5 to each chunk. For video v, method m, and chunk c, let $s _ { v , m , c }$ be the VLM score. Each method receives six chunk scores, so its videolevel score is:

$$
s _ { v , m } = \frac { 1 } { 6 } \sum _ { c = 1 } ^ { 6 } s _ { v , m , c } .
$$

The average pairwise-diference matrix of these 15 videos is defined as follows:

$$
\bar { D } ( i , j ) = \frac { 1 } { 1 5 } \sum _ { v = 1 } ^ { 1 5 } D _ { v } ( i , j ) = \frac { 1 } { 1 5 } \sum _ { v = 1 } ^ { 1 5 } \left| s _ { v , m _ { i } } - s _ { v , m _ { j } } \right| .
$$

Across 15 videos, the eight method means occupy only the interval 2.2889–2.8889 on the 1–5 scale, and Fig. 6 shows a mean pairwise absolute diference of only 0.3544. The result shows that many visually diferent color appearances receive nearly equivalent scores, indicating that the VLM does not provide suficiently discriminative evidence for comparing color modernization quality.

![](images/de97c6bf650fc5716e3b8179fc3c56f6b4696064fe3529f58e5939e61dc74b16.jpg)  
Figure 6: Average pairwise absolute diferences between the VLM scores of the eight evaluated methods across 15 videos.

Tab. 7 shows results in the 2AFC setting. For each method pair, we evaluate temporally aligned frames in both A/B orders and consider the result consistent only if the same underlying method wins. Swapping the order changes the winner for 158 of 420 pairs (37.6%). Candidate B is selected 65% of the time, 30 percentage points more often than A, revealing a substantial preference for the second position. The VLM sometimes changes its preference when the positions of the two candidates are swapped, revealing substantial positional inconsistency.

## D.2 User Study Results

We conduct a user study to evaluate whether each method successfully transforms the colors of legacy footage into a modern style. As shown in Fig. 6 and Tab. 7, we find that VLM-based evaluation does not provide suficiently consistent judgments. Determining whether a method produces modern colors and improves upon the original footage is inherently subjective.

We collect a total of 975 rankings from 65 participants and evaluate the results using Mean Rank, Rank-1, and Top-3 rates. Tab. 6 shows that our method outperforms all baselines across all three measures, indicating that it most successfully transforms the dated colors of legacy footage into a contemporary appearance. Interestingly, Neural Preset, SA-LUT, CogVideoX-5B, and Wan2.1-VACE-1.3B perform worse than the unedited original video. This suggests that these methods either fail to produce a suficiently contemporary color appearance or introduce content instability that reduces user preference.

## E Additional Qualitative Results

## E.1 Comparison of Keyframe Selection Strategies

Selecting an appropriate keyframe is critical, as it serves both as the input to the image editor and as the color reference for the entire shot. We compare our vision–language modelbased selector with four alternatives based on heuristic, statistical, and learned criteria. Firstframe selects the first frame of each shot, while the color medoid selects the frame whose mean color is closest to the shot-level average. MUSIQ (Ke et al. 2021) selects the frame with the highest no-reference image-quality score, and the DINOv2 medoid (Oquab et al. 2024) selects the frame closest to the mean DINOv2 representation of the shot.

Fig. 7 illustrates representative limitations of these selection criteria, with failure cases outlined in red. First frame is sensitive to shot transitions and rapid motion, and may select blurred frames or views captured before the main subject appears. MUSIQ prioritizes perceptual quality but does not assess shot representativeness, occasionally favoring title cards or poster-like graphics over actual scene content. The DINOv2 medoid favors semantically representative frames but does not explicitly account for image quality, while the color medoid similarly captures color centrality without assessing visual quality or whether the frame contains informative scene content. In contrast, our vision–language model jointly considers visual quality, shot representativeness, and whether each candidate is a valid content frame.

## E.2 Comparison with Baselines

Fig. 10 and Fig. 11 present additional qualitative comparisons between our method and the evaluated baselines on frames sampled from the evaluation set, covering diverse content and visual conditions.

## F Limitations and Future Directions

Our framework has three main limitations. First, joint modernization is restricted to groups of up to four temporally adjacent keyframes. It therefore promotes local cross-shot consistency but does not explicitly coordinate the appearance of distant or recurring shots throughout a long video. Second, shot-level propagation relies on a single global afine color mapping. While this design eficiently preserves spatial structure and temporal stability, it cannot reproduce spatially varying edits or nonlinear local adjustments introduced by the image editor. Third, the method cannot reliably distinguish unintended color degradation from deliberate stylistic choices, such as muted palettes or scene-specific color casts, and may therefore alter the intended visual style.

Future work could explore hierarchical or memory-based coordination across keyframes to improve long-range coherence while preserving shot-specific flexibility. Spatially adaptive propagation may better capture local edits without sacrificing temporal consistency. Incorporating user input or reference images could also help preserve the intended visual style by specifying the desired degree and direction of modernization. Broader evaluation across genres, production formats, and viewer populations would further clarify how perceptions of contemporary color vary across viewing contexts.

VLM-based Modern Color Scoring Prompt  
System Prompt   
["You are an expert evaluator for a video color-modernization study.   
Your task is to evaluate ONLY the color treatment, not the content or image quality.   
Use the full 1-5 scale. Do not be overly harsh or overly generous.   
A score of 3 means acceptable or mixed color modernization, not failure.   
Do NOT evaluate:   
whether the subject, clothes, objects, props, architecture, or location look old or modern   
image resolution   
sharpness or blur   
noise   
compression artifacts   
object content   
scene composition   
acting, story, aesthetics, or cinematic framing   
non-color effects   
temporal consistency or motion continuity"]   
Prompt   
["You will receive sampled frames from one generated video.   
Evaluate whether the color treatment looks modernized, natural, and color-balanced.   
Focus only on color appearance.   
Evaluate ONLY:   
natural\_saturation: colors are neither artificially oversaturated nor unnecessarily desaturated.   
cast\_neutrality: no unintended yellow, green, turquoise, blue or magenta cast of the old video and has a neutral color   
tone.   
naturalness: semantically familiar colors such as daytime, skin, objects, and neutrals look plausible   
Overall Scoring:   
Use a 1-5 scale for overall\_color\_modernity.   
5 = color looks clean and acceptable for contemporary viewing   
4 = mostly acceptable, with minor old-color artifacts   
3 = mixed; some old-color artifacts are clearly visible   
2 = mostly outdated; old-color artifacts dominate   
1 = severely outdated or color-biased   
A clean but ordinary color treatment can receive 4 or 5.   
Use low acceptability scores only when outdated colors are clearly visible.   
Score overall\_color\_modernity from 1 to 5.   
Example output:   
"frame\_ids": [the exact frame\_ids\_in\_order list],   
"overall\_color\_modernity": 3,   
"reason": "Although the color treatment appears somewhat modern, it is not fully natural."   
}"]

## VLM-based Pairwise Preference Comparison Prompt

```jsonl
System Prompt
["You are an expert evaluator for a video color-modernization study.
You will be given two matching frames from two candidate videos.
Your task is to evaluate ONLY the color treatment, not the content or image quality.
Use the full distinction between candidates. You must choose exactly one winner.
Do NOT evaluate:
whether the subject, clothes, objects, props, architecture, or location look old or modern
image resolution
sharpness or blur
noise
compression artifacts
object content
scene composition
acting, story, aesthetics, or cinematic framing
non-color effects
temporal consistency or motion continuity"]
Prompt
["Compare Candidate A and Candidate B for color modernization.
Focus only on color appearance.
Evaluate ONLY:
natural_saturation: colors are neither artificially oversaturated nor unnecessarily desaturated.
cast_neutrality: no unintended yellow, green, turquoise, blue or magenta cast of the old video and has a neutral color
tone.
naturalness: semantically familiar colors such as daytime, skin, objects, and neutrals look plausible
Choose exactly one winner: A or B. Do not return TIE.
If the candidates are very close, choose the one with slightly more plausible, neutral, and modern color treatment.
Example output:
"winner": "A",
"reason": "Candidate A has more neutral and natural color balance, with fewer outdated casts than Candidate B."
}"]
```

![](images/cc8b49794f657210ceb970466ceda252baccfe7aaa08fbd9baf60e54bf2073a2.jpg)  
Figure 7: Comparison of keyframe selection methods. Each row shows the keyframes selected by diferent methods for the same shot; failure cases are outlined in red.

![](images/bd19cbd118c2ef82b15c1f8cf6b11d9da3163e51ef00d435656a1c50cae931cf.jpg)

Figure 8: Instructions presented to participants in the user study.  
![](images/37970c3981964d751470d79067ef91c82600a094e781c8b0c62565d708bd7a84.jpg)  
Figure 9: Interface used by participants to rank the eight video versions.

Source  
CogVideoX  
Wan2.1-VACE  
Neural Preset  
SA-LUT  
VCG  
CAP-VSTNet  
Ours  
![](images/f1f2216292821b459fb093b3572ac07cfb9ef3302050c773e94b5261cee406ac.jpg)  
Figure 10: Additional qualitative comparisons between our method and the evaluated baselines on diverse frames from the evaluation set (Part 1).

Source  
CogVideoX  
Wan2.1-VACE  
Neural Preset  
SA-LUT  
VCG  
CAP-VSTNet  
Ours  
![](images/d99e6705321b4ad0c929597360fbae63ff4077844b5babaede44f7fd5e0b666d.jpg)  
Figure 11: Additional qualitative comparisons between our method and the evaluated baselines on diverse frames from the evaluation set (Part 2).