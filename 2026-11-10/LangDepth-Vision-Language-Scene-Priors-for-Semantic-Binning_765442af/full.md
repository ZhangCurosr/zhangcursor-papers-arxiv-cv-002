# LangDepth: Vision-Language Scene Priors for Semantic Binning and Cross-Domain Monocular Depth Estimation

Reyhaneh Ahani Manghotay<sup>1</sup> and Jie Liang<sup>2⋆</sup>

<sup>1</sup> School of Engineering Science, Simon Fraser University, Burnaby, BC, Canada raa112@sfu.ca

2 School of Information Science and Technology, Eastern Institute of Technology, Ningbo, China

Jliang@eitech.edu.cn

Abstract. Monocular depth estimation has advanced through largescale pretraining and strong visual backbones, yet metric depth remains sensitive to scene-level scale. Indoor rooms and outdoor driving scenes occupy very diferent depth ranges, but most depth decoders infer this global structure implicitly from visual features. We ask whether scenelevel semantic structure can instead act as an upstream prior that shapes the metric output space before dense prediction.

We present LangDepth (SemanticBins), a monocular depth architecture that couples strong visual features with a scene-conditioned output space. A frozen CLIP encoder and a learnable bank of scene- and geometry-oriented concept keys convert image features into a compact scene descriptor, without requiring per-image text or an external captioner. A Scene Prior Module (SPM) maps this descriptor to a sceneconditioned metric depth interval, providing a global range anchor before dense decoding. In parallel, a DINOv2-L backbone with a lightweight FPN extracts dense visual features, and a Per-Pixel Assignment Network (PPAN) assigns pixels to the scene-prescribed bin structure. A confidence gate fuses the dense visual estimate with the SPM prior.

We train LangDepth jointly on NYU Depth V2 and KITTI under a modest supervised depth budget, without web-scale depth supervision or pseudo-labelled depth data. A single model achieves $\delta _ { 1 } \mathrm { = } 0 . 9 6 0$ on NYU, 0.960 on KITTI, 0.887 on DIODE indoor, 0.946 on iBims-1, and 0.861 on SUN RGB-D. Beyond aggregate accuracy, the SPM predicts compact indoor ranges and much wider outdoor KITTI ranges, the gate becomes SPM-dominant on outdoor scenes, and perturbing the routed scene context causes large drops on KITTI and iBims-1. These results support scene-conditioned output spaces as a practical and interpretable mechanism for cross-domain monocular depth estimation.

Keywords: Monocular depth estimation · Vision-language models · CLIP · Semantic bins · Scene-conditioned depth · DINOv2 · Cross-domain transfer · Scene Prior Module

## 1 Introduction

Predicting metric depth from a single RGB image remains a central challenge in computer vision, with applications in robotics, augmented reality, autonomous navigation, and 3D scene understanding. Recent progress has been driven by large-scale pretraining [22], strong visual backbones [13], and carefully designed output spaces [2,12,15]. These models achieve impressive accuracy, but the metric scale of a scene is still strongly shaped by global scene context. A compact indoor room, a cluttered ofice, and an outdoor driving scene occupy very diferent depth ranges. Even when the output space is adaptive, most depth decoders infer this global structure from visual features alone. The decoder must therefore recover both local geometry and global scale from pixels, which becomes especially challenging when evaluation spans domains whose depth ranges and scene layouts difer substantially from those seen during training.

Scene semantics as an upstream structural prior. Vision-language models such as CLIP [17] encode scene-level information that is directly relevant to metric depth. Recognising that an image depicts a corridor, bedroom, kitchen, street, or long-range driving scene constrains the plausible metric scale before dense prediction begins. However, many language-augmented depth models use language as an auxiliary or late-stage conditioning signal, appended after much of the visual computation has already occurred [9,25–27]. Such conditioning can provide useful cues, but it does not necessarily define the metric output space over which dense visual prediction is performed. Caption-conditioned methods also introduce a dependency on per-image text or an external captioner at inference time. The gap we address is therefore one of scene-conditioned output structure: using semantic scene information, derived directly from the image through a reusable concept bank, without per-image text, to shape the depth range and bin layout upstream of dense decoding, so that the visual branch predicts within a scene-adapted metric space rather than a fixed or purely visual one.

Our approach: scene semantics set the range, vision assigns pixels. LangDepth (SemanticBins) addresses this gap by coupling a CLIP-derived scene prior with a strong DINOv2 visual decoder. A frozen CLIP image encoder maps each image through a learnable bank of scene- and geometry-oriented concept keys, producing a compact scene descriptor z without requiring perimage text. A Scene Prior Module (SPM), an MLP that consumes only $\mathbf { z } ,$ predicts a scene-conditioned metric interval $( d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } )$ and defines ordered bin centres $\{ c _ { k } \} _ { k = 1 } ^ { K }$ within that interval. A DINOv2-L [13] backbone with a multiscale FPN then extracts dense visual features, and a Per-Pixel Assignment Network (PPAN) assigns each pixel to the scene-prescribed bin structure. The final depth blends the dense visual estimate with the global SPM prior through a learned confidence gate (Fig. 1). Crucially, the SPM does not consume the dense pixel-level features used by the DINOv2–PPAN pathway; the predicted depth interval is determined by the scene descriptor, so the semantic pathway defines the output space rather than merely conditioning it.

Our analyses confirm that this structure is not merely cosmetic: the SPM predicts compact indoor ranges (≈ 8–10 m) and much wider KITTI ranges, the gate becomes SPM-dominant on outdoor scenes, and perturbing the routed descriptor causes large accuracy drops on KITTI and iBims-1. This design leads to three contributions. First, we introduce LangDepth (SemanticBins), a sceneconditioned monocular depth architecture in which a CLIP-derived SPM defines a metric depth range before dense visual decoding. Unlike adaptive visual discretisation methods, the predicted output range is derived from a routed scene descriptor rather than from the dense visual feature map itself. Second, we propose a concept-routing mechanism that converts frozen CLIP image features into a compact scene descriptor using learnable scene and geometry concept keys, providing a more efective scene representation for cross-domain depth than a single global CLIP embedding, as confirmed by component ablations on KITTI and DIODE transfer. Third, we combine this semantic range prior with a DINOv2-based PPAN and confidence-gated fusion, enabling domainadaptive balancing between local visual evidence and global scene-scale information. Trained jointly on NYU Depth V2 and KITTI under a modest supervised budget, LangDepth achieves strong performance across both training domains and transfers to DIODE indoor, iBims-1, and SUN RGB-D. Together, the accuracy results, range analysis, gate statistics, and scene-context perturbations demonstrate that scene-conditioned output spaces are a practical and interpretable mechanism for cross-domain monocular metric depth estimation.

## 2 Related Work

## 2.1 Monocular depth estimation and adaptive output spaces

Single-image depth has been studied as direct metric regression [5, 6], as multitask learning with surface normals and semantic segmentation, and more recently as a binned prediction problem in which the network discretises the depth axis and predicts soft per-pixel assignments [2, 3, 12]. Transformer backbones and CRF-style refinement improve local structure, boundaries, and long-range context [1, 11, 18, 24], while internal-discretisation methods learn adaptive output representations for depth prediction [15].

LangDepth follows the adaptive-output-space direction, but difers in how the output space is constructed. AdaBins [2] and BinsFormer [12] derive adaptive bin structures from visual feature maps, while iDisc [15] learns an internal discretisation from multi-scale visual representations. In these methods, the global metric range and the dense depth prediction are both inferred from the visual pathway. In contrast, LangDepth predicts the metric interval from a CLIP-routed scene descriptor before dense visual assignment. The SPM does not consume the dense DINOv2–PPAN feature map; it provides a scene-conditioned range anchor, after which the visual decoder assigns pixels within the prescribed bin structure. This separation makes the output space explicitly scene-conditioned rather than purely visual, enabling domain-adaptive range behaviour that can be inspected and verified empirically.

## 2.2 Language and depth

DepthCLIP [26] and CLIP2Depth [9] were among the first to exploit CLIP’s geometric signal for monocular depth. VPD [27] distils text-to-image difusion representations for dense prediction. WorDepth [25] is the most directly related recent language-conditioned method: it encodes a scene caption through CLIP’s text encoder and uses an image-conditional sampler to produce a depth map. These works show that language and vision-language representations contain useful geometric and scene-level information. However, language is often used as an auxiliary conditioning signal around an otherwise visual prediction pipeline, and caption-conditioned methods introduce a dependency on per-image text or an external captioner at inference time.

In contrast, LangDepth uses language-grounded scene semantics to condition the metric output space itself. Rather than requiring a generated caption, it routes frozen CLIP image features through a reusable bank of scene- and geometry-oriented concept keys, producing a compact scene descriptor. This descriptor is passed to the SPM before dense decoding, where it defines a sceneconditioned depth interval and global range anchor. The visual decoder then assigns pixels within this scene-prescribed bin structure. Unlike WorDepth, which builds depth prediction around visual features and a sampled language prior, the SPM in LangDepth defines the metric interval explicitly before dense visual decoding. Thus, semantic information is placed upstream of dense prediction, where it can shape the metric scale directly.

## 2.3 Web-scale monocular pipelines

ZoeDepth [4], Metric3D [23], Depth Anything V2 [22], and UniDepthV2 [16] achieve strong performance through large-scale pretraining, pseudo-labelling, or broad multi-dataset training. These systems are important reference points, but operate under substantially larger depth-data and training budgets than our supervised NYU+KITTI setting: ZoeDepth pre-trains on 12 relative-depth datasets, Metric3D uses over 8M images from 11 datasets, Depth Anything V2 uses 595K synthetic images plus 62M pseudo-labelled real images, and UniDepthV2 uses 24 datasets totalling 16M images [4, 16, 22, 23]. We therefore treat them as contextual comparisons. Our focus is distinct: rather than relying on web-scale depth supervision, pseudo-labelled depth data, or broad multi-dataset depth training to implicitly learn scene-scale disambiguation, LangDepth makes the metric range prediction explicit through a scene-conditioned SPM, yielding an inspectable scene-scale mechanism through its range predictions, gate behaviour, and context sensitivity.

## 2.4 Visual backbones and feature conditioning

DINOv2 [13] has emerged as a strong self-supervised visual backbone for dense prediction, providing transferable features that are useful beyond the datasets on which a downstream model is trained. Conditioning mechanisms such as FiLM [14] ofer a lightweight way to modulate visual features using auxiliary context. In LangDepth, we use these ideas in a separated design: DINOv2 provides dense visual features for per-pixel assignment, while the CLIP-derived scene pathway defines a scene-conditioned metric output space before dense decoding.

## 3 Method

## 3.1 Overview

Given an RGB image $\mathbf { I } \in \mathbb { R } ^ { 3 \times H \times W }$ , LangDepth predicts a metric depth map $\hat { \mathbf { D } } \in \mathbb { R } ^ { H \times W }$ through two complementary branches that meet at a confidencegated fusion step (Fig. 1). The semantic branch uses CLIP, concept routing, and the Scene Prior Module (SPM) to infer a scene-conditioned metric range and global depth prior. The vision branch uses DINOv2–FPN features and a Per-Pixel Assignment Network (PPAN) to assign each pixel to bins defined inside that scene-conditioned range. Thus, the semantic pathway shapes the output space before dense decoding, while the visual pathway resolves local geometry, object boundaries, and per-pixel depth variation.

This design gives scene semantics a structural role in metric depth prediction. LangDepth uses a routed CLIP descriptor to define a scene-conditioned range anchor that is shared by the dense prediction branch, rather than treating semantic information only as an auxiliary feature. The final confidence gate then determines, at each pixel, how strongly to rely on the dense visual estimate versus the global scene prior. This is especially useful for mixed indoor–outdoor depth prediction, where the valid metric range can change substantially across scene types.

## 3.2 Scene descriptor from CLIP

We run a frozen CLIP ViT-B/32 [17] on a $2 2 4 ^ { 2 }$ resize of the input. Let v denote the CLS embedding and P the $7 \times 7$ patch grid. A spatial query is formed by concatenating the global mean and four quadrant means of P and projecting the result into a learned routing space. This query attends over a bank of learnable concept keys $\mathbf { k } _ { j } \in \mathbb { R } ^ { 2 5 6 }$ with associated concept values. The concept bank contains 65 prompts in two groups: 20 scene-type prompts covering indoor layouts and outdoor/driving scenes, and 45 geometry/material prompts covering depth-range anchors, spatial structure, boundary types, lighting conditions, and outdoor geometry. These prompts were manually curated to cover scene-scale and geometric cues relevant to metric depth across indoor and outdoor domains. The concept keys are randomly initialised in a learned 256-dimensional routing space, while the concept values are initialised from frozen CLIP text embeddings of the corresponding prompts, projected to 512 dimensions, and subsequently made learnable. At inference, no per-image text encoding, captions, or external captioner is used; the concept bank is fixed after training and scene semantics are derived from the input image through the frozen CLIP image encoder. A linear projection fuses the retrieved context with v into the scene descriptor $\mathbf { z } \in \mathbb { R } ^ { 5 1 2 }$

![](images/b927fce86aca66e46ee72eb26bb96109efd8f49ed7c4510a6e033dd44997521c.jpg)  
Fig. 1: LangDepth (SemanticBins) pipeline. The input image is encoded by a frozen CLIP ViT-B/32, whose image features are routed through learnable scene and geometry concept keys to produce a compact scene descriptor z. The Scene Prior Module (SPM) uses only z to predict a scene-conditioned range $( d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } )$ , ordered bin centres $\{ c _ { k } \} _ { k = 1 } ^ { K } .$ , and a global prior over those centres. In parallel, DINOv2-L with a lightweight FPN extracts dense visual features. PPAN conditions these features on z and predicts per-pixel bin weights $\{ W _ { k } \} _ { k = 1 } ^ { K }$ and a gate map g. Depth fusion forms the visual estimate $\begin{array} { r } { d _ { \mathrm { v i s } } = \sum _ { k } W _ { k } c _ { k } } \end{array}$ and the SPM prior $d _ { \mathrm { S P M } } = \sum _ { k }$ π<sub>k</sub>c<sub>k</sub>, then outputs $\hat { d } = \mathbf { g } \odot d _ { \mathrm { v i s } } + ( 1 - \mathbf { g } ) \odot d _ { \mathrm { S P M } }$

Concept routing. The concept bank is deliberately separated into scene-type prompts and geometry/material prompts. Scene prompts encourage the descriptor to capture global layout cues such as corridor-like, bedroom-like, kitchenlike, or outdoor driving structure, while geometry prompts capture properties such as planar surfaces, clutter, shadows, and foreground–background transitions. Rather than using CLIP image-text similarity directly, we project both the spatial query and concept keys into a learned routing space. This lets the model adapt concept selection to the depth objective while still benefiting from CLIP text initialisation. We regularise the routing with entropy and key-repulsion terms to encourage selective but diverse concept usage.

## 3.3 Multi-layer CLIP routing

The full model uses multi-layer CLIP routing. Shallow CLIP ViT layers tend to emphasise texture and local appearance, while deeper layers encode higherlevel scene semantics. We therefore extract patch tokens from one early and one late intermediate block, apply the same five-region pooling, and concatenate the resulting vectors with the default CLIP statistics before projection. The CLIP backbone remains fully frozen; only the projection and routing weights are trained. This provides the SPM and PPAN with a descriptor that combines local geometric cues and global scene semantics.

## 3.4 Scene Prior Module (SPM)

The SPM is a lightweight MLP that consumes only the scene descriptor z. It predicts a scene-conditioned metric interval $( d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } )$ within the global valid depth bounds, using a parameterisation that keeps the interval ordered and clamps it to the maximum valid depth. The bin centres $\{ c _ { k } \} _ { k = 1 } ^ { K }$ are then uniformly placed inside $[ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$

The SPM also outputs logits $\ell \in \mathbb { R } ^ { K }$ over these centres, normalised as $\pi =$ softmax(ℓ). The SPM global prior is then:

$$
d _ { \mathrm { S P M } } = \sum _ { k = 1 } ^ { K } \pi _ { k } c _ { k } ,\tag{1}
$$

which is broadcast spatially after PPAN’s output resolution is known.

The key role of the SPM is to provide a scene-conditioned range anchor. By predicting $( d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } )$ from z before dense decoding, the model can adapt its metric output space to the scene type: compact indoor scenes can use a short range, while outdoor driving scenes can use a much wider range. The prior π provides a secondary scene-level weighting over the prescribed bin centres. Empirically, the learned prior over bin centres is highly concentrated, so we interpret the SPM primarily as a scene-conditioned range anchor. This motivates the range-only ablation, which keeps the predicted interval but replaces the learned prior with uniform bin weights.

## 3.5 DINOv2 encoder and FPN

Images are resized to $5 1 8 ^ { 2 }$ for DINOv2-L/14 [13]. We tap blocks {5, 11, 17, 23}, each yielding a feature map with channel width 1024. A lightweight FPN merges these features into a multi-scale representation with width 256. The last four ViT blocks and the final layer norm are trainable; earlier blocks remain frozen. This keeps the visual backbone strong while limiting the number of trainable parameters.

## 3.6 PPAN and gated depth

PPAN applies FiLM modulation $( \gamma , \beta )$ to the FPN features, conditioned on z, followed by a small convolutional head that outputs per-pixel softmax weights $\mathbf { W \in }$ R<sup>K×h×w</sup> over the SPM bin centres and a per-pixel gate map $\mathbf { g } \in [ 0 , 1 ] ^ { 1 \times h \times w }$ The visual depth estimate is:

$$
\hat { d } _ { \mathrm { v i s } } = \sum _ { k = 1 } ^ { K } W _ { k } c _ { k } .\tag{2}
$$

The final fused map is:

$$
\hat { d } = \mathbf { g } \odot \hat { d } _ { \mathrm { v i s } } + ( 1 - \mathbf { g } ) \odot d _ { \mathrm { S P M } } ,\tag{3}
$$

upsampled bilinearly to $( H , W )$ and clamped to the valid depth range.

The gate value has a direct interpretation: $\mathbf g \to 1$ favours the dense visual estimate, while $\mathbf { g }  \mathbf { \zeta } 0$ favours the SPM prior. The gate is therefore not an uncertainty estimator in the strict probabilistic sense; it is an adaptive fusion mechanism that decides where local visual evidence should dominate and where the global scene range prior should contribute more strongly.

Ablation of gate fusion. The $w / o$ gate fusion ablation, denoted as visual\_only in the implementation, replaces the final confidence-gated blend with the dense PPAN prediction $\hat { d } _ { \mathrm { v i s } }$ . This variant keeps the same scene-conditioned bin construction and routed scene descriptor as the full model, but removes the final adaptive mixing between the PPAN estimate and the SPM prior. It therefore isolates the contribution of the fusion stage while leaving the upstream sceneconditioned output space unchanged.

## 3.7 Flip-consistent confidence

We encourage the gate map g to reflect prediction stability under simple input perturbations. During training, we process both an image x and its horizontally flipped version x˜ in a single forward pass. After reversing the flip, we obtain aligned depth predictions $\hat { d } _ { a }$ and $\hat { d } _ { b }$ for the original and flipped views. A perpixel agreement target is defined as:

$$
\mathbf { a } = \exp \left( - \frac { \left| \log \hat { d } _ { a } - \log \hat { d } _ { b } \right| } { \tau } \right) ,\tag{4}
$$

where $\tau { = } 0 . 1 5$ controls sensitivity to prediction disagreement. Pixels whose predictions remain stable under flipping receive agreement values close to 1, while unstable regions receive lower values.

Let ${ \bf g } _ { a }$ and $\mathbf { g } _ { b }$ denote the gate maps predicted for the original and flipped views after spatial alignment. Both maps are supervised toward the detached agreement target a through:

$$
\mathcal { L } _ { \mathrm { c c } } = \frac { 1 } { 2 N } \sum _ { \mathbf { p } \in \mathcal { V } } \Big [ \big ( \mathbf { g } _ { a } ( \mathbf { p } ) - \mathbf { a } ( \mathbf { p } ) \big ) ^ { 2 } + \big ( \mathbf { g } _ { b } ( \mathbf { p } ) - \mathbf { a } ( \mathbf { p } ) \big ) ^ { 2 } \Big ] ,\tag{5}
$$

where V is the set of valid annotated pixels and $N { = } | \nu |$ . This loss is weighted by $\lambda _ { \mathrm { c c } } { = } 0 . 1$ in the total objective. The mechanism encourages the learned fusion to be consistent with prediction stability, while still allowing the gate to adapt by scene and region.

## 3.8 Training objectives

We train with a joint objective composed of standard depth supervision and regularisation terms for the scene-conditioned pathway:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { s p m } } \mathcal { L } _ { \mathrm { s p m } } + \lambda _ { \mathrm { c o n c e p t } } \mathcal { L } _ { \mathrm { c o n c e p t } } + \lambda _ { \mathrm { c c } } \mathcal { L } _ { \mathrm { c c } }\tag{6}
$$

We use $\lambda _ { \mathrm { c c } } { = } 0 . 1$ for flip-consistent confidence. The depth term ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ combines an Eigen-style scale-invariant log loss [6], residual gradient and edge-weighted gradient terms, an ordinal ranking loss over sampled pixel pairs, and an auxiliary SiLog head on PPAN features. $\mathcal { L } _ { \mathrm { s p m } }$ encourages the predicted range $( d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } )$ to cover the valid per-image depth extent while avoiding uninformatively wide intervals. $\mathcal { L } _ { \mathrm { c o n c e p t } }$ keeps the concept routing selective and discourages concentration on a small number of generic concepts. ${ \mathcal { L } } _ { \mathrm { c c } }$ is the flip-consistent confidence loss defined in Eq. (5).

We use $K { = } 6 4$ bins and train the final model jointly on NYU Depth V2 and KITTI. NYU uses a 10 m depth cap, while KITTI uses an 80 m cap, allowing the SPM to learn both compact indoor ranges and long-range outdoor scale.

## 4 Experiments

## 4.1 Setup

Training data. We train the final LangDepth model jointly on NYU Depth V2 [19] and KITTI Eigen [7]. NYU provides dense indoor supervision with a 10 m depth cap, while KITTI provides outdoor driving scenes with an 80 m cap. This mixed indoor, outdoor setting lets us evaluate whether the proposed scene-conditioned output space can support both compact indoor ranges and long-range outdoor geometry within a single model.

Test data. For evaluation, we use NYU Depth V2, KITTI Eigen, the DIODE indoor split [21], iBims-1 [10], and SUN RGB-D [20]. NYU and KITTI evaluate the supervised indoor and outdoor domains used during training, while DIODE indoor, iBims-1, and SUN RGB-D serve as held-out transfer benchmarks without target-domain fine-tuning. We use the standard Eigen crop and 10 m cap for NYU, the Garg crop and 80 m cap for KITTI, and per-image median scaling with a 10 m cap for the indoor transfer benchmarks.

Metrics. We report standard monocular depth metrics: AbsRel, RMSE, and threshold accuracies $\delta _ { 1 }$ and $\delta _ { 2 }$ . The threshold metrics measure the fraction of valid pixels whose predicted depth is within 1.25 and $1 . 2 5 ^ { 2 }$ of the ground truth, respectively. We omit $\delta _ { 3 }$ from the main comparison table to keep it readable.

Baselines and scope. We compare against representative monocular depth methods under the closest available protocols. Web-scale foundation models such as Depth Anything V2 [22], ZoeDepth [4], Metric3D [23], and UniDepthV2 [16] use substantially larger depth-data or training budgets and are listed only as contextual references. BinsFormer [12] is discussed in related work but omitted from Tab. 1 where comparable multi-benchmark numbers were unavailable. Full LangDepth uses concept routing, SPM, DINOv2-L, PPAN, and confidencegated fusion, and is fine-tuned for 20 epochs from a prior NYU+KITTI checkpoint using AdamW and cosine learning-rate decay on an H100 GPU with 3:1 NYU:KITTI sampling. At inference, it runs at 13.8 FPS at 416×544 resolution.

## 4.2 Unified results across supervised and transfer benchmarks

Tab. 1 reports performance across five benchmarks: NYU Depth V2, KITTI Eigen, DIODE indoor, iBims-1, and SUN RGB-D. NYU and KITTI evaluate the supervised indoor and outdoor domains used during training, while DIODE indoor, iBims-1, and SUN RGB-D evaluate transfer to held-out scene distributions. All evaluations in Tab. 1 follow the main benchmark protocol: median scaling on indoor transfer benchmarks, Garg crop on KITTI, no target-domain fine-tuning, and standard benchmark-specific depth caps.

For transparency, Tab. 1 combines numbers from multiple sources. We train and evaluate LangDepth on all five benchmarks. DepthFormer [re-train] is our reproduced SwinL checkpoint trained on NYU Depth V2 only; WorDepth is evaluated from the released checkpoint where comparable evaluation was possible. Remaining entries follow the closest matching protocol from the corresponding papers or reported baseline matrices, and unavailable comparisons are left blank rather than mixing incompatible protocols.

On NYU Depth V2, LangDepth achieves strong in-domain accuracy, reaching $\delta _ { 1 } \mathrm { = } 0 . 9 6 0$ with AbsRel 0.073. On KITTI Eigen, the same model maintains strong outdoor metric performance, reaching $\delta _ { 1 } \mathrm { = } 0 . 9 6 0$ . Methods such as NeWCRFs, PixelFormer, iDisc, and WorDepth report higher KITTI $\delta _ { 1 }$ under KITTI-specific training setups (0.975–0.979), while LangDepth is trained jointly across both NYU and KITTI. KITTI RMSE is also higher than in KITTI-specialist methods, suggesting that the mixed-domain objective preserves broad indoor–outdoor and transfer behaviour at the cost of some KITTI-specific fine-scale metric precision. This broader mixed-domain setting trades some KITTI-specific accuracy for a single model that also supports compact indoor depth and held-out transfer benchmarks.

On the held-out transfer benchmarks, under the evaluation protocol described above, LangDepth achieves the best $\delta _ { 1 }$ among the non-web-scale methods

Table 1: Unified results across supervised and transfer benchmarks. LangDepth is trained jointly on NYU Depth V2 and KITTI; baselines use their original recipes or the closest reproducible protocol. DIODE indoor, iBims-1, and SUN RGB-D are held-out transfer benchmarks (no target-domain fine-tuning). Indoor transfer: per-image median scaling, 10 m cap. KITTI: Eigen split, Garg crop, 80 m cap. “— ” = no comparable result. <sup>†</sup> Web-scale models listed for context only (larger depthdata or training budget). DepthFormer [re-train]: SwinL trained on NYU only; KITTI/DIODE indoor/iBims-1 reflect NYU-to-domain transfer. iDisc: DIODE/SUN δ<sub>1</sub> from paper; δ<sub>2</sub> and iBims-1 not reported. WorDepth: released-checkpoint evaluation. ↓ lower is better; ↑ higher is better.
<table><tr><td></td><td colspan="4">NYU Depth V2</td><td colspan="4">KITTI Eigen</td><td colspan="4">DIODE indoor</td><td colspan="4">iBims-1</td><td colspan="4">SUN RGB-D</td></tr><tr><td>Method</td><td>AbsRel ↓ RMSE ↓</td><td></td><td>δ1↑</td><td>δ2↑</td><td>AbsRel ↓</td><td>RMSE ↓</td><td>δ1↑</td><td>δ2 ↑</td><td></td><td>AbsRel ↓ RMSE ↓</td><td>δ1↑</td><td>δ2↑</td><td>AbsRel ↓</td><td>RMSE ↓</td><td>δ1↑</td><td>δ2↑</td><td>AbsRel ↓</td><td>RMSE ↓</td><td>δ1↑</td><td>δ2↑</td></tr><tr><td>AdaBins [2]</td><td>0.103</td><td>0.364</td><td>0.903</td><td>0.984</td><td>0.058</td><td>2.360</td><td>0.964</td><td>0.995</td><td>0.186</td><td>0.768</td><td>0.743</td><td>0.916</td><td>0.125</td><td>0.655</td><td>0.835</td><td>0.968</td><td>0.159</td><td>0.476</td><td>0.771</td><td>0.944</td></tr><tr><td>LocalBins [3]</td><td>0.087</td><td>0.362</td><td>0.936</td><td>0.990</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.210</td><td>0.880</td><td>0.560</td><td>0.880</td><td>0.156</td><td>0.470</td><td>0.777</td><td>0.949</td></tr><tr><td>DPT [18]</td><td>0.110</td><td>0.357</td><td>0.904</td><td>0.988</td><td>0.062</td><td>2.575</td><td>0.959</td><td>0.995</td><td>0.132</td><td>0.613</td><td>0.853</td><td>0.945</td><td>0.098</td><td>0.537</td><td>0.888</td><td>0.986</td><td>0.358</td><td>1.230</td><td>0.852</td><td>0.925</td></tr><tr><td>DepthFormer [11]</td><td>0.103</td><td>0.390</td><td>0.902</td><td>0.983</td><td>0.220</td><td>8.771</td><td>0.608</td><td>0.852</td><td>0.161</td><td>0.695</td><td>0.784</td><td>0.937</td><td>0.114</td><td>0.596</td><td>0.862</td><td>0.972</td><td>0.143</td><td>0.431</td><td>0.790</td><td>0.964</td></tr><tr><td>NeWCRFs [24]</td><td>0.095</td><td>0.334</td><td>0.922</td><td>0.992</td><td>0.052</td><td>2.071</td><td>0.975</td><td>0.997</td><td>0.160</td><td>0.703</td><td>0.805</td><td>0.931</td><td>0.112</td><td>0.599</td><td>0.864</td><td>0.976</td><td>0.367</td><td>1.320</td><td>0.836</td><td>0.922</td></tr><tr><td>PixelFormer [1]</td><td>0.090</td><td>0.322</td><td>0.929</td><td>0.991</td><td>0.051</td><td>2.081</td><td>0.976</td><td>0.997</td><td>0.153</td><td>0.670</td><td>0.808</td><td>0.936</td><td>0.114</td><td>0.617</td><td>0.859</td><td>0.969</td><td>0.373</td><td>1.345</td><td>0.831</td><td>0.922</td></tr><tr><td>iDisc [15]</td><td>0.086</td><td>0.313</td><td>0.940</td><td>0.993</td><td>0.050</td><td>2.067</td><td>0.977</td><td>0.997</td><td>0.156</td><td>0.721</td><td>0.810</td><td></td><td></td><td></td><td></td><td></td><td>0.128</td><td>0.387</td><td>0.838</td><td></td></tr><tr><td>DepthCLIP [26]</td><td>0.344</td><td>1.074</td><td>0.474</td><td>0.746</td><td>0.428</td><td>11.896</td><td>0.314</td><td>0.584</td><td>0.237</td><td>0.976</td><td>0.648</td><td>0.864</td><td>0.294</td><td>1.373</td><td>0.550</td><td>0.796</td><td>0.346</td><td>1.229</td><td>0.836</td><td>0.950</td></tr><tr><td>WorDepth [25]</td><td>0.088</td><td>0.317</td><td>0.932</td><td>0.992</td><td>0.049</td><td>2.039</td><td>0.979</td><td>0.998</td><td>0.165</td><td>0.804</td><td>0.787</td><td>0.897</td><td>0.144</td><td>0.895</td><td>0.812</td><td>0.921</td><td>0.374</td><td>1.397</td><td>0.810</td><td>0.911</td></tr><tr><td>GLPN [8]</td><td>0.116</td><td>0.459</td><td>0.880</td><td>0.974</td><td>0.057</td><td>2.297</td><td>0.967</td><td>0.996</td><td>0.182</td><td>0.819</td><td>0.772</td><td>0.923</td><td>0.116</td><td>0.685</td><td>0.861</td><td>0.969</td><td>0.366</td><td>1.276</td><td>0.843</td><td>0.923</td></tr><tr><td>Web-scale foundation models</td><td></td><td></td><td>0.955</td><td>(different data/training budget - listed for context only):</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ZoeDepth Depth Any. V2 [22]†</td><td>0.075 0.056</td><td>0.270 0.206</td><td>0.984</td><td>0.995 0.998</td><td>0.045</td><td>1.861</td><td>0.983</td><td>0.998</td><td>0.137</td><td>0.580</td><td>0.899</td><td>0.958</td><td>0.074</td><td>0.427</td><td>0.935</td><td>0.992</td><td></td><td></td><td></td><td></td></tr><tr><td>LangDepth (Ours)</td><td>0.073</td><td>0.292</td><td></td><td>0.960 0.993</td><td>0.064</td><td>2.782</td><td>0.960</td><td>0.995</td><td>0.110</td><td>0.533</td><td>0.887 0.964</td><td></td><td>0.070</td><td>0.397</td><td></td><td>0.946 0.987</td><td>0.353</td><td>1.195</td><td></td><td>0.861 0.924</td></tr></table>

included in Tab. 1: 0.887 on DIODE indoor, 0.946 on iBims-1, and 0.861 on SUN RGB-D. These results suggest that the scene-conditioned output space transfers beyond the supervised domains without target-domain fine-tuning. Overall, the results support scene-conditioned output spaces as a practical mechanism for cross-domain depth prediction while remaining compatible with strong visual backbones.

## 4.3 Component ablations

We next isolate the contribution of the main architectural components using matched training ablations. The ablation and perturbation studies follow a matched internal evaluation protocol for controlled component comparison; their absolute scores may therefore difer from the main benchmark results in Tab. 1 and should be interpreted only within the same table/protocol. All variants are trained under the same protocol and difer only in the component being ablated. The w/o gate fusion row corresponds to the implementation variant visual\_only: it bypasses the final SPM-vision fusion and uses only the PPAN visual estimate, while CLIP concept routing and SPM computation remain active. It should therefore not be interpreted as a no-text model.

The ablation results should be interpreted as evidence of architectural balance rather than as a large single-component accuracy gain. The full model is not uniformly best on every individual dataset; in particular, the variant without a learned SPM (fixed global bins) achieves KITTI $\delta _ { 1 } \mathrm { = } 0 . 9 6 5 3$ , compared to 0.9601 for the full model, at the cost of reduced DIODE and SUN performance. However, replacing concept routing with a single global CLIP descriptor reduces KITTI $\delta _ { 1 }$ and DIODE $\delta _ { 1 }$ , indicating that concept-specific routing provides a useful scene representation for cross-domain transfer. The SPM variants further show that the predicted range is the dominant functional component of the SPM: using the predicted range with a uniform weighting remains close to the full model. Overall, the full model provides a strong multi-dataset balance while preserving the interpretability of a scene-conditioned range prior and confidencegated fusion. This also explains why the $w / o$ gate fusion variant remains close to the full model: it removes only the final SPM-prior blend, while retaining the routed scene descriptor and SPM-defined bin structure used by PPAN. In contrast, the context perturbation study directly corrupts the scene descriptor before both the SPM and PPAN consume it, so it tests a stronger intervention on the scene-conditioned output space.

Table 2: Component ablation study. All variants use the same matched ablation protocol and are intended for relative component comparison. w/o gate fusion bypasses the confidence-gated SPM blend but still uses CLIP concept routing and the SPMdefined bin structure; it is not a no-text baseline. The ablation protocol applies perimage median scaling uniformly on all benchmarks, including KITTI, which accounts for the diference in KITTI AbsRel relative to Tab. 1 (0.054 vs. 0.064); the latter follows the benchmark-standard direct metric evaluation for KITTI.
<table><tr><td>Method</td><td>NYU AbsRel</td><td>NYU δ1</td><td>KITTI AbsRel</td><td>KITTIδ1</td><td>DIODE δ1</td><td>iBims δ1</td><td>SUN δ1</td></tr><tr><td>Ours</td><td>0.0727</td><td>0.9600</td><td>0.0540</td><td>0.9601</td><td>0.9030</td><td>0.9572</td><td>0.8663</td></tr><tr><td>w/o gate fusion</td><td>0.0725</td><td>0.9600</td><td>0.0539</td><td>0.9607</td><td>0.9028</td><td>0.9556</td><td>0.8664</td></tr><tr><td>w/o concept routing</td><td>0.0726</td><td>0.9600</td><td>0.0562</td><td>0.9554</td><td>0.8966</td><td>0.9543</td><td>0.8653</td></tr><tr><td>w/o learned SPM</td><td>0.0729</td><td>0.9597</td><td>0.0521</td><td>0.9653</td><td>0.8997</td><td>0.9565</td><td>0.8650</td></tr><tr><td>SPM range only</td><td>0.0729</td><td>0.9597</td><td>0.0543</td><td>0.9598</td><td>0.8991</td><td>0.9573</td><td>0.8661</td></tr></table>

Table 3: SPM range statistics. The SPM predicts compact indoor ranges and a much wider outdoor range for KITTI. The learned prior over bin centres is highly concentrated across datasets, indicating that the SPM primarily acts as a sceneconditioned range predictor and global scale anchor.
<table><tr><td>Dataset</td><td> $d _ { \mathrm { m i n } } \ \mathrm { ( m ) }$ </td><td> $d _ { \mathrm { m a x } } \ \mathrm { ( m ) }$ </td><td>Width (m)</td></tr><tr><td>NYU</td><td> $0 . 8 5 \pm 0 . 2 1$ </td><td> $8 . 8 4 \pm 0 . 2 1$ </td><td> $7 . 9 9 \pm 0 . 0 0$ </td></tr><tr><td>DIODE</td><td> $0 . 9 4 \pm 0 . 5 7$ </td><td> $9 . 0 0 \pm 1 . 4 3$ </td><td> $8 . 0 6 \pm 1 . 0 6$ </td></tr><tr><td>iBims</td><td> $0 . 9 2 \pm 0 . 3 5$ </td><td> $8 . 9 2 \pm 0 . 3 5$ </td><td> $7 . 9 9 \pm 0 . 0 0 3$ </td></tr><tr><td>SUN</td><td> $0 . 8 7 \pm 0 . 2 2$ </td><td> $8 . 8 6 \pm 0 . 2 2$ </td><td> $7 . 9 9 \pm 0 . 0 0 4$ </td></tr><tr><td>KITTI</td><td> $5 . 5 4 \pm 0 . 8 6$ </td><td> $6 1 . 1 3 \pm 1 1 . 0 1$ </td><td> $5 5 . 5 9 \pm 1 0 . 7 9$ </td></tr></table>

## 4.4 Scene-conditioned range and gate behaviour

To understand what the SPM learns, we inspect its predicted depth ranges across datasets. The SPM predicts compact ranges of approximately 8–10 m for indoor datasets and a much wider range for KITTI, whose mean predicted width is 55.6 m. This indicates that the CLIP-routed scene descriptor separates indoor and outdoor scale regimes without explicit domain labels.

Table 4: Confidence gate statistics. KITTI is strongly SPM-dominant, while indoor datasets are more balanced or slightly vision-dominant.
<table><tr><td>Dataset</td><td> ${ \mathrm { m e a n } } ( g )$ </td><td> ${ \mathrm { m e d i a n } } ( g )$ </td><td> $\operatorname { s t d } ( g )$ </td><td> $g < 0 . 5$ </td><td> $g > 0 . 9$ </td></tr><tr><td>NYU</td><td>0.539</td><td>0.532</td><td>0.058</td><td>8.8%</td><td>0.6%</td></tr><tr><td>KITTI</td><td>0.359</td><td>0.318</td><td>0.129</td><td>90.1%</td><td>1.5%</td></tr><tr><td>DIODE</td><td>0.514</td><td>0.517</td><td>0.039</td><td>23.3%</td><td>0.01%</td></tr><tr><td>iBims</td><td>0.528</td><td>0.527</td><td>0.037</td><td>14.7%</td><td>0.001%</td></tr><tr><td>SUN</td><td>0.510</td><td>0.504</td><td>0.091</td><td>46.6%</td><td>1.3%</td></tr></table>

The learned prior over bins is highly concentrated across all datasets, indicating that the SPM functions primarily as a scene-conditioned range predictor and global scale anchor. This behaviour is useful because the module learns a clear separation between compact indoor ranges and long-range outdoor KITTI scenes, exposing the scene-level structure that the visual decoder can use for depth prediction.

Tab. 4 shows the distribution of the per-pixel gate g across datasets. Since fusion is defined as $\ddot { d } = \mathbf { g } \odot d _ { \mathrm { v i s } } + ( 1 - \mathbf { g } )$ ⊙ d<sub>SPM</sub> $\left( \mathrm { E q . \ ( 3 ) } \right)$ , larger values favour the dense visual estimate and smaller values favour the SPM prior.

On KITTI, 90.1% of pixels have $\mathbf { g } < 0 . 5 .$ , so the model relies more heavily on the SPM range prior for outdoor driving scenes. Indoor datasets are closer to balanced or mildly vision-dominant. This suggests that the gate learns a domain-adaptive fusion strategy, weighting local visual evidence and the global range prior diferently as scene scale changes.

## 4.5 Scene-context perturbation

We further test whether the routed scene descriptor is actively used at inference time. For a fixed checkpoint, we intercept the output of the CLIP conceptrouting module and perturb the scene descriptor before it is consumed by the SPM and PPAN. All model weights remain fixed. Tab. 5 reports $\delta _ { 1 }$ under several perturbations.

Zeroing the context causes severe degradation on KITTI and iBims-1, and wrong-domain context degrades all evaluated datasets. Random, shufled, and batch-mean contexts also hurt KITTI substantially. These results show that the routed scene descriptor is a structurally active depth-scale signal on KITTI and iBims-1, which rely heavily on the SPM range prior. Zeroing or corrupting the descriptor disrupts the scene-scale anchor and causes large accuracy drops on these benchmarks. SUN is an exception, where zero context slightly improves $\delta _ { 1 }$ suggesting that the efect of scene context can be dataset-dependent on heterogeneous indoor scenes.

Table 5: Eval-time scene-context perturbation. Perturbing the routed scene descriptor at evaluation time causes large accuracy drops on KITTI and iBims-1. ∆ denotes the change in $\delta _ { 1 }$ relative to the original prediction.
<table><tr><td>Perturbation</td><td>DIODE</td><td>Δ</td><td>iBims</td><td>Δ</td><td>SUN</td><td>Δ</td><td>KITTI</td><td>Δ</td></tr><tr><td>Original</td><td>0.9030</td><td></td><td>0.9572</td><td></td><td>0.8663</td><td></td><td>0.9601</td><td></td></tr><tr><td>Zero context</td><td>0.7333</td><td>-0.170</td><td>0.5900</td><td>-0.367</td><td>0.8933</td><td>+0.027</td><td>0.4685</td><td>-0.492</td></tr><tr><td>Wrong domain</td><td>0.8711</td><td>-0.032</td><td>0.8452</td><td>-0.112</td><td>0.8518</td><td>-0.014</td><td>0.8747</td><td>-0.085</td></tr><tr><td>Shuffled</td><td>0.8940</td><td>-0.009</td><td>0.9448</td><td>-0.012</td><td>0.8640</td><td>-0.002</td><td>0.8833</td><td>-0.077</td></tr><tr><td>Batch mean</td><td>0.8957</td><td>-0.007</td><td>0.9497</td><td>-0.008</td><td>0.8667</td><td>+0.000</td><td>0.8614</td><td>-0.099</td></tr><tr><td>Gaussian</td><td>0.8965</td><td>-0.007</td><td>0.9508</td><td>-0.006</td><td>0.8664</td><td>-0.000</td><td>0.8611</td><td>-0.099</td></tr></table>

## 4.6 Discussion and limitations

The results indicate that LangDepth is best understood as a structured scenescale mechanism rather than only as an accuracy-oriented architectural change. Strong DINOv2 features already provide robust local geometry, while the proposed semantic pathway adds an interpretable way to adapt the metric range across scene domains. This is reflected consistently across the analysis: the SPM separates compact indoor ranges from long-range KITTI scenes, the gate adjusts the balance between local visual evidence and the global range prior by domain, and corrupting the routed scene descriptor causes large drops on KITTI and iBims-1. These findings show that the scene descriptor is actively used for scale prediction and that the predicted output space is meaningfully scene-conditioned. At the same time, SUN RGB-D remains a challenging heterogeneous benchmark, where scene cues can be noisier; improving robustness on mixed indoor–outdoor scenes is an important direction for future work.

## 5 Conclusion

We presented LangDepth, a monocular depth architecture that uses CLIPderived scene semantics to define a scene-conditioned metric output space before dense visual prediction. A Scene Prior Module predicts a depth range and global scale anchor from a routed scene descriptor, while a DINOv2–FPN visual branch assigns pixels within the resulting bin structure and a learned gate fuses local visual evidence with the SPM prior. Across NYU, KITTI, and held-out indoor transfer benchmarks, LangDepth provides a strong mixed-domain depth model with an interpretable scene-scale pathway. Range statistics, gate behaviour, and context perturbations show that the semantic branch is structurally active, especially on KITTI and iBims-1. These findings support scene-conditioned output spaces as a practical mechanism for cross-domain monocular depth estimation, while also highlighting the need for more robust scene priors on heterogeneous datasets such as SUN RGB-D.

## References

1. Agarwal, A., Arora, C.: PixelFormer: Global context and local features cooperated transformer for depth estimation. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2023) 3, 11

2. Bhat, S.F., Alhashim, I., Wonka, P.: AdaBins: Depth estimation using adaptive bins. In: IEEE Conf. Comput. Vis. Pattern Recog. (2021) 2, 3, 11

3. Bhat, S.F., Alhashim, I., Wonka, P.: LocalBins: Improving depth estimation by learning local distributions. In: Eur. Conf. Comput. Vis. (2022) 3, 11

4. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288 (2023) 4, 10, 11

5. Eigen, D., Fergus, R.: Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In: Int. Conf. Comput. Vis. (2015) 3

6. Eigen, D., Puhrsch, C., Fergus, R.: Depth map prediction from a single image using a multi-scale deep network. In: Adv. Neural Inform. Process. Syst. (2014) 3, 9

7. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The KITTI dataset. The International Journal of Robotics Research 32(11), 1231–1237 (2013) 9

8. Kim, D., Ka, W., Ahn, P., Joo, D., Chun, S., Kim, J.: Global-local path networks for monocular depth estimation with vertical cutdepth. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022) 11

9. Kim, D., Lee, S.: CLIP can understand depth. In: Int. Conf. Mach. Learn. (2024) 2, 4

10. Koch, T., Liebel, L., Fraundorfer, F., Körner, M.: Benchmark for evaluating monocular visual depth estimation on challenging scenes. arXiv preprint arXiv:1805.01328 (2018) 9

11. Li, Z., Chen, Z., Liu, X., Jiang, J.: DepthFormer: Exploiting long-range correlation and local information for accurate monocular depth estimation. Machine Intelligence Research (2023) 3, 11

12. Li, Z., Wang, X., Liu, X., Jiang, J.: BinsFormer: Revisiting adaptive bins for monocular depth estimation. IEEE Trans. Image Process. (2023) 2, 3, 10

13. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., et al.: DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023) 2, 5, 7

14. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: FiLM: Visual reasoning with a general conditioning layer. In: AAAI Conference on Artificial Intelligence (2018) 5

15. Piccinelli, L., Sakaridis, C., Yu, F.: iDisc: Internal discretization for monocular depth estimation. In: IEEE Conf. Comput. Vis. Pattern Recog. (2023) 2, 3, 11

16. Piccinelli, L., Yang, Y.H., Sakaridis, C., Segu, M., Li, S., Van Gool, L., Yu, F.: Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv:2502.20110 (2025) 4, 10

17. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: Int. Conf. Mach. Learn. (2021) 2, 5

18. Ranftl, R., Bochkovskiy, A., Koltun, V., Dosovitskiy, A.: Vision transformers for dense prediction. In: Int. Conf. Comput. Vis. (2021) 3, 11

19. Silberman, N., Hoiem, D., Kohli, P., Fergus, R.: Indoor segmentation and support inference from RGBD images. In: Eur. Conf. Comput. Vis. (2012) 9

20. Song, S., Lichtenberg, S.P., Xiao, J.: SUN RGB-D: A rgb-d scene understanding benchmark suite. In: IEEE Conf. Comput. Vis. Pattern Recog. (2015) 9

21. Vasiljevic, I., Kolkin, N., Zhang, S., Luo, R., Wang, H., Dai, F.Z., Daniele, A.F., Mostajabi, M., Basart, S., Walter, M.R., et al.: DIODE: A dense indoor and outdoor depth dataset. In: arXiv preprint arXiv:1908.00463 (2019) 9

22. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. In: Adv. Neural Inform. Process. Syst. (2024) 2, 4, 10, 11

23. Yin, W., Zhang, C., Chen, H., Cai, Z., Yu, G., Wang, K., Chen, X., Shen, C.: Metric3D: Towards zero-shot metric 3d prediction from a single image. In: Int. Conf. Comput. Vis. (2023) 4, 10

24. Yuan, W., Gu, X., Dai, Z., Zhu, S., Tan, P.: Neural window fully-connected crfs for monocular depth estimation. In: IEEE Conf. Comput. Vis. Pattern Recog. (2022) 3, 11

25. Zeng, Z., Wang, D., Yang, F., Park, H., Soatto, S., Lao, D., Wong, A.: WorDepth: Variational language prior for monocular depth estimation. In: IEEE Conf. Comput. Vis. Pattern Recog. (2024) 2, 4, 11

26. Zhang, R., Zeng, Z., Guo, Z., Li, Y.: DepthCLIP: Can language understand depth? In: ACM Int. Conf. Multimedia (2022) 2, 4, 11

27. Zhao, W., Rao, Y., Liu, Z., Liu, B., Zhou, J., Lu, J.: Unleashing text-to-image difusion models for visual perception. In: Int. Conf. Comput. Vis. (2023) 2, 4