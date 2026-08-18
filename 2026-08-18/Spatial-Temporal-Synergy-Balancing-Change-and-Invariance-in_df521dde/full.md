# Spatial-Temporal Synergy: Balancing Change and Invariance in Text-Driven 3D Human Motion Editing

Shaohui Lin, Zhenwu Shi, Jingyu Gong<sup>∗</sup>, Jiao Xie, Yu Zhou, Baochang Zhang, Lizhuang Ma and Chia-Wen Lin Fellow, IEEE

Abstract—Text-driven human motion editing aims to modify existing motion sequences according to natural language instructions while maintaining the structural consistency of the original motion. Existing diffusion-based approaches struggle to balance text-responsive “change” and inertial “invariance”. They often rely on coarse spatial constraints and rigid uniform time assumptions leading to spatial motion distortions and the destruction of intrinsic physical rhythms during variable-length editing. To handle these challenges, we propose Change and Invariance Motion Editing (CIME), a unified framework that comprehensively decouples change and invariance into spatial pose and temporal rhythm dimensions. For spatial poses, our method integrates an omni-supervised positive-negative learning mechanism comprising hierarchical retrospective feature supervision, subtle motion preservation, and triplet-based semantic alignment. For temporal rhythms, we introduce the Riemannian Non-uniform Integral Manifold Mapping (RNIMM) module, which achieves high-fidelity reproduction of physical beats in the edited text via kinematics-aware non-uniform timestamps. Extensive experiments on the MotionFix and STANCE Adjustment datasets demonstrate that CIME achieves state-of-the-art performance in editing alignment and structural fidelity, validating the effectiveness of our unified architecture. Our source codes and models have been released at: github.com/ZhenwuShi/CIME.git

Index Terms—Motion Editing, Human Motion Synthesis, Diffusion Models, Omni-supervised Positive-negative Learning, Riemannian Integral Geometry

## 1 INTRODUCTION

Text-driven 3D human motion editing has emerged as a foundational technology for intuitive animation generation, virtual reality interaction, and digital human behavior simu-

![](images/c9f8722ccbedf0ea7357df665b39ef2e1a9659be1f37d19dc58bf251168b0155.jpg)  
Slow motion: low energy Fast motion: high energy Slow motion: low energy  
Fig. 1: Display information on fast motion and slow motion.

lation. Early motion synthesis primarily focused on autoregressive pose forecasting [1], [2], [3], [4]. By bridging the semantic gap between human language and 3D kinematics, modern editing technology allows users to execute finegrained modifications on existing motion sequences simply by providing free-form natural language instructions. This capability not only significantly reduces the manual labor required by traditional animators but also enables highly efficient reuse and personalization of massive motion capture assets. Compared to generating entirely new kinematic sequences from scratch, the motion editing task aims to maximally preserve the structural properties and stylistic nuances of the original movements. Consequently, this technology demonstrates immense practical value across diverse multimedia applications, high-fidelity game development, and complex virtual reality ecosystems.

With the rapid advancement of deep generative learning, Denoising Diffusion Probabilistic Models (DDPMs) [5], [6] have established themselves as the dominant paradigm in human motion synthesis [7]. Building upon the solid foundations of standard text-to-motion generation, researchers have progressively moved toward more flexible, interactive, and controllable motion editing tasks. Recently, pioneering frameworks such as MotionFix [8] have introduced comprehensive text-guided benchmarks and standardized triplet datasets, driving the community toward true openvocabulary motion manipulation rather than relying on isolated action tags. On top of these benchmarks, state-ofthe-art approaches like SimMotionEdit [9] have incorporated explicit motion similarity prediction as an auxiliary target to constrain the generation process. Although these models achieve certain performance, they take a relatively simplified view of the temporal alignment between the source and target sequences. Moreover, how to perfectly balance text-responsive “change” and inertial “invariance” remains an unsolved challenge, which requires meticulous synergy across both the spatial pose and temporal rhythm dimensions. In this work, we formally formulate this core problem as Change and Invariance Motion Editing (CIME). The essence of CIME is to handle the editing task from two complementary perspectives, namely spatial pose and temporal rhythm, such that text-driven modifications can be accurately applied while the original motion structure is faithfully maintained.

![](images/a47f4790815bad0ad60c3c93a3da294fdc689f0314576545a8064a14f2eb3d8f.jpg)  
Fig. 2: CIME is a unified framework for text-driven 3D human motion editing. Given a source motion and a natural language instruction, CIME edits the source motion to produce the desired target motion while balancing change and invariance across both spatial pose and temporal rhythm dimensions.

At the spatial pose level, existing generative constraints are typically restricted to coarse-grained global conditions or a single terminal gradient feedback from the final network layer. Although several text-conditioned frameworks [10], [11] support editing, they often lose track of original local features during target motion generation. Without continuous intermediate constraints, such as a preservation factor $m \in [ 0 , 1 ] .$ , models executing local pose changes are easily dominated by global text instructions. This causes the network to completely overwrite the original motion details. Consequently, these methods fail to smoothly maintain structural invariance in unaffected body regions. This specific limitation directly motivates our design of a continuous spatial constraint. More critically, a severe temporal mapping dilemma arises during variable-length editing.

Because the input source motion and the generated target one often differ in total frame count, they inherently lack a direct, frame-to-frame correspondence. Traditional motion diffusion models and temporal generation frameworks [12], [13] almost universally default to a flat and uniform Euclidean time assumption. This assumption is strictly bound to rigid, equidistant frame indices. Unfortunately, when textual instructions demand macroscopic changes in speed or duration, this rigid time structure forces a uniform stretching or compressing of the timeline. This will definitely undermine the rhythmic modifications induced by the textual instructions for motion editing.

Our core motivation stems from the critical necessity to break this uniform time assumption and establish a flexible, dynamic mapping mechanism that naturally accommodates variable-length editing. Since the source and target sequences are not strictly one-to-one, we inherently require a non-uniform mapping strategy to automatically synchronize different kinematic timelines rather than relying on fixed frame indices. Furthermore, pure time-domain feature learning struggles to master subtle velocity fluctuations and complex rhythm variations, such as fast actions or sudden localized pauses (as visually illustrated in Fig. 1). To capture these minute temporal deviations and explicitly learn the complex transitions of motion speed, we need to rethink the temporal architecture and model the timing axis based on the intrinsic geometric properties of the motion itself. By mapping the motion progress to a non-uniform timeline, the network can naturally preserve the intrinsic physical dynamics without being distorted by macroscopic duration discrepancies.

To implement the aforementioned concepts, we propose a unified framework termed CIME (as illustrated in Fig. 2), which comprehensively decouples ”change and invariance” into two core dimensions: temporal rhythm and spatial pose. For the spatial pose, the framework integrates a comprehensive omni-supervised positive-negative learning mechanism to regulate spatial structural integrity. For the temporal rhythm, we introduce the Riemannian Nonuniform Integral Manifold Mapping (RNIMM) module to explicitly model the timing axis.

A preliminary version of this work was presented as OmniME [14] in CVPR 2026. The earlier version primarily focused on the spatial dimension, relying solely on the omni-supervised mechanism for pose editing. This journal manuscript fundamentally extends the preliminary work into a “spatial-temporal synergy” paradigm. Crucially, the newly proposed RNIMM module leverages Riemannian integral geometry to compute kinematics-aware non-uniform timestamps via arc-length parameterization, replacing the standard uniform time assumption and successfully preserving intrinsic physical beats against sequence length variations.

We conduct extensive evaluations on the MotionFix and STANCE Adjustment datasets. To thoroughly validate the proposed extensions, we significantly expand our experimental scope by introducing fine-grained diagnostic analyses and zero-shot cross-dataset generalization tests. The results systematically demonstrate that CIME not only achieves state-of-the-art retrieval accuracy but also exhibits exceptional robustness across different domains.

Our contributions are summarized as follows:

• We propose CIME, a novel and unified framework for text-driven 3D human motion editing that explicitly balances text-responsive change and inertial invariance across both spatial pose and temporal rhythm dimensions.

• We design a robust omni-supervised positive-negative learning mechanism that exerts tight constraints on pose change and invariance across feature, motion, and semantic levels, significantly enhancing spatial precision and preventing unwanted joint distortions.

• We propose the Riemannian Non-uniform Integral Manifold Mapping (RNIMM) module, which aims to tackle the temporal mapping dilemma under variablelength editing by utilizing kinematics-aware nonuniform timestamps and preserves intrinsic physical rhythms.

## 2 RELATED WORK

## 2.1 Human Motion Generation

Driven by large-scale kinematic datasets [15], [16], [17], [18], early motion synthesis primarily focused on autoregressive pose forecasting [1], [2], [3], [4]. To bridge the semantic gap, subsequent deep learning approaches introduced natural language and discrete action categories as conditioning signals to enhance behavioral alignment [12], [19], [20], [21], [22], [23], [24], [25], [26].

Recently, Denoising Diffusion Probabilistic Models (DDPMs) [5], [6] have fundamentally shifted motion synthesis by formulating it as a gradual denoising process, achieving state-of-the-art performance in text-driven generation [7], [11], [13], [27], [28], [29], [30], [31], [32], [33]. Moving beyond isolated action tags, recent frameworks leverage sentence-level datasets [15], [17], [34] to interpret openended linguistic descriptions [7], [13], [35], [36], [37]. To ensure sequence fidelity, methods like Lagrangian Motion Fields [38] enhance temporal dynamics for long-term generation, while multimodal frameworks like MotionLLM [39] leverage the reasoning capabilities of Large Language Models to achieve deeper alignment between complex linguistic instructions and fine-grained behaviors.

Building upon these foundations, diffusion-based methods have naturally extended to spatial-temporal editing tasks like inbetweening and inpainting [7], [35]. Concurrently, advanced approaches such as FineMoGen [40] harness LLMs to facilitate localized motion editing. This progression, alongside new benchmarks for open-vocabulary generation [41], underscores the ongoing demand for semantic-aware, highly generalizable motion editing systems. While these approaches excel at open-ended synthesis, our framework specifically targets the structural preservation of source motions, enabling semantic-aware editing through a continuous spatial constraint mechanism.

## 2.2 Human Motion Editing

Early motion editing heavily relied on spatial and temporal constraints [42], [43], [44] for tasks like trajectory alteration [45], skeletal retargeting [46], and style transfer [47]. Deep learning subsequently advanced stylistic translation [46], [48], localized pose refinement [49], and temporal in-betweening [35], [50], [51]. However, achieving fine-grained editing strictly through text remained a formidable challenge.

Diffusion models have recently revitalized this domain. Researchers have explored mask-based spatial inpainting [10] and latent query substitution [52]. While several text-conditioned frameworks [13], [53] support editing, they often hard-freeze unedited joints, which severely disrupts kinematic fluidity. Alternative compositional frameworks [12], [54], [55], [56], [57] offer smoother transitions but are bottlenecked by heavy manual annotation requirements.

To enable more intuitive interaction, recent pipelines like PoseFix [58] utilize free-form natural language. Concurrently, LLM-empowered techniques [40], [59] manipulate motions at textual or latent levels. For standardized evaluation, MotionFix [8] contributed a comprehensive textguided editing benchmark. Advanced systems [40], [59], [60] further refined semantic control, though they inherently suffer from dense annotation burdens. Additionally, recent efforts explicitly predict joint-wise motion differences via cross-axis feature fusion for precise text-based editing [61]. Motivated by these benchmarks and datasets [8], we propose a framework to balance change and invariance across both spatial poses and temporal rhythms.

## 2.3 Generative Model Supervision

Effective supervision mechanisms are crucial for synthesizing realistic and semantically faithful outputs. In deep generative models, perceptual constraints [62] and adversarial feature matching [63] frequently preserve high-fidelity structural details. Concurrently, metric-based optimization, such as triplet and contrastive regularization [64], [65], effectively organizes the underlying latent spaces.

For human motion editing, supervision must simultaneously address cross-modal semantic alignment and kinematic continuity. MotionCLIP [23] achieves semantic alignment using pretrained contrastive encoders. To standardize text-guided editing, MotionFix [8] provides a diffusion baseline optimized on annotated triplets comprising source motions, target behaviors, and textual instructions. Meanwhile, diffusion frameworks [7], [35] incorporate explicit temporalconsistency penalties to ensure smooth and natural kinematic transitions during editing tasks.

Recently, researchers have enriched supervisory signals via novel data curation and auxiliary objectives. Dynamic augmentations like MotionCutMix [66] expand the training distribution, while frameworks like SimMotionEdit [9] use auxiliary similarity-prediction targets to implicitly constrain generation. Furthermore, advanced systems like FineMoGen [40] leverage LLMs to construct fine-grained supervisory signals for localized spatial control. Different from existing discrete or auxiliary supervisory objectives, our framework employs a continuous geometric supervision approach that explicitly regulates the balance between temporal rhythm invariance and local pose adaptability.

## 2.4 Riemannian Geometry and Temporal Manifolds

Riemannian geometry and manifold learning have been widely applied in human motion analysis. However, recent state-of-the-art research has predominantly focused on modeling the spatial constraints of skeletal poses. For instance, FlashMo [67] proposes geometric interpolation on Lie group manifolds to ensure the consistency of joint rotations; NRMF [68] constructs neural distance fields on Riemannian motion manifolds to explicitly model the physical priors of pose, velocity, and acceleration; and RMG [69] further explores motion generation on non-Euclidean product manifolds via Riemannian flow matching.

Despite significant advancements in spatial geometric constraints, existing mainstream motion diffusion models [13] and temporal generation frameworks [70] almost universally default to a flat and uniform Euclidean time assumption for temporal modeling, relying entirely on discrete and equidistant integer frame indices for feature progression. Departing from these Euclidean paradigms, we pioneer the introduction of Riemannian geodesics directly into the modeling of the time axis itself. By mapping the cumulative physical work of the motion to geodesic arc length, we construct a non-uniform temporal architecture that locks the invariant intrinsic physical beats against macroscopic sequence length changes.

## 3 METHODS

In the following parts, we will first establish the preliminaries of motion representation and formulate our editing objective in Section 3.1. Then, we will give an overview of our comprehensive motion editing framework in Section 3.2. Next, we will introduce the Semantic Alignment via Triplet Learning in Section 3.3, followed by the RNIMM module for temporal alignment and rhythm preservation in Section 3.4. Finally, we will present two additional supervision mechanisms that co-optimize the diffusion backbone, namely the Retrospective Feature Supervision in Section 3.5, and the Motion Preservation during editing in Section 3.6.

## 3.1 Preliminary

Motion Representation. Let $\mathbf { X } = [ \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { F } ]$ denote a continuous sequence of human motion frames, where each $\mathbf { x } _ { i }$ captures the comprehensive full-body pose at a specific temporal step. Adhering to the standard configuration established by MotionFix [8], we encode the frame-level feature into a 207-dimensional vector:

$$
\mathbf { x } _ { i } = [ \mathbf { v } _ { i } , \mathbf { o } _ { i } , \mathbf { r } _ { i } , \mathbf { p } _ { i } ] \in \mathbb { R } ^ { 2 0 7 } .\tag{1}
$$

Within this formulation, the individual components explicitly correspond to the global root velocity, the global spatial orientation, the local joint rotations, and the localized 3D joint positions, respectively.

Motion Editing. The primary objective of text-guided motion editing is to synthesize a targeted kinematic sequence by leveraging a given source motion X alongside a natural language instruction L. This generative mapping can be abstracted as $\mathbf { M } = \mathcal { G } ( \mathbf { X } , \mathbf { L } )$ , with $\mathcal { G }$ serving as the conditional synthesis network.

Editing Objective. First, the synthesized motion must accurately match the specific requirements of the textual instruction. Second, it preserves the inherent characteristics and unmodified regions of the raw source motion. Third, the entire motion sequence retains rigorous kinematic continuity and spatiotemporal coherence, ensuring smooth transition between motion frames without jarring artifacts.

Following the standard Denoising Diffusion Probabilistic Models [5], [6], the primary diffusion loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon , \mathbf { c } } \left\| \epsilon - \epsilon _ { \theta } \left( \mathbf { x } _ { t } , t , \mathbf { c } \right) \right\| _ { 2 } ^ { 2 } , } \end{array}\tag{2}
$$

where $\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } }$ ϵ is the noisy target motion, c denotes the combined conditioning from the text instruction and source motion, and $\epsilon _ { \theta }$ is the denoising network.

## 3.2 Overview

As depicted in Fig. 3, our framework balances the core philosophy of “change and invariance”. First, a pre-trained CLIP [71] encoder projects both textual and kinematic modalities into a shared semantic space. A Fusion Transformer then aggregates them to generate fused features. Crucially, these fused features encapsulate the textual semantics that fundamentally drive the motion modifications. Alongside the noise target’s frame length, they are processed by the Riemannian Non-uniform Integral Manifold Mapping module. This module aligns the features to the target timeline and injects them into the noise target, resolving the conflict between macroscopic length variations and intrinsic rhythm preservation.

The composite features are subsequently routed into a Diffusion Transformer [72]. To ensure high-quality synthesis, this backbone is co-optimized by three distinct mechanisms. First, a retrospective strategy explicitly constrains intermediate representations. Second, a dynamic preservation module maintains foundational postures in unedited regions. Third, a contrastive triplet objective strictly fortifies semantic alignment. Ultimately, the synergy of these modules achieves a perfect equilibrium between precise textdriven changes and high-fidelity structural invariance.

## 3.3 Semantic Alignment via Triplet Learning

To achieve strict semantic synchronization between edited motions and text prompts, a contrastive triplet objective operating directly on the terminal representations of the transformer backbone is integrated. Assuming $\mathbf { h } ^ { ( L ) } \in \mathbb { R } ^ { B \times T \times D }$ acts as the hidden feature map extracted from the final network layer, we collapse the temporal dimension through global average pooling to derive a compact motion embedding $\mathbf { z } _ { m } \in \mathbb { R } ^ { B \times ^ { \mathbf { \lambda } } D }$

$$
\mathbf { z } _ { m } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { h } _ { t } ^ { ( L ) } ,\tag{3}
$$

where $T$ indicates the temporal duration of the sequence.

Subsequently, we designate $\mathbf { z } _ { p } ~ \in ~ \mathbb { R } ^ { B \times D }$ as the positive linguistic condition intrinsically matching the target behavior, while $\mathbf { z } _ { n } \in \mathbb { R } ^ { B \times D }$ serves as a randomly sampled negative textual feature. The optimization landscape driving this semantic alignment is governed by the following triplet formulation:

$$
\mathcal { L } _ { \mathrm { t r i p l e t } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ \lVert \mathbf { z } _ { m } ^ { i } - \mathbf { z } _ { p } ^ { i } \rVert _ { 2 } ^ { 2 } - \lVert \mathbf { z } _ { m } ^ { i } - \mathbf { z } _ { n } ^ { i } \rVert _ { 2 } ^ { 2 } + \alpha \right] _ { + } .\tag{4}
$$

Here, the hinge function $[ \cdot ] _ { + }$ outputs the maximum between its argument and zero, while α represents the minimum geometric margin enforced between positive and negative pairs, which we empirically configure to 0.2 throughout our experiments. This contrastive regularization strictly compels the generated motion manifold to gravitate towards the correct textual semantics while actively repelling mismatched instructions, thereby guaranteeing high fidelity to the user’s intent.

![](images/9bcca8c4424fc8292f269ea67bb44df09ce02acd08dcb9d1a33c1be864d79f40.jpg)  
Fig. 3: Overview of CIME: First, the input text and source motion are projected into a shared semantic space via a CLIP encoder and aggregated by a Fusion Transformer. Next, the RNIMM module aligns these text-infused features with the target temporal dimension and injects them into the noise target. Finally, the composite features are processed by a Diffusion Transformer. This backbone is co-optimized by three auxiliary supervision mechanisms to achieve an optimal balance between precise editing and structure preservation, ultimately yielding the target motion.

## 3.4 Riemannian Non-uniform Integral Manifold Mapping

Kinematics-Aware Riemannian Geodesic Construction. Existing motion editing models generally treat human motion as a sequence of discrete frames with uniform time mappings. However, real human movements naturally involve varying intensities of physical momentum, such as explosive fast punches or localized static pauses. In addition, textual instructions also impose rhythmic adjustments on a subset of the motions therein. To strictly preserve the intrinsic physical dynamics and to capture both the changes and invariances dictated by the textual instructions, we break the assumption of uniform time mapping. Instead, we construct non-uniform geodesic timestamps on a Riemannian manifold based on kinematic features.

Inspired by the arc-length parameterization in Riemannian geometry, we model a motion sequence as a continuous trajectory γ(t) evolving on an underlying latent motion manifold. The intrinsic length of this trajectory can be formulated as:

$$
L ( \gamma ) = \int _ { 0 } ^ { T } \sqrt { \dot { \gamma } ( t ) ^ { T } g ( \gamma ( t ) ) \dot { \gamma } ( t ) } d t ,\tag{5}
$$

where $g ( \gamma ( t ) )$ denotes the Riemannian metric tensor that measures the local geometric structure of the latent manifold. However, practical motion representations are inherently discrete, where each sequence consists of finite latent motion features $F = \{ f _ { 0 } , f _ { 1 } , \dotsc , f _ { T - 1 } \}$ . Therefore, the continuous geodesic length is approximated by accumulating the local displacements between adjacent frames:

$$
s _ { i } = \sum _ { k = 0 } ^ { i } { \tilde { v } } _ { k } ,\tag{6}
$$

where $s _ { i }$ represents the accumulated intrinsic arc length up to the i-th frame, and $\tilde { v } _ { k }$ denotes the regularized local variation. Specifically, the instantaneous motion variation between neighboring latent states is computed as:

$$
v _ { i } = \| f _ { i } - f _ { i - 1 } \| _ { 2 } ,\tag{7}
$$

where $v _ { i }$ measures the kinematic change magnitude between two consecutive motion states. To avoid zero-length intervals caused by static motion segments, we introduce a small positive constant ϵ:

$$
\tilde { v } _ { i } = v _ { i } + \epsilon .\tag{8}
$$

The intrinsic temporal coordinate of each frame is obtained by normalizing the accumulated arc length:

$$
t _ { i } ^ { \mathrm { i n t r i n s i c } } = \frac { s _ { i } } { s _ { T - 1 } } .\tag{9}
$$

Finally, we divide the cumulative arc length of the current frame by the total arc length of the last frame. This yields the normalized physical rhythm progress $t _ { s r c } ^ { ( i ) }$ :

$$
t _ { s r c } ^ { ( i ) } = \frac { s _ { i } } { s _ { T _ { s r c } - 1 } } , \quad t _ { s r c } ^ { ( i ) } \in [ 0 , 1 ]\tag{10}
$$

Through this mechanism, each source frame receives a nonuniform temporal coordinate in the normalized arc-length parameter space that reflects its kinematic intensity: as illustrated in Fig. 4 frames in high-velocity regions are distributed more sparsely along the [0, 1] progress axis, while those in quasi-static regions are clustered more densely. This non-uniform temporal parameterization causes the downstream cross-sequence alignment to adaptively allocate greater temporal resolution toward dynamically rich regions, faithfully preserving the underlying kinematic rhythm even as the sequence duration varies, and providing a robust geometric baseline for subsequent cross-length motion editing.

After establishing the geodesic time for the source features, we construct the temporal mapping for the target sequence. Since the target motion is generated from pure noise via diffusion, it lacks any initial physical structure. Unlike the non-uniform progress of the source, the target sequence is strictly modeled as a Euclidean time manifold with a constant progression rate.

Given the target length $T _ { t g t }$ determined by textual instructions, we perform uniform linear sampling within the closed interval [0, 1] to generate the target timestamps:

$$
t _ { t g t } ^ { ( j ) } = \frac { j } { T _ { t g t } - 1 } , \quad j \in \{ 0 , 1 , \ldots , T _ { t g t } - 1 \}\tag{11}
$$

Regardless of how the target duration scales, this uniform ruler—through the optimal transport alignment described below—adaptively re-samples features from the non-uniform Riemannian manifold. Consequently, it decouples macroscopic sequence length changes from the preservation of localized physical beats.

Rhythm Mapping. To inject the constructed onedimensional non-uniform timestamps and overcome spectral bias, we introduce a high-frequency implicit neural representation. We apply a Fourier positional encoding γ(t) with logarithmically decaying frequencies, followed by a multi-layer perceptron to yield high-dimensional temporal features $\boldsymbol { E _ { t } } ^ { \star } \in \dot { \mathbb { R } } ^ { D }$ . To compute the first-order semantic cost $M _ { c o s t } ,$ these temporal features are explicitly added to the spatial source motion features to construct the keys and values. This mapping is indispensable, as it amplifies minute temporal deviations from the Riemannian geodesic that standard linear projections would obscure.

For downstream alignment, we propose a log-domain Fused Gromov-Wasserstein mechanism. To preserve the source kinematic rhythm, we define the intra-manifold topological distances as:

![](images/11074a86734f861e6ff36b5c1206dfdc3e1c234980e8fc4126134bb94e1a7a4a.jpg)  
Fig. 4: Illustration of arc length on Riemannian manifolds and encoding on non-uniform Riemannian manifolds. The orange boxes indicate regions of intense motion, and the red bars reflect higher instantaneous kinematic intensity.

$$
C _ { s r c } = ( t _ { s r c } \ominus t _ { s r c } ) ^ { 2 } , \quad C _ { t g t } = ( t _ { t g t } \ominus t _ { t g t } ) ^ { 2 }\tag{12}
$$

Combined with the semantic cost $M _ { c o s t } ,$ , the optimal transport plan P is stably solved via log-space Sinkhorn iterations, effectively preventing numerical overflow. Finally, we resample the source features using P. This seamlessly aligns the non-uniform Riemannian manifold with the uniform Euclidean timeline, yielding the aligned representation for the subsequent diffusion process.

## 3.5 Retrospective Feature Supervision

Building upon SimMotionEdit [9], we incorporate a retrospective feature supervision mechanism [73], [74] to enhance the optimization stability of the generative process. Our denoising backbone is instantiated as a Diffusion Transformer [72] comprising 8 sequential blocks. Breaking away from the conventional paradigm that relies exclusively on the terminal layer for gradient feedback, we establish multi-level intermediate constraints. Specifically, we append lightweight linear projection heads to the hidden states of the 2-nd, 4-th, and 6-th intermediate blocks.

Formally, assume $\mathbf { h } ^ { ( l ) } \in \mathbb { R } ^ { B \times T \times D }$ represents the internal feature map extracted from the l-th block, where the dimensions indicate the batch size, sequence length, and hidden channels, respectively. The auxiliary projection layer $f ^ { ( l ) } ( \cdot )$ transforms these hidden features directly back into the target kinematic space:

$$
\hat { \mathbf { x } } ^ { ( l ) } = f ^ { ( l ) } ( \mathbf { h } ^ { ( l ) } ) , \quad l \in \{ 2 , 4 , 6 \} ,\tag{13}
$$

Here, $\hat { \textbf { x } } ^ { ( l ) } ~ \in ~ \mathbb { R } ^ { B \times T \times J }$ signifies the recovered motion sequence, with J denoting the dimensions of the motion features.

To enforce structural consistency, we calculate the mean squared error between these intermediate predictions and the actual ground-truth motion x:

$$
\mathcal { L } ^ { ( l ) } = \frac { 1 } { B T J } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T } \Vert \hat { \mathbf { x } } _ { b , t } ^ { ( l ) } - \mathbf { x } _ { b , t } \Vert _ { 2 } ^ { 2 } .\tag{14}
$$

The overall retrospective objective is subsequently formulated as a weighted summation of these intermediate penalties:

$$
\mathcal { L } _ { \mathrm { r e t r o } } = \sum _ { l \in \{ 2 , 4 , 6 \} } \lambda _ { l } \mathcal { L } ^ { ( l ) } ,\tag{15}
$$

The coefficient $\lambda _ { l }$ modulates the intensity of the supervisory signals injected at different architectural depths. In execution, the reconstruction errors derived from the designated intermediate blocks are averaged prior to integration with the primary reconstruction loss originating from the final 8- th block. This hierarchical aggregation guarantees that the terminal network layer retains its strict priority in guiding the motion generation. Ultimately, this deep supervision strategy explicitly forces the internal feature representations to converge toward the target data distribution at an earlier stage, thereby substantially mitigating training volatility during motion editing.

## 3.6 Motion Preservation during Editing

Building upon the foundational principles of SimMotionEdit [9], we first evaluate frame-level kinematic similarities to explicitly isolate the temporal regions that necessitate structural preservation during the modification process. Assuming a source sequence $\mathbf { x } = \{ x _ { i } \} _ { i = 1 } ^ { T }$ and its corresponding synthesized counterpart $\mathbf { m } = \{ m _ { j } \} _ { j = 1 } ^ { T } ,$ this similarity metric is derived through a three-stage progressive pipeline.

Raw Similarity. Initially, we quantify the structural proximity across both rotational and positional spaces by employing a local sliding window of size W. The rotational discrepancy is mathematically defined as:

$$
S _ { i } ^ { R r } = - \operatorname* { m i n } _ { | i - j | \leq W } d _ { r } ( x _ { i } , m _ { j } ) ,\tag{16}
$$

where $d _ { r } ( \cdot )$ acts as a standard distance metric, such as the Euclidean norm, applied within the orientation manifold. Let $S _ { i } ^ { R r }$ and $S _ { i } ^ { R l }$ represent the computed frame-wise similarities for joint rotations and local positions, respectively. The unified raw similarity score is subsequently formulated via a weighted linear combination:

$$
S _ { i } ^ { R } = w _ { 1 } S _ { i } ^ { R r } + w _ { 2 } S _ { i } ^ { R l } .\tag{17}
$$

Normalized Similarity. To guarantee scale-invariant evaluations across diverse action categories, these raw metric values undergo a stringent normalization bounding them within the closed interval [0, 1]. This transformation yields a continuous temporal similarity profile, which effectively accentuates the frames exhibiting substantial divergence between the source and target behaviors while preserving uniform numerical magnitudes across all processed sequences.

Motion Signal-to-Noise Filtering. To suppress irrelevant fluctuations and distill robust supervisory signals, a bespoke Motion Signal-to-Noise Ratio is calculated. This metric is formulated as:

$$
\mathrm { M o t i o n S N R } = \frac { \sum _ { x \in T ^ { R } } x } { \sum _ { x \in B ^ { R } } x } ,\tag{18}
$$

Here, $T ^ { R }$ and $B ^ { R }$ denote the subsets containing the topκ and bottom-κ frames, respectively, as ranked by their temporal similarity scores. We selectively retain data pairs exhibiting elevated MotionSNR values, as they more accurately distinguish the mutable editing windows from the rigid background context. Adhering to the empirical configurations established by SimMotionEdit [9] on the MotionFix [8] dataset, we configure the sliding window parameter W alongside weighting coefficients $w _ { 1 } = 1 . 0$ and $w _ { 2 } = 1 . 0$ . Specifically for the STANCE Adjustment dataset, we set $\kappa = 5$ while establishing the minimum signal-tonoise threshold at 1.5.

Motion Preservation. A pronounced MotionSNR empirically indicates that the synthesized behavior remains structurally homologous to the source motion, implying that the textual instruction demands only localized and nuanced spatial adjustments. For these specific motion pairs surpassing the predefined threshold $\tau ,$ we enforce a dedicated preservation loss to penalize any unwarranted deviations from the original kinematic blueprint:

$$
\mathcal { L } _ { \mathrm { p r e s v } } = \mathbb { I } \big ( \mathbf { M o t i o n S N R } ( \mathbf { x } , \mathbf { m } ) > \tau \big ) \cdot \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \| m _ { i } - x _ { i } \| _ { 2 } ^ { 2 } .\tag{19}
$$

where I(·) serves as the binary indicator function, with T denoting the total sequence duration. This regularization objective strictly reinforces the structural integrity within high-confidence samples, thereby steering the network to concentrate its computational capacity exclusively on the semantic modifications dictated by the text while shielding the foundational motion architecture from degradation.

Overall, our framework establishes comprehensive supervision over the “change and invariance” of poses. The $\mathcal { L } _ { \mathrm { c l s } }$ directly follows the definition from SimMotionEdit [9]. The global loss function is thus expressed as:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { c l s } } { \mathcal { L } } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e t r o } } { \mathcal { L } } _ { \mathrm { r e t r o } } } \\ & { ~ + \lambda _ { \mathrm { p r e s e r v e } } { \mathcal { L } } _ { \mathrm { p r e s v } } + \lambda _ { \mathrm { t r i p l e t } } { \mathcal { L } } _ { \mathrm { t r i p l e t } } . } \end{array}\tag{20}
$$

The configurations of the loss weights and implementation details are detailed in Section 4.4.

## 4 EXPERIMENTS

In the following parts, we will first introduce the experimental datasets, evaluation metrics, competitive baselines, and implementation details in Sections 4.1, 4.2, 4.3, and 4.4, respectively. Then, we will present the quantitative analysis and qualitative results to demonstrate the superiority of our comprehensive framework in Sections 4.5 and 4.6. Next, we will provide extensive ablation studies to validate the effectiveness of our core components and distinct supervision strategies in Section 4.7. Finally, we will assess the subjective perceptual quality of our generated motions in Section 4.8, and explore the out-of-distribution robustness of our model in Section 4.9.

## 5 EXPERIMENTAL SETUPS

Dataset. To train and evaluate our framework, we utilize the MotionFix dataset [8], which supplies meticulously annotated triplets comprising a source behavior, a target behavior, and a linguistic instruction. The curation of this corpus relies on the TMR model [75] to mine structurally similar kinematic pairs from extensive MoCap archives, after which human annotators articulate the precise semantic deltas between them. In total, this collection furnishes 6, 730 distinct triplets distributed across the training, validation, and testing phases.

![](images/75bdfdc8882e9dafb24c839ece2fcada9a1e8c0527dbeffdbd16cf814341563a.jpg)  
Fig. 5: Qualitative comparison between our method and SimMotionEdit on the MotionFix dataset. Our results surpass SimMotionEdit in terms of semantic consistency, motion smoothness, and source motion preservation.

![](images/ce4efb89592abdcdc75e5f0bbee624f441c7133ca30650be64f9195b0279e162.jpg)  
Fig. 6: Qualitative comparison between our method and SimMotionEdit on the STANCE Adjustment dataset. Our results surpass SimMotionEdit in terms of semantic consistency, motion smoothness, and source motion preservation.

Additionally, we employ the STANCE Adjustment dataset [66], an auxiliary corpus constructed upon the generative foundations of the MLD architecture [11]. During its compilation, the latent space of the generator is systematically perturbed to yield 16 distinct kinematic variants for every base instruction. Consequently, any given pair drawn from these variants encapsulates a nuanced spatial or temporal transformation. Human experts subsequently transcribe these subtle modifications into natural language prompts, culminating in a supplementary repository of 4, 411 curated triplets.

Evaluation metrics. The fundamental imperative in textguided kinematic editing is to ensure strict consistency between the synthesized outputs and the ground-truth target behaviors. Adhering to the standardized evaluation pipeline established by MotionFix [8], we implement a motion-tomotion retrieval paradigm. In practice, this entails projecting the generated sequences into a comparable latent space utilizing the pre-trained TMR encoder [75], followed by calculating the retrieval precision within a constrained batch of candidates. To provide a comprehensive quantitative assessment, we report the Top-1, Top-2, and Top-3 retrieval accuracies (commonly denoted as R@1, R@2, and R@3) evaluated both within a batch size of 32 and across the entire testing distribution, alongside the average retrieval rank for in-depth benchmarking.

TABLE 1: Comparison of retrieval performance (Generated-to-Target) on MotionFix dataset [8]. In the following tables and figures, R@k indicates recall at top-k (↑ higher is better, ↓ lower is better). \* : no explicit text conditioning in the DiT stage.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Generated-to-Target (Batch)</td><td colspan="4">Generated-to-Target (Test Set)</td></tr><tr><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>Ground Truth</td><td></td><td>100</td><td>100</td><td>100</td><td>1.00</td><td>64.36</td><td>88.75</td><td>95.56</td><td>1.74</td></tr><tr><td>MDM [7]</td><td>ICLR&#x27;23</td><td>4.03</td><td>7.56</td><td>10.48</td><td>15.55</td><td>0.10</td><td>0.10</td><td>0.10</td><td></td></tr><tr><td>MDM-BP [8]</td><td>SIGGRAPH Asia&#x27;24</td><td>39.10</td><td>50.09</td><td>54.84</td><td>6.46</td><td>8.69</td><td>14.71</td><td>18.36</td><td>180.99</td></tr><tr><td>TMED [8]</td><td>SIGGRAPH Asia&#x27;24</td><td>62.90</td><td>76.51</td><td>83.06</td><td>2.71</td><td>14.51</td><td>21.72</td><td>28.73</td><td>56.63</td></tr><tr><td>MotionReFit [66]</td><td>CVPR&#x27;25</td><td>66.33</td><td>80.05</td><td>84.98</td><td>2.64</td><td></td><td></td><td></td><td></td></tr><tr><td>SimMotionEdit [9]</td><td>CVPR&#x27;25</td><td>70.62</td><td>82.92</td><td>88.12</td><td>2.38</td><td>25.49</td><td>39.33</td><td>49.21</td><td>23.49</td></tr><tr><td>SimMotionEdit * [9]</td><td>CVPR&#x27;25</td><td>71.04</td><td>83.96</td><td>89.58</td><td>2.22</td><td>26.88</td><td>44.27</td><td>51.98</td><td>20.88</td></tr><tr><td>Han et al. [61]</td><td>CVPR&#x27;26</td><td>74.38</td><td>88.54</td><td>92.08</td><td>1.92</td><td>29.45</td><td>45.26</td><td>54.55</td><td>16.42</td></tr><tr><td>OmniME [14]</td><td>CVPR&#x27;26</td><td>77.29</td><td>88.54</td><td>91.88</td><td>1.79</td><td>32.02</td><td>50.20</td><td>59.88</td><td>13.06</td></tr><tr><td>CIME(Ours)</td><td></td><td>78.12</td><td>88.54</td><td>92.50</td><td>1.73</td><td>33.40</td><td>50.29</td><td>59.88</td><td>12.24</td></tr></table>

TABLE 2: Comparison of retrieval performance (Generated-to-Target) on STANCE Adjustment dataset [66].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Generated-to-Target (Batch)</td><td colspan="4">Generated-to-Target (Test Set)</td></tr><tr><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>TMED [8]</td><td>SIGGRAPH Asia&#x27;24</td><td>29.69</td><td>44.01</td><td>52.08</td><td>6.97</td><td>11.22</td><td>17.86</td><td>25.51</td><td>35.56</td></tr><tr><td>TMED w/ MCM [8], [66]</td><td>CVPR&#x27;25</td><td>32.22</td><td>45.03</td><td>54.83</td><td>6.56</td><td></td><td></td><td></td><td></td></tr><tr><td>SimMotionEdit * [9]</td><td>CVPR&#x27;25</td><td>36.46</td><td>48.96</td><td>57.81</td><td>5.71</td><td>12.76</td><td>23.98</td><td>29.59</td><td>29.05</td></tr><tr><td>MotionReFit [66]</td><td>CVPR&#x27;25</td><td>42.45</td><td>56.25</td><td>62.76</td><td>5.12</td><td></td><td></td><td></td><td></td></tr><tr><td>OmniME [14]</td><td>CVPR&#x27;26</td><td>43.75</td><td>56.25</td><td>66.15</td><td>4.66</td><td>22.45</td><td>31.63</td><td>36.22</td><td>22.77</td></tr><tr><td>CIME</td><td></td><td>47.92</td><td>61.46</td><td>71.35</td><td>4.34</td><td>29.59</td><td>36.22</td><td>41.33</td><td>22.44</td></tr></table>

Baselines. To rigorously evaluate our framework, we benchmark its performance against six representative stateof-the-art models: MDM [7], MDM-BP [8], TMED [8], SimMotionEdit [9], MotionReFit [66], and a recent crossaxis fusion approach [61]. Each competitor embodies a distinct paradigm within the motion editing domain. MDM [7] demonstrates pure diffusion editing without observing source kinematics. MDM-BP [8] injects source constraints but lacks targeted instruction-guided optimization. TMED [8] provides a strong baseline by systematically integrating linguistic conditions and source behaviors. Among the latest advancements, SimMotionEdit [9] enforces semantic fidelity via similarity prediction mechanisms. MotionRe-Fit [66] boosts generation flexibility and temporal coherence using a dedicated coordinator and dynamic augmentation. Additionally, Han et al. [61] propose a cross-axis feature fusion architecture and employ joint-wise motion difference prediction as an auxiliary task to achieve precise, localized edits.

Implementation details. During the generative process, the diffusion trajectory spans 300 steps governed by a cosine noise scheduler. Aligning with established practices in conditional generation [8], [76], we enforce a two-way conditioning strategy, allocating a uniform guidance scale of 2.0 for both the textual and source motion inputs. The semantic features of the language instructions are extracted utilizing a pre-trained CLIP ViT-L/14 encoder [71]. Architecturally, the fusion module and the diffusion backbone are instantiated as Transformers comprising 4 and 8 layers, respectively. Both networks share a unified configuration of 8 attention heads mapped to a 512-dimensional latent space.

Network optimization is executed via the AdamW solver [77] configured with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 128. For the RNIMM module, the scaling coefficient utilized to inject the rhythm-aligned source features into the noise target is explicitly set to $\lambda _ { \mathrm { R N I M M } } = 0 . 0 5$ Across all empirical setups, the global loss coefficients are rigidly maintained at $\bar { \lambda } _ { \mathrm { r e t r o } } = 1 . 0$ and $\lambda _ { \mathrm { t r i p l e t } } = 0 . 0 1$ . Meanwhile, the motion preservation weight $\lambda _ { \mathrm { p r e s e r v e } }$ is configured to 0.2 when training on the MotionFix dataset [8], and adjusted to 0.1 for the STANCE Adjustment corpus [66]. The entire framework undergoes 1, 500 epochs of training on a single NVIDIA A6000 GPU. Computationally, this requires approximately 19 hours to converge on the MotionFix dataset and 11 hours on the STANCE Adjustment dataset.

## 5.1 Quantitative Analysis

We present quantitative comparisons in Tab. 1 and Tab. 2. Our model consistently performs better than all baselines in motion-editing alignment. As expected, MDM [7] exhibits the weakest performance, highlighting that text instructions alone are insufficient to produce source-aligned edits. Incorporating GPT-based identification of editable body parts, MDM-BP [8] shows notable improvements over MDM. TMED [8], which models the influence of text instructions on the source motion, further surpasses MDM-BP. MotionReFit [66] benefits from MCM to enrich training data, providing additional gains, while SimMotionEdit [9] leverages source-target similarity curves to design auxiliary loss functions that enhance alignment. Moreover, Han et al. [61] propose a cross-axis feature fusion architecture and employ joint-wise motion difference prediction as an auxiliary task to achieve precise, localized edits. Although Han et al. [61], OmniME [14], and our proposed CIME achieve an identical Batch R@2 score of 88.54, sample-level analysis reveals this is merely a numerical coincidence with nonidentical retrieved subsets, and CIME ultimately demonstrates superior robustness by maintaining higher rankings for non-hit samples.

TABLE 3: Ablation study on the MotionFix dataset [8]. Comparison of retrieval performance (Generated-to-Target).
<table><tr><td rowspan="2">#</td><td colspan="4">Components</td><td colspan="4">Generated-to-Target (Batch)</td><td colspan="4">Generated-to-Target (Test Set)</td></tr><tr><td>Lretro</td><td> ${ \mathcal { L } } _ { \mathrm { t r i p l e t } }$ </td><td></td><td>Lpres MRNIMM</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>71.04</td><td>83.96</td><td>89.58</td><td>2.22</td><td>26.88</td><td>44.27</td><td>51.98</td><td>20.88</td></tr><tr><td>2</td><td>√</td><td></td><td></td><td></td><td>72.71</td><td>85.62</td><td>88.75</td><td>2.09</td><td>30.63</td><td>45.26</td><td>54.35</td><td>18.62</td></tr><tr><td>3</td><td></td><td>V</td><td></td><td></td><td>73.54</td><td>85.00</td><td>88.54</td><td>2.04</td><td>28.26</td><td>42.49</td><td>51.78</td><td>17.99</td></tr><tr><td>4</td><td></td><td></td><td>√</td><td></td><td>75.62</td><td>87.50</td><td>90.42</td><td>1.88</td><td>30.24</td><td>45.65</td><td>56.92</td><td>15.58</td></tr><tr><td>5</td><td></td><td></td><td></td><td>√</td><td>73.54</td><td>87.92</td><td>91.04</td><td>1.88</td><td>29.64</td><td>44.27</td><td>55.34</td><td>15.71</td></tr><tr><td>6</td><td>√</td><td>L</td><td>√</td><td>√</td><td>78.12</td><td>88.54</td><td>92.50</td><td>1.73</td><td>33.40</td><td>50.29</td><td>59.88</td><td>12.24</td></tr></table>

TABLE 4: Analysis of Negative Text Sampling Strategies. The evaluation is conducted on the MotionFix dataset [8].
<table><tr><td rowspan="2">Method</td><td colspan="4">Generated-to-Target (Batch)</td><td colspan="4">Generated-to-Target (Test Set)</td></tr><tr><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>Baseline</td><td>71.04</td><td>83.96</td><td>89.58</td><td>2.22</td><td>26.88</td><td>44.27</td><td>51.98</td><td>20.88</td></tr><tr><td>infoNCE [71]</td><td>68.33</td><td>83.12</td><td>86.04</td><td>2.38</td><td>25.10</td><td>38.14</td><td>48.62</td><td>24.14</td></tr><tr><td>K-means</td><td>70.21</td><td>82.50</td><td>87.92</td><td>2.35</td><td>27.47</td><td>42.29</td><td>50.79</td><td>21.06</td></tr><tr><td>Triplet(Ours)</td><td>73.54</td><td>85.00</td><td>88.54</td><td>2.04</td><td>28.26</td><td>42.49</td><td>51.78</td><td>17.99</td></tr></table>

## 5.2 Qualitative Results

We present a visual comparison of our method against SimMotionEdit [9]. Fig. 5 and 6 show examples from the MotionFix [8] and STANCE Adjustment [66] datasets.

Fig. 5 shows examples from MotionFix [8]. In Example 1, although SimMotionEdit generates roughly correct movements, it introduces an unnatural self-rotation. In contrast, our method produces stable and realistic motions. In Example 2, the motion generated by SimMotionEdit is completely unrelated to the textual instruction, whereas our approach aligns perfectly with the target semantics. In Example 3, SimMotionEdit struggles with incorrect temporal rhythm and spatial amplitude, while our method ensures accurate physical dynamics and appropriate motion scales.

Fig. 6 shows examples from STANCE Adjustment [66]. In Example 1, SimMotionEdit exhibits an abnormal armraising behavior that is absent from both the source motion and the instruction, whereas our method maintains proper structural integrity. In Example 2, given the instruction “Both arms are opened”, SimMotionEdit produces an exaggerated amplitude leading to highly unnatural physical deformities. Our method, however, achieves a natural and anatomically consistent adjustment. Finally, in Example 3, SimMotionEdit fails to follow the textual instruction and merely reconstructs the source motion, while our method accurately applies the intended modifications.

TABLE 5: Comparison with Curriculum Scheduling based on Generated-to-Target (Batch).
<table><tr><td>Method</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>Curriculum Scheduling</td><td>72.92</td><td>84.79</td><td>90.21</td><td>2.07</td></tr><tr><td>Ours</td><td>73.54</td><td>85.00</td><td>88.54</td><td>2.04</td></tr></table>

Overall, our approach outperforms SimMotionEdit [9] in semantic understanding, physical plausibility, and motion fidelity, producing motions that are more natural, coherent, and faithful to textual instructions.

## 5.3 Ablation Study

Tab. 3 presents the results of our ablation study on the MotionFix [8] dataset. We evaluate the individual contributions of the retrospective loss $( \mathcal { L } _ { \mathrm { r e t r o } } )$ , triplet loss $( \mathcal { L } _ { \mathrm { t r i p l e t } } )$ , preservation loss $( \mathcal { L } _ { \mathrm { p r e s e r v e } } ) .$ , and the RNIMM module. As shown, removing any of these components leads to a consistent performance drop. Specifically, $\mathcal { L } _ { \mathrm { r e t r o } }$ improves structural stability through multi-level feature constraints. $\mathcal { L } _ { \mathrm { p r e s e r v e } }$ ensures the original kinematics of unedited frames are maintained. The RNIMM module effectively aligns temporal rhythms across varying sequence lengths. Finally, $\bar { \mathcal { L } } _ { \mathrm { t r i p l e t } }$ enforces strict semantic alignment between the generated motions and the textual instructions. The full model, which combines all these mechanisms, achieves the best overall performance. This demonstrates the effectiveness and complementarity of our proposed modules.

We further compare different strategies for constructing negative texts, with the results presented in Table 4. (1) The Triplet Loss (Random Negatives) randomly samples negative texts, enabling the model to learn from a broad and diverse range of semantic contrasts, thereby improving generalization. (2) The Cluster-based Triplet Loss employs

TABLE 6: Trained on MotionFix and tested on STANCE adjustment. † indicates our re-implementation.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Generated-to-Target (Batch)</td><td colspan="4">Generated-to-Target (Test Set)</td></tr><tr><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>AvgR↓</td></tr><tr><td>SimMotionEdit*† [9]</td><td>CVPR&#x27;25</td><td>21.43</td><td>33.04</td><td>42.41</td><td>8.80</td><td>7.42</td><td>11.35</td><td>14.41</td><td>58.67</td></tr><tr><td>OmniME [14]</td><td>CVPR&#x27;26</td><td>22.40</td><td>35.94</td><td>47.40</td><td>7.44</td><td>6.63</td><td>10.71</td><td>15.82</td><td>41.23</td></tr><tr><td>CIME (Ours)</td><td></td><td>29.69</td><td>43.23</td><td>54.69</td><td>6.36</td><td>10.20</td><td>18.37</td><td>23.47</td><td>34.74</td></tr></table>

![](images/14cac2445381edc91ec8cc6670f153d34c8026468d0edd03b626aa1430450b1e.jpg)

![](images/1b06a448bee595b6cf2e521570b4a14704fdccdeaf60a10c48924c6a7c2b2924.jpg)  
Fig. 7: Subjective evaluation: CIME vs SimMotionEdit [9]. The charts report average human ratings (scale 1– 10) across four perceptual metrics: Semantic Alignment, Motion Preservation, Transition Smoothness, and Overall Naturalness. Results for the MotionFix [8] and STANCE Adjustment [66] datasets are displayed on the left and right, respectively. Our proposed CIME consistently outperforms the baseline across all metrics on both datasets.

K-means to partition all texts into three clusters, selecting negative samples from the same cluster as the positive text. Although this design aims to capture fine-grained distinctions within similar semantics, it limits the model’s exposure to diverse negative samples, leading to insufficient generalization. (3) The InfoNCE Loss [71], following the CLIP paradigm, treats all other samples within a batch as negatives. However, since motion editing instructions often repeat across samples, semantically identical instructions may appear as false negatives, confusing the contrastive objective and degrading performance.

Furthermore, we investigate a Curriculum Scheduling approach, which dynamically transitions from easy (random) negatives to hard (in-cluster) negatives during the training process. Intuitively, curriculum learning should refine semantic boundaries; however, our empirical results (Tab. 5) demonstrate that purely random negative sampling performs better and exhibits greater stability. The underlying reason is that, in the context of motion editing, incluster “hard” negatives are often structurally ambiguous. They typically share highly overlapping kinematic trajectories with the positive samples but differ only in nuanced semantics. Forcing the model to strictly repel these structurally similar samples in later training stages introduces conflicting supervision, which disrupts the continuous motion manifold and destabilizes convergence. Consequently, the random-negative triplet loss achieves the most optimal and stable balance between diversity and semantic discrimination.

## 5.4 Perceptual Study

To assess the perceptual quality of our approach compared to SimMotionEdit [9], we organized a subjective evaluation. The study involved 25 volunteers, comprising academic professors, graduate students, and industry professionals.

We selected 20 representative editing cases in total. Half of these were sourced from the MotionFix [8] dataset, and the remaining half from the STANCE Adjustment [66] dataset. During the test, participants reviewed the original kinematics, textual prompts, ground-truth targets, and the synthesized outputs. For a fair and unbiased comparison, the evaluated models were blindly labeled as Method A and Method B.

Evaluators scored the generated sequences from 1 (worst) to 10 (best) based on four criteria: Semantic Alignment, Motion Preservation, Transition Smoothness, and Overall Naturalness. As illustrated in Fig. 7, CIME achieved consistently higher scores than SimMotionEdit [9] across all evaluated dimensions. It demonstrates superior textual responsiveness and structural integrity. These subjective results validate the effectiveness of our design philosophy of change and invariance.

## 5.5 Out-of-Distribution Robustness

Although achieving optimal cross-dataset generalization typically necessitates dedicated Domain Adaptation techniques—a distinct and expansive research area designated for our future exploration—we nonetheless conduct cross-dataset benchmarking to validate the intrinsic robustness of our architecture. Specifically, the evaluated models undergo training exclusively on the MotionFix [8] corpus and are subsequently deployed directly onto the unseen STANCE Adjustment [66] dataset for zero-shot testing. As illustrated in Tab. 6, even in the complete absence of domainspecific fine-tuning, our proposed CIME framework consistently and significantly surpasses both SimMotionEdit [9] and the prior OmniME [14] architecture across all evaluated retrieval metrics. This substantial performance leap empirically confirms the formidable generalization capacity of our approach. Furthermore, it emphasizes that the delicate equilibrium between structural invariance and semantic modification established by CIME is not merely overfitted to the idiosyncratic distributions of a specific training dataset, but rather profoundly captures the fundamental principles and universal physical laws governing text-driven motion editing.

## 6 CONCLUSION

In this paper, we propose CIME, a unified framework for text-driven 3D human motion editing. CIME effectively balances text-responsive modifications with the preservation of original motion structures. We achieve this by decoupling the editing process into spatial and temporal dimensions. For spatial poses, we introduce an omni-supervised learning mechanism to maintain structural accuracy. For temporal rhythms, we present the RNIMM module to align varying sequence lengths while preserving intrinsic physical beats. Experiments confirm that CIME achieves state-of-the-art performance across multiple benchmarks.

Despite these advancements, our work has certain limitations. Achieving optimal generalization across entirely different motion datasets remains a challenge. Specifically, our framework currently lacks cross-dataset generalization capabilities. In future work, we plan to explore dedicated Domain Adaptation techniques. This future research direction will further enhance the model’s cross-dataset robustness and expand its generalizable applicability.

## REFERENCES

[1] K. Lyu, Z. Liu, S. Wu, H. Chen, X. Zhang, and Y. Yin, “Learning human motion prediction via stochastic differential equations,” in Proceedings of the 29th ACM International Conference on Multimedia, 2021, pp. 4976–4984.

[2] K. Lyu, H. Chen, Z. Liu, B. Zhang, and R. Wang, “3d human motion prediction: A survey,” Neurocomputing, vol. 489, pp. 345– 365, 2022.

[3] S. Aliakbarian, F. S. Saleh, M. Salzmann, L. Petersson, and S. Gould, “A stochastic conditioning scheme for diverse human motion prediction,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5223–5232.

[4] X. Yan, A. Rastogi, R. Villegas, K. Sunkavalli, E. Shechtman, S. Hadap, E. Yumer, and H. Lee, “Mt-vae: Learning motion transformations to generate multimodal human dynamics,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 265–281.

[5] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[6] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning. pmlr, 2015, pp. 2256–2265.

[7] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-or, and A. H. Bermano, “Human motion diffusion model,” in The Eleventh International Conference on Learning Representations, 2023.

[8] N. Athanasiou, A. Cseke, M. Diomataris, M. J. Black, and G. Varol, “Motionfix: Text-driven 3d human motion editing,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1–11.

[9] Z. Li, K. Cheng, A. Ghosh, U. Bhattacharya, L. Gui, and A. Bera, “Simmotionedit: Text-based human motion editing with motion similarity prediction,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27 827–27 837.

[10] J. Kim, J. Kim, and S. Choi, “Flame: Free-form language-based motion synthesis & editing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, 2023, pp. 8255–8263.

[11] X. Chen, B. Jiang, W. Liu, Z. Huang, B. Fu, T. Chen, and G. Yu, “Executing your commands via motion diffusion in latent space,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 000–18 010.

[12] M. Petrovich, M. J. Black, and G. Varol, “Temos: Generating diverse human motions from textual descriptions,” in European Conference on Computer Vision. Springer, 2022, pp. 480–497.

[13] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “Motiondiffuse: Text-driven human motion generation with diffusion model,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 6, pp. 4115–4128, 2024.

[14] Z. Shi, J. Gong, P. Wang, X. Wang, T. Qian, W. Li, Y. Fang, J. Xie, L. Ma, and S. Lin, “Omni-supervised motion editing: Balancing change and invariance through positive-negative learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 30 595–30 606.

[15] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3d human motions from text,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5152–5161.

[16] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “Amass: Archive of motion capture as surface shapes,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5442–5451.

[17] A. R. Punnakkal, A. Chandrasekaran, N. Athanasiou, A. Quiros-Ramirez, and M. J. Black, “Babel: Bodies, action and behavior with english labels,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 722–731.

[18] A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, “Ntu rgb+ d: A large scale dataset for 3d human activity analysis,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 1010–1019.

[19] X. Bie, W. Guo, S. Leglaive, L. Girin, F. Moreno-Noguer, and X. Alameda-Pineda, “Hit-dvae: Human motion generation via hierarchical transformer dynamical vae,” arXiv preprint arXiv:2204.01565, 2022.

[20] F. G. Harvey, M. Yurick, D. Nowrouzezahrai, and C. Pal, “Robust motion in-betweening,” ACM Transactions on Graphics (TOG), vol. 39, no. 4, pp. 60–1, 2020.

[21] F. Hong, M. Zhang, L. Pan, Z. Cai, L. Yang, and Z. Liu, “Avatarclip: zero-shot text-driven generation and animation of 3d avatars,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1–19, 2022.

[22] X. Lin and M. R. Amer, “Human motion modeling using dvgans,” arXiv preprint arXiv:1804.10652, 2018.

[23] G. Tevet, B. Gordon, A. Hertz, A. H. Bermano, and D. Cohen-Or, “Motionclip: Exposing human motion generation to clip space,” in European Conference on Computer Vision. Springer, 2022, pp. 358–374.

[24] Z. Wang, P. Yu, Y. Zhao, R. Zhang, Y. Zhou, J. Yuan, and C. Chen, “Learning diverse stochastic human-action generators by learning smooth latent transitions,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 12 281–12 288.

[25] C. Zhong, L. Hu, Z. Zhang, and S. Xia, “Attt2m: Text-driven human motion generation with multi-perspective attention mechanism,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 509–519.

[26] K. Fan, J. Zhang, R. Yi, J. Gong, Y. Wang, Y. Wang, X. Tan, C. Wang, and L. Ma, “Textual decomposition then sub-motionspace scattering for open-vocabulary motion generation,” arXiv preprint arXiv:2411.04079, 2024.

[27] R. Dabral, M. H. Mughal, V. Golyanik, and C. Theobalt, “Mofusion: A framework for denoising-diffusion-based motion synthesis,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 9760–9770.

[28] W. Dai, L.-H. Chen, J. Wang, J. Liu, B. Dai, and Y. Tang, “Motionlcm: Real-time controllable motion generation via latent consistency model,” in European Conference on Computer Vision. Springer, 2024, pp. 390–408.

[29] H. Liang, J. Bao, R. Zhang, S. Ren, Y. Xu, S. Yang, X. Chen, J. Yu, and L. Xu, “Omg: Towards open-vocabulary motion generation via mixture of controllers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 482–493.

[30] H. Liang, W. Zhang, W. Li, J. Yu, and L. Xu, “Intergen: Diffusionbased multi-human motion generation under complex interactions,” International Journal of Computer Vision, vol. 132, no. 9, pp. 3463–3483, 2024.

[31] J. Ren, M. Zhang, C. Yu, X. Ma, L. Pan, and Z. Liu, “Insactor: Instruction-driven physics-based characters,” Advances in Neural Information Processing Systems, vol. 36, pp. 59 911–59 923, 2023.

[32] S. Xu, Z. Li, Y.-X. Wang, and L.-Y. Gui, “Interdiff: Generating 3d human-object interactions with physics-informed diffusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14 928–14 940.

[33] M. Zhang, X. Guo, L. Pan, Z. Cai, F. Hong, H. Li, L. Yang, and Z. Liu, “Remodiffuse: Retrieval-augmented motion diffusion model,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 364–373.

[34] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, “Motion-x: A large-scale 3d expressive whole-body human motion dataset,” Advances in Neural Information Processing Systems, vol. 36, pp. 25 268–25 280, 2023.

[35] Y. Shafir, G. Tevet, R. Kapon, and A. H. Bermano, “Human motion diffusion as a generative prior,” in The Twelfth International Conference on Learning Representations, 2024.

[36] W. Wan, Y. Huang, S. Wu, T. Komura, W. Wang, D. Jayaraman, and L. Liu, “Diffusionphase: Motion diffusion in frequency domain,” arXiv preprint arXiv:2312.04036, 2023.

[37] Y. Xie, V. Jampani, L. Zhong, D. Sun, and H. Jiang, “Omnicontrol: Control any joint at any time for human motion generation,” arXiv preprint arXiv:2310.08580, 2023.

[38] Y. Yang, Z. Huang, C. Xu, and S. He, “Lagrangian motion fields for long-term motion generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[39] L.-H. Chen, S. Lu, A. Zeng, H. Zhang, B. Wang, R. Zhang, and L. Zhang, “Motionllm: Understanding human behaviors from human motions and videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[40] M. Zhang, H. Li, Z. Cai, J. Ren, L. Yang, and Z. Liu, “Finemogen: Fine-grained spatio-temporal motion generation and editing,” Advances in Neural Information Processing Systems, vol. 36, pp. 13 981– 13 992, 2023.

[41] R. Wang, C. Ma, G. Li, H. Xu, Y. Li, and Z. Wang, “You think, you act: The new task of arbitrary text to motion generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 12 012–12 022.

[42] M. Gleicher, “Motion editing with spacetime constraints,” in Proceedings of the 1997 symposium on Interactive 3D graphics, 1997, pp. 139–ff.

[43] J. Lee and S. Y. Shin, “A hierarchical approach to interactive motion editing for human-like figures,” in Proceedings of the 26th annual conference on Computer graphics and interactive techniques, 1999, pp. 39–48.

[44] M. Gleicher, “Motion path editing,” in Proceedings of the 2001 symposium on Interactive 3D graphics, 2001, pp. 195–202.

[45] M. Kim, K. Hyun, J. Kim, and J. Lee, “Synchronized multicharacter motion editing,” ACM transactions on graphics (TOG), vol. 28, no. 3, pp. 1–9, 2009.

[46] K. Aberman, P. Li, D. Lischinski, O. Sorkine-Hornung, D. Cohen-Or, and B. Chen, “Skeleton-aware networks for deep motion retargeting,” ACM Transactions on Graphics (TOG), vol. 39, no. 4, pp. 62–1, 2020.

[47] M. Unuma, K. Anjyo, and R. Takeuchi, “Fourier principles for emotion-based human figure animation,” in Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, 1995, pp. 91–96.

[48] W. Yin, H. Yin, K. Baraka, D. Kragic, and M. Bjorkman, “Dance¨ style transfer with cross-modal transformer,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 5058–5067.

[49] B. N. Oreshkin, F. Bocquelet, F. G. Harvey, B. Raitt, and D. Laflamme, “Protores: Proto-residual network for pose authoring via learned inverse kinematics,” in International Conference on Learning Representations, 2022.

[50] J. Qin, Y. Zheng, and K. Zhou, “Motion in-betweening via twostage transformers.” ACM Trans. Graph., vol. 41, no. 6, pp. 184–1, 2022.

[51] J. Tseng, R. Castellon, and K. Liu, “Edge: Editable dance generation from music,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 448–458.

[52] S. Raab, I. Gat, N. Sala, G. Tevet, R. Shalev-Arkushin, O. Fried, A. H. Bermano, and D. Cohen-Or, “Monkey see, monkey do: Harnessing self-attention in motion diffusion for zero-shot motion transfer,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1–13.

[53] E. Pinyoanuntapong, P. Wang, M. Lee, and C. Chen, “Mmm: Generative masked motion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1546–1555.

[54] N. Athanasiou, M. Petrovich, M. J. Black, and G. Varol, “Teach: Temporal action composition for 3d humans,” in 2022 International Conference on 3D Vision (3DV). IEEE, 2022, pp. 414–423.

[55] Y. Shi, J. Wang, X. Jiang, B. Lin, B. Dai, and X. B. Peng, “Interactive character control with auto-regressive motion diffusion models,” ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1–14, 2024.

[56] N. Athanasiou, M. Petrovich, M. J. Black, and G. Varol, “Sinc: Spatial composition of 3d human motions for simultaneous action generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9984–9995.

[57] M. Petrovich, O. Litany, U. Iqbal, M. J. Black, G. Varol, X. Bin Peng, and D. Rempe, “Multi-track timeline control for text-driven 3d human motion generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024, pp. 1911–1921.

[58] G. Delmas, P. Weinzaepfel, F. Moreno-Noguer, and G. Rogez, “Posefix: correcting 3d human poses with natural language,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15 018–15 028.

[59] Y. Huang, W. Wan, Y. Yang, C. Callison-Burch, M. Yatskar, and L. Liu, “Como: Controllable motion generation through language guided pose code editing,” in European Conference on Computer Vision. Springer, 2024, pp. 180–196.

[60] P. Goel, K.-C. Wang, C. K. Liu, and K. Fatahalian, “Iterative motion editing with natural language,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–9.

[61] G. Han and J. Kim, “Cross-axis feature fusion with joint-wise motion difference prediction for text-based 3d human motion editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 30 618–30 628.

[62] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for realtime style transfer and super-resolution,” in European conference on computer vision. Springer, 2016, pp. 694–711.

[63] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” Advances in neural information processing systems, vol. 29, 2016.

[64] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 815–823.

[65] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597– 1607.

[66] N. Jiang, H. Li, Z. Yuan, Z. He, Y. Chen, T. Liu, Y. Zhu, and S. Huang, “Dynamic motion blending for versatile motion editing,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22 735–22 745.

[67] Z. Zhang, Y. Wang, D. Li, D. Gong, I. Reid, and R. Hartley, “Flashmo: Geometric interpolants and frequency-aware sparsity for scalable efficient motion generation,” Advances in Neural Information Processing Systems, vol. 38, pp. 33 602–33 632, 2026.

[68] Z. Yu, S. Foti, L. Zhang, A. Zhao, C. Keskin, S. Zafeiriou, T. Birdal et al., “Geometric neural distance fields for learning human motion priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 2232–2242.

[69] F. Miao, J. Huang, and T. Li, “Riemannian motion generation: A unified framework for human motion representation and generation via riemannian flow matching,” arXiv preprint arXiv:2603.15016, 2026.

[70] Y. Wang, S. Wang, J. Zhang, K. Fan, J. Wu, Z. Xue, and Y. Liu, “Timotion: Temporal and interactive framework for efficient humanhuman motion generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7169–7178.

[71] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[72] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[73] X. Tan, Q. Ma, J. Gong, J. Xu, Z. Zhang, H. Song, Y. Qu, Y. Xie, and L. Ma, “Positive-negative receptive field reasoning for omnisupervised 3d segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 15 328–15 344, 2023.

[74] P. Xiang, X. Wen, Y.-S. Liu, H. Zhang, Y. Fang, and Z. Han, “Retrofpn: Retrospective feature pyramid network for point cloud semantic segmentation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 17 826–17 838.

[75] M. Petrovich, M. J. Black, and G. Varol, “Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9488–9497.

[76] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 392–18 402.

[77] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

![](images/4eb4cea967e4931bae71c862baef9f16a64d7d884f91ba4c0b02a078f80df8fb.jpg)

Shaohui Lin received Ph.D. degree from Xiamen University, Xiamen, China, in 2019. He is currently a Research Professor at East China Normal University. He is the author of about 40 scientific articles at top venues, including IEEE TPAMI, TIP, CVPR, ICML, and ICLR, etc. He serves as a Area Chair at top-tier conferences, such as CVPR, ICML and ICLR, as well as a reviewer for IEEE TPAMI, IJCV, TMM and CVPR, etc. His research interests include machine learning and computer vision.

![](images/1e627f9762f9db48ba18d24850a287776baff5e6c59ca1bffd055a0eb49be2d8.jpg)

Zhenwu Shi received the M.S. degree in Educational Information Technology from the National Engineering Research Center for E-Learning, Central China Normal University, in 2020. He is currently pursuing the Ph.D. degree in Intelligent Education at the Shanghai Institute of Intelligent Education, East China Normal University, Shanghai, China. His research interests include computer vision, human pose estimation, motion editing, and AI in education.

![](images/344e40dbd6d795bea788489754d7168c23e46d7aa0d40f03b864acee30cb67d7.jpg)

Jingyu Gong received his Ph.D. from the Department of Computer Science and Technologies, Shanghai Jiao Tong University. He is now an Associate Research Professor with the School of Computer Science and Technology, East China Normal University, China. His research interests cover image processing and computer vision. He has published 15 papers in major international journals and conferences including the IEEE TPAMI, IEEE TVCG, CVPR, ECCV, AAAI, IJCAI, ACM MM, etc.

![](images/2ef5d6e88bcd37812adb4c1d8f12da63b6bed50a6076d51a5dfe04a135a4ea7a.jpg)

Jiao Xie received a Ph.D. degree from Xiamen University, Fujian, China, in 2023. She is currently a PostDoc. at East China Normal University. Her research interests include Computer Vision and Deep Learning.

![](images/1a82824582b9f86bf1bdaef64566a7347716f62968c7b8daa4c01b10ac79c3be.jpg)

Zhou Yu , School of Statistics, Key Laboratory of Advanced Theory and Application in Statistics and Data Science, Ministry of Education (KLATASDS-MOE), East China Normal University, Shanghai, China. Zhou Yu received the B.Sc. degree from the Hefei University of Technology in 2004 and the Ph.D. degree from East China Normal University, China, in 2010. He is currently a Professor and the Ph.D. Supervisor with East China Normal University, and the Dean of the School of Statistics, ECNU. He has published more than 40 articles in well-known statistics journals, such as Annals of Statistics, Biometrika, and Journal of the American Statistical Association. His current research interests include statistical analysis of high-dimensional data and statistical machine learning.

![](images/3bfc288bf08c8c97e577acc322f215adaa60f6ae5ef01017cfa7617f92e49c77.jpg)

Baochang Zhang received his Ph.D. degree from Harbin Institute of Technology. He is a professor at Beihang University, and Zhongguancun National Laboratory and an academic advisor at Baidu’s Deep Learning Laboratory. His research interests include model compression and acceleration, as well as efficient object recognition.

![](images/24fd66f6884794c07bb5929479758eda7675938f701d676957d65cb74beaa6cb.jpg)

Lizhuang Ma received his B.S. and Ph.D. degrees from Zhejiang University, China, in 1985 and 1991, respectively. He is now a Distinguished Professor, at the Department of Computer Science and Engineering, Shanghai Jiao Tong University, China, and the School of Computer Science and Technology, East China Normal University, China. He was a Visiting Professor at the Frounhofer IGD, Darmstadt, Germany in 1998, and a Visiting Professor at the Center for Advanced Media Technology, Nanyang

Technological University, Singapore from 1999 to 2000. His research interests include computer vision, computer aided geometric design, computer graphics, scientific data visualization, computer animation, digital media technology, and theory and applications for computer graphics, CAD/CAM. He serves as the reviewer of IEEE TPAMI, IEEE TIP, IEEE TMM, CVPR, AAAI etc.

![](images/0ca8ce775cd58a007387f7709a1227fe98982112ea20c11a5eee100f85f9e725.jpg)

Chia-Wen Lin received his Ph.D. degree from the Department of Electrical Engineering, National Tsing Hua University (EE/NTHU), Hsinchu, Taiwan, in January 2000. He is a professor with EE/NTHU since August 2007. Prior to joining EE/NTHU in 2007, he worked for the Department of Computer Science and Information Engineering, National Chung Cheng University (CSIE/CCU), Chiayi, Taiwan from August 2000 to July 2007. He was with the Information and Communications Research Laboratories, Industrial Technology Research Institute (ICL/ITRI), Hsinchu, Taiwan, during 1992 2000, where his final post was Section Manager. He served as a visiting scholar at the Information Processing Laboratory, Department of Electrical Engineering, University of Washington, USA during April August 2000, a visiting professor with Microsoft Research Asia, Beijing, China, during July-August 2002, with National Institute of Informatics, Tokyo, Japan from August 2015 to February 2016, and with Nagoya University during June December 2019. His research interests include visual content analysis and processing, and video networking.

Dr. Lin was named IEEE Fellow for his contributions to multimedia coding and editing in 2018. He is serving on the Board-of-Governors (2022 2024) and Fellow Evaluation Committee (2021 2023) of IEEE Circuits and Systems Society (CASS). He was a Distinguished Lecturer of IEEE CASS (2018 2019). He also served as President of the Chinese Image Processing and Pattern Recognition Association, Taiwan (2019 2020). His paper won the Young Investigator Award of SPIE VCIP 2005 and Best Paper Award of IEEE VCIP 2015. He was a recipient of the Distinguished Research Award (2023) and the Ta-You Wu Memorial Award (2006) both presented by National Science & Technology Council (NSTC), Taiwan. He has served on the editorial board of IEEE Transactions on Image Processing, IEEE Transactions on Circuits and Systems for Video Technology, IEEE Transactions on Multimedia, IEEE Multimedia Magazine, Journal of Visual Communication and Image Representation, and Signal Processing: Image Communication. He also served as a Guest Editor of five special issues for IEEE Journal of Selected Topics in Signal Processing, IEEE Transactions on Multimedia, EURASIP Journal on Advances in Signal Processing, and Journal of Visual Communication and Image Representation, respectively. He was Chair of the Multimedia Systems and Applications Technical Committee of IEEE Circuits and Systems Society (2013 2015). He served as Steering Committee Chair of IEEE ICME (2020 2021), TPC Co-Chair of IEEE ICIP 2019 and ICME 2010, General Co-Chair of IEEE VCIP 2018 and PCS 2024, and Special Session Co-Chair of IEEE ICME 2009 and ICME 2018.