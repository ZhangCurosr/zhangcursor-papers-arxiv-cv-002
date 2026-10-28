# LociArt: A Unified Paradigm for Free-Style Local Video Stylization

Yuxuan Chen<sup>1∗</sup> Feng Xie<sup>2∗</sup> Jiagao Hu<sup>1∗B</sup> Fuhao Li<sup>1</sup> Zepeng Wang<sup>1</sup> Fei Wang<sup>1</sup> Daiguo Zhou<sup>1</sup>

<sup>1</sup> MiLM Plus, Xiaomi Inc. <sup>2</sup> Xi’an University of Electronic Science and Technology <sup>\*</sup>The first three authors contributed equally.

![](images/5e28402ee1abf02a65f211447d1ecdba086e4fdf0d04894d3983df2684a842ed.jpg)  
Figure 1 Examples of free-style local video stylization. LociArt supports local or global editing, shared or distinct styles across multiple regions, and style composition without compositional training pairs. Small images show the source videos and masks; white and black denote editing and preservation, respectively. Color masks identify styles.

## Abstract

Video stylization typically transforms entire scenes, while local artistic editing requires precise region–style association, spatial control, and temporal consistency. We introduce LociArt, a unified paradigm for free-style local video stylization that explicitly specifies where to stylize and what style to apply while preserving non-target content. At its core, GRNStyle builds on Generative Refinement Networks (GRN) and introduces a polar style-aware mask that embeds style identities as equal-norm directions within selected regions. Complementary representation–geometry conditioning combines encoded mask features with raw-mask spatial support, integrating spatiotemporal context with geometric control. To provide paired supervision, we develop an automated pipeline combining global stylization, video object removal, matting, and compositing, and construct the LociArt-100K Dataset with 100K local stylization pairs. We further establish the LociArt Benchmark, comprising 80 source videos and 84 target masks covering 168 foreground/background stylization cases, with a Style Prototype Similarity (SPS) evaluation scheme for regional style quality and complementary measures of non-target preserva-

tion and temporal consistency. Experiments show that GRNStyle outperforms existing methods in regional style quality and spatial control while maintaining strong temporal consistency. Qualitative examples further illustrate joint stylization of multiple regions with the same or diferent learned styles, despite training only on single-region, single-style pairs. Together, these components provide a unified foundation for learning and evaluating controllable local video stylization. Project page: https://higherhu.github.io/lociart/.

## 1 Introduction

Video stylization transforms recorded footage into artistic renditions for film, advertising, and digital art. Building on image style transfer [1, 2] and difusion-based image generation and editing [3, 4], video methods use text or reference images to guide appearance while preserving motion [5–7]. Yet artistic control often requires more than a global transformation: an artist may turn a moving subject into a sketch within an otherwise unchanged scene, reimagine the surroundings as a watercolor while keeping the subject intact, or bring distinct artistic styles to diferent objects in the same shot, as shown in Fig. 1. We call this task free-style local video stylization: controlling both where to stylize and what style to apply across space and time, while preserving non-target content.

Video segmentation provides a practical starting point. A few user interactions can define masks that propagate across frames, with streaming inference supporting real-time processing [8, 9]. Such masks ofer an accessible spatial prior for local stylization [10, 11], extending control from foreground subjects to the less-explored task of stylizing their surroundings. Yet selecting a region does not specify its artistic transformation: a binary mask indicates editing extent but not style identity. With style supplied separately through text, the model must learn to associate the intended appearance with the selected region.

Reliable local control also requires preserving region boundaries throughout motion. Despite advances in attention supervision [12, 13] and mask-conditioned editing [14–17], mask representations face a trade-of between spatiotemporal context and spatial fidelity. VAE-based processing aligns masks with learned video features [18], but compression can blur their geometry; direct resampling retains spatial structure without the same learned temporal context [19]. Thus, a useful conditioning mechanism must both bind styles explicitly to regions and retain the spatial detail needed to localize their efects over time.

These modeling challenges are compounded by gaps in supervision and evaluation. Training pairs must demonstrate local artistic changes, faithful preservation elsewhere, and natural transitions along moving boundaries. Global stylization pairs [7, 20] do not directly supply this supervision, while instruction-based video-editing datasets [13, 21] cover broader editing operations. Likewise, global similarity and video-averaged scores can conceal weak regional style, changes outside the target, and transient failures. Free-style local video stylization therefore calls for coordinated advances in paired data, region–style conditioning, and region-aware evaluation.

We introduce LociArt, a unified paradigm addressing these needs through the LociArt-100K Dataset, the GRNStyle model, and the LociArt Benchmark. Our automated data pipeline combines global stylization, video object removal, matting, and compositing to construct 100K curated pairs. Each example contains a source video, an editing mask, a style instruction, and a locally stylized target, supervising transformation and preservation together. Constructing these targets requires handling contour changes introduced by stylization. Object removal supplies clean background content for newly exposed areas, while alpha mattes estimated separately for original and stylized objects guide compositing along their respective contours, avoiding object remnants and enabling natural boundary transitions.

Trained on these pairs, GRNStyle builds on Generative Refinement Networks (GRN) [22] and the source-evidence editing formulation of GRNEdit [23]. Its core design is a polar style-aware mask, which embeds style identities as equal-norm directions within selected regions, unifying where to stylize and what style to apply without imposing a scalar ordering on styles. Complementary representation–geometry conditioning combines VAE-encoded mask features with raw-mask spatial support to integrate spatiotemporal context and geometric control, while source evidence guides content preservation throughout refinement.

To evaluate local stylization, the LociArt Benchmark contains 80 source videos and 84 object– mask units, each evaluated under both foreground and background stylization protocols, yielding 168 stylization cases. It uses Style Prototype Similarity (SPS) to measure regional style fidelity, together with complementary measures of non-target preservation and temporal consistency.

Our contributions are threefold:

• A Unified Paradigm for Local Video Stylization. We introduce LociArt, a unified computational paradigm that formulates local video stylization as region-conditioned artistic transformation, explicitly specifying where to stylize and what style to apply while preserving non-target content and temporal consistency.

• GRNStyle with Style-Aware Region Control. We propose GRNStyle, a GRN-based model that unifies region and style through polar style-aware masks and combines spatiotemporal context with geometric control. Qualitative examples further illustrate free-style multi-region, multi-style compositions learned from single-region, single-style supervision.

• Scalable Data Construction and Region-Aware Evaluation. We develop an automated pipeline to construct the LociArt-100K Dataset and establish the LociArt Benchmark with 80 source videos and 84 object–mask units, integrating SPS with preservation and temporal consistency measures for region-aware evaluation.

## 2 Related Work

Image style transfer. Neural style transfer progressed from per-image optimization [1] to feed-forward arbitrary transfer with AdaIN [2]. Building on latent difusion [3], StyleShot [24] separates style from content, while TeleStyle [25] learns content-preserving transfer from curated triplets and extends it to video. Localized editing, however, requires explicit spatial control.

Localized stylization. Image methods use masked feature or attention control [16], masklearned regional attention [12], or semantic-region matching [11]. For video, Xia et al.[26] use spatiotemporal alignment, while UniVST [10] combines mask propagation, AdaIN-guided latent and attention manipulation, and optical-flow smoothing without training. In contrast, we study locally artistic video stylization with paired supervision and fixed, equal-norm polar codes that bind style identity to editable regions.

Video stylization and generative editing. PickStyle [27] learns lightweight style adapters from synthetic clips derived from paired images. VISTA [7] and MoStyle [28] use style/source/stylizedvideo triplets; DreamStyle [20] supports text-, image-, and first-frame-guided stylization from paired videos. For broader editing, VACE [14] unifies video conditions including masks, while VideoPainter [15] performs context-conditioned inpainting. Beyond difusion, GRN [22] uses binary-state refinement, and GRNEdit [23] treats the source as editing evidence. GRNStyle extends this evidence-based formulation by combining VAE-encoded mask features with raw-mask geometry, thereby coupling temporal semantics with spatial fidelity.

![](images/02f2426b8543b79c0869cd59f6ada9a8df5116470a8e7460c102e7e7ffd3568f.jpg)  
Figure 2 Overview of the data construction pipeline. Source and object-removed videos are globally stylized and filtered, then combined through mask-guided matting to produce foregroundand background-stylized pairs for human verification.

Data and evaluation. Ditto-1M [21] scales instruction-based editing, while CoinVE-200K and CoinVE-Bench [13] target compositional editing, including regional stylization. StyleShot [24] and MegaStyle [29] provide image data for stylistic consistency and diversity; StyleShot also introduces StyleBench. Evaluation spans image quality with ArtFID [30], style similarity with CSD [31], and video quality and temporal consistency with VBench [32]; RegionRoute [12] also evaluates regional style and background preservation. LociArt-100K adds pairs aligning content, regions, styles, and targets; its benchmark measures regional style, non-target preservation, temporal consistency, and VLM-based perceptual quality.

## 3 LociArt-100K Dataset Construction

To supervise local stylization and non-target preservation, we construct the LociArt-100K Dataset of 100K pairs by separating global style generation from spatial composition. Specifically, we globally stylize source and object-removed videos, then recombine their components into complementary foreground and background targets. Each example contains a source video $\mathbf { V } _ { \mathrm { s r c } }$ , binary editing mask M, style instruction �, and locally stylized target $\mathbf { V } ^ { \mathrm { s t y } }$ . As illustrated in Fig. 2, the pipeline comprises the following four steps.

Step 1: Object identification and removal. We first identify the objects to separate from their surroundings. Source videos are collected from SA-V [8] and Ditto-1M [21]. Qwen3-VL-32B [33] identifies salient objects, and their category prompts guide SAM3 [9] to produce temporally tracked masks $\mathbf { M } _ { \mathrm { o b j } }$ . After vision-language, temporal-coverage, and area checks, each accepted mask guides SVOR [19] to remove the selected object, yielding $\mathbf { V } _ { \mathrm { r e m } }$ . This reconstructed background fills areas exposed by stylization-induced contour changes, avoiding original-object remnants during compositing. Appendix Section A.1 details the mask selection and filtering procedures.

Step 2: Global stylization and filtering. With source and reconstructed background videos available, we run both Ditto [21] and our internal fine-tuned VACE [14] on each under a stylization instruction $p .$ . For either global stylizer $G ,$ we get two stylized outputs:

$$
{ \bf V } _ { \mathrm { s r c } } ^ { \mathrm { s t y } } = G ( { \bf V } _ { \mathrm { s r c } } , p ) , \qquad { \bf V } _ { \mathrm { r e m } } ^ { \mathrm { s t y } } = G ( { \bf V } _ { \mathrm { r e m } } , p ) .\tag{1}
$$

Before using these outputs for local composition, GPT-5.5 is used to assess style quality, content preservation, and temporal quality through two cascaded gates: the source-video pair is evaluated first, followed by the object-removed pair. Only candidates passing both proceed to matting and compositing. More details appear in Appendix Section A.2.

Step 3: Matting and localized compositing. We next localize the accepted stylizations. Since stylization can change object contours, we use VideoMaMa [34] to estimate separate alpha mattes $\mathbf { A } _ { \mathrm { s r c } }$ and $\mathbf { A } _ { \mathrm { s t y } }$ from $\mathbf { V } _ { \mathrm { s r c } }$ and $\mathbf { V } _ { \mathrm { s r c } } ^ { \mathrm { s t y } }$ with the same tracked mask $\mathbf { M } _ { \mathrm { o b j } }$ , respectively. These mattes follow the respective object boundaries through time and are used for compositing instead of binary masks. The resulting composites stylize the foreground while retaining the background, or vice versa:

$$
\mathbf { V } _ { \mathrm { f g } } ^ { \mathrm { s t y } } = \mathbf { A } _ { \mathrm { s t y } } \odot \mathbf { V } _ { \mathrm { s r c } } ^ { \mathrm { s t y } } + ( 1 - \mathbf { A } _ { \mathrm { s t y } } ) \odot \mathbf { V } _ { \mathrm { r e m } } ,\tag{2}
$$

$$
\mathbf { V } _ { \mathrm { b g } } ^ { \mathrm { s t y } } = \mathbf { A } _ { \mathrm { s r c } } \odot \mathbf { V } _ { \mathrm { s r c } } + ( 1 - \mathbf { A } _ { \mathrm { s r c } } ) \odot \mathbf { V } _ { \mathrm { r e m } } ^ { \mathrm { s t y } } .
$$

Here $\odot$ denotes element-wise multiplication, with alpha values broadcast across color channels. Both targets are paired with $\mathbf { V } _ { \mathrm { s r c } }$ and $p$ . The directional editing mask is $\mathbf { M } = \mathbf { M } _ { \mathrm { o b j } }$ for the foreground target and $\mathbf { M } = 1 - \mathbf { M } _ { \mathrm { o b j } }$ for the background target.

Step 4: Human verification. Finally, because compositing may introduce artifacts, annotators inspect each completed example for content preservation, boundary quality, and temporal consistency. Only qualified samples are retained, yielding a final dataset of 100K video pairs covering 18 style categories. More details on the verification and final dataset statistics can be found in Section A.3.

## 4 GRNStyle: Free-Style Local Video Stylization

As the model within LociArt, GRNStyle binds style identities to target regions and combines encoded mask features with explicit geometric support (Fig. 3). Source evidence guides content preservation throughout generation, while the same spatial conditioning interface accommodates multiple regions at inference.

## 4.1 Problem Formulation and Binary Evidence

Given a source video $\mathbf { X } = \mathbf { V } _ { \mathrm { s r c } } \in [ 0 , 1 ] ^ { N \times H \times W \times 3 }$ with � frames of spatial size $H \times W _ { : }$ , an editing mask $\mathbf { M } \in \{ 0 , 1 \} ^ { N \times H \times W \times 1 }$ , and a text instruction �, we derive the target style $s \in \{ 1 , \ldots , K \}$ from �, where � is the number of style categories. We embed � into M to form a style-aware mask M and learn $\widehat { \mathbf { Y } } = F ( \mathbf { X } , \mathbf { M } _ { s } , p ; \boldsymbol { \Theta } )$ from locally stylized targets $\mathbf { Y } = \mathbf { V } ^ { \mathrm { s t y } }$ , aiming to transform the selected region while preserving non-target content.

To preserve source content during generation, we build on GRN [22] and GRNEdit’s sourceevidence formulation [23]. An encoder E and binary quantizer Q map the source to $\begin{array} { r l } { \mathbf { B } ^ { \mathbf { X } } } & { { } = } \end{array}$ $Q ( \mathcal { E } ( \mathbf { X } ) ) \in \{ 0 , 1 \} ^ { L \times D }$ , with � latent positions and � bits per position. GRN uses one-hot bit codes $\phi ( b ) = ( 1 - b , b )$ and parallel iterative refinement, while GRNEdit supplies source evidence for retaining or changing target bits. GRNStyle adds explicit style–region binding and geometric control to localize the transformation.

![](images/a5e018c55dc161a61f2c180262cb69e0886e951c31f21c3e0e75c550b3f1fb8a.jpg)  
Figure 3 Overview of GRNStyle. (a) Polar style-aware masks bind styles to regions and support multi-region composition. (b) Anchored mask evidence and geometric support are combined with source features through block-wise gating to guide local video stylization.

## 4.2 Equal-Norm Style–Space Conditioning

A binary mask specifies where to edit but leaves the associated style to a separate text condition. To bind style directly to space, we assign each style a fixed polar angle $\theta _ { s }$ at unit radius and embed its coordinates within the selected region (Fig. 3(a)). The first two channels of the three-channel VAE input store the rescaled Cartesian coordinates, i.e., $\begin{array} { r } { \pmb { a } _ { s } = ( \cos \theta _ { s } , } \end{array}$ sin $\theta _ { s } )$ , while the third retains binary occupancy for the polar style-aware mask:

$$
{ \bf M } _ { s } ( q ) = { \bf M } ( q ) \left( \frac { 1 + a _ { s , 1 } } { 2 } , \frac { 1 + a _ { s , 2 } } { 2 } , 1 \right) .\tag{3}
$$

Here � indexes spatiotemporal pixel positions. Since $\| \pmb { a } _ { s } \| _ { 2 } = 1$ , styles difer by direction rather than magnitude, avoiding an arbitrary scalar ordering. Unedited positions remain zero, and occupancy is directly recoverable from the third channel. This representation adds neither channels nor conditioning tokens relative to a repeated binary mask. Appendix Section B.1 provides encoding details.

Crucially, the resulting field contains the region–style interaction before video encoding, as a linear readout $\mathcal { R } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ recovers

$$
\mathcal { R } ( \mathbf { M } _ { s } ( q ) ) = \mathbf { M } ( q ) \mathbf { { a } } _ { s } .\tag{4}
$$

Thus, the input explicitly associates each edited position with its style code. Appendix Section B.3 derives this readout.

## 4.3 Representation–Geometry Mask Conditioning

The mask must be encoded on the backbone’s latent grid during generation. However, common mask compression strategies can compromise spatial localization [19]. We therefore combine encoded mask features with geometric support from binary occupancy, retaining spatiotemporal context while localizing mask evidence.

Representation and anchoring. To align mask conditioning with source representations, we encode both through the frozen map $\Phi = \phi \circ Q \circ \mathcal { E }$ , including input normalization and spatiotemporal flattening. Since an all-zero mask does not encode to zero, we anchor mask evidence to the encoding of an empty mask of the same shape:

$$
\begin{array} { r } { \mathbf { Z } ^ { \mathbf { X } } = \boldsymbol { \Phi } ( \mathbf { X } ) , \qquad \Delta \mathbf { Z } ^ { \mathbf { M } } = \boldsymbol { \Phi } ( \mathbf { M } _ { s } ) - \boldsymbol { \Phi } ( \mathbf { 0 } ) . } \end{array}\tag{5}
$$

This subtraction ensures zero mask evidence for an absent edit and represents nonempty masks through changes relative to the shared empty-condition baseline.

Geometric support. Following the VAE’s temporal grouping and spatial strides, we use average pooling AvgPool to map binary occupancy onto the latent grid, then apply spatial max pooling MaxPool and threshold:

$$
\Omega ( \mathbf { M } ) = \mathbf { 1 } [ \mathbf { M a x P o o l } ( \mathbf { A v g P o o l } ( \mathbf { M } ) ) \geq 0 . 5 ] .\tag{6}
$$

Max pooling uses unit stride and size-preserving padding, providing a kernel-controlled boundary margin without further downsampling. Complementing empty-mask anchoring, the flattened support $\Omega ( \mathbf { M } ) \in \{ 0 , 1 \} ^ { L \times 1 }$ localizes projected mask injection without hard-masking source features or output pixels.

## 4.4 Evidence-Guided Generation and Compositional Inference

Evidence fusion. To condition the backbone on source content and local style, we project their evidence into a shared �-dimensional space using $\pmb { P } ^ { \mathbf { X } }$ and $\pmb { P } ^ { \mathbf { M } }$ . The mask projection is bias-free to preserve the zero anchored response. These projections are shared across injection blocks and form $\mathbf { U } = P ^ { \mathbf { X } } ( \mathbf { Z } ^ { \mathbf { X } } ) + \Omega ( \mathbf { M } ) \odot P ^ { \mathbf { M } } \big ( \Delta \mathbf { Z } ^ { \mathbf { M } } \big ) \in \mathbb { R } ^ { L \times d }$ , combining source context with geometrically localized mask evidence. To adapt this shared signal to each block, we compute the channel gate $g _ { k } = g _ { \mathrm { v i s } , k } ( \mathbf { U } ) + g _ { \mathrm { t e x t } , k } ( p ) \in \mathbb { R } ^ { d }$ from visual evidence and text, broadcast it across the � spatiotemporal positions, and inject the gated residual:

$$
\mathbf { R } _ { k } = \mathbf { U } \odot \mathbf { g } _ { k } , \qquad \mathbf { H } _ { k } = \mathcal { G } _ { k } \big ( \mathbf { H } _ { k - 1 } + \boldsymbol { \Pi } _ { \mathrm { v i s } } ( \mathbf { R } _ { k } ) \big ) .\tag{7}
$$

Here $\mathcal { G } _ { k }$ is the �-th DiT block with output $\mathbf { H } _ { k } ,$ and $\Pi _ { \mathrm { v i s } }$ inserts residuals at existing visual positions without adding tokens. Blocks without evidence injection use ${ \mathbf { R } } _ { k } = 0$

Learning and generation. With clean source and mask evidence as conditions, GRNStyle predicts target bits from a partially randomized state. We retain GRN’s full-video bitwise cross-entropy to jointly supervise stylization and preservation. Training mixes style-specific text with generic mask following instructions, encouraging the model to read spatially encoded styles while retaining text guidance. At inference, random target bits are iteratively refined under the same conditioning and decoded into video.

Compositional inference. This mask-following interface also supports multiple regions through the same codebook. For � disjoint regions $\{ \mathbf { M } ^ { r } \} _ { r = 1 } ^ { R }$ and learned styles $\{ s _ { r } \} _ { r = 1 } ^ { R }$ , we compose the encodings in Equation 3:

$$
\mathbf { M } _ { s } ^ { \mathrm { m u l t i } } ( q ) = \sum _ { r = 1 } ^ { R } \mathbf { M } _ { s } ^ { r } ( q ) , \qquad \mathbf { M } ^ { \mathrm { u n i o n } } = \sum _ { r = 1 } ^ { R } \mathbf { M } ^ { r } .\tag{8}
$$

Here $\mathbf { M } _ { s } ^ { r }$ denotes the polar encoding of region $\mathbf { M } ^ { r }$ . Disjointness ensures a valid style code or empty condition at each position, with $\begin{array} { r } { \mathcal { R } ( { \bf M } _ { s } ^ { \mathrm { m u l t i } } ( { q } ) ) = \sum _ { r } { \bf M } ^ { r } ( { q } ) { \bf a } _ { s _ { i } } } \end{array}$ by Equation 4. We jointly encode the composite field and localize its mask evidence using $\Omega ( { \bf M } ^ { \mathrm { u n i o n } } )$ . Thus, all regions are generated in a single refinement trajectory under a generic mask-following instruction, without adding channels or conditioning tokens. Qualitative examples illustrate these compositions after single-region, single-style training, without further training on region–style combinations.

Table 1 Comparison on the LociArt Benchmark. The fg/bg sufixes denote stylization of the target mask � and its complement 1 − �. FT denotes local-stylization fine-tuning, and ∗ denotes training-free methods. <sup>†</sup> marks direct mask-compositing methods, for which PSNR is omitted because non-edited regions are copied from the source. <sup>‡</sup> marks image-based methods. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Venue</td><td>SPS-fg↑</td><td>SPS-bg↑</td><td>CLIPT-fg↑</td><td>CLIPT-bg↑</td><td>PSNR-fg↑</td><td>PSNR-bg↑</td><td>TC↑</td><td>Style↑</td><td>Preserve↑</td></tr><tr><td>LEAST* [35]</td><td>arXiv &#x27;24</td><td>0.283</td><td>0.302</td><td>0.158</td><td>0.173</td><td></td><td></td><td>0.953</td><td>2.20</td><td>4.50</td></tr><tr><td>VACE-1.3B [14]</td><td>ICCV &#x27;25</td><td>0.156</td><td>0.191</td><td>0.114</td><td>0.132</td><td>32.35</td><td>29.17</td><td>0.981</td><td>0.05</td><td>5.00</td></tr><tr><td>UniVST* [10]</td><td>TPAMI &#x27;25</td><td>0.191</td><td>0.286</td><td>0.127</td><td>0.165</td><td>27.69</td><td>23.92</td><td>0.969</td><td>1.83</td><td>4.40</td></tr><tr><td>RegionRoute [12]</td><td>CVPR &#x27;26</td><td>0.317</td><td>0.491</td><td>0.154</td><td>0.183</td><td>29.38</td><td>13.17</td><td>0.955</td><td>2.41</td><td>2.75</td></tr><tr><td>StyleGallery* [11]</td><td>CVPR &#x27;26</td><td>0.230</td><td>0.325</td><td>0.146</td><td>0.183</td><td></td><td></td><td>0.957</td><td>3.39</td><td>4.77</td></tr><tr><td>CoinVE-Edit [13]</td><td>arXiv ’26</td><td>0.248</td><td>0.251</td><td>0.153</td><td>0.159</td><td>20.71</td><td>16.76</td><td>0.983</td><td>3.11</td><td>3.93</td></tr><tr><td>Seedance 2.5 [36]</td><td>Commercial</td><td>0.347</td><td>0.358</td><td>0.173</td><td>0.175</td><td>18.50</td><td>18.76</td><td>0.984</td><td>3.98</td><td>2.40</td></tr><tr><td>VACE-1.3B (our FT)</td><td></td><td>0.386</td><td>0.676</td><td>0.155</td><td>0.185</td><td>25.43</td><td>25.49</td><td>0.980</td><td>4.52</td><td>4.56</td></tr><tr><td>GRNStyle-2B (ours)</td><td></td><td>0.407</td><td>0.705</td><td>0.160</td><td>0.199</td><td>25.92</td><td>26.15</td><td>0.983</td><td>4.62</td><td>4.64</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Training. We train the 2B GRNStyle model on LociArt-100K with 8 × NVIDIA H200 GPUs, using all 18 styles unless otherwise specified. We fully fine-tune the backbone and conditioning branches. More training details are provided in Appendix Section D.

Benchmark. We introduce the LociArt Benchmark for local video stylization; it contains 80 source videos and 84 object-mask units. Each video–mask pair is evaluated under two complementary stylization protocols: foreground stylization edits the tracked foreground-object mask $\mathbf { M } _ { \mathrm { o b j } }$ , while background stylization edits its complement $1 -  { \mathbf { M } } _ { \mathrm { o b j } }$ . The benchmark therefore contains 168 cases in total. More details can be found in Appendix Section C.1.

Metrics. We use SPS (Style Prototype Similarity) to measure whether the edited region matches the requested style using a prototype bank in the CSD [31] feature space. Regional CLIP-T (CLIPT) [37] measures alignment between the edited region and the canonical style phrase, while PSNR [38] measures fidelity to the source in the non-target region. TC (Temporal Consistency) [39] measures cosine similarity between CLIP embeddings of adjacent frames. Additionally, we use Gemini 3.6 Flash to provide separate Style and Preserve scores for regional stylization quality and non-target content preservation. Detailed definitions and implementation protocols are provided in Sections C.2 to C.4.

## 5.2 Comparison with Existing Methods

We compare with methods spanning regional image stylization and video editing: LEAST [35], UniVST [10], RegionRoute [12], and StyleGallery [11] are image-based methods, whereas VACE-1.3B [14], CoinVE-Edit [13], and the commercial system Seedance 2.5 [36] target video generation or editing. LEAST and StyleGallery directly composite the non-edited region from the source using the mask, while others generate or reconstruct the complete output; accordingly, PSNR is not reported for the former two because source-pixel copying makes it non-comparable. To isolate the contribution of task-specific supervision, we additionally fine-tune VACE-1.3B on LociArt-100K using the same architecture-independent hyperparameters as GRNStyle.

Quantitative and perceptual comparison. As shown in Table 1, GRNStyle achieves the best SPS for both foreground (0.407) and background (0.705) stylization, together with the best background CLIPT (0.199). It also maintains strong preservation, with the second-best background PSNR (26.15 dB) and near-best temporal consistency (TC: 0.983). The VLM scores further indicate a favorable balance between regional stylization and non-target preservation, achieving the highest Style score (4.62) while retaining a high Preserve score (4.64). The highest PSNR values from VACE coincide with its minimal stylization, so they do not indicate a better style–preservation trade-of. In addition, Appendix F shows that GRNStyle remains competitive with SOTA global stylization methods under all-one masks.

![](images/8ab12219f777cf52d241dda6b09261f117b63934f229a2c777de8f0fe5c54adf.jpg)  
Figure 4 Qualitative comparison of local video stylization. Examples evaluate stylization inside the editing region, preservation outside it, and temporal consistency. Foreground and background cases highlight leakage, distortion of preserved content, and boundary inconsistency. GRNStyle produces localized styles with coherent boundaries and stable non-target content.

Qualitative comparison. Fig. 4 presents representative resultsof local video stylization. GRNStyle renders the armchair and room in pixel art while preserving the dog, and adds pencil strokes to the bear without desaturating the surrounding rocks, demonstrating precise regional stylization and faithful preservation. Other methods tend to produce weak edits, spill stylization beyond the target, or lose subject detail. Additional examples are provided in Appendix I.

## 5.3 User Study

We conducted a user study with 20 participants on a 160-case subset (16 cases per style), with each case receiving one rating from this participant pool. Since the study evaluates generated videos, we compare GRNStyle only with the two video-based methods, CoinVE-Edit and Seedance 2.5. As shown in Fig. 5(a), GRNStyle achieves the highest mean ratings for style fidelity (4.34) and background preservation (4.47), indicating stronger artistic expression together with better scene preservation. Fig. 5(b) further shows consistent benefits across styles: GRNStyle outperforms CoinVE-Edit on both criteria for all ten styles, improves preservation over Seedance 2.5 for every style, and improves style fidelity for seven styles.

## 5.4 Ablation of Mask Conditioning

We evaluate two complementary mask-conditioning strategies: RG (representation–geometry conditioning), which combines empty-mask anchoring with raw-mask geometric support, and Polar conditioning, which encodes an explicit regional style identity through a polar representation.

Table 2 shows that the baseline performs worst overall. RG substantially improves PSNR while leaving SPS and CLIPT nearly unchanged, indicating improved preservation without sacrificing stylization. As illustrated in Fig. 6, empty-mask anchoring removes the reference response to an empty input, while geometric support suppresses spatial spillover. Polar encoding supplies an explicit regional style identity, yielding a clear SPS improvement and a modest PSNR gain. Combining the two components achieves the strongest overall trade-of between style conditioning and spatial control. The appendix analysis in Section G further shows that GRNStyle remains robust to imperfect masks.

![](images/6509d73efa60ad8997ecc544ddebc75e7b3c2f31cc54e6f1d67e87459e5b2083.jpg)  
(a) Overall ratings

<table><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>+1.88</td><td></td><td rowspan=1 colspan=1>+1.81</td><td rowspan=1 colspan=1>+1.06</td></tr><tr><td rowspan=1 colspan=4></td></tr></table>

Figure 5 Human evaluation results. Values show mean ± SD; per-style gains are shown for the ten-style subset.

![](images/a8e0b6fa23048609347eced22e3632a9c0f5ae59d067f1fccdce391952be4011.jpg)  
Figure 6 Mask-conditioning ablation. Representation–geometry conditioning improves spatial localization over the raw encoding.

Table 2 Component ablation. RG: representation–geometry conditioning. Polar: polar styleaware mask.
<table><tr><td>Configuration</td><td>SPS↑</td><td>CLIPT↑</td><td>PSNR↑</td></tr><tr><td>Base</td><td>0.541</td><td>0.180</td><td>20.09</td></tr><tr><td>Base + Polar</td><td>0.553</td><td>0.181</td><td>21.34</td></tr><tr><td>Base + RG</td><td>0.544</td><td>0.179</td><td>25.07</td></tr><tr><td>GRNStyle</td><td>0.554</td><td>0.181</td><td>25.78</td></tr></table>

## 5.5 Learning New Styles without Replay

To examine whether a predefined polar codebook limits expansion to new styles, we extend a separately trained 10-style model to 15 styles using only the five new classes and no replay of the original data. The polar and binary variants follow the same protocol, and retention is measured against their respective 10-class checkpoints.

Table 3 shows that polar conditioning supports this 50% vocabulary expansion with substantially less forgetting: old-class SPS drops by 0.022, compared with 0.102 for binary masks, while the corresponding CLIPT drops are 0.008 and 0.019; PSNR remains stable in both cases. The new-class scores are also slightly higher across all metrics, indicating that improved retention does not hinder adaptation. These results support the extensibility of explicit style identities: new styles can be added while better preserving existing capabilities, without replaying the original data.

Table 3 Replay-free style expansion from 10 to 15 classes. Expanded models are trained only on the five new classes. Retention and acquisition are evaluated on 1,680 original-style and 840 new-style cases, respectively.
<table><tr><td rowspan="2">Configuration</td><td colspan="3">10 classes (old)</td><td colspan="3">5 classes (new)</td></tr><tr><td>SPS↑</td><td>CLIPT↑</td><td>PSNR↑</td><td>SPS↑</td><td>CLIPT↑</td><td>PSNR↑</td></tr><tr><td>Binary, 10 classes</td><td>0.547</td><td>0.177</td><td>25.42</td><td></td><td></td><td></td></tr><tr><td>Binary, 15 classes</td><td>0.445-∆0.102</td><td>0.158-∆0.019</td><td>25.43 +∆0.01</td><td>0.588</td><td>0.197</td><td>25.39</td></tr><tr><td>Polar, 10 classes</td><td>0.553</td><td>0.180</td><td>25.75</td><td></td><td></td><td></td></tr><tr><td>Polar, 15 classes</td><td>0.531-∆0.022</td><td>0.172-∆0.008</td><td>25.77 +∆0.02</td><td>0.590</td><td>0.199</td><td>25.74</td></tr></table>

![](images/04f76c7104a807ffc2feae2e196ce750dbad4f170bae930bef02578095fed4e6.jpg)  
Figure 7 Data-construction ablation. For curly hair (a) and head contours (b), columns compare direct compositing, removal without matting, and the full pipeline. Rows show background- and foreground-stylized targets; insets highlight residual silhouettes, clipped contours, and boundary quality.

## 5.6 Efect of the Data Construction Pipeline

To assess the contribution of each data-construction step, we compare direct mask compositing, using object removal but no matting, and the complete pipeline for both foreground and background stylization. As shown in Fig. 7, background editing with direct compositing can leave ghost curls or head outlines when stylized subjects extend beyond their source silhouettes; object removal eliminates these artifacts. For foreground editing, source-aligned masks may clip stylized curls, hair tips, and noses, while removal alone cannot resolve the mismatch. Matting follows the actual foreground contour, preserving fine boundaries and providing cleaner supervision for style-induced shape changes.

## 6 Conclusion

We introduced LociArt, a unified framework comprising automated data construction, the LociArt-100K dataset, GRNStyle, and a region-aware benchmark. Built on GRN, GRNStyle combines representation–geometry conditioning with polar style codes to balance spatial control and temporal context while preserving non-target content, outperforming the evaluated baselines on the LociArt Benchmark. Category-aware polar style-aware masks further enable zero-shot multi-region compositions from single-region, single-style supervision, reducing the need for exhaustive paired data. Limitations and future works are discussed in Appendix Section H.

## Acknowledgements

This work uses the VideoMaMa model (https://github.com/cvlab-kaist/VideoMaMa), licensed under CC BY-NC 4.0. The Ditto dataset and model (https://github.com/EzioBy/ Ditto), licensed under CC BY-NC-SA 4.0. The authors confirm that all uses of the above resources are strictly for academic research purposes and not for any commercial application.

## References

[1] L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2414–2423.

[2] X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normalization,” in 2017 IEEE international conference on computer vision (ICCV). IEEE, 2017, pp. 1510–1519.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent difusion models,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). ieee, 2022, pp. 10 674–10 685.

[4] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 18 392–18 402.

[5] M. Geyer, O. Bar Tal, S. Bagon, and T. Dekel, “Tokenflow: Consistent difusion features for consistent video editing,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 1608–1620.

[6] Z. Ye, H. Huang, X. Wang, P. Wan, D. Zhang, and W. Luo, “Stylemaster: Stylize your video with artistic generation and translation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 2630–2640.

[7] Y. Song, W. Yao, H. Wang, and M. Z. Shou, “Vista: Triplet-supervised video style transfer with difusion transformers,” arXiv preprint arXiv:2605.17312, 2026.

[8] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything in images and videos,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 28 085–28 128.

[9] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris Coll-Vinent, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” in International conference on learning representations, vol. 2026, 2026, pp. 138 846–138 923.

[10] Q. Song, M. Lin, W. Zhan, S. Yan, L. Cao, and R. Ji, “Univst: A unified framework for trainingfree localized video style transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[11] B. He, Y. Ye, C. Liu, W. Wu, F. Liu, and Z. Cai, “Stylegallery: Training-free and semanticaware personalized style transfer from arbitrary image references,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 29 092–29 102.

[12] B. Chen, J. Zuena, A. C. Bovik, and D. Kothandaraman, “Regionroute: Regional style transfer with difusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 35 736–35 746.

[13] F. Long, C. Wang, Z. Gao, W. Zhong, Y. Cheng, X. Hou, Y. Li, X. Cao, X. Sun, X. Chen et al., “Coinve-200k: A large-scale high-quality dataset for compositional instruction-guided video editing,” arXiv preprint arXiv:2608.17566, 2026.

[14] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “Vace: All-in-one video creation and editing,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 17 191–17 202.

[15] Y. Bian, Z. Zhang, X. Ju, M. Cao, L. Xie, Y. Shan, and Q. Xu, “Videopainter: Any-length video inpainting and editing with plug-and-play context control,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025, pp. 1–12.

[16] S. Li and Y. Pan, “An eficient and harmonized framework for balanced cross-domain feature integration,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, 2026, pp. 6369–6377.

[17] D. Kang, J. Hwang, J. Park, M. Kang, Y. Lee, B. Ko, H. Roh, J. Shin, and H. Jang, “Mast: Mask-guided attention control for training-free regional-multi style transfer,” arXiv preprint arXiv:2604.12281, 2026.

[18] D. Yatim, R. Fridman, O. Bar-Tal, and T. Dekel, “Dynvfx: Augmenting real videos with dynamic content,” in Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 2025, pp. 1–12.

[19] J. Hu, Y. Chen, F. Li, Z. Wang, F. Wang, D. Zhou, and J. Luan, “From ideal to real: Stable video object removal under imperfect conditions,” arXiv preprint arXiv:2603.09283, 2026.

[20] M. Li, J. Chen, S. Zhao, W. Feng, P. Tu, and Q. He, “Dreamstyle: A unified framework for video stylization,” arXiv preprint arXiv:2601.02785, 2026.

[21] Q. Bai, Q. Wang, H. Ouyang, Y. Yu, H. Wang, W. Wang, K. L. Cheng, S. Ma, Y. Zeng, Z. Liu et al., “Scaling instruction-based video editing with a high-quality synthetic dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 37 971–37 981.

[22] J. Han, J. Liu, J. Wang, B. Peng, and Z. Yuan, “Generative refinement networks for visual synthesis,” in European Conference on Computer Vision. Springer, 2026, pp. 566–585.

[23] F. Xie, J. Hu, F. Li, Z. Wang, Y. Chen, D. Gao, F. Wang, and D. Zhou, “Grnedit: Eficient general video editing from a new binary-evidence perspective in generative refinement networks,” arXiv preprint arXiv:2608.16328, 2026.

[24] J. Gao, Y. Sun, Y. Liu, Y. Tang, Y. Zeng, D. Qi, K. Chen, and C. Zhao, “Styleshot: A snapshot on any style,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[25] S. Zhang, X. Yang, B. Zi, H. Huang, C. Zhang, and X. Li, “Telestyle: Content-preserving style transfer in images and videos,” arXiv preprint arXiv:2601.20175, 2026.

[26] X. Xia, T. Xue, W.-s. Lai, Z. Sun, A. Chang, B. Kulis, and J. Chen, “Real-time localized photorealistic video style transfer,” in 2021 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2021, pp. 1088–1097.

[27] S. Mehraban, V. Adeli, J. Rommann, B. Taati, and K. Truskovskyi, “Pickstyle: Video-to-video style transfer with context-style adapters,” arXiv preprint arXiv:2510.07546, 2025.

[28] X. Wei, Y. Fang, Y. Li, L. Wu, Q. Yang, M. Zhu, N. Wang, and X. Gao, “Stylized video generation via decoupled data synthesis and gated style token injection,” in European Conference on Computer Vision. Springer, 2026, pp. 19–36.

[29] J. Gao, S. Liu, J. Li, Y. Sun, Y. Tu, F. Shen, W. Zhang, C. Zhao, and J. Zhang, “Megastyle: Constructing diverse and scalable style dataset via consistent text-to-image style mapping,” arXiv preprint arXiv:2604.08364, 2026.

[30] M. Wright and B. Ommer, “Artfid: Quantitative evaluation of neural style transfer,” in DAGM German Conference on Pattern Recognition. Springer, 2022, pp. 560–576.

[31] G. Somepalli, A. Gupta, K. Gupta, S. Palta, M. Goldblum, J. Geiping, A. Shrivastava, and T. Goldstein, “Measuring style similarity in difusion models,” arXiv preprint arXiv:2404.01292, 2024.

[32] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 21 807–21 818.

[33] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[34] S. Lim, S. W. Oh, J. Huang, H. Yoon, S. Kim, and J.-Y. Lee, “Videomama: Mask-guided video matting via generative prior,” arXiv preprint arXiv:2601.14255, 2026.

[35] S. Singh, S. Jandial, S. Shahid, and A. Java, “Least:" local" text-conditioned image style transfer,” arXiv preprint arXiv:2405.16330, 2024.

[36] ByteDance Seed Team, “One-take creation, flexible referencing: Introducing seedance 2.5,” Jul. 2026, product announcement, July 31, 2026. [Online]. Available: https://seed.bytedan ce.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5

[37] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 8748–8763.

[38] A. Hore and D. Ziou, “Image quality metrics: Psnr vs. ssim,” in 2010 20th international conference on pattern recognition. IEEE, 2010, pp. 2366–2369.

[39] Z. Zhang, B. Wu, X. Wang, Y. Luo, L. Zhang, Y. Zhao, P. Vajda, D. Metaxas, and L. Yu, “Avid: Any-length video inpainting with difusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7162–7172.

[40] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool, “The 2017 davis challenge on video object segmentation,” arXiv preprint arXiv:1704.00675, 2017.

[41] S. P. Lloyd, “Least squares quantization in pcm,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982.

[42] P. J. Rousseeuw, “Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.

[43] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008.

[44] H. Li, J. Wang, S. Cai, H. Sheng, B. Deng, J. Ye, and W. Luo, “Echostyle: Unlocking highfidelity video stylization with reverse data synthesis,” in European Conference on Computer Vision. Springer, 2026, pp. 270–286.

## Appendix

The supplementary material is organized as follows:

• A. Detailed Data Pipeline (p. 16): data construction, filtering, verification, and statistics.

• B. Style-Code Design and Local Readout Analysis (p. 20): circular codebook, occupancy normalization, and readout analysis.

• C. Details of the Evaluation Protocol (p. 21): benchmark construction, localized metrics, and VLM evaluation.

• D. Model Reproduction Details (p. 26): training configuration, evidence injection, and checkpoint validation.

• E. Inference Eficiency (p. 27): latency and GPU-memory comparison.

• F. Global Stylization Comparison (p. 28): all-one-mask comparison with global stylization methods.

• G. Robustness to Imperfect Masks (p. 29): five mask perturbations at two severity levels.

• H. Limitations and Future Work (p. 29): current limitations and future directions.

• I. Additional Qualitative Results (p. 30): additional foreground and background examples.

## A Detailed Data Pipeline

All videos are standardized to 81 frames at 720p resolution. Frame correspondence is maintained throughout object removal, global stylization, matting, and compositing, so source, removed, stylized, and alpha videos remain aligned. Both Ditto [21] and our internal fine-tuned VACE [14] are run on the source and object-removed videos; each stylizer produces candidate outputs for the subsequent quality checks.

## A.1 Object Inventory and Mask Filtering Details

This stage identifies candidate foreground objects and obtains reliable masks for object removal and subsequent local composition. We first build an object inventory, track the selected instances, and then filter their masks for semantic correctness and temporal coverage.

Object inventory. Qwen3-VL-32B [33] analyzes each source video to identify prominent foreground objects, excluding broad background regions such as sky, roads, and walls. Each inventory entry contains a concise category label and an appearance- and location-based description. The category label provides the prompt for segmentation.

Instance handling. Using these category prompts, SAM3 [9] produces temporally tracked instance masks. When several objects share a category, their masks remain separate if their identities can be reliably tracked, preserving object-specific control for the later removal and compositing steps.

Mask filtering. We then use Qwen3-VL-32B [33] to reject missing, flickering, or incorrectly assigned masks. Additional checks require nonempty masks in more than 40% of frames and reject masks whose maximum per-frame area exceeds 85% of the frame. Temporal coverage counts all nonempty frames, not the length of a continuous track. These production thresholds select masks with suficient visibility and a bounded foreground extent before object removal.

## A.2 Global Stylization Quality Filtering Details

The source and object-removed videos provide complementary content for local composition, so both stylizations must preserve their inputs while expressing the requested style. We assess them through two cascaded gates with a shared sampling and scoring protocol, retaining only candidates that satisfy the acceptance criteria below.

Cascaded evaluation. The source-video pair is evaluated first; failure at this gate stops processing for the candidate. Passing candidates proceed to evaluation of the object-removed pair, and both pairs must pass before matting and compositing. To avoid repeated evaluation, source-gate results are reused for candidates sharing the same source, style instruction, and actual stylized output. Removal-gate results are cached separately for each object-removed video, style instruction, and corresponding stylized output. Distinct removal inputs or generated outputs require separate evaluations, even when the source video and style are shared.

Frame sampling and scoring. At either gate, we sample frame indices 0, 20, 40, 60, and 80 from the aligned 81-frame videos. GPT-5.5 receives ten images in chronological paired order, with an original and a stylized frame at each time point. It scores style adherence, content preservation, structural quality, style consistency, and temporal quality on a 1–5 scale.

Acceptance criteria. The resulting five scores must reach 4, 4, 3, 4, and 4, respectively. In addition, a pair is rejected if it introduces salient objects, removes or duplicates objects, changes semantics or identity, inserts text or logos, or exhibits major layout changes, severe deformation, or ghosting. The total score is reported for reference only and imposes no additional passing threshold.

Evaluation prompt. The following prompt (Listing 1) implements the same sampling, scoring, and acceptance protocol for both gates, with the corresponding input video and stylized output supplied in each case.

## A.3 Human Verification Details

After matting and compositing, human review checks the completed local edits for failures that global stylization filtering may miss. Reviewers assess each source–mask–instruction–target example against the criteria below, then retain qualified pairs to form the final dataset.

Listing 1 VLM prompt for global stylization quality filtering.

You are a strict content-preservation evaluator for a full-video stylization result.   
Target production style: {production\_prompt}   
Canonical style category: {style\_category}   
The input videos contain 81 aligned frames (indices 0 through 80). You receive 10 images from   
five uniformly sampled time points in chronological paired order:   
- Images 1 and 2: ORIGINAL frame 0 and STYLIZED frame 0   
- Images 3 and 4: ORIGINAL frame 20 and STYLIZED frame 20   
- Images 5 and 6: ORIGINAL frame 40 and STYLIZED frame 40   
- Images 7 and 8: ORIGINAL frame 60 and STYLIZED frame 60

![](images/c6c562344d3208764738cae2628601a52f19c21e1c54bad470df4c52bc4b08fb.jpg)  
Figure 8 Representative rejection cases from the data pipeline.

- Images 9 and 10: ORIGINAL frame 80 and STYLIZED frame 80   
Each original/stylized pair is the same time point. Compare every pair and use all five time   
points. Do not decide from the first pair alone.   
Evaluate the requested style while preserving semantic content strictly. Content preservation   
includes the same people, animals, objects, object count, identities, roles, actions,   
interactions, scene layout, camera view, and major background structures.   
Do not penalize normal style transformations such as changed colors, simplified details, stylized   
facial features that preserve identity, painterly/cartoon/anime/clay/pixel/vector rendering, or   
moderate texture simplification.   
However, be conservative: if an object identity, person identity, object count, action, or scene   
layout cannot be confidently matched between an original/stylized pair, score   
content\_preservation below 4 and reject. Do not assume an ambiguous or redesigned object is   
preserved.   
Reject a new salient person/object, disappearance or duplication, an object changing semantic   
category, identity/role/gender change, action transfer, major layout change, inserted text/logo/   
graphic, severe deformation, or ghosting.   
Judge style adherence against the explicitly requested target style, not against generic   
stylization or photorealism. Examples:   
Low Poly requires visible facets/polygonal shading.   
Pixel Art requires block-like pixels and a limited palette.   
Claymation requires plasticine material and rounded clay forms.   
Oil Painting requires painterly brushwork.   
Watercolor requires translucent washes.   
Anime or Ghibli requires recognizable hand-drawn animation treatment.   
Other styles must show their characteristic visual language.   
Score exactly these five dimensions from 1 to 5:   
style\_adherence   
content\_preservation   
structural\_quality   
style\_consistency   
temporal\_quality   
Passing criteria:   
style\_adherence >= 4   
content\_preservation >= 4   
structural\_quality >= 3   
style\_consistency >= 4   
temporal\_quality >= 4   
hard\_reject\_reasons must be empty   
Report total as the sum of the five scores for reference only; it is not an additional passing   
criterion.   
Return ONLY one valid JSON object. Do not output analysis, Markdown, or any text outside the JSON   
object:   
{   
"pass": false,   
"style\_adherence": 1,

![](images/2bc7eb1b2acb341367c3acad8a594d9bcae3d61915800bb729251d2deac63108.jpg)  
(a) Style-category distribution

![](images/077e270cb687fb6f8235a0a3c2eb1f1b6f11407787e92ea6e61d8075cfd8817a.jpg)  
(b) Foreground-object mask area ratios

Figure 9 Statistics of the LociArt-100K Dataset. The 100K qualified pairs cover 18 style categories. Area ratios refer to foreground object masks $\mathbf { M } _ { \mathrm { o b j } }$ , not the complementary background editing masks.  
"content\_preservation": 1,   
"structural\_quality": 1,   
"style\_consistency": 1,   
"temporal\_quality": 1,   
"total": 5,   
"hard\_reject\_reasons": [],   
"reason": "Brief evidence-based explanation"   
}

Verification requirements. A qualified sample must satisfy four requirements: (i) the requested style is expressed while semantic content, object identities and counts, actions, and scene layout remain consistent with the source; (ii) non-target content is preserved without unintended style spillover; (iii) masks and compositing boundaries remain coherent, without incomplete regions or visible unstylized remnants; and (iv) the video is temporally consistent, without frame breakage or flicker. These checks distinguish unintended content changes from valid artistic transformations: style-consistent changes in color, texture, detail, and object contours are permitted.

Acceptance and archiving. Candidates meeting all four requirements are labeled qualified and retained; those failing the checks are rejected and discarded. Typical rejection cases include weak or incorrect style, semantic or identity changes, inserted or missing objects, unintended changes to preserved content, incomplete masks, and temporal artifacts. Representative examples rejected during human verification are shown in Fig. 8 and our demo page: https://higherhu.githu b.io/lociart/.

Production statistics. The pass rate is 52.4% for global stylization VLM filtering and 65% for human verification of the resulting local composites. These rates are measured at their respective filtering stages. The final dataset contains 100K qualified video pairs covering 18 style categories. Figure 9 summarizes the style distribution and foreground object mask areas. The area histogram characterizes $\mathbf { M } _ { \mathrm { o b j } }$ , not its complementary background editing mask, and therefore does not represent the distribution of all editing regions.

## B Style-Code Design and Local Readout Analysis

## B.1 Codebook Assignment and Input Normalization

We use a fixed 100-slot circular codebook. For $K \leq 1 0 0 .$ , let $\nu : \{ 1 , \dots , K \} \to \{ 0 , \dots , 9 9 \}$ assign each style to a distinct slot, with $\theta _ { s } = 2 \pi \nu ( s ) / 1 0 0$ . Well-spaced slots are assigned deterministically, and the mapping remains fixed across training and inference. The resulting directions satisfy $\| \pmb { a } _ { s } \| _ { 2 } = 1$ , distinguishing styles by angle rather than magnitude, analogous to the equal-norm one-hot bit codes in GRN [22].

Following the main-text definition in Equation 3, the style-aware mask is

$$
{ \bf M } _ { s } ( q ) = { \bf M } ( q ) \left( \frac { 1 + a _ { s , 1 } } { 2 } , \frac { 1 + a _ { s , 2 } } { 2 } , 1 \right)\tag{9}
$$

The VAE input normalization $x \mapsto 2 x - 1$ therefore yields $c _ { s } = ( a _ { s , 1 } , a _ { s , 2 } , 1 )$ at edited positions and $e = \left( - 1 , - 1 , - 1 \right)$ at unedited positions. Thus, the normalized third channel retains the same binary occupancy cue for every style, while the first two channels encode its direction.

## B.2 Why Circular Rather Than Spherical Codes?

A three-coordinate spherical code ofers greater class-packing capacity, but uses every channel for identity. Local stylization also needs a stable distinction between edit and keep. We therefore reserve the third channel for occupancy and use the first two for style directions. The shared occupancy channel avoids requiring edit/keep status to be inferred from class-dependent coordinates.

A larger empty-condition margin. Reserving this occupancy axis keeps style codes away from the empty condition. For the circular and origin-centered unit-sphere alternatives,

$$
\begin{array} { r l r } & { \underset { \theta \in [ 0 , 2 \pi ) } { \operatorname* { m i n } } \ \lVert ( \cos \theta , \sin \theta , 1 ) - e \rVert _ { 2 } = \sqrt { 7 - 2 \sqrt { 2 } } \approx 2 . 0 4 , } & \\ & { \underset { \nu \in \mathbb { R } ^ { 3 } : \lVert \nu \rVert _ { 2 } = 1 } { \operatorname* { m i n } } \lVert \nu - e \rVert _ { 2 } = \sqrt { 3 } - 1 \approx 0 . 7 3 . } & \end{array}\tag{10}
$$

The circular representation therefore keeps every style farther from the empty condition than the spherical alternative. For the 100-slot codebook, the smallest distance between adjacent normalized style directions is 2 sin $( \pi / 1 0 0 ) \approx 0 . 0 6 3$ ; the empty-condition margin is the larger value $\sqrt { 7 - 2 \sqrt { 2 } } \approx 2 . 0 4$

A shared cue for the frozen encoder. These coordinates denote category identity, not visual color. The pretrained VAE nevertheless processes them through learned RGB features, so greater spherical spacing alone does not resolve the encoding problem. A binary occupancy channel supplies a common boundary cue across styles, while class-dependent variation remains confined to the other two channels. Our 100-slot VAE/HBQ validation found no exact token collisions. We therefore favor explicit occupancy and empty-condition separation over additional spherical capacity.

## B.3 Interaction Error and Exact Local Readout

This section isolates the style–region interaction in Equation 4. We compare an additive readout of separate conditions with a linear readout of the polar field, using the same region and style information.

Setup. At one pixel position, write $m = \mathbf { M } ( q ) \in \{ 0 , 1 \}$ and let ${ \pmb a } _ { s } \in \mathbb { R } ^ { 2 }$ be the unit style direction in Equation 3. The desired local instruction is $m { \pmb a } _ { s } .$ . To separate representation from datasetspecific correlations, consider independent $m \sim$ Bernoulli(�) and $s \sim \pi _ { : }$ , where $0 < \beta < 1$ and $\pi _ { s }$ is the prior probability of style $s ,$ and define $\begin{array} { r } { \pmb { \mu } = \sum _ { s } \pmb { \pi } _ { s } \pmb { a } _ { s } } \end{array}$ . For additive readouts $f ( m , s ) =$ $f _ { M } ( m ) + f _ { S } ( s )$ , the component functions may be arbitrary. We measure squared instruction error $\mathcal { T } ( f ) = \mathbb { E } \| m \pmb { \alpha } _ { s } - f ( m , s ) \| _ { 2 } ^ { 2 }$

Optimal additive error. The best additive readout and its error are

$$
f ^ { \star } ( m , s ) = m \pmb { \mu } + \beta ( \pmb { a } _ { s } - \pmb { \mu } ) ,\tag{11}
$$

$$
\operatorname* { m i n } _ { f = f _ { M } + f _ { S } } { \mathcal { T } } ( f ) = \beta ( 1 - \beta ) { \big ( } 1 - \| \pmb { \mu } \| _ { 2 } ^ { 2 } { \big ) } .\tag{12}
$$

To see this, decompose � $\pmb { a } _ { s } = f ^ { \star } ( m , s ) + ( m - \beta ) ( \pmb { a } _ { s } - \pmb { \mu } )$ . The residual has zero conditional mean given either � or �, so it is orthogonal to every additive readout. Consequently, $\mathcal { T } ( f ) = \mathcal { T } ( f ^ { \star } ) { + } \mathbb { E } \| f { - } f ^ { \star } \| _ { 2 } ^ { 2 }$ Independence and $\| \pmb { a } _ { s } \| _ { 2 } = 1$ then give Equation 12. The factor $1 - \| \pmb { \mu } \| _ { 2 } ^ { 2 }$ is the variance of the style directions. The minimum is positive when at least two distinct styles have positive probability. It is at most $\beta ( 1 - \beta ) \leq 1 / 4$ , with equality at $1 / 4$ when $\beta = 1 / 2$ and $\pmb { \mu } = 0$

Exact polar readout. For the same inputs, the encoded pixel ${ \bf M } _ { s } ( q )$ defined in Equation 9 satisfies

$$
\left( \begin{array} { c c c } { 2 } & { 0 } & { - 1 } \\ { 0 } & { 2 } & { - 1 } \end{array} \right) \mathbf { M } _ { s } ( q ) = m \pmb { a } _ { s } , \quad \mathbb { E } \| \mathcal { R } ( \mathbf { M } _ { s } ( q ) ) - m \pmb { a } _ { s } \| _ { 2 } ^ { 2 } = 0 .\tag{13}
$$

Polar encoding therefore supplies the interaction explicitly, without adding a new condition. This is a distinction between readout classes, not an error bound on a nonlinear generator: a nonlinear network can learn the interaction from separate inputs, whereas our representation computes it before encoding.

## C Details of the Evaluation Protocol

We first describe the construction and evaluation units of the LociArt Benchmark, then assess local stylization along three complementary axes: regional style, measured by Style Prototype Similarity (SPS) and regional CLIPT; temporal consistency and non-target preservation, measured by temporal consistency (TC) and regional PSNR; and full-video perceptual quality, measured by the VLM Style and Preserve scores.

## C.1 Benchmark Construction and Evaluation Units

The benchmark contains 80 source videos and 84 object-mask units, each with 81 aligned frames. Videos with two selected subjects contribute separate object-mask units. Of the 80 videos, 13 come from the rejected Ditto [21] pool and were excluded from our training data, 46 come from the DAVIS 2017 training split [40], and 21 come from Pexels<sup>1</sup>. The foreground area ratios range from 0.02 to 0.58, and 29 units contain human subjects. We denote the tracked foreground-object mask by $\mathbf { M } _ { \mathrm { o b j } }$ and the directional editing mask by M, with $\mathbf { M } = \mathbf { M } _ { \mathrm { o b j } }$ for foreground editing and $\mathbf { M } = 1 - \mathbf { M } _ { \mathrm { o b j } }$ for background editing. Throughout the metric definitions below, � denotes the number of frames in the evaluated aligned sequence and $n \in \{ 1 , \ldots , N \}$ indexes a frame; for benchmark clips, the nominal sequence length is 81.

![](images/903ebc69e5b7ea2858e0b90e6a68f4c4047b29f18f24abe54d5ec7c246a8c448.jpg)  
(a) Style prototype visualization.

![](images/5a0b98957683ea7f10a3da68d36b4e740b4ad37a874aec26968ab549b3401c60.jpg)  
(b) Nearest-prototype classification.  
Figure 10 SPS prototype visualization and validation. (a) t-SNE visualization of 80 CSD prototypes across 18 styles. (b) Row-normalized confusion matrix under leave-one-source-out validation, achieving 93.61% macro-averaged accuracy.

Styles and comparison coverage. In the quantitative and perceptual comparison, all methods receive the same input videos for the same target style and follow the same frame-alignment protocol for shared cases. Table 4 lists the 18-style vocabulary and the ten-style comparison subset (bench10). Each object-mask unit supports foreground and background editing, so the complete ten-style grid contains $8 4 \times 1 0 \times 2 = 1 , 6 8 0$ results (168 cases, 10 styles).

## C.2 Regional Style Metrics

## C.2.1 Localized Region Construction

Both SPS and regional CLIPT operate on a localized region rather than the full output frame. Let Y be an output video and $\mathbf { M } _ { \mathrm { o b j } }$ its tracked foreground-object mask. VideoMaMa [34] maskguided matting of the output yields a foreground alpha video $\mathbf { A _ { \mathrm { o u t } } } .$ , allowing localization to follow stylization-induced contour changes. For foreground stylization, each frame is cropped to the alpha-support bounding box with a 2% margin. For background stylization, the full frame is retained while foreground-object tokens are excluded from the region representation, avoiding an artificial filled foreground area. Thus, SPS and regional CLIPT share the same localization rule and difer only in how the localized region is scored. We denote the resulting CSD region descriptor by $\psi _ { \omega } ( \mathbf { Y } _ { n } , \mathbf { A } _ { \mathrm { o u t } , n } )$ and the corresponding CLIP image embedding by $\chi _ { \omega } ( \mathbf { Y } _ { n } , \mathbf { A } _ { \mathrm { o u t } , n } )$ , where $\omega \in \{ \mathrm { f g } , \mathrm { b g } \}$ specifies the editing direction.

## C.2.2 Style Prototype Similarity (SPS)

SPS evaluates whether an edited region resembles characteristic visual examples of the requested style. Unlike text–image similarity, it compares outputs with a class-specific bank of visual prototypes. SPS is the prototype-based scoring scheme built on the Contrastive Style Descriptors (CSD) [31].

Prototype construction. We collect globally stylized source and object-removed videos that pass the data pipeline’s quality gates. Every frame is encoded into a 768-dimensional CSD descriptor.

Table 4 Canonical style phrases for regional CLIPT. A check mark identifies the ten-style comparison subset.
<table><tr><td>Style category</td><td>Canonical phrase</td><td>in bench10</td></tr><tr><td>Children&#x27;s Book</td><td>children&#x27;s book illustration</td><td>√</td></tr><tr><td>Claymation</td><td>claymation</td><td>√</td></tr><tr><td>Comic Book</td><td>comic book</td><td></td></tr><tr><td>Disney</td><td>Disney animation</td><td></td></tr><tr><td>Ghibli</td><td>Studio Ghibli anime</td><td>√</td></tr><tr><td>Japanese Anime</td><td>Japanese anime</td><td></td></tr><tr><td>LEGO</td><td>LEGO brick</td><td></td></tr><tr><td>Low Poly Oil Painting</td><td>low-poly 3D</td><td>√ √</td></tr><tr><td></td><td>oil painting</td><td>√</td></tr><tr><td>Papercut</td><td>papercut</td><td></td></tr><tr><td>Pencil Sketch</td><td>pencil sketch Pixar 3D animation</td><td>√</td></tr><tr><td>Pixar Pixel Art</td><td></td><td>√</td></tr><tr><td></td><td>pixel art</td><td></td></tr><tr><td>Pop Art</td><td>pop art The Simpsons cartoon</td><td></td></tr><tr><td>Simpsons Ukiyo-e</td><td>Ukiyo-e woodblock print</td><td>√</td></tr><tr><td>Vector Animation</td><td>flat vector animation</td><td></td></tr><tr><td>Watercolor</td><td>watercolor painting</td><td>√</td></tr></table>

Within each style, we apply �-means clustering [41] to the CSD descriptors. The candidate set $C _ { s } \in \{ 3 , 5 \}$ is fixed in advance and shared across styles; for each candidate, we compute the mean silhouette score [42] over the descriptors and retain the partition with the larger score. This data-driven criterion adapts the prototype bank to within-style variation while keeping model selection deliberately small. The resulting bank contains 80 centroids across 18 styles. Multiple prototypes represent within-style variation rather than forcing distinct rendering modes into a single center. Fig. 10a visualizes the prototype bank with t-SNE [43].

Frame and video scores. For target style � and its prototype set $\mathcal { P } _ { s }$ , we compute

$$
\begin{array} { r l } & { \rho _ { n , s } = \underset { \pmb { \nu } \in \mathcal { P } _ { s } } { \operatorname* { m a x } } \cos \left( \psi _ { \omega } ( \mathbf { Y } _ { n } , \mathbf { A } _ { \mathrm { o u t } , n } ) , \pmb { \nu } \right) , } \\ & { \mathrm { S P S } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \rho _ { n , s } . } \end{array}\tag{14}
$$

Frame scores are first averaged within each evaluated video; dataset scores then average the corresponding video-level scores, with foreground and background editing reported separately.

Discriminative validation. To test whether SPS distinguishes the 18 style categories beyond source-specific appearance, we perform leave-one-source-out validation: all samples derived from the held-out source video are excluded when constructing its reference prototypes. Each frame is assigned the style of its nearest prototype by cosine similarity, and frame predictions are aggregated by video-level majority vote. This yields a macro-averaged classification accuracy of 93.61%. Fig. 10b shows the class-wise confusion matrix, complementing the prototype visualization in Fig. 10a. Together, these analyses characterize the prototype organization and support SPS’s ability to distinguish the evaluated style categories.

## C.2.3 Regional CLIPT

Regional CLIPT measures alignment with the canonical style phrase, complementing SPS’s visual prototypes. Using the same localized region construction as SPS, it extracts image and text features with CLIP ViT-L/14 [37]. Let $\chi _ { \omega } ( \mathbf { Y } _ { n } , \mathbf { A } _ { \mathrm { o u t } , n } )$ denote the region image embedding and let $\eta _ { s }$ denote the CLIP text embedding of the canonical phrase for style � in Table 4. Then

$$
\begin{array} { r l } & { \mathrm { C L I P T } _ { \mathrm { r e g } , n } = \cos \left( \chi _ { \omega } ( \mathbf { Y } _ { n } , \mathbf { A } _ { \mathrm { o u t } , n } ) , \eta _ { s } \right) , } \\ & { \mathrm { C L I P T } _ { \mathrm { r e g } } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathrm { C L I P T } _ { \mathrm { r e g } , n } . } \end{array}\tag{15}
$$

As with SPS, video-level means are averaged before dataset aggregation. The alpha video defines region selection; it is not itself the RGB image passed to the image encoder. SPS and regional CLIPT therefore difer in their scoring encoders and similarity targets, not in their spatial localization rule.

## C.3 Temporal and Preservation Metrics

## C.3.1 Temporal Consistency

Following AVID [39], TC (Temporal Consistency) uses full-frame image embeddings ${ \mathfrak { z } } _ { n }$ from a CLIP ViT-H/14 tower, a separate and larger encoder than the ViT-L/14 used for regional CLIPT:

$$
\mathrm { T C } = \frac { 1 } { N - 1 } \sum _ { n = 1 } ^ { N - 1 } \cos ( z _ { n } , z _ { n + 1 } ) , \quad z _ { n } = \mathrm { C L I P } _ { \mathrm { i m g } } ( \mathbf { Y } _ { n } ) .\tag{16}
$$

TC summarizes adjacent-frame feature consistency over the aligned output sequence. It is a global measure and does not replace regional style or preservation scores.

## C.3.2 Non-Target Preservation

Preservation is measured with regional PSNR [38] outside the intended edit. For foreground stylization, PSNR is computed on the preserved background region 1 − M; for background stylization, it is computed on the preserved foreground region M. In both cases, the valid region is derived from the foreground alpha in $\mathbf { A _ { \mathrm { o u t } } }$ on an 8-bit scale: the preserved background core comprises pixels with alpha at most 5, whereas the preserved foreground core comprises pixels with alpha above 200. The anti-aliased transition band in between is excluded.

We compare aligned output and source frames after resizing the source to the output resolution and use the common valid temporal range. For each frame, the RGB reconstruction error is averaged over the valid non-edit pixels and channels, and the resulting errors are aggregated over valid frames before conversion to PSNR. Frames with insuficient valid pixels are skipped; when an alpha estimate is empty, the original mask is used as a fallback.

## C.4 VLM Evaluation

We use Gemini 3.6 Flash to complement the objective metrics with full-video VLM assessment of two separate quantities: Style (edit\_region\_style) and Preserve (non\_edit\_preservation). Each is an integer from 0 to 5. The scores are not merged, so strong stylization cannot compensate for damage to non-target content, or vice versa. This evaluation is distinct from the GPT-5.5 data-construction filter, which uses five dimensions on a 1–5 scale.

Inputs and execution. The judge receives the source video first and the edited result second, aligned frame by frame and tagged SRC and RESULT. A red contour marks the foreground object in the source. For foreground editing, its interior is the target; for background editing, the exterior is the target. The text instruction explicitly identifies the direction, and the contour and display tags are not scored. We use the English system prompt and rubric below with temperature zero. Responses are parsed as JSON; if parsing fails, the same request is retried once.

Scoring anchors. Style jointly assesses style correctness, coverage of the edit region, and crossframe stability. A score of 5 requires all three, 3 indicates visible but imperfect style, incomplete coverage, or occasional flicker, and 0 indicates absent or incorrect style or severe instability. Preserve assesses source fidelity and temporal stability in the non-edit region: 5 indicates a close, stable match, 3 allows minor changes or occasional jitter, and 0 indicates substantial repainting or severe drift. The two video-level scores are aggregated separately across cases.

Prompt templates. The following template (Listing 2) retains the scoring rubric while making the edit direction explicit. For foreground editing, {target\_region} is “the foreground object inside the red outline”; for background editing, it is “the background outside the red outline.” The {region\_description} field names both the edit and non-edit regions accordingly, and {style} is replaced by the requested style.

Listing 2 VLM-as-a-judge prompt in our evaluation.  
You are a professional video local-editing judge. This review contains two videos: the first is   
the source video (the foreground object is outlined in red, with an "SRC" tag in the top-left   
corner), and the second is the edited result video (tagged "RESULT" in the top-left corner). The   
red outline and any display differences caused by it as a marker are not scored; judge only the   
actual visual changes inside the outlined region and in the rest of the frame. Watch both videos   
in full and compare them segment by segment before scoring. Score strictly based only on what is   
visible; do not deduct for video encoding noise, and do not let the order of the videos bias you.   
Output only one line of valid JSON, with no code fences or any other text.   
[Task] Local video stylization review   
- Edit request: convert only {target\_region} in the source video to the {style} style   
- Region: {region\_description}   
[Scoring dimensions] Each dimension is an integer from 0 to 5; score ONLY the following two   
dimensions:   
1. edit\_region\_style -- quality of the stylization in the edit region (combining style   
correctness, coverage completeness, and cross-frame stability):   
(i) Stylization: does it present a pure, typical {style} style   
(ii) Coverage: does the style evenly cover the entire edit region (no gaps, no missed spots)   
(iii) Temporal stability: the appearance of the edit region stays stable and coherent over   
time, with no flicker and no frame-by-frame style drift   
5 = a pure, typical {style} style evenly covering the entire edit region, with no noticeable   
abnormal flicker or style drift throughout; if any aspect clearly fails, a 5 must not be given   
3 = stylization is present but not faithful to the style, or coverage is incomplete / parts of   
the region are missed, or occasional flicker   
0 = almost no stylization, the style is completely wrong, or the edit region flickers or   
drifts severely   
2. non\_edit\_preservation -- preservation of the non-edit region (combining):   
(i) Consistency with the source: at every corresponding moment the non-edit region keeps the   
content, position, texture, and colors of the source video as closely as possible   
(ii) Temporal consistency: the non-edit region stays stable over time, with no frame-by-frame   
jitter and no color-shift drift   
5 = the non-edit region matches the source video closely and is temporally stable throughout   
3 = only minor changes, slight color shifts, or local differences, and most of the non-edit   
region still matches the corresponding source frames; or only occasional jitter   
0 = the content, texture, colors, or details of the non-edit region are visibly repainted or   
stylized, or severe frame-by-frame jitter appears   
[Output] Output only the following JSON and nothing else:   
{"edit\_region\_style":n,"non\_edit\_preservation":n,"comment":"summarize the most prominent issue in   
at most 20 English words; write none if there is no obvious issue"}

## D Model Reproduction Details

This appendix specifies the training configuration of the reported GRNStyle-2B model, defines the checkpoint validation protocol, and discusses the observed validation trends.

## D.1 Training Setup

Table 5 summarizes the main training configuration, which uses 100K paired examples across 18 style categories. The model is initialized from a pre-trained GRNEdit [23] general-purpose video editor and fully fine-tunes the GRN-2B backbone and conditioning branches, while the video tokenizer and T5 encoder remain frozen. This initialization provides a strong global stylization prior, so optimization primarily adapts the model to mask-guided local control rather than learning stylization from scratch.

Training applies the all-token bit cross-entropy over the full video. With probability 0.50, the style-specific instruction is replaced by a generic mask-following instruction, encouraging the model to read style identity from the polar style-aware mask while retaining text guidance. The training loader consumes 61-frame clips at 20 fps.

Table 5 Main training configuration for GRNStyle-2B.
<table><tr><td>Hyperparameter</td><td>Setting</td></tr><tr><td colspan="2">Model and representations</td></tr><tr><td>Backbone</td><td>GRN-2B</td></tr><tr><td>Initialization</td><td>Pre-trained GRNEdit</td></tr><tr><td>Trainable modules</td><td>GRN-2B backbone, conditioning branches</td></tr><tr><td>Transformer blocks / hidden width / heads</td><td>28 / 2,304 / 18</td></tr><tr><td colspan="2">Data and conditioning</td></tr><tr><td>Training pairs / style categories</td><td>100,000 / 18</td></tr><tr><td>Training clip length / sampling rate</td><td>61 / 20 fps</td></tr><tr><td>Aspect-ratio buckets</td><td>15 buckets, H/W ∈ [0.333, 3.0]</td></tr><tr><td>Class-aware polar mask</td><td>Enabled</td></tr><tr><td>Generic-prompt replacement probability</td><td>0.50</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer / momentum coefficients</td><td>AdamW / (0.9,0.999)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping (global norm)</td><td>0.1</td></tr><tr><td>Training objective</td><td>All-token bit cross-entropy</td></tr><tr><td>Mixed precision / TF32</td><td>BF16 /enabled</td></tr><tr><td>Training duration</td><td>10 epochs</td></tr><tr><td colspan="2">Learning-rate schedule</td></tr><tr><td>Initial / peak / minimum learning rate</td><td> $1 0 ^ { - 5 } / 4 { \times } 1 0 ^ { - 5 } / 2 { \times } 1 0 ^ { - 5 }$ </td></tr><tr><td>Linear warmup</td><td>100 optimizer updates</td></tr><tr><td colspan="2">Distributed execution</td></tr><tr><td>Distributed ranks / sharding</td><td></td></tr><tr><td>Random seed</td><td>8 / FSDP full sharding 1,789,151,675</td></tr></table>

Evidence injection. GRNStyle injects the source/mask residual at the input of the 1st, 13th, and 25th transformer blocks, covering early, middle, and late stages; no residual is injected at the other block boundaries. This conditioning path adds 8.50M parameters (∼0.43% of the 2B backbone).

![](images/7603118094a643f020b5a06a92d0f4261c7d48b51d2405d9d04d9ff4d59419b5.jpg)  
Figure 11 Validation stylization quality measured by mean foreground/background SPS.

![](images/b1f87e76567dbdb56811610235c5ed5d909240b647de9e31dd2ab96577bba5bc.jpg)  
Figure 12 Validation preservation quality measured by mean foreground/background PSNR on non-target regions.

Table 6 Inference cost of the two task-specific models in Table 1. Lower values are better for both metrics.
<table><tr><td>Model</td><td>Latency (s)↓</td><td>GPU memory (GB)↓</td></tr><tr><td>VACE-1.3B (our FT)</td><td>39.2</td><td>23.7</td></tr><tr><td>GRNStyle-2B (ours)</td><td>34.5</td><td>27.4</td></tr></table>

We monitor local-control quality with two region-wise measures: SPS on the edited region and PSNR on the preserved region. At each checkpoint, we compute foreground and background scores separately and combine them with an unweighted arithmetic mean. We evaluate nine checkpoints during training. These measures complement full-video bit-prediction statistics, which aggregate edited and preserved tokens and can therefore obscure localized failures.

Figures 11 and 12 report SPS and PSNR on the validation set. Markers denote the nine evaluated checkpoints, and the connecting lines are not smoothed. From update 2,000 to 16,000, mean SPS increases from 0.512 to 0.554, and mean PSNR from 23.01 to 26.00 dB. Both metrics change more rapidly before update 10,000 and more slowly thereafter, suggesting diminishing returns for local-control quality.

## E Inference Eficiency

Table 1 in the main paper contains two task-specific configurations developed in this work: the fine-tuned VACE-1.3B and GRNStyle-2B. Table 6 compares their inference cost to characterize the practical deployment profile of our approach. GRNStyle-2B requires 34.5 s per inference and 27.4 GB of GPU memory, compared with 39.2 s and 23.7 GB for VACE-1.3B (our FT). Relative to this 1.3B configuration, GRNStyle-2B reduces latency by 12.0% while increasing memory by 15.6%; the additional 3.7 GB therefore yields a 4.7 s reduction in inference time. These results indicate that competitive local stylization can be achieved with a 2B backbone, without relying on a 14B foundation model. Both configurations fit within a single-device memory footprint, and GRNStyle-2B achieves lower latency than the fine-tuned 1.3B variant, supporting practical deployment without a multi-device serving setup.

Table 7 Global stylization comparison. All methods use the same 80 benchmark videos and ten-style subset. <sup>†</sup>EchoStyle supports five styles and is evaluated on those only. Best and secondbest results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Params</td><td>Venue</td><td>CSD↑</td><td>CLIPT↑</td><td>TC↑</td><td>Style↑</td></tr><tr><td>StyleMaster [6]</td><td>1.3B</td><td>CVPR &#x27;25</td><td>0.428</td><td>0.189</td><td>0.962</td><td>2.12</td></tr><tr><td>Ditto [21]</td><td>14B</td><td>CVPR &#x27;26</td><td>0.649</td><td>0.207</td><td>0.976</td><td>4.73</td></tr><tr><td>EchoStyle† [44]</td><td>14B</td><td>ECCV &#x27;26</td><td>0.406</td><td>0.190</td><td>0.977</td><td>4.14</td></tr><tr><td>PickStyle [27]</td><td>14B</td><td>ECCV ’26</td><td>0.388</td><td>0.182</td><td>0.984</td><td>3.04</td></tr><tr><td>GRNStyle-2B</td><td>2B</td><td></td><td>0.683</td><td>0.190</td><td>0.977</td><td>4.67</td></tr></table>

Table 8 Mask perturbation sensitivity. NONE denotes the unperturbed baseline. All variants use the same source videos, prompts, seeds, and sampling configuration; only the input mask is perturbed. Subscripts denote the signed change from NONE; changes are computed from unrounded values.
<table><tr><td>Perturbation</td><td>SPS-fg↑</td><td>SPS-bg↑</td><td>CLIPT-fg↑</td><td>CLIPT-bg↑</td><td>PSNR-fg↑</td><td>PSNR-bg↑</td><td>TC↑</td></tr><tr><td>NONE</td><td>0.388</td><td>0.677</td><td>0.154</td><td>0.188</td><td>26.6</td><td>26.3</td><td>0.972</td></tr><tr><td colspan="8">Light</td></tr><tr><td>Boundary</td><td>0.388+0.000</td><td>0.676-0.001</td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 8 7 _ { - 0 . 0 0 1 }$ </td><td> $2 6 . 5 _ { - 0 . 0 5 }$ </td><td> $2 6 . 0 _ { - 0 . 2 7 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Dilation</td><td>0.394+0.006</td><td>0.677+0.000</td><td> $0 . 1 5 5 _ { + 0 . 0 0 1 }$ </td><td>0.187-0.001</td><td> $2 6 . 5 _ { - 0 . 0 4 }$ </td><td> $2 5 . 8 _ { - 0 . 4 9 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Erosion</td><td>0.382-0.006</td><td>0.676-0.001</td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td>0.187-0.001</td><td> $2 6 . 6 _ { + 0 . 0 6 }$ </td><td> $2 6 . 5 _ { + 0 . 2 9 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Temporal</td><td> $0 . 3 8 8 _ { + 0 . 0 0 0 }$ </td><td> $0 . 6 7 7 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 8 8 _ { + 0 . 0 0 0 }$ </td><td> $2 6 . 6 _ { - 0 . 0 1 }$ </td><td> $2 6 . 2 _ { - 0 . 0 7 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Dropout</td><td> $0 . 3 8 2 _ { - 0 . 0 0 6 }$ </td><td> $0 . 6 5 5 _ { - 0 . 0 2 2 }$ </td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 8 3 _ { - 0 . 0 0 5 }$ </td><td> $2 6 . 6 _ { + 0 . 0 3 }$ </td><td> $2 6 . 3 _ { + 0 . 0 5 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td colspan="8">Strong</td></tr><tr><td>Boundary</td><td> $0 . 3 8 8 _ { + 0 . 0 0 0 }$ </td><td>0.676-0.001</td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 8 8 _ { + 0 . 0 0 0 }$ </td><td>26.5-0.05</td><td>25.9-0.34</td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Dilation</td><td> $0 . 4 0 3 _ { + 0 . 0 1 5 }$ </td><td> $0 . 6 8 1 _ { + 0 . 0 0 4 }$ </td><td> $0 . 1 5 6 _ { + 0 . 0 0 2 }$ </td><td> $0 . 1 8 8 _ { + 0 . 0 0 0 }$ </td><td> $2 6 . 2 _ { - 0 . 3 2 }$ </td><td> $2 2 . 2 _ { - 4 . 0 1 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Erosion</td><td> $0 . 3 6 5 _ { - 0 . 0 2 3 }$ </td><td> $0 . 6 7 5 _ { - 0 . 0 0 2 }$ </td><td> $0 . 1 5 3 _ { - 0 . 0 0 1 }$ </td><td> $0 . 1 8 8 _ { + 0 . 0 0 0 }$ </td><td> $2 6 . 8 _ { + 0 . 2 7 }$ </td><td> $2 6 . 8 _ { + 0 . 5 9 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Temporal</td><td> $0 . 3 8 8 _ { + 0 . 0 0 0 }$ </td><td> $0 . 6 7 7 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 5 4 _ { + 0 . 0 0 0 }$ </td><td> $0 . 1 8 8 _ { + 0 . 0 0 0 }$ </td><td> $2 6 . 5 _ { - 0 . 0 7 }$ </td><td> $2 5 . 9 _ { - 0 . 3 5 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr><tr><td>Dropout</td><td> $0 . 3 6 8 _ { - 0 . 0 2 0 }$ </td><td> $0 . 5 9 9 _ { - 0 . 0 7 8 }$ </td><td> $0 . 1 5 2 _ { - 0 . 0 0 2 }$ </td><td> $0 . 1 7 6 _ { - 0 . 0 1 2 }$ </td><td> $2 6 . 7 _ { + 0 . 1 6 }$ </td><td> $2 6 . 4 _ { + 0 . 1 3 }$ </td><td> $0 . 9 7 2 _ { + 0 . 0 0 0 }$ </td></tr></table>

## F Global Stylization Comparison

To test whether GRNStyle extends beyond local editing, we set the editing mask to all ones and apply the same checkpoint globally, without retraining. We compare with existing global video stylization methods (including StyleMaster [6], Ditto [21], EchoStyle [44], and PickStyle [27]) on the 80 benchmark videos and the same ten-style subset used in the main comparison. All methods follow the same frame-alignment and metric implementations. EchoStyle supports only five of the ten styles and is therefore evaluated on those five.

Table 7 reports CSD, CLIPT, TC, and VLM-based Style score. GRNStyle obtains the highest CSD (0.683) and the second-highest VLM Style score (4.67). Its CLIPT (0.190) matches EchoStyle and is close to the best result (0.207), while its TC (0.977) remains competitive with the strongest methods. Notably, GRNStyle outperforms the 14B methods on CSD despite using a 2B backbone, indicating that our style-aware conditioning transfers from local to global stylization without a separate global model.

## G Robustness to Imperfect Masks

We evaluate the robustness of GRNStyle to imperfect user-provided masks while keeping the source videos, style prompts, random seeds, guidance scale, and sampling configuration fixed. Starting from a 40-case subset of the benchmark and ten styles, we obtain 400 video–style evaluations for each variant.

We consider five perturbation categories and evaluate each at light and strong severity levels:

• Boundary noise: pixels within a six-pixel band around the mask contour are independently flipped with probability 0.15 (light) or 0.35 (strong) per frame.

• Dilation: the white edit region is expanded by a fixed radius of 3 (light) or 9 (strong) pixels.

• Erosion: the white edit region is contracted by a fixed radius of 3 (light) or 9 (strong) pixels.

• Temporal jitter: every frame receives an independently sampled signed radius in [−3, 3] (light) or [−9, 9] (strong) pixels.

• Frame dropout: the entire mask is cleared in approximately 10% (light) or 30% (strong) of frames.

Radii are specified at a 480p short side and scaled to the actual input resolution. The perturbed mask replaces the original mask in both the polar style encoding and the geometric support path; no retraining or parameter update is involved.

Table 8 reports the results. Under light perturbations, all metrics remain close to the unperturbed baseline: absolute changes are within 0.006 for SPS and CLIPT and within 0.49 dB for PSNR, while TC remains at 0.972. Under strong perturbations, erosion and dropout produce the clearest reduction in stylization quality. Strong erosion lowers foreground SPS from 0.388 to 0.365, and strong dropout reduces background SPS from 0.677 to 0.599 and background CLIPT from 0.188 to 0.176. In contrast, strong dilation leaves the style-related metrics stable or slightly higher, but decreases background PSNR from 26.3 to 22.2 dB because the expanded edit region inevitably modifies content outside the original preserved area; its remaining metrics change little. Overall, GRNStyle remains robust to imperfect masks: style fidelity is largely preserved, and the remaining degradation is concentrated in the metric directly afected by each geometric perturbation.

## H Limitations and Future Work

Our current system has three notable limitations. First, temporal consistency can degrade for fast-moving objects, especially near motion boundaries. Rapid motion makes it dificult to maintain stable region alignment and coherent stylization across adjacent frames, which may lead to local flicker or inconsistent boundaries. Improving motion-aware conditioning and boundary tracking is an important direction for future work.

Second, the data-construction pipeline relies on an object-removal model that may also remove the object’s cast shadow. As a result, some paired targets lack the original shadow even when it should remain semantically consistent with the scene. Training on such pairs can encourage the stylization model to erase shadows, producing visually unnatural results. Future work should improve shadow-aware data construction and incorporate explicit supervision for preserving scene-consistent shadows.

Third, our evaluation suite does not include a boundary-adherence metric, although boundary accuracy is an important aspect of local editing quality. Such a metric is dificult to define in practice: some styles intentionally expand the target object (e.g., Simpsons-style stylization) while others shrink it, so no single fixed boundary can serve as the ground truth for all outputs. Future work should develop style-aware boundary evaluation that distinguishes intended shape deformation from mask leakage and inadvertent boundary violations.

## I Additional Qualitative Results

We apply GRNStyle to foreground and background regions in the LociArt Benchmark with diverse target styles. Figs. 13 to 17 show frames 15, 30, and 45 of each generated video, illustrating consistent regional stylization and preservation of surrounding content over time.

![](images/73221071d2fd33a25a4e56eeb74c7bf8bcbe4e3754d9881c3d7d9da60a250027.jpg)  
Figure 13 Additional qualitative results of GRNStyle. The 15th, 30th, and 45th frames of the generated videos are shown for each case.

![](images/976066a210855ffc76dfa18785f44b3a5004912caafe541c3a314a703cb0452f.jpg)  
Figure 14 Additional qualitative results of GRNStyle. The 15th, 30th, and 45th frames of the generated videos are shown for each case.

![](images/5a8a4640b35755af5da1f484b0286fdcfd7b14fbbd3af1f03557a18132dd4b24.jpg)  
Figure 15 Additional qualitative results of GRNStyle. The 15th, 30th, and 45th frames of the generated videos are shown for each case.

![](images/8c55a4d0c3bf6ce17a174a9f1f19e96d7333fc7014225559fc1f5f6b8ab4859d.jpg)  
Figure 16 Additional qualitative results of GRNStyle. The 15th, 30th, and 45th frames of the generated videos are shown for each case.

![](images/6fdd94e2cc6530a3a41e1cc0ff9b95bfbb47761711dd12fb9ee0e2a5dce60273.jpg)  
Figure 17 Additional qualitative results of GRNStyle. The 15th, 30th, and 45th frames of the generated videos are shown for each case.