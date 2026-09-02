Article

# IT-TextFusion: Iterative Text-Image Interaction with Text-Guided Residual Refinement for Degradation-Aware Image Fusion

![](images/eeb13fb4c65045d4c15419ddec1f62d04967d03368eaa47a3836d94ef1954ea4.jpg)

Siyang Liu<sup>2</sup>, Peiyi Zhou<sup>2</sup>, Tianle Jin<sup>2</sup>, Rongrong Bian<sup>2</sup>, Zheke Jin<sup>2</sup>, and Mengze Gao<sup>1\*</sup>

<sup>1</sup> School of Automation, Southeast University, Nanjing, China

<sup>2</sup> Chair of Robotics, Artificial Intelligence and Real-time Systems, Technical University of Munich, Munich, Germany

\* Corresponding author: Mengze Gao.

## Highlights:

• Hierarchical text conditioning guides fusion across multiple decoder stages.

• Degradation-aware prompts provide scenario-specific global semantic guidance.

• Cross-Gate Fusion adaptively aggregates visible and infrared features at multiple hierarchical scales.

• Text-guided residual refinement improves high-resolution feature reconstruction under degradations.

Abstract: Text-guided image fusion has recently emerged as an effective paradigm for integrating multimodal information while enabling flexible and task-oriented fusion control. However, existing text-guided fusion methods often rely on shallow semantic-visual interaction and limited attention mechanisms, which restrict their ability to robustly handle complex degradations and fully exploit textual guidance. In this paper, we propose an iterative text-guided image fusion framework that incorporates text-conditioned feature interaction across multiple fusion and refinement stages. The proposed method integrates deepestlevel Cross-Attention, multi-scale Cross-Gate Fusion, and stage-specific text-conditioned modulation, allowing the global text embedding to condition hierarchical feature fusion and residual refinement. By repeatedly injecting the pooled text embedding across hierarchical decoder and refinement stages, the proposed framework provides degradation-aware global semantic conditioning while preserving complementary information from the visible and infrared modalities. Experiments on several benchmark datasets show that the proposed method improves several information-preservation and perceptual-quality metrics, while exhibiting metric-dependent trade-offs on some datasets.

Keywords: infrared-visible image fusion; text-guided image fusion; iterative text-image interaction; degradation-aware fusion; text-guided residual refinement

## 1. Introduction

In recent years, autonomous driving has attracted increasing attention due to rapid progress in sensing, learning, and system integration technologies. Autonomous driving systems aim to reduce driver workload, improve road safety, and mitigate traffic congestion. Among the three core components of autonomous driving systems–perception, prediction, and motion planning–perception plays a fundamental role by providing accurate environmental understanding to support reliable decision-making in advanced driver-assistance systems (ADAS). Core perception tasks such as semantic segmentation [1] and object detection [2] rely heavily on visual sensing, with cameras remaining the most widely deployed sensors in autonomous vehicles due to their rich spatial and appearance information [3,4]. Cao et al. [5] further investigate fisheye object detection for autonomous driving, introducing a feature-aligned pyramid module and a location-aligned detection head.

Frame-based RGB images generally provide high visual fidelity, and recent advances in deep learning and end-to-end training have enabled strong perception performance under favorable conditions. However, perception systems based on a single sensing modality inherently suffer from limitations in robustness and reliability when operating in challenging environments. Conventional cameras are particularly vulnerable to adverse conditions such as low illumination, overexposure, motion blur, and sensor noise. Furthermore, complex traffic scenes frequently contain occlusions and large illumination variations, which further reduce the reliability of camera-based perception. These challenges significantly degrade the performance of downstream perception tasks, including object detection and semantic segmentation, highlighting the necessity for more robust sensing strategies. As a result, multi-modal sensor fusion has emerged as a promising approach to alleviate these limitations by leveraging complementary information from different sensing modalities [2,3,6,7].

While conventional RGB cameras remain strong standalone sensors, combining multiple sensing modalities enables a more comprehensive and resilient representation of the environment. Depth sensors provide geometric structure, event cameras excel in high-speed and high-dynamic-range scenarios [2,4,8], and LiDAR systems offer accurate spatial and angular measurements [4]. Infrared cameras, in particular, capture thermal radiation and are highly effective for nighttime perception and for detecting heat-emitting objects such as pedestrians and vehicles [9]. Although some approaches explore fusion across multiple modalities simultaneously [10], this work focuses on the fusion of visible and infrared imagery, which offers strong complementarity for safety-critical perception tasks.

A central challenge in multi-modal perception lies in determining how to effectively integrate heterogeneous sensor information, a problem for which no trivial solution exists [11]. Existing fusion strategies are typically categorized into early, middle, and late fusion. Early fusion combines raw sensor data at the input level but is highly sensitive to noise. Late fusion integrates modality-specific outputs at the decision level, often through algebraic or probabilistic methods, but fails to capture fine-grained cross-modal interactions at lower feature levels [11]. Middle fusion, which merges modality-specific features within intermediate network layers, has therefore become a widely adopted strategy. For example, NeuroGrasp [12] combines RGB and event information in a multi-modal neural network for robotic grasp pose estimation. Nonetheless, effectively exploiting informative signals from multiple modalities while suppressing noise and inconsistencies remains a fundamental challenge in multi-modal fusion. In this work, we adopt a middle-fusion strategy and aim to address this challenge by leveraging semantic guidance from textual prompts under complex degradation conditions.

![](images/7642a2ed6cec5578893fa642bc4e5e2211bd368f56744abd9da8af1f80ac37fd.jpg)  
Figure 1. Comparison of image fusion frameworks. (a) Previous frameworks: (i) purely visual fusion without textual guidance; and (ii) text-image fusion using generic text that provides only limited semantic guidance. (b) The proposed framework, in which degradation-aware text guides both iterative text-image interaction and the subsequent text-guided residual refinement.

Recent progress in vision-language models has highlighted the effectiveness of leveraging textual semantics to guide visual representation learning. Large-scale vision-language models trained on diverse image-text pairs exhibit strong cross-modal alignment and semantic generalization capabilities, providing transferable semantic representations for downstream visual tasks [9]. In particular, CLIP [13] demonstrates remarkable robustness and transferability, providing rich semantic embeddings that have been successfully adopted in various vision applications, including image fusion [14]. Building upon these advances, recent text-guided fusion frameworks such as Text-IF [9], TextFusion [15], and TITFormer [16] incorporate linguistic descriptions into the fusion pipeline, allowing semantic prompts to regulate feature aggregation and refinement. These methods demonstrate that textual representations can provide additional semantic conditions for feature aggregation and refinement beyond conventional purely visual fusion strategies.

Nevertheless, purely visual fusion remains dominant in many earlier approaches. As illustrated in Fig. 1, representative methods such as [17–19] rely exclusively on image-based fusion mechanisms without leveraging auxiliary semantic information. While effective under relatively clean imaging conditions, their robustness is often limited when facing complex real-world degradations, including low illumination, reduced contrast, and sensor noise. Despite the encouraging developments in text-guided fusion, several challenges remain insufficiently addressed. Textual information is commonly introduced as a high-level conditional signal, without explicitly modeling fine-grained semantic structures or degradation characteristics. Consequently, the interaction between linguistic cues and visual features often remains relatively shallow, restricting the model’s ability to exploit textual guidance across different feature resolutions. Moreover, many existing approaches rely on limited cross-modal interaction, where text features influence image representations through a single modulation stage or simple affine transformations. Such designs may restrict the effect of the global text condition on hierarchical feature fusion, particularly when different degradations affect visual information at different feature resolutions. In addition, although refinement modules are frequently employed, semantic guidance is not always propagated consistently throughout the correction stages. This may lead to incomplete suppression of residual artifacts under severe or spatially heterogeneous degradations.

To overcome these limitations, we propose a unified text-guided fusion framework that strengthens hierarchical text-conditioned feature fusion and residual refinement. The proposed approach improves semantic specificity through degradation-aware text engineering, which generates more informative and task-relevant linguistic guidance. In addition, the Cross-Gate Fusion module is introduced to regulate intermodal feature aggregation by adaptively weighting the reliability of visual responses under degradation. To maintain textual conditioning across hierarchical scales, we design an Iterative Text-Image Interaction Module (ITIM), which repeatedly injects the pooled global text embedding into decoder features through independently parameterized stage-specific modulation functions. Furthermore, a Text-Guided Residual Refinement Module (TG-RRM) is developed to apply global text-conditioned modulation within the multiscale refinement process, supporting high-resolution residual feature correction before final reconstruction. Overall, our contributions can be summarized as follows:

• We improve two complementary aspects of the fusion pipeline: degradation-aware text engineering increases the semantic specificity of textual guidance, while the Cross-Gate Fusion module improves multi-scale feature aggregation by adaptively suppressing unreliable visual responses.

• We propose ITIM, which incorporates textual guidance into visual features across multiple fusion stages, providing hierarchical global text conditioning throughout the fusion process.

• We propose TG-RRM, which integrates text-conditioned modulation into the refinement stage to enhance structural consistency and detail preservation.

• In addition, we evaluate the proposed method on an extended EMS dataset covering nine representative degradation types. Experimental results show improvements on selected fusion and no-reference perceptual-quality metrics, together with trade-offs in MI, $\mathrm { S S I M _ { \mathrm { s u m } } }$ , and CLIP-IQA.

## 2. Related Work

## 2.1. General Image Fusion Methods

With the success of deep convolutional networks, encoder-decoder architectures and residual learning have become the dominant paradigms for image fusion tasks [20,21]. Owing to skip connections and parallel modality-specific backbones, deep fusion models are able to effectively preserve complementary information from different sources [8,11,22]. More recently, transformer-based architectures have been introduced into image fusion to model long-range dependencies via self-attention [23]. Vision Transformer (ViT) [24] represents images as sequences of patch tokens, enabling global context modeling that has proven beneficial for fusion tasks [9,10,16,25]. CMX [4] proposes a cross-modal feature rectification module to refine RGB and arbitrary modality features before fusion, followed by cross-attention-based feature aggregation. Building upon CMX, CMNeXt [10] extends this framework to support an arbitrary number of modalities via dynamic feature selection and parallel pooling.

Alternative fusion strategies have also been explored. Adaptive Instance Normalization (AdaIN) [26] provides an efficient feature-statistics alignment operation and has inspired lightweight modulation strategies in multi-modal representation learning. In contrast, Bayesian fusion approaches [11] perform late fusion at the detection level by combining independently trained detectors through probabilistic inference, improving robustness under modality misalignment. Closely related in employing feedback mechanisms, the E2E-MFD model [6] utilizes an Object-Region-Pixel Phylogenetic Tree (ORPPT) for hierarchical, coarse-to-fine fusion. Finally, Text-IF [9] incorporates semantic text guidance into image fusion, enabling interactive and degradation-aware fusion through transformer-based cross-attention and convolutional decoding.

## 2.2. Text-Image Models

Transformer-based representation learning has significantly advanced multi-modal understanding by enabling joint modeling of visual and textual data. A major milestone in this direction is CLIP [13], which aligns images and text in a shared embedding space through large-scale contrastive pretraining, supporting zero-shot recognition and semantic retrieval. Building upon CLIP, DenseCLIP [27] extends visionlanguage pretraining to dense prediction tasks by introducing pixel-text similarity maps and context-aware prompting, improving performance in semantic segmentation and detection. Diffusion-based models such as Stable Diffusion [28] further demonstrate the effectiveness of textual conditioning for fine-grained visual control by integrating text encoders with generative architectures.

Motivated by these advances, incorporating textual semantics into image fusion has emerged as a natural extension. TextFusion [15] introduces affine text-guided modulation within a transformer-based fusion framework, enabling controllable fusion outcomes. TeRF [22] further explores region-aware fusion by integrating language commands into segmentation-guided pipelines. TITFormer [16] combines textual inputs with simulated infrared imagery for image enhancement, leveraging cross-modal attention to extract contextual features from text. Similarly, MGFusion [14] injects semantic information through CLIP-guided feature modulation, enhancing multi-modal representations without complex architectural designs. Text-IF [9] represents a unified framework that dynamically adapts fusion strategies based on semantic text inputs.

Existing text-guided fusion methods mainly differ in the location, representation, and frequency of textual conditioning. TextFusion primarily employs text-conditioned feature modulation, TeRF emphasizes region-aware language control, MGFusion uses CLIP-derived semantic features to modulate multi-modal representations, and Text-IF adapts fusion to degradation descriptions. In contrast, IT-TextFusion maintains global text conditioning across hierarchical decoding and residual feature refinement. The proposed method contributes a stage-specific textual conditioning framework that integrates multi-scale visual

fusion with high-resolution residual refinement.

## 3. Method

This section presents the overall workflow of our framework, as illustrated in Fig. 2. We first introduce Text Engineering, which constructs degradation-aware prompts to provide semantic guidance for image fusion. We then describe the Image Fusion Pipeline, including the image encoders, Cross-Attention, Cross-Gate Fusion, and the Iterative Text-Image Interaction Module (ITIM). Next, the Text Semantic Encoding and Feature Modulation subsection explains how the global text embedding is encoded and used to condition visual features at each hierarchical stage. We further present the Text-Guided Residual Refinement Module (TG-RRM), which performs text-conditioned residual feature correction before image reconstruction. Finally, the Loss Functions subsection describes the composition of the overall training objective.

## 3.1. Text Engineering

Textual guidance has recently emerged as an effective mechanism for regulating multi-modal fusion. Nevertheless, many existing text-guided fusion frameworks rely on coarse or generic task descriptions that primarily express high-level fusion objectives while failing to explicitly identify the affected modality, degradation category, or scenario-specific semantic priority. Such generic descriptions may provide insufficient degradation-specific conditioning because the same or similar prompt can be used under substantially different input conditions. Consequently, prompts that do not explicitly distinguish the affected modality and degradation type may produce similar global text embeddings, limiting their ability to differentiate different fusion scenarios.

To address this issue, we manually construct degradation-aware prompt sets offline to enrich the semantic content of the textual conditions. Instead of adopting minimal task descriptions, the prompts are designed to jointly encode the fusion objective, dominant degradation characteristics, and applicationrelevant semantic priorities. Since the prompts are organized into predefined category-specific sets, different textual conditions can be selected for different degradation scenarios without changing the fusion-network architecture. Concretely, beyond describing the fusion task itself, the textual guidance is augmented with manually specified semantic emphasis that varies according to the target scenario. In safety-critical applications such as autonomous driving, objects including vehicles, pedestrians, and road structures must remain clearly distinguishable despite adverse imaging conditions. By explicitly encoding such priorities, the textual input conveys not only what degradation is present but also which scene elements should receive greater attention during fusion. As an illustrative example, the basic task description “The task at hand is the fusion of infrared and visible light images, where visible images are affected by low light degradation” can be extended with additional semantic guidance such as “... Highlight road irregularities, such as potholes or debris, that may be obscured in darkness.” The resulting prompt therefore provides a predefined global text-conditioning signal containing more specific degradation information than a generic fusion-task description.

By exposing the model to degradation-aware and semantically enriched prompts, the fusion network receives category-specific global text-conditioning signals during training. The selected prompt is encoded by the frozen CLIP text encoder and used to condition hierarchical feature fusion and residual refinement, while the fusion-network architecture remains unchanged.

![](images/17dfda1484c3d6c81c7a828d96edd85a65e5152931312afbdec8d4798b10439e.jpg)  
Figure 2. Overall workflow of the proposed method. It consists of two main parts: a text semantic encoder that extracts a global text embedding from the detailed description produced by Text Engineering, and an image fusion pipeline comprising Cross-Attention, Cross-Gate Fusion, Self-Attention (SE-ATT), Iterative Text-Image Interaction Module (ITIM), and Text-Guided Residual Refinement Module (TG-RRM).

## 3.2. Image Fusion Pipeline

Image Encoders. The visible and infrared image encoders independently process the source visible and infrared images as inputs. To effectively capture both spatial structures and high-level semantic information, we employ Transformer-based blocks [29] as the backbone feature extractor. This design enables the encoder to learn comprehensive and discriminative representations by jointly modeling local details and long-range dependencies across different feature scales. The encoding process is formulated as follows:

$$
\begin{array} { r } { F _ { \mathrm { v i s } } = \mathcal { F } _ { \mathrm { v } } ^ { \mathrm { e } } ( I _ { \mathrm { v i s } } ) , \quad F _ { \mathrm { i r } } = \mathcal { F } _ { \mathrm { i } } ^ { \mathrm { e } } ( I _ { \mathrm { i r } } ) , } \end{array}\tag{1}
$$

where $I _ { \mathrm { v i s } } \in \mathbb { R } ^ { H \times W \times 3 }$ and $I _ { \mathrm { i r } } \in \mathbb { R } ^ { H \times W \times 3 }$ denote the visible and infrared network inputs, respectively. In the implementation, the infrared image is loaded in three-channel RGB format before being passed to the encoder. H and W represent the height and width of the input image. $\mathcal { F } _ { \mathrm { v } } ^ { \mathrm { e } }$ and $\mathcal { F } _ { \mathrm { i } } ^ { \mathrm { e } }$ denote the visible encoder and infrared encoder. $F _ { \mathrm { v i s } }$ and $F _ { \mathrm { i r } }$ collectively denote the four-level visible and infrared feature pyramids, respectively: $F _ { \mathrm { v i s } } = \{ F _ { \mathrm { v i s } } ^ { ( k ) } \} _ { k = 1 } ^ { 4 }$ and $F _ { \mathrm { i r } } = \{ F _ { \mathrm { i r } } ^ { ( k ) } \} _ { k = 1 } ^ { 4 }$ . Here, $F _ { m } ^ { ( k ) }$ denotes the feature of modality $m \in \{ \mathrm { v i s } , \mathrm { i r } \}$ at the k-th encoder level. In our framework, multi-level features are extracted at four different stages of the encoder, corresponding to hierarchical representations at progressively reduced spatial resolutions. These multi-scale features are later utilized as inputs to different stages of the fusion and decoding modules.

Cross-Attention. The Cross-Attention layer performs bidirectional cross-modal interaction at the deepest encoder level by using cross-modal affinities to aggregate modality-specific value features. Before generating the query, key, and value representations, the two features are independently normalized using 16-group GroupNorm. The normalized features are then projected by modality-specific $1 \times 1$ convolutions:

$$
[ Q _ { m } , K _ { m } , V _ { m } ] = \mathrm { R e s h a p e } _ { h } \left( \mathrm { S p l i t } _ { 3 } \left[ \mathrm { C o n v } _ { \mathrm { q k v } , m } ^ { 1 \times 1 } \left( \mathrm { G N } _ { m } \left( F _ { m } ^ { ( 4 ) } \right) \right) \right] \right) , \quad m \in \{ \mathrm { v i s } , \mathrm { i r } \} ,\tag{2}
$$

where $Q _ { m } , K _ { m } , V _ { m } \in \mathbb { R } ^ { B \times h \times d \times N _ { 4 } } , N _ { 4 } = H _ { 4 } W _ { 4 }$ , and $d = C _ { 4 } / h$ . In the specific implementation, $h = 1$ and $d = C _ { 4 }$

Subsequently, bidirectional cross-modal interaction is performed. In the visible branch, the infrared query is matched against the visible keys and values, and the attended response is projected and added back to the visible feature. Symmetrically, the visible query is used to retrieve information from the infrared keys and values for the infrared branch. The implemented Cross-Attention can therefore be written as

$$
F _ { \mathrm { v } } ^ { \mathrm { c a } } = F _ { \mathrm { v i s } } ^ { ( 4 ) } + \mathcal { P } _ { \mathrm { v } } \left[ \mathrm { S o f t m a x } \left( \frac { Q _ { \mathrm { i } } K _ { \mathrm { v } } ^ { \top } } { \sqrt { d } } \right) V _ { \mathrm { v } } \right] , \quad F _ { \mathrm { i } } ^ { \mathrm { c a } } = F _ { \mathrm { i r } } ^ { ( 4 ) } + \mathcal { P } _ { \mathrm { i } } \left[ \mathrm { S o f t m a x } \left( \frac { Q _ { \mathrm { v } } K _ { \mathrm { i } } ^ { \top } } { \sqrt { d } } \right) V _ { \mathrm { i } } \right] ,\tag{3}
$$

where $\sqrt { d }$ is the scaling factor and $\mathcal { P } _ { \mathrm { v } } ( \cdot )$ and $\mathcal { P } _ { \mathrm { i } } ( \cdot )$ denote the modality-specific output projections. The single-head attention responses are reshaped into spatial feature maps, passed through modality-specific $1 \times 1$ output projections, and added to the original unnormalized level-4 encoder features through residual connections.

Cross-Gate Fusion. While Cross-Attention enables effective information exchange between visible and infrared features, it does not explicitly regulate the reliability of the transferred information. Therefore, a Cross-Gate Fusion module is introduced to adaptively modulate the cross-attended features, allowing the network to selectively preserve informative cues and suppress degraded or irrelevant responses. At the deepest level, the Cross-Gate Fusion module takes the cross-attended features as inputs and can be expressed as $F _ { \mathrm { c g } } ^ { ( 4 ) } = \mathcal { F } _ { \mathrm { C G } } ^ { ( 4 ) } ( F _ { \mathrm { v } } ^ { \mathrm { c a } } , F _ { \mathrm { i } } ^ { \mathrm { c a } } )$ , where $\mathcal { F } _ { \mathrm { C G } } ^ { ( 4 ) } ( \cdot , \cdot )$ denotes the Cross-Gate Fusion operation at the deepest level. The application of Cross-Gate Fusion at the finer hierarchical levels is described in the subsequent ITIM formulation. The detailed architecture is illustrated in Fig. 3. First, we calculate the gating weights for the two image feature maps from the Cross-Attention layer:

$$
\mathrm { g a t e } _ { \mathrm { i } } = \sigma \left( \mathrm { C o n v } _ { \mathrm { i } } ^ { 1 \times 1 } \left( F _ { \mathrm { v } } ^ { \mathrm { c a } } \right) \right) , \quad \mathrm { g a t e } _ { \mathrm { v } } = \sigma \left( \mathrm { C o n v } _ { \mathrm { v } } ^ { 1 \times 1 } \left( F _ { \mathrm { i } } ^ { \mathrm { c a } } \right) \right) ,\tag{4}
$$

where $\sigma$ denotes the sigmoid function. The learned gating weights are then used to adaptively regulate the contribution of each modality at every spatial location, enabling selective preservation of informative features while suppressing less reliable responses, as formulated below:

$$
\widehat { F } _ { \mathrm { c g } } ^ { ( 4 ) } = F _ { \mathrm { v } } ^ { \mathrm { c a } } \odot \mathrm { g a t e _ { v } } + F _ { \mathrm { i } } ^ { \mathrm { c a } } \odot \mathrm { g a t e _ { i } } ,\tag{5}
$$

where $\odot$ denotes the Hadamard product. For the final output, an additional convolutional layer is introduced to further refine the fused features and enhance their representation capability: $F _ { \mathrm { c g } } ^ { ( 4 ) } =$ $\mathrm { C o n v } _ { 1 \times 1 } ^ { ( 4 ) } ( \widehat { F } _ { \mathrm { c g } } ^ { ( 4 ) } )$ .

Iterative Text-Image Interaction Module (ITIM). The deepest-level Cross-Gate Fusion output is first processed by a single-head Self-Attention (SE-ATT) module. Before attention, the feature is normalized using 16-group GroupNorm, and $\textbf { a } 1 \times 1$ convolution generates the query, key, and value representations. The attention response is output-projected and combined with a residual connection:

$$
[ Q _ { \mathrm { c g } } , K _ { \mathrm { c g } } , V _ { \mathrm { c g } } ] = \mathrm { T o k e n i z e } \left( \mathrm { S p l i t _ { 3 } } \left[ \mathrm { C o n v } _ { \mathrm { q k v } } ^ { 1 \times 1 } \left( \mathrm { G N } \left( F _ { \mathrm { c g } } ^ { ( 4 ) } \right) \right) \right] \right) , F _ { \mathrm { s e } } = F _ { \mathrm { c g } } ^ { ( 4 ) } + \mathcal { P } _ { \mathrm { s e } } \left[ \mathrm { S o f i m a x } \left( \frac { Q _ { \mathrm { c g } } K _ { \mathrm { c g } } ^ { \top } } { \sqrt { d } } \right) V _ { \mathrm { c g } } \right] ,\tag{6}
$$

where $Q _ { \mathrm { c g } } , K _ { \mathrm { c g } }$ , and $V _ { \mathrm { c g } }$ denote the query, key, and value projections of $F _ { \mathrm { c g } } ^ { ( 4 ) }$ , respectively, and $\mathcal { P } _ { \mathrm { s e } } ( \cdot )$ denotes the output projection. After this early fusion stage, the framework enters the phase where textual information actively participates in the fusion process.

Before applying text-conditioned feature modulation, the global text embedding is projected into a stage-specific semantic conditioning vector. At each hierarchical stage, the visual feature is represented as spatial tokens, whereas the single global text embedding is projected to provide the semantic conditioning signal. The resulting text-conditioned representation is subsequently mapped to affine modulation parameters that regulate the corresponding visual features.

Since a single global sentence-level text embedding is used rather than a sequence of token-level text features, the operation provides global semantic conditioning at each hierarchical stage instead of tokenwise language-vision attention. The main role of this design is therefore to repeatedly inject the prompt semantics into visual representations throughout hierarchical decoding. The stage-specific conditioning parameters allow the same textual guidance to affect features at different semantic resolutions, thereby enabling progressive text-conditioned feature refinement.

At the deepest encoder level $( k = 4 )$ , the visible and infrared features first undergo the Cross-Attention described above, followed by Cross-Gate Fusion and Self-Attention (SE-ATT). The resulting fused feature $F _ { \mathrm { s e } }$ is then conditioned on the global text embedding through the stage-specific text-conditioning function $\varphi ^ { ( 4 ) } ( \cdot , \cdot )$ and processed by a Transformer-based decoder block [29]:

$$
F _ { \mathrm { d } } ^ { ( 4 ) } = \mathcal { F } _ { \mathrm { D } } ^ { ( 4 ) } \left( \varphi ^ { ( 4 ) } \left( F _ { \mathrm { s e } } , F _ { \mathrm { t e x t } } \right) \right) .\tag{7}
$$

For the subsequent levels $k \in \{ 3 , 2 , 1 \}$ , the decoded feature from the immediately coarser level is first upsampled to the current spatial resolution and then modulated by the same global text embedding. In parallel, the visible and infrared encoder features at the corresponding level are reintroduced and fused using a stage-specific Cross-Gate Fusion module. The text-conditioned decoder feature and the cross-gated encoder feature are subsequently concatenated along the channel dimension:

$$
\begin{array} { r l } & { \widetilde { F } _ { \mathrm { u } } ^ { ( k ) } = \varphi ^ { ( k ) } \left( \mathcal { U } ^ { ( k ) } \left( F _ { \mathrm { d } } ^ { ( k + 1 ) } \right) , F _ { \mathrm { t e x t } } \right) , \quad F _ { \mathrm { e } } ^ { ( k ) } = \mathcal { F } _ { \mathrm { C G } } ^ { ( k ) } \left( F _ { \mathrm { v i s } } ^ { ( k ) } , F _ { \mathrm { i r } } ^ { ( k ) } \right) , } \\ & { F _ { \mathrm { d } } ^ { ( k ) } = \mathcal { F } _ { \mathrm { D } } ^ { ( k ) } \left( \mathcal { R } ^ { ( k ) } \left( [ \widetilde { F } _ { \mathrm { u } } ^ { ( k ) } ; F _ { \mathrm { e } } ^ { ( k ) } ] \right) \right) , \quad k = 3 , 2 , 1 . } \end{array}\tag{8}
$$

![](images/f897066d9e8858304ba010d983c1b3746e6f92d89e56730582de81c60642bc21.jpg)  
Figure 3. Architecture of the Cross-Gate Fusion module, illustrated using its application at the deepest level $( k = 4 )$ . Cross-attended visible and infrared features generate cross-modal gates to regulate the opposite modality, after which the cross-gated features are added and projected through a convolution.

Here, $\mathcal { U } ^ { ( k ) } ( \cdot )$ denotes the implemented upsampling operator, [·;·] denotes channel-wise concatenation, and $\mathcal { R } ^ { ( k ) } ( \cdot )$ denotes the channel-adjustment operation. Specifically, $\mathcal { R } ^ { ( 3 ) } ( \cdot )$ and $\mathcal { R } ^ { ( 2 ) } ( \cdot )$ are implemented using $1 \times 1$ convolutions, whereas $\mathcal { R } ^ { ( 1 ) } ( \cdot )$ is the identity mapping. Therefore, the concatenated level-1 feature is fed directly to the final decoder block without an additional channel-reduction convolution.

After the initial level-4 fusion, the encoder features reintroduced during the remaining three interactions are the level-3, level-2, and level-1 visible and infrared features, respectively. At the deepest level, Cross-Attention enables bidirectional high-level semantic interaction between the visible and infrared features. At the finer levels, the framework combines an upsampled text-conditioned decoder feature with a same-resolution cross-gated encoder feature. In this way, the global text embedding is repeatedly injected into the hierarchical decoding process while modality-specific visual information is progressively recovered from the encoder branches.

## 3.3. Text Semantic Encoding and Feature Modulation

This subsection describes the stage-specific text-conditioning function $\varphi ^ { ( k ) } ( \cdot , \cdot )$ used in Eqs. (7) and (8). Specifically, the visual feature provides the query representation, whereas the pooled CLIP text embedding provides the key and value representations. The resulting text-conditioned context is then projected to the visual channel dimension and used to generate feature-wise affine modulation parameters.

Text Semantic Encoder. The text semantic encoder transforms the input prompt into a compact sentence-level text embedding. Benefiting from large-scale vision-language pretraining, CLIP [13] provides text representations with strong semantic discrimination. To maintain linguistic consistency and avoid overfitting the text encoder to the fusion datasets, the CLIP text encoder is kept frozen throughout training. The text feature extraction process is formulated as

$$
F _ { \mathrm { t e x t } } = \mathcal { F } _ { \mathrm { c l i p } } \left( T _ { \mathrm { t e x t } } \right) ,\tag{9}
$$

where $F _ { \mathrm { t e x t } } \in \mathbb { R } ^ { B \times 5 1 2 }$ denotes the pooled global text embedding returned by the frozen CLIP ViT-B/32 text encoder. Only this single 512-dimensional embedding is retained for each prompt.

Stage-Specific Visual-Text Conditioning. Let $F _ { \mathrm { f } } ^ { ( k ) } \in \mathbb { R } ^ { B \times C _ { k } \times H _ { k } \times W _ { k } }$ denote the visual feature entering the text-conditioning module at the k-th interaction stage, where $C _ { k } , H _ { k }$ , and $W _ { k }$ denote its channel number, height, and width, respectively. The visual feature is first flattened into $N _ { k } = H _ { k } W _ { k }$ spatial tokens:

$$
X _ { \mathrm { f } } ^ { ( k ) } = { \mathrm { r e s h a p e } } \left( F _ { \mathrm { f } } ^ { ( k ) } \right) \in \mathbb { R } ^ { B \times N _ { k } \times C _ { k } } .\tag{10}
$$

The visual tokens and the global text embedding are independently normalized and projected to a common transformer width $D = 5 1 2$ :

$$
\begin{array} { r } { Z _ { \mathrm { f } } ^ { ( k ) } = W _ { \mathrm { F } } ^ { ( k ) } \mathrm { L N } _ { \mathrm { F } } ^ { ( k ) } \left( X _ { \mathrm { f } } ^ { ( k ) } \right) , \quad Z _ { \mathrm { t e x t } } ^ { ( k ) } = W _ { \mathrm { T } } ^ { ( k ) } \mathrm { L N } _ { \mathrm { T } } ^ { ( k ) } \left( F _ { \mathrm { t e x t } } \right) . } \end{array}\tag{11}
$$

Here, $Z _ { \mathrm { f } } ^ { ( k ) } \in \mathbb { R } ^ { B \times N _ { k } \times D }$ . The projected text embedding is regarded as a single text token, such that $Z _ { \mathrm { t e x t } } ^ { ( k ) } \in \bar { \mathbb { R } } ^ { B \times 1 \times D }$ . All projection parameters are independently parameterized at different hierarchical stages.

The query representation is generated from the projected visual tokens, whereas the key and value representations are generated from the projected text token:

$$
\begin{array} { r } { Q ^ { ( k ) } = W _ { \mathrm { Q } } ^ { ( k ) } Z _ { \mathrm { f } } ^ { ( k ) } , \quad K ^ { ( k ) } = W _ { \mathrm { K } } ^ { ( k ) } Z _ { \mathrm { t e x t } } ^ { ( k ) } , \quad V ^ { ( k ) } = W _ { \mathrm { V } } ^ { ( k ) } Z _ { \mathrm { t e x t } } ^ { ( k ) } . } \end{array}\tag{12}
$$

Multi-head attention is then computed using $Q ^ { ( k ) } , K ^ { ( k ) }$ , and $V ^ { ( k ) }$ to obtain the text-conditioned context $C _ { \mathrm { c t x } } ^ { ( k ) } \in \mathbb { R } ^ { B \times N _ { k } \times C _ { k } }$

Semantic Feature Modulation. The text-conditioned context $C _ { \mathrm { c t x } } ^ { ( k ) }$ obtained above contains the textual information used to regulate the visual feature at the k-th interaction stage. To inject this context into the visual feature, it is first mapped to a pair of affine modulation parameters:

$$
[ \gamma ^ { ( k ) } , \beta ^ { ( k ) } ] = \mathrm { S p l i t } _ { 2 } \left[ \mathrm { G E L U } \left( W _ { \gamma \beta } ^ { ( k ) } C _ { \mathrm { c t x } } ^ { ( k ) } + b _ { \gamma \beta } ^ { ( k ) } \right) \right] ,\tag{13}
$$

where $W _ { \gamma \beta } ^ { ( k ) } : \mathbb { R } ^ { C _ { k } }  \mathbb { R } ^ { 2 C _ { k } }$ . The $2 C _ { k } { \mathrm { - d i m e n s i o n a l } }$ output is evenly split into the scaling parameter $\gamma ^ { ( k ) }$ and the bias parameter $\beta ^ { ( k ) }$ . The two parameters are jointly generated by the same linear projection followed by GELU, while different hierarchical stages use independent parameter sets.

After reshaping $\gamma ^ { ( k ) }$ and $\beta ^ { ( k ) }$ from token form to spatial feature maps, they are applied to the input visual feature $F _ { \mathrm { f } } ^ { ( k ) }$ as

$$
\widehat { F } _ { \mathrm { f } } ^ { ( k ) } = \varphi ^ { ( k ) } \left( F _ { \mathrm { f } } ^ { ( k ) } , F _ { \mathrm { t e x t } } \right) = \left( 1 + \mathrm { r e s h a p e } \left( \gamma ^ { ( k ) } \right) \right) \odot F _ { \mathrm { f } } ^ { ( k ) } + \mathrm { r e s h a p e } \left( \beta ^ { ( k ) } \right) ,\tag{14}
$$

![](images/35d9b16a4b02cab63a9280017c956ede63176bf706701f03d886a881d25f4e58.jpg)  
Figure 4. The U-Net structured Text-Guided Residual Refinement Module (TG-RRM), which consists of four encoder-decoder stages with text-conditioned modulation.

Here, $\gamma ^ { ( k ) }$ controls the feature scaling, while $\beta ^ { ( k ) }$ provides an additive feature shift. $\widehat { F } _ { \mathrm { f } } ^ { ( k ) }$ denotes the text-modulated output feature, whereas $F _ { \mathrm { f } } ^ { ( k ) }$ denotes the visual feature before text conditioning. The function $\varphi ^ { ( k ) } ( \cdot , \cdot )$ represents the complete text-conditioning operation at the k-th stage, and its parameters are not shared across hierarchical stages.

## 3.4. Text-Guided Residual Refinement Module (TG-RRM)

After four hierarchical text-image interactions, the network produces a high-resolution fused feature representation that already encodes rich cross-modal information under textual guidance. Although this intermediate feature captures the global fusion structure, it may still contain local ambiguities and degradation-related residual responses before final image reconstruction.

To alleviate these limitations, refinement stages have been widely adopted in image restoration and fusion frameworks [29,30]. Such strategies enhance visual quality by correcting prediction errors and improving detail fidelity, typically through convolutional or transformer-based layers. Related refinement strategies have also been explored in event-based stereo depth estimation, where URNet [31] combines local-global refinement with uncertainty modeling to improve disparity prediction. For infrared-visible image fusion, however, refinement under challenging degradation conditions remains difficult. Coarse outputs often contain structural ambiguities and regionally inconsistent responses, which restrict the ability of shallow or single-scale refinement modules to fully recover fine details and coherent structures.

Motivated by these observations, we introduce TG-RRM, designed to enhance fine-grained details and improve structural coherence. As illustrated in Fig. 4, TG-RRM adopts a compact U-Net architecture that takes the high-resolution fused feature as input and predicts a residual feature correction. This correction is added to the input feature, after which the network’s output layer reconstructs the three-channel fused image. The encoder-decoder structure enables progressive feature abstraction and reconstruction, while skip connections directly propagate low-level spatial information, facilitating the preservation of fine textures and sharp boundaries. The refinement process is explicitly conditioned on the global CLIP text embedding $F _ { \mathrm { t e x t } }$ , allowing semantic guidance to remain active during residual correction. Specifically, the textual information is injected into both encoder and decoder stages through a lightweight feature-wise modulation mechanism. The text embedding is mapped via a small MLP to generate affine parameters $( \gamma _ { \mathrm { r } } , \beta _ { \mathrm { r } } )$ , which modulate intermediate feature responses according to $x  ( 1 + \gamma _ { \mathrm { r } } ) \odot x + \beta _ { \mathrm { r } }$ . The textconditioned affine transformations used at different encoder stages, the bottleneck, decoder stages, and the residual output stage are independently parameterized and do not share parameters across feature levels. At each modulation site, a separate two-layer MLP with dimensions $5 1 2  1 0 2 4  2 C$ and a LeakyReLU activation between the two linear layers jointly produces the 2C-dimensional affine vector, which is subsequently split into the scaling term $\gamma _ { \mathrm { r } }$ and bias term $\beta _ { \mathrm { r } }$ . Thus, the two affine parameters share the same MLP within one site, while different scales and refinement sites use independent MLP parameters.

This text-guided refinement strategy enables TG-RRM to adaptively emphasize task-relevant structures while suppressing degradation-related artifacts. By combining multi-scale U-Net refinement with semantic conditioning, TG-RRM produces fusion results with improved visual consistency, sharper structural details, and better alignment with the intended textual guidance.

## 3.5. Loss Functions

The loss function largely determines what source information is preserved and how different modalities contribute to the final fused result. Following prior text-guided fusion frameworks, the fusion-related losses include intensity loss, structural similarity (SSIM) loss [32], maximum gradient loss, and color consistency loss. Since no ground-truth fused images are available, the fusion network is trained using a combination of complementary loss terms that jointly encourage visual saliency, structural preservation, and color consistency. The overall training objective is formulated as a weighted sum of these losses.

Intensity loss: The intensity loss preserves the source pixel selected by a luminance comparison while retaining the three-channel value of the selected modality. Let

$$
\bar { I } _ { m } = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } I _ { m , c } , \quad M = \mathbf { 1 } [ \bar { I } _ { \mathrm { i r } } > \bar { I } _ { \mathrm { v i s } } ] ,\tag{15}
$$

where $m \in \{ \mathrm { v i s } , \mathrm { i r } \}$ and M is broadcast to the three image channels. The intensity target and loss are

$$
I _ { \mathrm { m a x } } = M \odot I _ { \mathrm { i r } } + ( 1 - M ) \odot I _ { \mathrm { v i s } } , \quad \mathcal { L } _ { \mathrm { i n t } } = \left| \left| I _ { \mathrm { f u s e d } } - I _ { \mathrm { m a x } } \right| \right| _ { 1 } ,\tag{16}
$$

where the $\ell _ { 1 }$ loss is averaged over pixels and channels. For synthetically degraded samples in EMS, the clean source pair stored with the training sample is used to construct the fusion losses. For standard datasets without separate clean counterparts, the available infrared and visible source images are used directly.

Structural similarity loss: To preserve structural information from both modalities, the implementation computes a visible-image SSIM term in RGB space and an infrared SSIM term after grayscale conversion. The structural similarity loss is defined as

$$
\mathcal { L } _ { \mathrm { S S I M } } = w _ { \mathrm { s s i m - v i s } } [ 1 - \mathrm { S S I M } ( I _ { \mathrm { f u s e d } } , I _ { \mathrm { v i s } } ) ] + w _ { \mathrm { s s i m - i r } } [ 1 - \mathrm { S S I M } ( G ( I _ { \mathrm { f u s e d } } ) , G ( I _ { \mathrm { i r } } ) ) ] ,\tag{17}
$$

where $G ( \cdot )$ denotes RGB-to-grayscale conversion. The SSIM index is computed in its standard form:

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{18}
$$

where $\mu , \sigma$ , and $\sigma _ { x y }$ denote local means, standard deviations, and cross-covariance computed within a sliding window. $C _ { 1 }$ and $C _ { 2 }$ are small constants that prevent division by zero.

Maximum gradient loss: The maximum gradient loss is introduced to preserve sharp edges from both modalities. In the implementation, the visible, infrared, and fused images are first converted to grayscale. Horizontal and vertical absolute Sobel responses are then computed separately, and the fused response is matched to the element-wise maximum source response in each direction:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g r a d } } = \| D _ { x } ( G ( I _ { \mathrm { f u s e d } } ) ) - \operatorname* { m a x } \left( D _ { x } ( G ( I _ { \mathrm { v i s } } ) ) , D _ { x } ( G ( I _ { \mathrm { i r } } ) ) \right) \| _ { 1 } } \\ & { \qquad + \left\| D _ { y } ( G ( I _ { \mathrm { f u s e d } } ) ) - \operatorname* { m a x } \left( D _ { y } ( G ( I _ { \mathrm { v i s } } ) ) , D _ { y } ( G ( I _ { \mathrm { i r } } ) ) \right) \right\| _ { 1 } , } \end{array}\tag{19}
$$

where $D _ { x } ( \cdot )$ and $D _ { y } ( \cdot )$ denote the absolute horizontal and vertical Sobel responses after grayscale conversion. The Sobel filters are applied after one-pixel replicate padding, and the absolute horizontal and vertical responses are matched separately.

Color consistency loss: The color consistency loss encourages the fused image to preserve the chrominance information of the visible image. Both images are transformed to YCbCr space, and the Cb and Cr channels are constrained with an $\ell _ { 1 }$ distance:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = \Vert C _ { \mathrm { b } } ( I _ { \mathrm { f u s e d } } ) - C _ { \mathrm { b } } ( I _ { \mathrm { v i s } } ) \Vert _ { 1 } + \Vert C _ { \mathrm { r } } ( I _ { \mathrm { f u s e d } } ) - C _ { \mathrm { r } } ( I _ { \mathrm { v i s } } ) \Vert _ { 1 } .\tag{20}
$$

Overall training objective: The overall training objective is composed of the intensity, structural similarity, color consistency, and maximum gradient losses:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = w _ { \mathrm { i n t } } \mathcal { L } _ { \mathrm { i n t } } + w _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { S S I M } } + w _ { \mathrm { c o l o r } } \mathcal { L } _ { \mathrm { c o l o r } } + w _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } ,\tag{21}
$$

where $w _ { \mathrm { i n t } } , w _ { \mathrm { s s i m } } , w _ { \mathrm { c o l o r } } ,$ and $w _ { \mathrm { g r a d } }$ denote the coefficients of the corresponding loss terms. The coefficients $w _ { \mathrm { s s i m - v i s } }$ and ${ w _ { \mathrm { s s i m - i r } } }$ are applied inside ${ \mathcal { L } } _ { \mathrm { S S I M } }$ , as defined in Eq. (17).

## 4. Results

In this section, we first describe the implementation details and experimental configurations. We then evaluate the effectiveness of the proposed method through both qualitative and quantitative comparisons, with a particular focus on text-guided image fusion. Finally, ablation experiments are conducted to analyze the contributions of different components in the proposed framework.

Table 1. Quantitative comparison of IT-TextFusion with existing image fusion methods and the baseline Text-IF [9] on the MSRS, LLVIP, and RoadScene datasets using a fixed generic prompt without degradation-aware or scene-specific semantic information (Bold: best performance).
<table><tr><td rowspan="2">Methods</td><td colspan="5">MSRS Dataset</td><td colspan="5">LLVIP Dataset</td><td colspan="5">RoadScene Dataset</td></tr><tr><td>SCD↑</td><td>SD↑</td><td>EN↑</td><td>VIFF↑</td><td> $\overline { { Q ^ { A B / F } \dag } }$ </td><td>SCD↑</td><td>SD↑</td><td>EN↑</td><td>VIFF↑</td><td> $\overline { { Q ^ { A B / F } \dag } }$ </td><td>SCD↑</td><td>SD↑</td><td>EN↑</td><td>VIFF↑</td><td> $\overline { { Q ^ { A B / F } \dag } }$ </td></tr><tr><td>UMF-CMGR [36]</td><td>0.981</td><td>20.819</td><td>5.600</td><td>0.430</td><td>0.266</td><td>1.029</td><td>31.501</td><td>6.569</td><td>0.509</td><td>0.352</td><td>1.613</td><td>36.251</td><td>6.973</td><td>0.554</td><td>0.429</td></tr><tr><td>TarDAL [37]</td><td>1.484</td><td>35.460</td><td>6.347</td><td>0.673</td><td>0.426</td><td>0.817</td><td>39.070</td><td>5.349</td><td>0.330</td><td>0.252</td><td>1.415</td><td>42.609</td><td>7.054</td><td>0.525</td><td>0.391</td></tr><tr><td>ReCoNet [38]</td><td>1.191</td><td>44.374</td><td>3.895</td><td>0.438</td><td>0.367</td><td>1.345</td><td>41.234</td><td>5.514</td><td>0.513</td><td>0.364</td><td>1.589</td><td>37.580</td><td>6.822</td><td>0.504</td><td>0.354</td></tr><tr><td>MURF [39]</td><td>0.868</td><td>16.4315.047</td><td></td><td>0.413</td><td>0.327</td><td>0.514</td><td>21.834</td><td>6.051</td><td>0.386</td><td>0.206</td><td>1.576</td><td>36.788</td><td>6.992</td><td>0.484</td><td>0.432</td></tr><tr><td>U2Fusion [17]</td><td>1.182</td><td>23.5415.246</td><td></td><td>0.506</td><td>0.372</td><td>0.757</td><td>23.614</td><td>5.972</td><td>0.552</td><td>0.341</td><td>1.498</td><td>30.969</td><td>6.739</td><td>0.513</td><td>0.467</td></tr><tr><td>MetaFusion [18]</td><td>1.486</td><td>39.432</td><td>6.368</td><td>0.726</td><td>0.478</td><td>1.317</td><td>42.446</td><td>6.823</td><td>0.833</td><td>0.493</td><td>1.581</td><td>50.613 7.223</td><td></td><td>0.512</td><td>0.338</td></tr><tr><td>DDFM [19]</td><td>1.550</td><td>32.749</td><td>5.693</td><td>0.622</td><td>0.431</td><td>1.414</td><td>38.346</td><td>6.979</td><td>0.549</td><td>0.220</td><td>1.864</td><td>44.925</td><td>7.226</td><td>0.544</td><td>0.413</td></tr><tr><td>Text-IF [9]</td><td>1.681</td><td>44.564</td><td>6.789</td><td>1.046</td><td>0.676</td><td>1.591</td><td>48.834</td><td>7.325</td><td>1.011</td><td>0.616</td><td>1.572</td><td>48.962</td><td>7.332</td><td>0.739</td><td>0.578</td></tr><tr><td>IT-TextFusion (Ours)</td><td>1.721</td><td>42.889</td><td>6.882</td><td>1.064</td><td>0.725</td><td>1.520</td><td>50.451</td><td>7.443</td><td>1.042</td><td>0.760</td><td>1.102</td><td>49.102</td><td>7.332</td><td>0.733</td><td>0.636</td></tr></table>

## 4.1. Implementation Details

All models are implemented in PyTorch and trained on NVIDIA RTX A5000 GPU. We adopt the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of $5 \times 1 0 ^ { - 2 }$ , and a batch size of 8. Training is conducted for 120 epochs, and the checkpoint achieving the lowest validation loss is selected for final evaluation. During training, input images are randomly cropped to a spatial resolution of $9 6 \times 9 6$ Horizontal and vertical flips are applied with a probability of 0.5 to improve generalization.

## 4.2. Datasets

Standard Datasets. We evaluate the model on several standard infrared-visible fusion datasets, including MSRS [33], LLVIP [34], MFNet [35], and RoadScene [17]. These datasets primarily represent singledegradation conditions. MSRS and LLVIP mainly contain low-light visible images, whereas MFNet focuses on infrared images with low contrast. RoadScene contains outdoor traffic scenes that exhibit illumination imbalance and exposure variations, providing an additional real-world evaluation scenario. Due to the large size of LLVIP, we follow common practice and randomly select 40% of the dataset, training on this subset for 50 epochs. Since MSRS and LLVIP do not provide ground-truth fused images, the original infrared and visible image pairs are directly used as supervision targets during training and loss computation.

EMS Degradation Dataset. In addition to the standard fusion datasets, to evaluate robustness under diverse synthetic degradation conditions that approximate common imaging artifacts, we construct an extended multi-degradation dataset based on three publicly available infrared-visible datasets: MSRS [33], LLVIP [34], and MFNet [35]. The extended dataset includes nine common degradation types: Low Light, Rain, Visible Blur, Overexposure, Visible Haze, Visible Random Noise, Infrared Low Contrast, Infrared Random Noise, and Infrared Stripe Noise. Each degraded sample consists of an RGB–IR image pair and is accompanied by a corresponding textual description, enabling text-conditioned fusion training. For degradation types without predefined descriptions, additional prompts are created following the structure and semantics of existing templates to maintain consistency across all degradation scenarios.

## 4.3. Evaluation Metrics and Comparison Methods

Evaluation Metrics. Following the evaluation protocol of Text-IF [9], we employ a comprehensive set of classical fusion metrics, including information entropy (EN) [40], standard deviation (SD), spatial frequency (SF) [41], mutual information (MI), sum of correlations of differences (SCD) [42], visual information fidelity (VIFF) [43], gradient-based fusion quality $Q ^ { A B / F }$ [40], and a fusion-specific summed metric constructed based on the Structural Similarity Index Measure (SSIM) [44], denoted as $\mathrm { S S I M _ { \mathrm { s u m } } }$

Since no ground-truth fused image is available, the structural similarity evaluation is computed between the fused image and each of the two source images and then summed:

$$
\mathrm { S S I M _ { s u m } } = \mathrm { S S I M } ( I _ { \mathrm { f } } , I _ { \mathrm { i r } } ) + \mathrm { S S I M } ( I _ { \mathrm { f } } , I _ { \mathrm { v i s } } ) ,\tag{22}
$$

where $I _ { \mathrm { f } } , I _ { \mathrm { i r } } ,$ , and $I _ { \mathrm { v i s } }$ denote the grayscale fused, infrared, and visible images, respectively. Each pairwise SSIM term is computed using the standard definition in Eq. (18), with the image intensity range set to 255. Therefore, $\mathrm { S S I M _ { \mathrm { s u m } } }$ is not bounded by 1 and may take values greater than 1. A higher value indicates that the fused image jointly preserves more structural information from the two source modalities.

For datasets without reference images, we additionally report CLIP-IQA [45], NIQE [46], MUSIQ [47], and BRISQUE [48], which assess no-reference perceptual quality and image naturalness using different statistical or learned representations. Although CLIP-IQA uses CLIP-derived representations, the adopted implementation receives only the fused image and does not use the input degradation-aware prompt. Therefore, it is treated as a no-reference perceptual-quality metric rather than a direct measure of image-prompt semantic alignment. Lower NIQE and BRISQUE values indicate higher perceived naturalness, whereas higher EN, $Q ^ { \mathrm { A B / F } }$ , VIFF, and $\mathrm { S S I M _ { \mathrm { s u m } } }$ values indicate better information, gradient, visual-fidelity, and joint structural preservation, respectively. To ensure consistency with prior literature, all metrics are computed using publicly released implementations commonly adopted in recent fusion works.

Comparison Methods. We compare the proposed method with several state-of-the-art (SOTA) methods on multiple datasets, which are described in Section 4.2. In particular, the proposed method is mainly compared with Text-IF [9], a recent representative framework for text-guided image fusion. Other methods for comparison include U2Fusion [17], MetaFusion [18], DDFM [19], MURF [39], ReCoNet [38], TarDAL [37], and UMF-CMGR [36], which are conventional image fusion approaches relying purely on visual information without involving any text or language-based guidance. For the synthesized EMS dataset described in Section 4.2, we compare only with Text-IF.

## 4.4. Comparison with a Fixed Generic Prompt

To evaluate the effect of degradation-specific prompt content, we first conduct experiments using a fixed generic prompt, i.e., “This is an infrared and visible image fusion task.” Under this setting, the textual branch remains active, while the input prompt contains no degradation-aware or scene-specific semantic information. This setting is also consistent with our primary baseline Text-IF [9]. Therefore, rather than treating this experiment as a text-free visual-only baseline, we regard it as an experiment using a fixed generic prompt and use it for controlled comparison. In this way, we can investigate whether the proposed

Text: In the context of infrared-visible light fusion, visible images may suffer from reduced quality in low-light scenarios.

![](images/f8af99ffb2483c2874c807dee580facc25cd590dc7b9c53eb9d38ab6cadde372.jpg)  
Infrared

![](images/573a72b82b0ae43c20571f7cb00296d75220eba714009cf1de7d9d5c81fdb23b.jpg)  
Visible

![](images/17a930de12be5033485b0201f9100c55250b52aa8064b9534735f82e9fb523f3.jpg)  
Text-IF

![](images/5d368fa0c2f37d20411a94f67d0e2d388cffd16c3feb8179af9059307a53d498.jpg)  
Ours

![](images/4891e38681d69f6bdbdee65a777e82c7a448818a3364ef76876c4e3a9be9cb85.jpg)  
Infrared

![](images/f9a9e47f962732d1a8686428a871a6d2cb46453fa9ed3eb5d1f7716d7075f5c8.jpg)  
Visible

![](images/5942360195f334defb346382a4510923bd7d9567f5d87adeda75d0939074e8c1.jpg)  
Text-IF

![](images/d5d52cc453d4f6cced842c363a2282d4fb252223216f967dea0c194e66bd14ea.jpg)  
Ours

Text: In this challenge, we focus on the low contrast degradation in the infrared images.  
![](images/c838fd51478c138c5735dda3685bbe665f22f75f748d211bb34b175edb7204cd.jpg)  
Infrared

![](images/ee108d6d6a2100962968f6833405597d5621e655e13bcaf0ef5c28661f006307.jpg)  
Visible

![](images/ddbcb35583cecd7b05c7632199d82689812fd50d733010f86c547e499a3040c9.jpg)  
Text-IF

![](images/64e8fd9ad1477f2bf6b29b444c9f064daaed051fe83b89de78861d1d528df2d0.jpg)  
Ours

![](images/1814eb1a80901f13c372910d13a5372125c465fbcb2f14b739b148341bf478b9.jpg)  
Infrared

![](images/1c7e6f5b3a137930e583994c7f9395f653362b1f87c84e7fe1035626574ba113.jpg)  
Visible

![](images/d8b7b698613cff03e4f16e11374c3c4b40f3ae3dd14d4161581c1a9b04aee285.jpg)  
Text-IF

![](images/ebd237926eb5a8109b428aecca4fb0f0ba6554175758f86d7a5a608ebf5b3585.jpg)  
Ours

Text: This pertains to the fusion of infrared and visible light images, with a low light degradation in the visible images.  
![](images/2b391e490e2ceeeb3064d2a64b9dda0eff440160a98d5b093ce37ed658801b3a.jpg)  
Infrared

![](images/64bfd3914090b1e5fee366f5464ea14e5c4a4172523efd7f897bb8d0c3c6e3d8.jpg)  
Visible

![](images/34a111c70a16f41beea9b03f0cd116583712d743a60f54714dbd1c8fb63517c3.jpg)  
Text-IF

![](images/6258b52161ddc09287c8d63d434a7cfa13d4ad991c7391e418b6a37525e3f3f5.jpg)  
Ours

![](images/70b288405d21d15fbf486c5bc301593c8506081bcf358fa4afadc6ed036b5d7a.jpg)  
Infrared

![](images/9fbe5f08897d7f4343c5dd460bff546fb1fe6004af0c38151e32ee550342dd04.jpg)  
Visible

![](images/3825e73131b844aed34b1d5da34bdf888d044ac9542cfe1e258ab2177ab462ab.jpg)  
Text-IF

![](images/ded03c1f6319d665ffa7ebe649d156e3d09bc27b72c36eca1747dbcbd941f7fa.jpg)  
Ours

Text: We’re tackling the infrared-visible light image fusion challenge, dealing with visible images suffering from overexposure degradation.  
![](images/9dd3bce6d8f299ae70aa87c55022c0b296ba2aa22104209318fcaf2debaa62a4.jpg)  
Infrared

![](images/49c4b0b6352a4a866fdf73818dfce1560370380b9c8277ba1d9dd33b7f2faff5.jpg)  
Visible

![](images/45c5e083bc0bcd0dd1ed0b46f1758706abed11a0fdf1fbfb43c4a7b2a3d25b83.jpg)  
Text-IF

![](images/d45242e57a79b61b14b29ccecaf72b6c46f0b0216330f1d34c741402ed152fd8.jpg)  
Ours

![](images/737589f582652e07b8cf34f43cfec7eb9fa4fe94124b6aaa8e26b435efdbf435.jpg)  
Infrared

![](images/543343e4371a0c1d4f0116eef2cd4cee7b931aca99c87701ba92783b90053faa.jpg)  
Visible

![](images/7934cb308c5d41e34fd7c46cb473e9c733beb622aac63d08df90074d5dd13814.jpg)  
Text-IF

![](images/2b9013ac0e2bf1d73cb6f70c52a8a1874ed9e9e6d7f3fb89ef315cc752dccb91.jpg)  
Ours

Figure 5. Qualitative comparison with the baseline method Text-IF [9] on the four standard datasets. Degradations from top to bottom: Low Light (LLVIP), Infrared Low Contrast (MFNet), Low Light (MSRS), and Overexposure (RoadScene). For each dataset, Text-IF and our method use the same degradation-aware text prompt for a fair comparison.

architecture itself provides improved fusion capability before introducing explicit degradation-aware text guidance.

Tab. 1 shows that the proposed method achieves competitive performance when using a fixed generic prompt, with different advantages across datasets and metrics. On MSRS, the proposed method improves SCD, EN, VIFF, and $Q ^ { A B / F }$ compared with Text-IF, although its SD is lower. On LLVIP, improvements are observed in SD, EN, VIFF, and $Q ^ { A B / F }$ , whereas SCD decreases slightly. On RoadScene, the proposed method obtains higher SD and $Q ^ { A B / F }$ , while Text-IF retains higher SCD and VIFF and both methods achieve the same EN. These results indicate that the architectural modifications alter the balance among different fusion properties rather than providing uniform improvements on every metric.

Compared with Text-IF, the proposed method introduces an iterative text-image interaction mechanism, where the same global text embedding modulates features across the four hierarchical decoding stages and is injected again during residual refinement. Even with the global text embedding produced from a fixed generic prompt, this repeated conditioning enables global semantic regulation at multiple feature resolutions and strengthens the interaction between progressively decoded fusion features and reintroduced encoder features. These results suggest that the performance improvement of the proposed framework does not rely solely on degradation-aware textual semantics; the hierarchical text-conditioned interaction and subsequent residual refinement themselves also contribute to the fusion performance.

Table 2. Quantitative comparison of our method under degradation-aware text guidance with Text-IF and combinations of existing image restoration (eir.) and fusion methods on source images with various types of degradations (MSRS and LLVIP datasets: low-light visible images; MFNet: low-contrast infrared images; RoadScene: overexposed visible images). (Bold: best performance)
<table><tr><td rowspan="2">Methods</td><td colspan="3">MSRS Dataset</td><td colspan="3">LLVIP Dataset</td><td colspan="3">MFNet Dataset</td><td colspan="3">RoadScene Dataset</td></tr><tr><td>CLIP-IQA↑</td><td>EN↑</td><td>NIQE↓</td><td>EN↑</td><td>NIQE↓</td><td>MUSIQ↑</td><td>SD↑</td><td>EN↑ MUSIQ↑</td><td>SF↑</td><td>NIQE↓</td><td></td><td>BRISQUE↓</td></tr><tr><td>eir.+UMF-CMGR</td><td>0.101</td><td>6.316</td><td>3.738</td><td>|7.087</td><td>3.891</td><td>47.543</td><td>23.6845.414</td><td></td><td>34.113</td><td>11.047</td><td>3.792</td><td>32.485</td></tr><tr><td>eir.+TarDAL</td><td>0.082</td><td>5.855</td><td>4.750</td><td>7.042</td><td>3.659</td><td>41.735</td><td>33.4546.142</td><td></td><td>25.120</td><td>11.789</td><td>3.667</td><td>32.436</td></tr><tr><td>eir.+ReCoNet</td><td>0.117</td><td>7.216</td><td>5.769</td><td>7.109</td><td>4.695</td><td>44.187</td><td>41.6545.161</td><td></td><td>29.299</td><td>10.312</td><td>4.785</td><td>37.775</td></tr><tr><td>eir.+MURF</td><td>0.111</td><td>5.872</td><td>4.199</td><td>6.757</td><td>4.177</td><td>50.589</td><td></td><td>23.7415.601</td><td>35.626</td><td>15.605</td><td>3.779</td><td>30.594</td></tr><tr><td>eir.+U2Fusion</td><td>0.127</td><td>6.724</td><td>3.997</td><td>7.439</td><td>3.969</td><td>48.481</td><td></td><td>33.9405.740</td><td>34.255</td><td>18.006</td><td>4.215</td><td>34.577</td></tr><tr><td>eir.+MetaFusion</td><td>0.106</td><td>7.302</td><td>3.584</td><td>7.495</td><td>3.722</td><td>49.620</td><td></td><td>42.0266.665</td><td>34.762</td><td>26.653</td><td>3.473</td><td>29.500</td></tr><tr><td>eir.+DDFM</td><td>0.094</td><td>6.723</td><td>3.465</td><td>7.150</td><td>5.184</td><td>35.933</td><td>30.465</td><td>6.480</td><td>26.902</td><td>10.493</td><td>3.717</td><td>32.334</td></tr><tr><td>Text-IF [9]</td><td>0.132</td><td>7.172</td><td>3.708</td><td>7.391</td><td>3.502</td><td>48.625</td><td></td><td>43.933 6.683</td><td>35.650</td><td>17.766</td><td>3.342</td><td>29.021</td></tr><tr><td>IT-TextFusion (Ours)</td><td>0.145</td><td>6.689</td><td>3.906</td><td>7.502</td><td>3.336</td><td>54.067</td><td></td><td>65.1414.137</td><td>33.942</td><td>15.332</td><td>3.330</td><td>27.571</td></tr></table>

## 4.5. Comparison with Degradation-Aware Text Guidance

## 4.5.1. Evaluation on Standard Datasets

After evaluating the model using a fixed generic prompt, we further investigate its performance with degradation-aware text guidance, where the input prompts explicitly describe the degradation characteristics of the source images. In real-world scenarios, source images often suffer from various types of degradations, such as poor illumination, noise, and low contrast. Among the compared methods introduced above, Text-IF [9] is the only framework that explicitly incorporates textual guidance and is designed to handle diverse degradation conditions in a unified manner. In contrast, the other comparison methods are conventional image fusion approaches that rely solely on visual information and are not specifically designed to address degraded inputs. Therefore, to ensure a fair comparison under degradation scenarios, we follow common practice by combining existing image fusion methods with appropriate image restoration models as preprocessing steps. SOTA image restoration models for different degradations include URetinex-Net [49] for low-light image enhancement, AirNet [50] for contrast enhancement, GDID [51] for denoising, and LMPEC [52] for overexposure correction.

Quantitative Comparison. The quantitative results on the four standard infrared-visible fusion datasets are reported in Tab. 2. Overall, the proposed method exhibits different advantages and trade-offs across datasets and evaluation metrics, rather than uniformly outperforming all competing methods.

On MSRS, IT-TextFusion achieves the highest CLIP-IQA score (0.145), indicating superior performance under this no-reference perceptual-quality metric. However, its EN and NIQE are not the

Text: This is the infrared-visible lightfusion task, where visible images are affected by blur degradation.

![](images/c42de16412584cf6448ab2799d0406732d1b436083e2025e1549a8c008b2c465.jpg)  
Infrared

![](images/0425741aea6eb472a08ac4f821dcee4d21b20942e2304a3b977e6b45dae8dc4a.jpg)  
Visible

![](images/aa059c5fd3fc614da94b3fff58335a1470639644bb2b8130462ec5f126c1d542.jpg)  
Text-IF

![](images/c021cdf9e4005ad07241f1769d286e11227895865b29ed46bd9d0f0efb9fd541.jpg)  
Ours

Text: This is the infrared visible lightfusion task.   
Infrared images have the low contrast degradation.

![](images/55f4c310b5f20d20860b6f9cb9ae0cdc9e438cce008ca1bfa538fd70f6bee985.jpg)  
Infrared

![](images/300726db763f28c41056a40492084a5caa446ab0706436a8251b9fe58361c3c7.jpg)  
Visible

![](images/605a111e5dba9a18d8f5895cdde78c43e4bde057c5a6afc7122f59af3dbdaa92.jpg)  
Text-IF

![](images/c97c6c14bed4d6655e4faa8201b78e5fae4b7223a48a14d809bb64c0311a4173.jpg)  
Ours

Text: This is the infrared visible light fusion task. Visible Text: This is the infrared visible light fusion task. Visible images have the overexposure degradation. images have the low light degradation.  
![](images/64d7b26bbf9dc52d205e5860915cb7fea0bde4d593f5a1168f9a546935234d06.jpg)  
Infrared

![](images/86df093ceff38c8d16f587c637f3c163d81118ecb7c1c6a68e0a469d6940dc62.jpg)  
Visible

![](images/56c2e8ea2dc39c97c434a842a9ad7a0745be9964ea043c3bd2ada274fe4981ed.jpg)  
Text-IF

![](images/07ae898164df9f3330f71047f0c702229a205e0b885a4fd175f5527e953cabbf.jpg)  
Ours

![](images/514c6ea26ee575b3e3e07228c9ca08f3e0218c8b6b9f0e7ae08192cb6701d0ef.jpg)  
Infrared

![](images/1efdf181b031cbc1fe262a760f16b8e77ced7c0d4280e366639d09fbd24f812f.jpg)  
Visible

![](images/62d4ad85e639cb6c935b7795585b1335ee257dc8cef88cb2d1578afe86401d9b.jpg)  
Text-IF

![](images/063b33a7ce03430320100321d4fae5d586f343f4b6d5a454030ffd8934889578.jpg)  
Ours

Text: This is the infrared-visible lightfusion task, where Text: This is the infrared-visible lightfusion task, where visible images are affected by haze degradation. visible images are affected by rain degradation.  
![](images/94c204cb8502370f5dac1d09718d8240663d6510ac8af7d02e7f3240efa6c827.jpg)  
Infrared

![](images/ce1d2170e3fdc449dfb9072017fe8f045922590ef0db2e660f90d7a262bdad1c.jpg)  
Visible

![](images/a29c8af34b5f7adcb9a55a8ff401b756246658ed2aea0dcd69f94940195c74bd.jpg)  
Text-IF

![](images/5694a0efe32903c2eac581bec737cc83ecef3e90e0f5581b5c8ba6b7510f1ec7.jpg)  
Ours

![](images/5e9ffe20ecd41f5313ff2b8b0aa50e44edc7f95f604aed5a046ceb8b7abc78b3.jpg)  
Infrared

![](images/a998d72c984eabad12bd9faa6f8529f26edaf41d162d7cda09890e27fd5cb4e4.jpg)  
Visible

![](images/9a64315d5f0c4602ce0568bfc4d0e435cf12880ce21bf7947a7f6fc94547fbe6.jpg)  
Text-IF

![](images/2c2bcc15b3d67fbd6a2f0e11029a2939c9e61db5fd92d302447c1087fb1c2cce.jpg)  
Ours

Figure 6. Qualitative comparison on the EMS dataset under six degradation types. Each group contains the infrared input, visible input, the result of Text-IF [9], and our result. The sentence above each group denotes the degradation-aware text prompt used for text-guided fusion.

best among the compared methods, indicating a trade-off between semantic/perceptual quality and conventional information or naturalness measures.

On LLVIP, IT-TextFusion achieves the highest EN (7.502) and MUSIQ (54.067), together with the lowest NIQE (3.336). These results indicate improvements in information content and selected no-reference perceptual-quality criteria, although the proposed method does not achieve the best result for every fusion metric.

On MFNet, IT-TextFusion obtains the highest SD (65.141), indicating stronger global intensity variation. However, its EN (4.137) is lower than that of Text-IF (6.683), and its MUSIQ (33.942) is also lower than the best competing score. Therefore, the main advantage on MFNet is reflected in global contrast rather than in entropy or no-reference perceptual quality.

On RoadScene, the proposed method achieves the lowest NIQE (3.330) and BRISQUE (27.571), whereas its SF (15.332) is lower than the best competing score. Thus, the method shows advantages in selected no-reference image-quality criteria but does not dominate all fusion metrics.

Qualitative Comparison. As shown in Fig. 5, the proposed method demonstrates clear qualitative improvements over Text-IF [9] on three standard datasets (LLVIP, MFNet, and RoadScene), while the visual difference on MSRS is relatively limited. These observations are generally consistent with the quantitative results in Tab. 2. On LLVIP, where visible images suffer from severe low-light degradation, our method produces fusion results with more appropriate brightness, improved color saturation, and clearer structural details compared with Text-IF. For MFNet, which mainly contains low-contrast infrared images, the improvement is particularly evident in terms of target-background separation. This visual enhancement is consistent with the significant increase in SD, indicating that our model effectively enhances global contrast and intensity variation, making thermal targets more distinguishable. On RoadScene, our method better alleviates overexposure effects by expanding the dynamic range of the fusion results, leading to more balanced brightness and clearer textures. In contrast, on MSRS, both methods produce visually comparable fusion results, suggesting that the performance margin under this degradation setting is relatively limited.

Table 3. Average quantitative comparison of our method with Text-IF [9] on the EMS dataset. For each metric, results are first averaged within each degradation type and then equally averaged across the nine degradation types. ↑ indicates that a higher value is better, and ↓ indicates that a lower value is better. (Bold: better result)
<table><tr><td rowspan="2">Method</td><td colspan="8">Fusion Metrics</td><td colspan="4">Perceptual Metrics</td></tr><tr><td>EN↑</td><td>SD↑</td><td>SF↑</td><td>MI↑</td><td>SCD↑</td><td>VIFF↑</td><td> $\boldsymbol { Q } ^ { A B / F } \uparrow$ </td><td> $\mathrm { S S I M _ { \mathrm { s u m } } \uparrow }$ </td><td>NIQE↓</td><td>MUSIQ↑</td><td>BRISQUE↓</td><td>CLIP-IQA↑</td></tr><tr><td>Text-IF [9]</td><td>7.287</td><td>48.372</td><td>15.899</td><td>14.481</td><td>1.447</td><td>0.784</td><td>0.566</td><td>1.198</td><td>3.405</td><td>46.999</td><td>26.929</td><td>0.224</td></tr><tr><td>IT-TextFusion (Ours)</td><td>7.296</td><td>48.875</td><td>16.402</td><td>14.474</td><td>1.525</td><td>0.809</td><td>0.577</td><td>1.195</td><td>3.373</td><td>47.125</td><td>25.747</td><td>0.212</td></tr></table>

## 4.5.2. Evaluation on EMS Dataset

We evaluate our method on the EMS dataset under diverse degradation conditions. Following the evaluation protocol adopted by Text-IF [9], we report eight fusion-related metrics: EN, SD, SF, MI, SCD, VIFF, $Q ^ { A B / F }$ , and $\mathrm { S S I M _ { \mathrm { s u m } } }$ . These metrics evaluate complementary properties of the fusion results, including information content, contrast, spatial activity, source-information preservation, structural consistency, and visual fidelity.

Quantitative Comparison. As shown in Tab. 3, our method outperforms Text-IF on several fusionrelated metrics, including EN, SD, SF, SCD, $Q ^ { A B / F }$ , and VIFF. The improvements in SCD, VIFF, and $Q ^ { A B / F }$ suggest that the proposed method can preserve complementary source information and structural details more effectively under diverse degradation conditions. However, the gains are metric-dependent: Text-IF achieves slightly higher MI and $\mathrm { S S I M _ { \mathrm { s u m } } }$ , while the difference in EN is relatively small. Therefore, the results should be interpreted as a set of metric-specific improvements rather than uniform superiority across all fusion criteria.

NIQE, MUSIQ, BRISQUE, and CLIP-IQA assess no-reference perceptual quality and image naturalness from different statistical or learned perspectives and are not direct measures of multi-modal information preservation. In particular, although CLIP-IQA uses CLIP-derived representations, the adopted implementation receives only the fused image and therefore does not directly measure alignment between the fusion result and the input degradation-aware prompt. Their values may also be affected by the statistical characteristics of natural images and may not fully reflect the effectiveness of infrared-visible image fusion. Accordingly, these metrics are reported as complementary evaluation criteria and are not combined with the fusion-related metrics into a single aggregate score. Compared with Text-IF, our method obtains lower NIQE and BRISQUE values and a higher MUSIQ score, while Text-IF achieves a higher CLIP-IQA score. These results indicate that the proposed method improves several aspects of perceptual quality but also exhibits trade-offs across different no-reference evaluation criteria.

Table 4. Ablation study of the proposed method on the LLVIP dataset. (Bold indicates the best performance)
<table><tr><td>TG -RRM</td><td>Deg.-Aware Text Eng.</td><td> $\mathrm { C r o s s \mathrm { - } G a t e }$  ITIM Fusion</td><td>EN SF</td><td>VIFF</td><td> $Q ^ { A B / F }$  NIQE</td></tr><tr><td>X</td><td>X</td><td>X X</td><td>7.443</td><td>316.153 1.045</td><td>0.751 3.600</td></tr><tr><td>X</td><td>X</td><td>X V</td><td>7.44416.2961.040</td><td>0.752</td><td>3.472</td></tr><tr><td>√</td><td>X</td><td>X √</td><td>7.44016.4111.041</td><td></td><td>0.765 3.353</td></tr><tr><td>√</td><td>X</td><td>√ √</td><td>7.43916.3941.043</td><td></td><td>0.758 3.363</td></tr><tr><td>V</td><td>V</td><td>√ V</td><td>7.502 16.464 1.037</td><td></td><td>0.765 3.336</td></tr></table>

Qualitative Comparison. The qualitative results on the EMS dataset under six representative degradation scenarios are shown in Fig. 6. Compared with Text-IF [9], the proposed method produces more stable and visually consistent fusion results across different degradation types.

For the Visible Blur, Low Light, Rain, and Overexposure degradation scenarios, visible images suffer from severe degradation in contrast, brightness, and texture, which significantly increases the difficulty of reliable fusion. Text-IF can partially enhance scene visibility; however, the fusion results still show constrained brightness, residual saturation, or reduced color vividness in degraded regions, indicating that the loss of visible information is not fully compensated for during fusion. By contrast, our method produces more balanced fusion results with more appropriate brightness, clearer structural details, and better preserved color consistency, allowing salient targets and scene content to remain distinguishable even under severe degradations. These representative examples suggest improved visual robustness under several visible-side degradation conditions. For low-contrast infrared images, thermal targets are weakly separated from the background, making effective fusion particularly challenging. In this case, the fusion results of our method are visually comparable to those of Text-IF [9], and both methods exhibit limited contrast enhancement. A plausible reason is that the degradation mainly affects the intrinsic contrast of the infrared modality, where discriminative thermal cues are inherently weak and difficult to recover through fusion alone. Without explicit infrared contrast enhancement priors, both methods tend to preserve the original infrared intensity distribution, resulting in similar fusion performance under low-contrast infrared conditions.

## 4.6. Ablation Study

To further analyze the contribution of each component in the proposed framework, we conduct a series of ablation experiments on the LLVIP dataset. Starting from a basic fusion backbone, key modules are progressively introduced, including degradation-aware text engineering, ITIM, Cross-Gate Fusion, and TG-RRM. Here, disabling degradation-aware text engineering does not remove the textual branch; instead, the fixed generic prompt is used in place of degradation-aware semantic prompts. The results are reported in Tab. 4 and Fig. 7.

![](images/2fa9f731eb67dcc2ef52dcfa2f993e48670597fbe8179fbb7d222f101d1f80bf.jpg)  
(a) Visible

![](images/525b78afb974ccb347d81c8c8b782a86983a035919bd3acad41c9ad616a82753.jpg)  
(b) Original

![](images/2e214feea1cb8ebefac8e5232e3af13fe46b491bffb32693a3c0aec141431561.jpg)  
(c) Only ITIM

![](images/9401b7930f78d0e9bde30652eec1ba794ba062e6a2912d7e3981d9b0080da785.jpg)  
(d) ITIM+TG-RRM

![](images/07d4a4c44ed49d5f12be4f2b32d891d077207fbc041ef478f89d87e40f8fc69b.jpg)  
(e) w/o Deg.-Aware Text Eng.

![](images/8f2d2c9fa2d599f0085bce78f487cd3939c37f2a100f0c1851d9bd290fd3bf6a.jpg)  
(f) Complete Model  
Figure 7. Qualitative comparison of different ablation variants on the LLVIP dataset.

The quantitative results reveal metric-dependent contributions from the proposed components. Under the setting with a fixed generic prompt, ITIM slightly improves EN, SF, and $Q ^ { A B / F }$ while reducing NIQE, indicating that iterative text-image interaction is beneficial even without degradation-specific semantics. Adding TG-RRM further improves SF and $Q ^ { A B / F }$ and achieves a lower NIQE, supporting its role in structural refinement and artifact suppression. Cross-Gate Fusion slightly improves VIFF relative to the preceding variant but causes small decreases in several other metrics, indicating a trade-off rather than a uniform gain. Finally, degradation-aware text engineering leads to clear improvements in EN, SF, $Q ^ { A B / F }$ , and NIQE, although VIFF decreases. Consequently, the complete model achieves the best EN, SF, and NIQE and ties for the best $Q ^ { A B / F }$ , but it does not outperform the baseline model in VIFF. These findings suggest that the proposed components jointly balance information preservation, structural fidelity, and perceptual naturalness rather than improving every metric independently. The qualitative comparisons show that different components exhibit distinct effects on fusion quality, while the complete model produces more balanced results with clearer structures and better visual consistency. Overall, the complete model achieves the best results on most metrics, indicating that the combined use of these components contributes complementary improvements to overall performance.

## 5. Conclusion

In this paper, we investigate text-guided infrared-visible image fusion under diverse degradation conditions and propose a unified framework that introduces global textual guidance into hierarchical feature fusion and refinement. Unlike existing text-guided fusion methods that rely on relatively shallow semanticvisual interaction, our approach maintains global text conditioning across hierarchical decoding and residual feature refinement. Specifically, we enhance prompt expressiveness via task- and degradationaware text engineering, introduce stage-specific text-conditioned modulation to inject the global text representation into hierarchical decoder features, and design a Cross-Gate Fusion block for more robust and content-adaptive multi-modal feature integration. Moreover, we incorporate a lightweight Text-Guided Residual Refinement Module to further improve detail preservation and visual consistency under severe degradations. Extensive experiments on standard fusion benchmarks and the multi-degradation EMS dataset demonstrate that the proposed method achieves competitive overall performance, with improvements on several fusion and perceptual-quality metrics and trade-offs on others. The ablation results indicate that hierarchical text conditioning, Cross-Gate Fusion, TG-RRM, and degradation-aware prompts can make complementary and synergistic contributions to model performance.

## Data availability statement

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## Declaration of generative AI and AI-assisted technologies

During the preparation of this manuscript, the authors used generative AI tools only to improve language and readability. Specifically, the authors used ChatGPT for rewriting in substantial portions of the manuscript. The authors take full responsibility for the content of the manuscript.

## Conflicts of interest

The authors declares no conflicts of interest.

## References

[1] Cao H, Chen G, Zhao H, Jiang D, Zhang X, et al. SDPT: Semantic-Aware Dimension-Pooling Transformer for Image Segmentation. IEEE Transactions on Intelligent Transportation Systems 2024, 25(11):15934–15946.

[2] Cao H, Chen G, Xia J, Zhuang G, Knoll A. Fusion-Based Feature Attention Gate Component for Vehicle Detection Based on Event Camera. IEEE Sensors Journal 2021, 21(21):24540–24548.

[3] Huang K, Shi B, Li X, Li X, Huang S, et al. Multi-Modal Sensor Fusion for Autonomous Driving Perception: A Survey. arXiv preprint arXiv:2202.02703 2022.

[4] Zhang J, Liu H, Yang K, Hu X, Liu R, et al. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers. IEEE Transactions on Intelligent Transportation Systems 2023, 24(12):14679–14694.

[5] Cao H, Sun D, Song R, Xia Y, Li X, et al. Feature-aligned Fisheye Object Detection Network for Autonomous Driving. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2025 pp. 7781–7787.

[6] Zhang J, Cao M, Xie W, Lei J, Li D, et al. E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection. Advances in Neural Information Processing Systems 2024, 37:52296–52322.

[7] Cui C, Ma Y, Cao X, Ye W, Zhou Y, et al. A Survey on Multimodal Large Language Models for Autonomous Driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2024 pp. 958–979.

[8] Cao H, Zhang Z, Xia Y, Li X, Xia J, et al. Embracing Events and Frames With Hierarchical Feature Refinement Network for Object Detection. In Proceedings of the European Conference on Computer Vision (ECCV). 2024 pp. 161–177.

[9] Yi X, Xu H, Zhang H, Tang L, Ma J. Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024 pp. 27016–27025.

[10] Zhang J, Liu R, Shi H, Yang K, Reiß S, et al. Delivering Arbitrary-Modal Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023 pp. 1136–1147.

[11] Chen YT, Shi J, Ye Z, Mertz C, Ramanan D, et al. Multimodal Object Detection via Probabilistic Ensembling. In Proceedings of the European Conference on Computer Vision (ECCV). 2022 pp. 139–158.

[12] Cao H, Chen G, Li Z, Hu Y, Knoll A. NeuroGrasp: Multimodal Neural Network With Euler Region Regression for Neuromorphic Vision-Based Grasp Pose Estimation. IEEE Transactions on Instrumentation and Measurement 2022, 71:1–11.

[13] Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, et al. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the International Conference on Machine Learning (ICML). 2021 pp. 8748–8763.

[14] Yang Z, Li Y, Tang X, Xie M. MGFusion: A Multimodal Large Language Model-Guided Information Perception for Infrared and Visible Image Fusion. Frontiers in Neurorobotics 2024, 18:1521603.

[15] Cheng C, Xu T, Wu XJ, Li H, Li X, et al. TextFusion: Unveiling the Power of Textual Semantics for Controllable Image Fusion. Information Fusion 2025, 117:102790.

[16] Li K, Li H, Cui M, Li J, Lv P, et al. TITFormer: Combining Textual Modality and Simulating Infrared Modality Based on Transformer for Image Enhancement. IEEE Transactions on Multimedia 2025, 27:4725–4735.

[17] Xu H, Ma J, Jiang J, Guo X, Ling H. U2Fusion: A Unified Unsupervised Image Fusion Network. IEEE Transactions on Pattern Analysis and Machine Intelligence 2020, 44(1):502–518.

[18] Zhao W, Xie S, Zhao F, He Y, Lu H. MetaFusion: Infrared and Visible Image Fusion via Meta-Feature Embedding From Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023 pp. 13955–13965.

[19] Zhao Z, Bai H, Zhu Y, Zhang J, Xu S, et al. DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023 pp. 8048–8059.

[20] Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2015 pp. 3431–3440.

[21] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Seg-

mentation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). 2015 pp. 234–241.

[22] Wang H, Zhang H, Yi X, Xiang X, Fang L, et al. TeRF: Text-Driven and Region-Aware Flexible Visible and Infrared Image Fusion. In Proceedings of the 32nd ACM International Conference on Multimedia (ACM MM). 2024 pp. 935–944.

[23] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, et al. Attention Is All You Need. Advances in Neural Information Processing Systems 2017, 30.

[24] Dosovitskiy A. An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929 2020.

[25] Cao H, Qu Z, Chen G, Li X, Thiele L, et al. GhostViT: Expediting Vision Transformers via Cheap Operations. IEEE Transactions on Artificial Intelligence 2023, 5(6):2517–2525.

[26] Huang X, Belongie S. Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017 pp. 1510– 1519.

[27] Rao Y, Zhao W, Chen G, Tang Y, Zhu Z, et al. DenseCLIP: Language-Guided Dense Prediction With Context-Aware Prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 18061–18070.

[28] Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. High-Resolution Image Synthesis With Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 10674–10685.

[29] Zamir SW, Arora A, Khan S, Hayat M, Khan FS, et al. Restormer: Efficient Transformer for High-Resolution Image Restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 5728–5739.

[30] Sun S, Ren W, Gao X, Wang R, Cao X. Restoring Images in Adverse Weather Conditions via Histogram Transformer. In European Conference on Computer Vision (ECCV). 2024 pp. 111–129.

[31] Cheng Y, Knoll A, Cao H. URNet: Uncertainty-aware Refinement Network for Event-based Stereo Depth Estimation. Visual Intelligence 2025, 3(1):18.

[32] Zhao H, Gallo O, Frosio I, Kautz J. Loss Functions for Image Restoration With Neural Networks. IEEE Transactions on Computational Imaging 2016, 3(1):47–57.

[33] Tang L, Yuan J, Zhang H, Jiang X, Ma J. PIAFusion: A Progressive Infrared and Visible Image Fusion Network Based on Illumination Awareness. Information Fusion 2022, 83:79–92.

[34] Jia X, Zhu C, Li M, Tang W, Zhou W. LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021 pp. 3489–3497.

[35] Ha Q, Watanabe K, Karasawa T, Ushiku Y, Harada T. MFNet: Towards Real-Time Semantic Segmentation for Autonomous Vehicles With Multi-Spectral Scenes. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2017 pp. 5108–5115.

[36] Wang D, Liu J, Fan X, Liu R. Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration. arXiv preprint arXiv:2205.11876 2022.

[37] Liu J, Fan X, Huang Z, Wu G, Liu R, et al. Target-Aware Dual Adversarial Learning and a

Multi-Scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 5792–5801.

[38] Huang Z, Liu J, Fan X, Liu R, Zhong W, et al. ReCoNet: Recurrent Correction Network for Fast and Efficient Multi-Modality Image Fusion. In Proceedings of the European Conference on Computer Vision (ECCV). 2022 pp. 539–555.

[39] Xu H, Yuan J, Ma J. MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence 2023, 45(10):12148–12166.

[40] Ma J, Ma Y, Li C. Infrared and Visible Image Fusion Methods and Applications: A Survey. Information Fusion 2019, 45:153–178.

[41] Eskicioglu AM, Fisher PS. Image Quality Measures and Their Performance. IEEE Transactions on Communications 2002, 43(12):2959–2965.

[42] Aslantas V, Bendes E. A New Image Quality Metric for Image Fusion: The Sum of the Correlations of Differences. AEU–International Journal ofElectronics and Communications 2015, 69(12):1890– 1896.

[43] Han Y, Cai Y, Cao Y, Xu X. A New Image Fusion Performance Metric Based on Visual Information Fidelity. Information Fusion 2013, 14(2):127–135.

[44] Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 2004, 13(4):600–612.

[45] Wang J, Chan KCK, Loy CC. Exploring CLIP for Assessing the Look and Feel of Images. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI). 2023 pp. 2555–2563.

[46] Mittal A, Soundararajan R, Bovik AC. Making a “Completely Blind” Image Quality Analyzer. IEEE Signal Processing Letters 2012, 20(3):209–212.

[47] Ke J, Wang Q, Wang Y, Milanfar P, Yang F. MUSIQ: Multi-Scale Image Quality Transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021 pp. 5128–5137.

[48] Mittal A, Moorthy AK, Bovik AC. No-Reference Image Quality Assessment in the Spatial Domain. IEEE Transactions on Image Processing 2012, 21(12):4695–4708.

[49] Wu W, Weng J, Zhang P, Wang X, Yang W, et al. URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 5891–5900.

[50] Li B, Liu X, Hu P, Wu Z, Lv J, et al. All-in-One Image Restoration for Unknown Corruption. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022 pp. 17431–17441.

[51] Chen H, Gu J, Liu Y, Magid SA, Dong C, et al. Masked Image Training for Generalizable Deep Image Denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023 pp. 1692–1703.

[52] Afifi M, Derpanis KG, Ommer B, Brown MS. Learning Multi-Scale Photo Exposure Correction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021 pp. 9157–9167.