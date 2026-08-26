# ReGround-Surg: Reliability-Guided Anchor Grounding for Referring Surgical Video Segmentation

Jiaxin Wen, Ming Yin, Lu Liu, and Zeyu Fu<sup>⋆</sup>

University of Exeter, Exeter, UK

Abstract. Referring surgical video segmentation requires segmenting a target instrument or tissue region across video frames according to a natural language expression. Recent Segment Anything Model 2 (SAM2) based two-stage methods (e.g., ReSurgSAM2) first ground the referred target in an initial or selected frame, then propagate the selected mask via tracking. Although efective, their performance is highly sensitive to the quality of the initial grounded mask: once an incorrect anchor is selected, subsequent tracking tends to propagate the error. This issue is especially challenging in surgical videos due to visually similar instruments, occlusion, and complex tissue-tool interactions. To address this issue, we propose ReGround-Surg, a lightweight reliability-guided anchor grounding framework to improve SAM2-based referring surgical video segmentation. It first predicts a text-conditioned spatial reliability map from the referring expression and current-frame visual features. The map is then reused in two complementary branches: a Gated Side Adapter enhances expression-relevant visual regions before textto-vision fusion, while a Reliability-Weighted Vision-to-Text Attention module suppresses of-target visual evidence during prompt-token aggregation. Experiments on Ref-EndoVis17 and Ref-EndoVis18 show consistent improvements over state-of-the-art methods across three evaluation splits with negligible speed reduction. Code is publicly available at https://github.com/JiaxinWen1/ReGround-Surg.

Keywords: Referring Surgical Instrument Segmentation · SAM2 · Reliability Gating

## 1 Introduction

Surgical video segmentation plays a fundamental role in computer-assisted intervention by providing precise, pixel-level identification of surgical instruments and tissue regions. Existing methods focus on category-level segmentation, evolving from semantic segmentation [18] to instance-based approaches [7], transformerbased tracking [28], and temporal consistency modeling [3]. However, these methods are inherently limited in flexibility, as they cannot distinguish between instruments of the same type based on context or respond to free-form user queries.

![](images/a3ecf3bcdb8322b74674a3863f3f731fb2553c4e39c525a55dcdfad3ad29abae.jpg)  
Fig. 1. Visualisation of ReSurgSAM2 initial frame grounding and subsequent tracking failures under multi-instrument confusion, showing per-frame segmentation results for the expression “large needle driver is manipulating tool” across four conditions: ground truth, ReSurgSAM2, ReSurgSAM2 (w/o Stage II), and ReGround-Surg (Ours, w/o Stage II).

To address this limitation, referring surgical video segmentation aims to segment surgical instruments or tissue regions in a video according to a natural language expression. Compared with category-level surgical instrument segmentation, referring segmentation provides a more flexible interface because the target can be specified by object category, spatial location, action, or tissue relation. This is directly useful for interactive surgical video analysis, robotic assistance, and downstream skill assessment [13]. RSVIS [20] establishes the foundations of this task by jointly learning video-level and instrument-level representations for cross-modal grounding. SurgRef [21] further shows that motion dynamics provide more robust grounding cues than static appearance alone, yet still relies on per-frame processing without exploiting video-level propagation.

Recent Segment Anything Model (SAM) based methods, such as ReSurgSAM2 [11], have shown promising performance on referring surgical video segmentation. They commonly follow a two-stage detection-and-tracking paradigm: Stage I grounds the referred target and selects a credible anchor frame with an anchor mask, while Stage II propagates this mask to the remaining frames using the tracking memory of SAM2 [17]. Despite this efective design, the two-stage pipeline remains vulnerable to errors in the initial grounding stage. As illustrated in Fig. 1, given the expression “large needle driver is manipulating tool”, the baseline incorrectly grounds an adjacent instrument in the anchor frame, and the tracking stage subsequently propagates this error throughout the video. To determine whether this failure arises from tracking or from initial grounding, we evaluate a Stage II-disabled variant that directly uses the Stage I detection mask without temporal propagation. The incorrect prediction persists, as shown in the third row of Fig. 1, indicating that the error primarily stems from erroneous anchor selection in Stage I. This reveals a key limitation of current SAM2-based referring surgical video segmentation methods: strong tracking cannot compensate for inaccurate initial grounding. Once a wrong anchor mask is selected, the error is propagated throughout the video, especially under instrument similarity, occlusion, and tissue-tool interactions.

![](images/0aed8fda2455be11dd50da6210840784f6558713175c9f9f0dde04bd981b76ff.jpg)  
Fig. 2. Overview of our proposed ReGround-Surg for Referring Surgical Video Segmentation.

These results highlight that robust referring surgical video segmentation demands reliable cross-modal grounding prior to tracking. Rather than modifying the SAM2 tracking stage, we focus on Stage I and propose ReGround-Surg, a lightweight reliability-guided anchor grounding framework that enhances expression-relevant regions and suppresses distractors during anchor selection, as illustrated in Fig. 2. Specifically, a Text-Guided Reliability Gate first predicts a text-conditioned spatial reliability map from max-pooled noun token features and compressed current-frame visual features. This map is shared by two complementary modules: a Gated Side Adapter (GSA) that modulates visual features before text-to-vision attention, encouraging the model to focus on regions consistent with the referring expression; and a Reliability-Weighted Vision-to-Text (RW-V2T) attention mechanism that reuses the same reliability map as a multiplicative weight on visual keys when aggregating visual features into the prompt token, reducing the influence of of-target or unreliable regions on grounding.

The main contributions of this paper are as follows: 1) We analyse the initial grounding bottleneck in the current SAM2-based two-stage referring surgical video segmentation and provide quantitative and qualitative evidence that the inaccurate anchor mask selection is associated with persistent tracking failures in two-stage pipelines. 2) We propose ReGround-Surg, in which a shared text-conditioned reliability map is estimated and used by two complementary modules: a GSA for visual feature modulation and an RW-V2T attention for suppressing of-target visual evidence during prompt-token aggregation. 3) Experiments on Ref-EndoVis17 and Ref-EndoVis18 show consistent improvements of +3.77, +3.09, and +0.94 over ReSurgSAM2 across three evaluation splits, with only 0.5 M additional parameters and negligible speed overhead.

## 2 Related Work

## 2.1 Foundation Models for Surgical Video Segmentation

SAM [9] and SAM2 [17] have demonstrated strong promptable and zero-shot segmentation capabilities, inspiring a range of surgical-specific adaptations. SurgicalSAM [27] adapts SAM for surgical instrument segmentation via class-level prompts. SurgicalSAM2 [12] further extends SAM2 to real-time surgical video segmentation via frame pruning strategies, while MA-SAM2 [26] introduces memory augmentation to improve long-sequence tracking stability. Despite these advances, these methods primarily rely on visual prompts such as points or boxes, limiting their ability to dynamically specify targets through natural language instructions in interactive surgical settings.

## 2.2 Referring Surgical Video Segmentation

Referring surgical video segmentation introduces natural language expressions to specify target instruments or tissue regions. In the general domain, referring video object segmentation (RVOS) methods such as ReferFormer [23], MUTR [24], and SAMWISE [6] have established strong baselines for languageguided video segmentation. However, these methods are designed for natural scenes and generalise poorly to surgical settings due to instrument similarity, occlusion, and challenging imaging conditions. In the surgical domain, RSVIS [20] introduces the referring surgical video instrument segmentation task and proposes VIS-Net with graph-based relation-aware representation learning. ReSurgSAM2 [11] combines CLIP text features [16] with SAM2 video propagation in a two-stage detection-and-tracking framework, where an anchor mask is first grounded through credible initial frame selection and then propagated via SAM2 memory. More recently, SurgRef [21] argues that existing methods over-rely on static visual cues and predefined instrument names, and proposes motion-guided grounding to improve robustness under occlusion, semantic ambiguity, and cross-institutional scenarios. Despite these advances, SAM2-based two-stage pipelines share a common assumption that the initial anchor grounding is reliable. When this assumption fails, the tracking stage propagates the error throughout the sequence with no mechanism for correction. This grounding bottleneck motivates the focus of our work.

## 2.3 Cross-Modal Grounding and Feature Modulation

To address the grounding bottleneck identified above, we draw on two complementary lines of work on cross-modal feature modulation. LAVT [25] introduces text-guided spatial gating over intermediate visual feature maps to suppress irrelevant regions, but treats all spatial responses uniformly without distinguishing which are trustworthy under occlusion or instrument confusion. XMem [5] demonstrates that explicitly assigning diferent weights to diferent elements is more efective than uniform aggregation, yet the reliability estimation relies solely on visual cues without any language conditioning to identify expressionrelevant regions. AdaptFormer [4] ofers a complementary perspective, showing that zero-initialised side adapters can modulate pre-trained representations efficiently; yet it is designed for unimodal vision and lacks any mechanism for cross-modal alignment or reliability-aware suppression. These limitations suggest that neither line of work alone sufices for reliable cross-modal grounding under surgical complexity. Building on these insights, our work introduces lightweight reliability-anchor grounding to improve the efectiveness of SAM2-based referring surgical video segmentation.

## 3 Method

## 3.1 Preliminaries

Given a surgical video $\boldsymbol { \nu } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ and a referring expression Q, the goal is to predict binary masks $\mathcal { M } = \{ M _ { t } \} _ { t = 1 } ^ { T }$ , where $M _ { t } \ \in \ \{ 0 , 1 \} ^ { H \times W }$ indicates the referred instrument or tissue region in frame $I _ { t }$

Our method builds upon ReSurgSAM2 [11], a two-stage framework for referring surgical video segmentation. In Stage I, CLIP text features [16] are fused with SAM2 visual features via a Cross-modal Spatial-Temporal Mamba (CST-Mamba) module to produce per-frame segmentation masks. A Credible Initial Frame Selection (CIFS) strategy then identifies a reliable anchor frame $\hat { M } _ { a }$ from a sliding window of high-confidence detections. In Stage II, this anchor mask is propagated across the remaining frames via SAM2’s memory-based tracking. Consequently, the accuracy of Stage I grounding is critical, as it directly determines the quality of the entire output sequence.

The CSTMamba module fuses text and visual features through a TwoWayTokenTransformer with $L { = } 3$ stacked blocks, each performing text selfattention, a Mamba temporal layer [8], text-to-vision (T2V) cross-attention, an MLP step, and vision-to-text (V2T) cross-attention. We identify two structural limitations that undermine grounding reliability. In the $T \to V$ direction, all $H \times W$ spatial positions are treated uniformly, allowing difuse activations that the mask decoder cannot resolve. In the $V {  } T$ direction, all visual tokens contribute equally to the prompt token [CLS’], regardless of occlusion or background clutter, potentially corrupting the grounding signal. As confirmed by Fig. 1, disabling Stage II does not resolve grounding failures, directly implicating CSTMamba as the source of error. To address these gaps, we propose a lightweight reliability-guided framework for improving Stage 1 grounding in SAM2-based referring surgical video segmentation as illustrated in Fig. 2.

![](images/4c7d51d60e350d19acb007e5ed1d383daf299198bcbd78850661510de8ba98ee.jpg)  
Fig. 3. Architecture of the proposed Text-Guided Reliability Gate, Gated Side Adapter (GSA), and Reliability-Weighted V2T Attention (RW-V2T).

## 3.2 Reliability-Guided Anchor Grounding

Text-Guided Reliability Gate. The Text-Guided Reliability Gate is the shared core of our method, as illustrated in Fig. 3. It predicts a spatial reliability map $G \in [ 0 , 1 ] ^ { H \times W }$ that indicates, for each spatial position, the degree of alignment between the visual content and the referring expression. This map is then reused by both the Gated Side Adapter and the Reliability-Weighted V2T Attention.

Let $\mathbf { Q } \in \mathbb { R } ^ { N \times C }$ denote the text token sequence output from the self-attention layer, where N is the number of text tokens and C is the feature dimension. To capture the most discriminative text features (which typically correspond to instrument-related nouns), inspired by LAVT [25], we apply element-wise maxpooling over all text tokens:

$$
\mathbf { n } _ { c } = \operatorname* { m a x } _ { i = 1 } ^ { N } \mathbf { Q } _ { : , i , c } \ \in \ \mathbb { R } ^ { B \times C } ,\tag{1}
$$

Both n and the current-frame visual features $\mathbf { K } _ { \mathrm { c u r } } \in \mathbb { R } ^ { B \times C \times H \times W }$ are projected to a 64-dimensional bottleneck: $\hat { \mathbf { n } } = W _ { \mathrm { g a t e } } \mathbf { n }$ and $\hat { \mathbf { K } } = \mathrm { C o n v } _ { 1 \times 1 } ( \mathbf { K } _ { \mathrm { c u r } } )$ The reliability map is then:

$$
\begin{array} { r } { G = \sigma \mathopen { } \mathclose \bgroup \left( \sum _ { d = 1 } ^ { 6 4 } \hat { \mathbf { K } } _ { : , d , : , : } \cdot \hat { \mathbf { n } } _ { : , d } \aftergroup \egroup \right) \in [ 0 , 1 ] ^ { B \times 1 \times H \times W } , } \end{array}\tag{2}
$$

where G≈1 indicates strong text–vision alignment and G≈0 indicates poor alignment or distractor regions. G is computed independently in each of the three TwoWayTokenAttentionBlock layers with separate parameters, allowing it to adapt to progressively refined representations.

Gated Side Adapter (GSA). The reliability map is used to modulate visual features before T2V cross-attention via a soft residual gate with floor $\alpha \in [ 0 , 1 )$ 2 where $\alpha { = } 0 . 3$ is selected by grid search (Table 6):

$$
\begin{array} { r } { \tilde { \bf K } = \hat { \bf K } \cdot \big ( \boldsymbol \alpha + ( 1 - \boldsymbol \alpha ) \boldsymbol G \big ) . } \end{array}\tag{3}
$$

We use residual modulation rather than direct masking to avoid suppressing potentially useful contextual information, especially when the expression is ambiguous, or the target is partially occluded [14]. The modulated features are expanded back to $C$ channels and added as a zero-initialised [4] residual:

$$
\mathbf { K } _ { \mathrm { c u r } } ^ { \prime } = \mathrm { L a y e r N o r m } \big ( \mathbf { K } _ { \mathrm { c u r } } + \mathrm { C o n v } _ { 1 \times 1 } ^ { \uparrow } ( \tilde { \mathbf { K } } ) \big ) .\tag{4}
$$

Zero-initialisation of $\mathrm { C o n v _ { 1 \times 1 } ^ { \uparrow } }$ ensures the adapter is functionally identical to the baseline at step zero, preserving pre-trained SAM2 representations. Only current-frame features are modulated; reference-frame features selected by CIFS are left unchanged, since gating historical memory would risk undermining the stable temporal anchor.

Reliability-Weighted V2T Attention (RW-V2T). Standard V2T crossattention lets all visual tokens contribute equally to the prompt token, regardless of occlusion or background. We incorporate the reliability map G as a multiplicative weight on the visual keys, so that expression-relevant positions contribute more strongly to prompt-token aggregation. The reliability-weighted keys are computed as:

$$
\mathbf { K } _ { \mathrm { v 2 t } } = \mathrm { L a y e r N o r m } \big ( \mathbf { K } + W _ { v } ( \mathbf { K } \cdot v _ { \mathrm { f u l l } } ) \big ) ,\tag{5}
$$

where $v _ { \mathrm { f u l l } } ~ = ~ [ ( \alpha + ( 1 - \alpha ) G _ { \mathrm { f a t } } ) ~ \parallel ~ { \bf 1 } _ { ( f - 1 ) H W } ]$ , f denotes the total number of frames (one current frame plus f−1 reference frames), and $W _ { v } \in \mathbb { R } ^ { C \times C }$ is a learnable linear projection initialised as the identity matrix. The residual is added to the original K rather than to the weighted version, so downstream visual features retain their full unattenuated content (information-lossless design). No explicit supervision is imposed on the reliability gate; it is learned implicitly through the segmentation loss.

Training and Inference. To preserve the pre-trained SAM2 representation, the SAM2 image encoder and tracking memory module are frozen throughout training. Only the CrossModalFusionModule (original CSTMamba weights plus the 0.5 M parameters of the two proposed modules) is updated. The model is trained with the same composite loss as ReSurgSAM2 [11]:

$$
\mathcal { L } = 2 0 \mathcal { L } _ { \mathrm { m a s k } } + \mathcal { L } _ { \mathrm { d i c e } } + \mathcal { L } _ { \mathrm { I o U } } + \mathcal { L } _ { \mathrm { c l s } } ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { m a s k } }$ is binary cross-entropy, ${ \mathcal { L } } _ { \mathrm { d i c e } }$ provides a region-overlap objective, ${ \mathcal { L } } _ { \mathrm { I o U } }$ directly optimises the confidence score consumed by CIFS, and ${ \mathcal L } _ { \mathrm { c l s } }$ supervises the occlusion prediction head.

For inference, the proposed modules are used only in Stage I detection. The resulting anchor mask is then passed to the original SAM2 tracking stage unchanged, making our method fully plug-and-play with respect to Stage II.

Table 1. Dataset statistics.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">Training</td><td colspan="4">Testing</td></tr><tr><td> ${ \mathrm { S e q . } }$ </td><td>Frame</td><td>Object</td><td>Pair</td><td>Seq.</td><td>Frame</td><td>Object</td><td>Pair</td></tr><tr><td>Ref-EndoVis17 (tool)</td><td>7</td><td>2100</td><td>20</td><td>4873</td><td>3</td><td>900</td><td>10</td><td>2265</td></tr><tr><td>Ref-EndoVis18 (tool)</td><td>11</td><td>1639</td><td>34</td><td>3787</td><td>4</td><td>596</td><td>15</td><td>1384</td></tr><tr><td>Ref-EndoVis18 (tissue)</td><td>11</td><td>1639</td><td>25</td><td>2995</td><td>4</td><td>596</td><td>7</td><td>807</td></tr></table>

## 4 Experiments

## 4.1 Datasets and Experimental Details

Datasets. We evaluate on Ref-EndoVis17 and Ref-EndoVis18 [20], following the same splits as ReSurgSAM2 [11]. Both datasets build on EndoVis17 [2] and EndoVis18 [1], reannotated with instance-specific labels. Ref-EndoVis18 includes tissue-specific annotations. Table 1 summarises the statistics.

Metrics. We adopt region similarity $\begin{array} { r l } { \mathcal { I } } & { { } = } & { \frac { | M \cap G | } { | M \cup G | } } \end{array}$ , contour accuracy ${ \mathcal { F } } =$ $\frac { 2 P _ { c } R _ { c } } { P _ { c } + R _ { c } }$ [15], and their mean $\mathcal { I } \& \mathcal { F } = \frac { 1 } { 2 } ( \mathcal { I } + \mathcal { F } )$ following the RVOS evaluation protocol [23], where M and G denote the predicted mask and ground-truth mask respectively, and $P _ { c } , \ : R _ { c }$ are the contour precision and recall. All metrics are higher-is-better. To better understand the efect of Stage I detection quality, we additionally report Initial Grounding metrics: $\mathcal { I } , \mathcal { F }$ , and $\mathcal { T } \& \mathcal { F }$ computed on the CIFS-selected anchor frame before SAM2 tracking. We further introduce AnchorAcc, defined as the fraction of anchor masks whose IoU with ground truth exceeds 0.5:

$$
\mathrm { A n c h o r A c c } = \frac { 1 } { S } \sum _ { i = 1 } ^ { S } \mathcal { k } \Big [ \mathrm { I o U } \big ( \hat { M } _ { a } ^ { ( i ) } , M _ { a } ^ { ( i ) } \big ) > 0 . 5 \Big ] .\tag{7}
$$

where S is the number of video-text pairs, $\hat { M } _ { a } ^ { ( i ) }$ is the predicted anchor frame mask, and $\boldsymbol { M } _ { a } ^ { ( i ) }$ is the corresponding ground-truth mask.

Implementation Details. We use the Hiera-Small backbone [19] initialised from the oficial ReSurgSAM2 checkpoint [11]. Input resolution is 512×512. The SAM2 image encoder, CLIP text encoder [16], SAM2 mask decoder, and all Stage II components are frozen; only the CrossModalFusionModule is updated. The model is trained for 60 epochs with a batch size of 16 (8 GPUs, 2 per GPU) and the AdamW optimiser. The original CSTMamba parameters use a learning rate of $5 \times 1 0 ^ { - 5 }$ , while our newly introduced adapter parameters use a larger rate of $3 \times 1 0 ^ { - 4 }$ to compensate for their zero/identity initialisation [4]. The schedule follows linear warm-up (30%) then cosine decay; gradients are clipped to $\ell _ { 2 }$ norm 0.1. For inference, we adopt the default ReSurgSAM2 hyperparameters: $\delta _ { o } { = } 0 . 9$ $\delta _ { i o u } \mathrm { = } 0 . 7 , \gamma _ { i o u } \mathrm { = } 0 . 9 5 , N _ { w } \mathrm { = } 5 , N _ { p } \mathrm { = } 5 , N _ { l } \mathrm { = } 4$ (see [11] for definitions).

Table 2. Comparison with state of the art on Ref-EndoVis17 and Ref-EndoVis18. †: ofline method. Bold: best; underline: second best. ∆: gain over ReSurgSAM2. FPS measured on NVIDIA A100-SXM4-80GB.
<table><tr><td></td><td colspan="4">EV17 (tool)</td><td colspan="3">EV18 (tool)</td><td colspan="3">EV18 (tissue)</td><td></td></tr><tr><td>Method</td><td></td><td>J&amp;F</td><td>J</td><td>F</td><td>J&amp;F</td><td>J</td><td>F</td><td>J&amp;F</td><td>J</td><td>F</td><td>FPS</td></tr><tr><td>ReferFormer†</td><td>[23]</td><td>52.64</td><td>53.06</td><td>52.22</td><td>65.43</td><td>66.14</td><td>64.72</td><td>66.37</td><td>72.18</td><td>60.56</td><td></td></tr><tr><td>MUTR† [24]</td><td></td><td>54.09</td><td>53.80</td><td>54.38</td><td>66.92</td><td>67.71</td><td>66.13</td><td>66.14</td><td>72.37</td><td>59.91</td><td></td></tr><tr><td>RSVIS [20]</td><td></td><td>61.76</td><td>62.06</td><td>61.46</td><td>71.28</td><td>71.85</td><td>70.71</td><td>70.64</td><td>76.26</td><td>65.02</td><td></td></tr><tr><td>OnlineRefer [22]</td><td></td><td>60.35</td><td>60.92</td><td>59.78</td><td>70.17</td><td>70.93</td><td>69.41</td><td>68.83</td><td>74.51</td><td>63.15</td><td></td></tr><tr><td>RefSAM [10]</td><td></td><td>63.56</td><td>63.84</td><td>63.28</td><td>72.86</td><td>73.40</td><td>72.32</td><td>71.90</td><td>77.48</td><td>66.32</td><td></td></tr><tr><td>ReSurgSAM2</td><td>[11]</td><td>77.73</td><td>77.77</td><td>77.69</td><td>80.62</td><td>80.94</td><td>80.31</td><td>75.09</td><td>80.93</td><td>69.25</td><td>54.2</td></tr><tr><td>Ours</td><td></td><td>81.50</td><td>81.48</td><td>81.52</td><td>83.71</td><td>83.83</td><td>83.60</td><td>76.03</td><td>82.90</td><td>69.33</td><td>53.9</td></tr><tr><td>Δ</td><td></td><td>+3.77</td><td>+3.71 +3.83</td><td></td><td>+3.09</td><td>+2.89</td><td>+3.29</td><td>+0.94 +1.97 +0.08|</td><td></td><td></td><td>-0.3</td></tr></table>

Table 3. Module ablation on Ref-EndoVis18 (tool).
<table><tr><td>GSA</td><td>RW-V2T</td><td>J&amp;F</td><td>J</td><td>F</td></tr><tr><td rowspan="3">√</td><td></td><td>80.62</td><td>80.94</td><td>80.31</td></tr><tr><td></td><td>82.43</td><td>82.68</td><td>82.18</td></tr><tr><td>√</td><td>81.22</td><td>81.40</td><td>81.04</td></tr><tr><td>√</td><td>√</td><td>83.71</td><td>83.83</td><td>83.60</td></tr></table>

## 4.2 Main Results

Table 2 compares our method against prior work on all three evaluation splits. Our method achieves the best performance on every metric across all datasets. Compared to the strong ReSurgSAM2 baseline, our method further improves J&F by +3.77 and +3.09 on the two tool splits, demonstrating clear benefits from reliability-aware grounding. The gain on the tissue split is more modest (+0.94), which we attribute to the spatially difuse nature of tissue regions: their boundaries are less well-defined than rigid instruments, making the reliability gate’s spatial focusing less impactful. Nevertheless, our method still establishes a new state of the art on this challenging split. We do not include SurgRef [21] in the quantitative comparison, as its pretrained checkpoints have not been publicly released at the time of submission.

## 4.3 Ablation Studies

All ablation experiments are conducted on Ref-EndoVis18 (tool) to examine the contribution of each design choice.

Module contribution. Table 3 reports the contribution of each proposed module. Since both GSA and RW-V2T consume the reliability map G produced by the Text-Guided Reliability Gate, the Gate is kept enabled throughout; its internal design is dissected separately in Table 4. Individually, GSA improves J&F by +1.81 and RW-V2T by +0.60, confirming that injecting reliability awareness into either attention direction is beneficial on its own. More importantly, combining the two modules yields a +3.09 gain, which exceeds the sum of their individual contributions by 0.68 points. This indicates that GSA and RW-V2T are complementary to each other. By anchoring both modules to the same reliability map, our design enables coordinated reliability-aware modulation in both attention directions.

Table 4. Gate design ablation on Ref-EndoVis18 (tool).
<table><tr><td>Gate design</td><td>J&amp;F</td></tr><tr><td>No gate (baseline)</td><td>80.62</td></tr><tr><td>Static learned gate</td><td>80.79</td></tr><tr><td>Visual-only gate</td><td>81.02</td></tr><tr><td>Text-guided gate (no residual)</td><td>81.94</td></tr><tr><td>Text-guided residual gate (ours)</td><td>82.43</td></tr></table>

Table 5. V2T attention design ablation on Ref-EndoVis18 (tool).
<table><tr><td>V2T design</td><td>J&amp;F</td></tr><tr><td>Vanilla V2T (baseline)</td><td>80.62</td></tr><tr><td>V ⊙ G + mixing</td><td>80.95</td></tr><tr><td> $V \odot G + { \mathrm { m i x i n g } } + { \mathrm { r e s i d u a l } } { \mathrm { ~ ( o u r s ) } }$ </td><td>81.22</td></tr></table>

Table 6. Efect of gate floor α on Ref-EndoVis18 (tool).  
Table 7. Efect of initialisation strategy on Ref-EndoVis18 (tool).
<table><tr><td>α</td><td>0.0 0.1</td><td>0.3 0.5</td><td>1.0</td></tr><tr><td></td><td>J&amp;F|81.94 82.85 83.71 83.12 81.35</td></tr></table>

<table><tr><td>Initialisation</td><td>|J&amp;F</td></tr><tr><td>Random init</td><td>79.53</td></tr><tr><td>Zero/identity init (ours)</td><td>83.71</td></tr></table>

Gate design. Table 4 dissects the reliability gate design. A static learned gate or visual-only gate yields only marginal gains (+0.17 and +0.40 over the no-gate baseline), confirming that text conditioning is the key contributor. Removing the residual structure from our text-guided gate further drops J &F from 82.43 to 81.94, showing that soft residual modulation outperforms direct gating.

V2T attention design. Table 5 examines the RW-V2T design. Value scaling with channel mixing yields a modest gain (+0.33), and adding the identityinitialised residual projection brings a further improvement (+0.27). The residual connection stabilises training by preserving original visual content.

Gate floor α. Table 6 reports sensitivity to the soft floor parameter α in Eq. 3. Performance follows an inverted-U pattern: α=0 aggressively suppresses lowconfidence regions and removes useful contextual cues (81.94), whereas α=1 degenerates into uniform weighting and disables reliability gating altogether (81.35). α=0.3 achieves the best trade-of between distractor suppression and context preservation (83.71).

Initialisation strategy. Table 7 compares random initialisation against our zero/identity initialisation for the newly introduced adapter parameters. Random initialisation severely degrades performance (−4.18 J&F), confirming that preserving the pre-trained CSTMamba representation at initialisation is critical for stable fine-tuning of the lightweight adapter modules [4].

Table 8. Initial grounding analysis on Ref-EndoVis18 (tool). Metrics are computed on the CIFS-selected anchor frame before SAM2 tracking.
<table><tr><td>Method</td><td>Init.  $\mathcal { I }$  Init.</td><td> $\mathcal { F }$  Init.</td><td> $\mathcal { I } \& \mathcal { F }$ </td><td>AnchorAcc</td><td>Final  $\mathcal { I } \& \mathcal { F }$ </td></tr><tr><td>ReSurgSAM2 [11]</td><td>63.30</td><td>62.49</td><td>62.90</td><td>76.19</td><td>80.62</td></tr><tr><td>ReGround-Surg (Ours)</td><td>72.31</td><td>72.52</td><td>72.42</td><td>85.17</td><td>83.71</td></tr><tr><td>Δ</td><td>+9.01</td><td>+10.03</td><td>+9.52</td><td>+8.98</td><td>+3.09</td></tr></table>

![](images/fd9e4c77303eb96fc65c2a950ba819452af9a425fad517a0c2f01a699442aed4.jpg)  
Fig. 4. Initial anchor IoU vs. final video IoU on Ref-EndoVis17 and Ref-EndoVis18 (tool). Each point corresponds to one video-text pair.

## 4.4 Further Analysis

Initial Grounding Analysis. To verify that our improvements stem from better Stage I grounding rather than from incidental interactions with the tracking stage, we report grounding quality on the CIFS-selected anchor frame before SAM2 tracking. Table 8 reports grounding quality on the CIFS-selected anchor frame before SAM2 tracking. Our method improves Init. $\mathcal { T } \& \mathcal { F }$ by +9.52 and AnchorAcc by +8.98, which translate into $\mathrm { ~ a ~ } + 3 . 0 9$ gain in final video $\mathcal { T } \& \mathcal { F }$ The smaller final gain reflects the smoothing efect of SAM2 tracking, which compensates for minor noise, but cannot recover from anchor errors, reinforcing the importance of reliable Stage I grounding. Fig. 4 plots the initial anchor IoU against the final video $\mathcal { T } \& \mathcal { F }$ on Ref-EndoVis17 and Ref-EndoVis18 (tool). The clear positive trend confirms that a correct anchor is necessary, though not suficient for high final performance, as the remaining variance arises from occlusion and scene complexity during tracking.

Dificult-Case Analysis. Table 9 stratifies the Ref-EndoVis18 test cases by scene complexity. Consistent improvements are observed across all challenging scenarios. Gains are particularly pronounced for same-type instrument confusion (+5.62), followed by partial occlusion (+4.17) and multi-instrument scenes (+4.15). These results suggest that the proposed reliability gate is especially beneficial when visually similar distractors compete with the referred target, improving robustness to ambiguous surgical conditions.

Table 9. Dificult-case analysis on Ref-EndoVis18, stratified by scene type (J&F).
<table><tr><td>Case type</td><td>ReSurgSAM2</td><td>Ours</td><td>Gain</td></tr><tr><td>Multiple instruments</td><td>75.33</td><td>79.48</td><td>+4.15</td></tr><tr><td>Same-type instruments</td><td>71.39</td><td>77.01</td><td>+5.62</td></tr><tr><td>Partial occlusion</td><td>75.43</td><td>79.60</td><td>+4.17</td></tr></table>

![](images/0f313c7abf60b1502926c5c87e1f6da8d4be5440f8bc2d30933f75a30a52ea17.jpg)  
Fig. 5. Qualitative comparison on Ref-EndoVis18. Rows (top to bottom): ground truth; ReSurgSAM2 prediction; ours. Columns: multi-instrument confusion; partial occlusion; complete grounding failure.

Qualitative Analysis. Fig. 5 presents qualitative comparisons on Ref-EndoVis18 across three representative failure modes: spatial confusion under multi-instrument scenes, first-frame failure under partial occlusion, and a complete grounding failure case. In the multi-instrument case, the baseline drifts onto an adjacent instrument, while the reliability gate produces higher responses around the referred region and lower responses around competing distractors. In the occlusion case, RW-V2T appears to reduce the influence of occluded or unreliable regions during prompt-token aggregation, enabling better first-frame localisation.

## 5 Conclusion

We presented ReGround-Surg, a lightweight reliability-guided anchor grounding framework to enhance the referring surgical video segmentation. Motivated by the observation that SAM2-based two-stage pipelines are highly sensitive to initial grounding errors, our method improves Stage I detection through a Gated Side Adapter and Reliability-Weighted V2T attention. Both modules share a single text-conditioned reliability map and are inert at initialisation, and require no changes to the SAM2 encoder, decoder, or tracking components. Experiments on Ref-EndoVis17 and Ref-EndoVis18 show consistent J&F improvements of +3.77, +3.09, and +0.94 over ReSurgSAM2 with only 0.5 M additional parameters. Initial grounding analysis, gate visualisation, and dificult-case breakdown further support the importance of addressing grounding failure at its source, particularly under multi-instrument confusion and partial occlusion.

Nevertheless, the current method has limitations: the reliability gate may produce weaker improvements on spatially difuse targets such as tissue regions, and the gate floor α requires manual tuning across diferent surgical domain. Moreover, since Stage II tracking remains unchanged, residual errors arising from memory drift or long-term occlusion after the anchor frame are not addressed by our current design. Future work will explore learnable gate parameters and extending reliability-aware modulation into Stage II memory selection to further reduce propagation errors.

## References

1. Allan, M., Kondo, S., Bodenstedt, S., Leger, S., Kadkhodamohammadi, R., Luengo, I., et al.: 2018 robotic scene segmentation challenge. arXiv preprint arXiv:2001.11190 (2020)

2. Allan, M., Shvets, A., Kurmann, T., Zhang, Z., Duggal, R., Su, Y.H., Rieke, N., Laina, I., et al.: 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 (2019)

3. Ayobi, N., P´erez-Rond´on, A., Rodr´ıguez, S., Arbelaez, P.: MATIS: Maskedattention transformers for surgical instrument segmentation. In: IEEE International Symposium on Biomedical Imaging (ISBI) (2023)

4. Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P.: AdaptFormer: Adapting vision transformers for scalable visual recognition. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 35 (2022)

5. Cheng, H.K., Schwing, A.G.: XMem: Long-term video object segmentation with an Atkinson-Shifrin memory model. In: European Conference on Computer Vision (ECCV). pp. 640–658 (2022)

6. Cuttano, C., Trivigno, G., Rosi, G., Masone, C., Averta, G.: SAMWISE: Infusing wisdom in SAM2 for text-driven video segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3395–3405 (2025)

7. Gonz´alez, C., Bravo-S´anchez, L., Arbelaez, P.: ISINet: An instance-based approach for surgical instrument segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI) (2020)

8. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023)

9. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Doll´ar, P., Girshick, R.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 3992–4003 (2023)

10. Li, Y., Zhang, J., Teng, X., Lan, L., Liu, X.: Refsam: Eficiently adapting segmenting anything model for referring video object segmentation. arXiv preprint arXiv:2307.00997 (2024)

11. Liu, H., Gao, M., Luo, X., Wang, Z., Qin, G., Wu, J., Jin, Y.: ReSurgSAM2: Referring segment anything in surgical video via credible long-term tracking. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025 (2025)

12. Liu, H., Zhang, E., Wu, J., Hong, M., Jin, Y.: Surgical SAM 2: Real-time segment anything in surgical video by eficient frame pruning. In: NeurIPS Workshop on Advancements in Medical Foundation Models (2024)

13. Maier-Hein, L., et al.: Surgical data science for next-generation interventions. Nature Biomedical Engineering 1(9), 691–696 (2017)

14. Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B., Glocker, B., Rueckert, D.: Attention U-Net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999 (2018)

15. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-Hornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: CVPR (2016)

16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (2021)

17. Ravi, N., et al.: SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

18. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI). pp. 234–241 (2015)

19. Ryali, C., Hu, Y.T., Bolya, D., Wei, C., Fan, H., Huang, P.Y., Aggarwal, V., Chowdhury, A., Poursaeed, O., Hofman, J., Malik, J., Li, Y., Feichtenhofer, C.: Hiera: A hierarchical vision transformer without the bells-and-whistles. In: International Conference on Machine Learning (ICML). pp. 29441–29454 (2023)

20. Wang, H., Yang, G., Zhang, S., Qin, J., Guo, Y., Xu, B., Jin, Y., Zhu, L.: Videoinstrument synergistic network for referring video instrument segmentation in robotic surgery. IEEE Transactions on Medical Imaging 43(12), 4457–4469 (2024)

21. Wei, M., Yuan, K., Li, S., Zhou, Y., Bai, L., Navab, N., Ren, H., Lee, H.J., Vercauteren, T., Padoy, N.: Where it moves, it matters: Referring surgical instrument segmentation via motion. In: Proceedings of the AAAI Conference on Artificia Intelligence (2026)

22. Wu, D., Wang, T., Zhang, Y., Zhang, X., Shen, J.: Onlinerefer: A simple online baseline for referring video object segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2761–2770 (2023)

23. Wu, Y., et al.: Language as queries for referring video object segmentation. In: CVPR (2022)

24. Yan, S., Zhang, R., Guo, Z., Chen, W., Zhang, W., Li, H., Qiao, Y., Dong, H., He, Z., Gao, P.: Referred by multi-modality: A unified temporal transformer for video object segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence (2024)

25. Yang, Z., Wang, J., Tang, Y., Chen, K., Zhao, H., Torr, P.H.: LAVT: Languageaware vision transformer for referring image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18155–18165 (2022)

26. Yin, M., Wang, F., Ye, X., Meng, Y., Fu, Z.: Memory-augmented SAM2 for training-free surgical video segmentation. arXiv preprint arXiv:2507.09577 (2025)

27. Yue, W., Zhang, J., Hu, K., Xia, Y., Luo, J., Wang, Z.: SurgicalSAM: Eficient class promptable surgical instrument segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 6890–6898 (2024)

28. Zhao, Z., Jin, Y., Heng, P.A.: TraSeTr: Track-to-segment transformer with contrastive query for instance-level instrument segmentation in robotic surgery. In: IEEE International Conference on Robotics and Automation (ICRA) (2022)