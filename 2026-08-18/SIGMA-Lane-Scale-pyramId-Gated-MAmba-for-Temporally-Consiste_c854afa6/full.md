# SIGMA-Lane: Scale-pyramId Gated MAmba for Temporally Consistent Video Lane Detection

Tiancheng Zhang , Yan Gao, Xiangjie Kong , Guojiang Shen, Jiaxin Du, and Mengmeng Wang <sup>†</sup>

Zhejiang Key Laboratory of Visual Information Intelligent Processing, College of Computer Science and Technology, Zhejiang University of Technology

Abstract. Video lane detection requires predictions that remain stable across frames, yet severe vehicle occlusions can break temporal cues. In streaming recurrent models, corrupted observations may enter the hidden state and produce errors that persist into later frames. Existing occlusion-aware refinements usually provide obstacle masks as auxiliary inputs, so the state-update path is only indirectly protected. We propose SIGMA-Lane, which treats this failure mode as state contamination in State Space Model (SSM)-based temporal modeling. SIGMA-Lane places occlusion-aware gates on the SSM write and residual-fusion paths, controlling how current observations enter temporal memory and are fused back after temporal propagation. After coordinate-consistent afine alignment, the model combines two complementary paths: SSM-consistent dual-gating for temporal filtering and Structural Spatial Retrieval (SSR) for recovering missing lane structure from aligned historical priors. Experiments on VIL-100 and OpenLane-V show improved temporal stability under heavy occlusion, with competitive F1 and mIoU scores.

Keywords: Video Lane Detection · State Space Models · Temporal Consistency · Occlusion Handling

## 1 Introduction

Lane detection provides structured road-boundary information for downstream planning and control in autonomous driving. Existing methods achieve strong per-frame accuracy with segmentation, parametric, or anchor-based representations [19, 23, 37]. However, in video, the detector must also keep the predictions consistent over time. Two temporal designs are common. Window-batched methods [27, 28, 35, 38] jointly process a temporal window (Fig. 1a). Recurrent methods [2, 12, 34, 41] propagate a state frame by frame for streaming inference (Fig. 1b). Both designs are vulnerable when large vehicles occlude lanes for several frames. If corrupted observations enter the temporal pipeline, they can afect later predictions unless the update itself suppresses them.

Existing recursive methods, such as OMR [11], feed obstacle masks into a ConvLSTM-based refinement module. This gives the model obstacle context, but it does not directly constrain how occluded features are written into the recurrent state. We examine this issue through the SSM update $h _ { t } = \bar { A } _ { t } h _ { t - 1 } + \bar { B } _ { t } x _ { t }$ , where the current observation enters the state through the write term. Mask-based noise filtering is common in vision, but in an SSM, this write term can still inject occlusion-corrupted features into the hidden state and leave residual errors across frames. Once stored, the corrupted feature becomes part of temporal memory and is reused by later predictions. We call this failure mode state contamination. This perspective motivates placing the obstacle prior on the SSM write path and the residual-fusion path, instead of using it only as an auxiliary input. It also exposes a second issue, coordinate inconsistency: if features and masks are warped independently, the gate may no longer match the occluded region it is meant to modulate.

We propose SIGMA-Lane (Fig. 1c), an SSM-based framework that gates the state update under occlusion. SSM-Consistent Dual-Gating reduces the efective write amplitude of $\bar { B } _ { t } { x } _ { t }$ at the input stage and controls residual fusion after temporal propagation. Lane-Guided Afine Warp applies the same afine transform to features, obstacle masks, and lane masks, keeping the gating signal in the same coordinate frame as the feature being modulated. SSR then recovers lane structure from aligned historical features with mask-guided cross-attention. A geometry-aware start token initializes the temporal state at the beginning of a stream. On VIL-100 [35], SIGMA-Lane achieves the best mIoU, F1@0.5, accuracy, and temporal stability scores, reducing $R _ { F } / R _ { M } @ 0 . 5$ to 0.023/0.030. On OpenLane-V [12], it obtains the best F1@0.5/F1@0.8 and the lowest persistent missing rate at the stricter threshold $\left( R _ { M } @ 0 . 8 = 0 . 3 8 4 \right)$ , while remaining competitive in mIoU (Tab. 1, Tab. 2).

## Our contributions are:

– We formulate state contamination as a failure mode in SSM-based video lane detection, and introduce a dual-gating mechanism that controls both the SSM write path and residual-fusion path under occlusion.

– We design a dual-path aggregation module with coordinate-consistent afine alignment, Mamba-based temporal propagation, and Structural Spatial Retrieval. The design separates temporal filtering from spatial recovery, addressing temporal-state contamination and loss of lane topology with diferent modules.

– On VIL-100 and OpenLane-V, SIGMA-Lane obtains competitive accuracy with lower flickering and missing rates. Ablations and controlled contamination analysis show that dual-gating accounts for a large share of the improvement.

## 2 Related Work

Lane Detection Representations. Modern lane detectors use several representation strategies for complex road scenes. Segmentation-based methods [9,10, 19–21,36] treat lane detection as pixel-wise classification and improve spatial coherence with message passing [19] or multi-scale aggregation [21,36]. Parametric methods [5, 16, 18, 24, 26, 31, 32, 39] model lanes as polynomials or Bézier curves; CondLaneNet [15], for example, uses conditional convolution for instance-level prediction. Anchor-based detectors [3,23,29,37] use predefined line priors for localization. These single-frame methods perform well on static benchmarks, but they process each image independently and cannot use temporal context to infer lanes hidden by occlusion.

![](images/30242f404f8420968cfb9e41ef3c39989e853079efe1579d142efc920c89b2c0.jpg)  
Fig. 1: Comparison of Video Lane Detection Paradigms. The window-batched, joint-aggregation paradigm (a) extracts features $F _ { t - T + 1 : t }$ from a fixed temporal window and aggregates them jointly, incurring high latency without reusing past computation. The recurrent, state-propagation paradigm (b) propagates the previous state $( F _ { t - 1 } , L _ { t - 1 } )$ frame by frame, enabling streaming inference, but ofers no mechanism to prevent corrupted observations from contaminating the hidden state under occlusion. Our memory-maintained, contamination-aware paradigm (c) maintains a video memory bank of historical features $F ,$ obstacle masks M, and lane masks L. After lane-guided afine alignment, two paths process the aligned data: SSM-Consistent Dual-Gating for occlusion-aware temporal filtering and SSR for lane-guided spatial recovery.

Temporal Context in Video Lane Detection. Extending detection to videos reduces the instability of single-frame methods. Early recurrent designs [34, 41] use ConvLSTM or ConvGRU to model frame-to-frame dependencies. Recent transformers [25], including LaneTCA [38] and MMA-Net [35], use attention for global temporal context. Existing temporal aggregators still face a trade-of: recursive methods such as RVLD [12] and OMR [11] are eficient but can propagate corrupted observations, while attention-based models are more expensive. Current eficient solutions often use auxiliary motion correction [12] or separate memory refinement [11], but they do not control how occluded evidence enters the temporal state. SIGMA-Lane targets this gap through a contaminationaware update rule.

SSMs for Sequence Modeling. SSMs [6, 7] model long sequences with linear complexity. They have been used in generic vision tasks [14, 17, 40] and autonomous driving tasks, including 3D perception [33], motion planning [22], and static lane detection [1]. Mamba adds selective scanning for content-aware sequence modeling. In occluded lane videos, however, contaminated observations may still enter the hidden state and accumulate over time. Our work complements Mamba’s internal selectivity with an obstacle prior at the write and residual-fusion paths.

## 3 Method

## 3.1 Preliminary

We build on OMR’s recursive video lane detection framework [11]. We retain its encoder and decoder but replace its aggregation module. Given a current frame $I _ { t } , \mathbf { \boldsymbol { \mathsf { \varepsilon } } } ^ { }$ a ResNet18 [8] encoder extracts a multi-scale feature map $F _ { t } \in \mathbb { R } ^ { B \times C \times H \times W }$ The aggregation module refines $F _ { t }$ into enhanced features $\tilde { F } _ { t }$ using temporalstate propagation and spatial retrieval. Following the eigenlane paradigm [13], the decoder predicts a probability map $P _ { t }$ and a coeficient map $C _ { t }$ . Lane instances are then extracted by NMS to obtain a binary lane mask $L _ { t }$

For streaming inference, a video memory bank stores the past $T$ frames’ aggregated features and obstacle masks. With spatial positions flattened, these tensors are $\mathbf { F } \in \mathbb { R } ^ { ( B H W ) \times T \times C }$ and $\mathbf { M } \in \mathbb { R } ^ { ( B H W ) \times T \times 1 }$ . The bank also stores the previous lane mask $L _ { t - 1 }$ for alignment and guidance. At the end of each frame, $\tilde { F } _ { t } , M _ { t } .$ , and $L _ { t }$ are written back to memory. A geometry-aware start token initializes this recursive process at a cold start (Sect. 3.5). OMR uses ConvLSTM for temporal refinement; we use Mamba2 [4] as the temporal update core and modify the aggregation module to address state contamination and coordinate inconsistency. SSM-Consistent Dual-Gating is the central temporal filter. Lane-guided alignment supplies coordinate-consistent masks, and SSR retrieves missing lane structure after gated temporal filtering. Fig. 2 shows the overall architecture.

![](images/1e815f13cf8206a13d43643986de1d4c8b81cca83d2455467a01149d40f2b9fb.jpg)  
Fig. 2: Overview of the SIGMA-Lane Aggregation Module. Video Memory stores historical features $F _ { t - T + 1 : t }$ , obstacle masks $M _ { t - T + 1 : t }$ , and the previous lane mask $L _ { t - 1 }$ . The Afine Warp module aligns historical data to current-frame coordinates. Warped features then pass through two paths: (1) SSM-Consistent Dual-Gating with a scale-pyramid design for multi-scale temporal propagation and occlusion-aware gating, (2) SSR where current features $F _ { t }$ retrieve spatial context from warped $F _ { t - 1 }$ enhanced by $L _ { t - 1 }$ embeddings. The outputs are aggregated with current-frame features $F _ { t }$ via Add&Norm and FFN blocks, producing enhanced features $\tilde { F } _ { t }$

![](images/5c072d38872bf06cd1584988566b8f6fd2b826ede770f11837535b8c40bc4d96.jpg)  
Fig. 3: SSM-Consistent Dual-Gating with a scale-pyramid design. Input features X pass through LN and are split into high- and low-resolution branches. Each branch applies Input Gate, concatenates the Geometry-Aware Start Token $s _ { 0 } ,$ , performs Mamba2 [4] temporal modeling, and applies Output Gate, with both gates modulated by the obstacle mask M. The low-resolution branch is upsampled and merged with the high-resolution branch through Add&Norm to produce $\tilde { X } .$ . Finally, the current-frame features $\tilde { x } _ { t }$ are extracted from $\tilde { X }$ and sent to subsequent modules.

## 3.2 SSM-Consistent Dual-Gating

Observations from occluded regions are noisy. Directly writing them into the temporal state causes state contamination that accumulates through recursive propagation. Starting from the linear update rule of SSMs,

$$
h _ { t } = \bar { A } _ { t } h _ { t - 1 } + \bar { B } _ { t } x _ { t } ,\tag{1}
$$

the central design question is where the obstacle prior enters the recursion. We gate two pathways where occlusion noise can enter the temporal output: the SSM write path, which controls what enters the temporal state, and the residual-fusion path, which controls how much current-frame evidence is mixed back after temporal propagation. The input gate acts before selective scanning and modulates the SSM write term. The output gate acts after temporal propagation and controls the residual path that mixes current features back into the temporal output. Mamba2’s selective scan [4] uses data-dependent parameters; conditioned on a given input sequence, each step can still be viewed as following the write-andpropagate structure in Eq. 1. This conditioned SSM view focuses the analysis on the write path afected by the external mask. Fig. 3 illustrates the module architecture.

Input gating. We employ Mask-Aware Input Modulation:

$$
x _ { t } ^ { \prime } = x _ { t } \odot ( 1 - m _ { t } ) ,\tag{2}
$$

where $m _ { t } \in [ 0 , 1 ]$ denotes the token-wise occlusion probability, broadcast over channels, and $\odot$ represents element-wise multiplication. The gate suppresses the input write term in Eq. 1:

$$
\bar { B } _ { t } x _ { t } \ \to \ \left( 1 - m _ { t } \right) \bar { B } _ { t } x _ { t } .\tag{3}
$$

Highly occluded tokens therefore contribute little to the state write, while visible tokens are written normally.

This placement follows the contamination recursion. Let $\boldsymbol { x } _ { t } ^ { \star }$ denote the clean observation, $x _ { t } = x _ { t } ^ { \star } + \eta _ { t }$ the occlusion-perturbed observation, and $\varepsilon _ { t }$ the noiseinduced state error. To isolate the explicit write path, we use a conditioned, linearized view in which the discretized parameters ${ \bar { A } } _ { t }$ and ${ \bar { B } } _ { t }$ are fixed around a trajectory. Without gating, the error evolves as

$$
\varepsilon _ { t } = \bar { A } _ { t } \varepsilon _ { t - 1 } + \bar { B } _ { t } \eta _ { t } .\tag{4}
$$

With input gating, let $D _ { t }$ denote the diagonal or channel-broadcast operator induced by $( 1 - m _ { t } )$ . Comparing the perturbed gated trajectory with its clean gated counterpart under the same local dynamics gives

$$
\varepsilon _ { t } ^ { \mathrm { g a t e } } = \bar { A } _ { t } \varepsilon _ { t - 1 } ^ { \mathrm { g a t e } } + D _ { t } \bar { B } _ { t } \eta _ { t } .\tag{5}
$$

When occlusion is severe $( m _ { t }  1 )$ , the entries of $D _ { t }$ approach zero and the write-path noise term is suppressed before it enters the state.

For a contiguous occlusion segment $[ s , s { + } L { - } 1 ]$ and a later frame $t \geq s { + } L { - } 1$ define the transition product

$$
\varPhi ( t , k ) = \bar { A } _ { t } \bar { A } _ { t - 1 } \cdot \cdot \cdot \bar { A } _ { k + 1 } , \qquad \varPhi ( t , t ) = I .\tag{6}
$$

Unrolling the gated error recursion over the segment yields

$$
\varepsilon _ { t } ^ { \mathrm { g a t e } } = \varPhi ( t , s - 1 ) \varepsilon _ { s - 1 } ^ { \mathrm { g a t e } } + \sum _ { k = s } ^ { s + L - 1 } \varPhi ( t , k ) D _ { k } \bar { B } _ { k } \eta _ { k } .\tag{7}
$$

Assume the conditioned dynamics are uniformly contractive, and the write matrices are bounded, with $\| \bar { A } _ { t } \| \le \alpha < 1$ and $\| \bar { B } _ { t } \| \le \beta$ . If $\| \eta _ { k } \| \le \sigma$ on the segment and $\| D _ { k } \| \leq \gamma _ { k } \leq 1$ , Eq. 7 gives

$$
\| \varepsilon _ { t } ^ { \mathrm { g a t e } } \| \leq \alpha ^ { t - s + 1 } \| \varepsilon _ { s - 1 } ^ { \mathrm { g a t e } } \| + \beta \sigma \sum _ { k = s } ^ { s + L - 1 } \alpha ^ { t - k } \gamma _ { k } .\tag{8}
$$

With $\gamma = \mathrm { m a x } _ { k \in [ s , s + L - 1 ] } \gamma _ { k }$ , this becomes

$$
\| \varepsilon _ { t } ^ { \mathrm { g a t e } } \| \leq \alpha ^ { t - s + 1 } \| \varepsilon _ { s - 1 } ^ { \mathrm { g a t e } } \| + \beta \sigma \gamma \alpha ^ { t - s - L + 1 } \frac { 1 - \alpha ^ { L } } { 1 - \alpha } .\tag{9}
$$

The ungated case is recovered by setting $D _ { k } = I$ and $\gamma = 1$ . Under the conditioned, contractive write-path view above, a longer occlusion segment contributes a larger truncated geometric accumulation term, while input gating scales this term by the mask-dependent factor γ. The complete module, including the output gate, is evaluated empirically through ablation and controlled-contamination studies.

Output gating. Input gating protects the state write, but the residual connection can still reintroduce noisy input features after temporal propagation. We control the residual fusion ratio at the output stage using the occlusion probability M (a token-aligned obstacle mask, broadcast along channels). We summarize occlusion severity as a scalar $\bar { m } = \mathrm { G A P } ( M )$ and compute the residual retention ratio

$$
\bar { m } = \mathrm { G A P } ( M ) , \quad r = \sigma ( \mathrm { M L P } ( \bar { m } ) ) \in ( 0 , 1 ) ,\tag{10}
$$

where r is then broadcast to match the shape of $X _ { \mathrm { i n } }$ . Let $X _ { \mathrm { i n } }$ denote the temporal block input and $X _ { \mathrm { o u t } }$ the Mamba output:

$$
X _ { \mathrm { b a s e } } = X _ { \mathrm { i n } } + X _ { \mathrm { o u t } } , \qquad X _ { \mathrm { h i s t } } = r \odot X _ { \mathrm { i n } } + X _ { \mathrm { o u t } } ,\tag{11}
$$

$$
X _ { \mathrm { t i m e } } = ( 1 - M ) \odot X _ { \mathrm { b a s e } } + M \odot X _ { \mathrm { h i s t } } .\tag{12}
$$

The gate is initialized to a small value, close to 0.2, at the beginning of training for stability. Under occlusion, it down-weights the contaminated input $X _ { \mathrm { i n } }$ while keeping the propagated output $X _ { \mathrm { o u t } }$ . When $M = 1$ (full occlusion), the output reduces to $r \odot X _ { \mathrm { i n } } + X _ { \mathrm { o u t } } ;$ when $M = 0$ (no occlusion), the standard residual $X _ { \mathrm { i n } } + X _ { \mathrm { o u t } }$ is used.

Relation to Mamba internal selectivity. Mamba uses input-dependent parameters $( \varDelta , B , C )$ learned from data for content-aware selectivity. Our dualgating adds an obstacle-mask prior at spatial positions likely to contain corrupted observations. The two mechanisms operate at diferent levels: Mamba’s internal selectivity acts on content representations, while the external gates use the obstacle mask to modulate state writing and residual fusion at occluded positions.

Multi-scale implementation. To aggregate information across the past T frames, we flatten the feature sequence $\{ x _ { t - T + 1 } , \ldots , x _ { t } \}$ and occlusion sequence $\{ m _ { t - T + 1 } , \dots , m _ { t } \}$ along spatial positions to obtain $\mathbf { \bar { \chi } } _ { X } \in \mathbb { R } ^ { ( B H W ) \times T \times C }$ and $M \doteq \mathbb { R } ^ { ( B H W ) \times T \times 1 } . \mathrm { ~ \dot { A } ~ }$ parallel low-resolution branch captures broader context: features are downsampled by a factor $s ,$ processed through Mamba with dual-gating, and upsampled back. The outputs are fused via a learnable residual:

$$
\tilde { x } _ { t } = x _ { \mathrm { o u t } } + \alpha _ { \mathrm { l o w } } \cdot \mathrm { U p s a m p l e } ( \mathrm { M a m b a } ( \mathrm { P o o l } _ { s } ( X ) ) ) ,\tag{13}
$$

where $\alpha _ { \mathrm { l o w } }$ is initialized to a small value for stable training. The scale-pyramid branch combines fine local details with coarse context.

## 3.3 Lane-Guided Afine Warp

SSM-Consistent Dual-Gating applies temporal modeling after flattening spatial positions into token sequences. Each sequence therefore assumes that the same spatial index across frames refers to a consistent road location. Ego-motion and camera motion violate this assumption: a lane point or obstacle boundary may move to a diferent feature coordinate between adjacent frames. If such misaligned features are scanned along the T dimension, the SSM can mix evidence from diferent physical locations. The obstacle gate can also shift away from the feature region it is meant to modulate. We therefore align the historical feature, lane prior, and obstacle mask before temporal propagation, using the synchronous strategy shown in Fig. 4(a).

We use a local short-range afine transformation for lightweight 2D alignment between adjacent frames. This design matches the streaming state update, where information is propagated recursively and only the most recent transition must be synchronized before it enters the next temporal scan. In forward-facing driving videos, this low-dimensional parameterization is a compact approximation for synchronizing the feature, lane-prior, and obstacle-mask coordinates used by the gates. It is also less sensitive than dense optical flow [12] to textureless or occluded regions. The resulting alignment supplies the token-aligned obstacle prior used by dual-gating. The lane mask $\ell _ { t - 1 }$ provides an additional geometric prior: masked global average pooling (GAP) focuses transformation estimation on structurally stable lane areas while ignoring cluttered backgrounds.

Let $x _ { t - 1 }$ and $x _ { t }$ denote adjacent frame features. The afine parameters are predicted as:

$$
p = \mathrm { t a n h } ( \mathrm { M L P } ( [ \mathrm { G A P } ( x _ { t - 1 } ; \ell _ { t - 1 } ) , ~ \mathrm { G A P } ( x _ { t } ) ] ) ) ,\tag{14}
$$

where $[ \cdot ]$ denotes concatenation. We zero-initialize the final MLP layer so that the transformation starts from identity and avoids unstable early estimates. Reshaping p yields a residual afine matrix $\varDelta \theta \in \mathbb { R } ^ { B \times 2 \times 3 }$ . The final afine matrix is obtained by adding it to the identity afine $I _ { 2 \times 3 } { \mathrm { : } }$

$$
\Delta \theta = \mathrm { r e s h a p e } ( p ) \in \mathbb { R } ^ { B \times 2 \times 3 } , \qquad \theta = I _ { 2 \times 3 } + \varDelta \theta .\tag{15}
$$

The transformation synchronously warps features, lane masks, and obstacle masks via bilinear sampling:

$$
\tilde { x } _ { t - 1 } = \mathrm { w a r p } ( x _ { t - 1 } , \Theta ) , \quad \tilde { \ell } _ { t - 1 } = \mathrm { w a r p } ( \ell _ { t - 1 } , \Theta ) , \quad \tilde { m } _ { t - 1 } = \mathrm { w a r p } ( m _ { t - 1 } , \Theta ) .\tag{16}
$$

This alignment keeps the occlusion signal $\tilde { m } _ { t - 1 }$ and lane prior $\tilde { \ell } _ { t - 1 }$ matched to the warped features $\tilde { x } _ { t - 1 }$ and supports better aligned gating and structural guidance.

## 3.4 Structural Spatial Retrieval

Dual-gating suppresses contaminated writes and residuals (Sect. 3.2), but its temporal propagation can be insuficient when the historical state is weak and the current frame is heavily occluded. In this case, occluded regions still lack local lane evidence after gated temporal filtering. With aligned features $\tilde { x } _ { t - 1 }$ and lane prior $\tilde { \ell } _ { t - 1 }$ from Lane-Guided Afine Warp, SSR adds a spatial retrieval step to strengthen these missing structural cues. It uses cross-frame correspondence between the current frame and aligned history to recover structural evidence, as illustrated in Fig. 4(b).

Current-frame features serve as the query, while warped previous-frame features form the key and value. To guide attention toward lane-relevant regions, we inject a learnable lane-mask embedding into the key/value features:

$$
Q = \mathrm { L N } ( x _ { t } ) , \qquad K = V = \mathrm { L N } ( \tilde { x } _ { t - 1 } ) + \mathrm { L N } ( e ( \tilde { \ell } _ { t - 1 } ) ) ,\tag{17}
$$

![](images/98a1dac1f9a76c3947f69e51cce7bb43cb1d41d895034f9f59779b18f878dafe.jpg)  
Fig. 4: Lane-Guided Afine Warp and Structural Spatial Retrieval. (a) In Lane-Guided Afine Warp, features $x _ { t - 1 }$ and lane mask $\ell _ { t - 1 }$ are fed into Lane-Masked $\mathrm { G A P }$ to compute lane-region-weighted statistics. Combined with current-frame features $x _ { t }$ , an MLP predicts afine parameters $p ,$ where tanh constrains the transformation magnitude. The Afine Warp operator synchronously transforms $x _ { t - 1 } , \ell _ { t - 1 }$ , and $m _ { t - 1 }$ using the same grid, yielding aligned outputs $\tilde { x } _ { t - 1 } , \tilde { \ell } _ { t - 1 } ,$ and $\tilde { m } _ { t - 1 } . ~ ( \mathrm { b } )$ In Structural Spatial Retrieval, current-frame features $x _ { t }$ pass through LN to form the query Q. Warped features $\tilde { x } _ { t - 1 }$ and lane-mask embedding $e ( \tilde { \ell } _ { t - 1 } )$ are normalized via LN and summed to form key K and value V. Multi-head cross-attention produces the enhanced feature output.

where LN denotes Layer Normalization and $e ( \cdot )$ is a lightweight convolutional network mapping the single-channel mask $\tilde { \ell } _ { t - 1 }$ to a C-dimensional feature space. Cross-attention follows the standard multi-head formulation [25]:

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right) V .\tag{18}
$$

The $Q K ^ { \top }$ similarity matrix computes soft spatial correspondence, so each query position can retrieve structural cues from history without explicit ofset prediction. Injecting the lane-mask embedding into the key/value features biases attention toward lane regions. Positions with stronger lane priors can receive higher matching scores, improving cross-frame completion under occlusion. SSR works with Dual-Gating (Sect. 3.2), which handles recursive temporal state propagation.

## 3.5 Geometry-Aware State Initialization

The previous modules control how observations enter an already running temporal state. At the beginning of a stream, however, this state is still uninitialized. Zero initialization $( h _ { 0 } = 0 )$ makes the first few state updates depend strongly on early frame observations, so an occluded or ambiguous first frame can dominate the initial memory. We use Geometry-Aware State Initialization to give the SSM a weak spatial prior before any image observation is written into the state. We prepend a learnable start token s to each spatial token sequence before Mamba processing. This token is read before the video frames and is not attenuated by the obstacle mask. To give it spatial semantics, we parameterize it as the sum of a context-agnostic shared token and a position-specific projection:

![](images/a3c938393d91f4b3466383e245262478f7bf998e8672fd75128bcf3ba85bc883.jpg)  
Fig. 5: Visualization of the spatial prior learned by Geometry-Aware State Initialization. Each strip shows the input image, position-prior heatmap, and ground-truth lanes. The learned prior is concentrated around lane-like structures and provides a cold-start bias before reliable temporal history is available.

$$
s = \mathrm { t o k e n } _ { \mathrm { s h a r e d } } + \mathrm { P r o j } ( \mathrm { P E } _ { 2 D } ) ,\tag{19}
$$

where $s \in \mathbb { R } ^ { ( B H W ) \times 1 \times C }$ and $\mathrm { P E _ { 2 D } }$ encodes normalized 2D coordinates. By processing $X ^ { + } = \operatorname { c o n c a t } ( s , X )$ , the SSM updates its hidden state $h _ { - 1 }  h _ { 0 }$ using s before seeing any video frames. This initialization complements dual-gating: the gates suppress contaminated writes, while the start token provides a geometryaware state from which reliable temporal propagation can begin. Fig. 5 visualizes this learned prior, which tends to concentrate around lane-like regions.

## 4 Experiments

## 4.1 Implementation Details

The base detector described in Sect. 3.1 is initialized from the pretrained OMR checkpoint and kept frozen. Only the temporal aggregation modules are trained. Following OMR, obstacle masks are produced by a frozen SegFormer-B5 [30]; the segmenter is not updated during training. The temporal module uses Mamba2 [4] with an 8-frame memory window. It is trained with the same lane classification and coeficient regression losses as the base detector. Detailed hyperparameters, dataset descriptions, and metric definitions are provided in the supplementary material.

## 4.2 Datasets and Evaluation Metrics

We evaluate SIGMA-Lane on VIL-100 [35] and OpenLane-V [12], two video lane benchmarks with persistent vehicle occlusion and diverse driving conditions. Following the CULane protocol [19], we report per-frame quality using mIoU, accuracy, and F1 at IoU thresholds 0.5 and 0.8. We also report temporal flickering and missing rates $R _ { F } / R _ { M }$ from [12], where lower values indicate fewer inconsistent detections across adjacent frames. The supplementary material gives the dataset details and metric definitions.

Table 1: Comparison on VIL-100. Best results are in bold, second best are underlined.
<table><tr><td>Method</td><td>Approach</td><td>mIoU(↑)</td><td>F1@0.5(↑)</td><td>F1@0.8(↑)</td><td>Accuracy(↑)</td><td> $R _ { F } @ 0 . 5 ( \downarrow )$ </td><td> $R _ { M }$  @0.5(↓)</td></tr><tr><td>LaneNet [18] LSTR [16] LaneATT [23]</td><td>Image-based</td><td>0.633 0.573 0.664 0.745</td><td>0.721 0.703 0.823</td><td>0.222 0.131</td><td>0.858 0.884 0.912</td><td></td><td></td></tr><tr><td>DiLane [3] MMA-Net [35] MLM-Net [28] RVLD [12]</td><td>Video-based</td><td>0.705 0.753 0.787</td><td>0.837 0.839 0.904 0.924</td><td>0.505 0.458 0.624 0.582</td><td>0.884 0.910</td><td>0.042 0.038</td><td>0.127 0.050</td></tr><tr><td>PHNet [2] OMR [11]</td><td></td><td>0.783 0.774</td><td>0.915 0.936</td><td>0.615 0.504</td><td>0.908 0.948</td><td>0.026</td><td>0.038</td></tr><tr><td>LaneTCA [38] SIGMA-Lane</td><td>Video-based</td><td>0.796 0.801</td><td>0.933 0.940</td><td>0.621 0.595</td><td>0.951 0.956</td><td>0.039 0.023</td><td>0.055 0.030</td></tr></table>

## 4.3 Comparative Results

VIL-100. Tab. 1 compares SIGMA-Lane with image-based [3, 16, 18, 23] and video-based [2,11,12,28,35,38] detectors. SIGMA-Lane obtains the best F1@0.5 (0.940), accuracy (0.956), and mIoU (0.801). Although SIGMA-Lane does not obtain the best F1@0.8, it has the lowest flickering and missing rates on VIL-100. Compared with LaneTCA, it reduces $R _ { F } / R _ { M }$ by 41.0%/45.5%. It also improves F1@0.8 by 0.091 over OMR, suggesting greater robustness when heavy occlusions suppress per-frame evidence. Fig. 6 shows examples where SIGMA-Lane preserves lane continuity when large vehicles obscure multiple lanes.

OpenLane-V. Tab. 2 reports results on the larger OpenLane-V benchmark. LaneTCA [38] obtains the highest mIoU (0.774), while SIGMA-Lane obtains the highest F1@0.5/F1@0.8 (0.838/0.591) and the lowest persistent missing rate at the stricter threshold $\left( R _ { M } @ 0 . 8 \ = \ 0 . 3 8 4 \right)$ . This pattern reflects the diferent operating modes of the methods: LaneTCA aggregates a temporal window and achieves stronger average overlap, whereas SIGMA-Lane performs streaming state propagation and explicitly filters occluded writes. The gains at the stricter threshold match the intended use case: preserving valid lane instances under occlusion. Fig. 7 visualizes results across diverse conditions, and Fig. 8 traces one occluded case from intermediate maps to the recovered lane prediction. Additional examples of frame-to-frame behavior are provided in the supplementary material.

![](images/360e03e34be180102ce32951f81f2170e4697e6ffca47861d35ca5022543fbec.jpg)  
Fig. 6: Comparison of lane detection results on the VIL-100 dataset.

Table 2: Comparison on OpenLane-V.
<table><tr><td>Method</td><td>Approach</td><td colspan="7">mIoU(↑) F1@0.5(↑) F1@0.8(↑) RF@0.5(↓) RM@0.5(↓) RF@0.8(↓) RM@0.8(↓)</td></tr><tr><td rowspan="5">MFIALane [21]  $\mathrm { C o n d L a n e N e t } \ [ 1 5 ]$  GANet [26]</td><td rowspan="5">Image-based</td><td>0.697</td><td>0.723</td><td>0.475</td><td>0.061</td><td>0.300</td><td>0.080</td><td>0.519</td></tr><tr><td>0.698</td><td>0.780</td><td>0.450</td><td>0.047</td><td>0.239</td><td>0.084</td><td>0.531</td></tr><tr><td>0.716</td><td>0.801</td><td>0.530</td><td>0.048</td><td>0.198</td><td>0.082</td><td>0.443</td></tr><tr><td>0.735</td><td>0.789</td><td>0.554</td><td>0.054</td><td>0.224</td><td>0.086</td><td>0.430</td></tr><tr><td>0.740</td><td>0.795</td><td>0.573</td><td>0.043</td><td>0.232</td><td>0.082</td><td>0.409</td></tr><tr><td rowspan="6">ConvLSTM [41] ConvGRUs [34]  $\mathrm { M M A  – N e t ~ [ \dot { 3 } 5 ] ~ }$  RVLD [12]</td><td rowspan="6">Video-based</td><td>0.529</td><td>0.641</td><td>0.353</td><td>0.058</td><td>0.282</td><td>0.091</td><td>0.574</td></tr><tr><td>0.540</td><td>0.641</td><td>0.355</td><td>0.064</td><td>0.288</td><td>0.094</td><td>0.576</td></tr><tr><td>0.574</td><td>0.573</td><td>0.328</td><td>0.044</td><td>0.461</td><td>0.071</td><td>0.671</td></tr><tr><td>0.727</td><td>0.825</td><td>0.566</td><td>0.014</td><td>0.167</td><td>0.051</td><td>0.406</td></tr><tr><td>0.742</td><td>0.836</td><td>0.582</td><td>0.016</td><td>0.162</td><td>0.055</td><td>0.393</td></tr><tr><td>0.757</td><td>0.804</td><td>0.588</td><td>0.025</td><td>0.196</td><td>0.064</td><td>0.399</td></tr><tr><td rowspan="2">LaneTCA [38] SIGMA-Lane</td><td></td><td>0.774</td><td>0.822</td><td>0.574</td><td>0.050</td><td>0.212</td><td>0.084</td><td>0.424</td></tr><tr><td>Video-based</td><td>0.761</td><td>0.838</td><td>0.591</td><td>0.018</td><td>0.162</td><td>0.052</td><td>0.384</td></tr></table>

![](images/557df132d824ed74bfdd3aa7b5f69f81271dfc8304acf6ed7b2097787cc87b5f.jpg)  
Fig. 7: Comparison of lane detection results on the OpenLane-V dataset.

![](images/35560fc4a84bb7fa66ebc8ee172100280a61a846a13352efee84c5f63c0d3874.jpg)  
Fig. 8: Qualitative analysis of SIGMA-Lane under vehicle occlusion. Current-frame maps $( F _ { t } , P _ { t } )$ show weakened lane responses in the occluded region, while refined maps $( \tilde { F } _ { t } , \tilde { P } _ { t } )$ recover the lane topology and produce a prediction closer to GT. Maps are min-max normalized.

## 4.4 Computational Cost

Because SIGMA-Lane targets streaming video inference, we compare the computational cost of representative video lane detectors under the same 384×640 input setting in Tab. 3. All timing is measured on a single NVIDIA RTX 4090 in full-validation mode, using 50 warmup frames and all remaining frames for measurement. Model FPS measures the network-only forward pass, while E2E FPS includes post-processing and NMS inside the per-frame inference loop. Main Params exclude auxiliary ILD models and count only the temporal model used in the final stage. Agg. Params count only temporal aggregation modules. SIGMA-Lane has the lowest FLOPs and temporal aggregation parameters in this comparison, and runs faster than LaneTCA. OMR gives the highest throughput; SIGMA-Lane uses fewer FLOPs and aggregation parameters while obtaining higher accuracy.

Table 3: Computational cost comparison on OpenLane-V. Model FPS denotes network-only forward speed, while E2E FPS includes post-processing and NMS.
<table><tr><td>Method</td><td>FLOPs(G)(↓) Model FPS(↑)</td><td></td><td>E2E FPS(↑)</td><td></td><td>Main Params(M) Agg. Params(M)</td></tr><tr><td>OMR [11]</td><td>54.6</td><td>215.96</td><td>141.15</td><td>7.15</td><td>3.54</td></tr><tr><td>LaneTCA [38]</td><td>28.9</td><td>41.57</td><td>28.48</td><td>1.62</td><td>0.96</td></tr><tr><td>SIGMA-Lane</td><td>27.1</td><td>118.79</td><td>96.74</td><td>3.96</td><td>0.35</td></tr></table>

## 4.5 Ablation Studies

We next isolate each component in the contamination-aware aggregation module. Row 1 in Tab. 4 replaces OMR’s ConvLSTM with a standard Mamba2 [4] operator, with no gating or alignment. This gives a direct backbone-swap reference.

Dual-gating (rows 1–4). The baseline (row 1) is a competitive temporal backbone-swap reference. Input gating (row 2) attenuates the write term $( 1 - m _ { t } ) \bar { B } _ { t } x _ { t }$ to reduce corrupted state writes. Output gating (row 3) reduces the residual contribution of the noisy current frame at the fusion stage. Combining both gates (row 4) gives +0.012 mIoU (0.775→0.787), which accounts for 46% of the total mIoU gain (0.775→0.801) in Tab. 4. Dual-gating also reduces ${ \cal R } _ { F } / { \cal R } _ { M }$ from 0.052/0.084 to 0.024/0.032. The full model reaches 0.023/0.030 after adding alignment, retrieval, and initialization.

Table 4: Ablation studies on VIL-100. DG-I/O: Input/Output Gating, Warp: Afine Warp, SSR: Structural Spatial Retrieval, Token: Geometry-Aware Token.
<table><tr><td colspan="11"># DG-I DG-O Warp SSR Token mIoU(↑) F1@0.5(↑) Accuracy(↑)</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>0.775</td><td>0.926</td><td>0.942</td><td> $R _ { F } @ 0 . 5 ( \downarrow )$  0.052</td><td> $R _ { M } @ 0 . 5 ( \downarrow )$ </td><td>0.084</td></tr><tr><td>2</td><td>√</td><td></td><td></td><td></td><td></td><td>0.781</td><td>0.930</td><td>0.947</td><td>0.036</td><td>0.045</td></tr><tr><td>3</td><td></td><td>√</td><td></td><td></td><td>0.778</td><td>0.929</td><td></td><td>0.946</td><td>0.040</td><td>0.053</td></tr><tr><td>4</td><td>√</td><td>√</td><td></td><td></td><td>0.787</td><td>0.935</td><td></td><td>0.950</td><td>0.024</td><td>0.032</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td></td><td>0.794</td><td></td><td>0.937</td><td>0.952</td><td>0.025</td><td>0.035</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>0.798</td><td>0.938</td><td>0.953</td><td>0.024</td><td>0.033</td></tr><tr><td>7</td><td>√</td><td>√</td><td>√</td><td>√</td><td>V</td><td>0.801</td><td>0.940</td><td>0.956</td><td>0.023</td><td>0.030</td></tr></table>

Lane-guided afine warp (row 5). Adding synchronous alignment gives +0.007 mIoU and keeps the obstacle mask m aligned with the regions it should modulate. Without it, misalignment between warped features and the static mask can leak contaminated signals or over-suppress clean areas, which also hurts F1 and accuracy.

Structural Spatial Retrieval (row 6). SSR adds +0.004 mIoU by retrieving lane-relevant spatial cues from warped historical features via lane-mask-guided cross-attention. It supplies structural information that gating alone cannot reconstruct, and also improves F1@0.5 and accuracy.

Geometry-aware initialization (row 7). With dual-gating, alignment, and retrieval already enabled, the start token further improves the reliability of the initial temporal state. Row 7 gives the best values across all reported metrics, including the lowest missing rate, indicating that a geometry-aware prior helps recurrent aggregation start from lane-relevant spatial structure before suficient history has accumulated.

Ablation summary. These component ablations indicate a clear division of labor. Dual-gating provides the main temporal-stability gain by suppressing contaminated writes and residual fusion, while alignment, SSR, and the start token improve spatial recovery and initialization.

## 4.6 Contamination and Occlusion Analysis

We further analyze whether the improvement comes from the proposed contamination-aware use of obstacle priors, especially under corrupted memory and heavier occlusion.

Mask and gate sensitivity. Tab. 5 separates two possible explanations: whether the gain comes from having an obstacle prior, or from how the temporal model uses that prior. Perturbing the mask has a limited efect. $\mathrm { ~ A ~ } 7 \times 7$ dilation changes F1/mIoU by onl $\mathrm { \Delta \cdot - 0 . 0 0 1 4 / - 0 . 0 0 1 0 }$ , and erasing 20% of obstacle pixels changes them by $+ 0 . 0 0 0 7 / - 0 . 0 0 0 5$ . By contrast, removing both gates while keeping the original mask drops F1/mIoU by $- 0 . 0 1 8 0 / - 0 . 0 1 5 0$ and increases $R _ { M } \ \mathrm { b y } + 0 . 0 7 4 0$ . The result indicates that mask precision is not the main factor; the critical operation is placing the prior on the write and residual-fusion paths.

Table 5: Mask and gate sensitivity on OpenLane-V validation. Deltas are measured relative to clean SIGMA-Lane. Lower is better for $\varDelta R _ { F }$ and $\varDelta R _ { M }$
<table><tr><td>Setting</td><td>∆F1(↑)</td><td>∆mIoU(↑)</td><td> $\varDelta R _ { F } ( \mathscr { L } )$ </td><td> $\varDelta R _ { M } ( \mathscr { L } )$ </td></tr><tr><td>No DG-I/O; original M</td><td>-0.0180</td><td>-0.0150</td><td>+0.0362</td><td> $+ 0 . 0 7 4 0$ </td></tr><tr><td>Full SIGMA; dilate M (7×7)</td><td>-0.0014</td><td>-0.0010</td><td>+0.0012</td><td> $+ 0 . 0 0 2 4$ </td></tr><tr><td>Full SIGMA; erase 20% M=1 +0.0007</td><td></td><td>-0.0005</td><td>+0.0008</td><td>-0.0010</td></tr></table>

Controlled contamination and occlusion severity. The supplementary material reports two additional analyses under stronger stress. Structured memory perturbations along occlusion boundaries increase $R _ { M } @ 0 . 5$ from 0.162 to 0.182 for SIGMA-Lane, but from 0.236 to 0.367 for the ungated control, a 6.6× larger degradation without the gates. In the occlusion-stratified evaluation on OpenLane-V, SIGMA-Lane and OMR are comparable in Light frames, while SIGMA-Lane gains +0.028 mIoU over OMR in both Moderate and Heavy bins.

## 5 Conclusion

This paper studies video lane detection under persistent occlusion from the perspective of SSM state updates. We identify state contamination: occlusioncorrupted observations can be written into temporal memory and afect later predictions. SIGMA-Lane addresses this failure mode by gating the SSM write and residual-fusion paths, with coordinate-consistent alignment, SSR, and geometryaware initialization supporting temporal propagation and structural recovery. Experiments on VIL-100 and OpenLane-V show improved temporal stability with competitive per-frame detection quality. The ablations, mask sensitivity analysis, and controlled contamination tests indicate that the obstacle prior is most useful when it directly controls how information enters and leaves the temporal update. The current design operates in 2D image-feature space, where afine warping provides a local short-range alignment approximation. A natural extension is to combine contamination-aware state updates with 3D geometric cues, camera pose, or depth-aware motion for more complex driving scenes.

Acknowledgements This work was supported by the National Natural Science Foundation of China (Grant Nos. 62403429, 62476247, and 62402442), the Hangzhou Key Research and Development Program (Grant No. 2025SZDA0100), the Zhejiang Provincial Natural Science Foundation of China (Grant Nos. LQN25F030008 and LQ24F020038), and the Fundamental Research Funds for the Provincial Universities of Zhejiang (Grant No. G26081190006).

## References

1. Chen, M., Jia, Q., Yang, J., Liu, S.: SR-LMamba: A lane detection model for complex scenes integrating curvelet transform with Mamba architecture. PLOS ONE 20 (2025)

2. Cheng, Z., Wang, C., Zhang, G., Zhou, W.: Parallel heterogeneous networks with adaptive routing for online video lane detection. IEEE Trans. Intell. Transp. Syst. 26(4), 5225–5235 (2025)

3. Cheng, Z., Zhang, G., Wang, C., Zhou, W.: DILane: Dynamic instance-aware network for lane detection. In: ACCV. pp. 2075–2091 (2022)

4. Dao, T., Gu, A.: Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In: ICML. pp. 10041–10071 (2024)

5. Feng, Z., Guo, S., Tan, X., Xu, K., Wang, M., Ma, L.: Rethinking eficient lane detection via curve modeling. In: CVPR (2022)

6. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023)

7. Gu, A., Goel, K., Ré, C.: Eficiently modeling long sequences with structured state spaces. In: ICLR (2022)

8. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016)

9. Hou, Y., Ma, Z., Liu, C., Hui, T.W., Loy, C.C.: Inter-region afinity distillation for road marking segmentation. In: CVPR (2020)

10. Hou, Y., Ma, Z., Liu, C., Loy, C.C.: Learning lightweight lane detection CNNs by self attention distillation. In: ICCV (2019)

11. Jin, D., Kim, C.S.: OMR: Occlusion-aware memory-based refinement for video lane detection. In: ECCV. pp. 129–145 (2024)

12. Jin, D., Kim, D., Kim, C.S.: Recursive video lane detection. In: ICCV. pp. 8473– 8482 (2023)

13. Jin, D., Park, W., Jeong, S.G., Kwon, H., Kim, C.S.: Eigenlanes: Data-driven lane descriptors for structurally diverse lanes. In: CVPR (2022)

14. Li, K., Li, X., Wang, Y., He, Y., Wang, Y., Wang, L., Qiao, Y.: VideoMamba: State space model for eficient video understanding. arXiv preprint arXiv:2403.06977 (2024), eCCV 2024

15. Liu, L., Chen, X., Zhu, S., Tan, P.: CondLaneNet: A top-to-down lane detection framework based on conditional convolution. In: ICCV (2021)

16. Liu, R., Yuan, Z., Liu, T., Xiong, Z.: End-to-end lane shape prediction with transformers. In: WACV (2021)

17. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., Liu, Y.: VMamba: Visual state space model. arXiv preprint arXiv:2401.10166 (2024), neurIPS 2024

18. Neven, D., De Brabandere, B., Georgoulis, S., Proesmans, M., Van Gool, L.: Towards end-to-end lane detection: An instance segmentation approach. In: Intelligent Vehicles Symposium (2018)

19. Pan, X., Zhan, X., Shi, J., Luo, P., Wang, X., Tang, X.: Spatial as deep: Spatial CNN for trafic scene understanding. In: AAAI. pp. 7276–7283 (2018)

20. Qin, Z., Wang, H., Li, X.: Ultra fast structure-aware deep lane detection. In: ECCV (2020)

21. Qiu, Z., Zhao, J., Sun, S.: Mfialane: Multiscale feature information aggregator network for lane detection. IEEE Transactions on Intelligent Transportation Systems 23(12), 24263–24275 (2022)

22. Su, H., Wu, W., Song, F., Zhang, J., Yang, Z., Yan, J.: DriveMamba: Task-centric scalable state space model for eficient end-to-end autonomous driving. In: ICLR (2026)

23. Tabelini, L., Berriel, R., Paixao, T.M., Badue, C., De Souza, A.F., Oliveira-Santos, T.: Keep your eyes on the lane: Real-time attention-guided lane detection. In: CVPR (2021)

24. Tabelini, L., Berriel, R., Paixao, T.M., Badue, C., De Souza, A.F., Oliveira-Santos, T.: PolyLaneNet: Lane estimation via deep polynomial regression. In: ICPR. pp. 6150–6156 (2021)

25. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)

26. Wang, J., Ma, Y., Huang, S., Hui, T., Wang, F., Qian, C., Zhang, T.: A keypointbased global association network for lane detection. In: CVPR (2022)

27. Wang, M., Zhang, Y., Feng, W., Zhu, L., Wang, S.: Video instance lane detection via deep temporal and geometry consistency constraints. In: ACM MM (2022)

28. Wang, X., Yin, Y., Huang, F., Bao, X.: MLM-Net: Streamlined multi-lane detection network with spatio-temporal memory for video instance lane detection. In: IEEE Int. Conf. Intell. Transp. Syst. (2023)

29. Xiao, L., Li, X., Yang, S., Yang, W.: ADNet: Lane shape prediction via anchor decomposition. In: ICCV (2023)

30. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: SegFormer: Simple and eficient design for semantic segmentation with transformers. In: NeurIPS. pp. 12077–12090 (2021)

31. Xu, H., Wang, S., Cai, X., Zhang, W., Liang, X., Li, Z.: CurveLane-NAS: Unifying lane-sensitive architecture search and adaptive point blending. In: ECCV (2020)

32. Xu, S., Cai, X., Zhao, B., Zhang, L., Xu, H., Fu, Y., Xue, X.: RCLane: Relay chain prediction for lane detection. In: ECCV (2022)

33. You, Z., Wang, N., Wang, H., Zhao, Q., Wang, J.: MambaBEV: An eficient 3D detection model with Mamba2. arXiv preprint arXiv:2410.12673 (2025)

34. Zhang, J., Deng, T., Yan, F., Liu, W.: Lane detection model based on spatiotemporal network with double convolutional gated recurrent units. IEEE Trans. Intell. Transp. Syst. 23(7), 6666–6678 (2021)

35. Zhang, Y., Zhu, L., Feng, W., Fu, H., Wang, M., Li, Q., Li, C., Wang, S.: VIL-100: A new dataset and a baseline model for video instance lane detection. In: ICCV. pp. 15681–15690 (2021)

36. Zheng, T., Fang, H., Zhang, Y., Tang, W., Yang, Z., Liu, H., Cai, D.: RESA: Recurrent feature-shift aggregator for lane detection. In: AAAI (2021)

37. Zheng, T., Huang, Y., Liu, Y., Tang, W., Yang, Z., Cai, D., He, X.: CLRNet: Cross layer refinement network for lane detection. In: CVPR. pp. 898–907 (2022)

38. Zhou, K., Li, L., Zhou, W., Wang, Y., Feng, H., Li, H.: LaneTCA: Enhancing video lane detection with temporal context aggregation. IEEE TCSVT 35(9), 8574–8585 (2025)

39. Zhou, K.: Lane2seq: Towards unified lane detection via sequence generation. In: CVPR. pp. 16944–16953 (2024)

40. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., Wang, X.: Vision mamba: Eficient visual representation learning with bidirectional state space model. In: ICML (2024)

41. Zou, Q., Jiang, H., Dai, Q., Yue, Y., Chen, L., Wang, Q.: Robust lane detection from continuous driving scenes using deep neural networks. IEEE Trans. Veh. Technol. 69(1), 41–54 (2020)