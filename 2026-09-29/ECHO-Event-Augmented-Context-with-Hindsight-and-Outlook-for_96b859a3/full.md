# ECHO: Event-Augmented Context with Hindsight and Outlook for Wrist-Only Manipulation

Xinyue Wang<sup>1∗</sup>, Yicheng Jiang<sup>1∗</sup>, Zesen Gan<sup>1</sup>, Junhao He<sup>2</sup>, Jiaxu Wang<sup>3</sup>, Junhao Li<sup>1</sup>, Jingtao Zhang<sup>1</sup>, Tianlun He<sup>3</sup>, Jianan Wang<sup>4</sup>, Isabel Guan<sup>1,5B</sup> and Qiming Shao<sup>1B</sup>

Abstract— Learning-based manipulation policies relying on RGB cameras often suffer from degraded observations under extreme exposure. Event cameras mitigate this degradation by asynchronously detecting pixel-level intensity changes to offer a high dynamic range. However, their observations heavily depend on camera placement, as fixed cameras miss static scene content while wrist-mounted camera motion causes previously visited regions to leave the field of view. To address these spatial-temporal limitations, we present ECHO (Eventaugmented Context with Hindsight and Outlook), a wristonly latent world action model that encodes wrist events into compact motion representations to provide temporal and spatial context for policy reasoning. Specifically, ECHO utilizes a pretrained event encoder to explain visual-feature changes between frames. Its hindsight module preserves the gripper trajectory with past event stream as addressable off-camera context. Concurrently, the outlook module introduces learnable event foresight queries supervised to anticipate the event window for future actions, enabling the policy to predict upcoming scene changes. Evaluated on wrist-only RLBench tasks, ECHO outperforms RGB and RGB+event baselines by 20.6 and 12.0 percentage points under normal lighting, and by 14.6 and 11.3 points under severe exposure drops, respectively, while also surpassing RGB references using a third-person camera. Realworld experiments with a wrist-mounted event camera validate that ECHO outperforms RGB-only and RGB+event baselines across multiple tasks under both nominal and severely dark lighting. Project page is at https://echo-wam.github.io/.

## I. INTRODUCTION

Recent years have seen remarkable progress in robotic manipulation, progressing from foundational imitation learning paradigms to sophisticated vision language action and world action models [1], [2], [3], [4], [5]. Although these models have achieved impressive success on diverse tasks, current evaluations mostly depend on standard RGB visual inputs captured in bright, ideal lighting conditions. Yet real-world deployments involving changing lighting conditions or rapid camera motion suffer from severe visual degradation, causing policies to fail with corrupted frames.

To address these visual failure modes, event cameras provide a compelling sensing modality by asynchronously report per-pixel brightness changes with microsecond timestamp resolution and high dynamic range [6]. They capture interactions such as approach, slip, and release when RGB frames are over-exposed, blur, or darken, provided the induced changes exceed the sensor’s contrast threshold. Events therefore offer complementary motion cues for manipulation when RGB observations become unreliable.

However, event cameras respond only to brightness changes, introducing a trade-off between fixed and wristmounted configurations. When the camera is fixed, it misses static scene details that produce no brightness change. Mounting the camera on the wrist instead allows it to actively probe scene layout through motion, but the same motion also removes previously visited regions from view. This loss of visual context calls for information beyond the current observation. Existing event-augmented policies primarily incorporate event cues through visual fusion or residual action pathways [7], [8], without jointly organizing trajectory context, current event observations, and event foresight in the policy prefix.

We present ECHO, a wrist-only latent world action model that integrates events as a policy modality across past, present, and future. A pretrained encoder transforms pixellevel motion into latent representations that support both policy conditioning and future-event prediction. Looking back, a bounded trajectory memory combines visited gripper locations with current event features to provide off-camera context; looking forward, event foresight queries predict a dense representation of the event window spanned by the current action, anticipating scene changes before they are observed.Following the mainstream prefix-expert design, a vision-language backbone encodes observations and instructions into a token prefix, and a flow-matching expert generates continuous actions.

Our contributions are summarized as follows:

• A hybrid convolution-attention event encoder pretrained via cross-modal visual reconstruction and time-reversal contrastive learning to extract robust spatio-temporal motion priors.

• A wrist-only perception framework integrating trajectory memory to keep out-of-view spatial cells addressable, and eventforesight queries to predict future actionconditioned event observations.

• Comprehensive evaluations on RLBench and real-robot platforms showing that ECHO consistently surpasses wrist RGB and RGB+event baselines under both standard and extreme low-light conditions (Fig. 1).

![](images/1ed1d19bfd018655cda239c627698e65db5054694ef23f7ee9369b339a273c93.jpg)  
Fig. 1. Success rates on the wrist-only RLBench under normal lighting and a −4 EV exposure setting. ECHO achieves the highest result in both conditions.

## II. RELATED WORK

## A. Vision-language-action policies

Vision-language-action (VLA) policies differ in action representation and decoding. RT-2 and OpenVLA generate action tokens from vision-language backbones [1], [2]; CogACT couples a vision-language model to a diffusion action module [3]; and $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ generate continuous action chunks with flow-matching experts [4], [5]. Following modern prefix-expert designs, ECHO augments an observationand-language prefix with event-derived context to condition the flow-matching action expert.

## B. World models and latent action prediction

World models predict future observations to support planning and policy learning. UniPi recovers actions from video plans through inverse dynamics [9], while GR-1 and World-VLA jointly predict future observations and actions [10], [11]. Ctrl-World supports policy evaluation and improvement through imagined rollouts [12]. Other approaches focus on changes between observations, encoding them as compact latent actions [13], [14], [15], [16], [17], [18], [19] or continuous motion representations [20]. FLARE and VLA-JEPA further connect latent world modeling with policy learning through future-feature prediction [21], [22]. Following this latent world action modeling perspective, ECHO treats events as an efficient representation of observation changes. Its event encoder maps these signals into a compact latent space for policy reasoning, trajectory memory, and event foresight.

## C. Memory for partial observability and wrist views

Memory supports manipulation when decisions depend on information no longer visible. MemoryVLA retrieves perceptual and semantic information from a memory bank [23]; HALO combines video question-answering supervision with sparse attention to retrieve task-relevant interaction history [24]; and $\mu \mathrm { V L A }$ carries learnable memory tokens across timesteps through recurrent self-attention [25].

Wrist-camera motion further motivates spatial memory. AtlasVLA combines a persistent voxel-hashed world state with a memory of robot state and task progress for manipulation from a single wrist camera [26]. Mem-World uses wristcentered, surfel-indexed memory to retrieve historical views for consistent prediction in a multi-view world model [27]. ECHO quantizes the end-effector trajectory into visited spatial cells and embeds their positions together with the current event summary, keeping previously visited places accessible after they leave the frame.

## D. Event cameras in robot learning

Event cameras provide high temporal resolution and dynamic range [6], supporting obstacle avoidance, visual servoing, grasping, and visuomotor control [28], [29], [30], [31]. Related work also explores early action prediction from event histories [32], event-language modeling [33], and synchronized event and manipulation datasets [34]. Recent event-augmented VLAs improve robustness to blur and low exposure by augmenting pretrained RGB policies with event cues. E-VLA fuses events into RGB inputs or visual features through overlays or hierarchical adapters [7], while Event-VLA introduces a gated residual fusion pathway after the VLA backbone[8]. ECHO instead treats events as a policy modality spanning past, present, and future within a shared prefix. Current event features tag previously visited gripper locations, observed event tokens support present perception, and foresight queries anticipate future event representations.

## III. PRELIMINARIES

## A. Policy interface and flow-matching action expert

Following common practice in modern manipulation policies, we adopt a prefix-expert architecture. At decision step t, let $I _ { t }$ denote the wrist RGB observation, ℓ the language instruction, and $s _ { t }$ the proprioceptive state. A vision-language backbone constructs the conditioning prefix as

$$
h _ { t } = B _ { \psi } ( I _ { t } , \ell ) .\tag{1}
$$

The action expert receives the proprioceptive state $s _ { t }$ as a dedicated suffix token alongside the noisy action tokens.

The action expert conditions on this prefix and generates a chunk of H actions. We write

$$
a _ { t : t + H - 1 } \triangleq \left( a _ { t } , \ldots , a _ { t + H - 1 } \right) \in \mathbb { R } ^ { H \times d _ { a } } ,\tag{2}
$$

where $d _ { a }$ is the action dimension.

Flow matching trains the expert to transport Gaussian noise to the demonstrated action chunk [35], [4]. For a training sample, draw $\varepsilon _ { a } \sim \mathcal { N } ( 0 , I )$ and interpolate with

$$
x _ { \tau } = \tau \varepsilon _ { a } + \left( 1 - \tau \right) a _ { t : t + H - 1 } , \qquad \tau \in [ 0 , 1 ] ,\tag{3}
$$

with target velocity

$$
v _ { a } ^ { \star } = \varepsilon _ { a } - a _ { t : t + H - 1 } .\tag{4}
$$

At inference, numerical integration starts at $\tau \ = \ 1$ and follows the learned velocity field toward $\tau = 0$ , yielding the action chunk.

![](images/0f8a2775405b8581ff1e3e653ac29cb2655ec9417954150bd155ff3aa88e9925.jpg)  
Fig. 2. Overview of ECHO. (a) The event encoder is pretrained to represent the visual transition between consecutive RGB observations using frozen DINOv3 features and a latent world model. (b) During policy learning, wrist RGB, current event latent tokens $E _ { t } ,$ , language, off-camera context $\mathbf { \bar { \boldsymbol { C } } } _ { t }$ built from visited poses and current events, and foresight queries $\dot { F } _ { t }$ supervised by future event latents form the vision-language prefix. A flow-matching action expert then predicts continuous action chunks from this prefix and proprioception.

## B. Event Stream and Voxel Representation

An event $\boldsymbol { e } _ { k } = ( x _ { k } , y _ { k } , t _ { k } , p _ { k } )$ records pixel coordinates, timestamp, and polarity $p _ { k } \in \{ - 1 , + 1 \}$ . As illustrated in Fig. 2, positive events (intensity increases) are shown in red, and negative events (intensity decreases) in blue. Events are triggered whenever the log intensity change crosses the contrast threshold [6].

To interface with a neural encoder, we accumulate asynchronous events over a decision window ending at t into a signed voxel volume:

$$
V _ { t } \in \mathbb { R } ^ { C \times H _ { e } \times W _ { e } } , \qquad C = 6 .\tag{5}
$$

Specifically, events within the window (e.g., the previous completed keystep segment in simulation) are aggregated into $C \ = \ 6$ equal-duration temporal bins. Signed event counts are halved and clipped to $[ - 1 , 1 ]$ , preserving both polarity and coarse temporal ordering. For training in simulation, events are rendered via the ESIM simulator [36].

The event encoder $f _ { \theta }$ maps this volume to N event latent tokens,

$$
E _ { t } = f _ { \theta } ( V _ { t } ) \in \mathbb { R } ^ { N \times D } , \qquad N = 6 4 ,\tag{6}
$$

which can be concatenated with the backbone prefix. ECHO trains $f _ { \theta }$ to explain changes in frozen visual teacher features and uses the event latent for current perception, spatial context, and future-event supervision. The encoder architecture and pretraining are described in Sec. IV-B, and its prefix blocks in Secs. IV-C and IV-D.

## IV. METHODOLOGY

## A. Overview

ECHO extends the policy in Sec. III-A with event features $E _ { t }$ , off-camera context $\mathcal { C } _ { t }$ , and foresight queries $\mathcal { F } _ { t }$ (Fig. 2). The resulting policy is

$$
a _ { t : t + H - 1 } = \pi ( I _ { t } , \ell , s _ { t } , E _ { t } , \mathcal { C } _ { t } , \mathcal { F } _ { t } ) .\tag{7}
$$

These prefix blocks provide present, past, and future context.

1) Event features (present). The encoder $f _ { \theta }$ extracts a compact event latent $E _ { t }$ from the observed volume $V _ { t }$ (Sec. IV-B).

2) Off-camera context (past). The gripper trajectory is quantized into visited spatial cells, and M context slots combine cell positions with the current event summary to keep visited places addressable outside the wrist view (Sec. IV-C).

3) Event foresight (future). Learnable queries $\mathcal { F } _ { t }$ are supervised to predict the latent representation of the future event window covered by the planned action (Sec. IV-D).

The prefix contains image, event, language, context, and foresight tokens in order. The action expert receives a proprioceptive state token and H noisy action tokens. Implementation settings are given in Sec. V-A.

## B. Event feature encoder and pretraining

To capture fine-grained interaction dynamics, we design a convolution-attention encoder $f _ { \theta } \ ( \mathrm { F i g } . \ 3 )$ that processes a wrist event volume $V _ { t }$ into a compact spatial latent grid:

$$
E _ { t } = f _ { \theta } ( V _ { t } ) \in \mathbb { R } ^ { N \times D } ,\tag{8}
$$

![](images/f23b91c02bade73bad3684188734774eb19d751780b4e04870a8201b11da3ccf.jpg)  
Fig. 3. Event encoder pretraining. The event volume is encoded into an event latent that can (i) reconstruct the frozen RGB teacher’s transition through a warp and synthesis head, and (ii) separate true motion window from its reversed twin under a motion-direction contrast (MDC).

where N denotes the number of spatial tokens and D is their feature dimension.

We pretrain $f _ { \theta }$ through a latent world model formulation, enabling event representations to predict visual-feature transitions and capturing temporal directionality under a frozen RGB teacher ϕ (DINOv3 [37]). Specifically, given $\begin{array} { r l } { \mathbf { u } _ { t } } & { { } = } \end{array}$ $\phi ( I _ { t } )$ and $\mathbf { u } _ { t + \Delta } = \phi ( I _ { t + \Delta } )$ , the event latent explains their transition while $\mathbf { u } _ { t }$ provides appearance. Multi-layer teacher features are aggregated, compressed, and spatially aligned to match the event token grid, grounding event tokens in rich visual semantics. Visual change. A warp head predicts pertoken displacements and backward-warps the current teacher features as

$$
W = \mathrm { W a r p } \big ( { \bf u } _ { t } , g _ { \mathrm { w } } ( E _ { t } ) \big ) .\tag{9}
$$

A synthesis branch applies attention to $\mathbf { u } _ { t } ,$ injecting event features at each block to produce

$$
Q = g _ { \mathrm { s } } \big ( E _ { t } , \mathbf { u } _ { t } \big ) .\tag{10}
$$

A gate maps [u<sub>t</sub>, W, Q] to two logits per token and applies a softmax across branches. With warp weight α, the prediction is

$$
\widehat { \mathbf { u } } _ { t + \Delta } = \alpha \odot W + ( 1 - \alpha ) \odot Q .\tag{11}
$$

Reconstruction targets the full future teacher feature using SmoothL1. Motion is emphasized through detached weights $w _ { i } = d _ { i } / \mathrm { m e a n } _ { j } d _ { j }$ , where $d _ { i } = \| ( \mathbf { u } _ { t + \Delta } ) _ { i } - ( \mathbf { u } _ { t } ) _ { i } \| _ { 2 }$ . The reconstruction loss is

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { H _ { \phi } W _ { \phi } } \sum _ { i = 1 } ^ { H _ { \phi } W _ { \phi } } w _ { i } \mathrm { S m o o t h L 1 } \big ( ( \hat { \mathbf { u } } _ { t + \Delta } ) _ { i } , ( \mathbf { u } _ { t + \Delta } ) _ { i } \big ) ,\tag{12}
$$

where $H _ { \phi }$ and $W _ { \phi }$ are the spatial dimensions of the teacher feature grid, so the loss averages over its spatial tokens.

The reversed branch swaps the frames and uses V<sup>rev</sup> (defined below) with the same weights and loss to reconstruct u<sub>t</sub>, yielding $\mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { r e v } }$

Motion-direction contrast. To encourage the encoder to capture motion directionality, we process an anchor event volume $V ^ { \mathrm { a n c } }$ , a duration-jittered positive view $V ^ { \mathrm { p o s } }$ , and a time-reversed view $V ^ { \mathrm { r e v } } = - \mathrm { F l i p } ( V ^ { \mathrm { a n c } } )$ created by reversing temporal bins and inverting event polarity. For each view $v \in$ {anc, pos, rev}, the encoder produces $E ^ { v } = f _ { \theta } ( V ^ { v } )$ and each token is $\ell _ { 2 } \cdot$ -normalized as $z _ { i } ^ { v } = E _ { i } ^ { v } / \| E _ { i } ^ { v } \| _ { 2 }$ for the contrastive loss. Spatial selection of informative regions is achieved by max-pooling teacher feature change magnitudes $\left\| \mathbf { u } _ { t + \Delta } - \mathbf { u } _ { t } \right\|$ onto the event token grid, which identifies the set of most active tokens M. The motion-direction contrastive (MDC) loss then enforces a margin m between the positive and time-reversed views:

$$
\mathcal { L } _ { \mathrm { M D C } } = \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \left[ \cos ( z _ { i } ^ { \mathrm { a n c } } , z _ { i } ^ { \mathrm { r e v } } ) - \cos ( z _ { i } ^ { \mathrm { a n c } } , z _ { i } ^ { \mathrm { p o s } } ) + m \right] _ { + } .\tag{13}
$$

Variance Regularization. We apply a batch-wise variance floor to the event encoder outputs E before $\ell _ { 2 }$ normalization:

$$
\mathcal { L } _ { \mathrm { v a r } } = \frac { 1 } { N D } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { D } \bigl [ \tau - \mathrm { s t d } _ { b } ( E _ { b , i , c } ) \bigr ] _ { + } ,\tag{14}
$$

where $E _ { b , i , c }$ denotes channel c of token i in batch sample $b ,$ and τ is the target standard deviation threshold. The complete pretraining objective combines spatial reconstruction, temporal contrast, and feature variance regularization as

$$
\mathcal { L } _ { \mathrm { p r e } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { r e v } } + \lambda _ { \mathrm { v a r } } \mathcal { L } _ { \mathrm { v a r } } + \lambda _ { \mathrm { M D C } } \mathcal { L } _ { \mathrm { M D C } } ,\tag{15}
$$

where $\lambda _ { \mathrm { v a r } }$ and λ<sub>MDC</sub> weight variance regularization and motion contrast. Both constraints act directly on the event latent tokens provided to the downstream policy.

## C. Off-camera context

To retain spatial context after visited regions drift out of the wrist camera’s field of view, we introduce a bounded trajectory memory. Given the end-effector position $p _ { \tau } \in \mathbb { R } ^ { 3 }$

at step τ relative to a fixed origin $p _ { 0 } .$ , we quantize the continuous motion into discrete spatial cells:

$$
c _ { \tau } = \mathrm { r o u n d } \bigg ( \frac { p _ { \tau } - p _ { 0 } } { \delta } \bigg ) \in \mathbb { Z } ^ { 3 } ,\tag{16}
$$

where $\delta$ is the grid resolution and $q _ { c } = p _ { 0 } + \delta c$ defines the cell center. The memory buffers the first K unique cells in order of visitation. To construct the final hindsight context, we retrieve the prefix sequence of length $M .$ , beginning with the origin cell $c _ { 0 }$ . Each context token combines a position embedding with the current event summary $\bar { e } _ { t } \ =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( E _ { t } ) _ { n } \in \bar { \mathbb { R } } ^ { D } } \end{array}$ as

$$
m _ { c } = \psi ( q _ { c } ) + W _ { e } \bar { e } _ { t } ,\tag{17}
$$

where $\psi$ is an MLP and $W _ { e }$ projects to backbone width. Visited cells persist across steps, while the shared event term is recomputed from the current window.

The off-camera context block $\mathcal { C } _ { t } = ( m _ { ( 1 ) } , \ldots , m _ { ( M ) } )$ follows language tokens and precedes foresight queries. Image, event, and language tokens share an attention segment; $\mathcal { C } _ { t }$ $\mathcal { F } _ { t }$ , and the action expert each begin a new one. Tokens attend bidirectionally within each segment and to all preceding segments, and the unused context slots are zero-padded.

## D. Event Foresight Queries

To capture future action-conditioned dynamics, we append $N _ { f }$ learnable tokens $\mathcal { F } _ { t } = Q + P + \tau _ { \mathrm { t y p e } } \in \mathbb { R } ^ { N _ { f } \times \bar { d _ { \mathrm { v l m } } } }$ to the observation prefix. These tokens attend over the entire observation prefix (including $I _ { t } , \ell ,$ , current event tokens, and $\mathcal { C } _ { t } )$

During training, foresight queries regress the future event latent $\mathbf { f } _ { t }$ generated by a frozen event encoder over the action horizon $H$ spanned by $\scriptstyle a _ { t : t + H - 1 }$ . This encoder output serves as the target of an MSE loss with stop-gradient

$$
{ \mathcal { L } } _ { \mathrm { f o r e s i g h t } } = \left\| \operatorname { R e a d o u t } ( { \mathcal { F } } _ { t } ) - \operatorname { s g } ( \mathbf { f } _ { t } ) \right\| _ { 2 } ^ { 2 } ,\tag{18}
$$

where $\mathbf { f } _ { t }$ is obtained by spatial average pooling of the frozen encoder’s event latent tokens into $N _ { f }$ target tokens. Samples lacking valid future event windows are pad-masked and excluded from the loss.

$$
\mathcal { L } = w ( \mathrm { s t e p } ) \mathcal { L } _ { \mathrm { a c t i o n } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f o r e s i g h t } } .\tag{19}
$$

Fig. 4 visualizes event foresight through decoded future events and RGB reconstructions. The decoded events capture the main polarity patterns and motion locations, while the RGB reconstructions preserve the future object layout with smoother details than the ground truth.

Fig. 5 shows that trajectory memory tokens and foresight queries attend to interaction-relevant regions when actionexpert attention is dominated by apparent background motion. The memory and foresight attention align with their intended functions, for spatial grounding and anticipating interaction dynamics.

![](images/50133cdcf8b422097349be294f5224197f29b0f3f8118bca25e65689b8de94b2.jpg)

Fig. 4. Qualitative visualization of future predictions. Current event and RGB observations are shown with future ground truth and decoded predictions. For visualization only, lightweight ViT heads decode events from foresight queries and reconstruct RGB images from future teacher features predicted by the latent world model conditioned on current teacher features and foresight queries  
![](images/0fdf2024e22d1594ed28e7419ba0378a447c49a1ce1d399bb868772f42daeade.jpg)  
Fig. 5. Qualitative attention visualization of different ECHO components. Under large wrist-camera motion where action tokens attend heavily to background movement, memory tokens concentrate precisely at gripper– object contact points. Meanwhile, foresight queries cover broader surrounding regions to anticipate interaction dynamics.

## V. EXPERIMENTS

## A. Implementation details

We evaluate our approach on six representative RLBench tasks [38] under a wrist-camera-only setting, utilizing wrist RGB images and the wrist event stream. For each task, we utilize episodes 0 to 99 for training, and episodes 100 to 124 for evaluation.

The policy is trained for 20k steps with a global batch size of 32 across two H20 GPUs. The event encoder, foresight queries, their shared readout, and context projections are jointly trained with a learning rate of $1 \times 1 0 ^ { - 4 }$ . The encoder starts from its pretrained weights (Sec. IV-B), while a frozen copy provides the foresight target. The queries are randomly initialized. Other modules are initialized from $\pi _ { 0 }$

Following Event-VLA [8], during policy training, the wrist RGB input is randomly pad-masked with a probability of 0.3 while retaining the complete event stream for all the methods incorporating event inputs. At closed-loop evaluation, event streams between consecutive RGB frames are generated online via the same dense ESIM pipeline used for training data generation [36]. Simulation uses ESIM events, and the real-robot experiments in Sec. V-D use a physical DVSync camera from DVSense [39].

Model structure. ECHO pairs a vision-language prefix backbone with a flow-matching action expert; both blocks are initialized from $\pi _ { 0 }$ [4]. The event encoder has approximately 60M parameters and maps six-channel event volumes to N=64 latent tokens of width $\scriptstyle D = 1 0 2 4$ , arranged on an $8 \times 8$ grid in row-major order.

w/o syn  
Event 0→1  
Frame 1 (GT)  
![](images/dfc34a09e4325bdf0216519d0fdd1c53984f7a7a8e280add87a92baf124d810a.jpg)  
Fig. 6. Qualitative reconstruction examples of the pretrained event encoder, on a simulation sequence and a real-robot sequence.

The context module utilizes a spatial resolution of $\delta { = } 0 . 0 2 5 \mathrm { m }$ , retains up to K=64 visited cells, and reads $M { = } 1 6$ context slots in first-seen order. Cell positions are embedded with a two-layer MLP. The foresight block appends $N _ { f } { = } 1 6$ queries, supervised by future event tokens pooled from an 8×8 to a 4×4 grid using 2×2 average pooling. We set the action horizon to $H { = } 1$ for RLBench keystep prediction following [40] and H=16 for real-robot deployment (Sec. V-D).

Encoder pretraining. DINOv3-ViT-L/16 processes images with a resolution of 256×256 pixels. Features from layers {6, 12, 18, 24} are concatenated, projected by offline PCA to 256 dimensions, LayerNormalized per token, and averaged over four one-pixel shifts to reduce patch-grid aliasing. Event tokens are aligned to the resulting 16×16 teacher grid by nearest replication, each covering a $2 \times 2$ block. Training pairs span $\Delta \in \{ 1 , 2 , 3 \}$ keystep segments. We use SmoothL1 with $\beta { = } 0 . 1$ , duration jitter up to 20%, the top-10% active tokens for MDC, a variance threshold τ=0.5, and $\lambda _ { \mathrm { v a r } } { = } 1$ . Both reconstruction branches have unit weight.

## B. Main results

In Table I, checkmarks identify the active components, with Traj-ctx and Event-ctx denoting trajectory and event context tokens. The <sup>∗</sup> variant places foresight queries in the action expert rather than the prefix. The highest value in each column and lighting condition is bold, and the next highest is underlined. Red parenthesized SR deltas are measured relative to the π<sub>0</sub> + event baseline.

Overview. On the six-task wrist-only benchmark (Table I), ECHO achieves a 69.3% average success rate under normal lighting, ranking first among wrist-only methods. It outperforms the RGB-only wrist baselines π<sub>0</sub> (48.7%) and CogACT (48.0%), the event-augmented wrist baseline π<sub>0</sub> + event (57.3%), and the third-person RGB references (55.3% frontonly, 64.7% front+wrist). Under an extreme −4 EV exposure shift (about 1/16 illuminance, right side of Table I), ECHO maintains 53.3% success, ahead of π<sub>0</sub> + event (42.0%), the wrist-only RGB baselines (38.7% π<sub>0</sub>, 34.7% CogACT), and the third-person RGB references (43.3% front-only, 37.3% front+wrist).

## C. Ablation Study

Results and Analysis. Table I reports ablations of the ECHO components.

1) Pretrained Event Representation. We first compare the wrist-only RGB baseline with $\pi _ { 0 } ~ +$ event, which adds pretrained event tokens without context or foresight. This increases average success from 48.7% to 57.3% under normal lighting and from 38.7% to 42.0% under severe darkness (−4 EV), indicating that current event observations provide useful cues for action generation (Table I). To isolate the contribution of pretraining, we then compare ECHO w/o foresight with ECHO w/o pretrained encoder,foresight. Both variants retain the trajectory and event context blocks and omit foresight queries, while the latter trains the event encoder from scratch during policy learning. Pretraining raises success from 56.0% to 65.3% under normal lighting and from 40.0% to 42.7% under darkness, giving gains of 9.3 and 2.7 percentage points, respectively. Fig. 6 further separates the roles of the two reconstruction branches. The warp head transports current semantic features along eventinduced displacements and captures the motion direction of content present in the current frame. The synthesis head can introduce newly visible information, although its output is less constrained to preserve the current semantic structure. In the real-robot example, variants containing the synthesis branch recover the changing gripper configuration, while the warp-only reconstruction does not. Together, these results indicate that latent-wm style encoder pretraining provides a useful initialization for downstream policy learning, with benefits even when RGB observations are well exposed.

2) Off-Camera Trajectory Memory. Incorporating the trajectory memory block onto the event-input baseline (ECHO w/o foresight vs. π + event) boosts normal-light success from 57.3% to 65.3% (Table I). The largest per-task gains appear in multi-stage tasks where the wrist camera temporarily loses sight of targets during workspace exit and re-entry, e.g., umbrella (5 → 10). However, under extreme darkness (Table I), maintaining trajectory memory alone fails to prevent low-light failure, yielding limited gain over wrist RGB (42.7% vs. 38.7%).

3) Event Foresight Queries. Event foresight queries serve as the primary mechanism for mitigating severe low-light degradation, boosting dark success from 42.7% to 53.3% (+10.6%, Table I), with substantial recoveries in contactheavy tasks such as umbrella (4 → 11) and toilet (16 → 23). Notably, foresight queries depend on spatial grounding from the trajectory memory, training without context (ECHO w/o context) drops normal-lighting success to 54.0% (below the 57.3% baseline). Furthermore, relocating foresight predictions from the prefix into the action expert (ECHO w/o prefix foresight) lowers performance significantly, suggesting that event foresight functions more effectively as a prefix context than inside the action expert.

In summary, pretrained event representations provide fundamental motion priors, trajectory memory anchors persistent spatial representations when workspace regions leave the wrist field of view, and event foresight yields predictive cues that sustain policy execution under severe photometric degradation.

TABLE I  
SUCCESS RATE (%) ON SIX RLBENCH TASKS USING THE WRIST-ONLY CAMERA AND EVENT PROTOCOL OF SEC. V. RESULTS ARE REPORTED UNDER NORMAL LIGHTING (LEFT) AND A −4 EV EXPOSURE SHIFT (RIGHT).
<table><tr><td>Method</td><td colspan="5">Event Pretrained Traj-ctx Event-ctx Foresight</td><td>laptop N</td><td>D N</td><td>toilet D</td><td>fridge N D</td><td>N</td><td>umbrella D</td><td>frame N D</td><td>N</td><td>plants D</td><td>SR (%)</td></tr><tr><td>π0 (Front only)</td><td></td><td></td><td></td><td></td><td>16</td><td>17</td><td>24</td><td>17</td><td>18</td><td>12</td><td>3</td><td>5</td><td>1</td><td></td><td>55.3 (-2.0)|43.3 (+1.3)</td></tr><tr><td>π0 (Front + wrist)</td><td></td><td></td><td></td><td></td><td>19</td><td>9</td><td>24</td><td>17</td><td>20</td><td>15</td><td>5 11</td><td>4-3</td><td>81</td><td>6-2</td><td>64.7(+7.4)|37.3 (−4.7)</td></tr><tr><td>π0 (Wrist only)</td><td></td><td></td><td></td><td></td><td>18</td><td>17</td><td>24</td><td>12</td><td>20</td><td>4</td><td>5</td><td>3</td><td>4</td><td></td><td>48.7 (−8.6)|38.7(−3.3)</td></tr><tr><td>CogACT (Wrist only)</td><td></td><td></td><td></td><td></td><td>23</td><td>10</td><td>20</td><td>17</td><td>19</td><td>0</td><td>0</td><td>8 4-0</td><td>3</td><td>32</td><td>48.0(−9.3)|34.7(−7.3)</td></tr><tr><td>π0 + event</td><td>√</td><td></td><td></td><td></td><td>24</td><td>20</td><td>24</td><td>14</td><td>17</td><td>5</td><td>9</td><td>3</td><td>8</td><td>3</td><td>57.3 (+0.0)|42.0 (+0.0)</td></tr><tr><td>ECHO (action-expert event foresight)</td><td>√</td><td></td><td>√</td><td>√*</td><td>8</td><td>10</td><td>23</td><td>17</td><td>22</td><td>5</td><td>3</td><td>6</td><td>7</td><td>1</td><td>|46.0(−11.3)|36.7(-5.3)</td></tr><tr><td>ECHO w/o pretrained encoder, foresight</td><td>V</td><td></td><td>√</td><td></td><td>19</td><td></td><td>23</td><td>20</td><td>21</td><td>8</td><td></td><td></td><td>6</td><td>0</td><td>56.0(−1.3)|40.0(−2.0)</td></tr><tr><td>ECHO w/o foresight</td><td>√</td><td></td><td>√</td><td></td><td>22</td><td>18</td><td>24</td><td>16</td><td>21</td><td>10</td><td>8 6</td><td>0 2</td><td>12</td><td>3</td><td>65.3(+8.0) 42.7 (+0.7)</td></tr><tr><td>ECHO w/o context</td><td>√</td><td></td><td></td><td>r</td><td>22</td><td>18</td><td>25</td><td>14</td><td>22</td><td>5</td><td>7</td><td>5</td><td>1</td><td>2</td><td>54.0(−3.3)44.0(+2.0)</td></tr><tr><td>ECHO w/o event context</td><td>√</td><td></td><td>√</td><td>√</td><td>21</td><td></td><td>23</td><td>22</td><td>22</td><td>13</td><td>10</td><td>3</td><td>3</td><td>1</td><td>60.7(+3.4)|48.7(+6.7)</td></tr><tr><td>ECHO (full)</td><td>√</td><td></td><td></td><td>√</td><td></td><td>24</td><td>25</td><td>23</td><td>21</td><td>17</td><td>6 11</td><td></td><td>8</td><td>1</td><td>69.3(+12.0) 53.3 (+11.3)</td></tr></table>

TABLE II

REAL-ROBOT SUCCESS RATE (%) ACROSS TASKS UNDER NORMAL AND SEVERE DARK LIGHTING (20 TRIALS PER CELL). RED PARENTHESIZED  
DELTAS ARE RELATIVE TO THE π<sub>0</sub> + EVENT BASELINE.
<table><tr><td></td><td></td><td colspan="2">Normal Lighting</td><td colspan="2">Severe Dark Lighting</td></tr><tr><td>Method</td><td>Perception</td><td>pick-place</td><td>bell-ring</td><td>pick-place</td><td>bell-ring</td></tr><tr><td>π0</td><td>wrist RGB</td><td>80 (+5)</td><td>15 (−15)</td><td>55 (-5)</td><td>5 (-10)</td></tr><tr><td></td><td>π0 + event wrist RGB + event</td><td>75 (+0)</td><td>30 (+0)</td><td>60 (+0)</td><td>15 (+0)</td></tr><tr><td>ECHO</td><td>wrist RGB + event 90 (+15)</td><td></td><td>45 (+15)</td><td>80 (+20)</td><td>30 (+15)</td></tr></table>

Ring the Bell Twice  
![](images/7b7b213fcc34c694670e4e889bd2585d2f8a4c7b3998987d514ca16f2dfd52c9.jpg)

Put the Toy Bee into the Plate  
![](images/2d8766576f43b593ceb1af92a59d3bb56f33a3018ff8909e2d3297a1b09cdf9a.jpg)  
Fig. 7. Real-robot setup for the tasks Ring the Bell Twice and Put the Toy Bee into the plate.

## D. Real-robot deployment

Experimental setups. The real-world setup uses an Astribot S1 [41] (Fig. 7). The policy is deployed on an RTX 3090. Visual and event observations are collected from a wristmounted DVSync Event camera [39]. We evaluate two tasks, Put the Toy Bee into the Plate and Ring the Bell twice. Each task provides 100 demonstrations, 75 recorded under normal lighting and 25 under severe dark lighting.

Each policy is run for 20 trials per task, under normal lighting at 286 lux and dark lighting below 10 lux, set by a dimmable lamp above the workspace.

In closed loop the policy is queried every H=16 RGB intervals and executes the returned chunk. The foresight queries and the context block are updated at this frequency. The foresight window is the event stream of the H intervals the chunk spans, and the context block is updated once per step from the end-effector history since the episode start. The observed event latent is recomputed per RGB interval, encoding the last 16 segments, each represented as a six-bin volume (Sec. IV-B).

![](images/436e107df1116edac55ef20be1cf6d16413e05ab5a8956d4ae601c3d3a6956c7.jpg)  
Fig. 8. Real-robot execution of ring the bell twice under normal and dark lighting. Wrist RGB (top) and events from the same interval (bottom) show contact between the gripper and bell.

## Results and Analysis.

1) Normal Lighting. Across both real-robot tasks, ECHO consistently outperforms the wrist-only RGB baseline (Table II). The performance gap is most pronounced in bell-ring (45% vs. 15%), where executing repeated strikes requires keeping track of the bell after the wrist-mounted camera moves past it. The spatial trajectory memory in ECHO keeps the target location addressable even when it leaves the camera frame.

2) Severe Dark Lighting. As shown in Fig. 8, extreme low light severely corrupts RGB inputs, whereas the synchronous event stream robustly captures temporal brightness changes during physical interaction. Quantitatively, ECHO maintains strong reliability: on pick-place, its performance margin over the baseline widens from +10% to +25% (80% vs. 55%, Table II). Notably, simply appending instant event features (π<sub>0</sub> + event) yields inconsistent results across tasks, confirming that raw event streams require explicit trajectory context and predictive grounding to handle lighting and view shifts effectively.

## VI. CONCLUSIONS

In this work, we introduce ECHO, an event-based framework for robot manipulation under challenging environments. ECHO leverages a pretrained event feature encoder to embed wrist-mounted event volumes into representation vectors that explicitly capture event-motion dynamics in a latent-wm style. Furthermore, its hindsight block injects gripper trajectories with event as off-camera context, while the outlook block optimizes foresight queries to predict future event horizons for action generation. Extensive evaluations on RLBench and two real-robot tasks demonstrate that ECHO consistently outperforms RGB-only baselines across both standard and severely degraded lighting conditions.

## ACKNOWLEDGMENT

Codex [42] with ChatGPT (GPT-5.6 [43]) was used to assist with coding in this project, and the robot-arm illustration in Fig. 2 was generated with GPT Image 2 [44]. All other content was created by the authors without generative models.

## REFERENCES

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, et al., “RT-2: Visionlanguage-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.

[2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, et al., “Open-VLA: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[3] Q. Li, Y. Liang, Z. Wang, et al., “CogACT: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation,” arXiv preprint arXiv:2411.19650, 2024.

[4] K. Black, N. Brown, D. Driess, et al., “π<sub>0</sub>: A vision-languageaction flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[5] Physical Intelligence, “π<sub>0.5</sub>: A vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[6] G. Gallego, T. Delbruck, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, ¨ S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis, and D. Scaramuzza, “Event-based vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 154–180, 2020.

[7] J. Zhai, H. Shi, S. Guo, K. Yang, and K. Wang, “E-VLA: Eventaugmented vision-language-action model for dark and blurred scenes,” arXiv preprint arXiv:2604.04834, 2026.

[8] J. Liu, X. Xu, Z. Zhang, H. Wang, R. Chen, et al., “Event-VLA: Action-conditioned event fusion for robust vision-language-action model,” arXiv preprint arXiv:2606.29384, 2026.

[9] Y. Du, M. Yang, B. Dai, H. Dai, O. Nachum, J. B. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via text-guided video generation,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[10] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pre-training for visual robot manipulation,” in International Conference on Learning Representations (ICLR), 2024.

[11] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, et al., “World-VLA: Towards autoregressive action world model,” arXiv preprint arXiv:2506.21539, 2025.

[12] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-World: A controllable generative world model for robot manipulation,” arXiv preprint arXiv:2510.10125, 2025.

[13] J. Bruce, M. D. Dennis, A. Edwards, et al., “Genie: Generative interactive environments,” in International Conference on Machine Learning (ICML), 2024.

[14] D. Schmidt and M. Jiang, “Learning to act without actions,” in International Conference on Learning Representations (ICLR), vol. 2024, 2024, pp. 9379–9395.

[15] S. Ye, J. Jang, B. Jeon, S. J. Joo, J. Yang, et al., “Latent action pretraining from videos,” in International Conference on Learning Representations (ICLR), 2025, arXiv:2410.11758.

[16] A. Liang, P. Czempin, M. M. Hong, Y. Zhou, J. Wang, et al., “CLAM: Continuous latent action models for robot learning from unlabeled demonstrations,” arXiv preprint arXiv:2505.04999, 2025.

[17] H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, et al., “Motus: A unified latent action world model,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, arXiv:2512.13030.

[18] Q. Garrido, T. Nagarajan, B. Terver, N. Ballas, Y. LeCun, and M. Rabbat, “Learning latent action world models in the wild,” in International Conference on Machine Learning (ICML), 2026.

[19] X. Chen, H. Wei, P. Zhang, C. Zhang, K. Wang, Y. Guo, R. Yang, Y. Wang, X. Xiao, L. Zhao, et al., “VILLA-X: Enhancing latent action modeling in vision-language-action models,” in International Conference on Learning Representations (ICLR), vol. 2026, 2026, pp. 70 673–70 703.

[20] J. Yang, Y. Shi, H. Zhu, M. Liu, K. Ma, Y. Wang, G. Wu, T. He, and L. Wang, “CoMo: Learning continuous latent motion from internet videos for scalable robot learning,” arXiv preprint arXiv:2505.17006, 2025.

[21] R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, et al., “FLARE: Robot learning with implicit world modeling,” arXiv preprint arXiv:2505.15659, 2025.

[22] J. Sun, W. Zhang, Z. Qi, S. Ren, Z. Liu, H. Zhu, G. Sun, X. Jin, and Z. Chen, “VLA-JEPA: Enhancing vision-language-action model with latent world model,” in European Conference on Computer Vision (ECCV). Springer, 2026, pp. 478–497.

[23] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, et al., “MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation,” in International Conference on Learning Representations (ICLR), 2026.

[24] R. Shah, Y. Li, F. Bello, Y. Zhu, and R. Mart´ın-Mart´ın, “Memory retrieval in visuomotor policies for long-horizon robot control,” in Robotics: Science and Systems (RSS), 2026.

[25] E. Cherepanov, N. Kachaev, D. Zelezetsky, A. Bulatov, A. Pshenitsyn, Y. Kuratov, A. Skrynnik, A. I. Panov, and A. K. Kovalev, “µVLA: On recurrent memory for partially observable manipulation in VLA models,” arXiv preprint arXiv:2606.12497, 2026.

[26] G. Zhao, L. Guo, Y. Mei, Z. Zhu, Y. Zhang, B. Cao, M. Yu, X. He, J. Jiang, and J. Liu, “AtlasVLA: Persistent world-ego state modeling for vision-language-action models,” arXiv preprint arXiv:2608.06729, 2026.

[27] Z. Zheng, J. Yu, X. Peng, J. Shi, M. Li, C. Zhang, W. Li, D. Wang, H. Lu, and X. Jia, “Mem-World: Memory-augmented actionconditioned world models for persistent robot manipulation,” arXiv preprint arXiv:2606.18960, 2026.

[28] N. J. Sanket, C. M. Parameshwara, et al., “EvDodgeNet: Deep dynamic obstacle dodging with event cameras,” in IEEE International Conference on Robotics and Automation (ICRA), 2020.

[29] K. Vinod, P. J. Ramesh, and B. Chakravarthi, “SEBVS: Synthetic event-based visual servoing for robot navigation and manipulation,” arXiv preprint arXiv:2508.17643, 2025.

[30] X. Huang, M. Halwani, R. Muthusamy, A. Ayyad, et al., “Realtime grasping strategies using event camera,” Journal of Intelligent Manufacturing, 2022.

[31] S. Vemprala, S. Mian, and A. Kapoor, “Representation learning for event-based visuomotor policies,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[32] D. Deniz, C. Fermuller, E. Ros, et al., “Event-based vision for early prediction of manipulation actions,” arXiv preprint arXiv:2307.14332, 2023.

[33] S. Liu, J. Li, G. Zhao, Y. Zhang, X. Meng, F. R. Yu, X. Ji, and M. Li, “EventGPT: Event stream understanding with multimodal large language models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 29 139–29 149.

[34] D. Sliwowski, S. Jadav, S. Stanovcic, J. Orbik, et al., “REASSEMBLE: A multimodal dataset for contact-rich robotic assembly and disassembly,” arXiv preprint arXiv:2502.05086, 2025.

[35] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations (ICLR), 2023, arXiv:2210.02747.

[36] H. Rebecq, D. Gehrig, and D. Scaramuzza, “ESIM: An open event camera simulator,” in Conference on Robot Learning (CoRL), 2018.

[37] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., “DI-NOv3,” arXiv preprint arXiv:2508.10104, 2025.

[38] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “RLBench: The robot learning benchmark & learning environment,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3019–3026, 2020.

[39] “DVSense,” https://www.dvsense.com/en/home en/, accessed: 2026- 09-15.

[40] J. Liu, H. Chen, P. An, Z. Liu, R. Zhang, C. Gu, X. Li, Z. Guo, S. Chen, M. Liu, et al., “HybridVLA: Collaborative diffusion and au-

toregression in a unified vision-language-action model,” arXiv preprint arXiv:2503.10631, 2025.

[41] Astribot, “Astribot s1: Ai robot platform,” https://www.astribot.com/ product, 2024, accessed: 2026-09-15.

[42] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021.

[43] OpenAI, “Gpt-5.6: Frontier intelligence that scales with your ambition,” https://openai.com/index/gpt-5-6/, July 2026.

[44] ——, “Introducing chatgpt images 2.0,” https://openai.com/index/ introducing-chatgpt-images-2-0/, April 2026.