# E-WAVE: Event-based Continuous Optical Flow via Warping-Aligned Visual Encoding

Jiale Wu<sup>1</sup>, Xiaoyang Bai<sup>2</sup>, Haoming Yu<sup>1</sup>, Yiwei Chen<sup>1</sup>, Yifan Peng<sup>2</sup>, Weiwei Xu<sup>1</sup>\*

![](images/a6f4531ceedc6a9e42a9a37cec7946c004798442dfcca1a1eac39c0e4bcaebf9.jpg)  
Figure 1: (a) We propose E-WAVE, an event-based continuous optical flow estimation method that leverages feature warping instead of correlation volumes to extract temporal and cross-modal coherence. b) Our head-mounted acquisition prototype, built to validate our model in real-world scenarios. c) E-WAVE operates at a higher resolution than existing methods such as BFlow, and it excels in challenging cases featuring nonlinear dynamics and motion blur. d) Our method is a promising candidate for modeling fast and continuous head, hand, and object motion and interaction in VR/AR applications.

## ABSTRACT

Temporally dense optical flow is essential for dynamic perception in immersive VR/AR systems, where rapid head, hand, and object motion must be continuously captured and tracked. Existing frame-based optical flow estimation methods are constrained by the tradeoff between temporal resolution and computational cost; while event cameras, with their high temporal resolution and energy efficiency, serve as a natural solution to the dilemma. However, eventbased approaches commonly rely on correlation volumes to capture pairwise voxel correspondences, which incur substantial memory and computation overhead. We present E-WAVE, a correlation-free framework for high-temporal-resolution (HTR) optical flow estimation from event streams. Instead of constructing all-pairs correlation volumes, E-WAVE employs global attention mechanism to model long-range feature dependencies and performs trajectoryguided feature warping using Bezier curve. Through iterative up-´ dates, it predicts trajectories that allow for querying at arbitrary timestamps without repeated inference. Experiments on MultiFlow and DSEC-Flow demonstrate a 25% lower trajectory error and comparable endpoint flow estimation accuracy relative to state-of-theart baselines. Additional evaluations on self-captured data using a head-mounted prototype validate that E-WAVE remains robust under challenging real-world conditions.

Index Terms: Event Camera, Optical Flow, Motion Perception

## 1 INTRODUCTION

As a crucial task in computer vision and graphics, optical flow estimation extracts motion vectors from 2D visual data and serves as an essential prior in a wide range of downstream tasks, such as 4D reconstruction [54], video deblurring [31], and frame interpolation [27]. Moreover, the flourishing of real-time robotics [4] and VR/AR [23] systems, which require accurate dynamic perception, also motivates the exploration of fast and robust optical flow estimation algorithms. However, conventional frame-based approaches such as RGB cameras encounter a tradeoff between hardware capacity and estimation quality. That is to say, temporally dense optical flow demands high frame rate inputs, increasing bandwidth requirements and computational loads, whereas lower frame rates can introduce large displacements and motion blur that degrade estimation accuracy. In immersive VR/AR applications, temporally dense motion estimates are essential to track fast, continuously evolving head, hand, and object interactions [1, 21, 9], and therefore resolving this bottleneck has become a pivotal yet unresolved challenge.

In recent years, event cameras have emerged as a compelling tool for optical flow estimation to complement the limitations of frame-based sensors. By capturing per-pixel intensity changes asynchronously, event sensors offer microsecond-level temporal resolution, continuous data acquisition, high dynamic range, and immunity to motion blur [10, 32]. Exploiting these attributes, prior works prove successful in both low-temporal-resolution (LTR, i.e., estimating only for RGB timestamps) [15] and high-temporalresolution (HTR, i.e., estimating for any arbitrary timestamp) [64] regimes, even with RGB frame-free, event-only setups. Nonetheless, state-of-the-art approaches often rely on correlation volumes to establish feature correspondences across time and between modalities, which incurs two downsides when applied to event streams. Firstly, all-pairs correlation scales quadratically with spatial feature size [48], as it constructs a 4D volume with quadratic memory usage for every pair of feature maps. Even with 1/8 spatial subsampling, the memory footprint and computation overhead can still be heavy when processing densely constructed event voxels. Secondly, correlation volumes only characterize pairwise correspondences and can fail to exploit long-range, continuous temporal trajectory inherently encoded by event streams.

An alternative solution to extract temporal and cross-modal coherence of features is through warping, rather than indexing correlation volumes. For optical flow estimation and dense point tracking on RGB inputs, representative methods such as WAFT [53] and CoWTracker [28] establish an effective and scalable pipeline that pairs feature warping with iterative refinement and replaces allpairs correlation with attention-driven feature alignment. On the other hand, warping-based contrast maximization (CM) has also long been leveraged across diverse event-based vision tasks, such as depth estimation [11] and feature tracking [12]. As an early exploration of applying warping to event-based optical flow, ID-Net [56] introduces a correlation-free framework that predicts optical flow through iterative event deblurring, validating the feasibility of leveraging feature warping, albeit limited by its underlying linear motion assumption.

Motivated by these attempts, we propose E-WAVE, an eventbased, correlation-free framework that achieves HTR optical flow estimation through trajectory-guidedfeature warping. Specifically, we leverage Vision Transformer (ViT) [7] as a replacement for correlation volumes to capture long-range dependencies, and we advance standard feature warping by parameterizing continuous pixel trajectories as Bezier curves. This design enables E-WAVE to re-´ cover temporally coherent, nonlinear motion from event streams at arbitrary timestamps. Analogous to state-of-the-art architectures [53, 28], our pipeline adapts an iterative warp-and-update process for Bezier parameter prediction, which further extends its esti-´ mation accuracy beyond single-step inference.

Experimental results show that E-WAVE achieves state-of-theart performance when estimating HTR optical flow on the challenging MultiFlow dataset [16] and can be extended with multimodal RGB inputs, bidirectional feature matching, and cross-dataset finetuning, yielding substantial improvements on the LTR DSEC-Flow benchmark [14]. To evaluate real-world robustness, we build a headset prototype with stereo event-RGB cameras and test on selfcaptured data featuring challenging cases such as underexposure and non-linear motion. These field evaluations reveal that E-WAVE reliably handles complex motion patterns induced by egocentric capturing with typical head-mounted or hand-held devices, highlighting its strong potential for immersive VR/AR applications. In all, our main contributions are summarized as follows:

• We propose E-WAVE, a trajectory-guided warping framework for event-based HTR optical flow estimation, that achieves state-of-the-art performance on the challenging MultiFlow dataset while maintaining competitive LTR capability. Moreover, extended versions of our model achieve state-of-the-art on LTR DSEC-Flow benchmark.

• We design a unified, flexible feature construction, warping and updating pipeline for both event-only and event-RGB setups. Given this, one can easily customize the pipeline across diverse input modalities and motion dynamics.

• Besides extensive benchmark evaluations, we also validate on real-world sequences captured by a head-mounted acquisition prototype, demonstrating its robustness and applicability to VR/AR scenarios.

## 2 RELATED WORK

## 2.1 Optical Flow Estimation

Deep learning methods for optical flow estimation has witnessed great progress in recent years since the pioneering FlowNet [8]. Follow-up works such as PWC-Net [46] improve estimation quality with components including pyramid processing, warping, and correlation-volume construction. The RAFT series [48, 55] introduce iterative refinement and demonstrate superior performance and efficiency, as well as the effectiveness of correlation volumes on handling large displacements. To avoid quadratic computational complexity of 4D correlation volumes, subsequent works adopt partial correlation volume [36] or global attention mechanism [58]. Most recently, WAFT [53] further combines warping with attention mechanism to estimate optical flow at a higher resolution, as will be discussed in detail below.

## 2.2 Event-based Optical Flow Estimation

Early approaches exploit event signals with hand-crafted heuristics [2, 3], typically under restrictive assumptions that limit their applicability to real-world scenarios [5, 65]. Later, the principle of Contrast Maximization is introduced to tackle various applications [11], optical flow estimation included. This formulation better reflects intrinsic characteristics of events by aligning moving edges to one or multiple timestamps via warping [44], Alternative works design specialized loss functions and supervise learningbased methods [17, 19, 25], but they can struggle with the inherently sparse and noisy nature of event signals [43, 20].

The availability of large-scale datasets [29], built through simulation [30, 26] or dedicated acquisition [6], has facilitated recent data-driven methods. The pioneering E-RAFT [15], inspired by RAFT [48], introduces iterative refinement with correlation volumes, supported by discrete event voxels [66]. To further capture intermediate motion, TMA [34] constructs multiple correlation volumes within each interval, a strategy later adopted in RGBor stereo-augmented approaches [63, 62]. BAT [57] extends this idea in a bidirectional manner, achieving remarkable results. However, the quadratically scaling memory usage of correlation volumes severely limits spatial resolution. To this end, IDNet [56] alternates the process with iterative deblurring, while EDC-Flow [33] combines feature differences on higher resolution with correlation volumes for improved estimation.

In HTR optical flow estimation, DCEIFlow [49] constructs pseudo-next-frame features for dense correlation volumes to predict intermediate optical flow. BFlow [16] replaces linear motion assumptions with correlation sampling guided by predicted Bezier´ curves [41]. ResFlow [64] and STSC-Flow [25] incorporate extra supervision to learn HTR trajectories with LTR ground truth. However, scalability remains a fundamental limitation: these approaches must subsample temporally due to the burden brought by correlation volumes. Similar challenges arise in event-based point tracking tasks [50]. Alternatively, EVA-Flow [61] explores a correlation-free framework to predict instant velocity, and then predict final optical flow via accumulated velocities, though drifting error progressively degrades performance on longer sequences.

In summary, although correlation volumes empower eventbased optical flow estimation with higher accuracy, their memoryexpensive nature systematically limits the scalability of relevant techniques towards higher resolution, both spatially and temporally.

## 2.3 Feature Alignment with Warping

Warping has been widely used by classical and early learningbased optical flow estimation methods across RGB and event inputs [40, 38], where it is considered as a tool for photometric brightness alignment or contrast maximization [37]. Following PWC-Net [46], SCFlow [24] warps feature in a pyramidal style and construct correlation volumes after warping. This reveals that aligned features can serve as tokens for feature matching. Recently, WAFT [53] further combines iterative warping with global attention mechanism for effective optical flow estimation, and CoW-Tracker [28] similarly utilizes attention mechanism to track spatially large motion. Additionally, SWIFT [52] combines warping with correlation volumes in lower resolution for better efficiency.

![](images/85401eea6375f423e7eb65fda4d22b9f4307041c704ab05003814fce9f6714d2.jpg)  
Figure $_ { 2 : }$ Correlation-based vs. Warping-based matching. (a) Correlation volumes store dense pixel-wise similarity between two feature frames, which is queried at a later stage to update estimated trajectories. (b) Our method leverages global attention to capture long-range correlation and directly derive warping parameters.

Existing warping-based approaches on event cameras [56, 61] often assume short temporal interval and constant motion for input data, but these assumptions may not hold for real-world applications on immersive systems, where accelerated, articulated, or otherwise nonlinear motion [18, 29] commonly occurs. In such cases, endpoint flow prediction with linear warping may fail to align motion between frames [41, 16], whereas how to unleash warping without the linear-motion assumption for optical flow estimation is still under-explored. To address this limitation, we extend warping to continuous event-guided trajectory represented as Bezier dis-´ placement fields, which allows warping to account for nonlinear motion in event-based scenarios.

## 3 E-WAVE METHOD

## 3.1 Event Camera and Event Voxelization

Unlike conventional sensory, event cameras asynchronously trigger events in response to changes in logarithmic brightness $L = \log I ,$ producing a sparse and variable-length event stream instead of regularly sampled image tensors [65]. An event activation is fired when $\Delta L \approx p C$ , where C is the contrast threshold and p is the event polarity (+1 or -1). Under the brightness-constancy assumption, local event rate r is determined jointly by the spatial intensity gradient ∇L and image motion v:

$$
^ { 6 , }\tag{1}
$$

and therefore event signal serves as a good motion indicator.

However, directly tensorizing raw events leads to inconsistent input dimensions and large memory overhead. The conventional practice is thus to convert event streams into compact, fixeddimensional voxel representations [63]. Specifically, given bin width $\Delta t$ and a set of anchor timestamps $\{ \tau _ { n } \} , n \in { 0 , 1 , \dots , K }$ between two adjacent RGB frames, we define the spatiotemporal interpolation weight for an event $\boldsymbol { e } _ { i } = ( x _ { i } , y _ { i } , t _ { i } , p _ { i } )$ at pixel $( x , y )$ as:

$$
\begin{array} { l } { { w _ { n } ( x , y ) = b ( x _ { i } - x ) b ( y _ { i } - y ) b \bigg ( \displaystyle \frac { t _ { i } - \tau _ { n } } { \Delta t } \bigg ) , } } \\ { { \mathrm { a n d } ~ b ( u ) = \operatorname* { m a x } ( 0 , 1 - | u | ) . } } \end{array}\tag{2}
$$

Then we accumulate the contribution of all events at each pixel, for positive and negative activations separately:

$$
V _ { n } ^ { s } ( x , y ) = \sum _ { i : p _ { i } = s } w _ { n } ( x , y ) , \qquad s \in \{ + 1 , - 1 \} .\tag{3}
$$

To align with the dimension of RGB images, we construct threechannel event voxels by additionally computing the difference between two polarity channels after clipping and normalization:

$$
{ \bf V } _ { n } = \left[ { V } _ { n } ^ { + } , { V } _ { n } ^ { - } , { V } _ { n } ^ { + } - { V } _ { n } ^ { - } \right] \in \mathbb { R } ^ { 3 \times H \times W } .\tag{4}
$$

## 3.2 Problem Definition and Feature Matching

Given the constructed event voxels $\{ \mathbf { V } _ { n } \} _ { n = 0 } ^ { K }$ between two boundary timestamps (endpoints), $\tau _ { 0 }$ and $\tau _ { K } .$ , and optionally the corresponding RGB frames $\mathbf { I } _ { 0 }$ and ${ \bf I } _ { 1 }$ , low-temporal-resolution $( L T R )$ optical flow estimation aims to retrieve a pixelwise flow map $\Psi$ from $\tau _ { 0 }$ to $\tau _ { K } ~ ( i . e .$ , from I<sub>0</sub> to $\mathbf { I } _ { 1 } ) _ { \cdot }$ , while high-temporal-resolution $( H T R )$ estimation is able to compute flow map $\Psi _ { t }$ corresponding to any intermediate timestamp $t \in [ \tau _ { 0 } , \tau _ { K } ]$

As scene motion inevitably causes spatial misalignment between voxels (and frames), a crucial step in optical flow estimation is feature matching. Given feature maps $\mathbf { F } ^ { a }$ and $\mathbf { F } ^ { b }$ , the classic correlation-based approach constructs a correlation volume $\mathbf { C } ^ { a , b }$ that stores similarities over candidate displacements ${ \mathcal { D } } \mathbf { : }$

$$
\mathbf { C } ^ { a , b } ( \mathbf { x , d } ) = \left. \mathbf { F } ^ { a } ( \mathbf { x } ) , \mathbf { F } ^ { b } ( \mathbf { x + d } ) \right. , \quad \mathbf { d } \in \mathcal { D } .\tag{5}
$$

In subsequent steps, $\mathbf { C } ^ { a , b }$ is queried by estimated flow maps to retrieve local correspondence information along each pixel trajectory. Alternatively, warping computes a per-pixel displacement matrix $\mathbf { U } ^ { a , b }$ and aligns two features using grid sampling:

$$
\widetilde { \mathbf { F } } ^ { b  a } ( \mathbf { x } ) : = \mathbf { F } ^ { b } ( \mathbf { x } + \mathbf { U } ^ { a , b } ( \mathbf { x } ) ) .\tag{6}
$$

Correlation relies on a separate matching hypothesis for each feature pair, whereas warping produces a single aligned feature map for a whole feature sequence. The latter design therefore allows for higher spatial and temporal resolution and subsequently leads to better performance, especially when operating on densely voxelized event data. See Fig. 2 for a visual comparison.

## 3.3 Warping-Aligned Visual Encoding

As illustrated in Fig. 3, our E-WAVE pipeline consists of three steps: (a) extracting multimodal features via RGB and event voxel encoders, (b) warping features with a parametrized displacement field, and (c) predicting residual updates using warped features with a global attention mechanism. Steps (b) and (c) are recursively applied for R iterations to yield the final output.

## 3.3.1 Feature extraction and fusion

To maintain architectural consistency across modalities, we employ the same ResNet [22] architecture for both RGB frame encoder $\Phi _ { \mathrm { r e s } } ^ { \mathrm { r g b } }$ and event voxel encoder $\Phi _ { \mathrm { r e s } } ^ { \mathrm { e v t } } .$ , which differ by only the last projection layer. We further complement RGB frame features by depth-aware features extracted from a frozen Depth Anything V2 [59] network, and then project them to the same dimension as event features via a standalone $1 \times 1$ convolution. The final perframe RGB features $\mathbf { R } _ { 0 } , \mathbf { R } _ { 1 }$ and per-voxel event features $\mathbf { F } _ { j }$ , are formally written as:

![](images/323cc6d1c2dfc3f7eca57a365682150879a6533ac8ca59382a3866af087ff593.jpg)  
Figure 3: Pipeline of E-WAVE. We firstly extract features from voxelized events and RGB frames (if available), and then repeat the refinement step by R times. At each iteration, visual features are warped using the predicted trajectory map from the previous iteration (purple), concatenated with the trajectory map and the hidden state (cyan), and processed by a ViT-DPT module to obtain an updated hidden state. Ultimately, two convolution heads convert the hidden state into the updated trajectory map and uncertainty vector ρ, respectively. An NLL loss on endpoints and a trajectory loss on intermediate ground truth are computed for each iteration to train the entire framework.

$$
\begin{array} { r l } & { \mathbf { R } _ { i } = \mathrm { C o n v } _ { 1 \times 1 } \left( \left[ \Phi _ { \mathrm { D A v 2 } } ( \mathbf { I } _ { i } ) , \Phi _ { \mathrm { r e s } } ^ { \mathrm { r g b } } ( \mathbf { I } _ { i } ) \right] \right) , \quad i \in \{ 0 , 1 \} , } \\ & { \mathbf { F } _ { j } = \Phi _ { \mathrm { r e s } } ^ { \mathrm { e v t } } ( \mathbf { V } _ { j } ) , \qquad \quad j = 0 , \dots , K . } \end{array}\tag{7}
$$

We also apply a shared 2D positional encoding to each event feature map before feature warping.

## 3.3.2 Trajectory parameterization

Throughout the estimation process, we maintain a parametrized displacement field U, where $\bar { \mathbf { U } } ( \mathbf { x } , t )$ denote the displacement of pixel x at time $t \in [ \tau _ { 0 } , \tau _ { K } ]$ relative to $\tau _ { 0 } . \mathrm { A t }$ each pixel location x (henceforth omitted for clarity), $\mathbf { u } ( t ) : = \mathbf { U } ( \mathbf { x } , t )$ is represented as a degree-L Bezier curve with anchors´ $\mathbf { p } \in \mathbb { R } ^ { L }$ . Formally:

$$
\mathbf { u } ( t ) = \sum _ { \ell = 0 } ^ { L } \beta _ { \ell } ^ { L } ( \hat { t } ) \mathbf { p } _ { \ell } , \qquad \beta _ { \ell } ^ { L } ( \hat { t } ) = \binom { L } { \ell } ( 1 - \hat { t } ) ^ { L - \ell } \hat { t } ^ { \ell } .\tag{8}
$$

Here tˆ is normalized time defined as $\hat { t } = ( t - \tau _ { 0 } ) / ( \tau _ { K } - \tau _ { 0 } )$

Since the trajectory represents displacement relative to $\tau _ { 0 } ,$ we set $\mathbf { p } _ { 0 } = 0$ , which ensures that $\mathbf { u } ( \tau _ { 0 } ) = \bar { 0 }$ . Instead of directly predicting p, our network equivalently outputs θ as differences between adjacent control points, that is,

$$
{ \bf p } _ { \ell } = \sum _ { q = 1 } ^ { \ell } \theta _ { q } , \qquad \ell = 1 , \ldots , L .\tag{9}
$$

We use Θ to denote the collection of $\theta \mathrm { { ^ { \circ } s } }$ over all pixel locations.

## 3.3.3 Feature warping and residual update

At the r-th iteration, we warp $\{ \mathbf { F } _ { j } \} , \mathbf { R } _ { 0 }$ and $\mathbf { R } _ { 1 }$ (if available) using the up-to-date $\mathbf { U } ^ { ( r ) }$ to obtain aligned features:

$$
\widetilde { \mathbf { F } } _ { j } ^ { ( r ) } \gets \mathrm { w a r p } ( \mathbf { F } _ { j } , \mathbf { U } ^ { ( r ) } ( \cdot , \tau _ { j } ) ) , \mathrm { ~ a n d ~ } \widetilde { \mathbf { R } } _ { 1 } ^ { ( r ) } \gets \mathrm { w a r p } ( \mathbf { R } _ { 1 } , \mathbf { U } ^ { ( r ) } ( \cdot , \tau _ { K } ) ) .\tag{10}
$$

Algorithm 1 Iterative trajectory refinement.   
Require: Event voxel features $\{ \mathbf { F } _ { j } \} _ { j = 0 } ^ { K } ;$ RGB image features   
$\mathbf { R } _ { 0 } , \mathbf { R } _ { 1 } ;$ initial state $\mathbf { h } ^ { ( 0 ) }$ ; number of iterations R.   
1: $\mathbf { U } ^ { ( 0 ) }  \mathbf { 0 }$   
2: for $r = 0 , \ldots , R - 1 { \bf d o }$   
3: $\widetilde { \mathbf { F } } _ { i } ^ { ( r ) } \gets \mathbf { F } _ { j } , \mathbf { U } ^ { ( r ) } ; \widetilde { \mathbf { R } } _ { 1 } ^ { ( r ) } \gets \mathbf { R } _ { 1 } , \mathbf { U } ^ { ( r ) }$ (Eq. 10)   
4: z<sup>(r)</sup> ← Fe<sup>(r)</sup>, . . . , Fe<sup>(r)</sup>, R<sub>0</sub>, Re <sup>(r)</sup>, h<sup>(r)</sup>, sg[Θ<sup>(r)</sup>] (Eq. 11)   
5: h<sup>(r+1)</sup> ← T (z<sup>(r)</sup>), h<sup>(r)</sup> (Eq. 12)   
6: δΘ<sup>(r)</sup> ← h<sup>(r+1)</sup> (Eq. 13)   
U<sup>(r+1)</sup> ← B(δΘ<sup>(r)</sup>) (Eq. 14 & Sec. 3.3.2)   
8: end for   
9: return $\boldsymbol { \Psi } _ { t } : = \mathbf { U } ^ { ( R ) }$

Here warp(·) is defined as applying Eq. 6 for each pixel. We initialize ${ \bf U } ^ { 0 }$ to be all-zeros, and subsequent ${ \bf U } ^ { ( r ) } { \bf s }$ are parametrized by estimated control parameters $\Theta ^ { ( r ) } \in \mathbb { R } ^ { L \times H \times W }$ , as detailed above.

Then, we assemble the latent feature map z from aligned event features ${ \widetilde { \mathbf { F } } } _ { j } ,$ reference image features $\mathbf { R } _ { 0 }$ and $\widetilde { \mathbf { R } } _ { 1 }$ , hidden recurrent state h, and current control parameters Θ using $1 \times 1$ convolution:

$$
\mathbf { z } ^ { ( r ) } = \mathrm { C o n v } _ { 1 \times 1 } \left( \left[ \widetilde { \mathbf { F } } _ { 0 } ^ { ( r ) } , \ldots , \widetilde { \mathbf { F } } _ { K } ^ { ( r ) } , \mathbf { R } _ { 0 } , \widetilde { \mathbf { R } } _ { 1 } ^ { ( r ) } , \mathbf { h } ^ { ( r ) } , \mathbf { s } \mathbf { g } [ \Theta ^ { ( r ) } ] \right] \right) .\tag{11}
$$

$\mathbf { R } _ { 0 }$ and $\widetilde { \mathbf { R } } _ { 1 }$ are set to zero if RGB inputs are unavailable. Next, we pass z through a ViT-DPT [39] refinement network $T _ { \mathrm { r e f } }$ , which embeds z using an $8 \times 8$ patch projection, adds an interpolated learnable positional embedding, and decodes intermediate transformer features into a dense output on the spatial grid, that is, $\mathbf { y } ^ { ( r ) } = T _ { \mathrm { r e f } } ( \mathbf { z } ^ { ( r ) } )$ . We then update the hidden state by:

$$
\begin{array} { r } { \mathbf { h } ^ { ( r + 1 ) } = \mathbf { C } \mathsf { o n v } _ { 1 \times 1 } \left( \left[ \mathbf { y } ^ { ( r ) } , \mathbf { h } ^ { ( r ) } \right] \right) . } \end{array}\tag{12}
$$

Finally, we use a prediction head to output increments of control parameters and an uncertainty head for NLL loss computation, each

Iteration 1

![](images/46a56b7e15527433c23d0bc9cad4dc7fe4255c62439ccb522d05ed3b83d84814.jpg)

![](images/df66b0d26e1aa97a2737b673d6f9f50999d6213c265bfa88d2cfb3d1dafe20cb.jpg)  
Iteration 2

![](images/76072d756f88ac6e2ce931b4f1221e0edfc8abf01d45876eaa202247b8a32885.jpg)  
Iteration 3

![](images/cfe3d190abf83fa64ecfbb9934aede274f3545036e9e4907a80d73107ab737e8.jpg)  
Iteration 4

Figure 4: Visualization of estimated optical flow and predicted (red) vs. ground truth (blue) trajectories across different iterations. The warp-andupdate process improves both pixelwise trajectory alignment and object-level flow coherence in the first few iterations and converges afterwards.

with two convolution layers:

$$
\begin{array} { r } { \delta \Theta ^ { ( r ) } = \mathrm { C o n v } _ { 1 \times 1 } \left[ \mathrm { R e L U } \left( \mathrm { C o n v } _ { 3 \times 3 } ( \mathbf { h } ^ { ( r + 1 ) } ) \right) \right] , } \\ { \rho ^ { ( r ) } = \mathrm { C o n v } _ { 1 \times 1 } \left[ \mathrm { R e L U } \left( \mathrm { C o n v } _ { 3 \times 3 } ( \mathbf { h } ^ { ( r + 1 ) } ) \right) \right] . } \end{array}\tag{13}
$$

Refer to Sec. 3.5 for more details on NLL loss. With the estimated $\delta \Theta ^ { ( r ) }$ , we may then update the control parameters as:

$$
\Theta ^ { ( r + 1 ) } = \mathrm { s g } [ \Theta ^ { ( r ) } ] + \delta \Theta ^ { ( r ) } ,\tag{14}
$$

and proceed into the next iteration. Algorithm 1 summarizes the complete iterative refinement procedure. We use B(δΘ) to denote the composition of Eq. 14 and Bezier curve construction in´ Sec. 3.3.2. Note that we operate on ${ \mathrm { ~  ~ \lambda ~ } } ^ { \mathrm { ~ 2 ~ } \times 2 }$ downsampled scale to reduce memory and time consumption, and a learned convex 3 × 3 kernel is used to upsample $\Theta ^ { R }$ back to the input resolution before outputting it as $\Psi _ { t }$

## 3.4 Analysis on Iterative Warp-and-Update

E-WAVE’s alternating warp-and-update process allows it to model long-range correlations and dependencies that are hard to capture with one-step inference, even with global attention. To further analyze its optimization dynamics, observe that Eq. 6, Eq. 8, Eq. 9 and Eq. 14 are all linear. Therefore, after r iterations, the trajectory maps and warped features can be written as:

$$
\begin{array} { r l } & { \mathbf { U } ^ { ( r + 1 ) } = \displaystyle \sum _ { i = 0 } ^ { r } B ( \delta \Theta ^ { ( i ) } ) = B ( \sum _ { i = 0 } ^ { r } \delta \Theta ^ { ( i ) } ) , } \\ & { \widetilde { \mathbf { F } } ^ { ( r + 1 ) } = \displaystyle \operatorname { w a r p } ( \mathbf { F } , \mathbf { U } ^ { ( r + 1 ) } ) = \operatorname { w a r p } ( \mathbf { F } , B ( \sum _ { i = 0 } ^ { r } \delta \Theta ^ { ( i ) } ) ) . } \end{array}\tag{15}
$$

That is, if we define $\begin{array} { r } { \bar { \Theta } : = \sum _ { i = 0 } ^ { r } \delta \Theta ^ { ( i ) } } \end{array}$ , then the first r iterations are equivalent to one single step of inference with the refinement network $T _ { \mathrm { r e f } }$ predicting Θ<sup>¯</sup> . In other words, increasing the number of iterations does not inherently cause the output to diverge.

However, this property does not guarantee iterative convergence, which by itself is hard to prove theoretically due to the nonconvex nature of $T _ { \mathrm { r e f } } .$ To empirically assess whether E-WAVE behaves as expected across iterations, we visualize its per-iteration outputs in Fig. 4. As shown in the figure, estimation quality improves significantly for the initial iterations and stabilizes afterwards, which supports our intuitive assumption that warp-and-update converges to a well-matched state.

## 3.5 Training Objectives

We supervise every refinement iteration with an endpoint uncertainty loss and, when temporally dense annotations are available, a trajectory loss. For the endpoint loss, we follow WAFT [53] and SEA-RAFT [55] to use a mixture-of-Laplace formulation. At iteration r, the prediction head (Eq. 13) maps the refined hidden state to $\pmb { \rho } \in \dot { \mathbb { R } } ^ { H \times \dot { W } \times 4 }$ , which represents two mixture logits and two logscales per pixel. Denote the ground truth optical flow as U<sup>∗</sup>, the endpoint residual e at pixel x can be written as:

$$
e = { \mathbf { U } } ^ { * } ( { \mathbf { x } } , \tau _ { K } ) - { \mathbf { U } } ^ { ( r ) } ( { \mathbf { x } } , \tau _ { K } ) ,\tag{16}
$$

and we model e using a two-component Laplace mixture and minimize the negative log-likelihood of the ground-truth flow:

$$
p ( e ) = \alpha \frac { \exp ( - | e | ) } { 2 } + ( 1 - \alpha ) \frac { \exp ( - | e | / \exp ( \beta ) ) } { 2 \exp ( \beta ) } ,\tag{17}
$$

$$
\ell _ { \mathrm { N L L } } ( e ) = - \log p ( e ) ,\tag{18}
$$

where α and $\beta$ are calculated from $\rho ( \mathbf { x } )$ . The final loss $L _ { \mathrm { n l l } } ^ { ( r ) }$ averages $\ell _ { \mathrm { N L I } }$ over all valid pixels and two flow directions. This loss respects the behavior of the standard $L _ { 1 }$ objective while taking into account large uncertainty in ambiguous or heavily occluded regions. That is, unpredictable samples are less likely to dominate optimization, reducing overfitting in presence of ambiguous cases and improving cross-dataset generalizability.

For dense trajectory supervision, let $\Omega _ { q }$ be the set of valid pixels at normalized time $\tau _ { q } ,$ we evaluate the predicted Bezier curve using´ a vector Charbonnier penalty:

$$
L _ { \mathrm { t r a j } } ^ { ( r ) } = \frac { 1 } { \left| \mathcal { T } \right| } \sum _ { q \in \mathcal { T } } \frac { 1 } { \left| \Omega _ { q } \right| } \sum _ { \mathbf { x } \in \Omega _ { q } } \left( \sqrt { \left\| \mathbf { U } ^ { ( r ) } ( \mathbf { x } , \tau _ { q } ) - \mathbf { U } ^ { * } ( \mathbf { x } , \tau _ { q } ) \right\| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } } - \varepsilon \right) ,\tag{19}
$$

where $\mathcal { T } = \{ \boldsymbol { q } : | \Omega _ { d } | > 0 \}$ . Note that averaging each timestamp by $\Omega _ { q }$ prevents timestamps with more valid pixels from dominating loss calculation.

Finally, we exponentially weight both loss terms from all R refinement iterations to obtain the overall loss value:

$$
L = \sum _ { r = 1 } ^ { R } \gamma ^ { R - r } \left( L _ { \mathrm { n l l } } ^ { ( r ) } + \lambda _ { \mathrm { t r a j } } L _ { \mathrm { t r a j } } ^ { ( r ) } \right) .\tag{20}
$$

## 3.6 Extensions for Improved Performance

While E-WAVE adopts a simple baseline configuration by default, its warping-based design can be readily extended with techniques commonly used in existing optical-flow methods. Specifically, we extend its capacity by incorporating bidirectional feature matching and cross-dataset finetuning, which improve model performance without altering the core warping-based framework.

Bidirectional feature matching has been explored in both RGB-based [42] and event-based [57] approaches. Specifically, we extend E-WAVE to jointly predict the forward and backward endpoint flows $\mathbf { U } _ { \mathrm { f w d } }$ and $\mathbf { U _ { \mathrm { b w d } } }$ . They are then used to construct a degree-2 motion trajectory:

![](images/5a4bff685193a68437ca251cd3ab404d8e71deec654d91c87772b6bb5d2a1c9d.jpg)  
Figure 5: Qualitative comparison on DSEC-Flow [15]. E-WAVE produces more detailed estimation thanks to its higher spatial resolution. We cropped bottom 60px of the figure for the lack of supervision in this region.

$$
\mathbf { U } ( \tau ) = \frac { 1 } { 2 } \left( \mathbf { U } _ { \mathrm { b w d } } + \mathbf { U } _ { \mathrm { f w d } } \right) \tau ^ { 2 } + \frac { 1 } { 2 } \left( \mathbf { U } _ { \mathrm { f w d } } - \mathbf { U } _ { \mathrm { b w d } } \right) \tau ,\tag{21}
$$

where $\tau \in [ - 1 , 1 ]$ denotes the normalized timestamp. Event voxel features are warped along the resulting trajectory in both temporal directions, while the core encoding and refinement modules remain unchanged. The forward and backward flows are jointly optimized using the corresponding bidirectional ground truth available in datasets such as DSEC-Flow [15].

Cross-dataset finetuning follows the strategy adopted by ECDPT [50] and STFlow [63], where we firstly pretrain E-WAVE on a larger synthetic dataset (e.g. MultiFlow [16]) and then finetune it on real-world data (e.g. DSEC-Flow). This strategy complements the spatially sparse annotation of real-world datasets and mitigates overfitting observed in our experiments.

## 4 EXPERIMENTS AND RESULTS

## 4.1 Datasets

We conduct extensive experiments on DSEC-Flow [15, 14] and MultiFlow [16]. DSEC-Flow is a well-established event-RGB driving dataset with 10 Hz ground truth annotations. Although it lacks HTR optical flows and therefore can only be used for evaluating LTR performance, we still conduct relevant training and validation to quantitatively assess our model’s performance in real-world scenarios. Predictions on the held-out test set are submitted to the official online evaluation server for evaluation.

MultiFlow is a more challenging synthetic dataset featuring dynamic, nonlinear motion patterns. It serves as the primary benchmark for evaluation continuous motion trajectories estimated by HTR methods. Each sample from the dataset contains dense ground-truth trajectories from $\tau _ { 0 } = 4 0 0$ ms to $\tau _ { K } = 9 0 0$ ms, with reference RGB frame rendered at $\tau _ { 0 } = 4 0 0$ ms. Due to storage constraints, we train our model on a subset of 2,000 samples, approximately one-fifth of the full training set, while evaluating on the complete test set.

Table 1: Quantitative results on DSEC-Flow. Methods up to “Ours” are event-only, while STFlow, BFlow and “Ours w/ img” works on event-RGB inputs. “+ bidir.” and “+ finetuning” correspond to two extensions in Sec. 3.6, respectively.
<table><tr><td>Method</td><td>1PE</td><td>2PE</td><td>3PE</td><td>EPE</td><td>AE</td><td>HTR</td></tr><tr><td>E-RAFT</td><td>12.74</td><td>4.74</td><td>2.68</td><td>0.79</td><td>2.85</td><td></td></tr><tr><td>TMA</td><td>10.86</td><td>3.97</td><td>2.30</td><td>0.74</td><td>2.68</td><td></td></tr><tr><td>IDNet</td><td>10.07</td><td>3.50</td><td>2.04</td><td>0.72</td><td>2.72</td><td></td></tr><tr><td>ECDDP</td><td>8.89</td><td>3.20</td><td>1.96</td><td>0.70</td><td>2.58</td><td></td></tr><tr><td>BAT</td><td>7.54</td><td>2.84</td><td>1.74</td><td>0.65</td><td>2.43</td><td></td></tr><tr><td>ResFlow</td><td>11.22</td><td>4.24</td><td>2.50</td><td>0.75</td><td>2.73</td><td>V</td></tr><tr><td>Ours</td><td>8.82</td><td>3.33</td><td>2.07</td><td>0.69</td><td>2.52</td><td>√</td></tr><tr><td>+ bidir.</td><td>7.70</td><td>2.88</td><td>1.75</td><td>0.63</td><td>2.37</td><td>√</td></tr><tr><td>STFlow</td><td>7.93</td><td>2.61</td><td>1.45</td><td>0.63</td><td>2.29</td><td></td></tr><tr><td>BFlow</td><td>9.70</td><td>3.42</td><td>1.88</td><td>0.69</td><td>2.42</td><td>√</td></tr><tr><td>Ours w/ img</td><td>8.15</td><td>2.83</td><td>1.73</td><td>0.64</td><td>2.40</td><td>√</td></tr><tr><td>+ finetuning</td><td>7.26</td><td>2.54</td><td>1.56</td><td>0.62</td><td>2.24</td><td>√</td></tr></table>

## 4.2 Baselines

For DSEC-Flow, we compare against representative baseline methods on the official leaderboard, including E-RAFT [15], TMA [34], IDNet [56], ECDDP [60], BAT [57], ResFlow [64], BFlow [16], and STFlow [63]. We directly use their reported metric scores on the official benchmark website for comparison. For MultiFlow, we retrain a wide range of open-sourced methods by ourselves, covering different input modalities:

Event-only methods. IDNet, E-RAFT, TMA, BFlow, and ResFlow are chosen as baselines. IDNet, E-RAFT, and TMA are LTR methods that produce only endpoint flow estimates, and we compute their intermediate trajectories by linearly interpolating two endpoint predictions. For the two baseline HTR methods, BFlow and Res-Flow, as well as our E-WAVE, we use the same degree of freedom for trajectory representation. Specifically, we set the degree of Bezier curves to 10 in E-WAVE and BFlow, and train ResFlow to´ predict 10 dense flow fields at uniformly sampled timestamps too.

Event-RGB method. We compare against the state-of-the-art STFlow in this category. Likewise, we extend STFlow to predict degree-10 Bezier curves to align with other HTR methods.´

RGB-only method. We additionally train WAFT [53] on RGB

Table 2: Quantitative results on MultiFlow. STFlow<sup>†</sup> denotes STFlow model trained with Bezier curve motion representation.
<table><tr><td>Methods</td><td>Input</td><td>TEPE</td><td>TAE</td><td>EPE</td><td>AE</td><td>HTR</td></tr><tr><td>WAFT</td><td>i</td><td>6.20</td><td>17.85</td><td>3.85</td><td>3.36</td><td></td></tr><tr><td>IDNet</td><td>e</td><td>7.08</td><td>20.74</td><td>7.60</td><td>10.85</td><td></td></tr><tr><td>E-RAFT</td><td>e</td><td>6.21</td><td>18.74</td><td>4.35</td><td>5.7</td><td></td></tr><tr><td>TMA</td><td>e</td><td>6.02</td><td>18.24</td><td>3.91</td><td>4.84</td><td></td></tr><tr><td>BFlow</td><td>e</td><td>2.74</td><td>7.00</td><td>5.00</td><td>7.35</td><td>√</td></tr><tr><td>ResFlow</td><td>e</td><td>2.28</td><td>5.32</td><td>3.74</td><td>4.70</td><td>√</td></tr><tr><td>Ours</td><td>e</td><td>1.81</td><td>3.89</td><td>3.03</td><td>3.51</td><td>√</td></tr><tr><td>STFlow</td><td>e+i</td><td>5.16</td><td>16.67</td><td>1.65</td><td>1.99</td><td></td></tr><tr><td>STFlow†</td><td>e+i</td><td>1.65</td><td>4.86</td><td>1.98</td><td>2.56</td><td>√</td></tr><tr><td>BFlow</td><td>e+i</td><td>2.05</td><td>5.33</td><td>3.55</td><td>5.00</td><td>√</td></tr><tr><td>Ours</td><td>e+i</td><td>1.25</td><td>2.95</td><td>2.04</td><td>2.45</td><td>√</td></tr></table>

frames only for a more thorough comparison.

## 4.3 Metrics

We evaluate endpoint flow estimation (LTR) using end-point error (EPE) and angular error (AE). On DSEC-Flow, we additionally report N-pixel error (NPE), i.e., the percentage of valid pixels whose EPE exceeds N pixels, where $N \in \{ 1 , 2 , 3 \}$ , which are also included in the official benchmark metrics.

On MultiFlow, we further evaluate HTR pixel trajectories using trajectory end-point error (TEPE) and trajectory angular error (TAE), originally proposed by BFlow. TEPE extends EPE by averaging it over K trajectory timestamps:

$$
\mathrm { T E P E } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { E P E } \big ( \mathbf { f } _ { \mathrm { p r e d } } ( t _ { k } ) , \mathbf { f } _ { \mathrm { g t } } ( t _ { k } ) \big ) , \qquad K \geq 2 ,
$$

and TAE is calculated analogously using AE. Lower values indicate better performance for all metrics.

## 4.4 Implementation Details

MultiFlow We train our model for 100 epochs on two NVIDIA V100 GPUs with 32 GB of memory each. We use a batch size of 4 and iteration count $R = 5 .$ . We use the AdamW [35] optimizer with the OneCycle [45] scheduler, setting the maximum learning rate to $1 . 2 \times 1 { \dot { 0 } } ^ { - 4 }$ , weight decay to $1 0 ^ { - 4 }$ , and the iterative loss discount factor to $\gamma = 0 . 8 5$ . Gradients are clipped to a maximum norm of 1.0. The predicted variance for $L _ { \mathrm { { N L L } } }$ is constrained to [0,10]. During training, inputs are randomly cropped to $2 8 8 \times 3 8 4$ and augmented with random horizontal and vertical flips. The temporal resolution of input event voxels is set to $K = 2 0$ , corresponding to a temporal interval of 25ms per voxel.

DSEC-Flow We make the following changes on the MultiFlow configuration to accommodate for the characteristics of DSEC-Flow. Our models are trained for 100 epochs on a single NVIDIA RTX 5090 GPU with 32 GB of memory. We apply random horizontal and vertical flips without cropping, and use a batch size of 2, while the iterative loss discount factor is set to $\gamma = 0 . 7 5 .$ The predicted variance for L<sub>NLL</sub> is constrained to [0,3]. We choose $K = 8 .$ , which corresponds to a temporal interval of 12.5 ms per voxel. Finally and most importantly, we set the Bezier degree to´ $L = 2$ due to lack of HTR supervision in DSEC-Flow.

## 4.5 Experimental Results

DSEC-Flow Table 1 reports quantitative results on the DSEC-Flow benchmark. As shown, E-WAVE achieves comparable performance as previous state-of-art methods under both event only and event-RGB settings. Specifically, under the event-only setting, our method performs slightly better than the newest correlation-free method IDNet [56] and achieve a 8% improvement compared to current best HTR method ResFlow [64]. Under the event-RGB setting, our EPE is 7.2% lower than BFlow [16], thanks to our higher spatiotemporal resolution. Additionally, our extended models (“+ bidir.” and “+ finetuning”) achieve state-of-the-art performance among event-only and event-RGB models, respectively, demonstrating the scalability of our model.

Table 3: Ablation study results. “B” denotes warping with degree-10 Bezier curves, “L” denotes warping with linear trajectories, “no”´ denotes prediction without warping process
<table><tr><td colspan="2">Methods</td><td colspan="2">Trajectory</td><td colspan="2">Endpoint</td></tr><tr><td> $\underline { { L _ { \mathrm { t r a j } } } }$ </td><td> $\mathsf { W a r p }$ </td><td>TEPE</td><td>TAE</td><td>EPE</td><td>AE</td></tr><tr><td>√</td><td>B</td><td>1.81</td><td>3.89</td><td>3.03</td><td>3.51</td></tr><tr><td>X</td><td>B</td><td>7.62</td><td>24.60</td><td>4.87</td><td>5.34</td></tr><tr><td>√</td><td>L</td><td>4.64</td><td>14.64</td><td>6.84</td><td>7.07</td></tr><tr><td>X</td><td>L</td><td>6.27</td><td>18.32</td><td>4.70</td><td>5.14</td></tr><tr><td>√</td><td>no</td><td>3.21</td><td>6.96</td><td>5.58</td><td>6.51</td></tr></table>

Qualitative results in Fig. 5 further demonstrate how higher feature resolution benefits estimation. By comparing E-WAVE against BFlow [16] (event-only checkpoint) and IDNet [56], operating on 1/8 and 1/4 resolutions respectively, we show that optical flows predicted on lower resolutions tend to be more blurry and ambiguous, sometimes corrupted by patch-like artifacts resulting from the upsampling process.

MultiFlow MultiFlow serves as the main benchmark to evaluate HTR optical flow prediction. As shown in Table 2, our model surpasses other baselines on both trajectory and endpoint metrics under the unimodal setting, beating the best-performing Res-Flow [64] by 19% in EPE and 20% in TEPE. We observe that most previous methods relying on endpoint correlation can estimate endpoint optical flows reasonably well in the presence of complex and non-linear motions in this challenging dataset, but all LTR models fail significantly on HTR metrics. Notably, IDNet [56] exhibits the worst performance among all baselines, possibly because motion patterns in MultiFlow violates its linear motion assumption.

Under the multimodal setting, E-WAVE witnesses a solid performance boost with RGB frames as input. In addition to BFlow and STFlow, we also train STFlow with Bezier curve motion repre-´ sentation and Bezier-assisted correlation volumes similar to BFlow,´ which improves the original linear design by a large margin on trajectory metrics, for a more thorough comparison. As reported in Table 2, E-WAVE outperforms both versions of STFlow and BFlow in trajectory metrics, and it achieves similar endpoint accuracy to the Bezier-enhanced STFlow model. While the original STFlow ranks´ first in endpoint prediction, its trajectory estimation is way worse than other models. Finally, we train the RGB-only WAFT model to probe the influence of event information on scene dynamics modeling. We find that WAFT achieves competitive endpoint metrics without any motion cue between two frames, but it is still outperformed by our model, revealing that event streams carry critical information that helps to capture object motion more accurately.

Figure 6 visually compares E-WAVE against BFlow and STFlow<sup>†</sup> from three aspects: optical flow, pixel trajectory, and image of warped events (IWE). Specifically, IWEs are obtained by warping events to the reference timestamp τ<sub>0</sub> using predicted (or ground truth) continuous optical flows and integrating them together, which serves as an indicator of the alignment between flow maps and event streams. In all three sampled instances, our method achieves the best trajectory alignment.

## 4.6 Ablation Study

We conduct ablation experiments for warping policies and trajectory loss on MultiFlow following the event-only settings and report our results in Table 3. Additionally, we also investigate how model performance changes with spatial and temporal resolutions, which validates our claim that E-WAVE gains advantage by scaling up to higher spatiotemporal resolutions.

![](images/a902707f9be65b6f3dc790c5f8d941019a0ea632ea9eb7261f7befb5d97fad87.jpg)  
Figure 6: Qualitative results on MultiFlow. For each scene, the first row visualizes the RGB reference image, the ground truth optical flow, and estimated flows from three methods, overlaid with ground truth (blue) and estimated (red) trajectories and zoomed in to a representative region. The second row visualizes the integrated event frame, the ground truth IWE, and IWEs for each method, also zoomed in to the same region.

Table 4: Ablation study results on spatial and temporal resolutions. Spatial resolutions are written with respect to the original data resolution, and temporal resolution is denoted by the number of event voxels used for prediction.
<table><tr><td colspan="2">Resolution</td><td colspan="2">Trajectory</td><td colspan="2">Endpoint</td></tr><tr><td>Spatial</td><td>Temporal</td><td>TEPE</td><td>TAE</td><td>EPE</td><td>AE</td></tr><tr><td>1/2</td><td>20</td><td>1.81</td><td>3.89</td><td>3.03</td><td>3.51</td></tr><tr><td>1/4</td><td>20</td><td>2.02</td><td>4.73</td><td>3.39</td><td>4.28</td></tr><tr><td>1/8</td><td>20</td><td>2.46</td><td>6.09</td><td>3.91</td><td>4.79</td></tr><tr><td>1/2</td><td>20</td><td>1.81</td><td>3.89</td><td>3.03</td><td>3.51</td></tr><tr><td>1/2</td><td>10</td><td>2.05</td><td>4.46</td><td>3.39</td><td>3.98</td></tr><tr><td>1/2</td><td>5</td><td>2.22</td><td>4.65</td><td>3.71</td><td>4.18</td></tr></table>

Trajectory-based warping. We compare three different warping policies: degree-10 Bezier curve warping, linear warping (real- ´ ized by setting Bezier degree to 1), and no warping. We observe that´ linear warping policy performs even worse than the no warping policy on both trajectory and endpoint metrics, while both of them fall behind the Bezier policy, demonstrating the necessity of represent-´ ing pixel trajectory and conducting feature warping at a sufficiently high degree of freedom to model highly nonlinear motion.

Trajectory loss. We further ablate with the trajectory loss $L _ { \mathrm { t r a j } }$ by setting $\lambda _ { \mathrm { t r a j } }$ to 0. Results in Table 3 show that degree-10 Bezier´ curves are highly underdetermined without dense trajectory supervision, which results in a severe degradation of prediction accuracy, especially in TEPE and TAE. However, when the linear warping policy is in effect, removing the trajectory loss leads to better endpoint prediction since the extra supervision between endpoints serves only as a noise factor in this case.

![](images/0ffe7207336d5769ee108f3e60ea7048622d2d5c45688d68a62989f7a0f84cf4.jpg)  
Figure 7: Results on real-world acquisitions. From left to right: reference RGB frames (not used for inference), event voxels at the beginning and end of inference window, and estimated results by three methods. For scenarios 2&4, we additionally visualize pixel trajectories (blue) to intuitively compare motion estimation accuracy across models.

Spatial Resolution We adjust the patch size of the DPT head to realize spatial resolution tuning without drastically changing the number of parameters. For example, spatial resolution is set to 1/4 and 1/8 by choosing patch sizes 4 × 4 and 2 × 2, respectively. As shown in Table 4, performance consistently degrades as spatial resolution decreases, demonstrating the importance of maintaining a high spatial resolution to preserve fine-grained spatial information.

Temporal Resolution Temporal resolution is characterized by the number of event voxels used for prediction, which reflects how densely the event stream is sampled over time. For all experiments, we keep the temporal duration of each voxel fixed at 25 ms. In Table 4, we observe that increasing the voxel count from 5 to 20 consistently improves all metrics. This trend shows that a denser temporal representation provides a more comprehensive description of intermediate motion between RGB timestamps.

## 4.7 Real-world Assessment

To assess real-world applicability of different models, we conduct zero-shot qualitative evaluations on self-captured data from diverse indoor and outdoor scenes, where ground truth motion is unavailable. All models are event-only and trained on the MultiFlow dataset. This Sim2Real setting better aligns with actual use cases where fine-tuning on large-scale labeled data is often infeasible.

We construct a rigid head-mounted prototype comprising a Prophesee EVK4 event camera and an Orbbec Femto Bolt RGB-D camera. EVK4 records events at 1,280 × 720 resolution with µs timestamps, while our Femto Bolt capture pipeline provides viewpoint-aligned 1,920 × 1,080 RGB-D frames at 30 FPS. We use a hardware trigger to synchronize each RGB-D frame with the corresponding event stream segment. For visualization, we refine depth maps with the pretrained LingBot-Depth v0.5 model [47] and project RGB images onto the event camera viewpoint using depth maps and calibrated RGB-to-event geometry.

Figure 7 visualizes our results. Rows 1–2 are captured in bright outdoor scenarios with both camera egomotion and object motion. In the “streetview” scenario, E-WAVE produces clearer and more stable motion boundaries for the moving truck and pedestrians. In the “badminton” scenario, where fast-moving objects (right arm and racket) can lead to motion blur for the frame-based RGB camera, our method still performs stably and estimates trajectories that are more realistic and coherent than ResFlow and BFlow, both stateof-the-art HTR models.

Rows 3 feature indoor captures and challenge each method’s ability to track and recognize hand and small objects (e.g., key chain), which move along highly nonlinear patterns. Thanks to the high spatial resolution of E-WAVE, our model is able to make finegrained predictions such as distinct boundaries for fingers, while correlation volume-based methods are restricted by their designs and can only give fuzzy estimates.

Row 4 is captured under extreme low-light condition, where illumination is restricted solely to a smartphone flashlight. As shown in columns 1–3, the RGB camera fails to capture usable scene detail, whereas the event camera still records valuable visual information. Nevertheless, low-light event streams are sparser and more noisy, making optical flow estimation even more challenging. Columns 4–6 show that E-WAVE not only recovers object boundaries (an arm wiping the blackboard), but also infer plausible trajectories for pixels invisible to RGB cameras.

## 5 DISCUSSIONS AND CONCLUSION

Limitations and Future Work. Our work has several limitations. Firstly, it relies on dense event voxel encoding, whose computational cost scales with the input event frame rate. Future research could incorporate more light-weighted encoders [5] or sliding-window approaches [13]. Secondly, our model trains on ground truth annotations, but it is potentially generalizable to a hybrid-supervision framework, following STSC-Flow [25] and ResFlow [64]. Finally, real-world event acquisition suffers from various noise factors such as flickering [20], reflection [51], and shadow. These artifacts can degrade flow accuracy in unconstrained environments, indicating that targeted noise-robustness mechanisms are necessary prior to deployment in VR/AR devices.

Conclusion. We present E-WAVE, a correlation-free framework for high temporal resolution (HTR) optical flow estimation from event streams. By replacing all-pairs correlation with the attention mechanism with warping-aligned features, E-WAVE effectively exploits continuous temporal information inherent in event data while maintaining its scalability to higher spatiotemporal resolutions. Extensive evaluations on DSEC-Flow and MultiFlow show that E-WAVE achieves state-of-the-art performance in HTR motion estimation, as well as LTR estimation with minor modifications. Real-world assessments using a head-mounted prototype further confirm its robustness under severe underexposure as well as rapid, non-linear motion dynamics. These results present E-WAVE as a promising candidate for facilitating temporally dense, dynamic perception in immersive VR/AR systems.

## REFERENCES

[1] P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol, et al. Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7061–7071. IEEE, 2025. 1

[2] R. Benosman, S.-H. Ieng, C. Clercq, C. Bartolozzi, and M. Srinivasan. Asynchronous frameless event-based optical flow. Neural Netw., 27:32–37, Mar. 2012. doi: 10.1016/j.neunet.2011.11.001 2

[3] T. Brosch, S. Tschechne, and H. Neumann. On event-based optical flow detection. Frontiers in Neuroscience, Volume 9 - 2015, 2015. doi: 10.3389/fnins.2015.00137 2

[4] H. Chao, Y. Gu, and M. Napolitano. A survey of optical flow techniques for robotics navigation applications. Journal of Intelligent & Robotic Systems, 73(1):361–372, 2014. 1

[5] T. Dalgaty, T. Mesquida, D. Joubert, A. Sironi, P. Vivet, and C. Posch. Hugnet: Hemi-spherical update graph neural network applied to lowlatency event-based optical flow. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3953–3962. IEEE, 2023. 2, 9

[6] E. Delaney, T. Brophy, E. Ward, F. Collins, E. Jones, B. Deegan, and M. Glavin. Evaluating event-based vision sensing in rain and fog. IEEE Sensors Journal, 2025. 2

[7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv, abs/2010.11929, 2020. 2

[8] A. Dosovitskiy, P. Fischer, E. Ilg, P. Hausser, C. Hazirbas, V. Golkov,¨ P. v. d. Smagt, D. Cremers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In 2015 IEEE International Conference on Computer Vision (ICCV), pp. 2758–2766, 2015. doi: 10. 1109/ICCV.2015.316 2

[9] X. Feng, Y. Liu, and S. Wei. Livedeep: Online viewport prediction for live virtual reality streaming using lifelong deep learning. In 2020 IEEE Conference on Virtual Reality and 3D User Interfaces (VR), pp. 800–808, 2020. doi: 10.1109/VR46266.2020.00104 1

[10] G. Gallego, T. Delbruck, G. Orchard, C. Bartolozzi, B. Taba, A. Censi,¨ S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis, et al. Eventbased vision: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(1):154–180, 2020. 1

[11] G. Gallego, H. Rebecq, and D. Scaramuzza. A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3867–3876. IEEE, 2018. 2

[12] D. Gehrig, H. Rebecq, G. Gallego, and D. Scaramuzza. Asynchronous, photometric feature tracking using events and frames. In European Conference on Computer Vision, pp. 766–781. Springer, 2018. 2

[13] D. Gehrig and D. Scaramuzza. Low-latency automotive vision with event cameras. Nature, 629(8014):1034–1040, 2024. 9

[14] M. Gehrig, W. Aarents, D. Gehrig, and D. Scaramuzza. Dsec: A stereo event camera dataset for driving scenarios. IEEE Robotics and Automation Letters, 2021. doi: 10.1109/LRA.2021.3068942 2, 6

[15] M. Gehrig, M. Millhausler, D. Gehrig, and D. Scaramuzza. E-raft:¨ Dense optical flow from event cameras. In 2021 International Conference on 3D Vision (3DV), pp. 197–206. IEEE, 2021. 1, 2, 6

[16] M. Gehrig, M. Muglikar, and D. Scaramuzza. Dense continuous-time optical flow from event cameras. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(7):4736–4746, 2024. 2, 3, 6, 7

[17] J. Hagenaars, F. Paredes-Valles, and G. De Croon. Self-supervised´ learning of event-based optical flow with spiking neural networks. Advances in Neural Information Processing Systems, 34:7167–7179, 2021. 2

[18] F. Hamann, D. Gehrig, F. Febryanto, K. Daniilidis, and G. Gallego. Etap: Event-based tracking of any point. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27186–27196, 2025. doi: 10.1109/CVPR52734.2025.02532 3

[19] F. Hamann, Z. Wang, I. Asmanis, K. Chaney, G. Gallego, and K. Daniilidis. Motion-prior contrast maximization for dense continuous-time motion estimation. In European Conference on Computer Vision, pp. 18–37. Springer, 2024. 2

[20] J. Han, Y. Yang, Z. Zhan, B. Shi, and I. Sato. Edef-net: Spatiotemporal association network for flicker removal in event streams. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 229–237, 2025. 2, 9

[21] S. Han, B. Liu, R. Cabezas, C. D. Twigg, P. Zhang, J. Petkau, T.-H. Yu, C.-J. Tai, M. Akbay, Z. Wang, A. Nitzan, G. Dong, Y. Ye, L. Tao, C. Wan, and R. Wang. Megatrack: monochrome egocentric articulated hand-tracking for virtual reality. ACM Trans. Graph., 39(4), Aug. 2020. doi: 10.1145/3386569.3392452 1

[22] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016. doi: 10.1109/CVPR .2016.90 3

[23] A. Holynski and J. Kopf. Fast depth densification for occlusion-aware augmented reality. ACM Transactions on Graphics (ToG), 37(6):1–11, 2018. 1

[24] L. Hu, R. Zhao, Z. Ding, L. Ma, B. Shi, R. Xiong, and T. Huang. Optical flow estimation for spiking camera. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17823–17832. IEEE, 2022. 3

[25] R. Hu, S. Wu, W. Yang, and J. Wu. From contrast to consistency: Rethinking event-based continuous-time optical flow estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15125–15134, 2026. 2, 9

[26] Y. Hu, S.-C. Liu, and T. Delbruck. v2e: From video frames to realistic dvs events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1312–1321, 2021. doi: 10.1109/CVPRW53098.2021.00144 2

[27] T. Kim, Y. Chae, H.-K. Jang, and K.-J. Yoon. Event-based video frame interpolation with cross-modal asymmetric bidirectional motion fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18032–18042, 2023. 1

[28] Z. Lai, E. Insafutdinov, E. Sucar, and A. Vedaldi. Cowtracker: Tracking by warping instead of correlation. arXiv preprint arXiv:2602.04877, 2026. 2, 3

[29] Y. Li, Z. Huang, S. Chen, X. Shi, H. Li, H. Bao, Z. Cui, and G. Zhang. Blinkflow: A dataset to push the limits of event-based optical flow estimation. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 3881–3888. IEEE, 2023. 2, 3

[30] Z. Li, X. Bai, J. Lu, P. Shen, E. Y. Lam, and Y. Peng. Eventtracer: Fast path tracing-based event stream rendering. IEEE Transactions on Visualization and Computer Graphics, 32(8):7606–7618, 2026. doi: 10.1109/TVCG.2026.3701141 2

[31] J. Lin, Y. Cai, X. Hu, H. Wang, Y. Yan, X. Zou, H. Ding, Y. Zhang, R. Timofte, and L. Van Gool. Flow-guided sparse transformer for video deblurring. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, eds., Proceedings of the 39th International Conference on Machine Learning, vol. 162 of Proceedings of Machine Learning Research, pp. 13334–13343. PMLR, 17–23 Jul 2022. 1

[32] S. Lin, G. Zheng, Z. Wang, R. Han, W. Xing, Z. Zhang, Y. Peng, and J. Pan. Embodied neuromorphic synergy for lighting-robust machine vision to see in extreme bright. Nature Communications, 15(1):10781, 2024. 1

[33] D. Liu, L. Cheng, T. Wang, and C. Sun. Edcflow: Exploring temporally dense difference maps for event-based optical flow estimation. In

2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1984–1993. IEEE, 2025. 2

[34] H. Liu, G. Chen, S. Qu, Y. Zhang, Z. Li, A. Knoll, and C. Jiang. Tma: Temporal motion aggregation for event-based optical flow. In ICCV, 2023. 2, 6

[35] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7

[36] H. Morimitsu, X. Zhu, R. M. Cesar, X. Ji, and X.-C. Yin. Dpflow: Adaptive optical flow estimation with a dual-pyramid framework. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17810–17820. IEEE, 2025. 2

[37] F. Paredes-Valles, K. Y. Scheper, C. De Wagter, and G. C. De Croon.´ Taming contrast maximization for learning sequential, low-latency, event-based optical flow. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9695–9705, 2023. 3

[38] F. Paredes-Valles and G. C. H. E. de Croon. Back to event basics:´ Self-supervised learning of image reconstruction for event cameras via photometric constancy. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3445–3454, 2021. doi: 10.1109/CVPR46437.2021.00345 2

[39] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12159–12168, 2021. doi: 10.1109/ ICCV48922.2021.01196 4

[40] A. Ranjan and M. J. Black. Optical flow estimation using a spatial pyramid network. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2720–2729, 2017. doi: 10.1109/ CVPR.2017.291 2

[41] H. Seok and J. Lim. Robust feature tracking in dvs event stream using bezier mapping. In´ 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 1647–1656, 2020. doi: 10. 1109/WACV45572.2020.9093607 2, 3

[42] X. Shi, Z. Huang, W. Bian, D. Li, M. Zhang, K. C. Cheung, S. See, H. Qin, J. Dai, and H. Li. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12435–12446, 2023. doi: 10.1109/ICCV51070.2023.01146 5

[43] S. Shiba, Y. Aoki, and G. Gallego. Event collapse in contrast maximization frameworks. Sensors, 22(14):5190, 2022. 2

[44] S. Shiba, Y. Klose, Y. Aoki, and G. Gallego. Secrets of event-based optical flow, depth and ego-motion estimation by contrast maximization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):7742–7759, 2024. 2

[45] L. N. Smith and N. Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications, vol. 11006, pp. 369–386. SPIE, 2019. 7

[46] D. Sun, X. Yang, M.-Y. Liu, and J. Kautz. PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume. 2018. 2, 3

[47] B. Tan, C. Sun, X. Qin, H. Adai, Z. Fu, T. Zhou, H. Zhang, Y. Xu, X. Zhu, Y. Shen, and N. Xue. Masked depth modeling for spatial perception. arXiv preprint arXiv:2601.17895, 2026. 9

[48] Z. Teed and J. Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pp. 402– 419. Springer, 2020. 2

[49] Z. Wan, Y. Dai, and Y. Mao. Learning dense and continuous optical flow from an event camera. IEEE Transactions on Image Processing, 31:7237–7251, 2022. 2

[50] Z. Wan, J. Luo, Y. Dai, and G. H. Lee. Event-aided dense and continuous point tracking: Everywhere and anytime. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7936–7946. IEEE, 2025. 2, 6

[51] J. Wang, D. Kai, H. Zhu, Q. Hu, Z. Xu, and X. Sun. Evreflection: Event-driven micro-dynamics for reflection removal. In Forty-third International Conference on Machine Learning, 2026. 9

[52] R. J. Wang and C. X. Ling. Swift: Efficient warping-only optical flow via scale-specialized refinement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3674– 3682, 2026. 3

[53] Y. Wang and J. Deng. Waft: Warping-alone field transforms for optical

flow. arXiv preprint arXiv:2506.21526, 2025. 2, 3, 5, 6

[54] Y. Wang and G. H. Lee. Flow4dgs-slam: Optical flow-guided 4d gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 33364–33373, 2026. 1

[55] Y. Wang, L. Lipson, and J. Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference on Computer Vision, pp. 36–54. Springer, 2024. 2, 5

[56] Y. Wu, F. Paredes-Valles, and G. C. H. E. de Croon. Lightweight ´ event-based optical flow estimation via iterative deblurring. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 14708–14715, 2024. doi: 10.1109/ICRA57147.2024.10610353 2, 3, 6, 7

[57] G. Xu, H. Lin, Z. Zhang, H. Luo, H. Sun, and X. Yang. Bat: learning event-based optical flow with bidirectional adaptive temporal correlation. In Proceedings of the Fortieth AAAI Conference on Artificial Intelligence and Thirty-Eighth Conference on Innovative Applications ofArtificial Intelligence and Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’26/IAAI’26/EAAI’26. AAAI Press, 2026. doi: 10.1609/aaai.v40i13.38100 2, 5, 6

[58] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8121– 8130, 2022. 2

[59] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024. 3

[60] Y. Yang, L. Pan, and L. Liu. Event camera data dense pre-training. In European Conference on Computer Vision, pp. 292–310. Springer, 2024. 6

[61] Y. Ye, H. Shi, K. Yang, Z. Wang, X. Yin, L. Sun, Y. Wang, and K. Wang. Towards anytime optical flow estimation with event cameras. Sensors, 25(10):3158, 2025. 2, 3

[62] P. Zhang, L. Zhu, X. Wang, L. Wang, and H. Huang. Ematch: A unified framework for event-based optical flow and stereo matching. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5845–5855. IEEE, 2025. 2

[63] Q. Zhou, J. Hou, M. Yang, Y. Deng, Y. Li, and J. Xiong. Spatiallyguided temporal aggregation for robust event-rgb optical flow estimation. IEEE Transactions on Multimedia, 28:5084–5095, 2026. doi: 10 .1109/TMM.2026.3664962 2, 3, 6

[64] Q. Zhou, Z. Zhu, J. Hou, Y. Deng, Y. Li, and J. Xiong. Resflow: Finetuning residual optical flow for event-based high temporal resolution motion estimation. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 1, 2, 6, 7, 9

[65] A. Zhu, L. Yuan, K. Chaney, and K. Daniilidis. Ev-flownet: Selfsupervised optical flow estimation for event-based cameras. In Proceedings of Robotics: Science and Systems. Pittsburgh, Pennsylvania, June 2018. doi: 10.15607/RSS.2018.XIV.062 2, 3

[66] A. Z. Zhu, L. Yuan, K. Chaney, and K. Daniilidis. Unsupervised event-based learning of optical flow, depth, and egomotion. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 989–997, 2019. doi: 10.1109/CVPR.2019.00108 2