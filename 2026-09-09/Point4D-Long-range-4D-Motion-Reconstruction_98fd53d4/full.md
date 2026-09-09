# Point4D: Long-range 4D Motion Reconstruction

Minsik Jeon Jay Karhade Deva Ramanan<sup>†</sup> Shubham Tulsiani<sup>†</sup> Carnegie Mellon University Project Page: https://point-4d.github.io

Robust Long-Range 4D Motion Reconstruction through 3D Query-based Trajectory Chaining

![](images/ed162d187787f76304dfc48a721bfd06b8ca6b95978eb88bbeee4317c7eba5ed.jpg)  
Figure 1: Point4D enables long-range 4D reconstruction via autoregressively chaining motion reconstruction across overlapping chunks. Existing 4D reconstruction methods (e.g., TraceAnything [19], Any4D [15], 4RC [20], VDPM [27]) predict 4D motion for query 2D pixels, and cannot be easily chained under occlusion as the tracked points may not be visible in overlapping frames. Point4D decodes motion for query 3D points instead of 2D pixels, allowing direct chaining for long-range motion reconstruction under occlusions.

## Abstract

We introduce Point4D, a feed-forward model for 4D reconstruction of long-range video sequences. Point4D is able to reliably infer dense per-point 3D trajectories across multi-hundred-frame videos, unlike existing 4D methods that are limited to short input windows of at most a few dozen frames. A key innovation that enables this is our flexible 3D query-based motion decoder that decouples trajectory prediction from image-plane visibility. The predicted 3D endpoints are then directly re-queried in the next chunk without re-projection or matching. Furthermore, we show that extracting and reusing a visual descriptor from an arbitrary frame where the point is visible leads to better performance than relying solely on the source patch. Overall, Point4D achieves state-of-the-art performance across diverse long video tracking benchmarks spanning over 200 frames and largely outperforms previous feed-forward 4D method. Project page: point-4d.github.io

(a) 2D Query-based Trajectory Chaining

## 1 Introduction

Consider the runner in Fig. 1: a person jogging down a long corridor, vanishing behind structural columns and reappearing moments later. We humans can effortlessly follow the runner’s continuing trajectory throughout the clip, understanding their long-range motion despite occlusions, and even reasoning about where they may be when not directly visible. In this work, we seek to build a computational system that can similarly perform long-range 4D motion reconstruction from a monocular video — recovering per-point 3D trajectories across hundreds of frames.

While initial approaches [34, 35, 44, 39] for such ‘4D reconstruction’ leveraged slow and expensive iterative optimization, there has been a marked shift towards feed-forward methods [15, 45, 20, 24, 19, 27] which, following the successes in feed-forward 3D reconstruction [37, 33], have extended such multi-view models to additionally perform motion prediction. Specifically, by adding scene flow prediction ‘heads’ to multi-view models, these approaches allow decoding each pixel’s 3D position at any queried timestep, thereby inferring the 4D motion of the scene in an efficient feed-forward manner. While these methods deliver impressive results, they are designed to operate over short input windows of a handful of frames. Long videos that span hundreds or more frames expose two challenges that short windows do not face. First, jointly processing the full frame set is computationally infeasible for the underlying transformer encoders, and second, the points to be tracked routinely become occluded or leave the field of view between observations.

The first challenge is not unique to long-range 4D reconstruction. Indeed, approaches for longrange static reconstruction are similarly bottlenecked by computational complexity, and a common solution is to chain short-range predictions across overlapping chunks [6]. Whereas chaining static reconstruction merely requires aligning the coordinate systems from independent per-chunk reconstructions, chainingfor 4D reconstruction requires autoregressive motion prediction – the motion for a point tracked in an earlier chunk needs to be queried again from a later chunk to continue the point track. Existing 4D methods that compute 3D track predictions for query pixels are not well suited for such chaining, either requiring reprojection from 3D to 2D for re-querying or a computationally expensive ‘matching’ of independently predicted 4D trajectories. However, these strategies break

![](images/dd067775ef432e556dd902a79e255c98c32a074337db033990c6ea3ce155dc4a.jpg)

![](images/ad9557f72780f60469e155d894f4ec579f3362821747205486d586b2742b6291.jpg)  
(b) 3D Query-based Trajectory Chaining  
Figure 2: Point4D uses 3D query representations for chaining motion prediction. A 2D-querybased motion decoder requires image reprojection and is not robust to chaining under occlusions. Our insight is to leverage 3D queries, allowing direct chaining without reprojection.

down when a point is occluded or outside the field of view at the chunk boundary which are the very situations that are routine in long videos. Our key insight is to query points, not pixels – a motion reconstruction decoder with 3D points as queries allows the predicted endpoint of one chunk to be re-queried in the next directly, with no reprojection or matching (see Fig. 2). More generally, 3D queries also encourage the decoder to focus on motion reconstruction instead of geometry estimation (which is already captured in the input) and can also help resolve ambiguous 2D pixel queries e.g., on object boundaries where it may be unclear if the query is a foreground or background pixel.

We operationalize this insight in Point4D , a feed-forward model that decodes 3D motion from a 3D-coordinate query. A shared ViT encoder produces a scene representation along with per-frame depth and camera poses; a lightweight cross-attention decoder takes a 3D query, target time and camera indices, and a visual descriptor extracted once from any prior frame where the point is visible; it then predicts the queried point’s 3D position at the target time. To process long videos, we partition the video into overlapping chunks, autoregressively re-querying each predicted 3D endpoint in the next chunk based on the decoded trajectory from the previous one, while reusing the same visual descriptor throughout. We validate our approach across multiple datasets and show that Point4D substantially improves over the prior state of the art on long-video 4D tracking while also improving over its 2D-query counterpart for short-range motion reconstruction.

In summary, our contributions are:

• We build a feed-forward pipeline for long-range 4D motion reconstruction, recovering per-point motion across hundreds of frames via trajectory chaining across chunks.

• We introduce a 3D-query-based motion decoding that decouples trajectory prediction from image-plane visibility. This also allows predicted 3D endpoints to be propagated directly across chunks.

• We achieve state-of-the-art long-video 4D tracking, outperforming both chaining-based feed-forward 4D methods and 3D point trackers.

## 2 Related Work

3D Tracking. To model scene motion, point tracking and optical flow [28] methods estimate pixel-level correspondences across frames. Subsequent work extended tracking to longer temporal ranges through correlation-based matching and iterative updates [7, 8, 14], yet these methods operate purely in 2D. The TAPVid-3D benchmark [17] motivated lifting tracks into 3D: DELTA [21] and SpatialTracker [39] combine 2D trackers with monocular depth to track points in camera coordinates. More recently, TAPIP3D [44] lifts video features into a camera-stabilized 3D point cloud and iteratively refines trajectories in 3D space, while SpatialTrackerV2 [40] learns geometry and point motion jointly in an end-to-end architecture, eliminating separate depth estimation. However, these methods only track sparse query points, often rely on off-the-shelf depth modules, and require iterative refinement that limits speed. Point4D can predict dense 3D trajectories in a single forward pass via query-based decoding, while remaining competitive with iterative trackers on sparse benchmarks.

4D Reconstruction of Dynamic Scenes. 4D reconstruction aims to recover both the 3D structure of a scene and how it changes over time. Early methods rely on per-scene optimization [35], achieving high fidelity but at a prohibitive cost. Recent feed-forward approaches extend 3D reconstruction architectures to the temporal dimension, predicting dense geometry and motion jointly from a short window of frames [9, 19, 15, 27, 20]. Although these methods differ in motion representation – scene flow from a canonical view [15], continuous trajectory fields [19], or dense point maps at queried timesteps [20, 27], they all decode motion for every pixel through DPT-style heads or similar dense decoders. D4RT [45] instead introduces a query-based 4D decoder that predicts the 3D position of a 2D query point at a target timestep and coordinate frame. In all cases, however, queries live on the image plane, so extending a track beyond the window requires reprojecting predictions back to pixels, which fails under occlusion. Together with the memory cost of joint multi-frame encoding, this confines prior work to a few dozen frames. We extend D4RT’s query-based decoding from 2D to 3D queries, which decouples tracking from visibility and enables longer video 4D reconstruction to over hundreds of frames through direct re-querying of predicted 3D positions.

Long-range 3D reconstruction. Feed-forward 3D reconstruction methods [37, 33, 18, 16] have shown impressive results in multi-view 3D reconstruction. However, these approaches only work for a limited number of frames, due to high memory requirements, and quickly run out of memory for longer sequence inputs. CUT3R [36], InfiniteVGGT [43], StreamingVGGT [50] proposed memory mechanisms for long-sequence reconstructions, while VGGT-Long [6] proposes a chunking and loop closure mechanism. More recently, Loger [47], TTT3R [3] and Zipmap [12] integrate test-time training mechanisms. These works demonstrate that 3D reconstruction models trained for small inputs can be effectively chained to handle longer sequences. Point4D extends this insight to the 4D setting: rather than simply chaining static geometry, it uses 3D point queries to chain motion across chunks, recovering dense scene flow and long-range 3D tracks in a scalable, feed-forward manner.

## 3 Method

Our goal is to recover dense 3D trajectories from long monocular video sequences. To achieve this, we propose Point4D , a feed-forward model that uses 3D coordinate queries as input instead of inferring motion from 2D pixels. By querying directly in 3D space, we decouple trajectory prediction from image-plane visibility. Building upon established visual geometry prediction models [33, 18, 20] and D4RT [45], we first encode the video into a feature representation while predicting depth and camera poses to support query decoding (Sec. 3.1). We then reformulate the query mechanism to use

Input Video � Baseline: 2D-Query based 4D Reconstruction (D4RT)  
![](images/12f7f3d587200f63064c3ef766e63c0fe0d454043983403de3d22bb67339c29c.jpg)  
Figure 3: Overview of Point4D . Unlike 2D-query methods, which bind a query to a source pixel and its patch, Point4D queries a 3D point with a descriptor drawn from any frame in which it is visible. Given a monocular video $V ,$ an encoder $\mathcal { E }$ with self-attention layers produces a scene representation $\mathcal { F }$ together with per-frame depth maps and camera poses. A 3D query ${ \bf p } = ( x , y , z )$ is constructed as the point’s position at time $t _ { \mathrm { s r c } } ,$ , expressed in the camera coordinates of frame $t _ { \mathrm { s r c } } ,$ and its visual patch is extracted from any timestep where the point is visible. The resulting query embedding q cross-attends to $\mathcal { F }$ to predict the point’s 3D position at the target timestep $t _ { \mathrm { t g t } } .$ . Together, the 3D query and its visibility-agnostic descriptor decouple a query point from any single frame’s image plane, so that occluded and out-of-frame points remain valid queries.

3D points $( x , y , z )$ in the source frame, paired with a visual descriptor extracted from any timestep at which the point is visible (Sec. 3.2). This formulation allows us to predict a point’s 3D position at any target time regardless of its image-plane visibility. Finally, we present a simple chaining framework for long-range 3D trajectory inference (Sec. 3.3). Since 3D queries remain well-defined under occlusion, we can propagate trajectories across overlapping video chunks by re-querying predicted 3D endpoints, remaining valid through occlusions and field-of-view exits. An overview of our method is shown in Fig. 3.

## 3.1 Preliminaries

Feed-forward visual geometry prediction. Recent feed-forward methods [18, 33, 46, 37, 4] process multi-view images, or video V through a ViT backbone E with alternating frame-wise and global self-attention layers, producing patch tokens $\mathbf { Z } _ { i }$ and camera tokens $\mathbf { c } _ { i }$ for each frame $i \in \bar { \{ 1 , . . . , T \} }$ . Dedicated heads decode camera poses $\mathbf { P } _ { i }$ from $\mathbf { c } _ { i }$ and depth maps $\mathbf { D } _ { i }$ from $\mathbf { Z } _ { i }$ via a DPT decoder [25]. To handle dynamic scenes, recent work augments this representation with a learnable time token $\mathbf { t } _ { i }$ per frame [20], initialized from a sinusoidal encoding of the normalized timestep in [0, 1] and refined through the same attention layers. This yields a scene representation ${ \mathcal { F } } = \{ \bar { \mathbf { Z } } _ { i } \} _ { i = 1 } ^ { T }$ along with per-frame tokens and predictions:

$$
\mathcal { F } , \{ \mathbf { c } _ { i } , \mathbf { t } _ { i } , \mathbf { P } _ { i } , \mathbf { D } _ { i } \} _ { i = 1 } ^ { T } = \mathcal { E } ( V ) .\tag{1}
$$

Query-based 4D decoding. While most 4D reconstruction methods utilize DPT-style heads for dense geometry, D4RT introduces on-demand query-based decoding. This mechanism defines a query as $( u , v , t _ { \mathrm { s r c } } , t _ { \mathrm { t g t } } , t _ { \mathrm { c a m } } , S )$ , where $( u , v )$ represents a 2D pixel in frame $t _ { \mathrm { s r c } }$ and S provides visual context via an RGB patch around $( u , v )$ . The decoder D then predicts the 3D position pˆ of the corresponding point pixel at a target timestep $t _ { \mathrm { t g t } } .$ expressed in camera $t _ { \mathrm { c a m } } \mathrm { \tilde { s } }$ coordinate system. Each query is mapped to an embedding q and independently cross-attends to $\mathcal { F }$ to produce a predicted point:

$$
\begin{array} { r } { \hat { \mathbf { p } } = \mathcal { D } ( \mathbf { q } , \mathcal { F } ) \in \mathbb { R } ^ { 3 } . } \end{array}\tag{2}
$$

This formulation enables flexible decoding of arbitrary pixel sets across varying timesteps and coordinate frames. However, like prior 4D methods, D4RT relies on 2D pixel coordinates, requiring the queried point to be visible in the source frame for initialization.

## 3.2 3D Query-based Decoder

Existing 4D reconstruction methods infer motion via predicting scene flow for 2D pixels and require the tracked 3D point to be visible in the source frame. This breaks down when the point is occluded or outside the field of ${ \mathrm { v i e w } } ,$ and is especially problematic across video chunks where the predicted 3D position at the end of one chunk may have no visible pixel to re-query in the next. We address this with two key design choices. First, we replace 2D pixels with 3D points as queries, which decouples point identity from image-plane visibility, and allows the decoder to focus on predicting motion rather than jointly recovering geometry. Second, we supply appearance context via a local image patch drawn from any frame where the point is visible, rather than tying visual context to the query frame. This enables cross-chunk trajectory chaining as the patch can be sourced from a different chunk entirely, even when the point is occluded or out of view in the current one.

3D query construction. To specify a query, we first select a pixel $( u , v )$ that is visible in some frame $t _ { \mathrm { r e f } }$ and extract a local image patch $\bar { S }$ around it as a visual descriptor. We then obtain the query coordinate ${ \bf p } = ( x , y , z )$ by taking this point’s 3D position at a source time $t _ { \mathrm { s r c } } ,$ expressed in frame $t _ { \mathrm { s r c } } \mathrm { \ ' } _ { \mathrm { s } }$ camera coordinate system. When $t _ { \mathrm { r e f } } \neq t _ { \mathrm { s r c } }$ , the point may have moved and may be occluded or outside the field of view at $t _ { \mathrm { s r c } }$ . While its corresponding 2D pixel is then ill-defined, the 3D coordinate remains valid since it does not depend on the pixel projection. The full query is $( \mathbf { p } , t _ { \mathrm { s r c } } , t _ { \mathrm { t g t } } , t _ { \mathrm { c a m } } , S )$ where is the 3D point at position p (in frame $t _ { s r c } \ ' _ { s }$ coordinate frame) with visual descriptor S at timestep $t _ { t g t } ,$ expressed in camera $t _ { c a m } \ ' _ { s }$ coordinateframe?

Query embedding and decoding. To build the query embedding ${ \bf q } ,$ the 3D spatial coordinates p are first encoded with sinusoidal positional encoding [31]. The temporal and camera indices are encoded using corresponding tokens from the encoder: $t _ { \mathrm { s r c } }$ and $t _ { \mathrm { { c a m } } }$ use camera tokens $\mathbf { c } _ { t _ { \mathrm { s r c } } }$ and $\mathbf { c } _ { t _ { \mathrm { c a m } } } ,$ while $t _ { \mathrm { t g t } }$ uses time token $\mathbf { t } _ { t _ { \mathrm { t g t } } } .$ . These encodings are summed with an embedding of $S$ to form the query embedding $\mathbf { q } .$ . The decoder then follows D4RT (Eq. 2), passing $\mathbf { q }$ through cross-attention layers attending to $\mathcal { F }$ and obtain predicted 3D position ${ \hat { \mathbf { p } } } .$ . It uses no self-attention between queries, so each query is decoded independently, enabling flexible batching of arbitrary query sets at inference.

Loss. The primary loss $\mathcal { L } _ { \mathrm { p o i n t } }$ is an L1 loss on the predicted 3D position, where both prediction and target are passed through a signed log-transform $\phi ( x ) = \mathrm { s i g n } ( x ) \log ( 1 + | x | )$ to dampen the influence of far-away points. The confidence loss ${ \mathcal { L } } _ { \mathrm { c o n f } }$ modulates $\mathcal { L } _ { \mathrm { p o i n t } }$ by a per-query confidence score, so that uncertain predictions are penalized less heavily. The auxiliary reprojection loss $\mathcal { L } _ { 2 \mathrm { d } }$ is an L1 loss on the predicted 2D projection of pˆ into frame $t _ { \mathrm { { c a m } } } .$ , enforcing consistency between the predicted 3D position and camera geometry. The auxiliary visibility loss ${ \mathcal { L } } _ { \mathrm { v i s } }$ is a binary cross-entropy loss on a per-query logit predicting whether the point is visible at $t _ { \mathrm { t g t } }$ . The total loss is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { p o i n t } } + \mathcal { L } _ { \mathrm { c o n f } } + \mathcal { L } _ { 2 \mathrm { d } } + \mathcal { L } _ { \mathrm { v i s } } .\tag{3}
$$

## 3.3 Trajectory Chaining

Long videos cannot be processed in a single forward pass due to memory constraints. We therefore partition the video into short, overlapping chunks and encode each chunk independently. Producing coherent long-range trajectories from these chunk-level predictions requires (1) aligning each chunk’s independent 3D coordinate frame into a single global frame, and (2) propagating point identity across chunk boundaries so that per-chunk predictions form a single continuous trajectory.

Alignment of chunk-level 4D reconstruction. Since each chunk’s geometry is predicted up to an unknown scale and its own coordinate frame, we align adjacent chunks via a Sim(3) transformation [30] estimated from dense depth predictions on the shared overlap frames. Composing these pairwise Sim(3) transforms places every chunk’s predictions into a single global coordinate frame. While this procedure suffices for static 3D reconstruction, we also need to propagate the temporal point identity across chunks for 4D scene reconstruction.

Chaining trajectories with 3D queries. Our 3D query formulation reduces this correspondence problem to a single coordinate transform. Within each chunk, we decode every query point’s trajectory by sweeping the target time $t _ { \mathrm { t g t } }$ across the chunk’s frames. At the first overlapping frame $t _ { k }$ between chunk k and chunk $k { \mathrel { + { 1 } } }$ , we take the predicted 3D position $P _ { i } ( t _ { k } )$ and apply the already-estimated Sim(3) to express it in chunk k+1’s local coordinates, yielding the re-query coordinate $\mathbf { p } _ { i }$ for the next chunk. The visual descriptor $S _ { i }$ , extracted once from the first frame where the point is visible, is reused across all subsequent chunks without re-extraction. This procedure chains trajectories regardless of whether the point is visible in the overlap: the 3D coordinate is always well-defined, so occluded and out-of-frame points propagate without failure. In contrast, 2D-query methods must project the predicted 3D position back to pixel coordinates to re-query, which is undefined for occluded points and compounds camera prediction error for visible ones. The overall trajectory chaining procedure is described in Algorithm 1.

Algorithm 1 Trajectory Chaining for Long Video   
Require: Video V of T frames; query pixels $\{ ( u _ { i } , v _ { i } ) \}$ visible in frame 0   
Ensure: 4D trajectories $\{ P _ { i } ( t ) \} _ { t = 0 } ^ { T - 1 }$ in a global coordinate frame   
1: Partition V into K overlapping chunks $\{ C _ { k } \} _ { k = 0 } ^ { K - 1 }$ , where $C _ { k }$ starts at global time $t _ { k }$   
2: $\mathbf { p } _ { i }  \mathrm { U N P R O J E C T } ( u _ { i } , v _ { i } , \mathbf { D } _ { 0 } ) \quad \forall i$ ▷ lift pixel query to 3D at t=0   
3: $\bar { S } _ { i } \gets$ EXTRACTPATCH(frame<sub>0</sub>, $u _ { i } , v _ { i } )$ ∀i ▷ visual descriptor, extracted once   
4: for $k = 0 , \ldots , K - 1$ do   
5: $F _ { k } \gets \mathrm { E N C O D E } ( C _ { k } )$ ▷ 4D scene representation for chunk k   
6: $\pi _ { t }  ( t _ { s r c } = 0 , t _ { t g t } = t - t _ { k } , t _ { c a m } = 0 )$ ▷ $( t _ { s r c } , t _ { t g t } , t _ { c a m } )$ local to $C _ { k }$   
7: $P _ { i } ( t ) \gets \mathcal { D } ( ( \mathbf { p } _ { i } , \pi _ { t } ^ { - } , S _ { i } ) , F _ { k } ) \quad \forall i , \forall t \in C _ { k }$ ▷ decode 3D position at each frame   
8: $\mathbf { i f } \ k < K - 1$ then   
9: $( c , \mathbf { R } , \mathbf { t } ) \gets$ ESTIMATESIM $3 ( C _ { k } \cap C _ { k + 1 } )$ ▷ align chunks via overlap depth   
10: $\mathbf { \Delta p } _ { i } \gets c \mathbf { R } P _ { i } ( t _ { k + 1 } ) + \mathbf { t } \quad \forall i$ ▷ propagate 3D query into $C _ { k + 1 } \mathbf { \dot { s } }$ frame; reuse $S _ { i }$   
11: end if   
12: end for   
13: return $\{ P _ { i } ( t ) \} _ { t = 0 } ^ { T - 1 }$ transformed to global frame-0 coordinates

## 3.4 Implementation Details

We initialize the encoder and geometry heads from Depth Anything 3 [18] pretrained weights, while the decoder is trained from scratch. We train on a mixture of dynamic datasets (PointOdyssey [48], Dynamic Replica [13], Bedlam2 [29], Kubric Movi-F [11], CoTracker-Kubric [14], Waymo [1], Omniworld [49]) and static datasets (ScanNet [5], ScanNet++ [42], BlendedMVS [41], Co3Dv2 [26], WildRGBD [38]), where static points are treated as stationary trajectories. Each training sequence contains 16 to 64 frames, with width sampled from [252, 518]. Further details are in Appendix A.

## 4 Experiments

## 4.1 Experimental Setting

Baselines. We compare against two categories of methods: (1) Feed-forward 4D reconstruction methods such as TraceAnything [19], Any4D [15], 4RC [20], and V-DPM [27], which predict dense per-pixel 3D positions at queried timesteps and are most directly comparable to ours. (2) 3D point trackers such as SpatialTrackerV2 [40] and TAPIP3D [44], which represent the current state of the art in per-point 3D tracking but do not support dense per-pixel queries and are slower (Appendix B).

Setup. We evaluate Point4D on (A) long-video tracking, which requires trajectory chaining across chunks, and a simpler (B) single-chunk tracking. For long-video tracking, we use PointOdyssey and Dynamic Replica sequences of 200 frames and TAPVid3D Panoptic Studio (PStudio) sequences of 150 frames. Each sequence is partitioned into chunks of 48 frames with 8-frame overlap, where, for Point4D , trajectories are chained across chunks using 3D queries (Sec. 3.3). Feed-forward 4D methods are evaluated with selection-based reprojection chaining, which picks the overlapping frame with the best depth agreement for each point (Appendix D). SpatialTrackerV2 and TAPIP3D use their sliding-window inference modes. For single-chunk tracking, we evaluate on LSFOdyssey, Dynamic Replica and PStudio with up to 64 frames per sequence.

Metrics. Following the benchmarking protocol of recent works [9, 15], we report endpoint error (EPE) and the average percentage of points within distance thresholds $\delta _ { 3 D } \in$ {0.1, 0.3, 0.5, 1.0} m (APD), after median-scale alignment of ground truth and predicted trajectories (Further details are in Appendix A).

![](images/67dfb648b07339368b1023a1237311905db9f83ef91fae6c08b192990a0989e3.jpg)

![](images/5ecd76ee73ccde28fa962b8166cb2d91a873ecc9cb02abb2414529eefb830724.jpg)

![](images/820ced1f25899b8f55af3b0d36274c645039dd3c42268f7bb79cde053a97e19b.jpg)

![](images/d6afa229e66d147fa77605f1f9419a9d41936dc1ca9ba060f2c7cda8551646dd.jpg)

![](images/79d9a001ee5ef1dc20c47a6540b8dce3d9e69106207a4a4327b9266b93679183.jpg)

![](images/df33515107780e42eebc6d6c9481fbe2e98818bf3165a2251dded0dd68efd39c.jpg)

![](images/5f7bc573870d5b927e8986ed28222bf47b9cd21dd77b4469d50aaed19641e902.jpg)  
Input Video

![](images/998127f9e20823160541fde25e389353da9aa77406da5c4f147f1fe929f5597a.jpg)  
SpatialTrackerV2

![](images/5dc571146c702dd160fd4c0fc355fbb39e285427b58f787ed035139cc1e0be38.jpg)

![](images/951bd33ef5747795ce6c46c179a1c5fac9390f7747a1c9baa19be0d9c9b619e1.jpg)  
4RC  
Point4D

Figure 4: Point4D produces reliable and consistent 4D tracking over 200-frame sequences via trajectory chaining. 4RC struggles to maintain correspondence and its tracks are wrongly chained. SpatialTrackerV2 only can predict a sparser trajectory due to memory constraints.
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">PointOdyssey [48]</td><td colspan="3">Dynamic Replica [13]</td><td colspan="3">PStudio [17]</td></tr><tr><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td></tr><tr><td rowspan="2">ter.</td><td>TAPIP3D</td><td>0.952</td><td>0.417</td><td>0.317</td><td>0.185</td><td>0.806</td><td>0.748</td><td>0.230</td><td>0.741</td><td>0.622</td></tr><tr><td>SpatialTrackV2</td><td>0.498</td><td>0.611</td><td>0.477</td><td>0.218</td><td>0.772</td><td>0.687</td><td>0.234</td><td>0.719</td><td>0.623</td></tr><tr><td rowspan="4">Re-ard</td><td>TraceAnything</td><td>2.147</td><td>0.121</td><td>0.070</td><td>0.767</td><td>0.464</td><td>0.391</td><td>0.665</td><td>0.408</td><td>0.312</td></tr><tr><td>Any4D</td><td>1.026</td><td>0.400</td><td>0.291</td><td>0.364</td><td>0.711</td><td>0.629</td><td>0.497</td><td>0.495</td><td>0.389</td></tr><tr><td>4RC</td><td>0.789</td><td>0.559</td><td>0.463</td><td>0.336</td><td>0.733</td><td>0.654</td><td>0.379</td><td>0.613</td><td>0.534</td></tr><tr><td>VDPM</td><td>0.736</td><td>0.559</td><td>0.461</td><td>0.386</td><td>0.666</td><td>0.587</td><td>0.280</td><td>0.719</td><td>0.634</td></tr><tr><td></td><td>Point4D</td><td>0.616</td><td>0.585</td><td>0.514</td><td>0.155</td><td>0.856</td><td>0.812</td><td>0.236</td><td>0.731</td><td>0.664</td></tr></table>

Table 1: Long-Video 4D Tracking via Trajectory Chaining. Point4D achieves the best average rank among all compared methods, outperforming every other feed-forward method, while running much faster than iterative trackers (see Appendix). We report EPE (↓), APD (↑), and survival rate (↑) on sequences of 200 frames, partitioned into chunks of 48 frames with 8-frame overlap. Red , Orange , and Yellow indicate the top three results.

$$
\mathrm { E P E } _ { i , t } = \Vert \hat { \mathbf { p } } _ { i } ^ { t } - \mathbf { p } _ { \mathrm { G T } , i } ^ { t } \Vert\tag{4}
$$

$$
\mathrm { A P D } = \sum _ { i , t } \mathbb { 1 } \cdot ( \mathrm { E P E } _ { i , t } < \delta _ { \mathrm { 3 D } } )\tag{5}
$$

For long-video, we additionally report Survival rate [48]: the average fraction of video length before tracking failure. A point i is considered failed at frame t if $\left. \hat { \mathbf { p } } _ { i } ^ { t } - \mathbf { \check { p } } _ { \mathrm { G T } , i } ^ { t } \right. _ { 2 } > \delta _ { \mathrm { 3 D } }$ , and we report:

$$
\mathrm { S u r v i v a l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \frac { t _ { i } ^ { \mathrm { f a i l } } - 1 } { T } }\tag{6}
$$

where $t _ { i } ^ { \mathrm { f a i l } }$ is the first failure frame and N is total queries. We average over $ { \delta _ { \mathrm { 3 D } } } \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \}$ m.

## 4.2 4D Tracking

Long-video trajectory chaining. Table 1 compares all methods on 200-frame sequences (150 for PStudio) that require chaining across multiple chunks. Point4D outperforms both categories of baselines on most sequences. Feed-forward 4D baselines use reprojection-based chaining, which is undefined for occluded points and compounds camera prediction error even for visible ones, causing trajectories to drift or break at chunk boundaries. Iterative 3D trackers propagate tracks within their own fixed windows, but their iterative refinement is slow and memory constraints preclude dense query sets. In contrast, Point4D re-queries predicted 3D coordinates directly in each subsequent chunk, avoiding both failure modes.

![](images/c4d20a6dc09a7c828eccb2d33115da82cf878bb4cd4d96c969e8959faba79555.jpg)  
Figure 5: Single-chunk 4D tracking on DAVIS [23]. Point4D produces consistent 3D trajectories across challenging real-world sequences.

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">LSFOdyssey [32]</td><td colspan="2">Dynamic Replica [13]</td><td colspan="2">PStudio [17]</td></tr><tr><td>EPE↓</td><td>APD↑</td><td>EPE↓</td><td>APD↑</td><td>EPE↓</td><td>APD↑</td></tr><tr><td rowspan="4">2D er</td><td>MonST3R + CoTracker3</td><td>0.61</td><td>0.51</td><td>0.81</td><td>0.43</td><td>0.51</td><td>0.52</td></tr><tr><td>MASt3R + CoTracker3</td><td>0.83</td><td>0.46</td><td>0.40</td><td>0.58</td><td>0.43</td><td>0.54</td></tr><tr><td>VGGT + CoTracker3</td><td>0.47</td><td>0.59</td><td>0.26</td><td>0.69</td><td>0.26</td><td>0.69</td></tr><tr><td>MapAnything + CoTracker3</td><td>0.63</td><td>0.35</td><td>0.25</td><td>0.71</td><td>0.63</td><td>0.51</td></tr><tr><td rowspan="2"></td><td>DepthAnything3 +CoTracker3</td><td>0.50</td><td>0.69</td><td>0.11</td><td>0.89</td><td>0.33</td><td>0.63</td></tr><tr><td>TAPIP3D</td><td>0.35</td><td>0.66</td><td>0.87</td><td>0.50</td><td>0.30</td><td>0.65</td></tr><tr><td rowspan="2">Iter</td><td>SpatialTrackv2</td><td>0.34</td><td>0.68</td><td>0.69</td><td>0.62</td><td>0.21</td><td>0.75</td></tr><tr><td>St4RTrack</td><td>0.56</td><td>0.48</td><td>0.17</td><td>0.81</td><td>0.41</td><td>0.53</td></tr><tr><td rowspan="5">Re-rd</td><td>TraceAnything</td><td>0.76</td><td>0.42</td><td>0.34</td><td>0.66</td><td>0.25</td><td>0.72</td></tr><tr><td>Any4D</td><td>0.27</td><td>0.72</td><td>0.07</td><td>0.93</td><td>0.28</td><td>0.66</td></tr><tr><td>4RC</td><td>0.16</td><td>0.87</td><td>0.07</td><td>0.95</td><td>0.29</td><td>0.66</td></tr><tr><td>VDPM</td><td>0.14</td><td>0.85</td><td>0.14</td><td>0.84</td><td>0.17</td><td>0.79</td></tr><tr><td>Point4D</td><td>0.27</td><td>0.73</td><td>0.09</td><td>0.91</td><td>0.23</td><td>0.73</td></tr></table>

Table 2: Single-chunk 4D tracking. Point4D performs comparably to existing methods on short sequences that do not require trajectory chaining, which confirms that the long-video gains in Table 1 stem from the proposed 3D query formulation, not a stronger single-chunk decoder. We report EPE (↓) and APD (↑). Red , Orange , and Yellow indicate the top three results.

Figure 4 shows qualitative results on 200-frame sequences, where we sample a regular grid of query points on the first frame and decode their trajectories across chunks. Point4D maintains continuous trajectories through occlusion and field-of-view exits, while 2D-query methods lose track at chunk boundaries. SpatialTrackerV2 uses a sparser query grid due to GPU memory constraints, resulting in visibly sparser trajectories. Additional results can be found in Appendix E and F.

Single-chunk tracking. Table 2 evaluates all methods on sequences short enough to fit within a single chunk (64 frames), isolating decoder accuracy from chaining. Point4D performs comparably to other feed-forward 4D methods while outperforming both 2D tracker + 3D reconstruction pipelines and iterative refinement-based methods. Combined with the long-video results, this shows that our advantage on longer sequences comes from reliable 3D-query chaining rather than a gap in single-chunk decoding. Figure 5 shows representative results on DAVIS sequences.

## 4.3 Ablations and Analysis

Ablation on query formulation. We ablate the two key design choices in Point4D’s query formulation: (1) using 3D coordinates instead of 2D pixels, and (2) training with visual descriptors from arbitrary visible frames rather than only the source frame. Table 3 compares three variants: 2D queries with pixel coordinates (u, v) as in D4RT [45]; 3D (source patch), which uses 3D coordinates but always extracts the visual descriptor from the source frame, so the model never sees occluded or out-of-frame queries during training; and 3D (Point4D ), which combines 3D coordinates with descriptors from arbitrary visible frames. The 2D variant must reproject each predicted point back to the image plane to continue a trajectory, as in other 2D-query based methods. The 3D source patch variant avoids reprojection but expects a visible query with descriptor from the source frame, where the point may be occluded or out of view at a chunk boundary. Point4D outperforms both, especially in long-video tracking, indicating that both design choices are needed for reliable chaining.

![](images/fec606eb052b227ed9d6f68ea3aee273c27f3a7373d2c95227a20d9ef56749f5.jpg)

<table><tr><td></td><td colspan="4">Long-Video Tracking</td><td colspan="4">Single-Chunk Tracking</td></tr><tr><td></td><td colspan="2">PointOdyssey [48]</td><td colspan="2">Dynamic Replica [13]</td><td colspan="2">LSFOdyssey [32]</td><td colspan="2">Dynamic Replica [13]</td></tr><tr><td>Query Formulation</td><td>EPE↓</td><td>SR↑</td><td>EPE↓</td><td>SR↑</td><td>EPE↓</td><td>APD↑</td><td>EPE↓</td><td>APD↑</td></tr><tr><td>2D</td><td>0.891</td><td>0.283</td><td>0.712</td><td>0.422</td><td>0.279</td><td>0.713</td><td>0.110</td><td>0.870</td></tr><tr><td>3D (source patch)</td><td>0.869</td><td>0.380</td><td>0.825</td><td>0.266</td><td>0.442</td><td>0.620</td><td>0.121</td><td>0.800</td></tr><tr><td>3D (Point4D)</td><td>0.616</td><td>0.514</td><td>0.155</td><td>0.812</td><td>0.274</td><td>0.731</td><td>0.091</td><td>0.906</td></tr></table>

Table 3: Point4D ’s proposed 3D query formulation is optimal for reliable decoding. We ablate this choice on both single-chunk and long-video chaining benchmarks. We report EPE (↓), APD (↑) and Survival Rate (SR,↑).

## Robustness of tracking across chunks.

To verify that our 3D query formulation enables reliable trajectory chaining across video chunks, we measure per-chunk APD and EPE on PStudio long tracking result. Figure 6 shows that Point4D degrades slowly across chunks in both APD and EPE, while baselines deteriorate faster as chaining proceeds. The gap widens because 2D-query methods compound error at every chunk boundary through reprojection, whereas our 3D re-querying avoids this entirely. Notably, VDPM starts with higher firstchunk accuracy, yet Point4D surpasses them within a few chunks. This confirms that the advantage of 3D queries lies not in single-chunk

Figure 6: Chunk-wise tracking accuracy on PStudio. Point4D (red) degrades slowly across chunks, while others worsen faster.

decoding but in chaining across many chunks, which matters for long-video 4D reconstruction.

## 5 Discussion

We presented Point4D, a feed-forward model for 4D reconstruction of long-range video sequences. Point4D infers motion using 3D coordinate queries compared to prior feed-forward methods which inferred motion from 2D pixels. This decouples trajectory prediction from image-plane visibility, and allows for direct requerying of 3D points across chunks even when occluded or out of field of view. Moreover, we show that our visual descriptor extracted from arbitrary frames improves over the source patch descriptor alone. We show these design choices enable Point4D to outperform existing methods on long-range videos. Ultimately, we believe Point4D will serve as a foundation step towards achieving reliable 4D reconstruction on in-the-wild videos of arbitrary length and serve for diverse applications in generative AI, AR/VR and robotics.

Limitations. The handoff between chunks carries only each query’s 3D coordinate and its patch descriptor; no scene representation or feature memory is retained. Since the encoder represents only what is observed within the current chunk, a point that remains occluded or out of frame for an entire chunk has no supporting evidence in that chunk, and its predicted position becomes unreliable. Our formulation also relies on predicted depth and on the Sim(3) alignment between consecutive chunks, so errors in depth prediction can compound across chunks.

Acknowledgements. We thank the members of the Physical Perception Lab at CMU for their valuable discussions. This work was supported in part by NSF Award IIS-2345610. This work used Bridges-2 at Pittsburgh Supercomputing Center through allocation CIS251064 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. This work was supported by Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0074. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

## References

[1] Arjun Balasingam, Joseph Chandler, Chenning Li, Zhoutong Zhang, and Hari Balakrishnan. Drivetrack: A benchmark for long-range point tracking in real-world videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22488–22497, 2024.

[2] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In European conference on computer vision, pages 611–625. Springer, 2012.

[3] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. In International Conference on Learning Representations, pages 50694–50718, 2026.

[4] Zhongxiao Cong, Qitao Zhao, Minsik Jeon, and Shubham Tulsiani. Flow3r: Factored flow prediction for scalable visual geometry learning. arXiv preprint arXiv:2602.20157, 2026.

[5] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[6] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggt-long: Chunk it, loop it, align it–pushing vggt’s limits on kilometer-scale long rgb sequences. arXiv preprint arXiv:2507.16443, 2025.

[7] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022.

[8] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10061–10072, 2023.

[9] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8503–8513, 2025.

[10] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal ofrobotics research, 32(11):1231–1237, 2013.

[11] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022.

[12] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T Barron, Noah Snavely, and Aleksander Hołynski. Zipmap: Linear-time stateful 3d reconstruction via test-time training. In ´ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21748–21759, 2026.

[13] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023.

[14] Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6013–6022, 2025.

[15] Jay Karhade, Nikhil Keetha, Yuchen Zhang, Tanisha Gupta, Akash Sharma, Sebastian Scherer, and Deva Ramanan. Any4d: Unified feed-forward metric 4d reconstruction. arXiv preprint arXiv:2512.10935, 2025.

[16] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction. In International Conference on 3D Vision (3DV). IEEE, 2026.

[17] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Joao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. Advances in Neural Information Processing Systems, 37:82149–82165, 2024.

[18] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[19] Xinhang Liu, Yuxi Xiao, Donny Y Chen, Jiashi Feng, Yu-Wing Tai, Chi-Keung Tang, and Bingyi Kang. Trace anything: Representing any video in 4d via trajectory fields. arXiv preprint arXiv:2510.13802, 2025.

[20] Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, and Chen Change Loy. 4rc: 4d reconstruction via conditional querying anytime and anywhere. arXiv preprint arXiv:2602.10094, 2026.

[21] Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang. Delta: Dense efficient long-range 3d tracking for any video. arXiv preprint arXiv:2410.24211, 2024.

[22] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7855–7862. IEEE, 2019.

[23] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016.

[24] Shenhan Qian, Ganlin Zhang, Shangzhe Wu, and Daniel Cremers. Flow4r: Unifying 4d reconstruction and tracking with scene flow. arXiv preprint arXiv:2602.14021, 2026.

[25] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

[26] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021.

[27] Edgar Sucar, Eldar Insafutdinov, Zihang Lai, and Andrea Vedaldi. V-dpm: 4d video reconstruction with dynamic point maps. arXiv preprint arXiv:2601.09499, 2026.

[28] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020.

[29] Joachim Tesch, Giorgio Becherini, Prerana Achar, Anastasios Yiannakidis, Muhammed Kocabas, Priyanka Patel, and Michael J. Black. BEDLAM2.0: Synthetic humans and cameras in motion. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

[30] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on pattern analysis and machine intelligence, 13(4):376–380, 2002.

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[32] Bo Wang, Jian Li, Yang Yu, Li Liu, Zhenping Sun, and Dewen Hu. Scenetracker: Long-term scene flow estimation network. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[33] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[34] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In International Conference on Computer Vision, 2023.

[35] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9660–9672, 2025.

[36] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025.

[37] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 20697–20709, 2024.

[38] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22378–22389, 2024.

[39] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20406–20417, 2024.

[40] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: Advancing 3d point tracking with explicit camera motion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 6726–6737, 2025.

[41] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1790–1799, 2020.

[42] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

[43] Shuai Yuan, Yantai Yang, Xiaotian Yang, Xupeng Zhang, Zhonghao Zhao, Lingming Zhang, and Zhipeng Zhang. Infinitevggt: Visual geometry grounded transformer for endless streams. arXiv preprint arXiv:2601.02281, 2026.

[44] Bowei Zhang, Lei Ke, Adam W Harley, and Katerina Fragkiadaki. Tapip3d: Tracking any point in persistent 3d geometry. arXiv preprint arXiv:2504.14717, 2025.

[45] Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar, Joëlle K Barral, Raia Hadsell, et al. Efficiently reconstructing dynamic scenes one d4rt at a time. arXiv preprint arXiv:2512.08924, 2025.

[46] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024.

[47] Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, and Deqing Sun. Loger: Long-context geometric reconstruction with hybrid memory. arXiv preprint arXiv:2603.03269, 2026.

[48] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19855–19865, 2023.

[49] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025.

[50] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025.

## A Implementation Details

Training Details. Table 4 summarizes the training datasets and their sampling ratios. Omni-World [49] contains dynamic content but lacks trajectory ground truth; we restrict its queries to $t _ { \mathrm { s r c } } = t _ { \mathrm { t g t } }$ , reducing the task to depth and relative-pose estimation without requiring cross-frame trajectory labels.

For each frame, we sample $N = 7 5 0$ query pixels, with 40% drawn from edge regions detected by Sobel filtering to encourage coverage of object boundaries. The query indices $t _ { \mathrm { s r c } } , t _ { \mathrm { t g t } } ,$ and $t _ { \mathrm { { c a m } } }$ are sampled uniformly from the frame indices, with 40% of queries constrained to $t _ { \mathrm { c a m } } = t _ { \mathrm { t g t } }$ so the model frequently predicts points in the target frame’s own coordinate system.

We use AdamW with a peak learning rate of $1 \times 1 0 ^ { - 4 }$ for 150 epochs, linearly warmed up till 10 epoch and then cosine-decayed. During training, the learning rate for models initialized from DepthAnything3 is scaled by 0.1. Training is done on 8 H100 GPUs, and the loss for dynamic points is upweighted relative to static points. We apply color jittering, Gaussian blurring, random rescaling, and aspect-ratio augmentation throughout the training.

Table 4: Training datasets. Sampling ratio denotes the proportion of samples drawn per epoch. Dynamic datasets provide ground-truth trajectories; static datasets treat all points as stationary.
<table><tr><td>Dataset</td><td>Dynamic</td><td>Sampling Ratio</td></tr><tr><td>PointOdyssey [48]</td><td>√</td><td>19.8%</td></tr><tr><td>Dynamic Replica [13]</td><td>√</td><td>19.8%</td></tr><tr><td>BEDLAM2 [29]</td><td>√</td><td>19.8%</td></tr><tr><td>CoTracker Kubric [14]</td><td>√</td><td>11.9%</td></tr><tr><td>Kubric Movi-F [11]</td><td>√</td><td>11.9%</td></tr><tr><td>Waymo Drivetrack [1]</td><td>√</td><td>5.8%</td></tr><tr><td>OmniWorld [49]</td><td>√t</td><td>2.0%</td></tr><tr><td>ScanNet [5]</td><td>x</td><td>2.0%</td></tr><tr><td>ScanNet++ [42]</td><td>x</td><td>2.0%</td></tr><tr><td>BlendedMVS [41]</td><td>x</td><td>2.0%</td></tr><tr><td>Co3Dv2 [26]</td><td>x</td><td>2.0%</td></tr><tr><td>WildRGBD [38]</td><td>x</td><td>1.0%</td></tr></table>

<sup>†</sup> Dynamic content but no trajectory GT; queries restricted to $t _ { \mathrm { s r c } } = t _ { \mathrm { t g t } } .$

Evaluation Details. For long-video 4D Tracking (Table 1), we evaluated both static and dynamic points, aligning ground-truth and predicted trajectories with a single global scale per sequence. We include static points because they are the ones that most often leave the field of view during chaining, and thus directly measure tracking accuracy for occluded and out-of-frame points. Results on dynamic points only are reported in Appendix E. For single-chunk tracking (Table 2), we evaluate dynamic points only and follow the protocol of Any4D [15], which rescales each frame pair independently.

## B Runtime Analysis

We measure the runtime of each method, scaling either the number of input frames or the number of query points at the first frame. When scaling the number of frames (16, 32, 48, 64), the number of queries are fixed to 100, and when scaling the number of queries (100 to 5,000) the number of frame is fixed to 48. All measurements are taken on a single A6000 (48 GB) at an input resolution of $2 9 4 \times 5 1 8$

Results are shown in Figure 7 on a log scale. When scaling the number of input frames, Point4D is the second-fastest method at every frame count. When scaling the number of queries, the runtime of Point4D grows, whereas other feed-forward methods remain constant: their DPT-based dense decoders predict every pixel regardless of how many points are queried. This growth is a property of query-based decoding in general, not of 3D queries specifically, and it is precisely what makes it flexible and sparse tracking cheap: querying 100 points costs proportionally little, while dense decoders pay their full cost either way. Finally, SpatialTrackerV2 [40] and TAPIP3D [44] are the slowest methods after VDPM [27], and SpatialTrackerV2 runs out of memory at 5,000 queries.

![](images/8433d2ffaf1ac434e944f5193fb29b1a1b89fec2bcadce957e42c2b003c09870.jpg)  
Figure 7: Runtime Comparison. Inference time vs. the number of input frames (left) and query points (right). Point4D is the second fastest with sparse queries; with dense queries it falls behind the dense DPT-head methods but still completes, where SpatialTrackV2 runs out of memory.

## C Video depth and camera pose estimation.

We evaluate video depth and camera pose estimation on Sintel [2], Bonn [22], and KITTI [10]. Depth accuracy is measured by absolute relative error (AbsRel, ↓) and the inlier ratio $\delta < 1 . 2 5 \left( \uparrow \right)$ . Camera pose is assessed using Absolute Trajectory Error (ATE), translational Relative Pose Error (RPE ), and rotational Relative Pose Error (RPE<sub>r</sub>), computed after global SE(3) alignment. As shown in Table 5, Point4D performs on par with DA3, while outperforming other feed-forward 4D reconstruction methods on most metrics. This indicates that Point4D , built on the DA3 backbone, maintains accurate scene geometry understanding — important for ensuring that the 3D queries used in motion reconstruction are reliably initialized from predicted depth.

<table><tr><td></td><td colspan="6">Video Depth Estimation</td><td colspan="6">Camera Pose Estimation</td></tr><tr><td></td><td colspan="2">KITTI [10]</td><td colspan="2">Bonn [22]</td><td colspan="2">Sintel [2]</td><td colspan="3">Bonn [22]</td><td colspan="3">Sintel [2]</td></tr><tr><td>Method</td><td>AbsRel</td><td> $\delta < 1 . 2 5$ </td><td>AbsRel</td><td> $\delta < 1 . 2 5$ </td><td>AbsRel</td><td> $\delta < 1 . 2 5$ </td><td>ATE</td><td>RPEt</td><td>RPET</td><td>ATE</td><td>RPEt</td><td>RPEr</td></tr><tr><td>DA3</td><td>0.053</td><td>0.976</td><td>0.069</td><td>0.967</td><td>0.238</td><td>0.655</td><td>0.029</td><td>0.011</td><td>0.667</td><td>0.124</td><td>0.053</td><td>0.479</td></tr><tr><td>TraceAnything</td><td>0.1055</td><td>0.903</td><td>6.964</td><td>0.462</td><td>0.506</td><td>0.409</td><td>0.040</td><td>0.015</td><td>43.280</td><td>0.499</td><td>0.322</td><td>10.801</td></tr><tr><td>Any4D</td><td>0.090</td><td>0.939</td><td>0.414</td><td>0.600</td><td>0.670</td><td>0.384</td><td>0.059</td><td>0.023</td><td>0.632</td><td>0.619</td><td>0.241</td><td>0.741</td></tr><tr><td>4RC</td><td>0.051</td><td>0.960</td><td>0.076</td><td>0.910</td><td>0.483</td><td>0.581</td><td>0.030</td><td>0.009</td><td>0.645</td><td>0.397</td><td>0.190</td><td>0.654</td></tr><tr><td>VDPM</td><td>0.068</td><td>0.945</td><td>0.086</td><td>0.925</td><td>0.239</td><td>0.651</td><td>0.028</td><td>0.009</td><td>0.664</td><td>0.133</td><td>0.063</td><td>0.508</td></tr><tr><td>Point4D</td><td>0.051</td><td>0.978</td><td>0.072</td><td>0.934</td><td>0.202</td><td>0.707</td><td>0.029</td><td>0.010</td><td>0.588</td><td>0.114</td><td>0.049</td><td>0.470</td></tr></table>

Table 5: Video depth and camera pose estimation. Point4D achieves state-of-the-art accuracy among feed-forward 4D reconstruction methods. We report AbsRel (↓) and $\delta < 1 . 2 5 \left( \uparrow \right)$ for depth, and ATE (↓), RPE (↓), RPE (↓) for pose on KITTI, Bonn, and Sintel.

## D Chaining Accuracy by Visibility at the Handoff Frame

Chaining requires re-acquiring each point where two chunks meet, and how difficult this is depends on whether the point is visible there. We therefore break down EPE by the visibility state at the handoff frame: Visible (the point projects into the frame and is unoccluded), Occluded (it projects into the frame but is hidden by another object), and OOF (it projects outside the frame boundary). We first identify the strongest handoff strategy for 2D-query baselines, and then compare 2D and 3D re-querying under each condition.

Choosing the handoff strategy for 2D-query baselines. Since 2D-query methods require projecting predicted 3D points to pixel coordinates for re-querying, the choice of handoff strategy affects chaining quality. Table 6 ablates four strategies using 4RC as the base method across three datasets. Each strategy decides, from the trajectory predicted in the current chunk, where to re-query the point in the next chunk. First-frame projects all points to the first overlapping frame and clips out-of-frame projections to the image boundary. Stationary and Linear apply the same projection but extrapolate out-of-frame points as stationary or with constant velocity, respectively. Selection projects each point at every overlapping frame, selects the frame where the projected depth best matches the depth head’s prediction, and re-queries at that frame, which reduces occlusion-related errors by choosing the frame where the point is most likely visible. Selection attains the lowest EPE overall on all three datasets, and we therefore use it for all 2D-query baselines.

Table 6: Chaining strategy ablation for 2D-query chaining (4RC), EPE by visibility. Persequence EPE averaged over sequences (regime columns average over sequences containing that regime). Selection-based chaining achieves the lowest EPE on Visible, Occluded, and overall across all datasets.
<table><tr><td></td><td colspan="4">PointOdyssey</td><td colspan="4">DynamicReplica</td><td colspan="4">PStudio</td></tr><tr><td>Chaining</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td></tr><tr><td>First-frame</td><td>0.845</td><td>1.105</td><td>1.198</td><td>0.960</td><td>0.318</td><td>0.598</td><td>0.340</td><td>0.361</td><td>0.423</td><td>0.531</td><td>0.644</td><td>0.454</td></tr><tr><td>Stationary</td><td>0.822</td><td>1.074</td><td>0.758</td><td>0.891</td><td>0.322</td><td>0.597</td><td>0.260</td><td>0.357</td><td>0.421</td><td>0.525</td><td>0.597</td><td>0.451</td></tr><tr><td>Linear</td><td>0.831</td><td>1.087</td><td>0.810</td><td>0.907</td><td>0.325</td><td>0.600</td><td>0.278</td><td>0.358</td><td>0.435</td><td>0.541</td><td>0.619</td><td>0.466</td></tr><tr><td>Selection</td><td>0.672</td><td>0.915</td><td>1.097</td><td>0.789</td><td>0.290</td><td>0.574</td><td>0.335</td><td>0.336</td><td>0.354</td><td>0.443</td><td>0.610</td><td>0.379</td></tr></table>

Table 7: EPE by visibility across datasets. Point4D achieves the lowest EPE in nearly every visibility, with the largest margins on Occluded points, where 2D re-querying fails but 3D re-querying carries the point through directly.
<table><tr><td></td><td colspan="4">PointOdyssey</td><td colspan="4">DynamicReplica</td><td colspan="4">PStudio</td></tr><tr><td>Method</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td><td>Vis.</td><td>Occ.</td><td>OOF</td><td>All</td></tr><tr><td>TraceAnything</td><td>2.049</td><td>2.258</td><td>2.748</td><td>2.147</td><td>0.728</td><td>0.856</td><td>1.203</td><td>0.767</td><td>0.635</td><td>0.744</td><td>1.187</td><td>0.665</td></tr><tr><td>Any4D</td><td>0.893</td><td>1.190</td><td>1.377</td><td>1.026</td><td>0.305</td><td>0.635</td><td>0.509</td><td>0.364</td><td>0.474</td><td>0.582</td><td>0.802</td><td>0.497</td></tr><tr><td>4RC</td><td>0.672</td><td>0.915</td><td>1.097</td><td>0.789</td><td>0.290</td><td>0.574</td><td>0.335</td><td>0.336</td><td>0.354</td><td>0.443</td><td>0.610</td><td>0.379</td></tr><tr><td>VDPM</td><td>0.636</td><td>0.860</td><td>1.171</td><td>0.736</td><td>0.346</td><td>0.607</td><td>0.379</td><td>0.386</td><td>0.261</td><td>0.332</td><td>0.493</td><td>0.280</td></tr><tr><td>Point4D</td><td>0.548</td><td>0.597</td><td>1.147</td><td>0.616</td><td>0.140</td><td>0.202</td><td>0.309</td><td>0.155</td><td>0.228</td><td>0.265</td><td>0.409</td><td>0.236</td></tr></table>

3D vs. 2D re-querying across visibility regimes. Table 7 compares Point4D with 2D-query baselines (all using Selection chaining) broken down by visibility regime. Point4D achieves lower EPE on visible and occluded points on every dataset, with the largest margin on occluded points. This indicates that even when a 2D method re-queries at the best overlapping frame they still cannot successfully chain occluded points, whereas a 3D query carries it through the occlusion directly. For out-of-frame points the advantage narrows, and on PointOdyssey 4RC is better. This is expected: some OOF points the next chunk never observes leave no trace in its scene representation, while 2D methods clip to the image boundary and re-query a wrong surface that can still yield a smaller error.

## E Additional Quantitative Results on Long-Video Tracking

4D tracking results on dynamic points. Table 1 evaluates all query points, including those on static background, whose trajectories are largely explained by camera motion alone. To verify that Point4D remains effective on points that actually move, we repeat the evaluation using only dynamic points. PStudio is excluded, as its ground-truth trajectories are annotated only for dynamic points and its numbers are therefore identical. As shown in Table 8, Point4D outperforms every feed-forward method on all metrics and attains the best average rank among all compared methods, showing that 3D-query chaining extends trajectories reliably even when the queried points are in motion.

4D tracking results on longer sequences. To assess how each method behaves under more chaining steps, we extend the evaluation to 500 frames on PointOdyssey and 300 frames on Dynamic Replica (the full length of its sequences), keeping the chunk size 48 and overlap 8; PStudio is excluded as it is already evaluated at its full length of 150 frames. The results are shown in Table 9. Point4D ranks first on every metric, surpasses all iterative trackers and feed-forward 4D models at longer horizons. This demonstrates the robustness of our method for long video tracking.

Additional chunk-wise tracking accuracy analysis. Complementing the chunk-wise robustness analysis in Figure 6, we report the full results on PointOdyssey and Dynamic Replica in Figure 8. On both datasets, the accuracy of Point4D remains stable across chunks, while the accuracy of feed-forward methods which rely on 2D re-querying degrades as chaining proceeds.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">PointOdyssey [48]</td><td colspan="3">Dynamic Replica [13]</td></tr><tr><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td></tr><tr><td rowspan="2">te</td><td>TAPIP3D</td><td>0.779</td><td>0.470</td><td>0.353</td><td>0.251</td><td>0.728</td><td>0.614</td></tr><tr><td>SpatialTrackV2</td><td>0.493</td><td>0.589</td><td>0.425</td><td>0.240</td><td>0.755</td><td>0.636</td></tr><tr><td rowspan="4">Re-ard</td><td>TraceAnything</td><td>1.863</td><td>0.120</td><td>0.070</td><td>0.722</td><td>0.462</td><td>0.361</td></tr><tr><td>Any4D</td><td>1.127</td><td>0.276</td><td>0.170</td><td>0.431</td><td>0.614</td><td>0.505</td></tr><tr><td>4RC</td><td>0.879</td><td>0.445</td><td>0.355</td><td>0.359</td><td>0.698</td><td>0.598</td></tr><tr><td>VDPM</td><td>0.784</td><td>0.467</td><td>0.373</td><td>0.386</td><td>0.660</td><td>0.557</td></tr><tr><td></td><td>Point4D</td><td>0.617</td><td>0.566</td><td>0.474</td><td>0.200</td><td>0.794</td><td>0.717</td></tr></table>

Table 8: Long-Video 4D Tracking on Dynamic Points Only. Point4D achieves the best average rank among all compared methods, outperforming every other feed-forward methods. We report metrics over dynamic points only, on sequences of 200 frames partitioned into chunks of 48 frames with 8-frame overlap. PStudio is omitted as it evaluates only dynamic queries, making the results identical to Table 1. Red , Orange , and Yellow indicate the top three results.

<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="3">PointOdyssey [48] (500 frames)</td><td colspan="3">Dynamic Replica [13] (300 frames)</td></tr><tr><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td><td>EPE↓</td><td>APD↑</td><td>Survival ↑</td></tr><tr><td rowspan="3">Iter</td><td>TAPIP3D</td><td>1.020</td><td>0.421</td><td>0.292</td><td>0.198</td><td>0.797</td><td>0.725</td></tr><tr><td>SpatialTrackV2</td><td>1.008</td><td>0.461</td><td>0.307</td><td>0.251</td><td>0.749</td><td>0.644</td></tr><tr><td>4RC</td><td>1.597</td><td>0.338</td><td>0.228</td><td>0.418</td><td>0.676</td><td>0.582</td></tr><tr><td rowspan="3">EE.</td><td>VDPM</td><td>1.458</td><td>0.318</td><td>0.213</td><td>0.478</td><td>0.610</td><td>0.513</td></tr><tr><td>Point4D</td><td>0.972</td><td>0.482</td><td>0.387</td><td>0.174</td><td>0.836</td><td>0.786</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Long-Video 4D Tracking on Extended Sequences. Point4D ranks first on every metric, surpassing not only feed-forward baselines but also iterative trackers. Chunking follows the same setting (48-frame chunks with 8-frame overlap), so longer sequences require more handoffs: 12 for PointOdyssey and 7 for Dynamic Replica. Red , Orange , and Yellow indicate the top three results.

## F Additional Qualitative Results on Long-Video Tracking

We provide additional tracking results for long videos, where the number of frames varies between 100 and 500 in Figure 9 and Figure 10. Each chunk has size 48 with 16 overlapping frames. Point4D show robust trajectory prediction across long videos, showing its effectiveness on trajectory chaining.

![](images/d80512d4d6ff1f7052e2fffddb7e72952286ff8b24b749edbb2ac524963625b5.jpg)

![](images/54e3e77feabb4264f4f2ad05473b9a1ce482c521d4526a700c2c0bb2df8fae49.jpg)

![](images/12cf64fd21e47b97c04abac2d434fb552020c013a2dbf5f780afd45e725c8d69.jpg)

![](images/6860c466c4702ff869c4b63e171bf291749ae357ceecaac4d07dcf25f22def82.jpg)  
Figure 8: Chunk-wise tracking accuracy on (Left) Dynamic Replica and (Right) PointOdyssey. On both datasets, the accuracy of Point4D (red) degrades more slowly than that of other methods as chaining proceeds, demonstrating the robustness of 3D-query trajectory chaining.

![](images/150bf2b92bb92cc1e8cc840ef4253439ffd6c8b1000659df9b82664953f75b53.jpg)

![](images/e97e63960f35e3433051f60b8345a67ac82917b780f3ed405ff8331cb15d98d9.jpg)

![](images/a407b473efdfc2839a29324f4f4abcbf9f01623ea97199a317cc3358612c2154.jpg)

![](images/3bce3a709960fc563471d0d72755e780377fdfeb531698bf3d1f369f78663913.jpg)

![](images/5dec628fe7d5b54c26045f8f876dbe7a01f24f32a3b7c7bd8e2338d4c0f2f502.jpg)

![](images/6962d16bacb18142e12a92c25598fe257041f494a09228311510f2105822e02b.jpg)

![](images/7c8873855f1899ade2f4e04331c2c3df61cbe77ceb4f3d5260350d6f04ba7334.jpg)

![](images/d1a6d4528b3c59f7cbd6ae58c4add9c6af12dff579388ae2ed0a8d956f4fa6dc.jpg)

![](images/9576d28e5de24fed8ebcedb89b775c3cd7233c3099b1a62b64d3cd764de6f53a.jpg)

![](images/1f3664816006c0845122676051dd44b76e79e9bebeaa214b25a43acdea22ae83.jpg)

![](images/404c0a5cee3f3d953fcfb7f3a22e6d52779fbc50aae6650294227502c89104ef.jpg)

![](images/37b9db59ddaada7422851f38e9282affa2e70c0e272bba9446e90d72a79f926d.jpg)

![](images/0afcf07aa72109bbaf50ccbbff37cb80188726d874c4951722e53c202b76d873.jpg)

![](images/cc2eb150acc8f80309c00213f5471cfa5941d29d56d323cad4ed5efa1163175c.jpg)

![](images/b2bc6f3ecad21134e8265f05c4c083df4a4f237c718f4a1d25c8992c87c045fa.jpg)

![](images/2524c1f0eb4d394b17f902b3d9eaf6b0d1fc357b2fbffb112e1115129f99c340.jpg)

![](images/2200b35f9d919fef8e2a6d599459409a76f41af619954740240a69d994d1fd8d.jpg)  
Input Video

![](images/9358b0d32cce464dba1d9158337bbf1549e0e9109bfc2e4ee7676a7bd5dca69e.jpg)  
SpatialTrackerV2

![](images/661ba1effbdb01ada352e01102c74c32ba553b3dd97ec3565d56de379b2af516.jpg)  
4RC

![](images/09520c33bd9e481f9ab91cf8f8b5c87dc9ab31163787320f03c205e7a64dfb12.jpg)  
Point4D

Figure 9: Additional results for long-video tracking via trajectory chaining. Point4D produces consistent 3D trajectories across long videos.

![](images/1e782982fff898ced42895a09c213052a0842594460282be04f723c241941d4b.jpg)  
Figure 10: Additional results for long-video tracking via trajectory chaining. Point4D produces consistent 3D trajectories across long videos.