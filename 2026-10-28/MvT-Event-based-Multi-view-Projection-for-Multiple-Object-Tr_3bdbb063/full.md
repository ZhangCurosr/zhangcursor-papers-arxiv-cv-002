# MvT: Event-based Multi-view Projection for Multiple Object Tracking

Muxi Zha, Banglei Guan, Minzu Liang, Zibin Liu, Yang Shang, Qifeng Yu, and Laurent Kneip

Abstract—Event cameras are increasingly used for Multiple Object Tracking (MOT), but their asynchronous event output often requires specialized methods. Existing processing methods primarily follow two paradigms, pseudo-frames and event-byevent. The former is the prevailing approach since its data format aligns with images, making image-based techniques applicable. However, it suffers from tracking failures when trajectories overlap or are spatially close on pseudo-frames. Facing this challenge, we propose a multi-view pipeline, Multi-view Tracking (MvT), which preserves the 2D data format to leverage imagebased techniques directly while introducing additional spatiotemporal views to resolve tracking ambiguities in a single view. MvT comprises a Multi-view Projection (MvP) module and a Multi-view Fusion (MvF) stage. MvP encodes events into three complementary spatio-temporal views while mitigating the pattern discretization. Within MvF, multi-view results are unified into a 3D coordinate system, and tracklets are associated through an optimization model subject to specific criteria combination. Evaluations on four datasets, including our self-collected Small Objects Dataset (SOD), show that MvT seamlessly integrates image-based methods and outperforms existing non-learning and learning trackers in generalized scenarios, and effectively resolves the single-view tracking ambiguities. Being training-free, MvT is applicable when ground-truth annotation is infeasible, thereby highlighting its practical, data-efficient potential. Code is available at https://github.com/zhazhabiu/MvTracking.

Index Terms—Event-based tracking, multiple object tracking, multi-view representation

## I. INTRODUCTION

ULTIPLE Object Tracking (MOT) serves as a funestimation [1], [2] and autonomous driving assistance [3]. Image-based trackers excel in precision but struggle under challenging conditions (e.g., ground-based UAV observation). Event cameras offer a promising alternative [4] with high dynamic range, low latency, and fast response to stimuli in these extreme environments. However, their asynchronous event output is fundamentally different from images, as shown in Fig. 1(a). Specialized event-based tracking methods have been developed [5]–[7], but they mainly leverage the high dynamic range and low latency of event cameras to achieve real-time tracking. With the special motion sensitivity of event cameras, this paper focuses on small object tracking, which is more challenging than standard MOT due to issues such as small sizes and similar appearance features.

![](images/76b0fd47af5263591162188dffd2f1f884d1dbd95618b68c4ba5cbe2ef7c1fd5.jpg)  
Fig. 1. Comparison of different event processing approaches. (a) Asynchronous event streams of the Raindrops sequence in SOD. (b)-(d) The first row shows data representations (0.3s period) of pseudo-frames (event images [8] here), event-by-event processing, and our multi-view representation. The second row shows the corresponding tracking results from: our singleview framework variant with region growing, E-MS [7], and the original MvT+RG. Different colors indicate distinct trajectories.

Each output event contains four elements, spatial coordinates (x, y), timestamps t, and polarity p. Current event-based tracking methods often process them in two formats, pseudoframes [8], [9] and event-by-event [7], [10], as depicted in Fig. 1(b) and (c). The first approach generates pseudoframes from events, including event images, voxel-grids, time surfaces, etc. Since this format aligns with images, which are intuitive and convenient to process, the majority of eventbased tracking works [5], [6], [11], [12] are built upon it. Compared to real images, most pseudo-frame representations lack dense color and texture information, which serves as critical appearance cues for tracking. Although some studies [13], [14] have successfully utilized event polarity, which encodes the luminance gradient fields, to recover texture and intensity images, commonly used event images lack such rich cues.

![](images/2228fbd8ab4a3b5ee45e9c2262d656ac4f8c811b19c3cc3f4acc2d9e0e1682e0.jpg)  
Fig. 2. Overview of our MvT pipeline and visualization of main stages, taking an image-based segmentation method as an example. (a) An event stream is divided into blocks using a sliding window and then fed into the processing stages one by one. (b) and (d) detail the key components of MvT, MvP and MvF, respectively. MvP generates multiple views through time-realigned events. In MvF, complete trajectories are recovered from the segmentation results of multiple views. (c) denotes the image-based segmentation stage.

Furthermore, they simply collapse the temporal dimension of events, which can lead to tracking failures when object trajectories overlap or intersect in the image plane.

Event-by-event processing preserves the original sequential temporal nature of events. Beyond emerging Spiking Neural Networks (SNN) [12], [15], clustering techniques [7], [16]– [18] are used widely since they are training-free. These methods regard events as point clouds distributed in t-x-y space. However, unlike standard 3D point clouds with homogeneous axis units in z-x-y, the heterogeneous units of t-x-y axes complicate parameter selections. Specifically, first, the high temporal resolution and asynchronous nature cause a relatively sparser distribution along t-axis compared to the spatial dimensions, thus requiring disparate clustering parameters for different axes. Second, the temporal distribution of an event trajectory is velocity-dependent, rendering it hard to determine parameters for objects with diverse motion patterns. Though this format avoids the tracking ambiguities in above pseudoframes, these issues limit its generalizability.

Given the processing ease and straightforward parameter determination of pseudo-frames, alongside the rich spatiotemporal information of event-by-event processing for trajectory disambiguation, an open problem arises. How can the tracking ambiguities inherent in single pseudo-frames be resolved? Considering the difficulty in obtaining ground truth annotations from events, we move away from prevailing supervised network designs.

In this paper, we provide a novel generalizable framework MvT, to integrate existing image-based methods into eventbased tracking task. To ensure compatibility with image-based methods, MvT preserves the pseudo-frame format. The overall pipeline is shown in Fig. 2. It encodes events into a multi-view representation through its component, MvP. The generated views include a spatial view, XOY, and two spatio-temporal views, YOT and TOX. This representation overcomes the information limitation of a single pseudo-frame. Image-based detection/segmentation methods are then applied to views. The other component of MvT, named MvF, fuses results from each view and maps them back into the original t-xy coordinates. If there are detection or segmentation errors in some views, MvF can leverage complementary information provided by other views to correct results through an isolation scheme. Subsequently, an association stage is performed on the corrected results. To accommodate the modality of event data, we design three association criteria based on the motion consistency of event trajectories.

In experiments, we integrate both non-learning and learningbased methods into MvT, and compare them against carefully devised pseudo-frame and event-by-event methods. Evaluations are conducted on four datasets, including Event Camera Dataset (ECD) [19], Bee Swarm Dataset (BEES) [20], EV-UAV [21], and our self-collected Small Objects Dataset (SOD). Except for ECD, the remaining three datasets cover small-sized objects and various motion patterns. The main contributions of this paper are as follows:

• We propose MvT, a generalized and training-free eventbased MOT framework that enables the direct application of image-based methods while preserving the temporal information of events for tracking.

• We devise an association model comprising three criteria, corresponding to three cases derived from short-term and long-term event trajectory relationships. These criteria can be flexibly combined to handle diverse association scenarios involving both normal-sized and small-sized objects.

• Extensive experiments validate the effectiveness and compatibility of MvT with various image-based detection and segmentation methods, as well as the efficacy of the multi-view representation.

## II. RELATED WORK

We provide a review of event-based tracking methods, covering both pseudo-frames and event-by-event paradigms. We then introduce the multi-view representation in image processing and spike-based coding, which collectively inspire the design of our event representation.

## A. Pseudo-frame Tracking

Pseudo-frames refer to 2D images constructed from events, such as event images [8], Image of Warped Events (IWE) [9], and voxel-grids [15], [22], with IWE being a well-known form that accumulates events warped by motion models. This type of frame is typically processed using maximization optimization, such as Entropy Maximization (EM) [23], [24], Expectation Maximization-Iterative Closest Point (EM-ICP) [25], Contrast Maximization (CM) [8], [9], [26]–[28], and Spatiotemporal Consistency Maximization [29]. For example, Zhu et al. [25] introduce an optical flow method based on Expectation Maximization that maximizes all correlation expectations of soft data associations for feature tracking. In other pseudoframe-based works, Zhang et al. [22] design cross-domain attention schemes that combine the information from RGB frames and voxel-grid-based event frames to guide object tracking. Zhang et al. [15] then generate positive-negative event frames and feed them into a transformer module for global spatial information, as well as a spiking neural network (SNN) for temporal clues. With these clues, a single object is accurately detected and tracked. Since event images cannot fully utilize temporal information, researchers have explored diverse pseudo-frames that integrate more temporal information. For example, the Surface of Active Events (SAE) [30] stores timestamps as pixel values and has been used to estimate optical flow. Subsequently, a Time-Surface map, which stores exponentially decayed timestamps, was introduced as a prototype of event-based object recognition tasks [31] and has since been extended as a popular representation in motion estimation [32], [33]. Chen et al. [5], [6] propose an Adaptive Time-Surface with Linear Time Decay (ATSLTD) frame representation and a Time-Surface with Linear Time Decay (TSLTD) representation. Almareafi et al. [34] introduce a prototype target texture that captures temporal distance variations in events. These representations retain temporal information by encoding timestamps into pixel values, but they may lose other details, like the number of triggers at each pixel. Mitrokhin et al. [35] propose using a Time-image to assist event images in jointly estimating motions.

## B. Event-by-event Tracking

To fully leverage the low latency of event cameras, many asynchronous processing methods bypass frame generation, categorizing object tracking into one-stage and two-stage approaches. In one-stage methods, deep learning models, like Vision Transformers (ViT) [36] and SNNs serve as a supervised solution mainly for Single Object Tracking (SOT) [12], [15], while unsupervised clustering offers a more general alternative applicable to MOT without extensive training [10], [16], [17]. However, varying target speeds cause unstable performance due to non-fixed clustering criteria. Two-stage pipelines separate detection and tracking. The detection stage utilizes not only clustering methods [7], [18], [37] but also some visual feature-based methods [38]–[40]. The tracking stage associates tracklets across batches similar to image-based tracking. Several methods have been developed, including searching features or templates in the neighborhood [37], [40], matching target descriptors across adjacent time steps [18], and employing Kalman filters [7], [38], [41].

![](images/d2fc8eaa42236520027c5813785f3b11ea4d3e6b0959fcf6750c91976461b846.jpg)  
Fig. 3. Illustrations of challenges in current event-based tracking paradigms. (a) In pseudo-frame representations, overlapping trajectories (green and blue) may lead to erroneous tracking. Arrows indicate the direction of motion. (b) In the TOX plane, each trajectory consists of a series of discrete events. Objects moving at different velocities (red, blue, and dark gray) generate trajectories with distinct temporal distributions, which necessitate different processing parameters.

## C. Multi-view based works

Event data can be regarded as point clouds in the T-X-Y space. In processing point clouds, leveraging multiple views is a common strategy to obtain more information than a single view provides. “Multiple views” typically carries two meanings, views at different timestamps (video sequences) and views from different perspectives. In our work, we focus on the latter, which is more relevant to event data. Among multiview forms, orthographic views are particularly effective, being used for reconstruction via geometric constraints [42], and for classification using morphological information [43]. This principle persists in modern neural representations. For example, the Triplane representation in Neural Radiance Fields (NeRF) [44], [45] uses three 2D feature planes, facilitating the application of 2D convolutional networks. Multi-view representation is also used in spike-based coding. Thorpe et al. [46] propose Rank Order Coding, which involves projecting spatiotemporal data onto three 2D planes (X-T, Y-T, and X-Y). This projection scheme has been explored in neuroscience [47] and applied to image processing [48], establishing a foundational method for converting temporal event streams into informative 2D maps.

## III. METHOD

For a better understanding of this work, we start with a motivation analysis in Sec. III-A to guide the design of our framework. A brief introduction to MvT is provided in Sec. III-B, followed by detailed information about MvP (Sec. III-C) and MvF (Sec. III-D).

## A. Motivation

While image-based detection or segmentation methods have demonstrated remarkable success, their direct application to event-based tracking remains challenging due to modality shift. As analyzed in Sec. I, existing event-based tracking approaches face two primary challenges. (1) Single pseudoframe representations contain limited spatio-temporal information. When objects overlap or their trajectories intersect in the 2D projection, these representations fail to distinguish them (Fig. 3(a)). (2) The spatio-temporal units (t-x-y) are not homogeneous. The inherent mismatch between temporal (µs) and spatial (pixel) units, combined with the asynchronous triggering mechanism, results in sparse and non-uniform event distributions. As illustrated in Fig. 3(b), varying target speeds lead to distinctly different event densities (total event counts per unit of recording period) along trajectories, thereby posing difficulties for parameter selection. We summarize this as Claim 1. The proof is provided in the supplementary material.

![](images/6f86a7c7c65134b27bef4b2b883466b2ac54f60094da2ece5ff4e2fd6b6ea6b0.jpg)  
Fig. 4. Pipeline of the proposed MvP. The raw event data first undergoes time realignment to compress temporal gaps. The realigned events are then projected onto three orthogonal planes (YOT, XOY, and TOX). Finally, each view is refined through a Gaussian pyramid to enhance structural continuity, producing th final multi-view representation. The projections are grayscale, and we binarize them only for readability.

Moreover, although the high temporal resolution of event cameras enables tracking to be treated as a trajectory segmentation problem, it introduces a side effect, discretization. From a macroscopic perspective, trajectories can be regarded as continuous 3D lines or volumes. However, at the micro-time scale, they comprise discretely distributed events (Fig. 3(b)). In this work, we consider both temporal discretization resulting from sparse event triggering and spatial discretization caused by the physical artifact of pixel-based sensor arrays.

For the first challenge, our multi-view representation adopts the 2D view format that enables the application of image-based algorithms while using multiple views to preserve essential spatio-temporal information for trajectory discrimination. As for the second challenge and the discretization issue, we introduce a time realignment method designed for asynchronously triggered events, alongside a Gaussian pyramid, to alleviate the tracklet pattern discontinuities caused by spatio-temporal discretization.

Claim 1. Events are triggered asynchronously, and under constant illumination, the same object moving at different speeds in the image plane will result in proportionally different local event densities.

## B. MvT Architecture

MvT processes raw events (i.e. Dynamic Vision Sensor signals) and outputs event trajectories rather than bounding boxes. Input event streams are partitioned into non-overlapping batches based on a fixed event count or a fixed time window.

Each batch is first encoded by MvP to generate the multi-view representation. Results obtained by image-based methods are then passed to MvF for complete trajectory recovery. Since views produced by MvP are independent, the image-based algorithm can be executed in parallel for efficiency.

## C. MvP

MvP projects an event batch onto three orthogonal planes in the 3D Cartesian coordinate system, XOY, YOT, and TOX, as illustrated in Fig. 4. Prior works in three-view reconstruction [42] and spike-based coding [46] demonstrate that those three views preserve sufficient information for recovering 3D structures. Despite reducing dimensionality to 2D, our representation retains essential event spatio-temporal information.

Ignoring polarities, each event is represented as $e \quad =$ $( t , x , y )$ , where $t , \ x , \ y$ denote the raw timestamp and 2D locations, respectively. For a batch $\Phi = \{ e _ { i } = ( t _ { i } , x _ { i } , y _ { i } ) | t _ { i } \in$ $[ t _ { 0 } , t _ { 0 } + B ] \}$ within a small window $[ t _ { 0 } , t _ { 0 } + B ]$ , its timestamp set is $T = \{ t _ { i } | e _ { i } \in \Phi \}$ and the range ran(T) of $T$ is

$$
\mathrm { r a n } ( T ) = [ \mathrm { m i n } ( T ) , \mathrm { m a x } ( T ) ] ,\tag{1}
$$

where min(·) denotes the minimum, and max(·) denotes the maximum. Taking YOT as an example, the projected event count view is generated by:

$$
\operatorname { Y O T } ( y , \ t ) = \sum _ { e _ { i } \in \Phi } \delta ( y - y _ { i } , \ t - t _ { i } ) ,\tag{2}
$$

where $\mathrm { Y O T } \subset \mathbb { R } ^ { H \times | \mathrm { r a n } ( T ) | } \ = \ \mathbb { R } ^ { H \times ( \operatorname* { m a x } ( T ) - \operatorname* { m i n } ( T ) + 1 ) }$ . H denotes the height of spatial resolution. | · | represents the cardinality. $\delta ( \cdot )$ is the Dirac delta function.

Time realignment. To mitigate temporal discretization, we apply time realignment via a time warping function g(·)

$$
\begin{array} { r l } & { \hat { \Phi } = \{ \hat { e } _ { i } = ( \hat { t } _ { i } , x _ { i } , y _ { i } ) | \forall \hat { t } _ { i } \in \hat { T } \} , } \\ & { \hat { T } = g ( T ) \triangleq \{ i | \forall t _ { i } \in T , i \in \mathbb { N } \} , } \end{array}\tag{3}
$$

where $\hat { \Phi }$ is the time-realigned event set, and $\hat { T }$ is the set of realigned timestamps t<sup>ˆ</sup>.

This process eliminates temporal gaps by renumbering timestamps sequentially from zero, as illustrated in Fig. 4. By discarding redundant “blank” timestamps where no events are triggered, it compacts the timeline into a series of pseudotimestamps. Therefore, $g ( \cdot )$ is an inhomogeneous warping function that removes the gaps between timestamps. Since $g ( \cdot )$ preserves order, it is also a monotonic function. This ensures that the mapping between $T$ and $\hat { T }$ is bijective, which is a prerequisite for timestamp recovery in the subsequent event trajectories.

The range of $\hat { T }$ is ran $( \hat { T } )$ :

$$
\mathrm { r a n } ( \hat { T } ) = [ 0 , | T | - 1 ] .\tag{4}
$$

The realigned projection $\mathrm { Y } \hat { \mathrm { O T } } \subset \mathbb { R } ^ { H \times | \mathrm { r a n } ( \hat { T } ) | }$ achieves dimensional reduction since $| \mathrm { r a n } ( \hat { T } ) | = | T | \leq | \mathrm { r a n } ( T ) |$ . Owing to the simple implementation and monotonic characteristic, this adaptive time realignment can be seamlessly integrated into the voxel-grid-based representation for further temporal scale [22].

The quality of realigned timestamps is affected by two primary factors, multiple object velocities and noise that often refers to background activity (BA) events [49]–[51]. Regarding the former, when objects exhibit varying planar velocities, the trajectories of slow-moving objects tend to be more fragmented and sparse along the temporal axis than those of fast-moving objects. However, the association model proposed in Sec. III-D can handle this fragmentation. With respect to the latter, BA noise introduces two distinct problems. First, it can distort the realigned timestamps away from their ideal (noise-free) values, leading to trajectory fragmentation. This issue is similarly addressed by the association model. Second, BA events may be wrongly detected as part of a tracklet. Given the inherently random and independent nature of BA events [52], even if a single noise event coincidentally overlaps with a tracklet in one 2D view, the probability of it consistently falling on the tracklet across all views is low. Consequently, the majority of BA events can be distinguished through crossview validation.

Gaussian pyramid. Time realignment alleviates temporal sparsity, but discontinuities in tracklet patterns across the views persist due to spatial discretization and residual temporal sparsity. We employ a Gaussian pyramid consisting of dual pathways, bottom-up and top-down, to compensate for missing trajectory details where events are not triggered. By default, we set two scales for each pathway. This generates views with relatively continuous trajectory patterns, facilitating accurate detection and segmentation.

The pyramid structure is shown in Fig. 4. In the bottom-up pathway, Gaussian filtering with a kernel size of k is applied prior to downsampling to prevent aliasing. The selection of k and the corresponding standard deviation σ are critical for effective anti-aliasing while mitigating pattern discontinuities. First, we determine the lower bound of $\sigma$ for anti-aliasing. According to the Nyquist sampling theorem, scaling by a factor of $s \in ( 0 , 1 )$ , i.e., a sampling factor of $1 / s ,$ reduces the Nyquist frequency from $f _ { N }$ to $f _ { N } ^ { \prime } = s \cdot f _ { N }$ , where $f _ { N } = 1 / 2$ in a frame. To attenuate the signal to a level α at $f _ { N } ^ { \prime }$ , the frequency response of the Gaussian function should satisfy ${ \cal H } ( \dot { f } _ { N } ^ { \prime } ) \stackrel { \cdot } { = } e ^ { - 2 \pi ^ { 2 } \sigma ^ { 2 } { f _ { N } ^ { \prime } } ^ { 2 } } \le \alpha$ , where α denotes the stopband edge gain and $\sigma$ is the standard deviation of the Gaussian kernel. Therefore, the lower bound of σ is $\begin{array} { r } { \sigma _ { a n t i } = \sqrt { \frac { - 2 \ln { \alpha } } { \pi ^ { 2 } s ^ { 2 } } } } \end{array}$ To determine the final $\sigma$ and k, we conduct a series of experiments to analyze the tracking performance under different combinations of $\sigma$ and k. The details are provided in Sec. IV-F. Although the spatial and temporal units are inhomogeneous, we still use isotropic Gaussian kernels. This is because our views, specifically YOT and TOX, are constructed by mapping physical timestamps onto discrete grid indices, which means that the Gaussian kernels operate on grid topology rather than physical timestamps. Moreover, our pyramid does not harm high-precision timestamps, since the smoothed views are used exclusively for detection/segmentation, whereas the other stages still retain the original timestamps.

BA events [53], [54] always occur independently and randomly, unlike events from objects, which are highly correlated across neighboring pixels. Therefore, BA events are highfrequency anomalies superimposed on the ideal event output. When the Gaussian kernel is applied, neighboring pixels are averaged, which can suppress the influence of BA events. The top-down pathway upsamples the scaled frame. The pyramid design compensates for missing textures while incorporating an anti-aliasing kernel to minimize the impact of artificial artifacts. A detailed experimental analysis of this pyramid design is provided in Sec. IV-G.

## D. MvF

MvF is the core stage for recovering complete trajectories from multi-view detection or segmentation results. As illustrated in Fig. 5, this stage operates in two sequential phases: (1) tracklet isolation, which integrates all results to generate isolated tracklets that are independent across views, and (2) tracklet association, which links tracklets for complete trajectories via optimal assignment.

Tracklet isolation. The isolation phase leverages complementary information from different views to resolve trajectory ambiguities caused by occlusions. This phase is further divided into two steps, temporal post-processing and multiview tracklet isolation. Due to the discretization of tracklet patterns along the t-axis or algorithmic errors, initial detection/segmentation results of YOT and TOX often consist of numerous small regions, as shown in Fig. 6(a) and (b). Since sparse event points yield less accurate direction vectors in the association phase, we introduce an iterative post-processing step to combine regions, as shown in Fig. 6(c) and (d).

Criterion 1. Tracklets $K _ { i }$ and $K _ { j }$ of one trajectory should lie within the time-extended region of each other.

$$
\| ( c _ { j } - c _ { i } ) \times \mathbf { d } _ { i } \| _ { 2 } \leq w _ { i } ,\tag{5}
$$

where $c _ { i } , \ \mathbf { d _ { i } }$ , and w<sub>i</sub> are the centroid, unit direction vector, and half-width of $K _ { i }$ . They are given by

$$
c _ { i } \triangleq ( E [ { \hat { T } } _ { i } ] , \ E [ X _ { i } ] , \ E [ Y _ { i } ] ) ,\tag{6}
$$

$$
\mathbf { d } _ { i } \triangleq \mathrm { n o r m a l i z e } ( c _ { i 2 } - c _ { i 1 } ) ,\tag{7}
$$

y  
![](images/5559b73de27858cf0d69e1de3b689c508022aa403ead0e980ed272cef51a8bca.jpg)  
Fig. 5. Pipeline of the $\mathbf { M v F }$ stage. It consists of two core phases: (a) Isolation, which resolves trajectory ambiguities by integrating segmentation results from multiple views to extract distinct 3D trajectories, and (b) Association, which associates continuous and discontinuous tracklets into complete trajectories through optimal assignment. A simple filtering rule based on event count is applied to the final output to exclude isolated noise.

![](images/2ea50207878de21c57ebc555d3917ad66b4d748d5055a8a924f5282717184fae.jpg)  
(a)

![](images/8756acef02f617c9cf6f1ff71bc0fb1179a729845ff0bf651ee6b0bfc237c588.jpg)  
(b)  
(c)  
(d)  
Fig. 6. Illustration of iterative region post-processing in the YOT view, where tracklets evolve along the t-axis. (a) Initial segmentation results, with colors denoting different detection or segmentation results. The tracklets are highly fragmented. (b) During post-processing, a direction vector $\mathbf { d } _ { i }$ (red lines) is calculated from its two centroids $c _ { i 1 }$ and $c _ { i 2 }$ (green dots) for each tracklet fragment. (c) Merged tracklets after post-processing. (d) The final direction vectors and centroids recalculated for the tracklets are shown.

$$
w _ { i } \triangleq \operatorname* { m a x } ( \lVert ( K _ { i } - c _ { i } ) \times \mathbf { d } _ { i } \rVert _ { 2 } ) ,\tag{8}
$$

where ${ \hat { T } } _ { i } , \ X _ { i } ,$ , and $Y _ { i }$ represent the realigned timestamp set and 2D location set of all events in $K _ { i }$ , respectively. $E [ \cdot ]$ means expectation. normalize(·) denotes L2 normalization. ∥· ∥<sub>2</sub> denotes L2-norm, and $c _ { i 1 }$ and $c _ { i 2 }$ are centroids of two parts of $K _ { i }$ partitioned by its mean timestamp.

Post-processing relies on Criterion 1, which is founded on a common assumption [25], [55] that trajectories approximate linear tracklets within small spatio-temporal windows. Though this criterion describes relationships between 3D tracklets, it also applies to 2D segmentation regions.

After region post-processing, we perform inverse projection to map the multi-view results into a common 3D coordinate system. This step reconstructs the time-realigned 3D events from 2D segmentation regions r<sub>XOY</sub>, r<sub>YOT</sub>, and $r _ { T O X }$ yielding corresponding 3D tracklet sets R<sub>XOY</sub>, R<sub>YOT</sub>, and $R _ { T O X }$ . Taking $R _ { X O Y }$ as an example, we formulate inverse projection in Eq. 9.

$$
R _ { X O Y } = \{ \hat { e } _ { i } = ( \hat { t } _ { i } , x _ { i } , y _ { i } ) \in \hat { \Phi } | \exists r _ { j } \in r _ { X O Y } , ( x _ { i } , y _ { i } ) \in r _ { j } \} ,\tag{9}
$$

![](images/d83763f7eda6d02a149f84fb4be0af8d7647d136192f59873f934d90a67621ad.jpg)

![](images/10f7cd92f33051fdaa181d332a39a301c80a71bf4c59ec10d827bc3076169741.jpg)  
Fig. 7. Illustration of occlusion handling via multi-view trajectory isolation. The extracted trajectories from the XOY and YOT views are shown, where trajectory a in the XOY corresponds to separated trajectories c and d in YOT, while trajectory b corresponds to e.

where $r _ { X O Y }$ consists of a set of regions $r _ { j }$ . Each region $r _ { j }$ comprises a series of 2D coordinates.

The dimensionality reduction in the pseudo-frame representation means that events mapped to the same pixel may originate from distinct objects, which prevents single pseudoframe methods from distinguishing intersecting or overlapping tracks. Our solution for disambiguation is guided by the principle that distinct 3D trajectories can be separated as long as they are non-overlapping in at least one of their 2D projections. We isolate tracklets by intersecting their inverseprojected 3D event sets, leveraging the fact that overlapping tracks in one view are often separated in others.

To implement this, we systematically isolate tracklets via intersection operations on $R _ { X O Y } , R _ { Y O T }$ , and $R _ { T O X }$ to obtain the final isolated tracklet set MR, as in Eq. 10. This step also helps identify outliers within tracklets, as shown in Fig. 7.

$$
M R = \{ K _ { i } \cap K _ { j } \cap K _ { k } | K _ { i } \in R _ { X O Y } , K _ { j } \in R _ { Y O T } , K _ { k } \in R _ { T O X } \} .\tag{10}
$$

Tracklets association. The association phase handles two types of associations: (1) intra-batch association, resolving potential over-fragmentation due to isolation or inherent discontinuous trajectory patterns, and (2) inter-batch association, linking tracklets across different batches. Although trajectories can be locally approximated as linear tracklets over short intervals, they often exhibit complex nonlinear patterns over extended periods due to the existence of varying forces, such as vehicle steering. The inherent sparsity and asynchronicity of events further compound this challenge by producing nonsmooth trajectory contours.

![](images/4927ae34479a8741a42e3f8245514e9ec2b951cc1f1a2003976f3708a783ca44.jpg)  
Fig. 8. Illustration of tracklet association criteria. $\theta _ { i }$ and $\theta _ { j }$ denote the angular deviations between the tracklet directions $( \mathbf { d } _ { i } , \mathbf { d } _ { j } )$ and the normalized centroid connection vector $\mathbf { c } _ { i } \mathbf { c } _ { j }$ , respectively. (a) Criterion 2: tracklets with small directional variation are associated through motion continuity constraints. (b) Criterion 3: tracklets exhibiting big directional changes are linked via spatiotemporal intersection verification.

Despite these complexities, adjacent tracklets typically maintain motion continuity. We formulate tracklet association as an optimal assignment problem governed by three geometric criteria, the previously introduced Criterion 1, along with Criterion 2 and 3 illustrated in Fig. 8. Criterion 2 evaluates the coplanarity of two tracklets’ directions and their positional variation, while Criterion 3 assesses whether a valid spatiotemporal intersection I exists between two tracklets.

Criterion 2. Tracklets $K _ { i }$ and $K _ { j }$ are considered to belong to the same trajectory if they achieve the minimal direction change among all candidate pairs, subject to constrained directional changes (not exceeding π/2) and consistent positional progression.

$$
\begin{array} { r } { \begin{array} { r l } & { \operatorname { a r g m i n } _ { K _ { i } , K _ { j } } | ( { \bf d _ { i } } \times { \bf d _ { j } } ) \cdot { \bf c } _ { i } { \bf c } _ { j } | , } \\ & { \quad \mathrm { ~ s . ~ t . ~ } \ \| { \bf d } _ { i } \times { \bf c _ { i } } { \bf c _ { j } } \| _ { 2 } \leq \beta , \quad \| { \bf c _ { i } } { \bf c _ { j } } \times { \bf d _ { j } } \| _ { 2 } \leq \beta , } \end{array} } \end{array}\tag{11}
$$

where $\mathbf { c } _ { i } \mathbf { c } _ { j }$ is the normalized centroid connection vector, and $\beta$ is a parameter constraining the allowable directional change. If multiple pairs achieve the minimum, the tie is broken by selecting the pair with the smallest angle $\angle ( \mathbf { d } _ { i } , \mathbf { d } _ { j } )$ , and subsequently the smallest centroid distance.

Criterion 3. For chronologically ordered tracklets $K _ { i }$ and $K _ { j }$ $( \hat { t } _ { c _ { i } } < \hat { t } _ { c _ { i } } )$ that cannot satisfy Criterion 2, their tangent vectors are extendedfrom centers $c _ { i }$ and $c _ { j }$ to compute an intersection point I. If I lies within the spatio-temporal region bounded by their centers, they are likely to belong to one trajectory.

$$
\begin{array} { r l } & { \textrm { \exists } I \in S , } \\ & { ~ s . t . \| \mathbf { d } _ { i } \times \mathbf { d } _ { j } \| _ { 2 } \neq 0 , ~ | ( \mathbf { d _ { i } } \times \mathbf { d _ { j } } ) \cdot \mathbf { c } _ { i } \mathbf { c } _ { j } | = 0 , } \\ & { S = \{ ( \hat { t } , x , y ) | x \in [ \operatorname* { m i n } ( x _ { c _ { i } } ^ { - } , x _ { c _ { j } } ^ { - } ) , \operatorname* { m a x } ( x _ { c _ { i } } ^ { + } , x _ { c _ { j } } ^ { + } ) ] , } \\ & { y \in [ \operatorname* { m i n } ( y _ { c _ { i } } ^ { - } , y _ { c _ { j } } ^ { - } ) , \operatorname* { m a x } ( y _ { c _ { i } } ^ { + } , y _ { c _ { j } } ^ { + } ) ] , ~ \hat { t } \in [ \hat { t } _ { c _ { i } } , \hat { t } _ { c _ { j } } ] \} , } \end{array}\tag{12}
$$

where $x _ { c _ { i } } ^ { + } = x _ { c _ { i } } + w _ { i } , ~ x _ { c _ { i } } ^ { - } = x _ { c _ { i } } - w _ { i } , ~ y _ { c _ { i } } ^ { + } = y _ { c _ { i } } + w _ { i } , ~ y _ { c _ { i } } ^ { - } =$ $y _ { c _ { i } } - w _ { i }$ define the spatial extents of x- and y-positions $o f c _ { i } ,$ $t _ { c _ { i } }$ is its realigned t-component, and $w _ { i }$ denotes the spatial half-width of the spatio-temporal volume $K _ { i }$ . These variables for $c _ { j }$ are defined analogously.

To determine $\beta$ in Criterion 2, we first assume the ideal case where two tracklets from the same trajectory have a directional variation of $\theta _ { v } \triangleq \angle ( \mathbf { d } _ { i } , \mathbf { d } _ { j } ) = \arcsin \| \mathbf { d } _ { i } \times \mathbf { d } _ { j } \| _ { 2 } \leq \pi / 2$ Let the angular deviation $\theta _ { i } \triangleq \angle ( \mathbf { d } _ { i } , \mathbf { c } _ { i } \mathbf { c } _ { j } ) , \theta _ { j } \triangleq \angle ( \mathbf { c } _ { i } \mathbf { c } _ { j } , \mathbf { d } _ { j } )$ These three angles satisfy ${ \theta } _ { i } ~ \leq ~ { \theta } _ { v }$ and ${ \theta } _ { j } ~ \le ~ { \theta } _ { v }$ . The constraints in Eq. 11 can be reformulated as

$$
\begin{array} { r } { \| \mathbf { d } _ { i } \times \mathbf { c _ { i } } \mathbf { c _ { j } } \| _ { 2 } \leq \sin \theta _ { v } , } \\ { \| \mathbf { c _ { i } } \mathbf { c _ { j } } \times \mathbf { d _ { j } } \| _ { 2 } \leq \sin \theta _ { v } . } \end{array}\tag{13}
$$

To account for computational errors in direction vectors, we introduce an angular error $\theta _ { e }$ of $\theta _ { i } , \theta _ { j } .$ , and $\theta _ { v } .$ , reformulating $\theta _ { v }$ as $\theta _ { v } + \theta _ { e }$ . Assuming $\theta _ { e } \in [ - \Delta \theta , \Delta \theta ]$ , the upper bound for $\beta$ is max $( \sin ( \theta _ { v } \pm \Delta \theta ) )$ . Given the computational error $\Delta \theta ,$ , the candidate pairs with the minimal triple product in Criterion 2 may no longer be the best matching pairs, requiring an adjustment to the objective. Since the original coplanarity objective minimizes the volume of the parallelepiped formed by unit vectors $\mathbf { d } _ { i } , \mathbf { d } _ { j }$ , and $\mathbf { c } _ { i } \mathbf { c } _ { j }$ , let $V = | ( \mathbf { d } _ { i } \times \mathbf { d } _ { j } ) \cdot \mathbf { c } _ { i } \mathbf { c } _ { j } |$ be the volume and $\begin{array} { r } { \mathbf { n } = \frac { \mathbf { { d } } _ { i } \times \mathbf { { d } } _ { j } } { \lVert \mathbf { { d } } _ { i } \times \mathbf { { d } } _ { j } \rVert _ { 2 } } } \end{array}$ be the normal vector. Then

$$
{ \begin{array} { r l } & { V = | \mathbf { d } _ { i } \times \mathbf { d } _ { j } | \cdot | \mathbf { c } _ { i } \mathbf { c } _ { j } | } \\ & { \quad = \sin \theta _ { v } \cdot | \mathbf { n } \cdot \mathbf { c } _ { i } \mathbf { c } _ { j } | } \\ & { \quad = \sin \theta _ { v } | \cos \phi | \leq \sin \theta _ { v } } \\ & { \quad \leq \sin ( \theta _ { i } + \theta _ { j } ) \leq \sin \theta _ { i } + \sin \theta _ { j } } \end{array} }\tag{14}
$$

where $\phi$ is the angle between n and $\mathbf { c } _ { i } \mathbf { c } _ { j }$ . The objective is simplified to minimizing the sum of the sines of these angular deviations, which aligns with the constraints.

$$
\begin{array} { r l } & { \operatorname { a r g m i n } _ { K _ { i } , K _ { j } } \| \mathbf { d _ { i } } \times \mathbf { c _ { i } c _ { j } } \| _ { 2 } + \| \mathbf { c _ { i } c _ { j } } \times \mathbf { d _ { j } } \| _ { 2 } } \\ & { \quad \mathrm { ~ s . ~ t . ~ } \| \mathbf { d _ { i } } \times \mathbf { c _ { i } c _ { j } } \| _ { 2 } \leq \operatorname* { m a x } ( \sin ( \theta _ { v } \pm \Delta \theta ) ) , } \\ & { \quad \quad \quad \quad \| \mathbf { c _ { i } c _ { j } } \times \mathbf { d _ { j } } \| _ { 2 } \leq \operatorname* { m a x } ( \sin ( \theta _ { v } \pm \Delta \theta ) ) . } \end{array}\tag{15}
$$

Similarly, the triple product constraint in Criterion 3 can be expressed with $\Delta \theta .$ Consequently, this criterion is rewritten as Eq. 16, where I is computed via vector operations and trigonometric relationships.

$$
\begin{array} { c } { \exists I \in S , } \\ { s . t . \ \lVert { \bf d } _ { i } \times { \bf d } _ { j } \rVert _ { 2 } \neq 0 , } \\ { \left. \left. ( { \bf d } _ { \bf i } \times { \bf d } _ { \bf j } ) \cdot { \bf c } _ { i } { \bf c } _ { j } \right. \leq \sin ( \theta _ { v } \pm \Delta \theta ) , \right. } \\ { S = \{ ( \hat { t } , x , y ) | x \in [ \operatorname* { m i n } ( x _ { c _ { i } } ^ { - } , x _ { c _ { j } } ^ { - } ) , \operatorname* { m a x } ( x _ { c _ { i } } ^ { + } , x _ { c _ { j } } ^ { + } ) ] , } \\ { y \in [ \operatorname* { m i n } ( y _ { c _ { i } } ^ { - } , y _ { c _ { j } } ^ { - } ) , \operatorname* { m a x } ( y _ { c _ { i } } ^ { + } , y _ { c _ { j } } ^ { + } ) ] , \ \hat { t } \in [ \hat { t } _ { c _ { i } } , \hat { t } _ { c _ { j } } ] \} , } \end{array}\tag{16}
$$

During intra-batch association, MR is associated using Criterion 1 and 2 sequentially, while in inter-batch association, Criterion 2 and 3 are executed sequentially. It should be noted that Criterion 3 has limited applicability for small objects. Since tracklets generated by small targets consist of only a few sparsely distributed events, their directional vectors are highly susceptible to local interference events. Due to geometric amplification, even a minor deviation in the estimated direction can cause the computed intersection I to drift far outside the ideal spatio-temporal regions. Conversely, larger objects often trigger a large number of events, yielding more stable direction estimates. Furthermore, their larger volumes provide more tolerance for spatial deviation. This is confirmed by our ablation study in the supplementary material. Although the combination of these criteria is flexible, Criterion 3 must be executed after Criterion 2. Specifically, it is applied only to the remaining unmatched tracklet pairs because it imposes a looser constraint, and applying this less restrictive condition as a fallback does not degrade tracking performance. Following the association process, successfully matched trajectories retain their original IDs, whereas newly detected trajectories are assigned new ones.

TABLE I  
THE CONFIGURATION OF SOD. $\because \mathrm { S E Q . } ^ { \prime \prime }$ MEANS SEQUENCES, AND “SEGMENTS” DENOTES THE NUMBER OF SEGMENTS IN EACH SEQUENCE.
<table><tr><td>Category</td><td>Source</td><td>Seq.</td><td>Segments</td><td>Objects</td><td>Type</td><td>Size</td><td>Scene</td><td>Motion Type</td><td>Speed (px/s)</td><td>Camera</td><td>GT</td><td>Period (s)</td></tr><tr><td rowspan="3">SOD-r</td><td rowspan="3">Real-world</td><td>Sphere-20 Sphere-30</td><td></td><td>20</td><td>Balls Balls</td><td rowspan="5"></td><td rowspan="3">Mixed</td><td>Non-linear Non-linear</td><td rowspan="3">≥ 2500</td><td rowspan="3"></td><td rowspan="3">Fixed</td><td rowspan="3">0.5 No 0.5 4.0</td></tr><tr><td></td><td>333</td><td>30</td><td>Raindrops</td><td>Mixed Rain</td></tr><tr><td>Raindrops</td><td></td><td>&gt; 100</td><td>5-10px</td><td>Linear</td></tr><tr><td rowspan="2">SOD-b</td><td>Tree</td><td></td><td>1</td><td>19 Balls</td><td></td><td>Tree</td><td></td><td></td><td>6DoF</td><td>Yes</td><td>3.0</td></tr><tr><td>Blender</td><td>Sky Smog</td><td>1 1</td><td>28 44</td><td>Balls Balls</td><td>Cloudy sky Full of smog</td><td>Free</td><td>≥ 100</td><td></td><td></td><td>3.0 3.0</td></tr></table>

![](images/ec6694def6cbefb0c485f907843be11766e4ee041e37be82e7d7826d2ed9275e.jpg)  
Fig. 9. The recording equipment and smog scenario setup in the XOY view used in our SOD-b dataset.

## IV. EXPERIMENTS

Our experiments have four purposes. First, assess the generalization capability and tracking effectiveness of the proposed framework MvT by incorporating both non-learning and learning-based image-based detection and segmentation methods. Second, compare MvT with existing pseudo-frame and event-by-event-based baselines, with a particular focus on small object tracking scenarios. Third, perform ablation studies to validate the contribution of components in MvT. Finally, sensitivity experiments on key parameters, including direction angle error $\Delta \theta ,$ , sampling scale s in Gaussian pyramid, and window size B, are provided in the supplementary material. All evaluations are conducted on a platform with an Intel Core i7-7700HQ CPU @ 2.8GHz, 16G pf 2600MHz RAM and an NVIDIA GTX 1050 Ti GPU. Since event cameras suffer from significant noise [56], we apply an event count filter to all methods’ results to remove trajectories with fewer than m events. m is set to 20, which is lower than the total event count of a legitimate and complete trajectory.

## A. Datasets

Datasets. Experiments are conducted on four datasets with varying object scales, including three public datasets, ECD Shapes [19], BEES [20], and Ev-UAV [21], and a selfcollected dataset SOD. ECD is a standard bounding boxbased MOT dataset, containing different shapes of objects with normal size $( > 2 5 \times 2 5 $ pixels) under three camera motions, translation, rotation, and 6DoF. BEES and Ev-UAV provide event-wise Ground Truth (GT) labels and focus on small objects $( \leq ~ 1 0 \times 1 0$ pixels) with complex and independent motion patterns. BEES, despite being a 5s recording, contains approximately 5 million events with 70 to 110 labeled bees. However, it only provides segmentation labels without distinguishing different trajectories. Ev-UAV includes both segmentation and tracking labels, but only a few UAVs (≤ 5) appear in the test and validation sets.

Due to the limited number of datasets for small and multiple object tracking in event-based vision, we collect SOD to include small, independently fast-moving targets for evaluation.

Unlike the previous three datasets with low spatial resolution (240×360) and a single moving element (either the camera or the objects), SOD is recorded using a high spatial resolution $( 7 2 0 \times 1 2 8 0 )$ event camera, Prophesee EVK4, and provides scenarios with both camera and object motion. It consists of two parts, real-world and simulated data, as listed in Table I. The real-world sequences are captured in three outdoor scenarios, tree, playground, and rain. Based on the target types and quantities, these are categorized as Sphere-20, Sphere-30, and Raindrops. Sphere-20 and Sphere-30 sequences record marbles thrown by hand in both tree and clean playground scenes, while the Raindrops sequences record falling raindrops. The simulated sequences are generated by screen-recording the scenarios rendered by Blender, as shown in Fig. 9, which is a common practice in event camera simulation [57]. In the simulation set, we design three different scenarios, tree, sky and smog, within which we simulate blobs moving randomly. Using the bounding box-based tracking GTs generated by Blender, we convert them into event-wise labels following the same procedure as in Ev-UAV. While Ev-UAV mainly contains trajectories with large spatio-temporal breaks, SOD features target overlapping and static background occlusion. SOD will be made available in our repository, and more information about the simulated subset (SOD-b) is provided in the supplementary material.

## B. Evaluation metrics

To comprehensively evaluate our framework, we employ both bounding box-based and event-wise MOT metrics. The former aligns with standard metrics in image-based MOT, including Precision (P), Recall (R), Multiple Object Tracking Accuracy (MOTA) [58], and Feature Age (FA) [40]. By default, FA is evaluated over all trajectories. For a fair comparison, we evaluate only the longest trajectories, with counts equalized to 20% and 50% of the minimum trajectory total across all methods, visualized via box plots. In these plots, boxes represent the interquartile range (25%-75%), with blue and red lines marking the median and mean, and whiskers showing the full data range. Event-wise metrics are derived from event-based segmentation metrics [21], specifically Segmentation IoU (Seg. IoU) and Segmentation Accuracy (Seg. Acc). We extend them to Track IoU and Track Acc, and add IDF1 [58] to evaluate tracking stability. As box-based metrics rely on GT bounding boxes, they may yield biased evaluations when events of one trajectory are temporarily misassigned to another trajectory. Therefore, event-wise metrics serve as the primary indicators.

## C. Integration with non-learning-based methods

We integrate three classic image-based segmentation methods, region growing [59], watershed [60], and Mean Shift [61]

![](images/97b9c940203915334530dae53524c7848ff9c9f36d64974adfd99dc62babd834.jpg)  
Fig. 10. Partial tracking results on ECD Shapes [19] are presented in 2D and 3D. In 2D images (the XOY view), red boxes denote GT, while green boxes represent the tracking results. For 3D visualizations, the displayed period is 200 ms around the timestamp of the left image. Distinct colors represent individual trajectories. From left to right, the sequences are translation (frame 532), 6DoF (frame 236), and 6DoF (frame 1014).

into MvT, denoted as MvT+RG, MvT+WS, and MvT+MS.   
Implementations are provided in the supplementary material.

Comparing Methods. In this section, we compare our MvT with existing event-based MOT methods. For pseudo-frame methods, we compare against ETD [5] and RMRNet [6]. For the event-by-event paradigm, we evaluate against AEB-Tracker [38] (AEBT), AEMOT [20], and E-MS [7]. Since ETD, RMRNet, and AEBT require GT initialization of object positions, they are evaluated exclusively on ECD. AEMOT incorporates a binary classifier trained on 28 × 28 intensity patches, which are only available in BEES. For BEES and SOD, we additionally employ the spatter tracking method from Prophesee software [62] as a baseline, denoted as PROPHE-SEE. Since PROPHESEE generates bounding boxes at each evaluation window, we convert the boxes into event trajectories for evaluation and visualization.

(1) Tracking on ECD Shapes. All compared methods are evaluated using their original settings. RMRNet is trained on the 6DoF sequence and tested on the remaining scenes. For MvT, we set B = 1800 events (consistent with E-MS) and $\Delta \theta = \pi / 6$ . Although ECD provides only bounding box-based GTs, we convert them into event-level GT trajectories using the same rule as in Ev-UAV to enable event-wise evaluation. Results are presented in Table II and Fig. 10. In Table II, methods requiring GT initialization always rank in the top two for Precision. This advantage is attributed not only to GT initialization but also to the upstream learning-based or appearance-based detectors. Event cameras are inherently sensitive to dynamic changes, and in ECD, environmental artifacts such as paper contours or wall shadows frequently trigger events. In non-learning-based methods, these events may be mistakenly identified as objects, increasing false positives and reducing overall precision.

In terms of the tracking metrics MOTA and IDF1, the MvT variants (MvTs) achieve the top three across all scenarios. Unlike Track IoU and Track Acc, which only reflect the percentage of true positive events within each GT trajectory, the leading MOTA and IDF1 scores of MvTs demonstrate stronger long-term tracking and association capabilities compared to other methods. As shown in Fig. 10, RMRNet produces less smooth trajectories than other methods, as it treats event-based tracking purely as an image-based task. MvT+MS, which utilizes the same clustering strategy Mean Shift as E-MS, yields more accurate tracking and continuous trajectories than E-MS. Among MvTs, MvT+RG and MvT+MS achieve the best overall performance, while MvT+WS consistently yields the lowest Track IoU and Track Acc, revealing that it fails to capture the majority of target events.

TABLE II  
TRANSLATION AND ROTATION SEQUENCES. “FPS” REFERS TO THE PROCESSING BATCHES/FRAMES PER SECOND. THE BEST AND SECOND BEST VALUES ARE HIGHLIGHTED.
<table><tr><td rowspan=1 colspan=1>Seq.</td><td rowspan=1 colspan=1>GTs</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>P(%)</td><td rowspan=1 colspan=1>R(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>TrackIoU(%)</td><td rowspan=1 colspan=1>TrackAcc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td><td rowspan=1 colspan=1>FPS</td></tr><tr><td rowspan=2 colspan=1>trans.</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>ETDRMRNetAEBT</td><td rowspan=1 colspan=1>48.7295.8881.87</td><td rowspan=1 colspan=1>40.6191.6981.73</td><td rowspan=1 colspan=1>-6.3154.11</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>7.44×106.672.51</td></tr><tr><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>E-MSMvT+RGMvT+WSMvT+MS</td><td rowspan=1 colspan=1>74.7475.8577.9276.49</td><td rowspan=1 colspan=1>97.2499.3697.6198.91</td><td rowspan=1 colspan=1>62.5664.6765.4964.49</td><td rowspan=1 colspan=1>30.5143.7326.9343.43</td><td rowspan=1 colspan=1>40.5755.6930.6854.46</td><td rowspan=1 colspan=1>40.3672.8474.6475.87</td><td rowspan=1 colspan=1>2.81×1042.58×1012.16×1011.8×10−1</td></tr><tr><td rowspan=3 colspan=1>rotat.</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>ETDRMRNetAEBT</td><td rowspan=1 colspan=1>41.0693.7269.19</td><td rowspan=1 colspan=1>48.0488.0086.01</td><td rowspan=1 colspan=1>-26.6331.98</td><td rowspan=1 colspan=1>一一</td><td rowspan=1 colspan=1>一1</td><td rowspan=1 colspan=1>一一一</td><td rowspan=1 colspan=1>7.78×105.832.51</td></tr><tr><td rowspan=2 colspan=1>No</td><td rowspan=2 colspan=1>E-MSMvT+RGMvT+WSMvT+MS</td><td rowspan=2 colspan=1>65.6465.7369.7269.89</td><td rowspan=2 colspan=1>95.8699.1197.4298.45</td><td rowspan=2 colspan=1>43.1142.4747.8049.56</td><td rowspan=2 colspan=1>29.3737.4623.7037.78</td><td rowspan=2 colspan=1>39.7748.8124.5646.33</td><td rowspan=1 colspan=1>57.68</td><td rowspan=2 colspan=1>2.67×1042.43×1011.93×1011.8×10−1</td></tr><tr><td rowspan=1 colspan=1>65.9766.3370.98</td></tr><tr><td rowspan=3 colspan=1>6dof</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>ETDRMRNetAEBT</td><td rowspan=1 colspan=1>48.8799.7283.38</td><td rowspan=1 colspan=1>34.6197.1872.01</td><td rowspan=1 colspan=1>-3.6248.76</td><td rowspan=1 colspan=1>一一一</td><td rowspan=1 colspan=1>一一</td><td rowspan=1 colspan=1>一一</td><td rowspan=1 colspan=1>7.23×10−77.783.72</td></tr><tr><td rowspan=2 colspan=1>No</td><td rowspan=2 colspan=1>E-MSMvT+RGMvT+WSMvT+MS</td><td rowspan=2 colspan=1>67.9377.9080.0777.18</td><td rowspan=2 colspan=1>97.2999.4897.4298.80</td><td rowspan=2 colspan=1>43.5567.8368.5664.73</td><td rowspan=2 colspan=1>23.2932.5419.3530.01</td><td rowspan=2 colspan=1>21.8037.0518.2836.33</td><td rowspan=1 colspan=1>24.6057.21</td><td rowspan=2 colspan=1>2.59×1042.45×1011.90×1011.9×10−1</td></tr><tr><td rowspan=1 colspan=1>56.3357.94</td></tr></table>

TABLE III

TRACKING PERFORMANCE ON BEES.
<table><tr><td>Method</td><td>Seg. IoU(%)</td><td>Seg. Acc(%)</td><td>Number of Tracks</td></tr><tr><td>AEMOT</td><td>36.73</td><td>37.83</td><td>297</td></tr><tr><td>E-MS</td><td>29.52</td><td>98.08</td><td>17686</td></tr><tr><td>PROPHESEE</td><td>28.39</td><td>33.99</td><td>954</td></tr><tr><td>MvT+RG</td><td>54.81</td><td>57.76</td><td>351</td></tr><tr><td>MvT+WS</td><td>29.93</td><td>30.42</td><td>276</td></tr><tr><td>MvT+MS</td><td>53.44</td><td>71.66</td><td>630</td></tr></table>

The runtime efficiency of MvT depends primarily on the adopted image-based method. As shown in Table II, MvT+RG and MvT+WS achieve near real-time speeds of 19-26 FPS, which is higher than RMRNet and ETD. MvT+MS drops to 0.18-0.19 FPS because the Mean Shift in the scikit-learn module is computationally expensive and cannot be executed in parallel. Nevertheless, MvT strikes a balance between efficiency and tracking accuracy.

TABLE IV  
RESULTS ON EV-UAV. “\*” MEANS THE IMAGE-BASED METHOD. “TRAIN” COLUMN INDICATES WHETHER THE METHOD NEEDS TO BE TRAINED. “REP.” MEANS THE REPRESENTATION. IN “EVENT REP.” COLUMN, “EVENT COUNT” IS EVENT COUNT IMAGES, ONE OF THE PSEUDO-FRAME REPRESENTATION, WHILE “POINT” IS THE ORIGINAL EVENT DATA FORMAT. EXCEPT FOR TWO SOT MODELS, THE FPS FOR THE REMAINING MODELS IS COMPUTED EXCLUDING THE DETECTION/SEGMENTATION TIME (OR XOY PROCESSING TIME FOR MVT).
<table><tr><td>Method</td><td>Task</td><td>Source</td><td>Train</td><td>Event Rep.</td><td>Seg. IoU(%)</td><td>Seg. Acc(%)</td><td>Track IoU(%)</td><td>Track Acc(%)</td><td>IDF1(%)</td><td>FPS</td></tr><tr><td>HDETrack</td><td>SOT</td><td>2024CVPR</td><td>Yes</td><td>Event Count</td><td>74.06</td><td>86.87</td><td>72.86</td><td>86.02</td><td>99.00</td><td>8.66</td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>65.29</td><td>72.71</td><td>60.91</td><td>67.17</td><td>73.15</td><td>147.68</td></tr><tr><td>TSETrack</td><td>SOT</td><td>2026IJCV</td><td>Yes</td><td>Event Count</td><td>78.03</td><td>88.55</td><td>76.00</td><td>88.22</td><td>99.50</td><td>13.98</td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>68.88</td><td>74.42</td><td>60.17</td><td>67.49</td><td>75.14</td><td>146.01</td></tr><tr><td>YOLOX*</td><td>Detect.</td><td>2021ArXiv</td><td>Yes</td><td>Event Count</td><td>69.58</td><td>70.86</td><td></td><td></td><td></td><td></td></tr><tr><td>+OCSORT*</td><td>MOT</td><td>2023CVPR</td><td>No</td><td>Event Count</td><td>61.25</td><td>62.02</td><td>35.70</td><td>29.44</td><td>39.72</td><td>615.75</td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>59.77</td><td>60.48</td><td>50.79</td><td>54.44</td><td>71.74</td><td>170.93</td></tr><tr><td>SEGMAN*</td><td>Segment.</td><td>2025CVPR</td><td>Yes</td><td>Event Count</td><td>57.65</td><td>58.34</td><td></td><td></td><td></td><td></td></tr><tr><td>+DEVA*</td><td>MOT</td><td>2023ICCV</td><td>Yes</td><td>Event Count</td><td>18.49</td><td>18.53</td><td>20.35</td><td>15.07</td><td>26.72</td><td>3.41</td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>48.62</td><td>49.10</td><td>34.67</td><td>43.73</td><td>70.41</td><td>19.10</td></tr><tr><td>RVT</td><td>Detect.</td><td>2023CVPR</td><td>Yes</td><td>Voxel-grid</td><td>18.88</td><td>19.96</td><td></td><td></td><td></td><td></td></tr><tr><td>+OCSORT*</td><td>MOT</td><td>2023CVPR</td><td>No</td><td>MvP</td><td>14.52</td><td>14.90</td><td>18.61</td><td>6.18</td><td>23.78</td><td>72.98</td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>15.81</td><td>16.36</td><td>33.04</td><td>13.99</td><td>39.22</td><td>214.75</td></tr><tr><td>EV-SpSegNet</td><td>Segment.</td><td>2025ICCV</td><td>Yes</td><td>Point</td><td>42.67</td><td>65.78</td><td></td><td></td><td></td><td></td></tr><tr><td>+MvT</td><td>MOT</td><td></td><td>No</td><td>MvP</td><td>45.16</td><td>74.31</td><td>60.26</td><td>64.95</td><td>64.69</td><td>22.44</td></tr></table>

(2) Tracking on Bee Swarm Dataset. We set $B = 5$ ms according to the evaluation protocol in the BEES dataset [20] for all methods. The configuration of MvT remains unchanged. The bandwidth of E-MS is set to 0.1, and AEMOT follows its original settings. Table III primarily reports segmentation performance, while tracking performance is reflected by the FA metric in Fig. 11. Regarding the two segmentation metrics, MvT variants rank first and second in Seg. IoU, and second and third in Seg. Acc. Since Seg. Acc is equivalent to the detection metric Recall, it only reflects the proportion of target events captured. As shown in Table III, E-MS achieves 98% Acc but a low IoU, indicating it captures nearly all target events at the cost of numerous false positives. In terms of FA, AEMOT achieves the longest trajectory lifetime, while other four methods except E-MS yield comparable mean and median values. As illustrated in Fig. 13, AEMOT also yields the cleanest event stream output, which is attributed to its learning-based detector’s ability to effectively distinguish targets from background noise. However, it misses trajectories in certain regions (highlighted in two colored boxes) that other methods capture. Among the remaining methods, MvT+RG outperforms PROPHESEE by generating a greater number of clear trajectories under noisy conditions (blue boxes).

Note that FA values in Fig. 11 exhibit an inverse correlation with the number of tracked objects. This is because, under a fixed number of targets and a fixed total duration, a higher trajectory count inherently implies that continuous trajectories are being fragmented regardless of misdetections. The FA comparisons for the Top-20% and Top-50% longest trajectories show the same trend as the overall FA. These results collectively confirm that our MvT framework is effective for tracking small objects with complex motion patterns.

(3) Tracking on SOD-r. All methods are evaluated with B = 10 ms. E-MS uses a bandwidth of 0.03, while MvTs are configured with $\Delta \theta = \pi / 1 0$ . Fig. 12 presents the FA box plots for the Raindrops sequences, while the remaining results are in the supplementary material. As shown in Fig. 12, our framework variants rank the top three. In Fig. 14, MvT+RG and MvT+WS maintain strong performance in both tracking accuracy and stability. However, they also exhibit limitations in trajectory separation and association. When tracklets contain an insufficient number of events, the parameter estimation is unstable or incorrect during the post-processing of views or tracklet association, as observed in Sphere-20 and Sphere-30. In contrast, both MvT+MS and E-MS struggle with these small, densely distributed objects. Despite parameter tuning, their segmentation results tend to incorrectly cluster target events. We attribute this failure to the shared Mean Shift core, which is ill-suited to handle dense distributions. PROPHESEE exhibits moderate performance. It accurately tracks some objects, but resulting trajectories are often too short. These results further validate the effectiveness and generalizability of our framework in tracking small and fast-moving objects compared to methods tailored to specific scenarios. Moreover, the clean outputs of MvTs further demonstrate the framework’s ability to mitigate the influence of noise on tracking.

![](images/6138d40025d6505b3393492a3280b5010c8567dc7910eef5ff4fd817935906e5.jpg)  
Fig. 11. Box plots of Feature Age on BEES.

![](images/669b52fb592bef4bb6da0f4239f58d524c577a25edb71826e3227749cae18530.jpg)  
Fig. 12. Box plots of Feature Age on SOD-r Raindrops.

![](images/f8c7713e5d3801a30ec9bb697fcb92a024d88ccede283a3760304aed8a2e044f.jpg)  
Fig. 13. Tracking result examples on the BEES dataset for GT, AEMOT, Prophesee SDK, and MvT+RG, visualized across two periods (2-2.5 s and 3.5-4 s). 2D images (XOY views) are synthesized by accumulating all tracked event trajectories, where black pixels indicate no events and other colors correspond to distinct trajectories. Note that official GTs do not distinguish between individual bee trajectories, so they are shown in a uniform color.

![](images/4f9442b485d15a5675d9c17ef6949e934c9802d7d4266b9f0b803d833f5afba0.jpg)  
Fig. 14. Visualization of tracking results on three scenes in SOD-r. Rows show the 2D and 3D tracking results for three sequences. In the 2D visualizations (XOY views), the background is an event accumulation, where black and white denote the absence and presence of events, respectively.

## D. Integration with learning-based methods

We integrate learning-based methods into MvT to expand its applicability to complex scenarios. Two event-based datasets are used, Ev-UAV and SOD-b. Since object trajectories exhibit distinct spatio-temporal patterns in the YOT and TOX views, appearance-based upstream models perform poorly in these views. Consequently, we adopt an asymmetric strategy, applying learning-based models to the XOY view while utilizing an appearance-independent region growing method for the YOT and TOX. We evaluate six learning-based models, including two image-based models (YOLOX [63] and SEGMAN [64]), two event-based models (RVT [65] and EV-SpSegNet [21]), and two event-based SOT models (HDETrack [66] and TSE-

TABLE V  
RESULTS ON SOD-B. HEREAFTER, ONLY THE UNIQUE BEST VALUES ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Task Type</td><td rowspan=1 colspan=1>Event Rep.</td><td rowspan=1 colspan=1>Seg. IoU(%)</td><td rowspan=1 colspan=1>Seg. Acc(%)</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc.(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=1 colspan=1>HDETrack+MvT</td><td rowspan=1 colspan=1>SOTMOT</td><td rowspan=1 colspan=1>Event CountMvP</td><td rowspan=1 colspan=1>0.700.70</td><td rowspan=1 colspan=1>24.9024.90</td><td rowspan=1 colspan=1>0.601.69</td><td rowspan=1 colspan=1>0.739.68</td><td rowspan=1 colspan=1>13.3338.38</td></tr><tr><td rowspan=1 colspan=1>TSETrack+MvT</td><td rowspan=1 colspan=1>SOTMOT</td><td rowspan=1 colspan=1>Event CountMvP</td><td rowspan=1 colspan=1>0.820.82</td><td rowspan=1 colspan=1>29.1429.13</td><td rowspan=1 colspan=1>1.232.78</td><td rowspan=1 colspan=1>1.559.83</td><td rowspan=1 colspan=1>13.6433.16</td></tr><tr><td rowspan=1 colspan=1>YOLOX*+OCSORT*+MvT</td><td rowspan=1 colspan=1>DetectionMOTMOT</td><td rowspan=1 colspan=1>Event CountMvPMvP</td><td rowspan=1 colspan=1>0.410.420.41</td><td rowspan=1 colspan=1>0.420.420.42</td><td rowspan=1 colspan=1>12.872.64</td><td rowspan=1 colspan=1>0.40V0.41</td><td rowspan=1 colspan=1>6.528.51</td></tr><tr><td rowspan=1 colspan=1>SEGMAN*+DEVA*+MvT</td><td rowspan=1 colspan=1>SegmentationMOTMOT</td><td rowspan=1 colspan=1>Event CountMvPMvP</td><td rowspan=1 colspan=1>0.080.000.08</td><td rowspan=1 colspan=1>0.120.000.12</td><td rowspan=1 colspan=1>0.000.35</td><td rowspan=1 colspan=1>0.000.09</td><td rowspan=1 colspan=1>0.005.29</td></tr></table>

Track [67]). For MOT comparisons, we select two prevailing image-based trackers, a bounding box-based OCSORT [68] and a segmentation-based DEVA [69], besides two event-based SOT models. Since SOD-b is a small-scale set, we only train models on the training set of Ev-UAV and evaluate on SOD-b and the remaining sets of Ev-UAV.

(1) Tracking on Ev-UAV. All methods are evaluated with B = 50 ms. Results are reported in Table IV and Fig. 15. Excluding the two SOT models, our method achieves the best and most stable tracking performance across the remaining four upstream models. Specifically, integrating MvT with YOLOX and RVT outperforms the detection-based OCSORT by over 15% Track IoU, 7% Track Acc, and 10% IDF1, and surpasses the segmentation-based DEVA by over 14% Track IoU, 28% Track Acc, and 43% IDF1. However, our method yields lower segmentation performance than the original upstream models. This degradation occurs because the multi-view strategy recognizes scattered events around trajectories as outliers. As illustrated in the first row of Fig. 16, TSETrack+MvT classifies surrounding events as outliers due to their irregular distribution in the TOX view (Fig. 16(c)). Although this discards some valid trajectory points, the resulting trajectories remain smooth and trackable. The two SOT models, HDETrack and TSE-Track, achieve IDF1 scores of nearly 100%. Their low false positives and false negatives can be attributed to supervised training, GT initialization, and the small number of tracking targets. To evaluate the generalization of these methods, we conduct further experiments on SOD-b without fine-tuning.

Regarding computational overhead, MvT operates at 146- 215 FPS (Table IV) when paired with bounding-box-based detectors, HDETrack, TSETrack, YOLOX and RVT. For segmentation networks, MvT still achieves 19-22 FPS, which is 6× faster than DEVA under SEGMAN. These results show the efficiency of our framework.

(2) Tracking on SOD-b. All methods are evaluated with $B = 1 6 . 6 7$ ms. Without training, several models fail to produce valid results, so we integrate the remaining four upstream models into MvT for evaluation. In Table V, all methods undergo a substantial performance drop due to severe domain shift. Since our method is a tracking framework that relies on detections, its absolute tracking performance is inherently bounded by the degraded detection accuracy. Even under such low accuracy, MvT still achieves the best relative tracking performance, surpassing HDETrack and TSETrack by over 7% in Track Acc and 20% in IDF1. These improvements further confirm the generalization and effectiveness of our multi-view framework in complex motion scenarios.

![](images/67a2303737aaab359e701c145fb848161647785367d0d570bd2d8afdb0b1d1df.jpg)  
Fig. 15. Tracking results of OCSORT and our framework on Ev-UAV based on YOLOX. 2D visualizations are XOY views. The sequences shown in the left and right two columns are test 017 and val 019, respectively.

![](images/8f0ec50900215d68386febf410259c8efb59e8780a7d5b1af97e1fa3988a8148.jpg)  
Fig. 16. Results on the sequence (test 013) in Ev-UAV where MvT exhibits a decrease in segmentation performance. The first row presents the GT, TSETrack, and TSETrack+MvT results. In the second row, (a) shows the detected trajectories in the 11th processing window of TSETrack. (b)-(d) present the detections in (a) encoded by our multi-view representation, different colors indicate different tracklets MvT detects.

## E. Case Study

This study is to validate the tracking disambiguation of our framework compared to the pseudo-frame representation. In Fig. 17, we present two cases involving occluded and closely distributed trajectories. Sub-figure (a) shows the results of MvT+TSETrack/RG and sub-figure (b) shows the results of MvF+TSETrack/RG. Since TSETrack is a voxel-grid-based model and RG operates on a single view, we integrate our MvF module with them to construct the voxel-grid variant and the single-view variant, denoted as MvF+TSETrack and MvF+RG, respectively. In the first row of Fig. 17, the single-view variant fails to distinguish trajectories from occlusions, whereas our multi-view model succeeds. For closely distributed objects, the single-view method cannot determine whether the events mapped to a pixel originate from one object. This causes several trajectory segments with the same color to be discretely distributed in the T-X-Y space. In contrast, our multi-view representation effectively separates them.

![](images/329dcb7ce3240431d47a2a2fab67478d0e8851d96c08b80a8bc30a5261356852.jpg)  
Fig. 17. Case study comparing existing pseudo-frames and multi-view representation under occlusion (first row, Ev-UAV test 007 sequence) and closely distributed trajectories (second row, SOD-r Raindrops). (a) shows the results of multi-view variants, MvT+TSETrack and MvT+RG. (b) shows the results of voxel-grid variant MvF+TSETrack and single-view variant MvF+RG.

## F. Gaussian Pyramid Design

The Gaussian pyramid is a crucial component in our MvP module, designed to mitigate the pattern discretization of event distributions in views. The design of Gaussian kernels and the number of layers are essential for its effectiveness. We test a series of values for these parameters to determine the best combination. The kernel design involves two parameters, kernel size k and standard deviation σ in Gaussian kernels. We initially set the downsampling scale $s ~ = ~ 0 . 8$ , and the stopband edge gain $\alpha ~ = ~ 2 6 \mathrm { { d B } }$ . Then the lower bound of σ for anti-aliasing is $\sigma _ { a n t i } = 0 . 9 7 4$ . The tested values of $\sigma$ and k are $\sigma \in \{ 0 . 9 7 4 , 1 . 0 , 1 . 1 , 1 . 2 , 1 . 3 , 1 . 4 , 1 . 5 , 1 . 6 , 1 . 7 \}$ and $k \in \{ 3 , 5 , 7 , 9 , 1 1 , 1 3 \}$ . We evaluate parameter sensitivity on the ECD dataset using RG (Fig. 18) and WS (Fig. 19). Fig. 18 reveals that variations in σ have almost no impact on final tracking performance. This is because RG identifies components based on non-zero pixel connectivity (> 0 for the target and = 0 for the background). Under this binary criterion, the Gaussian kernel with continuous weighting acts like a box filter. However, as k spans [3, 13], performance fluctuations become noticeable with a maximum of 15%.

In Fig. 19, when $k \ = \ 3 .$ , all σ values perform similarly well, with slightly higher Recall, Track Acc, and Track IDF1 when $\sigma \ge 1 . 4$ . Unlike RG, when $k \geq 5 ,$ , WS achieves its best performance when σ lies in [0.974, 1.1], which is around the anti-aliasing lower bound $\sigma _ { a n t i } .$ . As σ grows, performance degrades significantly. This discrepancy stems from their distinct mechanisms, as WS relies more heavily on structural patterns. To comprehensively accommodate different imagebased techniques, the Gaussian kernel is configured as $k = 3$ and $\sigma = 1 . 6$ . Although $k = 3$ does not yield the optimal tradeoff for RG, it clearly achieves the best performance for WS. Larger kernels may suppress more noise and connect broken tracklets, yielding higher Precision, but they cause noticeable performance fluctuations and risk over-smoothing adjacent trajectory patterns. This indicates that designing a dynamic kernel size selection strategy is a meaningful direction for future work, as discussed in Sec. V. The Gaussian filtering is implemented using OpenCV, with the kernel normalized to preserve the DC component.

![](images/a329403305749006d4d8c301aabdec1f12360535db23044a9d265f4df18b36d4.jpg)

Fig. 18. The results of different combinations of σ and k in Gaussian kernel design using RG on ECD.  
![](images/7eb9dfc24cad948120b815e84a92b5f8cf1604eb3397bd40d4ee87f93f611b27.jpg)

Fig. 19. The results of different combinations of σ and k in Gaussian kernel design using WS on ECD.  
![](images/f4370b5e283928afa0a7304ea238d4e9fdc52982e6b710088c6e204ad386e705.jpg)

![](images/73ed2f9120d0d3b5b2a1035db309d7cf9efb4cdf75dff2ff43a1a167967cc9a8.jpg)

![](images/a24c67ca543f6b30ff49019880cd03215cd5e8e700b0fb38fda1fb89c2dfac02.jpg)  
— RG -- WS -- Best Layer  MOTA  Track loU  Track Acc → Track IDF1  
Fig. 20. Results of different numbers of layers on Gaussian pyramid.

With respect to the number of layers, we evaluate configurations from 1 to 6, where one layer indicates that no pyramid is applied. As shown in Fig. 20, using only one layer significantly degrades the performance of both RG and WS, confirming the necessity of the Gaussian pyramid. RG achieves the optimal result with 2 layers, whereas WS performs best with 6 layers. This difference also stems from their distinct mechanisms. Furthermore, the WS results verify that the scale space built by the Gaussian pyramid helps mitigate pattern discretization. Because RG is more sensitive to the number of layers than WS, a 2-layer configuration is selected as the default.

TABLE VI  
ABLATION RESULTS OF THE MULTI-VIEW REPRESENTATION ON ECD USING MS.
<table><tr><td>Seq.</td><td>Rep.</td><td>P(%)</td><td>R(%)</td><td>MOTA(%)</td><td>IDF1(%)</td></tr><tr><td rowspan="3">translation</td><td>MvP</td><td>76.49</td><td>98.91</td><td>64.49</td><td>75.87</td></tr><tr><td>SingV</td><td>74.22</td><td>98.83</td><td>61.42</td><td>70.65</td></tr><tr><td>VG-① VG-②</td><td>30.21</td><td>97.98</td><td>-173.50</td><td>12.15</td></tr><tr><td rowspan="5">rotation</td><td>MvP</td><td>10.62 69.89</td><td>64.96 98.45</td><td>-529.21</td><td>2.05 70.98</td></tr><tr><td>SingV</td><td>65.42</td><td>98.61</td><td>49.56 41.14</td><td></td></tr><tr><td>VG-①</td><td></td><td></td><td></td><td>65.16</td></tr><tr><td>VG-②</td><td>27.86</td><td>97.73</td><td>-200.76</td><td>14.09</td></tr><tr><td>MvP</td><td>10.09</td><td>77.01</td><td>-669.07</td><td>1.87</td></tr><tr><td rowspan="4">6dof</td><td></td><td>77.18</td><td>98.80</td><td>64.73</td><td>57.94</td></tr><tr><td>SingV</td><td>76.90</td><td>98.95</td><td>63.45</td><td>57.13</td></tr><tr><td>VG-①</td><td>28.78</td><td>98.41</td><td>-193.71</td><td>6.03</td></tr><tr><td>VG-②</td><td>10.58</td><td>51.60</td><td>-417.65</td><td>1.41</td></tr></table>

TABLE VII  
ABLATION RESULTS OF THE MULTI-VIEW REPRESENTATION ON EV-UAV.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Rep.</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=1 colspan=1>YOLOX</td><td rowspan=1 colspan=1>MvPSingV</td><td rowspan=1 colspan=1>50.7934.20</td><td rowspan=1 colspan=1>54.4445.27</td><td rowspan=1 colspan=1>71.7461.73</td></tr><tr><td rowspan=1 colspan=1>RVT</td><td rowspan=1 colspan=1>MvPVG-③</td><td rowspan=1 colspan=1>33.0418.31</td><td rowspan=1 colspan=1>13.9911.67</td><td rowspan=1 colspan=1>39.2244.37</td></tr></table>

## G. Ablation Study

We conduct ablation studies on two components, multi-view representation and time realignment.

(1) Multi-view representation. We evaluate our multi-view representation (MvP) against a voxel-grid-based representation $( \mathbf { V } \mathbf { G } , \in \mathbb { R } ^ { H \times W \times T } , T = 2 0 0 )$ and a single-view representation $( \mathrm { S i n g V } , \in \mathbb { R } ^ { H \times W } )$ . While MvP and SingV share a 2D frame format, VG is inherently 3D data. To comprehensively evaluate VG against MvP, we design four experiments addressing different VG processing paradigms. First, project the 3D VG onto the same orthogonal views (XOY, YOT, TOX) used in MvP. This process is mathematically equivalent to applying MvP without the Time Realignment (TR) module. As shown in the subsequent ablation study (Table VIII and Table IX), MvP consistently outperforms this VG-projection variant. Second, apply the 3D Mean Shift to VG. Third, treat VG as a multichannel 2D image, where the same 2D Mean Shift is applied independently to each channel. Fourth, utilize an event-based VG detector, RVT, as the upstream model to compare tracking performance under the VG and MvP pipelines. The latter three experiments, denoted as VG-①, VG-②, and VG-③, are conducted on ECD and Ev-UAV, with results in Table VI and Table VII. All variants are evaluated under identical parameter settings.

In Table VI, the MvP variant significantly outperforms two VG variants across all metrics and leads the SingV variant in most metrics. In Table VII, the MvP variant again outperforms the VG variant by a large margin, achieving improvements of over 14% in Track IoU and 2% in Track Acc. This confirms that MvP successfully isolates true trajectories from adjacent trajectories or background noise. We also observe a drop in IDF1 for the MvP variant. This is because in the VG model, spatially close events are often merged into one trajectory, artificially maintaining ID consistency. In contrast, the MvP variant distinguishes individual tracklets across multiple views, inevitably increasing the number of identities and lowering IDF1, as illustrated in Fig. 17. Furthermore, we integrate an image-based detector, YOLOX, to evaluate the MvP and SingV representations on Ev-UAV. The MvP model outperforms the SingV model by 16% in Track IoU, 9% in Track Acc, and 10% in IDF1, further confirming the effectiveness of our representation.

(2) Time realignment. Time realignment aims to mitigate the effects of temporal discretization in subsequent framebased processing. We evaluate MvT+RG with and without this module, as reported in Table VIII and Table IX. The model with time realignment consistently surpasses its counterpart across all metrics, indicating the effectiveness of this module in processing sparsely and discretely distributed events.

TABLE VIII  
ABLATION RESULTS OF TIME REALIGNMENT (TR) ON ECD.
<table><tr><td rowspan=1 colspan=1>Seq.</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>P(%)</td><td rowspan=1 colspan=1>R(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=1 colspan=1>translation</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>75.8575.80</td><td rowspan=1 colspan=1>99.3699.33</td><td rowspan=1 colspan=1>64.6764.47</td><td rowspan=1 colspan=1>72.8472.23</td></tr><tr><td rowspan=1 colspan=1>rotation</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>65.7365.73</td><td rowspan=1 colspan=1>99.1199.11</td><td rowspan=1 colspan=1>42.4742.47</td><td rowspan=1 colspan=1>65.9765.81</td></tr><tr><td rowspan=1 colspan=1>6dof</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>77.9077.90</td><td rowspan=1 colspan=1>99.4899.48</td><td rowspan=1 colspan=1>67.8367.81</td><td rowspan=1 colspan=1>57.2156.52</td></tr></table>

TABLE IX

ABLATION RESULTS OF TIME REALIGNMENT (TR) ON EV-UAV.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=1 colspan=1>HDETrack</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>60.9159.19</td><td rowspan=1 colspan=1>67.1766.64</td><td rowspan=1 colspan=1>73.1571.66</td></tr><tr><td rowspan=1 colspan=1>TSETrack</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>60.1759.14</td><td rowspan=1 colspan=1>67.4967.45</td><td rowspan=1 colspan=1>75.1474.52</td></tr><tr><td rowspan=1 colspan=1>RVT</td><td rowspan=1 colspan=1>MvT+RGw/o TR</td><td rowspan=1 colspan=1>33.0433.02</td><td rowspan=1 colspan=1>13.9913.73</td><td rowspan=1 colspan=1>39.2238.82</td></tr></table>

## V. DISCUSSION

Comprehensive evaluations across four datasets featuring objects of varying scales and motion patterns demonstrate the effectiveness, generalization, and flexibility of the MvT framework. By integrating with diverse upstream detection or segmentation models, MvT successfully adapts to a wide range of tracking environments.

However, as the association model of our method relies on motion continuity rather than appearance features or specifically learned motion models, it may struggle during prolonged occlusions accompanied by significant directional changes. Specifically, while $\Delta \theta$ in association Criterion 2 bounds directional calculation errors, it restricts the motion model to near-linear assumptions. The optimal $\Delta \theta$ may vary for objects with different motion patterns. Future work will explore adaptive methods to dynamically determine $\Delta \theta$ based on object characteristics or complex motion models. Currently, MvT utilizes only the spatio-temporal coordinates (t, x, y) of each event. As introduced in Sec. I, some researchers [13], [14] have leveraged event polarity to reconstruct intensity images containing rich texture and appearance information. Integrating polarity and its underlying intensity gradient field to infer object motion represents a promising direction for adaptively determining ∆θ. Lastly, the isotropic Gaussian kernel used in MvP may not be optimal for all views, as pattern discontinuities vary across projection views. Furthermore, our Gaussian kernel experiments show that optimal kernel designs also differ across objects. It is a worthwhile direction for future to explore adaptive kernel shapes and sizes to better mitigate pattern discontinuities, especially in cases of complex motion combinations or varying object sizes.

## VI. CONCLUSION

In this work, we have presented MvT, a general framework that leverages an intuitive multi-view representation to effectively encode spatio-temporal information from event streams for event-based tracking. Our representation enables the direct application of image-based upstream detection or segmentation algorithms while leveraging complementary dimensional information to resolve tracking ambiguities, such as occlusion and overlap, in a single view. An association model consisting of three criteria under different scenarios further ensures robust tracklet linking and stable performance. Comprehensive experiments across four datasets demonstrate that MvT can flexibly integrate with learning-based and nonlearning-based models to reliably track objects of varying sizes, outperforming existing event-based methods. Crucially, as a training-free approach, MvT enables robust tracking even in extreme scenarios where GT annotations are unobtainable, positioning it as a practical and generalizable framework for event-based tracking in real-world applications.

## REFERENCES

[1] L. Shunkun, G. Banglei, L. Bin, Y. Qifeng, and S. Yang, “A pose-only geometric constraint for multi-camera pose adjustment,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–18, 2026.

[2] G. Banglei, Z. Ji, and K. Laurent, “A complete solution to generalized relative pose estimation from affine correspondences,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–15, 2026.

[3] H. N. Hu, Q. Z. Cai, D. Wang, J. Lin, M. Sun, P. Krahenbuhl, T. Darrell, and F. Yu, “Joint monocular 3d vehicle detection and tracking,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 5390–5399.

[4] C. Brandli, R. Berner, M. Yang, S.-C. Liu, and T. Delbruck, “A 240× 180 130 db 3 µs latency global shutter spatio-temporal vision sensor,” IEEE J. Solid State Circuits, vol. 49, no. 10, pp. 2333–2341, 2014.

[5] H. Chen, Q. Wu, Y. Liang, X. Gao, and H. Wang, “Asynchronous tracking-by-detection on adaptive time surfaces for event-based object tracking,” in Proc. ACM Int. Conf. Multimedia, 2019, pp. 473–481.

[6] H. Chen, D. Suter, Q. Wu, and H. Wang, “End-to-end learning of object motion estimation from retinal events for event-based object tracking,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 07, 2020, pp. 10 534–10 541.

[7] F. Barranco, C. Fermuller, and E. Ros, “Real-time clustering and multitarget tracking using event-based sensors,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Oct. 2018, pp. 5764–5769.

[8] G. Gallego and D. Scaramuzza, “Accurate angular velocity estimation with an event camera,” IEEE Robot. Autom. Lett., vol. 2, no. 2, pp. 632–639, 2017.

[9] G. Gallego, H. Rebecq, and D. Scaramuzza, “A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2018, pp. 3867–3876.

[10] Z. Ni, A. Bolopion, J. Agnus, R. Benosman, and S. Regnier, “Asyn-´ chronous event-based visual shape tracking for stable haptic feedback in microrobotics,” IEEE Trans. Robot., vol. 28, no. 5, pp. 1081–1089, 2012.

[11] Z. Zhu, J. Hou, and D. O. Wu, “Cross-Modal Orthogonal High-Rank Augmentation for RGB-Event Transformer-Trackers,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 22 045–22 055.

[12] J. Zhang, M. Zhang, Y. Wang, Q. Liu, B. Yin, H. Li, and X. Yang, “Spiking Neural Networks With Adaptive Membrane Time Constant for Event-Based Tracking,” IEEE Trans. Image Process., vol. 34, pp. 1009– 1021, 2025.

[13] F. Paredes-Valles and G. C. H. E. de Croon, “Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 3446–3455.

[14] C. Scheerlinck, H. Rebecq, D. Gehrig, N. Barnes, R. Mahony, and D. Scaramuzza, “Fast Image Reconstruction with an Event Camera,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2020, pp. 156–163.

[15] J. Zhang, B. Dong, H. Zhang, J. Ding, F. Heide, B. Yin, and X. Yang, “Spiking Transformers for Event-based Single Object Tracking,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 8791–8800.

[16] S. Schraml and A. N. Belbachir, “A spatio-temporal clustering method using real-time motion analysis on event-based 3d vision,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) Workshops, Jun. 2010, pp. 57–63.

[17] T. Bolten, R. Pohle-Frohlich, and K. D. T¨ onnies, “Application of¨ hierarchical clustering for object tracking with a dynamic vision sensor,” in Proc. Int. Conf. Intell. Comput. Control Syst., 2019, pp. 164–176.

[18] S. Hu, Y. Kim, H. Lim, A. J. Lee, and H. Myung, “ecdt: Event clustering for simultaneous feature detection and tracking,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Oct. 2022, pp. 3808–3815.

[19] E. Mueggler, H. Rebecq, G. Gallego, T. Delbruck, and D. Scaramuzza, “The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and slam,” Int. J. Robot. Res., vol. 36, no. 2, pp. 142–149, 2017.

[20] A. Apps, Z. Wang, V. Perejogin, T. L. Molloy, and R. Mahony, “Asynchronous multi-object tracking with an event camera,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2025, pp. 794–800.

[21] N. Chen, C. Xiao, Y. Dai, S. He, M. Li, and W. An, “Event-based Tiny Object Detection: A Benchmark Dataset and Baseline,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Honolulu, Hawaii, Oct. 2025.

[22] J. Zhang, X. Yang, Y. Fu, X. Wei, B. Yin, and B. Dong, “Object tracking by jointly exploiting frame and event domain,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 13 043–13 052.

[23] U. M. Nunes and Y. Demiris, “Entropy minimisation framework for event-based vision model estimation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Aug. 2020, pp. 161–176.

[24] Nunes, Urbano Miguel and Demiris, Yiannis, “Robust event-based vision model estimation by dispersion minimisation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 12, pp. 9561–9573, 2021.

[25] A. Z. Zhu, N. Atanasov, and K. Daniilidis, “Event-based feature tracking with probabilistic data association,” in Proc. IEEE Int. Conf. Robot Autom. (ICRA), Dec. 2017, pp. 4465–4470.

[26] T. Stoffregen, G. Gallego, T. Drummond, L. Kleeman, and D. Scaramuzza, “Event-based motion segmentation by motion compensation,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 7244–7253.

[27] S. Shiba, Y. Aoki, and G. Gallego, “Secrets of event-based optical flow,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Oct. 2022, pp. 628–645.

[28] T. Stoffregen and L. Kleeman, “Simultaneous optical flow and segmentation (sofas) using dynamic vision sensor,” arXiv preprint arXiv:1805.12326, 2018.

[29] D. Liu, A. Parra, and T. J. Chin, “Spatio-temporal registration for eventbased visual odometry,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 4937–4946.

[30] R. Benosman, C. Clercq, X. Lagorce, S.-H. Ieng, and C. Bartolozzi, “Event-based visual flow,” IEEE Trans. Neural Netw. Learn. Syst., vol. 25, no. 2, pp. 407–417, 2013.

[31] X. Lagorce, G. Orchard, F. Galluppi, B. E. Shi, and R. B. Benosman, “Hots: A hierarchy of event-based time-surfaces for pattern recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 7, pp. 1346–1359, 2016.

[32] X. Huang, Y. Zhang, and Z. Xiong, “Progressive spatio-temporal alignment for efficient event-based motion estimation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 1537–1546.

[33] A. Sironi, M. Brambilla, N. Bourdis, X. Lagorce, and R. Benosman, “Hats: Histograms of averaged time surfaces for robust event-based object classification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2018, pp. 1731–1740.

[34] M. Almatrafi, R. Baldwin, K. Aizawa, and K. Hirakawa, “Distance surface for event-based optical flow,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 7, pp. 1547–1556, 2020.

[35] A. Mitrokhin, C. Fermuller, C. Parameshwara, and Y. Aloimonos,¨ “Event-based moving object detection and tracking,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Oct. 2018, pp. 1–9.

[36] Z. Zhu, J. Hou, and D. O. Wu, “Cross-modal orthogonal high-rank augmentation for rgb-event transformer-trackers,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 22 045–22 055.

[37] J. P. Rodr´ıguez-Gomez, A. G. Egu´ıluz, J. R. Mart´ınez-de Dios, and A. Ollero, “Asynchronous event-based clustering and tracking for intrusion monitoring in uas,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2020, pp. 8518–8524.

[38] Z. Wang, T. Molloy, P. van Goor, and R. Mahony, “Asynchronous blob tracker for event cameras,” IEEE Trans. Robot., vol. 40, pp. 4750–4767, 2024.

[39] Z. Ni, C. Pacoret, R. Benosman, S. Ieng, and S. REGNIER, “Asyn-<sup>´</sup> chronous event-based high speed vision for microparticle tracking,” J. Microsc., vol. 245, no. 3, pp. 236–244, 2012.

[40] I. Alzugaray Lopez and M. Chli, “Haste: multi-hypothesis asynchronous speeded-up tracking of events,” in Proc. Brit. Mach. Vis. Conf. (BMVC), 2020, p. 744.

[41] D. J. Borer, “4d flow visualization with dynamic vision sensors,” Ph.D. dissertation, ETH Zurich, Zurich, Switzerland, 2014.¨

[42] M. H. Kuo, “Reconstruction of quadric surface solids from three-view engineering drawings,” Comput.-Aided Des., vol. 30, no. 7, pp. 517–527, 1998.

[43] Y. Guo, H. Wang, Q. Hu, H. Liu, L. Liu, and M. Bennamoun, “Deep learning for 3d point clouds: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 12, pp. 4338–4364, 2020.

[44] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. De Mello, O. Gallo, L. J. Guibas, J. Tremblay, S. Khamis et al., “Efficient geometry-aware 3d generative adversarial networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 16 123– 16 133.

[45] R. Khatib and R. Giryes, “Trinerflet: A wavelet based triplane nerf representation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Sept. 2024, pp. 358–374.

[46] S. Thorpe, A. Delorme, and R. Van Rullen, “Spike-based strategies for rapid processing,” Neural Netw., vol. 14, no. 6, pp. 715–725, 2001.

[47] E. H. Adelson and J. R. Bergen, “Spatio-temporal energy models for the perception of motion,” J. Opt. Soc. Am., vol. 2, no. 2, pp. 284–299, Feb. 1985.

[48] R. C. Bolles, H. H. Baker, and D. H. Marimont, “Epipolar-plane image analysis: An approach to determining structure from motion,” Int. J. Comput. Vis., vol. 1, no. 1, pp. 7–55, 1987.

[49] S. Guo and T. Delbruck, “Low Cost and Latency Event Camera Background Activity Denoising,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 1, pp. 785–795, 2023.

[50] B. Jiang, B. Xiong, B. Qu, M. Salman Asif, Y. Zhou, and Z. Ma, “EDformer: Transformer-Based Event Denoising Across Varied Noise Levels,” in Proc. Eur. Conf. Comput. Vis. (ECCV), A. Leonardis,

E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature Switzerland, Sept. 2024, pp. 200–216.

[51] Y. Duan, “Led: A Large-scale Real-world Paired Dataset for Event Camera Denoising,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024, pp. 25 637–25 647.

[52] Y. Chen, Y. Huang, F. Li, X. Zeng, W. Li, and M. Wang, “Denoising Method for Dynamic Vision Sensor Based on Two-Dimensional Event Density,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), May 2023, pp. 1–4.

[53] J. Wu, C. Ma, L. Li, W. Dong, and G. Shi, “Probabilistic Undirected Graph Based Denoising Method for Dynamic Vision Sensor,” IEEE Trans. Multimedia, vol. 23, pp. 1148–1159, 2021.

[54] P. Duan, Z. W. Wang, B. Shi, O. Cossairt, T. Huang, and A. K. Katsaggelos, “Guided event filtering: Synergy between intensity images and neuromorphic events for high performance imaging,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 8261–8275, 2021.

[55] A. Zihao Zhu, N. Atanasov, and K. Daniilidis, “Event-based visual inertial odometry,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 5391–5399.

[56] S. Ding, J. Chen, Y. Wang, Y. Kang, W. Song, J. Cheng, and Y. Cao, “E-mlb: Multilevel benchmark for event-based camera denoising,” IEEE Trans. Multimedia, vol. 26, pp. 65–76, 2023.

[57] T. J. Chin, S. Bagchi, A. Eriksson, and A. Van Schaik, “Star tracking using an event camera,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) Workshops, Jun. 2019, pp. 1646–1655.

[58] J. Luiten, A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taix˘ e, and´ B. Leibe, “HOTA: A Higher Order Metric for Evaluating Multi-object Tracking,” Int. J. Comput. Vis., vol. 129, no. 2, pp. 548–578, Feb. 2021.

[59] S. Hojjatoleslami and J. Kittler, “Region growing: A new approach,” IEEE Trans. Image Process., vol. 7, no. 7, pp. 1079–1084, 1998.

[60] L. Vincent and P. Soille, “Watersheds in digital spaces: An efficient algorithm based on immersion simulations,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 06, pp. 583–598, 1991.

[61] D. Comaniciu, V. Ramesh, and P. Meer, “Kernel-based object tracking,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 25, no. 5, pp. 564–577, 2003.

[62] Prophesee. (2021) Spatter tracking using python. [Online]. Available: https://docs.prophesee.ai/stable/samples/modules/analytics/ tracking spatter py.html

[63] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “YOLOX: Exceeding YOLO Series in 2021,” arXiv preprint arXiv:2107.08430, 2021.

[64] Y. Fu, M. Lou, and Y. Yu, “SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2025, pp. 19 077–19 087.

[65] M. Gehrig and D. Scaramuzza, “Recurrent Vision Transformers for Object Detection With Event Cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, Canada, Jun. 2023, pp. 13 884–13 893.

[66] X. Wang, S. Wang, C. Tang, L. Zhu, B. Jiang, Y. Tian, and J. Tang, “Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024, pp. 19 248–19 257.

[67] J. Zhang, X. Yang, H. Tang, Y. Wang, B. Yin, H. Wang, and X. Fu, “Efficient Vision Transformer with Token Sparsification for Event-Based Object Tracking,” Int. J. Comput. Vis., vol. 134, no. 2, pp. 1–20, 2026.

[68] J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, “Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, Canada, Jun. 2023, pp. 9686–9696.

[69] H. K. Cheng, S. W. Oh, B. Price, A. Schwing, and J.-Y. Lee, “Tracking Anything with Decoupled Video Segmentation,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 1316–1326.