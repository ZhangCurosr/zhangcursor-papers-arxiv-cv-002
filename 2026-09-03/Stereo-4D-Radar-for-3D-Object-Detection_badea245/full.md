# Stereo 4D Radar for 3D Object Detection:

# Integrating Geometric Alignment and Absolute Velocity Estimation

Seung-Hyun Song , Student Member, IEEE, Dong-Hee Paek , Member, IEEE, Woong-Chan Byun , Student Member, IEEE, and Seung-Hyun Kong , Senior Member, IEEE

Abstract—Four-dimensional (4D) Radar is a powerful sensing modality capable of detecting surrounding three-dimensional (3D) objects under diverse weather conditions and providing Doppler-based motion information. However, raw 4D Radar signals contain significant clutter from road surfaces, guardrails, and surrounding vehicles, along with multipath-induced ghost reflections and the receiver’s inherent noise floor. Consequently, preprocessing algorithms designed to remove such invalid measurements often make the Radar data excessively sparse. Moreover, the Doppler measurements provided by 4D Radar describe only the radial component of an object’s velocity, limiting their ability to recover the full motion state. In this paper, we introduce the stereo 4D Radar-based 3D object detection framework that exploits the geometric disparity between left and right Radars to estimate the absolute velocity of objects and to achieve more robust perception through the fusion of their complementary features. The effectiveness of the proposed framework is validated on our in-house stereo 4D Radar dataset, demonstrating performance gains of 8.82% in $A P _ { 3 D }$ and 9.0% in $A P _ { B E V }$ over state-of-the-art mono 4D Radar baselines. These results demonstrate that absolute velocity estimation combined with stereo geometry-aware feature fusion leads to substantial improvements in 3D object detection.

Index Terms—Stereo 4D Radar, 3D Object Detection, Velocity Estimation, Autonomous Driving

## I. INTRODUCTION

R <sup>OBUST</sup> <sup>perception</sup> <sup>that</sup> <sup>remains</sup> <sup>reliable</sup> <sup>under</sup> <sup>adverse</sup> weather conditions is essential for ensuring the safety and dependability of autonomous driving and intelligent transportation systems [1]–[3]. In particular, accurate object detection serves as a core component of perception, enabling precise understanding of the driving environment as well as reliable decision-making and control [4], [5].

Recent advances in deep learning-based visual perception have driven autonomous driving research to focus primarily on LiDAR and camera sensors [6]–[10]. However, these optical sensors rely on visible and near-infrared light, whose measurements become unreliable under adverse weather conditions such as rain, fog, and snow. In addition, they cannot directly measure object velocity, and this combination ultimately results in reduced perception reliability [11]–[13].

In contrast, Radar leverages long-wavelength millimeter-wave (mmWave) signals, providing robust perception performance even under harsh weather conditions. Moreover, Radar can directly measure the velocity of reflecting targets through the Doppler effect. Conventional automotive Radar has provided range, azimuth, Doppler velocity, and the power of each reflected signal. Recently, advances in high-resolution multiple-input multiple-output (MIMO) array technology have made it possible to measure elevation as well, giving rise to four-dimensional (4D) Radar systems. By providing threedimensional (3D) spatial structure together with Doppler information, 4D Radar has emerged as a key sensing modality for robust perception in autonomous driving [14]–[16].

However, 4D Radar-based perception systems still face two critical challenges. First, as illustrated in Fig. 1-(a), the Doppler velocity measured by Radar contains only the radial component projected onto the line-of-sight (LoS) direction between the sensor and the target, and therefore cannot fully represent the object’s true motion. Prior studies have attempted to exploit this incomplete Doppler velocity to distinguish dynamic objects or to infer their motion direction for improved classification, and some approaches attempt to mitigate the limitations of incomplete Doppler measurements by incorporating ego-velocity information [17]–[21]. Nevertheless, these methods remain limited by the fact that Doppler velocity is constrained to the LoS direction, making it difficult to recover the transverse components of object motion.

Second, 4D Radar measurements contain strong clutter from road surfaces, guardrails, and nearby vehicles, ghost reflections caused by multipath propagation, and receiver noise [22], [23]. Although preprocessing steps are applied to suppress these effects, they inevitably make the data excessively sparse and may remove valid returns along with the noise, limiting the reliability of subsequent perception modules [24], [25]. To address these limitations, various approaches have been proposed, including encoding-based methods [7], [16], [26], [27], context-based methods [28]–[30], and multi-representationbased methods [21], [31]–[35]. However, these approaches still rely on a mono 4D Radar sensor, whose limited field of view and inherently sparse point distribution reduce their ability to capture object shape and motion in real driving scenarios, resulting in performance that remains inferior to LiDAR- and camera-based perception systems.

To address these limitations, we propose a stereo 4D Radarbased 3D object detection framework that leverages geometric disparity between two Radars. First, to address the incompleteness of Doppler velocity, we deploy two identical 4D Radars separated by a fixed baseline distance $d ,$ allowing the same object to be observed simultaneously from two distinct viewpoints. As illustrated in Fig. 1-(b), the geometric baseline between the Radars and the azimuth angles measured at each Radar allow the recovery of an object’s absolute velocity from Doppler measurements. Building on this principle, we introduce a stereo 4D Radar-based absolute velocity estimation method and integrate it into a 3D object detection network. In addition, to overcome the representational limitations from sparse Radar measurements, we introduce the Stereo Radar Pillar Feature Encoding (SR-PFE) and the Stereo Radar Bilateral Fusion Module (SR-BFM). These modules align, refine, and fuse the features from the left and right Radars, enabling effective integration of stereo 4D Radar information. In this work, we design a framework that enhances incomplete motion information through stereo 4D Radar-based absolute velocity estimation and enriches sparse Radar representations through effective stereo fusion. Extensive experiments demonstrate that both components contribute substantially to improving 3D object detection performance.

![](images/0a7184f96cf8c5a3bc1f4019722b58cacfc4c303866f117da0e664b2a84e1e81.jpg)  
Fig. 1. Stereo 4D Radar-based velocity estimation. (a) Illustration of Doppler velocity, which represents the radial component of an object’s relative motion. (b) Stereo 4D Radar utilizes the Doppler measurements from the left and right Radars to estimate both the relative and absolute velocities.

In summary, the main contributions of this work are as follows:

• To the best of our knowledge, we propose the first stereo 4D Radar framework for 3D object detection, effectively mitigating the sparsity and incomplete velocity information inherent to mono 4D Radar.

• We introduce a stereo 4D Radar-based absolute velocity estimation method that recovers the true motion of dynamic objects from Doppler measurements and demonstrate its contribution to improving detection performance.

• We develop a stereo 4D Radar-based 3D object detection network equipped with SR-PFE and SR-BFM, achieving improvements of 8.82% in $A P _ { 3 D }$ and 9.0% in $A P _ { B E V }$ over state-of-the-art mono 4D Radar baselines.

The remainder of this paper is structured as follows. Section

II reviews related work on 4D Radar-based 3D object detection and Doppler utilization. Section III details the proposed absolute velocity estimation and detection network. Section IV presents the stereo 4D Radar dataset, experimental setup, and results. Section V concludes the paper and discusses future directions.

## II. RELATED WORK

## A. 3D Object Detection using 4D Radar

To effectively extract geometric and semantic information from sparse 4D Radar points, various network architectures have been proposed. Early encoding-based approaches sought to preserve spatial structure. PointPillars [7] forms pillarwise representations via vertical quantization and learns BEV features using 2D CNNs. RTNH [16] and RTNH+ [26] reconstruct Radar data into multi-dimensional tensors by separating the elevation dimension and independently processing spatial, velocity, and power channels to strengthen representation. Radar PillarNet [27] adapts pillar encoding to Radar, generating pseudo-BEV images and enhancing BEV feature learning through separated and fused spatial, velocity, and power information.

To address missing local interactions and density cues in encoding, context-based methods have emerged. RPFA-Net [28] adds self-attention to model intra-pillar interactions and restore fine-grained representations. SMURF [29] enhances pillar features with KDE-based density representations to reduce losses in sparse regions. DADAN [30] leverages a density-aware module and Doppler-guided dynamic augmentation to achieve robust detection across diverse environments.

Recently, multi-representation methods have been explored to overcome the limitations of single-level representations and to maintain spatial and temporal consistency from multiple perspectives. RadarMFNet [21] aggregates features over consecutive frames and compensates point displacements using

![](images/565d167d342a1fcb3d911985ebf86e4a5ac6de8da0bbc92af9ab21baf32e7f4b.jpg)  
Fig. 2. Overview of the proposed stereo 4D Radar-based 3D object detection pipeline. The framework consists of three key modules—Stereo 4D Radar Preprocessing, SR-PFE, and SR-BFM—along with a 2D backbone and an anchor-based detection head.

Doppler velocity, reducing motion distortion and yielding stable BEV representations. Radar M3-Net [31] employs a multiscale, multi-layer, and multi-frame architecture to enlarge the receptive field and hierarchically fuse features at different resolutions, improving detection of static and dynamic objects. MVFAN [32] extracts BEV and cylindrical representations in parallel and fuses their complementary spatial information through a multi-branch design. SMIFormer [33] models global interactions across BEV, front-view, and side-view representations using Transformer-based attention to integrate geometric relations among viewpoints. MUFASA [34] adaptively fuses range-angle, BEV, and cylindrical features via a spatialawareness module for robust representation in diverse environments. MAFF-Net [35] enhances motion-aware BEV features by integrating auxiliary cues such as velocity, intensity, and Doppler maps through attention mechanisms.

Research on 4D Radar-based 3D perception has primarily centered on pillar representations for efficient BEV processing of sparse Radar points, while recent commercial developments have also explored lightweight real-time deployment.<sup>1</sup> Yet, even with advances across encoding, context, and multirepresentation methods, mono 4D Radar with limited field of view and sparse point density continues to restrict perception performance.

## B. Use of Radar Doppler Information

In addition to range- and angle-based spatial measurements, 4D Radar provides Doppler velocity, and numerous studies have explored leveraging this information for perception. RATRack [17] employs Doppler- driven dynamic likelihood estimation to detect and track moving objects, while RSS-Net [18] incorporates Doppler as a class-discriminative feature for weakly-supervised semantic segmentation. CenterFusion [19] utilizes Doppler as an object-level motion hint to stabilize camera-Radar fusion, and RLNet [20] integrates Doppler into an adaptive fusion module to better distinguish static and dynamic regions. In multi-frame approaches such as MFNet [21], temporally aggregated Doppler is used to build velocityconsistent features for improved 3D object detection.

Meanwhile, most existing 4D Radar-based perception methods utilize the Doppler measurements in their incomplete form to support tasks. These approaches implicitly rely on Doppler as if it were sufficient motion information, even though it contains only the radial component.

## III. PROPOSED METHODS

In this section, we first provide an overview of the overall architecture and operation flow of the proposed stereo 4D Radar-based 3D object detection network, and then describe each component in detail.

## A. Overview

The overall architecture of the proposed stereo 4D Radarbased 3D object detection framework is illustrated in Fig. 2. The network consists of three key modules followed by a 2D convolutional backbone and an anchor-based detection head adapted from an open-source implementation,<sup>2</sup> which generate the final object predictions.

In the Stereo 4D Radar Preprocessing stage, point clouds collected from the left and right Radar are used to estimate the ego-velocity, separate static background points from dynamic points, cluster dynamic objects, and estimate absolute velocity. Radar points augmented with absolute velocity information are then forwarded to the detection network. Next, SR-PFE encodes the spatial, power, and velocity information of each Radar point and converts them into BEV feature maps, producing separate feature representations for the left and right Radars. SR-BFM consists of two components: an Attention Refinement (AR) block, which geometrically aligns and semantically refines the left and right features, and a Gated Correlation-Fusion (GCF) block, which generates a fused representation based on the reliability of the two Radar inputs. The following sections provide a detailed description of each module and its role within the framework.

![](images/b0e27b4c4fa85f22185b00e66acbbcd8258edf91dd7d0465a77ca255124d3cc0.jpg)  
Fig. 3. Stereo 4D Radar preprocessing pipeline and visualization of intermediate results. (a) Overall stereo 4D Radar preprocessing pipeline, (b) extraction of dynamic Radar points, (c) first-stage dynamic object clustering based on spatial proximity, (d) second-stage clustering using Doppler similarity, (e) absolute velocity estimation. The red-marked areas in (c) and (d) illustrate objects that spatial information alone fails to distinguish, whereas Doppler information enables clear separation.

## B. Stereo 4D Radar-based Velocity Estimation

We propose a three-stage absolute velocity estimation framework tailored to the characteristics of stereo 4D Radar data. First, dynamic points corresponding to moving objects are separated using Random Sample Consensus (RANSAC)- based Dynamic Point Segmentation [36]. This process is performed independently for the left and right Radars, and the robustness of RANSAC to outliers enables stable segmentation even in the presence of noise or moving objects. Second, dynamic object points from the left and right Radars are clustered using a two-step Density-Based Spatial Clustering of Applications with Noise (DBSCAN) procedure [37]. The first stage groups points based on spatial proximity, and the second stage refines the clusters using Doppler similarity, allowing accurate separation of adjacent objects with different motion patterns. Third, absolute velocity is estimated using stereo-based ridge-regularized least squares (RLS). By combining Doppler measurements observed from two different viewpoints, stereo 4D Radar can recover the full velocity vector—including both radial and transverse components—rather than only the radial component available from a mono 4D Radar.

1) Dynamic Point Segmentation: In this stage, we apply a RANSAC-based method grounded in a 2D translation velocity model to (i) estimate the ego-velocity on the horizontal plane, denoted as $\hat { \mathbf { v } } _ { s }$ , using Doppler measurements, and (ii) separate points corresponding to dynamically moving objects to obtain a dynamic mask.

(i) Each Radar point is given by $\mathbf p _ { i } = ( x _ { i } , y _ { i } , z _ { i } , \mathrm { P W } _ { i } , v _ { i } ^ { \mathrm { d o p } } )$ where PW<sub>i</sub> denotes the reflection power and $v _ { i } ^ { \mathrm { d o p } }$ is the measured Doppler velocity. The LoS unit vector of each point, $\mathbf { h } _ { i } ,$ is defined as:

$$
\mathbf { h } _ { i } = \frac { [ x _ { i } , y _ { i } , z _ { i } ] ^ { \top } } { \| [ x _ { i } , y _ { i } , z _ { i } ] \| } , \quad \mathbf { h } _ { i , x y } = [ h _ { i , x } , h _ { i , y } ] ^ { \top } ,\tag{1}
$$

where $\mathbf { h } _ { i , x y }$ denotes the projected unit vector on the horizontal plane (x, y). Based on the ground-plane prior that most object motion in road scenes occurs on the ground plane, the velocity estimation is restricted to the horizontal plane. For static background points, the observed Doppler velocity is assumed to be caused solely by the ego-vehicle’s 2D translational velocity $\mathbf { v } _ { s } \in \mathbb { R } ^ { 2 }$ . The expected Doppler velocity is therefore given by:

$$
\hat { v } _ { i } ^ { \mathrm { d o p } } = \mathbf { h } _ { i , x y } ^ { \top } \mathbf { v } _ { s } .\tag{2}
$$

The ego-velocity ${ \bf v } _ { s }$ is robustly estimated through RANSAC. At each iteration, a small subset of points is sampled to compute a tentative estimate of ${ \bf v } _ { s } .$ , and the residual errors between the estimated and measured Doppler values are evaluated over all points to form an inlier set $\tau _ { \mathrm { i n l i e r s } }$ . Finally, a least-squares regression is applied to the inlier set to obtain the final egovelocity estimate:

$$
\hat { \mathbf { v } } _ { s } = \operatorname* { m i n } _ { \mathbf { v } _ { s } } \sum _ { i \in \mathcal { T } _ { \mathrm { i n l i e r s } } } \left( v _ { i } ^ { \mathrm { d o p } } - \mathbf { h } _ { i , x y } ^ { \top } \mathbf { v } _ { s } \right) ^ { 2 } .\tag{3}
$$

Points located close to the sensor or at the periphery of the field of view may exhibit large Doppler variations even for the same static object due to differences in observation angle. Therefore, inlier candidates are prioritized from distant points (beyond 15 m) within the main viewing sector (±40<sup>◦</sup>) to ensure consistent and reliable estimation. The specific thresholds were empirically validated to provide the most reliable segmentation performance.

(ii) After estimating the ego-velocity $\hat { \mathbf { v } } _ { s }$ from Eq. 3, the expected Doppler velocity $\mathbf { h } _ { i , x y } ^ { \top } \hat { \mathbf { v } } _ { s }$ is computed for each point and compared with the measured Doppler value to obtain the residual $r _ { i } .$ . The residual indicates how similar each point’s velocity component is to the ego-motion: static background points exhibit small residuals, whereas independently moving objects show large values.

$$
r _ { i } = \big | \boldsymbol v _ { i } ^ { \mathrm { d o p } } - { \bf h } _ { i , x y } ^ { \top } \hat { \mathbf { v } } _ { s } \big | .\tag{4}
$$

Next, a threshold $\tau _ { \mathrm { b a s e } }$ is determined based on the median of the residuals within $\mathcal { T } _ { \mathrm { i n l i e r s } }$ . The median is less sensitive to outliers than the mean, preventing distortion from noise or a small number of dynamic points and providing a stable measure of the central tendency of the residual distribution. Using this threshold, the dynamic mask $\mathcal { M } _ { \mathrm { d y n } }$ is obtained as follows:

$$
\tau _ { \mathrm { b a s e } } = \mathrm { m e d i a n } \big ( \{ r _ { i } \mid i \in \mathcal { I } _ { \mathrm { i n l i e r s } } \} \big ) ,\tag{5}
$$

$$
\mathcal { M } _ { \mathrm { d y n } } = \{ i \mid r _ { i } > \tau _ { \mathrm { b a s e } } \} .\tag{6}
$$

As shown in Fig. 3-(b), the proposed method clearly separates dynamic points (blue) such as vehicles from static background structures (black), including guardrails and trees, while effectively suppressing Doppler noise and outlier-induced errors.

2) Dynamic Object Clustering: After separating dynamic points, a two-step clustering process is applied to group points belonging to the same moving object. The input to this stage consists of the dynamic points identified in the previous step through the residual-based thresholding, represented by the dynamic mask $\mathcal { M } _ { \mathrm { d y n } } .$

In the first stage, DBSCAN is applied to the dynamic points detected from the left and right Radars, grouping them into candidate clusters based on their 3D spatial proximity in $( x , y , z )$ . Here, $\epsilon _ { x y z }$ denotes the maximum inter-point distance that determines the spatial extent of a cluster, and minPts $x y z$ specifies the minimum number of points required to form a valid cluster:

$$
\mathrm { C l u s t e r } _ { x y z } = \mathrm { D B S C A N } ( \{ ( x _ { i } , y _ { i } , z _ { i } ) \} ; \epsilon _ { x y z } , \mathrm { m i n P t s } _ { x y z } ) .\tag{7}
$$

In the second stage, points within each spatial cluster are further grouped based on their Doppler velocities, allowing objects that are spatially adjacent but exhibit different motion patterns (e.g., two vehicles driving side by side) to be effectively separated:

$$
\mathrm { C l u s t e r } _ { d o p } = \mathrm { D B S C A N } ( \{ v _ { i } ^ { \mathrm { d o p } } \} ; \epsilon _ { d o p } , \mathrm { m i n P t s } _ { d o p } ) .\tag{8}
$$

As illustrated in Fig. 3-(c) and 3-(d), objects that cannot be distinguished using spatial information alone become clearly separable through the second-stage Doppler-based clustering.

3) Absolute Velocity Estimation: The absolute velocity estimation problem is formulated as a regularized RLS model using the Doppler measurements from the two sensors and the geometric relationship between each Radar and the target.

For a dynamic cluster consisting of $N _ { p }$ Radar points, let $\mathbf { h } _ { i } \in \mathbb { R } ^ { 2 }$ denote the LoS unit vector for point i, and let ${ \mathrm { d } } _ { i }$ represent the Doppler measurement corrected by the estimated ego-velocity. This can be written as:

$$
\mathrm { d } _ { i } = v _ { i } ^ { d o p } + \mathbf { h } _ { i } ^ { \top } \hat { \mathbf { v } } _ { s } .\tag{9}
$$

This relationship expands the observation model at each point i into $N _ { p }$ independent constraints, which can be stacked to form the following linear system:

$$
\begin{array} { r } { \mathbf { d } = \mathbf { H } \mathbf { v } , } \end{array}\tag{10}
$$

where $\mathbf { d } = [ \mathrm { d } _ { 1 } , \mathrm { d } _ { 2 } , \ldots , \mathrm { d } _ { N _ { p } } ] ^ { \top } \in \mathbb { R } ^ { N _ { p } }$ is the vector of egocorrected Doppler measurements, $\mathbf { H } = [ \mathbf { h } _ { 1 } ^ { \top } ; \ldots ; \mathbf { h } _ { N _ { v } } ^ { \top } ]$ is the matrix of LoS unit vectors, and $\mathbf { v } \in \mathbb { R } ^ { 2 }$ denotes the object’s

absolute (ground-truth) velocity. The velocity estimation can then be formulated as a regularized least-squares problem:

$$
\hat { \mathbf { v } } = \arg \operatorname* { m i n } _ { \tilde { \mathbf { v } } } \| \mathbf { H } \tilde { \mathbf { v } } - \mathbf { d } \| _ { 2 } ^ { 2 } + \lambda \| \tilde { \mathbf { v } } \| _ { 2 } ^ { 2 } ,\tag{11}
$$

where λ is a regularization parameter that suppresses numerical instability when the LoS vectors become nearly parallel, v˜ is the optimization variable, and vˆ denotes the estimated velocity obtained from the regularized least-squares solution.

The proposed stereo 4D Radar-based velocity estimation is significant in that it enables direct recovery of absolute velocity components that cannot be reconstructed using a mono sensor. By incorporating a regularization term, the method alleviates numerical instability that may arise when the LoS vectors are nearly parallel or when measurements are corrupted by noise, allowing robust estimation even under insufficient geometric constraints. The effectiveness and advantages of the proposed absolute velocity estimation are demonstrated in the experimental section through comparisons with mono 4D Radarbased velocity estimation methods.

## C. Stereo 4D Radar-based 3D Object Detection Network

In this section, we focus on the key components designed to effectively process stereo 4D Radar inputs: the Stereo Radar Pillar Feature Encoder (SR-PFE) and the Stereo Radar Bilateral Fusion Module (SR-BFM). SR-PFE encodes the various attributes of Radar measurements to generate rich BEV representations, while SR-BFM leverages stereo geometric information by aligning, refining, and fusing the left and right Radar features.

1) Stereo Radar Pillar Feature Encoding (SR-PFE): The proposed SR-PFE extends the encoding structure of Radar PillarNet [27]. It first partitions the input Radar point cloud into spatial pillars and then encodes the points within each pillar to generate BEV feature maps.

We use three types of Radar inputs: the left sensor (L), the right sensor (R), and an aggregated input (A) obtained by concatenating the left and right Radar point sets. For each input $k \in \{ \mathrm { L } , \mathrm { R } , \mathrm { A } \}$ , let $\mathcal { P } ^ { k } = \{ \mathbf { p } _ { j } ^ { k } \} _ { j = } ^ { M }$ denote the set of M points contained within a pillar. From these points, three types of feature vectors are constructed: a spatial feature vector $\mathbf { f } _ { s , j } ^ { k }$ using $\{ x , y , z \}$ , a reflection feature vector $\mathbf { f } _ { r , j _ { - } } ^ { k }$ using $\{ { \mathrm { P W } } \}$ and a velocity feature vector $\mathbf { f } _ { v , j } ^ { k }$ using $\{ v ^ { ' d o p } , v _ { x } ^ { a b s } , v _ { y } ^ { a b s } \}$ These feature vectors are passed into three encoders: a spatial encoder $\phi _ { s } ( \cdot )$ , a power encoder $\phi _ { r } ( \cdot )$ , and a velocity encoder $\phi _ { v } ( \cdot )$ . Each encoder produces a point-wise embedding, which is concatenated channel-wise, followed by max pooling within each pillar and scattering onto the BEV plane to obtain a sensor-specific feature map $\mathbf { F } _ { k }$ . The entire process can be summarized as:

$$
\mathbf { F } _ { k } = \mathrm { S c a t t e r } \left( \operatorname* { m a x } _ { j } \left[ \left[ \phi _ { s } ( \mathbf { f } _ { s , j } ^ { k } ) ; \phi _ { r } ( \mathbf { f } _ { r , j } ^ { k } ) ; \phi _ { v } ( \mathbf { f } _ { v , j } ^ { k } ) \right] \right] \right) .\tag{12}
$$

The generated feature maps ${ \mathbf { F } } _ { \mathrm { L } } , \ { \mathbf { F } } _ { \mathrm { R } } .$ , and $\mathbf { F } _ { \mathrm { A } }$ represent the BEV features extracted from the left Radar, right Radar, and aggregated input, respectively, and are used as inputs to the subsequent SR-BFM module.

![](images/f016fc48ad337bf5d7d1d8f8ad11b49361dcc44a4826f11a9de3d8a2edb8b47f.jpg)  
Fig. 4. Overview of the SR-BFM architecture. SR-BFM comprises an Attention Refinement (AR) block, which aligns and enhances stereo features, and a Gated Correlation-Fusion (GCF) block, which integrates them through correlation-aware refinement and pairwise gating.

2) Stereo Radar Bilateral Fusion Module (SR-BFM): To integrate the complementary features of stereo 4D Radar, we propose the SR-BFM. SR-BFM consists of three sequential components — the Sensor Identity-Positional Encoding (SI-PE) block, the Attention Refinement (AR) block, and the Gated Correlation-Fusion (GCF) block — which progressively align and fuse the left-right Radar features. Each block plays a distinct role in sensor identification, feature alignment, and feature fusion, and their effectiveness is quantitatively validated through ablation studies in Section IV-E.

a) SI-PE Block: Given the Radar feature maps extracted from the left, right, and aggregated encoders, the SI-PE Block enriches the representation by injecting sensor identity and positional information. First, a learnable Sensor Identity Embedding is applied to explicitly distinguish the inputs from each sensor. Then, a Fourier Positional Encoding module adds global spatial context over the BEV grid. Finally, a 1×1 convolution is used to align the feature dimensions, producing representations that are well suited for the subsequent fusion stages.

b) AR Block: The AR Block is responsible for aligning the left and right Radar features by mitigating spatial inconsistencies between sensors and producing representations that are interpretable within a shared embedding space. The block consists of two parallel branches, where each branch takes the features from one sensor as queries and applies local cross attention against the aggregated feature map $\mathbf { F _ { A } }$ used as keys and values. Formally, for the right- and left-Radar features $\mathbf { F _ { R } }$ and $\mathbf { F _ { L } }$ , the local cross attention operations are defined as:

$$
\tilde { \mathbf { F } } _ { \mathbf { R } } = \mathrm { C r o s s A t t n } ( \mathbf { F } _ { \mathbf { R } } , \mathbf { F } _ { \mathbf { A } } ) , \tilde { \mathbf { F } } _ { \mathbf { L } } = \mathrm { C r o s s A t t n } ( \mathbf { F } _ { \mathbf { L } } , \mathbf { F } _ { \mathbf { A } } ) ,\tag{13}
$$

where CrossAttn(·, ·) denotes a localized cross-attention operator that computes query-key similarity and weighted aggregation within a restricted spatial window. This process effectively aligns the left-right sensor observations to the unified coordinate frame provided by $\mathbf { F _ { A } }$ , enabling consistent fusion in subsequent stages.

After local geometric alignment, the AR Block further enhances feature quality by learning to reweight the channel and spatial dimensions, thereby amplifying semantically salient structures (e.g., vehicles, pedestrians) while suppressing background noise. Let CSAttn(·) denote the channel-spatial attention function. The refined stereo features are then obtained as:

$$
{ \bf F _ { R } ^ { \prime } } = \mathrm { C S A t t n } ( \tilde { \bf F } _ { \bf R } ) , \qquad { \bf F _ { L } ^ { \prime } } = \mathrm { C S A t t n } ( \tilde { \bf F } _ { \bf L } ) ,\tag{14}
$$

Through the combination of local cross attention-based geometric alignment and channel-spatial importance reweighting, the AR Block produces semantically refined and geometrically consistent stereo features, ${ \bf { F } } _ { \bf { R } } ^ { \prime }$ and $\mathbf { F _ { L } ^ { \prime } }$ . These refined representations are then passed to the GCF Block, which integrates complementary stereo information for final fusion.

c) GCF Block: The GCF block consists of two main components: (i) a pairwise gating module that adaptively selects reliable features from each sensor, and (ii) a correlationaware refinement module that further aligns and enhances the fused representation.

In the pairwise gating stage, the BEV-wise confidence weights $\mathbf { w } ^ { - } = [ \mathbf { w } _ { \mathrm { R } } , \mathbf { \bar { w } _ { \mathrm { L } } } ] \in \mathbb { R } ^ { 2 \times \bar { H } \times W }$ are estimated by jointly considering the refined left-right features ${ \bf { F } } _ { \bf { R } } ^ { \prime }$ and $\mathbf { F _ { L } ^ { \prime } }$ . As illustrated in Fig. 4, the two feature maps are concatenated along the channel dimension and passed through a lightweight head (1×1 Conv-BN-ReLU-Dropout-1×1 Conv), followed by a softmax normalization that enforces $\mathrm { w _ { R } } + \mathrm { w _ { L } } = 1$ at each BEV location:

$$
\begin{array} { r } {  { \mathbf { w } } = \mathrm { S o f t m a x } \big ( \Gamma ( [  { \mathbf { F } _ { \mathbf { R } } } ^ { \prime } ,  { \mathbf { F } _ { \mathbf { L } } } ^ { \prime } ] ) \big ) , \qquad { \mathbf { w } } = [  { \mathbf { w } } _ { \mathrm { R } } ,  { \mathbf { w } } _ { \mathrm { L } } ] , } \end{array}\tag{15}
$$

where $\Gamma ( \cdot )$ denotes a lightweight prediction module composed of a 1×1 convolution, normalization, non-linear activation, and dropout. The initial fused feature $\mathbf { F } _ { \mathrm { g a t e } }$ is then computed as:

$$
\mathbf { F _ { \mathrm { g a t e } } } = \mathrm { w _ { R } } \odot \mathbf { F _ { R } ^ { \prime } } + \mathrm { w _ { L } } \odot \mathbf { F _ { L } ^ { \prime } } .\tag{16}
$$

This weighted combination allows the network to rely more heavily on the viewpoint with higher confidence, ensuring robust fusion even when one sensor suffers from occlusion, weak reflections, or partial signal degradation.

In the correlation-aware refinement stage, the three inputs ${ \bf F } _ { \bf R } ^ { \prime } , \ { \bf F } _ { \bf L } ^ { \prime }$ , and $\begin{array} { r } { \Delta = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | \mathbf { F } _ { \mathbf { R } } ^ { \prime ( c ) } - \mathbf { F } _ { \mathbf { L } } ^ { \prime ( c ) } | } \end{array}$ are concatenated and processed in parallel by dilated convolutions $\mathrm { C o n v } _ { d } ( \cdot )$ with dilation rates $d \in \{ 1 , 2 , 3 \}$ . These operations highlight and correct regions of inconsistency between the two sensors, reducing geometric discrepancies:

TABLE I  
COMPARISON OF 3D OBJECT DETECTION PERFORMANCE WITH DIFFERENT RADAR INPUT CONFIGURATIONS, EXAMINING THE EFFECTS OF STEREO INPUT AND ABSOLUTE VELOCITY INCORPORATION.
<table><tr><td rowspan="2">Network</td><td rowspan="2">Mono</td><td rowspan="2"></td><td colspan="3">Stereo</td></tr><tr><td>without  $\overline { { ( v _ { x } ^ { a b s } , v _ { y } ^ { a b s } ) } }$ </td><td>with</td><td> $\overline { { ( v _ { x } ^ { a b s } , v _ { y } ^ { a b s } ) } }$ </td></tr><tr><td></td><td> $\overline { { A P _ { 3 D } } }$ </td><td> $\overline { { A P _ { B E V } } }$ </td><td> $\overline { { A P _ { 3 D } } }$   $\overline { { A P _ { B E V } } }$ </td><td> $\overline { { A P _ { 3 D } } }$ </td><td> $\overline { { A P _ { B E V } } }$ </td></tr><tr><td>RTNH [16]</td><td>49.10</td><td>49.82</td><td>56.72 57.37</td><td>57.03</td><td>57.71</td></tr><tr><td>RTNH+ [26]</td><td>50.10</td><td>50.73</td><td>57.14 57.83</td><td>57.29</td><td>57.93</td></tr><tr><td>RadarPillarNet [27]</td><td>48.02</td><td>49.52</td><td>57.23 58.25</td><td>58.01</td><td>58.95</td></tr><tr><td>SMURF [29]</td><td>39.93</td><td>41.84</td><td>50.37 50.89</td><td>50.46</td><td>51.05</td></tr><tr><td>RPFANet [28]</td><td>41.07</td><td>41.74</td><td>48.89 49.92</td><td>48.78</td><td>56.40</td></tr><tr><td>RadarMFNet [21]</td><td>47.77</td><td>48.86</td><td>49.71 50.48</td><td>50.20</td><td>57.93</td></tr><tr><td>MVFAN [32]</td><td>49.22</td><td>50.33</td><td>57.27 58.24</td><td>57.34</td><td>58.35</td></tr><tr><td>DADAN [30]</td><td>49.99</td><td>50.69</td><td>57.03 57.93</td><td>57.66</td><td>58.51</td></tr><tr><td>Proposed</td><td></td><td></td><td>58.56</td><td>59.56 58.92</td><td>59.73</td></tr></table>

$$
\mathbf { F _ { r e f } } { = } \mathrm { C o n v _ { 1 \times 1 } } \left( [ \mathrm { C o n v } _ { k } ( \mathbf { X } ) ] _ { k = 1 } ^ { 3 } \right) ,\tag{17}
$$

$$
{ \bf X } = \mathrm { C o n c a t } [ { \bf F } _ { \bf R } ^ { \prime } , { \bf F } _ { \bf L } ^ { \prime } , \Delta ] .\tag{18}
$$

The refined feature $\mathbf { F } _ { \mathrm { r e f } }$ is then concatenated with the gated fusion output $\mathbf { F } _ { \mathrm { g a t e } }$ along the channel dimension, and a 1×1 convolution produces the final fused feature map:

$$
\mathbf { F } _ { \mathrm { N e c k } } { = } \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { C o n c a t } [ \mathbf { F _ { r e f } } , \mathbf { F _ { g a t e } } ] ) .\tag{19}
$$

As a result, the GCF Block combines confidence-based gating with correlation-aware refinement to correct left-right discrepancies and produce a spatially consistent, semantically enhanced fused representation $\mathbf { F _ { N e c k } }$ This fused feature map is subsequently fed into the backbone and detection head to generate the final 3D object detection outputs.

## IV. EXPERIMENTS

In this section, we evaluate the performance of the proposed stereo 4D Radar-based 3D object detection framework. To this end, we construct a new stereo 4D Radar dataset that captures the same scenes from two spatially separated sensors. All experiments are conducted using this dataset, enabling a comprehensive assessment of both absolute velocity estimation and the resulting improvements in 3D object detection performance.

## A. Dataset

For this study, we constructed a multimodal dataset featuring stereo 4D Radars as the primary sensing modality, complemented by a LiDAR sensor for annotation and multiple forward-facing cameras for visualization.

The dataset comprises 37 sequences (22K frames) recorded in clear daytime driving scenarios, with about 66K 3D bounding boxes annotated for the Sedan class. Compared to existing public 4D Radar datasets, the scale of our dataset is sufficiently large for quantitative evaluation and generalization analysis [14], [15], [38], [39]. The structure and detailed specifications of the dataset are provided in the Appendix A.

## B. Experimental Setup and Metric

The proposed model is implemented using the PyTorch 1.12 framework and trained on an Ubuntu 22.04 system equipped with an NVIDIA RTX 4070 Ti GPU (12 GB). Training is performed for 30 epochs with an initial learning rate of 0.001 and a batch size of 8. The Region of Interest (RoI) for Radar points is set to $x \ \in \ [ 0 , 7 2 ] \mathrm { m } , \ y \in \ [ - 1 6 , 1 6 ] \mathrm { n }$ m, and $z \in [ - 2 , 7 . 6 ] \mathrm { m }$

For dynamic object clustering, the spatial proximity threshold is set to $\epsilon _ { x y z } = 2 . 5$ m and the Doppler similarity threshold to $\epsilon _ { d o p } = 0 . 5$ m/s, enabling both spatial separation of vehicles and fine-grained grouping based on velocity differences. We use min $\mathrm { P t s } _ { x y z } = 6$ for the first-stage spatial clustering to ensure sufficient point density, and minPts $_ { d o p } = 2$ in the second stage to separate objects with narrow velocity distributions (e.g., distant vehicles).

For 3D object detection evaluation, we follow the K-Radar benchmark [16] protocol and adopt Average Precision (AP) as the performance metric. Specifically, we report $A P _ { 3 D }$ and $A P _ { B E V }$ for the Sedan class at an intersection-over-union (IoU) threshold of 0.3.

C. Quantitative Analysis of Mono vs. Stereo 4D Radar Configurations

Table I presents a quantitative comparison of 3D object detection performance across various Radar-based networks under different input configurations. This evaluation aims to verify that the proposed stereo 4D Radar framework effectively mitigates the two fundamental limitations highlighted in the Introduction: (1) the sparsity of mono 4D Radar measurements after preprocessing, and (2) the incomplete and noisy Doppler velocity observations that lead to loss of reliable motion information.

First, we compare the performance of mono-Radar input with that of stereo 4D Radar input. Across all networks, introducing stereo input consistently improves both $A P _ { 3 D }$ and $A P _ { B E V }$ . For example, in RadarPillarNet, $A P _ { 3 D }$ increases from 48.02 to 57.23 and $A P _ { B E V }$ rises from 49.52 to 58.25. Similar trends appear in other networks such as RTNH [16] and RadarMFNet [21], demonstrating that combining spatially complementary observations from the left and right Radars effectively alleviates the extreme sparsity of mono 4D Radar measurements and enables the models to learn richer geometric representations.

![](images/4e4b3ed4ce79dafbf0cfd614e74fbffbbc8fdb7e0c0159b8838bea015e9c9cfc.jpg)  
Fig. 5. Qualitative comparison of velocity estimation between mono-based and stereo-based methods. Red and blue arrows indicate the ego-vehicle’s local X- and Y-axes, respectively, while green arrows represent the absolute velocity vectors of surrounding objects. Yellow boxes highlight notable dynamic objects for comparison.

Next, we compare stereo input using only Doppler measurements with stereo input augmented by the estimated absolute velocities. In most networks, adding absolute velocity information further improves both $A P _ { 3 D }$ and $A P _ { B E V }$ . This indicates that the motion information recovered through stereobased velocity estimation can be effectively utilized by existing detectors, enabling more accurate localization and motion interpretation of dynamic objects.

The proposed network achieves the highest performance with $A P _ { 3 D }$ of 58.92 and $A P _ { B E V }$ of 59.73, outperforming all existing 4D Radar-based detectors. These results confirm that our framework successfully integrates geometric complementarity from stereo 4D Radar with accurate absolute velocity estimation, thereby overcoming both the sparsity and incomplete Doppler limitations inherent to mono 4D Radar systems and significantly boosting 3D object detection performance.

## D. Absolute Velocity Estimation

In this section, we demonstrate the effectiveness of the proposed stereo 4D Radar-based velocity estimation in recovering the true motion of dynamic objects. To this end, we provide a qualitative comparison against conventional mono-Radar-based velocity estimation. Fig. 5 presents representative driving scenes, showing the camera image (left), the monobased estimation results (middle), and the proposed stereo 4D Radar-based estimation results (right).

TABLE II  
IMPACT OF SR-PFE AND SR-BFM ON STEREO 4D RADAR DETECTION. $N _ { a } , N _ { b } , N _ { c } ,$ AND $N _ { d }$ REPRESENT VARIANTS WHERE THE SI-PE BLOCK, AR BLOCK, GCF BLOCK, AND SR-PFE ARE REMOVED, RESPECTIVELY.
<table><tr><td rowspan="2">Network</td><td rowspan="2">SR-PFE</td><td colspan="2">SR-BFM</td><td rowspan="2"> $A P _ { B E V }$ </td><td rowspan="2"> $A P _ { 3 D }$ </td></tr><tr><td>SI-PE Block AR Block</td><td>GCF Block</td></tr><tr><td> $\overline { { N _ { a } } }$ </td><td>√</td><td></td><td>√ √</td><td>58.87</td><td>57.91</td></tr><tr><td> $N _ { b }$ </td><td>√</td><td>√</td><td>√</td><td>58.76</td><td>57.83</td></tr><tr><td> $N _ { c }$ </td><td> $\checkmark$ </td><td>√ √</td><td></td><td>58.02</td><td>56.72</td></tr><tr><td> $N _ { d }$ </td><td></td><td> $\checkmark$ </td><td>√  $\checkmark$ </td><td>59.10</td><td>58.01</td></tr><tr><td>ours</td><td> $\overline { { \checkmark } }$ </td><td> $\checkmark$ </td><td> $\checkmark$   $\overline { { \checkmark } }$ </td><td>59.73</td><td>58.92</td></tr></table>

Scene 1 and Scene 2 compare absolute velocity estimation for vehicles moving laterally. The mono-based method struggles to capture transverse motion, often producing distorted velocity vectors. In contrast, the stereo-based method leverages Doppler measurements from both left and right sensors, effectively compensating for missing motion components and recovering the true motion of the target object.

Scene 3 and Scene 4 highlight driving situations where vehicles exhibit complex motion patterns that cannot be reliably inferred from a mono 4D Radar viewpoint. In Scene 3, for example, the oncoming vehicle in the left lane (left yellow box) maintaining a straight trajectory, and the ego-lane vehicle (right yellow box) initiating a lane change to the left. The mono-based method misinterprets their motion directions due to the lack of complete velocity, whereas the stereo-based estimation clearly separates radial and transverse components, yielding accurate motion vectors.

Finally, Scene 5 highlights a case where the stereo configuration improves motion perception compared to the mono-Radar. The mono-based method fails to correctly classify a dynamic object, whereas the stereo approach utilizes the geometric complementarity between the two viewpoints to recover missing motion information and reliably identify the object as dynamic.

These results show that the proposed stereo 4D Radarbased velocity estimation robustly compensates for missing or incomplete motion information in various driving scenarios.

## E. Ablation Study

This section quantitatively analyzes the contribution of each component within the proposed SR-BFM and examines how the module’s architectural design influences the overall 3D object detection performance.

1) Contribution Analysis of SR-PFE and SR-BFM: Table II summarizes the contribution of each component within the proposed SR-BFM—namely the SI-PE Block, AR Block, and GCF Block—as well as the effect of removing SR-PFE.

Removing the GCF Block (N ) yields the largest degradation, with $A P _ { B E V } = 5 8 . 0 2 $ and $A P _ { 3 D } = 5 6 . 7 2$ , indicating that correlation-aware refinement and gating are central to effective stereo fusion. Eliminating the SI-PE Block $( N _ { a } )$ reduces performance to $A P _ { B E V } = 5 8 . 8 7$ and $A P _ { 3 D } = 5 7 . 9 1$ showing that sensor identity embeddings provide a stable reference frame for stereo alignment. Removing the AR Block $( N _ { b } )$ lowers accuracy to $A P _ { B E V } = 5 8 . 7 6$ and $A P _ { 3 D } = 5 7 . 8 3$ confirming that the cross-attention-based alignment mitigates left-right geometric inconsistencies. Excluding SR-PFE $( N _ { d } )$ also results in a noticeable drop $( A P _ { B E V } = 5 9 . 1 0 , A P _ { 3 D } =$ 58.01), demonstrating its role in producing discriminative BEV features prior to stereo fusion.

TABLE III  
PERFORMANCE COMPARISON OF THE SR-BFM SUB-COMPONENTS (COARSE BRANCH AND DILATED CONVOLUTION). $N _ { a }$ DENOTES THE VARIANT WITHOUT THE COARSE BRANCH, AND $N _ { b }$ DENOTES THE VARIANT WITHOUT DILATED CONVOLUTION.
<table><tr><td rowspan=1 colspan=1>Network</td><td rowspan=1 colspan=1>Coarse Branch D</td><td rowspan=1 colspan=1>ilated Convolution</td><td rowspan=1 colspan=1> $A P _ { B E V }$   $A P _ { 3 D }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { N _ { a } } }$  $N _ { b }$ </td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $\checkmark$ </td><td rowspan=1 colspan=1>59.39  58.3459.36  58.27</td></tr><tr><td rowspan=1 colspan=1>ours</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1>59.73  58.92</td></tr></table>

The full model (ours) achieves the highest performance, with $A P _ { B E V } = 5 9 . 7 3$ and $A P _ { 3 D } = 5 8 . 9 2 $ , verifying that SR-PFE and all SR-BFM components jointly enhance stereo 4D Radar detection.

2) Effectiveness ofDetailed Components in the GCF Block: Table III reports the effectiveness of the detailed components within the GCF Block, namely the Coarse Branch and the Dilated Convolution. Removing the Coarse Branch $( N _ { a } )$ results in a noticeable performance drop, with $A P _ { B E V } = 5 9 . 3 9$ and $A P _ { 3 D } = 5 8 . 3 4 $ . This suggests that capturing low-resolution global context is essential for maintaining wide-range alignment consistency during stereo feature fusion. When the Dilated Convolution module is removed $( N _ { b } )$ , the performance declines further to $A P _ { B E V } = 5 9 . 3 6$ and $A P _ { 3 D } = 5 8 . 2 7$ . This demonstrates that dilated receptive fields effectively correct geometric discrepancies. The full model (ours), which incorporates both components, achieves the highest performance with $A P _ { B E V } = 5 9 . 7 3$ and $A P _ { 3 D } = 5 8 . 9 2 $ , confirming that the multi-scale alignment and correlation-aware refinement jointly contribute to improving stereo fusion quality.

## V. CONCLUSION

In this paper, we addressed two fundamental limitations of 4D Radar perception—the severe sparsity of point clouds after preprocessing and the incomplete motion information caused by the radial nature of Doppler measurements. To overcome these challenges, we introduced a new stereo 4D Radar-based 3D object detection framework that restores the absolute motion of dynamic objects and effectively fuses the complementary observations from left and right sensors. The proposed design integrates stereo-based absolute velocity estimation with cross-attention-driven alignment and correlationaware refinement, enabling robust geometric consistency and enhanced representation quality beyond the capabilities of mono 4D Radar methods.

Extensive experiments on our stereo 4D Radar dataset demonstrate that the proposed framework outperforms existing state-of-the-art 4D Radar-based 3D object detection. In future work, we plan to extend the proposed framework toward tracking and multi-object motion reasoning using stereo 4D Radar, aiming to further enhance the completeness and reliability of Radar-based perception.

## APPENDIX

## A. Sensor Specification

To collect stereo 4D Radar data, we equipped a test vehicle with three types of sensors, as illustrated in Fig. 6. Two 4D Radars were mounted on the front bumper with a fixed baseline of 0.6 m. Each Radar operates in the 76-79 GHz band and is equipped with 12 transmit (Tx) and 16 receive (Rx) antennas. In addition, a 64-channel long-range LiDAR was installed at the center of the vehicle roof, and three forwardfacing cameras were mounted on the front roof section. The LiDAR and cameras were primarily used for data visualization and for generating annotations.

## B. Data Collection and Distribution

The dataset was collected in Daejeon, South Korea, using the sensor platform described earlier. All data were recorded under clear daytime conditions and span a diverse set of driving environments, including 10 urban, 11 highway, and 16 suburban sequences. This composition provides representative autonomous-driving scenarios while maintaining consistent environmental conditions. For training and evaluation, the dataset is split into 18K frames for training and 4K frames for testing.

A total of 66K 3D bounding boxes were annotated for the Sedan class. Specifically, an automated labeling pipeline [40] was first applied to the collected LiDAR data to obtain initial annotations, followed by manual refinement to improve label accuracy. The annotation focuses on a single class for two reasons. First, sedans constitute the majority of traffic participants in the collected sequences, providing sufficient data density and variability for reliable training and evaluation. Second, restricting the annotations to a dominant vehicle class avoids confounding factors arising from heterogeneous object types with differing Radar reflectivity patterns, enabling a clearer analysis of stereo 4D Radar-based velocity compensation and 3D detection performance. This design choice follows the common practice in prior Radar perception studies, where a primary vehicle class is analyzed first before extending to multi-class settings [16], [28], [29].

## REFERENCES

[1] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354–3361.

![](images/8ea1a7cae3807df5093701a8918801fdb44271a5e42d51746957dad1d6a30ce3.jpg)  
Fig. 6. Sensor configuration for the stereo 4D Radar dataset.

[2] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 621–11 631.

[3] M. J. Mirza, C. Buerkle, J. Jarquin, M. Opitz, F. Oboril, K.-U. Scholl, and H. Bischof, “Robustness of object detectors in degrading weather conditions,” in 2021 IEEE International Intelligent Transportation Systems Conference (ITSC). IEEE, 2021, pp. 2719–2724.

[4] G. Kim, S. Choi, and A. Kim, “Scan context++: Structural place recognition robust to rotation and lateral variations in urban environments,” IEEE Transactions on Robotics, vol. 38, no. 3, pp. 1856–1874, 2021.

[5] T. Shan and B. Englot, “Lego-loam: Lightweight and ground-optimized lidar odometry and mapping on variable terrain,” in 2018 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2018, pp. 4758–4765.

[6] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point cloud based 3d object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4490–4499.

[7] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, “Pointpillars: Fast encoders for object detection from point clouds,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 697–12 705.

[8] S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, “Pv-rcnn: Point-voxel feature set abstraction for 3d object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10 529–10 538.

[9] X. Chen, K. Kundu, Z. Zhang, H. Ma, S. Fidler, and R. Urtasun, “Monocular 3d object detection for autonomous driving,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2147–2156.

[10] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. L. Rus, and S. Han, “Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 2774–2781.

[11] M. Bijelic, T. Gruber, and W. Ritter, “Benchmarking image sensors under adverse weather conditions for autonomous driving,” in 2018 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2018, pp. 1773–1779.

[12] Y. Zhang, A. Carballo, H. Yang, and K. Takeda, “Perception and sensing for autonomous vehicles under adverse weather conditions: A survey,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 196, pp. 146–177, 2023.

[13] M. Kutila, P. Pyykonen, H. Holzh ¨ uter, M. Colomb, and P. Duthon,¨ “Automotive lidar performance verification in fog and rain,” in 2018 21st International conference on intelligent transportation systems (ITSC). IEEE, 2018, pp. 1695–1701.

[14] A. Palffy, E. Pool, S. Baratam, J. F. Kooij, and D. M. Gavrila, “Multiclass road user detection with 3+ 1d radar in the view-of-delft dataset,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4961–4968, 2022.

[15] L. Zheng, Z. Ma, X. Zhu, B. Tan, S. Li, K. Long, W. Sun, S. Chen, L. Zhang, M. Wan et al., “Tj4dradset: A 4d radar dataset for autonomous driving,” in 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022, pp. 493–498.

[16] D.-H. Paek, S.-H. Kong, and K. T. Wijaya, “K-radar: 4d radar object detection for autonomous driving in various weather conditions,” Advances in Neural Information Processing Systems, vol. 35, pp. 3819– 3829, 2022.

[17] Z. Pan, F. Ding, H. Zhong, and C. X. Lu, “Ratrack: moving object detection and tracking with 4d radar point cloud,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 4480–4487.

[40] M.-H. Sun, D.-H. Paek, S.-H. Song, and S.-H. Kong, “Efficient 4d radar data auto-labeling method using lidar-based object detection network,” in 2024 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2024, pp. 2616–2621.

[18] P. Kaul, D. De Martini, M. Gadd, and P. Newman, “Rss-net: Weaklysupervised multi-class semantic segmentation with fmcw radar,” in 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020, pp. 431–436.

[19] R. Nabati and H. Qi, “Centerfusion: Center-based radar and camera fusion for 3d object detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 1527–1536.

[20] R. Xu and Z. Xiang, “Rlnet: Adaptive fusion of 4d radar and lidar for 3d object detection,” in European Conference on Computer Vision. Springer, 2024, pp. 181–194.

[21] B. Tan, Z. Ma, X. Zhu, S. Li, L. Zheng, S. Chen, L. Huang, and J. Bai, “3-d object detection for multiframe 4-d automotive millimeter-wave radar point cloud,” IEEE Sensors Journal, vol. 23, no. 11, pp. 11 125– 11 138, 2022.

[22] M. Markel, Ed., Radar for Fully Autonomous Driving. Boston, MA: Artech House, 2022.

[23] J. Gamba, Radar Signal Processing for Autonomous Driving. Singapore: Springer, 2022.

[24] N. Scheiner, F. Kraus, N. Appenrodt, J. Dickmann, and B. Sick, “Object detection for automotive radar point clouds–a comparison,” AI Perspectives, vol. 3, no. 1, p. 6, 2021.

[25] Y. Zhou, L. Liu, H. Zhao, M. Lopez-Ben´ ´ıtez, L. Yu, and Y. Yue, “Towards deep radar perception for autonomous driving: Datasets, methods, and challenges,” Sensors, vol. 22, no. 11, p. 4208, 2022.

[26] S.-H. Kong, D.-H. Paek, and S. Lee, “Rtnh+: Enhanced 4d radar object detection network using two-level preprocessing and vertical encoding,” IEEE Transactions on Intelligent Vehicles, 2024.

[27] L. Zheng, S. Li, B. Tan, L. Yang, S. Chen, L. Huang, J. Bai, X. Zhu, and Z. Ma, “Rcfusion: Fusing 4-d radar and camera with bird’s-eye view features for 3-d object detection,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–14, 2023.

[28] B. Xu, X. Zhang, L. Wang, X. Hu, Z. Li, S. Pan, J. Li, and Y. Deng, “Rpfa-net: A 4d radar pillar feature attention network for 3d object detection,” in 2021 IEEE International Intelligent Transportation Systems Conference (ITSC). IEEE, 2021, pp. 3061–3066.

[29] J. Liu, Q. Zhao, W. Xiong, T. Huang, Q.-L. Han, and B. Zhu, “Smurf: Spatial multi-representation fusion for 3d object detection with 4d imaging radar,” IEEE Transactions on Intelligent Vehicles, 2023.

[30] X. Wang, J. Li, J. Wu, S. Wu, and L. Li, “Dadan: Dynamic-augmented and density-aware network for accurate 3d object detection with 4d radar,” IEEE Sensors Journal, 2025.

[31] Y. Yang, J. Liu, H. Liu, and G. Jiang, “Radar m3-net: Multi-scale, multilayer, multi-frame network with a large receptive field for 3d object detection,” Expert Systems with Applications, p. 127515, 2025.

[32] Q. Yan and Y. Wang, “Mvfan: Multi-view feature assisted network for 4d radar object detection,” in International Conference on Neural Information Processing. Springer, 2023, pp. 493–511.

[33] W. Shi, Z. Zhu, K. Zhang, H. Chen, Z. Yu, and Y. Zhu, “Smiformer: Learning spatial feature representation for 3d object detection from 4d imaging radar via multi-view interactive transformers,” Sensors, vol. 23, no. 23, p. 9429, 2023.

[34] X. Peng, M. Tang, H. Sun, K. Bierzynski, L. Servadei, and R. Wille, “Mufasa: Multi-view fusion and adaptation network with spatial awareness for radar object detection,” in International Conference on Artificial Neural Networks. Springer, 2024, pp. 168–184.

[35] X. Bi, C. Weng, P. Tong, B. Fan, and A. Eichberge, “Maff-net: Enhancing 3d object detection with 4d radar via multi-assist feature fusion,” IEEE Robotics and Automation Letters, 2025.

[36] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[37] M. Ester, H.-P. Kriegel, J. Sander, X. Xu et al., “A density-based algorithm for discovering clusters in large spatial databases with noise,” in kdd, vol. 96, no. 34, 1996, pp. 226–231.

[38] M. Meyer and G. Kuschk, “Automotive radar dataset for deep learning based 3d object detection,” in 2019 16th european radar conference (EuRAD). IEEE, 2019, pp. 129–132.

[39] J. Rebut, A. Ouaknine, W. Malik, and P. Perez, “Raw high-definition´ radar for multi-task learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17 021– 17 030.