# HuLiGen: Human LiDAR Generation from Parametric Body Models

Salma Galaaoui1,2 Nermin Samet1\*David Picard2\*

1 Valeo.ai, Paris, France

2 LIGM, CNRS, Univ Gustave Eiffel, ENPC, Institut Polytechnique de Paris, Marne-la-Vallée, France

![](images/b3f2f19a4cc29b26428ca6297351a4d190899352d8a19a6b13ed1e05b6be3fdf.jpg)  
Figure 1. Learning realistic human LiDAR observations from real data. Existing synthetic pipelines generate articulated human point clouds by ray casting parametric body models, which represent the underlying body surface but omit clothing, hair and other surface variations. This also affects the resulting point distribution, often leading to fewer and less consistent points than in real observations. In the left example, the woman's coat is not rendered and sensor-specific noise and measurement irregularities are largely absent from the simulated sample. On the right, the book is captured by our generation while absent in ray casting. In contrast, our method learns these characteristics directly from real data, producing point clouds with more realistic geometry, point distributions and sensor effects. The camera images are shown only for visualization and are not used by our pipeline.

## Abstract

LiDAR point clouds of humans are extremely expensive to collect and annotate, thus represent a scarce resource that hinders the development of human analysis using this modality. To alleviate this scarcity, prior work relies on simulated human LiDAR, but such samples do not fully reflect the geometry and sensing characteristics of real observations. In contrast, we introduce HuLiGen, a generative model that generates human LiDAR point clouds from a parametric body model, using a point transformer trained with a flow-matching objective. We show that our generated point clouds are closer to the real capture distribution. Using HuLiGen to generate synthetic data, we propose a synthetic-only pretraining scheme for LiDAR-based HPE that achieves state-of-the-art performance, with even larger gains in low-annotation and low-data regimes, where MPJPE is reduced by up to 50%. Code, models and generated samples are available at https://github.com/ valeoai/HuLiGen.

## 1. Introduction

3D human pose estimation (HPE) is an important task for human-centric perception in embodied systems. Accurate estimation of human body pose can provide fine-grained cues for pedestrian behavior understanding and intention prediction [32], as well as trajectory forecasting [2, 15], which are important for collision avoidance and interactionaware motion planning [28]. The geometric nature of Li-DAR makes it particularly attractive for these applications, but also introduces a fundamental bottleneck: human Li-DAR data with accurate 3D pose annotations remains difficult to obtain at scale.

Constructing such datasets requires substantial annotation effort. LiDAR points on humans are sparse, viewpoint dependent, and frequently affected by self-occlusion and external occlusions, while accurate pose supervision typically requires synchronized motion-capture systems, additional cameras or inertial sensors. Recent datasets such as SLOPER4D [7] and the Waymo Open Dataset (WOD) [39] provide real-world point clouds together with human pose annotations which help the progress in LiDAR-based human pose estimation. Nevertheless, the available data still covers only a limited range of human poses and body shapes. For example, SLOPER4D contains more than 100K LiDAR frames but only 12 recorded subjects, while WOD provides approximately 10K object frames with 3D keypoint annotations. In contrast, large-scale human bodymodel repositories such as AMASS [24], provide a much richer diversity of human poses and body configurations, but lack corresponding LiDAR observations.

To compensate for the limited amount and diversity of annotated human LiDAR data, a common strategy is to generate synthetic samples from parametric body models using ray casting [3, 35, 40, 42]. However, the realism of these samples is limited by the manually defined rendering process [25, 49]. Real LiDAR observations depend not only on the underlying body geometry, but also on clothing and surface variations, viewpoint, range, sensor configuration and measurement uncertainty. Modeling these effects requires explicit assumptions and hand-crafted noise or sensor models.

In this work, we explore a different paradigm: rather than explicitly simulating the interaction of the LiDAR with the body model to obtain the point cloud, we learn it directly from real lidar observations. Given real human point clouds paired with corresponding body models, we train a conditional flow-matching model that acts as a learned LiDAR renderer, modeling how a given human pose and shape is observed by a real sensor. This formulation learns real Li-DAR characteristics such as sparsity, viewpoint- and rangedependent sampling and uncertainty. It can also capture deviations from the underlying body model caused by clothing, hair and other surface variations (see Fig. 1). Finally, it decouples pose diversity from sensor realism: large-scale mocap data provides diverse human poses, while real Li-DAR data defines how these poses are observed.

Once trained, the model can generate LiDAR observations for previously unseen poses, allowing us to leverage existing large-scale mocap datasets with diverse poses and body configurations to generate sensor-specific human point clouds. Beyond evaluating generation quality using standard generative metrics, we further assess the generated point clouds on the downstream task of 3D human pose estimation. We show that pretraining a pose estimator on generated human LiDAR samples, followed by finetuning on real data, significantly improves performance in both full-data and low-data regimes. Experiments with different fractions of the real training set further show that our generated data can compensate for limited labeled real data.

Our main contributions are:

• We introduce HuLiGen, to the best of our knowledge, the first learned generative mapping from parametric human body models to LiDAR point-clouds, learning real LiDAR characteristics.

• We demonstrate the practical value of the learned Li-DAR distribution through downstream 3D HPE. Pretraining on generated human LiDAR observations followed by finetuning on real data reduces MPJPE from 6.02 cm to 5.04 cm on SLOPER4D and from 6.74 cm to 5.85 cm on WOD, achieving state-of-the-art performance on both datasets.

• We demonstrate strong gains in the low-annotation setting on WOD, where our pretraining pipeline reduces MPJPE by about 50% using only 1% and 10% of the available pose annotations.

• We further show that our pretraining pipeline remains effective in low-data settings, reducing MPJPE by about 30% and 25% on SLOPER4D [7], and by 42% and 36% on WOD [39], when using only 10% and 20% of the real training data, respectively.

## 2. Related works

Lidar-based 3D human pose estimation. Lidar-based human pose estimation is challenging due to the sparsity, non-uniform density and partial observation of the human body. Early approaches therefore focused on extracting more structured representations from sparse measurements. LidPose [17] projects human point clouds into structured 2D representations, LPFormer [45] combines BEV and sparse voxel features, and VoxelKP [37] operates directly on high-resolution sparse voxels. LiDAR-HMR [8] leverages Pose Regression Network (PRN) to estimate 3D keypoints directly from human LiDAR point clouds through probabilistic point-wise voting and attention-based refinement. Other works compensate for missing geometric information using complementary cues: MMVP [9] and HUM3DIL [46] fuse RGB and LiDAR information, while LiDAR-HMP [12] exploits temporal context. More recently, the focus has shifted toward learning stronger human-specific representations. UniPVU-Human [42] learns body-part and motion priors before downstream pose estimation, while DAPT [3] explicitly addresses variations in LiDAR point density through density-aware feature learning. Beyond architectural improvements, GC-KPL [40] tackles the limited availability of labeled LiDAR data by combining synthetic pretraining with geometric and temporal self-supervision on unlabeled real data.

Synthetic data for lidar-based 3D HPE. Beyond developing better architectures and representations, prior work has also attempted to address the scarcity of annotated human LiDAR data. One line of research introduces additional supervision from images, using 2D annotations or pseudo-labels together with LiDAR to reduce the dependence on 3D pose labels [4, 6, 11, 29, 48]. For LiDARonly methods, two strategies are commonly adopted to increase training diversity and quantity. VoxelKP [37], LP-Former [45], and LidPose [17] rely on extensive data augmentation, such as frustum dropout, frame mixing, body cropping and geometric transformations, to perturb existing real samples. A second direction generates entirely new LiDAR observations from parametric human models using simulated LiDAR ray casting and hand-designed perturbations, such as laser-level masks for occlusion [3, 35, 40, 42]. This makes it possible to exploit large-scale body-model repositories such as AMASS [24], even though they do not contain paired LiDAR measurements. In contrast, our approach preserves this ability to leverage large-scale human body data while learning the mapping from body models to LiDAR observations directly from real data.

Lidar point cloud generation. LiDAR generation has evolved from raycasting simulators in virtual environments [10] to learned generative models based on energy functions, discrete latents and diffusion [27, 31, 41, 49]. Most of these methods, however, focus on full-scene generation rather than the fidelity of individual objects. Recent object-level LiDAR generators explicitly model foreground objects. These methods generate localized objects with diffusion, using conditioning such as object class, position, or viewpoint, and can integrate the generated instances back into complete driving scenes [16, 43]. Despite this progress, existing LiDAR generators are not designed to generate humans conditioned on their underlying body pose and shape.

## 3. Method

## 3.1. Overview

Our goal is to learn to predict how a LiDAR would capture a human, as represented by a parametric body model. Formally, given a human point cloud P and its corresponding body model B, we want to learn the conditional distribution

$$
p ( \mathbf { P } \mid \mathbf { B } ) ,\tag{1}
$$

where B can be further decomposed into $[ \mathbf { o } , \beta , \theta ]$ the oriented location of the subject, the body shape and local body pose representations respectively.

We model this conditional distribution using flow matching (FM) parametrized by a neural network operating directly on the 3D point cloud. Starting from a random Gaussian point cloud, this flow-matching model learns to gradually move the 3D points towards that of a human Li-DAR capture, conditionally to $[ \mathbf { o } , \beta , \theta ]$ that correspond to the body model B of the human. Because each LiDAR produces very different distributions of point clouds, we train a separate model for each acquisition setting so that sensordependent characteristics are captured independently.

## 3.2. Human-centric parameterization

Because humans are articulated and their capture by a Li-DAR results in very local interactions, the choice of coordinate system is crucial to provide relevant information to the flow-matching model. Let us denote by subscript w the world coordinates and by subscript l the LiDAR coordinates.

![](images/6ce9f1d8a5a0ad8a370a4be8cf6fe469b2821b1e78908066f2dce7b36a7a8a89.jpg)  
Figure 2. Root-joint axes across different pose types from the SLOPER4D [7] dataset. We use the azimuth of the lateral axis for canonicalisation, as its projection onto the horizontal plane remains well defined across the different poses. In contrast, the forward axis can approach the vertical under strong hip flexion, causing its horizontal projection to shrink (at the risk of reaching 0°) and making the corresponding azimuth ill-conditioned.

The human point cloud $\mathbf { P } = \{ \mathbf { p } _ { i } \} _ { 1 \leq i \leq N }$ is usually not touching the surface of the body model because of clothing and noise. Since any side of the human can be captured, the distribution of P is centered around the body model in expectation over all possible captures. As such, we propose to use the center of the body model in LiDAR coordinates as the origin. Noting $\mathbf { t } _ { w }$ the center of the body model in world coordinates, its LiDAR counterpart is given by using a translation $\mathbf { t } _ { l } = \mathbf { T } _ { w  l } \mathbf { t } _ { w }$ , and each point of P is expressed relative to it:

$$
x _ { i } = \mathbf { T } _ { w  l } \mathbf { p } _ { i } - \mathbf { t } _ { l } .\tag{2}
$$

To encode the distance and orientation of the body with respect to the sensor (the $\mathbf { t } _ { l }$ part of B), we use cylindrical coordinates:

$$
\phi = \mathrm { a t a n 2 } ( t _ { l _ { y } } , t _ { l _ { x } } ) , \quad \rho = \sqrt { t _ { l _ { x } } ^ { 2 } + t _ { l _ { y } } ^ { 2 } } , \quad z = t _ { l _ { z } } .\tag{3}
$$

φ, $\rho ,$ and z respectively describe azimuth, radial distance and height relative to the LiDAR.

![](images/3c9ec80233a9e7a10482b1268c4360a24107af89ee72cdc6fafceef92c142310.jpg)  
Figure 3. Our synthetic pretraining pipeline. We sample local body poses θ from AMASS and bootstrap the remaining conditioning parameters, $\mathbf { o } , \beta$ as well as number of target points from the target dataset. These conditions are provided to our flow-matching model to generate novel human LiDAR observations. We then pretrain the HPE model on the generated samples and finetune it on real target dataset, yielding substantial performance gains

In addition to center on the body, we also adopt a cannonical reference frame aligned on the global body yaw. The reasoning behind this step is to let the flow-matching model operate in a frame that does not depend on the pitch bend in the pose (e.g., a person tying their shoes) as humans tend to not tilt side to side (see Fig. 2). More formally, denoting $\mathbf { R } _ { 0 }$ the root joint rotation matrix, the azimuth of the lateral axis is obtained with

$$
\psi = \mathrm { a t a n 2 } \left( [ { \bf R } _ { 0 } ] _ { 1 0 } , [ { \bf R } _ { 0 } ] _ { 0 0 } \right) .\tag{4}
$$

Then, the points are transformed into the canonical body frame with the rotation

$$
\mathbf { R } _ { z } ( - \psi ) \cdot \mathbf { x } _ { i } , \quad \mathbf { R } _ { z } ( - \psi ) = \left[ { \begin{array} { c c c } { \cos \psi } & { \sin \psi } & { 0 } \\ { - \sin \psi } & { \cos \psi } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} } \right]\tag{5}
$$

and the same rotation is applied to all body joints to keep a consistent reference frame. Similarly, we apply this correction to the azimuth encoding of the LiDAR by computing $\phi _ { \mathrm { r e l } } = \big ( ( \phi - \psi + \pi )$ mod $2 \pi ) - \pi$ . After the generation, the inverse rotation is applied to recover the generated point cloud in the original LiDAR frame.

## 3.3. Condition encoding

Our conditioning B combines three complementary sources of information:

$$
{ \tiny \left( \underbrace { \phi _ { \mathrm { r e l } } , \rho , z } _ { \mathrm { o r i e n t e d ~ b o d y ~ l o c a t i o n ~ { \bf o } ~ } } , \underbrace { \beta } _ { \mathrm { b o d y ~ s h a p e } } , \underbrace { \theta } _ { \mathrm { l o c a l ~ b o d y ~ p o s e } } \right) } ,\tag{6}
$$

that require specific encoding each.

Oriented body location condition. Because angular variables are discontinuous at —π and π, we first represent $\phi _ { \mathrm { r e l } }$ through its sine and cosine components and apply a cyclic embedding. The radial distance $\rho$ and height z are represented using Fourier features. This provides 3 sensorrelated condition tokens.

Body shape condition. Using the parametric body model for the shape, each of the shape coefficients is independently embedded into a different token using an MLP to keep these parameters disentangled, leading to different condition tokens.

Local body pose condition. Using the body model for the pose θ, we convert the axis-angle to 3D cartesian coordinates using forward kinematics and embed each of the 3D body joints using a PointNet [30] model to obtain one token per joint. To retain the semantic information of the joints, we keep their order the same and add a learned positional encoding.

The condition encoding results in the concatenation of all the tokens in a single sequence of tokens.

## 3.4. 3D Point cloud flow transformer

To generate the 3D LiDAR point cloud from the condition embedding tokens, we train a point-based Diffusion Transformer [26] with a flow-matching objective [20].

Given a human point cloud in canonical frame ${ \bf x } _ { 1 } ~ \in$ $\mathbb { R } ^ { N \times 3 }$ and a random initial point cloud $\mathbf { X } _ { 0 } \sim \mathcal { N } ( 0 , 1 )$ , we define the linear interpolant ${ \bf X } _ { t } = t { \bf X } _ { 1 } + ( 1 - t ) { \bf X } _ { 0 } , t \mathrm { ~ } \hat { \bf \Lambda }$ $\mathcal { U } ( 0 , 1 )$ , that corresponds to the following differential equation associated with the velocity field $v _ { t } \colon$

$$
\mathrm { d } \mathbf { X } _ { t } = v _ { t } \mathrm { d } t ,\tag{7}
$$

$$
\begin{array} { r } { v _ { t } = \mathbf { X } _ { 1 } - \mathbf { X } _ { 0 } . } \end{array}\tag{8}
$$

Following this velocity field allows to recover $\mathbf { X } _ { 1 }$ starting from the random point cloud $\mathbf { X } _ { 0 }$

The transformer is trained to approximate $v _ { t }$ It takes as input $\mathbf { X } _ { t } , t$ and the condition embedding tokens, and output a velocity field $v _ { \Theta } ( \mathbf { X } _ { t } , t , \mathbf { B } )$ , optimized using the following regression loss:

$$
\mathcal { L } = \Vert v _ { t } - v _ { \Theta } ( \mathbf { X } _ { t } , t , \mathbf { B } ) \Vert ^ { 2 } ,\tag{9}
$$

optimized by back-propagation.

Once the transformer is trained, we generate novel Li-DAR point clouds by simply sampling an initial random point cloud $\mathbf { X } _ { 0 } \sim \mathcal { N } ( 0 , 1 )$ and solving the differential equation $\mathbf { X } _ { t } = v _ { \Theta } ( \mathbf { X } _ { t } , t , \mathbf { B } ) \mathrm { d } t$ using an Euler sampling scheme. Notice that using a transformer based architecture allows us to sample varying amounts of points N without the need for architectural changes.

## 4. Experiments

Implementation details. To retain the characteristic of each LiDAR setup, we train separate models for different datasets. In the following experiments, we use a modified DiT-XS/4 configuration from [16], with 12 transformer blocks, a hidden dimension of 192, and three attention heads.

We train FM models for 200 epochs with a batch size of 128 and a learning rate of 1e-4. To handle variable-size point clouds, we pad each sample to the maximum number of points within the batch. For generation, we follow [23] and solve the corresponding ordinary differential equation using the Euler solver with 50 sampling steps. All models are trained on a single NVIDIA H200 GPU and take roughly 15 and 3 hours to train on SLOPER4D and WOD, respectively.

Datasets. We conduct experiments on SLOPER4D [7] and the Waymo Open Dataset (WOD) [39]. SLOPER4D is an outdoor human motion-capture dataset containing synchronized LiDAR point clouds and RGB images, with ground-truth human meshes and 3D keypoints obtained using motion-capture devices. Since the dataset does not provide an official train/test split, we follow LiDAR-HMR [8] and use 24,9k samples for training and 8,1k samples for testing.

For WOD, we use the human keypoint annotations provided with the v2.0 release, containing 8,1k annotated human instances for training and 1,8k for testing. As WOD does not provide ground-truth human meshes, we adopt the pseudo-mesh annotations from [8].

Generative quality metrics. We evaluate generation quality on the validation set of each dataset using both sample-level and distribution-level metrics.

We measure the geometric similarity between a generated point cloud and the real observation with the same conditioning input using the Chamfer Distance (CD).

We assess distributional realism using the Fréchet Point-Net Distance (FPD) [38], adapted from FID [13]. We extract point-cloud representations from the PRN network [8], using the features after its final multi-head self-attention layer and before the FCN head. We train a separate PRN for each dataset. We similarly adapt the Kernel Inception Distance (KID) [5], which is particularly suitable for smaller sample sets, and denote the resulting metric as Kernel Point-Net Distance (KPD).

Finally, we compare the real and generated pointcloud distributions using coverage (COV) [1] and nearestneighbor accuracy (1-NNA) [22, 44] following [26], computed using CD. COV measures how much of the real distribution is covered by generated samples, while 1-NNA measures how easily real and generated samples can be separated using nearest neighbors. A 1-NNA of 50% indicates indistinguishable distributions.

Downstream evaluation. To demonstrate the practical impact of our method, we use our trained FM models to generate new human LiDAR observations from poses in the AMASS database [24]. Specifically, we use KIT and HumanEva sequences and filter them to walking, standing and running, resulting in 47,494 unique body poses. We sample the local body pose from AMASS, while bootstrapping the global body orientated location, shape and target point count from the target dataset, i.e., SLOPER4D or WOD. This places novel poses in realistic target-domain observation settings and allows the generator to produce realistic point densities and off-surface structures, rather than being limited to the returns obtained from ray casting a bare body mesh. This provides a source of synthetic LiDAR samples with corresponding 3D keypoint annotations. We then pretrain a LiDAR-based 3D human pose estimator on the generated data, using the publicly available PRN [8] as the downstream model. In a second stage, we finetune the pretrained PRN on real LiDAR data. We illustrate this pipeline in Fig. 3. In both stages, we train PRN using 15 keypoints for 50 epochs using the Adam optimizer, with a batch size of 64 and a learning rate of 5e-4. We evaluate downstream performance using MPJPE and PA-MPJPE, reported in centimeters; in the tables, PA denotes PA-MPJPE.

## 4.1. Experimental analysis

Conditioning ablations. Since SLOPER4D provides ground-truth human meshes, we conduct our ablation experiments on this dataset and report the generative quality metrics on the validation set. We begin with a simple setup without canonical coordinate transformation, conditioning only on the body pose θ while discarding the shape parameter β. As shown in Tab. 1, both canonical transformation and shape conditioning improve performance when introduced separately and combining them yields further gains. Finally, replacing the angular pose with joint coordinates obtained through forward kinematics achieves the best performance.

We also compare our generated samples against a raycasting baseline. For this baseline, we raycast the meshes from the corresponding validation sets using the respective LiDAR sensor specifications: 64 beams with a 20° vertical FoV for WOD and 128 beams with a 45°vertical FoV for SLOPER4D. As shown in Tab. 4, our generated point clouds achieve better generative quality.

Table 1. Conditioning ablation study on SLOPER4D using generative quality metrics. Canonical align.: √ = canonical axis transformation applied; X = no canonical transformation. Shape param.: √ = shape parameters included in the conditioning; X = shape parameters excluded from the conditioning. Cartesian joints: √ = Cartesian joint coordinates with PointNet embeddings; X = 6D rotation coordinates with cyclic embeddings.
<table><tr><td>Canonical align.</td><td>Shape param.</td><td>Cartesian joints</td><td>CD↓ 1-NNA↓</td><td>COV↑</td><td>FPD↓</td><td>KPD↓</td></tr><tr><td>X</td><td>X</td><td>X</td><td>0.02983 90.65</td><td>38.93</td><td>0.1857</td><td>0.01357</td></tr><tr><td>√</td><td>X</td><td>X</td><td>0.02836 90.46</td><td>40.21</td><td>0.1890</td><td>0.01689</td></tr><tr><td>X</td><td>√</td><td>X</td><td>0.02859 91.43</td><td>39.20</td><td>0.1725</td><td>0.01560</td></tr><tr><td>J</td><td>√</td><td>X</td><td>0.02718 88.11</td><td>43.99</td><td>0.1277</td><td>0.01273</td></tr><tr><td>√</td><td></td><td>√</td><td>0.02648 84.45</td><td>45.17</td><td>0.1508</td><td>0.01124</td></tr><tr><td colspan="2">Raycasting</td><td></td><td>0.02470 95.45</td><td>37.77</td><td>0.1390</td><td>0.01177</td></tr></table>

Table 2. Generative quality comparison on WOD.
<table><tr><td>Method</td><td>CD</td><td>1-NNA</td><td>COV↑</td><td>FPD</td><td>KPD</td></tr><tr><td>Raycasting</td><td>0.0513</td><td>94.23</td><td>22.49</td><td>0.6051</td><td>0.07869</td></tr><tr><td>HuLiGen</td><td>0.0566</td><td>75.64</td><td>37.55</td><td>0.0961</td><td>0.00422</td></tr></table>

Table 3. Effect of synthetic pretraining on LiDAR-based 3D HPE. \* denotes results obtained by training on real data from scratch, without pretraining. ∆ values report the change in MPJPE relative to training from scratch without pretraining.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Syn. pre. data size</td><td colspan="2">Pretraining</td><td colspan="2">Finetuning</td><td rowspan="2">-Δ%</td></tr><tr><td>MPJPE</td><td>PA-</td><td>MPJPE</td><td>PA-</td></tr><tr><td>S4D</td><td>N/A</td><td>N/A</td><td>N/A</td><td>6.02*</td><td>4.58*</td><td>N/A</td></tr><tr><td></td><td>8k</td><td>15.13</td><td>9.31</td><td>5.18</td><td>4.07</td><td>13.95</td></tr><tr><td></td><td>24k</td><td>12.60</td><td>7.63</td><td>5.05</td><td>4.02</td><td>16.11</td></tr><tr><td></td><td>48k</td><td>11.09</td><td>7.03</td><td>5.04</td><td>3.97</td><td>16.28</td></tr><tr><td rowspan="4">WOD</td><td>N/A</td><td>N/A</td><td>N/A</td><td>6.74*</td><td>5.09*</td><td>N/A</td></tr><tr><td>8k</td><td>10.86</td><td>8.62</td><td>6.21</td><td>4.65</td><td>7.86</td></tr><tr><td>24k</td><td>10.63</td><td>8.53</td><td>6.03</td><td>4.45</td><td>10.53</td></tr><tr><td>48k</td><td>10.79</td><td>8.51</td><td>5.85</td><td>4.37</td><td>13.20</td></tr></table>

In Tab. 2, we also report generative quality metrics for our final model on WOD and compare it against the raycasting baseline. Consistent with the results on SLOPER4D, our model achieves better overall generative quality. We provide additional details regarding our method and ablations in the supplementary material.

Effect of pretraining with generated samples on LiDARbased 3D HPE. We further evaluate the quality and practical utility of our generated samples on downstream LiDAR-based 3D human pose estimation. We first pretrain PRN on our generated human LiDAR data and then finetune the pretrained model on real human LiDAR instances. The results are summarized in Tab. 3. For both datasets, pretraining consistently outperforms training from scratch across different synthetic pretraining set sizes, and performance further improves as the amount of generated data increases. We also observe that the pretrained model alone, without real-data finetuning, achieves reasonable performance.

Table 4. Comparison of raycasted and HuLiGen-generated samples using our pretraining pipeline on the downstream LiDAR-based 3D HPE task. ∆ values report the change in MPJPE relative to training from scratch without pretraining.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Pretraining</td><td colspan="2">Finetuning</td><td rowspan="2">-Δ%</td></tr><tr><td>MPJPE</td><td>PA-</td><td>MPJPE</td><td>PA-</td></tr><tr><td rowspan="3">SOR4D</td><td>HuLiGen</td><td>11.09</td><td>7.03</td><td>5.04</td><td>3.97</td><td>16.28</td></tr><tr><td>Rc — bootstr.</td><td>11.03</td><td>7.26</td><td>5.47</td><td>4.26</td><td>9.14</td></tr><tr><td>Rc — direct</td><td>21.95</td><td>9.73</td><td>5.27</td><td>4.16</td><td>12.45</td></tr><tr><td rowspan="3">WOD</td><td>HuLiGen</td><td>10.79</td><td>8.51</td><td>5.85</td><td>4.37</td><td>13.20</td></tr><tr><td>Rc — bootstr.</td><td>15.54</td><td>10.39</td><td>5.98</td><td>4.42</td><td>11.28</td></tr><tr><td>Rc — direct</td><td>15.11</td><td>10.04</td><td>6.35</td><td>4.71</td><td>5.79</td></tr></table>

Comparison with raycasting pretraining. We further compare pretraining with our generated samples against raycasted samples using the same set of approximately 48k AMASS [24] meshes. For ray casting, we consider two settings. First, we ray cast the AMASS meshes using their original pose and body parameters, while simulating only the LiDAR beam pattern at the median range of the targetdomain sensor. Second, for a fair comparison with our generated data, we also bootstrap the body orientation, translation and shape parameters using real LiDAR samples before ray casting. In the tables, we denote the first setting as Rc — direct and the second as Rc — bootstr. As shown in Tab. 4, pretraining with our generated samples consistently outperforms both raycasting variants in downstream 3D HPE.

## 4.2. Comparison to the-state-of-the-art

We compare our pretraining pipeline using HuLiGengenerated samples against existing LiDAR-based 3D HPE methods on the validation sets of WOD and SLOPER4D in Tab. 5, with qualitative results shown in Fig. 4 and

Table 5. SOTA comparison for LiDAR-based 3D HPE. Bench.: √ = the benchmark data is the only real-data source used in the pipeline; X = training includes extra large-scale real data. LiDAR: √ = input is only LiDAR point clouds; X = input includes images. (a) Comparison on the WOD [39] validation split.
<table><tr><td>Method</td><td>Reference</td><td>Bench.</td><td>LiDAR</td><td>MPJPE</td></tr><tr><td>DAPT</td><td>[3] aaai &#x27;25</td><td>X</td><td>J</td><td>5.16</td></tr><tr><td>WS-HPE</td><td>[48] cvprw&#x27;22</td><td>√</td><td>X</td><td>10.32</td></tr><tr><td>WS-Fusion</td><td>[4] ieee-iv ’23</td><td>√</td><td>X</td><td>8.58</td></tr><tr><td>HUM3DIL</td><td>[46] corl &#x27;22</td><td>√</td><td>X</td><td>6.72</td></tr><tr><td>GC-KPL</td><td>[40] cvpr&#x27;23</td><td>√</td><td>√</td><td>10.10</td></tr><tr><td>VoxelKP</td><td>[37] iccv ’25</td><td>√</td><td>√</td><td>8.87</td></tr><tr><td>PRN</td><td>[8] ieee-tmm&#x27;25</td><td>√</td><td>√</td><td>6.78</td></tr><tr><td>LPFormer</td><td>[45] icra’24</td><td>√</td><td>√</td><td>6.16</td></tr><tr><td>HuLiGen</td><td>ours</td><td></td><td></td><td>5.85</td></tr></table>

(b) Comparison on the SLOPER4D [7] validation split.
<table><tr><td>Method</td><td>Reference</td><td>Bench.</td><td>LiDAR</td><td>MPJPE</td></tr><tr><td>PE-Proj</td><td>[19] acm mm&#x27;24</td><td>X</td><td>X</td><td>5.55</td></tr><tr><td>DAPT</td><td>[3] aaai &#x27;25</td><td>X</td><td>√</td><td>2.80</td></tr><tr><td>LiDARCap</td><td>[18] cvpr&#x27;22</td><td>√</td><td>√</td><td>8.61</td></tr><tr><td>LIP</td><td>[34] ieee-tvcg ’24</td><td>√</td><td>X</td><td>9.48</td></tr><tr><td>LiveHPS</td><td>[36] cvpr’24</td><td>√</td><td>√</td><td>7.77</td></tr><tr><td>LiveHPS++</td><td>[33] eccv &#x27;24</td><td>√</td><td>√</td><td>7.70</td></tr><tr><td>NE-3D-HPE</td><td>[47] aaai &#x27;24</td><td>√</td><td>√</td><td>9.68</td></tr><tr><td>LiDAR-HMR</td><td>[8] ieee-tmm&#x27;25</td><td>√</td><td>√</td><td>5.10</td></tr><tr><td>PRN</td><td>[8] ieee-tmm&#x27;25</td><td>√</td><td>√</td><td>6.02</td></tr><tr><td>ReMP</td><td>[14] wacv ’25</td><td>√</td><td>√</td><td>21.66</td></tr><tr><td>HuLiGen</td><td>ours</td><td></td><td></td><td>5.04</td></tr></table>

Fig. 5. For a fair comparison, we primarily consider methods that do not rely on additional large-scale real-world training datasets. In particular, DAPT [3] additionally uses LiDARHuman26M [18], which contains 184K Li-DAR frames. PE-Proj [19] uses HmPEAR, which contains 250K frames with motion-capture-based 3D pose annotations. For SLOPER4D, some mesh-reconstruction methods do not explicitly report MPJPE. For these methods, we compute MPJPE from joints obtained from their predicted pose, shape and translation parameters to allow a consistent comparison. Our approach achieves state-of-the-art performance on both datasets.

## 4.3. Low-annotation regime

Our approach can be particularly useful when LiDAR data is abundant but only a small fraction is annotated with 3D human keypoints. We investigate this on WOD, where ground-truth meshes are unavailable and the body models are instead obtained using an off-the-shelf estimator [8]. We use the full training set to learn our FM model, generate \~48k human LiDAR samples from AMASS, and pretrain PRN on these samples. We then finetune PRN using only 1% or 10% of the labeled training data. As shown in

Table 6. Results in the low-annotation regime on the WOD validation set. Zero-shot denotes evaluation directly using models pretrained on synthetic data, without finetuning on real data. Pt. denotes pretraining.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pt.</td><td colspan="2">Zero-shot (0%)</td><td colspan="2">1 %</td><td colspan="2">10 %</td></tr><tr><td>MPJPE</td><td>PA-</td><td>MPJPE</td><td>PA-</td><td>MPJPE</td><td>PA-</td></tr><tr><td>Random init.</td><td>x</td><td>N/A</td><td>N/A</td><td>18.01</td><td>14.19</td><td>13.74</td><td>10.60</td></tr><tr><td>Rc — bootstr.</td><td>√</td><td>15.54</td><td>10.39</td><td>9.01</td><td>7.15</td><td>7.22</td><td>5.45</td></tr><tr><td>Rc — direct</td><td>√</td><td>15.11</td><td>10.04</td><td>10.89</td><td>8.37</td><td>7.98</td><td>6.16</td></tr><tr><td>GC-KPL [40]</td><td>√</td><td>17.70</td><td></td><td>18.52</td><td></td><td>15.10</td><td></td></tr><tr><td>HuLiGen</td><td>√</td><td>10.79</td><td>8.51</td><td>8.43</td><td>6.48</td><td>6.84</td><td>5.18</td></tr></table>

Table 7. Results in the low-data regime on the WOD and SLOPER4D validation sets using 10% and 20% of the real training data. Pt: pretraining; Ft: finetuning.

<table><tr><td rowspan="2">Fraction</td><td rowspan="2">Pt.</td><td rowspan="2">Ft.</td><td colspan="2">SLOPER4D</td><td rowspan="2">-Δ%</td><td colspan="2">WOD</td><td rowspan="2">-Δ%</td></tr><tr><td>MPJPE</td><td>PA-</td><td>MPJPE</td><td>PA-</td></tr><tr><td rowspan="3">10%</td><td>X</td><td>√</td><td>9.29</td><td>6.62</td><td></td><td>13.27</td><td>10.27</td><td></td></tr><tr><td>√</td><td>x</td><td>16.40</td><td>9.38</td><td></td><td>14.07</td><td>8.86</td><td></td></tr><tr><td>√</td><td>√</td><td>6.46</td><td>4.86</td><td>30.46</td><td>7.57</td><td>5.61</td><td>42.95</td></tr><tr><td rowspan="3">20%</td><td>X</td><td>√</td><td>7.91</td><td>5.60</td><td></td><td>10.53</td><td>8.28</td><td></td></tr><tr><td>√</td><td>X</td><td>13.83</td><td>8.65</td><td></td><td>13.78</td><td>8.77</td><td></td></tr><tr><td>√</td><td>√</td><td>5.87</td><td>4.53</td><td>25.79</td><td>6.69</td><td>5.01</td><td>36.47</td></tr></table>

Tab. 6, this reduces the error by more than 50% compared with training from scratch when using only 1% and 10% of the annotations. It also outperforms the raycasting baselines and GC-KPL [40] in both low-annotation settings.

## 4.4. Low-data regime

The other challenging setting where HuLiGen can be particularly effective is the low-data regime, where only a limited amount of real LiDAR data is available, rather than merely limited annotations. To simulate this scenario, we restrict the entire pipeline to only 10% and 20% of the available real data. We first train the FM model on the corresponding low-data subset and use it to generate \~48k synthetic human LiDAR samples from AMASS. We then pretrain the pose estimator on these generated samples and, in the final stage, finetune it using the same limited fraction of real data. As shown in Tab. 7, our pipeline substantially improves over training from scratch, reducing MPJPE and PA-MPJPE by approximately 30% on SLOPER4D and 45% on WOD. These results further demonstrate the effectiveness of our approach when real LiDAR data itself is scarce.

## 5. Conclusion

In this paper, we introduced HuLiGen, a point-transformerbased flow-matching model for human LiDAR generation conditioned on parametric body models. HuLiGen produces more realistic observations than standard raycasting and allows large pose repositories to be converted into synthetic LiDAR data. We show that pretraining HPE on these samples significantly improves performance, with particularly strong gains in low-data and low-annotation regimes.

OD

S4D

![](images/6d434e5f8097aab3c85bd6c0a5c390810b2767e24d41e8814353c7ddbdb8f08e.jpg)

Figure 4. Qualitative results on the WOD [39] and SLOPER4D [7] validation sets. For each sample, we show the camera crop, bodymodel surface, real LiDAR observation, raycasted result and our generation, with point counts below. Camera images are for visualization only, and all panels share the same viewpoint and scale. Ray casting stays on the bare body surface, while real and generated observations capture off-surface structure from clothing and hair.  
![](images/f80123d6aada5c7859616c010cb43edd7591cce2b284a762eb20ea296fbad55c.jpg)  
Figure 5. Qualitative results on AMASS. AMASS provides pose and shape but no sensor data, so we use the body surface as reference. For each sample, we show the body surface, raycast result and our generation, with point counts below. Ray casting remains on the bare body surface, yielding a median surface distance of 0 cm on both datasets. In contrast, our generations reach 1.41 cm on SLOPER4D (top row), close to 1.17 cm for real observations, and 4.5 cm on WOD compared with 3.0 cm for real data. We attribute the larger WOD (bottom row) gap mainly to ground-level noise.

## References

[1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3D point clouds. 2018. 5

[2] Vida Adeli, Mahsa Ehsanpour, Ian Reid, Juan Carlos Niebles, Silvio Savarese, Ehsan Adeli, and Hamid Rezatofighi. Tripod: Human trajectory and pose dynamics forecasting in the wild. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 13370–13380. IEEE, 2021. 1

[3] Xiaoqi An, Lin Zhao, Chen Gong, Jun Li, and Jian Yang. Pre-training a Density-Aware Pose Transformer for Robust LiDAR-based 3D Human Pose Estimation. In AAAI. 2, 3, 7

[4] Peter Bauer, Arij Bouazizi, Ulrich Kressel, and Fabian B. Flohr. Weakly Supervised Multi-Modal 3D Human Body Pose Estimation for Autonomous Driving. In 2023 IEEE Intelligent Vehicles Symposium (IV), 2023. 2, 7

[5] Mikolaj Binkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. 2018. 5

[6] Peishan Cong, Yiteng Xu, Yiming Ren, Juze Zhang, Lan Xu, Jingya Wang, Jingyi Yu, and Yuexin Ma. Weakly Supervised 3D Multi-Person Pose Estimation for Large-Scale Scenes Based on Monocular Camera and Single LiDAR. 2023. 2

[7] Yudi Dai, Yitai Lin, Xiping Lin, Chenglu Wen, Lan Xu, Hongwei Yi, Siqi Shen, Yuexin Ma, and Cheng Wang. SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments. In CVPR, 2023. 1, 2,3, 5,7,8

[8] Bohao Fan, Wenzhao Zheng, Jianjiang Feng, and Jie Zhou. LiDAR-HMR: 3D Human Mesh Recovery from LiDAR. IEEE Transactions on Multimedia, pages 1–14. 2, 5, 7

[9] Bohao Fan, Siqi Wang, Wenzhao Zheng, Jianjiang Feng, and Jie Zhou. Human-m3: A multi-view multi-modal dataset for 3d human pose estimation in outdoor scenes. arXiv preprint arXiv:2308.00628, 2023. 2

[10] Jin Fang, Dingfu Zhou, Feilong Yan, Tongtong Zhao, Feihu Zhang, Yu Ma, Liang Wang, and Ruigang Yang. Augmented lidar simulator for autonomous driving. IEEE Robotics and Automation Letters, 5(2):1931–1938, 2020. 3

[11] Michael Furst, Shriya T. P. Gupta, Rene Schuster, Oliver Wasenmuller, and Didier Stricker. HPERL: 3D Human Pose Estimation from RGB and LiDAR. In ICPR, 2021. 2

[12] Xiao Han, Yiming Ren, Yichen Yao, Yujing Sun, and Yuexin Ma. Towards Practical Human Motion Prediction with Li-DAR Point Clouds. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 7629–7638. 2

[13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. 2017. 5

[14] Hojun Jang and Young Min Kim. ReMP: Reusable Motion Prior for Multi-domain 3D Human Pose Estimation and Motion Inbetweening. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision, 2025. 7

[15] Jaewoo Jeong, Seohee Lee, Daehee Park, Giwon Lee, and Kuk-Jin Yoon. Multi-modal knowledge distillation-based

human trajectory forecasting. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24222–24233. IEEE, 2025. 1

[16] Ellington Kirby, Mickael Chen, Renaud Marlet, and Nermin Samet. Logen: Toward lidar object generation by point diffusion. arXiv preprint arXiv:2412.07385, 2024. 3, 5

[17] Lóránt Kovács, Balázs M. Bódis, and Csaba Benedek. Lid-Pose: Real-Time 3D Human Pose Estimation in Sparse Lidar Point Clouds with Non-Repetitive Circular Scanning Pattern. Sensors, 24(11):3427. 2

[18] Jialian Li, Jingyi Zhang, Zhiyong Wang, Siqi Shen, Chenglu Wen, Yuexin Ma, Lan Xu, Jingyi Yu, and Cheng Wang. Li-DARCap: Long-range Markerless 3D Human Motion Capture with LiDAR Point Clouds. In CVPR, 2022. 7

[19] Yitai Lin, Zhijie Wei, Wanfa Zhang, Xiping Lin, Yudi Dai, Chenglu Wen, Siqi Shen, Lan Xu, and Cheng Wang. Hm-PEAR: A Dataset for Human Pose Estimation and Action Recognition. In ACM MM, 2024. 7

[20] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 4

[21] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 1

[22] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. 2018. 5

[23] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024. 5

[24] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of Motion Capture as Surface Shapes. In ICCV, 2019. 2, 3, 5, 6

[25] Sivabalan Manivasagam, Shenlong Wang, Kelvin Wong, Wenyuan Zeng, Mikita Sazanovich, Shuhan Tan, Bin Yang, Wei-Chiu Ma, and Raquel Urtasun. Lidarsim: Realistic lidar simulation by leveraging the real world. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11164–11173. IEEE, 2020. 2

[26] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. DIT-3D: Exploring plain diffusion transformers for 3D shape generation. 2023. 4,5

[27] Kazuto Nakashima and Ryo Kurazume. Lidar data synthesis with denoising diffusion probabilistic models. 2024. 3

[28] Lukas Neumann and Andrea Vedaldi. Pedestrian and egovehicle trajectory prediction from monocular camera. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10204–10212, 2021. 1

[29] Zhiyu Pan, Zhicheng Zhong, Wenxuan Guo, Yifan Chen, Jianjiang Feng, and Jie Zhou. LiCamPose: Combining Multi-View LiDAR and RGB Cameras for Robust Singletimestamp 3D Human Pose Estimation. In WACV, 2025. 2

[30] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 4

[31] Haoxi Ran, Vitor Guizilini, and Yue Wang. Towards realistic scene generation with lidar diffusion models. 2024. 3

[32] Haziq Razali, Taylor Mordan, and Alexandre Alahi. Pedestrian intention prediction: A convolutional bottom-up multitask approach. Transportation research part C: emerging technologies, 130:103259, 2021. 1

[33] Yiming Ren, Xiao Han, Yichen Yao, Xiaoxiao Long, Yujing Sun, and Yuexin Ma. LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment. In Computer Vision - ECCV 2024 - 18th European Conference, pages 127– 144. Springer, . 7

[34] Yiming Ren, Chengfeng Zhao, Yannan He, Peishan Cong, Han Liang, Jingyi Yu, Lan Xu, and Yuexin Ma. LiDARaid Inertial Poser: Large-scale Human Motion Capture by Sparse Inertial and LiDAR Sensors. IEEE Transactions on Visualization and Computer Graphics, 29(5):2337–2347, . 7

[35] Yiming Ren, Xiao Han, Yichen Yao, Xiaoxiao Long, Yujing Sun, and Yuexin Ma. LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment. In ECCV, 2024. 2,3

[36] Yiming Ren, Xiao Han, Chengfeng Zhao, Jingya Wang, Lan Xu, Jingyi Yu, and Yuexin Ma. LiveHPS: LiDAR-Based Scene-Level Human Pose and Shape Estimation in Free Environment. In CVPR, 2024. 7

[37] Jian Shi and Peter Wonka. VoxelKP: A Voxel-based Network Architecture for Human Keypoint Estimation in Li-DAR Data. In ICCV, 2025. 2, 7

[38] Dong Wook Shu, Sung Woo Park, and Junseok Kwon. 3D point cloud generative adversarial network based on tree structured graph convolutions. 2019. 5

[39] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In CVPR, 2020. 1, 2, 5, 7, 8

[40] Zhenzhen Weng, Alexander S. Gorban, Jingwei Ji, Mahyar Najibi, Yin Zhou, and Dragomir Anguelov. 3D Human Keypoints Estimation from Point Clouds in the Wild without Human Labels. In CVPR, 2023. 2, 3, 7

[41] Yuwen Xiong, Wei-Chiu Ma, Jingkang Wang, and Raquel Urtasun. UltraLiDAR: Learning compact representations for lidar completion and generation. 2023. 3

[42] Yiteng Xu, Kecheng Ye, Xiao Han, Yiming Ren, Xinge Zhu, and Yuexin Ma. A Unified Framework for Human-centric Point Cloud Video Understanding. In CVPR, 2024. 2, 3

[43] Tianyi Yan, Junbo Yin, Xianpeng Lang, Ruigang Yang, Cheng-Zhong Xu, and Jianbing Shen. Olidm: Object-aware lidar diffusion models for autonomous driving. 2025. 3

[44] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. PointFlow: 3D point cloud generation with continuous normalizing flows. 2019. 5

[45] Dongqiangzi Ye, Yufei Xie, Weijia Chen, Zixiang Zhou, Lingting Ge, and Hassan Foroosh. LPFormer: LiDAR Pose Estimation Transformer with Multi-Task Network. In ICRA, 2024.2, 7

[46] Andrei Zanfir, Mihai Zanfir, Alexander Gorban, Jingwei Ji, Yin Zhou, Dragomir Anguelov, and Cristian Sminchisescu. HUM3DIL: Semi-supervised Multi-modal 3D Human Pose Estimation for Autonomous Driving. In CoRL, 2022. 2, 7

[47] Jingyi Zhang, Qihong Mao, Guosheng Hu, Siqi Shen, and Cheng Wang. Neighborhood-Enhanced 3D Human Pose Estimation with Monocular LiDAR in Long-Range Outdoor Scenes. 2024. 7

[48] Jingxiao Zheng, Xinwei Shi, Alexander Gorban, Junhua Mao, Yang Song, Charles R. Qi, Ting Liu, Visesh Chari, Andre Cornman, Yin Zhou, Congcong Li, and Dragomir Anguelov. Multi-modal 3D Human Pose Estimation with 2D Weak Supervision in Autonomous Driving. In CVPRW, 2022. 2, 7

[49] Vlas Zyrianov, Xiyue Zhu, and Shenlong Wang. Learning to generate realistic lidar point clouds. In European Conference on Computer Vision, pages 17–35. Springer, 2022. 2, 3

# HuLiGen: Human LiDAR Generation from Parametric Body Models

# Supplementary Material

This appendix supplements the main paper with additional details on our method and ablation experiments.

## 6. Complementary details on conditioning and embedding

Here, we provide additional details on the conditioning formulations explored in our ablation study. In particular, we describe how our Flow Matching framework is instantiated with the SMPL [21] body model and explain the different conditioning and embedding strategies considered.

Data format. We use the point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ , transformed into the sensor frame, together with the corresponding SMPL annotations: shape $\beta \in \mathbb { R } ^ { 1 0 }$ , full local body pose $\theta \in \mathbb { R } ^ { 7 2 }$ represented by 24 axis-angle triplets, and translation $t _ { l }$ . The global orientation $\theta _ { 0 }$ and translation $t _ { l }$ are also expressed in the sensor frame. All ablation experiments in Table 1 of the main paper use the same P, $t _ { l } ,$ and $\theta ;$ they differ only in whether the shape parameter $\beta$ is included and in how the pose $\theta$ is provided to the network as well as canonicalisation. We provide an overview of the ablation configurations in Table 8, detailing the representations, conditioning choices and dimentionality used in each setting.

Canonicalisation. For canonicalisation, our model uses the lateral axis of the SMPL root joint. We first convert the root axis-angle representation $\pmb { \theta } _ { 0 }$ into a rotation matrix,

$$
\mathbf { R } _ { 0 } = \mathcal { R } ( \pmb { \theta } _ { 0 } ) \in S O ( 3 )\tag{10}
$$

expressed in the sensor frame. The columns of $\mathbf { R } _ { 0 }$ correspond to the transformed SMPL canonical body axes: $\mathbf { R } _ { 0 } \hat { e } _ { 1 }$ denotes the lateral axis, $\mathbf { R } _ { 0 } \hat { e } _ { 2 }$ the upward axis, and ${ \bf R } _ { 0 } \hat { e } _ { 3 }$ the forward axis. We project the lateral axis onto the horizontal plane and compute its azimuth ψ, which defines the canonicalising rotation:

$$
\psi = \mathrm { a t a n 2 } \big ( [ \mathbf { R } _ { 0 } ] _ { 1 0 } , [ \mathbf { R } _ { 0 } ] _ { 0 0 } \big ) ,\tag{11}
$$

and $\mathbf { R } _ { z } ( - \psi )$ is applied consistently to all geometric quantities.

Using the projected lateral axis rather than the forward axis avoids degenerate cases in which the forward direction approaches the vertical axis, making its projection onto the horizontal plane, and consequently its azimuth, ill-defined. Examples from SLOPER4D [7] are shown in Fig. 2, where crouched poses illustrate this issue and motivate our choice. We present our canonicalization algorithm in Algorithm 1.

For the ablations without canonicalisation (corresponding to rows 1 and 3 of Table 1 in the main paper), we instead express the translation $t _ { l }$ in cylindrical coordinates.

Algorithm 1 Canonicalisation   
Require: points P, pose $\theta ,$ shape $\beta ,$ translation $t _ { l }$   
1: $t _ { l } \gets$ body centre; $( \phi , \rho , z ) \gets \mathrm { c y l } ( t )$   
2: $\mathbf { R } _ { 0 } \gets \mathcal { R } ( \theta _ { 0 } )$   
3: $\psi  \mathrm { a t a n 2 } ( [ \mathbf { R } _ { 0 } ] _ { 1 0 } , [ \mathbf { R } _ { 0 } ] _ { 0 0 } )$   
4: $\mathbf { P }  \mathbf { R } _ { z } ( - \psi ) \cdot ( \mathbf { P } - t _ { l } )$   
5: $\pmb { \theta } _ { 0 }  \mathrm { r o t v e c } \big ( \mathbf { R } _ { z } ( - \psi ) \cdot \mathbf { R } _ { 0 } \big )$   
6: $\phi _ { \mathrm { r e l } }  \mathrm { w r a p } ( \phi - \psi )$   
7: return $\mathbf { P } , \theta , \beta , \mathbf { o } = ( \phi _ { \mathrm { r e l } } , \rho , z )$

Its azimuth $\phi$ is encoded as (sin φ, cos φ) and provided to the network together with logρ and z. o thus becomes (sin φ, cos φ, log $\rho , z )$

Local body pose representations. Our model applies the SMPL forward kinematics (FK) function to the input axisangle poses θ to obtain the corresponding 3D joint positions, which are then embedded using the joint-position encoder described in Sec. 3.3 of the main paper. As part of our ablation study, we also consider an alternative pose representation in which the axis-angle joint rotations are converted to 6D rotations with cyclic embedding, together with a learned positional encoding analogous to that used in the FK-based formulation.

Table 8. Conditioning and embedding details. Canonical align.: ${ \surd } = \mathrm { c a n o n i c a l }$ axis transformation applied; $\scriptstyle x = \mathrm { n o }$ canonical transformation. Shape param.: $\checkmark = \mathrm { s h a p e }$ parameters included in the conditioning; X = shape parameters excluded from the conditioning. Cartesian joints: $\overline { { { \bf { \mathcal { V } } } = { \bf { \bar { \Phi } } } } }$ Cartesian joint coordinates with PointNet embeddings; $\boldsymbol { x } = 6 \mathrm { D }$ rotation coordinates with cyclic embeddings. abs cyl. denotes absolute cylindrical coordinates and canon. denotes canonicalized coordinates. The third column group reports the dimensionality of each conditioning representation before embedding; while the final column gives the total conditioning-token count used by each variant.
<table><tr><td></td><td>Canonical align. Shape param.</td><td>Cartesian joints</td><td> $\mathbf { o } _ { r e p }$ </td><td> $\theta _ { r e p }$ </td><td> $o _ { e m b }$ </td><td> $\beta _ { e m b }$ </td><td> $\theta _ { e m b }$ </td><td>token count</td></tr><tr><td>X</td><td>X</td><td>X</td><td>abs. cyl.</td><td> $2 4 \times 6 \mathrm { D }$  rotation</td><td>4</td><td>N/A</td><td>144</td><td>28</td></tr><tr><td>√</td><td>X</td><td>X</td><td>canon.</td><td> $2 4 \times 6 \mathrm { D }$  rotation</td><td>3</td><td>N/A</td><td>144</td><td>27</td></tr><tr><td>X</td><td>√</td><td>X</td><td>abs. cyl.</td><td> $2 4 \times 6 \mathrm { D }$  rotation</td><td>4</td><td>10</td><td>144</td><td>38</td></tr><tr><td>√</td><td>√</td><td>X</td><td>canon.</td><td> $2 4 \times 6 \mathrm { D }$  rotation</td><td>3</td><td>10</td><td>144</td><td>37</td></tr><tr><td>V</td><td>√</td><td>√</td><td>canon.</td><td> $2 4 \times 3 \mathrm { D }$  position</td><td>3</td><td>10</td><td>72</td><td>37</td></tr></table>