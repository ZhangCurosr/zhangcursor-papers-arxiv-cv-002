# PanoGS-SLAM: Panoramic 3D Gaussian Splatting SLAM

Yongqi Mao<sup>1</sup>, Hao Shi<sup>3,1</sup>, Yufan Zhang<sup>2</sup>, Zhonghua Yi<sup>1</sup>, Xiangfei Guo<sup>1</sup>, Kaiwei Wang<sup>1∗</sup> <sup>1</sup>Zhejiang University, <sup>2</sup>National University of Defense Technology, <sup>3</sup>Ant Group

Abstract— Real-time dense SLAM is a core capability for robotics applications that require robust localization and highquality mapping in dynamic or fast-changing environments. Recent 3D Gaussian Splatting (3DGS)-based SLAM methods have shown promising performance, but most are designed for narrow-FoV pinhole cameras, where limited angular coverage weakens pose observability and often leads to unstable photometric optimization under rapid motion and large viewpoint changes. We present PanoGS-SLAM, the first panoramic dense SLAM system built on 3D Gaussian Splatting. Our method performs differentiable rendering and pose optimization directly in the spherical domain, enabling omnidirectional photometric constraints for more stable tracking. To improve geometric consistency and robustness, we introduce (1) a sphere-consistent photometric loss that compensates for the area distortion of equirectangular projection, and (2) a depth-guided Gaussian initialization strategy that stabilizes incremental mapping in newly observed regions. Extensive experiments on both real and synthetic panoramic benchmarks (PALVIO and SynPano) show that PanoGS-SLAM consistently outperforms geometric and GS-based baselines in tracking accuracy and rendering quality, while achieving fast front-end convergence and real-time performance. In addition, controlled field-of-view experiments reveal a clear monotonic improvement in optimization conditioning and convergence stability as angular coverage increases, highlighting the fundamental role of sensing geometry in shaping the optimization landscape of differentiable Gaussian-based SLAM. The source code will be made publicly available.

## I. INTRODUCTION

Real-time dense SLAM is a fundamental capability for robotic systems that require reliable localization, scene understanding, and high-fidelity mapping, such as autonomous navigation, 3D reconstruction, and immersive perception. Classical dense SLAM pipelines typically rely on explicit scene representations (e.g., depth maps[1], voxels[2][3], or surfels[4][5]), which often face an unfavorable trade-off between scalability, efficiency, and rendering fidelity. Recent advances in differentiable scene representations, especially 3D Gaussian Splatting (3DGS)[6], provide a compelling alternative by combining explicit geometric primitives with efficient differentiable rendering. Building on 3DGS, several recent methods have extended Gaussian representations to dense visual SLAM[7][8][9][10][11][12]. By jointly optimizing camera poses and Gaussian primitives through photometric rendering losses, these systems achieve a favorable balance between real-time performance and reconstruction quality. This line of work has established Gaussian-based differentiable SLAM as a promising paradigm for highquality online mapping.

![](images/28045835dec0f6fa378c86b693651d6816f833929f260d4c9956aa550bab2477.jpg)  
Fig. 1. PanoGS-SLAM takes monocular panoramic input and performs differentiable rendering and pose optimization directly in the spherical domain. The large FoV provides strong constraints, enabling fast and stable pose convergence with millimeter-level trajectory accuracy.

However, existing GS-based SLAM systems are predominantly developed under the narrow-FoV pinhole camera assumption. This design choice introduces a fundamental limitation for differentiable photometric pose estimation: the optimization landscape is strongly influenced by the spatial and angular distribution of image gradients. Under limited FoV, gradients are concentrated within a narrow viewing cone, leading to an anisotropic information distribution, weakened rotational observability, and stronger coupling between translational and rotational degrees of freedom. As a consequence, the underlying optimization problem becomes poorly conditioned, exhibiting a reduced convergence basin and increased sensitivity to rapid motion or abrupt viewpoint changes. A second issue arises during incremental mapping. Under narrow-FoV sensing, newly observed regions are introduced frequently, but these regions initially lack sufficient Gaussian support and geometric constraints. This weakly constrained state can destabilize front-end pose optimization and slow down convergence, especially when tracking depends on a lagging global map. In practice, this behavior is evident in recent monocular GS-based systems such as MonoGS[8], where tracking quality can degrade significantly when the camera enters large unseen areas.

In this work, we revisit Gaussian-based dense SLAM from the perspective of sensing geometry. We hypothesize that wide-field perception fundamentally reshapes the optimization landscape of differentiable SLAM by redistributing photometric constraints across the full viewing sphere. To validate this hypothesis, we present PanoGS-SLAM, the first panoramic SLAM system built upon 3D Gaussian Splatting. Instead of decomposing panoramas into perspective crops, our framework performs differentiable rendering and joint optimization directly in the spherical domain, maintaining continuous gradients over the entire unit sphere. This design substantially improves numerical conditioning and enlarges the convergence basin of pose optimization. To ensure geometric consistency under equirectangular projection, we introduce a sphere-consistent photometric objective that compensates for non-uniform pixel area distortion, preventing polar regions from dominating the loss function. We further propose a depth-guided Gaussian initialization and insertion strategy to enhance geometric priors in newly observed regions and stabilize incremental mapping. Together, these components form a unified panoramic Gaussian SLAM framework for robust tracking and high-quality reconstruction from omnidirectional RGB input.

Extensive experiments on both synthetic and real panoramic benchmarks (SynPano[13] and PALVIO[14]) demonstrate the effectiveness of our approach. PanoGS-SLAM consistently achieves the best tracking accuracy and superior rendering quality over both geometric panoramic SLAM baselines and recent GS-based dense SLAM methods. On several sequences, it outperforms existing baselines by up to an order of magnitude. Moreover, our system converges much faster in front-end pose optimization, requiring only 15 iterations to achieve stable tracking—approximately one-seventh of the iterations required by MonoGS[8]. This leads to substantially improved runtime efficiency. Beyond benchmark comparisons, we conduct a controlled field-ofview study and show a clear monotonic trend: increasing angular coverage markedly improves convergence behavior and numerical stability in differentiable pose optimization. These results reveal that the field of view is not only a sensing choice but also a key factor governing the optimization properties of GS-based dense SLAM.

Our contributions are summarized as follows:

• Panoramic 3D Gaussian SLAM Framework. We present the first 3DGS-based panoramic SLAM system that directly performs joint camera tracking and Gaussian map optimization in the spherical domain.

• Sphere-Consistent Photometric Optimization. A geometrically consistent panoramic loss compensating for equirectangular area distortion to improve gradient balance and tracking stability.

• Depth-Guided Gaussian Initialization For Incremental Mapping. A depth-guided Gaussian insertion strategy that enhances geometric consistency in newly observed regions.

• Field-of-View Analysis For Differentiable SLAM. Through controlled FoV experiments, we empirically show that wider angular coverage substantially improves conditioning, convergence basin, and robustness in Gaussian-based dense SLAM.

## II. RELATED WORK

## A. Dense SLAM

Traditional sparse SLAM systems achieve remarkable performance in camera localization and mapping, as demonstrated by ORB-SLAM[15][16][17] and VINS-Mono[18]. However, the resulting maps are typically sparse and mainly optimized for pose estimation. In contrast, dense SLAM aims to reconstruct richer scene representations, providing enhanced perception for augmented reality and robotic applications. Dense SLAM systems fuse per-pixel observations to recover continuous surfaces, commonly using voxelbased representations[2][3] or point-based models[5][4]. Although effective, these explicit representations often incur high memory consumption and limited rendering quality. NeRF[19] introduced neural implicit representations for modeling scene geometry and appearance. Building on this idea, iMAP[20] incorporated implicit neural maps into SLAM, while NICE-SLAM[21] improved scalability through a hierarchical multi-resolution design. Subsequent works combined geometric representations with neural fields to enhance rendering fidelity[22][23], but these approaches generally require time-consuming optimization, limiting realtime applicability. More recently, 3DGS[6] proposed an explicit and differentiable representation that significantly reduces optimization time while maintaining high rendering quality. Methods such as SplaTAM[9], GS-SLAM[7], MonoGS[10], and Photo-SLAM[8] extended 3DGS to dense SLAM by jointly optimizing poses and Gaussian parameters, improving robustness and photometric consistency. GS-ICP[11] efficiently registers Gaussian spheres with an ICPlike optimization, and VINGS-SLAM[12] integrates visualinertial odometry with GS–based mapping. Nevertheless, these methods are largely based on the pinhole camera model, whose limited field of view may compromise tracking stability under rapid viewpoint changes.

## B. Omnidirectional SLAM and 3DGS

Omnidirectional images provide richer geometric constraints due to their wide field of view, which generally improves tracking robustness under rapid motion and abrupt scene transitions. OpenVSLAM[24] extends ORB-SLAM2[16] to support panoramic cameras. Building upon ORB-SLAM3[17], P2U-SLAM[25] further proposes a unified omnidirectional framework that explicitly models point and pose uncertainties. LF-VISLAM[26], derived from VINS-Mono[18], introduces a tightly coupled visual-inertial framework with loop closure capability that accommodates omnidirectional inputs. 360VO[27] extends the classical Direct Sparse Visual Odometry framework to the omnidirectional camera setting. 360-GS[28] extends 3DGS to omnidirectional scene modeling and demonstrates global reconstruction for panoramic images. Subsequent works such as ODGS[29] and OmniGS[30] introduce dedicated CUDA rasterizers that directly operate in spherical or equirectangular projection spaces, avoiding intermediate perspective decomposition and improving rendering efficiency. However, these methods focus on offline reconstruction and do not address camera pose estimation or global consistency in an online SLAM setting. Consequently, the integration of omnidirectional 3DGS into a globally consistent SLAM framework persists as a significant, unresolved challenge. In light of this, we present a novel, globally consistent omnidirectional 3DGS-based SLAM system that concurrently optimizes both camera poses and Gaussian parameters within a cohesive, unified framework.

## III. METHOD

The overall architecture of PanoGS-SLAM is illustrated in Fig. 2. Our system takes monocular panoramic sequences as input and maintains a unified 3D Gaussian representation for both tracking and mapping. The pipeline consists of a preprocessing module for initial depth estimation, a tracking front-end for pose optimization, and a mapping back-end for incremental scene reconstruction. In the following, we describe the core components of our framework.

## A. Gaussian Splatting

We adopt 3DGS as the scene representation, denoted as $\{ G _ { i } \} _ { i = 1 } ^ { N }$ . Each Gaussian is parameterized by its mean position $\mu _ { i } \ \in \ \mathbb { R } ^ { 3 }$ in the world coordinate frame, opacity $\alpha _ { i } \in [ 0 , 1 ]$ , and covariance matrix $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$

$$
\begin{array} { r } { G _ { i } ( \mathbf { x } ) = \exp \left( - \frac { 1 } { 2 } \left( \mathbf { x } - \pmb { \mu } _ { i } \right) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } _ { i } ) \right) . } \end{array}\tag{1}
$$

In addition, the appearance of each Gaussian is modeled using spherical harmonics (SH) to represent view-dependent color. Before splatting, each 3D Gaussian is projected onto the 2D image plane, yielding a 2D covariance matrix:

$$
\begin{array} { r } { \Sigma _ { \mathrm { 2 D } } = \mathbf { J } \mathbf { W } \Sigma \mathbf { W } ^ { \top } \mathbf { J } ^ { \top } . } \end{array}\tag{2}
$$

where J denotes the Jacobian of the projection function and W is the camera pose transformation matrix. 3DGS performs volumetric rendering without explicit geometric surfaces. The final color at each pixel is synthesized by compositing N Gaussians along the viewing direction:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{3}
$$

where $c _ { i }$ is the color of the i-th Gaussian. Unlike raybased volume rendering, 3DGS adopts a rasterization-based pipeline that traverses 2D Gaussians on the image plane for each pixel. This design fully exploits the sparsity of the 3D scene representation and enables highly efficient rendering.

## B. Camera Model

Standard 3DGS is primarily designed for perspective projection. To adapt it for omnidirectional SLAM, we replace the pinhole model with a spherical projection that maps 3D Gaussians onto an equirectangular projection (ERP) plane, following the prior panoramic Gaussian Splatting method[29]. Let $\bar { \boldsymbol { \mu } } = ( \bar { \mu } _ { x } , \bar { \mu } _ { y } , \bar { \mu } _ { z } ) ^ { \top }$ denote the mean position of a Gaussian in the camera coordinate frame. Its longitude $\phi _ { \mu }$ and latitude $\theta _ { \mu }$ in spherical coordinates are defined as:

$$
\phi _ { \mu } = \arctan \left( \frac { \mu _ { x } } { \mu _ { z } } \right) ,\tag{4}
$$

$$
\theta _ { \mu } = \arctan \left( \frac { - \mu _ { y } } { \sqrt { \mu _ { x } ^ { 2 } + \mu _ { z } ^ { 2 } } } \right) .\tag{5}
$$

The spherical coordinates are then mapped to pixel coordinates $( u , v )$ via the cylindrical projection function $\pi _ { o } ( \cdot ) \colon$

$$
\pi _ { o } \left( \pmb { \mu } \right) = \left( \frac { W } { 2 \pi } \phi _ { \mu } + \frac { W } { 2 } , \ - \frac { H } { \pi } \theta _ { \mu } + \frac { H } { 2 } \right) ^ { \top } .\tag{6}
$$

Here, W and H are the width and height of the panoramic image. To propagate covariance information during rasterization, we compute the Jacobian matrix J of the projection $\pi _ { o }$ by differentiating it with respect to the 3D Gaussian position. This Jacobian captures the local geometric deformation induced by the mapping from Euclidean space to the spherical domain, ensuring that the 3D Gaussians are properly stretched near the poles of the panoramic image:

$$
a = \frac { W } { 2 \pi \| \pmb { \mu } \| } , b = \frac { H } { \pi \| \pmb { \mu } \| } ,\tag{7}
$$

$$
{ \bf J } = \left( \begin{array} { c c c } { { a \sec { \theta _ { \mu } } \cos { \phi _ { \mu } } } } & { { 0 } } & { { - a \sec { \theta _ { \mu } } \sin { \phi _ { \mu } } } } \\ { { b \sin { \theta _ { \mu } } \sin { \phi _ { \mu } } } } & { { b \cos { \theta _ { \mu } } } } & { { b \sin { \theta _ { \mu } } \cos { \phi _ { \mu } } } } \end{array} \right)\tag{8}
$$

The final 2D covariance is computed as $\Sigma _ { \mathrm { 2 D } }$ in (2), completing the transformation from 3D Gaussians to 2D Gaussians on the ERP plane. The resulting 2D Gaussians are then rasterized and blended to synthesize pixel colors, following the standard Gaussian splatting procedure.

## C. SLAM Front-end

In the front-end, we optimize only the camera pose while keeping the Gaussian map fixed. Unlike narrow-FoV pinhole cameras, which often exhibit discontinuous pose gradients near image boundaries, the panoramic camera provides omnidirectional coverage. This ensures smooth gradients across the entire view, resulting in a more stable and efficient pose optimization. To further enhance tracking stability, we design a sphere-aware Panoramic Loss tailored to panoramic images and incorporate a keyframe-based management strategy to maintain map consistency.

Panoramic Loss. Equirectangular images introduce area distortion: pixels near the poles cover smaller solid angles but are over-represented in the image domain. Standard pho tometric losses therefore overweight polar regions, making optimization sensitive to noise and distortion. To address this, we propose a Panoramic Loss that weights each pixel by the cosine of its spherical latitude. This weighting explicitly accounts for the true area each pixel represents on the unit sphere, ensuring that the loss is consistent with spherical geometry rather than the distorted image domain:

$$
\mathcal { L } _ { p a n o } = \sum _ { \mathbf { u } } \cos \left( \theta ( \mathbf { u } ) \right) \left. I _ { \mathrm { r e n d e r } } ( \mathbf { u } ) - I _ { \mathrm { g t } } ( \mathbf { u } ) \right. .\tag{9}
$$

![](images/4f3e45a94e70d5687489b0393c7a46e84e30a12b56fe860cb90d5734b077e957.jpg)  
Fig. 2. PanoGS-SLAM Pipeline. The system consists of preprocessing, tracking, and mapping modules, operating on panoramic inputs with 3DGS as the sole representation. The initial map is constructed from the first frame using a pre-trained panoramic depth estimator[31]. In tracking, camera poses are estimated by minimizing the panoramic loss $L _ { p a n o }$ on the unit sphere, initialized using the previous-frame pose. Keyframes trigger opacity-guided Gaussian insertion. In mapping, newly inserted Gaussians are generated via equal-area unit-sphere sampling and depth initialization. The back-end jointly optimizes keyframe poses, randomly sampled non-keyframes, and Gaussian parameters within the keyframe window, while GS management performs cloning, splitting, and pruning to maintain map quality.

where $\theta ( \mathbf { u } )$ denotes the latitude angle of pixel u on the unit sphere. By enforcing area-consistent error accumulation, this sphere-aware loss significantly improves tracking stability.

Keyframe Management. While the front-end processes all incoming frames to maintain trajectory continuity, the back-end cannot optimize over every frame due to computational constraints. Therefore, the front-end is responsible for selecting keyframes and only forwards these to the backend for optimization. Following the keyframe management strategy of MonoGS[8], we select keyframes based on Gaussian point co-visibility and camera pose displacement. After keyframe selection, the rendered opacity map is sent to the backend to guide point cloud sampling.

## D. SLAM Back-end

In the back-end mapping stage, we jointly optimize camera poses and Gaussian parameters by minimizing the photometric error. The optimization spans all frames in the keyframe window, along with two randomly sampled non-keyframe views, to alleviate global forgetting and enforce cross-view consistency. We reuse the panoramic loss $L _ { p a n o }$ for backend supervision. Furthermore, we propose a Depth-Guided Gaussian Initialization Strategy (DGIS), which consists of map initialization and Gaussian insertion, to provide better geometric priors and improve structural accuracy.

Map Initialization. Upon receiving the first frame in the back-end, we employ a pre-trained panoramic depth estimation network[31] to predict an initial depth map, which enables the construction of a reliable Gaussian map. To initialize the map, Gaussian primitives are generated by downsampling the RGB-D observations. Specifically, we adopt an equal-area uniform sampling strategy on the unit sphere. This design alleviates Gaussian redundancy near the poles caused by equirectangular projection and ensures a more balanced spatial distribution across the panoramic field of view.

Gaussian Insertion. For each incoming keyframe, additional Gaussian primitives are inserted to progressively model newly observed regions. To reduce redundancy, the insertion density is adaptively controlled based on the opacity map, built upon the unit-sphere sampling scheme. Regions with opacity values below a predefined threshold are considered under-reconstructed and are therefore assigned a higher insertion density. The depth of newly inserted Gaussians is initialized from the rendered depth map of the current frame. If no valid depth is available at the corresponding pixel, the depth is assigned from its nearest valid neighbor, followed by a small random perturbation. This strategy preserves geometric consistency with the current reconstruction while introducing sufficient diversity to enhance optimization robustness and reduce the risk of convergence to poor local minima.

## IV. EVALUATION

## A. Experimental Setting

Baselines. We benchmark our method against classical geometric systems, including ORB-SLAM3[17] and VINS-Mono[18], and their wide-FoV extensions P2U-SLAM[25] and LF-VISLAM[26]. We also compare with Gaussian-based SLAM frameworks, specifically MonoGS[8] and Photo-SLAM[10]. Our method builds upon the Gaussian representation as applied in MonoGS, while extending it to panoramic modeling and improving tracking robustness. Therefore, MonoGS serves as the primary baseline for evaluating the effectiveness of our proposed extensions.

Datasets. We evaluate our method on the SynPano[13] and PALVIO datasets[14]. The SynPano dataset is synthetically generated using Blender and consists of indoor room scenes in the form of $3 6 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ equirectangular panoramas. It provides a controlled environment to systematically investigate the impact of sensing geometry on optimization stability. The PALVIO dataset features $3 6 0 ^ { \circ } \times [ 4 0 ^ { \circ } , 1 2 0 ^ { \circ } ]$ panoramic annular lens (PAL) sequences. It is captured by a real aerial platform and poses significant challenges for tracking due to aggressive 6-DOF maneuvers and rapid viewpoint changes.

![](images/9e054177dea27dc6e68751380121fcb36dca991b47bddf9385308c7c9c9989e2.jpg)

TABLE I  
TRAJECTORY ACCURACY COMPARISON ON PALVIO DATASET[14] (ATE RMSE [M]).
<table><tr><td>Category</td><td>Method</td><td>ID01</td><td>ID02</td><td>ID03</td><td>ID04</td><td>ID05</td><td>ID06</td><td>ID07</td><td>ID08</td><td>ID09</td><td>ID10</td></tr><tr><td rowspan="4">Sparse Slam</td><td>VINS-Mono*[18]</td><td>1</td><td>0.368</td><td>0.338</td><td>0.331</td><td>0.287</td><td>1</td><td>0.519</td><td>1</td><td>1</td><td>1</td></tr><tr><td>ORB-SLAM3*[17]</td><td>0.554</td><td>0.102</td><td>0.114</td><td>0.089</td><td>0.095</td><td>0.094</td><td>0.378</td><td>1.956</td><td>2.055</td><td>1.868</td></tr><tr><td>LF-VISLAM*[26]</td><td>1</td><td>0.148</td><td>0.180</td><td>0.198</td><td>0.178</td><td>0.203</td><td>0.230</td><td>0.199</td><td>0.219</td><td>1</td></tr><tr><td>P2U-SLAM[25]</td><td>0.108</td><td>0.073</td><td>0.081</td><td>0.086</td><td>0.082</td><td>0.091</td><td>0.110</td><td>0.109</td><td>0.102</td><td>0.109</td></tr><tr><td rowspan="3">GS-based Slam</td><td>Photo-SLAM[10]</td><td>1</td><td>0.068</td><td>0.094</td><td>1</td><td>0.129</td><td>0.087</td><td>1.468</td><td>0.854</td><td>0.330</td><td>0.641</td></tr><tr><td>MonoGS[8]</td><td>2.664</td><td>1.490</td><td>2.031</td><td>1.417</td><td>2.044</td><td>1.501</td><td>2.376</td><td>1.096</td><td>2.070</td><td>2.225</td></tr><tr><td>Ours</td><td>0.078</td><td>0.055</td><td>0.071</td><td>0.074</td><td>0.068</td><td>0.076</td><td>0.091</td><td>0.091</td><td>0.102</td><td>0.102</td></tr></table>

Best results are in bold, second-best in underlined. “/” indicates failure. <sup>∗</sup> indicates methods with IMU.

Fig. 3. Trajectory Comparison Against Ground Truth. Left: room3 from the SynPano dataset. Right: ID02 from the PALVIO dataset. Our method achieves substantially better alignment with the ground truth than all baselines. In contrast, MonoGS (our primary baseline) suffers from unstable front-end tracking and inconsistent trajectory scale over time.  
TABLE II  
TRAJECTORY ACCURACY COMPARISON ON SYNPANO DATASET[13] (ATE RMSE [M]).
<table><tr><td>Method</td><td>room1</td><td>room2</td><td>room3</td><td>room4</td><td>room5</td></tr><tr><td>ORB-SLAM3[17]</td><td>0.015</td><td>1.827</td><td>0.023</td><td>1</td><td>1.366</td></tr><tr><td>P2U-SLAM[25]</td><td>0.020</td><td>0.135</td><td>0.066</td><td>0.011</td><td>0.020</td></tr><tr><td>Photo-SLAM[10]</td><td>0.015</td><td>2.164</td><td>0.049</td><td>1</td><td>1</td></tr><tr><td>MonoGS[8]</td><td>0.237</td><td>1.190</td><td>0.354</td><td>0.317</td><td>1.058</td></tr><tr><td>Ours</td><td>0.005</td><td>0.018</td><td>0.003</td><td>0.003</td><td>0.010</td></tr></table>

VINS-Mono and LF-VISLAM require IMU data, which SynPano does not provide, so their results are unavailable.

Data Preprocessing. To accommodate the diverse camera models of the baselines, we implement a flexible preprocessing interface tailored to each dataset’s native format. For the SynPano dataset (native equirectangular), we project the panoramas into virtual pinhole views for pinhole-based systems and into Panoramic Annular Lens (PAL) images for those using the Scaramuzza (Taylor) model. Conversely, for the PALVIO dataset (native PAL), we map the raw frames into equirectangular images for our PanoGS-SLAM and into pinhole views for the other baselines. Crucially, all virtual pinhole views $( 1 2 0 ^ { \circ } \times 6 0 ^ { \circ }$ FoV) are extracted from the equatorial region to minimize resampling artifacts, ensuring a rigorous and fair evaluation across different sensing geometries. ORB-SLAM3, MonoGS, and Photo-SLAM use pinhole views as input, while other baselines use panoramic inputs.

Metrics. We evaluate tracking accuracy using the root mean squared error (RMSE) of the absolute trajectory error (ATE). To account for scale ambiguity, the estimated trajectory is aligned with the ground truth before evaluation. For reconstruction quality, we report photometric metrics including PSNR, SSIM, and LPIPS to evaluate the absolute rendering quality and photometric consistency of the reconstructed Gaussian maps.

Computational Platform. All experiments are conducted on a cloud server equipped with a single NVIDIA GeForce RTX 4090 GPU and 16 vCPUs based on an Intel(R) Xeon(R) Gold 6430 processor.

## B. Quantitative Evaluation

Tracking Accuracy. As summarized in Tables I and II, PanoGS-SLAM consistently achieves the lowest ATE RMSE across all evaluated sequences. While traditional geometric baselines such as LF-VISLAM and P2U-SLAM leverage wide-FoV inputs, their reliance on discrete feature matching can lead to instability under aggressive 6-DOF maneuvers. In contrast, PanoGS-SLAM maintains stable tracking by leveraging continuous photometric gradients from our differentiable rendering pipeline, which provides more resilient pose constraints during rapid viewpoint changes on the PALVIO benchmark (Tables I). Compared to Gaussian-based dense SLAM systems, our method demonstrates superior trajectory alignment (Fig. 3) by resolving the weakened rotational observability and poorly conditioned optimization inherent to narrow-FoV sensing. This advantage is most pronounced in SynPano’s room4 and room5 sequences (Tables II), where $3 6 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ coverage ensures reliable convergence in textureless regions where pinhole-based systems typically diverge.

![](images/bb4db8df3c2be0f6bdfeeb917e681e5edae2a39e8e4e71543b8afb22629e8244.jpg)  
Fig. 4. Comparison of Rendering Quality in Training and Novel Views on SynPano. MonoGS shows misaligned reconstructions due to trajector errors, while Photo-SLAM exhibits artifacts and overfitting in novel views. The black regions in the MonoGS renderings do not correspond to unknown areas, as evidenced by the Photo-SLAM renderings.

GT  
![](images/5e9de3dae8fded2846740ada837acbfa81ae30442f51e8e10ff6aad45e056f36.jpg)  
Fig. 5. Panoramic Rendering Examples. Our method produces highquality panoramic renderings.

Rendering Quality. Table III reports rendering quality compared with GS-based SLAM systems, where our method consistently outperforms all baselines. Fig. 4 shows visual comparisons where novel views are rendered from random poses outside the sequence. In these cases, MonoGS suffers from map distortions due to trajectory drift, while Photo-SLAM produces artifacts due to elongated Gaussians and overfitting. Notably, despite being trained with a panoramic model, our method still outperforms the baselines even on individual pinhole views corresponding to their input observations. Fig. 5 presents panoramic renderings, showing geometrically consistent and visually coherent reconstructions under full panoramic projection, though the larger FoV naturally yields lower PSNR than pinhole renderings.

TABLE III  
AVERAGE RENDERING QUALITY COMPARISON.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR(dB) ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">PALVIO[14]</td><td>Photo-SLAM[10]</td><td>20.10</td><td>0.83</td><td>0.40</td></tr><tr><td>MonoGS[8]</td><td>18.17</td><td>0.79</td><td>0.47</td></tr><tr><td>Ours</td><td>23.48</td><td>0.88</td><td>0.30</td></tr><tr><td rowspan="3">SynPano[13]</td><td>Photo-SLAM[10]</td><td>24.46</td><td>0.82</td><td>0.30</td></tr><tr><td>MonoGS[8]</td><td>21.47</td><td>0.81</td><td>0.31</td></tr><tr><td>Ours</td><td>30.58</td><td>0.91</td><td>0.22</td></tr></table>

Direct comparison between panoramic and pinhole renderings is non-trivial. To ensure a fair comparison with pinhole-based baselines, all methods are evaluated using pinhole rendering; three 120<sup>◦</sup> FoV views (forward/left/right) are rendered and averaged at each pose. For SynPano room1, only the forward view is used since the FoV of pinhole baselines does not cover the entire scene.

Ablative Analysis. Table IV presents an ablation study assessing the effectiveness of our proposed unit-sphere panoramic loss $\mathcal { L } _ { p a n o }$ and depth-guided Gaussian initialization strategy (DGIS). The unit-sphere loss $\mathcal { L } _ { p a n o }$ aligns geometrically with panoramic projection, effectively mitigating noise amplification near the poles, and thus substantially reduces trajectory errors across all sequences. For comparison, ${ \mathcal { L } } _ { G S }$ employs a Gaussian weighting scheme that decreases penalties from the equator toward the poles, serving as a baseline to confirm the geometric superiority of our cosine-based $\mathcal { L } _ { p a n o } .$ Our DGIS yields a more reliable initial Gaussian map and enhances the quality of newly inserted Gaussians during incremental mapping, further lowering trajectory errors in most sequences.

TABLE IV  
ABLATION STUDY ON SYNPANO DATASET[13].
<table><tr><td>Method</td><td>ATE(cm) ↓</td><td>PSNR(dB) ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>w/o  $\mathcal { L } _ { p a n o }$ </td><td>5.19</td><td>27.23</td><td>0.86</td><td>0.30</td></tr><tr><td>W  ${ \mathcal { L } } _ { G S }$ </td><td>0.84</td><td>29.88</td><td>0.91</td><td>0.22</td></tr><tr><td>w/o DGIS</td><td>1.27</td><td>28.68</td><td>0.89</td><td>0.24</td></tr><tr><td>Ours</td><td>0.78</td><td>30.58</td><td>0.91</td><td>0.22</td></tr></table>

DGIS denote Depth-Guided Gaussian Initialization Strategy. ${ \mathcal { L } } _ { G S }$ is a Gaussian-weighted loss whose weights decrease progressively from the equator to the poles.

![](images/b0461859355a97ac91cb33043818f7c1073cfbc06fb8c77f731e90b4edd20490.jpg)  
Fig. 6. Trajectory Comparison under Different FoVs. Larger FoV lead to smoother trajectories and closer alignment with the ground truth.

Effect of Field-of-View. Table V presents a controlled FoV study on the SynPano dataset. Fig. 6 illustrates detailed trajectory comparisons against the ground truth under different FoV settings. Results indicate a clear monotonic trend: trajectory accuracy improves markedly as the visible angular range increases across all sequences. Notably, challenging sequences (e.g., room2 and room3) degrade severely at a narrow FoV (120°), but they experience a critical performance turning point when the FoV exceeds 200°. This phenomenon suggests that wide-field sensing fundamentally alters the conditioning of the underlying pose optimization problem. Under limited FoV, photometric gradients are concentrated within a narrow angular region, leading to insufficient rotational observability and unstable optimization. As the FoV expands, gradient contributions are distributed across a broader spherical domain, significantly improving numerical stability and convergence behavior. Furthermore, under the same FoV setting, our method consistently outperforms MonoGS across all four scenes.

TABLE V  
FOV STUDY ON SYNPANO DATASET[13] (ATE RMSE [CM]).
<table><tr><td>Method</td><td>FOV</td><td>room1</td><td>room2</td><td>room3</td><td>room4</td><td>room5</td></tr><tr><td>MonoGS[8]</td><td>120°</td><td>23.71</td><td>118.98</td><td>35.45</td><td>31.38</td><td>105.80</td></tr><tr><td rowspan="5">Ours</td><td>120°</td><td>5.26</td><td>59.64</td><td>102.28</td><td>25.82</td><td>19.96</td></tr><tr><td> $2 0 0 ^ { \circ }$ </td><td>2.89</td><td>11.29</td><td>0.86</td><td>6.34</td><td>6.27</td></tr><tr><td> $2 8 0 ^ { \circ }$ </td><td>1.36</td><td>2.48</td><td>0.34</td><td>0.81</td><td>4.73</td></tr><tr><td> $3 6 0 ^ { \circ }$ </td><td>0.45</td><td>1.79</td><td>0.26</td><td>0.21</td><td>1.08</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE VI

CONVERGENCE BEHAVIOR UNDER DIFFERENT ITERATION COUNTS ON PALVIO DATASET[14].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Front-end / Back-end Iterations</td></tr><tr><td>15/30</td><td>30/50</td><td>50/80</td><td>100/100</td></tr><tr><td rowspan="2">MonoGS[8]</td><td>ATE(m) ↓</td><td>2.314</td><td>1.872</td><td>1.566</td><td>1.490</td></tr><tr><td>FPS↑</td><td>5.07</td><td>3.22</td><td>2.13</td><td>1.19</td></tr><tr><td rowspan="2">Ours</td><td>ATE(m) ↓</td><td>0.056</td><td>0.054</td><td>0.052</td><td>0.053</td></tr><tr><td>FPS↑</td><td>7.04</td><td>4.47</td><td>3.16</td><td>1.46</td></tr></table>

![](images/2045099d6ac71cf6fe4c159abe8b0070bf050015aa009055aeecaad00945b4cc.jpg)  
Fig. 7. Convergence Basin On SynPano Dateset. Top: Loss contour map on the X–Z plane. Bottom: 3D visualization of the X–Z–Loss surface. Our PanoGS-SLAM shows faster and more robust convergence with a wider attraction basin and consistent gradients.

Pose Optimization Convergence. In Table VI, we observe that PanoGS-SLAM converges significantly faster during camera pose optimization. Specifically, PanoGS-SLAM achieves pose convergence within only 15 optimization iterations, reaching 7 FPS, whereas MonoGS exhibits convergence trends only after approximately 100 iterations. To further analyze the reasons for more stable and faster optimization, we conduct a convergence basin study by perturbing the camera pose around the ground-truth pose. Perturbations are scaled relative to the median scene depth (with a factor of 0.5) to maintain consistency across diverse environments. As illustrated in Fig 7, PanoGS-SLAM manifests a significantly broader and more well-conditioned basin of attraction within the optimization landscape, ensuring enhanced convexity and robust convergence compared to the narrow-FoV MonoGS. This improvement stems from the synergy between our native spherical representation and the differentiable rendering pipeline. Unlike pinhole models, which clip gradients at image boundaries, our panoramic framework maintains a continuous gradient field over the entire viewing sphere. This ensures strong gradient consistency in all directions, as gradients are not truncated in space, making optimization easier while enhancing pose observability and enabling faster, more stable convergence.

## V. CONCLUSION

We presented PanoGS-SLAM, the first panoramic SLAM framework that jointly performs camera tracking and 3D Gaussian map optimization directly within the spherical domain. To address the inherent distortions of equirectangular projections and maintain strict geometric consistency, we introduced a sphere-consistent photometric objective alongside a depth-guided Gaussian initialization strategy. Furthermore, our work reveals a fundamental connection between sensing geometry and numerical robustness in differentiable SLAM. Future research will explore the integration of global optimization and loop closure for large-scale, long-term deployments.

## REFERENCES

[1] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense tracking and mapping in real-time,” in Proc. 10th IEEE Int. Symp. Mixed and Augmented Reality (ISMAR), 2011, pp. 2320–2327.

[2] A. Dai, M. Nießner, M. Zollhofer, S. Izadi, and C. Theobalt, “Bundle-¨ Fusion: Real-time globally consistent 3D reconstruction using on-thefly surface reintegration,” ACM Trans. Graph., vol. 36, no. 4, 2017.

[3] R. A. Newcombe et al., “KinectFusion: Real-time dense surface mapping and tracking,” in Proc. 10th IEEE Int. Symp. Mixed and Augmented Reality (ISMAR), 2011, pp. 127–136.

[4] M. Keller, D. Lefloch, M. Lambers, S. Izadi, T. Weyrich, and A. Kolb, “Real-time 3D reconstruction in dynamic scenes using point-based fusion,” in Proc. Int. Conf. 3D Vision (3DV), 2013, pp. 1–8.

[5] T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger, “ElasticFusion: Real-time dense SLAM and light source estimation,” Int. J. Robot. Res., vol. 35, no. 14, pp. 1697–1716, 2016.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian ¨ splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 1–14, 2023.

[7] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “GS-SLAM: Dense visual SLAM with 3D Gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[8] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian splatting SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[9] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, “SplaTAM: Splat, track & map 3D Gaussians for dense RGB-D SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 21357–21366.

[10] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-SLAM: Realtime simultaneous localization and photorealistic mapping for monocular, stereo, and RGB-D cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[11] S. Ha, J. Yeon, and H. Yu, “RGBD GS-ICP SLAM,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 180–197.

[12] K. Wu, Z. Zhang, M. Tie, Z. Ai, Z. Gan, and W. Ding, “VINGS-Mono: Visual-inertial Gaussian splatting monocular SLAM in large scenes,” IEEE Trans. Robot., vol. 41, pp. 5912–5931, 2025.

[13] X. Guo, “SynPano: A synthetic panoramic dataset for SLAM,” GitHub repository, 2026. [Online]. Available: https://github. com/guoxf304/SynPano-Dataset

[14] Z. Wang, K. Yang, H. Shi, P. Li, F. Gao, and K. Wang, “LF-VIO: A visual-inertial-odometry framework for large field-of-view cameras with negative plane,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2022, pp. 4423–4430.

[15] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM: A´ versatile and accurate monocular SLAM system,” IEEE Trans. Robot., vol. 31, no. 5, pp. 1147–1163, 2015.

[16] R. Mur-Artal and J. D. Tardos, “ORB-SLAM2: An open-source SLAM´ system for monocular, stereo, and RGB-D cameras,” IEEE Trans. Robot., vol. 33, no. 5, pp. 1255–1262, 2017.

[17] C. Campos, R. Elvira, J. J. Gomez, J. M. M. Montiel, and J. D. Tard ´ os,´ “ORB-SLAM3: An accurate open-source library for visual, visualinertial and multi-map SLAM,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890, 2021.

[18] T. Qin, P. Li, and S. Shen, “VINS-Mono: A robust and versatile monocular visual-inertial state estimator,” IEEE Trans. Robot., vol. 34, no. 4, pp. 1004–1020, 2018.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 405–421.

[20] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “iMAP: Implicit mapping and positioning in real-time,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 6229–6238.

[21] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “NICE-SLAM: Neural implicit scalable encoding for SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 12786–12796.

[22] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, “Vox-Fusion: Dense tracking and mapping with voxel-based neural implicit representation,” in Proc. IEEE Int. Symp. Mixed and Augmented Reality (ISMAR), 2022, pp. 499–507.

[23] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, “Point-SLAM:¨ Dense neural point cloud-based SLAM,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 18433–18444.

[24] S. Sumikura, M. Shibuya, and K. Sakurada, “OpenVSLAM: A versatile visual SLAM framework,” in Proc. 27th ACM Int. Conf. Multimedia, 2019, pp. 2292–2295.

[25] Y. Zhang, K. Yang, Z. Wang, and K. Wang, “P2U-SLAM: A monocular wide-FoV SLAM system based on point uncertainty and pose uncertainty,” IEEE Trans. Intell. Transp. Syst., 2026.

[26] Z. Wang, K. Yang, H. Shi, P. Li, F. Gao, J. Bai, and K. Wang, “LF-VISLAM: A SLAM framework for large field-of-view cameras with negative imaging plane on mobile agents,” IEEE Trans. Autom. Sci. Eng., 2023.

[27] H. Huang and S.-K. Yeung, “360VO: Visual odometry using a single 360 camera,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2022, pp. 5594–5600.

[28] J. Bai, L. Huang, J. Guo, W. Gong, Y. Li, and Y. Guo, “360-GS: Layout-guided panoramic Gaussian splatting for indoor roaming,” in Proc. Int. Conf. 3D Vision (3DV), 2025, pp. 1042–1053.

[29] S. Lee, J. Chung, J. Huh, and K. M. Lee, “ODGS: 3D scene reconstruction from omnidirectional images with 3D Gaussian splattings,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

[30] L. Li, H. Huang, S.-K. Yeung, and H. Cheng, “OmniGS: Fast radiance field reconstruction using omnidirectional Gaussian splatting,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 2260– 2268.

[31] F.-E. Wang, Y.-H. Yeh, M. Sun, W.-C. Chiu, and Y.-H. Tsai, “BiFuse: Monocular 360 depth estimation via bi-projection fusion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020.