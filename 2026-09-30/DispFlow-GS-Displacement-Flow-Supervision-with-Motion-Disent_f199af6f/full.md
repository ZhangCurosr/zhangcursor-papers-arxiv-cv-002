# DispFlow-GS: Displacement Flow Supervision with Motion Disentangling for Monocular Deformable 3D Gaussian Splatting

Thai Duy Nguyen<sup>1</sup>, Haitian Zhang<sup>1</sup>, and Addison Lin Wang<sup>1,∗</sup>

Abstract— Accurate dynamic scene reconstruction is important for robotic perception, where temporally consistent representations of dynamic environments are essential. Deformable 3D Gaussian Splatting (3DGS) models dynamic scenes through deformation fields, and recent methods incorporate motion supervision by aligning rendered Gaussian flow with optical flow. However, we find that such Gaussian-flow-based supervision provides only limited improvements in motion modeling. We identify a fundamental limitation of this supervision paradigm, namely a domain gap between rendered Gaussian flow and optical flow. To address this limitation, we propose a motion supervision framework built on Displacement Flow, which splats per-Gaussian 3D displacements onto the image plane to provide direct and stable optimization signals. We further disentangle scene motion from camera motion via intermediate-view rendering, enabling more reliable motion priors and targeted constraints on deformation and geometry. We also observe a discrepancy between motion fidelity and image-based evaluation, where improved motion awareness does not necessarily translate into better rendered image quality or higher image-based metric scores. Motivated by this mismatch, we introduce Deformation-Rendering Consistency (DRC), a motion-aware metric that measures the alignment between predicted deformation and rendering improvement. Experiments on dynamic scene benchmarks show substantial improvements in motion localization and motion–rendering consistency, reaching up to 39% and $6 \%$ , respectively, while image-based metrics change by only about 0.1%. These results confirm the observed mismatch between motion fidelity and image-based evaluation, demonstrating the significance of DRC for motion-aware evaluation.

## I. INTRODUCTION

Novel-view synthesis [1]–[3] and 3D reconstruction of dynamic scenes is important for applications such as Simultaneous Localization and Mapping (SLAM) [4], and robotic navigation [5]. Building upon neural radiance field (NeRF) [2] and 3D Gaussian Splatting (3DGS) [1], recent dynamic 3DGS methods have demonstrated efficient reconstruction and rendering of time-varying scenes [6]– [8]. Dynamic 3DGS approaches model temporal evolution through three main paradigms, including: explicit 4D primitive-based methods, deformation-field-based methods, and frame-wise training methods. Explicit 4D methods incorporate time directly into Gaussian representations [8], [9], while frame-wise methods optimize temporally varying Gaussian states [10]. In contrast, deformation-field-based 3DGS maintains a canonical Gaussian representation and learns a deformation field to predict time-dependent changes in Gaussian attributes [6], [7]. This provides a compact and continuous representation of scene dynamics, with better parameter efficiency than explicit 4D primitives and stronger temporal coherence than independent frame-wise methods. In this work, we focus specifically on this deformation-fieldbased paradigm for motion-aware reconstruction.

![](images/6f72e0384e25d96ef1dc305e22014ecb55050e3259f2cfce14f342a7f176e2da.jpg)  
Fig. 1. Highlight of our motion disentangling framework. In dynamic scenarios, between frames there are two types of motion being entangled together: the camera motion (from view<sub>1</sub> to view<sub>2</sub>), and the scene motion (from state at $t _ { 1 } \ t _ { 0 } \ t _ { 2 } ) .$ . For efficient motion supervision, such two motions need to be disentangle. Our method uses Deformable 3DGS to render an intermediate image at $\left( \boldsymbol { v i e w _ { 1 } } , t _ { 2 } \right)$ , such that its motion relative to each input frame isolates a single motion component by keeping either the viewpoint or the scene state fixed during motion estimation.

Most deformable 3DGS methods rely primarily on photometric supervision between rendered and input images. However, photometric supervision is ambiguous for dynamic scenes, as different deformation patterns can produce similar appearance consistency. Recent works therefore introduce motion-aware constraints to improve deformation learning [11]–[13]. GaussianFlow [11] renders the motion of projected Gaussians and supervises it using optical flow, while subsequent approaches further exploit Gaussian-derived motion for motion decomposition and deformation learning [12], [14]. Our work focuses on this Gaussian-flow-based motion supervision within deformationfield-based 3DGS. Despite different supervision targets and formulations, these approaches align motion rendered from deformed Gaussians with flow-derived motion cues. Our empirical analysis shows that such supervision provides only limited improvement in motion-aware deformation. We identify a key limitation: Gaussian flow and optical flow represent motion differently, creating a domain gap between the rendered motion representation and its supervision target.

To address this limitation, we propose a novel motion supervision framework using Displacement Flow, obtained by splatting per-Gaussian 3D displacements onto the image plane. Rather than enforcing pixel correspondence between heterogeneous motion representations, Displacement Flow captures relative motion magnitude induced by Gaussian deformation. We further disentangle scene motion from camera-induced motion via intermediate-view rendering, as depicted in Fig. 1, allowing motion priors to be estimated under a fixed viewpoint for more reliable supervision. Since our focus is motion-aware deformation learning, image-based metrics such as PSNR, SSIM, and LPIPS cannot fully reflect motion fidelity. Empirical results in Sec. V-B show that improvements in motion quality do not necessarily translate into gains in these metrics. This distinction is particularly relevant to robotic perception, where correctly identifying dynamic regions can matter even when rendered appearance changes little. We therefore introduce Deformation– Rendering Consistency (DRC), which measures whether predicted deformation aligns with rendering improvement.

We evaluate our method on the NeRF-DS [15] and HyperNeRF [16] benchmarks. Our approach consistently improves motion-aware metrics while image-based performance remains largely unchanged. We show that improved motion awareness does not necessarily translate into better rendered image quality or higher image-based metric scores, highlighting the need for motion-aware evaluation. Our main contributions are summarized as follows:

1) We identify a domain gap between rendered Gaussian flow and optical-flow supervision as a key limitation of motion learning in deformable 3DGS.

2) We propose a motion supervision framework using Displacement Flow and scene-motion decomposition for more efficient deformation learning.

3) We reveal a discrepancy between motion fidelity and image-based evaluation, motivating Deformation– Rendering Consistency for motion-aware evaluation.

## II. RELATED WORKS

Dynamic Scene Reconstruction. The NeRF framework [2] has been extended to dynamic scenes through deformation fields that map observations to a shared canonical space [17], [18], direct spatiotemporal modeling [19]– [21], and structured representations [22]–[24]. Similarly, many works explore dynamic scene reconstruction with 3DGS [25], [26]. Dynamic 3DGS methods can be broadly grouped into explicit 4D primitive-based, deformation-fieldbased, and frame-wise approaches. Explicit 4D methods incorporate time directly into Gaussian primitives [8], [27], while frame-wise methods optimize Gaussian states across timestamps [10], [28]. In contrast, deformation-field-based methods maintain canonical Gaussians and predict timedependent transformations through a learned deformation field [6], [7]. Our work focuses on the deformation-fieldbased paradigm and studies motion supervision for more reliable motion-aware deformation learning.

![](images/1e8ce0406b7d90f7db2732811e126000e90aa18db06725308158273da0776e9d.jpg)  
Fig. 2. Gaussian flow formulation and its discrepancy with optical flow. Gaussian flow is the α-composited weighted sum of pixel displacements induced by projected 2D Gaussians. It considers only the Gaussian composition at $t _ { 1 } ,$ , ignoring changes in contributing Gaussians and blending weights due to deformation and viewpoint variation. In contrast, optical flow uses pixel positions at both $t _ { 1 }$ and $t _ { 2 } .$ , leading to a domain gap between two representations.

Motion-Aware Deformable 3DGS. Recent methods introduce motion supervision to improve deformation learning in deformable 3DGS. GaussianFlow [11] supervises rendered Gaussian flow using optical flow, while MotionGS [12] further decomposes camera and scene motion before applying flow guidance. Guo et al. [13] incorporate uncertainty-aware flow supervision, whereas Xie et al. [29] estimate Gaussian motion using a differentiable Lucas–Kanade formulation without explicit optical-flow supervision. Despite different designs, these approaches still align Gaussian-derived motion with flow-based targets whose motion definitions are inherently different, introducing the domain gap studied in this work. In contrast, we supervise Displacement Flow, constructed from per-Gaussian 3D displacements, using directly estimated scene motion under a fixed viewpoint, avoiding direct correspondence matching between Gaussian-derived motion and optical flow, which represent motion differently.

Motion Disentanglement for Robotics. Robotic perception in dynamic environments requires distinguishing platforminduced motion from motion intrinsic to the scene, as both are entangled in temporal observations. Recent works address this through dynamic–static scene decomposition for autonomous driving [30], scene-flow-based motion reasoning [31], [32], and motion segmentation using scene-flow and ego-motion consistency with 4D radar [33] or temporal LiDAR [34]. Image-based methods further combine optical flow and geometric constraints to separate scene motion from camera-induced motion [35]. When camera-induced motion and scene motion are not separated, the observed motion is difficult to attribute reliably to actual scene dynamics. Such ambiguity also affects dynamic scene representations for robotic perception, where inter-frame motion jointly reflects viewpoint change and scene dynamics. We address it through intermediate-view rendering, fixing the viewpoint to isolate scene motion for more reliable deformation supervision.

![](images/e1cc18fc384e0bd4acdf8e85e29ec1efd3e8fd4207dfe09ffdedb792eeabd530.jpg)  
Fig. 3. Overview of the proposed DispFlow-GS framework. The optimization consists of two components: photometric loss and motion loss. The photometric loss is the standard rendered image loss. The motion loss is defined by our Displacement Flow Supervision, where displacement flow is obtained by splatting per-Gaussian 3D displacements onto the image plane and supervising them against scene motion.

## III. REVISIT THE MOTION SUPERVISION FOR DEFORMABLE 3DGS

This section revisits a widely adopted motion supervision scheme in existing works [11]–[13], [29] that aims to improve motion awareness. In this approach, Gaussian flow is rendered by compositing the motion of projected 2D Gaussians through α-blending. The rendered flow map is then aligned with target optical flow to provide supervision. We then examine whether these two representations encode motion consistently, revealing their domain mismatch.

## A. Formulation of Gaussian Flow Supervision

Motion supervision in deformable 3DGS typically extracts a motion representation from deformed Gaussians and supervises it using priors from off-the-shelf flow models. Despite different supervision targets and losses, many methods [11]– [13], [29] adopt Gaussian flow as the motion representation. Since the supervision directly compares this rendered motion with an external flow prior, its formulation determines how effectively the target motion constrains the deformation field.

Introduced in GaussianFlow [11], Gaussian flow $F _ { t _ { 1 }  t _ { 2 } } ^ { G }$ is the α-composited displacement of projected 2D Gaussians:

$$
\begin{array} { l } { { \displaystyle { \cal F } _ { t _ { 1 } \to t _ { 2 } } ^ { G } = \sum _ { i = 1 } ^ { K } w _ { i } \left( { \bf x } _ { i , t _ { 2 } } - { \bf x } _ { t _ { 1 } } \right) } } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { K } w _ { i } \left( \Sigma _ { i , t _ { 2 } } \Sigma _ { i , t _ { 1 } } ^ { - 1 } ( { \bf x } _ { t _ { 1 } } - \mu _ { i , t _ { 1 } } ) + \mu _ { i , t _ { 2 } } - { \bf x } _ { t _ { 1 } } \right) . } } \end{array}\tag{1}
$$

Here, K is the number of contributing Gaussians and $\begin{array} { r } { w _ { i } = \frac { T _ { i } \alpha _ { i } } { \sum _ { i } T _ { j } \alpha _ { j } } } \end{array}$ is the normalized $\alpha$ -blending weight. $\mathbf { x } _ { t _ { 1 } }$ and ${ \bf x } _ { i , t _ { 2 } }$ denote the pixel coordinates before and after Gaussian deformation, while $\mu _ { i , t }$ and $\Sigma _ { i , t }$ are the Gaussian mean and covariance. The computation is illustrated in Fig. 2. Applying (1) to all pixels produces the rendered Gaussian flow, supervised by pseudo ground-truth optical flow:

$$
\mathcal { L } _ { \mathrm { f l o w } } =  F _ { t _ { 1 }  t _ { 2 } } ^ { o } ( \mathbf { x } _ { t _ { 1 } } ) - F _ { t _ { 1 }  t _ { 2 } } ^ { G }  ,\tag{2}
$$

where $F _ { t _ { 1 }  t _ { 2 } } ^ { o }$ is estimated using a pre-trained flow model.

## B. Problems of Gaussian Flow Supervision Scheme

From (1), Gaussian flow at a pixel is computed using the composition weights $w _ { i }$ at time $t _ { 1 }$ , modeling pixel displacement as the weighted aggregation of the displacements of Gaussians contributing at $t _ { 1 }$ . This formulation implicitly assumes that Gaussian contributions to a pixel remain consistent across timestamps. In practice, however, both the contributing Gaussians and their α-blending weights change over time due to deformation and reprojection, leading to a mismatch between the supervision signal and the rendered motion representation used for deformation learning.

Consequently, Gaussian flow does not model explicit correspondence between pixel locations at $t _ { 1 }$ and $t _ { 2 } .$ , but instead represents accumulated motion induced by Gaussian deformation at $t _ { 1 }$ . In contrast, optical flow establishes a direct mapping between pixel positions at $t _ { 1 }$ and their landing positions at $t _ { 2 } ,$ as illustrated in Fig. 2. This difference introduces a domain gap between Gaussian flow and optical flow, preventing the semantic meaning of optical flow from being faithfully captured under this formulation.

## IV. METHODOLOGY

## A. Displacement Flow Supervision

Design Rationales. A natural way to address the domain gap in Sec. III-B is to model Gaussian composition at both $t _ { 1 }$ and $t _ { 2 }$ and establish cross-time correspondence. However, obtaining correspondence at $t _ { 2 }$ would already amount to solving optical flow, making this formulation circular. We therefore reinterpret Gaussian-derived motion as a motion magnitude map rather than a correspondence field. This motivates a representation that captures relative Gaussian motion without requiring explicit cross-time pixel correspondence.

Formulation of Displacement Flow. We propose Displacement Flow, derived by splatting per-Gaussian 3D displacements onto the image plane (see Fig. 3). At each step, a pair of images at timestamps $t _ { 1 }$ and $t _ { 2 }$ is processed, producing two sets of deformations from the canonical 3DGS:

$$
\delta _ { x _ { i } } ^ { t _ { 1 } } , \delta _ { r _ { i } } ^ { t _ { 1 } } , \delta _ { s _ { i } } ^ { t _ { 1 } } = \mathcal { D } ( x _ { i } , t _ { 1 } ) , \quad \delta _ { x _ { i } } ^ { t _ { 2 } } , \delta _ { r _ { i } } ^ { t _ { 2 } } , \delta _ { s _ { i } } ^ { t _ { 2 } } = \mathcal { D } ( x _ { i } , t _ { 2 } ) .\tag{3}
$$

The 3D displacement of each Gaussian from $t _ { 1 }$ to $t _ { 2 }$ is computed as

$$
\delta _ { x _ { i } } ^ { t _ { 1 } t _ { 2 } } = \delta _ { x _ { i } } ^ { t _ { 2 } } - \delta _ { x _ { i } } ^ { t _ { 1 } } .\tag{4}
$$

![](images/a07fc0c336325572b90a85ad9a2e86b26a0e31f54905621ce05637e2b9011312.jpg)  
Fig. 4. Illustration of scene motion calculation. For an input pair $( v i e w _ { 1 } , t _ { 1 } )$ and (view<sub>2</sub>, t<sub>2</sub>), total motion includes scene and camera motion. MotionGS estimates cross-view flow and subtracts camera motion, which is error-prone due to large cross-view displacements. Instead, we render an intermediate frame at $\left( \boldsymbol { v i e w _ { 1 } } , t _ { 2 } \right)$ and compute flow between $( v i e w _ { 1 } , t _ { 1 } )$ and $\left( \boldsymbol { v } i e w _ { 1 } , t _ { 2 } \right)$ to isolate scene motion under a fixed viewpoint.

Displacement Flow is then obtained by splatting these 3D displacements onto the image plane using α-blending:

$$
F _ { \mathrm { d i s p } } = \sum _ { i = 1 } ^ { K } w _ { i } \delta _ { x _ { i } } ^ { t _ { 1 } t _ { 2 } } ,\tag{5}
$$

where $w _ { i }$ is the α-blending weight and K is the number of contributing Gaussians. Unlike Gaussian flow, Displacement Flow is constructed directly from 3D center displacements, providing a direct gradient path to the deformation field.

Supervision. The displacement flow $F _ { \mathrm { d i s p } }$ is supervised against scene motion $F _ { \mathrm { s c e n e } } ,$ which excludes camera-induced motion (detailed in Sec. IV-B). Following the discussion above, both flows are normalized into a motion magnitude representation in [0, 1] using min-max normalization.

The motion loss is defined as

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \left. \hat { F } _ { \mathrm { s c e n e } } - \hat { F } _ { \mathrm { d i s p } } \right. _ { 1 } .\tag{6}
$$

Rather than enforcing exact pixel correspondence, this supervision aligns relative motion structures across the scene, encouraging deformation updates that are consistent with dynamic regions while mitigating the domain gap limitation discussed in Sec. III-B.

## B. Motion Disentangling via Intermediate Image Rendering

At each optimization step, we process an image pair from different viewpoints and timestamps. Two motion components occur simultaneously: scene motion (temporal deformation) and camera motion (viewpoint change), denoted as $( v i e w _ { 1 } , t _ { 1 } ) \to ( v i e w _ { 2 } , t _ { 2 } )$ . Since Displacement Flow models only scene motion, these components must be disentangled. Using the differentiable rendering of deformable 3DGS, we render an intermediate state at $\left( \boldsymbol { v i e w _ { 1 } } , t _ { 2 } \right)$ , decomposing the transformation as:

$$
( v i e w _ { 1 } , t _ { 1 } )  ( v i e w _ { 1 } , t _ { 2 } )  ( v i e w _ { 2 } , t _ { 2 } ) ,\tag{7}
$$

where the first stage isolates scene motion and the second camera motion. Rendering at $( v i e w _ { 2 } , t _ { 1 } )$ produces the reverse order. We adopt the former so scene motion is expressed under view<sub>1</sub>, which is directly constrained by photometric loss, leading to more stable supervision.

These decomposed pairs enable separate estimation of scene motion flow and camera flow using off-the-shelf optical flow models. The resulting flows provide pseudo groundtruth motion priors during optimization. This decomposition also improves flow reliability, as directly estimating flow between the original pair can be inaccurate under large or complex motion. See Fig. 4 for illustration.

## C. Optimization

The overall training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { p h o t o } } + \alpha _ { t } \mathcal { L } _ { \mathrm { m o t i o n } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { p h o t o } }$ is the photometric loss and ${ \mathcal { L } } _ { \mathrm { m o t i o n } }$ is the proposed motion supervision term. The coefficient $\alpha _ { t }$ controls the contribution of motion supervision during training.

To stabilize optimization, motion supervision is introduced after an initial warm-up period and its weight is gradually increased. For iterations $t \geq t _ { 0 } , \alpha _ { t }$ is linearly ramped from 0.1α to α, while $\alpha _ { t } = 0$ for $t < t _ { 0 }$ . This scheduling prevents motion supervision from dominating early training before photometric reconstruction stabilizes, allowing the deformation field to first learn a reasonable geometric initialization and later refine motion-consistent deformation.

## V. EXPERIMENTS

Implementation Details. Training runs for 20,000 iterations, with the first 3,000 iterations used for warm-up. Motion supervision is introduced at iteration 10,000 with a final weight of $\alpha = 0 . 1$ for both the NeRF-DS [15] and HyperNeRF [16] datasets. The spatial learning-rate scale for the deformation field is set to 12. We use the pre-trained GMFlow [36] model to estimate scene motion. Other hyperparameters follow the baseline method [7]. Experiments are conducted using PyTorch [37] on a single NVIDIA RTX 5090 GPU.

Benchmark Datasets. We evaluate our method on two representative real-world dynamic scene benchmarks: NeRF-DS [15] and HyperNeRF [16]. The image resolutions and train–test splits follow the original dataset protocols.

## A. Evaluation Metrics

Image-based Metrics. Following prior works, we report PSNR, SSIM, and LPIPS for image reconstruction quality.

TABLE I  
QUANTITATIVE COMPARISON USING IMAGE-BASED METRICS ON THE NERF-DS DATASET PER SCENE.  
WE HIGHLIGHT THE BEST AND SECOND-BEST RESULTS IN EACH SCENE.
<table><tr><td></td><td colspan="3">Sieve</td><td colspan="3">Plate</td><td colspan="3">Bell</td><td colspan="3">Press</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>3D-GS [1]</td><td>23.16</td><td>0.8203</td><td>0.2247</td><td>16.14</td><td>0.6970</td><td>0.4093</td><td>21.01</td><td>0.7885</td><td>0.2503</td><td>22.89</td><td>0.8163</td><td>0.2904</td></tr><tr><td>TiNeuVox [38]</td><td>21.49</td><td>0.8265</td><td>0.3176</td><td>20.58</td><td>0.8027</td><td>0.3317</td><td>23.08</td><td>0.8242</td><td>0.2568</td><td>24.47</td><td>0.8613</td><td>0.3001</td></tr><tr><td>HyperNeRF [16]</td><td>25.43</td><td>0.8798</td><td>0.1645</td><td>18.93</td><td>0.7709</td><td>0.2940</td><td>23.06</td><td>0.8097</td><td>0.2052</td><td>26.15</td><td>0.8897</td><td>0.1959</td></tr><tr><td>NeRF-DS [15]</td><td>25.78</td><td>0.8900</td><td>0.1472</td><td>20.54</td><td>0.8042</td><td>0.1996</td><td>23.19</td><td>0.8212</td><td>0.1867</td><td>25.72</td><td>0.8618</td><td>0.2047</td></tr><tr><td>Deformable-3DGS [7]</td><td>25.27</td><td>0.8682</td><td>0.1532</td><td>20.51</td><td>0.8090</td><td>0.2271</td><td>25.07</td><td>0.8446</td><td>0.1725</td><td>25.45</td><td>0.8619</td><td>0.1941</td></tr><tr><td>MotionGS [12]</td><td>25.49</td><td>0.8443</td><td>0.2193</td><td>20.19</td><td>0.7920</td><td>0.2685</td><td>24.89</td><td>0.8198</td><td>0.2465</td><td>25.62</td><td>0.8539</td><td>0.2415</td></tr><tr><td>DispFlow-GS</td><td>25.33</td><td>0.8685</td><td>0.1570</td><td>20.27</td><td>0.8045</td><td>0.2344</td><td>25.19</td><td>0.8459</td><td>0.1676</td><td>25.56</td><td>0.8644</td><td>0.1940</td></tr><tr><td rowspan="2"></td><td colspan="3">Cup</td><td colspan="3">As</td><td colspan="3">Basin</td><td colspan="3">Mean</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Method</td><td></td><td></td><td>0.2548</td><td>22.69</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-GS [1]</td><td>21.71</td><td>0.8304</td><td></td><td>21.26</td><td>0.8017</td><td>0.2994</td><td>18.42</td><td>0.7170</td><td>0.3153 0.2690</td><td>20.86 21.61</td><td>0.7816 0.8241</td><td>0.2920 0.3195</td></tr><tr><td>TiNeuVox [38]</td><td>19.71</td><td>0.8109</td><td>0.3643</td><td></td><td>0.8289</td><td>0.3967</td><td>20.66 20.41</td><td>0.8145 0.8199</td><td>0.1911</td><td>23.45</td><td>0.8488</td><td>0.1991</td></tr><tr><td>HyperNeRF [16]</td><td>24.59</td><td>0.8770 0.8741</td><td>0.1650 0.1737</td><td>25.58 25.13</td><td>0.8949 0.8778</td><td>0.1777 0.1741</td><td>19.96</td><td>0.8166</td><td>0.1855</td><td>23.60</td><td>0.8494</td><td>0.1816</td></tr><tr><td>NeRF-DS [15] Deformable-3DGS [7]</td><td>24.91 24.20</td><td>0.8816</td><td>0.1681</td><td>26.12</td><td>0.8814</td><td>0.1856</td><td>19.68</td><td>0.7927</td><td>0.1918</td><td>23.76</td><td>0.8485</td><td>0.1846</td></tr><tr><td>MotionGS [12]</td><td>24.23</td><td>0.8553</td><td>0.2109</td><td>26.02</td><td>0.8626</td><td>0.2256</td><td>19.79</td><td>0.7751</td><td>0.2426</td><td>23.75</td><td>0.8290</td><td>0.2364</td></tr><tr><td>DispFlow-GS</td><td>24.47</td><td>0.8876</td><td>0.1620</td><td>26.08</td><td>0.8770</td><td>0.1991</td><td>19.68</td><td>0.7914</td><td>0.1941</td><td>23.80</td><td>0.8485</td><td>0.1869</td></tr></table>

Motion-based Metrics. Image-based metrics mainly measure appearance and do not directly assess motion quality. We therefore introduce Deformation-Rendering Consistency (DRC) to measure whether predicted deformation aligns with rendering improvement. To avoid bias from evaluating solely with our proposed DRC, we also use MMAP, which follows the standard Average Precision (AP) evaluation protocol.

a) Deformation-Rendering Consistency (DRC): DRC measures whether deformation-induced motion improves reconstruction. It does not aim to measure true motion correspondence, but rather whether predicted motion is concentrated in regions where deformation reduces rendering error.

For consecutive frames t−1 and t, let $I _ { \mathrm { f r o z e n } }$ denote the render using the previous deformation field D(t−1) and $I _ { \mathrm { f u l l } }$ the render using the updated deformation D(t). The deformation gain is

$$
\Delta \mathcal { E } ( p ) = | I _ { \mathrm { f r o z e n } } ( p ) - I _ { \mathrm { g t } } ( p ) | - | I _ { \mathrm { f u l l } } ( p ) - I _ { \mathrm { g t } } ( p ) | ,\tag{9}
$$

where $\Delta \mathcal { E } ( p ) > 0$ indicates improved reconstruction. Let $\hat { F } _ { \mathrm { d i s p } } ( p ) \in [ 0 , 1 ]$ denote the normalized displacement flow. DRC is defined as

$$
\mathrm { D R C } = \frac { \sum _ { p } \hat { F } _ { \mathrm { d i s p } } ( p ) \mathbf { 1 } [ \Delta \mathcal { E } ( p ) > 0 ] } { \sum _ { p } \hat { F } _ { \mathrm { d i s p } } ( p ) + \varepsilon } \in [ 0 , 1 ] ,\tag{10}
$$

where higher DRC indicates stronger alignment between predicted motion and error-reducing deformation.

b) Motion Mask Average Precision (MMAP): MMAP evaluates motion localization using the standard AP evaluation protocol. Let $M _ { f } ( p ) \in \{ 0 , 1 \}$ denote the ground truth motion label and $\hat { F } _ { \mathrm { d i s p } , f } ( p ) \in [ 0 , 1 ]$ the normalized displacement flow as per-pixel motion confidence. For each frame, AP is computed from the precision–recall curve by thresholding $\hat { F } _ { \mathrm { d i s p } , f }$ against $M _ { f } .$ , and MMAP is the mean $\mathbf { A P }$ over all test frames. Higher MMAP indicates better alignment between predicted motion and true dynamic regions. Both metrics require displacement flow and are evaluated on image pairs, consistent with the training procedure.

TABLE II  
QUANTITATIVE COMPARISON USING MOTION-BASED METRICS ON THE  
NERF-DS DATASET PER SCENE.
<table><tr><td></td><td colspan="2">Sieve</td><td colspan="2">Plate</td><td colspan="2">Bell</td><td colspan="2">Press</td></tr><tr><td>Method</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Deformable-3DGS [7]</td><td>0.4220</td><td>0.6320</td><td>0.7514</td><td>0.7088</td><td>0.8562</td><td>0.7608</td><td>0.7346</td><td>0.6994</td></tr><tr><td>MotionGS [12]</td><td>0.5905</td><td>0.6600</td><td>0.7149</td><td>0.7095</td><td>0.8236</td><td>0.7395</td><td>0.7398</td><td>0.6867</td></tr><tr><td>DispFlow-GS</td><td>0.8724</td><td>0.6957</td><td>0.9204</td><td>0.7573</td><td>0.8936</td><td>0.7769</td><td>0.9128</td><td>0.7133</td></tr><tr><td></td><td colspan="2">Cup</td><td colspan="2">As</td><td colspan="2">Basin</td><td colspan="2">Mean</td></tr><tr><td>Method</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Deformable-3DGS [7]</td><td>0.6066</td><td>0.6060</td><td>0.3566</td><td>0.6422</td><td>0.8205</td><td>0.6127</td><td>0.6497</td><td>0.6660</td></tr><tr><td>MotionGS [12]</td><td>0.6733</td><td>0.6132</td><td>0.3818</td><td>0.6516</td><td>0.8538</td><td>0.6193</td><td>0.6825</td><td>0.6685</td></tr><tr><td>DispFlow-GS</td><td>0.9356</td><td>0.7036</td><td>0.8449</td><td>0.6868</td><td>0.9308</td><td>0.6231</td><td>0.9015</td><td>0.7081</td></tr></table>

## B. Results

1) Results on the NeRF-DS dataset: Quantitative results are reported in Tab. I and Tab. II. DispFlow-GS achieves the best motion-based performance across all scenes, improving mean MMAP and DRC over the photometriconly Deformable 3DGS [7] by approximately 39% and 6%, respectively, while PSNR changes by only about 0.1%. This indicates that the improvement brought by motion supervision is expressed primarily in the learned deformation rather than in conventional reconstruction quality. Qualitative results in Fig. 5 support this observation, where our method localizes deformation more consistently to dynamic regions and suppresses responses in static backgrounds. The basin scene provides a particularly clear example: MotionGS produces visibly blurrier reconstructions while attaining the highest PSNR among the three deformable-3DGS methods, showing that a better image-based score does not necessarily correspond to more faithful motion modeling.

This discrepancy reflects the different objectives of photometric and motion supervision. Photometric optimization can improve appearance by allowing flexible deformation that explains the observed images, without requiring the resulting deformation to correspond to the actual dynamic regions. In contrast, motion supervision explicitly constrains where and how deformation should occur, which may restrict this photometric flexibility while producing more meaningful scene motion. Thus, substantial improvements in motionaware deformation can occur with little change in imagebased metrics, motivating the use of motion-specific metrics when evaluating motion-supervised dynamic reconstruction.

Deformable 3DGS  
TABLE III  
QUANTITATIVE COMPARISON ON THE HYPERNERF DATASET PER SCENE.
<table><tr><td></td><td colspan="4">Chicken</td><td colspan="4">Banana</td><td colspan="4">Mean</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>MMAP↑</td><td>DRC↑</td><td>PSNR↑</td><td>SSIM↑</td><td>MMAP↑</td><td>DRC↑</td><td>PSNR↑</td><td>SSIM↑</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>HyperNeRF [16]</td><td>27.40</td><td>0.630</td><td>=</td><td></td><td>22.10</td><td>0.720</td><td></td><td></td><td>24.80</td><td>0.680</td><td>=</td><td></td></tr><tr><td>TiNeuVox [38]</td><td>28.20</td><td>0.790</td><td></td><td></td><td>24.40</td><td>0.640</td><td></td><td></td><td>26.30</td><td>0.720</td><td></td><td></td></tr><tr><td>Deformable-3DGS [7]</td><td>23.58</td><td>0.646</td><td>0.529</td><td>0.729</td><td>25.16</td><td>0.831</td><td>0.749</td><td>0.760</td><td>24.37</td><td>0.738</td><td>0.639</td><td>0.744</td></tr><tr><td>MotionGS [12]</td><td>23.44</td><td>0.638</td><td>0.625</td><td>0.729</td><td>24.90</td><td>0.829</td><td>0.751</td><td>0.752</td><td>24.17</td><td>0.733</td><td>0.688</td><td>0.741</td></tr><tr><td>DispFlow-GS</td><td>23.66</td><td>0.649</td><td>0.664</td><td>0.743</td><td>24.97</td><td>0.830</td><td>0.774</td><td>0.758</td><td>24.31</td><td>0.739</td><td>0.719</td><td>0.750</td></tr></table>

DispFlow-GS

Ground truth  
Deformable 3DGS  
MotionGS  
DispFlow-GS  
![](images/0cea42e0d06c2c777f73ae4dc473d861cba678c3979284d3aab957eaa4479622.jpg)  
Fig. 5. Qualitative results on representative scenes of NeRF-DS dataset. For each scene, the top row shows the rendered images, while the bottom row shows the corresponding Displacement Flow.

2) Results on the HyperNeRF dataset: We report results on the “chicken” and “banana” scenes in Tab. III, with qualitative comparisons in Fig. 6 for the “chicken” scene. We omit the remaining scenes because several HyperNeRF sequences contain noticeable camera-pose inaccuracies, as also reported in Deformable 3DGS [7]. Such errors can degrade deformation learning in deformable 3DGS methods and also affect intermediate-view rendering used for scenemotion estimation; corresponding failure cases are discussed in Sec. VI. DispFlow-GS consistently improves motionaware performance on the evaluated scenes, achieving the highest MMAP on both scenes and the best average MMAP and DRC. Based on the mean results, MMAP improves by approximately 4.5% over MotionGS and 12.5% over photometric-only Deformable 3DGS, while image-based performance remains comparable. Qualitatively, our method also produces deformation that is more concentrated around the moving object with fewer spurious responses in static regions, indicating that the proposed supervision generalizes beyond NeRF-DS while preserving reconstruction quality.

## C. Ablation Study

We conduct ablations on NeRF-DS dataset to evaluate key components and design choices of our framework. Results are averaged across scenes and reported in Tabs. IV–VII. Key components. Tab. IV evaluates Displacement Flow (DF), Motion Decomposition (MD), and Training Scheduling (TS). DF alone does not improve over the photometric baseline, indicating that changing the motion representation is insufficient when the supervision target still contains ambiguous motion. Once MD is introduced, MMAP increases by 47.7% and DRC by 12.5% over DF alone, showing that motion decomposition is the key component that enables Displacement Flow to learn meaningful scene dynamics. Without TS, this stronger motion constraint slightly degrades reconstruction quality; introducing TS restores the imagebased metrics to approximately the baseline level while retaining most of the motion improvement, resulting in the final balance between motion and appearance.

Ground truth  
MotionGS  
![](images/e1e6d49cebe58607ff85987256e3899b08a27406b6e012f299e86822fe468885.jpg)  
Fig. 6. Qualitative results on ‘chicken’ scene of the HyperNeRF dataset.  
TABLE IV

ABLATION ON KEY COMPONENTS: DISPLACEMENT FLOW (DF), MOTION  
DECOMPOSITION (MD), AND TRAINING SCHEDULING (TS).
<table><tr><td>Method</td><td>DF</td><td>MD</td><td>TS</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>23.76</td><td>0.8485</td><td>0.1846</td><td>0.6497</td><td>0.6660</td></tr><tr><td>1</td><td>√</td><td></td><td></td><td>23.63</td><td>0.8406</td><td>0.1941</td><td>0.6306</td><td>0.6455</td></tr><tr><td>2</td><td>√</td><td></td><td>√</td><td>23.77</td><td>0.8462</td><td>0.1878</td><td>0.6817</td><td>0.6577</td></tr><tr><td>3</td><td>√</td><td>√</td><td></td><td>23.62</td><td>0.8467</td><td>0.1902</td><td>0.9316</td><td>0.7261</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>23.80</td><td>0.8485</td><td>0.1869</td><td>0.9015</td><td>0.7081</td></tr></table>

TABLE V  
ABLATION ON MOTION REPRESENTATIONS AND SUPERVISION TARGETS.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Gaussian Flow + Indirect Scene Motion (MotionGS)</td><td>23.75</td><td>0.8290</td><td>0.2364</td><td>0.6825</td><td>0.6685</td></tr><tr><td>Gaussian Flow + Direct Scene Motion</td><td>23.85</td><td>0.8485</td><td>0.1846</td><td>0.6317</td><td>0.6673</td></tr><tr><td>Displacement Flow + Indirect Scene Motion</td><td>23.92</td><td>0.8498</td><td>0.1833</td><td>0.7258</td><td>0.6728</td></tr><tr><td>Displacement Flow + Direct Scene Motion (Ours)</td><td>23.80</td><td>0.8485</td><td>0.1869</td><td>0.9015</td><td>0.7081</td></tr></table>

Motion representation and supervision target. Tab. V further separates the effects of the motion representation and scene-motion target. Under the same indirect scene-motion supervision, replacing Gaussian Flow with Displacement Flow already improves MMAP, suggesting that it is better aligned with deformation learning. More importantly, switching from indirect to direct scene-motion estimation increases MMAP by 24.2% for Displacement Flow, whereas it reduces MMAP for Gaussian Flow. This result shows that Displacement Flow and direct scene-motion supervision are complementary, with their combination giving the strongest motion-aware performance.

TABLE VI  
ABLATION ON FLOW BACKBONES, LOSS WEIGHTS, AND MOTION MASKS.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Different flow model (RAFT [39])</td><td>23.78</td><td>0.8488</td><td>0.1847</td><td>0.9024</td><td>0.7083</td></tr><tr><td>Different flow model (MDFlow [40])</td><td>23.73</td><td>0.8476</td><td>0.1865</td><td>0.8084</td><td>0.6824</td></tr><tr><td>Smaller motion loss weight (α = 0.02)</td><td>23.78</td><td>0.8478</td><td>0.1873</td><td>0.8206</td><td>0.6866</td></tr><tr><td>Larger motion loss weight (α = 0.5)</td><td>23.61</td><td>0.8464</td><td>0.1910</td><td>0.9269</td><td>0.7216</td></tr><tr><td>Motion decomposition in reverse order</td><td>23.87</td><td>0.8472</td><td>0.1892</td><td>0.5566</td><td>0.6473</td></tr><tr><td>Motion mask filtering</td><td>23.78</td><td>0.8475</td><td>0.1889</td><td>0.8629</td><td>0.7063</td></tr><tr><td>Ours (GMFlow [36]; α = 0.1; w/o mask)</td><td>23.80</td><td>0.8485</td><td>0.1869</td><td>0.9015</td><td>0.7081</td></tr></table>

TABLE VII  
ABLATION STUDY ON MOTION NORMALIZATION STRATEGIES.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MMAP↑</td><td>DRC↑</td></tr><tr><td>Max-Absolute Normalization</td><td>23.75</td><td>0.8485</td><td>0.1869</td><td>0.8970</td><td>0.7041</td></tr><tr><td>Z-score Normalization</td><td>23.39</td><td>0.8388</td><td>0.2024</td><td>0.8926</td><td>0.6973</td></tr><tr><td>Interquartile Range Normalization</td><td>23.76</td><td>0.8483</td><td>0.1862</td><td>0.9254</td><td>0.7115</td></tr><tr><td>Min-Max Normalization (Ours)</td><td>23.80</td><td>0.8485</td><td>0.1869</td><td>0.9015</td><td>0.7081</td></tr></table>

Motion supervision settings. Tab. VI examines the flow backbone, motion-loss weight, decomposition order, and motion-mask filtering. RAFT [39] gives nearly identical results to the default GMFlow [36], while MDFlow [40] performs worse, suggesting that flow-prior quality matters but the gains are not specific to GMFlow; we retain GMFlow for fair comparison with MotionGS. Increasing the motion-loss weight improves motion metrics but sacrifices image quality, whereas reducing it weakens motion learning, supporting α = 0.1 as a balanced setting. Most notably, reversing the decomposition order reduces MMAP by approximately 38%, despite maintaining competitive image-based metrics. The decomposition order is therefore critical, as estimating scene motion under the photometrically constrained input viewpoint provides substantially more reliable supervision. Motion-mask filtering provides no consistent improvement, so it is omitted from the final model.

Normalization strategy. Tab. VII shows that our method is relatively insensitive to the normalization strategy. Interquartile-range normalization achieves the highest MMAP and DRC, but only modestly outperforms min-max, which gives slightly better image-based performance. We therefore adopt min-max for its simplicity and balanced performance across image- and motion-based metrics.

## VI. DISCUSSION AND LIMITATIONS

Potential for Robotics. Robotic perception and spatial reasoning in dynamic environments require separating egomotion from scene motion for reliable scene understanding. Our framework addresses this ambiguity by isolating scene motion under a fixed viewpoint and learning deformation concentrated on dynamic regions. Such motion-aware representations could support dynamic SLAM, navigation, and manipulation, where distinguishing persistent structure from transient motion is important. Our results also show that motion fidelity can improve substantially while image-based metrics remain nearly unchanged, suggesting that rendering quality alone may overlook properties relevant to robotic perception and spatial reasoning. Together, these results highlight the potential of motion-aware deformable 3DGS for robotic dynamic scene understanding and spatial reasoning. Failure Cases. Failure cases occur in the “3D printer” and “broom” scenes of HyperNeRF (see Fig. 7), mainly due to inaccurate camera poses, as also reported in Deformable-3DGS [7]. This leads to poor reconstruction of dynamic regions, such as the filament in “3D printer” and the broom in “broom”. Since our Displacement Flow supervision estimates scene motion from intermediate-frame rendering, it is particularly sensitive to pose errors.

Ground truth  
Deformable 3DGS  
MotionGS  
DispFlow-GS  
![](images/041358b175562e9e9aef10b24203d7124f4c40f03562d9b2c5b2a1cd528158ec.jpg)  
Fig. 7. Failure cases on the HyperNeRF dataset.

Limitations. Our framework is sensitive to optical-flow and camera-pose errors, affecting deformation learning and intermediate-frame rendering. Moreover, supervising motion magnitude rather than exact pixel correspondence may limit precise trajectory recovery in complex dynamic scenes.

## VII. CONCLUSION

In this paper, we revisited motion supervision in Deformable 3DGS and showed that Gaussian flow is fundamentally limited by a domain mismatch with optical flow, weakening its ability to learn meaningful scene dynamics. We introduced Displacement Flow and scene–camera motion disentangling to provide more direct and domain-consistent supervision for deformation learning. Our experiments reveal a central finding: improved motion awareness does not necessarily translate into better photometric reconstruction or higher image-based metric scores. To better evaluate this discrepancy, we introduced Deformation–Rendering Consistency (DRC), which measures the alignment between predicted motion and rendering improvement. Together, these findings suggest that motion loss may offer limited benefit when image quality is the primary goal, whereas motionaware deformation should be supervised and evaluated with motion-specific objectives and metrics. Building on these findings, future work will investigate task-oriented extensions of our motion disentangling and Displacement Flow for robotic dynamic scene understanding, with explicit integration into mapping and spatial-reasoning frameworks.

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.,” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99– 106, 2021.

[3] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in European conference on computer vision, pp. 333– 350, Springer, 2022.

[4] Z. Sun, J. Lo, and J. Hu, “Embracing dynamics: Dynamics-aware 4d gaussian splatting slam,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5331–5338, IEEE, 2025.

[5] Y. Wang, J. Morris, L. Wu, T. A. Vidal-Calleja, and V. Ila, “Dynorecon: Dynamic object reconstruction for navigation,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 3305– 3311, IEEE, 2025.

[6] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310–20320, 2024.

[7] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331–20341, 2024.

[8] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting for real-time dynamic view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8508–8520, 2024.

[9] Z. Yang, Z. Pan, X. Zhu, L. Zhang, J. Feng, Y.-G. Jiang, and P. H. Torr, “4d gaussian splatting: Modeling dynamic scenes with native 4d primitives,” arXiv preprint arXiv:2412.20720, 2024.

[10] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis,” in 2024 International Conference on 3D Vision (3DV), pp. 800–809, IEEE, 2024.

[11] Q. Gao, Q. Xu, Z. Cao, B. Mildenhall, W. Ma, L. Chen, D. Tang, and U. Neumann, “Gaussianflow: Splatting gaussian dynamics for 4d content creation,” arXiv preprint arXiv:2403.12365, 2024.

[12] R. Zhu, Y. Liang, H. Chang, J. Deng, J. Lu, W. Yang, T. Zhang, and Y. Zhang, “Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting,” Advances in Neural Information Processing Systems, vol. 37, pp. 101790–101817, 2024.

[13] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3d gaussian splatting for efficient dynamic scene reconstruction,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[14] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21136– 21145, 2024.

[15] Z. Yan, C. Li, and G. H. Lee, “Nerf-ds: Neural radiance fields for dynamic specular objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8285– 8295, 2023.

[16] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, “Hypernerf: A higherdimensional representation for topologically varying neural radiance fields,” arXiv preprint arXiv:2106.13228, 2021.

[17] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865–5874, 2021.

[18] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “Dnerf: Neural radiance fields for dynamic scenes,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10318–10327, 2021.

[19] Z. Li, Q. Wang, F. Cole, R. Tucker, and N. Snavely, “Dynibar: Neural dynamic image-based rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4273– 4284, 2023.

[20] W. Xian, J.-B. Huang, J. Kopf, and C. Kim, “Space-time neural irradiance fields for free-viewpoint video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9421–9431, 2021.

[21] C. Gao, A. Saraf, J. Kopf, and J.-B. Huang, “Dynamic view synthesis from dynamic monocular video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5712–5721, 2021.

[22] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 130–141, 2023.

[23] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-planes: Explicit radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479–12488, 2023.

[24] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, “Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16632–16642, 2023.

[25] K. Katsumata, D. M. Vo, and H. Nakayama, “A compact dynamic 3d gaussian representation for real-time dynamic view synthesis,” in European Conference on Computer Vision, pp. 394–412, Springer, 2024.

[26] J. Lei, Y. Weng, A. W. Harley, L. Guibas, and K. Daniilidis, “Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6165–6177, 2025.

[27] S. Sun, C. Zhao, Z. Sun, Y. V. Chen, and M. Chen, “Splatflow: Selfsupervised dynamic gaussian splatting in neural motion flow field for autonomous driving,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27487–27496, 2025.

[28] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photorealistic free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20675– 20685, 2024.

[29] L. Xie, J. Julin, K. Niinuma, and L. A. Jeni, “Gaussian splatting lucaskanade,” arXiv preprint arXiv:2407.11309, 2024.

[30] S. Doll, N. Hanselmann, L. Schneider, R. Schulz, M. Cordts, M. Enzweiler, and H. P. A. Lensch, “DualAD: Disentangling the dynamic and static world for end-to-end driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14728–14737, 2024.

[31] J. Kim, J. Woo, U. Shin, J. Oh, and S. Im, “Flow4D: Leveraging 4d voxel network for lidar scene flow estimation,” IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 3462–3469, 2025.

[32] A. Khoche, Q. Zhang, L. P. Sanchez, A. Asefaw, S. S. Mansouri, and P. Jensfelt, “SSF: Sparse long-range scene flow for autonomous driving,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 6394–6400, IEEE, 2025.

[33] Y. Liu, X. Chen, N. Wang, S. Andreev, A. Dvorkovich, R. Fan, and H. Lu, “Self-supervised diffusion-based scene flow estimation and motion segmentation with 4d radar,” IEEE Robotics and Automation Letters, vol. 10, no. 6, pp. 5895–5902, 2025.

[34] Z. Yi, F. Neumann, G. von Wichert, and D. Burschka, “Moving object segmentation via 3d lidar data: A learning-free real-time online alternative,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 12502–12509, IEEE, 2025.

[35] L. Goli, S. Sabour, M. Matthews, M. A. Brubaker, D. Lagun, A. Jacobson, D. J. Fleet, S. Saxena, and A. Tagliasacchi, “RoMo: Robust motion segmentation improves structure from motion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6155–6164, 2025.

[36] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao, “Gmflow: Learning optical flow via global matching,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8121–8130, 2022.

[37] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[38] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and Q. Tian, “Fast dynamic radiance fields with time-aware neural voxels,” in SIGGRAPH Asia 2022 Conference Papers, pp. 1–9, 2022.

[39] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in European conference on computer vision, pp. 402– 419, Springer, 2020.

[40] L. Kong and J. Yang, “Mdflow: Unsupervised optical flow learning by reliable mutual knowledge distillation,” IEEE Transactions on circuits and systems for video technology, vol. 33, no. 2, pp. 677–688, 2022.