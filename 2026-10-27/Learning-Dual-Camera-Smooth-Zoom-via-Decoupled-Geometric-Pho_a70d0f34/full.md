# Learning Dual-Camera Smooth Zoom via Decoupled Geometric-Photometric Data Synthesis

Junjie Fan, Zhilu Zhang, Renlong Wu, Dazhi Zhang and Wangmeng Zuo, Senior Member, IEEE

Abstract—When zooming between dual cameras on smartphones, prominent discontinuities in both geometric perspective and color rendition inevitably degrade the visual experience of users. To address this issue, we introduce the dual-camera smooth zoom (DCSZ) task, aiming to synthesize continuous intermediate frames for fluid zooming. However, directly applying existing video frame interpolation (VFI) models to this task is severely hindered by substantial domain gaps and the practical intractability of capturing real-world continuous ground-truth. To this end, we propose a novel data synthesis framework to generate high-fidelity DCSZ training sequences, named ZoomGS v2. It explicitly decouples geometric and color transformations from 3D scene representation modeling, thereby enabling extensive data generation from easily accessible single-camera 3D scenes. Furthermore, we develop ZoomFI, a specialized VFI network tailored to the DCSZ task, which leverages complementary bidirectional warping for comprehensive dual-camera information fusion. Extensive experiments on test datasets captured using two mobile phones demonstrate that training with the constructed DCSZ data significantly improves the performance of various VFI models. Moreover, ZoomFI achieves state-ofthe-art results in both quantitative metrics and visual quality. The datasets, code, and pretrained models are available at https://hitjunjief.github.io/DCSZ-v2.

Index Terms—Dual-Camera Smooth Zoom, Video Frame Interpolation, Data Synthesis, 3D Reconstruction.

## I. INTRODUCTION

D <sup>UE</sup> <sup>to</sup> <sup>physical</sup> <sup>space</sup> <sup>constraints,</sup> <sup>modern</sup> <sup>smartphones</sup>predominantly deploy multiple fixed-focal-length cam- predominantly deploy multiple fixed-focal-length cameras rather than variable-focal-length lenses for digital zoom [4]. In particular, dual-camera configurations are commonly deployed, typically pairing a wide-angle (W, e.g., ×1.0) lens for primary imaging with an ultra-wide-angle (UW, e.g., ×0.5 or ×0.6) lens to provide an expanded field of view. For zooming between the UW and W cameras, commercial smartphones (e.g., manufactured by Xiaomi, Huawei, Oppo, and Vivo) conventionally crop the central region of the UW image and digitally scale it up to the target resolution, as illustrated in Fig. 1(a). However, once the zoom magnification reaches the transition threshold (e.g., switching from ×0.9 to ×1.0), the system abruptly switches from the UW camera to the W camera. Due to the distinct deployment locations and internal characteristics [1], [2], [84] of the two sensors, this switch induces prominent preview discontinuities, manifested as abrupt shifts in both geometric perspective and color rendition (see Fig. 1(c)). These discontinuities substantially degrade the user experience. To overcome this limitation, the task of Dual-Camera Smooth Zoom (DCSZ) has emerged, which aims to synthesize continuous frames between dual cameras to deliver a fluid zoom preview on mobile phones, as shown in Fig. 1(b).

Video frame interpolation (VFI) techniques [20], [35], [42], [66], [85] offer a potential remedy by synthesizing intermediate frames between the endpoint views of dual cameras. However, directly applying off-the-shelf VFI models often fails to produce satisfactory results. It stems from the substantial domain gap between standard VFI benchmarks and dual-camera zoom scenarios that feature larger viewpoint displacements and severer color variations. A more principled approach is to collect specialized DCSZ datasets to retrain or fine-tune interpolation networks. Unfortunately, capturing real-world continuous ground-truth sequences is physically intractable, as hardware constraints preclude deploying a dense array of sensors between the dual cameras.

To address this issue, we propose to assemble continuous virtual cameras and render intermediate views from a 3D scene to get DCSZ data, rather than relying on direct physical capture. The formulation of such a virtual (V) camera requires accurate modeling of both geometric parameters (extrinsics and intrinsics) and photometric characteristics (e.g., the image signal processing (ISP) pipeline). While extrinsic and intrinsic parameters define the perspective projection from 3D world coordinates to the 2D image plane, the ISP pipeline dictates the color rendering and tonal characteristics of the image. Assuming that all parameters of both UW and W cameras are calibrated, virtual camera parameters can be obtained via smooth interpolation, allowing continuous views to be rendered from the underlying 3D scene.

In our preliminary conference work, ZoomGS [72], an initial step was taken toward virtual-camera-based DCSZ data synthesis. ZoomGS models the scene with 3D Gaussian Splatting (3DGS) [28] and uses a multi-layer perceptron (MLP)- based camera transition (CamTrans) module to learn spatial and color transitions between the two endpoint 3DGS. Nevertheless, ZoomGS suffers from two major limitations. First, ZoomGS lacks intermediate-view supervision. This results in insufficient constraints on the CamTrans module that couples color and position, leading to temporal instability and visual jitter in the synthesized sequences. Second, ZoomGS mandates dual-camera captures for the same scene, creating a data acquisition bottleneck that limits the scalability of dataset construction.

To overcome these challenges, we propose ZoomGS v2, a novel and scalable data synthesis pipeline designed for high-fidelity DCSZ sequence generation. Unlike ZoomGS, ZoomGS v2 explicitly decouples geometric and color transformations from 3D scene representation modeling. Specifically, the framework separately estimates the relative camera pose and photometric discrepancy between the dual cameras, smoothly interpolates these attributes, and applies them during 3D scene rendering. Crucially, benefiting from decoupling design, the underlying 3D scene representation to be reconstructed from standard single-camera multi-view data, which substantially improves data scalability. Furthermore, previous VFI architectures struggle with the asymmetric correspondences and large geometric displacements between dualcamera views. To address this issue, we suggest ZoomFI, a VFI model specifically designed for the DCSZ task. It leverages complementary bidirectional warping for comprehensive dual-camera information fusion.

![](images/4f5409bfeb6e96e613283d5ca86d06cb7f468a2aea3b2500e83a814c15c4bc13.jpg)  
Fig. 1: (a) Existing dual-camera digital zoom. For zoom transitions between the UW and W cameras (i.e., from ×0.6 to ×1.0), commercial smartphones generally crop the corresponding region from the UW image and resize it to the target dimensions. When the zoom factor transitions from ×0.9 to ×1.0, the active lens switches abruptly from UW to W, inducing severe geometric perspective and color rendition discontinuities in the preview. (b) The proposed dual-camera smooth zoom (DCSZ) pipeline. (c) Visual illustration of geometric and color rendition discontinuities in conventional dual-camera zoom previews.

To validate our approach, we collect real-world dual-camera image pairs from two mobile phones (Xiaomi and Huawei) for evaluation. Extensive experiments across multiple VFI backbones demonstrate that models fine-tuned on data synthesized by ZoomGS v2 achieve substantial quantitative and qualitative improvements over baseline models as well as ZoomGS, validating the effectiveness of the proposed introduced data synthesis pipeline. Moreover, the proposed ZoomFI consistently outperforms state-of-the-art VFI models on the evaluated DCSZ benchmarks.

The contributions can be summarized as follows:

• We explore a practical task, i.e., dual-camera smooth zoom (DCSZ), and propose a solution that synthesizes continuous zoom sequences to empower video frame interpolation (VFI) models for smooth zoom previews.

• We propose ZoomGS v2, a high-fidelity and scalable data synthesis framework that explicitly decouples geometric and color transformations from 3D scene representation modeling.

• We collect real-world dual-camera image pairs of two smartphones for evaluation. Extensive experiments validate the superiority of the proposed ZoomGS v2 and VFI model over existing state-of-the-art ones.

## II. RELATED WORK

## A. Frame Interpolation

Given an image pair, video frame interpolation (VFI) aims to generate intermediate frames, which can be roughly divided into kernel-based methods [26], [54], [86], hallucination-based methods [26], [29], [55], flow-based methods [20], [23], [25], [32], [35], [42], [47], [54], [66], [82], [85], and state space model (SSM)-based [81]. For example, RIFE [23] and IFR-Net [32] incorporate task-oriented flow distillation objectives to generate intermediate flow prior during training. AMT [35] builds bidirectional correlation volumes for all pairs of pixels to retrieve correlations for more accurate flow estimation. VFIMamba [81] uses SSMs for multi-directional serialization of inter-frame spatiotemporal modeling and inter-frame features for generating intermediate frames. In addition, some works [42] introduce vision transformer [13] for performance improvement. However, the domain gap between DCSZ data and their training datasets limits their effective application for the DCSZ task. Moreover, compared to traditional interpolation, DCSZ interpolation has a larger amplitude of inter-frame motion, making it difficult for these traditional interpolation models to accurately model inter-frame motion.

## B. 3D Reconstruction

With the development of deep learning, neural rendering algorithms [3], [45], [46], [51], [76] have outperformed traditional methods [21], [58] in 3D reconstruction. Neural Radiance Field (NeRF) [45] pioneers this field and facilitates high-quality novel view synthesis with the use of MLPs. Advancements have been made to handle multi-resolution inputs [3], sparse-view image inputs [76], dynamic scenes [51], and nearly real-time rendering capabilities [46]. In addition, a few works [60], [60], [78] exploit the 3D priors in NeRF models to construct data for downstream tasks, such as stereo depth estimation [60], object detection [19], and object dense description [78]. However, the NeRF-based 3D reconstruction methods struggle to achieve visually realistic rendering quality and real-time rendering speed simultaneously, due to the implicit scene representation and the dense sampling of rays.

In contrast, the recent 3D Gaussian Splatting (3DGS) [28] overcomes the issue with explicit scene representation and differentiable rasterization, attracting substantial attention [9]. For example, some works utilize it for 3D reconstruction in sparse-view scenarios [6], [7], [12], [59], [70], [73], [75], [83], [87], [88], dynamic scenarios [34], [39], [67], [67], [71], and COLMAP-free scenarios [10], [17], [18], [65]. Some works employ it in segmentation [5], [22], [24], [33], visual generation [11], [36], [44], [48], [52] and editing [14], [16], [61], [62], [89] tasks. Besides, many efforts have been made in antialiased rendering [74], [79] and storage efficiency [15], [49], [80]. In this work, further considering the specific imaging characteristics for each virtual camera, we enhance 3DGS [28] with the ability to specialize the 3D model for a specific camera, thus generating data for the DCSZ task.

## III. PROPOSED PROBLEM AND METHOD

In this section, we first formulate the problem in Sec. III-A and present the motivation of our ZoomGS v2 method in Sec. III-B. We then introduce the proposed ZoomGS v2 in Sec. III-C, Sec. III-D and Sec. III-E. Finally, we detail the proposed ZoomFI and provide the training details in Sec. III-F.

## A. Problem DCSZ Setup

As illustrated in Fig. 1(c), the distinct deployment locations and internal characteristics [1], [2], [84] of dual-zoom cameras lead to prominent preview discontinuities in geometric perspective and color rendition, which significantly affect the user’s zoom experience. The proposed dual-camera smooth zoom (DCSZ) task aims to achieve a fluid zoom preview, which can be accomplished by employing a video frame interpolation (VFI) model to synthesize N intermediate images $\{ \tilde { \mathbf { I } } _ { i } \} _ { i = } ^ { \tilde { N } }$ between dual-camera images. Denoted by $\mathbf { I } _ { u w }$ and $\mathbf { I } _ { w }$ the UW and W images respectively, the VFI model can be written as:

$$
\tilde { \mathbf { I } } _ { i } = \mathcal { T } ( \mathbf { I } _ { u w } , \mathbf { I } _ { w } , t _ { i } ; \boldsymbol { \Theta } _ { \mathcal { F } } ) ,\tag{1}
$$

where $\Theta _ { \mathcal { F } }$ denotes the parameters of the VFI model $\mathcal { F } , t _ { i }$ is the temporal encoding that indicates the relative relationship between the synthesized image and the dual-camera images. Due to the substantial domain gap between standard VFI benchmarks and dual-camera zoom scenarios, existing pretrained VFI models often fail to meet expectations. Retraining or fine-tuning the model is a more appropriate solution, but it is impractical to capture the continuous ground-truth sequences by deploying a dense array of sensors between the dual cameras.

To address the issue, we suggest assembling continuous virtual cameras to construct such zoom sequences. For each dual-camera image pair, we construct a set of N virtual cameras $\{ v _ { i } \} _ { i = 1 } ^ { N }$ along the transition between the UW and W cameras. Thus, synthetic ground-truth $\{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N }$ can be rendered by these virtual cameras. By performing the steps on multiple image pairs, a synthetic dataset for the DCSZ task can be constructed, thus being employed to fine-tune a VFI model, $i . e .$

$$
\Theta _ { \mathcal { T } } ^ { * } = \arg \operatorname* { m i n } _ { \Theta _ { \mathcal { F } } } \mathcal { L } _ { V F I } \left( \mathcal { T } ( \mathbf { I } _ { u w } , \mathbf { I } _ { w } , t _ { i } ; \Theta _ { \mathcal { F } } ) , \mathbf { I } _ { i } \right) ,\tag{2}
$$

where $\mathcal { L } _ { V F I }$ denotes the learning objective during training.

To realize this, we construct virtual cameras comprising two key components. The first component consists of the camera extrinsic and intrinsic parameters, as they determine the perspective projection from 3D world coordinates to the 2D image plane. The second component comprises the ISP pipeline parameters, which influence the color rendering and tonal characteristics of the image. Assuming that all parameters of UW and W cameras have been calibrated, virtual camera parameters can be obtained through smooth interpolation to render continuous views from the underlying 3D scene.

## B. Motivation of Methods

Our preliminary conference work, ZoomGS [72], attempted to address these issues by constructing 3D models using 3D Gaussian Splatting (3DGS) [28]. ZoomGS represents the 3D scenes of the UW and W cameras with 3DGS, and then utilizes an MLP-based camera transition (CamTrans) module to learn the position and color transformations of 3DGS representations between two endpoints, ultimately interpolating them through CamTrans to obtain virtual camera representations, as illustrated in Fig. 2 (a). However, ZoomGS lacks intermediate-view supervision. This results in insufficient constraints on the CamTrans module that couples color and position together, leading to discontinuous intermediate sequences (see Fig. 2(a)). Moreover, ZoomGS requires dualcamera data from the same scene, rendering scalable dataset expansion prohibitively costly, as existing open-source 3D Gaussian training datasets are captured by a single camera.

These observations motivate us to disentangle scene representation learning from device-specific camera transformation modeling. To this end, we propose a novel and scalable synthesis pipeline termed ZoomGS v2 (see Fig. 2(b)). Specifically, ZoomGS v2 first calibrates the geometric and color transformation gap between dual cameras from a small amount of dual-camera calibration data. Geometric transformation is modeled by Gaussian distribution statistics of dual-camera parameter gap, and color transformation is modeled by image signal processing (ISP) pipeline with a network introduced and trained to predict these parameters, as shown in Fig. 3. Then, ZoomGS v2 transfers the differences to large-scale single-camera scenes, and simultaneously models 3DGS scene with single-camera data, instead of jointly optimizing scene representation and cross-camera transformation from sparse dual-camera observations. By decoupling scene representation learning from cross-camera transformation modeling, ZoomGS v2 avoids the lack of intermediate-view supervision in ZoomGS and enables scalable synthesis of DCSZ training data from large-scale single-camera datasets.

We divide the introduction of ZoomGS v2 into three parts: Geometric Transformation Calibration of Dual-camera in Sec. III-C, Color Transformation Prediction of Dual-camera in Sec. III-D, and Data Generation from ZoomGS v2 in

![](images/f7287c9af112b88ef4259beaa807ebd80be17c84f0a7de6ce3074124803d1a29.jpg)  
(b) Overall Pipeline and Results of ZoomGS v2  
Fig. 2: Comparison of ZoomGS [72] and ZoomGS v2. (a) Overall Pipeline and Results of ZoomGS. ZoomGS models the scene with Base Gaussians and uses the CamTrans Module to learn spatial and color transitions between the two endpoint 3DGS [28] representations, and then transforms the Base Gaussians to target Gaussians of virtual cameras for rendering. (b) Overall Pipeline and Results of ZoomGS v2. ZoomGS v2 first calibrates the distribution of camera and color parameters from a small set of dual-camera data, and then applies them to Gaussians constructed from large-scale single-camera data to generate 3DGS representations for scene rendering.

Sec. III-E. In addition, to address the asymmetric correspondences and large geometric displacements between dualcamera views faced by VFI models in the DCSZ task, we design an effective VFI model tailored for the DCSZ task, termed ZoomFI, which incorporates both forward and backward warping to produce more photorealistic zoom transitions, as detailed in Sec. III-F.

## C. Geometric Transformation Calibration of Dual-camera

The geometric disparity between the UW and W images can be primarily reflected by their camera parameters, including both intrinsic and extrinsic parameters. Since 3DGS [28] enables rendering directly from camera parameters, this disparity can be characterized by analyzing the variations in these parameters. The camera extrinsic parameters $\mathbf { P } = \left[ \mathbf { R } \mid \mathbf { T } \right]$ consist of a rotation matrix $\textbf { R } \in \ \mathbb { R } ^ { 3 \times 3 }$ and a translation vector $\textbf { T } \in \ \mathbb { R } ^ { 3 \times 1 }$ , which encode the camera’s orientation and position in world coordinates, respectively. The intrinsic parameters include the focal length, which determines the field of view (FOV) of the captured images. For clarity, in the following sections, we denote a pair of dual-camera images as $\mathbf { I } _ { u w }$ and ${ \bf { I } } _ { w }$ , the corresponding extrinsic parameters as $\mathbf { P } _ { u w } = [ \mathbf { R } _ { u w } | \mathbf { T } _ { u w } ]$ and $\mathbf { P } _ { w } = [ \mathbf { R } _ { w } | \mathbf { T } _ { w } ]$ , and their intrinsic parameters as ${ \bf K } _ { u w }$ and $\mathbf { K } _ { w } .$ , respectively.

Camera intrinsics are primarily determined by the focal length. For a smartphone, the effective focal length ratio (or scaling factor) between the UW and W cameras is a fixed constant $s ,$ yielding a constant relative intrinsic disparity. This relationship can be formulated as, ${ \bf K } _ { w } = { \bf K } _ { u w } / s$

![](images/b119c080affb657d36d7a61917ce688ed60f2231441b0653cbdeaa7352abb374.jpg)  
Fig. 3: (a) Calibration of Camera Parameter Distribution. ZoomGS v2 collects multiple camera parameter offsets $\Delta { \bf V }$ (Eq. (3)) from UW-W pairs and models their distribution using a Gaussian distribution. (b) Calibration of Color Parameter Distribution. ZoomGS v2 trains a CNN-based [30] color prediction module on UW-W pairs to predict the parameters of the color transformation pipeline from UW to W.

To formulate the spatial relationship between the translation vectors $\mathbf { T } _ { u w }$ and $\mathbf { T } _ { w }$ of the dual cameras, we introduce a relative spatial displacement vector ∆V. Without loss of generality, the upright orientation of the mobile device is established as the reference coordinate frame. Owing to the rigid physical mounting of the dual-camera module on the mobile chassis, the relative spatial displacement between the two optical centers remains invariant within the coordinate frame of the device. Accordingly, the relative displacement vector $\Delta { \bf V }$ defined in the world coordinate system is formulated as:

![](images/d4d5e8a409130e7d1997e229442771c8dfe854a281cfecfe871533982c3ece7a.jpg)  
(a) Distribution of ∆V for Xiaomi smartphone

![](images/6ae00a20b20fc98cf5b188fafd01df2c40f8be3452e011adc9f4034af548cc5c.jpg)  
(b) Distribution of ∆V for Huawei smartphone  
Fig. 4: Statistical distributions and Gaussian fitting results of $\Delta \mathbf { V } ,$ which represents the relative vector offset between the UW and W images captured by Xiaomi (a) and Huawei (b) model smartphones. From left to right, the plots represent $\Delta \mathbf { V } _ { x } , \Delta \mathbf { V } _ { y }$ and $\Delta \mathbf { V } _ { z }$ , respectively.

$$
\Delta \mathbf { V } = \mathbf { R } _ { w } ^ { \mathsf { T } } \mathbf { T } _ { w } - \mathbf { R } _ { u w } ^ { \mathsf { T } } \mathbf { T } _ { u w } .\tag{3}
$$

Regarding the rotation matrices $\mathbf { R } _ { u w }$ and $\mathbf { R } _ { w } ,$ because the dual sensors are rigidly fixed to the same planar substrate, their relative orientation discrepancy is marginal in practical scenarios. Empirical statistical analysis on our collected dualcamera calibration dataset reveals that the mean absolute error (MAE) between the rotation parameters of the UW and W cameras is less than $3 . 5 \times 1 0 ^ { - \bar { 3 } }$ . Consequently, we reasonably approximate the orientations of both cameras as identical during data synthesis, $i . e . , \mathbf { R } _ { w } \approx \mathbf { R } _ { u w }$

To determine the empirical distribution of the relative displacement vectors $\Delta { \bf V }$ , we utilize a diverse set of calibrated dual-camera image pairs along with their estimated camera extrinsics. Theoretically, the relative vector offset ∆V should remain deterministic for a specific mobile device model. In practice, however, perturbations induced by subtle camera shake, sensor thermal variations, and numerical ambiguities inherent to Structure-from-Motion (SfM) estimation in COLMAP [53] introduce stochastic fluctuations into the calculated values of $\Delta { \bf V }$ . To account for this stochasticity, we model $\Delta { \bf V }$ as a multivariate Gaussian distribution fitted over the empirical measurements, where each coordinate component of the translation vector comprises a deterministic baseline offset perturbed by Gaussian noise (see Fig. 4).

Thus, given an arbitrary single-camera UW view along with its extrinsic parameters $\mathbf { R } _ { u w }$ and $\mathbf { T } _ { u w } ,$ a relative displacement vector $\Delta { \bf V }$ can be randomly sampled from the fitted multivariate Gaussian distribution. Under the approximation $\mathbf { R } _ { w } ~ \approx ~ \mathbf { R } _ { u w }$ , the corresponding translation vector $\mathbf { T } _ { w }$ of the W camera is analytically derived via Eq. (3) as $\mathbf { T } _ { w }$ ≈ $\mathbf { R } _ { w } ( \Delta \mathbf { V } + \mathbf { R } _ { u w } ^ { \mathsf { T } } \mathbf { T } _ { u w } )$ . In summary, the extrinsic parameters of the W camera can be systematically derived from those of the UW camera through the proposed decoupled geometric formulation.

## D. Color Transformation Prediction of Dual-camera

To model the photometric disparity between the dual cameras, we leverage the representative image signal processing (ISP) pipeline of mobile devices to formulate the color transformation between the RGB images $\mathbf { I } _ { u w } , \mathbf { I } _ { w }$ and their underlying linear RAW representations $\mathbf { X } _ { u w } , \mathbf { X } _ { w }$ , as illustrated in Fig. 3(b). Conceptually, mapping the color distribution of $\mathbf { I } _ { u w }$ to that of $\mathbf { I } _ { w }$ involves a tripartite workflow: first inverting $\mathbf { I } _ { u w }$ into the linear RAW domain $\mathbf { X } _ { u w }$ via an inverse ISP, subsequently mapping $\mathbf { X } _ { u w }$ to $\mathbf { X } _ { w }$ through a linear transformation, and finally rendering $\mathbf { X } _ { w }$ into the RGB domain $\mathbf { I } _ { w }$ via a forward ISP.

Specifically, a linear transformation parameterized by a weight matrix $\mathbf { W } _ { r a w } \in \mathbb { R } ^ { 3 \times 3 }$ and a bias vector $\mathbf { b } _ { r a w } \in \mathbb { R } ^ { \mathbf { \bar { 3 } } \times 1 }$ is initially employed to characterize the cross-sensor spectral discrepancy from $\mathbf { X } _ { u w }$ to $\mathbf { X } _ { w }$ . We then simplify the ISP pipeline into three primary stages governing color rendition, including color correction (CC), auto white balance (AWB), and gamma correction. We define full matrices $\mathbf { C } _ { u w } , \mathbf { C } _ { w } \in$ $\mathbb { R } ^ { 3 \times \bar { 3 } }$ to model the CC transformations, and diagonal matrices $\mathbf { D } _ { u w } , \mathbf { D } _ { w } \in \mathbb { R } ^ { 3 \times 3 }$ to characterize their respective AWB gains of the UW and W cameras, respectively. The non-linear gamma corrections in the two ISP pipelines are formulated channel-wise as:

$$
\mathbf { I } _ { u w , c } = \frac { ( \alpha _ { u w , c } \mathbf { I } _ { u w , c } ^ { a w b } + \beta _ { u w , c } ) ^ { \gamma _ { u w , c } } - ( \beta _ { u w , c } ) ^ { \gamma _ { u w , c } } } { ( \alpha _ { u w , c } + \beta _ { u w , c } ) ^ { \gamma _ { u w , c } } - ( \beta _ { u w , c } ) ^ { \gamma _ { u w , c } } } ,\tag{4}
$$

$$
\mathbf { I } _ { w , c } = \frac { ( \alpha _ { w , c } \mathbf { I } _ { w , c } ^ { a w b } + \beta _ { w , c } ) ^ { \gamma _ { w , c } } - ( \beta _ { w , c } ) ^ { \gamma _ { w , c } } } { ( \alpha _ { w , c } + \beta _ { w , c } ) ^ { \gamma _ { w , c } } - ( \beta _ { w , c } ) ^ { \gamma _ { w , c } } } ,\tag{5}
$$

where $c \in \{ r , g , b \}$ denotes the color channel, $\{ \alpha _ { u w , c } , \beta _ { u w , c } , \gamma _ { u w , c } \}$ and $\{ \alpha _ { w , c } , \beta _ { w , c } , \gamma _ { w , c } \}$ are parameters to be determined, and $\mathbf { I } _ { u w } ^ { a w b } , \mathbf { I } _ { w } ^ { a w b }$ represent the intermediate representations obtained after applying CC and AWB to $\mathbf { X } _ { u w }$ and $\mathbf { X } _ { w } ,$ respectively. Consequently, the intermediate linear operations can be algebraically unified. By defining $\begin{array} { r l r } { \mathbf { \bar { W } } } & { = } & { { \bf D } _ { w } { \bf C } _ { w } { \bf W } _ { r a w } \bar { \bf C } _ { u w } ^ { - 1 } { \bf D } _ { u w } ^ { - 1 } \quad \in \quad \mathbb { R } ^ { 3 \times 3 } } \end{array}$ and $ { \mathbf { b } } =  { \mathbf { D } } _ { w }  { \mathbf { C } } _ { w }  { \mathbf { b } } _ { r a w } \in \mathbb { R } ^ { 3 \times 1 }$ , the color transition from $\mathbf { I } _ { u w } ^ { a w b }$ to I<sup>awb</sup><sub>w</sub> is jointly expressed as a compact affine mapping, $i . e . ,$ ${ \bf { I } } _ { w } ^ { \tilde { a } w b } = { \bf { \bar { W } } } { \bf { I } } _ { u w } ^ { a \tilde { w } b } + { \bf { \bar { b } } }$

Because these physical parameters are inaccessible from closed-source mobile imaging systems, we design a learningbased estimation strategy. We collect a diverse set of realworld UW and W image pairs and establish geometric alignment for each pair via field-of-view cropping followed by optical-flow-based warping [56]. A convolutional neural network (CNN) [30] is then trained on these aligned pairs to directly predict the transformation parameters. The optimization is supervised by a color difference loss based on $\Delta \mathrm { E }$ [43], which projects the predicted and ground-truth images into the perceptually uniform CIELAB color space and computes their $\mathrm { L _ { 2 } }$ Euclidean distance. During inference, the estimated photometric parameters are applied to transform the color profile of the UW view into that of the W camera.

![](images/aa85fe16ca4654c2d0a0e841fd8e295671253c59b900927a9264faf05519b595.jpg)  
Fig. 5: Comparison of forward and backward warping for transforming UW to W (Left) and transforming W to UW (Right) using SEA-RAFT [64].

## E. Data Generation from ZoomGS v2

To synthesize continuous dual-camera zoom sequences, we curate diverse static scenes from DL3DV-10K [37] and reconstruct each scene as a 3DGS representation. For each reconstructed scene, ZoomGS v2 first randomly samples an original capture to define the reference UW view, with corresponding intrinsic matrix ${ \bf K } _ { u w }$ and extrinsic matrix $\mathbf { P } _ { u w } ~ = ~ [ \mathbf { R } _ { u w }$ $\mathbf { T } _ { u w } ]$ . The target W camera intrinsics ${ \bf K } _ { w }$ and orientation $\mathbf { R } _ { w }$ are obtained according to Sec. III-C, while the translation vector $\mathbf { T } _ { w }$ is determined by:

$$
{ \bf T } _ { w } = { \bf T } _ { u w } + { \bf R } _ { w } \Delta { \bf V } ,\tag{6}
$$

where $\Delta \mathbf { V } ~ \in ~ \mathbb { R } ^ { 3 \times 1 }$ is drawn from the fitted multivariate Gaussian distribution.

To simulate smooth camera movement between the two endpoints, we define a normalized progression factor $\tau _ { i } =$ $\frac { i - \bar { 1 } } { N - 1 } \in [ 0 , 1 ]$ for the i-th virtual camera view $( i \in \{ 1 , \ldots , N \} )$ Following the continuous trajectory modeling in [40], [63], the intermediate extrinsic parameters ${ \bf P } _ { v _ { i } } \in \mathrm { S E } ( 3 )$ are computed via geodesic interpolation in the Lie algebra of se(3):

$$
\mathbf { P } _ { v _ { i } } = \mathbf { P } _ { u w } \exp \left( \tau _ { i } \log \left( \mathbf { P } _ { w } \mathbf { P } _ { u w } ^ { - 1 } \right) \right) ,\tag{7}
$$

where exp(·) and log(·) denote the matrix exponential and logarithmic mappings between the Lie group SE(3) and its Lie algebra $\mathfrak { s e } ( 3 )$ , respectively. Simultaneously, the intrinsic parameters are smoothly interpolated as:

$$
{ \bf K } _ { v _ { i } } = ( 1 - \tau _ { i } ) { \bf K } _ { u w } + \tau _ { i } { \bf K } _ { w } .\tag{8}
$$

With the interpolated camera parameters $( \mathbf { K } _ { v _ { i } } , \mathbf { P } _ { v _ { i } } )$ , the intermediate geometric view $\mathbf { I } _ { v _ { i } }$ is rendered from the reconstructed 3DGS representation. Finally, the photometric transformation parameters predicted by the CNN [30] model are linearly modulated by $\tau _ { i }$ and applied to $\mathbf { I } _ { v _ { i } }$ via the formulation in Sec. III-D, thereby generating photometrically and geometrically continuous zoom sequences.

![](images/981b3cc8e6047762c367a114c340f4a03c81031342cb31d4fc2a70d976674060.jpg)  
Fig. 6: Overview pipeline of ZoomFI. The bidirectional warping is used for dual-camera information fusion.

## F. Pipeline and Training of ZoomFI

The DCSZ task departs substantially from conventional video frame interpolation (VFI) in that it entails significantly larger motion amplitudes, while the field of view (FoV) of the UW camera strictly subsumes that of the W camera. Conventional flow-based VFI architectures (e.g., RIFE [23] and AMT [35]) typically estimate backward optical flows from the intermediate target frame to the input frames, backward warp the input images toward the target time step, and fuse the aligned representations to generate the intermediate view. However, because backward warping executes target-oriented pixel sampling, the substantial scale disparity and large spatial displacement between dual-camera views often induce sampling ambiguities and severe peripheral distortions when backward warping the broad-FoV UW view toward the localized W view (see Fig. 5). To address this limitation, we avoid relying exclusively on backward warping and instead synergistically couple forward warping with backward warping. Because forward warping operates along the inverse sampling trajectory, it inherently accommodates scale expansion and preserves peripheral visual content, thereby providing information complementary to backward warping for DCSZ tasks. Furthermore, a specialized optical flow estimation network is integrated to robustly handle the substantial scale discrepancy between the dual-camera views.

Specifically, we utilize a pre-trained optical flow network O (e.g., SEA-RAFT [64]) to compute bidirectional flow fields between the input images $\mathbf { I } _ { u w }$ and $\mathbf { I } _ { w }$ at a 4× downsampled resolution:

$$
\mathbf { F } _ { u w  w } = \mathcal { O } ( \mathbf { I } _ { u w } , \mathbf { I } _ { w } ) , \quad \mathbf { F } _ { w  u w } = \mathcal { O } ( \mathbf { I } _ { w } , \mathbf { I } _ { u w } ) .\tag{9}
$$

Following the flow approximation scheme in SGM-VFI [38], intermediate backward flows $\mathbf { F } _ { i  u w }$ and $\mathbf { F } _ { i  w }$ are approximated via forward warping, while intermediate forward flows $\mathbf { F } _ { u w  i }$ and $\mathbf { F } _ { w  i }$ are derived through linear trajectory interpolation, providing a coarse motion initialization:

$$
\mathbf { F } _ { i  u w } = \overrightarrow { \mathcal { W } } \big ( \tau _ { i } \mathbf { F } _ { w  u w } , ( 1 - \tau _ { i } ) \mathbf { F } _ { w  u w } \big ) ,\tag{10a}
$$

$$
{ \mathbf { F } } _ { i  w } = \overrightarrow { \mathcal { W } } \big ( ( 1 - \tau _ { i } ) { \mathbf { F } } _ { u w  w } , \tau _ { i } { \mathbf { F } } _ { u w  w } \big ) ,\tag{10b}
$$

$$
{ \bf F } _ { u w  i } = \tau _ { i } { \bf F } _ { u w  w } , \quad { \bf F } _ { w  i } = ( 1 - \tau _ { i } ) { \bf F } _ { w  u w } ,\tag{10c}
$$

where $\vec { \mathcal { W } } ( \mathbf { X } , \mathbf { F } )$ denotes the forward warping operator that splats tensor X guided by the flow field F.

The overview pipeline of ZoomFI can be seen in Fig. 6. The backward flows $\mathbf { F } _ { i  u w }$ and $\mathbf { F } _ { i  w }$ are employed to backward warp $\mathbf { I } _ { u w }$ and ${ \bf { I } } _ { w }$ to the intermediate state i, yielding warped frames $\mathbf { W } _ { i  u w }$ and $\mathbf { W } _ { i  w } ,$ respectively. Simultaneously, the forward flows $\mathbf { F } _ { u w  i }$ and $\mathbf { F } _ { w  i }$ forward warp $\mathbf { I } _ { u w }$ and $\mathbf { I } _ { w }$ to state i, yielding $\mathbf { W } _ { u w  i }$ and $\mathbf { W } _ { w \to i }$ . Subsequently, the aligned feature tensors $\{ \mathbf { W } _ { i  u w } , \mathbf { W } _ { i  w } , \mathbf { W } _ { u w  i } , \mathbf { W } _ { w  i } \}$ and the intermediate flow fields $\{ \mathbf { F } _ { i  u w } , \mathbf { F } _ { i  w } , \mathbf { F } _ { u w  i } , \mathbf { F } _ { w  i } \}$ are concatenated along the channel dimension and fed into a multi-scale fusion module to synthesize the final intermediate frame ${ \tilde { \mathbf { I } } } _ { i } .$ The fusion module consists of initial convolutional feature extraction layers followed by a multi-scale U-Net architecture.

Furthermore, to facilitate real-time deployment on resourceconstrained mobile hardware, we introduce an efficient variant termed ZoomFI-Lite. By executing optical flow estimation at an 8× downsampling factor, ZoomFI-Lite achieves accelerated inference throughput and a reduction in floatingpoint operations (FLOPs) compared with the standard ZoomFI architecture.

To optimize the proposed interpolation framework, we employ a joint loss combining a multi-scale Laplacian pyramid reconstruction loss and a perceptual loss. Following established VFI methodologies [23], [47], the reconstruction loss is computed over the Laplacian pyramid representations of the predicted frame <sup>˜</sup>I<sub>i</sub> and the ground-truth frame I<sub>i</sub>. The overall optimization objective ${ \mathcal { L } } _ { \mathrm { V F I } }$ is formulated as:

$$
\mathcal { L } _ { \mathrm { V F I } } = \mathcal { L } _ { 1 } \big ( \mathrm { L P } ( \tilde { \mathbf { I } } _ { i } ) , \mathrm { L P } ( \mathbf { I } _ { i } ) \big ) + \lambda _ { \mathrm { p e r } } \mathcal { L } _ { 1 } \big ( \phi ( \tilde { \mathbf { I } } _ { i } ) , \phi ( \mathbf { I } _ { i } ) \big ) ,\tag{11}
$$

where $\lambda _ { \mathrm { p e r } }$ is a hyperparameter set to 1.0, $\operatorname { L P } ( \cdot )$ designates the Laplacian pyramid decomposition, and $\phi ( \cdot )$ denotes the features extracted from a pre-trained VGG-19 network [57].

## IV. EXPERIMENTS

To evaluate the generalization capability of our method across different devices and zooming paths from UW (zoom factor ×0.6) to W (zoom factor ×1.0), we conduct experiments using two smartphones from Xiaomi and Huawei, respectively. And we consider two zooming settings: $\times 0 . 6 $ $\times 1 . 0$ and $\times 0 . 8 5  \times 1 . 0$ In the $\times 0 . 6  \times 1 . 0$ setting, we directly apply continuous geometric and color transformations from the UW view to the W view. In the $\times 0 . 8 5  \times 1 . 0$ setting, we first apply center cropping to the UW image to obtain an intermediate view with a zoom factor close to that of the W view $( i . e . , \times 0 . 8 5 )$ , and then apply continuous geometric and color transformations from this intermediate view to the W view. We construct training and test datasets for both devices in the $\times 0 . 6  \times 1 . 0$ setting. For the $\times 0 . 8 5  \times 1 . 0$ setting, we derive the corresponding datasets by center-cropping the images in the $\times 0 . 6  \times 1 . 0$ datasets. Specifically, the cropping scale is set to $0 . 6 / 0 . 8 5$ for the UW image and 1 for the W image. For intermediate images in the training sequences, the cropping scale is linearly increased from 0.6/0.85 to 1 along the transition from the UW view to the W view.

## A. Experimental Settings

Data Construction for ZoomGS v2. We first capture multiple dual-camera image pairs to statistically characterize the geometric and color disparities between the UW and W cameras. For each scene, we capture 15 UW-W image pairs from different viewpoints using a smartphone. We then use COLMAP [53] to estimate the camera poses and intrinsic parameters of the captured images. For the Xiaomi device, we collect 48 indoor and 30 outdoor image pairs using a Redmi K50 Ultra across diverse scenarios, including classrooms, dining halls, shopping malls, campuses, and companies. For the Huawei device, we collect 25 indoor and 25 outdoor pairs using a Huawei Mate 40E. To construct the 3D scenes, we use 4,000 scenes from the DL3DV-10K [37] dataset, each of which contains more than 300 views. For each scene, we randomly sample four dual-camera parameter sets, yielding 16,000 synthetic DCSZ zoom sequences per device. After manually removing image sequences with poor 3DGS [28] reconstruction quality or severe rendering artifacts, 8,704 sequences remain for Xiaomi and 7,686 for Huawei.

Construction of Test Dataset. Obtaining suitable DCSZ test data for reference-based evaluation is challenging. Directly capturing real-world DCSZ sequences would require an array of smartphone cameras with different physical positions and camera parameters, making the setup impractical. Synthesizing DCSZ test data with ZoomGS [72] or ZoomGS v2 is also suboptimal. In ZoomGS, inaccuracies in Gaussian reconstruction can produce blurry rendered sequences with visible artifacts, making them unsuitable as ground-truth references. Moreover, the camera poses used by both ZoomGS and ZoomGS v2 are obtained through calibration and may not fully match the poses of real dual-camera systems. Consequently, evaluating VFI models on these rendered image sequences may not faithfully reflect their performance during real-world zooming. We therefore evaluate the generated sequences using noreference image and video quality metrics, which do not require ground-truth references. Therefore, we directly capture real-world UW/W image pairs and use them to evaluate the generated sequences with no-reference metrics. We assess the VFI results along three dimensions: image quality, video quality, and color continuity, to provide a multidimensional evaluation. For testing, we additionally collect 100 scenes for Xiaomi and 50 scenes for Huawei. Each scene contains a UW image and its corresponding W image.

Training Configurations. We optimize 3DGS representations in ZoomGS v2 using Adam optimizer [31] for 50k iterations, following the same rendering resolution and optimization schedule of the original 3DGS method. ZoomFI is first pretrained for 200k iterations on the data generated by ZoomGS v2 without color transformation and then fine-tuned for 100k iterations on color-transformed data. We augment the training data using random horizontal and vertical flips. Cosine annealing strategy [41] is employed to decrease the learning rate from $0 . 7 5 \times 1 0 ^ { - 4 }$ to $0 . 7 5 \times 1 0 ^ { - 6 }$ steadily. All experiments are conducted with PyTorch [50] on two NVIDIA GeForce RTX A6000 GPUs.

TABLE I: Quantitative comparisons of VFI models on the Xiaomi test dataset. † denotes models fine-tuned on data generated by ZoomGS [72], and ‡ denotes models fine-tuned on data generated by ZoomGS v2. ↑ indicates that higher values are better, and ↓ indicates that lower values are better. The best results are marked in red, while the second-best results are marked in blue. Models evaluated without fine-tuning produce excessively blurry results, which can artificially improve the color-continuity metrics by suppressing frame-to-frame color variations.
<table><tr><td>Finetuning Data</td><td>Networks</td><td colspan="2">MUSIQ↑ / MANIQA↑ / TOPIQ↑ (×0.6 ↔ ×1.0)</td><td colspan="2">DOVER↑ / FasterVQA↑</td><td colspan="2">Smooth-∆Emean↓ / Smooth-∆Emax↓ (×0.85 ↔ ×1.0)</td></tr><tr><td rowspan="3">None</td><td>RIFE [23]</td><td>33.058 / 0.213 / 0.290</td><td>48.26 / 0.666</td><td>0.142 / 0.445</td><td>42.830 / 0.246 / 0.321</td><td>67.65 / 0.688</td><td>0.016 / 0.074</td></tr><tr><td>AMT [35]</td><td>43.457 / 0.227 / 0.323</td><td>44.17 / 0.746</td><td>0.220 / 0.752</td><td>53.093 0.294 / 0.389</td><td>80.08 / 0.751</td><td>0.049 / 0.135</td></tr><tr><td>VFIMamba [81]</td><td>40.732 / 0.240 / 0.364</td><td>44.59 / 0.597</td><td>0.072 / 0.426</td><td>47.434 / / 0.272 / 0.397</td><td>69.36 / 0.752</td><td>0.059 / 0.438</td></tr><tr><td rowspan="5">ZoomGS [72]</td><td>RIFE† [23]</td><td>59.460 / 0.319 / 0.484</td><td>80.26 / 0.824</td><td>0.063 / 0.380</td><td>60.263 / 0.336 / 0.473</td><td>82.05 / 0.837</td><td>0.028 / 0.184</td></tr><tr><td>AMT† [35]</td><td>61.007 / 0.329 / 0.478</td><td>80.98 / 0.833</td><td>0.162 / 0.513</td><td>59.704 / 0.329 / 0.460</td><td>81.77 / 0.838</td><td>0.136 / 0.387</td></tr><tr><td>VFIMamba† [81]</td><td>53.870 / 0.288 / 0.406</td><td>80.48 / 0.801</td><td>0.149 / 0.466</td><td>53.569 / 0.302 / 0.414</td><td>80.48 / 0.797</td><td>0.059 / 0.309</td></tr><tr><td>ZoomFI-Lite†</td><td>63.139 / 0.358 3 / 0.511</td><td>81.94 / 0.837</td><td>0.032 / 0.167</td><td>59.988 / 0.345 / 0.473</td><td>82.39 / 0.840</td><td>0.022 / 0.145</td></tr><tr><td>ZoomFI†</td><td>63.689 / 0.354 / 0.518</td><td>81.98 / 0.838</td><td>0.032 / 0.195</td><td>61.438 / 0.348 /0.486</td><td>82.63 / 0.843</td><td>0.022 / 0.149</td></tr><tr><td rowspan="5">ZoomGS v2</td><td>RIFE‡[23]</td><td>65.238 / 0.371 / 0.556</td><td>81.51 / 0.837</td><td>0.032 / 0.133</td><td>63.574 / 0.359 / 0.513</td><td>82.53 / 0.852</td><td>0.047 / 0.136</td></tr><tr><td>AMT‡ [35]</td><td>64.704 / 0.366 / 0.546</td><td>82.36 / 0.845</td><td>0.162 / 0.510</td><td>62.694 / 0.355 / 0.498</td><td>82.45 / 0.847</td><td>0.107 / 0.401</td></tr><tr><td>VFIMamba‡ [81]</td><td>55.764 / 0.292 / 0.439</td><td>81.28 / 0.825</td><td>0.044 / 0.148</td><td>58.050 / 0.312 / 0.438</td><td>82.38 / 0.837</td><td>0.046 / 0.141</td></tr><tr><td> ${ \mathrm { Z o o m F I - L i t e ^ { \ddagger } } }$ </td><td>65.355 / 0.387 / 0.554</td><td>82.46 / 0.843</td><td>0.025 / 0.087</td><td>63.503 / 0.368 / 0.508</td><td>83.08 / 0.858</td><td>0.017 / 0.075</td></tr><tr><td> $\mathrm { Z o o m F I ^ { \dag } }$ </td><td>66.172 / 0.399 / 0.567</td><td>82.48 / 0.846</td><td>0.023 / 0.082</td><td>63.836 / 0.376 / 0.517</td><td>83.05 / 0.851</td><td>0.021 / 0.095</td></tr></table>

TABLE II: Quantitative comparisons of VFI models on the Huawei test dataset. † denotes models fine-tuned on data generated by ZoomGS [72], and <sup>‡</sup> denotes models fine-tuned on data generated by ZoomGS v2. ↑ indicates that higher values are better, and ↓ indicates that lower values are better. The best results are marked in red, while the second-best results are marked in blue. Models evaluated without fine-tuning produce excessively blurry results, which can artificially improve the color-continuity metrics by suppressing frame-to-frame color variations
<table><tr><td>Finetuning Data</td><td>Networks</td><td colspan="2">MUSIQ↑ / MANIQA↑ / TOPIQ↑ (×0.6 ↔ ×1.0)</td><td colspan="2">DOVER↑ / FasterVQA↑</td><td colspan="2"> $\mathrm { S m o o t h - } \Delta \mathrm { E } _ { m e a n } \downarrow$  / Smooth-  $\Delta \mathrm { E } _ { m a x } \downarrow$  (×0.85 ↔ ×1.0)</td></tr><tr><td rowspan="3">None</td><td>RIFE [23]</td><td>34.451 / 0.240 / 0.318</td><td>43.34 / 0.302</td><td>0.079 / 0.520</td><td>39.464 / 0.249 / 0.300</td><td>62.71 / 0.324</td><td>0.063 / 0.470</td></tr><tr><td>AMT [35]</td><td>40.899 / 0.233 / 0.311</td><td>73.58 / 0.339</td><td>0.157 / 0.570</td><td>51.503 / 0.287 /0.367</td><td>79.40 / 0.381</td><td>0.060 / 0.172</td></tr><tr><td>VFIMamba [81]</td><td>40.530 / 0.263 /0.369</td><td>40.99 / 0.280</td><td>0.085 / 0.573</td><td>44.737 / 0.271 0.388</td><td>59.82 / 0.339</td><td>0.064 / 0.452</td></tr><tr><td rowspan="4">ZoomGS [72]</td><td>RIFE† [23]</td><td>63.300 / 0.361 / 0.521</td><td>80.18 / 0.818</td><td>0.037 / 0.225</td><td>62.767 / 0.353 / 0.476</td><td>82.45 / 0.832</td><td>0.035 / 0.241</td></tr><tr><td>AMT† [35]</td><td>62.232 / 0.352 / 0.496</td><td>81.13 / 0.809</td><td>0.173 / 0.559</td><td>61.813 / 0.344 / 0.451</td><td>81.00 / 0.825</td><td>0.141 / 0.398</td></tr><tr><td>VFIMamba† [81]</td><td>47.315 / 0.270 / 0.350</td><td>79.84 / 0.754</td><td>0.191 / 0.663</td><td>51.827 / 0.304 / 0.398</td><td>80.60 / 0.780</td><td>0.109 / 0.731</td></tr><tr><td>ZoomFI-Lite† ZoomFI†</td><td>66.405 / 0.387 / 0.546 66.725 5/ 0.395 / 0.551</td><td>82.20 / 0.830 82.46 / 0.836</td><td>0.044 / 0.220 0.039 / 0.217</td><td>62.225 / 0.351 / 0.476 63.512 / 0.362 0.483</td><td>82.70 / 0.830 83.06 / 0.831</td><td>0.034 / 0.223 0.034 / 0.237</td></tr><tr><td rowspan="5">ZoomGS v2</td><td>RIFE[23]</td><td>67.918 / 0.402 / 0.578</td><td>82.37 / 0.830</td><td>0.027 / 0.086</td><td>66.287 / 0.378 / 0.516</td><td>83.08 / 0.851</td><td>0.041 / 0.114</td></tr><tr><td>AMT‡ [35]</td><td>65.806 / 0.374 / 0.546</td><td>83.10 / 0.838</td><td>0.178 / 0.536</td><td>65.699 / 0.366 / 0.500</td><td>83.60 / 0.855</td><td>0.124 / 0.415</td></tr><tr><td>VFIMamba‡ [81]</td><td>53.043 / 0.291 / 0.420</td><td>80.59 / 0.797</td><td>0.035 / 0.124</td><td>59.404 / 0.327  / 0.441</td><td>82.86 / 0.828</td><td>0.050 / 0.153</td></tr><tr><td> ${ \mathrm { Z o o m F I - L i t e } } ^ { \ddagger }$ </td><td>68.138 / 0.410 / 0.576</td><td>83.48 / 0.843</td><td>0.023 / 0.077</td><td>66.213 / 0.378 / 0.509</td><td>83.96 / 0.859</td><td>0.019 / 0.088</td></tr><tr><td> $\mathrm { Z o o m F I ^ { \dag } }$ </td><td>68.466 / 0.422 / 0.582</td><td>83.40 / 0.843</td><td>0.025 / 0.083</td><td>66.895 / 0.390 / 0.521</td><td>84.10 / 0.858</td><td>0.019 / 0.097</td></tr></table>

Evaluation Configurations. Since the test data lack groundtruth references, we employ several no-reference metrics, including MUSIQ [27], MANIQA [77], and TOPIQ [8] for single-image quality assessment, and DOVER [69], Faster-VQA [68] for video quality assessment. In addition, we design a protocol for evaluating color continuity. First, we align all frames with the W image using cropping center and optical-flow-based warping [56], with SEA-RAFT [64] used to estimate the flow. We then compute ∆E [43] between each aligned frame and the UW image. Next, we fit the color transition curve using a cubic polynomial, which serves as a compromise between modeling nonlinear color transitions and avoiding excessive model complexity. Finally, we compute the mean and maximum absolute differences between the fitted curve and the measured values across frames, denoted by $\mathrm { S m o o t h - } \Delta \mathrm { E } _ { m e a n }$ and $\mathrm { S m o o t h - } \Delta \mathrm { E } _ { m a x } ,$ , respectively.

## B. Comparison with State-of-the-Art Methods

Comparison Configurations. We compare our approach with three representative VFI methods (i.e., RIFE [23], AMT [35], and VFIMamba [81]) to evaluate the effectiveness of ZoomGS v2 and ZoomFI. All VFI models are trained under the same configurations to ensure a fair comparison.

Quantitative and Qualitative Results. As shown in Table I and Table II, models fine-tuned on data generated by ZoomGS v2 consistently outperform those fine-tuned on ZoomGS [72]

W Image

×0.6↔×1.0  
×0.85↔×1.0

![](images/12aaa5715a08937b2dfcd77af3f274899769c8c56d94dc7d8d1ed6a7ca2967ef.jpg)  
UW Image

UW Image  
![](images/663747c5b3b871fb6986c728014be0f5c188f9cfd8c22d7c5c1746d137bb5b96.jpg)  
W Image

![](images/0e8d00160117bd16065f0759ca3cf3f85e19c7012468c6e97685efd2b583cdfa.jpg)  
ZoomGS [72]

![](images/27548703f35282240432f7b84ebbf71bc4441f178c02676f471fe36b4b267fc1.jpg)  
ZoomGS v2

![](images/aa84ceca32056a6c5bc483911afcc4977fadb4ebfad2c84cda70c424ce29b977.jpg)

![](images/98ea1a12752cf446071c8b48273c34be289b4ff40ce96935619e6dc82d2d5d71.jpg)

![](images/0401127caaff7a624ba6e2b56e96df943f75369bae0b35d3cdfaba60a7acf230.jpg)  
ZoomGS [72]

![](images/cb9c3fb2eec632cdd8fa79035ae23514a91e7d5e1b5c9199f3601b5feee53cd0.jpg)  
ZoomGS v2

Fig. 7: Visual comparisons on the Xiaomi test dataset of ZoomFI models trained with data generated by ZoomGS [72] and ZoomGS v2. The top and bottom rows correspond to the ×0.6 ↔ ×1.0 and ×0.85 ↔ ×1.0 settings, respectively.  
![](images/9cc8f647bc9a4ee71e2932056e0add75886d0fc6bca30b882dd637d42d30689f.jpg)  
Fig. 8: Visual comparisons on the Huawei test dataset of ZoomFI models trained with data generated by ZoomGS [72] and ZoomGS v2. The top and bottom rows correspond to the ×0.6 ↔ ×1.0 and ×0.85 ↔ ×1.0 settings, respectively.

and their non-fine-tuned counterparts on most evaluation metrics, with particularly notable improvements in perceptual quality. These results support the effectiveness of ZoomGS v2 in improving VFI performance in real-world DCSZ scenarios. Non-fine-tuned models produce overly blurry results that suppress high-frequency color details. This can artificially reduce frame-to-frame fluctuations in ∆E [43], making the color continuity metrics appear better than the actual visual quality would suggest. Visual comparisons in Fig. 7 and Fig. 8 further support these findings, showing that models fine-tuned on ZoomGS v2 data preserve more photorealistic details and exhibit fewer visual artifacts. Across the evaluated settings, ZoomFI achieves the best performance on most evaluation metrics among the compared methods, as illustrated in Fig. 9 and Fig. 10.

Efficiency of VFI Models. As shown in Table III, under identical hardware and implementation settings, ZoomFI has a shorter inference time than VFIMamba and AMT, whereas ZoomFI-Lite achieves the lowest inference time and FLOPs count among all compared methods. These results indicate that ZoomFI balances visual quality and inference speed, whereas ZoomFI-Lite offers the lowest computational cost and maintains competitive generation quality relative to the other methods, except ZoomFI. Both models are promising candidates for mobile deployment. All evaluations are performed at a resolution of 1632 × 1216.

TABLE III: Comparison of FLOPs, parameters, and inference time of different VFI models at a test resolution of 1632 × 1216.
<table><tr><td>Networks</td><td>FLOPs (G)</td><td>Parameters (M)</td><td>Time (ms)</td></tr><tr><td>RIFE [23]</td><td>708.3</td><td>10.1</td><td>48.7</td></tr><tr><td>AMT [35]</td><td>540.3</td><td>3.0</td><td>183.2</td></tr><tr><td>VFIMamba [81]</td><td>1792.9</td><td>16.1</td><td>159.7</td></tr><tr><td>ZoomFI-Lite</td><td>422.7</td><td>12.5</td><td>40.9</td></tr><tr><td>ZoomFI</td><td>724.2</td><td>12.5</td><td>52.1</td></tr></table>

## C. Ablation Study

We conduct ablation studies to evaluate the two main components of our approach: the ZoomGS v2 data generation pipeline and ZoomFI. We also investigate how the amount of data generated by ZoomGS v2 affects fine-tuning. All ablation experiments are conducted on Xiaomi smartphone device and ×0.6 ↔ ×1.0 setting.

Effect of Sampling Strategy of ∆V in ZoomGS v2. We model the relative translation offsets ∆V with a multivariate Gaussian distribution (Sec. III-B) and sample from it to generate synthetic zoom data. For a fair comparison, Gaussian and uniform sampling use the same number of samples and identical sampling ranges, with the bounds of the uniform distribution are set to the minimum and maximum values observed in the real dual-camera dataset. All other training configurations follow Sec. IV-A. Compared with uniform

×0.6↔×1.0

![](images/a3738cfb20b9ae5d9eee99dd3ba39099e10dbf68dc02974f6f879fa784583280.jpg)

![](images/067f49ef39ae48bcbd501bfbc7a4b6555613ba4b0d474fd72d7534474339a90b.jpg)

![](images/dc072b0e7c1ee9d4e73cf5dd0648f86e73ee083cd29475c0b2a4f36adc668576.jpg)

![](images/cda543c1ef30d720b87caecf2385f39c3f0c448ef72f276333e035873012048f.jpg)

![](images/549b5a19f6d65612c859db3e765979d55d70b3abc4656a9b37c20c8c45d0463d.jpg)

![](images/03800e208b2d94fc33d75e5db67315235432582dbe2ff93d9966e9dca7a8a4be.jpg)  
×0.85↔×1.0

![](images/4019b8e898bba6dc423000b95828135984321cd84195de5e3f9e9fb52cd7c80a.jpg)  
UW Image

![](images/bb83c5de9b4551722f1ecd3cfd58097fee1f5851df236463b53854fcbbdca215.jpg)  
W Image

![](images/04da38b368be5fc1455fbeae82ff3cafdf77d8512186046a88d75b2d14baf0d6.jpg)  
AMT<sup>‡</sup> [35]

![](images/d532dea9347df3b9a6b6ade57d8c939b3cff42dec3d9fce045f48eca104c1744.jpg)  
VFIMamba<sup>‡</sup> [81]

![](images/1077dfecef8e9b67712cf9a0160fb0ce91936b2a603a3dbc9c28c641fd1884a8.jpg)  
RIFE<sup>‡</sup> [23]

![](images/9f55060d840f13a9b1c38a8d9a645243ad7b8caaf3ea255cf0f1d0500a0af9c2.jpg)  
ZoomFI<sup>‡</sup>  
×0.6↔×1.0

Fig. 9: Visual comparisons of different VFI models on the Xiaomi test dataset. The top and bottom rows correspond to the ×0.6 ↔ ×1.0 and ×0.85 ↔ ×1.0 settings, respectively.  
![](images/eaf26cbd04d29f946f0b6cd8beb9e43a08f0ecfaa6640c351aa75ca60aa165c7.jpg)

![](images/7ca937489743e3b1c03b234936840c25961fe6dc9e6a3a9f2a3bb84e864ac901.jpg)

![](images/e88b60c0656982cee2424beb2f1d9642031e0304193390cc19ba2e471fe4d121.jpg)

![](images/f2ec6f4ee5472360a5ccc451f0bce557ae2a9503f030a4b2e8eb3b92b7176983.jpg)  
×0.85↔×1.0

![](images/5c0ea19bd8b07dd68086fa133b4fe4ac6fcc967419daaf704f93084982ccaf0e.jpg)  
UW Image

![](images/ced6ada7754a9aa077b570f98b1d7f3aa2c8e419da2de5228fc18f4aab30b5d6.jpg)  
W Image

![](images/adef91b1fe7f06bce394de40af8fd063b28ba2f8d51b5b267bdd24ae8140c32b.jpg)

![](images/ddc2c38263b0998d610a908c8965eb54d9b46133e3461f7790592f4b3eaea42c.jpg)

![](images/c43e39dc763d3287d4de47c086d8b13b778c119686e98480b683b03b4c05da01.jpg)

AMT<sup>‡</sup> [35]  
![](images/d67c006ec8e19fad1fb413b9124e91ab63bcd01960365cd561b6bb933fb3fa3e.jpg)  
VFIMamba<sup>‡</sup> [81]

![](images/5c97ca3abc2e90cd4680fe7feda735a43881222dae08752a4a635ecfe291c8e9.jpg)  
RIFE<sup>‡</sup> [23]

![](images/aa4f377549677a51afdccb8c0bc0f91f2fe16288237b1a71743e4cbb9ab1eda5.jpg)  
ZoomFI<sup>‡</sup>

Fig. 10: Visual comparisons of different VFI models on the Huawei test dataset. The top and bottom rows correspond to the ×0.6 ↔ ×1.0 and ×0.85 ↔ ×1.0 settings, respectively.  
![](images/beb10abe678cdf17de806eacf2578a5188ebab866aa7ba489de7f7a8a02c70f4.jpg)  
UW Image

![](images/5787859a2b30e811edf242c5161ae4195eed36dd9d99bc03858cdd86d034c0b6.jpg)  
W Image

![](images/2badd98575334f81f63a0f09217b3b4d8b113eab85a81f4e9f51abea9bc93892.jpg)  
Uniform

![](images/948e001476ac7ad33c7881df6c1a8d60fa1a81011319bb468da05857160505b1.jpg)  
Fig. 11: Visual comparison of different sampling strategies for ∆V. Gaussian sampling produces fewer misalignment artifacts and better visual results than Uniform sampling.  
Gaussian

TABLE IV: Ablation studies of the ∆V sampling strategy in ZoomGS v2.
<table><tr><td>∆V sample manner</td><td>MUSIQ↑ / MANIQA↑ / TOPIQ↑</td><td>DOVER↑/ FasterVQA↑</td><td>Smooth-∆Emean↓/ Smooth-∆Emax↓</td></tr><tr><td>Uniform</td><td>65.692 / 0.397 / 0.561</td><td>82.12 / 0.841</td><td>0.021  / 0.096</td></tr><tr><td>Gaussian</td><td>66.172 / 0.399 / 0.567</td><td>82.48  / 0.846</td><td>0.023  / 0.082</td></tr></table>

sampling, Gaussian sampling improves all image and video quality metrics and one color continuity metric, supporting the effectiveness of the multivariate Gaussian sampling, as shown in Table IV. The visual comparisons in Fig. 11 are consistent with these results: Gaussian sampling produces substantially fewer misalignment artifacts than Uniform sampling. This improvement may be attributed to the closer match between Gaussian sampling and the empirical distribution of real dual-camera translations. As shown in Fig. 4, the measured relative translation offsets are concentrated in a limited region rather than distributed uniformly. Uniform sampling therefore produces more atypical camera configurations, increasing the distribution gap between the synthetic training data and real DCSZ data.

Effect of Color Transformation in ZoomGS v2. We evaluate the effectiveness of color transformation and its components. The training configurations follow Sec. IV-A, except that the fine-tuning stage with color-transformed data is omitted or changed. Table V reports the ablation results for color transformation. Adding color transformation improves all videoquality and color-continuity metrics over the baseline without color transformation, indicating that reducing the photometric gap improves VFI performance. Moreover, the full color transformation framework outperforms variants that omit either gamma correction or the simplified linear-mapping module on all seven metrics, demonstrating the contribution of both components. Visual results further support these findings. As illustrated in Fig. 12, we calculate ∆E [43] between each image sequences generated by ZoomFI trained on data with and without color transformation. For each sequence, we compute ∆E between each generated frame and its corresponding UW images. The color-transformed sequences exhibit smoother variations in ∆E than the untransformed sequences, indicating the effectiveness of color transformation in improving color continuity. Furthermore, color-transformed UW images are more color-consistent with the W images and exhibit less chromatic aberration (CA) than the uncorrected images, as shown in Fig. 13. These results further demonstrate the effectiveness of the color transformation module.

![](images/73126b1fe2b3f6109a6f3055899f51904fa8e6123670c8b2eee11cc413188ba3.jpg)  
w/ Color Transformation  
Fig. 12: Color-difference trajectories of sequences generated by ZoomFI models trained with and without color transformation. For each sequence, we compute ∆E [43] between each generated frame and its corresponding UW image. Four representative scenes are shown as examples, with each column corresponding to one scene. The ∆E values are normalized within each plot.

TABLE V: Ablation study of color-transformation module and its gamma-correction and linear-mapping components.
<table><tr><td>Gamma Params</td><td>Linear Map Params</td><td>MUSIQ↑ / MANIQA↑ / / TOPIQ↑</td><td>DOVER↑/ FasterVQA↑</td><td>Smooth-∆Emean↓/ Smooth-∆Emax↓</td></tr><tr><td>×</td><td>×</td><td>66.198 / 0.400 / 0.566</td><td>82.36 / 0.845</td><td>0.029 / 0.090</td></tr><tr><td>×</td><td>√</td><td>66.028 / 0.397 / 0.565</td><td>82.39 / 0.844</td><td>0.026 / 0.084</td></tr><tr><td>√</td><td>×</td><td>63.022 / 0.370 / 0.520</td><td>81.76 / 0.829</td><td>0.189 / 0.441</td></tr><tr><td>√</td><td>√</td><td>66.172 /  0.399  / 0.567</td><td>82.48  / 0.846</td><td>0.023 / 0.082</td></tr></table>

Effect of Training Data Scale from ZoomGS v2. To evaluate the contribution of ZoomGS v2, we increase the proportion of scenes used for training from 1% to 100% according to an exponential schedule. The training configurations follow Sec. IV-A, except that the number of training iterations is scaled in proportion to the dataset size so that each training sample is accessed approximately the same number of times. As summarized in Table VI, the VFI model achieves its best overall performance when trained on the full ZoomGS v2 dataset. Overall, performance improves as the amount of synthetic training data increases, demonstrating the benefits of scaling up the training dataset. Moreover, when trained on only 1.0% of the ZoomGS v2 scenes, the VFI model performs comparably to or better than the model trained on ZoomGS data, as reported in Table I. The number of scenes in this subset is comparable to that in the ZoomGS dataset, suggesting that ZoomGS v2 provides more effective training data even at a similar scene count.

![](images/f18d05b1072423325deb31e66893f161b95323d9a7a87d272edd27398ffed3a3.jpg)  
Fig. 13: Visual comparison of the UW images before and after color transformation.

![](images/472fe25f4ee9f40f0205125ad0b908b0065b0a43328bdebe9ba5524ef963fa95.jpg)  
Fig. 14: Visual comparisons of ZoomFI with different combinations of forward and backward warping. Combining both warping directions yields better visual results than using either direction alone.

TABLE VI: Effect of training data scale on VFI performance using data generated by ZoomGS v2.
<table><tr><td>Percentage of Data</td><td>MUSIQ↑ / MANIQA↑ / TOPIQ↑</td><td>DOVER↑/ FasterVQA↑</td><td>Smooth-∆Emean↓/ Smooth-∆Emax↓</td></tr><tr><td>1.0%</td><td>64.324 / 0.385 / 0.539</td><td>82.02 / 0.837</td><td>0.036 / 0.117</td></tr><tr><td>10%</td><td>65.174 / 0.393 / 0.551</td><td>82.06 / 0.841</td><td>0.034 / 0.110</td></tr><tr><td>100%</td><td>66.172 / 0.399 / 0.567</td><td>82.48  / 0.846</td><td>0.023 / 0.082</td></tr></table>

TABLE VII: Ablation study of backward and forward warping in ZoomFI.
<table><tr><td>backward warp</td><td>forward warp</td><td>MUSIQ↑ / MANIQA↑ / TOPIQ↑</td><td>DOVER↑/ FasterVQA↑</td><td> $\begin{array} { c } { \mathrm { S m o o t h - } \Delta \mathrm { E } _ { m e a n } \downarrow / } \\ { \mathrm { S m o o t h - } \Delta \mathrm { E } _ { m a x } \downarrow } \end{array}$ </td></tr><tr><td>√</td><td>×</td><td>65.917 / 0.391 / 0.552</td><td>82.39 / 0.847</td><td>0.032 / 0.096</td></tr><tr><td>×</td><td>√</td><td>65.229 / 0.384 / 0.553</td><td>82.05 / 0.839</td><td>0.045 / 0.129</td></tr><tr><td>√</td><td>√</td><td>66.172 / 0.399 / 0.567</td><td>82.48  / 0.846</td><td>0.023  / 0.082</td></tr></table>

Effect of Bidirectional Warping in ZoomFI. We investigate the respective contributions of forward and backward flow in ZoomFI. The training configurations follow those described in Sec. IV-A. As shown in Table VII, using both forward and backward flows consistently improves performance across all evaluation metrics. This improvement can be attributed to the complementary sampling properties of the two warping directions. Specifically, backward warping samples sourceframe features at target locations, whereas forward warping maps source locations toward the target. Their complementary properties can improve robustness in regions affected by occlusion and large motion. Combining the two warping directions therefore enables more accurate feature alignment and reduces interpolation artifacts. Visual comparisons in Fig. 14 further support this finding: using both flows produces fewer blurring artifacts and structural distortions than using either direction alone.

## V. CONCLUSION

In this paper, we introduced the Dual-Camera Smooth Zoom (DCSZ) task to address abrupt transitions in smartphone dual-camera zooming. To mitigate the lack of practical realworld supervision, we reformulated DCSZ data synthesis as a decoupled geometry–photometry modeling problem and proposed ZoomGS v2, a 3D Gaussian Splatting (3DGS)- based pipeline that synthesizes scalable training data from single-camera datasets by modeling inter-camera geometric and color discrepancies. Building upon the synthesized data, we developed ZoomFI, a DCSZ-oriented frame interpolation network that leverages bidirectional warping to generate visually plausible intermediate frames. Extensive experiments on test datasets captured using Xiaomi and Huawei devices demonstrate the data synthesis quality of our approach, and verify that ZoomGS v2 can be extended to various mobile devices. Future work will explore more accurate camera and ISP modeling as well as efficient deployment on mobile platforms.

## REFERENCES

[1] Abdelhamed, A., Punnappurath, A., Brown, M.S.: Leveraging the availability of two cameras for illuminant estimation. In: CVPR. pp. 6637– 6646 (2021) 1, 3

[2] Alzayer, H., Abuolaim, A., Chan, L.C., Yang, Y., Lou, Y.C., Huang, J.B., Kar, A.: Dc2: Dual-camera defocus control by learning to refocus. In: CVPR. pp. 21488–21497 (2023) 1, 3

[3] Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: ICCV. pp. 5855–5864 (2021) 2

[4] Blahnik, V., Schindelbeck, O.: Smartphone imaging technology and its applications. ADV OPT TECHNOL 10(3), 145–232 (2021) 1

[5] Cen, J., Fang, J., Yang, C., Xie, L., Zhang, X., Shen, W., Tian, Q.: Segment any 3d gaussians. In: AAAI. vol. 39, pp. 1971–1979 (2025) 3

[6] Chan, K.C., Xiao, J., Goshu, H.L., Lam, K.M.: Point cloud densification for 3d gaussian splatting from sparse input views. In: Proc. ACM Int. Conf. Multimed. pp. 896–904 (2024) 3

[7] Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: CVPR. pp. 19457–19467 (2024) 3

[8] Chen, C., Mo, J., Hou, J., Wu, H., Liao, L., Sun, W., Yan, Q., Lin, W.: Topiq: A top-down approach from semantics to distortions for image quality assessment. TIP 33, 2404–2418 (2024). https://doi.org/10.1109/TIP.2024.3378466 8

[9] Chen, G., Wang, W.: A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890 (2024) 3

[10] Chen, Y., Potamias, R.A., Ververas, E., Song, J., Deng, J., Lee, G.H.: Zerogs: Training 3d gaussian splatting from unposed images. arXiv preprint arXiv:2411.15779 (2024) 3

[11] Chen, Z., Wang, F., Wang, Y., Liu, H.: Text-to-3d using gaussian splatting. In: CVPR. pp. 21401–21412 (2024) 3

[12] Chung, J., Oh, J., Lee, K.M.: Depth-regularized optimization for 3d gaussian splatting in few-shot images. In: CVPR. pp. 811–820 (2024) 3

[13] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021) 2

[14] Fan, L., Yang, Y., Li, M., Li, H., Zhang, Z.: Trim 3d gaussian splatting for accurate geometry representation. arXiv preprint arXiv:2406.07499 (2024) 3

[15] Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z.: Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. NeurIPS 37, 140138–140158 (2024) 3

[16] Feng, Q.Y., Cao, G.C., Chen, H.X., Xu, Q.C., Mu, T.J., Martin, R., Hu, S.M.: Evsplitting: an efficient and visually consistent splitting algorithm for 3d gaussian splatting. In: SIGGRAPH Asia ’24. pp. 1–11 (2024) 3

[17] Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A.A., Wang, X.: Colmapfree 3d gaussian splatting. In: CVPR. pp. 20796–20805 (2024) 3

[18] Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A.A., Wang, X.: Colmapfree 3d gaussian splatting. In: CVPR. pp. 20796–20805 (2024) 3

[19] Ge, Y., Behl, H., Xu, J., Gunasekar, S., Joshi, N., Song, Y., Wang, X., Itti, L., Vineet, V.: Neural-sim: Learning to generate training data with nerf. In: ECCV. pp. 477–493. Springer (2022) 2

[20] Guo, Z., Li, W., Loy, C.C.: Generalizable implicit motion modeling for video frame interpolation. In: NeurIPS. vol. 37, pp. 63747–63770 (2024) 1, 2

[21] Hanrahan, M.L.P.: Light field rendering. SIGGRAPH96, Computer Graphics Proceeding (1996) 2

[22] Hu, X., Wang, Y., Fan, L., Fan, J., Peng, J., Lei, Z., Li, Q., Zhang, Z.: Semantic anything in 3d gaussians. arXiv preprint arXiv:2401.17857 (2024) 3

[23] Huang, Z., Zhang, T., Heng, W., Shi, B., Zhou, S.: Real-time intermediate flow estimation for video frame interpolation. In: ECCV. pp. 624–642. Springer (2022) 2, 6, 7, 8, 9, 10

[24] Jain, U., Mirzaei, A., Gilitschenski, I.: Gaussiancut: Interactive segmentation via graph cut for 3d gaussian splatting. NeurIPS 37, 89184–89212 (2024) 3

[25] Jin, X., Wu, L., Chen, J., Chen, Y., Koo, J., Hahm, C.h.: A unified pyramid recurrent network for video frame interpolation. In: CVPR. pp. 1578–1587 (2023) 2

[26] Kalluri, T., Pathak, D., Chandraker, M., Tran, D.: Flavr: Flow-agnostic video representations for fast frame interpolation. In: ICCV. pp. 2071– 2082 (2023) 2

[27] Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: ICCV. pp. 5148–5157 (2021) 8

[28] Kerbl, B., Kopanas, G., Leimkuhler, T., Drettakis, G.: 3d gaussian¨ splatting for real-time radiance field rendering. TOG 42(4) (2023) 1, 3, 4, 7

[29] Kim, S.Y., Oh, J., Kim, M.: Fisr: Deep joint frame interpolation and super-resolution with a multi-scale temporal loss. In: AAAI. vol. 34, pp. 11278–11286 (2020) 2

[30] Kim, Y.: Convolutional neural networks for sentence classification. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). pp. 1746–1751 (2014) 4, 5, 6

[31] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 7

[32] Kong, L., Jiang, B., Luo, D., Chu, W., Huang, X., Tai, Y., Wang, C., Yang, J.: Ifrnet: Intermediate feature refine network for efficient frame interpolation. In: CVPR. pp. 1969–1978 (2022) 2

[33] Lan, K., Li, H., Shi, H., Wu, W., Wang, L., Liao, Y.: 2d-guided 3d gaussian segmentation. In: ASIANComNet. pp. 1–5. IEEE (2024) 3

[34] Li, Z., Chen, Z., Li, Z., Xu, Y.: Spacetime gaussian feature splatting for real-time dynamic view synthesis. In: CVPR. pp. 8508–8520 (2024) 3

[35] Li, Z., Zhu, Z.L., Han, L.H., Hou, Q., Guo, C.L., Cheng, M.M.: Amt: All-pairs multi-field transforms for efficient frame interpolation. In: CVPR. pp. 9801–9810 (2023) 1, 2, 6, 8, 9, 10

[36] Li, Z., Chen, Y., Zhao, L., Liu, P.: Controllable text-to-3d generation via surface-aligned gaussian splatting. In: 3DV. pp. 1113–1123. IEEE (2025) 3

[37] Ling, Lu, Y.S., Tu, Z.: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. CVPR pp. 22160–22169 (2024) 6, 7

[38] Liu, C., Zhang, G., Zhao, R., Wang, L.: Sparse global matching for video frame interpolation with large motion. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19125–19134. IEEE (2024) 6

[39] Liu, I., Su, H., Wang, X.: Dynamic gaussians mesh: Consistent mesh reconstruction from dynamic scenes. arXiv preprint arXiv:2404.12379 (2024) 3

[40] Liu, P., Zuo, X., Larsson, V., Pollefeys, M.: Mba-vo: Motion blur aware visual odometry. In: ICCV. pp. 5550–5559 (2021) 6

[41] Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016) 8

[42] Lu, L., Wu, R., Lin, H., Lu, J., Jia, J.: Video frame interpolation with transformer. In: CVPR. pp. 3532–3542 (2022) 1, 2

[43] McLaren, K.: Xiii—the development of the cie 1976 (l\* a\* b\*) uniform colour space and colour-difference formula. Journal of the Society of Dyers and Colourists 92(9), 338–341 (1976) 5, 8, 9, 11

[44] Melas-Kyriazi, L., Laina, I., Rupprecht, C., Neverova, N., Vedaldi, A., Gafni, O., Kokkinos, F.: Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. In: PMLR 235. pp. 35310– 35323 (2024) 3

[45] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65(1), 99–106 (2021) 2

[46] Muller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics¨ primitives with a multiresolution hash encoding. ToG 41(4), 1–15 (2022) 2

[47] Niklaus, S., Liu, F.: Softmax splatting for video frame interpolation. In: CVPR. pp. 5437–5446 (2020) 2, 7

[48] Pan, Z., Yang, Z., Zhu, X., Zhang, L.: Efficient4d: Fast dynamic 3d object generation from a single-view video. Int. J. Comput. Vis. 134(1), 14 (2026) 3

[49] Papantonakis, P., Kopanas, G., Kerbl, B., Lanvin, A., Drettakis, G.: Reducing the memory footprint of 3d gaussian splatting. P ACM COMPUT GRAPH 7(1), 1–17 (2024) 3

[50] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-performance deep learning library. NeurIPS 32 (2019) 8

[51] Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural radiance fields for dynamic scenes. In: CVPR. pp. 10318–10327 (2021) 2

[52] Ren, J., Pan, L., Tang, J., Zhang, C., Cao, A., Zeng, G., Liu, Z.: Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142 (2023) 3

[53] Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: CVPR. pp. 4104–4113 (2016) 5, 7

[54] Shen, T., Li, D., Gao, Z., Tian, L., Barsoum, E.: Ladder: An efficient framework for video frame interpolation. arXiv preprint arXiv:2404.11108 (2024) 2

[55] Shi, Z., Xu, X., Liu, X., Chen, J., Yang, M.H.: Video frame interpolation transformer. In: CVPR. pp. 17482–17491 (2022) 2

[56] Shimizu, J., Sun, H., Katto, J.: Forward and backward warping for optical flow-based frame interpolation. In: 2022 International Conference on Artificial Intelligence in Information and Communication (ICAIIC). pp. 082–086 (2022). https://doi.org/10.1109/ICAIIC54071.2022.9722682 5, 8

[57] Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014) 7

[58] Szeliski, R., Gortler, S., Grzeszczuk, R., Cohen, M.F.: The lumigraph. In: SIGGRAPH 1996. pp. 43–54 (1996) 2

[59] Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast single-view 3d reconstruction. In: CVPR. pp. 10208–10217 (2024) 3

[60] Tosi, F., Tonioni, A., De Gregorio, D., Poggi, M.: Nerf-supervised deep stereo. In: CVPR. pp. 855–866 (2023) 2

[61] Wan, Y., Shao, M., Wu, R., Zuo, W.: Color3d: Controllable and consistent 3d colorization with personalized colorizer (2026) 3

[62] Wang, J., Fang, J., Zhang, X., Xie, L., Tian, Q.: Gaussianeditor: Editing 3d gaussians delicately with text instructions. In: CVPR. pp. 20902– 20911 (2024) 3

[63] Wang, P., Zhao, L., Ma, R., Liu, P.: Bad-nerf: Bundle adjusted deblur neural radiance fields. In: CVPR. pp. 4170–4179 (2023) 6

[64] Wang Yihan, L.L., Deng, J.: Sea-raft: Simple, efficient, accurate raft for optical flow. ECCV pp. 36–54 (2024) 6, 8

[65] Wei, Y., Zhang, J., Zhang, X., Shao, L., Lu, S.: Pcr-gs: Colmap-free 3d gaussian splatting via pose co-regularizations. In: ICCV. pp. 26499– 26508 (2025) 3

[66] Wu, G., Tao, X., Li, C., Wang, W., Liu, X., Zheng, Q.: Perceptionoriented video frame interpolation via asymmetric blending. In: CVPR. pp. 2753–2762 (2024) 1, 2

[67] Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: CVPR. pp. 20310–20320 (2024) 3

[68] Wu, H., Chen, C., Liao, L., Hou, J., Sun, W., Yan, Q., Gu, J., Lin, W.: Neighbourhood representative sampling for efficient end-to-end video quality assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(12), 15185–15202 (2023) 8

[69] Wu, H., Zhang, E., Liao, L., Chen, C., Hou, J., Wang, A., Sun, W., Yan, Q., Lin, W.: Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 20144– 20154 (2023) 8

[70] Wu, J., Liu, K., Gao, H., Jiang, X., Zhang, L.: Dig3d: Marrying gaussian splatting with deformable transformer for single image 3d reconstruction. CoRR (2024) 3

[71] Wu, R., Zhang, Z., Chen, M., Yan, Z., Zuo, W.: Deblur4dgs: 4d gaussian splatting from blurry monocular video. In: AAAI. vol. 40, pp. 10727– 10735 (2026) 3

[72] Wu, R., Zhang, Z., Yang, Y., Zuo, W.: Dual-camera smooth zoom on mobile phones. In: ECCV. pp. 250–269. Springer (2024) 1, 3, 4, 7, 8, 9

[73] Xiong, H., Muttukuru, S., Upadhyay, R., Chari, P., Kadambi, A.: Sparsegs: Real-time 360° sparse view synthesis using gaussian splatting. arXiv e-prints pp. arXiv–2312 (2023) 3

[74] Yan, Z., Low, W.F., Chen, Y., Lee, G.H.: Multi-scale 3d gaussian splatting for anti-aliased rendering. In: CVPR. pp. 20923–20931 (2024) 3

[75] Yang, C., Li, S., Fang, J., Liang, R., Xie, L., Zhang, X., Shen, W., Tian, Q.: Gaussianobject: Just taking four images to get a high-quality 3d object with gaussian splatting. arXiv preprint arXiv:2402.10259 (2024) 3

[76] Yang, J., Pavone, M., Wang, Y.: Freenerf: Improving few-shot neural rendering with free frequency regularization. In: CVPR. pp. 8254–8263 (2023) 2

[77] Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., Yang, Y.: Maniqa: Multi-dimension attention network for no-reference image quality assessment. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1191–1200 (2022) 8

[78] Yen-Chen, L., Florence, P., Barron, J.T., Lin, T.Y., Rodriguez, A., Isola, P.: Nerf-supervision: Learning dense object descriptors from neural radiance fields. In: ICRA. pp. 6496–6503. IEEE (2022) 2

[79] Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-splatting: Aliasfree 3d gaussian splatting. In: CVPR. pp. 19447–19456 (2024) 3

[80] Zhang, F., Luo, Y., Zhang, T., Zhang, L., Huang, Z.: Gaussianforest: Hierarchical-hybrid 3d gaussian splatting for compressed scene modeling. arXiv preprint arXiv:2406.08759 (2024) 3

[81] Zhang, G., Liu, C., Cui, Y., Zhao, X., Ma, K., Wang, L.: Vfimamba: Video frame interpolation with state space models. NeurIPS 37, 107225– 107248 (2024) 2, 8, 9, 10

[82] Zhang, G., Zhu, Y., Wang, H., Chen, Y., Wu, G., Wang, L.: Extracting motion and appearance via inter-frame attention for efficient video frame interpolation. In: CVPR. pp. 5682–5692 (2023) 2

[83] Zhang, J., Li, J., Yu, X., Huang, L., Gu, L., Zheng, J., Bai, X.: Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In: ECCV. pp. 335–352. Springer (2024) 3

[84] Zhang, Z., Wang, R., Zhang, H., Chen, Y., Zuo, W.: Self-supervised learning for real-world super-resolution from dual zoomed observations. In: ECCV. pp. 610–627. Springer (2022) 1, 3

[85] Zhong, Z., Krishnan, G., Sun, X., Qiao, Y., Ma, S., Wang, J.: Clearer frames, anytime: Resolving velocity ambiguity in video frame interpolation. In: ECCV. pp. 346–363. Springer (2024) 1, 2

[86] Zhou, K., Li, W., Han, X., Lu, J.: Exploring motion ambiguity and alignment for high-quality video frame interpolation. In: CVPR. pp. 22169–22179 (2023) 2

[87] Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: Fsgs: Real-time few-shot view synthesis using gaussian splatting. In: ECCV. pp. 145–163. Springer (2024) 3

[88] Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: Fsgs: Real-time few-shot view synthesis using gaussian splatting. In: ECCV. pp. 145–163. Springer (2024) 3

[89] Zhuang, J., Kang, D., Cao, Y.P., Li, G., Lin, L., Shan, Y.: Tip-editor: An accurate 3d editor following both text-prompts and image-prompts. ToG 43(4), 1–12 (2024) 3