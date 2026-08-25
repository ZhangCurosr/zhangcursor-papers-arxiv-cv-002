# Neighbor-Aware View Synthesis for Restoring Missing Views in Light-Field Camera Arrays

Sakshi Goel, Ayush Goyal, K S Venkatesh, Koteswar Rao Jerripothula

Indian Institute of Technology Kanpur (IIT Kanpur), India

Email: {sakshigl, gayush23, venkats, kotesrj}@iitk.ac.in

Abstract—In light-field (LF) imaging systems, dense spatial sampling from a camera array enables powerful post-capture capabilities such as refocusing and depth estimation. However, real-world LF capture is often affected by hardware malfunctions, where one or more cameras in the array fail, leading to missing sub-aperture images and degraded reconstruction quality. This paper addresses the problem of defective or missing view restoration in light-field camera arrays. We propose a novel generative framework that synthesizes the absent views by exploiting information from a carefully selected subset of neighboring cameras. These selected images, along with a positional encoding map indicating both their locations and the desired target view, are fed into a conditional Generative Adversarial Network (cGAN) trained to generate the missing viewpoint in a geometrically consistent manner. Extensive experiments on synthetic and real-world LF datasets demonstrate that our method produces visually plausible and photometrically accurate reconstructions, outperforming baselines for view interpolation both quantitatively and qualitatively. The proposed framework thus offers a robust and efficient solution for fault-tolerant lightfield image acquisition.

## I. INTRODUCTION

Light field imaging has emerged as a revolutionary technology, attracting significant attention in both academia and industry, especially with the advent of commercial plenoptic cameras [1], [5] and recent dedication in the field of Virtual Reality (VR) and Augmented Reality (AR) [12]. Light field (LF) cameras need camera arrays, which can be incorporated in a single camera as well by placing a microlens array between the main lens and the image sensor, thereby capturing both the intensity and direction of light rays from real world scenes. While such camera array setups offer high spatial resolution and wide baselines, to achieve dense angular sampling using a synchronized grid of cameras, building a dense one is quite complex.

Moreover, the hardware malfunctions in the multi-camera architecture can lead to missing Sub-Aperture Images (SAIs), or simply views, leading to gaps in the 4D LF data structure. These missing views can significantly degrade performance of downstream algorithms resulting in incorrect depth maps and visible discontinuities in the final rendering. One possible solution to this problem is missing view synthesis and is an important research direction to explore for building efficient and reliable LF acquisition systems.

Inspired by traditional approaches such as learnable interpolation or geometry based rendering, Kalantari et.al [4] breaks down the goal of view synthesis into disparity estimator and predictor modeled by Convolutional Nueral Network (CNN) and achieves it using the four corner views, outperforming prior methods [6], [7], but continues to struggle in reconstructing occluded and non-lambertain surfaces. Wu et.al [8] present a ”blur-restoration deblur” framework using Epiplolar Plane Images (EPIs) without estimating the geometry of the scene. Recently, deep learning-based methods [10], [11], have shown remarkable success in image reconstruction and novel view synthesis. However, many existing frameworks do not promise efficient LF missing view restoration, as they may require large amounts of training data, may not explicitly leverage the underlying geometric consistency of the Light Field or exploit all the available input views.

![](images/f7838dbd9f13c9b866859ba32dc386dc94bb5415f4d0938385f71d406ddcaae3.jpg)  
Fig. 1: Light-field missing view synthesis using 4 neighboring views supplied to our novel conditional GAN architecture.

To address the challenges of the LF missing view synthesis problem, this paper proposes a novel, neighbor-aware generative framework, as shown in Fig. 1. We argue that neighboring views capture relevant and accurate information, given that they share similar angular properties as the missed one. Moreover, our method is quite flexible/adaptive in choosing the 4 input views: any neighbor based on the availability along cardinal or diagonal axes within certain distance can be chosen. In contrast, [4] limits itself by having only corner views as input views. What if one of the corners is defective?

The main contributions of this work are listed below.

• We propose a new synthetic dataset consisting of 121 Light Field scenes, covering a variety of objects, to serve as a challenging benchmark.

![](images/162bd68e31261941b81a801b2576c8edcaae7fc53858320008c3b29e3bd67c7a.jpg)

![](images/1c3431be7e4bfbfb38546dcd8e71e52e79ed0b0923e97afc7bfe4916aaa22793.jpg)  
Fig. 2: Architecture of the proposed U-Net framework (a) U-Net Architecture (b) Model Structure

• We propose a novel generative framework for lightfield missing view restoration that relies on an adaptive selection of four neighboring views, making the system efficient and robust.

• We introduce a Positional Encoding Map as a key conditional input to our generator, providing an explicit geometric prior that significantly improves view synthesis consistency.

• We propose a novel 3D-2D fusion generator that effectively integrates multi-view information for high-fidelity image reconstruction.

## II. METHODOLOGY

## A. Approach Overview

The approach proposed to restore the missing or defective views in Light Field data, is based on our novel conditional Generative Adversarial Network (cGAN), which carefully leverages the geometric and photometric redundancy present in LF data. The process can be decomposed as follows: an adaptive neighbor selection mechanism with positional encoding, a tailored conditional GAN architecture, and a robust loss function for exhaustive training and implementation.

## B. Adaptive Neighbor Selection and Positional Encoding

Instead of indiscriminately using all the available input views from LF data, our framework initially starts with strategically selecting a geometrical optimal subset of input images. For a target missing view at angular coordinates $\mathbf { c } _ { t } ~ = ~ ( u _ { t } , v _ { t } )$ we adaptively identify four available neighboring SAIs and randomly choose one of the two configurations, ‘plus’ (+) or $\mathbf { \dot { x } } ^ { \mathbf { \gamma } } ( \times )$ , and choose neighbors within a certain neighborhood.

In Plus-Shape Configuration (+), the four input neighbors are selected along the cardinal axes, one from each direction (Right, Left, Up, Down), each at a random distance between 1 to 10. Similarly, in X-Shape Configuration (×), they are selected along the diagonal axes (Right-Up, Left-Up, Right-Down, Left-Down), each at a random distance between 1-10.

The selected four neighboring SAIs, $\mathbf { I } _ { i } ~ \in ~ \mathbb { R } ^ { 3 \times H \times W }$ for $i \in \{ 1 , 2 , 3 , 4 \}$ , are stacked along the channel dimension to form a 4D input tensor $\mathbf { X } \ \in \ \mathbb { R } ^ { 4 \times 3 \times H \times W }$ . We generate a Positional Encoding map as a Side Frame, P, which is a low resolution single channel image that encodes the coordinates of the selected neighboring SAIs as per the configuration and the missing target view. We utilize the Manhattan Distance $( D _ { m a n h a t t a n } )$ between the missing target view and its farthest neighbor to dynamically weight the reconstruction loss, enabling the network to focus and overcome the challenges in view synthesis.

![](images/b0189af26a407ffbd9aad1d8ca2b7d0a9cb4639aaca6c41424849ddab54049c8.jpg)  
Fig. 3: Central Views of scenes in our dataset

## C. Conditional Generative Adversarial Network (cGAN)

Our missing view restoration model is realized using a cGAN comprising of a specialized Generator G and a Patch-based Discriminator D. This architecture is inspired by the Pix 2 Pix framework, conditioned on the input stack X of SAIs and the positional encoding map P. Fig. 2 shows the architecture of the proposed U-Net framework.

1) Generator G: UNet with 3D-2D Fusion: The Generator G follows a modified U-Net architecture (UNetGenerator3DWithSideInput) designed to effectively fuse multi-view information and geometric constraints:

1) 3D Fusion Block: The input stack $\mathbf { X } \in \mathbb { R } ^ { B \times { 4 } \times { 3 } \times H \times W }$ is first permuted to $\mathbb { R } ^ { B \times 3 \times 4 \times H \times W }$ and processed by a 3D convolution layer with a kernel size of (4, 4, 4)

and a stride of (1, 2, 2). This operation compresses the 4 input views into a single, fused feature map $\mathbf { F } ~ \in ~ \mathbb { R } ^ { \sum _ { } ^ { } \times 6 4 \times H / 2 \times W / 2 }$ , effectively projecting the 4D LF data onto a 2D feature space while retaining multiview spatial correlations. This fused map F serves as the initial feature representation (e0).

2) Side Input Processing: The low-resolution Positional Encoding Map $\textbf { P } ( \in \bar { \mathbb { R } ^ { B \times 1 \times 5 1 \times 5 1 } } )$ is upsampled to the current feature map size $( 6 4 \times 6 4 )$ and processed by a 2-layer 2D convolutional block.

3) Adaptive Fusion in Encoder: The processed side input is concatenated with the features after the first 2D downsampling layer. This adaptive fusion at an intermediate feature stage ensures the geometric prior guides the multi-view feature integration.

4) U-Net Structure: The network then follows the standard U-Net encoding-decoding path with skip connections, ensuring detailed high-frequency information is preserved. The final output is the synthesized target view $\hat { \mathbf { I } } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$

2) Discriminator D: PatchGAN: We employ a PatchGAN Discriminator (PatchGANDiscriminator) to enforce local realism in the synthesized image. The discriminator takes the restored missing image <sup>ˆ</sup>I<sub>t</sub> (or the ground truth I<sub>t</sub>) and the conditional input, which is defined as the mean of the four neighboring SAIs $\begin{array} { r } { ( \mathbf { X } _ { a v g } = \frac { 1 } { 4 } \sum \mathbf { I } _ { i } ) } \end{array}$ , concatenated along the channel dimension. The discriminator outputs a probability map which indicates the realness of local image patches, thereby driving the generator to produce photometrically and structurally plausible results.

## D. Training and Loss Functions

The cGAN is trained end-to-end to minimize a composite loss function $\mathcal { L } _ { \mathcal { G } }$ for the Generator and $\mathcal { L } _ { D }$ for the Discriminator. 1) Discriminator Loss: The Discriminator loss $\mathcal { L } _ { D }$ is a standard adversarial loss based on Binary Cross-Entropy (BCE)[14]:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathcal { D } } = \frac { 1 } { 2 } [ { \mathbb { E } } _ { \mathbf { I } _ { t } } [ \log \mathcal { D } ( \mathbf { X } _ { a v g } , \mathbf { I } _ { t } ) ]  } \ ~ } \\ { { \displaystyle  + { \mathbb { E } } _ { \mathbf { X } , \mathbf { P } } [ \log ( 1 - \mathcal { D } ( \mathbf { X } _ { a v g } , \mathcal { G } ( \mathbf { X } , \mathbf { P } ) ) ) ] } ] } \end{array}\tag{1}
$$

2) Generator Loss: The Generator loss $\mathcal { L } _ { \mathcal { G } }$ is a weighted combination of adversarial, reconstruction, structural, and perceptual terms [9][14][15][16]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { G } } = \mathcal { L } _ { G A N } + \lambda _ { L 1 } \mathcal { L } _ { L 1 } + \lambda _ { S S I M } \mathcal { L } _ { S S I M } } \\ { + \lambda _ { P e r c } \mathcal { L } _ { P e r c } + \lambda _ { E d g e } \mathcal { L } _ { E d g e } } \end{array}\tag{2}
$$

1. Adversarial Loss $( \mathcal { L } _ { G A N } ) { : }$ This term encourages the synthesized image $\hat { \mathbf { I } } _ { t }$ to fool the Discriminator:

$$
\mathcal { L } _ { G A N } = \mathbb { E } _ { \mathbf { X } , \mathbf { P } } \left[ \log \left( 1 - \mathcal { D } ( \mathbf { X } _ { a v g } , \mathcal { G } ( \mathbf { X } , \mathbf { P } ) ) \right) \right]\tag{3}
$$

2. Weighted Reconstruction Loss $( \mathcal { L } _ { L 1 }$ and $\mathcal { L } _ { S S I M } ) \colon$ To enforce photometric accuracy, we use a combination of L1

![](images/d0798b572cf496612af8f2719255c70f0d8788f3c1c246ec65b50585e56bb1c2.jpg)  
Fig. 4: Visual Comparison for different Light Field Scenes

loss and Structural Similarity Index Measure (SSIM) loss, both weighted by the inverse difficulty of the synthesis task, represented by the Manhattan distance of the target view:

$$
\mathbf { W } = 1 . 0 + \lambda _ { d i s t } \cdot \frac { D _ { m a n h a t t a n } } { D _ { m a x } }\tag{4}
$$

• Weighted L1 Loss:

$$
\mathcal { L } _ { L 1 } = \mathbb { E } \left[ \mathbf { W } \cdot \left. \hat { \mathbf { I } } _ { t } - \mathbf { I } _ { t } \right. _ { 1 } \right]
$$

• Weighted SSIM Loss: SSIM is calculated on images normalized to [0, 1].

$$
\mathcal { L } _ { S S I M } = \mathbb { E } \left[ { \bf W } \cdot \left( 1 - { \bf S } { \bf S } \mathbf { I M } ( \hat { \mathbf { I } } _ { t } , \mathbf { I } _ { t } ) \right) \right]
$$

3. Perceptual Loss $( \mathcal { L } _ { P e r c } ) { : }$ To improve perceptual quality and visual plausibility, we introduce a VGG-19 based perceptual loss. This loss minimizes the L1 distance between the feature representations of the generated and ground truth images extracted from specific layers of a pre-trained VGG-19 network:

$$
\mathcal { L } _ { P e r c } = \sum _ { j } \left\| \phi _ { j } ( \hat { \mathbf { I } } _ { t } ) - \phi _ { j } ( \mathbf { I } _ { t } ) \right\| _ { 1 }\tag{5}
$$

where $\phi _ { j } ( \cdot )$ denotes the feature map extracted from the j-th selected layer of the VGG network.

4. Edge Loss $( \mathcal { L } _ { E d g e } )$ : During model evaluation, a recurring vertical blur artifact was observed in the generated images. To mitigate this and enforce sharper, more defined object boundaries, a dedicated Edge Loss was introduced. We use a pre-trained model, TEED, [9] as an expert edge extractor, to compute edge maps for both the generated image $( \hat { \mathbf { I } } _ { t } )$ and the ground truth target $\left( \mathbf { I } _ { t } \right)$ . The loss function adapts TEED’s proposed “Double Loss” $( \mathcal { L } _ { d l o s s } )$ , which combines two distinct components to ensure both the completeness and quality of the detected edges. A Weighted Cross-Entropy loss $( \mathcal { L } _ { w c e } )$ is applied to intermediate edge maps to encourage the detection of all possible edges, while a Tracing Loss $( \mathcal { L } _ { t r c g } )$ is applied to the final fused edge map to promote thinner and clearer boundaries. By minimizing the difference between the edge map hierarchies of the generated and target images, this loss provides a strong, explicit gradient that guides the generator to produce structurally coherent images with crisp outlines.

TABLE I: Quantitative comparison of on our and MPI datasets. The proposed model consistently achieves the best performance across all metrics.
<table><tr><td rowspan="2">Model</td><td colspan="3">Our Dataset</td><td colspan="3">MPI Dataset</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Baseline (Avg 4 Inputs)</td><td>0.6519</td><td>19.58</td><td>0.2582</td><td>0.6337</td><td>21.12</td><td>0.2778</td></tr><tr><td>3D U-Net</td><td>0.6792</td><td>13.73</td><td>0.2421</td><td>0.6232</td><td>17.73</td><td>0.2245</td></tr><tr><td>3D PatchGAN</td><td>0.6833</td><td>23.61</td><td>0.3298</td><td>0.6367</td><td>20.72</td><td>0.3472</td></tr><tr><td>3D PatchGAN + Side-Input 0.7244</td><td></td><td>22.43</td><td>0.1430</td><td>0.6712</td><td>21.34</td><td>0.1323</td></tr><tr><td>Proposed</td><td>0.7572</td><td>24.24</td><td>0.1023</td><td>0.7006</td><td>22.51</td><td>0.1184</td></tr></table>

The key loss weights used in the training are: $\lambda _ { L 1 } = 1 0 . 0$ $\lambda _ { S S I M } = 5 . 0 , \lambda _ { P e r c } = 1 . 0 ,$ , and $\lambda _ { d i s t } = 1 . 0 $

## III. EXPERIMENTS AND RESULTS

## A. Datasets and Implementaion Details

Datasets: We capture our light-field scenes using a costeffective system using a precision ball screw–based 2D actuation platform with a single moving camera. Designed for static scenes, the setup captures high-resolution images (51×51 views) with 2 mm horizontal and 4 mm vertical steps, controlled via Arduino and MATLAB for automated movement and sequential capture. Built using three industrial axes supporting up to 1 kg payload, the platform ensures high positioning accuracy. Calibration is performed by selecting four extreme corners of the scene to define the capture area, with horizontal and vertical steps computed for uniform coverage for better light field reconstruction and view synthesis. Fig. 3 provides Central Views of captured 121 Light Field Scenes. We also use MPI Light Field Intrinsic Dataset [13], containing complex scenes with varying depths and lighting.

All models were trained on an NVIDIA RTX A6000 GPU for 200 epochs using the Adam optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ and betas $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 5 , 0 . 9 9 9 )$ ). We used an 80/20 train-test split.

## B. Evaluation Metrics and Baselines

We employ three standard metrics to quantitatively evaluate the quality of the synthesized views against the ground truth images: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).

As a simple baseline for comparison, we compute the pixelwise average of the four selected neighboring input Sub-Aperture Images. This baseline helps to quantify the improvement gained from our learning-based approach over trivial interpolation.

## C. Ablation Study and Quantitative Analysis

To analyze the contribution of each key component in our framework, we performed an ablation study by systematically building up our final model. We started with a basic U-Net generator and progressively added the discriminator, the positional encoding side-input, and the full composite loss function. 3D U-Net: We first trained our generator architecture as a standalone model, using only the weighted $L _ { 1 }$ reconstruction loss. This configuration serves to evaluate the raw synthesis capability of the 3D-2D fusion U-Net. 3D PatchGAN: Next, we incorporated the PatchGAN discriminator and trained the model within the complete cGAN framework using an adversarial loss combined with the $L _ { 1 }$ loss. This step assesses the improvement in image realism and sharpness from adversarial training. 3D PatchGAN with Side-Input: We then integrated the Positional Encoding Map (P) as a conditional side-input to the generator. This variant tests our core hypothesis that providing explicit geometric guidance is crucial for accurate view synthesis. Proposed (Full Model): Finally, we trained our complete model as described in Section II-D, utilizing the full composite loss function which includes adversarial, weighted reconstruction $( L _ { 1 } + L _ { \mathrm { S S I M } } + L _ { \mathrm { e d g e } } )$ , and perceptual $\scriptstyle ( L _ { \mathrm { P e r c } } )$ terms. This represents our final, optimized framework.

## D. Discussion

The qualitative comparisons in Figs. 4 demonstrate the effectiveness of the proposed model in reconstructing missing light-field views. Our generated results preserve fine spatial details, exhibit sharper object boundaries, and maintain consistent illumination across views. The inclusion of positional encoding clearly enhances geometric accuracy, aligning the synthesized structures more faithfully with the ground truth. These improvements highlight the model’s ability to leverage multi-view and geometric cues for photometrically and structurally coherent view synthesis.

The quantitative results of our ablation study are summarized in Table I. The results clearly demonstrate the incremental benefits of each proposed component. The baseline averaging method yields poor results, highlighting the complexity of the task. While a basic 3D U-Net provides a reasonable starting point, incorporating the PatchGAN discriminator significantly boosts PSNR. The most substantial leap in performance, especially in SSIM and LPIPS, is observed upon introducing the positional encoding side-input, confirming its critical role in preserving geometric consistency. Our final model, trained with the comprehensive loss function, achieves the best scores across all metrics, producing images that are not only pixelwise accurate but also structurally coherent and perceptually convincing.

## IV. CONCLUSION

This paper addresses the challenge of restoring missing views in light-field camera arrays due to hardware failures. It introduces a novel conditional GAN framework that uses a Positional Encoding Map and a 3D-2D fusion generator to synthesize accurate and geometrically consistent views from sparse neighboring inputs. Extensive experiments show impressive results.

## REFERENCES

[1] Adelson, E.H., Bergen, J.R.: The plenoptic function and the elements of early vision. In: Computational Models of Visual Processing. pp. 3–20. MIT Press (1991).

[2] Levoy, M., Hanrahan, P.: Light field rendering. In: Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques. pp. 31–42 (1996).

[3] Levin, A., Durand, F.: Linear view synthesis using a dimensionality gap light field prior. In: Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. pp. 1831–1838. IEEE (2010).

[4] Kalantari, N.K., Wang, T.C., Ramamoorthi, R.: Learning-based view synthesis for light field cameras. ACM Transactions on Graphics (TOG) 35(6), 193 (2016).

[5] Ng, R., Levoy, M., Br´edif, M., Duval, G., Horowitz, M., Hanrahan, P.: Light field photography with a hand-held plenoptic camera. Computer Science Technical Report CSTR 2(11), 1–11 (2005)

[6] Wang, T.C., Efros, A.A., Ramamoorthi, R.: Depth estimation with occlusion modeling using light-field cameras. IEEE transactions on pattern analysis and machine intelligence (TPAMI) 38(11), 2170–2181 (2016).

[7] Liu, F., Hou, G., Sun, Z., Tan, T.: High quality depth map estimation of object surface from light-field images. Neurocomputing 252, 3–16 (2017).

[8] Wu, G., Zhao, M., Wang, L., Dai, Q., Chai, T., Liu, Y.: Light field reconstruction using deep convolutional network on epi. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1638–1646 (2017).

[9] Soria, X., Li, Y., Rouhani, M., Sappa, A. D. (2023). Tiny and Efficient Model for the Edge Detection Generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops (pp. 1356-1365).

[10] C. Jia, X. Zhang, S. Wang, S. Wang, and S. Ma, “Light field image compression using generative adversarial network-based view synthesis,”IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 177–189, 2019, doi: 10.1109/JET-CAS.2018.2886642.

[11] A. Wafa and P. Nasiopoulos, “Light Field GAN-based View Synthesis using full 4D information,” in Proc. ACM SIGGRAPH European Conf. on Visual Media Production (CVMP), London, U.K., Dec. 1-2, 2022, pp. 1-7, doi: 10.1145/3565516.3565519.

[12] Huang, F.C., Chen, K., Wetzstein, G.: The light field stereoscope: immersive computer graphics via factored near-eye light field displays with focus cues. ACM Transactions on Graphics (TOG) 34(4), 60 (2015)

[13] S. Shekhar, S. Beigpour, M. Ziegler, M. Chwesiuk, D. Palen, K.´ Myszkowski, J. Keinert, R. Mantiuk and P. Didyk, “Light-Field Intrinsic Dataset,” in British Machine Vision Conference (BMVC), Newcastle, UK, Sept. 3-6 2018, p. 120.

[14] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, “Generative Adversarial Networks,” arXiv preprint arXiv:1406.2661, 2014.

[15] J. Johnson, A. Alahi and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in European Conference on Computer Vision (ECCV), Amsterdam, Netherlands, Oct. 8-16 2016.

[16] L. Jiang, B. Dai, W. Wu, and C. C. Loy, “Focal Frequency Loss for Image Reconstruction and Synthesis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 1–10.