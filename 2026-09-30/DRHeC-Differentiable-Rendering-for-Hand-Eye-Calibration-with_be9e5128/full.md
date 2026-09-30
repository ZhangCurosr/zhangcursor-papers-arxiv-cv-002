# DRHeC: Differentiable Rendering for Hand-Eye Calibration with

RGB-Based Gradients

Xiaotian Zhang<sup>1</sup>, Yusheng Wang<sup>2</sup>, Naoya Kagawa<sup>3</sup>, Noritaka Takamura<sup>3</sup>, Keiji Okuhara<sup>3</sup>, Hiroyasu Baba<sup>3</sup>, Jun Ota<sup>2</sup>

Abstract—Accurate hand-eye calibration is crucial for precision manipulation. Traditional methods rely on markers, with their precision dependent on marker accuracy and observability. In contrast, markerless methods, such as learning-based approaches, use deep neural networks to directly extract keypoints or features from images, enabling the computation of hand-eye transformation with a single image and without the need for physical markers. However, these methods often face challenges related to dataset scale and data quality. Recently, differentiable rendering-based methods for hand-eye calibration have leveraged physical models to render binary masks and compare them with observations, enabling hand-eye calibration without fiducial markers in the calibration stage and providing interpretable optimization. While the state-of-the-art differentiable rendering methods achieve remarkable accuracy, the use of binary masks can result in the loss of internal profile details, reducing precision. Additionally, these methods are prone to convergence issues during optimization due to the lack of global information in the loss function, which can lead to local minima.

In this study, we propose a novel RGB-based differentiable rendering framework that provides richer geometric and appearance cues by incorporating color and mask geometric features, thereby improving calibration accuracy and optimization stability. Additionally, we propose a mask-guided image-to-image translation (I2IT) method to ensure explicit preservation of color and geometric consistency throughout the translation. Our approach is validated through both simulation and real-world experiments, with results demonstrating strong accuracy and robustness and clear improvements over existing differentiable rendering methods. Our method achieves a grasping success rate of 88.9% and insertion success rate of 57.4% on the UR5e real-world experiment, outperforming the state-of-the-art differentiable rendering hand-eye calibration method EasyHeC by 46.3 and 48.1 percentage points, respectively.

Index Terms—Hand-eye calibration, differentiable rendering, vision-guided robots (VGR), image-to-image translation (I2IT).

## I. INTRODUCTION

Robot manipulators are widely used across various fields, and vision-guided robots (VGR) are becoming increasingly popular for pick-and-place tasks due to their superior adaptability and ease of implementation [1]–[5]. A typical VGR employs cameras as sensors to provide feedback signals, allowing the robot to move precisely to target positions. To accurately determine an object’s 3D position and orientation within a robot’s workspace, it is crucial to establish the relative poses among several key frames: from the robot base frame to the end-effector frame, from the end-effector frame to the camera frame, and from the camera frame to the object or world frame [6]–[8]. Hand-eye calibration is essential for determining the transformation matrix between the end-effector and camera frames for robotic manipulators [9]–[12]. This calibration is essential for vision-guided tasks, as it aligns the camera’s coordinate system with the robot’s, enabling accurate visual data processing and precise robot control [13]–[15].

Hand-eye calibration methods can be broadly categorized into marker-based and markerless methods. Traditional marker-based methods rely on specific patterns, such as chessboards, AprilTags, or ChArUco markers, to determine the relative pose between the pattern frame and the camera frame [16]–[20]. Their accuracy depends on marker precision, measuring poses, and measurement quality, often requiring extensive design and tuning for specific tasks, which makes them time-consuming and expensive. Wang et al. employed dual matrix operators to express the hand-eye calibration equation in the dual quaternion domain, and proposed a new simultaneous method that solves rotation and translation jointly [21]. Jin et al. proposed Single3D, a unified singlemarker hand-eye calibration method that avoids intermediate pose estimation and achieves high accuracy and efficiency [2]. In contrast, markerless methods do not require markers, making them easier to execute in real-world robot manipulation tasks. These methods estimate camera poses by extracting features such as color and texture from the environment, often without prior knowledge of these features. Although there has been some research into markerless hand-eye calibration, most studies have focused on learning-based approaches that estimate keypoints and use PnP for pose estimation [22], [23]. These methods face several challenges, including dependence on dataset scale and quality, as well as susceptibility to noise. Recent studies have introduced differentiable rendering for markerless hand-eye calibration, leveraging physical models to enhance interpretability and generalization with less training data compared to keypoint-based methods [24]–[26]. However, differentiable rendering also faces challenges such as vanishing or exploding gradients in the absence of global gradient direction, which can lead to an optimization process that lacks sufficient guidance to accurately converge to the target pose. This instability can lead to inaccuracies in the handeye transformation matrix. Moreover, relying solely on binary masks in differentiable rendering is inherently problematic, as binary masks contain only silhouette information. This means that only boundary pixels provide useful gradients, while interior pixels contribute nothing, resulting in an underconstrained optimization. Pose changes that do not alter the silhouette produce zero gradients, which can lead to vanishinggradient regions and may also lead to pose ambiguity in some cases where different robot poses produce similar binary masks.

These limitations suggest that binary-mask-based supervision may be insufficient for robust and stable hand-eye calibration in some cases. To obtain informative and nondegenerate gradients, richer appearance cues such as RGB derivatives and additional global geometric constraints are required to make the optimization both precise and robust. This observation motivates our framework, which integrates RGBbased differentiable rendering with global geometric features to improve calibration accuracy and optimization stability under binary-mask-based supervision. This combination is not simply an addition of image features. By using local RGB appearance cues together with the mask centroid and area terms, it provides more reliable gradient guidance and improves optimization stability.

In this study, we propose a novel framework for handeye calibration that leverages differentiable rendering with RGB derivatives and mask geometric features to address these challenges. Given the need for a wide field of view, the simplicity of the experimental setup, and the fact that capturing the entire manipulator in the image improves the robustness of the differentiable rendering process, we primarily adopt the eye-to-hand configuration. Specifically, our main contributions are:

• We introduce an integrated differentiable rendering-based calibration framework that combines RGB derivatives with global geometric information, improving optimization accuracy and stability.

• Within this framework, we design a loss function that explicitly incorporates mask centroids, mask areas, and RGB derivatives to provide reliable gradient guidance.

• We develop a mask-guided I2IT method that preserves geometric consistency and improves real-to-simulation alignment under mild illumination, color, and material variations.

• We collect two real robot datasets, each with approximately 1,500 paired images from UR5e and DENSO VS060 for real-to-simulation, to train the real-to-sim translation models. Real-world experiments on grasping and insertion tasks demonstrate clear improvements over existing binary-mask-based differentiable renderer handeye calibration methods. Comparisons with marker-based baselines are also provided as reference baselines under the tested setup.

## II. RELATED RESEARCH

## A. Hand-Eye Calibration

The marker-based method tracks the relative motions of the camera and the robot’s end-effector, forming a loop closure equation to estimate the hand-eye transformation matrix. The commonly used formulation is:

$$
A X = X B ,\tag{1}
$$

where A and B represent the robot’s end-effector motion and the camera motion, respectively, and X is the desired hand-eye transformation matrix.

Depending on the robot setup, this equation can be adapted to account for additional coordinate transformations or external reference frames, such as $A X = Y B , A X B = Y C Z$ . Methods for solving the equation are typically classified into separate and simultaneous approaches [27]–[29]. Tsai et al. first compute the rotation using SVD, then estimate the translation using least squares [7], while Daniilidis uses dual quaternions to solve both rotation and translation simultaneously, offering greater robustness to noise [8]. However, the precision of these methods depends on marker accuracy, measurement poses, computational precision, and this process needs to be carried out before a manipulation task with a prepared marker.

Learning-based methods leverage deep learning to estimate the hand-eye transformation from the image. Lee et al. apply a Perspective-n-Point (PnP) algorithm to extract key points and compute the transformation [22], [23], while Valassakis et al. propose a neural network-based direct regression approach [30]. However, these approaches rely on large-scale, highquality datasets, which are costly to obtain and often lack transparency.

Recently, differentiable rendering-based hand-eye calibration has emerged as a promising alternative. These methods optimize the hand-eye transformation by comparing rendered and real images, computing gradients through a differentiable pipeline. While effective, existing differentiable rendering approaches face convergence issues and optimization errors, especially under limited binary-mask supervision. In the next section, we discuss the challenges of current differentiable rendering-based methods and introduce our improvements.

## B. Differentiable Renderer Hand-eye Calibration

Forward rendering involves generating a 2D image from a 3D model by considering various scene parameters such as shapes, materials, object poses, and lighting. Differentiable rendering extends the rendering process by enabling the computation of the gradient between the 2D image and the 3D scene, facilitating optimization tasks such as pose estimation. This technique has recently gained significant attention due to its ability to optimize 3D parameters in an end-to-end differentiable manner, making it a valuable tool in numerous applications [31]–[34]. One notable application is pose estimation, where differentiable rendering compares a rendered image with a real-world image to calculate the pose error. By iteratively optimizing the pose parameters, the rendered image progressively aligns with the real image, achieving precise pose estimation [35], [36].

![](images/45a74d1e2deeca57534969a0859669bd9f7b978af5b81b4c25c6cd10a9fd92b0.jpg)  
Fig. 1. Comparison of optimization processes using EasyHeC [25] (first row), IPE [24] (second row), and DRHeC (third row) on the synthetic UR5e dataset. All methods start from the same initial pose and target ground-truth pose. EasyHeC and IPE struggle to converge when the initial pose deviates significantly from the ground truth due to large or incorrect gradients, often leading to optimization failure. In those methods, the manipulator flies out of the frame during the optimization process. In contrast, the proposed DRHeC method effectively mitigates these issues and successfully converges.

Recent studies explore the use of differentiable rendering specifically for pose estimation in robotics. For example, Lu et al. proposed a method for precise pose estimation by employing differentiable rendering with distance maps and appearance differences, as well as reconstructing robot shapes from images using differentiable rendering in conjunction with a keypoint detector [24]. Additionally, EasyHeC proposes an approach for precise and automatic hand-eye calibration by combining differentiable rendering with space exploration techniques to optimize the calibration process [25], [26].

These methods depend on binary masks for differentiable rendering. However, binary masks often lose internal profile details, resulting in reduced precision and stability, and may also lead to pose ambiguity in some cases. The complexity of robotic models makes automatic differentiation difficult, often leading to inaccurate gradient estimations and suboptimal convergence, as is shown in Fig 1. To address these issues, we propose a novel approach that leverages RGB derivatives and mask geometric features such as centroid and area to enhance both robustness and accuracy.

## C. Image-to-Image Translation

Image-to-Image Translation (I2IT) maps images between source and target domains, evolving from early classificationand regression-based methods [37]–[39]. The advent of Generative Adversarial Networks (GANs) significantly advanced this field, enabling realistic image synthesis [40]. Conditional GANs, such as Pix2Pix and Pix2PixHD, further improved control over generation by leveraging paired training data [41]–[45]. However, acquiring such paired datasets remains challenging.

CycleGAN addresses this limitation by introducing cycle consistency loss, enabling unpaired image translation by enforcing bidirectional consistency [46]. Recent works extend CycleGAN across various aspects [47], such as Geo-MaskGAN, which incorporates segmentation masks to improve geometric consistency in I2IT [48]. While recent studies have mainly used diffusion models for I2IT [49]–[51], CycleGAN has been shown to be more effective than diffusion models when dealing with unpaired datasets. Our study focuses on translating robotic images from real to simulation domains, where discrepancies in geometric sizes pose challenges. To mitigate this, we enhance CycleGAN with a mask loss to reduce geometric inconsistencies and improve translation accuracy.

## III. PRELIMINARIES

Before presenting our method, we first review the foundational concepts of differentiable rendering.

## A. Differentiable Rendering

The forward rendering process is defined as:

$$
I = f ( \rho ) ,\tag{2}
$$

where I represents the rendered image, f defines the mapping from the 3D model to the 2D image, and $\rho$ denotes the scene

![](images/b49eec2b56a0b3b7eed20e8420a4e7624eaddc4a7ec78dcc534f08e51ec08cef.jpg)  
Fig. 2. The architecture of the differentiable renderer hand-eye calibration. The observed real image is first transformed into a simulation-like (sim-like) image using I2IT, which is then compared with the rendered image for differentiable renderer hand-eye calibration.

parameters. By differentiating Eq. (2) with respect to the scene parameters $\rho ,$ we have:

$$
\frac { \partial I } { \partial \rho } = \frac { \partial f ( \rho ) } { \partial \rho } .\tag{3}
$$

In inverse rendering, the goal is to recover the optimal scene parameters $\rho$ by minimizing the discrepancy between the rendered image $f ( \rho )$ and the observed image $I _ { \mathrm { o b s } }$ . This can be formulated as the following optimization problem:

$$
\hat { \rho } = \arg \operatorname* { m i n } _ { \rho } \mathcal { L } ^ { ( 2 ) } ( f ( \rho ) , I _ { \mathrm { o b s } } ) .\tag{4}
$$

where $\mathcal { L } ^ { ( 2 ) }$ denotes the pixel-wise squared error between the rendered image and the observed image.

## B. Differentiable Renderer Hand-Eye Calibration

For single-image hand-eye calibration, the EasyHeC [25] method can be simplified to differentiable rendering using a single binary image. Assuming that the robot model is known, and the robot base pose relative to the camera frame can be determined based on an initial estimate of the hand-eye transformation, which is roughly obtained through manual measurement or approximate calibration. Given this information, the binary image of the robot can be rendered as:

$$
I _ { b } = \mathcal { B } ( f ( q , ^ { c } T _ { b } ) ) ,\tag{5}
$$

where $I _ { b }$ is the binary rendered image, $\mathcal { B }$ is the binary mask extraction operator, and $f$ is the forward rendering function that maps the robot model to the image. $q$ and $^ c T _ { b }$ represent the scene parameters, with $q$ denoting the joint angles that define the robot configuration and $^ c T _ { b }$ representing the hand-eye transformation. In our formulation, the robot joint configuration $q$ is assumed to be known from the robot state and is not estimated during hand-eye calibration. Accordingly, the unknown variable to be optimized is the hand-eye transformation $^ c T _ { b }$ . The proposed differentiable renderer calibration framework does not require exact appearance consistency between the real robot and the simulation model, since the I2IT module is introduced to reduce the real-to-sim appearance gap. However, the robot geometry is still assumed to be accurate enough for the rendered observations to remain meaningfully aligned with the sim-like images.

The desired estimate of the hand-eye transformation can then be obtained by solving the following optimization problem until convergence:

$$
^ { c } \hat { T } _ { b } = \arg \operatorname* { m i n } _ { ^ { c } T _ { b } } \mathcal { L } _ { 2 } \left( I _ { b } , \mathcal { B } ( I _ { \mathrm { o b s } } ) \right) .\tag{6}
$$

To minimize the loss function, we employ the Adam [52] optimizer. By iteratively updating $^ c T _ { b } .$ , the rendered binary mask $I _ { b }$ is gradually aligned with the observed binary mask $\mathcal { B } ( I _ { \mathrm { o b s } } )$ , leading to an accurate estimate of the hand-eye transformation.

While EasyHeC is effective in many cases, it still faces several challenges. For instance, the lack of global gradient direction can lead to instability in the optimization process, causing vanishing or exploding gradients, which can even result in the rendered mask moving out of the frame. Additionally, since binary masks lack color information, the calibration precision is limited and may result in pose ambiguity in some cases.

## IV. METHOD

To overcome these issues, we reformulate the calibration objective by combining local RGB-based appearance information with global mask-geometric constraints. First, RGB derivatives provide dense local appearance cues that improve precision and reduce the risk of local minima compared with binarymask-based supervision. Second, centroid and area losses provide global geometric guidance for the rendered robot mask, which stabilizes the optimization when the initial pose error is large. In this sense, the proposed method does not simply add image features, but provides more reliable gradient guidance for stable optimization.

The proposed framework consists of two main components: RGB-based differentiable-renderer hand-eye calibration and geometry-preserving real-to-simulation I2IT. Compared with EasyHeC [25] and IPE [24], which mainly rely on silhouette cues derived from binary masks, the proposed framework additionally incorporates RGB appearance information and both mask centroid and area constraints to improve the stability of differentiable-rendering optimization. For the I2IT module, our method builds upon CycleGAN and introduces a mask loss to reduce geometric inconsistencies during image translation. Together, these components form a unified RGB-based differentiable-rendering framework for hand-eye calibration.

## A. I2IT

To leverage RGB derivatives as differentiable parameters, it is essential to compute the loss between the RGB mask and the rendered image. However, these two images often exhibit significant differences due to factors such as variations in model size, color, material properties, and lighting conditions. These differences arise from the variations between the real robot and the simulated model, making it challenging to directly compare the two images.

A feasible solution to address this challenge is the I2IT method, which transforms real-world robot images into simulation-like images, which are images that resemble those generated by rendering in simulation, as shown in the red section on the left of Fig. 2. First, the RGB mask is extracted from the real observed image, and then an I2IT transformation is applied to convert the RGB mask into a simulationlike image. However, it is well-known that no existing I2IT model can be generally applied across different types of robot manipulators for real-to-sim image translation. Consequently, a robot-specific I2IT model must be trained for each specific application. By minimizing the disparities between the rendered and real images, particularly when both share the same hand-eye transformation and robot configuration, this method enhances the robustness of the optimization process and improves the accuracy of the RGB loss calculation. In this work, we refine the CycleGAN method, where the generator’s loss function in the standard CycleGAN framework is defined as:

$$
\mathcal { L } = \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } ^ { ( 1 ) } + \lambda _ { \mathrm { G A N } } \mathcal { L } _ { \mathrm { G A N } } + \lambda _ { \mathrm { c y c l e } } \mathcal { L } _ { \mathrm { c y c l e } } ^ { ( 1 ) } ,\tag{7}
$$

where ${ \mathcal { L } } _ { \mathrm { i d } }$ represents the identity loss, ensuring that when images from the target domain are fed into the generator, their domain-specific features are preserved without unnecessary changes. ${ \mathcal { L } } _ { \mathrm { G A N } }$ denotes the adversarial loss, which ensures that the generated images are indistinguishable from real images in the target domain, while ${ \mathcal L } _ { \mathrm { c y c l e } }$ is the cycle consistency loss, which guarantees that an image can be reconstructed to its original form after being translated to the target domain and then back to the source domain. The coefficients $\lambda _ { \mathrm { i d } } , \lambda _ { \mathrm { G A N } } .$ and $\lambda _ { \mathrm { c y c l e } }$ serve as the weighting factors for each loss term.

Despite the I2IT process, minor pose misalignments may still arise due to configuration discrepancies between the real and simulation robot models. These misalignments can introduce errors in subsequent hand-eye calibration. To mitigate such effects, we introduce two practical measures for the I2IT training stage: (1) an offline preliminary hand-eye calibration with binary images to improve geometric alignment between the real and simulated models when preparing the I2IT training pairs, and (2) a mask loss term in the CycleGAN training process to enhance geometric consistency. The preliminary calibration is used only as a coarse data-alignment step for I2IT training, and it is not used to initialize or constrain the final DRHeC optimization. After the I2IT network is trained, its parameters are fixed, and no final DRHeC calibration result is fed back to retrain or update the I2IT module. In our current implementation, the pretrained I2IT module provides real-tosimulation appearance alignment while maintaining geometric consistency before RGB-based differentiable-renderer handeye calibration. The revised loss function is expressed as:

Algorithm 1 Hand-Eye Calibration via Differentiable Render  
ing with RGB and Mask Geometry Features   
Input: Observed image $I _ { r e f } ,$ initial hand-eye pose q<sub>init</sub>   
Output: Optimized hand-eye pose $q _ { o p t }$   
1: $q _ { o p t } \gets q _ { i n i t }$   
2: optimizer $ \mathrm { A d a m } ( [ q _ { o p t } ] , l r _ { b a s e } )$   
3: for $i = 1$ to $i t e r _ { m a x }$ do   
4: $I _ { o p t }  f ( q _ { o p t } )$   
5: $\boldsymbol { L _ { r g b } } \gets \mathbf { M S E } ( I _ { r e f } , I _ { o p t } )$   
6: L<sub>centroid</sub> ← centroid loss $\left( \mathcal { B } ( I _ { r e f } ) , \mathcal { B } ( I _ { o p t } ) \right)$   
7: $L _ { a r e a } \gets \mathrm { a r e a \_ l o s s } ( \mathcal { B } ( I _ { r e f } ) , \mathcal { B } ( I _ { o p t } ) )$   
8: if $L _ { c e n t r o i d } >$ threshold then   
9: $L _ { t o t a l } \gets L _ { r g b } + \lambda _ { c } L _ { c e n t r o i d } + \lambda _ { a } L _ { a r e a }$   
10: else   
11: $L _ { t o t a l }  L _ { r g b }$   
12: end if   
13: optimizer.zero grad()   
14: $L _ { t o t a l } .$ .backward()   
15: if $L _ { c e n t r o i d } >$ threshold then   
16: with $\mathtt { t o r c h . n o \_ g r a d ( ) }$ do   
17: $\nabla q _ { o p t } [ 3 : ] = 0$ {Freeze rotation gradient}   
18: end if   
19: optimizer.step()   
20: end for   
21: return $q _ { o p t }$

$$
\mathcal { L } = \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } ^ { ( 1 ) } + \lambda _ { \mathrm { G A N } } \mathcal { L } _ { \mathrm { G A N } } + \lambda _ { \mathrm { c y c l e } } \mathcal { L } _ { \mathrm { c y c l e } } ^ { ( 1 ) } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } ^ { ( 1 ) } .\tag{8}
$$

where the mask loss $\mathcal { L } _ { \mathrm { m a s k } } ^ { ( 1 ) } = \mathbb { E } [ \| \mathcal { B } ( G ( x ) ) - \mathcal { B } ( x ) \| _ { 1 } ]$ penalizes discrepancies between the binary masks of the generated images and those of the inputs. $\lambda _ { \mathrm { m a s k } }$ is a weighting factor that controls the contribution of the mask loss.

These improvements facilitate better geometric and color consistency, ensuring more accurate and reliable image translations between the real and simulated domains.

## B. RGB-Based Differentiable Rendering

As shown in Fig. 2, after transforming the real observed image into a simulation-like image, we can compare it with the rendered simulation image to infer the hand-eye transformation. To improve the precision of the hand-eye calibration, we initially attempt to use RGB derivatives instead of binary derivatives during calibration, as described by the following equation:

$$
^ { c } \hat { T } _ { b } = \arg \operatorname* { m i n } _ { ^ { c } T _ { b } } \mathcal { L } ^ { ( 2 ) } \left( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } \right) ,\tag{9}
$$

where $I _ { r g b }$ is the rendered RGB image. $I _ { \mathrm { s i m - l i k e } }$ is the simulation-like image obtained via the I2IT transformation.

However, based on experimental observations, we found that this approach does not yield satisfactory results. The use of RGB derivatives introduces more detailed color information, which leads to uneven gradient propagation compared to binary masks. This unevenness can cause the optimization to diverge and increase the likelihood of the solution going out of frame during differentiable rendering. To address this issue, we propose a novel method that introduces a loss function aimed at capturing the global gradient direction with respect to the robot mask. By combining the original RGB derivatives loss with the centroid distance loss and mask area loss, we aim to improve both calibration precision and optimization stability. The revised equation is shown as:

$$
\begin{array} { r l } & { c \hat { T } _ { b } = \arg \operatorname* { m i n } _ { c _ { T _ { b } } } \Bigg ( \lambda _ { \mathrm { r g b } } \mathcal { L } ^ { ( 2 ) } \left( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } \right) } \\ & { \quad \quad \quad \quad + \lambda _ { \mathrm { c e n t r o i d } } \mathcal { L } _ { \mathrm { c e n t r o i d } } ^ { ( 2 ) } \left( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } \right) } \\ & { \quad \quad \quad \quad + \lambda _ { \mathrm { a r e a } } \mathcal { L } _ { \mathrm { a r e a } } ^ { ( 2 ) } \left( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } \right) \Bigg ) . } \end{array}\tag{10}
$$

where $\mathcal { L } ^ { ( 2 ) } \left( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } \right)$ calculates the RGB loss between the rendered and the simulation-like images. $\mathcal { L } _ { \mathrm { c e n t r o i d } } ^ { ( 2 ) }$ represents the centroid distance loss between the binary masks of the two images, and $\mathcal { L } _ { \mathrm { a r e a } } ^ { ( 2 ) }$ computes their area loss. $\lambda _ { \mathrm { r g b } } , \lambda _ { \mathrm { c e n t r o i d } }$ , and $\lambda _ { \mathrm { a r e a } }$ define the weighting factors for each loss.

The centroid loss quantifies the Euclidean distance between soft-weighted centroids of the rendered and simulation-like images:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c e n t r o i d } } ^ { ( 2 ) } ( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } ) = \left\| \mathbf { c } ( I _ { r g b } ) - \mathbf { c } ( I _ { \mathrm { s i m - l i k e } } ) \right\| _ { 2 } , } \end{array}\tag{11}
$$

where $\mathbf { c } ( \cdot )$ denotes the soft centroid derived from the grayscale intensity, computed as the weighted average of pixel coordinates:

$$
\mathbf { c } ( I ) = \sum _ { p \in P } \frac { \exp { ( \mathcal { G } ( I ) [ p ] / \tau ) } } { \sum _ { r \in P } \exp { ( \mathcal { G } ( I ) [ r ] / \tau ) } } \cdot \mathbf { x } _ { p } ,\tag{12}
$$

where P denotes the set of all pixel indices in the image, and $p , r \in P$ serve as index variables. The vector $\mathbf { x } _ { p }$ is the 2D image coordinate of pixel $p .$ The denominator computes a softmax normalization over all pixels, ensuring that the weights sum to 1. τ is a temperature parameter, typically set to 1.0.

$$
\mathcal { G } ( I ) [ p ] = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } I [ p , k ] .\tag{13}
$$

where $K = 3$ is the number of RGB channels, and $k = 1 , 2 , 3$ indexes these color channels.

Compared to binary masks that assign equal weights to all object pixels and have hard edges causing non-smooth, nondifferentiable boundaries, soft weights provide a smooth, fully differentiable way to measure differences. This smoothness helps the centroid change continuously with small image changes, improving gradient flow during optimization and leading to more stable and accurate results.

The area loss is defined based on a smooth and differentiable approximation of the object area computed from the grayscale image.

$$
\mathbf { a } ( I ) = \sum _ { p } \sigma { \big ( } { \mathcal { G } } ( I ) [ p ] { \big ) } ,\tag{14}
$$

where $\sigma ( \cdot )$ is the sigmoid function and $\mathcal { G } ( I ) [ p ]$ denotes the grayscale intensity at pixel $p .$ Then area loss can be defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a r e a } } ^ { ( 2 ) } ( I _ { r g b } , I _ { \mathrm { s i m - l i k e } } ) = \left\| \mathbf { a } ( I _ { r g b } ) - \mathbf { a } ( I _ { \mathrm { s i m - l i k e } } ) \right\| _ { 2 } . } \end{array}\tag{15}
$$

In practice, we observed that during early optimization stages, large pose errors can make the calibration objective under-constrained and poorly conditioned, which can lead to divergence or poor convergence of the hand-eye pose estimation. To address this, we adopt a two-step optimization strategy: Initially, we freeze the gradients of the three Euler rotation variables when the centroid loss exceeds a threshold, indicating that the current estimate is still in the coarse-alignment stage, effectively restricting updates to the translation parameters. This allows the pose to move closer to the true position under the guidance of our improved combined loss function, stabilizing the optimization. Once the centroid loss falls below a predefined switching threshold, these three rotation gradients are unfrozen for fine-tuning using the full RGB loss, marking the transition to the fine-refinement stage and enabling precise rotation refinement. This staged approach improves stability and accuracy by preventing erratic updates in rotation during early optimization. The switching threshold controls the trade-off between optimization stability and refinement efficiency. If the threshold is too small, rotation updates are enabled only after the centroid misalignment becomes very small, which improves early-stage stability but may delay rotation refinement and slow convergence. If the threshold is too large, rotation updates are enabled earlier, which may accelerate refinement but can also expose the optimization to noisy or misleading rotation gradients when the pose is still poorly aligned. Since the calibration module does not feed its outputs back into the I2IT module, the overall pipeline does not introduce recursive feedback between I2IT training and hand-eye calibration. This approach effectively stabilizes early optimization and improves final accuracy. Pseudocode for this procedure is provided in Algorithm 1.

In contrast to the IPE method [24], which relies on the rendered image mask multiplied by a distance map, this results in a non-smooth loss function. To address this issue, the gradient at the edge requires special processing [53], but the current implementation still faces gradient issues. Moreover, for non-convex robot configurations, the gradient directions can become inconsistent, further hindering convergence. To overcome these issues, our method incorporates a differentiable mask centroid loss that facilitates smoother gradient propagation and is more robust to shape complexity. However, while the centroid loss effectively aligns the mask centers, it may not always precisely correspond to the object’s true center, potentially resulting in misalignment along the z-axis. To address this, we incorporate an area loss to stabilize ztranslation by enforcing mask area consistency.

## C. Implementation Details

The key hyperparameters in our optimization process were determined through preliminary small-scale experiments to ensure stable convergence and reliable performance. For the I2IT training stage, we follow the original CycleGAN settings by keeping the weights $\lambda _ { \mathrm { i d } } , \lambda _ { \mathrm { G A N } }$ , and $\lambda _ { \mathrm { c y c l e } }$ unchanged, and set $\lambda _ { \mathrm { m a s k } } = 1 . 0$ . The network is trained for 400 epochs, with a constant learning rate for the first 200 epochs, followed by a linear decay over the remaining epochs. For the differentiable rendering-based optimization, we use the Adam optimizer with a fixed learning rate of 0.002 and a maximum of 2000 iterations. The loss weights are set as follows: $\lambda _ { \mathrm { r g b } } = 1 . 0$ $\lambda _ { \mathrm { c e n t r o i d } } = 0 . 0 1$ , and $\lambda _ { \mathrm { a r e a } } = 0 . 0 0 0 1$ . These values were chosen to reflect the intended roles of each loss component: the RGB loss functions as the dominant optimization signal, the centroid distance loss introduces geometric regularization, and the mask area loss enforces additional shape consistency. We set the threshold for the two-step optimization to 0.25. This value serves as a practical criterion for controlling the transition from coarse translation-dominant alignment to full pose refinement. As shown in the ablation study, the method performs consistently within a nearby range of threshold values, indicating that the optimization is not strongly dependent on this exact value. Note that the threshold and loss weights may require tuning for substantially different environments or datasets. Imbalanced settings could lead to optimization instability or suboptimal convergence, highlighting the importance of appropriate loss balancing in the rendering-based calibration process.

TABLE I  
EXPERIMENTAL RESULTS ON THE SYNTHETIC UR5E DATASET.
<table><tr><td rowspan="2">Perturbation Level</td><td rowspan="2">Method</td><td colspan="4">Translation Error (meters)</td><td colspan="4">Rotation Error (radians)</td><td rowspan="2">Success Rate (%)</td></tr><tr><td>Q1</td><td>Q3</td><td>Max</td><td>Mean</td><td>Q1</td><td>Q3</td><td>Max</td><td>Mean</td></tr><tr><td rowspan="5">Low (LP)</td><td>EasyHeC [25]</td><td>0.0002</td><td>0.0005</td><td>1.1282</td><td>0.0089</td><td>0.0010</td><td>0.0029</td><td>0.7012</td><td>0.0122</td><td>91.5</td></tr><tr><td>IPE [24]</td><td>0.0008</td><td>0.0021</td><td>0.0084</td><td>0.0016</td><td>0.0034</td><td>0.0100</td><td>0.0693</td><td>0.0095</td><td>100.0</td></tr><tr><td>RGB</td><td>0.0002</td><td>0.1037</td><td>1.8703</td><td>0.3419</td><td>0.0008</td><td>0.4508</td><td>1.7163</td><td>0.3254</td><td>79.0</td></tr><tr><td>RGB+Mask Centroid</td><td>0.0001</td><td>0.0003</td><td>0.0012</td><td>0.0003</td><td>0.0007</td><td>0.0016</td><td>0.0080</td><td>0.0013</td><td>100.0</td></tr><tr><td>DRHeC</td><td>0.0002</td><td>0.0003</td><td>0.0013</td><td>0.0003</td><td>0.0006</td><td>0.0015</td><td>0.0053</td><td>0.0012</td><td>100.0</td></tr><tr><td rowspan="5">Medium (MP)</td><td>EasyHeC [25]</td><td>0.0005</td><td>0.3570</td><td>1.8940</td><td>0.2815</td><td>0.0016</td><td>0.4274</td><td>1.8958</td><td>0.2899</td><td>91.5</td></tr><tr><td>IPE [24]</td><td>0.0034</td><td>0.1573</td><td>1.4977</td><td>0.1469</td><td>0.0121</td><td>0.3683</td><td>1.4743</td><td>0.2719</td><td>96.5</td></tr><tr><td>RGB</td><td>0.7112</td><td>1.6945</td><td>2.1994</td><td>1.2572</td><td>0.5230</td><td>1.3933</td><td>1.7245</td><td>1.0278</td><td>28.5</td></tr><tr><td>RGB+Mask Centroid</td><td>0.0002</td><td>0.2933</td><td>1.5972</td><td>0.1862</td><td>0.0007</td><td>0.2041</td><td>1.9684</td><td>0.1703</td><td>99.0</td></tr><tr><td>DRHeC</td><td>0.0002</td><td>0.0009</td><td>1.9937</td><td>0.0766</td><td>0.0007</td><td>0.0031</td><td>1.7313</td><td>0.0930</td><td>99.0</td></tr><tr><td rowspan="5">High (HP)</td><td>EasyHeC [25]</td><td>0.0293</td><td>1.3432</td><td>1.9862</td><td>0.6572</td><td>0.0709</td><td>1.3728</td><td>1.8319</td><td>0.6508</td><td>71.0</td></tr><tr><td>IPE [24]</td><td>0.0729</td><td>0.4965</td><td>1.8722</td><td>0.4055</td><td>0.1910</td><td>0.9723</td><td>1.9769</td><td>0.5866</td><td>83.5</td></tr><tr><td>RGB</td><td>1.3399</td><td>1.8105</td><td>2.1614</td><td>1.5204</td><td>1.1760</td><td>1.3681</td><td>1.8452</td><td>1.1892</td><td>11.0</td></tr><tr><td>RGB+Mask Centroid</td><td>0.0005</td><td>0.4068</td><td>1.8095</td><td>0.3323</td><td>0.0012</td><td>0.3923</td><td>2.7546</td><td>0.3124</td><td>97.5</td></tr><tr><td>DRHeC</td><td>0.0005</td><td>0.2103</td><td>1.6269</td><td>0.1495</td><td>0.0011</td><td>0.2351</td><td>2.0007</td><td>0.2187</td><td>99.5</td></tr></table>

## V. SIMULATIONS

To evaluate the effectiveness of the proposed method, we conduct simulation experiments on both synthetic and realworld datasets. The synthetic dataset allows us to compare our approach with differentiable rendering-based methods under various initial pose errors, while the real-world dataset enables direct evaluation against learning-based hand-eye calibration approaches.

## A. Synthetic Datasets

In this subsection, we conduct studies on the synthetic UR5e dataset to evaluate the individual contributions of key components in our proposed framework. We generate a synthetic dataset using a UR5e robot to systematically evaluate our method under controlled conditions. For each synthetic sample, the robot joint configuration is generated by sampling around the UR5e robot pose $\left[ 0 , - 9 0 ^ { \circ } , 9 0 ^ { \circ } , - 9 0 ^ { \circ } , - 9 0 ^ { \circ } , 0 \right]$ , with an independent perturbation of $\pm 1 0 ^ { \circ }$ applied to each joint.

![](images/2d1a4167aceb990aa24211ec99c231d063cb53a8bf85a2912c20f81167f6b90d.jpg)

![](images/8ac2ae822bfa49a835dedef3df755385812b493ff43d23061b539855e3abf989.jpg)  
(a)

![](images/38259cc13f936e3420c7e4ddb390c38055679d2820d1b446e015e34a5b802ff6.jpg)

![](images/2bec40c26ab1f77ed2aced22468f49b5a3fe179c949f239ae9080765cf60fba4.jpg)  
(b)

![](images/90ad10f41f7d63bd9690e356b6c772ed9ea15f7cc24aa644489b4d450ed3159d.jpg)

![](images/ffe4f5dae7fef60427e75949e76ce8ab8b9c7b3bd57d7cb9de27f4490ebee21f.jpg)  
(c)  
Fig. 3. Distributions of calibration errors under varying levels of initial perturbations for different differentiable rendering methods: EasyHeC [25], IPE [24], RGB, RGBC (RGB and Mask Centroid), and DRHeC. (a) Low perturbations, (b) Moderate perturbations, (c) High perturbations.

To ensure a comprehensive assessment, we generate 200 test samples for each perturbation level, randomly selected within predefined error ranges. Three experimental settings are designed to simulate different levels of initial misalignment: (LP) low translation errors within [−0.1,0.1] meters and rotation errors within $[ - \pi / 1 8 , \pi / 1 8 ]$ radians, (MP) moderate translation errors from $[ - 0 . 2 , - 0 . 1 ]$ to [0.1, 0.2] meters, and (HP) high translation errors from [−0.3, −0.2] to [0.2, 0.3] meters, while the rotation error range remains fixed at $[ - \pi / 1 8 , \pi / 1 8 ]$ radians. These settings allow us to evaluate the precision and robustness of each method under varying levels of misalignment. We compare five rendering-based calibration methods: EasyHeC [25], IPE [24], the RGB loss method, the RGB loss combined with centroid distance loss method (RGBC), and the DRHeC method. Table I provides the quantitative results, while Figs. 3 (a), 3 (b), and 3 (c) visualize the corresponding outcomes.

Overall, the results demonstrate that adding the mask centroid loss to the RGB loss significantly reduces translation and rotation errors, confirming the importance of geometric constraints in guiding optimization. Further inclusion of the mask area loss in DRHeC yields additional improvements in error metrics and notably increases success rates, especially under challenging high-perturbation conditions. This confirms that leveraging both RGB-based gradients and mask-derived geometric information enhances the accuracy and robustness of hand-eye calibration.

More specifically, DRHeC consistently achieves the lowest Q3 (third quartile) and mean translation and rotation errors across all perturbation levels, demonstrating superior robustness to large initial misalignments compared to all baselines. In particular, under MP and HP settings, DRHeC consistently outperforms other methods, demonstrating superior robustness against large initial misalignments. Although the IPE method exhibits a slightly lower maximum rotation error in the MP setting, this can be attributed to the strong local constraints imposed by its distance map loss. However, this advantage is offset by its higher mean and Q3 values compared to DRHeC, indicating less consistent overall performance.

Moreover, the inclusion of mask area gradients further improves performance, as evidenced by the increase in success rates from 97.5% (RGBC) to 99.5% (DRHeC) in the HP setting. This result highlights the advantage of leveraging both RGB-based gradients and mask-derived geometric constraints, which enhances stability and accuracy.

## B. Real World Datasets

To further validate our approach in real-world conditions, we use the real-world Baxter dataset from [22]. This dataset allows direct comparison with EasyHeC and other learningbased hand-eye calibration methods. It consists of 100 images captured from 20 distinct robot poses, with each image annotated with GT (Ground Truth) keypoints for quantitative evaluation.

Following the EasyHeC setup, we manually initialize the camera pose relative to the Baxter robot. The collected images are used to construct I2IT training pairs through an offline coarse alignment procedure. This alignment step is used only to prepare the I2IT training pairs and is not part of the final DRHeC optimization. The trained I2IT model then transforms real images into a simulation-like style, reducing the appearance gap between real and simulated robot images while preserving geometric consistency. Finally, we apply the DRHeC method to calibrate the hand-eye pose and evaluate the results. Together with the UR5e and DENSO VS060 exper iments presented later, these results provide initial, although not exhaustive, evidence of the applicability of the proposed framework across different robot morphologies, visual appearances, and data sources.

TABLE II  
EVALUATION RESULTS OF 2D PCK ON THE REAL-WORLD DATASET.
<table><tr><td>Method</td><td>20px</td><td>30px</td><td>40px</td><td>50px</td><td>100px</td><td>150px</td><td>200px</td></tr><tr><td>Dream [23]</td><td>0.16</td><td>0.23</td><td>0.29</td><td>0.33</td><td>0.52</td><td>0.62</td><td>0.64</td></tr><tr><td>OK [22] IPE (box) [24]</td><td>0.34</td><td>0.54</td><td>0.66</td><td>0.69</td><td>0.88</td><td>0.93</td><td>0.95</td></tr><tr><td></td><td>1</td><td>1</td><td></td><td>0.65</td><td>0.94</td><td>0.95</td><td>0.95</td></tr><tr><td>IPE (cylinder) [24]</td><td></td><td></td><td></td><td>0.80</td><td>0.91</td><td>0.93</td><td>0.95</td></tr><tr><td>IPE (CAD) [24]</td><td></td><td></td><td></td><td>0.74</td><td>0.90</td><td>0.95</td><td>0.95</td></tr><tr><td>EasyHeC [25]</td><td>0.35</td><td>0.55</td><td>0.75 0.75</td><td>0.90</td><td>0.95</td><td>0.95</td><td>1.00</td></tr><tr><td>EasyHeC++ [26]</td><td>0.50</td><td>0.75</td><td></td><td>0.85</td><td>0.90</td><td>0.95</td><td>1.00</td></tr><tr><td>DRHeC</td><td>0.65</td><td>0.75</td><td>0.90</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

The evaluation employs the 2D PCK (Percentage of Correct Keypoints) metric, which quantifies the proportion of estimated keypoints that fall within a specified pixel distance from the GT keypoints. Results are evaluated across multiple pixel thresholds (from 20px to 200px) to offer a comprehensive view of performance at various accuracy levels. As shown in Table II, our proposed method consistently outperforms both learning-based and the differentiable renderer hand-eye calibration methods.

Notably, DRHeC outperforms other methods across all pixel thresholds, suggesting that our method not only improves calibration accuracy but also enhances robustness to small perturbations in real-world data. This demonstrates that our method offers both superior accuracy and greater stability for hand-eye calibration.

## VI. REAL-WORLD EXPERIMENTS

To validate the proposed method, we conduct two types of real-world experiments. We primarily evaluate our approach on the UR5e robot, as described in Sec. VI-A through Sec. VI-D. Additionally, we test our method on a DENSO VS060 robot, which has a nearly colorless appearance with the entire robot being white except for its black end-effector, to further demonstrate the effectiveness of our approach in scenarios with minimal color information (Sec. VI-E). Finally, we present a discussion in Sec. VI-F.

## A. Experiment Setup

We conduct real-world experiments using a UR5e robot (Universal Robots A/S, Odense, Denmark). The experimental setup includes an industrial CMOS camera, the Basler acA2440-20gc (Basler AG, Schleswig-Holstein, Germany), equipped with a fixed-focus lens, the Kowa LM8JC10M (Kowa Optronics Co., Ltd., Aichi, Japan). To ensure the camera can capture the entire UR5e robot within its field of view, the camera tripod is positioned approximately 2 meters away from the UR5e robot along the x-axis, with both remaining at roughly the same horizontal plane. It is assumed that both the camera and the robot are well calibrated. For differentiable rendering, we utilize nvdiffrast [54] as the rendering engine, while image segmentation is performed using SAM [55]. SAM is used only to generate the initial mask proposal, with manual prompt selection and mask correction when necessary, in order to ensure reliable masks and to prevent segmentation errors from influencing the evaluation of the proposed calibration framework. This mask-preparation step is not part of the differentiable-renderer-based handeye optimization itself, and the segmentation module can be replaced by another task-specific automatic method without changing the proposed calibration framework. All experiments are conducted on an NVIDIA RTX 4090 GPU. All real-world experiments were performed under stable indoor lighting, with the workspace illuminated by fixed overhead LED lights, ensuring approximately constant illumination across all trials.

![](images/0515e280eff21438679480cd275a3b6d91ba02849775d3338121d2761bd58d82.jpg)  
(a)

![](images/bf1a717f9f7d4ed057d8356e0545d9b65e816cec9de1baed9468a6b16bd32231.jpg)  
(b)  
Fig. 4. Experimental setup for two robot tasks. (a) Grasping experiment to verify the robot’s ability to grasp an object after hand-eye calibration. (b) Insertion experiment to verify the robot’s ability to accurately insert into a hole after hand-eye calibration.

For this study, we selected two representative tasks that highlight key capabilities required for robotic manipulation: grasping and insertion, as shown in Fig. 4. These tasks are sensitive to hand-eye calibration accuracy. Grasping requires precise positioning to avoid missing the object, while insertion demands accurate alignment. Thus, both serve as effective benchmarks for evaluating calibration performance. To evaluate the proposed method, we compared it against several baseline approaches, including two classical marker-based methods (Tsai’s [7] and Daniilidis’ [8]), which are widely used in industrial settings, as well as two recent marker-based methods, Single3D [2] and Wang et al. [21], for a more comprehensive academic comparison. We also include the differentiable rendering method EasyHeC [25] as a baseline. In both tasks, the object pose relative to the camera frame was estimated using 12 cm AprilTags and the PnP algorithm. These AprilTags were used only as an auxiliary evaluation setup for object pose estimation in the downstream validation experiments, and this evaluation setup was applied uniformly to all compared methods. They were not used in the proposed differentiable-renderer-based hand-eye calibration procedure. Based on this pose estimation, the gripper then calculated the object’s position relative to the robot’s base frame. Notably, in the grasping experiments, object collision was not considered, making precise hand-eye calibration essential for achieving

![](images/ef3e796ca038b96fbbd958c6e9aa49e78c5b74fef2c58c48ebcda35c281bd82f.jpg)  
Fig. 5. Example of real-world image translated into a simulation-like image for differentiable rendering-based hand-eye calibration.

successful grasps.

## B. I2IT Preprocessing

Prior to performing DRHeC, the captured images must be transformed into the simulation-image domain, as shown in Fig. 2. To train the I2IT network, we utilized a dataset comprising 10 robot configurations, each associated with 147 camera poses, resulting in a total of 1,470 images. Robot masks were extracted from the observed images, and these RGB masks were paired with their corresponding simulation images to train the I2IT model. While paired datasets are not strictly required for I2IT training, empirical evidence suggests that paired datasets often lead to improved performance. To refine the training dataset, a preprocessing step was introduced, as the binary masks share the same mask area as the RGB masks. In this process, a preliminary binary-mask-based calibration was used only to improve the alignment between the observed images and the rendered simulation images when constructing the I2IT training pairs, thereby ensuring more accurate color correspondence. This preprocessing calibration was not used in the subsequent DRHeC optimization. The realto-simulation paired dataset collected and annotated from the UR5e experiments is used as one dataset in this work, while the additional DENSO dataset described in Section VI.E serves as another real-world dataset for I2IT training and evaluation.

The output of the I2IT network is shown in Fig. 5. All 10 robot configurations were successfully transformed from the real-image domain to the simulation-image domain, preserving geometric information while altering only color correspondences. There were no changes to the poses, apart from minor discrepancies resulting from mask extraction and slight differences in the model configurations between the real and simulation robots.

TABLE III  
PERFORMANCE OF I2IT METHODS IN THE REAL-WORLD EXPERIMENT.
<table><tr><td>Method</td><td>MSE</td><td>MAE</td><td>IoU</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>CycleGAN [46]</td><td>294.35</td><td>3.19</td><td>0.9300</td><td>23.68</td><td>0.9725</td></tr><tr><td>CUT [47]</td><td>354.60</td><td>3.46</td><td>0.9284</td><td>22.76</td><td>0.9680</td></tr><tr><td>GeoMaskGAN [48]</td><td>304.74</td><td>3.26</td><td>0.9286</td><td>23.54</td><td>0.9705</td></tr><tr><td>DRHeC (ours)</td><td>292.12</td><td>3.19</td><td>0.9350</td><td>23.70</td><td>0.9722</td></tr></table>

As shown in Table III, the proposed DRHeC method outperforms the state-of-the-art I2IT approaches across most evaluation metrics. While its SSIM score reaches 0.9722, slightly lower than that of CycleGAN at 0.9725, DRHeC achieves the lowest MSE and MAE, the highest PSNR, and

Pose1

Pose2

![](images/b16ba61f6967d2844f2f221c672987de06044ce5e1058a4790d85a4ae20215dd.jpg)  
RGB Mask

![](images/55e06a90d05ee88151c1f5e46b0e6eb48069e24d18bf3a52170d10f5c82153bb.jpg)  
CycleGAN

![](images/bcaed063a5a761786d249929029de0d8a5b5709bf8a7a9aace1660c384b35819.jpg)  
CUT

![](images/a5c9d5f01f1b3ab12950c7441e6f7cd1f4bd014a82b344256897da0dee572ca6.jpg)  
GeoMaskGAN

![](images/3f0c0ba0ed6410371492b87ebb0d260f4766b59fb7e91ab6f015dc7f08911bce.jpg)  
DRHeC

Fig. 6. Visual comparison of real-world RGB images and predicted simulation-like images from different I2IT methods across robot poses. DRHeC shows better performance in color consistency for challenging robot configurations compared to other methods.

a notably higher mask IoU, indicating superior geometric consistency. In addition to these quantitative results, the visual comparisons in Fig. 6 further demonstrate that DRHeC more accurately preserves both the appearance fidelity and geometric structure of simulation-style images. Notably, our method maintains consistent color translation even under challenging robot poses, while other methods often fail around joint regions. This robust I2IT performance provides a reliable image translation backbone and facilitates accurate downstream RGB-based differentiable renderer hand-eye calibration.

## C. Grasping Experiments

Grasping tasks are essential in many robotic applications, where accurate hand-eye calibration is crucial for enabling the robot to effectively pick and place objects. The accuracy of the calibration method directly impacts the robot’s ability to estimate the object’s pose and perform a successful grasp. Therefore, the grasping experiments were designed to evaluate the performance of different calibration methods in guiding the robot to grasp objects accurately.

For the differentiable rendering-based method, we employed two types of robot configurations, three marker positions, and nine camera poses, resulting in a total of 54 combinations. In comparison, the marker-based method uses images captured from 10 different robot poses, leading to 18 trials. Three marker positions were used in each trial, yielding another 54 combinations, allowing for a comparison with the differentiable rendering-based approach. The results of the grasping experiments are summarized in Table IV.

The experimental results show that DRHeC substantially improves over the differentiable renderer hand-eye calibration baseline EasyHeC. Compared with the closed-form markerbased baselines included in this setup, DRHeC also shows strong performance, while Single3D (iterations) still achieves the highest success rate when fiducial markers are available.

Under our experimental setup, when using 10 images for calibration, the traditional marker-based methods (Tsai et al. and Daniilidis et al.) achieve success rates of 5.6% and 9.3%, respectively. The recent closed-form marker-based methods, Single3D (closed-form) and Wang et al., also exhibit low success rates, achieving only 3.7% and 24.1% success rates, respectively. In contrast, the iterative variant Single3D (iterations) performs substantially better and achieves the highest success rate among all methods. The relatively low success rates of some marker-based baselines may be associated with the specific imaging conditions in our setup, particularly the approximately 2 m distance between the camera and the robot base. At this distance, the fiducial markers occupy relatively small image regions, which may reduce the accuracy of marker detection and pose estimation. Moreover, their performance may also be affected by measurement-pose selection and marker precision in our experimental setup. The strong performance of Single3D (iterations) suggests that its iterative refinement is beneficial under the tested conditions.

In contrast, the differentiable rendering method does not rely on fiducial markers during hand-eye calibration. Instead, it leverages the robot’s own geometry for hand-eye calibration, where pose estimation is determined using the entire robot mask rather than relying on specific corner points or feature points. While the EasyHeC method performs better with a success rate of 42.6%, it still falls short of optimal performance due to the limitations of binary-mask-based supervision, which provides limited geometric information and lacks discriminative appearance cues. In comparison, DRHeC achieves a success rate of 88.9%, substantially improving over EasyHeC under this experimental setup. It also outperforms several closed-form marker-based baselines, although Single3D (iterations) remains the strongest overall method in these grasping trials. These results support the effectiveness of the proposed markerless hand-eye calibration under the tested conditions. Although Single3D (iterations) attains the highest overall success rate, it relies on fiducial-marker observations. In contrast, DRHeC achieves strong performance without relying on fiducial markers for solving the hand-eye calibration problem, although AprilTags are used only for downstream object-pose estimation in the task evaluation setup. This demonstrates the effectiveness of the proposed differentiable-renderer hand-eye calibration framework.

TABLE IV  
SUCCESS RATES OF DIFFERENT EXPERIMENTS ON UR5E ROBOT.
<table><tr><td rowspan="2">Method</td><td colspan="5">Marker-based</td><td colspan="2">Differentiable-rendering-based (Markerless)</td></tr><tr><td>Tsai et al. [7]</td><td>Daniilidis et al. [8]</td><td>Single3D (closed-form) [2]</td><td>Single3D (iterations) [2]</td><td>Wang et al. [21]</td><td>EasyHeC [25]</td><td>DRHeC</td></tr><tr><td>Grasping</td><td>5.6%</td><td>9.3%</td><td>3.7%</td><td>96.3%</td><td>24.1%</td><td>42.6%</td><td>88.9%</td></tr><tr><td>Insertion</td><td>5.6%</td><td>9.3%</td><td>1.8%</td><td>81.5%</td><td>16.7%</td><td>9.3%</td><td>57.4%</td></tr></table>

## D. Insertion Experiments

Insertion tasks are also critical for robot manipulators, as they require precise alignment and positioning to ensure successful interaction with objects. Such tasks are common in manufacturing, assembly lines, and maintenance operations, where robots are required to insert components into specific slots or holes with high accuracy. The ability to perform these tasks efficiently and precisely is a key indicator of a robot’s manipulation capabilities, especially when dealing with complex geometries and tight tolerances. In this set of experiments, we assessed the capability of each method to achieve accurate and successful insertion. The setup for the insertion experiment follows the same configuration as the grasping task.

As shown in Table IV, the success rates for the insertion experiment follow a similar trend to those observed in the grasping task. The marker-based methods (Tsai et al., Daniilidis et al., Single3D (closed-form), Wang et al.) again show low success rates of 5.6%, 9.3%, 1.8% and 16.7%, respectively. Single3D (iterations) again maintains the highest success rate, consistent with its performance in the grasping task. The EasyHeC method also performs poorly, with a success rate of 9.3%. In the insertion task, DRHeC achieves 57.4%, showing a substantial improvement over EasyHeC and higher success rates than several closed-form markerbased baselines under the tested setup, while remaining below Single3D (iterations). The lower success rate compared to the grasping task is due to the increased difficulty of the insertion task, which requires more precise alignment. Additionally, we apply a stricter criterion for success in insertion: any collision with the object is considered a failure. These factors contribute to the lower overall success rate in the insertion experiment.

Although both the Daniilidis et al. method and the Easy-HeC method yield the same 9.3% success rate, they exhibit a notable difference in performance. The EasyHeC method results in only a slight deviation from the target, whereas the marker-based methods, particularly the Tsai and Daniilidis approaches, tend to produce much larger errors, especially in translation.

To better understand the calibration differences among the compared methods, we further conduct a numerical analysis on one representative trial. Among all methods, Single3D (iterations) yields the most accurate and stable calibration result under the tested setup, and its solution is therefore used only as a reference calibration result for relative comparison in this representative trial. It should not be interpreted as absolute ground truth. The DRHeC method shows a translation error of 0.03 m, and the EasyHeC method shows a translation error of 0.05 m, both smaller than those of the Tsai et al. method (0.14 m), the Daniilidis et al. method (0.16 m), the Wang et al. method (0.10 m), and the Single3D (closed-form) method (0.52 m). Rotation errors show a similar trend: DRHeC and EasyHeC yield smaller deviations, whereas other closedform methods exhibit significantly larger angular errors. (with rotation errors of 0.02, 0.03, 0.13, 0.09, 0.04, and 0.85 rad corresponding to DRHeC, EasyHeC, Tsai, Daniilidis, Wang, and Single3D (closed-form), respectively). Using Single3D (iterations) as the reference, these results indicate that DRHeC provides the closest estimation in this trial, followed by Easy-HeC, while the compared closed-form marker-based methods show larger deviations overall. The compared closed-form marker-based methods also exhibit larger translation and rotation errors, which likely contributed to their insertion failures. Although EasyHeC still does not achieve successful insertion, it shows a smaller deviation from the reference than several closed-form marker-based methods in this representative trial.

In conclusion, DRHeC shows strong performance among the evaluated differentiable-rendering methods and achieves higher success rates than several closed-form marker-based methods under the tested setup, while remaining below Single3D (iterations). This supports the effectiveness of DRHeC for successful insertion without relying on fiducial markers.

## E. DENSO VS060 Experiments

To provide an additional validation beyond the UR5e experiments, we conducted 18 parallel trials using the DENSO VS060 robot (DENSO WAVE Inc., Aichi, Japan) equipped with an OnRobot RG2 gripper (OnRobot A/S, Odense, Denmark). In contrast to the UR5e robot, which is visually distinctive due to richer color variation, the VS060 has a nearly uniform appearance across all joints, with only minor visual differences at the end effector. This reduced color variation makes RGB-based optimization more challenging and therefore provides a useful additional test case for the proposed framework. The success rates for each method are presented in Table V.

Compared with the baseline methods, DRHeC achieved strong success rates in both grasping and insertion tasks, outperforming EasyHeC by 22.2 and 44.5 percentage points, respectively, although Single3D (iterations) achieved the highest success rates overall. This demonstrates the robustness of DRHeC across different robotic platforms. Despite the lack of distinctive visual features on the DENSO VS060, DRHeC still performed strongly, highlighting its ability to handle challenging visual conditions and generalize effectively. We also provide a relative quantitative comparison using one representative trial on the DENSO VS060 robot, taking the Single3D (iterations) result only as a reference calibration result under the tested setup, rather than as absolute ground truth. The translation errors for DRHeC, EasyHeC, Tsai et al., Daniilidis et al., Wang et al., and Single3D (closed-form) are 0.03, 0.05, 0.20, 0.08, 0.07, and 0.05 m, respectively. The corresponding rotation errors are 0.03, 0.04, 0.03, 0.01, 0.01, and 0.09 rad. Considering both translation and rotation, DRHeC provides the closest alignment to the Single3D (iterations) reference result, followed by EasyHeC, while the compared closed-form marker-based methods tend to show larger deviations. Overall, these results support the practicality of DRHeC for offline or setup-time industrial hand-eye calibration scenarios.

TABLE V  
SUCCESS RATES OF DIFFERENT EXPERIMENTS ON DENSO VS060 ROBOT.
<table><tr><td rowspan="2">Method</td><td colspan="5">Marker-based</td><td colspan="2">Differentiable-rendering-based (Markerless)</td></tr><tr><td>Tsai et al. [7]</td><td>Daniilidis et al. [8]</td><td>Single3D (closed-form) [2]</td><td>Single3D (iterations) [2]</td><td>Wang et al. [21]</td><td>EasyHeC [25]</td><td>DRHeC</td></tr><tr><td>Grasping</td><td>11.1%</td><td>11.1%</td><td>38.9%</td><td>94.4%</td><td>44.4%</td><td>55.6%</td><td>77.8%</td></tr><tr><td>Insertion</td><td>11.1%</td><td>11.1%</td><td>27.8%</td><td>83.3%</td><td>22.2%</td><td>33.3%</td><td>77.8%</td></tr></table>

![](images/941c939e55abe53994c886a245c92c39de5113e90f7ddb3b4f371909963f1d56.jpg)  
(1-a)

![](images/89550e57010a262646c1aa69499c650f3c0347d5446814042ebcae54e170ba80.jpg)

![](images/9cfddfc6cd6f2ef710b65fc7bbf84b6947b84221bf8499c1600a23d4f6e3ba6d.jpg)  
(2-a)

(1-b)  
![](images/dc4a0d701cb7c2b28b5bcd999ff5757e67f1ecf235a80afb159c1d9b3e33a933.jpg)  
(2-b)  
Fig. 7. Reprojection of calibration results on real-world images for two scenarios (1 and 2). Results from DRHeC are shown in the left column (1-a and 2-a), whereas those from EasyHeC are shown in the right column (1-b and 2-b).

## F. Discussion

The experimental results show that DRHeC substantially improves over EasyHeC under the present setup and achieves higher success rates than several closed-form marker-based baselines under the tested conditions. This improvement is attributed to DRHeC’s ability to integrate color and mask geometric features. Marker-based methods, while computationally efficient, may be sensitive to marker occlusion and variations in imaging conditions. Moreover, in our experimental setup, the relatively long calibration distance (2 meters), limited camera poses (only 10), and the use of printed markers with limited precision may have further degraded accuracy and constrained the performance of traditional marker-based methods in this study. Similarly, EasyHeC achieves higher success rates than several closed-form marker-based baselines under this experimental setup, but it still suffers from several limitations, including restricted supervision from binary mask configurations and the lack of color information, which can reduce robustness and precision. In contrast, DRHeC improves calibration performance by incorporating RGB cues and mask geometric features.

While the proposed DRHeC method achieves strong performance, there are still areas for improvement. As shown in Fig. 7, DRHeC exhibits slight misalignment between the robot in the simulation and the real world. This issue arises from the limitations of the I2IT method, which struggles to fully transform the real-world robot into a simulation-like image. Adding more diverse input to the I2IT training set may help address the misalignment issues by better capturing realworld variations. Despite CycleGAN’s ability to train I2IT transformations without paired datasets, discrepancies still arise in the simulation robot model, such as differences in color configuration and texture that cannot be fully avoided. Additionally, imperfect mask extraction introduces slight errors in color correspondence. Optimization remains challenging as it may fall into local minima when using only a single image.

From the perspective of single-frame observability, the calibration is more likely to be well constrained when perturbations of $^ c T _ { b }$ produce sufficiently distinguishable changes in the rendered RGB and mask observations under a known robot configuration. In practice, this is more likely to hold when the robot structure is sufficiently visible, self-occlusion is limited, projection symmetry is weak, and the image contains informative RGB and mask gradients. In contrast, degradation may occur when the known robot configuration and camera viewpoint produce highly similar observations under different perturbations of $^ { c } T _ { b } .$ , for example under projection symmetry, severe self-occlusion, limited visible structure, or weak appearance variation. In this context, pose ambiguity should be understood only as one possible qualitative manifestation of degraded single-frame observability. A formal observability analysis and a quantitative tolerance boundary for such degraded cases were beyond the scope of the present study.

![](images/dd1e6c172fa51091ca1d7aed328c95dd68253e05c76f414fe22170d6c4242df1.jpg)  
(a)

![](images/dc8dc93aba8f2ae170fc7ff58bc730f156329e88aeb68b7d8c96ac15d51b96f8.jpg)  
(b)

![](images/9dfacbdbaaa62a31c4700376ee2614f57866d540c749effc88ef2d64cfe39bdd.jpg)  
(c)

![](images/730c85de06c9528151317552039addb74b14e80c2b1ca8fb316f6aa5615b8293.jpg)  
(d)

![](images/7c9cdab10c039edc9b5e6580b66ab9da9d1b3efc5a02daa649dd296833ee595b.jpg)  
(e)

![](images/a9d73acd21655881c155005824e176b504eb964cd60a289b2b852f7ce2a5dc45.jpg)  
(f)  
Fig. 8. Illustrative example of pose ambiguity in binary-mask-based calibration. RGB images (a-c) and corresponding binary masks (d-f) represent three distinct robot poses: (a,d): [0°, -90°, 0°, -90°, 0°, 0°]; (b,e): [90°, -90°, 0°, - 90°, 0°, 0°]; and (c,f): [-90°, -90°, 0°, -90°, 0°, 0°]. Despite the 180° difference in Joint 1 between (b) and (c), their binary masks (e,f) appear nearly identical due to projection ambiguity. Methods relying on binary-mask loss may have difficulty distinguishing these poses during optimization. In contrast, DRHeC incorporates RGB information, which can help distinguish such cases.

Pose ambiguity may arise in some binary-mask-based calibration cases. As illustrated in Fig. 8, distinct robot poses can produce nearly identical binary masks due to projection symmetry and occlusion effects. The additional color and texture information encoded in the RGB data provides discriminative cues that help distinguish robot poses which are otherwise indistinguishable by binary masks alone. In such cases, methods relying only on binary-mask supervision may have difficulty distinguishing between different poses, which can reduce calibration accuracy. In contrast, DRHeC leverages RGB image derivatives to provide additional color and texture cues, which can help distinguish such cases during optimization. This example illustrates a potential additional benefit of RGB information in cases where binary masks become insufficiently discriminative. Since this work does not provide a systematic quantitative evaluation specifically targeting such ambiguity cases, we present this observation as a qualitative example rather than a central validated claim.

In addition, although the DENSO VS060 robot provides minimal color variation, DRHeC still achieved strong calibration performance. This result suggests that DRHeC effectively leverages mask geometric features, specifically the mask centroid and mask area, to provide global information for gradient-based optimization, even when explicit color information is limited. This capability is particularly advantageous in industrial scenarios where robots may have uniform color schemes. Furthermore, in practical applications, robots often have manufacturer labels or markings, and end-effectors may be equipped with tools that introduce additional color variations. In the current pipeline, the influence of such appearance changes mainly depends on the I2IT stage. If these changes do not significantly affect feature extraction or the generation of real-to-sim images, then they are not expected to have a substantial impact on the subsequent DRHeC calibration. However, if they affect the I2IT translation process, then the generated sim-like images may no longer satisfy the assumptions of the current experimental setting, and the final calibration performance may also be affected. A dedicated evaluation with deliberate appearance modifications was beyond the scope of the present study, and a more systematic evaluation under deliberate color and texture variations will be investigated in future work. Although the current experiments on Baxter, UR5e, and DENSO VS060 provide initial evidence of applicability across different robot settings, the generalization capability of the framework has not yet been systematically evaluated on a broader range of robot platforms. More extensive validation on a broader range of robot platforms will be an important direction for future work.

Our method assumes reasonably stable indoor illumination. Although it is not designed to handle illumination variations, it remains stable under the mild indoor lighting fluctuations present during all our real-world experiments. However, extreme lighting conditions or occlusions on the robot mask may affect the reliability of the pipeline. When the scene becomes excessively bright or dark, the observed images may contain more noise and reduced contrast, which will negatively influence both the segmentation results and the I2IT translation process. Meanwhile, occlusions or blocking on the robot mask will interfere with the mask consistency used by I2IT, causing the translated sim-like images to contain noticeable errors. These inaccurate translated images may then propagate through the differentiable rendering pipeline and ultimately reduce the accuracy of the final hand-eye calibration result.

We further evaluated the sensitivity of DRHeC to geometric model mismatch in simulation by rendering reference images with globally anisotropic Gaussian scaling perturbations with standard deviations of 0%, 1%, 3%, and 5%. The experiment comprised 20 paired trials using the UR5e robot model in the simulation environment. The mean translation errors were 7.41, 21.92, 53.05, and 86.13 mm, respectively, while the corresponding mean rotation errors were 0.0197, 0.0190, 0.0287, and 0.0396 rad. These results indicate that a 1% geometric mismatch already substantially degrades translation accuracy, whereas the effect on rotation accuracy appears to become more noticeable at larger mismatch levels. A systematic quantitative evaluation of the remaining factors, including controlled illumination changes, external occlusions, segmentation failures, and deliberate robot appearance variations, remains an important direction for future work.

In terms of computational complexity, our differentiable rendering-based optimization requires approximately 7.19 seconds per 1000 iterations for DRHeC, which is comparable to EasyHeC (7.18 seconds) and significantly faster than IPE (78.25 seconds). Under the 2000-iteration setting used in this study, DRHeC requires about 14 seconds per calibration. Since this cost is incurred only once during calibration and the estimated hand-eye transformation can be reused for subsequent manipulation tasks, we consider this runtime practical for offline hand-eye calibration or occasional recalibration. Further acceleration would be beneficial for online recalibration or fully automated high-throughput deployment.

![](images/496b409d0ea18d888df64372fa3cc160ccbe639d309dd0316022430c11f42812.jpg)

![](images/e9b5297e9edd39a642206531d914c2a2140bb5398c21fb55199903aac1bfe11f.jpg)  
(a)

![](images/931ce3c714271f132b210180f80b505a485226e1e3e50aeee901b311b20be1a3.jpg)

![](images/106b3dc8df489b5a36d4de8c737395003dcf3e63affdf79cfe2a5ed0fca9cc99.jpg)  
(b)

![](images/3cdc68eb879326657ab2f4d9f9af0afb17b394ca403fe0c767172b1d673d0f43.jpg)

![](images/50669e07de3585bd181fd4dc920af47869414d0ab6962686c6105c394497f087.jpg)  
(c)  
Fig. 9. Ablation study on (a) $\lambda _ { \mathrm { c e n t r o i d } } .$ , (b) $\lambda _ { \mathrm { a r e a } } ,$ and (c) threshold for the two-step optimization parameters. Each pair shows translation and rotation errors, respectively.

## VII. ABLATION STUDY

The simulation experiments in Sec. V-A already form an ablation study over different combinations of loss terms. Specifically, we compare four configurations: (1) EasyHeC (binary mask only), (2) RGB (using RGB mask without geometric terms), (3) RGB + mask centroid, and (4) our full DRHeC method (RGB + mask centroid + mask area). The results consistently show that DRHeC achieves the best performance, followed by RGB + mask centroid and RGB alone. This performance hierarchy clearly demonstrates the importance of incorporating mask geometric terms for improving stability and accuracy.

We also conduct an ablation study to determine the appropriate hyperparameters for the loss function in Eq. 10 under the LP and MP settings, as the HP setting produces errors that are unrealistically large for practical hand-eye calibration applications. We set $\lambda _ { \mathrm { r g b } } = 1$ as the reference weight. Since the RGB, centroid, and area losses are defined over different quantities and therefore have different numerical scales, logarithmically spaced candidate weights were evaluated for $\lambda _ { \mathrm { c e n t r o i d } }$ and $\lambda _ { \mathrm { a r e a } }$ . The final values were subsequently selected based on the ablation results. All ablation experiments are implemented using our DRHeC framework, and each configuration is evaluated over 20 independent parallel trials to ensure statistical reliability.

As shown in Table VI, Table VII and Table VIII, the proposed hyperparameter configuration achieves the most robust and accurate calibration results. Specifically, the parameters $\lambda _ { \mathrm { c e n t r o i d } } = 0 . 0 1 , \lambda _ { \mathrm { a r e a } } = 0 . 0 0 0 1$ , and threshold $= 0 . 2 5$ yield the lowest translation errors, while maintaining comparable rotation accuracy across all conditions. Compared with the baseline configuration without these loss terms, our configuration significantly improves translation accuracy. When $\lambda _ { \mathrm { c e n t r o i d } } = 0 .$ , the translation error increases from 5.05 mm to 86.06 mm, indicating that the mask centroid loss term is essential for providing a geometric constraint that stabilizes convergence. Likewise, when $\lambda _ { \mathrm { a r e a } } = 0 .$ , the translation error increases from 2.30 mm to 134.86 mm, demonstrating that the mask area loss term contributes to scale alignment between the rendered and observed masks. For the threshold parameter, the results show that the proposed two-step optimization performs well over a reasonable range of threshold values. In particular, threshold = 0.20, 0.25, and 0.30 all achieve relatively low errors, while threshold = 0.25 gives the best overall result. This indicates that the method is not overly sensitive to the specific threshold value. Consistent with the optimization behavior discussed in Sec. IV.B, excessively small or large thresholds may degrade performance by delaying rotation refinement or activating rotation updates too early. Therefore, the main advantage lies in the proposed coarseto-fine two-step optimization strategy, while 0.25 is regarded as a practical choice within a stable operating range rather than as a theoretically unique value. In addition, the same hyperparameter configuration was used for all simulation and real-world experiments without environment-specific retuning. The experimental results indicate that the selected parameters remain effective across different robot models and in both simulation and real-world scenarios.

TABLE VI  
ABLATION STUDY RESULTS OF THE $\lambda _ { c e n t r o i d }$ OVER 2000 ITERATIONS.
<table><tr><td> $\overline { { \lambda _ { c e n t r o i d } } }$ </td><td>0</td><td>0.1</td><td>0.01</td><td>0.001</td><td>0.0001</td></tr><tr><td>Translation</td><td>86.06</td><td>10.67</td><td>5.05</td><td>158.62</td><td>188.10</td></tr><tr><td>Rotation</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.03</td><td>0.00</td></tr></table>

TABLE VII  
ABLATION STUDY RESULTS OF THE $\lambda _ { a r e a }$ OVER 2000 ITERATIONS.
<table><tr><td> $\overline { { \lambda _ { a r e a } } }$ </td><td>0</td><td>0.1</td><td>0.01</td><td>0.001</td><td>0.0001</td></tr><tr><td>Translation</td><td>134.86</td><td>111.17</td><td>21.81</td><td>16.99</td><td>2.30</td></tr><tr><td>Rotation</td><td>0.05</td><td>0.11</td><td>0.01</td><td>0.04</td><td>0.01</td></tr></table>

We also perform an ablation study to demonstrate that the I2IT module plays an essential role in the real world experiment, and that without it the calibration results would suffer from large errors. Since the real world setup does not provide ground truth for the hand-eye transformation, we evaluate the influence of the I2IT module by comparing the rendered images obtained through differentiable calibration using the real captured images and the I2IT transformed images. A more accurate calibration should produce a rendered image that is closer to the simulated view. As shown in Table IX, using I2IT clearly improves all image based metrics, including a large reduction in MSE and MAE as well as notable gains in PSNR and SSIM. These results indicate that the I2IT module effectively reduces the appearance gap between real and synthetic images, which enables more accurate and stable hand-eye calibration.

TABLE VIII  
ABLATION STUDY RESULTS OF THE THRESHOLD FOR THE TWO-STEP OPTIMIZATION OVER 2000 ITERATIONS.
<table><tr><td>Threshold</td><td>0</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.50</td></tr><tr><td>Translation (mm)</td><td>183.83</td><td>18.32</td><td>13.65</td><td>28.23</td><td>150.11</td></tr><tr><td>Rotation (rad)</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.03</td><td>0.07</td></tr></table>

TABLE IX  
ABLATION STUDY RESULTS OF THE I2IT MODULE.
<table><tr><td>Method</td><td>MSE</td><td>MAE</td><td>IoU</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>Real world image</td><td>0.0099</td><td>0.0162</td><td>0.6111</td><td>20.88</td><td>0.9477</td></tr><tr><td>I2ITed img</td><td>0.0026</td><td>0.0053</td><td>0.8520</td><td>25.95</td><td>0.9727</td></tr></table>

Overall, the ablation study confirms that all loss terms and the I2IT module are necessary, and our chosen configuration achieves the best performance among all candidates. The selected nonzero weights of the RGB derivative, centroid, and area losses stabilize the optimization process, making the differentiable rendering based hand-eye calibration more robust and less prone to failure.

## VIII. CONCLUSION

In this paper, we introduce a novel framework for robotic hand-eye calibration that utilizes RGB derivatives and mask geometry features in differentiable rendering, which improves calibration accuracy and optimization stability under binarymask-based supervision. Additionally, we propose a modified loss function for training the I2IT network, which preserves essential geometric information. Experimental results demonstrate that our approach achieves strong accuracy and robustness in both simulations and real-world scenarios.

While the proposed method shows promising results, the reliance on a model in differentiable hand-eye calibration can introduce errors due to model inaccuracies. To address this, we plan to integrate advanced model reconstruction techniques that will reduce errors and enhance calibration precision. Additionally, because the camera’s field of view often requires capturing the entire robot, our current implementation is limited to the eye-to-hand configuration. To overcome this limitation, we propose leveraging partial observations of the robot to enable a more practical and flexible eye-in-hand calibration. This approach would allow for either the reconstruction of the full model or direct calibration using only visible segments, thereby increasing the method’s flexibility and applicability in various practical settings. Another practical consideration is that the current mask preparation and runtime are mainly suitable for offline or setup-time hand-eye calibration rather than online recalibration. Nevertheless, this setting is consistent with many practical hand-eye calibration scenarios, where calibration is performed before task execution and the estimated transformation is reused afterward. Future work will focus on automatic mask generation and acceleration of the differentiable-renderer-based hand-eye calibration process. Finally, the method is mainly applicable to indoor scenes with moderate lighting and unobstructed robot visibility, as extreme illumination or occlusions may require additional preprocessing. Broader validation beyond the tested settings remains an important direction for future work, including formal observability analysis for arbitrary single-frame configurations, calibration uncertainty evaluation with independent reference measurements, and more comprehensive sensitivity analyses covering broader types of robot-model deviations. Comparisons with recent model-free or weak-model calibration methods under inaccurate-model conditions are also left for future work.

## REFERENCES

[1] J. Kofman, X. Wu, T. J. Luu, and S. Verma, “Teleoperation of a robot manipulator using a vision-based human-robot interface,” IEEE transactions on industrial electronics, vol. 52, no. 5, pp. 1206–1219, 2005.

[2] G. Jin, X. Yu, Y. Chen, and J. Li, “Hand-eye parameter estimation based on 3d observation of a single marker,” IEEE Transactions on Instrumentation and Measurement, 2024.

[3] Y. Zhang, Z. Qiu, and X. Zhang, “A simultaneous optimization method of calibration and measurement for a typical hand–eye positioning system,” IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1–11, 2020.

[4] J. Wu, Y. Sun, M. Wang, and M. Liu, “Hand-eye calibration: 4-d procrustes analysis approach,” IEEE Transactions on Instrumentation and Measurement, vol. 69, no. 6, pp. 2966–2981, 2019.

[5] S. Qiu, M. Wang, and M. R. Kermani, “A new formulation for hand–eye calibrations as point-set matching,” IEEE Transactions on Instrumenta tion and Measurement, vol. 69, no. 9, pp. 6490–6498, 2020.

[6] S. Sarabandi, J. M. Porta, and F. Thomas, “Hand-eye calibration made easy through a closed-form two-stage method,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 3679–3686, 2022.

[7] R. Y. Tsai, R. K. Lenz et al., “A new technique for fully autonomous and efficient 3 d robotics hand/eye calibration,” IEEE Transactions on robotics and automation, vol. 5, no. 3, pp. 345–358, 1989.

[8] K. Daniilidis, “Hand-eye calibration using dual quaternions,” The International Journal of Robotics Research, vol. 18, no. 3, pp. 286–298, 1999.

[9] J. Schmidt and H. Niemann, “Data selection for hand-eye calibration: A vector quantization approach,” The International Journal of Robotics Research, vol. 27, no. 9, pp. 1027–1053, 2008.

[10] Z. Fu, J. Pan, E. Spyrakos-Papastavridis, X. Chen, and M. Li, “A dual quaternion-based approach for coordinate calibration of dual robots in collaborative motion,” IEEE Robotics and Automation Letters, vol. 5, no. 3, pp. 4086–4093, 2020.

[11] G. Tan, J. Du, R. Guo, W. Wang, and Z. Ju, “Robust hand-eye calibration with a single plane for 3d robot measurement,” IEEE Transactions on Instrumentation and Measurement, 2025.

[12] L. Li, L. Zhou, T. Zhang, and J. Zheng, “Industrial robot hand–eye calibration combining data augmentation and actor-critic network,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–12, 2023.

[13] E. Pedrosa, M. Oliveira, N. Lau, and V. Santos, “A general approach to hand–eye calibration through the optimization of atomic transformations,” IEEE Transactions on Robotics, vol. 37, no. 5, pp. 1619–1633, 2021.

[14] K. Koide and E. Menegatti, “General hand–eye calibration based on reprojection error minimization,” IEEE Robotics and Automation Letters, vol. 4, no. 2, pp. 1021–1028, 2019.

[15] M. Li, Z. Du, X. Ma, W. Dong, and Y. Gao, “A robot hand-eye calibration method of line laser sensor based on 3d reconstruction,” Robotics and Computer-Integrated Manufacturing, vol. 71, p. 102136, 2021.

[16] Z. Liu, X. Liu, G. Duan, and J. Tan, “Precise hand–eye calibration method based on spatial distance and epipolar constraints,” Robotics and Autonomous Systems, vol. 145, p. 103868, 2021.

[17] J. Ha, “Probabilistic framework for hand–eye and robot–world calibration ax = yb,” IEEE Transactions on Robotics, vol. 39, no. 2, pp. 1196– 1211, 2022.

[18] J. Wu, M. Liu, Y. Zhu, Z. Zou, M.-Z. Dai, C. Zhang, Y. Jiang, and C. Li, “Globally optimal symbolic hand-eye calibration,” IEEE/ASME Transactions on Mechatronics, vol. 26, no. 3, pp. 1369–1379, 2020.

[19] R. A. Boby and S. K. Saha, “Single image based camera calibration and pose estimation of the end-effector of a robot,” in 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 2435–2440.

[20] D. Zhu, H. Wu, T. Ding, and L. Hua, “Point cloud registration-enabled globally optimal hand–eye calibration,” IEEE/ASME Transactions on Mechatronics, 2024.

[21] X. Wang, H. Sun, C. Liu, and H. Song, “Dual quaternion operations for rigid body motion and their application to the hand–eye calibration,” Mechanism and Machine Theory, vol. 193, p. 105566, 2024.

[22] J. Lu, F. Richter, and M. C. Yip, “Pose estimation for robot manipulators via keypoint optimization and sim-to-real transfer,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4622–4629, 2022.

[23] T. E. Lee, J. Tremblay, T. To, J. Cheng, T. Mosier, O. Kroemer, D. Fox, and S. Birchfield, “Camera-to-robot pose estimation from a single image,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 9426–9432.

[24] J. Lu, F. Liu, C. Girerd, and M. C. Yip, “Image-based pose estimation and shape reconstruction for robot manipulators and soft, continuum robots via differentiable rendering,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 560–567.

[25] L. Chen, Y. Qin, X. Zhou, and H. Su, “Easyhec: Accurate and automatic hand-eye calibration via differentiable rendering and space exploration,” IEEE Robotics and Automation Letters, 2023.

[26] Z. Hong, K. Zheng, and L. Chen, “Easyhec++: Fully automatic handeye calibration with pretrained image models,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 816–823.

[27] F. C. Park and B. J. Martin, “Robot sensor calibration: solving ax= xb on the euclidean group,” IEEE Transactions on Robotics and Automation, vol. 10, no. 5, pp. 717–721, 1994.

[28] R. Horaud and F. Dornaika, “Hand-eye calibration,” The international journal of robotics research, vol. 14, no. 3, pp. 195–210, 1995.

[29] N. Andreff, R. Horaud, and B. Espiau, “On-line hand-eye calibration,” in Second International Conference on 3-D Digital Imaging and Modeling (Cat. No. PR00062). IEEE, 1999, pp. 430–436.

[30] E. Valassakis, K. Dreczkowski, and E. Johns, “Learning eye-in-hand camera calibration from a single image,” in Conference on Robot Learning. PMLR, 2022, pp. 1336–1346.

[31] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[32] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[33] A. Rosinol, J. J. Leonard, and L. Carlone, “Nerf-slam: Real-time dense monocular slam with neural radiance fields,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437–3444.

[34] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039–18 048.

[35] A. Palazzi, L. Bergamini, S. Calderara, and R. Cucchiara, “End-toend 6-dof object pose estimation through differentiable rasterization,” in Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 2018, pp. 0–0.

[36] K. Park, A. Mousavian, Y. Xiang, and D. Fox, “Latentfusion: End-toend differentiable reconstruction and rendering for unseen object pose estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10 710–10 719.

[37] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3431–3440.

[38] S. Xie and Z. Tu, “Holistically-nested edge detection,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 1395– 1403.

[39] R. Zhang, P. Isola, and A. A. Efros, “Colorful image colorization,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14. Springer, 2016, pp. 649–666.

[40] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.

[41] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros, “Context encoders: Feature learning by inpainting,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2536–2544.

[42] C. Li and M. Wand, “Precomputed real-time texture synthesis with markovian generative adversarial networks,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14. Springer, 2016, pp. 702–716.

[43] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014.

[44] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125– 1134.

[45] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catanzaro, “High-resolution image synthesis and semantic manipulation with conditional gans,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 8798–8807.

[46] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232.

[47] T. Park, A. A. Efros, R. Zhang, and J.-Y. Zhu, “Contrastive learning for unpaired image-to-image translation,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16. Springer, 2020, pp. 319–345.

[48] C. Lu, Q. Zhang, K. Krishnakumar, J. Chen, H. Fuchs, S. Talathi, and K. Liu, “Geometry-aware eye image-to-image translation,” in 2022 Symposium on Eye Tracking Research and Applications, 2022, pp. 1–7.

[49] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[50] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[51] B. Li, K. Xue, B. Liu, and Y.-K. Lai, “Bbdm: Image-to-image translation with brownian bridge diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2023, pp. 1952–1961.

[52] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[53] T.-M. Li, M. Aittala, F. Durand, and J. Lehtinen, “Differentiable monte carlo ray tracing through edge sampling,” ACM Transactions on Graphics (TOG), vol. 37, no. 6, pp. 1–11, 2018.

[54] S. Laine, J. Hellsten, T. Karras, Y. Seol, J. Lehtinen, and T. Aila, “Modular primitives for high-performance differentiable rendering,” ACM Transactions on Graphics, vol. 39, no. 6, 2020.

[55] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” arXiv:2304.02643, 2023.

![](images/7fa844af987921f87bbf4392d19f1d4617c6619f7eab419827c16a2c47da92fb.jpg)  
Xiaotian Zhang (Member, IEEE) is a Ph.D. student in the Department of Precision Engineering, School of Engineering, The University of Tokyo, Japan. He received his M.S. degree in Precision Engineering from The University of Tokyo in 2021.  
His research interests include robot perception and computer vision.

![](images/fcc3c65acf44cfdae97078a60984bf98c3930b829f1dc1bf6118db6ed19fb1f9.jpg)

Yusheng Wang (Member, IEEE) received the B.E. degree in mechanical engineering from the Dalian University of Technology, Dalian, China, in 2017, and the M.S. and Ph.D. degrees in precision engineering from The University of Tokyo, Bunkyo, Japan, in 2019 and 2022, respectively.

From 2022 to 2023, he was a Project Assistant Professor with the Department of Precision Engineering, The University of Tokyo. He is currently an Assistant Professor with Research into Artifacts, Center for Engineering, The University of Tokyo.

His research interests include robot perception, computer vision, human-robot interaction, and marine robotics.

![](images/f7b2b911b2471bd912ae5faa959ff9ba5045b4968c34232b8e587189af5bbb16.jpg)  
Keiji Okuhara is an Engineer in Factory Products Business Unit, Engineering Division, DENSO WAVE INCORPORATED, Japan.

Dr. Wang is a Member of SICE, JSME, and RSJ.

![](images/1e864b2bd2b00845b34434e5e85dca425e4bad0561dcdf968aae2c7762265fc0.jpg)

Hiroyasu Baba is a Manager in Factory Products Business Unit, Strategy Planning Division, DENSO WAVE INCORPORATED, Japan.

Naoya Kagawa is a Manager in Factory Products Business Unit, Engineering Division, DENSO WAVE INCORPORATED, Japan.

![](images/318b99f5b009d41832188e88a099c78a8f25ac6095adf25c4ac3321d6f9dcaa8.jpg)

![](images/3ed96f9c48977dcbcadee32911a3c99bcd5855771ab5594348ed2741133f2643.jpg)

Jun Ota (Member, IEEE) received the B.E., M.E., and Ph.D. degrees from the Faculty of Engineering, The University of Tokyo, Bunkyo, Japan, in 1987, 1989, and 1994, respectively.

From 1989 to 1991, he worked with Nippon Steel Corperation. In 1991, he was a Research Associate with The University of Tokyo. He became a Lecturer and Associate Professor in 1994 and 1996, respectively. In April 2009, he became a Professor with the Graduate School of Engineering, The University of Tokyo. In June 2009, he became a Professor with

![](images/adb32309515be7a524480aeba56cb14894b77b3951f81dc316a7427c8edf272e.jpg)

Noritaka Takamura is an Engineer in Factory Products Business Unit, Engineering Division, DENSO WAVE INCORPORATED, Japan.

Research into Artifacts, Center for Engineering, (RACE), The University of Tokyo. From 1996 to 1997, he was a Visiting Scholar with Stanford University. From 2015 to 2021, he was a Guest Professor with the South China University of Technology. He is a Professor with RACE, School of Engineering, The University of Tokyo. His research interests include multiagent robotic systems, embodied-brain systems science, design support for large-scale production/material handling systems, and human behavior analysis and support.

Dr. Ota was the recipient of the fellowships from the Robotics Society of Japan in 2016 and from the Japan Society of Mechanical Engineers in 2021.