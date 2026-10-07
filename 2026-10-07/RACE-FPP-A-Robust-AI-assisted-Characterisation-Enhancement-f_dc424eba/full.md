# RACE-FPP: A Robust AI-assisted Characterisation Enhancement for Fringe Projection Profilometry

Osman Ali<sup>1</sup> <sup>∗</sup>, Xiangjun Kong<sup>1</sup>, Tibebe Yalew<sup>1</sup>, Waiel Elmadih<sup>2</sup>, and Samanta Piano<sup>1</sup>

<sup>1</sup>Manufacturing Metrology Team, Faculty of Engineering, University of Nottingham, Nottingham, NG8 1BB, UK

<sup>2</sup>Taraz Metrology Ltd., Strelley Hall, Nottingham NG8 6PE, UK

<sup>∗</sup>Correspondence to: Osman Ali: osman.ali@nottingham.ac.uk

## Abstract

Fringe Projection Profilometry (FPP) requires precise system characterisation to achieve reliable three-dimensional (3D) reconstructions; however, characterisation accuracy strongly depends on robust checkerboard feature localisation, which can deteriorate under challenging imaging conditions such as lens blur and characterisation target orientations. Existing deep learning-based corner detectors are typically assessed using detection metrics and camera reprojection error alone, without considering their wider impact on projector characterisation, cameraprojector stereo characterisation consistency, or overall measurement accuracy. In this work, we introduce a complete FPP characterisation pipeline that incorporates deep learning-based corner detection into the standard camera characterisation workflow. We also characterise the projector by sampling phase values at the centres of the white squares in the characterisation target. Rather than treating corner detection as an isolated task, the proposed framework explicitly analyses how localisation errors propagate throughout the entire FPP characterisation chain. Performance is evaluated using detection metrics (e.g., precision and recall), camera and projector reprojection errors, and the camera and projector stereo characterisation. Across a mixed dataset of clean and degraded images, the camera reprojection error is reduced from 1.237 pixels to 0.259 pixels, while the projector reprojection error is reduced by roughly 50%. Dimensional evaluation of reconstructed artefacts shows improved geometric accuracy compared with those resulting from the conventional pipeline. Overall, the findings indicate increased robustness of system-level characterisation under challenging imaging conditions, thereby enabling more reliable industrial FPP measurements.

Keywords: Fringe Projection Profilometry, Camera characterisation, Projector characterisation, Deep learning, Checkerboard corner detection.

## Introduction

Fringe Projection Profilometry (FPP) is a non-contact optical measurement technique widely used in manufacturing, aerospace, and biomedical applications [1, 2]. In FPP, a projector emits structured fringe patterns onto an object, and a camera captures the deformed patterns to reconstruct its three-dimensional (3D) geometry. Accurate camera and projector characterisation is essential, as errors in intrinsic (focal lengths, principal points, skewness) and extrinsic parameters (camera and projector poses and their stereo relationship) directly afect final reconstruction accuracy [3, 4].

Among other characterisation targets, planar checkerboards are commonly used for camera and projector characterisation because of their simple geometry and well-defined feature locations [5]. Characterisation quality critically depends on accurate localisation of these corners; even sub-pixel errors can propagate through the characterisation chain and distort 3D reconstruction [6]. In FPP systems, characterisation images are often afected by lens defocus and environment-induced noise (vibrations, light variations, etc.), which make accurate point localisation more dificult [7, 8].

Camera characterisation in FPP systems is commonly performed using established techniques such as Zhang’s characterisation method, which estimates intrinsic and extrinsic parameters from multiple views of a planar checkerboard pattern [9]. While conventional detection algorithms, such as those implemented in OpenCV, perform well under controlled conditions, their performance degrades in practical FPP scenarios [10, 11]. These limitations introduce biases in the estimated characterisation parameters and reduce reconstruction accuracy. [9, 12].

To address these challenges, robust checkerboard corner detection becomes essential. This process typically involves global board identification followed by accurate corner localisation with sub-pixel refinement [13]. Classical corner detection approaches can be classified into three categories: intensity-, contour-, and binary-based methods [14]. Intensity-based detection is a widely used approach that identifies corners by computing image derivatives (gradients) to locate points where the intensity varies across multiple directions. Examples of intensity-based methods are the Harris detector [15] and its variants [16, 17, 18], which provide robustness to rotation and illumination changes but remain computationally demanding and limited in localisation precision. Enhancements combining FAST and SUSAN detectors [19, 20] improve eficiency; however, performance degrades under defocus blur, noise, and extreme board poses commonly encountered in FPP [21, 14].

Recent advances in deep learning have reformulated corner detection as a regression or detection task using convolutional neural networks (CNNs) [22]. Some studies focus on image restoration to mitigate blur [23, 24, 25]. In contrast, other methods directly predict corner locations [11, 10, 26, 27, 28, 29]. In these methods, deep learning models learn to output the pixel coordinates of checkerboard corners via coordinate regression or heatmap estimation, without relying on explicit edge detection or geometric intersection steps. The YOLO family [30, 31] has gained particular attention due to its real-time performance and regression-based localisation capability. Variants such as YOLOX and YOLOv8 have demonstrated improved robustness under degraded imaging conditions [14, 32, 33].

However, most existing studies evaluate such methods primarily in terms of detection metrics or camera characterisation reprojection error. The efect of feature-localisation robustness on the complete FPP characterisation chain - including projector characterisation, camera-projector stereo consistency, 3D reconstruction, and ultimately dimensional measurement accuracy- has received considerably less attention.

Projector characterisation follows a formulation analogous to camera characterisation, treating the projector as an inverse camera [34], in which phase measurements establish a correspondence between camera pixels and projector pixel coordinates [7]. In this process, each point on the characterisation board is first mapped from image space to projector space via the phase information, and then associated with its known world coordinates to estimate the projector parameters [35, 36]. While these mapping strategies can improve the accuracy of projector coordinate estimation, they remain sensitive to surface reflectivity and low signal-to-noise ratio [12]. In particular, the sharp changes in reflectivity at the boundaries between black and white checkerboard squares can degrade the captured fringe signals and introduce phase errors [37, 38]. Variations in surface reflectivity may also reduce fringe contrast or cause local underexposure and saturation, further afecting phase estimation [39]. Since the absolute phase is used to establish the corresponding projector coordinates, these phase errors can reduce the accuracy of the world-toprojector correspondences and consequently afect projector characterisation.

Errors arising from both camera and projector characterisation can propagate into the camera–projector stereo characterisation, afecting the estimation of their relative geometric relationship and consequently introducing systematic errors into 3D reconstruction [9, 40]. Since stereo characterisation relies on consistent geometric parameters for both devices, biases in either characterisation can lead to errors in their estimated relative pose. These errors subsequently afect triangulation, degrading reconstruction accuracy in both local geometry and global dimensional measurements [12].

In this paper, we develop a robust end-to-end characterisation framework for FPP systems operating under non-ideal imaging conditions. The framework combines data-driven checkerboard feature localisation with a centrebased projector correspondence strategy while retaining the conventional geometric camera-projector characterisation model. A YOLOv11 pose-estimation model is used to localise checkerboard corners under clean and degraded imaging conditions. The detected corners are subsequently reordered and supplied to the standard camera characterisation procedure. For projector characterisation, the corner coordinates are used to determine the geometric centres of the white checkerboard squares, and the corresponding absolute phase values are sampled at these locations. This avoids direct phase sampling at black-white reflectivity discontinuities and provides camera-projector correspondences from more uniform reflective regions.

The primary focus of the proposed framework is characterisation under degraded imaging conditions, where conventional checkerboard localisation becomes increasingly unreliable; noise-free images are additionally evaluated to provide a reference baseline under favourable conditions. The contributions of this work are threefold. First, we investigate robust learned checkerboard localisation specifically in the context of FPP characterisation and assess its behaviour under both uniform and spatially varying image degradation. Second, we introduce a centrebased projector correspondence strategy in which phase information is sampled at white-square centres derived from the detected checkerboard geometry. Third, rather than evaluating feature detection in isolation, we examine the complete propagation of characterisation performance through camera, projector, and stereo estimation to 3D reconstruction and dimensional measurement. The resulting reconstructions are evaluated against Coordinate Measuring Machine (CMM) reference measurements of physical artefacts, providing both local geometric and global dimensional measures of reconstruction fidelity.

![](images/656b25321ba56016cf8a5394d02f0cfbb89e811ede0bcc4b1a28466338102ff8.jpg)  
Fig. 1. Overview of the proposed FPP characterisation framework. (a) Camera characterisation: checkerboard corners are localised using the YOLOv11 pose- estimation model, reordered according to the checkerboard topology, and supplied to the conventional geometric camera-characterisation procedure. (b) Projector characterisation: white-square centres are calculated from the detected checkerboard corners, and absolute phase values sampled at these locations are converted into projector pixel coordinates. (c) Geometric camera-projector model used for camera, projector, and stereo characterisation. The subscripts (c) and (p) denote camera and projector quantities, respectively.

## Materials and methods

## Overview of the proposed characterisation framework

Figure 1 illustrates the proposed FPP characterisation framework, termed RACE-FPP (Robust AI-assisted Characterisation Enhancement for Fringe Projection Profilometry). The workflow comprises three main stages: camera characterisation, projector characterisation, and cameraprojector stereo characterisation, followed by dense 3D reconstruction.

For camera characterisation, checkerboard images acquired at multiple poses are first processed using a YOLOv11 pose-estimation model to localise the checkerboard corners. Unlike conventional approaches, such as OpenCV, where corner extraction relies on analytical image processing techniques, RACE-FPP learns corner representations from a diverse training dataset, improving robustness under challenging imaging conditions such as lens blur, noise, and extreme checkerboard orientations.

The predicted corner coordinates are subsequently reordered into a consistent checkerboard grid structure before being supplied to the standard camera characterisation procedure based on Zhang’s method. The underlying mathematical characterisation model remains unchanged; the proposed approach replaces only the image featurelocalisation stage. Following camera characterisation, the detected corner coordinates are used to compute the centres of the checkerboard white squares. Absolute phase values are then sampled at these locations and converted into projector pixel coordinates, establishing correspondences between world and projector coordinates for projector characterisation. Sampling within the whitesquare centres avoids the reflectivity discontinuities present at checkerboard corners and is intended to provide more reliable phase information for projector characterisation.

The estimated camera and projector parameters are subsequently used to determine the relative rotation and translation between the two devices through stereo characterisation. Finally, the resulting camera-projector geometry, together with the recovered absolute phase maps, is used for dense 3D reconstruction. The performance of the complete framework is assessed at several levels: checkerboard localisation accuracy and detection robustness, camera and projector reprojection errors, camera-projector stereo consistency, and CMM-referenced dimensional accuracy of reconstructed artefacts.

Camera characterisation Camera characterisation establishes the relationship between a 3D point in the target coordinate system, $\begin{array} { l l l } { \bar { { \bf P } } _ { w } } & { = } & { [ X _ { W } , Y _ { W } , Z _ { W } , 1 ] ^ { \top } } \end{array}$ , and its corresponding 2D location in the camera image, $\begin{array} { r l } { \mathbf { p } _ { c } } & { { } = } \end{array}$ $[ u _ { c } , \nu _ { c } , 1 ] ^ { \mathsf { T } }$ , in the homogeneous form. This mapping underpins the subsequent camera-projector characterisation and 3D reconstruction pipeline [9]. As shown in Figure 1(c), the mapping from the world coordinates to the image coordinates can be described using the pinhole model as follows [7, 41]:

$$
\mathbf { s } \left[ \begin{array} { l } { \left[ u _ { c } \right] } \\ { \nu _ { c } } \\ { \left[ \begin{array} { l } { \left[ R _ { c } \right] t _ { c } } \\ { 1 } \end{array} \right] } \end{array} \right] \left[ \begin{array} { l } { \left[ X _ { W } \right] } \\ { Y _ { W } } \\ { Z _ { W } } \\ { 1 } \end{array} \right] ,\tag{1}
$$

where � is an arbitrary scale factor, $K _ { c }$ is the camera intrinsic matrix (including the focal length $f _ { c }$ to the original point $O _ { c } )$ , and $[ R _ { c } | t _ { c } ]$ is the rotation and translation between the target and the camera coordinate systems. Camera characterisation therefore consists of estimating $K _ { c } ,$ $[ R _ { c } | t _ { c } ]$ and lens distortion by minimising the reprojection error between the observed 2D image points and their corresponding projected points.

Before camera characterisation, the predicted keypoints are reordered into a fixed grid sequence corresponding to the checkerboard topology. Since the point-wise detections are returned according to detection confidence rather than their spatial arrangement, a spatial sorting procedure is applied in which the points are grouped into rows according to their vertical coordinates and subsequently sorted horizontally within each row. Owing to variations in checkerboard orientation between poses, the spatial grouping thresholds in the horizontal and vertical directions are adjusted for individual images to obtain a consistent row-major ordering. The resulting ordered image points are then supplied to the standard planar characterisation procedure based on Zhang’s method [9]. The geometric camera model and parameter-estimation procedure therefore remain unchanged. For comparison, the same characterisation procedure is also performed using corner coordinates obtained with OpenCV’s findChessboardCornersSB() implementation. This sector-based detector provides direct sub-pixel corner localisation with improved robustness to image noise and runs faster on large images compared to the conventional findChessboardCorners() implementation [13, 42]. Consequently, diferences between the two camera-characterisation pipelines arise from the image coordinates provided to the common geometric estimation procedure. To evaluate performance under diferent imaging conditions, three characterisation datasets are considered:

1. noise-free images: checkerboard images acquired under favourable imaging conditions, with uniform illumination and minimal visible degradation,

2. mixed images: a combination of noise-free and degraded checkerboard images. The degraded images include both experimentally observed and synthetically introduced efects and may contain spatially nonuniform degradation (including partial, directional, or full-image defocus) and combinations of blur and image noise,

3. noisy images: checkerboard images containing substantial image degradation, including synthetic or experimentally observed noise and blur, used to assess characterisation robustness under challenging acquisition conditions.

The reason for these three diferent sets of images is to demonstrate the broad applicability of RACE-FPP, which yields good characterisation results under various conditions. Each set is processed independently using both RACE-FPP and OpenCV’s checkerboard detection implementation. The evaluation is conducted using two complementary metrics. First, we directly compare the predicted keypoints with the manually annotated reference keypoints described in the Annotation strategy subsection to quantify pixel-level detection accuracy. Second, the reprojection error is used to assess the camera characterisation accuracy, measuring the discrepancy between observed image points and their projections based on estimated parameters.

Centre-based projector characterisation Similar to the camera, the projector characterisation can be performed based on the pinhole model (Figure 1(c)), where the corresponding 2D projector points are represented as $p _ { p } .$ Assuming the projector behaves as an inverse imaging system that follows the same pinhole model, it can be characterised using a planar checkerboard pattern in accordance with Zhang and Huang’s method [43]. Unlike a camera, however, the projector does not directly observe the calibration target. Correspondences between points observed by the camera and coordinates in the projector image plane must therefore be established indirectly from the projected fringe patterns, as shown in Equation 6.

![](images/665b332ca21bfaff0b43ae06bf5d26657fef1b3ae3fbbfb39b1d10a6fd4c31d7.jpg)

![](images/fe4ea9426fb89eb958073458319ecc1e6b3cbf21011b9b7e7d8e0d07dab5efe5.jpg)  
Fig. 2. The modulation of the high-frequency horizontal fringes in a sample image: (a) a selected profile region in the image and (b) the intensity modulation in the checkerboard square centre and corners.

Two orthogonal multi-frequency sinusoidal fringe patterns are sequentially projected onto the checkerboard. The fringes are projected horizontally (H) and vertically (V) to obtain the projector pixel coordinates $( u _ { p } , \nu _ { p } )$ corresponding to each camera pixel. The use of multiple fringe frequencies enables reliable phase unwrapping and recovery of the absolute phase map. Let $\left( u _ { c } , \nu _ { c } \right)$ denote a point in the camera image coordinates and $( u _ { p } , \nu _ { p } )$ the corresponding projector coordinates. For each fringes orientation $d \in \{ \mathrm { V } , \mathrm { H } \}$ , an �-step phase-shifting sequence, $\delta _ { i }$ , with known phase shifts is projected and captured:

$$
\delta _ { i } = \frac { 2 \pi ( i - 1 ) } { N } , \quad i = 1 , \ldots , N .\tag{2}
$$

The recorded intensity at each pixel is given by:

$$
\begin{array} { c } { { I _ { i } ^ { ( d ) } ( u _ { c } , \nu _ { c } ) = A ^ { ( d ) } ( u _ { c } , \nu _ { c } ) + B ^ { ( d ) } ( u _ { c } , \nu _ { c } ) } } \\ { { \times \cos \big ( \phi ^ { ( d ) } ( u _ { c } , \nu _ { c } ) + \delta _ { i } \big ) . } } \end{array}\tag{3}
$$

where $A ^ { ( d ) } ( u _ { c } , \nu _ { c } )$ and $B ^ { ( d ) } ( u _ { c } , \nu _ { c } )$ represent the background intensity and modulation amplitude, respectively, and $\phi ^ { ( d ) } ( u _ { c } , \nu _ { c } )$ is the phase to be recovered. Vertical fringes $\begin{array} { l c l } { ( d } & { = } & { \mathrm { V } ) } \end{array}$ encode the projector $u _ { p }$ coordinate, while horizontal fringes $( d = \mathrm { H } )$ encode the projector $\nu _ { p }$ coordinate. The wrapped phase (Equation 4) is first obtained from Equation 3 using the phase-shifting demodulation via the least-squares method and, subsequently, unwrapped to yield the absolute phase $( \Phi ^ { ( d ) } ( u _ { c } , \nu _ { c } ) )$ by adding an integer that represents the ”fringe order” (the number of cycles that the phase has wrapped), as shown in Equation 5:

$$
\phi ^ { ( d ) } ( u _ { c } , \nu _ { c } ) = t a n ^ { - 1 } \frac { \sum _ { i = 1 } ^ { N } I _ { i } ^ { ( d ) } ( u _ { c } , \nu _ { c } ) s i n ( \delta _ { i } ) } { \sum _ { i = 1 } ^ { N } I _ { i } ^ { ( d ) } ( u _ { c } , \nu _ { c } ) c o s ( \delta _ { i } ) }\tag{4}
$$

$$
\Phi ^ { ( d ) } ( u _ { c } , \nu _ { c } ) = \phi ^ { ( d ) } ( u _ { c } , \nu _ { c } ) + 2 \pi K ( u _ { c } , \nu _ { c } )\tag{5}
$$

The projector coordinates are then computed as:

$$
u _ { p } = \frac { \Phi ^ { \mathrm { ( V ) } } ( u _ { c } , \nu _ { c } ) } { 2 \pi f _ { \mathrm { V } } } , \quad \nu _ { p } = \frac { \Phi ^ { \mathrm { ( H ) } } ( u _ { c } , \nu _ { c } ) } { 2 \pi f _ { \mathrm { H } } } ,\tag{6}
$$

where $f _ { \mathrm { V } }$ and �<sub>H</sub> denote the spatial carrier frequencies of the vertical and horizontal fringe patterns, respectively.

In contrast to traditional methods that assign phase values to the checkerboard’s corner points, this work associates the phase information with the centres of the white squares. This strategy enhances the robustness of the phase data because the white areas exhibit smoother and more uniform fringe modulation, while the corner regions include greylevel transitions that introduce phase ambiguities, as can be seen in Figure 2. In conventional checkerboardbased projector characterisation, the phase values used to determine projector coordinates are commonly evaluated at the checkerboard corner locations. These positions coincide with abrupt transitions between black and white regions of the target. Such reflectivity discontinuities can reduce local fringe modulation and introduce underexposure, saturation, or low signal-to-noise ratio, thereby increasing uncertainty in the recovered phase [37, 12].

To reduce sensitivity to these efects, the proposed framework associates the projector phase information with the centres of the white checkerboard squares rather than directly with the checkerboard corners. As illustrated in Figure 2, the white-square interiors provide more spatially uniform reflective regions and exhibit stronger fringe modulation than the black-white transition regions.

The square centre coordinates are derived from the checkerboard corner points detected in the previous stage. The centre position of the white square is computed as the arithmetic mean of the coordinates of its four surrounding corners. Since the detected corner locations are usually sub-pixel values, the resulting centre does not usually coincide with a discrete pixel location in the phase map. Consequently, the phase value at the square centre is not directly available and must be estimated from the surrounding phase-map pixels. To obtain a sub-pixel phase estimate, bilinear interpolation is applied. For a centre located at $( x _ { s } , y _ { s } )$ , the interpolated phase $\Phi ( x _ { s } , y _ { s } )$ is computed as:

$$
\Phi ( x _ { s } , y _ { s } ) = \sum _ { i = 1 } ^ { 2 } \sum _ { j = 1 } ^ { 2 } w _ { i j } \Phi ( x _ { i } , y _ { j } ) ,\tag{7}
$$

where $( x _ { i } , y _ { j } )$ denote the four neighbouring phasemap pixels surrounding the centre location, and $w _ { i j }$ are interpolation weights determined by the relative distances between the centre location and these pixels, satisfying $\begin{array} { r } { \sum _ { i , j } w _ { i j } \ = \ 1 } \end{array}$ Neighbouring pixels closer to the centre are assigned larger interpolation weights than those farther away.

The projector characterisation followed the same image grouping strategy adopted for camera characterisation, using the clear, mixed, and noisy datasets. For each set, intrinsic and extrinsic parameters are estimated independently. The reprojection error of each configuration is used to compare the proposed square-centre approach against the conventional corner-based method implemented in OpenCV.

Following the individual characterisation of the camera and projector, stereo characterisation is performed to estimate the relative transformation between them. This step determines the rotation matrix and translation vector $( \mathsf { R } , \mathsf { t } ) _ { p c }$ that define the projector pose relative to the camera, which, in this work, serves as the reference coordinate system for subsequent reconstruction. These parameters are essential for triangulating 3D points during reconstruction.

Using the estimated camera, projector, and stereo characterisation parameters together with the recovered absolute phase maps, 3D reconstruction is then performed. The characterisation parameters define the geometric relationship between the devices, while the absolute phase establishes correspondences between camera pixels and projector coordinates. The resulting reconstructed geometries are compared against reference measurements obtained using a Coordinate Measuring Machine (CMM; Mitutoyo Crysta Apex S7106) and analysed in terms of dimensional and geometric deviations, following established approaches for the metrological validation of optical 3D measurement systems [44, 2]. This provides a practical validation of the complete characterisation pipeline.

## Deep learning model

In this work, the checkerboard corner detection problem is reformulated as a keypoint detection task, where each corner is treated as an independent target. A YOLOv11 pose estimation model is adopted as the base architecture due to its eficient inference and improved localisation capability compared to earlier YOLO variants [45]. YOLOv11 follows a unified backbone–neck–head architecture. The backbone extracts visual features from the input images at multiple scales via its internal layers; the neck combines information from diferent resolutions; and the pose head predicts the locations of the checkerboard corner keypoints. This design enables robust localisation under varying image conditions and checkerboard poses.

Dataset preparation The keypoint detection of the deep learning model heavily relies on the availability of an accurate labelled training dataset. In this work, the dataset includes both original images and augmented variations to increase robustness to diferent imaging conditions such as blur and noise.

High-resolution images (3000 × 2000 pixels) of a characterisation target are captured using a fringe projection system to train the deep learning model. The characterisation plate is a $1 2 \times 1 3$ checkerboard pattern, resulting in 132 interior corner points with a square size of 6 mm. The checkerboard is captured 200 times in diferent positions and orientations relative to the camera to ensure geometric diversity. The images were taken using an FPP system, developed by Taraz Metrology Ltd, with a Digital Light Processing (DLP) projector that emits sinusoidal fringe patterns generated by a bespoke high-performance computer (Windows 10, 64- bit architecture, and 32 GB RAM). The imaging camera employs a 16 mm focal length lens with an F/1.8 aperture and provides a field of view (FOV) of approximately 255 mm × 174 mm. To improve training generalisation and better reflect realistic FPP imaging conditions, the dataset was augmented by synthetically applying a range of photometric and blur-related degradations to the original checkerboard images. These included changes in brightness and contrast, gamma adjustment, Poisson and Gaussian noise, and both full-frame and directional defocus blur. Several augmentations were also applied in combination, as shown in Figure 3.

Annotation strategy Each checkerboard image was annotated using LabelMe [46], an open-source image annotation tool that enables annotations to be created and stored locally. Manual annotation was adopted because no automatic corner detection method could be assumed to provide errorfree ground-truth labels under the challenging imaging conditions considered in this work. The annotations were performed at high image magnification and subsequently reviewed to maintain consistent corner placement and ordering. In the adopted point-wise formulation, each internal checkerboard corner is treated as an independent instance; consequently, each label file contains 132 annotation lines, one for each of the 132 internal corners. Each annotation consists of a bounding box and its associated keypoint coordinate in the form: class $x _ { b o x }$ ��<sub>��</sub> ��<sub>��</sub> $h _ { b o x } \ x _ { k } \ y _ { k } \ \nu _ { k }$ , where class is a number representing the object class (in our case, it is set to 0 as we only have one class, the checkerboard), $x _ { b o x } , y _ { b o x } , w _ { b o x } , h _ { b o x }$ denote the normalised centre and size of the instance’s bounding box, and each triplet $( x _ { k } , y _ { k } , \nu _ { k } )$ represents the normalised keypoint coordinates and their visibility flag, where $\nu _ { k } ~ \in ~ \{ 0 , 1 , 2 \}$ A value of $\nu _ { k } = 2$ indicates the keypoint is visible, $\nu _ { k } = 1$ means labelled but occluded, and $\scriptstyle \nu _ { k } = 0$ marks absent points. Most of our checkerboard corners remain fully visible within the FOV and are therefore assigned a visibility value of2, following the standard YOLO pose annotation convention [47]. For a few captured poses, however, portions of the checkerboard extend outside the image boundaries, resulting in invisible points. Maintaining a consistent annotation structure across all 132 corners is essential to preserving stable point correspondence during training.

(a) Raw image  
![](images/f8819c7bbd81b52a58c725a82df3d6044827fb63c7f32eb6002941a1da261d7a.jpg)

(b) Brightness change  
![](images/2e06e5ebd446b9fb9445c42a369470e6284ac48d062bbbfc591ab3a49324953d.jpg)  
(e) Lens blur

(c) Gaussian noise  
![](images/c49eb6318e3cede8a2dadcc1088eb00d0b08f3a4b435ca9f81d8ad41de699d32.jpg)  
(f) Lens blur + Gaussian

(d) Poisson noise  
![](images/f2ae034abf0e797ff997f1a588703e7dfa5ecda937bfb8de9bfb6148cca3e009.jpg)  
(g) Brightness change + Gaussian + Poisson  
Fig. 3. Dataset samples used for training with: (a) the original raw image and (b) - (g) the added noise types for the same image in a unified area.

Unlike conventional checkerboard detection, which requires successful identification of the complete predefined corner pattern for an image to be accepted, the proposed approach adapts the target representation by treating each checkerboard corner as an independent detection target, without modifying the underlying network architecture. Consequently, local degradation afecting individual corners does not necessarily invalidate detections obtained from the remaining regions of the checkerboard, while confidence scores provide an additional measure for filtering uncertain predictions. The practical efect of this point-wise formulation on the number of successfully detected characterisation images is evaluated later in Table 3.

Model training and testing The YOLOv11 pose estimation network is trained using the annotated checkerboard dataset described in the previous section. The dataset is divided into 80% for training and 20% for validation, ensuring that both subsets contain data spanning the full range of imaging conditions. Instead of dedicating a percentage of the dataset to testing, the trained model is later evaluated on real characterisation images against groundtruth point correspondences. The training is performed using the YOLOv11 framework built on PyTorch with an NVIDIA Quadro P5000 Graphics Processing Unit (GPU).

A higher input resolution (1440 × 960 pixels) than the commonly used YOLO settings (usually $6 4 0 ~ \times ~ 6 4 0$ pixels [48]) is selected to preserve better checkerboard corner details required for accurate characterisation, while remaining within the available GPU memory constraints. The training hyperparameters are set to 200 epochs with a batch size of four at a 0.001 learning rate, decaying linearly over the training schedule. Hyperparameters are empirically tuned based on validation performance. AdamW is used as the optimiser. Training employed early stopping with a patience of 20 epochs, terminating when the validation performance no longer improved, thereby preventing overfitting.

The model training is optimised using five loss components associated with bounding box localisation, keypoint localisation, keypoint objectness, classification, and distribution focal loss. Since the present application focuses on checkerboard corner localisation, particular attention is given to the pose and keypoint objectness losses. During training, model performance is monitored using the validation losses together with precision, recall, and mean average precision (mAP) [48]. Precision measures the proportion of predicted detections that are correct, whereas recall measures the proportion of ground-truth instances that are successfully detected. For keypoint evaluation, mAP summarises the precision–recall performance based on intersection-over-union (IoU), with mAP@0.5 evaluated at an IoU threshold of 0.5 and mAP@0.5:0.95 averaged over IoU thresholds from 0.5 to 0.95. Once trained, the model predictions are evaluated against manually annotated reference keypoints prior to characterisation. The reference keypoints are obtained using the annotation procedure described in the Annotation strategy subsection. The predictions are also compared with those obtained using OpenCV’s conventional checkerboard detection algorithm.

(a)  
![](images/236aa3ebd4c75a3aa9e9ac0d6b7061adcd739030f38e29a5fcf61b49fd832c4f.jpg)

(b)  
![](images/a7135a27106a5d088a3a7a64109a88c4113fa4dc87945535df01b15569e8a1e1.jpg)

(c)  
![](images/e8faeb023b9c4bc81f6a13a36476c1164b713913b35970c6f0875d42037a01c5.jpg)

(d)  
![](images/999bf8c4dba754f088c63911fe71890ca25185ee0cbee80d072fec4e2fa25a9c.jpg)  
Fig. 4. YOLOv11 pose estimation model metrics: (a) and (b) are the pose and keypoint objectness losses for training optimisation, respectively, and (c) and (d) are precision and recall, and Mean Average Precision graphs, respectively, fo performance evaluation.

## Results and discussion

Keypoint detection performance The training behaviour of the corner-detection model is first examined to assess convergence before evaluating its localisation performance on the independent characterisation datasets. Figure 4 represents the evolution of the training and validation pose and keyobjectness losses (Figure 4(a) and (b)), precision and recall (Figure 4(c)), and mean average precision (mAP)

(Figure 4(d)). Training terminated after 110 epochs (25.231 hours) through the early-stopping criterion, when no further improvement was observed over the specified patience period.

As seen in Figure 4(a) and (b), the training losses for pose and keypoint objectness decreased smoothly from approximately 0.208 and 0.17 to $2 . 9 3 \ \times 1 0 ^ { - 3 }$ and $9 . 4 \times 1 0 ^ { - 4 }$ , respectively, before converging toward stable values as the model progressively learned more accurate corner localisation patterns from the training data. The corresponding validation curves exhibit similar convergence behaviour and remain close to the training curves, with no evident divergence as training progresses. These losses’ trends indicate stable optimisation of the model on both the training and validation datasets.

The detection metrics in Figure 4(c) show a rapid improvement during the initial training epochs before stabilising, with final precision and recall values of approximately 99.8% and 98.7%, respectively. The high precision indicates that few predicted corner instances are false detections, while the recall indicates that the majority of reference corner instances are successfully detected. Figure 4(d) further shows that both mAP@0.5 and mAP@0.5:0.95 converge to approximately 99.5%. The close agreement between the two mAP measures indicates that the high detection performance is maintained across the evaluated IoU thresholds.

Table 1: Percentage of correct keypoints (PCK) at diferent thresholds for RACE-FPP and OpenCV detections.
<table><tr><td rowspan="2">Image dataset</td><td colspan="4">RACE-FPP</td><td colspan="4">OpenCV</td></tr><tr><td>PCK@0.5</td><td>PCK@1</td><td>PCK@5</td><td>PCK@10</td><td>PCK@0.5</td><td>PCK@1</td><td>PCK@5</td><td>PCK@10</td></tr><tr><td>Noise-free</td><td>19.70</td><td>100.00</td><td>100.00</td><td>100.00</td><td>12.12</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>Mixed images</td><td>16.67</td><td>58.33</td><td>100.00</td><td>100.00</td><td>16.67</td><td>49.24</td><td>96.21</td><td>97.73</td></tr><tr><td>Noisy images</td><td>3.03</td><td>19.70</td><td>99.24</td><td>100.00</td><td>5.30</td><td>15.15</td><td>77.27</td><td>87.88</td></tr></table>

To complement these global metrics, the network’s spatial accuracy was further examined using the Percentage of Correct Keypoints (PCK), a standard metric in keypoint detection and pose estimation tasks [49]. PCK@� denotes the percentage of detected keypoints whose localisation error is within a threshold � (in pixels):

$$
\mathrm { P C K @ } t = \frac { N _ { \mathrm { c o r r e c t } } } { N _ { \mathrm { t o t a l } } } \times 1 0 0 \% ,\tag{8}
$$

where $N _ { \mathrm { c o r r e c t } }$ is the number of keypoints with localisation error below the threshold �. In this study, thresholds of five pixels and 10 pixels (px) were initially adopted to represent moderate and relaxed localisation tolerances relative to the image resolution, enabling the detector’s robustness to be evaluated under increasingly degraded imaging conditions.

The resulting PCK@5 px and PCK@10 px values reached 100% across the noise-free and mixed datasets and over 99% for noisy images, as illustrated in Table 1, indicating that almost all detected corners were localised within five pixels of the reference annotations. This outcome aligns with the high precision and recall trends observed during training, reinforcing the model’s spatial consistency and prediction reliability. At stricter localisation thresholds, RACE-FPP achieves comparable or higher PCK values than OpenCV at 1 px, while both methods exhibit relatively low PCK at 0.5 px. This indicates that, when corners are successfully detected, the localisation performance of the two methods becomes more comparable at demanding sub-pixel thresholds. However, RACE-FPP maintains successful detection of the complete checkerboard across the investigated image conditions, as will be seen in the Camera characterisation subsection, providing a more consistent set of image observations for the subsequent characterisation stages.

Corner localisation accuracy and robustness To further evaluate the spatial accuracy of the proposed deep learning detector, its keypoint predictions are compared against the outputs of OpenCV’s findChessboardCornersSB()

function and the manually annotated ground truth. The analysis uses three representative test images acquired with a fixed checkerboard pose, representing diferent image qualities -noise-free, partial defocus, and full defocus images- to assess detection robustness under diferent characterisation scenarios. Figure 5 presents the resulting corner locations superimposed on the corresponding checkerboard images. RACE-FPP predictions are marked in green, the OpenCV detections in red, and the ground truth (GT) corners in blue. The zoomed-in region is kept consistent across all three images to facilitate a direct comparison of corner localisation under diferent image degradations. The partially defocused image is particularly representative of practical characterisation conditions, where variations in depth and viewing geometry result in focused and defocused regions within the same image. Visual inspection reveals that the proposed model reliably identifies all 132 corner points with accurate spatial alignment for all the images, while the OpenCV method exhibits localisation deviations, particularly in degraded regions. Inaccurate detections are observed for OpenCV under non-ideal conditions in Figure 5 (b) and (c), as will be discussed further in the following subsection. Interestingly, the OpenCV detections in the partially defocused image exhibit localisation errors comparable to, and in some locations greater than, those observed under full-image defocus. This behaviour may be attributed to the transition region containing spatially varying blur, producing locally inconsistent image gradients on which conventional gradient-based corner detectors rely. In contrast, the fully defocused image exhibits more uniform blur characteristics, despite the overall loss of sharpness. Similar challenges associated with spatially varying defocus have been reported in the literature on blur estimation and image feature analysis [50, 51, 52].

For a quantitative assessment, the per-point localisation (Euclidean) error $( e _ { i } )$ , defined in Equation 9, is computed for each detected corner, providing a direct measure of geometric accuracy in image-pixel coordinates [49]. For each checkerboard corner, the predicted keypoint is matched to its corresponding ground-truth point using the predefined grid ordering, and the resulting localisation error directional components are visualised in Figure 6 for the representative checkerboard images presented in Figure 5. The localisation errors are then summarised by their mean and standard deviation over all 132 checkerboard corners and reported in Table 2, along with Euclidean magnitudes. The error statistics for RACE-FPP and the OpenCV detections in the horizontal and vertical directions, together with the overall mean Euclidean error, are presented.

![](images/efaf14530f653ab8b01f37f5082d84b1dd6acf1744162d51b790977ccd6c0ffd.jpg)  
Fig. 5. Comparison of checkerboard corner localisation using the proposed method (green), OpenCV (red), and manually annotated ground truth (blue) under three representative image qualities: (a) noise-free image, (b) partial horizontal defocus, and (c) severe full-image defocus.

$$
e _ { i } = \sqrt { ( x _ { i } ^ { p r e d } - x _ { i } ^ { g t } ) ^ { 2 } + ( y _ { i } ^ { p r e d } - y _ { i } ^ { g t } ) ^ { 2 } } ,\tag{9}
$$

where $x _ { i } ^ { p r e d } , y _ { i } ^ { p r e d } , x _ { i } ^ { g t }$ and $y _ { i } ^ { g t }$ indicate the $i ^ { t h }$ point’s coordinates from the predictions and the ground truth values, respectively.

Figure 6 illustrates the spatial distribution of localisation errors for both methods across the same images in Figure 5. RACE-FPP errors are labeled in green, and the OpenCV ones are in red. Figure 6(a) shows compact localisationerror distributions for both methods under noise-free conditions, with most errors remaining below one pixel. Both distributions exhibit a similar systematic ofset from the origin, with mean directional errors of approximately −0.4 px in both axes (Table 2). The close agreement between the ofsets observed for RACE-FPP and OpenCV indicates that this displacement is common to both methods relative to the manually annotated reference, rather than being specific to the proposed detector. In contrast, the OpenCV detections exhibit larger variations and noticeable outliers reaching almost 18 pixels when blurry conditions exist in Figure 6(b) and (c). The largest OpenCV errors are concentrated in a small number of corner points rather than being uniformly distributed across the checkerboard, indicating that localisation failures occur locally in degraded image regions. These outliers correspond to incorrectly localised corner points, which typically occur in regions afected by image degradation. Compared with OpenCV, the proposed method exhibits a tighter clustering of localisation errors without the presence of extreme outliers in Figure 6(b) and (c), indicating that the proposed detector maintains consistently low localisation errors even when conventional gradient-based corner detection produces isolated large localisation failures.

These observations are quantitatively confirmed by the error statistics reported in Table 2, where RACE-FPP consistently exhibits lower mean localisation errors and substantially smaller standard deviations than OpenCV, particularly under partial and severe defocus.

The trends observed qualitatively in Figures 5 and 6 are quantitatively supported by the error statistics reported in Table 2. Under noise-free imaging conditions, both methods exhibit similar localisation performance, with mean Euclidean errors of 0.63 px and 0.64 px for RACE-FPP and OpenCV, respectively. The corresponding mean directional errors are also nearly identical, with $D _ { x } = - 0 . 4 2$ px and $D _ { \mathrm { y } } = - 0 . 4 2 \ : \mathrm { p x }$ for RACE-FPP and $D _ { x } = - 0 . 4 3 \ : \mathrm { p x }$ and $D _ { \mathrm { y } } = - 0 . 4 4$ px for OpenCV, confirming the common ofset observed in Figure 6(a). As the image degradation increases from partial to full defocus, the localisation error increases for both methods. However, RACE-FPP consistently maintains lower mean Euclidean errors of 1.22 px and 2.00 px than OpenCV, compared with 2.27 px and 3.30 px for OpenCV. Moreover, the proposed method exhibits substantially smaller standard deviations under both degraded conditions (0.94 px and 0.86 px) than OpenCV (4.47 px and 4.43 px). This indicates that, beyond achieving a lower average localisation error, the proposed detector provides considerably more consistent corner localisation with far fewer large localisation failures. These observations are consistent with the tighter error distributions and reduced number of outliers shown in Figure 6, confirming that RACE-FPP maintains stable localisation performance even under challenging imaging conditions.

![](images/4faa9d9b4777d45056b3b83b37ab9e80a2ed86b7f6a748a51d51aa3e7d94b07f.jpg)

![](images/941ddef3485c13dc97bb51bb159ad5275142cb2d2f9ff01927c7f8c39756d12d.jpg)

![](images/1fcaf388f34648b9b4c7c80dbed01c803bbfb56164188f544fa3df1be1d61499.jpg)  
Fig. 6. Spatial distribution of the directional localisation errors $( D _ { x } , D _ { y } )$ relative to the manually annotated ground truth for the same representative checkerboard images shown in Figure 5.

Table 2: Mean and standard deviation (std) of the directional $( D _ { x } , D _ { y } )$ and Euclidean localisation errors in pixels (px) relative to the manually annotated ground truth (GT).
<table><tr><td rowspan="3">Error</td><td colspan="6">RACE-FPP</td><td colspan="6">OpenCV</td></tr><tr><td colspan="2">Noise-free</td><td colspan="2">partial defocus</td><td colspan="2">full defocus</td><td colspan="2">Noise-free</td><td colspan="2">partial defocus</td><td colspan="2">full defocus</td></tr><tr><td>Mean</td><td>std</td><td>Mean</td><td>std</td><td>Mean</td><td>std</td><td>Mean</td><td>std</td><td>Mean</td><td>std</td><td>Mean</td><td>std</td></tr><tr><td>Dx</td><td>-0.42</td><td>0.23</td><td>-0.28</td><td>1.17</td><td>1.02</td><td>1.32</td><td>-0.43</td><td>0.20</td><td>0.17</td><td>3.41</td><td>1.00</td><td>3.71</td></tr><tr><td>Dy</td><td>-0.42</td><td>0.16</td><td>-0.12</td><td>0.96</td><td>0.34</td><td>1.36</td><td>-0.44</td><td>0.13</td><td>0.34</td><td>3.65</td><td>1.00</td><td>3.84</td></tr><tr><td>Euclidean</td><td>0.63</td><td>0.19</td><td>1.22</td><td>0.94</td><td>2.00</td><td>0.86</td><td>0.64</td><td>0.16</td><td>2.27</td><td>4.47</td><td>3.30</td><td>4.43</td></tr></table>

Camera characterisation To assess the impact of the proposed model on camera characterisation accuracy, three separate characterisations are performed using (i) noise-free images, (ii) a mixed set containing noise-free and noisy images, and (iii) noisy images only. Each characterisation is executed twice: once using RACE-FPP predicted keypoints, and the other using the points detected by OpenCV’s findChessboardCornersSB(). The characterisation follows the standard pinhole camera model with radial and tangential distortion terms [9]. The root mean square reprojection error (RPE), in Equation 10, is used to quantify the overall geometric accuracy.

$$
\mathrm { R P E } = \sqrt { \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( p _ { i } ^ { \mathrm { d e t } } - p _ { i } ^ { \mathrm { e s t } } ) ^ { 2 } } ,\tag{10}
$$

where $p _ { i } ^ { \mathrm { d e t } }$ and $p _ { i } ^ { \mathrm { e s t } }$ are the coordinates of the $i ^ { t h }$ point in the $j ^ { t h }$ image of the detected image point (YOLO prediction or OpenCV detection) and the corresponding reprojected point obtained using the estimated characterisation parameters, respectively; � and � are the total number of points and images, respectively. The resulting errors for all three datasets are summarised in Table 3.

In addition to reprojection accuracy, the number of characterisation images successfully processed was evaluated, as a complete set of checkerboard correspondences is required for each pose included in the camera characterisation. RACE-FPP successfully provided the required correspondences for all 16 images in each dataset. The OpenCV baseline also processed all 16 noise-free images, but this number decreased to 10/16 for the mixed dataset and 5/16 for the noisy dataset. Thus, the efect of image degradation was observed not only in the localisation accuracy of the detected corners, but also in the number of poses available for characterisation. Moreover, some of the degraded images successfully accepted by OpenCV exhibited comparatively large per-image reprojection errors. For example, the maximum per-image root mean square (RMS) reprojection error increased from 0.124 px in the noise-free dataset to 3.801 px and 4.788 px in the mixed and noisy datasets, respectively. In contrast, RACE-FPP retained all characterisation poses across the three imaging conditions, demonstrating greater robustness to the imposed image degradation.

Table 3: Detection rates and reprojection error (RPE) in pixels (px) for camera characterisation using RACE-FPP and OpenCV pipelines.
<table><tr><td>Dataset</td><td>Images detected (RACE-FPP)</td><td>RPE (RACE-FPP)</td><td>Images detected (OpenCV)</td><td>RPE (OpenCV)</td></tr><tr><td>Noise-free</td><td>16/16</td><td>0.2281</td><td>16/16</td><td>0.0806</td></tr><tr><td>Mixed images</td><td>16/16</td><td>0.2591</td><td>10/16</td><td>1.2365</td></tr><tr><td>Noisy images</td><td>16/16</td><td>0.4500</td><td>5/16</td><td>2.8514</td></tr></table>

![](images/7309f57729a86a03051fcd997ef30fa51cc1b79b27221e948abde200064a2231.jpg)  
Fig. 7. Visualisation of projected 3D points using the estimated camera intrinsic and extrinsic parameters. Areas of diferent defocus within the image are zoomed in.

The reprojection errors for both methods were calculated using the same OpenCV camera characterisation routine and evaluation procedure. Under noise-free conditions, OpenCV achieves a lower global RMS reprojection error of 0.0806 px, compared with 0.2281 px for RACE-FPP. This diference can be attributed, at least in part, to the diferent localisation strategies. The OpenCV checkerboard detector is specifically designed for geometric corner localisation and provides sub-pixel corner estimates, which is advantageous when the checkerboard edges and corner transitions are clearly resolved. In contrast, RACE-FPP estimates corner locations through a learned keypoint representation trained across a broad range of image degradation conditions. Consequently, under favourable noise-free conditions, the specialised OpenCV detector provides more precise corner localisation, while both methods retain sub-pixel reprojection accuracy. RACE-FPP is primarily developed to address corner localisation under degraded FPP imaging conditions, for which conventional checkerboard detection becomes unreliable. As the imaging conditions deteriorate, however, the RACE-FPP characterisations yield global RMS errors of 0.2591 px and 0.4500 px for the mixed and noisy datasets, respectively, whereas the corresponding OpenCV errors increase to 1.2365 px and 2.8514 px. These values should be interpreted together with the detection success rates: the OpenCV errors under degraded conditions are calculated only from the successfully detected subsets (10/16 and 5/16 poses), whereas RACE-FPP uses all 16 poses in each case. Consequently, the degraded-condition reprojection errors do not constitute a point-for-point comparison over identical pose subsets; rather, they characterise the calibration obtained from the correspondences that each method successfully provides. Taken together, the pose-retention rate and reprojection results show a trade-of between the higher precision of OpenCV under noise-free conditions and the greater robustness of RACE-FPP as image quality deteriorates.

Table 4: Estimated camera intrinsic parameters obtained using the RACE-FPP and OpenCV pipeline for the three image datasets.
<table><tr><td>Method</td><td>Dataset</td><td> $f _ { x } \times 1 0 ^ { 3 }$  px</td><td> $f _ { \mathrm { y } } \times 1 0 ^ { 3 } \ : \mathrm { l }$ </td><td>0X cx × 103 px</td><td> $c _ { \mathrm { y } } \times 1 0 ^ { 3 } ~ \mathrm { p x }$ </td></tr><tr><td rowspan="3">RACE-FPP</td><td>Noise-free</td><td>6.708</td><td>6.705</td><td>1.432</td><td>0.910</td></tr><tr><td>Mixed images</td><td>6.733</td><td>6.728</td><td>1.429</td><td>0.896</td></tr><tr><td>Noisy images</td><td>6.714</td><td>6.711</td><td>1.420</td><td>0.903</td></tr><tr><td rowspan="3">OpenCV</td><td>Noise-free</td><td>6.712</td><td>6.711</td><td>1.446</td><td>0.913</td></tr><tr><td>Mixed images</td><td>6.597</td><td>6.634</td><td>1.397</td><td>1.018</td></tr><tr><td>Noisy images</td><td>6.352</td><td>6.472</td><td>1.293</td><td>1.207</td></tr></table>

Figure 7 provides a visual illustration of the projection of the 3D points onto the checkerboard. An image with a tilted checkerboard pose and partial defocus is selected as an example, as it reflects realistic FPP imaging conditions. The figure displays the projected points using the parameter matrices estimated from the two pipelines: RACE-FPP is indicated by green crosses, and OpenCV by red triangles. The reference points, manually annotated using the LabelMe image annotation tool, are shown as blue circles to enable a comprehensive assessment, clearly highlighting the deviation of the OpenCV points in the blurred regions of the image.

Beyond reprojection accuracy, the intrinsic parameters estimated using RACE-FPP exhibited greater stability across all three datasets. The estimated focal lengths $( f _ { x } ,$ $f _ { y } )$ , principal point coordinates $( c _ { x } , \ c _ { y } )$ , and distortion coeficients showed only minor variations between the three independent characterisations, whereas noticeably larger fluctuations were observed for the OpenCV-based pipeline under degraded imaging conditions. The estimated camera intrinsic parameters for the three datasets are summarised in Table 4, confirming that RACE-FPP yields more stable camera intrinsic parameters across the diferent imaging conditions.

All the aforementioned findings underline the link between detection accuracy and camera characterisation reliability. Accurate and structurally coherent keypoints not only minimise reprojection error but also ensure stable parameter estimation even when the imaging conditions deviate from ideal laboratory settings. In practical terms, this means that a single model trained on diverse image conditions can efectively synergise characterisation routines that traditionally rely on carefully controlled data acquisition.

Projector and stereo characterisation Following camera characterisation, the performance of the proposed approach is further evaluated through projector and stereo characterisation. As described in the Centre-based projector characterisation subsection, the detected checkerboard corners are used to determine the centres ofthe white squares by averaging the coordinates of their four surrounding corners. The corresponding absolute phase values are then obtained at these sub-pixel centre locations using bilinear interpolation and mapped to projector coordinates. This centre-based strategy enables phase information to be sampled within the white-square regions, where stronger fringe modulation is observed, rather than directly at checkerboard corners. The projector parameters are subsequently estimated using OpenCV’s calibrateCamera function. Stereo characterisation then determines the relative rotation and translation between the camera and projector, with the camera coordinate system used as the reference for subsequent 3D reconstruction. Table 5 summarises the root mean square reprojection errors obtained for projector and stereo characterisation using the same three image datasets evaluated previously.

Table 5: Reprojection error (RPE) in pixels (px) for projector and stereo characterisation using RACE-FPP and the OpenCV pipeline.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">RACE-FPP RPE</td><td colspan="2">OpenCV RPE</td></tr><tr><td>Projector</td><td>Stereo</td><td>Projector</td><td>Stereo</td></tr><tr><td>Noise-free</td><td>0.0512</td><td>0.1646</td><td>0.0357</td><td>0.1323</td></tr><tr><td>Mixed images</td><td>0.3935</td><td>0.3778</td><td>0.8571</td><td>1.6647</td></tr><tr><td>Noisy images</td><td>0.6305</td><td>0.1518</td><td>1.6849</td><td>3.1638</td></tr></table>

Under noise-free conditions, both pipelines achieve low projector and stereo characterisation reprojection errors, with OpenCV yielding slightly lower values of 0.0357 px and 0.1323 px, respectively, compared with 0.0512 px and 0.1646 px for the proposed approach. This behaviour is consistent with the camera characterisation results, where conventional sub-pixel corner localisation performs particularly well for clean, high-contrast checkerboard images. As the image quality deteriorates, however, a clear divergence between the two pipelines is observed. For the mixed dataset, the projector reprojection error obtained using the proposed approach is 0.3935 px compared with 0.8571 px for OpenCV, while the corresponding stereo errors are 0.3778 px and 1.6647 px, respectively. Under the noisy condition, the diference increases further as the proposed approach yields a projector error of 0.6305 px compared with 1.6849 px for OpenCV, while the stereo reprojection error remains at 0.1518 px compared with 3.1638 px for the conventional pipeline.

The improved projector characterisation under degraded imaging conditions benefits from both the robustness of the detected checkerboard features and the centre-based projector correspondence strategy. Deriving each whitesquare centre from four surrounding detected corners reduces the influence of individual localisation deviations, while sampling the corresponding phase within the whitesquare region avoids direct reliance on the checkerboard intensity-transition regions, where weaker fringe modulation is observed. In contrast, the conventional pipeline relies directly on detected checkerboard corners and their corresponding phase values; consequently, degraded corner localisation and less reliable phase information can both afect the resulting projector correspondences.

![](images/a3439c2b6bd3f30901cb28a8d7d743af492f8d6495e6520dec70896c547b7784.jpg)  
Fig. 8. The measured artefacts: (a) raw image and (b) high-frequency fringes image. The dumbbell sphere is on the left, and the multi-feature artefact is on the right in each image.

![](images/fab2dc91549214c93395c6e3dff4137f9da4bc4d76cd1fc995c71758bcad3cc0.jpg)  
Fig. 9. 3D reconstructions obtained using the proposed and conventional characterisation pipelines: (a–c) RACE-FPP and (d–f) Traditional FPP for noise-free, mixed, and noisy image datasets, respectively; (g) and (h) side views of the corresponding noisy-condition reconstructions, highlighting diferences in geometric distortion.

The stereo results further demonstrate the stability of the estimated camera–projector geometry. Although OpenCV achieves a slightly lower stereo reprojection error under noise-free conditions, its error increases substantially for the mixed and noisy datasets, reaching 3.1638 px for the latter. In comparison, the proposed pipeline maintains subpixel stereo reprojection errors across all three conditions. In addition, the proposed detector successfully processes all characterisation images without dataset-specific adjustment of detection parameters, preserving the available characterisation poses under degraded imaging conditions. These results indicate that the proposed characterisation strategy maintains a more stable camera–projector relationship as image quality deteriorates, the dimensional consequences of which are examined through 3D reconstruction in the following subsection.

3D reconstruction and dimensional validation The final validation of the proposed characterisation pipeline is performed through the 3D reconstruction of two artefacts: a precision dumbbell sphere, used as a standard reference artefact, and an additively manufactured component incorporating multiple geometric features (Figure 8). For each image dataset -noise-free, mixed, and noisy- the characterisation parameters are estimated independently, resulting in six 3D reconstructions, as shown in Figure 9. The reconstructed point clouds have an average spatial resolution of approximately 50 ��.

Table 6: Sphere radius and spacing distance (SD) measurements (mm) of the precision dumbbell sphere, with deviations from the corresponding CMM measurements.
<table><tr><td>Feature</td><td>Dataset</td><td>CMM</td><td>RACE-FPP</td><td>Deviation</td><td>Traditional FPP</td><td>Deviation</td></tr><tr><td rowspan="3">SD</td><td>Noise-free</td><td rowspan="3">49.945</td><td>49.89</td><td>-0.056</td><td>49.93</td><td>-0.015</td></tr><tr><td>Mixed images</td><td>49.87</td><td>-0.075</td><td>51.37</td><td>+1.425</td></tr><tr><td>Noisy images</td><td>49.83</td><td>-0.115</td><td>54.33</td><td>+4.385</td></tr><tr><td rowspan="3">Sphere 1 radius</td><td>Noise-free</td><td rowspan="3">4.997</td><td>5.03</td><td>+0.033</td><td>5.01</td><td>+0.013</td></tr><tr><td>Mixed images</td><td>4.99</td><td>-0.007</td><td>5.12</td><td>+0.123</td></tr><tr><td>Noisy images</td><td>5.00</td><td>+0.003</td><td>5.59</td><td>+0.593</td></tr><tr><td rowspan="3">Sphere 2 radius</td><td>Noise-free</td><td></td><td>5.04</td><td>+0.044</td><td>5.04</td><td>+0.044</td></tr><tr><td>Mixed images</td><td>4.996</td><td>5.03</td><td>+0.034</td><td>5.16</td><td>+0.164</td></tr><tr><td>Noisy images</td><td></td><td>5.01</td><td>+0.014</td><td>5.76</td><td>+0.764</td></tr></table>

Table 7: Average hood step-height measurements (mm) for the multi-feature artefact, with standard deviations (std) and deviations from the CMM reference.
<table><tr><td>Dataset</td><td>CMM</td><td>RACE-FPP</td><td>std</td><td>Deviation</td><td>Traditional FPP</td><td>std</td><td>Deviation</td></tr><tr><td>Noise-free</td><td></td><td>4.91</td><td>0.04</td><td>-0.007</td><td>4.90</td><td>0.05</td><td>-0.017</td></tr><tr><td>Mixed images</td><td>4.917</td><td>4.91</td><td>0.03</td><td>-0.007</td><td>5.07</td><td>0.05</td><td>+0.153</td></tr><tr><td>Noisy images</td><td></td><td>4.92</td><td>0.04</td><td>+0.003</td><td>4.97</td><td>0.05</td><td>+0.053</td></tr></table>

The two artefacts provide complementary validation of the reconstruction performance. The precision dumbbell provides a controlled metrological reference through measurements of the sphere radii and centre-to-centre spacing distance (SD), enabling assessment of the reconstructed feature dimensions and relative spacing. The multifeature artefact provides additional geometric validation through measurement of the step height between the planar surfaces of the Hood feature and Upper side, assessing the ability of the reconstruction to preserve the separation between surfaces at diferent depths. Visual inspection indicates that RACE-FPP produces geometrically consistent reconstructions across the diferent datasets, whereas distortions and misalignment become increasingly apparent in the Traditional FPP reconstructions under noisy conditions. In particular, the side view in Figure 9(h) highlights the geometric distortion in the Traditional FPP reconstruction obtained from the noisy dataset. For quantitative evaluation, the dumbbell spheres are fitted using a sphere-fitting algorithm to determine their radii and centreto-centre spacing, while the step height of the multi-feature artefact is determined as the mean perpendicular distance between the hood-top point cloud and the plane fitted to the surrounding upper surface. These measurements are compared with the corresponding CMM reference values, with the resulting deviations reported in Tables 6 and 7.

Table 6 presents the dimensional evaluation of the precision dumbbell sphere, a calibrated reference artefact with known geometry. Under noise-free conditions, both RACE-FPP and Traditional FPP produce comparable measurements with deviations below 0.05 mm, indicating that both characterisation pipelines achieve high reconstruction accuracy when corner localisation is reliable. As the image quality deteriorates, however, the behaviour of the two methods diverges significantly. RACE-FPP maintains stable reconstruction performance across all datasets, with sphere radius deviations remaining within approximately 0.04 mm and the SD deviation increasing only marginally to -0.115 mm under the noisy condition. In contrast, the Traditional FPP geometry exhibits a progressive increase in both local and global dimensional errors, with sphere radius deviations reaching up to 0.764 mm and the SD deviation increasing to 4.385 mm. The larger deviation observed in the SD measurement compared with the sphere radii suggests that the accumulated efects of corner localisation errors become increasingly pronounced when reconstructing the relative positions of spatially separated features. Although both measurements depend on the characterisation accuracy, the sphere radii are estimated from locally fitted surface points, whereas the SD additionally reflects the preservation of the overall reconstructed geometry. Consequently, inaccuracies in the camera, projector, and stereo characterisation are more readily manifested in the centre-to-centre distance than in the fitted sphere radii.

Table 7 presents the step-height measurements obtained from the multi-feature artefact. RACE-FPP results remain consistent across the three characterisation datasets, with measured heights of 4.91, 4.91, and 4.92 mm for the noise-free, mixed, and noisy conditions, respectively, compared with the CMM reference value of 4.917 mm. The corresponding deviations remain within 0.01 mm across all conditions, indicating that the reconstructed depth separation is maintained despite degradation of the characterisation images. In comparison, the Traditional FPP measurements show greater variation, with deviations ranging from −0.02 to +0.15 mm and the largest deviation occurring for the mixed-image dataset. These results provide additional evidence that the proposed characterisation approach maintains dimensional stability under the investigated image-degradation conditions.

The 3D reconstruction and dimensional evaluations demonstrate that both characterisation pipelines provide comparable performance under noise-free conditions, wherea their behaviour diverges substantially as image quality deteriorates. The proposed characterisation framework maintains geometrically consistent reconstructions and bounded dimensional deviations across both reference artefacts, while the conventional pipeline exhibits greater geometric distortion and substantially larger dimensional deviations under degraded conditions. These findings demonstrate that the improved stability of the proposed camera–projector characterisation is reflected in the dimensional fidelity of the reconstructed geometry.

## Conclusion

This work developed a robust characterisation framework for FPP systems operating under non-ideal imaging conditions, particularly where regions of varying image quality coexist within the same frame due to object orientation, depth variation, and optical efects. The framework combines deep learning-based checkerboard corner localisation with a centre-based projector characterisation strategy, while retaining the conventional geometric camera–projector model. White-square centres are derived from the detected checkerboard geometry and used to sample phase information for establishing camera–projector correspondences. The approach therefore modifies the feature-localisation and projector-correspondence stages without replacing the underlying physical characterisation model.

The proposed framework demonstrated substantially greater robustness as image quality deteriorated. Although the conventional OpenCV pipeline achieved slightly lower reprojection errors for clean, high-contrast images, its performance degraded markedly under lens blur and noise, with several checkerboard images no longer successfully detected. In contrast, the proposed method successfully processed all characterisation images and maintained substantially lower reprojection errors under degraded conditions. For the noisy-image dataset, the camera characterisation reprojection error declined from 2.85 px with the conventional pipeline to 0.45 px, while the projector characterisation reprojection error decreased from 1.68 px to 0.63 px. Sub-pixel stereo reprojection errors were maintained across all tested imaging conditions, indicating greater stability of the estimated camera–projector geometry.

The practical significance of this robustness was confirmed through 3D reconstruction and CMM-referenced dimensional measurements. For a precision dumbbell artefact, the centre-to-centre spacing deviation under noisy conditions was -0.115 mm with the proposed framework, compared with 4.385 mm for the conventional pipeline. Measurements of a step height on a multi-feature artefact further demonstrated dimensional stability across the different characterisation conditions, with deviations from the CMM reference remaining within 0.01 mm for the proposed s framework. These results demonstrate that the increased stability of the proposed characterisation framework extends beyond image-domain reprojection metrics to the dimensional fidelity of the reconstructed geometry.

Overall, the results show that combining data-driven feature localisation with centre-based projector correspondences can substantially improve the robustness of the complete FPP characterisation chain while preserving its geometric and physically interpretable formulation. This is particularly relevant to practical dimensional measurements using FPP, where spatially non-uniform blur, noise, and other image degradations cannot always be avoided.

The present study considered a single FPP configuration. Future work will therefore examine transferability across diferent camera resolutions, fields of view, optical geometries, and measurement systems. It will also investigate the influence of training-data composition on the balance between performance under ideal conditions and robustness to image degradation, with the aim of reducing the small accuracy gap observed for clean images without sacrificing performance under challenging acquisition conditions.

## Funding

United Kingdom Research and Innovation - Engineering & Physical Sciences Research Council (EPSRC); Taraz Metrology Ltd., UK.

## Acknowledgement

Thanks to the EPSRC and Taraz Metrology Ltd. for funding this work. Thanks are also extended to Xiangjun Kong and Tibebe Yalew with the EPSRC project number EP/X010929/1. Also, thanks to Mohamed Isa from the Manufacturing Metrology Team (MMT) for the CMM measurements, Mark East from the Centre for Additive Manufacturing (CfAM), University of Nottingham, for 3D printing the multi-feature artefact, and Sean Lau and Qian Kai Loh from Taraz Metrology Ltd for the assistance with the CMM measurement of the dumbbell sphere and FPP system backend work.

## Author contributions

Osman Ali: Conceptualization, data collection, methodology, experiment, analysis, and original draft. Xiangjun Kong: Conceptualization, draft review and editing. Tibebe Yalew: Deep learning model proposal and draft review. Waiel Elmadih: Funding acquisition, industrial supervision, draft review. Samanta Piano: Funding acquisition, draft review and editing.

Conflict of interest

The authors declare no conflict of interest.

## References

[1] Xu, J. & Zhang, S. Status, challenges, and future perspectives of fringe projection profilometry. Optics and Lasers in Engineering 135, 106193 (2020).

[2] Shaheen, A. et al. Characterisation of a multi-view fringe projection system based on the stereo matching of rectified phase maps. Measurement Science and Technology 32, 045006 (2021).

[3] Zhang, S. Flexible and high-accuracy method for unidirectional structured light system calibration. Optics and Lasers in Engineering 143, 106637 (2021).

[4] Liu, J. et al. High-precision calibration to zoom lens of optical measurement machine based on fnn. Optics Express 30, 23511–23530 (2022).

[5] Sun, W. et al. Robust checkerboard recognition for eficient nonplanar geometry registration in projectorcamera systems. In Proceedings of the 5th ACM/IEEE International Workshop on Projector camera systems, 1–7 (2008).

[6] Chen, B., Liu, Y. & Xiong, C. Automatic checkerboard detection for robust camera calibration. In 2021 IEEE International Conference on Multimedia and Expo (ICME), 1–6 (IEEE Computer Society, 2021).

[7] Zhang, W. et al. Sub-pixel projector calibration method for fringe projection profilometry. Optics express 25, 19158–19169 (2017).

[8] Lu, J. et al. Flexible calibration of phase-to-height conversion in fringe projection profilometry. Applied optics 55, 6381–6388 (2016).

[9] Zhang, Z. A flexible new technique for camera calibration. IEEE Transactions on pattern analysis and machine intelligence 22, 1330–1334 (2002).

[10] Wu, H. & Wan, Y. A highly accurate and robust deep checkerboard corner detector. Electronics Letters 57, 317–320 (2021).

[11] Zuo, Z. et al. A checkerboard corner detection method for infrared thermal camera calibration based on physics-informed neural network. In Photonics, Vol. 12, 847 (MDPI, 2025).

[12] Rao, L. & Da, F. Local blur analysis and phase error correction method for fringe projection profilometry systems. Applied optics 57, 4267–4276 (2018).

[13] Duda, A. & Frese, U. Accurate detection and localization of checkerboard corners for calibration. In BMVC, Vol. 126 (2018).

[14] Wang, G., Zheng, H. & Zhang, X. A robust checkerboard corner detection method for camera calibration based on improved yolox. Frontiers in Physics 9, 819019 (2022).

[15] Harris, C., Stephens, M. et al. A combined corner and edge detector. In Alvey vision conference, Vol. 15, 10–5244 (Manchester, UK, 1988).

[16] Wang, W. et al. An improved algorithm for harris corner detection. Optics and Precision Engineering 16, 1995–2001 (2008).

[17] Shang, M. & WANG, K. An image registration algorithm based on multi-scale harris corner detection. Electronics Optics & Control 31, 28–32 (2024).

[18] Shang, S. et al. Research on corner detection algorithm in machine vision. comput. Meas. Control 32, 1–14 (2024).

[19] Wang, M. et al. Harris corner detection algorithm based on pixel point gray diference. Computer Engineering 41, 227–230 (2015).

[20] Feng, J. et al. A feature detection and matching algorithm based on harris algorithm. In 2019 International Conference on Communications, Information System and Computer Engineering (CISCE), 616–621 (IEEE, 2019).

[21] Yuan, Q. et al. Unsupervised-learning-based calibration method in microscopic fringe projection profilometry. Applied Optics 62, 7299–7315 (2023).

[22] Jiang, N. et al. Photohelper: portrait photographing guidance via deep feature retrieval and fusion. IEEE Transactions on Multimedia 25, 2226–2238 (2022).

[23] Zhang, J. et al. Deep-learning-based adaptive camera calibration for various defocusing degrees. Optics Letters 46, 5537–5540 (2021).

[24] Abuolaim, A. & Brown, M. S. Defocus deblurring using dual-pixel data. In European conference on computer vision, 111–126 (Springer, 2020).

[25] Ha, H.-G. et al. Target-specified reference-based deep learning network for joint image deblurring and resolution enhancement in surgical zoom lens camera calibration. Computers in Biology and Medicine 183, 109309 (2024).

[26] Du, F. et al. Correlation-guided attention for corner detection based visual tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6836–6845 (2020).

[27] Dantas, M. S. M. et al. Automatic template detection for camera calibration. Research, Society and Development 11, e173111436168–e173111436168 (2022).

[28] Kang, J. et al. Sparse checkerboard corner detection from global perspective. In 2021 IEEE International Conference on Signal and Image Processing Applications (ICSIPA), 12–17 (IEEE, 2021).

[29] Zhu, H. et al. Sub-pixel checkerboard corner localization for robust vision measurement. IEEE Signal Processing Letters 31, 21–25 (2023).

[30] Redmon, J. et al. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, 779–788 (2016).

[31] Chen, Z. et al. Mngnas: distilling adaptive combination of multiple searched networks for oneshot neural architecture search. IEEE Transactions on pattern analysis and machine intelligence 45, 13489– 13508 (2023).

[32] Xicai, L., Qinqin, W. & Yuanqing, W. Binocular vision calibration method for a long-wavelength infrared camera and a visible spectrum camera with diferent resolutions. Optics Express 29, 3855–3872 (2021).

[33] Zhou, K. et al. Checkerboard corner point detection for enhanced accuracy in fish-eye camera images. The Visual Computer 41, 11261–11274 (2025).

[34] Wu, Z. et al. Review of fringe projection profilometry: from geometric triangulation to computational 3d imaging. Light: Advanced Manufacturing 7, 541–599 (2026).

[35] Li, B., Karpinsky, N. & Zhang, S. Novel calibration method for structured-light system with an out-offocus projector. Appl. Opt. 53, 3415–3426 (Jun 2014).

[36] Moreno, D. & Taubin, G. Simple, accurate, and robust projector-camera calibration. In 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, 464–471 (2012). http://dx.doi.org/10.1109/3DIMPVT. 2012.77.

[37] Liu, Y. et al. An improved projector calibration method by phase mapping based on fringe projection profilometry. Sensors 23, 1142 (2023).

[38] Hu, C. et al. Phase error model and compensation method for reflectivity and distance discontinuities in fringe projection profilometry. Optics Express 31, 4405–4422 (2023).

[39] Li, H. et al. Fringe projection profilometry based on saturated fringe restoration in high dynamic range scenes. Sensors 23, 3133 (2023).

[40] Feng, S. et al. Calibration of fringe projection profilometry: A comparative review. http://dx. doi.org/10.1016/j.optlaseng.2021.106622 (8 2021).

[41] Kong, X. et al. Focus-guided structured-light profilometry for microscopic 3d reconstruction with extended depth-of-field. Optics and Lasers in Engineering 196, 109433 (2026).

[42] Hillen, M. et al. Enhanced checkerboard detection using gaussian processes. Mathematics 11, 4568 (2023).

[43] Zhang, S. & Huang, P. S. Novel method for structured light system calibration. Optical Engineering 45, 083601–083601 (2006).

[44] Catalucci, S. et al. Measurement of complex freeform additively manufactured parts by structured light and photogrammetry. Measurement 164, 108081 (2020).

[45] Khanam, R. & Hussain, M. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725 (2024).

[46] Wada, K. et al. Labelme: Image polygonal annotation with python (2016).

[47] Saha, D. et al. Deep-pose-tracker: a unified model for behavioural studies of caenorhabditis elegans. bioRxiv (2025).

[48] Sapkota, R. & Karkee, M. Ultralytics yolo evolution: An overview of yolo26, yolo11, yolov8 and yolov5 object detectors for computer vision and pattern recognition. https://arxiv.org/abs/ 2510.09653 (2026).

[49] Yu, M. et al. Yolo-mousepose: A novel framework and dataset for mouse pose estimation from a topdown view. IEEE Transactions on Instrumentation and Measurement (2025).

[50] Zhuo, S. & Sim, T. Defocus map estimation from a single image. Pattern Recognition 44, 1852–1858 (2011).

[51] Shi, J., Xu, L. & Jia, J. Just noticeable defocus blur detection and estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 657–665 (2015).

[52] Deng, M. et al. A multi-scale refinement corner detection algorithm based on shi-harris. Digital Signal Processing 161, 105137 (2025).