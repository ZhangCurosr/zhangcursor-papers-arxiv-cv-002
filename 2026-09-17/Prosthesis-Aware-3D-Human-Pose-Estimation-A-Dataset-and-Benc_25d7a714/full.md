# Prosthesis-Aware 3D Human Pose Estimation: A Dataset and Benchmark for RSP Users

Yilin Wen<sup>1</sup> , Kechuan Dong<sup>1</sup>, Fumiya Suginaka<sup>1</sup>, Ken Endo<sup>2</sup> , and Yusuke Sugano<sup>1∗</sup>

<sup>1</sup> The University of Tokyo, Tokyo, Japan <sup>2</sup> Sony Computer Science Laboratories, Tokyo, Japan {fylwen,kchdong,fumisugi,sugano}@iis.u-tokyo.ac.jp kene@csl.sony.co.jp

Abstract. Recovering 3D human body motion from video is important for applications such as rehabilitation assessment and sports performance evaluation. For prosthesis users, this requires capturing both natural body joints and the geometry of the prosthetic device, a challenge that existing methods are not designed to address. Model-based estimators rely on body models trained on non-amputee individuals and cannot represent prosthesis geometry, while model-free methods lack body kinematic priors and are unreliable under occlusion. This challenge is particularly prominent for users of running-specific prostheses (RSPs), where the RSP has a complex curved geometry and moves dynamically during exercise. To fill this gap, we collect RSP3D, the first 3D dataset of RSP users, covering essential daily-life and exercise actions from participants with varied amputation conditions, using a multi-camera marker-based motion capture setup. We formally define the task of prosthesis-aware 3D pose estimation, evaluate representative methods in a zero-shot setting, and confirm their individual limitations. We further propose a hybrid baseline combining model-based body joint estimation with model-free RSP shape recovery, establishing a starting point for future research. Our project page is available at https://ut-vision.github.io/RSP3D/

Keywords: 3D Human Pose Estimation · Prosthesis Reconstruction · Dataset and Benchmark · Running-Specific Prosthesis

## 1 Introduction

Accurately recovering 3D human body motion from videos is a key technology for understanding human activities. For prosthesis users, capturing the 3D positions of both natural body joints and prosthetic components is essential for applications such as rehabilitation assessment and sports performance evaluation. Such prosthesis-aware 3D body recovery can directly improve the quality of life for prosthesis users and contribute to building a more inclusive society. While professional athletes typically have access to coaches who can monitor their movement and provide feedback, for the majority of running-specific prostheses (RSP) users who train independently or in recreational settings, such expert guidance is not always available. Automated prosthesis-aware 3D analysis can therefore fill this gap, providing objective motion feedback that supports training and rehabilitation without requiring direct expert supervision. However, existing methods have not been designed to handle prosthesis users, and there is a clear lack of both data and methods for this purpose.

![](images/6c4cad2db2de9f5711d479725aa358a9c173f23a6d3401d7f42681d3d6a27ee0.jpg)  
Fig. 1: Samples of collected data (left) and challenges faced by existing works in handling pose estimation for running-specific prostheses (RSP) users (right).

This task poses a unique challenge: part of the human body is replaced by an object that does not follow human anatomical structure, and prostheses lack shared statistical geometric structure across diferent users. This breaks the fundamental assumption shared by existing approaches, and neither model-based nor model-free methods can solve it alone (see Fig. 1). Model-based 3D human pose estimators [45, 52] rely on predefined anatomical models [12, 16, 22] trained on large-scale datasets of non-amputee individuals, and by design cannot generalize to bodies outside this learned distribution. This means that while they can robustly estimate body joint positions even under occlusion, they cannot represent prosthesis geometry. Moreover, prostheses vary widely in shape, length, and attachment depending on the individual’s amputation condition and activity, making it dificult to define a common geometric prior. Conversely, modelfree approaches [3, 39, 40] can reconstruct arbitrary 3D shapes without relying on predefined templates, making them capable of recovering prosthesis geometry. However, without body-specific priors, they often fail under occlusion and cannot capture the correct kinematics of body joints. In other words, this task inherently requires integrating two distinct approaches: one that understands the human body and one that can handle arbitrary shapes.

In this study, we address this challenge by constructing RSP3D, the first dataset that captures 3D poses of prosthesis users, with a particular focus on individuals using running-specific prostheses (RSPs). Our dataset comprises essential actions for daily activities and specific drills for physical exercise, collected from diverse participants spanning a range of ages, amputation conditions, and RSP experience. We use an indoor environment equipped with 16 GoPro RGB cameras and a marker-based motion capture system, providing 3D annotations of both natural body joints and RSP shapes. With this dataset, we formally define the task of prosthesis-aware 3D human pose estimation, which involves recovering both natural body joints and keypoints representing the prosthesis shape in 3D space from video observations.

We then use the collected data for zero-shot evaluation, considering representative model-based and model-free methods. For model-based approaches, we evaluate SAM3D Body [45], MotionBERT [52], and the only existing amputationaware method AJAHR [4]. For model-free approaches, we evaluate SpatialTrackerV2 [39], which enables 3D point tracking and scene reconstruction from video. Our evaluation confirms their individual limitations discussed above. Based on these findings, we propose a hybrid baseline that combines the two approaches: we use model-based methods for robust body joint estimation and model-free methods for flexible prosthesis shape recovery, aligning their outputs through scale and position adjustment, therefore providing a reference for future research in prosthesis-aware 3D pose estimation.

Our contributions are summarized as follows:

– We introduce the task of prosthesis-aware 3D human pose estimation, which requires recovering both natural body joints and prosthesis geometry from video observations.

– We construct RSP3D, the first 3D dataset of RSP users, covering actions from daily activities to physical exercise, with 3D annotations for both body joints and RSP shapes, serving as a resource for zero-shot benchmarking.

– We evaluate existing model-based and model-free methods, revealing their individual limitations, and propose a hybrid baseline that integrates both approaches to establish a starting point for this new research direction.

## 2 Related Work

3D Human Pose Estimation 2D human pose estimation aims to detect body joint locations in image space. Mainstream solutions [10, 25, 32, 43] typically involve detecting the human bounding box and regressing the heatmap of a fixed set of body joints. However, these methods face a specific challenge for prosthesis users, as joint detectors trained on non-amputee individuals may fail to localize joints near prosthetic limbs. 2D-to-3D lifting approaches convert 2D joint detections into 3D space [23, 51, 52]. For prosthesis users, these methods inherit errors from 2D detection, and the lifting models themselves assume a complete set of body joints, making them sensitive to missing or incorrectly detected joints.

Direct 3D estimation methods recover body joints and mesh directly from images [14, 17–19, 21, 31, 45]. By training on large datasets that cover a variety of body poses and environments, these solutions learn consistent spatiotemporal relationships among diferent joints, enabling robust performance in diverse settings and accurate predictions even when joints are occluded. Recent works [5,38,41,42] have further extended this to reconstructing humans alongside the objects they interact with. However, these models cannot represent bodies outside their training distribution, making them unsuitable for prosthesis users.

Table 1: Comparison with existing datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Data Source</td><td>#</td><td>#</td><td colspan="3">Prosthesis Prosthesis Body</td></tr><tr><td>Participants Frames</td><td></td><td>Users</td><td>Ann.</td><td>Ann.</td></tr><tr><td>Human3.6M [16]</td><td>Real-world Collection</td><td>11</td><td>3.6M</td><td>X</td><td></td><td>3D</td></tr><tr><td>MPI-INF-3DHP [24]</td><td>Real-world Collection</td><td>8</td><td>1.3M</td><td>X</td><td></td><td>3D</td></tr><tr><td>3DPW [33]</td><td>Real-world Collection</td><td>5</td><td>51k</td><td>X</td><td></td><td>3D</td></tr><tr><td>LDPose [47]</td><td>Web Images/Videos</td><td></td><td>28k</td><td>√</td><td></td><td>2D</td></tr><tr><td>InclusiveVidPose [8]</td><td>Web Videos</td><td></td><td>327k</td><td>√</td><td></td><td>2D</td></tr><tr><td>ProGait [46]</td><td>Real-world Collection</td><td>4</td><td>412 clips</td><td>√</td><td>2D</td><td>2D</td></tr><tr><td>A3D [4]</td><td>Synthetic Data</td><td></td><td>1.0M</td><td>√</td><td></td><td>3D</td></tr><tr><td>Ours</td><td>Real-world Collection</td><td>6</td><td>5.6M</td><td>√</td><td>3D</td><td>3D</td></tr></table>

Across these approaches, the fundamental limitation lies in the assumption of a complete set of body joints. While 2D detectors struggle to localize joints near prosthetic limbs, lifting-based and direct 3D methods are further constrained by body models trained exclusively on non-amputee individuals. As a result, none of these methods can capture amputations or the geometry of RSPs.

Model-free 3D Reconstruction Model-free solutions recover 3D structures of arbitrary objects from image observations or 2D keypoints, without relying on predefined shape templates. Earlier research tackled this using non-rigid structure-from-motion techniques [1, 7, 20], which recover 3D structure from 2D point trajectories of deforming objects. 3D-LFM [6] proposes a unified framework for freeform 2D-to-3D lifting that captures object deformation while preserving point permutation equivariance, showing generalizability to novel objects. However, these methods rely heavily on the accurate 2D detection of keypoints, and often struggle with resolving depth ambiguity due to the sparse nature of inputs.

More recently, foundational backbones for scene depth and 3D pointwise features [26, 34, 37, 44] have enabled unified solutions for dynamic 3D/4D point correspondence, supporting simultaneous scene reconstruction and point tracking [2, 11, 35, 36, 39, 40, 48–50]. Without relying on predefined object templates, these methods can recover 3D shapes corresponding to the 2D detection and segmentation of body parts and RSPs, thus providing a feasible solution for our task. However, this flexibility complicates the incorporation of body kinematic priors, reducing the accuracy of natural body joint estimation, as these methods cannot exploit the kinematic relationships among joints.

Pose Estimation for Prosthesis Users Several recent works have addressed body pose estimation for prosthesis users. LDPose [47] and InclusiveVidPose [8] focus on limb deficiency-aware 2D pose estimation, where they collect data using web images and videos, and concentrate on detecting intact or residual body joints separately. ProGait [46] further considers the prosthesis shape in 2D space for gait analysis. This study involves collecting data from human subjects with transfemoral prosthetic legs and examines 2D human pose estimation and object segmentation to facilitate gait classification. AJAHR [4] extends this to 3D by predicting amputation-aware body meshes. This is achieved by first classifying the body part amputation and generating a mesh reflecting this amputation status. Furthermore, they propose a synthesis pipeline to generate data for training and evaluation.

![](images/d14c15c337f5eda62e5db7c46370941e311194ab6cd2a271baafecebf8efe54a.jpg)  
Fig. 2: Our recording setup.

Despite these advances, existing solutions still cannot capture both body joints and prosthesis geometry in 3D space. The lack of datasets tailored for this task remains a fundamental obstacle, limiting the development of dedicated methods (see Tab. 1).

## 3 Dataset

We collect RSP3D, a motion dataset from prosthesis users focusing specifically on those using RSPs. We capture and annotate both the 3D positions of natural body joints and the RSP shape, which has not been addressed in 3D in prior research. This study was approved by the Institutional Review Board under approval number E2025ALS275. Our data and code are accessible through our project page. Users must agree to the license and terms of use outlined on our project page and submit an access request form. Approved applicants will receive access to the videos and annotations.

## 3.1 Recording Setup

Camera Setup As shown in Fig. 2, we conduct our dataset collection in an indoor laboratory, where we record using a combination of an OptiTrack motion capture system and 16 GoPro video cameras. We place reflective markers on participants to capture their 3D positions. The OptiTrack system employs 12 Prime cameras for marker tracking and one Prime Color camera for video recording. The 16 GoPro cameras surround the recording area to capture video at 60 FPS from various angles.

![](images/d1f41f7f9163890a620f851b1b6cebf53d98bfc0aea8f13df8c27e8f591902da.jpg)  
Fig. 3: Actions that highlight diferences in motion patterns among participants with varying amputation conditions. We show more cases in the supplementary materials.

Time Synchronization and Calibration We follow the oficial procedure to synchronize and calibrate the OptiTrack cameras, including hardware-based time synchronization and waving a calibration wand throughout the recording space. To synchronize the OptiTrack system with the GoPro cameras, we employ a QR video method inspired by Ego-Exo4D [15]. At the start of each recording session, we display a QR video to all GoPro cameras and the Prime Color camera, where each frame encodes a timestamp at a resolution of ∼0.033 sec (i.e., 30 FPS). Each camera’s time code is synchronized by detecting this QR video. Through manual checks, we ensure synchronization accuracy within one frame at 60 FPS. We calibrate the intrinsic parameters of each GoPro camera and the Prime Color camera using a ChArUco board [13] via OpenCV, following the standard procedure of capturing the board from multiple viewpoints. Extrinsic calibration among these cameras is conducted using another ChArUco board with a known physical size, establishing the transformation between each camera’s coordinate system and the world coordinate system defined by the OptiTrack system.

## 3.2 Action

We design the recording sessions in collaboration with two experienced prosthetists and orthotists to ensure that the selected actions are both practically relevant and representative of the challenges faced by RSP users. To build an initial action list, we attended a training session for first-time RSP users led by the prosthetists, observing which movements were practiced and which proved dificult. We then refined the list through a follow-up meeting with the prosthetists, incorporating their feedback on which actions were unexpectedly important for daily life and which presented particular dificulty for RSP users.

![](images/9d13e533f30a928d77fc51803790996ac87ae4d9e38efd451406f0a592d5676e.jpg)  
Fig. 4: Total recording duration by action. We report on all 6 participants.

On one hand, in addition to basic activities like walking and jogging, we include actions crucial for daily life $( e . g .$ , body rotation in place, standing up from $t h e \ f t o o r )$ , as well as those that test proficiency in using the RSP while maintaining balance $( e . g . ,$ , stepping over hurdles, vertical jumps). On the other hand, we incorporate actions designed to highlight motion pattern diferences among participants with varying amputation conditions (see Fig. 3). For instance, noticeable diferences are observed in actions like walking up stairs and squatting between individuals with below-knee and above-knee amputations, primarily due to the absence of the knee joint. Similarly, users with left-foot versus right-foot amputations may demonstrate diferent strategies for standing up from the $f l o o r ,$ often shifting weight to the non-amputated side. Furthermore, for users with bilateral amputations, movements such as hip rotation introduce additional challenges in maintaining balance and body control. A detailed description of these actions, along with examples, is provided in the supplementary materials.

## 3.3 Annotation

As illustrated in Fig. 2, we attach reflective markers to the body parts and RSPs of participants. OptiTrack cameras capture the positions of these markers within their world coordinate system, allowing us to reconstruct the 3D positions of natural body joints and the RSP surfaces.

Body Joints We use the 17-joint definition from the Human3.6M dataset [16] to annotate the natural body joints. For the upper body, including the hips, we use the upper body template provided by the OptiTrack system, following its guidelines for marker placement and joint position regression. For the available knees and ankles, we place two markers symmetrically on the left and right sides of each available knee or ankle, using their midpoint as the corresponding annotation.

RSP Surface To capture the RSP geometry, we densely and symmetrically place markers on both sides of the RSP along its curve (Fig. 2). Let the marker set be $\bar { \mathcal { M } } = \bar { \mathcal { M } } ^ { 1 } \bigcup \bar { \mathcal { M } } ^ { 2 }$ , where $\bar { \mathcal { M } } ^ { 1 }$ and $\bar { \mathcal { M } } ^ { 2 }$ collect markers on each side of the RSP. Each M<sup>¯</sup> <sup>i</sup> contains n sequential markers $( \bar { m } _ { 1 } ^ { i } , \dots , \bar { m } _ { n } ^ { i } )$ that are ordered from the near-body side to the near-floor side. As shown in Fig. 2, markers are placed at intervals of approximately 5 cm, allowing for the capture of the RSP curvature.

Table 2: Information of participants.
<table><tr><td>ID Age Group Gender</td><td></td><td>Amputation Site</td><td>RSP Experience</td></tr><tr><td>P1 30s</td><td>M</td><td>Below-knee (Right)</td><td>&gt; 5 yrs</td></tr><tr><td>P2 40s</td><td>M</td><td>Above-knee (Left)</td><td>&lt; 1 yr</td></tr><tr><td>P3 20s</td><td>F</td><td>Below-knee (Left)</td><td>&gt; 5 yrs</td></tr><tr><td>P4 60s</td><td>M</td><td>Above-knee (Right)</td><td>&gt; 5 yrs</td></tr><tr><td>P5 10s</td><td>F</td><td>Below-knee (Bilateral)</td><td>&gt; 5 yrs</td></tr><tr><td>P6 10s</td><td>M</td><td>Below-knee (Bilateral)</td><td>&lt; 1 yr</td></tr></table>

We then construct the RSP surface $\bar { \bar { S } }$ using a ruled surface approach, which efectively captures the RSP curve shape and surface width. Specifically, $\bar { \boldsymbol { S } }$ is defined by a family of lines $\bar { \boldsymbol { r } } _ { u } ( v )$ that span between $\bar { \mathcal { M } } ^ { 1 }$ and $\bar { \mathcal { M } } ^ { 2 }$ , connecting markers $\bar { m } _ { u } ^ { 1 } , \bar { m } _ { u } ^ { 2 }$ with the same index from each side. $\bar { \boldsymbol { S } }$ is then expressed as

$$
\bar { \mathcal { S } } = \bigcup _ { u = 1 } ^ { n } \{ \bar { r } _ { u } ( v ) = ( 1 - v ) \bar { m } _ { u } ^ { 1 } + v \bar { m } _ { u } ^ { 2 } \ | \ v \in [ 0 , 1 ] \} .\tag{1}
$$

## 3.4 Statistics

We collect data from 4 male and 2 female RSP users, ensuring diversity in age, amputation conditions, and years of RSP experience. Since RSP users span a wide range of training backgrounds, from beginners who have recently adopted the device to experienced recreational runners, capturing this diversity is important for developing methods that generalize beyond elite users who have access to professional coaching. We provide the demographic statistics in Tab. $^ { 2 , }$ and a breakdown of recording lengths by action in Fig. 4. Additionally, we compare RSP3D with existing related datasets in Tab. 1. Overall, our data collection offers a comprehensive resource for motion analysis of RSP users across diferent body conditions, including substantial recording lengths and a variety of camera viewpoints for diverse participants and actions.

## 4 Methods and Evaluation Protocol

## 4.1 Task Definition

As illustrated in Fig. 1, our task input is a monocular video $\mathcal { V } = \{ \mathbf { I } _ { i } \in \mathbb { R } ^ { H \times W \times 3 } | i =$ $1 , \ldots , T \}$ featuring an RSP user, where T represents the number of frames, and H, W denote the image height and width. The task objective is to recover the 3D positions of both the available natural body joints $\mathbf { J } \in \mathbb { R } ^ { N _ { 1 } \times 3 }$ and points describing the RSP shape $\mathbf { P } \in \mathbb { R } ^ { N _ { 2 } \times 3 }$ for each image $\mathbf { I } \in \mathcal { V } .$ . By capturing both natural body parts and the unique geometry of the RSP, our task aims to provide a foundation for comprehensive motion analysis, facilitating applications such as rehabilitation assessment and prosthesis design optimization.

![](images/18d8104f0ab4cffe862f50c52397acfa1dc737b2e5fd7c10854ea4ac259d48cd.jpg)  
Fig. 5: Proposed hybrid baseline solution, which solves scale ambiguity to align the model-based and model-free solutions.

## 4.2 Evaluation Metrics

Natural Body Joints We follow established practices in human body pose estimation research to compute the Mean Per Joint Position Error (MPJPE), which considers all available natural body joints J<sup>¯</sup> and compares the ground truth positions with the estimated J in a root-aligned space. We further report the Procrustes-aligned MPJPE (MPJPE-PA), which eliminates global diferences in scale, translation, and rotation, resulting in a more pose-focused error metric. Both metrics are reported in mm.

RSP Shape We evaluate the RSP by first examining it within the body space. This is essential for assessing the accurate interaction between the RSP and the body under the overall body configuration, which is crucial for prosthesisaware pose estimation. Subsequently, we isolate the body pose estimation and focus on the local geometry of the RSP. We evenly sample the ground truth RSP surface S<sup>¯</sup> to derive a point cloud P<sup>¯</sup> . Inspired by research in human object reconstruction [41], we report bidirectional Chamfer Distance and its F-score between P<sup>¯</sup> and the estimated P.

Specifically, we first translate both the estimated and ground truth RSPs by aligning them using the body pelvis positions. In this context, we report the bidirectional Chamfer Distance (CD) and its F-score at thresholds of 100 mm and 50 mm (F@100, F@50). Next, to focus on the local geometry, we align the average positions of P, P<sup>¯</sup> . We then re-evaluate the Chamfer Distance (CD-C) and report the F-score at thresholds of 50 mm, 30 mm (F-C@50, F-C@30).

## 4.3 Baseline Methods

We evaluate all methods in a zero-shot setting, where pre-trained models are applied directly to our task without fine-tuning on our dataset. For each recorded action segment, we select one of the 16 GoPro cameras that ofers moderate occlusion as the starting point for prosthesis-aware pose estimation. We examine methods in two groups: model-based 3D human pose estimators and model-free reconstruction and tracking solutions. We then establish the baseline for our task by integrating these two approaches.

Model-based The model-based 3D human pose estimator aims to recover the 3D positions of natural body joints $\mathbf { J } _ { 1 : T }$ from the video observations V. Existing methods output joints covering the standard body structure [9, 27, 31, 45, 52], while recent research enables amputation-aware body estimation [4].

We evaluate three representative model-based approaches. The first is SAM3D Body [45], a state-of-the-art method that regresses body joints and mesh directly from input images. The second is MotionBERT [52], a lifting-based approach that converts 2D joint detections into 3D space. We assess MotionBERT using both 2D detections from AlphaPose [10] and the 2D ground truth, with the latter providing an oracle analysis. The third approach, AJAHR [4], difers from SAM3D Body and MotionBERT, which assume a standard set of body joints. This is a recent prosthesis-aware solution that identifies amputations and outputs only the available body joints from images.

Model-free To recover both natural body joints $\mathbf { J } _ { 1 : T }$ and RSP points $\mathbf { P } _ { 1 : T }$ we apply model-free solutions that can handle arbitrary shapes. Utilizing 2D joint detection and prosthesis-aware mask segmentation, these methods output P corresponding to the masked RSP points $ { \mathbf { p } } \in \mathbb { R } ^ { N _ { 2 } \times 2 }$ and assign 3D coordinates to J corresponding to detected 2D body joints $ { \mathbf { j } } \in \mathbb { R } ^ { N _ { 1 } \times 2 }$

We refer to SpatialTrackerV2 [39] as the representative work, which facilitates eficient simultaneous 3D point tracking and scene reconstruction. This approach ofers two solutions for our task: (1) STv2-Pointmap, which reads out surface points $\mathbf { J } _ { 1 : T } , \mathbf { P } _ { 1 : T }$ from the reconstructed scene point map by referencing the frame-wise 2D detections $\mathbf { j } _ { 1 : T } , \mathbf { p } _ { 1 : T } ;$ (2) STv2-Tracking, which tracks the first frame detection $\mathbf { j } _ { 1 } , \mathbf { p } _ { 1 }$ in 3D space to obtain $\mathbf { J } _ { 1 : T } , \mathbf { P } _ { 1 : T }$ for the entire video. We examine both solutions, employing AlphaPose [10] and SAM2 [28–30] to obtain $\mathbf { j } _ { 1 : T } , \mathbf { p } _ { 1 : T }$ , respectively. We also report oracle results using the 2D ground truth.

For implementation details, we obtain the detected RSP mask p by using SAM2 to track from the initial 2D ground truth bounding box of the RSP in the first frame, and we restart the tracking process every 10 sec. For SpatialTrackerV2, we further divide the video into short chunks by setting $T = 6 0$ , which spans 1 sec under the original 60 FPS recording. Furthermore, when evaluating model-free solutions, to address the inherent scale ambiguity in monocular observations, we compare the bounding box size in the XY dimensions between the first frame $\mathbf { J } _ { 1 }$ and its ground truth counterpart $\bar { \mathbf { J } } _ { 1 }$ . This comparison allows us to recover the scale $s ,$ converting the outputs into mm measurements for evaluation.

Hybrid Baseline As illustrated in Fig. 5, we establish the baseline for our proposed task by combining model-based and model-free solutions. Let the modelbased outputs be denoted as $\mathbf { J } ^ { \prime } { } _ { ; }$ and the model-free outputs as $\mathbf { J } ^ { * }$ and $\mathbf { P } ^ { * }$ . Given the groundtruth amputation site label, our alignment derives J directly from $\mathbf { J } ^ { \prime }$ by removing the hallucinated joints, while $\mathbf { P }$ is obtained by resolving the scale ambiguity in $\mathbf { P } ^ { * }$ and aligning it with the position of $\mathbf { J } ^ { \prime }$

$$
{ \bf J } = { \bf J } ^ { \prime } , \quad { \bf P } = s ( { \bf P } ^ { * } - r ^ { * } ) + r ^ { \prime } ,\tag{2}
$$

where $s \in \mathbb { R }$ is the scale factor and $\boldsymbol { r } ^ { * } , \boldsymbol { r } ^ { \prime }$ are the root joint of $\mathbf { J } ^ { * } , \mathbf { J } ^ { \prime }$

We obtain s through a three-step process to stabilize the computation. We begin by comparing the bounding box sizes in the XY-dimensions between $\mathbf { J } ^ { \prime }$ and $\mathbf { J } ^ { * }$ . This comparison yields the scale factor ${ \tilde { s } } \in \mathbb { R }$ , which transforms $\mathbf { J } ^ { * }$ to a scale comparable with $\mathbf { J } ^ { \prime }$ . Next, we align the root $( i . e . , \mathrm { p e l v i s } )$ of the model-free outputs to that of the model-based $\mathbf { J } ^ { \prime }$ , addressing the depth discrepancy between $\mathbf { J } ^ { \prime }$ and ${ \tilde { s } } \mathbf { J } ^ { * }$ caused by inaccurate depth estimation. The transformation is given by:

$$
\hat { { \bf J } } ^ { * } = \tilde { s } ( { \bf J } ^ { * } - r ^ { * } ) + r ^ { \prime } , \quad \hat { { \bf P } } ^ { * } = \tilde { s } ( { \bf P } ^ { * } - r ^ { * } ) + r ^ { \prime } .\tag{3}
$$

Finally, we refine the scaling factor to ensure consistent 2D projection between $\mathbf { P } ^ { * }$ and $\mathbf { P } \colon$

$$
\hat { s } = \arg \operatorname* { m i n } _ { \hat { s } > 0 } \left\| \pi \left( \hat { s } \left( \left[ \hat { \mathbf { J } } ^ { * } \right] - r ^ { \prime } \right) + r ^ { \prime } \right) - \pi \left( \left[ \mathbf { J } ^ { * } \right] \right) \right\| ,\tag{4}
$$

where $\pi ( . )$ denotes the perspective projection using the recording intrinsics. We therefore obtain $s = \hat { s } \tilde { s }$

We use SAM3D Body [45] and STv2-Pointmap [39] as baseline methods due to their superior performance. Similar to the evaluation of model-free methods, we examine performance using either 2D detection data or 2D GT counterparts when implementing STv2-Pointmap.

## 5 Results

Tab. 3 and Fig. 6 summarize the quantitative and qualitative results of all evaluated methods on RSP3D. We begin by discussing the specific limitations of both model-based and model-free approaches. Following this, we analyze how our hybrid baseline efectively addresses these limitations and further examine performance variations across diferent amputation conditions.

## 5.1 Model-based and Model-free Solutions

Model-based solutions are fundamentally unable to capture the 3D geometry of prostheses, as their outputs are limited to a predefined set of body joints based on human anatomy, as exemplified by the qualitative results of SAM3D Body in Figs. 6 and 7. Although they leverage learned spatial relationships among joints to estimate the positions of missing body joints, this prediction often fails to align with the observed prosthesis due to the domain gap caused by the appearance diference between prosthetic and natural limbs.

Table 3: Quantitative results on the collected dataset. For model-free solutions, outputs are aligned with the GT to resolve scale ambiguity.
<table><tr><td></td><td>Method</td><td>|MPJPE ↓ MPJPE-PA ↓</td><td>CD ↓</td><td></td><td>F@100 ↑ F@50 ↑ CD-C ↓ F-C@50 ↑ F-C@30 ↑</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Model-Based</td><td>AJAHR [4]</td><td>125.06</td><td>90.85</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAM3D Body [45]</td><td>78.35</td><td>50.89</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MotionBERT [52] (w/ 2D Det.)</td><td>115.06</td><td>64.94</td><td>2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MotionBERT [52] (w 2D GT)</td><td>112.41</td><td>60.03</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Model-Free (w/ 2D Detection)</td><td>STv2-Tracking [39]</td><td>174.84</td><td>125.45</td><td>299.59</td><td>40.31</td><td>15.32</td><td>122.42</td><td>48.32</td><td>22.49</td></tr><tr><td>Hybrid Alignment</td><td>STv2-Pointmap [39] SAM3D Body [45]</td><td>159.64</td><td>113.87</td><td>277.15</td><td>45.44</td><td>18.82</td><td>115.00</td><td>50.60</td><td>24.33</td></tr><tr><td>(w/ 2D Detection)</td><td>+ STv2-Pointmap [39]</td><td>78.35</td><td>50.89</td><td>288.44</td><td>45.61</td><td>19.92</td><td>119.60</td><td>50.45</td><td>25.15</td></tr><tr><td rowspan="2">Model-Free (w/ 2D GT)</td><td>STv2-Tracking [39]</td><td>141.14</td><td>109.71</td><td>273.98</td><td>46.17</td><td>19.30</td><td>124.67</td><td>50.99</td><td>24.16</td></tr><tr><td>STv2-Pointmap [39]</td><td>119.59</td><td>93.94</td><td>238.71</td><td>53.81</td><td>25.74</td><td>115.63</td><td>54.60</td><td>26.72</td></tr><tr><td>Hybrid Alignment (w/ 2D GT)</td><td>SAM3D Body [45] + STv2-Pointmap [39]</td><td>78.35</td><td>50.89</td><td>251.15</td><td>50.72</td><td>22.41</td><td>117.23</td><td>54.94</td><td>27.52</td></tr></table>

![](images/8f525583d79db234e1b1804225c71ad34a662ab22f27df206f09c05f102b04da.jpg)  
Fig. 6: Qualitative comparisons with SAM3D Body, SpatialTrackerV2-Pointmap and the proposed hybrid alignment baseline.

Conversely, model-free solutions are not limited to predefined object categories and can thus handle prosthesis estimation. However, they have limited ability to use kinematic and geometric priors of the human body. As demonstrated in Tab. 3 and illustrated in Fig. 6, this limitation hinders reliable estimation under occlusion, leading to worse accuracy for natural body joints.

Among model-based approaches, SAM3D Body outperforms MotionBERT and AJAHR in estimating body joints. We attribute this to SAM3D Body’s large and diverse training data, along with its ability to use richer appearance information from input images. As shown in Fig. 7, while AJAHR accounts for amputations in its output, it struggles to correctly classify amputation status under occlusion, and such misclassifications of natural limbs as amputated penalize its accuracy.

![](images/1e60b7fc4ea5193a12f7b17304a911b9a81f5c4ceb0c8afe6c90e30307f61ed8.jpg)  
Fig. 7: Failure cases of SAM3D Body, AJAHR, and STv2-Tracking. SAM3D Body hallucinates amputated-side meshes that serve fundamentally diferent biomechanical functions from RSPs and have drastically diferent geometries. AJAHR shows reduced accuracy due to dificulties in classifying amputation under occlusion. STv2-Tracking struggles with the thin geometry of RSPs and textureless clothing.

Table 4: Comparison among users with diferent amputation types. Both methods leverage 2D detection for prediction.
<table><tr><td>Amputation Type|</td><td>Method</td><td>|MPJPE ↓ MPJPE-PA ↓|</td><td>CD↓</td><td>F@100 ↑ F@50 ↑|CD-C ↓ F-C@50 ↑ F-C@30 ↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Bilateral</td><td>STv2-Pointmap Hybrid Alignment</td><td>155.60 91.91</td><td>98.89 51.32</td><td>|294.21 342.22</td><td>45.88 37.41</td><td>19.56 13.80</td><td>106.80 113.32</td><td>54.60 54.25</td><td>25.97 27.35</td></tr><tr><td>STv2-Pointmap</td><td>162.02</td><td>122.71</td><td>267.07</td><td>45.19</td><td>18.38</td><td>119.84</td><td>48.25</td><td>23.36</td></tr><tr><td rowspan="2">Unilateral</td><td>Hybrid Alignment</td><td>70.35</td><td>50.63</td><td>256.64</td><td>50.45</td><td>23.53</td><td>123.31</td><td>48.21</td><td>23.85</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

For SpatialTrackerV2, retrieving outputs from the reconstructed scene point map yields better results than those obtained through tracking. As exemplified in Fig. 7, frequent tracking loss is observed, primarily due to the lack of texture on clothing and the thin geometry of the RSP. This problem is more severe for model-free methods because they cannot incorporate prior knowledge of RSP shape or body kinematics. Furthermore, comparing results with 2D detection inputs against those with 2D ground truth reveals that point selection based on 2D detections is a significant bottleneck for model-free solutions: even with a fixed 3D scene reconstruction, incorrect 2D detections cause wrong points to be sampled from the point map, directly hurting accuracy. This suggests that improving 2D detection and segmentation for prosthesis users is a promising direction for future work.

## 5.2 Hybrid Baseline

The two limitations identified above, namely the inability of model-based methods to capture prosthesis geometry and the lack of body priors in model-free methods, directly motivate us to establish the baseline with a hybrid approach. Results in Tab. 3 confirm that combining the two components addresses both limitations: the model-based component provides accurate body joint estimation, while the model-free component recovers the RSP geometry. Qualitative results in Fig. 6 also show that our hybrid baseline achieves robust estimation of natural body joints, even when joints are absent or occluded. Additionally, it captures the 3D geometry and pose of the RSP.

As we directly adopt the estimated body joints from the model-based component, we further discuss by focusing on the performance of RSP estimation to evaluate the efectiveness of scale alignment. Our results demonstrate overall comparable performance to model-free solutions that resolve scale ambiguity by referring to the 3D ground truth, which verifies the efectiveness of our proposed scale alignment. However, we find that accurate estimation in the root-aligned space remains challenging, particularly reflected by the bidirectional Chamfer Distance. We attribute this to the sensitivity of alignment to inaccurate depth estimation, highlighting that achieving global alignment between body parts and RSP is a key challenge for future work.

Moreover, as reported in Tab. 4, participants with bilateral amputations generally show higher estimation errors compared to those with unilateral amputations, largely due to the increased dificulty in alignment when both legs are replaced by prostheses. On one hand, the model-based component has fewer intact joints available as anchor points for below-knee scale alignment, making it less stable in capturing the relative positions between both RSPs and the body. On the other hand, higher errors in each component for participants with bilateral amputations further complicate accurate scale recovery. We further discuss per-participant results in the supplementary materials.

## 6 Conclusion

Recovering 3D human body motion from video has important applications in rehabilitation assessment and sports performance evaluation, yet existing methods have not been designed for prosthesis users. In this work, we introduce the task of prosthesis-aware 3D human pose estimation and collect RSP3D, the first 3D dataset of RSP users, covering a range of daily-life and exercise actions from participants with diverse amputation conditions. Our evaluation confirms that model-based methods cannot represent RSP geometry, while model-free methods lack body kinematic priors and are unreliable under occlusion. Our hybrid baseline combines the strengths of both approaches and establishes a starting point for future research on this task.

Limitation and Future Work The current dataset scale is insuficient for reliable training. Dataset expansion and extension to marker-less, in-the-wild settings remain important future directions. The zero-shot hybrid baseline serves only as a reference, and more advanced solutions could be explored, such as leveraging synthetic data or incorporating biomechanical constraints and RSP consistency into an optimization framework. Amputation site labels are currently assigned manually; integrating the temporal cue for automatic prediction remains an open direction. Finally, while the unsaturated baselines demonstrate task dificulty, a more comprehensive evaluation of RSP shapes and poses and stricter error thresholds are left for future work.

Acknowledgment This research is supported by JSPS KAKENHI Grant Number JP25K03134, Toyota Foundation Grant Number D24-ST-0030, JST ASPIRE

Grant Number JPMJAP2303, and The Telecommunications Advancement Foundation. The authors are also grateful to Atsuro Okino (OSPO), Motohiko Takahashi (Step4ward), and Hideto Naito (Xiborg) for helpful discussions and support.

## A Overview

In this supplementary material, we present detailed action descriptions and perparticipant baseline results. Additionally, we include a video titled supp.mp4, which is compatible with most media players. This video demonstrates the recorded actions, exemplifies the various motion patterns among participants, and ofers qualitative comparisons, thereby supporting the discussion in our main text.

## B Dataset Actions

We demonstrate the detailed actions we ask participants to perform. To ensure safety, we made slight adjustments to the treadmill speed and the number of repetitions based on the participant’s stamina and ability during the recording. We encourage participants to perform the actions to the best of their ability, even if they find it challenging to execute them in the standard manner. The following sections describe the actions, with demos available in the supplementary video.

A1: Treadmill Walking Begin walking at a speed of 1.0 km/h, and increase the speed by 1 km/h every 15 seconds, up to a maximum of 4 km/h or the participant’s maximum comfortable speed. Maintain this speed for 30 seconds. Then, decrease the speed by 1 km/h every 15 seconds until returning to 1.0 km/h.

A2: Treadmill Jogging Start jogging at a speed of 4.0 km/h, and increase the speed by 1 km/h every 15 seconds, up to a maximum of 7 km/h or the participant’s maximum comfortable speed. Maintain this speed for 30 seconds. Then, decrease the speed by 1 km/h every 15 seconds until returning to 4.0 km/h.

A3: Incline Walking Maintain a fixed speed while setting the incline to 5% or 10% and walk for 30 seconds.

B1: Sit-to-Stand (Stairs) Sit on the third step of the stairs and stand up.

B2: Stair Ascent/Descent Go up and down the stairs.

C1: Stepping Over Low Hurdles Continuously step over mini hurdles, performing the exercise while facing both forward and sideways.

C2: Stepping Over High Hurdles Step over hurdles approximately at knee height. Perform the exercise by facing forward and leading with either the right or left foot. Additionally, perform round trips while facing sideways.

D1: Free Walking Walk along a straight line or circle.

D2: Free Jogging Jog lightly along a straight line or circle.

D3: Hip Rotation Rotate the hips while standing. Perform both clockwise and counterclockwise.

D4: Sit-to-Stand (Chair) Sit on the chair and then stand up.

D5: Side Steps Move sideways while keeping the body facing forward, then return to the original position.

D6: Squats From a standing position, bend the knees and lower the hips while pulling the buttocks backward, then stand up.

D7: Single-leg T-Balance Stand on the non-amputated leg and lean the upper body forward. Extend the upper limbs sideways to form a T-shape with the body and hold the position for 3 seconds.

D8: Rotation in Place Pivot on one foot and rotate in place. Use either the right or left foot as the axis, performing rotations both clockwise and counterclockwise.

D9: Sit-to-Stand (Floor) Sit down on the floor and then stand up.

E1: High Knees Perform a rapid jogging motion in place, lifting each knee to hip height with each step.

E2: Crouching Start From a standing position, lower the posture, place both hands on the ground, place one knee up, and shift the weight forward, preparing for the start signal, then return to a standing position.

E3: Vertical Jump From a stationary position, jump straight up as high as possible, using maximum efort for each jump. Avoid taking a running start and focus on explosive power.

E4: Straight Leg Jump (Scissors) With knees extended, jump by alternately extending each foot forward.

E5: Toe Jump (Pogo) Jump lightly in place with both feet together and knees slightly bent, maintaining a continuous bouncing motion.

![](images/122fa549ece3396b07ec8426b27581fd482471ecf11644523e5304e6043c441e.jpg)  
Fig. 8: Sample image of each participant.

Table 5: Per-participant results for the proposed Hybrid Alignment (w/ 2D Detection) baseline.
<table><tr><td>ID</td><td>Amputation Site</td><td>MPJPE ↓ MPJPE-PA ↓</td><td></td><td>CD F@100</td><td>↑F@50 ↑</td><td>CD-C</td><td></td><td>↓F-C@50 ↑F-C@30 ↑</td></tr><tr><td>P1</td><td>Below-knee (Right)</td><td>61.91</td><td>49.71</td><td>255.20 46.37</td><td>21.43</td><td>131.22</td><td>44.47</td><td>22.05</td></tr><tr><td>P2</td><td>Above-knee (Left)</td><td>60.30</td><td>42.74</td><td>283.87 46.18</td><td>19.18</td><td>114.23</td><td>52.62</td><td>25.64</td></tr><tr><td>P3</td><td>Below-knee (Left)</td><td>83.27</td><td>52.16</td><td>247.49 51.64</td><td>24.07</td><td>132.04</td><td>44.66</td><td>23.10</td></tr><tr><td>P4</td><td>Above-knee (Right)</td><td>73.30</td><td>55.92</td><td>248.96 57.52</td><td>28.89</td><td>109.44</td><td>54.05</td><td>25.74</td></tr><tr><td></td><td>P5 Below-knee (Bilateral)</td><td>102.20</td><td>55.07</td><td>319.29 37.77</td><td>14.47</td><td>115.57</td><td>52.12</td><td>24.76</td></tr><tr><td></td><td>P6 Below-knee (Bilateral)</td><td>83.34</td><td>48.19</td><td>361.34 37.11</td><td>13.25</td><td>111.45</td><td>56.03</td><td>29.51</td></tr></table>

## C Result for Each Participant

We report the per-participant results for the proposed hybrid alignment in Tab. 5, where 2D detection is referenced during the inference stage. Two observations emerge from this per-participant analysis:

First, compared to participants with unilateral amputation, those with bilateral amputation (i.e., P5, P6) exhibit higher joint error in terms of MPJPE. This increased error may be attributed to the fact that both lower limbs are replaced by running-specific prostheses (RSPs), which increase the dificulty for model-based methods to generalize well. As a result, the estimated positions of natural body joints become less accurate. The imprecise estimation of natural body joints, coupled with the reduced number of intact joints available for belowknee scale alignment, makes the subsequent hybrid alignment less reliable. This leads to inaccuracies when evaluating prosthetic blades in the body space (i.e., CD and its F-score), which is consistent with the discussion in the main text.

Second, when evaluating the local geometry of the RSP estimation by aligning the average positions of the estimation with the corresponding ground truth, we find significantly worse results for P1 and P3, in terms of CD-C and its F-score. We attribute this to the fact that P1 and P3 are elite athletes using RSPs with more pronounced curvature (see Fig. 8), which complicates the ability of modelfree methods to capture the prior knowledge of RSP geometry for accurate depth estimation.

## References

1. Bregler, C., Hertzmann, A., Biermann, H.: Recovering non-rigid 3d shape from image streams. In: Proceedings IEEE Conference on Computer Vision and Pattern Recognition. CVPR 2000 (Cat. No. PR00662). vol. 2, pp. 690–696. IEEE (2000)

2. Chen, X., Chen, Y., Xiu, Y., Geiger, A., Chen, A.: Easi3r: Estimating disentangled motion from dust3r without training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9158–9168 (2025)

3. Chen, X., Chu, F.J., Gleize, P., Liang, K.J., Sax, A., Tang, H., Wang, W., Guo, M., Hardin, T., Li, X., et al.: Sam 3d: 3dfy anything in images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7220–7232 (2026)

4. Cho, H., Choi, G., Choi, J.: Ajahr: Amputated joint aware 3d human mesh recovery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7925–7935 (2025)

5. Cseke, A., Tripathi, S., Dwivedi, S.K., Lakshmipathy, A.S., Chatterjee, A., Black, M.J., Tzionas, D.: Pico: Reconstructing 3d people in contact with objects. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1783– 1794 (2025)

6. Dabhi, M., Jeni, L.A., Lucey, S.: 3d-lfm: Lifting foundation model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10466–10475 (2024)

7. Dai, Y., Li, H., He, M.: A simple prior-free method for non-rigid structure-frommotion factorization. International Journal of Computer Vision 107(2), 101–122 (2014)

8. Du, H., Ying, J., Wang, S., Li, X., Zhang, K., Yu, X.: Inclusivevidpose: Bridging the pose estimation gap for individuals with limb deficiencies in video-based motion. In: International Conference on Learning Representations (ICLR) (2026)

9. Dwivedi, S.K., Sun, Y., Patel, P., Feng, Y., Black, M.J.: Tokenhmr: Advancing human mesh recovery with a tokenized pose representation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1323– 1333 (2024)

10. Fang, H.S., Li, J., Tang, H., Xu, C., Zhu, H., Xiu, Y., Li, Y.L., Lu, C.: Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time. IEEE transactions on pattern analysis and machine intelligence 45(6), 7157–7173 (2022)

11. Feng, H., Zhang, J., Wang, Q., Ye, Y., Yu, P., Black, M.J., Darrell, T., Kanazawa, A.: St4rtrack: Simultaneous 4d reconstruction and tracking in the world. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8503–8513 (2025)

12. Ferguson, A., Osman, A.A.A., Bescos, B., Stoll, C., Twigg, C., Lassner, C., Otte, D., Vignola, E., Prada, F., Bogo, F., Santesteban, I., Romero, J., Zarate, J., Lee, J., Park, J., Yang, J., Doublestein, J., Venkateshan, K., Kitani, K., Kavan, L., Farra, M.D., Hu, M., Ciofi, M., Fabris, M., Ranieri, M., Modarres, M., Kadlecek, P., Khirodkar, R., Abdrashitov, R., Prévost, R., Rajbhandari, R., Mallet, R., Pearsall, R., Kao, S., Kumar, S., Parrish, S., Yu, S.I., Saito, S., Shiratori, T., Wang, T.L., Tung, T., Xu, Y., Dong, Y., Chen, Y., Xu, Y., Ye, Y., Jiang, Z.: Mhr: Momentum human rig. arXiv preprint arXiv:2511.15586 (2025)

13. Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F.J., Marín-Jiménez, M.J.: Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition 47(6), 2280–2292 (2014)

14. Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., Malik, J.: Humans in 4d: Reconstructing and tracking humans with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14783–14794 (2023)

15. Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., et al.: Ego-exo4d: Understanding

skilled human activity from first-and third-person perspectives. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19383–19400 (2024)

16. Ionescu, C., Papava, D., Olaru, V., Sminchisescu, C.: Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence 36(7), 1325–1339 (2013)

17. Kanazawa, A., Black, M.J., Jacobs, D.W., Malik, J.: End-to-end recovery of human shape and pose. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7122–7131 (2018)

18. Kocabas, M., Athanasiou, N., Black, M.J.: Vibe: Video inference for human body pose and shape estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5253–5263 (2020)

19. Kolotouros, N., Pavlakos, G., Black, M.J., Daniilidis, K.: Learning to reconstruct 3d human pose and shape via model-fitting in the loop. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2252–2261 (2019)

20. Kumar, S.: Non-rigid structure from motion: Prior-free factorization method revisited. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 51–60 (2020)

21. Li, J., Cao, J., Zhang, H., Rempe, D., Kautz, J., Iqbal, U., Yuan, Y.: Genmo: A generalist model for human motion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11766–11776 (2025)

22. Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia) 34(6), 248:1–248:16 (Oct 2015)

23. Martinez, J., Hossain, R., Romero, J., Little, J.J.: A simple yet efective baseline for 3d human pose estimation. In: Proceedings of the IEEE international conference on computer vision. pp. 2640–2649 (2017)

24. Mehta, D., Rhodin, H., Casas, D., Fua, P., Sotnychenko, O., Xu, W., Theobalt, C.: Monocular 3d human pose estimation in the wild using improved cnn supervision. In: 2017 international conference on 3D vision (3DV). pp. 506–516. IEEE (2017)

25. Newell, A., Yang, K., Deng, J.: Stacked hourglass networks for human pose estimation. In: European conference on computer vision. pp. 483–499. Springer (2016)

26. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal (2024)

27. Patel, P., Black, M.J.: Camerahmr: Aligning people with perspective. In: 2025 International Conference on 3D Vision (3DV). pp. 1562–1571. IEEE (2025)

28. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. In: International Conference on Learning Representations. vol. 2025, pp. 28085–28128 (2025)

29. Ren, T., Jiang, Q., Liu, S., Zeng, Z., Liu, W., Gao, H., Huang, H., Ma, Z., Jiang, X., Chen, Y., Xiong, Y., Zhang, H., Li, F., Tang, P., Yu, K., Zhang, L.: Grounding dino 1.5: Advance the "edge" of open-set object detection. arXiv preprint arXiv:2405.10300 (2024)

30. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Zhang, L.: Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159 (2024)

31. Shin, S., Kim, J., Halilaj, E., Black, M.J.: Wham: Reconstructing world-grounded humans with accurate 3d motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2070–2080 (2024)

32. Sun, K., Xiao, B., Liu, D., Wang, J.: Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5693–5703 (2019)

33. Von Marcard, T., Henschel, R., Black, M.J., Rosenhahn, B., Pons-Moll, G.: Recovering accurate 3d human pose in the wild using imus and a moving camera. In: Proceedings of the European conference on computer vision (ECCV). pp. 601–617 (2018)

34. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)

35. Wang, Q., Ye, V., Gao, H., Zeng, W., Austin, J., Li, Z., Kanazawa, A.: Shape of motion: 4d reconstruction from a single video. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9660–9672 (2025)

36. Wang, S., Jiang, Z., Yang, X., Wang, X.: C4d: 4d made from 3d through dual correspondences. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7570–7580 (2025)

37. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20697–20709 (2024)

38. Wen, B., Huang, D., Zhang, Z., Zhou, J., Deng, J., Gong, J., Chen, Y., Ma, L., Li, Y.L.: Reconstructing in-the-wild open-vocabulary human-object interactions. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17426–17436 (2025)

39. Xiao, Y., Wang, J., Xue, N., Karaev, N., Makarov, Y., Kang, B., Zhu, X., Bao, H., Shen, Y., Zhou, X.: Spatialtrackerv2: Advancing 3d point tracking with explicit camera motion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6726–6737 (2025)

40. Xiao, Y., Wang, Q., Zhang, S., Xue, N., Peng, S., Shen, Y., Zhou, X.: Spatialtracker: Tracking any 2d pixels in 3d space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20406–20417 (2024)

41. Xie, X., Bhatnagar, B.L., Lenssen, J.E., Pons-Moll, G.: Template free reconstruction of human-object interaction with procedural interaction generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10003–10015 (2024)

42. Xie, X., Wen, B., Chang, Y., Rabeti, H., Li, J., Yuan, Y., Pons-Moll, G., Birchfield, S.: Cari4d: Category agnostic 4d reconstruction of human-object interaction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14006–14016 (2026)

43. Xu, Y., Zhang, J., Zhang, Q., Tao, D.: Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems 35, 38571–38584 (2022)

44. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. Advances in Neural Information Processing Systems 37, 21875–21911 (2024)

45. Yang, X., Kukreja, D., Pinkus, D., Fan, T., Park, J., Shin, S., Cao, J., Liu, J.W., Ugrinovic, N., Sagar, A., et al.: Sam 3d body: Robust full-body human mesh recovery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7209–7219 (2026)

46. Yin, X., Yang, B., Liu, W., Xue, Q., Alamri, A., Fiedler, G., Gao, W.: Progait: A multi-purpose video dataset and benchmark for transfemoral prosthesis users. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8984–8993 (2025)

47. Ying, J., Du, H., Zhang, K., Li, L., Yu, X.: Ldpose: Towards inclusive human pose estimation for limb-deficient individuals in the wild. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9865–9875 (2025)

48. Zhang, B., Ke, L., Harley, A., Fragkiadaki, K.: Tapip3d: Tracking any point in persistent 3d geometry. Advances in Neural Information Processing Systems 38, 135284–135303 (2025)

49. Zhang, C., Le Moing, G., Koppula, S., Rocco, I., Momeni, L., Xie, J., Sun, S., Sukthankar, R., Barral, J.K., Hadsell, R., et al.: Eficiently reconstructing dynamic scenes one d4rt at a time. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7382–7392 (2026)

50. Zhang, J., Herrmann, C., Hur, J., Jampani, V., Darrell, T., Cole, F., Sun, D., Yang, M.H.: Monst3r: A simple approach for estimating geometry in the presence of motion. In: International Conference on Learning Representations. vol. 2025, pp. 82863–82886 (2025)

51. Zheng, C., Zhu, S., Mendieta, M., Yang, T., Chen, C., Ding, Z.: 3d human pose estimation with spatial and temporal transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11656–11665 (2021)

52. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., Wang, Y.: Motionbert: A unified perspective on learning human motion representations. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 15085–15099 (2023)