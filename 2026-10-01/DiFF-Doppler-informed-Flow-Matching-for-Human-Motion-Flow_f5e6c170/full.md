# DiFF: Doppler-informed Flow Matching for Human Motion Flow

Kai Wang and Mingle Zhao

Abstract— Perceiving human motion via privacy-preserving 4D millimeter-wave (mmWave) radar is critical for nextgeneration human-robot interaction (HRI), where point cloud scene flow serves as a foundational motion representation. Yet the extreme sparsity and noise of 4D radar point clouds make non-rigid motion flow estimation severely ill-posed–a challenge that existing rigid-centric methods and prior works fail to adequately address, largely because they neglect the rich Doppler velocity cues inherent in 4D radar. We propose DiFF, a generative framework that marries Doppler-informed motion priors with a Kolmogorov-Arnold Network (KAN)- based conditional flow matching model. At its core, a KANattention mechanism enables expressive feature extraction, while a prior-guided generative process harnesses Doppler cues to regularize the ill-posed solution space. Extensive experiments show that DiFF achieves state-of-the-art (SOTA) performance across diverse real-world datasets, reducing 3D endpoint error to the millimeter scale on the mmBody benchmark. The source code is released at: https://github.com/keroseus/DiFF/.

## I. INTRODUCTION

Perceiving and understanding human motion is a critical capability for next-generation HRI, enabling safer humanrobot collaboration, personalized assistive care, and intuitive smart home environments. Traditional approaches often rely on cameras or wearable sensors. However, these technologies are not only susceptible to environmental factors like poor illumination, but more importantly, they raise significant privacy concerns, rendering them unsuitable for deployment in private spaces. In light of these challenges, researchers have begun exploring alternative sensing modalities [1].

Single-chip mmWave radar has emerged as a low-cost and highly integrated sensor. Due to its Multiple-Input-Multiple-Output (MIMO) transceiver design, mmWave radar can provide reliable point clouds of a scene and effectively adapt to environmental dynamics. Building on these advantages, single-chip radars have seen widespread adoption and have been extensively researched in human sensing, spanning applications from vital sign monitoring and motion recognition [2] to fine-grained gesture recognition [3].

In the context of human motion perception, milliFlow [5] reveals that accurate scene flow estimation is crucial for a robot to better comprehend dynamic human behaviors. Nevertheless, extracting fine-grained motion information from sparse and noisy point clouds from low-cost radars is a formidable challenge. The inherent data sparsity makes establishing reliable point-to-point correspondences between consecutive scans exceptionally difficult, a problem exacerbated by the non-rigid and complex nature of human articulation. This intrinsic data ambiguity renders the estimation of human motion flow from sparse radar point clouds a severely ill-posed problem.

![](images/d9cf4f76ee7e24458065b4b69edab00204a43a8260e2174adaf058dc0e2af821.jpg)  
Fig. 1. Data pipeline overview. (Left) Dataset setup and multi-modal annotations from [4]. (Right) Motion label generation from mesh vertices, skeleton, and images, and the proposed Doppler-informed radar motion flow.

However, a limitation of previous works is that they could not take advantage of Doppler velocity information, as this feature is not available on the Vayyar mmWave radar in [5]. A growing body of research shows that Doppler information inherent in 4D Radar/LiDAR sensors provides crucial additional constraints for key tasks such as motion sensing, perception, and estimation [6]–[13], thus helping to address several long-standing and highly challenging problems at the sensor level. To fill this gap, we introduce DiFF, a novel framework that integrates Doppler-informed motion prior with Kolmogorov-Arnold Network (KAN) conditioned flow matching for estimating high-fidelity human motion flow. Our core innovation lies in combining an efficient flow matching paradigm with a powerful conditioning feature extractor network, KAN-Attention. This network is built upon the principle of KAN and incorporates a global attention structure. By replacing fixed activation functions with learnable rational polynomial functions, our conditioning network can adaptively capture complex geometric and temporal correlations within sparse point clouds, guiding the flow matching process to regress an accurate velocity field. The experimental results demonstrate that DiFF is a fast and accurate framework for estimating human motion flow through 4D mmWave radars. The contributions are:

1) To the best of our knowledge, this work is the first to introduce a flow matching network for human motion flow estimation on point clouds. Our method achieves SOTA accuracy and overall performance, reducing the motion flow prediction error to the millimeter level.

2) We propose a KAN-based point cloud feature extractor. Combined with a global attention mechanism, the proposed architecture can capture both local geometric structures and global contextual information more effectively, while also improving inference efficiency.

3) We design a Doppler-informed motion prior module, leveraging the Doppler information embedded in the 4D radar to estimate an initial radial flow, which serves as a physics-plausible guidance for flow matching. This design facilitates faster convergence and higher estimation accuracy. Building upon this, we incorporate a dynamic conditioning mechanism, which further enhances training stability.

## II. RELATED WORK

## A. Scene Flow in Rigid and Deformable Scenes

Scene flow estimation on point clouds has been predominantly driven by advancements in autonomous driving, where LiDAR sensors provide high-resolution data. Early pioneering works like FlowNet3D [14] and PointPWC-Net [15] established end-to-end deep learning frameworks by constructing cost volumes to learn point correspondences. Subsequent research has significantly improved performance by designing more sophisticated correlation mechanisms and network architectures. For instance, PV-RAFT [16] introduced point-voxel correlation fields to enhance longrange motion modeling. To reduce the reliance on largescale labeled data, self-supervised methods have become a major trend, often leveraging the local rigidity of scenes to generate supervisory signals [17]–[19]. Diffusion models like DifFlow3D [20] have been introduced to improve robustness in noisy environments by modeling motion uncertainty.

Inspired by these successes, researchers have begun to adapt these principles for 4D mmWave radar in automotive scenarios. These works often focus on building robust selfsupervised learning strategies by leveraging radar’s unique properties. For example, Ding et al. first utilized the Dopplerderived radial velocity for self-supervision [21] and later explored cross-modal supervision from other sensors like cameras and LiDAR [22]. Other works like DMRFlow [23] and TARS [24] have further refined the network architecture for traffic scenes. However, a fundamental limitation shared by both the LiDAR and automotive radar literature is their inherent assumption of a world composed of rigid or piecewise-rigid objects (e.g., vehicles, cyclists). This assumption breaks down when faced with the highly articulated and non-rigid nature of human motion, rendering these methods suboptimal for fine-grained human motion analysis.

## B. Motion Sensing and Estimation via Vision and Radar

The challenge of modeling non-rigid motion has been extensively studied in the computer vision community. Methods using RGB images [25]–[29] have demonstrated the ability to estimate ego motions, dense 3D environments and motion fields. Furthermore, multi-modal approaches fusing RGB with depth or event data have shown increased robustness. CamLiFlow [30], for instance, leverages LiDAR’s geometric prior to correct image depth errors, while RPEFlow [31] and BlinkVision [32] integrate event-camera data to better capture high-speed motion. These vision-based methods are effective in capturing deformations of the human body.

However, their reliance on cameras introduces two critical drawbacks for human-centric applications: severe privacy risks and susceptibility to environmental conditions. The intrusive nature of cameras limits their deployment in homes, hospitals, and elderly care facilities. In addition, their performance degrades significantly in poor lighting, smoke, or fog. This fundamental conflict between performance and privacy motivates a shift towards sensing modalities that are both effective and non-invasive. The mmWave radar, with its ability to “see” through darkness and preserve anonymity by capturing sparse points instead of detailed appearances, emerges as an ideal solution to bridge this gap.

## C. Radar-based Human Motion Sensing and Doppler Prior

In fact, a large body of research in human sensing has begun to exploit the multi-modal characteristics of mmWave radar to achieve finer-grained perception. These works are no longer limited to processing the final sparse point clouds but delve into the rich features available at different stages of the radar signal processing pipeline. For example, in the cutting-edge area of 3D human mesh reconstruction, works like mmMesh [33] and M4esh [34] have demonstrated the remarkable ability to reconstruct dynamic, high-fidelity 3D human meshes directly from radar signals, proving that the raw data contains rich information sufficient to recover detailed body posture and shape. In Human Activity Recognition (HAR) [35], [36], some advanced methods also go beyond point cloud geometry, directly utilizing Range-Doppler Heatmaps [36]. Using micro-Doppler effects within this sparse data [37], [38], these models can capture finegrained motion signatures specific to actions like walking or waving, enabling more robust recognition. These works fully demonstrate that the rich features of the mmWave radar are crucial for a deep understanding of the human body, especially the Doppler information, which greatly inspires us to design a Doppler-informed generation network of human motion flow. Our experimental findings further validate the advantageous role of Doppler information from 4D radars/LiDARs in estimating human motion flow.

## III. METHODOLOGY

## A. Problem Formulation

Let $\mathrm { P C } _ { 1 } = \{ \mathbf { p } _ { i } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { N }$ and $\mathrm { P C } _ { 2 } = \{ \mathbf { q } _ { j } \in \mathbb { R } ^ { 3 } \} _ { j = 1 } ^ { M }$ denote consecutive source and target radar point clouds. The forward ground-truth flow is $S _ { \mathrm { g t } } = \{ \mathbf { s } _ { i } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { N }$ , where $\mathbf { p } _ { i } + \mathbf { s } _ { i }$ is the location of source point i at the target time. This definition does not assume that an observed target return $\mathbf { q } _ { j }$ exists at exactly that location, because radar detections are not persistent across frames. Each source return may also carry intensity, amplitude, and a signed Doppler measurement $d _ { i }$ . The task is to estimate $\widehat { S } = \{ \widehat { \mathbf { s } } _ { i } \} _ { i = 1 } ^ { N }$ on the source points from the two point sets and their available attributes.

![](images/ebcb9f554a6cbed167b4c2179149488e215a41e481c2dbc9143cac1420790cbd.jpg)  
Fig. 2. Overview of DiFF. A weight-shared point-wise projection, three KAN blocks, and frame-wise global attention encode the source $\mathrm { P C _ { 1 } }$ and target PC<sub>2</sub>. During training, the intermediate flow $\dot { \boldsymbol { S } } _ { t }$ warps the source points toward the target to construct the state-dependent condition $c _ { t } .$ The KAN decoder combines $c _ { t } , s _ { t } .$ , and the time embedding to predict $v _ { \mathrm { p r e d } }$ . During inference, the condition computed at t = 0 is cached while the ODE updates $\boldsymbol { S } _ { t }$

## B. Framework Overview

DiFF contains a KAN-attention geometric encoder and a conditional flow-matching backbone (Fig. 2). The key idea is to use Doppler to initialize the radial part of motion, and then use a state-dependent conditioning signal to guide the recovery of the unobserved tangential components. A training sample is processed as follows:

1) Geometric Encoding: A weight-shared encoder maps $\mathrm { P C _ { 1 } }$ and $\mathrm { P C _ { 2 } }$ to point-wise features. It preserves input cardinality, refines local neighborhoods with three KAN blocks, and adds frame context via self attention.

2) Conditional Flow Matching: At a sampled path time t, the intermediate flow $S _ { t }$ warps the source points. Cross-frame neighbor aggregation around the warped source returns forms the conditioning signal $c _ { t }$ . A dynamic condition uses the current state $S _ { t }$ , while a static condition uses the initial state $s _ { \mathrm { 0 } } ;$ this distinction is evaluated in the ablation study. A two-layer KAN decoder then predicts the velocity field $v _ { \mathrm { p r e d } } ( S _ { t } , t , c _ { t } )$

The training target is the derivative of a straight path from a Doppler-informed initialization $ { \boldsymbol { S } } _ { 0 }$ to $ { S _ { \mathrm { g t } } }$ . At inference, the model integrates the learned field from $t = 0$ to t = 1.

## C. Network Architecture

This section details the architecture of two core modules that are designed to effectively process sparse point clouds and model their spatiotemporal relationships. Compared with rigid-scene point clouds, human radar returns are sparse in local neighborhoods and strongly coupled across distant articulated body parts. This motivates our use of KAN-based local refinement together with global attention.

1) KAN-Attention Feature Extractor: To generate an informative condition vector y, we design a single-scale extractor inspired by PointKAN-Elite [39]. Unlike traditional hierarchical methods [40], [41], our network maintains full point cloud resolution to prevent critical information loss. It comprises three stages:

a) Siamese Deep Local Feature Extraction: The network first employs a shared-weight convolutional feature extractor to process the two input point clouds, generating initial coarse spatial feature representations. These features are then passed through a series of KAN Blocks to extract fine-grained geometric features, as illustrated in Fig. 2. The design of this module resembles the Local Point Feature (LPF) module in PointKAN [39], but we omit the convolutional steps and directly apply KANs to refine geometric features. During the embedding process, for each neighboring point, we concatenate its relative positional coordinates and features with the features of the source point, expanding the dimension along the neighborhood axis within the point cloud dimension. Point-wise KAN-based refinement is subsequently performed. Each KAN layer utilizes a lightweight Kolmogorov-Arnold network with learnable rational activations, defined as:

$$
\phi ( x ) = \mathrm { S i L U } ( x ) + w \frac { \sum _ { i = 0 } ^ { m } a _ { i } x ^ { i } } { \sqrt { 1 + ( \sum _ { j = 1 } ^ { n } b _ { j } x ^ { j } ) ^ { 2 } } }\tag{1}
$$

where $\left\{ { a } _ { i } \right\}$ and $\{ b _ { j } \}$ are learnable coefficients and w is a learnable scaling factor. Horner’s method for polynomial evaluation and explicit gradient calculation accelerates both training and inference. Unlike an MLP with fixed node activations, KAN learns nonlinear functions on edges [42]; this provides adaptive local bases for sparse radar neighborhoods, where fine-scale geometry, amplitude changes, and Doppler cues can be smoothed by a standard MLP. This design is consistent with the MLP ablation in Table III.

b) Global Context Attention: Following local extraction, features pass through a Global Attention module. This standard multi-head self-attention mechanism aggregates information across the entire frame, capturing global structural dependencies essential for recognizing articulated human motion. It allows each point to integrate features from all other points in its cloud, effectively modeling long-range dependencies. This is particularly important for human radar sensing, where sparse limb returns can be locally ambiguous: similar local clusters may correspond to different body parts or motion directions unless interpreted with torso-level and whole-body context.

2) Flow Matching Backbone: The backbone network is responsible for regressing the velocity field. Its processing pipeline consists of the following steps:

a) PC Warp Module: In this module, we warp the target point cloud toward the source point cloud to facilitate estimation. For each point in the source point cloud, we identify its neighbors in the warped target point cloud. The relative positions between them are computed and concatenated with their corresponding spatial features and the features of the source point. A key advantage of this design is its dynamic re-evaluation: as the flow field $S _ { t }$ evolves during the flow matching process, the warping distance changes, allowing the next module to adaptively refine the cost volume. This contributes to stable loss convergence, mitigating the training oscillations often observed with KANs.

b) Cost Volume Module: This module closely resembles the KAN Block. After the PC warp module, the aggregated information is processed by another KAN-based aggregator, followed by a max-pooling operation over the neighbor features. While this operation is identical to that in the KAN Block, we omit the final normalization and ReLU activation. We posit that these steps would potentially discard conditional information, which contrasts with the KAN Block’s objective of refining geometric features. This process yields a point-wise cost volume that quantifies crossframe similarity. It is subsequently concatenated with the source point cloud’s own features to form the final condition vector c. In our notation, the state-dependent version is denoted as $c _ { t } { : }$ dynamic conditioning recomputes this signal from the current flow state $S _ { t }$ , whereas static conditioning fixes it at the initial state $S _ { 0 }$

c) KAN Decoder Module: This module fuses the multisource feature representations to derive the final velocity field. To integrate temporal information, we employ a sinusoidal positional encoding scheme. Both this temporal embedding and the current flow representation $S _ { t }$ are independently processed by an MLP-based projection layer to achieve dimensional alignment with the conditional vector c. The holistic state representation is then constructed via an element-wise summation of the flow embedding, the temporal encoding, and the conditional context. Finally, a two-layer lightweight Kolmogorov-Arnold Network (KAN) serves as the decoder to regress the refined 3D velocity for each point within the current flow state $S _ { t }$ . During inference, we compute the condition once at $t ~ = ~ 0$ and cache $c _ { 0 }$ to reduce repeated neighbor search, while the decoder still receives the evolving $S _ { t }$ and t.

## D. Doppler-informed Flow Matching

A central contribution of our work is the integration of Doppler velocity priors into the Flow Matching framework. Instead of initiating the ODE trajectory from a standard Gaussian noise distribution $\mathcal { N } ( 0 , \bf { I } )$ , we start from a distribution that is already informed by the observed motion, as shown in Fig. 1. Specifically, we define the starting point of our flow trajectory $ { \boldsymbol { S } } _ { 0 }$ as:

$$
S _ { 0 } = \alpha \cdot \frac { \mathbf { V } _ { \mathrm { d o p p l e r } } } { C } + \epsilon , \quad \mathrm { w h e r e } \quad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbf { I } )\tag{2}
$$

Here, $\mathbf { V } _ { \mathrm { d o p p l e r } }$ is the Doppler velocity, C is a constant for the radar frame rate (e.g., 30 for Arbe [43]), α is a constant to scaling the mean of the Doppler radial flow (typically set to 1), and ϵ is a small Gaussian noise to maintain stochasticity.

We adopt the Rectified Flow formulation [44], which defines a straight-line path between the starting flow $ { \boldsymbol { S } } _ { 0 }$ and the target ground truth flow $\begin{array} { r } { S _ { g t } \mathrm { : } } \end{array}$

$$
S _ { t } = ( 1 - t ) S _ { 0 } + t S _ { g t }\tag{3}
$$

As in [45], the target velocity for a path with mean $\mu _ { t }$ and standard deviation $\sigma _ { t }$ is given by $\begin{array} { r } { v _ { t a r g e t } ( S _ { t } , t ) = \frac { \sigma _ { t } ^ { \prime } } { \sigma _ { t } } ( S _ { t } - } \end{array}$ $\mu _ { t } ) + \mu _ { t } ^ { \prime }$ . For the straight-line path in (3), these terms are:

$$
\mu _ { t } = ( 1 - t ) \frac { \alpha \mathbf { V } _ { \mathrm { d o p p l e r } } } { C } + t S _ { g t }\tag{4}
$$

$$
\sigma _ { t } = ( 1 - t ) \sigma _ { 0 } + t \sigma _ { 1 } = ( 1 - t ) \sigma\tag{5}
$$

$$
\mu _ { t } ^ { \prime } = { \frac { d \mu _ { t } } { d t } } = S _ { g t } - { \frac { \alpha \mathbf { V } _ { \mathrm { d o p p l e r } } } { C } }\tag{6}
$$

$$
\sigma _ { t } ^ { \prime } = \frac { d \sigma _ { t } } { d t } = - \sigma\tag{7}
$$

Substituting these into the general formula, the target velocity field that our network $v _ { p r e d }$ is trained to regress simplifies to the time derivative of the path:

$$
v _ { t a r g e t } = \frac { d S _ { t } } { d t } = S _ { g t } - S _ { 0 }\tag{8}
$$

The training objective is to minimize the Mean Squared Error between the network’s prediction and this target velocity:

$$
\mathcal { L } = \mathbb { E } _ { t , \mathrm { P C _ { 1 } } , \mathrm { P C _ { 2 } } } \left[ \| v _ { p r e d } ( S _ { t } , t , c _ { t } ) - ( S _ { g t } - S _ { 0 } ) \| ^ { 2 } \right]\tag{9}
$$

By initializing the flow trajectory closer to the final solution, this Doppler-informed approach provides a strong inductive bias, significantly accelerating the training process and improving the final accuracy of the scene flow estimation. At inference, the cached condition $c _ { 0 }$ is used for efficiency, while $S _ { t }$ and t still evolve through the ODE solver.

![](images/3c7b79fd0b1f326692f74e6e949338d58f508ea7d3f62587736712f30b3d7c7c.jpg)  
Fig. 3. The effect of inference steps on EPE3D and average inference time

## IV. EXPERIMENTS

## A. Experimental Setup

## 1) Implementation Details:

a) Datasets: All experiments are conducted on two datasets: the milliFlow dataset [5] and the mmBody dataset [4]. The milliFlow dataset provides sparser and noisier point clouds with only a single feature (“intensity”). The mmBody dataset is a large-scale, high-quality multi-modal dataset designed for 3D human mesh reconstruction from millimeterwave radar. The employed Arbe radar [43] is equipped with a 48×48 antenna array, which surpasses many commercial single-chip radars and provides rich point clouds where each point carries features including intensity, Doppler velocity, and amplitude. To focus on the core challenge of non-rigid motion estimation, we use only the non-occluded sequences. We adopt a 4:1 train-test split, using sequences 0-5, 7-16 for training and sequences 6, 17-19 for testing, ensuring that subjects in the test set are unseen during training. We use a subset to evaluate model generalization.

b) Data Processing and Ground Truth Generation: The mmBody dataset provides not only rich radar point clouds, often exceeding 6000 points per frame, but also synchronized 2D images, SMPL-X 3D meshes, and skeletal poses. As the raw data contains significant environmental clutter, we first filter the point cloud for each frame by retaining only points within a 0.15 m radius of the 3D human SMPL-X skeleton. Each of these points is then assigned a semantic body part label based on the nearest skeleton. Following the automatic annotation technique from milliFlow [5], we compute the SE(3) transformation matrix for each body part between two consecutive frames. The ground truth flow $\mathcal { S } _ { g t }$ for each point p on a body part j is then derived using the kinematic transformation $s _ { i } = ( T _ { j } \circ p _ { i } ) - p _ { i }$ . For training, we process pairs of frames and uniformly sample or pad the filtered point clouds to a fixed size of $N = 5 1 2$ . During inference, we do not perform sampling and directly estimate scene flow on the variable-sized point clouds.

## 2) Training and Inference Details:

a) Doppler Prior Generation: For milliFlow, which does not contain Doppler measurements, we follow the firstsubmission diagnostic setting and construct radial priors by projecting available flow annotations onto radial directions. This setting allows us to analyze the behavior of the flowmatching backbone on sparser radar point clouds; a fully deployable no-Doppler setting can be addressed by replacing this prior with a learned or uninformed initialization.

For mmBody training, the radial component is obtained by projecting the labeled flow onto the radar line of sight: $\mathbf { s } _ { \mathrm { r a d } , i } = ( \mathbf { s } _ { i } ^ { \top } \widehat { \mathbf { r } } _ { i } ) \widehat { \mathbf { r } } _ { i }$ , and we set ${ \bf s } _ { i } ^ { D } = { \bf s } _ { \mathrm { r a d } , i }$ in (2). The random component is sampled with a scaler parameter, and the initial trajectory preserves the physically observed radial motion while still allowing exploration of the unobserved tangential components. At mmBody inference, ${ \bf s } _ { i } ^ { D }$ is computed from the measured Doppler velocity and frame interval, with $\epsilon _ { \perp , i } =$ 0.

b) Conditioning: The conditioning signal refers to the cross-frame cost-volume feature used by the velocity decoder. In the static variant, this signal is computed from the initial state ${ \cal S } _ { 0 } ,$ i.e., the Doppler-informed flow at $t ~ = ~ 0$ In the dynamic variant used for training, the source points are warped by the sampled intermediate state $S _ { t }$ , and the condition $c _ { t }$ is recomputed from the corresponding target neighbors. Thus static conditioning uses the input at $t = 0$ whereas dynamic conditioning uses the time-dependent state along the flow path. During inference, we compute $c _ { 0 }$ once and cache it to avoid repeated neighbor searches, while the decoder still receives the evolving $S _ { t }$ and t.

c) Parameter Settings: The PyTorch model uses 64- dimensional encoder features, three KAN blocks, a 128- dimensional condition and state embedding, and a two-layer KAN decoder. We train end-to-end on one NVIDIA RTX 4080 GPU for 200 epochs with batch size 16, AdamW, an initial learning rate of $1 \times 1 0 ^ { - 3 }$ , and cosine annealing. Inference uses 10 ODE steps.

## B. Comparison with State-of-the-Art Methods

We compare DiFF with several SOTA methods, including the leading radar-based approach for human motion, and several strong LiDAR-based methods retrained on the milliFlow and mmBody datasets, as reported in Tables I and II.

a) Evaluation Metrics: For a comprehensive evaluation, we employ four key metrics: 3D End-Point-Error (EPE3D) in mm, Strict 3D Accuracy (Acc3D Strict) in (%), Relaxed 3D Accuracy (Acc3D Relax) in (%), and the EPE3D distribution across test samples. To rigorously assess the model’s performance on fine-grained human motion, we adopt significantly stricter accuracy thresholds than those established in the autonomous driving domain. Besides, EPE3D distribution of test samples serves as a key indicator of an algorithm’s robustness, reflecting its ability to mitigate the effects of noise inherent in sparse mmWave radar point clouds.These evaluation metrics are defined as follows:

• EPE3D (mm): The average L2 endpoint error between the predicted and ground-truth scene-flow vectors over all evaluated points. Lower values indicate more accurate scene-flow estimation.

TABLE I  
QUANTITATIVE COMPARISON ON THE MILLIFLOW DATASET. THE CUMULATIVE ERROR DISTRIBUTION IS SHOWN IN FIG. 4 (LEFT). SINCE MILLIFLOW DOES NOT PROVIDE DOPPLER MEASUREMENTS, THE DIFF ROW FOLLOWS THE RADIAL-PRIOR CONSTRUCTION DESCRIBED IN THE EXPERIMENTAL SETUP. BOLD MARKS THE BEST AGGREGATE VALUE, INCLUDING TIES.
<table><tr><td>Model</td><td>EPE3D (mm) ↓</td><td>Acc3D Strict (%) ↑</td><td>Acc3D Relax (%) ↑</td><td>0-1 mm (%)</td><td>1-10 mm (%)</td><td>10-100 mm (%)</td><td>100-1000 mm (%)</td><td>&gt;1000 mm (%)</td><td>Time (ms) ↓</td></tr><tr><td>FlowNet3D [14]</td><td>1198.131</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>35.56</td><td>64.44</td><td>11.43</td></tr><tr><td>DiffSF [46]</td><td>787.278</td><td>1.22</td><td>1.38</td><td>0.01</td><td>0.01</td><td>0.46</td><td>74.33</td><td>25.19</td><td>32.87</td></tr><tr><td>FLOT [47]</td><td>200.508</td><td>1.54</td><td>8.53</td><td>0.00</td><td>0.00</td><td>22.53</td><td>77.38</td><td>0.09</td><td>11.07</td></tr><tr><td>milliFlow [5]</td><td>177.132</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>99.99</td><td>0.01</td><td>30.72</td></tr><tr><td>DifFlow3D [20]</td><td>120.295</td><td>74.39</td><td>86.51</td><td>0.01</td><td>31.24</td><td>60.92</td><td>4.71</td><td>3.12</td><td>84.63</td></tr><tr><td>FlowStep3D [48]</td><td>84.264</td><td>29.63</td><td>55.24</td><td>0.00</td><td>2.66</td><td>75.01</td><td>22.32</td><td>0.01</td><td>17.83</td></tr><tr><td>DiFF (Ours)</td><td>27.119</td><td>74.61</td><td>86.81</td><td>4.07</td><td>29.71</td><td>62.75</td><td>3.46</td><td>0.01</td><td>19.26</td></tr></table>

TABLE II

QUANTITATIVE COMPARISON ON THE MMBODY DATASET. THE CUMULATIVE ERROR DISTRIBUTION IS SHOWN IN FIG. 4 (RIGHT). LOWER EPE3D AND INFERENCE TIME ARE BETTER; HIGHER ACC3D VALUES ARE BETTER. BOLD MARKS THE BEST AGGREGATE VALUE, INCLUDING TIES.
<table><tr><td>Model</td><td>EPE3D (mm) ↓</td><td>Acc3D Strict (%) ↑</td><td>Acc3D Relax (%) ↑</td><td>0-1 mm (%)</td><td>1-10 mm (%)</td><td>10-100 mm (%)</td><td>100-1000 mm (%)</td><td>&gt;1000 mm (%)</td><td>Time (ms) ↓</td></tr><tr><td>DiffSF [46]</td><td>319.4</td><td>4.42</td><td>4.60</td><td>0.06</td><td>0.39</td><td>7.32</td><td>92.19</td><td>0.04</td><td>29.90</td></tr><tr><td>milliFlow [5]</td><td>80.2</td><td>0.03</td><td>0.20</td><td>0.00</td><td>0.00</td><td>76.81</td><td>23.19</td><td>0.00</td><td>33.30</td></tr><tr><td>FLOT [47]</td><td>30.4</td><td>47.44</td><td>86.26</td><td>0.00</td><td>4.03</td><td>95.47</td><td>0.50</td><td>0.00</td><td>19.05</td></tr><tr><td>FlowNet3D [14]</td><td>14.2</td><td>86.01</td><td>90.34</td><td>0.96</td><td>41.83</td><td>57.21</td><td>0.00</td><td>0.00</td><td>35.50</td></tr><tr><td>FlowStep3D [48]</td><td>13.7</td><td>88.96</td><td>95.49</td><td>0.00</td><td>63.50</td><td>36.09</td><td>0.42</td><td>0.00</td><td>22.83</td></tr><tr><td>DifFlow3D [20]</td><td>2.8</td><td>98.63</td><td>99.62</td><td>2.67</td><td>92.24</td><td>5.09</td><td>0.00</td><td>0.00</td><td>89.82</td></tr><tr><td>DiFF (Ours)</td><td>2.1</td><td>98.63</td><td>99.63</td><td>63.36</td><td>31.78</td><td>4.86</td><td>0.00</td><td>0.00</td><td>10.10</td></tr></table>

![](images/e4e1b7f93519fc16e7637080763c82b44685648482c018752433aee450a9dd0c.jpg)

![](images/bd0efc8c9322eb733bb9ba2661f7ee6ab2bd976277b6936a72c3500a95338cdf.jpg)  
Fig. 4. Cumulative distributions of per-sample EPE3D on (Left) milliFlow [5] and (Right) mmBody [4], corresponding to Tables I and II, respectively.

• Acc3D Strict (%): For each test sample, this metric measures the proportion of points whose point-wise endpoint error is no greater than 0.025 m or whose relative error, normalized by the magnitude of the corresponding ground-truth flow vector, is no greater than 2.5%. The reported value is averaged over all test samples. Higher values are better.

• Acc3D Relax (%): This metric is calculated in the same manner as Acc3D Strict, but uses relaxed thresholds of 0.05 m for the point-wise endpoint error and 5.0% for the relative error. The reported value is averaged over all test samples. Higher values are better.

• EPE3D Distribution: For each test sample, we compute the mean endpoint error across evaluated points and assign it to one of five ranges: 0–1 mm, 1–10 mm, 10–100 mm, 100–1000 mm, or above 1000 mm. A larger share in lower-error ranges indicates greater accuracy and consistency.

Beyond accuracy, our comprehensive evaluation also considers computational efficiency. As these demanding metrics highlight our algorithm’s SOTA performance, our analysis of the average inference time further demonstrates its practicality and superiority for real-world applications.

b) Results: In Table I, DiFF maintains SOTA performance even with sparser radar points, with 33.78% of the flow estimates exhibiting an EPE3D below 10 mm, further confirming its robustness. The cumulative error curves in Fig. 4 show that, regardless of the dataset tested, DiFF exhibits stronger robustness compared to others, with a substantial portion of test data achieving errors under 10 mm.

In Table II, DiFF outperforms by a significant margin. It reduces the EPE3D to only 2.1 mm, which represents an order-of-magnitude improvement over the previous leading method, milliFlow. While other diffusion-based models such as DifFlow3D [20] achieve competitive accuracy, their flow errors are mostly concentrated in the range of 1-10 mm, and their inference speed is considerably slower. In contrast, DiFF not only achieves a lower EPE3D, but also locates 63.36% of the samples within the precise 0-1 mm range, demonstrating a clear advantage for real-time applications.

These advancements can be attributed to two key innovations in our method. First, the substantial performance leap is driven by the effective integration of Doppler priors into the generative flow-matching framework. Second, by replacing the cumbersome downsampling backbones (e.g., PointNet++) with an efficient hybrid architecture that leverages KANs and global attention on raw point clouds, we are able to robustly capture both local and global geometric structures in sparse point clouds while mitigating the impact of noise. This architectural efficiency significantly reduces training and inference time, enabling a compelling combination of high accuracy and real-time performance.

TABLE III  
ABLATION STUDY ON CORE COMPONENTS. CONFIGURATION NAMES MATCH THE VARIANTS DEFINED IN SECTION IV.
<table><tr><td>Configuration</td><td>EPE3D (mm) ↓</td></tr><tr><td>DiFF (Full Model)</td><td>2.1</td></tr><tr><td>w/o Doppler Prior</td><td>4.4</td></tr><tr><td>Using MLP</td><td>2.7</td></tr><tr><td>Static Conditioning</td><td>3.3</td></tr></table>

## C. Ablation Experiment

To analyze the sources of our model’s performance gains, we conduct a series of ablation studies, with results summarized in Table III and Table IV.

a) Effect of Core Components: Table III validates our key design choices.

• w/o Doppler Prior: We replace the Doppler-informed initialization with a standard Gaussian noise prior. The sharp performance drop confirms that our Doppler initialization provides a strong inductive bias.

• Using MLP: We replace all KAN-based components with standard MLPs. The performance degradation indicates that KAN’s adaptive activations are more effective in capturing complex geometric relationships in sparse point clouds.

• Static Conditioning: We compare the dynamic cost volume with a training variant that uses static conditioning. The results show that dynamic conditioning provides more robust guidance throughout the ODE solving process.

b) Effect of Initial Noise Variance: We also study the impact of the variance $\sigma ^ { 2 }$ of the noise in the Doppler prior. In Table IV, the performance is optimal at $\sigma ^ { 2 } = 0 . 0 1 ^ { 2 }$ . A larger variance introduces more noise, making it difficult to learn the path from the Doppler mean to the target flow. Conversely, a smaller variance reduces the model’s ability to correct for errors in the initial prior, as the ODE trajectory becomes too deterministic and can be led astray by inaccuracies in the Doppler measurement and velocity errors made in the ODE solver. Notably, the setting with the best EPE3D does not produce the lowest training loss, indicating that training loss alone is not sufficient for selecting the noise variance.

TABLE IV  
SENSITIVITY TO NOISE VARIANCE. THE LOWEST EPE3D IS HIGHLIGHTED; TRAINING LOSS IS REPORTED FOR REFERENCE AND IS NOT MINIMIZED AT THE BEST-EPE3D SETTING.
<table><tr><td>Variance  $( \sigma ^ { 2 } )$ </td><td></td><td>EPE3D (mm) ↓ Avg. Training Loss</td></tr><tr><td>1.0</td><td>3.0</td><td> $3 . 2 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $0 . 1 ^ { 2 }$ </td><td>2.3</td><td> $9 . 6 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $\mathbf { 0 . 0 1 ^ { 2 } }$ </td><td>2.1</td><td> $3 . 2 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $0 . 0 0 1 ^ { 2 }$ </td><td>2.4</td><td> $9 . 6 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $0 . 0 0 0 1 ^ { 2 }$ </td><td>2.6</td><td> $4 . 8 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $0 . 0 0 0 0 1 ^ { 2 }$ </td><td>3.0</td><td> $4 . 8 0 \times 1 0 ^ { - 6 }$ </td></tr></table>

## V. CONCLUSION

In this work, we presented DiFF, a novel framework for human motion scene flow estimation from sparse mmWave radar point clouds. By integrating Doppler velocity priors into a flow matching model and leveraging a KAN-based feature extractor, our approach achieves state-of-the-art performance on the mmBody dataset. Ablation studies confirm the significance of each component: the Doppler prior, the KAN architecture, and the dynamic correlation mechanism.

Despite its strengths, DiFF has limitations that suggest future directions: (1) replacing the linear flow path with a learned, nonlinear transport to better exploit the anisotropic Doppler prior; (2) extending the framework to multi-person scenarios with occlusion handling. Overall, DiFF represents a meaningful step toward human motion analysis with applications in assistive robotics and human-robot interaction.

## REFERENCES

[1] F. Adib and D. Katabi, “See through walls with wifi!” in Proceedings of the ACM SIGCOMM 2013 conference on SIGCOMM, 2013, pp. 75–86.

[2] J. Lien, N. Gillian, M. E. Karagozler, P. Amihood, C. Schwesig, E. Olson, H. Raja, and I. Poupyrev, “Soli: Ubiquitous gesture sensing with millimeter wave radar,” ACM Transactions on Graphics (ToG), vol. 35, no. 4, pp. 1–19, 2016.

[3] B. Jin, X. Ma, B. Hu, Z. Zhang, Z. Lian, and B. Wang, “Gesturemmwave: Compact and accurate millimeter-wave radar-based dynamic gesture recognition for embedded devices,” IEEE Transactions on Human-Machine Systems, vol. 54, no. 3, pp. 337–347, 2024.

[4] A. Chen, X. Wang, S. Zhu, Y. Li, J. Chen, and Q. Ye, “mmbody benchmark: 3d body reconstruction dataset and analysis for millimeter wave radar,” in Proceedings ofthe 30th ACM International Conference on Multimedia, 2022, pp. 3501–3510.

[5] F. Ding, Z. Luo, P. Zhao, and C. X. Lu, “milliflow: Scene flow estimation on mmwave radar point cloud for human motion sensing,” in European Conference on Computer Vision. Springer, 2024, pp. 202–221.

[6] K. Harlow, H. Jang, T. D. Barfoot, A. Kim, and C. Heckman, “A new wave in robotics: Survey on recent mmwave radar applications in robotics,” IEEE Transactions on Robotics, vol. 40, pp. 4544–4560, 2024.

[7] B. Hexsel, H. Vhavle, and Y. Chen, “DICP: Doppler Iterative Closest Point Algorithm,” in Proceedings of Robotics: Science and Systems, New York City, NY, USA, June 2022.

[8] Y. Gu, H. Cheng, K. Wang, D. Dou, C. Xu, and H. Kong, “Learning moving-object tracking with fmcw lidar,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 3747–3753.

[9] Y. Wu, D. J. Yoon, K. Burnett, S. Kammel, Y. Chen, H. Vhavle, and T. D. Barfoot, “Picking up speed: Continuous-time lidar-only odometry using doppler velocity measurements,” IEEE Robotics and Automation Letters, vol. 8, no. 1, pp. 264–271, 2022.

[10] D. J. Yoon, K. Burnett, J. Laconte, Y. Chen, H. Vhavle, S. Kammel, J. Reuther, and T. D. Barfoot, “Need for speed: Fast correspondencefree lidar-inertial odometry using doppler velocity,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 5304–5310.

[11] M. Zhao, J. Wang, T. Gao, C. Xu, and H. Kong, “Fmcw-lio: A doppler lidar-inertial odometry,” IEEE Robotics and Automation Letters, vol. 9, no. 6, pp. 5727–5734, 2024.

[12] ——, “Free-init: Scan-free, motion-free, and correspondence-free initialization for doppler lidar-inertial systems,” IEEE Robotics and Automation Letters, vol. 9, no. 12, pp. 11 329–11 336, 2024.

[13] A. Khoche, Q. Zhang, Y. Cai, S. S. Mansouri, and P. Jensfelt, “Dogflow: Self-supervised lidar scene flow via cross-modal doppler guidance,” IEEE Robotics and Automation Letters, 2026.

[14] X. Liu, C. R. Qi, and L. J. Guibas, “Flownet3d: Learning scene flow in 3d point clouds,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 529–537.

[15] W. Wu, Z. Y. Wang, Z. Li, W. Liu, and L. Fuxin, “Pointpwc-net: Cost volume on point clouds for (self-) supervised scene flow estimation,” in European conference on computer vision. Springer, 2020, pp. 88–107.

[16] Y. Wei, Z. Wang, Y. Rao, J. Lu, and J. Zhou, “Pv-raft: Pointvoxel correlation fields for scene flow estimation of point clouds,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 6954–6963.

[17] R. Li, C. Zhang, G. Lin, Z. Wang, and C. Shen, “Rigidflow: Selfsupervised scene flow learning on point clouds by local rigidity prior,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 959–16 968.

[18] S. A. Baur, D. J. Emmerichs, F. Moosmann, P. Pinggera, B. Ommer, and A. Geiger, “Slim: Self-supervised lidar scene flow and motion segmentation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 13 126–13 136.

[19] H. Mittal, B. Okorn, and D. Held, “Just go with the flow: Selfsupervised scene flow estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 177–11 185.

[20] J. Liu, G. Wang, W. Ye, C. Jiang, J. Han, Z. Liu, G. Zhang, D. Du, and H. Wang, “Difflow3d: Toward robust uncertainty-aware scene flow estimation with iterative diffusion-based refinement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 109–15 119.

[21] F. Ding, Z. Pan, Y. Deng, J. Deng, and C. X. Lu, “Self-supervised scene flow estimation with 4-d automotive radar,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8233–8240, 2022.

[22] F. Ding, A. Palffy, D. M. Gavrila, and C. X. Lu, “Hidden gems: 4d radar scene flow learning using cross-modal supervision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9340–9349.

[23] M. Zhai, B.-K. Bao, and X. Xiang, “Dmrflow: 4d radar scene flow estimation with decoupled matching and refinement,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[24] J. Wu, M. Braun, D. Spata, and M. Rottmann, “Tars: Traffic-aware radar scene flow estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 26 075–26 084.

[25] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison, “Codeslam—learning a compact, optimisable representation for dense visual slam,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2560–2568.

[26] G. Yang and D. Ramanan, “Upgrading optical flow to 3d scene flow through optical expansion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 1334–1343.

[27] J. Hur and S. Roth, “Self-supervised monocular scene flow estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7396–7405.

[28] M. Zhao, D. Zhou, X. Song, X. Chen, and L. Zhang, “Dit-slam: realtime dense visual-inertial slam with implicit depth representation and tightly-coupled graph optimization,” Sensors, vol. 22, no. 9, p. 3389, 2022.

[29] R. Li and T. Nguyen, “Monoplflownet: Permutohedral lattice flownet for real-scale 3d scene flow estimation with monocular images,” in European Conference on Computer Vision. Springer, 2022, pp. 322– 339.

[30] H. Liu, T. Lu, Y. Xu, J. Liu, W. Li, and L. Chen, “Camliflow: bidirectional camera-lidar fusion for joint optical flow and scene flow estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5791–5801.

[31] Z. Wan, Y. Mao, J. Zhang, and Y. Dai, “Rpeflow: Multimodal fusion of rgb-pointcloud-event for joint optical flow and scene flow estimation,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 030–10 040.

[32] Y. Li, Y. Shen, Z. Huang, S. Chen, W. Bian, X. Shi, F.-Y. Wang, K. Sun, H. Bao, Z. Cui et al., “Blinkvision: A benchmark for optical flow, scene flow and point tracking estimation using rgb frames and events,” in European conference on computer vision. Springer, 2024, pp. 19–36.

[33] H. Xue, Y. Ju, C. Miao, Y. Wang, S. Wang, A. Zhang, and L. Su, “mmmesh: Towards 3d real-time dynamic human mesh construction using millimeter-wave,” in Proceedings of the 19th annual international conference on mobile systems, applications, and services, 2021, pp. 269–282.

[34] H. Xue, Q. Cao, Y. Ju, H. Hu, H. Wang, A. Zhang, and L. Su, “M4esh: mmwave-based 3d human mesh construction for multiple subjects,” in Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems, 2022, pp. 391–406.

[35] Z. Gu, X. He, G. Fang, C. Xu, F. Xia, and W. Jia, “Millimeter wave radar-based human activity recognition for healthcare monitoring robot,” arXiv preprint arXiv:2405.01882, 2024.

[36] L. Cao, S. Liang, Z. Zhao, D. Wang, C. Fu, and K. Du, “Human activity recognition method based on fmcw radar sensor with multidomain feature attention fusion network,” Sensors, vol. 23, no. 11, p. 5100, 2023.

[37] X. Yang, W. Gao, X. Qu, and H. Meng, “Generalizable indoor human activity recognition method based on micro-doppler corner point cloud and dynamic graph learning,” IEEE Transactions on Aerospace and Electronic Systems, vol. 61, no. 2, pp. 5195–5209, 2024.

[38] Z. Zeng, M. G. Amin, and T. Shan, “Automatic arm motion recognition based on radar micro-doppler signature envelopes,” IEEE Sensors Journal, vol. 20, no. 22, pp. 13 523–13 532, 2020.

[39] Y. Shi, Q. He, Y. Liu, X. Liu, and J. Su, “Kan or mlp? point cloud shows the way forward,” arXiv preprint arXiv:2504.13593, 2025.

[40] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 652–660.

[41] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” Advances in neural information processing systems, vol. 30, 2017.

[42] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Y. Hou, and M. Tegmark, “Kan: Kolmogorov-arnold networks,” arXiv preprint arXiv:2404.19756, 2024.

[43] Arbe Robotics, “Arbe,” https://arberobotics.com/, 2022.

[44] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022.

[45] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[46] Y. Zhang, B. Wandt, M. Magnusson, and M. Felsberg, “Diffsf: Diffusion models for scene flow estimation,” Advances in Neural Information Processing Systems, vol. 37, pp. 111 227–111 247, 2024.

[47] G. Puy, A. Boulch, and R. Marlet, “Flot: Scene flow on point clouds guided by optimal transport,” in European conference on computer vision. Springer, 2020, pp. 527–544.

[48] Y. Kittenplon, Y. C. Eldar, and D. Raviv, “Flowstep3d: Model unrolling for self-supervised scene flow estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4114–4123.