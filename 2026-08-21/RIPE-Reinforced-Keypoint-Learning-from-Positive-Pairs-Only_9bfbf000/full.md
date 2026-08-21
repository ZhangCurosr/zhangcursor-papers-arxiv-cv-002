# RIPE++: Reinforced Keypoint Learning from Positive Pairs Only

Johannes Künzel<sup>1,2[0000−0002−3561−2758]</sup>, Peter Eisert<sup>1,2[0000−0001−8378−4805]</sup>, and Anna Hilsmann<sup>1[0000−0002−2086−0951]</sup>

<sup>1</sup> Fraunhofer Heinrich-Hertz-Institute, HHI, Germany <sup>2</sup> Humboldt University Berlin, Germany

![](images/c1461951552e9b1146c73b1772993b9a235c353dee6c083cd4f297a4ca949d7c.jpg)  
Fig. 1: We propose a reward formulation for weakly-supervised keypoint extraction training that enables training from positive pairs only and is directly applicable to the training of learned matching architectures, demonstrated with LightGlue [25].

Abstract. Sparse keypoint extraction and matching underpin core tasks in geometric computer vision, including structure-from-motion, visual SLAM, augmented reality, and medical image registration. Learning robust local feature representations, however, typically requires accurate camera poses or depth supervision, which are often unavailable in realworld settings. Reinforcement learning (RL) has recently emerged as a promising alternative, requiring only the information if two images show the same scene or not. However, existing RL formulations such as RIPE rely on coarse binary rewards and carefully constructed negative training pairs, limiting training stability and descriptor discriminability. In this paper, we revisit RL-based keypoint learning and propose a reward that fully exploits the geometric consistency signal, deriving both reward and penalty from a single positive pair without contrasting against negatives. This richer signal provides suficient supervisory contrast to learn discriminative detectors and descriptors from positive image pairs alone, enabling representation learning under extremely limited supervision. Furthermore, we show that the same RL objective can be extended to the matching stage by adapting LightGlue, raising AUC@5 on MegaDepth1500 from 56.58 to 59.65 and enabling weakly-supervised training of the full sparse matching pipeline from image pairs with partial visual overlap. We validate our approach on established benchmarks, demonstrating competitive results compared to fully-supervised methods. We further show that the method can be even trained on low texture medical video sequences, where camera poses are usually unavailable and standard SfM pipelines often fail. Code and data are available at https://github.com/fraunhoferhhi/RIPEpp.

## 1 Introduction

Sparse keypoint extraction and matching is a cornerstone of geometric computer vision, underpinning structure-from-motion (SfM), visual SLAM, augmented reality, and medical image registration. Despite enormous progress, modern learned pipelines still depend on dense geometric supervision, i.e. ground-truth depth and camera poses, which presupposes calibrated hardware or ofline SfM reconstruction and is expensive or impossible to acquire at scale. The limiting factor in modern keypoint learning is thus no longer model capacity but the availability of geometric supervision, making representation learning under limited supervision a key challenge for geometric vision.

Reinforcement learning (RL) circumvents the fact that the discrete nature of keypoint selection prevents direct gradient-based optimization and greatly reduces the required supervision, as shown in RIPE [19]. RIPE trained an RL-based detector and descriptor from image pairs annotated only with a binary samescene/diferent-scene label, using RANSAC-based fundamental matrix estimation as a geometry-aware reward. This reduces supervision to a single bit per pair, obtainable from the temporal structure of a video stream at no annotation cost. Yet two limitations remain: the binary reward is coarse, limiting training stability and descriptor discriminability while requiring carefully constructed negative pairs; and the pipeline still depends on a matcher trained with full pose or depth supervision, negating much of the practical benefit.

We revisit RL-based keypoint learning and show that the efectiveness of weak supervision critically depends on how geometric consistency is translated into a reward signal. We therefore propose a new reward based on the number of geometrically plausible inliers and outliers. This richer signal makes negative pairs unnecessary: the reward itself provides suficient contrast to learn discriminative detectors and descriptors from positive pairs alone. We further extend the RL paradigm to the matching stage by adapting LightGlue to the same weaklysupervised objective, raising the AUC@5 on MegaDepth1500 from 56.58 to 59.65, and validate on established benchmarks and on medical video sequences, a domain where ground-truth poses are unavailable and standard SfM pipelines such as COLMAP cannot be applied. Our method thus operates in the very-limitedresource regime along two axes, annotation (only binary) and data curation (raw video streams, no negative-pair mining), demonstrating that accurate local feature representations can be learned with minimal supervision and dataset preparation. Our major contributions are as follows:

– We address the limitations of the binary reward formulation in RIPE and propose a geometric reward that generates reward and penalty by taking inliers and outliers into account, which provides a richer and more stable training signal for RL-based keypoint learning.

– This improved reward formulation naturally removes the need for negative training pairs, simplifying the training pipeline and dataset curation.

– We show that this allows training from data sources as simple as video streams, as we show for medical image data.

– We extend the same reward to a transformer-based matcher (LightGlue), improving AUC@5 on MegaDepth1500 from 56.58 to 59.65 under weak supervision, removing the remaining dependence on fully-supervised matching. We validate the approach on standard benchmarks, performing favorably compared to fully-supervised training methods.

## 2 Related Work

Hand-crafted keypoint extraction. Classical detectors and descriptors such as SIFT [27], SURF [5], and ORB [34] rely on hand-crafted image statistics to identify repeatable interest points and encode their local appearance. While highly eficient, these methods lack robustness under severe appearance changes such as day–night transitions or seasonal variation, motivating the shift toward learned alternatives.

Supervised and self-supervised keypoint learning. The first wave of learned detectors coupled detection and description in a single end-to-end network. Super-Point [8] bootstraps training from synthetic shapes and refines via homographic adaptation, D2-Net [9] jointly detects and describes dense feature maps, R2D2 [32] explicitly decouples repeatability and reliability, and ALIKED [46] introduces differentiable matching layers with attention-weighted local descriptors. Li et al. [23] demonstrated that decoupling detection and description mitigates the adverse influence of weak descriptors on detector accuracy, a paradigm further refined by DeDoDe [11,13], which learns detectors directly from 3D feature tracks extracted via SfM. SiLK [15] and DomainFeat [50] additionally leverage style-transfer and triplet losses to bridge domain gaps induced by illumination changes. A persistent limitation shared by all these methods is their dependence on MegaDepth [24] or similar reconstruction-derived datasets, whose phototourism bias and reliance on SIFT-bootstrapped COLMAP [41] models constrain generalization to challenging real-world conditions.

Reinforcement learning for keypoint detection. The inherently discrete nature of keypoint selection prevents direct gradient-based optimization of the detector, motivating an RL formulation. Bhowmik et al. [6] first treated the downstream matching and pose estimation pipeline as a non-diferentiable black box and computed rewards from known camera poses. DISK [45] refined this idea with per-match rewards derived from ground-truth depth, while DEAL [30] extended DISK to handle non-rigid deformations. S-TREK [36] addressed patch-boundary artifacts via sequential of-policy sampling with equivariant convolutions, and DaD [12] closed the resulting train-inference gap by aligning the sampling strategy across both stages, training a pure detector via policy-gradient repeatability objectives – though still with depth-supervised reprojection rewards. RaCo [43] trains its detector from single-image homographies, requiring no multi-view su pervision, yet still relies on the supervised ALIKED descriptor. This partial weak supervision contrasts with our approach, which trains both detector and descriptor from image pairs alone. RIPE [19] takes a diferent route: it trains jointly on detection and description using only image pairs annotated with a same-scene/diferent-scene label, exploiting the epipolar constraint via RANSACbased fundamental matrix estimation as a geometry-aware reward signal. Our work builds directly on RIPE and addresses its two key limitations: the coarse binary reward and the reliance on negative training pairs.

Feature matchers. Sparse keypoint correspondences must be established by a dedicated matcher, which itself requires training supervision. SuperGlue [38] introduced graph neural networks with self- and cross-attention to leverage geometric context from both images. LightGlue [25] replaced this with a more eficient transformer architecture with adaptive early stopping. Subsequent work explored state-space models [35], difusion-based assignment [51], and foundation-model guidance [18, 42]. While dense and semi-dense matchers such as RoMa [14], the DUSt3R family [22, 47], and LoFTR [44, 48] have pushed accuracy benchmarks, sparse matchers remain the dominant choice in production SfM and SLAM systems due to their favorable computational scaling, modularity, and compatibility with established pipelines. Crucially, all existing sparse matchers are trained using pose or depth supervision, a dependency that mirrors and compounds the data constraints on the detector side. In this work, we demonstrate that the weakly-supervised RL training paradigm can be extended to the matching step, removing this bottleneck.

## 3 Method

In this section, we present our approach for weakly supervised keypoint detection and description from image pairs, without requiring geometric ground truth such as depth, camera pose, or annotated correspondences (Sec. 3.1). Our main contribution is a reformulation of the geometric reward signal used in reinforcement-learning-based keypoint learning, resulting in a simpler training process and improved results. We additionally explore how the same learning scheme can be used to train a matcher (Sec. 3.2).

## 3.1 Weakly-supervised Keypoint Extraction

We begin with a brief recap of RIPE [19] to establish the foundation for our modifications. We then show how to train a keypoint extractor using paired images only, to reshape the reward to reflect actual match quality and how to enforce precise keypoint localization in the heatmap through entropy regularization.

Recap on RIPE RIPE [19] demonstrates how RL can be used to train a keypoint extractor using only image pairs that are labeled as either positive (showing the same scene, i.e. 2D projections of 3D points exist in both images) or negative (images are from diferent scenes). An overview is shown in Fig. 2. Given an input image $\mathbf { A } \in \mathbb { R } ^ { h \times w \times c }$ , a neural network predicts a heatmap $\mathbf { H } ^ { A } \in \mathbb { R } ^ { h \times w }$ indicating potential keypoint locations. During training, the heatmap $\mathbf { H } ^ { A }$ is divided into n quadratic cells of size q. The logit values, within a cell, define a categorical distribution, from which exactly on keypoint location is sampled. These sampled keypoint locations $\mathbf { K } ^ { A } \in \mathbb { R } _ { 2 \times n }$ are associated with initial probabilities $\hat { \mathbf { p } } \in \mathbb { R } _ { 1 \times n }$ (calculated from the distribution) and corresponding logit values $\mathbf { z } ^ { A } \in \mathbb { R } _ { 1 \times n } .$ . This formulation allows the network to shape the heatmap distribution according to the reward signal. To account for unreliable locations (sky, texture-less or overexposed regions), RIPE introduces an acceptance indicator acc ${ \bf \Sigma } ^ { A } = \mathrm { S i g m o i d } ( { \bf z } ^ { A } )$ to reject keypoint locations based on their logit value. The final probability of a keypoint is calculated as $\mathbf { p } ^ { A } = \hat { \mathbf { p } } ^ { \mathbf { A } } \odot \mathbf { a c c } ^ { \tilde { A } }$ , with $\odot$ denoting the elementwise product. Keypoint locations are then associated with their descriptors $ { \mathbf Ḋ \mathbf Ḋ A Ḍ }$ computed as hypercolumn descriptor [16] from the layers of the encoder part of the network (see [19] for details). During training, the same procedure yields acceptance indicators $\mathbf { a c c } ^ { B }$ , location ${ \bf K } ^ { B }$ and probabilities $\mathbf { p } ^ { B }$ , for a paired image B. Using REINFORCE [49], the gradient for maximizing the expected reward can be approximated as

![](images/358a34098b37203a57ef8952483b1e7f0ece6e162710f8f47e86cf1115a73619.jpg)  
Fig. 2: Schematic overview of the proposed weakly-supervised keypoint extraction framework. During training, the framework operates on pairs of images, from which a neural network generates heatmaps H from which keypoint locations K with a probability p are sampled. Descriptors D are extracted from the encoder, as Hypercolumn Features. Using these descriptors, keypoints are matched and filtered using epipolar constraints to calculate the reward matrix R, which weights the combined log-probabilities L of the keypoint locations, resulting in an approximation of the gradients, using REINFORCE [49].

$$
\nabla _ { \theta } \mathbb { E } _ { K } [ \mathbf { R } ] \approx \boldsymbol { \hat { g } } = \sum _ { \kappa } \nabla _ { \theta } \mathbf { R } ( \log \mathbf { p } ^ { A } \oplus \log \mathbf { p } ^ { B } ) ,\tag{1}
$$

with denoting the outer sum, resulting in the combined matrix of log-probabilities $\mathbf { L } \in \mathbb { R } ^ { n \times n }$ and $\kappa$ representing all possible combinations of keypoints. Note, that this formulation could also be viewed as a Contextual Bandit [21], as each image pair constitutes a context, on which the drawn actions (keypoint locations) directly depend and receive an immediate reward.

RIPE‘s key component lies in the calculation of the reward matrix $\mathbf { R } \in \mathbb { R } ^ { n \times n }$ which translates geometric consistency into a learning signal. The reward is derived from the number of detected keypoints that can be matched and are geometrically plausible. Concretely, mutual nearest-neighbors are established between descriptors, yielding a list M $\in \mathbb { R } ^ { m \times 2 }$ of index pairs for matched keypoints, with $m < < n$ . Subsequently, these candidates are filtered by a robust estimation of the fundamental matrix (assuming rigid transformations only and pinhole cameras), resulting in a list of inlier indicators $\mathbf { I } \in \mathbb { R } ^ { m }$ . The reward matrix R, is then constructed by

![](images/9d7771cd90da416dbfea46a76c7cc922115322c17f62ded53307ac8b5e9269f4.jpg)  
Fig. 3: Visual comparison of the reward matrix construction to shape the logprobabilities in L for the gradient computation (Eq. (1)). In RIPE (left) rewards are assigned at image pair level: geometric consistent matches are rewarded for positive pairs and penalized for negative sample pairs, outliers are ignored. In RIPE++ (right), rewards are defined at correspondence level within positive pairs only. Geometrically consistent matches are rewarded and geometrically inconsistent matches are explicitly penalized. This finer-grained reward makes negative pairs unnecessary and yields a more informative training signal.

$$
r _ { p , q } = \left\{ \begin{array} { l l } { \rho _ { \mathrm { i n } } , } & { ( i , j ) \in \mathbf { M } \mathrm { ~ a n d ~ } \mathbf { I } _ { i , j } \mathrm { ~ i s ~ T r u e } } \\ { \lambda , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{2}
$$

with $\rho$ being the reward and λ being a small negative penalty for not finding a suficient keypoint. For a negative image pair, the reward is inverted to penalize the network for finding geometrically plausible matches $\left( \rho _ { \mathrm { o u t } } \ : = \ : - \rho _ { \mathrm { i n } } \right)$ for a negative pair, and give a small reward (positive λ) for detecting only keypoints not resulting in plausible matches. A visualization of the resulting reward matrices is shown in Fig. 3 (left). As a result, the gradients from Eq. (1) allow the network to shape the heatmap in such a way that geometrically consistent correspondences in positive pairs have a higher probability of being selected and vice versa for the negative case.

Training From Positive Samples Only The reliance on both positive and negative pairs in RIPE is closely tied to the structure of its reward formulation. Since the reward for positive pairs is solely based on counting geometrically consistent inliers, negative pairs are required to introduce a counter-signal that penalizes such matches. This design, however, has several limitations. First, mislabeled negative pairs – images incorrectly marked as depicting diferent scenes – can mislead the detector and can destabilize training. Second, and more fundamentally, the inversion of the reward for negative pairs penalizes only the number of geometrically consistent matches. As a consequence, the network is able to produce many false matches without penalty, as long as they are filtered out by

RANSAC. In the extreme case where all keypoints are matched but subsequently rejected by robust estimation, the network receives a neutral gradient signal, resulting in a degraded training performance. We therefore reformulate the training objective to use only positive pairs, directly rewarding inlier matches and penalizing outliers.

This requires revisiting the reward formulation itself. Instead of counting inliers, we fully exploit the supervision from the robust fundamental matrix estimation and assign a reward $\rho _ { \mathrm { i n } }$ to every element $r _ { i , j }$ in R that was identified to be an inlier and a penalty ρ<sub>out</sub> to outliers. Extending Eq. (2) gives

$$
r _ { i , j } = \left\{ \begin{array} { l l } { \rho _ { \mathrm { i n } } , } & { ( i , j ) \in \mathbf { M } \mathrm { ~ a n d ~ } \mathbf { I } _ { i , j } \mathrm { ~ i s ~ T r u e } } \\ { \rho _ { \mathrm { o u t } } , } & { ( i , j ) \in \mathbf { M } \mathrm { ~ a n d ~ } \mathbf { I } _ { i , j } \mathrm { ~ i s ~ F a l s e } , } \\ { \lambda , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{3}
$$

for learning from positive samples only. Beyond simplifying supervision, this reformulated reward also enables the matcher training in Sec. 3.2.

Entropy Regularization To prevent the generation of low-probability keypoints, RIPE introduces, inspired by Potje et al. [30], a regularization term $\begin{array} { r } { \mathcal { L } _ { \mathrm { l o w } } = - \sum _ { i } \log p _ { i } \cdot \epsilon _ { : } } \end{array}$ , where ϵ is a small negative constant and $p _ { i } \in \mathbf { p }$ denotes the logit values in the i-th cell. While this regularization has been proven efective, we observe that the resulting heatmaps lack suficient spatial precision, particularly at lower input resolutions (see Sec. 4.4). Although $\mathcal { L } _ { \mathrm { l o w } }$ encourages high probability for the selected keypoint, it only implicitly shapes the surrounding probability landscape: since keypoints are sampled from a categorical distribution over each patch, a high probability for the selected location necessitates low probabilities elsewhere, but does not explicitly enforce a concentrated distribution. To address this, we replace $\mathcal { L } _ { \mathrm { l o w } }$ with the negative entropy

$$
\mathcal { L } _ { \mathrm { H } } = - \sum _ { i } \sum _ { j } \mathbf { p } _ { i , j } \log \mathbf { p } _ { i , j } ,\tag{4}
$$

where $\mathbf { p } _ { i , j }$ represents the probabilities of the individual keypoint locations in one cell. This directly encourages the distribution within each patch to approach a one-hot encoding. Unlike $\mathcal { L } _ { \mathrm { l o w } } .$ , minimizing ${ \mathcal { L } } _ { \mathrm { H } }$ produces gradients for all positions in the patch, explicitly shaping the entire distribution rather than only penalizing the selected keypoint.

Final Loss We define the loss of the detector as ${ \mathcal { L } } _ { \mathrm { d e c t } } = - \mathbb { E } _ { \kappa } [ \mathbf { R } ]$ . As the Reinforcement Learning only optimizes the locations of the keypoints, but not their descriptors, we utilize the contrastive descriptor loss ${ \mathcal { L } } _ { \mathrm { d e s c } }$ from RIPE [19] to pull the descriptors of putative matches closer and repel others. Combining these with our introduced entropy regularization results in $\mathcal { L } = \mathcal { L } _ { \mathrm { d e c t } } + \omega \mathcal { L } _ { H } + \psi \mathcal { L } _ { \mathrm { d e s c } } ,$ with ω and $\psi$ balancing the influence of the regularization and the descriptor loss respectively.

![](images/f41c504ea5339f8526d2a177c66c997d4960cafbe37c2a15d59973ffb1947bf0.jpg)  
Fig. 4: Schematic overview of the proposed weakly-supervised training framework for LightGlue. Given keypoint locations and descriptors from two images, LightGlue refines the representations through multiple layers of self- and cross-attention, incorporating positional and cross-image context. Rather than supervising with ground-truth correspondences derived from depth or pose data, we formulate matching as a reinforcement learning problem: matches deemed geometrically consistent by RANSAC receive a positive reward, while outliers are penalized. This enables end-to-end training of the matcher using only image pairs.

Further Modifications Beyond the core contributions described above, we explored several additional directions, including curriculum learning strategies and alternative descriptor loss formulations. We document these investigations in the supplementary material, as they provide useful insights for practitioners working on RL-based keypoint learning.

## 3.2 Extension to Weakly-supervised Matcher

To be competitive, modern sparse keypoint extractors rely on dedicated, trained matchers such as SuperGlue [8] or LightGlue [25]. However, these methods require pose or depth annotations for training, which contradicts our goal of learning from weak supervision via image pairs alone. As an additional experiment, we therefore explore whether the same geometric reward can also be used to train a transformer-based matcher. Concretely, we adapt LightGlue to enable training using only geometric consistency as a reward signal, drawing on the policy gradient formulation of DISK [45]. Fig. 4 illustrates our approach.

Recap on LightGlue Given two sets of local features from images A and B, each with normalized 2D positions $\mathbf { p } _ { i }$ and descriptors $\mathbf { d } _ { i } \in \mathbb { R } ^ { d }$ , LightGlue refines point representations through L stacks of self- and cross-attention layers. A lightweight head then computes pairwise scores

$$
S _ { i , j } = \mathrm { L i n e a r } \left( \mathbf { x } _ { i } ^ { A } \right) ^ { \top } \mathrm { L i n e a r } \left( \mathbf { x } _ { j } ^ { B } \right) ,\tag{5}
$$

and per-point matchability scores $\sigma _ { i } = { \mathrm { S i g m o i d } } \left( { \mathrm { L i n e a r } } \left( \mathbf { x } _ { i } \right) \right) \in \left[ 0 , 1 \right]$ , which are combined into a soft partial assignment matrix

$$
P _ { i , j } = \sigma _ { i } ^ { A } \sigma _ { j } ^ { B } \operatorname { S o f t m a x } _ { k \in \mathcal { A } } ( \mathbf { S } _ { k , j } ) _ { i } \operatorname { S o f t m a x } _ { k \in \mathcal { B } } ( \mathbf { S } _ { i , k } ) _ { j } .\tag{6}
$$

Weakly-supervised LightGlue The original LightGlue loss maximizes the log-likelihood of ground-truth assignments, requiring known correspondences from pose or depth. To avoid this supervision and allow training from image pairs only, we reformulate the optimization using a reward-driven objective. Inspired by DISK [45], we develop a policy gradient formulation, interpreting matching as a decision process and optimize the expected geometric reward instead of a supervised assignment loss. Concretely, we decompose Eq. (6) into the probability of a match,

$$
P ( i , j \mid A , B ) = \operatorname { S o f t m a x } ( \mathbf { S } _ { k , j } ) _ { i } \operatorname { S o f t m a x } ( \mathbf { S } _ { i , k } ) _ { j } ,\tag{7}
$$

and

$$
P _ { i } ( \mathbf { x } _ { i } \mid I ) = { \mathrm { S i g m o i d } } \left( \operatorname { L i n e a r } \left( \mathbf { x } _ { i } \right) \right) ,\tag{8}
$$

the probability of accepting keypoint i as matchable. The expected reward gradient then takes the form

$$
\nabla _ { \theta } \underset { A  B } { \mathbb { E } } R ( M _ { A  B } ) = \underset { F _ { A } , F _ { B } } { \mathbb { E } } \sum _ { i , j } P ( i , j \mid A , B ) \cdot r ( i , j ) \cdot \nabla _ { \theta } r _ { i j } ,\tag{9}
$$

where $r ( i  j )$ is a per-match reward based on geometric consistency and $\begin{array} { r } { T _ { i j } = \log P ( i  j \mid A , B ) + \log P _ { i } ( \mathbf { x } _ { i } \mid A ) + \log P _ { j } ( \mathbf { x } _ { j } \mid B ) } \end{array}$ . Since all match probabilities can be computed in closed form, no sampling is required and the expectation is exact up to the empirical approximation over feature sets $F _ { A } , F _ { B }$ . We provide a detailed derivation of the connection between the DISK and LightGlue formulations in the supplementary material.

The reward signal $r ( i , j )$ is derived from the geometric consistency of the predicted correspondences. To this end, we estimate the fundamental matrix using a robust estimator and classify the resulting correspondences as inliers or outliers. Inliers receive a reward $\nu _ { \mathrm { i n } } ,$ , while outliers receive a penalty $\nu _ { \mathrm { o u t } }$ . Since this reward signal only applies to matched keypoints, the network could trivially minimize the loss by predicting all keypoints as non-matchable. To avoid this degenerate solution, we add a small penalty λ for each keypoint classified as unmatchable.

To prevent the network from trivially labeling every keypoint as not matchable according to Eq. (8) we introduce a regularization term $\begin{array} { r } { \mathcal { L } _ { \mathrm { n m } } = \sum _ { i } P _ { i } ( \mathbf { x } _ { i } | A ) + } \end{array}$ $\begin{array} { r } { \sum _ { i } P _ { i } ( \mathbf { x } _ { i } | B ) } \end{array}$ which corresponds to the expected number of keypoints the model classifies as non-matchable. We define the loss of the matcher as $\mathcal { L } _ { \mathrm { m a t c h } } ~ { = }$ $- \mathbb { E } _ { M _ { A  B } } [ R ( M _ { A  B } ) ]$ and the final training loss as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { m a t c h } } + \eta \mathcal { L } _ { \mathrm { n m } } , } \end{array}\tag{10}
$$

with η controlling the strength of the non-matchable regularization term.

## 4 Experiments

Implementation Details To ensure a fair comparison with RIPE, we use the same torchvision VGG-19 backbone combined with the depthwise convolutional refiners proposed by Edstedt et al. [10]. We use the GPU-accelerated implementation of nearest-neighbor matching from Kornia [33]. For robust fundamental matrix estimation, we replace PoseLib [20] with the $\mathrm { U S A C } _ { - }$ magsac RANSAC variant from OpenCV [7], which achieves comparable accuracy at lower computational cost, reducing overall training time. The keypoint extractor is trained with AdamW [26] using a learning rate that decays linearly from $1 0 ^ { - 3 }$ to 10<sup>−6</sup> over 80k steps. We use a batch size of 6 with gradient accumulation over 4 batches, and no data augmentation beyond normalization, resizing the longer side to 560 pixels and padding to preserve the aspect ratio. As usual, the cell size is set to 8 pixels. Training takes 26 hours on a single A100, compared to 72 hours for RIPE – another benefit of using positive samples only, since RANSAC can exit early for positive samples but must run the maximum iterations for negative ones. Please refer to the supplementary for the remaining hyperparameters.

Training data To enable a controlled comparison with existing methods and to isolate the efects of our training scheme, we train on the MegaDepth dataset, using the subset proposed by DISK [45] for the keypoint extractor and the subset proposed by LightGlue [25] for the matcher. Since our method requires only image pairs without depth or pose annotations, it naturally generalizes to domains where such supervision is notoriously dificult to acquire; we demonstrate this by training on endoscopic video streams in Sec. 4.2.

Inference While training operates on image pairs, at inference the keypoint extractor processes a single image: the top-k keypoints are selected from the heatmap H based on their score, after applying non-maximum suppression in a $3 \times 3$ window and subpixel refinement following ALIKED [53], as also adopted by RaCo [43] and DaD [12].

## 4.1 Relative Pose Estimation

Dataset We evaluate relative pose estimation on the MegaDepth-1500 benchmark introduced by LoFTR [44], which comprises two scenes (Brandenburger Tor and St. Peters Square) from the 196-scene MegaDepth dataset. The benchmark poses challenges through large viewpoint changes, varying illumination, and repetitive structures. We resize the longer image side to 1200 pixels and to isolate the quality of the extracted keypoints, all sparse methods are evaluated using mutual-nearest-neighbor matching.

Metrics Following standard practice, match quality is measured indirectly through the accuracy of the resulting relative poses. The pose error is defined as the maximum of the angular errors in rotation and translation. We report the Area Under the Curve (AUC) of the pose error at thresholds of 5°, 10°, and 20°. All experiments use the top 2048 keypoints and robust pose estimation via PoseLib [20], implemented within the glue-factory framework [25, 29].

Table 1: Relative pose estimation on MegaDepth1500. Methods are grouped by whether training requires geometric ground truth (pose, depth) for the detector or descriptor. RIPE++ trains both detector and descriptor from positive image pairs alone, without geometric supervision, yet remains comparable to geometric ground truth group and outperforms RIPE despite discarding its negative pairs. Best and second-best are highlighted within each group.
<table><tr><td></td><td>Method</td><td>AUC@5°</td><td>AUC@10°</td><td>AUC@20°</td><td>Supervision</td></tr><tr><td></td><td>ALIKED [53]TIM&#x27;23</td><td>56.66</td><td>69.64</td><td>79.35</td><td>Pose+Homog.</td></tr><tr><td></td><td>DaD [12]CoRR&#x27;25</td><td>56.46</td><td>70.07</td><td>80.18</td><td>Depth</td></tr><tr><td></td><td>DeDoDe-B [13]CVPRW&#x27;24</td><td>56.01</td><td>69.20</td><td>78.72</td><td>Pose</td></tr><tr><td>Geo rG</td><td>DISK [45]NeurIPS&#x27;20</td><td>50.69</td><td>64.35</td><td>74.74</td><td>Pose/ Depth</td></tr><tr><td></td><td>RACO [43]3DV&#x27;26</td><td>57.96</td><td>71.28</td><td>81.00</td><td>Homog. (Det.) + Pose (Desc.)</td></tr><tr><td>G</td><td>SuperPoint [8]CVPRW&#x27;18</td><td>47.26</td><td>60.89</td><td>70.65</td><td>Homography</td></tr><tr><td></td><td>SIFT [27]1JCV’04</td><td>38.97</td><td>53.21</td><td>64.99</td><td></td></tr><tr><td>e.</td><td>RIPE [19]ICCV’25</td><td>53.47</td><td>67.02</td><td>77.62</td><td>Pos. + Neg. Pairs</td></tr><tr><td>ON</td><td>RIPE++</td><td>56.58</td><td>69.53</td><td>79.33</td><td>Pos. Pairs</td></tr></table>

![](images/22c45151bcba3b2b788a8fbab3ab2f04e2b1ce17b22d28568b09ffc2b81a96d0.jpg)  
Fig. 5: Qualitative results of raw matches for RIPE++ and RIPE++ in combination with our weakly-supervised LightGlue matcher on the MegaDepth1500 benchmark dataset on the left and right respectively. The middle image shows qualitative results for RIPE++ Medical on the SCARED1500 test dataset. For more qualitative results please refer to the supplementary material.

Baselines We compare RIPE++ against SotA methods for sparse keypoint extraction. As RaCo and DaD are only keypoint detectors, we combine both with ALIKED-n16 descriptors for evaluation. For the all methods, we use mutualnearest-neighbor matching to establish correspondences, and a RANSAC inlier threshold of 0.5.

Results Tab. 1 groups methods by whether their training requires geometric ground truth. Among methods that use none, RIPE++ is the strongest by a clear margin, improving over RIPE by 3.11 pp AUC@5° while discarding the negative pairs RIPE depends on. More notably, this parity extends across the supervision boundary: RIPE++ matches the strongly-supervised extractors, trailing ALIKED by only 0.07 pp on average and outperforming DeDoDe [13] and DaD [12], both of which use separate, fully-supervised detection and description networks. RaCo [43] reports higher absolute numbers, but only its detector is trained without pose supervision; its descriptor is taken from the supervised ALIKED. RIPE++, by contrast, trains both detector and descriptor from image pairs alone, without artificial homographies, pose annotations, or depth from pre-registered 3D models. Qualitative results are provided in Fig. 5 and the supplementary material.

Table 2: Evaluation of the relative pose estimation on Scared 1500, showing the benefits of our proposed training scheme. It allows to easily train a specialized keypoint extractor, simply from video frames. The best and second-best performances are highlighted.
<table><tr><td>Method</td><td>AUC@5°</td><td>AUC@10°</td><td>AUC@20°</td><td>Supervision</td></tr><tr><td>ALIKED [53]TIM&#x27;23</td><td>18.24</td><td>40.61</td><td>61.23</td><td>Pose+Homography</td></tr><tr><td>DaD [12]CoRR&#x27;25</td><td>18.44</td><td>40.56</td><td>61.40</td><td>Depth</td></tr><tr><td>DeDoDe-B [13]CVPRW&#x27;24</td><td>18.85</td><td>41.69</td><td>62.80</td><td>Pose</td></tr><tr><td>DISK [45]NeurIPS’20</td><td>16.58</td><td>36.77</td><td>55.63</td><td>Pose/ Depth</td></tr><tr><td>RACO [43]3DV’26</td><td>19.35</td><td>42.36</td><td>63.34</td><td>Homography</td></tr><tr><td>SIFT [27]1JCV’04</td><td>13.24</td><td>32.17</td><td>51.66</td><td></td></tr><tr><td>SuperPoint [8]CVPRW&#x27;18</td><td>19.01</td><td>42.71</td><td>64.54</td><td>Homography</td></tr><tr><td>RIPE [19]ICCV&#x27;25</td><td>17.43</td><td>38.63</td><td>58.70</td><td>Pos. + Neg. Pairs</td></tr><tr><td>RIPE++</td><td>17.53</td><td>38.09</td><td>57.86</td><td>Pos. Pairs</td></tr><tr><td>RIPE++ Medical</td><td>20.90</td><td>46.51</td><td>68.72</td><td>Video Data</td></tr></table>

Table 3: Ablation study of our proposed methods on the MegaDepth1500 test set.
<table><tr><td colspan="2">Parameter</td><td colspan="5">Metrics</td></tr><tr><td>pos only</td><td>entropy</td><td># RANSAC inlier</td><td>% RANSAC inlier</td><td>AUC@5°</td><td>AUC@10°</td><td>AUC@20°</td></tr><tr><td></td><td></td><td>297.5</td><td>34.4</td><td>51.83</td><td>65.37</td><td>75.94</td></tr><tr><td>√</td><td></td><td>427.5</td><td>41.5</td><td>52.42</td><td>66.78</td><td>78.03</td></tr><tr><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>√</td><td> $\begin{array} { l } { 1 \times 1 0 ^ { - 4 } } \\ { 1 \times 1 0 ^ { - 5 } } \end{array}$ </td><td>266</td><td>33.6</td><td>49.62</td><td>62.7</td><td>72.8</td></tr><tr><td>√</td><td> $1 \times { 1 0 } ^ { - 6 }$ </td><td>366.5</td><td>40.6</td><td>56.59</td><td>69.44</td><td>79.18</td></tr></table>

## 4.2 SCARED1500

The weak supervision of RIPE++ enables training keypoint extractors in domains where ground-truth pose or depth information is dificult to obtain, such as for instance endoscopic imaging. We demonstrate this by training directly from endoscopic camera streams, which are readily available in large quantities.

Dataset We introduce SCARED1500, a benchmark derived from the SCARED dataset [2], recorded with a da Vinci Xi surgical robot with accurate camera poses, which we only use for our evaluation. The dataset contains 7 training and 2 test scenes of porcine subjects, each with 4–5 keyframes representing unique viewpoints with structured-light depth. Test pairs are formed by randomly sampling image pairs within the same keyframe sequence and retaining only those with 40–90% visual overlap, estimated by projecting 3D points between frames. Images are undistorted using the provided calibration and resized to 1200 pixels on the longer side at inference. Training pairs for RIPE++ are formed by frames separated by 60 frames, yielding 17 514 pairs, augmented with random afine transformations (in-plane rot. within ±30°, transl. up to 10 % of image size, and isotropic scaling in [0.9,1.1]) to mitigate overfitting. While this simple pairing strategy can produce degenerate pairs with negligible motion or no overlap, we find the model robust to such noise given a reasonable frame distance.

![](images/07e73a8390cda40c725dc014311d5246ccd653f1a319b9a66bb199a01969f860.jpg)  
Fig. 6: Efect of our entropy-based regularization on keypoint localization. Without it, heatmaps tend to become difuse, particularly at low input resolutions (top-left). Regularization encourages sharper, more peaked responses (bottom).

Metrics We integrate SCARED1500 into the glue-factory framework to leverage its evaluation pipeline. As in Sec. 4.1, we evaluate relative pose estimation using the same metrics, with a RANSAC threshold of 1.0.

Results The value of weak supervision is clearest under domain shift (Tab. 2). Retrained on endoscopic video, RIPE++ Medical outperforms all baselines across every threshold, adapting both detector and descriptor from raw frames without pose or depth. This adaptability is the operative advantage: zero-shot, every learned method collapses on the unseen domain, and RIPE++ itself only matches RIPE, since its positive-only reward specializes to the training distribution rather than generalizing across domains. What distinguishes RIPE++ is not zero-shot transfer but the ability to retrain cheaply wherever geometric ground truth is unavailable, which no pose- or depth-supervised method can do here. RaCo [43] could in principle fine-tune its homography-trained detector, but its supervised ALIKED descriptor cannot follow, leaving the adaptation incomplete. Qualitative results are provided in Fig. 5 and the supplementary material.

## 4.3 RIPE++ with Learned Matcher

To demonstrate the versatility of our RL-based training approach, we trained LightGlue as dedicated matcher on top of our learned RIPE++ extractor, by extending the oficial LightGlue [25] implementation. We integrated our proposed training method, but retained the original two-stage training protocol: pretraining on synthetic image pairs generated from the distractor dataset [31], followed by fine-tuning on MegaDepth using the oficial data split. Since the inlier-to-outlier ratio during matcher training is more balanced than during extraction, we use rewards and penalties of equal magnitude. Tab. 4 shows the clear improvements gained for the MegaDepth1500 test set. While the absolute pose accuracy remains below that of LightGlue variants trained with full pose and depth supervision (e.g., 66.1% AUC@5<sup>◦</sup> when paired with ALIKED), our matcher is trained entirely from image pairs without any geometric ground truth, making the average improvement 3.9 pp a promising indication that RL-based weak supervision can transfer efectively to the matching stage.

Table 4: Improvements on MegaDepth1500 from training LightGlue with our proposed approach on positive image pairs only.
<table><tr><td>Method</td><td>AUC@5°</td><td>AUC@10°</td><td>AUC@20°</td></tr><tr><td>RIPE++</td><td>56.58</td><td>69.53</td><td>79.33</td></tr><tr><td>↓ + LightGlue</td><td>+3.07</td><td>+4.16</td><td>+4.53</td></tr><tr><td>Matched-RIPE++</td><td>59.65</td><td>73.69</td><td>83.86</td></tr></table>

## 4.4 Ablations

To ablate the influence of our proposed methods, we evaluated our diferent configurations on the MegaDepth1500 test set with the configurations described in Sec. 4.1. Table 3 shows that our improved reward calculation purely based on positive pairs not only simplifies the data generation, but also improves over training from positive and negative pairs. Adding our regularization to ensure a precise localization of keypoints improves the AUC@5<sup>◦</sup> by over 4 pp. But it requires careful weighting; otherwise, it causes a degraded performance (ω = 1e  5) or even a collapsed training (ω = 1e  4). Yet, carefully tuned, it improves the quality of the heatmap, especially for low-resolution inputs, as can be seen in Fig. 6. Please refer to the supplementary for additional ablations.

## 5 Conclusion

We presented a novel geometric reward for reinforcement learning–based training of sparse keypoint extractors under weak supervision. Importantly, this formula tion fully exploits geometric consistency within a single image pair, eliminating the need for negative training pairs and further reducing the data requirements for RL-based keypoint learning. Our experiments demonstrate that the resulting RIPE++ extractor achieves competitive accuracy while relying on substantially weaker supervision than competing approaches. The simplified training setup also enables learning directly from video sequences. In particular, we show that training on endoscopic video streams is suficient to obtain a specialized keypoint extractor for the medical domain, where geometric ground truth is often unavailable.

Beyond keypoint extraction, the positive-pair training paradigm also allows the construction of a reward signal for the matching stage. Together, these results highlight the potential of reinforcement learning for jointly training keypoint extractors and matchers under minimal supervision.

## Acknowledgements

This work was partly funded by German Federal Ministry for Economic Afairs and Climate Action (DeepTrain, grant no. 19S23005D).

# Supplementary Material for RIPE++: Reinforced Keypoint Learning from Positive Pairs Only

## 6 Hyperparameters

Table 5 documents our hyperparameters for the training of the keypoint extractor (left side) and the keypoint matcher (right side).

Table 5: Overview of training hyperparameters for the keypoint extractor (left) and the matcher (right).
<table><tr><td colspan="3">Extractor</td><td colspan="3">Matcher</td></tr><tr><td>Name</td><td>Value</td><td>Purpose</td><td>Name</td><td>Value</td><td>Purpose</td></tr><tr><td> $\rho _ { \mathrm { i n } }$ </td><td>1.0</td><td>reward Eq. (3)</td><td> $\nu _ { \mathrm { i n } }$ </td><td>1.0</td><td>reward matcher training</td></tr><tr><td> $\rho _ { \mathrm { o u t } }$ </td><td>-0.1</td><td>penalty Eq. (3)</td><td> $\nu _ { \mathrm { o u t } }$ </td><td>-1.0</td><td>penalty matcher training</td></tr><tr><td>λ</td><td>-1e-7</td><td>penalty Eq. (3)</td><td>λ</td><td>-1e-07</td><td>penalty no match</td></tr><tr><td>ψ</td><td>5.0</td><td>weight  $\mathcal { L } _ { \mathrm { d e s c } }$ </td><td>η</td><td>0.0001</td><td>weight  $\dot { \mathcal { L } } _ { \mathrm { n m } }$  Eq. (10)</td></tr><tr><td>ω</td><td>1e-6</td><td>weight  $\mathcal { L } _ { H }$ </td><td></td><td></td><td></td></tr></table>

## 7 Additional Architectural Experiments

## 7.1 Methods

Beyond our core contributions, we explored several additional directions. We document these investigations in the following, as they provide useful insights for practitioners working on RL-based keypoint learning.

Distance-based Reward For further improvement of the reward signal, we replace the binary inlier/outlier reward with a distance-aware reward signal based on the Sampson distance [17], measuring how well a correspondence satisfies the epipolar constraint. Given a putative match for the two keypoint locations $\mathbf { l } ^ { A } \in \mathbf { K } _ { A }$ and $\mathbf { l } _ { B } \in \mathbf { K } ^ { B }$ and the estimated fundamental matrix F, the Sampson distance is defined as:

$$
d _ { S } ( \mathbf { l } _ { A } , \mathbf { l } _ { B } , \mathbf { F } ) = { \frac { ( \mathbf { l } _ { B } \mathrm { { ^ T F l } } _ { A } ) ^ { 2 } } { ( \mathbf { F l } _ { A } ) _ { 1 } ^ { 2 } + ( \mathbf { F l } _ { A } ) _ { 2 } ^ { 2 } + ( \mathbf { F } ^ { \mathsf { T } } \mathbf { l } _ { B } ) _ { 1 } ^ { 2 } + ( \mathbf { F } ^ { \mathsf { T } } \mathbf { l } _ { B } ) _ { 2 } ^ { 2 } } } ,\tag{11}
$$

where $( \cdot ) _ { i }$ denotes the i-th component of a vector. We convert this distance into a reward using a truncated $L _ { 2 }$ kernel:

$$
r _ { i , j } ( d _ { S } ) = \left\{ \begin{array} { l l } { 1 - \frac { 1 } { 2 } \left( \frac { d _ { S } } { \delta } \right) ^ { 2 } , } & { \mathrm { i f ~ } d _ { S } \leq \tau } \\ { \rho _ { \mathrm { o u t } } , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{12}
$$

where δ controls the curvature of the reward fallof, τ is the outlier threshold, and $\rho _ { \mathrm { o u t } }$ is again the penalty for outliers.

This formulation provides a smooth gradient for correspondences near the epipolar constraint, rather than treating all inliers equally, while explicitly penalizing geometric outliers. Matches with zero Sampson distance receive maximal reward $( r = 1 )$ , with the reward decreasing quadratically as the distance increases, until the threshold τ is exceeded. For the given values in Fig. 7 and a RANSAC threshold of 1.5, inliers with $d _ { s } > 1 . 4 1$ already receive a (small) penalty, as they are deemed to be too far away from their respective epipolar line. Outliers with $1 . 5 > d _ { s } < 2 . 0$ receive an increasing penalty, as they were almost labeled inliers. On the other hand, outliers with $d _ { s } \geq 2 . 0$ receive a smaller constant penalty, as they are already suficiently distant from deemed as inliers.

By replacing hard inlier decisions with a continuous geometric signal, the reward better reflects matching quality and improves gradient stability during training.

Curriculum Learning Training keypoint detectors with weak supervision presents a challenge: early in training, the network produces many lowquality matches, resulting in noisy reward signals that can destabilize learning. Following [4], we address this with a curriculum learning strategy that gradually increases training dificulty.

![](images/38897d5c95a6fd987e3105e0a4c89df61cb977b22861874d2be20a20b9c0bb2f.jpg)  
Fig. 7: Illustration Distance-based Reward

For each batch, we compute the reward $r _ { i }$ for each sample i and rank samples by their reward in descending order. Only the top-k% of samples contribute to the loss; the remaining samples are masked out. This ensures that the network initially learns from high-confidence matches where the reward signal is reliable.

The curriculum schedule follows a linear annealing:

$$
k ( t ) = \operatorname* { m i n } \left( k _ { \operatorname* { m a x } } , k _ { \operatorname* { i n i t } } + \left\lfloor \frac { t } { T _ { \operatorname* { i n c } } } \right\rfloor \cdot \varDelta k \right) ,\tag{13}
$$

where t is the current training step, $k _ { \mathrm { i n i t } }$ is the initial percentage of samples used, $\varDelta k$ is the increment step, $T _ { \mathrm { i n c } }$ is the number of steps between increments, and $k _ { \mathrm { m a x } }$ is the maximum percentage.

In practice, we set $k _ { \mathrm { i n i t } } = 3 0 \% , \varDelta k = 5 \% , T _ { \mathrm { i n c } } = 1 0 0 0$ , and $k _ { \mathrm { m a x } } = 8 0 \%$ . This schedule allows the network to first establish reliable keypoint detection on easy samples before gradually incorporating harder cases with noisier supervision. The upper bound $k _ { \mathrm { m a x } } < 1 0 0 \%$ ensures that the most dificult samples – which may contain erroneous rewards due to missing or too small visual overlap – never dominate the training signal.

Table 6: Ablation of the diferent components on the MegaDepth1500 benchmark set with the longer side of the input rescaled to 1200 pixels and 2048 keypoints detected.
<table><tr><td colspan="6">Parameter</td><td colspan="5">Metrics</td></tr><tr><td></td><td></td><td>diste raard</td><td></td><td></td><td>RASNAC#</td><td>AAA</td><td></td><td></td><td></td><td></td></tr><tr><td> soy</td><td>contive</td><td>nufonce</td><td></td><td>curuuum</td><td>enttropy</td><td>lier</td><td>nlier</td><td>AUCC</td><td>AU110°</td><td>AU20°</td></tr><tr><td></td><td>ノノV</td><td></td><td></td><td></td><td></td><td>297</td><td>34</td><td>51.83</td><td>65.37</td><td>75.94</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>427</td><td>41</td><td>52.42</td><td>66.78</td><td>78.03</td></tr><tr><td></td><td></td><td></td><td>√</td><td></td><td></td><td>384</td><td>40</td><td>54.75</td><td>68.33</td><td>78.92</td></tr><tr><td></td><td></td><td></td><td></td><td>√</td><td></td><td>347</td><td>38</td><td>52.57</td><td>65.84</td><td>76.43</td></tr><tr><td></td><td></td><td>ン&gt;</td><td></td><td></td><td></td><td>430</td><td>45</td><td>51.72</td><td>65.0</td><td>75.9</td></tr><tr><td></td><td></td><td></td><td>√</td><td></td><td></td><td>395</td><td>43</td><td>50.02</td><td>63.66</td><td>74.66</td></tr><tr><td>√√</td><td></td><td></td><td></td><td>V√</td><td></td><td>370</td><td>41</td><td>49.18</td><td>61.93</td><td>72.85</td></tr><tr><td></td><td></td><td></td><td>√</td><td></td><td></td><td>362</td><td>40</td><td>51.04</td><td>64.29</td><td>74.6</td></tr><tr><td></td><td>√</td><td></td><td>√</td><td></td><td> $1 \times { 1 0 } ^ { - 6 }$ </td><td>374</td><td>40</td><td>67.12</td><td>78.01</td><td>53.01</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td></td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>366</td><td>41</td><td>69.44</td><td>79.18</td><td>56.59</td></tr></table>

Improved Descriptor Loss RIPE [19] used a hinge loss as auxiliary training signal to optimize the descriptors with

$$
L _ { \mathrm { d e s c } } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m a x } ( 0 , \mu + \delta _ { + } ^ { n } - \delta _ { h } ^ { n } ) , } & { \lambda _ { \kappa } = 1 } \\ { \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m a x } ( 0 , \mu - \delta _ { + } ^ { n } ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{14}
$$

where N is the number of inliers after the RANSAC filtering, µ is a positive threshold, and δ the L2-distance between two descriptors (δ<sup>n</sup> being the L2- distance between two keypoint descriptors which got matched and validated $( { \mathrm { i . e . } }$ identified as inliers) by geometric filtering while $\delta _ { h } ^ { n }$ is the distance to the second closest neighbor). In summary, the loss pulls putative matches closer in features space and repells matched descriptors for negative pairs, as their should be no matches ideally.

Despite having been proved efective the major drawback lies in the sparsity of the loss, as only the nearest and/ or the second nearest neighbor are taken into account.

To this end we draw inspiration from self-supervised works and aim to replace Eq. (14) with the InfoNCE [28] loss.

For two images A and P sets of descriptors $\mathbf { D } _ { A } \in \mathbb { R } _ { d \times N }$ and $\mathbf { D } _ { P } \in \mathbb { R } _ { d \times M }$ are extracted and L2-normalized. We then calculate the similarity for every combination of descriptors with

$$
\mathbf { S } = \frac { 1 } { \tau } \mathbf { D } _ { A } ^ { \phantom { } T } \mathbf { D } _ { P } \in \mathbb { R } ^ { N \times M }\tag{15}
$$

with $\tau$ being a temperature hyperparameter that controls the sharpness of the resulting distribution. Applying softmax on each row of S with

$$
p _ { i j } = \frac { e ^ { s _ { i j } } } { \sum _ { k = i } ^ { M } e ^ { s _ { i k } } } ,\tag{16}
$$

efectively results in the probabilities of some descriptor $d _ { i } ^ { a }$ matching some descriptor $d _ { j } ^ { b }$

With $k ^ { i }$ giving the putative correct match (from NN-matching and RANSAC filtering, not groundtruth) of descriptor $d _ { i }$ we formulate the final cross-entropy loss

$$
L _ { i } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log p _ { i , k ^ { i } }\tag{17}
$$

efectively enforcing the probability mass to concentrate on the correct match.

Evaluation The ablation in Tab. 6 confirms that training on positive pairs only consistently improves results. Replacing the descriptor loss, however, does not: while the number of RANSAC inliers increases notably, this does not translate into improved pose accuracy. Curriculum learning, entropy regularization, and the distance-based reward each individually improve results, though their combination does not yield further cumulative gains. The best overall configuration uses positive-only training, entropy regularization, and the contrastive descriptor loss.

## 8 Influence of Removing Negative Samples

While removing the negative pairs from RIPE++’s training proved beneficial in our evaluations, we examine whether a model trained without negative pairs produces more spurious correspondences on geometrically unrelated image pairs. To this end, we sample 1500 negative pairs from the MegaDepth-1500 test set by pairing images from two diferent scenes, which therefore share no common underlying 3D geometry. We extract keypoints and match them using the same settings as in Sec. 4.1, and visualize in Fig. 8 the histograms of false nearest-neighbor matches and of RANSAC inliers after filtering with the fundamental matrix. The results clearly show that the absence of negative pairs does not increase the number of false matches; in fact, it decreases them by 8.4 %, indicating that the negative pairs are not necessary to suppress spurious correspondences.

## 9 Derivation of Weakly-Supervised LightGlue Training

We provide a detailed derivation of the connection between the DISK [45] and LightGlue [25] formulations.

![](images/581c4172e580fd1363d3edf1843aa7c15b6e4e05bae9ff26388783b5d9d44c4d.jpg)

![](images/98b5e77ff303a38c25bdd7689eb28c3ba607eca6510b635f1ab5b065d8efedc1.jpg)  
Fig. 8: Number of false NN matches (left) and RANSAC inliers (right) for negative image pairs.

Background: LightGlue Given two sets of local features from images A and B, each with normalized 2D positions $\mathbf { p } _ { i }$ and descriptors $\mathbf { d } _ { i } \in \mathbb { R } ^ { d }$ , LightGlue processes point representations through L stacked self- and cross-attention layers. After each layer, a classifier determines whether inference can be halted early to reduce computation. A lightweight head then computes pairwise similarity scores

$$
S _ { i , j } = \mathrm { L i n e a r } \left( \mathbf { x } _ { i } ^ { A } \right) ^ { \top } \mathrm { L i n e a r } \left( \mathbf { x } _ { j } ^ { B } \right) ,\tag{18}
$$

and per-point matchability scores

$$
\sigma _ { i } = { \mathrm { S i g m o i d } } \left( \operatorname { L i n e a r } \left( \mathbf { x } _ { i } \right) \right) \in [ 0 , 1 ] ,\tag{19}
$$

encoding the likelihood that point i has a corresponding point in the other image. Both terms are combined into a soft partial assignment matrix

$$
P _ { i , j } = \sigma _ { i } ^ { A } \sigma _ { j } ^ { B } \operatorname { S o f t m a x } _ { k \in \mathcal { A } } ( \mathbf { S } _ { k , j } ) _ { i } \operatorname { S o f t m a x } _ { k \in \mathcal { B } } ( \mathbf { S } _ { i , k } ) _ { j } .\tag{20}
$$

The network is trained by minimizing the negative log-likelihood of ground-truth assignments at each layer ℓ:

$$
\begin{array} { l } { \displaystyle \mathcal { L } = - \frac { 1 } { L } \sum _ { \ell } \Bigg ( \frac { 1 } { | \mathcal { M } | } \sum _ { ( i , j ) \in \mathcal { M } } \log \ell P _ { i , j } } \\ { \displaystyle \qquad + \frac { 1 } { 2 | \bar { \mathcal { M } } | } \sum _ { i \in \bar { \mathcal { A } } } \log \left( 1 - \ell \sigma _ { i } ^ { A } \right) } \\ { \displaystyle \qquad + \frac { 1 } { 2 | \bar { \mathcal { B } } | } \sum _ { j \in \bar { \mathcal { B } } } \log \left( 1 - \ell \sigma _ { j } ^ { B } \right) \Bigg ) , } \end{array}\tag{21}
$$

where $\mathcal { M }$ denotes ground-truth matches, and ${ \bar { \mathcal { A } } } \subseteq { \mathcal { A } } , { \bar { \mathcal { B } } } \subseteq B$ are the unmatched points in each image.

Background: DISK Policy Gradient DISK [45] formulates keypoint detection and matching as a reinforcement learning problem, optimizing the expected reward over sampled matches rather than a supervised loss. A keypoint at location $p$ within patch u is selected and accepted with probability

$$
P ( p \mid \mathbf { K } ^ { u } ) = \operatorname { S o f t m a x } ( \mathbf { K } ^ { u } ) _ { p } \cdot \operatorname { S i g m o i d } ( \mathbf { K } _ { p } ^ { u } ) ,\tag{22}
$$

where $\mathbf { K } _ { p } ^ { u }$ is the detection heatmap logit at location $p .$ Match probabilities are derived from the L2-distance matrix d between descriptors, where rows and columns define categorical distributions for the forward $( A \to B )$ and reverse $( B \to A )$ directions:

$$
\begin{array} { r } { P _ { A \to B } ( j \mid { \bf d } , i ) = \mathrm { S o f t m a x } ( - \theta _ { M } { \bf d } ( i , \cdot ) ) _ { j } , } \end{array}\tag{23}
$$

with $\theta _ { M }$ the inverse softmax temperature. The negative sign converts L2 distance into afinity, analogous to how LightGlue uses the dot product directly (where zero denotes perpendicular, rather than identical, descriptors).

The policy gradient then takes the form

$$
\begin{array} { r l } { \nabla _ { \theta } \underset { M _ { A  B } } { \mathbb { E } } R ( M _ { A  B } ) = \underset { F _ { A } , F _ { B } } { \mathbb { E } } \sum _ { i , j } \Big [ P ( i , j \mid F _ { A } , F _ { B } ) \cdot r ( i , j ) \cdot \nabla _ { \theta } T _ { i j } \Big ] , ~ } & { } \\ { T _ { i j } = \log P ( i , j \mid F _ { A } , F _ { B } ) + \log P ( F _ { A , i } \mid A ) + \log P ( F _ { B , j } \mid B ) , ~ } & { } \end{array}\tag{24}
$$

where the sum runs only over matched pairs, since unmatched keypoints receive zero reward. Crucially, since all match probabilities $P ( i , j )$ can be computed in closed form, the inner expectation is evaluated exactly – no sampling is required and no variance is introduced at this level. Variance arises only from the empirical approximation of the outer expectation over feature sets $F _ { A } , F _ { B }$

Adapting DISK’s Policy Gradient to LightGlue The structural parallel between DISK and LightGlue is direct: both decompose a soft assignment into a matchability/ keypointness term and a match term. We therefore substitute DISK’s keypoint selection probability (Eq. (22)) with LightGlue’s matchability score (Eq. (19)):

$$
P _ { i } ( \mathbf { x } _ { i } \mid I ) = { \mathrm { S i g m o i d } } \left( \operatorname { L i n e a r } \left( \mathbf { x } _ { i } \right) \right) ,\tag{25}
$$

and replace the descriptor-distance-based match probability with LightGlue’s attention-based formulation:

$$
P ( i , j \mid A , B ) = \operatorname { S o f t m a x } ( \mathbf { S } _ { k , j } ) _ { i } \operatorname { S o f t m a x } ( \mathbf { S } _ { i , k } ) _ { j } .\tag{26}
$$

Substituting into Eq. (24) yields the gradient estimator used in our training:

$$
\begin{array} { r l } { \nabla _ { \theta } \underset { M 4  B } { \mathbb { E } } R ( M _ { A  B } ) = \underset { F _ { A } , F _ { B } } { \mathbb { E } } \sum _ { i , j } \bigg [ P ( i , j \mid A , B ) \cdot r ( i , j ) \cdot \nabla _ { \theta } T _ { i j } \bigg ] , ~ } & { } \\ { T _ { i j } = \log P ( i , j \mid A , B ) + \log P _ { i } ( \mathbf { x } _ { i } \mid A ) + \log P _ { j } ( \mathbf { x } _ { j } \mid B ) . } & { } \end{array}\tag{27}
$$

## 10 Outdoor localization day-night

Dataset We evaluate our approach on visual localization, i.e. the estimation of the 6-DoF pose of a query image relative to a 3D scene model, using the

Table 7: Results on outdoor visual localization using Aachen Day-Night v1.1.
<table><tr><td rowspan="2">Method</td><td colspan="3">Day</td><td colspan="3">Night</td></tr><tr><td> $. 2 5 \mathrm { m } / 2 ^ { \circ }$ </td><td> $. 5 \mathrm { m } / 5 ^ { \circ }$ </td><td> $5 \mathrm { m } / 1 0 ^ { \circ }$ </td><td> $. 2 5 \mathrm { m } / 2 ^ { \circ }$ </td><td> $. 5 \mathrm { m } / 5 ^ { \circ }$ </td><td> $5 \mathrm { m } / 1 0 ^ { \circ }$ </td></tr><tr><td>ALIKED</td><td>85.5</td><td>92.0</td><td>95.8</td><td>66.0</td><td>80.6</td><td>92.7</td></tr><tr><td>DaD</td><td>88.9</td><td>93.2</td><td>97.0</td><td>69.6</td><td>85.9</td><td>96.9</td></tr><tr><td>DeDoDe</td><td>81.8</td><td>89.1</td><td>93.4</td><td>55.5</td><td>68.1</td><td>78.0</td></tr><tr><td>DISK</td><td>81.9</td><td>91.4</td><td>95.4</td><td>61.8</td><td>75.9</td><td>86.9</td></tr><tr><td>RaCo</td><td>87.4</td><td>93.4</td><td>97.1</td><td>72.8</td><td>86.9</td><td>96.9</td></tr><tr><td>RIPE</td><td>80.9</td><td>88.5</td><td>92.8</td><td>45.0</td><td>60.2</td><td>74.9</td></tr><tr><td>RIPE++</td><td>81.3</td><td>89.0</td><td>93.7</td><td>54.5</td><td>67.5</td><td>80.1</td></tr><tr><td>↓+ Tokyo</td><td>-2.5</td><td>-1.9</td><td>-1.5</td><td>+5.7</td><td>+4.8</td><td>+3.7</td></tr><tr><td>RIPE++</td><td>78.8</td><td>87.1</td><td>92.2</td><td>60.2</td><td>72.3</td><td>83.8</td></tr></table>

Aachen v1.1 dataset [39, 40, 52]. This dataset is particularly challenging due to large viewpoint and illumination changes, including day-to-night transitions.

Metrics Using the HLoc framework [37] (commit c13273b with pycolmap 3.13.0), we first triangulate a 3D model from the 6,697 reference images, then retrieve 50 candidate images per query via NetVLAD [3] and match them against each of the 1,015 queries (824 daytime, 191 nighttime). Camera poses are estimated using a Perspective-n-Point solver with RANSAC, and we report AUC at thresholds of 0.25 m/2°, 0.5 m/5°, and $1 . 0 \mathrm { m } / 1 0 ^ { \circ }$

Baselines We compare RIPE++ against state-of-the-art methods for sparse keypoint extraction. Since RaCo and DaD are pure detectors, we pair both with ALIKED-n16 descriptors for evaluation. Images are resized to 1600 pixels on the longer side, and all methods are evaluated using mutual nearest-neighbor matching to isolate the quality of the extracted keypoints.

![](images/7cfa880104fc0576f1e41db5184939f5aac11d2a12e35233be572ba9d816eedb.jpg)  
Fig. 9: Qualitative results on the MegaDepth1500 benchmark dataset for, RIPE [19] (top), RIPE++ (ours, middle) and RaCo [43] (bottom).

![](images/bed85c99647daad2ee97e1103224f08a03548cabf5e29b1a717fff67e42fbc64.jpg)  
Fig. 10: Qualitative results on MegaDepth1500 for RIPE++ paired with our weaklysupervised LightGlue matcher.

Results Section 9 shows results on the Aachen v1.1 benchmark. RIPE++ consistently improves over RIPE across all thresholds, with particularly notable gains under nighttime conditions (+9.5 pp at 0.25 m/2°, +7.3 pp at 0.5 m/5°, +5.2 pp at 5 m/10°). On daytime queries, RIPE++ performs on par with DeDoDe and DISK. A gap to fully-supervised methods such as DaD and RaCo remains, which is expected given that RIPE++ trains without ground-truth depth or pose and, unlike RaCo, without synthetic homography augmentation. The authors of RIPE [19] also showed that the weakly-supervised training scheme facilitates the integration of additional training data. To verify that this still holds for RIPE++, we replace 20 % of the training data with day/night pairs from the query images of the Tokyo 24/7 [1] dataset. As Sec. 9 shows, this yields a slight degradation in the daytime setting—expected, given the reduced number of remaining daytime pairs—but a clear improvement for nighttime localization.

![](images/114a20b3b9da94c79e7898f10b4203efe9ea136ab20e7e2414db774eb1a91a9e.jpg)  
Fig. 11: Qualitative results on the Scared1500 benchmark dataset for RIPE++ (ours, top), RIPE++ Medical (ours, middle) and RaCo [43] (bottom).

## 11 Qualitative Results

We show qualitative results on MegaDepth1500 for sparse keypoint extraction (Fig. 9), RIPE++ in combination with LightGlue (Fig. 10) and on our proposed SCARED1500 benchmark (Fig. 11), using the same settings as in the respective quantitative evaluations. Figure 9 shows that both RIPE++ and RIPE learn to suppress keypoint detection in uninformative regions such as sky, while RaCo [43] occasionally detects keypoints on clouds (e.g., the right-most image), likely due to its reliance on synthetic homographies during training. Compared to RIPE, our method tends to localize keypoints more precisely at object boundaries, as visible around the Quadriga in the right-most image. Figure 11 illustrates the substantial domain gap between the typical MegaDepth training distribution and the medical domain, and demonstrates how training RIPE++ on medical data yields both more numerous and more accurate matches.

## References

1. A. Torii et al.: 24/7 Place Recognition by View Synthesis. CVPR (2015)

2. Allan, M., Mcleod, J., Wang, C., Rosenthal, J.C., Hu, Z., Gard, N., Eisert, P., Fu, K.X., Zefiro, T., Xia, W., Zhu, Z., Luo, H., Jia, F., Zhang, X., Li, X., Sharan, L., Kurmann, T., Schmid, S., Sznitman, R., Psychogyios, D., Azizian, M., Stoyanov, D., Maier-Hein, L., Speidel, S.: Stereo Correspondence and Reconstruction of Endoscopic Data Challenge. arXiv (2021). https://doi.org/10.48550/arxiv. 2101.01133

3. Arandjelović, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). vol. 40, pp. 5297–5307 (2016). https://doi.org/10.1109/cvpr.2016.572

4. Barroso-Laguna, A., Munukutla, S., Prisacariu, V.A., Brachmann, E.: Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 4852–4863 (2024). https://doi.org/10.1109/cvpr52733.2024.00464

5. Bay, H., Tuytelaars, T., Gool, L.V.: SURF: Speeded Up Robust Features. ECCV 2006, 9th European Conference on Computer Vision pp. 404–417 (2006). https: //doi.org/10.1007/11744023\_32

6. Bhowmik, A., Gumhold, S., Rother, C., Brachmann, E.: Reinforced Feature Points: Optimizing Feature Detection and Description for a High-Level Task. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 4947–4956 (2020). https://doi.org/10.1109/cvpr42600.2020.00500

7. Bradski, G.: The OpenCV Library. Dr. Dobb’s Journal of Software Tools (2000)

8. DeTone, D., Malisiewicz, T., Rabinovich, A.: SuperPoint: Self-Supervised Interest Point Detection and Description. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) pp. 337–349 (2018). https://doi. org/10.1109/cvprw.2018.00060

9. Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler, T.: D2-Net: A Trainable CNN for Joint Description and Detection of Local Features. CVPR pp. 8084–8093 (2019). https://doi.org/10.1109/cvpr.2019.00828

10. Edstedt, J., Athanasiadis, I., Wadenbäck, M., Felsberg, M.: DKM: Dense Kernelized Feature Matching for Geometry Estimation. arXiv (2022). https://doi.org/10. 48550/arxiv.2202.00667

11. Edstedt, J., Bökman, G., Wadenbäck, M., Felsberg, M.: DeDoDe: Detect, Don’t Describe — Describe, Don’t Detect for Local Feature Matching. 2024 International Conference on 3D Vision (3DV) pp. 148–157 (2024). https://doi.org/10.1109/ 3dv62453.2024.00035

12. Edstedt, J., Bökman, G., Wadenbäck, M., Felsberg, M.: DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection. arXiv (2025). https://doi.org/ 10.48550/arxiv.2503.07347

13. Edstedt, J., Bökman, G., Zhao, Z.: DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) pp. 4245–4253 (2024). https: //doi.org/10.1109/cvprw63382.2024.00428

14. Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., Felsberg, M.: RoMa: Robust Dense Feature Matching. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 19790–19800 (2024). https://doi.org/10.1109/ cvpr52733.2024.01871

15. Gleize, P., Wang, W., Feiszli, M.: SiLK: Simple Learned Keypoints. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 22442–22451 (2023). https://doi.org/10.1109/iccv51070.2023.02056

16. Hariharan, B., Arbelaez, P., Girshick, R., Malik, J.: Hypercolumns for Object Segmentation and Fine-Grained Localization. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 447–456 (2015). https://doi.org/ 10.1109/cvpr.2015.7298642

17. Hartley, R.I., Zisserman, A.: Multiple View Geometry in Computer Vision. Cambridge University Press, ISBN: 0521540518, second edn. (2004)

18. Jiang, H., Karpur, A., Cao, B., Huang, Q., Araujo, A.: OmniGlue: Generalizable Feature Matching with Foundation Model Guidance. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 19865–19875 (2024). https://doi.org/10.1109/cvpr52733.2024.01878

19. Künzel, J., Hilsmann, A., Eisert, P.: RIPE: Reinforcement learning on unlabeled image pairs for robust keypoint extraction. arXiv (2025)

20. Larsson, V., contributors: PoseLib - Minimal Solvers for Camera Pose Estimation (2020), https://github.com/vlarsson/PoseLib

21. Lattimore, T., Szepesvári, C.: Bandit Algorithms. Cambridge University Press (2020)

22. Leroy, V., Cabon, Y., Revaud, J.: Grounding Image Matching in 3D with MASt3R. In: European Conference on Computer Vision. pp. 71–91 (2024). https://doi.org/ 10.1007/978-3-031-73220-1\_5, https://api.semanticscholar.org/CorpusID: 270521424

23. Li, K., Wang, L., Liu, L., Ran, Q., Xu, K., Guo, Y.: Decoupling Makes Weakly Supervised Local Feature Better. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 15817–15827 (2022). https://doi.org/10. 1109/cvpr52688.2022.01538

24. Li, Z., Snavely, N.: MegaDepth: Learning Single-View Depth Prediction from Internet Photos. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition pp. 2041–2050 (2018). https://doi.org/10.1109/cvpr.2018.00218

25. Lindenberger, P., Sarlin, P.E., Pollefeys, M.: LightGlue: Local Feature Matching at Light Speed. In: International Conference on Computer Vision (ICCV) (2023)

26. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2017), https://api.semanticscholar. org/CorpusID:53592270

27. Lowe, D.G.: Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision pp. 91–110 (2004). https://doi.org/10.1023/b: visi.0000029664.99615.94

28. Oord, A.v.d., Li, Y., Vinyals, O.: Representation Learning with Contrastive Predictive Coding. arXiv (2018). https://doi.org/10.48550/arxiv.1807.03748

29. Pautrat\*, R., Suárez\*, I., Yu, Y., Pollefeys, M., Larsson, V.: GlueStick: Robust Image Matching by Sticking Points and Lines Together. In: International Conference on Computer Vision (ICCV) (2023)

30. Potje, G., Cadar, F., Araujo, A., Martins, R., Nascimento, E.R.: Enhancing Deformable Local Features by Jointly Learning to Detect and Describe Keypoints . In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1306–1315. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2023). https://doi.org/10.1109/CVPR52729.2023.00132, https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.00132

31. Radenović, F., Iscen, A., Tolias, G., Avrithis, Y., Chum, O.: Revisiting oxford and paris: Large-scale image retrieval benchmarking. In: CVPR (2018)

32. Revaud, J., Weinzaepfel, P., Souza, C.D., Pion, N., Csurka, G., Cabon, Y., Humenberger, M.: R2D2: Repeatable and Reliable Detector and Descriptor. arXiv (2019). https://doi.org/10.48550/arxiv.1906.06195

33. Riba, E., Mishkin, D., Ponsa, D., Rublee, E., Bradski, G.: Kornia: an open source diferentiable computer vision library for pytorch. In: Winter Conference on Appli cations of Computer Vision (2020), https://arxiv.org/pdf/1910.02190.pdf

34. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: ORB: an eficient alternative to SIFT or SURF. 2011 International Conference on Computer Vision pp. 2564–2571 (2011). https://doi.org/10.1109/iccv.2011.6126544

35. Ryoo, K., Lim, H., Myung, H.: MambaGlue: Fast and Robust Local Feature Matching with Mamba. 2025 IEEE International Conference on Robotics and Automation (ICRA) pp. 5758–5765 (2025). https://doi.org/10.1109/icra55743. 2025.11128473

36. Santellani, E., Sormann, C., Rossi, M., Kuhn, A., Fraundorfer, F.: S-TREK: Sequential translation and rotation equivariant keypoints for local feature extraction. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9694–9703 (2023). https://doi.org/10.1109/iccv51070.2023.00892

37. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From Coarse to Fine: Robust Hierarchical Localization at Large Scale. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019). https://doi.org/10.1109/cvpr. 2019.01300

38. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: SuperGlue: Learning Feature Matching with Graph Neural Networks. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 4937–4946 (2020). https://doi.org/10.1109/cvpr42600.2020.00499

39. Sattler, T., Maddern, W., Toft, C., Torii, A., Hammarstrand, L., Stenborg, E., Safari, D., Okutomi, M., Pollefeys, M., Sivic, J., Kahl, F., Pajdla, T.: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2018)

40. Sattler, T., Weyand, T., Leibe, B., Kobbelt, L.: Image Retrieval for Image-Based Localization Revisited. In: British Machine Vision Conference (BMCV) (2012)

41. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

42. Shen, X., Cai, Z., Yin, W., Müller, M., Li, Z., Wang, K., Chen, X., Wang, C.: Gim: Learning generalizable image matcher from internet videos. In: The Twelfth International Conference on Learning Representations (2024)

43. Shenoi, t., Lindenberger, P., Sarlin, P.E., Pollefeys, M.: RaCo: Ranking and Covari ance for Practical Learned Keypoints. In: Thirteenth International Conference on 3D Vision (2026), https://openreview.net/forum?id=BWtdgrdcBH

44. Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X.: LoFTR: Detector-Free Local Feature Matching with Transformers. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8918–8927 (2021). https://doi.org/10.1109/ cvpr46437.2021.00881

45. Tyszkiewicz, M., Fua, P., Trulls, E.: DISK: Learning local features with policy gradient. In: Advances in Neural Information Processing Systems. pp. 14254–14265. Curran Associates, Inc. (2020). https://doi.org/10.5555/3495724

46. Wang, C., Xu, R., Lu, K., Xu, S., Meng, W., Zhang, Y., Fan, B., Zhang, X.: Attention Weighted Local Descriptors. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 10632–10649 (2023). https://doi.org/10.1109/tpami. 2023.3266728

47. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: DUSt3R: Geometric 3D Vision Made Easy . In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20697–20709. IEEE Computer Society, Los Alamitos, CA, USA (2024). https://doi.org/10.1109/CVPR52733.2024.01956, https://doi.ieeecomputersociety.org/10.1109/CVPR52733.2024.01956

48. Wang, Y., He, X., Peng, S., Tan, D., Zhou, X.: Eficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 21666–21675 (2024). https://doi.org/10.1109/cvpr52733.2024.02047

49. Williams, R.J.: Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning pp. 229–256 (1992). https://doi.org/ 10.1007/bf00992696

50. Xu, R., Wang, C., Xu, S., Meng, W., Zhang, Y., Fan, B., Zhang, X.: DomainFeat: Learning Local Features With Domain Adaptation. IEEE Transactions on Circuits and Systems for Video Technology pp. 46–59 (2024). https://doi.org/10.1109/ tcsvt.2023.3282956

51. Zhang, S., Ma, J.: DifGlue: Difusion-Aided Image Feature Matching. Proceedings of the 32nd ACM International Conference on Multimedia pp. 8451–8460 (2024). https://doi.org/10.1145/3664647.3681069

52. Zhang, Z., Sattler, T., Scaramuzza, D.: Reference Pose Generation for Visual Localization via Learned Features and View Synthesis. arXiv 2005.05179 (2020)

53. Zhao, X., Wu, X., Chen, W., Chen, P.C.Y., Xu, Q., Li, Z.: Aliked: A lighter keypoint and descriptor extraction network via deformable transformation. IEEE Transactions on Instrumentation and Measurement pp. 1–16 (2023). https://doi. org/10.1109/TIM.2023.3271000