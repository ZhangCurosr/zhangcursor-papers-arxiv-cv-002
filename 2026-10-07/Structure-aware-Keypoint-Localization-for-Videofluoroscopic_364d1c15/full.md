# Structure-aware Keypoint Localization for Videofluoroscopic Swallowing Study

Kai Zhou<sup>1∗</sup>, Chuanshen Chen<sup>1∗</sup>, Runhao Zeng<sup>2∗</sup>, Meng Dai<sup>3</sup>, Yifan Yang<sup>4</sup>, Jinwu Hu<sup>1</sup>, Daiyuan Li<sup>1</sup>, Mingkui Tan<sup>1†</sup>, Fei Liu<sup>1†</sup>

<sup>1</sup>South China University of Technology, <sup>2</sup>Shenzhen MSU-BIT University,

<sup>3</sup>The Third Affiliated Hospital of Sun Yat-sen University, <sup>4</sup>Electric Power Research Institute, China South Grid {kayjoe0723, chuanshenc888, youngyif1, fhujinwu, ldyabroad}@gmail.com, zengrh@smbu.edu.cn daim@mail3.sysu.edu.cn, {mingkuitan, feiliu}@scut.edu.cn

Abstract—Videofluoroscopic Swallowing Study (VFSS) is one of the gold standard for diagnosing swallowing disorders, providing dynamic X-ray imaging of the swallowing process. Automated kinematic analysis in VFSS relies fundamentally on precise anatomical keypoint localization. However, existing studies focus on limited keypoints (e.g., cervical vertebrae or the hyoid) and overlook critical regions such as the soft palate, while annotating only active swallowing segments and ignoring abundant non-swallowing data, resulting in poor data efficiency. Moreover, leveraging this unlabeled data via standard semisupervised learning is suboptimal, as generic methods are prone to spatial bias. In medical X-rays with fixed layouts, models tend to memorize absolute coordinates rather than understanding anatomical structures. To tackle these challenges, we introduce VFSSKep, a novel dataset that extends annotations to the soft palate and incorporates large-scale unlabeled data. We further propose S<sup>3</sup>KL, a Structure-aware Semi-Supervised Keypoint Localization framework designed to overcome spatial bias. It integrates a Structure-Aware Learning strategy to extract highresolution structural cues for structure-aware representation learning, and a Structural Representation Consistency Learning strategy with block shuffling to enforce invariant structural recognition. Experiments show our method achieves state-ofthe-art semi-supervised performance, even with unlabeled and 25% labeled data surpassing fully supervised learning with 100% labeled data. Code and data will be made publicly available at: https://github.com/kaai520/S3KL.

Index Terms—Videofluoroscopic swallowing study, keypoint localization, semi-supervised learning.

## I. INTRODUCTION

Swallowing disorders (dysphagia) [1] are a serious illness that frequently lead to life-threatening complications, such as severe malnutrition and aspiration pneumonia [2]. Over 30% of individuals aged 65 and older are affected by swallowing disorders [3]. The Videofluoroscopic Swallowing Study (VFSS) is one of the gold-standard diagnostic tools, providing dynamic X-ray imaging of the swallowing process [4]. Recently, clinical assessment has increasingly relied on quantitative kinematic analysis; for example, hyoid bone displacement is a key indicator for detecting aspiration and pharyngeal residues [5]. This trend has driven the need for automated and precise localization of critical anatomical landmarks (e.g., the keypoints of the cervical vertebrae [6] and hyoid bone [7]).

![](images/7f87ba7f223b0465ff494cf8b6fb29cb8987d973dcf340736c72cc139dcbb1d5.jpg)  
Fig. 1. Illustration of the eight keypoints in our VFSSKep dataset.

Despite recent progress, existing VFSS keypoint (landmark) localization benchmarks [6], [7] are limited in anatomical scope and data utilization. First, current datasets focus exclusively on rigid structures like the hyoid bone, overlooking the soft palate. This deformable stucture is critical for preventing nasal reflux [8], yet its non-rigid motion makes manual annotation notoriously difficult [9], leaving it absent from automated systems. Second, existing benchmarks suffer from poor data efficiency. While clinical videos are long (often minutes), annotations are typically restricted to brief “active swallowing” windows (2-4 seconds), resulting in the exclusion of most non-swallowing frames and leaving a substantial amount of potentially informative anatomical data unused.

Leveraging this abundance of unlabeled data via Semi-Supervised Learning (SSL) is a natural solution, yet applying generic computer vision SSL paradigms to VFSS is non-trivial. Existing semi-supervised keypoint localization methods [10]– [12] typically rely on global geometric perturbations (e.g., rotation) to enforce consistency. In the context of medical Xray imaging, which exhibits low tissue contrast and relatively fixed anatomical layouts, these methods are prone to shortcut learning. Previous models tend to memorize absolute keypoint coordinates (see Fig. 6) rather than learning to distinguish the anatomical structure, leading to spatial bias that limits generalization diverse anatomical variations.

TABLE I  
COMPARISON OF VFSS KEYPOINT DATASETS. OUR VFSSKEP IS THE FIRST TO INCLUDE SOFT PALATE ANNOTATIONS AND LEVERAGE LARGE-SCALE UNLABELED DATA FOR SEMI-SUPERVISED LEARNING.
<table><tr><td>Dataset</td><td>Year</td><td>Anatomical Structure</td><td>Labeled Frames</td><td>Unlabeled Frames</td></tr><tr><td>Zhang et al. [6]</td><td>2021</td><td>Cervical Vertebrae</td><td>60k</td><td>=</td></tr><tr><td>Hsiao et al. [7]</td><td>2023</td><td>Hyoid + Vertebrae</td><td>7k</td><td>=</td></tr><tr><td>Ours (VFSSKep)</td><td>2025</td><td>Hyoid + Vertebrae + Soft Palate</td><td>14k</td><td>32k</td></tr></table>

![](images/ca4a3834e2961480379922d24ab2cb227f69406a531f2f0824050bbc051278a5.jpg)  
Fig. 2. Comparison of the traditional (a) and our proposed (b) paradigms for utilizing unlabeled data in semi-supervised keypoint localization. Our paradigm leverages dense anatomical context to assist keypoint localization, learning structural priors without extra data or annotations.

To bridge these gaps, we present a holistic solution comprising a new benchmark and a tailored learning framework. First, we introduce VFSSKep, the first VFSS keypoint dataset to provide keypoint annotations for the deformable soft palate and incorporate large-scale non-swallowing segments as unlabeled data (see Fig. 1 and Tab. I). Second, to effectively leverage these data, we propose a Structure-aware Semi-Supervised Keypoint Localization framework (S<sup>3</sup>KL). Unlike conventional approaches, our method explicitly prevents shortcut learning by prioritizing anatomical understanding over spatial heuristics. We achieve this through two core paradigms: 1) A Structure-Aware Learning strategy. To compensate for the lack of dense medical annotations, we leverage the robust structure awareness of vision foundation models (e.g., DINOv2 [13]). We design a module to recover high-resolution details from these general priors, guiding the encoder to perceive subtle anatomical structures (see Fig. 2). 2) A Structural Representation Consistency Learning strategy. We introduce a “Block Shuffle” operation to disrupt global spatial contexts at the input level. By restoring the feature positions and enforcing consistency with the original image, we compel the model to identify local anatomical parts independent of their absolute locations, significantly enhancing generalization capability.

Our contributions are as follows:

![](images/087585c624bdb90fa0692c37283ba7cc2d1cade41daac0f8f81d63bf104afbeb.jpg)  
Fig. 3. Summary of the construction of our VFSSKep dataset.

• We introduce VFSSKep, a novel VFSS keypoint dataset that extends anatomical scope to the soft palate keypoints and integrates large-scale unlabeled data, enabling dataefficient clinical analysis via semi-supervised learning.

• We propose S<sup>3</sup>KL framework, designed to overcome spatial bias through two core paradigms: a Structure-Aware Learning strategy that recovers high-resolution anatomical details from foundation model features to enforce dense structural alignment, and a Structural Representation Consistency Learning strategy for learning invariant features via input-level spatial disruption and feature-level restoration.

• Experiments demonstrate the superiority of our approach. Notably, our S<sup>3</sup>KL trained with only 25% labeled data outperforms fully supervised baselines using 100% labels, highlighting its exceptional data efficiency and generalization capability.

## II. VFSS KEYPOINT DATASET CONSTRUCTION

To address the limitations of existing benchmarks, we construct VFSSKep, a large-scale fluoroscopic video dataset tailored for semi-supervised keypoint localization.

Dataset Overview and Statistics. The dataset was collected from a collaborating medical center using digital fluoroscopic systems. As shown in Fig. 3, it comprises 13,895 labeled frames extracted from active swallowing segments and 31,536 unlabeled frames from non-swallowing segments. For the labeled data, we follow a subject-independent split ratio of roughly 6:1:3, resulting in 8,302 training, 1,518 validation, and 4,075 test images. Crucially, distinct individuals are strictly separated across subsets to prevent data leakage. The unlabeled set, derived from the remaining videos, captures diverse static and motion patterns to support semi-supervised learning. Detailed acquisition protocols, filtering strategies, and demographic statistics are provided in the Appendix.

Keypoint Definition. As shown in Fig. 1, we annotate eight anatomical landmarks essential for swallowing kinematics: the superior, anterior, and inferior points of the hyoid bone; the anterior–inferior corners of C2 and C4 vertebrae; and three points on the soft palate (anterior, midpoint, inferior).

![](images/5949bc931a62bada69e2745c6b3ba9e3e5dea184cef2d588b9e47b73abefc4d6.jpg)  
Fig. 4. An overview of our proposed $\mathrm { S ^ { 3 } K L }$ framework. Our proposed $\mathrm { s ^ { 3 } K L }$ framework applies easy augmentation $T _ { e }$ to an unlabeled image for teacher predictions, transformed via easy-to-hard mapping $T _ { e  h }$ as pseudo-labels for the student. Beyond keypoint supervision, we propose a Structure-Aware Learning (SAL) strategy to leverage dense anatomical cues, assisting keypoint localization by aligning the auxiliary prediction head output with structural prior from the Structure Pairing Module. The output from the Structure Parsing Module is visualized using PCA [14]. In addition, we introduce a Structural Representation Consistency (SRC) Learning strategy, where the Block Shuffle operation disrupts spatial structure, encouraging the model to maintain consistent representations despite such perturbations. The training process of labeled images is omitted for simplicity.

Annotations were performed by clinical experts with strict multi-step verification.

Comparison with Existing Datasets. VFSSKep differs in two aspects: 1) Deformable structures: Unlike prior datasets [6], [7], it includes soft palate keypoints, enabling oral-phase analysis. 2) Semi-supervised design: It provides abundant unlabeled non-swallowing frames, reflecting clinical label scarcity and offering a semi-supervised setting.

## III. METHOD

## A. Preliminaries and Method Overview

Problem Formulation. We address the problem of semisupervised keypoint localization. Given a labeled dataset $\mathcal { D } _ { s } =$ $\{ ( \bar { \mathbf { I } } _ { i } ^ { s } , \mathbf { H } _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } }$ and a larger unlabeled dataset $\mathbf { \mathcal { D } } _ { u } = \{ \mathbf { I } _ { i } ^ { u } \} _ { i = 1 } ^ { N _ { u } }$ our goal is to train a robust model $f _ { \theta }$ by leveraging both sets. For each image I, the ground truth H consists of $K$ Gaussian heatmaps, each indicating the probability distribution of a specific anatomical keypoint.

Standard SSL Paradigm. Existing methods [10], [12] typically adopt a Teacher-Student framework. The student network is trained on labeled data via a supervised loss $\mathcal { L } _ { s }$ and on unlabeled data via an unsupervised loss $\mathcal { L } _ { u }$ . Taking SemiPose [10] as a baseline, the learning objective is:

$$
\mathcal { L } _ { b a s e } = \mathcal { L } _ { s } + \lambda \mathcal { L } _ { u } ,\tag{1}
$$

$$
\mathcal { L } _ { s } = \mathbb { E } _ { ( \mathbf { I } ^ { s } , \mathbf { H } ^ { s } ) \sim \mathcal { D } _ { s } } [ \| f _ { \theta } ( T _ { e } ( \mathbf { I } ^ { s } ) ) - T _ { e } ( \mathbf { H } ^ { s } ) \| ^ { 2 } ] ,\tag{2}
$$

$$
\mathcal { L } _ { u } = \mathbb { E } _ { \mathbf { I } ^ { u } \sim \mathcal { D } _ { u } } [ \| f _ { \theta } ( T _ { h } ( \mathbf { I } ^ { u } ) ) - T _ { e  h } ( f _ { \theta ^ { \prime } } ( T _ { e } ( \mathbf { I } ^ { u } ) ) ) \| ^ { 2 } ] ,\tag{3}
$$

where $T _ { e }$ denotes an easy augmentation. Consistency is enforced between the student under a hard augmentation $T _ { h }$ and the teacher under $T _ { e }$ , with $T _ { e  h }$ aligning the teacher output to the student view. Here, θ and $\theta ^ { \prime }$ denote the student and teacher parameters, and $\lambda _ { 1 } , \lambda _ { 2 }$ balance the loss terms.

Method Overview. Due to the fixed anatomical layouts and potential spatial bias in VFSS X-ray imaging, learning robust features from unlabeled data without shortcut learning is a highly challenging task. To address this, we propose a structure-aware semi-supervised keypoint localization framework, namely $\mathrm { S ^ { 3 } K L }$ . As illustrated in Fig. 4, the framework comprises two core parts: 1) Structure-Aware Learning. It is implemented through an auxiliary branch that extracts dense structural representations from the encoder’s features. This branch is supervised by a reference signal generated through our proposed Structure Parsing Module, which produces meaningful anatomical patterns from the input image without requiring additional annotations. 2) Structural Representation Consistency Learning. We design a “Block Shuffle” mechanism to explicitly disrupt global spatial contexts at the input level. By compelling the model to restore feature consistency under such perturbations, this module enforces the learning of invariant anatomical representations, significantly enhancing generalization capability.

## B. Structure-Aware Learning

To prioritize anatomical understanding, it is essential to enforce dense structural constraints on the model. While prior studies [15] typically achieve this by introducing auxiliary semantic segmentation tasks, obtaining such pixel-wise annotations for VFSS is prohibitively costly. Defining anatomically meaningful regions is also ambiguous due to the complexity of soft tissues. Although auto-segmentation methods such as SAM [16] and MedSAM [17] produce coarse contours, they lack VFSS-specific semantics and often fail to distinguish functionally relevant regions (Fig. 5(a)).

Structure Parsing Module. To address this, we propose a Structure Parsing Module (SPM) that identifies critical anatomical regions without manual annotations. We leverage large-scale self-supervised models (e.g., DINOv2 [13]) for contextual priors, but their limited output resolution causes blurred boundaries and loss of detail (Fig. 5(b)). We therefore incorporate FeatUp [18] to recover high-resolution, semantically consistent features. As shown in Fig. 5(c), the enhanced SPM yields clear anatomical structures such as the soft palate (white), hyoid bone (yellow), and vertebrae (red) that align with the target keypoints, improving interpretability and providing strong structural guidance without medical annotations.

![](images/d8d0971b90e3f0fadef9baa5d0cf4512a4fcf0e43b2e9d7eb3df0da28f0289a3.jpg)  
(a) SAM

![](images/f4708921a644b84602cbc780557ca0bec4a840c3f7bbfef9ba84c33cb789adac.jpg)  
(b) DINOv2

![](images/423a3a69aefd212335cb205bcf4148f551b1ed592b35e8ef7a6aca0ff94a9223.jpg)  
(c) Our SPM  
Fig. 5. (a) X-ray image segmented by SAM [16] with regions highlighted in different colors, but without semantic labels. (b) Output features from DINOv2 [13] for the same image, visualized by mapping PCA results to RGB channels. (c) Output features from our Structure Parsing Module (SPM), also visualized using PCA mapping.

Structure-Aware Branch. To effectively distill these dense priors into the localization model, we propose a Structure-Aware Branch (SAB). Specifically, we append an auxiliary prediction head $f _ { \theta , s }$ to the shared backbone $f _ { \theta , \mathrm { e n c } }$ . Unlike traditional multi-task learning that relies on discrete semantic segmentation labels [15], our SPM yields continuous, highresolution structural features. Accordingly, we replace the conventional cross-entropy loss with a cosine similarity loss to align the student representations with the structural targets provided by SPM. The loss function is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o s } } ( \mathbf { x } ) = \mathbb { E } _ { ( i , j ) } \left[ \mathbb { I } ( \mathsf { s i m } _ { i j } \le \alpha ) ( 1 - \mathsf { s i m } _ { i j } ) \right] , } \end{array}\tag{4}
$$

where si $\mathbf { m } _ { i j } = \cos ( f _ { \theta , s } ( \mathbf { x } ) _ { i j } , f _ { \mathrm { S P M } } ( \mathbf { x } ) _ { i j } )$ denotes the cosine similarity between the SAB prediction and SPM target at location $( i , j )$ . The indicator function I(·) works with the threshold α to focus learning on misaligned regions.

Finally, we apply this constraint to unlabeled data under hard augmentation to enforce robust structural representation learning. The overall Structure-Aware Learning loss is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S A L } } = \mathbb { E } _ { \mathbf { I } ^ { u } \sim \mathcal { D } _ { u } } [ \mathcal { L } _ { \mathrm { c o s } } ( T _ { h } ( \mathbf { I } ^ { u } ) ) ] . } \end{array}\tag{5}
$$

## C. Structural Representation Consistency Learning

To explicitly disrupt the spatial bias where models memorize absolute coordinates, we introduce a Structural Representation Consistency (SRC) Learning strategy. Unlike photometric augmentations that only alter pixel values, our approach destroys global geometric context via a Block Shuffle (BS) operation, compelling the encoder to recognize anatomical structures based on local visual patterns rather than absolute positions.

Formally, given an input image x, we partition it into a grid of non-overlapping blocks and apply a random permutation $T _ { \mathrm { B S } }$ , yielding a shuffled image $\mathbf { x } _ { s } = T _ { \mathrm { B S } } ( \mathbf { x } )$ . This shuffled input is processed by the encoder $f _ { \boldsymbol { \theta } , \mathrm { e n c } } ,$ . Crucially, unlike pretext tasks that predict permutation indices [19], our goal is feature invariance. Therefore, we apply an inverse operation $T _ { \mathrm { U S } }$ (Unshuffle) to the encoded features, restoring the original spatial order before the prediction heads:

$$
\mathcal { F } _ { o } = T _ { \mathrm { U S } } ( f _ { \theta , \mathrm { e n c } } ( T _ { \mathrm { B S } } ( \mathbf { x } ) ) ) .\tag{6}
$$

This process forces the encoder to generate logically consistent features even when the global spatial layout is corrupted, effectively enforcing local-to-global anatomical reasoning.

We integrate this mechanism into the semi-supervised consistency framework. The student model processes the blockshuffled image, and its spatially restored features are forced to align with the teacher’s stable predictions. The modified unsupervised loss $\mathcal { L } _ { u } ^ { \mathrm { S R C } }$ is formulated as:

$$
\mathcal { L } _ { u } ^ { \mathrm { S R C } } = \mathbb { E } _ { \mathbf { I } ^ { u } \sim \mathcal { D } _ { u } } [ \| g _ { \boldsymbol \theta } ( T _ { h } ( \mathbf { I } ^ { u } ) ) - T _ { e  h } ( f _ { \boldsymbol \theta ^ { \prime } } ( T _ { e } ( \mathbf { I } ^ { u } ) ) ) \| ^ { 2 } ] ,\tag{7}
$$

where $g _ { \theta } ( \cdot ) = f _ { \theta , k } \circ T _ { \mathrm { U S } } \circ f _ { \theta , \mathrm { e n c } } \circ T _ { \mathrm { B S } }$ represents the student’s forward pass with Block Shuffle. Here, $f _ { \theta , k }$ denotes the keypoint prediction head, and $T _ { \mathrm { U S } }$ restores the spatial order of features extracted by the encoder $f _ { \boldsymbol { \theta } , \mathrm { e n c } } ,$ . Note that this strategy is similarly applied to the Structure-Aware Learning loss $( \mathcal { L } _ { \mathrm { S A L } } )$ to further robustify structural feature extraction.

## D. Overall Learning Objectives

The total learning objective combines the supervised loss, standard unsupervised loss, and our proposed structure-aware regularization terms:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { s } + \lambda _ { 2 } ( \mathcal { L } _ { u } + \mathcal { L } _ { u } ^ { \mathrm { S R C } } ) + \lambda _ { 3 } ( \mathcal { L } _ { \mathrm { S A L } } + \mathcal { L } _ { \mathrm { S A L } } ^ { \mathrm { S R C } } ) , } \end{array}\tag{8}
$$

where $\lambda _ { 1 , 2 , 3 }$ are balancing coefficients. Here, $\mathcal { L } _ { u } ^ { \mathrm { S R C } }$ and $\mathcal { L } _ { \mathrm { S A L } } ^ { \mathrm { S R C } }$ denote the consistency and structure-aware losses applied to the block-shuffled branch, respectively.

Dual Networks Learning. Following SemiPose [10], we employ two networks $f _ { \theta }$ and $f _ { \xi }$ to mutually supervise each other. Instead of a standard EMA teacher, each network acts as a pseudo-label generator for the other on unlabeled data. Specifically, for the consistency terms $( \mathcal { L } _ { u }$ and $\mathcal { L } _ { u } ^ { \mathrm { S R C } } )$ , we compute the loss by aligning $f _ { \boldsymbol { \theta } } { } ^ { \prime } { \mathbf { s } }$ predictions with the fixed pseudo-labels from $f _ { \xi }$ (and vice versa). This interaction prevents coupling errors and stabilizes training. Final predictions are obtained by fusing the outputs of both networks.

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Settings

Datasets and Metrics. We conduct experiments on our VFSSKep Dataset (Sec. II), as existing benchmarks [6], [7] are unavailable. We use 25%, 50%, and 100% of the labeled training data, each combined with the full unlabeled set, and evaluate on the VFSSKep test set. Evaluation relies on two metrics: Probability of Correct Keypoint (PCK) and Mean Euclidean Distance (MED). For PCK, we adopt a strict threshold (PCK@0.02), considering a keypoint correct if it falls within 2% of the image diagonal. MED measures the average physical distance (mm) for complementary precision analysis.

TABLE II  
COMPARISON WITH METHODS USING DIFFERENT PROPORTIONS OF LABELED DATA. PCK@0.02 (%) AND MED (MM) ARE REPORTED. OUR $\mathrm { S ^ { 3 } K L }$ ISPLUG-AND-PLAY, WHERE VARIATIONS IN UNSUPERVISED LOSS $\mathcal { L } _ { u }$ ACROSS DIFFERENT METHODS DO NOT AFFECT ITS INTEGRATION. BEST RESULTS AREIN BOLD, AND BLUE NUMBERS SHOW RELATIVE IMPROVEMENTS OVER SEMIPOSE AND G2LCPS.
<table><tr><td rowspan="2">Method</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK@0.02 (%)↑</td><td>MED (mm)↓</td><td>PCK@0.02 (%)↑</td><td>MED (mm)↓</td><td>PCK@0.02 (%)↑</td><td>MED (mm)↓</td></tr><tr><td>Supervised [20]</td><td>63.2</td><td>4.34</td><td>73.5</td><td>3.47</td><td>75.6</td><td>3.29</td></tr><tr><td>SemiPose [10]</td><td>77.2</td><td>3.63</td><td>79.0</td><td>3.14</td><td>81.9</td><td>2.65</td></tr><tr><td>+S³KL (Ours)</td><td>78.8 (↑2.1%)</td><td>2.88 (↓20.7%)</td><td>82.1 (↑3.9%)</td><td>2.64 (↓15.9%)</td><td>83.6 (↑2.1%)</td><td>2.48 (↓6.4%)</td></tr><tr><td>G2LCPS [12]</td><td>78.2</td><td>2.98</td><td>81.0</td><td>2.76</td><td>83.1</td><td>2.54</td></tr><tr><td>+S3KL (Ours)</td><td>79.4 (↑1.5%)</td><td>2.87 (↓3.7%)</td><td>82.9 (↑2.3%)</td><td>2.57 (↓6.9%)</td><td>83.8 (↑0.8%)</td><td>2.46 (↓3.1%)</td></tr></table>

Implementation Details. We employ SimpleBaseline [20] with a ResNet18 [21] backbone (256 × 256) as the backbone. The Structure Parsing Module integrates a fixed DINOv2 with JBU upsamplers [18]. Models are trained for 100 epochs (batch size 64) using Adam with an initial learning rate of $1 0 ^ { - 3 }$ , decayed at epochs 60 and 90. Following SemiPose [10], we apply easy (rotation ±30<sup>◦</sup>) and hard (rotation $\pm 6 0 ^ { \circ }$ with our Block Shuffle) augmentations to the teacher and student, respectively. Loss weights are set to $\lambda _ { 1 } ~ = ~ 0 . 5 , ~ \lambda _ { 2 } ~ = ~ 1 . 0$ and feature alignment $\lambda _ { 3 } = 1 0 ^ { - 3 }$ , with a similarity threshold $\alpha = 0 . 8 5$ . Architectural details of SAB and SPM, full hyperparameter analysis, and extended experiments $( e . g .$ , backbone variants) are provided in the Appendix.

Baselines. We compare $\mathrm { S ^ { 3 } K L }$ against methods representing key milestones in the field: the supervised lower bound SimpleBaseline [20], the classic semi-supervised method Semi-Pose [10], and the recent state-of-the-art G2LCPS [12]. Comprehensive related works are discussed in the Appendix.

## B. Main Results

Effectiveness of semi-supervised learning. As shown in Tab. II, all semi-supervised methods, using only 25% labeled data, surpass the supervised baseline trained on 100% labeled data. This demonstrates the effectiveness of semi-supervised learning with our unlabeled non-swallowing segments.

Effectiveness of S<sup>3</sup>KL. As shown in Tab. II, integrating our S<sup>3</sup>KL into SemiPose [10] and G2LCPS [12] outperforms existing methods. Compared to SemiPose, S<sup>3</sup>KL achieves PCK gains of 2.1%, 3.9%, and 2.1% at 25%, 50%, and 100% supervision, respectively, while notably reducing MED by 20.7% and 15.9% at 25% and 50% supervision. When combined with G2LCPS, S<sup>3</sup>KL further improves PCK by 1.5%, 2.3%, and 0.8%, again accompanied by lower MED, and achieves the best overall performance. These results confirm that our plug-and-play design effectively leverages both labeled and unlabeled data for robust keypoint localization, and can serve as a complementary module to existing methods.

## C. Ablation Studies

We evaluate the effect of key components in our $\mathrm { S ^ { 3 } K L }$ , using SemiPose [10] as the semi-supervised baseline.

Effect of our proposed Structure-Aware Learning and Structural Representation Consistency Learning. Tab. III shows that both Structure-Aware Learning (SAL) and Structural Representation Consistency (SRC) Learning contribute to performance gains. SAL enhances keypoint localization by leveraging structural context, while SRCL improves robustness by enforcing consistency in structural representations. Their combination, S<sup>3</sup>KL, achieves the best results, with PCK improvements along with notable reductions in MED.

TABLE III  
ABLATION STUDY ON THE PROPOSED STRUCTURE-AWARE LEARNING (SAL) AND STRUCTURAL REPRESENTATION CONSISTENCY (SRC) LEARNING. PCK@0.02 (%) AND MEAN EUCLIDEAN DISTANCE (MED, IN MM) ARE REPORTED.
<table><tr><td rowspan="2">Method</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td>Baseline [10]</td><td>77.2</td><td>3.63</td><td>79.0</td><td>3.14</td><td>81.9</td><td>2.65</td></tr><tr><td>Baseline w/ SAL</td><td>78.1</td><td>3.22</td><td>81.6</td><td>2.69</td><td>82.4</td><td>2.58</td></tr><tr><td>Baseline w/ SRC</td><td>78.4</td><td>3.62</td><td>81.2</td><td>2.72</td><td>82.7</td><td>2.57</td></tr><tr><td>S³KL (Ours)</td><td>78.8</td><td>2.88</td><td>82.1</td><td>2.64</td><td>83.6</td><td>2.48</td></tr></table>

TABLE IV

EFFECT OF AUXILIARY MODELS AND UPSAMPLING STRATEGIES ON THE STRUCTURE PARSING MODULE. PCK@0.02 (%) AND MEAN EUCLIDEAN DISTANCE (MED, IN MM) ARE REPORTED.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Upsample</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td>CLIP</td><td>FeatUp</td><td>77.8</td><td>3.73</td><td>80.5</td><td>2.86</td><td>81.9</td><td>2.74</td></tr><tr><td>DINOv2</td><td>FeatUp</td><td>78.1</td><td>3.22</td><td>81.6</td><td>2.69</td><td>82.4</td><td>2.58</td></tr><tr><td>DINOv2</td><td>Bilinear</td><td>77.8</td><td>2.97</td><td>80.9</td><td>2.79</td><td>81.9</td><td>2.68</td></tr></table>

Effect of auxiliary models and upsampling strategies of the Structure Parsing Module. The Structure Parsing Module, part of Structure-Aware Learning, can leverage different pre-trained models as auxiliaries. In addition to DINOv2, we explore CLIP [22], another large-scale pre-trained model, as an auxiliary model. As shown in Tab. IV, using CLIP outperforms the baseline but lags behind DINOv2, as it relies on imagetext pairs rather than self-supervised learning from image data, limiting its ability to provide adequate structural priors. We also evaluate alternative upsampling techniques, finding that replacing FeatUp with bilinear interpolation reduces performance, highlighting FeatUp’s effectiveness in preserving structures while restoring high-resolution spatial details.

Effect of unshuffle position in Structure Representation Consistency Learning. We investigate the impact of applying the unshuffle operation at different stages: (1) Block Shuffle (BS) with unshuffle on encoder output and (2) BS with unshuffle on final output. Tab. V shows that both variants improve performance over the baseline (w/o BS). Applying it to the encoder output achieves the best overall results, with consistently higher PCK and lower MED at 50% and 100% supervision. Although its MED at 25% supervision is relatively high, this effect diminishes with more labeled data, and the encoder variant remains clearly superior in most settings. These results indicate that restoring disrupted features at the encoder stage yields stronger spatial and structural representations, making it the preferred choice.

EFFECT OF UNSHUFFLE POSITION ON THE BLOCK SHUFFLE (BS) OPERATION. “FINAL” INDICATES APPLYING UNSHUFFLE TO THE FINAL OUTPUT, WHILE “ENCODER” INDICATES APPLYING UNSHUFFLE TO THE ENCODER OUTPUT.
<table><tr><td rowspan="2">Method</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td>w/o BS</td><td>77.2</td><td>3.63</td><td>79.0</td><td>3.14</td><td>81.9</td><td>2.65</td></tr><tr><td>BS (final)</td><td>77.7</td><td>3.16</td><td>80.9</td><td>2.77</td><td>82.5</td><td>2.59</td></tr><tr><td>BS (encoder)</td><td>78.4</td><td>3.62</td><td>81.2</td><td>2.72</td><td>82.7</td><td>2.57</td></tr></table>

![](images/fee30b8a849e9872b73f34afc934af036f4c0c44da30a964434fca755697bc2b.jpg)  
Fig. 6. Comparison of keypoint localization results between the baseline method [10] and our S<sup>3</sup>KL under 25% supervision. Green points represent the ground truth positions, while blue points represent the predicted keypoints.

## D. Visualization Results

We visualize the keypoint localization results of the baseline method [10] and our proposed $\mathrm { S ^ { 3 } K L }$ under the 25% supervision level. As shown in Fig. 6, our method consistently achieves more accurate keypoint predictions, with the predicted blue points aligning closely with the ground truth green points. From columns 1–3, the baseline fails to perceive the hyoid bone and instead memorizes average training coordinates rather than tracking its true anatomical location. In contrast, $\mathrm { S ^ { 3 } K L }$ demonstrates better performance, highlighting the effectiveness of our approach in capturing anatomical structures and improving localization accuracy.

## V. CONCLUSION

We introduce the VFSSKep dataset and the $\mathrm { S ^ { 3 } K I }$ method for semi-supervised VFSS keypoint localization. By mitigating spatial bias with structure-aware priors and consistency learning, our method learns robust anatomical semantics rather than memorizing coordinates. Experiments show that it outperforms fully supervised methods with only 25% labeled data, providing a data-efficient solution for clinical kinematic analysis.

[1] N. Rommel and S. Hamdy, “Oropharyngeal dysphagia: manifestations and diagnosis,” Nature reviews Gastroenterology & hepatology, vol. 13, no. 1, pp. 49–59, 2016.

[2] S. Carrion, A. Costa, O. Ortega, E. Verin, P. Clav´ e, and A. Laviano,´ “Complications of oropharyngeal dysphagia: malnutrition and aspiration pneumonia,” Dysphagia: Diagnosis and treatment, pp. 823–857, 2019.

[3] J. T. Lee, E. Park, and T.-D. Jung, “Automatic detection of the pharyngeal phase in raw videos for the videofluoroscopic swallowing study using efficient data collection and 3d convolutional networks,” Sensors, vol. 19, no. 18, p. 3873, 2019.

[4] Y. J. Na, J. S. Jang, K. H. Lee, Y. J. Yoon, M. S. Chung, and S. H. Han, “Thyroid cartilage loci and hyoid bone analysis using a video fluoroscopic swallowing study (vfss),” Medicine, vol. 98, no. 30, p. e16349, 2019.

[5] S. A. Kraaijenga, L. van der Molen, W. D. Heemsbergen, G. B. Remmerswaal, F. J. Hilgers, and M. W. van den Brekel, “Hyoid bone displacement as parameter for swallowing impairment in patients treated for advanced head and neck cancer,” European Archives of Oto-Rhino-Laryngology, vol. 274, pp. 597–606, 2017.

[6] Z. Zhang, S. Mao, J. Coyle, and E. Sejdic, “Automatic annotation´ of cervical vertebrae in videofluoroscopy images via deep learning,” Medical image analysis, vol. 74, p. 102218, 2021.

[7] M.-Y. Hsiao, C.-H. Weng, Y.-C. Wang, S.-H. Cheng, K.-C. Wei, P.- Y. Tung, J.-Y. Chen, C.-Y. Yeh, and T.-G. Wang, “Deep learning for automatic hyoid tracking in videofluoroscopic swallow studies,” Dysphagia, vol. 38, no. 1, pp. 171–180, 2023.

[8] J. A. Logemann, “Swallowing disorders,” Best practice & research Clinical gastroenterology, vol. 21, no. 4, pp. 563–573, 2007.

[9] E. C. Goldfield, C. Buonomo, K. Fletcher, J. Perez, S. Margetts, A. Hansen, V. Smith, S. Ringer, M. J. Richardson, and P. H. Wolff, “Premature infant swallowing: patterns of tongue-soft palate coordination based upon videofluoroscopy,” Infant Behavior and Development, vol. 33, no. 2, pp. 209–218, 2010.

[10] R. Xie, C. Wang, W. Zeng, and Y. Wang, “An empirical study of the collapsing problem in semi-supervised 2d human pose estimation,” in ICCV, 2021, pp. 11 240–11 249.

[11] L. Huang, Y. Li, H. Tian, Y. Yang, X. Li, W. Deng, and J. Ye, “Semisupervised 2d human pose estimation driven by position inconsistency pseudo label correction module,” in CVPR, June 2023, pp. 693–703.

[12] Z. Ren, V. S. Dinh, P.-M. Wong, C.-B. Chng, J. J.-Y. Too, T.-W. Foong, W. N.-H. Loh, and C.-K. Chui, “G2lcps: End-to-end semi-supervised landmark prediction with global-to-local cross pseudo supervision for airway difficulty assessment,” Computers in Biology and Medicine, vol. 183, p. 109246, 2024.

[13] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, 2024.

[14] K. Pearson, “Liii. on lines and planes of closest fit to systems of points in space,” The London, Edinburgh, and Dublin philosophical magazine and journal of science, vol. 2, no. 11, pp. 559–572, 1901.

[15] F. Xia, P. Wang, X. Chen, and A. L. Yuille, “Joint multi-person pose estimation and semantic part segmentation,” in CVPR, 2017, pp. 6769– 6778.

[16] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023, pp. 4015–4026.

[17] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, no. 1, p. 654, 2024.

[18] S. Fu, M. Hamilton, L. E. Brandt, A. Feldmann, Z. Zhang, and W. T. Freeman, “Featup: A model-agnostic framework for features at any resolution,” in ICLR, 2024.

[19] M. Noroozi and P. Favaro, “Unsupervised learning of visual representations by solving jigsaw puzzles,” in ECCV, 2016, pp. 69–84.

[20] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in ECCV, 2018, pp. 466–481.

[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[22] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.

# Supplementary of “Structure-aware Keypoint Localization for Videofluoroscopic Swallowing Study”

I. RELATED WORK

## A. Videofluoroscopic Swallowing Study

Videofluoroscopic Swallowing Study (VFSS) is the gold standard for diagnosing swallowing disorders [1], providing dynamic X-ray imaging of the swallowing process. In VFSS, quantitative diagnosis relies on temporal and kinematic analysis. Temporal analysis [2], [3] involves identifying the start and end times of swallowing actions or micro-actions in the video, while kinematic analysis [4], [5] requires localizing anatomical landmarks to extract kinematic parameters associated with swallowing. Existing studies have explored various approaches for kinematic analysis. For instance, Kellen et al. [6] manually label the hyoid region and use Sobel edge detection to track its motion. Zhang et al. [4] propose a deep learning-based method to localize keypoints of cervical vertebrae, while Hsiao et al. [5] develop a deep learning-based approach to localize the hyoid bone and cervical vertebrae. However, these studies focus on limited keypoints (e.g., cervical vertebrae or hyoid regions) and neglect critical areas like the soft palate [7]. In contrast, we introduce a new keypoint localization dataset for VFSS, annotating 8 categories of keypoints, including the soft palate, cervical vertebrae, and hyoid regions. Building on this dataset, we propose a semi-supervised keypoint localization framework for VFSS to further improve localization performance.

## B. Keypoint Localization

Keypoint Localization has been widely applied in various domains, including human pose estimation [8]–[16], facial landmark detection [17], [18]. Keypoint localization can be broadly categorized into regression-based methods [8]–[10] and heatmap-based methods [11]–[16], depending on the type of supervision signal. Regression-based methods directly predict the 2D coordinates of keypoints by learning the mapping from images to these coordinates. Toshev et al. [8] were the first to use deep convolutional networks for pose estimation tasks, employing AlexNet [19] as the backbone network and proposing a deep convolutional regressor named DeepPose to predict keypoint coordinates. Carreira et al. [9] employed GoogLeNet [20] as the backbone network and introduced an Iterative Error Feedback network, a self-correcting model that progressively improves the prediction of keypoint coordinates by feeding prediction errors back into the input space. However, directly learning the mapping from images to sparse keypoint coordinates is quite challenging, and the performance of such methods often falls short compared to heatmap-based methods. Heatmap-based methods do not directly predict the 2D coordinates of keypoints; instead, they predict heatmaps for each keypoint, where the ground truth heatmap is typically generated by a 2D Gaussian distribution centered at the keypoint location. Newell et al. [13] proposed the Stacked Hourglass model, whose hourglass-shaped structure enhances the receptive field while minimizing computational demand for pose estimation. Xiao et al. [14] proposed the SimpleBaseline model, which employs ResNet [21] as the backbone network and adds only a few transposed convolutional layers to generate keypoint heatmaps. Sun et al. [15] recognized the benefits of high-resolution features for keypoint localization and innovatively proposed the High-Resolution Network (HRNet). This model maintains high-resolution feature representations throughout the network, gradually incorporates sub-networks of decreasing resolutions, and integrates their features at multiple scales to form a multi-resolution feature map. In addition to CNN-based methods, transformers have shown strong performance in keypoint localization. For example, ViTPose [16] leverages self-attention to capture long-range dependencies and achieves superior performance. In this paper, we utilize SimpleBaseline [14] as the main framework of keypoint localization to extract features following [22]. Our method allows for the replacement of the keypoint localization framework with various network architectures (e.g., ViTPose [16]) in semi-supervised training.

## C. Semi-supervised Keypoint Localization

Semi-supervised keypoint localization employs pseudolabeling [23], [24] and consistency regularization [22], [25]– [27] paradigms. DataDistill [23] uses ensembled predictions from a teacher model on transformed unlabeled images as pseudo-labels to train a student model, while PLACL [24] introduces a reinforcement learning-based strategy for pseudolabel selection. However, pseudo-labeling is often limited by the fixed teacher model pre-trained on labeled data. To address this, Xie et al. [22] propose consistency regularization with an easy-hard augmentation strategy to avoid the collapsing problem (i.e., predicting all pixels as background). SSKL [25] jointly optimizes keypoint heatmaps and category-level representations using supervised and unsupervised constraints, and Huang et al. [26] refine pseudo-labels by ensembling the least inconsistent predictions from two teacher networks. Yu et al. [28] introduce a denoising scheme to generate reliable pseudo-heatmaps. G2LCPS [27] further enhances semisupervised keypoint localization with a global filter for selecting unlabeled data and a local filter for removing unreliable pseudo heatmaps. However, these methods ignore anatomical cues and structural relationships vital for keypoint localization in VFSS. Our $\mathrm { S ^ { 3 } K I }$ L bridges this gap by leveraging structural priors to improve robustness and accuracy.

TABLE I: Demographic distribution of VFSSKep subjects. Values are percentages; n: number of subjects.
<table><tr><td></td><td>Unlabeled  $( n = 6 2 )$ </td><td>Train  $( n = 2 8 )$ </td><td>Val  $( n = 9 )$ </td><td>Test  $( n = 2 8 )$ </td></tr><tr><td>Male</td><td>37.1%</td><td>39.3%</td><td>55.6%</td><td>39.3%</td></tr><tr><td>Female</td><td>62.9%</td><td>60.7%</td><td>44.4%</td><td>60.7%</td></tr><tr><td>Age 10–30</td><td>9.7%</td><td>7.1%</td><td>33.3%</td><td>7.1%</td></tr><tr><td>Age 30–50</td><td>25.8%</td><td>25.0%</td><td>11.1%</td><td>21.4%</td></tr><tr><td>Age 50–70</td><td>41.9%</td><td>42.9%</td><td>44.4%</td><td>46.4%</td></tr><tr><td>Age 70+</td><td>22.6%</td><td>25.0%</td><td>11.1%</td><td>25.0%</td></tr></table>

## II. DATASET CONSTRUCTION DETAILS

Dataset Acquisition Details. We use two digital fluoroscopic systems to enhance clinical relevance and diversity: 1) Lanmage Dynamic Digital Radiography Machine (Athena Plus 7500; Shenzhen Lanmage Medical Technology Co., Ltd., China), accounting for 65% of the videos, and 2) Gastrointestinal X-ray Machine (Toshiba DBA-300; Toshiba Co. Ltd., Tokyo, Japan), accounting for the remaining 35%. All fluoroscopic videos were recorded at 30 FPS, and frames were converted into grayscale PNG format after extraction.

Dataset Statistics. We collect 288 videos and curate them based on specific criteria, including clear visualization of swallowing phases and diverse swallowing patterns. From these, we extract 13,895 frames from 210 videos, focusing on active swallowing segments. These frames are divided into three subsets: 8,302 images (from 131 videos) for the labeled training set, 1,518 images (from 16 videos) for the validation set, and 4,075 images (from 63 videos) for the test set. We maintain a 6:1:3 ratio for the training, validation, and test sets, ensuring that each subset includes images from distinct individuals to prevent data overlap. The remaining 78 videos constitute the unlabeled dataset. To ensure diversity, both active swallowing and non-swallowing segments are included, with the latter constituting the majority of the duration. Since non-swallowing segments contain relatively static frames, we sample one frame every ten frames, resulting in 39,838 images. After filtering out frames with poor visibility of the oral cavity or throat, as well as those with excessive motion blur, we finalize the unlabeled dataset with 31,536 clean images.

Demographic Information. Table I summarizes the demographic distribution of subjects in VFSSKep. The dataset covers a wide age range (10–70+ years) and maintains a balanced sex distribution across training, validation, test, and unlabeled subsets, ensuring diversity for model development and evaluation.

TABLE II: An example implementation of the SAB prediction head using ResNet18 as the backbone. The deconvolutional layers $( \mathrm { d e c _ { 1 } , d e c _ { 2 } , d e c _ { 3 } ) }$ progressively upsample the features, and a final $1 \times 1$ convolution (pred) produces the output.
<table><tr><td>Stage</td><td>Architecture</td><td>Output sizes  $C \times H \times W$ </td></tr><tr><td>input</td><td></td><td> $5 1 2 \times 8 \times 8$ </td></tr><tr><td> ${ \mathrm { d e c } } _ { 1 }$ </td><td> $\mathrm { d e c o n v \ 4 \times 4 , 2 5 6 }$   $( \mathrm { s t r i d e } { = } 2 , \mathrm { p a d d i n g { = } 1 ) }$ </td><td> $2 5 6 \times 1 6 \times 1 6$ </td></tr><tr><td> ${ \mathrm { d e c } } _ { 2 }$ </td><td> $\mathrm { d e c o n v \ 4 \times 4 , 2 5 6 }$   $( \mathrm { s t r i d e { = } } 2 , \mathrm { p a d d i n g { = } 1 ) }$ </td><td> $2 5 6 \times 3 2 \times 3 2$ </td></tr><tr><td> $\mathrm { { d e c } _ { 3 } }$ </td><td> $\mathrm { d e c o n v \ 4 \times 4 , 2 5 6 }$   $( \mathrm { s t r i d e { = } } 2 , \mathrm { p a d d i n g { = } 1 ) }$ </td><td> $2 5 6 \times 6 4 \times 6 4$ </td></tr><tr><td>pred</td><td> $\mathrm { c o n v \ 1 \times 1 , 3 8 4 }$   $( \mathrm { s t r i d e } { = } 1 , \mathrm { p a d d i n g { = } 0 ) }$ </td><td> $3 8 4 \times 6 4 \times 6 4$ </td></tr></table>

Annotation Tool. We utilized the open-source CVAT tool<sup>1</sup>. A cross-validation process was implemented where annotations were reviewed by independent experts to ensure consistency.

Ethical Approval and Informed Consent. All videos were collected from the Third Affiliated Hospital of Sun Yat-Sen University. This study was approved by the hospital’s Ethics Committee (approval number: [2018]02-374-01). Informed consent was waived due to the retrospective nature of the study and the use of anonymized data.

## III. MORE EXPERIMENTAL RESULTS

## A. More Implementation Details

Following SemiPose [22], we employ SimpleBaseline [14] as the keypoint localization network, utilizing ResNet18 [21] as its backbone (shared encoder). We train our model on 8 TITAN Xp GPUs using PyTorch. To compute MED across devices, we normalize each image using its original pixel spacing and resize it to $2 5 6 \times 2 5 6$ , resulting in a unified scale of 0.6 mm/pixel. For the prediction head of Structure-Aware Branch (SAB), we employ a straightforward yet efficient design. It consists of three deconvolutional layers that enhance the resolution of the feature maps, along with a convolutional layer that generates the feature maps. The structure of the deconvolutional layers is inspired by the decoder architecture of SimpleBaseline. The detailed architecture is presented in Table II, with the input coming from the output of the final convolutional stage in ResNet. Our Structure Parsing Module (SPM) extracts low-resolution features via a fixed DINOv2 and passes them, along with the original image, through a 4-layer Joint Bilateral Upsampler (JBU) from FeatUp [29]. Each JBU layer upsamples by a factor of 2, achieving 16× upsampling for high-resolution output. In our experiments,we train for 100 epochs using the Adam optimizer with an initial learning rate of $1 0 ^ { - 3 }$ , decayed to $1 0 ^ { - 4 }$ at epoch 60 and $1 0 ^ { - 5 }$ at epoch 90. The batch size is 64. For data augmentation, we apply easy augmentation (e.g., Affine Transformation with rotation angles sampled from $[ - 3 0 ^ { \circ } , 3 0 ^ { \circ } ] )$ to the teacher network and hard augmentation (e.g., Affine Transformation with rotation angles sampled from $[ - 6 0 ^ { \circ } , 6 0 ^ { \circ } ]$ combined with our proposed Block Shuffle) to the student network. Following SemiPose [22], we set the supervised loss weight $\lambda _ { 1 }$ and unsupervised loss weight $\lambda _ { 2 }$ to 0.5 and 1.0, respectively. In addition to the default hyperparameters, we set the feature alignment loss weight $\lambda _ { 3 } = 1 0 ^ { - 3 }$ . The similarity threshold $\alpha$ is 0.85, and the block size of Block Shuffle is $6 4 \times 6 4$ . We analyze the effects of these hyperparameters in hyperparameter analyses.

TABLE III: Effect of loss weight $\lambda _ { 3 }$ for feature alignment loss. PCK@0.02 (%) and Mean Euclidean Distance (MED, in mm) are reported.
<table><tr><td rowspan="2"> $\lambda _ { 3 }$ </td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td>69.2</td><td>3.60</td><td>73.7</td><td>4.08</td><td>75.3</td><td>3.14</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>78.1</td><td>3.56</td><td>80.0</td><td>2.90</td><td>81.1</td><td>2.75</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>78.8</td><td>2.88</td><td>82.1</td><td>2.64</td><td>83.6</td><td>2.48</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>78.4</td><td>3.62</td><td>80.6</td><td>3.32</td><td>82.4</td><td>2.56</td></tr></table>

TABLE IV: Effect of the similarity threshold α in the cosine similarity loss $\mathcal { L } _ { c o s }$ . PCK@0.02 (%) and Mean Euclidean Distance (MED, in mm) are reported.
<table><tr><td rowspan="2">α</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td>0.7</td><td>78.4</td><td>2.94</td><td>82.0</td><td>2.87</td><td>83.4</td><td>2.49</td></tr><tr><td>0.85</td><td>78.8</td><td>2.87</td><td>82.1</td><td>2.64</td><td>83.6</td><td>2.48</td></tr><tr><td>1.0</td><td>78.7</td><td>3.66</td><td>81.6</td><td>2.87</td><td>82.8</td><td>2.58</td></tr></table>

## B. Hyperparameter Analyses

Effect of loss weight $\lambda _ { 3 } .$ . Tab. III shows the model’s performance with varying loss weights $\lambda _ { 3 }$ for the auxiliary task. A high weight $( e . g . , \ \lambda _ { 3 } \ = \ 1 0 ^ { - 1 } )$ overemphasizes the auxiliary task, degrading performance on the primary task. A low weight $( e . g . , \lambda _ { 3 } = 1 0 ^ { - 4 } )$ diminishes the auxiliary task’s impact. The best performance occurs at $\lambda _ { 3 } = 1 0 ^ { - 3 }$ , balancing both tasks.

Effect of similarity threshold α. Tab. IV shows how different similarity threshold values α affect the calculation of the Structure-Aware Learning loss. When we set α a low threshold $( e . g . , 0 . 7 )$ , the model excludes a large proportion of features from the Structure-Aware Learning loss, potentially hindering its alignment learning. Conversely, when we set a high threshold (e.g., 1.0), the model includes all features, even those already well-optimized, which may lead to overfitting. The optimal performance is achieved at $\alpha = 0 . 8 5$ , balancing relevant feature inclusion for better generalization.

Effect of input block size on the Block Shuffle operation. The Block Shuffle operation rearranges image blocks before inputting them into the encoder. A small block size (32×32) causes excessive shuffling, disrupting local spatial relationships and lowering performance. A large block size (128×128) provides insufficient shuffling, failing to prevent overfitting. As shown in Tab. V, the optimal block size of 64×64 strikes a balance, preserving spatial structures while enabling effective shuffling, resulting in the best performance across all supervision levels.

TABLE V: Effect of input block size on the Block Shuffle operation. PCK@0.02 (%) and Mean Euclidean Distance (MED, in mm) are reported.
<table><tr><td rowspan="2">Size</td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td><td>PCK↑</td><td>MED↓</td></tr><tr><td>32</td><td>78.5</td><td>2.95</td><td>81.8</td><td>2.75</td><td>82.6</td><td>2.68</td></tr><tr><td>64</td><td>78.8</td><td>2.87</td><td>82.1</td><td>2.64</td><td>83.6</td><td>2.48</td></tr><tr><td>128</td><td>78.4</td><td>3.65</td><td>81.4</td><td>3.13</td><td>82.3</td><td>2.60</td></tr></table>

TABLE VI: Effect of different backbones.

<table><tr><td rowspan="2">Backbone Param FLOPs</td><td rowspan="2"></td><td colspan="2">25%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td></td><td>PCK↑ MED↓</td><td>PCK↑ MED↓</td><td></td><td>PCK↑ MED↓</td><td></td></tr><tr><td>ResNet1815.6M ViT-B</td><td>2.4G 87.6M 21.6G</td><td>78.8 79.9</td><td>2.88 2.74</td><td>82.1 82.3</td><td>2.64 2.55</td><td>83.6 83.5</td><td>2.48 2.51</td></tr></table>

## C. Extension Experiments

Effect of different backbones. As shown in Tab. VI, employing a stronger backbone ViT-B [30] (adopted from ViTPose [16]) yields better performance under limited supervision (25% and 50%), demonstrating its advantage in capturing global structural information. Interestingly, when full supervision is available, the lightweight ResNet18 slightly outperforms ViT-B. We attribute this to the relatively modest dataset size, where the strong capacity of ViT-B is not fully exploited and may even lead to overfitting, while ResNet18 benefits from the inherent inductive bias of convolutional networks that better capture local structural patterns. Considering the significantly higher computational and memory cost of ViT-B (87.6M/21.6G vs. 15.6M/2.4G), we mainly adopt ResNet18, as it provides a better balance between accuracy and efficiency for resource-constrained medical scenarios.

## IV. MORE DISCUSSIONS

## A. Clinical Implications

Accurate and efficient keypoint localization in VFSS has direct clinical value. Anatomical landmarks underpin quantitative kinematic measures, such as hyoid displacement and soft palate elevation, which are essential for dysphagia assessment. By reducing reliance on labor-intensive annotations, $\mathrm { S ^ { 3 } K I }$ lowers deployment barriers in resource-constrained clinical settings. Its plug-and-play design further ensures compatibility with existing pipelines, facilitating seamless integration into current workflows. Moreover, by leveraging both swallowing and non-swallowing phases, $\mathrm { S ^ { 3 } K L }$ shows potential for automated pre-screening, helping reduce clinician workload and improve efficiency.

![](images/62be7c32067045cd16a9ef9d4f1b1e2f172a584bfdb7a45499904de6518e894b.jpg)

<table><tr><td colspan="4">Onset and Offset of Swallow Micro-events</td></tr><tr><td>Oral Delivery</td><td>3.31</td><td>~</td><td>4.59</td></tr><tr><td>Soft Palate Lift</td><td>4.34</td><td>~</td><td>S 5.58 s</td></tr><tr><td>Hyoid Exercise</td><td>4.51 ~</td><td>5.96</td><td>S</td></tr><tr><td>UES Open</td><td>4.92 ~</td><td>5.74</td><td>s</td></tr><tr><td>Throat Swallow</td><td>4.65</td><td>~ 5.73</td><td>S</td></tr><tr><td>Throat Transport</td><td>4.33</td><td>~ 5.63</td><td>s</td></tr><tr><td>Laryngeal Vestibule Closure</td><td>4.86</td><td>~ 5.61</td><td>s</td></tr><tr><td>Whole Swallow</td><td>3.31</td><td>~ 5.96</td><td>s</td></tr></table>

![](images/dee2f5e5a67c9910a9ab1404f7294932b4d936e92ef65fef04342cea131f2887.jpg)  
Fig. 1: Partial screenshot of the clinical prototype system.

To bridge the gap to real-world application, we have developed a clinician-in-the-loop prototype system integrating our framework (see Fig. 1). The system automatically extracts kinematic parameters (e.g., hyoid trajectories) and temporal metrics (e.g., swallow micro-events) to support diagnostic workflows. While large-scale deployment requires formal ethical approval, this prototype demonstrates the practical clinical relevance of our method.

## B. Limitations and Future Directions

Despite the above contributions, there remain opportunities for further improvement. First, while VFSSKep provides more comprehensive annotations than existing datasets, its scale is still modest compared to large benchmarks in computer vision. Expanding the dataset in size and diversity, including multicenter data collection, would enhance generalizability. Second, our framework currently operates on static frames without explicitly modeling temporal swallowing dynamics, which are known to yield additional diagnostic insights. Incorporating temporal modules or sequence-based architectures represents a promising direction to further strengthen clinical applicability.

## REFERENCES

[1] Y. J. Na, J. S. Jang, K. H. Lee, Y. J. Yoon, M. S. Chung, and S. H. Han, “Thyroid cartilage loci and hyoid bone analysis using a video fluoroscopic swallowing study (vfss),” Medicine, vol. 98, no. 30, p. e16349, 2019.

[2] J. T. Lee, E. Park, and T.-D. Jung, “Automatic detection of the pharyngeal phase in raw videos for the videofluoroscopic swallowing study using efficient data collection and 3d convolutional networks,” Sensors, vol. 19, no. 18, p. 3873, 2019.

[3] X. Ruan, M. Dai, Z. Chen, Z. You, Y. Zhang, Y. Li, Z. Dou, and M. Tan, “Temporal micro-action localization for videofluoroscopic swallowing study,” IEEE Journal of Biomedical and Health Informatics, 2023.

[4] Z. Zhang, S. Mao, J. Coyle, and E. Sejdic, “Automatic annotation´ of cervical vertebrae in videofluoroscopy images via deep learning,” Medical image analysis, vol. 74, p. 102218, 2021.

[5] M.-Y. Hsiao, C.-H. Weng, Y.-C. Wang, S.-H. Cheng, K.-C. Wei, P.- Y. Tung, J.-Y. Chen, C.-Y. Yeh, and T.-G. Wang, “Deep learning for automatic hyoid tracking in videofluoroscopic swallow studies,” Dysphagia, vol. 38, no. 1, pp. 171–180, 2023.

[6] P. M. Kellen, D. L. Becker, J. M. Reinhardt, and D. J. Van Daele, “Computer-assisted assessment of hyoid bone motion from videofluoroscopic swallow studies,” Dysphagia, vol. 25, pp. 298–306, 2010.

[7] J. A. Logemann, “Swallowing disorders,” Best practice & research Clinical gastroenterology, vol. 21, no. 4, pp. 563–573, 2007.

[8] A. Toshev and C. Szegedy, “Deeppose: Human pose estimation via deep neural networks,” in CVPR, Columbus, OH, USA, June 23-28, 2014. IEEE Computer Society, 2014, pp. 1653–1660.

[9] J. Carreira, P. Agrawal, K. Fragkiadaki, and J. Malik, “Human pose estimation with iterative error feedback,” in CVPR, Las Vegas, NV, USA June 27-30, 2016. IEEE Computer Society, 2016, pp. 4733–4742.

[10] X. Sun, J. Shang, S. Liang, and Y. Wei, “Compositional human pose regression,” in ICCV, Venice, Italy, October 22-29, 2017. IEEE Computer Society, 2017, pp. 2621–2630.

[11] J. Tompson, A. Jain, Y. LeCun, and C. Bregler, “Joint training of a convolutional network and a graphical model for human pose estimation,” in NeurIPS, December 8-13 2014, Montreal, Quebec, Canada, 2014, pp. 1799–1807.

[12] S. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh, “Convolutional pose machines,” in CVPR, Las Vegas, NV, USA, June 27-30, 2016. IEEE Computer Society, 2016, pp. 4724–4732.

[13] A. Newell, K. Yang, and J. Deng, “Stacked hourglass networks for human pose estimation,” in ECCV, Amsterdam, The Netherlands, October 11-14, 2016. Springer, 2016, pp. 483–499.

[14] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in ECCV, 2018, pp. 466–481.

[15] K. Sun, B. Xiao, D. Liu, and J. Wang, “Deep high-resolution representation learning for human pose estimation,” in CVPR, Long Beach, CA, USA, June 16-20, 2019. Computer Vision Foundation / IEEE, 2019, pp. 5693–5703.

[16] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “Vitpose: Simple vision transformer baselines for human pose estimation,” in NeurIPS, 2022.

[17] P. Gao, K. Lu, J. Xue, L. Shao, and J. Lyu, “A coarse-to-fine facial

landmark detection method based on self-attention mechanism,” IEEE Transactions on Multimedia, vol. 23, pp. 926–938, 2020.

[18] S. Mahpod, R. Das, E. Maiorana, Y. Keller, and P. Campisi, “Facial landmarks localization using cascaded neural networks,” Computer Vision and Image Understanding, vol. 205, p. 103171, 2021.

[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[20] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1–9.

[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[22] R. Xie, C. Wang, W. Zeng, and Y. Wang, “An empirical study of the collapsing problem in semi-supervised 2d human pose estimation,” in ICCV, 2021, pp. 11 240–11 249.

[23] I. Radosavovic, P. Dollar, R. B. Girshick, G. Gkioxari, and K. He,´ “Data distillation: Towards omni-supervised learning,” in CVPR, Salt Lake City, UT, USA, June 18-22, 2018. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 4119–4128.

[24] C. Wang, S. Jin, Y. Guan, W. Liu, C. Qian, P. Luo, and W. Ouyang, “Pseudo-labeled auto-curriculum learning for semisupervised keypoint localization,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview. net/forum?id=6Q52pZ-Th7N

[25] O. Moskvyak, F. Maire, F. Dayoub, and M. Baktashmotlagh, “Semi-supervised keypoint localization,” in International Conference on Learning Representations, 2021. [Online]. Available: https: //openreview.net/forum?id=yFJ67zTeI2

[26] L. Huang, Y. Li, H. Tian, Y. Yang, X. Li, W. Deng, and J. Ye, “Semisupervised 2d human pose estimation driven by position inconsistency pseudo label correction module,” in CVPR, June 2023, pp. 693–703.

[27] Z. Ren, V. S. Dinh, P.-M. Wong, C.-B. Chng, J. J.-Y. Too, T.-W. Foong, W. N.-H. Loh, and C.-K. Chui, “G2lcps: End-to-end semi-supervised landmark prediction with global-to-local cross pseudo supervision for airway difficulty assessment,” Computers in Biology and Medicine, vol. 183, p. 109246, 2024.

[28] Z. Yu, M. Wang, Y. Chen, P. Favaro, and D. Modolo, “Denoising and selecting pseudo-heatmaps for semi-supervised human pose estimation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 6280–6289.

[29] S. Fu, M. Hamilton, L. E. Brandt, A. Feldmann, Z. Zhang, and W. T. Freeman, “Featup: A model-agnostic framework for features at any resolution,” in ICLR, 2024.

[30] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.