# ProGeo-RC: Prototype-guided Semantic Reliability and Geometric Error Calibration for Semi-supervised Cardiac Image Segmentation

Yunfei Zhu<sup>1\*</sup>, Wenfeng Zhang<sup>1\*</sup>, Pan Zeng<sup>1</sup>, Xin Huang<sup>1</sup>, Tao Jia<sup>1</sup>, Weiting Shao<sup>2</sup>, Wei Hu<sup>1†</sup>, Jianming Hu<sup>3†</sup> <sup>1</sup>College of Computer and Information Science,

Research Center of Biomimetic Intelligence and Computational Biomedicine,

Chongqing Normal University

<sup>2</sup>School of Educational Sciences, Chongqing Normal University

<sup>3</sup>School of Physics and Electrical Engineering, Chongqing Normal University Chongqing, China

Abstract—Semi-supervised medical image segmentation aims to train reliable segmentation models using limited labeled data and abundant unlabeled data. However, existing methods typically rely on prediction confidence or model consistency to estimate pseudo-label reliability, but these criteria are insufficient to characterize both semantic reliability and geometric plausibility. To address this issue, we propose ProGeo-RC, a prototype-guided semantic reliability and geometric error calibration framework for semi-supervised medical image segmentation. ProGeo-RC consists of two key modules: Prototype-guided Semantic Reliability Estimation (PSRE) and Structured Semantic-Geometric Error Calibration (S-SGEC). PSRE uses probabilistic class prototypes to estimate the semantic reliability of unlabeled predictions and calibrate pixel-wise teacher pseudo labels, while S-SGEC introduces signed-distance-field-based geometric constraints to refine ambiguous boundaries and suppress spurious foreground responses outside target structures. Experiments on the ACDC and MVAA-CT datasets show that ProGeo-RC achieves better performance compared with representative semi-supervised segmentation methods. Under the 10% labeled setting on ACDC, ProGeo-RC improves DSC and Jaccard by 0.37% and 0.49% over the corresponding second-best results, while reducing 95HD and ASD by 0.61 and 0.21, respectively. These results indicate that joint semantic reliability estimation and geometric error calibration can provide more reliable pseudo supervision for semi-supervised medical image segmentation.

Index Terms—Semi-supervised Learning, Cardiac Image Segmentation, Prototype Learning, Geometric Calibration

## I. INTRODUCTION

Computer-aided diagnosis systems rely on accurate anatomical segmentation for disease assessment, treatment planning, and postoperative follow-up. In cardiac image analysis, reliable segmentation supports ventricular volume quantification, ejection fraction estimation, valve morphology analysis, and personalized treatment planning. Deep learning networks, including U-Net, V-Net, 3D U-Net, and nnU-Net, have achieved strong performance in medical image segmentation [1]–[4] and cardiac segmentation [5]–[9]. However, they usually require extensive pixel-level annotations, which are labor-intensive, time-consuming, and clinically demanding. Learning from limited labeled data and abundant unlabeled data therefore remains a key challenge in semi-supervised cardiac image segmentation.

Existing semi-supervised segmentation methods mainly exploit unlabeled data through pseudo-label learning, consistency regularization, teacher-student frameworks, and cross pseudo supervision [10]–[13]. Although these methods have demonstrated the effectiveness of exploiting unlabeled data, their performance largely depends on the reliability of pseudo supervision. Common strategies assess pseudo-label reliability using prediction confidence or model consistency, either by selecting high-confidence regions or by enforcing consistency across perturbations, branches, or teacher-student models [11], [12]. However, these criteria are imperfect surrogates for true reliability. High-confidence predictions may still deviate from the semantic distributions of their predicted classes [14], while consistent predictions do not necessarily guarantee geometrically plausible structures [15]–[17]. Therefore, the key challenge in semi-supervised cardiac image segmentation lies not only in exploiting unlabeled data, but also in identifying reliable pseudo supervision from unlabeled predictions.

This challenge motivates assessing pseudo-supervision reliability from both semantic and geometric perspectives. Semantically, models may produce overconfident errors, where high-confidence pixels do not match the feature distribution of the predicted class [14], [18], [19]; agreement between two models also does not necessarily imply feature-space reliability. Geometrically, cardiac structures exhibit anatomical continuity and clear boundaries, making confidence-based filtering insufficient for boundary ambiguity, isolated false positives, and spurious responses outside target structures [20]– [22]. Although prototypes, intra-class distribution modeling, boundary-aware learning, and distance-based constraints can improve semantic discrimination and structural plausibility, they are usually used as auxiliary regularization or representation learning constraints. These observations suggest that reliable pseudo supervision should jointly consider feature-space semantic consistency and image-space geometric plausibility.

To this end, we propose ProGeo-RC, a prototype-guided semantic reliability and geometric error calibration framework for semi-supervised cardiac image segmentation. Unlike methods relying mainly on prediction confidence or model consistency, ProGeo-RC assesses pseudo-supervision reliability from two complementary aspects: whether pixel features match the predicted class distribution in feature space and whether predicted regions are geometrically plausible in image space. By jointly modeling semantic consistency and geometric plausibility, ProGeo-RC provides more reliable pseudo supervision and reduces the accumulation of erroneous pseudo labels.

Our main contributions are summarized as follows:

• We propose ProGeo-RC, a prototype-guided framework for semi-supervised cardiac image segmentation that jointly models pseudo-supervision reliability from complementary semantic and geometric perspectives, thereby mitigating the accumulation of unreliable pseudo labels during training.

• We develop Prototype-guided Semantic Reliability Estimation (PSRE) and Structured Semantic-Geometric Error Calibration (S-SGEC) to improve pseudo-supervision quality. PSRE evaluates unlabeled predictions using classwise probabilistic prototypes, while S-SGEC uses signed distance fields to calibrate boundary ambiguity and suppress structurally implausible foreground responses.

• Extensive experiments on the ACDC and MVAA-CT datasets demonstrate that ProGeo-RC consistently outperforms representative semi-supervised segmentation methods under different annotation settings. Ablation studies and qualitative visualizations further verify the effectiveness and generalizability of the proposed framework.

## II. METHOD

## A. Overview

We propose ProGeo-RC for semi-supervised cardiac image segmentation under limited annotation. As shown in Fig. 1, the framework contains two student networks with identical architectures but independent parameters. Given a labeled dataset $\mathcal { D } _ { l } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \hat { N } _ { l } }$ and an unlabeled dataset $\mathcal { D } _ { u } = \{ x _ { i } \} _ { i = 1 } ^ { N _ { u } }$ the k-th student network $f _ { k }$ outputs a segmentation probability map $P _ { k }$ and a feature map $F _ { k }$

$$
( P _ { k } , F _ { k } ) = f _ { k } ( x ) , \quad k \in \{ 1 , 2 \} .\tag{1}
$$

The probability $P _ { k , c } ( i )$ denotes the prediction of the k-th student for class c at position i, where $c \in \{ 0 , 1 , . . . , C - 1 \}$ The background class is denoted by $c = 0$ , and the foreground anatomical classes are denoted by $\mathcal { C } _ { f } = \{ 1 , \ldots , C - 1 \}$

ProGeo-RC calibrates unlabeled pseudo-supervision from semantic and geometric perspectives. PSRE models foreground class distributions with probabilistic prototypes to generate a prototype-calibrated teacher pseudo label $\tilde { P }$ and update the prototype memory with reliable feature-label pairs. S-SGEC then constructs signed distance fields from $\tilde { P }$ to regularize ambiguous boundaries and suppress abnormal foreground responses during student optimization.

## B. Prototype-guided Semantic Reliability Estimation

PSRE estimates whether an unlabeled foreground prediction is reliable in the feature space. Because softmax confidence and dual-student agreement may retain overconfident errors in locally similar cardiac structures, we introduce foreground class-wise probabilistic prototypes as semantic reliability estimators. Background prototypes are not maintained because background regions are heterogeneous and less suitable for compact prototype modeling.

For the k-th student network and foreground class $c \in { \mathcal { C } } _ { f }$ , the probabilistic prototype memory is defined as:

$$
\mathcal { M } _ { k , c } = \{ \mu _ { k , c } , \sigma _ { k , c } ^ { 2 } \} , \quad c \in \mathcal { C } _ { f } .\tag{2}
$$

The mean $\mu _ { k , c }$ and variance $\sigma _ { k , c } ^ { 2 }$ describe the class center and intra-class feature dispersion, respectively. During training, $\mathcal { M } _ { k , c }$ is updated using labeled feature-ground-truth pairs and reliable unlabeled feature-pseudo-label pairs selected by the semantic reliability gate.

Given an unlabeled pixel feature $\hat { F } _ { k } ( i )$ , its prototype response to foreground class c is computed as:

$$
Z _ { k , c } ^ { \mathrm { p r o t o } } ( i ) = - \frac { 1 } { d } \sum _ { m = 1 } ^ { d } \frac { \left( \hat { F } _ { k , m } ( i ) - \mu _ { k , c , m } \right) ^ { 2 } } { \sigma _ { k , c , m } ^ { 2 } + \epsilon } , \quad c \in \mathcal { C } _ { f } ,\tag{3}
$$

where d is the feature dimension and ϵ is a numerical stabilizer. A larger $Z _ { k , c } ^ { \mathrm { p r o t o } } ( i )$ indicates stronger consistency between the pixel feature and the prototype distribution of class c. The responses are normalized over foreground classes:

$$
Q _ { k , c } ( i ) = \frac { \exp \Big ( Z _ { k , c } ^ { \mathrm { p r o t o } } ( i ) \Big ) } { \sum _ { r \in \mathscr { C } _ { f } } \exp \Big ( Z _ { k , r } ^ { \mathrm { p r o t o } } ( i ) \Big ) } , \quad c \in \mathscr { C } _ { f } .\tag{4}
$$

$Q _ { k , c } ( i )$ denotes the prototype support for foreground class $c ,$ while the background probability is retained from the segmentation prediction.

We denote the predicted class and prediction confidence of the k-th student as:

$$
\hat { y } _ { k } ( i ) = \underset { c } { \operatorname { a r g m a x } } P _ { k , c } ( i ) , \quad p _ { k } ( i ) = \underset { c } { \operatorname* { m a x } } P _ { k , c } ( i ) ,\tag{5}
$$

and define the prototype-predicted foreground class and prototype confidence as:

$$
\hat { y } _ { k } ^ { \mathrm { p r o t o } } ( i ) = \underset { c \in \mathcal { C } _ { f } } { \operatorname { a r g m a x } } Q _ { k , c } ( i ) , \quad q _ { k } ( i ) = \underset { c \in \mathcal { C } _ { f } } { \operatorname* { m a x } } Q _ { k , c } ( i ) .\tag{6}
$$

![](images/9d0a06d63d4f0a35885dfde9e4ceb8f184ba705e106fddee8763b5d253bb3912.jpg)  
Fig. 1. Overall framework of ProGeo-RC. Two student networks generate predictions and features from labeled and unlabeled images. PSRE estimate semantic reliability using probabilistic prototypes and a semantic gate, while S-SGEC applies SDF-based calibration to boundary regions and far-background false positives.

The semantic reliability gate is formulated as:

$$
\begin{array} { r l } & { R _ { \mathrm { s e m } } ^ { k } ( i ) = \mathbb { I } [ \hat { y } _ { k } ( i ) \in \mathcal { C } _ { f } ] \mathbb { I } [ p _ { k } ( i ) > \tau _ { p } ] } \\ & { \quad \quad \cdot \mathbb { I } [ \hat { y } _ { k } ( i ) = \hat { y } _ { k } ^ { \mathrm { p r o t o } } ( i ) ] \mathbb { I } [ q _ { k } ( i ) > \tau _ { p } ] } \\ & { \quad \quad \cdot \mathbb { I } [ \hat { y } _ { 1 } ( i ) = \hat { y } _ { 2 } ( i ) ] . } \end{array}\tag{7}
$$

The indicator function is denoted by I[·], and $\tau _ { p }$ is the reliability threshold. This gate jointly evaluates foreground prediction, prediction confidence, prediction-prototype consistency, prototype confidence, and dual-student agreement. Pixels satisfying the gate are treated as reliable foreground feature-label pairs for prototype memory update.

For each foreground class $c \in \mathcal { C } _ { f }$ , the prototype update set is constructed as:

$$
\begin{array} { r l } & { \boldsymbol { \mathcal { S } } _ { k , c } ^ { l } = \{ \hat { F } _ { k } ( i ) \mid ( x , y ) \in \mathcal { D } _ { l } , y ( i ) = c \} , } \\ & { \boldsymbol { \mathcal { S } } _ { k , c } ^ { u } = \{ \hat { F } _ { k } ( i ) \mid x \in \mathcal { D } _ { u } , R _ { \mathrm { s e m } } ^ { k } ( i ) = 1 , \hat { y } _ { k } ( i ) = c \} , } \\ & { \boldsymbol { \mathcal { S } } _ { k , c } = \boldsymbol { \mathcal { S } } _ { k , c } ^ { l } \cup \boldsymbol { \mathcal { S } } _ { k , c } ^ { u } . } \end{array}\tag{8}
$$

When $\mathcal { S } _ { k , c }$ is non-empty, its batch-wise mean and variance are computed by:

$$
\begin{array} { r l } & { \bar { \mu } _ { k , c } = \displaystyle \frac { 1 } { | \mathcal { S } _ { k , c } | } \sum _ { \hat { F } _ { k } ( i ) \in \mathcal { S } _ { k , c } } \hat { F } _ { k } ( i ) , } \\ & { \bar { \sigma } _ { k , c } ^ { 2 } = \displaystyle \frac { 1 } { | \mathcal { S } _ { k , c } | } \sum _ { \hat { F } _ { k } ( i ) \in \mathcal { S } _ { k , c } } \left( \hat { F } _ { k } ( i ) - \bar { \mu } _ { k , c } \right) ^ { 2 } . } \end{array}\tag{9}
$$

The prototype memory is updated using exponential moving average:

$$
\begin{array} { r } { \mu _ { k , c }  \eta \mu _ { k , c } + ( 1 - \eta ) \bar { \mu } _ { k , c } , } \\ { \sigma _ { k , c } ^ { 2 }  \eta \sigma _ { k , c } ^ { 2 } + ( 1 - \eta ) \bar { \sigma } _ { k , c } ^ { 2 } , } \end{array}\tag{10}
$$

where $\eta$ denotes the EMA momentum. If no reliable sample of class c is available in the current mini-batch, $\mathcal { M } _ { k , c }$ remains unchanged. This reduces the risk of updating the prototype memory with unreliable pseudo labels.

PSRE further constructs a prototype-calibrated soft teacher pseudo label from the two student predictions. Instead of introducing an additional teacher network, we use a pixel-wise selector to choose the student branch with higher prediction confidence at each unlabeled pixel:

$$
\begin{array} { r l } & { \quad s ( i ) = \underset { k \in \{ 1 , 2 \} } { \mathrm { a r g m a x } } p _ { k } ( i ) , } \\ & { \quad P _ { c } ^ { T } ( i ) = P _ { s ( i ) , c } ( i ) , \quad c \in \{ 0 , 1 , \ldots , C - 1 \} , } \\ & { \quad Q _ { c } ^ { T } ( i ) = Q _ { s ( i ) , c } ( i ) , \quad c \in \mathcal { C } _ { f } . } \end{array}\tag{11}
$$

The selected branch index is $s ( i )$ . Thus, $P _ { c } ^ { T } ( i )$ denotes the pixel-wise teacher prediction probability, and $Q _ { c } ^ { T } ( i )$ denotes the prototype probability queried from the same selected branch. We define:

$$
\begin{array} { r } { \boldsymbol { { \hat { y } } } ^ { T } ( i ) = \underset { c } { \arg \operatorname* { m a x } } \ : P _ { c } ^ { T } ( i ) , } \\ { \boldsymbol { { \hat { y } } } ^ { \mathrm { p r o t o } , T } ( i ) = \underset { c \in \mathcal { C } _ { f } } { \arg \operatorname* { m a x } } \ : Q _ { c } ^ { T } ( i ) , } \\ { \boldsymbol { q } ^ { T } ( i ) = \underset { c \in \mathcal { C } _ { f } } { \arg \operatorname* { m a x } } \ : Q _ { c } ^ { T } ( i ) . } \end{array}\tag{12}
$$

The prototype calibration indicator is:

$$
m ( i ) = \mathbb { I } [ \hat { y } ^ { T } ( i ) \in \mathcal { C } _ { f } ] \mathbb { I } [ \hat { y } ^ { T } ( i ) = \hat { y } ^ { \mathrm { p r o t o } , T } ( i ) ] \mathbb { I } [ q ^ { T } ( i ) > \tau _ { p } ] .\tag{13}
$$

When the selected teacher prediction passes the foreground prototype-consistency criterion, its foreground probabilities are calibrated by prototype responses. To preserve the classifierderived background probability, we define the foreground probability mass as:

$$
\pi ^ { T } ( i ) = \sum _ { c \in \mathcal { C } _ { f } } P _ { c } ^ { T } ( i ) = 1 - P _ { 0 } ^ { T } ( i ) .\tag{14}
$$

The prototype-calibrated teacher is given by:

$$
\begin{array} { r } { \tilde { P } _ { 0 } ( i ) = P _ { 0 } ^ { T } ( i ) , } \end{array}\tag{15}
$$

and for $c \in \mathcal { C } _ { f }$

$$
\begin{array} { r l } & { \tilde { P } _ { c } ( i ) = ( 1 - m ( i ) ) P _ { c } ^ { T } ( i ) } \\ & { \qquad + m ( i ) \big [ ( 1 - \lambda ) P _ { c } ^ { T } ( i ) + \lambda \pi ^ { T } ( i ) Q _ { c } ^ { T } ( i ) \big ] . } \end{array}\tag{16}
$$

The coefficient λ controls the contribution of the prototype probability. Since $Q ^ { T }$ is normalized only over foreground classes, multiplication by $\pi ^ { T } ( i )$ preserves the foreground probability mass and keeps $\tilde { P }$ a valid class distribution. The output $\tilde { P } ( i ) = \{ \tilde { P } _ { c } ( i ) \} _ { c = 0 } ^ { C - 1 }$ is used for subsequent unlabeled supervision. The prototype memory is not updated by ${ \tilde { P } } ,$ but only by the reliable feature-label pairs selected by $\bar { R _ { \mathrm { s e m } } ^ { k } } .$

## C. Structured Semantic-Geometric Error Calibration

Semantic reliability alone does not ensure spatial plausibility, as pseudo-label errors often occur near ambiguous boundaries or outside target structures. Motivated by boundaryand distance-based constraints [20]–[22], S-SGEC constructs geometric supervision from the prototype-calibrated teacher prediction ${ \tilde { P } } .$

The soft teacher prediction is first converted into a hard pseudo label:

$$
\tilde { Y } ( i ) = \underset { c } { \mathrm { a r g m a x } } \tilde { P } _ { c } ( i ) .\tag{17}
$$

For each foreground class $c \in \mathcal { C } _ { f }$ , a signed distance field (SDF) is constructed as:

$$
S _ { c } = \mathrm { S D F } \left( \mathbb { I } [ \tilde { Y } = c ] \right) .\tag{18}
$$

$S _ { c } ( i ) < 0 , S _ { c } ( i ) \approx 0 .$ , and $S _ { c } ( i ) > 0$ indicate inside, boundaryadjacent, and outside regions of class $c ,$ respectively. If class c is absent from $\tilde { Y } .$ , its geometric weight is set to zero.

To obtain scale-consistent responses, the SDF is normalized by:

$$
\bar { S } _ { c } ( i ) = \frac { S _ { c } ( i ) } { \operatorname* { m a x } _ { j } | S _ { c } ( j ) | + \epsilon } .\tag{19}
$$

The boundary-region and far-background-region weights are defined as:

$$
W _ { c } ^ { b d } ( i ) = \left( 1 - | \bar { S } _ { c } ( i ) | \right) ^ { 2 } , \quad W _ { c } ^ { f a r } ( i ) = \left[ \operatorname* { m a x } ( \bar { S } _ { c } ( i ) , 0 ) \right] ^ { 2 } .\tag{20}
$$

$W _ { c } ^ { b d } ( i )$ emphasizes boundary-adjacent regions, while $W _ { c } ^ { f a r } ( i )$ highlights outside regions with potential abnormal foreground responses.

Let Ω denote all pixel or voxel positions in the current image or patch. We use the maximum teacher response as a continuous reliability weight:

$$
\rho ( i ) = \operatorname* { m a x } _ { c } \tilde { P } _ { c } ( i ) .\tag{21}
$$

The boundary calibration loss is:

$$
\mathcal { L } _ { b d } ^ { k } = \frac { \underset { i \in \Omega , c \in \mathcal { C } _ { f } } { \sum } \left| P _ { k , c } ( i ) - \tilde { P } _ { c } ( i ) \right| W _ { c } ^ { b d } ( i ) \rho ( i ) Q _ { k , c } ( i ) } { \underset { i \in \Omega , c \in \mathcal { C } _ { f } } { \sum } W _ { c } ^ { b d } ( i ) \rho ( i ) Q _ { k , c } ( i ) + \epsilon } .\tag{22}
$$

This term aligns the student prediction with the prototypecalibrated teacher in boundary-sensitive regions.

To suppress abnormal foreground responses, we define a class-specific prototype conflict indicator:

$$
\begin{array} { r l } & { C _ { k } ( i , c ) = \mathbb { I } [ c = \tilde { Y } ( i ) ] \mathbb { I } [ \hat { y } _ { k } ^ { \mathrm { p r o t o } } ( i ) \neq \tilde { Y } ( i ) ] } \\ & { \quad \quad \quad \cdot \mathbb { I } [ q _ { k } ( i ) > \tau _ { p } ] , \quad c \in \mathcal { C } _ { f } . } \end{array}\tag{23}
$$

This indicator acts only on the teacher-indicated class to avoid indiscriminate foreground suppression. The abnormal foreground weight is:

$$
A _ { k } ( i , c ) = W _ { c } ^ { f a r } ( i ) \big ( 1 - Q _ { k , c } ( i ) \big ) + C _ { k } ( i , c ) , \quad c \in \mathcal { C } _ { f } .\tag{24}
$$

The abnormal foreground suppression loss is:

$$
\mathcal { L } _ { a b n } ^ { k } = \frac { \displaystyle \sum _ { i \in \Omega , c \in \mathcal { C } _ { f } } P _ { k , c } ^ { 2 } ( i ) A _ { k } ( i , c ) \rho ( i ) } { \displaystyle \sum _ { i \in \Omega , c \in \mathcal { C } _ { f } } A _ { k } ( i , c ) \rho ( i ) + \epsilon } .\tag{25}
$$

This term penalizes foreground responses in structure-external and class-conflicting regions. The geometric calibration objective for the k-th student is:

$$
\begin{array} { r } { \mathcal { L } _ { g e o } ^ { k } = \mathcal { L } _ { b d } ^ { k } + \mathcal { L } _ { a b n } ^ { k } . } \end{array}\tag{26}
$$

Thus, $\mathcal { L } _ { b d } ^ { k }$ calibrates boundary consistency, while $\mathcal { L } _ { a b n } ^ { k }$ suppresses abnormal foreground responses based on the semantically calibrated teacher pseudo label from PSRE.

## D. Training Objective

The overall objective consists of supervised learning, prototype-calibrated pseudo supervision, and geometric calibration. For a labeled sample $( x ^ { l } , Y ^ { l } ) \in \mathcal { D } _ { l } .$ , both student networks are optimized by:

$$
\mathcal { L } _ { s u p } = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { 2 } \left[ \mathcal { L } _ { c e } ( P _ { k } ^ { l } , Y ^ { l } ) + \mathcal { L } _ { d i c e } ( P _ { k } ^ { l } , Y ^ { l } ) \right] ,\tag{27}
$$

where $( P _ { k } ^ { l } , F _ { k } ^ { l } ) = f _ { k } ( x ^ { l } )$ , and $\mathcal { L } _ { c e }$ and $\mathcal { L } _ { d i c e }$ denote crossentropy and Dice losses, respectively.

For unlabeled data, the prototype-calibrated soft teacher pseudo label $\tilde { P }$ supervises both student networks. To avoid gradients flowing through the generated pseudo target, $\tilde { P }$ is treated as a detached target. With sg[·] denoting stop-gradient, we define:

$$
\begin{array} { r } { \tilde { P } ^ { \mathrm { s g } } ( i ) = \mathrm { s g } \left[ \tilde { P } ( i ) \right] . } \end{array}\tag{28}
$$

The unlabeled pseudo-supervision loss is:

$$
\mathcal { L } _ { u } = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { 2 } \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \mathcal { H } \big ( \tilde { P } ^ { \mathrm { s g } } ( i ) , P _ { k } ( i ) \big ) ,\tag{29}
$$

where $\mathcal { H } ( \cdot , \cdot )$ is the soft cross-entropy:

$$
\mathcal { H } \big ( \tilde { P } ^ { \mathrm { s g } } ( i ) , P _ { k } ( i ) \big ) = - \sum _ { c = 0 } ^ { C - 1 } \tilde { P } _ { c } ^ { \mathrm { s g } } ( i ) \log \left( P _ { k , c } ( i ) + \epsilon \right) .\tag{30}
$$

The geometric calibration loss is averaged over the two student networks:

$$
\mathcal { L } _ { g e o } = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { 2 } \mathcal { L } _ { g e o } ^ { k } .\tag{31}
$$

The final training objective is:

$$
\mathcal { L } = \mathcal { L } _ { s u p } + \lambda _ { u } \mathcal { L } _ { u } + \lambda _ { g } \mathcal { L } _ { g e o } .\tag{32}
$$

The coefficients $\lambda _ { u }$ and $\lambda _ { g }$ balance prototype-calibrated pseudo supervision and geometry-aware calibration. In our implementation, they are treated as learnable loss-balancing weights and optimized jointly with the segmentation networks.

## III. EXPERIMENTS

## A. Datasets and Evaluation Metrics

We evaluated ProGeo-RC on ACDC [5] and MVAA-CT [23]. ACDC is a cardiac MRI benchmark from the MIC-CAI 2017 Automatic Cardiac Diagnosis Challenge, containing 100 cine MR cases annotated for RV, Myo, and LV. Following common semi-supervised settings, we split ACDC into 70 training, 10 validation, and 20 test cases, and used 10% or 20% of the training cases as labeled data. MVAA-CT is from the MICCAI 2026 Mitral Valve Anatomy Analysis Using Multimodal Imaging Data Challenge, containing 27 labeled mitral valve CT volumes and 1,040 official unlabeled volumes. Since the official validation and test sets were unavailable, we fixed 7 labeled cases as the held-out test split and used the remaining 20 as the training pool. For each low-annotation setting, 2 or 4 labeled cases were sampled five times, with the remaining training-pool cases treated as unlabeled data. Together with the official unlabeled volumes, the 2-labeled and 4-labeled settings contained 1,058 and 1,056 unlabeled cases, respectively, and results are reported as the mean and standard deviation over five samplings.

We evaluated segmentation performance using DSC, Jaccard, 95% Hausdorff Distance (95HD), and Average Surface Distance (ASD). DSC and Jaccard quantify regional overlap, while 95HD and ASD measure boundary distance errors.

## B. Implementation Details

All experiments were implemented in PyTorch on a single NVIDIA GeForce RTX 3090 GPU. For ACDC, images were cropped to $2 5 6 \times 2 5 6$ , with each mini-batch containing 8 labeled and 8 unlabeled images. For MVAA-CT, 3D patches of size $6 4 \times 1 2 8 \times 1 2 8$ were used, with 2 labeled and 2 unlabeled patches per mini-batch. Training lasted 30,000 iterations for ACDC and 15,000 for MVAA-CT. All reproduced baselines and ProGeo-RC were optimized with SGD using an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , momentum of 0.9, and weight decay of $1 \times 1 0 ^ { - 4 }$ . The EMA momentum for prototype memory was 0.99, $\lambda _ { u }$ and $\lambda _ { g }$ were learnable non-negative weights, and ϵ was fixed to $1 0 ^ { - 6 }$

## C. Comparison with State-of-the-art Methods

We compared ProGeo-RC with representative semisupervised methods, including CPS [12], MC-Net/MC-Net+ [24], [25], BCP [13], DiffRect [26], MLRP [15], W2SPC [27], BoCLIS [22], AdaMix [28], CAML [29], LeFeD [30], SDCL [31], and PICK [32]. U-Net [1] and 3D U-Net [3] served as supervised baselines for ACDC and MVAA-CT, respectively.

TABLE I  
COMPARISON ON THE ACDC TEST SET. BEST AND SECOND-BEST SEMI-SUPERVISED RESULTS WITHIN EACH LABELED SPLIT ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. <sup>†</sup> INDICATES RESULTS REPRODUCED BY US.
<table><tr><td>Method</td><td>Lab.</td><td></td><td></td><td>Unlab. |DSC↑ Jaccard↑ 95HD↓ ASD↓</td><td></td><td></td></tr><tr><td rowspan="2">U-Net† [1]</td><td>7(10%)</td><td>0</td><td>77.07</td><td>65.14</td><td>8.47</td><td>2.65</td></tr><tr><td>14(20%) 70(AIl)</td><td>0</td><td>84.12</td><td>74.21</td><td>6.46</td><td>2.03</td></tr><tr><td>CPS† [12]</td><td></td><td>0</td><td>91.47</td><td>84.68</td><td>1.40</td><td>0.34</td></tr><tr><td rowspan="5">MC-Net† [24] MC-Net+† [25] BCP† [13] DiffRect† [26]</td><td></td><td>86.74</td><td>77.43</td><td>6.64</td><td>1.80</td></tr><tr><td></td><td>87.17</td><td>77.97</td><td>8.09</td><td>2.06</td></tr><tr><td></td><td>86.31</td><td>76.71</td><td>8.23</td><td>2.20</td></tr><tr><td>7(10%) 63(90%)</td><td>88.23</td><td>79.71</td><td>4.54</td><td>1.41</td></tr><tr><td></td><td>89.53</td><td>81.57</td><td>4.33</td><td>1.21</td></tr><tr><td>MLRP† [15] W2SPC [27]</td><td></td><td>87.99</td><td>79.28</td><td>3.28</td><td>0.90</td></tr><tr><td>BoCLIS [22]</td><td></td><td>88.92</td><td>79.65</td><td>4.32</td><td>1.19</td></tr><tr><td>AdaMix† [28]</td><td></td><td>88.96</td><td>79.72</td><td>4.23</td><td>1.21</td></tr><tr><td>Ours</td><td></td><td>89.02</td><td>80.57</td><td>2.15</td><td>0.78</td></tr><tr><td>CPS† [12]</td><td></td><td>89.90</td><td>82.06</td><td>1.54</td><td>0.57</td></tr><tr><td></td><td></td><td>88.48</td><td>80.28</td><td>5.47</td><td>1.59</td></tr><tr><td>MC-Net† [24]</td><td></td><td>87.62</td><td>78.92</td><td>5.55</td><td>1.73</td></tr><tr><td>MC-Net+† [25]</td><td></td><td>87.94</td><td>79.50</td><td>5.77</td><td>1.66</td></tr><tr><td>BCP† [13]</td><td>14(20%) 56(80%)</td><td>89.20</td><td>81.13</td><td>4.80</td><td>1.56</td></tr><tr><td>DiffRect† [26]</td><td></td><td>89.92</td><td>82.27</td><td>3.44</td><td>0.93</td></tr><tr><td>MLRP† [15]</td><td></td><td>88.76</td><td>80.64</td><td>4.49</td><td>1.16</td></tr><tr><td>W2SPC [27]</td><td></td><td>90.23</td><td>81.24</td><td>4.01</td><td>1.06</td></tr><tr><td>BoCLIS [22]</td><td></td><td>90.10</td><td>81.11</td><td>4.14</td><td>1.06</td></tr><tr><td> $\operatorname { A d a M i x } ^ { \dagger }$  [28]</td><td></td><td>89.52</td><td>81.29</td><td>1.97</td><td>0.64</td></tr><tr><td>Ours</td><td></td><td>90.72</td><td>83.45</td><td>1.42</td><td>0.45</td></tr></table>

The results on ACDC and MVAA-CT are reported in Tables I and II, respectively. On ACDC, ProGeo-RC achieved 89.90% DSC, 82.06% Jaccard, 1.54 95HD, and 0.57 ASD with 10% labeled data, improving the corresponding secondbest results by 0.37%, 0.49%, 0.61, and 0.21, respectively. With 20% labeled data, it also ranked first across all metrics, reaching 90.72% DSC, 83.45% Jaccard, 1.42 95HD, and 0.45 ASD. On the held-out MVAA-CT test split, results are reported over five random labeled-set samplings. ProGeo-RC achieved 75.80% DSC, 62.76% Jaccard, 2.25 95HD, and 0.67 ASD in the 2-labeled setting, and 82.21% DSC, 69.89% Jaccard, 1.35 95HD, and 0.43 ASD in the 4-labeled setting, ranking first across all metrics in both cases. It also obtained the smallest standard deviations among semi-supervised methods, indicating stable performance across labeled-set samplings. These results demonstrate improved regional completeness and boundary localization on ACDC, as well as effective CT-based mitral valve region segmentation on MVAA-CT.

## D. Ablation Study

We conducted ablation experiments on the ACDC test set to evaluate the contributions of PSRE and S-SGEC, as reported in Table III. Adding PSRE improved DSC and Jaccard to 88.77% and 80.46%, while reducing 95HD and ASD to 3.40 and 1.01, indicating improved pseudo-label reliability. Adding S-SGEC also yielded clear gains, especially in boundary-related metrics, reducing 95HD and ASD to 3.20 and 0.76. Combining both modules achieved the best performance across all metrics, showing that semantic reliability estimation and geometric error calibration act complementarily. We also analyzed the reliability threshold $\tau _ { p }$ and teacher calibration weight λ in Table IV. ProGeo-RC remained stable when $\tau _ { p }$ varied from 0.6 to 0.8 and λ from 0.1 to 0.3, with the best overall balance obtained at $\tau _ { p } = 0 . 7$ and λ = 0.2.

LeFeD  
CAML  
GT  
AdaMix  
Image  
GT  
TABLE II  
COMPARISON ON THE HELD-OUT MVAA-CT TEST SPLIT. BEST AND SECOND-BEST SEMI-SUPERVISED RESULTS WITHIN EACH LABELED SPLIT ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. <sup>†</sup>INDICATES RESULTS REPRODUCED BY US.
<table><tr><td rowspan=1 colspan=10>Method       |Lab. Unlab.|DSC↑  Jaccard↑95HD↓  ASD↓</td></tr><tr><td rowspan=1 colspan=1>3D U-Net† [3]</td><td rowspan=1 colspan=5>2    04    0</td><td rowspan=1 colspan=4>|74.55±3.83 60.13±4.90 4.56±2.10 1.18±0.4481.06±0.9368.34±1.271.83±0.43 0.59±0.14</td></tr><tr><td rowspan=6 colspan=1>CPS† [12]MC-Net+† [25]CAML† [29]LeFeD† [30]MLRP† [15]SDCL† [31]</td><td rowspan=3 colspan=5></td><td rowspan=2 colspan=4>|72.78±3.56 58.74±3.71 3.42±1.34 1.05±0.5673.28±2.21 58.92±3.36 3.23±1.02 0.74±0.25</td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=2 colspan=4>72.33±3.17 57.25±4.02 3.79±1.47 0.80±0.4275.48±4.33 61.14±5.35 3.23±1.31 0.94±0.33</td></tr><tr><td rowspan=1 colspan=2>2</td><td></td><td rowspan=1 colspan=2>1058</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=4>71.46±3.52 56.94±4.13 4.26±1.67 1.15±0.9474.04±3.05 60.45±3.43 3.06±1.21 0.78±0.60</td></tr><tr><td rowspan=1 colspan=5></td></tr><tr><td rowspan=2 colspan=1>PICK† [32]AdaMix† [28]Ours</td><td rowspan=2 colspan=5></td><td rowspan=1 colspan=3>73.55±5.21 59.16±5.53</td><td rowspan=1 colspan=1>3.42±1.24 0.77±0.18</td></tr><tr><td rowspan=1 colspan=3>72.59±3.02 57.26±3.6775.80±1.8062.76±2.33 2</td><td rowspan=1 colspan=1>4.11±2.481.52±0.49.25±0.60 0.67±0.11</td></tr><tr><td rowspan=3 colspan=1>CPS† [12]MC-Net+† [25]CAML† [29]</td><td rowspan=3 colspan=5></td><td rowspan=1 colspan=2>|79.83±1.50</td><td rowspan=1 colspan=1>65.74±1.65</td><td rowspan=1 colspan=1>1.97±0.54 0.67±0.31</td></tr><tr><td rowspan=1 colspan=2>80.32±1.446</td><td rowspan=1 colspan=1>6.75±1.561</td><td rowspan=1 colspan=1>.98±0.52 0.65±0.28</td></tr><tr><td rowspan=1 colspan=2>79.87±1.22</td><td rowspan=1 colspan=1>66.43±1.37</td><td rowspan=1 colspan=1>2.02±0.580.72±0.26</td></tr><tr><td rowspan=1 colspan=1>LeFeD† [30]</td><td rowspan=1 colspan=5>4  1056</td><td rowspan=1 colspan=2>81.27±1.19</td><td rowspan=1 colspan=1>68.73±1.591</td><td rowspan=1 colspan=1>.62±0.47 0.51±0.19</td></tr><tr><td rowspan=2 colspan=1>MLRP† [15]SDCL† [31]</td><td rowspan=2 colspan=5></td><td rowspan=1 colspan=3>80.99±1.7868.28±2.44 1</td><td rowspan=2 colspan=1>.66±0.56 0.53±0.22.67±0.66 0.65±0.27</td></tr><tr><td rowspan=1 colspan=3>80.05±1.54 67.30±1.561</td></tr><tr><td rowspan=1 colspan=1>PICK† [32]</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=3>80.08±1.13 67.10±1.36</td><td rowspan=1 colspan=1>2.04±0.62 0.65±0.20</td></tr><tr><td rowspan=1 colspan=1>AdaMix† [28]</td><td rowspan=2 colspan=5></td><td rowspan=1 colspan=4>81.04±1.60 68.71±2.261.44±0.43 0.55±0.23</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=4>82.21±1.01 69.89±1.20 $\overline { { 1 . 3 5 \pm 0 . 4 1 } }$ 0.43±0.17</td></tr></table>

TABLE III

ABLATION STUDY OF THE PROPOSED COMPONENTS ON THE ACDC DATASET.
<table><tr><td>Method</td><td>PSRE</td><td>S-SGEC | DSC↑</td><td></td><td>Jaccard↑</td><td>95HD↓</td><td>ASD↓</td></tr><tr><td>Baseline</td><td>一</td><td>一</td><td>86.47</td><td>76.52</td><td>6.72</td><td>2.05</td></tr><tr><td>Baseline + PSRE</td><td>√</td><td>1</td><td>88.77</td><td>80.46</td><td>3.40</td><td>1.01</td></tr><tr><td>Baseline + S-SGEC</td><td>一</td><td>√</td><td>88.69</td><td>80.29</td><td>3.20</td><td>0.76</td></tr><tr><td>ProGeo-RC</td><td>√</td><td>√</td><td>89.90</td><td>82.06</td><td>1.54</td><td>0.57</td></tr></table>

TABLE IV

SENSITIVITY ANALYSIS OF RELIABILITY THRESHOLD AND TEACHER CALIBRATION WEIGHT ON THE ACDC DATASET.
<table><tr><td>Setting</td><td></td><td>Value | DSC↑</td><td>Jaccard↑</td><td>95HD↓</td><td>ASD↓</td></tr><tr><td rowspan="3">Reliability threshold  $\tau _ { p }$ </td><td>0.6</td><td>89.82</td><td>82.01</td><td>1.57</td><td>0.60</td></tr><tr><td>0.7</td><td>89.90</td><td>82.06</td><td>1.54</td><td>0.57</td></tr><tr><td>0.8</td><td>89.88</td><td>82.04</td><td>1.53</td><td>0.61</td></tr><tr><td rowspan="3">Calibration weight λ</td><td>0.1</td><td>89.91</td><td>82.01</td><td>1.60</td><td>0.59</td></tr><tr><td>0.2</td><td>89.90</td><td>82.06</td><td>1.54</td><td>0.57</td></tr><tr><td>0.3</td><td>89.78</td><td>81.97</td><td>1.66</td><td>0.61</td></tr></table>

![](images/17b19c3f5fb993a2d79d564c5719658681a6e31c810e05825b197cda4fd42a60.jpg)  
Fig. 2. Visual comparison on representative ACDC test slices. Red, green, and blue denote RV, Myo, and LV, respectively.

Image  
CPS  
Ours  
![](images/5245b9afccd29e0884bf574420a1766c77539c4fab590bdf4dfc8642228cf49d.jpg)  
Fig. 3. Visual comparison on representative slices from the held-out MVAA-CT test split. Yellow contours denote the annotated segmentation regions.

TABLE V  
ANALYSIS OF MODEL PARAMETERS AND COMPUTATIONAL COMPLEXITY ON ACDC WITH AN INPUT SIZE OF 256 × 256.
<table><tr><td>Method</td><td>Params (M)</td><td>GFLOPs</td></tr><tr><td>AdaMix</td><td>3.27</td><td>7.40</td></tr><tr><td>BCP</td><td>1.81</td><td>6.01</td></tr><tr><td>CPS</td><td>1.81</td><td>6.01</td></tr><tr><td>DiffRect</td><td>1.81</td><td>6.01</td></tr><tr><td>MC-Net</td><td>3.6</td><td>14.02</td></tr><tr><td>MC-Net+</td><td>3.5</td><td>13.81</td></tr><tr><td>MLRP</td><td>2.45</td><td>9.85</td></tr><tr><td>Ours</td><td>1.81</td><td>6.01</td></tr></table>

## E. Analysis of Visualization

Figure 2 and Figure 3 show visual comparisons on ACDC and MVAA-CT, respectively. On ACDC, ProGeo-RC produced more stable segmentation around myocardial boundaries, ventricular contours, and low-contrast regions than AdaMix [28], BCP [13], CPS [12], and MC-Net+ [25], with better preservation of RV, Myo, and LV shapes and fewer boundary shifts. On MVAA-CT, where thin valve structures are prone to discontinuities, boundary shifts, and background false positives, ProGeo-RC showed more accurate localization, better structural continuity, and fewer isolated false-positive regions. These visual observations are consistent with the quantitative results.

## F. Computational Efficiency

Computational efficiency is important for clinical deployment. As shown in Table V, ProGeo-RC requires 1.81M parameters and 6.01 GFLOPs under single-model inference, matching BCP, CPS, and DiffRect while using fewer resources than AdaMix, MC-Net, MC-Net+, and MLRP. Since PSRE and S-SGEC act only as training-stage pseudo-supervision calibration strategies, they introduce no additional inferencetime branches. Thus, ProGeo-RC improves segmentation performance while preserving the inference complexity of the underlying backbone.

## IV. CONCLUSION

We presented ProGeo-RC, a semi-supervised cardiac image segmentation framework that improves pseudo-supervision by coupling feature-space semantic reliability with image-space geometric plausibility. PSRE estimates reliable unlabeled pixels and calibrates teacher pseudo labels using probabilistic prototypes, while S-SGEC applies signed-distance-fieldbased constraints to refine ambiguous boundaries and suppress abnormal foreground responses. Experiments on ACDC and MVAA-CT, together with ablation studies and visualizations, demonstrated consistent gains across annotation settings, imaging modalities, and segmentation targets. The efficiency analysis further showed that ProGeo-RC introduces no extra inference-time architectural overhead. These results indicate that joint semantic and geometric calibration can provide reliable and efficient pseudo-supervision for low-annotation cardiac image analysis.

## REFERENCES

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2015, pp. 234– 241.

[2] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in Proc. Int. Conf. 3D Vision, 2016, pp. 565–571.

[3] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2016, pp. 424–432.

[4] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, 2021.

[5] O. Bernard et al., “Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved?” IEEE Trans. Med. Imag., vol. 37, no. 11, pp. 2514–2525, 2018.

[6] W. Bai et al., “Semi-supervised learning for network-based cardiac MR image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2017, pp. 253–260.

[7] G. Luo et al., “Commensal correlation network between segmentation and direct area estimation for bi-ventricle quantification,” Med. Image Anal., vol. 59, Art. no. 101591, 2020.

[8] S. Dong et al., “Deep atlas network for efficient 3D left ventricle segmentation on echocardiography,” Med. Image Anal., vol. 61, Art. no. 101638, 2020.

[9] G. Luo et al., “Dynamically constructed network with error correction for accurate ventricle volume estimation,” Med. Image Anal., vol. 64, Art. no. 101723, 2020.

[10] D.-H. Lee, “Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks,” in Proc. Workshop Challenges Representation Learning, ICML, 2013.

[11] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Proc. Adv. Neural Inf. Process. Syst., 2017.

[12] X. Chen, Y. Yuan, G. Zeng, and J. Wang, “Semi-supervised semantic segmentation with cross pseudo supervision,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 2613–2622.

[13] Y. Bai, D. Chen, Q. Li, W. Shen, and Y. Wang, “Bidirectional copy-paste for semi-supervised medical image segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 11514–11524.

[14] E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness, “Pseudo-labeling and confirmation bias in deep semi-supervised learning,” in Proc. Int. Joint Conf. Neural Networks, 2020, pp. 1–8.

[15] J. Su, Z. Luo, S. Lian, D. Lin, and S. Li, “Mutual learning with reliable pseudo label for semi-supervised medical image segmentation,” Med. Image Anal., vol. 94, Art. no. 103111, 2024.

[16] X. Li, G. Luo, W. Wang, K. Wang, and S. Li, “Curriculum label distribution learning for imbalanced medical image segmentation,” Med. Image Anal., vol. 89, Art. no. 102911, 2023.

[17] Z. Zhu, X. Ma, W. Wang, S. Dong, K. Wang, L. Wu, G. Luo, G. Wang, and S. Li, “Boosting knowledge diversity, accuracy, and stability via trienhanced distillation for domain continual medical image segmentation,” Med. Image Anal., vol. 94, Art. no. 103112, 2024.

[18] I. Alonso, A. Sabater, D. Ferstl, L. Montesano, and A. C. Murillo, “Semisupervised semantic segmentation with pixel-level contrastive learning from a class-wise memory bank,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 8219–8228.

[19] W. Lu et al., “UPCoL: Uncertainty-informed prototype consistency learning for semi-supervised medical image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2023, pp. 662–672.

[20] H. Kervadec, J. Bouchtiba, C. Desrosiers, E. Granger, J. Dolz, and I. B. Ayed, “Boundary loss for highly unbalanced segmentation,” Med. Image Anal., vol. 67, Art. no. 101851, 2021.

[21] D. Karimi and S. E. Salcudean, “Reducing the Hausdorff distance in medical image segmentation with convolutional neural networks,” IEEE Trans. Med. Imag., vol. 39, no. 2, pp. 499–513, 2020.

[22] Y. Yang, J. Zhuang, G. Sun, R. Wang, and J. Su, “Boundary-guided contrastive learning for semi-supervised medical image segmentation,” IEEE Trans. Med. Imag., vol. 44, no. 7, pp. 2973–2988, 2025.

[23] MVAA 2026 Challenge Organizers, “MVAA 2026: Mitral Valve Anatomy Analysis Using Multimodal Imaging Data,” Codabench, 2026. [Online]. Available: https://www.codabench.org/competitions/15662/

[24] Y. Wu, M. Xu, Z. Ge, J. Cai, and L. Zhang, “Semi-supervised left atrium segmentation with mutual consistency training,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2021, pp. 297–306.

[25] Y. Wu, Z. Ge, D. Zhang, M. Xu, L. Zhang, Y. Xia, and J. Cai, “Mutual consistency learning for semi-supervised medical image segmentation,” Med. Image Anal., vol. 81, Art. no. 102530, 2022.

[26] X. Liu, W. Li, and Y. Yuan, “DiffRect: Latent diffusion label rectification for semi-supervised medical image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2024, pp. 56–66.

[27] Y. Yang, G. Sun, T. Zhang, R. Wang, and J. Su, “Semi-supervised medical image segmentation via weak-to-strong perturbation consistency and edge-aware contrastive representation,” Med. Image Anal., vol. 101, Art. no. 103450, 2025.

[28] Z. Shen, P. Cao, J. Su, J. Yang, and O. R. Zaiane, “Adaptive Mix for semi-supervised medical image segmentation,” Med. Image Anal., vol. 108, Art. no. 103857, 2026.

[29] S. Gao, Z. Zhang, J. Ma, Z. Li, and S. Zhang, “Correlation-aware mutual learning for semi-supervised medical image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2023, pp. 98–108.

[30] Q. Zeng, Y. Xie, Z. Lu, M. Lu, J. Zhang, and Y. Xia, “Consistencyguided differential decoding for enhancing semi-supervised medical image segmentation,” IEEE Trans. Med. Imag., vol. 44, no. 1, pp. 44–56, 2025.

[31] B. Song and Q. Wang, “SDCL: Students discrepancy-informed correction learning for semi-supervised medical image segmentation,” in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention, 2024, pp. 567–577.

[32] Q. Zeng et al., “PICK: Predict and mask for semi-supervised medical image segmentation,” Int. J. Comput. Vis., vol. 133, pp. 3296–3311, 2025.