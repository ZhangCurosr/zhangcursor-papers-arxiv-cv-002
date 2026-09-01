# MEOM: Multi-View Expected-OKS Maximization for Human Pose Triangulation

Ziliang Xiong Henglin Shi Per-Erik Forssen´ Computer Vision Laboratory, Department of Electrical Engineering, Linkoping University, Sweden¨

{ziliang.xiong, henglin.shi, per-erik.forssen}@liu.se

## Abstract

Conventional algebraic triangulation solves 3D human pose estimation (HPE) from multi-view 2D keypoints. The typical approach, decoding 2D keypoints from predicted heatmaps, is unreliable as heatmaps can be multimodal under occlusion, and collapsing them into single peaks discards their spatial distribution. We seek to use the entire heatmap to estimate 3D poses more accurately, which requires solving two problems: how to robustlyfuse heatmaps across views, and how to assess the reliability ofheatmaps. For the former, we introduce a novel objective, Multi-view Expected-OKS Maximization (MEOM), that locates a 3D joint where the views agree in probability mass. For the latter, we adopt highest-density-region (HDR) calibration as a diagnostic of that mass, independently of distance-based metrics. The proposed framework covers two settings, with and without 3D supervision. Without 3D supervision, we optimize 3D poses from pretrained heatmap predictors by maximizing MEOM, achieving comparable performance with state-of-the-art methods that rely on larger backbones, temporal fusion, and simulated 3D data. On ambiguous Human3.6M (H36MA) and occluded CMU Panoptic frames, the advantage is substantial. When 3D labels are available, we train the model end-to-end with a combined MEOM and MSE loss, achieving 19.11 mm absolute MPJPE on Human3.6M outperforming the state-of-the-art volumetric approach on absolute MPJPE at half the inference cost.

## 1. Introduction

Multi-view 3D HPE is classically solved by triangulation [11]: 2D keypoints are detected in each calibrated<sup>1</sup> view, and the 3D joint is recovered by e.g., view-weighted direct linear transform (DLT) [10]. The geometry is explicit and the cost negligible, but every view is reduced to one point per joint: the spatial mass of each heatmap is collapsed before triangulation. Volumetric methods keep that mass [14, 15, 25], but have cubic memory cost and learn a 3D prior tied to the capture geometry. Fig. 1 shows why fusion of collapsed peaks can be unreliable: each mode of a multimodal heatmap defines its own camera ray, so a pointbased objective may settle on a sharp, low-mass intersection, as under ambiguity and occlusion.

![](images/8316dea40b333de183bd2912d46da00e44ebd79edc61f69456e2548349abe8bd.jpg)  
Figure 1. Multimodal heatmaps induce ambiguous triangulation. The upper view is unimodal; the lower view is bimodal, with a sharp outlier mode and a broader mode with most of the probability mass around. Our proposed MEOM selects a 3D point in the high-mass region of this implicit 3D density (illustration only). With argmax decoding, reprojection error and raw heatmap likelihood coincide, both selecting the sharp, low-mass intersection (red arrow). A 3D MSE loss instead collapses the implicit volume to isotropic Gaussians.

Recent 2D HPE has moved away from heatmap collapse.

ProbPose [22] treats each heatmap as a spatial distribution over the keypoint, calibrates it with temperature scaling, and reads the joint by Object Keypoint Similarity (OKS) decoding, which slightly improves 2D accuracy. Similarly, we wish to use the entire heatmap for 3D HPE, which requires solving two problems: how to robustly fuse heatmaps across views into a 3D estimate, and how to assess whether the heatmap distribution is reliable.

For the former, we locate the 3D joint where the views agree in probability mass. Each heatmap is convolved with a fixed OKS kernel, and a candidate 3D joint is scored by the aggregated response over cameras. Maximizing that score is our proposed Multi-view Expected-OKS Maximization (MEOM). The OKS kernel acts as a sliding window that seeks the largest probability mass rather than the highest peak. Thus, MEOM is a multi-view fusion method that optimizes 3D pose where the views agree in mass.

For the latter, we adopt highest-density-region (HDR) calibration [4] as a reliability diagnostic of mass distribution in heatmaps. Because expected-OKS decoding reads and MEOM fuses probability mass, better-calibrated heatmaps are expected to result in more accurate triangulation. HDR-ECE numerically summarizes the deviation from perfect HDR calibration. Additionally, we identify that ProbPosestyle map calibration is the discrete case of HDR calibration, and we further show that temperature scaling, which originally handles classification calibration [9], locates the HDR calibration optimum, though it does not attain perfect calibration.

The MEOM score admits dual uses depending on the availability of 3D supervision. When 3D supervision is not available, we freeze a pretrained heatmap predictor and optimize 3D joints: view-weighted DLT initialization is refined by maximizing MEOM. When 3D supervision is available, we project the 3D labels to the predicted heatmaps and optimize them: after an MSE-only stage, a triangulator is trained end-to-end through differentiable expected-OKS decoding with a combined MEOM and 3D MSE loss, without subsequent MEOM 3D refinement.

We evaluate both settings on Human3.6M [13], its ambiguous subset H36MA [26], and CMU Panoptic [15]. When 3D supervision is not available, MEOM outperforms methods using reprojection error and heatmap likelihood as optimization targets, with larger gains on H36MA and occluded CMU Panoptic, and is comparable to methods that use larger backbones, temporal fusion, and simulated 3D data [5]. On CMU Panoptic, COCO-pretrained ProbPose with no target-domain 3D training remains competitive with the end-to-end algebraic baseline [14]. When 3D supervision is available, MEOM surpasses state-of-the-art volumetric method [14] in absolute MPJPE at around half the inference cost (Tab. 4).

## Contributions.

1. MEOM score for algebraic triangulation. We introduce MEOM for multi-view 3D HPE: a 3D candidate is scored by the aggregated Expected-OKS response across cameras, so fusion selects points on which the views agree in probability mass.

2. MEOM optimization without 3D labels. With a frozen heatmap predictor, maximizing MEOM is competitive with methods that rely on larger backbones, temporal fusion, and simulated 3D data [5], without using 3D joint labels of the target dataset.

3. End-to-end MEOM training. With 3D labels, an algebraic triangulator trained with expected-OKS decoding and a combined MEOM+MSE loss surpasses volumetric triangulation [14] in absolute MPJPE on Human3.6M at roughly half the inference cost.

4. HDR calibration for mass-based triangulation. We compare quantile, HDR, and copula diagnostics under temperature scaling and measure how each predicts triangulation accuracy. On H36MA, over-sharp maps cost 9.6 mm Abs. MPJPE; the HDR-ECE-minimizing temperature is optimal for reprojection error and within 0.17 mm of the MEOM optimum.

Sec. 2 reviews related work. Sec. 3 presents MEOM, its two uses with and without 3D labels, and HDR calibration. Sec. 4 reports datasets, then frozen-network results, end-toend training, and HDR diagnostics. Sec. 5 concludes.

## 2. Related Work

3D multiview HPE Algebraic triangulation first detects 2D keypoints in each view and recovers 3D joints by a viewweighted DLT [10, 11], either as post-hoc refinement under predicted 2D uncertainty [8] or trained end-to-end through differentiable soft-argmax [14]. It is fast and geometrically explicit, yet every heatmap is collapsed to one point and at most an additional scalar confidence. Beyond point triangulation, Joo et al. [15] discretize the 3D mocap space into a voxel grid and back-project each voxel center to all camera views, aggregating the corresponding heatmap scores into a 3D node-likelihood map by spatial voting to propose candidate 3D joints. Subsequent work unprojects multiview features into a voxel grid and fuses them with a 3D CNN, either for a single person with a learned body-shape prior [14] or for multi-person detection inside per-person cuboids [25]. Voxel grids raise laboratory accuracy, but at cubic memory cost, and a 3D CNN trained on them can overfit to the capture geometry and camera layout. Learned 2D-to-3D lifters further replace explicit triangulation with cross-view and temporal fusion networks [3, 5, 6]; strong benchmark numbers can still come with limited robustness when projective geometry is only implicit and training camera topologies are narrow. Our method instead retains triangulation: Expected-OKS decoding and MEOM operate on predicted heatmaps without constructing a voxel grid or learning a 3D human prior.

2D human pose estimation. Modern 2D HPE [4, 22, 24, 27, 28] localizes keypoints with dense heatmaps per joint, which represent the spatial uncertainty distribution over the keypoint. HRNet [24] maintains high-resolution representations throughout the network and remains a strong CNN baseline; ViTPose [27] shows that plain vision transformers are competitive when trained at scale. Both are typically supervised by pixelwise MSE against isotropic Gaussian heatmap targets, yet evaluated on COCO by mean Average Precision (mAP) under OKS [7, 19]. OKS is a scale- and joint-normalized Gaussian similarity to the annotated keypoint. OKS assigns a distinct per-joint scale (larger for hips, smaller for eyes), computed as the standard deviation of independent annotators repeatedly labeling the same people. ProbPose [22] instead predicts calibrated non-parametric heatmaps with temperature scaling, together with a presence score, and trains with OKS loss [21]. We adopt the expected-OKS kernel to regularize each heatmap with the annotation noise, in both DLT initialization and MEOM fusion.

Uncertainty and calibration in pose estimation. In 3D HPE, aleatoric uncertainty has been modeled with structured Gaussians for weighted multi-view triangulation [8]. For monocular 2D-to-3D lifting, probabilistic methods represent ambiguous poses with normalizing flows [26], diffusion models [12], and flow matching [18]. However, these methods do not directly evaluate the predicted distribution of 3D poses. In 2D HPE, Gu et al. [7] show that standard confidences (e.g., heatmap maxima) are poorly calibrated to OKS, and propose a light post-hoc network to realign confidence with pose accuracy. In our work, scalar confidenceto-OKS alignment is not enough: heatmaps must satisfy distributional calibration properties. Bramlage et al. [1] instead regress parametric location uncertainties and apply post-hoc quantile recalibration [16] independently per joint and axis. Axis-wise quantile calibration is not enough as marginal calibration does not imply joint calibration [29]. Purkrabek et al. [22] investigate heatmap calibration by temperature scaling [9], yet do neither show the benefits of calibration nor theoretically prove its optimality. A multivariate predictive distribution must first be reduced by a prerank that maps the forecast to a scalar [17]; different preranks then induce quantile [16], HDR [4], and copula [29] calibration. We therefore compare these diagnostics under inference-time temperature scaling and study how they correlate with multi-view triangulation accuracy.

## 3. Method

## 3.1. Motivation: why Expected OKS

Let $\textbf { X } \in \ \mathbb { R } ^ { 3 }$ denote a 3D joint. Under skeletal and anthropometric constraints its distribution $p ( \mathbf { X } )$ lies on a lowdimensional, typically anisotropic manifold $\mathcal { M } \subset \mathbb { R } ^ { 3 }$ . A calibrated camera with projection π induces a heatmap as the projective marginal

$$
p ( \mathbf { x } ) = \int _ { \mathbb { R } ^ { 3 } } p ( \mathbf { X } ) \delta \big ( \pi ( \mathbf { X } ) - \mathbf { x } \big ) d \mathbf { X } ,\tag{1}
$$

where $\mathbf { x } \in \mathbb { R } ^ { 2 }$ is the keypoint location in the image plane and δ is the Dirac delta that restricts the integral to the camera ray projecting to x, so $p ( \mathbf { x } )$ accumulates the 3D probability mass along that ray. Training heatmaps with pixelwise MSE against a fixed isotropic Gaussian [24, 27] therefore misspecifies the marginal whenever uncertainty is elongated (depth ambiguity), truncated (crop boundary), or multimodal (occlusion).

Under a uniform spatial prior and conditional independence of views given X, with $v \in \{ 1 , \ldots , V \}$ indexing the V calibrated views and $I ^ { ( v ) }$ the image in view v, the multiview posterior satisfies

$$
\log p ( \mathbf { X } \mid \{ I ^ { ( v ) } \} ) = \sum _ { v = 1 } ^ { V } \log p \Big ( I ^ { ( v ) } \mid \mathbf { X } \Big ) + C ,\tag{2}
$$

where C is a constant (See full derivation in Appendix A). Let $\mathbf { \mathcal { H } } ^ { ( v ) }$ denote the predicted spatial localization distribution (heatmap) on $\mathbb { R } ^ { \hat { W } \times H }$ in view v, and let $\mathbf { P } ^ { ( v ) }$ be the corresponding camera matrix with projection $\pi ( \mathbf { P } ^ { ( v ) } , \cdot )$ . In practice, using $\mathbf { \mathcal { H } } ^ { ( v ) }$ directly in place of the view likelihood in (2), as a training loss or as a refinement objective, severely degrades performance due to sparsity.

To mitigate this, we follow ProbPose [22] and convolve $\mathbf { \mathcal { H } } ^ { ( v ) }$ with an OKS kernel K to obtain the expected-OKS response $\pmb { S } ^ { ( v ) } = \pmb { \mathcal { H } } ^ { ( v ) } * \mathbf { K }$ . The COCO OKS [19] between a query location x and a reference $\mathbf { x } ^ { \prime }$ for joint k is given as

$$
\mathrm { O K S } _ { k } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \exp \left( - \frac { \| \mathbf { x } - \mathbf { x } ^ { \prime } \| _ { 2 } ^ { 2 } } { 2 s ^ { 2 } \sigma _ { k } ^ { 2 } } \right) ,\tag{3}
$$

where s is the subject scale and $\sigma _ { k }$ is the fixed standard deviation for each keypoint. When the kernel is discretized on a heatmap, the subject scale s is implied by the activationwindow resolution. Then, we introduce the MEOM score by aggregating the expected-OKS responses of the 3D joint over views:

$$
\sum _ { v = 1 } ^ { V } w ^ { ( v ) } { \pmb S } ^ { ( v ) } \Big ( \pi \big ( { \pmb P } ^ { ( v ) } , { \pmb X } \big ) \Big ) ,\tag{4}
$$

which is a practical surrogate for multi-view maximum a posteriori inference with finitely many views; here $w ^ { ( v ) } \geq 0$ is a view weight, usually predicted by the model.

![](images/ba2f7b3649ba8af97b6cbed85cb8dc3f5ba19c49317de81e170e10346a48da63.jpg)  
Figure 2. MEOM triangulation without 3D labels. First, a frozen ProbPose [22] produces per-view heatmaps and view weights; then view-weighted DLT gives an initial 3D pose estimation $\mathbf { X } ^ { ( 0 ) }$ , and iterative refinement finds $\mathbf { X } ^ { * }$ that maximizes the MEOM score.

The same score admits dual uses that differ in which variable is free: when 3D labels are unavailable, Sec. 3.2 freezes the heatmaps and finds $\mathbf { X } ^ { * }$ that maximizes the MEOM score; when 3D labels are available, Sec. 3.3 fixes $\mathbf { X } = \mathbf { X } ^ { \mathrm { g t } }$ and optimizes heatmaps (hence S) to raise that score. Sec. 3.4 further treats heatmap temperature and HDR calibration as a reliability diagnostic for both uses.

## 3.2. MEOM triangulation without 3D labels

This subsection studies triangulation with heatmaps fixed: 3D ground truth is typically available only in instrumented labs, so outside that setting a pretrained heatmap predictor is frozen and we optimize only the multi-view fusion of its detections via DLT initialization and iterative refinement. See the pipeline in Fig. 2. We compare three algebraic optimization objectives for post-hoc refinement.

Reprojection error. Minimize the weighted sum of reprojection residuals [8]:

$$
\mathbf { X } _ { k } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { X } _ { k } } \sum _ { v = 1 } ^ { V } w _ { k } ^ { ( v ) } \left\| \hat { \mathbf { x } } _ { k } ^ { ( v ) } - \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } _ { k } \big ) \right\| _ { 2 } ,\tag{5}
$$

where indices k and v run over joints and calibrated views; ${ \bf X } _ { k }$ is a candidate 3D joint, $\hat { \mathbf { x } } _ { k } ^ { \left( v \right) }$ the 2D network output, and $w _ { k } ^ { ( v ) }$ the view weights.

Heatmap likelihood. Maximize the weighted localization likelihood of the projected point under the frozen heatmap $\mathcal { H } _ { k } ^ { ( v ) }$

$$
\mathbf { X } _ { k } ^ { * } = \underset { \mathbf { X } _ { k } } { \arg \operatorname* { m a x } } ~ \sum _ { v = 1 } ^ { V } w _ { k } ^ { ( v ) } \mathcal { H } _ { k } ^ { ( v ) } \left( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } _ { k } \big ) \right) .\tag{6}
$$

MEOM. The response S of Sec. 3.1 is shared by DLT initialization and MEOM refinement; $\hat { \mathbf { x } } _ { k } ^ { ( v ) } = \arg$ max $\pmb { S } _ { k } ^ { ( v ) }$ is used only to form the DLT initialization. With heatmaps held fixed, refinement then maximizes Eq. (4) over the 3D joint,

$$
\mathbf { X } _ { k } ^ { * } = \underset { \mathbf { X } _ { k } } { \arg \operatorname* { m a x } } ~ \sum _ { v = 1 } ^ { V } w _ { k } ^ { ( v ) } \pmb { S } _ { k } ^ { ( v ) } \Big ( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } _ { k } \big ) \Big ) .\tag{7}
$$

In experiments, for all three objectives, we instantiate $w _ { k } ^ { ( v ) }$ with either ProbPose presence probability or OKS prediction [22] and compare both choices.

## 3.3. End-to-end triangulation with MEOM

Sec. 3.2 treats heatmaps as fixed observations and applies MEOM only at inference by searching over X. When 3D supervision is available, we instead use the dual form of the MEOM score (4): fix $\mathbf { X } = \mathbf { X } ^ { \mathrm { g t } }$ and train the 2D network end-to-end to raise that score, jointly with a 3D MSE term that back-propagates through differentiable expected-OKS decoding and DLT (Fig. 3). Gradients from both losses flow into each view’s heatmaps and act as a geometric pullback: probability mass is encouraged to concentrate where multiview rays agree with the supervised 3D pose, without constructing an explicit voxel grid.

## 3.3.1. Differentiable OKS decoding

Iskakov et al. [14] recover 2D joints by soft-argmax on raw heatmaps and then apply DLT. Soft-argmax is convenient for training, but on multimodal heatmaps it returns the mass center that can fall between modes (expectation collapse). We therefore keep the expected-OKS response, and decode it with temperatured-soft-argmax:

![](images/139f7cabeafb84e6580100a0a808d060e5aaabe868f54304245d8639132eb142.jpg)  
Figure 3. End-to-end MEOM training. A shared backbone produces unit-scale heatmaps and view confidences. The MEOM loss L<sub>MEOM</sub> samples $s$ at projections of the 3D ground truth and takes a confidence-weighted mean; the MSE loss $\mathcal { L } _ { \mathrm { M S E } }$ triangulates soft Expected OKS detections by confidence-weighted DLT and compares to the same 3D ground truth.

$$
\hat { \mathbf { x } } _ { k } ^ { ( v ) , \mathrm { s o f t } } = \sum _ { ( x , y ) } ( x , y ) \operatorname { s o f t m a x } \left( \frac { \pmb { S } _ { k } ^ { ( v ) } } { \tau } \right) _ { ( x , y ) } ,\tag{8}
$$

with a small temperature τ . $\mathbf { A s } \tau  0$ , the soft weights concentrate on the dominant OKS peak, so the proxy tracks a mode rather than a broad center of mass on H (We demonstrate this in Appendix A). This soft path is used whenever 3D MSE must back-propagate through triangulation.

## 3.3.2. Joint MSE and MEOM training

Under the projective marginal (1), a 3D MSE in [14] on soft-decoded, triangulated joints is well matched only to a homoscedastic isotropic Gaussian prior on $\mathbf { X } ;$ it misspecifies anisotropic or multimodal uncertainty. As in Sec. 3.1, the two uses of the MEOM score (4) differ in what is free. With supervision, we fix $\mathbf { X } _ { k } = \mathbf { X } _ { k } ^ { \mathrm { g t } }$ and instead optimize the heatmaps and the predictor,

$$
\{ \mathcal { H } _ { k } ^ { ( v ) } \} ^ { * } = \underset { \{ \mathcal { H } _ { k } ^ { ( v ) } \} } { \arg \operatorname* { m a x } } ~ \sum _ { k } \sum _ { v = 1 } ^ { V } w _ { k } ^ { ( v ) } \pmb { S } _ { k } ^ { ( v ) } \Big ( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } _ { k } ^ { \mathrm { g t } } \big ) \Big ) .\tag{9}
$$

Equivalently, we minimize the training loss

$$
\mathcal { L } _ { \mathrm { M E O M } } = - \sum _ { k } \sum _ { v = 1 } ^ { V } w _ { k } ^ { ( v ) } \ \pmb { S } _ { k } ^ { ( v ) } \Big ( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } _ { k } ^ { \mathrm { g t } } \big ) \Big ) .\tag{10}
$$

Pure MEOM supervision can under-constrain absolute 3D scale and geometry. We therefore retain a 3D MSE term and train with the joint objective

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { M E O M } } \mathcal { L } _ { \mathrm { M E O M } } + \lambda _ { \mathrm { M S E } } \mathcal { L } _ { \mathrm { M S E } } , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { M S E } }$ is the smooth 3D keypoint MSE [14]. We train in two stages : Stage 1 uses MSE only $( \lambda _ { \mathrm { M E O M } } = 0 )$ ;

Stage 2 fine-tunes with the joint loss Eq.(11). Unless stated otherwise, we omit subsequent MEOM 3D refinement after DLT so that reported gains are attributable to decoding and training rather than post-hoc optimization.

## 3.4. Heatmap temperature and HDR calibration

Expected-OKS decoding and MEOM depend on the spatial mass of each heatmap, so poor calibration can distort triangulation. A heatmap is a multivariate forecast, thus calibration requires a pre-rank that reduces it to a scalar [17]; different pre-ranks induce quantile [16], HDR [4], and copula [29] diagnostics. Sec. 4.4 compares all three under temperature scaling against triangulation accuracy. Unlike distance-based metrics e.g., OKS [19] and MPJPE, we adopt HDR-ECE because it scores the frequency of groundtruth keypoints falling in the highest-density region of the predicted mass at each level. Similar to HDR, ProbPose states a requirement operationally for discrete heatmaps: the top 5% of map mass should contain 5% of ground-truth keypoints, the top 10% should contain 10%, and so on. We make this connection precise: ProbPose map calibration is the discrete case of HDR calibration.

Following [4], define the λ-density region of a predicted density ${ \hat { f } } _ { Y \mid x }$ by $\mathrm { D R } _ { x } ( \lambda ) : = \{ y : { \hat { f } } _ { Y | x } ( y ) \geq \lambda \}$ . For each level $p \in ( 0 , 1 )$ , the highest-density region is the smallest such level set whose predictive mass is at least $p ,$

$$
\begin{array} { r l } & { \operatorname { H D R } _ { x } ( p ) : = \operatorname { D R } _ { x } ( \lambda ^ { * } ) , } \\ & { \lambda ^ { * } = \operatorname* { s u p } \Big \{ \lambda : P \big ( \hat { Y } \in \operatorname { D R } _ { x } ( \lambda ) \mid X = x \big ) \ge p \Big \} . } \end{array}\tag{12}
$$

HDR calibration requires

$$
P \left( Y \in { \mathrm { H D R } } _ { X } ( p ) \right) = p \qquad \forall p \in ( 0 , 1 ) .\tag{13}
$$

Proposition 1 (ProbPose map calibration is HDR calibration). On a discrete spatial heatmap, sorting pixels by descending predicted density and taking the shortest prefix whose cumulative mass reaches a nominal level p yields exactly $\mathrm { H D R } _ { x } ( p )$ in (12). Consequently, requiring that ground-truth keypoints fall in these prefixes at rate p for all $p \in ( 0 , 1 )$ is equivalent to (13).

Proof. Order the heatmap pixels so that ${ \hat { f } } _ { Y \mid x } ( y _ { ( 1 ) } ) \geq$ ${ \hat { f } } _ { Y \mid x } ( y _ { ( 2 ) } ) \geq \cdot \cdot \cdot$ . For any threshold $\lambda , \operatorname { D R } _ { x } ( \lambda )$ is a prefix of this ordering. The supremum $\lambda ^ { * }$ in (12) selects the shortest prefix whose predictive mass is at least $p ,$ which is the “top-p mass” set used by ProbPose. HDR calibration (13) is therefore identical to requiring that these top-p sets cover the ground truth with empirical frequency p. □

Temperature as a discrete HDR recalibrator. To improve heatmap calibration, ProbPose applies a temperature T before spatial normalization, softmax $( \mathbf { \mathcal { H } } ^ { ( v ) } / T )$ , the post-hoc scalar that Guo et al. [9] use on classifiers. Prop. 1 makes (12) computable by sorting alone, so a heatmap needs neither density estimation nor continuity, and a onedimensional sweep over $T$ is a cheaper surrogate for the learned mapping of Chung et al. [4] (Appendix B). Flattening the softmax draws mass away from pixels denser than the ground truth, so per-level coverage error is V-shaped in $T$ and thus HDR-ECE has an interior minimum for each heatmap. These per-map minima need not share an optimal T, but the network predictions tends be to over-orunderconfident in the same direction, so a shared scalar still yields a global HDR-ECE optimum (Appendix B). The limitation is that temperature scaling never changes the density ordering thus cannot attain perfect calibraiton. That shared $T$ is nevertheless the practical operating point for triangulation: over-sharp maps degrade 3D error, and the HDR-ECEminimizing T is optimal for reprojection-error refinement and lies on the MEOM accuracy plateau (Sec. 4.4).

## 4. Experiments

## 4.1. Datasets and evaluation metrics

We follow the multi-view algebraic evaluation protocol in [14] by using absolute MPJPE in global coordinates as the primary metric, and reporting pelvis-relative MPJPE and PA-MPJPE only as secondary. Throughout the paper, expected-OKS decoding and MEOM use OKS kernels with the predefined per-joint constants $\sigma _ { k }$ of the COCO protocol [19]; retuning $\sigma _ { k }$ for Human3.6M and CMU Panoptic may further improve accuracy, but it is outside the scope of this work.

Human3.6M. We test on subjects S9/S11 with the four standard cameras and the common 17-joint skeleton. As in [14], scenes with erroneous 3D annotations are removed from Human3.6M. When 3D labels are unavailable (Sec. 4.2), we subsample every 64th frame. End-to-end training (Sec. 4.3) instead follows [14] by retaining every 5th frame with official lens undistortion. We additionally evaluate on H36MA [26], an ambiguous subset of Human3.6M selected for occlusion and complex pose.

Table 1. Performance of methods that do not use 3D joint labels on Human3.6M. DLT and UPose3D results are reported from [5], where absolute MPJPE is not reported.
<table><tr><td>Method</td><td>Backbone</td><td>Frames</td><td>MPJPE↓</td><td>Abs. ↓</td><td>PA↓</td><td>Params (M)</td></tr><tr><td>DLT [23]</td><td>ResNet-152</td><td>1</td><td>36.3</td><td></td><td></td><td>68.6</td></tr><tr><td>DLT [2]</td><td>CPN</td><td>1</td><td>30.5</td><td></td><td>27.6</td><td>27.0</td></tr><tr><td>UPose3D† [5] ResNet-152</td><td></td><td>1</td><td>31.0</td><td></td><td>29.0</td><td>68.6</td></tr><tr><td>UPose3D† [5] CPN</td><td></td><td>1</td><td>26.9</td><td></td><td>24.1</td><td>64.9</td></tr><tr><td>UPose3D† [5] CPN</td><td></td><td>27</td><td>26.4</td><td></td><td>23.4</td><td>64.9</td></tr><tr><td>MEOM (ours)</td><td>ProbPose-small (ViT-S)</td><td>1</td><td>32.42</td><td>36.04</td><td>30.38</td><td>24</td></tr></table>

<sup>†</sup>Trained on AMASS [20] with simulated 3D data.

Table 2. Triangulation results using three objectives without 3D labels on Human3.6M and H36MA [26] $\scriptstyle ( N = 6 2 3 8 )$ . Minimum per column in bold.
<table><tr><td></td><td colspan="3">Human3.6M</td><td colspan="3">H36MA</td></tr><tr><td>Method</td><td>MPJPE↓</td><td>Abs. ↓</td><td>PA↓</td><td>MPJPE↓</td><td>Abs. ↓</td><td>PA↓</td></tr><tr><td>Init (DLT, unweighted)</td><td>36.53</td><td>40.23</td><td>35.85</td><td>42.21</td><td>46.26</td><td>43.37</td></tr><tr><td>Init (DLT, OKS-weighted)</td><td>34.27</td><td>37.94</td><td>32.71</td><td>37.12</td><td>41.28</td><td>36.17</td></tr><tr><td>Reprojection error</td><td>34.69</td><td>38.11</td><td>32.89</td><td>37.84</td><td>41.70</td><td>36.63</td></tr><tr><td>Heatmap likelihood</td><td>69.79</td><td>61.48</td><td>61.31</td><td>115.10</td><td>92.25</td><td>94.21</td></tr><tr><td>MEOM</td><td>32.42</td><td>36.04</td><td>30.38</td><td>33.80</td><td>37.72</td><td>31.55</td></tr></table>

CMU Panoptic. We use the Panoptic validation frame ranges and four-camera set of [14]. Compared with Human3.6M, most CMU cameras do not see the full person for the entire sequence, leading to strong occlusions and missing body parts; the resulting incomplete or multimodal heatmaps require a robust objective to fuse across views.

## 4.2. MEOM triangulation without 3D labels

We evaluate the frozen-network pipeline of Sec. 3.2 on Human3.6M and CMU Panoptic. Unlike Iskakov et al. [14], we do not undistort the images and project with the provided camera matrices as-is.

Human3.6M. Because COCO and Human3.6M use different keypoint layouts, we fine-tune a COCO-pretrained ProbPose-small on Human3.6M 2D labels, then freeze it for triangulation. We initialize with view-weighted DLT and refine for 80 Adam steps, using ProbPose OKS predictions as view weights at both stages. Tab. 1 compares MEOM to multi-view methods that likewise use no 3D joint labels of the target dataset. Tab. 2 then ablates the refinement objective on Human3.6M and on H36MA under this frozen backbone. See full ablation results in Appendix D.1.

Against UPose3D [5] (Tab. 1), single-frame MEOM reaches 32.42 mm MPJPE (36.04 mm absolute), close to their ResNet-152 result (31.0 mm) and above DLT with the same backbone (36.3 mm). Their lower errors (26.9/26.4 mm) come with temporal fusion and a compiler trained on AMASS [20] large-scale simulated 3D data, neither of which MEOM uses. Among the three refinement objectives (Tab. 2), MEOM attains the lowest absolute MPJPE on both splits (36.04 mm on Human3.6M; 37.72 mm on H36MA), improving over OKS-weighted DLT and over reprojection error. The gap is larger on H36MA, where ambiguous 2D detections make point-based fusion brittle: MEOM reduces absolute error by 4.0 mm relative to reprojection error, whereas heatmap-likelihood refinement collapses (92.25 mm) and remains far worse than the initialization. Raw heatmap likelihood is likewise weak on Human3.6M (61.48 mm absolute): predicted heatmaps are sparse, with vanishing mass on most pixels, so maximizing H at the projected location (6) stalls in flat zero-mass regions.

![](images/fe2b26ed6db4fb990e9aae7f0306aa45f6b4d148ccb8b24336c419b9d74dd60f.jpg)  
Figure 4. View-count sensitivity on H36MA: absolute MPJPE vs. number of camera views. Fixed setting per objective across 4/3/2 views (y-axis truncated above 60 mm).

View-count sensitivity on H36MA. We subsample the first $V \in \{ 2 , 3 , 4 \}$ cameras from the fixed four-camera setting on H36MA. Fig. 4 reports absolute MPJPE versus V (y-axis truncated above 60 mm for readability). Full results are in Appendix D.1. Absolute error increases with fewer views; expected OKS remains the most accurate objective across all view counts.

CMU Panoptic. On this set, 2D observations are frozen COCO-pretrained ProbPose heatmaps with no CMU 2D or 3D fine-tuning; only algebraic DLT + refinement is evaluated. Person crops use Mask R-CNN (MRCNN) detections following [14]. Refinement objectives and view-weighting toggles match the Human3.6M protocol above. Reported numbers from Iskakov et al. [14] are absolute MPJPE with four cameras (Tab. 3 in [14]). Rankings among the three refinement objectives under the same frozen network are in Appendix D.2.

Table 3. Performance comparison on CMU Panoptic. Absolute MPJPE in millimeters. Results of Iskakov et al. are from [14] (Tab. 3). Minimum absolute MPJPE in bold.
<table><tr><td>Method</td><td>Backbone</td><td>3D labels Abs. MPJPE ↓</td><td></td></tr><tr><td>RANSAC [14] Algebraic [14]</td><td>ResNet-152</td><td>√</td><td>39.50 21.30</td></tr><tr><td>Volumetric [14]</td><td></td><td></td><td>13.70</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>MEOM (ours)</td><td>ProbPose-small (ViT-S)</td><td>一</td><td>23.99</td></tr></table>

With a frozen COCO ProbPose backbone and no CMU adaptation, MEOM reaches 23.99 mm absolute MPJPE— close to the algebraic baseline of Iskakov et al. [14] (21.30 mm), which is trained end-to-end on the target domain, and far below their RANSAC triangulation (39.50 mm). This near-parity under a mismatched, frozen 2D network indicates that Expected-OKS fusion is already at the level of strong algebraic multi-view methods, and that MEOM remains effective when self-occlusion and multiperson occlusion corrupt per-view heatmaps. The full ablation is in Appendix D.2; Expected OKS again ranks first among the three refinement objectives under the frozen COCO ProbPose backbone.

## 4.3. End-to-end triangulation with MEOM

We next train an algebraic triangulator end-to-end when 3D labels are available (Sec. 3.3; N≈26 610). The baselines are the official algebraic and volumetric checkpoints of Iskakov et al. [14]. We use the same ResNet-152 backbone as their algebraic model (Tab. 4), replacing only their softargmax 2D decoding by soft Expected-OKS decoding (8) with τ=0.02, followed by confidence-weighted DLT without post-hoc 3D refinement. Training follows the two-stage schedule of Sec. 3.3.2. Tab. 4 summarizes the main comparison; the volumetric and algebraic entries of Iskakov et al. [14] use the authors’ official pretrained weights.

Stage 1 reaches 19.38 mm / 22.15 mm, already improving over the algebraic baseline, which indicates that simply switching to expected-OKS decoding is already beneficial. Stage 2 with unfrozen confidences and the default MSE weight $\lambda _ { \mathrm { M S E } } { = } 1 0 0$ further improves to 19.11 mm absolute and 21.86 mm relative MPJPE—0.06 mm below the official volumetric model on absolute error while remaining 0.86 mm relative above it. The improvement brought by stage 2 over stage 1 shows that MEOM could further optimize the 3D joints. A broader grid over λ<sub>MSE</sub> (Appendix D.3) yields similar relative errors, indicating that our proposed loss remains stable.

These comparisons indicate that Expected-OKS-aligned algebraic training remains beneficial after lens undistortion, and that MSE+MEOM fine-tuning with an adaptable confidence head further improves over Stage 1, though volumetric aggregation still leads on relative MPJPE. Stage 1 and

Table 4. Performance of end-to-end algebraic triangulation on Human3.6M (every 5th frame). Absolute and root-relative MPJPE in millimeters. GFLOPs are fvcore inference multiply–adds. Best MPJPE in each error column in bold.
<table><tr><td>Method</td><td>Abs. MPJPE ↓</td><td>Rel. ↓</td><td>GFLOPs ↓</td></tr><tr><td>LTHP (volumetric) [14]</td><td>19.17</td><td>21.00</td><td>301.4</td></tr><tr><td>LTHP (algebraic) [14]</td><td>19.89</td><td>22.59</td><td>158.0</td></tr><tr><td>Stage 1 (MSE, soft Exp-OKS)</td><td>19.38</td><td>22.15</td><td>158.2</td></tr><tr><td>Stage 2 (MSE+MEOM)</td><td>19.11</td><td>21.86</td><td>158.2</td></tr></table>

Stage 2 use the same image-to-3D architecture at test time, so Tab. 4 reports 158.2 GFLOPs for both, matching the official algebraic model (158.0) and remaining at roughly half the computation of volumetric triangulation (301.4). The added Expected-OKS convolution is 0.162 GFLOPs— 919× cheaper than unprojection and the voxel-to-voxel (V2V) 3D CNN (149 GFLOPs); isolated-lift latency, memory, and the measurement protocol are in Appendix D.4.

## 4.4. Temperature Scaling and HDR calibration

Temperature sweep when 3D labels are unavailable. ProbPose’s heatmap head has a temperature T that controls heatmap calibration and sharpness. We freeze the trained checkpoint and regenerate heatmaps at various T on H36MA. The full grid of calibration scores and the three refinement objectives are in Appendix D.5; here Fig. 5 reports HDR-ECE and NLL against MEOM and reprojection-error refinement.

Better-calibrated map leads to better triangulation. On H36MA, HDR-ECE is unimodal in T and lowest at $T { = } 0 . 5$ , while over-sharp maps degrade MEOM triangulation (Fig. 5). Reprojection-error refinement isolates the effect of calibration on decoding: it reads the heatmap only through the OKS-decoded 2D joint and the OKS view weights, and never evaluates the response map during 3D optimization. Its absolute MPJPE is likewise unimodal in T and attains its minimum exactly where HDR-ECE does, at T=0.5 (Fig. 5, bottom left). This again shows that simply switching to expected-OKS decoding is already beneficial yet small. MEOM reads probability mass rather than an isolated peak, so the same miscalibration is more costly: it peaks slightly later at T=1.0 but is within 0.17 mm of that optimum at the HDR-optimal T=0.5 (Fig. 5, top left), i.e. inside a plateau spanning $T \in [ 0 . 5 , 1 . 5 ]$ ], while over-sharp maps cost up to 9.6 mm.

Which calibration score to trust. Relative to its own minimum, HDR-ECE spans an order of magnitude over the sweep, whereas NLL stays within about 2× and is nearly flat near T=0.75 (Fig. 5, right). It shows that temperature scaling swiftly improves HDR calibration, which is more indicative to the mass distribution reliability than NLL. Both scores place their optima on the MEOM plateau, but only HDR-ECE’s sharp V coincides with the reprojectionerror minimum at T=0.5. In Appendix D.5, we further show that both marginal quantile calibration and copula calibration improve over large T but fail to predict the MPJPE minimum for reprojection error and MEOM.

![](images/2dad6b1d31b7fc7f868c8b1597ab2d05dee618cc72e7a18d93223706c5662dff.jpg)  
Figure 5. Temperature scaling affects both heatmap calibration and triangulation accuracy. Calibration metrics relative to its minimum (blue, shared 1×–10× log scale) and refined absolute MPJPE (red) vs. ProbMapHead temperature $T ;$ stars mark each minimum. Reprojection error attains its optimum at the same T=0.5 that minimizes HDR-ECE; MEOM peaks at $T { = } 1 . 0 .$ within 0.17 mm of its value at T=0.5. HDR-ECE spans an order of magnitude; NLL remains within about 2× of its minimum. Note the different MPJPE ranges: 9.6 mm (MEOM) versus 0.73 mm (reprojection).

## 5. Conclusion

We presented MEOM, an Expected-OKS objective that unifies 2D decoding and multi-view algebraic fusion: a 3D joint is inferred where the views agree in probability mass, without collapsing heatmaps to peaks or constructing a voxel grid. With frozen 2D networks, maximizing MEOM outperforms point-based and likelihood refinement, especially under ambiguity and occlusion, and remains competitive with methods that use larger backbones, temporal fusion, and simulated 3D data. When 3D labels are available, the dual form of the same score trains an algebraic triangulator that matches the volumetric method accuracy but at a substantially lower inference cost.

Because MEOM fuses probability mass, poorly calibrated heatmaps degrade triangulation. A temperature that minimizes HDR-ECE is a practical operating point for both reprojection-error and MEOM refinement; NLL is less indicative of that mass reliability near the same optimum. We therefore recommend HDR calibration as a first-class diagnostic for mass-based multi-view pose, to be reported alongside MPJPE.

## References

[1] L. Bramlage, M. Karg, and C. Curio. Plausible uncertainties for human pose regression. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 15087–15096. IEEE, 2023. 3

[2] Y. Chen, Z. Wang, Y. Peng, Z. Zhang, G. Yu, and J. Sun. Cascaded pyramid network for multi-person pose estimation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7103–7112. IEEE, 2018. 6

[3] A. Chharia, W. Gou, and H. Dong. Mv-ssm: Multi-view state space modeling for 3d human pose estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 11590–11599, 2025. 2

[4] Y. Chung, I. Char, and J. Schneider. Sampling-based multidimensional recalibration. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 5, 6, 12

[5] V. Davoodnia, S. Ghorbani, M.-A. Carbonneau, A. Messier, and A. Etemad. Upose3d: Uncertainty-aware 3d human pose estimation with cross-view and temporal cues. In European Conference on Computer Vision (ECCV), 2024. 2, 6

[6] S. A. Ghasemzadeh, A. Alahi, and C. De Vleeschouwer. Rumpl: Ray-based transformers for universal multi-view 2d to 3d human pose lifting. arXiv preprint arXiv:2512.15488, 2025. 2

[7] K. Gu, R. Chen, and A. Yao. On the calibration of human pose estimation. arXiv preprint arXiv:2311.17105, 2023. 3

[8] N. B. Gundavarapu, D. Srivastava, R. Mitra, A. Sharma, and A. Jain. Structured aleatoric uncertainty in human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2019. 2, 3, 4

[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017. 2, 3, 6, 12

[10] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, 2 edition, 2003. 1, 2

[11] R. I. Hartley and P. Sturm. Triangulation. Computer Vision and Image Understanding, 68(2):146–157, 1997. 1, 2

[12] K. Holmquist and B. Wandt. Diffpose: Multi-hypothesis human pose estimation using diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15977–15987, 2023. 3

[13] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7): 1325–1339, 2013. 2

[14] K. Iskakov, E. Burkov, V. Lempitsky, and Y. Malkov. Learnable triangulation of human pose. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7718–7727, 2019. 1, 2, 4, 5, 6, 7, 8, 14, 15

[15] H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, and Y. Sheikh. Panoptic studio: A massively multiview system for social motion capture. In

Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3334–3342, 2015. 1, 2

[16] V. Kuleshov, N. Fenner, and S. Ermon. Accurate uncertainties for deep learning using calibrated regression. In International conference on machine learning, pages 2796–2804. PMLR, 2018. 3, 5

[17] A. Laajil, E. Zhalieva, N. Desobry, and S. Ben Taieb. Calibrated multivariate distributional regression with pre-rank regularization. In AISTATS 2026 Workshop on Towards Trustworthy Predictions: Theory and Applications of Cali brationfor Modern AI, Tangier, Morocco, 2026. 3, 5

[18] C. Le, P. Melnyk, B. Wandt, and M. Wadenback. Flow¨ matching for probabilistic monocular 3d human pose estimation. arXiv preprint arXiv:2601.16763, 2026. Accepted to TMLR. 3

[19] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft COCO: Com-´ mon objects in context. In European Conference on Computer Vision (ECCV), 2014. 3, 5, 6, 11

[20] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. Black. Amass: Archive of motion capture as surface shapes. in 2019 ieee. In CVF International Conference on Computer Vision (ICCV), pages 5441–5450, 2019. 6, 7

[21] D. Maji, S. Nagori, M. Mathew, and D. Poddar. Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2636–2645. IEEE, 2022. 3

[22] M. Purkrabek and J. Matas. Probpose: A probabilistic approach to 2d human pose estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27124–27133, 2025. 2, 3, 4, 11, 12

[23] H. Qiu, C. Wang, J. Wang, N. Wang, and W. Zeng. Cross view fusion for 3d human pose estimation. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4341–4350. IEEE, 2019. 6

[24] K. Sun, B. Xiao, D. Liu, and J. Wang. Deep high-resolution representation learning for human pose estimation. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 5686–5696. IEEE, 2019. 3

[25] H. Tu, C. Wang, and W. Zeng. Voxelpose: Towards multi camera 3d human pose estimation in wild environment. In European conference on computer vision, pages 197–212. Springer, 2020. 1, 2

[26] T. Wehrbein, M. Rudolph, B. Rosenhahn, and B. Wandt. Probabilistic monocular 3d human pose estimation with normalizing flows. In Proceedings of the IEEE/CVF Interna tional Conference on Computer Vision, pages 11199–11208, 2021. 2, 3, 6

[27] Y. Xu, J. Zhang, Q. Zhang, and D. Tao. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35:38571– 38584, 2022. 3

[28] Y. Yuan, R. Fu, L. Huang, W. Lin, C. Zhang, X. Chen, and J. Wang. Hrformer: High-resolution vision transformer for dense predict. Advances in neural information processing systems, 34:7281–7293, 2021. 3

[29] J. F. Ziegel and T. Gneiting. Copula calibration. Electronic Journal ofStatistics, 8(2):2619–2638, 2014. 3, 5

# MEOM: Multi-View Expected-OKS Maximization for Human Pose Triangulation

Supplementary Material

## A. Multi-view OKS as a factorized score

This appendix records notation shared by Sec. 3.2 in the main paper, the Expected-OKS decoding formulas, and a short probabilistic reading of multi-view fusion used in Sec. 3.1 in the main paper. The likelihood discussion is a modeling justification for finite V, not a claim that infinite cameras reconstruct the true 3D density by tomographic inversion.

Notation. Unless stated otherwise, $k \in \{ 1 , \ldots , K \}$ indexes joints and $v \in \{ 1 , \ldots , V \}$ indexes views. We write $\mathbf { X } _ { k } \in \mathbb { R } ^ { 3 }$ for a candidate 3D joint position, with DLT initialization $\mathbf { X } ^ { ( 0 ) }$ and refined estimate $\mathbf { X } ^ { * }$ as in Fig. 2 in the main paper; $\hat { \mathbf { x } } _ { k } ^ { ( v ) } \in \mathbb { R } ^ { 2 }$ for the 2D network output in view v $( \mathrm { F i g } . \ 2$ in the main paper: xˆ); $\mathbf { P } ^ { ( v ) }$ for the camera matrix and $\pi ( \mathbf { P } ^ { ( v ) } , \cdot )$ for its projection; $w _ { k } ^ { ( v ) }$ for the view weight in DLT and refinement; $\mathcal { H } _ { k } ^ { ( v ) }$ for the predicted heatmap on $\mathbb { R } ^ { 2 } ;$ ; and $\pmb { S } _ { k } ^ { ( v ) }$ for the Expected-OKS response map used by MEOM.

Factorized multi-view score. Let $\mathbf { X } \in \mathbb { R } ^ { 3 }$ be a 3D joint and write $\mathcal { I } = \{ I ^ { ( v ) } \} _ { v = 1 } ^ { V }$ for the calibrated views. Bayes rule gives the posterior

$$
p ( \mathbf { X } \mid { \mathcal { T } } ) = { \frac { p ( { \mathcal { T } } \mid \mathbf { X } ) p ( \mathbf { X } ) } { p ( { \mathcal { T } } ) } } .\tag{14}
$$

Conditional independence of the views given X factorizes the joint image likelihood into per-view terms,

$$
p ( \mathcal { T } \mid \mathbf { X } ) = \prod _ { v = 1 } ^ { V } p \Big ( I ^ { ( v ) } \mid \mathbf { X } \Big ) .\tag{15}
$$

Under a uniform spatial prior, $p ( \mathbf { X } )$ does not depend on X in the region of interest, so taking logarithms yields Eq. (2) in the main paper:

$$
\log p ( \mathbf { X } \mid { \mathcal { T } } ) = \sum _ { v = 1 } ^ { V } \log p { \Big ( } I ^ { ( v ) } \mid \mathbf { X } { \Big ) } + C ,\tag{16}
$$

where $C = \log p ( \mathbf { X } ) - \log p ( \mathcal { T } )$ collects the X-independent prior and evidence. Modeling

$$
p \Big ( I ^ { ( v ) } \mid \mathbf { X } \Big ) \propto \pmb { S } _ { k } ^ { ( v ) } \Big ( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } \big ) \Big )\tag{17}
$$

for the joint of interest yields the MEOM aggregation maximized in Eq. (7) in the main paper (up to monotone transforms and optional view weights $w _ { k } ^ { ( v ) } )$ . Equivalently, one may maximize

$$
\sum _ { v = 1 } ^ { V } \log \pmb { S } _ { k } ^ { ( v ) } \Big ( \pi \big ( \mathbf { P } ^ { ( v ) } , \mathbf { X } \big ) \Big )\tag{18}
$$

as an implicit continuous field over X without voxelization; our method evaluates related objectives via 2D OKS decoding and algebraic DLT/refinement rather than dense 3D search.

OKS kernel. The COCO OKS [19] between a query location x and a reference $\mathbf { x } ^ { \prime }$ for joint k is given by Eq. (3) in the main paper:

$$
\mathrm { O K S } _ { k } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \exp \left( - \frac { \| \mathbf { x } - \mathbf { x } ^ { \prime } \| _ { 2 } ^ { 2 } } { 2 s ^ { 2 } \sigma _ { k } ^ { 2 } } \right) ,\tag{19}
$$

where s is the subject scale and $\sigma _ { k }$ is the official perkeypoint constant (larger for hips and shoulders, smaller for eyes and wrists). We use the COCO $\sigma _ { k }$ without retuning; when the kernel is discretized on a heatmap, s is the subject scale implied by the activation-window resolution.

Expected-OKS response and hard decoding. Following ProbPose [22], let $p _ { L } ^ { ( v ) }$ be the localization distribution on the activation window AW. For each query pixel $( x , y ) \in A W$ the Expected-OKS response is the expectation of Eq. (19) under $p _ { L } ^ { ( v ) }$

$$
\begin{array} { l } { { \displaystyle { \pmb S } _ { k } ^ { ( v ) } ( x , y ) = \sum _ { ( x ^ { \prime } , y ^ { \prime } ) \in A W } p _ { L } ^ { ( v ) } ( x ^ { \prime } , y ^ { \prime } ) } } \\ { { \mathrm {  ~ \Omega ~ } \cdot \mathrm { O K S } _ { k } \big ( ( x , y ) , ( x ^ { \prime } , y ^ { \prime } ) \big ) , } } \end{array}\tag{20}
$$

or equivalently $\pmb { S } _ { k } ^ { ( v ) } = \pmb { \mathcal { H } } _ { k } ^ { ( v ) }$ ∗ ${ \bf K } _ { k }$ with the discretized kernel $\mathbf { K } _ { k } ( \mathbf { x } ) = \mathrm { O K } \dot { \mathrm { S } } _ { k } ( \mathbf { x } , \mathbf { 0 } )$ ,

$$
\pmb { S } _ { k } ^ { ( v ) } = \pmb { \mathcal { H } } _ { k } ^ { ( v ) } * \mathbf { K } _ { k } ,\tag{21}
$$

Hard decoding reads

$$
\hat { \mathbf { x } } _ { k } ^ { ( v ) } = \underset { ( x , y ) } { \arg \operatorname* { m a x } } \pmb { S } _ { k } ^ { ( v ) } ( x , y ) ,\tag{22}
$$

The response $\pmb { S } _ { k } ^ { ( v ) }$ is shared by DLT initialization and by MEOM (Eq. (7) in the main paper) in the no 3D supervision pipeline; hard-decoded $\hat { \mathbf { x } } _ { k } ^ { \left( v \right) }$ is used only to initialize DLT. Soft Expected-OKS decoding in Sec. 3.3.1 in the main paper likewise operates on $\pmb { S } _ { k } ^ { ( v ) }$

Temperature soft-argmax on the OKS response. The soft Expected-OKS decoding in Eq. (8) in the main paper applies softmax $( \pmb { S } _ { k } ^ { ( v ) } / \tau )$ on the OKS response. If two modes of $\pmb { S } _ { k } ^ { ( v ) }$ differ by a margin $\epsilon > 0$ , their soft weight ratio scales as $\mathrm { e } ^ { \epsilon / \tau } . \mathrm { A s } \tau  0$ , secondary modes are exponentially suppressed and the soft mean approaches arg max $\pmb { S } _ { k } ^ { ( v ) }$ , i.e., a differentiable mode tracker on the OKS map. Operating on $\pmb { S } _ { k } ^ { ( v ) } = \pmb { \mathcal { H } } _ { k } ^ { ( v ) } * \mathbf { K } _ { k }$ rather than on raw $\mathcal { H } _ { k } ^ { ( v ) }$ further attenuates single-pixel spikes before this selection, which mitigates (but does not categorically eliminate) expectation collapse under multimodality. Figures 6 and 7 illustrate these contrasts on synthetic 64×48 heatmaps.

## B. Temperature Scaling as a Discrete HDR Recalibrator

Sec. 3.4 in the main paper defines highest-density regions and shows that ProbPose map calibration is HDR calibration. Here we prove the monotonicity claimed there— raising T increases the HDR coverage for all levels $p \in$ (0, 1), so a one-dimensional sweep locates the HDR-ECE optimum and serves as a cheaper surrogate for the learned mapping of Chung et al. [4], and then bound what that single parameter can achieve, since minimizing this error and attaining exact HDR calibration are requirements at two different levels.

In [22], temperature scaling [9] is used as a post-hoc sharpness knob on a discrete heatmap. Write $h _ { i }$ for the predicted score of pixel $i \in \{ 1 , \ldots , N \}$ (assumed distinct), g for the ground-truth pixel, and

$$
A : = \{ i : h _ { i } > h _ { g } \} , \qquad U ( T ) : = \sum _ { i \in A } \pi _ { i } ( T ) ,\tag{23}
$$

with $\pi _ { i } ( T ) \propto \exp ( h _ { i } / T )$ , so that by Prop. 1 in the main paper the ground truth lies in $\mathrm { H D R } _ { x } ( p )$ if and only if $U ( T ) ~ < ~ p$ . With $\beta ~ = ~ 1 / T , ~ Z ~ = ~ \sum _ { i }$ exp(βh<sub>i</sub>) and $Z _ { A } = \sum _ { i \in A }$ exp(βh ), we have $U = Z _ { A } / \dot { Z }$ and

$$
\frac { \mathrm { d } } { \mathrm { d } \beta } \log \frac { Z _ { A } } { Z } = \mathbb { E } _ { A } [ h ] - \mathbb { E } [ h ] > 0\tag{24}
$$

whenever $A \neq \emptyset$ , since every score in A exceeds every score outside it. Raising T therefore decreases U on every sample simultaneously, from $U \ \to \ 1$ as $T  0 ^ { + }$ to $U \ \to \ | A | / N$ as $T \ \to \ \infty ;$ flattening the softmax draws mass away from the pixels denser than the ground truth, so the share of mass lying above it falls. Write ${ \widehat { c } } ( p ; T )$ for the empirical coverage at level $p ,$ the fraction of samples with $U ( T ) ~ < ~ p$ . Since every U decreases in $T ,$ a sample once counted stays counted, so ${ \widehat { c } } ( p ; \cdot )$ is non-decreasing and crosses the level p at most once. The per-level calibration error is $| { \widehat { c } } ( p ; T ) - p |$ , an absolute value composed with a monotone function of T, hence V-shaped: it falls while ${ \widehat { c } } ( p ; T ) < p ,$ vanishes at the crossing, and rises once ${ \widehat { c } } ( p ; T ) ~ > ~ p .$ HDR-ECE averages these V-shaped curves over levels whose crossings need not coincide, so monotonicity guarantees an interior optimum reachable by a onedimensional sweep, though not a unique one in general; on our data it is unique, at $T { \approx } 0 . 5$ (Sec. 4.4 in the main paper). Per-map crossings need not agree, but modern networks tend to be overconfident or underconfident in the same direction [9] when tested on the same dataset, so those crossings cluster and the dataset operating interval stays tight. Thus, a global temperature learned on the validation set can minimize HDR-ECE on the test set, which is why we report HDR-ECE alongside 3D error.

![](images/4fd4f340f7010cc61919b437eb0ff1fc09d34ef59dfbab76991b52692fadf8ee.jpg)

![](images/e60d1599654779b512ff9e2146a618d094794f8385dd1dd04bcaac97d7678b40.jpg)

![](images/b8be888916f665b59c9a66d52639d8dc5f04b1c8524756d79435e8e56e818bfd.jpg)

![](images/d181375d03b43e4fd320d2de3882c52372effacb5b2b95152b29ddaf006e6590.jpg)

![](images/014a08a4be57330b6f868b897349ec9d4344b2c53e3e504730c5afaf9988d2f4.jpg)

![](images/396a072d30661caaa575a6157299a6ba436f9a1f31a37220fc47ecebcc00be6a.jpg)  
Figure 6. OKS decoding vs. argmax and soft-argmax on synthetic heatmaps $( 6 4 \times 4 8 )$ . Left: raw heatmap H; right: OKS-convolved response $\pmb { S } = \pmb { \mathcal { H } } * \mathbf { K }$ . Markers show arg max on the raw map (red ×), soft-argmax / center of mass on the raw map (blue +), hard OKS decode arg max S (green ◦), and—in the bottom row— soft-argmax on S (brown ⋄). First row: On a unimodal Gaussian, all estimators coincide. Second row: Hard OKS decoding selects the mass-dominant right mode from a bimodal mixture, while raw arg max prefers the sharper left peak, and raw soft-argmax collapses between modes; Third row: Soft-argmax with $\tau { = } 1$ on S coincides with Soft-argmax with τ=1 on raw H.

![](images/d8e43ca0c06e6ce90abbd18b9e133fdba3cfe099742581c8ec099f7711c00f02.jpg)  
Figure 7. Temperature soft-argmax on a bimodal heatmap approaches hard OKS decode. Left: raw H; right: OKS response $s .$ As temperature τ decreases (light→dark along the orange path), soft-argmax moves from a broad center of mass toward a mode. On S, the operating point τ=0.02 (red star) coincides with hard OKS decode (green ◦), providing a differentiable proxy used in Eq. (8) in the main paper. Soft-argmax on raw H remains farther from the OKS mode even at the same τ.

Temperature scaling is not without limitations. Exact HDR calibration (Eq. (13) in the main paper) is instead a distributional requirement on the whole dataset, $U \sim$ Unif(0, 1), which asks one shared scalar to satisfy every level at once when the per-sample sets A are fixed and mutually inconsistent. Minimizing HDR-ECE is thus attainable while uniformity in general is not: a temperature corrects sharpness, but no T moves a ground truth past a competing mode. Post-hoc calibration of a trained heatmap is bounded by this, which is why we report HDR-ECE alongside 3D error rather than treating a recalibrated map as fully calibrated.

## C. ProbPose fine-tuning on Human3.6M

COCO and Human3.6M both use 17 keypoints, but the joint sets and channel orders differ: COCO is organized around face and limb landmarks, whereas Human3.6M follows a kinematic tree with a pelvis root, spine, thorax, and neck. A COCO-pretrained heatmap head is therefore not aligned with Human3.6M annotations. We start from the official COCO-pretrained ProbPose-small checkpoint and fine-tune it on the standard Human3.6M training subjects S1, S5, S6, S7, and S8, so that the output heatmaps follow the Human3.6M joint order. Subjects S9 and S11 are held out for testing; bbox-normalized PCK@0.1 on this split peaks at 0.976 by epoch 5.

## D. Additional experimental tables

This appendix collects triangulation tables, the end-to-end compute protocol, and temperature-sweep results deferred from Sec. 4 in the main paper.

Table 5. Full triangulation ablation on Human3.6M S9/S11 every 64th frame $( N { = } 2 1 8 1 )$ . Presence and OKS view weights at DLT / refinement. Minimum per column in bold.
<table><tr><td>Objective</td><td>DLT / Ref.</td><td>MPJPE↓</td><td>Abs. ↓</td><td>PA↓</td></tr><tr><td>Init (DLT only)</td><td> $- / -$ </td><td>36.53</td><td>40.23</td><td>35.85</td></tr><tr><td>Init (DLT only)</td><td>OKS / -</td><td>34.27</td><td>37.94</td><td>32.71</td></tr><tr><td>Reprojection error</td><td>Presence (-/-)</td><td>35.76 35.76</td><td>39.21</td><td>34.38</td></tr><tr><td></td><td>Presence  $( - / \check { \mathbf { \nabla } } )$ </td><td></td><td>39.21</td><td>34.37</td></tr><tr><td></td><td>Presence (√/-)</td><td>35.76</td><td>39.21</td><td>34.38</td></tr><tr><td></td><td>Presence  $( \surd / \surd )$ </td><td>35.76</td><td>39.21</td><td>34.37</td></tr><tr><td></td><td>OKS (−/-)</td><td>35.76</td><td>39.21</td><td>34.38</td></tr><tr><td></td><td>OKS  $( - / \check { \mathbf { \nabla } } )$ </td><td>34.67</td><td>38.10</td><td>32.88</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>35.78</td><td>39.23</td><td>34.39</td></tr><tr><td></td><td>OKS (√/√)</td><td>34.69</td><td>38.11</td><td>32.89</td></tr><tr><td>Heatmap likelihood</td><td>Presence (-/-)</td><td>69.76</td><td>61.65</td><td>62.07</td></tr><tr><td></td><td>Presence (-/√)</td><td>69.78</td><td>61.66</td><td>62.12</td></tr><tr><td></td><td>Presence  $( \checkmark / - )$ </td><td>69.84</td><td>61.73</td><td>62.18</td></tr><tr><td></td><td>Presence  $( \surd / \surd )$ </td><td>69.84</td><td>61.72</td><td>62.15</td></tr><tr><td></td><td>OKS (-/-)</td><td>69.76</td><td>61.65</td><td>62.07</td></tr><tr><td></td><td>OKS  $( - / \check { \mathbf { \nabla } } )$ </td><td>71.93</td><td>62.13</td><td>62.51</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>68.35</td><td>60.93</td><td>60.62</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>69.79</td><td>61.48</td><td>61.31</td></tr><tr><td>Expected OKS</td><td>OKS (-/-)</td><td>33.12</td><td>36.81</td><td>31.51</td></tr><tr><td></td><td>OKS  $( - / \check { \mathbf { \nabla } } )$ </td><td>32.75</td><td>36.39</td><td>30.95</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>32.74</td><td>36.42</td><td>30.83</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>32.42</td><td>36.04</td><td>30.38</td></tr></table>

## D.1. Human3.6M triangulation without 3D labels: full presence/OKS grids

Tabs. 5 and 6 expand Tabs. 2 in the main paper with presence-probability vs. predicted OKS view weighting and DLT / refinement ON/OFF toggles. Presence weighting barely moves reprojection error, while Expected OKS remains best under OKS weights in both splits.

## D.2. CMU Panoptic triangulation ablation

Tab. 7 expands the CMU comparison of Sec. 4.2 in the main paper with the full objective and view-weighting grid under MRCNN boxes. The absolute MPJPE 23.99 mm reported in the main text is the Expected-OKS row with OKS weighting at both DLT and refinement.

## D.3. End-to-end MSE weight ablation

Tab. 8 expands Tab. 4 in the main paper with Stage 2 joint MSE+MEOM fine-tuning under MEOM weight 1.0 and λ<sub>MSE</sub> ∈ {10, 50, 100, 200, 500}. Relative MPJPE is best at $\lambda _ { \mathrm { M S E } } { = } 1 0 0 ;$ absolute MPJPE is marginally best at $\lambda _ { \mathrm { M S E } } { = } 5 0 0 .$

## D.4. Computational cost of 2D-to-3D lifting

Tab. 4 in the main paper reports full-pipeline GFLOPs for the three end-to-end methods that share a ResNet-152 heatmap backbone on $3 8 4 \times 3 8 4$ crops from four views. The distinguishing computation is the lift from heatmaps (or deconvolution features) to 3D joints: confidence-weighted DLT after soft-argmax in the algebraic baseline of Iskakov et al. [14], the same DLT after Expected-OKS decoding (Eq. (8) in the main paper) in our method, and dense unprojection followed by a V2V 3D CNN in the volumetric model [14]. Stage 1 and Stage 2 use the same Expected- $\mathrm { O K S + D L T }$ inference graph, so they share the 158.2 GFLOPs entry.

Table 6. Full triangulation ablation on H36MA (N=6 238). Presence and OKS view weights at DLT / refinement. Minimum per column in bold.
<table><tr><td>Objective</td><td>DLT / Ref.</td><td>MPJPE↓</td><td> $\operatorname { A b s . } \downarrow$ </td><td> $\mathrm { P A } \downarrow$ </td></tr><tr><td>Init (DLT only)</td><td> $- / -$ </td><td>42.21</td><td>46.26</td><td>43.37</td></tr><tr><td>Init (DLT only)</td><td> $\mathrm { O K S } / -$ </td><td>37.12</td><td>41.28</td><td>36.17</td></tr><tr><td>Reprojection error</td><td>Presence  $( - / - )$ </td><td>40.36</td><td>44.17</td><td>40.20</td></tr><tr><td></td><td>Presence  $( - / \check { \pmb { \mathscr { I } } } )$ </td><td>40.36</td><td>44.17</td><td>40.19</td></tr><tr><td></td><td>Presence  $( \checkmark / - )$ </td><td>40.36</td><td>44.17</td><td>40.20</td></tr><tr><td></td><td>Presence  $( \surd / \surd )$ </td><td>40.36</td><td>44.17</td><td>40.19</td></tr><tr><td></td><td>OKS  $( - / - )$ </td><td>40.36</td><td>44.17</td><td>40.20</td></tr><tr><td></td><td>OKS  $( - / \check { \pmb { \mathscr { I } } } )$ </td><td>37.80</td><td>41.67</td><td>36.61</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>40.40</td><td>44.19</td><td>40.23</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>37.84</td><td>41.70</td><td>36.63</td></tr><tr><td>Heatmap likelihood</td><td>Presence (−/-)</td><td>112.94</td><td>90.36</td><td>94.49</td></tr><tr><td></td><td>Presence  $( - / \check { \pmb { \mathscr { I } } } )$ </td><td>112.85</td><td>90.31</td><td>94.45</td></tr><tr><td></td><td>Presence  $( \checkmark / - )$ </td><td>112.82</td><td>90.32</td><td>94.40</td></tr><tr><td></td><td>Presence  $( \surd / \surd )$ </td><td>112.86</td><td>90.30</td><td>94.42</td></tr><tr><td></td><td>OKS  $( - / - )$ </td><td>112.94</td><td>90.36</td><td>94.49</td></tr><tr><td></td><td>OKS  $( - / \check { \pmb { \mathscr { I } } } )$ </td><td>116.99</td><td>91.90</td><td>95.48</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>111.62</td><td>90.58</td><td>92.94</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>115.10</td><td>92.25</td><td>94.21</td></tr><tr><td>Expected OKS</td><td>OKS (-/-)</td><td>35.29</td><td>39.26</td><td>34.08</td></tr><tr><td></td><td>OKS  $( - / \check { \pmb { \mathscr { I } } } )$ </td><td>34.67</td><td>38.47</td><td>32.92</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>34.28</td><td>38.31</td><td>32.30</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>33.80</td><td>37.72</td><td>31.55</td></tr></table>

To isolate the lift we wrap each as a standalone module, feed dummy tensors with the Human3.6M inference shapes (batch size 1, four views, $9 6 \times 9 6$ heatmaps, $6 4 ^ { 3 }$ volume), and count multiply–adds with the fvcore flop counter. Peak allocated GPU memory is recorded by resetting the CUDA peak tracker immediately before a forward pass and reading the subsequent maximum; latency is the mean of 30 CUDAevent timings after warmup on an NVIDIA A100.

Tab. 9 reports the lift alone and the implied full imageto-3D budget obtained by adding the shared 2D backbone (≈ 158 GFLOPs for the algebraic models; 152 GFLOPs for volumetric triangulation, which omits the algebraic confidence head). Analytic convolution counts agree with fvcore on the OKS kernels and on the V2V network. Expected-OKS decoding adds a per-joint convolution of

Table 7. Triangulation ablation on CMU Panoptic validation [14] — MRCNN bounding boxes $( N = 8 1 1 0 )$ . Pre-computed Mask R-CNN detections from [14]. Four views per sample. Viewweight columns list DLT / refinement $( \mathrm { O N } = \surd , \mathrm { O F F = - } )$ . Init: unweighted and OKS-weighted DLT (no refinement). Minimum per metric column in bold.
<table><tr><td>Objective</td><td>DLT / Ref.</td><td>MPJPE↓</td><td>Abs. ↓</td><td> $\mathrm { P A } \downarrow$ </td></tr><tr><td>Init (DLT only)</td><td> $- / -$ </td><td>50.74</td><td>37.18</td><td>42.45</td></tr><tr><td>Init (DLT only)</td><td> $\mathrm { O K S } / -$ </td><td>44.86</td><td>28.52</td><td>31.41</td></tr><tr><td>Reprojection error</td><td>Presence  $( - / - )$ </td><td>49.10</td><td>35.65</td><td>41.47</td></tr><tr><td></td><td>Presence  $( - / \sqrt { } )$ </td><td>49.06</td><td>35.60</td><td>41.39</td></tr><tr><td></td><td>Presence  $( \checkmark / - )$ </td><td>49.10</td><td>35.64</td><td>41.45</td></tr><tr><td></td><td>Presence  $( \checkmark / \check { \checkmark } )$ </td><td>49.05</td><td>35.58</td><td>41.37</td></tr><tr><td></td><td>OKS (−/-)</td><td>49.10</td><td>35.65</td><td>41.47</td></tr><tr><td></td><td>OKS  $( - / \sqrt { } )$ </td><td>46.14</td><td>31.00</td><td>35.70</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>49.18</td><td>35.82</td><td>41.59</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>46.05</td><td>31.15</td><td>35.70</td></tr><tr><td>Heatmap likelihood</td><td>Presence  $( - / - )$ </td><td>128.11</td><td>113.65</td><td>138.97</td></tr><tr><td></td><td>Presence  $( - / \sqrt { } )$ </td><td>128.01</td><td>113.54</td><td>138.89</td></tr><tr><td></td><td>Presence  $( \checkmark / - )$ </td><td>127.74</td><td>113.24</td><td>138.65</td></tr><tr><td></td><td>Presence  $( \checkmark / \check { \checkmark } )$ </td><td>127.72</td><td>113.22</td><td>138.64</td></tr><tr><td></td><td>OKS  $( - / - )$ </td><td>128.11</td><td>113.65</td><td>138.97</td></tr><tr><td></td><td>OKS  $( - / \sqrt { } )$ </td><td>133.89</td><td>119.25</td><td>145.01</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>128.05</td><td>113.62</td><td>140.53</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>133.96</td><td>119.57</td><td>147.10</td></tr><tr><td>Expected OKS</td><td>OKS (-/-)</td><td>42.80</td><td>24.95</td><td>26.47</td></tr><tr><td></td><td>OKS  $( - / \sqrt { } )$ </td><td>42.71</td><td>24.39</td><td>25.82</td></tr><tr><td></td><td>OKS  $( \checkmark / - )$ </td><td>42.29</td><td>24.40</td><td>25.64</td></tr><tr><td></td><td>OKS  $( \surd / \surd )$ </td><td>42.33</td><td>23.99</td><td>25.18</td></tr></table>

Table 8. Human3.6M end-to-end Stage 2 MSE weight ablation on the undistorted protocol of Sec. 4.1 in the main paper (every 5th frame). Absolute and root-relative MPJPE in millimeters (lower is better). Soft Expected-OKS decoding $\scriptstyle ( \tau = 0 . 0 2 )$ ; MEOM weight 1.0. Official baselines and Stage 1 repeated from Tab. 4. Best algebraic numbers in each column are in bold.
<table><tr><td>Method</td><td>Abs. MPJPE ↓</td><td>Rel. MPJPE ↓</td></tr><tr><td>Iskakov et al. (volumetric) [14] Iskakov et al. (algebraic) [14]</td><td>19.17 19.89</td><td>21.00 22.59</td></tr><tr><td></td><td></td><td></td></tr><tr><td>hm× 1 Stage 1 (MSE, soft Exp-OKS) hm×1 Stage 2 (MSE+MEOM,</td><td>19.38 19.30</td><td>22.15 22.12</td></tr><tr><td> $\lambda _ { \mathrm { M S E } } { = } 1 0 )$ </td><td>19.20</td><td></td></tr><tr><td>hm×1 Stage 2 (MSE+MEOM,  $\lambda _ { \mathrm { M S E } } { = } 5 0 )$  hm×1 Stage 2 (MSE+MEOM,</td><td>19.11</td><td>21.98 21.86</td></tr><tr><td> $\lambda _ { \mathrm { M S E } } { = } 1 0 0 )$  hm×1 Stage  $2 \ ( \mathbf { M S E + M E O M } .$   $\lambda _ { \mathrm { M S E } } { = } 2 0 0 )$ </td><td>19.24</td><td>22.10</td></tr><tr><td>hm× 1 Stage 2 (MSE+MEOM,</td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { M S E } } { = } 5 0 0 )$ </td><td>19.10</td><td>22.03</td></tr></table>

0.162 GFLOPs—two orders of magnitude above algebraic soft-argmax and DLT (1.3 MFLOPs), yet 919× cheaper than volumetric unprojection and V2V (149 GFLOPs). Consequently the full algebraic MSE+MEOM pipeline remains at 158 GFLOPs, whereas volumetric triangulation reaches 301 GFLOPs. Wall-clock time and memory follow the same ordering: the Expected-OKS lift runs in 1.9 ms with 12 MB peak allocation, versus 23 ms and 547 MB for the volumetric lift.

![](images/1323b648ebeaf76f81920faa59eed771fac2dd928a19dec363653f5cfe539a98.jpg)  
Figure 8. Same H36MA temperature sweep as Fig. 5 in the main paper, with every left axis showing the calibration score relative to its minimum on a shared log scale (1×–10×). HDR-ECE spans an order of magnitude; ECE, copula ECE, and NLL remain within about 2× of their minima, and NLL is nearly flat near the MEOM optimum.

Table 9. Cost of the 2D-to-3D lift with the shared ResNet-152 backbone excluded (batch size 1, four 384 × 384 views, 96 × 96 heatmaps, $6 4 ^ { 3 }$ volume). Multiply–adds are counted by fvcore. Full GFLOPs add the 2D backbone measured on the same dummy input. Latency and peak allocated memory are measured on an NVIDIA A100. Algebraic: soft-argmax + DLT. Ours: Expected-OKS convolution + temperature soft-argmax + DLT. Volumetric: 1×1 feature projection, unprojection, and V2V [14].

<table><tr><td>Method</td><td>Lift (GFLOPs)</td><td>Full (GFLOPs)</td><td>Latency (ms)</td><td>Peak mem. (MB)</td></tr><tr><td>Algebraic (soft-argmax + DLT)</td><td>0.001</td><td>158.0</td><td>0.74</td><td>4.9</td></tr><tr><td>Ours (Expected-OKS + DLT)</td><td>0.162</td><td>158.2</td><td>1.91</td><td>12.0</td></tr><tr><td>Volumetric (unproject + V2V) [14]</td><td>149.0</td><td>301.4</td><td>22.67</td><td>547.3</td></tr></table>

## D.5. H36MA temperature sweep

Sec. 4.4 in the main paper uses $T \in$ {0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}. For each T we evaluate heatmap calibration on visible joints (visibility = 2): axis-wise ECE, copula ECE, HDR-ECE, NLL, and energy score; and algebraic DLT followed by 80-step refinement under the three objectives of Sec. 4.2 in the main paper, with view weighting on at both stages (expected OKS and reprojection error use OKS weights; heatmap likelihood uses presence). Reported 3D errors are absolute MPJPE after refinement. Tab. 10 lists per-temperature calibration metrics and refined absolute MPJPE for the H36MA sweep discussed in Sec. 4.4 in the main paper. Fig. 8 replots the same dual-axis curves with a shared relative left axis (1×–10×) so that basin amplitudes can be compared across scores.

Table 10. H36MA temperature sweep (N=6 238). Inference-time ProbMapHead temperature T vs. heatmap calibration and refined absolute MPJPE (mm) for expected OKS (ExpOKS), reprojection error (Repro), and heatmap likelihood (Lik), all with DLT/refine view weighting ON. Minimum Abs. MPJPE per objective in bold.
<table><tr><td>T</td><td>ECEx</td><td>ECEy</td><td>Copula</td><td>HDR</td><td>NLL</td><td>ExpOKS</td><td>Repro</td><td>Lik</td></tr><tr><td>0.05</td><td>0.328</td><td>0.354</td><td>0.309</td><td>0.265</td><td>7.015</td><td>47.15</td><td>42.43</td><td>267.24</td></tr><tr><td>0.10</td><td>0.320</td><td>0.347</td><td>0.307</td><td>0.211</td><td>5.892</td><td>43.27</td><td>42.00</td><td>213.90</td></tr><tr><td>0.15</td><td>0.313</td><td>0.340</td><td>0.301</td><td>0.173</td><td>5.104</td><td>40.68</td><td>41.87</td><td>180.41</td></tr><tr><td>0.25</td><td>0.298</td><td>0.328</td><td>0.294</td><td>0.109</td><td>4.108</td><td>38.66</td><td>41.77</td><td>135.88</td></tr><tr><td>0.50</td><td>0.268</td><td>0.303</td><td>0.279</td><td>0.026</td><td>3.645</td><td>37.72</td><td>41.70</td><td>90.30</td></tr><tr><td>0.75</td><td>0.249</td><td>0.286</td><td>0.269</td><td>0.051</td><td>3.600</td><td>37.58</td><td>41.71</td><td>77.67</td></tr><tr><td>1.00</td><td>0.233</td><td>0.272</td><td>0.262</td><td>0.087</td><td>3.618</td><td>37.55</td><td>41.77</td><td>72.30</td></tr><tr><td>1.50</td><td>0.208</td><td>0.253</td><td>0.254</td><td>0.143</td><td>3.704</td><td>37.60</td><td>41.86</td><td>66.39</td></tr><tr><td>2.00</td><td>0.190</td><td>0.241</td><td>0.250</td><td>0.178</td><td>3.798</td><td>37.65</td><td>41.94</td><td>62.79</td></tr><tr><td>3.00</td><td>0.171</td><td>0.222</td><td>0.241</td><td>0.224</td><td>3.968</td><td>37.79</td><td>42.06</td><td>59.55</td></tr></table>

## D.6. Likelihood under stronger softening

Because likelihood refinement continued to improve up to T=3 in Sec. 4.4 in the main paper, we extended the H36MA likelihood sweep to $T ~ \in ~ \{ 4 , 5 , 6 , 8 , 1 0 , 1 5 , 2 0 \}$ (Fig. 9). Absolute MPJPE decreases further and plateaus near 56.4 mm at $T { = } 2 0$ , still ≈19 mm above the best expected-OKS result (37.5 mm at $T { = } 1 . 0 )$ Thus temperature can partially mitigate the mismatch between raw heatmap likelihood and 3D accuracy, but expected-OKS decoding remains substantially more robust and accurate under the same ProbPose predictions.

![](images/4791fce5f7e2f87b3ffa55edcd891d8ad8bf54a2fb9289bc53cc15f7e0ec111e.jpg)  
Figure 9. H36MA: heatmap-likelihood absolute MPJPE vs. ProbMapHead temperature (log x-axis), extended to T=20. The dashed line marks the best expected-OKS absolute MPJPE over $T \in [ 0 . 0 5 , 3 ]$ (37.5 mm). Likelihood improves with softening but saturates well above expected OKS.