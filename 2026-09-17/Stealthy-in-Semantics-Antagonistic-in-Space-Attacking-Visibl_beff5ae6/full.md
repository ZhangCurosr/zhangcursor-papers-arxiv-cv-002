# Stealthy in Semantics, Antagonistic in Space: Attacking Visible-Infrared Object Detectors via Object-Level Misalignment

Yueqi Zhu<sup>1,∗</sup>, Qi Ming<sup>1,∗</sup>, Guo Cheng<sup>1</sup>, Yongkang Zhang<sup>1</sup>, Feiran Liu<sup>1</sup>, Juan Fang<sup>1</sup>, Jiahuan Zhou<sup>2</sup>, Jiangmeng Li<sup>3</sup>, Yuhan Zhang<sup>4</sup>

<sup>1</sup>Beijing University of Technology <sup>2</sup>Peking University

<sup>3</sup>National Key Laboratory of Space Integrated Information System, Institute of Software, Chinese Academy of Sciences

<sup>4</sup>Intelligent Science & Technology Academy of CASIC

<sup>∗</sup>Equal contribution. 15510323848@emails.bjut.edu.cn

## Abstract

Visible-infrared object detectors are usedfor robust perception under challenging illumination and weather conditions. Current physical attacks apply conspicuous patches to spatially aligned target regions, which are noticeable to human observers. Meanwhile, most of these methods only perturb the appearance within the aligned region, without explicitly targeting the correspondence between modalities or thefusion process. In this paper, we propose CamoShift, an adversarial frameworkfor visible-infrared object detection. By combining visual camouflage with object-level infrared shifting, CamoShift breaks cross-modal spatial alignment and disrupts fusion. Specifically, the Semantic Camouflage Module (SCM) generates a stealthy camouflaged patch that can be attached to the host object and maintains its effectiveness in the infrared branch through an RGB-IR adapter. The Object-level Spatial Decoupling Module (OSDM) shifts the infrared target evidence in a scale-aware manner, so as to break object-level correspondence and disrupt cross-modal fusion. Then, the Harmonic Adversarial loss (HarAdv loss) further balances attack strength and visual stealth during optimization. To the best of our knowledge, we are the first to target both visual stealthiness and attack success in visible-infrared object detection. Extensive experimental results show that CamoShift achieves a superior balance between attack effectiveness and visual stealth. Code and models will be available on GitHub.

## 1. Introduction

Object detection enables systems to rapidly identify and locate target objects [23, 27, 50]. Visible-infrared object detectors are increasingly being adopted to tackle challenging adverse environments. While RGB cameras capture fine visual details, infrared sensors provide robust infrared cues that effectively overcome glare and fog [3, 10, 16, 25]. Despite their robustness to environmental challenges, the underlying deep learning models remain highly vulnerable to adversarial attacks [5, 14, 28].

![](images/443539ad8dfe610ba6786cb0503fa6ac746f386be99e54cc2c9638362daca3a9.jpg)  
Figure 1. Comparison of adversarial attacks on cross-modal detectors. (a) Single-modal attacks manipulate only visible textures, inherently failing in the infrared domain. (b) Cross-modal adversarial attacks rely on conspicuous artificial patterns and strictly maintain object-level spatial alignment, failing to disrupt the fusion mechanism. (c) Our framework achieves both visual stealthiness and high attack success. It combines semantic camouflage with object-level spatial decoupling to break cross-modal alignment, successfully misleading the detector.

Physical adversarial patches aim to mislead object detection systems by deploying crafted patterns or materials in the physical world. To achieve this, mainstream methods have evolved from single-modal to visible-infrared strategies. Early approaches primarily targeted single-modal object detectors by optimizing textures or geometric shapes [1, 9, 35, 40]. However, these pure RGB patches fail in the infrared domain due to the lack of infrared signatures. To mislead both sensors simultaneously, recent advancements have begun to explore cross-modal physical attacks by deploying crafted multispectral adversarial patches (e.g., combining visual patterns with infrared materials) [34, 48, 49]. As a foundational step, CDUPatch [21] exploits the physical mapping between RGB colors and infrared absorption to generate patches that mislead both branches.

![](images/8177863fd763366c11d486941bc0cfe31a5a6bce3b46a6f0b0ed579be7e99743.jpg)  
Figure 2. Performance comparison of visual stealthiness and attack success on the DroneVehicle dataset.

Despite their initial success, existing cross-modal adversarial patch attacks exhibit two critical limitations: (1) Highly conspicuous visual appearance. To generate sufficient infrared differences, existing methods [7, 21, 36, 49] are forced to use extreme RGB color contrasts (e.g., pure black and white blocks). Consequently, these patches manifest as highly unnatural, conspicuous artificial patterns that cannot blend into the background, as shown in Fig. 1(b). They are instantly detectable by human observers, failing the basic requirement of remaining unnoticeable. (2) Strict reliance on object-level spatial alignment. Current crossmodal detectors fundamentally assume strict spatial alignment between RGB and IR inputs. These systems strictly treat the spatial misalignment as a global, image-level sensor calibration error. Existing attacks [15, 32, 36] blindly inherit this assumption, confining their perturbations to corrupting features within perfectly matched bounding boxes. However, they ignore a more fundamental issue that an adversarial patch can actively shift the location of the target object in a single modality. As illustrated in Fig. 1(c), target features no longer spatially align at the object level across the two modalities, which prevents the network from pairing them and causes the fusion mechanism to fail.

In this paper, to expose the vulnerability of spatial alignment mechanisms and enhance attack stealthiness, we propose a framework named CamoShift. First, we design a Semantic Camouflage Module (SCM) to generate visually stealthy adversarial patches. SCM uses style-consistent, color-driven generation techniques to create adversarial textures, enabling them to seamlessly blend into the background. Compared to traditional attacks that rely on conspicuous color blocks, our method achieves excellent visua deception, thereby reducing the risk of human detection. Second, we introduce an Object-level Spatial Decoupling Module (OSDM) to disrupt cross-modal spatial alignment. OSDM utilizes a scale-aware offset generator to stealthily shift the infrared target at the object level away from its visible location. This manipulation disrupts the strict objectlevel spatial alignment assumption relied upon by crossmodal detectors. Consequently, this spatial decoupling triggers severe cross-modal feature antagonism. It forces the fusion mechanism to aggregate real features with conflicting background noise, subsequently flattening the network’s attention features. Finally, we formulate a Harmonic Adversarial loss (HarAdv loss) to dynamically balance adversarial strength with stealthiness. HarAdv loss allows the attack to adapt robustly to complex environments. As illustrated in Fig. 2, CamoShift achieves SOTA comprehensive performance among existing baselines, which demonstrates that our attack is both effective and visually stealthy.

In summary, our contributions are threefold:

• We propose CamoShift to attack visible-infrared object detectors by combining semantic camouflage with objectlevel spatial decoupling. CamoShift is the pioneering attempt to target both visual stealthiness and attack success in visible-infrared object detection.

• We design SCM and OSDM in CamoShift for stealthy cross-modal attacks. Specifically, SCM generates a semantically camouflaged and thermally valid patch, while OSDM spatially shifts the infrared target to break objectlevel alignment and disrupt fusion.

• The HarAdv loss is designed to harmonize attack strength and visual stealth. Experimental results validate that this loss design is crucial to achieving a superior balance between adversarial effectiveness and visual camouflage compared with existing methods.

## 2. Related Work

## 2.1. Visible-Infrared Object Detection

Visible-infrared object detection combines visual details with infrared features to operate robustly under diverse weather conditions, which is crucial for challenging scenarios like UAV surveillance [26, 43]. While early methods relied on naive feature concatenation [10, 18, 29, 39], recent advanced models utilize smarter fusion strategies [2, 6, 24, 42, 44, 46].

Due to the physical separation of visible and infrared sensors, the images they capture never align perfectly. To mitigate this, modern detectors employ spatial alignment modules. Networks like MBNet [47] and C2Former [41], along with the offset-guided fusion in COMO [17], and weakly aligned learning methods [45], calculate dense similarities to explicitly align the features from both sensors. However, this creates a critical vulnerability by assuming the misalignment is merely a global camera shift.

![](images/e4711b6ff761bb681fb1ca9e7eeb88615336dd0f29516182aea45af3825e780f.jpg)  
Figure 3. Overview of our CamoShift framework. SCM generates a visually stealthy patch and its valid infrared counterpart via an RGB-IR adapter. Concurrently, OSDM utilizes an FPN to compute a spatial offset for the infrared target, using inpainting and recomposition to break cross-modal spatial alignment. Finally, a harmonic adversarial loss is introduced to dynamically balance the attack effectiveness and visual stealthiness during optimization.

## 2.2. Adversarial Attacks

Early adversarial attacks mainly focused on digital perturbations, including pixel-level noise and spatial transformations such as pixel warping [38]. Their effectiveness often degrades under physical-world variations. Physical attacks later introduced printable patches [1, 28] and wearable materials [37]. Camouflaged methods such as AdvCam [4] and NAP [8] further improve visual stealth, but their original single-modal setting does not affect both sensor branches.

Cross-modal physical attacks, including UNIPatch [36] and CDUPatch [21], use infrared-blocking materials or visible-to-infrared properties to attack both modalities. Most methods rely on high-contrast patterns and confine perturbations to the aligned target region. CamoShift targets object-level cross-modal correspondence through semantic camouflage and infrared spatial decoupling.

## 3. Methodology

## 3.1. Overall Structure

Given an aligned visible-infrared pair $I \ = \ ( I ^ { v } , I ^ { i r } )$ and a target object o with class label $c _ { o }$ and bounding box $b _ { o } .$

a visible-infrared detector D predicts a set of detections $Y = D ( I )$ . Current visible-infrared patch attacks mainly perturb the appearance inside the aligned target box. Although such perturbations may suppress local confidence, they still preserve the cross-modal correspondence assumed by the fusion network. We find that, alongside appearance corruption, the fusion networks of current object detectors are highly vulnerable to object-level spatial misalignment.

As illustrated in Fig. 3, our CamoShift framework jointly optimizes a semantically camouflaged visible pattern and a scale-aware spatial offset in the infrared branch. It consists of three components: 1) a Semantic Camouflage Module (SCM) that makes the visible patch blend into the host object while enforcing thermal feasibility; 2) an Object-level Spatial Decoupling Module (OSDM) that shifts the infrared evidence to break the alignment prior used by feature fusion; and 3) a Harmonic Adversarial loss (HarAdv loss) that balances attack effectiveness and visual stealthiness. For a sampled transformation, the visible and infrared patching process is written as follows:

$$
\begin{array} { r l } & { P ^ { i r } = F _ { A } ( P ^ { v } ) , } \\ & { I _ { a d v } ^ { v } = ( 1 - M _ { \tau } ) \odot I ^ { v } + M _ { \tau } \odot \tau ( P ^ { v } ) , } \\ & { I _ { p } ^ { i r } = ( 1 - M _ { \tau } ) \odot I ^ { i r } + M _ { \tau } \odot \tau ( P ^ { i r } ) , } \end{array}\tag{1}
$$

where $P ^ { v }$ is the learnable visible patch, $P ^ { i r }$ is its infrared counterpart, $I _ { a d v } ^ { v }$ and $I _ { p } ^ { i r }$ are the patched visible and infrared images, τ is the sampled transformation, $F _ { A }$ is the RGB-IR adapter, $M _ { \tau }$ is the transformed patch mask. $F _ { A }$ learns an empirical response mapping from paired RGB-IR data rather than reproducing the thermal imaging process. SCM optimizes the aligned patched pair, while OSDM moves infrared object evidence away from the visible target.

## 3.2. Semantic Camouflage Module

SCM is designed to generate visually stealthy adversarial patches. In the visible domain, the patch camouflages itself as the natural surface patterns of the host object while maintaining its attack performance. Meanwhile, in the infrared domain, the same patch must still produce a realistic and effective infrared response through the RGB-IR adapter. Therefore, SCM is not only an appearance regularizer. It is the stage that establishes a valid cross-modal attack state on which the OSDM can operate.

A visually stealthy patch should look like a plausible part of the object surface instead of an external marker [31, 33]. Let $C _ { o } ^ { v } = \mathrm { C r o p } ( I ^ { v } , b _ { o } )$ denote the visible crop of the host object and let $I ^ { s t y }$ be a natural reference texture, such as rust, dried mud, or surface stain. Using a pretrained VGG feature extractor, we define the style, content, and color consistency terms as follows:

$$
\mathcal { L } _ { s t y } = \sum _ { l \in S _ { s } } \frac { 1 } { C _ { l } ^ { 2 } } \left. G ( \phi _ { l } ( P ^ { v } ) ) - G \big ( \phi _ { l } ( I ^ { s t y } ) \big ) \right. _ { 1 } ,\tag{2}
$$

$$
\mathcal { L } _ { c n t } = \sum _ { l \in S _ { c } } \frac { 1 } { C _ { l } H _ { l } W _ { l } } \left. \phi _ { l } ( P ^ { v } ) - \phi _ { l } ( C _ { o } ^ { v } ) \right. _ { 1 } ,\tag{3}
$$

$$
\mathcal { L } _ { c o l } = \| \mu ( P ^ { v } ) - \mu ( C _ { o } ^ { v } ) \| _ { 1 } + \| \sigma ( P ^ { v } ) - \sigma ( C _ { o } ^ { v } ) \| _ { 1 } ,\tag{4}
$$

where $\phi _ { l } ( \cdot )$ extracts the feature map at the l-th layer of VGG, with shape $C _ { l } \times H _ { l } \times W _ { l } . \ S _ { s }$ and $S _ { c }$ denote the sets of layers used for style and content extraction, respectively. $G ( \cdot )$ computes the Gram matrix, while $\mu ( \cdot )$ and $\sigma ( \cdot )$ compute the spatial mean and standard deviation. The style term aligns the patch with the texture family of the reference pattern, the content term keeps the patch anchored to the local structure of the host object, and the color term prevents the patch from degenerating into conspicuous artificial patterns.

Beyond achieving visual stealth, SCM must also effectively suppress the confidence of the detector. To this end, we define an aligned attack loss based on the predictions generated from the patched pair in Eq. (1):

$$
\mathcal { L } _ { a t t } ^ { S C M } = \frac { 1 } { \left| Q _ { o } \right| } \sum _ { k \in Q _ { o } } s _ { k } p _ { k } ( c _ { o } ) ,\tag{5}
$$

where $Q _ { o }$ collects the residual predictions associated with the target region, $s _ { k }$ is the objectness score, and $p _ { k } ( c _ { o } )$ is the class probability for the target class. Minimizing Eq. (5) makes SCM a true attack module rather than a pure camouflage pre-processor.

The corresponding SCM cost term gathers all

appearance-related constraints into a single branch:

$$
\begin{array} { r l } & { \mathcal { L } _ { c o s t } ^ { S C M } = \lambda _ { s t y } \mathcal { L } _ { s t y } + \lambda _ { c n t } \mathcal { L } _ { c n t } + \lambda _ { c o l } \mathcal { L } _ { c o l } } \\ & { \quad \quad \quad + \lambda _ { t v } \mathcal { L } _ { t v } + \lambda _ { s s i m } \big ( 1 - \mathrm { S S I M } ( P ^ { v } , C _ { o } ^ { v } ) \big ) } \\ & { \quad \quad \quad \quad + \lambda _ { l p i p s } \mathrm { L P I P S } ( P ^ { v } , C _ { o } ^ { v } ) , } \end{array}\tag{6}
$$

where $\mathcal { L } _ { t v }$ denotes the total variation loss used to enhance local smoothness, SSIM and LPIPS measure structural similarity and perceptual distance, and the λ<sub>·</sub> terms are scalar loss weights. The attack term keeps the patch adversarial, while the cost term keeps it semantically camouflaged and visually printable. All SCM terms are optimized under Expectation Over Transformation (EOT), so the patch is not tied to a fixed placement or orientation.

## 3.3. Object-level Spatial Decoupling Module

OSDM disrupts object-level alignment by shifting target evidence only in the infrared branch. The modeled shift represents RGB-IR mismatch in moving scenes caused by differences in exposure time, frame rate, or sensor readout. Coupled with SCM, the resulting mismatch breaks the alignment prior used by cross-modal fusion.

To ensure the offset is both effective and stealthy, we utilize a feature pyramid network (FPN) to adaptively adjust the offset based on the object scale. For each target object, the pyramid level is assigned according to the object size:

$$
l _ { s e l } = \arg \operatorname* { m i n } _ { l } \left| \log _ { 2 } \frac { \sqrt { w _ { o } h _ { o } } } { s _ { l } } - \kappa \right| ,\tag{7}
$$

where $w _ { o }$ and $h _ { o }$ are the width and height of the target box, l indexes the pyramid levels, $s _ { l } \in \{ 8 , 1 6 , 3 2 \}$ is the stride of level l, and κ is the canonical scale constant. We then extract the corresponding RoI feature $z _ { o }$ and predict a bounded offset via a generator network $G _ { \theta } \mathrm { : }$

$$
\Delta d _ { o } = ( \Delta x _ { o } , \Delta y _ { o } ) = \alpha s _ { l _ { s e l } } \operatorname { t a n h } \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( G _ { \theta } ( z _ { o } ) \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) ,\tag{8}
$$

where $\Delta x _ { o }$ and $\Delta y _ { c }$ are the horizontal and vertical offsets, θ denotes the parameters of $G _ { \theta }$ , and α is set to 1.5 in our experiments. The hyperbolic tangent bounds the displacement, while the factor $\alpha s _ { l _ { s e l } }$ ties the offset magnitude to the receptive-field stride of the assigned pyramid level. As a result, the infrared target is encouraged to cross the corresponding grid-cell boundary on the feature map, even when the image-plane shift remains visually small.

However, directly translating the infrared target creates empty holes and sharp boundaries in the background. To synthesize a plausible misaligned pair, we first remove the original target from the infrared image by inpainting, and then paste the shifted target back with a soft mask:

$$
\begin{array} { r l } & { \hat { I } ^ { i r } = H ( I _ { p } ^ { i r } , M _ { o } ) , } \\ & { I _ { a d v } ^ { i r } = \left( 1 - S _ { \Delta d _ { o } } ( M _ { o } ^ { s } ) \right) \odot \hat { I } ^ { i r } + S _ { \Delta d _ { o } } \left( M _ { o } ^ { s } \odot I _ { p } ^ { i r } \right) , } \end{array}\tag{9}
$$

where $\hat { I } ^ { i r }$ is the inpainted infrared image, $I _ { a d \iota } ^ { i r }$ is the shifted infrared image, $M _ { o }$ is the target mask, $M _ { o } ^ { s }$ is its Gaussiansmoothed version, $H ( \cdot )$ denotes LaMa-based inpainting, and $S _ { \Delta d _ { o } } ( \cdot )$ is the spatial translation operator.

After Eq. (9), the visible branch still observes the target at the original position, whereas the infrared branch places the main infrared evidence at the shifted position. This spatial mismatch breaks the strict alignment prior relied upon by cross-modal fusion, inducing the cross-modal feature antagonism discussed in the introduction. To describe this mechanism more clearly, we separate the OSDM attack term from the OSDM cost term.

The OSDM attack branch has three parts. The first reduces the residual confidence around the original and shifted semantic positions. The second weakens visibleinfrared agreement at the original target region. The third flattens the fused response over the two competing regions:

$$
\begin{array} { r l } & { \mathcal { L } _ { d i s } \ = \displaystyle \frac { 1 } { | R _ { o } | } \sum _ { u \in R _ { o } } \frac { \langle \Phi _ { u } ^ { v } , \Phi _ { u } ^ { i r } \rangle } { \| \Phi _ { u } ^ { v } \| _ { 2 } \| \Phi _ { u } ^ { i r } \| _ { 2 } + \varepsilon } \mathrm { , } } \\ & { \mathcal { L } _ { f l a t } = \mathrm { S t d } \Big ( \Phi ^ { f u s } \Big | _ { R _ { o } \cup R _ { o } ^ { \Delta } } \Big ) \mathrm { , } } \end{array}\tag{10}
$$

where $R _ { o }$ and $R _ { o } ^ { \Delta }$ denote the projected regions of the original box and the shifted box on the selected feature map. $\Phi _ { u } ^ { v }$ and $\Phi _ { u } ^ { i r }$ represent the visible and infrared feature vectors at spatial location u. $\Phi ^ { f u s }$ is the fused feature representation, $\Phi ^ { f u s } | _ { R }$ denotes the fused features restricted to region $R ,$ $\operatorname { S t d } ( \cdot )$ denotes standard deviation, and ε is a small constant for numerical stability. The similarity term measures how strongly the two modalities match at the original location, and the flattening term captures whether the fused representation still forms a concentrated target response.

We further define the OSDM detection-suppression term and the full OSDM attack loss as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { d e t } ^ { O S D M } = \displaystyle \frac { 1 } { | Q _ { o } ^ { \Delta } | } \sum _ { k \in Q _ { o } ^ { \Delta } } s _ { k } p _ { k } ( c _ { o } ) , } \\ & { \mathcal { L } _ { a t t } ^ { O S D M } = \lambda _ { d e t } \mathcal { L } _ { d e t } ^ { O S D M } + \lambda _ { d i s } \mathcal { L } _ { d i s } + \lambda _ { f l a t } \mathcal { L } _ { f l a t } , } \end{array}\tag{11}
$$

where $Q _ { o } ^ { \Delta }$ collects the predictions overlapping either the original visible position or the shifted infrared position, and $\lambda _ { d e t } , \lambda _ { d i s } ,$ , and $\lambda _ { f l a t }$ are scalar loss weights. Eq. (11) makes OSDM explicit: it is not merely moving pixels, but jointly suppressing prediction confidence, cross-modal agreement, and fused localization concentration after the misalignment is introduced. The OSDM cost term is defined as:

$$
\mathcal { L } _ { c o s t } ^ { O S D M } = \lambda _ { r e g } \frac { \| \Delta d _ { o } \| _ { 1 } } { s _ { l _ { s e l } } } ,\tag{12}
$$

where $\lambda _ { r e g }$ is a scalar loss weight. This term penalizes large offsets. Therefore, OSDM seeks the smallest object-level shift needed to break the fusion prior.

## 3.4. Harmonic Adversarial Loss

An attack should remain effective without sacrificing visual stealthiness. Based on the above decomposition, the final attack loss and the final cost loss are no longer written in a mixed and unordered manner. Instead, they are assembled module by module:

$$
\begin{array} { r } { \mathcal { L } _ { a t t } = \omega _ { s } \mathcal { L } _ { a t t } ^ { S C M } + \omega _ { o } \mathcal { L } _ { a t t } ^ { O S D M } , } \\ { \mathcal { L } _ { c o s t } = \rho _ { s } \mathcal { L } _ { c o s t } ^ { S C M } + \rho _ { o } \mathcal { L } _ { c o s t } ^ { O S D M } , } \end{array}\tag{13}
$$

where $\omega _ { s }$ and $\omega _ { o }$ weight the SCM and OSDM attack losses, $\rho _ { s }$ and $\rho _ { o }$ weight their corresponding cost losses. SCM contributes the aligned adversarial pressure and semantic camouflage cost, while OSDM contributes the misalignmentdriven adversarial pressure and the offset regularization cost. This makes the source of each optimization signal transparent.

A fixed linear combination of the final attack loss and the cost loss is often unstable because the desired optimization balance shifts during training [13]. We therefore use crossadaptive harmonic weights:

$$
\beta _ { a t t } = \exp { \left( - \frac { \mathrm { s g } ( \mathcal { L } _ { c o s t } ) } { T _ { c o s t } } \right) } , \beta _ { c o s t } = \exp { \left( - \frac { \mathrm { s g } ( \mathcal { L } _ { a t t } ) } { T _ { a t t } } \right) } ,\tag{14}
$$

where $\operatorname { s g } ( \cdot )$ is the stop-gradient operator and $T _ { a t t } , T _ { c o s t }$ are temperature parameters. The final HarAdv loss is then defined as follows:

$$
\mathcal { L } _ { H a r A d v } = ( 1 + \beta _ { a t t } ) \mathcal { L } _ { a t t } + \lambda _ { h } ( 1 + \beta _ { c o s t } ) \mathcal { L } _ { c o s t } ,\tag{15}
$$

where $\lambda _ { h }$ controls the contribution of the cost loss. When the attack becomes strong but visually less plausible, the harmonic weight of the attack branch decreases and the optimization focuses on camouflage. Conversely, when the patch is plausible but the detector remains stable, the cost branch is relatively down-weighted and the optimization shifts toward attack effectiveness.

To demonstrate the necessity of the stop-gradient operator, we analyze the gradient of the total objective with respect to the learnable parameters. Let the learnable parameters W contain the visible patch and the OSDM offset generator. Since the dynamic weights are detached from the opposite branch, the gradient simplifies to:

$$
\begin{array} { r l } & { \cfrac { \partial \mathcal { L } _ { H a r A d v } } { \partial W } = ( 1 + \beta _ { a t t } ) \cfrac { \partial \mathcal { L } _ { a t t } } { \partial W } + \lambda _ { h } ( 1 + \beta _ { c o s t } ) \cfrac { \partial \mathcal { L } _ { c o s t } } { \partial W } , } \\ & { \cfrac { \partial } { \partial W } \ c \big ( ( 1 + \beta _ { a t t } ) \mathcal { L } _ { a t t } \big ) = ( 1 + \beta _ { a t t } ) \cfrac { \partial \mathcal { L } _ { a t t } } { \partial W } + \mathcal { L } _ { a t t } \cfrac { \partial \beta _ { a t t } } { \partial W } . } \end{array}\tag{16}
$$

The first line is the gradient actually used by our optimization. The second line shows the pathological leakage that would appear without stop-gradient. Because $\beta _ { a t t }$ is inversely controlled by the cost branch, the extra term would encourage the optimizer to increase the cost merely to shrink the attack weight, thereby creating a harmful shortcut that ruins camouflage. The stop-gradient prevents this leakage and ensures that the two branches only interact through scalar balancing rather than gradient contamination.

<table><tr><td rowspan="2">Method</td><td colspan="3">DroneVehicle[26]</td><td colspan="3">M3FD [19]</td><td colspan="3">LLVIP [11]</td><td colspan="3">VEDAI [22]</td></tr><tr><td>YOLOv8</td><td>C2Former</td><td>COMO</td><td>YOLOv8</td><td>C2Former</td><td>COMO</td><td>YOLOv8</td><td>C2Former</td><td>COMO</td><td>YOLOv8</td><td>C2Former</td><td>COMO</td></tr><tr><td>Clean</td><td>84.09</td><td>91.03</td><td>88.14</td><td>77.02</td><td>70.37</td><td>85.78</td><td>92.47</td><td>89.23</td><td>92.80</td><td>88.86</td><td>84.63</td><td>86.27</td></tr><tr><td>RandomPatch†</td><td>[1] 4.76/80.07</td><td>5.39/87.66</td><td>3.28/85.30</td><td>2.02/76.53</td><td>2.21/68.54</td><td>2.88/84.71</td><td>4.02/90.76</td><td>3.76/87.20</td><td>7.22/90.51</td><td></td><td>17.56/79.5613.36/78.189.38/74.27</td><td></td></tr><tr><td>DPatch† [20]</td><td>5.29/80.62</td><td>5.85/86.63</td><td>5.34/84.52</td><td>8.04/75.59</td><td>8.12/68.62</td><td>7.77/84.55</td><td>9.33/90.35</td><td>9.51/86.51</td><td></td><td>9.42/90.77 25.78/67.2724.05/66.23 28.39/59.73</td><td></td><td></td></tr><tr><td>stAdv† [38]</td><td>46.92/65.98 27.85/83.9129.82/75.99 44.09/56.70 46.14/47.9734.55/70.43 25.85/80.9718.03/75.8710.65/89.79 54.44/59.32 55.95/53.55 55.40/59.63</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AdvCam† [4]</td><td>43.39/70.23 60.15/70.05 53.74/67.22 61.02/46.3938.92/53.11 41.23/64.23 29.67/79.7515.68/77.23 32.70/80.3540.67/75.88 30.92/71.91 56.40/73.99</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NAP† [8]</td><td>29.24/76.1939.70/79.56 37.23/74.54 40.67/59.24 25.61/64.12 31.62/70.96 30.02/81.6014.82/79.42 34.26/86.02 55.11/69.04 27.48/73.22 49.29/69.21</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YOLOPatch† [28]36.34/73.6046.72/78.1751.90/68.04 49.40/50.34 35.15/52.9938.18/65.8920.14/85.9314.39/77.8820.35/86.8046.67/78.6824.43/75.9735.07/78.97</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FCA† [30]</td><td>16.88/80.9348.85/77.52 46.95/70.4740.22/58.35 22.93/63.64 26.37/75.0231.54/78.3211.97/82.7737.67/76.6734.67/77.2717.94/75.9842.65/73.34</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HOTCOLD†[32]</td><td>47.21/61.5459.37/58.78 62.70/54.0128.95/71.28 36.19/55.4929.79/76.0024.07/84.13 20.95/73.2235.38/78.54 37.78/83.3717.56/79.54 38.39/78.23</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNIPatch [36]</td><td>51,20/62,98 41.08/75.13 39,06/71,46 40.42/59,99 32.56/57.5851.16/59,81 23,63/80.45 16.26/79,65 22,84/82.03 54,22/69.37 39,69/66.50 53.51/69.64</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TOUAP [7]</td><td>34.07/72.64 48.99/72.9748.02/68.1938.13/59.8944.91/34.3744.28/58.4720.40/82.5720.08/67.72 20.09/84.62 45.33/74.95 20.99/73.02 51.66/65.99</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CDUPatch [21]</td><td>51.64/60.52 56.60/68.90 54.54/42.5346.77/58.3747.00/39.3944.40/60.5918.07/86.9731.34/59.3324.29/82.24 49.33/66.8126.34/72.0949.76/64.22</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

CamoShift (ours) 78.44/20.26 86.96/19.40 64.23/21.81 76.71/27.52 67.08/15.95 65.84/28.71 64.28/25.27 73.64/22.43 57.32/25.62 67.72/27.52 71.72/39.91 62.50/33.45  
Table 1. Overall comparison on four visible-infrared datasets. The Clean row reports m $\mathsf { A P } _ { 5 0 }$ (%). All attack rows report Strict-ASR (%) / post-attack mA $\mathrm { { \cdot P _ { 5 0 } } }$ (%). Methods marked with † are single-modal attacks adapted to the visible-infrared setting.

The final parameters are obtained by minimizing the expectation of the HarAdv loss under the transformation distribution, and the optimized infrared patch is generated by the RGB-IR adapter from the optimized visible patch.

## 4. Experiments

## 4.1. Experimental Setup

We conduct experiments on four widely used visibleinfrared datasets: DroneVehicle [26], M3FD [19], LLVIP [11], and VEDAI [22]. We select YOLOv8 [12], C2Former [41], and COMO [17] as our victim models. While YOLOv8 serves as a strong baseline, C2Former and COMO explicitly model cross-modal interaction, providing a strict protocol to evaluate whether our attack effectively disrupts the fusion correspondence prior.

We compare generic attacks, including RandomPatch [1], DPatch [20], YOLOPatch [28], and stAdv [38]. Other baselines include the camouflage attacks AdvCam [4], NAP [8], and FCA [30], the infrared attack HOTCOLD [32], and the visible-infrared attacks UNIPatch [36], TOUAP [7], and CDUPatch [21]. Single-modal baselines are applied to visible images, while the same RGB-IR adapter used by CamoShift generates their infrared perturbations. Thus, all methods are evaluated under a unified cross-modal protocol.

We introduce Strict Attack Success Rate (Strict-ASR) to require suppression at both semantic locations. Let N be the number of samples, Y (b) the predictions matched to box b, $b _ { o } ^ { ( i ) }$ the original box, $b _ { o } ^ { \Delta ( i ) }$ the shifted box, and $\mathbb { I } ( \cdot )$ the indicator function:

$$
\mathrm { S t r i c t - A S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \Big ( | Y _ { i } ( b _ { o } ^ { ( i ) } ) | = 0 \land | Y _ { i } ( b _ { o } ^ { \Delta ( i ) } ) | = 0 \Big ) .\tag{17}
$$

For baselines without spatial shifting, we set $b _ { o } ^ { \Delta ( i ) } ~ =$ $b _ { o } ^ { ( i ) }$ . Therefore, Eq. (17) is mathematically equivalent to conventional ASR for these baselines. On DroneVehicle with YOLOv8, CamoShift achieves a conventional ASR of 93.7%, while its Strict-ASR is 78.4%. We also report postattack m $\mathrm { { A P } _ { 5 0 } }$ , SSIM, and LPIPS.

For Table 1, patches are optimized on the corresponding COMO model for each dataset and evaluated on YOLOv8, C2Former, and COMO. All detectors are trained on their respective datasets, with clean m $\mathrm { { A P } _ { 5 0 } }$ reported in Table 1. CamoShift is implemented in PyTorch and optimized with Adam for 50 epochs. Implementation details, including hyperparameters, runtime, memory usage, and hardware, are provided in the Supplementary Material.

## 4.2. Main Results

Table 1 summarizes the main results on all datasets and detectors. A consistent pattern can be observed across all settings. Methods that mainly perturb appearance, including generic baselines and RGB-specific camouflage attacks, have a limited impact in the visible-infrared setting. Their perturbation may reduce the confidence of one branch, but the complementary modality still preserves enough evidence for the detector to localize the target. This effect is particularly clear on COMO, where the retained m $\mathrm { { A P } _ { 5 0 } }$ of AdvCam and NAP remains high across all datasets.

![](images/90a9a0d616cef37501ee59275b4706bc7d854cce8759bfb3ef12831a9438b237.jpg)  
Figure 4. Qualitative results of CamoShift against the COMO detector. To better visualize the attack effects, visible-light images are utilized as the base for daylight scenarios, whereas shifted infrared images are employed for dark environments.

Recent cross-modal attacks achieve much stronger results, but their gains remain limited when the perturbation is still constrained to the aligned object region. For example, CDUPatch clearly outperforms single-modality baselines, yet its Strict-ASR on COMO still only ranges from 24.29% to 54.54% across the four datasets. At the same time, the detector still retains substantial performance after the attack. This suggests that appearance corruption alone is not sufficient once the detector can continue to associate visible and infrared evidence at the same object position.

In contrast, CamoShift achieves the best results in all settings. On COMO, our method raises Strict-ASR to 64.23%, 65.84%, 57.32%, and 62.50% on the four datasets, while reducing the retained mAP<sub>50</sub> to 21.81%, 28.71%, 25.62%, and 33.45%, respectively. Similar trends are also observed on YOLOv8 and C2Former. The qualitative attack effects on the COMO detector are further visualized in Fig. 4. These results support our main claim: once the infrared features are spatially shifted from the visible target, the fusion module is forced to aggregate mismatched cues, leading to a substantially more severe performance degradation than that caused by spatially aligned perturbations.

## 4.3. Component-wise Ablation Study

Table 2 presents the ablation study on DroneVehicle using COMO. The RGB-IR adapter alone already produces a valid visible-infrared patch, but its visual quality is poor, with only 0.892 SSIM. Adding semantic camouflage greatly improves the visual appearance to 0.937 SSIM, although the attack success rate drops from 53.1% to 44.2%. Stronger camouflage constrains the optimization space, making effective attacks harder to generate.

The key gain appears when a spatial shift is introduced. Once OSDM is enabled, the Strict-ASR rises sharply to 60.2% while the patch still preserves reasonable visual quality. This confirms that the main contribution of CamoShift is not a stronger texture perturbation, but the deliberate breaking of object-level cross-modal correspondence. After adding the harmonic optimization objective, both attack strength and stealth improve further, reaching 64.2% Strict-ASR and 0.975 SSIM. This result indicates that HarAdv loss is effective in balancing the conflict between attack loss and camouflage constraints during optimization.

<table><tr><td>IR-Ada</td><td>Camo</td><td>Spat</td><td>HarAdv</td><td>Strict-ASR ↑</td><td>SSIM ↑</td></tr><tr><td></td><td></td><td></td><td></td><td>32.3</td><td>0.972</td></tr><tr><td>√</td><td></td><td></td><td></td><td>53.1</td><td>0.892</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>44.2</td><td>0.937</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>60.2</td><td>0.923</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>64.2</td><td>0.975</td></tr></table>

Table 2. Ablation study. Strict-ASR is reported in percent. IR-Ada denotes the RGB-IR adapter; Camo, semantic camouflage; Spat, OSDM; and HarAdv, the harmonic adversarial loss.
<table><tr><td>Method</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>HOTCOLD [32]</td><td>0.8855</td><td>0.1484</td></tr><tr><td>DPatch [20]</td><td>0.9303</td><td>0.0724</td></tr><tr><td>UNIPatch [36]</td><td>0.9390</td><td>0.0921</td></tr><tr><td>CDUPatch [21]</td><td>0.9464</td><td>0.0920</td></tr><tr><td>AdvCam [4]</td><td>0.9476</td><td>0.0938</td></tr><tr><td>YOLOPatch [28]</td><td>0.9483</td><td>0.0808</td></tr><tr><td>CamoShift (ours)</td><td>0.9745</td><td>0.0648</td></tr></table>

Table 3. Stealthiness comparison on DroneVehicle using COMO. SSIM and LPIPS compare the patched and original regions.

## 4.4. Stealthiness Evaluation

Attack strength alone is not sufficient for a patch, since an obviously artificial pattern is easy to detect and remove in practice. We therefore compare the visual quality of the patched region against the original host appearance. The results in Table 3 show a clear advantage of the proposed semantic camouflage module. Compared with CDUPatch, our patch improves SSIM from 0.9464 to 0.9745 and reduces LPIPS from 0.0920 to 0.0648 on DroneVehicle. More importantly, this gain is achieved in the visible-infrared setting, where the patch must remain visually plausible while still inducing an effective infrared response.

The result is also competitive with single-modal camouflage attacks. AdvCam produces a natural visible pattern, but it does not need to satisfy the cross-modal constraint. In contrast, CamoShift preserves or even improves visible stealth while remaining effective against both modalities. This behavior is consistent with the design of SCM: the style, content, and color regularizers keep the patch close to the host texture distribution, while the RGB-IR adapter ensures that the optimized appearance remains meaningful in the infrared branch.

We also test whether the attack depends on a specific visible appearance. As visualized in Fig. 5, Table 4 shows that different natural textures, including rust, dried mud, and urban camouflage, all lead to high attack success while preserving good visual similarity. The gap between these styles is small compared with the gap between our method and CDUPatch. This observation is important for deployment, since the patch can be adapted to different environments without changing the underlying attack mechanism.

Rust texture  
Urban camo  
Floral pattern  
Dried mud  
![](images/dd83f888eb66728320b474c654af08dc58589976e4e89b0568549ee0c8c1a683.jpg)  
Figure 5. Qualitative results of CDUPatch and our CamoShift across diverse environments.

<table><tr><td>Style</td><td>Strict-ASR ↑</td><td> $\mathrm { \ m A P _ { 5 0 } ~ } .$ </td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Rust texture</td><td>66.21</td><td>22.57</td><td>0.9738</td><td>0.0602</td></tr><tr><td>Dried mud</td><td>65.42</td><td>22.62</td><td>0.9735</td><td>0.0592</td></tr><tr><td>Urban camo</td><td>65.28</td><td>22.51</td><td>0.9732</td><td>0.0603</td></tr><tr><td>Floral pattern</td><td>66.19</td><td>22.56</td><td>0.9734</td><td>0.0610</td></tr></table>

Table 4. Effect of different camouflage styles on COMO. Strict-ASR and $\mathrm { \ m A P 5 0 }$ are reported in percent.

## 4.5. Generalization and Robustness

We further evaluate whether the proposed attack generalizes beyond the white-box setting. Table 5 reports crossmodel transfer results on DroneVehicle. A patch optimized on YOLOv8 still reaches 90.61% Strict-ASR on C2Former and 59.26% on COMO. Similarly, the patch trained on COMO transfers to YOLOv8 and C2Former with 78.44% and 86.96% Strict-ASR, respectively. This result indicates that the gain of CamoShift does not mainly come from overfitting to a specific detector head. Instead, it exposes a common vulnerability in visible-infrared object detectors by exploiting their heavy reliance on consistent cross-modal correspondence.

We further evaluate CamoShift against three common defenses on DroneVehicle with YOLOv8. Strict-ASR remains 52.1%, 73.3%, and 74.5% under adversarial training, feature denoising, and input purification, respectively, compared with 78.4% without defense.

Finally, we evaluate the sensitivity of our attack to the detection confidence threshold. When the threshold varies from 0.3 to 0.7, the attack against COMO remains stable across all four datasets. Even at a strict threshold of 0.3, the Strict-ASR stays above 56.0% on DroneVehicle. Crossmodel evaluation on DroneVehicle further shows robust effectiveness across all threshold levels. The corresponding curves are provided in the Supplementary Material.

<table><tr><td>Source model</td><td>YOLOv8 [12]</td><td>C2Former [41]</td><td>COMO [17]</td></tr><tr><td>YOLOv8 [12]</td><td>88.86</td><td>90.61</td><td>59.26</td></tr><tr><td>C2Former [41]</td><td>82.78</td><td>91.26</td><td>63.95</td></tr><tr><td>COMO [17]</td><td>78.44</td><td>86.96</td><td>64.23</td></tr></table>

Table 5. Evaluation of cross-model adversarial transferability. Rows denote source models and columns denote target models. Results are evaluated by Strict-ASR (%).

## 5. Conclusion

In this work, we presented CamoShift, an adversarial framework for visible-infrared object detection. Unlike prior cross-modal attacks that mainly rely on conspicuous patterns and perturbations confined to spatially aligned object regions, our method targets a more fundamental weakness in visible-infrared fusion. This weakness lies in the strong dependence on object-level cross-modal correspondence. Specifically, SCM generates a visually plausible patch that remains consistent with the texture and color distribution of the host object, while the OSDM shifts the infrared evidence in a scale-aware manner to break the alignment prior used by fusion networks. Together with the harmonic adversarial loss, these components jointly improve attack effectiveness and stealth, and induce severe cross-modal feature conflict during fusion.

Extensive evaluations across four benchmark datasets and three representative detector families demonstrate that CamoShift consistently outperforms existing baselines in both attack strength and visual stealthiness. The performance gains are particularly significant on fusion models that rely heavily on strict spatial alignment. This validates our core insight: disrupting spatial alignment is fundamentally more destructive than simply applying stronger, spatially aligned appearance perturbations. Transferability, camouflage-style, and component-wise ablation results further support the robustness and generality of CamoShift. Overall, our study demonstrates that spatial decoupling is a critical yet underexplored attack surface in visible-infrared detection. We hope this work inspires research into fusion mechanisms robust to spatial decoupling, defense strategies [14], and evaluation protocols for multi-sensor vision systems.

## References

[1] Tom B Brown, Dandelion Mane, Aurko Roy, Mart´ ´ın Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017. 1, 3, 6

[2] Yanpeng Cao, Xing Luo, Jiangxin Yang, Yanlong Cao, and Michael Ying Yang. Locality guided cross-modal feature aggregation and pixel-level fusion for multispectral pedestrian detection. Information Fusion, 88:1–11, 2022. 2

[3] Kinjal Dasgupta, Arindam Das, Sudip Das, Ujjwal Bhattacharya, and Senthil Yogamani. Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving. IEEE transactions on intelligent transportation systems, 23(9):15940–15950, 2022. 1

[4] Ranjie Duan, Xingjun Ma, Yisen Wang, James Bailey, A Kai Qin, and Yun Yang. Adversarial camouflage: Hiding physical-world attacks with natural styles. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 997–1005. IEEE, 2020. 3, 6, 7

[5] Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1625–1634, 2018. 1

[6] Xiao He, Chang Tang, Xin Zou, and Wei Zhang. Multispectral object detection via cross-modal conflict-aware learning. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1465–1474, 2023. 2

[7] Chengyin Hu, Weiwen Shi, Wen Yao, Tingsong Jiang, Ling Tian, and Wen Li. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world. Applied Soft Computing, 171:112818, 2025. 2, 6

[8] Yu-Chih-Tuan Hu, Bo-Han Kung, Daniel Stanley Tan, Jun-Cheng Chen, Kai-Lung Hua, and Wen-Huang Cheng. Naturalistic physical adversarial patch for object detectors. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7848–7857, 2021. 3, 6

[9] Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Fuchun Sun, Bo Zhang, and Xiaolin Hu. Adversarial texture for fooling person detectors in the physical world. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13297–13306. IEEE, 2022. 1

[10] Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon. Multispectral pedestrian detection: Benchmark dataset and baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1037–1045, 2015. 1, 2

[11] Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. Llvip: A visible-infrared paired dataset for lowlight vision. In 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 3489–3497. IEEE, 2021. 6

[12] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics yolo, 2023. 6, 8

[13] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018. 5

[14] Taeheon Kim, Youngjoon Yu, and Yong Man Ro. Defending physical adversarial attack on object detection via adversarial patch-feature energy. In Proceedings of the 30th ACM International Conference on Multimedia, pages 1905–1913, 2022. 1, 8

[15] Taeheon Kim, Youngjoon Yu, and Yong Man Ro. Multispec tral invisible coating: Laminated visible-thermal physical attack against multispectral object detectors using transparent low-e films. In Proceedings ofthe AAAI Conference on Arti ficial Intelligence, pages 1151–1159, 2023. 2

[16] Chengyang Li, Dan Song, Ruofeng Tong, and Min Tang. Illumination-aware faster r-cnn for robust multispectral pedestrian detection. Pattern Recognition, 85:161–171, 2019. 1

[17] Chang Liu, Xin Ma, Xiaochen Yang, Yuxiang Zhang, and Yanni Dong. Como: Cross-mamba interaction and offsetguided fusion for multimodal object detection. Information Fusion, 125:103414, 2026. 3, 6, 8

[18] Jingjing Liu, Shaoting Zhang, Shu Wang, and Dimitris N Metaxas. Multispectral deep neural networks for pedestrian detection. arXiv preprint arXiv:1611.02644, 2016. 2

[19] Jinyuan Liu, Xin Fan, Zhanbo Huang, Guanyao Wu, Risheng Liu, Wei Zhong, and Zhongxuan Luo. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In 2022 IEEE/CVF conference on computer vision and pat tern recognition (CVPR), pages 5792–5801. IEEE, 2022. 6

[20] Xin Liu, Huanrui Yang, Ziwei Liu, Linghao Song, Hai Li, and Yiran Chen. Dpatch: An adversarial patch attack on object detectors. arXiv preprint arXiv:1806.02299, 2018. 6, 7

[21] Jiahuan Long, Wen Yao, Tingsong Jiang, Jiacheng Hou, Shuai Jia, Junqi Wu, Xiaoya Zhang, Xiaohu Zheng, and Chao Ma. Cdupatch: Color-driven universal adversarial patch attack for dual-modal visible-infrared detectors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 1462–1470, 2025. 2, 3, 6, 7

[22] Sebastien Razakarivony and Frederic Jurie. Vehicle detec tion in aerial imagery: A small target detection benchmark. Journal of Visual Communication and Image Representation, 34:187–203, 2016. 6

[23] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016. 1

[24] Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, and Wankou Yang. Icafusion: Iterative cross-attention guided feature fusion for multispectral object detection. Pattern Recognition, 145:109913, 2024. 2

[25] Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. Detfusion: A detection-driven infrared and visible image fusion network. In Proceedings of the 30th ACM international con ference on multimedia, pages 4003–4011, 2022. 1

[26] Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning. IEEE Transactions on Circuits and Systems for Video Technology, 32(10):6700–6713, 2022. 2, 6

[27] Karasawa Takumi, Kohei Watanabe, Qishen Ha, Antonio Tejero-De-Pablos, Yoshitaka Ushiku, and Tatsuya Harada. Multispectral object detection for autonomous vehicles. In Proceedings of the on Thematic Workshops of ACM Multimedia 2017, pages 35–43, 2017. 1

[28] Simen Thys, Wiebe Van Ranst, and Toon Goedeme. Fool-´ ing automated surveillance cameras: adversarial patches to attack person detection. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 49–55. IEEE, 2019. 1, 3, 6, 7

[29] Jorg Wagner, Volker Fischer, Michael Herman, Sven¨ Behnke, et al. Multispectral pedestrian detection using deep fusion convolutional neural networks. In Esann, pages 509– 514, 2016. 2

[30] Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Zhiqiang Gong, Xiaoya Zhang, Wen Yao, and Xiaoqian Chen. Fca: Learning a 3d full-coverage vehicle camouflage for multi-view physical adversarial attack. In Proceedings of the AAAI conference on artificial intelligence, pages 2414– 2422, 2022. 6

[31] Jiakai Wang, Aishan Liu, Zixin Yin, Shunchang Liu, Shiyu Tang, and Xianglong Liu. Dual attention suppression attack: Generate adversarial camouflage in physical world. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8565–8574, 2021. 4

[32] Hui Wei, Zhixiang Wang, Xuemei Jia, Yinqiang Zheng, Hao Tang, Shin’ichi Satoh, and Zheng Wang. Hotcold block: Fooling thermal infrared detectors with a novel wearable design. In Proceedings of the AAAI conference on artificial intelligence, pages 15233–15241, 2023. 2, 6, 7

[33] Hui Wei, Hanxun Yu, Kewei Zhang, Zhixiang Wang, Jianke Zhu, and Zheng Wang. Moire backdoor attack (mba): A´ novel trigger for pedestrian detectors in the physical world. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8828–8838, 2023. 4

[34] Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin’ichi Satoh, Luc Van Gool, and Zheng Wang. Physical adversarial attack meets computer vision: A decade survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9797–9817, 2024. 2

[35] Xingxing Wei, Ying Guo, and Jie Yu. Adversarial sticker: A stealthy attack method in the physical world. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3): 2711–2725, 2022. 1

[36] Xingxing Wei, Yao Huang, Yitong Sun, and Jie Yu. Unified adversarial patch for visible-infrared cross-modal attacks in the physical world. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2348–2363, 2023. 2, 3, 6, 7

[37] Zuxuan Wu, Ser-Nam Lim, Larry S Davis, and Tom Goldstein. Making an invisibility cloak: Real world adversarial attacks on object detectors. In European Conference on Computer Vision, pages 1–17. Springer, 2020. 3

[38] Chaowei Xiao, Jun-Yan Zhu, Bo Li, Warren He, Mingyan Liu, and Dawn Song. Spatially transformed adversarial examples. arXiv preprint arXiv:1801.02612, 2018. 3, 6

[39] Dan Xu, Wanli Ouyang, Elisa Ricci, Xiaogang Wang, and Nicu Sebe. Learning cross-modal deep representations for robust pedestrian detection. In 2017 IEEE Conference on computer vision and pattern recognition (CVPR), pages 4236–4244. IEEE, 2017. 2

[40] Kaidi Xu, Gaoyuan Zhang, Sijia Liu, Quanfu Fan, Mengshu Sun, Hongge Chen, Pin-Yu Chen, Yanzhi Wang, and Xue Lin. Adversarial t-shirt! evading person detectors in a physical world. In European conference on computer vision, pages 665–681. Springer, 2020. 1

[41] Maoxun Yuan and Xingxing Wei. C<sup>2</sup>former: Calibrated and complementary transformer for rgb-infrared object detection. IEEE Transactions on Geoscience and Remote Sens ing, 62:1–12, 2024. 3, 6, 8

[42] Maoxun Yuan, Xiaorong Shi, Nan Wang, Yinyan Wang, and Xingxing Wei. Improving rgb-infrared object detection with cascade alignment-guided transformer. Information Fusion, 105:102246, 2024. 2

[43] Jiaqing Zhang, Jie Lei, Weiying Xie, Zhenman Fang, Yunsong Li, and Qian Du. Superyolo: Super resolution assisted object detection in multimodal remote sensing im agery. IEEE Transactions on Geoscience and Remote Sensing, 61:1–15, 2023. 2

[44] Lu Zhang, Zhiyong Liu, Shifeng Zhang, Xu Yang, Hong Qiao, Kaizhu Huang, and Amir Hussain. Cross-modality interactive attention network for multispectral pedestrian de tection. Information Fusion, 50:20–29, 2019. 2

[45] Lu Zhang, Xiangyu Zhu, Xiangyu Chen, Xu Yang, Zhen Lei, and Zhiyong Liu. Weakly aligned cross-modal learning for multispectral pedestrian detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5127–5137, 2019. 3

[46] Yi Zhang, Wang Zeng, Sheng Jin, Chen Qian, Ping Luo, and Wentao Liu. When pedestrian detection meets multimodal learning: Generalist model and benchmark dataset. In European Conference on Computer Vision, pages 430–448. Springer, 2024. 2

[47] Kailai Zhou, Linsen Chen, and Xun Cao. Improving multi spectral pedestrian detection by addressing modality imbalance problems. In European Conference on Computer Vision (ECCV), 2020. 3

[48] Xiaopei Zhu, Xiao Li, Jianmin Li, Zheyao Wang, and Xiaolin Hu. Fooling thermal infrared pedestrian detectors in real world using small bulbs. In Proceedings of the AAAI conference on artificial intelligence, pages 3616–3624, 2021. 2

[49] Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, and Xiaolin Hu. Infrared invisible clothing: Hiding from infrared detectors at multiple angles in real world. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13307–13316. IEEE, 2022. 2

[50] Zhengxia Zou, Keyan Chen, Zhenwei Shi, Yuhong Guo, and Jieping Ye. Object detection in 20 years: A survey. Proceed ings ofthe IEEE, 111(3):257–276, 2023. 1