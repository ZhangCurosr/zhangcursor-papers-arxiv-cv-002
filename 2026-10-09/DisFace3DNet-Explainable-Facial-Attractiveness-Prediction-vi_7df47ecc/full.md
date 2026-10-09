# DisFace3DNet: Explainable Facial Attractiveness Prediction via 3D Component Disentanglement

Fenggui Rao , Yan Luximon , and Jie Zhang , Member, IEEE

Abstract—Facial attractiveness prediction usually assigns one overall rating, leaving the roles of shape, appearance, and viewing conditions implicit. We propose DisFace3DNet, which uses 3D component disentanglement to learn seven component reference scores from overall ratings with auxiliary weak semantic supervision, without human-labeled component targets. Designated 3D representations and image cues feed jointly learned routes for identity, skin, hair, light, background, expression, and pose. A constrained fit then combines five static and two signed dynamic scores into the overall rating, exposing each component’s numerical contribution and supporting component-specific comparisons across images. On SCUT-FBP5500, DisFace3DNet achieves a Pearson correlation of 0.8904 ± 0.0063 (mean ± standard deviation across five folds) with average human ratings; its component terms reconstruct every held-out prediction to numerical precision. Skin, hair, and facial shape account for the largest component-wise prediction variation. Human evaluation supports the score directions for facial shape, skin, and hair; expression agreement is weaker. DisFace3DNet thus connects overall prediction to quantitative analysis of the facial and contextual cues entering each estimate.

Index Terms—Facial attractiveness explanation, facial attractiveness prediction, 3D component disentanglement

## I. INTRODUCTION

Facial attractiveness prediction (FAP) aims to estimate human attractiveness evaluations from face images [1], [2], while facial attractiveness explanation (FAE) seeks to identify the visual factors associated with these evaluations and to understand how predictive models use such information. Both tasks are relevant to the study and modeling of facial perception: attractiveness evaluations vary with facial shape and texture [3], expression, and pose [4]. Digital-human design [5], hairstyle try-on [6], and live-video retouching [7] involve comparing alternative facial appearances. Prediction provides a quantitative basis for such comparisons, while explanation helps identify the visual features to consider when refining a design or retouching setting.

Existing FAP methods predict overall ratings through regression, retain rating spread through label-distribution learning, and model relative preferences through ranking [1], [8], [9], [10]. Geometric, attribute-aware, and multimodal features enrich their representations [11], [12], [7]. However, their overall predictions do not reveal separately reportable scores for facial shape, appearance, and capture conditions, or how these scores contribute to the predicted rating. Post-hoc explanation methods use visual attribution, concept analysis, or image editing to identify relevant regions, semantic sensitivities, or appearance changes [13], [14], [15]. Part-based prediction methods also learn regional sub-scores [16]. Figure 1 contrasts overall prediction (a), post-hoc explanation (b), and DisFace3DNet (c). Our objective is to separate 3D structure, surface appearance, and viewing conditions into component scores with explicit additive roles in prediction. Obtaining reliable component-level training labels, however, poses a fundamental challenge: facial shape, albedo, and lighting jointly determine a 2D face image, making it difficult for human annotators to assign an absolute attractiveness score to each factor in isolation. Attribute-focused pairwise comparisons can test relative preferences, but do not supply independently calibrated scores for these abstract components.

To address these limitations, we propose Dis-Face3DNet. To the best of our knowledge, Dis-Face3DNet is the first model to use 3D component disentanglement for explainable facial attractiveness prediction through jointly learned component scores and their explicit additive contributions. Designated 3D and image inputs separate structural, appearance, and viewingcondition evidence. Auxiliary weak semantic labels orient the component scores under overall-rating supervision, and constrained fusion fixes their numerical relationship to the overall prediction.

![](images/4a1c5d87d6bafc5175a1a684ad7e193c1177988ca1527edd2df3ea2208fa8ba3.jpg)  
Fig. 1. Prediction and explanation interfaces. (a) Overall prediction through regression, distribution learning, or ranking [1], [8], [10]. (b) Post-hoc explanation [13], [17], [18], illustrated by Grad-CAM. (c) DisFace3DNet uses separate component predictors to produce component reference scores, which are combined by weighted fusion into the overall rating. Head renderings illustrate the components; scores are symbolic. E and D denote encoder and decoder.

This paper makes three contributions:

• We propose DisFace3DNet<sup>1</sup>, an explainable facial attractiveness prediction model based on 3D component disentanglement that produces seven separately reportable component reference scores, supporting component-specific image comparisons and analysis of each component’s contribution to the overall rating.

• We develop designated 3D and image inputs with jointly learned branches for five static and two signed dynamic components. Auxiliary semantic guidance orients the scores, and constrained fusion yields explicit additive prediction terms.

• We evaluate prediction accuracy and use matched controls to test inputs, semantic guidance, and joint learning alongside component correspondence and dynamic-score variation. Human evaluation supports the score directions for identity, skin, and hair.

## II. RELATED WORK

## A. Facial Attractiveness Prediction

Facial attractiveness prediction methods estimate overall human ratings, rating distributions, or relative preferences from face images [19], [20], [21]. Regressionbased methods map handcrafted or learned facial features to mean ratings [1], [22], [23]; label-distribution learning methods preserve rating spread [8], [24], and rankingbased methods incorporate relative preferences [9], [10]. All three method types learn from overall attractiveness evaluations.

Convolutional neural network (CNN) methods use searched architectures, network ensembles, or ensembles trained with different losses [25], [26], [27], [28]. Transformer-based methods model image-wide dependencies [29], with TransFBP adding cross-attention and attention-guided augmentation [30]. Attentionbased methods emphasize informative cues through co-attention [31], spatial/channel weighting [32], or dynamic attentive convolution [33]. Other methods incorporate attributes through generated filters [11], share features across tasks [34], [35], or learn from paired images [36], while retaining overall-score outputs. Geometry-aware methods use shape–texture fusion [37], [38], structural measurements [39], or depthaware landmarks [40]. GPNet combines full-face and local features with landmark-based regularization [12]. Multimodal methods such as FPEM fuse face-prior and vision–language features [7].

These methods use geometric, attribute, and semantic cues for overall prediction, leaving factor-specific attractiveness scores and contributions implicit. DisFace3DNet addresses this gap through 3D component disentanglement and jointly learned component scoring branches. Designated inputs separate component evidence, while constrained fusion links the seven reference scores to the overall rating through explicit additive terms, enabling component-specific comparisons and contribution analysis.

## B. Facial Attractiveness Explanation

FAE studies examine the factors underlying human evaluation and model predictions. Perceptual studies consider shape, averageness, and skin appearance [41], [42], [43], as well as expression, head tilt, imaging conditions, and observer context [44], [4], [45], [46]. Feature-level analyses examine shape categories, structural measurements, skin texture, and geometric morphometrics [47], [48]; an artificial-face trustworthiness review discusses related feature and evaluation-design effects [49]. Controlled 3D studies vary shape and reflectance separately to measure perceptual responses [3], [50]. These studies motivate factor selection; explaining a model prediction also requires specifying each factor’s role in the scoring rule.

Post-hoc explanation methods analyze a trained prediction model. Grad-CAM uses gradients to weight feature maps and localize relevant regions [13]; Grad-CAM++, Score-CAM, and Ablation-CAM use alternative localization mechanisms [17], [18], [51]. Conceptbased methods measure sensitivity along learned semantic directions [14], while beautification and attractiveness-guided editing methods reveal scoreassociated appearance changes [15], [52]. These post-hoc methods characterize model responses without assigning component attractiveness scores. Part-based explanation methods instead learn sub-scores within a prediction model. SCAT [16] uses multilayer perceptrons (MLPs) for facial-part sub-scores and a Transformer to aggregate 2D part embeddings into an overall rating. A consistency loss encourages SCAT’s overall rating to match the mean sub-score, but does not enforce an exact additive prediction rule. The facial-part decomposition also does not separate shape, appearance, and viewing conditions.

Concept bottleneck models connect supervised intermediate concepts to predictions [53]. For FAE, the remaining gap is to connect separated factors to component scores and explicit contributions. DisFace3DNet builds designated inputs from reconstructed geometry, albedo, lighting, pose, and expression [54], [55], [56], together with segmented hair and background. Weak semantic guidance orients component scores under overall-rating supervision, and constrained fusion makes their additive roles explicit without human-labeled component targets.

## III. METHOD

## A. Overview

DisFace3DNet takes a face image and its benchmarkgroup label and returns an overall estimate, seven scores, and their weighted contributions (Fig. 2). Component Disentanglement prepares inputs for the Static and Dynamic Component Encoders; Component Score Fusion combines their outputs. Demographic Conditioning supplies the shared embedding g (Section III-C).

Here, 3D component disentanglement links each designated input to a component-specific branch and its prediction term. For a single image, identity, skin, and hair describe structure and appearance, while light and background describe capture context. These five static scores form a nonnegative base; signed expression and pose scores adjust it for facial state and view.

## B. Component Disentanglement

The seven inputs are cached before attractiveness training. The FFHQ-UV pipeline [54], using Deep3DFaceRecon [55] and the HiFi3D++ topology [58], provides neutral geometry, albedo, and spherical-harmonic lighting. 3DDFA-V3 [56] provides pose and expression, and a 19-class BiSeNet [59] provides hair and background masks. Complete inputconstruction details are provided in Section S2.A of the supplemental materials.

1) Identity, Skin, and Light: Identity represents expression-neutral facial structure with nine UV channels: three normals, three positions, and three curvatures. Three symmetry and four scale descriptors augment the encoded geometry without supplying skin texture. Skin concatenates intrinsic RGB albedo with fine-, middle-, and coarse-scale RGB detail from the unwrapped texture into 12 channels, excluding shape coordinates. Light evaluates 27 spherical-harmonic coefficients on neutralmesh normals to produce three UV channels.

2) Pose and Expression: Expression uses expressive and neutral Basel Face Model (BFM) meshes [60] with shared identity coefficients α. Expression coefficients are δ for the expressive mesh and zero for the neutral mesh:

$$
\Delta V = V _ { \mathrm { e x p } } ( \alpha , \delta ) - V _ { \mathrm { n e u } } ( \alpha , { \bf 0 } ) .\tag{1}
$$

Transferring this displacement to HiFi3D++ and UV space yields three displacement-normal channels, three displacement channels standardized and compressed with a hyperbolic tangent, and one log-magnitude channel. Pose uses the original-image pitch–yaw–roll vector. Together, these inputs recover state and view information suppressed by neutral frontal fitting.

3) Hair and Background: Hair retains segmented RGB hair pixels, with non-hair pixels zeroed before square padding and resizing. To summarize the surrounding scene, Background uses a 15-dimensional vector of color, texture, area, and skin-contrast statistics. All routes also receive the supplied group embedding.

## C. Benchmark-Group Conditioning

Demographic Conditioning in Fig. 2 maps the supplied benchmark-group label to an eight-dimensional

![](images/d744679a95cee006b81716459756636b48db365f93d46445cc3f328a186d5c0a.jpg)  
Fig. 2. Architecture of DisFace3DNet. Component Disentanglement provides designated inputs to the Static and Dynamic Component Encoders, which produce five nonnegative scores and signed expression and pose scores, respectively. Component Score Fusion yields the overall rating and per-image weighted contributions. Demographic Conditioning provides a shared embedding to the component-specific feature-wise linear modulation (FiLM) [57] layers.

embedding g shared by all branches. Before score prediction, each branch applies its own feature-wise linear modulation (FiLM) [57] transformation,

$$
\mathrm { F i L M } _ { k } ( f _ { k } , g ) = \left( 1 + \gamma _ { k } ( g ) \right) \odot f _ { k } + \eta _ { k } ( g ) ,\tag{2}
$$

where $f _ { k }$ is component $k ' s$ feature vector, $\gamma _ { k }$ and $\eta _ { k }$ map $g$ to feature-wise scales and offsets, and $\odot$ denotes element-wise multiplication. This lets each branch condition its score on the benchmark group while sharing its encoder across groups.

## D. Static Component Encoder

Four EfficientNet-B0 [61] encoders separately process the Identity, Skin, and Light UV Maps and Hair Image; an MLP encodes the 15-dimensional background vector. Each pooled feature passes through FiLM [57] and a score-prediction MLP; the identity route appends symmetry and scale descriptors. Sigmoids produce $S _ { \mathrm { i d e } } .$ $S _ { \mathrm { s k i n } } , S _ { \mathrm { l i g h t } } , S _ { \mathrm { h a i r } }$ , and $S _ { \mathrm { b g } }$

## E. Dynamic Component Encoder

Expression and pose can have signed associations with attractiveness across facial states and views [62], [63], [44]. An MLP encodes the 3D Pose Vector, and an EfficientNet-B0 [61] encodes the Expression UV Map. FiLM [57] and a final MLP produce $S _ { \mathrm { p o s e } }$ and $S _ { \mathrm { e x p } }$

## F. Component Score Fusion

During representation learning, positive normalized coefficients combine the static scores $S _ { k } ( x _ { k } , g ) \in [ 0 , 1 ]$ For $K _ { \mathrm { s t } } = \{ \mathrm { i d e } ,$ skin, hair, light, bg}, the static score is

$$
\begin{array} { r l } & { S _ { \mathrm { s t } } = \displaystyle \sum _ { k \in { \mathcal K } _ { \mathrm { s t } } } W _ { k } S _ { k } + b _ { \mathrm { s t } } , } \\ & { W _ { k } = \frac { \mathrm { s o f t p l u s } \left( \alpha _ { k } \right) } { \displaystyle \sum _ { j \in { \mathcal K } _ { \mathrm { s t } } } \mathrm { s o f t p l u s } \left( \alpha _ { j } \right) } . } \end{array}\tag{3}
$$

Expression and pose use separate learned positive scales on signed readouts,

$$
S _ { \mathrm { e x p } } = W _ { \mathrm { e x p } } \operatorname { t a n h } h _ { \mathrm { e x p } } ( x _ { \mathrm { e x p } } , g ) ,\tag{4}
$$

$$
S _ { \mathrm { p o s e } } = W _ { \mathrm { p o s e } } \operatorname { t a n h } h _ { \mathrm { p o s e } } ( x _ { \mathrm { p o s e } } , g ) ,\tag{5}
$$

$$
S _ { \mathrm { d y } } = S _ { \mathrm { e x p } } + S _ { \mathrm { p o s e } } .\tag{6}
$$

The training score $\tilde { y } _ { n } = S _ { \mathrm { s t } } + S _ { \mathrm { d y } }$ lets overall-rating supervision jointly shape the component branches; n denotes the normalized rating scale.

After freezing the network, we standardize the static and dynamic groups to calibrate their relative influence while retaining the learned static-component proportions. We use their inner-training means $\mu _ { \ell }$ and standard deviations (SDs) $\sigma _ { \ell } \colon$

$$
Z _ { \ell , i } = ( S _ { \ell , i } - \mu _ { \ell } ) / \sigma _ { \ell } , \qquad \ell \in \{ \mathrm { s t } , \mathrm { d y } \} .\tag{7}
$$

The normalized score before clipping is

$$
\hat { y } _ { n , i } ^ { \mathrm { p r e } } = b _ { 0 } + W _ { \mathrm { s t } } Z _ { \mathrm { s t } , i } + W _ { \mathrm { d y } } Z _ { \mathrm { d y } , i } .\tag{8}
$$

The final fit estimates these three coefficients from the inner-training set $\tau$ by minimizing

$$
\mathcal { L } _ { \mathrm { f i t } } = \sum _ { i \in \mathcal { T } } \left[ ( \hat { y } _ { n , i } ^ { \mathrm { p r e } } - y _ { i } ) ^ { 2 } + 0 . 3 ( \hat { y } _ { n , i } ^ { \mathrm { p r e } } - t _ { n , i } ) ^ { 2 } \right] ,\tag{9}
$$

subject to $W _ { \mathrm { s t } } , W _ { \mathrm { d y } } \geq 0$ , with $b _ { 0 }$ unconstrained. Here, $r _ { i }$ is the benchmark mean rating and $t _ { i }$ is the foldsafe FPEM [7] prediction for image i; $y _ { i } = ( r _ { i } - 1 ) / 4$ and $t _ { n , i } ~ = ~ ( t _ { i } - 1 ) / 4$ normalize these ratings. This deterministic convex fit updates only fusion coefficients. The frozen network, statistics, and coefficients produce held-out ratings as $\mathrm { c l i p } ( 1 + 4 \hat { y } _ { n } ^ { \mathrm { p r e } } , 1 , 5 )$

Expanding the fitted score gives component terms for $k \in \mathcal { K } _ { \mathrm { s t } }$ and $d \in \{ \exp , \mathrm { p o s e } \}$ :

$$
C _ { i , k } = \frac { W _ { \mathrm { s t } } W _ { k } S _ { i , k } } { \sigma _ { \mathrm { s t } } } , \qquad C _ { i , d } = \frac { W _ { \mathrm { d y } } S _ { i , d } } { \sigma _ { \mathrm { d y } } } .\tag{10}
$$

The effective intercept collects the fixed offsets:

$$
b _ { \mathrm { e f f } } = b _ { 0 } + \frac { W _ { \mathrm { s t } } ( b _ { \mathrm { s t } } - \mu _ { \mathrm { s t } } ) } { \sigma _ { \mathrm { s t } } } - \frac { W _ { \mathrm { d y } } \mu _ { \mathrm { d y } } } { \sigma _ { \mathrm { d y } } } .\tag{11}
$$

The vector $\mathbf { c } _ { i } ~ = ~ [ C _ { i , \mathrm { { i d e } } } , C _ { i , \mathrm { { s k i n } } } , C _ { i , \mathrm { { h a i r } } } , C _ { i , \mathrm { { l i g h t } } } , C _ { i , \mathrm { { b g } } } ,$ $C _ { i , \mathrm { e x p } } , C _ { i , \mathrm { p o s e } } ]$ satisfies $\begin{array} { r } { \hat { y } _ { n , i } ^ { \mathrm { p r e } } = b _ { \mathrm { e f f } } + \sum _ { k } C _ { i , k } . } \end{array}$ . Each $C _ { i , k }$ contributes $4 C _ { i , k }$ rating points before clipping.

## G. Joint Learning and Frozen Fusion Fitting

As shown in Fig. 3, Stage 1 (Component Encoders Joint Training) jointly optimizes all component routes, heads, static-component weights, and dynamic scales in shared batches. After inner-validation checkpoint selection, Stage 2 (Static–Dynamic Weight Fusion Training) fits only fusion coefficients on frozen, standardized static/dynamic outputs using inner-training data (Section III-F).

Benchmark ratings and scalar FPEM targets supervise the provisional score [7], [65]; teacher targets receive uniform weights across samples. A 21-bin distribution head predicts $\pi _ { i }$ against a smoothed Gaussian target $q _ { i }$ derived from the benchmark mean and SD. Labeldistribution learning (LDL) uses Kullback–Leibler (KL) divergence [8], [66]; agreement aligns the scalar and distribution means. For batch size B,

$$
\mathcal { L } _ { \mathrm { s c o r e } } ^ { \mathrm { G T } } = B ^ { - 1 } \sum _ { i } | \tilde { y } _ { n , i } - y _ { i } | ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { s c o r e } } ^ { \mathrm { K D } } = B ^ { - 1 } \sum _ { i } | \tilde { y } _ { n , i } - t _ { n , i } | ,\tag{13}
$$

$$
\mathcal { L } _ { \mathrm { L D L } } = B ^ { - 1 } \sum _ { i , m } q _ { i m } \log ( q _ { i m } / \pi _ { i m } ) ,\tag{14}
$$

$$
\bar { y } _ { \pi , i } = ( \sum _ { m } c _ { m } \pi _ { i m } - 1 ) / 4 ,\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { a g r e e } } = B ^ { - 1 } \sum _ { i } | \tilde { y } _ { n , i } - \mathrm { s g } ( \bar { y } _ { \pi , i } ) | .\tag{16}
$$

Here, $c _ { m }$ is a bin center and sg stops gradients: LDL trains the distribution head, while agreement trains the component-score path. GT and teacher ordering losses [9], [10] use random batch pairs whose absolute normalized target difference exceeds 0.15; both use a margin of 0.1.

a) Weak semantic guidance: Weak semantic labels guide what each branch scores. In our experiments, all 5,500 SCUT-FBP5500 images have seven jointly generated GPT-5.5 [64] weak labels on a 1–5 ordinal scale from the OpenAI API. We map static labels to [0, 1] and center expression/pose labels at neutral on $[ - 0 . 1 0 , 0 . 1 0 ] / [ - 0 . 0 5 , 0 . 0 5 ]$ . These ranges define training targets; standardization and fitted fusion determine their contributions to the final rating. For valid targets $v _ { i k }$

$$
\mathcal { L } _ { \mathrm { s e m } } = \frac { 1 } { 7 } \sum _ { k } \frac { 1 } { N _ { k } } \sum _ { i } M _ { i k } | S _ { i k } - v _ { i k } | .\tag{17}
$$

Here, $M _ { i k }$ selects observed labels and $N _ { k }$ counts valid batch targets for component k. These weak labels orient component branches under primary overall-rating supervision. The full prompts and teacher protocol are provided in Section S2.B of the supplemental materials.

b) Static-score balance: Overall-rating supervision can be satisfied even when a static branch produces nearly constant scores. To encourage variation across all five static branches, we use

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b a l } } = { \mathrm { V a r } } _ { k \in \mathcal { K } _ { \mathrm { s t } } } \left[ \log \left( { \mathrm { S t d } _ { i } ( S _ { i k } ) + 0 . 0 1 } \right) \right] . } \end{array}\tag{18}
$$

The loss reduces differences in static-score dispersion within a batch. Expression and pose learn their signed scores through semantic, score, KD, agreement, and ordering supervision.

The complete joint objective is

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { s c o r e } } ^ { \mathrm { G T } } + \lambda _ { \mathrm { K D } } \mathcal { L } _ { \mathrm { s c o r e } } ^ { \mathrm { K D } } + \mathcal { L } _ { \mathrm { L D L } } + \lambda _ { \mathrm { a g r e e } } \mathcal { L } _ { \mathrm { a g r e e } } } \\ & { \qquad + \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } } \\ & { \qquad + \lambda _ { \mathrm { r a n k } } ( \mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { G T } } + \mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { K D } } ) . } \end{array}\tag{19}
$$

FPEM [7], weak labels, and the distribution head are used only for training; group embeddings remain active in deployed FiLM [57].

## IV. EXPERIMENTS

## A. Dataset

SCUT-FBP5500 [2] contains 5,500 facial images with mean human attractiveness ratings on a 1–5 scale. It provides landmarks, rating distributions, and four benchmark-group labels: Asian male (AM; 2,000 images), Asian female (AF; 2,000 images), Caucasian male (CM; 750 images), and Caucasian female (CF; 750 images). In this evaluation, the learned scores characterize mean human ratings within the population and rating context represented by SCUT-FBP5500 [41], [45].

![](images/e97cfce045e7c714937e24532822dcd5505b07ba4ef106128bb8198bf9bbaa9c.jpg)  
Fig. 3. Two-stage training in DisFace3DNet. Stage 1 jointly trains seven component routes using ground-truth (GT) ratings, scalar knowledge distillation (KD) from FPEM [7], and GPT-5.5 weak labels [64]. The provisional score ${ \tilde { y } } _ { n }$ and distribution head π receive the indicated loss terms. Stage 2 freezes the network and fits fusion coefficients on the standardized static/dynamic outputs through ${ \mathcal { L } } _ { \mathrm { f i t } }$

The publisher’s fixed five-fold protocol [2] allocates 4,400 training and 1,100 test images per split. DisFace3DNet and matched controls share folds and paired seeds, with a deterministic 3,960/440 innertraining/validation split. Each of the 5,500 images is evaluated once by a model trained on the other folds.

## B. Evaluation Metrics

We evaluate overall predictions against mean human ratings using the Pearson correlation coefficient (PCC), Spearman rank correlation coefficient (SRCC), mean absolute error (MAE), and root mean square error (RMSE). PCC and SRCC measure linear and rank association; MAE and RMSE measure prediction error on the 1–5 rating scale. Higher correlations and lower errors indicate better prediction.

## C. Implementation Details

1) Input Preparation: All UV/image inputs use 384× 384 resolution; pose and background are encoded as vectors. Section S2.A of the supplemental materials gives input-construction, normalization, and augmentation details.

2) Training Configuration: EfficientNet-B0 [61] encoders use ImageNet initialization [74]. We train each fold independently in PyTorch [75] with AdamW [76], batch size 12, 70 joint epochs at $3 \times 1 0 ^ { - 4 }$ , and 30 refinement epochs at $3 \times 1 0 ^ { - 5 }$ . The semantic coefficient is prespecified as $\lambda _ { \mathrm { s e m } } = 0 . 2$ using inner-validation prediction and component diagnostics. Inner-validation PCC selects the checkpoint; Stage 2 weight fusion training uses only inner-training data. All choices, standardization statistics, and fitted coefficients are fixed before testfold evaluation. Layer widths, initialization rules, loss weights, and inference settings are given in Section S2.A of the supplemental materials.

## D. Baselines

Inception-V3 [67] and ResNeXt-50 [68] provide convolutional reference models, while REX-INCEP [27] combines their features in a two-branch architecture. ViT-B [70] represents transformer-based prediction. 2D-FAP [73] uses a lightweight network with dual label distributions. FPEM [7] is a state-of-the-art FAP method that combines face-prior and vision–language features, making it a strong reference for overall prediction.

## V. RESULTS

## A. Overall Score Prediction

FPEM achieves higher correlations and lower errors than DisFace3DNet (Table I). Its face-prior and vision– language features support flexible overall prediction. DisFace3DNet restricts each branch to designated 3D, segmented, or statistical inputs and combines seven scores additively. These constraints trade predictive flexibility for a decomposition that can be inspected component by component.

To compare average prediction levels, Fig. 4 tests mean differences between DisFace3DNet and FPEM. No significant difference is detected overall or within groups after Holm correction. Figure 5 compares individual predictions with human ratings across the four groups, showing close estimates as well as rating and ordering errors. Section S5 of the supplemental materials gives the sampling protocol and statistical analyses.

TABLE I  
OVERALL ATTRACTIVENESS PREDICTION ON SCUT-FBP5500 [2], GROUPED BY MODEL CATEGORY
<table><tr><td rowspan="2">Model</td><td rowspan="2">Year</td><td rowspan="2">Backbone</td><td colspan="2">Correlation ↑</td><td colspan="2">Error ↓</td></tr><tr><td>PCC</td><td>SRCC</td><td>MAE</td><td>RMSE</td></tr><tr><td colspan="6"></td></tr><tr><td>CNN-based learning</td><td>2016</td><td>Inception-V3</td><td>0.918</td><td>0.905</td><td></td><td></td></tr><tr><td>Inception-V3 [67] ResNeXt-50 [68]</td><td>2017</td><td>ResNeXt-50</td><td>0.911</td><td>0.899</td><td></td><td></td></tr><tr><td>DALDL [24]</td><td>2019</td><td>ResNeXt-50</td><td>0.920</td><td></td><td>0.200</td><td>0.269</td></tr><tr><td>R3CNN [9]</td><td>2019</td><td>ResNeXt-50</td><td>0.914</td><td></td><td>0.212</td><td>0.280</td></tr><tr><td>NAS4FBP [25]</td><td>2022</td><td>Custom</td><td>0.939</td><td></td><td>0.180</td><td>0.238</td></tr><tr><td>Dynamic ER-CNN [26]</td><td>2023</td><td>ResNeXt-50 + Inception-V3</td><td>0.921</td><td></td><td>0.204</td><td>0.267</td></tr><tr><td>REX-INCEP [27]</td><td>2022</td><td>ResNeXt-50 + Inception-V3</td><td>0.917</td><td>0.907</td><td></td><td></td></tr><tr><td>Loss ensemble [28]</td><td>2023</td><td>FIAC-Net [69]</td><td>0.931</td><td></td><td>0.203</td><td>0.261</td></tr><tr><td>RankNet [10]</td><td>2024</td><td>ResNet-50</td><td>0.928</td><td></td><td>0.191</td><td>0.255</td></tr><tr><td colspan="6">Transformer-based learning</td><td></td></tr><tr><td>ViT-B [70]</td><td>2021</td><td>ViT-B</td><td>0.880</td><td>0.869</td><td></td><td></td></tr><tr><td>FBPFormer [29]</td><td>2023</td><td>ViT</td><td>0.918</td><td></td><td>0.205</td><td>0.273</td></tr><tr><td colspan="6">Attention-based learning</td><td></td></tr><tr><td>DyAttenConv [33]</td><td>2024</td><td>ResNet-50</td><td>0.906</td><td></td><td>0.220</td><td>0.295</td></tr><tr><td colspan="6">Semi-supervised learning</td><td></td></tr><tr><td>NFME [71]</td><td>2020</td><td>VGG-Face</td><td>0.866</td><td></td><td>0.268</td><td>0.346</td></tr><tr><td>FME [72]</td><td>2022</td><td>VGG-Face + ResNet-50</td><td>0.911</td><td></td><td>0.221</td><td>0.287</td></tr><tr><td colspan="6">Lightweight learning</td><td></td></tr><tr><td>CoAttention [31]</td><td>2019</td><td>MobileNetV2</td><td>0.926</td><td>0.916</td><td>0.202</td><td>0.266</td></tr><tr><td>RIRSCA [32]</td><td>2020</td><td>Custom</td><td>0.878</td><td></td><td>0.252</td><td>0.332</td></tr><tr><td>2D-FAP (60/40) [7]</td><td>2025</td><td>MobileNetV2</td><td>0.915</td><td>0.903</td><td></td><td></td></tr><tr><td>2D-FAP (five-fold) [73]</td><td>2025</td><td>MobileNetV2</td><td>0.928</td><td></td><td>0.196</td><td>0.259</td></tr><tr><td colspan="6">Attribute-aware learning</td><td></td></tr><tr><td>AaNet [11]</td><td>2019</td><td>ResNet-18</td><td>0.906</td><td></td><td>0.224</td><td>0.295</td></tr><tr><td>P-AaNet [11]</td><td>2019</td><td>ResNet-18</td><td>0.897</td><td></td><td>0.229</td><td>0.304</td></tr><tr><td colspan="6">Multi-task learning</td><td></td></tr><tr><td>HMTNet [34]</td><td>2019</td><td>Custom</td><td>0.878</td><td></td><td>0.250</td><td>0.326</td></tr><tr><td>BranchMTL [35]</td><td>2020</td><td>ResNet-50</td><td>0.937</td><td></td><td>0.183</td><td>0.242</td></tr><tr><td>BSN [36]</td><td>2024</td><td>ResNeXt-50</td><td>0.926</td><td></td><td>0.198</td><td>0.263</td></tr><tr><td colspan="6">Multimodal learning</td><td></td></tr><tr><td>FPEM [7]</td><td>2025</td><td>Swin-T + CLIP</td><td>0.937</td><td>0.932</td><td>0.185</td><td>0.242</td></tr><tr><td colspan="6">3D component-based learning</td><td></td></tr><tr><td>DisFace3DNet (Ours)</td><td>2026</td><td>EfficientNet-B0 + MLP</td><td>0.890</td><td>0.877</td><td>0.258</td><td>0.337</td></tr></table>

## B. Component Score Prediction

1) Qualitative Component Analysis: Figure 6 shows 48 lower- and higher-score identity and skin heads. Neutral gray identity heads reveal facial contours and feature proportions. Higher-score skin examples generally look more uniform; lower-score examples show stronger local color and texture variation. For visualization, skin heads combine personal-albedo-guided lowfrequency color with FFHQ-UV [54] detail under shared rendering conditions.

Figure 7 extends this comparison to 168 examples covering all seven components. Hair rows contrast silhouette and framing, while light and background rows show differences in capture context. Expression and pose rows associate signed scores with eye/mouth appearance and head orientation.

2) Component Score Distribution Analysis: Figure 8 shows broad static-score subranges, whereas expression and pose concentrate near zero, consistent with their roles as signed adjustments. Light and background show the largest static-score group-mean shifts. Shared axes and bandwidths allow direct comparisons of the raw scores $S _ { k }$ across groups; the overall distribution retains the dataset’s group proportions.

## 3) Component Contribution Analysis:

a) Component contribution variation: For image i in official fold f(i), we measure component variation using centered rating-point contributions:

![](images/00b8d698a97f171cecfd96103b52b9be76a3520c64a33d6aa1659a243c438bab.jpg)

![](images/48b5d5acea3558398650a8b188f3e5d7fc7bf3ec1e10f48fae0477ee2e35a3ca.jpg)

![](images/2a03cdb7d005004dac8d4a5c55a6c93f0e0bdb3ad0adc4d92872ae97a37c8497.jpg)  
Caucasian Male

![](images/a20127a4ee03e1d06e681c07de2b551a26e52168d4539fc5411462790f2cf7d2.jpg)  
Caucasian Female

![](images/1da4c9775f06d5720388366f1649cf4b670b588a366d6b438ccebe8fb89d38a0.jpg)

<sub>o</sub>r Overall Asian Male Asian Female Caucasian Male Caucasian Female<sub>Fig. 4. DisFace3DNet and FPEM [7] prediction distributions for 1,000 held-out images. White dots and black bars indicate medians and</sub> 5sinterquartile ranges. Paired t-test p-values are unadjusted overall and Holm-adjusted across the four groups; ns denotes p ≥ 0.05.  
![](images/2eea034fe27bb1c2d0087138ebd9e89baf116b2f24c910bbf05b7a1f64b005ed.jpg)  
Fig. 5. Qualitative prediction examples. Four images per group are ordered by increasing human mean rating. Blue/red encodes lower/higher ratings on a fixed 1–5 scale.Lo(a)

![](images/22a395c0254ad11ef51edfdd08aa7b1fd5597b718d4d31f26022b2bac15cafd6.jpg)  
0.26 0.28 0.31 0.61 0.68 0.74 0.32 0.35 0.36 0.73 0.76 0.80<sub>Fig. 6. 3D identity and skin examples: (a) Asian Male, (b) Asian Female, (c) Caucasian Male, (d) Caucasian Female. Each row shows three</sub> lower-score and three higher-score heads. Identity displays neutral geometry; skin uses personal-albedo-guided color and FFHQ-UV detail for display. Labels give raw component scores on [0, 1].

$$
d _ { i k } = 4 \left( C _ { i k } - \overline { { C } } _ { f ( i ) , k } \right) ,
$$

$$
a _ { k } = \frac { 1 } { N } \sum _ { i } | d _ { i k } | , \qquad p _ { k } = 1 0 0 \frac { a _ { k } } { \sum _ { j } a _ { j } } .\tag{20}
$$

N counts evaluated images, and $\overline { { C } } _ { f \left( i \right) , k }$ is the foldspecific component mean. The mean absolute deviation $a _ { k }$ measures component k’s variation in rating points; $p _ { k }$ expresses its share of the sum across components (100% in total).

![](images/a6193fdd33f286932c9e8f7b2f17ce73133b70ff84985a8eaab3471e636a6f03.jpg)  
Fig. 7. Component-score examples by benchmark group (columns), with three lower- and three higher-score images per component. Labels show $S _ { k }$ for static components ([0, 1]), 10S<sub>k</sub> for expression, and 100S<sub>k</sub> for pose. Displayed expression/pose ranges are [−0.48, +0.80]/[−0.29, +0.65].

![](images/d2c35d467de5bd912892b630dc85a179bacdfdaa5c9807c9bcda658cab75e569.jpg)  
Fig. 8. Component-score distributions for all 5,500 held-out SCUT-FBP5500 images and the four benchmark groups. Static scores lie in [0, 1]; expression and pose use signed axes. Density curves have equal peak heights, and labels report mean ± sample SD. Scores precede fold-specific standardization and fusion.

For AM, AF, CM, and CF, centering within each fold– group measures variation among images from the same group. The 95% intervals resample images within fold– group strata with the fitted models fixed (supplemental Section S3.A).

For all 5,500 held-out images, $b _ { \mathrm { e f f } } + \sum _ { k } C _ { i , k }$ matches the normalized prediction within $1 . 5 3 \times 1 0 ^ { - 7 }$ maximum absolute error. All predicted ratings lie within [1, 5] before clipping, preserving the additive decomposition in the final output.

b) Relative contribution magnitudes: Skin, hair, and identity-related facial shape lead prediction variation in every fold; skin has the largest overall share (Table II; Eq. (20)).

c) Within-group contributions: The same three components lead within all four groups (Table II). Background’s smaller within-group shares show that its overall variation includes substantial group-mean differences. Light’s share is highest in AF and lowest in CF.

TABLE II  
RELATIVE COMPONENT CONTRIBUTION MAGNITUDES (%) WITH 95% IMAGE-BOOTSTRAP INTERVALS CONDITIONAL ON THE FROZEN MODELS. EACH COLUMN SUMS TO 100% BEFORE ROUNDING. IDENTITY DENOTES FACIAL SHAPE AND STRUCTURE. AM, AF, CM, AND CF DENOTE ASIAN MALE, ASIAN FEMALE, CAUCASIAN MALE, AND CAUCASIAN FEMALE, RESPECTIVELY
<table><tr><td>Component</td><td>Overall</td><td>AM</td><td>AF</td><td>CM</td><td>CF</td></tr><tr><td>Identity</td><td>18.22 [17.98, 18.44]</td><td>18.32 [17.84, 18.79]</td><td>19.86 [19.47, 20.23]</td><td>20.96 [20.15, 21.73]</td><td>21.96 [21.33, 22.56]</td></tr><tr><td>Skin</td><td>34.29 [33.96, 34.63]</td><td>35.85 [35.24, 36.46]</td><td>35.92 [35.36, 36.51]</td><td>36.21 [35.14, 37.29]</td><td>39.71 [38.81, 40.60]</td></tr><tr><td>Hair</td><td>20.90 [20.55, 21.25]</td><td>22.25 [21.58, 22.92]</td><td>24.20 [23.60, 24.79]</td><td>25.54 [24.36, 26.67]</td><td>24.14 [23.17, 25.09]</td></tr><tr><td>Light</td><td>6.82 [6.72, 6.92]</td><td>6.92 [6.73, 7.12]</td><td>7.08 [6.90, 7.25]</td><td>6.12 [5.75, 6.51]</td><td>4.34 [4.05, 4.63]</td></tr><tr><td>Background</td><td>10.71 [10.56, 10.87]</td><td>6.02 [5.65, 6.38]</td><td>3.98 [3.69, 4.28]</td><td>0.91 [0.84, 0.99]</td><td>0.76 [0.70, 0.84]</td></tr><tr><td>Expression</td><td>8.45 [8.28, 8.63]</td><td>9.99 [9.63, 10.37]</td><td>8.37 [8.10, 8.65]</td><td>9.51 [8.89, 10.18]</td><td>8.52 [8.06, 8.99]</td></tr><tr><td>Pose</td><td>0.60 [0.59, 0.62]</td><td>0.65 [0.62, 0.68]</td><td>0.60 [0.58, 0.62]</td><td>0.74 [0.70, 0.79]</td><td>0.57 [0.54, 0.60]</td></tr></table>

![](images/8a555d4f76c51428d5d8f902467443d911994889f6f19cbee1d42e670961dc08.jpg)

![](images/c35015d28c1005becb6a7f74bb130caf850dacdbfde88bb349eed36b5b1731e5.jpg)  
(b) Pairwise agreement  
Fig. 9. Component-specific human evaluation with 20 participants. (a) Restyled skin-trial interface with the original stimulus pair and response options; no response is shown. (b) Pairwise agreement. Yellow circles: fraction of decisive responses matching the model’s score ordering. Blue diamonds: mean per-response agreement fraction with other participants on the same component and image pair. Both exclude unsure responses and repeats; diamonds require a decisive peer. Bars: 95% participant-clustered bootstrap intervals. Vertical line: 50% agreement between independent fair binary choices.

## C. Human Evaluation

Twenty consenting adults (10 male and 10 female) compared randomly sampled image pairs for facial shape (identity), skin, hair, and expression under IRB approval HSEARS20240313013. Responses were Left, Right, or Unsure, with 400 non-repeat responses per component. The post-training study tests whether participants’ choices agree with component-score orderings. Figure 9(a) illustrates the interface, and Fig. 9(b) compares human–model and between-human agreement; Section S4 gives the protocol and statistical analysis.

Human–model agreement exceeds the corresponding between-human point estimates for identity, skin, and hair (Fig. 9(b)), supporting these component rankings. Expression shows weak model alignment and low between-human agreement. This task requires comparison with an imagined neutral expression, and the benchmark lacks identity-matched expression variation for controlled validation. Light, background, and pose remain untested in the formal human evaluation; Section S4 explains their omission using informal pilot feedback.

## D. Ablation Studies

Six retrained controls test component scoring, training objectives, and joint learning using overall and component diagnostics (Table III). Section S3 reports auxiliary controls and teacher diagnostics.

1) Effect of Component Scoring: Replacing component scoring with a capacity-matched scalar head lowers MAE; the paired 95% CI for its PCC difference includes zero. Thus, the additive scores incur a modest absoluteerror cost. Removing dynamic components also gives a PCC interval spanning zero. Replacing componentspecific inputs with a common UV texture lowers PCC, supporting the value of organizing inputs by component (Table III).

2) Effect of the Training Objectives: Removing weak semantic supervision lowers weak-label correspondence for all seven components in every fold despite little change in overall PCC, showing why component correspondence must be assessed alongside overall accuracy. FPEM guidance improves correlations in four folds, but its paired PCC interval includes zero and neither error metric improves.

Five seed-specific sweeps over five semantic coefficients show a trade-off between overall prediction and component correspondence with weak labels. The prespecified $\lambda _ { \mathrm { s e m } } ~ = ~ 0 . 2$ is retained; Fig. 11 of the supplemental materials gives all distributions and paired seed trajectories.

TABLE III  
FIVE-FOLD ABLATIONS OF COMPONENT SCORING, TRAINING OBJECTIVES, AND TRAINING STRATEGY. METRIC SUMMARIES ARE MEAN ± SAMPLE SD ACROSS FOLDS. ∆PCC IS DISFACE3DNET MINUS THE VARIANT (POSITIVE FAVORS DISFACE3DNET), WITH MARGINAL 95% PAIRED IMAGE-BOOTSTRAP CIS
<table><tr><td>Variant</td><td>PCC↑</td><td>∆PCC [95% CI]</td><td>Additional metrics or matched diagnostic</td></tr><tr><td colspan="4">A. Component scoring</td></tr><tr><td>Without component scoring</td><td></td><td>0.8931 ± 0.0063 -0.0027 [-0.0058, 0.0003]</td><td>SRCC 0.8781 ± 0.0066; MAE 0.2472 ± 0.0055; RMSE 0.3229 ± 0.0092</td></tr><tr><td>Without dynamic components</td><td></td><td>0.8891 ± 0.0043 0.0013 [-0.0012, 0.0038]</td><td>SRCC 0.8752 ± 0.0047; MAE  $0 . 2 5 3 4 \pm 0 . 0 0 5 0 \ddagger$  RMSE 0.3314 ± 0.0062</td></tr><tr><td>Without component-specific inputs</td><td></td><td>0.8715 ± 0.0079 0.0189 [0.0141, 0.0238]</td><td>Common UV texture for image branches; auxiliary vectors zeroed</td></tr><tr><td colspan="4">B. Training objectives</td></tr><tr><td>Without FPEM [7] teacher guidance</td><td></td><td>0.8882 ± 0.0054 0.0022 [-0.0001, 0.0044]</td><td>SRCC 0.8730 ± 0.0052; MAE 0.2575 ± 0.0044; RMSE 0.3349 ± 0.0058</td></tr><tr><td>Without weak semantic supervision (λsem = 0)</td><td></td><td>0.8896 ± 0.0048 0.0008 [-0.0015, 0.0030]</td><td>Lower weak-label correspondence for all seven components in every fold; shifted score distributions</td></tr><tr><td colspan="4">C. Training strategy</td></tr><tr><td>Without joint learning throughout</td><td></td><td>0.8875 ± 0.0061 0.0028 [0.0003, 0.0053]</td><td>Full staged curriculum; dynamic-score variation and fitted dynamic weight reduced</td></tr><tr><td>DisFace3DNet (Ours)</td><td>0.8904 ± 0.0063</td><td></td><td>SRCC 0.8766 ± 0.0060; MAE 0.2580 ± 0.0072; RMSE 0.3369 ± 0.0114</td></tr></table>

![](images/d6518077f25e30852c811f5bd710789e25aaf97b222f937a9dc7aef347de4ee9.jpg)  
(a)  
(b)  
(c)  
Fig. 10. Illustrative applications of DisFace3DNet. Component reference scores support (a) digital-human and bio-inspired robot face design, (b) beauty-filter assessment for livestreaming, photography, and social media, and (c) hairstyle design, salon consultation, and wig try-on.

3) Effect of the Training Strategy: Joint learning outperforms the compute-matched staged curriculum in PCC and retains more dynamic-score variation.

## VI. APPLICATIONS

Figure 10 uses component reference scores to illustrate geometry selection for digital-human and bio-inspired robot faces, beauty-filter assessment, and hairstyle comparison for salon consultation and wig try-on. Section S6 of the supplemental materials details these workflows.

## VII. CONCLUSION

DisFace3DNet predicts overall attractiveness through seven reference scores whose weighted contributions sum to each estimate. On SCUT-FBP5500, skin, hair, and identity lead prediction variation, and human comparisons support their score directions. Expression alignment is weaker, while light, background, and pose have not undergone formal human evaluation. The illustrated workflows show how the scores can organize design references and appearance comparisons.

## REFERENCES

[1] Y. Eisenthal, G. Dror, and E. Ruppin, “Facial attractiveness: Beauty and the machine,” Neural Computation, vol. 18, no. 1, pp. 119–142, 2006.

[2] L. Liang, L. Lin, L. Jin, D. Xie, and M. Li, “SCUT-FBP5500: A diverse benchmark dataset for multi-paradigm facial beauty prediction,” in Proc. Int. Conf. Pattern Recognit., 2018, pp. 1598– 1603.

[3] K. Nakamura and K. Watanabe, “Data-driven mathematical model of East-Asian facial attractiveness: The relative contributions of shape and reflectance to attractiveness judgements,” R. Soc. Open Sci., vol. 6, no. 5, 2019, art. no. 182189.

[4] C. A. M. Sutherland, A. W. Young, and G. Rhodes, “Facial first impressions from another angle: How social judgements are influenced by changeable and invariant facial properties,” Br. J. Psychol., vol. 108, no. 2, pp. 397–415, 2017.

[5] C. Wang et al., “MeGA: Hybrid mesh-Gaussian head avatar for high-fidelity rendering and head editing,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 26 274–26 284.

[6] S. Khwanmuang, P. Phongthawee, P. Sangkloy, and S. Suwajanakorn, “StyleGAN Salon: Multi-view latent optimization for pose-invariant hairstyle transfer,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 8609–8618.

[7] H. Li et al., “FPEM: Face prior enhanced facial attractiveness prediction for live videos with face retouching,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 11 458–11 468.

[8] Y.-Y. Fan et al., “Label distribution-based facial attractiveness computation by deep residual learning,” IEEE Trans. Multimedia, vol. 20, no. 8, pp. 2196–2208, 2018.

[9] L. Lin, L. Liang, and L. Jin, “Regression guided by relative ranking using convolutional neural network (R3CNN) for facial beauty prediction,” IEEE Trans. Affect. Comput., vol. 13, no. 1, pp. 122–134, 2022.

[10] H. Lyu, J. Li, Y. Ye, and C.-C. Chang, “A ranking information based network for facial beauty prediction,” IEICE Trans. Inf. Syst., vol. E107.D, no. 6, pp. 772–780, 2024.

[11] L. Lin, L. Liang, L. Jin, and W. Chen, “Attribute-aware convolutional neural networks for facial beauty prediction,” in Proc. Int. Joint Conf. Artif. Intell., 2019, pp. 847–853.

[12] T. Peng, M. Li, F. Chen, Y. Xu, and D. Zhang, “Geometric prior guided hybrid deep neural network for facial beauty analysis,” CAAI Trans. Intell. Technol., vol. 9, no. 2, pp. 467–480, 2024.

[13] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis., 2017, pp. 618–626.

[14] B. Kim et al., “Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV),” in Proc. Int. Conf. Mach. Learn., 2018, pp. 2668–2677.

[15] T. Peng et al., “ISFB-GAN: Interpretable semantic face beautification with generative adversarial network,” Expert Syst. Appl., vol. 236, 2024, art. no. 121131.

[16] D. E. Boukhari and A. Chemsa, “SCAT: The self-correcting aesthetic transformer for explainable facial beauty prediction,” Research Square preprint, Jul. 2025, pp. 1–17. [Online]. Available: https://www.researchsquare.com/article/rs-7003463/v1

[17] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks,” in Proc. IEEE Winter Conf. Appl. Comput. Vis., 2018, pp. 839–847.

[18] H. Wang et al., “Score-CAM: Score-weighted visual explanations for convolutional neural networks,” in Proc. IEEE/CVF Conf Comput. Vis. Pattern Recognit. Workshops, 2020, pp. 111–119.

[19] A. Laurentini and A. Bottino, “Computer analysis of face beauty: A survey,” Comput. Vis. Image Underst., vol. 125, pp. 184–199, 2014.

[20] S. Liu, Y.-Y. Fan, A. Samal, and Z. Guo, “Advances in computational facial attractiveness methods,” Multimedia Tools Appl., vol. 75, no. 23, pp. 16 633–16 663, 2016.

[21] D. E. Boukhari, F. Dornaika, A. Chemsa, and A. Taleb-Ahmed, “A comprehensive review of facial beauty prediction using deep learning techniques,” Eng. Appl. Artif. Intell., vol. 161, 2025, art. no. 112009.

[22] S. Wang, M. Shao, and Y. Fu, “Attractive or not?: Beauty prediction with attractiveness-aware encoders and robust late fusion,” in Proc. ACM Int. Conf. Multimedia, 2014, pp. 805–808.

[23] J. Xu, L. Jin, L. Liang, Z. Feng, D. Xie, and H. Mao, “Facial attractiveness prediction using psychologically inspired convolutional neural network (PI-CNN),” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2017, pp. 1657–1661.

[24] L. Chen and W. Deng, “Facial attractiveness prediction by deep adaptive label distribution learning,” in Proc. Chin. Conf. Biometric Recognit., 2019, pp. 198–206.

[25] P. Zhang and Y. Liu, “NAS4FBP: Facial beauty prediction based on neural architecture search,” in Proc. Int. Conf. Artif. Neural Netw., 2022, pp. 225–236.

[26] F. Bougourzi, F. Dornaika, N. Barrena, C. Distante, and A. Taleb-Ahmed, “CNN based facial aesthetics analysis through dynamic robust losses and ensemble regression,” Appl. Intell., vol. 53, no. 9, pp. 10 825–10 842, 2023.

[27] F. Bougourzi, F. Dornaika, and A. Taleb-Ahmed, “Deep learning based face beauty prediction via dynamic robust losses and ensemble regression,” Knowl.-Based Syst., vol. 242, 2022, art. no. 108246.

[28] J. N. Saeed, A. M. Abdulazeez, and D. A. Ibrahim, “Automatic facial aesthetic prediction based on deep learning with loss ensembles,” Appl. Sci., vol. 13, no. 17, 2023, art. no. 9728.

[29] Q. Liu, L. Lin, Z. Shen, and Y. Yu, “FBPFormer: Dynamic convolutional transformer for global-local-contexual facial beauty prediction,” in Proc. Int. Conf. Artif. Neural Netw., 2023, pp. 223–235.

[30] D. E. Boukhari and F. Dornaika, “Enhancing facial beauty prediction with a cross-attention vision transformer and attentionguided augmentation,” Cogn. Comput., vol. 18, no. 1, 2026, art. no. 41.

[31] S. Shi, F. Gao, X. Meng, X. Xu, and J. Zhu, “Improving facial attractiveness prediction via co-attention learning,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2019, pp. 4045–4049.

[32] K. Cao, K.-n. Choi, H. Jung, and L. Duan, “Deep learning for facial beauty prediction,” Information, vol. 11, no. 8, 2020, art. no. 391.

[33] Z. Sun, Z. Xiao, Y. Yu, and L. Lin, “Dynamic attentive convolution for facial beauty prediction,” IEICE Trans. Inf. Syst., vol. E107.D, no. 2, pp. 239–243, 2024.

[34] L. Xu, H. Fan, and J. Xiang, “Hierarchical multi-task network for race, gender and facial attractiveness recognition,” in Proc. IEEE Int. Conf. Image Process., 2019, pp. 3861–3865.

[35] E. Vahdati and C. Y. Suen, “Facial beauty prediction using transfer and multi-task learning techniques,” in Proc. Int. Conf. Pattern Recognit. Artif. Intell., 2020, pp. 441–452.

[36] Y. Li, T. Zhang, and C. L. P. Chen, “Broad Siamese network for facial beauty prediction,” IEEE Trans. Artif. Intell., vol. 5, no. 11, pp. 5786–5800, 2024.

[37] Q. Xiao, Y. Wu, D. Wang, Y.-L. Yang, and X. Jin, “Beauty3DFaceNet: Deep geometry and texture fusion for 3D facial attractiveness prediction,” Computers & Graphics, vol. 98, pp. 11–18, 2021.

[38] Y. Liu, E. Huang, Z. Zhou, K. Wang, and S. Liu, “3D facial attractiveness prediction based on deep feature fusion,” Comput. Animat. Virtual Worlds, vol. 35, no. 1, 2024, art. no. e2203.

[39] S. Liu, E. Huang, Y. Xu, K. Wang, and D. K. Jain, “Computation of facial attractiveness from 3D geometry,” Soft Comput., vol. 26, no. 19, pp. 10 401–10 407, 2022.

[40] Y. Xie, Y. Sun, T. Peng, and D. Zhang, “Crucial and irreplaceable 3D features for facial beauty analysis,” Knowl.-Based Syst., vol. 330, 2025, art. no. 114675.

[41] J. H. Langlois, L. Kalakanis, A. J. Rubenstein, A. Larson, M. Hallam, and M. Smoot, “Maxims or myths of beauty? a metaanalytic and theoretical review,” Psychol. Bull., vol. 126, no. 3, pp. 390–423, 2000.

[42] G. Rhodes, “The evolutionary psychology of facial beauty,” Annu. Rev. Psychol., vol. 57, no. 1, pp. 199–226, 2006.

[43] A. C. Little, B. C. Jones, and L. M. DeBruine, “Facial attractiveness: Evolutionary based research,” Philos. Trans. R. Soc. B, Biol. Sci., vol. 366, no. 1571, pp. 1638–1659, 2011.

[44] P. Marshall, A. Bartolacci, and D. Burke, “Human face tilt is a dynamic social signal that affects perceptions of dimorphism, attractiveness, and dominance,” Evol. Psychol., vol. 18, no. 1, pp. 1–15, 2020.

[45] C. A. M. Sutherland, X. Liu, L. Zhang, Y. Chu, J. A. Oldmeadow, and A. W. Young, “Facial first impressions across culture: Datadriven modeling of Chinese and British perceivers’ unconstrained facial impressions,” Pers. Soc. Psychol. Bull., vol. 44, no. 4, pp. 521–537, 2018.

[46] C. A. M. Sutherland and A. W. Young, “Understanding trait impressions from faces,” Br. J. Psychol., vol. 113, no. 4, pp. 1056–1078, 2022.

[47] J. Zhao, M. Zhang, C. He, X. Xie, and J. Li, “A novel facial attractiveness evaluation system based on face shape, facial structure features and skin,” Cogn. Neurodyn., vol. 14, no. 5, pp. 643–656, 2020.

[48] T. Sano and H. Kawabata, “A computational approach to investigating facial attractiveness factors using geometric morphometric analysis and deep learning,” Sci. Rep., vol. 13, no. 1, 2023, art. no. 19797.

[49] L. Wu, V. Chellappa, and Y. Luximon, “Facial trustworthiness in artificial faces: A systematic review,” Int. J. Hum.-Comput. Interact., vol. 42, no. 4, pp. 2057–2080, 2026.

[50] D. Oh, R. Dotsch, and A. Todorov, “Contributions of shape and reflectance information to social judgments from faces,” Vision Res., vol. 165, pp. 131–142, 2019.

[51] S. Desai and H. G. Ramaswamy, “Ablation-CAM: Visual explanations for deep convolutional network via gradient-free localization,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2020, pp. 972–980.

[52] L. Li, J. Hou, W. Liu, Y. Fang, and J. Yan, “Diffusion-based facial aesthetics enhancement with 3D structure guidance,” IEEE Trans. Image Process., vol. 34, pp. 1879–1894, 2025.

[53] P. W. Koh et al., “Concept bottleneck models,” in Proc. Int. Conf. Mach. Learn., 2020, pp. 5338–5348.

[54] H. Bai, D. Kang, H. Zhang, J. Pan, and L. Bao, “FFHQ-UV: Normalized facial UV-texture dataset for 3D face reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 362–371.

[55] Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, “Accurate 3D face reconstruction with weakly-supervised learning: From single image to image set,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops, 2019, pp. 285–295.

[56] Z. Wang, X. Zhu, T. Zhang, B. Wang, and Z. Lei, “3D face reconstruction with the geometric guidance of facial part segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 1672–1682.

[57] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., 2018, pp. 3942–3951.

[58] Z. Chai et al., “REALY: Rethinking the evaluation of 3D face reconstruction,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 74– 92.

[59] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, “BiSeNet: Bilateral segmentation network for real-time semantic segmentation,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 334–349.

[60] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter, “A 3D face model for pose and illumination invariant face recognition,” in Proc. IEEE Int. Conf. Adv. Video Signal Based Surveillance, 2009, pp. 296–301.

[61] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 6105–6114.

[62] S. Kalayci, H. K. Ekenel, and H. Gunes, “Automatic analysis of facial attractiveness from video,” in Proc. IEEE Int. Conf. Image Process., 2014, pp. 4191–4195.

[63] N. Weng, J. Wang, A. Li, and Y. Wang, “Two-stream temporal convolutional network for dynamic facial attractiveness prediction,” in Proc. Int. Conf. Pattern Recognit., 2021, pp. 10 026– 10 033.

[64] OpenAI, “GPT-5.5 system card,” Apr. 2026, accessed: Sep. 10, 2026. [Online]. Available: https://openai.com/index/ gpt-5-5-system-card/

[65] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, pp. 1–9, 2015.

[66] S. Kullback and R. A. Leibler, “On information and sufficiency,” Ann. Math. Statist., vol. 22, no. 1, pp. 79–86, 1951.

[67] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the Inception architecture for computer vision,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 2818–2826.

[68] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He, “Aggregated´ residual transformations for deep neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 5987–5995.

[69] J. N. Saeed, A. M. Abdulazeez, and D. A. Ibrahim, “FIAC-Net: Facial image attractiveness classification based on light deep convolutional neural network,” in Proc. Int. Conf. Comput. Sci., Eng. Appl., 2022, pp. 1–6.

[70] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn Represent., 2021.

[71] F. Dornaika, K. Wang, I. Arganda-Carreras, A. Elorza, and A. Moujahid, “Toward graph-based semi-supervised face beauty prediction,” Expert Syst. Appl., vol. 142, 2020, art. no. 112990.

[72] F. Dornaika and A. Moujahid, “Multi-view graph fusion for semi-supervised learning: Application to image-based face beauty prediction,” Algorithms, vol. 15, no. 6, 2022, art. no. 207.

[73] S. Liu et al., “Lightweight facial attractiveness prediction using dual label distribution,” IEEE Trans. Cogn. Develop. Syst., vol. 17, no. 4, pp. 953–963, 2025.

[74] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2009, pp. 248–255.

[75] A. Paszke et al., “PyTorch: An imperative style, highperformance deep learning library,” in Adv. Neural Inf. Process Syst., 2019, pp. 1–12.

[76] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent., 2019.

# Supplemental Materials for DisFace3DNet: Explainable Facial Attractiveness Prediction via 3D Component Disentanglement

Fenggui Rao , Yan Luximon , and Jie Zhang

## S1. OVERVIEW

These materials detail implementation, ablation and sensitivity analyses, human evaluation, prediction comparisons, and application workflows. Figure, table, and equation numbering continues from the main paper.

## S2. IMPLEMENTATION AND SUPERVISION

## A. Input Construction and Implementation

The curvature channels in Section III-B of the main paper contain mean and Gaussian curvature and shape index. Three symmetry descriptors compare UV halves; four standardized scale descriptors are [Std(X), Std(Y ), Std(Z), Range(Z)].

Expression displacement $\Delta V$ is embedded, transferred, and scattered onto HiFi3D++ before UV unwrapping:

$$
\Delta U = \mathcal { U } ( { \mathcal { T } } _ { \mathrm { B F M  H i F i } } ( \Delta V ) ) ,\tag{21}
$$

$$
\begin{array} { r l } & { X _ { \mathrm { e x p } } = [ \mathcal { N } ( \Delta U ) , D _ { \mathrm { e x p } } , M _ { \mathrm { e x p } } ] , } \\ & { D _ { \mathrm { e x p } } = 3 \operatorname { t a n h } ( z ( \Delta U ) / 3 ) , } \\ & { M _ { \mathrm { e x p } } = \log ( 1 + 1 0 0 \| \Delta U \| _ { 2 } ) . } \end{array}\tag{22}
$$

Here, $\tau$ transfers BFM displacement to HiFi3D++, U unwraps it into UV coordinates, and $\mathcal { N }$ computes three normal channels from raw displacement. We standardize each displacement channel using fixed cached statistics from 500 samples, shared across folds, then compress large values with the hyperbolic tangent and zero invalid pixels. 3DDFA-V3 uses RetinaFace–ResNet50 [1], [2] and outputs pitch, yaw, and roll in radians. BiSeNet classes 17 and 0 define hair and background. The background vector summarizes RGB means/SDs (six), gradient-magnitude mean/SD (two), area ratio (one), RGB distance from the skin mean (one), HSV means (three), and saturation/value SDs (two).

a) Input normalization and augmentation: Identity positions are standardized over valid pixels, with invalid pixels zeroed. Finite-difference normals from unstandardized positions map from [−1, 1] to [0, 1]. Mean and Gaussian curvature are divided by 5 and 10; shape index keeps its prepared [−1, 1] scale. During the first 70 epochs, skin-albedo brightness and contrast factors are independently uniform on [0.85, 1.15] with probability 0.3. With probability 0.15, one of hair, light, or all skin-detail channels is uniformly selected and zeroed; identity-normal Gaussian noise has SD 0.02 and probability 0.15. Refinement disables augmentation.

b) Training configuration: Each CNN’s 1,280 pooled features are projected to 256 dimensions. Background and pose MLP widths are 15–64–32 and 3– 32–16; the expression readout is 256–64–1. Identity appends its seven descriptors before scoring. Dropout and stochastic-depth probabilities are 0.3. EfficientNet feature blocks 0–5 remain frozen for RGB inputs and 1–5 for adapted inputs. Adapted stems initialize their first three channels at 0.5 times the pretrained weights and cycle those weights at factor 0.1 for extra channels. FiLM scale/offset mappings start at zero.

Training uses an NVIDIA H200 GPU, weight decay 0.01, mixed precision, and gradient-norm clipping at 5.

For LDL, we normalize a discrete Gaussian over 21 equally spaced bin centers on [1, 5], using the rating SD clipped to [0.20, 1.0], then mix it with 2% uniform mass. In Eq. (19), $\lambda _ { \mathrm { K D } } = \lambda _ { \mathrm { a g r e e } } = \lambda _ { \mathrm { b a l } } = 0 . 3 , \lambda _ { \mathrm { s e m } } = 0 . 2 ,$ and $\lambda _ { \mathrm { r a n k } } = 0 . 1 ;$ GT-score and LDL weights are one. Model selection and frozen fusion fitting are described in Sections IV-C and III-G of the main paper.

c) Inference configuration: Inference averages overall and component predictions from original and horizontally flipped inputs. Mirroring reverses horizontal geometry and displacement components and yaw. Demographic Conditioning through FiLM [3] remains active in both views.

## B. Weak Semantic Supervision and Teacher Protocol

In the reported experiments, the following prompts provided auxiliary semantic guidance by jointly generating seven GPT-5.5 integer labels (1–5) per image for all 5,500 images. Shape denotes the identity component (facial shape and structure). For an integer label $v ,$ static, expression, and pose targets map to $( v - 1 ) / 4 , ( v - 3 ) / 2 0$ and $( v - 3 ) / 4 0$ , respectively; Section III-G of the main paper defines the masked loss.

a) System prompt: You are a careful image-factor annotator for a released academic facial-image benchmark.

Your task is NOT to identify the person, infer demographics, or make a personal judgment. Treat the image only as a benchmark sample with an existing aggregate face-score. Your task is to estimate how specific visible factors would push that benchmark face-score up or down in the current image.

Definition: The benchmark face-score is the aggregate visual preference / appeal rating that dataset raters would assign to the face image, based only on visible image evidence.

## Important rules:

1. Evaluate only the requested factor.

2. Treat all other factors as fixed as much as possible.

3. Judge contribution to the benchmark face-score, not objective quality alone.

4. A score of 3 means neutral / ordinary / little clear effect, not bad.

5. Do not reward intensity by itself.

6. Subtle, harmonious, natural effects may score higher than stronger but less flattering ones.

7. Do not let identity, age, gender, ethnicity, clothing, or personal preference dominate the judgment.

8. Use 1 or 5 only when the effect is clearly strong.

b) User prompt: Evaluate the contribution of the following 7 factors to the benchmark face-score of the face in this image. Rate each factor on a 1-5 integer scale.

Unified scoring rule for every factor:

1 = strongly lowers the benchmark face-score

2 = slightly lowers the benchmark face-score

3 = little or no clear effect / neutral / ordinary

4 = slightly raises the benchmark face-score

5 = strongly raises the benchmark face-score

Factor definitions:

## 1. Facial Structure Contribution

Judge how the face geometry itself contributes to the benchmark face-score: facial proportion, symmetry, contour, spacing, and structural harmony of features. Ignore skin texture, hairstyle, expression, lighting, and background.

## 2. Skin Contribution

Judge how the visible skin contributes to the benchmark face-score: clarity, smoothness, evenness, cleanliness, healthy-looking texture/tone. Ignore facial shape, hairstyle, expression, lighting direction, and background.

## 3. Hair Contribution

Judge how the hairstyle, tidiness, framing, volume, and harmony of the hair with the face contribute to the benchmark face-score. Ignore facial shape, skin quality, expression, lighting, and background.

## 4. Expression Contribution

Compared with a relaxed neutral expression of the same person under the same pose, lighting, and background, judge how much the current facial expression lowers or raises the benchmark face-score. Do not reward smile size or smile intensity by itself. A subtle natural smile may score higher than a wide or exaggerated smile. Judge only the contribution of the expression to attractiveness.

## 5. Pose Contribution

Judge how the head angle, tilt, and direction of gaze contribute to the benchmark face-score. Consider whether the angle is flattering, whether the gaze engages the viewer, and whether the head position looks natural and well-composed. A standard frontal pose with no special flattering effect is 3. A slight natural three-quarter angle or subtle tilt that flatters the face can be 4. An extreme angle, looking away, or very awkward tilt is 1-2. Ignore facial structure, skin quality, hairstyle, expression, lighting, and background.

## 6. Lighting Contribution

Judge how the current lighting pushes the benchmark face-score up or down: shadow placement, softness, balance, visibility of features, and whether the lighting flatters the face. Ignore the person’s intrinsic facial structure, skin condition, hairstyle, expression, and background content.

## 7. Background Contribution

Judge how the current background helps or hurts the benchmark face-score: distraction, clutter, color harmony, and whether it supports the face as the visual focus. Ignore the face itself, skin, hair, expression, and lighting quality on the face.

Calibration examples:

\- Example (expression): a slight relaxed smile that flatters the face -> 4

\- Example (expression): a forced or awkward smile that hurts the look -> 2

\- Example (lighting): harsh shadows under the eyes that make the face less flattering -> 2

\- Example (skin): clear, even, healthy-looking skin that supports attractiveness -> 4

\- Example (pose): slight three-quarter turn that shows the jawline well -> 4

Important:

\- Judge contribution to the benchmark face-score, not general image quality.

\- Do not rate the person; annotate visible image factors for research weak supervision.

\- Do not let the overall face-score dominate your answer.

\- Keep the seven factors as independent as possible.

First, give one short observation phrase for each factor. Then output your final scores on the last line in EXACTLY this format: SCORES: [shape],[skin],[hair], [expression],[pose],[light],[background]

![](images/407a80d7c6ed734e0a2c1fd76305893aad305533c884fbce75fd52db8b7457ff.jpg)

![](images/64de43ed284272ea3979c6ed911071d100624c9c71a18155cec717355907605f.jpg)  
Fig. 11. Sensitivity to $\lambda _ { \mathrm { s e m } }$ . Half violins (left) show (a–g) component scores and (h) predicted ratings; white-centered markers and black bars mark medians and interquartile ranges. Curves (right) show (a–g) weak-label Spearman correlations and (h) PCC with mean human ratings. Colors identify seeds; curves connect same-seed results.

c) Teacher: The FPEM [4] teacher follows the student’s official folds. Starting from the public checkpoint, it is fine-tuned separately on each training fold, with checkpoint selection by internal validation. We average its original/flip five-class probabilities to obtain the expected scalar rating used for uniformly weighted score distillation, ordering, and frozen fusion fitting. Benchmark rating statistics define the 21-bin LDL target.

## S3. EXTENDED ABLATIONS AND SENSITIVITY

## A. Statistical Procedures

Five-fold metrics and component diagnostics are reported as equally weighted fold means with sample SDs (denominator four). Weak-supervision diagnostics compare raw scores and within-fold weak-label Spearman correlations on the same held-out images. Ablation differences are DisFace3DNet minus the variant. Their 95% percentile confidence intervals (CIs) use 10,000 paired within-fold image-bootstrap resamples [5], preserving fold sizes and model pairing. Each resample recomputes fold PCC/MAE and differences between five-fold means.

For Table II, 10,000 fold–group-stratified imagebootstrap resamples (seed 20260905) recompute component means and shares while preserving group composition. Both procedures keep the trained models fixed and measure uncertainty from image sampling. Within each fold, static-score scale coefficient of variation (CV) is the population SD of the five held-out component SDs divided by their mean.

## B. Control Definitions

All five-fold controls retain the common settings except for the named change.

a) Component scoring: Without component scoring, one scalar head replaces the seven component readouts and their additive fusion, with 224 fewer parameters $( < ~ 0 . 0 0 1 \% )$ . Inputs, encoders, conditioning, and schedule remain the same. Without dynamic components, the five static routes are retrained. Without component-specific inputs, all image branches receive one UV texture and auxiliary vectors (pose, background, symmetry, and scale) are zeroed; heads, conditioning, and joint training remain.

b) Training objectives: Removing teacher guidance excludes FPEM [4] scalar and ordering targets from learning and its scalar target from frozen fusion fitting. Removing weak semantic supervision sets $\lambda _ { \mathrm { s e m } } = 0 .$ Auxiliary controls remove balance loss, distribution consistency (LDL and scalar–distribution agreement), both ordering losses, or the group embedding separately.

c) Training strategy: Joint learning optimizes all seven routes throughout both learning-rate periods. The compute-matched staged control learns static routes, then jointly trains expression and pose, and finally refines all routes. Both end with the same frozen fusion fitting.

## C. Prediction Error and Component-Score Diagnostics

a) Component scoring: Predicting through seven component scores increases mean MAE by 0.0108 rating points (95% CI: 0.0078 to 0.0138) relative to the capacity-matched scalar control, quantifying the additional error associated with the additive scoring structure.

b) Weak semantic supervision: Without weak supervision, expression weak-label correlation falls from $0 . 1 1 6 \pm 0 . 0 6 7 ~ { \mathrm { t o } } ~ - 0 . 1 8 8 \pm 0 . 1 0 4$ , and pose correlation from $0 . 2 1 2 \pm 0 . 0 3 5 ~ \mathrm { t o } ~ - 0 . 0 2 2 \pm 0 . 0 6 0$ . Expression-score SD changes from 0.0190±0.0016 to 0.0295±0.0301, indicating greater fold-to-fold variation in scale; mean pose score shifts from $0 . 0 0 0 7 \pm 0 . 0 0 0 5$ to $0 . 0 0 8 2 \pm 0 . 0 0 5 1$

c) Training strategy: Compared with joint learning, staged training reduces expression-score SD from $0 . 0 1 9 0 \pm 0 . 0 0 1 6 \ \mathrm { { \ t o } \ 0 . 0 0 2 3 \pm 0 . 0 0 0 8 }$ and the fitted dynamic coefficient from $0 . 0 2 4 2 \pm 0 . 0 0 5 0 \ \mathrm { t o } \ 0 . 0 0 3 4 \pm$ 0.0014 across held-out folds. The staged model therefore uses a narrower range of expression scores and gives dynamic variation less influence on the final rating.

d) Teacher guidance: Adding FPEM guidance changes mean MAE by 0.0006 points (95% CI −0.0017– 0.0028) and raises within-fold weak-label Spearman correlation from $0 . 3 2 9 \pm 0 . 1 9 8$ to $0 . 4 5 0 \pm 0 . 1 0 4$ for background, while lowering it from $0 . 2 3 3 \pm 0 . 0 8 2$ to $0 . 1 1 6 \pm 0 . 0 6 7$ for expression and from $0 . 2 6 4 \pm 0 . 0 4 6$ to $0 . 2 1 2 \pm 0 . 0 3 5$ for pose.

## D. Semantic-Weight Sensitivity

1) Evaluation Design: Five semantic coefficients are evaluated with five seeds (20260801–20260805), giving

25 model fits on fold 0 inner validation. Within each seed, coefficients share a 3,960/440 partition of the 4,400 official training images. Seeds vary the partition and training randomness; the 1,100-image test fold is unused. In Fig. 11, Scott-bandwidth half violins have equal maximum widths; axes are fixed within components without per-run standardization.

2) Results: $\mathbf { A s } \lambda _ { \mathrm { s e m } }$ increases from zero to 0.2, weaklabel Spearman correlation improves for all seven components in all five seeds. Averaged over seeds and components, static correlation rises from 0.430 to 0.564, and expression/pose correlation from −0.077 to 0.175. All five static-score SDs and interquartile ranges decrease in every seed.

From 0.1 to 0.2, mean weak-label Spearman correlation rises for every component, while PCC falls from $0 . 8 8 7 3 { \scriptstyle \pm 0 . 0 0 5 7 }$ to 0.8863±0.0064 (Fig. 11(h); mean ± sample SD across five seeds). At both 0.5 and 1.0, skin, hair, and expression correlations exceed their 0.2 values in every seed, whereas mean light correlation falls from 0.564 at 0.2 to 0.492 at 1.0. Mean PCC decreases to $0 . 8 8 4 0 \pm 0 . 0 0 6 9$ at 0.5 and $0 . 8 7 5 8 \pm 0 . 0 0 5 3$ at 1.0.

## E. Auxiliary-Control Results

Table IV shows that balance loss reduces scale CV, making static-branch variability more comparable. Removing distribution consistency worsens both held-out distribution diagnostics in every fold. Without pairwise ordering, SRCC changes from $0 . 8 7 6 6 \pm 0 . 0 0 6 0$ to $0 . 8 7 5 6 \pm 0 . 0 0 6 7 ;$ the unrounded GT-concordance difference is below 0.02 percentage points. The paired PCC intervals for all four auxiliary controls include zero.

Changing only the supplied AM/AF/CM/CF token in the fixed model produces a mean rating range of 0.074 points (95% CI 0.060–0.086).

## S4. HUMAN EVALUATION

## A. Evaluation Design

Twenty consenting adults (10 male, 10 female) completed 2,060 responses under The Hong Kong Polytechnic University IRB approval HSEARS20240313013. Each completed 103 trials: four practice, 80 non-repeat component, five component repeats, ten separate overallattractiveness, two identical-image attention, and two instruction checks.

Images were sampled uniformly from SCUT-FBP5500 without group, fold, or target-gap restrictions; left– right order was randomized. Participants compared facial shape (identity), skin, hair, or expression, with an unsure option. The post-training comparisons evaluate whether component-score orderings agree with participants’ choices. The Chinese/English interface requested a laptop/desktop display and loaded both images before allowing a response.

TABLE IV  
AUXILIARY-DESIGN ABLATIONS. INTERVALS FOLLOW SECTION S3.A. PAIRED DIAGNOSTIC VALUES LIST THE VARIANT BEFORE DISFACE3DNET
<table><tr><td>Variant</td><td>PCC↑</td><td>∆PCC [95% CI]</td><td>Matched diagnostic</td></tr><tr><td>Without balance loss</td><td></td><td>0.8909 ± 0.0069 -0.0005 [-0.0028, 0.0017]</td><td> $\mathrm { S t a t i c - s c o r e ~ s c a l e ~ C V : ~ 0 . 3 7 7 \pm 0 . 0 2 9 ~ v s . }$   $0 . 0 8 2 \pm 0 . 0 1 1$ </td></tr><tr><td>Without distribution consistency</td><td></td><td>0.8893 ± 0.0056 0.0011 [-0.0016, 0.0037]</td><td> $\mathrm { K L ~ d i v e r g e n c e } \colon 0 . 5 0 8 \pm 0 . 0 0 2 ~ \mathrm { v s } .$   $0 . 1 0 0 \pm 0 . 0 0 5 ; \mathrm { s c o r e - d i s t r i b u t i o n ~ M A E } .$ </td></tr><tr><td>Without pairwise ordering</td><td></td><td>0.8899 ± 0.0055 0.0005 [-0.0015, 0.0024]</td><td> $0 . 4 6 4 \pm 0 . 0 1 5 ~ \mathrm { v s . ~ 0 . 0 4 6 \pm 0 . 0 0 2 ~ p o i n t s }$  GT-pair concordance:  $( 9 6 . 7 9 \pm 0 . { \bar { 3 5 } } ) \%$  vS.  $( 9 6 . \dot { 8 } 1 \pm 0 . 3 2 ) \%$ </td></tr><tr><td>Without group conditioning</td><td> $0 . 8 8 8 5 \pm 0 . 0 0 4 2$ </td><td>0.0019 [-0.0003, 0.0040]</td><td>Group embedding removed</td></tr><tr><td>DisFace3DNet (Ours)</td><td> $0 . 8 9 0 4 \pm 0 . 0 0 6 3 - $ </td><td></td><td>Reference for all comparisons</td></tr></table>

Informal pilot feedback guided component selection. Participants found it difficult to isolate lighting’s contribution, discern pose differences in the candidate pairs, or apply a clear attractiveness criterion to background differences. The formal study therefore focused on identity, skin, hair, and expression.

Instructions restricted attention to the named attribute, excluded overall attractiveness, and offered “No clear difference / unsure”. Shape trials excluded skin, hair, and makeup; skin trials excluded shape and hair; hair trials excluded the face. Expression trials asked: “Whose expression appears to increase that face’s attractiveness more compared with a neutral expression?” Identicalimage checks instructed unsure, and instruction checks requested Left. Figure 9(a) in the main paper illustrates a skin trial.

## B. Statistical Analysis

Each component contributes 400 non-repeat responses; the primary agreement estimates exclude unsure responses. For this analysis, held-out component scores are standardized by their official-fold means and sample SDs; model direction follows the higher standardized score. No evaluated pair has tied model scores. Human– model agreement is the fraction of decisive responses following this direction. For between-human agreement, each decisive response is compared with all other participants’ decisive responses to the same component and image pair, aligning the chosen image rather than its screen position. We average these per-response agreement fractions, weighting responses equally in both estimates. A response without another decisive response to that pair is omitted only from the between-human estimate.

Both estimates use 95% percentile intervals from 100,000 participant-clustered bootstrap draws [5] (seed 20260724). Each draw samples 20 participants with replacement, retains their response clusters, and recomputes both estimates. To compare different people, between-human agreement uses responses from other original participants, even when a draw samples the same participant more than once. A response without such a peer on the same pair is omitted from that draw’s between-human estimate. For reference, response-level Wilson intervals [6] for human–model agreement are 68.2%–77.7%, 67.7%–77.8%, 60.2%– 70.3%, and 36.3%–47.5% for identity, skin, hair, and expression, respectively.

Secondary analyses of identity, skin, and hair examine pairs whose component-score and benchmark-rating directions conflict (excluding equal ratings), count unsure as non-agreement, and compare human majority choices with model directions. Majority analysis excludes unsure and tied votes; each component-specific pair counts once, including unanimous choices.

## C. Agreement Counts and Secondary Analyses

a) Response counts: All participants passed the attention and instruction checks. Figure 9(b) reports the four components separately. Human–model agreeing/decisive counts for identity, skin, hair, and expression are 241/329, 214/293, 221/338, and 123/294. Betweenhuman estimates use 329, 291, 338, and 292 responses across 80, 78, 80, and 78 image pairs, respectively: two skin and two expression responses lack another decisive response. There are 1,066, 852, 1,144, and 856 responseto-response comparisons, counting both reference directions; the reported rates average per-response fractions rather than divide aggregate comparison counts.

b) Pooled static-component agreement: Pooling responses for identity, skin, and hair gives 676/960 (70.4%; participant-clustered 95% CI: 67.1%–73.7%).

c) Conflicting directions, unsure responses, and majority votes: For the same three components, participants follow the component score in 205/334 decisive conflicting-direction responses (61.4%). Counting the 240 unsure responses for identity, skin, and hair as nonagreement gives 56.3% (676/1,200). Human majority choices match the model for 178/214 non-tied pairs from these three components (83.2%).

## S5. PREDICTION COMPARISONS WITH FPEM

## A. Mean-Prediction Comparison Protocol

For Fig. 4, 1,000 held-out images were sampled uniformly without replacement (AM/AF/CM/CF: 370/351/138/141). Two-sided paired t-tests compare DisFace3DNet and FPEM predictions; subgroup tests use Holm correction at $\alpha = 0 . 0 5$

## B. Overall and Group Mean Differences

On the sampled images, mean predictions are 2.927 (DisFace3DNet) and 2.940 (FPEM); the overall paired test gives $t ( 9 9 9 ) ~ = ~ - 1 . 6 1 9$ and $\begin{array} { l l l } { p } & { = } & { 0 . 1 0 6 } \end{array}$ . The AM/AF/CM/CF mean offsets (DisFace3DNet minus FPEM) are $+ 0 . 0 1 1 / - 0 . 0 2 7 / - 0 . 0 4 0 / - 0 . 0 1 2$ points.

## C. Image-Level Prediction Errors

The 16 held-out examples in Fig. 5 of the main paper show both shared and model-specific prediction errors. Both models overestimate the first CF and third AF images and reverse the second/third CM ordering. Only DisFace3DNet reverses the first two AF images; FPEM preserves their order but overestimates the first. Several DisFace3DNet estimates closely match human ratings: using unrounded values, absolute errors are 0.06 (second AF), 0.06 (second AM), 0.15 (fourth CM), and 0.04 (fourth AM) points.

## S6. APPLICATIONS

a) Digital-Human and Robot Face Design: A designer could select geometry references by identity score, then compare skin and hair with geometry fixed (Fig. 10(a)). This separates shape and appearance choices.

b) Beauty-Filter Assessment: For presets on the same image, users could compare skin, light, and identity scores alongside the overall rating when selecting surface treatments; Fig. 10(b) illustrates this comparison.

c) Hairstyle Design and Virtual $T r y – O n .$ Hair scores could shortlist hairstyle references for transfer to the same face (Fig. 10(c)). Try-on images and user preferences would then guide salon or wig selection.

## REFERENCES

[1] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “RetinaFace: Single-shot multi-level face localisation in the wild,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 5202–5211.

[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[3] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., 2018, pp. 3942–3951.

[4] H. Li et al., “FPEM: Face prior enhanced facial attractiveness prediction for live videos with face retouching,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 11 458–11 468.

[5] B. Efron, “Bootstrap methods: Another look at the jackknife,” Ann. Statist., vol. 7, no. 1, pp. 1–26, 1979.

[6] E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” J. Amer. Statist. Assoc., vol. 22, no. 158, pp. 209–212, 1927.