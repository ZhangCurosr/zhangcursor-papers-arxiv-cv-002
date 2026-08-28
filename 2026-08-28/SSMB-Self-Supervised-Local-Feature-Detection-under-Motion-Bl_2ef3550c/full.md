# SSMB: Self-Supervised Local Feature Detection under Motion Blur

Zhenjun Zhao, Fabio Bellavia, Wenting Wang, Fan Zhu, Jiajun Wu, Suryansh Kumar, Mingqiang Wei, Haoang Li, Javier Civera

Abstract—Keypoint detection under motion blur remains a significant challenge, as blur distorts local image structure and degrades the repeatability of feature localization. Existing approaches either rely on computationally expensive deblur-thendetect pipelines that may introduce restoration artifacts, or learn to regress the image positions of handcrafted keypoints extracted on sharp images, which reflects the assumptions of the handcrafted detector rather than what is truly repeatable under blur. We present SSMB, a deblur-free, self-supervised keypoint detector for motion-blurred images that requires neither handcrafted detectors nor external pseudo-labels. SSMB introduces the Local Discriminability Enhancement (LDE) module, which restores fine-grained local discriminability after global feature mixing. Training is performed in two stages. First, geometric pretraining on synthetic shapes bootstraps spatially discriminative keypoint detection without any external detector, just from the rendered geometry. Second, blur-aware training on real sharp-blur image pairs learns blur-invariant detection through a multi-component self-supervised objective that enforces cross-domain consistency, geometric alignment, and spatial coverage. Extensive evaluations on keypoint detection, image matching, relative pose estimation, and visual localization under motion blur demonstrate that SSMB establishes a new state-of-the-art among sparse keypoint detectors, consistently outperforming both supervised and selfsupervised baselines across all tasks. Code, models, and datasets will be publicly available upon paper acceptance.

Index Terms—Local feature detection, motion blur, selfsupervised learning, keypoint detection.

## I. INTRODUCTION

Accurately detecting salient keypoints across images is a fundamental component in many computer vision applications, including Simultaneous Localization and Mapping (SLAM), Structure-from-Motion (SfM), camera calibration, image retrieval, and visual localization [1]–[12]. An effective keypoint detector should produce features that are well distributed across the image, highly repeatable across viewpoints and time, and robust to photometric and geometric variation. While remarkable progress has been made on sharp images [13]–[23], motion blur remains a major and largely overlooked challenge. Arising from camera shake, object motion, or long exposure times in low-light conditions, motion blur smears local image structure across spatial regions. This severely degrades the repeatability of keypoint detection, compromising the feature correspondences required by downstream vision tasks.

A natural approach to addressing motion blur is the deblurthen-detect pipeline, which first restores a sharp image with a deblurring network and then applies any existing feature detector. However, deblurring networks [24]–[26] are computationally demanding and rarely run in real time. Moreover, their restored images often contain artifacts, particularly under severe blur, which propagate into the detection stage and degrade the reliability of the extracted features.

In contrast, a deblur-free detector that operates directly on blurred images avoids the computational overhead of image restoration while eliminating artifact propagation, providing a more efficient and robust solution. In this paper, we focus on this deblur-free paradigm. A recent step in this direction is BALF [27], which introduces a Multi-Layer-Perceptron (MLP)-based detector trained directly on blurred images with real-time inference capabilities. However, BALF is trained to regress the SIFT [13] keypoints detected on the corresponding sharp images, which fundamentally limits its ability to learn general blur-invariant features. Since SIFT is handcrafted for sharp images rather than motion-blurred ones, the learned detector inevitably reproduces its detection preferences instead of discovering the structures that are genuinely repeatable under blur. More generally, supervising with pseudo-labels, whether from SIFT or any other handcrafted detector, constrains the learned representation to imitate the behavior of the chosen detector rather than directly learning the cues that characterize repeatable keypoints in blurred images.

In this paper, we propose SSMB, a deblur-free keypoint detector for motion-blurred images that requires neither handcrafted detectors nor external pseudo-labels. Since SSMB must learn on its own which local structures are relevant, without any external signal to guide it, we introduce a novel Local Discriminability Enhancement (LDE) module, which explicitly preserves and reinforces these local cues throughout the network, enabling effective self-supervised learning in the presence of blur. Additionally, we propose a two-stage training pipeline. At the first stage, that we denote as geometric pretraining, SSMB learns to detect spatially discriminative keypoints from synthetic images of simple geometric forms, in which reliable corner labels are derived directly from the rendered geometry without any external detector. At the second stage, that we denote as blur-aware training, SSMB is trained on real sharp-blur pairs from the GoPro dataset [24] using a multi-component self-supervised objective. This objective aligns detections across the sharp and blurred domains, anchors them to geometrically consistent locations through online homographic adaptation [14], and promotes a welldistributed set of keypoints across the image, preventing the network from collapsing to a small set of dominant responses. In our experiments we observed that, without this spatial spreading term, self-supervised training degenerates into a trivial solution in which detection concentrates on a handful of salient locations, leaving most of the image without any meaningful response. SSMB reliably detects keypoints on a real-world blurred image and achieves the highest image matching accuracy among sparse methods under motion blur, as illustrated in Fig. 1.

![](images/587008480c0925351107beae1699da0666ba29939bbafa912edd32fcc937e274.jpg)  
Input

![](images/151643462a7e9807b247b31465db83cce2b274768ad260d32ae1d425c817e8a8.jpg)  
SSMB (Ours)

![](images/731e486795b62d74126b061bca938b170c4dc5eef9a091cd1d8ac74417251639.jpg)  
Fig. 1. SSMB under motion blur. Left: a real-world blurred image from RealBlur [28] (top) and the keypoints (green circles) detected by SSMB on it (bottom), without any deblurring preprocessing. Right: image matching accuracy (Mean Matching Accuracy (MMA) [16] at a 3-pixel threshold) on the Blur-HPatches TOUGH split [27] under motion blur. SSMB, our self-supervised sparse detector trained without handcrafted or external supervision, substantiall outperforms all sparse baselines.

In summary, our main contributions are as follows:

• We introduce the Local Discriminability Enhancement (LDE) module, which restores fine-grained local discriminability after global feature mixing, a necessary condition for self-supervised training to succeed under blur.

• We propose a two-stage training pipeline with geometric pretraining on synthetic data followed by blur-aware training on real sharp-blur pairs, and show that geometric pretraining is an essential prerequisite for effectively bootstrapping self-supervised learning under blur.

• We design a multi-component self-supervised loss combining homographic adaptation, blur consistency, position consistency, and spatial diversity terms, and identify spatial diversity as the key term that prevents the network from collapsing to a degenerate solution.

Extensive evaluations on keypoint detection, image matching, relative pose estimation, and visual localization under motion blur demonstrate that SSMB establishes a new state of the art among sparse keypoint detectors, consistently outperforming both supervised and self-supervised sparse baselines, and even surpassing detector-free matching methods at low pixel thresholds.

## II. RELATED WORK

## A. Hand-crafted Local Feature Detection

Hand-crafted keypoint detectors identify salient image structures using operators that were manually designed to capture certain local image statistics. The Harris corner detector [29] identifies corners from first-order image derivatives, while the Hessian detector [30] detects blob-like structures from secondorder derivatives. SIFT [13] became the most widely adopted due to its robustness to scale and rotation, and it has long served as a source of pseudo-labels for supervised learningbased keypoint detectors.

## B. Learning-Based Local Feature Detection

Learning-based methods overcome the limitations of handcrafted operators by training neural networks to detect repeatable keypoints directly from data. SuperPoint [14] introduces a self-supervised framework that bootstraps keypoint labels from synthetic geometry via homographic adaptation on real images, producing a joint detector and descriptor without manual annotation. Subsequent methods explored increasingly powerful network architectures and learning objectives. Key.Net [15] combines hand-crafted filters with learned Convolutional Neural Network (CNN) features for multi-scale detection. D2-Net [16] presents a single CNN that jointly detects and describes features by identifying local maxima across both spatial and channel dimensions of deep feature maps. R2D2 [17] simultaneously predicts keypoint locations and descriptors using dilated convolutions, training the detector to be both repeatable and reliable. DISK [18] trains an endto-end detector and descriptor pipeline using reinforcement learning rewards derived from matching performance, bypassing the non-differentiability of sparse keypoint selection.

REKD [31] proposes a self-supervised rotation-equivariant detector. NeSS-ST [19] combines hand-crafted keypoints with a learned neural stability score to select high-quality feature points. ALIKED [20] proposes sparse deformable descriptors that learn geometrically adaptive supporting features for each keypoint, emphasizing precise keypoint localization accuracy. DeDoDe v2 [23] decouples detection from description for improved pose estimation. XFeat [32] prioritizes real-time efficiency with a lightweight architecture suitable for resourcelimited devices.

## C. Detector-Free Matching Methods

An alternative to sparse keypoint detection is dense or semidense feature matching, where correspondences are established directly between image pairs without explicit keypoint detection. LoFTR [33] pioneered this paradigm by leveraging a transformer architecture [34] to compute dense correlation maps between image patches, enabling semi-dense matching even in weakly textured regions, where sparse detectors typically fail. Subsequent methods, such as MatchFormer [35] and ASpanFormer [36], improve upon LoFTR with interleaved attention and adaptive span mechanisms, respectively. More recently, RoMa [37] combines frozen DINOv2 [38] features with CNN features and a transformer-based matching decoder to achieve dense correspondences, while RoMa v2 [39] improves both the matching architecture and training distribution to establish the current state of the art in dense matching. Despite their impressive performance, detector-free methods have important practical limitations. Since correspondences are computed jointly for each image pair, they cannot produce reusable keypoints or descriptors, making them poorly suited to retrieval-based visual localization pipelines [11], which are designed around pre-computed, reusable keypoint maps of the environment and become prohibitively expensive when every image pair must be matched from scratch. Furthermore, their inference cost is also substantially higher than that of sparse detectors, as they require processing the full image pair at inference time. Finally, although these methods significantly advanced the state of the art on standard benchmarks, neither sparse nor detector-free approaches are designed nor trained to handle motion blur, which distorts the local image structure on which both paradigms rely.

## D. Keypoint Detection on Motion-Blurred Images

Image deblurring. A common strategy for handling motion blur is to first restore a sharp image using a deblurring network and then apply an existing keypoint detector. Early deep learning approaches estimate the blur kernels from the degraded image and restore the blur-less image through learned deconvolution [40]. More recent methods address deblurring in an end-to-end manner, formulating it as an image-toimage translation problem. Nah et al. [24] introduce a multiscale CNN with a coarse-to-fine strategy for blind deblurring. Kupyn et al. propose DeblurGAN [41] and its follow-up DeblurGAN-v2 [25], employing adversarial training to improve perceptual quality. SRN-DeblurNet [26] uses a scalerecurrent architecture for progressive multi-scale restoration.

While these methods produce visually plausible results in most cases, they introduce restoration artifacts, in particular under severe blur, and require significant computational resources that preclude real-time operation.

Deblur-free detection. Rather than restoring the image before detection, a more principled approach is to design a detector that operates directly on blurred images, without any intermediate restoration. To the best of our knowledge, BALF [27] is the first method to explicitly follow this paradigm, proposing a pure MLP-based architecture trained on sharp-blur image pairs with SIFT keypoints on sharp images as supervision. Although BALF achieves real-time inference and strong performance on blurred benchmarks, its supervision is fundamentally tied to a hand-crafted detector. Consequently, its learned representation is constrained to replicate SIFT’s detections rather than discovering structures that are genuinely repeatable under blur. In contrast, our SSMB removes the need for external supervision entirely by learning directly from geometric and cross-domain consistency constraints in a self-supervised framework.

## III. METHODOLOGY

## A. Overview

We propose SSMB, a deblur-free, self-supervised keypoint detector specifically designed for motion-blurred images. In contrast to BALF [27], which learns to regress SIFT keypoints extracted from sharp images, SSMB is trained without any handcrafted or external pseudo-labels, eliminating the SIFT dependency that limits generalization. Without an external signal such as SIFT to guide it, however, SSMB must learn on its own which local structures are worth attending to. To this end, we introduce a novel Local Discriminability Enhancement (LDE) module that explicitly preserves and reinforces these local cues throughout the network, making self-supervised learning under blur effective in the first place.

Fig. 2 illustrates the network architecture. SSMB consists of two main components: (i) an MLP-based encoder with an LDE module inserted in each of its four blocks, and (ii) a detector head that produces a dense keypoint probability map and sub-pixel localization offsets.

Training is performed in two self-supervised stages, illustrated in Fig. 4. In the first stage (geometric pretraining), the model learns to detect spatially discriminative keypoints from synthetic geometric shapes, where reliable corner annotations are derived directly from the rendered geometry. In the second stage (blur-aware training), the model leverages real sharpblur image pairs from the GoPro dataset [24] and a multicomponent self-supervised loss that enforces cross-domain detection consistency and spatial spreading over the whole image space.

## B. Network Architecture

MLP-based encoder. The encoder adopts the multi-axis gated MLP design used in BALF [27], which is built upon the MAXIM architecture [42]. It consists of four cascaded blocks. Each block contains a linear projection layer, a ResidualSplitHeadMultiAxisGmlpLayer (RSMA), a ResidualChannelAttentionBlock (RCAB), and a max-pooling downsampler (disabled in the last block). The encoder progressively reduces spatial resolution from $H \times W$ to $\frac { H } { 8 } \times \frac { W } { 8 }$ while expanding channel dimensions from 3 to {32, 64, 128, 256}. The RSMA layer splits the feature map into two branches processed in parallel: a GridGmlpLayer for global spatial mixing via dilated grid partitioning, and a BlockGmlpLayer for local spatial mixing via dense block partitioning [42]. This dual-path design captures both long-range structural context and local texture patterns, which is particularly important for motionblurred inputs where pixel intensities are spread across neighborhoods. Intuitively, grid partitioning subsamples spatially distant tokens into each group, giving each token a coarse, image-wide receptive field at low computational cost, while block partitioning groups spatially adjacent tokens, preserving fine local detail. We refer readers to the original MAXIM paper [42] for a complete architectural specification.

![](images/c4a47aba0a5c8263be5f2d7b6bf7ab846e4a6cc84c8c72d1da70190f311d1aa4.jpg)  
Fig. 2. Network architecture of SSMB. Given an input image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , the encoder progressively downsamples the feature map through four cascaded blocks, from $H \times W$ to $\textstyle { \frac { H } { 8 } } \times { \frac { W } { 8 } }$ , while expanding the channel dimension. Each block contains an RSMA layer, within which the Local Discriminability Enhancement (LDE) module is inserted between its two internal branches, and a subsequent RCAB layer. The detector head then takes the encoder output and branches into two independent towers: a probability tower (two convolutions followed by a Softmax) that produces a keypoint probability map $\mathbf { P } \in \mathbb { R } ^ { H \times W }$ and an offset tower (two convolutions followed by a Sigmoid) that produces sub-pixel position offsets δ $\equiv \stackrel { \bullet } { \mathbb { R } } ^ { 2 } \times H _ { c } \times W _ { c } \enspace \breve { ( H _ { c } = H / 8 , W _ { c } = W / 8 ) }$ . Keypoints are localized at the cells where P peaks, refined by δ.

Local Discriminability Enhancement (LDE). Global MLP mixing improves contextual understanding but can dilute the fine-grained local discriminability that precise keypoint localization depends on. This risk is particularly acute for SSMB, since it is trained without any external signal to anchor the network toward geometrically meaningful locations. To compensate for that, we introduce the LDE module, inserted within the RSMA layer immediately after the GridGmlpLayer and before the BlockGmlpLayer. LDE operates on the intermediate feature tensor $\mathbf { u } \in \dot { \mathbb { R } ^ { B \times H \times W \times C } }$ (the global-mixing branch output). It first applies layer normalization, then extracts local gradient features via a depthwise convolution, gated by a learned, feature-dependent channel attention, as illustrated in Fig. 3:

$$
\mathrm { L D E } ( \mathbf { u } ) = \mathbf { u } + \mathrm { L N } ( \mathbf { u } ) + f _ { \mathrm { l o c a l } } ( \mathrm { L N } ( \mathbf { u } ) ) \odot g _ { \mathrm { b l u r } } ( \mathrm { L N } ( \mathbf { u } ) ) ,\tag{1}
$$

where LN(·) denotes layer normalization, $f _ { \mathrm { l o c a l } } ( \cdot )$ is a depthwise convolution capturing local spatial patterns, and $g _ { \mathrm { b l u r } } ( \cdot )$ is a two-layer pointwise convolution bottleneck that adaptively weights the local enhancement based on channel statistics. Both operations are applied in the spatial format via permutation, with a residual connection preserving the input features. This design introduces a negligible parameter overhead (≈ 50K) while meaningfully improving keypoint localization under blur. We visually confirm the necessity of this design in Sec. IV-I, where removing LDE causes the predicted probability map to collapse toward near-zero almost everywhere.

![](images/6400c7388e7fc82f7c64342bd459da22e79eb127f62ea374a3395dc68667889c.jpg)  
Fig. 3. Architecture of the LDE module. The input feature u is first normalised by Layer Norm to produce $\mathbf { x } = \mathrm { L N } ( \mathbf { u } )$ . Two parallel branches then operate on x: a depthwise 3×3 convolution $f _ { \mathrm { l o c a l } }$ that captures local gradient structure, and a lightweight, feature-dependent gating branch $g _ { \mathrm { b l u r } }$ consisting of a two-layer pointwise convolution bottleneck with a ReLU in between, followed by a Sigmoid activation. The branch outputs are fused by element-wise multiplication (⊙) and added to x via an inner residual, and then to the original input u via an outer residual, yielding $\mathrm { L D E } ( \mathbf { u } ) = \mathbf { u } + \mathrm { L N } ( \mathbf { u } ) + f _ { \mathrm { l o c a l } } ( \mathrm { L N } ( \mathbf { \bar { u } } ) ) \odot g _ { \mathrm { b l u r } } ( \mathrm { L N } ( \mathbf { u } ) )$

Detector head. Following [14], [27], the detector head takes the encoder output $\mathcal { F } \in \mathsf { \bar { R } } ^ { B \times C \times H _ { c } \times W _ { c } }$ <sup>c</sup> (where $H _ { c } = H / 8$ $W _ { c } = W / 8 )$ and splits into two independent branches. The probability branch produces the keypoint probability map $\mathbf { \bar { P } } ~ \in ~ \mathbb { R } ^ { \bar { B } \times H \times W }$ : two convolutional layers followed by a Softmax yield logits $\mathbf { L } \in \mathbb { R } ^ { B \times 6 5 \times H _ { c } \times \dot { W _ { c } } }$ over $8 \times 8 = 6 4$ pixel positions per cell plus one dustbin channel, which are then converted to a full-resolution probability map via pixelshuffle after discarding the dustbin channel. The offset branch produces sub-pixel position offsets $\delta \in \mathbb { R } ^ { B \times 2 \times H _ { c } \times W _ { c } }$ : two convolutional layers followed by a Sigmoid predict the $( x , y )$ offset within each cell. Keypoints are localized at the cells where P peaks, refined by δ.

![](images/df535a0d42ff63b0f9e577f6aa093326eb3ae0a99945ac6c231f48c7c2b1202f.jpg)  
Fig. 4. Two-stage self-supervised training pipeline of SSMB. (a) Stage 1 (geometric pretraining): SSMB is pretrained on synthetic geometric shapes with corner ground truth rendered directly from the synthetic images (no handcrafted or learned detector involved), supervised by the cross-entropy loss ${ \mathcal { L } } _ { \mathrm { s y n } } .$ (b) Stage 2 (blur-aware training): a single SSMB network with shared weights, initialized from the Stage 1 checkpoint, processes sharp $I ^ { s }$ and blurred $\dot { I } ^ { b }$ GoPro image pairs. The sharp-image branch is supervised by the homographic-adaptation loss $\mathcal { L } _ { \mathrm { { h a } } } .$ The blur-image branch is supervised by the blur consistency loss ${ \mathcal { L } } _ { \mathrm { b l u r } } ,$ the position consistency loss ${ \mathcal { L } } _ { \mathrm { p o s } }$ , and the spatial diversity loss ${ \mathcal { L } } _ { \mathrm { d i v } }$ . All four terms are combined into the total training loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { i } \lambda _ { i } \mathcal { L } _ { i } ^ { \cdot } } \end{array}$ $i \in \{ \mathrm { h a , b l u r , p o s , d i v } \}$

## C. Self-Supervised Training

SSMB is trained entirely without manual keypoint annotations, in the two self-supervised stages shown in Fig. 4. We describe each stage and the loss functions used in each.

1) Geometric Pretraining: This first stage initializes the detector with meaningful spatial discriminability before any blur-specific training. Without this initialization, we observe score map collapse: all spatial locations receive nearly identical scores (Pearson correlation ≈ 1.0 across different images), making subsequent blur-aware training ineffective.

Synthetic pretraining. We generate synthetic images onthe-fly by randomly composing geometric primitives (checkerboards, line segments, polygons, ellipses, and star patterns) on grayscale backgrounds, inspired by the synthetic pre-training paradigm of SuperPoint [14]. Since the geometry is fully controlled, reliable corner labels can be derived directly from the rendered images without requiring any external supervision or handcrafted annotations. For each synthetic image, we produce a label map $\mathbf { Y } \in \{ 0 , \dots , 6 4 \} ^ { H _ { c } \times \dot { W } _ { c } }$ : each 8×8 cell is assigned the index of the corner-occupied pixel, or the dustbin class (64) if no corner is present. The network is trained with a masked cross-entropy (CE) loss:

$$
\mathcal { L } _ { \mathrm { s y n } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( i , j ) \in \mathcal { V } } \mathrm { C E } \big ( \mathbf { L } _ { i j } , \mathbf { Y } _ { i j } \big ) ,\tag{2}
$$

where V denotes the set of valid (non-border) cells and $\mathbf { L } _ { i j }$ are the model logits for cell (i, j).

2) Blur-Aware Training: In this second stage, we train the model on real sharp-blur image pairs from the GoPro dataset [24]. A single SSMB network performs two forward passes per training iteration, one on the sharp image $I ^ { s }$ and one on its paired blurred image $I ^ { b } ,$ , sharing all weights. The sharp forward pass serves a dual role. It generates the homographic adaptation pseudo-label target and provides the consistency supervision signal for the blur forward pass. The total training loss combines four terms:

$$
\mathcal { L } = \lambda _ { \mathrm { h a } } \mathcal { L } _ { \mathrm { h a } } + \lambda _ { \mathrm { b l u r } } \mathcal { L } _ { \mathrm { b l u r } } + \lambda _ { \mathrm { p o s } } \mathcal { L } _ { \mathrm { p o s } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } .\tag{3}
$$

Homographic adaptation loss $\mathcal { L } _ { \mathbf { h a } } .$ For each sharp image in the batch, we perform online Homographic Adaptation (HA) [14]. Specifically, we generate $N _ { H }$ random homographies $\{ \mathbf { H } _ { k } \} _ { k = 1 } ^ { N _ { H } }$ , warp the sharp image, and run the current detector on each warped image to obtain a probability map $P _ { k }$ . We then aggregate these warped probability maps back to the original image coordinates via:

$$
\hat { P } ( i , j ) = \frac { \sum _ { k = 1 } ^ { N _ { H } } \mathbf { H } _ { k } ^ { - 1 } \left[ P _ { k } ( i , j ) \right] \cdot \mathcal { M } _ { k } ( i , j ) } { \sum _ { k = 1 } ^ { N _ { H } } \mathcal { M } _ { k } ( i , j ) + \epsilon } ,\tag{4}
$$

where $\mathcal { M } _ { k }$ is the validity mask for homography k, and homographies covering less than 50% of the image are discarded. The aggregated map is discretized into per-cell pseudo-labels Y<sup>ˆ</sup> , and the HA loss supervises the sharp-image forward pass:

$$
\mathcal { L } _ { \mathrm { h a } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( i , j ) \in \mathcal { V } } \mathrm { C E } \big ( \mathbf { L } _ { i j } ^ { s } , \hat { \mathbf { Y } } _ { i j } \big ) ,\tag{5}
$$

where $\mathbf { L } _ { i j } ^ { s }$ are the cell $( i , j ) \mathrm { ^ s }$ logits from the sharp-image forward pass and $\hat { \mathbf { Y } } _ { i j }$ are their respective HA pseudo-labels.

This loss anchors the network’s detection response to stable, geometry-consistent locations.

Blur consistency loss ${ \mathcal { L } } _ { \mathbf { b l u r } }$ . The core self-supervised signal for blur robustness enforces that the blurred image produces keypoints at the same locations as its paired sharp image. We treat the sharp image’s per-cell arg max prediction as a hard target for the blurred image:

$$
\mathcal { L } _ { \mathrm { b l u r } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( i , j ) \in \mathcal { V } } \mathbf { C E } \big ( \mathbf { L } _ { i j } ^ { b } , \underbrace { \mathrm { a r g } \operatorname* { m a x } _ { c } \mathbf { L } _ { i j } ^ { s } } _ { \mathrm { d e t a c h e d } } \big ) ,\tag{6}
$$

where $\mathbf { L } _ { i j } ^ { b }$ and $\mathbf { L } _ { i j } ^ { s }$ are the cell $( i , j ) \ d s$ logits from the blurredand sharp-image forward passes, respectively. “detached” indicates that gradients are not backpropagated through the sharp arg max target, so it is treated as a fixed pseudolabel rather than being optimized jointly with the blurredimage prediction. This hard consistency formulation differs from soft Kullback–Leibler (KL) divergence distillation, as it is computationally efficient and directly aligns the predicted keypoint cells rather than the full distribution.

Position consistency loss $\mathcal { L } _ { \mathrm { p o s } } .$ The detector head also predicts sub-pixel offsets $\delta \in [ \dot { 0 } , 1 ] ^ { 2 \times H _ { c } \times W _ { c } }$ <sup>c</sup> within each cell. To ensure that the blurred image predicts the same sub-pixel positions as the sharp image, we apply an $\ell _ { 2 }$ regression loss:

$$
\mathcal { L } _ { \mathrm { p o s } } = \frac { 1 } { H _ { c } W _ { c } } \big \| \delta ^ { s } - \delta ^ { b } \big \| _ { F } ^ { 2 } ,\tag{7}
$$

where $\delta ^ { s }$ and $\delta ^ { b }$ are the position offsets from the sharp and blur forward passes, respectively.

Spatial diversity loss $\mathcal { L } _ { \bf d i v }$ . Without explicit coverage enforcement, the network tends to collapse to detecting only the most salient locations (typically image centers or high-contrast edges), leaving most of the image with near-zero response. We prevent this by partitioning the feature map into a $G _ { h } \times G _ { w }$ grid of spatial cells and penalizing low maximum response in any cell:

$$
\mathcal { L } _ { \mathrm { d i v } } = - \frac { 1 } { G _ { h } G _ { w } } \sum _ { m = 1 } ^ { G _ { h } G _ { w } } \log \bigl ( \operatorname* { m a x } _ { p \in \mathcal { C } _ { m } } \hat { P } _ { p } ^ { b } + \epsilon \bigr ) ,\tag{8}
$$

where ${ \hat { P } } ^ { b }$ is the probability map (dustbin excluded) from the blurred forward pass, ${ \mathcal { C } } _ { m }$ indexes the pixels within the $m -$ th grid cell, and $\epsilon = 1 0 ^ { - 6 }$ ensures numerical stability. This loss encourages the detector to maintain at least one confident keypoint in every spatial region, promoting well-distributed detections across the image.

The loss weights are set to $\lambda _ { \mathrm { h a } } = 0 . 5 , \lambda _ { \mathrm { b l u r } } = 0 . 5 , \lambda _ { \mathrm { p o s } } =$ 0.1, and $\lambda _ { \mathrm { d i v } } = 0 . 0 0 5$ . Despite the small weight, ${ \mathcal { L } } _ { \mathrm { d i v } }$ proves to be the most critical component (see Sec. IV-I). Without it, detection again collapses onto a handful of salient locations even after the geometric pretraining stage.

## D. Implementation Details

Stage 1 training. We pretrain on 25, 000 synthetic geometric images for 10 epochs (batch size 8, 320 × 320 crops, Adam optimizer with initial $\mathrm { l r } = 1 0 ^ { - 5 }$ , linear warmup over 4 epochs).

Stage 2 training. We train on the GoPro dataset [24] (2, 912 sharp-blur pairs from 30 sequences) for 36 epochs starting from the Stage 1 checkpoint, with batch size 8, $3 2 0 \times 3 2 0$ random crops, $N _ { H } = 1 0 0$ homographies per image for online HA, spatial diversity grid $G _ { h } ~ = ~ G _ { w } ~ = ~ 8 ,$ and rotation augmentation up to $\pm 9 0 ^ { \circ }$ . The Adam optimizer uses initial $\mathrm { l r } = 1 0 ^ { - 5 }$ , decaying at 60% and 80% of total epochs by a factor of 0.1.

Data augmentation. During Stage 2, each sharp-blur pair undergoes synchronized geometric augmentation (random homography, horizontal flip) followed by independent color jitter (brightness ±0.4, contrast ±0.3, saturation ±0.3, hue ±0.1) and random grayscale conversion $( p = 0 . 1 )$ . All training is conducted on a single NVIDIA GeForce RTX 3090.

## IV. EXPERIMENTS

## A. Datasets

Blur-HPatches. As no specific benchmark exists for evaluating keypoint detectors directly on motion-blurred images, we use the Blur-HPatches dataset introduced in [27]. It is derived from the original HPatches dataset [43], which provides 116 sequences (59 viewpoint changes and 57 illumination changes), each with one reference image and five target images accompanied by ground-truth homographies. For each image, motion blur is synthesized by convolving with a randomly generated point spread function. Three difficulty levels are defined by increasing blur kernel size and motion irregularity: EASY, HARD, and TOUGH. We evaluate SSMB under two settings: blur-to-sharp (reference sharp, target blurred) and blurto-blur (both images blurred). The original HPatches dataset (all images sharp) is additionally used to assess performance on clean images.

Deblur-HPatches. Also introduced in [27], this dataset applies two state-of-the-art deblurring networks, SRN-DeblurNet [26] and DeblurGAN-v2 [25], to the Blur-HPatches images without any fine-tuning. It is used to evaluate whether a deblur-then-detect pipeline can match the performance of a deblur-free approach.

ArchViz. For relative pose estimation, we use the ArchViz dataset [44], generated using the Unreal Engine to simulate rapid back-and-forth camera movements. It provides groundtruth camera poses and paired sharp and blurred images, along with texture-less regions and repetitive patterns that make pose estimation particularly challenging. The evaluation set consists of 3, 321 image pairs at 768×480 resolution, with up to 2, 048 keypoints per image.

Aachen Day-Night. For visual localization, we use the Aachen Day-Night benchmark [3], which covers 824 daytime and 98 nighttime query images against a 3D map built from 4, 328 daytime reference images. Following the established protocol for evaluating motion-blur robustness on real photographic benchmarks [27], [44], [45], we synthesize motion blur on both query and database images using the same procedure as Blur-HPatches. We use the original Aachen v1.0 benchmark to maintain consistency with this protocol.

## B. Evaluation Metrics

Repeatability. We use the repeatability metric [46] to evaluate keypoint detection quality. For an image pair, repeatability is the ratio of mutually corresponding keypoints to the smaller of the two detection counts. Correspondences are identified via overlap error $\epsilon _ { \mathrm { I o U } }$ between keypoint regions [47], with a match accepted if $\epsilon _ { \mathrm { I o U } } < 0 . 4$ (region overlap greater than 60%). We use 1, 000 keypoints per image.

MMA. For image matching, we report the Mean Matching Accuracy (MMA) [16], defined as the ratio of correctly matched keypoints at pixel thresholds of 3, 5, and 10 pixels, using up to 2, 048 keypoints per image.

Pose AUC. For relative pose estimation, we report the Area Under the Cumulative Curve (AUC) of the pose error at thresholds 5<sup>◦</sup>, 10<sup>◦</sup>, 20<sup>◦</sup>, and 30<sup>◦</sup>, following [33], [48]. The pose error is the maximum angular error in rotation and translation. We solve the essential matrix using the five-point algorithm [49] with RANSAC [50], with the inlier threshold tuned per method on the test data, following [51], since evaluating RANSAC itself is not the focus of this comparison.

Localization accuracy. For visual localization, we report the percentage of queries correctly localized within error thresholds (0.25m, 2<sup>◦</sup>), (0.5m, 5<sup>◦</sup>), and (1.0m, 10<sup>◦</sup>).

## C. Baseline Methods

We compare SSMB against a comprehensive set of methods spanning three categories.

Sparse methods. We compare against classical SIFT [13], DoG [13], and a comprehensive set of learning-based methods: Key.Net [15], SuperPoint [14], LF-Net [52], D2-Net [16], R2D2 [17], DISK [18], SiLK [21], REKD [31], NeSS-ST [19], ALIKED [20], DeDoDe v2 [23], XFeat [32], and BALF [27]. For downstream tasks (image matching, pose estimation, and visual localization), each method is paired with its native descriptor (or HardNet [53]/HyNet [54] when a native descriptor is unavailable) and matched via non-learned matching (mutual nearest neighbor, MNN, or Dual-Softmax), except where noted otherwise. SuperPoint, DISK, and ALIKED are additionally evaluated with LightGlue [51] as a learned matcher.

Detector-free methods. For image matching, pose estimation, and visual localization, we include LoFTR [33], Match-Former [35], ASpanFormer [36], and RoMa v2 [39]. These methods operate on full image pairs and represent a complementary paradigm to sparse detectors. They are reported separately and highlights are applied within sparse methods only, as the two paradigms are not directly comparable in terms of computational cost and target downstream tasks.

For evaluation in downstream tasks, SSMB keypoints are paired with HardNet [53] or HyNet [54] descriptors and MNN matching.

Note that the red, orange, and yellow highlights in all tables indicate the 1st , 2nd , and 3rd best performing sparse method for each metric, respectively.

TABLE I  
KEYPOINT DETECTION REPEATABILITY ON THE BLUR-HPATCHES DATASET [27]. FOR COMPACTNESS, WE ONLY REPORT THE OVERALL REPEATABILITY. THE DISAGGREGATED RESULTS FOR VIEWPOINT AND ILLUMINATION CHANGES CAN BE FOUND IN SEC. A OF THE APPENDIX.
<table><tr><td></td><td colspan="3">Reference: Sharp Target: Blur</td><td colspan="3">Reference: Blur Target: Blur</td></tr><tr><td>Method</td><td>EASY ↑</td><td>HARD ↑</td><td>TOUGH ↑</td><td>EASY ↑</td><td>HARD ↑</td><td>TOUGH ↑</td></tr><tr><td>SIFT [13]</td><td>55.92</td><td>56.80</td><td>53.49</td><td>56.99</td><td>53.49</td><td>45.94</td></tr><tr><td>Key.Net [15]</td><td>60.34</td><td>54.71</td><td>44.69</td><td>62.77</td><td>58.17</td><td>49.25</td></tr><tr><td>SuperPoint [14]</td><td>65.64</td><td>62.22</td><td>52.84</td><td>58.60</td><td>50.03</td><td>43.28</td></tr><tr><td>LF-Net [52]</td><td>63.54</td><td>61.19</td><td>56.78</td><td>60.45</td><td>59.07</td><td>57.71</td></tr><tr><td>D2-Net [16]</td><td>49.71</td><td>47.30</td><td>44.32</td><td>51.80</td><td>51.05</td><td>50.53</td></tr><tr><td>R2D2 [17]</td><td>57.99</td><td>51.73</td><td>40.57</td><td>57.49</td><td>55.31</td><td>46.86</td></tr><tr><td>DISK [18]</td><td>60.24</td><td>58.18</td><td>55.97</td><td>61.89</td><td>60.87</td><td>60.65</td></tr><tr><td>REKD [31]</td><td>55.15</td><td>53.15</td><td>48.52</td><td>52.57</td><td>48.54</td><td>44.02</td></tr><tr><td>NeSS-ST [19]</td><td>57.93</td><td>56.12</td><td>53.83</td><td>60.97</td><td>59.80</td><td>59.21</td></tr><tr><td>ALIKED [20]</td><td>58.88</td><td>54.12</td><td>50.36</td><td>62.19</td><td>59.49</td><td>57.57</td></tr><tr><td>DeDoDe v2 [23]</td><td>52.78</td><td>49.41</td><td>46.78</td><td>60.17</td><td>57.39</td><td>54.99</td></tr><tr><td>XFeat [32]</td><td>54.82</td><td>51.59</td><td>49.56</td><td>59.70</td><td>58.29</td><td>57.74</td></tr><tr><td>BALF [27]</td><td>74.12</td><td>74.45</td><td>71.84</td><td>70.48</td><td>68.43</td><td>67.71</td></tr><tr><td>SSMB (Ours)</td><td>77.24</td><td>77.18</td><td>77.16</td><td>74.20</td><td>74.33</td><td>74.48</td></tr></table>

## D. Keypoint Detection

Although SSMB is designed for motion-blurred images, we additionally verify its performance on the original, all-sharp HPatches dataset, where it remains competitive with methods explicitly optimized for sharp images. Full results are provided in Sec. A of the Appendix.

Results on Blur-HPatches. Tab. I presents the main evaluation on motion-blurred images. In the blur-to-sharp setting, SSMB achieves 77.24%, 77.18%, and 77.16% on EASY, HARD, and TOUGH respectively, outperforming BALF at every difficulty level. Importantly, SSMB’s performance remains remarkably stable across difficulty levels, whereas BALF drops from 74.12% to 71.84% and most other methods degrade substantially as blur severity increases. This stability is a direct consequence of the blur consistency training objective, which explicitly aligns detection responses across different levels of blur degradation. In the blur-to-blur setting, SSMB outperforms all methods across all difficulty levels, achieving 74.20%, 74.33%, and 74.48% on EASY, HARD, and TOUGH respectively. Unlike most detectors, which degrade as blur severity increases, SSMB exhibits a slight improvement, a further consequence of the blur consistency loss.

Results on Deblur-HPatches. Tab. II evaluates the deblurthen-detect pipeline, where blurred images are first restored by SRN-DeblurNet or DeblurGAN-v2 and then fed to each keypoint detector. While deblurring networks improve the performance of other detectors, neither deblurring-detector pair is able close the performance gap to SSMB. For example, under the TOUGH deblur-to-sharp setting, the best competing result (DeblurGAN-v2 with SuperPoint) reaches 58.22%, while SSMB directly on the blurred image achieves 77.16%. This confirms that deblurring introduces artifacts that limit downstream keypoint detection, and that a deblur-free approach is more effective.

TABLE II  
KEYPOINT DETECTION REPEATABILITY ON THE DEBLUR-HPATCHES DATASET [27], WHERE BLURRED IMAGES ARE FIRST RESTORED USINGSRN-DEBLURNET OR DEBLURGAN-V2 BEFORE DETECTION.
<table><tr><td></td><td colspan="6">Reference: Sharp Target: Deblur</td><td colspan="6">Reference: Deblur Target: Deblur</td></tr><tr><td></td><td colspan="3">SRN-DeblurNet [26]</td><td colspan="3">DeblurGAN-v2 [25]</td><td colspan="3">SRN-DeblurNet [26]</td><td colspan="3">DeblurGAN-v2 [25]</td></tr><tr><td>Method</td><td>EASY ↑</td><td>HARD ↑</td><td>TOUGH ↑</td><td>EASY ↑</td><td>HARD ↑</td><td>TOUGH ↑</td><td>EASY ↑</td><td>HARD ↑</td><td>TOUGH ↑</td><td>EASY ↑</td><td>HArD ↑</td><td>TOUGH ↑</td></tr><tr><td>SIFT [13]</td><td>56.62</td><td>55.36</td><td>53.83</td><td>57.63</td><td>56.52</td><td>56.50</td><td>59.75</td><td>58.13</td><td>50.63</td><td>59.44</td><td>57.98</td><td>51.21</td></tr><tr><td>Key.Net [15]</td><td>63.28</td><td>58.01</td><td>47.10</td><td>63.99</td><td>59.16</td><td>49.35</td><td>62.86</td><td>60.44</td><td>50.74</td><td>62.73</td><td>60.58</td><td>52.96</td></tr><tr><td>SuperPoint [14]</td><td>67.72</td><td>64.05</td><td>55.26</td><td>67.95</td><td>65.86</td><td>58.22</td><td>66.38</td><td>63.16</td><td>49.52</td><td>66.50</td><td>63.71</td><td>52.09</td></tr><tr><td>LF-Net [52]</td><td>62.22</td><td>59.90</td><td>54.73</td><td>62.59</td><td>60.24</td><td>54.81</td><td>63.06</td><td>62.03</td><td>57.28</td><td>63.00</td><td>61.79</td><td>57.85</td></tr><tr><td>D2-Net [16]</td><td>51.81</td><td>49.49</td><td>45.94</td><td>52.64</td><td>50.21</td><td>45.88</td><td>53.60</td><td>53.00</td><td>50.93</td><td>53.93</td><td>53.29</td><td>50.74</td></tr><tr><td>R2D2 [17]</td><td>60.31</td><td>55.43</td><td>43.26</td><td>60.46</td><td>55.68</td><td>45.38</td><td>58.11</td><td>54.80</td><td>45.77</td><td>57.95</td><td>55.03</td><td>47.86</td></tr><tr><td>DISK [18]</td><td>63.44</td><td>61.74</td><td>57.18</td><td>63.25</td><td>61.76</td><td>58.12</td><td>64.10</td><td>63.46</td><td>59.46</td><td>63.86</td><td>62.86</td><td>59.86</td></tr><tr><td>REKD [31]</td><td>55.18</td><td>53.87</td><td>49.50</td><td>55.27</td><td>53.82</td><td>50.65</td><td>54.22</td><td>51.99</td><td>45.35</td><td>53.86</td><td>51.61</td><td>45.59</td></tr><tr><td>NeSS-ST [19]</td><td>61.26</td><td>59.61</td><td>55.37</td><td>61.40</td><td>59.88</td><td>56.27</td><td>63.11</td><td>62.18</td><td>59.33</td><td>62.69</td><td>62.01</td><td>59.39</td></tr><tr><td>ALIKED [20]</td><td>66.24</td><td>63.51</td><td>54.96</td><td>66.81</td><td>64.16</td><td>57.28</td><td>68.45</td><td>66.26</td><td>58.19</td><td>68.03</td><td>66.29</td><td>60.78</td></tr><tr><td>DeDoDe v2 [23]</td><td>56.80</td><td>55.15</td><td>47.88</td><td>57.75</td><td>56.18</td><td>51.12</td><td>61.67</td><td>60.89</td><td>52.91</td><td>61.34</td><td>60.75</td><td>56.97</td></tr><tr><td>XFeat [32]</td><td>58.59</td><td>57.50</td><td>52.47</td><td>59.13</td><td>57.77</td><td>53.34</td><td>61.68</td><td>61.14</td><td>58.01</td><td>61.43</td><td>60.87</td><td>58.59</td></tr><tr><td>BALF [27]</td><td>74.12</td><td>74.45</td><td>71.84</td><td>74.12</td><td>74.45</td><td>71.84</td><td>70.48</td><td>68.43</td><td>67.71</td><td>70.48</td><td>68.43</td><td>67.71</td></tr><tr><td>SSMB (Ours)</td><td>77.24</td><td>77.18</td><td>77.16</td><td>77.24</td><td>77.18</td><td>77.16</td><td>74.20</td><td>74.33</td><td>74.48</td><td>74.20</td><td>74.33</td><td>74.48</td></tr></table>

TABLE III

IMAGE MATCHING ON THE BLUR-HPATCHES DATASET [27]. MEAN MATCHING ACCURACY (MMA) IS REPORTED AT MULTIPLE PIXEL THRESHOLDS.FIRST SECOND , AND THIRD BEST RESULTS ARE HIGHLIGHTED WITHIN SPARSE METHODS ONLY, AS DETECTOR-FREE METHODS OPERATE ON FULLIMAGES AND ARE NOT DIRECTLY COMPARABLE TO SPARSE DETECTORS
<table><tr><td rowspan="2"></td><td colspan="3">Overall MMA ↑</td><td colspan="3">Illumination MMA ↑</td><td colspan="3">Viewpoint MMA ↑</td></tr><tr><td>@3px</td><td>@5px</td><td>@10px</td><td>@3px</td><td>@5px</td><td>@10px</td><td>@3px</td><td>@5px</td><td>@10px</td></tr><tr><td>Detector-free (Dense/Semi-dense)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoFTR [33]</td><td>26.40</td><td>35.28</td><td>64.96</td><td>39.97</td><td>42.32</td><td>73.23</td><td>13.29</td><td>28.48</td><td>56.96</td></tr><tr><td>MatchFormer [35]</td><td>38.08</td><td>48.86</td><td>73.67</td><td>57.69</td><td>60.57</td><td>86.33</td><td>19.14</td><td>37.54</td><td>61.45</td></tr><tr><td>ASpanFormer [36]</td><td>22.91</td><td>33.20</td><td>65.51</td><td>29.35</td><td>30.87</td><td>63.24</td><td>16.69</td><td>35.46</td><td>67.70</td></tr><tr><td>RoMa v2 [39]</td><td>28.51</td><td>54.58</td><td>88.54</td><td>36.24</td><td>67.24</td><td>96.47</td><td>21.04</td><td>42.34</td><td>80.89</td></tr><tr><td>Sparse (detector + learned matcher)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SuperPoint [14] + LightGlue [51]</td><td>9.93</td><td>24.58</td><td>61.17</td><td>10.88</td><td>26.72</td><td>65.55</td><td>9.02</td><td>22.51</td><td>56.93</td></tr><tr><td>DISK [18] + LightGlue [51]</td><td>12.58</td><td>29.52</td><td>63.69</td><td>15.24</td><td>35.01</td><td>71.83</td><td>10.00</td><td>24.21</td><td>55.84</td></tr><tr><td>ALIKED [20] + LightGlue [51]</td><td>11.51</td><td>27.36</td><td>62.91</td><td>11.77</td><td>27.83</td><td>62.74</td><td>11.27</td><td>26.91</td><td>63.07</td></tr><tr><td>Sparse (detector + non-learned matching)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SIFT [13] + MNN</td><td>13.08</td><td>21.79</td><td>31.74</td><td>12.85</td><td>21.79</td><td>32.42</td><td>13.31</td><td>21.79</td><td>31.08</td></tr><tr><td>DoG [13] + AffNet [55] + HardNet [53] + MNN</td><td>7.01</td><td>13.50</td><td>24.72</td><td>6.26</td><td>12.25</td><td>22.80</td><td>7.73</td><td>14.71</td><td>26.58</td></tr><tr><td>Key.Net [15] + AffNet [55] + HardNet [53] + MNN</td><td>10.12</td><td>20.68</td><td>41.65</td><td>11.62</td><td>22.37</td><td>43.76</td><td>8.68</td><td>19.05</td><td>39.61</td></tr><tr><td>SuperPoint [14] + MNN</td><td>5.63</td><td>13.34</td><td>30.76</td><td>7.21</td><td>16.91</td><td>38.03</td><td>4.09</td><td>9.90</td><td>23.72</td></tr><tr><td>D2-Net [16] + MNN</td><td>12.79</td><td>27.43</td><td>54.80</td><td>16.92</td><td>34.76</td><td>66.48</td><td>8.81</td><td>20.35</td><td>43.52</td></tr><tr><td>R2D2 [17] + MNN</td><td>15.39</td><td>28.83</td><td>52.28</td><td>19.05</td><td>33.83</td><td>60.31</td><td>11.84</td><td>23.99</td><td>44.52</td></tr><tr><td>DISK [18] + MNN</td><td>9.18</td><td>21.61</td><td>47.87</td><td>11.47</td><td>26.56</td><td>56.68</td><td>6.97</td><td>16.82</td><td>39.35</td></tr><tr><td>SiLK [21] + MNN</td><td>2.06</td><td>4.13</td><td>7.77</td><td>4.17</td><td>8.39</td><td>15.67</td><td>0.02</td><td>0.02</td><td>0.15</td></tr><tr><td>DeDoDe v2 [23] + Dual-Softmax</td><td>10.76</td><td>21.71</td><td>46.29</td><td>15.90</td><td>29.22</td><td>57.83</td><td>5.80</td><td>14.45</td><td>35.13</td></tr><tr><td>BALF [27] + HardNet [53] + MNN</td><td>25.03</td><td>44.71</td><td>72.64</td><td>34.58</td><td>56.06</td><td>84.90</td><td>15.80</td><td>33.74</td><td>60.81</td></tr><tr><td>BALF [27] + HyNet [54] + MNN</td><td>20.55</td><td>38.12</td><td>67.84</td><td>27.51</td><td>45.90</td><td>78.03</td><td>13.82</td><td>30.62</td><td>58.00</td></tr><tr><td>SSMB (Ours) + HardNet [53] + MNN</td><td>40.10</td><td>51.43</td><td>77.22</td><td>62.02</td><td>62.78</td><td>85.60</td><td>18.91</td><td>40.46</td><td></td></tr><tr><td>SSMB (Ours) + HyNet [54] + MNN</td><td>38.45</td><td>49.52</td><td>76.67</td><td>59.93</td><td>60.23</td><td>84.91</td><td>18.21</td><td>39.18</td><td>69.12 68.72</td></tr></table>

## E. Image Matching

Dataset and protocol. We evaluate image matching on the Blur-HPatches TOUGH split under the blur-to-blur setting, using MMA at pixel thresholds of 3, 5, and 10.

Results. As shown in Tab. III, SSMB with HardNet achieves 40.10%, 51.43%, and 77.22% MMA at 3, 5, and 10 pixels overall, ranking first among all sparse methods by a substantial margin. The next best sparse method is BALF with HardNet, which reaches 72.64% at 10 pixels overall, benefiting from the same deblur-free detection paradigm but without the selfsupervised training that removes SIFT dependency. The improvement is most pronounced under illumination changes, where SSMB+HardNet achieves 62.02% at 3 pixels compared to 34.58% for the next best sparse method (BALF+HardNet). Under viewpoint changes, SSMB+HardNet achieves 69.12% at 10 pixels, also ranking first among sparse methods, ahead of BALF+HardNet at 60.81%. SSMB+HyNet ranks second across all metrics, confirming that the gains stem from the detector rather than the descriptor choice. Among detector-free methods, RoMa v2 achieves 88.54% at 10 pixels. However, this performance comes at the cost of operating on complete image pairs during inference, resulting in substantially higher computation than that of our sparse detector (see Tab. VI). Note also that SSMB outperforms all detector-free methods in overall MMA at the strict threshold of 3 pixels.

TABLE IV  
RELATIVE POSE ESTIMATION ON THE ARCHVIZ DATASET [44]. THE AREA UNDER THE CUMULATIVE CURVE (AUC) IS REPORTED AT MULTIPLE ANGULAR ERROR THRESHOLDS IN BLUR-TO-SHARP AND BLUR-TO-BLUR SETTINGS. FIRST SECOND , AND THIRD BEST RESULTS ARE HIGHLIGHTED WITHIN SPARSE METHODS ONLY.
<table><tr><td rowspan="3">Method</td><td colspan="4">Blur-to-Sharp</td><td colspan="4">Blur-to-Blur</td></tr><tr><td colspan="4">Pose AUC ↑</td><td colspan="4">Pose AUC ↑</td></tr><tr><td>@5°</td><td>@10°</td><td>@20°</td><td>@30°</td><td>@5°</td><td>@10°</td><td>@20°</td><td>@30°</td></tr><tr><td>Detector-free (Dense/Semi-dense)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoFTR [33]</td><td>10.14</td><td>23.15</td><td>41.94</td><td>54.72</td><td>6.64</td><td>19.53</td><td>41.54</td><td>52.25</td></tr><tr><td>MatchFormer [35]</td><td>2.50</td><td>4.32</td><td>10.45</td><td>18.39</td><td>0.00</td><td>2.55</td><td>9.74</td><td>14.68</td></tr><tr><td>ASpanFormer [36]</td><td>4.31</td><td>11.00</td><td>20.20</td><td>28.87</td><td>3.56</td><td>7.40</td><td>15.85</td><td>22.31</td></tr><tr><td>RoMa v2 [39]</td><td>21.45</td><td>38.50</td><td>59.01</td><td>70.76</td><td>16.71</td><td>35.02</td><td>57.09</td><td>68.26</td></tr><tr><td>Sparse (detector + learned matcher)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SuperPoint [14] + LightGlue [51]</td><td>14.86</td><td>26.50</td><td>40.72</td><td>51.07</td><td>4.25</td><td>8.67</td><td>16.86</td><td>28.13</td></tr><tr><td>DISK [18] + LightGlue [51]</td><td>12.01</td><td>18.63</td><td>30.33</td><td>39.12</td><td>2.47</td><td>9.36</td><td>17.14</td><td>22.54</td></tr><tr><td>ALIKED [20] + LightGlue [51]</td><td>6.83</td><td>16.39</td><td>30.07</td><td>39.54</td><td>5.99</td><td>14.31</td><td>27.56</td><td>36.63</td></tr><tr><td>Sparse (detector + non-learned matching)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SIFT [13] + MNN</td><td>9.28</td><td>19.60</td><td>34.44</td><td>45.92</td><td>5.20</td><td>12.53</td><td>23.16</td><td>32.26</td></tr><tr><td>DoG [13] + AffNet [55] + HardNet [53] + MNN</td><td>8.56</td><td>16.29</td><td>27.48</td><td>35.58</td><td>3.43</td><td>8.96</td><td>18.42</td><td>25.55</td></tr><tr><td>Key.Net [15] + AffNet [55] + HardNet [53] + MNN</td><td>9.91</td><td>16.61</td><td>29.51</td><td>39.74</td><td>5.53</td><td>13.88</td><td>24.36</td><td>33.64</td></tr><tr><td>SuperPoint [14] + MNN</td><td>7.07</td><td>12.91</td><td>24.43</td><td>32.94</td><td>3.73</td><td>6.57</td><td>11.28</td><td>17.17</td></tr><tr><td>D2-Net [16] + MNN</td><td>7.18</td><td>19.83</td><td>40.19</td><td>53.38</td><td>4.90</td><td>17.01</td><td>37.16</td><td>50.05</td></tr><tr><td>R2D2 [17] + MNN</td><td>14.26</td><td>23.48</td><td>38.37</td><td>48.00</td><td>9.93</td><td>22.25</td><td>34.75</td><td>44.85</td></tr><tr><td>DISK [18] + MNN</td><td>6.98</td><td>14.90</td><td>24.59</td><td>31.02</td><td>3.66</td><td>6.57</td><td>9.97</td><td>13.94</td></tr><tr><td>SiLK [21] + MNN</td><td>0.00</td><td>0.00</td><td>0.67</td><td>3.34</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.82</td></tr><tr><td>DeDoDe v2 [23] + Dual-Softmax</td><td>9.46</td><td>19.86</td><td>35.95</td><td>45.12</td><td>5.87</td><td>11.10</td><td>17.85</td><td>24.07</td></tr><tr><td>BALF [27] + HardNet [53] + MNN</td><td>14.89</td><td>27.29</td><td>47.84</td><td>59.15</td><td>12.63</td><td>25.17</td><td>46.09</td><td>58.42</td></tr><tr><td>BALF [27] + HyNet [54] + MNN</td><td>14.29</td><td>26.57</td><td>45.17</td><td>58.44</td><td>10.06</td><td>23.11</td><td>42.19</td><td>55.20</td></tr><tr><td>SSMB (Ours) + HardNet [53] + MNN</td><td>15.08</td><td>27.34</td><td>48.67</td><td>63.01</td><td>10.99</td><td></td><td>46.95</td><td></td></tr><tr><td>SSMB (Ours) + HyNet [54] + MNN</td><td>16.51</td><td>31.59</td><td>52.94</td><td>66.61</td><td>11.07</td><td>25.61 24.65</td><td>48.39</td><td>60.24 60.83</td></tr></table>

## F. Relative Pose Estimation

Dataset and protocol. We evaluate on the ArchViz dataset [44] under both blur-to-sharp and blur-to-blur settings. We use up to 2, 048 keypoints per image and report Pose AUC at $5 ^ { \circ } , 1 0 ^ { \circ } , 2 0 ^ { \circ }$ , and $3 0 ^ { \circ }$ thresholds.

Results. Tab. IV shows that SSMB consistently achieves top-2 results among sparse keypoint detectors across both settings and all thresholds, with at least one of its two descriptor variants (HardNet or HyNet) ranking in the top two in every case. In the blur-to-sharp setting, SSMB+HyNet achieves an AUC of 66.61 at $3 0 ^ { \circ }$ , the highest among all sparse methods and a substantial improvement over the next best result of 59.15 (BALF+HardNet). In the blur-to-blur setting, SSMB+HyNet again leads sparse methods with 60.83 at 30<sup>◦</sup>, ahead of the next best sparse method, BALF+HardNet, at 58.42. The only exception to this trend occurs at the $5 ^ { \circ }$ threshold in the blur-to-blur setting, where BALF+HardNet (12.63) narrowly surpasses both SSMB variants (11.07 and 10.99). At this very strict threshold, pose accuracy is highly sensitive to a small number of high-quality correspondences, and both methods perform close to the noise floor of the benchmark. The consistent performance of SSMB across both settings and the remaining seven threshold-setting combinations demonstrates that blur-robust detection translates directly to improved pose estimation accuracy.

## G. Visual Localization

Dataset and protocol. We evaluate on the Aachen Day-Night benchmark [3] with motion blur on both query and database images. We use the hierarchical localization framework [11] with NetVLAD [56] retrieval (top-20 candidates), COLMAP [5] for 3D reconstruction, and 4, 096 keypoints per image at 640 × 480 resolution.

Results. As shown in Tab. V, SSMB outperforms all sparse baselines under both daytime and nighttime conditions. For daytime queries, SSMB+HyNet achieves 58.5%, 76.2%, and 93.0% at the three error thresholds, compared to 51.2%, 68.9%, and 89.4% for the best competing sparse method, BALF+HyNet. The improvement is even more pronounced for nighttime queries, where SSMB+HyNet achieves 31.6%, 59.2%, and 86.7%, above the next best result of 21.4%, 36.7%, and 80.6%, achieved by BALF+HardNet. The larger nighttime gains are expected, as nighttime images naturally exhibit motion blur due to longer exposures, making blur-robust detection particularly beneficial in this condition. Interestingly, BALF’s stronger descriptor pairing differs by condition (HyNet for daytime, HardNet for nighttime), whereas SSMB+HyNet remains the best configuration in both conditions, suggesting that SSMB’s detections are more consistently compatible with a single descriptor across illumination changes. Among detectorfree methods, LoFTR, MatchFormer, and ASpanFormer all trail SSMB at every threshold and condition (e.g., LoFTR

TABLE V

VISUAL LOCALIZATION ON THE AACHEN DAY-NIGHT BENCHMARK [3]. THE PERCENTAGE OF QUERIES SUCCESSFULLY LOCALIZED WITHIN THREE ERROR THRESHOLDS IS REPORTED. FIRST SECOND AND THIRD BEST RESULTS ARE HIGHLIGHTED WITHIN SPARSE METHODS ONLY. <sup>†</sup>ROMA V2 COULD NOT BE EVALUATED ON THIS

BENCHMARK DUE TO INSUFFICIENT SYSTEM MEMORY DURING LARGE-SCALE BUNDLE ADJUSTMENT (4, 328 DATABASE IMAGES).
<table><tr><td rowspan="3">Method</td><td colspan="2">Correctly Localized Queries (%) ↑</td></tr><tr><td>Day</td><td>Night</td></tr><tr><td colspan="2">(0.25m,2°) / (0.5m,5°) / (1.0m,10°)</td></tr><tr><td>Detector-free (Dense/Semi-dense)</td><td></td><td></td></tr><tr><td>LoFTR [33]</td><td>49.3 / 63.3 / 80.1</td><td>21.4 / 41.8 / 67.3</td></tr><tr><td>MatchFormer [35]</td><td>6.9 / 24.0 / 76.0</td><td>3.1 / 14.3 / 49.0 12.4 / 29.6 / 58.4</td></tr><tr><td>ASpanFormer [36] RoMa v2 [39]†</td><td>43.3 / 57.2 /  76.5 00M</td><td>00M</td></tr><tr><td>Sparse (detector + learned matcher)</td><td></td><td></td></tr><tr><td>SuperPoint [14] + LightGlue [51]</td><td>31.9 / 51.1 / 79.5</td><td>11.2 / 23.5 / 62.2</td></tr><tr><td>DISK [18] + LightGlue [51]</td><td>42.1 / 59.7 / 79.6</td><td>10.2 / 26.5 / 56.1</td></tr><tr><td>ALIKED [20] + LightGlue [51]</td><td>39.3 / 55.2 / 77.5</td><td>18.4 / 28.6 / 60.2</td></tr><tr><td>Sparse (detector + non-learned matching)</td><td></td><td></td></tr><tr><td>SIFT [13] + MNN</td><td>49.7 / 59.8 / 68.8</td><td>13.3 /  18.4 / 28.6</td></tr><tr><td>Key.Net [15] + AffNet [55] + HardNet [53] + MNN</td><td>35.7 / 52.3 / 74.6</td><td>7.1 / 23.5 / 40.8</td></tr><tr><td>SuperPoint [14] + MNN</td><td>33.3 / 56.4 / 76.5</td><td>9.2 / 21.4 / 60.2</td></tr><tr><td>D2-Net [16] + MNN</td><td>10.6 / 26.8 / 74.4</td><td>3.1 / 12.2 / 57.1</td></tr><tr><td>R2D2 [17] + MNN</td><td>47.5 / 60.9 /  78.4</td><td></td></tr><tr><td>DISK [18] + MNN</td><td>43.7 / 58.0 / 76.7</td><td>20.1 / 35.7 / 60.3</td></tr><tr><td>DeDoDe v2 [23] + Dual-Softmax</td><td>44.9 / 58.7 / 75.4</td><td>13.7 / 30.1 / 63.8 20.4 / 32.7 / 54.1</td></tr><tr><td>BALF [27] + HardNet [53] + MNN</td><td>51.0 / 68.0 / 89.2</td><td></td></tr><tr><td>BALF [27] + HyNet [54] + MNN</td><td>51.2 68.9 / 89.4</td><td>21.4 36.7 / 80.6 15.3 / 30.6 / 72.4</td></tr><tr><td></td><td></td><td></td></tr><tr><td>SSMB (Ours) + HardNet [53] + MNN</td><td>55.3 73.8</td><td>92.6 28.6 53.1 84.7</td></tr><tr><td>SSMB (Ours) + HyNet [54] + MNN</td><td>58.5 76.2</td><td>93.031.6 59.2 86.7</td></tr></table>

achieves 21.4%, 41.8%, and 67.3% at night, compared to SSMB’s 31.6%, 59.2%, and 86.7%), despite operating on full image pairs rather than sparse keypoints. RoMa v2 could not be evaluated on this benchmark, as its dense per-pair correspondence estimation exhausted the available system memory during bundle adjustment over this largescale reconstruction (4,328 database images), consistent with its substantially higher per-pair computational cost reported in Tab. VI. This result highlights the importance of blurrobust sparse detection even in comparison to dense matching approaches, particularly at the scale required by real-world visual localization deployments.

## H. Computational Efficiency

Tab. VI reports detection-only inference time for all methods at two resolutions, 240×320 and 480×640 pixels, measured on an NVIDIA RTX 3090 GPU. For methods that jointly predict descriptors alongside keypoints, we exclude the descriptor head from the forward pass to ensure a fair, detection-only comparison. Detector-free methods have no separate detection step, so the reported time is for full pairwise dense/semi-dense matching on an image pair, which is not directly comparable to the detection-only timings of sparse methods. SSMB runs at 11.65 ms and 30.63 ms, comfortably within real-time requirements (33 FPS at VGA resolution, i.e., 480 × 640 pixels) for robotic applications. Compared to BALF, SSMB incurs a modest overhead (30.63 ms vs. 27.04 ms at VGA resolution) due to the added LDE module, which is justified by the substantial gains in self-supervised training without external supervision (see Tab. VII).

TABLE VI  
COMPUTATIONAL COST (IN MILLISECOND) FOR KEYPOINT DETECTION ON A SINGLE IMAGE, MEASURED ON AN NVIDIA RTX 3090 GPU (INTEL XEON E5-2686 V4 CPU FOR SIFT). <sup>†</sup>FOR D2-NET, DETECTION AND DESCRIPTION CANNOT BE SEPARATED IN A SINGLE FORWARD PASS, SO THE REPORTED TIME INCLUDES BOTH.
<table><tr><td>Method</td><td>240×320 pixels↓</td><td>480×640 pixels↓</td></tr><tr><td>Detector-free (Dense/Semi-dense)</td><td></td><td></td></tr><tr><td>LoFTR [33]</td><td>30.35</td><td>75.22</td></tr><tr><td>MatchFormer [35]</td><td>22.81</td><td>37.92</td></tr><tr><td>ASpanFormer [36]</td><td>53.22</td><td>108.54</td></tr><tr><td>RoMa v2 [39]</td><td>486.64</td><td>497.96</td></tr><tr><td>Sparse</td><td></td><td></td></tr><tr><td>SIFT [13]</td><td>12.97</td><td>57.18</td></tr><tr><td>Key.Net [15]</td><td>14.80</td><td>33.54</td></tr><tr><td>SuperPoint [14]</td><td>2.14</td><td>3.40</td></tr><tr><td>D2-Net [16]†</td><td>4.97</td><td>17.84</td></tr><tr><td>R2D2 [17]</td><td>4.77</td><td>17.60</td></tr><tr><td>DISK [18]</td><td>4.89</td><td>13.18</td></tr><tr><td>REKD [31]</td><td>7.86</td><td>27.18</td></tr><tr><td>NeSS-ST [19]</td><td>10.54</td><td>14.45</td></tr><tr><td>ALIKED [20]</td><td>5.62</td><td>5.63</td></tr><tr><td>DeDoDe v2 [23]</td><td>14.92</td><td>42.77</td></tr><tr><td>XFeat [32]</td><td>5.45</td><td>5.44</td></tr><tr><td>BALF [27]</td><td>9.42</td><td>27.04</td></tr><tr><td>SSMB (Ours)</td><td>11.65</td><td>30.63</td></tr></table>

## I. Ablation Study

We conduct ablation studies to analyze the contribution of each design choice in SSMB. All ablations are evaluated on the Blur-HPatches TOUGH split under both blur-to-sharp and blur-to-blur settings, reporting repeatability.

Loss components. Tab. VII ablates each loss term by removing it while keeping all others. Removing ${ \mathcal { L } } _ { \mathrm { d i v } }$ causes the largest performance drop, from 77.16% to 64.07% in blur-to-sharp and from 74.48% to 65.26% in blur-to-blur, confirming that spatial diversity is the most critical component despite its small weight $( \lambda _ { \mathrm { d i v } } = 0 . 0 0 5 )$ ). Without ${ \mathcal { L } } _ { \mathrm { d i v } }$ , detection again collapses onto a handful of salient locations during GoPro training, even after geometric pretraining. Removing ${ \mathcal { L } } _ { \mathrm { b l u r } }$ causes the second largest drop (to 65.81% and 67.84%), demonstrating that explicit cross-domain consistency supervision is essential for blur robustness. Removing $\mathcal { L } _ { \mathrm { h a } }$ leads to a moderate drop (to 69.03% and 69.78%), as the HA loss anchors detection to geometrically stable locations on sharp images. Removing ${ \mathcal { L } } _ { \mathrm { p o s } }$ has the smallest effect (72.69% and 73.77%), indicating that sub-pixel position consistency provides only a marginal benefit under the repeatability metric used throughout this evaluation. Since this loss only enforces consistency between the sharp and blurred branches without an absolute geometric target, we additionally verify that the offset head does not collapse to a degenerate constant field. Applying the predicted offset δ to the final keypoint coordinates provides a modest but consistent improvement on the TOUGH split in both settings, from 75.74% to 77.16% in blur-to-sharp and from 73.89% to 74.48% in blur-to-blur, confirming that the offset head learns non-trivial sub-pixel corrections.

Multi-stage training. Skipping synthetic pretraining leads to a dramatic drop from 77.16% to 59.58% in blur-to-sharp repeatability and from 74.48% to 61.44% in blur-to-blur repeatability. This confirms that geometric pretraining is an essential prerequisite. Without it, the network cannot escape score map collapse during blur-aware training, and the blur consistency loss has no stable detection signal to align.

TABLE VII  
ABLATION STUDY. REPEATABILITY IS REPORTED ON THE BLUR-HPATCHES TOUGH SPLIT [27] UNDER BLUR-TO-SHARP AND BLUR-TO-BLUR SETTINGS.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Repeatability (%)</td></tr><tr><td>Blur-to-Sharp</td><td>Blur-to-Blur</td></tr><tr><td>Loss components</td><td></td><td></td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { h a } }$ </td><td>69.03</td><td>69.78</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { b l u r } }$ </td><td>65.81</td><td>67.84</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { p o s } }$ </td><td>72.69</td><td>73.77</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { d i v } }$ </td><td>64.07</td><td>65.26</td></tr><tr><td>Training pipeline</td><td></td><td></td></tr><tr><td>w/o Synthetic pretraining</td><td>59.58</td><td>61.44</td></tr><tr><td>Architecture w/o LDE</td><td></td><td></td></tr><tr><td></td><td>33.83</td><td>50.13</td></tr><tr><td>Full (ours)</td><td>77.16</td><td>74.48</td></tr></table>

LDE module. Removing LDE causes a dramatic drop from 77.16% to 33.83% in blur-to-sharp repeatability and from 74.48% to 50.13% in blur-to-blur repeatability, at a parameter saving of only 0.05M. This confirms that LDE is the most impactful architectural component in SSMB. Notably, the w/o LDE variant is architecturally identical to the BALFstyle backbone [27], trained with our self-supervised objective rather than SIFT supervision, showing that this backbone alone is insufficient to support stable self-supervised training under blur. This is because, without LDE, the global MLP mixing in the GridGmlpLayer destroys the local gradient structure that distinguishes repeatable keypoint locations under blur, leaving the network unable to identify discriminative spatial positions. Fig. 5 visualizes this effect directly. The response collapses toward near-zero almost everywhere, and the few residual peaks that remain no longer trace real structure, falling instead on featureless regions with a spacing that resembles a uniform grid rather than genuine edge localization.

## J. Qualitative Results

Detection on RWBI. Fig. 6 presents keypoint detection results on a real-world blurred image from the RWBI dataset [57], captured with real cameras rather than synthetic blur. This is the same image used earlier to visualize the effect of LDE in Fig. 5. SSMB produces well-localized keypoints that remain concentrated on genuine image structure, whereas SIFT and DISK detect comparatively sparser responses in heavily blurred regions, and SuperPoint, NeSS-ST, and ALIKED scatter detections more uniformly without consistently aligning to salient structure. Compared to BALF, SSMB produces more accurate keypoints around text regions and other heavily blurred structures, despite being trained without any SIFT-derived supervision. The enlarged regions highlight this difference clearly, particularly in areas with fine text or high-frequency edge structure that are otherwise easy to overlook at full-image scale. Additional keypoint detection results on the RealBlur dataset [28] are provided in Fig. 10 of the Appendix.

![](images/44aa43c0b488731eec89428fb51784422c953dfdf3b0597dbd6b239dc6a0907a.jpg)

![](images/7dd31f574644f14feac610cef0bf0b25eedc10007d6275ef12ae77ba66e26d5f.jpg)  
Fig. 5. Effect of the LDE module on the predicted probability map. Top row: probability heatmaps on the same real-world blurred image from the RWBI dataset [57], overlaid on a lightened version of the input. Full SSMB produces dense, well-localized responses that closely trace real structure such as storefront signage, arched window frames, and shop dividers. Without LDE, the response collapses toward near-zero almost everywhere, with only a handful of residual peaks remaining. Bottom row: the pixel-value histogram of the probability map, together with the extracted keypoints (green circles) for each variant. Full SSMB’s keypoints densely and precisely trace the storefront geometry. In contrast, the w/o LDE variant yields far fewer keypoints overall, which no longer trace real structure. Most fall on featureless regions such as bare pavement, with a spacing that resembles a uniform grid rather than genuine edge localization, consistent with the near-flat residual field revealed by the histogram. This visually explains the dramatic repeatability drop observed in Tab. VII (from 77.16% to 33.83% on the blur-to-sharp TOUGH setting). Without local discriminability, the network can no longer anchor its predictions to repeatable geometric structure.

Matching on RealBlur. Fig. 7 presents a qualitative feature matching result on a real-world sharp-blur image pair from the RealBlur dataset [28], using SSMB with HardNet [53] and MNN matching. The pair exhibits real camera motion blur, together with a viewpoint change between the sharp and blurred images. SSMB produces well-localized and repeatable keypoints from both images despite the cross-domain appearance gap, yielding dense and geometrically consistent matches, confirming that the gains on synthetic benchmarks transfer to real camera blur. Additional matching results across indoor and outdoor scenes are provided in Fig. 11 of the Appendix.

## V. CONCLUSION

We present SSMB, a self-supervised keypoint detector for motion-blurred images that requires no external supervision. The key architectural contribution is the Local Discriminability Enhancement (LDE), which restores fine-grained local discriminability after global feature mixing. The training pipeline proceeds in two stages, where geometric pretraining on synthetic shapes bootstraps spatial discriminability, and bluraware training on real sharp-blur pairs reinforces it through a multi-component self-supervised loss. Both stages are individually necessary and, in particular, the spatial diversity loss term proves essential for preventing the network from degenerating during training on blurred images. Evaluations across keypoint detection, image matching, relative pose estimation, and visual localization tasks demonstrate that SSMB achieves state-ofthe-art performance among sparse detectors under motion blur.

![](images/c693721147b8fe11affbc156b059ce12f46465e8498b2649088cf8f33d81c57a.jpg)  
Input

![](images/38e5b2a886bfa07da384e3ddf7e21f0f1d8dac2197631dbd444c1f0ca5e543e1.jpg)  
SIFT [13]

![](images/18b7c051287a9230c06c2540de03385039daa6a95f495f7d6e759b9f39e63d1d.jpg)  
SuperPoint [14]

![](images/05a7f9171bb2ba899cffa6dd71cede0bfbfbea8865aaf40ea4a9c99717e9d484.jpg)  
DISK [18]

![](images/947d4b12ee9ac1ef53f2acbf414359d79ef1639de0536424a00f6ea54f6b94c4.jpg)  
NeSS-ST [19]

![](images/a208ef768388bc4e9d3c23054921bfd08463b78cc49544ef25bfe5c18f29ded1.jpg)  
ALIKED [20]

![](images/4d88e7577306c95ba6aa0139f90d66c69715d5a5ae3444df5383e95ebde38ce7.jpg)  
BALF [27]

![](images/0064275895b67bdcc852ddc11ca3a037620dd0aee41322975a9c55a5faf77e39.jpg)  
SSMB (Ours)  
Fig. 6. Qualitative detection results on real-world blurred images from the RWBI dataset [57]. SSMB detects well-localized and consistently distributed keypoints under real camera motion blur, while several baselines either miss salient structure or produce noisy responses in heavily blurred regions. The red and blue boxes mark two representative regions, shown enlarged below each image for closer comparison. Best viewed in color.

![](images/d8f2f9b27531f35fde95d29db89bcfe1bcc7d7ca054cf34937983ceb590aa831.jpg)  
Fig. 7. Qualitative matching results on a real-world sharp-blur image pair from the RealBlur dataset [28]. Top: the original image pair (left: sharp, right: blurred). Bottom: SSMB detects well-localized and repeatable keypoints from both images for matching under real camera motion blur and viewpoint changes. Green lines indicate correct matches. Best viewed in color.

## ACKNOWLEDGMENTS

This work was supported by Spanish grants PID2021- 127685NB-I00 and PID2024-155886NB-I00 and Aragon grant´ T45 23R.

## APPENDIX

In this appendix, we provide additional analysis and experimental results that are omitted from the main paper due to space limitations:

• Complete keypoint detection results on the original, all-sharp HPatches dataset, together with the disaggregated viewpoint and illumination repeatability for Blur-HPatches and Deblur-HPatches (see Sec. A).

• Additional image matching results across the full range of pixel thresholds, complementing the single-threshold comparison reported in the main paper (see Sec. B).

• Additional qualitative results for relative pose estimation on the ArchViz dataset (see Sec. C).

• Additional qualitative detection and matching results on real-world blurred images from the RealBlur dataset (see Sec. D).

## A. Keypoint Detection

Due to limited space in the main paper, we only report the overall repeatability in Tabs. I and II of the main paper, and omit results on the original all-sharp HPatches dataset [43] entirely. We present the complete results here, including the viewpoint and illumination repeatability separately.

Results on HPatches (sharp images). Tab. VIII shows repeatability on the original HPatches dataset with all-sharp images. SSMB achieves 75.17% overall repeatability, ranking second among all methods and outperforming BALF (70.28%). This result is notable given that SSMB is specifically designed for blurred images and trained exclusively on GoPro blur-sharp pairs without any sharp-image-specific supervision. Homographic adaptation provides invariance to planar image warps, but does not fully model the 3D viewpoint changes, parallax, and occlusion present in HPatches’ viewpoint sequences, which explains the more modest viewpoint repeatability of 67.50% compared to methods such as REKD (80.94%) and ALIKED (72.85%) that are specifically designed for geometric invariance on sharp images. Meanwhile, the blur consistency objective and photometric augmentations may favor robustness to illumination and blur variations, consistent with the strong illumination repeatability of 83.11%. Motion blur primarily degrades fine spatial structure rather than overall photometric appearance, so learning to be consistent across blur levels naturally generalizes to illumination changes as well.

Results on Blur-HPatches. Tabs. IX and X report the detailed repeatability scores under blur-to-sharp and blurto-blur settings, respectively. SSMB’s viewpoint repeatability remains stable around 64–67% across all difficulty levels and both settings, while its illumination repeatability stays above 88% under the blur-to-sharp setting (peaking at 88.57% on EASY) and above 84% under the blur-to-blur setting (peaking at 84.55% on TOUGH). This consistency confirms that the illumination-viewpoint trade-off discussed above originates from the blur consistency training objective itself, rather than from any particular difficulty level or evaluation setting. Compared against BALF, the only other deblur-free method in this evaluation, SSMB outperforms it across all difficulty levels under both the blur-to-sharp and blur-to-blur settings on Blur-HPatches. On Blur-HPatches blur-to-blur, SSMB outperforms BALF by 3.72 on EASY, 5.90 on HARD, and 6.77 on TOUGH, a margin that widens with increasing blur severity. This widening margin reflects the benefit of training directly on cross-domain consistency between sharp and blurred images, rather than regressing fixed SIFT pseudo-labels that do not adapt to blur severity.

TABLE VIII  
KEYPOINT DETECTION REPEATABILITY ON THE HPATCHES DATASET [43]. FOR EACH METHOD, WE REPORT THE AVERAGE REPEATABILITY SCORE FOR VIEWPOINT CHANGES, ILLUMINATION CHANGES, AND FOR ALL IMAGE SEQUENCES.
<table><tr><td rowspan=2 colspan=4>Reference: SharpTarget: SharpMethod           Viewpoint ↑    Illumination ↑</td></tr><tr><td rowspan=1 colspan=1>Overall ↑</td></tr><tr><td rowspan=1 colspan=1>SIFT [13]</td><td rowspan=1 colspan=1>60.29</td><td rowspan=1 colspan=1>60.44</td><td rowspan=1 colspan=1>60.36</td></tr><tr><td rowspan=1 colspan=1>Key.Net [15]</td><td rowspan=1 colspan=1>68.99</td><td rowspan=1 colspan=1>67.47</td><td rowspan=1 colspan=1>68.24</td></tr><tr><td rowspan=1 colspan=1>SuperPoint [14]</td><td rowspan=1 colspan=1>69.53</td><td rowspan=1 colspan=1>68.92</td><td rowspan=1 colspan=1>69.23</td></tr><tr><td rowspan=1 colspan=1>LF-Net [52]</td><td rowspan=1 colspan=1>68.41</td><td rowspan=1 colspan=1>73.61</td><td rowspan=1 colspan=1>70.96</td></tr><tr><td rowspan=1 colspan=1>D2-Net [16]</td><td rowspan=1 colspan=1>53.99</td><td rowspan=1 colspan=1>62.80</td><td rowspan=1 colspan=1>58.32</td></tr><tr><td rowspan=1 colspan=1>R2D2 [17]</td><td rowspan=1 colspan=1>61.68</td><td rowspan=1 colspan=1>61.93</td><td rowspan=1 colspan=1>61.80</td></tr><tr><td rowspan=1 colspan=1>DISK [18]</td><td rowspan=1 colspan=1>67.22</td><td rowspan=1 colspan=1>66.68</td><td rowspan=1 colspan=1>66.95</td></tr><tr><td rowspan=1 colspan=1>REKD [31]</td><td rowspan=1 colspan=1>80.94</td><td rowspan=1 colspan=1>76.29</td><td rowspan=1 colspan=1>78.65</td></tr><tr><td rowspan=1 colspan=1>NeSS-ST [19]</td><td rowspan=1 colspan=1>64.45</td><td rowspan=1 colspan=1>64.98</td><td rowspan=1 colspan=1>64.71</td></tr><tr><td rowspan=1 colspan=1>ALIKED [20]</td><td rowspan=1 colspan=1>72.85</td><td rowspan=1 colspan=1>70.37</td><td rowspan=1 colspan=1>71.63</td></tr><tr><td rowspan=1 colspan=1>DeDoDe v2 [23]</td><td rowspan=1 colspan=1>61.95</td><td rowspan=1 colspan=1>61.47</td><td rowspan=1 colspan=1>61.71</td></tr><tr><td rowspan=1 colspan=1>XFeat [32]</td><td rowspan=1 colspan=1>59.07</td><td rowspan=1 colspan=1>64.61</td><td rowspan=1 colspan=1>61.79</td></tr><tr><td rowspan=1 colspan=1>BALF [27]</td><td rowspan=1 colspan=1>67.21</td><td rowspan=1 colspan=1>73.51</td><td rowspan=1 colspan=1>70.28</td></tr><tr><td rowspan=1 colspan=1>SSMB (Ours)</td><td rowspan=1 colspan=1>67.50</td><td rowspan=1 colspan=1>83.11</td><td rowspan=1 colspan=1>75.17</td></tr></table>

Results on Deblur-HPatches. Tabs. XI and XII report results using SRN-DeblurNet [26] for restoration, and Tabs. XIII and XIV report results using DeblurGAN-v2 [25]. The bottom row in each table shows SSMB’s repeatability on the corresponding blurred images, without any deblurring preprocessing, for direct comparison against the deblur-then-detect pipeline. The gap between SSMB and the deblur-then-detect pipeline widens as blur severity increases, since deblurring artifacts become more pronounced under HARD and TOUGH conditions and increasingly degrade the restored image structure that downstream detectors rely on. This is most evident on TOUGH sequences, where SSMB’s overall repeatability (77.16% blur-to-sharp, 74.48% blur-to-blur) exceeds every deblur-then-detect combination, including the strongest ones, DeblurGAN-v2 with SuperPoint (58.22% under deblur-tosharp setting) and DeblurGAN-v2 with ALIKED (60.78% under deblur-to-deblur setting).

## B. Image Matching

Fig. 8 further validates the effectiveness of SSMB for image matching under motion blurred conditions. The plot on the left shows the Mean Matching Accuracy (MMA) curves for different methods at various matching thresholds, providing a comprehensive view of each method’s performance. SSMB with HardNet consistently outperforms all sparse methods, excelling in overall, illumination, and viewpoint MMA. At the strict 3-pixel threshold, SSMB with HardNet even surpasses the strongest detector-free methods, RoMa v2 and Match-Former, despite operating on a single sparse keypoint set rather than full dense correspondences.

TABLE IX  
KEYPOINT DETECTION REPEATABILITY ON THE BLUR-HPATCHES DATASET UNDER THE BLUR-TO-SHARP SETTING. SSMB ACHIEVES THE HIGHEST ILLUMINATION REPEATABILITY AND RANKS FIRST IN OVERALL REPEATABILITY AT THE TOUGH LEVEL.
<table><tr><td></td><td></td><td>EASY</td><td></td><td></td><td>HARD</td><td></td><td></td><td>TOUGH</td><td></td></tr><tr><td>Method</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>50.64</td><td>61.39</td><td>55.92</td><td>51.55</td><td>62.23</td><td>56.80</td><td>47.36</td><td>59.83</td><td>53.49</td></tr><tr><td>Key.Net [15]</td><td>58.02</td><td>62.74</td><td>60.34</td><td>50.54</td><td>59.02</td><td>54.71</td><td>39.20</td><td>50.38</td><td>44.69</td></tr><tr><td>SuperPoint [14]</td><td>66.06</td><td>65.21</td><td>65.64</td><td>61.96</td><td>62.49</td><td>62.22</td><td>51.60</td><td>54.13</td><td>52.84</td></tr><tr><td>LF-Net [52]</td><td>59.57</td><td>67.65</td><td>63.54</td><td>57.64</td><td>64.87</td><td>61.19</td><td>52.14</td><td>61.57</td><td>56.78</td></tr><tr><td>D2-Net [16]</td><td>44.30</td><td>55.31</td><td>49.71</td><td>41.60</td><td>53.19</td><td>47.30</td><td>38.39</td><td>50.47</td><td>44.32</td></tr><tr><td>R2D2 [17]</td><td>55.10</td><td>60.99</td><td>57.99</td><td>47.37</td><td>56.25</td><td>51.73</td><td>33.46</td><td>47.92</td><td>40.57</td></tr><tr><td>DISK [18]</td><td>60.42</td><td>60.05</td><td>60.24</td><td>58.43</td><td>57.91</td><td>58.18</td><td>55.81</td><td>56.13</td><td>55.97</td></tr><tr><td>REKD [31]</td><td>34.53</td><td>76.50</td><td>55.15</td><td>31.87</td><td>75.18</td><td>53.15</td><td>27.87</td><td>69.89</td><td>48.52</td></tr><tr><td>NeSS-ST [19]</td><td>57.68</td><td>58.19</td><td>57.93</td><td>56.09</td><td>56.15</td><td>56.12</td><td>53.85</td><td>53.81</td><td>53.83</td></tr><tr><td>ALIKED [20]</td><td>60.49</td><td>57.21</td><td>58.88</td><td>56.19</td><td>51.97</td><td>54.12</td><td>52.08</td><td>48.58</td><td>50.36</td></tr><tr><td>DeDoDe v2 [23]</td><td>52.71</td><td>52.85</td><td>52.78</td><td>49.17</td><td>49.66</td><td>49.41</td><td>45.94</td><td>47.64</td><td>46.78</td></tr><tr><td>XFeat [32]</td><td>53.56</td><td>56.13</td><td>54.82</td><td>50.38</td><td>52.83</td><td>51.59</td><td>48.15</td><td>51.03</td><td>49.56</td></tr><tr><td>BALF [27]</td><td>72.58</td><td>75.74</td><td>74.12</td><td>72.93</td><td>76.07</td><td>74.45</td><td>67.26</td><td>76.54</td><td>71.84</td></tr><tr><td>SSMB (Ours)</td><td>66.29</td><td>88.57</td><td>77.24</td><td>66.23</td><td>88.51</td><td>77.18</td><td>66.22</td><td>88.48</td><td>77.16</td></tr></table>

TABLE X

KEYPOINT DETECTION REPEATABILITY ON THE BLUR-HPATCHES DATASET UNDER THE BLUR-TO-BLUR SETTING. SSMB ACHIEVES THE HIGHEST ILLUMINATION AND OVERALL REPEATABILITY ACROSS ALL DIFFICULTY LEVELS.
<table><tr><td rowspan="2">Method</td><td colspan="3">EASY</td><td colspan="3">HARD</td><td colspan="3">TOUGH</td></tr><tr><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>56.67</td><td>57.31</td><td>56.99</td><td>53.15</td><td>53.85</td><td>53.49</td><td>46.23</td><td>46.63</td><td>45.94</td></tr><tr><td>Key.Net [15]</td><td>61.81</td><td>63.77</td><td>62.77</td><td>56.82</td><td>59.57</td><td>58.17</td><td>45.68</td><td>52.94</td><td>49.25</td></tr><tr><td>SuperPoint [14]</td><td>58.01</td><td>59.22</td><td>58.60</td><td>49.69</td><td>50.37</td><td>50.03</td><td>42.34</td><td>44.25</td><td>43.28</td></tr><tr><td>LF-Net [52]</td><td>52.45</td><td>68.74</td><td>60.45</td><td>51.20</td><td>67.21</td><td>59.07</td><td>49.68</td><td>66.02</td><td>57.71</td></tr><tr><td>D2-Net [16]</td><td>46.65</td><td>57.14</td><td>51.80</td><td>45.84</td><td>56.44</td><td>51.05</td><td>45.11</td><td>56.13</td><td>50.53</td></tr><tr><td>R2D2 [17]</td><td>54.10</td><td>61.00</td><td>57.49</td><td>51.87</td><td>58.87</td><td>55.31</td><td>40.88</td><td>53.05</td><td>46.86</td></tr><tr><td>DISK [18]</td><td>62.34</td><td>61.43</td><td>61.89</td><td>60.81</td><td>60.94</td><td>60.87</td><td>60.89</td><td>60.40</td><td>60.65</td></tr><tr><td>REKD [31]</td><td>34.28</td><td>71.51</td><td>52.57</td><td>31.20</td><td>66.50</td><td>48.54</td><td>27.24</td><td>61.39</td><td>44.02</td></tr><tr><td>NeSS-ST [19]</td><td>61.18</td><td>60.74</td><td>60.97</td><td>60.07</td><td>59.53</td><td>59.80</td><td>59.42</td><td>59.00</td><td>59.21</td></tr><tr><td>ALIKED [20]</td><td>64.48</td><td>59.82</td><td>62.19</td><td>61.64</td><td>57.28</td><td>59.49</td><td>58.90</td><td>56.19</td><td>57.57</td></tr><tr><td>DeDoDe v2 [23]</td><td>61.88</td><td>58.41</td><td>60.17</td><td>59.40</td><td>55.31</td><td>57.39</td><td>56.28</td><td>53.66</td><td>54.99</td></tr><tr><td>XFeat [32]</td><td>58.79</td><td>60.64</td><td>59.70</td><td>57.30</td><td>59.32</td><td>58.29</td><td>56.77</td><td>58.76</td><td>57.74</td></tr><tr><td>BALF [27]</td><td>69.44</td><td>71.56</td><td>70.48</td><td>67.13</td><td>70.22</td><td>68.43</td><td>65.90</td><td>69.60</td><td>67.71</td></tr><tr><td>SSMB (Ours)</td><td>64.60</td><td>84.14</td><td>74.20</td><td>64.66</td><td>84.34</td><td>74.33</td><td>64.75</td><td>84.55</td><td>74.48</td></tr></table>

TABLE XI

KEYPOINT DETECTION REPEATABILITY ON DEBLURRED IMAGES FROM SRN-DEBLURNET [26] UNDER DEBLUR-TO-SHARP SETTING. THE BOTTOM ROW SHOWS THE RESULTS OF SSMB ON THE CORRESPONDING BLURRED IMAGES.
<table><tr><td rowspan="2">Method</td><td colspan="3">EASY</td><td colspan="3">HARD</td><td colspan="3">TOUGH</td></tr><tr><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>53.98</td><td>59.35</td><td>56.62</td><td>52.44</td><td>58.38</td><td>55.36</td><td>47.95</td><td>59.92</td><td>53.83</td></tr><tr><td>Key.Net [15]</td><td>62.86</td><td>63.72</td><td>63.28</td><td>56.37</td><td>59.71</td><td>58.01</td><td>42.08</td><td>52.30</td><td>47.10</td></tr><tr><td>SuperPoint [14]</td><td>68.65</td><td>66.76</td><td>67.72</td><td>65.33</td><td>62.73</td><td>64.05</td><td>54.18</td><td>56.37</td><td>55.26</td></tr><tr><td>LF-Net [52]</td><td>54.40</td><td>70.31</td><td>62.22</td><td>52.52</td><td>67.53</td><td>59.90</td><td>47.77</td><td>61.93</td><td>54.73</td></tr><tr><td>D2-Net [16]</td><td>46.72</td><td>57.08</td><td>51.81</td><td>44.30</td><td>54.87</td><td>49.49</td><td>40.00</td><td>52.07</td><td>45.94</td></tr><tr><td>R2D2 [17]</td><td>58.36</td><td>62.31</td><td>60.31</td><td>52.58</td><td>58.38</td><td>55.43</td><td>37.40</td><td>49.32</td><td>43.26</td></tr><tr><td>DISK [18]</td><td>64.06</td><td>62.79</td><td>63.44</td><td>62.13</td><td>61.32</td><td>61.74</td><td>57.30</td><td>57.05</td><td>57.18</td></tr><tr><td>REKD [31]</td><td>36.08</td><td>74.96</td><td>55.18</td><td>34.88</td><td>73.52</td><td>53.87</td><td>29.96</td><td>69.73</td><td>49.50</td></tr><tr><td>NeSS-ST [19]</td><td>61.41</td><td>61.13</td><td>61.27</td><td>59.75</td><td>59.46</td><td>59.61</td><td>55.64</td><td>55.10</td><td>55.37</td></tr><tr><td>ALIKED [20]</td><td>68.29</td><td>64.11</td><td>66.24</td><td>65.77</td><td>61.18</td><td>63.51</td><td>57.14</td><td>52.71</td><td>54.96</td></tr><tr><td>DeDoDe v2 [23]</td><td>57.51</td><td>56.07</td><td>56.80</td><td>55.80</td><td>54.47</td><td>55.15</td><td>48.26</td><td>47.48</td><td>47.88</td></tr><tr><td>XFeat [32]</td><td>57.33</td><td>59.89</td><td>58.59</td><td>56.56</td><td>58.47</td><td>57.50</td><td>51.58</td><td>53.40</td><td>52.47</td></tr><tr><td>BALF [27]</td><td>72.58</td><td>75.74</td><td>74.12</td><td>72.93</td><td>76.07</td><td>74.45</td><td>67.26</td><td>76.54</td><td>71.84</td></tr><tr><td>SSMB (Ours)</td><td>66.29</td><td>88.57</td><td>77.24</td><td>66.23</td><td>88.51</td><td>77.18</td><td>66.22</td><td>88.48</td><td>77.16</td></tr></table>

TABLE XII  
KEYPOINT DETECTION REPEATABILITY ON DEBLURRED IMAGES FROM SRN-DEBLURNET [26] UNDER DEBLUR-TO-DEBLUR SETTING. THE BOTTOM ROW SHOWS THE RESULTS OF SSMB ON THE CORRESPONDING BLURRED IMAGES.
<table><tr><td></td><td colspan="3">EASY</td><td colspan="3">HARD</td><td colspan="3">TOUGH</td></tr><tr><td>Method</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>60.31</td><td>59.16</td><td>59.75</td><td>57.98</td><td>58.27</td><td>58.13</td><td>48.56</td><td>52.78</td><td>50.63</td></tr><tr><td>Key.Net [15]</td><td>62.71</td><td>63.01</td><td>62.86</td><td>59.69</td><td>61.22</td><td>60.44</td><td>46.57</td><td>55.06</td><td>50.74</td></tr><tr><td>SuperPoint [14]</td><td>67.77</td><td>64.93</td><td>66.38</td><td>64.61</td><td>61.67</td><td>63.16</td><td>49.04</td><td>50.02</td><td>49.52</td></tr><tr><td>LF-Net [52]</td><td>55.36</td><td>71.03</td><td>63.06</td><td>54.60</td><td>69.72</td><td>62.03</td><td>49.17</td><td>65.68</td><td>57.28</td></tr><tr><td>D2-Net [16]</td><td>49.05</td><td>58.32</td><td>53.60</td><td>48.58</td><td>57.57</td><td>53.00</td><td>45.70</td><td>56.33</td><td>50.93</td></tr><tr><td>R2D2 [17]</td><td>55.71</td><td>60.60</td><td>58.11</td><td>52.30</td><td>57.38</td><td>54.80</td><td>40.83</td><td>50.88</td><td>45.77</td></tr><tr><td>DISK [18]</td><td>65.02</td><td>63.15</td><td>64.10</td><td>64.13</td><td>62.76</td><td>63.46</td><td>59.52</td><td>59.40</td><td>59.46</td></tr><tr><td>REKD [31]</td><td>35.65</td><td>73.44</td><td>54.22</td><td>34.01</td><td>70.59</td><td>51.99</td><td>27.89</td><td>63.42</td><td>45.35</td></tr><tr><td>NeSS-ST [19]</td><td>63.36</td><td>62.86</td><td>63.11</td><td>62.63</td><td>61.72</td><td>62.18</td><td>59.90</td><td>58.74</td><td>59.33</td></tr><tr><td>ALIKED [20]</td><td>71.35</td><td>65.45</td><td>68.45</td><td>69.43</td><td>62.99</td><td>66.26</td><td>60.62</td><td>55.68</td><td>58.19</td></tr><tr><td>DeDoDe v2 [23]</td><td>63.73</td><td>59.54</td><td>61.67</td><td>63.24</td><td>58.45</td><td>60.89</td><td>55.10</td><td>50.65</td><td>52.91</td></tr><tr><td>XFeat [32]</td><td>60.72</td><td>62.67</td><td>61.68</td><td>60.66</td><td>61.65</td><td>61.14</td><td>57.54</td><td>58.51</td><td>58.01</td></tr><tr><td>BALF [27]</td><td>69.44</td><td>71.56</td><td>70.48</td><td>67.13</td><td>70.22</td><td>68.43</td><td>65.90</td><td>69.60</td><td>67.71</td></tr><tr><td>SSMB (Ours)</td><td>64.60</td><td>84.14</td><td>74.20</td><td>64.66</td><td>84.34</td><td>74.33</td><td>64.75</td><td>84.55</td><td>74.48</td></tr></table>

TABLE XIII

KEYPOINT DETECTION REPEATABILITY ON DEBLURRED IMAGES FROM DEBLURGAN-V2 [25] UNDER DEBLUR-TO-SHARP SETTING. THE BOTTOM ROW SHOWS THE RESULTS OF SSMB ON THE CORRESPONDING BLURRED IMAGES.
<table><tr><td></td><td></td><td>EASY</td><td></td><td></td><td>HARD</td><td></td><td></td><td>TOUGH</td><td></td></tr><tr><td>Method</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>56.28</td><td>59.03</td><td>57.63</td><td>54.02</td><td>59.10</td><td>56.52</td><td>51.29</td><td>61.88</td><td>56.50</td></tr><tr><td>Key.Net [15]</td><td>63.78</td><td>64.20</td><td>63.99</td><td>57.46</td><td>60.91</td><td>59.16</td><td>44.92</td><td>53.92</td><td>49.35</td></tr><tr><td>SuperPoint [14]</td><td>68.85</td><td>67.02</td><td>67.95</td><td>66.83</td><td>64.84</td><td>65.86</td><td>57.37</td><td>59.10</td><td>58.22</td></tr><tr><td>LF-Net [52]</td><td>55.01</td><td>70.43</td><td>62.59</td><td>53.03</td><td>67.70</td><td>60.24</td><td>47.73</td><td>62.14</td><td>54.81</td></tr><tr><td>D2-Net [16]</td><td>47.68</td><td>57.77</td><td>52.64</td><td>44.96</td><td>55.64</td><td>50.21</td><td>40.17</td><td>51.80</td><td>45.88</td></tr><tr><td>R2D2 [17]</td><td>58.60</td><td>62.38</td><td>60.46</td><td>52.72</td><td>58.74</td><td>55.68</td><td>40.55</td><td>50.38</td><td>45.38</td></tr><tr><td>DISK [18]</td><td>63.65</td><td>62.85</td><td>63.25</td><td>62.08</td><td>61.42</td><td>61.76</td><td>58.61</td><td>57.61</td><td>58.12</td></tr><tr><td>REKD [31]</td><td>35.87</td><td>75.34</td><td>55.27</td><td>34.29</td><td>74.03</td><td>53.82</td><td>30.46</td><td>71.54</td><td>50.65</td></tr><tr><td>NeSS-ST [19]</td><td>61.24</td><td>61.56</td><td>61.40</td><td>59.87</td><td>59.89</td><td>59.88</td><td>56.33</td><td>56.22</td><td>56.27</td></tr><tr><td>ALIKED [20]</td><td>68.47</td><td>65.09</td><td>66.81</td><td>66.05</td><td>62.21</td><td>64.16</td><td>59.21</td><td>55.29</td><td>57.28</td></tr><tr><td>DeDoDe v2 [23]</td><td>58.14</td><td>57.35</td><td>57.75</td><td>56.36</td><td>55.99</td><td>56.18</td><td>51.47</td><td>50.75</td><td>51.12</td></tr><tr><td>XFeat [32]</td><td>57.69</td><td>60.63</td><td>59.13</td><td>56.60</td><td>58.98</td><td>57.77</td><td>52.47</td><td>54.24</td><td>53.34</td></tr><tr><td>BALF [27]</td><td>72.58</td><td>75.74</td><td>74.12</td><td>72.93</td><td>76.07</td><td>74.45</td><td>67.26</td><td>76.54</td><td>71.84</td></tr><tr><td>SSMB (Ours)</td><td>66.29</td><td>88.57</td><td>77.24</td><td>66.23</td><td>88.51</td><td>77.18</td><td>66.22</td><td>88.48</td><td>77.16</td></tr></table>

TABLE XIV

KEYPOINT DETECTION REPEATABILITY ON DEBLURRED IMAGES FROM DEBLURGAN-V2 [25] UNDER DEBLUR-TO-DEBLUR SETTING. THE BOTTOM ROW SHOWS THE RESULTS OF SSMB ON THE CORRESPONDING BLURRED IMAGES.
<table><tr><td></td><td colspan="3">EASY</td><td colspan="3">HARD</td><td colspan="3">TOUGH</td></tr><tr><td>Method</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td><td>Viewpoint ↑</td><td>Illumination ↑</td><td>Total ↑</td></tr><tr><td>SIFT [13]</td><td>59.17</td><td>59.73</td><td>59.44</td><td>57.33</td><td>58.66</td><td>57.98</td><td>50.12</td><td>52.33</td><td>51.21</td></tr><tr><td>Key.Net [15]</td><td>62.38</td><td>63.08</td><td>62.73</td><td>59.39</td><td>61.81</td><td>60.58</td><td>49.12</td><td>56.93</td><td>52.96</td></tr><tr><td>SuperPoint [14]</td><td>67.49</td><td>65.49</td><td>66.50</td><td>64.89</td><td>62.48</td><td>63.71</td><td>51.85</td><td>52.35</td><td>52.09</td></tr><tr><td>LF-Net [52]</td><td>55.57</td><td>70.69</td><td>63.00</td><td>54.58</td><td>69.25</td><td>61.79</td><td>50.48</td><td>65.49</td><td>57.85</td></tr><tr><td>D2-Net [16]</td><td>49.40</td><td>58.62</td><td>53.93</td><td>48.65</td><td>58.08</td><td>53.29</td><td>45.69</td><td>55.97</td><td>50.74</td></tr><tr><td>R2D2 [17]</td><td>55.66</td><td>60.33</td><td>57.95</td><td>52.31</td><td>57.84</td><td>55.03</td><td>43.12</td><td>52.76</td><td>47.86</td></tr><tr><td>DISK [18]</td><td>64.33</td><td>63.37</td><td>63.86</td><td>62.98</td><td>62.74</td><td>62.86</td><td>60.06</td><td>59.66</td><td>59.86</td></tr><tr><td>REKD [31]</td><td>35.17</td><td>73.21</td><td>53.86</td><td>33.62</td><td>70.23</td><td>51.61</td><td>28.74</td><td>63.03</td><td>45.59</td></tr><tr><td>NeSS-ST [19]</td><td>62.72</td><td>62.65</td><td>62.69</td><td>62.45</td><td>61.55</td><td>62.01</td><td>59.95</td><td>58.80</td><td>59.39</td></tr><tr><td>ALIKED [20]</td><td>70.38</td><td>65.59</td><td>68.03</td><td>68.62</td><td>63.87</td><td>66.29</td><td>62.70</td><td>58.79</td><td>60.78</td></tr><tr><td>DeDoDe v2 [23]</td><td>62.48</td><td>60.17</td><td>61.34</td><td>62.20</td><td>59.25</td><td>60.75</td><td>58.64</td><td>55.23</td><td>56.97</td></tr><tr><td>XFeat [32]</td><td>59.95</td><td>62.96</td><td>61.43</td><td>59.81</td><td>61.96</td><td>60.87</td><td>58.10</td><td>59.08</td><td>58.59</td></tr><tr><td>BALF [27]</td><td>69.44</td><td>71.56</td><td>70.48</td><td>67.13</td><td>70.22</td><td>68.43</td><td>65.90</td><td>69.60</td><td>67.71</td></tr><tr><td>SSMB (Ours)</td><td>64.60</td><td>84.14</td><td>74.20</td><td>64.66</td><td>84.34</td><td>74.33</td><td>64.75</td><td>84.55</td><td>74.48</td></tr></table>

![](images/1cdfb797fd771aebd8c615b32e39575dac3fa5b04caee99368ba2f5489d58fb3.jpg)  
Fig. 8. Image matching on the Blur-HPatches dataset [27]. Left: Mean Matching Accuracy (MMA) curves at different thresholds. Solid lines denote sparse detector + non-learned matching methods, dashed lines with triangle markers (▲) denote sparse detector + learned matcher methods, and dotted lines with square markers (■) denote detector-free methods. SSMB (thick solid lines) is highlighted for clarity. Right: MMA at a 3-pixel threshold. first , second and third best results are highlighted within sparse methods only. SSMB with HardNet achieves the highest MMA among all sparse methods, and at this strict threshold also surpasses every detector-free method.

## C. Relative Pose Estimation

This section presents additional qualitative results for relative pose estimation on the ArchViz dataset [44], following the blur-to-blur setting discussed in the main paper. The figure compares qualitative matching results across multiple methods, highlighting correct matches and mismatches under motion blur and viewpoint changes. Correct matches are plotted in green, and incorrect matches (i.e., epipolar errors beyond $5 \times 1 0 ^ { - 4 } )$ in red.

Fig. 9 illustrates matching results for different methods on the same image pair. These results show that SSMB, when paired with HardNet [53] or HyNet [54], achieves more correct matches and fewer mismatches compared to existing methods, demonstrating superior robustness to motion blur and viewpoint changes. These reinforce the conclusions in the main paper, confirming SSMB’s effectiveness for relative pose estimation under challenging blur conditions.

## D. More Qualitative Results

Detection on RealBlur. Fig. 10 presents additional keypoint detection results on a real-world blurred image from the Real-Blur dataset [28], complementing the RWBI results in Fig. 6 of the main paper with a broader set of baseline comparisons. Eleven methods are shown, including four methods not in the main figure: Key.Net, REKD, DeDoDe $\mathbf { v } 2 ,$ and XFeat. SSMB consistently produces well-localized keypoints concentrated on salient image structure, while most baselines either respond sparsely in blurred regions or distribute detections without aligning to meaningful structure. The enlarged regions in Fig. 10 further highlight this advantage, where SSMB maintains dense and well-aligned detections in areas that most baselines fail to cover. The results confirm that SSMB’s blurrobust behavior generalizes across different real-world blur sources and scene types.

Matching on RealBlur. Fig. 11 presents additional qualitative feature matching results on real-world sharp-blur image pairs from the RealBlur dataset [28], complementing the result in Fig. 7 of the main paper with coverage of two indoor and two outdoor scenes. Each pair contains real camera motion blur and viewpoint changes between the sharp and blurred images, following the same setting as the main paper. SSMB is combined with HardNet [53] and MNN matching. Across all four scenes, SSMB consistently detects well-localized and repeatable keypoints from both images despite the cross-domain appearance gap, yielding dense and geometrically consistent matches under challenging real-world blur conditions.

## REFERENCES

[1] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D. Tardos, “Orb-slam3: An accurate open-source library for visual, visual–´ inertial, and multimap slam,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890, 2021.

[2] H. Li, J. Zhao, J.-C. Bazin, P. Kim, K. Joo, Z. Zhao, and Y.-H. Liu, “Hong kong world: Leveraging structural regularity for line-based slam,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 13 035– 13 053, 2023.

[3] T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, and T. Pajdla, “Benchmarking 6dof outdoor visual localization in changing conditions,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2018, pp. 8601– 8610.

[4] X. Meng, P. Hou, Z. Zhao, J. Civera, D. Cremers, H. Wang, and H. Li, “Dream-slam: Dreaming the unseen for active slam in dynamic environments,” arXiv preprint arXiv:2602.21967, 2026.

[5] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2016, pp. 4104– 4113.

[6] M. Li, D. Li, S. Hu, K. Wang, Z. Zhao, and H. Wang, “Slam-x: Generalizable dynamic removal for nerf and gaussian splatting slam,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 1132–1140.

[7] T. Sattler, T. Weyand, B. Leibe, and L. Kobbelt, “Image retrieval for image-based localization revisited.” in Proc. Brit. Mach. Vis. Conf., vol. 1, no. 2, 2012, p. 4.

[8] F. Zhu, Z. Chen, Z. Zhao, Z. Xu, H. Zhu, M. Li, C. Jiang, and J. Civera, “Mygo-splat: Multi-objective closed-loop geometric feedback for rgbonly gaussian slam,” arXiv preprint arXiv:2606.29738, 2026.

[9] C. Toft, W. Maddern, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, T. Pajdla, F. Kahl, and T. Sattler, “Long-term visual localization revisited,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 4, pp. 2074–2088, 2020.

[10] Y. Gu, S. Yan, Z. Zhao, Y. Kou, J. Luo, P. Shi, and J. Li, “Ulfloc: Unbiased landmark feature for robust visual localization with 3d gaussian splatting,” arXiv preprint arXiv:2605.04730, 2026.

[11] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, “From coarse to fine: Robust hierarchical localization at large scale,” in Proc. IEEE/CVF

![](images/13ca2ca1511d767a7f3b954fd0999a2c8b2616a9d22329ec4f7c571f5548f7e2.jpg)

![](images/d12cf205d8e68b6dd59760e1332be91e48036c26c45f25170bcdcb5e5ab08fb7.jpg)  
SIFT [13] + MNN  
DeDoDe v2 [23] + Dual-Softmax

![](images/07c7f88d1767e37a44769e6682820d74fd89aa02fe21a92da8ccadbbeaf95f6d.jpg)  
ALIKED [20] + LightGlue [51]

![](images/a8471d019c7ee355a6bdacb5bea712eb37edca5794b1c99df42d4ab2514e2d67.jpg)  
BALF [27] + HardNet [53] + MNN

![](images/ae07cc2d3d4789c35a8ed95b937670ab29234c84e066d30117b9b9a301e16167.jpg)  
SSMB (Ours) + HardNet [53] + MNN

Conf. Comput. Vis. Pattern Recog., 2019, pp. 12 708–12 717.

[12] Z. Zhao, H. Yang, B. Liao, Y. Zeng, S. Yan, Y. Gu, P. Liu, Y. Zhou, H. Li, and J. Civera, “Advances in global solvers for 3d vision,” arXiv preprint arXiv:2602.14662, 2026.

[13] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” Int. J. Comput. Vis., vol. 60, no. 2, pp. 91–110, 2004.

[14] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Selfsupervised interest point detection and description,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog. Worksh., 2018, pp. 224–236.

[15] A. B. Laguna, E. Riba, D. Ponsa, and K. Mikolajczyk, “Key.net: Keypoint detection by handcrafted and learned cnn filters,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2019, pp. 5835–5843.

[16] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler, “D2-net: A trainable cnn for joint description and detection of local features,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog.,

![](images/b51f5530cfa1d3f3d2703466330962de9bc8789a0cddb0439797981e44d14c9a.jpg)  
Key.Net [15] + AffNet [55] + HardNet [53] + MNN

![](images/3565c97e2ec3c8b61e4946597f47e711de9e2ed2cf17d2e78af64c1d3967e77a.jpg)  
SuperPoint [14] + LightGlue [51]

![](images/f542fd4670ace595791a383e13ba20308494c876954207ea4a44a1b3ab2dda0c.jpg)  
LoFTR [33]

![](images/23f2011d46c70bc53bff4376b15c2aaf725d1cc4398680216766bf03ed9d66d5.jpg)  
BALF [27] + HyNet [54] + MNN

![](images/def98c183d8d013758eb2fe53ee07f63c41a8c11707acf65590ac788cea703b3.jpg)  
SSMB (Ours) + HyNet [54] + MNN  
Fig. 9. Qualitative matching results on the ArchViz dataset [44] under the blur-to-blur setting. Each subfigure displays the number of correct matches and precision in the top-left corner. SSMB, when paired with HardNet [53] or HyNet [54], achieves more correct matches and fewer mismatches, effectively handling motion blur and viewpoint changes. Green lines indicate correct matches (epipolar error below $5 \times 1 0 ^ { - 4 } )$ and red lines indicate incorrect matches. Best viewed in color.

![](images/ad26893f7b78495b600df5fe974b5b09356fe66dbd1fc9743b22903253bd997e.jpg)  
Input

![](images/00acf0e6a0e435e24440e19f7a934f236accd1ecd9cbcecc43873ed6d64991b3.jpg)  
SIFT [13]

![](images/52d1864287c00d39008742995bc3a2596846ed9e694b41c27dea3653741040d7.jpg)  
Key.Net [15]

![](images/8342eab5ab373fa8124d0c4a2f010655d127765662a1c06d0e547566b14c4375.jpg)  
SuperPoint [14]

![](images/382831dddf72a761d45fc70e5d92a536e8b5350124a81dc075eeef88555ebcd8.jpg)  
DISK [18]

![](images/0c84f26ad057b46a0652e36082f7bcd5fc272f92382dbde75c2b44a1341b1871.jpg)  
REKD [31]

![](images/f31e774e1715a54361270e99069258dcf0ef323e6bd150bd2454f2d8ab00a2d0.jpg)  
NeSS-ST [19]

![](images/e297405799dc729526526d27a226e8af6383bc8a422ca691d4e78c5272fbcf34.jpg)  
ALIKED [20]

![](images/dc1adb7c88ea6f76df462c4091e2b4a3411c37c1ee1cab1b6598d9938a083ef0.jpg)  
DeDoDe v2 [23]

![](images/c36f2cfc81819ac690f5bb4fa746fd958c2786c1a924df0f6a1823cfcba589ff.jpg)  
XFeat [32]

![](images/2dd9be17a7322bcea9205d171db23a5fb4fcee776bc23f7d6a41a5b8ddbf1957.jpg)  
BALF [27]

![](images/ca09c85d25c3b684d597cb9604ee9d8b4f90166dbf8923f87cecc976c7aee0ce.jpg)  
SSMB (Ours)  
Fig. 10. Qualitative detection results on real-world blurred images from the RealBlur dataset [28]. SSMB detects well-localized and consistentl distributed keypoints under real camera motion blur, while several baselines either miss salient structure or produce noisy responses in heavily blurred regions The red and blue boxes mark two representative regions, shown enlarged below each image for closer comparison. Best viewed in color.

2019, pp. 8084–8093.

[17] J. Revaud, P. Weinzaepfel, C. R. de Souza, N. Pion, G. Csurka, Y. Cabon, and M. Humenberger, “R2d2: Repeatable and reliable detector and descriptor,” in Proc. Adv. Neural Inf. Process. Syst., vol. 32, 2019.

[18] M. Tyszkiewicz, P. Fua, and E. Trulls, “Disk: Learning local features with policy gradient,” in Proc. Adv. Neural Inf. Process. Syst., vol. 33, 2020, pp. 14 254–14 265.

[19] K. Pakulev, A. Vakhitov, and G. Ferrer, “Ness-st: Detecting good and stable keypoints with a neural stability score and the shi-tomasi detector,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 9544–9554.

[20] X. Zhao, X. Wu, W. Chen, P. C. Chen, Q. Xu, and Z. Li, “Aliked: A lighter keypoint and descriptor extraction network via deformable transformation,” IEEE Trans. Instrum. Meas., vol. 72, pp. 1–16, 2023.

[21] P. Gleize, W. Wang, and M. Feiszli, “Silk: Simple learned keypoints,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 22 499–22 508.

[22] F. Bellavia, Z. Zhao, L. Morelli, and F. Remondino, “Image matching filtering and refinement by planes and beyond,” arXiv preprint arXiv:2411.09484, 2024.

[23] J. Edstedt, G. Bokman, and Z. Zhao, “Dedode v2: Analyzing and¨ improving the dedode keypoint detector,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog. Worksh., 2024, pp. 4245–4253.

[24] S. Nah, T. H. Kim, and K. M. Lee, “Deep multi-scale convolutional neural network for dynamic scene deblurring,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2017, pp. 257–265.

[25] O. Kupyn, T. Martyniuk, J. Wu, and Z. Wang, “Deblurgan-v2: Deblur-

ring (orders-of-magnitude) faster and better,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2019, pp. 8877–8886.

[26] X. Tao, H. Gao, Y. Wang, X. Shen, J. Wang, and J. Jia, “Scale-recurrent network for deep image deblurring,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2018, pp. 8174–8182.

[27] Z. Zhao, “Balf: Simple and efficient blur aware local feature detector,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2024, pp. 3362–3372.

[28] J. Rim, H. S. Chwa, and S. Cho, “Real-world blur dataset for learning and benchmarking deblurring algorithms,” in Proc. Eur. Conf. Comput. Vis., 2020.

[29] C. G. Harris and M. J. Stephens, “A combined corner and edge detector,” in Alvey Vision Conference, 1988, pp. 147–151.

[30] P. R. Beaudet, “Rotational invariant image operators,” in Proc. Int. Conf. Pattern Recog., 1978, pp. 579–583.

[31] J. Lee, B. Kim, and M. Cho, “Self-supervised equivariant learning for oriented keypoint detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022, pp. 4847–4857.

[32] G. Potje, F. Cadar, A. Araujo, R. Martins, and E. R. Nascimento, “Xfeat: Accelerated features for lightweight image matching,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2024, pp. 2682–2691.

[33] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “Loftr: Detector-free local feature matching with transformers,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2021, pp. 8922–8931.

[34] A. Vaswani, N. M. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst., 2017.

[35] Q. Wang, J. Zhang, K. Yang, K. Peng, and R. Stiefelhagen, “Matchformer: Interleaving attention in transformers for feature matching,” in Proc. Asian Conf. Comput. Vis., 2022, pp. 2746–2762.

[36] H. Chen, Z. Luo, L. Zhou, Y. Tian, M. Zhen, T. Fang, D. McKinnon, Y. Tsin, and L. Quan, “Aspanformer: Detector-free image matching with adaptive span transformer,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 20–36.

[37] J. Edstedt, Q. Sun, G. Bokman, M. Wadenb¨ ack, and M. Felsberg, “Roma:¨ Robust dense feature matching,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2024, pp. 19 790–19 800.

[38] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[39] J. Edstedt, D. Nordstrom, Y. Zhang, G. B¨ okman, J. Astermark, V. Lars-¨ son, A. Heyden, F. Kahl, M. Wadenback, and M. Felsberg, “Roma¨ v2: Harder better faster denser feature matching,” arXiv preprint arXiv:2511.15706, 2025.

[40] J. Sun, W. Cao, Z. Xu, and J. Ponce, “Learning a convolutional neural network for non-uniform motion blur removal,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2015, pp. 769–777.

[41] O. Kupyn, V. Budzan, M. Mykhailych, D. Mishkin, and J. Matas, “Deblurgan: Blind motion deblurring using conditional adversarial networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2018, pp. 8183–8192.

[42] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. C. Bovik, and Y. Li, “Maxim: Multi-axis mlp for image processing,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022, pp. 5759–5770.

[43] V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk, “Hpatches: A benchmark and evaluation of handcrafted and learned local descriptors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2017, pp. 3852– 3861.

[44] P. Liu, X. Zuo, V. Larsson, and M. Pollefeys, “Mba-vo: Motion blur aware visual odometry,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 5550–5559.

[45] P. Wang, L. Zhao, Y. Zhang, S. Zhao, and P. Liu, “Mba-slam: Motion blur aware dense visual slam with radiance fields representation,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[46] K. Mikolajczyk and C. Schmid, “A performance evaluation of local descriptors,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 27, no. 10, pp. 1615–1630, 2005.

[47] X. Zhang, F. X. Yu, S. Karaman, and S.-F. Chang, “Learning discriminative and transformation covariant local feature detectors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2017, pp. 6818–6826.

[48] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superglue: Learning feature matching with graph neural networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020, pp. 4938–4947.

[49] D. Nister, “An efficient solution to the five-point relative pose problem,”´ IEEE Trans. Pattern Anal. Mach. Intell., vol. 26, no. 6, pp. 756–770, 2004.

[50] M. Fischler and R. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, pp. 381–395, 1981.

[51] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “Lightglue: Local feature matching at light speed,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 17 581–17 592.

[52] Y. Ono, E. Trulls, P. V. Fua, and K. M. Yi, “Lf-net: Learning local features from images,” in Proc. Adv. Neural Inf. Process. Syst., vol. 31, 2018.

[53] A. Mishchuk, D. Mishkin, F. Radenovic, and J. Matas, “Working hard to´ know your neighbor’s margins: Local descriptor learning loss,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[54] Y. Tian, A. Barroso Laguna, T. Ng, V. Balntas, and K. Mikolajczyk, “Hynet: Learning local descriptor with hybrid similarity measure and triplet loss,” Proc. Adv. Neural Inf. Process. Syst., vol. 33, pp. 7401– 7412, 2020.

[55] D. Mishkin, F. Radenovic, and J. Matas, “Repeatability is not enough: Learning affine regions via discriminability,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 284–300.

[56] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, “Netvlad: Cnn architecture for weakly supervised place recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2016, pp. 5297–5307.

[57] K. Zhang, W. Luo, Y. Zhong, L. Ma, B. Stenger, W. Liu, and H. Li, “Deblurring by realistic blurring,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020, pp. 2737–2746.

![](images/7e1fb86528892fe96899c4078a6e476730d81776a2f31a91cfa576ef272f2383.jpg)  
Fig. 11. Qualitative matching results on real-world sharp-blur image pairs from the RealBlur dataset [28]. Each row shows a matched sharp-blur pair (left: sharp; right: blurred) with real camera motion blur and viewpoint changes. SSMB detects well-localized and repeatable keypoints across diverse indoo and outdoor scenes despite the cross-domain appearance gap. Green lines indicate correct matches. Best viewed in color.