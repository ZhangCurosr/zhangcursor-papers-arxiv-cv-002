# NaCR: Visual Localization via NeRF-aided Camera Ray Regression

Yesheng Zhang, Xiang Dai, Xu Zhao, Chongyang Zhang

Shanghai Jiao Tong University

Abstract. Visual localization (VL) is a fundamental technology for vision applications such as virtual reality. Recently, a novel VL paradigm, Camera Ray Regression (CRR), has emerged, which maps 2D image patches to 3D camera rays, but its accuracy is limited. To improve CRR accuracy, we notice a compelling duality: the inverse of this mapping is inherently performed by the novel view synthesis model, i.e., Neural Radiance Fields (NeRF). While NeRF renders image patches from camera rays via diferentiable ray marching, CRR predicts the rays from image patches. Motivated by this complementary relationship, we propose NeRF-aided Camera Ray Regression (NaCR), a unified framework that seamlessly bridges NeRF and CRR at the ray level. First, NaCR incorporates three simple yet efective enhancements into the CRR baseline. Second, leveraging a pre-trained NeRF, NaCR augments the training data by synthesizing novel views tailored for eficient, patch-level consumption. Finally, exploiting the diferentiability of NeRF, NaCR forms a closed-loop supervision pipeline where photometric rendering errors are back-propagated to optimize the predicted camera rays. To ensure stable convergence within the highly non-convex image space, we introduce a two-stage training curriculum. Extensive experiments across indoor and outdoor benchmarks demonstrate that NaCR achieves competitive accuracy. Comprehensive ablation studies validate the eficacy of each proposed component.

Keywords: Visual Localization · Camera Ray · NeRF

## 1 Introduction

Visual Localization (VL) targets estimating the 6-Degree-of-Freedom camera pose for a given query image in a known scene. It has a variety of applications, such as SLAM/SfM [27], Virtual Reality (VR), and Augmented Reality (AR) [5]. Although it is a long-standing research area, achieving high-precision visual localization remains a consistent pursuit.

In the era of deep learning, end-to-end VL methods employ neural networks as implicit maps to predict camera poses for query images. They are computationally more eficient than conventional VL methods [31, 32], which require storing explicit 3D maps. As a representative end-to-end paradigm, Scene Coordinate Regression (SCR) [4] introduces an over-parameterization trick for VL. It first estimates dense scene coordinates corresponding to image patches, and then calculates the pose based on RANSAC-PnP methods. Among the SCR series, ACE [3] achieves eficient model optimization through Gradient Decorrelation Training (GDT), yielding precise results. Nevertheless, the spatial distribution of 3D scene points exhibits high variance (especially background points), leading to unstable training. Thus, SCR methods [10,21,37] heavily rely on depth priors and hyperparameter tuning to mask out unstable predictions during training.

![](images/ffd27dd59f7382ed827e0a9c2911d2f50e3b38f02e35186bc0d7e6bff0af9659.jpg)  
Fig. 1: Camera Ray Regression (CRR) and Neural Radiance Fields (NeRF) Represent Two Complementary Processes: CRR learns to map an image patch to a camera ray , while NeRF renders an image patch from a camera ray . Recognizing this synergy, we introduce NeRF into the CRR training pipeline, creating an end-to-end diferentiable loop. In this loop, photometric rendering error provides a direct guidance for CRR optimization. Additionally, NeRF allows for generating synthetic training image patches from arbitrary rays, serving as a powerful patch-level data augmentation technique.

To eliminate this hyperparameter restriction, DIMM [43] proposes a novel VL paradigm, i.e., Camera Ray Regression (CRR). In this paradigm, the model regresses a camera ray for each image patch. As a patch-level pose representation, camera rays are naturally over-parameterized and compatible with GDT. Bounded ray errors yield robust training dynamics, thus eliminating the need for depth priors. After ray regression, the camera pose can be recovered by two linear solvers for rotation and translation, respectively. This decoupled approach is more robust than the PnP solvers in SCR. Collectively, these benefits of camera rays enable CRR-based methods [16, 43] to achieve promising VL accuracy in outdoor scenes. However, their accuracy remains limited, particularly in indoor scenes, possibly due to constrained training scalability and reliance on purely geometric supervision.

Early on, camera rays were utilized as the medium for rendering in the novel view synthesis (NVS) model: NeRF [25]. Its Ray Marching process establishes a diferentiable pathway from rays (poses) to NeRF and further to images, which can be integrated in VL inference for pose optimization [18, 22, 46]. However, recalling CRR, we recognize that it performs the exact inverse operation (see Fig. 1): a learned, diferentiable mapping from an image patch back to its corresponding camera ray. By integrating NeRF into the CRR training, the gradient propagation can realize a closed loop from images to rays and back to images. Then, the photometric error from NeRF’s rendering provides a powerful, self-supervised signal that directly optimizes the CRR network predictions. In essence, this provides a mechanism to align the implicit scene representation learned by the CRR model with the volumetric scene representation encoded in NeRF. This complementary relationship is beneficial for improving CRR accuracy. Furthermore, this integration is highly practical. Because NeRF renders on a ray-level basis, it seamlessly integrates with the patch-level, ray-based GDT [3] of CRR [43].

![](images/9c6fc5f16d820a3129c1324a1175af59bbf77b50f67a5bd7b1c03f36b5410f08.jpg)  
Fig. 2: The Improved DIMM. Our work introduces three simple yet efective enhancements to DIMM. ❶ We substitute the original DINOv2 trained on 2D tasks with the DA3-DINOv2 backbone pre-trained for 3D reconstruction, which provides the model with richer geometric priors. ❷ We transform the challenging absolute camera ray regression into a more tractable task by conditioning the model on tokens from fixed reference views. ❸ We augment the DIMM output head to predict confidence for each ray. This efectively leverages the over-parameterized nature of camera rays by downweighting uncertain predictions.

The integration of NeRF also provides a native data augmentation mechanism for CRR. Since end-to-end VL requires training data of image-pose pairs, using NVS models to expand this dataset is a direct choice. Indeed, recent work has demonstrated that augmenting VL training with randomized-appearance views generated by 3D Gaussian Splatting (3DGS) [15] can significantly boost accuracy [16, 17]. As mentioned above, a pre-trained NeRF can be a supervised module in CRR training, using it for data augmentation incurs minimal overhead. Furthermore, CRR can leverage this synthetic data with superior eficiency. Unlike current pose-level training [16,17], where an entire image supports a single pose prediction [17], CRR training consumes these synthetic views at the ray (patch) level. This enables a more eficient use of the augmented data, maximizing the benefit of generated views.

Thus, we propose NeRF-aided Camera Ray Regression, namely NaCR, a novel framework that tightly integrates NeRF and CRR at the ray level to achieve accurate VL. We first introduce three simple yet efective architectural improvements to the baseline CRR model (DIMM), inspired by recent advances in ray-based 3D reconstruction [19]. Subsequently, NaCR incorporates NeRF into the CRR training, serving two synergistic purposes. 1) Data Augmentation: NaCR exploits NeRF to render novel views, which are comprehensively consumed at the patch level during CRR training. To mitigate the domain gap between real and synthetic images, we also incorporate adversarial loss in training [17]. 2) Closed-Loop Supervision: NaCR leverages the diferentiability of NeRF to enforce closed-loop consistency. This enables the patch-level photometric rendering loss to supervise ray-level regression, guiding the CRR model to align with the robust 3D scene priors embedded in NeRF. However, naively optimizing rendering loss from scratch can collapse the CRR model due to the highly non-convex nature of the image space [22]. Thus, we propose a two-stage training pipeline. In the first stage, the CRR model is optimized using standard ray-confidence loss alongside NeRF-based data augmentation. Once the CRR performance stabilizes, the second stage introduces the rendering loss. This strategy allows NeRF’s image-space supervision to gracefully fine-tune the regressed rays within a reliable local neighborhood, yielding significant accuracy gains.

In summary, our contributions are as follows:

1. We present NaCR, a novel visual localization framework that tightly integrates Neural Radiance Fields (NeRF) and Camera Ray Regression (CRR) at the ray level. NaCR harnesses NeRF as both a dense patch-level data augmenter and a diferentiable module providing closed-loop photometric supervision for ray regression.

2. We introduce three simple yet highly efective architectural enhancements over the baseline DIMM network, improving the ray regression precision.

3. Extensive experiments on indoor and outdoor benchmarks demonstrate that NaCR improves the CRR accuracy and achieves comparable performance in VL. Comprehensive ablation studies are provided to validate the eficacy of each proposed component.

## 2 Related Work

## 2.1 Visual Localization

Early localization methods [31, 32] matched query features [20, 38] to explicit 3D point-cloud maps and recovered poses with PnP. The storage cost of such maps has motivated learning-based methods [3, 10, 12, 37] that encode scenes in network parameters. These methods mainly follow three paradigms. Absolute and Relative Pose Regression (APR/RPR) directly predict compact poses from query images; they are eficient but noise-sensitive and often less precise. Scene Coordinate Regression (SCR) predicts an over-parameterized pose representation, i.e., scene coordinates for image patches. Although accurate, SCR uses unbounded coordinates, which can destabilize training and require depth priors for outlier filtering. Camera Ray Regression (CRR) instead predicts camera rays, whose bounded errors enable smoother training without carefully tuned hyperparameters. We build on CRR and use camera rays as a geometric bridge to incorporate Neural Radiance Fields (NeRF), providing both synthetic data augmentation and robust image-level supervision.

## 2.2 Camera Ray Representation

Camera ray representation lifts a compact camera pose matrix to an imagealigned ray map. This dimensional consistency makes rays useful across posecentric 3D vision tasks. In novel view synthesis (NVS), rays serve as view conditions [9, 11, 45] or rendering media [25, 36]; in dense prediction, ray maps are often used as targets [33, 40, 42]. Recent large-scale 3D reconstruction models [19, 23, 44] show that data-driven models can learn generalizable mappings from image patches to camera rays. Camera rays also exhibit favorable convergence in optimization-based 3D reconstruction [28, 29]. For VL, rays share patch-level separability with scene coordinates [16], enabling scene-wise Gradient Decorrelation Training and competitive localization performance [43]. We leverage this geometric and dimensional alignment to closely couple CRR with a NVS rendering pipeline (NeRF).

![](images/93d43a0535ed782f4761a0373db326839d5852893898c66c24e1e09770dbee47.jpg)  
Fig. 3: The two-stage NeRF-aided CRR training pipeline. (a) Stage 1 trains CRR with patch-level Gradient Decorrelation Training (GDT) on real and NeRF-synthesized patches, while adversarial feature alignment bridges the render-to-real gap. (b) Stage 2 adds a diferentiable rendering loss, using the photometric error between rendered and original patches as a closed-loop signal for ray refinement.

## 2.3 Novel View Synthesis in Visual Localization

Novel View Synthesis (NVS) models such as NeRF [25] and 3DGS [15] can render high-fidelity images from arbitrary viewpoints, providing dense imagepose pairs for VL training. Their diferentiability also allows photometric errors from rendered images to update pose parameters. NVS has therefore been used in VL in two mostly decoupled roles. 1) Matching-based methods integrate NVS into pose refinement [22]: CrossFire [26] uses NeRF for iterative rendering and pose optimization, and NeRFMatch [46] propagates photometric errors to a pose optimization module. 2) End-to-end methods use NVS for scene-specific data augmentation: RAP [17] and GRLoc [16] render appearance-varied novel views with 3DGS to enrich training data and improve localization. Our method also introduces NeRF into VL, but difers by using camera rays to tightly couple NeRF with CRR. Rather than treating NeRF only as an ofline augmentation engine or post-hoc refinement module, we exploit both its data generation and ray-level diferentiability during CRR training.

## 3 Methodology

## 3.1 Preliminaries

Camera Ray Regression (CRR). A camera ray is defined as the ray originating from the camera center and passing through an image pixel [42]. A minimum of two camera rays can determine the camera origin, while the direction of a camera ray can establish the camera rotation. Therefore, the set of camera rays determined by the centers of uniformly divided image patches (i.e., the ray map) can serve as an over-parameterized representation of the camera pose. Recently, this representation has proven pivotal in 3D vision tasks such as Novel View Synthesis [9, 11, 45], 3D Reconstruction [19, 23, 44], and Visual Localization [16, 43]. DIMM [43] pioneers the CRR paradigm by designing an end-to-end framework to estimate patch-level camera rays for query images in visual localization. Specifically, for an input image $\boldsymbol { I } \in \mathbb { R } ^ { \breve { W } \times H \times 3 }$ , it first extracts patch-level image feature tokens $\mathcal { T } _ { i } \in \bar { \mathbb { R } } ^ { D }$ , where D is the feature dimension, corresponding to image patches $p _ { i } \in \mathbb { R } ^ { s \times s \times 3 }$ of size s, with $i = \{ 1 , \ldots , N \}$ and $N = W H / s ^ { 2 }$ . Then, the model regresses the patch-level camera rays $r _ { i } \in \mathbb { R } ^ { 6 }$ Each ray has six parameters. The first three are ray direction $r d _ { i } \in \mathbb { R } ^ { 3 }$ and $| | r d _ { i } | | _ { 2 } ~ = ~ 1$ . The last three are ray moment $r m _ { i } \in \mathbb { R } ^ { 3 }$ . After obtaining the predicted rays $\begin{array} { r } { \hat { \mathcal { R } } = \{ r _ { i } \} _ { i = 1 } ^ { N } , } \end{array}$ rotation and translation can be solved separately through two linear problems [42], thereby achieving VL for the query image.

Baseline DIMM. The DIMM model primarily consists of a feature extractor E and a ray regression model F. It first employs a fine-tuned DINOv2 as the feature extractor to acquire patch-level feature tokens $\{ \mathcal { T } _ { i } \} _ { i = 0 } ^ { N } ,$ each formed by concatenating local features, global features, and image-location features [43]:

$$
\{ \mathcal { T } _ { i } \} _ { i = 0 } ^ { N } = \mathcal { E } ( I ) ,\tag{1}
$$

Then, for each feature token, the corresponding ray is achieved by:

$$
r _ { i } = \mathcal { F } ( \mathcal { T } _ { i } ) .\tag{2}
$$

F includes a parallel multi-mapper and a semantic attention block. Please refer to [43] for more details. During training, DIMM maps a single scene via Gradient Decorrelation Training (GDT) to model the spatial information of camera rays into F. In localization, it performs patch-level ray regression on the target image and then obtains the camera pose using two decoupled RANSAC-based solvers. Our method builds upon this, making three simple improvements to the original DIMM architecture and introducing NeRF into the training pipeline.

## 3.2 Improved DIMM

Next, we elaborate our three improvements over the baseline DIMM (Fig. 2): the utilization of a DA3-pretrained DINOv2 backbone, Reference Token conditioning, and a Ray-Confidence head.

Pretrained DINOv2 from DA3. Recently, the Depth Anything 3 (DA3) [19] framework demonstrated highly accurate and generalized camera ray regression with a concise DINOv2 backbone. Trained on massive data, DA3 achieves robust geometric priors of camera rays. Thus, transferring this knowledge for scene-wise mapping is beneficial for VL. However, since the DA3’s DPT decoder predicts rays at the image level, it is fundamentally incompatible with the spatially decoupled GDT [3], which is essential for fast convergence and high accuracy on individual scenes. Therefore, to harness the powerful priors from DA3 without sacrificing the eficiency of GDT, we adopt a hybrid approach: using the pre-trained DINOv2 backbone from DA3 while retaining DIMM’s patch-level decoder.

![](images/391b0d12f2125b9581c33e70e6170cf2fb8b283bfa4e00462ab417ff6b42fb4c.jpg)  
Fig. 4: Qualitative results in 7-Scenes. The top row compares the ground truth image with a view rendered from our estimated camera pose, which validates the precision of our method. The bottom row visualizes the predicted and GT camera rays within the 3D scene.

Reference Token Conditioning. The CRR task involves estimating absolute ray parameters within a fixed scene. Thus, the model needs to learn the scene’s reference coordinate system during training to fit the ground truth. However, our DINOv2 backbone from DA3, pre-trained for multi-view reconstruction, naturally operates in a relative frame of reference, conditioning its features on a reference view. Inspired by recent RPR methods [2], we fix the reference frame by inputting F reference images to the DINO feature extractor. Thus, the query image features can be computed relative to the fixed reference images, and each target patch token $\mathcal { T } _ { i }$ is concatenated with its corresponding reference tokens $\{ \mathcal { T } _ { i } ^ { j } \} _ { j = 0 } ^ { F }$ . This provides the CRR model with explicit reference information, thereby simplifying its learning task.

$$
r _ { i } = \mathcal { F } ( \mathcal { T } _ { i } \mid \{ \mathcal { T } _ { i } ^ { j } \} _ { j = 0 } ^ { F } ) ,\tag{3}
$$

where i is the patch index and j is the reference image index. In practice, we empirically use 3 reference images by clustering the training poses and choosing the 3 cluster centroids.

Ray Confidence Head. DIMM directly predicts ray parameters without considering their confidence, forcing the model to treat all predictions as equally reliable. To address this limitation, and inspired by DA3 [19], we introduce a ray confidence head to DIMM. This mechanism efectively leverages the overparameterized nature of camera rays by actively down-weighting the influence of uncertain predictions, leading to a more robust final pose estimate. At inference time, the predicted confidence scores are also used to weight the RANSAC-based ray solver [19, 43], enhancing the robustness of the final pose.

## 3.3 NeRF-aided CRR

In this section, we introduce the NeRF integration of NaCR, which serves two primary purposes: providing dense data augmentation via novel view synthesis, and enabling self-supervised ray refinement through a diferentiable closed-loop renderer. To efectively achieve this, we address two critical challenges, i.e., the render-to-real domain gap and non-convex photometric ambiguities, by introducing adversarial feature alignment and a robust patch-level rendering loss within a two-stage training paradigm.

Motivation and Challenges. While recent works [16, 17] have shown that integrating NVS models for data augmentation in VL significantly improves accuracy, this benefit can be amplified in the context of CRR. As CRR treats each camera ray as an independent sample [43], a single rendered image provides a dense set of training data, maximizing data eficiency. Furthermore, the integration extends beyond augmentation. NeRF’s ray-marching rendering is natively compatible with CRR’s output, allowing us to connect the two models. This creates a diferentiable, closed-loop pipeline where photometric rendering loss can be back-propagated to provide a powerful supervisory signal for the ray regressor. This signal guides the CRR model to become consistent with the implicit 3D scene representation encoded in NeRF. However, a naive integration faces two major challenges: the render-to-real gap, and the non-convex photometric ambiguities.

Bridging the Render-to-Real Gap. Firstly, the rendered images inherently contain noise compared to real images, which negatively impacts VL model training. To mitigate this, we follow RAP [17], using generative adversarial training to align the features generated from real and rendered images. Since we use the pre-trained DA3 DINO backbone, fine-tuning all its parameters is computationally expensive and risks destroying its semantic priors. Hence, we only unfreeze the last two attention layers of the DINO backbone and train them via the Adversarial Loss, allowing robust adaptation.

Patch-level Rendering Loss. Secondly, the rendering loss is evaluated entirely within the highly non-linear RGB space, which sufers from ambiguity where identical colors appear at multiple spatial locations. Directly using pixellevel diferences to supervise ray regression can lead to unstable optimization directions. To address this, we modify the rendering pipeline of NeRF. Instead of rendering only the patch center, we uniformly sample rays across the entire patch based on the predicted ray parameters. Then, the Rendering Loss is computed at the patch level, increasing its robustness compared to the pixellevel loss.

Two-stage Training Pipeline. Additionally, RGB-guided pose optimization requires a good initial solution [22]. We therefore formulate a Two-stage Training Pipeline. In the first stage, we solely use NeRF to render novel views for data augmentation. Once the localization model can predict rays reasonably well, NaCR initiates the second stage and back-propagates the rendering loss into the CRR model for fine-grained ray refinement.

Next, we elaborate our Two-stage Training Pipeline in detail.

1. NeRF Model Training and Rendering. For each scene, we first train a NeRFacto [36] model using the training images $( I \in \mathcal { T } _ { r e a l } )$ and poses $( P \in \mathcal { P } _ { r e a l } )$ . We then create a set of novel camera poses $\mathcal { P } _ { s y n }$ by applying random translation and rotation noise $\delta _ { t } , \delta _ { r }$ . Simultaneously, we perturb the image appearance embeddings [36] to generate visual variations. Using these poses and appearance codes, we generate a synthetic image dataset $( \{ \mathcal { T } _ { s y n } , \mathcal { P } _ { s y n } \} )$ . To avoid introducing excessive noise from low-quality novel views, we filter the synthetic dataset using BRISQUE scores.

2. Training Stage 1 $( F i g . \ 3 a )$ . We train the CRR model on a 2:1 mixture of real and synthetic data using the GDT scheme. During each step, image patch features are passed to two heads: the Ray Regression Model that predicts rays and their confidence, and the Discriminator that classifies the feature domain. The model is optimized using only the Ray-Confidence loss on the regression output and the Adversarial loss driven by the Discriminator.

3. Training Stage $\mathcal { Q } \ ( F i g . \ 3 b )$ . This stage introduces the Rendering Loss to provide direct image-space supervision. While retaining the data sampling strategy and loss functions from Stage 1, the predicted ray parameters are now also fed into the NeRFacto renderer. This enables the computation of the photometric loss between the rendered image patch and the ground-truth image patch. This loss is then back-propagated through NeRFacto and added to the existing Stage 1 objectives, jointly updating the Ray Regression Model.

Finally, benefiting from NeRF’s rendering capabilities, we can feed the model’s predicted poses into NeRFacto to render RGB and Depth. We then calculate the relative pose to the query image via 2D-3D matching to obtain a refined localization result, following [16, 17], denoted as NaCR+NPR.

## 3.4 Training Loss

Ray-Confidence Loss This loss optimizes the geometric accuracy of regressed rays. For each ray $r _ { i } ,$ the training objective is:

$$
L _ { R C } = c _ { i } ^ { d } | | r d _ { i } ^ { g t } - \hat { r d } _ { i } | | _ { 2 } ^ { 2 } + c _ { i } ^ { m } | | r m _ { i } ^ { g t } - r \hat { m } _ { i } | | _ { 2 } ^ { 2 } - \alpha \log ( c _ { i } ^ { m } c _ { i } ^ { d } )\tag{4}
$$

where $r d _ { i } ^ { g t } , r m _ { i } ^ { g t }$ are the ground-truth ray direction and moment parameters; $\hat { r d } _ { i } , r \hat { m } _ { i }$ are the prediction; $c _ { i } ^ { d } , c _ { i } ^ { m }$ are the predicted confidence; and α is a hyperparameter scaling the confidence regularization (typically set to 0.1).

Adversarial Loss This loss encourages the feature extractor to generate domain invariant features for both real and rendered images, training the Discriminator simultaneously. For any input image I sampled from the sets of real $\left( \mathcal { T } _ { r e a l } \right)$ or synthetic images $( \mathcal { T } _ { s y n } )$ , the loss function is:

$$
L _ { A } = | \mathcal { D } ( \mathcal { E } ( I ) ) - y | _ { 1 }\tag{5}
$$

where D is the Discriminator, E represents the DINOv2 extractor specifically including the last two trainable attention layers, and y is the domain label $( y = 1$ if $I \in \mathcal { T } _ { r e a l }$ , otherwise $y = 0 )$

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Chess(cm/°)</td><td rowspan=1 colspan=1>Fire(cm/°)</td><td rowspan=1 colspan=1>Heads(cm/°)</td><td rowspan=1 colspan=1>Office(cm/°)</td><td rowspan=1 colspan=1>[Pumpkin|K(cm/°)</td><td rowspan=1 colspan=1>itchen|(cm/°)</td><td rowspan=1 colspan=1>Stairs(cm/°)</td><td rowspan=1 colspan=1>Average(cm/°)</td></tr><tr><td rowspan=2 colspan=1>SCR</td><td rowspan=2 colspan=1>DSAC* [4]ACE [3]GLACE [37]</td><td rowspan=1 colspan=1>0.5/0.20.5/0.2</td><td rowspan=1 colspan=1>0.8/0.30.8/0.3</td><td rowspan=1 colspan=1>0.5/0.30.6/0.3</td><td rowspan=1 colspan=1>1.2/0.31.0/0.3</td><td rowspan=1 colspan=1>1.2/0.31.0/0.2</td><td rowspan=1 colspan=1>0.7/0.20.8/0.2</td><td rowspan=1 colspan=1>2.7/0.82.9/0.8</td><td rowspan=1 colspan=1>1.1/0.31.1/0.3</td></tr><tr><td rowspan=1 colspan=1>0.6/0.2</td><td rowspan=1 colspan=1>0.9/0.3</td><td rowspan=1 colspan=1>0.5/0.3</td><td rowspan=1 colspan=1>1.1/0.3</td><td rowspan=1 colspan=1>0.9/0.2</td><td rowspan=1 colspan=1>0.8/0.2</td><td rowspan=1 colspan=1>3.2/0.9</td><td rowspan=1 colspan=1>1.2/0.3</td></tr><tr><td rowspan=1 colspan=1>RPR</td><td rowspan=1 colspan=1>ExReNet [39]Reloc3r [8]</td><td rowspan=1 colspan=1>|5.0/1.63.0/0.9</td><td rowspan=1 colspan=1>7.0/2.5|3.0/0.8</td><td rowspan=1 colspan=1>3.0/2.7|1.0/1.0</td><td rowspan=1 colspan=1>6.0/1.84.0/0.9</td><td rowspan=1 colspan=1>7.0/2.06.0/1.1</td><td rowspan=1 colspan=1>|7.0/2.1|4.0/1.3</td><td rowspan=1 colspan=1>19.0/4.97.0/1.3</td><td rowspan=1 colspan=1>8.0/2.54.0/1.0</td></tr><tr><td rowspan=3 colspan=1>APR</td><td rowspan=1 colspan=1>marepo [6]</td><td rowspan=1 colspan=1>|2.1/1.2</td><td rowspan=1 colspan=1>2.3/1.4 </td><td rowspan=1 colspan=1>1.8/2.0 ||</td><td rowspan=1 colspan=1>2.8/1.3</td><td rowspan=1 colspan=1>3.5/1.5</td><td rowspan=1 colspan=1>|4.2/1.7 |</td><td rowspan=1 colspan=1>5.6/1.7|</td><td rowspan=1 colspan=1>3.2/1.5</td></tr><tr><td rowspan=2 colspan=1>RAPGs[17]GRLocGs [16]</td><td rowspan=2 colspan=1>1.0/0.81.0/0.3</td><td rowspan=2 colspan=1>6.0/3.42.9/1.0</td><td rowspan=1 colspan=1>4.0/5.5</td><td rowspan=1 colspan=1>5.0/1.9</td><td rowspan=1 colspan=1>4.0/1.7</td><td rowspan=1 colspan=1>7.0/2.1</td><td rowspan=2 colspan=1>9.0/2.13.4/0.8</td><td rowspan=2 colspan=1>5.0/2.52.8/0.8</td></tr><tr><td rowspan=1 colspan=1>1.4/0.9</td><td rowspan=1 colspan=1>|5.1/1.1</td><td rowspan=1 colspan=1>2.3/0.5</td><td rowspan=1 colspan=1>3.1/0.7</td></tr><tr><td rowspan=2 colspan=1>CRR</td><td rowspan=2 colspan=1>DIMM [43]   |NaCRNeRF (Ours)</td><td rowspan=2 colspan=1>2.3/0.90.9/0.21</td><td rowspan=2 colspan=1>|2.6/1.2.0/0.22</td><td rowspan=2 colspan=1>2.0/1.3|1.3/0.4</td><td rowspan=2 colspan=1>5.7/1.04.7/0.6</td><td rowspan=1 colspan=1>2.7/0.8</td><td rowspan=2 colspan=1>3.2/0.81.9/0.33</td><td rowspan=2 colspan=1>6.2/1.1.8/0.62</td><td rowspan=2 colspan=1>|3.5/1.0.2/0.4</td></tr><tr><td rowspan=1 colspan=1>1.7/0.2</td></tr><tr><td rowspan=5 colspan=1>NPR</td><td rowspan=2 colspan=1>CrossFireNeRF[26]NeRFMatchNeRF [46]</td><td rowspan=2 colspan=1>1.0/0.41.0/0.3</td><td rowspan=1 colspan=1>5.0/1.9</td><td rowspan=1 colspan=1>|3.0/2.3</td><td rowspan=1 colspan=1>5.0/1.6</td><td rowspan=1 colspan=1>3.0/0.8</td><td rowspan=1 colspan=1>2.0/0.8</td><td rowspan=1 colspan=1>12.0/1.9</td><td rowspan=1 colspan=1>4.4/1.4</td></tr><tr><td rowspan=1 colspan=1>1.1/0.4</td><td rowspan=1 colspan=1>1.3/0.9</td><td rowspan=1 colspan=1>3.1/0.9</td><td rowspan=1 colspan=1>2.2/0.6</td><td rowspan=1 colspan=1>1.0/0.2</td><td rowspan=1 colspan=1>9.3/1.7</td><td rowspan=1 colspan=1>2.7/0.7</td></tr><tr><td rowspan=3 colspan=1>RAPGs [17]GRLocGs [16]NaCRNeRF (Ours)</td><td rowspan=3 colspan=1>0.3/0.10.3/0.10.2/0.1</td><td rowspan=3 colspan=1>0.5/0.20.3/0.10.2/0.6</td><td rowspan=2 colspan=1>0.4/0.30.2/0.1</td><td rowspan=1 colspan=1>0.6/0.1</td><td rowspan=1 colspan=1>0.8/0.2</td><td rowspan=1 colspan=1>0.5/0.1</td><td rowspan=1 colspan=1>1.1/0.3</td><td rowspan=1 colspan=1>0.6/0.2</td></tr><tr><td rowspan=1 colspan=1>0.6/0.2</td><td rowspan=1 colspan=1>0.8/0.2</td><td rowspan=1 colspan=1>0.2/0.1</td><td rowspan=2 colspan=1>0.6/ 0.20.7/ 0.2</td><td rowspan=2 colspan=1>0.5/0.10.3/0.2</td></tr><tr><td rowspan=1 colspan=1>0.3/0.1</td><td rowspan=1 colspan=1>0.4/0.3</td><td rowspan=1 colspan=1>0.3/0.2</td><td></td></tr></table>

Table 1: Localization accuracy on 7-Scenes [35] using SfM poses as ground truth. We report median position errors in cm and orientation errors in degrees (<sup>◦</sup>). The best results within the CRR and NPR groups, together with NVS-aided methods , are highlighted. “Method $N e R F / G S ^ { " }$ indicates that “Method” uses NeRF/GS-based nove view synthesis for data augmentation.

Rendering Loss The rendering loss is the photometric L2 distance between the color rendered by NeRF from a predicted ray and the ground-truth color of the input image patch. For each ray with image center $( u , v )$ and patch size $s \times s ,$ the rendering loss is defined as:

$$
\mathcal { L } _ { R } = \sum _ { x = u - s / 2 } ^ { u + s / 2 } \sum _ { y = v - s / 2 } ^ { v + s / 2 } \Big \| \hat { \mathbf { I } } ( x , y ) - \mathbf { I } ( x , y ) \Big \| _ { 2 } ^ { 2 }\tag{6}
$$

where $\hat { \bf \cal I } ( x , y )$ is the image color rendered by NeRFacto at coordinates $( x , y )$ and I denotes the input image.

In summary, the total objective for Stage 1 optimization is $L = \beta L _ { R C } + \gamma L _ { A }$ For Stage 2 fine-tuning, the overall multi-task objective becomes $L = \beta L _ { R C } +$ $\gamma L _ { A } + \mu L _ { R }$ . The $\beta , \gamma$ and $\mu$ are hyperparameters that balance the contribution of each loss term.

## 4 Results

## 4.1 Implementation Details

For each scene, we construct a synthetic dataset by first training a Nerfacto model [36] and subsequently rendering images from sampled novel-view poses. Detailed descriptions of the Nerfacto training process and the pose sampling strategy are provided in our Suppl. The two-stage training of the NaCR takes place on an NVIDIA A800 GPU. To accommodate the pre-trained DA3 DINO backbone, we fix the image resolution to $5 0 4 \times 3 7 8$ and the patch size to 14, predicting the ray parameters of the patch center. The parameters of the last two layers of the DA3 DINO backbone are unfrozen for updates. For our ray regression model, we adopt the DIMM decoder but replace its original CLS Token with the Camera Token from DA3 DINOv2 backbone, which also suficiently summarizes the global image content.

<table><tr><td rowspan="2">Scene</td><td colspan="2">SCR</td><td colspan="4">APR</td><td colspan="3">CRR</td></tr><tr><td>DSAC* [4]</td><td>ACE [3]</td><td>PN [12–14]</td><td>MST [34]</td><td>marepo [6]</td><td>marepos [6]</td><td>DIMM [43]</td><td>NaCR Ours</td><td>NaCR+NPR Ours</td></tr><tr><td>Throughput (fps)|</td><td>17.9</td><td>17.9</td><td>166.7</td><td>28.4</td><td>55.6</td><td>55.6</td><td>20.1</td><td>33.4</td><td>16.5</td></tr><tr><td>Bears</td><td>|82.6%/91.6%</td><td>80.7%/92.6%</td><td>12.9%/35.7%</td><td>0.5%/12.8%</td><td>80.7%/99.3% 80.7%/99.5%|</td><td></td><td>|82.4%/97.1%|</td><td></td><td>95.5%/99.3% 96.4%/99.7%</td></tr><tr><td>Cubes</td><td>83.8%/98.1% 97.0%/98.1%</td><td></td><td>0.0%/0.4%</td><td>0.00%/9.9%</td><td>72.4%/96.9%71.8%/96.9%</td><td></td><td>71.5%/89.8%</td><td>72.5%/85.7%</td><td>73.8%/87.9%</td></tr><tr><td>Inscription</td><td>54.1%/69.7%</td><td>49.0%/69.6%</td><td>1.1%/6.3%</td><td>1.3%/9.7%</td><td>37.8%/74.2% 37.1%/74.1%</td><td></td><td>21.3%/72.6%</td><td></td><td>57.8%/75.1% 56.1%/77.3%</td></tr><tr><td>Lawn</td><td>34.7% /38.0%</td><td>35.8%/38.5%</td><td>0.0%/0.2%</td><td>0.0%/0.0%</td><td>32.6%/41.6%34.2%/41.1%</td><td></td><td>22.4%/46.3%</td><td></td><td>34.2%/47.5%35.1%/48.8%</td></tr><tr><td>Map</td><td>56.7% /87.1%</td><td>56.5%/84.7%</td><td>14.9%/49.1%</td><td>5.6%/25.7%</td><td>53.9%/</td><td>/87.7% 55.1%/87.9%</td><td>57.3%/89.2%</td><td>68.0%/95.2%</td><td>68.7%/96.3%</td></tr><tr><td>Square Bench</td><td>69.5%/97.9%</td><td>66.7%/97.8%</td><td>0.0%/3.0%</td><td>0.0%/0.0%</td><td>68.6%/100%70.7%/100%</td><td></td><td>54.3%/98.9%</td><td>70.9%/97.1%</td><td>72.3%/98.4%</td></tr><tr><td>Statue</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>0.0%/0.0%</td><td>1.3%/4.1%</td><td>0.5%/2.2%</td></tr><tr><td>Tendrils</td><td>25.1%/26.5%</td><td>34.9%/36.8%</td><td>0.0%/0.0%</td><td>0.9%/23.6%</td><td>27.9%/33.4%</td><td>29.3%/34.8%</td><td>24.1%/43.6%</td><td></td><td>46.8%/50.4% 47.3%/52.6%</td></tr><tr><td>The Rock Winter Sign</td><td>100%/100%</td><td>100%/100% 1.0%/7.6%</td><td></td><td></td><td>24.2%/77.5%10.7%/52.6% 98.1%/100%</td><td>99.8%/100% 0.0%/0.3%</td><td>95.4%/100%</td><td>97.9%/100%</td><td>98.6%/100%</td></tr><tr><td>Average</td><td>0.2%/5.7% |50.7%/61.5%</td><td>52.2%/62.6%</td><td>0.0%/0.0% 5.3%/17.2%</td><td>0.0%/0.0% 1.9%/13.4%</td><td>0.0%/0.7% 47.2%/63.4%</td><td>47.9%/63.5%|42.9%/64.3%</td><td>0.0%/5.4%</td><td>0.2% 8.3% 54.5%/66.3% 54.9%/67.1%</td><td>0.0% 7.6%</td></tr></table>

Table 2: Pose accuracy comparison on the outdoor Wayspots [1] dataset. Results are reported as the percentage of frames below $1 0 c m / 5 ^ { \circ }$ and $0 . 5 m / 5 ^ { \circ }$ error.

For the first training stage, we employ the AdamW optimizer with a learning rate set to $1 0 ^ { - 4 }$ and weight decay at 0.01, utilizing GDT for 40K steps. In the second stage, we incorporate the patch-level rendering loss and lower the learning rate empirically to $1 0 ^ { - 6 }$ for another 20K steps. A batch size of 5120 is consistently used during training. Empirically, the loss parameters are set to: $\beta = 1 , \gamma = 0 . 2$ , and $\mu = 0 . 1$

## 4.2 7-Scenes Results

Dataset. We first evaluate the localization accuracy on the indoor 7-Scenes [35] dataset. This dataset provides seven indoor scenes, each containing several video sequences captured from diferent trajectories. The resolution of images is low and features serious motion blur. We follow ACE [3] to train and test with accurate SfM poses.

Baselines. On this dataset, we selected methods from several diferent paradigms for comparisons: Scene Coordinate Regression (SCR) paradigm including DSAC\* [4], ACE [3], and GLACE [37]; methods based on Absolute Pose Regression (APR) [6] and Relative Pose Regression (RPR) [8, 39]; RAP [17] and GRLoc [16] which apply 3DGS for data augmentation; and finally, DIMM [43] under the Camera Ray Regression (CRR) paradigm. Besides, the NVS models can assist in post-inference optimization [26, 46], which is also feasible in our method. Following RAP, we denote these methods as NVS-based Pose Refinement (NPR) and compare their performances. We report the median translation and rotation errors across scenes.

Results. As shown in Tab. 1, NaCR substantially improves accuracy over DIMM, reducing the average translation error from 3.5 cm to 2.2 cm. This improvement comes from both incorporating NeRF and improving the baseline. Ablation studies of these two modules are presented in Tab. 6 and Tab. 9 of the supplementary material. Compared with methods adopting NVS models for data augmentation, i.e., RAP and GRLoc, NaCR also yields superior accuracy (e.g., GRLoc’s translation error is up to 2.8 cm, while NaCR reduces it to 2.2 cm). These results highlight the primary advantage of NaCR: applying an NVS model for both data augmentation and in-training supervision yields better performance than augmentation alone. While NaCR trails the top SCR-based methods on this dataset, this is expected because the bounded indoor environment minimizes SCR’s weakness, namely, instability from unbounded background points. However, leveraging NaCR’s initial pose estimates, NaCR+NPR achieves a 0.3 cm translation error, surpassing both SCR and other post-optimization methods. The qualitative results are presented in Figure 4.

Table 3: Detailed eficiency report. Mapping time is measured per scene, and inference time is measured per query image.
<table><tr><td>Component</td><td>Model Params (M)</td><td>Time</td><td>Role</td></tr><tr><td>ACE Mapping</td><td>2.10</td><td>5 min</td><td>SCR mapping</td></tr><tr><td>GLACE Mapping</td><td>9.45</td><td>33 min</td><td>SCR mapping</td></tr><tr><td>DIMM Mapping</td><td>18.31</td><td></td><td>1.32 h baseline mapping</td></tr><tr><td>NaCR Mapping Cost</td><td>39.53</td><td>3.63 h</td><td>mapping</td></tr><tr><td>↔ Scene NeRF training</td><td>16.79</td><td>0.42 h</td><td>mapping</td></tr><tr><td>→ Stage-1 training</td><td>39.53</td><td>2.08 h</td><td>mapping</td></tr><tr><td>↔ Stage-2 training</td><td>39.53</td><td>1.13 h</td><td>mapping</td></tr><tr><td>DIMM Inference</td><td>18.31</td><td>0.02 s</td><td>baseline VL</td></tr><tr><td>NaCR Inference Cost</td><td>39.53</td><td>0.03 s</td><td>VL</td></tr></table>

## 4.3 Wayspots Results

Dataset. We also conduct experiments on the outdoor Wayspots dataset [1]. Wayspots involves 10 outdoor scenes, each enclosing two video sequences dedicated separately to mapping and localization. Due to low-texture and repetitive areas within views, this dataset provides remarkable challenges for visual localization.

Baselines. In this experiment, we compare against SCR methods specifically DSAC\* [4] and ACE [3]; APR methods including PN [12–14], MST [34], and marepo [6]; as well as the CRR baseline DIMM [43]. Performance is measured using localization recall at $1 0 c m / 5 ^ { \circ }$ and $0 . 5 m / 5 ^ { \circ }$ error margins.

Results. As demonstrated in Tab. 2, NaCR yields state-of-the-art VL results across scenes. The integration of NeRF into CRR results in an overall precision boost. NaCR reaches 54.5% for the $1 0 \mathrm { c m } / 5 ^ { \circ }$ recall metric, comparing favorably with the 42.9% of baseline DIMM. Unlike SCR methods, NaCR exhibits high robustness to the unbounded backgrounds of the Wayspots dataset. This allows NaCR (54.5%) to outperform ACE (52.2%). Our NeRF-based post-optimization step (NaCR+NPR) generally provides further gains in most scenes. However, we also observe degraded optimization in a few scenes, such as “State” and “Winter Sign”. This is likely due to the prevalence of repetitive textures in these environments, which reduces matching accuracy and introduces errors into the optimization process. Qualitative results are provided in the supplementary material A.

## 4.4 Eficiency Report

We report the cost of each NaCR component in Tab. 3. The NeRF is used as a one-time ofline mapping module: it is trained once per scene and then provides both synthetic views and diferentiable rendering supervision during NaCR training. This increases the per-scene mapping time from 1.32 h for DIMM to 3.63 h for NaCR, including 0.42 h for scene NeRF training, 2.08 h for stage-1 training, and 1.13 h for stage-2 training. At localization time, NaCR keeps the feed-forward CRR inference path and does not require NeRF optimization, resulting in a comparable per-query runtime of 0.03 s. As a complementary analysis, Suppl. D provides a Mapping Time Profile that studies the accuracy– cost trade-of under diferent data amounts of rendering augmentation.

## 4.5 Ablation Study on Training Pipeline

We ablate the two-stage strategy on 7-Scenes by comparing single-stage training with and without the rendering loss. As summarized in Tab. 4, applying the rendering loss from the start destabilizes optimization (see Fig. 6 in the Suppl.) and degrades accuracy, while removing it is also sub-optimal. The two-stage pipeline is therefore critical: it first stabilizes CRR with GDT, then uses the rendering loss for local ray refinement.

## 4.6 Complete Component Ablation

We further isolate the contributions of the architectural upgrades, NeRF-based data augmentation, and patch-level rendering loss on the full 7-Scenes and Wayspots benchmarks. Table 4 reports results averaged over three random seeds. The matched pair NaCR (µ = 0) and NaCR difers only in the rendering-loss weight and therefore directly measures the efect of $\mathcal { L } _ { R }$ under the same two-stage schedule. The rendering loss improves the full model from 2.4/0.5 to 2.2/0.4 on 7-Scenes and from 52.7/65.5 to 54.1/66.1 on Wayspots, while the single-stage variant with rendering loss is substantially less stable. These results support the complementary roles of the three components.

## 5 Conclusion

This work proposes a novel Camera Ray Regression method, NaCR, for Visual Localization. Building upon a previous baseline, NaCR first introduces three simple yet efective improvements inspired by advanced camera ray-based reconstruction methods. The core technical insight of NaCR is the deep integration of NeRF and CRR, driven by the observation that camera rays hold a central role in both techniques. Specifically, NaCR exploits NeRF’s novel view synthesis capability to generate abundant pose-image pairs for ray-level synthetic data augmentation. Furthermore, taking advantage of the diferentiable rendering of NeRF, NaCR constructs a closed-loop gradient flow between the image and ray space, providing photometric supervision for ray regression. To overcome the training instability caused by the highly non-linear nature of the image space, we design a stable two-stage training scheme, leveraging rendering errors to finetune the rays locally while ensuring optimization stability. Extensive experiments on both indoor and outdoor datasets demonstrate that NaCR achieves competitive performance. Our ablation studies thoroughly validate the contribution of each module.

Table 4: Complete component ablation. Except for the two single-stage variants, all configurations use a 60K-step two-stage schedule. Results are averaged over seeds 42, 23, and 24 and across all scenes in 7-Scenes and Wayspots. “Arch.” denotes the architectural upgrades. Each entry reports translation/rotation error on 7-Scenes and recall at $1 0 \mathrm { c m } / 5 ^ { \circ }$ and $0 . 5 \mathrm { m } / 5 ^ { \circ }$ on Wayspots.
<table><tr><td>Method</td><td>Arch. NVS aug.</td><td></td><td> $\mathcal { L } _ { R }$ </td><td>TST</td><td>7-Scenes ↓</td><td>STD</td><td>Wayspots ↑</td><td>STD</td></tr><tr><td>DIMM</td><td>X</td><td>X</td><td>X</td><td>√</td><td>3.6/1.0</td><td>0.034</td><td>42.3/64.0</td><td>0.215</td></tr><tr><td> $\hookrightarrow \mathrm { D A 3 \ o n l y }$ </td><td>DA3</td><td>X</td><td>X</td><td>√</td><td>3.3/0.9</td><td>0.037</td><td>45.9/63.8</td><td>0.052</td></tr><tr><td>DIMM*</td><td>Full</td><td>X</td><td>X</td><td>√</td><td>3.2/0.9</td><td>0.019</td><td>45.8/63.7</td><td>0.555</td></tr><tr><td>NaCR  $( \mu = 0 )$ </td><td>Full</td><td>√</td><td>X</td><td>√</td><td>2.4/0.5</td><td>0.022</td><td>52.7/65.5</td><td>0.040</td></tr><tr><td> $\mathrm { D I M M ^ { * } } + \mathcal { L } _ { R }$ </td><td>Full</td><td>X</td><td>√</td><td>√</td><td>2.9/0.7</td><td>0.023</td><td>51.0/64.3</td><td>0.117</td></tr><tr><td> $\mathrm { N a C R - S S T }$ </td><td>Full</td><td>√</td><td>√</td><td>X</td><td>6.7/3.3</td><td>0.027</td><td>39.1/60.0</td><td>0.099</td></tr><tr><td> $\mathrm { N a C R - S S T } \ ( \mu = 0 )$ </td><td>Full</td><td>√</td><td>X</td><td>X</td><td>2.8/0.7</td><td>0.048</td><td>50.2/64.3</td><td>0.119</td></tr><tr><td>NaCR</td><td>Full</td><td>√</td><td>√</td><td>√</td><td>2.2/0.4</td><td>0.033</td><td>54.1/66.1</td><td>0.156</td></tr></table>

A limitation of the current work is its reliance on a pre-trained NeRF for each scene, which increases the ofline mapping cost and may limit scalability to large or changing environments. Nevertheless, this reliance may also be viewed not merely as a drawback, but as a potentially useful design choice. This raylevel interface also provides a natural path for future extension, as it can readily incorporate advances in NeRF designed for large-scale environments. In future work, we plan to build on this design to improve scalability and reduce the need for scene-specific NeRF preparation. More details can be found in our Suppl.

## References

1. Arnold, E., Wynn, J., Vicente, S., Garcia-Hernando, G., Monszpart, Á., Prisacariu, V.A., Turmukhambetov, D., Brachmann, E.: Map-free visual relocalization: Metric pose relative to a single image. In: ECCV (2022) 11, 12

2. Barroso-Laguna, A., Cavallari, T., Prisacariu, V., Brachmann, E.: A scene is worth a thousand features: Feed-forward camera localization from a collection of image features. In: International Conference on Learning Representations (ICLR) (2026) 7

3. Brachmann, E., Cavallari, T., Prisacariu, V.A.: Accelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses. In: CVPR (2023) 2, 3, 4, 6, 10, 11, 12, 22

4. Brachmann, E., Rother, C.: Visual camera re-localization from rgb and rgb-d images using dsac. IEEE transactions on pattern analysis and machine intelligence 44(9), 5847–5865 (2021) 1, 10, 11, 12

5. Brachmann, E., Wynn, J., Chen, S., Cavallari, T., Monszpart, Á., Turmukhambetov, D., Prisacariu, V.A.: Scene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer. In: ECCV (2024) 1

6. Chen, S., Cavallari, T., Prisacariu, V.A., Brachmann, E.: Map-relative pose regression for visual re-localization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20665–20674 (2024) 10, 11, 12

7. Do, T., Miksik, O., DeGol, J., Park, H.S., Sinha, S.N.: Learning to detect scene landmarks for camera localization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11132–11142 (2022) 22

8. Dong, S., Wang, S., Liu, S., Cai, L., Fan, Q., Kannala, J., Yang, Y.: Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. CVPR (2025) 10, 11

9. Jiang, H., Tan, H., Wang, P., Jin, H., Zhao, Y., Bi, S., Zhang, K., Luan, F., Sunkavalli, K., Huang, Q., Pavlakos, G.: Rayzer: A self-supervised large view synthesis model. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4918–4929 (October 2025) 4, 6, 23

10. Jiang, X., Wang, F., Galliani, S., Vogel, C., Pollefeys, M.: R-score: Revisiting scene coordinate regression for robust large-scale visual localization. arXiv preprint arXiv:2501.01421 (2025) 2, 4

11. Jin, H., Jiang, H., Tan, H., Zhang, K., Bi, S., Zhang, T., Luan, F., Snavely, N., Xu, Z.: Lvsm: A large view synthesis model with minimal 3d inductive bias. The Thirteenth International Conference on Learning Representations (2025) 4, 6

12. Kendall, A., Cipolla, R.: Modelling uncertainty in deep learning for camera relocalization. In: 2016 IEEE international conference on Robotics and Automation (ICRA). pp. 4762–4769. IEEE (2016) 4, 11, 12

13. Kendall, A., Cipolla, R.: Geometric loss functions for camera pose regression with deep learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5974–5983 (2017) 11, 12

14. Kendall, A., Grimes, M., Cipolla, R.: Posenet: A convolutional network for realtime 6-dof camera relocalization. In: Proceedings of the IEEE international conference on computer vision. pp. 2938–2946 (2015) 11, 12

15. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., et al.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023) 3, 5

16. Li, C., Ma, X., Liu, L., Li, Z., Yan, Q., Xu, Y.: Grloc: Geometric representation regression for visual localization (2025), https://arxiv.org/abs/2511.13864 2, 3, 5, 6, 8, 9, 10, 11, 20

17. Li, S., Tan, S., Chang, B., Zhang, J., Feng, C., Li, Y.: Unleashing the power of data synthesis. In: International Conference on Computer Vision (ICCV) (2025) 3, 5, 8, 9, 10, 11, 20

18. Lin, C.H., Ma, W.C., Torralba, A., Lucey, S.: Barf: Bundle-adjusting neural radiance fields. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 5721–5731 (2021) 2

19. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. ICLR (2026) 3, 5, 6, 7

20. Lindenberger, P., Sarlin, P.E., Pollefeys, M.: LightGlue: Local Feature Matching at Light Speed. In: ICCV (2023) 4

21. Lu, D., Xiao, W., Ran, T., Yuan, L., Lv, K., Zhang, J.: Attention-based accelerated coordinate encoding network for visual relocalization. In: 2024 IEEE 7th Information Technology, Networking, Electronic and Automation Control Conference

(ITNEC). vol. 7, pp. 1675–1680 (2024). https://doi.org/10.1109/ITNEC60942. 2024.10733333 2

22. Lu, S.Y., Chen, Y.Y., Wu, Y.T., Lin, H.C., Jhong, S.Y., Cheng, W.H.: Radiance field-based pose estimation via decoupled optimization under challenging initial conditions. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2664–2673. IEEE (2025) 2, 4, 5, 8

23. Lu, Y., Zhang, J., Fang, T., Nahmias, J.D., Tsin, Y., Quan, L., Cao, X., Yao, Y., Li, S.: Matrix3d: Large photogrammetry model all-in-one. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 11250–11263 (2025) 5, 6

24. Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.: Deblurnerf: Neural radiance fields from blurry images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12861–12870 (2022) 20

25. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European conference on computer vision. pp. 405–421. Springer (2020) 2, 4, 5

26. Moreau, A., Piasco, N., Bennehar, M., Tsishkou, D., Stanciulescu, B., de La Fortelle, A.: Crossfire: Camera relocalization on self-supervised features from an implicit representation. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 252–262 (2023) 5, 10, 11

27. Mur-Artal, R., Montiel, J.M.M., Tardos, J.D.: Orb-slam: A versatile and accurate monocular slam system. IEEE transactions on robotics 31(5), 1147–1163 (2015) 1

28. Murai, R., Dexheimer, E., Davison, A.J.: Mast3r-slam: Real-time dense slam with 3d reconstruction priors pp. 16695–16705 (2025). https://doi.org/10.1109/ CVPR52734.2025.01556 5

29. Pan, L., Barath, D., Pollefeys, M., Schönberger, J.L.: Global Structure-from-Motion Revisited. In: European Conference on Computer Vision (ECCV) (2024) 5

30. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction pp. 12159–12168 (2021). https://doi.org/10.1109/ICCV48922.2021.01196 21, 22

31. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From coarse to fine: Robust hierarchical localization at large scale. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12716–12725 (2019) 1, 4

32. Sarlin, P.E., Unagar, A., Larsson, M., Germain, H., Toft, C., Larsson, V., Pollefeys, M., Lepetit, V., Hammarstrand, L., Kahl, F., et al.: Back to the feature: Learning robust camera localization from pixels to pose. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3247–3257 (2021) 1, 4

33. Schops, T., Larsson, V., Pollefeys, M., Sattler, T.: Why having 10,000 parameters in your camera model is better than twelve. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2535–2544 (2020) 5

34. Shavit, Y., Ferens, R., Keller, Y.: Learning multi-scene absolute pose regression with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2733–2742 (2021) 11, 12

35. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene coordinate regression forests for camera relocalization in rgb-d images. In: 2013 IEEE Conference on Computer Vision and Pattern Recognition. pp. 2930–2937 (2013). https://doi.org/10.1109/CVPR.2013.377 10, 11

36. Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Wang, T., Kristofersen, A., Austin, J., Salahi, K., Ahuja, A., et al.: Nerfstudio: A modular framework for neural radiance

field development. In: ACM SIGGRAPH 2023 conference proceedings. pp. 1–12 (2023) 4, 8, 9, 10, 18, 20

37. Wang, F., Jiang, X., Galliani, S., Vogel, C., Pollefeys, M.: Glace: Global local accelerated coordinate encoding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21562–21571 (2024) 2, 4, 10, 11

38. Wang, Y., He, X., Peng, S., Tan, D., Zhou, X.: Eficient loftr: Semi-dense local feature matching with sparse-like speed. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21666–21675 (2024) 4

39. Winkelbauer, D., Denninger, M., Triebel, R.: Learning to localize in new environments from synthetic training data. In: 2021 IEEE International Conference on Robotics and Automation (ICRA). p. 5840–5846. IEEE Press (2021). https://doi.org/10.1109/ICRA48506.2021.9560872 10, 11

40. Xu, Y., Chen, D., Liu, K., Zakharov, S., Ambrus, R., Daniilidis, K., Guizilini, V.: Se(3) equivariant ray embeddings for implicit multi-view depth estimation. In: Proceedings of the 38th International Conference on Neural Information Processing Systems. NIPS ’24, Curran Associates Inc., Red Hook, NY, USA (2024) 5

41. Ye, V., Li, R., Kerr, J., Turkulainen, M., Yi, B., Pan, Z., Seiskari, O., Ye, J., Hu, J., Tancik, M., Kanazawa, A.: gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research 26(34), 1–17 (2025) 20

42. Zhang, J.Y., Lin, A., Kumar, M., Yang, T.H., Ramanan, D., Tulsiani, S.: Cameras as rays: Pose estimation via ray difusion. In: International Conference on Learning Representations (ICLR) (2024) 5, 6

43. Zhang, Y., Zhao, X.: Semantic-guided camera ray regression for visual localization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 25639–25648 (October 2025) 2, 3, 5, 6, 7, 8, 10, 11, 12, 22

44. Zhao, Q., Lin, A., Tan, J., Zhang, J.Y., Ramanan, D., Tulsiani, S.: Difusionsfm: Predicting structure and motion via ray origin and endpoint difusion. In: CVPR (2025) 5, 6

45. Zhao, Q., Tan, H., Wang, Q., Bi, S., Zhang, K., Sunkavalli, K., Tulsiani, S., Jiang, H.: E-rayzer: Self-supervised 3d reconstruction as spatial visual pre-training. In: CVPR (2026) 4, 6, 23

46. Zhou, Q., Maximov, M., Litany, O., Leal-Taixé, L.: The nerfect match: Exploring nerf features for visual localization. European Conference on Computer Vision (2024) 2, 5, 10, 11

## Supplementary Material

## A Qualitative Results on Wayspots

We provide qualitative results on the outdoor Wayspots dataset in Fig. 5. The top row compares each query image with the view rendered from our estimated camera pose. The close visual agreement indicates that the predicted pose is geometrically consistent with the scene representation learned by Nerfacto, even under outdoor appearance changes and repetitive structures. The bottom row further visualizes the predicted camera rays in the reconstructed scene. These ray visualizations show that NaCR produces spatially coherent ray predictions rather than only fitting the final pose after aggregation, supporting the efectiveness of coupling CRR with NeRF at the ray level.

![](images/c1e3b9d22fc96920af44e371f2b1c4c5cf5ff254d1b769d5965b4cd2d2a5fc22.jpg)  
Fig. 5: Qualitative results in Wayspots.The top row compares the ground truth image with a view rendered from our estimated camera pose. The bottom row visualizes the predicted camera rays within the 3D scene.

## B Implementation Details of Synthesis Data Augmentation

For each scene, we train a Nerfacto model [36] using the Adam optimizer for 20K steps on an NVIDIA RTX 4090 GPU, with a learning rate set to 0.01. Then, we use the Nerfacto model to generate a novel view dataset: $\{ \mathcal { T } _ { s y n } , \mathcal { P } _ { s y n } \}$ . Specifically, we compute the variation range in translation and rotation for the poses in the training set $\left( \mathcal { P } _ { r e a l } \right)$ , and compute the maximum variations for translation and rotation: $\varDelta _ { t } , \varDelta _ { r } .$ respectively. Then, the novel view poses $( P _ { s y n } \in \mathcal { P } _ { s y n } )$ are generated based on random perturbations $\delta _ { t } \in ( 0 , \varDelta _ { t } )$ and $\delta _ { r } \in ( 0 , \varDelta _ { r } )$ on the real poses. We render images at these poses and filter them using a BRISQUE score threshold of 50, resampling any low-quality views.

## C Ablation Study on Adversarial Loss

To evaluate the efectiveness of the adversarial loss, which is designed to bridge the render-to-real domain gap, we conducted an ablation study on the 7-Scenes

Table 5: Ablation Study on the adversarial loss. The median rotation and translation errors are reported.

<table><tr><td>NaCR Scene NaCR  $\mathrm { w } / \mathrm { o }$  Adversarial Loss</td></tr><tr><td>chess 0.9/0.2 1.0/0.3</td></tr><tr><td>fire 1.0/0.2 1.2/0.5</td></tr><tr><td>heads 1.3/0.4 1.7/0.6</td></tr><tr><td>office 4.7/0.6 4.9/0.9</td></tr></table>

dataset. The results, presented in Table 5, show that removing the adversarial component leads to a consistent increase in localization error. This confirms that adversarial training acts as a crucial regularizer, making our feature extractor robust to the domain shift between synthetic and real data.

<table><tr><td colspan="2">Train Loss</td><td colspan="2"></td><td rowspan="2">Treln-Loss</td></tr><tr><td>Training Stage 1</td><td>Training Stage 2 Add Render Loss</td><td>Training Stage 1</td><td>Training Stage 2 Add Render Loss</td></tr><tr><td></td><td></td><td></td><td></td><td>Single Stage Training</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Fig. 6: Training loss curves on the 7-Scenes Chess dataset, comparing our two-stage training strategy against a single-stage baseline. Our two-stage training demonstrates that while the rendering loss introduces slight instability to the training error (left), it yields superior final translation accuracy (middle). Conversely, the single-stage baseline (right), which applies the rendering loss from the start, struggles to converge with optimization instability, due to the unreliable supervision signal from the renderer when ray predictions are still coarse.

## D Mapping Time Profile

Table 6: Mapping time versus performance. The 10cm $. / 5 ^ { \circ }$ recall ↑ and training steps are reported on Wayspots-Bears, varying the proportion of augmented synthesized images.
<table><tr><td>Proportion of Synthesized Images</td><td colspan="3">Recall (%) Time (h) Training Steps</td></tr><tr><td>DIMM (Baseline)</td><td>82.4</td><td>0.83</td><td>20K</td></tr><tr><td>0% (Improved DIMM)</td><td>84.1</td><td>1.15</td><td>20K</td></tr><tr><td>20%</td><td>84.6</td><td>1.87</td><td>30K</td></tr><tr><td>50%</td><td>87.1</td><td>2.25</td><td>40K</td></tr><tr><td>70%</td><td>92.3</td><td>2.67</td><td>50K</td></tr><tr><td>100%</td><td>95.5</td><td>3.22</td><td>60K</td></tr></table>

This section investigates the trade-of between training duration and model performance relative to the volume of synthetic data utilized. As the training steps are increased, total training time scales linearly with the amount of rendered data added for augmentation. To quantify this relationship, we evaluated VL recall on the Wayspots-Bears dataset across varying ratios of synthetic data, with results summarized in Table 6. Our findings indicate that increasing the volume of augmented data yields progressively higher accuracy, albeit at the

Table 7: Ablation study on NeRF augmentation. The median translation/rotation errors $\downarrow ,$ along with rendering PSNR ↑, are reported on four 7-Scenes scenes.
<table><tr><td>Scene</td><td>NaCR</td><td>NaCR</td><td>NaCR</td><td></td><td>RAP [17] GRLoc [16]</td></tr><tr><td></td><td></td><td>NVS model Nerfacto [36] Deblur-NeRF [24]</td><td>3DGS [41] (w/o rendering loss)</td><td>3DGS</td><td>3DGS</td></tr><tr><td>chess</td><td>0.9/0.2/23.67</td><td>1.0/0.2/22.91</td><td>1.1/0.5/23.95</td><td>1.0/0.8/-</td><td>1.0/0.3/-</td></tr><tr><td>fire</td><td>1.0/0.2/22.75</td><td>1.1/0.5/21.33</td><td>1.4/0.3/23.86</td><td>6.0/3.4/-</td><td>2.9/1.0/-</td></tr><tr><td>heads</td><td>1.3/0.4/18.89</td><td>1.5/1.0/18.09</td><td>1.7/1.3/19.07</td><td>4.0/5.5/-</td><td>1.4/0.9/-</td></tr><tr><td>office</td><td>4.7/0.6/21.98</td><td>3.2/0.8/22.85</td><td>5.3/1.3/22.69</td><td>5.1/1.1/-</td><td>5.0/1.9/-</td></tr></table>

expense of longer training periods. This clear trade-of enables the selection of an optimal augmentation level that balances predictive accuracy against computational overhead. Furthermore, in the absence of synthetic data, our improved DIMM outperforms the original DIMM, demonstrating the inherent efectiveness of the proposed architectural enhancements over the baseline model.

## E Augmentation from Diferent NVS Models

As NaCR embeds an NVS model into CRR, we ablate NVS choices on the first four motion-blurred 7-Scenes scenes. We compare our default Nerfacto [36], Deblur-NeRF [24], and Splatfacto/3DGS [41]; the last provides high-quality augmentation but lacks the explicit ray geometry required by our rendering loss. Tab. 7 also reports PSNR and APR baselines with 3DGS augmentation. Nerfacto gives the best average localization, while Deblur-NeRF helps only the heavily blurred “ofice” scene. Although 3DGS achieves the highest PSNR, the missing rendering loss limits localization accuracy. Across all NVS choices, NaCR outperforms APR methods with NVS augmentation, supporting both the CRR paradigm and tight NVS-CRR integration.

## F NVS Reference-View Sensitivity

To quantify the efect of NVS training coverage, we vary the number of reference views used to train the scene representation from 25% to 100% of the original training images. Nerfacto and 3DGS use the same reference-view subsets at every setting. Table 8 reports both test-view rendering quality and downstream localization accuracy. NaCR improves steadily as reference-view coverage increases, even though Nerfacto has lower image-level PSNR than 3DGS. This trend is consistent with patch-level ray supervision: local rendering signals can still provide useful gradients for ray refinement, and better NVS coverage yields more reliable supervision.

## G Ablation Study on Improved DIMM

In this section, an ablation study is conducted on four scenes from the Wayspots dataset to validate our three proposed enhancements to the DIMM baseline, using recall at 10cm/5<sup>◦</sup> as the metric. The results in Tab. 9 confirm that every component provides a distinct benefit. Specifically, adopting the pre-trained DINOv2 backbone from DA3 consistently improves performance, such as 91.7 vs. 92.5 in the “Bears” scene. For the reference token condition, we found that our pose-clustering method is superior to random sampling, which can degrade accuracy. This suggests that a well-distributed set of reference images can produce more robust query features. Finally, the inclusion of the ray confidence head also boosts performance in most cases, except for low-complexity environments such as “The Rock”. Furthermore, a cross-table comparison reveals that the NeRF integration (Tab. 9: DIMM vs. DIMM + Nerfacto) provides a greater performance contribution than our architectural enhancements to the baseline (Tab. 6: DIMM vs. Improved DIMM). However, both of them are significant and necessary to achieve the final performance of NaCR.

Table 8: NVS reference-view sensitivity on 7-Scenes. Nerfacto and 3DGS use matched reference-view subsets. We report average median translation/rotation error $\left( \mathrm { c m } / ^ { \circ } \right)$ and test-view PSNR over all scenes.
<table><tr><td>NVS reference views (%)</td><td></td><td>NaCR ↓ Nerfacto PSNR ↑</td><td>RAP↓</td><td>3DGS PSNR ↑</td></tr><tr><td>0</td><td>2.9/0.7</td><td>N/A</td><td>7.1/2.9</td><td>N/A</td></tr><tr><td>25</td><td>3.2/1.1</td><td>13.7</td><td>8.3/3.6</td><td>15.3</td></tr><tr><td>50</td><td>2.6/0.6</td><td>17.4</td><td>7.5/3.2</td><td>19.4</td></tr><tr><td>75</td><td>2.3/0.4</td><td>19.8</td><td>6.8/2.8</td><td>22.3</td></tr><tr><td>100</td><td>2.2/0.4</td><td>21.0</td><td>5.0/2.5</td><td>23.4</td></tr></table>

Table 9: Ablation Study on Improved DIMM components. The recall ↑ within 10cm $/ 5 ^ { \circ }$ is reported on the Wayspots dataset.
<table><tr><td>Model</td><td>Bears Cubes Lawn The Rock</td><td></td><td></td></tr><tr><td>I: DIMM</td><td>82.4</td><td>71.5 22.4</td><td>95.4</td></tr><tr><td> $\mathrm { I I } { \boldsymbol { : } } \ \mathrm { D I M M } + \mathrm { N e r f a c t o }$ </td><td>91.7 72.1</td><td>29.6</td><td>96.0</td></tr><tr><td> $\mathrm { I I I } \colon \mathrm { I I } + \mathrm { D A } 3 \ \mathrm { D I N O v } 2$ </td><td>92.5 72.3</td><td>30.2</td><td>96.5</td></tr><tr><td>IV: III + Ref. Token Cond. (random)</td><td>92.1 72.1</td><td>29.8</td><td>96.9</td></tr><tr><td> $\mathrm { V \colon I I I } + \mathrm { R e f . ~ T o k e n }$  Cond. (centered)</td><td>94.6 72.3</td><td>32.5</td><td>98.1</td></tr><tr><td>VI: V + Ray-Confidence Head</td><td>95.5 72.5</td><td>34.2</td><td>97.9</td></tr></table>

![](images/5a7321ec3ed78d84b44aae7de90bd3897dd068430290553253019e7574563fca.jpg)

![](images/facb200638a44dce0a66bf6332a92604e4b88ddda673dad9593d5c2befa1e101.jpg)

![](images/6c07cfa6af513207b2509ec1d1256809e400680f23c91e54cfcb4f15ac7ff550.jpg)  
Fig. 7: The performance of image-level fine-tuning of DA3 DPT [30] decoder in 7- Scenes Chess. This is because the small VL training set generates highly correlated ray gradients within each image, which corrupts the powerful geometric priors learned during DA3’s large-scale pre-training.

![](images/e13bbf442be6f08297e776d802bb4cf5dcfbf92e0edfbb67f72ea4ac00b70110.jpg)  
Fig. 8: Poor Rendering Image Quality of Nerfacto in 6-Scenes [7].The severe illumination changes in the 6-Scenes dataset degrade the rendering quality of the NVS model, causing it to produce significant visual artifacts. This, in turn, negatively impacts NaCR’s localization performance by introducing noise into both of its NeRF-dependent components: (1) the synthetic images used for data augmentation are corrupted, and (2) the supervisory signal from the end-to-end rendering loss becomes unreliable.

## H Fine-tuning DPT of DA3

We also evaluate an alternative training approach: directly fine-tuning the original DA3 Dense Predictive Transformer [30] (DPT) decoder at the image level, instead of using the patch-based DIMM decoder with GDT. However, this strategy yields poor localization performance, as shown in Fig. 7. We attribute this failure to two primary factors: 1) Catastrophic Forgetting: The training data for a single VL scene is vastly smaller and less diverse than DA3’s pre-training dataset. Fine-tuning on this narrow distribution can corrupt the powerful, generalizable geometric priors of DPT acquired during large-scale pre-training. 2) Training Ineficiency: Image-level training is ineficient because the gradients from the rays within a single image are highly correlated. This lack of sample diversity per step significantly slows convergence on the single scene and limits the VL accuracy [3]. Given these limitations, the original image-level decoder is ill-suited for the scene-specific VL task. We therefore adopt the more robust and eficient combination of a patch-level DIMM decoder [43] and the GDT scheme [3].

## I Limitation and Future Work

## I.1 Limitation from NeRF Reliance

A key limitation of NaCR is its dependence on a high-quality, pre-trained NeRF for each scene. In challenging localization scenarios sufering from sparse image coverage or severe illumination changes, reconstructing a high-quality NeRF remains dificult. Therefore, the inherent noise within the NeRF model can degrade the VL performance of NaCR. For instance, we attempt to apply NaCR to the 6-Scenes dataset [7], which is characterized by large spatial ranges and lighting variations. These conditions lead to a low-quality NeRF reconstruction containing significant visual artifacts (as shown in Fig. 8). When integrated into the NaCR pipeline, these rendering errors introduce substantial noise into both the data augmentation and the supervisory signal, ultimately causing NaCR’s VL performance to fall below that of the DIMM baseline.

## I.2 Future work of Jointly NeRF-CRR Optimization

A promising direction for future work is the end-to-end, joint optimization of the NeRF and CRR models. While our diferentiable pipeline theoretically supports this, it presents a significant challenge: NeRF training relies on accurate camera rays, while our CRR model requires a high-fidelity NeRF for supervision. Successfully resolving this co-dependency could lead to a framework capable of simultaneous localization and mapping. This function is similar to recent Large View Synthesis Models (LVSM) [9,45], but is explicitly grounded in geometry of the camera ray, potentially ofering greater robustness and interpretability.