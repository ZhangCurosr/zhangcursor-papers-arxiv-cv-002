# Preserving Knowledge across Space and Time for Continual Video Deepfake Detection

Taehoon Kim<sup>1⋆</sup> , Jongwook Choi<sup>1⋆</sup> , Heejae Jo<sup>2</sup> , Byungmin Park<sup>1</sup> , and Jongwon Choi<sup>1,2,3†</sup>

<sup>1</sup> GS. of AI, Chung-Ang University, Republic of Korea

2 Dept. of Advanced Imaging, GSAIM, Chung-Ang University, Republic of Korea 3 Dept. of Metaverse Convergence, Chung-Ang University, Republic of Korea {kimth,cjw,jzoe,byungmin\_park}@vilab.cau.ac.kr, choijw@cau.ac.kr

Abstract. The continuous emergence of high-quality video deepfakes requires detectors that continually adapt to new forgery patterns, yet existing approaches, which are designed for deepfake images, fail to capture video-specific cues. Unlike deepfake images that contain only spatial artifacts, deepfake videos leave distinct evidence along both spatial and temporal axes, necessitating the separate preservation of each modality during sequential model updates. To overcome this limitation, we introduce a continual deepfake video detection framework, Modality-Specific Frequency Distillation (MSFD), that explicitly decomposes video features into spatial, temporal, and spatiotemporal modalities in the frequency domain. This decomposition enables independent preservation of each modality, as diferent deepfake video types exhibit varying reliance on spatial and temporal cues across tasks. Furthermore, MSFD adopts a cross-modality decorrelation loss that encourages spatiotemporal representations to remain orthogonal to single-modality cues. Extensive experiments show that our framework achieves stronger adaptation and preserves performance more efectively than state-of-the-art methods across diverse continual deepfake video scenarios.

Keywords: Continual Learning · Deepfake Detection · Video Processing

## 1 Introduction

To address the continuously evolving video deepfakes, continual learning approaches have been proposed to enable detection models to adapt to newly emerging forgery techniques while preserving their ability to detect previously learned ones [4, 19, 36]. The rapid advancement of deepfake generation techniques has led to the continuous emergence of new video forgery methods, with the quality and realism of generated videos improving significantly [57]. Recent video generation approaches [1,52,63] produce high-quality video deepfakes that pose substantial challenges to deepfake detection systems trained on limited forgery types. While recent video deepfake detectors improve robustness by modeling generalizable spatiotemporal representations [12, 53, 64], they are typically trained once under a fixed distribution and evaluated in a static setting. In practice, however, forgery techniques evolve after deployment and are encountered sequentially, requiring detectors to be updated over time.

![](images/95cee0b055fdebe7b605587ee458a02d36cf3e9efc6b56444b0179cfc35cbfcc.jpg)

![](images/dc7c4b8f76fe0572f6f390fbafeacca84793e981ca6a148c48623b332fe465f6.jpg)

![](images/451f49ebb044d5e0024c95aea695bf393af21e8bce24c8cd310e817e6ed47cfb.jpg)

![](images/6f042387695438d8a0b424baf966683fcf2f359f313345c4e2515b977022a274.jpg)

![](images/9e61c2fcf9553e227ff54a53427a0b238be16e7b829b6f02c20e695a33e9245e.jpg)  
Fig. 1: Performance in the continual video deepfake detection. The top row illustrates representative temporal, spatiotemporal, and spatial artifacts across sequential tasks. The bottom row shows the performance degradation on the initial task (FF++) and average performance across all tasks as the model is continually trained on six sequential deepfake video datasets (FF++, DFD, CDF, DFDCp, FFIW, and KoDF). Our method (red) maintains the highest performance with the least degradation compared to existing continual deepfake detection approaches.

However, most existing continual deepfake detection frameworks [4, 19, 32, 36,48] were primarily designed under image-level assumptions, which limit their ability to capture video-level spatiotemporal artifacts. Deepfake videos have complementary but distinct evidence in spatial artifacts (e.g., blending boundaries) and temporal inconsistencies (e.g., flickering, unnatural motion) [34, 53, 64]. Such image-centric designs are therefore inadequate for videos, where deepfakes exhibit characteristics that difer from those observed in manipulated images [53,64]. Consequently, these image-based approaches struggle when applied to video-based detection, often leading to unbalanced adaptation and severe forgetting, as shown in Fig. 1.

Our preliminary experiments further support this observation (Fig. 2). When we trained models using only spatial or only temporal cues in a continual-training setting, their layer-wise weight updates and activation responses diverged substantially (Fig. 2 (a) and (b)). Moreover, performance retention under spatial and temporal perturbations varies markedly across generation paradigms—some rely primarily on spatial cues, others on temporal cues, and some on both—even within the same dataset (e.g., DFDCp-A vs. DFDCp-B) (Fig. 2 (c)). These results explain why existing methods that treat spatial and temporal cues as a single entangled representation fail to retain both during continual updates, and motivate the core design of our framework: preserving spatial, temporal, and spatiotemporal representations separately.

Layer-wise Weight Changes from the first to last tasks  
![](images/8855642f1212c7336f657f5aca97c6a3fa31bdb382b02d34f3a01326205be6fc.jpg)  
(a) Parameter Update Magnitudes

![](images/2fba0e18365e53eeee4d7d7f5789e2d2ec1951d819e8951a26df09be8650a142.jpg)  
(b) Activation Maps

![](images/d5a0d7dca6b3f77392932f528c08c4d70752eec8df82a8df228fb8b53541b6a3.jpg)  
(c) Modality Sensitivity  
Fig. 2: Analysis of Spatial and Temporal Modalities. We visualize (a) parameter update magnitudes across layers relative to the iCaRL [39] baseline and (b) activation patterns when models are continually trained with either spatial-only or temporal-only cues, alongside (c) a dataset-wise sensitivity map showing the AUC retention ratio of dataset-specific finetuned models under spatial and temporal perturbations. Detailed settings are provided in the supplementary material.

Motivated by these observations, we develop Modality-Specific Frequency Distillation (MSFD), which decomposes features into spatial, temporal, and spatiotemporal modalities and preserves each by aligning the current model’s frequency spectra with those of the previous model at each continual learning step. This design enables independent preservation of each modality, as diferent deepfake video types carry distinct spatial and temporal characteristics that need to be retained separately (Fig. 2 (c)). We perform this decomposition in the spectral domain because it ofers a simple yet highly efective approach specifically for deepfakes. Unlike complex disentanglement in the raw feature space, frequency transforms explicitly isolate distinct forgery signatures: spatial forgeries manifest high-frequency boundary artifacts captured by 2D spatial spectra [15, 30], while point-wise 1D temporal frequency spectra capture fine-grained temporal irregularities characteristic of temporal manipulations [20]. We further employ a 3D spatiotemporal spectrum to capture joint space-time dynamics inaccessible to single-modality transforms, and introduce a cross-modality decorrelation loss that encourages the spatiotemporal representation to encode frequency patterns specific to the joint use of spatial and temporal cues while remaining orthogonal to those captured by single-modality branches.

Through extensive experiments across realistic continual deepfake video detection scenarios, including sequential domain shifts, emerging forgery techniques, and restricted data, our method shows strong adaptability while preserving prior knowledge. We further confirm its robustness through evaluations on diverse perturbation settings, and our ablation studies validate the efectiveness of each component in the modality-specific design. Our contributions are summarized as follows:

We introduce the first framework that addresses the preservation of spatial and temporal forgery cues for continual deepfake video detection.

– We design the Modality-Specific Frequency Distillation (MSFD) framework, which decomposes video features into three complementary frequency modalities to preserve modality-specific deepfake cues.

We propose a Cross-Modality Decorrelation Loss (CDL) that encourages spatiotemporal frequency representations orthogonal to single-modality cues.

– We conduct comprehensive experiments across multiple continual deepfake video settings, demonstrating our proposed framework’s ability to improve generalization, robustness, and forgetting mitigation.

## 2 Related Work

Video-based Deepfake Detection. Early deepfake detectors [6, 15, 16, 30, 45, 51, 58] focused on spatial artifacts in static images, failing to account for temporal inconsistencies that often expose video manipulations. Later studies [5, 10, 11, 20, 33, 34, 53, 55, 59, 64] showed that modeling spatiotemporal representations improves robustness to unseen forgery types. For example, FTCN [64] leverages temporal artifacts as transferable cues, while AltFreezing [53] disentangles spatial and temporal branches to stabilize detection, and recent detectors further highlight the importance of balanced spatiotemporal fusion [12, 34, 59]. Despite these advances, most detectors still assume a fixed closed-world distribution. However, new deepfake techniques are continuously evolving after deployment, necessitating detectors that can be incrementally updated over time. Continual Deepfake Detection. Continual deepfake detection aims to adapt detectors to newly emerging forgeries while retaining performance on previously learned ones [19, 24, 36, 47, 48, 62]. Existing methods such as CoReD [19], DFIL [36], and SURLID [4] preserve prior knowledge via distillation or feature alignment, but are designed under image-level assumptions. Unlike deepfake images that contain only spatial artifacts, video deepfakes present complementary evidence along both spatial and temporal axes, whose divergent characteristics make a single entangled representation insuficient to retain during continual updates. SARB-DF [38], the only prior efort toward continual deepfake video detection, addresses only spatial cues and follows generic regularization (e.g., EWC [21]) rather than deepfake-specific mechanisms. These limitations motivate our design that separately preserves spatial, temporal, and spatiotemporal forensic cues throughout continual model updates.

Continual Learning for Video Representations. Continual learning has been widely studied for video understanding, where models are sequentially adapted to evolving spatial and temporal dynamics [3,14,23,28,50]. These methods are typically designed to preserve high-level semantic representations (e.g., playing tennis) under sequential updates. However, deepfake video detection depends on modality-sensitive, low-level forensic signals (e.g., frequency artifacts and fine-grained temporal inconsistencies) that are fragile and can be easily washed out under semantics-focused objectives. As a result, directly applying conventional video continual learning techniques leads to suboptimal adaptation, motivating forensic-aware preservation targets for continual deepfake detection.

![](images/69e1cfc8af6278c33da5aa97910af075497c9c9c4cae6ce2ea5604bb634c278b.jpg)  
Fig. 3: Our proposed framework, Modality-Specific Frequency Distillation (MSFD) for continual deepfake video detection. (a) Overall pipeline: The current model (θ<sub>t</sub>) is trained on task t data and memory (R), distilling knowledge from the frozen previous model $\left( \theta _ { t - 1 } \right)$ . (b) Modality-Specific Frequency Distillation (MSFD): Intermediate features from $l ^ { t h }$ layer are decomposed into temporal (T), spatial (S), and spatiotemporal (ST) frequency spectra via 1D, 2D, and 3D FFTs. The Modality-Specific Adaptor (MSA) applies modality-specific reweighting $\big ( \boldsymbol { \varPhi } _ { d } ^ { ( l ) } \big )$ and distillation masking $( M _ { d } ^ { ( l ) } )$ to compute frequency-domain distillation loss (L<sub>FDL</sub>), while the cross-modality decorrelation loss (L<sub>CDL</sub>) is computed in parallel.

## 3 Proposed Method

We address continual deepfake video detection, where the model encounters a sequence of T tasks. At each task $t ,$ the previous model $\theta _ { t - 1 }$ is frozen as the teacher, and the current model $\theta _ { t }$ is updated using the task data $\mathcal { D } _ { t }$ and a small subset of prior data stored in the replay memory R. The primary objective at task t is to efectively learn the new deepfake domain introduced in the current dataset, while maintaining high accuracy on all previously learned tasks by preserving knowledge from the previous model $\theta _ { t - 1 }$

## 3.1 Framework Overview

Our proposed framework, Modality-Specific Frequency Distillation (MSFD), employs a unified optimization scheme that leverages frequency-domain distillation and cross-modality regularization. The overall pipeline, visualized in Fig. 3, explicitly decomposes intermediate features [37, 60] into three complementary modalities—spatial, temporal, and spatiotemporal—via Fast Fourier Transforms (FFT), enabling distillation of distinct deepfake artifacts per modality. Our framework involves three key mechanisms: (1) the modality-wise frequency transform explicitly decomposes intermediate features into frequency spectra for each modality; (2) the Modality-Specific Adaptor (MSA) performs adaptive calibration and selective masking on the current model’s spectra to compute the frequency-domain distillation loss $\mathcal { L } _ { F D L }$ , thereby more efectively preserving modality-specific cues; (3) the Cross-Modality Decorrelation Loss (CDL) L<sub>CDL</sub> promotes distinct spatiotemporal representations while remaining orthogonal to those captured by single-modality branches. These components are integrated with standard classification loss $\mathcal { L } _ { C E }$ and logit distillation loss $\mathcal { L } _ { P D }$

## 3.2 Modality-wise Frequency Transform

At task t, the current model $\theta _ { t }$ is optimized with respect to the frozen previous model $\theta _ { t - 1 }$ using replay samples from R. For a set of intermediate layers L in the video backbone, we extract ${ { l } ^ { t h } }$ layer’s feature maps $F _ { t - 1 } ^ { ( l ) } , F _ { t } ^ { ( l ) } \in \bar { \mathbb R } ^ { \bar { C } _ { l } \times T _ { l } \times H _ { l } \times W _ { l } }$ from both models, where $l \in L , C _ { l }$ denotes the number of channels, $T _ { l }$ is the temporal dimension, and $H _ { l } , W _ { l }$ are the spatial dimensions at ${ { l } ^ { t h } }$ layer.

To explicitly decompose these representations into modality-specific components, we apply Fast Fourier Transform (FFT) along diferent dimensions for feature maps $F _ { t } ^ { ( l ) }$ and $F _ { t - 1 } ^ { ( l ) }$ to obtain three complementary magnitude spectra for each modality $d \in \{ \mathsf { T } , \mathsf { S } , \mathsf { S } \mathsf { T } \}$ , corresponding to temporal, spatial, and spatiotemporal modalities. We then take the magnitude to construct modality-wise spectra $Z _ { t } ^ { ( d , l ) }$ and $Z _ { t - 1 } ^ { ( d , l ) }$

Temporal (T): 1D FFT along the temporal axis $T _ { l }$ to capture temporal frequency patterns.

– Spatial (S): 2D FFT over the spatial dimensions $( H _ { l } , W _ { l } )$ to extract spatial frequency characteristics.

– Spatiotemporal (ST): 3D FFT across both temporal and spatial axes $( T _ { l } , H _ { l } , W _ { l } )$ to capture joint spatiotemporal frequency, targeting coupled space–time artifacts not recoverable from single-modality spectra alone.

## 3.3 Modality-Specific Adaptor

To more efectively handle modality-specific cues during distillation, we introduce the Modality-Specific Adaptor (MSA), which integrates modality-specific reweighting and modality-specific distillation masking.

Modality-Specific Reweighting. For each modality $d \in \{ \mathsf { T } , \mathsf { S } , \mathsf { S } \mathsf { T } \}$ and $l ^ { t h }$ layer, the modality-specific reweighting module applies a learnable, modalitywise reweighting parameter $\boldsymbol { \varPhi } _ { d } ^ { ( l ) }$ to recalibrate the current model’s spectrum $Z _ { t } ^ { ( d , l ) }$ via a residual transformation:

$$
\widetilde { Z } _ { t } ^ { ( d , l ) } = \varPhi _ { d } ^ { ( l ) } \odot Z _ { t } ^ { ( d , l ) } + Z _ { t } ^ { ( d , l ) } ,\tag{1}
$$

where the reweighting parameter $\boldsymbol { \varPhi } _ { d } ^ { ( l ) }$ is defined with a modality-dependent shape and is automatically broadcast to match the spectrum $Z _ { t } ^ { ( d , l ) } \in \mathbb { R } ^ { B \times C \times T _ { l } \times H _ { l } \times W _ { l } }$ enabling element-wise spectral modulation. Specifically, for the temporal modality (T), the reweighting parameter spans only the temporal axis, resulting in $\phi _ { \mathsf { T } } ^ { ( l ) } \in \mathbb { R } ^ { 1 \times C \times T _ { l } \times 1 \times 1 }$ . For the spatial modality (S), the parameter covers the spatial dimensions, giving $\phi _ { \mathsf { S } } ^ { ( l ) } \in \mathbb { R } ^ { 1 \times C \times 1 \times H _ { l } \times W _ { l } }$ . Finally, for the spatiotemporal modality (ST), all axes are reweighted jointly, yielding $\Phi _ { \mathsf { S T } } ^ { ( l ) } \in \mathbb { R } ^ { 1 \times C \times T _ { l } \times H _ { l } \times W _ { l } }$ Modality-Specific Distillation Masking. The modality-specific distillation masking module learns a sparse, binary mask ${ \cal M } _ { d } ^ { ( l ) }$ to select discriminative frequency bands based on data, rather than hand-crafted priors. The frequency spectrum’s radial axis is discretized into $K _ { d }$ bins. A learnable logit vector $\mathbf { a } _ { d , k } ^ { ( l ) } \in$ $\mathbb { R } ^ { 2 }$ for each bin k is converted into a binary pass/block decision $m _ { d , k } ^ { ( l ) } \in \{ 0 , 1 \}$ using the Gumbel-Softmax operator $\mathcal { G } ( \cdot ; \tau )$

$$
m _ { d , k } ^ { ( l ) } = \mathcal { G } ( \mathbf { a } _ { d , k } ^ { ( l ) } ; \tau ) \cdot \mathbf { e } _ { \mathrm { p a s s } } ,\tag{2}
$$

where ${ \mathbf { e } } _ { \mathrm { p a s s } } = [ 0 , 1 ] ^ { \top }$ selects the $\mathrm { \Delta ^ { \circ } p a s s ^ { \circ } }$ component, and the temperature $\tau$ is linearly annealed from 1.0 to 0.05 to encourage sharp selections.

The radial mask at frequency location u is then defined as:

$$
M _ { d } ^ { ( l ) } ( \mathbf { u } ) = m _ { d , \kappa _ { d } ^ { ( l ) } ( \mathbf { u } ) } ^ { ( l ) } \in \{ 0 , 1 \} ,\tag{3}
$$

which directly maps each frequency location to its corresponding bin’s pass/block decision. The learned radial mask is broadcast to modality-specific dimensions, such as $( 1 , 1 , T _ { l } , 1 , 1 )$ for temporal, $( 1 , 1 , 1 , H _ { l } , W _ { l } )$ for spatial, and $( 1 , 1 , T _ { l } , H _ { l } , W _ { l } )$ for spatiotemporal, and then applied element-wise to each corresponding spectrum to obtain the final masked representations:

$$
\widehat { Z } _ { t } ^ { ( d , l ) } = M _ { d } ^ { ( l ) } \odot \widetilde { Z } _ { t } ^ { ( d , l ) } , \quad \widehat { Z } _ { t - 1 } ^ { ( d , l ) } = M _ { d } ^ { ( l ) } \odot Z _ { t - 1 } ^ { ( d , l ) } ,\tag{4}
$$

where $\odot$ denotes element-wise multiplication after broadcasting. We empirically set the radial bins $K _ { d }$ for each modality as $K _ { \mathsf { T } } = 4 , K _ { 5 } = 1 6$ , and $K _ { S \mathsf { T } } = 3 2 $

## 3.4 Training Objective

Frequency-domain distillation loss. The frequency-domain distillation loss then enforces knowledge transfer in the most informative spectral regions:

$$
\mathcal { L } _ { \mathrm { F D L } } = \sum _ { l \in \mathcal { L } } \sum _ { d \in \{ \mathsf { T } , \mathsf { S } , \mathsf { S } \mathsf { T } \} } \lambda _ { d } \Vert \widehat { Z } _ { t } ^ { ( d , l ) } - \widehat { Z } _ { t - 1 } ^ { ( d , l ) } \Vert _ { 2 } ^ { 2 } ,\tag{5}
$$

where $\lambda _ { d }$ is a modality-wise weighting coeficient.

Cross-modality decorrelation loss. To ensure the spatiotemporal representation captures unique joint space-time dynamics and remains orthogonal to the single-modality cues, we introduce the Cross-Modality Decorrelation Loss

(CDL). For each l-th layer, CDL measures the correlation between the spatiotemporal spectrum and its spatial or temporal counterparts:

$$
\mathcal { L } _ { \mathrm { C D L } } = \sum _ { l \in L } \left[ \begin{array} { c } { \mathcal { L } _ { \mathrm { c o r r } } ( P _ { \mathsf { S T } \to \mathsf { S } } ^ { ( l ) } ( \widetilde { Z } _ { t } ^ { ( \mathsf { S T } , l ) } ) , P _ { \mathsf { S } } ^ { ( l ) } ( \widetilde { Z } _ { t } ^ { ( \mathsf { S } , l ) } ) ) } \\ { + \mathcal { L } _ { \mathrm { c o r r } } ( P _ { \mathsf { S T } \to \mathsf { T } } ^ { ( l ) } ( \widetilde { Z } _ { t } ^ { ( \mathsf { S T } , l ) } ) , P _ { \mathsf { T } } ^ { ( l ) } ( \widetilde { Z } _ { t } ^ { ( \mathsf { T } , l ) } ) ) } \end{array} \right] ,\tag{6}
$$

where $P _ { d } ^ { ( l ) }$ represents modality-specific projection heads that map the spectra into an alignment space. Before projection, spatial spectra are obtained by averaging over the temporal axis, temporal spectra by averaging across spatial dimensions, and spatiotemporal spectra are used directly without reduction. Each spectrum is then projected through its corresponding head $( P _ { 5 } ^ { ( l ) } , P _ { \mathsf { T } } ^ { ( l ) } , P _ { 5 \mathsf { T }  5 } ^ { ( l ) } ,$ $P _ { \mathsf { S T }  \mathsf { T } } ^ { ( l ) } )$ into an alignment space with dimensionality equal to the number of frequency bins for that modality $( e . g . , K _ { 5 } , K _ { \mathsf { T } } )$ .

For projected tensors $A , B \in \dot { \mathbb { R } } ^ { B \times C \times k }$ , the mean per-channel Pearson correlation is defined as:

$$
\mathrm { c o r r } ( A , B ) = \frac { 1 } { B C } \sum _ { b = 1 } ^ { B } \sum _ { c = 1 } ^ { C } \left( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } A _ { b , c , i } ^ { \prime } B _ { b , c , i } ^ { \prime } \right) ,\tag{7}
$$

where $A ^ { \prime }$ and $B ^ { \prime }$ are channel-wise z-score normalized along the frequency dimension. In practice, $\mathcal { L } _ { \mathrm { c o r r } } ( A , B )$ is computed as the mean squared correlation, ${ \mathcal { L } } _ { \mathrm { c o r r } } ( A , B ) = \operatorname { c o r r } ( A , B ) ^ { 2 }$ , equally penalizing positive and negative correlations. Final objective loss. The final objective function integrates classification, logitlevel distillation [19, 39], and our frequency-based regularization terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { P D } } \mathcal { L } _ { \mathrm { P D } } + \lambda _ { \mathrm { F D L } } \mathcal { L } _ { \mathrm { F D L } } + \lambda _ { \mathrm { C D L } } \mathcal { L } _ { \mathrm { C D L } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the standard binary cross-entropy loss computed on the combined data $( \mathcal { D } _ { t } \cup \mathcal { R } )$ ). The term $\mathcal { L } _ { \mathrm { P D } }$ is the memory-based logit-level distillation loss, which is applied exclusively to samples from the replay memory R to align the student’s logits with the frozen previous model’s predictions. Finally, $\lambda _ { \mathrm { P D } } , \lambda _ { \mathrm { F D L } } .$ and $\lambda _ { \mathrm { C D I } }$ are empirically set weighting coeficients.

## 4 Experiments

Datasets. We conduct experiments on ten publicly available video-based deepfake datasets spanning a broad range of forgery techniques: FF++ [42], DFD [8], CDF [27], DFDCp [7], FFIW [65], KoDF [22], FSh [26], DFo [17], DF40 [57], and AIGVDBench [31]. The first nine are face-centric video datasets capturing the temporal dynamics of facial forgeries; AIGVDBench additionally covers non-facial AI-generated content to evaluate generalization beyond face-centric scenarios. For DF40 [57], we adopt six manipulation methods: BlendFace [46], FaceDancer [41], and MobileSwap [56] for Face Swap (FS), and HyperReenact [2], MCNet [13], and SadTalker [61] for Face Reenactment (FR) deepfakes.

Continual Video Deepfake Detection Protocols. We evaluate our framework under three complementary continual learning protocols tailored to videobased deepfake detection.

– Protocol 1: Dataset Incremental (DI). This protocol emulates a practical scenario where detectors must continually adapt as new deepfake datasets emerge. The model is sequentially trained on FF++ (Task 1), followed by DFD, CDF, DFDCp, FFIW, and KoDF. For a balanced evaluation, we sample 200 real and 200 fake videos from each newly introduced dataset.

– Protocol 2: Fake Incremental (FI). This protocol evaluates robustness against structural shifts across generation paradigms in AIGVDBench [31], inspired by [4]: Text-to-Video (T2V) setup using EasyAnimate [54], Image-to-Video (I2V) via Pyramid-Flow [18], Video-to-Video (V2V) via LTX [9], and closedsource generation (SORA [35]), each relying on distinct generative priors that produce fundamentally diferent forgery patterns. Specifically, subsequent tasks introduce only new fake videos without any real data.

Protocol 3: Few-shot Incremental (FSI). Following the continual deepfake detection setting [36], this protocol evaluates adaptation under a constrained data regime (only 25/25 real/fake videos per task, excluding the first task), which mirrors realistic scenarios where new forgeries first appear with limited samples. The model is sequentially updated across FF++, $\mathrm { D F D C p } ,$ DFD, and CDF.

Evaluation Metrics. For each task $t ,$ we compute the video-level AUC, denoted AUC\_t , and summarize the overall performance as: $\begin{array} { r } { A U C _ { a l l } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } A U C _ { t } } \end{array}$ Forgetting is measured as the average performance drop relative to when each task was first learned: $\begin{array} { r } { F _ { a v g } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { \bar { T } - 1 } \left( a _ { t , t } - a _ { T , t } \right) } \end{array}$ , where $\boldsymbol { a } _ { t , i }$ denotes the AUC on Task t immediately after learning it, and ${ a } _ { T , t }$ is the AUC on the same task after completing all $T$ tasks. A higher $A U C _ { a l l }$ and lower $F _ { a v g }$ indicate better retention and a more favorable stability–plasticity balance. For Protocol 3, following [36], we report clip-level accuracy with Average Accuracy (AA) and Average Forgetting (AF), where AA is the mean accuracy over all seen tasks, and AF measures the average performance drop from when each task was first learned.

Comparison Methods. We compare against a broad spectrum of continual learning baselines: naive fine-tuning, EWC [21], DCE [25], replay-based methods (ER [40], iCaRL-NME/FC [39], IDER [29]), video continual learning methods (Drift [14], HCE [28], vCLIMB [50], STSP [3], ESSENTIAL [23]), and continual deepfake detection methods (SARB-DF [38], DFIL [36], HDP [47], SURLID [4]). Unless otherwise specified, all methods adopt 3D ResNet-18 [49] and are initialized from Task 1 weights for a fair comparison. SARB-DF and ESSENTIAL are evaluated with their native backbones, as their architectures cannot be directly adapted.

Implementation Details. For preprocessing, as described in [64], we detect and align faces using RetinaFace. The model processes a 32-frame clip at a spatial resolution of 224 \times 224 . During training, up to 270 frames are extracted from each video to form multiple 32-frame clips via temporal sampling, while evaluation uses up to 110 frames to generate non-overlapping or minimally overlapping clips. Video-level scores are computed by averaging the clip-level logits. All models are trained for 20 epochs per task using the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 8. The replay memory stores 200 clips (each containing 32 frames) for replay-based continual learning methods. We adopt iCaRL [39] with a linear classifier as a baseline and employ its exemplar-based memory update strategy to focus on the efectiveness of our preservation mechanism. Additional implementation details are provided in the Supplementary.

Table 1: Comparison under Protocol 1: Dataset-Incremental (DI). We evaluate continual adaptation as new deepfake video datasets are sequentially introduced, and report video-level AUC (%) after the final task, overall $A U C _ { a l l }$ , and average forgetting $F _ { a v g }$ . Unless otherwise noted, all methods use 3DResNet-18 as backbone; all replay-based methods set memory size to 200 clips. Best results are in bold and second best are underlined. ↑/↓ indicates higher/lower is better.
<table><tr><td>Method</td><td>Backbone</td><td>[FF++ DFD CDF DFDCp FFIW KoDF|AUCall↑ Favg ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">&gt;General continual learning methods</td></tr><tr><td>Finetuning</td><td></td><td>89.47</td><td></td><td>89.2971.13</td><td>73.82</td><td>84.98</td><td>99.77</td><td>84.74</td><td>14.07</td></tr><tr><td>EWC [21]</td><td></td><td>81.76</td><td>86.16</td><td>67.67</td><td>69.07</td><td>83.23</td><td>99.89</td><td>81.30</td><td>18.69</td></tr><tr><td>ER [40]</td><td>3DResNet-18</td><td>92.36</td><td>94.21</td><td>72.14</td><td>74.13</td><td>91.68</td><td>99.75</td><td>87.38</td><td>10.77</td></tr><tr><td>iCaRL-NME [39]</td><td></td><td>86.67</td><td>92.22 74.76</td><td></td><td>80.59</td><td>87.60</td><td>96.86</td><td>86.45</td><td>8.14</td></tr><tr><td>iCaRL-FC [39]</td><td></td><td>96.82</td><td>94.65</td><td>80.50</td><td>79.20</td><td>88.77</td><td>99.07</td><td>89.84</td><td>6.81</td></tr><tr><td>DCE [25]</td><td></td><td>99.63</td><td>96.05 72.97</td><td></td><td>83.93</td><td>90.40</td><td>90.91</td><td>88.98</td><td>0.07</td></tr><tr><td>IDER [29]</td><td></td><td>95.32</td><td>96.26 93.62</td><td></td><td>80.09</td><td>83.80</td><td>97.48</td><td>91.10</td><td>2.97</td></tr><tr><td>Video continual learning methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>vCLIMB [50]</td><td></td><td>80.07</td><td>90.46 66.61</td><td></td><td>77.71</td><td>85.51</td><td>96.36</td><td>82.79</td><td>11.27</td></tr><tr><td>Drift [14]</td><td>3DResNet-18</td><td>85.03</td><td>82.8067.56</td><td></td><td>76.14</td><td>94.36</td><td>99.99</td><td>84.31</td><td>14.53</td></tr><tr><td>HCE [28]</td><td></td><td>81.54</td><td>91.56 62.18</td><td></td><td>78.96</td><td>92.08</td><td>99.28</td><td>84.27</td><td>13.06</td></tr><tr><td>STSP [3]</td><td></td><td>99.48</td><td>92.67 83.76</td><td></td><td>77.15</td><td>82.78</td><td>98.35</td><td></td><td></td></tr><tr><td>ESSENTIAL [23] -</td><td>ViT-B/16(CLIP)</td><td>68.10</td><td>95.50~74.40</td><td></td><td>89.70</td><td>41.30</td><td>99.30</td><td>89.03 78.05</td><td>3.98 20.04</td></tr><tr><td colspan="10">Continual deepfake detection methods</td></tr><tr><td>DFIL [36]</td><td></td><td>90.51</td><td>92.02 78.86</td><td></td><td>74.58</td><td>91.52</td><td>99.76</td><td>87.88</td><td>10.02</td></tr><tr><td>HDP [47]</td><td>3DResNet-18</td><td>92.22</td><td>86.20 68.58</td><td></td><td>70.21</td><td>83.07</td><td>99.95</td><td>83.37</td><td>16.14</td></tr><tr><td>SURLID [4]</td><td></td><td>94.98</td><td>90.25 83.20</td><td></td><td>81.30</td><td>76.81</td><td>99.12</td><td>87.61</td><td>8.19</td></tr><tr><td>SARB-DF [38]  Xception + Transformer</td><td></td><td>81.99</td><td>90.60~83.26</td><td></td><td>83.55</td><td>80.46</td><td>99.16</td><td>86.50</td><td>11.63</td></tr><tr><td>Ours</td><td>3DResNet-18</td><td>98.42</td><td></td><td>91.15 90.17</td><td>84.14</td><td>94.88</td><td>99.36</td><td>93.02</td><td>2.29</td></tr></table>

## 4.1 Comparisons on Continual Deepfake Video Detection

Comparison on Protocol 1: Dataset Incremental (DI). As shown in Table 1, our method consistently outperforms existing continual learning strategies under Protocol 1, achieving the highest $A U C _ { a l l }$ . Conventional video continual learning methods (e.g., vCLIMB [50], HCE [28]) primarily focus on preserving high-level semantic representations, which are insuficient for retaining low-level forensic cues, resulting in pronounced forgetting. Continual deepfake detection methods (e.g., DFIL [36], SURLID [4]) rely on a single joint representation under image-level assumptions, failing to disentangle spatial and temporal forensic signals. MSFD overcomes these limitations by independently preserving modality-specific forensic cues in the frequency domain, achieving the highest overall $A U C _ { a l l }$ . While DCE [25] and IDER [29] show competitive forgetting, this comes at the cost of reduced plasticity. Furthermore, as shown in Fig. 4, our method maintains near-zero forgetting throughout the sequence while achieving a steady increase in AUC, demonstrating both stability and adaptability.

Table 2: Protocol 2: Fake-Incremental. We evaluate continual adaptation to structurally distinct generation paradigms and report AUC (%) after the final task, overall $A U C _ { a l l } ,$ and average forgetting $F _ { a v g } .$  
Table 3: Protocol 3: Few-shot Incremental. We evaluate continual adaptation under a restricted data regime and report clip-level accuracy (%) after the final task, Average Accuracy (AA), and Average Forgetting (AF), following [36].
<table><tr><td>Method</td><td>T2V</td><td>I2V</td><td>V2V</td><td>SORA AUCaul↑ Favg↓</td><td></td><td></td></tr><tr><td>iCaRL-FC [39]</td><td>98.49</td><td>87.99</td><td>96.51</td><td>96.71</td><td>94.93</td><td>0.30</td></tr><tr><td>IDER [29]</td><td>99.69</td><td>84.87</td><td>95.92</td><td>96.51</td><td>94.25</td><td>-0.32</td></tr><tr><td>STSP [3]</td><td>99.57</td><td>83.28</td><td>93.65</td><td>89.53</td><td>91.51</td><td>-0.37</td></tr><tr><td>DFIL [36]</td><td>93.24</td><td>83.88</td><td>95.35</td><td>97.74</td><td>92.55</td><td>4.59</td></tr><tr><td>SURLID [4]</td><td>97.31</td><td>83.94</td><td>96.97</td><td>95.62</td><td>93.46</td><td>0.25</td></tr><tr><td>Ours</td><td>98.89</td><td>91.47 97.49</td><td></td><td>98.84</td><td>96.67</td><td>1.37</td></tr></table>

<table><tr><td>Method</td><td>FF++ DFDCp DFD</td><td></td><td>CDF</td><td></td><td>AA↑AF↓</td></tr><tr><td>iCaRL-FC [39]</td><td>95.95</td><td>68.23</td><td>88.08</td><td>75.53</td><td>81.95 2.39</td></tr><tr><td>IDER [29]</td><td>95.50</td><td>77.04</td><td>94.67 66.67</td><td></td><td>83.47 -2.10</td></tr><tr><td>STSP [3]</td><td>87.81</td><td>76.78</td><td>52.86</td><td>82.90</td><td>75.09 1.59</td></tr><tr><td>DFIL [36]</td><td>94.50</td><td>63.14</td><td>81.93</td><td>79.08</td><td>79.66 5.47</td></tr><tr><td>SURLID [4]</td><td>95.98</td><td>71.91</td><td>93.19</td><td>71.24</td><td>83.08 0.58</td></tr><tr><td>Ours</td><td>97.10</td><td>72.04</td><td>90.32</td><td>75.13</td><td>83.65 -1.25</td></tr></table>

Comparison on Protocol 2: Fake Incremental (FI). MSFD achieves the highest overall $A U C _ { a l l }$ across generation paradigms (Table 2). We select the five most competitive methods from Protocol 1, including iCaRL-FC [39], IDER [29], STSP [3], DFIL [36], and SURLID [4], for evaluation under Protocols 2 and 3. The gain is particularly notable on I2V, where MSFD outperforms all baselines by a large margin, suggesting that modality-specific frequency distillation efectively preserves generation-paradigm-specific forensic cues that singlerepresentation approaches fail to retain across tasks. While STSP and IDER report lower forgetting, this comes at the cost of substantially lower overall accuracy, indicating that they sacrifice adaptability to new paradigms to retain prior knowledge. MSFD achieves a more favorable balance, attaining the highest $A U C _ { a l l }$ with moderate forgetting.

Comparison on Protocol 3: Few-shot Incremental (FSI). Under the fewshot setting (Table 3), MSFD achieves the highest AA while also exhibiting negative forgetting $( \mathrm { A F } = - 1 . 2 5 )$ , indicating positive backward transfer to previously learned tasks. Although IDER reports a lower AF, its overall AA is lower than that of MSFD, suggesting that MSFD provides a stronger few-shot adaptability. Notably, STSP, which showed competitive performance in Protocol 1, sufers significant degradation under both Fake-Incremental and Few-shot Incremental settings, revealing that conventional video continual learning approaches lack the robustness to handle the extreme distribution shifts and data scarcity inherent in evolving deepfake scenarios.

## 4.2 Ablation Study and Analysis

We conduct ablation studies to validate the contribution of each component in our framework. Further analyses are provided in the supplementary material. Analysis on Memory Size Sensitivity. We analyze the sensitivity of our framework to the size of the replay memory bufer. As shown in Fig. 5, our method (red) consistently outperforms all baselines across memory sizes from 50 to 400 in both $A U C _ { a l l }$ and $F _ { a v g }$ under Protocol 1. Most notably, with only 50 memory samples, our framework already matches or surpasses DFIL and

![](images/84c2dd45d41254c016c74c077fbbb84fc1a97db3af3e48abc18c71ab05defb0a.jpg)

![](images/0093060a6f25c42c5d73a6c6d29f9410ea7212518dc336e5d69178696ca713d8.jpg)  
Fig. 4: Performance trends across sequential tasks. We visualize the average performance $A U C _ { a l l }$ and forgetting $F _ { a v g }$ across sequential tasks under Protocol 1.  
Fig. 5: Sensitivity for memory bufer size. We report average performance $A U C _ { a l l }$ and forgetting $F _ { a v g }$ across a memory size from 50 to 400 in Protocol 1.

SURLID operating with 400 samples. This eficiency stems from our modalityspecific distillation, which focuses on the most discriminative spectral components as replay exemplars, preserving critical knowledge without requiring large memory budgets. These results confirm that MSFD remains efective even in resource-constrained continual learning scenarios.

Forgetting Analysis Across Modalities. As shown in Fig. 6, our method consistently occupies the upper-right across most benchmarks, demonstrating superior preservation of spatial and temporal forgery cues under continual updates. To analyze modality-specific forgetting behavior beyond aggregate metrics, Fig. 6 visualizes the spatial and temporal retention of each method across benchmarks in Protocol 1, defined as the ratio of the final model’s AUC under spatial and temporal perturbation (grid shufling/frame shufling) to its AUC when first learning each dataset. While competing methods scatter toward lower retention on one or both modalities, our results confirm that explicitly preserving spatial and temporal representations is critical for robust con tinual deepfake detection. Details will be provided in the supplementary material.

![](images/f60c5c39f0cb7921579c12f23a5516d985035fc6f5495c5cb00ced54c1474685.jpg)  
Fig. 6: Forgetting analysis across spatial and temporal modalities. Each point represents a method–dataset pair, plotting spatial and temporal retention, defined as the ratio of the final model’s AUC under each perturbation to its AUC when first learned.

Contribution of Components. Each proposed component addresses a distinct bottleneck, and their combination achieves the best balance between adaptation and retention (Table 4-(a)). Multilayer feature distillation provides only marginal improvement, confirming that preserving entangled spatial-temporal features in the raw space is insuficient. MSFD (Naive) yields a substantial gain by decomposing features into modalityspecific frequency spectra, validating frequency-domain decomposition as the key enabler. MSA improves stability by suppressing noisy frequency bands, while CDL reduces cross-modality redundancy. The full model achieves the strongest overall performance, confirming their complementary roles.

Table 4: Ablation studies of MSFD. (a) Contribution of major components (MSFD, MSA, and CDL) over the baseline. Multi-layer Distillation extends naive feature distillation to multiple intermediate layers of the baseline. (b) Efect of modality decomposition: S (spatial), T (temporal), and ST (spatiotemporal). (c) Efect of CDL on Protocol 1 and robustness on $\mathrm { F F + + / D F 4 0 }$ (FS: face swap, FR: face reenactment). (d) Breakdown of MSA (MSR and MSDM). (e) Mask types: fixed high/low-pass vs. learnable Gumbel–Softmax in MSDM. (f) Backbone generalization under continual learning settings. Unless otherwise specified, all results are reported under Protocol 1.  
(a) Components.
<table><tr><td colspan="4">Model MSFD MSA CDL AUCall ↑ Favg↓</td></tr><tr><td>Baseline Multi-layer Distill.</td><td></td><td>89.84 90.64</td><td>6.81 6.34</td></tr><tr><td>MSFD (Naive)</td><td>vv</td><td>92.65</td><td>4.09</td></tr><tr><td>MSFD + MSA</td><td>√</td><td>92.14</td><td>3.20</td></tr><tr><td></td><td>√</td><td>92.77</td><td>2.65</td></tr><tr><td>MSFD + CDL</td><td>√ √ √ √</td><td>93.02</td><td>2.29</td></tr></table>

(d) MSA components.

(b) Modalities.
<table><tr><td>Modality</td><td>ST ST AUCall↑ Favg ↓</td><td></td></tr><tr><td>w/o Decompose Spatial Only Temporal Only</td><td></td><td>90.64 6.34 90.69 5.29 91.45 4.41</td></tr><tr><td>Spatiotemp. Only Full (Ours)</td><td>√ √√√</td><td>92.06 3.27 93.02 2.29</td></tr></table>

(c) Efect of CDL.
<table><tr><td rowspan="2">Method</td><td>Protocol 1</td><td></td><td>FF++</td><td colspan="2">DF40</td></tr><tr><td>AUCau ↑ Favg ↓</td><td></td><td>FS FR</td><td>FS</td><td>FR</td></tr><tr><td>w/o CDL</td><td>92.14</td><td>3.20</td><td>99.5 97.5</td><td></td><td>92.8 75.6</td></tr><tr><td>Ours</td><td>93.02</td><td>2.29</td><td>99.6 97.8</td><td></td><td>93.3 85.6</td></tr></table>

(f) Backbone.

<table><tr><td>Configuration</td><td>MSR MSDM</td><td> $A U C _ { a l l } \uparrow F _ { a v g } \downarrow$ </td><td></td></tr><tr><td>w/o MSA + MSR Only</td><td>√ 一</td><td>92.77 92.97</td><td>2.65</td></tr><tr><td>+ MSDM Only</td><td>一 √</td><td>91.91</td><td>2.43</td></tr><tr><td></td><td></td><td></td><td>3.08</td></tr><tr><td>MSA (Full)</td><td>√ √</td><td>93.02</td><td>2.29</td></tr></table>

(e) Mask type in MSDM.
<table><tr><td>Mask Type</td><td>Trainable</td><td>AUCaul↑ Favg↓</td><td></td></tr><tr><td>None High-pass</td><td>Fixed</td><td>92.97 92.94</td><td>2.43 2.53</td></tr><tr><td>Low-pass Ours</td><td>Fixed Learnable</td><td>92.38 93.02</td><td>3.14 2.29</td></tr></table>

<table><tr><td>Backbone</td><td>Method</td><td> $A U C _ { a l l } \uparrow F _ { a v g } \downarrow$ </td><td></td></tr><tr><td>R3D-18</td><td>|iCaRL-FC| Ours</td><td>89.84 93.02</td><td>6.81 2.29</td></tr><tr><td>R(2+1)D-18</td><td>iCaRL-FC Ours</td><td>86.10 89.24</td><td>8.75 2.17</td></tr><tr><td>FTCN</td><td>iCaRL-FC Ours</td><td>74.05 75.34</td><td>16.39 8.03</td></tr></table>

Analysis of Modality Decomposition. All three frequency modalities are necessary, and their combination yields the most robust performance (Table 4- (b)). Spatial-only (S) provides the smallest gain, as spatial cues are well preserved by standard distillation. Temporal-only (T) shows a larger improvement, consistent with prior findings that temporal cues are more generalizable forensic signals [11, 64]. Spatiotemporal-only (ST) achieves the strongest result by capturing joint space-time dynamics inaccessible to 1D or 2D transforms. The full model further improves, confirming complementarity across the three spectra.

Efect of Cross-modality Decorrelation Loss (CDL). CDL improves both accuracy and stability by preventing the ST branch from redundantly encoding cues already preserved by the spatial and temporal branches (Table 4-(c)). Without CDL, such redundancy wastes capacity and compounds forgetting when overlapping representations shift during later tasks. Furthermore, we find the largest improvement on Face Reenactment (FR) in DF40 [57], an advanced variant of FF++. Since FR modifies expression dynamics, its artifacts appear as temporally inconsistent spatial deformations, making spatiotemporal cues more critical than in Face Swap (FS).

![](images/7ce39caedc32ff361f446fede8a7e0c3b407301124d70fd5d3a36a52374973ae.jpg)  
Fig. 7: Efect of the Modality-Specific Adaptor (MSA) on frequency distributions. We visualize frequency spectra of real (blue) and fake (red) samples at Layer 1 across three modalities (spatial, temporal, and spatiotemporal). For each modality, the left plot shows spectra before MSA and the right plot after MSA.

Analysis of the Modality-Specific Adaptor (MSA). The combination of Modality-Specific Reweighting (MSR) and Modality-Specific Distillation Masking (MSDM) achieves the highest performance and lowest forgetting, demonstrating strong synergy between the two components as shown in Table 4-(d). MSR adaptively calibrates channel-wise spectral responses to reduce forgetting through modality-aware spectral reweighting, while MSDM selectively highlights discriminative frequency components via a Gumbel–Softmax mechanism. Their complementary roles, MSR for calibration and MSDM for frequency-wise selection, enable comprehensive spectral adaptation across all modalities.

To further validate the learnable masking mechanism, we compare the modalityspecific distillation masking against a fixed frequency filter. As shown in Table 4-(e), replacing the learnable mask with a static high-pass or low-pass filter degrades performance, indicating that fixed frequency selection cannot adapt to task-specific spectral characteristics. In contrast, our learnable mask dynamically allocates attention across frequency regions, achieving the best performance and confirming that adaptive frequency selection is crucial for capturing discriminative spectral cues and maintaining stability in continual deepfake video detection.

To analyze the impact of MSA, we visualize the frequency distributions of real and fake samples at Layer 1, representing low-level features, across spatial, temporal, and spatiotemporal modalities using our model. As shown in Fig. 7, the top row presents the distributions before applying MSA, while the bottom row illustrates those after applying MSA. Without MSA, the spectral responses of real and fake samples largely overlap. In contrast, after MSA is introduced, the separation between real and fake distributions becomes more pronounced across all modalities, demonstrating that MSA enables the model to learn forgeryrelevant cues even from the low-level representation stage.

Backbone Generalization Analysis. Our modality-aware frequency distillation generalizes across architecturally diverse backbones, consistently outperforming the baseline on all tested networks (Table 4-(f)). To further validate this generality, we evaluate our framework across diferent backbone architectures under identical continual learning settings, including the standard 3D ResNet-18 and R(2+1)D-18 video backbones [49], as well as the FTCN [64], a deepfakespecific backbone. Note that FTCN is originally equipped with a transformer, from which we remove the transformer modules due to memory constraints.

Failure Case. Despite consistent overall gains, our proposed method exhibits two failure modes. (i) Race-specific distribution shifts: after learning KoDF (predominantly Asian faces), MSFD’s forgetting rises from near-zero to ∼2 (Fig. 4). This suggests a demographic shift in KoDF: diferences in facial appearance change the spatial/frequency patterns learned earlier, causing more forgetting. (ii) Residual asymmetric retention: the spatial and temporal branches still degrade at uneven rates (Fig. 6).

## 5 Conclusion and Future Work

In this work, we introduced a novel framework for continual deepfake video detection that mitigates catastrophic forgetting through modality-specific frequencydomain knowledge preservation. Specifically, we proposed a Modality-Specific Frequency Distillation (MSFD) strategy that decomposes intermediate representations into spatial, temporal, and spatiotemporal spectra, enabling modalityspecific knowledge transfer. We further introduced the Modality-Specific Adaptor (MSA) for discriminative spectral calibration and masking, along with the Cross-Modality Decorrelation Loss (CDL) to encourage orthogonal frequency representations across modalities. Through comprehensive experiments across multiple continual learning protocols, unseen forgery domains, and perturbation conditions, our method demonstrated superior generalization, robustness, and stability compared to existing continual deepfake detection frameworks.

Future Work. Building upon the strong performance of memory-based replay methods in continual deepfake video detection, future work will explore deepfake video-specific memory update strategies that explicitly consider both spatial and temporal information to guide exemplar selection and maintenance. Additionally, developing adaptive mechanisms that dynamically balance the importance of each modality per task could further improve robustness across diverse generation paradigms and reduce sensitivity to hyperparameter choices.

## Acknowledgements

This work was supported in part by the Institute of Information & Communication Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2025-02263841, Development of a Real-time Multimodal Framework for Comprehensive Deepfake Detection Incorporating Common Sense Error Analysis; RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)), and by [Ministry of Science and ICT] (MSIT) through the [National Research Foundation of Korea] (NRF) funded by the Ministry of Science and ICT (RS-2025-16066667).

# Preserving Knowledge across Space and Time for Continual Video Deepfake Detection Supplementary Material

## A Additional Experiments

## A.1 Additional Analysis

Visualization of Representation Preservation To examine how each method retains previously learned knowledge, we visualize Grad-CAM [44] activation maps from the initial task (FF++) before and after continual training (Fig. A).

While existing methods exhibit significant drift in attention, shifting focus to irrelevant regions or losing sensitivity to deepfake artifacts, our approach maintains consistent activation around key forgery regions such as the mouth, eyes, and facial boundaries. This qualitative result highlights the effectiveness of our modality-specific frequency distillation in preserving early spatial–temporal representations even after multiple continual updates.

Generalization for Unseen Datasets As shown in Table A, our method achieves the highest average AUC across all unseen forgery datasets, outperforming prior continual detectors such as DFIL [36] and SURLID [4]. To assess unseen deepfake generalization, models that performed highly under Protocol 1 were trained under Pro-

Ours  
iCaRL  
STSP  
DFIL  
SURLID  
![](images/61d167bbe62ad0b864d9e1245712200d55e4eaf707c5e2b55f3da4ef0f28d313.jpg)  
Fig. A: Grad-CAM visualization of representation preservation. We compare Grad-CAM attention maps on FF++ (Task 1) between the initial model and each continual learning method after completing all tasks.

tocol 1 up to the DFDCp stage and subsequently evaluated on four unseen domains: FFIW, KoDF, FSh, and DFo. This result highlights that our modalityspecific frequency distillation achieves adaptive knowledge transfer by preserving the most relevant past representations while aligning them with new spatiotemporal cues, thus generalizing robustly to unseen deepfakes.

Robustness Analysis To further assess the robustness of our method, we evaluate the models on FF++ under four common perturbations: H.264 video compression (FF++ C40), resize, grid shufle, and frame shufle. As shown in

Table A: Unseen deepfake generalization. All models are trained sequentially on datasets up to DFDCp on Protocol 1, and evaluated on unseen forgery datasets (FFIW, KoDF, FSh, DFo).  
Table B: Perturbation robustness analysis. After completing all tasks, we evaluate robustness on FF++ under four perturbations: compression, resize, frame shufle, and grid shufle.
<table><tr><td>Method</td><td>FFIW KoDF</td><td>FSh</td><td>DFo</td><td>Avg. AUC ↑</td></tr><tr><td>Finetune</td><td>66.45</td><td>85.53 69.26</td><td>82.51</td><td>75.94</td></tr><tr><td>iCaRL-FC [39]</td><td>69.11</td><td>84.58 79.61</td><td>92.66</td><td>81.49</td></tr><tr><td>DFIL [36]</td><td>68.33</td><td>86.89 76.03</td><td>83.65</td><td>78.73</td></tr><tr><td>SURLID [4]</td><td>69.29</td><td>82.60 71.78</td><td>90.72</td><td>78.60</td></tr><tr><td>Ours</td><td>72.53</td><td>85.63</td><td>86.1890.47</td><td>83.70</td></tr></table>

<table><tr><td>Method</td><td colspan="4">Compression Resize Frame Shuffle Grid Shuffle</td></tr><tr><td>Finetune</td><td>67.21</td><td>73.31</td><td>70.28</td><td>68.43</td></tr><tr><td>iCaRL-FC [39]</td><td>72.29</td><td>79.61</td><td>71.22</td><td>71.23</td></tr><tr><td>DFIL [36]</td><td>70.74</td><td>77.88</td><td>64.36</td><td>72.81</td></tr><tr><td>SURLID [4]</td><td>70.54</td><td>78.07</td><td>65.87</td><td>64.94</td></tr><tr><td>Ours</td><td>77.61</td><td>85.86</td><td>77.71</td><td>77.34</td></tr></table>

Table C: Long-sequence (12-task Fake-Incremental). We construct a 12-task long sequence under the Fake-Incremental setting: starting from FF++, we sequentially add 6 DF40 [57] forgery types (BlendFace, FaceDancer, MobileSwap, Hyperreenact, MCNet, Sadtalker), followed by 5 deepfake datasets (DFD, CDF, DFDCp, FFIW, KoDF) as fake-only tasks.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1> $A U C _ { \mathrm { a l l } }$ ↑ $F _ { \mathrm { a v g } }$ ↓</td></tr><tr><td rowspan=1 colspan=1>iCaRL-FC</td><td rowspan=1 colspan=1>86.76   3.21</td></tr><tr><td rowspan=1 colspan=1>DFIL</td><td rowspan=1 colspan=1>85.11   2.39</td></tr><tr><td rowspan=1 colspan=1>SURLID</td><td rowspan=1 colspan=1>85.16   2.79</td></tr><tr><td rowspan=1 colspan=1>STSP</td><td rowspan=1 colspan=1>67.43   -4.58</td></tr><tr><td rowspan=1 colspan=1>MSFD (Ours)</td><td rowspan=1 colspan=1>87.44  0.85</td></tr></table>

Table B, while existing continual learning approaches (e.g., iCaRL-FC, DFIL, and SURLID) exhibit notable drops in performance, our model maintains consistently higher performance across all conditions. In particular, ours performs robustly under both temporal (frame shufling) and spatial (grid shufling) perturbations, demonstrating its ability to efectively preserve both spatial and temporal information.

Scalability to long task sequences We construct a 12-task long sequence under the Fake-Incremental setting to evaluate scalability to long task sequences. MSFD achieves the highest $A U C _ { \mathrm { a l l } }$ (87.44) with competitive forgetting (Tab. C), confirming graceful scaling under long-term forgery evolution. STSP attains a lower nominal $F _ { \mathrm { a v g } }$ but only by sacrificing plasticity $( A U C _ { \mathrm { a l l } } = 6 7 . 4 3 )$ , failing to adapt to new forgeries; a negative $F _ { \mathrm { a v g } }$ here reflects a model that barely learns rather than one that forgets little.

## A.2 Additional Ablation Study

Additional Analysis for Efect of the Modality-Specific Adaptor (MSA) To analyze the impact of MSA, we further visualize the frequency distributions of real and fake samples at layers 2-4, representing features, across spatial, temporal, and spatiotemporal modalities using our model. As shown in Fig. B, each subfigure compares the real–fake frequency spectra before (top) and after (bottom) applying the Modality-Specific Adaptor (MSA) across Layers 2–4. Before applying MSA, the spectral responses of real and fake videos substantially overlap, indicating limited discriminability. After applying MSA, however, the separation between real and fake distributions becomes consistently more pronounced in all modalities—spatial, temporal, and spatiotemporal—demonstrating that MSA efectively amplifies forgery-relevant spectral cues at every layer.

Spatial  
![](images/5ec0561b013733053ed11f225f4735e2c0fd826610970f4a09c6523459763db3.jpg)

Temporal  
![](images/7e991ba8217ea36e147dd457372cb53c602dc711fd6a305b79c985d380baddee.jpg)  
(a) Layer 2

Spatiotemporal  
![](images/6f2b2f3c52a520f3027ada211abe95ad4f060ac7ed4f2df201c353e410de56f4.jpg)

![](images/4cc0dcd2644aef00baf340c69d6fc24aa1109ab9e347802db7fc558510873375.jpg)

![](images/a9b665e844d66272a26be6a5776c268dfba7c7cdcea0a25d703f4664aa3d9110.jpg)  
(b) Layer 3

Spatiotemporal  
![](images/a8619e1c1976a899ebd0aef2474911de19ff2504beb691eeea74289e0024d89d.jpg)

![](images/0a6b5dc9ba421e8a945971ecec0dd9161818d39fa38939cd3ef3e7da6f284e9b.jpg)

![](images/30c1ca903da55bcf86f42a90982a6c75ac3d6c3c3b9ece9db8c2852fbc185019.jpg)

Spatiotemporal  
![](images/382a74aa998041e82666d055a9cc4b31cc22b0fe96eb56d27fb0d8c12d8cbb95.jpg)  
(c) Layer 4  
Fig. B: Additional analysis for efect of the Modality-Specific Adaptor (MSA) across Layers 2–4. Frequency spectra of real (blue) and fake (red) samples are visualized across the spatial, temporal, and spatiotemporal modalities for Layers 2 to 4. For each layer, the upper row shows distributions before applying MSA, and the lower row shows distributions after applying MSA. Each distribution is computed using 100 real and 100 fake samples.

Analysis on Modality Decomposition Eficacy To verify whether our framework efectively decomposes video features into meaningful spatial and temporal components, we analyze the performance of individual modality branches: Spatial (S), Temporal (T), and Spatiotemporal (ST). Each model using only the corresponding single modality is trained without CDL. Also, we categorize the manipulation types in FF++ into Face Swap (DF, FS), which involves replacing the entire face identity, and Face Reenactment (F2F, NT), which manipulates expressions and movements while maintaining identity.

Table D: Additional analysis for the efect of modality decomposition. We analyze the contribution of each frequency modality: spatial (S), temporal (T), and spatiotemporal (ST). Results are reported on FF++ with four manipulation types, grouped by their characteristics: face swap deepfakes (DF, FS) and face reenactment deepfakes (F2F, NT). We denote video-level AUC (%).
<table><tr><td>Model</td><td>Face Swap DF FS</td><td>F2F</td><td>Face Reenactment NT</td><td>Avg.</td></tr><tr><td>Spatial Only (S)</td><td>99.88 97.95</td><td>97.79</td><td>94.93</td><td>97.64</td></tr><tr><td>Temporal Only (T)</td><td>99.88 97.86</td><td>98.36</td><td>95.44</td><td>97.89</td></tr><tr><td>Spatiotemporal Only (ST)</td><td>99.95 99.42</td><td>97.94</td><td>94.94</td><td>98.06</td></tr><tr><td>Ours (Full)</td><td>99.83 99.33</td><td>99.01</td><td>96.67</td><td>98.71</td></tr></table>

As presented in Table D, the performance trends across diferent manipulation types provide empirical evidence for the efectiveness of our modality decomposition. For Face Swap (DF, FS), which introduces significant spatial and temporal anomalies, every model achieves high accuracy, indicating that artifacts are prominent in both domains. However, for Face Reenactment (F2F, NT), where temporal consistency (e.g., mouth movement, expression jitter) is a key discriminatory factor, the temporal only model (T) consistently outperforms the spatial only model (S), confirming that our framework efectively decomposes and utilizes spatial and temporal cues. While single-modality models show limitations in specific categories, our Full Model (Ours) integrates these complementary cues to achieve the highest Average AUC. This substantial gain validates that the synergy between spatial and temporal representations is essential for robustly detecting sophisticated manipulation types.

Analysis on Frequency Components Frequency representations consist of two components: magnitude and phase. To examine their respective efects, we compare models built on the same overall framework, difering only in the frequency component used for distillation, as shown in Table E.

The results show that using magnitude leads to better continual deepfake detection performance than using phase, achieving higher $A U C _ { a l l }$ and lower average forgetting $F _ { a v g } .$ In contrast, when phase is used, the training process frequently becomes unstable during continual updates, and the distillation loss often diverges, eventually producing NaN values, as illustrated in Fig. C. We believe this is be-

![](images/c6ce4157cd798839f7791472d7e8326564ece66005bfed6ddebd688c3d321b7e.jpg)  
Fig. C: Visualization distillation loss. We compare the feature distillation loss $\mathcal { L } _ { F D L }$ when using phase and magnitude components.

Table E: Analysis of frequency component. We analyze the frequency components, including phase and magnitude.
<table><tr><td>Component</td><td> $\overline { { A U C _ { a l l } } }$  ↑</td><td> $\overline { { F _ { a v g } \downarrow } }$ </td></tr><tr><td>Phase</td><td>90.04</td><td>5.53</td></tr><tr><td>Magnitude (Ours)</td><td>93.02</td><td>2.29</td></tr></table>

Table F: Analysis of memory update strategy. We evaluate diferent memory update strategies in our framework.
<table><tr><td>Strategy</td><td> $\overline { { A U C _ { a l l } } }$  ↑</td><td> $\overline { { F _ { a v g } \downarrow } }$ </td></tr><tr><td>Random [40]</td><td>93.01</td><td>3.89</td></tr><tr><td>Center (default) [39]</td><td>93.02</td><td>2.29</td></tr></table>

cause the phase component is highly sensitive to small changes in the input and can vary abruptly across tasks, making it less stable for preserving transferable knowledge in continual learning. These observations support our choice of magnitude as the frequency representation in the proposed method.

Analysis of Memory Update Strategy We analyze the efect of diferent memory update strategies within our framework, including random replacement and class-center based update, as shown in Table F. The overall performance remains largely consistent across strategies, with only marginal diferences in $A U C _ { a l l }$ , suggesting that our framework is stable with respect to the choice of memory update policy.

Nevertheless, the diference in average forgetting $F _ { a v g }$ is more pronounced. Although random replacement achieves comparable $A U C _ { a l l } ,$ , it leads to higher forgetting than the center-based update. This implies that the memory update strategy plays an important role in preserving previously acquired knowledge over time. In particular, our results suggest that future memory construction schemes may benefit from explicitly considering the relative importance of spatial and temporal cues, rather than treating all samples equally during memory update.

Further Analysis on Modality Decomposition To isolate the contribution of the frequency transform itself, we add two variants (Tab. G): (i) MSFD w/o $F F T ,$ a deeper ablation that removes FFT from MSFD while keeping other components (e.g., MSA, CDL); and (ii) Conv. Decomp., a simpler alternative that replaces frequency decomposition with per-modality 3D convolutions (1×3×3 / 3×1×1 / 3×3×3 for ${ \mathsf { S } } / { \mathsf { T } } / { \mathsf { S } } { \mathsf { T } } ,$ ). Both fall short of MSFD, suggesting frequencydomain decomposition contributes beyond axis-separation alone.

Table G: Further analysis of modality decomposition. We compare MSFD against a simpler convolutional alternative and an FFT-free variant under Protocol 1. Conv. Decomposition replaces the frequency transform with per-modality 3D convolutions; MSFD w/o FFT removes the FFT while keeping MSA and CDL. We report video-level AUC<sub>all</sub> (%) and average forgetting $F _ { \mathrm { a v g } } .$
<table><tr><td rowspan=1 colspan=2>Variant                             $A U C _ { \mathrm { a l l } }$ ↑ $F _ { \mathrm { a v g } } \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>Multi-layer Distill. (raw feature)Conv. Decomposition (3D conv)MSFD $\mathrm { w } / \mathrm { o }$ FFT</td><td rowspan=1 colspan=1>90.64   6.3491.64   4.6492.21   3.87</td></tr><tr><td rowspan=1 colspan=1>Full MSFD (Ours)</td><td rowspan=1 colspan=1>93.02  2.29</td></tr></table>

Table H: Robustness to frequency-suppressing perturbations. After completing all tasks under Protocol 1, we report video-level AUC (%) on FF++ under three perturbations that attenuate high-frequency forensic cues: Gaussian blur, WebP compression, and a 3D-FFT low-pass filter (cutof = 0.5).
<table><tr><td>Method</td><td colspan="3">Gaussian Blur WebP 3D-FFT Low-pass</td></tr><tr><td>iCaRL-FC</td><td>47.4</td><td>77.2</td><td>71.3</td></tr><tr><td>DFIL</td><td>53.3</td><td>66.6</td><td>66.5</td></tr><tr><td>MSFD (Ours)</td><td>74.9</td><td>85.5</td><td>74.5</td></tr></table>

Table I: Comparison with static video-based deepfake detection approaches. Results without a marker are quoted from the original papers, whereas <sup>∗</sup> indicates reproduced results under a diferent evaluation setting.
<table><tr><td>Method</td><td>Architecture</td><td>[Params.</td><td>[CDF FFIW</td></tr><tr><td>STA [59] (CVPR 2025)</td><td>CLIP-B16 + Adapter</td><td>106M</td><td>86.6</td></tr><tr><td>PwTF [20] (ICCV 2025)</td><td>FTCN + Transformer</td><td>164M</td><td>89.7  ${ 7 5 . 2 ^ { * } }$ </td></tr><tr><td>Ours(MSFD)</td><td>3DResNet-18</td><td>33M</td><td>90.2 94.9</td></tr></table>

Table J: Comparison with deepfake video detectors (sequential FT).
<table><tr><td>Method</td><td> $\mathrm { F F + + \ D F D }$  CDF  $\mathrm { D F D C p }$  FFIW KoDF</td><td> $A U C _ { a l l }$ </td><td>↑  $F _ { \mathrm { a v g } } \ 、$ </td></tr><tr><td>AltFreezing [53] (seq. FT) PwTF [20] (seq. FT)</td><td>62.97 778.38 58.10 62.00 57.25 98.71 97.28 84.80 85.26 76.01 52.95</td><td>69.57 82.37</td><td>27.11 9.27</td></tr><tr><td>Ours</td><td>98.42 91.15 90.17 84.14 94.88</td><td>97.93 99.36</td><td>93.02 2.29</td></tr></table>

Robustness to Frequency-Suppressing Perturbations We apply three frequency-suppressing perturbations on FF++ (Gaussian blur, WebP, 3D-FFT low-pass cutof): MSFD retains substantially higher AUC than both iCaRL and DFIL in every case, confirming our frequency preservation provides complementary cues robust to frequency-domain manipulation.

Comparison with Static Deepfake Video Detection Approaches Our continual learning framework achieves competitive detection performance with ≈ 5× fewer parameters compared to static video-based detectors (Table I). This highlights the practical advantage of continual adaptation: rather than training a single large model on all forgery types at once, MSFD incrementally absorbs new forgery patterns while preserving previously learned cues through modality-wise distillation. Tab. J verifies the necessity: the strongest static spatiotemporaldisentangled detectors AltFreezing [53] and PwTF [20] collapse under sequential fine-tuning $( F _ { \mathrm { a v g } }$ of 27.11 and 9.27), confirming that decomposition-for-detection does not transfer to continual learning.

Computational Overhead We report deployment and adaptation costs in Tab. K.

While MSFD consumes more training memory than baselines, this is a necessary consequence of multilayer modality decomposition (distilling {S, T, ST} at intermediate layers). Crucially, all auxiliary modules are training-only and discarded after adaptation: the deployed model has identical FLOPs and memory as baselines while achieving the highest $A U C _ { a l l }$ . Replay storage is also identical to other memory-based methods (iCaRL, DFIL, SURLID).

Table K: Computational & deployment overhead. We report deploy-time cost (FLOPs, clips/sec), training cost (memory, number of knowledge distillation (KD) layers), and overall performance $\left( A U C _ { a l l } \right)$ on Protocol 1.
<table><tr><td>Method</td><td>Deploy FLOPs(G) Clip/s</td><td>Train Mem(MB) KD</td><td> $A U C _ { a l l } ~ ^ { \prime }$  1</td></tr><tr><td>iCaRL-FC</td><td>325.9 45.5</td><td>14,756 0</td><td>89.84</td></tr><tr><td>SURLID</td><td>325.9 44.3</td><td>12,869 1</td><td>87.61</td></tr><tr><td>DFIL</td><td>325.9 45.5</td><td>14,756 1</td><td>87.88</td></tr><tr><td>Ours</td><td>325.9 45.5</td><td>28,287 4</td><td>93.02</td></tr></table>

## B Additional Implementation Details

## B.1 Modality-Specific Distillation Mask Generation

As shown in Fig. D, we construct a modality-specific radial mask to select informative frequency regions in a data-driven manner. For each modality $d \in$ {T, S, ST}, the radial axis of the corresponding frequency spectrum is discretized into $K _ { d }$ bins, and each bin k is assigned a learnable logit vector $\mathbf { a } _ { d , k } ^ { ( l ) } \in \mathbb { R } ^ { 2 }$ . The Gumbel–Softmax operator produces a binary pass/block decision $m _ { d , k } ^ { ( l ) } \in \{ 0 , 1 \}$ , which is then mapped to radial frequency locations via $\kappa _ { d } ^ { ( l ) } ( \mathbf { u } )$ to form the final mask ${ \cal M } _ { d } ^ { ( l ) } ( { \bf u } )$ . The resulting mask is broadcast to the corresponding modality spectrum and applied element-wise, enabling selective preservation of discriminative frequency regions. As illustrated in Fig. D-(b), this yields modality-specific mask structures: 1D bands for T, concentric radial bands for ${ \mathsf { S } } ,$ and spherical frequency shells for ST.

![](images/39e295e5123e5bdca533f37f42369a48be68ac9f537af91fd1ae972001eab9f3.jpg)  
Fig. D: Visualization of mask generation. (a) Learnable bin-wise logits are transformed into binary pass/block decisions via Gumbel–Softmax and mapped to radial frequency locations to generate the modality-specific distillation mask. (b) Generated mask examples for the temporal, spatial, and spatiotemporal modalities.

## B.2 Dataset Configurations

For continual learning protocols, Task 1 (FF++) uses the original training and test splits. For subsequent tasks (Task 2-6), we sample 200 real and 200 fake videos for training from each dataset. Test set sizes are as follows: FaceForensics++ (FF++) with 1400 videos, DFD contains 3,000 videos (2,531 fake), DFDCp contains 758 videos (488 fake), CDF contains 918 videos (510 fake), FFIW [65] contains 500 videos (250 fake), and KoDF [22] contains 3,000 videos (2,500 fake). CDF denotes the Celeb-DF-V2 dataset. For DF40 [57], we use deepfakes sourced from FF++ [42]. For KoDF, due to the large size of the original training set, we construct non-overlapping train and validation sets using only the validation portion of the dataset. To ensure balanced evaluation, we randomly sample 500 videos per forgery method from the remaining test videos, excluding those used for training. For AIGVDBench [31], due to the large size of the original training set, we construct non-overlapping train and validation sets through random sampling. Specifically, we first sample 2,000 real videos and 2,000 fake videos per forgery category. From these, 200 samples are allocated for training, while the remaining 1,800 videos are used for validation to ensure a robust evaluation.

## B.3 Perturbation Settings

We evaluate the robustness of our model using four perturbation types. First, to simulate real-world quality degradation, we employ Video Compression (using the FF++ C40 dataset) and Resizing, where frames are downsampled to a scale of 0.5 and subsequently upsampled to the original resolution. These perturbations allow us to assess model stability under common transmission artifacts and reduced resolutions. Secondly, to investigate the model’s reliance on specific modalities, we apply structural perturbations, including grid shufle and frame shufle, as visualized in Fig. E. Grid Shufle rearranges local patches within frames using a G×G grid (we set G = 14, yielding 16×16 pixel patches at 224 × 224 resolution) to break global spatial structure, whereas Frame Shufle groups every K consecutive frames into segments and randomizes the segment order (we set K = 1, i.e., individual frames are fully shufled) to disrupt motion continuity. For the visualization in Fig. 2, we use $G = 7$ and $K = 2$ for clearer visual illustration. These settings allow us to efectively decouple the contribution of spatial and temporal features under adversarial conditions.

To investigate how continual learning afects spatial and temporal knowledge independently, we measure modality-specific retention after all tasks have been learned sequentially (Fig. 2, Fig. 6). Specifically, we define spatial retention as the ratio of the final model’s AUC under grid-shufle perturbation to its AUC when the dataset was first encountered, and temporal retention analogously under frame-shufle perturbation. A value closer to 1.0 indicates that the model preserves its original discriminative capability in that modality despite subsequent task updates.

## B.4 Training Configuration

During training, all frames in a clip share the same augmentation parameters to preserve temporal coherence. Each frame is first resized to an intermediate scale $s \sim [ 2 5 6 , 3 2 0 ]$ , followed by a random crop with size c ∼ [frame\_size, s]. A horizontal flip with probability 0.5 is then applied at the clip level. We additionally apply stochastic masking with up to three holes, where the generated binary mask is broadcast across channels and multiplied with each frame. After augmentation, frames are resized to 224 × 224 and normalized using ImageNet [43] statistics (mean: (0.485, 0.456, 0.406); std: (0.229, 0.224, 0.225)). The processed frames are then stacked along the temporal dimension to form a clip tensor of size [C, T, H, W], where $C = 3 , T = 3 2$ , and $H = W = 2 2 4$ . At evaluation time, only resizing and normalization are applied. In the R3D-18 [49] backbone, which we generally adopt, the number of layers used for feature extraction is $L = 4$ The loss weighting coeficients are set as λ<sub>PD</sub> is 1.0, $\lambda _ { \mathrm { F D L } }$ is 1.0, and λ<sub>CDL</sub> is 0.5. For modality-specific frequency distillation, the weights are $\lambda _ { S }$ is 0.5, $\lambda _ { T }$ is 1.0, and $\lambda _ { S T }$ is 1.0.

Most experiments were carried out on either four NVIDIA A6000 GPUs (48GB each) paired with an AMD Ryzen Threadripper Pro 3955WX 16-core CPU, or on two NVIDIA A100 GPUs (40GB each) coupled with an Intel Xeon Gold 6240R CPU. Training on Protocol 1 takes approximately 30 minutes per epoch using A100 GPUs.

## B.5 Configuration of Comparison Methods

We provide detailed implementation information for each continual learning method.

For the Nearest Mean of Exemplars classifier (NME), the probability of a sample x being fake is computed as:

$$
p _ { \mathrm { f a k e } } ( x ) = \sigma ( d _ { \mathrm { r e a l } } - d _ { \mathrm { f a k e } } ) ,\tag{1s}
$$

where the distances to the class prototypes are defined as $d _ { \mathrm { c l s } } = \lVert f - \mu _ { \mathrm { c l s } } \rVert _ { 2 }$ , cls $\in$ $\{ r e a l , f a k e \}$ , and $\begin{array} { r } { f = \phi ( x ) , \quad \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } } } \end{array}$ . Here, f denotes the L2-normalized feature vector extracted from the backbone $\theta , \mu _ { \mathrm { r e a l } }$ and $\mu _ { \mathrm { f a k e } }$ represent the class mean prototypes computed from exemplars, $d _ { \mathrm { r e a l } }$ and $d _ { \mathrm { f a k e } }$ are their Euclidean distances, and $\sigma ( \cdot )$ is the sigmoid function.

![](images/7e17472e89739915d23de8b2aafbc789f96c2d4798add281aa77b71948ec0d2c.jpg)  
Fig. E: Visualization of frame and grid shufle. To assess the stability of our learned representations and conduct the preliminary analysis (as discussed in Fig. 2 in the main paper), we apply frame shufling and grid shufling to perturb temporal continuity and spatial structure, respectively. The first row shows the raw video clip. The second row shows frame shufling, where the temporal order is randomized. The third row depicts grid shufling, where spatial patches are rearranged within each frame using a consistent permutation across the clip while preserving the temporal order. The numbers overlaid on each frame indicate the original frame index.

ER [40]. We reproduce ER [40] following the original paper. The method maintains a memory bufer populated with randomly selected exemplars, where class balance is preserved via simple class-wise pruning. During continual training, current-task samples are mixed with replayed exemplars to form each minibatch, and the model is optimized solely with standard classification loss on this combined batch. No additional regularization, distillation, or task-specific constraints are introduced, ensuring a faithful reproduction of the baseline ER protocol.

EWC [21]. We reproduce EWC [21] following the original paper. After completing each task, we estimate the Fisher Information Matrix using samples from the current task and accumulate it across tasks to model parameter importance. The quadratic regularization term is then applied during training to penalize deviations from important parameters of previous tasks. Following common practice, we set the regularization coeficient to $\lambda = 1 0 0 0$ for all experiments.

iCaRL [39]. We reproduce iCaRL [39] using herding-based exemplar selection as described in the original work. During training, the model is optimized using a combination of classification loss on mixed current–memory batches and a distillation loss computed on memory samples, both weighted equally at 1.0. For evaluation, we report results using both the Nearest Mean of Exemplars (NME) classifier, computed with exemplar-based class means described in Equation 1s, and the standard Fully-Connected (FC) linear classifier

IDER [29]. We reproduce IDER [29] following their oficial implementation, adapting its idempotent experience replay to the binary deepfake setting. The

R3D-18 backbone is split into a feature stage $f _ { 1 }$ and a label-prior-conditioned prediction stage $f _ { 2 }$ , and IDER enforces idempotency by applying $f _ { 2 }$ twice with cross-entropy on both passes so the prediction converges to a stable fixed point.

DCE [25]. We adapt the domain-incremental learning method DCE [25] to continual deepfake video detection following their oficial implementation. DCE expands a pool of three collaborative experts per task: a naive expert trained with plain cross-entropy, together with balanced and reverse experts whose objectives add the log class prior once and twice, so the three heads specialize to the natural, balanced, and minority-emphasizing views of the same features. Following the original setup, the backbone is shared and frozen after the first task, and at each subsequent task, only the current task’s three experts are updated while all earlier experts remain fixed. After expert training, we estimate per-class Gaussian prototypes and train a dynamic selector on synthetic features sampled from these prototypes with task-distance-dependent scaling, so no raw exemplars are retained; at inference the selector produces per-expert weights that fuse all experts’ logits into the final real–fake decision.

vCLIMB [50]. We reproduce vCLIMB [50] following their oficial implementation code. The method extends iCaRL replay with a temporal consistency regularization that enforces prediction invariance between original and temporally downsampled clips. During training, each clip is paired with a temporally downsampled view reconstructed via trilinear interpolation, and the final loss blends their classification objectives. We adopt herding-based exemplar selection and report results using NME classifiers.

Drift [14]. We adapt the video domain-incremental learning method Drift [14] to the deepfake video detection setting. Drift employs a dual-loss formulation that combines standard classification loss on both current and replayed samples with a knowledge distillation term to preserve past representations. Following the original setup, we set the distillation weight to 1.0 and the temperature to 2.0.

HCE [28]. We compare the conventional video class-incremental learning method HCE [28] with our approach for continual deepfake video detection. HCE computes spatial hypercorrelation matrices between intermediate features of the current and previous models across backbone layers, aggregates them through 4D convolutions, and uses the resulting hypercorrelation weights to modulate the feature-level distillation objective, following the original setting where both distillation and hypercorrelation supervision terms are given equal importance. In addition, the Correlation Refinement Mechanism (CRM) adjusts the crossentropy loss based on class-wise hypercorrelation variance ratios, with CRM contributing half the weight of the main classification term to address distribution shifts between old and new data. For memory construction, we adopt herding-based exemplar selection, and inference follows the original design using an NME classifier for real–fake prediction.

STSP [3]. We reproduce STSP [3] following the original paper. The method learns compact subspace bases per class via a Task-Specific Classifier (TSC), where each class is represented by an orthogonal basis of rank $\tau = 8$ in the feature space. Classification is performed by comparing subspace distances: the logit is computed as $d _ { \mathrm { r e a l } } - d _ { \mathrm { f a k e } }$ , where $d _ { c l s }$ measures the Frobenius distance between the feature’s self-correlation and the class subspace projection. The training objective combines classification loss $( \alpha = 1 . 0 )$ , a feature orthogonality loss $( \beta = 0 . 0 5 )$ , and a weight orthogonality loss $( \gamma = 0 . 0 5 )$ that enforces both intra-class orthonormality and inter-class decorrelation of the subspace bases. To mitigate forgetting, Subspace Gradient Projection (SGP) constrains gradient updates to the orthogonal complement of previously learned input subspaces, with the energy retention threshold set to 0.99. SGP is applied to all convolutional and linear layers in the backbone as well as the TSC bases, and no replay bufer is used.

ESSENTIAL [23]. We reproduce ESSENTIAL [23] following their oficial implementation code. The method builds on a CLIP-based backbone and introduces learnable semantic prompts that capture task-specific knowledge through cross-attention with input features. Training combines a static matching loss and a temporal matching loss, both weighted equally at 1.0, to align prompt representations with frame-level and sequence-level features. We set the semantic prompt length to 32 (matching the number of input frames), use cross-attention as the prompt interaction mode, and adopt task-wise memory construction with a bufer size of 200 per task. Each task is trained for 30 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$

SARB-DF [38]. We reproduce SARB-DF [38] following the original paper. For memory update, we use a self-adaptive replay bufer mechanism that dynamically adjusts the composition of stored exemplars based on sample dificulty and representativeness. The training objective combines standard classification loss with a replay-based regularization term (EWC [21]) to mitigate forgetting across sequential deepfake detection tasks. We set the total memory bufer size to 500 and follow the original hyperparameter settings for all experiments.

DFIL [36]. We reproduce DFIL following their oficial implementation code. The training objective consists of a classification loss, a supervised contrastive loss $( \tau = 0 . 1 )$ , feature-level distillation, and a logit-level knowledge distillation term $( T = 2 0 )$ , each applied with equal weight (1.0). For memory selection, we adopt DFIL’s hybrid replay strategy: half of the exemplars are chosen based on high entropy (hard samples), while the remaining half are selected according to their proximity to the class centroid (central samples).

HDP [47]. We reproduce HDP [47] following the oficial implementation. Instead of storing raw exemplars, the method utilizes Universal Adversarial Perturbations (UAPs) generated from each task’s real videos as compact surrogates for historical distributions. Each UAP (δ) is optimized over 10 iterations with a step size of $\eta = 0 . 0 1$ , targeted to maximize the model’s prediction toward the fake class. During subsequent tasks, pseudo-fake samples are synthesized on the fly by adding a randomly selected historical UAP to the current real videos. The input size is set to 224×224 with 32 frames per clip. Crucially, for our experiments, the UAPs are stored with a temporal dimension of 32 $( 1 \times 3 \times 3 2 \times 2 2 4 \times 2 2 4 )$ , matching the clip length to ensure precise frame-wise additive synthesis throughout the video sequence.

SURLID [4]. We reproduce SURLID [4] following their oficial implementation code. The method introduces aligned feature isolation to reduce catastrophic forgetting in incremental deepfake detection and builds on three core components: (1) Sparse Uniform Replay (SUR), which selects stable and uniformly distributed exemplars based on shufle consistency and spatial distribution; (2) Distribution Refilling (DR), which augments replay features by mixing samples with their domain centroids; and (3) Incremental Decision Alignment (IDA), which encourages angular consistency between new and previous task-specific classifiers $( \gamma = 0 . 0 0 1 )$ ). Training follows the original SURLID loss, combining isolation loss (supervised contrastive), feature distillation with weight 0.5, and detection loss also weighted by 0.5, which consistently yielded better performance in our continual deepfake video detection experiments. For memory construction, we adopt SUR sampling, selecting exemplars whose representations remain stable across perturbations (measured using grid shufling).

## References

1. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., Rombach, R.: Stable video difusion: Scaling latent video difusion models to large datasets. CoRR abs/2311.15127 (2023). https://doi.org/10.48550/ARXIV.2311.15127, https://doi.org/10.48550/arXiv.2311.15127

2. Bounareli, S., Tzelepis, C., Argyriou, V., Patras, I., Tzimiropoulos, G.: Hyperreenact: one-shot reenactment via jointly learning to refine and retarget faces. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 7149–7159 (2023)

3. Cheng, H., Yang, S., Wang, C., Zhou, J.T., Kot, A.C., Wen, B.: Stsp: Spatialtemporal subspace projection for video class-incremental learning. In: European Conference on Computer Vision. pp. 374–391. Springer (2024)

4. Cheng, J., Yan, Z., Zhang, Y., Hao, L., Ai, J., Zou, Q., Li, C., Wang, Z.: Stacking brick by brick: Aligned feature isolation for incremental face forgery detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13927–13936 (2025)

5. Choi, J., Kim, T., Jeong, Y., Baek, S., Choi, J.: Exploiting style latent flows for generalizing deepfake video detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1133–1143 (2024)

6. Chollet, F.: Xception: Deep learning with depthwise separable convolutions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1251–1258 (2017)

7. Dolhansky, B., Howes, R., Pflaum, B., Baram, N., Canton-Ferrer, C.: The deepfake detection challenge (DFDC) preview dataset. CoRR abs/1910.08854 (2019), http://arxiv.org/abs/1910.08854

8. Dufour, N., Gully, A.: Contributing Data to Deepfake Detection Research — ai.googleblog.com. https://ai.googleblog.com/2019/09/contributing-datato-deepfake-detection.html (2019), [Accessed 30-07-2023]

9. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., Bibi, O.: Ltx-video: Realtime video latent difusion. CoRR abs/2501.00103 (2025). https://doi.org/10.48550/ARXIV.2501.00103, https://doi.org/10.48550/arXiv.2501.00103

10. Haliassos, A., Mira, R., Petridis, S., Pantic, M.: Leveraging real talking faces via self-supervision for robust forgery detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14950– 14962 (June 2022)

11. Haliassos, A., Vougioukas, K., Petridis, S., Pantic, M.: Lips don’t lie: A generalisable and robust approach to face forgery detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5039–5049 (2021)

12. Han, Y.H., Huang, T.M., Hua, K.L., Chen, J.C.: Towards more general videobased deepfake detection through facial component guided adaptation for foundation model. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22995–23005 (2025)

13. Hong, F.T., Xu, D.: Implicit identity representation conditioned memory compensation network for talking head video generation. In: ICCV (2023)

14. Hu, Y., Hou, J., Liu, X., Sun, X., Guo, W.: Video domain incremental learning for human action recognition in home environments. In: International Conference on Image and Graphics. pp. 316–327. Springer (2025)

15. Jeong, Y., Kim, D., Min, S., Joe, S., Gwon, Y., Choi, J.: Bihpf: Bilateral high-pass filters for robust deepfake detection. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 48–57 (2022)

16. Jeong, Y., Kim, D., Ro, Y., Choi, J.: Frepgan: robust deepfake detection using frequency-level perturbations. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 1060–1068 (2022)

17. Jiang, L., Li, R., Wu, W., Qian, C., Loy, C.C.: DeeperForensics-1.0: A large-scale dataset for real-world face forgery detection. In: CVPR (2020)

18. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for eficient video generative modeling. In: International Conference on Learning Representations. vol. 2025, pp. 23378–23402 (2025)

19. Kim, M., Tariq, S., Woo, S.S.: Cored: Generalizing fake media detection with continual representation using distillation. In: Proceedings of the 29th ACM International Conference on Multimedia. pp. 337–346 (2021)

20. Kim, T., Choi, J., Jeong, Y., Noh, H., Yoo, J., Baek, S., Choi, J.: Beyond spatial frequency: Pixel-wise temporal frequency-based deepfake video detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 11198–11207 (October 2025)

21. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114(13), 3521–3526 (2017)

22. Kwon, P., You, J., Nam, G., Park, S., Chae, G.: Kodf: A large-scale korean deepfake detection dataset. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10744–10753 (October 2021)

23. Lee, J., Bae, K., Min, K., Park, G.M., Choi, J.: Essential: Episodic and semantic memory integration for video class-incremental learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17546–17556 (2025)

24. Li, C., Huang, Z., Paudel, D.P., Wang, Y., Shahbazi, M., Hong, X., Van Gool, L.: A continual deepfake detection benchmark: Dataset, methods, and essentials. In: Winter Conference on Applications of Computer Vision (WACV) (2023)

25. Li, L., Zhou, D.W., Ye, H.J., Zhan, D.C.: Addressing imbalanced domainincremental learning through dual-balance collaborative experts. ICML (2025)

26. Li, L., Bao, J., Yang, H., Chen, D., Wen, F.: Faceshifter: Towards high fidelity and occlusion aware face swapping. arXiv preprint arXiv:1912.13457 (2019)

27. Li, Y., Yang, X., Sun, P., Qi, H., Lyu, S.: Celeb-df: A large-scale challenging dataset for deepfake forensics. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3207–3216 (2020)

28. Liang, S., Zhu, K., Zhai, W., Liu, Z., Cao, Y.: Hypercorrelation evolution for video class-incremental learning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 3315–3323 (2024)

29. Liu, Z., Li, Y., Gao, H., Li, Y., Kong, L., Sun, L., Huang, W.: IDER: IDempotent experience replay for reliable continual learning. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum? id=Vr5f3kRvLD

30. Luo, Y., Zhang, Y., Yan, J., Liu, W.: Generalizing face forgery detection with highfrequency features. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16317–16326 (June 2021)

31. Ma, L., Xue, Z., Wang, Y., Yan, Z., Xu, J., Jiang, X., Yu, H., Liao, Y., Bi, Z.: Your one-stop solution for ai-generated video detection. arXiv preprint arXiv:2601.11035 (2026)

32. Ma, X., Tian, J., Cai, Y., Chai, Y., Li, Z., Dai, J., Zang, L., Han, J.: Hidd: Humanperception-centric incremental deepfake detection. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2024)

33. Masi, I., Killekar, A., Mascarenhas, R.M., Gurudatt, S.P., AbdAlmageed, W.: Twobranch recurrent network for isolating deepfakes in videos. In: Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16. pp. 667–684. Springer (2020)

34. Nguyen, D., Astrid, M., Kacem, A., Ghorbel, E., Aouada, D.: Vulnerability-aware spatio-temporal learning for generalizable deepfake video detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10786–10796 (2025)

35. OpenAI: Sora. https://openai.com/index/sora/ (2024), [Accessed 28-06-2026]

36. Pan, K., Yin, Y., Wei, Y., Lin, F., Ba, Z., Liu, Z., Wang, Z., Cavallaro, L., Ren, K.: Dfil: Deepfake incremental learning by exploiting domain-invariant forgery clues. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 8035–8046 (2023)

37. Pham, C., Nguyen, V.A., Le, T., Phung, D., Carneiro, G., Do, T.T.: Frequency attention for knowledge distillation. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 2277–2286 (2024)

38. Prathibha, P., Tamizharasan, P., Panthakkan, A., Mansoor, W., Al Ahmad, H.: Sarb-df: A continual learning aided framework for deepfake video detection using self-attention residual block. IEEE Access 12, 189088–189101 (2024)

39. Rebufi, S.A., Kolesnikov, A., Sperl, G., Lampert, C.H.: icarl: Incremental classifier and representation learning. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 2001–2010 (2017)

40. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G.: Experience replay for continual learning. Advances in neural information processing systems 32 (2019)

41. Rosberg, F., Aksoy, E.E., Alonso-Fernandez, F., Englund, C.: Facedancer: Poseand occlusion-aware high fidelity face swapping. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 3454–3463 (January 2023)

42. Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., Nießner, M.: Faceforensics++: Learning to detect manipulated facial images. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1–11 (2019)

43. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV) 115(3), 211–252 (2015). https://doi.org/10.1007/s11263-015-0816-y

44. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: 2017 IEEE International Conference on Computer Vision (ICCV). pp. 618–626 (2017). https://doi.org/10.1109/ICCV.2017.74

45. Shiohara, K., Yamasaki, T.: Detecting deepfakes with self-blended images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18720–18729 (2022)

46. Shiohara, K., Yang, X., Taketomi, T.: Blendface: Re-designing identity encoders for face-swapping. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2023)

47. Sun, K., Chen, S., Yao, T., Sun, X., Ding, S., Ji, R.: Continual face forgery detection via historical distribution preserving. International Journal of Computer Vision 133(3), 1067–1084 (2025)

48. Tian, J., Yu, C., Wang, X., Chen, P., Xiao, Z., Han, J., Chai, Y.: Dynamic mixedprototype model for incremental deepfake detection. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 8129–8138 (2024)

49. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., Paluri, M.: A closer look at spatiotemporal convolutions for action recognition. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 6450–6459 (2018)

50. Villa, A., Alhamoud, K., Escorcia, V., Caba, F., Alcázar, J.L., Ghanem, B.: vclimb: A novel video class incremental learning benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19035– 19044 (2022)

51. Wang, S.Y., Wang, O., Zhang, R., Owens, A., Efros, A.A.: Cnn-generated images are surprisingly easy to spot... for now. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8695–8704 (2020)

52. Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang, Y., Yang, C., He, Y., Yu, J., Yang, P., et al.: Lavie: High-quality video generation with cascaded latent difusion models. International Journal of Computer Vision 133(5), 3059–3078 (2025)

53. Wang, Z., Bao, J., Zhou, W., Wang, W., Li, H.: Altfreezing for more general video face forgery detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4129–4138 (2023)

54. Xu, J., Zou, X., Huang, K., Chen, Y., Liu, B., Cheng, M., Shi, X., Huang, J.: Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991 (2024)

55. Xu, Y., Liang, J., Jia, G., Yang, Z., Zhang, Y., He, R.: Tall: Thumbnail layout for deepfake video detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22658–22668 (2023)

56. Xu, Z., Hong, Z., Ding, C., Zhu, Z., Han, J., Liu, J., Ding, E.: Mobilefaceswap: A lightweight framework for video face swapping. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 2973–2981 (2022)

57. Yan, Z., Yao, T., Chen, S., Zhao, Y., Fu, X., Zhu, J., Luo, D., Wang, C., Ding, S., Wu, Y., Yuan, L.: DF40: toward next-generation deepfake detection. In: Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J.M., Zhang, C. (eds.) Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 (2024), http://papers. nips.cc/paper\_files/paper/2024/hash/34239f60eca7ce9bee5280aaf81362d8- Abstract-Datasets\_and\_Benchmarks\_Track.html

58. Yan, Z., Zhang, Y., Fan, Y., Wu, B.: Ucf: Uncovering common features for generalizable deepfake detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 22412–22423 (2023)

59. Yan, Z., Zhao, Y., Chen, S., Guo, M., Fu, X., Yao, T., Ding, S., Wu, Y., Yuan, L.: Generalizing deepfake video detection with plug-and-play: Video-level blending and spatiotemporal adapter tuning. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12615–12625 (2025)

60. Yu, F., Pan, H., Zhang, K., Guan, J., Jiang, H.: Uhkd: A unified framework for heterogeneous knowledge distillation via frequency-domain representations. arXiv preprint arXiv:2510.24116 (2025)

61. Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., Wang, F.: Sadtalker: Learning realistic 3d motion coeficients for stylized audio-driven single image talking face animation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8652–8661 (2023)

62. Zhang, X., Zhu, P., Zhang, C., Yan, Z., Cheng, J., Lao, M., Cai, S., Guo, Y.: Generalization-preserved learning: Closing the backdoor to catastrophic forgetting in continual deepfake detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 3798–3808 (October 2025)

63. Zhang, Y., Liu, Y., Xia, B., Peng, B., Yan, Z., Lo, E., Jia, J.: Magicmirror: Idpreserved video generation in video difusion transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14464–14474 (2025)

64. Zheng, Y., Bao, J., Chen, D., Zeng, M., Wen, F.: Exploring temporal coherence for more general video face forgery detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15044–15054 (October 2021)

65. Zhou, T., Wang, W., Liang, Z., Shen, J.: Face forensics in the wild. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5778–5788 (June 2021)