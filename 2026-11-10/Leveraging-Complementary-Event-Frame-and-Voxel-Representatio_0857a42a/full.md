# Leveraging Complementary Event Frame and Voxel Representations for RGB-Guided Multimodal Pre-training

Wentao Wu, Xiao Wang\*, Member, IEEE, Yi Wang, Chenglong Li\*, Senior Member, IEEE, Yonghong Tian, Fellow, IEEE, Bin Luo

Abstract—Existing Event-based multimodal pre-training methods primarily achieve joint multimodal learning by mapping Event features into the feature spaces of pre-trained models. However, these methods mainly focus on feature associations within existing representation spaces, lacking unified modeling of continuous RGB-Event observations and sufficient learning of cross-modal dependencies, thereby limiting the multimodal representational capacity of pre-trained models. To address these issues, we propose CM3AE 2.0, a unified RGB Frame and Event-Voxel/-Frame pre-training framework. By jointly learning multimodal spatiotemporal representations and exploiting latent cross-modal dependencies through cross-modal generation, the framework enhances multimodal information interaction and collaborative modeling capabilities. Specifically, we first propose an Event-aware Spatiotemporal Masked Reconstruction (ESMR) strategy to learn unified multimodal spatiotemporal representations. It leverages event responses to guide masking toward dynamically salient regions. Through spatiotemporal masked reconstruction, it learns spatial structures and their temporal variations from continuous RGB-Event observations. In addition, it incorporates multimodal fusion reconstruction and cross-modal contrastive learning to facilitate the modeling of cross-modal complementary information and the learning of shared representations. Building upon this, we further propose a Hierarchical Cross-modal Generation (HCG) strategy to learn latent cross-modal dependencies through visual and language generation. At the visual level, Event-conditioned RGB generation is performed using a pre-trained diffusion model to capture visual associations between the two modalities. At the language level, video-level text description generation based on joint RGB-Event representations establishes connections between multimodal visual features and linguistic semantics, further enriching the learning of cross-modal dependencies. Extensive experiments on four diverse downstream tasks demonstrate that CM3AE 2.0 consistently improves performance, validating the effectiveness and generalization capability of the proposed pre-training framework. The source code of this paper is available at https://github.com/Event-AHU/CM3AE.

Index Terms—Pre-trained Vision Model; Masked Auto-Encoder; Multimodal Learning; Vision and Language; Self-Supervised Learning

✦

## 1 INTRODUCTION

consumption, have become an important sensing modality for visual perception [1]. Unlike conventional frame-based cameras, event cameras asynchronously capture pixel-level brightness changes, providing fine-grained motion cues and strong robustness to rapid motion and challenging illumination conditions. These properties have enabled a wide range of applications, including action recognition [2], object detection [3], visual object tracking [4], semantic understanding [5], and image reconstruction [6]. However, most event-based perception models are still developed for individual tasks and rely heavily on task-specific annotations, motivating the development of transferable representations through large-scale pre-training.

Large-scale self-supervised pre-training has demonstrated strong transferability across conventional vision tasks [7], [8]. Recent vision foundation models, including MAE [9], iBOT [10], DINOv2 [11], and BEiT v2 [12], have shown the effectiveness of learning generalizable visual representations from large-scale unlabeled data. Extending to sequential visual understanding, VideoMAE [13] and Video-MAE V2 [14] further explore temporal representation learning. Meanwhile, multimodal pre-training methods, such as CLIP [15] and MultiMAE [16], exploit cross-modal information to enhance representation learning. Inspired by these advances, increasing efforts have been devoted to eventbased pre-training. Early studies primarily focus on Event representation learning through masked reconstruction [17] and contrastive learning [18]. Subsequent studies further explore spatiotemporal representation learning. Huang et al. [19] introduce disentangled masked modeling to learn local spatiotemporal features and global representations, while TESPEC [20] investigates long-term temporal modeling through pseudo-grayscale video reconstruction. Recently, researchers have increasingly leveraged pre-trained foundation models for cross-modal learning. EventBind [21] aligns Event representations with the shared image-text space of CLIP, while Jeong et al. [22] and ScaleEvent [23] transfer visual knowledge through cross-modal alignment and distillation. Furthermore, EventGPT [24] incorporates

![](images/9be9fca8b21492089b8e80744e4b7ec759aed196762e55ad71e82abe4823ad48.jpg)  
Fig. 1. Comparison of existing pre-training paradigms and CM3AE 2.0. (a) Representative Event-related pre-training paradigms. (b) Overview of the proposed CM3AE 2.0 framework. (c) Performance comparisons on action recognition and event-guided low-light image enhancement.

Event representations into multimodal large language models for semantic understanding.

Despite these advances, three key challenges remain in unified RGB-Event pre-training, as illustrated in Fig. 1. First, although existing studies have explored Event spatiotemporal learning [19], [20] and cross-modal representation alignment [21], unified modeling of temporal dynamics across continuous RGB-Event observations remains insufficient. Existing Event-based methods have explored temporal modeling within Event streams, while cross-modal alignment methods primarily focus on establishing correspondences between Event representations and pre-trained visual or vision-language features. Consequently, the temporal evolution of RGB appearance and Event activity is not fully exploited within a unified multimodal pre-training framework. Second, existing approaches such as Event-Bind [21] and ScaleEvent [23] primarily transfer knowledge from pre-trained models through feature alignment and distillation, with limited explicit integration of complementary information from paired RGB and Event observations. Consequently, complementary cues from RGB appearance and Event motion may not be sufficiently combined, while shared features across modalities may not be adequately learned. Third, although existing studies have begun to investigate semantic understanding of Event data [24], the latent dependencies among Event dynamics, RGB visual content, and scene semantics remain insufficiently modeled, limiting the ability to exploit information that can be inferred across modalities.

To address these limitations, we propose CM3AE 2.0, a unified RGB Frame and Event-Voxel/-Frame pre-training framework. The framework learns unified multimodal spatiotemporal representations and further explores latent cross-modal dependencies through cross-modal generation, enhancing multimodal information interaction and collaborative modeling. Specifically, to address the insufficient modeling of temporal dynamics in continuous RGB-Event observations and the limited exploitation of complementary information, we introduce an Event-aware Spatiotemporal Masked Reconstruction (ESMR) strategy. It leverages event responses to guide masking toward dynamically salient regions, and employs modality-specific spatiotemporal masked reconstruction to learn spatial structures and their temporal variations from continuous RGB-Event observations. To further preserve Event-specific information, the event-specific reconstruction loss supervises event occurrence and polarity prediction. Meanwhile, multimodal fusion reconstruction integrates RGB, Event, and Voxel information to exploit complementary cues, while crossmodal contrastive learning promotes shared representation learning across modalities. To further explore latent dependencies among Event dynamics, RGB visual content, and scene semantics, we introduce a Hierarchical Crossmodal Generation (HCG) strategy. At the visual level, eventconditioned RGB image generation uses Event representations to condition a pre-trained diffusion model, transferring visual generative priors to Event representations and encouraging them to capture RGB appearance and structural information. At the language level, video-level text generation leverages joint RGB-Event representations to guide a pre-trained language model in generating scene descriptions, establishing connections between multimodal visual features and linguistic semantics.

To comprehensively evaluate the transferability and generalization capability of the proposed framework, we conduct extensive experiments under both Event-only and RGB-Event multimodal settings, covering four representative downstream tasks, including action recognition, object detection, visual object tracking, and event-guided lowlight image enhancement. Experimental results demonstrate that CM3AE 2.0 provides effective pre-trained initialization for diverse downstream architectures and consistently improves the representation capability of both Event and RGB-Event models.

To sum up, the contributions of this work can be summarized as follows:

• We propose CM3AE 2.0, a unified RGB Frame and Event-Voxel/-Frame pre-training framework that learns unified multimodal spatiotemporal representations and further explores latent cross-modal dependencies through cross-modal generation.

• We propose an Event-aware Spatiotemporal Masked Reconstruction strategy that learns unified multimodal spatiotemporal representations by modeling temporal dynamics in continuous RGB-Event observations and exploiting complementary information and shared features through fusion reconstruction and cross-modal contrastive learning.

• We propose a Hierarchical Cross-modal Generation strategy that combines event-conditioned RGB image generation and video-level text generation to explore latent cross-modal dependencies by leveraging visual generative priors and linguistic knowledge.

• We conduct extensive experiments on four downstream tasks, demonstrating the effectiveness of CM3AE 2.0 across diverse perception tasks, as well as the strong transferability and generalization capability of the learned representations.

A preliminary version of this work appeared in the proceedings of ACM Multimedia 2025 (ACM MM 2025) under the title CM3AE [25]. Compared with the conference version, this journal extension substantially advances the original work in three aspects. (1) We extend the original framelevel multimodal pre-training framework to support unified multimodal spatiotemporal representation learning. Specifically, we introduce an Event-aware spatiotemporal masked reconstruction strategy. Building upon the original framework, this strategy leverages Event information to guide masking toward dynamically salient regions, employs spatiotemporal masked reconstruction to model spatiotemporal relationships across continuous RGB-Event observations, and incorporates an event-specific reconstruction loss to learn Event-specific structural information, thereby enabling unified multimodal spatiotemporal representation learning. (2) We introduce a new hierarchical cross-modal generation strategy, which combines Event-conditioned RGB image generation using a pre-trained diffusion model with videolevel text generation using a pre-trained language model. By leveraging visual generative priors and linguistic knowledge, this strategy explores latent cross-modal dependencies among Event dynamics, RGB appearance, and high-level scene semantics. (3) We extend the evaluation scope by introducing an event-guided low-light image enhancement task and conducting experiments across diverse datasets under both RGB-Event and Event-only settings. Furthermore, comprehensive ablation studies, transferability analyses, computational efficiency evaluations, and qualitative visualizations validate the effectiveness and generalization capability of the proposed framework. Overall, CM3AE 2.0 goes beyond the original frame-level cross-modal learning framework by establishing a two-stage pre-training paradigm that integrates unified multimodal spatiotemporal representation learning with generative supervision for learning latent cross-modal dependencies.

## 2 RELATED WORK

## 2.1 Event-based Vision

Research on event-based visual perception primarily focuses on modeling the spatial structure and temporal dynamics of event streams, with applications in recognition, detection, and tracking. For recognition, EventMamba [26] learns event representations through hierarchical pointbased processing and state-space modeling, while EMP [27] combines accumulated and reconstructed event frames for action recognition. Wang et al. [2], [28] introduce the HARDVS and CeleX-HAR benchmarks and propose ESTF and EVMamba, respectively, to learn spatiotemporal features. For object detection, RVTs [29] and SATE [30] combine spatial feature extraction with recurrent temporal modeling, whereas MvHeat-DET [31] explores event feature extraction using mixture-of-experts heat-conduction operators. For tracking and segmentation, Wang et al. [32] introduce the EventVOT benchmark and HDETrack for event-only tracking, while Zhu et al. [33] predict subsequent object masks using recurrent temporal embeddings and historical eventmask references. Event streams have also been used to assist conventional image enhancement, improving image quality under challenging illumination conditions. EvLight [34] performs low-light image enhancement through signal-tonoise ratio-guided regional feature selection, while Weng et al. [35] address local overexposure and underexposure in high-dynamic-range scenes. More recently, RetinEV [36] estimates illumination from event timestamps to improve low-light image enhancement. These studies demonstrate the potential of event-based vision across diverse perception tasks and highlight the importance of effective spatial and temporal feature modeling for improving event-based visual perception.

## 2.2 RGB-Event based Fusion

RGB frames provide rich appearance and semantic information but are vulnerable to low illumination and motion blur, whereas event streams capture rapid motion with high temporal resolution but offer limited appearance information. Their complementarity motivates RGB-Event fusion, although modality discrepancies and spatiotemporal misalignment remain challenging. For object tracking, MAMTrack [37] combines Mamba-based fusion with historical information to model target appearance changes. AlignTrack [38] leverages target-related priors to guide cross-modal feature alignment, while CRSOT [39] employs uncertainty-aware fusion to address spatiotemporal misalignment between sensors. For object detection, FlexEvent [40] adaptively fuses RGB semantics with highfrequency event observations to support detection at varying operational frequencies. Beyond Duality [41] explicitly disentangles shared and modality-specific features, enabling adaptive exploitation of RGB appearance details and eventbased motion cues under different scene conditions. For recognition, MMHCO-HAR [42] combines heat-conductionbased feature interaction with policy-driven adaptive fusion for human activity recognition. Wang et al. [43] introduce the EventPAR benchmark and an asymmetric RWKV fusion framework, extending RGB-Event fusion to pedestrian attribute recognition, including appearance and emotion attributes. For optical flow estimation, Diff-ABFlow [44] leverages appearance cues from RGB frames and boundary information from event streams to condition diffusion-based estimation. Overall, existing methods primarily address modality discrepancies within downstream tasks. However, the limited scale and diversity of downstream datasets constrain the generalizability of the learned representations.

## 2.3 Pre-trained Foundation Models

Early pre-training primarily focused on RGB images or videos. For example, MoCo [45] learns discriminative visual representations through contrastive learning, MAE [9] learns image representations via pixel reconstruction, and VideoMAE V2 [14] extends masked modeling to video pre-training. With the development of multimodal pretraining, CLIP [15] achieves cross-modal alignment through image-text contrastive learning, MultiMAE [16] learns crossmodal representations by jointly reconstructing RGB, depth, and semantic information, while VehicleMAE [46] leverages vehicle structural priors and image-text knowledge to guide vehicle appearance reconstruction. Recent studies further explore autoregressive modeling and generative priors. AIMv2 [47] learns visual representations by autoregressively predicting image patches and text tokens, while CleanDIFT [48] and VLV [49] enhance visual representations through diffusion feature distillation and generative model knowledge transfer.

Inspired by these studies, researchers have begun to explore pre-training methods for event data. Early works primarily focused on event representation learning. ECDP [18] constructs contrastive learning objectives using paired images, MEM [17] learns event representations through masked event modeling, and TESPEC [20] captures temporal information using long event sequences and pseudograyscale video reconstruction. Subsequent studies have expanded toward multimodal learning. Liang et al. [50] transfer pre-trained image knowledge to the event domain through adaptive prompting, ScaleEvent [23] employs structure-aware distillation from visual foundation models, and EventBind [21] aligns event representations with pretrained image-text feature spaces. CM3AE [25] explores joint RGB-Event representation learning through multimodal reconstruction and cross-modal contrastive learning. Furthermore, EventGPT [24] and EventDrive [51] investigate the use of language knowledge for event-stream understanding and the integration of RGB, events, and language for driving perception and reasoning, respectively. However, existing studies primarily focus on pre-trained knowledge transfer and cross-modal semantic associations, leaving the unified spatiotemporal modeling of continuous RGB-Event observations insufficiently explored.

## 3 METHODOLOGY

## 3.1 Overview

As illustrated in Fig. 2, we propose CM3AE 2.0, a unified RGB Frame and Event-Voxel/-Frame pre-training framework. The framework consists of two stages, Event-aware Spatiotemporal Masked Reconstruction (ESMR) and Hierarchical Cross-modal Generation (HCG), aiming to learn unified multimodal spatiotemporal representations and explore latent cross-modal dependencies. In the first stage, to effectively model spatiotemporal relationships and exploit cross-modal complementary information from continuous RGB-Event observations, we introduce ESMR. Specifically, event-guided masking identifies dynamically salient regions based on Event activity, while spatiotemporal masked reconstruction captures spatiotemporal relationships through spatial encoding and cross-frame interactions. An eventspecific reconstruction loss further enhances the representation of Event-specific information. Moreover, multimodal fusion reconstruction exploits complementary information by integrating RGB, Event, and Voxel features, while crossmodal contrastive learning promotes shared representation learning through feature alignment. In the second stage, to further explore cross-modal associations and highlevel semantic knowledge that are insufficiently captured by feature-level learning, we introduce HCG. Specifically, event-conditioned RGB image generation leverages visual generative priors from a pre-trained diffusion model to establish associations between Event dynamics and RGB appearance. Meanwhile, video-level text generation generates video descriptions from joint RGB-Event representations, establishing connections between multimodal visual features and high-level linguistic semantics.

## 3.2 CM3AE 2.0 Pre-training Framework

The first stage consists of four components, i.e., eventguided masking, spatiotemporal masked reconstruction, multimodal fusion reconstruction, and cross-modal contrastive learning.

• Stage 1: Event-aware Spatiotemporal Masked Reconstruction. Existing RGB-Event pre-training methods lack unified modeling of spatiotemporal relationships across continuous RGB-Event observations and fail to fully exploit complementary information between modalities. To address these limitations, we develop an Event-aware Spatiotemporal Masked Reconstruction (ESMR) strategy, which consists of four components: event-guided masking, spatiotemporal masked reconstruction, multimodal fusion reconstruction, and cross-modal contrastive learning. The overall objective of Stage 1 is to learn unified multimodal spatiotemporal representations. Specifically, event-guided masking and spatiotemporal masked reconstruction model temporal dynamics and cross-frame relationships in continuous RGB-Event observations. Meanwhile, multimodal fusion reconstruction exploits complementary information across modalities, while cross-modal contrastive learning promotes shared representation learning.

Event-guided Masking. Given a paired RGB-Event sequence of T frames, we denote the RGB and Event inputs as $\begin{array} { r } { \mathcal { T } ^ { r } ~ = ~ \{ I _ { t } ^ { r } \} _ { t = 1 } ^ { T } } \end{array}$ and $\mathcal { T } ^ { e } ~ = ~ \{ I _ { t } ^ { e } \} _ { t = 1 } ^ { T } ,$ , respectively. Each frame is divided into $N \mathrm { ~ = ~ } 1 9 6$ non-overlapping patches and projected into D = 768-dimensional patch embeddings using modality-specific patch embedding layers.

Unlike conventional random masking that assigns equal masking probabilities to all spatial patches, our strategy exploits Event activity to identify dynamically informative regions. Specifically, positive and negative events are first converted into a binary Event activity map, where pixels with Event activity are assigned a value of one and inactive pixels zero. Based on this map, the frame-wise Event response $s _ { t , i }$ is defined as the proportion of active Event pixels within the i-th patch of the t-th frame, quantifying local Event activity at each spatial position.

![](images/9599df0d2e3c035ffca53df5b67ba927dbe5a79b4857de28252f2d684b000e8c.jpg)  
Fig. 2. Overview of the proposed CM3AE 2.0 framework, comprising two stages: (1) Event-aware Spatiotemporal Masked Reconstruction (ESMR), which learns unified multimodal spatiotemporal representations by modeling cross-frame dynamics and exploiting complementary information and shared features across modalities; and (2) Hierarchical Cross-modal Generation (HCG), which explores latent cross-modal dependencies through Event-conditioned RGB Image Generation and Video-level Text Generation.

For genuine video sequences, we aggregate frame-wise responses across valid frames to capture both persistent motion and short-term salient variations:

$$
s _ { i } = \alpha \ \mathrm { M e a n } _ { t \in \mathcal { V } } ( s _ { t , i } ) + ( 1 - \alpha ) \ \mathrm { M a x } _ { t \in \mathcal { V } } ( s _ { t , i } ) ,\tag{1}
$$

where V denotes the set of valid frames and $\alpha = 0 . 7$ . The resulting response $s _ { i }$ is normalized across spatial positions to obtain $\bar { s } _ { i } ,$ reflecting the relative Event activity at each position.

Based on these responses, we perform event-guided probabilistic masking. The sampling weight of the i-th patch is defined as:

$$
w _ { i } = ( 1 - \gamma ) + \gamma \bar { s } _ { i } + 0 . 0 1 \eta \xi _ { i } ,\tag{2}
$$

where $\gamma$ controls the strength of Event guidance, $\xi _ { i } \sim$ $\mathcal { U } ( 0 , 1 )$ introduces a small stochastic perturbation, and η controls its magnitude. We set $\gamma = 0 . 6$ and $\eta = 0 . 1$ . Patches with stronger Event responses are assigned larger sampling weights and are therefore more likely to be selected as masked reconstruction targets. The uniform baseline term prevents the sampling weights from being determined solely by Event activity, while the stochastic perturbation introduces small random variations. The default masking ratio is set to 75%, leaving the remaining 25% of patches visible.

To preserve spatial correspondence, paired RGB and Event inputs share the same masking pattern. For genuine video sequences, temporally aggregated Event responses guide a shared spatial mask across frames, whereas each image in an image-level pseudo-sequence is independently masked based on its own Event response.

Spatiotemporal Masked Reconstruction. After eventguided masking, the visible patch embeddings of RGB and Event are obtained as $\mathbf { E } _ { t } ^ { r } , \dot { \mathbf { E } _ { t } ^ { e } } \in \mathbb { R } ^ { 4 9 \times 7 6 8 }$ , respectively. We add spatial and temporal positional embeddings to these features and prepend a CLS token, yielding the encoder inputs Ee <sup>r</sup><sub>t</sub> , $\widetilde { \mathbf { E } } _ { t } ^ { e } \in \mathbb { R } ^ { 5 0 \times 7 6 8 }$ . They are then fed into two modalityspecific ViT encoders, each consisting of 12 Transformer blocks. Temporal attention is inserted after every four blocks to model spatiotemporal relationships across consecutive frames. Specifically, for modality $m \in \{ r , e \}$ , the features at the same spatial position across $T$ consecutive frames are organized into $\dot { \bf Q } _ { i } ^ { m } \in \mathbb { R } ^ { T \times 7 6 8 }$ . Cross-frame temporal interaction is formulated as

$$
\widetilde { \mathbf { Q } } _ { i } ^ { m } = \mathbf { Q } _ { i } ^ { m } + \mathrm { T A t t n } \left( \mathrm { L N } ( \mathbf { Q } _ { i } ^ { m } ) \right) ,\tag{3}
$$

where TAttn denotes temporal self-attention. After spatial encoding and cross-frame temporal interaction, the RGB and Event encoder outputs are obtained as $\mathbf { F } _ { t } ^ { r }$ and $\mathbf { F } _ { t } ^ { e } .$ , respectively. For image-level pseudo-sequences without genuine temporal continuity, cross-frame temporal attention is disabled.

We further introduce a Voxel branch. Given a Voxel sequence $\mathbf { \nabla } \mathcal { X } ^ { v } ~ = ~ \{ \mathbf { V } _ { t } \} _ { t = 1 } ^ { T } ,$ , the raw Voxel features of each frame are divided into 196 patches and projected into a 768- dimensional feature space, yielding the patch embeddings $\mathbf { E } _ { t } ^ { v } \in \mathbb { R } ^ { 1 9 6 \times 7 6 8 }$ . Unlike the RGB and Event branches, the Voxel branch is not spatially masked, and all patches are retained. After adding a CLS token and spatial and temporal positional embeddings, the resulting features $\widetilde { \mathbf { E } } _ { t } ^ { v } \in \mathbb { R } ^ { 1 9 \widetilde { 7 } \times 7 6 8 }$ are fed into a separate ViT encoder. Spatial encoding and temporal interaction produce the complete Voxel representation $\mathbf { F } _ { t } ^ { v }$ . Cross-frame temporal interaction is applied only to valid video frames with Voxel observations. The RGB and Event encoder outputs, $\mathbf { F } _ { t } ^ { r }$ and $\mathbf { F } _ { t } ^ { e } ,$ , are then projected from 768 to 512 dimensions through separate linear projection layers. We insert 147 learnable mask tokens at the original masked positions, yielding the decoder inputs $\widehat { \bf E } _ { t } ^ { r } , \widehat { \bf E } _ { t } ^ { e } \in$ $\mathbb { R } ^ { 1 9 7 \times 5 1 2 }$ . These inputs are fed into the corresponding RGB and Event decoders, each consisting of eight Transformer blocks. The decoders produce the feature representations ${ \bf { H } } _ { t } ^ { r }$ and $\mathbf { H } _ { t } ^ { e } ,$ from which linear prediction layers generate the reconstructed RGB and Event contents, denoted by $\hat { \mathbf { X } } _ { t } ^ { r }$ and $\hat { \bf X } _ { t } ^ { e } ,$ , respectively. Reconstruction losses are computed only over masked patches.

For the Event branch, we further introduce an eventspecific reconstruction loss. In addition to pixel-level Event reconstruction, two lightweight prediction heads are applied to predict event occurrence and dominant polarity in masked patches. The occurrence head predicts whether a masked patch contains valid Events, whereas the polarity head predicts the dominant polarity only for patches containing valid Events. These objectives enable the model to capture Event-specific structural information beyond pixellevel reconstruction.

Multimodal Fusion Reconstruction. To further exploit complementary information across modalities, we perform multimodal fusion reconstruction using the RGB, Event, and Voxel encoder features, $\mathbf { F } _ { t } ^ { r } , \mathbf { F } _ { t } ^ { e } ,$ , and $\mathbf { F } _ { t } ^ { v } .$ . Specifically, the visible RGB and Event features are concatenated and fed into a Transformer-based fusion module, where multimodal self-attention is used to obtain the RGB-Event fused representation $\mathbf { F } _ { t } ^ { r e }$ . For RGB-Event-Voxel fusion, the same fusion process is adopted by combining the visible RGB and Event features with the complete Voxel patch features, yielding the joint representation $\bar { \mathbf { F } _ { t } ^ { r e v } }$ . The two fused representations are then separately projected to 512 dimensions. By inserting 147 learnable mask tokens at the original masked positions, we obtain the decoder inputs $\widehat { \mathbf { E } } _ { t } ^ { r e } , \widehat { \mathbf { E } } _ { t } ^ { r e v } \in \mathbb { R } ^ { 1 9 7 \times 5 1 2 }$ Both branches use the same RGB decoder to generate the reconstructed RGB contents $\hat { \mathbf { X } } _ { t } ^ { r e }$ and $\hat { \mathbf { X } } _ { t } ^ { r e v }$ , respectively. The reconstruction losses are computed over masked RGB patches to encourage the model to exploit complementary information across modalities.

Cross-modal Contrastive Learning. In addition to reconstruction, we employ cross-modal contrastive learning to promote shared representation learning across modalities. Specifically, the RGB and Event decoder features, ${ \bf { H } } _ { t } ^ { r }$ and $\mathbf { H } _ { t } ^ { e }$ , are aggregated through learnable token pooling to obtain the frame-level representations $\mathbf { z } _ { t } ^ { r } , \mathbf { z } _ { t } ^ { e } \in \mathbb { \hat { \mathbb { R } } } ^ { 5 1 2 }$ . After $L _ { 2 }$ normalization, the similarity between RGB and Event representations is computed to construct a bidirectional contrastive objective. Corresponding RGB-Event observations are treated as positive pairs, while representations from other observations serve as negatives. To reduce falsenegative interference, other frames from the same genuine video are excluded from the negative set, whereas different images in image-level pseudo-sequences are treated as independent observations. Furthermore, frames with extremely low Event activity are filtered during RGB-Event contrastive learning.

Similarly, we construct an RGB-Voxel contrastive objective. The complete Voxel encoder features $\mathbf { F } _ { t } ^ { v }$ are aggregated through learnable token pooling and projected from 768 to 512 dimensions, yielding the Voxel representation $\mathbf { z } _ { t } ^ { v } \in \mathbb { R } ^ { 5 1 2 }$ . The same contrastive learning procedure is then applied to RGB-Voxel pairs, using only frames with valid Voxel observations. Through these components, Stage 1 learns unified multimodal spatiotemporal representations from continuous RGB-Event observations while exploiting complementary information and shared features across modalities.

• Stage 2: Hierarchical Cross-modal Generation. Stage 1 learns unified multimodal spatiotemporal representations through reconstruction and contrastive learning. To further explore cross-modal dependencies at the visual and semantic levels, we introduce a Hierarchical Cross-modal Generation (HCG) strategy consisting of event-conditioned RGB image generation and video-level text generation. The former transfers visual generative priors from a pre-trained diffusion model to Event representations, while the latter introduces linguistic knowledge through video-level description generation.

Event-conditioned RGB Image Generation. To establish the relationship between Event representations and corresponding RGB appearance, we employ the pre-trained Stable Diffusion v1.4 model [52], conditioned on the Event decoder features $\mathbf { H } _ { t } ^ { e } .$ . After removing the CLS token, the remaining Event patch features are projected from 512 dimensions into the diffusion conditioning space through a learnable linear projection layer, yielding $\mathbf { \Delta C } _ { t } ^ { e } \in \mathbb { R } ^ { 1 9 6 \times 7 6 \breve { 8 } }$

For the corresponding RGB image $I _ { t } ^ { r } , \mathrm { ~ a ~ }$ frozen VAE encoder extracts its latent representation $\mathbf { z } _ { t } .$ . Gaussian noise $\epsilon _ { t } \sim \mathcal { N } ( 0 , \bf { I } )$ is added at a randomly sampled timestep τ to obtain the noisy latent $\mathbf { z } _ { t , \tau }$ . The frozen diffusion U-Net then predicts the injected noise conditioned on $\mathbf { C } _ { t } ^ { e }$

$$
\begin{array} { r } { \mathbf { z } _ { t , \tau } = \sqrt { \bar { \alpha } _ { \tau } } \mathbf { z } _ { t } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \mathbf { \epsilon } _ { t } , } \\ { \hat { \mathbf { \epsilon } } _ { t } = \epsilon _ { \theta } \left( \mathbf { z } _ { t , \tau } , \tau , \mathbf { C } _ { t } ^ { e } \right) . \qquad } \end{array}\tag{4}
$$

During training, the VAE and diffusion U-Net remain frozen, while the noise-prediction objective updates the conditioning projection and upstream Event representations. Through this process, the Event representations learn dependencies between Event dynamics and RGB appearance under the guidance of pre-trained visual generative priors.

Video-level Text Generation. To introduce high-level linguistic semantics into multimodal representation learning, we employ the RGB-Event fusion decoder features $\mathbf { H } _ { t } ^ { r e }$ and Event decoder features $\mathbf { H } _ { t } ^ { e }$ obtained in Stage 1. After removing the CLS tokens, the patch features from valid frames are concatenated and aggregated into eight visual tokens through adaptive token pooling. A learnable projection layer further maps the aggregated features from

512 dimensions into the embedding space of a pre-trained language model, yielding $\mathbf { V } \in \mathbb { R } ^ { 8 \times 4 0 9 6 }$ :

$$
\mathbf { V } = \psi _ { l } \left( \mathrm { P o o l } \left( \mathrm { C a t } _ { t } \left[ \mathbf { H } _ { t , 1 : N } ^ { r e } ; \mathbf { H } _ { t , 1 : N } ^ { e } \right] \right) \right) ,\tag{5}
$$

where Pool(·) denotes visual token aggregation and $\psi _ { l } ( \cdot )$ represents the visual-to-language projection. The resulting visual tokens are combined with a textual prompt and fed into a pre-trained Llama-3-8B model to generate video-level descriptions.

Since the pre-training datasets lack textual annotations, we use Qwen2.5-VL [53] to generate video-level descriptions offline as pseudo-text labels. During training, the pretrained Llama-3-8B model [54] remains frozen, and the autoregressive language modeling loss is computed only over the target description tokens. The gradients are propagated through the visual-to-language projection to the upstream multimodal representations, enabling them to learn scenelevel semantic relationships from linguistic supervision.

Through these two generation tasks, HCG models crossmodal dependencies at both the visual appearance and linguistic semantic levels.

## 3.3 Training Objectives

To learn unified multimodal spatiotemporal representations and further explore latent cross-modal dependencies, we formulate a two-stage training objective. Stage 1 combines masked reconstruction and event-specific reconstruction loss to model temporal dynamics and strengthen Event-specific representations, while multimodal fusion reconstruction and cross-modal contrastive learning exploit complementary information and promote shared representation learning. Stage 2 retains the reconstruction objectives and introduces visual and language generation to further explore latent cross-modal dependencies.

For masked reconstruction, let $\mathcal { M } _ { t }$ denote the set of masked patches in the t-th frame, and let $\mathbf { X } _ { t , i } ^ { r }$ and $\mathbf { X } _ { t , i } ^ { e }$ denote the ground-truth RGB and Event patch contents, respectively. Based on the reconstruction predictions $\hat { \mathbf { X } } _ { t } ^ { r }$ and $\hat { \mathbf { X } } _ { t } ^ { e }$ defined above, the joint masked reconstruction loss is computed over the masked regions as:

$$
\begin{array} { r } { \mathcal { L } _ { m } = \displaystyle \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { t } \sum _ { i \in \mathcal { M } _ { t } } \Big ( \left\| \mathbf { X } _ { t , i } ^ { r } - \hat { \mathbf { X } } _ { t , i } ^ { r } \right\| _ { 2 } ^ { 2 } } \\ { + \left\| \mathbf { X } _ { t , i } ^ { e } - \hat { \mathbf { X } } _ { t , i } ^ { e } \right\| _ { 2 } ^ { 2 } \Big ) , } \end{array}\tag{6}
$$

where |M| denotes the total number of valid masked patches.

To explicitly preserve event-specific properties beyond pixel reconstruction, we introduce event occurrence and polarity prediction based on the Event decoder features. The event occurrence target $y _ { t , i } ^ { \mathrm { o c c } }$ indicates whether the ith patch in the t-th frame contains any positive or negative event pixels. The polarity target $y _ { t , i } ^ { \mathrm { p o l } }$ is determined by the dominant event polarity, with a label of 1 assigned when the number of positive event pixels is greater than or equal to that of negative event pixels, and 0 otherwise.

Let $\mathbf { \Psi } _ { a _ { t , i } }$ denote the predicted event occurrence logit and $\mathbf { b } _ { t , i }$ denote the logits for the two polarity classes. The event occurrence loss is computed using binary cross-entropy over masked patches in valid frames:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { o c c } } = \displaystyle - \frac { 1 } { N _ { \mathrm { o c c } } } \sum _ { t \in \mathcal { V } } \sum _ { i \in \mathcal { M } _ { t } } \Big [ y _ { t , i } ^ { \mathrm { o c c } } \log \sigma ( a _ { t , i } ) } \\ & { ~ \qquad + ( 1 - y _ { t , i } ^ { \mathrm { o c c } } ) \log \big ( 1 - \sigma ( a _ { t , i } ) \big ) \Big ] , } \end{array}\tag{7}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function, V denotes the set of valid frames, $\mathcal { M } _ { t }$ denotes the set of masked patches in the t-th frame, and $N _ { \mathrm { o c c } }$ denotes the total number of valid masked patches.

The polarity loss is computed using cross-entropy only for masked patches containing valid events:

$$
\mathcal { L } _ { \mathrm { p o l } } = - \frac { 1 } { N _ { \mathrm { p o l } } } \sum _ { t \in \mathcal { V } } \sum _ { i \in \mathcal { M } _ { t } } y _ { t , i } ^ { \mathrm { o c c } } \log \mathrm { S o f t m a x } ( \mathbf { b } _ { t , i } ) _ { y _ { t , i } ^ { \mathrm { p o l } } } ,\tag{8}
$$

where $N _ { \mathrm { p o l } }$ denotes the number of masked patches containing valid events. The resulting event-specific reconstruction loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e s p } } = \mathcal { L } _ { \mathrm { o c c } } + \mathcal { L } _ { \mathrm { p o l } } . } \end{array}\tag{9}
$$

For cross-modal complementary representation learning, the fused representation $\mathbf { F } _ { t } ^ { r e }$ is decoded to recover the corresponding RGB content $\hat { \mathbf { X } } _ { t } ^ { r e }$ . The fusion reconstruction loss is defined as:

$$
\mathcal { L } _ { f u s } ^ { r e } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { t } \sum _ { i \in \mathcal { M } _ { t } } \left. \mathbf { X } _ { t , i } ^ { r } - \hat { \mathbf { X } } _ { t , i } ^ { r e } \right. _ { 2 } ^ { 2 } .\tag{10}
$$

Similarly, the RGB-Event-Voxel reconstruction loss $\mathcal { L } _ { f u s } ^ { r e v }$ is computed only on frames with valid Voxel observations. The overall fusion reconstruction loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { f u s } = \mathcal { L } _ { f u s } ^ { r e } + \mathcal { L } _ { f u s } ^ { r e v } . } \end{array}\tag{11}
$$

The fusion reconstruction objective therefore encourages both RGB-Event and RGB-Event-Voxel representations to integrate dynamic information from Event and Voxel with the appearance cues provided by RGB observations.

Meanwhile, we employ symmetric contrastive learning to promote shared representation learning across different modalities. Given the $L _ { 2 }$ -normalized RGB and Event representations $\mathbf { z } _ { i } ^ { r }$ and $\mathbf { z } _ { i } ^ { e } ,$ , their pairwise similarity is computed as $s _ { i j } ^ { r e } = l _ { s } ( \mathbf { \bar { z } } _ { i } ^ { r } ) ^ { \top } \mathbf { z } _ { j } ^ { e } .$ , where $\bar { l } _ { s }$ denotes the learnable scaling factor.

The RGB-Event contrastive objective considers both RGB-to-Event and Event-to-RGB directions, treating corresponding observations as positive pairs and other valid observations as negatives. Similarly, RGB-Voxel contrastive learning is performed in both directions. Following the original CM3AE formulation, the overall contrastive objective is expressed as:

$$
\mathcal { L } _ { c o n } = \mathcal { L } _ { r  e } + \mathcal { L } _ { e  r } + \mathcal { L } _ { r  v } + \mathcal { L } _ { v  r } .\tag{12}
$$

To reduce false-negative interference in sequential observations, other frames from the same video are excluded from the negative set. RGB-Event contrastive learning further filters frames with extremely low Event activity, while RGB-Voxel contrastive learning is performed only on frames with valid Voxel observations.

For Event-conditioned RGB image generation, we employ the standard diffusion denoising objective to optimize the Event conditioning representations using a frozen pretrained diffusion model. Given the Event-conditioned representation ${ \bf C } _ { t } ^ { e }$ , the diffusion loss is formulated as:

$$
\mathcal { L } _ { d i f f } = \mathbb { E } _ { t , \tau , \epsilon } \left[ \left. \epsilon - \epsilon _ { \theta } \left( \mathbf { z } _ { t , \tau } , \tau , \mathbf { C } _ { t } ^ { e } \right) \right. _ { 2 } ^ { 2 } \right] ,\tag{13}
$$

where ϵ denotes the Gaussian noise added to the RGB latent representation and $\epsilon _ { \theta } ( \cdot )$ denotes the frozen diffusion U-Net. Since the diffusion backbone remains frozen, gradients from $\mathcal { L } _ { d i f f }$ are propagated through the conditioning pathway to guide the Event representations in learning RGB appearance-related information under the visual generative priors of the diffusion model.

For video-level text generation, the projected RGB-Event visual tokens V and a textual prompt p are fed into the frozen Llama-3-8B model to predict the corresponding video-level description $\textbf { y } = \{ y _ { l } \} _ { l = 1 } ^ { L _ { y } }$ . The text generation objective is defined as:

$$
\mathcal { L } _ { t e x t } = - \frac { 1 } { L _ { y } } \sum _ { l = 1 } ^ { L _ { y } } \log p \left( y _ { l } \mid \mathbf { V } , \mathbf { p } , y _ { < l } \right) .\tag{14}
$$

Only the target caption tokens contribute to $\mathcal { L } _ { t e x t , \ }$ while the visual tokens and textual prompt are excluded from loss computation. The pre-trained language model remains frozen, such that the language semantic prior is transferred to the RGB-Event representations through the learnable visual conditioning pathway.

Based on the above objectives, CM3AE 2.0 is optimized in two stages. In the first stage, we focus on event-aware spatiotemporal masked modeling and cross-modal representation learning. The overall objective is defined as

$$
\mathcal { L } _ { \mathrm { I } } = \lambda _ { m } \mathcal { L } _ { m } + \lambda _ { e s p } \mathcal { L } _ { e s p } + \lambda _ { f } \mathcal { L } _ { f u s } + \lambda _ { c } \mathcal { L } _ { c o n } ,\tag{15}
$$

where $\lambda _ { m } , \lambda _ { e s p } , \lambda _ { f } ,$ and $\lambda _ { c }$ balance the contributions of different objectives. Spatiotemporal masked reconstruction models temporal relationships across continuous RGB-Event observations, while multimodal fusion reconstruction and cross-modal contrastive learning capture complementary information and shared representations across modalities. Together, these objectives promote unified multimodal spatiotemporal representation learning.

The second stage is initialized from the model learned in Stage 1 and further incorporates visual- and language-level generative semantic supervision. Accordingly, the overall objective is formulated as

$$
\mathcal { L } _ { \mathrm { I I } } = \lambda _ { m } \mathcal { L } _ { m } + \lambda _ { e s p } \mathcal { L } _ { e s p } + \lambda _ { f } \mathcal { L } _ { f u s } + \lambda _ { d } \mathcal { L } _ { d i f f } + \lambda _ { t } \mathcal { L } _ { t e x t } ,\tag{16}
$$

where $\lambda _ { d }$ and $\lambda _ { t }$ control the contributions of the diffusion and text generation objectives, respectively. Through this two-stage optimization, CM3AE 2.0 first learns unified multimodal spatiotemporal representations and then explores latent cross-modal dependencies by linking Event dynamics with RGB appearance and joint RGB-Event representations with scene-level semantics.

## 3.4 Downstream Tasks

To comprehensively evaluate the transferability and generalization capability of CM3AE 2.0, we conduct experiments on four representative downstream tasks, including action recognition, object detection, visual object tracking, and event-guided low-light image enhancement. These tasks cover both Event-based unimodal perception and RGB-Event multimodal understanding, spanning temporal recognition, spatial perception, cross-modal fusion, and low-level visual enhancement. For downstream adaptation, the reconstruction and generation branches used during pre-training are discarded, while the pre-trained modalityspecific encoders and temporal modeling components relevant to each task are transferred to task-specific networks.

• Task 1. Action Recognition Task. Action recognition requires effective modeling of spatial appearance and temporal motion patterns in video sequences. We adopt TimeSformer [55] as the downstream framework and evaluate CM3AE 2.0 under both Event-only and RGB-Event settings. The pre-trained representations are transferred to the corresponding recognition networks for task-specific finetuning. This task evaluates the transferability of the learned multimodal spatiotemporal representations.

• Task 2. Object Detection Task. Object detection aims to recognize and localize objects in complex scenes. We adopt ViTDet [56] and SPFD [41] as the downstream frameworks for Event-only and RGB-Event detection, respectively. The pre-trained representations of CM3AE 2.0 are incorporated into the corresponding detection networks for fine-tuning. This task evaluates their effectiveness in spatial perception and multimodal feature learning.

• Task 3. Visual Object Tracking Task. Visual object tracking aims to continuously localize a target throughout a video sequence, requiring robust spatial and temporal representations. We adopt ODTrack [57] and CEUTrack [58] for Event-only and RGB-Event tracking, respectively. The pre-trained representations of CM3AE 2.0 are transferred to the corresponding tracking networks for fine-tuning. This task evaluates their transferability to sequential target perception.

• Task 4. Event-guided Low-Light Image Enhancement Task. Event-guided low-light image enhancement aims to improve image quality under challenging illumination conditions by leveraging complementary RGB and Event information. We adopt EvLight [34] as the downstream framework and incorporate the pre-trained representations of CM3AE 2.0 for task-specific fine-tuning. This task evaluates the transferability of the learned representations to low-level visual enhancement.

## 4 EXPERIMENTS

## 4.1 Datasets and Evaluation Metrics

We construct a large-scale RGB-Event dataset REV2M for pre-training and evaluate the proposed method on multiple downstream benchmarks covering action recognition, object detection, visual object tracking, and event-guided low-light image enhancement. In the following, we first introduce the pre-training and downstream datasets, and then describe the evaluation metrics adopted for different tasks.

• Pre-training Dataset. To construct a large-scale RGB-Event pre-training dataset, we integrate four publicly available datasets, including HARDVS [2], N-ImageNet [63], CO-ESOT [58], and VisEvent [64]. The source video portion contains 66,035 video sequences, while the image-level portion consists of 1,025,012 valid images. These data provide a large-scale and diverse foundation for unified RGB-Event pre-training. Representative examples of the video-level temporal data and image-level paired data are shown in Fig. 3. To accommodate different data formats, we construct the pre-training samples in a unified sequential form. For video data with continuous temporal information, the original videos are divided into fixed-length RGB-Event clips, resulting in 292,364 candidate clips as pre-training samples after filtering. For image-level data without genuine temporal continuity, the images are organized into fixed-length pseudo-sequences, with each sample containing four images, and temporal interaction is disabled during training. In total, the 1,025,012 images are organized into 256,253 image-level pseudo-sequence samples. After the above data processing and filtering procedures, the resulting pre-training dataset contains 548,617 training samples in total, including 292,364 video clip samples and 256,253 image-level pseudo-sequence samples. This large-scale pretraining corpus supports unified multimodal spatiotemporal representation learning by modeling temporal dynamics and exploiting complementary information and shared features across modalities. To prevent data leakage in downstream evaluation, we ensure that the pre-training samples do not overlap with the validation and test sequences of the downstream benchmarks.

TABLE 1  
Comparison with existing pre-trained models across three RGB-Event downstream tasks, including action recognition, object detection, and visua object tracking.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Dataset</td><td colspan="2">Action Recognition (HARDVS Dataset [2])</td><td colspan="2">Object Detection (DSEĆ-Det Dataset [59])</td><td colspan="3">Visual Object Tracking (COESOT&#x27; Dataset [58])</td></tr><tr><td>Top-1</td><td>Top-5</td><td>AP[0.5:0.95] AP0.5</td><td>AP0.75</td><td>SR</td><td>PR</td><td>NPR</td></tr><tr><td>Scratch</td><td></td><td>41.63</td><td>50.66</td><td>38.6</td><td>58.7 43.5</td><td>38.9</td><td>44.7</td><td>46.4</td></tr><tr><td>iBOT [10]</td><td>ImageNet-1K</td><td>45.12</td><td>51.36</td><td>40.9</td><td>62.5 45.1</td><td>58.3</td><td>71.3</td><td>70.3</td></tr><tr><td>BEiT v2 [12]</td><td>ImageNet-1K</td><td>45.01</td><td>51.76</td><td>41.3</td><td>63.3 46.5</td><td>43.1</td><td>51.6</td><td>53.1</td></tr><tr><td>EVA [60]</td><td>ImageNet-1K</td><td>48.14</td><td>55.06</td><td>41.1</td><td>62.5 45.7</td><td>58.1</td><td>70.7</td><td>70.5</td></tr><tr><td>PixMIM [61]</td><td>ImageNet-1K</td><td>47.17</td><td>53.44</td><td>42.0</td><td>62.7 47.0</td><td>59.4</td><td>72.8</td><td>71.2</td></tr><tr><td>MAE [9]</td><td>ImageNet-1K</td><td>50.23</td><td>59.65</td><td>41.7</td><td>62.9 46.4</td><td>60.6</td><td>73.6</td><td>72.8</td></tr><tr><td>MEM [17]</td><td>N-ImageNet</td><td>47.70</td><td>56.73</td><td>40.6</td><td>62.2 45.4</td><td>54.4</td><td>62.8</td><td>64.2</td></tr><tr><td>EventBind [21]</td><td>WIT</td><td>49.13</td><td>58.43</td><td>41.7</td><td>62.9 47.1</td><td>56.7</td><td>68.5</td><td>67.3</td></tr><tr><td>ECDDP [62]</td><td>E-TartanAir</td><td>50.13</td><td>59.29</td><td>41.2</td><td>62.5 45.9</td><td>58.3</td><td>70.5</td><td>71.6</td></tr><tr><td>MAE [9]</td><td>REV2M</td><td>50.98</td><td>62.10</td><td>42.2</td><td>62.7 46.7</td><td>61.0</td><td>73.8</td><td>73.1</td></tr><tr><td>CM3AE [25]</td><td>REV2M</td><td>53.40</td><td>65.82</td><td>43.1</td><td>64.6 48.9</td><td>62.6</td><td>75.8</td><td>74.9</td></tr><tr><td>CM3AE 2.0</td><td>REV2M</td><td>53.95</td><td>66.27</td><td>44.5</td><td>65.4 50.6</td><td>63.5</td><td>77.0</td><td>76.2</td></tr></table>

TABLE 2

Comparison with existing pre-trained models across four event-based downstream tasks, including action recognition, object detection, visual object tracking, and event-guided low-light image enhancement.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Dataset</td><td colspan="2">Action Recognition</td><td colspan="2">Object Detection</td><td colspan="3">Visual Object Tracking</td><td colspan="2">Event-guided Image Enhancement</td></tr><tr><td>(HARDVS Dataset [2])</td><td></td><td>(EvDET200K Dataset [31])</td><td></td><td></td><td></td><td>(EventVOT Dataset [32])</td><td></td><td>(SDE-in Dataset [34])</td></tr><tr><td>Top-1</td><td>Top-5</td><td>AP[0.5:0.95] AP0.5</td><td>AP0.75</td><td>SR</td><td>PR</td><td>NPR</td><td>SSIM</td><td>PSNR</td><td>PSNR*</td></tr><tr><td>Scratch</td><td></td><td>20.23</td><td>37.74</td><td>42.2 73.0</td><td>44.0 49.1</td><td>44.8 55.7</td><td>41.8</td><td>57.9 70.6</td><td>70.2 74.2</td><td>20.2</td><td>23.2</td></tr><tr><td>iBOT [10]</td><td>ImageNet-1K</td><td>22.99</td><td>39.99</td><td>46.2</td><td>76.3</td><td></td><td>59.7</td><td></td><td></td><td>21.6</td><td>24.1</td></tr><tr><td>BEiT v2 [12]</td><td>ImageNet-1K</td><td>26.61</td><td>43.52</td><td>46.6</td><td>78.4 48.4</td><td>51.1</td><td>52.0</td><td>65.0</td><td>73.8</td><td>21.5</td><td>24.0</td></tr><tr><td>EVA [60]</td><td>ImageNet-1K</td><td>36.44</td><td>50.96</td><td>47.5</td><td>78.7 50.7</td><td>56.0</td><td>60.3</td><td>71.1</td><td>74.6</td><td>21.5</td><td>24.2</td></tr><tr><td>PixMIM [61]</td><td>ImageNet-1K</td><td>35.07</td><td>49.67</td><td>47.3</td><td>77.5 50.7</td><td>55.6</td><td>59.7</td><td>70.5</td><td>75.0</td><td>21.8</td><td>24.6</td></tr><tr><td>MAE [9]</td><td>ImageNet-1K</td><td>49.21</td><td>59.01</td><td>47.4</td><td>78.1 50.9</td><td>56.6</td><td>60.7</td><td>71.4</td><td>75.2</td><td>21.9</td><td>24.9</td></tr><tr><td>MEM [17]</td><td>N-ImageNet</td><td>45.10</td><td>53.80</td><td>46.5</td><td>77.5 49.1</td><td>51.3</td><td>53.5</td><td>65.4</td><td>72.0</td><td>21.2</td><td>23.7</td></tr><tr><td>EventBind [21]</td><td>WIT</td><td>44.22</td><td>52.96</td><td>45.0</td><td>74.9 47.7</td><td>49.3</td><td>49.2</td><td>63.1</td><td>73.7</td><td>21.8</td><td>24.0</td></tr><tr><td>ECDDP [62]</td><td>E-TartanAir REV2M</td><td>49.64</td><td>56.02</td><td>47.3</td><td>77.8 50.3</td><td>54.3</td><td>59.6</td><td>69.0</td><td>75.4</td><td>22.0</td><td>24.7</td></tr><tr><td>MAE [9] CM3AE [25]</td><td>REV2M</td><td>49.32 52.45</td><td>60.41</td><td>47.5</td><td>78.0 50.3 52.4</td><td>56.8 57.6</td><td>60.5</td><td>71.5</td><td>75.5 76.0</td><td>21.9</td><td>24.6</td></tr><tr><td>CM3AE 2.0</td><td>REV2M</td><td>53.26</td><td>63.76 64.58</td><td>49.0 50.3</td><td>80.0 81.5</td><td>53.1 59.0</td><td>62.3 63.2</td><td>72.8 74.1</td><td>77.0</td><td>22.2 22.5</td><td>24.7 25.3</td></tr></table>

(a) Video-level temporal data  
![](images/652fb1edae699ebc6a63e13212e4318afb325dbd822b9234d312d7b5fa15e0d5.jpg)  
Fig. 3. Representative samples in our pre-training dataset.

• Downstream Datasets. We evaluate the transferability of the learned representations on multiple downstream benchmarks, covering action recognition, object detection, visual object tracking, and event-guided low-light image enhancement. The datasets used in our experiments are introduced as follows.

HARDVS [2] is a large-scale benchmark for human action recognition using event data. It contains 107,646 event sequences covering 300 action categories, with diverse action types and scene variations. We evaluate CM3AE 2.0 on

TABLE 3  
Ablation results of the proposed pre-training framework on four representative downstream tasks.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Stage</td><td colspan="2">Action Recognition (HARDVS Dataset)</td><td colspan="3">Object Detection (DSEC-Det Dataset)</td><td colspan="3">Visual Object Tracking (COESÓT Dataset)</td><td colspan="3">Event-guided Image Enhancement</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>SR</td><td>PR</td><td>NPR</td><td>SSIM</td><td>(SDE-in Dataset)</td><td></td></tr><tr><td>Top-1</td><td>Top-5</td><td> $\overline { { A P _ { [ 0 . 5 : 0 . 9 5 ] } } }$ </td><td>AP0.5</td><td>AP0.75</td><td></td><td></td><td></td><td></td><td>PSNR 21.5</td><td>PSNR*</td></tr><tr><td>Baseline + Multimodal Fusion Reconstruction</td><td>1</td><td>49.29</td><td>59.99</td><td>41.1</td><td>62.7</td><td>46.3</td><td>58.7</td><td>71.4</td><td>70.7</td><td>72.9</td><td></td><td>24.1</td></tr><tr><td>and Cross-modal Contrastive Learning</td><td>1</td><td>50.51</td><td>61.39</td><td>42.4</td><td>63.6</td><td>47.8</td><td>60.4</td><td>72.9</td><td>72.6</td><td>73.9</td><td>21.9</td><td>24.2</td></tr><tr><td>+ Event-guided Masking</td><td>1</td><td>51.47</td><td>62.41</td><td>42.8</td><td>64.2</td><td>48.1</td><td>61.0</td><td>73.7</td><td>73.4</td><td>74.5</td><td>22.2</td><td>24.3</td></tr><tr><td>+ Spatiotemporal Masked Reconstruction</td><td>1</td><td>52.76</td><td>64.86</td><td>43.8</td><td>65.1</td><td>49.0</td><td>62.3</td><td>75.5</td><td>74.7</td><td>75.1</td><td>22.3</td><td>24.6</td></tr><tr><td>+ Event-specific Reconstruction Loss</td><td>1</td><td>53.28</td><td>65.63</td><td>44.1</td><td>65.2</td><td>49.5</td><td>62.8</td><td>75.9</td><td>75.1</td><td>75.6</td><td>22.3</td><td>24.7</td></tr><tr><td>+ Event-conditioned Image Generation</td><td>2</td><td>53.64</td><td>66.01</td><td>44.3</td><td>65.4</td><td>50.2</td><td>63.3</td><td>76.6</td><td>75.9</td><td>76.8</td><td>22.6</td><td>25.1</td></tr><tr><td>+ Video-level Text Generation (Full Model)</td><td>2</td><td>53.95</td><td>66.27</td><td>44.5</td><td>65.4</td><td>50.6</td><td>63.5</td><td>77.0</td><td>76.2</td><td>77.0</td><td>22.5</td><td>25.3</td></tr></table>

![](images/e8ea40fe6dbde06fb454b8b7f5ea4e3cf8eff13bff944039d4c115c95c5fbf0e.jpg)  
(a) RGB-Event Action Recognition

![](images/88f1c5920e9301a0a2346c662e99d8883ce212928a60c23242807974925ddfe3.jpg)  
(b) RGB-Event Object Detection

![](images/2149345083986b5ba1de54f8a7f6f86e7e3dec1c16db73806b62b49e3c54ee98.jpg)  
(c) RGB-Event Visual Object Tracking

![](images/b1729ab6722a1edb64428d1ba74bf53ce8ef2cd7749fb8540ab72cde0640e7f9.jpg)  
(d) Event-guided Low-Light Image Enhancement  
Fig. 4. The impact of pre-training epochs on experimental results across different downstream tasks.

HARDVS under both Event-only and RGB-Event settings. The Event-only setting uses event sequences as input, while the RGB-Event setting jointly uses the corresponding RGB observations and Event data. Following the protocol in the original paper, the dataset is divided into training, validation, and testing sets with ratios of 60%, 10%, and 30%, respectively.

DSEC-Det [59] is an RGB-Event object detection dataset for autonomous driving scenarios, containing synchronized image and event data collected in complex traffic environments such as urban roads. The dataset consists of 60 sequences and eight traffic-related object categories, providing 70,379 annotated frames and 390,118 bounding boxes. According to the official split, the training, validation, and testing sets contain 41, 6, and 13 sequences, respectively.

EvDET200K [31] is a large-scale event-based object detection dataset captured using a high-resolution Prophesee EVK4-HD event camera. It contains diverse real-world scenes and targets under different motion conditions. The dataset includes 10,054 event samples covering 10 object categories and provides more than 200K bounding-box annotations.

COESOT [58] is a large-scale multimodal dataset for RGB-Event single-object tracking, where RGB images and event data are synchronously captured using an event camera. It covers diverse indoor and outdoor environments as well as challenging conditions such as low illumination, fast motion, occlusion, and scale variation. The dataset contains 1,354 RGB-Event video sequences covering 90 target categories, with 827 sequences for training and 527 sequences for testing.

EventVOT [32] is a high-resolution benchmark for eventbased visual object tracking. Compared with existing lowresolution event tracking datasets, EventVOT provides event data at a resolution of $1 2 8 0 ~ \times ~ 7 2 0$ and includes diverse targets such as pedestrians, vehicles, and UAVs under complex motion and scene variations. The dataset contains 1,141 event videos covering 19 target categories and 14 challenging attributes. The training, validation, and testing sets contain 841, 18, and 282 videos, respectively.

SDE-in [34] is the indoor subset of the SDE dataset and is mainly used for event-guided low-light image enhancement. SDE is captured using a DAVIS346 camera in realworld low-light environments and provides spatially and temporally aligned low-light images, normal-light images, and event data. The complete dataset contains 91 sequences, including 43 indoor and 48 outdoor sequences. The original SDE dataset uses 76 sequences for training and 15 sequences for testing. In our experiments, we follow the original setting and use its indoor subset, SDE-in, for evaluation.

• Evaluation Metrics. For action recognition, we adopt Top-1 and Top-5 classification accuracies as the evaluation metrics. Top-1 accuracy measures the proportion of samples for which the class with the highest predicted probability matches the ground-truth label, while Top-5 accuracy measures the proportion of samples for which the ground-truth label is included among the five classes with the highest predicted probabilities. For object detection, we use the COCOstyle Average Precision (AP), including $A P _ { 0 . 5 } , A P _ { 0 . 7 5 } .$ , and $A P _ { [ 0 . 5 : 0 . 9 5 ] }$ . Specifically, $A P _ { 0 . 5 }$ and $A P _ { 0 . 7 5 }$ denote the average precision at IoU thresholds of 0.5 and 0.75, respectively, while $A P _ { [ 0 . 5 : 0 . 9 5 ] }$ represents the average AP over multiple IoU thresholds ranging from 0.5 to 0.95 with a step size of 0.05. For visual object tracking, we adopt Success Rate (SR), Precision Rate (PR), and Normalized Precision Rate (NPR). SR mainly evaluates the overlap between the predicted and ground-truth bounding boxes, PR measures the localization accuracy based on the center distance between the predicted and ground-truth targets, and NPR further normalizes the center localization error to reduce the influence of variations in target scale and image resolution. For event-guided lowlight image enhancement, we employ Peak Signal-to-Noise Ratio (PSNR), brightness-adjusted PSNR (PSNR ), and Structural Similarity Index Measure (SSIM). PSNR evaluates the pixel-level reconstruction quality between the enhanced image and the reference image, while PSNR<sup>∗</sup> further compensates for global brightness differences. SSIM measures the similarity between two images in terms of luminance, contrast, and structural information. Higher values indicate better performance for all the above metrics.

## 4.2 Implementation Details

During the pre-training phase, we adopt ViT-B/16 as the visual backbone and train CM3AE 2.0 using a two-stage pre-training strategy. All experiments are implemented in Python based on PyTorch [65], with training conducted in BF16 precision. In the first stage, the model is trained for 100 epochs using the AdamW [66] optimizer. The base and minimum learning rates are set to $2 . { \overset {  } { 5 } } \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 6 }$ respectively, while the weight decay is gradually increased from 0.04 to 0.4. A 10-epoch learning-rate warm-up is employed. The batch size is set to 45 per GPU, resulting in an effective batch size of 180 across four GPUs. The masking ratio is set to 0.75, the temporal interaction interval is set to 4, and the guided pairwise masking strategy is adopted with a noise ratio of 0.1. In the second stage, the model is initialized with the weights obtained from the first stage and further trained for 100 epochs. AdamW is again adopted as the optimizer, with the base and minimum learning rates set to $5 \dot { \times } 1 0 ^ { - 5 }$ and $1 \times 1 0 ^ { - 6 }$ , respectively. The weight decay is gradually increased from 0.04 to 0.4, and a 2-epoch learningrate warm-up is employed. The batch size is set to 24 per GPU, corresponding to an effective batch size of 96 across four GPUs. All pre-training experiments are conducted on four NVIDIA A800 GPUs, and the complete two-stage pretraining process takes approximately 240 hours.

During the fine-tuning stage, task-specific training configurations are adopted for different downstream tasks. For human action recognition on HARDVS, the model is trained for 15 epochs with a learning rate of 0.008 and a batch size of 7. For Event-based object detection on EvDET200K, the model is trained for 15 epochs with a learning rate of 0.0001 and a batch size of 8. For RGB-Event multimodal object detection on DSEC-Det, the model is trained for 120K iterations using the AdamW optimizer, with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a total batch size of 2. For Event-based visual object tracking on EventVOT, the model is fine-tuned for 200 epochs with a learning rate of 0.0001 and a batch size of 16. For RGB-Event visual object tracking on COESOT, the model is trained for 50 epochs with a learning rate of 0.0001 and a batch size of 16. For eventguided low-light image enhancement on SDE-in, the model is trained for 80 epochs using the Adam optimizer, with a learning rate of $\mathrm { { \bar { 1 . 5 } \times 1 0 ^ { - 4 } } }$ and a batch size of 6. The training configurations for the downstream tasks largely follow those of their corresponding baseline methods. All downstream experiments are conducted on NVIDIA RTX 4090 GPUs.

## 4.3 Comparison with State-of-the-Art Algorithms

We systematically evaluate the transferability of CM3AE 2.0 across seven downstream evaluation settings covering four representative tasks under Event-only and RGB-Event configurations. As shown in Table 1 and Table 2, the compared methods mainly fall into two categories. The first category consists of general-purpose visual pre-training models trained on large-scale RGB images, including iBOT [10], BEiT v2 [12], EVA [60], PixMIM [61], and MAE [9]. The second category includes pre-training methods specifically designed for Event data, including MEM [17], EventBind [21], and ECDDP [62]. In addition, we adopt MAE pre-trained on REV2M [25] as a baseline under the same pre-training data setting, and further compare our method with CM3AE [25] proposed in our previous conference version.

TABLE 4  
Effect of different masking ratios on action recognition.
<table><tr><td rowspan="3">Mask Ratio</td><td colspan="3">Action Recognition</td></tr><tr><td>RGB+Event</td><td></td><td>Event</td></tr><tr><td>Top-1</td><td>Top-5 Top-1</td><td>Top-5</td></tr><tr><td>0.50</td><td>52.62</td><td>64.67</td><td>52.28 63.83</td></tr><tr><td>0.75</td><td>53.95 66.27</td><td>53.26</td><td>64.58</td></tr><tr><td>0.90</td><td>51.78</td><td>62.83 51.03</td><td>61.57</td></tr></table>

• Results of Action Recognition Task. On HARDVS, CM3AE 2.0 achieves 53.95% Top-1 and 66.27% Top-5 accuracy under the RGB-Event setting. Compared with MAE, a widely used general-purpose visual pre-training baseline, CM3AE 2.0 improves Top-1 accuracy by 3.72% and also outperforms ECDDP. Under the same REV2M pre-training data setting, MAE achieves 50.98% Top-1 accuracy, while CM3AE and CM3AE 2.0 further improve it to 53.40% and 53.95%, respectively. A consistent trend is also observed under the Event-only setting, where CM3AE 2.0 achieves 53.26% Top-1 accuracy, outperforming MAE, ECDDP, and CM3AE. These results demonstrate the transferability of the learned representations to action recognition under both Event-only and RGB-Event settings.

• Results of Object Detection Task. On the RGB-Event object detection task on DSEC-Det, CM3AE 2.0 achieves 44.5%, 65.4%, and 50.6% in $A P _ { [ 0 . 5 : 0 . 9 5 ] } , A P _ { 0 . 5 } ,$ , and $A P _ { 0 . 7 5 } ,$ respectively. In particular, its $A \dot { P } _ { [ 0 . 5 : 0 . 9 5 ] }$ exceeds that of the strong general-purpose pre-training model PixMIM by 2.5% and that of MAE pre-trained on the same REV2M dataset by 2.3%. On the Event-only EvDET200K dataset, CM3AE 2.0 further achieves 50.3% $A P _ { [ 0 . 5 : 0 . 9 5 ] } ,$ , outperforming methods such as MAE and ECDDP, as well as CM3AE with 49.0%. These results demonstrate the transferability of the learned representations to both Event-only and RGB-Event object detection, supporting their effectiveness in spatial perception and multimodal feature learning.

• Results of Visual Object Tracking Task. On COESOT, CM3AE 2.0 achieves 63.5% SR, 77.0% PR, and 76.2% NPR under the RGB-Event setting. Compared with MAE pretrained on ImageNet-1K, the three metrics are improved by 2.9%, 3.4%, and 3.4%, respectively, while consistent improvements are also obtained over MAE pre-trained on REV2M. On EventVOT, CM3AE 2.0 further achieves 59.0% SR, 63.2% PR, and 74.1% NPR, outperforming MAE, ECDDP, and CM3AE. These results further demonstrate the effectiveness of unified multimodal spatiotemporal representation learning in capturing temporal relationships and supporting robust sequential visual perception.

• Results of Event-guided Low-Light Image Enhancement

RGB Video  
Event Stream  
![](images/7ecc957b31964e6c32c680a5cca48f8320ee75234cbbae8372024d2dce3e6be7.jpg)  
Fig. 5. Visualization of masked image reconstruction results using MAE, CM3AE, and CM3AE 2.0.

![](images/9df09c2a17544f5be2e280928c32d60c2079e053571aa9b909db4d0d9c38edc6.jpg)  
The person in the image is seen jumping rope in place, moving their arms in a i rhythmic back and forth motion while jumping. In the beginning of the sequence, ! the person is standing with their arms outstretched, and as the rope is swung, the i person jumps and their arms move up, then down

Fig. 6. Qualitative visualization of video-level text generation.  
![](images/63c2134812677851ed89d1a61ca8df5dbb910e7757057590f743c2a3b1387d77.jpg)  
Fig. 7. Qualitative visualization of RGB images generated from Eventconditioned visual representations by CM3AE 2.0.

Event Stream  
RGB Video  
![](images/c68437a1e6d4a252891e9670fa13520976ebcb29f2ecd9c430330770c316ebca.jpg)  
The person in the video is performing the Simplified Tai Chi Right Kick. At the beginning of the sequence, the person stands upright with the left leg supporting the body. As the movement progresses, the right leg is raised and extended forward, while the arms adjust to maintain balance.

Task. On SDE-in, CM3AE 2.0 achieves an SSIM of 77.0%, a PSNR of 22.5 dB, and a PSNR<sup>∗</sup> of 25.3 dB. Compared with the general-purpose visual pre-training model MAE, all three metrics are improved, while consistent advantages are also observed over the Event-oriented pre-training method ECDDP and MAE pre-trained on REV2M. Moreover, compared with CM3AE, CM3AE 2.0 improves SSIM from 76.0% to 77.0%, PSNR from 22.2 dB to 22.5 dB, and PSNR<sup>∗</sup> from 24.7 dB to 25.3 dB. These results demonstrate that the learned representations effectively exploit complementary RGB-Event information for low-light image enhancement, indicating their transferability to low-level visual perception.

## 4.4 Component Analysis

To investigate the contributions of the ESMR and HCG strategies, we progressively evaluate their components, as shown in Table 3. The baseline achieves 49.29% Top-1 accuracy on action recognition, 41.1% $A P _ { [ 0 . 5 : 0 . 9 5 ] }$ on object detection, and 58.7% SR on visual object tracking. Introducing multimodal fusion reconstruction and crossmodal contrastive learning improves performance across downstream tasks, supporting the benefits of exploiting complementary information and learning shared features across modalities. Event-guided masking further improves the learned representations by emphasizing dynamically salient regions. Incorporating spatiotemporal masked reconstruction increases the Top-1 accuracy from 51.47% to 52.76% and the tracking SR from 61.0% to 62.3%, highlighting the importance of modeling temporal dynamics in continuous observations. The event-specific reconstruction loss further improves performance by supervising event occurrence and polarity prediction.

![](images/1bac50f9dc33a3676256eb14a203531d5dde1e48f7416050953102e3d1e59cd9.jpg)  
Fig. 8. Visualization of the experimental results of each downstream task.

TABLE 5  
Effect of different numbers of input frames on action recognition.
<table><tr><td rowspan="3">Frames</td><td colspan="3">Action Recognition</td></tr><tr><td>RGB-Event</td><td></td><td>Event</td></tr><tr><td>Top-1</td><td>Top-5</td><td>Top-1</td><td>Top-5</td></tr><tr><td>2</td><td>52.13</td><td>64.85</td><td>52.27</td><td>63.74</td></tr><tr><td>4</td><td>53.95</td><td>66.27</td><td>53.26</td><td>64.58</td></tr><tr><td>8</td><td>54.03</td><td>66.14</td><td>53.19</td><td>64.52</td></tr></table>

In Stage 2, event-conditioned RGB image generation further improves performance, particularly on low-light image enhancement, where SSIM increases from 75.6% to 76.8%. Incorporating video-level text generation yields the full CM3AE 2.0 model, further increasing SSIM to 77.0% and improving performance on other downstream tasks. Compared with the baseline, the full model improves Top-1 action recognition accuracy from 49.29% to 53.95%, detection $A P _ { [ 0 . 5 : 0 . 9 5 ] }$ from 41.1% to 44.5%, tracking SR from 58.7% to 63.5%, and low-light enhancement SSIM from 72.9% to 77.0%. These results demonstrate that the ESMR strategy enables unified multimodal spatiotemporal representation learning, while the HCG strategy further explores latent cross-modal dependencies through visual and language generation.

## 4.5 Ablation Study

• Effects of Pre-training Epochs. We first investigate the influence of the number of pre-training epochs on different downstream tasks. As shown in Fig. 4, the performance of action recognition, object detection, visual object tracking, and low-light image enhancement generally improves as pre-training progresses. Although slight fluctuations are observed for some metrics, the overall performance continues to improve and gradually becomes stable in the later stage of pre-training. This trend indicates that sufficient pre-training is important for learning unified multimodal spatiotemporal representations and exploring cross-modal dependencies.

• Effects of Masking Ratio. Table 4 reports the results with different masking ratios. When the masking ratio increases from 0.50 to 0.75, the RGB-Event Top-1 accuracy improves from 52.62% to 53.95%, while the Event-only Top-1 accuracy increases from 52.28% to 53.26%. However, further increasing the masking ratio to 0.90 causes noticeable performance degradation under both settings. These results indicate that a moderate masking ratio of 0.75 provides a better balance between reconstruction difficulty and the amount of visible contextual information, and is therefore adopted as the default setting.

• Effects of the Number of Input Frames. We further evaluate the effect of different numbers of input frames in Table 5. Increasing the number of frames from 2 to 4 consistently improves both RGB-Event and Event-only action recognition. When the number of frames is further increased to 8, the RGB-Event Top-1 accuracy only slightly increases from 53.95% to 54.03%, while the other metrics show no further improvement. Considering both performance and computational cost, four input frames provide a better overall trade-off and are adopted in our experiments.

![](images/95d15c01a1f0bf3b11b90488f115423c3a207f6a4187bc1d827c9fd9fd6e4f2a.jpg)  
RGB-Event based Object Detection  
Event-guided Low-Light Image Enhancement

Fig. 9. Visualization of attention maps on different downstream tasks.  
TABLE 6  
Few-shot transfer results of different pre-trained models on multiple downstream tasks using 10% of the labeled training data for fine-tuning.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pre-training Data</td><td colspan="2">Action Recognition (HARDVS Dataset)</td><td colspan="2">Object Detection (DSEC-Det Dataset)</td><td colspan="2">Visual Object Tracking (COESÓT Dataset)</td><td colspan="2">Low-Light Enhancement (SDE-in Dataset)</td></tr><tr><td>Top-1</td><td>Top-5</td><td>AP[0.5:0.95]  $\overline { { A P _ { 0 . 5 } } }$   $\overline { { A P _ { 0 . 7 5 } } }$ </td><td>SR</td><td>PR</td><td>NPR</td><td>SSIM PSNR</td><td>PSNR*</td></tr><tr><td>Scratch</td><td></td><td>5.63</td><td>14.52</td><td>17.1 29.2 17.2</td><td>26.3</td><td>26.9</td><td>30.1</td><td>59.4</td><td>19.9</td><td>20.1</td></tr><tr><td>MAE [9]</td><td>ImageNet-1K</td><td>6.65</td><td>18.04</td><td>21.1 34.2</td><td>40.1</td><td>48.7</td><td>50.7</td><td>61.3</td><td>20.2</td><td>21.3</td></tr><tr><td>MAE [9]</td><td>REV2M</td><td>33.31</td><td>49.01</td><td>23.1 35.8</td><td>21.4 23.7 41.5</td><td>50.8</td><td>52.3</td><td>63.2</td><td>20.7</td><td>21.7</td></tr><tr><td>CM3AE [25]</td><td>REV2M</td><td>43.68</td><td>57.23</td><td>26.6 39.8</td><td>26.1 43.9</td><td>55.5</td><td>56.0</td><td>63.8</td><td>21.1</td><td>22.2</td></tr><tr><td>CM3AE 2.0</td><td>REV2M</td><td>44.96</td><td>59.78</td><td>28.4 42.1</td><td>27.9 45.7</td><td>59.1</td><td>57.9</td><td>69.4</td><td>22.1</td><td>22.8</td></tr></table>

<table><tr><td rowspan="2">Method</td><td rowspan="2">Pre-training Data</td><td colspan="2">Action Recognition (HARDVS Dataset)</td><td colspan="3">Object Detection (EvDET200K Dataset)</td><td colspan="3">Visual Object Tracking (EventVOT Dataset)</td></tr><tr><td>Top-1</td><td>Top-5</td><td> $\overline { { A P _ { [ 0 . 5 : 0 . 9 5 ] } } }$ </td><td> $\overline { { A P _ { 0 . 5 } } }$ </td><td> $\overline { { A P _ { 0 . 7 5 } } }$ </td><td>SR</td><td>PR</td><td>NPR</td></tr><tr><td>Scratch</td><td></td><td>2.40</td><td>8.06</td><td>26.7</td><td>47.3</td><td>26.9</td><td>25.4</td><td>20.0</td><td>30.7</td></tr><tr><td>MAE [9]</td><td>ImageNet-1K</td><td>9.31</td><td>24.63</td><td>33.3</td><td>60.2</td><td>33.3</td><td>37.0</td><td>35.5</td><td>45.9</td></tr><tr><td>MAE [9]</td><td>REV2M</td><td>16.94</td><td>37.27</td><td>34.2</td><td>62.1</td><td>34.9</td><td>39.2</td><td>39.8</td><td>48.1</td></tr><tr><td>CM3AE [25]</td><td>REV2M</td><td>32.34</td><td>52.18</td><td>37.7</td><td>65.9</td><td>38.7</td><td>41.1</td><td>42.7</td><td>50.5</td></tr><tr><td>CM3AE 2.0</td><td>REV2M</td><td>34.52</td><td>55.41</td><td>38.3</td><td>67.1</td><td>39.6</td><td>43.6</td><td>44.2</td><td>51.9</td></tr></table>

• Few-shot Transferability. To further evaluate transferability under limited supervision, we fine-tune different pretrained models using only 10% of the labeled training data, as shown in Table 6. Compared with CM3AE, CM3AE 2.0 improves SSIM from 63.8% to 69.4% on low-light enhancement, Top-1 accuracy from 32.34% to 34.52% on Eventonly action recognition, and SR from 41.1% to 43.6% on Event-only tracking. Consistent gains are observed across the remaining tasks and metrics, demonstrating the effectiveness and transferability of the learned representations with limited labeled data.

## 4.6 Efficiency Analysis

Table 7 compares the computational complexity and model size of different pre-training variants. From MAE to CM3AE, FLOPs increase from 287.42 G to 391.56 G, and the parameter count increases from 335.82 M to 350.00 M, mainly due to multimodal fusion and cross-modal contrastive learning. Incorporating ESMR further increases the computational cost to 404.81 G FLOPs and the parameter count to 357.09 M, indicating limited additional overhead for unified multimodal spatiotemporal representation learning. The main increase comes from HCG in Stage 2. Introducing the language generation branch increases the parameter count to 8389.46 M, while adding the diffusion branch raises FLOPs to 1682.85 G. With both branches enabled, CM3AE 2.0 reaches 2088.18 G FLOPs, 1044.09 G MACs, and 9333.03 M parameters. These costs mainly arise from the pretrained generative models used to introduce visual generative priors and linguistic knowledge for latent cross-modal dependency learning. Importantly, the diffusion model and language model are used only during pre-training and are discarded during downstream fine-tuning and inference. Only the pre-trained modality-specific encoders and temporal modeling components required by each downstream task are transferred to downstream networks. Therefore, the additional computational and parameter overhead of HCG does not carry over to downstream deployment.

TABLE 7  
Comparison of computational complexity and model size of different pre-training configurations.
<table><tr><td>Configuration</td><td>FLOPs (G)</td><td>MACs (G)</td><td>Params (M)</td></tr><tr><td>MAE</td><td>287.42</td><td>143.71</td><td>335.82</td></tr><tr><td>CM3AE CM3AE 2.0: Stage 1 only</td><td>391.56 404.81</td><td>195.78</td><td>350.00</td></tr><tr><td>+ text generation</td><td>810.14</td><td>202.41</td><td>357.09</td></tr><tr><td></td><td></td><td>405.07</td><td>8389.46</td></tr><tr><td>+ image generation</td><td>1682.85</td><td>841.43</td><td>1300.66</td></tr><tr><td>+ both (full model)</td><td>2088.18</td><td>1044.09</td><td>9333.03</td></tr></table>

## 4.7 Visualization

In this section, we visualize masked reconstruction, video-level text generation, and Event-conditioned RGB image generation during pre-training, followed by downstream predictions and attention maps. As shown in Fig. 5, CM3AE 2.0 reconstructs more complete object structures and clearer visual details than MAE and CM3AE, indicating improved reconstruction quality through unified multimodal representation learning. Fig. 6 presents the video-level text generation results. The generated descriptions capture the main scene content and human actions, demonstrating the ability to establish connections between joint RGB-Event representations and linguistic descriptions. Fig. 7 further shows the RGB images generated from Eventconditioned representations. The results recover reasonable scene structures and appearance information, illustrating how visual generative priors contribute to Event representation learning.

We further visualize the downstream results in Fig. 8, including action recognition, object detection, visual object tracking, and Event-guided low-light image enhancement. The results demonstrate the transferability of CM3AE 2.0 to diverse perception tasks under both RGB-Event and Eventonly settings. Finally, Fig. 9 compares the attention maps of MAE, CM3AE, and CM3AE 2.0 across different downstream tasks. CM3AE 2.0 exhibits stronger responses in task-relevant regions and more focused attention on target objects, providing qualitative evidence of improved visual representation learning.

## 4.8 Limitation Analysis

Although CM3AE 2.0 achieves consistent improvements across various RGB-Event and Event-only downstream tasks, two limitations remain. First, the framework relies on paired RGB-Event data for pre-training, while collecting high-quality synchronized multimodal data is costly. Future work could explore pre-training with unpaired or weakly paired data to leverage larger-scale data resources. Second, the hierarchical cross-modal generation strategy relies on large pre-trained diffusion and language models, introducing considerable computational overhead during pre-training. Although these models are discarded during downstream transfer and inference, developing more efficient approaches to exploit visual generative priors and linguistic knowledge for cross-modal dependency learning remains an important direction for future research.

## 5 CONCLUSION AND FUTURE WORK

In this paper, we propose CM3AE 2.0, a unified RGB Frame and Event-Voxel/-Frame pre-training framework. In the first stage, we introduce an Event-aware spatiotemporal masked reconstruction strategy to learn unified multimodal spatiotemporal representations. It models spatiotemporal relationships through masked reconstruction while exploiting cross-modal complementary information and learning shared features. In the second stage, we further propose a hierarchical cross-modal generation strategy to explore latent cross-modal dependencies. It combines event-conditioned RGB image generation and video-level text generation to leverage visual generative priors and linguistic knowledge. We evaluate CM3AE 2.0 on four downstream tasks under both Event-only and RGB-Event settings. Extensive experiments, ablation studies, and qualitative visualizations demonstrate the effectiveness and transferability of the learned representations across diverse perception tasks, supporting the benefits of unified multimodal spatiotemporal representation learning and generative cross-modal dependency modeling. In future work, we will explore more efficient approaches to transferring visual generative priors and linguistic knowledge while reducing pre-training overhead. We also plan to extend RGB-Event pre-training to larger-scale and more diverse datasets, including unpaired or weakly paired data, to improve the scalability and generalization of multimodal pre-training.

## REFERENCES

[1] G. Gallego, T. Delbruck, G. Orchard, C. Bartolozzi, B. Taba, ¨ A. Censi, S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis et al., “Event-based vision: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 1, pp. 154–180, 2020.

[2] X. Wang, Z. Wu, B. Jiang, Z. Bao, L. Zhu, G. Li, Y. Wang, and Y. Tian, “Hardvs: Revisiting human activity recognition with dynamic vision sensors,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 6, 2024, pp. 5615–5623.

[3] H. Zhang, C. Xu, X. Wang, B. Liu, G. Hua, L. Yu, and W. Yang, “Detecting every object from events,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 7171–7178, 2025.

[4] N. Messikommer, C. Fang, M. Gehrig, G. Cioffi, and D. Scaramuzza, “Data-driven feature tracking for event cameras with and without frames,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3706–3717, 2025.

[5] L. Kong, Y. Liu, L. X. Ng, B. R. Cottereau, and W. T. Ooi, “Openess: Event-based semantic scene understanding with open vocabularies,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 15 686–15 698.

[6] J. Yang, X. Zhang, S. Wang, Y. Sun, and B. Yin, “Unsp: Improving event-to-image reconstruction with uncertainty guided self-paced learning,” Displays, vol. 87, p. 102985, 2025.

[7] X. Wang, G. Chen, G. Qian, P. Gao, X.-Y. Wei, Y. Wang, Y. Tian, and W. Gao, “Large-scale multi-modal pre-trained models: A comprehensive survey,” Machine Intelligence Research, vol. 20, no. 4, pp. 447–482, 2023.

[8] X. Wang, S. Wang, Y. Ding, Y. Li, W. Wu, Y. Rong, W. Kong, J. Huang, S. Li, H. Yang et al., “State space model for new generation network alternative to transformers: a survey,” Visual Intelligence, vol. 4, no. 1, p. 29, 2026.

[9] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009.

[10] J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong, “Image bert pre-training with online tokenizer,” in International Conference on Learning Representations, 2022.

[11] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[12] Z. Peng, L. Dong, H. Bao, Q. Ye, and F. Wei, “Beit v2: Masked image modeling with vector-quantized visual tokenizers,” arXiv preprint arXiv:2208.06366, 2022.

[13] Z. Tong, Y. Song, J. Wang, and L. Wang, “Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training,” Advances in neural information processing systems, vol. 35, pp. 10 078–10 093, 2022.

[14] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao, “Videomae v2: Scaling video masked autoencoders with dual masking,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 14 549–14 560.

[15] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning, 2021, pp. 8748–8763.

[16] R. Bachmann, D. Mizrahi, A. Atanov, and A. Zamir, “Multimae: Multi-modal multi-task masked autoencoders,” in European conference on computer vision. Springer, 2022, pp. 348–367.

[17] S. Klenk, D. Bonello, L. Koestler, N. Araslanov, and D. Cremers, “Masked event modeling: Self-supervised pretraining for event cameras,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 2378–2388.

[18] Y. Yang, L. Pan, and L. Liu, “Event camera data pre-training,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 10 665–10 675.

[19] Z. Huang, C. Li, H. Chen, Y. Deng, Y. Geng, and L. Wang, “Data-efficient event camera pre-training via disentangled masked modeling,” arXiv preprint arXiv:2403.00416, 2024.

[20] M. Mohammadi, Z. Wu, and I. Gilitschenski, “Tespec: Temporallyenhanced self-supervised pretraining for event cameras,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 7782–7793.

[21] J. Zhou, X. Zheng, Y. Lyu, and L. Wang, “Eventbind: Learning a unified representation to bind them all for event-based openworld understanding,” in European Conference on Computer Vision, 2024, pp. 477–494.

[22] S. Jeong, H. Chen, S. Yun, S. Cho, W. Huang, X. Liu, and M. Imani, “Cross-modal event encoder: Bridging image–text knowledge to event streams,” in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2026, pp. 3213–3222.

[23] Z. Chen, J. Hou, Z. Zhu, J. Wu, and G. Shi, “Scaling dense eventstream pretraining from visual foundation models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2026, pp. 8011–8022.

[24] S. Liu, J. Li, G. Zhao, Y. Zhang, X. Meng, F. R. Yu, X. Ji, and M. Li, “Eventgpt: Event stream understanding with multimodal large language models,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 29 139– 29 149.

[25] W. Wu, X. Wang, C. Li, B. Jiang, J. Tang, B. Luo, and Q. Liu, “Cm3ae: A unified rgb frame and event-voxel/-frame pre-training framework,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 2159–2168.

[26] H. Ren, Y. Zhou, J. Zhu, X. Lin, H. Fu, Y. Huang, Y. Fang, F. Ma, H. Yu, and B. Cheng, “Rethinking efficient and effective pointbased networks for event camera classification and regression,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[27] M. Cao, X. Shu, X. Jiang, R. Yan, Y. Yao, and J. Tang, “Exploiting frequency dynamics for enhanced multimodal event-based action recognition,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 5969–5979.

[28] X. Wang, S. Wang, P. Shao, L. Zhu, B. Jiang, and Y. Tian, “Event stream based human action recognition: A high-definition benchmark dataset and algorithms,” International Journal of Computer Vision, vol. 134, no. 4, p. 181, 2026.

[29] M. Gehrig and D. Scaramuzza, “Recurrent vision transformers for object detection with event cameras,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 13 884–13 893.

[30] L. Zhu, T. Long, X. Wang, L. Wang, and H. Huang, “Rethinking scale-aware temporal encoding for event-based object detection,” Advances in Neural Information Processing Systems, vol. 38, pp. 144 008–144 032, 2026.

[31] X. Wang, Y. Jin, W. Wu, W. Zhang, L. Zhu, B. Jiang, and Y. Tian, “Object detection using event camera: A moe heat conduction based detector and a new benchmark dataset,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 29 321–29 330.

[32] X. Wang, S. Wang, C. Tang, L. Zhu, B. Jiang, Y. Tian, and J. Tang, “Event stream-based visual object tracking: A high-resolution benchmark dataset and a novel baseline,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 19 248–19 257.

[33] L. Zhu, X. Chen, L. Wang, X. Wang, Y. Tian, and H. Huang, “Continuous-time object segmentation using high temporal resolution event camera,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 2, pp. 807–824, 2024.

[34] G. Liang, K. Chen, H. Li, Y. Lu, and L. Wang, “Towards robust event-guided low-light image enhancement: A large-scale real-world event-image dataset and novel approach,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 23–33.

[35] J. Weng, B. Li, and K. Huang, “Event-based image enhancement under high dynamic range scenarios,” in Asian Conference on Computer Vision. Springer, 2024, pp. 346–360.

[36] L. Sun, Y. Bao, J. Zhai, J. Liang, Y. Zhang, K. Wang, D. P. Paudel, and L. Van Gool, “Low-light image enhancement using eventbased illumination estimation,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 6667–6677.

[37] C. Sun, J. Zhang, Y. Wang, H. Ge, Q. Xia, B. Yin, and X. Yang, “Exploring historical information for rgbe visual tracking with mamba,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 6500–6509.

[38] C. Sun, J. Zhang, Y. Wang, Y. Wang, Y. Jiang, B. Yin, and X. Yang, “Aligntrack: Top-down spatiotemporal resolution alignment for rgb-event visual tracking,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 11, 2026, pp. 9171–9179.

[39] Y. Zhu, X. Wang, C. Li, B. Jiang, L. Zhu, Z. Huang, Y. Tian, and J. Tang, “Crsot: Cross-resolution object tracking using unaligned frame and event cameras,” IEEE Transactions on Multimedia, 2025.

[40] D. Lu, L. Kong, G. H. Lee, C. Simon Chane, and W. Ooi, “Flexevent: towards flexible event-frame object detection at varying operational frequencies,” Advances in Neural Information Processing Systems, vol. 38, pp. 89 009–89 044, 2026.

[41] K. Wang, S. Liu, H. Shi, L. Shi, and H. Chen, “Beyond duality: A hybrid framework of leveraging shared and private features for rgb-event object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 4415–4424.

[42] S. Wang, X. Wang, B. Jiang, L. Zhu, G. Li, Y. Wang, Y. Tian, and J. Tang, “Human activity recognition using rgb-dvs cameras: A multi-modal heat conduction model and a benchmark dataset,” International Journal of Computer Vision, vol. 134, no. 8, p. 377, 2026.

[43] X. Wang, H. Wang, S. Wang, Q. Chen, J. Jin, H. Song, B. Jiang, and C. Li, “Rgb-event based pedestrian attribute recognition: A benchmark dataset and an asymmetric rwkv fusion framework,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 32 745–32 755.

[44] H. Wang, H. Zhou, H. Liu, and L. Yan, “Injecting frame-event complementary fusion into diffusion for optical flow in challenging scenes,” Advances in Neural Information Processing Systems, vol. 38, pp. 31 614–31 633, 2026.

[45] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in 2020

IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2020, pp. 9726–9735.

[46] W. Wu, X. Wang, C. Li, J. Tang, and B. Luo, “Vehicle-centric perception via multimodal structured pre-training,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, pp. 8615–8630, 2026.

[47] E. Fini, M. Shukor, X. Li, P. Dufter, M. Klein, D. Haldimann, S. Aitharaju, V. G. T. da Costa, L. Bethune, Z. Gan ´ et al., “Multimodal autoregressive pre-training of large vision encoders,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 9641–9654.

[48] N. Stracke, S. A. Baumann, K. Bauer, F. Fundel, and B. Ommer, “Cleandift: Diffusion features without noise,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 117–127.

[49] T. Zhang, Y. Li, Y.-C. Chou, J. Chen, A. Yuille, C. Wei, and J. Xiao, “Vision-language-vision auto-encoder: Scalable knowledge distillation from diffusion models,” Advances in Neural Information Processing Systems, vol. 38, pp. 79 174–79 202, 2026.

[50] Q. Liang, Q. Li, S. Liu, X. Cao, J. Lu, F. Yang, W. Zhang, K. Huang, and Y. Tian, “Efficient event camera data pretraining with adaptive prompt fusion,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 8656–8667.

[51] D. Lu, R. Li, A. Liang, L. Kong, W. Yin, L. X. Ng, B. R. Cottereau, C. S. Chane, and W. T. Ooi, “Eventdrive: Event cameras for visionlanguage driving intelligence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 22 312–22 322.

[52] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 684–10 695.

[53] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[54] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[55] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” arXiv preprint arXiv:2102.05095, 2021.

[56] Y. Li, H. Mao, R. Girshick, and K. He, “Exploring plain vision transformer backbones for object detection,” in European conference on computer vision. Springer, 2022, pp. 280–296.

[57] Y. Zheng, B. Zhong, Q. Liang, Z. Mo, S. Zhang, and X. Li, “Odtrack: Online dense temporal token learning for visual tracking,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 7, 2024, pp. 7588–7596.

[58] C. Tang, X. Wang, J. Huang, B. Jiang, L. Zhu, S. Chen, J. Zhang, Y. Wang, and Y. Tian, “Revisiting color-event based tracking: A unified network, dataset, and metric,” Pattern Recognition, p. 112718, 2025.

[59] D. Gehrig and D. Scaramuzza, “Low-latency automotive vision with event cameras,” Nature, vol. 629, no. 8014, pp. 1034–1040, 2024.

[60] Y. Fang, W. Wang, B. Xie, Q. Sun, L. Wu, X. Wang, T. Huang, X. Wang, and Y. Cao, “Eva: Exploring the limits of masked visual representation learning at scale,” in 2023 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2023, pp. 19 358–19 369.

[61] Y. Liu, S. Zhang, J. Chen, K. Chen, and D. Lin, “Pixmim: Rethinking pixel reconstruction in masked image modeling,” Transactions on Machine Learning Research.

[62] Y. Yang, L. Pan, and L. Liu, “Event camera data dense pretraining,” in European Conference on Computer Vision. Springer, 2024, pp. 292–310.

[63] J. Kim, J. Bae, G. Park, D. Zhang, and Y. M. Kim, “N-imagenet: Towards robust, fine-grained object recognition with event cameras,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 2146–2156.

[64] X. Wang, J. Li, L. Zhu, Z. Zhang, Z. Chen, X. Li, Y. Wang, Y. Tian, and F. Wu, “Visevent: Reliable object tracking via collaboration of frame and event flows,” IEEE Transactions on Cybernetics, vol. 54, no. 3, pp. 1997–2010, 2023.

[65] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[66] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

![](images/335d17e4638b2e9eb1e5f7ba78128b6c8bc42e0d71b53432765314185f9a22b5.jpg)

Wentao Wu received the B.S. degree from Jiangxi University of Science and Technology, Nanchang, China, in 2022. He is currently working toward the Ph.D. degree with the school of Artificial Intelligence, Anhui University. His research interests include computer vision and deep learning.

![](images/3065b5f59abaaa154e6fe2861f8fc1d1e49a0ba114f009aa2c65c902366e155a.jpg)

Xiao Wang (Member, IEEE) received the B.S. degree from West Anhui University, Luan, China, in 2013. He received the Ph.D. degree in computer science from Anhui University, Hefei, China, in 2019. From 2015 and 2016, he was a visiting student at the School of Data and Computer Science, Sun Yat-sen University, Guangzhou, China. He also has a visit to UBTECH Sydney Artificial Intelligence Centre, the Faculty of Engineering, the University of Sydney, in 2019. He finished the postdoc research at Peng Cheng Laboratory, Shenzhen, China, from April 2020 to April 2022. He is now an Associate Professor at the School of Computer Science and Technology, Anhui University, Hefei, China. His current research interests are mainly Computer Vision, Event-based Vision, Machine Learning, and Pattern Recognition. He serves as a reviewer for a number of journals and conferences, such as IEEE TCSVT, TNNLS, TIP, IJCV, CVIU, PR, CVPR, ICCV, AAAI, ECCV, ACCV, ACM-MM, and WACV. Homepage: https://wangxiao5791509.github.io

![](images/4e58bf700dbdaf6c2ca73a57ac035af5b64316e1a5b93e20ac69ca8937769c4b.jpg)

Yi Wang received the B.S. degree from Anhui Normal University, Wuhu, China. He is currently working toward the M.S. degree with the School of Computer Science and Technology, Anhui University, Hefei, China. His research interests include time-series classification, signal processing, and edge-localized mode (ELM) recognition.

![](images/972e9ca07d5aef2aebce1b76f2e8495ce873a9618fe81e4500fac94b340f3d26.jpg)

Chenglong Li (Senior Member, IEEE) received the M.S. and Ph.D. degrees from the School of Computer Science and Technology,Anhui University, Hefei, China, in 2013 and 2016, respectively. From 2014 to 2015, he was a Visiting Student at the School of Artificial Intelligence, Sun Yat-sen University, Guangzhou, China. He was a Post-Doctoral Research Fellow with the Center for Research on Intelligent Perception and Computing (CRIPAC), National Laboratory of Pattern Recognition (NLPR), Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing, China. He is currently a Professor and a Ph.D. Supervisor with the School of Artificial Intelligence, Anhui University. His research interests include computer vision and deep learning

![](images/4f761c6cb76d3cde9994750a63c32a2002515cd86dad0399bf373c522ce8d94e.jpg)

Yonghong Tian (Fellow, IEEE) is currently a Boya Distinguished Professor with the Department of Computer Science and Technology, Peking University, China, and is also the deputy director of Artificial Intelligence Research Center, PengCheng Laboratory, Shenzhen, China. His research interests include neuromorphic vision, braininspired computation and multimedia big data. He is the author or coauthor of over 200 technical articles in refereed journals

ACM CSUR/TOIS/TOMM and conferences such as NeurIPS/CVPR/ICCV/AAAI/ACMMM/WWW. Prof. Tian was/is an Associate Editor of IEEE TCSVT (2018.1-), IEEE TMM (2014.8- 2018.8), IEEE Multimedia Mag. (2018.1-), and IEEE Access (2017.1-). He co-initiated IEEE Int’l Conf. on Multimedia Big Data (BigMM) and served as the TPC Co-chair of BigMM 2015, and aslo served as the Technical Program Co-chair of IEEE ICME 2015, IEEE ISM 2015 and IEEE MIPR 2018/2019, and General Co-chair of IEEE MIPR 2020 and ICME2021. He is the steering member of IEEE ICME (2018-) and IEEE BigMM (2015-), and is a TPC Member of more than ten conferences such as CVPR, ICCV, ACM KDD, AAAI, ACM MM and ECCV. He was the recipient of the Chinese National Science Foundation for Distinguished Young Scholars in 2018, two National Science and Technology Awards and three ministerial-level awards in China, and obtained the 2015 EURASIP Best Paper Award for Journal on Image and Video Processing, and the best paper award of IEEE BigMM 2018. He is a senior member of IEEE, CIE and CCF, a member of ACM.

![](images/e387f2aa56cae8773ae03960ff76d96009ec42bc5824f54fec0b03c7e054602b.jpg)

statistical pattern recognition, digital watermarking, and information security.