# Multi-Scale Latent Temporal Compression Attention and Video-Text Cross Attention Fusion for Video Action Detection

<sub>Yicheng</sub> <sub>Qiu</sub>[0009-0006-8841-8648] <sub>and</sub> <sub>Keiji</sub> <sub>Yanai</sub>(B)[0000-0002-0431-183X]

The University of Electro-Communications, Tokyo, Japan qiu-y@mm.inf.uec.ac.jp, yanai@cs.uec.ac.jp

Abstract. In video action detection, modeling the temporal dependencies in continuous video streams is crucial. Previous methods based on deep neural networks can perform frame-level video recognition tasks. However, these methods struggle to model dynamic temporal patterns, facing challenges such as redundant frame information, dificulty balancing global and local features, and insuficient fusion of visual and textual features. To address these issues, we propose a video temporal attention module based on the Multi-Scale Latent Temporal Compression Attention, which aggregates continuous temporal sequences into feature representations across diverse temporal scales while eficiently reducing redundancy. We also introduce a Video-Text Cross Attention Fusion module that leverages cross-attention at both latent and standard feature levels, achieving fine-grained fusion between video and textual features to enhance semantic alignment and feature interaction. Experimental results across multiple video recognition benchmarks demonstrate that our proposed methods achieve superior performance compared to existing methods, validating the efectiveness and generalization capability.

Keywords: Video Action Detection · Temporal Attention · Video-Text Fusion · Anomaly Detection

## 1 Introduction

Video action detection is a critical task in multimodal video understanding, with a wide range of real-world applications such as intelligent surveillance [26,35,37], autonomous driving [6, 30], human behavior analysis [34, 55], and embodied intelligence [9]. Robust models are required to recognize and temporally localize predefined actions as well as abnormal events such as accidents or illegal activities in long, untrimmed video streams. Early approaches extract spatio-temporal features with 3D convolutional networks, while recent methods build on Transformer architectures, state space models [14, 38], and multimodal pre-training to model longer temporal context and to align visual content with category semantics. Although recent Multimodal Large Language Models (MLLMs) [60,63] generalize well on general video tasks, their limited capacity for long-video temporal reasoning, generative uncertainty, and heavy computational cost make them ill-suited for fine-grained video action detection, motivating specialized architectures with eficient temporal attention and multimodal fusion. Transformer attention [47] is widely used to model temporal context in videos, but two problems remain. First, attention over long sequences repeatedly processes a large amount of redundant frame information, which limits both feature quality and training eficiency. Existing remedies either apply pooling [44], which irreversibly discards fine-grained details, or window-based attention [27], which restricts the receptive field and misses long-range dependencies; recent eficient temporal models [4] still struggle to obtain global context and local precision at the same time. To address this issue, we propose a video temporal attention module based on Multi-Scale Latent Temporal Compression Attention, termed MLTC-Attention. The key insight is to first project features into a compact latent space and then compress the temporal dimension at multiple scales before performing attention, rather than attending to all raw features directly. This compress-then-attend design, inspired by recent sparse and compressed attention [50,61], filters temporal redundancy while larger compression scales preserve global context and smaller scales preserve local details, and a gate re-weights the scales according to the input content.

![](images/ef35d02d89647b1607970aa8ab870ca7b8154f3a95bd56d3903d32365de2ea99.jpg)  
(a) Framework overview

![](images/35b4e5b3926d5c28b98bf4aba599bd79bfb938aee9a27ce9effed9f4f1fad922.jpg)  
(b) Performance vs. eficiency  
Fig. 1: (a) MLTC-Attention models multi-scale temporal dependencies via compress-then-attend in latent space, and VTCA-Fusion enforces hierarchical video-text alignment at both standard and latent feature levels. (b) Performance–eficiency on video anomaly detection (blue, left axis) and video action detection (orange, right axis): our method (red) attains the best accuracy at the highest inference speed (FPS) with the fewest FLOPs on both tasks.

Second, multimodal learning has become a key route to better video understanding, but previous video-text methods [49, 68] align video and text features at a single representation level, which limits the depth of cross-modal interaction. To address this limitation, we propose a multimodal feature fusion module, termed Video-Text Cross Attention Fusion (VTCA-Fusion), which enforces cross-modal alignment at two complementary levels: the standard feature space captures explicit semantic correspondences, while a parallel latent space branch, inspired by latent space feature interaction [62], captures compressed cross-modal patterns. By exploiting the shared category information between video and text, the module achieves fine-grained alignment and complementarity, producing fused features that are more discriminative. As shown in Fig. 1, our framework integrates MLTC-Attention and VTCA-Fusion for video action detection. To summarize, the main contributions of this paper are as follows:

– MLTC-Attention, which compresses features in a multi-scale temporal latent space and adaptively extracts dynamic features across diferent temporal scales. It strikes a balance between global and local feature extraction, thereby enhancing the ability to eficiently process temporal features in video.

– VTCA-Fusion, which utilizes cross attention between video features and enhanced category text features at both the standard and the latent level. It overcomes the limitations of existing methods in fusing video and text features, and achieves fine-grained alignment and complementarity between the two modalities.

– The framework, built around the MLTC-Attention and VTCA-Fusion modules, is applicable to various video action detection tasks. Experimental validation on four benchmarks demonstrates the efectiveness of the proposed modules.

## 2 Related Work

## 2.1 Video Understanding Models

Video representation learning has progressed from 3D CNNs such as C3D [46] and I3D [3], to Transformers such as ViViT [1] and Video Swin Transformer [27], self-supervised VideoMAE [48] and InternVideo [51], and SSM-based Video-Mamba [19]. On the multimodal side, CLIP [40], ImageBind [13], and Language-Bind [68] align vision with language and other modalities, Video-LLaVA [22] and InternVL [8] connect video encoders with large language models, and M2- CLIP [49] adapts multimodal models to video action recognition.

## 2.2 Video Action Detection Models

Temporal action detection has evolved from boundary-matching methods like BMN [23] and graph-based G-TAD [56] to Transformer-based approaches such as ActionFormer [64]. TriDet [42] employs multi-scale feature pyramids, TemporalMaxer [44] uses max-pooling, ActionMamba [4] and eficient temporal attention with state space models [36, 39] leverage SSM [14] for global temporal modeling, and recent works [12,20] explore cross-layer decoupling and boundary refinement.

## 2.3 Video Anomaly Detection Models

MIL-based methods [43] learn anomaly scores from video-level labels and were later improved with graph convolutions [66], feature magnitude learning [45], multi-sequence Transformers [21], and dual memory units [67]. Recent methods transfer CLIP [40] knowledge to weakly supervised anomaly detection [33,53,62], use text prompts for normality guidance [59] or MIL enhancement [5], and model short and long anomalies across temporal scales [31].

![](images/330bde81bc96dad125b9e91b521ac7daff3d99502a8aea63a5c267522172f4d8.jpg)  
Fig. 2: Overview of the overall architecture of our proposed video action detection framework. Our method integrates the VTCA-Fusion module and MLTC-Attention Blocks to process temporal and categorical features, with the resulting predictions feeding into various downstream tasks.

## 3 Method

## 3.1 Overview

Figure 2 shows the overall framework. Given a video of $\tau$ frames, the pretrained video encoder $f _ { \mathcal { V } } \left( \cdot \right)$ of the multimodal model extracts frame features $\mathcal { V } = [ v _ { 1 } , \cdots , v _ { \tau } ] \in \mathbb { R } ^ { \tau \times d }$ . For each of the c categories, the category text $t _ { c }$ is concatenated with the learnable prompt $t _ { L }$ , the Enhanced Context, and encoded by the pre-trained text encoder $f _ { \mathcal { T } } \left( \cdot \right) ;$ the resulting text features $\mathcal { T } \in \mathbb { R } ^ { c \times d }$ are projected into the video feature space. The VTCA-Fusion module fuses $\nu$ and $\tau$ through standard and latent space cross attention, producing the fused sequence $\hat { \mathcal { V } } = [ \bar { \hat { v } } _ { 1 } , \cdots , \hat { v } _ { \tau } ] \in \mathbb { R } ^ { \tau \times d }$ with the temporal resolution preserved, together with a video-text similarity map $S _ { \mathcal { V } \times \mathcal { T } } \in \mathbb { R } ^ { \tau \times c }$ . The MLTC-Attention module then compresses and emphasizes critical temporal information of $\hat { \mathcal { V } }$ in the latent space via multi-scale compression attention, outputting temporal features $O _ { \nu } \in \mathbb { R } ^ { \tau }$ ×d that capture both fine-grained details and long-range temporal context. Integrating $O _ { \nu }$ with $S _ { \mathcal { V } \times \mathcal { T } }$ yields the frame-level temporal scores $\boldsymbol { P } _ { t } \in \mathbb { R } ^ { \tau }$ and the class scores $\boldsymbol { P } _ { c l s } \in \mathbb { R } ^ { \tau \times c }$ , which feed the task-specific loss functions.

## 3.2 Video-Text Cross Attention Fusion Module

Standard cross-modal fusion operates at a single representation level, which may overfit to surface-level correlations while missing deeper semantic structures. To overcome this, the VTCA-Fusion module performs cross attention at both standard and latent feature levels simultaneously: the standard-level branch preserves full-dimensional semantic details for explicit alignment, while the latent-level branch operates in a compressed space that encourages the model to learn more abstract cross-modal representations. Its architecture is illustrated in the bottom-left part of Figure 2. We utilize the cross attention, where video frame features attend to text prompt features. Specifically, the video features $\mathcal { V } \in \mathbb { R } ^ { \tau \times d }$ are treated as queries, while the enhanced text prompt embeddings $\mathcal { T } \in \mathbb { R } ^ { c \times d }$ serve as both keys and values within a multi-head attention. We denote $\pmb { Q } _ { \mathcal { V } } \in \mathbb { R } ^ { \tau \times d } , \pmb { K } _ { \mathcal { T } } \in \mathbb { R } ^ { c \times d } , \pmb { V } _ { \mathcal { T } } \in \mathbb { R } ^ { c \times d }$ represent the input for standard multihead attention, and $\pmb { Q } _ { \mathcal { V } } ^ { l } \in \mathbb { R } ^ { \tau \times d _ { l } } , \pmb { K } _ { \mathcal { T } } ^ { l } \in \bar { \mathbb { R } } ^ { c \times d _ { l } } , \pmb { V } _ { \mathcal { T } } ^ { l } \in \bar { \mathbb { R } } ^ { \bar { c } \times d _ { l } }$ represent the input for attention of latent space features from the Latent Projection, $d _ { l } < d \in \mathbf { \hat { Z } } ^ { + }$ denotes the output feature dimension in the latent space, which reduces the dimension to compress the feature information, thereby enabling the interaction between video and enhanced text features within the latent space. The attention outputs are computed separately, as follows:

$$
A t t n \gamma _ { \times T } = \mathrm { s o f t m a x } \bigg ( \frac { Q _ { \gamma } K _ { T } ^ { \top } } { \sqrt { d } } \bigg ) V _ { \mathcal { T } } , \quad A t t n _ { \nu \times T } ^ { l } = \mathrm { s o f t m a x } \bigg ( \frac { Q _ { \nu } ^ { l } K _ { T } ^ { l ^ { \top } } } { \sqrt { d _ { l } } } \bigg ) V _ { \mathcal { T } } ^ { l } ,\tag{1}
$$

In Eq. 1, $A t t n _ { \mathcal { V } \times T } \in \mathbb { R } ^ { \tau \times d }$ represents the result of standard multi-head attention, where $Q _ { \nu } K _ { \tau } ^ { \mathrm { 7 } } \mathrm { y i e l d s }$ the similarity $S _ { \nu \times \tau }$ between the video and enhanced text features, and $\mathbf { \sigma } _ { A t t n _ { \mathcal { V } \times T } ^ { l } } \in \mathbb { R } ^ { \tau \times d _ { l } }$ represents the computation result of multihead attention in latent space. Finally, we construct an output projection weight tensor $W _ { o } \in \mathbb { R } ^ { ( d + d _ { l } ) \times d }$ to fuse the outputs of the standard attention and latent space attention, the final result of VTCA-Fusion module is computed, as follows:

$$
\hat { \mathcal { V } } = \mathcal { V } + W _ { o } ( A t t n _ { \mathcal { V } \times \mathcal { T } } | | A t t n _ { \mathcal { V } \times \mathcal { T } } ^ { l } ) .\tag{2}
$$

where ∥ denotes the operation of concatenating tensors. Through this computation, we obtain the temporal feature $\hat { \mathcal { V } } \in \mathbb { R } ^ { \tau \times d }$ that fuses the video and text features.

## 3.3 Multi-Scale Latent Temporal Compression Attention Block

Standard self-attention [47] over the full temporal sequence incurs $O ( \tau ^ { 2 } )$ complexity, while pooling-based methods [44] reduce cost but irreversibly discard fine-grained information. Window-based attention [27] restricts the receptive field and cannot model long-range dependencies. Our MLTC addresses this by first projecting features into a lower-dimensional latent space to reduce per-token cost, then compressing the temporal dimension at multiple scales so that larger scales capture global context while smaller scales preserve local dynamics. Attending to the compressed keys and values reduces the cost of each compression branch from $O ( \tau ^ { 2 } )$ to $O ( \tau ^ { 2 } / \sigma _ { i } )$ , while a single global self-attention branch is retained to preserve the full receptive field. The architecture of the MLTC is shown in the bottom-center and bottom-right parts of Figure 2. The overall input to the module is the temporal feature $\boldsymbol { \mathcal { X } } _ { \mathcal { V } } \in \mathbb { R } ^ { \tau \times d }$ . First, the Latent Projection maps $\mathcal { X } _ { \mathcal { V } }$ into the latent space, giving $Q _ { t } ^ { l } , K _ { t } ^ { l } , V _ { t } ^ { l } \in \mathbb { R } ^ { \tau \times \frac { d } { 2 } }$ as the inputs of the multiscale latent compression attention. In the bottom-right part of Figure 2, the scales set $[ s c a l e { - } 1 , \cdot \cdot \cdot$ , scale-n] denote the compression sizes along the temporal dimension, with each scale corresponding to a Compression Attention module.

The total temporal dimension of the Compression Attention input is $\tau ; K _ { t } ^ { l }$ and $\mathbf { } V _ { \mathrm { ~ } t } ^ { l }$ are each transformed along the temporal dimension by their own Compress Network of scale-i, as follows:

$$
\mathbf { { K } } _ { t _ { i } } ^ { l } = { W } _ { i , K } ^ { o u t } \mathrm { { G E L U } } ( { W } _ { i , K } ^ { i n } { K } _ { t } ^ { l } ) , \quad { V } _ { t _ { i } } ^ { l } = { W } _ { i , V } ^ { o u t } \mathrm { { G E L U } } ( { W } _ { i , V } ^ { i n } { V } _ { t } ^ { l } ) ,\tag{3}
$$

where $W _ { i , * } ^ { i n } \in \mathbb { R } ^ { \frac { \tau } { \sigma _ { i } } \times \tau } , W _ { i , * } ^ { o u t } \in \mathbb { R } ^ { \frac { \tau } { \sigma _ { i } } \times \frac { \tau } { \sigma _ { i } } } , * \in \{ K , V \}$ , denote the input and output weights of the two Compress Networks, which compress the temporal dimension from $\tau$ to $\textstyle { \frac { \tau } { \sigma _ { i } } } ; i$ indexes the scales set [scale-1, · · · , scale-n], $\sigma _ { i } ~ \in ~ \mathbb { Z } ^ { + }$ is the compression size of scale-i, and GELU is the activation function. The outputs $K _ { t _ { i } } ^ { l } , V _ { t _ { i } } ^ { l } \in \mathbb { R } ^ { \frac { \tau } { \sigma _ { i } } \times \frac { d } { 2 } }$ are the inputs of the i-th Compression Attention module. In the Compression Attention module, the computation is first performed based on the input tensors $\boldsymbol { Q } _ { t } ^ { l }$ with the compressed $\hat { { K } } _ { t _ { i } } ^ { l }$ and $\boldsymbol { V } _ { t _ { i } } ^ { l }$ as follows:

$$
\mathcal { S } _ { i } = \mathrm { s o f t m a x } \left( \frac { Q _ { t } ^ { l } { K _ { t _ { i } } ^ { l } } ^ { \top } } { \sqrt { d / 2 } } \right) , \quad A t t { n } _ { i } ^ { c p } = \mathcal { S } _ { i } V _ { t _ { i } } ^ { l } ,\tag{4}
$$

where $\boldsymbol { S } _ { i } \in \mathbb { R } ^ { \tau \times \frac { \tau } { \sigma _ { i } } }$ represents the similarity scores tensor between $\boldsymbol { Q } _ { t } ^ { l }$ and $\boldsymbol { K } _ { t _ { i } } ^ { l }$ $A t t n _ { i } ^ { c p } \in \mathbb { R } ^ { \tau \times \frac { d } { 2 } }$ represents the computation result of the attention, $c p$ means the compressed $\boldsymbol { K } _ { t _ { i } } ^ { l } , \boldsymbol { V } _ { t _ { i } } ^ { l }$ . Secondly, to capture the critical temporal information from $\boldsymbol { K } _ { t _ { i } } ^ { l }$ and $\boldsymbol { V } _ { t _ { i } } ^ { l }$ , we perform top-k selection based on $S _ { i }$ , as follows:

$$
\begin{array} { r } { \mathcal { T } _ { i } = \mathbf { T o p - K } \left( S _ { i } ( t ) , k \right) , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { K _ { t _ { i } } ^ { \mathcal { T } _ { i } } V _ { t _ { i } } ^ { \mathcal { T } _ { i } } = \big \| _ { j \in \mathbb { Z } _ { i } } K _ { t } ^ { l } V _ { t } ^ { l } [ j \sigma _ { i } : ( j + 1 ) \sigma _ { i } ] , } \end{array}\tag{6}
$$

where $\mathcal { T } _ { i } \in \mathbb { R } ^ { k }$ denotes the indices of the top-k most relevant temporal segments in $\boldsymbol { K } _ { t _ { i } } ^ { l } , \boldsymbol { V } _ { t _ { i } } ^ { l }$ with $\boldsymbol { Q } _ { t } ^ { l }$ . Following the top-k selection strategy in [43], we set $k =$ ma $\mathrm { x } ( \mathrm { i } , \lfloor \frac { \tau } { \sigma _ { i } } / 8 \rfloor )$ , which selects the top $1 / 8$ of the compressed blocks at each scale with a minimum of 1. The $\pmb { K } _ { t _ { i } } ^ { \mathcal { Z } _ { i } } \in \mathbb { R } ^ { \sigma _ { i } k \times \frac { d } { 2 } } , \pmb { V } _ { t _ { i } } ^ { \mathcal { Z } _ { i } } \in \mathbb { R } ^ { \sigma _ { i } k \times \frac { d } { 2 } }$ represent the key and value pairs retrieved via $\mathcal { T } _ { i } ,$ and they together with $\boldsymbol { Q } _ { t } ^ { l }$ , form the input tensors for computing the attention of critical temporal segments, as follows:

$$
{ A t t n } _ { i } ^ { t k } = \mathrm { s o f t m a x } \left( \frac { Q _ { t } ^ { l } K _ { t _ { i } } ^ { \mathcal { T } _ { i } ^ { \top } } } { \sqrt { d / 2 } } \right) V _ { t _ { i } } ^ { \mathcal { T } _ { i } } ,\tag{7}
$$

where $A t t n _ { i } ^ { t k } \in \mathbb { R } ^ { \tau \times \frac { d } { 2 } }$ represents the attention result obtained from the top-k filtering method, enabling the model to enhance feature learning for critical temporal segments. The results $A t t n _ { i } ^ { c p } , A t t n _ { i } ^ { t k }$ obtained through the above methods are concatenated, as follows:

$$
\begin{array} { r } { { \cal A } t t n _ { i } = { \cal A } t t n _ { i } ^ { c p } \| { \cal A } t t n _ { i } ^ { t k } , } \end{array}\tag{8}
$$

where $A t t n _ { i } \ \in \ \mathbb { R } ^ { \tau \times d }$ is the complete i-th Compression Attention output result. We also apply a Global Attention $A t t n _ { g l o b a l } :$ , which performs standard selfattention [47] over the entire video sequence $\mathcal { X } _ { \mathcal { V } }$ to capture long-range temporal

dependencies and holistic context. Finally, we introduce a gate network with weights applied to each output to fuse Compression Attention across various scales and Global Attention, with the process described as follows:

$$
O _ { \mathcal { V } } = W _ { g a t e } \odot ( A t t n _ { g l o b a l } \| \underbrace { A t t n _ { 1 } \| \cdots \| A t t n _ { n } } _ { n } ) ,\tag{9}
$$

where $W _ { g a t e } \in \mathbb { R } ^ { \tau \times ( n + 1 ) }$ denotes the gating weights used to allocate weights to each output, and $O _ { \nu } \in \mathbb { R } ^ { \tau \times d }$ indicates the fused output. It is further processed by a standard LayerNorm and FFN layer to obtain the final representation of the MLTC-Attention module. Based on temporal feature output $O _ { \nu }$ and the Video-Text Similarity map $S _ { \mathcal { V } \times \mathcal { T } }$ , the post-processing step is performed to generate the final temporal and categorical predictions, the $\boldsymbol { P } _ { t } \in \mathbb { R } ^ { \tau }$ represents the temporal prediction scores along the temporal dimension, and the $\ b { P _ { c l s } } \in \mathbb { R } ^ { \tau \times c }$ provides the class-level scores across time.

## 3.4 Loss Functions

Loss of Video Anomaly Detection. Following the standard weakly supervised setting [43, 53, 67], only video-level annotations are available for training: given the video-level label $Y \in \{ 0 , 1 \}$ , 0 for normal and 1 for abnormal, we broadcast it along the temporal dimension as $y ^ { i } = Y$ for $i = 1 , \cdots , \tau .$ so that no frame-level annotation is used. A binary cross-entropy loss $\mathcal { L } _ { B C E } ^ { a - n }$ is computed between the predicted scores $P _ { t }$ and $\mathbf { \pmb { y } } .$ Since an abnormal video still contains normal frames, this broadcast label is inevitably noisy for abnormal videos; the top-k lowest-scoring frames of an abnormal video are therefore additionally selected as normal instances $P _ { k - n } \in \mathbb { R } ^ { k }$ with loss $\mathcal { L } _ { B C E } ^ { k - n }$ , which lets the model learn normal features from abnormal videos and mitigates the label noise introduced by the broadcast. Following MIL [43], the mean of the top-k scores of each category in $P _ { c l s }$ along the temporal dimension gives the predicted logits $\tilde { P } _ { c l s } \in \mathbb { R } ^ { c }$ , trained with a multi-class cross-entropy $\mathcal { L } _ { C E } ^ { c l s }$ against the category labels $\ b { y } _ { c l s } \in \mathbb { R } ^ { c }$

$$
\begin{array} { r l } & { \mathcal { L } _ { B C E } ^ { a - n } = \displaystyle - \frac { 1 } { \tau } \sum _ { i = 1 } ^ { \tau } [ { \pmb y } ^ { i } \log ( { \pmb P } _ { t } ^ { i } ) + ( 1 - { \pmb y } ^ { i } ) \log ( 1 - { \pmb P } _ { t } ^ { i } ) ] , \quad \mathcal { L } _ { B C E } ^ { k - n } = \displaystyle - \frac { 1 } { k } \sum _ { i \in k } \log ( 1 - { \pmb P } _ { k - n } ^ { i } ) , } \\ & { \mathcal { L } _ { C E } ^ { c l s } = \displaystyle - \sum _ { i = 1 } ^ { c } { \pmb y } _ { c l s } ^ { i } \log ( \tilde { \pmb P } _ { c l s } ^ { i } ) , \quad \mathcal { L } _ { a - n } = \mathcal { L } _ { B C E } ^ { a - n } + \mathcal { L } _ { B C E } ^ { k - n } + \mathcal { L } _ { C E } ^ { c l s } , } \end{array}\tag{10}
$$

where the total loss $\mathcal { L } _ { a - n }$ preserves both the binary anomaly classification and the multi-category loss commonly used in MIL.

Loss of Temporal Action Detection. We use Focal Loss [24] for action classification, $\mathcal { L } _ { c l s }$ , and Distance-IoU [65] for temporal segment regression, $\mathcal { L } _ { i o u } \mathrm { : }$

$$
\mathcal { L } _ { a c t } = \frac { 1 } { \tau ^ { + } } \sum _ { i = 1 } ^ { \tau } [ \mathcal { L } _ { c l s } ( P _ { c l s } ^ { i } , \pmb { y } _ { c l s } ^ { i } ) + \lambda \mathcal { L } _ { i o u } ( P _ { t } ^ { i } , \pmb { y } _ { t } ^ { i } ) ] ,\tag{11}
$$

where $\pmb { y } _ { c l s } , \pmb { y } _ { t } \in \mathbb { R } ^ { \tau }$ are the labels for action categories and temporal segments, $\tau ^ { + }$ denotes the number of positive samples assigned during training, and Soft Non-Maximum Suppression is applied as post-processing at inference, and λ is the weight of $\mathcal { L } _ { i o u }$

## 4 Experiments

## 4.1 Datasets and Evaluation Metrics

For video anomaly detection we use UCF-Crime [43], over 1,900 surveillance videos covering 13 real-world abnormal events together with normal activities, and XD-Violence [52], about 4,700 videos from movies and social media covering 6 categories of abnormal events. For temporal action detection we use THUMOS14 [16], 413 untrimmed videos with temporal annotations over 20 action categories, and ActivityNet [2], 19,994 videos with temporal annotations over 200 action categories. Following common practice, we report the framelevel AUC of the ROC curve on UCF-Crime and the frame-level Average Precision (AP) on XD-Violence, which better reflects the precision–recall balance under class imbalance; for action detection we report the mean Average Precision (mAP) averaged over multiple temporal IoU (tIoU) thresholds between predicted and ground-truth segments.

## 4.2 Implementation Details

We implemented our method on a single NVIDIA L40S GPU. Both pre-trained encoders $f _ { \mathcal { V } } ( \cdot )$ and $f _ { T } ( \cdot )$ are frozen; only the projection layers, the learnable prompts, VTCA-Fusion, MLTC-Attention and the task head are trained, so all text-encoder variants share the same protocol and schedule. During the training process, we use the AdamW [28] optimizer and employ a scheduler that integrates linear warm-up and cosine annealing to adjust the learning rate, which is set to 1e-5. The batch size was set to 64. For video data input, we utilized pretrained encoder weights to extract video features, and the sequence length of video feature input is 512 frames. Each frame was resized to a resolution of 224 × 224 pixels. For the text input with the enhanced text prompt design, where each textual class label is embedded with 10 learnable prompt sequence length. The maximum text sequence length is set to 79, following the configuration of CLIP [40]. The model was trained for 20 epochs.

## 4.3 Evaluation and Comparison

Results of Video Anomaly Detection. Table 1 compares our method with previous state-of-the-art methods on UCF-Crime and XD-Violence. Our method achieves the highest AUC on UCF-Crime and, without incorporating any audio features, the highest AP on XD-Violence. Compared with recent LLM-based approaches, LAVAD [63], EventVAD [41], and VERA [60], our method obtains clearly higher accuracy on both datasets without relying on large language models, indicating that LLM-based approaches show no clear advantage on finegrained temporal localization while requiring substantially higher computational cost, whereas our targeted attention and fusion designs are more efective for fine-grained temporal anomaly detection.

Table 1: Comparison of frame-level performance between our Method and Previous methods on the UCF Crime and XD-Violence benchmarks. <sup>†</sup>: results reproduced by us with the listed vision encoder.
<table><tr><td>Method</td><td>Vision Encoder</td><td>UCF-Crime AUC(%)</td><td>XD-Violence AP(%)</td></tr><tr><td>LAVAD [63]</td><td>ImageBind [13]</td><td>80.28</td><td>62.01</td></tr><tr><td>DMU† [67]</td><td>I3D [3]</td><td>86.14</td><td>81.46</td></tr><tr><td>AnomalyCLIP [62]</td><td>ViT-B16 [11]</td><td>86.30</td><td>77.40</td></tr><tr><td>VERA [60]</td><td>InternVL2 [8]</td><td>86.55</td><td>70.54</td></tr><tr><td>UMIL [29]</td><td>ViT-B32 [11]</td><td>86.75</td><td></td></tr><tr><td>PE-MIL [5]</td><td>I3D [3]</td><td>86.83</td><td></td></tr><tr><td>MGFN [7]</td><td>VideoSwin [27]</td><td>86.97</td><td>79.31</td></tr><tr><td>OE-CTŠT [31]</td><td>I3D [3]</td><td>86.99</td><td>81.78</td></tr><tr><td>CLIP-TSA [17</td><td>ViT-B16 [11]</td><td></td><td>82.15</td></tr><tr><td>EventVAD [41]</td><td>ViT-B16 [11]</td><td>87.51</td><td>64.04</td></tr><tr><td>VadCLIP† [53]</td><td>I3D [3]</td><td>87.57</td><td>82.48</td></tr><tr><td>TPWNG [59]</td><td>ViT-B16 [11]</td><td>87.79</td><td>83.68</td></tr><tr><td>MLTC+VTCA+CLIP</td><td>I3D [3]</td><td>88.68</td><td>83.44</td></tr><tr><td>MLTC+VTCA+CLIP</td><td>ViT-B16 11</td><td>88.63</td><td>84.63</td></tr><tr><td>MLTC+VTCA+LangBind</td><td>ViT-B16 11]</td><td>89.17</td><td>84.89</td></tr><tr><td>MLTC+VTCA+DeBERTa</td><td>ViT-B16 11</td><td>89.84</td><td>86.53</td></tr></table>

Table 2: Comparison on the THUMOS14 and ActivityNet-1.3 with the mAP of tIoU. <sup>†</sup>: results reproduced by us with the listed vision encoder.
<table><tr><td rowspan="2">Model</td><td rowspan="2" colspan="2">Vision Encoder</td><td colspan="6">THUMOS14</td><td colspan="4">ActivityNet-1.3</td></tr><tr><td></td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg</td></tr><tr><td>ActionFormer [64]</td><td>I3D [3]</td><td></td><td>82.10</td><td>77.80</td><td>71.00</td><td>59.40</td><td>43.90</td><td>66.80</td><td>55.50</td><td>38.12</td><td>8.78</td><td>37.05</td></tr><tr><td>TemporalMaxer [44]</td><td>InternVidv2</td><td>[51]</td><td>85.23</td><td>81.27</td><td>74.17</td><td>63.70</td><td>49.14</td><td>70.70</td><td>60.99</td><td>40.53</td><td>9.79</td><td>38.59</td></tr><tr><td>TriDet [42]</td><td>InternVidv2</td><td>[51</td><td>85.81</td><td>81.78</td><td>75.61</td><td>64.64</td><td>50.50</td><td>71.67</td><td>56.40</td><td>38.74</td><td>9.60</td><td>37.60</td></tr><tr><td>Dyfadet [58]</td><td>VideoMAEv2</td><td>[48]</td><td>85.99</td><td>81.66</td><td>76.32</td><td>64.46</td><td>50.08</td><td>71.70</td><td>58.19</td><td>39.30</td><td>8.63</td><td>38.62</td></tr><tr><td>AdaTAD† [25]</td><td>InternVidv2</td><td>[51]</td><td>86.75</td><td>82.25</td><td>76.10</td><td>65.28</td><td>50.29</td><td>72.13</td><td>59.99</td><td>39.53</td><td>9.79</td><td>38.69</td></tr><tr><td>CLTDR-GMG [20]</td><td>InternVidv2</td><td>51</td><td>87.00</td><td>84.00</td><td>78.40</td><td>67.90</td><td>54.00</td><td>74.30</td><td>55.00</td><td>38.00</td><td>8.60</td><td>37.10</td></tr><tr><td>ActionMamba [4]</td><td>InternVidv2</td><td>51</td><td>87.05</td><td>82.66</td><td>76.51</td><td>65.56</td><td>49.87</td><td>72.33</td><td>62.20</td><td>43.14</td><td>9.57</td><td>41.73</td></tr><tr><td>BDRC-Net [12]</td><td>InternVidv2</td><td>51</td><td>88.60</td><td>83.80</td><td>78.00</td><td>66.80</td><td>51.20</td><td>73.70</td><td>61.40</td><td>42.10</td><td>10.20</td><td>41.10</td></tr><tr><td>DiGIT [18]</td><td>InternVidv2</td><td>51</td><td>87.60</td><td>84.20</td><td>77.60</td><td>67.30</td><td>52.50</td><td>73.80</td><td>62.00</td><td>43.10</td><td>11.30</td><td>42.00</td></tr><tr><td>MLTC+TCA+CLIP [40]</td><td>InternVidv2</td><td>51</td><td>88.95</td><td>85.12</td><td>78.40</td><td>68.02</td><td>52.51</td><td>74.60</td><td>63.81</td><td>45.34</td><td>10.92</td><td>43.59</td></tr><tr><td>MLTC+VTCA+LangBind [68]</td><td>InternVidv2</td><td>51</td><td>88.99</td><td>85.22</td><td>78.73</td><td>68.30</td><td>52.78</td><td>74.80</td><td>63.84</td><td>45.35</td><td>11.17</td><td>43.82</td></tr><tr><td>MLTC+VTCA+DeBERTa [15]</td><td>InternVidv2</td><td>51</td><td>89.50</td><td>85.38</td><td>78.82</td><td>69.05</td><td>54.15</td><td>75.37</td><td>63.87</td><td>45.43</td><td>11.36</td><td>43.90</td></tr></table>

Table 3: Ablation for module components and for the latent projection in each module.
<table><tr><td></td><td colspan="4">Module</td><td>UCF-C</td><td>THUMOS</td></tr><tr><td>MLTC</td><td>Top-k</td><td>VTCA</td><td>Latent (MLTC)</td><td>Latent (VTCA)</td><td>AUC</td><td>mAP</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>86.00</td><td>72.53</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>88.27</td><td>73.75</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>86.65</td><td>73.23</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>88.19</td><td>73.68</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>89.09</td><td>74.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>88.92</td><td>74.66</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>89.84</td><td>75.37</td></tr></table>

Table 4: Ablation results for alternative module design.
<table><tr><td>Attention Modules</td><td></td><td>UCF-C</td><td>THUMOS</td></tr><tr><td>Temporal Attn</td><td>V&amp;T Fusion</td><td>AUC</td><td>mAP</td></tr><tr><td>Self-Attn [47]</td><td>VTCA</td><td>86.00</td><td>72.53</td></tr><tr><td>Soft-Attn [57]</td><td>VTCA</td><td>86.14</td><td>72.64</td></tr><tr><td>MS-TCT [10]</td><td>VTCA</td><td>88.01</td><td>73.44</td></tr><tr><td>MLTC</td><td>Cross-Attn [32]</td><td>88.19</td><td>73.23</td></tr><tr><td>MLTC</td><td>VTCA</td><td>89.84</td><td>75.37</td></tr></table>

Results of Temporal Action Detection. Table 2 reports the results on THU-MOS14 and ActivityNet-1.3. On THUMOS14, our method achieves the best average mAP over tIoU thresholds from 0.3 to 0.7 and consistently maintains the best mAP across all thresholds; on ActivityNet-1.3, it also achieves the best overall performance and continues to lead at every tIoU threshold. Our method consistently outperforms both earlier Transformer- and SSM-based detectors and the most recent state-of-the-art methods on both datasets.

Computational Eficiency Analysis. Table 7 compares the computational cost and inference speed of our MLTC module with previous methods under the same input resolution and hardware settings, on a single NVIDIA L40S GPU with 512-frame inputs at 224 × 224 resolution, excluding I/O and preprocessing overhead. Our method achieves the highest throughput with the fewest FLOPs among all compared methods while also being the most accurate, reducing the computational cost compared with prior methods while simultaneously improving accuracy. By projecting features into a compact latent space and compressing temporal redundancy before the attention computation, our compress-then-attend paradigm achieves a better accuracy–eficiency trade-of than prior methods that increase complexity for marginal gains.

Table 5: Ablation for diferen temporal compression scales.
<table><tr><td rowspan="2">Scales</td><td>UCF-C</td><td>THUMOS</td></tr><tr><td>AUC</td><td>mAP</td></tr><tr><td>w/o</td><td>86.00</td><td>72.53</td></tr><tr><td>[4,16,32]</td><td>88.77</td><td>74.32</td></tr><tr><td>[8,16,64]</td><td>89.05</td><td>74.57</td></tr><tr><td>[8,32,64]</td><td>89.84</td><td>75.37</td></tr><tr><td>[16,64,128]</td><td>88.91</td><td>74.65</td></tr></table>

Table 6: Ablation for text prompt types.
<table><tr><td rowspan="2">Prompt Types</td><td>UCF-C</td><td>THUMOS</td></tr><tr><td>AUC</td><td>mAP</td></tr><tr><td>w/o prompt</td><td>86.65</td><td>73.23</td></tr><tr><td>Random</td><td>86.72</td><td>73.31</td></tr><tr><td>Generic</td><td>86.84</td><td>73.45</td></tr><tr><td>Class-Only</td><td>88.74</td><td>74.62</td></tr><tr><td>Learnable</td><td>89.84</td><td>75.37</td></tr></table>

Table 7: Eficiency comparison with prior methods.
<table><tr><td>Method</td><td>FPS</td><td>FLOPs</td></tr><tr><td>MGFN [7]</td><td>61.30</td><td>57.46G</td></tr><tr><td>AnomCLIP [62]</td><td>69.98</td><td>50.4G</td></tr><tr><td>STPrompt [54]</td><td>79.19</td><td>49.6G</td></tr><tr><td>VadCLIP [53]</td><td>78.34</td><td>44.8G</td></tr><tr><td>Ours</td><td>83.77</td><td>42.1G</td></tr></table>

![](images/b2cb6e24139cf7e6e6ba68628b8f428caa7fe0247440bb678dd848e84f93655b.jpg)

![](images/95ceb832a93ae9afe47427c3febb78d87bf9c0d532ae5909f877fe5feb6618fa.jpg)

![](images/f106e20e616c85c3f16112a935f2c7d94c11bcf3b3134bb2784112d2defa2746.jpg)

![](images/9a7252ef231bab134f02d5be000174dfcf62243734fb988579d3e71645e05ef5.jpg)

![](images/503fed3c8b7ed0231ef6ff7caed3ec82adb0ba78ed74d604b32d58bbdef0e483.jpg)

![](images/d365ab31a7e2dfc4ecf7a3af661d42b5f43f2f89757a126aee0728103f7024c4.jpg)

Fig. 3: Qualitative results of our proposed methods on Video Anomaly Detection tasks.  
![](images/6b16e1b11cd652f1501d681461de521f0b7307e3e4ff12435e8dce39bdebc3d2.jpg)  
Fig. 4: Qualitative results of our proposed methods on temporal action detection tasks.

## 4.4 Qualitative Analyses

We present visualization examples of temporal predictions produced by our model. Figure 3 shows Video Anomaly Detection results, where the curves are predicted anomaly scores over time and the red shaded regions are ground-truth (GT) anomaly intervals, and Figure 4 shows temporal action detection results, where colored segments are predicted action categories and temporal intervals; compared with the baseline, our method yields more precise temporal predictions on both tasks. Figure 5 visualizes the category-wise temporal attention: for anomaly videos the model correctly activates on the corresponding category during the GT interval, while for normal videos “Normal” dominates the response, reflecting the joint efect of MLTC-Attention, which captures fine-grained dynamics across multiple scales, and VTCA-Fusion, which aligns video features with category-specific textual semantics.

![](images/45f7a7ada27d81d1e0394ca0539a5a66af5f232791ac75cdd3f2e52c5ad30a70.jpg)  
Fig. 5: Visualization of internal representation category-wise temporal feature.

## 4.5 Ablation Studies

To better analyze the contribution of each component in our model, we conduct ablation studies on the representative benchmarks UCF-Crime [43] and THUMOS14 [16].

Contribution of Our Proposed Modules. As shown in Table 3, we ablate each module within the overall architecture and evaluate on representative datasets for each task. First, for the MLTC-Attention module, we replace it with standard attention. The results on both UCF-Crime and THUMOS14 datasets show a noticeable performance decline, demonstrating the superiority of MLTC-Attention in capturing temporal features. We also conducted experiments on the efect of the Top-k selection, which retains the top 1/8 of the compressed blocks at each scale, at least one block, following strategies in previous works [42]. Its removal led to a slight performance decrease on both datasets, further validating the efectiveness of this filtering mechanism in focusing on the most informative temporal segments. When the VTCA-Fusion module is removed, the correlation between video and text features weakens, resulting in a performance drop on both tasks. Finally, the latent component, which applies latent space attention within both the MLTC-Attention and VTCA-Fusion modules, is crucial for understanding key semantic features in latent space; its removal degrades performance significantly.

Alternative of Our Proposed Modules. We replace MLTC-Attention or VTCA-Fusion with several previously used temporal and cross-modal attention modules. As shown in Table 4, substituting MLTC with Self-Attention, Soft-Attention, or MS-TCT consistently lowers performance, indicating that multi-scale temporal compression is crucial for modeling long-range dependencies. Similarly, replacing VTCA with standard Cross-Attention also degrades results, demonstrating that our multi-level video-text fusion achieves more effective cross-modal alignment.

Multi-scale Compression factors. To analyze the efect of diferent temporal compression scales on the MLTC-Attention module, we conducted an ablation study by varying scales. As shown in the Table 5, the w/o denotes the absence of compression scales, corresponding to the standard attention. The smaller compression set [4, 16, 32] preserves fine-grained temporal details but introduces redundant computations, while the larger compression set [16, 64, 128] overly compresses the temporal dimension, leading to the loss of local dynamic characteristics. The configuration [8, 32, 64] achieves a better balance between feature compression and redundancy reduction. Intuitively, the larger scales 32 and 64 capture global anomaly regions across the temporal extent, while the smaller scale 8 refines local action boundaries, and their combination achieves sharper activation within ground-truth intervals.

Prompt Types. As shown in Table 6, removing textual input, denoted w/o in the table, clearly degrades performance, indicating that text provides useful semantic priors. Random and generic prompts introduce only weak guidance, as they lack category-aligned semantics. Using Class-Only prompts brings some class information but remains limited. In contrast, the Learnable prompt achieves the best results, showing that learnable prompt tokens efectively capture category-specific semantics and significantly improve multimodal feature learning.

## 5 Conclusion

We tackle key problems in multimodal video action detection by our novel solutions. First, we propose the MLTC-Attention module, which utilizes a multiscale temporal latent space compression attention. It is designed to adaptively capture important dynamic features across diferent temporal scales, efectively reducing the computational redundancy and ineficiency. Second, we present the VTCA-Fusion module, which employs cross attention between video content and enhanced multi-category text. This enables precise alignment and complementary integration of visual and textual features, addressing limitations in existing multimodal feature integration methods. Extensive experiments on temporal action detection and video anomaly detection benchmarks demonstrate that our methods consistently improve accuracy and achieve superior overall performance.

Acknowledgments. This work was supported by JST CRONOS JPMJCS24K4 and JSPS KAKENHI 26K23875.

Disclosure of Interests. The authors declare no competing interests.

## References

1. Arnab, A., Dehghani, M., Heigold, G., et al.: ViViT: A video vision transformer. In: ICCV. pp. 6836–6846 (2021)

2. Caba Heilbron, F., Escorcia, V., Ghanem, B., Carlos Niebles, J.: ActivityNet: A large-scale video benchmark for human activity understanding. In: CVPR. pp. 961–970 (2015)

3. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: CVPR. pp. 6299–6308 (2017)

4. Chen, G., Huang, Y., Xu, J., et al.: Video Mamba suite: State space model as a versatile alter native for video understanding. IJCV 134(1), 20 (2025)

5. Chen, J., Li, L., Su, L., et al.: Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection. In: CVPR. pp. 18319–18329 (2024)

6. Chen, L., Ma, N., Wang, P., et al.: Survey of pedestrian action recognition techniques for autonomous driving. Tsinghua Science and Technology 25(4), 458–470 (2020)

7. Chen, Y., Liu, Z., Zhang, B., et al.: MGFN: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection. In: AAAI. vol. 37, pp. 387–395 (2023)

9. Cheng, S., Fang, K., Yu, Y., et al.: VidEgoThink: Assessing egocentric video understanding capabilities for embodied ai. arXiv preprint arXiv:2410.11623 (2024)

10. Dai, R., Das, S., Kahatapitiya, K., et al.: MS-TCT: Multi-scale temporal convtransformer for action detection. In: CVPR. pp. 20041–20051 (2022)

11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2021)

12. Fang, Z., Yu, J., Hong, R.: Boundary discretization and reliable classification network for temporal action detection. IEEE TMM (2025)

13. Girdhar, R., El-Nouby, A., Liu, Z., et al.: ImageBind: One embedding space to bind them all. In: CVPR. pp. 15180–15190 (2023)

14. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: First Conference on Language Modeling (COLM) (2024)

15. He, P., Gao, J., Chen, W.: DeBERTaV3: Improving DeBERTa using electra-style pre-training with gradient-disentangled embedding sharing. arXiv preprint arXiv:2111.09543 (2021)

16. Jiang, Y.G., Liu, J., Roshan Zamir, A., et al.: THUMOS challenge: Action recognition with a large number of classes (2014)

17. Joo, H.K., Vo, K., Yamazaki, K., Le, N.: CLIP-TSA: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In: 2023 IEEE International Conference on Image Processing (ICIP). pp. 3230–3234 (2023)

18. Kim, H.J., Lee, Y., Hong, J.H., Lee, S.W.: DiGIT: Multi-dilated gated encoder and centraladjacent region integrated decoder for temporal action detection transformer. In: CVPR. pp. 24286–24296 (2025)

19. Li, K., Li, X., Wang, Y., et al.: VideoMamba: State space model for eficient video understanding. In: ECCV. pp. 237–255 (2024)

20. Li, Q., Liu, D., Kong, J., et al.: Temporal action localization with cross layer task decoupling and refinement. In: AAAI. vol. 39, pp. 4878–4886 (2025)

21. Li, S., Liu, F., Jiao, L.: Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection. In: AAAI. vol. 36, pp. 1395–1403 (2022)

22. Lin, B., Ye, Y., Zhu, B., et al.: Video-LLaVA: Learning united visual representation by alignment before projection. In: EMNLP. pp. 5971–5984 (2024)

23. Lin, T., Liu, X., Li, X., et al.: BMN: Boundary-matching network for temporal action proposal generation. In: ICCV. pp. 3889–3898 (2019)

24. Lin, T.Y., Goyal, P., Girshick, R., et al.: Focal loss for dense object detection. In: ICCV. pp. 2980–2988 (2017)

25. Liu, S., Zhang, C.L., Zhao, C., Ghanem, B.: End-to-end temporal action detection with 1B parameters across 1000 frames. In: CVPR. pp. 18591–18601 (2024)

26. Liu, Y., Ju, B., Yang, D., et al.: Memory-enhanced spatial-temporal encoding framework for industrial anomaly detection system. Expert Systems with Applications 250, 123718 (2024)

27. Liu, Z., Ning, J., Cao, Y., et al.: Video Swin transformer. In: CVPR. pp. 3202–3211 (2022)

28. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

29. Lv, H., Yue, Z., Sun, Q., et al.: Unbiased multiple instance learning for weakly supervised video anomaly detection. In: CVPR. pp. 8022–8031 (2023)

30. Ma, Y., Yuan, L., Abdelraouf, A., et al.: M2DAR: Multi-view multi-scale driver action recognition with vision transformer. In: CVPR. pp. 5286–5293 (2023)

31. Majhi, S., Dai, R., Kong, Q., et al.: OE-CTST: Outlier-embedded cross temporal scale transformer for weakly-supervised video anomaly detection. In: WACV. pp. 3218–3228 (2024)

32. Ouenniche, K., Tapu, R., Zaharia, T.: Vision-text cross-modal fusion for accurate video captioning. IEEE Access 11, 115477–115492 (2023)

33. Pu, Y., Wu, X., Yang, L., Wang, S.: Learning prompt-enhanced context features for weaklysupervised video anomaly detection. IEEE TIP (2024)

34. Qiu, Y., Niu, L., Sha, F.: Multipath 3D-Conv encoder and temporal-sequence decision for repetitive-action counting. Expert Systems with Applications 249, 123760 (2024)

35. Qiu, Y., Sha, F., Niu, L.: Dka-yolo: Enhanced small object detection via dilation kernel aggregation convolution modules. IEEE Access 12, 187353–187366 (2024)

36. Qiu, Y., Sha, F., Niu, L.: Eficient temporal attention with state space model for temporal action localization. In: ICONIP. pp. 183–197 (2024)

37. Qiu, Y., Sha, F., Niu, L., Zhang, G.: Fire anomaly detection based on low-rank adaption finetuning and localization using gradient filtering. Applied Soft Computing p. 112782 (2025)

38. Qiu, Y., Yanai, K.: DST-Mamba: Spatial–temporal adapter with boundary-aware Mamba for video temporal action detection. Big Data and Cognitive Computing 10(7), 231 (2026)

39. Qiu, Y., Yanai, K.: Eficient spatial-temporal focal adapter with SSM for temporal action detection. arXiv preprint arXiv:2604.09164 (2026)

40. Radford, A., Kim, J.W., Hallacy, C., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)

41. Shao, Y., He, H., Li, S., et al.: EventVAD: Training-free event-aware video anomaly detection. In: ACM MM. pp. 2586–2595 (2025)

42. Shi, D., Zhong, Y., Cao, Q., et al.: TriDet: Temporal action detection with relative boundary modeling. In: CVPR. pp. 18857–18866 (2023)

43. Sultani, W., Chen, C., Shah, M.: Real-world anomaly detection in surveillance videos. In: CVPR. pp. 6479–6488 (2018)

44. Tang, T.N., Kim, K., Sohn, K.: TemporalMaxer: Maximize temporal context with only max pooling for temporal action localization. arXiv preprint arXiv:2303.09055 (2023)

45. Tian, Y., Pang, G., Chen, Y., et al.: Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In: ICCV. pp. 4975–4986 (2021)

46. Tran, D., Bourdev, L., Fergus, R., et al.: Learning spatiotemporal features with 3D convolutional networks. In: ICCV. pp. 4489–4497 (2015)

47. Vaswani, A., Shazeer, N., Parmar, N., et al.: Attention is all you need. NeurIPS 30 (2017)

48. Wang, L., Huang, B., Zhao, Z., et al.: VideoMAE v2: Scaling video masked autoencoders with dual masking. In: CVPR. pp. 14549–14560 (2023)

49. Wang, M., Xing, J., Jiang, B., et al.: A multimodal, multi-task adapting framework for video action recognition. In: AAAI. vol. 38, pp. 5517–5525 (2024)

50. Wang, S., Li, B.Z., Khabsa, M., Fang, H., Ma, H.: Linformer: Self-attention with linear com plexity. arXiv preprint arXiv:2006.04768 (2020)

51. Wang, Y., Li, K., Li, X., et al.: InternVideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377 (2024)

52. Wu, P., Liu, J., Shi, Y., et al.: Not only look, but also listen: Learning multimodal violence detection under weak supervision. In: ECCV. pp. 322–339 (2020)

53. Wu, P., Zhou, X., Pang, G., et al.: VadCLIP: Adapting vision-language models for weakly su pervised video anomaly detection. In: AAAI. vol. 38, pp. 6074–6082 (2024)

54. Wu, P., Zhou, X., Pang, G., et al.: Weakly supervised video anomaly detection and localization with spatio-temporal prompts. In: ACM MM. pp. 9301–9310 (2024)

55. Xarles, A., Escalera, S., Moeslund, T.B., Clapés, A.: ASTRA: An action spotting transformer for soccer videos. In: Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports. pp. 93–102 (2023)

56. Xu, M., Zhao, C., Rojas, D.S., et al.: G-TAD: Sub-graph localization for temporal action detection. In: CVPR. pp. 10156–10165 (2020)

57. Yan, S., Smith, J.S., Lu, W., Zhang, B.: Hierarchical multi-scale attention networks for action recognition. Signal Processing: Image Communication 61, 73–84 (2018)

58. Yang, L., Zheng, Z., Han, Y., et al.: DyFADet: Dynamic feature aggregation for temporal action detection. In: ECCV. pp. 305–322 (2024)

59. Yang, Z., Liu, J., Wu, P.: Text prompt with normality guidance for weakly supervised video anomaly detection. In: CVPR. pp. 18899–18908 (2024)

60. Ye, M., Liu, W., He, P.: VERA: Explainable video anomaly detection via verbalized learning of vision-language models. In: CVPR. pp. 8679–8688 (2025)

61. Yuan, J., Gao, H., Dai, D., et al.: Native sparse attention: Hardware-aligned and natively trainable sparse attention. In: ACL. pp. 4644–4664 (2025)

62. Zanella, L., Liberatori, B., Menapace, W., et al.: Delving into CLIP latent space for video anomaly recognition. Computer Vision and Image Understanding 249, 104163 (2024)

63. Zanella, L., Menapace, W., Mancini, M., et al.: Harnessing large language models for trainingfree video anomaly detection. In: CVPR. pp. 18527–18536 (2024)

64. Zhang, C.L., Wu, J., Li, Y.: ActionFormer: Localizing moments of actions with transformers. In: ECCV. pp. 492–510 (2022)

65. Zheng, Z., Wang, P., Liu, W., et al.: Distance-IoU loss: Faster and better learning for bounding box regression. In: AAAI. vol. 34, pp. 12993–13000 (2020)

66. Zhong, J.X., Li, N., Kong, W., et al.: Graph convolutional label noise cleaner: Train a plug-andplay action classifier for anomaly detection. In: CVPR. pp. 1237–1246 (2019)

67. Zhou, H., Yu, J., Yang, W.: Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In: AAAI. vol. 37, pp. 3769–3777 (2023)

68. Zhu, B., Lin, B., Ning, M., et al.: LanguageBind: Extending video-language pretraining to nmodality by language-based semantic alignment. arXiv preprint arXiv:2310.01852 (2023)