# SMART: MLLM-guided Temporal Alignment for Unifying Sign Language Recognition and Spotting

Eunjee Choi<sup>∗</sup> 6eun09ji@dankook.ac.kr

Dankook University Yongin, Korea

<sup>2</sup>JungHoon Sung<sup>∗</sup> jason\_dku77@dankook.ac.kr

Seongwhan Cho uchosw0812@dankook.ac.kr

AChu Xin chuxin@dankook.ac.kr

2Younggeun Choi younggch@dankook.ac.kr

## Abstract

Continuous sign language recognition (CSLR) aims to recognize gloss sequences from unsegmented sign videos under weak sequence-level supervision. However, existing methods rely on sentence-level gloss annotations, providing limited temporal and semantic guidance for fine-grained representation learning. Conventional video-text alignment also requires large batch sizes, making it inefficient for memory-intensive sign language video training. In this work, we propose SMART, an MLLM-guided temporal alignment framework for joint sign recognition and spotting. SMART uses MLLMgenerated motion descriptions as auxiliary semantic cues and performs stable videotext alignment under small-batch training. To improve temporal representation learning, we introduce a Multi-Scale Temporal Adapter that models temporal interactions during transformer encoding. For dense temporal localization, SMART incorporates CSFormer, a CSLR-guided spotting module that injects recognition-derived gloss evidence into a boundary-aware spotting network. This unified framework enables CSLR features to benefit spotting, while spotting supervision complements weak CTC-based recognition. Experiments on four sign language benchmarks, including PHOENIX14-T, CSL-Daily, Large-scale KSL, and Disaster and Safety KSL datasets, demonstrate the effectiveness of SMART across both recognition and spotting tasks.

## 1 Introduction

Sign language is a main communication medium for deaf and hard-of-hearing people. Developing effective sign language understanding systems is important for reducing communication barriers between signers and non-signers. Sign language understanding includes several tasks, such as sign language recognition, sign language translation, and sign spotting. In this work, we focus on Continuous Sign Language Recognition (CSLR) and sign spotting, which aim to recognize gloss sequences and localize signs in continuous videos, respectively.

![](images/90e5941a78445c19d2a0daf757153c130b77a0adf12021451a9983ec107a2eeb.jpg)  
Figure 1: SMART unifies sign language recognition and spotting through complementary recognition and localization supervision. The spotting branch refines weak CTC-based temporal alignments with dense temporal supervision, while the CSLR branch provides recognition-derived gloss evidence for spotting.

CSLR recognizes gloss sequences from unsegmented sign videos using only sentencelevel gloss annotations. Since frame-level gloss annotations are not available in CSLR, most CSLR methods rely on Connectionist Temporal Classification (CTC) loss [6, 8, 9, 15]. Although CTC enables training without frame-level labels, it often produces peaky alignments [9, 35, 40], where each gloss is predicted by only a few frames while most frames are assigned to blank class. This weak temporal supervision limits dense frame-level representation learning and makes it difficult to capture fine-grained sign dynamics. Sign spotting can complement this limitation because it requires frame-level localization of target signs. As shown in Fig. 1, we integrate CSLR and spotting into a unified framework. We introduce CSFormer, a boundary-aware temporal segmentation backbone adapted for sign spotting, which refines temporal boundaries and provides dense frame-level predictions. Prior sign spotting methods have mainly been formulated as retrieval or localization tasks using short query templates or external sign dictionaries [28, 37, 42, 44, 45, 46, 47], or have relied on sparsely annotated large-scale resources such as BSL-1K [3]. Thus, dense frame-level sign spotting with large vocabularies and accurate boundary localization remains underexplored. To address this, we introduce a CSLR-aware injection mechanism that explicitly feeds gloss probabilities learned by the recognition module into the spotting network to guide temporal boundary estimation. Consequently, we demonstrate that CSLR features are highly effective for spotting segmentation, and conversely, the dense temporal supervision from the spotting module complements weak CTC-based recognition.

Recent CSLR methods have improved visual representation learning by adapting largescale vision-language models. In particular, Adaptsign [25] adopts a frozen CLIP [41] backbone with lightweight adaptation module, showing that pretrained CLIP features can be efficiently transferred to CSLR. However, existing CLIP-based CSLR methods use CLIP as a visual feature extractor and still rely on CTC-based gloss supervision, without using language-based semantic guidance. To address these limitations, we use MLLM-generated motion descriptions as auxiliary semantic cues. Since sign language is expressed through hand movements, facial expressions, and body motions, these descriptions provide motionaware language guidance for representation learning [30]. We align video and text representations using a SigLIP [50], which enables stable video-text alignment under small-batch video training. To further improve temporal representation learning, we introduce a Multi-Scale Temporal Adapter (MSTA) inside the CLIP backbone. Although the CLIP image encoder provides image representations, it does not model temporal dependencies across frames. MSTA addresses this limitation by capturing multi-scale temporal interactions during transformer encoding while preserving the pretrained CLIP representations. In addition, we incorporate sign spotting as dense temporal supervision to complement weak CTC-based alignments. Therefore, we introduce CSFormer, a CSLR-aware spotting module that injects gloss evidence from the recognition branch into a boundary-aware temporal segmentation backbone, enabling fine-grained boundary estimation for sign spotting. Based on these reasons, we propose SMART, an MLLM-guided Temporal Alignment framework for CSLR and spotting. Our contributions are summarized as follows:

• We propose SMART, an MLLM-guided temporal alignment framework for CSLR and spotting. SMART uses MLLM-generated motion descriptions as auxiliary semantic cues and enables stable video-text alignment under small-batch training.

• A Multi-Scale Temporal Adapter is introduced to capture multi-scale temporal dependencies within the CLIP backbone for enhanced temporal representation learning.

• We jointly integrate sign spotting and CSLR via CSFormer, using recognition-derived gloss guidance for dense frame-level localization and late fusion to complement weak CTC-based recognition.

• We conduct experiments on four sign language benchmarks, including PHOENIX14- T, CSL-Daily, Large-scale KSL, and Disaster and Safety KSL. The two Korean sign language datasets are used for CSLR and sign spotting for the first time, and the results demonstrate the complementarity between gloss-level recognition guidance and dense temporal supervision across different sign languages.

## 2 Related Works

Continuous Sign Language Recognition. Continuous Sign Language Recognition (CSLR) aims to recognize gloss sequences from unsegmented sign videos, where only sentence-level gloss annotations are available during training [9, 32]. Early CSLR approaches mainly relied on hand-crafted visual features [13, 31, 43] or HMM-based recognition systems [14, 18, 39] for temporal modeling. With the development of deep learning, CTC loss [15] has been widely adopted for end-to-end CSLR [7, 8]. Most CTC-based methods consist of a spatial backbone for frame-wise feature extraction and a temporal module, such as a 1D CNN, LSTM, or transformer, for sequence modeling [6, 7, 8, 38]. Since CSLR datasets provide only sentence-level gloss annotations without frame-level gloss boundaries or temporal alignments, CTC-based methods receive only weak temporal supervision. As a result, CTC often produces peaky alignments [35], where each gloss is predicted by only a few frames while most frames are assigned to the blank class. Although effective for sentence-level recognition, such alignments provide limited supervision for modeling fine-grained temporal dynamics. Recent methods have advanced CSLR by improving visual representations [6], refining temporal modeling [38], or adapting pretrained vision-language models. In particular, AdaptSign [25] adopts a frozen CLIP [41] backbone and introduces lightweight adaptation modules to transfer pretrained visual representations to CSLR. This shows the potential of adapting large-scale pretrained vision-language models for CSLR. However, existing CSLR methods still rely on sequence-level gloss supervision, while the use of MLLM-generated language descriptions as auxiliary semantic cues for CSLR remains underexplored.

Peaky Alignment in CSLR. Due to the absence of ground-truth frame-level alignment, the Connectionist Temporal Classification (CTC) loss is widely adopted as the standard objective [5, 15]. As a side effect of this formulation, models typically suffer from the peaky alignment problem; they predict each gloss as a single time-step spike within the downsampled feature space while filling the remaining sequence with blank (background) tokens [35, 49]. Although this peakiness is not reflected in sentence-level Word Error Rate (WER) scores alone, it poses an inherent limitation for CTC-trained CSLR models [19, 36, 53]. Because downsampled feature predictions are restricted to a combination of blanks and isolated spikes, the model has difficulty capturing the full temporal structure of each sign. Furthermore, the lack of meaningful gradient flow to adjacent frames significantly prevents the model’s ability to capture the intrinsic temporal consistency of each sign. To overcome this limitation, we propose a framework that integrates CSFormer to provide dense, framelevel predictions. Consequently, we demonstrate that our approach alleviates the frame-level peaky limitation and further complements sentence-level recognition.

Action segmentation and Sign Spotting. Frame-level action segmentation is a representative task requiring dense temporal modeling, making it the field most closely related to sign spotting. MS-TCN [12] proposes a multi-stage temporal convolutional network architecture that progressively refines frame-wise class probabilities at each stage, which is subsequently advanced by MS-TCN++ [34] to enhance dilated information exchange across stages. Among transformer-based approaches, ASFormer [48] substantially boosts segmentation accuracy by combining sliding-window self-attention with a multi-stage decoder. Additionally, ASRF [26] introduces a refinement module equipped with a dedicated boundary regression head to post-process and refine the temporal boundaries of predicted segments. More recently, LTContext [4] effectively captures long-range dependencies in long-duration videos by disentangling windowed attention from long-term attention. However, these methods rely on visual features from pre-trained models, or optical flow as inputs, and the effectiveness of backbones designed for the sign language domain has yet to be investigated. Sign spotting is a recently introduced task that aims to predict the temporal intervals in which signs from a predefined vocabulary occur in a video. However, prior sign spotting research has primarily focused on retrieval-based formulations that rely on short query templates [28, 37, 42, 44, 45, 46, 47] or on sparsely annotated datasets such as BSL-1K [3]. Consequently, sign spotting formulated as a dense, frame-level recognition task under large vocabularies and abundant training data remains largely unexplored.

## 3 Methods

We propose SMART: MLLM-guided Temporal Alignment for Sign Recognition and Spotting. SMART consists of three key components: (1) an MLLM-guided video-text alignment module that aggregates frame-level motion descriptions into video-level semantic cues, (2) a video encoder enhanced by the proposed Multi-Scale Temporal Adapter (MSTA) for spatiotemporal feature extraction, and (3) a CSLR-aware sign spotting module, CSFormer, that injects recognition probabilities into a boundary-aware backbone for fine-grained temporal refinement. As illustrated in Fig. 2, these components are jointly modeled for Continuous Sign Language Recognition (CSLR) and sign spotting. Each component is described in Sec. 3.1–3.3.

![](images/9e4c2e15252b965192fb415b73fb83f6e27e53f3c9694427a9f7774c17f34ad0.jpg)  
Figure 2: Overview of the proposed SMART framework. Given an input sign video, the visual encoder extracts spatio-temporal representations enhanced by the proposed Multi-Scale Temporal Adapter. MLLM-generated motion descriptions provide auxiliary semantic cues through video-text alignment. In addition, the CSLR-guided sign spotting branch provides temporal supervision for boundary refinement, enabling joint optimization of CSLR and sign spotting.

## 3.1 MLLM Frame-Level Text Generation

Prior work [30] uses MLLM-generated sign descriptions for sign language translation. We extend this idea to CSLR by generating frame-level motion descriptions and using them as auxiliary semantic cues for video-text alignment. Before training, each sign video is processed with LLaVA-OneVision-7B [33] to generate descriptions focusing on the signer’s hand movements and facial expressions.

## 3.2 Sign Language Recognition

The SMART architecture consists of a CLIP ViT-B/16 visual encoder [11, 41], which is augmented with Spatial Adapters and Prefix embeddings [25], and the proposed MSTA. The extracted visual features are then fed into a sequence model composed of a 1D CNN and a two-layer bidirectional LSTM. The final features are passed to a NormLinear classifier and CTC decoder for gloss sequence prediction. For semantic supervision, video representations are aligned with pre-extracted BERT text embeddings via video-text alignment during training.

Visual Encoder. Given a sign language video with T frames $\mathbf { x } = \{ x _ { t } \} _ { t = 1 } ^ { T } \in \mathbb { R } ^ { T \times 3 \times 2 5 6 \times 2 5 6 }$ each frame is processed by a frozen CLIP ViT-B/16 image encoder. Spatial Adapters and

![](images/439351b198f03b4ab4fa117fb447c76be632d12509176615a24ffd1c7c61c63b.jpg)  
(a) Multi-Scale Temporal Adapter module.

![](images/b8b83dd38b00cf277fdd6008c114ea22f27735e28b2d31863eb18915360f2096.jpg)  
(b) CSFormer sign spotting module.  
Figure 3: Illustration of the proposed Multi-Scale Temporal Adapter and CSFormer sign spotting module.

Prefix embeddings [25] are incorporated into each transformer block for spatial adaptation to the sign language domain. However, since CLIP processes each frame independently, temporal dynamics across frames are not captured within the encoder. To address this, we propose the MSTA, inserted into the last 4 transformer blocks to model inter-frame temporal dependencies at the feature level. The features are then passed through a 1D CNN and a two-layer bidirectional LSTM, followed by a NormLinear Classifier and CTC decoder for gloss sequence prediction.

Multi-Scale Temporal Adapter. The architecture of MSTA is illustrated in Fig. 3(a). Formally, given visual token features $\mathbf { Z } \in \mathbb { R } ^ { L \times B T \times d }$ , where L, B, T, and d denote the number of visual tokens, batch size, number of frames, and feature dimension, respectively, MSTA first projects Z into a low-dimensional temporal adaptation space:

$$
\mathbf { h } = \phi \big ( \mathrm { L N } ( \mathbf { Z } \mathbf { W } _ { d o w n } ) \big ) ,\tag{1}
$$

where h $\in \mathbb { R } ^ { L \times B T \times d _ { m } }$ and $d _ { m } = d / r$ . The feature h is then reshaped along the temporal dimension, and multi-scale temporal operators are applied:

$$
\hat { \mathbf { h } } = \sum _ { k \in \{ 3 , 5 , 7 \} } \alpha _ { k } T _ { k } ( \mathbf { h } ) , \qquad \alpha = \mathrm { s o f t m a x } ( \theta ) ,\tag{2}
$$

where $\mathcal { T } _ { k } ( \cdot )$ denotes a temporal operator with kernel size k, and the fusion weights α are learned through a softmax-normalized learnable parameter vector $\theta \in \mathbb { R } ^ { 3 }$ . The multi-scale temporal operators capture motion patterns with different temporal receptive fields, enabling the model to encode both short- and long-range temporal dependencies.

The adapted temporal feature is projected back to the original feature dimension and added to the input through a residual connection:

$$
\begin{array} { r } { \mathbf { Z } ^ { \prime } = \mathbf { Z } + \hat { \mathbf { h } } \mathbf { W } _ { u p } . } \end{array}\tag{3}
$$

The up-projection $\mathbf { W } _ { u p }$ is zero-initialized to preserve the pretrained CLIP representations at the beginning of training.

Text Encoder. The generated frame-level motion descriptions are encoded using BERT [10]. For each frame description, the final-layer representation of the [CLS] token is extracted as a 768-dimensional text embedding, resulting in per-frame text embeddings $\mathbf { T } \in \mathbb { R } ^ { F \times 7 6 8 }$ where $F$ denotes the number of sampled frames. These embeddings are averaged across frames to produce a single global text embedding $\mathbf { t } \in \mathbb { R } ^ { 7 6 8 }$ for each video, which is used for video-text alignment during training.

Video-Text Alignment. Existing CSLR models rely solely on CTC loss, which provides only sentence-level supervision. MLLM-generated motion descriptions provide semantic information about hand movements and facial expressions, but directly fusing text features into the visual encoder risks degrading the pretrained CLIP representations and increases inference cost. Instead, we use text embeddings as auxiliary supervision through video-text alignment, which bridges the semantic gap between visual and textual modality during training. InfoNCE losses [41] rely on in-batch negative pairs, requiring large batch sizes to provide sufficient negative samples. This setting is inefficient for memory-intensive CSLR training, where batch sizes are small. We instead adopt SigLIP [50], which applies a sigmoid loss independently to each pair, enabling stable optimization in small batch settings.

Specifically, after the CLIP visual encoder extracts frame-wise features, we apply temporal pooling over frames to obtain a video-level representation, which is then projected to a 768-dimensional video embedding $\mathbf { v } \in \mathbb { R } ^ { 7 6 8 }$ via a video projector. The global text embedding $\mathbf { t } \in \mathbb { R } ^ { 7 6 8 }$ from Sec. 3.1 is projected via a text projector. Both embeddings are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ before alignment. The alignment loss is defined as:

$$
\mathcal { L } _ { \mathrm { A l i g n } } = - \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \log \sigma \big ( y _ { i j } \cdot s \cdot \mathbf { v } _ { i } ^ { \top } \mathbf { t } _ { j } \big ) ,\tag{4}
$$

where B denotes the batch size, $\mathbf { v } _ { i }$ and $\mathbf { t } _ { j }$ represent the normalized video and text embeddings of the i-th and j-th samples, respectively. The pair label $y _ { i j }$ is set to +1 for matched video-text pairs and −1 otherwise, $\sigma ( \cdot )$ is the sigmoid function, and s is a learnable logit scale.

Training Details. The total training loss combines CTC-based recognition losses, a knowledge distillation loss, and the proposed video-text alignment loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { s e q } } \mathcal { L } _ { \mathrm { S e q C T C } } + \lambda _ { \mathrm { c o n v } } \mathcal { L } _ { \mathrm { C o n v C T C } } + \lambda _ { \mathrm { k d } } \mathcal { L } _ { \mathrm { K D } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { A l i g n } } ,\tag{5}
$$

where $\mathcal { L } _ { \mathrm { C o n v C T C } }$ and $\mathcal { L } _ { \mathrm { S e q C T C } }$ are CTC losses applied to the 1D CNN and BiLSTM outputs respectively, and ${ \mathcal { L } } _ { \mathrm { K D } }$ is a knowledge distillation loss between the two predictions [25, 36]. The weights are set to $\lambda _ { \mathrm { s e q } } = 1 . 0 , \lambda _ { \mathrm { c o n v } } = 1 . 0 , \lambda _ { \mathrm { k d } } = 2 5 . 0$ , and $\lambda _ { \mathrm { a l i g n } } = 0 . 0 5$ . During inference, the alignment module is discarded, and recognition is performed using CTC beam search on the final BiLSTM logits to obtain the gloss sequence $\hat { \mathbf { y } } = \{ \hat { y } _ { n } \} _ { n = 1 } ^ { N }$ , where N denotes the number of predicted glosses.

## 3.3 Sign Language Spotting

We inject the gloss-sequence probabilities produced by the recognition module in Sec. 3.2 into our spotting model, so that recognition-derived gloss evidence can guide the temporal boundary estimation in spotting. Existing sign spotting models predict gloss boundaries from visual features, which limits precision on short or visually ambiguous signs. However, CSLR predictions provide sparse but discriminative cues as peaky CTC spikes around gloss positions, whereas the visual encoder learns temporally continuous cues. Therefore, a naive channel-wise concatenation followed by a shared encoder can cause the sparse CSLR signal to be averaged out at the input stage. Built on the backbone [48], our model introduces three key modules: (1) a CSLR-aware injection that projects the visual features and the CSLR sequence logits into two independent streams, (2) a layer-wise bidirectional cross-attention that exchanges information between the two streams at every encoder layer, and (3) a boundary head on every stage that predicts gloss transitions jointly with frame-level classification. The overall structure is illustrated in Fig. 3(b).

CSLR-aware injection. Given frame-wise visual features $\mathbf { F } \in \mathbb { R } ^ { T \times 5 1 2 }$ and the sequence logits $\mathbf { P } \in \mathbb { R } ^ { T _ { c } \times C }$ produced by the BiLSTM classifier of Sec. $3 . 2 \ : ( C )$ gloss vocabulary size), we first align P to length T via 1D linear interpolation to obtain $\bar { \mathbf { P } } \in \mathbb { R } ^ { T \times C }$ . Each modality is projected to the encoder dimension D through a 1D convolution and kept as an independent stream throughout the encoder:

$$
{ \bf V } ^ { ( 0 ) } = \mathrm { C o n v } _ { 1 \times 1 } ( { \bf F } ) , ~ { \bf C } ^ { ( 0 ) } = \mathrm { C o n v } _ { 1 \times 1 } ( \bar { \bf P } ) .\tag{6}
$$

The recognition module is frozen during spotting training, so the spotting model receives recognition-derived gloss evidence only through P<sup>¯</sup> . At each encoder layer l, the two streams are first updated by sliding-window self-attention, then exchange information through a bidirectional cross-attention block. The visual stream attends to the peak locations of the CSLR predictions through the attention weights, while the CSLR stream refines its sparse spikes using the temporally continuous cues from the visual features. After L layers, the final encoder output is defined as the concatenation of the two streams, which is forwarded to the decoder stages.

Boundary Head and Multi-Task Objective. To explicitly model gloss transitions, we attach a boundary head implemented as a 1D convolution on top of the last feature of every stage. The boundary targets $b _ { t } \in \{ 0 , 1 \}$ are derived from frame-level GT labels by marking frames where the class index differs from the previous valid frame. To address the heavy positive/negative imbalance, ${ \mathcal { L } } _ { \mathrm { b n d } }$ is defined as a weighted binary cross-entropy:

$$
\mathcal { L } _ { \mathrm { s p o t } } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { \mathrm { m s e } } \mathcal { L } _ { \mathrm { T - M S E } } + \lambda _ { \mathrm { b n d } } \mathcal { L } _ { \mathrm { b n d } } .\tag{7}
$$

The weights are set to $\lambda _ { \mathrm { m s e } } = 0 . 1 5$ and $\lambda _ { \mathrm { b n d } } = 0 . 5$ in all experiments.

Inference and Late Fusion. At inference, the classification logits from the last decoder stage are used for frame-level gloss prediction. To exploit the complementarity between recognition and spotting, we fuse the gloss probabilities of the two models after temporal alignment via a simple linear combination, followed by CTC beam-search decoding:

$$
\mathbf { P } _ { \mathrm { f u s e } } = \alpha \mathbf { P } _ { \mathrm { r e c } } + ( 1 - \alpha ) \mathbf { P } _ { \mathrm { s p o t } } .\tag{8}
$$

We set $\alpha = 0 . 7$ in our grid search experiments.

## 4 Experiments

## 4.1 Experimental Settings

Datasets. Experiments are conducted on four continuous sign language recognition (CSLR) benchmarks: PHOENIX14-T [5], CSL-Daily [52], the Large-scale KSL [1, 17] and Disaster and Safety KSL (DS KSL) [2]. Dataset statistics are summarized in Tab. 1. PHOENIX14-T is a German sign language dataset collected from weather forecast broadcasts, while CSL-Daily consists of Chinese sign language videos covering daily-life scenarios. The two KSL datasets contain Korean sign language videos collected from daily-life and disaster domains.

Table 1: Statistics of the datasets used in our experiments.
<table><tr><td>Dataset</td><td>PHOENIX14-T</td><td>CSL-Daily</td><td>Large-scale KSL</td><td>DS KSL</td></tr><tr><td>Language</td><td>German</td><td>Chinese</td><td>Korean</td><td>Korean</td></tr><tr><td>Domain</td><td>Weather</td><td>Daily-life</td><td>Daily-life</td><td>Disaster(Weather)</td></tr><tr><td># Videos</td><td>8,257</td><td>20,654</td><td>35,987</td><td>28,250</td></tr><tr><td>Split</td><td></td><td></td><td>7096 / 519 / 642 18401 / 1077 / 1176 25590 / 6397 / 400020089 / 5023 / 3138</td><td></td></tr><tr><td># Glosses</td><td>1,066</td><td>2,000</td><td>440</td><td>2,496</td></tr><tr><td># Signers</td><td>9</td><td>10</td><td>18</td><td>20</td></tr></table>

![](images/43634e8e88da6d50fec541dacea74798178e825c093e74b6da6f6e02556ed8d8.jpg)  
Figure 4: Examples of sign language videos from the DS KSL dataset.

Unlike PHOENIX14-T and CSL-Daily, the KSL datasets provide temporal boundary annotations for sign spotting evaluation, enabling evaluation on both CSLR and sign spotting tasks. Representative frames from the DS KSL dataset are shown in Fig. 4. Since the KSL datasets only provide train and test splits, we further divide the training set into training and validation subsets using an 8:2 ratio.

Implementation Details. A CLIP-pretrained ViT-B/16 model [11, 41] is used as the visual backbone. The model is trained for 40 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$ and batch size 8. For video-text alignment, frame-level motion descriptions are generated using LLaVA-OneVision-7B [33] by processing videos, and encoded via BERT [10] to obtain text embeddings. The Multi-Scale Temporal Adapter (MSTA) is inserted into the last four CLIP transformer blocks with reduction factor $r = 3 2$ . For the sign spotting stage, CSFormer is built on an ASFormer [48] encoder-decoder with $L { = } 1 0$ encoder layers, a hidden dimension of $D { = } 6 4$ , and 3 decoder stages. The recognition module is frozen during spotting training, and CSFormer is trained for 20 epochs with a batch size of 16 using Adam, a learning rate of $5 \times 1 0 ^ { - 4 }$ , and a 5-epoch warm-up. All experiments are conducted on a single NVIDIA RTX 6000 Ada GPU.

Evaluation Metrics. For CSLR, we use Word Error Rate (WER) as the evaluation metric. WER evaluates the difference between the predicted gloss sequence and the reference gloss sequence based on substitution, insertion, and deletion operations. Lower WER indicates better performance.

$$
\mathrm { W E R } = \frac { \# s u b s t i t u t i o n s + \# i n s e r t i o n s + \# d e l e t i o n s } { \# r e f e r e n c e }\tag{9}
$$

For sign spotting, we report F1 score:

$$
\mathrm { F 1 } = \frac { 2 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }\tag{10}
$$

Table 2: Comparison with state-of-the-art methods on PHOENIX14-T, CSL-Daily, Large-scale KSL, and DS KSL datasets. Results are reported in WER, where lower is better. Here, SMART (Ours) denotes our recognition model without CSFormer. Best and second-best results are highlighted in bold and underline, respectively. † indicates that the baseline results on Large-scale KSL and DS KSL are reproduced using the official implementations under the original experimental settings.
<table><tr><td rowspan="2">Method</td><td colspan="2">PHOENIX14-T</td><td colspan="2">CSL-Daily</td><td colspan="2">Large-scale KSL†</td><td colspan="2">DS KSL†</td></tr><tr><td>Dev(%)</td><td>Test(%)</td><td>Dev(%)</td><td>Test(%)</td><td>Dev(%)</td><td>Test(%)</td><td>Dev(%)</td><td>Test(%)</td></tr><tr><td>VAC []</td><td></td><td></td><td></td><td></td><td>2.72</td><td>2.81</td><td>31.80</td><td>27.45</td></tr><tr><td>SMKD []</td><td>20.80</td><td>22.40</td><td></td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>TLP []</td><td>19.40</td><td>21.20</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEN []</td><td>19.30</td><td>20.70</td><td></td><td></td><td>1.60</td><td>1.40</td><td>29.70</td><td>25.60</td></tr><tr><td>AdaBrowse []</td><td>19.50</td><td>20.60</td><td>31.20</td><td>30.70</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>SSSLR []</td><td>20.50</td><td>22.30</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>C2SLR []</td><td>20.20</td><td>20.40</td><td>31.90</td><td>31.00</td><td></td><td></td><td></td><td></td></tr><tr><td>CoSign []</td><td>19.50</td><td>20.10</td><td>28.10</td><td>27.20</td><td></td><td></td><td></td><td></td></tr><tr><td>SignBERTplus []</td><td>18.80</td><td>19.90</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CTCA []</td><td>19.30</td><td>20.30</td><td>31.30</td><td>29.40</td><td></td><td></td><td></td><td></td></tr><tr><td>CVT-SLR []</td><td>19.40</td><td>20.30</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CorrNet []</td><td>18.90</td><td>20.05</td><td>30.60</td><td>30.10</td><td>2.62</td><td>2.64</td><td>28.27</td><td>25.79</td></tr><tr><td>AdaptSign []</td><td>18.60</td><td>19.80</td><td>26.70</td><td>26.30</td><td>0.89</td><td>0.76</td><td>30.14</td><td>26.58</td></tr><tr><td>SMART (Ours)</td><td>17.58</td><td>19.50</td><td>26.50</td><td>26.20</td><td>0.64</td><td>0.48</td><td>27.27</td><td>23.92</td></tr></table>

where a predicted segment is counted as a true positive if its temporal overlap with the ground truth exceeds a predefined IoU threshold. Higher F1 indicates better performance. We report F1 scores at varying IoU thresholds (0.10, 0.25, and 0.50) to simultaneously evaluate temporal localization and boundary alignment.

## 4.2 Comparison with State-of-the-Art

Sign Language Recognition Results. Tab. 2 compares SMART with state-of-the-art methods on four benchmarks: PHOENIX14-T, CSL-Daily, Large-scale KSL and DS KSL. SMART achieves the best performance across all datasets, outperforming previous methods in terms of WER. These results show the effectiveness of the proposed MLLM-guided video-text alignment and MSTA for CSLR.

Comparison with Segmentation and Spotting Baselines. As shown in Tab. 3, we evaluate SMART against conventional action segmentation methods and a sign spotting baseline [47]. On Large-scale KSL, the spotting baseline improves over CSLR-only at low IoU thresholds, but does not improve boundary localization at a higher IoU threshold; for example, Test F1@50 changes from 7.37 to 7.13. On DS KSL, the same baseline performs even worse than CSLR-only across all IoU thresholds. This indicates that sparse query-based spotting methods are not well suited for dense frame-level spotting in continuous sign videos. Although action segmentation methods achieve stronger localization, they suffer from significantly higher WER, indicating that visual temporal boundaries alone are insufficient without gloss-level recognition guidance. In contrast, SMART achieves strong performance on both spotting and recognition, obtaining 96.72 F1@50 with 0.48 WER on Large-scale KSL and 59.77 F1@50 with 22.93 WER on DS KSL.

To further validate CSFormer, we integrate it into existing CSLR models (e.g., VAC,

Table 3: Comparison with state-of-the-art methods on the Large-scale KSL and DS KSL benchmarks for sign spotting and spotting-enhanced recognition. For a fair comparison, all baseline methods are reproduced using the official implementations under the original experimental settings.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Dev</td><td></td><td colspan="5">Test</td></tr><tr><td>F1@10</td><td>F1@25</td><td>F1@50</td><td>Acc</td><td>WER(%)</td><td>|F1@10</td><td>F1@25</td><td>F1@50</td><td>Acc</td><td>WER(%)</td></tr><tr><td rowspan="10">LS KSL</td><td>SMART (CSLR-Only)</td><td>35.23</td><td>18.88</td><td>7.47</td><td>59.05</td><td>0.64</td><td>36.58</td><td>19.59</td><td>7.37</td><td>59.64</td><td>0.48</td></tr><tr><td>HS-I3D []</td><td>52.77</td><td>33.01</td><td>8.43</td><td>57.79</td><td></td><td>49.62</td><td>30.84</td><td>7.13</td><td>57.37</td><td></td></tr><tr><td>ASFormer []</td><td>94.40</td><td>94.31</td><td>89.39</td><td>88.93</td><td>6.27</td><td>94.30</td><td>94.25</td><td>89.32</td><td>88.58</td><td>6.69</td></tr><tr><td>ASRF []</td><td>91.23</td><td>90.04</td><td>75.05</td><td>81.11</td><td>12.89</td><td>88.09</td><td>86.78</td><td>72.18</td><td>80.38</td><td>16.24</td></tr><tr><td>LTContext []</td><td>93.32</td><td>93.18</td><td>89.47</td><td>89.47</td><td>6.79</td><td>92.55</td><td>92.40</td><td>86.58</td><td>87.54</td><td>9.27</td></tr><tr><td>VAC + CSFormer</td><td>88.44</td><td>87.48</td><td>83.76</td><td>87.39</td><td>2.72</td><td>95.57</td><td>95.22</td><td>93.02</td><td>90.27</td><td>2.81</td></tr><tr><td>SEN + CSFormer</td><td>90.68</td><td>90.14</td><td>87.64</td><td>89.35</td><td>1.60</td><td>96.20</td><td>95.99</td><td>93.79</td><td>90.48</td><td>1.40</td></tr><tr><td>CorrNet + CSFormer</td><td>91.45</td><td>90.95</td><td>88.33</td><td>89.66</td><td>2.62</td><td>95.72</td><td>95.36</td><td>93.49</td><td>90.89</td><td>2.64</td></tr><tr><td>AdaptSign + CSFormer</td><td>89.38</td><td>88.72</td><td>85.94</td><td>88.73</td><td>0.89</td><td>95.64</td><td>95.23</td><td>93.19</td><td>91.04</td><td>0.76</td></tr><tr><td>SMART (Ours)</td><td>92.99</td><td>92.64</td><td>90.40</td><td>90.17</td><td>0.64</td><td>98.10</td><td>97.95</td><td>96.72</td><td>91.96</td><td>0.48</td></tr><tr><td rowspan="10">DS KSL</td><td>SMART (CSLR-Only)</td><td>53.25</td><td>38.54</td><td>5.50</td><td>62.40</td><td>27.27</td><td>53.81</td><td>37.15</td><td>5.90</td><td>60.36</td><td>23.92</td></tr><tr><td>HS-I3D []</td><td>23.40</td><td>19.56</td><td>3.11</td><td>33.43</td><td></td><td>29.88</td><td>20.75</td><td>3.34</td><td>34.23</td><td></td></tr><tr><td>ASFormer []</td><td>55.20</td><td>52.04</td><td>40.40</td><td>67.54</td><td>52.41</td><td>61.71</td><td>58.75</td><td>45.75</td><td>69.19</td><td>43.46</td></tr><tr><td>ASRF []</td><td>48.81</td><td>44.74</td><td>32.43</td><td>66.67</td><td>62.78</td><td>59.39</td><td>55.41</td><td>40.57</td><td>68.34</td><td>50.57</td></tr><tr><td>LTContext []</td><td>61.20</td><td>54.10</td><td>38.50</td><td>65.40</td><td>61.42</td><td>64.80</td><td>57.70</td><td>41.11</td><td>65.71</td><td>60.30</td></tr><tr><td>VAC + CSFormer</td><td>69.87</td><td>66.64</td><td>52.69</td><td>72.42</td><td>30.83</td><td>74.56</td><td>71.25</td><td>57.17</td><td>73.68</td><td>26.68</td></tr><tr><td>SEN + CSFormer</td><td>72.28</td><td>69.30</td><td>56.33</td><td>74.49</td><td>28.33</td><td>76.29</td><td>73.52</td><td>59.22</td><td>74.72</td><td>24.55</td></tr><tr><td>CorrNet + CSFormer</td><td>72.23</td><td>69.08</td><td>55.16</td><td>74.03</td><td>26.66</td><td>76.13</td><td>73.16</td><td>58.35</td><td>74.23</td><td>23.49</td></tr><tr><td>AdaptSign + CSFormer</td><td>70.48</td><td>69.12</td><td>56.40</td><td>74.14</td><td>26.75</td><td>74.67</td><td>72.73</td><td>58.16</td><td>74.57</td><td>23.23</td></tr><tr><td>SMART (Ours)</td><td>72.45</td><td>69.53</td><td>56.51</td><td>74.51</td><td>26.34</td><td>76.31</td><td>73.56</td><td>59.77</td><td>74.77</td><td>22.93</td></tr></table>

Table 4: Ablation study of SMART components.
<table><tr><td>Fusion</td><td>Align MSTA</td><td>Dev(%)</td><td>Test(%)</td></tr><tr><td></td><td></td><td>18.60</td><td>19.80</td></tr><tr><td></td><td>√</td><td>17.88</td><td>19.82</td></tr><tr><td></td><td>√</td><td>17.98</td><td>19.75</td></tr><tr><td>√</td><td>√</td><td>18.25</td><td>20.31</td></tr><tr><td>√</td><td>√ √</td><td>19.05</td><td>21.08</td></tr><tr><td></td><td>√ √</td><td>17.58</td><td>19.50</td></tr></table>

Table 5: Ablation study of PEFT configurations.
<table><tr><td>Method</td><td>Layers</td><td>Rank</td><td>Dev(%)</td><td>Test(%)</td></tr><tr><td rowspan="5">Adapter</td><td>11-12</td><td>16</td><td>18.84</td><td>20.59</td></tr><tr><td>11-12</td><td>32</td><td>18.28</td><td>20.47</td></tr><tr><td>9-12</td><td>16</td><td>18.62</td><td>20.33</td></tr><tr><td>9-12</td><td>32</td><td>17.58</td><td>19.50</td></tr><tr><td>5-12</td><td>16</td><td>18.09</td><td>20.05</td></tr><tr><td>LoRA</td><td>5-12 9-12</td><td>32 32</td><td>17.88 24.40</td><td>19.39 25.20</td></tr></table>

SEN, CorrNet, and AdaptSign). With CSFormer, CSLR models consistently obtain dense spotting predictions, showing that sparse recognition cues can be transformed into framelevel localization. In particular, SMART improves Test F1@50 from 7.37 to 96.72 on Largescale KSL while preserving WER and from 5.90 to 59.77 on DS KSL while reducing WER from 23.92 to 22.93. These results show that CSFormer bridges CSLR and sign spotting through gloss-level recognition guidance and dense temporal supervision.

## 4.3 Ablation study

## Analysis of each module in SMART.

Sign Language Recognition. Tab. 4 reports the component ablation of the recognition module on the PHOENIX14-T dataset. Compared with the baseline, MSTA reduces the Dev WER from 18.60 to 17.88, showing the effectiveness of MSTA. The alignment objective also improves recognition performance, reducing the Test WER from 19.80 to 19.75. By jointly applying MSTA and alignment, the model achieves the best performance, with 17.58 Dev WER and 19.50 Test WER. In contrast, using MLLM-generated text features via a gated fusion module leads to performance degradation, indicating that the generated descriptions are more effective as auxiliary semantic supervision than as input features for recognition. Sign Language Spotting. Tab. 7 reports the ablation study on the LS KSL and DS KSL datasets to validate the proposed CSFormer. The baseline without CSLR-derived guidance shows limited spotting performance, while adding CSLR features improves Test F1@50 from 94.50 to 95.44 on Large-scale KSL and from 58.54 to 59.68 on DS KSL. This confirms that recognition-derived gloss probabilities provide effective gloss-level guidance for sign spotting. Furthermore, combining CSLR features with cross-attention achieves the strongest spotting performance, reaching 97.06 Test F1@50 on Large-scale KSL and 59.81 on DS KSL, indicating that layer-wise interaction between the visual and CSLR streams is beneficial for dense temporal localization. Although the full model slightly sacrifices F1@50 compared with the best partial variant, it maintains competitive spotting performance and achieves the lowest WER on DS KSL, reducing Test WER from 23.72 to 22.93. Therefore, we use the full CSFormer as the default configuration for balancing sign spotting and sequence-level recognition.

Table 6: Effect of batch size and video-text alignment methods.
<table><tr><td>Batch Size</td><td>Dev(%)</td><td>Test(%) |</td><td>Alignment</td><td>Dev(%)</td><td>Test(%)</td></tr><tr><td>2</td><td>19.13</td><td>20.36</td><td>None</td><td>18.60</td><td>19.80</td></tr><tr><td>4</td><td>18.41</td><td>20.64</td><td>InfoNCE</td><td>18.62</td><td>20.76</td></tr><tr><td>8</td><td>17.58</td><td>19.50</td><td>SigLIP</td><td>17.98</td><td>19.75</td></tr></table>

Table 7: Ablation study of CSFormer on Large-scale KSL and DS KSL. F1@50 and WER are reported to evaluate sign spotting and spotting-enhanced recognition, respectively.
<table><tr><td></td><td></td><td></td><td colspan="4">Large-scale KSL</td><td colspan="4">DS KSL</td></tr><tr><td></td><td></td><td></td><td colspan="2">Dev</td><td colspan="2">Test</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>Bnd head</td><td>Xattn</td><td>CSLR feat</td><td>F1@50</td><td>WER(%)</td><td>F1@50</td><td>WER(%)</td><td>F1@50</td><td>WER(%)</td><td>F1@50</td><td>WER(%)</td></tr><tr><td>一</td><td>一</td><td></td><td>87.42</td><td></td><td>94.50</td><td></td><td>54.25</td><td>27.13</td><td>58.54</td><td>23.72</td></tr><tr><td>√</td><td></td><td></td><td>88.24</td><td></td><td>94.92</td><td></td><td>54.09</td><td>27.07</td><td>57.96</td><td>23.74</td></tr><tr><td>一</td><td></td><td>√</td><td>88.80</td><td></td><td>95.44</td><td></td><td>56.50</td><td>26.92</td><td>59.68</td><td>23.68</td></tr><tr><td>√</td><td></td><td>√</td><td>89.58</td><td></td><td>94.75</td><td></td><td>56.45</td><td>26.95</td><td>58.78</td><td>23.65</td></tr><tr><td>1</td><td>√</td><td>√</td><td>90.35</td><td></td><td>97.06</td><td></td><td>56.43</td><td>27.05</td><td>59.81</td><td>23.66</td></tr><tr><td>√</td><td>√</td><td>√</td><td>90.40</td><td>0.64</td><td>96.72</td><td>0.48</td><td>56.51</td><td>26.34</td><td>59.77</td><td>22.93</td></tr></table>

Parameter-Efficient Fine-Tuning (PEFT). Tab. 5 evaluates MSTA applied to the last $N \in$ {2,4, 8} transformer layers with reduction factors $r \in \{ 1 6 , 3 2 \}$ on PHOENIX14-T. Additional results with smaller reduction factors are provided in the supplementary material. Applying MSTA to the last 4 layers with $r = 3 2$ achieves the best Dev WER of 17.58. Although applying MSTA to the last 8 layers improves Test WER from 19.50 to 19.39, it doubles the adapter parameters and degrades Dev WER from 17.58 to 17.88. Therefore, we adopt the last 4 layers with $r = 3 2$ for parameter-efficient adaptation. Under the same configuration, LoRA performs worse, with 24.40 Dev and 25.20 Test WER.

Video-Text Alignment Analysis. Tab. 6 analyzes the effect of batch size and alignment method on PHOENIX14-T. Increasing the batch size generally improves recognition performance, with batch size 8 achieving the best results of 17.58 Dev WER and 19.50 Test WER. Since larger batch sizes exceed GPU memory capacity for sign language video training, batch size 8 is the practical upper bound in our setting. For the alignment objective, we use the same baseline model [25] without MSTA and vary only the alignment objective. InfoNCE [41] performs worse than the no-alignment baseline, increasing Test WER from 19.80 to 20.76, while SigLIP [50] improves it to 19.75.

![](images/0651796b6ca83d77b8a2fe2841cab555e05f0737013c48fbdebdbe250936d40c.jpg)  
Figure 5: Qualitative comparison of sign spotting performance on (a) Large-scale KSL and (b) DS KSL datasets. Different colors denote distinct glosses, while the block widths correspond to their temporal durations.

<table><tr><td></td><td rowspan=1 colspan=3>Ground Truth</td><td rowspan=1 colspan=1>JETZT</td><td rowspan=1 colspan=1>WETTER</td><td rowspan=1 colspan=1>MORGEN</td><td rowspan=1 colspan=2>SAMSTAG</td><td rowspan=1 colspan=1>ZEHNTE</td><td rowspan=1 colspan=1>APRIL</td><td rowspan=1 colspan=2>WIE-AUSSEHEN</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2>SAMSTAG</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=2>WIE-AUSSEHEN</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=4 colspan=1>ZEIGEN-BILDSCHIRMZEIGEN-BILDSCHIRMZEIGEN-BILDSCHIRMZEIGEN-BILDSCHIRM</td></tr><tr><td></td><td rowspan=1 colspan=3>Baseline</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>WETTER</td><td rowspan=1 colspan=1>MORGEN</td><td rowspan=1 colspan=1>ZEIGEN-</td></tr><tr><td></td><td rowspan=1 colspan=3>Baseline + CL</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>WETTER</td><td rowspan=1 colspan=1>MORGEN</td><td rowspan=1 colspan=2>SAMSTAG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>APRIL</td><td rowspan=1 colspan=2>WIE-AUSSEHEN</td><td rowspan=1 colspan=1>ZEIG</td></tr><tr><td rowspan=4 colspan=4>Ground Truth</td><td rowspan=1 colspan=2>SMART(Ours)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>JETZT</td><td rowspan=1 colspan=1>WETTER</td><td rowspan=1 colspan=1>MORGEN</td><td rowspan=1 colspan=1>SAMSTAG</td><td rowspan=1 colspan=1>ZEHNTE</td><td rowspan=1 colspan=1>APRIL</td><td rowspan=1 colspan=1>WIE-AUSSEHEN</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>HEUTE</td><td rowspan=2 colspan=1>NACHT</td><td rowspan=2 colspan=2>SCHON WEST</td><td rowspan=2 colspan=3>REGEN</td><td rowspan=2 colspan=1>KOMMEN</td><td rowspan=2 colspan=1>FLUSS</td><td rowspan=2 colspan=1>NOCH</td></tr><tr><td rowspan=4 colspan=1></td></tr><tr><td rowspan=1 colspan=4>Baseline</td><td rowspan=1 colspan=1>HEUTE</td><td rowspan=1 colspan=1>NACHT</td><td rowspan=1 colspan=2>SCHON WEST</td><td rowspan=1 colspan=3>REGION   REGEN</td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>REGION</td></tr><tr><td></td><td rowspan=1 colspan=3>Baseline + CL</td><td rowspan=1 colspan=1>HEUTE</td><td rowspan=1 colspan=1>NACHT</td><td rowspan=1 colspan=2>SCHON WEST</td><td rowspan=1 colspan=3>REGION   REGEN</td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=1>FLUSS</td><td rowspan=1 colspan=1>NOCH</td></tr><tr><td></td><td rowspan=1 colspan=3>SMART(Ours)</td><td rowspan=1 colspan=1>HEUTE</td><td rowspan=1 colspan=1>NACHT</td><td rowspan=1 colspan=2>SCHON WEST</td><td rowspan=1 colspan=3>REGEN</td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=1>FLUSS</td><td rowspan=1 colspan=1>NOCH</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td></tr><tr><td></td><td rowspan=3 colspan=3>Ground Truth</td><td rowspan=3 colspan=1>A</td><td rowspan=3 colspan=1>HOCH</td><td rowspan=3 colspan=1>KOMMEN</td><td rowspan=3 colspan=1>SAMS</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=5 colspan=2>TEMPERATUR  STEIGENTEMPERATUR  STEIGENSTEIGENSTEIGEN</td></tr><tr><td></td><td rowspan=1 colspan=1>TAG</td><td rowspan=1 colspan=1>SONNTAG</td><td rowspan=1 colspan=1>MEHR</td><td rowspan=1 colspan=1>SONNE</td></tr><tr><td></td><td rowspan=1 colspan=3>Baseline</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>MEHRSONNE</td></tr><tr><td rowspan=1 colspan=4>Baseline + CL</td><td rowspan=1 colspan=1>EUROPA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=2>SAMSTAG</td><td rowspan=1 colspan=1>SONNTAG</td><td rowspan=1 colspan=3>MEHRSONNE</td></tr><tr><td rowspan=1 colspan=4>SMART(Ours)</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>HOCH</td><td rowspan=1 colspan=1>KOMMEN</td><td rowspan=1 colspan=2>SAMSTAG</td><td rowspan=1 colspan=1>SONNTAG</td><td rowspan=1 colspan=3>MEHRSONNE</td></tr></table>

Figure 6: Gloss prediction comparison between the baseline model, the baseline model with contrastive learning (CL), the proposed SMART framework, and ground truth annotations. Incorrect gloss predictions are highlighted in pink.

## 4.4 Qualitative Analysis

Qualitative results on Sign Spotting. Fig. 5 visualizes spotting results on the Large-scale KSL and DS KSL datasets. SMART (CSLR-Only) exhibits typical peaky alignments: it captures gloss identities at sparse time steps, but fails to cover the full temporal duration of each sign. Conversely, the baseline segmentation method, ASFormer [48], generates dense segments but lacks gloss-level discrimination, suffering from frequent misclassifications and merging adjacent signs. By injecting CSLR-derived gloss guidance into CSFormer, our unified SMART framework overcomes these limitations and produces predictions that better match the ground truth in both temporal boundaries and gloss labels.

Visualization of Gloss Prediction Results. Fig. 6 shows qualitative gloss prediction results on PHOENIX14-T. The baseline model [25] shows frequent deletion and substitution errors, particularly for less frequent signs. Adding contrastive learning (CL) to the baseline improves predictions, but deletion and substitution errors remain in several cases. For example, SMART correctly predicts JETZT and ZEHNTE in the first sequence and A and HOCH in the third, which are missed by the baseline. Overall, SMART achieves predictions closer to the ground truth with fewer deletion and substitution errors.

![](images/3300e5e8591636ae336fa648e537c38bb28f5ebc4bc60fe50d18d23785997194.jpg)  
Figure 7: Qualitative comparison of attention maps between the baseline and SMART on dev and test samples across four CSLR benchmarks.

Qualitative Analysis of Attention Maps. Fig. 7 shows attention visualizations on four CSLR benchmarks using samples from the dev and test sets. Compared with the baseline [25], SMART focuses more on hand movements and facial expressions with less attention to the background. These results suggest that MLLM-guided SigLIP alignment and MSTA improve spatial and temporal feature representations.

## 5 Conclusions

In this work, we proposed SMART, an MLLM-guided Temporal Alignment framework for sign language recognition and spotting. SMART introduces video-text alignment with MLLM-generated motion descriptions, temporal representation learning with a Multi-Scale Temporal Adapter (MSTA), and CSLR-guided sign spotting via CSFormer for temporal boundary refinement. CSFormer injects recognition-derived gloss probabilities into a boundaryaware spotting module to produce dense frame-level predictions for accurate sign localization. Experiments on four sign language benchmarks show that SMART achieves state-ofthe-art performance across sign languages. Additional spotting experiments show that CSLR representations provide effective gloss-level guidance for sign spotting, while dense spotting supervision improves or preserves recognition performance depending on dataset characteristics. These results indicate that gloss-level recognition guidance and dense temporal supervision are complementary for improving continuous sign language understanding.

## 6 Acknowledgement

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP), funded by the Korean government (Ministry of Science and ICT), through the ITRC (IITP-2026-RS-2024-00437102, 50%) and ICAN (IITP-2026-RS-2024-00437027, 50%) programs.

## References

[1] AI Hub. Large-scale korean sign language dataset. https://www.aihub.or.kr, 2020.

[2] AI Hub. Disaster and safety korean sign language dataset. https://www.aihub. or.kr, 2021.

[3] Samuel Albanie, Gül Varol, Liliane Momeni, Triantafyllos Afouras, Joon Son Chung, Neil Fox, and Andrew Zisserman. BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues. In ECCV, 2020.

[4] Emad Bahrami, Gianpiero Francesca, and Juergen Gall. How much temporal longterm context is needed for action segmentation? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10351–10361, 2023.

[5] Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. Neural sign language translation. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7784–7793, 2018.

[6] Ka Leong Cheng, Zhaoyang Yang, Qifeng Chen, and Yu-Wing Tai. Fully convolutional networks for continuous sign language recognition. In European Conference on Computer Vision, pages 697–714. Springer, 2020.

[7] Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, and Richard Bowden. Subunets: End-to-end hand shape and continuous sign language recognition. In Proceedings of the IEEE international conference on computer vision, pages 3056–3065, 2017.

[8] Runpeng Cui, Hu Liu, and Changshui Zhang. Recurrent convolutional neural networks for continuous sign language recognition by staged optimization. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7361–7369, 2017.

[9] Runpeng Cui, Hu Liu, and Changshui Zhang. A deep neural framework for continuous sign language recognition by iterative training. IEEE Transactions on Multimedia, 21 (7):1880–1891, 2019.

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[12] Yazan Abu Farha and Jurgen Gall. Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3575–3584, 2019.

[13] William T Freeman and Michal Roth. Orientation histograms for hand gesture recognition. In International workshop on automaticface and gesture recognition, volume 12, pages 296–301. Zurich, Switzerland, 1995.

[14] Wen Gao, Gaolin Fang, Debin Zhao, and Yiqiang Chen. A chinese sign language recognition system based on sofm/srn/hmm. Pattern Recognition, 37(12):2389–2402, 2004.

[15] Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd international conference on Machine learning, pages 369–376, 2006.

[16] Leming Guo, Wanli Xue, Qing Guo, Bo Liu, Kaihua Zhang, Tiantian Yuan, and Shengyong Chen. Distilling cross-temporal contexts for continuous sign language recognition. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10771–10780, 2023.

[17] Soomin Ham, Kibaek Park, YeongJun Jang, Youngtaek Oh, Seokmin Yun, Sukwon Yoon, Chang Jo Kim, Han-Mu Park, and In So Kweon. Ksl-guide: A large-scale korean sign language dataset including interrogative sentences for guiding the deaf and hard-of-hearing. In 2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021), pages 1–8. IEEE, 2021.

[18] Junwei Han, George Awad, and Alistair Sutherland. Modelling and segmenting subunits for sign language recognition based on hand motion analysis. Pattern recognition letters, 30(6):623–633, 2009.

[19] Aiming Hao, Yuecong Min, and Xilin Chen. Self-mutual distillation learning for continuous sign language recognition. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 11303–11312, 2021.

[20] Hezhen Hu, Weichao Zhao, Wengang Zhou, and Houqiang Li. Signbert+: Hand-modelaware self-supervised pre-training for sign language understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):11221–11239, 2023.

[21] Lianyu Hu, Liqing Gao, Zekang Liu, and Wei Feng. Temporal lift pooling for continuous sign language recognition. In European conference on computer vision, pages 511–527. Springer, 2022.

[22] Lianyu Hu, Liqing Gao, Zekang Liu, and Wei Feng. Continuous sign language recognition with correlation network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2529–2539, 2023.

[23] Lianyu Hu, Liqing Gao, Zekang Liu, and Wei Feng. Self-emphasizing network for continuous sign language recognition. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 854–862, 2023.

[24] Lianyu Hu, Liqing Gao, Zekang Liu, Chi-Man Pun, and Wei Feng. Adabrowse: Adaptive video browser for efficient continuous sign language recognition. In Proceedings ofthe 31st ACM international conference on multimedia, pages 709–718, 2023.

[25] Lianyu Hu, Tongkai Shi, Liqing Gao, Zekang Liu, and Wei Feng. Improving continuous sign language recognition with adapted image models. arXiv preprint arXiv:2404.08226, 2024.

[26] Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, and Hirokatsu Kataoka. Alleviating over-segmentation errors by detecting action boundaries. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2322–2331, 2021.

[27] Youngjoon Jang, Youngtaek Oh, Jae Won Cho, Myungchul Kim, Dong-Jin Kim, In So Kweon, and Joon Son Chung. Self-sufficient framework for continuous sign language recognition. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[28] Tao Jiang, Necati Cihan Camgöz, and Richard Bowden. Looking for the signs: Identifying isolated sign instances in continuous video footage. In 2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021), pages 1–8. IEEE, 2021.

[29] Peiqi Jiao, Yuecong Min, Yanan Li, Xiaotao Wang, Lei Lei, and Xilin Chen. Cosign: Exploring co-occurrence signals in skeleton-based continuous sign language recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 20676–20686, 2023.

[30] Jungeun Kim, Hyeongwoo Jeon, Jongseong Bae, and Ha Young Kim. Leveraging the power of mllms for gloss-free sign language translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21048–21058, 2025.

[31] Oscar Koller, Jens Forster, and Hermann Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer vision and image understanding, 141:108–125, 2015.

[32] Oscar Koller, Necati Cihan Camgoz, Hermann Ney, and Richard Bowden. Weakly supervised learning with multi-stream cnn-lstm-hmms to discover sequential parallelism in sign language videos. IEEE transactions on pattern analysis and machine intelligence, 42(9):2306–2320, 2019.

[33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[34] Shijie Li, Yazan Abu Farha, Yun Liu, Ming-Ming Cheng, and Juergen Gall. Ms-tcn++: Multi-stage temporal convolutional network for action segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):6647–6658, 2023.

[35] Hu Liu, Sheng Jin, and Changshui Zhang. Connectionist temporal classification with maximum entropy regularization. Advances in Neural Information Processing Systems, 31, 2018.

[36] Yuecong Min, Aiming Hao, Xiujuan Chai, and Xilin Chen. Visual alignment constraint for continuous sign language recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11542–11551, 2021.

[37] Liliane Momeni, Gul Varol, Samuel Albanie, Triantafyllos Afouras, and Andrew Zisserman. Watch, read and lookup: learning to spot signs from multiple supervisors. In Proceedings ofthe Asian conference on computer vision, 2020.

[38] Zhe Niu and Brian Mak. Stochastic fine-grained labeling of multi-state sign glosses for continuous sign language recognition. In European conference on computer vision, pages 172–186. Springer, 2020.

[39] Sylvie CW Ong and Surendra Ranganath. Automatic sign language analysis: A survey and the future beyond lexical meaning. IEEE Transactions on Pattern Analysis & Machine Intelligence, 27(06):873–891, 2005.

[40] Junfu Pu, Wengang Zhou, and Houqiang Li. Iterative alignment network for continuous sign language recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4165–4174, 2019.

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[42] Katrin Renz, Nicolaj C. Stache, Samuel Albanie, and Gül Varol. Sign language segmentation with temporal convolutional networks. In ICASSP, 2021.

[43] Chao Sun, Tianzhu Zhang, Bing-Kun Bao, Changsheng Xu, and Tao Mei. Discriminative exemplar coding for sign language recognition with kinect. IEEE Transactions on Cybernetics, 43(5):1418–1428, 2013.

[44] Gul Varol, Liliane Momeni, Samuel Albanie, Triantafyllos Afouras, and Andrew Zisserman. Read and attend: Temporal localisation in sign language videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16857–16866, 2021.

[45] Gül Varol, Liliane Momeni, Samuel Albanie, Triantafyllos Afouras, and Andrew Zisserman. Scaling up sign spotting through sign language dictionaries. International Journal ofComputer Vision, 130(6):1416–1439, 2022.

[46] Manuel Vázquez Enríquez, José L Alba Castro, Laura Docio Fernandez, Julio CS Jacques Junior, and Sergio Escalera. Eccv 2022 sign spotting challenge: dataset, design and results. In European Conference on Computer Vision, pages 225–242. Springer, 2022.

[47] Ryan Wong, Necati Cihan Camgöz, and Richard Bowden. Hierarchical i3d for sign spotting. In European Conference on Computer Vision, pages 243–255. Springer, 2022.

[48] Fangqiu Yi, Hongyu Wen, and Tingting Jiang. Asformer: Transformer for action segmentation. arXiv preprint arXiv:2110.08568, 2021.

[49] Albert Zeyer, Ralf Schlüter, and Hermann Ney. Why does ctc result in peaky behavior? arXiv preprint arXiv:2105.14849, 2021.

[50] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

[51] Jiangbin Zheng, Yile Wang, Cheng Tan, Siyuan Li, Ge Wang, Jun Xia, Yidong Chen, and Stan Z Li. Cvt-slr: Contrastive visual-textual transformation for sign language recognition with variational alignment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23141–23150, 2023.

[52] Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, and Houqiang Li. Improving sign language translation with monolingual data by sign back-translation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 1316– 1325, 2021.

[53] Ronglai Zuo and Brian Mak. C2slr: Consistency-enhanced continuous sign language recognition. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5131–5140, 2022.