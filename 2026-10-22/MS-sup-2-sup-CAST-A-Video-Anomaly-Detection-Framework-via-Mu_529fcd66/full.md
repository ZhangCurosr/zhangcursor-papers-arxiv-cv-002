# MS<sup>2</sup>CAST: A Video Anomaly Detection Framework via Multi-Scale Spatial Cross-Attention and Efficient Spatiotemporal Networks

Ali Safaei<sup>a</sup>, Mohammad Kazemi<sup>b</sup>

<sup>a</sup> Department of Computer Science, University of Milan, Milan, Italy

<sup>b</sup> Department of Electrical Engineering, University of Isfahan, Isfahan, Iran

Corresponding author: Mohammad Kazemi

Email address: m.kazemi@eng.ui.ac.ir

## Abstract

Video anomaly detection (VAD) is a challenging task for surveillance systems due to the scarcity and diversity of anomalous events. While semi-supervised VAD approaches have advanced the field, many rely on computationally intensive optical flow networks to capture motion dynamics. Convolutional neural networks (CNNs) are also widely used for spatial encoding, but their effective receptive field is highly local. To address these limitations, we introduce $\mathbf { M S } ^ { 2 } \mathbf { C A S T } ,$ an end-to-end semi-supervised predictive VAD framework. For spatial encoding, we employ a multi-scale cross-attention encoder that fuses the hierarchical feature representations from a vision transformer (MViTv2-S) and a CNN (ConvNeXt-T) to exploit their distinct receptive fields. To model spatiotemporal dynamics without optical flow estimation, we integrate two interchangeable spatiotemporal modules in the latent space. One module (SwinLSTM) incorporates a Swin Transformer, while the other (VMRNN) integrates Vision Mamba blocks. Our objective function combines L1 loss, Gradient Difference Loss (GDL), and Multi-Scale Structural Similarity Index Measure (MS-SSIM) to ensure pixel-wise accuracy, sharpness, and structural integrity. Evaluations across the UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets demonstrate frame-level Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) scores that are competitive with top-tier techniques while maintaining a peak GPU memory usage of under 1 GB.

Keywords: Video anomaly detection; Multi-scale cross-attention; VMamba; Vision transformers; MS-SSIM

## 1. Introduction

Anomaly detection in videos aims to detect events that are non-conforming to what is normal and expected across both spatial and temporal dimensions. To ensure safe environments, video anomaly detection (VAD) plays an integral role in modern surveillance systems. However, VAD is a challenging task due to the scarcity and diversity of anomalies. Beyond traditional approaches [1], [2] recent advances in deep learning have revolutionized VAD by defining different learning tasks that are based on various supervision signals. Notably, semi-supervised VAD holds a prominent position in this area of research because it bypasses the need for exhaustively collecting rare anomalous samples. Semi-supervised VAD operates on the assumption that only normal samples are available during training, so the model will only learn normal patterns [3]. Consequently, during model inference, anomalies are detected by flagging any sample that deviates from the learned patterns. Frame reconstruction, frame prediction, and a combination of both prediction and reconstruction are three commonly used paradigms for semi-supervised VAD. For the paradigm based on frame reconstruction, the model learns how to reconstruct current normal input frames during training, which can lead to a high reconstruction error for anomalous frames during inference. However, it is possible for a highcapacity model to be occasionally good at reconstructing abnormal frames with low error, thereby increasing the false negative rate. To prevent the reconstruction of anomalous frames, subsequent studies integrated a memory module into autoencoder architecture [4], [5]. However, memory modules can undesirably increase the reconstruction errors for normal frames [6]. Moreover, a memory-augmented architecture can potentially have poor performance in cross-scene applications because memory items are set up for a specific scene [6]. The frame prediction approach can effectively mitigate the overgeneralization issue associated with reconstruction-based approaches [3].

Although predictive approaches for VAD can address the limitations of reconstruction-based methods, they introduce several challenges. Due to the success of generative adversarial networks (GANs) [7] for future frame prediction [8], they are also frequently used in the VAD literature [9], [10], [11], [12], [13], [14]. However, adversarial training is often unstable and sensitive to hyperparameter choices [15]. Incorporating optical flow to enforce motion constraints is effective and widely used [6], [9], [11], [12], [16], but it is computationally intensive. Moreover, because pre-trained optical flow networks like FlowNet 2.0 [17] and RAFT [18] are usually tuned for specific types of data, using them to estimate optical flow can pose limitations on the adaptability of a VAD model depending on the dataset [19]. Lastly, while L2 loss in pixel space is widely used as one of the training objectives [9], [10], [16], [19], it can result in blurry outputs [8], [20].

Architectures based on convolutional neural networks (CNNs) are commonly used for VAD to encode both spatial features and spatiotemporal dynamics [3]. Although recent VAD studies have introduced transformer-based architectures [19], [21], approaches combining both transformers and CNNs remain underexplored. CNNs and vision transformers (ViTs) generate remarkably different feature maps [22]. While CNNs rely on local receptive fields at early layers that grow gradually, ViTs possess a significantly more global receptive field overall [22]. Recent research on feature fusion has shown promising applications of cross-attention mechanisms across different feature representations [23], [24]. This type of fusion has also been effectively used for VAD architectures in various settings [16], [19]. Cross-attention fusion mechanisms provide a way to compute the correlation between two distinct feature representations. Because anomalies typically occur at different spatial scales, a cross-attention mechanism that fuses distinct feature maps from a CNN and a ViT can enable multi-scale anomaly detection.

Beyond spatial feature extraction, predictive VAD models should also effectively capture temporal dependencies. In the broader context of spatiotemporal forecasting, SwinLSTM [25] and VMRNN [26] were proposed to improve predictive performance over ConvLSTM [27]. SwinLSTM incorporates Swin Transformer blocks [28], utilizing window-based and shifted window-based multi-head self-attention. This design achieves the representational power of global self-attention with significantly reduced computational demands. Alternatively, VMRNN integrates Vision Mamba blocks [29], employing the 2D Selective Scan (SS2D) mechanism to extend 1D sequential scanning to 2D spatial scanning, which makes state space models (SSMs) capable of processing spatial information. As alternatives to ConvLSTM, both SwinLSTM and VMRNN are computationally much more efficient and generate higher-quality frames in future frame prediction tasks [26].

In this paper, we propose $\mathbf { M S } ^ { 2 } \mathbf { C A S T } ,$ a predictive VAD framework that integrates a spatial cross-attention mechanism to leverage the feature representations of both ViTs and CNNs for spatial encoding. Our architecture employs parallel pathways to extract multi-scale spatial feature maps from RGB frames. Then, these maps are fused via a spatial cross-attention mechanism and passed through the latent space and skip connections. To effectively capture spatiotemporal dynamics, we integrate VMRNN and SwinLSTM as interchangeable modules in the latent space. Inspired by previous research in future frame prediction [8] and image restoration [20], our loss function combines L1 loss, Multi-Scale Structural Similarity Index Measure (MS-SSIM), and Gradient Difference Loss (GDL). In summary, our main contributions are:

• We introduce MS<sup>2</sup>CAST, a novel end-to-end framework for semi-supervised VAD, leveraging the distinct receptive fields of ViTs and CNNs through a hierarchical spatial cross-attention mechanism.

• We seamlessly integrate VMRNN and SwinLSTM as alternative components into our architecture to efficiently capture motion dynamics without relying on computationally intensive optical flow networks.

• We design a loss function by merging L1 loss to preserve pixel-space fidelity, MS-SSIM to maintain perceptual integrity, and GDL to generate sharp edges within predicted frames.

Our comprehensive evaluations across three public datasets demonstrate that our method achieves strong frame-level Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) scores on par with top-tier techniques while maintaining peak GPU memory usage under 1 GB.

## 2. Related works

Semi-supervised VAD techniques have evolved to capture different kinds of anomalies in various scenarios. Whether a framework relies on single-modal or multi-modal input affects both its detection accuracy and computational demands. This section reviews the literature across these two input settings to contextualize the design of our proposed framework.

## 2.1. Multi-modal approaches

To capture complex motion dynamics, a prominent category of VAD frameworks relies on multi-modal inputs, particularly pairing optical flow with RGB frames. In general, multi-modal VAD frameworks utilize either traditional optical flow algorithms [30], [31], [32] or deep pretrained optical flow networks [16], [33], [34]. While traditional algorithms are lightweight, they often lack the accuracy of modern deep pre-trained optical flow networks. However, the performance of these deep models can degrade when applied to domains that differ from their training data [19].

Among traditional optical flow estimation approaches, the TV-L1 algorithm [35] is widely used. For instance, Wu et al. [30] introduced an encoder-decoder architecture to process modalities independently via 3D CNNs. Their framework linearly fuses these features in the latent space to maintain a compact latent representation while reconstructing both motion and appearance. Utilizing the same TV-L1 algorithm [35], Rai et al. [31] employed two isolated 3D CNN-based autoencoders alongside pseudo-anomaly generation for the separate reconstruction of optical flow and RGB frames. Furthermore, they leveraged features from a pre-trained ViFi-CLIP [36] model to train a discriminator, providing a semantic anomaly detection criterion during inference. Focusing on computational speed, Wu et al. [32] proposed a sparse coding framework that estimates optical flow based on the method introduced in [37]. Using an adapted BN-Inception network [38], their model processes both modalities independently and fuses the spatial and temporal feature maps via channel-wise concatenation and a convolutional operation. Ultimately, it detects anomalies based on deviations from a learned normal dictionary.

To overcome the accuracy limitations of traditional algorithms, several frameworks incorporate deep pre-trained optical flow networks. AMMC-Net [34] maintains semantic consistency between appearance and motion features by utilizing feed-forward neural networks to integrate a cross-modality mapping mechanism within the latent space. This mechanism maps appearance features to motion features and vice versa. Furthermore, the network integrates memory modules in the latent space to detect anomalies based on both prediction errors and discrepancies from memory prototypes. Employing a similar anomaly detection criterion, Ning et al. [33] proposed a predictive encoder-decoder framework utilizing the pre-trained FlowNet 2.0 [17] model alongside a memory module in the latent space. They incorporated a channel attention mechanism into the motion encoder and refined the appearance features via elementwise multiplication with an attention map obtained from the fusion of spatial and temporal feature maps. Taking a different predictive approach, Ma et al. [16] leveraged a pre-trained RAFT [18] model with a cross-attention fusion mechanism for the motion-guided representation of appearance features, followed by a memory module and a decoder equipped with channel and spatial attention mechanisms. It is noteworthy that memory modules, despite being commonly integrated into VAD architectures [16], [33], [34], can limit cross-scene generalization [6]. To bypass the computational demands of optical flow estimation, many frameworks adopt single-modal architectures.

## 2.2. Single-modal approaches

Relying exclusively on RGB information to process both spatial and temporal representations requires careful architectural design to establish a cohesive single-modal VAD framework. A foundational category of these single-modal strategies relies on architectures that employ autoencoders for frame reconstruction [4], [39], [40]. For instance, Luo et al. [40] integrated ConvLSTM [27] into a convolutional autoencoder to address the motion-modeling limitations of ConvAE [39]. To prevent the reconstruction of anomalous frames, Gong et al. [4] introduced a memory-augmented architecture in which the encoded input retrieves a sparse set of stored normal patterns for reconstruction, ensuring that anomalous events yield a high reconstruction error. To capture more diverse patterns than single-encoder convolutional autoencoders, MESDnet [41] introduces an end-to-end multi-encoder architecture. It utilizes one encoder to capture motion via the difference between consecutive frames, alongside another encoder that extracts spatial information. The resulting feature maps are concatenated along the channel dimension and passed to the decoder to reconstruct the current frame. While MESDnet processes only RGB frames as inputs, it notably incorporates an adversarial loss and an optical flow loss into its training objective.

Predictive single-modal frameworks widely employ a U-Net alongside an adversarial training strategy and optical flow-based motion constraints [9], [11], [12]. Adopting an adversarial training strategy, Liu et al. [9] proposed a foundational VAD framework that employs a U-Net for future frame prediction while leveraging a per-trained optical flow network to impose motion constraints during training. Building on Frame-Pred [9], Luo et al. [11] adapted a U-Net by using a graph neural network in the latent space and the Mask R-CNN [42] object detector to model foreground visual relationships. However, object detectors can struggle to detect small objects and may underperform in crowded scenes with occluded objects, whereas choosing an object detector with higher detection accuracy usually comes at the cost of speed and efficiency [43]. To prioritize foreground elements without object-level detection, another architecture [6] integrated a spatial attention mechanism into a U-Net skip connections alongside a channel attention mechanism for spatial encoding. Exploiting rich semantic features of a latent space, Zhang et al. [12] adopted a predictive strategy across multiple spaces by not only predicting the future frame in the pixel space but also encoding those predictions into a latent space to compare them against the corresponding ground truth. Inspired by research in neuroscience, Ye et al. [44] proposed a dual-module architecture consisting of a predictive coding module that uses ConvLSTM [27] and a U-Net to encode future frames, coupled with a refinement module to model the prediction error. Following a hybrid approach while adopting an adversarial training strategy, Tang et al. [10] combined frame reconstruction and prediction strategies using only RGB frames, comparing their work against Frame-Pred [9].

Over the past few years, architectures utilizing transformers have emerged within single-modal VAD. As one of the early VAD frameworks to leverage transformers, TransAnomaly [21] adopts an adversarial training strategy and incorporates a video vision transformer into the latent space of a U-Net to serve as a spatiotemporal module capturing motion dynamics. Additionally, Tran et al. [19] proposed a predictive VAD framework utilizing transformers for aerial videos. They designed a spatial transformer encoder to extract spatial information and a temporal transformer encoder to model temporal relationships across consecutive frames, introducing a temporal cross-attention mechanism to fuse the obtained temporal tokens into a single temporal token.

While these single-modal approaches have advanced the field, many still rely on adversarial training, object detectors, or an optical flow loss. To employ an effective streamlined optimization process, we instead adopt an objective function that combines MS-SSIM, GDL, and L1 loss. Furthermore, rather than relying solely on CNNs or transformers, we leverage a spatial cross-attention mechanism to fuse the distinct feature maps of both a ViT and a CNN and pair this spatial encoding with an efficient spatiotemporal module to capture motion dynamics.

## 3. Proposed method

## 3.1. Overall architecture

We propose $\mathbf { M S } ^ { 2 } \mathbf { C A S T } .$ , a semi-supervised framework for VAD that leverages a ViT and a CNN to extract spatial features and fuses them using a cross-attention mechanism. As illustrated in Fig. 1, the architecture includes three main components: a multi-scale cross-attention encoder, a spatiotemporal module, and a decoder. It takes a video clip consisting of four frames $\left( I _ { t - 3 : t } \right)$ to predict the next frame $( \hat { I } _ { t + 1 } )$ . Grounded in the U-Net architecture [45], it first extracts two distinct sets of multi-level spatial features from each input frame using the ViT and CNN independently. It then fuses them at each level, using the CNN features to guide the feature maps obtained from the ViT. The fused feature map at the lowest resolution passes through the spatiotemporal module and then into the decoder, while the remaining fused feature maps from the last input frame $\left( I _ { t } \right)$ are used in skip connections to pass spatial information at varying resolutions to the decoder for prediction.

## 3.2. Multi-scale cross-attention encoder

To effectively capture a broad range of spatial patterns in video frames, our spatial encoder employs two distinct parallel pathways. We choose MViTv2-S [46], a multi-scale vision transformer, and ConvNeXt-T [47], a modern convolutional neural network, as our spatial feature extractors. We selected these architectures to generate the multi-scale representations required for future frame prediction in our U-Net-based framework, while leveraging the distinct receptive fields of a ViT and a CNN. High-capacity feature extractors are often computationally heavy; however, the “Tiny” variant of ConvNeXt and the “Small” variant of MViT-v2 provide an optimal balance between computational requirements and representational power. Additionally, they inherently generate multi-scale feature representations with matching resolutions at each stage. We exploit these attributes in these two architectures to fuse their feature maps at different scales.

![](images/534cc2987cb861291422e0ebc1d14aa38d64c1b7afaa7ca65d712971a6b1e45c.jpg)  
Fig. 1. Overview of the proposed MS<sup>2</sup>CAST framework. The architecture takes a clip of four consecutive frames to predict the future frame by leveraging a multi-scale cross-attention encoder, a spatiotemporal module, and a decoder.

## 3.2.1. Multi-scale spatial feature extractors

MViTv2 [46] is a hierarchical vision transformer that integrates decomposed relative positional embeddings and residual pooling connections alongside a hybrid window and pooling attention to address the high computational demands of global self-attention while maintaining high performance. As the other feature extractor in our spatial encoder, ConvNeXt [47] incorporates inverted bottlenecks, large kernels, and depthwise convolution. Both ConvNeXt and MViTv2 gradually reduce the spatial resolution of the input frames while expanding the channel dimension. The spatial downsampling ratios of these two networks mirror each other. At each stage, the outputs from ConvNeXt-T and MViTv2-S share perfectly aligned spatial dimensions and feature depths. We extract the intermediate feature maps from four distinct structural stages across both models and apply the cross-attention mechanism to the paired feature maps. Both backbones are initialized with ImageNet-1k [48] pre-trained weights and kept frozen during training.

Given an input sequence of � = 4 RGB frames $\mathbf { I } \in \mathbb { R } ^ { B \times T \times 3 \times H \times W }$ within a batch with size of �, where $H = 2 2 4$ and $W = 2 2 4$ , both models extract features across four hierarchical stages $l \in \{ 1 , 2 , 3 , 4 \}$ . The output features at stage � for MViTv2-S and ConvNeXt-T are defined as $\mathbf { F } _ { \mathrm { M V i T v } 2 \lnot S } ^ { ( l ) } \in \mathbb { R } ^ { B \times T \times C _ { l } \times H _ { l } \times W _ { l } }$ and $\mathbf { F } _ { \mathrm { C o n v N e X t - T } } ^ { ( l ) } \in \mathbb { R } ^ { B \times T \times C _ { l } \times H _ { l } \times W _ { l } }$ , with the specific channel and spatial dimensions at each stage � defined as:

• Stage $\textbf { 1 } ( l = 1 ) \colon C _ { 1 } = 9 6 , \quad H _ { 1 } = W _ { 1 } = 5 6$

• Stage $2 ( l = 2 ) \colon C _ { 2 } = 1 9 2 , \quad H _ { 2 } = W _ { 2 } = 2 8$

• Stage $3 ( l = 3 ) \colon C _ { 3 } = 3 8 4 , \quad H _ { 3 } = W _ { 3 } = 1 4$

• Stage $4 ( l = 4 ) \colon C _ { 4 } = 7 6 8 , \quad H _ { 4 } = W _ { 4 } = 7$

## 3.2.2. 2D sinusoidal spatial positional encoding

The cross-attention fusion used in this study operates on flattened sequences of spatial feature maps. This characteristic makes it an inherently permutation-invariant mechanism. To preserve 2D spatial topology across spatial feature maps, we utilize a non-learnable 2D sinusoidal positional encoding [49], adapted from the foundational 1D sequence formulation [50]. Considering coordinates $( x , y , c )$ where $x \in \{ 0 , \ldots , W _ { l } - 1 \} , \quad y \in \{ 0 , \ldots , H _ { l } - 1 \}$ and $c \in$ $\{ 0 , \ldots , C _ { l } - 1 \}$ , the channel dimension $C _ { l }$ is partitioned equally into �-axis and �-axis representations. For a given position $( x , y , c )$ and channel indices $k \in \{ 0 , \ldots , C _ { l } / 4 - 1 \}$ , the frequency constant $\omega _ { k }$ and the encodings are defined as:

$$
\omega _ { k } = ( \frac { 1 } { 1 0 0 0 0 } ) ^ { 4 k / C _ { l } }\tag{1}
$$

$$
\begin{array} { r } { \mathrm { P E } _ { l } ( x , y , c ) = \left\{ \begin{array} { l c } { \sin ( y \cdot \omega _ { k } ) , } & { \mathrm { i f } c = 2 k } \\ { \cos ( y \cdot \omega _ { k } ) , } & { \mathrm { i f } c = 2 k + 1 } \\ { \sin ( x \cdot \omega _ { k } ) , } & { \mathrm { i f } c = C _ { l } / 2 + 2 k } \\ { \cos ( x \cdot \omega _ { k } ) , } & { \mathrm { i f } c = C _ { l } / 2 + 2 k + 1 } \end{array} \right. } \end{array}\tag{2}
$$

This encoding is scaled dynamically to fit the resolution $H _ { l } \times W _ { l }$ at each stage and broadcast across the batch and temporal dimensions.

## 3.2.3. Multi-head cross-attention mechanism

The feature maps at each stage � are flattened into representations of shape $\mathbb { R } ^ { B T \times H _ { l } W _ { l } \times C _ { l } }$ and subsequently passed into the cross-attention mechanism. Considering flattened representations $\mathbf { Z } _ { \mathrm { M V i T v 2 - S } } ^ { ( l ) } \in \mathbb { R } ^ { B T \times H _ { l } W _ { l } \times C _ { l } }$ and ${ \bf Z } _ { \mathrm { C o n v N e X t - T } } ^ { ( l ) } \in \mathbb { R } ^ { B T \times H _ { l } W _ { l } \times C _ { l } }$ , we define Query (�), Key (�), and Value (�) for our cross-attention mechanism at each stage �:

$$
\mathbf { Q } ^ { ( l ) } = \mathbf { Z } _ { \mathrm { M V i T v 2 . S } } ^ { ( l ) } , \quad \mathbf { K } ^ { ( l ) } = \mathbf { Z } _ { \mathrm { C o n v N e X t - T } } ^ { ( l ) } , \quad \mathbf { V } ^ { ( l ) } = \mathbf { Z } _ { \mathrm { C o n v N e X t - T } } ^ { ( l ) }\tag{3}
$$

Using $h = 6$ parallel attention heads, the head projection dimension is $d _ { h } = C _ { l } / h$ . For head $i \in \{ 1 , \ldots , h \}$ , linear transformations with learnable weights $\mathbf { W } _ { i } ^ { Q } , \mathbf { W } _ { i } ^ { K } , \mathbf { W } _ { i } ^ { V } \in \mathbb { R } ^ { C _ { l } \times d _ { h } }$ project the sequence representations into subspace projections:

$$
\mathbf { Q } _ { i } = \mathbf { Q } ^ { ( l ) } \mathbf { W } _ { i } ^ { Q } , \quad \mathbf { K } _ { i } = \mathbf { K } ^ { ( l ) } \mathbf { W } _ { i } ^ { K } , \quad \mathbf { V } _ { i } = \mathbf { V } ^ { ( l ) } \mathbf { W } _ { i } ^ { V }\tag{4}
$$

The Multi-Head Attention (MHA) operation across all heads is formulated as:

$$
\mathrm { H e a d } _ { i } = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { \top } } { \sqrt { d _ { h } } } \right) \mathbf { V } _ { i }\tag{5}
$$

$$
\operatorname { M H A } \bigl ( \mathbf { Q } ^ { ( l ) } , \mathbf { K } ^ { ( l ) } , \mathbf { V } ^ { ( l ) } \bigr ) = \operatorname { C o n c a t } ( \operatorname { H e a d } _ { 1 } , \operatorname { H e a d } _ { 2 } , \dots , \operatorname { H e a d } _ { h } ) \mathbf { W } ^ { O }\tag{6}
$$

where $\mathbf { W } ^ { O } \in \mathbb { R } ^ { C _ { l } \times C _ { l } }$ represents the output projection matrix.

In transformer architecture, layer normalization is traditionally applied after the residual connections. This configuration is known as Post-Layer Normalization (Post-LN). In the Post-LN configuration, positional encodings are directly added to raw inputs, and layer normalization is applied after residual summation:

$$
\mathbf { Q } _ { \mathrm { p e } } ^ { ( l ) } = \mathbf { Q } ^ { ( l ) } + \mathbf { P E } ^ { ( l ) } , \quad \mathbf { K } _ { \mathrm { p e } } ^ { ( l ) } = \mathbf { K } ^ { ( l ) } + \mathbf { P E } ^ { ( l ) }\tag{7}
$$

$$
\mathbf { A } ^ { ( l ) } = \mathrm { M H A } \Big ( \mathbf { Q } _ { \mathrm { p e } } ^ { ( l ) } , \mathbf { K } _ { \mathrm { p e } } ^ { ( l ) } , \mathbf { V } ^ { ( l ) } \Big )\tag{8}
$$

$$
\mathbf { Z } _ { \mathrm { i n t e r m e d i a t e } } ^ { ( l ) } = \mathbf { Q } ^ { ( l ) } + \mathbf { A } ^ { ( l ) }\tag{9}
$$

$$
\mathbf { Z } _ { \mathrm { f u s e d } } ^ { ( l ) } = \mathrm { L N } \Big ( \mathbf { Z } _ { \mathrm { i n t e r m e d i a t e } } ^ { ( l ) } \Big )\tag{10}
$$

There is also another configuration called Pre-Layer Normalization (Pre-LN) in which layer normalizations $\mathrm { L N } _ { Q }$ and $\mathrm { L N } _ { K }$ are applied to Query and Key/Value representations prior to the positional encoding process and attention computation:

$$
\mathbf { 0 } _ { \mathrm { n o r m } } ^ { ( l ) } = \mathrm { L N } _ { Q } \big ( \mathbf { Q } ^ { ( l ) } \big ) , \quad \mathbf { K } _ { \mathrm { n o r m } } ^ { ( l ) } = \mathrm { L N } _ { K } \big ( \mathbf { K } ^ { ( l ) } \big ) , \quad \mathbf { V } _ { \mathrm { n o r m } } ^ { ( l ) } = \mathbf { K } _ { \mathrm { n o r m } } ^ { ( l ) }\tag{11}
$$

$$
\mathbf { Q } _ { \mathrm { p e } } ^ { ( l ) } = \mathbf { Q } _ { \mathrm { n o r m } } ^ { ( l ) } + \mathbf { P E } ^ { ( l ) } , \quad \mathbf { K } _ { \mathrm { p e } } ^ { ( l ) } = \mathbf { K } _ { \mathrm { n o r m } } ^ { ( l ) } + \mathbf { P E } ^ { ( l ) }\tag{12}
$$

$$
\mathbf { A } ^ { ( l ) } = \mathrm { M H A } \Big ( \mathbf { Q } _ { \mathrm { p e } } ^ { ( l ) } , \mathbf { K } _ { \mathrm { p e } } ^ { ( l ) } , \mathbf { V } _ { \mathrm { n o r m } } ^ { ( l ) } \Big )\tag{13}
$$

$$
\mathbf { Z } _ { \mathrm { f u s e d } } ^ { ( l ) } = \mathbf { Q } ^ { ( l ) } + \mathbf { A } ^ { ( l ) }\tag{14}
$$

In our experiments, we tested both the Pre-LN (Fig. 2(a)) and Post-LN (Fig. 2(b)) crossattention configurations to compare their effects on our framework. Experimental details regarding these configurations are discussed in Section 4.

Following cross-attention fusion, the flattened sequence $\mathbf { Z } _ { \mathrm { f u s e d } } ^ { ( l ) } \in \mathbb { R } ^ { B T \times H _ { l } W _ { l } \times C _ { l } }$ is reshaped back to spatial feature dimension $\mathbf { F } _ { \mathrm { f u s e d } } ^ { ( l ) } \in \mathbb { R } ^ { B \times T \times C _ { l } \times H _ { l } \times W _ { l } }$ . The lowest-resolution feature map $( l =$ 4) is forwarded to the spatiotemporal module, while the spatial feature maps obtained from the last input frame $\left( I _ { t } \right)$ at the first three stages are passed via skip connections to the decoder.

![](images/f1133ea0d6e705c7971f6acf02806da90d96bbfd5909821a467970e88c506826.jpg)  
Fig. 2. Structural configurations for the multi-head cross-attention fusion mechanism. The diagrams illustrate the (a) Pre-Layer Normalization (Pre-LN) and (b) Post-Layer Normalization (Post-LN) configurations.

## 3.3. Spatiotemporal module

To model motion dynamics across consecutive frames, the bottleneck features $\mathbf { F } _ { \mathrm { f u s e d } } ^ { ( 4 ) }$ extracted by the multi-scale cross-attention encoder, are fed into a spatiotemporal module.

The effective receptive field in deep CNNs is highly local [51]. This attribute poses a constraint on ConvLSTM [27], as its reliance on CNNs limits its ability to capture long-range spatial dependencies during spatiotemporal forecasting [26]. SwinLSTM [25] and VMRNN [26] are two alternatives to ConvLSTM that are not only significantly lighter computationally but also possess a global receptive field [26]. Furthermore, they generate higher-quality predictions compared to ConvLSTM when employed in future frame prediction tasks [26].

## 3.3.1. SwinLSTM

SwinLSTM [25] integrates Swin Transformer blocks [28] to leverage window-based and shifted window-based multi-head self-attention. At each timestep �, the fused spatial feature

map $\mathbf { F } _ { \mathrm { f u s e d } , t } ^ { ( 4 ) } \in \mathbb { R } ^ { B \times C _ { 4 } \times H _ { 4 } \times W _ { 4 } }$ is first projected into a sequence of spatial tokens via a patch embedding layer with a $1 \times 1$ patch size:

$$
\mathbf { X } _ { t } = \mathrm { P a t c h E m b e d } \Big ( \mathbf { F } _ { \mathrm { f u s e d } , t } ^ { ( 4 ) } \Big ) \in \mathbb { R } ^ { B \times H _ { 4 } W _ { 4 } \times C _ { 4 } }\tag{15}
$$

where $H _ { 4 } W _ { 4 } = 4 9$ represents the spatial token count and $C _ { 4 } = 7 6 8$ denotes the feature dimension.

To propagate temporal memory, the input token sequence $\mathbf { X } _ { t }$ and the previous hidden state $\mathbf { H } _ { t - 1 } \in \mathbb { R } ^ { B \times H _ { 4 } W _ { 4 } \times C _ { 4 } }$ are fused at the entry of the first Swin Transformer block. Specifically, both states are normalized using layer normalization and concatenated along the channel dimension:

$$
\mathbf { Z } _ { t } ^ { ( 0 ) } = \mathbf { W } [ \mathrm { L N } ( \mathbf { X } _ { t } ) \| \mathrm { L N } ( \mathbf { H } _ { t - 1 } ) ]\tag{16}
$$

where $\mathbf { W } \in \mathbb { R } ^ { C _ { 4 } \times 2 C _ { 4 } }$ is a learnable projection matrix and ‖ represents channel-wise concatenation.

The combined state $\mathbf { Z } _ { t } ^ { ( 0 ) }$ is sequentially processed through a cascade of $D = 6$ Swin Transformer blocks, each equipped with residual connections. To continuously guide the temporal representations with current spatial information, the initial token sequence $\mathbf { X } _ { t }$ is concatenated with the intermediate feature maps at the entry of every odd-indexed block $m >$ 1, where $m \in \{ 1 , \ldots , D \}$ . The �-th Swin Transformer block alternates between window-based multi-head self-attention (W-MSA) when � is odd and shifted window-based multi-head selfattention (SW-MSA) when � is even. For a local window of size $M \times M$ , where $M = 7$ , the self-attention operation is defined as:

$$
\operatorname { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { S o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { { \sqrt { d } } } } + \mathbf { B } _ { \mathrm { r e l } } \right) \mathbf { V }\tag{17}
$$

where �, �, $\mathbf { V } \in \mathbb { R } ^ { M ^ { 2 } \times d }$ are Query, Key, and Value matrices projected from token representations, $d = C _ { 4 } / h$ is the dimension of Query/Key across $h = 2 4$ attention heads, and $\mathbf { B } _ { \mathrm { r e l } } \in \mathbb { R } ^ { M ^ { 2 } \times M ^ { 2 } }$ is a learnable relative position bias matrix.

After passing through all D blocks, the transformed representation $\mathbf { F } _ { t } = \mathbf { Z } _ { t } ^ { ( D ) } \in \mathbb { R } ^ { B \times H _ { 4 } W _ { 4 } \times C _ { 4 } }$ updates the cell state $\mathbf { C } _ { t }$ and hidden state $\mathbf { H } _ { t }$ through a recurrent gating mechanism:

$$
\mathbf { G } _ { t } = \sigma ( \mathbf { F } _ { t } )\tag{18}
$$

$$
\tilde { \mathbf { C } } _ { t } = \operatorname { t a n h } ( \mathbf { F } _ { t } )\tag{19}
$$

$$
\mathbf { C } _ { t } = \mathbf { G } _ { t } \odot \left( \mathbf { C } _ { t - 1 } + \tilde { \mathbf { C } } _ { t } \right)\tag{20}
$$

$$
\mathbf { H } _ { t } = \mathbf { G } _ { t } \odot \mathrm { t a n h } ( \mathbf { C } _ { t } )\tag{21}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function, $\odot$ represents elementwise Hadamard multiplication, and initial states are set to zero.

## 3.3.2. VMRNN

Considering a continuous-time state space model with an input $x ( t ) \in \mathbb { R }$ , an output $y ( t ) \in \mathbb { R }$ and a hidden state vector $h ( t ) \in \mathbb { R } ^ { N _ { \mathrm { s t a t e } } }$ , where $N _ { \mathrm { s t a t e } } = 1 6$ , the state space model equations are defined as [29]:

$$
h ^ { \prime } ( t ) = \mathbf { A } h ( t ) + \mathbf { B } x ( t )\tag{22}
$$

$$
y ( t ) = \mathbf { C } h ( t ) + \mathbf { D } x ( t )\tag{23}
$$

where $\mathbf { A } \in \mathbb { R } ^ { N _ { \mathrm { s t a t e } } \times N _ { \mathrm { s t a t e } } }$ , $\mathbf { B } \in \mathbb { R } ^ { N _ { \mathrm { { s t a t e } } } \times 1 }$ , $\mathbf { C } \in \mathbb { R } ^ { 1 \times N _ { \mathrm { s t a t e } } }$ , and $\mathbf { D } \in \mathbb { R } ^ { 1 }$ represent the state space parameters. To apply this continuous formulation to discrete input sequences in a deep learning framework, � and � are discretized to $\bar { \bf A }$ and $\mathbf { \overline { { B } } } ,$ respectively, by using a zero-order hold (ZOH) approach with a timescale parameter [29]. This results in the following discrete-time equations at sequence index �:

$$
h _ { k } = \overline { { \mathbf { A } } } h _ { k - 1 } + \overline { { \mathbf { B } } } x _ { k }\tag{24}
$$

$$
y _ { k } = \mathbf { C } h _ { k } + \mathbf { D } x _ { k }\tag{25}
$$

The core architectural component of VMRNN [26] is the Visual State Space block (VSB) [29]. At timestep �, the fused spatial feature map $\mathbf { F } _ { \mathrm { f u s e d , } } ^ { ( 4 ) }$ is first projected into a sequence of spatial tokens. To propagate temporal memory, this sequence is concatenated with the previous hidden state $\mathbf { H } _ { t - 1 }$ along the channel dimension and linearly projected at the entry of the first VSB, as formulated in Eqs. (15) and (16). In the VSB, the tokens are reshaped back to their 2D spatial orientation. The features are then linearly projected to an expanded inner dimension $D _ { \mathrm { i n n e r } } =$ $2 C _ { 4 }$ and split into two parallel streams $\mathbf { X } _ { \mathrm { i n } } \in \mathbb { R } ^ { B \times H _ { 4 } \times W _ { 4 } \times D _ { \mathrm { i } } }$ nner and $\mathbf { Z } _ { \mathrm { i n } } \in \mathbb { R } ^ { B \times H _ { 4 } \times W _ { 4 } \times D _ { \mathrm { i n n e r } } }$ . Then, the stream ${ \bf X } _ { \mathrm { i n } }$ is transposed to a tensor of shape $\mathbb { R } ^ { B \times D _ { \mathrm { i n n e r } } \times H _ { 4 } \times W _ { 4 } }$ and passes through a $3 \times 3$ depthwise convolution followed by a SiLU activation function. Employing the 2D Selective Scan (SS2D) mechanism, the result is scanned along four distinct trajectories to generate 1D sequence streams $\mathbf { x } ^ { ( v ) } \in \mathbb { R } ^ { B \times D _ { \mathrm { i n n e r } } \times H _ { 4 } W _ { 4 } }$ , where $v \in \{ 1 , 2 , 3 , 4 \}$ : raster scan, reversed raster scan, column-major scan, reversed column-major scan. Each sequence $\mathbf { x } ^ { ( v ) }$ is processed by an independent discrete SSM to ensure spatial context aggregation from all directions. All the outputs $\mathbf { y } ^ { ( v ) }$ are restored to their original 2D spatial orientations via inverse scanning operators and then aggregated by summation. Finally, the aggregated output $\mathbf { Y } _ { \mathrm { S S 2 D } }$ is linearly projected back to dimension $C _ { 4 }$

$$
\mathbf { Y } _ { \mathrm { o u t } } = \mathbf { W } _ { \mathrm { o u t } } \big ( \mathrm { L N } ( \mathbf { Y } _ { \mathrm { S S 2 D } } ) \odot \mathrm { S i L U } ( \mathbf { Z } _ { \mathrm { i n } } ) \big )\tag{26}
$$

The initial input passes through a network that integrates $D = 6 \mathrm { \ V S B s }$ equipped with residual connections. After passing through all � blocks, the final block output $\mathbf { F } _ { t } = \mathbf { Z } _ { t } ^ { \left( D \right) } \in$ $\mathbb { R } ^ { B \times H _ { 4 } W _ { 4 } \times C _ { 4 } }$ passes through the exact same recurrent gating mechanism described for SwinLSTM in Eqs. (18)–(21) to update the cell state $\mathbf { C } _ { t }$ and hidden state $\mathbf { H } _ { t }$ [26].

## 3.4. Decoder

As the final architectural component of $\mathbf { M S } ^ { 2 } \mathbf { C A S T } ,$ the decoder is designed to generate the future frame $\hat { I } _ { t + 1 }$ by mapping the final hidden state $\mathbf { H } _ { t }$ produced by the spatiotemporal module. The decoder utilizes five distinct upsampling stages to generate output frames with spatial dimensions of $2 2 4 \times 2 2 4$

Prior to upsampling, $\mathbf { H } _ { t }$ is processed by a Double Convolution block, which is composed of two identical sequences of operations, each consisting of a $3 \times 3$ convolution followed by a group normalization layer and a ReLU activation function. As illustrated in Fig. 3, the decoder employs a hierarchical upsampling strategy to generate the output frame:

• Stages 1 to 3 (With Skip Connections): Each of the first three upsampling stages applies bilinear interpolation with a scale factor of 2 to expand the spatial dimensions of a feature map, immediately followed by a 3 × 3 convolution to reduce the channel dimension. These convolution results are concatenated along the channel dimension with the corresponding fused intermediate feature maps obtained from the final input frame $\mathbf { I } _ { t }$ in the multi-scale cross-attention encoder, including $\mathbf { F } _ { \mathrm { f u s e d } , t } ^ { ( 3 ) } \in \mathbb { R } ^ { B \times C _ { 3 } \times H _ { 3 } \times W _ { 3 } }$ for the first stage, $\mathbf { F } _ { \mathrm { f u s e d } , t } ^ { ( 2 ) } \in \mathbb { R } ^ { B \times C _ { 2 } \times H _ { 2 } \times W _ { 2 } }$ for the second, and $\mathbf { F } _ { \mathrm { f u s e d } , t } ^ { ( 1 ) } \in \mathbb { R } ^ { B \times C _ { 1 } \times H _ { 1 } \times W _ { 1 } }$ for the third. After concatenation, the obtained representations are processed by a Double Convolution block.

• Stages 4 to 5 (Without Skip Connections): The fourth and fifth upsampling stages operate without skip connections, relying entirely on all the steps introduced for the first three stages except concatenation.

• Output Layer: The last upsampling stage results in a feature map with 24 channels and 224 × 224 spatial dimensions. Subsequently, the output layer applies a 1 × 1 convolution to reduce the channel dimension to 3, resulting in the RGB representation. To generate the predicted frame, a sigmoid activation function is then applied to map the output values to a continuous range between 0 and 1.

We chose group normalization over batch normalization in the Double Convolution blocks because batch normalization relies on the statistical mean and variance across the batch dimension. This reliance can cause performance degradation if the batch statistics do not accurately represent the actual dataset domain statistics. Conversely, group normalization calculates the mean and variance independently for each channel group within an individual sample, which makes the normalization process inherently independent of batch size.

![](images/87055b9ef150fc093edbb91d8a9bf9dedc5641c3772b4848079fdccf87b3b8d4.jpg)  
Fig. 3. The hierarchical upsampling strategy employed by the decoder to generate the predicted frame.

## 3.5. Training loss

To train our predictive model to generate sharp frames with high quality, we adopt an objective function that integrates different criteria focused on pixel-wise accuracy, sharpness of edges, and perceptual integrity. The selection of these specific criteria is driven by the findings of both Mathieu et al. [8] and Zhao et al. [20]. To elaborate, Mathieu et al. [8] demonstrated that choosing L1 loss over L2 loss for future frame prediction yields sharper predictions with higher quality, while pairing L1 loss with GDL is an improvement over using L1 loss alone. Similarly, Zhao et al. [20] showed that employing L1 loss instead of L2 loss is a more effective strategy for super-resolution, leading to higher-quality output images. Additionally, they highlighted that a combination of L1 loss and MS-SSIM for their total loss function yields better results than using either loss function alone. Building on these established insights, we formulate our objective function.

Given a predicted frame $\hat { I } _ { t + 1 }$ and its corresponding ground truth $I _ { t + 1 }$ , the total loss function is a weighted combination of three distinct components including L1 loss $( L _ { 1 } ) , \mathrm { G D L } ( L _ { \mathrm { G D L } } )$ , and MS-SSIM loss $\left( L _ { \mathrm { M S \_ S S I M } } \right)$

The L1 loss computes the pixel-wise mean absolute difference between the predicted frame and the ground truth frame:

$$
L _ { 1 } = \frac { 1 } { N } \left. \hat { I } _ { t + 1 } - I _ { t + 1 } \right. _ { 1 }\tag{27}
$$

where $\Vert . \Vert _ { 1 }$ denotes the L1 norm and � represents the total number of pixels in the frame.

To generate sharp edges within a predicted frame, GDL penalizes the mean absolute error between the absolute spatial gradients of the prediction and the target along both horizontal and vertical axes:

$$
L _ { G D L } = \frac { 1 } { N } \Big ( \big \| | \nabla _ { x } I _ { t + 1 } | - | \nabla _ { x } \widehat { I } _ { t + 1 } | \big \| _ { 1 } + \big \| | \nabla _ { y } I _ { t + 1 } | - | \nabla _ { y } \widehat { I } _ { t + 1 } | \big \| _ { 1 } \Big )\tag{28}
$$

where ‖ . ‖ denotes the L1 norm and | . | denotes the element-wise absolute value, and � is the total number of pixels. The operators $\nabla _ { x }$ and $\nabla _ { y }$ compute the spatial gradients of the frames along the horizontal and vertical axes, respectively.

To consider the holistic structural integrity of the generated frame, we utilize MS-SSIM. The Structural Similarity Index Measure (SSIM) evaluates frame quality using three components: luminance (�), contrast (�), and structure (�). Considering a local window � in the predicted future frame and a local window � in the actual future frame, both extracted via an $1 1 \times 1 1$ sliding Gaussian kernel with standard deviation $\sigma = 1 . 5$ , the three mentioned components are defined as [52]:

$$
l ( \mathbf { x } , \mathbf { y } ) = \frac { 2 \mu _ { x } \mu _ { y } + C _ { 1 } } { \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } }\tag{29}
$$

$$
c ( \mathbf { x } , \mathbf { y } ) = \frac { 2 \sigma _ { x } \sigma _ { y } + C _ { 2 } } { \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } }\tag{30}
$$

$$
s ( \mathbf { x } , \mathbf { y } ) = \frac { \sigma _ { x y } + C _ { 3 } } { \sigma _ { x } \sigma _ { y } + C _ { 3 } }\tag{31}
$$

where $\mu _ { x }$ and $\mu _ { y }$ denote the spatial means within each local window, $\sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ represent the variance within each local window, and $\sigma _ { x y }$ is the cross-covariance of windows. The constants $C _ { 1 } = 0 . 0 0 0 1 , C _ { 2 } = 0 . 0 0 0 9$ , and $C _ { 3 } = C _ { 2 } / 2$ prevent division by zero in each fraction.

For the introduced local windows � and $y ,$ the SSIM is defined as [52]:

$$
\mathrm { S S I M } ( \mathbf { x } , \mathbf { y } ) = l ( \mathbf { x } , \mathbf { y } ) \cdot c ( \mathbf { x } , \mathbf { y } ) . s ( \mathbf { x } , \mathbf { y } ) = \frac { \big ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } \big ) \big ( 2 \sigma _ { x y } + C _ { 2 } \big ) } { \big ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } \big ) \big ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } \big ) }\tag{32}
$$

The final SSIM value is computed for a frame by averaging all the SSIM values obtained for the local windows across that frame.

MS-SSIM applies a similar computational process hierarchically at $M = 5$ different frame scales by employing a progressive downscaling process. MS-SSIM is computed as [52]:

$$
\mathrm { M S - S S I M } ( \mathbf { x } , \mathbf { y } ) = [ l _ { M } ( \mathbf { x } , \mathbf { y } ) ] ^ { \alpha _ { M } } \cdot \prod _ { m = 1 } ^ { M } [ c _ { m } ( \mathbf { x } , \mathbf { y } ) . s _ { m } ( \mathbf { x } , \mathbf { y } ) ] ^ { \beta _ { m } }\tag{33}
$$

where $\beta _ { 1 } = 0 . 0 4 4 8$ , � = 0.2856, � = 0.3001, � = 0.2363, and $\alpha _ { M } = \beta _ { M } = 0 . 1 3 3 3$

Because MS-SSIM yields a bounded score between 0 and 1, where a higher score indicates better frame quality, our MS-SSIM loss component $\left( L _ { \mathrm { M S \_ S S I M } } \right)$ is defined as:

$$
L _ { \mathrm { M S \_ S S I M } } = 1 - \mathrm { M S \_ S S I M } \left( \hat { I } _ { t + 1 } , I _ { t + 1 } \right)\tag{34}
$$

Our total loss function, as a linear combination of the previously mentioned components, is defined as:

$$
L _ { \mathrm { t o t a l } } = \alpha L _ { 1 } + \alpha L _ { \mathrm { G D L } } + ( 2 - 2 \alpha ) L _ { \mathrm { M S \_ S S I M } }\tag{35}
$$

The coefficient $\alpha \in ( 0 , 1 )$ ensures all loss weights remain positive. Because $L _ { 1 }$ and $L _ { \mathrm { G D L } }$ both penalize pixel-level errors by focusing on pixel intensities and spatial gradients, respectively, they share the same weight. This design explicitly partitions the objective function, weighing the combined pixel-level errors against the holistic structural integrity measured by $L _ { \mathrm { M S } \_ S S I M }$ The optimization process for the coefficient � is discussed in detail in Section 4.3.

## 3.6. Anomaly score

Our predictive model is trained exclusively on normal samples. This should result in high prediction errors for anomalous samples during model inference. To detect anomalies based on prediction errors, we define an anomaly scoring mechanism based on both Peak Signal-to-Noise Ratio (PSNR) and MS-SSIM. We integrate both metrics because PSNR effectively captures pixel-level prediction errors, whereas MS-SSIM quantifies structural deviations.

While our training objective utilizes L1 loss to preserve pixel-wise accuracy, we employ PSNR for anomaly scoring because its reliance on the squared L2 error quadratically magnifies large pixel-wise deviations, thereby making anomalous events more distinct. Given a predicted frame $\hat { I } _ { t + 1 }$ and its corresponding ground truth $I _ { t + 1 }$ , PSNR is defined as:

$$
\mathrm { P S N R } \big ( I _ { t + 1 } , \hat { I } _ { t + 1 } \big ) = 1 0 \log _ { 1 0 } \left( \frac { { I _ { \operatorname* { m a x } } } ^ { 2 } } { { \left. \hat { I } _ { t + 1 } - I _ { t + 1 } \right. } _ { 2 } / N } \right)\tag{36}
$$

where $\Vert . \Vert$ denotes the L2 norm, � is the total number of pixels in the frame, and $I _ { \mathrm { m a x } }$ is the maximum possible pixel intensity value.

To measure structural frame quality across various scales, we compute the MS-SSIM score between $\hat { I } _ { t + 1 }$ and $I _ { t + 1 }$ as defined in Eq. (33). Consequently, anomaly scores $S _ { \mathrm { P S N R } } ( t )$ and $S _ { \mathrm { M S \_ S S I M } } ( t )$ are defined as:

$$
S _ { \mathrm { P S N R } } ( t ) \ : = \ : - \mathrm { P S N R } \big ( I _ { t + 1 } , \hat { I } _ { t + 1 } \big )\tag{37}
$$

$$
\begin{array} { r } { S _ { \mathrm { M S \_ S S I M } } ( t ) = 1 - \mathrm { M S \_ S S I M } \big ( \hat { I } _ { t + 1 } , I _ { t + 1 } \big ) } \end{array}\tag{38}
$$

Subsequently, we apply Min-Max normalization independently to both $S _ { \mathrm { P S N R } } ( t )$ and $S _ { \mathrm { M S \_ S S I M } } ( t )$ within each individual video clip. Each of the obtained normalized scores, denoted as $\bar { S } _ { \mathrm { P S N R } } ( t )$ and $\bar { S } _ { \mathrm { M S \_ S S I M } } ( t )$ , falls into the range [0, 1]:

$$
\bar { S } _ { \mathrm { P S N R } } ( t ) = \frac { { S } _ { \mathrm { P S N R } } ( t ) - \underset { t ^ { \prime } } { \operatorname* { m i n } } ( S _ { \mathrm { P S N R } } ( t ^ { \prime } ) ) } { \underset { t ^ { \prime } } { \operatorname* { m a x } } ( S _ { \mathrm { P S N R } } ( t ^ { \prime } ) ) - \underset { t ^ { \prime } } { \operatorname* { m i n } } ( S _ { \mathrm { P S N R } } ( t ^ { \prime } ) ) }\tag{39}
$$

$$
\bar { S } _ { \mathrm { M S \_ S S I M } } ( t ) = \frac { { S _ { \mathrm { M S \_ S S I M } } ( t ) - \operatorname* { m i n } _ { t ^ { \prime } } ( S _ { \mathrm { M S \_ S S I M } } ( t ^ { \prime } ) ) } } { \operatorname* { m a x } _ { t ^ { \prime } } ( S _ { \mathrm { M S \_ S S I M } } ( t ^ { \prime } ) ) - \operatorname* { m i n } _ { t ^ { \prime } } ( S _ { \mathrm { M S \_ S S I M } } ( t ^ { \prime } ) ) }\tag{40}
$$

where $t ^ { \prime }$ ranges over all frame indices within a given video clip.

Next, $\bar { S } _ { \mathrm { P S N R } } ( t )$ and $\bar { S } _ { \mathrm { M S \_ S S I M } } ( t )$ are combined linearly using the coefficient �:

$$
S ( t ) = w \bar { S } _ { \mathrm { P S N R } } ( t ) + ( 1 - w ) \bar { S } _ { \mathrm { M S \_ S S I M } } ( t )\tag{41}
$$

Lastly, to obtain the final anomaly score, we utilize a 1D Gaussian filter to smooth $S ( t )$ . The coefficient $w \in [ 0 , 1 ]$ guarantees a valid convex combination of the normalized scores. The optimal value of this coefficient is analyzed further in Section 4.

## 4. Experiments

## 4.1. Datasets

To evaluate the performance of the $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ framework following a semi-supervised approach, we conduct experiments on three widely used, publicly available benchmark datasets: UCSD Ped2 [53], CUHK Avenue [54], and ShanghaiTech [55].

The UCSD Ped2 [53] dataset captures a pedestrian walkway in a single scene, where the training data consists of normal pedestrian walking. The test set contains 12 anomalous events, with anomalies defined as non-pedestrian objects, including bicycles, skateboards, and vehicles entering the walkway. The dataset includes 16 training videos and 12 testing videos with a resolution of $2 4 0 \times 3 6 0$ pixels.

The CUHK Avenue [54] dataset contains 16 training videos and 21 testing videos at a resolution of $3 6 0 \times 6 4 0$ pixels, recorded from a single scene of a campus avenue. The dataset includes 47 anomalous events such as individuals running, walking in abnormal directions, and throwing objects.

The ShanghaiTech [55] dataset is one of the most comprehensive and challenging benchmark datasets that can be used to evaluate multi-scene learning capability of a VAD model. It encompasses 13 distinct scenes, consisting of 330 training videos and 107 testing videos with a resolution of $4 8 0 \times 8 5 6$ pixels. The test set includes 130 abnormal events, covering a wide range of anomalies like people running, fighting, and riding. A comparative summary of the benchmark datasets is provided in Table 1.

Table 1  
Comparative summary of the three benchmark datasets
<table><tr><td rowspan="2">Dataset</td><td colspan="2">#Videos</td><td colspan="2">#Frames</td><td rowspan="2">#Anomalous Events</td><td rowspan="2">#Scenes</td></tr><tr><td>Training</td><td>Testing</td><td>Training</td><td>Testing</td></tr><tr><td>UCSD Ped 2</td><td>16</td><td>12</td><td>2,550</td><td>2,010</td><td>12</td><td>1</td></tr><tr><td>CUHK Avenue</td><td>16</td><td>21</td><td>15,328</td><td>15,324</td><td>47</td><td>1</td></tr><tr><td>ShanghaiTech</td><td>330</td><td>107</td><td>274,515</td><td>42,883</td><td>130</td><td>13</td></tr></table>

## 4.2. Evaluation criterion and implementation details

Following established protocols in VAD research [9], we use the frame-level Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) to quantitatively evaluate our proposed framework.

The $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ framework is implemented in PyTorch and evaluated on a single NVIDIAA100 GPU (40 GB VRAM). Each input video clip consists of four consecutive frames used to predict one future frame. Input frames are extracted and resized to a resolution of $2 2 4 \times 2 2 4$ pixels in RGB format, and their pixel values are scaled to a continuous range of [0, 1]. For training our model on each dataset, we use an AdamW optimizer with a weight decay of 0.01 and linear warmup followed by

cosine annealing to schedule the learning rate. We use a batch size of 16 with a peak learning rate of $2 \times 1 0 ^ { - 4 }$ over 25 epochs for the UCSD Ped2 dataset; a batch size of 32 with a peak learning rate of $5 \times 1 0 ^ { - 4 }$ over 15 epochs for the CUHK Avenue dataset; and a batch size of 128 with a peak learning rate of $1 \times 1 0 ^ { - 3 }$ over 5 epochs for the ShanghaiTech dataset.

## 4.3. Loss function optimization

As defined in Eq. (35), the hyperparameter � plays an integral role in determining the impact of each component within the proposed total loss function, which is a linear combination of three different criteria. To specify the optimal value for �, we conducted experiments tuning � across the set $\{ 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ using the architecture featuring Post-LN Cross-Attention and the SwinLSTM temporal module. We evaluated the model on the CUHK Avenue and UCSD Ped2 datasets using $w = 1 . 0$ and $w = 0 . 6 .$ , respectively, which are the optimal values for the coefficient $w$ in Eq. (41) for each dataset. Table 2 summarizes the frame-level AUC scores obtained for each value of $\alpha$

The results highlight the importance of the perceptual loss component $\left( L _ { \mathrm { M S \_ S S I M } } \right)$ over the pixel-wise loss component $\left( L _ { 1 } \right)$ and the GDL $( L _ { \mathrm { G D L } } )$ . Across both benchmark datasets, the architecture achieved its peak performance at $\alpha = 0 . 2$ . Consequently, $\alpha = 0 . 2$ was selected as the hyperparameter for all subsequent experiments.

Table 2  
Frame-level AUC (%) across different � values for the architecture with Post-Layer Normalization (Post-LN) cross-attention and the SwinLSTM temporal module on the UCSD Ped2 and CUHK Avenue datasets.
<table><tr><td>α</td><td>CUHK Avenue  $( w = 1 . 0 )$ </td><td>UCSD Ped2  $( w = 0 . 6 )$ </td></tr><tr><td>0.2</td><td>86.4</td><td>96.2</td></tr><tr><td>0.3</td><td>85.6</td><td>95.8</td></tr><tr><td>0.4</td><td>85.4</td><td>93.5</td></tr><tr><td>0.5</td><td>85.1</td><td>95.6</td></tr></table>

## 4.4. Architectural ablations and performance evaluation

## 4.4.1. Spatial encoder and spatiotemporal module effect evaluation

To evaluate the effectiveness of our proposed multi-scale spatial cross-attention mechanisms and to compare the interchangeable spatiotemporal modules, we conducted a comprehensive ablation study. We compared the anomaly detection capabilities of ten different models on the UCSD Ped2 and CUHK Avenue datasets. For each variant of the spatiotemporal module, we tested five distinct spatial encoders: a multi-scale Pre-LN cross-attention encoder, a multi-scale Post-LN cross-attention encoder, a pre-trained MViTv2-S encoder, a fine-tuned MViTv2-S encoder, and a pre-trained ConvNeXt-T encoder.

In addition to comparing the pre-trained MViTv2-S as a single spatial encoder against the encoders that are using the two variants of the spatial cross-attention mechanism, we investigated the effect of fine-tuning the pre-trained MViTv2-S encoder for domain adaptation. We adopted a layer-wise learning rate strategy for this fine-tuning process. To stabilize the spatiotemporal module and the decoder, the MViTv2-S backbone was kept frozen during the initial training warmup steps. Following this warmup, the pre-trained model was unfrozen to begin fine-tuning, utilizing a block-wise learning rate decay factor of 0.9. Under this strategy, the deepest block of the pre-trained model received the lowest learning rate, while the shallowest block shared the same learning rate as the spatiotemporal module and the decoder. In all evaluated single-encoder architectures, the three integrated skip connections originate from the intermediate stages of the respective encoder. Table 3 and Table 4 present the framelevel AUC comparisons for each evaluated architecture across the UCSD Ped2 and CUHK Avenue datasets, respectively.

Table 3  
Frame-level AUC (%) comparison across different combinations of spatial encoders and spatiotemporal modules on UCSD Ped2 dataset, where � is the coefficient used to calculate the anomaly score.
<table><tr><td>Spatiotemporal Module</td><td>Spatial Encoder</td><td>Optimal w</td><td>AUC (%)</td></tr><tr><td rowspan="5">VMRNN</td><td>Pre-trained ConvNeXt-T</td><td>0.5</td><td>95.2</td></tr><tr><td>Pre-trained MViTv2-S</td><td>0.5</td><td>95.5</td></tr><tr><td>Fine-tuned MViTv2-S</td><td>0.5</td><td>95.1</td></tr><tr><td>Pre-LN cross-attention</td><td>0.6</td><td>96.5</td></tr><tr><td>Post-LN cross-attention</td><td>0.5</td><td>95.7</td></tr><tr><td rowspan="5">SwinLSTM</td><td>Pre-trained ConvNeXt-T</td><td>0.3</td><td>95.7</td></tr><tr><td>Pre-trained MViTv2-S</td><td>0.6</td><td>96.0</td></tr><tr><td>Fine-tuned MViTv2-S</td><td>0.6</td><td>95.6</td></tr><tr><td>Pre-LN cross-attention</td><td>0.6</td><td>96.1</td></tr><tr><td>Post-LN cross-attention</td><td>0.6</td><td>96.2</td></tr></table>

## Table 4

Frame-level AUC (%) comparison across different combinations of spatial encoders and spatiotemporal modules on CUHK Avenue dataset, where � is the coefficient used to calculate the anomaly score.
<table><tr><td>Spatiotemporal Module</td><td>Spatial Encoder</td><td>Optimal w</td><td>AUC (%)</td></tr><tr><td rowspan="5">VMRNN</td><td>Pre-trained ConvNeXt-T</td><td>0.9</td><td>84.6</td></tr><tr><td>Pre-trained MViTv2-S</td><td>0.8</td><td>86.2</td></tr><tr><td>Fine-tuned MViTv2-S</td><td>0.8</td><td>86.1</td></tr><tr><td>Pre-LN cross-attention</td><td>0.8</td><td>85.6</td></tr><tr><td>Post-LN cross-attention</td><td>0.9</td><td>85.6</td></tr><tr><td rowspan="5">SwinLSTM</td><td>Pre-trained ConvNeXt-T</td><td>0.8</td><td>85.6</td></tr><tr><td>Pre-trained MViTv2-S</td><td>0.8</td><td>86.1</td></tr><tr><td>Fine-tuned MViTv2-S</td><td>0.7</td><td>84.8</td></tr><tr><td>Pre-LN cross-attention</td><td>0.9</td><td>85.8</td></tr><tr><td>Post-LN cross-attention</td><td>1.0</td><td>86.4</td></tr></table>

In all cases, fine-tuning the MViTv2-S model pre-trained on the ImageNet-1k [48] dataset resulted in AUC degradation. This decline is likely because fine-tuning a high-capacity pretrained model requires a substantial amount of data, and fine-tuning such models with an insufficient number of samples can potentially disrupt the generalized feature representations initially learned from the massive ImageNet-1k dataset. As reported in Table 1, the CUHK Avenue and UCSD Ped2 datasets have much fewer training frames than the number of images in the ImageNet-1k dataset, which contains more than one million samples across 1,000 classes in its training set [48].

The results presented in this section exhibit different trends across these datasets. On the UCSD Ped2 dataset, using cross-attention mechanisms for the spatial encoder consistently resulted in a higher AUC score. Conversely, on the CUHK Avenue dataset, utilizing VMRNN as the spatiotemporal module and incorporating cross-attention mechanisms in the spatial encoder to leverage both MViTv2-S and ConvNeXt-T did not improve the AUC over using MViTv2-S alone. Notably, the configuration pairing the multi-scale Post-LN cross-attention spatial encoder with a SwinLSTM as the spatiotemporal module achieved the highest AUC on the CUHK Avenue dataset across all ten evaluated configurations, whereas using the multi-scale Pre-LN spatial cross-attention encoder demonstrated inferior performance compared with employing pre-trained MViTv2-S alone.

Although the model that incorporates the multi-scale Post-LN spatial cross-attention encoder and SwinLSTM exhibited superior performance across all examined models on both datasets, we further evaluated all configurations in noisy environments to assess their noise resilience and robustness. The noise resilience of these models is discussed in Section 4.4.2.

## 4.4.2. Robustness to noise

Surveillance systems commonly suffer from environmental noise that adversely affects the quality of their recordings [56]. Consequently, it is crucial to consider the noise resilience of a VAD model as a factor in model selection for such applications. To assess the performance of the models introduced in Section 4.4.1 under various noisy conditions, we added zero-mean additive white Gaussian noise (AWGN) to the input frames across different standard deviations (�) during model evaluation. All other model evaluation protocols remained identical to those described in Section 4.4.1. Table 5 and Table 6 present the frame-level AUC of each model under various AWGN standard deviations (�) on the UCSD Ped2 and CUHK Avenue datasets, respectively.

Table 6  
Table 5  
Frame-level AUC (%) comparison under various additive white Gaussian noise (AWGN) standard deviations (�) on the UCSD Ped2 dataset.
<table><tr><td rowspan="2">Spatiotemporal Module</td><td rowspan="2">Spatial Encoder</td><td colspan="4">AUC (%)</td></tr><tr><td>No Noise</td><td>σ = 5</td><td>σ = 10</td><td>σ = 15</td></tr><tr><td rowspan="4">VMRNN</td><td>Pre-trained ConvNeXt-T</td><td>95.2</td><td>94.9</td><td>89.4</td><td>69.8</td></tr><tr><td>Pre-trained MViTv2-S</td><td>95.5</td><td>93.2</td><td>91.2</td><td>82.7</td></tr><tr><td>Pre-LN cross-attention</td><td>96.5</td><td>95.4</td><td>92.8</td><td>89.4</td></tr><tr><td>Post-LN cross-attention</td><td>95.7</td><td>95.1</td><td>93.5</td><td>85.0</td></tr><tr><td rowspan="4">SwinLSTM</td><td>Pre-trained ConvNeXt-T</td><td>95.7</td><td>94.5</td><td>85.0</td><td>72.6</td></tr><tr><td>Pre-trained MViTv2-S</td><td>96.0</td><td>95.5</td><td>90.7</td><td>87.6</td></tr><tr><td>Pre-LN cross-attention</td><td>96.1</td><td>95.9</td><td>93.6</td><td>90.5</td></tr><tr><td>Post-LN cross-attention</td><td>96.2</td><td>95.5</td><td>95.2</td><td>91.9</td></tr></table>

Frame-level AUC (%) comparison under various additive white Gaussian noise (AWGN) standard deviations (�) on the CUHK Avenue dataset.
<table><tr><td rowspan="2">Spatiotemporal Module</td><td rowspan="2">Spatial Encoder</td><td colspan="4">AUC (%)</td></tr><tr><td>No Noise</td><td>σ = 5</td><td>σ = 10</td><td>σ = 15</td></tr><tr><td rowspan="4">VMRNN</td><td>Pre-trained ConvNeXt-T</td><td>84.6</td><td>84.3</td><td>82.5</td><td>81.5</td></tr><tr><td>Pre-trained MViTv2-S</td><td>86.2</td><td>86.2</td><td>85.9</td><td>82.9</td></tr><tr><td>Pre-LN cross-attention</td><td>85.6</td><td>85.6</td><td>83.2</td><td>79.9</td></tr><tr><td>Post-LN cross-attention</td><td>85.6</td><td>84.1</td><td>79.2</td><td>75.1</td></tr><tr><td rowspan="4">SwinLSTM</td><td>Pre-trained ConvNeXt-T</td><td>85.6</td><td>85.1</td><td>82.2</td><td>78.2</td></tr><tr><td>Pre-trained MViTv2-S</td><td>86.1</td><td>86.1</td><td>84.5</td><td>81.9</td></tr><tr><td>Pre-LN cross-attention</td><td>85.8</td><td>85.8</td><td>85.4</td><td>82.4</td></tr><tr><td>Post-LN cross-attention</td><td>86.4</td><td>84.2</td><td>77.5</td><td>68.7</td></tr></table>

Across all configurations, the AUC monotonically decreases as the AWGN standard deviation (�) increases. However, the severity of AUC degradation varies depending on the configuration.

Although pairing the multi-scale Post-LN cross-attention encoder with SwinLSTM outperforms the other configurations in environments without noise, the AUC of this configuration on the CUHK Avenue dataset drops significantly under noisy conditions. Conversely, using the multi-scale Pre-LN cross-attention encoder along with SwinLSTM generally exhibits highly robust anomaly detection performance in the presence of noise on both datasets. This difference in noise resilience stems from how layer normalization interacts with feature representations within the integrated cross-attention mechanism. In the Post-LN cross-attention mechanism (Eqs. (7)–(10)), layer normalization operates on the residual sum obtained after the scaled dot-product attention (Eq. (5)), which is computed from unnormalized, noisy Query and Key/Value representations. Consequently, this can lead to a distorted attention weight matrix that is summed with the noisy representation of the Query. In contrast, the Pre-LN cross-attention mechanism (Eqs. (11)–(14)) normalizes the Query and Key/Value representations prior to the computation of the attention weight matrix and the subsequent residual summation. This pre-normalization statistically standardizes feature maps before the cross-attention computations, resulting in feature representations that are less disrupted by noise.

It is worth noting that pairing the pre-trained MViTv2-S encoder with VMRNN shows robustness on the CUHK Avenue dataset, even though the configuration that uses the multiscale Pre-LN cross-attention encoder along with SwinLSTM consistently outperforms it on the UCSD Ped2 dataset. Overall, our findings confirm that pairing the multi-scale Pre-LN crossattention encoder with SwinLSTM results in a robust architecture capable of maintaining high AUC scores under various noisy conditions across both datasets. Consequently, we selected this specific configuration to evaluate its multi-scene learning capability on the ShanghaiTech dataset, which is a comprehensive, multi-scene benchmark.

## 4.4.3. Computational efficiency analysis

Beyond anomaly detection capability and noise resilience, operational efficiency is a critical factor for deploying VAD models in real-world scenarios. To assess the computational efficiency of our proposed architecture during inference and compare it with other alternatives introduced in our ablation studies, we measured latency (milliseconds), throughput (frames per second), and peak GPU memory usage (MB) for six distinct architectural configurations. Table 7 compares the computational efficiency of the evaluated architectures using three different metrics.

Table 7  
Computational efficiency comparison across different combinations of spatial encoders and spatiotemporal modules.
<table><tr><td>Spatiotemporal Module</td><td>Spatial Encoder</td><td>Latency (ms)</td><td>Throughput (FPS)</td><td>Peak GPU Memory (MB)</td></tr><tr><td rowspan="3">VMRNN</td><td>Pre-trained ConvNeXt-T</td><td>38</td><td>161</td><td>536</td></tr><tr><td>Pre-trained MViTv2-S</td><td>59</td><td>106</td><td>617</td></tr><tr><td>Pre-LN cross-attention</td><td>72</td><td>64</td><td>750</td></tr><tr><td rowspan="3">SwinLSTM</td><td>Pre-trained ConvNeXt-T</td><td>31</td><td>192</td><td>592</td></tr><tr><td>Pre-trained MViTv2-S</td><td>51</td><td>118</td><td>674</td></tr><tr><td>Pre-LN cross-attention</td><td>62</td><td>68</td><td>806</td></tr></table>

The results demonstrate a clear trade-off between anomaly detection capability and computational efficiency across different architectural configurations. Using a single-encoder configuration yields lower latency and higher throughput compared with the configuration that employs a multi-scale cross-attention encoder. However, as established in Section 4.4.1 and Section 4.4.2, relying on a single-encoder configuration limits the anomaly detection capability of a model. A comparison of the two interchangeable spatiotemporal modules reveals that SwinLSTM consistently operates with lower latency and higher throughput than VMRNN across all configurations, though it requires more GPU memory.

The peak GPU memory usage of our selected architecture that uses the multi-scale Pre-LN cross-attention encoder along with SwinLSTM is 806 MB. This indicates that our framework can successfully maintain peak GPU memory usage under 1 GB while demonstrating strong anomaly detection capability.

## 4.5. Comparison with existing methods

To further evaluate the multi-scene learning capability of our selected configuration for the $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ framework, which employs the multi-scale Pre-LN cross-attention encoder along with SwinLSTM, we conducted experiments on the ShanghaiTech dataset. On this benchmark, the configuration achieved an AUC of 75.3% using an anomaly score coefficient $w = 0 . 2$ Table 8 compares the frame-level AUC scores of $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ against top-tier semi-supervised VAD methods across the three benchmark datasets, categorizing these methods into multimodal and single-modal techniques. Our proposed framework achieves performance comparable to these methods on the UCSD Ped2 and CUHK Avenue datasets, while outperforming all of them on the challenging ShanghaiTech dataset.

Table 8  
Frame-level AUC (%) comparison with top-tier semi-supervised video anomaly detection methods on the UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets.
<table><tr><td>Type</td><td>Method</td><td>UCSD Ped2</td><td>CUHK Avenue</td><td>ShanghaiTech</td></tr><tr><td></td><td>DeepOC [30]</td><td>96.9</td><td>86.6</td><td>一</td></tr><tr><td></td><td>FSCN [32]</td><td>92.8</td><td>85.5</td><td></td></tr><tr><td>Multi-Modal</td><td>AMMC-Net [34]</td><td>96.6</td><td>86.6</td><td>73.7</td></tr><tr><td></td><td>MAMC [33]</td><td>96.7</td><td>87.6</td><td>71.5</td></tr><tr><td></td><td>CF2M-Net [16]</td><td>96.9</td><td>85.7</td><td>73.9</td></tr><tr><td></td><td>ConvLSTM-AE [40]</td><td>88.1</td><td>77.0</td><td>一</td></tr><tr><td></td><td>TransAnomaly [21]</td><td>96.4</td><td>87.0</td><td>一</td></tr><tr><td></td><td>MNAD-R [5]</td><td>90.2</td><td>82.8</td><td>69.8</td></tr><tr><td></td><td>MemAE [4]</td><td>94.1</td><td>83.3</td><td>71.2</td></tr><tr><td></td><td>Frame-Pred [9]</td><td>95.4</td><td>85.1</td><td>72.8</td></tr><tr><td>Single-Modal</td><td>Zhang et al. [6]</td><td>95.8</td><td>84.9</td><td>71.4</td></tr><tr><td></td><td>F2PN [11]</td><td>96.2</td><td>85.7</td><td>73.0</td></tr><tr><td></td><td>MGAN-CL [14]</td><td>96.5</td><td>87.1</td><td>73.6</td></tr><tr><td></td><td>Dual GroupGAN [13]</td><td>96.6</td><td>85.5</td><td>73.1</td></tr><tr><td></td><td>Multispace [12]</td><td>95.4</td><td>86.8</td><td>73.6</td></tr><tr><td></td><td>MS²CAST (Pre-LN + SwinLSTM)</td><td>96.1</td><td>85.8</td><td>75.3</td></tr></table>

Although MAMC [33] and $\mathrm { C F ^ { 2 } M - N e t }$ [16] demonstrate strong performance on the singlescene datasets, they utilize pre-trained optical flow networks that introduce computational overhead. Additionally, they both incorporate memory modules, which can lead to poor crossscene generalization [6]. Among single-modal techniques, Dual GroupGAN [13] and MGAN-CL [14] achieve high AUC scores on the single-scene datasets. However, their reliance on adversarial training can pose challenges regarding training stability and hyperparameter tuning [15]. In contrast, our end-to-end framework demonstrates superior multi-scene learning capability on the ShanghaiTech dataset and exhibits competitive anomaly detection performance across the single-scene benchmarks. Moreover, $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ maintains peak GPU memory usage under 1 GB and avoids the computational overhead associated with adversarial training strategies and heavy pre-trained optical flow models.

To qualitatively demonstrate the effectiveness of the MS<sup>2</sup>CAST framework, Figs. 4(a), 4(b), and 4(c) visualize the actual frame, the predicted frame, and the corresponding error map during anomalous events across the UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets, respectively. Each error map shows the pixel-wise deviations between the prediction and actual frame. Furthermore, Figs. 5(a), 5(b), and 5(c) illustrate the anomaly score over time across specific video clips for these respective datasets, while the red background indicates ground truth anomalous frames.

![](images/3a2837f425902af68cf1988b2088384071ed6642271f6be07656bc751fd6ce07.jpg)

![](images/3c1fb4747a5800a748a04589e4f9e2eed9a6e80ab32cb4b3094efd3328160923.jpg)

Predicted Frame  
![](images/4c669fc28d3d0fedf6a606c7a7784c24a4598fade8fe7b7b25c60baa7697863f.jpg)

![](images/9bc15eb20673aac59b04e853cee453f4b58537772c12ecf181746a72432a146b.jpg)

![](images/76eb82bfa25645e3a551c815d0e06c4d034a52b97dcbde48dd3b9497b3c99c27.jpg)

![](images/c5bc9c279e0e5e985375829c588ff3c479487c87fb6694aa4ab16b6312157e1c.jpg)

![](images/f00df5c27c80492f7730dce5eeb81cdf398def282f50d468754896cef7e65a81.jpg)  
(a)

![](images/c667da48e2165409301708910437f36bb4fc725b860d4cf981a8ba4aeb67e127.jpg)  
(b)

![](images/259d51f731632252df0f5b8d4ecf16fcd82f2796bdab4b87717c0b7c7000fbb1.jpg)  
(c)  
Fig. 4. Visualization of the actual frame, the corresponding predicted frame, and the pixel-wise error map during anomalous events across the (a) UCSD Ped2, (b) CUHK Avenue, (c) ShanghaiTech datasets.

![](images/4abdcdccaf1f9d4a8345c5e3a4a806b0e21cbbca4cb7124c9fd452790ea32fad.jpg)  
(a)

![](images/73ee24f8eeb4fbdedaa889296002e1cdc435eacf05f675e84c5b3b7a04e7ac92.jpg)  
(b)

![](images/d8556970c86cee55f699630bb4963e5270c11c0f33474c78a6f4cf030fa9beaf.jpg)  
(c)  
Fig. 5. Anomaly scores plotted over time for sample video clips from the (a) UCSD Ped2, (b) CUHK Avenue, and (c) ShanghaiTech datasets. Ground truth anomalous frames are highlighted by the red background.

## 4.6. Hyperparameter sensitivity analysis

As defined in Eq. (41), the final anomaly score is computed as a weighted linear combination of the normalized PSNR-based score and the normalized MS-SSIM-based score. The relative contribution of each component depends on the hyperparameter �, where the final anomaly score with $w = 0$ exclusively relies on MS-SSIM, and the score with $w = 1 . 0$ relies solely on PSNR. To investigate the effect of this hyperparameter on the anomaly detection capability of the selected configuration for $\mathbf { M S } ^ { 2 } \mathbf { C A S T }$ , which employs the multi-scale Pre-LN crossattention encoder along with SwinLSTM, we conducted a sensitivity analysis by varying � from 0 to 1 in increments of 0.1. The frame-level AUC scores for each value of the coefficient � are illustrated in Fig. 6 for the UCSD Ped2 (Fig. 6(a)), CUHK Avenue (Fig. 6(b)), and ShanghaiTech (Fig. 6(c)) datasets. The results demonstrate that the relationship between the frame-level AUC and � is highly dependent on the dataset, yielding different optimal values of � for each benchmark.

On the UCSD Ped2 dataset, as � increases from 0, the AUC monotonically increases from 93.9%, achieving its highest value of $9 6 . 1 \%$ at $w = 0 . 6$ . The AUC score remains unchanged at $w = 0 . 7$ ， $w = 0 . 8$ , and $w = 0 . 9$ before slightly decreasing to $9 6 . 0 \%$ at $w = 1 . 0$ . Similarly, on the CUHK Avenue dataset, the AUC exhibits a consistent upward trend, starting from 79.7% at $w = 0$ and reaching a maximum of $8 5 . 8 \%$ at $w = 0 . 9$ , followed by a slight decrease to 85.7% at $w = 1 . 0$ . Unlike the single-scene datasets, the AUC score on the ShanghaiTech dataset initially increases, reaching its maximum value at $w = 0 . 2$ and sustaining this score at $w = 0 . 3$ and $w = 0 . 4$ . After this point, further increments of � yield a monotonic decrease in the AUC score, ultimately reaching a value of 73.9% at $w = 1 . 0$

![](images/86e73db916dd553119f28e9f4047922fb4b852946bf02cf0a1c44513b40116a9.jpg)  
(a)

![](images/f21ac555822e8c01eb4f8faa18a3377050c2f69ef9d315a1812cf8f8e34f2d7e.jpg)  
(b)

![](images/29c65dd6c958d36c01b5bfebe4051f27a982fb427035f8e497c510fdc75e7d04.jpg)  
(c)  
Fig. 6. Frame-level AUC (%) of the selected configuration for $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ across different values of the coefficient �, evaluated on the (a) UCSD Ped2, (b) CUHK Avenue, and (c) ShanghaiTech datasets.

Generally, assigning a higher weight to MS-SSIM to determine the final anomaly score yields higher AUC scores on the ShanghaiTech dataset, whereas assigning a higher weight to PSNR results in the maximum AUC scores on the CUHK Avenue and UCSD Ped2 datasets. These dataset-dependent variations confirm the necessity of our dual-metric anomaly scoring mechanism, which provides the flexibility required to adapt to the specific spatial and temporal characteristics of different surveillance environments.

## 5. Conclusion

In this paper, we introduce $\mathrm { M S } ^ { 2 } \mathrm { C A S T }$ an end-to-end semi-supervised predictive VAD framework. This framework fuses the distinct hierarchical feature representations of a ViT and a CNN via a multi-scale cross-attention mechanism to encode spatial features. This spatial encoding strategy is paired with an efficient spatiotemporal module to capture motion dynamics without relying on computationally demanding deep optical flow networks. Furthermore, the objective function for this framework focuses on both pixel-wise accuracy and structural integrity while enforcing sharp edges in the predicted frames through spatial gradients. Our comprehensive ablation studies revealed several critical insights regarding architectural design choices, confirming the effectiveness of leveraging the multi-scale spatial cross-attention mechanism in the spatial encoder. We demonstrated that employing the multiscale Pre-LN cross-attention encoder along with a SwinLSTM spatiotemporal module yields competitive anomaly detection performance on the UCSD Ped2 and CUHK Avenue datasets while outperforming top-tier VAD methods on the ShanghaiTech dataset. This architectural configuration maintains peak GPU memory usage under 1 GB while exhibiting strong robustness across various types of additive noise. While the anomaly scoring currently exhibits dataset-dependent sensitivity to the hyperparameter �, future work will focus on exploring alternative metrics and formulating a unified, parameter-free anomaly score function.

## References

[1] J. Kim and K. Grauman, “Observe locally, infer globally: A space-time MRF for detecting abnormal activities with incremental updates,” in IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 2921–2928. doi: 10.1109/cvpr.2009.5206569.

[2] A. D. Giorno, J. A. Bagnell, and M. Hebert, “A discriminative framework for anomaly detection in large videos.” in Lecture Notes in Computer Science. pp. 334-349, 2016. doi: 10.1007/978-3-319-46454-1\_21.

[3] P. Wu, C. Pan, Y. Yan, G. Pang, Q. Yan, and P. Wang, “Deep learning for video anomaly detection: a review,” IEEE Transactions on Neural Networks and Learning Systems, vol. 37, no. 7, pp. 3010-3030, 2026, doi: 10.1109/tnnls.2025.3647892.

[4] D. Gong et al., “Memorizing normality to detect anomaly: memory-augmented deep autoencoder for unsupervised anomaly detection,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2019. doi: 10.1109/iccv.2019.00179.

[5] H.-J. Park, J. Noh, and B. Ham, “Learning memory-guided normality for anomaly detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 14360-14369,. doi: 10.1109/cvpr42600.2020.01438.

[6] X. Zhang, J. Fang, B. Yang, S. Chen, and B. Li, “Hybrid attention and motion constraint for anomaly detection in crowded scenes,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 5, pp. 2259-2274, 2023, doi: 10.1109/tcsvt.2022.3221622.

[7] I. J. Goodfellow et al., “Generative adversarial nets,” in Advances in Neural Information Processing Systems, vol. 27, 2014.

[8] M. Mathieu, C. Couprie, and Y. LeCun, “Deep multi-scale video prediction beyond mean square error,” in International Conference on Learning Representations (ICLR), 2016.

[9] W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection - A new baseline,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6536-6545,. doi: 10.1109/CVPR.2018.00684.

[10] Y. Tang, L. Zhao, S. Zhang, C. Gong, G. Li, and J. Yang, “Integrating prediction and reconstruction for anomaly detection,” Pattern Recognition Letters, vol. 129, pp. 123- 130, 2020, doi: 10.1016/j.patrec.2019.11.024.

[11] W. Luo, W. Liu, D. Lian, and S. Gao, “Future frame prediction network for video anomaly detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7505-7520, 2022, doi: 10.1109/tpami.2021.3129349.

[12] Y. Zhang, X. Nie, R. He, M. Chen, and Y. Yin, “Normality learning in multispace for video anomaly detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 9, pp. 3694-3706, 2021, doi: 10.1109/tcsvt.2020.3039798.

[13] Z. Sun, P. Wang, W. Zheng, and M. Zhang, “Dual GroupGAN: An unsupervised fourcompetitor (2V2) approach for video anomaly detection,” Pattern Recognition, vol. 153, p. 110500, 2024, doi: 10.1016/j.patcog.2024.110500.

[14] D. Li, X. Nie, R. Gong, X. Lin, and H. Yu, “Multi-branch GAN-based abnormal events detection via context learning in surveillance videos,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 5, pp. 3439-3450, 2024, doi: 10.1109/tcsvt.2023.3325451.

[15] K. Kurach, M. Lučić, X. Zhai, M. Michalski, and S. Gelly, “A Large-Scale Study on Regularization and Normalization in GANs,” in Proceedings of the 36th International Conference on Machine Learning, K. Chaudhuri and R. Salakhutdinov, Eds., in Proceedings of Machine Learning Research, vol. 97. PMLR, Jun. 2019, pp. 3581–3590. [Online]. Available: https://proceedings.mlr.press/v97/kurach19a.html

[16] Q. Ma, C. Wang, and X. Zhou, “CF2M-Net: Cross-feature fusion and memoryconstraint network for video anomaly detection,” Information Sciences, vol. 723, p. 122673, 2026, doi: 10.1016/j.ins.2025.122673.

[17] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox, “FlowNet 2.0: Evolution of optical flow estimation with deep networks,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. doi: 10.1109/cvpr.2017.179.

[18] Z. Teed and J. Deng, “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.” in Lecture Notes in Computer Science. pp. 402-419, 2020. doi: 10.1007/978-3-030-58536- 5\_24.

[19] T. M. Tran, D. C. Bui, T. V. Nguyen, and K. Nguyen, “Transformer-based spatiotemporal unsupervised traffic anomaly detection in aerial videos,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 9, pp. 8292-8309, 2024, doi: 10.1109/tcsvt.2024.3376399.

[20] H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss functions for image restoration with neural networks,” IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 47- 57, 2017, doi: 10.1109/tci.2016.2644865.

[21] H. Yuan, Z. Cai, H. Zhou, Y. Wang, and X. Chen, “TransAnomaly: Video anomaly detection using video vision transformer,” IEEE Access, vol. 9, pp. 123977-123986, 2021, doi: 10.1109/access.2021.3109102.

[22] M. Raghu, T. Unterthiner, S. Kornblith, C. Zhang, and A. Dosovitskiy, “Do vision transformers see like convolutional neural networks?,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 12116-12128,.

[23] S. Li and H. Tang, “Multimodal alignment and fusion: A survey,” International Journal of Computer Vision, vol. 134, no. 3, 2026, doi: 10.1007/s11263-025-02667-1.

[24] H. Tang, L. Shao, N. Sebe, and L. Gool, “Enhanced multi-scale cross-attention for person image generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3377-3393, 2025, doi: 10.1109/tpami.2025.3531220.

[25] S. Tang, C. Li, P. Zhang, and R. Tang, “SwinLSTM: Improving spatiotemporal prediction accuracy using Swin Transformer and LSTM,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 13424-13433,. doi: 10.1109/iccv51070.2023.01239.

[26] Y. Tang, P. Dong, Z. Tang, X. Chu, and J. Liang, “VMRNN: Integrating Vision Mamba and LSTM for efficient and accurate spatiotemporal forecasting,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 5663- 5673,. doi: 10.1109/cvprw63382.2024.00575.

[27] X. Shi, Z. Chen, W. Hao, D. Y. Yeung, W. K. Wong, and W. C. Woo, “Convolutional LSTM network: A machine learning approach for precipitation nowcasting,” in Advances in Neural Information Processing Systems, vol. 28, 2015.

[28] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, and Z. Zhang, “Swin Transformer: Hierarchical vision transformer using shifted windows,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9992-10002,. doi: 10.1109/iccv48922.2021.00986.

[29] Y. Liu et al., “Vmamba: Visual state space model,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 103031-103063,.

[30] P. Wu, J. Liu, and F. Shen, “A deep one-class neural network for anomalous event detection in complex scenes,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1-14, 2019, doi: 10.1109/tnnls.2019.2933554.

[31] A. K. Rai et al., “Video anomaly detection via spatio-temporal pseudo-anomaly generation : A unified approach,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3887-3899,. doi: 10.1109/cvprw63382.2024.00393.

[32] P. Wu, J. Liu, M. Li, Y. Sun, and F. Shen, “Fast sparse coding networks for anomaly detection in videos,” Pattern Recognition, vol. 107, p. 107515, 2020, doi: 10.1016/j.patcog.2020.107515.

[33] Z. Ning, Z. Wang, Y. Liu, J. Liu, and L. Song, “Memory-enhanced appearance-motion consistency framework for video anomaly detection,” Computer Communications, vol. 216, pp. 159-167, 2024, doi: 10.1016/j.comcom.2024.01.004.

[34] R. Cai, H. Zhang, W. Liu, S. Gao, and Z. Hao, “Appearance-motion memory consistency network for video anomaly detection,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 2, pp. 938-946, 2021, doi: 10.1609/aaai.v35i2.16177.

[35] C. Zach, T. Pock, and H. Bischof, “A duality based approach for realtime TV-L 1 optical flow.” in Lecture Notes in Computer Science. pp. 214-223, 2007. doi: 10.1007/978-3- 540-74936-3\_22.

[36] H. Rasheed, M. U. Khattak, M. Maaz, S. Khan, and F. S. Khan, “Fine-tuned CLIP models are efficient video learners,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6545–6554. doi: 10.1109/cvpr52729.2023.00633.

[37] T. Brox, A. Bruhn, N. Papenberg, and J. Weickert, “High accuracy optical flow estimation based on a theory for warping.” in Lecture Notes in Computer Science. pp. 25-36, 2004. doi: 10.1007/978-3-540-24673-2\_3.

[38] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in International Conference on Machine Learning, 2015, pp. 448-456,.

[39] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, “Learning temporal regularity in video sequences,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 733-742,. doi: 10.1109/cvpr.2016.86.

[40] W. Luo, W. Liu, and S. Gao, “Remembering history with convolutional LSTM for anomaly detection,” in 2017 IEEE International Conference on Multimedia and Expo (ICME), 2017, pp. 439-444,. doi: 10.1109/icme.2017.8019325.

[41] Z. Fang, J. T. Zhou, Y. Xiao, Y. Li, and F. Yang, “Multi-encoder towards effective anomaly detection in videos,” IEEE Transactions on Multimedia, vol. 23, pp. 4106- 4116, 2021, doi: 10.1109/tmm.2020.3037538.

[42] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2017. doi: 10.1109/iccv.2017.322.

[43] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, “Object detection in 20 years: A survey,” Proceedings of the IEEE, vol. 111, no. 3, pp. 1-20, 2023, doi: 10.1109/JPROC.2023.3238524.

[44] M. Ye, X. Peng, W. Gan, W. Wu, and Y. Qiao, “AnoPCN: Video anomaly detection via deep predictive coding network,” in Proceedings of the 27th ACM International Conference on Multimedia, 2019, pp. 1805-1813,. doi: 10.1145/3343031.3350899.

[45] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds., Cham: Springer International Publishing, 2015, pp. 234–241.

[46] Y. Li et al., “MViTv2: Improved multiscale vision transformers for classification and detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, doi: 10.1109/cvpr52688.2022.00476.

[47] Z. Liu, H. Mao, C. Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11966-11976,. doi: 10.1109/cvpr52688.2022.01167.

[48] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248-255,. doi: 10.1109/cvpr.2009.5206848.

[49] N. Parmar et al., “Image Transformer,” in Proceedings of the 35th International Conference on Machine Learning, J. Dy and A. Krause, Eds., in Proceedings of Machine Learning Research, vol. 80. PMLR, Jul. 2018, pp. 4055–4064. [Online]. Available: https://proceedings.mlr.press/v80/parmar18a.html

[50] A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2017/file/3f5ee243547dee91fbd053c1c 4a845aa-Paper.pdf

[51] W. Luo, Y. Li, R. Urtasun, and R. Zemel, “Understanding the effective receptive field in deep convolutional neural networks,” in Advances in Neural Information Processing Systems, vol. 29, 2016.

[52] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image quality assessment,” in The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003. doi: 10.1109/acssc.2003.1292216.

[53] V. Mahadevan, W. Li, V. Bhalodia, and N. Vasconcelos, “Anomaly detection in crowded scenes,” in IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2010. doi: 10.1109/cvpr.2010.5539872.

[54] C. Lu, J. Shi, and J. Jia, “Abnormal Event Detection at 150 FPS in MATLAB,” in 2013 IEEE International Conference on Computer Vision (ICCV), 2013. doi: 10.1109/iccv.2013.338.

[55] W. Luo, W. Liu, and S. Gao, “A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework,” in 2017 IEEE International Conference on Computer Vision (ICCV), 2017. doi: 10.1109/iccv.2017.45.

[56] J. Dai, Q. Li, H. Wang, and L. Liu, “Understanding images of surveillance devices in the wild,” Knowledge-based systems, vol. 284, pp. 111226-111226, 2024, doi: 10.1016/j.knosys.2023.111226.