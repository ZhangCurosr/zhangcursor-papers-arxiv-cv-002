# PCT-PROMPT: A PROMPT-GUIDED TRANSFORMER FRAMEWORK FOR DENSE PREDICTION TASKS IN POINT CLOUDS

Dejun Zhang China University of Geosciences zhangdejun@cug.edu.cn

Yanzi Bai Li Auto Inc baiyanzi123@cug.edu.cn

Yiqi Wu<sup>∗</sup> China University of Geosciences wuyq@cug.edu.cn

August 18, 2026

## ABSTRACT

Standard Transformers have proven effective in point cloud object classification, but their performance in dense prediction tasks within complex scenes is often hindered by weak prior assumptions. To address this challenge, we propose PCT-Prompt, a novel framework that enhances standard Trans formers by introducing a prompt-guided feature branch to improve performance in dense prediction tasks. The standard Transformer branch leverages pre-trained models for global feature extraction from point cloud data, serving as the backbone for processing high-level features. Meanwhile, the prompt-guided feature branch consists of two key components: a fine-grained feature extraction block that captures multi-scale geometric features using geometry-sensitive abstraction layer, along with the PnP-3D layer to integrate local context with global regularization. The second component, the prompt-refined feature learning block generates prompt tokens, which are subsequently refined through cross-attention mechanisms. Additionally, we introduce a prompt drop mechanism that pro gressively removes prompt information across Transformer layers, balancing local details and global consistency. Experimental results on the ShapeNetPart, S3DIS, and DALES datasets demonstrate that PCT-Prompt significantly improves the adaptability of standard Transformers to dense prediction tasks, achieving strong performance in real-world scenarios.

Keywords point cloud · Transformer architecture · dense prediction tasks · prompt tokens · pre-trained models

## 1 Introduction

The widespread use of point cloud data in autonomous driving and robotics has significantly driven the development of deep learning methods specifically designed for point clouds. Unlike traditional deep learning architectures [1, 2, 3, 4], Transformers have emerged as particularly effective, excelling in handling sets and offering unparalleled adaptability to point cloud data. Point cloud Transformers are particularly proficient at capturing both local features and complex global relationships inherent in point clouds, which enables them to deliver superior performance in key tasks such as classification [5], part segmentation [6], semantic segmentation [7] and registration [8]. The transformative advantages offered by Transformers have fueled innovation in point cloud data processing, driving notable advancements in both autonomous driving and robotics.

Transformer-based methods for point cloud processing often incorporate specific inductive biases, such as neighbor embedding, vector self-attention, and offset-attention. These biases are designed to enhance the perception of geometric structures and improve the ability to capture complex local features. Transformers that leverage these inductive biases are commonly referred to as variant Transformers, with notable examples including Point Transformer [9] and Point Cloud Transformer [10]. While these variant Transformers offer substantial improvements in dense prediction tasks, they may sacrifice generality and limit the effective representation of multi-modal features. In contrast, standard Transformers [11, 12, 13, 14, 15] provide several advantages, including ease of deployment, a flexible architecture, rich pre-training strategies, and the capacity to handle multi-modal data more effectively.

However, standard Transformers have several limitations: (1) Lack of inductive bias: without inductive biases, they require large amounts of training data for good generalization. These models [15, 11] are typically pre-trained on small datasets like ShapeNetPart [16], resulting in poor generalization. (2) Difficulty with multi-scale context: single-scale Transformers struggle to capture local context at different scales, which limits their performance in tasks involving varying object sizes. (3) Dependence on initial self-attention: standard Transformers rely on initial selfattention mechanisms, restricting their performance in dense prediction tasks. Fine-tuning methods [17, 18] show some improvement in classification and part segmentation but still fall short for real-world scene segmentation.

We present PCT-Prompt, a novel framework that combines a standard Transformer branch with a prompt-guided feature branch. This approach leverages the strengths of standard Transformers while improving their performance on dense prediction tasks. The standard Transformer branch includes patch embedding and multiple Transformer blocks, which can load various pre-trained weights, providing a foundation for processing point cloud data and extracting global features. In parallel, the prompt-guided feature branch introduces two key components to enhance performance on dense prediction tasks: (1) The Fine-grained Feature Extraction (FFE) block, which employs a hierarchical architecture to extract multi-scale geometric features using geometry-sensitive abstraction (GSA) layer, along with a plug-and-play layer, PnP-3D, that integrates local context fusion and global bilinear regularization. (2) The Prompt-refined Feature Learning (PFL) block, which includes a prompt generator that dynamically transforms multi-scale geometric features into fine-grained prompt tokens. These tokens are then refined by a prompt refiner, which iteratively integrates both global and local features through cross-attention and shared-weight MLPs. Additionally, a prompt drop mechanism progressively removes prompt information across Transformer layers, enabling the model to focus on local geometric details while preserving global semantic consistency.

To summarize, this work makes the following contributions:

• We propose PCT-Prompt, a framework combining standard Transformers with a prompt-guided feature branch for improved dense prediction tasks.

• We propose a Fine-grained Feature Extraction (FFE) block that leverages a hierarchical combination of the GSA and PnP-3D layers to capture multi-scale geometric features.

• We present a Prompt-refined Feature Learning (PFL) block that generates and refines prompt tokens, incorporating a prompt drop mechanism for balanced feature refinement.

• Experimental evaluations on ShapeNetPart, S3DIS, and DALES datasets show that PCT-Prompt significantly enhances the standard Transformer’s adaptability to dense prediction tasks.

The remainder of this article is organized as follows: Section 2 reviews relevant literature. Section 3 details the proposed PCT-Prompt framework, explaining its architecture and how the standard Transformer branch is enhanced with the prompt-guided feature branch for improved performance on dense prediction tasks. Section 4 describes the datasets, training setup, evaluation metrics, experimental comparisons with state-of-the-art methods, and ablation studies. Finally, Section 5 provides the conclusion of the article.

## 2 Related Work

In this section, we provide an overview of related works in three key areas: deep learning-based point cloud processing, the application of transformers to point cloud data, and the emerging field of prompt tuning. We highlight the latest advancements in each domain, focusing on their relevance to current research and their role in addressing complex 3D perception challenges.

## 2.1 Deep Learning for Point Cloud

Point clouds are essential in various domains, including autonomous driving, robotics, and computer vision, due to their ability to represent 3D environments. They are commonly used in tasks such as classification [19], segmentation [20], 3D reconstruction [21], scene flow estimation [22, 23], and cross-view geo-localization [24], which are crucial for understanding and interacting with 3D scenes. These tasks support applications like object recognition, scene reconstruction, and environmental mapping. Consequently, the development of deep learning methods for efficient and accurate point cloud processing has become a significant research focus.

Deep learning methods for point cloud processing can be broadly classified into three main categories: projection-based, voxel-based, and point-based approaches. Projection-based methods, such as PointCLIP [25], and PointCLIPV2 [26], project point clouds onto multiple 2D views and utilize image feature extraction techniques to construct 3D representations. While these methods are efficient, they may suffer from occlusion and geometric loss. Voxel-based methods, including VoxNet [27], OctNet [28], and O-cnn [29], convert point clouds into voxel grids and apply 3D convolutions for feature extraction. These methods offer an effective way to handle large-scale data but can be memory-intensive and may lose resolution due to voxelization. Point-based methods, such as PointNet [30], PointNet++ [1], KPConv[31], Point Transformer [9], and PCT [10], process raw point cloud data directly, preserving its geometric structure. PointNet [30] uses point-wise MLPs with global pooling to extract features, while PointNet++ [1] employs a hierarchical approach for fine-grained local feature capture. KPConv [31] uses a continuous space point kernel to perform feature extraction, mimicking the behavior of convolutions on 3D data.

While projection-based and voxel-based methods offer simplified point cloud processing, they often face challenges like occlusion, geometric distortion, or high memory demands. In contrast, point-based methods retain the original geometry of point clouds, allowing for more precise and intuitive feature extraction. Building on this foundation, this paper adopts a point-based approach to extend the standard Transformer, enhancing its ability to capture multi-scale and detailed geometric features for dense prediction tasks.

## 2.2 Transformer for Point Cloud

Research on point cloud transformers can be categorized into two main directions: 1) Variant Transformers with inductive biases tailored to point clouds [10, 9, 32, 33]; 2) Standard Transformers adhering to the foundational goal of minimizing inductive bias [13, 12, 11, 14, 15].

The first focuses on variant transformers, which incorporate inductive biases tailored to point clouds. For example, PCT [10] enhances the model’s geometric awareness by introducing inductive biases such as neighborhood embedding and offset attention, alongside the standard attention mechanism. Point Transformer [9] integrates vector self-attention layers and position encoding to refine how the Transformer handles point clouds. Swin3D [33] and Patchformer [32] further optimize computational efficiency by limiting attention mechanisms to local regions. Although these methods show strong performance in point cloud tasks, the careful design of inductive biases introduces significant time costs, which can somewhat deviate from the original aim of minimizing inductive bias in Transformer models.

The second direction focuses on standard transformers, which aim to minimize inductive bias while leveraging pretraining strategies to enhance feature extraction capabilities. Point-BERT [11] extends the BERT [34] self-supervised learning approach to point clouds, achieving state-of-the-art results in classification and part segmentation through masked point modeling. MaskPoint [15] introduces a discriminative pre-training framework that converts fuzzy masked point reconstruction into a more discriminative task, capturing rich features. Point-MAE [12] employs random masking of point cloud patches for self-supervised pre-training, reconstructing unmasked patches to extract advanced latent features. ACT [14] and ReCon [13] further demonstrate the effectiveness of cross-modal learning, leveraging image and NLP Transformers for point cloud pre-training.

Minimizing inductive bias in standard Transformers has attracted considerable attention due to their ability to generalize across tasks. Building on this foundation, our study seeks to extend the pre-trained standard Transformer framework, enhancing its capabilities to tackle dense prediction tasks in complex scenes.

## 2.3 Prompt Tuning

Prompt tuning [35], originally developed for adapting pre-trained models to downstream tasks in natural language processing (NLP) [36], has since been successfully extended to image, video, and multimodal domains. In NLP, GPT-3 demonstrated strong performance across various tasks through prompt-based few-shot learning, accelerating the use of prompt tuning. In image-related tasks, CLIP [37] leverages contrastive learning to map images and text into a shared embedding space, enabling zero-shot learning through prompts for image classification and retrieval. Similarly, ALIGN [38] integrates vision and language information, optimizing multimodal tasks through prompts, thus improving image-text retrieval. VQGAN-CLIP [39] combines the Generative Adversarial Network (VQGAN) [40] with CLIP [37], utilizing a pre-trained image-text joint encoder and natural language prompts for high-quality image generation and semantic editing.

In point cloud Transformer research, prompt tuning has also gained traction. For instance, Point-PEFT [17] combines point-granular prompts with a geometry-aware adapter, improving classification performance while maintaining a small parameter size via fine-tuning the Transformer pre-trained model. Point-MAE [12] employs a masked autoencoder (MAE) method, using prompts to help the model reconstruct missing point cloud information, demonstrating strong results in point cloud reconstruction. IDPT [18] introduces dynamic prompt tuning, applied to 3D point cloud Transformers, and achieves good performance in classification and simple part segmentation tasks.

However, existing prompt tuning methods have not fully realized the potential of standard Transformers in dense prediction tasks like 3D object detection and semantic segmentation, leading to performance bottlenecks. Standard Transformers lack inductive biases and struggle with multi-scale local information, limiting their effectiveness in these tasks. To address this, this paper proposes a novel prompt tuning framework that enhances standard Transformers’ performance in dense prediction tasks through refined feature extraction and prompt-refined learning blocks.

## 3 PCT-Prompt

As shown in Fig. 1, the PCT-Prompt framework consists of two main branches: the standard Transformer branch and the prompt-guided feature branch. (1) The standard Transformer branch is capable of loading weights from various pre-trained models. It takes sub-clouds, which are obtained using Farthest Point Sampling (FPS) and K-Nearest Neighbor (KNN) algorithms, as input. By employing patch embedding, it constructs input sequences, which are then processed by a backbone comprising N Transformer blocks of the same scale. (2) The prompt-guided feature branch is composed of two key components: the Fine-grained Feature Extraction (FFE) block and the Prompt-refined Feature Learning (PFL) block. The FFE block employs a hierarchical structure to extract detailed geometric features from the point cloud, organizing them into a multi-scale feature pyramid. The PFL block, consisting of N blocks similar to the Transformer backbone, enables bidirectional feature interaction with the Transformer block, introducing point cloud-specific prompt features into the backbone and facilitating updates to the multi-scale feature pyramid for enhanced dense prediction tasks.

![](images/b435bea427c0e6c0bf16b576191662a84d9cf3d37425bea3e1e2539429d6003d.jpg)  
Figure 1: Overall architecture of PCT-Prompt.

## 3.1 Transformer Branch

Although existing pre-trained Transformers, such as Point-BERT [11], Point-MAE [12], ACT [14], and ReCon [13], employ different pre-training strategies, they share a common Transformer backbone and a unified pre-training parameter schema. To effectively utilize the weights from multiple pre-training strategies, the proposed PCT-Prompt adopts the same standard Transformer backbone for point cloud feature extraction. The following section provides a detailed overview of the standard Transformer backbone used in PCT-Prompt.

## 3.1.1 Patch Embedding

Given an input point cloud $\mathbf { P } \in \mathbb { R } ^ { n \times 3 }$ , the FPS algorithm is employed to select S center points, denoted as $\hat { \mathbf { P } }$ . For each center point, its local neighborhood is constructed by aggregating K nearest points using the KNN algorithm, resulting in S local sub-clouds. To mitigate the influence of global coordinate attributes and ensure the invariance of local sub-clouds, each sub-cloud is translated by subtracting its corresponding center point. Subsequently, a lightweight PointNet [30] is utilized to project these localized sub-clouds into point cloud embeddings. This process can be formally expressed as:

$$
\mathbf { F } _ { \mathrm { e m b } } = \mathbf { M a x P o o l } \left( \mathbf { M L P } \left( \mathbf { K N N } \left( \mathbf { F P S } \left( \mathbf { P } \right) \right) - \widehat { \mathbf { P } } \right) \right) .\tag{1}
$$

here, MaxPool(·) denotes channel-wise max pooling; MLP(·) comprises 1D convolutions (1 × 1 convolutions) followed by Batch Normalization and ReLU activation functions; KNN(·) denotes the KNN algorithm; and FPS(·) denotes for the FPS algorithm, which ensures uniform point distribution.

Given the pivotal role of positional embeddings in the Transformer architecture, we incorporate positional information into the S patch-wise features. To achieve this, we utilize a multilayer perceptron (MLP) comprising two linear layers with a GELU activation function, which transforms the S center coordinates into D-dimensional vectors, thereby generating the positional embeddings $\mathbf { F } _ { \mathrm { p o s } } .$ . Subsequently, the point cloud embeddings are combined with the positional embeddings to form the input to the standard Transformer, denoted as $\mathbf { F } _ { \mathrm { s t } } ^ { 0 }$ , as expressed in the following equation:

$$
\mathbf { F } _ { \mathrm { s t } } ^ { 0 } = \mathbf { F } _ { \mathrm { p o s } } \oplus \mathbf { F } _ { \mathrm { e m b } } ,\tag{2}
$$

where ⊕ denotes the concatenation operator.

## 3.1.2 Transformer Backbone

The standard Transformer backbone, consisting of L uniform-scale Transformer layers, is organized into N blocks. The input to the i-th Transformer block is denoted as $\mathbf { F } _ { \mathrm { s t } } ^ { i - 1 } \in \mathbb { R } ^ { S \times D }$

To enhance the Transformer backbone’s ability to learn features beneficial for dense prediction tasks, we introduce the prompt tokens learned by the prompt-guided feature branch into the backbone. Specifically, during the i-th feature interaction between the prompt generator and the Transformer, the prompt tokens $\mathbf { F } _ { \mathrm { p r o m p t } } ^ { i ^ { - } }$ are combined with the feature tokens $\mathbf { F } _ { \mathrm { s t } } ^ { i - 1 }$ and passed together into the i-th Transformer block. The output of the i-th Transformer block is given by:

$$
\left[ \widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i } ; \widehat { \mathbf { F } } _ { \mathrm { p r o m p t } } ^ { i } \right] = \operatorname { B l o c k } \left( \left[ \mathbf { F } _ { \mathrm { s t } } ^ { i - 1 } ; \mathbf { F } _ { \mathrm { p r o m p t } } ^ { i } \right] \right) ,\tag{3}
$$

where Block(·) represents the operation of the Transformer block in the backbone. Analogous to the class token in a standard Transformer, the prompt tokens not only encapsulate the global features of the point cloud but also guide the Transformer backbone by directing its focus toward local fine-grained details.

Following the configuration of the pre-trained standard Transformer, we set L = 12, with the FPS sampling number S and the KNN parameter K adaptively adjusted based on the specific requirements of the dense prediction task at hand. The number of Transformer blocks, N, is chosen to be 6, based on empirical observations regarding the impact of the number of interactions on model performance.

## 3.2 Prompt-guided Feature Branch

The prompt-guided feature branch improves dense prediction tasks by extracting multi-scale geometric features through the Fine-grained Feature Extraction (FFE) block and refining them with prompt tokens in the Prompt-refined Feature Learning (PFL) block. Next, we provide a detailed description of these two blocks.

## 3.2.1 Fine-grained Feature Extraction

The proposed Fine-grained Feature Extraction (FFE) block is illustrated in Fig. 2. Initially, the input point cloud $\mathbf { P } \in \mathbb { R } ^ { n \times 3 }$ is encoded into feature representations $\mathbf { F } \in \mathbb { R } ^ { n \times d }$ using a Multi-Layer Perceptron (MLP). A hierarchical structure with four levels is designed, where each level leverages a geometry-sensitive abstraction (GSA) layer followed by a PnP-3D [41] layer. These components extract point cloud features at multiple scales, specifically $C = \{ n / 4 , \stackrel { . } { n } / 1 6 , n / 6 4 , n / 2 5 6 \}$ . Finally, the extracted features are aggregated into a unified feature pyramid.

![](images/038ac0282965cb265e8703c8f0a1f7c4ff72ada7728505bfd82fcf740c2c465f.jpg)  
Figure 2: The structure of the fine-grained feature extraction block.

For the j-th feature extraction layer, FPS and KNN operations are applied to the input coordinates $\mathbf { P } ^ { j - 1 } \in \mathbb { R } ^ { C ^ { j - 1 } } ;$ ×3 and features $\mathbf { F } ^ { j - 1 } \in \mathbb { R } ^ { C ^ { j - 1 } \times D ^ { j - 1 } }$ , selecting center points and their corresponding neighborhoods. Relative positional encoding [42] is introduced to capture geometric relationships between the center points and neighbors. To enhance robustness, particularly in handling sparse and irregular structures, a learnable geometric affine mechanism is adopted. The output of this stage is defined as:

$$
\left[ \mathbf { P } ^ { j } ; \mathbf { F } _ { \mathrm { G S A } } ^ { j } \right] = \mathbf { G S A } \left( \left[ \mathbf { P } ^ { j - 1 } ; \mathbf { F } ^ { j - 1 } \right] \right) ,\tag{4}
$$

where GSA(·) denotes the geometry-sensitive abstraction layer. The features are further refined by PnP-3D, a lightweight module that integrates local context fusion and global bilinear regularization. The local branch captures fine-grained geometric structures via neighborhood graph aggregation (e.g., EdgeConv), while the global branch enhances feature consistency through bilinear interactions across channel and point dimensions. Finally, an MLP is applied to fuse the outputs of the GSA and PnP-3D layers, generating the point cloud features at the j-th scale:

$$
\mathbf { F } ^ { j } = \mathrm { M L P } \left( \mathbf { F } _ { \mathrm { G S A } } ^ { j } \oplus \mathrm { P n P 3 D } \left( \left[ \mathbf { P } ^ { j } ; \mathbf { F } _ { \mathrm { G S A } } ^ { j } \right] \right) \right) ,\tag{5}
$$

where PnP3D(·) represents the PnP-3D layer, and ⊕ represents the concatenation operation.

Finally, the outputs from different scales are concatenated and transformed into a unified multi-scale feature pyramid $\mathbf { F } _ { \mathrm { p y r a m i d } } ^ { 0 } \in \mathbb { R } ^ { \left( \frac { n } { 4 } , \frac { n } { 1 6 } , \frac { n } { 6 4 } , \frac { n } { 2 5 6 } \right) \times D }$

## 3.2.2 Prompt-refined Feature Learning

As shown in Fig. 3, the feature interaction between the Prompt-refined Feature Learning (PFL) block and the Transformer block is bidirectional, consisting of both sending and feedback phases. In the sending phase, the prompt generator injects multi-scale prompt tokens into the Transformer backbone, thereby augmenting its capacity to construct detailed feature representations. In the feedback phase, the Transformer block returns global feature information to the prompt refiner, enabling the feature pyramid to integrate high-level semantic knowledge and produce more refined multi-scale representations. A prompt drop mechanism is also introduced to progressively remove prompt tokens layer by layer, ensuring its influence extends effectively to deeper Transformer blocks and enhances feature modeling for dense segmentation tasks.

![](images/29d1f1bb6c3750751490d1535cc8465905f9ebd3766653e94695f6a8c42d0626.jpg)  
Figure 3: The structure of the prompt-refined feature learning block.

Prompt Generator. The prompt generator is designed to convert the multi-scale geometric features extracted in the previous stage into task-specific prompt tokens, which are subsequently passed into the Transformer backbone. This process consists of two main steps:

First, adaptive max pooling is applied to extract global representations from the multi-scale feature pyramid $\mathbf { F } _ { \mathrm { p y r a m i d } } ^ { i - 1 } .$ which are subsequently combined with learnable positional encodings $\mathbf { E } _ { \mathrm { p o s } } ^ { i }$ to yield the global feature representation $\mathbf { F } _ { \mathrm { g l o b a l } } ^ { i } .$ , as expressed in the following equation:

$$
\mathbf { F } _ { \mathrm { g l o b a l } } ^ { i } = \mathrm { A d a p t i v e M a x P o o l 1 D } \left( \mathrm { M L P } \left( \mathbf { F } _ { \mathrm { p y r a m i d } } ^ { i - 1 } \right) \right) + \mathbf { F } _ { \mathrm { p o s } } ^ { i } .\tag{6}
$$

here, MLP(·) comprises 1D convolutions (1 × 1 convolutions) followed by Batch Normalization and ReLU activation functions; AdaptiveMaxPool1D(·) denotes the adaptive max pooling operation, which compresses the spatial dimensions of the multi-scale input features from $\{ \mathbb { R } ^ { \frac { n } { 4 } \times D } , \mathbb { R } ^ { \frac { n } { 1 6 } \times \dot { D } } , \mathbb { R } ^ { \frac { n } { 6 4 } \times D } , \mathbb { R } ^ { \frac { n } { 2 5 6 } \times D } \}$ to $\{ \mathbb { R } ^ { 1 \times \hat { D } } , \mathbb { R } ^ { 1 \times D } , \mathbb { R } ^ { \hat { 1 } \times D } , \mathbb { R } ^ { 1 \times D } \}$ thereby producing compact global descriptors from each scale.

Second, the global features are refined and normalized through a dynamic processing module, formulated as:

$$
\mathbf { F } _ { \mathrm { p r o m p t } } ^ { i } = \mathrm { L a y e r N o r m } \left( \mathrm { L e a k y R e L U } \left( \mathrm { C o n v 1 D } \left( \mathbf { F } _ { \mathrm { g l o b a l } } ^ { i } \right) \right) \right) .\tag{7}
$$

Here, Conv1D(·) denotes a one-dimensional convolution $( 1 \times 1$ convolution) that performs linear transformation and feature reconstruction on local representations; LeakyReLU(·) is a nonlinear activation function; and LayerNorm(·) applies normalization across the channel dimension for each point feature.

As illustrated in Eq. 3, the prompt tokens $\mathbf { F } _ { \mathrm { p r o m p t } } ^ { i }$ are integrated with the feature tokens $\mathbf { F } _ { \mathrm { s t } } ^ { i - 1 }$ extracted from the (i−1)-th Transformer block, serving as the input tokens to the i-th Transformer block.

Prompt Refiner. The feature interaction from the Transformer block to the prompt refiner utilizes cross-attention to refine features at each scale within the feature pyramid. Additionally, a shared-weight Multi-Layer Perceptron (MLP) is employed to efficiently propagate information across scales.

Specifically, the output feature $[ \widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i } ; \widehat { \mathbf { F } } _ { \mathrm { p r o m p t } } ^ { i } ]$ from the i-th Transformer block serves as the key and value, while the multi-scale feature pyramid $\mathbf { F } _ { \mathrm { p y r a m i d } } ^ { i - 1 }$ acts as the query. The integration of global features into the multi-scale feature pyramid through cross-attention is formulated as:

$$
\widehat { \mathbf { F } } _ { \mathrm { p r a m i d } } ^ { i } = \mathrm { S o f t m a x } \left( \frac { \mathrm { L N } \left( \mathbf { F } _ { \mathrm { p y r a m i d } } ^ { i - 1 } \right) \cdot \mathrm { L N } \left( \left[ \widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i } ; \widehat { \mathbf { F } } _ { \mathrm { p r o m p l } } ^ { i } \right] \right) ^ { T } } { \sqrt { d } } \right) \cdot \mathrm { L N } \left( \left[ \widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i } ; \widehat { \mathbf { F } } _ { \mathrm { p r o m p l } } ^ { i } \right] \right) .\tag{8}
$$

Here, Softmax(·) normalizes the attention scores into a probability distribution, LN(·) denotes layer normalization, and d represents the dimensionality of the query and key vectors, which corresponds to the feature dimension D introduced previously.

The shared-weight MLP further refines the propagation of features across scales, as described by:

$$
\mathbf { F } _ { \mathrm { p y r a m i d } } ^ { i } = \mathrm { M L P } _ { \mathrm { s h a r e d } } \left( \widehat { \mathbf { F } } _ { \mathrm { p y r a m i d } } ^ { i } \right) ,\tag{9}
$$

where $\mathrm { M L P } _ { \mathrm { s h a r e d } } ( \cdot )$ represents the shared-weight multi-layer perceptron. The reorganized multi-scale feature pyramid results in a more robust representation, thereby enhancing the Transformer’s ability to capture detailed point cloud features and adapt effectively to dense prediction tasks.

Prompt Drop. As the number of Transformer blocks increases, each subsequent block’s ability to effectively utilize prompt features diminishes, which restricts feature modeling in dense segmentation tasks. To address this, we introduce the prompt drop mechanism, ensuring that each Transformer block leverages prompt information effectively while preserving global semantic consistency and refining local geometric details.

As shown in Fig. 3, the output features of the i-th Transformer block consist of two components: prompt tokens $\widehat { \mathbf { F } } _ { \mathrm { p r o m p t } } ^ { i }$ and feature tokens $\widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i }$ . The prompt drop mechanism removes the prompt tokens, with the final output features of the i-th Transformer block defined as:

$$
\mathbf { F } _ { \mathrm { s t } } ^ { i } = [ \widehat { \mathbf { F } } _ { \mathrm { s t } } ^ { i } ; \widehat { \mathbf { F } } _ { \mathrm { p r o m p t } } ^ { i } ] \setminus \widehat { \mathbf { F } } _ { \mathrm { p r o m p t } } ^ { i } ,\tag{10}
$$

where \ represents the removal operation. By progressively eliminating prompt tokens, the mechanism ensures their influence is propagated to deeper Transformer blocks, thereby enhancing the network’s feature modeling capability for dense segmentation.

## 3.3 Feature Fusion and Dense Prediction.

The fusion process between the output features of the standard Transformer branch and the prompt-guided feature branch is designed to integrate global semantic context with local geometric details. Specifically, the output features from the Transformer branch are first upsampled to four different resolutions using the point feature upsampling strategy, generating a set of multi-scale features that are resolution-aligned with the output feature from the prompt-guided feature branch. The corresponding multi-scale features from the standard Transformer branch and the prompt-guided feature branch are then combined via element-wise addition to obtain the fused representation. This fused representation is subsequently passed to a decoder module, which is structurally analogous to the segmentation head in PointNet++ [1]. The decoder comprises a series of feature propagation layers that integrate interpolation and MLPs to progressively recover fine-grained point-level semantic predictions.

## 3.4 Loss Function

We adopt two loss functions tailored to the characteristics of the datasets: cross-entropy loss and weighted cross-entropy loss. For the ShapeNetPart dataset, which features a balanced class distribution, we employ standard cross-entropy loss to quantify the divergence between the predicted and ground truth class distributions. It is defined as:

$$
\mathcal { L } _ { \mathrm { c e } } = - \sum _ { i = 1 } ^ { C } y _ { i } \log ( \hat { y } _ { i } ) ,\tag{11}
$$

where $y _ { i }$ denotes the ground truth label for class i, and $\hat { y } _ { i }$ represents the predicted probability for class i. For the S3DIS and DALES datasets, which are characterized by significant class imbalance, we utilize weighted cross-entropy loss to address this challenge. The loss function is expressed as:

$$
\mathcal { L } _ { \mathrm { w c e } } = - \sum _ { i = 1 } ^ { C } w _ { i } \cdot y _ { i } \log ( \hat { y } _ { i } ) ,\tag{12}
$$

where w is the weight assigned to class $i ,$ calculated as the ratio of the number of points in class $i \left( f _ { i } \right)$ to the total number of points across all classes $\begin{array} { r } { ( N ) , \mathrm { i } . \mathrm { e } . , w _ { i } = \frac { N } { f _ { i } } } \end{array}$ . This weighting mechanism assigns higher importance to less frequent classes, thereby mitigating the bias towards dominant classes and enhancing the model’s performance on imbalanced datasets.

## 4 Experiments

## 4.1 Experiments Setting

## 4.1.1 ShapeNetPart

The ShapeNetPart dataset [16] contains 16,881 objects, spanning 16 categories and 50 part labels, with each instance consisting of 2 to 6 parts. The model setup strictly follows that of Point-BERT. For each object, 2,048 points are randomly sampled as input. The standard Transformer backbone is configured with a sampling count $S \ : = \ : 1 2 8$ corresponding to 128 point cloud patches, and the KNN parameter K is set to 32. The Prompt reduces the number of GSA layers to 3, with the downsampling ratio set to $\mathbf { C } = \left. { \frac { n } { 8 } } , { \frac { n } { 1 6 } } , { \frac { n } { 3 2 } } \right.$ , while maintaining the same KNN parameters a the backbone.

For the training setup, we use cross-entropy loss (Eq. 11) to measure the discrepancy between the predicted probability distribution for each point and the ground truth labels. The model is trained for 300 epochs on an NVIDIA RTX 4090 with 24GB of memory using the AdamW optimizer, with a learning rate of 0.0005 and a batch size of 6. The quantitative evaluation metrics include class mIoU, instance mIoU, and the IoU for each individual class.

## 4.1.2 S3DIS

We validate the semantic segmentation performance of PCT-Prompt on the real-world indoor scene S3DIS dataset [43]. S3DIS includes six large indoor areas from three different buildings, with a total of 273 million points annotated with 13 categories (ceiling, floor, table, etc.). Area 5, the most challenging region, is used for testing, while the remaining areas are used for training.

For model configuration, due to the large point cloud scene data, we randomly sample 12,000 points as the input for each scene. The standard Transformer backbone is set with a sampling number $S = 2 5 6 .$ , corresponding to 256 point cloud patches, and the KNN parameter K = 32. In the Prompt, the number of GSA layers is set to $^ { 4 , }$ and the downsampling ratios are $\begin{array} { r } { \mathbf { C } = \{ \frac { n } { 4 } , \frac { \dot { n } } { 1 6 } , \frac { n } { 6 4 } , \frac { n } { 2 5 6 } \} } \end{array}$ , with KNN set to 64.

For training, due to the significant class imbalance in the real-world dataset, we use a weighted cross-entropy loss (Eq. 12) to address the class imbalance issue. The class weights are calculated based on the frequency of each class in the input point clouds. The model is trained for 150 epochs using the AdamW optimizer with a learning rate of 0.0005 and a batch size of 6 on an NVIDIA RTX 4090 with 24GB of memory. The quantitative evaluation metrics include mIoU, mAcc, OA, and IoU for each class.

## 4.1.3 DALES

DALES is a 10 km<sup>2</sup> airborne LiDAR dataset that includes 40 urban and rural scenes with a total of 500 million points, of which 12 scenes are used for evaluation.

For model configuration, due to the large number of points in each frame, each scene is divided into multiple grids of size $4 0 \times 4 0$ based on the area. 20,000 points from each point cloud scene are selected as input. The standard Transformer backbone is set with a sampling number $S = 3 \bar { 8 } 4$ , corresponding to 384 point cloud patches, and the KNN parameter $K = 3 2$ . In the Prompt, the number of GSA layers is set to 4, and the downsampling ratios are $\begin{array} { r } { { \bf { C } } = \left\{ \frac { n } { 4 } , \frac { n } { 1 6 } , \frac { n } { 6 4 } , \frac { n } { 2 5 6 } \right\} } \end{array}$ , with KNN set to 64.

For training, weighted cross-entropy loss (Eq. 12) is used to address class imbalance, where the class weights are derived from the frequency of each class in the labels. The model is trained for 250 epochs using the AdamW optimizer with a learning rate of 0.0005 and a batch size of 1 on an NVIDIA RTX 4090 with 24GB of memory. The quantitative evaluation metrics include mIoU and IoU for each class.

## 4.2 Quantitative Analysis

## 4.2.1 Performance on ShapeNetPart

We leveraged pre-trained weights from Point-BERT [11] for both the standard Transformer (Point-BERT) and PCT-Prompt, conducting part segmentation on the ShapeNetPart dataset. The experimental results, presented in Table 1, reveal that, when using identical pre-trained parameters for the standard Transformer backbone, PCT-Prompt enhances part segmentation performance. Specifically, it improves the category mIoU and instance mIoU by 0.9% and 0.6%, respectively, by incorporating the prompt structure. Moreover, PCT-Prompt effectively reduces the performance gap between the standard Transformer and its variants, such as PCT [10] and pointCAT [44]. These findings demonstrate that PCT-Prompt introduces an effective task-specific prompting mechanism, substantially benefiting the standard Transformer architecture.

Table 1: Object part segmentation results on the ShapeNetPart dataset. “T”, “S”, and “V” represent traditional neural network models, standard Transformers, and variant Transformers, respectively.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Type</td><td rowspan="2">ins. mIoU</td><td rowspan="2">cls. mIoU</td><td rowspan="2">aero</td><td rowspan="2">bag</td><td rowspan="2">cap</td><td rowspan="2">car</td><td rowspan="2">chr.</td><td rowspan="2">e.ph.</td><td rowspan="2">gtar</td><td rowspan="2">knf.</td><td rowspan="2">lmp.</td><td rowspan="2">ltp.</td><td rowspan="2">m.b.</td><td rowspan="2">mg</td><td rowspan="2">pstl</td><td rowspan="2">s.b. skte</td><td rowspan="2"></td><td rowspan="2">tbl.</td></tr><tr><td></td></tr><tr><td>PointNet[30]</td><td>T</td><td>83.7</td><td>80.4</td><td>83.4</td><td>78.7</td><td>82.5</td><td>74.9</td><td>89.6</td><td>73.0</td><td>91.5</td><td>85.9</td><td>80.8</td><td>95.3</td><td>65.2</td><td>93.0</td><td>81.2</td><td>57.9</td><td>72.8</td><td>80.6</td></tr><tr><td>PointNet++[1]</td><td>T</td><td>85.1</td><td>81.9</td><td>82.4</td><td>79.0</td><td>87.7</td><td>77.3</td><td>90.8</td><td>71.8</td><td>91.0</td><td>85.9</td><td>83.7</td><td>95.3</td><td>71.6</td><td>94.1</td><td>81.3</td><td>58.7</td><td>76.4</td><td>82.6</td></tr><tr><td>PointCNN[3]</td><td>T</td><td>86.1</td><td>84.6</td><td>84.1</td><td>86.5</td><td>86.0</td><td>80.8</td><td>90.6</td><td>79.7</td><td>92.3</td><td>88.4</td><td>85.3</td><td>96.1</td><td>77.2</td><td>95.2</td><td>84.2</td><td>64.2</td><td>80.0</td><td>83.0</td></tr><tr><td>DGCNN[2]</td><td>T</td><td>85.2</td><td>82.3</td><td>84.0</td><td>83.4</td><td>86.7</td><td>77.8</td><td>90.6</td><td>74.7</td><td>91.2</td><td>87.5</td><td>82.8</td><td>95.7</td><td>66.3</td><td>94.9</td><td>81.1</td><td>63.5</td><td>74.5</td><td>82.6</td></tr><tr><td>PointMLP[45]</td><td>T</td><td>86.1</td><td>84.6</td><td>83.5</td><td>83.4</td><td>87.5</td><td>80.5</td><td>90.3</td><td>78.2</td><td>92.2</td><td>88.1</td><td>82.6</td><td>96.2</td><td>77.5</td><td>95.8</td><td>85.4</td><td>64.6</td><td>83.3</td><td>84.3</td></tr><tr><td>PointASNL[46]</td><td>T</td><td>86.1</td><td></td><td>84.1</td><td>84.7</td><td>87.9</td><td>79.7</td><td>92.2</td><td>73.7</td><td>91.0</td><td>87.2</td><td>84.2</td><td>95.8</td><td>74.4</td><td>95.2</td><td>81.0</td><td>63.0</td><td>76.3</td><td>83.2</td></tr><tr><td>PointCAT[44]</td><td>V</td><td>86.0</td><td>84.4</td><td>83.0</td><td>83.8</td><td>90.1</td><td>79.8</td><td>90.2</td><td>83.4</td><td>91.8</td><td>87.8</td><td>82.5</td><td>95.9</td><td>76.1</td><td>95.4</td><td>84.9</td><td>68.5</td><td>83.1</td><td>84.1</td></tr><tr><td>PCT[10]</td><td>V</td><td>86.4</td><td></td><td>85.0</td><td>89.0</td><td>82.4</td><td>81.2</td><td>91.9</td><td>71.5</td><td>91.3</td><td>88.1</td><td>86.3</td><td>95.8</td><td>64.6</td><td>95.8</td><td>83.6</td><td>62.2</td><td>77.6</td><td>83.7</td></tr><tr><td>PT[9]</td><td>V</td><td>86.6</td><td>83.7</td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Point-BERT[11]</td><td>SS</td><td>85.6</td><td>84.1</td><td>84.3</td><td>84.8</td><td>88.0</td><td>79.8</td><td>91.0</td><td>81.7</td><td>91.6</td><td>87.9</td><td>85.2</td><td>95.6</td><td>75.6</td><td>94.7</td><td>84.3</td><td>63.4</td><td>76.3</td><td>81.5</td></tr><tr><td>PCT-Prompt</td><td></td><td>86.2</td><td>85.0</td><td>83.8</td><td>85.1</td><td>90.5</td><td>80.0</td><td>90.5</td><td>83.6</td><td>92.1</td><td>88.0</td><td>83.5</td><td>96.0</td><td>77.1</td><td>94.7</td><td>85.6</td><td>64.4</td><td>81.0</td><td>83.9</td></tr></table>

## 4.2.2 Performance on S3DIS

We initialize both the standard Transformer and PCT-Prompt with the same pre-trained parameters and evaluate their performance on the real-world indoor dataset S3DIS for semantic segmentation. As shown in Table 2, when both models are initialized with Point-BERT pre-trained weights, PCT-Prompt outperforms the standard Transformer, achieving a 3.1% improvement in mAcc and a 7.4% improvement in mIoU. Furthermore, PCT-Prompt surpasses several state-of-the-art point cloud Transformer variants, including Superpoint Transformer [47], PCT [10], pointCAT [44], PatchFormer [32], and pointTransformer [9]. These results not only validate the design principles underlying PCT-Prompt but also highlight the effectiveness of the prompt mechanism in extending the standard Transformer to a diverse range of downstream tasks, thereby enabling more effective utilization of the feature representation capabilities of the general backbone.

Furthermore, we visually compare the results of the standard Transformer (Point-BERT) and PCT-Prompt across three scenes in Area 5. As depicted in Fig. 4, PCT-Prompt demonstrates a clear fine-grained enhancement effect on the Point-BERT. These results show that the PCT-Prompt effectively extends the standard Transformer to downstream tasks, proves the rationality of the PCT-Prompt design, and can deeply explore the feature extraction potential of the general point cloud Transformer backbone.

## 4.2.3 Performance on DALES

We evaluated both the standard Transformer and PCT-Prompt models using the same pre-trained parameters on the real-world DALES dataset for semantic segmentation. As shown in Table 3, PCT-Prompt outperforms the standard

Ground Truth  
Input  
Table 2: Semantic segmentation results on S3DIS dataset, tested on Area 5. \*means we report the best results among public codebases and our own reproduction.
<table><tr><td>Methods</td><td>Type</td><td>mIoU</td><td>mAcc</td><td>OA</td><td>ceil.</td><td>flr</td><td>wl</td><td>bm</td><td>col.</td><td>wnd.</td><td>dr</td><td>tbl.</td><td>chr</td><td>sf.</td><td>b.c.</td><td>brd</td><td>cltr.</td></tr><tr><td>PointNet[30]</td><td>T</td><td>41.1</td><td>49.0</td><td></td><td>88.8</td><td>97.3</td><td>69.8</td><td>0.1</td><td>3.9</td><td>46.3</td><td>10.8</td><td>59.0</td><td>52.6</td><td>5.9</td><td>40.3</td><td>26.4</td><td>33.2</td></tr><tr><td>PointCNN[3]</td><td>T</td><td>57.3</td><td>63.9</td><td>85.9</td><td>92.3</td><td>98.2</td><td>79.4</td><td>0.0</td><td>17.6</td><td>22.8</td><td>62.1</td><td>74.4</td><td>80.6</td><td>31.7</td><td>66.7</td><td>62.1</td><td>56.7</td></tr><tr><td>SPG[48]</td><td>T</td><td>58.0</td><td>66.5</td><td>86.4</td><td>89.4</td><td>96.9</td><td>78.1</td><td>0.0</td><td>42.8</td><td>48.9</td><td>61.6</td><td>84.7</td><td>75.4</td><td>69.8</td><td>52.6</td><td>2.1</td><td>52.2</td></tr><tr><td>PointWeb[49]</td><td>T</td><td>60.3</td><td>66.6</td><td>87.0</td><td>92.0</td><td>98.5</td><td>79.4</td><td>0.0</td><td>21.1</td><td>59.7</td><td>34.8</td><td>76.3</td><td>88.3</td><td>46.9</td><td>69.3</td><td>64.9</td><td>52.5</td></tr><tr><td>KPConv[31]</td><td>T</td><td>67.1</td><td>72.8</td><td></td><td>92.8</td><td>97.3</td><td>82.4</td><td>0.0</td><td>23.9</td><td>58.0</td><td>69.0</td><td>81.5</td><td>91.0</td><td>75.4</td><td>75.3</td><td>66.7</td><td>58.9</td></tr><tr><td>PAT[50]</td><td>V</td><td>60.1</td><td>70.8</td><td></td><td>93.0</td><td>98.5</td><td>72.3</td><td>1.0</td><td>41.5</td><td>85.1</td><td>38.2</td><td>57.7</td><td>83.6</td><td>48.1</td><td>67.0</td><td>61.3</td><td>33.6</td></tr><tr><td>SPT[47]</td><td>V</td><td>68.9</td><td>77.3</td><td>89.5</td><td>91.5</td><td>98.2</td><td>81.4</td><td>0.0</td><td>23.3</td><td>65.3</td><td>40.0</td><td>75.5</td><td>87.7</td><td>58.5</td><td>67.8</td><td>65.6</td><td>49.4</td></tr><tr><td>PatchFormer[32]</td><td>V</td><td>67.3</td><td></td><td></td><td>91.8</td><td>98.7</td><td>86.2</td><td>0.0</td><td>34.1</td><td>48.9</td><td>62.4</td><td>81.6</td><td>89.8</td><td>47.2</td><td>74.9</td><td>74.4</td><td>58.6</td></tr><tr><td>PT*[9]</td><td>V</td><td>70.0</td><td>76.8</td><td>90.4</td><td>94.0</td><td>98.5</td><td>86.3</td><td>0.0</td><td>38.0</td><td>63.4</td><td>74.3</td><td>89.1</td><td>82.4</td><td>74.3</td><td>80.2</td><td>76.0</td><td>59.3</td></tr><tr><td>PCT[10]</td><td>V V</td><td>61.3</td><td>67.7</td><td></td><td>92.5</td><td>98.4</td><td>80.6</td><td>0.0</td><td>19.3</td><td>61.6</td><td>48.0</td><td>76.6</td><td>85.2</td><td>46.2</td><td>67.7</td><td>67.9</td><td>52.3</td></tr><tr><td>PointCAT[44]</td><td></td><td>64.0</td><td>71.0</td><td>88.2</td><td>94.2</td><td>98.3</td><td>80.5</td><td>0.0</td><td>18.6</td><td>55.5</td><td>58.9</td><td>77.2</td><td>88.0</td><td>64.8</td><td>72.2</td><td>68.9</td><td>55.4</td></tr><tr><td>Point-BERT[11]</td><td>S</td><td>63.5</td><td>75.7</td><td>85.0</td><td>91.3</td><td>92.3</td><td>73.1</td><td>0.0</td><td>33.9</td><td>65.6</td><td>60.4</td><td>76.5</td><td>82.7</td><td>86.8</td><td>64.0</td><td>41.7</td><td>43.0</td></tr><tr><td>PCT-Prompt</td><td>S</td><td>70.9</td><td>78.8</td><td>90.3</td><td>95.2</td><td>98.1</td><td>83.6</td><td>0.0</td><td>63.4</td><td>82.2</td><td>71.1</td><td>78.5</td><td>86.8</td><td>76.8</td><td>70.8</td><td>59.2</td><td>55.4</td></tr></table>

![](images/4674068fc4781d6999f48eb2b409eef923dee14f840942017280236350a1ac5c.jpg)

![](images/6337b12b2ec10add29e246c3e9d37c1821942e0724ad88fb3bf361f11b0c65c1.jpg)

![](images/1ec6510844a656e75419e2d0a9fb6f557d6582cdba1643577def59caeed823d5.jpg)

![](images/549d8f037c56c742c5ce9ff8f2893b68df71dda2e5f67108d8241bd7d2064886.jpg)

![](images/f4235318dfc440e4ff0263db8733a728861ed842a8bf557b7acd2b2c3bbee14b.jpg)

![](images/8fee697a73dc193cff5f49b2a2eb83f82539e6eb60ef94781fac54235a5965e9.jpg)

![](images/57d1a51a99cecea406db7e266e2f0f843e79e6b5572441b82f838b6ee1d01741.jpg)

![](images/9ad6ea120b30462539ae4a9c0b1dfee421e7a35460f0ca6873e1df45c5292d59.jpg)

![](images/6f992fe3a9f8923f7e96fd4e28626e3d6871945feb2a0fdbe4ef2304632b19b6.jpg)

![](images/da8d27270f5be4c50190836b4f21fa209e9835e64e242c06bf4d5e8dee097779.jpg)

![](images/e242494600af99f6bb60a9976e38a47eeb6720ea54d037162762678d9c792bbc.jpg)  
Point-BERT

![](images/aa0894f0422b8a7484cd8c809d96365ed24b9bd8363ce5f1bf751190a7591689.jpg)  
PCT-Prompt  
Figure 4: Visualizing the comparison between Point-BERT and PCT-Prompt on the S3DIS Dataset.

Transformer by achieving a 6.5% improvement in mIoU. This performance boost in more complex and diverse realworld scenarios can be attributed to PCT-Prompt’s effective integration of multi-scale geometric features and its ability to complement the standard Transformer architecture, thereby enhancing the overall semantic segmentation performance.

Table 3: Comparison of semantic segmentation performance on the DALES dataset.
<table><tr><td>Methods</td><td>Type</td><td>mIoU</td><td>mAcc</td><td>OA</td><td>Ground</td><td>Buildings</td><td>Cars</td><td>Trucks</td><td>Poles</td><td>Power Lines</td><td>Fences</td><td>Vegetation</td></tr><tr><td>PointNet++[1]</td><td>T</td><td>68.3</td><td></td><td>95.7</td><td>94.1</td><td>89.1</td><td>75.4</td><td>30.3</td><td>40.0</td><td>79.9</td><td>46.2</td><td>91.2</td></tr><tr><td>SPG[48]</td><td>T</td><td>60.6</td><td></td><td>95.5</td><td>94.7</td><td>93.4</td><td>62.9</td><td>18.7</td><td>28.5</td><td>65.2</td><td>33.6</td><td>87.9</td></tr><tr><td>PointCNN[3]</td><td>T</td><td>58.4</td><td></td><td>97.2</td><td>97.5</td><td>95.7</td><td>40.6</td><td>4.8</td><td>57.6</td><td>26.7</td><td>52.6</td><td>91.7</td></tr><tr><td>SPT[47]</td><td>V</td><td>79.6</td><td></td><td>97.5</td><td>96.7</td><td>93.1</td><td>86.1</td><td>52.4</td><td>94.0</td><td>52.7</td><td>65.3</td><td>96.7</td></tr><tr><td>Point-BERT[11]</td><td>S</td><td>72.0</td><td>79.6</td><td>96.0</td><td></td><td></td><td></td><td>27.8</td><td>93.3</td><td></td><td></td><td></td></tr><tr><td>PCT-Prompt</td><td>S</td><td>78.5</td><td>84.9</td><td>97.3</td><td>92.9 98.8</td><td>91.9 96.8</td><td>63.3 91.8</td><td>45.9</td><td>97.5</td><td>49.0 68.9</td><td>65.4 76.6</td><td>93.1 96.9</td></tr></table>

## 4.3 Ablation Studies

## 4.3.1 Components of PCT-Prompt

The purpose of the ablation study on the PCT-Prompt component is to assess the individual contributions of each block. As shown in Table 4, PCT-Prompt consists of three primary components: the Transformer backbone, the FFE block, and the PFL block. The PFL block is further divided into the generator, refiner, and drop mechanisms. Using only the Transformer yields a baseline mIoU of 63.5, while using the FFE block alone results in a significantly lower mIoU of 55.6, indicating that FFE block requires the Transformer backbone for effective feature modeling. When FFE block is combined with the Transformer backbone using the simplest interaction scheme (i.e., direct feature addition), the mIoU improves by 2.0%, suggesting its complementary role. Adding either the generator or the refiner leads to additiona gains, with the refiner providing slightly better performance. Combining both results in further improvement, and incorporating the drop mechanism yields the best overall performance, confirming the effectiveness of the complete design. These results demonstrate that each block contributes meaningfully, and their integration enhances performance in dense point cloud segmentation tasks.

Table 4: Component ablation study of PCT-Prompt on the S3DIS.
<table><tr><td rowspan="2">Transformer</td><td rowspan="2">FFE</td><td colspan="3">PFL</td><td rowspan="2">mIoU</td><td rowspan="2">mAcc</td></tr><tr><td>Generator</td><td>Refiner</td><td>Drop</td></tr><tr><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>63.5</td><td>75.7</td></tr><tr><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td><td>55.6</td><td>65.7</td></tr><tr><td>√</td><td>V</td><td>X</td><td>X</td><td>×</td><td>65.5</td><td>75.8</td></tr><tr><td>V</td><td>L</td><td>√</td><td>×</td><td>×</td><td>67.2</td><td>76.0</td></tr><tr><td>V</td><td>L</td><td>×</td><td>√</td><td>×</td><td>66.7</td><td>76.1</td></tr><tr><td>V</td><td>V</td><td>√</td><td>V</td><td>×</td><td>69.0</td><td>77.5</td></tr><tr><td>V</td><td>V</td><td>V</td><td>×</td><td>√</td><td>68.3</td><td>77.0</td></tr><tr><td>V</td><td>V</td><td>V</td><td>√</td><td>√</td><td>70.9</td><td>78.8</td></tr></table>

## 4.3.2 Effects of prompt Frequency

To refine the feature representation capabilities, PCT-Prompt divides the Transformer backbone into N blocks and constructs N corresponding Prompt-refined Feature Learning (PFL) blocks. This multi-block structure facilitates feature interactions, improving the model’s ability to capture complex features. To evaluate the impact of interaction frequency N, we conducted experiments with N values set to 0, 1, 2, 4, 6, and 8, keeping training configurations constant to determine the optimal N.

Table 5 shows the semantic segmentation performance on the S3DIS dataset for different interaction frequencies. When N = 0, PCT-Prompt is equivalent to the standard Transformer, achieving an mIoU of 63.5%. As the interaction frequency increases, performance improves, peaking at $N = 6 ,$ , where the mIoU reaches 70.9%. This demonstrates that increasing the frequency of interactions improves the fusion quality between the PFL block and the Transformer, with N = 6 being the optimal value. These results highlight the importance of frequent feature interactions in improving PCT-Prompt’s performance.

Table 5: Quantitative comparisons of different numbers of interactions on the S3DIS.
<table><tr><td>Frequency</td><td>mIoU</td><td>mAcc</td></tr><tr><td>N = 0</td><td>63.5</td><td>75.7</td></tr><tr><td>N = 1</td><td>65.9</td><td>76.1</td></tr><tr><td> $N = 2$ </td><td>68.4</td><td>76.0</td></tr><tr><td>N = 4</td><td>68.9</td><td>76.2</td></tr><tr><td> $N = 6$ </td><td>70.9</td><td>78.8</td></tr><tr><td> $N = 8$ </td><td>67.4</td><td>75.5</td></tr></table>

## 4.3.3 Effects of Different Pre-training Models

To assess the generalization capability of PCT-Prompt across various pre-trained models in dense prediction tasks, we evaluated its performance with different pre-trained weights. The pre-trained models considered in this study include ACT [14], Point-MAE [12], MaskPoint [15], Point-BERT [11], and ReCon [13].

Table 6: Quantitative comparisons among various pre-trained models.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Pre-trained Models</td><td rowspan=1 colspan=1>mIoU  mAcc</td></tr><tr><td rowspan=2 colspan=1>standardTransformer</td><td rowspan=2 colspan=1>ACT[14]Point-MAE[12]MaskPoint[15]Point-BERT[11]ReCon[13]</td><td rowspan=1 colspan=1>72.1    62.9</td></tr><tr><td rowspan=1 colspan=1>63.1    73.263.1    72.164.4   72.763.5   75.764.8   73.3</td></tr><tr><td rowspan=1 colspan=1>PCT-Prompt</td><td rowspan=1 colspan=1>ACT[14]Point-MAE[12]MaskPoint[15]Point-BERT[11]ReCon[13]</td><td rowspan=1 colspan=1>64.2   73.765.2   75.465.1    73.967.1    73.570.9   78.866.1    77.6</td></tr></table>

As shown in Table 6, the second and eighth rows represent the performance of the standard Transformer and PCT-Prompt without pre-trained weights, respectively. The third to seventh rows correspond to the standard Transformer with various pre-trained weights, while the ninth to final rows show the performance of PCT-Prompt under the same pre-trained weight configurations. The comparative analysis reveals the following key insights: (1) The integration of prompt-based features consistently improves the performance of the standard Transformer. (2) All pre-trained models contribute positively to the performance of PCT-Prompt across dense prediction tasks. (3) Among the pre-trained models tested, Point-BERT leads to the best performance when applied to PCT-Prompt.

These results highlight the robustness and generalizability of PCT-Prompt in leveraging diverse pre-trained models to enhance the adaptability and performance of the standard Transformer across various dense prediction tasks. Furthermore, our experimental findings conclusively demonstrate that the proposed PCT-Prompt framework not only boosts existing standard Transformer architectures for dense prediction but also holds the potential to further amplify performance as more advanced standard Transformers are developed.

## 5 Conclusion

This paper presents the Point Cloud Transformer Prompt (PCT-Prompt), a novel framework that enhances the capability of standard Transformers for dense prediction tasks in point cloud data. PCT-Prompt augments the feature representation of the standard Transformer without altering its core architecture, by integrating a pre-training-free feature extraction module and a prompt-guided feature learning mechanism. This framework improves model performance in dense prediction tasks by effectively incorporating multi-scale geometric features and dynamically refined prompt tokens, which interact with the Transformer backbone to iteratively enhance the feature pyramid. Moreover, PCT-Prompt is designed as a flexible and generalizable framework that can seamlessly load weights from various pre-trained models, thereby offering enhanced adaptability across diverse datasets. Through extensive experiments on segmentation tasks, we demonstrate that PCT-Prompt effectively reduces the performance gap between standard and variant Transformers, expanding the task applicability of standard Transformers and improving their effectiveness in real-world scenarios.

## References

[1] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017.

[2] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019.

[3] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. Advances in neural information processing systems, 31, 2018.

[4] Jaesung Choe, Chunghyun Park, Francois Rameau, Jaesik Park, and In So Kweon. Pointmixer: Mlp-mixer for point cloud understanding. In European Conference on Computer Vision, pages 620–640. Springer, 2022.

[5] Jun Sun, Junbo Zhang, Xuesong Gao, Mantao Wang, Dinghua Ou, Xiaobo Wu, and Dejun Zhang. Fusing spatial attention with spectral-channel attention mechanism for hyperspectral image classification via encoder–decoder networks. Remote Sensing, 14(9):1968, 2022.

[6] Jinyoung Park, Sanghyeok Lee, Sihyeon Kim, Yunyang Xiong, and Hyunwoo J Kim. Self-positioning point-based transformer for point cloud understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21814–21823, 2023.

[7] Dejun Zhang, Fazhi He, Zhigang Tu, Lu Zou, and Yilin Chen. Pointwise geometric and semantic learning network on 3d point clouds. Integrated Computer-Aided Engineering, 27(1):57–75, 2020.

[8] Xiaobo Hu, Dejun Zhang, Jinzhi Chen, Yiqi Wu, and Yilin Chen. Nrtnet: An unsupervised method for 3d non-rigid point cloud registration based on transformer. Sensors, 22(14):5128, 2022.

[9] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16259–16268, 2021.

[10] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. Pct: Point cloud transformer. Computational Visual Media, 7:187–199, 2021.

[11] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19313–19322, 2022.

[12] Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In European conference on computer vision, pages 604–621. Springer, 2022.

[13] Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining. arXiv preprint arXiv:2302.02318, 2023.

[14] Runpei Dong, Zekun Qi, Linfeng Zhang, Junbo Zhang, Jianjian Sun, Zheng Ge, Li Yi, and Kaisheng Ma. Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning? arXiv preprint arXiv:2212.08320, 2022.

[15] Haotian Liu, Mu Cai, and Yong Jae Lee. Masked discrimination for self-supervised learning on point clouds. In European Conference on Computer Vision, pages 657–675. Springer, 2022.

[16] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[17] Yiwen Tang, Ray Zhang, Zoey Guo, Xianzheng Ma, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Point-peft: Parameter-efficient fine-tuning for 3d pre-trained models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 5171–5179, 2024.

[18] Yaohua Zha, Jinpeng Wang, Tao Dai, Bin Chen, Zhi Wang, and Shu-Tao Xia. Instance-aware dynamic prompt tuning for pre-trained point cloud models. arXiv preprint arXiv:2304.07221, 2023.

[19] Yiqi Wu, Huachao Wu, Ronglei Hu, Yilin Chen, and Dejun Zhang. Multimodal 3d few-shot classification via gaussian mixture discriminant analysis. Computer Graphics Forum, 44(7):e70268, 2025.

[20] Dejun Zhang, Shifeng Xu, Yanzi Bai, Yiqi Wu, and Jun Liu. Sam-zero3d: Extending segment anything to zero shot 3d scene segmentation via iterative global–local interaction. IEEE Transactions on Circuits and Systemsfor Video Technology, 36(7):9137–9149, 2026.

[21] Xuefeng Tan, Dejun Zhang, Long Tian, Yiqi Wu, and Yilin Chen. Coarse-to-fine pipeline for 3d wireframe reconstruction from point cloud. Computers & Graphics, 106:288–298, 2022.

[22] Dejun Zhang, Mian Zhang, Xuefeng Tan, and Jun Liu. Bridging the domain gap in scene flow estimation via hierarchical smoothness refinement. ACM Transactions on Multimedia Computing, Communications and Applications, 20(8):1–21, 2024.

[23] Xiaohu Yan, Mian Zhang, Xuefeng Tan, Yiqi Wu, and Dejun Zhang. Flowst-net: Tackling non-uniform spatial and temporal distributions for scene flow estimation in point clouds. Neurocomputing, 619:129183, 2025.

[24] Shifeng Xu, Jun Sun, Xujie Long, Jun Liu, and Dejun Zhang. Weather-aware multi-granularity representation learning for cross-view geo-localization under adverse conditions. Pattern Recognition, page 114445, 2026.

[25] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8552–8562, 2022.

[26] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. Pointclip v2: Prompting clip and gpt for powerful 3d open-world learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2639–2650, 2023.

[27] Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In 2015 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 922–928. IEEE, 2015.

[28] Gernot Riegler, Ali Osman Ulusoy, and Andreas Geiger. Octnet: Learning deep 3d representations at high resolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3577–3586, 2017.

[29] Peng-Shuai Wang, Yang Liu, Yu-Xiao Guo, Chun-Yu Sun, and Xin Tong. O-cnn: Octree-based convolutional neural networks for 3d shape analysis. ACM Transactions On Graphics (TOG), 36(4):1–11, 2017.

[30] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings ofthe IEEE conference on computer vision andpattern recognition, pages 652–660, 2017.

[31] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, and Leonidas J Guibas. Kpconv: Flexible and deformable convolution for point clouds. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6411–6420, 2019.

[32] Cheng Zhang, Haocheng Wan, Xinyi Shen, and Zizhao Wu. Patchformer: An efficient point transformer with patch attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11799–11808, 2022.

[33] Yu-Qi Yang, Yu-Xiao Guo, Jian-Yu Xiong, Yang Liu, Hao Pan, Peng-Shuai Wang, Xin Tong, and Baining Guo. Swin3d: A pretrained transformer backbone for 3d indoor scene understanding. Computational Visual Media, 11(1):83–101, 2025.

[34] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

[35] Yusheng Su, Xiaozhi Wang, Yujia Qin, Chi-Min Chan, Yankai Lin, Huadong Wang, Kaiyue Wen, Zhiyuan Liu, Peng Li, Juanzi Li, et al. On transferability of prompt tuning for natural language processing. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3949–3969, 2022.

[36] Dejun Zhang, Mingbo Hong, Lu Zou, Fei Han, Fazhi He, Zhigang Tu, and Yafeng Ren. Attention pooling-based bidirectional gated recurrent units model for sentimental classification. International Journal ofComputational Intelligence Systems, 12(2):723–732, 2019.

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[38] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021.

[39] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance. In European Conference on Computer Vision, pages 88–105. Springer, 2022.

[40] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

[41] Shi Qiu, Saeed Anwar, and Nick Barnes. Pnp-3d: A plug-and-play for 3d point clouds. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):1312–1319, 2021.

[42] Renrui Zhang, Liuhui Wang, Yali Wang, Peng Gao, Hongsheng Li, and Jianbo Shi. Parameter is not all you need: Starting from non-parametric networks for 3d point cloud analysis. arXiv preprint arXiv:2303.08134, 2023.

[43] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1534–1543, 2016.

[44] Xincheng Yang, Mingze Jin, Weiji He, and Qian Chen. Pointcat: Cross-attention transformer for point cloud. arXiv preprint arXiv:2304.03012, 2023.

[45] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. arXiv preprint arXiv:2202.07123, 2022.

[46] Xu Yan, Chaoda Zheng, Zhen Li, Sheng Wang, and Shuguang Cui. Pointasnl: Robust point clouds processing using nonlocal neural networks with adaptive sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5589–5598, 2020.

[47] Jiahao Sun, Chunmei Qing, Junpeng Tan, and Xiangmin Xu. Superpoint transformer for 3d scene instance segmentation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 2393–2401, 2023.

[48] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 4558–4567, 2018.

[49] Hengshuang Zhao, Li Jiang, Chi-Wing Fu, and Jiaya Jia. Pointweb: Enhancing local neighborhood features for point cloud processing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5565–5573, 2019.

[50] Jiancheng Yang, Qiang Zhang, Bingbing Ni, Linguo Li, Jinxian Liu, Mengdie Zhou, and Qi Tian. Modeling point clouds with self-attention and gumbel subset sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3323–3332, 2019.