# MARS-CLIP: MULTI-RESOLUTION AND ATTENTION REFINED ZERO-SHOT IMAGE SEGMENTATION

Nagito Saito, Shintaro Ito, Koichi Ito, and Takafumi Aoki

Graduate School of Information Sciences, Tohoku University, Japan.

## ABSTRACT

Contrastive Language-Image Pre-training (CLIP) has demonstrated impressive capabilities in zero-shot transfer but often struggles with dense prediction tasks due to low spatial resolution and the loss of structural information. To address these limitations, we propose MARS-CLIP (Multi-resolution and Attention Refined Segmentation for CLIP), a novel framework for zero-shot semantic segmentation. Our approach introduces two key strategies: (i) a multi-resolution feature extraction module that fuses local fine-grained features with global context to overcome input resolution constraints, and (ii) an attention refinement mechanism that injects spatial and color biases from intermediate layers into the final self-attention block to accurately restore object boundaries. A set of experiments on six public datasets demonstrates that MARS-CLIP significantly outperforms state-of-the-art methods.

Index Terms— zero-shot semantic segmentation, CLIP, visionlanguage models, multi-resolution, attention mechanism

## 1. INTRODUCTION

Image segmentation, which performs pixel-level object identification and classification, serves as a foundational technology for autonomous driving [1], medical image analysis [2,3], and quality control [4, 5]. In recent years, deep learning-based methods [6–8] have become the major approach due to their robustness and high accuracy; however, they require large-scale datasets with dense pixellevel annotations. Consequently, extending these methods to recognize unseen classes beyond the training data requires additional annotation and model retraining, resulting in prohibitive time and labor costs.

To address this challenge, zero-shot image segmentation methods [9–11], which can recognize unseen classes based on text descriptions, have attracted significant attention. Most of these methods leverage Contrastive Language-Image Pre-Training (CLIP) [12] to achieve segmentation by exploiting the similarity between image and text in a shared feature space. However, CLIP is inherently designed to capture global representations of an entire image and is not optimized for preserving local, pixel-level details. Therefore, applying CLIP to dense prediction tasks presents two fundamental challenges that remain to be solved. The first challenge is the insufficient spatial resolution. Since the feature maps output by the CLIP image encoder are of a fixed low resolution, existing methods such as MaskCLIP [9] and SCLIP [13] often fail to detect tiny objects or accurately delineate complex boundaries. The second challenge is the loss of spatial structural information. Although SCLIP [13] improved spatial consistency by introducing a self-attention mechanism based on Key-Key similarity in the final layer, semantic abstraction in deep ViT layers inevitably dilutes the original shape information. Spatial layouts preserved in intermediate layers and lowlevel features like edges and colors are not fully utilized, often leading to ambiguous segmentation results.

In this paper, we propose MARS-CLIP (Multi-resolution and Attention Refined Segmentation for CLIP), a zero-shot segmentation framework capable of accurately recognizing fine image details through high-definition feature extraction by multi-resolution inputs and attention map refinement using low-level features. The main contributions of this work are summarized as follows: First, we introduce a multi-resolution strategy. By partitioning the input image into local grid patches for encoding, we circumvent input resolution limitations. These detailed local features are then adaptively fused with global features extracted from the entire image to obtain highresolution feature maps that preserve context. Second, we propose a refined attention mechanism by bias injection. In contrast to the standard Key-Key correlation used in SCLIP [13], we inject spatial biases from intermediate layers and color affinity biases, which capture sharp object boundaries, into the attention mechanism. Furthermore, by eliminating residual connections and feed-forward networks in the final layer, we reduce noise and achieve a balance between semantic consistency and spatial fidelity. We demonstrate the effectiveness of our proposed method through a set of experiments on PASCAL VOC 2012 [14], PASCAL Context [15], ADE20K [16], Cityscapes [17], COCO-Object [18], and COCO-Stuff [19], showing significant improvements over state-of-the-art methods.

## 2. RELATED WORK

In this section, we give a brief overview of zero-shot image segmentation utilizing CLIP, focusing on recent approaches centered on attention mechanism improvements and spatial structure preservation.

## 2.1. Contrastive Language-Image Pre-training (CLIP)

CLIP [12] is a vision-language model trained on 400 million imagetext pairs, achieving high generalization performance by embedding images and texts into a common feature space. While CLIP demonstrates excellent capabilities in image-level zero-shot classification, its training process focuses on global feature alignment, making it difficult to apply directly to dense prediction tasks like image segmentation at the pixel level.

## 2.2. Zero-Shot Semantic Segmentation

Early zero-shot segmentation methods [20, 21] had limited ability to handle unseen classes due to insufficient general alignment between visual and textual features. Since the advent of CLIP, research applying its strong zero-shot capabilities to segmentation has accelerated. A representative method, MaskCLIP [9], achieved segmentation by modifying the last attention layer to extract pixel-wise features, but the resulting prediction masks contained significant noise.

![](images/2436809723dd9da4b63ba273254a018384f1980b177057ba01fe19f4375be205.jpg)  
Fig. 1. Overview of MARS-CLIP. Global and local features are fused by a multi-resolution strategy. The final layer (highlighted in orange) injects spatial and color biases to refine boundaries, as detailed in Fig. 2.

To improve pixel-level feature correspondence, approaches modifying CLIP’s attention mechanism itself have become mainstream. CLIP Surgery [10] pointed out that standard Query-Key attention causes spatial redundancy and improved feature distinctiveness by introducing Value-Value attention. SCLIP [13] revisited the selfattention mechanism and proposed correlative self-attention based on Query-Query and Key-Key similarities, resolving spatial feature entanglement. Furthermore, ClearCLIP [22] identified residual connections in the final ViT layer as a source of segmentation noise and successfully sharpened masks by removing them. ResCLIP [23] aims for further accuracy improvement by introducing a residual attention mechanism. While these methods improve semantic consistency by manipulating $\mathrm { C L I P } ^ { \prime } \mathrm { s }$ internal attention mechanisms, the constraint of fixed input resolution remains, leaving challenges in recognizing fine structures and complex boundaries. In addition to attention mechanism improvements, attempts to reinforce spatial consistency have also been made. NACLIP [24] imposes smoothing constraints on attention maps between neighboring tokens, while OPMapper [25] reduces local ambiguity by integrating global context information. PnP-OVSS [26] adopts a method of refining masks through an iterative inference process. Meanwhile, approaches utilizing external foundation models have also been proposed. Proxy-CLIP [27] utilizes DINO’s local features, and FreeDA [28] leverages Stable Diffusion’s generative capabilities to expand feature representation. Although these methods using external knowledge show high performance, they require running multiple models in parallel, making increased computational cost unavoidable. In this paper, we solve the conventional issues of insufficient resolution and missing boundary information by combining CLIP’s internal features with multi-resolution inputs, without relying on external models, thereby maintaining computational efficiency.

## 3. MARS-CLIP

In this paper, we propose MARS-CLIP, a zero-shot image segmentation method designed to improve pixel-level dense prediction accuracy while maintaining the high generalization capability of CLIP. Fig. 1 illustrates the overview of our proposed framework. MARS-CLIP consists of three steps. The first step is feature extraction by multi-resolution inputs, where high-resolution feature maps integrating local details and global context are generated by partitioning the input image. The second step involves refining the attention mechanism using low-level features. We restore object boundary consistency by injecting spatial information from intermediate layers and color information as biases into the final attention layer. The final step generates segmentation masks by calculating the similarity between the obtained image features and text features.

## 3.1. Feature Extraction with Multi-resolution Images

Since the CLIP image encoder $E _ { i m g }$ typically has a fixed input size, e.g., 224 224 pixels, directly feeding high-resolution images results in the loss of fine structural details due to downsampling. To address this issue, we adopt a multi-resolution strategy that integrates local fine-grained features with global context features. Specifically, given an input image $\mathbf { I } \in \mathbb { R } ^ { 3 \times H \times W }$ , we first apply reflective padding to generate $\mathbf { I } _ { p a d }$ such that its dimensions are multiples of the crop size $S _ { c r o p } .$ Next, $\mathbf { I } _ { p a d }$ is divided into an $N \times M$ grid of local regions, and each region is resized to the encoder’s input resolution $K \times K$ Since typically $S _ { c r o p } < K .$ , this process corresponds to zooming into local regions, enabling the extraction of fine-grained features. The feature maps $\mathbf { F } _ { l o c a l } ^ { ( n , m ) } = E _ { i m g } ( \mathbf { x } _ { n , m } )$ extracted from the image patch $_ { \mathbf { x } _ { n , m } }$ at grid position $( n , m )$ are recombined according to their original spatial arrangement to form a single high-resolution feature map $\mathbf { F } _ { l o c a l } .$ . However, since processing local regions independently may result in the loss of global context, we also resize the entire image I and feed it into the encoder to extract a global feature map $\mathbf { F } _ { g l o b a l }$ . The final refined feature map ${ \bf F } _ { r e f i n e d }$ is obtained by integrating these features as follows:

$$
\mathbf { F } _ { r e f i n e d } = \alpha \cdot \mathbf { U p } ( \mathbf { F } _ { g l o b a l } ) + ( 1 - \alpha ) \cdot \mathbf { F } _ { l o c a l } ,\tag{1}
$$

where $\mathrm { U p } ( \cdot )$ denotes bilinear upsampling to align spatial resolutions, and α is a hyperparameter that balances the contributions of global and local information.

## 3.2. Structure-Aware Attention Refinement

While the self-attention maps in the final layer of CLIP’s ViT encoder possess strong capabilities for semantic grouping, they tend to exhibit spatially ambiguous object boundaries. Recent methods, such as SCLIP [13], suggest that employing KK<sup>⊤</sup> instead of standard $\mathbf { Q } \mathbf { K } ^ { \top }$ attention improves segmentation performance. However, boundary inconsistencies persist in complex scenes. To address this issue, we inject spatial correlations from intermediate attention maps and color affinities derived from the input image as biases into the final attention mechanism, as shown in Fig. 2. The final attention map $\mathbf { A } _ { f i n a l }$ is computed by adding a bias term $\mathbf { B } _ { t o t a l }$ to the $\mathbf { K K } ^ { \top }$ similarity matrix followed by the Softmax function:

![](images/3675429c4cb608dbc8aaff22a28a56807b44448dd05c26faac18b974f56d5a05.jpg)  
Fig. 2. Details of the proposed refined attention block. Unlike standard self-attention, we employ $\mathbf { K K } ^ { \top }$ similarity instead of $\mathbf { Q } \mathbf { K } ^ { \top }$ and inject spatial biases from intermediate layers along with color affinity biases from the input image.

$$
\mathbf { A } _ { f i n a l } = \mathrm { S o f t m a x } \left( \frac { \mathbf { K } _ { f i n a l } \mathbf { K } _ { f i n a l } \top } { \sqrt { d } } + \mathbf { B } _ { t o t a l } \right) ,\tag{2}
$$

where ${ \bf K } _ { f i n a l }$ denotes the $\operatorname { K e y }$ features of the final layer, and d is the feature dimension. We compute the product of this attention map $\mathbf { A } _ { f i n a l }$ and the final Value features $\mathbf { V } _ { f i n a l }$ , then apply a linear layer to obtain the output $\mathbf { O } _ { f i n a l }$ of the refined attention block. The bias term $\mathbf { B } _ { t o t a l }$ is defined as the sum of an internal bias $\mathbf { B } _ { i n t }$ based on intermediate attention maps and an external bias $\mathbf { B } _ { c o l o r }$ based on color information. For the internal bias $\mathbf { B } _ { i n t } .$ , we utilize attention maps from specific intermediate layers that strongly preserve the spatial layout of the input image. Furthermore, we integrate color information, which lacks semantic discriminability but retains precise boundary details, as the external bias $\mathbf { B } _ { c o l o r } .$ . Specifically, we convert the input image from sRGB to the CIELAB color space and use color features $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ downsampled to the patch level. The color affinity between patches i and $j$ is defined using a Gaussian kernel as follows:

$$
\mathbf { B } _ { c o l o r } ( i , j ) = \exp \left( - \frac { \| \mathbf { c } _ { i } - \mathbf { c } _ { j } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{3}
$$

where $\sigma$ is a bandwidth parameter, set to $\sigma = 3 0 . 0$ in our method. Note that color bias is not applied to the class token since it lacks spatial positional information. Additionally, following the insights from

ClearCLIP [22], we remove the residual connections and the feedforward network from the final Transformer block. This prevents the original features from being dominated by residual components, thereby maximizing the effect of the refined attention mechanism.

## 3.3. Zero-Shot Semantic Segmentation

The final segmentation is performed by matching the obtained image features with the text features. To generate text features, we insert the class name $y _ { k }$ of the target dataset into the 80 predefined ImageNet templates [12]. The text feature vector $\mathbf { F } _ { t e x t } ^ { k }$ is computed as the average of the embeddings obtained from the text encoder. Regarding image features, let ${ \bf F } _ { r e f i n e d }$ denote the high-resolution and boundary-aligned feature map. We calculate the cosine similarity score $S _ { i , j , k }$ between the image feature vector $\mathbf { F } _ { r e f i n e d } ^ { ( i , j ) }$ at pixel position (i, j) and the text feature vector $\mathbf { F } _ { t e x t } ^ { k }$ for class k as follows:

$$
S _ { i , j , k } = \frac { { \bf F } _ { r e f i n e d } ^ { ( i , j ) } \cdot { \bf F } _ { t e x t } ^ { k } } { | | { \bf F } _ { r e f i n e d } ^ { ( i , j ) } | | | { \bf F } _ { t e x t } ^ { k } | | } .\tag{4}
$$

Finally, the resulting similarity map is upsampled to the original image size, and the final prediction mask is obtained by assigning the class with the maximum score to each pixel.

## 4. EXPERIMENTS AND DISCUSSION

In this section, to verify the effectiveness of the proposed MARS-CLIP, we present ablation studies analyzing the contribution of each component and comparative experiments against state-of-the-art methods using standard benchmark datasets.

## 4.1. Dataset and Evaluation Metrics

We conduct evaluations on six standard segmentation benchmarks, following the protocols of previous studies such as SCLIP [13] and NACLIP [24]. For simplicity, we define the following abbreviations for each dataset setting. For PASCAL VOC 2012 [14] (1,449 images), we denote the 21-class setting including the background as $\mathrm { \Omega ^ { \ } \mathfrak { s v } } 2 1 ^ { \mathfrak { s } }$ and the 20-class setting excluding the background as “V20”. Similarly, for PASCAL Context [15] (5,105 images), we denote the 60-class setting $\mathrm { a s \ ^ { 6 4 } P C 6 0 ^ { 3 } }$ and the 59-class setting as “PC59”. Additionally, we denote ADE20K [16] (2,000 images) as “ADE”, COCO-Stuff [19] (5,000 images) as “C-Stf”, Cityscapes [17] (500 images) as $\mathrm { ^ { * } C i t y ^ { * } }$ , and COCO-Object [18] (5,000 images) as $^ { \mathrm { 6 6 } } \mathrm { C } \mathrm { - } \mathrm { O b j } ^ { \mathrm { 9 } }$ . For quantitative evaluation, we use the mean Intersection over Union (mIoU), the standard metric in semantic segmentation, calculated as the average IoU across all classes.

## 4.2. Implementation Details

We use the pre-trained CLIP ViT-B/16 [12] as the backbone model and freeze the weights of both image and text encoders. We employ 80 standard prompt templates for text embedding. During inference, input images are resized such that the shorter side is 336 pixels (560 pixels for “City”). We then apply sliding window processing with a window size of $2 2 4 \times 2 2 4$ pixels and a stride of 112. Hyperparameters specific to our method were set as follows based on the results of the ablation study. The crop size for multi-resolution feature extraction is set to 112, and the weight α for integrating global and local features is set to 0.8. For attention mechanism refinement, we utilize the intermediate attention maps from the 8th layer, which preserve spatial structure. For final mask generation, Pixel Adaptive

![](images/0d27302fa279f798bfd10b849848543d3140f7674c76bdec2e52686c93f9d565.jpg)  
Fig. 3. Impact of fusion weight α across different crop sizes on ${ \bf { \bar { \tau } } } \mathrm { { \bar { v } } } 2 1 ^ { , , , }$

Table 1. Ablation study on attention refinement using intermediate features from the 8th layer. “Feat. Sim.” indicates cosine similarity between features. The bottom row (Attn. + Color) represents our proposed setting, which adds color similarity bias to the standard attention.
<table><tr><td>Source</td><td>V21 PC60</td><td>C-Obj</td><td>V20</td><td>City PC59</td><td>ADE</td><td>C-Stf</td></tr><tr><td>Attn.  $( \mathbf { Q K } ^ { \top } )$ </td><td>58.6 32.6</td><td>33.8</td><td>83.0</td><td>32.7 35.9</td><td>17.1</td><td>23.9</td></tr><tr><td>Feat. Sim.</td><td>58.0 32.4</td><td>33.6</td><td>82.6</td><td>632.0 35.6</td><td>16.9</td><td>23.8</td></tr><tr><td> $\mathbf { K K } ^ { \top }$ </td><td>58.7 32.7</td><td>33.9</td><td>82.9</td><td>33.0 35.9</td><td>17.3</td><td>24.0</td></tr><tr><td>QQT</td><td>58.6 32.6</td><td>33.8</td><td>82.9</td><td>932.7 35.8</td><td>17.2</td><td>23.9</td></tr><tr><td> $\mathbf { V V } ^ { \top }$ </td><td>58.5 32.6</td><td>33.8</td><td>82.8</td><td>332.7 35.8</td><td>17.1</td><td>23.9</td></tr><tr><td>Attn. + Color</td><td>59.4</td><td>32.9 34.3</td><td></td><td>83.4 33.2</td><td>36.2 17.5</td><td>24.3</td></tr></table>

Mask Refinement (PAMR) [29] is applied as post-processing, with the background class threshold set to 0.1. All experiments were implemented using PyTorch and conducted on an NVIDIA A100 GPU.

## 4.3. Ablation Study

In this section, we validate the design choices of each component in MARS-CLIP and determine the optimal hyperparameters. Specifically, we analyze parameter sensitivity in the multi-resolution strategy, the selection of feature sources for attention refinement, and the individual contributions of each proposed module.

First, we present the results of examining the integration weight α for global and local features and the crop size in multi-resolution feature extraction in Fig. 3. The graph indicates that mIoU improves significantly when integrating both features compared to using single resolutions, such as $\alpha = 0 . 0$ (local only) or $\alpha = 1 . 0$ (global only). In particular, performance tends to saturate and maximize around $\alpha = 0 . 8 .$ . This suggests the importance of a balance where CLIP’s inherent global semantic understanding serves as the backbone, while missing detailed information is moderately supplemented by local features. Regarding crop size, 112 proved to be optimal. Smaller crop sizes result in insufficient context within patches, leading to reduced recognition accuracy. Based on these results, we fix the crop size to 112 and α to 0.8 for subsequent experiments.

Next, we investigate the optimal feature sources and extraction layers for refining the attention mechanism. While our method uses intermediate attention maps and color information as biases, Table 1 presents a comparison with other features (self-correlations of Q, K, V, and raw features). The comparison reveals that the setting using attention maps QK<sup>⊤</sup> (Attn.) combined with color information (+Color) achieves the highest accuracy. In particular, the gain from adding color information is substantial, confirming that low-level visual information contributes to defining object boundaries. Table 2 shows the comparison regarding the layer depth for feature extraction. Scores peak around the 8th layer (L8) and tend to decrease thereafter. This can be interpreted as the loss of spatial layout information as feature abstraction progresses in deeper layers. In our experiments, we adopt information from the 8th layer (L8), which showed the most stable performance on average.

Table 2. Ablation study on the layer depth for extracting attention maps. “Base” denotes using only the intermediate attention map as a bias, while “+Col” indicates the addition of color similarity as a bias.
<table><tr><td>Data Meth. L1 L2 L3 L4 L5 L6 L7 L8</td><td>L9 L10 L11</td></tr><tr><td>V21</td><td>Base 57.8 58.3 58.3 58.4 58.4 58.4 58.6 6 58.6 58.5 58.3 58.2 +Col 58.7 59.0 59.15 59.1 59.2 59.2 59.3 59.4 59.3 59.0 59.0</td></tr><tr><td>PC60</td><td>Base32.4 32.5 32.5 32.5 32.5 32.5 32.6 32.6 32.5 32.4 32.4 +Col 32.8 32.9 32.9 32.9 32.9 32.9 32.9 32.9 32.9 32.8 32.8</td></tr><tr><td>C-Obj</td><td>Base 33.5 533.733.73 33.7 33.8 33.8 33.8 33.8 33.8 33.7 33.6 +Col 34.0 34.2 34.2 234.1 34.2 34.2 34.3 34.3 34.2 34.1 34.1</td></tr><tr><td>V20</td><td>Base 82.5 82.7 82.9 82.8 82.9 82.9 83.0 83.0 83.18 83.0 82.8 +Col 82.9 83.1 83.3 83.2 83.2 83.3 83.4 83.4 83.4 83.3 83.2</td></tr><tr><td>City</td><td>Base 31.8 32.2 32.5 532.4 32.6 32.6 32.7 32.7 32.5 32.2 32.2 +Col 32.4 32.7 33.0 32.9 33.1 33.1 33.2 33.2 33.0 32.8 32.7</td></tr><tr><td>PC59</td><td>Base 35.6 35.8 35.8 35.8 35.8 35.8 35.8 35.9 35.8 35.7 35.6 +Col36.0 36.1 36.1 36.1 36.1 36.1 36.2 36.2 36.1 36.0 36.0</td></tr><tr><td>ADE</td><td>Base 16.9 17.0 17.1 17.0 17.1 17.1 17.1 17.1 17.0 17.0 17.0 +Col 17.3 17.4 17.4 17.4 17.4 17.4 17.5 17.5 17.417.317.3</td></tr><tr><td>C-Stf</td><td>Base23.7 23.9 23.9 23.9 23.9 23.9 23.9 23.9 23.9 23.8 23.8 +Col 24.1 24.2 24.2 24.2 24.2 24.2 24.2 24.3 24.2 24.1 24.1</td></tr></table>

Table 3. Ablation study on the components of MARS-CLIP. “Multi” refers to the adoption of the multi-resolution strategy, and “Bias” denotes attention refinement using low-level features.
<table><tr><td>Multi</td><td>Bias</td><td>V21</td><td>City</td><td>ADE</td><td>C-Stf</td><td>Avg Gain</td></tr><tr><td></td><td></td><td>58.9</td><td>35.0</td><td>17.4</td><td>23.4</td><td></td></tr><tr><td></td><td>V</td><td>59.9</td><td>35.5</td><td>17.5</td><td>23.4</td><td>+0.4</td></tr><tr><td>√</td><td></td><td>61.7</td><td>37.7</td><td>18.5</td><td>24.5</td><td>+1.9</td></tr><tr><td>√</td><td>√</td><td>62.0</td><td>38.2</td><td>18.5</td><td>24.4</td><td>+2.1</td></tr></table>

Finally, Table 3 presents the contribution of each component of our method. Compared to the standard CLIP baseline, introducing the multi-resolution strategy (Multi) resulted in significant accuracy improvements. Furthermore, combining this with attention mechanism refinement (Bias) led to further improvements, particularly on datasets containing high-definition and complex scenes such as “City” and “ADE”. These results demonstrate the complementary effects of enhancing resolution and restoring boundary information.

## 4.4. Comparison with State-of-the-Art Methods

We present the quantitative evaluation results on each benchmark in Table 4. To ensure a fair comparison, we conducted experiments under two settings: without post-processing and with post-processing using PAMR [29]. Our proposed method, MARS-CLIP, achieved accuracy surpassing existing state-of-the-art methods across all datasets and settings. In particular, our method demonstrates substantial performance improvement on datasets containing highresolution images and fine-grained objects, such as “City” and “ADE”. For instance, without PAMR, MARS-CLIP recorded an mIoU of 38.2% on “City”, outperforming the runner-up NACLIP (35.5%) by 2.7 points. This suggests that the extraction of detailed features by the multi-resolution strategy and boundary refinement by color information bias effectively overcome the loss of fine structures caused by insufficient resolution, which is a common weakness in conventional methods. Furthermore, the superiority of our method remains consistent even when PAMR is applied. On “City”, our method achieved 40.1% mIoU, realizing a 1.8-point improvement over NACLIP [24] (38.3%). This corroborates that the raw segmentation masks generated by our method already possess high spatial consistency without relying heavily on post-processing. Qualitative evaluation results are shown in Fig. 4. Compared to the conventional method NACLIP [24], our method detects small objects and regions with complex boundaries more accurately. These results demonstrate that the integrated mechanism of multi-resolution input and attention refinement in MARS-CLIP dramatically enhances the capability to recognize fine details in zero-shot segmentation.

Table 4. Comparison with state-of-the-art methods. The “PAMR” column indicates the application of post-processing. Best results are bold.
<table><tr><td>Method</td><td>PAMR [29]</td><td>V21</td><td>PC60</td><td>C-Obj</td><td>V20</td><td>City</td><td>PC59</td><td>ADE</td><td>C-Stf</td></tr><tr><td>CLIP [12]</td><td></td><td>18.6</td><td>7.8</td><td>6.5</td><td>49.1</td><td>6.7</td><td>11.2</td><td>3.2</td><td>5.7</td></tr><tr><td>MaskCLIP [9]</td><td></td><td>43.4</td><td>23.2</td><td>20.6</td><td>74.9</td><td>24.9</td><td>26.4</td><td>11.9</td><td>16.7</td></tr><tr><td>CLIP Surgery [10]</td><td></td><td>41.2</td><td>30.5</td><td></td><td></td><td>31.4</td><td></td><td>12.9</td><td>21.9</td></tr><tr><td>GEM [11]</td><td></td><td>46.2</td><td></td><td></td><td></td><td></td><td>32.6</td><td>15.7</td><td></td></tr><tr><td>SCLIP [13]</td><td></td><td>59.1</td><td>30.4</td><td>30.5</td><td>80.4</td><td>32.2</td><td>34.2</td><td>16.1</td><td>22.4</td></tr><tr><td>ClearCLIP [22]</td><td></td><td>51.8</td><td>32.6</td><td>33.0</td><td>80.9</td><td>30.0</td><td>35.9</td><td>16.7</td><td>23.9</td></tr><tr><td>NACLIP [24]</td><td></td><td>58.9</td><td>32.2</td><td>33.2</td><td>79.7</td><td>35.5</td><td>35.2</td><td>17.4</td><td>23.3</td></tr><tr><td>MARS-CLIP (Ours)</td><td></td><td>62.0</td><td>33.8</td><td>34.5</td><td>81.1</td><td>38.2</td><td>36.8</td><td>18.5</td><td>24.4</td></tr><tr><td>SCLIP [13]</td><td>V</td><td>61.7</td><td>31.5</td><td>32.1</td><td>83.5</td><td>34.1</td><td>36.1</td><td>17.8</td><td>23.9</td></tr><tr><td>NACLIP [24]</td><td>√</td><td>64.1</td><td>35.0</td><td>36.2</td><td>83.0</td><td>38.3</td><td>38.4</td><td>19.1</td><td>25.7</td></tr><tr><td>MARS-CLIP (Ours)</td><td>√</td><td>65.8</td><td>35.9</td><td>36.7</td><td>83.7</td><td>40.1</td><td>39.2</td><td>19.8</td><td>26.1</td></tr></table>

![](images/035a963213c080dc3cec10b37ac6e5fcef2a252e682ce284b41296f79125e466.jpg)  
Fig. 4. Qualitative comparison between existing methods and the proposed method.

We also compare inference cost on “V21” (shorter side 336 px, no sliding window). MARS-CLIP requires 1.1 GB peak memory and 0.0403 s per image, compared to 1.1 GB / 0.0390 s for NACLIP [24]. The peak memory matches because the memory footprint is dominated by dense image–text similarity rather than multi-resolution processing, and the 0.0013 s latency overhead is negligible.

## 5. LIMITATIONS

The color affinity bias may degrade in low-light or low-contrast scenes. Although the intermediate-layer spatial bias partially mitigates this, full illumination robustness is left as future work.

## 6. CONCLUSION

In this paper, we proposed MARS-CLIP, a zero-shot segmentation framework designed to accurately capture fine image details through multi-resolution inputs and attention mechanism refinement. Our approach successfully restored object boundaries while preserving semantic consistency by achieving high-resolution feature representation through the integration of local and global features, and by injecting spatial information from intermediate layers and color information as biases into the final layer. Through a set of experiments on six public datasets, we demonstrated the effectiveness of our proposed method, showing that it significantly outperforms state-of-theart methods.

## 7. ACKNOWLEDGMENT

This work was supported in part by JSPS KAKENHI 23H00463 and 25K03131, JST BOOST JPMJBS2421, and the WISE Program for AI Electronics, Tohoku University.

## 8. REFERENCES

[1] S. Minaee, Y. Boykov, F. Porikli, N. Plaza, A. Kehtarnavaz, and D. Terzopoulos, “Image segmentation using deep learning: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 7, pp. 3523–3542, July 2022.

[2] Y. Xu, R. Quan, W. Xu, Y. Huang, X. Chen, and F. Liu, “Advances in medical image segmentation: A comprehensive review of traditional, deep learning and hybrid approaches,” Bioengineering, vol. 11, no. 10, pp. 1034, Oct. 2024.

[3] Y. Xia, J. Zhou, X. Xun, L. Johnston, T. Wei, R. Gao, Y. Zhang, B. Reddy, C. Liu, and G. Kim, “Deep learning for oncologic treatment outcomes and endpoints evaluation from CT scans in liver cancer,” npj Precis. Oncol., vol. 8, no. 1, pp. 263, Nov. 2024.

[4] E. Cumbajin, N. Rodrigues, P. Costa, R. Miragaia, L. Fraz˜ao, N. Costa, A. Fern´andez-Caballero, J. Carneiro, L. H. Buruberri, and A. Pereira, “A systematic review on deep learning with cnns applied to surface defect detection,” J. Imaging, vol. 9, no. 10, pp. 193, Sept. 2023.

[5] H. Wu, Y. Liu, and J. Yang, “An automatic and unsupervised image mask acquisition method based on generative adversarial networks,” Syst. Sci. Control Eng., vol. 12, no. 1, pp. 2300835, Jan. 2024.

[6] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 3431–3440, June 2015.

[7] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” Eur. Conf. Comput. Vis., pp. 833– 851, Sept. 2018.

[8] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. Alvarez, and P. Luo, “SegFormer: Simple and efficient design for semantic segmentation with transformers,” Adv. Neural Inform. Process. Syst., vol. 34, pp. 12077–12090, Dec. 2021.

[9] C. Zhou, C. C. Loy, and B. Dai, “Extract free dense labels from CLIP,” Eur. Conf. Comput. Vis., pp. 696–712, Oct. 2022.

[10] Y. Li, H. Wang, Y. Duan, J. Zhang, and X. Li, “A closer look at the explainability of contrastive language-image pretraining,” Pattern Recognition, vol. 162, no. 111409, pp. 1–13, June 2025.

[11] W. Bousselham, F. Petersen, V. Ferrari, and H. Kuehne, “Grounding everything: Emerging localization properties in vision-language transformers,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 3828–3837, June 2024.

[12] A. Radford, J. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” Int. Conf. Mach. Learn., pp. 8748–8763, July 2021.

[13] F. Wang, J. Mei, and A. Yuille, “Sclip: Rethinking selfattention for dense vision-language inference,” Eur. Conf. Comput. Vis., pp. 315–332, Oct. 2024.

[14] M. Everingham, S. Eslami, L. Van Gool, C. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes challenge: A retrospective,” Int. J. Comput. Vis., vol. 111, pp. 98–136, Jan. 2015.

[15] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Urtasun, and A. Yuille, “The role of context for object detection and semantic segmentation in the wild,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 891–898, June 2014.

[16] B. Zhou, H. Zhao, Xa. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” Int. J. Comput. Vis., vol. 127, pp. 302–321, Dec. 2019.

[17] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes dataset for semantic urban scene understanding,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 3213–3223, June 2016.

[18] T. Y. Lin, M. Maire, S. Belongie, J. Hays, and P. Peroma, “Microsoft COCO: Common objects in context,” Eur. Conf. Comput. Vis., pp. 740–755, Sept. 2014.

[19] H. Caesar, J. Uijlings, and V. Ferrari, “COCO-Stuff: Thing and stuff classes in context,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 1209–1218, June 2018.

[20] M. Bucher, T.-H. Vu, M. Cord, and P. P´erez, “Zero-shot semantic segmentation,” Adv. Neural Inform. Process. Syst., vol. 32, no. 43, pp. 468–479, Dec. 2019.

[21] D. Baek, Y. Oh, and B. Ham, “Exploiting a joint embedding space for generalized zero-shot semantic segmentation,” Int. Conf. Comput. Vis., pp. 9536–9545, Oct. 2021.

[22] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Clearclip: Decomposing clip representations for dense visionlanguage inference,” Eur. Conf. Comput. Vis., pp. 143–160, Oct. 2024.

[23] Y. Yang, J. Deng, W. Li, and L. Duan, “ResCLIP: Residual attention for training-free dense vision-language inference,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 29968– 29978, June 2025.

[24] S. Hajimiri, I. B. Ayed, and J. Dolz, “Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation,” IEEE/CVF Winter Conf. App. Comput. Vis., pp. 5061– 5071, Mar. 2025.

[25] X. Wang, C. Si, X. Yang, Y. Zhao, W. Wang, X. Yang, and W. Shen, “OPMapper: Enhancing open-vocabulary semantic segmentation with multi-guidance information,” Adv. Neural Inform. Process. Syst., pp. 1–28, Dec. 2025.

[26] J. Luo, S. Khandelwal, L. Sigal, and B. Li, “Emergent openvocabulary semantic segmentation from off-the-shelf visionlanguage models,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 4029–4040, June 2024.

[27] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Proxyclip: Proxy attention improves clip for open-vocabulary segmentation,” Eur. Conf. Comput. Vis., pp. 70–88, Oct. 2024.

[28] L. Barsellotti, R. Amoroso, M. Cornia, L. Baraldi, and R. Cucchiara, “FreeDA: Training-free open-vocabulary segmentation with offline diffusion-augmented prototype generation,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 3689–3698, June 2024.

[29] N. Araslanov and S. Roth, “Single-stage semantic segmentation from image labels,” IEEE/CVF Conf. Comput. Vis. Pattern Recog., pp. 4253–4262, June 2020.