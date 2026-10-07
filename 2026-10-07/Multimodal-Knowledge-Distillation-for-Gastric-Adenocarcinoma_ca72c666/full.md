# Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images

Shrihari Dumbre<sup>1</sup> and Bikash Santra<sup>1</sup>

Indian Institute of Technology, Jodhpur b23ee1069@iitj.ac.in, bikash@iitj.ac.in

Abstract. Gastric adenocarcinoma (GA) is a leading cause of cancerrelated mortality worldwide, and accurate histopathological subtype classification from whole-slide images (WSIs) is essential for efective treatment planning. While multimodal approaches that integrate pathology report text with WSIs can improve classification, existing methods often depend on computationally expensive transformer architectures and large language models. We propose a multimodal knowledge distillation (MKD) framework that combines a pretrained WSI image encoder and a clinical text encoder using Low-Rank Multimodal Fusion (LMF) to eficiently model cross-modal interactions during training. Each WSI is represented as a bag of patches paired with a slide-level diagnostic caption. The teacher model learns fused image–text representations for subtype classification, while the student model distills this knowledge to enable accurate image-only inference. We evaluate our method on the PatchGastric benchmark dataset and achieve at least 3.35% higher mean accuracy than state-of-the-art approaches, without relying on transformer-based fusion, multi-task learning, or large language models. The source code is available at https://github.com/helomelo1/MKD-LMF.

Keywords: Multimodal Learning · Low-Rank Multimodal Fusion · Knowledge Distillation · Gastric Cancer Classification · Histopathology Image Analysis

## 1 Introduction

Accurate histological subtype classification of gastric adenocarcinoma from wholeslide images (WSIs) is clinically critical for treatment planning, yet morphological similarity between subtypes makes this task challenging even for experienced pathologists [15]. Most recent multimodal approaches [5–10] demonstrate that pairing WSI features with complementary data sources, such as genomic sequencing data, gene expression profiles and clinical records, improves classification. But most of them either require genomic sequencing data that is unavailable in routine clinical workflows [5, 6] or rely on highly complex architectures, such as PathM3 [10]. It employs a 12-layer Query Transformer initialized from

BLIP-2 [11] together with a frozen FlanT5-XL decoder [22] for classification and caption generation jointly.

The PatchGastric dataset [15] is one of the few publicly available datasets that provide both histopathology patch images and WSI-level diagnostic captions (i.e., report summaries), making it well-suited for multimodal analysis of gastric adenocarcinoma. However, the dataset presents a significant challenge. Many WSIs across diferent subtypes share near-identical descriptions, limiting their standalone discriminating power. Due to real-world data scarcity, methods on this dataset follow a data-scarce training protocol, using only 20% of the WSIs for training. Prior methods on this dataset, such as PathM3 [10] address these challenges using complex multi-task learning-based frameworks that jointly optimize classification and caption-generation using large-language models. While efective, these methods introduce substantial computational overhead.

On the contrary, we propose an LMF-based multimodal fusion approach that fuses patch-level image features with pathology report captions using Low-Rank Multimodal Fusion (LMF) [12]. Notably, each WSI in the PatchGastric dataset [15] is provided as a set of pre-extracted patches of size 300 × 300 at 20× magnification. Following the Multiple Instance Learning paradigm [1], each WSI is treated as a bag of patch instances. Each patch embedding is obtained and mean-pooled via LMF to fuse with the WSI-level caption embedding. LMF models multiplicative cross-modal interactions through a low-rank tensor decomposition without requiring attention-based cross-modal modules [6, 10] or generative objectives [11, 10]. Pathology report captions are routinely generated during standard diagnostic workflows, making them a practical, readily available second modality alongside H&E-stained histopathology images. Unlike PG-MLIF [14], which applies LMF to fuse WSIs with genomic data for survival prediction, we apply LMF to the image-report setting for histological subtype classification of gastric adenocarcinoma.

Our experiments compare the proposed method against ABMIL [1], CLAM [3], DSMIL [4], TransMIL [2], CITE [18], ILRA-MIL [19] and PathM3 [10]. We use the same PatchGastric dataset and the same train-validation-test split ratio as in CITE [18] and PathM3 [10]. We use three gastric adenocarcinoma subtypes, including “well diferentiated tubular adenocarcinoma”, “moderately diferentiated tubular adenocarcinoma”, and “poorly diferentiated adenocarcinoma”, similar to these baselines, thus enabling a direct comparison. We further conduct experiments on diferent multimodal fusion strategies and ablation studies across multiple LMF rank settings. The contributions of this work are the following:

– Lightweight multimodal framework: We propose an eficient multimodal knowledge distillation (MKD) approach that fuses WSI features and clinical text using Low-Rank Multimodal Fusion (LMF), avoiding heavy transformer-based architectures and large language models.

– Knowledge distillation for image-only inference: We introduce a teacher –student setup where multimodal knowledge learned during training is distilled into a student model, enabling accurate and eficient inference using only WSIs.

– Improved performance with eficiency: We conduct extensive experiments to analyze the contributions of individual modalities and the efect of the LMF rank in improving the gastric adenocarcinoma classification performance.

## 2 Related Works

## 2.1 Multiple Instance Learning for Histopathology

Multiple Instance Learning (MIL) is a weakly supervised paradigm where training examples are organized into bags of instances, with labels available only at the bag level [20]. In computational pathology, a whole-slide image (WSI) is treated as a bag containing hundreds to thousands of patch-level instances (small image regions extracted at specific magnification), with the number of patches varying across WSIs. The WSI-level label is known, but individual patch labels are not, making MIL a natural fit for WSI analysis. Ilse et al. [1] proposed Attention-based MIL (ABMIL), which learns attention weights to aggregate patch features into a bag-level representation. In another work, Shao et al. [2] introduced Transformer-based correlated MIL (TransMIL), employing transformer-based self-attention with Nyström approximation [21] to model correlations among patches eficiently. Whereas, Lu et al. [3] proposed Clustering-Constrained-Attention Multiple-Instance Learning (CLAM), a data-eficient and weakly supervised framework for WSI classification. While Li et al. [4] introduced Dual-Stream MIL (DSMIL), a dual-stream network that incorporates self-supervised contrastive learning. All of these methods rely exclusively on image features for histopathology classification. On the contrary, we introduce an MIL-based multimodal framework that uses a bag-of-image-patches representation and a caption generated from the pathology report.

## 2.2 Multimodal Fusion on Computation Pathology

A number of recent works [5–10] have explored multimodal fusion for cancer prognosis and classification by combining histopathology images along with complementary data modalities. Chen et al. [5] proposed Pathomic Fusion, integrating histopathology and genomic features via the Kronecker product for survival prediction. While efective in modelling cross-modal relationships, its tensor fusion is computationally expensive, requires paired genomic data, and assumes the availability of both modalities during inference. In another work, Chen et al. [6] introduced Multimodal Co-Attention Transformer (MCAT), a co-attention Transformer fusing WSIs with genomic data, while Xu and Chen [7] proposed Multimodal Optimal Transport Co-Attention Transformer (MOTCat) using optimal transport-based co-attention. On the other hand, Zhou and Chen [8] introduced Cross-Modal Translation and Alignment (CMTA) for cross-modal translation and alignment, and Ding et al. [9] proposed PathOmics, a pathologygenomics Transformer for survival prediction. On the other hand, recent visionlanguage approaches focus on image-text representations that leverage widely available diagnostic captions, enabling more flexible multimodal learning. For example, most closely related to our work, Zhou et al. [10] proposed PathM3, an MIL-based multimodal framework. PathM3 aligns WSIs with diagnostic captions using a query-based transformer initialized with BLIP-2 [11], jointly training for classification and caption generation. While PathM3 achieves strong performance, it relies on a computationally expensive architecture with a 12-layer query Transformer and a frozen FlanT5-XL decoder. In contrast, we employ a knowledge distillation framework with a low-rank multimodal fusion approach that directly learns joint image-text representations for WSI-only classification, avoiding the need for caption-generation objectives and large language models while achieving superior classification performance.

## 2.3 Low-Rank Multimodal Fusion and Knowledge Distillation

Low-Rank Multimodal Fusion (LMF) was introduced by Liu et al. [12] as an eficient alternative to Tensor Fusion Networks [13]. LMF uses parallel low-rank decomposition to model pairwise interactions between features from the image and text modalities while avoiding the exponential parameter growth of full tensor fusion [13]. For cancer survival prediction, Pan et al. [14] applied LMF to fuse histopathology images with genomic data. PG-MLIF [14] requires genomic profiling data, which is expensive and not always clinically available. Unlike PG-MLIF [14], we apply LMF to fuse histopathology images with pathology report summaries, which are routinely generated as part of standard diagnostic workflows. Knowledge distillation [23] transfers knowledge from a large teacher model to a smaller student model. FitNets [24] extended this by distilling intermediate representations, thereby enabling richer feature learning. In our approach, we employ knowledge distillation to transfer the multimodal representation learned from both image and text to a student model that operates solely on images.

## 3 Methodology

## 3.1 Problem Formulation

Consider a dataset of M whole-slide images (WSIs), where each WSI is represented as a bag of N patches $\boldsymbol { B } = \{ p _ { 1 } , p _ { 2 } , \dots , p _ { N } \}$ , with N varying across WSIs based on tissue regions. Each WSI is paired with a diagnostic caption c provided in the dataset, derived from the corresponding pathology report. The goal is to classify WSIs into $K = 3$ subtypes of gastric adenocarcinoma: well-diferentiated, moderately diferentiated, and poorly diferentiated tubular adenocarcinoma.

We address this 3-way classification of gastric adenocarcinoma in two stages. First, a multimodal teacher $f _ { \mathrm { t e a c h e r } } : ( \boldsymbol { B } , \boldsymbol { c } ) \to \boldsymbol { y }$ is trained using both image and text modalities. Second, an image-only student $f _ { \mathrm { s t u d e n t } } : B  y$ is trained via knowledge distillation, enabling WSI-only inference without the pathology report. The proposed complete pipeline is illustrated in Fig. 1.

![](images/f5b1b6cf668e29b855089b56804eae6c2b7a665d07f40b733ee8717cd07b129e.jpg)  
Fig. 1: Overview of the proposed two-stage MKD-LMF framework. WSI patches are encoded and mean-pooled to obtain a bag-level image representation $\left( \mathbf { z } _ { i m g } \right)$ while the corresponding caption is passed through a text encoder to extract text features $\left( \mathbf { z } _ { t x t } \right)$ . In Stage 1, the teacher network employs a Low-Rank Multimodal Fusion (LMF) module to combine $\left( \mathbf { z } _ { i m g } \right)$ and $\left( \mathbf { z } _ { t x t } \right)$ , producing a fused representation $( { \bf z } _ { f u s e d } )$ , which is used to train a linear classifier (shared with the student) for gastric adenocarcinoma (GA) subtype classification. In Stage 2, the student network takes only $\left( \mathbf { z } _ { i m g } \right)$ as input and learns, via a linear knowledge distillation module, to approximate $\mathbf { z } _ { f u s e d }$ generated by the teacher’s frozen LMF module. The synthesized representation is then passed to the shared classifier to enable image-only GA subtype classification.

## 3.2 Image Feature Extraction

First, the patches of a WSI $p _ { i } \in B \subset \mathbb { R } ^ { H \times W \times 3 }$ with height H and width $W$ are resized to $2 2 4 \times 2 2 4$ . Subsequently, each patch $p _ { i }$ is passed through a pretrained image encoder $\phi _ { \mathrm { i m g } }$ to obtain a $d _ { v }$ -dimensional patch-level image embedding:

$$
\mathbf { v } _ { i } = \phi _ { \mathrm { i m g } } ( p _ { i } ) \in \mathbb { R } ^ { d _ { v } } .\tag{1}
$$

In our implementation, Phikon [16], a Vision Transformer (ViT-B/16) [30] pretrained on large-scale histopathology data via self-supervised learning [31] is used as our image encoder, where $d _ { v } = 7 6 8$ . The weights of $\phi _ { \mathrm { i m g } }$ are kept frozen during training of the proposed model.

## 3.3 Bag-Level Feature Representation via Aggregation

Following the MIL paradigm, we aggregate the patch-level representations $\mathbf { v } _ { i }$ within each bag $\boldsymbol { B }$ to obtain a single bag-level representation. Since patches within a bag have no inherent ordering, the aggregation function must be permutation invariant, meaning the output should not change when the order of input patches is rearranged. This property is essential because the model should produce the same prediction regardless of how patches are arranged. Thus, we use mean pooling as the permutation-invariant aggregation function:

$$
\mathbf { z } _ { i m g } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \mathbf { v } _ { k } \in \mathbb { R } ^ { d _ { v } } ,\tag{2}
$$

to treat all patches equally, providing a simple yet efective summarization of the bag contents.

## 3.4 Text Feature Extraction

The caption c, a summarized pathology report, associated with the input WSI is encoded using a pretrained text encoder $\phi _ { \mathrm { t x t } }$ to obtain a $d _ { t }$ -dimensional bag-level text embedding:

$$
\mathbf { z } _ { t x t } = \phi _ { \mathrm { { t x t } } } ( c ) \in \mathbb { R } ^ { d _ { t } } .\tag{3}
$$

We use BGE-small-en-v1.5 [17], a pretrained sentence-embedding model, as our text encoder, with $d _ { t } = 3 8 4$ . Since all patches within a bag share the same WSIlevel caption, the text embedding $\mathbf { z } _ { t x t }$ is paired with the bag-level aggregated and fixed-dimensional representation $\mathbf { z } _ { i m g }$ of WSI patches. The weights of $\phi _ { \mathrm { t x t } }$ are kept frozen during the training of our model.

## 3.5 Low-Rank Multimodal Fusion

Multimodal GA subtype classification requires capturing high-order correlations between visual and textual embeddings $\mathbf { z } _ { i m g }$ and $\mathbf { z } _ { t x t }$ . Simple additive fusion schemes (such as feature concatenation or summation) assume that modalities contribute independently, which fails to model inter-modal dependencies. Conversely, multiplicative fusion captures the cross-modal interactions by computing pairwise products between every image feature and every text feature. However, mapping this full tensor to the output space requires a high-dimensional fusion weight tensor whose parameters scale quadratically with the number of features, leading to severe memory overhead. To capture and approximate multiplicative cross-modal interactions between the image and text modalities, we employ Low-Rank Multimodal Fusion (LMF) [12].

Given a bag-level representation of patches $\mathbf { z } _ { i m g } ~ \in ~ \mathbb { R } ^ { d _ { v } }$ and the caption embedding $\mathbf { z } _ { t x t } \in \mathbb { R } ^ { d _ { t } }$ , LMF computes a fused representation. It introduces two sets of learnable projection matrices $\mathbf { W } _ { v } ^ { ( k ) }$ and $\bar { \mathbf { W } } _ { t } ^ { ( k ) }$ specific to the image and text modalities, respectively We append a constant 1 to each embedding vector to incorporate a bias term into the low-rank factorization, enabling the model to learn an additive ofset without a separate bias parameter. Appending a constant 1 allows the augmented dimensions to act as multiplicative identities, enabling the factorization to represent image-only, text-only, and bias terms in addition to cross-modal interactions as follows:

$$
\bar { \mathbf { z } } _ { i m g } = [ \mathbf { z } _ { i m g } ; 1 ] \in \mathbb { R } ^ { d _ { v } + 1 } , \quad \bar { \mathbf { z } } _ { t x t } = [ \mathbf { z } _ { t x t } ; 1 ] \in \mathbb { R } ^ { d _ { t } + 1 } .\tag{4}
$$

Rather than explicitly constructing the full fusion tensor, LMF factorizes the fusion weight tensor into r distinct components. For each factor $j \in \{ 1 , \ldots , r \}$ , learnable projection matrices $\mathbf { W } _ { v } ^ { ( j ) } \in \mathbb { R } ^ { d _ { f } \times ( d _ { v } + 1 ) }$ and $\mathbf { W } _ { t } ^ { ( j ) } \in \mathbb { R } ^ { d _ { f } \times ( d _ { t } + 1 ) }$ project the augmented embeddings into a common $d _ { f } .$ -dimensional space, and the fused representation is obtained by summing element-wise products across all factors:

$$
\mathbf { z } _ { f u s e d } = \sum _ { j = 1 } ^ { r } \left( \mathbf { W } _ { v } ^ { ( j ) } \bar { \mathbf { z } } _ { i m g } \right) \odot \left( \mathbf { W } _ { t } ^ { ( j ) } \bar { \mathbf { z } } _ { t x t } \right) \in \mathbb { R } ^ { d _ { f } } ,\tag{5}
$$

where ⊙ denotes the Hadamard (element-wise) product, r is the rank, a hyperparameter controlling the expressiveness of the decomposition, and $d _ { f }$ is the fused embedding dimension. This formulation efectively retains the rich cross-modal interactions. Conventional multiplicative fusion requires $O ( d _ { f } \cdot d _ { v } \cdot d _ { t } )$ trainable parameters. In contrast, LMF reduces this parameter count to $O ( r \cdot d _ { f } \cdot ( d _ { v } + d _ { t } ) )$ , and thereby drastically improves parameter eficiency.

## 3.6 Classification

The proposed teacher-student MKD framework uses a shared classifier trained on the teacher side, which is kept frozen on the student side to distil the teacher’s knowledge. As shown in Fig. 1, on the teacher network, the fused multimodal representation $\mathbf { z } _ { f u s e d }$ is fed to a linear classifier $\left( \phi _ { \mathrm { c l s } } \right)$ to obtain the predicted GA subtype class probabilities:

$$
\begin{array} { r } { \hat { \mathbf { y } } _ { T } = \phi _ { \mathrm { c l s } } ( \mathbf { z } _ { f u s e d } ) \in \mathbb { R } ^ { K } . } \end{array}\tag{6}
$$

## 3.7 Knowledge Distillation for Synthesizing Multimodal Embedding

The multimodal model often achieves high accuracy by using both image and text data. However, it requires the pathology report at inference time. Pathology reports for histopathology images are generated after pathologists have made the diagnostic decision and identified the GA subtype. This limits its clinical applicability, as our objective is to automate this GA subtype identification. To address this limitation, we introduce a student network in which a Knowledge Distiller H learns to predict the teacher’s multimodal fused representation $\mathbf { z } _ { f }$ used from only the mean-pooled image features $\mathbf { z } _ { i m g }$ . Thus,

$$
\hat { \mathbf { z } } _ { f u s e d } = H ( \mathbf { z } _ { i m g } ) \in \mathbb { R } ^ { d _ { f } } ,\tag{7}
$$

where $\hat { \mathbf { z } } _ { f u s e d }$ is an approximated representation of $\mathbf { z } _ { f u s e d }$ (obtained from the teacher). The architecture of the proposed Knowledge Distiller is demonstrated in Fig. 2. It is then passed to the frozen shared classifier $\phi _ { \mathrm { c l s } }$ for WSI-only GA subtype classification.

![](images/f707cf8e8db019ce01d5871626a7fdb353bb7d93259e8cd42ec8dc7e565a6ad8.jpg)  
Fig. 2: Architecture of the proposed Knowledge Distiller, a residual multi-layer perceptron (MLP), comprising an input projection, stacked residual blocks, and an output projection for mapping image features to the fused feature space.

## 3.8 Two-Stage Training

Our proposed framework employs a two-stage training pipeline.

Stage 1: Multimodal Training. On the teacher side, we train the LMF-based multimodal model using both the image and text modalities by optimizing the cross-entropy loss $\mathcal { L } _ { C E }$ between y and $\hat { y } _ { T }$ . The teacher learns to produce discriminative fused representations $\mathbf { z } _ { f u s e d }$ that combine pathological information from WSI patches and diagnostic information from text captions. After training, the teacher’s module parameters (encoders, LMF module, and classifier) are frozen, as illustrated in Fig. 1.

Stage 2: Knowledge Distillation for Image-Only Inference. Using the teacher’s frozen modules, we train the Knowledge Distiller network H to predict the teacher’s fused representations directly from image features. The teacher’s classifier is shared with the student and kept frozen during this stage. We employ a combined distillation loss, $\mathcal { L } _ { d i s t i l l }$ as follows:

$$
\begin{array} { r } { \mathcal { L } _ { d i s t i l l } = \mathcal { L } _ { C E } + \lambda _ { f e a t } \underbrace { \mathcal { L } _ { f e a t } { ( \hat { \mathbf { z } } _ { f u s e d } , \ \mathbf { z } _ { f u s e d } ) } } _ { \mathrm { f e a t u r e ~ m a t c h i n g ~ l o s s } } + \lambda _ { K D } \underbrace { \mathcal { L } _ { K D } { ( \hat { \mathbf { y } } _ { S } , \ \hat { \mathbf { y } } _ { T } ) } } _ { \mathrm { l o g i t ~ d i s t i l l a t i o n ~ l o s s } } } \end{array}\tag{8}
$$

The cross-entropy loss $\mathcal { L } _ { C E }$ supervises the student’s predictions $\hat { \mathbf { y } } _ { S }$ against the ground-truth labels $y .$ While the feature matching loss uses Smooth L1 [29] to maximize the teacher-student agreement in synthesizing the multimodal fused representation by minimizing the distance between the representations predicted by them. On the other hand, the logit distillation loss uses KL divergence [23] with temperature τ to align the student’s output distribution with the teacher’s:

$$
\mathcal { L } _ { K D } = \tau ^ { 2 } \cdot D _ { \mathrm { K L } } \bigg ( \mathrm { s o f t m a x } \bigg ( \frac { \hat { \mathbf { y } } _ { S } } { \tau } \bigg ) ~ \left\| ~ \mathrm { s o f t m a x } \bigg ( \frac { \hat { \mathbf { y } } _ { T } } { \tau } \bigg ) \right) ,\tag{9}
$$

where we experimentally set $\tau = 2 , \lambda _ { f e a t } = 1 . 0$ , and $\lambda _ { K D } = 0 . 5$

Why triple supervision? Cross-entropy alone trains the student’s knowledge distillation module from scratch without leveraging the teacher’s learned structure. Feature-level supervision enforces the student to reproduce the teacher’s intermediate representations, but might not preserve class boundaries if the representation space contains redundant information. Logit-level supervision transfers the teacher’s soft class relationships but does not capture the full structure of the fused features. Combining all three ensures the Knowledge Distiller learns representations that are structurally faithful to the teacher’s, discriminative for classification, and grounded in the true labels.

## 3.9 Inference

At inference time, the proposed pipeline requires only the WSI, as shown in the student block of the Fig. 1. Using the frozen Phikon [16] encoder $\phi _ { i m g } .$ , we extract patch-level features, as discussed in Section 3.2. We aggregate the patches using mean pooling to obtain a fixed-dimensional WSI-level representation ${ \bf z } _ { i m g } ,$ as explained in Section 3.3. We pass the $\mathbf { z } _ { i m g }$ through the knowledge distiller H to obtain $\hat { \mathbf { z } } _ { f u s e d } ,$ a sythesized representation of the image-text fused embedding $\mathbf { z } _ { f u s e d }$ generated by the teacher. Subsequently, this synthesized representation is then passed through the frozen shared classifier $\phi _ { c l s }$ to obtain the final predictions, $\hat { \mathbf { y } } _ { S } ,$ as described in Section 3.7.

Note that the pathology report is used only during training. At inference time, the model requires only the WSI, making it suitable for real-time clinical decision support when reports are not yet available.

## 4 Experiments and Results

## 4.1 Dataset

In our experiments evaluating the proposed method, we use the PatchGastric [15] dataset, which originally consists of 991 WSIs, from which patches are extracted and paired with captions from diagnostic reports. There are 262,777 patches in the dataset, each 300x300 pixels in size, extracted at 20x magnification. The dataset contains 9 subtypes of gastric adenocarcinoma, out of which we use 3 following the baselines [18, 10] for a fair comparison: "well diferentiated tubular adenocarcinoma", "moderately diferentiated tubular adenocarcinoma" and "poorly diferentiated adenocarcinoma". This reduces the patches to 179,285 across 693 WSIs. The data split ratio is the same as in all previous methods. We randomly split the dataset into train (20%), validate (40%) and test (40%).

## 4.2 Implementation Details

For image feature extraction, we use Phikon [16], a ViT-B/16 encoder with an input image size of 224×224, in which each input image is split into nonoverlapping 16×16 patches. Each WSI patch is resized to 224x224 for input, and then the encoder converts it into a $d _ { v } = 7 6 8$ dimensional embedding. The final output after passing all the patches from the WSI is $N \mathrm { ~ x ~ } 7 6 8$ , where N is the number of patches varying across WSIs. For caption encoding we use BGEsmall-en-v1.5 [17], a retrieval-optimized sentence encoder from the BGE model family. The text encoder returns a $d _ { t } = 3 8 4$ dimensional embedding. The LMF module is implemented with the rank $r = 1 2 8$ (experimentally set as discussed in Section 4.3) to obtain a fused embedding of size $d _ { f } = 1 2 8$ . The shared classifier is a fully connected layer mapping from an input size of 128 to 3, consisting of a ReLU activation and a dropout layer with probability 0.3. The teacher model is trained for 20 epochs using the Adam [32] optimzer with a learning rate of 1e-4, optimizing cross-entropy loss.

Table 1: Comparison with state-of-the-art (SOTA) methods under image-only inference. Results are reported as mean accuracy (%) ± standard deviation. ∆ denotes the diference in mean accuracy of SOTA methods compared to our method (MKD-LMF). Red-colored negative values indicate lower performance than ours, while positive values indicate higher performance.
<table><tr><td>Method</td><td>Accuracy (%)</td><td>Δ</td></tr><tr><td>ABMIL [1] CLAM [3]</td><td> $\overline { { 6 6 . 6 7 \pm 2 . 6 3 } }$   $6 7 . 8 3 \pm 1 . 1 2$ </td><td>-8.16 -7.00</td></tr><tr><td>DSMIL [4]</td><td> $6 9 . 0 2 \pm 0 . 4 2$ </td><td>-5.81</td></tr><tr><td>TransMIL [2] CITE [18]</td><td> $6 7 . 4 8 \pm 1 . 1 6$ </td><td>-7.35</td></tr><tr><td>ILRA-MIL [19]</td><td> $6 9 . 6 3 \pm 0 . 9 1 $ </td><td>-5.20</td></tr><tr><td>PathM3 [10]</td><td> $7 0 . 1 6 \pm 1 . 1 1$ </td><td>-4.67</td></tr><tr><td>MKD-LMF (Ours)</td><td> $7 1 . 4 8 \pm 1 . 3 0$   $\overline { { { \bf 7 4 . 8 3 \pm 1 . 4 9 } } }$ </td><td>-3.35</td></tr></table>

As illustrated in Fig. 2, the Knowledge Distiller (KD) is implemented as a residual network [25] consisting of three components. First, an input projection maps a $d _ { v } = 7 6 8$ dimensional input to $d _ { h } = 5 1 2$ via a linear layer followed by Layer Normalization [26], GELU [27] activation, and dropout with probability 0.3. Second, each residual block preserves the 512-dimensional hidden representation and consists of two (512 → 512) linear layers, each followed by Layer Normalization and GELU activation, with a skip connection between the block input and output. A dropout layer with a probability of 0.4 is applied within each residual block. Third, the hidden representation $d _ { h } = 5 1 2$ is then mapped to the output $d _ { f } = 1 2 8$ via a linear layer followed by Layer Normalization [26]. The student model is trained for 50 epochs using the Adam [32] optimizer with a learning rate of 1e-4, optimizing the distillation loss $\mathcal { L } _ { d i s t i l l }$ defined in (8).

The proposed method is implemented in PyTorch, and all experiments are evaluated on a computing system with NVIDIA RTX 6000 ADA 48GB GPU, 512 GB of RAM and an Intel Xeon Gold 6426Y processor. Reported classification accuracy is the mean and standard deviation over three independent runs.

Table 2: Ablation study of the proposed MKD-LMF framework. The results of the experiments in the table are reported for a single run. The table analyzes (i) the performance of the teacher model using both image and text during inference, which is not clinically relevant, (ii) the efect of diferent distillation loss components in combination with the classification loss (cross-entropy) on the student-side and (iii) the impact of diferent inference modalities. Results show that combining both losses yields the best student performance. The student model in our multimodal knowledge distillation (KD) framework substantially outperforms unimodal (image-only or text-only) inference settings.
<table><tr><td>Study</td><td>Configuration</td><td>Accuracy (%)</td></tr><tr><td>Multimodal Inference</td><td>Proposed teacher model</td><td>92.74</td></tr><tr><td rowspan="2">Distillation Loss</td><td>Feature loss (Lfeat) KD loss (LKD)</td><td>73.63</td></tr><tr><td>Feature loss + KD loss</td><td>73.13 74.83</td></tr><tr><td rowspan="3">Inference Modality</td><td>Text-only (unimodal)</td><td>68.76</td></tr><tr><td>Image-only (unimodal)</td><td>70.35</td></tr><tr><td>Image-only (proposed multimodal KD)</td><td>74.83</td></tr></table>

## 4.3 Results and Analysis

Comparison with state-of-the-art methods. We compare our method against state-of-the-art WSI classification methods under the image-only inference setting. Results for all baseline methods are taken from PathM3 [10], which uses identical experimental settings, including the same dataset, data split ratio, and three GA subtypes for classification, thereby enabling direct comparison. As shown in Table 1, our method achieves 74.83% accuracy, outperforming all baselines, including PathM3 (71.48%) by 3.35 percentage points and the best attention-based MIL method ILRA-MIL (70.16%) by 4.67 percentage points. The consistent margin over established MIL baselines, including ABMIL, CLAM, DSMIL, TransMIL, and CITE, demonstrates the efectiveness of our approach in gastric adenocarcinoma subtype classification.

Ablation study. We conduct ablation studies to validate the key design choices in our framework, as described in the following three paragraphs. Notably, results of our ablation study are summarized in Table 2, where accuracy is determined in only one run of each model.

Multimodal Inference. We evaluated the teacher model independently under a multimodal inference setting using both image and text inputs. The teacher achieves the best performance among all ablated variants (see Table 2) by leveraging complementary information from the whole-slide image and the diagnostic report summary to classify GA subtypes. However, as discussed in Section 1, this multimodal setup is not clinically practical, since report generation requires clinician involvement and is assumed to be not available at inference time.

![](images/504ea4ecaceda162e51c20749c076faa82fb52915c18ed0930c405393f4859a3.jpg)  
(a) Efect of $\lambda _ { K D }$

![](images/7111d1bd92c42e971d2203bbadeca45ae054179de6408dffe6fe030c55c2c32f.jpg)  
(b) Efect of LMF rank.  
Fig. 3: Results for diferent choices of (a) the coeficient of logit distillation loss, $\lambda _ { K D }$ and (b) the rank in the low-rank multimodal fusion (LMF) approach

Importance of Distillation Loss Components. We ablate the proposed dualobjective distillation loss by training the student model with each component independently and in combination, while keeping the cross-entropy loss fixed. Using only the feature loss $\left( \mathcal { L } _ { \mathrm { f e a t } } \right)$ or only the KD logit loss $\left( \mathcal { L } _ { \mathrm { K D } } \right)$ leads to lower performance, lagging behind the combined objective by 1.20% and 1.70% points, respectively. In contrast, jointly optimizing both losses yields the best performance, demonstrating that they provide complementary supervision signals for more efective knowledge transfer.

Unimodal vs. Image-Only Multimodal KD. To quantify the contribution of multimodal fusion, we evaluate classifiers trained on individual modalities without fusion or KD. The image-only unimodal model achieves an accuracy of 70.35%, whereas the text-only model obtains 68.76%. Compared with these unimodal baselines, the distilled student model $( { \mathrm { i . e . } }$ , our image-only multimodal KD framework) improves the performance by 4.48 percentage points over the image-only model and by 6.07 percentage points over the text-only model. Furthermore, the full multimodal teacher model achieves a substantial gain of 17.91 percentage points over the distilled student model. These results demonstrate that the LMF-based fusion efectively leverages complementary information across multiple modalities and that the Feature Distiller successfully transfers this multimodal knowledge to our image-only inference pipeline.

Choice of logit distillation loss coeficient. We also experimented with tuning the $\lambda _ { K D }$ parameter to four settings (0.5, 1.0, 2.0, & 5.0). Our best results (74.83%) were achieved with $\lambda _ { K D } = 0 . 5$ . The accuracy of training the student with other values is presented in Figure 3a.

Choice of rank in LMF. We experimented with the teacher model’s rank hyperparameter for implementing LMF. The student model distils knowledge from the teacher model, which is trained with four diferent ranks: 16, 32, 64, and 128. The highest accuracy was observed with $r = 1 2 8 .$ , as shown in Fig. 3b.

![](images/e35d19cd81ba0c2f534f29669af7c65019771965098b694ed75e3ce0c8ccb51b.jpg)  
Fig. 4: t-SNE visualization of learned embeddings with corresponding DBI scores: image-only (5.2641), text-only (2.3107), teacher (0.7309), and distilled student (1.1078). Blue represents "moderately diferentiated tubular adenocarcinoma". Orange represents "poorly difrentiated adenocarcinoma". Green represents "well difrentiated tubular adenocarcinoma"

It is likely that a higher rank enables the teacher to learn richer multimodal representations via cross-modal interactions.

Feature visualizations. Fig. 4 shows t-SNE plots of learned representations which are $\mathbf { z } _ { i m g }$ (image-only features), $\mathbf { z } _ { t x t }$ (text-only features), $\mathbf { z } _ { f u s e d }$ (LMF representation of fusion of text and image embeddings) and $\hat { \mathbf { z } } _ { f u s e d }$ (Predicted embeddings by the knowledge distiller from the patch features only). To qualitatively assess the representation quality, we compute the Davies-Bouldin Index (DBI) on the learned embeddings. It measures the ratio of within-cluster dispersion to between-cluster separation, where lower values indicate better-defined clusters. The teacher model achieves the lowest DBI (0.7309), followed by the distilled student model (1.1078), both substantially outperforming the text-only (2.3107) and image-only (5.2641) representations, confirming the improved cluster compactness and inter-class separation.

## 5 Conclusion

We proposed an LMF-based multimodal fusion framework for image-only classification of gastric adenocarcinoma, fusing diagnostic captions with patches extracted from histopathological whole-slide images (WSIs). The proposed method outperforms the strongest baseline by 3.35%. Our results suggest that carefully chosen fusion mechanisms can outperform architecturally complex methods in computational pathology. While our framework demonstrates strong performance, it is limited to a single dataset and a single cancer type. Future work could explore evaluation across diverse datasets of various cancer types.

## References

1. Ilse, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: International Conference on Machine Learning, pp. 2127–2136. PMLR (2018)

2. Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X.: TransMIL: Transformer based correlated multiple instance learning for whole slide image classification. Advances in Neural Information Processing Systems 34, 2136–2147 (2021)

3. Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5(6), 555–570 (2021)

4. Li, B., Li, Y., Eliceiri, K.W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, (2021)

5. Chen, R.J., Lu, M.Y., Wang, J., Williamson, D.F., Rodig, S.J., Lindeman, N.I., Mahmood, F.: Pathomic fusion: An integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis. IEEE Transactions on Medical Imaging 41(4), 757–770 (2020)

6. Chen, R.J., Lu, M.Y., Weng, W.H., Chen, T.Y., Williamson, D.F., Manz, T., Shady, M., Mahmood, F.: Multimodal co-attention transformer for survival prediction in gigapixel whole slide images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4025 (2021)

7. Xu, Y., Chen, H.: Multimodal optimal transport-based co-attention transformer with global structure consistency for survival prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21241–21251 (2023)

8. Zhou, F., Chen, H.: Cross-modal translation and alignment for survival analysis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21485–21494 (2023)

9. Ding, K., Zhou, M., Metaxas, D.N., Zhang, S.: Pathology-and-genomics multimodal transformer for survival outcome prediction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 622–631. Springer (2023)

10. Zhou, Q., Zhong, W., Guo, Y., Xiao, M., Ma, H., Huang, J.: PathM3: A multimodal multi-task multiple instance learning framework for whole slide image classification and captioning. arXiv preprint arXiv:2403.08967 (2024)

11. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning, pp. 19730–19742. PMLR (2023)

12. Liu, Z., Shen, Y., Lakshminarasimhan, V.B., Liang, P.P., Zadeh, A., Morency, L.P.: Eficient low-rank multimodal fusion with modality-specific factors. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pp. 2247–2256 (2018)

13. Zadeh, A., Chen, M., Poria, S., Cambria, E., Morency, L.P.: Tensor fusion network for multimodal sentiment analysis. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 1103–1114 (2017)

14. Pan, X., An, Y., Lan, R., Liu, Z., Liu, Z., Lu, C., Yang, H.: PG-MLIF: Multimodal low-rank interaction fusion framework integrating pathological images and genomic data for cancer prognosis prediction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 1–10. Springer (2023)

15. Tsuneki, M., Kanavati, F.: Inference of captions from histopathological patches. In: International Conference on Medical Imaging with Deep Learning, pp. 1235–1250. PMLR (2022)

16. Filiot, A., Ma, Y., Mac Kain, A., Walker, C., Veuthey, B., Frati, P., Kestelman, F., Saillard, C., Bernigot, A., Benhenda, S.: Scaling self-supervised

learning for histopathology with masked image modeling. medRxiv (2023). https://doi.org/10.1101/2023.07.21.23292757

17. Xiao, S., Liu, Z., Zhang, P., Muennighof, N.: C-Pack: Packaged resources to advance general Chinese embedding. arXiv preprint arXiv:2309.07597 (2023)

18. Zhang, Y., Gao, J., Zhou, M., Wang, X., Qiao, Y., Zhang, S., Wang, D.: Text-guided foundation model adaptation for pathological image classification. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 272–282. Springer (2023)

19. Xiang, J., Zhang, J.: Exploring low-rank property in multiple instance learning for whole slide image classification. In: The Eleventh International Conference on Learning Representations (2023)

20. Dietterich, T.G., Lathrop, R.H., Lozano-Pérez, T.: Solving the multiple instance problem with axis-parallel rectangles. In: Artificial Intelligence Conference, pp. 234–248 (1997)

21. Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V.: Nyströmformer: A Nyström-based algorithm for approximating self-attention. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 14138–14148 (2021)

22. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022)

23. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. In: NIPS Deep Learning and Representation Learning Workshop (2015)

24. Romero, A., Ballas, N., Kahaei, S.E., Vincent, P., Bengio, Y.: FitNets: Hints for thin deep nets. In: International Conference on Learning Representations (2015)

25. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016)

26. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016)

27. Hendrycks, D., Gimpel, K.: Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415 (2016)

28. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research 15(1), 1929–1958 (2014)

29. Girshick, R.: Fast R-CNN. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 1440–1448 (2015)

30. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

31. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009 (2022)

32. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (2015)