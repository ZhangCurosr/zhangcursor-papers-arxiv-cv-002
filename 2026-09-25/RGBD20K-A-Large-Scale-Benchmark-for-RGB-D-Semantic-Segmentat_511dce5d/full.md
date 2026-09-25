# RGBD20K: A Large-Scale Benchmark for RGB-D Semantic Segmentation

Shaohua Dong, Zexuan Meng, Haiyan Sun, Bing Fan, Cuicui Zhang, Dylan Joseph, Kewei Sha, Yunhe Feng and Heng Fan

Abstract— In this paper, we propose RGBD20K, a novel dataset for facilitating the development of more robust and general RGB-D semantic segmentation by encompassing abundant categories and high-quality annotations. RGBD20K possesses several attractive properties: (1) Expanded Semantic Space. In particular, it covers 160 fine-grained categories, largely surpassing the category diversity of existing popular RGB-D benchmarks (e.g., NYUv2 with 40 classes and SUN RGB-D with 37 classes). With such enriched semantic coverage, we expect to promote the learning of more generalizable segmentation models. (2) Larger Scale. Compared with current benchmarks, RGBD20K offers 20,000 RGB-D image pairs, providing a substantially larger training resource that benefits the development of more powerful deep models. (3) High-Fidelity Annotation. We perform rigorous re-evaluation and correction of existing labels to resolve long-standing annotation noise, resulting in a clean and reliable ground-truth foundation. Furthermore, we propose a novel score-purified fusion (SPF) method, which achieves state-of-the-art performance across all evaluated benchmarks, demonstrating the effectiveness of our approach in leveraging high-quality multimodal information for RGB-D semantic segmentation. The dataset is here: RGBD20K.

## I. INTRODUCTION

Visual perception [1], [2], [3], [4] is a fundamental problem in computer vision, with semantic segmentation serving as a core task for achieving dense and structured scene understanding. It has been widely applied in robotics, autonomous systems, and intelligent perception, where pixellevel recognition of complex scenes is essential. Despite significant progress in deep learning-based RGB-D semantic segmentation, current methods [5], [6], [7] are still far from achieving robust and generalizable performance in real-world environments. A key limiting factor lies not only in model design, but more fundamentally in severe limitations of existing RGB-D datasets, as described in the following.

Limited data scale. NYUv1 [8] and NYUv2 [9] contain 2,347 and 1,449 annotated RGB-D image pairs, respectively (see Figure 1(a)), and serve as early benchmarks for RGB-D semantic segmentation. However, their limited scale significantly restricts the learning capacity of modern deep models. To address this limitation, SUN RGB-D [10] extends the dataset scale to 10,335 RGB-D images (see Figure 1 (a)). Although it significantly improves data availability and has played an important role in advancing RGB-D semantic segmentation, its scale is still insufficient for modern deep neural networks and vision transformers [11], [12], which typically require large-scale and diverse training data to fully exploit their representation capacity and achieve strong generalization performance.

Limited semantic coverage and scene diversity. Beyond data scale, existing benchmarks are also constrained by limited semantic coverage and restricted scene diversity. For example, NYUv1 [8], NYUv2 [9] and SUN RGB-D [10] contain only 13, 40 and 37 semantic categories, respectively (see Figure 1 (b)), which are insufficient to capture the fine-grained semantic structures present in real-world environments. In addition, data collection is largely limited to relatively constrained indoor settings, leading to insufficient variation in spatial layouts, lighting conditions, occlusions, and object arrangements (see Figure 1 (c)). Together, these limitations in both semantic richness and environmental diversity restrict the generalization ability of models when applied to more complex and open-world scenarios.

Limited annotation quality. The effectiveness of RGB-D semantic segmentation relies heavily on reliable annotations. However, existing datasets [10] often suffer from imperfect or noisy labels (see Figure 5). Such deficiencies not only compromise the reliability of evaluation but also impede the development of more advanced algorithms (see Section V-B.0.b). Consequently, improving annotation quality is crucial for enabling robust multimodal perception and more reliable model learning.

These limitations collectively highlight the need for a new generation of RGB-D benchmarks that provide richer semantic coverage, larger scale, and more reliable crossmodal alignment. To this end, we introduce RGBD20K, a large-scale benchmark designed to advance RGB-D semantic segmentation under real-world conditions. RGBD20K contains 20,000 high-quality RGB-D image pairs collected from diverse environments, with carefully refined annotations to ensure reliable supervision. Specifically, RGBD20K makes the following contributions:

(1) Large-scale high-quality RGB-D data. RGBD20K contains 20,000 high-quality RGB-D image pairs, providing a significantly larger-scale benchmark compared to existing datasets.

(2) Rich semantic coverage and diverse scene distribution. RGBD20K covers 160 fine-grained semantic categories, substantially exceeding existing benchmarks such as NYUv2 and SUN RGB-D, which typically contain fewer than 40 classes. In addition, the dataset spans 75 different scene types, further increasing its diversity and real-world complexity. This expanded data scale and semantic space jointly enable more comprehensive and generalizable learning of

![](images/0238f590ca5103ea0f213cda953740517b2d468885d26146d6885d0e6520808e.jpg)  
(a) Number of Images

![](images/00ae1ffaf680d3170ec0b107bf889ed59795c2707b716c347f12202e8acea932.jpg)  
(b) Object Categories

![](images/e6b0127807dd6ffcfebec0a8e5f864e8536386143f14b1cc301b48b51dbab6e9.jpg)  
(c) Scene Categories  
Fig. 1: Comparison of the proposed RGBD20K with other RGB-D datasets.

complex scene structures.

(3) High-quality annotation refinement. Unlike previous datasets that suffer from noisy labels, RGBD20K adopts a rigorous multi-stage manual refinement process to produce accurate pixel-level annotations. This process ensures clear semantic boundaries and high annotation consistency, thereby providing a more reliable supervision signal for model training.

Building upon this foundation, we further introduce the score-purified fusion (SPF) model, which follows a simple “purify-then-attend” design principle. By benefiting from the improved data quality and larger scale provided by RGBD20K, our method achieves more robust and generalizable semantic segmentation performance. By releasing RGBD20K and the SPF model, we aim to provide both a large-scale benchmark and a strong baseline to facilitate future research in robust and general-purpose RGB-D perception.

## II. RELATED WORK

RGB-D Semantic Segmentation Benchmarks. Benchmarks have been fundamental to the progression of multimodal scene understanding. Early RGB-D benchmarks were primarily indoor-centric and designed for small-scale evaluation. NYUv2 [9] and SUN RGB-D [10] established the initial standards, providing depth maps alongside semantic labels. However, these datasets are limited to fewer than 40 categories and often exhibit significant sensor noise and boundary misalignment. Later, ScanNet [13] offered a larger scale of 3D indoor data, but its focus remains on voxelized reconstruction rather than high-precision 2D semantic masks. Matterport3D [14] and 2D-3D-S [15] introduced ”buildingscale” data. Matterport3D offers 194,400 RGB-D images across 90 buildings, while 2D-3D-S provides 70,496 images. Despite their massive scale, these building-level datasets were designed with different objectives. 2D-3D-S focuses on structural parsing into only 13 coarse categories (e.g., wall, floor, ceiling), while Matterport3D primarily facilitates 3D reconstruction and room-level classification. Furthermore, because these datasets are captured as continuous scans, they often contain high redundancy and ”projected” labels that lack the pixel-level boundary precision. More recently, large-scale datasets such as RealSee3D [16] have introduced 10,000 unique indoor scenes combining real-world LiDAR captures with procedurally generated environments. While RealSee3D provides an unprecedented volume of multi-view panoramic data (nearly 300,000 viewpoints), its primary focus is on 3D reconstruction, floor plan generation, and 3D detection.

Despite the emergence of such large-scale resources, there remains a critical gap in fine-grained 2D semantic perception. Many massive datasets rely on automated or coarse annotations that lack the pixel-level precision and taxonomic depth required for nuanced scene understanding. To alleviate this, our RGBD20K provides 20,000 high-fidelity image pairs with a rigorously refined 160-class taxonomy. By bridging the gap between the massive scale of modern captures like RealSee3D and the high-precision requirements of semantic segmentation, RGBD20K serves as a more challenging and reliable foundation for next-generation multimodal fusion.

RGB-D Semantic Segmentation Algorithms. RGB-D semantic segmentation [17], [18], [5], [19] aims to improve recognition performance by incorporating depth information, which provides complementary 3D geometric cues that are often absent in RGB-only settings. Early mainstream approaches focused on designing complex interaction modules to fuse RGB and depth features extracted from two parallel pretrained backbones. For instance, CMX [20], TokenFusion [18], and GeminiFusion [21] integrate multimodal representations either within the encoder or during decoding to enhance performance. However, these dual-stream architectures face two key limitations: (1) the use of separate backbones introduces significant computational overhead, and (2) initializing depth streams with RGB-pretrained weights often leads to distribution mismatch. To address these issues, recent methods such as DPLNet [5] explore prompt-based designs to reduce the number of trainable parameters, while DFormer [6], [7] investigates unified RGB-D representation learning. By acknowledging the lower information density of depth data, DFormer allocates fewer channels to depth encoding, improving efficiency while mitigating distribution shift.

In contrast, our approach is instantiated as the scorepurified fusion (SPF) network, following a simple “purifythen-attend” design principle. By performing score-based feature purification prior to cross-modal interaction, the model enables a more direct and efficient utilization of multimodal cues compared to traditional interaction-heavy or unified-backbone paradigms.

Other Multi-modal Segmentation Benchmarks and Algorithms. Beyond the RGB-D domain, multi-modal semantic segmentation has been extensively explored to enhance robustness in adverse environments. In the field of RGB-Thermal (RGB-T) segmentation, benchmarks such as MFNet [22] and PST900 [23] were introduced to address challenges in low-illumination and nighttime scenarios. Building on these, SemanticRT [24] and the Multispectral Video Semantic Segmentation benchmark [25] have further scaled up the data volume and complexity, facilitating the development of multispectral algorithms that leverage the complementary nature of thermal and visual spectra. Recently, the DeLiVER benchmark [26] has pushed the boundaries of multi-modal research by providing a massive dataset covering Depth, LiDAR, multiple Views, Events, and RGB. The development of these benchmarks has driven a variety of multi-modal algorithms [27], [28], [29], [30], [31] designed to handle diverse sensing data. Early RGB-T segmenters focused on cross-modal fusion modules to align thermal and spatial features.

![](images/be3c41fc86a906f28f4e329d3c92bded168827cfbd63ff1ef050e13f0a134a7e.jpg)

(a) Pixel-level distribution  
![](images/10a6eb56ea217ea4080ef86978ee9914a7d8a95321ce789f636623716d93ebcd.jpg)  
(b) Category-level distribution  
Fig. 2: Visualization of RGBD20K distribution.

## III. THE PROPOSED RGBD20K

## A. Construction Principles

The primary goal of RGBD20K is to establish a comprehensive benchmark that provides a large-scale collection of images, rich object categories, and high-precision semantic annotations, thereby facilitating the development of more generalizable and robust RGB-D semantic segmentation methods. To this end, we follow the following principles in constructing RGBD20K:

TABLE I: Comparison of RGB-D semantic segmentation benchmarks.
<table><tr><td>Dataset</td><td>Year</td><td>Categories</td><td>Images</td><td>Scenarios</td><td>Masks</td><td>Avg./img</td></tr><tr><td>NYUv1 [8]</td><td>2011</td><td>13</td><td>2,347</td><td>7</td><td></td><td></td></tr><tr><td>NYUv2 [9]</td><td>2012</td><td>40</td><td>1,449</td><td>26</td><td>33,749</td><td>23.3</td></tr><tr><td>SUN RGB-D [10]</td><td>2015</td><td>37</td><td>10,335</td><td>47</td><td>146,617</td><td>14.2</td></tr><tr><td>RGBD20K (Ours)</td><td>2026</td><td>160</td><td>20,000</td><td>75</td><td>368,312</td><td>18.4</td></tr></table>

![](images/c92a26a5233aef3c416d058f8cbf824f42d33a7d99597bad9593d2961c998662.jpg)  
Fig. 3: Example images with annotations from RGBD20K.

Larger Scale. Large-scale data is crucial for training datadriven models. RGBD20K contains 20,000 RGB-D image pairs, exhibiting rich variations in illumination conditions, object arrangements, and scene layouts. Compared with existing RGB-D benchmarks, it significantly increases the data scale, providing stronger support for training more powerful segmentation models.

Vast Categories. A key objective of RGBD20K is to improve category diversity and generalization ability in RGB-D semantic segmentation. To this end, the dataset includes 160 fine-grained object classes covering a wide range of common indoor objects, enabling more detailed scene understanding and semantic reasoning. In addition, it includes 75 scene types, further enhancing environmental diversity and realworld complexity.

High-Quality Annotation. Annotation quality is critical for both training and evaluation in semantic segmentation. To ensure the high quality of RGBD20K, each RGB-D pair undergoes multiple rounds of manual inspection and refinement, which significantly improves boundary accuracy and annotation consistency while effectively reducing noise.

## B. Construction Principles

## C. Data Acquisition

RGBD20K is built through a large-scale curation and unification process of heterogeneous RGB-D sources to ensure both environmental and semantic diversity. We collect 20,000 depth-aligned image pairs from scene-centric and tracking-oriented benchmarks (see Table II), including SUN RGB-D [10], as well as subsets from RGB-D Mirror [32], VidSOD [33], DepthTrack [34], and ARKitTrack [36], together with 8,365 pairs from the DIML RGB-D dataset [37]. This integration covers a broad range of real-world indoor environments and scenarios.

TABLE II: Summary of RGBD20K Dataset Collection Sources. (SS: Semantic Segmentation, SOD: Salient Object Detection, VOT: Visual Object Tracking, PT: Pre-training.)
<table><tr><td>Dataset</td><td>Images</td><td>Task</td></tr><tr><td>SUN RGB-D [10]</td><td>10,335</td><td>SS</td></tr><tr><td>RGB-D mirror [32]</td><td>824</td><td>SS</td></tr><tr><td>VidSOD [33]</td><td>100</td><td>SOD</td></tr><tr><td>DepthTrack [34]</td><td>98</td><td>VOT</td></tr><tr><td>RGBD1K [35]</td><td>235</td><td>VOT</td></tr><tr><td>ARKitTrack [36]</td><td>43</td><td>VOT</td></tr><tr><td>DIML RGB-D [37]</td><td>8,365</td><td>PT</td></tr><tr><td>RGBD20K</td><td>20,000</td><td>SS</td></tr></table>

To ensure high fidelity, we perform a rigorous manual cleaning and re-annotation process, unifying these disparate sources under a single 160-class taxonomy. Each selected category has been verified by domain experts to ensure it is meaningful for semantic perception. The resulting dataset follows a natural long-tail distribution (see Figure 2), mirroring real-world object frequencies to encourage the development of models that generalize effectively across both common and infrequent classes. Ultimately, RGBD20K offers a vastly larger and more precise semantic foundation than legacy benchmarks, facilitating research in supervised, open-vocabulary, and zero-shot perception tasks.

## D. Annotation

We follow the similar principle as in [38], [39] for the semantic segmentation annotation. All images are annotated through a unified manual labeling process. Each RGB-D pair is processed by trained annotators using an interactive labeling interface, producing pixel-level semantic masks for all visible regions. A hierarchical labeling scheme is used, organizing concepts from coarse categories (e.g., furniture, appliances) to fine-grained classes (e.g., types of tables, electronic devices).

To handle occlusions in indoor scenes, we apply depthaware ordering when constructing final masks. Objects are assigned relative depth layers from the depth map, with background regions such as walls and floors placed at the farthest level. For overlaps, depth cues and mask geometry are used to determine consistent ordering, ensuring correct foreground–background relationships.

Unlike fixed-label benchmarks, RGBD20K supports flexible category refinement, allowing new semantic classes to be added during annotation for better coverage of real-world concepts. All regions are labeled at the semantic level to support segmentation and scene understanding.

Object parts are also annotated when applicable and linked to their parent objects, forming a lightweight hierarchical structure that reflects real-world composition (e.g., drawer–cabinet). Figure 3 displays several annotation examples.

## E. Dataset Split

RGBD20K consists of 20,000 RGB-D image pairs collected from diverse indoor environments. We adopt a standard benchmark split for training and evaluation, using 18,000 pairs for training and 2,000 pairs for testing. The split is performed in a stratified manner to preserve the distributions of scene types, object categories, and depth characteristics across both subsets. All 160 semantic categories are included in both training and testing sets, while maintaining a long-tailed distribution consistent with realworld indoor scenes. Although the test set accounts for only 10% of the data, it is designed to be representative of the full dataset while enabling efficient evaluation. This split follows common practice in large-scale indoor vision benchmarks, where a compact but diverse test set is used to balance efficiency and robustness.

## IV. METHODOLOGY: SCORE-PURIFIED FUSION MODEL

In RGB-D semantic segmentation, effectively fusing complementary information from heterogeneous modalities remains a challenging problem. Existing fusion methods, particularly those based on standard cross-attention, often suffer from attention dilution. This issue arises because the attention mechanism must simultaneously handle cross-modal inconsistencies (e.g., sensor noise and misaligned depth boundaries) while aggregating long-range contextual information, which can weaken discriminative feature learning. To address this problem, we propose the score-purified fusion (SPF) Network, following a simple “purify-then-attend” design principle. Instead of directly applying attention on raw projected features, SPF explicitly filters and refines the Key (K) and Value (V) representations at the linear projection stage before attention computation. Specifically, we introduce cross-examined reliability scores to assess feature consistency across modalities, enabling adaptive suppression of unreliable responses and enhancement of semantically consistent regions.

## A. Overall architecture

Our SPF model follows the GeminiFusion method [21], featuring a four-stage hierarchical encoder similar to Seg-Former [1]. As illustrated in Fig. 4, the network takes RGB and Depth images as inputs. Each modality is processed through shared encoder layers, which comprises Multi-Head Attention (MHA) and Feed-Forward Network (FFN) blocks to extract multi-scale features, which are then fused at each stage. For conciseness, Fig. 4 illustrates only the transformer blocks within the first stage rather than depicting all four hierarchical stages in detail. Different from GeminiFusion [21], our key contribution lies in the proposed Score-Purified Fusion module, which replaces the original fusion strategy for more effective multimodal feature integration. Finally, the fused features are passed to a SegFormer head decoder to produce the segmentation predictions.

![](images/28adc3c68a10a77e9076bee22b3aa2594eb84c886ae0b4b1f412f9a9aeae5632.jpg)  
Fig. 4: The overall architecture of the proposed score-purified fusion model.

## B. Score-Purified Fusion

The core of the SPF module lies in Reciprocal Score Generation and Score-Guided Manifold Purification. As shown in Fig. 4, for simplicity, we omit the block index i in the following formulations and present the operations at a representative layer without loss of generality.

a) Reciprocal Score Generation.: Let $Q _ { r g b } , Q _ { d } \in$ $\mathbb { R } ^ { N \times D ^ { \prime } }$ be the outputs of the Multi-Head Attention. To estimate the dynamic reliability scores of features from each modality, we introduce the Score Head. We first project the raw features into aligned embeddings to enable cross-modal comparison within a balanced representation space:

$$
F _ { r g b } = W _ { r g b } Q _ { r g b } + b _ { r g b } , \quad F _ { d } = W _ { d } Q _ { d } + b _ { d }\tag{1}
$$

where $W _ { r g b } , W _ { d } \in \mathbb { R } ^ { D \times D }$ and $b _ { r g b } , b _ { d } \in \mathbb { R } ^ { D }$ are learnable projection parameters. Rather than applying a simple heuristic fusion, a Relation Arbiter is introduced to perform a finegrained cross-examination. We estimate the fused importance scores $S ^ { f }$ that capture the pixel-wise reliability of each stream:

$$
S _ { r g b } ^ { f } = \sigma ( \mathrm { M L P } ( [ Q _ { r g b } ; F _ { d } ] ) ) , \quad S _ { d } ^ { f } = \sigma ( \mathrm { M L P } ( [ Q _ { d } ; F _ { r g b } ] ) )\tag{2}
$$

where [ ] denotes channel-wise concatenation and σ is the Sigmoid function. These scores are subsequently decomposed into intra-modal $( S _ { r g b } , S _ { d } )$ and cross-modal $( S _ { r g b  d } , S _ { d  r g b } )$ components via a Split operation.

$$
S _ { r g b } , S _ { r g b  d } = \mathrm { S p l i t } ( S _ { r g b } ^ { f } ) , S _ { d } , S _ { d  r g b } = \mathrm { S p l i t } ( S _ { d } ^ { f } )\tag{3}
$$

b) Score-Guided Component Enhancement.: The critical innovation of our approach is the construction of purified Keys (K) and Values (V). By adaptively weighting the learnable noise e and the cross-modal features F by their respective reliability scores, we perform a Purified Alignment:

$$
\begin{array} { c } { { { \cal K } _ { r g b } = \displaystyle { \frac { ( Q _ { r g b } + e _ { r g b } ^ { k } ) \cdot S _ { r g b } + F _ { d } \cdot S _ { r g b  d } } { S _ { r g b } + S _ { r g b  d } } } , } } \\ { { { \cal V } _ { r g b } = \displaystyle { \frac { ( Q _ { r g b } + e _ { r g b } ^ { v } ) \cdot S _ { r g b } + F _ { d } \cdot S _ { r g b  d } } { S _ { r g b } + S _ { r g b  d } } } . } } \\ { { { } } } \\ { { { \cal K } _ { d } = \displaystyle { \frac { ( Q _ { d } + e _ { d } ^ { k } ) \cdot S _ { d } + F _ { r g b } \cdot S _ { d  r g b } } { S _ { d } + S _ { d  r g b } + \epsilon } } , } } \\ { { { } } } \\ { { { \cal V } _ { d } = \displaystyle { \frac { ( Q _ { d } + e _ { d } ^ { v } ) \cdot S _ { d } + F _ { r g b } \cdot S _ { d  r g b } } { S _ { d } + S _ { d  r g b } + \epsilon } } . } } \end{array}\tag{4}
$$

(5)

where · denotes the Hadamard product, $e ^ { k } , e ^ { v } \in \mathbb R ^ { N \times D }$ are learnable noise components capturing modality-specific uncertainty, and ϵ is a stability constant. This mathematical formulation allows the model to selectively filter out crossmodal noise (e.g., depth edge artifacts) before the attention mechanism is invoked, preventing attention dilution.

With the purified Key (K) and Value (V) manifolds established, standard Multi-Head Attention (MHA) operates on a noise-robust latent space. This eliminates the burden of noise resolution from the attention mechanism, allowing it to focus entirely on high-fidelity context aggregation:

$$
\begin{array} { r l } & { \boldsymbol { \mathrm { O } } _ { r g b } = \boldsymbol { \mathrm { M H A } } ( Q _ { r g b } , K _ { r g b } , V _ { r g b } ) , } \\ & { \quad \boldsymbol { \mathrm { O } } _ { d } = \boldsymbol { \mathrm { M H A } } ( Q _ { d } , K _ { d } , V _ { d } ) . } \end{array}\tag{6}
$$

![](images/3b45a19a75a3f56ac814fa6bd1d4a30ae02a3b68918727435d8716b08347b168.jpg)  
Label Omission

![](images/901cd79f2a7c2678c82d173a10d224e80a4cb42a4532574d8053e16866e6ad0a.jpg)

![](images/614ee5f43c6b532e064c72cb98297a5ddb52e616798321485e5f87caefa4c79f.jpg)  
Semantic Mislabeling

Complete Labeling  
![](images/2c2f363518bb0c95b4571413786e8d8f40711ac9b58f82c3eae67a85013a7fad.jpg)  
Accurate Categorization

![](images/80990fe346ec04a0079fd955b8392e94bad8d98817c0a908c29b394fd6a6064e.jpg)

![](images/e5c3c24aae3c2dcda523ffb824ab4b6d630f4dea5a70e474f01be00f75746309.jpg)  
Labeling Consistency

Annotation Inconsistency  
![](images/a947c202afd74f17914853243431dd7039d93669cdfe6cce0bb1c406096c6167.jpg)  
Coarse Boundaries

![](images/0e8f364bd9b4176c1a542b542b5f674a2668497af19cca98cdbd0cbc6112b9f4.jpg)  
Precise Boundaries  
(a) Previous SUN RGB-D  
(b) Our Re-annotation  
Fig. 5: Comparison of annotation quality. (a) Previous dataset, (b) our high-quality results.

The output features are then added back to the original identity streams via residual skip connections, followed by the Feed-Forward Network (FFN) within the Transformer block.

## V. EXPERIMENTS

## A. Datasets and Implementation Details

a) Datasets and Metrics: To comprehensively evaluate our multimodal semantic segmentation method, we conduct experiments on three widely adopted benchmarks: NYUv2 [9], SUN RGB-D [10], and our newly proposed RGBD20K dataset, which together cover diverse indoor scenes and object categories, enabling a thorough assessment of model generalization across different data scales and complexities. Specifically, NYUv2 contains 795 training images and 654 testing images across 40 semantic categories, and all inputs are processed at a resolution of 480 × 640 following GeminiFusion [21] for fair comparison. SUN RGB-D includes 5,285 training images and 5,050 testing images across 37 categories, making it approximately 7× larger than NYUv2, and uses an input resolution of $4 8 0 \times 4 8 0$ for evaluation. Lastly, our RGBD20K dataset consists of 18,000 training images and 2,000 testing images spanning 160 categories, and is evaluated with an input resolution of $4 8 0 \times 6 4 0 .$ , providing more fine-grained annotations and significantly increasing task difficulty. Following standard evaluation protocols, we report mean Intersection-over-Union (mIoU), computed as the average IoU across all semantic categories.

b) Implementation Details: Following the standard training setting in GeminiFusion [21], our backbone is Swin-

Transformer Large, we optimize our models using a weight decay of 0.01. The entire training process spans 300 epochs, utilizing a learning rate schedule divided into three equal stages of 100 epochs each. Specifically, we set initial learning rates (LR) of $6 \times 1 0 ^ { - 5 } , 3 \times 1 0 ^ { - 5 }$ , and $1 . 5 \times 1 0 ^ { - 5 }$ for the first, second, and final 100 epochs, respectively. Throughout training, the learning rate in each stage is scheduled using the poly decay strategy with a power of 0.01. Our models are trained on four NVIDIA H100 GPUs.

## B. Comparisons with the State of the Art

a) Main Results: We compare our proposed SPF against a wide range of representative RGB-D semantic segmentation methods, including PGDENet [17], TokenFusion [18], GeminiFusion [21], MultiMAE [40], CMX [20], CMNeXt [26], DFormer v2 [7] and DPLNet [5], on the NYUv2 [9], SUN RGB-D [10], and our proposed RGBD20K datasets. For RGBD20K, we follow the original evaluation settings used by each method on SUN RGB-D for a fair comparison. The results are reported in Table III.

b) SUN RGB-D Annotation Quality Analysis: During dataset construction, we identified several annotation issues in SUN RGB-D, including category ambiguity, inaccurate labels, and noisy object boundaries. As shown in Figure 5, we provide qualitative comparisons between the original and refined annotations to illustrate these issues. To further quantify their impact, we refine only the test set annotations and evaluate a pretrained DPLNet [5] on the refined test set. This improves mIoU from 52.8 to 55.0, demonstrating that annotation inconsistencies can substantially affect evaluation and lead to an underestimation of model performance. We replace the original annotations with our refined labels while retaining the original 37-category setting, yielding SUN $\mathbf { R G B - D } ^ { \dagger }$ . We retrain representative RGB-D semantic segmentation models on this refined dataset. Under this setting, DPLNet [5] achieves an mIoU of 59.0, demonstrating that annotation refinement can substantially improve model performance when applied to both training and test data. As shown in Table III, all evaluated methods consistently achieve higher performance on SUN RGB-D<sup>†</sup>, demonstrating the effectiveness of our annotation refinement. These results highlight the significant impact of annotation quality on both evaluation and model training. Annotation inconsistencies in the training data can introduce noisy supervision, potentially impairing feature learning and model generalization. Based on these observations, we adopt the same annotation refinement principles throughout the construction of RGBD20K to ensure consistent and high-quality semantic annotations.

## C. Ablation Studies

We conduct extensive ablation experiments on RGBD20K to validate the effectiveness of each component in our Score-Purified Fusion method.

Effect of RGB and Depth Alignment Projections. We first analyze the impact of cross-modal alignment. We remove the specific projection layers for the RGB and depth modalities, processing both branches independently without explicit alignment. As shown in Table IV, this variant leads to a noticeable performance drop, demonstrating that crossmodal projection is essential for effective feature alignment and interaction.

TABLE III: Comparison of RGBD20K to existing datasets using mean IoU. <sup>†</sup> denotes our re-annotation.
<table><tr><td></td><td>Backbone</td><td>Params</td><td>NYUv2</td><td>SUN RGB-D</td><td>SUN RGB-D†</td><td>RGBD20K</td></tr><tr><td>PGDENet [17]</td><td>ResNet-34</td><td>100.7M</td><td>53.7</td><td>51.0</td><td>54.6</td><td>22.4</td></tr><tr><td>TokenFusion [18]</td><td>MiT-B3</td><td>45.9M</td><td>54.2</td><td>51.4</td><td>56.2</td><td>32.8</td></tr><tr><td>GeminiFusion [21]</td><td>Swin-L-384</td><td>369.2M</td><td>60.2</td><td>54.6</td><td>61.5</td><td>45.4</td></tr><tr><td>MultiMAE [40]</td><td>ViT-B</td><td>95.2M</td><td>56.0</td><td>51.1</td><td>55.7</td><td>42.2</td></tr><tr><td>CMX [20]</td><td>MiT-B5</td><td>181.1M</td><td>56.9</td><td>52.4</td><td>55.7</td><td>43.9</td></tr><tr><td>CMNeXt [26]</td><td>MiT-B4</td><td>119.6M</td><td>56.9</td><td>51.9</td><td>55.4</td><td>43.1</td></tr><tr><td>DFormer v1 [6]</td><td>DFormer-L</td><td>39.0M</td><td>57.2</td><td>52.5</td><td>59.1</td><td>41.9</td></tr><tr><td>DFormer v2 [7]</td><td>DFormer-v2-L</td><td>95.5M</td><td>58.4</td><td>53.3</td><td>59.7</td><td>44.1</td></tr><tr><td>DPLNet [5]</td><td>MiT-B5</td><td>88.6M</td><td>59.3</td><td>52.8</td><td>59.0</td><td>28.0</td></tr><tr><td>SPF (Ours)</td><td>Swin-L-384</td><td>416.4M</td><td>60.5</td><td>55.0</td><td>62.2</td><td>46.3</td></tr></table>

TABLE IV: Ablation study on different components.
<table><tr><td>Structure</td><td>mIoU (%)</td></tr><tr><td>Score-Purified Fusion</td><td>46.3</td></tr><tr><td>without Projection</td><td>45.7 (-0.6)</td></tr><tr><td>without Score with Score (K)</td><td>45.5 (-0.8) 46.0 (-0.3)</td></tr><tr><td>with Score (V)</td><td>45.9 (-0.4)</td></tr><tr><td>with Score (RGB) with Score (Depth)</td><td>45.8 (-0.5) 46.0 (-0.3)</td></tr></table>

Evaluation of the Score-Guided Enhancement Strategy. We conduct a comprehensive analysis of our method by evaluating the application of our score to enhance the Key (K) and Value (V) representations across different modalities and components. First, restricting the score enhancement to either the RGB branch or the depth branch alone, while reverting the other to a simple summation, yields inferior results, confirming that bidirectional enhancement is essential for balanced cross-modal learning. Second, independently applying the purification score to either the Key or Value alone while using simple summation for the other underperforms the complete model. The best results are achieved by jointly enhancing both components, demonstrating that consistent refinement of both attention computation (Key) and feature retrieval (Value) is critical. Results are shown in Table IV.

Effect of RGB and Depth noise selection. We also conduct ablation studies on noise selection strategies, with results reported in Table V. Our findings show that the best performance is achieved by introducing a learnable parameter into the key, where this parameter is independently defined for each layer. Replacing this design with a simple Multiply operation leads to a slight performance drop of 0.2% (46.1% mIoU). This observation is consistent with the findings of GeminiFusion [21], further demonstrating the effectiveness of layer-specific learnable noise for cross-modal alignment.

Effect of Relation Arbiter Design. We investigate the architectural design of the Relation Arbiter on the RGBD20K dataset, with results summarized in Table VI. Our experiments compare different transformation layers and activation functions to determine the most effective way to modulate cross-modal relationships. We find that a 2-layer MLP combined with a Sigmoid activation achieves the highest performance, reaching 46.3% mIoU.

TABLE V: Ablation about the noise selection on the RGBD20K dataset.
<table><tr><td>Structure</td><td>mIoU (%)</td></tr><tr><td>Learnable Noise, Add</td><td>46.3</td></tr><tr><td>Learnable Noise, Multiply Random Gaussian Noise, Add</td><td>46.1 (-0.2)</td></tr><tr><td>Random Gaussian Noise, Multiply</td><td>45.3 (-1.0) 45.5 (-0.8)</td></tr></table>

TABLE VI: Ablation about the Relation Arbiter on the RGBD20K dataset.

<table><tr><td>Structure</td><td>mIoU (%)</td></tr><tr><td>2layer-MLP + Sigmoid</td><td>46.3</td></tr><tr><td> $2 \mathrm { l a y e r – M L P } + \mathrm { S o f t m a x }$ </td><td>45.8 (-0.5)</td></tr><tr><td> $1 \times 1 \mathrm { C N N } + \mathrm { S i g m o i d }$ </td><td> $4 5 . 5 ~ ( - 0 . 8 )$ </td></tr><tr><td> $3 \times 3 \mathrm { \ C N N + { \cal S i g m o i d } }$ </td><td>45.4 (-0.9)</td></tr></table>

## VI. CONCLUSIONS

We introduce RGBD20K, a large-scale benchmark for RGB-D semantic segmentation. To bridge the gap in taxonomic diversity and annotation quality, RGBD20K provides 20,000 RGB-D image pairs annotated across 160 fine-grained categories. As one of the most comprehensive RGB-D benchmarks to date, it establishes a high-fidelity foundation for training general-purpose perception models. Furthermore, its dense, depth-aligned ground truth enables a deeper exploration of multimodal synergy, addressing the limitations of RGB-only approaches in complex scenes. To set a robust baseline for future research, we extensively evaluate representative segmentation models on RGBD20K. Additionally, we propose the score-purified fusion method, which achieves state-of-the-art performance across all evaluated benchmarks, demonstrating its effectiveness. By releasing RGBD20K, we aim to advance next-generation RGB-D semantic perception for robotic and autonomous systems.

[1] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.

[2] L. Peng, J. Gao, X. Liu, W. Li, S. Dong, Z. Zhang, H. Fan, and L. Zhang, “Vasttrack: Vast category visual object tracking,” Advances in Neural Information Processing Systems, vol. 37, pp. 130 797– 130 818, 2024.

[3] S. Dong, Y. Feng, Q. Yang, Y. Lin, and H. Fan, “Loretrack: efficient and accurate low-resolution transformer tracking,” arXiv preprint arXiv:2405.17660, 2024.

[4] W. Li, S. Dong, H. Lu, Y. Zhang, H. Fan, and L. Zhang, “Dmtrack: Spatio-temporal multimodal tracking via dual-adapter,” arXiv preprint arXiv:2508.01592, 2025.

[5] S. Dong, Y. Feng, Q. Yang, Y. Huang, D. Liu, and H. Fan, “Efficient multimodal semantic segmentation via dual-prompt learning,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 14 196–14 203.

[6] B. Yin, X. Zhang, Z. Li, L. Liu, M.-M. Cheng, and Q. Hou, “Dformer: Rethinking rgbd representation learning for semantic segmentation,” arXiv preprint arXiv:2309.09668, 2023.

[7] B.-W. Yin, J.-L. Cao, M.-M. Cheng, and Q. Hou, “Dformerv2: Geometry self-attention for rgbd semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 19 345–19 355.

[8] N. Silberman and R. Fergus, “Indoor scene segmentation using a structured light sensor,” in 2011 IEEE international conference on computer vision workshops (ICCV workshops). IEEE, 2011, pp. 601– 608.

[9] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in European conference on computer vision. Springer, 2012, pp. 746–760.

[10] S. Song, S. P. Lichtenberg, and J. Xiao, “Sun rgb-d: A rgb-d scene understanding benchmark suite,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 567–576.

[11] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[12] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 012–10 022.

[13] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[14] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3d: Learning from rgb-d data in indoor environments,” arXiv preprint arXiv:1709.06158, 2017.

[15] I. Armeni, S. Sax, A. R. Zamir, and S. Savarese, “Joint 2d-3d-semantic data for indoor scene understanding,” arXiv preprint arXiv:1702.01105, 2017.

[16] L. Li, Y. Wu, X. Li, L. Wang, T. Rao, J. Zhou, C. Pan, and X. Hui, “Realsee3d: A large-scale multi-view rgb-d dataset of indoor scenes (version 1.0),” 2025. [Online]. Available: https: //doi.org/10.5281/zenodo.17826243

[17] W. Zhou, E. Yang, J. Lei, J. Wan, and L. Yu, “Pgdenet: Progressive guided fusion and depth enhancement network for rgb-d indoor scene parsing,” IEEE Transactions on Multimedia, vol. 25, pp. 3483–3494, 2022.

[18] Y. Wang, X. Chen, L. Cao, W. Huang, F. Sun, and Y. Wang, “Multimodal token fusion for vision transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 186–12 195.

[19] J. Cai, J. Su, Q. Li, W. Yang, S. Wang, T. Zhao, S. He, and W. Liu, “Keep the balance: A parameter-efficient symmetrical framework for rgb+ x semantic segmentation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 10 587–10 598.

[20] J. Zhang, H. Liu, K. Yang, X. Hu, R. Liu, and R. Stiefelhagen, “Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers,” IEEE Transactions on intelligent transportation systems, vol. 24, no. 12, pp. 14 679–14 694, 2023.

[21] D. Jia, J. Guo, K. Han, H. Wu, C. Zhang, C. Xu, and X. Chen, “Geminifusion: Efficient pixel-wise multimodal fusion for vision transformer,” arXiv preprint arXiv:2406.01210, 2024.

[22] Q. Ha, K. Watanabe, T. Karasawa, Y. Ushiku, and T. Harada, “Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 5108–5115.

[23] S. S. Shivakumar, N. Rodrigues, A. Zhou, I. D. Miller, V. Kumar, and C. J. Taylor, “Pst900: Rgb-thermal calibration, dataset and segmentation network,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 9441–9447.

[24] W. Ji, J. Li, C. Bian, Z. Zhang, and L. Cheng, “Semanticrt: A large-scale dataset and method for robust semantic segmentation in multispectral images,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 3307–3316.

[25] W. Ji, J. Li, C. Bian, Z. Zhou, J. Zhao, A. L. Yuille, and L. Cheng, “Multispectral video semantic segmentation: A benchmark dataset and baseline,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1094–1104.

[26] J. Zhang, R. Liu, H. Shi, K. Yang, S. Reiß, K. Peng, H. Fu, K. Wang, and R. Stiefelhagen, “Delivering arbitrary-modal semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1136–1147.

[27] W. Zhou, S. Dong, C. Xu, and Y. Qian, “Edge-aware guidance fusion network for rgb–thermal scene parsing,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, 2022, pp. 3571–3579.

[28] W. Zhou, S. Dong, J. Lei, and L. Yu, “Mtanet: Multitask-aware network with hierarchical multimodal fusion for rgb-t urban scene understanding,” IEEE Transactions on Intelligent Vehicles, vol. 8, no. 1, pp. 48–58, 2022.

[29] W. Zhou, S. Dong, M. Fang, and L. Yu, “Cacfnet: Cross-modal attention cascaded fusion network for rgb-t urban scene parsing,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 1919–1929, 2023.

[30] S. Dong, W. Zhou, X. Qian, and L. Yu, “Gebnet: Graph-enhancement branch network for rgb-t scene parsing,” IEEE Signal Processing Letters, vol. 29, pp. 2273–2277, 2022.

[31] S. Dong, W. Zhou, C. Xu, and W. Yan, “Egfnet: Edge-aware guidance fusion network for rgb–thermal urban scene parsing,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 1, pp. 657– 669, 2023.

[32] H. Mei, B. Dong, W. Dong, P. Peers, X. Yang, Q. Zhang, and X. Wei, “Depth-aware mirror segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3044–3053.

[33] J. Lin, L. Zhu, J. Shen, H. Fu, Q. Zhang, and L. Wang, “Vidsod-100: A new dataset and a baseline model for rgb-d video salient object detection,” International Journal of Computer Vision, vol. 132, no. 11, pp. 5173–5191, 2024.

[34] S. Yan, J. Yang, J. Kapyl ¨ a, F. Zheng, A. Leonardis, and J.-K. ¨ Kam¨ ar¨ ainen, “Depthtrack: Unveiling the power of rgbd tracking,” in¨ Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 725–10 733.

[35] X.-F. Zhu, T. Xu, Z. Tang, Z. Wu, H. Liu, X. Yang, X.-J. Wu, and J. Kittler, “Rgbd1k: A large-scale dataset and benchmark for rgb-d object tracking,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 3870–3878.

[36] H. Zhao, J. Chen, L. Wang, and H. Lu, “Arkittrack: a new diverse dataset for tracking using mobile rgb-d data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5126–5135.

[37] J. Cho, D. Min, Y. Kim, and K. Sohn, “Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes,” arXiv preprint arXiv:2110.11590, 2021.

[38] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” International journal of computer vision, vol. 127, no. 3, pp. 302–321, 2019.

[39] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” International journal of computer vision, vol. 88, no. 2, pp. 303–338, 2010.

[40] R. Bachmann, D. Mizrahi, A. Atanov, and A. Zamir, “Multimae: Multi-modal multi-task masked autoencoders,” in European conference on computer vision. Springer, 2022, pp. 348–367.