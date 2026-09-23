# MorphoSHAP: Rethinking the Unit of Attribution in Explanation for Deep Visual Models

Anirudh Prabhakaran, Alexandre Rocchi, Gianni Franchi

AMIAD, Pole Recherche, Palaiseauˆ

Corresponding author: anirudh.prabhakaran@ip-paris.fr

## Abstract

Visual attribution methods typically explain predictions using pixels, superpixels, or regular patches. These representations can localize important regions, but provide limited information about their structure. We introduce MOR-PHOSHAP, a model-agnostic post-hoc method that instead uses morphological shapes as the players of a Shapley attribution game. Using the Tree of Shapes, each shape is described by its scale, geometry, and signed contribution, providing explanations of where the evidence lies, what type of structure carries it, and how strongly it affects the prediction. This shared morphological vocabulary enables spatial, textual, and global class-level explanations beyond image-specific heatmaps. To the best of our knowledge, MORPHOSHAP is the first SHAP-based image attributionframework to combine these differentforms ofexplanation. Acrossfive diverse datasets and three architectures, MORPHOSHAP achieves strong insertion/deletion performance and outperforms competing attribution methods on several benchmarks. Finally, a user study shows that MOR-PHOSHAP provides explanations that are easy to use and are preferred over standard attribution baselines.

## 1. Introduction

Understanding why a Deep Neural Network (DNN) makes a prediction remains a central challenge in explainable artificial intelligence (XAI). In computer vision, attribution methods assign importance scores to parts of an image to identify the evidence supporting or opposing a prediction. Existing approaches include pixel-level saliency maps [2, 4, 11, 12, 26, 42, 45], activation-based explanations [40], local surrogate methods [36], and Shapley-based attributions [32]. Despite their differences, these methods share the same goal: determining which elements of the image matter for the prediction.

![](images/0fc5d9cc1182f1a350a2d617c37343e0520f30a8d6af37405890ceb59b07eedb.jpg)  
Figure 1. Rethinking the unit of visual attribution. While conventional methods explain predictions through pixels, patches, or superpixels, MORPHOSHAP uses morphological shapes as Shap ley players.

This raises a more basic question: what should be the elementary unit of visual attribution? Pixel-based methods use individual pixels, perturbation methods often rely on superpixels, and vision transformers naturally suggest regular patches. However, this choice directly defines the space in which the explanation is expressed. Changing the elementary units changes the players of the attribution problem, and therefore changes the explanation itself. We argue that the choice of the explanatory representation should be treated as a central design decision in visual XAI.

Pixels, superpixels, and patches are useful representations, but they do not necessarily correspond to interpretable visual structures. Pixels are highly localized but have little meaning in isolation. Superpixels provide spatially coherent regions, but may split one structure or merge several structures. Regular patches are convenient computational units, but their boundaries are defined by a grid rather than by image geometry. As a result, conventional attribution maps can tell us where important evidence is located, but provide limited information about what type of visual structure the model relies on.

In this work, we investigate a different representation: morphological shapes as elementary units of attribution. Images contain connected structures and shapes at different scales that can be described through simple geometric properties such as size, elongation, circularity, or rectangularity. These properties are both spatially meaningful and nameable. Shape therefore provides a natural bridge between the localization offered by attribution maps and a structured description of the visual evidence used by the model.

Mathematical morphology [41] provides a natural framework for extracting such structures. Based on this idea, we introduce MORPHOSHAP, a model-agnostic post-hoc framework for shape-based attribution. Given an image, MORPHOSHAP uses the Tree of Shapes [1, 7, 19] to extract a hierarchical set of morphological shapes. Each shape is then characterized by its scale and geometric type, yielding descriptions such as small circle, medium rectangle, or large elongated structure. These morphological shapes, rather than pixels or superpixels, are finally used as the players of a cooperative game, and their contribution to the prediction is estimated using Shapley values [32]. The main difference with standard Shapley-based image explanations therefore lies not in the attribution rule itself, but in the representation on which the Shapley game is defined.

This representation provides richer explanations than a conventional attribution map. For each image, MOR-PHOSHAP associates every relevant structure with its spatial support, scale, geometric type, and signed contribution. The same explanation can therefore be shown as a heatmap or with a classical SHAP plot (Waterfall, or Beeswarm plot, ...) or expressed in words, e.g., “the prediction is mainly supported by a large elongated structure and a small circular region.” Moreover, because scale and geometric labels have the same meaning across images, these explanations can be aggregated over a dataset. This allows us to study, for example, whether a class is systematically associated with small circular structures, medium rectangles, or large elongated shapes. Thus, the same morphological vocabulary supports both local and global model analysis.

Our contributions are summarized as follows:

• We introduce MORPHOSHAP, a model-agnostic posthoc method that uses morphological shapes as the elementary units of a Shapley attribution game.

• We define a structured morphological explanation in which each shape is described by its scale, geometry, and signed contribution.

• This shared vocabulary enables local, textual, and global explanations beyond standard heatmaps. To the best of our knowledge, MORPHOSHAP is the first SHAP-based image attribution framework to provide this combination of spatial, geometric, textual, and global explanations.

• Experiments across datasets and architectures show that MORPHOSHAP provides faithful shape-level attributions with more structured and interpretable explanations than pixel-, patch-, and superpixel-based alternatives.

## 2. Related Work

## 2.1. Gradient-Based Attribution

Gradient-based methods such as Grad-CAM [40], Grad-CAM++ [10], Layer-CAM [25], Score-CAM [47] and ShapleyCAM [5] exploit model gradients or activation maps to produce spatial heatmaps. These techniques are computationally efficient and widely adopted for visualizing classifier decisions. However, they are inherently model-dependent: they require access to the internal layer activations and gradients, and they lack precise object boundaries, often highlighting diffuse regions rather than distinct entities. Furthermore, gradient saturation and reliance on specific architectural choices limit their applicability across diverse model families.

## 2.2. Hierarchical and Structured SHAP

Several works adapt Shapley-based attribution to structured image representations in order to reduce the cost and improve the coherence of visual explanations [20, 21, 23, 32, 35, 46]. Hierarchical methods such as h-Shap [46] and ShapBPT [35] organize image regions into trees and compute hierarchical Shapley/Owen-style contributions, while superpixel-based methods exploit region correlations or affinities to reduce the attribution game [20, 21]. Other approaches use learned segmentation or concepts, e.g., Explain Any Concept [44], which relies on SAM [27] to define interpretable image regions. In contrast, MORPHOSHAP defines the Shapley players directly from a multi-scale morphological hierarchy, providing each region with an explicit scale and geometric description without relying on a regular partition or an external segmentation model.

## 2.3. Mathematical Morphology for XAI

Mathematical morphology provides a principled framework for describing image structures through shape, connectivity, and scale [3, 41], and has been widely used to characterize visual patterns in microscopic and remote-sensing imagery [8, 9, 15–17]. Morphological operators have also been integrated into neural networks [18, 33, 34, 38], mainly as feature-extraction or architectural components rather than as post-hoc explanatory units. There have been multiple morphological decompositions; in particular, the Tree of Shapes provides a self-dual hierarchical decomposition of an image into nested connected structures at multiple scales [1, 7, 19]. In contrast to prior uses of morphology, MORPHOSHAP exploits these structures directly as the units of attribution, linking mathematical morphology with XAI.

## 3. Method

## 3.1. Overview and Problem Formulation

Let $x \in \mathbb { R } ^ { H \times W \times C }$ be an input image and $f : \mathbb { R } ^ { H \times W \times C } $ $\mathbb { R } ^ { K }$ a black-box classifier. For a target class y, we explain the corresponding model output $f _ { y } ( x )$ , taken as the class logit unless stated otherwise.

The key idea of MORPHOSHAP is to define attribution over morphological shapes, rather than pixels, superpixels, or regular patches. Given $x ,$ we extract M morphological shapes and progressively enrich each shape $b _ { i }$ with its scale $s _ { i } ,$ geometric category $g _ { i } ,$ and estimated contribution $\ddot { \phi } _ { i }$

$$
b _ { i } \longrightarrow ( b _ { i } , s _ { i } ) \longrightarrow ( b _ { i } , s _ { i } , g _ { i } ) \longrightarrow ( b _ { i } , s _ { i } , g _ { i } , \hat { \phi } _ { i } ) .\tag{1}
$$

The resulting explanation is

$$
\mathcal { E } _ { y } ( x ) = \Big \{ ( b _ { i } , s _ { i } , g _ { i } , \hat { \phi } _ { i } ) \Big \} _ { i = 1 } ^ { M } .\tag{2}
$$

Each explanatory unit therefore specifies where the evidence lies, at what scale, with which geometry, and how strongly it contributes to the prediction.

## 3.2. Morphological Shape Decomposition

Tree of Shapes. Among morphological decompositions [1], we use the Tree of Shapes because it directly represents an image as a hierarchy of connected shapes. Although extensions to color images exist [6], we present the formulation for a grayscale image u for simplicity.

Upper and lower level sets. For a gray level $\lambda ,$ the lower and upper level sets are

$$
L _ { \lambda } ( u ) = \{ p \in \Omega : u ( p ) < \lambda \} ,\tag{3}
$$

$$
U _ { \lambda } ( u ) = \{ p \in \Omega : u ( p ) \geq \lambda \} .\tag{4}
$$

Their connected components capture dark and bright structures at different intensity levels [1, 7, 19]. The Tree of Shapes combines both families into a self-dual representation.

Shapes and hierarchy. To obtain shapes, holes inside each connected component Γ are filled through the saturation operator

$$
\mathrm { S a t } ( \Gamma ) = \Omega \setminus \mathrm { C C } \left( \Omega \setminus \Gamma , p _ { \infty } \right) ,\tag{5}
$$

where $\operatorname { C C } ( A , p )$ denotes the connected component of A containing p. The resulting family of shapes is

$$
S ( u ) = S ^ { < } ( u ) \cup S ^ { \geq } ( u ) ,\tag{6}
$$

where $s ^ { < } ( u )$ and $\scriptstyle { \mathcal { S } } ^ { \geq } ( u )$ are obtained from the lower and upper level sets, respectively. Two shapes are either disjoint or nested, and the inclusion relation

$$
b _ { i } \preceq b _ { j } \quad \iff \quad b _ { i } \subseteq b _ { j }\tag{7}
$$

organizes them into the Tree of Shapes.

Morphological explanatory units. We use these structures as the explanatory units of MORPHOSHAP:

$$
B ( x ) = \{ b _ { 1 } , \ldots , b _ { M } \} , \qquad b _ { i } \in { \mathcal { S } } ( u ) ,\tag{8}
$$

excluding the root corresponding to the full image. Each shape is represented by a binary mask

$$
m _ { i } ( p ) = { \mathbf { 1 } } _ { \{ p \in b _ { i } \} } , \qquad m _ { i } \in \{ 0 , 1 \} ^ { H \times W } .\tag{9}
$$

Unlike a flat segmentation or regular patch grid, this representation preserves the hierarchy of image structures, allowing fine shapes to be nested inside larger ones.

## 3.3. Multi-Scale Morphological Characterization

The Tree of Shapes provides the morphological structures, which we further characterize by scale. For each shape $b _ { i }$ we compute its relative area $\begin{array} { r } { \rho _ { i } = \frac { | b _ { i } | } { H W } } \end{array}$ , and assign it to one of eight ordered scale levels $S _ { \mathrm { s c a l e } } = \{ S _ { 1 } , . . . , S _ { 8 } \}$ , from fine to coarse:

$$
s _ { i } = S _ { k } \quad \Longleftrightarrow \quad \tau _ { k - 1 } \leq \rho _ { i } < \tau _ { k } ,
$$

where $0 = \tau _ { 0 } < \cdot \cdot \cdot < \tau _ { 8 } = 1$ . The exact scale intervals are reported in Appendix A.1. The resulting representation is

$$
\mathcal { B } _ { \mathrm { s c a l e } } ( x ) = \{ ( b _ { i } , s _ { i } ) \} _ { i = 1 } ^ { M } .\tag{10}
$$

3.4. Geometric Characterization and Shape Naming

Geometric Shape Vocabulary. Scale alone cannot describe the geometry of a structure. We therefore associate each morphological shape with a discrete geometric label $g _ { i } \in \mathcal { G }$ , where

$$
\begin{array} { r } { \mathcal { G } = \{ \mathtt { E l o n g a t e d } , \mathtt { C i r c l e } , \mathtt { T r i a n g l e } , } \\ { \mathtt { R e c t a n g l e } , \mathtt { P o l y g o n } , \mathtt { C o m p l e x } \} . } \end{array}\tag{11}
$$

We investigate two mechanisms for assigning these names: (i) a deterministic rule-based geometric classifier, and (ii) a learned classifier. Both operate on the same geometric information and produce labels from the same vocabulary G.

![](images/e9479864bf164ad3a0f9a8de3794f4bcfd076b033bfee9b8adcf907647706246.jpg)  
Figure 2. Overview of the MorphoSHAP explanation pipeline. 1) Extraction: The input image is decomposed into hierarchical morphological regions using the Tree of Shapes. 2) Semantic Description: Each isolated region is mathematically evaluated and assigned a semantic tag (e.g., Triangular, Elongated, Complex). 3) Attribution: A region-based Shapley value estimator quantifies the precise contribution of each shape to the model’s prediction. 4) Explanation: The framework outputs rich, multi-modal explanations, including local spatial heatmaps, structured textual summaries, and global class-wise shape distributions.

Geometric descriptors. For every shape $b _ { i } .$ , we extract its external contour and compute a compact descriptor vector

$$
d _ { i } = [ A _ { i } , P _ { i } , \mathrm { A R } _ { i } , C _ { i } , Q _ { i } , n _ { i } ] ,\tag{12}
$$

where $A _ { i }$ is the area, $P _ { i }$ the perimeter, $\mathrm { A R } _ { i }$ the aspect ratio of its minimum-area bounding rectangle, $C _ { i }$ its circularity, $Q _ { i }$ its solidity, and $n _ { i }$ the number of vertices obtained from a polygonal approximation.

Rule-based geometric naming. Our first strategy is entirely deterministic. It applies a sequence of geometric rules to $d _ { i }$ . The order of the rules is intentional and makes the resulting categories mutually exclusive. See Appendix $\mathrm { A } . 2$ for more information on the Rule-based geometric naming.

We denote this deterministic mapping by

$$
g _ { i } = h _ { \mathrm { r u l e } } ( d _ { i } ) .\tag{13}
$$

Learned geometric naming. The hand-designed thresholds above provide a transparent and fully deterministic naming mechanism. However, fixed geometric rules may not always capture the interaction between different descriptors. We therefore also consider a data-driven variant. Let $h _ { \mathrm { D T } } ( \cdot ; \psi ) : \mathbb { R } ^ { D } \to \mathcal G$ denotes an MLP classifier parameterized by ψ. It receives an enhanced geometric descriptor vector $d _ { i }$ and predicts

$$
g _ { i } = h _ { \mathrm { D T } } ( d _ { i } ; \psi ) .\tag{14}
$$

The training data, hyperparameters, and classifier information used are reported in the Appendix $_ { \mathrm { A } . 3 , }$ . Appendix A.4 compares the two naming strategies. Both mechanisms lead to the same structured representation

$$
\mathcal { B } _ { \mathrm { m o r p h } } ( x ) = \{ ( b _ { i } , s _ { i } , g _ { i } ) \} _ { i = 1 } ^ { M } .\tag{15}
$$

## 3.5. Morphological Shape Contribution

The previous stages define what the explanatory units are. We now quantify how much each morphological shape contributes to the prediction. Each shape $b _ { i } \in B ( x )$ is treated

as a player in a cooperative game. Its scale $s _ { i }$ and geometric label $g _ { i }$ only describe the player; the model is perturbed through the spatial mask $m _ { i }$

Coalitions of morphological shapes. Let $S \subseteq B ( x )$ be a coalition of active shapes, with support

$$
m _ { S } = \bigvee _ { b _ { i } \in S } m _ { i } ,\tag{16}
$$

where $\vee$ denotes the pixel-wise logical OR. Because the Tree of Shapes is hierarchical, shapes may be nested, and $m _ { S }$ keeps every pixel belonging to at least one active shape.

Let

$$
m _ { B } = \bigvee _ { i = 1 } ^ { M } m _ { i } , \qquad m _ { \mathrm { b g } } = 1 - m _ { B } .\tag{17}
$$

Using a constant mid-gray reference image $x _ { 0 }$ to represent an absent shape, the image associated with coalition S is

$$
\begin{array} { c } { { x _ { S } = m _ { \mathrm { b g } } \odot x + m _ { S } \odot x } } \\ { { + \left( m _ { B } - m _ { S } \right) \odot x _ { 0 } , } } \end{array}\tag{18}
$$

where ⊙ denotes element-wise multiplication. Active shapes therefore keep their original pixels, inactive shapes are replaced by $x _ { 0 } .$ , and the residual background remains unchanged. The coalition value for class y is

$$
v _ { y } ( S ) = f _ { y } ( x _ { S } ) .\tag{19}
$$

Shapley contribution. The Shapley value of shape $b _ { i }$ is its average marginal contribution over all coalitions of the remaining shapes:

$$
\phi _ { i } = \sum _ { S \subseteq \mathcal { B } \setminus \{ b _ { i } \} } \frac { | S | ! ( M - | S | - 1 ) ! } { M ! } \left[ v _ { y } ( S \cup \{ b _ { i } \} ) - v _ { y } ( S ) \right] .\tag{20}
$$

Positive values support class $y ,$ while negative values oppose it. Since exact computation requires an exponential number of coalitions, we use KernelSHAP [32], which samples coalitions and fits a Shapley-weighted additive surrogate whose coefficients $\hat { \phi } _ { i }$ estimate the contribution of each morphological shape.

## 3.6. Local Morphological Explanations

For a single image, MORPHOSHAP produces a structured explanation

$$
\mathcal { E } _ { y } ( x ) = \Big [ ( b _ { i } , s _ { i } , g _ { i } , \hat { \phi } _ { i } ) \Big ] _ { i = 1 } ^ { M } ,\tag{21}
$$

which we rank according to $| \hat { \phi } _ { i } |$

This representation supports three complementary visualizations.

Shape-level explanation. Figure 3 illustrates how MOR-PHOSHAP summarizes a prediction through a SHAP-style explanation defined over morphological primitives rather than over raw pixels. Each bar in the waterfall plot corresponds to one extracted shape and is directly associated with three interpretable attributes: its scale level (e.g., S<sub>6</sub>), its geometric category (e.g., Elongated), and its signed contribution $\hat { \phi } _ { i } .$ For instance, the entry

$$
\begin{array} { r } { \textsf { S 6 } \mathrm { ~ -- ~ } \mathrm { ~ E 1 o n g a t e d } \mathrm { ~ -- ~ } \hat { \phi } = + 0 . 4 1 } \end{array}
$$

indicates that a relatively large elongated structure provides positive evidence for the considered prediction, whereas a negative contribution would indicate that the corresponding shape opposes it. This representation differs fundamentally from classical image-based explanations. Conventional saliency or attribution maps generally indicate where the model looks, but they do not explicitly describe what kind of visual structure supports the decision.

Morphological attribution map. MORPHOSHAP can also produce classical attribution map since the shape contributions can also be projected back into image space:

$$
H _ { y } ( p ) = \sum _ { i = 1 } ^ { M } \hat { \phi } _ { i } m _ { i } ( p ) .\tag{22}
$$

Because the Tree of Shapes is hierarchical, a pixel can belong to several nested structures. Equation (22) therefore accumulates evidence from the different morphological scales containing that pixel.

Textual explanation. Most importantly, the scale and geometric vocabulary allow the same attribution to be expressed in words without requiring an external language model. For the most influential positive and negative shapes, we generate templates of the form

![](images/a97573eb247403e349e1764778dde57216f753369a3ab48264382cefbac363ea.jpg)  
Figure 3. Shape-level explanation produced by MOR-PHOSHAP. In contrast to conventional image attribution methods that associate importance with pixels or image regions, each contribution in MORPHOSHAP corresponds to a morphological primitive characterized by both its geometric type and scale. Positive Shapley values indicate structures supporting the prediction, while negative values indicate opposing evidence. For example, the prediction is primarily supported by an $S _ { 6 }$ elongated structure and an S<sub>2</sub> circular structure, while an $S _ { 4 }$ rectangular structure provides negative evidence.

“The prediction is mainly supported by an $S _ { 6 }$ elongated structure and an $S _ { 2 }$ circular structure, while an $S _ { 4 }$ complex shape provides negative evidence.”

Thus, unlike a conventional attribution heatmap that only identifies relevant locations, MORPHOSHAP can describe the morphological nature of the evidence used by the model.

## 3.7. Global Morphological Explanations

A key consequence of using a fixed scale and geometric vocabulary is that explanations from different images can be compared and aggregated. Pixels from different images have no direct correspondence, and neither do arbitrary superpixels. In contrast, an $S _ { 2 }$ Circle or an $S _ { 6 }$ Elongated structure has the same interpretation across samples.

Let $\mathcal { D } _ { c }$ denote the set of images considered when analyzing class $c .$ For an image x, we first normalize its shape contributions as

$$
\widetilde { \phi } _ { i } ^ { ( x ) } = \frac { \hat { \phi } _ { i } ^ { ( x ) } } { \sum _ { j } \left| \hat { \phi } _ { j } ^ { ( x ) } \right| + \varepsilon } .\tag{23}
$$

For a geometric category g and scale $s ,$ we separately

aggregate positive and negative evidence:

$$
G _ { c } ^ { + } ( g , s ) = \frac { 1 } { \left| \mathcal { D } _ { c } \right| } \sum _ { x \in \mathcal { D } _ { c } } \sum _ { i = 1 } ^ { M _ { x } } \mathbb { 1 } \left[ g _ { i } = g , s _ { i } = s \right] \operatorname* { m a x } \left( \widetilde { \phi } _ { i } ^ { ( x ) } , 0 \right) ,\tag{24}
$$

$$
G _ { c } ^ { - } ( g , s ) = \frac { 1 } { \left| \mathcal { D } _ { c } \right| } \sum _ { x \in \mathcal { D } _ { c } } \sum _ { i = 1 } ^ { M _ { x } } \mathbb { 1 } \left[ g _ { i } = g , s _ { i } = s \right] \operatorname* { m a x } \left( - \widetilde { \phi } _ { i } ^ { ( x ) } , 0 \right)\tag{25}
$$

$G _ { c } ^ { + }$ therefore characterizes the morphological structures that systematically support class $c ,$ whereas $G _ { c } ^ { - }$ characterizes structures that systematically oppose it.

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We evaluate MorphoSHAP on five diverse benchmarks spanning natural, astronomical, satellite, biomedical and fine-grained recognition domains:

• ImageNet [13]: 1,000-class natural image classification. We evaluate on the validation set.

• Galaxy10 [28]: 10-class astronomical object classification (17,736 images). We evaluate on 5,000 stratified test images.

• EuroSAT [24]: 10-class satellite land-use classification (27,000) images. We evaluate on 5,000 stratified test images.

• Waterbirds [39]: 2-class bird classification with spurious background correlation (5,794 test images). We evaluate on 5,000 stratified test images.

• TissueMNIST [30] from MedMNIST [48]: 8-class histology tissue classification (47,280 test images). We evaluate on 5,000 stratified test images.

Models. We use three standard architectures: ResNet-50 [22], ViT-B/16 [14], and ConvNeXt-Tiny [29]. ImageNet models use ImageNet-1K pretrained weights; all other models are fine-tuned on their respective training splits. For Grad-CAM family methods, we target the final feature representations of each architecture: the last convolutional block in ResNet (layer4[-1]) and ConvNeXt (features[-1][-1]), and the final layer normalization block in ViT (encoder.layers[-1].ln 1).

Baselines. We compare against three categories of explanation methods:

• Gradient-based: Grad-CAM [40], Grad-CAM++ [10], Layer-CAM [25], Score-CAM [47], Shapley-CAM [5] and Integrated Gradients (IG) [45].

• Perturbation-based region methods: KernelSHAP [31] with SLIC superpixels (K = 50) and KernelSHAP with

Otsu binary thresholding, and PartitionSHAP [32] (axisaligned)

• Tree-based Shapley: ShapBPT [35] (binary partition tree) mode.

Evaluation Metrics. Following standard practice in explainability evaluation, we report:

• Insertion AUC (↑): Area under the curve of predictedclass probability as pixels are inserted from most to least important.

• Deletion AUC (↓): Area under the curve of predictedclass probability as pixels are deleted from most to least important.

• Runtime (seconds per image, ↓).

Further details are provided in Appendix B.

Hyperparameters and Ablations. MORPHOSHAP relies on specific hyperparameter configurations to control explanation granularity, which are detailed in Appendix C.2. To evaluate the effects of these structural parameters and our core algorithmic components on the final attributions, we conducted comprehensive ablation studies, the results of which are presented in Appendix C.3.

## 4.2. Quantitative Comparison

We report Insertion and Deletion AUC for all methods across the five datasets, averaged across three architectures (ResNet-50, ViT-B/16, and ConvNeXt-Tiny). We also report the average runtime on ImageNet.

Key findings. MorphoSHAP achieves the highest Insertion AUC on ImageNet, EuroSAT, Galaxy10, and TissueM-NIST, alongside the lowest Deletion AUC on ImageNet, EuroSAT, and Waterbirds. This indicates that our blob-level attributions consistently and accurately capture the regions the model relies on. The performance gap is notably pronounced on EuroSAT and Galaxy10, where objects possess strong macro-morphological structures (e.g., agricultural fields, galactic cores) that our method natively isolates. On datasets like Waterbirds, MorphoSHAP remains highly competitive with state-of-the-art tree-based methods while simultaneously providing the structured textual outputs that region methods fundamentally lack. In Appendix C.3 we can see that the results are stable to the change of the hyperparameters.

Comparison with region-based SHAP. KernelSHAP with SLIC superpixels performs poorly on structured domains because SLIC boundaries do not respect object contours, diluting attribution. KernelSHAP with Otsu suffers heavily from coarse binary segmentation. ShapBPT provides stronger baselines but still falls short of MorphoSHAP’s insertion scores on most datasets and fundamentally lacks the multi-scale shape vocabulary and geometric tagging of our method.

Table 1. Quantitative comparison of explanation methods across datasets, averaged across ResNet-50, ViT-B/16, and ConvNeXt-Tiny architectures. Bold indicates best performance. Lower Deletion AUC is better (↓); higher Insertion AUC is better (↑). Results are Mean ± Standard Deviation.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $\mathrm { I m a g e N e t }$ </td><td colspan="2"> $\mathrm { G a l a x y 1 0 }$ </td><td colspan="2"> $\mathrm { E u r o S A T }$ </td><td colspan="2">Waterbirds</td><td colspan="2">TissueMNIST</td></tr><tr><td>Del ↓</td><td>Ins ↑</td><td> $\mathrm { D e l } \downarrow$ </td><td>Ins ↑</td><td> $\mathrm { D e l } \downarrow$ </td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td></tr><tr><td>Grad-CAM</td><td>0.253 ±0.180</td><td>0.685 ±0.179</td><td> $0 . 2 5 8 \pm 0 . 2 9 2$ </td><td>0.555 ±0.302</td><td> $0 . 4 5 3 \pm 0 . 2 4 2$ </td><td> $0 . 5 5 5 \pm 0 . 2 2 8$ </td><td>0.420 ±0.289</td><td>0.852 ±0.184</td><td>0.174 ±0.192</td><td>0.460 ±0.239</td></tr><tr><td>Grad-CAM++</td><td>0.265 ±0.183</td><td> $0 . 6 7 0 \pm 0 . 1 8 4$ </td><td> $0 . 2 4 5 \pm 0 . 2 7 3$ </td><td> $0 . 5 5 2 \pm 0 . 2 8 5$ </td><td>0.458 ±0.237</td><td> $0 . 5 4 2 \pm 0 . 2 2 4$ </td><td>0.452 ±0.297</td><td> $0 . 8 3 0 \pm 0 . 1 9 1$ </td><td>0.191 ±0.211</td><td>0.452 ±0.239</td></tr><tr><td>Layer-CAM</td><td>0.259 ±0.181</td><td> $0 . 6 7 4 \pm 0 . 1 8 4$ </td><td> $0 . 1 7 0 \pm 0 . 1 9 8$ </td><td> $0 . 6 0 4 \pm 0 . 2 4 4$ </td><td> $0 . 4 5 9 \pm 0 . 2 3 6$ </td><td> $0 . 5 5 0 \pm 0 . 2 2 1$ </td><td> $0 . 4 5 1 \pm 0 . 2 9 6$ </td><td> $0 . 8 3 2 \pm 0 . 1 9 1$ </td><td>0.189 ±0.228</td><td>0.394 ±0.259</td></tr><tr><td>Score-CAM</td><td>0.292 ±0.197</td><td>0.656 ±0.202</td><td> $0 . 1 9 1 \pm 0 . 2 4 6$ </td><td>0.622 ±0.287</td><td> $0 . 4 1 6 \pm 0 . 2 3 3$ </td><td>0.562 ±0.214</td><td>0.444 ±0.295</td><td> $0 . 8 3 4 \pm 0 . 2 0 5$ </td><td>0.240 ±0.238</td><td>0.377 ±0.247</td></tr><tr><td>Shapley-CAM</td><td>0.253 ±0.180</td><td>0.685 ±0.179</td><td> $0 . 2 3 0 \pm 0 . 2 8 6$ </td><td>0.581 ±0.296</td><td> $0 . 4 3 4 \pm 0 . 2 4 6$ </td><td>0.574 ±0.219</td><td>0.420 ±0.289</td><td> $0 . 8 5 2 \pm 0 . 1 8 4$ </td><td>0.175 ±0.192</td><td>0.458 ±0.241</td></tr><tr><td>IG</td><td>0.144 ±0.182</td><td>0.435 ±0.295</td><td> $0 . 1 4 7 \pm 0 . 2 2 6$ </td><td>0.572 ±0.350</td><td> $0 . 1 9 7 \pm 0 . 2 2 7$ </td><td> $0 . 2 7 7 \pm 0 . 2 8 3$ </td><td>0.312 ±0.285</td><td> $0 . 8 1 8 \pm 0 . 2 7 9$ </td><td>0.170 ±0.273</td><td>0.215 ±0.290</td></tr><tr><td>KernelSHAP (SLIC)</td><td>0.223 ±0.163</td><td>0.736 ±0.173</td><td> $0 . 1 3 8 \pm 0 . 1 5 4$ </td><td>0.776 ±0.200</td><td> $0 . 3 6 4 \pm 0 . 2 1 9$ </td><td> $0 . 6 1 6 \pm 0 . 2 5 8$ </td><td>0.337 ±0.280</td><td> $0 . 8 9 0 \pm 0 . 2 0 8$ </td><td>0.157 ±0.218</td><td>0.445 ±0.292</td></tr><tr><td>KernelSHAP (Otsu)</td><td>0.406 ±0.175</td><td> $0 . 5 5 3 \pm 0 . 1 9 1$ </td><td> $0 . 2 8 0 \pm 0 . 2 5 9$ </td><td>0.656 ±0.278</td><td> $0 . 4 9 2 \pm 0 . 2 0 9$ </td><td> $0 . 5 0 7 \pm 0 . 2 1 1$ </td><td>0.591 ±0.241</td><td> $0 . 6 8 4 \pm 0 . 1 6 9$ </td><td>0.377 ±0.194</td><td>0.407 ±0.224</td></tr><tr><td>SHAP-BPT (AA)</td><td>0.280 ±0.217</td><td> $0 . 8 4 7 \pm 0 . 1 2 6$ </td><td> $0 . 0 8 3 \pm 0 . 1 0 9$ </td><td> $0 . 8 4 3 \pm 0 . 1 5 3$ </td><td> $0 . 3 8 4 \pm 0 . 1 9 7$ </td><td> $0 . 6 8 3 \pm 0 . 1 9 5$ </td><td>0.280 ±0.268</td><td> $0 . 8 8 9 \pm 0 . 2 1 4$ </td><td>0.143 ±0.190</td><td> $0 . 5 5 2 \pm 0 . 2 4 5$ </td></tr><tr><td>SHAP-BPT (BPT)</td><td>0.175 ±0.160</td><td> $0 . 7 9 2 \pm 0 . 1 5 1$ </td><td> $\mathbf { 0 . 0 7 4 \pm 0 . 1 1 7 }$ </td><td> $0 . 8 1 1 \pm 0 . 1 9 7$ </td><td> $0 . 3 3 0 \pm 0 . 2 1 5$ </td><td> $0 . 6 3 3 \pm 0 . 2 5 6$ </td><td>0.292 ±0.273</td><td> $\mathbf { 0 . 9 0 7 \pm 0 . 2 1 4 }$ </td><td>0.139 ±0.212</td><td> $0 . 4 5 2 \pm 0 . 2 9 4$ </td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.126 ±0.130</td><td> $\mathbf { 0 . 8 7 6 \pm 0 . 0 8 1 }$ </td><td> $0 . 1 1 6 \pm 0 . 0 8 6$ </td><td> $\mathbf { 0 . 8 7 6 \pm 0 . 0 8 4 }$ </td><td> $\mathbf { 0 . 1 1 6 \pm 0 . 1 1 9 }$ </td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 1 5 1 }$ </td><td> $\mathbf { 0 . 2 5 7 \pm 0 . 2 0 8 }$ </td><td> $0 . 8 8 9 \pm 0 . 1 8 5$ </td><td>0.144 ±0.187</td><td> $\mathbf { 0 . 8 2 0 \mathop { \pm } 0 . 1 0 7 }$ </td></tr></table>

Table 2. Average runtime per image (seconds) on ImageNet, aggregated across all three model architectures.
<table><tr><td>Method</td><td>Time (s) ↓</td></tr><tr><td>Grad-CAM Layer-CAM</td><td>0.02 0.02</td></tr><tr><td>Score-CAM Integrated Gradients (IG)</td><td>0.79 1.64</td></tr><tr><td>KernelSHAP (SLIC) SHAP-BPT (BPT)</td><td>0.82 0.29</td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.13</td></tr></table>

Computational cost. As shown in Table 2, MorphoSHAP is superior to existing perturbation methods in speed. Averaging 0.13 seconds per image on ImageNet, it is over 6× faster than SLIC-SHAP (0.82s), over 12× faster than Integrated Gradients (1.64s), and more than 2× faster than ShapBPT (0.29s). Because $M \ll H \cdot W$ , MorphoSHAP scales gracefully with image resolution while rivaling the speed of heavier CAM variants.

Detailed comparison of performance metrics are provided in Appendix C.1.

## 4.3. Qualitative and Semantic Explanations

Unlike pixel-level methods that output unstructured heatmaps, MorphoSHAP translates model behavior into a structured semantic vocabulary. By cross-referencing the blob registry B with the estimated Shapley values, we isolate the specific geometric primitives driving the prediction.

invariant, MorphoSHAP seamlessly adapts to vastly different object morphologies without requiring any domainspecific re-tuning. Unlike standard pixel-level attributions, MorphoSHAP isolates distinct geometric primitives—such as Circular clock faces, Rectangular street signs or agricultural parcels, and Complex galactic structures—and quantifies their exact contribution f(x) to the final prediction via native SHAP waterfall plots. This bridge between pristine visual saliency and structured textual summarization is achieved without relying on external vision-language models, ensuring the explanation remains strictly faithful to the underlying mathematical morphology. Details about the generation of the textual summary are provided in Appendix A.6. One can see in Figure 4 the two kinds of local explanations provided by MorphoSHAP. Appendix A.5 provides the qualitative results of the global explanation.

To demonstrate the versatility of our framework, Figure 4 visualizes this qualitative analysis across three distinct domains. Because the Tree of Shapes is scale- and contrast-

## 4.4. Cross-Domain Global Semantic Insights

Unlike pixel-attribution methods that are restricted to perimage heatmaps, MORPHOSHAP structurally tags each cooperative player prior to attribution, enabling the aggregation of local explanations into dataset-wide global insights. By auditing the primary positive morphological primitives across natural images (ImageNet), overhead satellite imagery (EuroSAT), and astronomical morphology (Galaxy10), we reveal clear, domain-specific geometric dependencies that align closely with physical ground truth.

For instance, MORPHOSHAP shows that predictions for industrial zones rely heavily on Complex primitives, while spiral galaxies depend on a dual representation of Rectangle central cores and Complex arms. Importantly, this vocabulary is modular: additional shape categories can be introduced depending on the structures that are relevant to a given application or domain. A comprehensive breakdown, including our full quantitative distribution table and further analysis of these global morphological dependencies, is provided in Appendix A.5.

![](images/12f67c4654a96cf016b0d1e056ac2978ac73f88478cedec94a3c125d89cda04d.jpg)  
Figure 4. Cross-domain qualitative explanations using MorphoSHAP on ImageNet, EuroSAT, and Galaxy10. The grid displays the original image, the MorphoSHAP attribution heatmap, and a corresponding waterfall plot that isolates specific geometric primitives (e.g., L3: Circle, L4: Rectangle) and quantifies their direct impact on the model’s final prediction.

## 4.5. User Study

While automated metrics like Insertion/Deletion AUC measure faithfulness to the model, they do not inherently measure human interpretability. To validate the practical utility of MorphoSHAP, we conducted a user study with 43 participants across 976 independent image evaluation trials using a randomly sampled subset of 150 images spanning 150 distinct ImageNet classes. To ensure non-expert participants could identify the objects, these highly specific classes were augmented with broad, recognizable category labels. The survey was structured into two core tasks: forward simulation (guess-the-class) and subjective preference.

Forward Simulation. Participants were shown an image masked by the explanation baseline and asked to identify the underlying object class. MorphoSHAP achieved a 89.7% human guessing accuracy, rivaling the dense pixelhighlighting of Grad-CAM (91.7%) and heavily outperforming hierarchical region baselines such as SHAP-BPT (85.6%) and PartitionSHAP (81.6%). Furthermore, MorphoSHAP imposed the lowest cognitive load on users, requiring a median decision time of only 6.78 seconds per image, the fastest across all evaluated methods.

Subjective Interpretability. When presented with sideby-side attributions and asked to select the most interpretable explanation for a given prediction, MorphoSHAP was overwhelmingly preferred. Participants selected MorphoSHAP as the best explanation in 60.1% of all trials, vastly exceeding SHAP-BPT (24.6%) and Grad-CAM (15.2%). These results demonstrate that extracting discrete, mathematically grounded geometric primitives aligns far better with human visual perception than raw pixel gradients or arbitrary superpixel segmentations.

Detailed information and results from the user study can be found in Appendix D.

## 5. Conclusion

In this work, we introduced MORPHOSHAP, a modelagnostic, post-hoc attribution framework that rethinks the fundamental unit of visual explanation. By leveraging the Tree of Shapes, MORPHOSHAP transitions the Shapley game from arbitrary pixel or superpixel grids to meaningful morphological primitives. This shift enables rich, multimodal explanations that simultaneously capture the spatial location, morphological scale, and geometric category of the visual evidence driving a model’s prediction. Our extensive evaluation across diverse domains - including natural, satellite, astronomical and medical imagery - demonstrates that MORPHOSHAP not only provides superior insertion and deletion performance compared to existing perturbation methods, but also aligns significantly better with human interpretability.

Limitations and Future Work. MORPHOSHAP may be less effective when predictions rely mainly on fine textures rather than well-defined shapes. In addition, the quality of the structured explanation depends on the tagging function used to describe each shape. Future work will therefore consider texture-aware decompositions and more advanced domain-specific or learned tagging strategies.

## References

[1] Coloma Ballester, Vicent Caselles, and Pascal Monasse. The tree of shapes of an image. ESAIM: Control, Optimisation and Calculus ofVariations, 9:1–18, 2003. 2, 3

[2] Emirhan Bilgic¸, Baptiste Caramiaux, Zhi Yan, and Gianni Franchi. Disentangling hallucinations: Orthogonal semantic projection for robust interpretability. In European Conference on Computer Vision, pages 1–18. Springer, 2026. 1

[3] Agustina Bouchet, Pedro Alonso, Juan Ignacio Pastore, Susana Montes, and Irene D´ıaz. Fuzzy mathematical morphology for color images defined by fuzzy preference relations. Pattern Recognition, 60:720–733, 2016. 2

[4] Walid Bousselham, Angie Boggust, Sofian Chaybouti, Hendrik Strobelt, and Hilde Kuehne. Legrad: An explainability method for vision transformers via feature formation sensitivity. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1–10. IEEE, 2025. 1

[5] Huaiguang Cai. Cams as shapley value-based explainers. arXiv preprint arXiv:2501.06261, 2025. 2, 6

[6] Edwin Carlinet and Thierry Geraud. Mtos: A tree of shapes´ for multivariate images. IEEE Transactions on Image Processing, 24(12):5330–5342, 2015. 3

[7] Edwin Carlinet, Sebastien Crozet, and Thierry G ´ eraud. The ´ tree of shapes turned into a max-tree: a simple and efficient linear algorithm. In 2018 25th IEEE International Conference on Image Processing (ICIP), pages 1488–1492. IEEE, 2018. 2, 3

[8] Gabriele Cavallaro. Spectral-Spatial Classification of Remote Sensing Optical Data with Morphological Attribute Profiles using Parallel and Scalable Methods. PhD thesis, University of Iceland, 2016. 2

[9] Gabriele Cavallaro, Nicola Falco, Mauro Dalla Mura, and Jon Atli Benediktsson. Automatic attribute profiles.´ IEEE Transactions on Image Processing, 26(4):1859–1872, 2017. 2

[10] Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader, and Vineeth N Balasubramanian. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE winter conference on applications of computer vision (WACV), pages 839–847. IEEE, 2018. 2, 6

[11] Hila Chefer, Shir Gur, and Lior Wolf. Generic attentionmodel explainability for interpreting bi-modal and encoderdecoder transformers. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 387–396. IEEE, 2021. 1

[12] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 782–791. IEEE, 2021. 1

[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6, 17

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,

Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl vain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6

[15] Gianni Franchi and Jesus Angulo. Comparative study on morphological principal component analysis of hyperspectral images. In 2014 6th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHIS-PERS), pages 1–4. IEEE, 2014. 2

[16] Gianni Franchi and Jesus Angulo. Morphological principal component analysis for hyperspectral image analysis. ISPRS International Journal ofGeo-Information, 5(6):83, 2016.

[17] Gianni Franchi, Jesus Angulo, Maxime Moreaud, and Lo¨ıc Sorbier. Enhanced edx images by fusion of multimodal sem images using pansharpening techniques. Journal ofmi croscopy, 269(1):94–112, 2018. 2

[18] Gianni Franchi, Amin Fehri, and Angela Yao. Deep morpho logical networks. Pattern Recognition, 102:107246, 2020. 2

[19] Thierry Geraud, Nicolas Boutry, S ´ ebastien Crozet, Edwin´ Carlinet, and Laurent Najman. A proof of the tree of shapes in nd. arXiv preprint arXiv:2206.05109, 2022. 2, 3

[20] Vahidin Hasic, Amar Halilovi ´ c, and Senka Krivi ´ c. Super-´ pixel correlation for explainable image classification. In World Conference on Explainable Artificial Intelligence, pages 27–44. Springer, 2025. 2

[21] Vahidin Hasic, Amar Halilovi ´ c, and Senka Krivi ´ c. Aa-shap:´ Superpixel affinity for explainable image classification. Ex pert Systems, 43(9):e70376, 2026. 2

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[23] Yizhou He, Jia Zheng, and Erbo Zou. Sharpen-cam: efficient hierarchical shap-based visual explanation for deep convolu tional neural networks. Multimedia Systems, 32(1):3, 2026. 2

[24] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019. 6

[25] Peng-Tao Jiang, Chang-Bin Zhang, Qibin Hou, Ming-Ming Cheng, and Yunchao Wei. Layercam: Exploring hierarchical class activation maps for localization. IEEE transactions on image processing, 30:5875–5888, 2021. 2, 6

[26] Remi Kazmierczak, Steve Azzolin, Elo´ ¨ıse Berthier, Goran Frehse, and Gianni Franchi. Enhancing concept localization in clip-based concept bottleneck models. arXiv preprint arXiv:2510.07115, 2025. 1

[27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In 2023 IEEE/CVF international conference on computer vision (ICCV), pages 3992–4003. IEEE, 2023. 2

[28] W. Henry Leung and Jo Bovy. Galaxy10 decals, 2024. 6

[29] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feicht enhofer, Trevor Darrell, and Saining Xie. A convnet for the

2020s. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 11966–11976. IEEE, 2022. 6

[30] Vebjorn Ljosa, Katherine L Sokolnicki, and Anne E Carpenter. Annotated high-throughput microscopy image sets for validation. Nature methods, 9(7):637–637, 2012. 6

[31] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Curran Associates, Inc., 2017. 6

[32] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. NeurIPS, 30, 2017. 1, 2, 5, 6

[33] Jonathan Masci, Jesus Angulo, and J´ urgen Schmidhuber.¨ A learning framework for morphological operators using counter–harmonic mean. In International Symposium on Mathematical Morphology and Its Applications to Signal and Image Processing, pages 329–340. Springer, 2013. 2

[34] Ranjan Mondal, Sanchayan Santra, and Bhabatosh Chanda. Dense morphological network: An universal function approximator. arXiv preprint arXiv:1901.00109, 2019. 2

[35] Muhammad Rashid, Elvio G Amparore, Enrico Ferrari, and Damiano Verda. ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 25099– 25107, 2026. 2, 6

[36] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ” why should i trust you?” explaining the predictions of any classifier. In Proceedings ofthe 22ndACM SIGKDD interna tional conference on knowledge discovery and data mining, pages 1135–1144, 2016. 1

[37] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal ofComputer Vision (IJCV), 115 (3):211–252, 2015. 17

[38] Faraz Saeedan, Nicolas Weber, Michael Goesele, and Stefan Roth. Detail-preserving pooling in deep networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9108–9116, 2018. 2

[39] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worstcase generalization. arXiv preprint arXiv:1911.08731, 2019. 6

[40] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, pages 618–626, 2017. 1, 2, 6

[41] Jean Serra. Image Analysis and Mathematical Morphology. Academic Press, Inc., Orlando, FL, USA, 1983. 2

[42] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013. 1

[43] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, (0):–, 2012. 16

[44] Ao Sun, Pingchuan Ma, Yuanyuan Yuan, and Shuai Wang. Explain any concept: Segment anything meets conceptbased explanation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2

[45] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In ICML, pages 3319–3328. PMLR, 2017. 1, 6

[46] Jacopo Teneggi, Alexandre Luster, and Jeremias Sulam. Fast hierarchical games for image explanations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4): 4494–4503, 2022. 2

[47] Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, and Xia Hu. Score-cam: Score-weighted visual explanations for convolutional neural networks. In 2020 IEEE/CVF conference on computer vision and pattern recognition workshops (CVPRW), pages 111– 119. IEEE, 2020. 2, 6

[48] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41, 2023. 6

# MorphoSHAP: Rethinking the Unit of Attribution in Explanation for Deep Visual Models

– Supplementary Material –

## A. Complementary information on MOR-PHOSHAP

## A.1. Multi-Scale Morphological Characterization

The Tree of Shapes provides a hierarchy of morphological structures with different spatial extents. To obtain a simple and shared vocabulary for their size, we assign each extracted shape $b _ { i }$ to one of eight scale levels.

For a shape $b _ { i }$ with binary mask $m _ { i }$ , we define its area as

$$
\left| b _ { i } \right| = \sum _ { p \in \Omega } m _ { i } ( p ) ,\tag{26}
$$

and its relative area with respect to the full image as

$$
\rho _ { i } = \frac { | b _ { i } | } { H W } , \qquad \rho _ { i } \in [ 0 , 1 ] ,\tag{27}
$$

where H and W denote the image height and width, respectively. Using a relative area rather than the raw number of pixels makes the scale definition independent of image resolution.

We consider the ordered scale vocabulary

$$
S _ { \mathrm { s c a l e } } = \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { 8 } \} ,\tag{28}
$$

where $S _ { 1 }$ corresponds to the finest structures and $S _ { 8 }$ to the coarsest ones. The scale boundaries are

$$
\tau = [ 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 4 0 , 0 . 5 0 , 1 . 0 0 ] _ { \cdot }\tag{29}
$$

For $k < 8 ,$ , a shape is assigned to scale $S _ { k }$ according to

$$
s _ { i } = S _ { k } \quad \Longleftrightarrow \quad \tau _ { k - 1 } \leq \rho _ { i } < \tau _ { k } ,\tag{30}
$$

while the last interval includes its upper boundary, $\rho _ { i } ~ \in$ [0.50, 1].

Table S3 gives the complete scale definition.

Importantly, these scale levels do not modify the Treeof-Shapes decomposition and are not used to generate or filter shapes. They are assigned after shape extraction and serve only as an interpretable attribute of each morphological primitive. The scale-enriched representation is therefore

$$
\mathcal { B } _ { \mathrm { s c a l e } } ( x ) = \{ ( b _ { i } , s _ { i } ) \} _ { i = 1 } ^ { M } .\tag{31}
$$

This discretization provides a common size vocabulary across images and allows MORPHOSHAP to describe, for example, whether a prediction relies on fine structures such as an $S _ { 1 }$ or $S _ { 2 }$ shape, or on larger structures such as $S _ { 6 } – S _ { 8 }$

Table S3. Definition of the eight morphological scale levels. The relative area $\rho _ { i } = | b _ { i } | / ( H W )$ measures the fraction of the image occupied by shape b<sub>i</sub>.
<table><tr><td>Scale</td><td>Relative area  $\rho _ { i }$ </td><td>Image area</td></tr><tr><td> $S _ { 1 }$ </td><td> $0 \leq \rho _ { i } < 0 . 0 1$ </td><td>&lt; 1%</td></tr><tr><td> $S _ { 2 }$ </td><td> $0 . 0 1 \leq \rho _ { i } < 0 . 0 5$ </td><td>1-5%</td></tr><tr><td> $S _ { 3 }$ </td><td> $0 . 0 5 \le \rho _ { i } < 0 . 1 0$ </td><td>5-10%</td></tr><tr><td> $S _ { 4 }$ </td><td> $0 . 1 0 \leq \rho _ { i } < 0 . 2 0$ </td><td>10-20%</td></tr><tr><td> $S _ { 5 }$ </td><td> $0 . 2 0 \leq \rho _ { i } < 0 . 3 0$ </td><td>20-30%</td></tr><tr><td> $S _ { 6 }$ </td><td> $0 . 3 0 \leq \rho _ { i } < 0 . 4 0$ </td><td>30-40%</td></tr><tr><td> $S _ { 7 }$ </td><td> $0 . 4 0 \leq \rho _ { i } < 0 . 5 0$ </td><td>40-50%</td></tr><tr><td> $S _ { 8 }$ </td><td> $0 . 5 0 \leq \rho _ { i } \leq 1$ </td><td>≥ 50%</td></tr></table>

## A.2. Rule-based geometric naming

One of the strategies to attribute the label to the shape is entirely deterministic. It applies a sequence of geometric rules to $d _ { i }$ . The order of the rules is intentional and makes the resulting categories mutually exclusive.

## 1. Elongated. If

$$
\mathrm { A R } _ { i } > 2 . 5 ,\tag{32}
$$

the shape is labeled Elongated. This rule is evaluated first so that strongly anisotropic structures are described primarily by their elongation, independently of their precise polygonal contour.

2. Circle. For the remaining shapes, if

$$
C _ { i } > 0 . 8 2 ,\tag{33}
$$

the shape is labeled Circle. The circularity measure reaches 1 for an ideal circle and decreases as the contour departs from circularity.

3. Triangle, Rectangle, and Polygon. For the remaining shapes, we approximate the contour using the Douglas– Peucker algorithm with tolerance

$$
\epsilon _ { i } = 0 . 0 4 P _ { i } .\tag{34}
$$

Let $n _ { i }$ denote the number of vertices of the simplified contour. We assign

$$
g _ { i } = \left\{ \begin{array} { l l } { \mathrm { T r i a n g l e , } } & { n _ { i } = 3 , } \\ { } & { } \\ { \mathsf { R e c t a n g l e , } } & { n _ { i } = 4 , } \\ { } & { } \\ { \mathsf { P o l y g o n , } } & { n _ { i } \in \{ 5 , 6 \} . } \end{array} \right.\tag{35}
$$

4. Complex shape. Every structure that does not satisfy the previous criteria is labeled Complex. This category captures morphological structures whose geometry cannot be faithfully summarized by one of the elementary geometric primitives in our vocabulary.

We denote this deterministic mapping by

$$
g _ { i } = h _ { \mathrm { r u l e } } ( d _ { i } ) .\tag{36}
$$

## A.3. Learned Geometric Naming (MLP Variant)

While the rule-based approach operates on the compact 6- dimensional descriptor $d _ { i }$ , the learned approach requires a richer representation to capture complex structural nuances. For the MLP variant, we extract an expanded 24- dimensional geometric feature vector for each shape contour.

This expanded feature set includes:

• Normalized Basic Metrics: Area and perimeter normalized by the total image area and dimensions.

• Bounding Geometries: Ratios of the contour area to its bounding box, minimum enclosing circle, convex hull, and minimum area rectangle.

• Polygonal and Convexity Features: Vertex counts at multiple ϵ-tolerances (0.02, 0.05, 0.10) using the Douglas-Peucker algorithm, alongside normalized convexity defect counts and depths.

• Invariant Moments: The seven Hu Moments (logtransformed) to capture rotation-, scale-, and translationinvariant shape characteristics.

• Spatial Distribution: The standard deviation and maximum of the distances from the contour points to the shape’s geometric centroid.

Our classifier, ShapeMLP, is parameterized as a Multi-Layer Perceptron (MLP) consisting of three hidden layers with dimensions (256, 128, 64). To ensure stable training across disparate geometric scales, we apply 1D Batch Normalization directly to the input features and after each hidden linear layer. The network utilizes ReLU activations and a dropout rate of $p = 0 . 3$ to prevent overfitting.

Dataset Generation and Training. To train the classifier without requiring exhaustive human annotation, we programmatically generated a diverse synthetic dataset of isolated morphological primitives. Using OpenCV, we rendered over 5,000 shapes with highly randomized geometric parameters - varying scales, aspect ratios, rotation angles, and contour noise - across randomized contrast intensities to accurately emulate the varied output of the Tree of Shapes. The dataset was split into training, validation, and holdout test sets. The network was trained using the Adam optimizer and Cross-Entropy Loss, relying on 1D Batch Normalization and dropout $( p = 0 . 3 )$ to ensure stable convergence and prevent overfitting. Evaluated on a 500-sample holdout test set, the trained MLP achieved near-perfect accuracy $( \approx ~ 9 9 \% )$ , demonstrating that the expanded MLPbased descriptor provides a highly separable feature space for identifying our five target morphological categories.

## A.4. Comparisons of the Two Namings

The rule-based mapping $\left( h _ { \mathrm { r u l e } } \right)$ and the learned classifier (ShapeMLP) offer distinctly complementary advantages for morphological attribution.

Rule-Based $( h _ { \mathrm { r u l e } } ) { : }$ The primary advantage of the rulebased heuristic is absolute transparency and zero-shot generalization. Because the thresholds rely on explicit, mathematically bounded properties (e.g., circularity and solidity), it seamlessly adapts to vastly different domains—from macroscopic satellite structures to microscopic histology— without requiring any domain-specific training data. Furthermore, it operates strictly on the compact descriptor $d _ { i } ,$ making it computationally inexpensive.

Learned Variant (ShapeMLP): Conversely, the MLP excels at resolving ambiguous edge cases by leveraging the richer 24-dimensional feature space. For instance, realworld object contours frequently suffer from camera perspective skew, discretization noise, and partial occlusion, which easily trip up strict rule-based polygonal approximations. Instead of depending on basic aspect ratios, the MLP navigates this noise by combining more complex shape features like Hu moments and convexity defects. However, this method requires a labeled morphological dataset for training and inherently binds the explanation quality to the distribution of that training data.

For the core experiments in the main text, we utilized the Learned Variant (MLP) approach to ensure maximum transparency, domain-independence, and reproducibility across the five diverse benchmark datasets. In Table S14 we compare the performance of the two naming strategies on a real dataset.

## A.5. Cross-Domain Global Semantic Insights

Unlike pixel-attribution methods that are restricted to perimage heatmaps, MorphoSHAP structurally tags each cooperative player prior to attribution, enabling the aggregation of local explanations into dataset-wide global insights. To demonstrate the cross-domain utility of MorphoSHAP, we audited the primary positive morphological primitives across three distinct vision domains: natural images (ImageNet), overhead satellite imagery (EuroSAT), and astronomical morphology (Galaxy10).

As detailed in Table S4, aggregating MorphoSHAP primitives reveals clear, domain-specific geometric dependencies that align closely with physical ground truth:

• Natural Images (ImageNet): Man-made object classes are heavily driven by canonical geometric bounds. ResNet-50 relies predominantly on Rectangle primitives when identifying Street Signs (46.2%) and Wall Clocks (34.3%).

• Satellite Remote Sensing (EuroSAT): Overhead imagery features complex, multi-facility structures. MorphoSHAP reveals that predictions for Industrial Zones are overwhelmingly driven by Complex Polygon primitives (64.0%), accurately reflecting the non-convex, sprawling layouts of warehouses and shipping docks.

• Astronomical Imaging (Galaxy10): MorphoSHAP successfully disentangles galactic structures. While Disk Edge-On galaxies rely heavily on rectilinear central profiles (76.0% Rectangle), Spiral Galaxies exhibit a strong dual reliance on central cores (44.0% Rectangle) and sprawling spiral arms (36.0% Complex Polygon).

This cross-domain auditing capability demonstrates that MorphoSHAP captures authentic, domain-appropriate structural representations, revealing the underlying geometric priors learned by the neural network. Furthermore, while the insights presented in Table S4 were generated using our lightweight, rule-based geometric heuristic, the MorphoSHAP framework is inherently extensible. The vocabulary of the shapes exposes a modular tagging interface allowing practitioners to seamlessly integrate custom tagging functions. By defining any function that maps an isolated binary morphological blob to a semantic string, researchers can deploy sophisticated domain-specific taggers—including specialized deep neural networks or pretrained vision-language models—to extract arbitrarily complex semantic explanations tailored to their specific use case.

<table><tr><td>Dataset</td><td>Class</td><td>Triangle</td><td>Circle</td><td>Rectangle</td><td>Elongated</td><td>Complex</td></tr><tr><td rowspan="4">ImageNet</td><td>Wall Clock</td><td>0.0%</td><td>32.0%</td><td>26.0%</td><td>22.0%</td><td>20.0%</td></tr><tr><td>Monitor</td><td>0.0%</td><td>30.0%</td><td>34.0%</td><td>14.0%</td><td>22.0%</td></tr><tr><td>Street Sign</td><td>2.0%</td><td>20.0%</td><td>10.0%</td><td>32.0%</td><td>36.0%</td></tr><tr><td>Traffic Light</td><td>0.0%</td><td>18.4%</td><td>10.2%</td><td>40.8%</td><td>30.6%</td></tr><tr><td rowspan="6">EuroSAT</td><td>AnnualCrop</td><td>4.0%</td><td>16.0%</td><td>10.0%</td><td>40.0%</td><td>30.0%</td></tr><tr><td>Highway</td><td>0.0%</td><td>0.0%</td><td>6.0%</td><td>24.0%</td><td>70.0%</td></tr><tr><td>Industrial</td><td>0.0%</td><td>10.0%</td><td>14.0%</td><td>8.0%</td><td>68.0%</td></tr><tr><td>Residential</td><td>0.0%</td><td>0.0%</td><td>8.0%</td><td>12.0%</td><td>80.0%</td></tr><tr><td>River</td><td>4.0%</td><td>6.0%</td><td>20.0%</td><td>50.0%</td><td>20.0%</td></tr><tr><td>SeaLake</td><td>0.0%</td><td>8.1%</td><td>18.9%</td><td>16.2%</td><td>56.8%</td></tr><tr><td rowspan="5">Galaxy10</td><td>Merging Galaxies</td><td>0.0%</td><td>48.0%</td><td>16.0%</td><td>18.0%</td><td>18.0%</td></tr><tr><td>Round Smooth Galaxies</td><td>0.0%</td><td>64.0%</td><td>24.0%</td><td>2.0%</td><td>10.0%</td></tr><tr><td>In-between Round Smooth Galaxies</td><td>0.0%</td><td>22.0%</td><td>36.0%</td><td>12.0%</td><td>30.0%</td></tr><tr><td>Barred Spiral Galaxies</td><td>0.0%</td><td>66.0%</td><td>16.0%</td><td>6.0%</td><td>12.0%</td></tr><tr><td>Unbarred Tight Spiral Galaxies</td><td>0.0%</td><td>70.0%</td><td>14.0%</td><td>10.0%</td><td>6.0%</td></tr></table>

Table S4. Cross-Domain Global Semantic Aggregation. Distribution of the highest-attributed MorphoSHAP primitives across natural, satellite, and astronomical datasets. The data aggregates shape counts across all 8 scale layers.

## A.6. Textual explanation

MORPHOSHAP translates spatial attributions into structured textual summaries. This is achieved deterministically through a rule-based natural language generation pipeline rather than relying on external Vision-Language Models (VLMs) or Large Language Models (LLMs). This guarantees that the generated explanation remains strictly faithful to the underlying Shapley values and avoids linguistic hallucinations.

Generation Pipeline. Given the blob registry B and estimated Shapley values $\hat { \phi } ,$ the text generation module operates in five sequential steps:

1. Aggregation and Area Tracking: We ignore background blobs (where $g _ { i } = \mathrm { b a c k g r o u n d } )$ . For each active morphological shape category $g \in { \mathcal { G } }$ , we sum the total Shapley contributions $\begin{array} { r } { \Phi ( g ) = \sum _ { b _ { i } \in g } \hat { \phi } _ { i } } \end{array}$ . Simultaneously, we track the maximum area ratio $\rho _ { \mathrm { m a x } } ( g ) =$ $\operatorname* { m a x } _ { b _ { i } \in g } { \frac { | b _ { i } | } { H W } }$ occupied by the largest blob of that shape category.

2. Filtering and Ranking: We filter for shape categories that actively support the target class prediction $( \Phi ( g ) >$ 0). The surviving categories are sorted in descending order by their aggregated Shapley contribution, and we select up to the top 3 dominant shape contributors.

3. Size Adjective Mapping: To describe spatial scale, the maximum relative area ratio $\rho _ { \mathrm { m a x } } ( g )$ of each top contributor is mapped to a discrete size adjective:

$$
\begin{array} { r } { \mathrm { S i z e } ( g ) = \left\{ \begin{array} { l l } { \mathrm { l a r g e , ~ } } & { \mathrm { i f } \rho _ { \mathrm { m a x } } ( g ) > 0 . 1 5 , } \\ { \mathrm { m e d i u m , } } & { \mathrm { i f } 0 . 0 5 < \rho _ { \mathrm { m a x } } ( g ) \leq 0 . 1 5 , } \\ { \mathrm { s m a l l , } } & { \mathrm { i f } \rho _ { \mathrm { m a x } } ( g ) \leq 0 . 0 5 . } \end{array} \right. } \end{array}\tag{37}
$$

4. Geometry Adjective Conversion: Discrete geometric labels are converted into descriptive adjectives (e.g., Circle → circular, Rectangle → rectangular, Triangle → triangular, Elongated → elongated, Complex → complex). Each contributor is formatted into a noun phrase of the form: “[size] [geometry] structure”.

5. Grammatical Assembly: The extracted phrases are injected into deterministic template strings based on the number of top positive contributors $( K \in \{ 0 , 1 , 2 , 3 \} )$ :

• K = 0: The prediction is primarily supported by the background context.

• K = 1: The prediction is mainly supported by a [Phrase<sub>1</sub>].

• K = 2: The prediction is mainly supported by a [Phrase ] and a [Phrase ].

• K = 3: The prediction is mainly supported by a [Phrase<sub>1</sub>], a [Phrase<sub>2</sub>], and a [Phrase<sub>3</sub>].

For example, if the top three positive shape categories for a prediction are a large complex blob $( \Phi = 0 . 4 2 , \rho _ { \mathrm { m a x } } =$ 0.22), a medium rectangular blob $( \Phi = 0 . 1 8 , \rho _ { \mathrm { m a x } } = 0 . 0 8 )$ and a small circular blob $( \Phi = 0 . 0 9 , \rho _ { \mathrm { m a x } } = 0 . 0 2 )$ , the pipeline outputs:

## B. Details on the Evaluation Metrics

We evaluate explanation faithfulness using the standard $I n \_$ sertion and Deletion metrics, together with the runtime required to generate an explanation.

Insertion AUC. Insertion measures whether the regions identified as important are sufficient to recover the model prediction. Starting from a reference image $x _ { 0 } ,$ , explanatory units are progressively restored from the most to the least important. Let $\pi = ( \pi _ { 1 } , \ldots , \pi _ { M } )$ denote the ranking of the M explanatory units. After inserting the first k units, we evaluate the predicted-class score

$$
I ( k ) = f _ { y } \big ( x _ { k } ^ { \mathrm { i n s } } \big ) , \qquad k = 0 , \ldots , M .\tag{38}
$$

The Insertion AUC is the area under the curve $f _ { y } ( x _ { k } ^ { \mathrm { i n s } } )$ as increasingly more evidence is added. A higher AUC indicates that the most important units identified by the explanation rapidly recover the model prediction.

Deletion AUC. Deletion follows the opposite procedure and measures whether removing the most important regions rapidly decreases the prediction. Starting from the original image x, explanatory units are removed according to the same importance ranking:

$$
D ( k ) = f _ { y } \left( x _ { k } ^ { \mathrm { d e l } } \right) , \qquad k = 0 , \ldots , M .\tag{39}
$$

A faithful explanation should identify regions whose removal strongly affects the prediction; therefore, a lower Deletion AUC is better.

MorphoSHAP evaluation. For standard pixel- or regionbased methods, insertion and deletion operate on their corresponding explanatory units. For MORPHOSHAP, the units are the morphological shapes $B ( x ) = \{ b _ { 1 } , \ldots , b _ { M } \}$ rather than individual pixels. The curves are therefore constructed by progressively inserting or deleting morphological blobs according to their attribution ranking. When a shape is absent, its pixels are replaced by the same reference value $x _ { 0 }$ used in the Shapley perturbation process.

This shape-level evaluation is consistent with the main goal of MORPHOSHAP: faithfulness is measured in the same morphological space in which the explanation is defined.

Runtime. We additionally report the average runtime required to produce an explanation, in seconds per image. Lower runtime indicates a more computationally efficient explanation method.

Sampling budget. For a fair comparison between perturbation-based approaches, we use the same KernelSHAP sampling budget of

$$
N = 1 0 2 4\tag{40}
$$

coalitions for all such methods.

## C. Experiments

## C.1. Detailed Quantitative Results per Architecture

In the main text (Table 1), we report the Insertion and Deletion AUC metrics averaged across all three evaluated architectures (ResNet-50, ViT-B/16, and ConvNeXt-Tiny) to provide a holistic view of explanation faithfulness.

In this section, we provide the detailed, per-architecture breakdowns. Tables S5, S6, and S7 report the individual performance on ResNet-50, ViT-B/16, and ConvNeXt-Tiny, respectively. These unaggregated results demonstrate that MORPHOSHAP maintains consistent faithfulness across structurally diverse model families, from standard convolutions to patch-based vision transformers.

Computational Cost per Architecture. In addition to the insertion and deletion metrics, we provide a detailed breakdown of the computational cost across different model architectures. Table S8 reports the average runtime (in seconds per image) on the ImageNet validation set for ResNet-50, ViT-B/16, and ConvNeXt-Tiny. MORPHOSHAP consistently remains significantly faster than both standard perturbation approaches (KernelSHAP) and tree-based Shapley estimators (SHAP-BPT) across all structural families, scaling efficiently even on computationally heavy Vision Transformers.

## C.2. Algorithm details

The performance and semantic granularity of the MOR-PHOSHAP framework are primarily governed by four core hyperparameters:

• Scale Thresholds (threshold percents): This dictates the multi-scale granularity of the Tree of Shapes decomposition. As defined in Appendix A.1, we utilize eight relative area thresholds $( S _ { 1 }$ through $S _ { 8 } )$ to extract shapes ranging from microscopic (0-1% of image area) to global (>50% of image area). Changing these boundaries directly alters the size and number of the morphological structures evaluated by the model.

• Geometric Tagging Function (tagging fn): This dictates how visual semantics are assigned to the extracted blobs. The framework supports switching between a deterministic, Rule-Based heuristic (relying on strict geometric bounds like circularity and aspect ratio) and a Learned MLP (which evaluates a 24-dimension feature vector).

Table S5. Quantitative comparison of explanation methods on ResNet-50. Lower Deletion AUC is better (↓); higher Insertion AUC is better (↑). Bold indicates best performance.
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet</td><td colspan="2">Galaxy10</td><td colspan="2">EuroSAT</td><td colspan="2">Waterbirds</td><td colspan="2">TissueMNIST</td></tr><tr><td> $\mathrm { D e l } \downarrow$ </td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td></tr><tr><td>Grad-CAM</td><td>0.260 ±0.184</td><td>0.705 ±0.173</td><td> $0 . 1 4 7 \pm 0 . 1 8 3$ </td><td>0.668 ±0.190</td><td> $0 . 3 5 4 \pm 0 . 2 6 7$ </td><td> $0 . 5 5 2 \pm 0 . 2 6 5$ </td><td>0.412 ±0.277</td><td> $0 . 8 7 6 \pm 0 . 1 5 8$ </td><td>0.173 ±0.241</td><td>0.371 ±0.280</td></tr><tr><td>Grad-CAM++</td><td>0.271 ±0.187</td><td>0.690 ±0.180</td><td> $0 . 1 4 9 \pm 0 . 1 8 4$ </td><td>0.664 ±0.190</td><td> $0 . 3 5 5 \pm 0 . 2 7 0$ </td><td> $0 . 5 4 9 \pm 0 . 2 6 7$ </td><td>0.428 ±0.279</td><td> $0 . 8 6 2 \pm 0 . 1 5 4$ </td><td>0.194 ±0.272</td><td>0.349 ±0.281</td></tr><tr><td>Layer-CAM</td><td>0.267 ±0.186</td><td>0.695 ±0.179</td><td>0.150 ±0.185</td><td>0.662 ±0.192</td><td>0.358 ±0.269</td><td> $0 . 5 4 6 \pm 0 . 2 6 8$ </td><td> $0 . 4 3 2 \pm 0 . 2 7 9$ </td><td> $0 . 8 6 1 \pm 0 . 1 5 3$ </td><td>0.195 ±0.274</td><td>0.341 ±0.283</td></tr><tr><td>Score-CAM</td><td>0.283 ±0.194</td><td>0.694 ±0.182</td><td> $0 . 1 4 8 \pm 0 . 1 8 9$ </td><td>0.662 ±0.198</td><td> $0 . 3 5 7 \pm 0 . 2 6 5$ </td><td> $0 . 5 4 9 \pm 0 . 2 7 6$ </td><td> $0 . 4 2 5 \pm 0 . 2 7 5$ </td><td> $0 . 8 6 4 \pm 0 . 1 5 1$ </td><td>0.206 ±0.277</td><td>0.314 ±0.285</td></tr><tr><td>Shapley-CAM</td><td>0.260 ±0.184</td><td>0.705 ±0.173</td><td> $0 . 1 4 7 \pm 0 . 1 8 3$ </td><td>0.668 ±0.190</td><td> $0 . 3 5 4 \pm 0 . 2 6 7$ </td><td> $0 . 5 5 2 \pm 0 . 2 6 5$ </td><td>0.412 ±0.277</td><td> $0 . 8 7 6 \pm 0 . 1 5 8$ </td><td>0.173 ±0.241</td><td>0.371 ±0.280</td></tr><tr><td>IG</td><td>0.142 ±0.183</td><td>0.397 ±0.287</td><td> $0 . 1 9 4 \pm 0 . 2 6 7$ </td><td>0.639 ±0.353</td><td> $0 . 1 3 8 \pm 0 . 2 4 3$ </td><td> $0 . 1 3 8 \pm 0 . 2 4 3$ </td><td>0.260 ±0.252</td><td> $0 . 6 4 0 \pm 0 . 2 7 6$ </td><td>0.187 ±0.355</td><td>0.187 ±0.355</td></tr><tr><td>KernelSHAP (SLIC)</td><td>0.227 ±0.165</td><td> $0 . 7 4 9 \pm 0 . 1 7 1$ </td><td> $0 . 1 8 1 \pm 0 . 1 8 2$ </td><td>0.700 ±0.208</td><td> $0 . 2 9 7 \pm 0 . 2 5 2$ </td><td> $0 . 4 7 8 \pm 0 . 3 0 7$ </td><td>0.304 ±0.257</td><td> $0 . 9 0 2 \pm 0 . 1 7 1$ </td><td>0.172 ±0.263</td><td>0.307 ±0.295</td></tr><tr><td>KernelSHAP (Otsu)</td><td>0.416 ±0.179</td><td> $0 . 5 7 2 \pm 0 . 1 8 8$ </td><td> $0 . 3 2 0 \pm 0 . 2 4 4$ </td><td> $0 . 5 2 7 \pm 0 . 2 6 6$ </td><td> $0 . 4 7 5 \pm 0 . 2 3 1$ </td><td> $0 . 4 3 0 \pm 0 . 2 3 9$ </td><td>0.538 ±0.244</td><td> $0 . 6 7 3 \pm 0 . 1 7 8$ </td><td>0.378 ±0.188</td><td>0.350 ±0.237</td></tr><tr><td>SHAP-BPT (AA)</td><td>0.301 ±0.226</td><td> $0 . 8 6 9 \pm 0 . 1 1 0$ </td><td> $0 . 0 9 5 \pm 0 . 1 3 0$ </td><td> $0 . 7 9 6 \pm 0 . 1 7 9$ </td><td> $0 . 3 4 6 \pm 0 . 2 1 5$ </td><td> $0 . 6 6 1 \pm 0 . 2 4 0$ </td><td>0.271 ±0.252</td><td> $\mathbf { 0 . 9 2 6 \pm 0 . 1 6 9 }$ </td><td>0.140 ±0.184</td><td>0.503 ±0.280</td></tr><tr><td>SHAP-BPT (BPT)</td><td>0.182 ±0.166</td><td> $0 . 8 0 2 \pm 0 . 1 4 6$ </td><td> $\mathbf { 0 . 0 9 1 \pm 0 . 1 4 1 }$ </td><td> $0 . 7 6 0 \pm 0 . 2 0 7$ </td><td> $0 . 2 7 3 \pm 0 . 2 3 9$ </td><td> $0 . 5 3 0 \pm 0 . 2 9 3$ </td><td>0.259 ±0.239</td><td> $0 . 9 1 1 \pm 0 . 1 7 6$ </td><td>0.142 ±0.243</td><td>0.298 ±0.304</td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.131 ±0.137</td><td> $\mathbf { 0 . 8 8 1 \pm 0 . 0 8 0 }$ </td><td> $0 . 1 3 7 \pm 0 . 1 0 8$ </td><td>0.851 ±0.112</td><td> $\mathbf { 0 . 1 1 9 \pm 0 . 1 2 5 }$ </td><td> $\mathbf { 0 . 8 6 4 \ : \pm 0 . 1 6 6 }$ </td><td>0.224 ±0.185</td><td> $0 . 8 9 7 \pm 0 . 1 6 3$ </td><td>0.146 ±0.239</td><td>0.873 ±0.053</td></tr></table>

Table S6. Quantitative comparison of explanation methods on ViT-B/16. Lower Deletion AUC is better (↓); higher Insertion AUC is better (↑). Bold indicates best performance.
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet</td><td colspan="2">Galaxy10</td><td colspan="2">EuroSAT</td><td colspan="2">Waterbirds</td><td colspan="2">TissueMNIST</td></tr><tr><td>Del ↓</td><td>Ins ↑</td><td>Del↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td></tr><tr><td>Grad-CAM</td><td>0.408 ±0.252</td><td>0.416 ±0.238</td><td> $0 . 4 9 5 \pm 0 . 3 3 4$ </td><td>0.319 ±0.309</td><td> $0 . 4 5 9 \pm 0 . 2 2 0$ </td><td> $0 . 4 4 1 \pm 0 . 2 1 3$ </td><td>0.360 ±0.277</td><td> $\mathbf { 0 . 9 3 0 \overset { \cdot } { = } 0 . 1 0 8 }$ </td><td>0.176 ±0.183</td><td>0.477 ±0.216</td></tr><tr><td>Grad-CAM++</td><td>0.427 ±0.229</td><td>0.424 ±0.202</td><td> $0 . 4 1 4 \pm 0 . 3 2 6$ </td><td>0.386 ±0.306</td><td> $0 . 4 8 1 \pm 0 . 2 1 3$ </td><td> $0 . 4 3 0 \pm 0 . 1 9 7$ </td><td> $0 . 3 8 7 \pm 0 . 2 5 7$ </td><td> $0 . 8 9 2 \pm 0 . 1 5 7$ </td><td>0.197 ±0.196</td><td>0.488 ±0.205</td></tr><tr><td>Layer-CAM</td><td>0.405 ±0.224</td><td>0.400 ±0.183</td><td> $0 . 2 0 8 \pm 0 . 2 1 5$ </td><td>0.525 ±0.248</td><td> $0 . 4 8 9 \pm 0 . 2 1 3$ </td><td> $0 . 4 3 8 \pm 0 . 1 8 6$ </td><td> $0 . 5 3 9 \pm 0 . 3 3 2$ </td><td> $0 . 8 6 6 \pm 0 . 1 8 0$ </td><td>0.190 ±0.246</td><td>0.323 ±0.254</td></tr><tr><td>Score-CAM</td><td>0.312 ±0.229</td><td>0.511 ±0.243</td><td> $0 . 2 4 3 \pm 0 . 2 9 2$ </td><td> $0 . 5 4 6 \pm 0 . 3 3 1$ </td><td> $0 . 3 7 3 \pm 0 . 2 1 1$ </td><td> $0 . 5 2 6 \pm 0 . 1 8 2$ </td><td> $0 . 4 6 7 \pm 0 . 3 1 7$ </td><td> $0 . 8 8 0 \pm 0 . 1 7 5$ </td><td>0.247 ±0.241</td><td>0.386 ±0.234</td></tr><tr><td>Shapley-CAM</td><td>0.440 ±0.275</td><td>0.369 ±0.211</td><td> $0 . 4 1 1 \pm 0 . 3 7 1$ </td><td> $0 . 3 9 7 \pm 0 . 3 4 3$ </td><td> $0 . 4 0 1 \pm 0 . 2 2 9$ </td><td> $0 . 5 0 0 \pm 0 . 2 1 1$ </td><td> $0 . 4 0 9 \pm 0 . 3 1 3$ </td><td> $0 . 9 1 7 \pm 0 . 1 1 9$ </td><td>0.178 ±0.184</td><td>0.470 ±0.222</td></tr><tr><td>IG</td><td>0.137 ±0.208</td><td>0.654 ±0.286</td><td> $0 . 0 8 3 \pm 0 . 1 3 1$ </td><td>0.474 ±0.310</td><td> $0 . 2 1 9 \pm 0 . 2 3 3$ </td><td> $0 . 4 0 0 \pm 0 . 3 1 9$ </td><td>0.143 ±0.281</td><td> $0 . 9 0 7 \pm 0 . 2 4 4$ </td><td>0.142 ±0.205</td><td>0.212 ±0.241</td></tr><tr><td>KernelSHAP (SLIC)</td><td>0.291 ±0.231</td><td>0.818 ±0.156</td><td> $0 . 1 1 7 \pm 0 . 1 3 6$ </td><td> $0 . 8 0 9 \pm 0 . 1 8 8$ </td><td> $0 . 3 5 0 \pm 0 . 1 7 9$ </td><td> $0 . 5 9 8 \pm 0 . 2 0 6$ </td><td>0.386 ±0.299</td><td> $0 . 8 8 2 \pm 0 . 2 3 0$ </td><td>0.169 ±0.241</td><td>0.429 ±0.271</td></tr><tr><td>KernelSHAP (Otsu)</td><td>0.476 ±0.209</td><td>0.631 ±0.193</td><td> $0 . 2 2 9 \pm 0 . 2 4 8$ </td><td>0.696 ±0.265</td><td>0.430 ±0.178</td><td>0.504 ±0.206</td><td>0.647 ±0.233</td><td>0.685 ±0.167</td><td>0.361 ±0.200</td><td>0.418 ±0.225</td></tr><tr><td>SHAP-BPT (AA)</td><td>0.244 ±0.215</td><td>0.771 ±0.140</td><td> $\mathbf { 0 . 0 6 1 \pm 0 . 0 7 8 }$ </td><td>0.862 ±0.163</td><td> $0 . 3 8 5 \pm 0 . 2 0 1$ </td><td> $0 . 6 4 9 \pm 0 . 2 0 1$ </td><td>0.312 ±0.291</td><td> $0 . 8 9 5 \pm 0 . 2 3 5$ </td><td>0.176 ±0.250</td><td>0.482 ±0.233</td></tr><tr><td>SHAP-BPT (BPT)</td><td>0.215 ±0.208</td><td>0.866 ±0.133</td><td> $0 . 0 6 7 \pm 0 . 1 1 4$ </td><td> $0 . 8 3 3 \pm 0 . 1 9 1$ </td><td> $0 . 3 0 9 \pm 0 . 1 9 0$ </td><td> $0 . 5 7 3 \pm 0 . 2 2 3$ </td><td>0.343 ±0.300</td><td> $0 . 9 0 4 \pm 0 . 2 4 0$ </td><td>0.152 ±0.240</td><td>0.443 ±0.254</td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.165 ±0.159</td><td>0.892 ±0.113</td><td>0.106 ±0.078</td><td>0.889 ±0.062</td><td>0.102 ±0.125</td><td> $\mathbf { 0 . 8 7 2 \pm 0 . 1 7 9 }$ </td><td>0.296 ±0.220</td><td>0.882 ±0.200</td><td>0.162 ±0.200</td><td>0.778 ±0.124</td></tr></table>

• Overlap Threshold: Because the Tree of Shapes is hierarchical, it inherently produces nested and overlapping structures. To constrain the number of Shapley players and prevent redundant attributions, candidate shapes are sorted by area, and any shape that overlaps with previously registered blobs by more than 50% is discarded.

• Shapley Sampling Budget (nsamples): When computing the morphological attributions via shap.KernelExplainer, we constrain the number of sampled coalitions. For our main experiments, the budget is set to N = 1024 to balance attribution stability with computational efficiency.

Shapley Sampling Budget To evaluate the convergence of the KernelSHAP estimator, we swept the sampling budget N from 32 to 4096 coalitions. As shown in Table S9, faithfulness metrics improve rapidly at lower budgets. However, both Insertion and Deletion AUC effectively plateau after 1024 samples, which the computational runtimes continues to scale. This mathematically validates our baseline choice of $N = 1 0 2 4$ as the optimal balance between attribution stability and speed.

## C.3. Ablation study

Morphological Overlap Threshold The hierarchical nature of the Tree of Shapes naturally produces overlapping nested structures. We ablated our overlap rejection threshold from 10% to 90% (Table S10) to test player redundancy. Stricter thresholds (e.g. 10%) aggressively prune players, resulting in fast computation (1.86s) but heavily penalizing Deletion AUC by discarding critical structural boundaries. Conversely, highly permissive thresholds (e.g. 90%) marginarrly improve Insertion AUC but cause the player count and runtime to explode (7.59s). The 50% baseline ensures distinct morphological players while maintaining a highly competitive runtime.

To systematically isolate the impact of our design choices on explanation faithfulness and computational efficiency, we conducted an ablation study across a randomly sampled subset of 200 images (100 from ImageNet, 100 from EuroSAT). These were evalauted across all three base architectures (ResNet-50, ViT-B/16 and ConvNeXt-Tiny). We measured Insertion and Deletion AUC alongside runtime. Note that the reported runtimes in this section include the heavy computational overload needed to compute the AUC curves, making them higher than the pure explanation generation times reported in the main text.

Scale Granularity Finally, we compare our baseline 8- level scale definition against uniformly spaced 4-bin, 10- bin and 15-bin configurations, as well as an 8-bin logarithmically spaced distribution (Table S11). Coarse granularity (4 bins) struggles to isolate specific visual evidence, resulting in a severe drop in Insertion AUC. While a logspaced distribution mathematically maximized faithfulness, its computational cost is prohibitive (over 2.5× slower than the baseline). The empirical 8-bin baseline provides robust multi-scale representations without introducing extreme computational overhead.

Table S7. Quantitative comparison of explanation methods on ConvNeXt-Tiny. Lower Deletion AUC is better (↓); higher Insertion AUC is better (↑). Bold indicates best performance.
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet</td><td colspan="2">Galaxy10</td><td colspan="2">EuroSAT</td><td colspan="2">Waterbirds</td><td colspan="2">TissueMNIST</td></tr><tr><td> $\mathrm { D e l } \downarrow$ </td><td>Ins ↑</td><td> $\mathrm { D e l } \downarrow$ </td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td><td>Del ↓</td><td>Ins ↑</td></tr><tr><td>Grad-CAM</td><td>0.218 ±0.152</td><td>0.583 ±0.172</td><td> $0 . 1 3 2 \pm 0 . 1 6 0$ </td><td>0.678 ±0.243</td><td> $0 . 5 4 6 \pm 0 . 1 9 4$ </td><td> $0 . 6 7 2 \pm 0 . 1 1 8$ </td><td>0.427 ±0.301</td><td> $0 . 8 2 8 \pm 0 . 2 0 5$ </td><td> $0 . 1 7 4 \pm 0 . 1 3 9$ </td><td>0.532 ±0.181</td></tr><tr><td>Grad-CAM++</td><td>0.233 ±0.159</td><td>0.567 ±0.169</td><td> $0 . 1 7 3 \pm 0 . 2 0 2$ </td><td>0.607 ±0.266</td><td> $0 . 5 3 8 \pm 0 . 1 8 0$ </td><td> $0 . 6 4 7 \pm 0 . 1 2 8$ </td><td> $0 . 4 7 5 \pm 0 . 3 1 2$ </td><td> $0 . 7 9 8 \pm 0 . 2 1 8$ </td><td>0.181 ±0.144</td><td>0.519 ±0.183</td></tr><tr><td>Layer-CAM</td><td>0.216 ±0.149</td><td>0.571 ±0.176</td><td>0.153 ±0.185</td><td> $0 . 6 2 4 \pm 0 . 2 6 5$ </td><td>0.530 ±0.181</td><td> $0 . 6 6 6 \pm 0 . 1 1 9$ </td><td>0.470 ±0.310</td><td>0.803 ±0.218</td><td>0.181 ±0.143</td><td>0.517 ±0.185</td></tr><tr><td>Score-CAM</td><td>0.336 ±0.206</td><td> $0 . 4 6 3 \pm 0 . 1 9 2$ </td><td> $0 . 1 8 2 \pm 0 . 2 3 6$ </td><td> $0 . 6 5 8 \pm 0 . 3 0 1$ </td><td> $0 . 5 1 9 \pm 0 . 1 7 9$ </td><td> $0 . 6 1 1 \pm 0 . 1 5 6$ </td><td> $0 . 4 6 4 \pm 0 . 3 1 2$ </td><td> $0 . 8 0 3 \pm 0 . 2 4 3$ </td><td>0.266 ±0.181</td><td>0.431 ±0.200</td></tr><tr><td>Shapley-CAM</td><td>0.218 ±0.152</td><td> $0 . 5 8 3 \pm 0 . 1 7 2$ </td><td> $0 . 1 3 2 \pm 0 . 1 6 0$ </td><td> $0 . 6 7 8 \pm 0 . 2 4 3$ </td><td> $0 . 5 4 6 \pm 0 . 1 9 4$ </td><td> $0 . 6 7 2 \pm 0 . 1 1 8$ </td><td>0.427 ±0.301</td><td> $0 . 8 2 8 \pm 0 . 2 0 5$ </td><td>0.174 ±0.139</td><td>0.532 ±0.181</td></tr><tr><td>IG</td><td>0.156 ±0.176</td><td>0.607 ±0.266</td><td> $0 . 1 6 4 \pm 0 . 2 4 0$ </td><td>0.602 ±0.364</td><td> $0 . 2 3 4 \pm 0 . 1 8 8$ </td><td> $0 . 2 9 2 \pm 0 . 2 1 2$ </td><td>0.269 ±0.423</td><td> $\mathbf { 0 . 9 0 8 \pm 0 . 2 2 4 }$ </td><td>0.181 ±0.232</td><td>0.246 ±0.257</td></tr><tr><td>KernelSHAP (SLIC)</td><td>0.195 ±0.133</td><td> $0 . 6 6 3 \pm 0 . 1 6 9$ </td><td> $0 . 1 1 7 \pm 0 . 1 3 0$ </td><td> $0 . 8 2 0 \pm 0 . 1 8 0$ </td><td> $0 . 4 4 4 \pm 0 . 1 9 4$ </td><td> $0 . 7 7 1 \pm 0 . 1 4 0$ </td><td>0.320 ±0.275</td><td> $0 . 8 8 7 \pm 0 . 2 1 8$ </td><td>0.131 ±0.119</td><td>0.598 ±0.231</td></tr><tr><td>KernelSHAP (Otsu)</td><td>0.348 ±0.130</td><td> $0 . 4 5 3 \pm 0 . 1 7 2$ </td><td> $0 . 2 9 0 \pm 0 . 2 7 5$ </td><td> $0 . 7 4 5 \pm 0 . 2 5 5$ </td><td> $0 . 5 7 1 \pm 0 . 1 8 9$ </td><td> $0 . 5 8 9 \pm 0 . 1 4 5$ </td><td>0.586 ±0.235</td><td> $0 . 6 9 3 \pm 0 . 1 6 0$ </td><td>0.392 ±0.191</td><td>0.453 ±0.196</td></tr><tr><td>SHAP-BPT (AA)</td><td>0.179 ±0.120</td><td> $0 . 7 4 1 \pm 0 . 1 4 5$ </td><td> $0 . 0 9 4 \pm 0 . 1 1 0$ </td><td> $0 . 8 7 2 \pm 0 . 0 9 2$ </td><td> $0 . 4 2 2 \pm 0 . 1 6 3$ </td><td> $0 . 7 4 0 \pm 0 . 1 0 7$ </td><td>0.258 ±0.255</td><td> $0 . 8 4 6 \pm 0 . 2 2 3$ </td><td>0.112 ±0.097</td><td>0.672 ±0.159</td></tr><tr><td>SHAP-BPT (BPT)</td><td>0.131 ±0.108</td><td> $0 . 7 3 3 \pm 0 . 1 5 8$ </td><td> $\mathbf { 0 . 0 6 5 \pm 0 . 0 8 9 }$ </td><td> $0 . 8 3 9 \pm 0 . 1 8 3$ </td><td> $0 . 4 0 6 \pm 0 . 1 9 0$ </td><td> $0 . 7 9 6 \pm 0 . 1 4 0$ </td><td>0.275 ±0.270</td><td> $0 . 9 0 6 \pm 0 . 2 2 2$ </td><td>0.122 ±0.134</td><td>0.615 ±0.228</td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.097 ±0.071</td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 0 7 6 }$ </td><td> $0 . 1 0 5 \pm 0 . 0 6 1$ </td><td> $\mathbf { 0 . 8 8 8 \pm 0 . 0 6 2 }$ </td><td> $\mathbf { 0 . 1 2 7 \pm 0 . 1 0 2 }$ </td><td> $\mathbf { 0 . 9 2 1 } \pm \mathbf { 0 . 0 8 4 }$ </td><td>0.252 ±0.211</td><td> $0 . 8 8 8 \pm 0 . 1 9 0$ </td><td>0.124 ±0.088</td><td>0.807 ±0.107</td></tr></table>

<table><tr><td>Method</td><td>ResNet-50</td><td>ViT-B/16</td><td>ConvNeXt-Tiny</td></tr><tr><td>Grad-CAM</td><td>0.01</td><td>0.02</td><td>0.02</td></tr><tr><td>Layer-CAM</td><td>0.01</td><td>0.02</td><td>0.02</td></tr><tr><td>Score-CAM</td><td>0.74</td><td>1.24</td><td>0.40</td></tr><tr><td>Integrated Gradients (IG)</td><td>0.81</td><td>3.05</td><td>1.08</td></tr><tr><td>KernelSHAP (SLIC)</td><td>0.42</td><td>1.35</td><td>0.70</td></tr><tr><td>SHAP-BPT (BPT)</td><td>0.24</td><td>0.34</td><td>0.28</td></tr><tr><td>MORPHOSHAP (Ours)</td><td>0.09</td><td>0.16</td><td>0.13</td></tr></table>

Table S8. Detailed runtime breakdown (seconds/image) on ImageNet for each architecture. Lower is better $( \downarrow )$

Robustness to Alternate Background Baselines. In our primary experiments, we utilize a fixed mid-gray background image. To ensure that our performance gains are not artifacts of this specific imputation method, we evaluated the robustness of MORPHOSHAP across diverse background imputation strategies, including Gaussian blur, random uniform noise, pure white, pure black, and per-image global mean color. As shown in Table S12, MORPHOSHAP remains highly robust across masking strategies. While using the image’s global mean slightly optimizes the metrics (Insertion AUC: 0.882), the gray baseline (0.875) provides excellent stability without requiring dynamic per-image tensor recalculations. Notably, the pure black baseline degrades performance significantly (Insertion AUC drops to 0.786), primarily driven by severe out-of-distribution artifacts induced in the satellite imagery (EuroSAT) domain.

Isolated vs. Pipeline Resilience. As shown in Table S13, under pristine isolated conditions, the featurelearned MLP achieves near-perfect classification accuracy (99.60%). However, when deployed end-to-end on multishape composite images, the extracted morphological components inherently suffer from extraction noise, boundary discretization artifacts, and occlusion from overlapping shapes. Consequently, hand-crafted geometric rules drop sharply to 59.83% global accuracy. In contrast, our featurelearned MLP maintains high resilience to this algorithmic extraction noise, achieving 77.86% global accuracy across all extracted pipeline components, proving its necessity for downstream applications.

Real-World Geometry Recovery on GTSRB. To quantitatively evaluate whether MORPHOSHAP’s morphological decomposition successfully isolates true geometric primitives in noisy, natural imagery (overcoming background clutter and camera perspective skew), we conducted an empirical validation on the German Traffic Sign Recognition Benchmark (GTSRB) [43]. Traffic signs possess legally standardized, ground-truth geometry (e.g., speed limits as Circles, warning signs as Triangles, and priority signs as Rectangles).

We evaluated $ { N _ { \mathrm { ~ \scriptsize ~ = ~ } 5 0 0 } }$ test images spanning these classes. As detailed in Table S14, the neural MLP tagger achieves an overall primitive recovery rate of 74.44%, significantly outperforming the rule-based heuristic (57.00%). The accuracy is particularly robust for sharp, bounded geometry such as Triangles (86.6%) and Rectangles (85.3%). This confirms that MORPHOSHAP’s hierarchical decomposition successfully isolates macro-morphological boundaries, and that our MLP effectively maps warped contours to human-interpretable geometric tags.

Table S9. Ablation on Shapley Sampling Budget. The baseline configuration is bolded.
<table><tr><td>Budget (N)</td><td>Insertion AUC (↑)</td><td>Deletion AUC (↓)</td><td>Runtime (s)*</td></tr><tr><td>32</td><td>0.871</td><td>0.264</td><td>1.34</td></tr><tr><td>64</td><td>0.873</td><td>0.248</td><td>1.46</td></tr><tr><td>128</td><td>0.875</td><td>0.245</td><td>1.71</td></tr><tr><td>256</td><td>0.875</td><td>0.244</td><td>2.04</td></tr><tr><td>512</td><td>0.875</td><td>0.244</td><td>2.50</td></tr><tr><td>1024 (Baseline)</td><td>0.875</td><td>0.244</td><td>3.09</td></tr><tr><td>2048</td><td>0.875</td><td>0.244</td><td>3.76</td></tr><tr><td>4096</td><td>0.875</td><td>0.244</td><td>4.64</td></tr></table>

<sup>∗</sup>Runtime includes full AUC curve evaluation overhead.

Table S10. Ablation on Morphological Overlap Threshold. The baseline configuration is bolded.
<table><tr><td>Overlap Threshold</td><td>Insertion AUC (↑)</td><td>Deletion AUC (↓)</td><td>Runtime (s)*</td></tr><tr><td>0.10</td><td>0.854</td><td>0.249</td><td>1.68</td></tr><tr><td>0.20</td><td>0.857</td><td>0.255</td><td>2.06</td></tr><tr><td>0.30</td><td>0.863</td><td>0.254</td><td>2.33</td></tr><tr><td>0.40</td><td>0.871</td><td>0.250</td><td>2.71</td></tr><tr><td>0.50 (Baseline)</td><td>0.875</td><td>0.244</td><td>3.09</td></tr><tr><td>0.60</td><td>0.882</td><td>0.234</td><td>4.08</td></tr><tr><td>0.70</td><td>0.889</td><td>0.224</td><td>5.01</td></tr><tr><td>0.80</td><td>0.895</td><td>0.218</td><td>6.30</td></tr><tr><td>0.90</td><td>0.902</td><td>0.213</td><td>7.59</td></tr></table>

<sup>∗</sup>Runtime includes full AUC curve evaluation overhead.

## D. Detailed User Study Setup and Results

## D.1. Study Setup

Stimuli. We compiled a dataset of 150 images from the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) [13, 37] validation dataset. Since most of the ImageNet classes are highly specific or technical, we used a Large Language Model (Google Gemini) to generate a broad, easily recognizable category for each label. From this augmented set, we randomly selected 150 unique classes and sampled one random image per class. To ensure participants could accurately identify the objects without requiring domain expertise, the stimuli were presented with the original label followed by the simplified category in parentheses (e.g. ”Schipperke (Dog)”, ”Titi (Monkey)”.

Participants. A total of 43 participants took the survey. There were no specific inclusion criteria. The recruited participants are from 7 countries across 3 continents.

Study Procedure. Prior to the main evaluation, we provided a detailed tutorial to familiarize participants with the interface and the task (see Figure S5). Each participant inspected 20 images. For each image, we prepared heatmaps using four different methods: MORPHOSHAP, PartitionSHAP (Axis Aligned Partitioning), Grad-CAM, and ShapBPT.

The evaluation for each image consisted of two tasks. For the first task, participants were evaluated on a single heatmap and corresponding masked image. To ensure balanced representation per user, the assigned method was strictly controlled so that every participant saw exactly 5 heatmaps from each of the 4 methods across their 20 trials, with the presentation sequence fully randomized. For each image, users completed the following steps:

• Classification. Participants were presented with a single heatmap and its corresponding masked image sideby-side and asked, ”What is this object?”. Their task was to identify the correct class of the image from five multiple-choice options (four specific classes and a fifth ”Not sure” option). To ensure a fair comparison, the size of the masked region was strictly controlled across all methods. For MORPHOSHAP, the mask was generated by selecting its most important morphological blob. For the pixel-attribution baselines (PartitionSHAP, Grad-CAM, and ShapBPT), we selected their top salient pixels until the total area exactly matched the area of the MOR-PHOSHAP blob. This constraint guarantees that no baseline benefited from revealing a larger portion of the image.

Table S11. Ablation on Scale Granularity and Distribution. The baseline configuration is bolded.
<table><tr><td>Scale Definition</td><td>Insertion AUC (↑)</td><td>Deletion AUC (↓)</td><td>Runtime (s)* *</td></tr><tr><td>4 Bins (Linear)</td><td>0.737</td><td>0.455</td><td>1.02</td></tr><tr><td>10 Bins (Linear)</td><td>0.828</td><td>0.308</td><td>1.20</td></tr><tr><td>15 Bins (Linear)</td><td>0.853</td><td>0.275</td><td>1.58</td></tr><tr><td>8 Bins (Baseline)</td><td>0.875</td><td>0.244</td><td>3.09</td></tr><tr><td>8 Bins (Log-spaced)</td><td>0.898</td><td>0.194</td><td>7.67</td></tr></table>

<sup>∗</sup>Runtime includes full AUC curve evaluation overhead.

Table S12. Robustness of MORPHOSHAP across different background imputation strategies, averaged across ImageNet and EuroSAT benchmarks.
<table><tr><td>Background Imputation Strategy</td><td>Insertion AUC (↑)</td><td>Deletion AUC (↓)</td></tr><tr><td>Global Image Mean</td><td>0.882</td><td>0.252</td></tr><tr><td>Mid-Gray (Baseline)</td><td>0.875</td><td>0.244</td></tr><tr><td>Uniform Noise</td><td>0.852</td><td>0.197</td></tr><tr><td>Pure White</td><td>0.839</td><td>0.186</td></tr><tr><td>Gaussian Blur</td><td>0.822</td><td>0.192</td></tr><tr><td>Pure Black</td><td>0.786</td><td>0.337</td></tr></table>

• Subjective Preference. Immediately following the classification task, participants were shown the unmasked original image alongside its correct class label. All four generated heatmaps were displayed simultaneously. Participants were then asked, ”Which heatmap provides the clearest and most accurate explanation?” To guide their evaluation, they were explicitly instructed to: (1) select the top one or two explanations that best helped them understand the class, and (2) choose explanations where the highlighted regions accurately identified relevant pixels without being so broad that they became uninformative.

## D.2. Participant Demographics

We collected responses from 42 participants from 7 unique countries. Figure S6 provides a comprehensive overview of the participant demographics, highlighting the diversity in age, gender, education and AI expertise.

## D.3. Detailed Metrics, Statistical Analysis and Subgroup Analysis

In addition to the overall method comparison presented in the main text, Figure S7 shows detailed breakdowns of accuracy, decision time and preference.

Furthermore, we analyzed the results across different demographic subgroups, as shown in Figures S9 and S8, indicating that MORPHOSHAP’s improvements hold consistently regardless of the participants ML expertise or educational background.

Statistical Analysis. To validate our findings, we conducted a one-way Analysis of Variance (ANOVA) across the evaluated explanation methods. The analysis reports a statistically significant main effect of the explanation method on user accuracy $( F = 4 . 4 , p < 0 . 0 1 )$ . However, the effect of the method on user decision time was not found to be statistically significant $( F = 0 . 4 , p < 0 . 7 5 )$ . These results indicate that MORPHOSHAP significantly improves the users’ ability to correctly identify the target class without requiring additional cognitive processing time compared to baselines. Additionally, a Chi-Square goodnessof-fit test on the multi-select preference data confirmed that user selections for the best heatmap were not uniformly distributed $( \chi ^ { 2 } = 5 5 2 . 4 , p < 0 . 0 0 1 )$ , demonstrating a highly significant qualitative preference for MORPHOSHAP over all baseline methods.

Table S13. Comparative performance of geometry tagging approaches on isolated synthetic shapes versus components extracted end-to-end by the MORPHOSHAP pipeline from complex synthetic composites.
<table><tr><td rowspan="2">Classifier Framework</td><td>Isolated Synthetic</td><td colspan="2">Pipeline Extracted Components</td></tr><tr><td>Accuracy (%)</td><td>Global Accuracy (%)</td><td>Macro F1-Score</td></tr><tr><td>Rule-based geometric naming</td><td>89.30%</td><td>59.83%</td><td>0.554</td></tr><tr><td>Learned MLP geometric naming</td><td>99.60%</td><td>77.86%</td><td>0.777</td></tr></table>

Table S14. Real-world primitive recovery rate on GTSRB traffic sign contours across 500 test instances. A recovery is counted if the MORPHOSHAP pipeline successfully extracts and correctly names the ground-truth geometric shape of the traffic sign.
<table><tr><td>Shape Class</td><td>Sample Size (N)</td><td>Rule-Based (geom_tagger)</td><td>Neural MLP (mlp-tagger)</td></tr><tr><td>Circle</td><td>317</td><td>49.5%</td><td>67.8%</td></tr><tr><td>Triangle</td><td>142</td><td>67.6%</td><td>86.6%</td></tr><tr><td>Rectangle</td><td>34</td><td>82.4%</td><td>85.3%</td></tr><tr><td>Overall Recovery Rate</td><td>493</td><td>57.00%</td><td>74.44%</td></tr></table>

![](images/548788ee7e5f8b9c72a81c687f1bd163cfda97db9de20fb54a1daccc65d6a78c.jpg)

![](images/62a673ae7e423a91430a551af18e61975db404c65c927122f42decbd8a414997.jpg)

![](images/f7fe3921de1a789dfafbc8a64ed4b6106f46c0a16d81751ae174cb318d8f0745.jpg)

![](images/58bc6e1eb25e8951e5410d4e47a942d3448c1f7bb28bdf76d55ba17353a337db.jpg)  
Figure S5. User Study Interface. From left to right: the first two images show the tutorial given to the users to explain the task. The last two images provide an example of the main interface.

![](images/7954b3aedb283e3a94971f5f1032ddb4d44c38e59dbadebe59e2e552ee2b83cd.jpg)

![](images/90c28f859dd65d9a586dd1a71ba7c97a5420a52d4e200697c7dafc04253f0930.jpg)

![](images/3fe6fff8eefb5474965169dfbb83da489182ba80e9e54376173e83540772b748.jpg)

![](images/375b7b9d44d3521046e028c969c2ceeb9fd4455cf212a78788ca032a142910ae.jpg)

Figure S6. Participant Demographics including age, gender, education level and ML expertise.  
![](images/19c6365350f52c96c674524d2f6b6d1c0e7b20a8cdd46e92c04f77b4979740fc.jpg)

![](images/cb3ff9f6c943af85a1c55ea048ea40214d677078faacd9c2212f256934ca5154.jpg)

![](images/9957f345f50239391b489aa4db0193e17066230d2280b209e1c174c009d45bcf.jpg)

Figure S7. Detailed comparison of confidence and trust scores across methods.  
![](images/4c1734b93bd8f7546400b7ea328fcc4323a47376f51884352c2ca875c8bc9541.jpg)

![](images/e7b60d99d3e2931286a76d6c1777dd09a90168dc9ad483634716e5d5022fce08.jpg)

![](images/3a15c385f465595320048248c8eeb4fbcccb7e0054d33fffdfcf2e9bc20f4ad9.jpg)  
Figure S8. Performance Metrics categorized by participants’ ML expertise.

![](images/2a56d74386254b4a858bda754addb8effed21f54521732fae41f834b9777081e.jpg)

![](images/f7dbab8a492b90492f2ac77ae05e7f74531c7ea9ef337af3d5f3590c1d8335c2.jpg)  
Figure S9. Performance Metrics categorized by participants’ education levels.

![](images/e7bf1b4ea2a9e638d6720dea78e5d14eb976d36ead236886d3f49863cfa6ab66.jpg)