# RouteGraph-Mona: Confusion-Aware Routing Fine-Tuning for Mineral Image Classification

Jierui Li<sup>⋆1</sup> , Zhiyuan Qi<sup>⋆2</sup> , Hao Zhu<sup>1</sup> , Yufan Liu<sup>3</sup> , Jixian Liu<sup>1</sup> , Shaojie Jiang<sup>1</sup> , Jianda Wang<sup>1</sup> , Yaqi Liu<sup>3</sup> , Xiaotong Li<sup>B1</sup> , Wei Wang<sup>B3</sup>

<sup>1</sup> Xidian University, Xi’an, China

<sup>2</sup> Tsinghua University, Beijing, China

3 Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), China jerryli@stu.xidian.edu.cn, qzy24@mails.tsinghua.edu.cn lixiaotong@xidian.edu.cn, wangwei@gml.ac.cn

Abstract. Mineral image classification is important for geological exploration and resource development, but it remains challenging due to substantial intra-class variations in appearance and high inter-class visual similarity. Multi-cognitive Visual Adapter (Mona) is a vision-oriented parameter-eficient adapter that adapts pre-trained visual models by tuning only a few parameters. However, Mona statically aggregates responses from multiple scales, limiting its ability to accommodate sample-specific scale preferences and model confusion among visually similar mineral categories. To address this issue, we propose RouteGraph-Mona, a lightweight route-space regularization method built on Mona. Specifically, we replace Mona’s static multi-scale aggregation with sampleadaptive routing. The resulting branch-selection behavior defines a compact routing space that captures each image’s scale preferences. We then regularize the resulting routing signatures with class-wise route anchors and confusion-weighted margins. The route anchors encourage class-consistent routing patterns, while the margins promote greater separation between visually similar categories in the routing space. Experiments on three public mineral image datasets with two visual backbones show that RouteGraph-Mona consistently outperforms Mona in mean accuracy and remains competitive with representative fine-tuning methods and mineral image classification baselines. The project repository is available at https://github.com/rgmona/RouteGraph-Mona.

Keywords: Mineral Image Classification · Parameter-Eficient Fine-Tuning · Adaptive Routing · Route-Space Regularization · Mona Adapter

## 1 Introduction

Mineral image classification is important for geological exploration, resource development, and resource evaluation [23]. Traditional mineral identification relies on expert knowledge and specialized instruments, including Raman spectroscopy, X-ray fluorescence spectroscopy, X-ray difraction, scanning electron microscopy, and automated mineralogy systems [9,1,2]. Although reliable, these methods are costly and complicated, limiting large-scale, low-cost automated identification [30,15,26]. In contrast, image-based classification is easy to deploy, and existing CNN-based, feature-fusion-based, DenseNet-based, YOLOv8-based, and Transformer-based methods have shown strong potential for mineral recognition [4,15,26,14,21,3].

![](images/606fde4645013c72e6106c932980c35cad81d181bdc25dae446b95e966fdf418.jpg)  
Fig. 1. Overview of RouteGraph-Mona. The method summarizes adaptive route weights over Mona’s multi-scale branches into route signatures and regularizes them with class-wise route anchors and an online confusion graph for confusion-aware routespace separation.

However, deep classification models require large annotated datasets and substantial computational resources, whereas collecting and labeling mineral images is dificult [4,15,26,14]. Pre-training and fine-tuning address this challenge by transferring visual knowledge from large-scale models to downstream tasks [28,19,18]. Full fine-tuning adapts all model parameters, while parametereficient fine-tuning updates only lightweight modules and freezes most of the backbone [11,12,16,10].

Among parameter-eficient visual fine-tuning methods, Mona is a representative vision-oriented adapter [27]. It introduces filtering structures and scaled normalization to improve visual transferability [27]. However, its direct application to mineral image classification remains challenging. Mineral images often exhibit substantial intra-class variations in appearance and high inter-class visual similarity, where samples from the same category may vary in shape, color, and texture, while diferent categories may share similar local cues [15,26,14]. Under these conditions, Mona’s static averaging of multi-scale branch responses may fail to capture image-specific scale preferences, while standard classificationdriven adaptation does not explicitly exploit category-level confusion.

We observe that routing weights induced by adaptive multi-scale aggregation define a low-dimensional route space that reflects image-specific preferences over diferent receptive-field scales. Compared with visual feature embeddings, this space emphasizes branch-selection patterns rather than appearance details, making it suitable for mineral images with substantial intra-class variation. We therefore regularize the routing signatures generated by adaptive routing to promote class-consistent routing patterns and improve separation between visually confusing categories.

To this end, we propose RouteGraph-Mona, a lightweight route-space regularization method for mineral image classification. As illustrated in Fig. 1, RouteGraph-Mona replaces Mona’s static multi-scale aggregation with sampleadaptive routing, constructs route signatures from the routing weights, and regularizes them with class-wise route anchors and a confusion-weighted route margin. The route anchors provide category-level route prototypes, while the online confusion graph assigns larger weights to route-margin penalties for visually confusing category pairs. This design improves relative route-space separation without requiring labels, graph updates, or auxiliary losses during inference.

The main contributions are summarized as follows:

– We propose RouteGraph-Mona, which introduces sample-adaptive routing into Mona’s multi-scale branches and forms a low-dimensional route space for mineral image classification.

We design class-wise route anchors and an online confusion graph to regularize routing signatures, guiding category-level routing tendencies and enlarging route-level separation between visually ambiguous categories.

– Experiments on three public mineral image datasets and two visual backbones show that RouteGraph-Mona improves the mean accuracy over Mona across the evaluated settings and achieves competitive performance against representative baselines.

## 2 Related Work

## 2.1 Mineral Image Classification Methods

Early mineral image classification methods mainly relied on handcrafted descriptors, such as color, texture, and shape features, together with traditional machine learning classifiers [23,30,15]. With the development of deep learning, CNN-based and Transformer-based models have been increasingly used for mineral image recognition, enabling discriminative visual representations to be learned directly from images [4,15,26,14]. Recent studies have further explored DenseNet variants, YOLOv8-CLS, and task-specific transformer models for mineral classification, typically combined with transfer learning, feature fusion, data augmentation, or multi-scale contextual modeling to improve recognition performance [3,21,14]. These methods mainly improve mineral recognition by designing stronger classifiers or feature extractors. In contrast, our work focuses on parameter-eficient adaptation of pre-trained visual backbones, while regularizing class-aware routing behavior in the route space.

## 2.2 Parameter-Eficient Visual Fine-Tuning

Pre-training and fine-tuning have become a common paradigm for transferring large-scale visual models to downstream recognition tasks [28,19,18]. Although full fine-tuning usually provides strong adaptation, it updates all model parameters and incurs high computational and storage costs. Parameter-eficient fine-tuning methods reduce this cost by updating only a small number of parameters or lightweight modules, such as adapters, LoRA, visual prompt tuning, and other eficient adaptation modules [11,12,16,10]. Mona further introduces vision-friendly filtering structures and scaled normalization, providing an efective adapter design for visual transfer tasks [27]. However, these methods are mainly optimized by the classification objective and do not explicitly model the category-level structure of mineral images. To address this limitation, RouteGraph-Mona regularizes the route space induced by adaptive multi-scale aggregation, using class-wise route anchors and an online confusion graph to improve route-level separation for visually ambiguous mineral categories.

## 3 Method

## 3.1 Overall Framework

We propose RouteGraph-Mona, a lightweight route-space regularization method built on Mona and designed for mineral image classification. As shown in Fig. 2(a), RouteGraph-Mona replaces Mona’s static averaging of multi-scale branch responses with sample-adaptive routing and summarizes the resulting routing weights into compact route signatures. To avoid directly constraining diverse visual appearances, these signatures describe Mona’s branch-selection behavior rather than visual feature embeddings. They are further regularized by class-wise route anchors and an online confusion graph, so that visually confusing categories can be better separated in the route space. During inference, RouteGraph-Mona uses a standard forward pass and predicts from the classification logits.

## 3.2 Parameter-eficient Fine-tuning Protocol

RouteGraph-Mona follows a parameter-eficient fine-tuning protocol. To adapt pretrained visual backbones to mineral recognition while reducing training cost, we initialize ViT [7] and Swin [20] with ImageNet-1K pretrained weights [6] and freeze the main backbone parameters. Mona modules are inserted after the self-attention and feed-forward sublayers in ViT blocks, and after the window attention and MLP sublayers in Swin blocks.

To keep the overall adaptation lightweight, only task-specific components are updated, including the Mona projections, multi-scale depthwise convolution branches, the scale router, class-wise route anchors, and the classification head. The online confusion graph is updated from model predictions during training and is not a trainable parameter.

![](images/e98065c63bafa7f87244d8979ef6bf18ebc63e8005b03eb05324928c7bb3f532.jpg)  
Fig. 2. Architecture of RouteGraph-Mona. (a) Overall backbone with route signatures and route-space regularization. (b) Static average fusion in Mona. (c) Router-based weighted fusion in RouteGraph-Mona; anchors and the confusion graph regularize the route space.

## 3.3 Routing Signature Generation

Each Mona module contains three depthwise convolution branches with kernel sizes $3 \times 3 , 5 \times 5$ , and $7 \times 7$ . Let $\mathcal { Q } = \{ 3 , 5 , 7 \}$ denote the set of branch scales, and let ${ \bf B } _ { q } ( { \bf h } _ { i } ^ { m } )$ be the response of branch q for sample i at the m-th Mona module. As shown in Fig. 2(b), the original Mona statically averages multi-scale branch responses:

$$
\mathbf { z } _ { i , \mathrm { { m o n a } } } ^ { m } = \frac { 1 } { | \mathcal { Q } | } \sum _ { q \in \mathcal { Q } } \mathbf { B } _ { q } ( \mathbf { h } _ { i } ^ { m } ) .\tag{1}
$$

To accommodate sample-specific scale preferences under substantial intraclass appearance variations, RouteGraph-Mona replaces this static fusion with a lightweight sample-adaptive scale router, as illustrated in Fig. 2(c). For each sample, $\mathbf { p } _ { i } ^ { m }$ is obtained by fusing a global context (the CLS token for ViT or mean-pooled tokens for Swin) with mean-pooled projected tokens. The router then predicts:

$$
\pmb { \alpha } _ { i } ^ { m } = \mathrm { s o f t m a x } \left( \mathbf { W } _ { r } ^ { m } \mathbf { p } _ { i } ^ { m } + \mathbf { b } _ { r } ^ { m } \right) ,\tag{2}
$$

where ${ \pmb { \alpha } } _ { i } ^ { m } = [ \alpha _ { i , 3 } ^ { m } , \alpha _ { i , 5 } ^ { m } , \alpha _ { i , 7 } ^ { m } ]$ and $\begin{array} { r } { \sum _ { q \in \mathcal { Q } } \alpha _ { i , q } ^ { m } = 1 } \end{array}$ . The adaptive multi-scale response is computed as:

$$
\mathbf { z } _ { i } ^ { m } = \sum _ { q \in \mathcal { Q } } \alpha _ { i , q } ^ { m } \mathbf { B } _ { q } ( \mathbf { h } _ { i } ^ { m } ) .\tag{3}
$$

When $\alpha _ { i , q } ^ { m } = 1 / | \mathcal { Q } |$ , this formulation reduces to Mona’s static average fusion.

For a backbone with L Transformer blocks, we collect routing weights from all $M = 2 L$ inserted Mona modules and compute their mean and standard deviation across modules:

$$
\pmb { \mu } _ { i } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \pmb { \alpha } _ { i } ^ { m } , \quad \pmb { \sigma } _ { i } = \sqrt { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left( \pmb { \alpha } _ { i } ^ { m } - \pmb { \mu } _ { i } \right) ^ { \odot 2 } } .\tag{4}
$$

The operations in $\sigma _ { i }$ are element-wise. The final route signature is obtained by concatenation followed by $\ell _ { 2 }$ normalization:

$$
\mathbf { r } _ { i } = \frac { \left[ \pmb { \mu } _ { i } ; \pmb { \sigma } _ { i } \right] } { \left\| \left[ \pmb { \mu } _ { i } ; \pmb { \sigma } _ { i } \right] \right\| _ { 2 } } .\tag{5}
$$

Thus, $\mathbf { r } _ { i }$ compactly characterizes sample-specific routing preferences across Mona modules.

## 3.4 Class-wise Route Anchors

For mineral images, samples from the same category may difer significantly in shape, color, and texture. To regularize category-level routing tendencies without directly constraining visual feature embeddings, we introduce class-wise route anchors in the route space. These anchors serve as compact route prototypes. In the general formulation, each class c is associated with K learnable route anchors:

$$
\mathcal { A } _ { c } = \{ \mathbf { a } _ { c , k } \} _ { k = 1 } ^ { K } .\tag{6}
$$

In our main experiments, we set $K = 1$ , which corresponds to one compact route prototype for each mineral category. This setting is simple, stable, and empirically suficient for the evaluated mineral datasets, while the formulation still allows multiple anchors when finer-grained route prototypes are required.

The route similarity between sample i and class c is computed using the closest route anchor:

$$
\rho _ { i c } = \operatorname* { m a x } _ { k } \cos \left( \mathbf { r } _ { i } , \mathbf { a } _ { c , k } \right) ,\tag{7}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity. When $K = 1$ , this similarity reduces to the cosine similarity between the sample route signature and the class-level route prototype. Let $\pmb { \rho } _ { i } = [ \rho _ { i 1 } , \dots , \rho _ { i C } ]$ denote the route-level class similarity vector. To encourage samples to align with their category-level route prototype, we define the anchor loss as:

$$
\mathcal { L } _ { a n c h o r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \ell _ { \mathrm { r o u t e } } ( \rho _ { i } , y _ { i } ) + \left[ m _ { g } + \operatorname* { m a x } _ { c \neq y _ { i } } \rho _ { i c } - \rho _ { i y _ { i } } \right] _ { + } \right] ,\tag{8}
$$

where $\ell _ { \mathrm { r o u t e } }$ denotes the cross-entropy loss computed on the route-level similarity vector, and $m _ { g }$ is the route margin.

## 3.5 Online Confusion-guided RouteGraph Regularization

Some mineral categories are visually similar and are more likely to be confused by the classifier. To focus route-level regularization on such dificult class pairs, we construct an online confusion graph G during training. The graph is initialized as a row-normalized of-diagonal matrix, where each entry $\mathbf { G } _ { u v }$ measures the confusion tendency from class u to class v.

For each mini-batch, we estimate confusion using the detached prediction distribution, so that the graph reflects current model predictions without introducing extra gradient paths. The diagonal prediction probability is removed and the remaining probabilities are normalized:

$$
\tilde { p } _ { i } ( v ) = \frac { \mathbb { I } [ v \neq y _ { i } ] \operatorname { s g } ( p _ { i } ( v ) ) } { \sum _ { c \neq y _ { i } } \operatorname { s g } ( p _ { i } ( c ) ) + \epsilon } ,\tag{9}
$$

where $p _ { i } ( v )$ is the predicted probability of class $v , \ \mathrm { s g ( \cdot ) }$ denotes stop-gradient, and ϵ is a small constant for numerical stability. The confusion graph is updated with momentum:

$$
\mathbf { G } _ { u v }  \beta \mathbf { G } _ { u v } + ( 1 - \beta ) \mathbb { E } _ { i : y _ { i } = u } [ \tilde { p } _ { i } ( v ) ] , \quad u \neq v .\tag{10}
$$

After row-wise re-normalization, a larger $\mathbf { G } _ { u v }$ indicates stronger confusion from class u to class v.

Using this graph, we impose a confusion-weighted route margin:

$$
\mathcal { L } _ { g r a p h } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { v \neq y _ { i } } \mathbf { G } _ { y _ { i } v } \left[ m _ { g } + \rho _ { i v } - \rho _ { i y _ { i } } \right] _ { + } .\tag{11}
$$

This loss assigns larger penalties to confusing category pairs and encourages stronger route-level separation.

## 3.6 Training Objective and Inference Stage Details

To combine classification learning with route-space regularization, the final training objective is:

$$
\mathcal { L } = \mathcal { L } _ { c e } + \lambda _ { m } \left( \mathcal { L } _ { a n c h o r } + \mathcal { L } _ { g r a p h } \right) ,\tag{12}
$$

where $\mathcal { L } _ { c e }$ is the standard classification loss and $\lambda _ { m }$ controls the strength of route-level regularization. The anchor loss guides class-level routing tendencies, while the graph loss enlarges the separation between confusing classes.

During training, the main backbone parameters are frozen, while Mona modules, scale routers, route anchors, and the classification head are updated by back-propagation. The online confusion graph is updated from detached model predictions and is not trainable. During inference, no labels, graph updates, or auxiliary loss computations are required. The adaptive scale router remains active, and the model predicts the class using the standard classification logits.

## 4 Experiments

## 4.1 Datasets

We evaluate RouteGraph-Mona on three public mineral image classification datasets: Minet [5], MinetV2 [29], and MineralPhotos [8]. Before splitting, we identify exact duplicate images by file-hash matching and remove duplicate entries with cross-class label conflicts to reduce potential data leakage and ambiguous supervision. The dataset sizes below correspond to the image sets used in our experiments.

Minet contains 955 hand-specimen images from seven categories: biotite, bornite, chrysocolla, malachite, muscovite, pyrite, and quartz. It is small and classimbalanced, making it suitable for evaluating limited-data adaptation.

MinetV2 follows the same seven-class taxonomy and contains 2,091 images with higher diversity. Compared with Minet, it shows stronger intra-class variation and inter-class ambiguity, such as chrysocolla–malachite and muscovite–quartz similarities.

MineralPhotos contains 39,354 images from 15 categories, providing a larger and more diverse setting for evaluating RouteGraph-Mona.

## 4.2 Compared Methods

We compare RouteGraph-Mona with two groups of baselines: general fine-tuning methods and representative mineral image classification methods. The fine-tuning baselines include linear probing, full fine-tuning, adapter tuning [11], LoRA [12], and Mona [27]. Linear probing updates only the classification head, while full fine-tuning updates the entire pre-trained backbone. Adapter tuning and LoRA represent widely used parameter-eficient fine-tuning strategies. Mona is the most direct baseline, since RouteGraph-Mona is built on top of Mona by introducing adaptive routing, route-anchor regularization, and confusion-guided route-margin regularization. For mineral image classification baselines, we include DenseNet-201 [13], EficientNet-B0 [24], YOLOv8-cls [21], and SwinMin [14]. These methods cover conventional convolutional networks, eficient convolutional architectures, modern classification frameworks, and a Transformer-based model designed for mineral recognition.

## 4.3 Experimental Settings

We use ViT-B/16 [7] and Swin-B [20] as pretrained backbones. Each cleaned dataset is split into training, validation, and test sets at 60%/20%/20%, with exact-hash grouping to prevent duplicate leakage. All experiments are repeated with seeds 0, 666, and 2026.

Accuracy is the main evaluation metric. For each run, the checkpoint with the lowest validation loss is evaluated once on the test set. Models supporting pretraining use ImageNet-1K weights [6]. For fairness, all fine-tuning baselines share the same splits, input resolution, augmentation, optimizer, early-stopping criterion, seeds, and checkpoint-selection rule. Mineral classification baselines are trained on the same cleaned splits following standard settings or oficial implementations when available.

Table 1. Comparison of fine-tuning methods on three mineral image datasets. Accuracy is reported as mean±std over three independent runs. The best and second-best results under each backbone are marked in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">ViT-B/16</td><td colspan="3">Swin-B</td></tr><tr><td>Minet</td><td>MinetV2</td><td>MineralPhotos</td><td>Minet</td><td>MinetV2</td><td>MineralPhotos</td></tr><tr><td>Linear</td><td>87.04±1.04</td><td>72.17±1.45</td><td>63.77±0.27</td><td>87.88±0.41</td><td>71.78±1.35</td><td>64.51±0.66</td></tr><tr><td>Full FT</td><td>90.74±0.95</td><td>76.73±4.46</td><td>72.91±0.25</td><td>92.93±1.24</td><td>80.50±1.06</td><td>80.98±0.82</td></tr><tr><td>Adapter</td><td>90.57±1.67</td><td>76.81±1.28</td><td>76.02±0.73</td><td>91.75±0.63</td><td>78.93±0.89</td><td>79.63±0.71</td></tr><tr><td>LoRA</td><td>91.25±0.48</td><td>75.79±0.87</td><td>74.78±0.67</td><td>93.10±1.56</td><td>77.59±0.19</td><td>81.05±0.61</td></tr><tr><td>Mona</td><td>90.91±0.71</td><td>76.34±1.42</td><td>73.85±0.27</td><td>92.26±0.86</td><td>78.22±1.42</td><td> $7 8 . 5 0 { \scriptstyle \pm 0 . 4 9 }$ </td></tr><tr><td>RouteGraph-Mona</td><td>92.09±1.05</td><td>77.12±2.87</td><td>76.25±0.51</td><td>93.94±0.71</td><td>79.64±0.98</td><td>81.09±0.51</td></tr></table>

Unless otherwise specified, we use 224 × 224 inputs, AdamW [22], a weight decay of $1 \times 1 0 ^ { - 4 }$ , and an early-stopping patience of 15. For RouteGraph-Mona, we set one route anchor per class, $\lambda _ { m } = 0 . 0 3$ $m _ { g } = 0 . 0 5$ , and $\beta = 0 . 9 5$

## 4.4 Main Comparison with Fine-Tuning Methods

As shown in Table 1, RouteGraph-Mona improves the mean accuracy over its direct baseline Mona across all three mineral datasets and both visual backbones. Under ViT-B/16, the gains over Mona are 1.18, 0.78, and 2.40 percentage points on Minet, MinetV2, and MineralPhotos, respectively; under Swin-B, the corresponding gains are 1.68, 1.42, and 2.59 percentage points.

RouteGraph-Mona also outperforms linear probing and provides stable gains over generic parameter-eficient fine-tuning methods such as adapter tuning and LoRA. Compared with full fine-tuning, it achieves competitive performance in most settings while keeping the main pretrained backbone frozen. These results indicate that mineral image classification benefits from explicit route-space regularization in addition to general feature adaptation. In addition, on crossdataset evaluation between Minet and MinetV2, RouteGraph-Mona improves Mona from 73.35% to 73.98% for Minet → MinetV2 and from 94.44% to 94.61% for MinetV2 → Minet, suggesting mild robustness under dataset shift.

## 4.5 Class-Balanced Evaluation and Statistical Analysis

Since mineral datasets are often class-imbalanced, we further report macro-F1 and mean class accuracy (mA) for the main Swin-B setting on MinetV2 using the same three seeds as Table 1. Mona obtains 78.22±1.42 accuracy, 77.36±1.36 macro-F1, and 77.38±1.53 mA, while RouteGraph-Mona obtains 79.64±0.98 accuracy, 79.00±0.84 macro-F1, and 78.97±1.12 mA. Beyond the 1.42 percentagepoint gain in overall accuracy, RouteGraph-Mona improves macro-F1 and mA by 1.64 and 1.59 percentage points, respectively. The consistent gains across overall accuracy and both class-balanced metrics suggest that the improvement also extends to more balanced class-wise recognition performance. A paired comparison over matched seeds shows positive gains, although the paired t-test does not reach statistical significance due to the small number of seeds $( p = 0 . 1 5 1$ for accuracy). We therefore treat these results as supporting evidence of a consistent trend rather than a standalone claim of statistical significance.

![](images/0e3ebaac3159d8e8ce5f8fc70770cd9b4c324e87dc2f1960e33c38d550bfb6c8.jpg)  
Fig. 3. Accuracy–parameter trade-of on Minet under two visual backbones. The xaxis denotes the percentage of trainable parameters, the y-axis denotes the mean test accuracy over three independent runs, and the bubble size represents the peak GPU memory measured on seed 0 using the same training-mode profiling protocol. Orange markers indicate RouteGraph-Mona.

## 4.6 Accuracy–Eficiency Trade-of

Figure 3 compares accuracy, trainable parameter ratio, and peak GPU memory. Under both ViT-B/16 and Swin-B, RouteGraph-Mona improves the accuracy over Mona while retaining a low trainable-parameter ratio. This indicates that the performance gain does not rely on large-scale unfreezing of the pretrained backbone, but is achieved through lightweight adaptation and routespace regularization. The bubble sizes also show that introducing the scale router and route-space regularization incurs additional training-time memory compared with Mona, while preserving the parameter-eficient update scheme. RouteGraph-Mona updates only lightweight adaptation components, scale routers, route anchors, and the classification head, without large-scale backbone updates. During inference, no labels, graph updates, or auxiliary losses are required; the scale router remains active, and predictions use the classification logits.

## 4.7 Comparison with Mineral Image Classification Methods

We further compare RouteGraph-Mona with representative mineral image classification methods, including DenseNet-201, EficientNet-B0, YOLOv8-cls, and

Table 2. Comparison with representative mineral image classification methods on three mineral datasets. Accuracy is reported as mean±std over three independent runs. The best and second-best results are marked in bold and underlined, respectively.
<table><tr><td>Method</td><td>Minet</td><td>MinetV2</td><td>MineralPhotos</td></tr><tr><td>DenseNet-201</td><td> $8 9 . 9 0 { \pm } 0 . 4 1 $ </td><td> $7 5 . 5 5 { \pm } 1 . 8 5$ </td><td> $7 8 . 8 4 { \pm } 0 . 2 4$ </td></tr><tr><td>EfficientNet-B0</td><td> $9 0 . 5 7 { \pm } 2 . 4 9$ </td><td> $7 6 . 1 0 { \pm } 0 . 9 5 $ </td><td> $7 6 . 5 1 { \pm } 1 . 2 9$ </td></tr><tr><td>YOLOv8-cls</td><td> $8 5 . 5 2 { \pm } 1 . 8 6 $ </td><td> $6 8 . 8 7 { \pm } 2 . 0 3 $ </td><td> $6 0 . 5 4 { \pm } 0 . 4 5$ </td></tr><tr><td>SwinMin</td><td> $9 1 . 4 1 { \pm } 1 . 0 9 \ $ </td><td> $7 9 . 0 1 { \pm } 2 . 7 8 $ </td><td> $8 0 . 3 2 { \pm } 1 . 5 6 \ $ </td></tr><tr><td>RouteGraph-Mona</td><td> $\mathbf { 9 3 . 9 4 } \pm \mathbf { 0 . 7 1 }$ </td><td> $\mathbf { 7 9 . 6 4 \pm 0 . 9 8 }$ </td><td> $\mathbf { 8 1 . 0 9 } \pm \mathbf { 0 . 5 1 }$ </td></tr></table>

Table 3. Ablation study on Minet and MinetV2 using the Swin-B backbone. Accuracy is reported as mean±std over three runs.
<table><tr><td rowspan="3">Variant</td><td colspan="3">Component</td><td colspan="2">Accuracy</td></tr><tr><td></td><td></td><td>Router Anchor Conf. Margin</td><td>Minet</td><td>MinetV2</td></tr><tr><td>Mona</td><td></td><td></td><td></td><td> $9 2 . 2 6 { \pm } 0 . 8 6 $ </td><td>78.22±1.42</td></tr><tr><td>+ Adaptive Routing</td><td>√</td><td></td><td></td><td> $9 2 . 4 2 { \pm } 1 . 0 1 $ </td><td> $7 8 . 3 8 { \pm } 1 . 5 7$ </td></tr><tr><td>+ Route Anchor</td><td>√</td><td>√</td><td></td><td> $9 2 . 4 2 { \pm } 0 . 8 7$ </td><td> $7 8 . 4 6 { \pm } 1 . 7 1 $ </td></tr><tr><td>+ Confusion Margin</td><td>√</td><td></td><td>√</td><td> $9 2 . 2 6 { \pm } 1 . 2 7 $ </td><td> $7 8 . 7 7 { \pm } 2 . 1 6 $ </td></tr><tr><td>RouteGraph-Mona</td><td>√</td><td>√</td><td>√</td><td> $\mathbf { 9 3 . 9 4 } \pm \mathbf { 0 . 7 1 }$  </td><td> $\mathbf { 7 9 . 6 4 \pm 0 . 9 8 }$ </td></tr></table>

SwinMin. These baselines cover conventional convolutional networks, eficient convolutional architectures, modern classification frameworks, and a Transformerbased model specifically designed for mineral image recognition. As reported in Table 2, RouteGraph-Mona achieves the best accuracy among the compared mineral classification baselines on all three datasets.

Specifically, RouteGraph-Mona obtains 93.94%, 79.64%, and 81.09% accuracy on Minet, MinetV2, and MineralPhotos, respectively. Compared with the strongest mineral classification baseline SwinMin, RouteGraph-Mona improves the accuracy by 2.53, 0.63, and 0.77 percentage points. This comparison shows that adapting strong pre-trained visual backbones with route-aware regularization provides a promising strategy for mineral image classification.

## 4.8 Ablation Study

We conduct an ablation study to examine the contribution of adaptive routing, class-wise route anchors, and the confusion-guided route margin. As shown in Table 3, adding adaptive routing to Mona brings small but consistent gains on both Minet and MinetV2, indicating that replacing static multi-scale aggregation with sample-adaptive routing provides a useful basis for route-space modeling. On top of adaptive routing, route anchors slightly improve accuracy on MinetV2 and reduce the standard deviation on Minet, suggesting that category-level route prototypes help regularize routing signatures. The confusion-guided route margin provides a larger gain on MinetV2, where inter-class visual ambiguity is stronger, increasing the accuracy from 78.38% to 78.77%.

Table 4. Feature-space versus route-space regularization on Minet and MinetV2 using Swin-B. Accuracy is reported as mean±std over three runs.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Space</td><td colspan="2">Minet</td><td colspan="2">MinetV2</td></tr><tr><td>Acc.</td><td>Gain</td><td>Acc.</td><td>Gain</td></tr><tr><td>Mona</td><td></td><td>92.26±0.86</td><td></td><td>78.22±1.42</td><td></td></tr><tr><td>Center Loss</td><td>Feature</td><td>92.09±1.27</td><td>-0.17</td><td>77.99±0.72</td><td>-0.23</td></tr><tr><td>SupCon</td><td>Feature</td><td>91.92±2.02</td><td>-0.34</td><td>78.93±0.49</td><td>+0.71</td></tr><tr><td>Feature Anchor+Graph Feature</td><td></td><td>91.08±2.28</td><td>-1.18</td><td>77.44±1.80</td><td>-0.78</td></tr><tr><td>RouteGraph-Mona</td><td>Route</td><td>93.94±0.71 +1.68</td><td></td><td>79.64±0.98</td><td>+1.42</td></tr></table>

When all components are combined, RouteGraph-Mona achieves the highest accuracy among the ablation variants, reaching 93.94% on Minet and 79.64% on MinetV2. The standalone gains of individual components are relatively modest, especially on the smaller Minet dataset. This suggests that the three components play complementary roles: adaptive routing defines the route space, route anchors provide category-level references, and the confusion-guided margin emphasizes separation between visually confusing categories.

## 4.9 Feature-space versus Route-space Regularization

Table 4 compares RouteGraph-Mona with several feature-space regularization variants using the Swin-B backbone. Center Loss [25] and supervised contrastive learning [17] are included as generic feature-level discriminative losses. Center Loss slightly decreases accuracy on both datasets, while supervised contrastive learning improves MinetV2 but decreases Minet, indicating that generic featurespace constraints do not consistently benefit mineral image classification.

To isolate the efect of the regularization space, we construct Feature Anchor+Graph, which uses the same anchor-based prototype constraint, confusionweighted margin, and online graph construction as RouteGraph-Mona, but applies them to visual feature embeddings rather than routing signatures. This provides a matched comparison between feature-space and route-space regularization.

As shown in Table 4, Feature Anchor+Graph performs worse than RouteGraph-Mona on both datasets, indicating that the gains do not simply arise from prototype- or graph-based regularization, but from applying these structural constraints in the route space induced by adaptive multi-scale aggregation.

## 4.10 Efect of Confusion Graph Construction

We compare diferent graph construction strategies on MinetV2 with the Swin-B backbone, where stronger inter-class visual ambiguity makes confusion-aware regularization more informative. As shown in Table 5, the no-graph setting achieves 78.46±1.71% accuracy, while the uniform graph slightly decreases the result to 78.30±2.86%, indicating that assigning equal weights to all negative classes does not provide a stable benefit. Random graph weighting increases the mean accuracy to 78.69%, but also raises the standard deviation to 2.61 and lacks semantic correspondence to actual category confusion.

![](images/ac465dccbf3269fc4cdfa8e63c45869a20a2203897a7f01fddc5ef4336cb586f.jpg)  
Fig. 4. Hyperparameter sensitivity on MinetV2 with the Swin-B backbone. Accuracy is reported as mean±std over three runs, and the diamond marker indicates the default setting.

Table 5. Efect of diferent graph construction strategies on MinetV2 using the Swin-B backbone. Accuracy is reported as mean±std over three independent runs. The online confusion graph corresponds to the main RouteGraph-Mona setting.
<table><tr><td>Graph Strategy</td><td>Accuracy</td></tr><tr><td>No Graph</td><td>78.46±1.71</td></tr><tr><td>Uniform Graph</td><td> $7 8 . 3 0 { \pm } 2 . 8 6 $ </td></tr><tr><td>Random Graph</td><td> $7 8 . 6 9 { \pm } 2 . 6 1 $ </td></tr><tr><td>Online Confusion Graph</td><td> $\mathbf { 7 9 . 6 4 \pm 0 . 9 8 }$ </td></tr></table>

In contrast, the online confusion graph achieves the best result of 79.64±0.98%, improving the no-graph setting by 1.18 percentage points while exhibiting the lowest variance among the evaluated graph strategies. These results suggest that prediction-derived confusion provides a more efective and stable weighting strategy than fixed uniform or random graph weighting. By upweighting route-margin penalties for confusing category pairs, the online graph focuses regularization on mineral categories that are more likely to be visually confused.

## 4.11 Hyperparameter Sensitivity

Figure 4 analyzes the efects of the number of route anchors K, the regularization weight $\lambda _ { m } ,$ , and the route margin $m _ { g }$ on model performance. A single route anchor per class already provides strong mean accuracy, while increasing K does not produce a consistent improvement; in particular, the performance drops noticeably at $K = 5$ . For the regularization weight, the default setting $\lambda _ { m } = 0 . 0 3$ achieves competitive performance, while the other evaluated values show only moderate fluctuations. For the route margin, $m _ { g } = 0 . 0 5$ performs well, whereas further increasing it to 0.07 leads to a performance decrease. Overall, the default settings provide stable performance, and increasing the number of anchors or the regularization strength does not consistently yield additional gains. Given the overlap among the error bars across several settings, these results are interpreted mainly as overall sensitivity trends rather than statistically significant diferences.

![](images/086249ae718a38e0b55f65da2633656414d19f9c5a6cc031cd72f9af34f226f3.jpg)

![](images/b8f75b8780294229b7f5f33e7d38654c12bc1c716e40ca6f88acc636f85f7be8.jpg)  
Fig. 5. Visualization analysis on MinetV2 using the Swin-B backbone. (a) Confusion change from Mona to RouteGraph-Mona. Negative values indicate reduced of-diagonal confusion errors, and the total of-diagonal error is decreased by 15.3%. (b) Routedistance distributions in the routing space. RouteGraph-Mona widens the separation gap between intra-class and confusing inter-class distances.

## 4.12 Analysis of Route-Space Geometry

We further examine whether RouteGraph-Mona improves the route-space geometry behind classification decisions. Figure 5 analyzes RouteGraph-Mona from the perspectives of class confusion and route-space geometry.

As shown in Figure 5, RouteGraph-Mona reduces the total of-diagonal confusion error by 15.3% compared with Mona, indicating that the proposed method alleviates misclassification among visually similar mineral categories. Quantitatively, RouteGraph-Mona increases the confusing inter-class route distance from $0 . 8 6 2 \times 1 0 ^ { - 2 } \mathrm { ~ t o ~ } 1 . 2 0 6 \times 1 0 ^ { - 2 }$ and widens the separation gap from $0 . 0 7 6 \times 1 0 ^ { - 2 }$ to $0 . 1 4 4 \times 1 0 ^ { - 2 }$ , making the route-space separation gap 1.90× as large as that of Mona. This suggests that the improvement mainly comes from better separation of visually confusing categories in the route space, rather than simply compressing samples from the same class.

## 5 Conclusion

In this paper, we propose RouteGraph-Mona, a lightweight route-space regularization framework for mineral image classification. The proposed method replaces Mona’s static multi-scale aggregation with sample-adaptive routing and applies class-wise route anchor constraints and an online confusion-guided route margin to routing signatures formed by routing weights. By regularizing routing signatures rather than visual feature embeddings directly, RouteGraph-Mona improves the relative route-space separation between visually confusing mineral categories. Experiments on three mineral image datasets and two visual backbones show that RouteGraph-Mona improves the mean accuracy over Mona across the evaluated settings and achieves competitive performance compared with representative fine-tuning and mineral image classification baselines. In our experiments, one route anchor per class is empirically suficient; nevertheless, future work may explore class-adaptive anchor allocation and more robust confusion graph updating strategies on larger and more diverse mineral datasets.

## References

1. Ali, A., Chiang, Y.W., Santos, R.M.: X-ray difraction techniques for mineral characterization: A review for engineers of the fundamentals, applications, and research directions. Minerals 12, 205 (2022)

2. Ali, A., Zhang, N., Santos, R.M.: Mineral characterization using scanning electron microscopy (SEM): A review of the fundamentals, advancements, and research directions. Appl. Sci. 13, 12600 (2023)

3. Attallah, Y., Zigh, E., Kaçmaz, S., Maazouzi, A.: Mineral classification using densenet architectures: leveraging deep feature extraction for enhanced accuracy. Int. Adv. Res. Eng. J. 9, 201–210 (2025)

4. Brempong, E.A., Agangiba, M., Aikins, D.: Minet: a convolutional neural network for identifying and categorising minerals. arXiv preprint arXiv:2111.11260 (2021)

5. Brempong, K.A.: Minerals identification dataset. Kaggle dataset (2019), available at: https://www.kaggle.com/datasets/asiedubrempong/minerals-identificationdataset

6. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)

7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

8. Geillon, F.: Mineral photos. Kaggle dataset (2024), available at: https://www.kaggle.com/datasets/floriangeillon/mineral-photos

9. Haskin, L.A., Wang, A., Rockow, K.M., Jollif, B.L., Korotev, R.L., Viskupic, K.M.: Raman spectroscopy for mineral identification and quantification for in situ planetary surface analysis: A point count method. J. Geophys. Res. Planets 102, 19293– 19306 (1997)

10. He, X., Li, C., Zhang, P., Yang, J., Wang, X.E.: Parameter-eficient model adaptation for vision transformers. In: AAAI. pp. 817–825 (2023)

11. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-eficient transfer learning for nlp. In: ICML. pp. 2790–2799 (2019)

12. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

13. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: CVPR. pp. 4700–4708 (2017)

14. Jia, L., Chen, F., Yang, M., Meng, F., He, M., Liu, H.: Swinmin: A mineral recognition model incorporating convolution and multi-scale contexts into swin transformer. Comput. Geosci. 184, 105532 (2024)

15. Jia, L., Yang, M., Meng, F., He, M., Liu, H.: Mineral photos recognition based on feature fusion and online hard sample mining. Minerals 11, 1354 (2021)

16. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: ECCV. pp. 709–727 (2022)

17. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. In: NeurIPS. pp. 18661– 18673 (2020)

18. Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., Houlsby, N.: Big transfer (BiT): General visual representation learning. In: ECCV. pp. 491–507 (2020)

19. Kornblith, S., Shlens, J., Le, Q.V.: Do better imagenet models transfer better? In: CVPR. pp. 2661–2671 (2019)

20. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022 (2021)

21. Long, B., Zhou, C.: Enhanced mineral image classification using yolov8-cls with optimized feature extraction and dataset augmentation. Informatica 49 (2025)

22. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

23. Lou, W., Zhang, D., Bayless, R.C.: Review of mineral recognition and its future. Appl. Geochem. 122, 104727 (2020)

24. Tan, M., Le, Q.V.: EficientNet: Rethinking model scaling for convolutional neural networks. In: ICML. pp. 6105–6114 (2019)

25. Wen, Y., Zhang, K., Li, Z., Qiao, Y.: A discriminative feature learning approach for deep face recognition. In: ECCV. pp. 499–515 (2016)

26. Wu, B., Ji, X., He, M., Yang, M., Zhang, Z., Chen, Y., Wang, Y., Zheng, X.: Mineral identification based on multi-label image classification. Minerals 12, 1338 (2022)

27. Yin, D., Hu, L., Li, B., Zhang, Y., Yang, X.: 5%> 100%: Breaking performance shackles of full fine-tuning on visual recognition tasks. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 20071–20081 (2025)

28. Yosinski, J., Clune, J., Bengio, Y., Lipson, H.: How transferable are features in deep neural networks? In: NeurIPS. pp. 3320–3328 (2014)

29. Youcefattallah97: Minerals identification & classification. Kaggle dataset (2022), available at: https://www.kaggle.com/datasets/youcefattallah97/mineralsidentification-classification

30. Zeng, X., Xiao, Y., Ji, X., Wang, G.: Mineral identification based on deep learning that combines image and mohs hardness. Minerals 11, 506 (2021)