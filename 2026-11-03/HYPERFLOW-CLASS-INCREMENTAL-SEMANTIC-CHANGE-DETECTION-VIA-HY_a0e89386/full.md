# HYPERFLOW: CLASS-INCREMENTAL SEMANTIC CHANGE DETECTION VIA HYPERBOLIC FLOW

Manon Bechaz´ <sup>1∗</sup>, Leo Buecher´ <sup>2,3</sup>, Emanuele Dalsasso<sup>4</sup>, Ciprian Tomoiaga<sup>5</sup>, Patrick Jayet<sup>5</sup>, Marcin Detyniecki<sup>5</sup>, Laetitia Chapel<sup>3,6</sup>, Lucas Drumetz<sup>7</sup>, Nicolas Courty<sup>2,3</sup>, Devis Tuia<sup>1</sup> <sup>1</sup>EPFL, Switzerland; <sup>2</sup>Universite de Bretagne Sud, France; ´ <sup>3</sup>IRISA, France <sup>4</sup>INRIA, France; <sup>5</sup>AXA GDAI Research, Switzerland; <sup>6</sup>Institut Agro Rennes-Angers, France <sup>7</sup>Lab-STICC, IMT Atlantique, France. Correspondence : manon.bechaz@epfl.ch

## ABSTRACT

Semantic change detection in remote sensing aims to identify and characterize land-cover transitions observed in pairs of satellite or aerial images. Existing methods assume a fixed set of change categories, limiting their applicability in real-world scenarios where changes of interest may vary, for example across geographies or over time. In this work, we tackle the problem of class-incremental semantic change detection, where a model progressively learns to detect new types of semantic change. To do so, we exploit the hierarchical structure of land-cover semantics, where newly introduced categories correspond to fine-grained refinements of existing ones, and represent said semantic categories in hyperbolic space, which is particularly well suited to represent such hierarchical relationships. To capture semantic changes, we introduce HyperFlow, a method relying on Busemann prototypes, modeling change as an anomaly in a learned semantic flow within the hyperbolic embedding space induced by segmentation. HyperFlow couples change detection to class-incremental semantic segmentation, enabling class-incremental semantic change detection. We also present HiChange, a new large-scale, high resolution remote sensing dataset with detailed, fine-grained hierarchical labels. Experimental results demonstrate that HyperFlow achieves both strong change detection and semantic segmentation performance across incremental steps. In particular, we show that progressively refining semantic categories, rather than considering them all directly, yields improved change detection, highlighting the benefits of incremental learning for semantic change detection.

## 1 INTRODUCTION

Semantic change detection (SCD) aims at identifying and characterizing land cover transitions over time by comparing pairs of co-registered satellite or aerial images. Despite being a well established research field (Jiang et al., 2022) with applications in disaster response (Zheng et al., 2021; Ahn et al., 2025), urban planning (Wang et al., 2024; Zhao et al., 2025; Zheng et al., 2021) or land cover analysis (Li et al., 2016; Lv et al., 2023; Daudt et al., 2019), current approaches remain static and assume a fixed set of change categories. This assumption is misaligned with real-world scenarios, where changing environmental and societal priorities imply evolving monitoring objectives (Tasar et al., 2019; Weng et al., 2024; Zhang et al., 2023). Adapting models to new types of changes typically requires costly retraining and annotation, making current approaches unsuitable for long-term deployment. A key observation is that land-cover categories are naturally structured in hierarchies, from broad categories like urban or vegetation to fine-grained ones such as residential building, crop field, or forest. In the context of land use monitoring, new types of changes are rarely entirely new categories; rather, they usually correspond to refining coarse classes into more specific ones, as illustrated in fig. 1, panel (1). Leveraging this hierarchy is therefore essential to enable adaptable SCD methods and calls for change detection (CD) models that (i) explicitly represent hierarchical semantics and (ii) adapt to different levels of this hierarchy over time.

To tackle these challenges, we consider the task of class-incremental semantic change detection (CI-SCD), in which a model continuously expands the types of semantic changes it detects. First, we model land-cover semantics in hyperbolic space, which is particularly well suited to represent hierarchical structures (Nickel & Kiela, 2017; Ganea et al., 2018a,b). Second, we leverage advances in class-incremental semantic segmentation (CISS) to advance CI-SCD. We introduce a dynamic hyperbolic formulation of CI-SCD, where semantic categories are obtained from hierarchical embeddings in hyperbolic space, while change is modeled as an anomaly in a learned semantic flow within this structured space. This approach couples SCD to CISS, enabling CI-SCD, and ensures that the detected changes are adapted to the updated semantic classes (see fig. 1, panel (2)).

![](images/afc3e13098559b2d73862f1fdc7067b2424629e2355439ce7556919eed357d3e.jpg)  
Figure 1: Overview of CI-SCD via Hyperbolic Flow. (1) Land-cover semantics are naturally organized in hierarchies. (2) During CI-SCD, the model leverages this taxonomy to progressively refine semantic categories from coarse to fine-grained classes. Change maps evolve to account for the progressively refined segmentation. (3) Hyperbolic semantic flow: a vector field transforming the embeddings of the first image into the embeddings of the second image is learned. The hyperbolic distance between the positions of the predicted embeddings and the observed positions indicates change. (4) Prototype-based incremental learning: we refine a coarse class by splitting it into leaves prototype nodes. The vector field in (3) is then retrained to adapt to the new semantic classes.

Specifically, we represent pixels at different time steps as points in a shared hyperbolic embedding space induced by the segmentation. We then model change as an anomaly in a learned semantic flow in this space, i.e., a displacement between embeddings at times $t _ { 0 }$ and $t _ { 1 } { : }$ when a pixel changes (e.g., from vegetation to building), its embedding moves between regions of the manifold, producing a displacement in embedding space between $t _ { 0 }$ and $t _ { 1 }$ , as illustrated in fig. 1, panel (3). By learning the expected flow accounting for small domain shifts naturally occurring between images (e.g. due to seasonal or illumination changes), the remaining deviations indicate the unexpected changes to be detected. To learn this semantic flow in hyperbolic space, we introduce HyperFlow, a method based on the Busemann function. Building on prior work in hyperbolic segmentation (Ghadimi Atigh et al., 2022; Berg et al., 2024), CISS (Hindel et al., 2024) and Riemannian vector fields (Katsman et al., 2023), we show that Busemann prototypes are the key to link this approach to CISS within a single hyperbolic method. In particular, incremental learning is handled directly through those prototypes, as depicted in fig. 1, panel (4): when a coarse category is refined, its prototype is split to initialize fine-grained subclasses, preserving geometric proximity to the parent while enabling specialization. The semantic flow is then retrained in this updated space, thus becoming specific to the newly introduced subclasses and allowing the detection of new semantic transitions.

In summary, this work makes three main contributions. First, we propose HyperFlow, a method that represents semantic categories using Busemann prototypes and models change via a semantic flow in a shared embedding space, coupling segmentation and CD in a geometrically grounded manner. Second, we show how HyperFlow can be leveraged to address the task of CI-SCD by extending CISS methods. In this formulation, incremental refinement is achieved by splitting parent prototypes to initialize newly introduced subclasses, preserving hierarchical structure while ensuring that the semantic flow, retrained to account for this new embedding space, adapts consistently as the taxonomy evolves. Third, as current benchmarks in SCD are limited to a small number of generic semantic categories (Zhang et al., 2025) lacking hierarchical structure, we present HiChange, a large-scale dataset with a rich hierarchical label structure designed to enable CI-SCD models to detect fine grained changes of increasing semantic complexity.

## 2 RELATED WORK

## 2.1 SEMANTIC CHANGE DETECTION IN REMOTE SENSING

CD enables the identification of differences between remotely sensed images of the same location acquired at different times. Initially based on hand-crafted features (Malila, 1980; Volpi et al., 2012), recent methods leverage deep learning, typically with Siamese designs (Daudt et al., 2018; Jiang et al., 2022). However, such approaches rely on the quality of ground-truth change maps, which are difficult to obtain and often noisy (Daudt et al., 2019). To overcome the need for accurate change labels, unsupervised CD has been widely studied (Malila, 1980; Nielsen & Conradsen, 1997; Volpi et al., 2012), and is currently framed as anomaly detection, linking changes to high reconstruction errors (Touati et al., 2020; Andresini et al., 2022; Bechaz et al., 2026; R´ u˚ziˇ cka et al., 2022; Nohˇ et al., 2022; Liu et al., 2021; Luppino et al., 2021). While most methods focus on binary change detection (BCD), real world scenarios require specifying the semantics of change. SCD therefore extends BCD by assigning semantic labels to changes (Daudt et al., 2019). Early SCD methods used post-segmentation comparisons (Huang et al., 2010; Hu & Zhang, 2013; Wu et al., 2017) and suffered from error accumulation and lack temporal modeling (Volpi et al., 2013). Deep learning SCD methods have since become dominant and adopt end-to-end architectures that couple semantic segmentation and change detection (Daudt et al., 2019; Zhu et al., 2022; Ding et al., 2024). Despite this progress, existing SCD methods remain rigid: they assume fixed categories, and treat them as independent labels, ignoring the hierarchical structure that naturally underlies land-cover semantics. These assumptions are unrealistic in real-world settings, where categories are progressively refined and semantically nested, and motivate our study of CI-SCD.

## 2.2 CLASS-INCREMENTAL SEMANTIC SEGMENTATION

CISS addresses the problem of extending a segmentation model to new categories without forgetting previously learned ones. Existing CISS methods are replay-based, expansion-based, and distillationbased. Replay-based methods store past examples (Cha et al., 2021), or encode them using prototypes (Chen et al., 2023; Zhu et al., 2025) and generative models (Maracani et al., 2021). Expansionbased approaches expand the model architecture to allocate components for new classes, e.g. via parallel branches or class-specific modules (Zhang et al., 2022; Xiao et al., 2023), or by preserving prior knowledge by freezing parameters associated with previously learned classes (Srivastava et al., 2019; Cha et al., 2021). Distillation-based methods constrain the updated model to preserve the previous responses on old classes, e.g. by distilling logits (Cermelli et al., 2020; Michieli & Zanuttigh, 2019; Baek et al., 2022), intermediate features (Douillard et al., 2021; Michieli & Zanuttigh, 2019) or attention relations (Qiu et al., 2023). Moreover, recent work showed that explicitly modeling semantic hierarchies, particularly using hyperbolic embedding spaces, significantly improves classincremental segmentation via structured refinement (Hindel et al., 2024; Lin et al., 2022). In this work, we build upon these advances in CISS to address the specific problem of CI-SCD, where the taxonomy of the semantic classes is progressively refined on a corpus of images that remains fixed across incremental steps.

## 2.3 HYPERBOLIC SPACES

Hyperbolic geometry is particularly adapted to model hierarchical labels (Nickel & Kiela, 2017; Ganea et al., 2018a,b), with applications in classification (Khrulkov et al., 2020), object detection (Liu et al., 2020; Lang et al., 2022), and semantic segmentation (Weber et al., 2024). Ghadimi Atigh et al. (2022) showed that hyperbolic multinomial logistic regression improves performance and hierarchical consistency, particularly in low dimension, while Hindel et al. (2024) showed the importance of leveraging taxonomies. In this context, Busemann functions have shown promise (Chami et al., 2021; Fan et al., 2023; Ghadimi Atigh et al., 2021; Berg et al., 2024, 2025), extracting geometrically meaningful features that quantify dissimilarity to ideal prototypes located at infinity.

More recently, hyperbolic spaces have been explored for class-incremental settings, both in image classification (Cui et al., 2023) and taxonomy-aware segmentation (Hindel et al., 2024), showing that hyperbolic embeddings can mitigate catastrophic forgetting by preserving hierarchical relations between classes. In our setting, where incremental refinement corresponds to splitting coarse landcover categories into finer-grained ones, such hierarchical modeling is particularly well suited to maintain semantic consistency. We therefore adopt Busemann prototypes (Berg et al., 2024, 2025) not only to structure hierarchical segmentation, but also to define the geometric basis of our semantic flow formulation for CI-SCD.

## 2.4 RIEMANNIAN VECTOR FIELDS

A natural way to model displacements in hyperbolic space is through vector fields and their flow. Parameterizable vector fields on manifolds for deep learning tasks were first considered for extending neural Ordinary Differential Equations to the Riemannian setting (Rezende et al., 2020; Falorsi & Forre, 2020; Mathieu & Nickel, 2020; Lou et al., 2020; Katsman et al., 2021). While most ap-´ proaches rely on extrinsic, Euclidean, formulations, ignoring the geometry of the manifold, Katsman et al. (2023) designed intrinsic vector fields in hyperbolic space for Riemannian ResNets, relying on the Busemann functions. Other designs were specifically investigated for images (van Spengler et al., 2023). In this work, we adopt the approach of Katsman et al. (2023): we construct a vector field induced by the Busemann functions, in order to model natural shifts of the hyperbolic embeddings between the temporal images.

## 3 METHODOLOGY

## 3.1 CLASS-INCREMENTAL SEMANTIC CHANGE DETECTION

Let $I _ { 0 } , I _ { 1 } \in \mathbb { R } ^ { H \times W \times C _ { \mathrm { i n } } }$ denote a co-registered pair of remote sensing (RS) images acquired at two times t<sub>0</sub> and $t _ { 1 }$ . SCD aims at characterizing, for each changed pixel, the type of change. This is achieved by predicting a binary change map $C \in \{ 0 , 1 \} ^ { H \times W }$ , indicating which pixels changed, together with semantic labels $\breve { Y _ { 0 } } , Y _ { 1 } \in \breve { \mathcal { V } } ^ { H \times \mathbf { \breve { W } } }$ detailing the change type (Daudt et al., 2019).

Traditional methods assume that the semantic label space $\mathcal { V }$ is fixed and fully specified during training. Instead, we consider a class-incremental scenario where categories are progressively refined over a sequence of incremental steps $s \in \{ 0 , . . . , n \}$ . Let $T _ { K } ^ { s }$ be the task linked to the s-th incremental step associated with a task-specific label set $\mathcal { { D } } _ { s }$ , of size $K _ { s } = | \mathcal { D } _ { s } |$ . This label space forms a taxonomic tree $\mathcal { H }$ whose internal nodes represent coarse concepts (e.g., vegetation, built-up), and leaves represent fine-grained ones $( e . g .$ ., coniferous forest, lake), see fig. 1, panel (1). The same input images are considered for all incremental steps, but for each task the labels are drawn from the task-specific set $\mathcal { D } _ { s } \subseteq \mathcal { H }$ . Note that new labels in $\mathcal { { \tt y } } _ { s }$ arise from refinements of existing categories and are appended to the existing ones. For each task $T _ { K _ { s - 1 } } ^ { s - 1 } , \ s > 0$ , if a coarse class $k \in \mathcal { D } _ { s - 1 }$ is refined into children chil $\mathrm { d } ( k ) = \{ k _ { 1 } , \dots , k _ { m } \}$ , then for the task $T _ { K _ { i } } ^ { s }$ the label set is

$$
\mathcal { V } _ { s } = \mathcal { V } _ { s - 1 } \backslash k \cup \mathrm { c h i l d } ( k ) .\tag{1}
$$

Incremental steps affect both the semantic segmentation and the CD: as the semantic label space is refined, the change maps are correspondingly updated to reflect transitions between categories in $\mathcal { \mathrm { V } } _ { s }$ only, as depicted in fig. 1, panel (2).

## 3.2 HYPERBOLIC SEMANTIC SEGMENTATION

To tackle semantic segmentation in hyperbolic space, we adopt the horospherical classifier from Berg et al. (2024). It is based on Busemann prototypes, that play a central role in the change detection mechanism introduced later in section 3.5. We use a segmentation backbone where only the last network layer is mapped to hyperbolic space (Hindel et al., 2024; Ghadimi Atigh et al., 2022; Khrulkov et al., 2020). Following Hindel et al. (2024), we use the DeepLabV3 model with a ResNet-101 backbone pre-trained on ImageNet, and consider the d-dimensional Poincare ball model´ of hyperbolic space defined as $\mathbb { B } _ { d } = \{ z \in \overline { { \mathbb { R } ^ { d } } } | \| z \| ^ { 2 } < 1 \}$ . Unless stated otherwise, $d = 2 ,$ , which already provides competitive performance compared to baselines. Experiments in higher dimensions and with other backbones are in the Supplementary Materials F.2 and F.5.

Let e denote the Euclidean embedding vector produced by the segmentation backbone for pixel x. These embeddings are mapped onto the Poincare ball via the exponential map at the origin:´

$$
z = \exp _ { 0 } ( e ) = \operatorname { t a n h } \big ( \| e \| _ { 2 } \big ) \cdot e / \| e \| _ { 2 } \in \mathbb { B } _ { d } .\tag{2}
$$

Each semantic class $k \in \mathcal { V } _ { s }$ <sub>s</sub> is parameterized by a Busemann prototype $p _ { k } \in \mathbb { S } ^ { d - 1 }$ located on the boundary of the Poincare ball together with a scalar bias´ $a _ { k } \in \mathbb { R } .$ , both learned during training. They define a horosphere, which can be interpreted as the hyperbolic analog of a Euclidean hyperplane:

$$
H _ { k } = \left\{ z \in \mathbb { B } _ { d } \vert - B _ { p _ { k } } ( z ) + a _ { k } = 0 \right\} ,\tag{3}
$$

where $B _ { p _ { k } }$ is the Busemann function (Chami et al., 2021; Bridson & Haefliger, 1999):

$$
B _ { p _ { k } } ( z ) = \log \left( \frac { \| p _ { k } - z \| _ { 2 } ^ { 2 } } { 1 - \| z \| _ { 2 } ^ { 2 } } \right) .\tag{4}
$$

The logit associated with embedding z and class k is then defined as $\xi _ { k } ( z ) = - B _ { p _ { k } } ( z ) + a _ { k }$ , and class probabilities are obtained via softmax. See Supplementary Materials C for more details on hyperbolic spaces and Busemann functions.

## 3.3 CLASS-INCREMENTAL SEMANTIC SEGMENTATION

When the label set evolves from $\mathcal { \partial } _ { s - 1 }$ to $\mathcal { \partial } _ { s }$ , prototypes associated with newly introduced finegrained categories are initialized from their parent prototypes. For a coarse class k being refined into subclasses $\{ k _ { 1 } , . . . , k _ { m } \}$ , each subclass prototype $p _ { k _ { j } }$ and bias $a _ { k _ { j } }$ are initialized as

$$
p _ { k _ { j } } = \frac { p _ { k } + \epsilon _ { j } } { \| p _ { k } + \epsilon _ { j } \| _ { 2 } } \in \mathbb { S } ^ { d - 1 } , \qquad a _ { k _ { j } } = a _ { k } \in \mathbb { R } ,\tag{5}
$$

where $p _ { k } \in \mathbb { S } ^ { d - 1 }$ and $a _ { k } \in \mathbb { R }$ are the parent prototype and bias. $\epsilon _ { j } ~ \in ~ \mathbb { R } ^ { d }$ is a small random perturbation drawn from a d-dimensional Gaussian distribution. This preserves semantic continuity: new subclasses are positioned close to their parent yet far enough to support discrimination through learning. Crucially, expansion occurs only in the prototype space, avoiding modifications of the segmentation backbone. During incremental steps, parent prototypes are frozen, and only the child prototypes and biases, as well as the backbone are learned.

## 3.4 SEGMENTATION LOSS

To train the segmentation module, we follow previous work on hierarchical semantic segmentation (Li et al., 2023; Hindel et al., 2024) and leverage the class taxonomy. We frame the problem as a pixel-wise multi-label classification problem: for each pixel, the true leaf class and all its ancestors are labeled positive, while all other categories are labeled negative. We adopt a tree-min loss (Li et al., 2023) that penalizes hierarchical inconsistencies. For each class $k \in \mathcal { V } _ { s }$ , let $A _ { k }$ and $D _ { k }$ denote its ancestors and descendants in the label hierarchy (including k), and let $l _ { k } \in \{ 0 , 1 \}$ indicate whether k is labeled as positive. The tree-min loss is defined as:

$$
\mathcal { L } _ { \mathrm { h i e r } } = \sum _ { k \in \mathcal { V } _ { s } } - l _ { k } \log \left( \operatorname* { m i n } _ { u \in A _ { k } } s _ { u } \right) - ( 1 - l _ { k } ) \log \left( 1 - \operatorname* { m a x } _ { u \in D _ { k } } s _ { u } \right) ,\tag{6}
$$

where $s _ { k } = \mathrm { s i g m o i d } ( \xi _ { k } )$ . In addition to this hierarchical constraint, previous approaches (Hindel et al., 2024) use a categorical cross-entropy loss at each hierarchical level to enforce competition between sibling classes. However, since our logits are derived from Busemann distances to hyperbolic prototypes, this approach was found to lead to bias toward parent prototypes and hinder fine-grained discrimination. We instead apply cross-entropy $\mathcal { L } _ { \mathrm { C E } }$ on active leaf classes only. The same applies during inference, where only leaf logits are considered. The final segmentation loss is a weighted combination of the tree-min and cross-entropy terms:

$$
\mathcal { L } _ { \mathrm { s e g } } = \beta \cdot \mathcal { L } _ { \mathrm { h i e r } } + \mathcal { L } _ { \mathrm { C E } } .\tag{7}
$$

## 3.5 UNSUPERVISED CHANGE DETECTION VIA BUSEMANN VECTOR FIELDS

To detect changes, we freeze the trained segmentation module and monitor differences between the hyperbolic embeddings obtained from the segmentation. Due to natural perturbations such as seasonal or lighting differences, we cannot identify changes by relying on the embeddings distance directly. Instead, we learn the natural shift within semantic classes as the hyperbolic semantic flow of a learned vector field V, and reconstruct the expected second embeddings (z ) from the first ones (z ). Remaining differences between the predicted and true embeddings will indicate change, as depicted in fig. 1, panel (3), framing CD as an anomaly detection problem. This approach is based on the assumption that structural changes constitute a minority that the model does not learn and fails to reconstruct. More precisely, the predicted position of the second embeddings is obtained as $\tilde { z } _ { 2 } = \exp _ { z _ { 1 } } \left( V ( z _ { 1 } ) \right)$ , which corresponds to following the vector field and is the hyperbolic analog of the Euclidean update $\tilde { z } _ { 2 } = z _ { 1 } + V ( z _ { 1 } )$ To define and parameterize such a vector field V in hyperbolic space, we follow Katsman et al. (2023) and construct it based on Busemann functions and their gradients. Given any neural network $n _ { \omega } : \mathbb { R } ^ { K }  \mathbb { R } ^ { K }$ , the vector field is defined as:

$$
V _ { P , \omega } ( z ) = \sum _ { k \in \mathcal { V } _ { s } } \left[ n _ { \omega } ( f _ { P } ( z ) ) \right] _ { k } \nabla B _ { p _ { k } } ( z ) ,\tag{8}
$$

where $f _ { P } ( z ) = \left( B _ { p _ { 1 } } ( z ) , \ldots , B _ { p _ { K _ { t } } } ( z ) \right)$ and ∇ is the Riemannian gradient. More details are in the Supplementary Materials C.3. While $\{ p _ { 1 } , \ldots , p _ { K _ { s } } \}$ could in principle be any set of (learned or fixed) prototypes (See Appendix F.3), we reuse the segmentation prototypes and keep them fixed. This vector field is trained separately from the segmentation task, in an unsupervised way, with a loss corresponding to the mean reconstruction error:

$$
\begin{array} { r } { { \mathcal { L } } _ { C D } = d \big ( z _ { 2 } , \exp _ { z _ { 1 } } ( V _ { P , \omega } ( z _ { 1 } ) ) \big ) , } \end{array}\tag{9}
$$

where d is the hyperbolic distance, see Supplementary Materials C.1. We predict change when

$$
d \bigl ( z _ { 2 } , \exp _ { z _ { 1 } } ( V _ { P , \omega } ( z _ { 1 } ) ) \bigr ) > d _ { \mathrm { t h r e s h } } ,\tag{10}
$$

where $d _ { \mathrm { t h r e s h } }$ is computed using the triangle method (Zack et al., 1977), see Supplementary Materials F.1 for details and values. During incremental steps, when new prototypes are introduced, the vector field is relearned from scratch to capture transitions involving newly introduced classes.

## 4 DATASETS

## 4.1 HICHANGE

The analysis of hierarchical and incremental change detection requires the creation of a specialized dataset that accurately represents hierarchical relations within remote sensing images, as existing SCD datasets are limited to a small number of generic and static classes. To address this gap, we introduce HiChange, a novel dataset specifically designed for hierarchical and incremental semantic change detection in aerial imagery. HiChange is constructed from BDOrtho aerial imagery. The dataset comprises 1000 RGB-NIR tiles, each of size $5 0 0 0 \times 5 0 0 0$ pixels at 0.2m spatial resolu tion, sampled over France, and partitioned geographically into training, validation, and test sets $( 7 0 \% - \bar { 1 0 \% } - 2 0 \% )$ , ensuring no spatial overlap across splits. HiChange is fully annotated following a hierarchical taxonomy. Coarse semantic information derived from OCSGE is combined with detailed labels from FLAIR (Garioud et al., 2025) and BDTopo to label each image at different semantic levels, for which the user can specify between 4 and 32 classes of interest. Additional information on the dataset, including the complete hierarchical taxonomy, details on the label generation pipeline and the sampling strategy as well as statistics and comparison with existing benchmarks, is provided in the Supplementary Material A.

## 4.2 ADDITIONAL DATASETS

While HiChange is the only available dataset with sufficient semantic richness to support hierarchical CI-SCD, we complement our evaluation with experiments on two widely used SCD benchmarks: SECOND (Yang et al., 2020) and Hi-UCD-mini (Tian et al., 2020). Although these datasets do not support incremental evaluation, they allow us to validate the segmentation backbone and Busemann vector field–based change detection mechanism, demonstrating that our approach generalizes beyond HiChange. More details on these datasets can be found in Supplementary Materials B.

![](images/f7459729ee13fdab7377dd948dd90541bfa25c5463412d85804885ba1abc2a9a.jpg)  
Figure 2: Qualitative illustration of HyperFlow. (1) SCD at two incremental steps. For each step, we show segmentation results at both timestamps, predicted change map, and a naive baseline obtained by differencing the two segmentation maps (Segm. diff.). (2) Hyperbolic 2-d embeddings of the first and second image, and predicted second embeddings obtained via the learned vector field. Pixels corresponding to true changes (in red) exhibit significant displacement between the timestamps, while unchanged regions show minor displacements. (3) Evolution of the embedding space across incremental steps. We visualize embeddings for a single image before and after the introduction of new classes. More qualitative examples are provided in Supplementary Materials S5.

## 5 EXPERIMENTS

## 5.1 DOES HYPERBOLIC SEMANTIC FLOW ENABLE EFFECTIVE CD?

We first compare HyperFlow against unsupervised CD baselines across the three datasets. We also report the performance of the fully supervised SCD method HRSCD4 (Daudt et al., 2019) as an upper-bound reference. Implementation details are given in the Supplementary Materials D, and qualitative examples are provided in fig. 2, panel (1). Numerically, table 1 shows that HyperFlow consistently outperforms competing approaches across all datasets. To isolate the contribution of the learned vector field, we additionally compare against a naive baseline that defines change as the difference between the segmentation of the two input images (Segm. diff.). Although this baseline already surpasses several unsupervised methods, and thus suggests that strong semantic segmentation alone can outperform many unsupervised CD approaches, an important performance gap remains with HyperFlow. This difference highlights that in the naive baseline small variations in appearance between the images can slightly modify the semantic embeddings, potentially leading to label changes in the output even when no real semantic change occurred. In contrast, HyperFlow operates directly in the embedding space and measures deviations of a semantic flow accounting for these variations. These irrelevant changes are therefore not accounted for, and only deviations that cannot be explained by the learned semantic flow, i.e., semantically significant movement in the embedding space, are identified as changes. This mechanism is illustrated in fig. 2, panel (2): only pixels corresponding to true changes (shown in red) undergo significant displacement between timestamps. The hyperbolic flow therefore controls segmentation-induced errors, explaining the improvement over simple segmentation differencing, also visible in fig. 2, panel (1).

When compared with existing unsupervised methods, the strong CD results in table 1 may raise concerns about the unsupervised nature of HyperFlow, given that not only the prototypes but also the embeddings are obtained from supervised segmentation, therefore influencing the performance of Hyperflow. We emphasize that our unsupervised claim concerns the CD stage only. For CD, the supervised segmentation backbone acts purely as a feature extractor, and in this sense, Hyperflow follows common practices in unsupervised CD of using supervised pretrained feature extractors (Saha et al., 2019; Lei et al., 2026; Bergamasco et al., 2022; Du et al., 2018; Zheng et al., 2024). HyperFlow differs in that the features are specifically adapted to the target domain and semantic task, thus yielding stronger features for CD, but HyperFlow could also operate on embeddings obtained through either supervised or unsupervised pretraining, with their quality and task/domain specificity naturally affecting downstream CD performance. To support this statement, table 1 also reports results using embeddings from two foundation models: SAM3 (Carion et al., 2026) and AnySat (Astruc et al., 2025), the latter being RS-specific. Even with these fully unsupervised embeddings, HyperFlow achieves performance comparable to SOTA unsupervised CD methods. Moreover, for HiChange and Hi-UCD-mini, performance improves from SAM3 to AnySat, consistent with increasing specialization to the RS domain. Thus, while supervised segmentation embeddings contribute to the strong results observed with HyperFlow, they are not required for competitive unsupervised CD performance. More generally, HyperFlow benefits from strong representations, regardless of how they are obtained.

Table 1: Comparison of HyperFlow with state-of-the-art unsupervised CD methods across the three datasets, for the task $T _ { K _ { 0 } } ^ { 0 } ,$ i.e. before any incremental step is performed. For HyperFlow, we additionally report CD performance using different embedding sources: HyperFlow uses the in-domain embeddings produced by the segmentation pipeline, while HyperFlow (SAM3, AnySat) use embeddings from the corresponding foundation models. Bold and underline indicate the first and secondbest unsupervised scores, respectively. † The HRSCD4 model is supervised.
<table><tr><td></td><td colspan="4">HiChange</td><td colspan="4">SECOND</td><td colspan="4">Hi-UCD-mini</td></tr><tr><td></td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td><td>| Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td>CVA (Malila, 1980)</td><td>74.2</td><td>8.0</td><td>4.5</td><td>39.3</td><td>67.6</td><td>30.5</td><td>25.9</td><td>37.1</td><td>75.5</td><td>14.4</td><td>9.3</td><td>32.3</td></tr><tr><td>SiROC (Kondmann et al., 2021)</td><td>78.0</td><td>9.3</td><td>5.3</td><td>39.1</td><td>69.4</td><td>29.6</td><td>26.5</td><td>33.4</td><td>74.4</td><td>15.6</td><td>9.9</td><td>36.9</td></tr><tr><td>CDRL (Noh et al., 2022)</td><td>73.0</td><td>11.6</td><td>6.4</td><td>61.9</td><td>62.3</td><td>33.2</td><td>25.2</td><td>48.8</td><td>81.6</td><td>17.9</td><td>12.5</td><td>31.3</td></tr><tr><td>X-Net (Luppino et al., 2021)</td><td>69.0</td><td>11.5</td><td>6.3</td><td>70.1</td><td>60.8</td><td>34.7</td><td>25.5</td><td>54.2</td><td>84.9</td><td>20.4</td><td>15.4</td><td>30.1</td></tr><tr><td>ACE-Net (Luppino et al., 2021)</td><td>69.9</td><td>11.0</td><td>6.0</td><td>65.0</td><td>65.7</td><td>33.2</td><td>26.5</td><td>44.4</td><td>78.1</td><td>20.0</td><td>13.1</td><td>42.6</td></tr><tr><td>KPCAMNet (Wu et al., 2021)</td><td>92.5</td><td>20.2</td><td>14.6</td><td>33.2</td><td>77.9</td><td>24.4</td><td>35.6</td><td>18.6</td><td>89.2</td><td>16.8</td><td>16.7</td><td>17.0</td></tr><tr><td>AnyChange (Zheng et al., 2024)</td><td>94.0</td><td>19.4</td><td>15.8</td><td>25.1</td><td>81.2</td><td>33.0</td><td>52.3</td><td>24.1</td><td>88.2</td><td>26.5</td><td>22.1</td><td>33.1</td></tr><tr><td>2Player (Béchaz et al., 2026)</td><td>92.0</td><td>17.1</td><td>12.2</td><td>28.7</td><td>69.0</td><td>34.1</td><td>28.6</td><td>42.1</td><td>76.7</td><td>21.4</td><td>13.7</td><td>48.6</td></tr><tr><td>Segm. diff.</td><td>86.7</td><td>25.1</td><td>14.9</td><td>78.8</td><td>60.9</td><td>41.9</td><td>29.3</td><td>73.3</td><td>73.3</td><td>28.1</td><td>17.0</td><td>80.8</td></tr><tr><td>HyperFlow</td><td>92.9</td><td>29.6</td><td>20.5</td><td>52.8</td><td>73.7</td><td>43.0</td><td>36.8</td><td>51.8</td><td>91.7</td><td>44.3</td><td>39.0</td><td>51.4</td></tr><tr><td>HyperFlow (SAM3)</td><td>93.9</td><td>15.1</td><td>12.3</td><td>19.6</td><td>48.5</td><td>40.3</td><td>25.9</td><td>90.8</td><td>49.7</td><td>18.3</td><td>10.2</td><td>88.0</td></tr><tr><td>HyperFlow (AnySat)</td><td>93.5</td><td>20.1</td><td>14.2</td><td>34.5</td><td>45.0</td><td>36.5</td><td>23.4</td><td>82.2</td><td>84.3</td><td>23.7</td><td>17.4</td><td>37.1</td></tr><tr><td>HRSCD4† (Daudt et al., 2019)</td><td>94.2</td><td>42.9</td><td>29.8</td><td>76.4</td><td>66.5</td><td>52.0 35.9</td><td></td><td>94.4</td><td>94.0</td><td>60.9</td><td>52.3</td><td>72.9</td></tr></table>

## 5.2 CLASS-INCREMENTAL SETTINGS

When considering the incremental setting (table 2), HyperFlow maintains strong SCD performance even as the number of semantic classes grows over incremental steps. While these scenarios are inherently more challenging, not only does HyperFlow still outperform the strongest unsupervised CD baselines presented in table 1, but it is also capable of characterizing the nature of a change at a much finer lever of the semantic taxonomy.

Table 2: SCD performance of HyperFlow across incremental steps for the HiChange dataset. We report segmentation performance with mIoU, CD with F1, and combined CD and segmentation using the Semantic Change Segmentation (SCS) score from Toker et al. (2022).
<table><tr><td></td><td> $T _ { 4 } ^ { 0 }$ </td><td> $T _ { 7 } ^ { 1 }$ </td><td> $T _ { 1 3 } ^ { 2 }$ </td><td> $T _ { 1 6 } ^ { 3 }$ </td><td> $T _ { 2 0 } ^ { 4 }$ </td><td> $T _ { 2 2 } ^ { 5 }$ </td><td> $T _ { 3 2 } ^ { 6 }$ </td></tr><tr><td>Segm. [mIoU]</td><td>62.0</td><td>56.2</td><td>33.7</td><td>28.4</td><td>23.5</td><td>20.7</td><td>16.1</td></tr><tr><td>CD [F1]</td><td>29.6</td><td>29.6</td><td>29.8</td><td>29.2</td><td>27.7</td><td>31.6</td><td>26.0</td></tr><tr><td>SCD [SCS]</td><td>39.0</td><td>34.4</td><td>22.4</td><td>19.9</td><td>17.3</td><td>17.6</td><td>13.8</td></tr></table>

Table 3 compares HyperFlow segmentation results against state-of-the-art CISS methods across incremental steps for the three datasets considered. HyperFlow achieves competitive performance and, while improvements over prior approaches remain moderate, the results demonstrate that considering Busemann prototypes and incorporating CD do not degrade CISS performance.

Table 3: Comparison of Hyperflow with state-of-the-art CISS methods for the three datasets. The mIoU is reported for each method and all incremental steps. Note that since SECOND and Hi-UCD do not enable incremental scenarios, only the base task is considered (Hyp: hyperbolic).
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="6">HiChange T0</td><td rowspan="2">SECOND T0</td><td rowspan="2">Hi-UCD T9</td></tr><tr><td>Hyp.</td><td>T1</td><td> $T _ { 1 3 } ^ { 2 }$ </td><td> $T _ { 1 6 } ^ { 3 }$ </td><td> $\mathrm { \dot { ~ } } T _ { 2 0 } ^ { 4 }$ </td><td> $T _ { 2 2 } ^ { 5 }$   $T _ { 3 2 } ^ { 6 }$ </td></tr><tr><td>PLOP (Douillard et al., 2021)</td><td>X</td><td>66.3</td><td>11.4</td><td>2.0</td><td>0.4</td><td>0.9</td><td>0.8 0.6</td><td>53.7</td><td>41.8</td></tr><tr><td>MiB (Cermelli et al., 2020)</td><td>X</td><td>66.3</td><td>52.9</td><td>36.0</td><td>27.8</td><td>22.2</td><td>20.7 16.1</td><td>53.7</td><td>41.8</td></tr><tr><td>MiB+AWT (Goswami et al., 2023)</td><td>X</td><td>66.3</td><td>54.7</td><td>37.6</td><td>28.2</td><td>22.9</td><td>20.2 15.9</td><td>53.7</td><td>41.8</td></tr><tr><td>DKD (Baek et al., 2022)</td><td>x</td><td>64.7</td><td>57.4</td><td>35.0</td><td>27.8</td><td>22.1</td><td>21.5 17.2</td><td>38.8</td><td>45.8</td></tr><tr><td>TOPICS (Hindel et al., 2024)</td><td>√</td><td>67.6</td><td>54.5</td><td>24.6</td><td>15.9</td><td>8.5</td><td>6.2 3.8</td><td>46.8</td><td>37.5</td></tr><tr><td>SAM3 (Carion et al., 2026)</td><td>X</td><td>29.8</td><td>24.3</td><td>20.5</td><td>13.8</td><td>11.2</td><td>9.3 6.4</td><td>28.0</td><td>20.9</td></tr><tr><td>HyperFlow</td><td>√</td><td>62.0</td><td>56.2</td><td>33.7</td><td>28.4</td><td>23.5</td><td>20.7 16.1</td><td>47.3</td><td>42.4</td></tr></table>

We also investigate whether progressively refining the taxonomy provides advantages over directly training on the full set of fine-grained categories. Results in table 4 show that structuring semantic representations progressively, rather than directly considering fine-grained distinctions, leads to improved SCD. Figure 2, panel (3) visualizes this effect, showing how incremental refinement preserves global organization while progressively increasing semantic resolution. More embeddings comparisons across incremental and non-incremental scenarios are presented in the Supplementary Materials E.

To isolate the effect of hierarchical initialization, we also investigate in table 4 a non-incremental scenario where prototypes are initialized according to the class hierarchy. Results improve compared to the non-incremental baseline, indicating that respecting class hierarchies at initialization helps segmentation. Performance remains however lower than in the full incremental setting with progressive taxonomy refinement, showing that benefits of incremental learning go beyond hierarchical initialization alone.

Table 4: Comparison between (i) class-incremental training with progressive taxonomy refinement, (ii) non-incremental training on the full fine-grained taxonomy, and (iii) non-incremental training with hierarchical prototype initialization for the HiChange dataset. The SCS score is reported for all incremental steps. At each step, the incremental and non incremental models are trained for the same total number of epochs.
<table><tr><td>Incr.</td><td>Hier.</td><td> $T _ { 4 } ^ { 0 }$ </td><td>T7</td><td> $T _ { 1 3 } ^ { 2 }$ </td><td> $T _ { 1 6 } ^ { 3 }$ </td><td> $T _ { 2 0 } ^ { 4 }$ </td><td> $T _ { 2 2 } ^ { 5 }$ </td><td> $T _ { 3 2 } ^ { 6 }$ </td></tr><tr><td>√</td><td>√</td><td>39.0</td><td>34.4</td><td>22.4</td><td>19.9</td><td>17.3</td><td>17.6</td><td>13.8</td></tr><tr><td>x</td><td>x</td><td>39.0</td><td>30.9</td><td>16.6</td><td>13.7</td><td>11.7</td><td>11.7</td><td>9.1</td></tr><tr><td>x</td><td>√</td><td>39.0</td><td>31.7</td><td>17.8</td><td>16.2</td><td>14.4</td><td>14.1</td><td>10.0</td></tr></table>

## 6 CONCLUSION

In this work, we introduce HyperFlow, a method that bridges CISS and SCD via hyperbolic semantic flow. By representing semantic categories with Busemann prototypes and modeling temporal evolution of semantic classes as a learned vector field in hyperbolic space, we formulate CD as the identification of deviations from expected temporal dynamics. We train and evaluate HyperFlow on HiChange, a new large-scale, very high resolution and hierarchical SCD dataset that we propose. Our results demonstrate that formulating semantic change as a hyperbolic flow enables effective unsupervised CD while preserving competitive segmentation performance, both in standard and in in cremental settings. Furthermore, we show that progressively refining the taxonomy, i.e. the semantic changes considered, rather than considering all changes from the beggining, helps considerably the task of fine-grained semantic change detection. Although the CD mechanism does not require supervision, HyperFlow still depends on the quality of the semantic segmentation model. Future works will focus on learning better representations for the semantic segmentation backbone, for instance leveraging foundation models based on large scale self-supervised pretraining (Sumbul et al., 2025; Cong et al., 2022). Moreover, we will explore leveraging longer time series, which could enable the modeling of more complex or gradual temporal transitions, such as seasonal vegetation cycles or multi-stage urban development, thereby providing a richer representation of temporal dynamics.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with debugging and improving existing code, and to reformulate some paragraphs of the manuscript to improve readability. The original code and scientific text were however written by the authors and were not generated by generative AI. Generative AI tools were not used to generate research ideas or design the methodology and experiments, nor to interpret the results. All AI-generated modifications were carefully reviewed by the authors. Code modifications were verified for correctness, and changes to the manuscript carefully reviewed by multiple authors to ensure that they accurately reflected the intended meaning. We take full responsibility for the final content of this work.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, all code used in our experiments will be made publicly available upon acceptance of the paper. Detailed information on the generation of the HiChange dataset is provided in the supplementary material A. The images and label sources used to construct the HiChange dataset are publicly available, with their respective sources and links provided in Appendix A. The code used to fuse the different label sources will also be published. The resulting HiChange dataset will be publicly released upon acceptance of the paper. All other datasets used in our experiments are publicly available, and their sources are provided in Appendix B. Implementation details are provided in Appendix D. Pretrained model checkpoints used to obtain the reported results will also be released alongside the source code.

## REFERENCES

Kyeongjin Ahn, Sungwon Han, Sungwon Park, Jihee Kim, Sangyoon Park, and Meeyoung Cha. Generalizable disaster damage assessment via change detection with vision foundation model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 27784–27792, 2025.

Giuseppina Andresini, Annalisa Appice, Daniele Iaia, Donato Malerba, Nicolo Taggio, and Antonello Aiello. Leveraging autoencoders in change vector analysis of optical satellite images. Journal ofIntelligent Information Systems, 58(3):433–452, 2022.

Guillaume Astruc et al. Anysat: One earth observation model for many resolutions, scales, and modalities. In CVPR, 2025.

Donghyeon Baek, Youngmin Oh, Sanghoon Lee, Junghyup Lee, and Bumsub Ham. Decomposed knowledge distillation for class-incremental semantic segmentation. Advances in neural information processing systems, 35:10380–10392, 2022.

Manon Bechaz, Emanuele Dalsasso, Ciprian Tomoiag´ a, Marcin Detyniecki, and Devis Tuia.˘ 2player: A general framework for self-supervised change detection via cooperative learning. IS-PRS Journal ofPhotogrammetry and Remote Sensing, 232:34–47, 2026.

Paul Berg, Bjoern Michele, Minh-Tan Pham, Laetitia Chapel, and Nicolas Courty. Horospherical learning with smart prototypes. In British Machine Vision Conference (BMVC), 2024.

Paul Berg, Leo Buecher, Bj ´ orn Michele, Minh-Tan Pham, Laetitia Chapel, and Nicolas Courty.¨ Multi-prototype hyperbolic learning guided by class hierarchy: P. berg et al. International Journal ofComputer Vision, 133(11):7969–7984, 2025.

Luca Bergamasco, Sudipan Saha, Francesca Bovolo, and Lorenzo Bruzzone. Unsupervised change detection using convolutional-autoencoder multiresolution features. IEEE Transactions on Geoscience and Remote Sensing, 60:1–19, 2022.

Silvere Bonnabel. Stochastic Gradient Descent on Riemannian Manifolds.\` IEEE Transactions on Automatic Control, 58(9):2217–2229, September 2013. ISSN 1558-2523. doi: 10.1109/TAC. 2013.2254619.

Martin R. Bridson and Andre Haefliger.´ Metric Spaces of Non-Positive Curvature, volume 319 of Grundlehren Der Mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 1999. ISBN 978-3-642-08399-0 978-3-662-12494-9.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Lil-¨ iane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan ´ Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2026. URL https://arxiv.org/abs/2511.16719.

Fabio Cermelli, Massimiliano Mancini, Samuel Rota Bulo, Elisa Ricci, and Barbara Caputo. Modeling the background for incremental learning in semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9233–9242, 2020.

Sungmin Cha, YoungJoon Yoo, Taesup Moon, et al. Ssul: Semantic segmentation with unknown label for exemplar-based class-incremental learning. Advances in neural information processing systems, 34:10919–10930, 2021.

Ines Chami, Albert Gu, Dat P. Nguyen, and Christopher Re. HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical Projections. In Proceedings of the 38th International Conference on Machine Learning, pp. 1419–1429. PMLR, July 2021.

Jinpeng Chen, Runmin Cong, Yuxuan Luo, Horace Ip, and Sam Kwong. Saving 100x storage: Prototype replay for reconstructing training sample distribution in class-incremental semantic segmentation. Advances in Neural Information Processing Systems, 36:35988–35999, 2023.

Hyunghoon Cho, Benjamin DeMeo, Jian Peng, and Bonnie Berger. Large-Margin Classification in Hyperbolic Space. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, pp. 1832–1840. PMLR, April 2019.

Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David Lobell, and Stefano Ermon. Satmae: Pre-training transformers for temporal and multispectral satellite imagery. Advances in Neural Information Processing Systems, 35:197–211, 2022.

Yawen Cui, Zitong Yu, Wei Peng, Qi Tian, and Li Liu. Rethinking few-shot class-incremental learning with open-set hypothesis in hyperbolic geometry. IEEE Transactions on Multimedia, 26: 5897–5910, 2023.

Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In 2018 25th IEEE international conference on image processing (ICIP), pp. 4063–4067. IEEE, 2018.

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. Multitask learning for large-scale semantic change detection. Computer Vision and Image Understanding, 187: 102783, 2019.

Lei Ding, Jing Zhang, Haitao Guo, Kai Zhang, Bing Liu, and Lorenzo Bruzzone. Joint spatiotemporal modeling for semantic change detection in remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 62:1–14, 2024.

Arthur Douillard, Yifu Chen, Arnaud Dapogny, and Matthieu Cord. Plop: Learning without forgetting for continual semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4040–4050, 2021.

Bo Du, Lixiang Ru, Chen Wu, and Liangpei Zhang. Unsupervised deep slow feature analysis for change detection in multi-temporal remote sensing images. arXiv preprint arXiv:1812.00645, 2018.

Luca Falorsi and Patrick Forre. Neural ordinary differential equations on manifolds. ´ arXiv preprint arXiv:2006.06663, 2020.

Xiran Fan, Chun-Hao Yang, and Baba Vemuri. Horospherical Decision Boundaries for Large Margin Classification in Hyperbolic Space. Advances in Neural Information Processing Systems, 36: 11194–11204, December 2023.

Octavian Ganea, Gary Becigneul, and Thomas Hofmann. Hyperbolic Entailment Cones for Learning Hierarchical Embeddings. In Proceedings of the 35th International Conference on Machine Learning, pp. 1646–1655. PMLR, July 2018a.

Octavian Ganea, Gary Becigneul, and Thomas Hofmann. Hyperbolic Neural Networks. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018b.

Anatol Garioud, Sebastien Giordano, Nicolas David, and Nicolas Gonthier. Flair-hub: Large-scale´ multimodal dataset for land cover and crop mapping. arXiv preprint arXiv:2506.07080, 2025.

Mina Ghadimi Atigh, Martin Keller-Ressel, and Pascal Mettes. Hyperbolic busemann learning with ideal prototypes. Advances in neural information processing systems, 34:103–115, 2021.

Mina Ghadimi Atigh, Julian Schoep, Erman Acar, Nanne Van Noord, and Pascal Mettes. Hyperbolic image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4453–4462, 2022.

Dipam Goswami, Rene Schuster, Joost Van de Weijer, and Didier Stricker. Attribution-aware weight ´ transfer: A warm-start initialization for class-incremental semantic segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3195–3204, 2023.

Julia Hindel, Daniele Cattaneo, and Abhinav Valada. Taxonomy-aware continual semantic segmentation in hyperbolic spaces for open-world perception. IEEE Robotics and Automation Letters, 10(2):1904–1911, 2024.

Jinrong Hu and Yuanzhi Zhang. Seasonal change of land-use/land-cover (lulc) detection using modis data in rapid urbanization regions: A case study of the pearl river delta region (china). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 6(4):1913–1920, 2013.

Zhi Huang, Xiuping Jia, and Linlin Ge. Sampling approaches for one-pass land-use/land-cover change mapping. International Journal ofRemote Sensing, 31(6):1543–1554, 2010.

Deyi Ji, Siqi Gao, Mingyuan Tao, Hongtao Lu, and Feng Zhao. Changenet: Multi-temporal asymmetric change detection dataset. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2725–2729. IEEE, 2024.

Huiwei Jiang, Min Peng, Yuanjun Zhong, Haofeng Xie, Zemin Hao, Jingming Lin, Xiaoli Ma, and Xiangyun Hu. A survey on deep learning-based change detection from high-resolution remote sensing images. Remote Sensing, 14(7):1552, 2022.

Isay Katsman, Aaron Lou, Derek Lim, Qingxuan Jiang, Ser Nam Lim, and Christopher M De Sa. Equivariant manifold flows. Advances in Neural Information Processing Systems, 34:10600– 10612, 2021.

Isay Katsman, Eric Chen, Sidhanth Holalkere, Anna Asch, Aaron Lou, Ser Nam Lim, and Christopher M De Sa. Riemannian residual neural networks. Advances in Neural Information Processing Systems, 36:63502–63514, 2023.

Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6418–6428, 2020.

Lukas Kondmann, Aysim Toker, Sudipan Saha, Bernhard Scholkopf, Laura Leal-Taix¨ e, and Xiao Xi-´ ang Zhu. Spatial context awareness for unsupervised change detection in optical satellite images. IEEE Transactions on Geoscience and Remote Sensing, 60:1–15, 2021.

Christopher Lang, Alexander Braun, Lars Schillingmann, and Abhinav Valada. On hyperbolic embeddings in object detection. In DAGM German Conference on Pattern Recognition, pp. 462–476. Springer, 2022.

Tao Lei, Shuxin Zhang, Shaoxiong Lin, Tongfei Liu, Zhiyong Lv, Tao Gao, Maoguo Gong, and Asoke K Nandi. Remote sensing image change detection using deep learning techniques: a comprehensive survey. Artificial Intelligence Review, 2026.

Liulei Li, Wenguan Wang, Tianfei Zhou, Ruijie Quan, and Yi Yang. Semantic hierarchy-aware segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2123– 2138, 2023.

Xiaodong Li, Feng Ling, Giles M Foody, and Yun Du. A superresolution land-cover change detection method using remotely sensed images with different spatial resolutions. IEEE Transactions on Geoscience and Remote Sensing, 54(7):3822–3841, 2016.

Zhiqiu Lin, Deepak Pathak, Yu-Xiong Wang, Deva Ramanan, and Shu Kong. Continual learning with evolving class ontologies. Advances in Neural Information Processing Systems, 35:7671– 7684, 2022.

Shaoteng Liu, Jingjing Chen, Liangming Pan, Chong-Wah Ngo, Tat-Seng Chua, and Yu-Gang Jiang. Hyperbolic visual embedding learning for zero-shot recognition. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 9273–9281, 2020.

Zhun-Ga Liu, Zuo-Wei Zhang, Quan Pan, and Liang-Bo Ning. Unsupervised change detection from heterogeneous data based on image translation. IEEE Transactions on Geoscience and Remote Sensing, 60:1–13, 2021.

Aaron Lou, Derek Lim, Isay Katsman, Leo Huang, Qingxuan Jiang, Ser Nam Lim, and Christopher M De Sa. Neural manifold ordinary differential equations. Advances in Neural Information Processing Systems, 33:17548–17558, 2020.

Luigi Tommaso Luppino, Michael Kampffmeyer, Filippo Maria Bianchi, Gabriele Moser, Sebastiano Bruno Serpico, Robert Jenssen, and Stian Normann Anfinsen. Deep image translation with an affinity-based change prior for unsupervised multimodal change detection. IEEE Transactions on Geoscience and Remote Sensing, 60:1–22, 2021.

Zhiyong Lv, Ming Zhang, Weiwei Sun, Jon Atli Benediktsson, Tao Lei, and Nicola Falco. Spatial-´ contextual information utilization framework for land cover change detection with hyperspectral remote sensed images. IEEE Transactions on Geoscience and Remote Sensing, 61:1–11, 2023.

William A Malila. Change vector analysis: An approach for detecting forest changes with landsat. In LARS symposia, pp. 385, 1980.

Andrea Maracani, Umberto Michieli, Marco Toldo, and Pietro Zanuttigh. Recall: Replay-based continual learning in semantic segmentation. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 7026–7035, 2021.

Emile Mathieu and Maximilian Nickel. Riemannian continuous normalizing flows. Advances in neural information processing systems, 33:2503–2515, 2020.

Umberto Michieli and Pietro Zanuttigh. Incremental learning techniques for semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision workshops, pp. 0–0, 2019.

Maximillian Nickel and Douwe Kiela. Poincare embeddings for learning hierarchical representa-´ tions. Advances in neural information processing systems, 30, 2017.

Allan Aasbjerg Nielsen and Knut Conradsen. Multivariate alteration detection (mad) in multispectral, bi-temporal image data: a new approach to change detection studies. Dept. Mathematical Modelling, Tech. Univ. Denmark, Lyngby, Denmark, Tech. Rep, 11:1997, 1997.

Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, and Dong-Geol Choi. Unsupervised change detection based on image reconstruction loss. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1352–1361, 2022.

Nobuyuki Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979. doi: 10.1109/TSMC.1979.4310076.

Yiqiao Qiu, Yixing Shen, Zhuohao Sun, Yanchong Zheng, Xiaobin Chang, Weishi Zheng, and Ruixuan Wang. Sats: Self-attention transfer for continual semantic segmentation. Pattern Recognition, 138:109383, 2023.

Danilo Jimenez Rezende, George Papamakarios, Sebastien Racaniere, Michael Albergo, Gurtej´ Kanwar, Phiala Shanahan, and Kyle Cranmer. Normalizing flows on tori and spheres. In International Conference on Machine Learning, pp. 8083–8092. PMLR, 2020.

V´ıt Ru˚ziˇ cka, Anna Vaughan, Daniele De Martini, James Fulton, Valentina Salvatelli, Chris Bridges,ˇ Gonzalo Mateo-Garcia, and Valentina Zantedeschi. Ravæn: unsupervised change detection of extreme events using ml on-board satellites. Scientific reports, 12(1):16939, 2022.

Sudipan Saha, Francesca Bovolo, and Lorenzo Bruzzone. Unsupervised deep change vector analysis for multiple-change detection in vhr images. IEEE transactions on geoscience and remote sensing, 57(6):3677–3693, 2019.

Ryohei Shimizu, Yusuke Mukuta, and Tatsuya Harada. Hyperbolic Neural Networks++. In International Conference on Learning Representations, October 2020.

Shivangi Srivastava, Maxim Berman, Matthew B Blaschko, and Devis Tuia. Adaptive compressionbased lifelong learning. arXiv preprint arXiv:1907.09695, 2019.

Gencer Sumbul, Chang Xu, Emanuele Dalsasso, and Devis Tuia. Smarties: Spectrum-aware multisensor auto-encoder for remote sensing images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5569–5578, 2025.

Onur Tasar, Yuliya Tarabalka, and Pierre Alliez. Incremental learning for semantic segmentation of large-scale remote sensing data. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(9):3524–3537, 2019.

Shiqi Tian, Ailong Ma, Zhuo Zheng, and Yanfei Zhong. Hi-ucd: A large-scale dataset for urban semantic change detection in remote sensing imagery. arXiv preprint arXiv:2011.03247, 2020.

Aysim Toker, Lukas Kondmann, Mark Weber, Marvin Eisenberger, Andres Camero, Jingliang Hu,´ Ariadna Pregel Hoderlein, C¸ aglar S¸enaras, Timothy Davis, Daniel Cremers, et al. Dynamicearth- ˘ net: Daily multi-spectral satellite dataset for semantic change segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21158–21167, 2022.

Redha Touati, Max Mignotte, and Mohamed Dahmane. Anomaly feature learning for unsupervised change detection in heterogeneous images: A deep sparse residual model. IEEE Journal of Se lected Topics in Applied Earth Observations and Remote Sensing, 13:588–600, 2020.

Max van Spengler, Erwin Berkhout, and Pascal Mettes. Poincare resnet. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5419–5428, 2023.

Michele Volpi, Devis Tuia, Gustavo Camps-Valls, and Mikhail Kanevski. Unsupervised change detection with kernels. IEEE Geoscience and Remote Sensing Letters, 9(6):1026–1030, 2012.

Michele Volpi, Devis Tuia, Francesca Bovolo, Mikhail Kanevski, and Lorenzo Bruzzone. Supervised change detection in vhr images using contextual information and support vector machines. International Journal of Applied Earth Observation and Geoinformation, 20:77–85, 2013.

Mingze Wang, Lili Su, Cilin Yan, Sheng Xu, Pengcheng Yuan, Xiaolong Jiang, and Baochang Zhang. Rsbuilding: Towards general remote sensing image building extraction and change detection with foundation model. IEEE Transactions on Geoscience and Remote Sensing, 2024.

Simon Weber, Bar Zong¨ ur, Nikita Araslanov, and Daniel Cremers. Flattening the parent bias: Hier-¨ archical semantic segmentation in the poincare ball. In´ Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28223–28232, 2024.

Lean Weng, Wenqing Yang, Boni Hu, Pengcheng Han, Shaocheng Xue, Yu Zhang, Haowei Li, Jie Jin, and Shuhui Bu. Mdinet: Multidomain incremental network for change detection. IEEE Transactions on Geoscience and Remote Sensing, 62:1–15, 2024.

Chen Wu, Bo Du, Xiaohui Cui, and Liangpei Zhang. A post-classification change detection method based on iterative slow feature analysis and bayesian soft fusion. Remote Sensing of Environment, 199:241–255, 2017.

Chen Wu, Hongruixuan Chen, Bo Du, and Liangpei Zhang. Unsupervised change detection in multitemporal vhr images based on deep kernel pca convolutional mapping network. IEEE Transactions on Cybernetics, 52(11):12084–12098, 2021.

Jia-Wen Xiao, Chang-Bin Zhang, Jiekang Feng, Xialei Liu, Joost van de Weijer, and Ming-Ming Cheng. Endpoints weight fusion for class incremental semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7204–7213, 2023.

Kunping Yang, Gui-Song Xia, Zicheng Liu, Bo Du, Wen Yang, Marcello Pelillo, and Liangpei Zhang. Semantic change detection with asymmetric siamese networks. arXiv preprint arXiv:2010.05687, 2020.

Gregory W Zack, William E Rogers, and Samuel A Latt. Automatic measurement of sister chromatid exchange frequency. Journal ofHistochemistry & Cytochemistry, 25(7):741–753, 1977.

Chang-Bin Zhang, Jia-Wen Xiao, Xialei Liu, Ying-Cong Chen, and Ming-Ming Cheng. Representation compensation networks for continual semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7053–7064, 2022.

Hanchao Zhang, Ruiqian Zhang, Xiaogang Ning, Xiao Huang, You He, Yixin Chen, Mingzhu Li, Wei Cui, and Jiaming Wang. Lim-cd: A large-scale remote sensing change detection dataset for incremental monitoring. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 10:903–910, 2023.

Haotian Zhang, Han Guo, Keyan Chen, Hao Chen, Zhengxia Zou, and Zhenwei Shi. Foba: A foreground–background co-guided method and new benchmark for remote sensing semantic change detection. IEEE Transactions on Geoscience and Remote Sensing, 63:1–19, 2025.

Like Zhao, Mingwang Yang, Yiran Zhao, Yuqing Liu, Huawei Jiang, and Zhen Yang. Df2net: A differential-feature dynamic-fusion network for building change detection in remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 64:1–15, 2025.

Zhuo Zheng, Yanfei Zhong, Junjue Wang, Ailong Ma, and Liangpei Zhang. Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: From natural disasters to man-made disasters. Remote Sensing of Environment, 265:112636, 2021.

Zhuo Zheng, Yanfei Zhong, Liangpei Zhang, and Stefano Ermon. Segment any change. Advances in Neural Information Processing Systems, 37:81204–81224, 2024.

Yanpeng Zhou, Jinjie Wang, Jianli Ding, Bohua Liu, Nan Weng, and Hongzhi Xiao. Signet: A siamese graph convolutional network for multi-class urban change detection. Remote Sensing, 15 (9):2464, 2023.

Guilin Zhu, Dongyue Wu, Changxin Gao, Runmin Wang, Weidong Yang, and Nong Sang. Adaptive prototype replay for class incremental semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10932–10940, 2025.

Qiqi Zhu, Xi Guo, Weihuan Deng, Sunan Shi, Qingfeng Guan, Yanfei Zhong, Liangpei Zhang, and Deren Li. Land-use/land-cover change detection based on a siamese global learning framework for high spatial resolution remote sensing imagery. ISPRS Journal ofPhotogrammetry and Remote Sensing, 184:63–78, 2022.

## APPENDIX

Organization of the Appendix. These supplementary materials provide additional details and experiments complementing the main paper:

• Section A presents further information about the proposed HiChange dataset, including comparisons with existing SCD datasets, the sampling strategy, the label generation pipeline, and dataset statistics.

• Section B summarizes the characteristics of the SECOND and Hi-UCD-mini datasets used in our experiments.

• Section C reviews the hyperbolic geometry concepts underlying our method.

• Section D provides implementation details and the definition of the incremental tasks.

• Section E presents additional qualitative results.

• Section F reports complementary experiments analyzing the influence of the threshold, backbone architecture and embedding dimension.

## A ADDITIONAL INFORMATION ON THE HICHANGE DATASET

## A.1 COMPARISON WITH EXISTING SCD DATASETS

Table 5 compares HiChange with commonly used SCD datasets. The latter are typically limited to a small number of semantic categories. This limitation motivates the creation of HiChange, as such restricted semantic diversity makes them less suitable for studying CI-SCD scenarios. HiChange addresses these limitations by introducing a larger and hierarchically structured taxonomy designed to support progressive semantic refinement.

Table 5: Existing SCD datasets, and comparison with the proposed HiChange dataset.
<table><tr><td>Dataset</td><td>Image pairs</td><td>Image size</td><td>Res [m]</td><td>Categories</td></tr><tr><td>HRSCD Daudt et al. (2019)</td><td>291</td><td>10000 × 10000</td><td>0.5</td><td>5</td></tr><tr><td>Hi-UCD-mini Tian et al. (2020)</td><td>1293</td><td>1024 × 1024</td><td>0.1</td><td>9</td></tr><tr><td>SECOND Yang et al. (2020)</td><td>4662</td><td>512 × 512</td><td>not indicated</td><td>6</td></tr><tr><td>CNAM-CD Zhou et al. (2023)</td><td>2503</td><td>512 × 512</td><td>0.5</td><td>5</td></tr><tr><td>DynamicEarthNet Toker et al. (2022)</td><td>75 time series of 730 images</td><td>1024 × 1024</td><td>3</td><td>7</td></tr><tr><td>ChangeNet Ji et al. (2024)</td><td>31000</td><td>~ 1900 × 1200</td><td>0.3</td><td>6</td></tr><tr><td>LevirSCD Zhang et al. (2025)</td><td>3225</td><td>256 × 256</td><td>1-2</td><td>16</td></tr><tr><td>HiChange</td><td>1000</td><td>5000 × 5000</td><td>0.2</td><td>4 - 32, organized hierarchically</td></tr></table>

## A.2 SAMPLING OF THE IMAGES

In this section, we detail the sampling procedure used to construct the HiChange dataset. The objective was to select a subset of tiles from the French national BDOrtho aerial imagery database<sup>1</sup> that maximizes the diversity and relevance of observed semantic changes. We fixed the total number of tiles to N = 1000, a choice made to balance dataset size, semantic diversity, and feasible training time for deep-learning models. Each tile has a spatial resolution of 0.2 m and a size of 5000 × 5000 pixels. The sampling procedure consists of three main stages described below.

Initial Filtering Based on the Number of Changes Observed. To ensure that the final dataset focuses on regions with actual changes, we perform an initial selection of tiles based on the density of changes within each tile. For this purpose, we use the OCSGE database<sup>2</sup>, which we also employ for change label generation. As OCSGE is available only for specific years depending on the region, we select BDOrtho acquisition years to match the available OCSGE years to ensure temporal consistency. Acquisition years range from 2016 to 2023, and time between the 2 acquisitions for each pair is 2 to 3 years. OCSGE is provided as a vector dataset in which each changed object is represented as a polygon. For each tile in BDOrtho, we count the number of OCSGE polygons falling within its spatial extent. We retain only tiles containing more than 30 polygons of change. This threshold was chosen to filter out tiles without meaningful change activity. This procedure yields approximately 20000 tiles. As anticipated, the retained tiles concentrate around dynamic, predominantly urban, regions as depicted in fig. 3, panel (a).

![](images/4c9882337ade415a053cd6bc2d108e15de525235ce92a1bfea744f6ea9d30b36.jpg)  
(a)

![](images/318fdfccb780db3fab237124f9038f4f8ed2808a033e92e48dd2e9a0f7cdccce.jpg)  
(b)

![](images/6747f457edb33b415461ce8117988b6e6ac49a9537c64380c63f4b590b6d3c63.jpg)  
(c)

![](images/76a7b52a91bbe9879f8aac708ae196c212774ed05da0d79a9f06e43de3ebf142.jpg)  
(d)  
Figure 3: Sampling procedure for constructing the HiChange dataset. (a) Tiles retained after the initial filtering step based on the number of changes observed. (b) Geographical partitioning of French administrative departments used to define train, test and validation splits. (c) Assignment of tiles to the corresponding split according to their geographical location. (d) Final selection of the tiles after the greedy sampling algorithm, ensuring diverse and balanced semantic change types in all splits of the dataset.

Geographical train-validation-test Assignment. The second step consists in defining the dataset splits. We opt for a geographical partitioning based on administrative departments to prevent spatial overlap between training, validation, and testing. Each of the roughly 20000 candidate tiles is assigned to one of the three splits (train/validation/test), as depicted in fig. 3, panel (b) and (c).

Greedy Selection for Diversity and Class Balance. Even after the initial filtering, the number of candidate tiles remains far larger than the desired dataset size. The final selection aims to (i) maximize diversity in the types of observed semantic changes and (ii) balance the relative frequency of change categories. To this end, we exploit the land-cover attributes provided by OCSGE and adopt an iterative greedy selection procedure. For each change polygon, its change type is defined as the combination of the dominant semantic labels before and after change (i.e., an ordered pair $( \mathrm { c a t e g o r y _ { b e f o r e } , c a t e g o r y _ { a f t e r } } ) )$ ). For each train-test-validation split, we run the tile selection proce dure as follows:

1. Initialize the selected tile set as empty.

2. For each unselected tile in the split:

• Simulate adding the tile to the current selection.

• Compute the deviation between the resulting distribution of change types and the target uniform distribution across all change types.

• Measure the reduction in deviation that would result from adding the tile.

3. Select the tile that yields the greatest improvement.

4. Repeat until the desired number of tiles K for the split is reached.

This greedy algorithm progressively builds a tile set that is both diverse and balanced in terms of semantic change types. We sample K = 700 tiles for the train set, K = 100 for the validation set and K = 200 for the test set. The final selected tiles are depicted in fig. 3, panel (d).

![](images/364d7e6729395e24a8450f93b8c40ece8283e20f762b7a5f1bf177ea14cb878e.jpg)  
Figure 4: Pipeline for the generation of the labels.

## A.3 GENERATION OF THE LABELS

The pipeline for the generation of the labels is presented in Figure 4. The change detection labels are derived from the OCSGE database, which provides coarse annotations of both land-cover segmentation and change between two temporal acquisitions. We directly exploit the change information from OCSGE as a binary indication of change. This change map is later refined at usage time to align with the selected semantic granularity, as detailed later in this section. However, the semantic segmentation labels provided by OCSGE remain coarse and cover only a limited set of semantic categories<sup>3</sup>. This level of detail is insufficient for the objectives of HiChange, which require fine-grained semantic information. Therefore, additional sources of information are incorporated to construct richer semantic labels. In particular, we leverage the BDTopo database<sup>4</sup> from the French National Institute of Geographic and Forest Information (IGN) which provides high-precision vector layers describing buildings, roads, and other man-made infrastructure for multiple years. A key limitation, however, is that BDTopo is not spatially exhaustive: several regions of the territory remain unlabeled or only partially covered. One possible remedy would be to supplement BDTopo with the OCSGE semantic segmentation data, but the latter is too coarse for our target spatial resolution (0.2 m). To overcome this limitation, we employ the FLAIR land-cover model Garioud et al. (2025), which was also trained on BDOrtho imagery and has demonstrated high accuracy on segmentation tasks. We run FLAIR inference over all tiles and use its predicted land-cover labels in regions not covered by BDTopo. The resulting semantic segmentation combines BDTopo-derived vector information where available and FLAIR predictions elsewhere, thereby achieving both fine-grained semantic resolution and full spatial coverage.

Hierarchical Label Selection. The above procedure yields semantic labels at the finest level of granularity. These labels are organized into a hierarchical taxonomy structured in three levels, as shown in fig. 5. When using the dataset, the user selects the desired classes to consider. Finegrained categories are then automatically mapped to their parent categories when coarser levels are requested.

Consistency Filtering of Change Labels. To ensure coherence between the semantic segmentation labels and the change annotations, we apply a filtering mechanism to the change map at usage time. Specifically, if the selected classes are coarse and a tile exhibits a change only between child categories belonging to the same parent category (e.g., a change from forest to grass when only vegetation is considered), the change should not be reported. We therefore mark as changed only pixels that are both (i) marked as changed by OCSGE and (ii) present different semantic labels between the two timestamps at the chosen semantic level. This guarantees that the binary change masks remain semantically meaningful and consistent across hierarchical configurations, and solves at the same time the problem of the coarse resolution of OCSGE.

![](images/b6182da3025394e0b402013f77977681315e36572675f525704a8532a3a226f7.jpg)  
Figure 5: Taxonomy of the semantic labels of the HiChange dataset.

Limitations The construction of such a large dataset involves a trade-off between annotation quality and coverage. Some regions in the dataset where BDTopo is unavailable are labeled using FLAIR. These automatically generated labels may inherit biases and errors from the model, and may therefore introduce some noise into the benchmark. In addition, the use of OCSGE may exclude changes that fall below its minimum mapping unit. However, given the scale of the dataset, manual annotation is not practically feasible. The adopted methodology provides a compromise that enables large-scale evaluation while acknowledging the limitations associated with automatically generated labels.

## A.4 EXAMPLES FROM THE DATASET

Some example images from the HiChange dataset are provided in fig. 6.

## A.5 DATASET STATISTICS

The distribution of the semantic labels at the three levels of depth of the hierarchy are presented in fig. 7. The confusion matrix for the land-cover for pixels that contain change is presented in fig. 8. Although not all types of changes are present in the dataset, especially at the finest level of the hierarchy, we observe a large variety.

## B ADDITIONAL INFORMATION ON THE SECOND AND HI-UCD-MINI DATASETS

SECOND The SECOND dataset consists of 4662 pairs of 512 × 512 aerial images collected from multiple platforms and sensors over several Chinese cities. Spatial resolution is not indicated.

![](images/f5680a33ecbc5632d7ab45111b71f6d0d241c64740993f0cbccc1a76fe0f4ef9.jpg)  
Figure 6: Examples from the HiChange dataset, illustrating the refinement of semantic labels. In the initial step, only coarse distinctions between vegetation and constructed areas are considered. At step 1, we have additional information on the type of vegetation and, at step 2, about the type of construction. Note that the change maps adapt to the set of semantic labels considered: in the first example, change happens within the vegetation, and is captured only at steps 1 and 2. Similarly, in the second example, a building is constructed and the change is only captured in the last incremental step.

Images are annotated by experts on six land-cover classes: low vegetation, non-vegetated ground surface, trees, water, building and playground. Semantic annotation is provided for changed regions only, and therefore only evaluated on those regions. The dataset is split in 2968 pairs for training and 1694 for testing.

Hi-UCD-mini The Hi-UCD-mini dataset is an urban SCD dataset built from very-high-resolution aerial images (0.1 m spatial resolution). It focuses on the city of Tallinn and contains 1293 image pairs of size 1024×1024 acquired between 2017 and 2019. The images are annotated for both binary change and segmentation, covering 9 semantic categories: water, grass, building, green house, road, bridge, bare land, woodland and undefined.

## C HYPERBOLIC SPACE AND VECTOR FIELDS

This section briefly reviews the hyperbolic geometry and vector field formulation that form the mathematical foundation of our method.

## C.1 HYPERBOLIC SPACE

Hyperbolic spaces are smooth manifolds, meaning that they can be endowed with Cartesian coordinates locally around every point. This structure makes it possible to define tangent vectors in each point, corresponding to the derivatives of all possible smooth curves going through it. The set of tangent vectors in some point x of a manifold M is called the tangent space at x and is denoted $T _ { x } \bar { \mathcal { M } }$ . It is a vector space that can be seen as the first order approximation of the manifold around this point. Riemannian manifolds are smooth manifolds where each tangent space $T _ { x } { \mathcal { M } }$ is endowed with a scalar product $g _ { x } : T _ { x } { \mathcal { M } } \times T _ { x } { \mathcal { M } } \to \mathbb { R }$ , called the Riemannian metric, and which completely characterizes the geometric structure of M. Indeed, it makes it possible to define the length of a

(a)

![](images/a16fa0e6e04906691776b3bd9e983a37579eadcd93517cac8a08a90d72be1a37.jpg)

![](images/5b4a955b7fe8e354819ffdf8539694205f1644162a39d04eb094d07f8c46bcd5.jpg)  
(b)

![](images/0f8e57f1c0e359cc99c91ea9271600d2785e6878957548e7b34315cc7ed6f4de.jpg)  
(c)  
Figure 7: Distribution of the land-cover categories in the HiChange dataset at different levels of the taxonomy.

smooth curve $\gamma : [ a , b ] \to { \mathcal { M } }$ as

$$
L ( \gamma ) = \int _ { a } ^ { b } { \sqrt { g _ { \gamma ( t ) } ( { \dot { \gamma } } ( t ) , { \dot { \gamma } } ( t ) ) } } \mathrm { d } t ,\tag{11}
$$

and thus geodesics as minimal-length paths between two points. We can then define the distance between two points as the length of such a geodesic. By extension, a curve $\gamma : \mathbb { R }  \mathcal { M }$ which is a geodesic on every segment $[ a , b ]$ is also called a geodesic.

![](images/c92e90cbb1e255cc4a06c9b880334e8bf37ca8028dc04c3faac3684c582c0333.jpg)

![](images/163142af97ae0c560722f7626ac8b629287c22b8f6c3e6213ed2c7be9fee6e0c.jpg)

![](images/3516a9386d4ae2ade99fd1ce9273240999afa2107e903ece2e199149efdd7c63.jpg)

![](images/fe439bfe809fc9546498b1e355a9e56cfe703b3ba12fc35ea07039347403b3b2.jpg)

(a)  
![](images/65b1918804c46d9372d5e5cfebec489f958e346311c06d3555db3b38c78dab1f.jpg)

![](images/20c7a095f86314d753018c39876b118ab807aed68ea979d6927d7367c263ad41.jpg)  
(b)

![](images/11d23e3bfc167cf2baa6bd670cd5b0a72274628520572cfb9a18a4c98327e8ec.jpg)

![](images/acac430d1cdc9598e5980e6c824eb78f0ea4fceed4c664e3bbac45248198c346.jpg)  
(c)

![](images/5865d5195a39369bacd216a509941850b92f42ee8666459d9434ec7d354a861f.jpg)  
Figure 8: Semantic changes in HiChange. Confusion matrices of land-cover transitions for pixels labeled as change at the three hierarchical levels of the taxonomy.

Hyperbolic spaces are Riemannian manifolds with the particular property of having constant negative curvature. This curvature controls how much the space looks like a tree. The more negative, the more the geometry becomes comparable to that of an (infinite) tree. The closer to 0, the more the hyperbolic space gets similar to a Euclidean space. There exist different models of hyperbolic space that are, for a given dimension d and curvature $\kappa < 0$ , all isometric to each other. The Poincare ball´ model is one of them. Using $c = - \kappa ,$ , it is defined as

$$
\mathbb { B } _ { d } = \left\{ z \in \mathbb { R } ^ { d } \mid \| z \| _ { 2 } ^ { 2 } < \frac { 1 } { c } \right\} ,\tag{12}
$$

endowed with the Riemannian metric $g _ { x } ( u , v ) \ = \ \lambda _ { x } ^ { 2 } \langle u , v \rangle$ , where $\begin{array} { r } { \lambda _ { x } \ = \ \frac { 2 } { 1 - c \| x \| _ { 2 } ^ { 2 } } } \end{array}$ is called the conformal factor.

In such a space, geodesics are given by arcs of circle (or straight lines) orthogonal to the ”boundary” $\mathbb { S } _ { c } ^ { d - 1 } = \left\{ \bar { z } \in \mathbb { R } ^ { \breve { d } } | \Vert z \Vert _ { 2 } ^ { 2 } = 1 / { c } \right\}$ (which is not reachable in finite time), and the Riemannian distance

between two points $x , y \in \mathbb { B } _ { d }$ is given by

$$
d ( x , y ) = \frac { 1 } { \sqrt { c } } \cosh ^ { - 1 } \left( 1 + 2 c \frac { \| x - y \| _ { 2 } ^ { 2 } } { ( 1 - c \| x \| _ { 2 } ^ { 2 } ) ( 1 - c \| y \| _ { 2 } ^ { 2 } ) } \right) .\tag{13}
$$

Given a tangent vector $v \in T _ { x } \mathbb { B } _ { d }$ in a point $\boldsymbol { x } \in \mathbb { B } _ { d }$ , the exponential map ex $\mathrm { p } _ { x } : T _ { x } \mathbb { B } _ { d } \to \mathbb { B } _ { d }$ returns $\gamma _ { v } ( 1 ) \in \mathbb { B } _ { d } ^ { }$ where $\gamma _ { v } : \mathbb { R }  \mathbb { B } _ { d }$ is the constant-speed geodesic passing through x in $t = 0$ with velocity $\gamma ^ { \prime } ( 0 ) = v ,$ , as such geodesic is unique. The exponential map therefore gives a way to move in the manifold $\mathbb { B } _ { d }$ from the starting point x following a displacement vector v. Its expression is given by

$$
\exp _ { x } ( v ) = x \oplus { \frac { 1 } { \sqrt { c } } } \operatorname { t a n h } \left( { \sqrt { c } } \| v \| _ { 2 } \right) { \frac { v } { \| v \| _ { 2 } } } ,\tag{14}
$$

where ⊕ denotes the Mobius addition, expressed, for¨ $x , y \in \mathbb { B } _ { d } ,$ as

$$
x \oplus y = { \frac { ( 1 + 2 c \langle x , y \rangle + c \| y \| _ { 2 } ^ { 2 } ) x + ( 1 - c \| x \| _ { 2 } ^ { 2 } ) y } { 1 + 2 c \langle x , y \rangle + c ^ { 2 } \| x \| _ { 2 } ^ { 2 } \| y \| _ { 2 } ^ { 2 } } } .\tag{15}
$$

## C.2 BUSEMANN FUNCTION

In a complete Riemannian manifold $\mathcal { M } ,$ , for a unit-speed geodesic $\gamma : \mathbb { R }  { \mathcal { M } } .$ , the Busemann function along γ is defined as

$$
B _ { \gamma } : x \in \mathcal { M } \mapsto \operatorname* { l i m } _ { t \to \infty } d ( \gamma ( t ) , x ) - t ,\tag{16}
$$

when the limit exists. It can be interpreted as a sort of coordinate of the point x along γ.

This Busemann function is well-defined in Euclidean space, and for a unit-speed geodesic $\gamma _ { w } : t \mapsto $ tw, with $w \in \mathbb { S } ^ { d - 1 }$ , going through the origin, we have $\mathsf { \tilde { B } } _ { w } ( x ) = - \langle x , w \rangle$ ⟩. The Busemann function can therefore be seen as a generalization of the (negative) scalar product to other manifolds, and thus can be used as a tool for Riemannian machine learning.

In hyperbolic space, the Busemann function exists as well. For a prototype $p$ at infinity $( p \in \mathbb { S } _ { c } ^ { d - 1 } )$ , which can be seen as a direction, the unit-speed geodesic going to p is written:

$$
\gamma _ { p } ( t ) = \exp _ { 0 } \left( \frac { \sqrt { c } t } { 2 } p \right) = \operatorname { t a n h } \left( \frac { \sqrt { c } t } { 2 } \right) p .\tag{17}
$$

The Busemann function can then be shown to be

$$
B _ { p } ( x ) = \frac { 1 } { \sqrt { c } } \log \left( \frac { c \| x - p \| _ { 2 } ^ { 2 } } { 1 - c \| x \| _ { 2 } ^ { 2 } } \right) .\tag{18}
$$

Its level-sets

$$
H _ { p , a } = \left\{ z \in \mathbb { B } _ { d } \mid - B _ { p } ( z ) + a = 0 \right\}\tag{19}
$$

are called horospheres. In the Poincare model, they are represented by spheres tangent to the proto-´ type. They can be seen as hyperbolic equivalents for Euclidean hyperplanes. Other generalizations of hyperplanes in hyperbolic space exist, such as geodesic hyperplanes that have been used in different machine learning works on hyperbolic geometry Ganea et al. (2018b); Cho et al. (2019); Shimizu et al. (2020). However, one advantage of horospherical learning is that the distance from a point to a horosphere admits the simple expression

$$
d ( x , H _ { p , a } ) = | B _ { p } ( x ) - a | ,\tag{20}
$$

which is analogous to the relation between Euclidean hyperplanes and scalar product. This expression is straightforward to compute and shows that the distance between two horospheres sharing the same prototype $p$ is constant. In particular, this distance is independent of the points at which it is evaluated, which provides a natural interpretation of horospherical classification.

## C.3 BUSEMANN VECTOR FIELDS

The change detection part of HyperFlow rely on the flow of a hyperbolic vector field. In a general manifold M, several possible approaches exist to define and parameterize such a vector field. A straightforward approach is to see the manifold M of dimension d as a subset of a Euclidean embedding space $\mathbb { R } ^ { \hat { n } }$ where $n \geq d ,$ and to naively project the vectors to their corresponding tangent space. However, this construction is not geometrically grounded and may prove inefficient.

We instead adopt the intrinsic approach proposed in Katsman et al. (2023). Let us assume there exists a parameterizable feature function $g _ { \theta } : \mathcal { M }  \mathbb { R }$ on the manifold. By considering K different parameter values $\theta _ { 1 } , . . . , \theta _ { K }$ , we can construct a map $f _ { \Theta } : \mathcal { M } \to \mathbb { R } ^ { K }$ by concatenating the corresponding feature functions:

$$
f _ { \Theta } = [ g _ { \theta _ { 1 } } , . . . , g _ { \theta _ { K } } ] ,\tag{21}
$$

with $\Theta = ( \theta _ { 1 } , \dots , \theta _ { K } )$ . We interpret this function as a coordinate system on $\mathcal { M } .$ , although it is not required to be either injective or surjective. At a point $x \in \mathcal { M }$ , its differential $D _ { x } f _ { \Theta }$ maps $T _ { x } { \mathcal { M } }$ to $\dot { T } _ { f _ { \Theta } ( x ) } \mathbb { R } ^ { K } \cong \mathbb { R } ^ { K }$ . The corresponding pullback operator $( D _ { x } f _ { \Theta } ) ^ { * }$ maps the dual space $( \mathbb { R } ^ { K } ) ^ { * }$ of $\mathbb { R } ^ { K }$ to the dual space $( T _ { x } { \mathcal { M } } ) ^ { * }$ of $T _ { x } { \mathcal { M } }$ . By identifying Euclidean spaces with their dual spaces, we can see the pullback operator as a map

$$
( D _ { x } f _ { \Theta } ) ^ { * } : \mathbb { R } ^ { K }  T _ { x } { \mathcal { M } } .\tag{22}
$$

It is expressed by the gradients of the different coordinate features, which together form what we can see as a set of elementary vector fields.

Following Katsman et al. (2023), a vector field can then be defined on M as

$$
V _ { \varTheta , \omega } : x \mapsto ( D _ { x } f _ { \varTheta } ) ^ { \ast } \Bigl ( n _ { \omega } ( f _ { \varTheta } ( x ) ) \Bigr ) = \sum _ { k = 1 } ^ { K } n _ { \omega } ( f _ { \varTheta } ( x ) ) _ { k } \nabla g _ { \theta _ { k } } ( x ) ,\tag{23}
$$

where $n _ { \omega } : \mathbb { R } ^ { K }  \mathbb { R } ^ { K }$ is any neural network and $\nabla$ is the Riemannian gradient. While $f _ { \Theta }$ feeds the neural network with some kind of coordinates, the outputs of this network serve as weights in a linear combination of the elementary vector fields. In the hyperbolic case, Busemann functions are natural candidates for feature functions, as proposed in Katsman et al. (2023).

Adaptation to Computer Vision Tasks. Under the above formulation, the vector field, when applied to the embeddings of the pixels of an image, does not take the spatial context into account. Consequently, the displacement of a pixel embedding is determined solely by its position in space, independently of the embeddings of neighboring pixels. In this work, since we argue that the spatial context provides valuable information, we instead replace the neural network $n _ { \omega } .$ , acting pixel-wise, by a neural network $N _ { \omega } : ( \mathbb { R } ^ { K } ) ^ { H \times W } \to ( \mathbb { R } ^ { K } ) ^ { H \times W }$ that takes the full image of Busemann features as input (in our case, a UNet). The Busemann functions $f _ { P }$ and their gradients $\nabla B _ { p _ { k } }$ are however still computed pixel-wise. This creates a vector field on $( \mathbb { B } _ { d } ) ^ { H \times W }$ , and is equivalent to applying the design defined by eq. (23) to $( \mathbb { B } _ { d } ) ^ { H \times W }$ as a product manifold, with the particular choice of weight-sharing feature functions $( g _ { \theta _ { k } , i , j } ) _ { i , j , k }$ defined by

$$
g _ { \theta _ { k } , i , j } : E \in ( \mathbb { B } _ { d } ) ^ { H \times W } \mapsto g _ { \theta _ { k } } ( E _ { i , j } ) \in \mathbb { R } ,\tag{24}
$$

for every $1 \leq i \leq H , 1 \leq j \leq W$ and $1 \leq k \leq K$

## D IMPLEMENTATION DETAILS

Segmentation. Our segmentation model is composed of a DeepLabV3 model, based on a ResNet101 backbone and pretrained on ImageNet, together with the horospherical classification head from Berg et al. (2024). We train it on 60 epochs with a batch size of 4 and a learning rate of 0.01. It is fed with RGB images of size $5 1 2 \times { \bar { 5 } } 1 2$ , obtained by randomly cropping the original images. We choose a hyperbolic embedding dimension of 2 in all our experiments, and we study the impact of the dimension in Supplementary Materials F.5. The prototypes from the horospherical head are optimized on the sphere, together with all the other weights, using a Riemannian stochastic gradient descent algorithm (Bonnabel, 2013).The weighting term $\beta$ in eq. (7) is set to $1 e - 5$ . During incremental steps, the new prototypes are initialized according to eq. (5), with $\epsilon _ { j } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ and dispersion $\sigma ^ { 2 } = 0 . 1$

Change detection. The change detection part of HyperFlow is trained on the embeddings produced by the segmentation model (which is, for this CD part, frozen) and learns the vector field introduced in section 3.5. As we choose to reuse the segmentation prototypes to define this vector field and to fix them, the only thing that needs to be learned is the network $n _ { \omega }$ that we implement with a UNet architecture. The model is trained with an ADAM optimizer, with learning rate $1 0 ^ { - 4 }$ batch size 4 on 20 epochs.

Incremental settings. As more incremental steps are considered, the segmentation model’s training is resumed for 60 more epochs with the same hyperparameters. Note that its architecture is slightly different as its (hyperbolic) head now contains new prototypes. The change detection model’s architecture is then redefined to take into account these new prototypes, which leads to a higher number of channels in the UNet, and is retrained from scratch.

For HiChange, classes are introduced progressively according to the incremental scheme summarized in table 6. The table reports only the classes introduced at each step; the full class set at step t includes all classes introduced in previous steps together with those added at step t. Newly introduced classes are added simultaneously rather than one after another. Class notation is consistent with fig. 5.

For SECOND and Hi-UCD-mini, no incremental setting is used. The tasks $T _ { 6 } ^ { 0 }$ and $T _ { 9 } ^ { 0 }$ correspond to the standard setting where all dataset classes are considered from the beginning.

Table 6: Incremental tasks for the HiChange dataset. The table summarizes the classes considered at each incremental step.  
Step Task Classes introduced   
0 $T _ { 4 } ^ { 0 }$ [1, 2, 3, 4]   
1 $T _ { 7 } ^ { \dot { 1 } }$ [31, 32, 33, 34]   
2 $T _ { 1 3 } ^ { \dot { 2 } }$ [10, 11, 12, 13, 14, 15, 16]   
3 $\overline { { T ^ { 3 } } } ^ { \mathrm { ~ } }$ $\mathit { I } _ { 1 6 } ^ { \mathrm { { ' } 0 } }$ [40, 41, 42, 43]   
4 $T _ { 2 0 } ^ { 4 }$ [20, 21, 22, 23, 24]   
5 $T _ { 2 2 } ^ { 5 }$ [321, 322, 323]   
6 $T _ { 3 2 } ^ { 6 }$ [110, 111, 112, 113, 114, 121, 122, 123, 141, 142, 143]

Semantic change detection maps and consistency between change and segmentation maps. Following established work in SCD, our formulation deliberately keeps (and evaluates) the change map and the semantic segmentation maps as separate outputs rather than combining them into a single from-to transition map. The latter would couple the errors of individual predictions without providing additional information. The notion of change consistency in our work should therefore be understood at the level of the incremental task rather than as a hard pixel-wise constraint. HyperFlow does not enforce change and semantic prediction to be consistent at every pixel. However, as the semantic taxonomy evolves, both semantic segmentation and change detector evolve to enable the change maps to describe changes within the new semantic classes considered.

## E ADDITIONAL QUALITATIVE RESULTS

This section presents additional qualitative results illustrating the behavior of HyperFlow across datasets.

## E.1 HICHANGE

For the HiChange dataset, we provide additional visualizations illustrating different aspects of the proposed approach. First, fig. 9 presents additional examples of segmentation prediction across incremental steps, together with the associated hyperbolic embeddings. These examples illustrate the capacity of HyperFlow to capture semantic information of increasing complexity as well as the evolution of the organization of the embedding space as the taxonomy is progressively refined.

![](images/d39376d3543587317a8c3f6b9fc8b858b96aa4a7d950dc4c37e47d5ed4f72d17.jpg)  
Figure 9: Qualitative examples on HiChange of segmentation across incremental steps and associ ated embedding space visualizations. For each step, we show the predicted segmentation together with the corresponding hyperbolic embeddings, illustrating how the embedding space evolves as the taxonomy is refined and new classes are introduced. The color of each embedding corresponds to the ground-truth semantic label.

Second, fig. 10 presents additional examples of semantic change detection for the second incremental step $( T _ { 1 5 } ^ { 2 } )$ ). For each pair of images, we show the predicted segmentation at both timestamps, the resulting change map, and the embeddings of the first and second image, as well as the predicted embeddings of the second image. These examples illustrate how pixels corresponding to changes exhibit larger displacements in the hyperbolic space compared to unchanged regions, enabling HyperFlow to detect changes in an unsupervised way.

Finally, fig. 11 compares both prediction and embedding spaces obtained under incremental and non-incremental settings. These visualizations provide an intuitive illustration of the effect discussed in the main paper: incremental refinement preserves the global hierarchical organization of the embedding space whereas directly training on the full fine-grained taxonomy tends to produce less structured representations. For the non incremental but hierarchical scenario, the hierarchical organization is also preserved. However, because the model must learn all fine-grained categories from scratch, the task is harder and the resulting performance is lower.

## E.2 SECOND

For the SECOND dataset, incremental scenarios are not possible. We therefore present qualitative examples of SCD only. In fig. 12, we visualize the predicted segmentation at both timesteps, the predicted change map and the embeddings of the first and second image, compared with the predicted embeddings of the second image.

## E.3 HI-UCD-MINI

Similarly, for the Hi-UCD-mini dataset, we provide qualitative examples of SCD in fig. 13. We show for each pair of images the segmentation and change predictions, as well as the associated embedding space visualizations.

## F ADDITIONAL EXPERIMENTS

## F.1 INFLUENCE AND CHOICE OF THE THRESHOLD

The CD part of HyperFlow, based on hyperbolic semantic flows, needs a threshold at inference to transform a continuous reconstruction error into a binary prediction. To obtain binary CD from the distance between the embeddings without relying on ground-truth labels, we require an unsupervised thresholding method. Otsu’s method (Otsu, 1979), although commonly used, is unsuitable here, as it assumes a bimodal histogram, whereas our distance distribution is dominated by the unchanged mode with a much smaller tail of change scores, as observed in fig. 14. Under this imbalance, Otsu does not provide a suitable threshold. We instead use the Triangle algorithm (Zack et al., 1977), which is designed for exactly this type of skewed distribution: it selects the threshold as the point of maximum perpendicular distance between the histogram and the line connecting its peak to its tail.

The triangle method, computed on the validation set, for HiChange and the base task $T _ { 4 } ^ { 0 }$ gives the value $d _ { \mathrm { t h r e s h } } = 4 . 2$ . We study the impact of this threshold in fig. 15. 4.2 appears to be a relevant choice, although not being the best one. While the performance depends on the choice of the threshold, we note that the range in which our model outperforms other unsupervised methods remains large. We therefore keep this value for all experiments involving HiChange. The threshold value is however adapted for the other two datasets, and when using SAM3 and AnySat as embeddings encoder, as the distribution of the change scores may vary significantly in those cases. Similarly, when considering the Euclidean vector fields, the thresholds are adapted to the range of change scores values encountered. The values retained are summarized in table 7.

## F.2 INFLUENCE OF THE BACKBONE ARCHITECTURE

All experiments presented in the main paper rely on a single backbone (Deep-LabV3 with a ResNet-101 backbone pre-trained on ImageNet) to ensure fair comparison across methods. We now investigate how the choice of backbone affects both segmentation and change detection performance. In particular, we consider a simpler UNet backbone. Results are provided in table 8. As expected, the performance obtained with the UNet backbone is lower than with DeepLabV3. This difference is mainly explained by the smaller model size (1.3M parameters compared to 58M), and the absence of pretraining. Consequently, segmentation performance decreases, which impacts CD and SCD, as good change detection relies on the quality of the underlying segmentation. Nevertheless, it can be observed that for $T _ { 4 } ^ { 0 }$ , the CD performance for the UNet backbone (F1= 14.5) remains simila to those of the unsupervised methods reported in table 1. This result indicates that HyperFlow still provides effective CD even when relying on a weaker segmentation model.

![](images/5032450b5acad78964085dc05c36cdf8917e84f9b79b2e52920374aba92d9925.jpg)  
Figure 10: Qualitative examples of SCD at task $T _ { 1 5 } ^ { 2 }$ . For each image pair, we show the predicted segmentation at both timesteps, the predicted change map, the embeddings of the two images and the predicted embeddings of the second image. Pixels corresponding to true semantic changes (in red) exhibit larger displacement in the embedding space, that are not captured by the learned flow and therefore not accounted for in the predicted embeddings.

![](images/43f6f69dda875766f761dd1ec59592493ab2567ad6be71c12af0cf278de673c8.jpg)  
Figure 11: Comparison between incremental and non-incremental scenarios. We visualize the segmentation and change predictions at (a) $T _ { 1 3 } ^ { 2 }$ and (b) $T _ { 3 2 } ^ { 6 } ,$ , along with corresponding embedding spaces. Incremental refinement preserves the hierarchical organization, enabling better prediction, whereas directly learning fine-grained categories leads to less structured embeddings and lower performances.

![](images/fe1570d7809d6c52d162b09fc4b74fc64e2c5010402e670ed887692dbd871f35.jpg)  
Figure 12: Qualitative examples on the SECOND dataset. For each image pair, we show segmentation and change prediction, and the associated embeddings.

![](images/93514d39333ef2b18a66035c0ad484865c2898df0eecc7dbd5333c9958aa4dfe.jpg)  
Figure 13: Qualitative examples on the Hi-UCD-mini dataset. For each image pair, we show segmentation and change prediction, and the associated embeddings.

Distribution of validation change scores  
![](images/44cb2c94a6aca4e9fc1659ba8f0bd1922f4bc2bd190b1f4f854ff2b778b00457.jpg)

Figure 14: Distribution of change scores and triangle threshold for the HiChange dataset and the base task $T _ { 4 } ^ { 0 }$  
![](images/04759a34c6529c1d122bbb83cfdcae77ed60f51a0f99f4ec2eeb4d91388429c4.jpg)  
Figure 15: Influence of $d _ { \mathrm { t h r e s h } }$ on CD performance. We present the F1 score of HyperFlow with respect to the threshold used, for HiChange and the base task $T _ { 4 } ^ { 0 }$ . Results are compared with the best unsupervised CD baseline, KPCAMNet.

## F.3 DO WE REALLY NEED TO USE THE SEGMENTATION PROTOTYPES?

The current formulation reuses the Busemann prototypes learned for segmentation as anchors for the vector field. An alternative is to use uniformly distributed prototypes, either learned or fixed. While this provides greater flexibility and allows HyperFlow to operate without segmentation-derived embeddings, which is particularly relevant for non-semantic CD tasks or when using general embeddings not obtained from segmentation pretext tasks, as in table 1 with SAM3 and AnySat, it may discard part of the semantic information encoded in the learned segmentation prototypes. Table 9 compares CD performance using prototypes derived from the segmentation task and prototypes independent of segmentation, both with and without prototype learning. While using non-segmentation prototypes leads to a performance drop, the resulting CD performance remains competitive, demonstrating that HyperFlow can also operate on prototypes not derived from a segmentation pretext task. Learning the prototypes partially closes the gap with segmentation-based prototypes, although not entirely. This can be explained by the fact that CD ultimately concerns changes in semantic content. Segmentation embeddings and prototypes therefore provide a natural basis for modeling this semantic flow.

Table 7: Thresholds values obtained from the triangle method. Unless stated otherwise, these values are used for all experiments.
<table><tr><td>Dataset</td><td>Embeddings</td><td>Hyperbolic/Euclidean</td><td>Threshold</td></tr><tr><td>HiChange</td><td>Segmentation</td><td>Hyp.</td><td>4.2</td></tr><tr><td>HiChange</td><td>Segmentation</td><td>Euc.</td><td>0.2</td></tr><tr><td>HiChange</td><td>Segmentation</td><td>Euc-Bus.</td><td>0.2</td></tr><tr><td>HiChange</td><td>SAM3</td><td>Hyp.</td><td>6.3</td></tr><tr><td>HiChange</td><td>AnySat</td><td>Hyp.</td><td>3.6</td></tr><tr><td>SECOND</td><td>Segmentation</td><td>Hyp.</td><td>7.0</td></tr><tr><td>SECOND</td><td>SAM3</td><td>Hyp.</td><td>6.0</td></tr><tr><td>SECOND</td><td>AnySat</td><td>Hyp.</td><td>5.4</td></tr><tr><td>Hi-UCD-mini</td><td>Segmentation</td><td>Hyp.</td><td>5.1</td></tr><tr><td>Hi-UCD-mini</td><td>SAM3</td><td>Hyp.</td><td>6.0</td></tr><tr><td>Hi-UCD-mini</td><td>AnySat</td><td>Hyp.</td><td>5.6</td></tr></table>

Table 8: Influence of the backbone architecture on the performances of HyperFlow.
<table><tr><td>Metric</td><td>Backbone</td><td> $T _ { 4 } ^ { 0 }$ </td><td> $T _ { 7 } ^ { 1 }$ </td><td> $T _ { 1 3 } ^ { 2 }$ </td><td> $T _ { 1 6 } ^ { 3 }$ </td><td> $T _ { 2 0 } ^ { 4 }$ </td><td> $T _ { 2 2 } ^ { 5 }$ </td><td> $T _ { 3 2 } ^ { 6 }$ </td></tr><tr><td>Segm. [mIoU]</td><td>DeepLabV3</td><td>62.0</td><td>56.2</td><td>33.7</td><td>28.4</td><td>23.5</td><td>20.7</td><td>16.1</td></tr><tr><td></td><td>ÜNet DeepLabV3</td><td>50.5 29.6</td><td>24.3 29.6</td><td>12.3 29.8</td><td>9.1 29.2</td><td>8.0 27.7</td><td>6.9 31.6</td><td>4.7 26.0</td></tr><tr><td>CD [F1]</td><td>UNet</td><td>14.5</td><td>16.3</td><td>17.1</td><td>17.0</td><td>17.2</td><td>18.8</td><td>17.3</td></tr><tr><td rowspan="2">SCD [SCS]</td><td>DeepLabV3</td><td>39.0</td><td>34.4</td><td>22.4</td><td>19.9</td><td>17.3</td><td>17.6</td><td>13.8</td></tr><tr><td>UNet</td><td>31.1</td><td>16.7</td><td>10.3</td><td>8.8</td><td>8.6</td><td>8.5</td><td>7.1</td></tr></table>

Table 9: Comparison of segmentation-based and independent prototypes for CD on the HiChange dataset, showing the impact of prototype initialization on HyperFlow performance.
<table><tr><td>Segmentation prototypes</td><td>Trained</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td>√</td><td>x</td><td>92.9</td><td>29.2</td><td>20.2</td><td>52.6</td></tr><tr><td>x</td><td>x</td><td>93.1</td><td>26.9</td><td>19.1</td><td>45.5</td></tr><tr><td>x</td><td>√</td><td>93.9</td><td>27.9</td><td>20.8</td><td>42.5</td></tr></table>

## F.4 DOES WORKING IN HYPERBOLIC SPACE HELP?

One of the main hypotheses of this work is that exploiting hyperbolic geometry can improve the CI-SCD task. While Hindel et al. (2024) demonstrated the benefits of hyperbolic representations for CISS on natural images, it remains unclear whether these advantages also extend to change detection (CD) in remote sensing imagery. To investigate this question, we consider two Euclidean counterparts to the proposed Busemann vector field framework. The first preserves the overall methodology while replacing the hyperbolic Busemann function with its Euclidean analogue, namely a scalar product. This variant therefore retains the same formulation and training procedure, differing only in the underlying geometry. The second removes the Busemann formulation entirely, as it is no longer required in Euclidean space, and instead trains a neural network to directly predict the next embeddings. We refer to these two variants as $E u c – B u s$ . and Euc, respectively, and compare their performance on the CD task with the proposed hyperbolic approach in table 10a.

We observe a clear superiority of the hyperbolic formulation. However, in our experiments, the threshold obtained using the triangle-based method for the Euclidean variants was not optimal. This can be attributed to the quite different distribution of embeddings in Euclidean space: compared to hyperbolic embeddings, the embeddings exhibit smaller pairwise distances, as those distances do not grow exponentially. To provide a fairer comparison and remove artifacts of thresholding behavior, we therefore manually selected the optimal threshold for both the Euc-Bus. and Euc formulations. Table 10b reports their performance using these thresholds, alongside that of the standard hyperbolic HyperFlow with its optimal threshold $( d _ { \mathrm { t h r e s h } } = 6 _ { \cdot }$ , see fig. 15). Although the performance gap is reduced, the overall trend remains unchanged: the hyperbolic version outperforms its Euclidean counterparts.

Table 10: Comparison of hyperbolic and Euclidean formulations for HyperFlow on the HiChange dataset and the base task $T _ { 4 } ^ { 0 }$ , using (a) triangle thresholds and (b) optimal thresholds.  
(a)
<table><tr><td>Space</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td>Hyp.</td><td>92.9</td><td>29.6</td><td>20.5</td><td>52.8</td></tr><tr><td>Euc-Bus.</td><td>82.1</td><td>18.9</td><td>10.8</td><td>74.7</td></tr><tr><td>Euc.</td><td>83.1</td><td>19.7</td><td>11.4</td><td>73.4</td></tr></table>

(b)
<table><tr><td>Space</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td>Hyp.</td><td>97.1</td><td>35.9</td><td>47.1</td><td>29.0</td></tr><tr><td>Euc-Bus.</td><td>94.4</td><td>34.9</td><td>25.8</td><td>53.8</td></tr><tr><td>Euc.</td><td>94.4</td><td>35.5</td><td>26.5</td><td>53.7</td></tr></table>

## F.5 EMBEDDING DIMENSION

We evaluate the impact of the embedding dimension d on performance for the base task $T _ { 4 } ^ { 0 }$ . Results are reported in table 11. Increasing the embedding dimension does not provide a clear improvement on both semantic segmentation and CD performance. The results fluctuate across dimensions, without showing a clear superiority. This behavior is likely influenced by the choice of the threshold that might not be optimal in other dimensions. In our experiments, we used the same threshold value $( d _ { \mathrm { t h r e s h } } = 4 . 2 )$ for all dimensions, while the optimal threshold can be expected to vary slightly with the embedding dimension.

Despite these results and the fact that higher dimension could potentially provide better performances, especially if we consider the SCD performance, we keep d = 2 in the main experiments for three reasons. First, this configuration already surpasses competing methods. Second, 2-dimensional embeddings enable direct visualization of the learned representations, which provides useful insights into the semantic structure and the learned flow. Third, it reduces the number of parameters in the model. While higher dimensions slightly improve the SCD (SCS) score, the gains remain limited and do not justify the additional parameters.

## F.6 FAILURE CASES

Figure 16 presents typical examples of failure cases. Two main sources of errors can be identified. The first is errors in the predicted segmentation. As illustrated by the first two examples, inaccuracies in the predicted land-cover propagate to the change map. This is consistent with the results discussed in section 5.1: the quality of the embeddings directly affects the quality of the resulting CD, and, for the segmentation embeddings used here, this quality is closely related to the accuracy of the land-cover segmentation. The second reason comes from errors in the ground-truth annotations, as observed in the last two examples.

## F.7 COMPUTATIONAL COMPLEXITY

Our model has 59.2M parameters (225.6 MB) and requires 276.7 GFLOPs per forward pass.

Table 11: Impact of the embedding dimension d on segmentation, CD, and SCD performance for HiChange and the base task $T _ { 4 } ^ { 0 }$
<table><tr><td>Dimension</td><td>Segm. [mIoU]</td><td>CD [F1]</td><td>SCD [SCS]</td></tr><tr><td>2</td><td>62.0</td><td>29.6</td><td>39.0</td></tr><tr><td>16</td><td>60.8</td><td>21.7</td><td>37.1</td></tr><tr><td>64</td><td>62.2</td><td>28.4</td><td>39.8</td></tr><tr><td>128</td><td>58.6</td><td>32.6</td><td>41.8</td></tr></table>

![](images/9cfec3c94ce8545f76f8ce06f8e9e29e5b3f410f8318a9242f0363ab3d4ec680.jpg)  
Figure 16: Failure cases of HyperFlow. The first two examples illustrate errors caused by inaccuracies in the predicted land-cover, which propagate to the change map. The last two illustrate cases where the ground-truth annotations present errors.