# Scale Matters: Adaptive Granularity Selection for Cross-Species 3D Plant Organ Segmentation

Carla Salazar<sup>1,2</sup> and Lazaros Nalpantidis<sup>1,2</sup>

<sup>1</sup> Technical University of Denmark (DTU), Kongens Lyngby, Denmark {ccsna,lanalpa}@dtu.dk

<sup>2</sup> Pioneer Centre for Artificial Intelligence, Copenhagen, Denmark

![](images/ffaadd0d2a1efecee24a6efb6d52ed1fac4bea8b04b5429491d5c54c614ac793.jpg)  
Fig. 1: The necessity of adaptive granularity in plant phenotyping. Cosine similarity (green: low, red: high) for leaves (top) and stems (bottom) between prototypes and Utonia features across diferent input scales for two example plants. A single fixed granularity fails to generalize across varying plant morphologies. The first plant requires a much finer granularity (scale=7) compared to the second (scale=1).

Abstract. Recent 3D foundation models provide powerful feature representations for point cloud learning by controlling spatial granularity. However, relying on a fixed spatial granularity severely limits generalization in applications like plant phenotyping, where organ morphology and size vary substantially across species and growth stages. To address this, we propose AGS-PlantSeg, a few-shot 3D plant organ segmentation method that leverages the frozen Utonia [22] foundation model combined with Adaptive Granularity Selection. By dynamically selecting the best granularity levels for each specific plant model, our method extracts optimized geometric features for a lightweight MLP segmentation head. Extensive experiments across PLANesT-3D [7], Pheno4D [12], and Crops3D [23] demonstrate that AGS-PlantSeg significantly improves cross-species generalization, achieving 88.9% average mIoU performance and outperforming fixed-granularity baselines by 2.5 mIoU points. Despite requiring minimal annotated data, our approach is highly competitive with fully supervised, plant-specific architectures. Project page: https://github.com/DTU-PAS/ags-plantseg

Keywords: 3D plant organ segmentation · adaptive granularity selection · cross-species generalization · few-shot learning · foundation model

## 1 Introduction

Computer vision is increasingly used for plant phenotyping as a tool to measure traits such as leaf area, fruit count, and organ morphology [2]. A key requirement for these measurements is plant organ segmentation, where organs such as leaves, stems, flowers, and fruits must be identified within a plant image or 3D scan. While 2D image-based methods are limited by occlusions and missing spatial information [15], 3D point cloud-based approaches provide geometric cues that motivate recent work on 3D plant organ segmentation [7, 15, 23].

Recent 3D foundation models have transformed point cloud learning by enabling strong feature representations across tasks and domains. Models such as Concerto [21] and Utonia [22] have shown that pretrained 3D encoders can generalize beyond a single dataset, moving seamlessly from robotic manipulation to large-scale outdoor scenes. In Utonia, this generalization is supported by the control of spatial granularity at which the model represents local structures. In many tasks, this granularity can be selected before deployment as the expected object size, scene scale, or point cloud density is relatively consistent.

However, this reliance on consistent spatial granularity presents a significant challenge in plant phenotyping, where organ size and morphology vary substantially across datasets, species, growth stages, and even between organs within the same plant. As a result, manually selecting a single fixed granularity is nontrivial: the granularity level that accurately represents stems and leaves for one species may not be suitable for another, especially when organs difer in size, density, or spatial arrangement. Similarly, even within the same species, the appropriate scale may change as plant organs shift in proportion across growth stages. As diferent granularities produce diferent feature representations, relying on a fixed choice limits generalization when plant morphology changes. Fig. 1 illustrates this efect by comparing the cosine similarity between Utonia features and learned stem and leaf prototypes across two distinct example plants: Ribes 07 and Rose 06. The preferred granularity difers significantly between the two, demonstrating that a plant’s specific morphology directly afects which granularity level best captures the spatial boundary between stem and leaf regions.

To address this limitation, we introduce AGS-PlantSeg, a plant organ segmentation method that combines the Utonia foundation model with Adaptive Granularity Selection. Instead of training a plant-specific segmentation network from scratch or manually choosing a fixed granularity level, we keep the Utonia encoder frozen and allow AGS-PlantSeg to automatically select the feature levels that best match each individual plant model. These selected levels are then used to extract frozen features that accurately reflect the morphology of the target plant. Finally, a lightweight MLP-based segmentation head operates on these optimized representations to classify plant organs. These design choices, endow AGS-PlantSeg with the following properties: 1 It remains competitive with fully supervised baselines while using frozen 3D foundation model features, 2 it can transfer to unseen species and datasets, and finally 3 its Adaptive Granularity Selection module improves segmentation performance compared with fixedgranularity alternatives.

The main contributions of this paper are:

– We propose AGS-PlantSeg, a few-shot 3D plant organ segmentation method that combines frozen 3D foundation model features with an Adaptive Granularity Selection (AGS) module.

We introduce a prototype-based granularity scoring strategy that selects suitable feature granularity levels during both training and inference, utilizing metrics for organ separation, within-organ compactness, and boundary consistency.

– We demonstrate that adaptive granularity selection improves cross-species generalization for plant organ segmentation, achieving performance competitive with fully supervised, plant-specific baselines without requiring retraining for every new plant model.

## 2 Related Work

3D Segmentation. 3D point cloud segmentation has evolved from early pointbased architectures such as PointNet [9] and PointNet++ [10] to more recent architectures with improved local feature aggregation, such as PointNeXt [11], and attention-based designs, such as Point Transformer variants [17–19]. These developments have also influenced 3D plant organ segmentation, where several methods train dedicated networks to segment organs such as leaves and stems at the semantic or instance level. For example, PSegNet [5] incorporates attention modules within a deep learning neural network for both semantic and instance segmentation. GCASSN [24] combines graph convolutional network with self-attention to improve feature learning on plant point clouds. Other methods build on established point cloud backbones such as PointNet++ [10] or Point-NeXt [11]. For example, Dong et al. [3] combine PointNeXt with Quickshift++ to increase the generalization ability of the model. These methods achieve strong performance in supervised settings, and some report improved generalization across plant species [3,24]. However, generalization remains challenging because plant morphology, organ size, and point cloud structure can vary substantially across species, growth stages, and acquisition settings.

3D Foundation Models. Recent work has started to explore pretrained 3D point cloud encoders that can provide transferable representations across different tasks and domains. Models such as Sonata [16] and Concerto [21] show that self-supervised pretraining can produce strong 3D representations for downstream tasks, reducing the need to learn task-specific features entirely from scratch. Utonia [22] extends this further, demonstrating cross-domain and crossscale generalization within a single model. Such pretrained encoders are of particular interest for 3D plant organ segmentation, where datasets are scarce and often small, and where morphology varies substantially across species, growth stages, and acquisition settings. However, using frozen 3D foundation models for plant organ segmentation introduces an additional challenge: the desired feature granularity may depend on the target structure. This issue is related to recent work in 3D part segmentation, where several methods have explored controllable or promptable segmentation granularity for object part segmentation [6, 14, 20]. These works indicate that part-level segmentation is not only a question of feature quality, but also of selecting an appropriate level of detail for the target task.

![](images/f7ad41c7a9bac9dfd0837a5c76125a801d77bb8dab2ef4bdaac9dc54bc980a3b.jpg)  
Fig. 2: Overview of AGS-PlantSeg. The method consists of five stages: (a) prototype computation at a candidate granularity level g, (b) training-time granularity selection using prototype-based scores, (c) segmentation head training on the selected granularity levels $G _ { \mathrm { t r a i n } } ^ { * }$ , (d) inference-time granularity selection using stored training prototypes and predicted labels, and (e) final prediction from the selected granularity levels $G _ { \mathrm { i n f e r } } ^ { * } .$ Orange and blue denote the two semantic organ labels. Feature spaces are visualized with PCA for illustration.

## 3 Methodology

The proposed AGS-PlantSeg selects the feature granularity that is most suitable for each plant scenario, where a scenario may difer in acquisition setup, plant species, growth stage, or organ morphology. Instead of assuming that one fixed granularity is optimal for all plants, we evaluate how well each granularity represents the semantic structure needed for organ segmentation.

Intuitively, a suitable granularity should satisfy three criteria. First, features belonging to diferent organs should be well separated, so that stem and leaf regions occupy distinct areas of the feature space. Second, clusters of features belonging to the same organ should be compact, so that points from the same class remain close to each other. Third, transition regions between organs should be treated carefully, since points near stem-leaf boundaries may contain mixed geometric or semantic information.

Granularity levels are selected during both training and inference using the Adaptive Granularity Selection (AGS) module of AGS-PlantSeg. The selection is implemented through a prototype-based scoring system, inspired by the use of prototypes in few-shot learning [13]. During training, labeled plants are evaluated at diferent candidate granularity levels by computing boundary-filtered prototypes and scoring how well each granularity separates organs, preserves compact organ representations, and handles boundary regions. The selected granularity levels are then used to train the segmentation head. During inference-time granularity selection, ground-truth labels are not available, so AGS-PlantSeg uses predicted labels from the trained segmentation head to compare test-plant features against the stored training prototype dictionary and select suitable granularity levels. This full process is summarized in Fig. 2.

## Utonia Features and Boundary-filtered Prototypes.

We build our method on features from the frozen Utonia model [22]. Given a plant point cloud and a granularity level g, Utonia extracts a feature vector $f _ { i } ^ { g }$ for each point i. Changing g changes the granularity of the resulting feature representation, allowing the model to emphasize diferent spatial structures of the plant.

To summarize each semantic organ in this feature space, we use class prototyping [13]. A prototype is the average feature vector of all points belonging to a given class. Since the feature representation depends on the granularity level, prototypes are also computed separately for each g. For example, the stem prototype at granularity level g is obtained by averaging the Utonia feature vectors of stem points extracted at g, while the leaf prototype is obtained from leaf points. These prototypes are later used to evaluate how well a candidate granularity level g separates organ features, preserves compact class representations, and handles boundary regions.

However, not all labeled points are equally reliable for defining an organ prototype. Points near the boundary between two organs may contain mixed information, especially around stem–leaf transitions. Including these points in the prototype computation can shift the prototype away from the interior representation of the organ. Therefore, before computing prototypes, we remove boundary points from the labeled plant.

A point is considered a boundary point if at least one of its K nearest neighbors belongs to a diferent semantic class. Only the remaining interior points are used to compute the class prototypes. This produces boundary-filtered prototypes that better represent the central feature distribution of each organ. In Fig. 2(a), this corresponds to applying the boundary mask, masking each label separately, and computing one prototype per semantic class at candidate granularity level g.

## Training-time Granularity Selection.

During the training-time granularity selection stage, corresponding to Fig. 2(b), ground-truth labels are available, so each candidate granularity level g can be

evaluated directly against the criteria introduced above. The first criterion requires features from diferent organs to be well separated. We measure this using the inter-class distance between class prototypes:

$$
D _ { \mathrm { i n t e r } } ^ { g } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( c , c ^ { \prime } ) \in \mathcal { P } } d _ { \cos } \left( p _ { c } ^ { g } , p _ { c ^ { \prime } } ^ { g } \right) ,\tag{1}
$$

where $\mathcal { P }$ is the set of class pairs, $p _ { c } ^ { g }$ is the prototype of class c at granularity level $^ { g , }$ and $d _ { \mathrm { c o s } }$ denotes cosine distance.

The second criterion requires clusters of features from the same organ to remain compact. We measure this using the intra-class distance between interior point features and their corresponding class prototype:

$$
D _ { \mathrm { i n t r a } } ^ { g } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \frac { 1 } { | \mathscr { T } _ { c } | } \sum _ { i \in \mathscr { T } _ { c } } d _ { \cos } \left( f _ { i } ^ { g } , p _ { c } ^ { g } \right) ,\tag{2}
$$

where $\mathcal { C }$ is the set of semantic classes, $\mathcal { T } _ { c }$ is the set of interior points with label $c ,$ and $f _ { i } ^ { g }$ is the feature vector of point i at granularity level $g .$

A good granularity level should maximize inter-class distance while minimizing intra-class distance. Inspired by Fisher-style class separability [4], where discriminative representations should have high between-class separation and low within-class variation, we define the training discriminability score as:

$$
S _ { \mathrm { d i s c } } ^ { g } = \frac { D _ { \mathrm { i n t e r } } ^ { g } } { D _ { \mathrm { i n t r a } } ^ { g } + \epsilon } ,\tag{3}
$$

where ϵ is a small constant for numerical stability. Higher values of $S _ { \mathrm { d i s c } } ^ { g }$ indicate that the feature space better separates organs while preserving compact organ representations.

The third criterion concerns boundary regions, where features may contain mixed information from neighboring organs. To evaluate whether these points remain consistent with their semantic class, we also compute an intra-class boundary distance from boundary points to their corresponding class prototype:

$$
D _ { \mathrm { b o u n d a r y } } ^ { g , c } = \frac { 1 } { | \mathcal { B } _ { c } | } \sum _ { i \in \mathcal { B } _ { c } } d _ { \mathrm { c o s } } \left( f _ { i } ^ { g } , p _ { c } ^ { g } \right)\tag{4}
$$

Here, $\boldsymbol { B } _ { c }$ is the set of boundary points with ground-truth label c. Since lower distances indicate stronger consistency with the class prototype, we define the boundary score as:

$$
S _ { \mathrm { b o u n d a r y } } ^ { g , c } = 1 - D _ { \mathrm { b o u n d a r y } } ^ { g , c }\tag{5}
$$

Thus, in our binary setting with classes A and $B ,$ each candidate granularity level is assigned three training-time scores. AGS selects the top-ranked granularity for each score. If multiple scores select the same granularity, it is retained only once. The corresponding class prototypes are stored in a training prototype dictionary D, which is later used during inference:

$$
\mathbf { s } _ { \mathrm { t r a i n } } ^ { g } = \left[ S _ { \mathrm { d i s c } } ^ { g } , \quad S _ { \mathrm { b o u n d a r y } } ^ { g , A } , \quad S _ { \mathrm { b o u n d a r y } } ^ { g , B } \right]\tag{6}
$$

$$
G _ { \mathrm { t r a i n } } ^ { * } = \{ g _ { \mathrm { d i s c } } ^ { * } , \quad g _ { A } ^ { * } , \quad g _ { B } ^ { * } \} .\tag{7}
$$

## Segmentation Head Training.

In the segmentation head training stage, corresponding to Fig. $2 ( \mathrm { c ) }$ , Utonia features are extracted at the selected training granularity levels $G _ { \mathrm { t r a i n } } ^ { * }$ . Each point-wise feature vector extracted at a selected granularity level is treated as an individual training sample for the shared MLP, paired with the corresponding ground-truth organ label. We keep the segmentation head small so that the evaluation focuses on the selected Utonia features and the AGS module, rather than on model capacity.

## Inference-time Granularity Selection.

The inference-time scoring procedure corresponds to Fig. 2(d). At inference time, ground-truth labels are not available. Therefore, AGS-PlantSeg cannot compute ground-truth prototypes for the test plant or directly evaluate inter-class separation as in training. Instead, candidate granularity levels are evaluated against a stored training prototype dictionary. This dictionary contains the class prototypes computed from the training plants and summarizes the feature distributions on which the segmentation head was trained. We therefore select granularity levels that produce test-plant features similar to this training feature structure.

At inference time, the stored prototype dictionary D is used to evaluate candidate granularity levels. Each entry $d \in \mathcal { D }$ contains the class prototypes $\tilde { p } _ { d , c }$ from one training plant. For a candidate granularity level $^ { g , }$ the segmentation head first predicts labels $\hat { y } _ { i } ^ { g }$ for each point. These predicted labels are used to define predicted interior points $\hat { \mathcal { T } } _ { c } ^ { g }$ and predicted boundary points $\hat { B } _ { c } ^ { g }$ for each class c.

To approximate the intra-class compactness criterion at inference time, we measure how close predicted interior-point features are to the stored training prototypes. Since the test plant does not need to match every training plant, we compute this distance against each stored prototype pair and keep the best match.

$$
D _ { \mathrm { i n t e r i o r } } ^ { g , d } = \frac { 1 } { \vert \mathcal { C } \vert } \sum _ { c \in \mathcal { C } } \frac { 1 } { \vert \hat { Z } _ { c } ^ { g } \vert } \sum _ { i \in \hat { Z } _ { c } ^ { g } } d _ { \cos } \left( f _ { i } ^ { g } , \tilde { p } _ { d , c } \right) .\tag{8}
$$

$$
S _ { \mathrm { i n t e r i o r } } ^ { g } = \operatorname* { m a x } _ { d \in \mathcal { D } } \left( 1 - D _ { \mathrm { i n t e r i o r } } ^ { g , d } \right) .\tag{9}
$$

Higher values of $S _ { \mathrm { i n t e r i o r } } ^ { g }$ indicate that the candidate granularity level $g$ produces interior features that are close to at least one stored training representation. This is used as an inference-time proxy for selecting a granularity level compatible with the feature space on which the segmentation head was trained.

Following the same boundary-consistency criterion used during training, we evaluate whether predicted boundary points remain close to the prototype dictionary for their predicted class. For each class $c ,$ we compare predicted boundarypoint features to the corresponding class prototypes in the dictionary and keep the best matching score:

$$
D _ { \mathrm { b o u n d a r y } } ^ { g , c , d } = \frac { 1 } { \lvert \hat { B } _ { c } ^ { g } \rvert } \sum _ { i \in \hat { B } _ { c } ^ { g } } d _ { \cos } \left( f _ { i } ^ { g } , \tilde { p } _ { d , c } \right) .\tag{10}
$$

$$
S _ { \mathrm { b o u n d a r y } } ^ { g , c } = \operatorname* { m a x } _ { d \in \mathcal { D } } \Big ( 1 - D _ { \mathrm { b o u n d a r y } } ^ { g , c , d } \Big ) .\tag{11}
$$

Thus, each candidate granularity level is assigned three inference-time scores. As in training-time, AGS selects the top-ranked granularity level for each score:

$$
\begin{array} { r } { \mathbf { s } _ { \mathrm { i n f e r } } ^ { g } = \left[ S _ { \mathrm { i n t e r i o r } } ^ { g } , \quad S _ { \mathrm { b o u n d a r y } } ^ { g , A } , \quad S _ { \mathrm { b o u n d a r y } } ^ { g , B } \right] . } \end{array}\tag{12}
$$

$$
G _ { \mathrm { i n f e r } } ^ { * } = \left\{ g _ { \mathrm { i n t e r i o r } } ^ { * } , \quad g _ { A } ^ { * } , \quad g _ { B } ^ { * } \right\} .\tag{13}
$$

Prediction.

As shown in Fig. $2 ( \mathrm { e } )$ , the trained MLP is applied to Utonia features extracted at the selected inference granularity levels $G _ { \mathrm { i n f e r } } ^ { * }$ . For each point i, the MLP outputs a class probability distribution $p _ { i , g , c }$ for each selected granularity level g and class c:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c } \sum _ { g \in G _ { \mathrm { i n f e r } } ^ { * } } w _ { i , g } p _ { i , g , c } ,\tag{14}
$$

where $w _ { i , g }$ is the normalized confidence weight for point i at granularity level $g ,$ computed from the maximum predicted class probability at that level.

## 4 Experimental Setup

## 4.1 Dataset

In this paper, we use three public datasets: PLANesT-3D [7], Crops3D [23] and Pheno4D [12]. Our approach has only been trained on PLANesT-3D data, using Crops3D and Pheno4D exclusively as test sets to evaluate generalization of our model to unseen species and datasets. This protocol is applied across all experiments and ablations.

The PLANesT-3D [7] is formed by point clouds of 10 pepper plants, 14 ribes plants and 10 rose plants, annotated by leaf and stem. This dataset provides a confidence score per point and, following the original protocol, points with confidence below 6 are removed. We evaluate three training settings. The fullsplit is the oficial PLANesT-3D split, where pepper (01, 03, 07), ribes (03, 10, 11, 14), and rose (01, 03, 09) are used for testing, and the remaining are used for training, corresponding to approximately a 70-30 split. In the few-shot setting, we chose one plant of each species for training: pepper 02, ribes 02 and rose 02.

In the one-shot setting, we train only on pepper 02 and test on the remaining pepper plants as well as all ribes and rose plants.

From Crops3D [23], we use the 3D scans of tomato and potato plants as an additional test set. The dataset contains 83 tomato and 118 potato plants annotated with stem and leaf labels. For the tomato subset, we exclude 10 scans that include fruit annotations, as the training data does not contain a fruit class.

From Pheno4D [12], we used their 77 annotated 3D point clouds of 7 tomato plants scanned daily during a 3 week period. This dataset is annotated by soil, stem and leaf, however, we remove the soil data for these experiments as the training data only contains stem and leaf.

## 4.2 Implementation Details

All point clouds are downsampled to 200K points to reduce memory consumption. During feature extraction, Utonia is kept frozen and features are computed at the granularity levels selected by our Adaptive Granularity Selection module. Following the oficial Utonia recommendation for object-level inputs [8], we apply coordinate normalization and a scale transformation. Coordinate normalization is used in all experiments to bring point clouds from diferent datasets into a comparable coordinate range, since the original scans may difer substantially in units or spatial scale. The scale transform is then used as the granularity parameter searched by our Adaptive Granularity Selection.

The granularity search is initialized at a scale of 4.0 and evaluates candidate scales with a step size of 0.5 within the range [1.0, 12.0]. Candidate scales are ranked using the proposed scoring system. We compute the prototypes excluding boundary points, where boundary detection is performed using a k-nearestneighbor filter with k = 1000.

The extracted multi-scale features are used to train a MLP-based lightweight segmentation head. For each training plant, the point-wise features are split into 80% training and 20% validation sets using stratified sampling to preserve the class distribution. For the full-split setting, storing features for every point would exceed GPU memory. We therefore randomly sample 50K points from each 200Kpoint cloud after Utonia feature extraction, reducing memory requirements. For few-shot and one-shot splits, we maintain all extracted points without subsampling.

The segmentation head is a two-layer MLP with a 256-dimensional hidden layer, trained on frozen Utonia features using Adam with a learning rate of $1 0 ^ { - 3 }$ batch size 2048, and cross-entropy loss. Training runs for at most 200 epochs with early stopping after 15 epochs without validation improvement, and uses dropout of 0.3. All experiments were conducted on a single NVIDIA GeForce RTX 5090.

## 4.3 Evaluation Metrics

We report mean Intersection over Union (mIoU) and Accuracy (Acc). The mIoU is computed as the average IoU over all semantic classes, while Accuracy is

computed as the fraction of correctly classified points over the total number of evaluated points.

## 5 Experiments and Results

We conduct experiments to support our claimed properties of AGS-PlantSeg, as stated in Sec. 1: 1 We first show that AGS-PlantSeg remains competitive with fully supervised baselines while using frozen 3D foundation model features. 2 We then evaluate its transfer to unseen species and datasets. 3 Finally, we show that Adaptive Granularity Selection improves segmentation performance compared with fixed-granularity alternatives.

## 5.1 Baseline Comparison 1

We compare AGS-PlantSeg against PointNet++ [10], RoseSegNet [15], and SP-LSCnet [7], as reported on the oficial PLANesT-3D split [7]. GCASSN [24] uses a diferent 80/20 split and preprocessing protocol and is included for reference. Since source code is unavailable, we report published results.

Table 1 shows that in the full-split setting, our method achieves a mIoU of 96.3%. This matches the best reported average mIoU while obtaining the highest average accuracy among the compared methods. In particular, our method performs strongly on pepper and ribes, reaching 96.9% and 97.5% mIoU, respectively. The few-shot setting remains close to the full-split setting. When trained with only one plant per species, our method obtains 94.9% average mIoU, compared with 96.3% in the full-split setting. This suggests that the frozen Utonia features provide a strong representation for plant organ segmentation even with limited annotated data. In the one-shot setting, performance remains high for pepper, which is included in training, and for ribes, which is unseen during training. This indicates that the frozen Utonia features can support cross-species generalization with limited supervision. However, performance drops on rose, reducing the average mIoU to 93.3%. Since rose is a more challenging morphology, this suggests that additional fine-tuning may still be needed for dificult species.

## 5.2 Generalization to Unseen Plant Species and Datasets 2

Figure 3 and Table 2 show the generalization performance on unseen species and datasets. Overall, the proposed method transfers well beyond the plants used for supervision. The full, few-shot, and one-shot models achieve comparable average performance, reaching 84.7%, 82.9%, and 82.5% mIoU, respectively. This suggests that the frozen Utonia features already provide a transferable representation for organ segmentation, so only limited annotated data is needed to adapt the classifier to the task.

On Pheno4D tomato, GCASSN reported 60.8% mIoU when trained on pepper and tested on tomato. In comparison, our full-split model reaches 86.0% mIoU, and the few-shot model reaches 83.7% mIoU. This suggests that frozen foundation features, combined with scale selection, provide a strong cross-species generalization. Interestingly, the one-shot model achieves the highest performance on potato with an mIoU of 85.9%, despite being trained on a single pepper plant. However, this result is not uniform across plant species: performance is strong on Ribes and Crops3D potato, but lower on Rose and tomato plants. This suggests that one-shot supervision can work very well when the target morphology is close to the training example, but is less stable across larger morphological shifts.

Table 1: Comparison with baseline methods on PLANesT-3D (%). Point-Net++, RoseSegNet, and SP-LSCnet are reported from PLANesT-3D [7], using the same 70/30 plant-level split as our full setting. GCASSN [24] uses a diferent 80/20 split and preprocessing protocol. Best result is shown in bold and second-best is underlined.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Supervision</td><td colspan="2">Pepper</td><td colspan="2">Rose</td><td colspan="2">Ribes</td><td colspan="2">Average</td><td rowspan="3">Rel. Training</td></tr><tr><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td>Speed</td></tr><tr><td>PointNet++ [7]</td><td>Full</td><td>97.9</td><td>94.3</td><td>96.7</td><td>89.8</td><td>98.4</td><td>94.2</td><td>97.7</td><td>92.8</td><td></td></tr><tr><td>RoseSegNet [7]</td><td>Full</td><td>98.3</td><td>95.5</td><td>97.4</td><td>92.0</td><td>98.6</td><td>95.1</td><td>98.1</td><td>94.2</td><td>=</td></tr><tr><td>SP-LSCnet [7]</td><td>Full</td><td>98.1</td><td>95.0</td><td>97.1</td><td>89.8</td><td>98.3</td><td>94.5</td><td>97.8</td><td>93.1</td><td>I</td></tr><tr><td>GCASSN [24]</td><td>Full</td><td>95.6</td><td>97.1</td><td>92.8</td><td>95.4</td><td>93.7</td><td>96.5</td><td>94.0</td><td>96.3</td><td>=</td></tr><tr><td>Ours</td><td>Full</td><td>98.8</td><td>96.9</td><td>98.5</td><td>94.4</td><td>99.2</td><td>97.5</td><td>98.8 96.3</td><td></td><td>1.0×</td></tr><tr><td>Ours</td><td>Few-Shot</td><td>99.0</td><td>97.0</td><td>97.9</td><td>92.6</td><td>99.0</td><td>95.2</td><td>98.6</td><td>94.9</td><td>2.0×</td></tr><tr><td>Ours</td><td>One-Shot</td><td>98.9</td><td>96.9</td><td>96.8</td><td>88.0</td><td>99.0</td><td>95.0</td><td>98.2</td><td>93.3</td><td>6.0×</td></tr></table>

Table 2: Generalization to unseen plant species and datasets (%). AGS-PlantSeg is evaluated with full, few-shot, and one-shot supervision. Baseline results are reported as published in the corresponding papers. Best result is shown in bold and second-best is underlined.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Supervision</td><td rowspan="2">Pheno4D Tomato</td><td colspan="4">Crops3D</td><td colspan="2">Avg</td></tr><tr><td></td><td>Tomato</td><td>Potato</td><td></td><td></td><td></td></tr><tr><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td></tr><tr><td>GCASSN [24]</td><td>Full</td><td>91.5</td><td>60.8</td><td>- 一</td><td>-</td><td>-</td><td></td><td>- 一</td></tr><tr><td>Ours</td><td>Full</td><td>97.1</td><td>86.0</td><td>94.3 82.6</td><td>97.2</td><td>85.6</td><td>96.2</td><td>84.7</td></tr><tr><td>Ours</td><td>Few-Shot</td><td>96.6</td><td>83.7</td><td>93.6 81.3</td><td>96.7</td><td>83.6</td><td>95.6</td><td>82.9</td></tr><tr><td>Ours</td><td>One-Shot</td><td>94.0</td><td>81.4</td><td>93.2 80.2</td><td>97.5</td><td>85.9</td><td>94.9</td><td>82.5</td></tr></table>

Qualitatively, Figure 3 shows predictions and ground-truth labels for Pheno4D tomato and Crops3D tomato and potato. While some local errors remain visible, the predicted organ structure is largely preserved across unseen datasets.

## 5.3 Ablations 3

We conduct ablation studies in the few-shot setting to analyze the contribution of the AGS module. The ablations are divided into two parts. First, we compare

Point Cloud

Prediction

Ground Truth

![](images/193be2e29aa69c3c3d9d9607bc4d1459956e41fa094378ab8018f30e86960096.jpg)  
Fig. 3: Qualitative generalization results on unseen species and datasets. Predictions are shown alongside ground truth for Pheno4D tomato and Crops3D tomato and potato. Orange denotes leaves and blue denotes stems.

AGS against fixed-granularity alternatives, where Utonia features are extracted at a single fixed granularity for both training and inference. These results are shown in Table 3. Second, we evaluate diferent components of AGS, including boundary filtering, granularity scoring, and confidence-weighted prediction. These results are shown in Table 4.

## Fixed-granularity Alternatives

We compare AGS against the common fixed-granularity setting, where the same MLP segmentation head is trained using Utonia features extracted at a single scale transform. We evaluate seven fixed scales in Table 3, ranging from 1.0 to 7.0, to sample the granularity range densely enough to capture the variation in plant morphology across species and datasets. This comparison isolates the efect of adaptive granularity selection from the segmentation architecture.

In contrast, AGS does not commit to a single global granularity. While it does not achieve the best mIoU for every individual species, it ranks first or second across all evaluated species and datasets. This consistency leads to the highest average performance, with 88.9% mIoU. The results show that adaptive granularity selection improves robustness over fixed-granularity feature extraction, while the cases where a fixed granularity still performs better indicate that our granularity selection strategy could be further improved.

Table 3: Comparison with fixed-granularity alternatives in the few-shot setting. Fixed baselines use a single Utonia scale for both training and inference, while AGS selects granularity levels adaptively. Best result is shown in bold and second-best is underlined.
<table><tr><td rowspan="3">Scale</td><td colspan="6">PLANesT-3D</td><td colspan="2">Pheno4D</td><td colspan="4">Crops3D</td><td rowspan="2" colspan="2">Avg</td></tr><tr><td colspan="2">Pepper</td><td colspan="2">Rose</td><td colspan="2">Ribes</td><td colspan="2">Tomato</td><td colspan="2">Tomato</td><td colspan="2">Potato</td></tr><tr><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td></td><td>Acc mIoU</td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td><td>Acc mIoU</td><td></td></tr><tr><td>1.0</td><td>97.4</td><td>92.7</td><td>96.1</td><td>86.7</td><td>97.3</td><td>87.7</td><td>93.8</td><td>70.6</td><td>92.0</td><td>77.2</td><td>97.1</td><td>84.8</td><td>95.6</td><td>83.3</td></tr><tr><td>2.0</td><td>98.3</td><td>95.2</td><td>97.5</td><td>91.0</td><td>98.3</td><td>92.3</td><td>89.5</td><td>74.8</td><td>93.8</td><td>81.2</td><td>95.7</td><td>79.0</td><td>95.5</td><td>85.6</td></tr><tr><td>3.0</td><td>98.7</td><td>96.4</td><td>97.4</td><td>90.7</td><td>98.8</td><td>94.5</td><td>95.1</td><td>83.4</td><td>91.1</td><td>76.4</td><td>94.3</td><td>77.2</td><td>95.9</td><td>86.4</td></tr><tr><td>4.0</td><td>98.8</td><td>96.8</td><td>97.6</td><td>91.5</td><td>98.9</td><td>94.9</td><td>95.6</td><td>83.0</td><td>91.1</td><td>76.5</td><td>92.0</td><td>71.3</td><td>95.7</td><td>85.7</td></tr><tr><td>5.0</td><td>98.8</td><td>96.7</td><td>98.0</td><td>93.0</td><td>98.6</td><td>93.3</td><td>91.7</td><td>75.3</td><td>88.5</td><td>72.2</td><td>90.4</td><td>68.2</td><td>94.3</td><td>83.1</td></tr><tr><td>6.0</td><td>98.9</td><td>97.0</td><td>97.8</td><td>91.8</td><td>99.1</td><td>95.8</td><td>94.1</td><td>74.6</td><td>89.7</td><td>73.0</td><td>94.2</td><td>75.5</td><td>95.6</td><td>84.6</td></tr><tr><td>7.0</td><td>98.9</td><td>96.8</td><td>97.7</td><td>91.3</td><td>98.9</td><td>94.7</td><td>92.8</td><td>70.0</td><td>90.5</td><td>73.3</td><td>94.2</td><td>76.0</td><td>95.5</td><td>83.7</td></tr><tr><td>Adaptive (ours)</td><td>99.0 97.0</td><td></td><td>97.9</td><td>92.6</td><td>99.0</td><td>95.2</td><td>96.6</td><td>83.7</td><td>93.6</td><td>81.3</td><td>96.7</td><td>83.6</td><td>97.1</td><td>88.9</td></tr></table>

Table 4: Ablation of AGS components in the few-shot setting. BFP denotes boundary-filtered prototype computation. $S _ { \mathrm { b o u n d a r y } } ^ { g , c }$ indicates whether boundary-aware granularity scores are used. CWA denotes confidence-weighted aggregation at inference time. Best result is shown in bold and second-best is underlined.
<table><tr><td>Method</td><td>BFP  $S _ { \mathrm { b o u n d a r y } } ^ { g , c }$ </td><td>CWA Avg. mIoU</td></tr><tr><td rowspan="3">Ablation 1 Ablation 2 Ablation 3</td><td></td><td>87.3</td></tr><tr><td>√</td><td>86.8</td></tr><tr><td>√ √</td><td>88.3</td></tr><tr><td>Ablation 4</td><td>√</td><td>88.5</td></tr><tr><td>Ours</td><td>√ √</td><td>√ 88.9</td></tr></table>

## AGS Component Ablation

Table 4 evaluates three components of AGS in the few-shot setting. BFP denotes boundary-filtered prototype computation. When BFP is disabled, class prototypes are computed using all points; when enabled, points near organ transitions are removed before prototype computation. $S _ { \mathrm { b o u n d a r y } } ^ { g , c }$ denotes the boundary-aware granularity score. When disabled, granularity selection uses only the internal score based on inter-class separation and intra-class compactness; when enabled, separate class-wise boundary scores are also used. CWA denotes confidence-weighted aggregation, where the selected granularity predictions are combined using the average softmax confidence of the MLP. The results show that the components are most efective when used together. BFP alone does not improve performance, because boundary points are removed from prototype computation but are not explicitly evaluated during granularity selection. Adding $\bar { S } _ { \mathrm { b o u n d a r y } } ^ { g , c }$ improves the average mIoU, and combining it with BFP improves it further. The full AGS pipeline, using BFP, $S _ { \mathrm { b o u n d a r y } } ^ { g , c }$ , and CWA, achieves the best average mIoU of 88.9%.

## 5.4 Runtime and Memory Footprint

We also report the runtime and memory footprint of our AGS module. In our current implementation, AGS requires roughly 15 s per plant during both training and inference. This includes feature extraction and granularity selection, and could likely be reduced through a more eficient search strategy and parallelization. In the full-split setting, AGS takes approximately 6 min, followed by 15 min for MLP training, resulting in a total training time of about 21 min. In the few-shot setting, AGS takes approximately 45 s and MLP training takes about 10 min, for a total of roughly 11 min (resulting in a 2× relative speedup). The one-shot setting is substantially cheaper, requiring approximately 3.5 min in total (resulting in a 6× relative speedup). The main practical diference is memory usage. In our implementation, the full-split setting exceeded the available memory on an RTX 5090, requiring heavy subsampling of the training features. In contrast, the few-shot and one-shot settings use fewer annotated plants and therefore have a much smaller memory footprint.

## 6 Conclusion

We propose AGS-PlantSeg, a few-shot 3D plant organ segmentation method that leverages the frozen Utonia model through Adaptive Granularity Selection. Instead of relying on a single fixed granularity, AGS selects suitable granularity levels for each plant using internal class separability and boundary-aware scores, and applies them with a lightweight MLP segmentation head. The results show that the best fixed granularity level varies across species and datasets. AGS improves cross-species robustness and achieves the best average performance, reaching 88.9% mIoU, and outperforming the best fixed-granularity baseline by 2.5 mIoU points. The few-shot model remains close to the full-split setting while requiring substantially fewer annotated plants, suggesting that frozen 3D foundation features can reduce annotation requirements for plant organ segmentation.

Limitations and Future Work. Although AGS is not always optimal for every individual species, its strong average performance shows that adaptive granularity selection is a promising direction for generalizable 3D plant phenotyping. Future work could investigate bigger segmentation heads, stronger training-time augmentation, and more eficient granularity search and scoring strategies. AGS could also be extended to a broader range of plant morphologies, including crops such as wheat and rice, and to additional semantic classes, such as fruits and soil. This work focuses specifically on evaluating whether adaptive granularity selection improves generalization across plant species and supervision settings. Future work could include a broader comparison with state-of-the-art few-shot segmentation approaches, particularly prototype-based methods such as COSeg [1], as well as an analysis of how the selected granularity levels vary with species, plant morphology, and growth stage.

## Acknowledgements

This work has been supported by the Novo Nordisk Foundation under grant agreement No. NNF25SA0104538 (Robotic Intercropping).

## References

1. An, Z., Sun, G., Liu, Y., Liu, F., Wu, Z., Wang, D., Van Gool, L., Belongie, S.: Rethinking few-shot 3d point cloud semantic segmentation. In: CVPR. pp. 3996– 4006 (2024)

2. Das Choudhury, S., Samal, A., Awada, T.: Leveraging image analysis for highthroughput plant phenotyping. Frontiers in plant science 10, 508 (2019)

3. Dong, S., Fan, X., Li, X., Liang, Y., Zhang, M., Yao, W., Yang, X., Wang, Z.: Automatic 3d plant organ instance segmentation method based on pointnext and quickshift++. Plant Phenomics 7(3), 100065 (2025). https://doi.org/10.1016/j. plaphe.2025.100065, https://www.sciencedirect.com/science/article/pii/ S2643651525000718

4. Fisher, R.A.: The use of multiple measurements in taxonomic problems. Annals of Eugenics 7(2), 179–188 (1936). https://doi.org/10.1111/j.1469- 1809.1936.tb02137.x, https://onlinelibrary.wiley.com/doi/abs/10.1111/ j.1469-1809.1936.tb02137.x

5. Li, D., Li, J., Xiang, S., Pan, A.: Psegnet: Simultaneous semantic and instance segmentation for point clouds of plants. Plant Phenomics 2022, 9787643 (2022). https://doi.org/10.34133/2022/9787643, https://www.sciencedirect.com/ science/article/pii/S2643651524001006

6. Liu, M., Uy, M.A., Xiang, D., Su, H., Fidler, S., Sharp, N., Gao, J.: Partfield: Learning 3d feature fields for part segmentation and beyond. In: ICCV (2025)

7. Mertoğlu, K., Şalk, Y., Sarıkaya, S.K., Turgut, K., Evrenesoğlu, Y., Çevikalp, H., Gerek, Ö.N., Dutağacı, H., Rousseau, D.: Planest-3d: A new annotated dataset for segmentation of 3d plant point clouds (2025), https://arxiv.org/abs/2407. 21150

8. Pointcept: Utonia: Oficial github repository. https://github.com/Pointcept/ Utonia (2026), accessed: 2026-07-10

9. Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation. CoRR abs/1612.00593 (2016), http://arxiv. org/abs/1612.00593

10. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. CoRR abs/1706.02413 (2017), http: //arxiv.org/abs/1706.02413

11. Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H., Elhoseiny, M., Ghanem, B.: Pointnext: Revisiting pointnet++ with improved training and scaling strategies. In: NeurIPS (2022)

12. Schunck, D., Magistri, F., Rosu, R.A., Cornelißen, A., Chebrolu, N., Paulus, S., Léon, J., Behnke, S., Stachniss, C., Kuhlmann, H., Klingbeil, L.: Pheno4d: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis. PLoS ONE 16 (2021), https://api. semanticscholar.org/CorpusID:237214002

13. Snell, J., Swersky, K., Zemel, R.S.: Prototypical networks for few-shot learning. CoRR abs/1703.05175 (2017), http://arxiv.org/abs/1703.05175

14. Su, H., Huang, T., Wan, Z., Wu, X., Zuo, W.: S<sup>2</sup>AM3D: Scale-controllable part segmentation of 3d point clouds. In: CVPR. pp. 14357–14366 (June 2026)

15. Turgut, K., Dutagaci, H., Rousseau, D.: Rosesegnet: An attention-based deep learning architecture for organ segmentation of plants. Biosystems Engineering 221, 138–153 (2022). https://doi.org/10.1016/j.biosystemseng.2022.06.016, https://www.sciencedirect.com/science/article/pii/S1537511022001556

16. Wu, X., DeTone, D., Frost, D., Shen, T., Xie, C., Yang, N., Engel, J., Newcombe, R., Zhao, H., Straub, J.: Sonata: Self-supervised learning of reliable point representations. In: CVPR (2025)

17. Wu, X., Jiang, L., Wang, P.S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., Zhao, H.: Point transformer v3: Simpler, faster, stronger. In: CVPR (2024)

18. Wu, X., Lao, Y., Jiang, L., Liu, X., Zhao, H.: Point transformer v2: Grouped vector attention and partition-based pooling. In: NeurIPS (2022)

19. Wu, X., Tian, Z., Wen, X., Peng, B., Liu, X., Yu, K., Zhao, H.: Towards large-scale 3d representation learning with multi-dataset point prompt training. In: CVPR (2024)

20. Yang, Y., Huang, Y., Guo, Y.C., Lu, L., Wu, X., Lam, E.Y., Cao, Y.P., Liu, X.: Sampart3d: Segment any part in 3d objects. arXiv preprint arXiv:2411.07184 (2024)

21. Zhang, Y., Wu, X., Lao, Y., Wang, C., Tian, Z., Wang, N., Zhao, H.: Concerto: Joint 2d-3d self-supervised learning emerges spatial representations. In: NeurIPS (2025)

22. Zhang, Y., Wu, X., Yang, Y., Fan, X., Li, H., Zhang, Y., Huang, Z., Wang, N., Zhao, H.: Utonia: Toward one encoder for all point clouds. In: ICML (2026)

23. Zhu, J., Zhai, R., Ren, H., et al.: Crops3d: A diverse 3d crop dataset for realistic perception and segmentation toward agricultural applications. Scientific Data 11, 1438 (2024). https://doi.org/10.1038/s41597-024-04290-0

24. Zou, Y., Wang, H., Zhang, F., Ge, Y., Wang, W., Chen, M.: Gcassn: a graph convolutional attention synergistic segmentation network for 3d plant point cloud segmentation. Frontiers in Plant Science Volume 16 - 2025 (2025). https://doi. org/10.3389/fpls.2025.1621934, https://www.frontiersin.org/journals/ plant-science/articles/10.3389/fpls.2025.1621934