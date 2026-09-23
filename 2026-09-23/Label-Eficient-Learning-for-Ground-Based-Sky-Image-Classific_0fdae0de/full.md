# Label-Eficient Learning for Ground-Based Sky-Image Classification: A Benchmark of Transfer Learning, Active Learning, and Pseudo-Labeling on GCD

Esther Bou Dagher<sup>1</sup>, Viktoriya Bu-Dager<sup>2\*</sup>, Boguslaw Zegarlinski<sup>3\*</sup>

<sup>1</sup>CEREMADE, Universit´e Paris Dauphine-PSL, Paris, France.

<sup>2\*</sup>School of Computing and Information Science, Anglia Ruskin University, Cambridge, United Kingdom.

<sup>3\*</sup>Institute of Mathematics, Polish Academy of Sciences, Warsaw, Poland.

\*Corresponding author(s). E-mail(s): Viktoriya.Bu-Dager@aru.ac.uk; bzegarlinski@impan.pl;

Contributing authors: esther.bou-dagher@dauphine.psl.eu;

## Abstract

Accurate ground-based cloud classification is important for atmospheric monitoring, solar-energy forecasting, aviation weather assessment, and climate observation systems. However, reliable sky-image annotation is time-consuming, especially when cloud types are visually similar or mixed. In this work, we study the label eficiency of deep learning for ground-based cloud classification using the Ground-based Cloud Dataset (GCD). Rather than proposing a new architecture, we benchmark three practical learning strategies under limited annotation budgets: supervised transfer learning, uncertainty-based active learning, and high-confidence pseudo-labeling. An ImageNet-pretrained ResNet50 is used as a common frozen backbone, and experiments are repeated over five random seeds for label budgets ranging from 1% to 100% of the training labels. Supervised transfer learning is already highly label-eficient: test accuracy increases from 0.635 ± 0.018 with 1% labels to 0.730 ± 0.002 with 40% labels, approaching the full-label performance of 0.735 ± 0.003. Active learning and pseudo-labeling are competitive with supervised sampling and provide small improvements for some metrics and budgets, but neither produces a large or consistent aggregate gain. Diagnostic analyses show that accepted pseudo-labels

are highly reliable, with accuracy ranging from 0.946 to 0.977, but are biased toward easier, high-confidence sky-type groups. In contrast, uncertainty sampling preferentially queries uncertain samples from visually challenging groups, including Mixed and the visually confusable Stratocumulus and Cumulonimbus groups, but these targeted acquisitions yield only modest performance improvements. Overall, these results show that transfer learning can substantially reduce annotation requirements for GCD, while simple active and semi-supervised strategies provide limited additional gains over a strong supervised baseline.

Keywords: Ground-based cloud classification, label-eficient learning, transfer learning, active learning, pseudo-labeling

## 1 Introduction

Cloud observation is essential for weather monitoring, climate research, solar energy forecasting, aviation weather assessment, and local environmental observation. Cloud type, structure, and temporal evolution influence radiative transfer, precipitation processes, surface irradiance, and short-term atmospheric conditions [1, 2]. Ground-based sky imaging complements satellite observations by providing local, high-resolution views of cloud fields from below. Such images are therefore useful for operational tasks such as sky-condition assessment, cloud-type classification, and short-term forecasting.

Automatic ground-based cloud classification remains a challenging task. Cloud categories are often dificult to distinguish, particularly when cloud fields contain multiple cloud types or exhibit gradual transitions between morphologies. Some classes, such as clear sky, are relatively easy to identify, whereas others, including altocumulus, stratocumulus, cumulonimbus, and mixed cloud scenes, can share similar texture, illumination, scale, and shape. Early cloud classification approaches relied on handcrafted features, such as colour, texture, and statistical descriptors. More recently, deep learning has significantly improved image-based cloud classification by learning features directly from data, and convolutional neural networks and related architectures have become widely used for this task [2, 3].

Most recent work on ground-based cloud classification has focused on model design. Proposed methods include lightweight convolutional networks, graph-based models, attention mechanisms, transformer-inspired architectures, and hybrid feature-fusion approaches [3–6]. These studies show that specialised deep networks can achieve strong classification performance on public cloud-image datasets. However, practical deployment depends not only on architecture, but also on annotation cost. In many observational settings, labelled cloud images are expensive to obtain because reliable annotation requires domain expertise, consistent interpretation across cloud categories, and careful handling of ambiguous or mixed cases. Therefore, an important practical question is not only which architecture achieves the highest performance with fully labelled data, but also how much labelled data is required to achieve useful classification performance.

Label-eficient learning addresses this question by reducing dependence on manually labelled data. In remote sensing, label scarcity has motivated transfer learning, self-supervised learning, semi-supervised learning, and active learning approaches [7]. In ground-based cloud classification, contrastive self-supervised learning has been proposed as a way to reduce dependence on labelled data [2]. Pseudo-labeling provides another simple semi-supervised strategy: a model trained on a small labelled subset assigns labels to high-confidence unlabelled samples, which are then used for further training. Active learning takes a diferent approach by selecting unlabelled samples that are expected to provide the greatest information gain, such as those for which model predictions are most uncertain. Both strategies are attractive in cloud-image analysis, where large collections of unlabelled sky images are often available while expert-labelled data remain limited.

Despite this motivation, there is still a need for a systematic empirical benchmark of label-eficient methods for ground-based cloud classification. In particular, it is important to compare active learning and pseudo-labeling against a strong supervised transfer-learning baseline under the same label budgets. Such a benchmark should evaluate not only overall classification accuracy, but also class-balanced performance metrics, variability across random seeds, and diagnostic analyses that explain why label-eficient strategies do or do not improve performance. This is especially important for imbalanced cloud datasets, where high overall accuracy can hide poor performance on minority or visually ambiguous classes.

In this study, we focus on the practical problem of reducing annotation requirements for ground-based cloud classification. Using the Ground-based Cloud Dataset (GCD), we evaluate whether simple label-eficient learning strategies can improve upon a strong supervised transfer-learning baseline. Rather than proposing a new neuralnetwork architecture, the aim is to provide a controlled benchmark and diagnostic analysis of when active and semi-supervised strategies are useful in this setting.

The main contributions of this work are as follows:

• We quantify the supervised label-eficiency curve of frozen ImageNet-pretrained ResNet50 on GCD, showing how performance changes as the labelled training budget increases.

• We compare supervised stratified random sampling, uncertainty-based active learning, and high-confidence pseudo-labeling under matched label budgets, a fixed training protocol, and five random seeds.

• We evaluate performance using overall and class-balanced metrics, including accuracy, macro-F1 score, balanced accuracy, weighted-F1 score, and per-class F1 scores.

• We analyse pseudo-label coverage, accepted accuracy, rejected prediction accuracy, confidence, and class-relative selection within the unlabelled training pool.

• We examine active-learning query enrichment relative to the remaining unlabelled pool to determine whether uncertainty sampling preferentially selects visually ambiguous cloud categories.

## 2 Related work

## 2.1 Ground-based cloud image classification

Ground-based cloud classification has been studied using both traditional imageprocessing methods and modern deep-learning approaches. Early methods relied on hand-crafted descriptors designed to capture colour, texture, shape, and local image structure. Representative examples include feature extraction from whole-sky images, automatic whole-sky cloud classification, block-based statistical and texture descrip tors, weighted local binary patterns, and texton-based representations[8–12]. Later descriptor-based approaches used region covariance descriptors and Riemannian bagof-features representations to improve ground-based cloud classification [13]. These methods provided important baselines, but their performance depends strongly on the quality of the manually designed representation and can be limited by the large intra-class variability of cloud images.

Deep learning has substantially changed the methodology for ground-based cloud classification by allowing feature representations to be learned directly from images. DeepCloud showed that convolutional visual features extracted from convolutional neural networks can improve ground-based cloud image categorization compared with traditional descriptors [14]. CloudNet proposed a dedicated convolutional neuralnetwork for meteorological cloud classification and introduced the Cirrus Cumulus Stratus Nimbus (CCSN) dataset with cloud categories under meteorological standards [1]. Subsequent studies developed more specialised architectures, including graph convolutional models, heterogeneous feature-learning methods, attention-based networks, transformer-inspired models, and hybrid CNN–transformer approaches [5, 6, 15].

Recent work has also focused on models designed for larger public ground-based cloud datasets. CloudDenseNet proposed a lightweight reconstructed DenseNet architecture and evaluated it on large-scale ground-based cloud datasets, including GCD [3]. Improved RepVGG-style models with attention mechanisms have similarly been proposed to capture both local texture and longer-range spatial structure in cloud images [4]. These architecture-focused studies demonstrate the strong performance of deep networks for cloud classification. However, they typically evaluate performance using the available labelled training set, whereas the question of how performance changes under systematically reduced label budgets has received less attention.

## 2.2 Label scarcity and label-eficient learning in remote sensing

Label scarcity is a common challenge in remote sensing and environmental image analysis. Labelling often requires expert knowledge, quality control, and consistency across sensors, locations, and acquisition conditions. In ground-based cloud classification, these dificulties are amplified by mixed cloud fields, ambiguous visual boundaries, and gradual transitions between cloud types. Label-eficient learning is therefore important for deploying cloud-classification systems in new observational settings.

Several label-eficient learning paradigms have been explored in remote sensing, including transfer learning, self-supervised learning, semi-supervised learning, and active learning. Transfer learning is particularly attractive when the target dataset is not large enough to train a deep model from scratch. A model pretrained on a large image dataset can provide generic visual features, while the final layers are adapted to the target task. In this work, we use ResNet50 as the shared backbone, following the residual-learning framework introduced by He et al. [16], and initialise the model with ImageNet-pretrained weights, following the transfer-learning paradigm enabled by large-scale datasets such as ImageNet [17]. This provides a strong supervised baseline against which active learning and pseudo-labeling can be evaluated.

Self-supervised learning is another strategy for reducing dependence on labels. In remote sensing, self-supervised approaches have been studied as a way to exploit large unlabelled image archives [7]. In ground-based cloud classification, contrastive self-supervised learning has been used to learn cloud-image representations before supervised classification [2]. Our study is related in motivation, but diferent in scope: instead of proposing a new representation-learning method, we benchmark simple and reproducible label-eficient strategies under matched annotation budgets.

## 2.3 Active learning for remote sensing classification

Active learning aims to reduce annotation cost by selecting informative unlabelled samples for labelling [18]. In remote sensing, active learning has long been studied because labelled data are expensive and random sampling may not capture the full spatial, spectral, or visual diversity of the data. Common strategies include uncertainty sampling, margin sampling, entropy-based criteria, query-by-committee, and diversity-aware selection. Surveys of active learning in remote sensing emphasise that training-set quality is critical for classification performance and that uncertainty-based heuristics provide simple and practical sample-selection rules [19].

Deep active learning extends these ideas to neural networks and large image archives. Recent work in remote-sensing image classification has considered singlelabel, multi-class, and multi-label settings, often combining uncertainty with diversity to avoid selecting redundant samples [20]. These studies motivate active learning as a practical strategy for reducing annotation cost in remote-sensing classification. In the present study, we examine whether this strategy provides additional benefit over a strong transfer-learning baseline for ground-based cloud classification.

## 2.4 Pseudo-labeling and semi-supervised learning

Semi-supervised learning uses both labelled and unlabelled data during training. Pseudo-labeling is one of the simplest semi-supervised strategies: a model trained on labelled data predicts labels for unlabelled samples, and high-confidence predictions are treated as additional training labels [21]. This approach is attractive because it is easy to implement and can be applied to standard supervised architectures. However, its efectiveness depends strongly on pseudo-label quality. Incorrect pseudo-labels can reinforce model errors, while overly conservative thresholds may select mostly easy examples and add little new information.

Pseudo-labeling and related semi-supervised methods have been studied in remote sensing, especially for scene classification and semantic segmentation, where annotation costs are high [22–24]. These studies highlight both the potential and the limitations of pseudo-label-based learning. Reliable pseudo-labels can improve performance, but noisy or class-biased pseudo-labels may limit gains. Because pseudo-labeling can be afected by confirmation errors and class bias, diagnostic analyses of pseudo-label quality are important when evaluating semi-supervised methods.

## 2.5 Positioning of this work

The existing literature shows that deep learning can achieve strong ground-based cloud classification performance, and that label-eficient learning is important in remote sensing. However, architecture-focused cloud-classification studies do not directly answer how much labelled data is needed, or whether simple active and semi-supervised strategies improve over a strong transfer-learning baseline. This study addresses that gap by focusing on annotation eficiency rather than architectural novelty.

## 3 Materials and methods

## 3.1 Dataset and data partitioning

Experiments were conducted using the publicly available Ground-based Cloud Dataset, which contains seven sky-type groups defined according to the International Cloud Classification System and practical visual similarity. The seven GCD groups are: Cumulus; Altocumulus/Cirrocumulus; Cirrus/Cirrostratus; Clear sky; Stratocumulus/Stratus/Altostratus; Cumulonimbus/Nimbostratus; and Mixed cloud. Images with cloudiness no greater than 10% are included in the Clear sky group [15]. For readability, we use shortened labels throughout the manuscript: Cumulus, Altocumulus, Cirrus, Clear sky, Stratocumulus, Cumulonimbus, and Mixed. These shortened labels refer to the corresponding GCD sky-type groups rather than to individual cloud types. We use this convention consistently in figures, tables, and the subsequent analysis.

The dataset comprises a 10,000-image training set and an independent 9,000- image test set. Representative examples from each GCD sky-type group are shown in Figure 1.

In the final benchmark, no labelled validation subset was used for model selection. Instead, label-budget subsets were sampled from the full training set, and fixed training schedules were used across all methods, label budgets, and random seeds. Thus, the reported label budgets correspond directly to the labelled images used for model optimisation, without an additional labelled validation resource. The independent test set was loaded without shufling and was used exclusively for final performance evaluation. It was excluded from training, pseudo-label generation, active-learning sample acquisition, hyperparameter selection, and all other model-development steps.

The dataset exhibits substantial class imbalance. Within the training set, the Cumulonimbus group was the largest (3,003 images), whereas the Mixed group was the smallest (348 images). A similar distribution was observed in the independent test set. The complete distribution of the seven GCD sky-type groups is presented in Table 1.

![](images/07f196abdfda3ad869da85f9e3e351f0a0ffb0cb1cc8ea0b29709373ad5d6c88.jpg)  
Fig. 1 Representative training images from the seven GCD sky-type groups. Shortened labels are used for readability and refer to the corresponding GCD sky-type groups. The examples illustrate both visually distinctive groups, such as Clear sky, and more visually ambiguous groups, such as Mixed, Stratocumulus, and Cumulonimbus.

Table 1 Distribution of the seven GCD sky-type groups in the training and independent test sets. Label-budget subsets were sampled from the full 10,000-image training set, with no additional labelled validation subset used for model selection.
<table><tr><td>Sky-type group</td><td>Training</td><td>Test</td></tr><tr><td>Cumulus</td><td>775</td><td>750</td></tr><tr><td>Altocumulus</td><td>725</td><td>750</td></tr><tr><td>Cirrus</td><td>1,153</td><td>753</td></tr><tr><td>Clear sky</td><td>2,150</td><td>1,589</td></tr><tr><td>Stratocumulus</td><td>1,846</td><td>1,790</td></tr><tr><td>Cumulonimbus</td><td>3,003</td><td>2,761</td></tr><tr><td>Mixed</td><td>348</td><td>607</td></tr><tr><td>Total</td><td>10,000</td><td>9,000</td></tr></table>

Note: Shortened labels correspond to the GCD sky-type groups defined in Section 3.1.

The class imbalance motivated the use of class-sensitive evaluation metrics in addition to overall accuracy. Specifically, macro-F1 score and balanced accuracy were included to ensure that performance on majority groups did not obscure weaker performance on minority or visually ambiguous cloud categories.

## 3.2 Image preprocessing and data pipeline

All images were loaded using the TensorFlow/Keras image data pipeline. Images were resized to 224 × 224 pixels, corresponding to the standard input resolution used by ResNet50. Before being passed to the network, images were converted to 32- bit floating-point tensors and preprocessed by the standard ResNet50 preprocessing function used for the ImageNet-pretrained backbone.

For each seed and label budget, the selected labelled indices were used to construct a dataset from the preprocessed training set. This resulting dataset was batched with a mini-batch size of 32 and prefetched using TensorFlow automatic tuning.

The same preprocessing pipeline was applied to all training and test images, ensuring that diferences between the supervised, active-learning, and pseudo-labeling experiments arose from the label-selection and training protocols rather than from diferences in input preparation.

No data augmentation was applied in the main experiments. This provided a controlled comparison of supervised transfer learning, active learning, and pseudo-labeling under identical preprocessing and modelling conditions. Consequently, diferences in performance reflect the learning strategies and label availability rather than augmentation efects.

## 3.3 Transfer-learning backbone

All experiments used the same transfer-learning architecture. An ImageNet-pretrained ResNet50 was selected as the backbone because it provides a strong and widely used visual feature extractor, making it well suited for benchmarking labeleficient learning strategies. The network was loaded without its original classification head using include top=False, and a task-specific classification head was attached. The ResNet50 backbone was kept frozen throughout the benchmark using base model.trainable = False, so that the comparisons focused on labelled-data availability and learning strategy rather than backbone fine-tuning.

Given a preprocessed input image ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ the ResNet50 backbone produced a spatial feature representation,

$$
z = f _ { \mathrm { R e s N e t 5 0 } } ( \pmb { x } ) .\tag{1}
$$

The resulting feature map was processed by a global average pooling (GAP) layer, followed by a dense layer with 256 units and ReLU activation, dropout (0.5), and a final seven-class softmax output layer. The resulting architecture can be summarised as

$$
\begin{array} { r l } & { x \longmapsto f _ { \mathrm { R e s N e t 5 0 } } ( x ) \longmapsto \mathrm { G A P } } \\ & { \longmapsto \mathrm { D e n s e } ( 2 5 6 ) + \mathrm { R e L U } \longmapsto \mathrm { D r o p o u t } ( 0 . 5 ) \longmapsto \mathrm { S o f t m a x } ( 7 ) . } \end{array}\tag{2}
$$

All models were trained using sparse categorical cross-entropy loss and the Adam optimiser. To ensure that the reported label budgets did not depend on an additional labelled validation resource, no validation-loss-based early stopping, validation-based model checkpointing, or validation-based learning-rate scheduling was used. Instead, models were trained for fixed schedules, with the model obtained at the final training epoch evaluated on the independent test set. Class weights were not used; class imbalance was instead accounted for through stratified label sampling and class-sensitive evaluation metrics.

The same backbone architecture and preprocessing pipeline were used across the supervised, active-learning, and pseudo-labeling experiments. Method-specific training schedules, including learning rates and numbers of epochs, are described in the corresponding subsections.

## 3.4 Label-budget protocol and supervised baseline

To quantify the efect of annotation availability, models were trained using label budgets of 1%, 3%, 5%, 10%, 20%, 40%, and 100% of the 10,000-image training set. For each budget, labelled samples were selected by class-stratified random sampling without replacement, retaining at least one image from each class whenever possible. This produced labelled sets of 100, 299, 500, 1000, 2001, 3999, and 10000 images, respectively; small deviations from the nominal percentages arose from class-wise rounding. The supervised baseline used only the labelled subset at each budget.

The sampling procedure was repeated for five random seeds, 0, 1, 2, 3, and 4. For each seed and label budget, a fresh ImageNet-pretrained ResNet50 model with a randomly initialised classification head was trained using the architecture and optimisation framework described in Section 3.3 for a fixed schedule of 10 epochs. The training subset was shufled during optimisation, while the full training set was loaded in a fixed order to ensure reproducible index-based sampling. The independent test set was not used for training, sample selection, pseudo-label generation, active-learning acquisition, or model selection, and was used only for final evaluation. The 100%- label model served as the reference baseline for comparison with the reduced-label supervised, active-learning, and pseudo-labeling experiments.

## 3.5 Uncertainty-based active learning

To evaluate whether annotation eficiency could be improved by selecting informative samples, we implemented a pool-based active-learning strategy using uncertainty sampling. Active learning was evaluated at 1%, 3%, 5%, 10%, 20%, and 40% label budgets. The 100% budget was excluded because no unlabelled samples remained for acquisition.

For each random seed, active learning was initialised with the same class-stratified 1% labelled subset used in the supervised experiments, containing 100 images. The remaining training images formed the initial unlabelled pool. A model was first trained on the initial labelled subset using the fixed supervised protocol described in Section 3.3. It was then applied to the current unlabelled pool to compute class probabilities for each unlabelled image.

For an unlabelled image x, the uncertainty score was defined as

$$
u ( \pmb { x } ) = 1 - \operatorname* { m a x } _ { c } p _ { \theta } ( y = c \mid \pmb { x } ) ,\tag{3}
$$

where $p _ { \theta } ( y ~ = ~ c ~ \mid ~ { \pmb x } )$ is the predicted probability of class c under the current model parameters θ. Larger values of $u ( { \pmb x } )$ correspond to lower maximum predicted confidence and therefore higher uncertainty.

At each acquisition step, the most uncertain samples were selected from the current unlabelled pool, their ground-truth labels were revealed, and they were added to the labelled training set. The labelled set was expanded according to the sequence

$$
1 \%  3 \%  5 \%  1 0 \%  2 0 \%  4 0 \% .\tag{4}
$$

The resulting labelled-set sizes were 100, 299, 500, 1000, 2001, and 3999 images, respectively, matching the rounded label-budget sizes used in the supervised experiments.

After each acquisition step, the model was warm-started from the previous activelearning stage and further trained on the enlarged labelled set. The initial 1% model was trained for 10 epochs using Adam with learning rate $1 0 ^ { - 3 }$ . Subsequent acquisition stages were trained for a fixed 5 epochs using Adam with learning rate $5 \times 1 0 ^ { - 4 }$ . After each stage, the final model was evaluated on the independent test set.

## 3.6 High-confidence pseudo-labeling

To evaluate a simple semi-supervised learning strategy, we implemented highconfidence pseudo-labeling at label budgets of 1%, 3%, 5%, 10%, 20%, and 40% of the 10,000-image training set. The 100% budget was excluded because no unlabelled training samples remain for pseudo-label generation.

For each random seed and label budget, a class-stratified labelled subset was sampled from the training set, while the remaining images formed the unlabelled pool. A first-stage supervised model was trained for 10 epochs with a learning rate of $1 0 ^ { - 3 }$ on the labelled subset only, using the fixed training protocol described in Section 3.3. This model was then used to predict class probabilities for all images in the unlabelled pool.

For an unlabelled image x, the predicted pseudo-label $\widehat { y }$ and confidence $q ( { \pmb x } )$ were defined as

$$
\widehat { y } = \arg \operatorname* { m a x } _ { c } p _ { \theta } ( y = c \mid \pmb { x } ) , \qquad q ( \pmb { x } ) = \operatorname* { m a x } _ { c } p _ { \theta } ( y = c \mid \pmb { x } ) .\tag{5}
$$

Only predictions satisfying

$$
q ( { \pmb x } ) \geq \tau , \qquad \tau = 0 . 9 5 ,\tag{6}
$$

were retained as pseudo-labels.

The accepted pseudo-labelled images were combined with the original labelled subset and shufled together to form second-stage training set. The second-stage model was initialised from the first-stage checkpoint and trained for a fixed 5 additional epochs using Adam with learning rate $5 \times 1 0 ^ { - 4 }$ . The number of optimiser updates in the second stage was determined by the size of the combined labelled and pseudo-labelled dataset.

To separate the efect of pseudo-label information from the efect of additional optimisation, we also trained a continued-supervised control for each seed and label budget. This control was initialised from the same first-stage checkpoint as the semisupervised model and trained using the same learning rate and the same total number of optimiser updates, but using only the original labelled subset. Thus, the semisupervised model and the continued-supervised control difered only in whether the second-stage updates used pseudo-labelled samples in addition to the labelled data.

The first-stage model, the continued-supervised control, and the semi-supervised model were all evaluated on the independent test set. The main pseudo-labeling efect was measured by comparing the semi-supervised model with the update-matched continued-supervised control. Pseudo-label coverage and mean confidence were also recorded to characterise the size and confidence of the accepted pseudo-labelled set.

## 3.7 Evaluation metrics

Model performance was evaluated on the independent 9,000-image test set. For each method, label budget, and random seed, we computed accuracy, balanced accuracy, macro-F1, weighted-F1, per-class metrics, and the confusion matrix. Results are reported as the mean and standard deviation over five random seeds.

Accuracy was defined as the proportion of correctly classified test images:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \widehat { y } _ { i } = y _ { i } \right) ,\tag{7}
$$

where $N$ is the number of test images, $y _ { i }$ is the true class label, $\widehat { y } _ { i }$ is the predicted class label, and $\mathbb { I } ( \cdot )$ is the indicator function.

Because the GCD dataset is imbalanced across sky-type groups, we also used class-balanced metrics. For each class c, precision, recall, and F1-score were computed from the corresponding true positives $( \mathrm { T P } _ { c } )$ , false positives $( \mathrm { F P } _ { c } )$ , and false negatives $( \mathrm { F N } _ { c } )$

$$
\begin{array} { r l } & { \mathrm { P r e c i s i o n } _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } , } \\ & { \mathrm { R e c a l l } _ { c } \qquad = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } , } \\ & { \mathrm { F 1 } _ { c } \qquad = \frac { 2 \mathrm { P r e c i s i o n } _ { c } \mathrm { R e c a l l } _ { c } } { \mathrm { P r e c i s i o n } _ { c } + \mathrm { R e c a l l } _ { c } } . } \end{array}\tag{8}
$$

Macro-averaged metrics assign equal weight to each of the $C = 7$ classes. For example,

$$
\mathrm { M a c r o – F 1 } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { F } 1 _ { c } .\tag{9}
$$

Balanced accuracy was defined as the mean recall across the seven sky-type groups:

$$
\mathrm { B a l a n c e d ~ a c c u r a c y } = { \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } \mathrm { R e c a l l } _ { c } .\tag{10}
$$

These metrics are important here because strong performance on frequent sky-type groups, such as Cumulonimbus or Clear sky, may otherwise hide weaker performance on minority or visually ambiguous groups.

Weighted-F1 was calculated by weighting each class-specific F1-score by its test-set support:

$$
\mathrm { W e i g h t e d - F 1 } = \sum _ { c = 1 } ^ { C } \frac { N _ { c } } { N } \mathrm { F } 1 _ { c } ,\tag{11}
$$

where $N _ { c }$ is the number of test samples belonging to class c.

Confusion matrices were row-normalised for visualisation, so that each entry represents the fraction of images from a given true sky-type group assigned to each predicted group. They were used to identify persistent confusions between visually similar or mixed cloud groups.

## 3.8 Diagnostic analyses

## 3.8.1 Active-query enrichment

To interpret the behaviour of uncertainty sampling, we analysed the sky-type-group composition of the samples selected at each acquisition step. The acquisition strategy selected samples solely according to predictive uncertainty and did not use class labels. True sky-type-group labels were inspected only after acquisition for diagnostic purposes.

For each acquisition step from label budget a to label budget b, let $\mathcal { U } ^ { ( a ) }$ denote the unlabelled pool immediately before acquisition, and let $U ^ { ( a ) } = | \mathcal { U } ^ { ( a ) } |$ . Let $U _ { c } ^ { ( a ) }$ denote the number of remaining unlabelled samples from sky-type group c immediately before that acquisition step. Similarly, let $Q ^ { ( a  \bar { b } ) }$ denote the total number of queried samples, and let $Q _ { c } ^ { ( a  b ) }$ denote the number of queried samples from sky-type group c.

The queried fraction and the corresponding pre-acquisition pool fraction were defined as

$$
r _ { c } ^ { ( a  b ) } = \frac { Q _ { c } ^ { ( a  b ) } } { Q ^ { ( a  b ) } } , \qquad \pi _ { c } ^ { ( a ) } = \frac { U _ { c } ^ { ( a ) } } { U ^ { ( a ) } } .\tag{12}
$$

Here, $\pi _ { c } ^ { ( a ) }$ is computed using the remaining unlabelled pool immediately before the acquisition step, rather than the original full training set. This accounts for the fact that earlier acquisitions may deplete some sky-type groups from the unlabelled pool.

The active-query enrichment ratio was then calculated as

$$
E _ { c } ^ { ( a  b ) } = \frac { r _ { c } ^ { ( a  b ) } } { \pi _ { c } ^ { ( a ) } } .\tag{13}
$$

Values above one indicate that sky-type group c was over-represented among queried samples relative to its availability in the current unlabelled pool, whereas values below one indicate under-representation. The analysis was performed for each active-learning acquisition step in (4) and summarised across five random seeds.

## 3.8.2 Pseudo-label quality

To interpret the behaviour of pseudo-labeling, we performed a post hoc diagnostic analysis of pseudo-label coverage, accuracy, confidence, and sky-type-group selection.

Because GCD provides ground-truth labels for all training images, pseudo-label correctness could be evaluated after pseudo-label generation. These ground-truth labels were used only for diagnostic analysis and were not used during pseudo-label selection or second-stage training.

For each label budget and random seed, the first-stage supervised model was applied to the corresponding unlabelled training pool. Let U denote the unlabelled pool, with $U = \left| \mathcal { U } \right|$ . For each image $i \in \mathcal { U }$ , the model produced a predicted pseudolabel $\widehat { y } _ { i }$ and a confidence score $q _ { i } ,$ defined as the maximum predicted class probability. The accepted pseudo-label set was defined as

$$
\begin{array} { r } { A = \{ i \in \mathcal { U } : q _ { i } \geq \tau \} , \qquad \tau = 0 . 9 5 , } \end{array}\tag{14}
$$

with $A = | { \mathcal { A } } |$ . The rejected set was defined as $\mathcal { R } = \mathcal { U } \backslash A .$

Pseudo-label coverage and accepted pseudo-label accuracy were defined as

$$
\mathrm { C o v e r a g e } = \frac { A } { U } , \qquad \mathrm { A c c e p t e d ~ a c c u r a c y } = \frac { 1 } { A } \sum _ { i \in \mathcal { A } } \mathbb { I } \left( \widehat { y } _ { i } = y _ { i } \right) ,\tag{15}
$$

where $y _ { i }$ is the ground-truth label used only for post hoc evaluation, and $\mathbb { I } ( \cdot )$ is the indicator function. For comparison, we also computed the accuracy of rejected predictions,

$$
\mathrm { R e j e c t e d ~ a c c u r a c y } = \frac { 1 } { \left| \mathcal { R } \right| } \sum _ { i \in \mathcal { R } } \mathbb { I } \left( \widehat { y } _ { i } = y _ { i } \right) .\tag{16}
$$

Mean accepted confidence was computed as

$$
{ \mathrm { A c c e p t e d ~ c o n f i d e n c e } } = { \frac { 1 } { A } } \sum _ { i \in A } q _ { i } .\tag{17}
$$

We also examined whether accepted pseudo-labels were concentrated in particular GCD sky-type groups. For each true sky-type group $^ { c , }$ true-group coverage was defined as the fraction of unlabelled samples from that group that were accepted:

$$
\mathrm { C o v e r a g e } _ { c } = \frac { | \{ i \in \mathcal { A } : y _ { i } = c \} | } { | \{ i \in \mathcal { U } : y _ { i } = c \} | } .\tag{18}
$$

To account for diferences in sky-type-group prevalence within the unlabelled pool, we computed an accepted-group enrichment ratio,

$$
{ \mathrm { E n r i c h m e n t } } _ { c } = { \frac { | \{ i \in { \mathcal { A } } : y _ { i } = c \} | / A } { | \{ i \in { \mathcal { U } } : y _ { i } = c \} | / U } } .\tag{19}
$$

A value above one indicates that $_ \mathrm { s k y - t y p e }$ group c was over-represented among accepted pseudo-labels relative to its prevalence in the unlabelled pool, while a value below one indicates under-representation.

Finally, for each predicted pseudo-label class $^ { c , }$ we computed the precision of accepted pseudo-labels assigned to that class:

$$
\operatorname { P r e c i s i o n } _ { c } ^ { \mathrm { p s e u d o } } = { \frac { | \{ i \in { \mathcal { A } } : { \widehat { y } } _ { i } = c , \ y _ { i } = c \} | } { | \{ i \in { \mathcal { A } } : { \widehat { y } } _ { i } = c \} | } } .\tag{20}
$$

Class-wise accepted counts were averaged over all five random seeds, including seeds in which zero samples from a class were accepted. Coverage and enrichment were computed from pooled counts across seeds, while predicted-class precision was computed over accepted pseudo-labels assigned to each class. If no accepted pseudolabels were assigned to a class, the corresponding predicted-class precision was treated as undefined.

## 4 Results

## 4.1 Supervised label-eficiency curve

We first evaluated the supervised transfer-learning baseline across label budgets to quantify how classification performance changes with decreasing annotation availability. Figure 2 shows label-budget curves for test accuracy and macro-F1, while Table 2 reports accuracy, balanced accuracy, macro-F1, and weighted-F1 values.

![](images/be7d3754b65e7c5a9c08f9ca48915da589ce68e9cb86fa556792229cdcb4a844.jpg)

![](images/e5d19f7462bf8b7b513da1930c282a0f5df62a679565b42e2e867ee23055fa6f.jpg)  
Fig. 2 Supervised label-budget performance of the ImageNet-pretrained ResNet50 baseline on GCD. Panel (a) shows test accuracy and panel (b) shows macro-F1. Points denote the mean over five random seeds and error bars denote one standard deviation. Accuracy increases rapidly at low label budgets and begins to plateau from approximately 20–40% labels, while macro-F1 shows a more gradual improvement with some seed-level variability.

Test accuracy increased rapidly in the low-label regime and then improved more gradually as additional labels were added. With 1% of the training labels, corresponding to 100 labelled images, the model achieved $0 . 6 3 4 9 \pm 0 . 0 1 7 8$ accuracy, increasing to 0.7078 ± 0.0117 at 5% labels, 0.7237 ± 0.0035 at 20%, and 0.7298 ± 0.0024 at 40%. Under full supervision, accuracy was 0.7346±0.0034. Thus, 20% of the available labels recovered most of the full-label accuracy, within approximately 1.1 percentage points of the 100%-label baseline, while 40% labels reduced the gap to approximately 0.5 percentage points.

The class-balanced metrics showed a more gradual improvement and were not strictly monotonic across label budgets. $\mathrm { M a c r o - F 1 }$ increased from $0 . 5 5 6 2 \pm 0 . 0 1 5 1$ at 1% labels to $0 . 6 5 4 8 \pm 0 . 0 1 6 0$ at 5%, $0 . 6 8 6 1 \pm 0 . 0 1 5 2$ at 20%, and $0 . 6 9 9 7 \pm 0 . 0 0 7 3$ under full supervision. Balanced accuracy similarly increased from $0 . 5 4 2 8 \pm 0 . 0 1 9 9$ at 1% labels to $0 . 6 7 6 2 \pm 0 . 0 0 7 5$ under full supervision. The small decrease in macro-F1 and balanced accuracy between 20% and 40% labels falls within the observed seedlevel variability and should not be interpreted as a systematic degradation. Overall, the class-balanced metrics indicate that additional labelled data remained useful for improving performance across the sky-type groups, even after overall accuracy begins to show diminishing gains.

The gap between overall accuracy and class-balanced metrics suggests that label eficiency was not uniform across sky-type groups. Overall accuracy approached the full-label baseline relatively early, whereas macro-F1 and balanced accuracy improved more gradually. This indicates that the benefits of additional labels were not uniform across groups. The corresponding group-level behaviour is examined in Section 4.3.

Table 2 Supervised ResNet50 performance across labelled-data budgets. Results are reported as mean ± standard deviation over five random seeds.
<table><tr><td>Labels (%)</td><td>Labelled images</td><td>Accuracy</td><td>Balanced accuracy</td><td>Macro-F1</td><td>Weighted-F1</td></tr><tr><td>1</td><td>100</td><td> $0 . 6 3 4 9 \pm 0 . 0 1 7 8$ </td><td> $0 . 5 4 2 8 \pm 0 . 0 1 9 9$ </td><td> $0 . 5 5 6 2 \pm 0 . 0 1 5 1$ </td><td> $0 . 6 1 1 6 \pm 0 . 0 2 0 1$ </td></tr><tr><td>3</td><td>299</td><td> $0 . 6 9 9 5 \pm 0 . 0 0 6 0$ </td><td> $0 . 6 2 9 1 \pm 0 . 0 0 6 6$ </td><td> $0 . 6 4 7 8 \pm 0 . 0 0 7 8$ </td><td> $0 . 6 8 8 8 \pm 0 . 0 0 5 8$ </td></tr><tr><td>5</td><td>500</td><td> $0 . 7 0 7 8 \pm 0 . 0 1 1 7$ </td><td> $0 . 6 3 6 4 \pm 0 . 0 1 3 7$ </td><td> $0 . 6 5 4 8 \pm 0 . 0 1 6 0$ </td><td> $0 . 6 9 6 6 \pm 0 . 0 1 2 8$ </td></tr><tr><td>10</td><td>1,000</td><td> $0 . 7 1 4 5 \pm 0 . 0 1 3 8$ </td><td> $0 . 6 5 1 7 \pm 0 . 0 2 2 9$ </td><td> $0 . 6 6 9 5 \pm 0 . 0 2 2 6$ </td><td> $0 . 7 0 5 4 \pm 0 . 0 1 5 8$ </td></tr><tr><td>20</td><td>2,001</td><td> $0 . 7 2 3 7 \pm 0 . 0 0 3 5$ </td><td> $0 . 6 6 5 6 \pm 0 . 0 1 5 6$ </td><td> $0 . 6 8 6 1 \pm 0 . 0 1 5 2$ </td><td> $0 . 7 1 7 3 \pm 0 . 0 0 8 3$ </td></tr><tr><td>40</td><td>3,999</td><td> $0 . 7 2 9 8 \pm 0 . 0 0 2 4$ </td><td> $0 . 6 5 9 8 \pm 0 . 0 0 7 6$ </td><td> $0 . 6 8 2 6 \pm 0 . 0 1 1 9$ </td><td> $0 . 7 1 8 4 \pm 0 . 0 0 5 9$ </td></tr><tr><td>100</td><td>10,000</td><td> $0 . 7 3 4 6 \pm 0 . 0 0 3 4$ </td><td> $0 . 6 7 6 2 \pm 0 . 0 0 7 5$ </td><td> $0 . 6 9 9 7 \pm 0 . 0 0 7 3$ </td><td> $0 . 7 2 7 0 \pm 0 . 0 0 7 5$ </td></tr></table>

## 4.2 Comparison of supervised, active-learning, and pseudo-labeling strategies

We next compared the supervised stratified random-sampling, uncertainty-based active learning, and high-confidence pseudo-labeling under matched label budgets. Figure 3 shows the test-accuracy curves up to 40% labels, and Table 3 reports the corresponding accuracy, macro-F1, balanced accuracy, and weighted-F1 values.

Overall, the three strategies produced broadly similar test-accuracy curves, indicating that the frozen ResNet50 transfer-learning baseline was already highly labeleficient on GCD. With only 1% of the training labels, supervised learning achieved $0 . 6 3 5 \pm 0 . 0 1 8$ accuracy, while pseudo-labeling increased this to $0 . 6 4 6 \pm 0 . 0 1 6$ . Active learning was identical to the supervised baseline at 1% because both methods used the same initial labelled subset before any acquisition step.

![](images/163f864127752599b11e14ce4d4e6effb8ab0dea27c92dd0bc53e6e518b4c0be.jpg)  
Fig. 3 Label-eficiency results on GCD. Points show mean test accuracy over five random seeds and error bars show one standard deviation. The horizontal reference line indicates the 100%-label supervised result. (a) Supervised ResNet50 performance across labelled-data budgets from 1% to 100%. (b) Comparison of supervised stratified random sampling, uncertainty-based active learning, and high-confidence pseudo-labeling up to 40% labels.

At 3% and 5% labels, supervised learning achieved the highest mean accuracy, with 0.699 ± 0.006 and 0.708 ± 0.012, respectively. Active learning had lower accuracy at these budgets, although its macro-F1 and balanced accuracy were competitive, particularly at 5% labels. From 10% labels onward, the methods became increasingly close. At 10% labels, active learning and pseudo-labeling both reached approximately 0.718 accuracy, compared with 0.715±0.014 for the supervised baseline. At 40% labels, active learning and pseudo-labeling both achieved 0.733 mean accuracy, close to the 100%-label supervised reference of $0 . 7 3 5 \pm 0 . 0 0 3$

Class-balanced metrics showed small advantages for the label-eficient methods at some budgets. Active learning had the highest macro-F1 at 3%, 5%, 10%, and 40% labels, while pseudo-labeling had the highest macro-F1 at 1% and 20% labels. However, these diferences were modest in absolute terms. For example, at 40% labels, active learning improved macro-F1 from 0.683 ± 0.012 to $0 . 6 9 7 { \scriptstyle \pm 0 . 0 0 7 }$ relative to supervised sampling, while test accuracy increased only from $0 . 7 3 0 \pm 0 . 0 0 2$ to $0 . 7 3 3 \pm 0 . 0 0 3$

To control for the additional optimisation introduced by the second pseudolabeling stage, we also trained a continued-supervised model from the same first-stage checkpoint for the same number of optimiser updates. As shown in Appendix B, pseudo-labeling did not consistently outperform this update-matched control, with semi-supervised minus control diferences remaining close to zero across label budgets. Thus, the apparent gains of pseudo-labeling over the first-stage supervised model should be interpreted cautiously.

These results show that simple uncertainty sampling and high-confidence pseudolabeling remained competitive with supervised stratified sampling and produced small gains for some metrics and label budgets. However, neither method produced a large or consistent improvement over the strong transfer-learning baseline. The diagnostic analyses in Sections 4.4 and 4.5 examine possible reasons for these limited gains.

Table 3 Comparison of supervised learning, active learning, and pseudo-labeling across labelled-data budgets. Results are reported as mean ± standard deviation over five random seeds.
<table><tr><td>Methods</td><td>Labels (%)</td><td>Accuracy</td><td>Macro-F1</td><td>Balanced accuracy</td><td> $\mathrm { W e i g h t e d } { - } \mathrm { F } 1$ </td></tr><tr><td>Supervised</td><td>1</td><td> $0 . 6 3 5 \pm 0 . 0 1 8$ </td><td> $0 . 5 5 6 \pm 0 . 0 1 5$ </td><td> $0 . 5 4 3 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 2 \pm 0 . 0 2 0$ </td></tr><tr><td>Active learning</td><td>1</td><td> $0 . 6 3 5 \pm 0 . 0 1 8$ </td><td> $0 . 5 5 6 \pm 0 . 0 1 5$ </td><td> $0 . 5 4 3 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 2 \pm 0 . 0 2 0$ </td></tr><tr><td>Pseudo-labeling</td><td>1</td><td> $0 . 6 4 6 \pm 0 . 0 1 6$ </td><td> $0 . 5 7 3 \pm 0 . 0 1 6$ </td><td> $0 . 5 6 3 \pm 0 . 0 1 9$ </td><td> $0 . 6 2 3 \pm 0 . 0 1 7$ </td></tr><tr><td>Supervised</td><td>3</td><td> $0 . 6 9 9 \pm 0 . 0 0 6$ </td><td> $0 . 6 4 8 \pm 0 . 0 0 8$ </td><td> $0 . 6 2 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 8 9 \pm 0 . 0 0 6$ </td></tr><tr><td>Active learning</td><td>3</td><td> $0 . 6 8 9 \pm 0 . 0 0 8$ </td><td> $0 . 6 5 1 \pm 0 . 0 0 8$ </td><td> $0 . 6 2 5 \pm 0 . 0 1 6$ </td><td> $0 . 6 8 3 \pm 0 . 0 0 9$ </td></tr><tr><td>Pseudo-labeling</td><td>3</td><td> $0 . 6 9 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 2 9 \pm 0 . 0 1 0$ </td><td> $0 . 6 1 7 \pm 0 . 0 0 5$ </td><td> $0 . 6 7 5 \pm 0 . 0 0 7$ </td></tr><tr><td>Supervised</td><td>5</td><td> $0 . 7 0 8 \pm 0 . 0 1 2$ </td><td> $0 . 6 5 5 \pm 0 . 0 1 6$ </td><td> $0 . 6 3 6 \pm 0 . 0 1 4$ </td><td> $0 . 6 9 7 \pm 0 . 0 1 3$ </td></tr><tr><td>Active learning</td><td>5</td><td> $0 . 6 9 7 \pm 0 . 0 0 8$ </td><td> $0 . 6 6 6 \pm 0 . 0 1 0$ </td><td>0.645 ± 0.010</td><td> $0 . 6 9 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Pseudo-labeling</td><td>5</td><td> $0 . 7 0 6 \pm 0 . 0 1 3$ </td><td> $0 . 6 5 4 \pm 0 . 0 1 3$ </td><td> $0 . 6 3 8 \pm 0 . 0 1 1$ </td><td> $0 . 6 9 5 \pm 0 . 0 1 1$ </td></tr><tr><td>Supervised</td><td>10</td><td> $0 . 7 1 5 \pm 0 . 0 1 4$ </td><td>0.669 ± 0.023</td><td> $0 . 6 5 2 \pm 0 . 0 2 3$ </td><td> $0 . 7 0 5 \pm 0 . 0 1 6$ </td></tr><tr><td>Active learning</td><td>10</td><td> $0 . 7 1 8 \pm 0 . 0 0 3$ </td><td>0.689 ± 0.009</td><td> $0 . 6 6 4 \pm 0 . 0 0 7$ </td><td> $0 . 7 1 3 \pm 0 . 0 0 4$ </td></tr><tr><td>Pseudo-labeling</td><td>10</td><td> $0 . 7 1 8 \pm 0 . 0 1 0$ </td><td> $0 . 6 7 5 \pm 0 . 0 1 3$ </td><td> $0 . 6 5 9 \pm 0 . 0 1 1$ </td><td> $0 . 7 1 0 \pm 0 . 0 1 0$ </td></tr><tr><td>Supervised</td><td>20</td><td> $0 . 7 2 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 8 6 \pm 0 . 0 1 5$ </td><td> $0 . 6 6 6 \pm 0 . 0 1 6$ </td><td> $0 . 7 1 7 \pm 0 . 0 0 8$ </td></tr><tr><td>Active learning</td><td>20</td><td> $0 . 7 2 1 \pm 0 . 0 0 2$ </td><td> $0 . 6 8 5 \pm 0 . 0 0 4$ </td><td> $0 . 6 6 3 \pm 0 . 0 0 5$ </td><td> $0 . 7 1 4 \pm 0 . 0 0 2$ </td></tr><tr><td>Pseudo-labeling</td><td>20</td><td> $0 . 7 2 9 \pm 0 . 0 0 6$ </td><td> $0 . 6 9 1 \pm 0 . 0 1 1$ </td><td> $0 . 6 7 0 \pm 0 . 0 0 7$ </td><td> $0 . 7 2 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Supervised</td><td>40</td><td> $0 . 7 3 0 \pm 0 . 0 0 2$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 2$ </td><td> $0 . 6 6 0 \pm 0 . 0 0 8$ </td><td> $0 . 7 1 8 \pm 0 . 0 0 6$ </td></tr><tr><td>Active learning</td><td>40</td><td> $0 . 7 3 3 \pm 0 . 0 0 3$ </td><td> $0 . 6 9 7 \pm 0 . 0 0 7$ </td><td> $0 . 6 7 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 2 6 \pm 0 . 0 0 5$ </td></tr><tr><td>Pseudo-labeling</td><td>40</td><td> $0 . 7 3 3 \pm 0 . 0 0 5$ </td><td> $0 . 6 9 5 \pm 0 . 0 1 2$ </td><td> $0 . 6 7 2 \pm 0 . 0 1 3$ </td><td> $0 . 7 2 5 \pm 0 . 0 0 8$ </td></tr><tr><td>Supervised</td><td>100</td><td> $0 . 7 3 5 \pm 0 . 0 0 3$ </td><td>0.700 ± 0.007</td><td> $0 . 6 7 6 \pm 0 . 0 0 7$ </td><td> $0 . 7 2 7 \pm 0 . 0 0 7$ </td></tr></table>

## 4.3 Per-class performance and confusion patterns

To examine which sky-type groups contributed most to the remaining performance diferences, we analysed per-class F1 scores. Table 4 reports results for the three learning strategies at 10% and 40% labels. These budgets were selected to represent an intermediate low-label setting and the largest label-eficient budget considered before full supervision.

Clear sky was consistently the easiest group, with F1 scores close to 0.96–0.97 across methods and label budgets. Cumulus and Cumulonimbus also achieved relatively strong performance, with F1 scores around 0.71–0.75. These groups contributed substantially to the high overall accuracy observed in the label-budget curves.

The most dificult groups were Mixed, Stratocumulus, and Altocumulus. Mixed had the lowest F1 scores across the selected budgets, reflecting the visual heterogeneity of this group. Active learning improved Mixed substantially relative to supervised sampling, increasing the F1 score from $0 . 3 9 4 \pm 0 . 0 8 6$ to $0 . 4 9 7 \pm 0 . 0 3 3$ at 10% labels and from $0 . 3 7 6 \pm 0 . 0 6 6$ to $0 . 5 1 1 \pm 0 . 0 4 6$ at 40% labels. Pseudo-labeling also improved Mixed at 40% labels, reaching $0 . 4 7 1 \pm 0 . 0 6 0$ , although the gain was smaller than that of active learning.

Table 4 Per-class F1 scores for supervised learning, active learning, and pseudo-labeling at selected label budgets. Results are reported as mean ± standard deviation over five random seeds.
<table><tr><td></td><td colspan="2">Supervised</td><td colspan="2">Active learning</td><td colspan="2">Pseudo-labeling</td></tr><tr><td>Class</td><td>10%</td><td>40%</td><td>10%</td><td>40%</td><td>10%</td><td>40%</td></tr><tr><td>Altocumulus</td><td> $0 . 6 2 4 \pm 0 . 0 2 8$ </td><td> $0 . 6 2 5 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 1 \pm 0 . 0 1 9$ </td><td> $0 . 5 8 6 \pm 0 . 0 1 2$ </td><td> $0 . 6 4 9 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 4 \pm 0 . 0 2 3$ </td></tr><tr><td>Cirrus</td><td> $0 . 6 6 3 \pm 0 . 0 2 3$ </td><td> $0 . 7 1 6 \pm 0 . 0 1 3$ </td><td> $0 . 7 0 2 \pm 0 . 0 1 5$ </td><td> $0 . 7 1 1 \pm 0 . 0 1 2$ </td><td> $0 . 6 6 3 \pm 0 . 0 2 1$ </td><td> $0 . 7 1 2 \pm 0 . 0 1 7$ </td></tr><tr><td>Clear sky</td><td> $0 . 9 5 7 \pm 0 . 0 1 2$ </td><td> $0 . 9 7 4 \pm 0 . 0 0 4$ </td><td> $0 . 9 6 4 \pm 0 . 0 0 7$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 4$ </td><td> $0 . 9 5 7 \pm 0 . 0 0 7$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 3$ </td></tr><tr><td>Cumulonimbus</td><td> $0 . 7 3 8 \pm 0 . 0 1 7$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 1 1 \pm 0 . 0 1 5$ </td><td> $0 . 7 5 1 \pm 0 . 0 0 9$ </td><td> $0 . 7 4 1 \pm 0 . 0 1 5$ </td><td> $0 . 7 4 9 \pm 0 . 0 0 9$ </td></tr><tr><td>Cumulus</td><td> $0 . 7 3 2 \pm 0 . 0 2 2$ </td><td> $0 . 7 4 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 3 0 \pm 0 . 0 1 4$ </td><td> $0 . 7 5 0 \pm 0 . 0 1 3$ </td><td> $0 . 7 3 0 \pm 0 . 0 1 7$ </td><td> $0 . 7 4 8 \pm 0 . 0 0 8$ </td></tr><tr><td>Mixed</td><td> $0 . 3 9 4 \pm 0 . 0 8 6$ </td><td> $0 . 3 7 6 \pm 0 . 0 6 6$ </td><td> $0 . 4 9 7 \pm 0 . 0 3 3$ </td><td> $0 . 5 1 1 \pm 0 . 0 4 6$ </td><td> $0 . 3 9 9 \pm 0 . 0 7 5$ </td><td> $0 . 4 7 1 \pm 0 . 0 6 0$ </td></tr><tr><td>Stratocumulus</td><td> $0 . 5 7 8 \pm 0 . 0 2 5$ </td><td> $0 . 5 9 7 \pm 0 . 0 1 7$ </td><td> $0 . 6 0 6 \pm 0 . 0 1 1$ </td><td> $0 . 5 9 8 \pm 0 . 0 2 0$ </td><td> $0 . 5 8 4 \pm 0 . 0 1 6$ </td><td> $0 . 5 9 7 \pm 0 . 0 2 2$ </td></tr></table>

Note: Shortened labels correspond to the GCD sky-type groups defined in Section 3.1.

Altocumulus remained challenging across methods, with F1 scores around 0.59– 0.65 at the selected budgets. Stratocumulus also remained dificult, with F1 scores around 0.58–0.61. These results indicate that the gap between overall accuracy and macro-F1 was driven by weaker performance on heterogeneous or visually overlapping groups rather than by uniform errors across all categories.

The class-level results are consistent with the diagnostic analyses in Sections 4.4 and 4.5. High-confidence pseudo-labeling tended to select easier examples, while active learning queried uncertain samples from dificult groups, including Mixed and the Stratocumulus–Cumulonimbus boundary. However, these targeted acquisitions produced only modest improvements, suggesting that the remaining errors involve ambiguous or visually overlapping cases. Row-normalised confusion matrices, reported in Appendix $\mathrm { A } ,$ further illustrate persistent confusion involving Mixed scenes and confusion between Stratocumulus and Cumulonimbus groups.

## 4.4 Pseudo-label quality and selection bias

The pseudo-labeling results in Section 4.2 showed limited improvements over the supervised and continued-supervised baselines. To understand why, we analysed the pseudo-labels accepted using the confidence threshold $\tau { \it \Delta \phi } = 0 . 9 5$ . Figure 4 shows accepted pseudo-label accuracy and coverage as a function of label budget, while Table 5 reports the corresponding numerical values. Class-relative pseudo-label selection diagnostics are reported in Appendix C.

The accepted pseudo-labels were substantially more reliable than the rejected predictions. Accepted pseudo-label accuracy ranged from 0.946 ± 0.028 at 1% labels to $0 . 9 7 7 \pm 0 . 0 0 3$ at 40% labels, whereas the accuracy of rejected predictions remained much lower. Coverage increased from $0 . 3 4 5 \pm 0 . 0 5 9$ to $0 . 6 6 6 \pm 0 . 0 1 2$ as the labelleddata budget increased. The absolute number of accepted pseudo-labels at 20% and

![](images/fa5b449002b1e3e44d7a8bb12246573602d033a18f55b27c8a6c9d0141fd0422.jpg)  
Fig. 4 Pseudo-label quality on the unlabelled training pool. Accepted pseudo-label accuracy denotes the fraction of retained pseudo-labels that matched the ground-truth labels, using training labels only for post hoc diagnostic evaluation. Coverage denotes the fraction of the unlabelled pool assigned a pseudo-label with confidence at least $\tau = 0 . 9 5$ . Points show the mean over five random seeds and error bars show one standard deviation.

40% labels was nevertheless lower than at 10% labels, despite higher coverage, because the unlabelled pool was smaller at the larger labelled-data budgets.

However, high pseudo-label quality did not translate into consistent gains over the update-matched continued-supervised control. This indicates that the limitation was not only pseudo-label noise, but also which samples were selected. The class-relative diagnostics in Appendix C show that Clear sky was consistently over-represented among accepted pseudo-labels, with enrichment above one at every budget, while Mixed was consistently under-represented, with low true-group coverage and enrichment below one. This supports the conclusion that high-confidence pseudo-labeling selected easy, visually distinctive examples more readily than ambiguous mixed-cloud scenes.

This selection pattern helps explain why high-confidence pseudo-labeling did not consistently improve macro-F1 or balanced accuracy. The pseudo-labels were highly accurate, but they mainly added examples that the model already classified confidently. Consequently, they contributed relatively few additional examples from the groups associated with the remaining classification errors, particularly Mixed and visually similar groups such as Stratocumulus and Cumulonimbus. Thus, in this setting, the main limitation of pseudo-labeling was not simply pseudo-label noise, but the limited additional information provided by confidence-selected pseudo-labels.

Table 5 Pseudo-label quality across labelled-data budgets at threshold $\tau = 0 . 9 5 .$ Results are reported as mean ± standard deviation over five random seeds. Coverage is the fraction of the unlabelled pool assigned a pseudo-label. Accepted accuracy is the accuracy of predictions retained as pseudo-labels, while rejected accuracy is the accuracy of predictions below the confidence threshold, computed only for diagnostic purposes.
<table><tr><td>Labels Labelled (%)</td><td>images</td><td>Pseudo labels</td><td>Coverage</td><td>Accepted accuracy</td><td>Rejected accuracy</td><td>Accepted confidence</td></tr><tr><td>1</td><td>100</td><td> $3 4 1 8 \pm 5 8 7$ </td><td> $0 . 3 4 5 \pm 0 . 0 5 9$ </td><td> $0 . 9 4 6 \pm 0 . 0 2 8$ </td><td> $0 . 6 6 4 \pm 0 . 0 1 2$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td></tr><tr><td>3</td><td>299</td><td> $4 2 6 0 \pm 2 1 5$ </td><td> $0 . 4 3 9 \pm 0 . 0 2 2$ </td><td> $0 . 9 7 4 \pm 0 . 0 0 7$ </td><td> $0 . 7 3 0 \pm 0 . 0 1 1$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 1$ </td></tr><tr><td>5</td><td>500</td><td> $5 1 2 2 \pm 3 1 1$ </td><td> $0 . 5 3 9 \pm 0 . 0 3 3$ </td><td> $0 . 9 6 6 \pm 0 . 0 0 6$ </td><td> $0 . 6 9 2 \pm 0 . 0 1 9$ </td><td> $0 . 9 8 9 \pm < 0 . 0 0 1$ </td></tr><tr><td>10</td><td>1000</td><td> $5 2 0 7 \pm 1 0 5$ </td><td> $0 . 5 7 9 \pm 0 . 0 1 2$ </td><td> $0 . 9 7 6 \pm 0 . 0 0 2$ </td><td> $0 . 7 1 9 \pm 0 . 0 1 3$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 1$ </td></tr><tr><td>20</td><td>2001</td><td> $5 0 5 1 \pm 1 3 9$ </td><td> $0 . 6 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 9 7 3 \pm 0 . 0 0 4$ </td><td> $0 . 7 2 1 \pm 0 . 0 1 0$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 1$ </td></tr><tr><td>40</td><td>3999</td><td> $4 0 0 0 \pm 7 3$ </td><td> $0 . 6 6 6 \pm 0 . 0 1 2$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 3$ </td><td> $0 . 7 1 6 \pm 0 . 0 1 1$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 1$ </td></tr></table>

## 4.5 Active-learning query behaviour

We analysed the samples selected by uncertainty-based active learning to characterise the acquisition behaviour of the method. Figure 5 shows the active-query enrichment ratio for each sky-type group and acquisition step. Enrichment was computed relative to the remaining unlabelled pool immediately before each acquisition step, rather than relative to the original full training distribution. A value of one therefore indicates that the queried fraction for a class matched its availability in the current unlabelled pool, while values above one indicate over-representation and values below one indicate under-representation. The corresponding queried counts, queried fractions, pre-acquisition pool fractions, and enrichment ratios are reported in Appendix D.

Uncertainty sampling did not simply reproduce the distribution of the remaining unlabelled pool. At the earliest acquisition step, from 1% to 3% labels, Mixed was strongly enriched, with an enrichment ratio of $6 . 2 6 \pm 1 . 2 3$ , despite representing only a small fraction of the available unlabelled pool. Cirrus and Altocumulus were also over-represented at this stage, with enrichment ratios of $2 . 6 1 \pm 0 . 8 0$ and $1 . 9 4 \pm 0 . 7 4$ 2 respectively. In contrast, Clear sky was consistently under-represented across acquisition steps, indicating that the model rarely regarded Clear sky images as highly uncertain.

At later acquisition steps, the queried samples became increasingly concentrated in the visually confusable Stratocumulus and Cumulonimbus groups. During the $2 0 \% $ 40% acquisition step, Stratocumulus and Cumulonimbus accounted for the largest queried fractions, $0 . 3 6 0 \pm 0 . 0 2 0$ and $0 . 3 6 5 \pm 0 . 0 1 8$ , respectively. Their corresponding enrichment ratios were $2 . 1 5 \pm 0 . 1 0$ and $1 . 2 2 \pm 0 . 0 7$ . Mixed also remained enriched at this stage, with an enrichment ratio of $2 . 5 5 \pm 0 . 2 1$ , although its queried fraction was smaller because few Mixed samples remained in the unlabelled pool.

The analysis shows that maximum-softmax uncertainty sampling targeted sky-type groups associated with visual ambiguity, particularly Mixed in the early acquisition steps and the Stratocumulus–Cumulonimbus boundary at higher budgets. This pattern is consistent with the confusion-matrix and per-class analyses, where Mixed scenes and the Stratocumulus and Cumulonimbus groups remained among the more dificult cases. Despite this targeted acquisition behaviour, active learning produced only limited improvement over the supervised label-budget baseline. This suggests that the remaining errors were not resolved simply by adding uncertain examples, possibly because the queried samples were dificult boundary cases, redundant within feature space, or insuficient to overcome the strong baseline provided by ImageNet-pretrained transfer learning. More sophisticated acquisition strategies, such as uncertainty combined with diversity, class-balanced uncertainty sampling, or methods explicitly targeting persistent confusion pairs, may be needed to obtain larger gains.

![](images/931788c0e0fd29fc675bf0264ad67b2809a8244c742d43777c1be84e91b21578.jpg)  
Fig. 5 Sky-type group enrichment among samples queried by uncertainty-based active learning. Enrichment is computed relative to the remaining unlabelled pool immediately before each acquisition step and averaged over five random seeds. Values above one indicate that a skytype group was over-represented among queried samples relative to its availability in the current unlabelled pool, while values below one indicate under-representation. The dashed horizontal line indicates an enrichment ratio of one.Shortened labels correspond to the GCD sky-type groups defined in Section 3.1.

## 5 Discussion

In this study, we investigated label-eficient ground-based cloud classification on GCD using a frozen ImageNet-pretrained ResNet50 backbone. We compared supervised stratified sampling, uncertainty-based active learning, and high-confidence pseudolabeling under matched label budgets and a fixed training protocol. The main finding is that the supervised transfer-learning baseline was already highly label-eficient. Test accuracy increased rapidly from 1% to 5% labels and then improved more gradually, with diminishing gains at larger label budgets. With 20–40% of the training labels, performance approached that obtained with the full labelled set. This suggests that generic visual features learned from ImageNet can transfer efectively to ground-based cloud imagery, even when the labelled pool is severely limited.

However, class-balanced metrics showed a more nuanced picture. Although overall accuracy approached the full-label result relatively early, macro-F1 and balanced accuracy improved more gradually as additional labels were added. This indicates that the benefits of additional annotation were not uniform across sky-type groups. The per-class analysis confirmed this interpretation. Clear sky was classified accurately across all budgets, whereas Mixed, Altocumulus, and Stratocumulus remained persistently challenging. Mixed scenes were particularly dificult, likely because they contain visual characteristics associated with multiple sky-type groups. The persistent confusion between the Stratocumulus/Stratus/Altostratus and Cumulonimbus/Nimbostratus groups further suggests that some errors arise from substantial visual overlap rather than simply insuficient labelled data.

The active-learning results show that maximum-softmax uncertainty sampling was competitive with supervised stratified sampling and provided small improvements for some class-balanced metrics. In particular, active learning improved performance on Mixed at selected budgets and achieved the highest macro-F1 at several label budgets. However, these gains were modest and did not translate into a large improvement in overall accuracy. The query-enrichment analysis showed that uncertainty sampling did not simply reproduce the distribution of the remaining unlabelled pool. Instead, it strongly enriched Mixed during the earliest acquisition step and later concentrated on the visually confusable Stratocumulus and Cumulonimbus groups, while consistently under-selecting Clear sky. Thus, the limited gains were not because active learning failed to query dificult sky-type groups. Rather, the queried samples may have represented dificult boundary cases, partially redundant examples, or insuficient additional information to overcome the strong transfer-learning baseline.

High-confidence pseudo-labeling also produced only limited gains. At higher budgets, it slightly improved accuracy or macro-F1 relative to the first-stage supervised model, but the controlled comparison showed that pseudo-labeling did not consistently outperform an update-matched continued-supervised control. This indicates that part of the apparent benefit may have resulted from additional optimisation rather than from the pseudo-labelled samples themselves. The pseudo-label quality analysis showed that accepted pseudo-labels were substantially more accurate than rejected predictions, so the limitation was not primarily pseudo-label noise. Instead, the class-relative diagnostics showed a selection efect: the confidence threshold preferentially accepted easier or more visually distinctive sky-type groups, especially Clear sky, while Mixed remained under-represented relative to its dificulty. As a result, pseudo-labeling added many reliable examples, but not necessarily the examples most likely to improve class-balanced performance.

Overall, these findings suggest that simple label-eficient strategies provide limited additional benefit when the supervised transfer-learning baseline is already strong. A relatively small labelled subset may be suficient to obtain strong overall accuracy on GCD, but improving performance on ambiguous sky-type groups requires more targeted approaches. The results also show why diagnostic analyses are important: pseudo-labeling and active learning can produce similar aggregate performance while behaving diferently in terms of sample selection. Pseudo-labeling mainly adds high-confidence examples, whereas uncertainty sampling actively queries ambiguous examples. Nevertheless, neither strategy fully resolved the persistent group-level errors.

Several directions could improve upon these baselines. For pseudo-labeling, classaware thresholds, calibration-aware confidence scores, or methods that explicitly balance pseudo-label selection across sky-type groups may reduce the bias toward easy high-confidence examples. For active learning, uncertainty could be combined with diversity, representation-space coverage, or class-conditional acquisition objectives to reduce redundant selections and improve coverage of dificult regions. Hybrid active semi-supervised approaches may also be useful, for example by using active learning to label dificult regions of the feature space while using pseudo-labeling for reliable high-confidence samples.

This study has several limitations. First, the benchmark used a single frozen backbone architecture and a fixed preprocessing pipeline. This design enabled a controlled comparison of learning strategies, but other architectures, fine-tuning protocols, or augmentation strategies may yield diferent absolute performance. Second, the active-learning and pseudo-labeling methods were deliberately simple baselines; more advanced strategies may perform diferently. Third, the analysis was restricted to GCD, so the conclusions should be validated on additional ground-based cloud datasets with diferent imaging conditions, class definitions, and geographic settings. Despite these limitations, the results provide a reproducible benchmark for labeleficient ground-based cloud classification and highlight the importance of class-level diagnostics when evaluating active and semi-supervised learning methods.

## 6 Conclusion

This study presented a systematic benchmark of label-eficient learning strategies for ground-based cloud classification on GCD. Under matched label budgets and a fixed training protocol, supervised transfer learning with a frozen ImageNet-pretrained ResNet50 backbone proved highly label-eficient. Test accuracy increased rapidly at low label budgets, and using only 20–40% of the available training labels approached the performance obtained with the full labelled set.

Active learning and high-confidence pseudo-labeling provided only limited additional benefit over the supervised baseline. The results show that the main challenge was not simply the number of labelled images, but the dificulty of improving performance on visually ambiguous sky-type groups. Pseudo-labeling produced reliable high-confidence labels, but these were biased toward easier sky-type groups and did not consistently outperform an update-matched continued-supervised control. In contrast, uncertainty-based active learning did query dificult groups, including Mixed and the Stratocumulus–Cumulonimbus boundary, but these targeted acquisitions produced only modest improvements in overall performance.

Overall, these results provide a reproducible benchmark for label-eficient groundbased cloud classification and show that strong transfer-learning baselines are dificult to improve with simple active or semi-supervised strategies. Future gains are likely to require more targeted methods, such as class-aware pseudo-labeling, uncertainty sampling combined with diversity, or hybrid active semi-supervised approaches designed specifically for visually ambiguous sky-type groups.

## Declarations

Funding. E.B.D. is funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Sk lodowska-Curie grant agreement n<sup>o</sup> 101034255.

Competing interests. The authors have no competing interests.

Ethics approval and consent to participate. Not applicable.

Consent for publication. Not applicable.

Data availability. The Ground-based Cloud Dataset (GCD) used in this study is publicly available at https://github.com/shuangliutjnu/ TJNU-Ground-based-Cloud-Dataset.

Materials availability. Not applicable.

Code availability. The code used for the experiments is available at: https:// github.com/EstherBD/Label-Eficient-Ground-Based-Cloud-Classification-on-GCD. git

Author contribution. E.B.D. designed the study, implemented the experiments, analysed the results, and drafted the manuscript. V.B.D. contributed to the study design, interpretation of the results, and revision of the manuscript. B.Z. supervised the project, contributed to the interpretation of the results, and revised the manuscript.

## Appendix A Supervised confusion matrices

Figure A1 reports row-normalised confusion matrices for the supervised ResNet50 baseline at selected label budgets. These matrices complement the per-class F1 analysis in Section 4.3 by showing the main error patterns between GCD sky-type groups. Across the selected label budgets, Clear sky is classified with high recall, whereas the main persistent errors involve Mixed scenes and confusion between visually similar groups, particularly Stratocumulus/Stratus/Altostratus and Cumulonimbus/Nimbostratus.

## Appendix B Controlled pseudo-labeling results

Table B1 reports the controlled pseudo-labeling results. The semi-supervised model was compared with both the first-stage labelled-only model and an update-matched continued-supervised control initialised from the same first-stage checkpoint. This

![](images/e57e66efb5c80a7796b76597a8b469dd77a43511a0c5601e01c8c5ad495434c7.jpg)  
Fig. A1 Row-normalised confusion matrices for the supervised ResNet50 baseline at selected label budgets for seed 0. Rows correspond to true labels and columns to predicted labels. Shortened class labels are used for readability and correspond to the GCD sky-type groups defined in Section 3.1. Clear sky is classified accurately across all selected budgets, whereas the main persistent errors involve Mixed scenes and confusion between Stratocumulus and Cumulonimbus groups.

control separates the efect of pseudo-label information from the efect of additional optimisation.

## Appendix C Class-relative pseudo-label selection

Table C2 reports class-relative pseudo-label selection diagnostics for the highconfidence pseudo-labeling experiment at threshold $\tau ~ = ~ 0 . 9 5$ . These diagnostics account for the diferent class sizes in the unlabelled pool. The accepted true count is the number of unlabelled samples from a given true class that passed the confidence threshold. True-class coverage is the fraction of unlabelled samples from that true class that were accepted as pseudo-labels. Accepted enrichment compares the accepted true-class fraction with the corresponding class fraction in the unlabelled pool. Predicted-class precision is the pooled precision among accepted pseudo-labels assigned to the corresponding predicted class.

Table B1 Controlled pseudo-labeling results at threshold $\tau = 0 . 9 5 ,$ . Results are reported as mean ± standard deviation over five random seeds. The continued-supervised control was initialised from the same first-stage checkpoint as the semi-supervised model and trained for the same number of second-stage optimiser updates using only the labelled subset.  
Panel A: Pseudo-label selection
<table><tr><td>Labels (%)</td><td>Labelled images</td><td>Pseudo labels</td><td>Pseudo-label coverage</td></tr><tr><td>1</td><td>100</td><td> $3 4 1 8 \pm 5 8 7$ </td><td> $0 . 3 4 5 \pm 0 . 0 5 9$ </td></tr><tr><td>3</td><td>299</td><td> $4 2 6 0 \pm 2 1 5$ </td><td> $0 . 4 3 9 \pm 0 . 0 2 2$ </td></tr><tr><td>5</td><td>500</td><td> $5 1 2 2 \pm 3 1 1$ </td><td> $0 . 5 3 9 \pm 0 . 0 3 3$ </td></tr><tr><td>10</td><td>1000</td><td> $5 2 0 7 \pm 1 0 5$ </td><td> $0 . 5 7 9 \pm 0 . 0 1 2$ </td></tr><tr><td>20</td><td>2001</td><td> $5 0 5 1 \pm 1 3 9$ </td><td> $0 . 6 3 1 \pm 0 . 0 1 7$ </td></tr><tr><td>40</td><td>3999</td><td> $4 0 0 0 \pm 7 3$ </td><td> $0 . 6 6 6 \pm 0 . 0 1 2$ </td></tr></table>

Panel B: Test performance
<table><tr><td>Labels (%)</td><td>Stage 1 accuracy</td><td>Control accuracy</td><td>Semi-supervised accuracy</td><td>Semi-supervised - control</td></tr><tr><td>1</td><td> $0 . 6 3 4 9 \pm 0 . 0 1 7 8$ </td><td> $0 . 6 4 9 5 \pm 0 . 0 1 2 5$ </td><td> $0 . 6 4 6 2 \pm 0 . 0 1 6 2$ </td><td> $- 0 . 0 0 3 3 \pm 0 . 0 0 6 2$ </td></tr><tr><td>3</td><td> $0 . 6 9 9 5 \pm 0 . 0 0 6 0$ </td><td> $0 . 7 0 0 7 \pm 0 . 0 0 8 0$ </td><td> $0 . 6 9 4 2 \pm 0 . 0 0 4 1$ </td><td> $- 0 . 0 0 6 5 \pm 0 . 0 0 5 0$ </td></tr><tr><td>5</td><td> $0 . 7 0 7 8 \pm 0 . 0 1 1 7$ </td><td> $0 . 7 0 9 4 \pm 0 . 0 1 2 2$ </td><td> $0 . 7 0 5 9 \pm 0 . 0 1 3 2$ </td><td> $- 0 . 0 0 3 6 \pm 0 . 0 0 2 9$ </td></tr><tr><td>10</td><td> $0 . 7 1 4 5 \pm 0 . 0 1 3 8$ </td><td> $0 . 7 1 8 2 \pm 0 . 0 1 1 9$ </td><td> $0 . 7 1 8 2 \pm 0 . 0 1 0 2$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 1 0 0$ </td></tr><tr><td>20</td><td> $0 . 7 2 3 7 \pm 0 . 0 0 3 5$ </td><td> $0 . 7 2 7 8 \pm 0 . 0 0 4 2$ </td><td> $0 . 7 2 9 0 \pm 0 . 0 0 6 1$ </td><td> $0 . 0 0 1 2 \pm 0 . 0 0 8 8$ </td></tr><tr><td>40</td><td> $0 . 7 2 9 8 \pm 0 . 0 0 2 4$ </td><td> $0 . 7 3 5 0 \pm 0 . 0 0 3 4$ </td><td> $0 . 7 3 3 2 \pm 0 . 0 0 4 7$ </td><td> $- 0 . 0 0 1 8 \pm 0 . 0 0 3 9$ </td></tr></table>

## Appendix D Active-learning queried-class distribution and enrichment

Table D3 reports the sky-type-group composition of samples acquired by uncertaintybased active learning at each acquisition step. Queried counts, queried fractions, pre-acquisition pool fractions, and enrichment ratios are reported as mean ± standard deviation over five random seeds. The queried fraction denotes the proportion of the acquired batch belonging to each sky-type group. The pool fraction denotes the proportion of the remaining unlabelled pool belonging to that sky-type group immediately before the acquisition step. Enrichment is defined as the ratio between the queried fraction and this pre-acquisition pool fraction. Values above one indicate that a skytype group was over-represented among queried samples relative to its availability in the current unlabelled pool, while values below one indicate under-representation.

Zero-count classes are included when computing the means and standard deviations. This table complements the active-learning diagnostics in Section 4.5 by showing both the absolute number and relative fraction of queried samples from each sky-type group.

Table C2 Class-relative pseudo-label selection diagnostics at threshold $\tau = 0 . 9 5 .$ Accepted true counts are reported as mean ± standard deviation over five random seeds, with zero-selection seeds included. True-class coverage and accepted enrichment are computed from pooled counts across seeds. Predicted-class precision is the pooled precision among accepted pseudo-labels assigned to each class.
<table><tr><td></td><td></td><td>Accepted true count</td><td>True-class coverage</td><td>Accepted enrichment</td><td>Predicted-class precision</td></tr><tr><td>Labels (%)</td><td>Class</td><td></td><td>0.430</td><td>1.25</td><td>0.988</td></tr><tr><td>1 1</td><td>Altocumulus Cirrus</td><td> $3 0 9 \pm 6 9$   $1 0 4 \pm 6 0$ </td><td>0.091</td><td>0.26</td><td>0.943</td></tr><tr><td>1</td><td>Clear sky</td><td> $1 5 2 1 \pm 1 0 1$ </td><td>0.715</td><td>2.07</td><td>0.993</td></tr><tr><td>1</td><td>Cumulonimbus</td><td> $9 6 2 \pm 3 1 7$ </td><td>0.324</td><td>0.94</td><td>0.929</td></tr><tr><td>1</td><td>Cumulus</td><td> $1 7 4 \pm 1 0 3$ </td><td>0.226</td><td>0.66</td><td>0.947</td></tr><tr><td>1</td><td>Mixed</td><td> $1 5 \pm 8$ </td><td>0.043</td><td>0.12</td><td>0.333</td></tr><tr><td>1</td><td>Stratocumulus</td><td> $3 3 4 \pm 2 7 7$ </td><td>0.182</td><td>0.53</td><td>0.734</td></tr><tr><td>3</td><td>Altocumulus</td><td> $4 3 3 \pm 2 2$ </td><td>0.617</td><td>1.40</td><td>0.986</td></tr><tr><td>3</td><td>Cirrus</td><td> $3 3 7 \pm 6 9$ </td><td>0.301</td><td>0.69</td><td>0.980</td></tr><tr><td>3</td><td>Clear sky</td><td> $1 6 9 5 \pm 1 1 5$ </td><td>0.812</td><td>1.85</td><td>0.995</td></tr><tr><td>3</td><td>Cumulonimbus</td><td> $1 2 1 7 \pm 1 8 9$ </td><td>0.418</td><td>0.95</td><td>0.953</td></tr><tr><td>3</td><td>Cumulus</td><td> $2 7 1 \pm 1 0 2$ </td><td>0.360</td><td>0.82</td><td>0.984</td></tr><tr><td>3</td><td>Mixed</td><td> $2 1 \pm 1 1$ </td><td>0.062</td><td>0.14</td><td>0.690</td></tr><tr><td>3</td><td>Stratocumulus</td><td> $2 8 7 \pm 1 4 1$ </td><td>0.160</td><td>0.36</td><td>0.897</td></tr><tr><td>5</td><td>Altocumulus</td><td> $5 1 0 \pm 3 8$ </td><td>0.740</td><td>1.37</td><td>0.979</td></tr><tr><td>5</td><td>Cirrus</td><td> $4 1 4 \pm 5 0$ </td><td>0.378</td><td>0.70</td><td>0.975</td></tr><tr><td>5</td><td>Clear sky</td><td> $1 7 6 3 \pm 5 9$ </td><td>0.863</td><td>1.60</td><td>0.994</td></tr><tr><td>5</td><td>Cumulonimbus</td><td> $1 4 6 9 \pm 2 2 6$ </td><td>0.515</td><td>0.96</td><td>0.940</td></tr><tr><td>5</td><td>Cumulus</td><td> $4 2 8 \pm 9 8$ </td><td>0.582</td><td>1.08</td><td>0.976</td></tr><tr><td>5</td><td>Mixed</td><td> $3 4 \pm 1 1$ </td><td>0.103</td><td>0.19</td><td>0.836</td></tr><tr><td>5</td><td>Stratocumulus</td><td> $5 0 4 \pm 1 4 6$ </td><td>0.287</td><td>0.53</td><td>0.910</td></tr><tr><td>10</td><td>Altocumulus</td><td> $5 0 0 \pm 2 3$ </td><td>0.766</td><td>1.32</td><td>0.984</td></tr><tr><td>10</td><td>Cirrus</td><td> $5 1 6 \pm 5 6$ </td><td>0.497</td><td>0.86</td><td>0.980</td></tr><tr><td>10</td><td>Clear sky</td><td> $1 7 6 8 \pm 3 3$ </td><td>0.914</td><td>1.58</td><td>0.992</td></tr><tr><td>10</td><td>Cumulonimbus</td><td> $1 4 1 8 \pm 1 2 3$ </td><td>0.525</td><td>0.91</td><td>0.963</td></tr><tr><td>10</td><td>Cumulus</td><td> $5 1 1 \pm 4 4$ </td><td>0.733</td><td>1.27</td><td>0.979</td></tr><tr><td>10</td><td>Mixed</td><td> $5 0 \pm 1 7$ </td><td>0.159</td><td>0.28</td><td>0.874</td></tr><tr><td>10</td><td>Stratocumulus</td><td> $4 4 5 \pm 1 2 6$ </td><td>0.268</td><td>0.46</td><td>0.939</td></tr><tr><td>20</td><td>Altocumulus</td><td> $4 5 9 \pm 4 7$ </td><td>0.791</td><td>1.25</td><td>0.987</td></tr><tr><td>20</td><td>Cirrus</td><td> $5 5 2 \pm 2 6$ </td><td>0.599</td><td>0.95</td><td>0.981</td></tr><tr><td>20</td><td>Clear sky</td><td> $1 6 0 6 \pm 3 5$ </td><td>0.933</td><td>1.48</td><td>0.992</td></tr><tr><td>20</td><td>Cumulonimbus</td><td> $1 3 4 5 \pm 1 8 9$ </td><td>0.560</td><td>0.89</td><td>0.959</td></tr><tr><td>20</td><td>Cumulus</td><td> $4 9 9 \pm 3 2$ </td><td>0.805</td><td>1.27</td><td>0.982</td></tr><tr><td>20</td><td>Mixed</td><td> $8 2 \pm 1 8$ </td><td>0.294</td><td>0.46</td><td>0.925</td></tr><tr><td>20</td><td>Stratocumulus</td><td> $5 0 9 \pm 1 3 5$ </td><td>0.345</td><td>0.55</td><td>0.928</td></tr><tr><td>40</td><td>Altocumulus</td><td> $3 8 5 \pm 1 4$ </td><td>0.885</td><td>1.33</td><td>0.979</td></tr><tr><td>40</td><td>Cirrus</td><td> $4 5 0 \pm 4 9$ </td><td>0.650</td><td>0.98</td><td>0.985</td></tr><tr><td>40</td><td>Clear sky</td><td> $1 2 1 0 \pm 2 4$ </td><td>0.938</td><td>1.41</td><td>0.993</td></tr><tr><td>40</td><td>Cumulonimbus</td><td> $1 1 3 5 \pm 8 0$ </td><td>0.630</td><td>0.94</td><td>0.962</td></tr><tr><td>40</td><td>Cumulus</td><td> $4 1 1 \pm 1 8$ </td><td>0.884</td><td>1.33</td><td>0.980</td></tr><tr><td>40</td><td>Mixed</td><td> $4 7 \pm 8$ </td><td>0.227</td><td>0.34</td><td>0.972</td></tr><tr><td>40</td><td>Stratocumulus</td><td> $3 6 2 \pm 1 0 1$ </td><td>0.327</td><td>0.49</td><td>0.955</td></tr></table>

Note: Shortened labels correspond to the GCD sky-type groups defined in Section 3.1.

## References

[1] Zhang, J., Liu, P., Zhang, F., Song, Q.: CloudNet: Ground-Based Cloud Classification With Deep Convolutional Neural Network. Geophysical Research Letters 45(16), 8665–8672 (2018) https://doi.org/10.1029/2018GL077787

[2] Lv, Q., Li, Q., Chen, K., Lu, Y., Wang, L.: Classification of Ground-Based Cloud Images by Contrastive Self-Supervised Learning. Remote Sensing 14(22), 5821 (2022) https://doi.org/10.3390/rs14225821

[3] Li, S., Wang, M., Sun, S., Wu, J., Zhuang, Z.: CloudDenseNet: Lightweight Ground-Based Cloud Classification Method for Large-Scale Datasets Based on Reconstructed DenseNet. Sensors 23(18), 7957 (2023) https://doi.org/10.3390/ s23187957

[4] Shi, C., Han, L., Zhang, K., Xiang, H., Li, X., Su, Z., Zheng, X.: Improved RepVGG ground-based cloud image classification with attention convolution. Atmospheric Measurement Techniques 17(3), 979–997 (2024) https://doi.org/10. 5194/amt-17-979-2024

[5] Li, X., Qiu, B., Cao, G., Wu, C., Zhang, L.: A Novel Method for Ground-Based Cloud Image Classification Using Transformer. Remote Sensing 14(16), 3978 (2022) https://doi.org/10.3390/rs14163978

[6] Liu, S., Li, M., Zhang, Z., Cao, X., Durrani, T.S.: Ground-Based Cloud Classification Using Task-Based Graph Convolutional Network. Geophysical Research Letters 47(5), 2020–087338 (2020) https://doi.org/10.1029/2020GL087338

[7] Wang, Y., Albrecht, C.M., Braham, N.A.A., Mou, L., Zhu, X.X.: Self-Supervised Learning in Remote Sensing: A review. IEEE Geoscience and Remote Sensing Magazine 10(4), 213–247 (2022) https://doi.org/10.1109/MGRS.2022.3198244

[8] Calb´o, J., Sabburg, J.: Feature Extraction from Whole-Sky Ground-Based Images for Cloud-Type Recognition. Journal of Atmospheric and Oceanic Technology 25(1), 3–14 (2008) https://doi.org/10.1175/2007JTECHA959.1 . Chap. Journal of Atmospheric and Oceanic Technology

[9] Heinle, A., Macke, A., Srivastav, A.: Automatic cloud classification of whole sky images. Atmospheric Measurement Techniques 3(3), 557–567 (2010) https://doi. org/10.5194/amt-3-557-2010

[10] Cheng, H.-Y., Yu, C.-C.: Block-based cloud classification with statistical features and distribution of local texture features. Atmospheric Measurement Techniques 8(3), 1173–1182 (2015) https://doi.org/10.5194/amt-8-1173-2015

[11] Liu, S., Zhang, Z., Mei, X.: Ground-based cloud classification using weighted local binary patterns. Journal of Applied Remote Sensing 9(1), 095062 (2015)

[12] Dev, S., Lee, Y.H., Winkler, S.: Categorization of cloud image patches using an improved texton-based approach. In: 2015 IEEE International Conference on Image Processing (ICIP), pp. 422–426 (2015). https://doi.org/10.1109/ICIP.2015. 7350833

[13] Tang, Y., Yang, P., Zhou, Z., Pan, D., Chen, J., Zhao, X.: Improving cloud type classification of ground-based images using region covariance descriptors. Atmospheric Measurement Techniques 14(1), 737–747 (2021) https://doi.org/10.5194/ amt-14-737-2021

[14] Ye, L., Cao, Z.-G., Xiao, Y.: DeepCloud: Ground-Based Cloud Image Categorization Using Deep Convolutional Features. IEEE Transactions on Geoscience and Remote Sensing PP, 1–12 (2017) https://doi.org/10.1109/TGRS.2017.2712809

[15] Liu, S., Duan, L., Zhang, Z., Cao, X., Durrani, T.S.: Ground-Based Remote Sensing Cloud Classification via Context Graph Attention Network. IEEE Transactions on Geoscience and Remote Sensing 60, 1–11 (2022) https://doi.org/10. 1109/TGRS.2021.3063255

[16] He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016)

[17] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L.: ImageNet: A largescale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255 (2009). https://doi.org/10.1109/CVPR. 2009.5206848

[18] Settles, B.: Active Learning Literature Survey. Technical Report, University of Wisconsin-Madison Department of Computer Sciences (2009)

[19] Tuia, D., Volpi, M., Copa, L., Kanevski, M., Munoz-Mari, J.: A Survey of Active Learning Algorithms for Supervised Remote Sensing Image Classification. IEEE Journal of Selected Topics in Signal Processing 5(3), 606–617 (2011) https://doi. org/10.1109/JSTSP.2011.2139193

[20] M¨ollenbrok, L., Sumbul, G., Demir, B.: Deep Active Learning for Multi-Label Classification of Remote Sensing Images. IEEE Geoscience and Remote Sensing Letters 20, 1–5 (2023) https://doi.org/10.1109/LGRS.2023.3305647

[21] Lee, D.-H., et al.: Pseudo-label: The simple and eficient semi-supervised learning method for deep neural networks. In: Workshop on Challenges in Representation Learning, ICML, vol. 3, p. 896 (2013). Atlanta

[22] Wang, J.-X., Chen, S.-B., Ding, C.H.Q., Tang, J., Luo, B.: Semi-Supervised

Semantic Segmentation of Remote Sensing Images With Iterative Contrastive Network. IEEE Geoscience and Remote Sensing Letters 19, 1–5 (2022) https: //doi.org/10.1109/LGRS.2022.3157032

[23] Feng, J., Luo, H., Gu, Z.: Improving semi-supervised remote sensing scene classification via Multilevel Feature Fusion and pseudo-labeling. International Journal of Applied Earth Observation and Geoinformation 136, 104335 (2025) https://doi.org/10.1016/j.jag.2024.104335

[24] Ran, L., Li, Y., Liang, G., Zhang, Y.: Pseudo Labeling Methods for Semi-Supervised Semantic Segmentation: A Review and Future Perspectives. IEEE Transactions on Circuits and Systems for Video Technology 35(4), 3054–3080 (2025) https://doi.org/10.1109/TCSVT.2024.3508768

Table D3 Sky-type-group distribution and enrichment of samples queried by uncertainty-based active learning. Queried counts, queried fractions, pre-acquisition pool fractions, and enrichment ratios are reported as mean ± standard deviation over five random seeds.
<table><tr><td>Acquisition</td><td></td><td>Queried count</td><td>Queried fraction</td><td>Pool fraction</td><td>Enrichment</td></tr><tr><td>step</td><td>Class</td><td></td><td></td><td></td><td></td></tr><tr><td>1 → 3%</td><td>Cumulus</td><td> $1 7 \pm 1 3$  28 ± 11</td><td> $0 . 0 8 4 \pm 0 . 0 6 6$ </td><td> $0 . 0 7 7 \pm 0 . 0 0 0$ </td><td> $1 . 0 9 \pm 0 . 8 6$   $1 . 9 4 \pm 0 . 7 4$ </td></tr><tr><td>1 → 3% 1 → 3%</td><td>Altocumulus Cirrus</td><td>60 ± 18</td><td> $0 . 1 4 1 \pm 0 . 0 5 4$   $0 . 3 0 1 \pm 0 . 0 9 2$ </td><td> $0 . 0 7 3 \pm 0 . 0 0 0$   $0 . 1 1 5 \pm 0 . 0 0 0$ </td><td> $2 . 6 1 \pm 0 . 8 0$ </td></tr><tr><td>1 → 3%</td><td>Clear sky</td><td></td><td> $5 \pm 3 0 . 0 2 5 \pm 0 . 0 1 6$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 0$ </td><td> $0 . 1 2 \pm 0 . 0 8$ </td></tr><tr><td>1 → 3%</td><td>Stratocumulus</td><td>27 ± 9</td><td> $0 . 1 3 8 \pm 0 . 0 4 5$ </td><td> $0 . 1 8 5 \pm 0 . 0 0 0$ </td><td> $0 . 7 5 \pm 0 . 2 4$ </td></tr><tr><td>1 → 3%</td><td>Cumulonimbus</td><td></td><td> $1 9 \pm 7 0 . 0 9 3 \pm 0 . 0 3 4$ </td><td> $0 . 3 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 3 1 \pm 0 . 1 1$ </td></tr><tr><td>1 → 3%</td><td>Mixed</td><td>43 ± 9</td><td> $0 . 2 1 8 \pm 0 . 0 4 3$ </td><td> $0 . 0 3 5 \pm 0 . 0 0 0$ </td><td> $6 . 2 6 \pm 1 . 2 3$ </td></tr><tr><td>3 → 5%</td><td>Cumulus</td><td> $3 1 \pm 1 3$ </td><td> $0 . 1 5 3 \pm 0 . 0 6 3$ </td><td> $0 . 0 7 7 \pm 0 . 0 0 1$ </td><td> $1 . 9 7 \pm 0 . 8 0$ </td></tr><tr><td>3 → 5%</td><td>Altocumulus</td><td></td><td> $1 1 \pm 6 0 . 0 5 6 \pm 0 . 0 2 8$ </td><td> $0 . 0 7 1 \pm 0 . 0 0 1$ </td><td> $0 . 7 8 \pm 0 . 4 0$ </td></tr><tr><td>3 → 5%</td><td>Cirrus</td><td></td><td> $4 5 \pm 1 1 0 . 2 2 5 \pm 0 . 0 5 7$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 2$ </td><td> $2 . 0 1 \pm 0 . 4 8$ </td></tr><tr><td>3 → 5%</td><td>Clear sky</td><td></td><td> $1 9 \pm 6 0 . 0 9 4 \pm 0 . 0 3 0$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 0$ </td><td> $0 . 4 3 \pm 0 . 1 4$ </td></tr><tr><td>3 → 5%</td><td>Stratocumulus</td><td></td><td> $4 9 \pm 9 0 . 2 4 2 \pm 0 . 0 4 5$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 1$ </td><td> $1 . 3 0 \pm 0 . 2 4$ </td></tr><tr><td>3 → 5%</td><td>Cumulonimbus</td><td>28 ± 11</td><td> $0 . 1 4 0 \pm 0 . 0 5 3$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 4 6 \pm 0 . 1 7$ </td></tr><tr><td>3 → 5%</td><td>Mixed</td><td></td><td> $1 8 \pm 9 0 . 0 9 1 \pm 0 . 0 4 5$ </td><td> $0 . 0 3 1 \pm 0 . 0 0 1$ </td><td> $2 . 9 1 \pm 1 . 4 4$ </td></tr><tr><td>5 → 10%</td><td>Cumulus</td><td>30 ± 13</td><td></td><td></td><td></td></tr><tr><td>5 → 10%</td><td>Altocumulus</td><td>19 ± 6</td><td> $0 . 0 6 0 \pm 0 . 0 2 6$  0.039 ± 0.012</td><td> $0 . 0 7 6 \pm 0 . 0 0 1$   $0 . 0 7 1 \pm 0 . 0 0 1$ </td><td> $0 . 7 9 \pm 0 . 3 5$   $0 . 5 4 \pm 0 . 1 7$ </td></tr><tr><td>5 → 10%</td><td>Cirrus</td><td>82 ± 12</td><td>0.164 ± 0.025</td><td> $0 . 1 0 9 \pm 0 . 0 0 1$ </td><td> $1 . 5 1 \pm 0 . 2 2$ </td></tr><tr><td>5 → 10%</td><td>Clear sky</td><td></td><td>31 ± 7 0.062 ± 0.015</td><td> $0 . 2 2 1 \pm 0 . 0 0 1$ </td><td> $0 . 2 8 \pm 0 . 0 7$ </td></tr><tr><td>5 → 10%</td><td>Stratocumulus</td><td>117 ± 21</td><td> $0 . 2 3 3 \pm 0 . 0 4 1$ </td><td> $0 . 1 8 4 \pm 0 . 0 0 1$ </td><td> $1 . 2 7 \pm 0 . 2 3$ </td></tr><tr><td>5 → 10%</td><td>Cumulonimbus</td><td>173 ± 31</td><td> $0 . 3 4 6 \pm 0 . 0 6 2$ </td><td> $0 . 3 0 8 \pm 0 . 0 0 1$ </td><td> $1 . 1 2 \pm 0 . 2 0$ </td></tr><tr><td>5 → 10%</td><td>Mixed</td><td>48±11</td><td> $0 . 0 9 6 \pm 0 . 0 2 3$ </td><td> $0 . 0 3 0 \pm 0 . 0 0 1$ </td><td> $3 . 1 9 \pm 0 . 6 4$ </td></tr><tr><td>10 → 20%</td><td>Cumulus</td><td>59 ± 17</td><td></td><td></td><td></td></tr><tr><td>10 → 20%</td><td>Altocumulus</td><td>33 ± 10</td><td> $0 . 0 5 9 \pm 0 . 0 1 7$   $0 . 0 3 3 \pm 0 . 0 1 0$ </td><td> $0 . 0 7 7 \pm 0 . 0 0 2$   $0 . 0 7 3 \pm 0 . 0 0 1$ </td><td> $0 . 7 7 \pm 0 . 2 1$   $0 . 4 5 \pm 0 . 1 3$ </td></tr><tr><td>10 → 20%</td><td>Cirrus</td><td> $1 5 0 \pm 3 3$ </td><td> $0 . 1 5 0 \pm 0 . 0 3 3$ </td><td> $0 . 1 0 6 \pm 0 . 0 0 1$ </td><td> $1 . 4 1 \pm 0 . 3 0$ </td></tr><tr><td>10 → 20%</td><td>Clear sky</td><td>29 ± 15</td><td> $0 . 0 2 9 \pm 0 . 0 1 5$ </td><td> $0 . 2 3 0 \pm 0 . 0 0 0$ </td><td> $0 . 1 3 \pm 0 . 0 6$ </td></tr><tr><td>10 → 20%</td><td>Stratocumulus</td><td> $2 9 8 \pm 3 7$ </td><td> $0 . 2 9 7 \pm 0 . 0 3 7$ </td><td> $0 . 1 8 2 \pm 0 . 0 0 3$ </td><td> $1 . 6 4 \pm 0 . 2 1$ </td></tr><tr><td>10 → 20%</td><td>Cumulonimbus</td><td> $3 6 8 \pm 3 3$ </td><td> $0 . 3 6 7 \pm 0 . 0 3 3$ </td><td> $0 . 3 0 6 \pm 0 . 0 0 3$ </td><td> $1 . 2 0 \pm 0 . 1 1$ </td></tr><tr><td>10 → 20%</td><td>Mixed</td><td>65 ± 16</td><td> $0 . 0 6 5 \pm 0 . 0 1 6$ </td><td> $0 . 0 2 6 \pm 0 . 0 0 1$ </td><td> $2 . 4 7 \pm 0 . 5 7$ </td></tr><tr><td>20 → 40%</td><td>Cumulus</td><td> $1 1 0 \pm 4 7$ </td><td> $0 . 0 5 5 \pm 0 . 0 2 4$ </td><td> $0 . 0 7 9 \pm 0 . 0 0 3$ </td><td> $0 . 6 9 \pm 0 . 2 7$ </td></tr><tr><td>20 → 40%</td><td>Altocumulus</td><td> $4 8 \pm 1 4$ </td><td> $0 . 0 2 4 \pm 0 . 0 0 7$ </td><td> $0 . 0 7 8 \pm 0 . 0 0 1$ </td><td> $0 . 3 1 \pm 0 . 0 9$ </td></tr><tr><td>20 → 40%</td><td>Cirrus</td><td> $1 6 7 \pm 1 6$ </td><td> $0 . 0 8 3 \pm 0 . 0 0 8$ </td><td> $0 . 1 0 0 \pm 0 . 0 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 9$ </td></tr><tr><td>20 → 40%</td><td>Clear sky</td><td> $1 1 7 \pm 4 1$ </td><td> $0 . 0 5 8 \pm 0 . 0 2 0$ </td><td> $0 . 2 5 6 \pm 0 . 0 0 2$ </td><td> $0 . 2 3 \pm 0 . 0 8$ </td></tr><tr><td>20 → 40%</td><td>Stratocumulus</td><td> $7 1 9 \pm 4 1$ </td><td> $0 . 3 6 0 \pm 0 . 0 2 0$ </td><td></td><td></td></tr><tr><td>20 → 40%</td><td>Cumulonimbus</td><td> $7 2 9 \pm 3 7$ </td><td> $0 . 3 6 5 \pm 0 . 0 1 8$ </td><td> $0 . 1 6 7 \pm 0 . 0 0 7$ </td><td> $2 . 1 5 \pm 0 . 1 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $0 . 2 9 8 \pm 0 . 0 0 4$ </td><td> $1 . 2 2 \pm 0 . 0 7$ </td></tr><tr><td>20 → 40%</td><td>Mixed</td><td> $1 0 9 \pm 1 5$ </td><td> $0 . 0 5 5 \pm 0 . 0 0 8$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 2$ </td><td> $2 . 5 5 \pm 0 . 2 1$ </td></tr></table>

Note: Shortened labels correspond to the GCD sky-type groups defined in Section 3.1.