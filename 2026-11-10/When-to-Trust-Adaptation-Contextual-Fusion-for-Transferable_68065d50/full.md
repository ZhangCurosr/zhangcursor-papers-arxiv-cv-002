# When to Trust Adaptation: Contextual Fusion for Transferable Vision-Language Prompting

Zhenyuan Chen<sup>1</sup> Wenshuo Tao<sup>1</sup> Zhuo Huang<sup>4</sup> Shuo Chen<sup>5</sup> Mingming Cheng<sup>1</sup> Xiang Li<sup>2,1,3\*</sup>

<sup>1</sup>VCIP, College of Computer Science, Nankai University, China; <sup>2</sup>NKIARI, Shenzhen Futian, Nankai University, China;

<sup>3</sup>AAIS, Nankai University, China; <sup>4</sup>Australian Institute for Machine Learning;

<sup>5</sup>School of Intelligence Science and Technology, Nanjing University, China.

zhenyuanchen@mail.nankai.edu.cn; xiang.li.implus@nankai.edu.cn

## Abstract

When a vision–language model encounters a new class or domain, how much should it trust its learned prompts? Because source adaptation can correct pretrained errors on some target images while disturbing valid predictions on others, we introduce Contextual Adaptive Fusion (CAF), which retains four prediction branches by pairing frozen and prompted visual and textual representations. To arbitrate when branches disagree on an image, CAF combines each branch’s predictive confidence with how frequently the branches favor its proposed class across the unlabeled target collection, placing greater trust in predictions that are locally confident yet less dominated by collection-wide frequency bias. The fusion coefficients are learned once on source data and remain fixed at inference. Trained on ImageNet, CAF achieves 69.96% average accuracy across ten cross-dataset benchmarks (1.92 percentage points above ManiPT) and remains highly competitive across four ImageNet variants, setting a new best of 52.21% on ImageNet-S. In base-to-novel transfer across eleven datasets, CAF further reaches 79.77% novel-class accuracy and an 82.81% harmonic mean. Code is available at https://github.com/zhenyuanchenai/CAF.

## 1. Introduction

Vision-language models such as CLIP [28] enable visual recognition through text descriptions of candidate categories. As illustrated by CoOp in Fig. 1(a), prompt learning adapts these models with a small set of trainable tokens while keeping the pretrained encoders frozen [18, 50]. Transferable vision-language prompting aims to adapt pretrained models with limited labeled data while preserving their ability to generalize to new images, categories, and visual domains [4, 31]. However, source adaptation can improve recognition of seen classes while degrading performance on unseen classes, even relative to the frozen pretrained model [49]. A central challenge is therefore to exploit source supervision without sacrificing the pretrained model’s generalizable recognition ability.

![](images/b015ce0967020c585d6efffd5ae51ef3d544d9fdf7a0877685396fbd089741a1.jpg)  
Figure 1. Comparison of CLIP prompting strategies: (a) text prompting in CoOp [50]; (b) frozen-reference regularization in PromptSRC [19]; (c) fixed feature fusion in ManiPT [42]; and (d) contextual fusion in CAF, where instance-level confidence and collection-level context determine image-dependent weights for four prediction branches. All backbone encoders remain frozen.

To address this challenge, existing methods regularize adaptation against frozen references [12, 13, 19, 24, 33] (Fig. 1(b)). ManiPT [42] further retains pretrained information at inference through fixed fusion of frozen and prompted features (Fig. 1(c)). Because multimodal prompt learning adapts both visual features and textual class representations [3, 18, 19, 29, 46], retaining their frozen counterparts provides four possible image–text pairings and hence four prediction branches. Fusing these four branches is challenging because the relative reliability of pretrained and adapted predictions can vary across images [52], while shifts in visual distributions or class vocabularies can further complicate the transfer of source-derived fusion rules [49, 52]. Consequently, fusion weights optimized for source-domain accuracy can compromise robustness under distribution shift, as the optimal balance between pretrained and adapted predictions may differ across distributions [52]. The central question is therefore when to trust adaptation: how can we assign appropriate contributions to these four branches for each target image without access to target labels?

Determining these contributions requires evidence of each branch’s relative reliability for the current image [9, 32, 52]. Predictive confidence characterizes how certain a branch is about the current image, while predictions over the unlabeled target collection reveal which classes the branches tend to favor. We use this collective prediction behavior as a common reference for interpreting each branch’s confidence, linking the assessment of an individual prediction to the statistical context of its proposed class. This motivates two complementary types of features for adaptive fusion: instance-level confidencefeatures and collection-level contextual features. We instantiate these feature types with normalized predictive entropy [35] and a shared predictionfrequency statistic, respectively. The latter is computed by averaging class probabilities across unlabeled target images and all four branches, with each branch reading the statistic associated with its proposed class. Together, these features provide a contextual basis for deciding when to trust adaptation, allowing branch contributions to depend on both image-specific confidence and the collective prediction tendencies of the target collection.

Building on these two feature types, we propose Contextual Adaptive Fusion (CAF), which combines frozen and prompted predictions through image-dependent weighting (Fig. 2). CAF first learns independent visual and textual prompts with frozen encoders, using source classification together with regularization for visual margin preservation and textual prototype alignment. Using the resulting prompted representations alongside their frozen counterparts, CAF evaluates all four image–text pairings to retain both pretrained and adapted predictions for fusion. During source training, we learn two scalar fusion coefficients from labeled source examples to balance instance-level confidence and collection-level context. The fusion loss updates only these coefficients, while a separate objective trains the prompts. These coefficients are held fixed across target datasets, while the features they combine are computed from the target predictions. At inference, CAF follows a transductive protocol, pooling class probabilities over the full unlabeled target collection and all four branches to obtain the shared prediction-frequency statistic. For each image, this statistic supplies a negative-log-frequency correction at each branch’s top-ranked class, which complements its negative normalized entropy when the branches disagree. The two terms are combined using the learned coefficients, and the resulting scores are normalized with a softmax to produce image-dependent weights for fusing the original branch logits. This procedure produces predictions over the full target class vocabulary without a trainable scoring network, target labels, or target-time parameter updates.

Trained on ImageNet, CAF achieves state-of-the-art (SOTA) cross-dataset transfer with 69.96% average accuracy across ten benchmarks, surpassing ManiPT [42] by 1.92 percentage points and ranking first on eight datasets. CAF also achieves the best accuracy on ImageNet-S at 52.21%, exceeding ManiPT by 1.58 points, and remains competitive across the four ImageNet variants. In baseto-novel transfer across eleven datasets, CAF achieves the highest average novel-class accuracy (79.77%) and harmonic mean (82.81%), surpassing ManiPT by 1.10 and 0.72 percentage points, respectively.

Our main contributions are summarized as follows.

• Four-Branch Transfer Formulation. We formulate transferable prompting over four pairings of frozen and prompted visual–textual representations, decoupling unimodal and joint adaptation.

• Contextual Adaptive Fusion (CAF). We dynamically weight the four branches per image by combining branch confidence with target-collection prediction frequency via two fixed, source-learned coefficients.

• Mechanistic and Empirical Insights. We show that CAF concentrates its gains on images with branch disagreement, adapts modality preferences across transfer regimes, and achieves SOTA transfer.

## 2. Related Work

## 2.1. Transferable Vision-Language Prompting

Prompt learning adapts pretrained vision-language models such as CLIP [28] through a small set of trainable tokens while keeping the backbone encoders frozen. Early approaches learn textual contexts [50], condition prompts on individual images [49], or introduce visual prompts [17], while MaPLe [18] jointly adapts the visual and textual modalities via cross-modal prompt coupling. Subsequent methods enrich prompt learning with class attributes [10, 22, 37], language-model knowledge [47], taskaware context [14], and hierarchical cross-modal interactions [46]. Other approaches improve the learned representations through feature disentanglement [29], category- and sample-level confusion modeling [34], and spectral guidance [48]. Prompt diversity is explored through variational modeling [8], surrogate feature generation [23], and repulsive Bayesian sampling [1]. Complementary approaches expand the knowledge available to prompt learning through large-vocabulary pretraining [31], teacher-guided distillation [21], and cascaded prompt boosting and adaptation [40]. Beyond source-time prompt learning, TPT [35] adapts prompts at test time through entropy minimization with confidence-based selection of augmented views.

![](images/0cfa636dd71340cc2ca4012a85fc70cf9a54d155273385094d8010fb6f14cbed.jpg)  
Figure 2. Overview of CAF. (a) Frozen and prompted visual and textual representations form four prediction branches. (b) Normalized entropy measures instance-level uncertainty, while class probabilities averaged across images and branches provide collection-level context. (c) Two source-learned coefficients combine these signals into image-dependent weights that fuse the original logits over all classes. Prompt learning and fusion-coefficient learning use separate gradients; all learned parameters remain fixed at inference.

Preserving pretrained knowledge during adaptation is another central direction for generalizable prompt learning. ProGrad [51] constrains prompt updates using gradients derived from pretrained predictions, whereas Kg-CoOp [43] regularizes learned textual embeddings toward those obtained from hand-crafted prompts. PromptSRC and CoPrompt [19, 33] encourage consistency with pretrained representations, and SPTR [6] combines textual regularization with a similarity-based formulation to retain knowledge from hand-crafted prompts. More recent methods, including MMRL, MMRL++, and AlignedNorm [12, 13, 24], combine multimodal adaptation with regularization informed by pretrained representations. ManiPT [42] further retains frozen features alongside prompted features at inference. CAF builds on regularized multimodal prompting to learn adapted representations while retaining their frozen counterparts for prediction fusion. It complements representation-level knowledge preservation with contextual weighting of four prediction branches, using instancelevel confidence and collection-level context.

## 2.2. Prediction Fusion for Transferable Adaptation

Combining pretrained and adapted knowledge provides a complementary route to preserving transferability. WiSE-FT [39] interpolates the parameters of zero-shot and finetuned models, while Tip-Adapter [45] augments zero-shot logits with cache-based predictions retrieved from labeled support examples using a global mixing coefficient. ManiPT [42] combines frozen and prompted features through fixed normalized addition before computing image–text similarities. These approaches incorporate complementary information through global mixing coefficients or fixed feature addition.

Adaptive fusion determines model contributions using input-dependent evidence. Ding et al. [9] compare the indistribution scores of zero-shot and prompt-tuned classifiers to obtain sample-dependent fusion weights. Variance Reduction Fine-tuning (VRF) [52] uses proximity to a reference set of zero-shot failures to adjust the contribution of the fine-tuned model. In multi-backbone ensembling, Rodriguez-Opazo et al. [32] learn input-dependent temperature coefficients from visual representations to combine predictions from different CLIP backbones. These approaches derive their weighting signals from classifier outputs, reference-set proximity, or visual representations.

The unlabeled target collection provides an additional source of information for adapting predictions. TransCLIP [44] exploits the structure of this collection through transductive inference regularized by pretrained visionlanguage predictions. CAF incorporates collection-level information into the weighting of four prediction branches formed from frozen and prompted representations. It combines instance-level confidence features with collectionlevel contextual features, using predictive entropy and shared soft prediction frequencies to assess individual predictions in their target context. The two fusion coefficients are learned during source training and remain fixed at target inference, while the shared prediction frequencies are computed from the full unlabeled target collection. CAF thus performs transductive prediction fusion through a directly computed weighting rule, without a trainable routing network or target-time parameter updates.

## 3. Method

## 3.1. Problem Setup and Four-Branch Predictions

We consider transferable vision-language prompting from a labeled source dataset $\mathcal { D } _ { s }$ with class vocabulary $\mathcal { C } _ { s }$ to an unlabeled target collection $\mathcal { U } _ { t }$ with vocabulary $\mathcal { C } _ { t } .$ The visual distributions and class vocabularies may differ across source and target. Target class names and unlabeled images are available for transductive inference, without target labels or parameter updates.

As illustrated in Fig. 2(a), CAF learns independent visual and textual prompts while keeping both encoders frozen. We denote the frozen and prompted image features by $v _ { 0 } ( x )$ and $v _ { V } ( x )$ , and their textual counterparts by $t _ { 0 } ( c )$ and $t _ { T } ( c )$ All four representations are $\ell _ { 2 ^ { - } }$ normalized. Following ManiPT and CoPrompt [33, 42], $t _ { 0 } ( c )$ is formed by averaging normalized embeddings of fixed GPT-generated class descriptions and renormalizing the result. The prompted feature $t _ { T } ( c )$ encodes the class name with learned textual prompts.

CAF retains four image–text pairings, indexed by $a \in$ $\{ 0 , V \}$ and $b \in \{ 0 , T \}$ :

$$
\boldsymbol { z } _ { a b } ( \boldsymbol { x } , c ) = \tau \boldsymbol { v } _ { a } ( x ) ^ { \top } t _ { b } ( c ) ,\tag{1}
$$

where $\tau$ is the frozen pretrained logit scale. The indices a and b specify the visual and textual states: 0 denotes an unprompted representation, and V or T denotes prompting. This yields four branches, $\mathcal { K } = \{ 0 0 , 0 T , V 0 , V T \}$ , which share the same class ordering and logit scale and reuse the two visual and two textual representations. Their fullvocabulary logits provide the prediction sources for subsequent contextual fusion.

## 3.2. Regularized Multimodal Prompt Learning

CAF learns visual and textual prompts through source classification while regularizing each modality against its frozen reference. For a source minibatch $B ,$ classification supervision is applied to the fully prompted VT branch:

$$
\mathcal { L } _ { \mathrm { c l s } } = \frac { 1 } { | \mathcal { B } | } \sum _ { ( x , y ) \in \mathcal { B } } \mathrm { C E } ( z _ { V T } ( x ) , y ) .\tag{2}
$$

To preserve reliable pretrained decisions during visual adaptation, we define the true-class margin as $m _ { k } ( x , y ) =$ $z _ { k } ( x , y ) \ - \ \operatorname* { m a x } _ { c \in { \mathcal C } _ { s } \setminus \{ y \} } z _ { k } ( x , c )$ When the frozen 00 branch has a positive margin, we penalize any reduction in the corresponding $V 0$ margin:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m a r g i n } } = \displaystyle \frac { 1 } { | \mathcal { B } | } \sum _ { ( x , y ) \in \mathcal { B } } \mathbf { 1 } \{ m _ { 0 0 } ( x , y ) > 0 \} } \\ { \times \left[ m _ { 0 0 } ( x , y ) - m _ { V 0 } ( x , y ) \right] _ { + } , } \end{array}\tag{3}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ . Comparing 00 and $V 0$ keeps the textual reference fixed, thereby constraining the effect of visual prompting.

For textual adaptation, we align the prompted representations with the frozen class prototypes over the entire source vocabulary:

$$
\mathcal { L } _ { \mathrm { t e x t } } = \frac { 1 } { \vert \mathcal { C } _ { s } \vert } \sum _ { c \in \mathcal { C } _ { s } } \left[ 1 - t _ { T } ( c ) ^ { \top } t _ { 0 } ( c ) \right] .\tag{4}
$$

Since both representations are normalized, this term penalizes their angular discrepancy and anchors textual adaptation to the pretrained semantic references.

The overall prompt-learning objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r o m p t } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { m a r g i n } } \mathcal { L } _ { \mathrm { m a r g i n } } + \lambda _ { \mathrm { t e x t } } \mathcal { L } _ { \mathrm { t e x t } } , } \end{array}\tag{5}
$$

where the two coefficients control the regularization strengths. Classification updates both prompt sets, while margin preservation and prototype alignment update only the visual and textual prompts, respectively. The prompts are optimized with this objective during source training and fixed at target inference.

## 3.3. Confidence and Context Features

CAF extracts instance-level confidence and collectionlevel contextual features from the four prediction branches (Fig. 2(b)). We describe these features for target inference; source training uses the same definitions with source-class probabilities and the historical EMA table introduced in Sec. 3.5. Let $C = | \mathcal { C } _ { t } |$ and $p _ { k } ( c \mid x ) = \mathrm { s o f t m a x } ( z _ { k } ( x ) ) _ { c }$ We characterize each branch’s predictive uncertainty using normalized entropy:

$$
h _ { k } ( x ) = - { \frac { 1 } { \log C } } \sum _ { c \in { \mathcal { C } } _ { t } } p _ { k } ( c \mid x ) \log p _ { k } ( c \mid x ) .\tag{6}
$$

A lower entropy indicates a more concentrated prediction for the current image.

To interpret this confidence within the target context, we construct a shared soft prediction-frequency statistic for each unlabeled target collection:

$$
\bar { p } _ { t } ( c ) = \frac { 1 } { 4 | \mathcal { U } _ { t } | } \sum _ { x \in \mathcal { U } _ { t } } \sum _ { k \in \mathcal { K } } p _ { k } ( c \mid x ) .\tag{7}
$$

This statistic, denoted by $f ( c )$ in Fig. 2(b), aggregates class probabilities across all target images and all four branches, providing a common reference for their collective prediction tendencies.

For branch k, let $c _ { k } ( x ) = { \arg \operatorname* { m a x } } _ { c \in { \mathcal { C } } _ { t } } z _ { k } ( x , c )$ denote its top-ranked class. We obtain the corresponding contextual feature through a negative-log-frequency transformation:

$$
r _ { k } ( x ) = - \log ( C \operatorname* { m a x } \{ \bar { p } _ { t } ( c _ { k } ( x ) ) , \epsilon \} ) ,\tag{8}
$$

where $\epsilon > 0$ ensures numerical stability. The factor C sets the uniform frequency $1 / C$ as the reference: classes predicted less frequently receive larger contextual values. Thus, $h _ { k } ( x )$ describes the concentration of an individual prediction, while $r _ { k } ( x )$ relates its proposed class to the collective prediction behavior. These two features are subsequently combined to determine image-dependent fusion weights.

## 3.4. Contextual Adaptive Prediction Fusion

CAF combines confidence and context to determine each branch’s contribution (Fig. 2(c)). Let $d ( x ) = \mathbf { 1 } \{ \exists k \in K$ $c _ { k } ( x ) \neq c _ { 0 0 } ( x ) \}$ indicate disagreement among the branch predictions. We compute the fusion scores and weights as

$$
\begin{array} { l } { { \displaystyle { s _ { k } } ( x ) = - \alpha { h _ { k } } ( x ) + \beta d ( x ) { r _ { k } } ( x ) } , } \\ { { \displaystyle { w _ { k } } ( x ) = \frac { { \exp ( s _ { k } ( x ) ) } } { \sum _ { j \in \mathcal { K } } { \exp ( s _ { j } ( x ) ) } } } , } \end{array}\tag{9}
$$

where α and $\beta$ are learnable scalar coefficients balancing instance-level confidence and collection-level context. The contextual term adjusts branch scores according to the collective frequency of their proposed classes when the branches disagree.

The weights are nonnegative, sum to one, and are shared across all classes for each image. CAF uses them to fuse the original branch logits:

$$
\begin{array} { r l r } {  { z _ { \mathrm { C A F } } ( x , c ) = \sum _ { k \in { \mathcal K } } w _ { k } ( x ) z _ { k } ( x , c ) , } } \\ & { } & { \hat { y } ( x ) = \arg \underset { c \in \mathcal C _ { t } } { \operatorname* { m a x } } z _ { \mathrm { C A F } } ( x , c ) . } \end{array}\tag{10}
$$

Thus, branch-proposed classes inform the weighting rule, while the final prediction considers the full target vocabulary.

We learn the fusion coefficients through cross-entropy on labeled source examples. We first detach the branch logits, $\tilde { z } _ { k } ( x ) = \mathrm { s g } ( z _ { k } ( x ) )$ , and compute both confidence and contextual features from these detached predictions. The fusion objective is

$$
\mathcal { L } _ { \mathrm { f u s e } } = \frac { 1 } { | \mathcal { B } | } \sum _ { ( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { B } } \mathrm { C E } \Bigg ( \sum _ { \boldsymbol { k } \in \mathcal { K } } w _ { \boldsymbol { k } } ( \boldsymbol { x } ; \alpha , \beta ) \tilde { z } _ { \boldsymbol { k } } ( \boldsymbol { x } ) , \boldsymbol { y } \Bigg ) .\tag{11}
$$

The overall training loss is

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { p r o m p t } } + { \mathcal { L } } _ { \mathrm { f u s e } } .\tag{12}
$$

The stop-gradient operation ensures that ${ \mathcal { L } } _ { \mathrm { f u s e } }$ updates only α and $\beta ,$ while $\mathcal { L } _ { \mathrm { { p r o m p t } } }$ optimizes the visual and textual prompts. At inference, the learned coefficients remain fixed, while fusion weights vary across images according to their branch predictions and shared target statistics, without target labels or parameter updates.

## 3.5. Optimization and Transductive Inference

During source training, CAF maintains a shared exponential moving average (EMA) of predicted class probabilities over the source vocabulary. Let $C _ { s } = | \mathcal { C } _ { s } |$ and initialize the table uniformly as $f ^ { ( 0 ) } ( c ) = 1 / C _ { s }$ . For the augmented source minibatch $B _ { n }$ at iteration $n ,$ we compute

$$
\hat { f } ^ { ( n ) } ( c ) = \frac { 1 } { 4 | \mathcal { B } _ { n } | } \sum _ { ( x , y ) \in \mathcal { B } _ { n } } \sum _ { k \in \mathcal { K } } \mathrm { s g } ( p _ { k } ( c \mid x ) ) ,\tag{13}
$$

where branch probabilities are evaluated over $\mathcal { C } _ { s } .$ . This statistic averages predictions across all image views and branches, without using ground-truth class frequencies.

The current minibatch uses the existing table $f ^ { ( n ) }$ to compute contextual features and fusion weights. Prompt parameters and fusion coefficients receive gradients from their respective objectives, with the branch outputs and frequency table detached from fusion learning. After processing the minibatch, we update the table using its previously computed probabilities:

$$
f ^ { ( n + 1 ) } ( c ) = 0 . 9 9 f ^ { ( n ) } ( c ) + 0 . 0 1 \hat { f } ^ { ( n ) } ( c ) .\tag{14}
$$

Thus, each minibatch is weighted using historical statistics before its predictions enter the EMA.

At inference, all learned parameters are fixed, and the source EMA is replaced by target-specific statistics. For each target collection $\mathcal { U } _ { t } .$ , we first compute and cache the four branch logits while accumulating the shared probability average $\bar { p } _ { t }$ over all unlabeled images. We then traverse the cached logits to compute confidence and contextual features, derive image-dependent weights, and produce fused predictions. The second pass requires no additional encoder evaluations. Each target collection uses its own statistics, without target labels or parameter updates.

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We consider eleven image-classification benchmarks: ImageNet [7], Caltech101 [11], OxfordPets [27], StanfordCars [20], Flowers102 [26], Food101 [2], FGV-CAircraft [25], SUN397 [41], DTD [5], EuroSAT [15], and UCF101 [36]. Four additional ImageNet variants, ImageNetV2 [30], ImageNet-S [38], ImageNet-A, and ImageNet-R [16], provide distribution-shifted evaluation data.

Implementation details. We freeze both CLIP ViT-B/16 encoders and train eight prompt tokens per layer with SGD and the fusion coefficients with Adam. For cross-dataset and cross-domain transfer, we train on 16-shot ImageNet for 5 epochs. For base-to-novel evaluation, we train separately on each dataset for 20 epochs using 16 labeled images per base class and compute fusion statistics separately

Table 1. Cross-dataset and cross-domain evaluation. The backbone is ViT-B/16; accuracies are percentages. CAF results are mean ± sample standard deviation over three seeds, with standard deviations placed below the means in smaller type. The ImageNet column reports source accuracy. The two averages cover ten cross-dataset targets and four ImageNet variants, respectively. The best mean accuracy in each column is shown in bold.
<table><tr><td></td><td colspan="10">Source</td><td colspan="5">Cross-domain</td></tr><tr><td>Method</td><td>ImageNet Caltech (IN)</td><td>101</td><td>Oxford Stanford Flowers Pets</td><td>Cars</td><td>102</td><td>Food 101</td><td>FGVC SUN Aircraft 397</td><td>DTD</td><td>EuroSAT</td><td>UCF 101</td><td>Avg-10</td><td>IN-V2</td><td>IN-S</td><td>IN-A</td><td>IN-R</td><td>Avg-4</td></tr><tr><td>CoOp [50]</td><td>71.51</td><td>93.70</td><td>89.10</td><td>64.50</td><td>68.70</td><td>85.30 18.50</td><td>64.20</td><td>41.90</td><td>46.40</td><td>66.60</td><td>63.90</td><td>64.20</td><td>47.99</td><td>49.71</td><td>75.21</td><td>59.28</td></tr><tr><td>CoCoOp [49]</td><td>71.02</td><td>94.40</td><td>90.10</td><td>65.30</td><td>71.90</td><td>86.10</td><td>22.90</td><td>67.40 45.70</td><td>45.40</td><td>68.20</td><td>65.70</td><td>64.07</td><td>48.75</td><td>50.63</td><td>76.18</td><td>59.91</td></tr><tr><td>TPT [35]</td><td>68.98</td><td>94.10</td><td>87.70</td><td>66.80</td><td>68.90</td><td>84.60 24.70</td><td>65.50</td><td>47.70</td><td>42.40</td><td>68.00</td><td>65.10</td><td>63.45</td><td>47.94</td><td>54.77</td><td>77.06</td><td>60.81</td></tr><tr><td>VPT [8]</td><td>70.70</td><td>93.70</td><td>90.60</td><td>65.00</td><td>70.90</td><td>86.30</td><td>24.90</td><td>67.50 46.10</td><td>45.90</td><td>68.70</td><td>66.00</td><td>64.23</td><td>49.20</td><td>51.33</td><td>77.00</td><td>60.44</td></tr><tr><td>PromptSRC [19]</td><td>71.27</td><td>93.60</td><td>90.30</td><td>65.70</td><td>70.30</td><td>86.20</td><td>24.00</td><td>67.10 46.90</td><td>45.50</td><td>68.80</td><td>65.80</td><td>64.35</td><td>49.55</td><td>50.90</td><td>77.80</td><td>60.65</td></tr><tr><td>CoPrompt [33]</td><td>70.80</td><td>94.50</td><td>90.73</td><td>65.67</td><td>72.30</td><td>86.43 24.00</td><td>67.57</td><td>47.07</td><td>51.90</td><td>69.73</td><td>67.00</td><td>64.25</td><td>49.43</td><td>50.50</td><td>77.51</td><td>60.42</td></tr><tr><td>ArGue [37]</td><td>71.57</td><td>94.30</td><td>88.90</td><td>67.20</td><td>69.80</td><td>85.00 24.40</td><td>66.80</td><td>46.20</td><td>46.10</td><td>69.20</td><td>65.80</td><td>64.57</td><td>48.92</td><td>50.93</td><td>76.56</td><td>60.25</td></tr><tr><td>MMRL [12]</td><td>72.03</td><td>94.67</td><td>91.43</td><td>66.10</td><td>72.77</td><td>86.40 26.30</td><td>67.57</td><td>45.90</td><td>53.10</td><td>68.27</td><td>67.25</td><td>64.47</td><td>49.17</td><td>51.20</td><td>77.53</td><td>60.59</td></tr><tr><td>TAC [14]</td><td>72.77</td><td>94.53</td><td>90.67</td><td>65.30</td><td>72.20</td><td>85.83 23.53</td><td>67.63</td><td>47.57</td><td>48.07</td><td>70.00</td><td>66.53</td><td>65.97</td><td>50.30</td><td>51.73</td><td>78.50</td><td>61.63</td></tr><tr><td>HiCroPL [46]</td><td>70.84</td><td>94.48</td><td>90.13</td><td>65.68</td><td>72.03</td><td>86.46 26.58</td><td>68.78</td><td>53.19</td><td>49.19</td><td>70.31</td><td>67.68</td><td>64.33</td><td>49.47</td><td>50.79</td><td>77.15</td><td>60.44</td></tr><tr><td>SurPL-G [23]</td><td>73.33</td><td>93.73</td><td>90.16</td><td>64.67</td><td>70.53</td><td>85.52 24.80</td><td>67.43</td><td>48.54</td><td>52.85</td><td>67.86</td><td>66.61</td><td>66.59</td><td>50.45</td><td>50.00</td><td>78.45</td><td>61.37</td></tr><tr><td>MMRL++ [13]</td><td>71.87</td><td>94.63</td><td>91.43</td><td>66.60</td><td>73.53 86.73</td><td>26.07</td><td>67.77</td><td>46.13</td><td>53.00</td><td>69.03</td><td>67.49</td><td>64.67</td><td>49.30</td><td>51.00</td><td>77.43</td><td>60.60</td></tr><tr><td>AlignedNorm [24]</td><td>71.83</td><td>94.50</td><td>91.57</td><td>66.50</td><td>73.00 86.67</td><td>26.30</td><td>67.97</td><td>46.63</td><td>53.47</td><td>69.50</td><td>67.61</td><td>65.13</td><td>49.53</td><td>51.10</td><td>78.07</td><td>60.96</td></tr><tr><td>MMRL + ReBaPL [1]</td><td>71.00</td><td>94.60</td><td>91.97</td><td>66.63</td><td>72.57 86.43</td><td>26.33</td><td>67.90</td><td>47.27</td><td>52.97</td><td>69.50</td><td>67.62</td><td>62.50</td><td>48.40</td><td>49.63</td><td>75.63</td><td>59.04</td></tr><tr><td>MMRL + SpecPL [48]</td><td>72.07</td><td>94.85</td><td>91.99</td><td>66.48</td><td>72.96 86.86</td><td>26.58</td><td>67.94</td><td>46.43</td><td>52.88</td><td>69.10</td><td>67.61</td><td>64.60</td><td>49.33</td><td>51.43</td><td>77.67</td><td>60.76</td></tr><tr><td>ManiPT [42]</td><td>72.92</td><td>95.13</td><td>90.87</td><td>66.27</td><td>72.14</td><td>86.66 27.00</td><td>68.50</td><td>51.83</td><td>50.88</td><td>71.11</td><td>68.04</td><td>66.12</td><td>50.63</td><td>51.97</td><td>78.79</td><td>61.88</td></tr><tr><td>CAF (ours)</td><td>72.26 ±0.03</td><td>94.10 ±0.12</td><td>93.11 ±0.15</td><td>68.39 ±0.07</td><td>71.88 ±0.26</td><td>87.31 29.82 ±0.08</td><td>70.65 ±0.33 ±0.17</td><td>54.83 ±0.22</td><td>56.36 ±1.35</td><td>73.09</td><td>69.96 ±0.14</td><td>65.63 ±0.15</td><td>52.21 ±0.11</td><td>49.63 ±0.26</td><td>77.97 ±0.08</td><td>61.36 ±0.08</td></tr></table>

on the unlabeled base and novel test splits. Training hyperparameters are detailed in Sec. A.2 of the supplement.

## 4.2. Cross-Dataset and Cross-Domain Generalization

As shown in Tab. 1, CAF achieves the highest average accuracy of 69.96% ± 0.14% across the ten cross-dataset benchmarks, surpassing the strongest compared baseline, ManiPT [42], by 1.92 percentage points. It also outperforms the recent MMRL-based ReBaPL [1] and SpecPL [48] variants by 2.34 and 2.35 percentage points, respectively. CAF ranks first on eight of the ten datasets, spanning fine-grained recognition [25], scene classification [41], texture recognition [5], remote sensing [15], and action recognition [36], indicating broad transferability across diverse visual domains and class vocabularies. These target improvements are achieved despite lower ImageNet accuracy than ManiPT and TAC [14], showing that higher source accuracy does not necessarily translate into stronger cross-dataset generalization. CAF also maintains competitive robustness on the four ImageNet variants, achieving an average accuracy of $6 1 . 3 6 \% \pm 0 . 0 8 \%$ , exceeding 13 of the 16 compared baselines, and obtaining the best result among them on ImageNet-S at 52.21%. These results indicate that CAF mitigates overfitting to the source class vocabulary, delivering substantial advantages under joint visual and semantic shifts while maintaining competitive robustness across domain shifts.

## 4.3. Base-to-Novel Generalization

With a single loss-weight pair used for all eleven datasets, CAF attains 79.52% average novel-class accuracy and a harmonic mean (HM) of 82.53%, outperforming every baseline on both metrics, while its average base accuracy (85.78%) is within 0.04 percentage points of ManiPT [42], the strongest baseline on novel accuracy and HM in Tab. 2. This 0.85-point novel gain, with base accuracy essentially unchanged (0.04 points lower), raises HM by 0.44 points and yields the smallest base–novel gap among the promptlearning methods compared (6.26 vs. 7.15 points for ManiPT). Compared with the baseline with the highest base accuracy (SurPL-G [23], 86.37%), CAF incurs a 0.59-point drop in base accuracy but gains 3.20 points on novel classes and 1.50 points on HM, indicating a more favorable base– novel trade-off.

CAF<sup>∗</sup>, which uses dataset-specific loss weights, further raises base accuracy, novel accuracy, and HM to 86.09%, 79.77%, and 82.81%, exceeding ManiPT on all three metrics (+0.27 base, +1.10 novel, and +0.72 HM). CAF<sup>∗</sup> achieves the best HM on nine of the eleven datasets and matches or exceeds ManiPT’s HM on every dataset. Its largest novel-class gains over the best baseline result on each dataset are 2.88 and 2.70 points on FGVCAircraft (44.63%) and DTD (71.26%). These are also the two benchmarks with the lowest best-baseline novel accuracy. Despite this per-dataset tuning, the average gain over CAF is at most 0.31 points on any metric. The shared setting thus captures most of the benefit of loss-weight selection, and the improvement over baselines does not hinge on datasetspecific tuning.

Table 2. Base-to-novel generalization on 11 datasets. Subscripts give sample standard deviations over three seeds, using seed-level HMs and dataset averages. <sup>∗</sup> denotes dataset-specific loss weights. Bold and underline mark the best and second-best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">(a) Average</td><td colspan="3">(b) ImageNet</td><td colspan="3">(c) Caltech101</td><td colspan="3">(d) OxfordPets</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP [28]</td><td>69.34</td><td>74.22</td><td>71.70</td><td>72.43</td><td>68.14</td><td>70.22</td><td>96.84</td><td>94.00</td><td>95.40</td><td>91.17</td><td>97.26</td><td>94.12</td></tr><tr><td>CoOp [50]</td><td>82.69 80.47</td><td>63.22 71.69</td><td>71.66 75.83</td><td>76.47 75.98</td><td>67.88 70.43</td><td>71.92</td><td>98.00</td><td>89.81</td><td>93.73</td><td>93.67</td><td>95.29</td><td>94.47</td></tr><tr><td>CoCoOp [49]</td><td>84.26</td><td>76.10</td><td>79.97</td><td>77.60</td><td></td><td>73.10</td><td>97.96</td><td>93.81</td><td>95.84</td><td>95.20 95.33</td><td>97.69</td><td>96.43</td></tr><tr><td>PromptSRC [19]</td><td>84.00</td><td>77.23</td><td>80.48</td><td>77.67</td><td>70.73 71.27</td><td>74.01</td><td>98.10</td><td>94.03</td><td>96.02</td><td>95.67</td><td>97.30</td><td>96.30</td></tr><tr><td>CoPrompt [33]</td><td>84.75</td><td>77.63</td><td>81.04</td><td>77.97</td><td>70.40</td><td>74.33</td><td>98.27 98.90</td><td>94.90</td><td>96.55</td><td>95.80</td><td>98.10 97.73</td><td>96.87</td></tr><tr><td>TAP [10] TAC [14]</td><td>85.24</td><td>77.60</td><td>81.24</td><td>78.57</td><td>71.03</td><td>73.99 74.61</td><td>98.57</td><td>95.50 95.27</td><td>97.17 96.89</td><td>95.93</td><td>98.17</td><td>96.76</td></tr><tr><td>HiCroPL [46]</td><td>85.89</td><td>77.99</td><td>81.75</td><td>78.07</td><td>71.72</td><td>74.76</td><td>98.77</td><td>95.96</td><td>97.34</td><td>96.28</td><td>97.76</td><td>97.04</td></tr><tr><td>SurPL-G [23]</td><td>86.37</td><td>76.32</td><td>81.03</td><td>78.74</td><td>70.49</td><td></td><td></td><td></td><td></td><td>96.37</td><td></td><td>97.01</td></tr><tr><td></td><td>85.53</td><td>78.32</td><td>81.77</td><td>77.63</td><td>71.50</td><td>74.39</td><td>98.77</td><td>95.16</td><td>96.93</td><td>95.60</td><td>97.41 97.43</td><td>96.89</td></tr><tr><td>MMRL++ [13] MMRL + ReBaPL [1]</td><td>85.74</td><td>77.44</td><td>81.38</td><td>77.90</td><td>68.83</td><td>74.44</td><td>99.07</td><td>94.53</td><td>96.75</td><td>95.93</td><td></td><td>96.51</td></tr><tr><td>MMRL + SpecPL [48]</td><td>85.90</td><td>77.55</td><td>81.51</td><td>78.05</td><td>71.50</td><td>73.09 74.63</td><td>99.10 99.10</td><td>94.27</td><td>96.62</td><td>96.20</td><td>97.57</td><td>96.74</td></tr><tr><td>ManiPT [42]</td><td>85.82</td><td>78.67</td><td>82.09</td><td>78.67</td><td>72.01</td><td>75.20</td><td>98.95</td><td>94.43</td><td>96.71</td><td>96.52</td><td>97.40 98.15</td><td>96.80</td></tr><tr><td>CAF</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>95.94</td><td>97.42</td><td></td><td></td><td>97.33</td></tr><tr><td>CAF*</td><td>85.78±0.05</td><td>79.52±0.50</td><td> $\overline { { 8 2 . 5 3 _ { \pm 0 . 2 7 } } }$ </td><td> $7 8 . 3 6 _ { \pm 0 . 0 3 }$ </td><td>73.15±0.15</td><td>75.67±0.09</td><td>98.77±0.23</td><td>95.89±0.17</td><td>97.31 ±0.19</td><td>96.10±0.26</td><td>98.02±0.12</td><td>97.05±0.08</td></tr><tr><td></td><td>86.09±0.15</td><td>79.77±0.23</td><td>82.81 ±0.19</td><td> $7 8 . 6 7 _ { \pm 0 . 1 0 }$ </td><td>73.27±0.14</td><td>75.87 ±0.04</td><td> $9 8 . 7 5 _ { \pm 0 . 2 9 } ^ { ^ { - - } }$ </td><td>96.14±0.25</td><td> $\mathbf { 9 7 . 4 3 _ { \pm 0 . 2 6 } }$ </td><td>96.72±0.25</td><td>97.97 ±0.14</td><td>97.34±0.12</td></tr><tr><td>Method</td><td></td><td>(e) StanfordCars</td><td></td><td></td><td>(f) Flowers102</td><td></td><td></td><td>(g) Food101</td><td></td><td></td><td>(h) FGVCAircraft</td><td></td></tr><tr><td></td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP [28] CoOp [50]</td><td>63.37</td><td>74.89</td><td>68.65</td><td>72.08</td><td>77.80</td><td>74.83</td><td>90.10</td><td>91.22</td><td>90.66</td><td>27.19</td><td>36.29</td><td>31.09</td></tr><tr><td>CoCoOp [49]</td><td>78.12</td><td>60.40</td><td>68.13</td><td>97.60</td><td>59.67</td><td>74.06</td><td>88.33</td><td>82.26</td><td>85.19</td><td>40.44</td><td>22.30</td><td>28.75</td></tr><tr><td>PromptSRC [19]</td><td>70.49</td><td>73.59 74.97</td><td>72.01 76.58</td><td>94.87</td><td>71.75</td><td>81.71</td><td>90.70</td><td>91.29</td><td>90.99</td><td>33.41</td><td>23.71</td><td>27.74</td></tr><tr><td></td><td>78.27</td><td></td><td>75.66</td><td>98.07</td><td>76.50</td><td>85.95</td><td>90.67</td><td>91.53</td><td>91.10</td><td>42.73</td><td>37.87</td><td>40.15</td></tr><tr><td>CoPrompt [33] TAP [10]</td><td>76.97</td><td>74.40 74.27</td><td></td><td>97.27</td><td>76.60</td><td>85.71</td><td>90.73</td><td>92.07</td><td>91.40</td><td>40.20</td><td>39.33</td><td>39.76</td></tr><tr><td></td><td>80.70</td><td></td><td>77.35</td><td>97.90</td><td>75.57</td><td>85.30</td><td>90.97</td><td>91.83</td><td>91.40</td><td>44.40</td><td>36.50</td><td>40.06</td></tr><tr><td>TAC [14]</td><td>81.63</td><td>74.17</td><td>77.72 78.14</td><td>97.97</td><td>76.87</td><td>86.15</td><td>90.87</td><td>91.87</td><td>91.37</td><td>44.60</td><td>37.70</td><td>40.86</td></tr><tr><td>HiCroPL [46]</td><td>81.51</td><td>75.04</td><td></td><td>98.29</td><td>75.46</td><td>85.38</td><td>90.96</td><td>91.67</td><td>91.31</td><td>48.38</td><td>41.75</td><td>44.82</td></tr><tr><td>SurPL-G [23]</td><td>83.57</td><td>72.77</td><td>77.80</td><td>98.90</td><td>72.88</td><td>83.92</td><td>90.92</td><td>91.81</td><td>91.36</td><td>49.20</td><td>36.93</td><td>42.19</td></tr><tr><td>MMRL++ [13]</td><td>81.33</td><td>75.27</td><td>78.18</td><td>98.53</td><td>77.90</td><td>87.01</td><td>90.47</td><td>91.73</td><td>91.10</td><td>46.40</td><td>38.77</td><td>42.24</td></tr><tr><td>MMRL + ReBaPL [1]</td><td>81.20</td><td>75.37</td><td>78.17</td><td>98.80</td><td>77.23</td><td>86.70</td><td>90.73</td><td>91.60</td><td>91.16</td><td>45.13</td><td>38.57</td><td>41.59</td></tr><tr><td>MMRL + SpecPL [48]</td><td>82.20</td><td>75.50</td><td>78.71</td><td>98.80</td><td>77.20</td><td>86.67</td><td>91.00</td><td>91.70</td><td>91.35</td><td>46.20</td><td>37.80</td><td>41.58</td></tr><tr><td>ManiPT [42]</td><td>82.53</td><td>75.61</td><td>78.92</td><td>98.12</td><td>76.84</td><td>86.19</td><td>91.17</td><td>92.16</td><td>91.66</td><td>48.28</td><td>39.87</td><td>43.68</td></tr><tr><td>CAF CAF*</td><td> $8 2 . 9 5 _ { \pm 0 . 5 7 }$ </td><td> $7 6 . 9 6 _ { \pm 0 . 3 0 }$ </td><td>79.84±0.30 80.08±0.41</td><td> $9 8 . 1 6 _ { \pm 0 . 2 4 }$ </td><td>76.83±0.50</td><td> $8 6 . 2 0 _ { \pm 0 . 3 7 }$ </td><td>90.45±0.23</td><td>92.53±0.02</td><td>91.48±0.13</td><td>45.70±0.44 46.18±1.08</td></table>

## 4.4. Ablation Study

Fusion weights. Tab. 3(a) shows that CAF improves generalization beyond both single-branch prediction $( { \bf e } _ { V T } )$ and static fusion. Uniform fusion $( \textstyle { \frac { 1 } { 4 } } \mathbf { 1 } )$ outperforms both source-fitted $\mathbf { \Pi } ( \mathbf { w } _ { \mathrm { s r c } } )$ and learned image-independent weights $\left( \mathbf { w } _ { \theta } \right)$ on the target benchmarks, suggesting that optimizing a global branch preference on the source domain does not necessarily improve transfer. CAF further exceeds uniform fusion by 1.89 and 0.48 percentage points on Avg-10 and $\mathrm { { A v g } \mathrm { { - } 4 , } }$ , respectively, while also improving ImageNet accuracy. These results support adapting branch contributions to individual images rather than applying the same weights to every input.

Confidence and context. Tab. 3(b) shows the benefit of incorporating collection-level context into prediction fusion. CAF improves over confidence-only fusion by 1.40 and 0.54 percentage points on Avg-10 and $\mathrm { { A v g } \mathrm { { - } 4 , } }$ respectively. With branch-specific soft frequencies $( r _ { k } ^ { \mathrm { b r } } )$ , combining confidence and context also outperforms either signal alone, supporting their complementary roles. Among the evaluated context estimators, shared soft frequencies $( r _ { k } ^ { \mathrm { s h } } )$ achieve the highest average accuracies: they improve Avg-10 over branch-specific statistics while maintaining comparable Avg-4, and outperform shared hard frequencies $( r _ { k } ^ { \mathrm { h a r d } } )$ on both metrics. The measured end-to-end inference overhead is only 0.61% relative to confidence-only fusion, indicating that the contextual information can be incorporated at little additional computational cost.

Prompt regularization. Tab. 3(c) reveals different generalization trade-offs for the two regularizers. The margin-only and text-only configurations are selected by Avg-4 and Avg-10, respectively. Text regularization alone increases Avg-10 from 69.043% to 69.845%, whereas margin regularization alone achieves the highest Avg-4 of 61.484% among the reported configurations. The combined setting obtains the highest Avg-10 of 69.955% and improves Avg-4 over text regularization alone, although it remains below the marginonly result. Thus, the selected combination favors crossdataset transfer while recovering part of the cross-domain performance lost with text regularization alone. The complete coefficient sweep is provided in Fig. B1.

Table 3. CAF ablations. Accuracies (%) on ImageNet (IN), ten cross-dataset targets (Avg-10), and four cross-domain targets (Avg-4). Time denotes median end-to-end inference time in seconds; Rel. is its ratio to the confidence-only baseline. ↑/↓ indicate that higher/lower is better.  
(a) Fusion weights  
(b) Confidence and context  
(c) Prompt regularization
<table><tr><td>Weights</td><td>IN↑</td><td>Avg-10↑</td><td>Avg-4↑</td></tr><tr><td> ${ \mathbf { e } } _ { V T }$ </td><td>72.624</td><td>65.426</td><td>60.589</td></tr><tr><td> $\mathbf { w _ { \mathrm { s r c } } }$ </td><td>72.590</td><td>65.812</td><td>60.671</td></tr><tr><td> $\mathbf { w } _ { \theta }$ </td><td>72.316</td><td>66.241</td><td>60.611</td></tr><tr><td> $\scriptstyle { \frac { 1 } { 4 } } \mathbf { 1 }$ </td><td>72.058</td><td>67.638</td><td>61.169</td></tr><tr><td>w(x) (CAF)</td><td>72.724</td><td>69.524</td><td>61.649</td></tr></table>

<table><tr><td>Inputs  $( h _ { k } , r _ { k } )$ </td><td>Avg-10↑</td><td> $\mathbf { A v g } { - } 4 \uparrow$ </td><td>Time (s)↓</td><td>Rel.↓</td></tr><tr><td> $( h _ { k } , 0 )$ </td><td>68.122</td><td>61.106</td><td>111.876</td><td>1.0000×</td></tr><tr><td> $( 0 , r _ { k } ^ { \mathrm { b r } } )$ </td><td>69.236</td><td>61.444</td><td>111.776</td><td>0.9991×</td></tr><tr><td> $( h _ { k } , r _ { k } ^ { \mathrm { b r } } )$ </td><td>69.405</td><td>61.641</td><td>112.592</td><td>1.0064×</td></tr><tr><td> $( h _ { k } , r _ { k } ^ { \mathrm { h a r d } } )$ </td><td>69.237</td><td>61.465</td><td>112.437</td><td>1.0050×</td></tr><tr><td> $( h _ { k } , r _ { k } ^ { \mathrm { s h } } ) \left( \mathbf { C } \mathbf { A } \mathbf { F } \right)$ </td><td>69.524</td><td>61.649</td><td>112.560</td><td>1.0061×</td></tr></table>

<table><tr><td> $\mathcal { L } _ { \mathrm { m a r g i n } } \mathcal { L } _ { \mathrm { t e x t } }$ </td><td></td><td colspan="2">IN↑ Avg-10↑ Avg-4↑</td></tr><tr><td>一</td><td></td><td>73.216</td><td>69.043 61.353</td></tr><tr><td>√</td><td></td><td>72.958</td><td>68.745 61.484</td></tr><tr><td>一</td><td>√</td><td>72.510</td><td>69.845 61.103</td></tr><tr><td>√</td><td>√</td><td>72.263</td><td>69.955 61.361</td></tr></table>

![](images/6a2c9d088fec4703c7e74d663351cff6c7250ec71cccb832458a47d717148dc7.jpg)

![](images/38b72f3cb31ab2ad7301a94416d5627bdbf3383c22421c8cb420a532c776a672.jpg)

![](images/bdd4b85c70a563f1c9244815f90fc85600497a157bbded1f6efa96637892e431.jpg)  
Figure 3. Adaptive fusion across transfer settings. (a,b) Accuracy by branch disagreement across ten datasets. Braces give differences in percentage points. (c) Mean branch weights on OxfordPets and EuroSAT. Outlines mark the largest weight in each row.

## 4.5. Analysis

Branch disagreement. Fig. 3(a,b) groups test images by the number of distinct top-1 predictions among the four branches. When all branches agree, adaptive and uniform fusion produce the same prediction. At every disagreement level under both protocols, adaptive fusion outperforms uniform fusion, with the largest gain when three distinct classes are proposed (+11.8 and +14.1 percentage points for CAF in cross-dataset transfer and CAF<sup>∗</sup> on novel classes, respectively); the gain remains positive even when all four predictions differ. A ground-truth branch oracle that selects a correct branch whenever one exists remains 19–31 points above adaptive fusion across disagreement levels, indicating that the branches already contain complementary correct predictions and that estimating which branch to trust remains the main bottleneck.

Branch preferences. Fig. 3(c) compares mean fusion weights on a natural-object benchmark (OxfordPets) and a satellite-imagery benchmark (EuroSAT). On OxfordPets, the fully prompted VT branch receives the largest weight under both protocols. On EuroSAT, the dominant branch changes across protocols: with ImageNet-trained prompts, 0T (frozen visual features with prompted text) receives the largest weight, whereas after base-class training, V 0 (prompted visual features with frozen text) leads on novel classes. The preferred branch is therefore not a fixed property of the method but depends on the transfer setting, which is why a single fixed branch or global weighting rule cannot be optimal across settings. Sec. C.2 reports the full comparison across datasets.

## 5. Conclusion

Our study suggests a clear answer to when to trust adaptation, namely neither globally nor uniformly, but locally and contextually. Contextual Adaptive Fusion (CAF) treats frozen and prompted visual and textual representations as four complementary branches, weighting them on each image by combining instance-level branch confidence with collection-level prediction frequency. With fusion coefficients fixed after source training, target-time arbitration relies solely on unlabeled target evidence, elevating the frozen model from a training-time regularizer to an active inference-time evidence source. This perspective explains why a single static balance is insufficient, as adaptive gains concentrate on images with branch disagreement while dominant branch preferences shift across transfer regimes even within the same domain. Consequently, CAF achieves SOTA cross-dataset transfer, strong robustness under ImageNet distribution shifts, and superior base-to-novel generalization without per-target tuning. The remaining oracle headroom further confirms that the four branches already contain the evidence needed for harder cases, making richer branch interactions and online context estimation promising directions for future work.

## Acknowledgments

This research was supported by the Fund of the National Natural Science Foundation of China (Grant No. 62576177), Shenzhen Science and Technology Program (QNXMB20250701090801002, JCYJ20250604184027034, JCYJ20240813114237048), Guangdong Basic and Applied Basic Research Foundation (2026A1515011435), the Fundamental Research Funds for the Central Universities (070-63263247), Youth Interdisciplinary Research Project of Nankai University (Grant No. NKQNJC202620), and the Tianjin Key Laboratory of Visual Computing and Intelligent Perception (VCIP). Additional support was provided by the “Science and Technology Yongjiang 2035” key technology breakthrough plan project (2025Z053), Chinese government-guided local science and technology development fund projects (scientific and technological achievement transfer and transformation projects) (254Z0102G), and the National Science Fund of China (Grant No. 62361166670). Computation is supported by the Supercomputing Center of Nankai University (NKSC).

## LLM Usage Statement

OpenAI Codex was used to assist with manuscript drafting and language revision, LAT<sub>E</sub>X formatting, preparation of analysis figures and literature-comparison tables, and consistency checks of references and reported results. The authors verified all citations, numerical results, scientific claims, and final content and take full responsibility for the manuscript.

## References

[1] Yassir Bendou, Omar Ezzahir, Eduardo Fernandes Montesuma, Gabriel Mahuas, Victoria Shevchenko, and Mike Gartrell. ReBaPL: Repulsive bayesian prompt learning. In CVPR, pages 39962–39971, 2026. 3, 6, 7

[2] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 - mining discriminative components with random forests. In ECCV, pages 446–461, 2014. 5, 12

[3] Mohamad Hassan N C, Divyam Gupta, Mainak Singha, Sai Bhargav Rongali, Ankit Jha, Muhammad Haris Khan, and Biplab Banerjee. OSLoPrompt: Bridging lowsupervision challenges and open-set domain generalization in CLIP. In CVPR, pages 10110–10120, 2025. 2

[4] Zhenyuan Chen, Lingfeng Yang, Shuo Chen, Zhaowei Chen, Jiajun Liang, and Xiang Li. Revisiting prompt pretraining of vision-language models. arXiv preprint arXiv:2409.06166, 2024. 1

[5] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In CVPR, pages 3606–3613, 2014. 5, 6, 12

[6] Fangming Cui, Jan Fong, Rongfei Zeng, Xinmei Tian, and Jun Yu. A similarity paradigm through textual regularization without forgetting. In AAAI, pages 16100–16108, 2025. 3

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 5, 12

[8] Mohammad Mahdi Derakhshani, Enrique Sanchez, Adrian Bulat, Victor G. Turrisi da Costa, Cees G. M. Snoek, Georgios Tzimiropoulos, and Brais Martinez. Bayesian prompt learning for image-language model generalization. In ICCV, pages 15237–15246, 2023. 3, 6

[9] Kun Ding, Haojian Zhang, Qiang Yu, Ying Wang, Shiming Xiang, and Chunhong Pan. Weak distribution detectors lead to stronger generalizability of vision-language prompt tun ing. In AAAI, pages 1528–1536, 2024. 2, 3

[10] Tong Ding, Wanhua Li, Zhongqi Miao, and Hanspeter Pfister. Tree of attributes prompt learning for vision-language models. In ICLR, pages 50495–50514, 2025. 2, 7

[11] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPRW, pages 178–178, 2004. 5, 12

[12] Yuncheng Guo and Xiaodong Gu. MMRL: Multi-modal representation learning for vision-language models. In CVPR, 2025. 1, 3, 6

[13] Yuncheng Guo and Xiaodong Gu. MMRL++: Parameterefficient and interaction-aware representation learning for vision-language models. IJCV, 134:11, 2026. 1, 3, 6, 7

[14] Fusheng Hao, Fengxiang He, Fuxiang Wu, Tichao Wang, Chengqun Song, and Jun Cheng. Task-aware clustering fo prompting vision-language models. In CVPR, pages 14745– 14755, 2025. 2, 6, 7

[15] Patrick Helber, Benjamin Bischke, Andreas R. Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE JSTARS, 12:2217–2226, 2019. 5, 6, 12

[16] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Lixuan Zhu, Samyak Parajuli, Mike Guo, Dawn Xiaodong Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pages 8320–8329, 2021. 5, 12

[17] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Vi sual prompt tuning. In ECCV, pages 709–727. Springer, 2022. 2

[18] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. MaPLe: Multi-modal prompt learning. In CVPR, pages 19113– 19122, 2023. 1, 2

[19] Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Self-regulating prompts: Foundational model adaptation without forgetting. In ICCV, pages 15190–15200, 2023. 1, 2, 3, 6, 7

[20] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In ICCVW, pages 554–561, 2013. 5, 12

[21] Zheng Li, Xiang Li, Xinyi Fu, Xin Zhang, Weiqiang Wang, Shuo Chen, and Jian Yang. Promptkd: Unsupervised prompt distillation for vision-language models. In CVPR, pages 26617–26626, 2024. 3

[22] Zheng Li, Yibing Song, Ming-Ming Cheng, Xiang Li, and Jian Yang. Advancing textual prompt learning with anchored attributes. In ICCV, pages 3618–3627, 2025. 2

[23] Liangchen Liu, Nannan Wang, Xi Yang, Xinbo Gao, and Tongliang Liu. Surrogate prompt learning: Towards efficient and diverse prompt learning for vision-language models. In ICML, pages 39755–39773. PMLR, 2025. 3, 6, 7

[24] Qi Ma, Chen-Yang Wang, Dehong Gao, and Deng-Ping Fan. AlignedNorm: Prompting vision-language models via coupled prompt field. In ICML, pages 84989–85009. PMLR, 2026. 1, 3, 6

[25] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, abs/1306.5151, 2013. 5, 6, 12

[26] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In ICVGIP, pages 722–729, 2008. 5, 12

[27] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In CVPR, pages 3498–3505. IEEE, 2012. 5, 12

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 1, 2, 7

[29] Umaima Rahman, Mohammad Yaqub, and Dwarikanath Mahapatra. DiMPLe - disentangled multi-modal prompt learning: Enhancing out-of-distribution alignment with invariant and spurious feature separation. In ICCV, pages 1634–1643, 2025. 2

[30] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In ICML, pages 5389–5400, 2019. 5, 12

[31] Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alex Smola, and Xu Sun. Prompt pretraining with twenty-thousand classes for open-vocabulary visual recognition. In NeurIPS, pages 12569–12588, 2023. 1, 3

[32] Cristian Rodriguez-Opazo, Ehsan Abbasnejad, Damien Teney, Hamed Damirchi, Edison Marrese-Taylor, and Anton van den Hengel. Synergy and diversity in CLIP: Enhancing performance through adaptive backbone ensembling. In ICLR, 2025. 2, 3

[33] Shuvendu Roy and Ali Etemad. Consistency-guided prompt learning for vision-language models. In ICLR, 2024. 1, 3, 4, 6, 7, 12

[34] Maoyuan Shao, Yutong Gao, Xinyang Huang, Lijuan Sun, Guoshun Nan, and Chuang Zhu. Capt: Confusion-aware prompt tuning for reducing vision-language misalignment. In CVPR, pages 3154–3164, 2026. 3

[35] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test time prompt tuning for zero-shot generalization in vision language models. In NeurIPS, pages 14274–14289, 2022. 2, 3, 6

[36] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, abs/1212.0402, 2012. 5, 6, 12

[37] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Argue: Attribute-guided prompt tuning for vision-language models. In CVPR, pages 28578–28587, 2024. 2, 6

[38] Haohan Wang, Songwei Ge, Eric P. Xing, and Zachary Chase Lipton. Learning robust global representations by penalizing local predictive power. In NeurIPS, pages 3256–3266, 2019. 5, 12

[39] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models. In CVPR, pages 7959–7971, 2022. 3

[40] Ge Wu, Xin Zhang, Zheng Li, Zhaowei Chen, Jiajun Liang, Jian Yang, and Xiang Li. Cascade prompt learning for vision-language model adaptation. In ECCV, pages 304– 321. Springer, 2024. 3

[41] Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, pages 3485–3492, 2010. 5, 6, 12

[42] Xi Yang, Yuanrong Xu, Weigang Zhang, Guangming Lu, David Zhang, and Jie Wen. Prompt tuning for CLIP on the pretrained manifold. In ICML, pages 146554–146578. PMLR, 2026. 1, 2, 3, 4, 6, 7, 12

[43] Hantao Yao, Rui Zhang, and Changsheng Xu. Visual language prompt tuning with knowledge-guided context optimization. In CVPR, pages 6757–6767, 2023. 3

[44] Maxime Zanella, Benoˆıt Gerin, and Ismail B Ayed. Boost-´ ing vision-language models with transduction. In NeurIPS, pages 62223–62256, 2024. 3

[45] Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-Adapter: Training-free adaption of CLIP for few-shot classification. In ECCV, pages 493–510. Springer, 2022. 3

[46] Hao Zheng, Shunzhi Yang, Zhuoxin He, Jinfeng Yang, and Zhenhua Huang. Hierarchical cross-modal prompt learning for vision-language models. In ICCV, pages 1891–1901, 2025. 2, 6, 7

[47] Zhaoheng Zheng, Jingmin Wei, Xuefeng Hu, Haidong Zhu, and Ram Nevatia. Large language models are good prompt learners for low-shot image classification. In CVPR, pages 28453–28462, 2024. 2

[48] Jingtao Zhou, Xirui Kang, Feiyang Huang, and Lai-Man Po. SpecPL: Disentangling spectral granularity for prompt learning. In ICML, 2026. 3, 6, 7

[49] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In CVPR, pages 16816–16825, 2022. 1, 2, 6, 7

[50] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. IJCV, 130(9):2337–2348, 2022. 1, 2, 6, 7

[51] Beier Zhu, Yulei Niu, Yucheng Han, Yue Wu, and Hanwang Zhang. Prompt-aligned gradient for prompt tuning. In ICCV, pages 15659–15669, 2023. 3

[52] Beier Zhu, Jiequan Cui, and Hanwang Zhang. Robust fine-tuning of zero-shot models via variance reduction. In NeurIPS, 2024. 2, 3

Table A1. Dataset statistics. Training counts exclude validation images and precede few-shot sampling and base/novel partitioning. A dash indicates a target-only dataset.
<table><tr><td>Dataset</td><td>Task / domain</td><td>Classes</td><td>Train</td><td>Test</td></tr><tr><td>Standard benchmarks</td><td></td><td></td><td></td><td></td></tr><tr><td>ImageNet [7]</td><td>Objects</td><td>1,000</td><td>1,281,167</td><td>50,000</td></tr><tr><td>Caltech101 [11]</td><td>Objects</td><td>100</td><td>4,128</td><td>2,465</td></tr><tr><td>OxfordPets [27]</td><td>Fine-grained</td><td>37</td><td>2,944</td><td>3,669</td></tr><tr><td>StanfordCars [20]</td><td>Fine-grained</td><td>196</td><td>6,509</td><td>8,041</td></tr><tr><td>Flowers102 [26]</td><td>Fine-grained</td><td>102</td><td>4,093</td><td>2,463</td></tr><tr><td>Food101 [2]</td><td>Fine-grained</td><td>101</td><td>50,500</td><td>30,300</td></tr><tr><td>FGVCAircraft [25]</td><td>Fine-grained</td><td>100</td><td>3,334</td><td>3,333</td></tr><tr><td>SUN397 [41]</td><td>Scenes</td><td>397</td><td>15,880</td><td>19,850</td></tr><tr><td>DTD [5]</td><td>Textures</td><td>47</td><td>2,820</td><td>1,692</td></tr><tr><td>EuroSAT [15]</td><td>Remote sensing</td><td>10</td><td>13,500</td><td>8,100</td></tr><tr><td>UCF101 [36]</td><td>Actions</td><td>101</td><td>7,639</td><td>3,783</td></tr><tr><td colspan="5">ImageNet variants</td></tr><tr><td>ImageNetV2 [30]</td><td>Natural shift</td><td>1,000</td><td></td><td>10,000</td></tr><tr><td>ImageNet-Sketch [38] Sketches</td><td></td><td>1,000</td><td></td><td>50,889</td></tr><tr><td>ImageNet-A</td><td>Adversarial</td><td>200</td><td></td><td>7,500</td></tr><tr><td>ImageNet-R [16]</td><td>Renditions</td><td>200</td><td></td><td>30,000</td></tr></table>

## A. Datasets and Experimental Details

## A.1. Datasets and Evaluation Protocols

We consider eleven standard image-classification benchmarks and four distribution-shifted ImageNet variants, summarized in Tab. A1. These datasets cover general object recognition, fine-grained recognition, scene classification, texture recognition, remote sensing, and action recognition. We use RGB images for EuroSAT and extracted video frames for UCF101.

For cross-dataset evaluation, ImageNet provides the labeled source data, and the remaining ten standard datasets serve as targets. Cross-domain evaluation uses ImageNetV2, ImageNet-Sketch, ImageNet-A, and ImageNet-R. ImageNetV2 and ImageNet-Sketch retain the 1,000 ImageNet classes, whereas ImageNet-A and ImageNet-R contain subsets of 200 classes. ImageNet-S in Tab. 1 denotes ImageNet-Sketch. Avg-10 and Avg-4 average accuracies over the ten cross-dataset and four cross-domain targets, respectively, excluding source ImageNet accuracy.

For base-to-novel evaluation, each of the eleven standard datasets is partitioned into disjoint base and novel class sets. Prompts are trained using labeled images from base classes and evaluated separately on base and novel classes, with novel classes unseen during training. We report base accuracy, novel accuracy, and their harmonic mean (HM) to assess the balance between adaptation and generalization.

Table A2. Dataset-specific loss weights used by $\mathrm { C A F ^ { * } }$ for baseto-novel generalization in Tab. 2.
<table><tr><td>Dataset(s)</td><td>λmargin</td><td> $\lambda _ { \mathrm { t e x t } }$ </td></tr><tr><td>ImageNet, EuroSAT</td><td>1</td><td>2.5</td></tr><tr><td>Caltech101</td><td>0.5</td><td>0</td></tr><tr><td>OxfordPets</td><td>2</td><td>20</td></tr><tr><td>StanfordCars, UCF101</td><td>0</td><td>2.5</td></tr><tr><td>Flowers102</td><td>4</td><td>2.5</td></tr><tr><td>Food101</td><td>4</td><td>40</td></tr><tr><td>FGVCAircraft</td><td>2</td><td>0</td></tr><tr><td>SUN397</td><td>2</td><td>5</td></tr><tr><td>DTD</td><td>4</td><td>5</td></tr></table>

## A.2. Implementation Details

Backbone and prompts. We use CLIP ViT-B/16 with eight learnable prompt tokens at each of the 12 layers in both the visual and text encoders. The pretrained encoders and logit scale remain frozen. Following ManiPT and CoPrompt [33, 42], we construct frozen textual prototypes from fixed GPTgenerated class descriptions. Throughout this section, lossweight pairs are given in the order $\left( \lambda _ { \mathrm { m a r g i n } } , \lambda _ { \mathrm { t e x t } } \right)$

Source training. For cross-dataset and cross-domain evaluation, we train on 16-shot ImageNet for 5 epochs. Each minibatch contains 32 source images with four augmented views per image at a resolution of $2 2 4 \times 2 2 4$ . We optimize the prompts with SGD at a peak learning rate of 0.01 and the fusion coefficients $\alpha , \beta$ with Adam at a peak learning rate of $1 0 ^ { - 3 }$ , using a one-epoch warm-up followed by cosine decay. We use the same loss weights (1, 40) and the same trained model for all ten cross-dataset targets and four ImageNet variants. The fusion loss has unit weight. During training, we maintain a shared exponential moving average of branch prediction probabilities with a decay of 0.99. At inference, we fix all learned parameters and recompute the shared statistics from the full unlabeled test set of each target dataset. Tab. 1 reports mean accuracies and sample standard deviations over three seeds.

Base-to-novel training. We train a separate model on each dataset using a fixed base/novel class split and 16 labeled images per base class. On the ten downstream datasets, we train for 20 epochs using minibatches of four images with four augmented views each (16 views per update) and a peak prompt learning rate of 0.0025. On ImageNet, the split contains 500 base and 500 novel classes, giving 8,000 base training images. We use 32 images per minibatch with four views each (128 views per update) and a peak prompt learning rate of 0.01. The 10- and 20-epoch ImageNet runs are trained independently from scratch. Tab. 2 uses the final checkpoint of the 20-epoch run for every dataset. Across all eleven datasets, prompt SGD uses momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 }$ , and a first-epoch learning rate of $1 0 ^ { - 5 }$ followed by cosine decay from the dataset-specific peak. We retain the Adam optimizer and unit fusion-loss weight used in source training. At evaluation, we use batches of 64 images and compute the shared statistics separately over the full unlabeled base and novel test splits.

Table B1. Individual branches and prediction fusion. Branch and uniform-fusion accuracies (%) share one historical checkpoint; CAF repeats the learned-coefficient, three-seed means from Tab. 1.
<table><tr><td>Prediction</td><td>ImageNet↑</td><td> $\mathbf { A v g - } 1 0 \uparrow$ </td><td> $\mathrm { \ A v g { - } } 4 \uparrow$ </td></tr><tr><td>00</td><td>69.642</td><td>66.873</td><td>60.024</td></tr><tr><td>V0</td><td>70.882</td><td>66.272</td><td>59.871</td></tr><tr><td>OT</td><td>71.220</td><td>66.766</td><td>60.762</td></tr><tr><td>VT</td><td>72.184</td><td>65.884</td><td>60.487</td></tr><tr><td>Uniform fusion</td><td>71.616</td><td>67.517</td><td>60.962</td></tr><tr><td>CAF (ours)</td><td>72.263</td><td>69.955</td><td>61.361</td></tr></table>

For CAF, the shared loss weights (4, 2.5) give the highest harmonic mean of the ten-dataset average base and novel accuracies among the 42 configurations in Fig. B2. We use this pair on all eleven datasets, including ImageNet. For $\mathrm { C A F ^ { * } }$ , we allow the loss weights to vary by dataset and select the combination that maximizes the same harmonic mean over the ten downstream datasets. On ImageNet, we select (1, 2.5) by fused base accuracy among the three configurations evaluated for 20 epochs. Tab. A2 lists the resulting dataset-specific settings.

For both CAF variants, we use fusion on novel classes for all datasets. On base classes, we use fusion on ImageNet and the V T branch on the other ten datasets. Tab. 2 reports three-seed mean base and novel accuracies and computes HM from these means.

## B. Additional Ablation Studies

## B.1. Individual Branches and Fusion

Tab. B1 compares the four individual prediction branches with uniform fusion using cached logits from the same prompt checkpoint. Among the individual branches, VT achieves the highest ImageNet accuracy but the lowest Avg-10 accuracy, indicating that stronger source performance does not necessarily translate into better cross-dataset generalization. The frozen 00 branch achieves the highest Avg-10, whereas 0T performs best on Avg-4, revealing different strengths across the evaluation settings.

Uniform fusion exceeds every individual branch on both target averages, reaching 67.517% on Avg-10 and 60.962% on Avg-4. These results support the complementarity of frozen and prompted predictions and motivate retaining multiple branches for transfer. The complete CAF results from Tab. 1 are included as a reference; these use learned fusion coefficients and three-seed averaging, whereas the individual-branch and uniform-fusion controls share one checkpoint.

Table B2. Unlabeled statistics: scope and budget. Single-seed accuracies (%) with a fixed prompt checkpoint and fusion coefficients. N denotes the collection size; budgeted references exclude the query image.
<table><tr><td>Statistics</td><td>Available images</td><td> $\mathbf { A v g - } 1 0 \uparrow$ </td><td> $\mathrm { \ A v g { - } } 4 \uparrow$ </td></tr><tr><td>Full collection</td><td>All N</td><td>69.524</td><td>61.649</td></tr><tr><td>Leave-one-out</td><td>N-1</td><td>69.530</td><td>61.650</td></tr><tr><td>Two-fold cross-fit</td><td>Other fold  $\left( \approx N / 2 \right)$ </td><td>69.532</td><td>61.567</td></tr><tr><td>Budget: 256</td><td> $\leq \operatorname* { m i n } ( 2 5 6 , N )$ </td><td>68.917</td><td>60.597</td></tr><tr><td>Budget: 1,024</td><td> $\leq \operatorname* { m i n } ( 1 0 2 4 , N )$ </td><td>69.210</td><td>61.056</td></tr><tr><td>Budget: 4,096</td><td> $\leq \operatorname* { m i n } ( 4 0 9 6 , N )$ </td><td>69.462</td><td>61.462</td></tr><tr><td>Online cumulative</td><td>Past batches</td><td>69.343</td><td>61.526</td></tr><tr><td>Online EMA</td><td>Past batches</td><td>69.287</td><td>61.543</td></tr><tr><td>Batch leave-one-out</td><td> $\leq 3 1$  others</td><td>67.809</td><td>60.735</td></tr></table>

## B.2. Scope and Budget of Unlabeled Context

Tab. B2 examines how the scope and amount of unlabeled data affect contextual fusion. We keep the prompt checkpoint and fusion coefficients fixed and vary only the construction of the shared soft-frequency table. Full-collection statistics use all images in the target dataset. Leave-one-out statistics exclude the query image, while two-fold crossfitting estimates each image’s context from the opposite fold. Budgeted variants restrict the number of reference images and also exclude the query.

Excluding the query image leaves performance virtually unchanged: leave-one-out statistics achieve 69.530% on Avg-10 and 61.650% on Avg-4, compared with 69.524% and 61.649% using the full collection. Two-fold crossfitting also retains comparable performance. Thus, the observed benefit does not depend on including an image in its own context estimate. Increasing the reference budget from 256 to 4,096 images improves both target averages, with the latter remaining within 0.19 percentage points of full-collection performance on both metrics.

We additionally consider online cumulative and EMA statistics, which use only previously observed batches. Both predict before incorporating the current batch, using a batch size of 32 and an EMA decay of 0.99 where applicable. These online variants remain within 0.24 percentage points of full-collection performance on both averages in the recorded evaluation order. In contrast, estimating context solely from the other images in the current batch reduces Avg-10 to 67.809% and Avg-4 to 60.735%. Together, these results indicate that accumulating context across images is beneficial, while useful contextual estimates can still be obtained with limited reference data or sequential access.

a  
![](images/a3e98f554be66a6b7652b90a30c1921295de6a34be84ea0094ea69a3c921cce3.jpg)

b  
![](images/09bbe931f081cab74edf4ebe5e3c5cedb4cc4151ba7bf04b9982b23e612bf67f.jpg)  
Figure B1. Regularization-weight results. Accuracy (%) for (a) cross-dataset and (b) cross-domain transfer, with separate color scales. Results combine historical fixed-coefficient and learned-coefficient runs. Outlined cells at (1, 40) report the three-seed means from Tab. 1; other cells report individual runs.

a  
![](images/f4b756b81ff96bbaef9f19879ce7db51c819f1ee605803e8b13f53178bc8be7e.jpg)

b  
![](images/35ed2d192d3d377bc2581328d25f29f3ad63a564e6a999b5997779efe2d6bb02.jpg)  
Figure B2. Regularization weights for base-to-novel generalization. Accuracy (%) averaged over ten datasets and three seeds: (a) base classes using the VT branch and (b) novel classes using fusion, with separate color scales. Outlined cells mark the shared (4, 2.5) setting used for CAF in Tab. 2.

## B.3. Regularization-Weight Results

Figs. B1 and B2 examine 42 combinations of regularization weights, with $\lambda _ { \mathrm { m a r g i n } } ~ \in ~ \{ 0 , 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ and $\lambda _ { \mathrm { t e x t } } \in \{ 0 , 2 . 5 , 5 , 1 0 , 2 0 , 4 0 , 8 0 \}$ . The grid in Fig. B1 combines runs with fixed and learned fusion coefficients. The outlined (1, 40) configuration reports the three-seed CAF means from Tab. 1, while the other cells report individual runs.

In Fig. B1, the (1, 40) configuration adopted in Tab. 1 achieves 69.96% on the ten-dataset cross-dataset average (Avg-10), within 0.07 percentage points of the grid maximum (70.02%), alongside 61.36% on the four-variant crossdomain average (Avg-4). Shifting to (4, 2.5) raises Avg-4 to its peak of 61.73% while reducing Avg-10 to 69.02%. This trade-off is consistent with the distinct nature of the two evaluation tracks. When transferring from the 1,000- class ImageNet source to new target vocabularies (Avg-10), strong textual anchoring $( \lambda _ { \mathrm { t e x t } } = 4 0 )$ may help preserve generalizable class semantics. Under visual shifts among classes drawn from the source vocabulary (Avg-4), the highest accuracy occurs with stronger regularization of reliable frozen classification margins $( \lambda _ { \mathrm { m a r g i n } } = 4 )$

Fig. B2 reveals a complementary trade-off between baseclass fitting and novel-class generalization in the few-shot base-to-novel setting, where we evaluate the VT branch on base classes and adaptive fusion on novel classes across ten datasets and three seeds. Novel-class accuracy consistently peaks at a stronger margin weight of $\lambda _ { \mathrm { m a r g i n } } = 4$ across all tested text weights, whereas base-class accuracy peaks at a milder (1, 2.5) setting. This pattern is consistent with margin regularization limiting visual overfitting to the few labeled examples per base class. Within the $\lambda _ { \mathrm { m a r g i n } } = 4$ regime, increasing $\lambda _ { \mathrm { t e x t } }$ from 2.5 to 5 yields a small novel-class gain (80.15% to 80.20%) at the cost of a larger drop in base accuracy (86.52% to 86.30%). Consequently, the (4, 2.5) setting secures near-optimal novelclass transfer while preserving strong base-class discrimination, achieving the highest harmonic mean of the ten-dataset base and novel averages (83.21%) across all 42 configurations.

## C. Analysis Experiments

## C.1. Interventions on Dynamic Weight Assignment

To examine the importance of matching fusion weights to individual images, Tab. C1 modifies weight assignments while keeping the branch logits and frequency statistics fixed. These interventions use the Branch-soft variant as the unmodified baseline; Shared-soft CAF is included separately for reference. All controls operate within each dataset without retraining or using target labels.

We consider four interventions. Dataset-mean weights replace each image’s weights with the average vector estimated from the opposite fold of a two-fold split. Predictedclass mean weights instead average within groups defined by the top-1 prediction of uniform fusion, using the opposite fold and falling back to its global mean when a group is absent. Shuffle across all images randomly permutes complete four-dimensional weight vectors, preserving their distribution while disrupting their correspondence with images. Shuffle within disagreement samples applies this permutation only to images whose branches predict different classes. Both shuffling experiments average results over 20 permutations.

Table C1. Dynamic weight assignment interventions. Interventions use Branch-soft weights from one checkpoint; shuffled rows average 20 permutations. Shared-soft CAF is a separate reference. Accuracies are percentages.
<table><tr><td>Weight assignment</td><td>Avg-10↑</td><td>Avg-4↑</td></tr><tr><td>Original (Branch-soft)</td><td>69.405</td><td>61.641</td></tr><tr><td>Dataset-mean weights</td><td>67.670</td><td>61.097</td></tr><tr><td>Predicted-class mean weights</td><td>68.058</td><td>61.137</td></tr><tr><td>Shuffle across all images</td><td>67.058</td><td>60.827</td></tr><tr><td>Shuffle within disagreement samples</td><td>67.071</td><td>60.666</td></tr><tr><td>Shared-soft CAF (reference)</td><td>69.524</td><td>61.649</td></tr></table>

The original assignment outperforms all four controls on both target averages. Replacing image-specific weights with dataset means reduces Avg-10 and Avg-4 by 1.74 and 0.54 percentage points, respectively. Conditioning the mean weights on predicted classes partially recovers performance but remains below the original assignment. Shuffling across all images causes larger reductions of 2.35 and 0.81 percentage points, despite preserving the distribution of weight vectors. Performance also decreases when shuffling is restricted to disagreement samples. These results support the value of assigning weights to the images for which they were computed, beyond retaining dataset-level or predictedclass-level weighting preferences.

## C.2. Dataset-Wise Fusion Weights

Fig. C1 shows how the same ImageNet-trained CAF model distributes its mean fusion weights across evaluation benchmarks. On ImageNet and its four cross-domain variants, the visually prompted branches V0 and VT consistently receive the two largest shares (29.7%–31.0% for V0 and 27.3%–30.7% for V T). Among the variants, the share of the frozen 00 branch ranges from 19.6% on ImageNetV2 to 22.9% on ImageNet-A. Across the ten cross-dataset targets, three patterns emerge. First, the jointly prompted V T branch leads on Flowers102 (36.4%), OxfordPets (32.3%), FGVCAircraft (32.0%), and the texture benchmark DTD (30.6%). Second, V0, which pairs adapted visual features with frozen text prototypes, receives the largest share on SUN397 (32.7%), UCF101 (30.8%), StanfordCars (30.5%), Food101 (29.0%), and Caltech101 (27.5%). On these datasets, the leading branch thus retains the pretrained text representation of each target class. Third, the preference reverses on EuroSAT satellite imagery, where V 0 receives 15.8% and 0T receives 37.1%. Even with fixed model parameters, the balance between visual and textual adaptation therefore varies across target domains.

![](images/fae2c6d3d21ea1bce62ada2ddb3faf185488cca6d4812c5bf4edcb657b3fcf4b.jpg)  
Figure C1. Mean CAF fusion weights for cross-dataset and cross-domain evaluation. Values (%) are averaged over test images in each dataset, with 25% corresponding to uniform weighting.

Fig. C2 reports mean branch weights for CAF<sup>∗</sup> on novel classes after base-class training on each of the eleven datasets. Here the visual prompts are learned within the target visual domain, while the textual prompts are learned using only the base categories. Across all eleven benchmarks, V0 and VT jointly receive more than half of the total weight (53.1%–85.0%). The text-only adapted branch 0T receives 4.8%–21.8% and has the smallest share on nine datasets. On OxfordPets and FGVCAircraft, 00 has the smallest share. The concentration of the weights also varies by task. On Caltech101, all four shares are close to the uniform weight of 25% (21.0%–27.8%). On DTD, by contrast, V0 alone receives 55.8%, compared with 4.8% for 0T. ImageNet (36.7% versus 14.7%) and SUN397 (36.0% versus 14.7%) show the same preference for in-domain visual prompts paired with frozen text prototypes. EuroSAT illustrates how the leading branch can change on the same benchmark across transfer settings. With ImageNet-trained prompts, 0T leads at 37.1% in Fig. C1. With prompts trained on EuroSAT base classes and evaluated on its novel classes, V0 leads at 32.9%, while 0T receives 13.7% in Fig. C2.

![](images/4854f2e6148c8a3a181cdb6f57f3deb26c13a4816a3b72105de45eab76ab504a.jpg)  
Figure C2. Mean fusion weights on novel classes across eleven datasets. CAF<sup>∗</sup> uses the dataset-specific loss weights from Tab. 2. Values are mean branch weights (%) on each novel-class test split. Uniform weighting assigns 25% to each branch.