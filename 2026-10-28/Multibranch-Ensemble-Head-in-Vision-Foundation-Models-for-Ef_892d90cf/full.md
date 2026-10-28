# Multibranch Ensemble Head in Vision Foundation Models for Eficient Cucumber Phenological Stage Classification

Kostiantyn Khabarlak

Dnipro University of Technology, Ukraine

Khabarlak.K.S@nmu.one

## Abstract

Vision Foundation Models (VFMs) pretrained with selfsupervised learning, such as DINOv2, provide highly transferablefeatures, yet adapting them to small, domainshifted datasets is still challenging. We propose a regularizing multibranch ensemble head: a training and inference scheme that pairs low- and high-capacity branches, samples a single branch stochastically at each training step, and averages both branches in probability space at inference. The design adds negligible parameters and essentially no inference overhead over a single heavy head, while improving the accuracy-eficiency trade-of under data-scarce conditions. Accurate classification of plant growth stages is critical for precision agriculture, enabling timely decisions on fertilization, irrigation, and harvesting. Therefore, as a challenging practical case we use cucumber phenological stage classification, in which flowers and fruits are frequently occluded by foliage, and stage transitions manifest primarily as subtle changes in leaf morphology. The NALEF Food Safety dataset contains images of five cucumber growth stages captured under diverse real-world conditions. On it we evaluate a broad range of backbones, including DINOv2 and DINOv3 ViT and ConvNeXt variants, against strongly supervised counterparts (ResNet, EficientNet, ViT). Theproposed DINOv2 ViT-S with multibranch head achieves a Macro F1 of0.978 and Cohen’s Kappa of 0.971, matching the performance of models more than ten times larger in parameters and GFlops, while requiring only 12.2 GFlops and 22.3M pa-

rameters, making it deployable on smartphones and edge devicesfor phenological stage classification.

Keywords: Plant growth stage classification, Vision Foundation Models, Self-supervised learning, DINOv2, DINOv3, Regularization, Transfer learning, Cucumber phenology.

## 1. Introduction

The timing and progression of plant developmental phases are highly dependent on environmental factors, particularly temperature, precipitation, and photoperiod [1]. Accurate classification of plant growth stages is critical for optimizing agricultural management, including decisions on fertilization, herbicide application, and harvesting. Automating this process reduces reliance on laborintensive and subjective manual analysis, ultimately supporting higher and more consistent crop yields.

Monitoring of plant growth can be performed using ground-level cameras, unmanned aerial vehicles (UAVs), or remote-sensing satellite imagery.

Remote-sensing satellite imagery can be used to estimate growth stages of large farmlands. For instance, Sentinel-2 satellites provide public access to multi-spectral imagery, including red-edge bands specifically designed for vegetation analysis [2]. However, this approach is unsuitable for use in cloudy conditions or in greenhouses, and its low spatial resolution leads to poor performance at early growth stages when the canopy does not fully cover the field, or for small fields.

UAVs and stationary cameras ofer much greater flexibility and precision. They can be used regardless of weather conditions and inside greenhouses. Additionally, smartphones enable rapid, on-site plant growth evaluation by non-experts. Camera-based plant growth monitoring is often formulated as a classification task [3], [4].

In particular, automated classification of cucumber growth stages from visual data remains an open problem despite the crop’s considerable agronomic significance. While computer vision methods have been applied to cucumbers across several related tasks including disease detection [5], fruit localization for robotic harvesting [6], quality grading [7], and morphological phenotyping [8], the problem of growth stage classification has received comparatively little attention in the literature. This task poses unique challenges: flowers and fruits are frequently occluded by foliage, and cucumber fruit shares near-identical coloration with the surrounding canopy. As a result, transitions between intermediate stages manifest primarily as subtle changes in leaf morphology.

Vision-based plant monitoring is typically performed by fine-tuning a backbone trained with strong supervision on a large labeled dataset. However, recent works have shown that backbones trained with self-supervised objectives ofer far more universal features that can be adapted to diverse downstream tasks using a simple linear probe [9], [10]. These Vision Foundation Models (VFMs) have demonstrated applicability across object classification, detection, tracking, scene depth estimation, and many other tasks; yet their use for plant growth stage classification remains comparatively underexplored.

In this work we study how to adapt self-supervised

VFMs to small, domain-shifted image datasets through the design of the classification head. We introduce a regularizing multibranch ensemble head: a general training and inference scheme in which a low-capacity linear branch and a higher-capacity MLP branch are attached to the backbone. A single branch is sampled stochastically at each training step, and the two branches are averaged in probability space at inference. This yields an in-model ensemble that regularizes head training under data scarcity while adding negligible parameters and essentially no inference cost. We use cucumber phenological stage classification as a challenging fine-grained case study: on the recently introduced NALEF Food Safety [11] dataset, the proposed head substantially improves the accuracy-eficiency tradeof over the DINOv2 ViT-L fine-tuning approach previously reported for this dataset [12].

Our main contributions are as follows:

1. We propose the regularizing multibranch ensemble head, a lightweight, backbone-agnostic scheme for training classification heads on top of Vision Models under data-scarce conditions, combining stochastic per-step branch sampling with probability-space aggregation at inference.

2. Using a cucumber phenological stage classification dataset, we systematically compare a broad range of self-supervised (DINOv2, DINOv3) and strongly supervised backbones, and show that self-supervised pretraining is particularly beneficial in the smallmodel regime despite the large domain gap between its pre-training data and plant imagery.

3. The resulting DINOv2 ViT-S model substantially improves over existing solutions on the NALEF Food Safety dataset at a small fraction of its computational cost, requiring few GFlops and parameters and remaining deployable on smartphones and edge devices.

## 2. Literature Overview

Fine-tuning from a strongly pretrained backbone is the standard approach for training deep neural networks on downstream tasks such as plant growth stage classification. The authors of [3] collect a strawberry growth dataset and report that a fine-tuned EficientNet-B7 achieves 83.7% classification accuracy. In [13], the authors highlight the importance of camera-based plant growth monitoring as (1) such systems are low-cost and (2) the collected data is crucial for nutrient management and harvesting scheduling; they develop a YOLOv8-based detection system for multi-plant basil growth tracking. In [4], a ResNet-50- based butterhead lettuce growth stage classification system is presented.

Cucumber-specific research is an active area. [14] investigate the relationship between cucumber root traits and crop yield to support the breeding of high-yielding, stresstolerant cultivars. Cucumber production further benefits from automated harvesting systems. Accurate detection of harvest-ready cucumbers remains challenging due to the visual similarity between fruits and surrounding foliage, as well as frequent occlusions. To address this problem, [6] propose a lightweight YOLO-based detection network tailored for cucumber harvesting scenarios. Quality assessment has also been explored using deep learning, with [7] introducing a CNN-based framework for classifying cucumber fruits into defective, curved, and straight/sound categories. For phenotyping applications, [8] present a lightweight keypoint detection model that enables automated extraction of fruit phenotypic traits. [15] propose a pest classification method applicable to diferent plant species, including cucumber. Closest to our task, [12] finetune a single DINOv2 ViT-L backbone with progressive layer unfreezing for cucumber growth stage classification.

In contrast to strongly supervised approaches, large language models (LLMs) have demonstrated remarkable progress through self-supervised learning, typically by training on large text corpora with a next-token prediction objective. This approach enables the network to learn strong linguistic representations from unlabeled data.

Vision Transformers have successfully adapted the concept of tokens to images, often surpassing convolutional neural networks. Initially, however, they were still trained with strong supervision, as the design of efective selfsupervised objectives for vision is more challenging, and the notion of a “next image token” is less self-evident.

To pretrain a vision backbone with self-supervision, several pretext tasks have been proposed: image recolorization, prediction of image transformations, and others. DINOv2 uses masked image modeling combined with a feature-consistency loss between teacher and student networks observing diferent crops of the same image. A large Vision Transformer (1.1B parameters [9]) is trained and then distilled into several smaller ViTs. This approach was later extended by DINOv3 [10], which identified limitations in DINOv2’s architecture scaling under extended training and proposed a Gram loss, among other improvements. The resulting 7B model achieves state-of-the-art performance on many downstream tasks, and the authors distill it into both ViT and ConvNeXt variants. Models trained in this fashion are commonly referred to as Vision Foundation Models, as the rich features they produce transfer broadly across diverse downstream tasks.

The self-supervised training objective allows the collection of large unlabeled datasets for training. For instance, the authors of DINOv2 assembled the LVD-142M dataset comprising 142 million images without any labels, including ImageNet-22K and many other sources. Most of these are not plant-focused; while the authors include the small Flowers dataset, the larger nature-focused iNaturalist dataset is absent from the training set [9]. The DINOv3 authors introduce the LVD-1689M dataset (1,689 million Instagram photos). While substantially larger, this dataset likely entails a greater domain gap than DINOv2’s websourced collection. Despite the domain gap, recent works have shown that self-supervised methods can be successfully adapted to plant-related tasks [16], [17].

Despite these advances, most existing approaches to plant growth stage classification still rely on strongly supervised backbones, limiting their transferability to specialized agricultural domains with scarce labeled data. Furthermore, despite its agronomic importance, cucumber growth stage classification has received only limited attention in the literature, with VFM-based solutions for this specific task explored only in a single large-backbone configuration [12]. Finally, the task stands to benefit from additional in-model regularization, as classification must rely primarily on subtle leaf morphology cues due to frequent occlusion of flowers and fruits; the potential of lightweight ensemble head designs to improve the accuracy-eficiency trade-of in this setting remains unexplored.

## 3. Materials and Methods

To address plant growth stage classification, we develop an approach based on Vision Foundation Models paired with a regularizing multibranch ensemble head. We compare the proposed method against the architectures listed in Table 1, spanning both convolutional neural networks (CNNs) and Transformer architectures.

The classification head is an important component of the proposed architecture. Given the rich features extracted by the backbone, the head must adapt them to the downstream task. We develop three head variants. Fig. 1a shows a lightweight head, also known as a linear probe, often used with VFMs as recommended in [9], [10], and frequently suficient given the descriptive power of VFM features. To address the domain shift between VFM pretraining and downstream agricultural data, a head with a larger number of parameters may be required. Fig. 1b is a heavy head comprising two MLP layers with GELU activation followed by Dropout regularization (probability 0.3). Fig. 1c shows the proposed multibranch ensemble head, described in detail below.

![](images/8ffd8f6d8b7616a1abb67b5baaa199251d4f5eeb395ebc8012a28c696f42f188.jpg)  
Figure 1. Neural network heads proposed for fine-tuning.

To improve the performance of VFMs, an ensemble of models can be used; however, this increases inference time by a factor of N. Several works instead build more robust representations within a single network, efectively forming an in-model ensemble. The prototypical example is Dropout [23]: randomly masking activations during training samples a diferent weight-sharing subnetwork at each step, and test-time weight scaling approximates averaging the predictions of exponentially many such subnetworks. The authors of PTA [24] make this idea explicit, building an in-model ensemble from multiple parallel branches of diferent depths within the backbone, improving classification quality. StochasticDepth [25] demonstrates that dynamically varying network depth during training acts as a regularizer; this technique has been adopted by EfficientNet V1 and V2, DINOv2, and DINOv3. [26] proposes collaborative networks where multiple branches are trained simultaneously, each supervised by a combination of ground-truth labels and soft targets from the other heads. [27] proposes a multi-head multi-loss classifier in which each head is assigned higher weight to a subset of classes, inducing specialization, with the final prediction obtained by averaging across heads.

In this work we propose a multibranch ensemble head (Fig. 1c). The two branches difer in depth and capacity: the left branch is a simple linear probe, while the right branch consists of several MLP layers with Dropout, equivalent to the heavy head described above (Fig. 1b). During training, the active branch is randomly selected for each batch, so each branch learns to solve the task independently. The diference in capacity allows the light and heavy branches to capture simpler and more complex feature transformations, respectively. At inference, an inmodel ensemble is formed by aggregating the outputs of both branches:

$$
\hat { y } = \log \left( \frac { 1 } { 2 } \left( p ^ { ( \mathrm { l i g h t } ) } + p ^ { ( \mathrm { h e a v y } ) } \right) + \epsilon \right) ,\tag{1}
$$

where the branch output $p ^ { ( k ) }$ , k ∈ {light, heavy} is defined as:

$$
p ^ { ( k ) } = \operatorname { s o f t m a x } \left( f ^ { ( k ) } ( z ) \right) .\tag{2}
$$

Here $f ^ { ( k ) }$ is the function computed by branch $k ,$ and $z =$ g(X) are the features produced by the backbone g for the input batch X. The vector $\hat { y } \in \mathbb { R } ^ { C }$ holds the aggregated log-probabilities, and ϵ is a small positive constant that ensures numerical stability of the logarithm. Aggregation is performed in probability space rather than logit space (Eqs. 1–2), since logits from branches of diferent capacity may have substantially diferent scales. Probability-space aggregation is consistent with standard mixture-of-experts averaging and corresponds to a uniform Bayesian model average treating each head as an equally plausible hypothesis about the class distribution.

Table 1. Backbones considered.
<table><tr><td>Model Type</td><td>Sizes</td></tr><tr><td>Strongly Supervised Backbones</td><td></td></tr><tr><td>ResNet [18] CNN</td><td>50,101</td></tr><tr><td>ConvNeXt [19] CNN</td><td>Tiny (T)</td></tr><tr><td>EfficientNet [20] CNN</td><td>B3, B7</td></tr><tr><td>EfficientNetV2 [21] CNN</td><td>Small (S), Large (L)</td></tr><tr><td>ViT [22] Transformer</td><td>Base (B), Large (L)</td></tr><tr><td colspan="2">Self-Supervised Backbones</td></tr><tr><td>DINOv2 ViT [9]</td><td>Transformer Small (S), Large (L)</td></tr><tr><td>DINOv3 ViT [10]</td><td>Transformer Small (S), Large (L)</td></tr><tr><td>DINOv3 ConvNeXt [10] CNN</td><td>Tiny (T), Small (S), Large (L)</td></tr></table>

The branch selector shown in Fig. 1c activates exactly one branch per training step. At each step a branch is drawn independently from a Bernoulli distribution, $k \sim$ Bernoulli(π), with π = 0.5 assigning equal probability to the two branches. Only the selected branch produces the prediction for that step, and only its parameters, together with those of the backbone, receive gradients from the cross-entropy loss; the other branch is left unchanged. No knowledge-distillation or cross-branch consistency term is applied, so each branch learns to solve the task independently. Algorithm 1 summarizes the procedure.

Models are trained using the AdamW optimizer with a batch size of 32. The head is randomly initialized while the backbone uses pretrained weights. To mitigate catastrophic forgetting, training proceeds in two stages:

Algorithm 1: Training step with the multibranch   
ensemble head   
Input : mini-batch (X, Y); backbone $^ { g ; }$ branches   
$f ^ { \mathrm { ( l i g h t ) } }$ $f ^ { \mathrm { ( h e a v y ) } }$ ; branch probability $\pi$   
1 $z  g ( X )$ % backbone features   
2 draw k ∼ Bernoulli(π), k ∈ {light, heavy}   
3 ℓ ← CrossEntropy $\big ( f ^ { ( k ) } ( z ) , Y \big )$   
4 update parameters of $g$ and $f ^ { ( k ) }$ using $\nabla \ell$   
% other branch weights are not   
updated

15 epochs of head-only fine-tuning at a learning rate of $3 \times 1 0 ^ { - 3 }$ , followed by 15 epochs of full-network finetuning at a reduced learning rate of $5 \times 1 0 ^ { - 6 }$

## 4. Experiments

To evaluate the proposed approach we use the NALEF Food Safety dataset [11], published in 2026. The dataset contains images of cucumber crops captured under realworld conditions, annotated with five growth stages: plantula, early vegetative, vegetative growth, flowering, and fruit setting. Overall there are 787 training and 239 test images. As the public test set does not include groundtruth annotations, we perform a stratified train/validation split with an 80/20 ratio for all reported experiments. Images are captured using diferent devices (professional and smartphone cameras) under varying environmental conditions.

The dataset is strongly imbalanced across growth stages, as summarized in Table 2. The smallest class (plantula) contributes an order of magnitude fewer samples than the largest (vegetative growth), leaving only about ten validation images for the minority stages. Accuracy is unsuitable as the primary metric for highly imbalanced datasets. We therefore evaluate all models using Macro F1 score and Cohen’s Kappa.

Table 2. Per-class image counts for the stratified 80/20 train/validation split.
<table><tr><td>Growth stage</td><td>Train</td><td>Val</td><td>Total</td></tr><tr><td>Plantula</td><td>40</td><td>10</td><td>50</td></tr><tr><td>Early vegetative</td><td>42</td><td>11</td><td>53</td></tr><tr><td>Vegetative growth</td><td>307</td><td>77</td><td>384</td></tr><tr><td>Flowering</td><td>70</td><td>17</td><td>87</td></tr><tr><td>Fruit setting</td><td>170</td><td>43</td><td>213</td></tr><tr><td>Total</td><td>629</td><td>158</td><td>787</td></tr></table>

To improve model generalization, the following augmentations are applied: randomly resized crop, random horizontal flip, and color jitter. All images are resized to $2 2 4 \times 2 2 4$

## 5. Results

Experiment results are presented in Table 3. For each model we report the head configuration that achieved the best result, the type of backbone supervision, evaluation metrics, and computational complexity. For strongly supervised models we report the model name and size; for self-supervised backbones we report the DINO version, backbone architecture, and size. Most strongly supervised backbones are outperformed by self-supervised ones of comparable size. The exception is ViT-L, which is on par with the best DINO-pretrained models and attains the highest Macro F1 overall (0.979). Therefore, the advantage of self-supervised pretraining is concentrated in the small- and mid-sized models (Fig. 2).

Fig. 2 shows Macro F1 score as a function of computational complexity (GFlops). Strongly supervised mod-

![](images/37b96d71ef315839d0258ff1573530bf87dd44d02fd39b74c7584a821903a40d.jpg)  
Figure 2. Macro F1 score vs. model computational complexity (GFlops). Self-supervised backbones (blue) hold a clear advantage in the small- and mid-compute regime.

els are shown in red and self-supervised models in blue. The advantage of DINO-pretrained backbones is most pronounced in the small-model regime. The best result for a small model (Macro F1 of 0.978) is achieved by DI-NOv2 ViT-S with the proposed multibranch ensemble head (highlighted in red in Table 3), matching the strongly supervised ViT-L model, which is over 10× larger. A similarly sized ConvNeXt-Tiny model achieves a substantially lower Macro F1 of 0.959; DINOv3-style pretraining of ConvNeXt-Tiny raises this to 0.968.

DINOv2 models also outperform their DINOv3 ViT and ConvNeXt counterparts. As discussed above, DI-NOv3 models are trained on a larger dataset sourced primarily from Instagram, and the greater domain shift is the likely cause of their slightly inferior performance.

## 6. Discussion

Table 4 details the incremental improvements to DI-NOv2 ViT-S proposed in this work. As a baseline we train DINOv2 ViT-S with a light head (Fig. 1a) for 15 epochs. Replacing the linear probe with a heavy head improves F1 to 0.955. Switching to the multibranch head further raises F1 to 0.960. Subsequently fine-tuning the full backbone yields the final F1 of 0.978. Importantly, the multibranch head adds negligible parameters and essentially no additional inference overhead compared to the single-branch heavy head, since backbone computation dominates.

Table 3. Evaluation results of the considered models and heads. DINOv2 ViT-S with the multibranch head (highlighted in red) achieves the best quality-eficiency trade-of.
<table><tr><td>Model</td><td>Head</td><td>B. Supervision</td><td>F1</td><td>Kappa</td><td>GFlops</td><td>Params (M)</td></tr><tr><td>ViT-L</td><td>Heavy</td><td>Strong</td><td>0.979</td><td>0.972</td><td>123.108</td><td>303.831</td></tr><tr><td>DINOv2 ViT-L</td><td>Heavy</td><td>Self</td><td>0.978</td><td>0.971</td><td>162.025</td><td>304.898</td></tr><tr><td>DINOv2 ViT-S</td><td>Multibranch</td><td>Self</td><td>0.978</td><td>0.971</td><td>12.248</td><td>22.260</td></tr><tr><td>DINOv3 ConvNeXt-L</td><td>Light</td><td>Self</td><td>0.970</td><td>0.962</td><td>68.720</td><td>196.238</td></tr><tr><td>DINOv3 ViT-S</td><td>Heavy</td><td>Self</td><td>0.968</td><td>0.953</td><td>9.397</td><td>21.797</td></tr><tr><td>DINOv3 ConvNeXt-T</td><td>Light</td><td>Self</td><td>0.968</td><td>0.962</td><td>8.910</td><td>27.824</td></tr><tr><td>DINOv3 ViT-L</td><td>Heavy</td><td>Self</td><td>0.963</td><td>0.952</td><td>125.681</td><td>303.659</td></tr><tr><td>ConvNeXt-T</td><td>Heavy</td><td>Strong</td><td>0.959</td><td>0.952</td><td>8.910</td><td>28.218</td></tr><tr><td>DINOv3 ConvNeXt-S</td><td>Light</td><td>Self</td><td>0.955</td><td>0.943</td><td>17.366</td><td>49.459</td></tr><tr><td>ViT-B</td><td>Heavy</td><td>Strong</td><td>0.934</td><td>0.914</td><td>35.127</td><td>86.196</td></tr><tr><td>EfficientNet V2-L</td><td>Heavy</td><td>Strong</td><td>0.931</td><td>0.894</td><td>24.461</td><td>117.893</td></tr><tr><td>ResNet101</td><td>Multibranch</td><td>Strong</td><td>0.913</td><td>0.875</td><td>15.601</td><td>43.562</td></tr><tr><td>ResNet50</td><td>Heavy</td><td>Strong</td><td>0.897</td><td>0.874</td><td>8.176</td><td>24.560</td></tr><tr><td>EfficientNet-B7</td><td>Heavy</td><td>Strong</td><td>0.883</td><td>0.874</td><td>10.337</td><td>65.101</td></tr><tr><td>EfficientNet V2-S</td><td>Heavy</td><td>Strong</td><td>0.875</td><td>0.845</td><td>5.699</td><td>20.836</td></tr><tr><td>EfficientNet-B3</td><td>Light</td><td>Strong</td><td>0.871</td><td>0.810</td><td>1.922</td><td>10.704</td></tr></table>

Table 4. Ablation study of the proposed improvements over DINOv2 ViT-S.
<table><tr><td>Modification</td><td>F1</td><td>Kappa</td><td>GFlops</td><td>Params (M)</td></tr><tr><td>DINOv2 ViT-S (baseline)</td><td>0.915</td><td>0.895</td><td>12.247</td><td>22.059</td></tr><tr><td>+ multibranch head</td><td>0.960</td><td>0.944</td><td>12.248</td><td>22.260</td></tr><tr><td>+ backbone fine-tuning</td><td>0.978</td><td>0.971</td><td>12.248</td><td>22.260</td></tr></table>

As shown in Fig. 3, the proposed DINOv2 ViT-S with multibranch head is a robust classifier: it makes no errors between non-adjacent growth stages, and despite significant class imbalance it remains accurate on minority classes.

The only misclassifications occur between the vegetative growth and flowering stages. At first glance these stages should be visually distinct, separable by the presence of flowers. However, closer inspection of the NALEF Food Safety dataset reveals that images primarily show cucumber leaves, with flowers or fruits often occluded by foliage. The model must therefore discriminate stages based on subtle diferences in leaf and vine appearance alone, which is a considerably harder task.

Limitations. The dataset is captured from a leaf-centric viewpoint in which flowers and fruits are frequently occluded, which is also the main source of the residual vegetative-growth vs. flowering confusion. Typically, field examination is performed during daytime and under good environmental conditions. We have not evaluated robustness to adverse conditions such as heavy shading, rain, night-time capture, or lens contamination on smartphone and edge devices. In this work we target the NALEF Food

![](images/163305e934d97b27439f559561232586cdaba21933ac3a38d03b0ec41c85ccec.jpg)  
Figure 3. Confusion matrix of DINOv2 ViT-S with multibranch head.

Safety dataset. Therefore, retraining will be needed to adapt the model to other cultivars or environmental conditions. Also, the validation split is highly imbalanced, so small diferences in Macro F1 between the top models should be interpreted with this in mind.

The multibranch ensemble head targets any finegrained classification task where a lightweight head on top of frozen or fine-tuned features must be trained on limited data and is at risk of overfitting. Applying and evaluating it on other fine-grained benchmarks, for instance, species recognition or disease severity grading, is a natural next step that would establish how far the regularization benefit observed here carries over.

## 7. Conclusions

In this work we proposed a regularizing multibranch ensemble head for adapting Vision Foundation Models to small, domain-shifted datasets, and studied it on cucumber growth stage classification as a fine-grained case study. Experiments across a broad set of strongly supervised and self-supervised backbones on the NALEF Food Safety dataset indicate that self-supervised pretraining yields stronger feature representations for this task in the small- and mid-sized model regime. DINOv2 and DI-NOv3 backbones outperform ResNet, EficientNet, and strongly supervised ViT variants of comparable size, while at the largest scale the strongly supervised ViT-L remains competitive; overall, VFMs transfer well to agricultural imagery despite substantial pre-training domain gaps, with the clearest benefit when model capacity is limited. Notably, DINOv2 ViT-S paired with the proposed multibranch ensemble head achieves a Macro F1 of 0.978 and Cohen’s Kappa of0.971, matching models with over 300M parameters while requiring only 12.2 GFlops and 22.3M parameters, making it suitable for deployment on smartphones and edge devices.

The introduced multibranch ensemble head trains a lightweight linear branch and a heavier MLP branch simultaneously via stochastic branch sampling. At inference, predictions are aggregated in probability space as a uniform Bayesian model average. Ablation analysis confirms that the multibranch head improves over both the linear probe and the heavy head, and that subsequent fullnetwork fine-tuning yields further substantial gains. The method adds negligible parameters and essentially zero inference overhead compared to a single-branch heavy head.

Error analysis reveals that misclassifications are confined to adjacent phenological stages, specifically between vegetative growth and flowering, a consequence of the dataset’s leaf-centric imaging perspective in which flowers and fruits are frequently occluded. The absence of errors between non-adjacent stages indicates well-structured internal representations.

Future work includes extending the approach to additional crop species and growth stage taxonomies, as well as incorporating explainability by integrating large language models.

## Funding

This research was carried out as part of the scientific project “Software and algorithmic solutions for agricultural information and analytical systems based on artificial intelligence in variable agroclimatic conditions” (state registration number 0126U001122) funded by the Ministry of Education and Science of Ukraine at the expense of the state budget.

## Declaration on Generative AI

During the preparation of this work, the author used ChatGPT-5.6 for grammar and spelling checks. The author sub-

sequently reviewed and edited the content as necessary and takes full responsibility for the content of this publication.

## References

[1] N. Katal, M. Rzanny, P. Mäder, and J. Wäldchen, “Deep learning in plant phenological research: A systematic literature review,” Frontiers in Plant Science, vol. Volume 13 - 2022, 2022, ISSN: 1664-462X. DOI: 10 . 3389 / fpls . 2022 . 805738 [Online]. Available: https : / / www . frontiersin . org / journals / plant - science / articles / 10 . 3389 / fpls . 2022.805738

[2] “Sentinel-2 | Copernicus Data Space Ecosystem,” Accessed: May 14, 2026. [Online]. Available: https : / / dataspace . copernicus . eu / data - collections / copernicus - sentinel-missions/sentinel-2

[3] D. Darlan et al., “SmartBerry for AI-based growth stage classification and precision nutrition management in strawberry cultivation,” Scientific Reports, vol. 15, no. 1, p. 14 019, Apr. 23, 2025, ISSN: 2045- 2322. DOI: 10.1038/s41598-025-97168- z [Online]. Available: https://www.nature. com/articles/s41598-025-97168-z

[4] J.-S. G. Kim, S. Chung, M. Ko, J. Song, and S. H. Shin, “Comparison of Image Preprocessing Strategies for Convolutional Neural Network-Based Growth Stage Classification of Butterhead Lettuce in Industrial Plant Factories,” Applied Sciences, vol. 15, no. 11, p. 6278, Jun. 3, 2025, ISSN: 2076- 3417. DOI: 10.3390/app15116278 [Online]. Available: https://www.mdpi.com/2076- 3417/15/11/6278

[5] J. Liu, X. Wang, Q. Chen, P. Yan, and X. Liu, “Deep learning method for cucumber disease detection in complex environments for new agricultural productivity,” BMC Plant Biology, vol. 25, no. 1, p. 888, Jul. 7, 2025, ISSN: 1471-2229. DOI: 10. 1186 / s12870 - 025 - 06841 - y Accessed:

May 25, 2026. [Online]. Available: https : / / bmcplantbiol . biomedcentral . com / articles / 10 . 1186 / s12870 - 025 - 06841-y

[6] B. Li, G. Zhong, and W. Ke, “LMS-Res-YOLO: Lightweight and Multi-Scale Cucumber Detection Model with Residual Blocks,” Sensors, vol. 25, no. 23, p. 7305, Dec. 1, 2025, ISSN: 1424-8220. DOI: 10.3390/s25237305 Accessed: May 25, 2026. [Online]. Available: https : / / www . mdpi.com/1424-8220/25/23/7305

[7] S. Haseli Golzar, H. Bagherpour, and J. Amiri Parian, “A New Method to Optimize Deep CNN Model for Classification of Regular Cucumber Based on Global Average Pooling,” Journal ofFood Processing and Preservation, vol. 2024, no. 1, p. 5 818 803, Jan. 2024, ISSN: 0145-8892, 1745-4549. DOI: 10 . 1155 / 2024 / 5818803 Accessed: May 25, 2026. [Online]. Available: https : / / onlinelibrary . wiley . com / doi / 10 . 1155/2024/5818803

[8] H. Yu, X. Zhang, J. Yan, and X. Meng, “YOLO11m-SCFPose: An Improved Detection Framework for Keypoint Extraction in Cucumber Fruit Phenotyping,” Horticulturae, vol. 11, no. 7, p. 858, Jul. 20, 2025, ISSN: 2311-7524. DOI: 10.3390/ horticulturae11070858 Accessed: May 25, 2026. [Online]. Available: https : / / www . mdpi.com/2311-7524/11/7/858

[9] M. Oquab et al., “Dinov2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., vol. 2024, 2024. [Online]. Available: https://openreview.net/forum?id= a68SUt6zFt

[10] O. Siméoni et al., “Dinov3,” CoRR, vol. abs/2508.10104, 2025. DOI: 10 . 48550 / ARXIV . 2508 . 10104 arXiv: 2508 . 10104. [Online]. Available: https://doi.org/10. 48550/arXiv.2508.10104

[11] A. C. Menco Tovar, E. Puertas, J. C. Martinez-Santos, and J. Serrano Castañeda, Nalef food safety 2026, Zenodo, Mar. 2026. DOI: 10 . 5281/zenodo.19192941 [Online]. Available: https://doi.org/10.5281/zenodo. 19192941

[12] K. Khabarlak, “Fine-tuned DINOv2 ViT-L for cucumber growth stage classification: NALEF food safety challenge solution,” in Proceedings of the NALEF 2026 Workshop, Cartagena, Colombia, ser. CEUR Workshop Proceedings, In press, 2026.

[13] J.-S. G. Kim, S. H. Shin, and S. Chung, “Instancelevel phenotype-based growth stage classification of basil in multi-plant environments,” Frontiers in Plant Science, vol. 16, p. 1 707 985, Nov. 18, 2025, ISSN: 1664-462X. DOI: 10 . 3389 / fpls . 2025.1707985 [Online]. Available: https:// www.frontiersin.org/articles/10. 3389/fpls.2025.1707985/full

[14] C. Zhu, H. Yu, T. Lu, Y. Li, W. Jiang, and Q. Li, “Deep learning-based association analysis of root image data and cucumber yield,” The Plant Journal, vol. 118, no. 3, pp. 696–716, May 2024, ISSN: 0960-7412, 1365-313X. DOI: 10 . 1111 / tpj . 16627 Accessed: May 25, 2026. [Online]. Available: https : / / onlinelibrary . wiley . com/doi/10.1111/tpj.16627

[15] K. Khabarlak, I. Laktionov, and G. Diachenko, “Feature knowledge distillation using group convolutions for eficient plant pest recognition,” in Proceedings of the 12th International Conference Information Control Systems & Technologies (ICST 2024), Odesa, Ukraine, September 23–25, ser. CEUR Workshop Proceedings, vol. 3790, 2024, pp. 377–387. [Online]. Available: https : / / ceur-ws.org/Vol-3790/paper33.pdf

[16] R. Singh, R. Bidese-Puhl, K. Dhakal, and S. Sornapudi, “Few-shot adaptation of grounding DINO for agricultural domain,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Work-

shops, CVPR Workshops 2025, Nashville, TN, USA, June 11-15, 2025, Computer Vision Foundation / IEEE, 2025, pp. 5332–5342. [Online]. Available: https : / / openaccess . thecvf . com / content/CVPR2025W/V4A/html/Singh\_ Few-Shot\_Adaptation\_of\_Grounding\_ DINO \_ for \_ Agricultural \_ Domain \_ CVPRW\_2025\_paper.html

[17] A. Picon et al., Mitigating domain drift in multi species segmentation with dinov2: A cross-domain evaluation in herbicide research trials, 2026. arXiv: 2508 . 07514 [cs.CV]. [Online]. Available: https://arxiv.org/abs/2508.07514

[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27- 30, 2016, IEEE Computer Society, 2016, pp. 770– 778. DOI: 10.1109/CVPR.2016.90

[19] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, IEEE, 2022, pp. 11 966– 11 976. DOI: 10.1109/CVPR52688.2022. 01167 [Online]. Available: https : / / doi . org/10.1109/CVPR52688.2022.01167

[20] M. Tan and Q. V. Le, “EficientNet: Rethinking model scaling for convolutional neural networks,” in Proceedings ofthe 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, ser. Proceedings of Machine Learning Research, vol. 97, PMLR, 2019, pp. 6105–6114.

[21] M. Tan and Q. V. Le, “EficientNetV2: Smaller models and faster training,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, vol. 139, PMLR, 2021, pp. 10 096–10 106.

[22] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, OpenReview.net, 2021. [Online]. Available: https://openreview.net/ forum?id=YicbFdNTTy

[23] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” CoRR, vol. abs/1207.0580, 2012. arXiv: 1207.0580. [Online]. Available: http: //arxiv.org/abs/1207.0580

[24] K. Khabarlak, “Post-train adaptive MobileNet for fast anti-spoofing,” in Proceedings ofthe 3rd International Workshop on Intelligent Information Technologies & Systems ofInformation Security, Khmelnytskyi, Ukraine, March 23-25, 2022, ser. CEUR Workshop Proceedings, vol. 3156, CEUR-WS.org, 2022, pp. 44–53. [Online]. Available: http : / / ceur-ws.org/Vol-3156/keynote5.pdf

[25] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, “Deep networks with stochastic depth,” in Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, the Netherlands, October 11-14, 2016, Proceedings, Part IV, ser. Lecture Notes in Computer Science, vol. 9908, Springer, 2016, pp. 646–661. DOI: 10 . 1007 / 978 - 3 - 319-46493-0\_39

[26] G. Song and W. Chai, “Collaborative learning for deep neural networks,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18, Montréal, Canada: Curran Associates Inc., 2018, pp. 1837– 1846.

[27] A. Galdran, J. W. Verjans, G. Carneiro, and M. A. González Ballester, “Multi-head multi-loss model calibration,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023,

Cham: Springer Nature Switzerland, 2023, pp. 108– 117, ISBN: 978-3-031-43898-1.