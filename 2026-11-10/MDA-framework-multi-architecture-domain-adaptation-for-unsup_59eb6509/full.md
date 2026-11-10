# MDA framework: multi-architecture domain adaptation for unsupervised person re-identification<sup>⋆</sup>

Wei Zhang<sup>a,1</sup>, Peijun Ye<sup>b,2</sup>, Dihu Chen<sup>c,3</sup> and Tao Su<sup>a,∗</sup>

<sup>a</sup>School ofElectronics and Information Technology, Sun Yat-sen University, Guangzhou, Guangdong, China

<sup>b</sup>Peng Cheng Laboratory, Shenzhen, Guangdong, China

<sup>c</sup>School of Integrated Circuits, Sun Yat-sen University, Shenzhen, Guangdong, China

## A R T I C L E I N F O

Keywords:   
person re-identification   
domain adaptation   
Mamba

## A BS T RA C T

Unsupervised person re-identification not only faces the challenge of domain gaps, but existing methods also rarely support multiple network architectures. To tackle both issues, this paper proposes a multi-architecture auto-adaptive domain adaptation method: Mamba augmented multi-architecture domain adaptation framework. Specifically, this paper constructs the multi-architecture domain adaptation capability at two levels. At the first level, we propose a composite operator(MDA operator) that encapsulates Mamba as a component for enhancing domain adaptation. At the second level, based on this operator, we propose a Mamba-enhanced multi-architecture Domain Adaptation framework(MDA framework). Comprehensive experiments on person re-identification benchmarks validate the consistency and flexibility of the proposed framework. The MDA framework supports versatile deployment strategies without being confined to a single integration point. Under a singledeployment configuration, it surpasses several SOTA methods. Moreover, it demonstrates substantial performance improvements across both CNN and Transformer architectures. When embedded into a CNN backbone, the framework achieves a 10.2% mAP improvement on the adaptation task from CUHK03 to Market-1501. Integrated with a Transformer backbone, it yields a 19.8% mAP improvement on the same task.

## 1. Introduction

As a research topic related to security and intelligence, person re-identification (person Re-ID) has long been a popular research direction. Person re-identification is a sub-task of person retrieval. As a branch of person re-identification, unsupervised domain adaptation person re-identification (UDA person Re-ID) aims to transfer the recognition capability from a supervised source domain to an unsupervised target domain. Owing to its wide application in fields such as intelligent person search and smart cities, an increasing number of researchers are becoming engaged in this field.

The domain gap in person Re-ID is primarily manifested by variations in pedestrian appearance across diferent datasets. This domain divergence is chiefly demonstrated through: discrepancies arising from changes in lighting conditions; diferences caused by viewing angles; variations induced by background clutter; and disparities stemming from camera characteristics. Consequently, the robustness of neural network-extracted pedestrian features across domains is crucial for enhancing domain adaptation.

In the realm of person Re-ID, mainstream domain adaptation approaches, while making significant contributions, exhibit several limitations requiring further investigation. These methodologies encompass: feature alignment techniques, cross-domain image style translation approaches, and clustering-based methods. Regarding feature alignment techniques, such methods frequently necessitate additional annotated data to efectively bridge domain gaps. As for cross-domain image style translation, these approaches often present persistent challenges in training stability and pose significant training dificulties. Clustering-based methods commonly contend with the persistent challenge of imbalanced clustered data distribution.

Based on analysis of the Mamba architecture, we observe the potential to leverage Mamba’s distinctive characteristics for enhancing domain adaptation. In various computer vision domains, Mamba[8] and its derivative architectures are emerging as solutions to address the spatial ineficiency inherent in self-attention mechanisms, efectively replacing Transformer architecture. The Mamba architecture possesses the following operational property: it selectively propagates or discards information along the sequence based on the current token. The strategic utilization of this property holds significant promise for improving domain adaptation performance.

To address the shortcomings of mainstream methods, we propose a simple yet eficient method from the perspective of enhancing domain adaptation with Mamba[8]. Under domain adaptation settings, our method operates by selectively attending to identity-relevant tokens(feature points) while disregarding identity-irrelevant ones, thereby improving recognition performance. This approach not only ofers a novel alternative for domain adaptation but also overcomes limitations of current methods, as evidenced by: (1) Eficient Training and Convergence. The method demonstrates ease of training and rapid convergence. (2)Minimal Information Requirements. It eliminates the need for auxiliary information from either the source or target domains. (3) Enhanced Clustering Stability. The training process inherently fosters stability in feature clustering. Remarkably, this methodology can even outperform some SOTA domain adaptation solutions.

In summary, the contributions of this article are as follows:

• Composite Domain Adaptation Operator. This work proposes a Mamba augmented Domain Adaptation operator (MDA operator) as a foundational building block for domain adaptation.

• Mamba augmented multi-architecture domain adap tation framework (MDA framework). Following a scalable design methodology, we construct a domain adaptation framework based on the aforementioned composite operator.

• Extensive Experimental Validation. Comprehensive experiments on three Re-ID benchmarks demonstrate that the MDA framework exhibits robust positional independence and enhances domain adaptation across low-, medium-, and high-level feature stages. The method proves efective across multiple baselines and outperforms several SOTA approaches.

• Architectural Compatibility: The MDA framework demonstrates compatibility with both Transformer and CNN-based backbone networks, enabling seamless integration and switching between architectures.

## 2. Related Work

## 2.1. domain adaptation techniques for person Re-ID

Feature-level alignment methods reduce the domain gap by aligning features between the training and inference domains. During the unsupervised domain adaptation process, HUDA[15] jointly modeled both global distribution alignment and local instance alignment to discover transferable knowledge from the source domain. Employing a dual-branch network with an identity inference space, TJ-AIDL[21] leveraged not only the label information but also the attribute annotations of source domain images. This facilitated improved transfer of knowledge learned from the source domain to the target domain. Such approaches frequently rely on auxiliary information or additional annotations. This reliance often incurs additional costs.

Methods for cross-domain image style transfer typically employ various variants of Generative Adversarial Network[7] to migrate image styles across domains, generating images that possess the style of the target domain while preserving the labels from the source domain. PSI-StarGAN[4] introduced semantic information of humans into the image conversion process to enhance domain adaptation performance. From the perspective of reducing sample annotation costs, PTGAN[23] generated images preserving the identity from the source domain with backgrounds from the target domain, to bridge the domain gap. Such approaches generally entail higher training dificulty and longer training durations compared to conventional Convolutional Neural Networks, and the quality of synthesized images is constrained by the GAN’s fitting capability.

Clustering-based approaches employ pre-trained models to extract pedestrian features while estimating pseudo-labels based on clustering results. This model is then fine-tuned on the target domain using images assigned these pseudolabels. The MLC framework [16] leveraged a multi-scale network to generate multi-granularity pedestrian features from both global and local perspectives. Subsequently, a multi-label learning module utilized a memory feature bank to assign a multi-label vector to each image based on its similarity to features within the bank. Finally, after several iterations of multi-label training, self-paced clustering capable of filtering out noisy samples is incorporated into the training process to assign a pseudo-label to each image. The CSA method[9] incorporated a dual cluster consistency refinement mechanism designed to improve the quality of the clustered pseudo-labels. This category of methods is inherently limited by the performance ceiling of the clustering algorithm and susceptibility to label noise, frequently resulting in intra-cluster data imbalance.

Our approach difers distinctively from the three aforementioned technical pathways while efectively mitigating certain limitations inherent in them. Furthermore, our method does not require any auxiliary information, whether sourced from the source or target domains. Notably, our solution is characterized by ease of training, rapid convergence, and flexible deployment.

## 2.2. Mamba in Re-ID

Re-identification has now incorporated Mamba. MambaPro[22] introduced a Mamba aggregation block to model interactions between long sequences from diferent modalities, enhancing performance in multi-modal object re-identification. MambaReID[25] applied Mamba to multi-modal object reidentification, deploying visual Mamba and dense Mamba to extract multi-modal features. The CLIMB-ReID framework[24] proposed a multi-temporal Mamba module for extracting multi-granular spatiotemporal information. While previous Mamba-related research has primarily focused on supervised learning, we extend its application to unsupervised learning, further broadening the scope of Mamba.

## 3. Proposed Methodology

## 3.1. Preliminary

State space models(SSMs) have been widely applied to various sequential data modeling tasks. Inspired by continuous systems, SSMs are adept at capturing dynamic patterns within long sequences. Specifically, a state space model can map an input sequence $\bar { \boldsymbol { X } } ( t ) \in \dot { \mathbb { R } } ^ { L }$ to an output sequence $Y ( t ) \in \mathbb { R } ^ { L }$ , represented by the following expression:

$$
\left\{ \begin{array} { l l } { \mathrm { h } ^ { \prime } ( \mathrm { t } ) = \mathrm { A } ^ { * } \mathrm { h } ( \mathrm { t } ) + \mathrm { B } ^ { * } \mathrm { X } ( \mathrm { t } ) } \\ { \mathrm { Y } ( \mathrm { t } ) = \mathrm { C } ^ { * } \mathrm { h } ( \mathrm { t } ) + \mathrm { D } ^ { * } \mathrm { X } ( \mathrm { t } ) } \end{array} \right.\tag{1}
$$

Where $A \in \mathbb { R } ^ { N \times N } , B \in \mathbb { R } ^ { N }$ and $C \in \mathbb { R } ^ { N }$ are model parameters, and $_ { \textit { D } \in }$ ℝ denotes the residual term. Here, N represents the dimension of the hidden state h(t). When the input is a continuous sequence, the above equation can model it. When the input is in a discrete form, such as an image or text, the SSMs can switch to the "zero-order hold" method to achieve discretization. Specifically, the discretization process is described by the following expression:

$$
\left\{ \begin{array} { l l } { \overline { { A } } = \mathrm { E X P } ( \triangle A ) } \\ { \overline { { B } } = ( \triangle A ) ^ { - 1 } \left( \mathrm { E X P } ( \triangle A )  – \mathrm { I } \right) ^ { \ast } \triangle B } \\ { \overline { { C } } = \mathrm { C } } \end{array} \right.\tag{2}
$$

Where $\bigtriangleup \in \mathbb { R } ^ { D }$ denotes a predefined time-scale parameter that maps the continuous parameters A and B into a discrete space, and $\overline { { B } } \in \mathbb { R } ^ { D \times N } , \overline { { C } } \in \mathbb { R } ^ { D \times N }$ are discrete matrices. Following discretization, the computational process of the State Space Model can be represented as follows:

$$
\left\{ \begin{array} { l } { { h _ { k } = \overline { { A } } \ h _ { k - 1 } + \overline { { B } } \ X _ { k } } } \\ { { Y _ { k } = \overline { { C } } \ h _ { k } + \overline { { D } } \ X _ { k } } } \end{array} \right.\tag{3}
$$

Finally, the discretized SSMs can model discrete signals with linear complexity.

Selective Scan Mechanism. While SSMs are useful for eficient sequence modeling, traditional SSMs may struggle to capture complex patterns within diverse input sequences. Lacking data-dependent parameters, conventional SSMs cannot dynamically focus on or ignore specific information. To address this limitation, the Selective Scan Mechanism for SSMs termed S6[8] was proposed. In S6, the matrices $B \in \mathbb { R } ^ { L \times N } , C \in \mathbb { R } ^ { L \times N }$ and the timescale parameter $\triangle \ \in \ \mathbb { R } ^ { L \times D }$ are generated directly from the input data $\boldsymbol { X } \in \mathbb { R } ^ { L \times D }$ . This enables S6 to be contextually aware of the input, significantly enhancing its flexibility and eficiency.

Early SSMs updated their states in a fixed, input-invariant manner, making them poorly suited for processing language data requiring selective memorization. In contrast, Mamba addresses this limitation by introducing Selective SSMs, which dynamically determine which information to retain or discard based on the current input data.

## 3.2. Workflow and Baseline Descriptions

The complete domain adaptation workflow is outlined as follows. The proposed method follows a joint training paradigm, where the training dataset consists of both labeled source samples and unlabeled target samples. During training, data are fed into the network in batches of a predetermined size. Each batch is composed of an equal proportion of labeled source samples and unlabeled target samples. The overall training pipeline comprises a backbone network, a flexibly deployable MDA framework, a clustering computation module DBSCAN[5], and the re-identification loss function $L _ { R e I D } .$ . The backbone network is divided into several stages according to feature levels. A hybrid classifier is deployed at the output of the backbone network. We apply $L _ { R e I D }$ comprising the classification loss $L _ { c l s } [ 1 0 ]$ and triplet loss $L _ { t r i }$ [13] to both the features and predictions from source and target domains. Figure 1 illustrates the domain adaptation workflow.

$$
{ \cal L } _ { R e I D } = { \cal L } _ { c l s } + { \cal L } _ { t r i }\tag{4}
$$

The deployment of the MDA framework emphasizes achieving a trade-of between computational overhead and performance gains. As a plug-and-play module, the MDA framework is deployed during both training and inference. The number of stacked MDA operators can be varied, enabling scalability of the MDA framework size. Inspired by the bottleneck structure, we implement an MDA framework comprising three composite operators. Notably, we do not concurrently deploy the MDA framework after multiple stages of the backbone network. Figure 2 depicts the structure of the MDA framework.

The DBSCAN[5] settings are kept uniform across diferent datasets, although separately setting diferent EPS values for each dataset could yield better performance. Here, to evaluate the proposed method more precisely, this paper uniformly sets EPS to a fixed value across diferent target datasets.

To clarify our method, we categorize the features extracted from all stages of the backbone network into three distinct levels: (1) Low-level features, corresponding to Stage 1 and Stage 2 outputs. (2) Mid-level features, corresponding to Stage 3 outputs. (3) High-level features, corresponding to Stage 4 and Stage 5 outputs.

To validate the generalizability of the proposed method, the entire training pipeline is deployed on several baseline architectures with varied backbones.

• Baseline 1: A standard baseline using a pre-trained ResNet50 [11] backbone.

• Baseline 2: A high-performance baseline using a pretrained IBN-ResNet50 [19] backbone.

• Baseline 3: A baseline based on SwinTransformer [18].

These three baseline types are constructed by substituting the backbone network while keeping all other components consistent.

## 3.3. Introduction to MDA framework

The design objectives of the MDA framework are briefly outlined as follows: (1) Flexibility, enabling performance improvements at diferent feature stages; and (2) Generality, allowing adaptation to diverse baselines. To achieve these objectives, this work progressively constructs domain adaptation capabilities, beginning with the fundamental composite operator(MDA Operator) and culminating in the complete MDA framework.

Serving as the fundamental operator for enhanced domain adaptation, the design of the MDA Operator inherently achieves robustness, symmetry, flexibility, consistency, and computational eficiency. (1) Robustness: A residual structure is deployed to enhance robustness. (2) Symmetry: The operator exhibits symmetry in the transformation patterns across feature map channels. (3) Flexibility: This design can avoid dependence on specific locations, thereby improving deployment flexibility. (4) Consistency: The design pursues consistent parameter settings, enabling simple configuration across multiple deployment points. (5) Computational Eficiency: A bottleneck structure is adopted to conserve computational resources. Detailed operator design is illustrated in Figure 2.

![](images/5a953c06fc25d6dfd92547e824be3f7ea57ae5568a144e2e3ebc41f9342f379c.jpg)  
Figure 1: Workflow of domain adaptation

Targeting multi-architecture adaptation, this paper proposes a Mamba-enhanced domain adaptation framework(MDA framework). The MDA framework employs MDA operator as its core operator, deploys batch normalization operators to perform cross-domain normalization, adopts SiLU as the activation function, and uses Dropout to alleviate overfitting. Inside the framework, the MDA operator, together with batch normalization and SiLU operators, forms a basic stacking block; at the output, Dropout that is enabled only during training is applied to mitigate overfitting. Detailed design is illustrated in Figure 2.

The scalability of the MDA framework is achieved in two ways. The first approach involves increasing the stacking number of stackable components within the framework. The second approach allows the stacking blocks themselves to be stacked to form a larger-scale framework. Scalability is a flexible option. If, at a given deployment location, a larger-scale network can improve performance, increasing the stacking number is chosen; otherwise, a single stacking instance can be adopted.

## 4. Experiments

## 4.1. Datasets and Evaluation Protocols

We validated the proposed framework using three benchmark datasets (Market-1501[27], MSMT17[23], and CUHK03[17]). Performance was evaluated using mean Average Precision(mAP) and Cumulative Matching Characteristics(CMC) at ����−1. During joint training, no additional supervisory information was employed apart from the ground-truth labels in the source domain and the generated pseudo-labels in the target domain.

## 4.2. Implementation Details

Based on the PyTorch framework, we designed and implemented a comprehensive domain adaptation scheme. The experimental configuration is detailed as follows:

• Input Image Specifications: For baseline 1 and baseline 2, input images were resized to 256 × 128 pixels with augmentations including random flipping, random cropping, and random erasing. Baseline 3 utilized 224 × 224 pixel inputs.

• Optimizer Configuration: The Adam optimizer [14] was deployed during training with a weight decay of 0.0005.

• Learning Rate Schedule: A step decay policy was applied throughout training, reducing the learning rate by a factor of 10 every 20 epochs. No warm-up strategy was implemented at the initial training stage.

• Fine-tuning Protocol: Classifier parameters were finetuned over 50 epochs.

• For baselines 1 & 2: Initial learning rate: 0.001, Batch size : 128, EPS of DBSCAN : 0.6.

![](images/1a1ea478488cb5fc2adf1fe7896b4e3435a9ded8555ab0cd26f7aae0457582fd.jpg)  
Figure 2: MDA framework and MDA Operator

• For baseline 3: Initial learning rate: 0.00035, Batch size : 64, EPS of DBSCAN : 0.4.

To validate the framework’s transferability, we applied uniform hyperparameter settings and consistent stacking counts for MDA operator in both Baseline 1 and Baseline 3. For MDA operator, the hyperparameters include only SSM state expansion factor, local convolution width, and block expansion factor.

## 4.3. Comparisons with SOTA Methods

From the perspectives of generality, flexibility, and performance, this paper compares the proposed algorithm with existing algorithms. We selected representative unsupervised person re-identification methods from recent years to conduct performance comparisons with our approach. As shown in Table 1, when generalizing from Market-1501 to MSMT17, our method outperforms most methods from this period. When generalizing from MSMT17 to Market1501, among the methods listed in the table, some have higher baselines than our method, while others adopt optimizations on clustering settings. For a fairer comparison, our method does not apply clustering optimization tailored to diferent datasets, and thus its performance is lower than some methods. It should be emphasized that the methods in the table can typically only be deployed on a single architecture. In contrast, our solution can be deployed on multiple architectures with high flexibility in deployment locations.

Comparison with the SOTA method on multiple benchmarks
<table><tr><td rowspan="2">algorithm</td><td colspan="4">MSMT17 → Market Market → MSMT17</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>MDJL [1]</td><td>一</td><td>一</td><td>13.4</td><td>34.3</td></tr><tr><td>SECRET [12]</td><td>81.0</td><td>92.6</td><td>31.3</td><td>60.4</td></tr><tr><td>PPLR[3]</td><td>81.5</td><td>92.8</td><td>31.4</td><td>61.1</td></tr><tr><td>CDCL[20]</td><td>81.9</td><td>93.1</td><td>29.5</td><td>59.7</td></tr><tr><td>MTNet[2]</td><td>82.7</td><td>93.0</td><td>32.1</td><td>59.6</td></tr><tr><td>ADA framework[26]</td><td>83.4</td><td>92.8</td><td>23.1</td><td>46.0</td></tr><tr><td>MDA framework (Ours) 84.3</td><td></td><td>93.7</td><td>24.1</td><td>47.4</td></tr></table>

## 4.4. Ablation Study

This paper validates the efectiveness, generalizability, and consistency of the framework from the perspectives of deployment flexibility, backbone network adaptability, and parameter consistency. Experiments were conducted on multiple common datasets(CUHK03 [17], Market1501 [27], MSMT17 [23]), deployed sequentially on three backbone networks, with extensive ablation studies performed using reasonable dataset combinations.

Unless otherwise specified, all subsequent experiments adhere to the following configuration: three MDA operators are deployed within the MDA framework; the MDA framework is stacked only once; and during a single experiment, the MDA framework is deployed at only one position.

## 4.4.1. Experiment on Baseline 1

To validate the performance improvement potential of the MDA framework, this paper implements a standard Baseline 1. Baseline 1 consists of the following components: backbone network ResNet-50, DBSCAN clustering, and loss function $L _ { R e I D }$ . The performance improvement introduced by the MDA framework was tested using various dataset combinations. Table 2 summarizes the performance of different dataset combinations and deployment locations under Baseline 1.

## 4.4.2. Experiment on Baseline 2

To assess the adaptability of the MDA framework on a high-performance baseline, this paper implements a highperformance Baseline 2. Baseline 2 consists of the following components: backbone network IBN-ResNet50, DBSCAN clustering, and loss function $L _ { R e I D }$ . The generalization capability of the MDA framework was tested using various dataset combinations. Table 3 summarizes the MDA framework’s performance under Baseline 2 across diferent dataset combinations and deployment locations.

## 4.4.3. Experiment on Baseline 3

To demonstrate the cross-architecture generalization capability of the MDA framework, this paper implements a Swin Transformer-based Baseline 3. Baseline 3 consists of the following components: backbone network Swin Transformer, DBSCAN clustering, and loss function $L _ { R e I D } .$ . The cross-architecture generalization capability of the framework was tested using various dataset combinations. Table 4 summarizes the MDA framework’s performance under Baseline 3 across diferent dataset combinations and deployment locations.

## 4.4.4. Parameter Analysis ofMamba

When migrating from Baseline 1 to Baseline 3, the hyperparameters of Mamba required zero adjustment. Deployed at all efective positions, the hyperparameters remained unchanged (local convolution width = 4, block expansion factor = 2, state expansion factor = 32).

The ease of hyperparameter tuning further validates the advantages of the framework. When migrating from Baseline 1 to Baseline 2, only minor adjustments were made to Mamba’s hyperparameters. The local convolution width remained unchanged (= 4). In this paper, the state expansion factor and the block expansion factor were moderately tuned according to the spatial size and channel values of the feature maps. When deployed after Stage 1, the block expansion factor is 2 and the state expansion factor is 64. When deployed after Stages 2 and 3, the block expansion factor is 4 and the state expansion factor is 128. When deployed after Stage 4, the block expansion factor is 4 and the state expansion factor is 32.

Through a comparison of parameters and FLOPs between the MDA framework and ResNet50, the deployment of the MDA framework exhibits the following pattern: the higher the feature level, the greater the demands on FLOPs and parameters after deploying the MDA framework. Therefore, the MDA framework is most competitive when deployed at the low-level feature stage. Detailed data are presented in the table 5 below.

## 4.4.5. Analysis of Experimental Results

Through an in-depth analysis of the experimental results obtained on the aforementioned three baselines, we draw the following conclusions:

• The MDA framework demonstrates superior generalization performance.

• The MDA framework exhibits multi-architecture applicability. When migrating from CNN architectures to Transformer architectures, it achieves performance improvements without re-tuning of Mamba parameters.

• The MDA framework provides deployment flexibility. It demonstrated this flexibility across diferent architectures and can be deployed at stages corresponding to low-, medium-, and high-level features.

## 4.5. Visualization

We employed the visualization tool Torchcam[6] to visualize images from the test set, using respectively the weights of Baseline 1 and the weights obtained after deploying the MDA framework at diferent positions within Baseline 1. The resulting visualizations were then arranged sequentially in a row. In these images, darker-colored regions (e.g., red areas) correspond to larger weights compared to lighter-colored ones. Through an in-depth analysis of the weight distributions, we conclude that, relative to the baseline output, the weight distribution exhibits pronounced selectivity after the deployment of the MDA framework. Specifically, regions relevant to identity recognition become weight-intensive, and this selectivity is introduced by the framework. Hence, the visualization results validate the effectiveness of the MDA framework. Sample visualizations are presented in Figure 3.

Table 2  
Ablation experiment of MDA framework deployment on baseline 1.
<table><tr><td rowspan="2">Experimental setup</td><td colspan="2">C to M</td><td colspan="2">M to C</td><td colspan="2">M to MS</td><td colspan="2">MS to M</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>Baseline1</td><td>61.2</td><td>78.9</td><td>6.3</td><td>5.5</td><td>13.0</td><td>29.1</td><td>71.2</td><td>86.5</td></tr><tr><td>MDA framework after stage 1</td><td>65.0</td><td>81.5</td><td>7.2</td><td>5.4</td><td>14.6</td><td>32.5</td><td>71.2</td><td>86.5</td></tr><tr><td>MDA framework after stage 2</td><td>68.9</td><td>84.9</td><td>8.2</td><td>6.9</td><td>15.3</td><td>33.7</td><td>73.7</td><td>88.5</td></tr><tr><td>MDA framework after stage 3</td><td>66.0</td><td>82.5</td><td>7.8</td><td>6.1</td><td>15.0</td><td>33.0</td><td>74.4</td><td>88.6</td></tr><tr><td>MDA framework after stage 4</td><td>71.4</td><td>86.5</td><td>8.0</td><td>6.4</td><td>16.5</td><td>35.9</td><td>76.2</td><td>89.7</td></tr><tr><td>MDA framework after stage 5</td><td>30.8</td><td>52.9</td><td>2.1</td><td>1.4</td><td>2.7</td><td>6.5</td><td>14.7</td><td>32.7</td></tr></table>

Note: The characters C, M, and MS are abbreviations for CUHK03, Market1501, and MSMT17, respectively.

Ablation experiment of MDA framework deployment on baseline 2.
<table><tr><td rowspan="2">Setting</td><td colspan="2">C to M</td><td colspan="2">M to C</td><td colspan="2">M to MS</td><td colspan="2">MS to M</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>Baseline2</td><td>75.3</td><td>88.6</td><td>10.1</td><td>7.9</td><td>20.2</td><td>41.3</td><td>80.1</td><td>91.8</td></tr><tr><td>MDA framework after stage 1</td><td>78.4</td><td>90.2</td><td>12.5</td><td>11.3</td><td>22.8</td><td>47.1</td><td>81.7</td><td>92.9</td></tr><tr><td>MDA framework after stage 2</td><td>77.0</td><td>89.8</td><td>11.5</td><td>10.3</td><td>22.8</td><td>46.1</td><td>80.6</td><td>91.7</td></tr><tr><td>MDA framework after stage 3</td><td>78.4</td><td>89.8</td><td>12.0</td><td>10.6</td><td>21.7</td><td>44.2</td><td>82.0</td><td>92.0</td></tr><tr><td>MDA framework after stage 4</td><td>80.8</td><td>91.6</td><td>12.7</td><td>10.9</td><td>24.1</td><td>47.4</td><td>84.3</td><td>93.7</td></tr><tr><td>MDA framework after stage 5</td><td>45.4</td><td>68.9</td><td>9.2</td><td>7.2</td><td>3.3</td><td>7.1</td><td>43.2</td><td>67.4</td></tr></table>

## 5. Discussion

(1) Why the framework can enhance domain adaptation. Based on the above experiments, we analyze that when an image is transformed into a long linear sequence (a large number of tokens), the framework selectively remembers or forgets information along the sequence depending on the current token. In the domain adaptation setting, this property can be converted into selectively attending to identity-relevant tokens and forgetting identity-irrelevant tokens, thereby improving recognition performance.

(2) Limitations. The built-in atomic addition operation in Mamba introduces randomness, which leads to slight performance fluctuations. This issue can be resolved through hyperparameter tuning. When Mamba-derived networks adopt this method, adjusting the network structure of the framework may be required.

(3) Why do multiple deployments and multiple stacking of the framework not further improve performance? In the experiments, the frameworks deployed at multiple positions did not form an efective synergistic efect, resulting in only minor changes in the weights of identity-relevant regions relative to identity-irrelevant regions. A similar weight modulation efect was observed with repeatedly stacked frameworks.

Ablation experiment of MDA framework deployment on baseline 3.
<table><tr><td rowspan="2">Experimental setup</td><td colspan="2">C to M</td><td colspan="2">M to C</td><td colspan="2">M to MS</td><td colspan="2">MS to M</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>Baseline3</td><td>23.9</td><td>48.0</td><td>6.9</td><td>6.3</td><td>4.8</td><td>14.6</td><td>33.2</td><td>59.1</td></tr><tr><td>MDA framework after stage 1</td><td>40.8</td><td>64.6</td><td>8.5</td><td>7.4</td><td>4.3</td><td>10.6</td><td>48.3</td><td>70.6</td></tr><tr><td>MDA framework after stage 2</td><td>41.5</td><td>63.2</td><td>8.6</td><td>7.5</td><td>5.4</td><td>13.3</td><td>53.0</td><td>72.2</td></tr><tr><td>MDA framework after stage 3</td><td>43.7</td><td>64.9</td><td>8.3</td><td>6.9</td><td>4.7</td><td>11.2</td><td>56.7</td><td>76.4</td></tr><tr><td>MDA framework after stage 4</td><td>36.5</td><td>58.9</td><td>5.3</td><td>4.8</td><td>5.7</td><td>14.2</td><td>46.8</td><td>68.2</td></tr><tr><td>MDA framework after stage 5</td><td>40.5</td><td>62.6</td><td>9.9</td><td>8.4</td><td>4.1</td><td>10.3</td><td>57.2</td><td>76.9</td></tr></table>

1  
1  
6  
4  
5  
5  
![](images/d67eefbb7ade6149c3f6aac437947d770c9f52f71e75f47c6b2aa5ebcf5f8256.jpg)

![](images/7e5a98c5eb2291b317a36cb5015970e9a79c942c50f9574e3e5f601d1c80691c.jpg)

![](images/3e30fdc3a58f556b3f5ee7557d3db0d286794df733a2da82de48ab1cd7496694.jpg)

![](images/4f98025c538fdc12138e7b52090f7cd216d8a096d44b95c3d86ade0f21b26f39.jpg)

![](images/eed395686512d06f5d8956a91d65d99cc0aaef0a4f201811f87223f4e00e80d3.jpg)  
2

![](images/b503c797a2a6ea79ce2e240266c7be694bcd397e033746294a001849fad6d0db.jpg)

![](images/92b0610e5c6dcd45c5981212fe118ce6e44de03cc958cbea99a3faeb5614bb39.jpg)

![](images/41e310857f8bc98b67a6129b7bca5abc153926f32c744b568cc89b17e04b482a.jpg)

3  
![](images/fa8747aac1456f017f308dd6fea3f740adcc1d3b2432ac4aafedf9d1e83ba323.jpg)

4  
![](images/7e591afb34d2cd726b03e7181a9be9a3394e845466e3b4f57505d626be040a08.jpg)

![](images/98fdfabf13486a4681537454e17a7574a1a98832fa6ba044d751b57cc9130900.jpg)

2  
3  
![](images/5b2258fd14236fa9735f811a4ce835d1f04795760717da8ffbfc14262a142225.jpg)

![](images/56262e32f089d818262d03926fb6564eb674f8a7922376a0136f979e129d0c5a.jpg)

![](images/d6f074467cf00c765dcb63f8b69af832128261fadf7b6f4bb88c28871e8d2752.jpg)

![](images/4c83d1cbd556e81e0385982e3ddf66b3d5f0cbabf013bc9416a95492c03157c3.jpg)

![](images/6896b558ce77aa2ef50a390253640f5d97f1eeac1b18093fd674560ec2b20cf0.jpg)  
Figure 3: Visualization for MDA framework. Note: From left to right are figure one to six in each group of images. Figure-1 : original image. Figure-2 : heatmap of baseline. Figure-3 : heatmap for MDA framework after stage1. Figure-4: heatmap for MDA framework after stage2. Figure-5: heatmap for MDA framework after stage3. Figure-6: heatmap for MDA framework after stage4.

Table 5  
MDA framework VS ResNet50 stage 1/2/3/4/5
<table><tr><td>Part of network</td><td>Params(M) FLOPs(G)</td></tr><tr><td>ResNet50 stage 1</td><td>0.010 0.158</td></tr><tr><td>MDA framework after stage 1</td><td>0.070 1.048</td></tr><tr><td>ResNet50 stage 2</td><td>0.216 0.895</td></tr><tr><td>MDA framework after stage 2</td><td>1.004 3.846</td></tr><tr><td>ResNet50 stage 3</td><td>1.220 1.358</td></tr><tr><td>MDA framework after stage 3</td><td>3.930 3.789</td></tr><tr><td>ResNet50 stage 4</td><td>7.098 1.924</td></tr><tr><td>MDA framework after stage 4</td><td>15.553 3.760</td></tr><tr><td>ResNet50 stage 5</td><td>14.965 3.837</td></tr><tr><td>MDA framework after stage 5</td><td>61.875 14.981</td></tr></table>

(4) What are the implications of this study for related fields? Constructing composite operators based on Mamba and building domain adaptation networks upon these composite operators is a feasible technical option. Since Mamba-derived networks share properties similar to those of Mamba, these derivative networks are also promising for improving domain adaptation.

## 6. Conclusions

In the field of UDA person re-identification, this paper is the first to exploit the properties of Mamba to significantly improve recognition performance. The performance improvement stems from a characteristic of the framework: The memorization of information is transformed into attention enhancement, while the forgetting of information is transformed into attention suppression. It is precisely because of this characteristic that the proposed framework can enhance recognition performance at diferent feature stages. The framework’s adaptability to backbone networks of diferent architectures facilitates its application in a wider range of domains.

## References

[1] Chen, F., Wang, N., Tang, J., Yan, P., Yu, J., 2023a. Unsupervised person re-identification via multi-domain joint learning. Pattern Recognition 138, 109369. doi:https://doi.org/10.1016/j.patcog. 2023.109369.

[2] Chen, S., Qiu, L., Tian, Z., Yan, Y., Wang, D.H., Zhu, S., 2023b. MT-Net: Mutual tri-training network for unsupervised domain adaptation on person re-identification. Journal of Visual Communication and Image Representation 90, 103749.

[3] Cho, Y., Kim, W.J., Hong, S., Yoon, S.E., 2022. Part-based pseudo label refinement for unsupervised person re-identification, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7308–7318.

[4] Dung, D.A., Yasuhiro, N., 2025. Person semantic informationintegrated stargan for unsupervised domain adaptive person reidentification. International Symposium on Artificial Life and Robotics (AROB 30th 2025) .

[5] Ester, M., Kriegel, H.P., Sander, J., Xu, X., et al., 1996. A densitybased algorithm for discovering clusters in large spatial databases with noise., in: kdd, pp. 226–231.

[6] Fernandez, F.G., 2020. Torchcam: class activation explorer. https: //github.com/frgfm/torch-cam.

[7] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2020. Generative adversarial networks. Commun. ACM 63, 139–144. URL: https://doi.org/10. 1145/3422622, doi:10.1145/3422622.

[8] Gu, A., Dao, T., 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 .

[9] GUO, J., WANG, Y., GUO, J., WEI, S., SHI, X., 2025. Camera style alignment for unsupervised person re-identification. IEICE Transactions on Information and Systems , 2024EDP7135.

[10] Hauptmann, A., Yang, Y., Zheng, L., 2016. Person re-identification: Past, present and future. arXiv preprint arXiv:1610.02984 .

[11] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778.

[12] He, T., Shen, L., Guo, Y., Ding, G., Guo, Z., 2022. Secret: Selfconsistent pseudo label refinement for unsupervised domain adaptive person re-identification, in: Proceedings of the AAAI conference on artificial intelligence, pp. 879–887.

[13] Hermans, A., Beyer, L., Leibe, B., . In defense of the triplet loss for person re-identification. arXiv preprint arXiv:1703.07737 .

[14] Kingma, D.P., Ba, J., 2014. Adam: A method for stochastic optimization. CoRR abs/1412.6980.

[15] Lan, X., Zhu, X., Gong, S., 2022. Unsupervised cross-domain person re-identification by instance and distribution alignment. Pattern Recognition 124, 108514. doi:https://doi.org/10.1016/j.patcog. 2021.108514.

[16] Li, Q., Peng, X., Qiao, Y., Hao, Q., 2022. Unsupervised person reidentification with multi-label learning guided self-paced clustering. Pattern Recognition 125, 108521. URL: https://www.sciencedirect. com/science/article/pii/S0031320322000024, doi:https://doi.org/10. 1016/j.patcog.2022.108521.

[17] Li, W., Zhao, R., Xiao, T., Wang, X., 2014. Deepreid: Deep filter pairing neural network for person re-identification, in: 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp. 152– 159. doi:10.1109/CVPR.2014.27.

[18] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992–10002. doi:10.1109/ICCV48922.2021.00986.

[19] Pan, X., Luo, P., Shi, J., Tang, X., 2018. Two at once: Enhancing learning and generalization capacities via ibn-net.

[20] Tian, Q., Sun, J., 2023. Cluster-based dual-branch contrastive learning for unsupervised domain adaptation person reidentification. Knowledge-Based Systems 280, 111026. URL: https: //www.sciencedirect.com/science/article/pii/S0950705123007761, doi:https://doi.org/10.1016/j.knosys.2023.111026.

[21] Wang, J., Zhu, X., Gong, S., Li, W., 2018. Transferable joint attributeidentity deep learning for unsupervised person re-identification, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2275–2284.

[22] Wang, Y., Liu, X., Yan, T., Liu, Y., Zheng, A., Zhang, P., Lu, H., 2025. Mambapro: Multi-modal object re-identification with mamba aggregation and synergistic prompt, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 8150–8158.

[23] Wei, L., Zhang, S., Gao, W., Tian, Q., 2018. Person transfer gan to bridge domain gap for person re-identification, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 79– 88.

[24] Yu, C., Liu, X., Zhu, J., Wang, Y., Zhang, P., Lu, H., 2025. Climbreid: A hybrid clip-mamba framework for person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 9589–9597.

[25] Zhang, R., Xu, L., Yang, S., Wang, L., 2024. Mambareid: Exploiting vision mamba for multi-modal object re-identification. Sensors 24, 4639.

[26] Zhang, W., Ye, P., Chen, D., Su, T., 2026. Ada framework for unsupervised domain adaptation person re-identification. Pattern Recognition 171, 112238. URL: https://www.sciencedirect. com/science/article/pii/S0031320325008994, doi:https://doi.org/10. 1016/j.patcog.2025.112238.

[27] Zheng, L., Shen, L., Tian, L., Wang, S., Bu, J., Tian, Q., 2015. Person re-identification meets image search. arXiv preprint arXiv:1502.02171 .