# Vision Transformer-Based Multi-Level Feature Fusion for Multi-Label Sewer Defect Classification

Xu Fang<sup>1</sup>, Zhuoran Wang<sup>2</sup>, Qing Li<sup>3</sup>, Shengyu Zhang<sup>1</sup>, Guanzhi Deng<sup>2</sup>, Jianbiao $\mathrm { H e } ^ { 1 }$ , and Qingquan Li<sup>4</sup>

<sup>1</sup>Shenzhen Polytechnic University, Shenzhen, China. E-mails: fangxu.hello@gmail.com; zsyhcxs@szpu.edu.cn. Corresponding author: Jianbiao He; E-mail: hejianbiao@szpu.edu.cn <sup>2</sup>City University of Hong Kong, Hong Kong, China. E-mails: wangzr040220@gmail.com; guanzdeng2-c@my.cityu.edu.hk

<sup>3</sup>Pengcheng Laboratory, Shenzhen, China. E-mail: lqing900205@gmail.com <sup>4</sup>Shenzhen University, Shenzhen, China. E-mail: liqq@szu.edu.cn

## ABSTRACT

Automated classification of sewer defects is essential for infrastructure condition assessment and maintenance decision-making, but existing deep learning methods struggle to balance classification accuracy and computational complexity in large-scale multi-label scenarios. This study develops Sewer-Transformer-ML, a hierarchical vision Transformer with multi-level feature fusion, together with two lightweight architectures, Sewer-MobileNet-ML and Sewer-Mobile-TransNet, for resource-constrained inspection scenarios. On the Sewer-ML test set, Sewer-Transformer-ML-Base achieved an �2 of 65.68% and an $F 1 _ { \mathrm { N o r m a l } }$ of 92.68%, ranking first on the public leaderboard and exceeding the second-ranked method by 7.6 percentage points in $F 2 _ { \mathrm { C I W } }$ Sewer-MobileNet-ML achieved an $F 2 _ { \mathrm { C I W } }$ of 65.73% with only 17 M parameters, representing an approximately 95% parameter reduction relative to the base model. Under the standard Sewer-Capsule data split, Sewer-Mobile-TransNet achieved 96.43% classification accuracy. When the training set was reduced to 1,177 images, pretraining on Sewer-ML consistently improved model performance. Ablation experiments further showed that direct concatenation was more efective for Transformer features, whereas attention-based fusion better supported multiscale CNN features. These findings provide a computational basis for automated sewer inspection, lightweight model design, and adaptation across civil infrastructure inspection platforms.

## PRACTICAL APPLICATIONS

Municipal sewer inspections generate large volumes of closed-circuit television images and videos that are commonly reviewed manually, making the process time-consuming and dependent on inspector experience and workload. The proposed models can support this workflow by automatically screening inspection images, distinguishing normal conditions from diferent defect types, and assigning multiple labels when several defects occur in the same image. The resulting classifications can help engineers prioritize images or pipe segments for further examination and provide supporting information for condition assessment, maintenance prioritization, and rehabilitation planning. Because the lightweight models substantially reduce the number of parameters while retaining high classification accuracy, they have the potential to be integrated into sewer inspection robots, embedded computing devices, or field video-processing systems. The cross-dataset experiments also indicate that knowledge learned from large CCTV data sets can improve adaptation to images collected by emerging sewer-capsule platforms, reducing the amount of labeled data required when introducing a new inspection device. These results demonstrate technical application potential; however, inference speed, energy consumption, and long-term reliability on operational field hardware remain to be validated.

## INTRODUCTION

Accurate inspection and quantitative assessment of urban underground drainage systems are critical components of municipal infrastructure health monitoring, essential for ensuring urban environmental safety and resilient operation in the context of smart city development and sustainable urbanization. As subsurface assets that form the backbone of modern urban areas, these pipeline networks require continuous condition monitoring for public health protection and flood prevention, where failures can lead to significant economic losses and social disruption. However, pipelines are susceptible to natural environmental factors and engineering construction impacts during longterm service, developing various defects such as cracking, displacement, corrosion, or blockage, which may cause safety accidents like water leakage, ground subsidence, or even collapse. Therefore, automated inspection and intelligent analysis of drainage pipelines are necessary to avoid pipeline performance deterioration or sudden structural failure, establishing a quantitative basis for intelligent infrastructure management in modern cities.

Although deep learning models have achieved success in speech recognition, face recognition and other fields relying on large-scale datasets like ImageNet and MSCOCO, they still face three major challenges in automated multi-label classification of sewer defect images from closedcircuit television (CCTV) and robotic inspections for large-scale municipal asset management: (1) Environmental complexity: insuficient lighting and high noise in internal pipeline imaging under complex field conditions, diverse defect morphologies and significant scale variations, leading to insuficient classification accuracy of existing models; (2) Computational resource constraints: deep models have large parameter counts, making edge deployment on mobile or embedded devices dificult; (3) Data and evaluation bottlenecks: commercial restrictions result in a lack of publicly available large-scale benchmark datasets, and emerging robotic collection methods like sewer pipeline capsule systems produce low-quality data with few samples, making it dificult to reproduce and fairly compare existing methods.

To address the above challenges in infrastructure condition assessment, the main contributions of this paper include:

1. Proposing the Sewer-Transformer-ML model as a novel intelligent inspection framework: Based on a hierarchical vision Transformer architecture, it enhances defect representation capability by integrating multi-level features, achieving first place on the largest open-source benchmark Sewer-ML (establishing a new classification benchmark with $F 2 _ { \mathrm { C I W } } 6 5 . 6 8 \%$ , $F 1 _ { \mathrm { N o r m a l } } 9 2 . 6 8 \% )$ , significantly outperforming the second-best method by 7.6 percentage points (65.68% vs. 58.08%).

2. Designing lightweight solutions: Constructing Sewer-MobileNet-ML based on MobileNetV3 and Sewer-Mobile-TransNet, achieving a good trade-of between accuracy and complexity with approximately 95% parameter reduction while maintaining state-of-theart accuracy (65.73% $F 2 _ { \mathrm { C I W } } , F 1 _ { \mathrm { N o r m a l } } 8 9 . 5 3 \% )$ , validating that small models can achieve superior performance for automated field inspection.

3. Revealing feature fusion principles: Through comprehensive and systematic comparison, we reveal that direct stacking of Transformer multi-level features achieves optimal results, while CNN multi-scale features are more suitable for fusion through multi-head attention mechanisms, providing scientific basis for deep learning network structure design in infrastructure defect detection.

4. Validating cross-domain transferability: Through comprehensive transfer learning experiments, we demonstrate that transferring pre-trained knowledge from large-scale CCTV inspection datasets to the small-scale Sewer-Capsule Dataset collected by novel robotic systems significantly improves classification performance. Sewer-Mobile-TransNet achieved 96.43% accuracy under the standard data split, and pretraining consistently improved model performance when the training set was reduced to 1,177 images, validating cross-domain transferability and enhancing practicality for municipal infrastructure maintenance.

## RELATED WORK

Previous automated sewer inspection methods can be broadly grouped into traditional imageprocessing and machine-learning approaches and more recent deep-learning approaches.

## Traditional Image Processing and Machine Learning Methods for Pipeline Defect Detection

Early systems combined hand-crafted texture, shape, or edge features with neural-network, support-vector-machine, or random-forest classifiers (Yang and Su 2008; Shehab and Moselhi 2005; Myrans et al. 2019). Other pipelines used edge detection and morphological processing to identify visually distinctive defects such as cracks and open joints (Su and Yang 2014; Halfawy and Hengmeechai 2014). Although these approaches reduced part of the manual inspection workload, their multi-stage processing and task-specific feature design limited generalization across variable illumination, noise, pipe materials, and defect morphologies.

## Deep Learning Methods for Pipeline Defect Detection

CNNs subsequently enabled end-to-end feature learning for sewer defect classification. Early multi-label systems used multiple binary subnetworks or shallow CNNs (Kumar et al. 2018; Meijer et al. 2019), while hierarchical approaches separated defect screening from class prediction to address imbalance and task complexity (Li et al. 2019; Xie et al. 2019). The release of Sewer-ML established a large-scale multi-label benchmark with class-importance-weighted evaluation, enabling fairer comparison of these methods (Haurum and Moeslund 2021).

Recent research has examined stronger CNN backbones, vision Transformers, and hybrid CNN-Transformer architectures. Cross-domain benchmarking indicates that pretrained CNNs can remain competitive on small target datasets (Meng et al. 2025), whereas hybrid models combine local convolutional features with long-range attention (Goharinezhad et al. 2025; Zhang et al. 2024). Nevertheless, existing studies provide limited systematic evidence on how multi-level Transformer features and multiscale CNN features should be fused for large-scale multi-label sewer classification. The balance between classification performance and lightweight deployment, as well as transfer from large CCTV datasets to data collected by emerging robotic platforms, also remains insuficiently studied. Accordingly, this work compares pure Transformer, lightweight CNN, and hybrid architectures and evaluates their feature-fusion and cross-domain transfer behavior (Liu et al. 2021; Dosovitskiy et al. 2021; Nguyen et al. 2025).

## METHODOLOGY

To address the multi-label classification challenge in sewer pipeline defect detection, this paper proposes a hierarchical vision Transformer-based inspection framework with Swin Transformer as the backbone. The main model, Sewer-Transformer-ML, achieves high recognition accuracy through multi-level feature fusion. To balance performance and computational eficiency, two lightweight variants are developed: Sewer-MobileNet-ML (pure CNN) and Sewer-Mobile-TransNet (hybrid CNN-Transformer). Furthermore, a Category Importance Weighting loss is introduced to mitigate class imbalance.

## Multi-Label Classification Formulation

Unlike single-label tasks where each image belongs to exactly one class, sewer inspection images may contain multiple co-occurring defects, necessitating multi-label classification. While the overall architecture (Backbone, Neck, Head) remains similar, the key distinction lies in the output encoding and loss function.

Multi-label classification treats each defect class as an independent binary task. The model outputs $\mathbf { x } \in \mathbb { R } ^ { C }$ are passed through element-wise Sigmoid activation, and Binary Cross-Entropy (BCE) loss is applied:

$$
L ( x , y ) = \frac { 1 } { C } \sum _ { i = 1 } ^ { C } - \Bigl [ p _ { i } y _ { i } \log ( \sigma ( x _ { i } ) ) + ( 1 - y _ { i } ) \log ( 1 - \sigma ( x _ { i } ) ) \Bigr ]\tag{1}
$$

where � is the number of defect classes, $\mathbf { y } \in \{ 0 , 1 \} ^ { C }$ is the ground-truth label vector, $\sigma ( \cdot )$ is the Sigmoid function, and $p _ { i }$ is a class-specific weighting factor to address dataset imbalance.

## Sewer-Transformer-ML: Multi-level Vision Transformer

## Sewer-Transformer-ML Network Structure

To improve the accuracy and practicality of sewer pipeline defect recognition, this section proposes a multi-label classification model based on a multi-level vision Transformer, namely Sewer-Transformer-ML. This model uses Swin Transformer(Liu et al. 2021) as the backbone network, eliminating convolutional operations in the backbone and instead extracting defect features through self-attention mechanisms and fusing multi-level semantic information to enhance feature representation capability. The overall architecture is shown in Figure1, and is divided into Tiny, Small, Base, and Large versions based on the number of Swin Transformer Blocks in Stage 3, with Block counts of 12, 12, 16, and 24 respectively.

Sewer-Transformer-ML adopts a hierarchical design. First, the Patch Partition module divides the input image into non-overlapping patches. Setting patch size as $4 \times 4$ , each patch is flattened to a 48-dimensional feature vector $( 4 \times 4 \times 3 )$ . Subsequently, the Linear Embedding layer maps it to dimension �, with corresponding � values of 96, 96, 128, and 192 for the four versions.

Multi-level feature representation draws on the idea of multi-scale feature extraction in convolutional neural networks. The entire network consists of multiple feature extraction stages (Stage), finally connecting to fully connected layers after feature fusion. As network depth increases, the Patch Merge module gradually reduces feature map resolution to decrease computational complexity, while each Swin Transformer Block performs modeling within local windows. The feature map dimensions evolve as follows:

• Stage 1: Maintained as $( H / 4 , W / 4 , C )$

• Stage 2: Downsampled to (�/8<sub>,</sub> �/8<sub>,</sub> 2�)

• Stage 3: Downsampled to (�/16<sub>,</sub> �/16<sub>,</sub> 4�)

• Stage 4: Downsampled to (�/16<sub>,</sub> �/16<sub>,</sub> 8�)

This hierarchical design produces multi-scale features similar to ResNet and MobileNet, enabling flexible adaptation to diverse computer vision tasks.

![](images/e45d9e61a4de1181d48710f1004b0bf48247f4117edc74ef81d645f3ddf765af.jpg)  
Fig. 1. Multi-label sewer defect classification model based on a multi-level vision Transformer: Sewer-Transformer-ML (Base version)

Following the original Swin Transformer formulation (Liu et al. 2021), each stage alternates window-based multi-head self-attention (W-MSA) and shifted-window multi-head self-attention (SW-MSA). W-MSA limits attention computation to local windows, while SW-MSA enables information exchange across adjacent windows. This hierarchical local-attention design provides computationally eficient modeling of defect patterns at diferent spatial scales; the present study focuses on how features from these stages are fused for multi-label sewer defect classification rather than modifying the standard Swin block.

## Multi-level Vision Transformer Feature Fusion

The hierarchical feature representation in vision Transformer structures has inherent similarity with multi-scale feature extraction in CNN. In CNN, fusing features at diferent scales (i.e., high-level semantics and low-level details) has significantly improved model performance in various visual task network structure designs. Inspired by this, we propose fusing multi-level features of Transformer to enhance representation capability for pipeline defects with complex and variable morphologies. The overall architecture is shown in Figure 2. To fully fuse multilevel features extracted by Swin Transformer Blocks at diferent stages and prevent information loss from continuous downsampling, low-level features are downsampled to the same size as high-level features and then concatenated before being fed into fully connected layers for multilabel classification. Meanwhile, to explore the characteristics of CNN multi-scale features and Transformer multi-level features, this study compares the following two attention fusion strategies:

![](images/dd89ed2586b3d8b282686f78e49045bd3e7c13b7b097ab08fbb212d34b312d83.jpg)  
Fig. 2. Multi-label sewer defect classification based on a vision Transformer and multi-level feature fusion

1. Multi-scale feature fusion based on separated attention mechanism structure units (Zhang et al. 2022);

2. Convolutional module based on channel-spatial attention (Woo et al. 2018).

Specifically, after concatenating multi-level features upsampled or downsampled to uniform size, they are input into fusion units for attention screening to obtain recalibrated features (emphasizing important features and suppressing unimportant ones), which are then fed into fully connected layers for classification.

## Lightweight Variants for Edge Deployment

To reduce deployment cost, the two lightweight variants use MobileNetV3 (Howard et al. 2019), whose standard inverted residual bottlenecks combine depthwise separable convolution, channel expansion and projection, and squeeze-and-excitation. These established components are adopted without structural modification; the distinction between the variants lies in whether the extracted multi-scale CNN features are classified directly or fused using shifted-window self-attention.

## Sewer-MobileNet-ML Network Structure

Sewer-MobileNet-ML adapts the MobileNetV3-Large backbone to multi-label sewer defect classification by replacing its original classifier with a sigmoid-based multi-label prediction head. The resulting model contains approximately 17M parameters and serves as the pure-CNN lightweight baseline for evaluating the performance–complexity trade-of.

![](images/e59497e1ee3addb394689380758c5c5c03088d672798c40803bf1b679be9bcc0.jpg)  
Fig. 3. Sewer-Mobile-TransNet for sewer defect classification based on a lightweight network and multi-head shifted-window self-attention

## Sewer-Mobile-TransNet Network Structure

Sewer-Mobile-TransNet is a multi-label sewer defect classification model based on lightweight networks and multi-head shifted-window self-attention, with its overall structure shown in Figure 3. This model uses MobileNetV3 (Howard et al. 2019) as the backbone, with Swin Transformer Block based on the shifted-window multi-head self-attention mechanism serving as an important module in the Sewer-Mobile-TransNet structure. Multiple cascaded Bottleneck modules extract multi-scale CNN features. To fully fuse multi-scale information, Swin Transformer Block is introduced as a feature fusion module: first, feature maps at diferent scales are aligned to the same size through upsampling or downsampling and concatenated, then the shifted-window multi-head self-attention mechanism is used to screen and semantically correlate the concatenated features, modeling their spatial relationships to enhance feature representation capability for pipeline defects.

## DATASETS AND EVALUATION METHODS

## Dataset Introduction

This section employs two types of datasets: the CCTV pipeline image dataset Sewer-ML Dataset and the pipeline capsule dataset Sewer-Capsule Dataset. The former is primarily used to validate the capability of visual attention mechanisms in pipeline defect classification tasks and provide pre-trained weights for the latter for transfer learning.

The Sewer-ML dataset is currently the largest open-source multi-label dataset in the field of sewer defect classification, containing over 1.3 million unique images collected by CCTV equipment. The images were extracted and annotated by sewer pipeline inspection professionals from 75,618 inspection videos between 2011 and 2019. The dataset covers 17 types of pipeline defects and normal images, with each sewer pipeline defect assigned a weight factor (Class Importance Weights, CIW) based on actual operational conditions, representing the socio-economic impact of the defect. Larger weights indicate more severe hazards and higher detection priority. The defect codes, descriptions, and weight factors are shown in Table 1. The data covers main and branch pipelines of diferent materials, shapes, and sizes, with significant diferences across pipelines, posing considerable challenges for classification models.

TABLE 1. Sewer-ML Dataset Defect Category Codes and Weight Factors (Haurum and Moeslund 2021)
<table><tr><td>Code</td><td>Description</td><td>CIW</td></tr><tr><td>RB</td><td>Cracks, breaks, and collapses</td><td>1.0000</td></tr><tr><td>OB</td><td>Surface damage</td><td>0.5518</td></tr><tr><td>PF</td><td>Production error</td><td>0.2896</td></tr><tr><td>DE</td><td>Deformation</td><td>0.1622</td></tr><tr><td>FS</td><td>Displaced joint</td><td>0.6419</td></tr><tr><td>IS</td><td>Intruding sealing material</td><td>0.1847</td></tr><tr><td>RO</td><td>Roots</td><td>0.3559</td></tr><tr><td>IN</td><td>Infiltration</td><td>0.3131</td></tr><tr><td>AF</td><td>Settled deposits</td><td>0.0811</td></tr><tr><td>BE</td><td>Attached deposits</td><td>0.2275</td></tr><tr><td>FO</td><td>Obstacle</td><td>0.2477</td></tr><tr><td>GR</td><td>Branch pipe</td><td>0.0901</td></tr><tr><td>PH</td><td>Chiseled connection</td><td>0.4167</td></tr><tr><td>PB</td><td>Drilled connection</td><td>0.4167</td></tr><tr><td>OS</td><td>Lateral reinstatement cuts</td><td>0.9009</td></tr><tr><td>OP</td><td>Connection with transition pro-</td><td>0.3829</td></tr><tr><td>OK</td><td>file Construction changes</td><td>0.4396</td></tr></table>

The Sewer-ML Dataset is divided into training, validation, and test sets by video segments, consisting of 60,356, 7,692, and 7,570 independent videos respectively. Images from diferent datasets come from diferent sewer pipeline videos, ensuring no duplicate images across subsets with uniform distribution of normal and defect samples. The overall dataset is divided into 80% training set and 20% validation test set (roughly equal split). The statistics of image numbers for each subset are shown in Table 2. The ground truth labels for the test set are not publicly available, and model evaluation requires submission through the oficial website to ensure fair algorithm comparison.

TABLE 2. Sewer-ML Dataset Multi-label Dataset Image Quantity Distribution
<table><tr><td>Type</td><td>Training</td><td>Validation</td><td>Test</td></tr><tr><td>Normal images</td><td>552820</td><td>68681</td><td>69221</td></tr><tr><td>Defect images</td><td>487309</td><td>61365</td><td>60805</td></tr><tr><td>Total</td><td>1040129</td><td>130046</td><td>130026</td></tr></table>

The Sewer-Capsule Dataset is collected based on pipeline capsule inspection equipment and is a single-label dataset used to verify algorithm efectiveness and generalization capability. Compared with v1, this dataset version has redefined and labeled defect categories, containing 5 classes including detachment, breakage, deformation, obstacles, and normal images, with 2,353 images in the training set and 1,177 in the validation set. Detailed category statistics are shown in Table 3.

TABLE 3. Sewer-Capsule Dataset Defect Categories and Quantity Distribution
<table><tr><td>Type</td><td>Detachment</td><td>Breakage</td><td>Deformation</td><td>Obstacle</td><td>Normal</td></tr><tr><td>Training set</td><td>620</td><td>371</td><td>227</td><td>197</td><td>938</td></tr><tr><td>Validation set</td><td>281</td><td>158</td><td>93</td><td>84</td><td>561</td></tr><tr><td>Total</td><td>901</td><td>529</td><td>320</td><td>281</td><td>1499</td></tr></table>

## Evaluation Metrics

Performance is evaluated using accuracy, precision, recall, $F 1 _ { \mathrm { N o r m a l } }$ , mean average precision (mAP), and the Sewer-ML benchmark’s class-importance-weighted $F _ { 2 }$ metric $( F 2 _ { \mathrm { C I W } } )$ . Because sewer defect categories have diferent operational consequences, $F 2 _ { \mathrm { C I W } }$ combines the classwise $F _ { 2 }$ scores with the CIW factors listed in Table 1:

$$
F 2 _ { \mathrm { C I W } } = { \frac { \sum _ { c = 1 } ^ { C } F _ { 2 _ { c } } \cdot { \mathrm { C I W } } _ { c } } { \sum _ { c = 1 } ^ { C } { \mathrm { C I W } } _ { c } } }\tag{2}
$$

where $F _ { 2 _ { c } }$ is the $F _ { 2 }$ score for class � and $\mathrm { C I W } _ { c }$ is its importance weight. $F 1 _ { \mathrm { N o r m a l } }$ evaluates recognition of normal pipe images, and mAP summarizes performance across defect classes.

## COMPREHENSIVE EXPERIMENTS ON SEWER-ML DATASET

## Comprehensive Multi-label Classification Evaluation

This experiment evaluates the performance of the multi-level vision Transformer-based Sewer-Transformer-ML model and the lightweight MobileNet-V3-based Sewer-MobileNet-ML model in multi-label classification of sewer defect images, and compares them with multiple SOTA models. Specifically, we trained four models: Sewer-Transformer-ML-Tiny, Sewer-Transformer-ML-Small, Sewer-Transformer-ML-Base, and Sewer-MobileNet-ML. Among them, Transformer-based models were trained for 300 epochs, while Sewer-MobileNet-ML was trained for 200 epochs. Table 4 shows the performance comparison of diferent methods on validation and test sets. The ground truth labels of the test set are not publicly available, requiring prediction results to be uploaded to the Sewer-ML Defect Classification Challenge oficial website for automatic evaluation to ensure leaderboard fairness.

Leaderboard Performance. Notably, on the oficial Sewer-ML Defect Classification Challenge test set, our Sewer-Transformer-ML-Base achieves 65.68% $F 2 _ { \mathrm { C I W } }$ and 92.68% $F 1 _ { \mathrm { N o r m a l } } ,$ establishing a new state-of-the-art and ranking first on the public leaderboard. This represents a significant improvement of 7.6 percentage points over the second-best method (58.08%), and surpasses the previous best published result (TResNet-XL, 54.24%) by over 11 percentage points. Remarkably, even our lightweight Sewer-MobileNet-ML achieves 65.73% $F 2 _ { \mathrm { C I W } }$ with merely 17M parameters (approximately 95% parameter reduction compared to Sewer-Transformer-ML-Base and 94% reduction compared to TResNet-XL), outperforming all existing CNN-based methods and validating that small models can achieve state-of-the-art accuracy in this domain.

TABLE 4. Comparison of Metrics Among Diferent Methods (Sewer-ML Dataset)
<table><tr><td rowspan="2">Method</td><td colspan="3">Validation dataset</td><td colspan="2">Test dataset</td><td rowspan="2">size</td></tr><tr><td>mAP</td><td> $F 2 _ { \mathrm { C I W } }$ </td><td> $F 1 _ { \mathrm { N o r m a l } }$ </td><td> $F 2 _ { \mathrm { C I W } }$ </td><td> $F 1 _ { \mathrm { N o r m a l } }$ </td></tr><tr><td>Xie et al. (2019)</td><td></td><td>48.57</td><td>91.08</td><td>48.34</td><td>90.62</td><td>35M+35M</td></tr><tr><td>Chen et al. (2018)</td><td></td><td>42.03</td><td>3.96</td><td>41.74</td><td>3.59</td><td>93 M</td></tr><tr><td>Hassan et al. (2019)</td><td></td><td>13.14</td><td>0.00</td><td>12.94</td><td>0.00</td><td>220 M</td></tr><tr><td>Myrans et al. (2019)</td><td></td><td>4.01</td><td>26.03</td><td>4.11</td><td>27.48</td><td>140 M</td></tr><tr><td>ResNet-101 (He et al. 2016)</td><td></td><td>53.26</td><td>79.55</td><td>53.21</td><td>78.57</td><td>162 M</td></tr><tr><td>KSSNet (Wang et al. 2020)</td><td></td><td>54.42</td><td>80.60</td><td>54.55</td><td>79.29</td><td>173 M</td></tr><tr><td>TResNet-M (Ridnik et al. 2021)</td><td></td><td>53.83</td><td>81.23</td><td>53.79</td><td>79.91</td><td>112 M</td></tr><tr><td>TResNet-L (Ridnik et al. 2021)</td><td></td><td>54.63</td><td>81.22</td><td>54.75</td><td>79.88</td><td>205 M</td></tr><tr><td>TResNet-XL (Ridnik et al. 2021)</td><td></td><td>54.42</td><td>81.81</td><td>54.24</td><td>80.42</td><td>290 M</td></tr><tr><td>Sewer-Transformer-ML-Tiny</td><td>62.97</td><td>60.1</td><td>86.35</td><td>60.28</td><td>85.07</td><td>107 M</td></tr><tr><td>Sewer-Transformer-ML- Small</td><td>63.19</td><td>65.59</td><td>89.16</td><td>66.63</td><td>88.24</td><td>188M</td></tr><tr><td>Sewer-Transformer-ML-Base</td><td>68.28</td><td>67.09</td><td>93.2</td><td>65.68</td><td>92.68</td><td>333 M</td></tr><tr><td>Sewer-MobileNet-ML</td><td>63.09</td><td>66.01</td><td>90.1</td><td>65.73</td><td>89.53</td><td>17M</td></tr></table>

The models were evaluated on the complete validation set every 2 epochs. As shown by the $F 2 _ { \mathrm { C I W } }$ curves in Figure 4, Sewer-MobileNet-ML stabilized after approximately 50 epochs, whereas the Sewer-Transformer-ML variants stabilized after about 150 epochs. Sewer-MobileNet-ML was trained for 200 epochs and the Transformer variants for 300 epochs.

All experiments were conducted on 4 A100 GPUs (40GB per card) using multi-GPU distributed training. The average training time for 200 epochs, including validation every 2 epochs, is reported in Table 5; the lightweight model required substantially less training time than the Transformer variants.

TABLE 5. Average Time Reference for Training 200 Epochs
<table><tr><td>Method</td><td>Time (h)</td></tr><tr><td>Sewer-Transformer-ML-Tiny</td><td>~ 40 h</td></tr><tr><td>Sewer-Transformer-ML-Small</td><td>~79 h</td></tr><tr><td>Sewer-Transformer-ML-Base</td><td>~110 h</td></tr><tr><td>Sewer-MobileNet-ML</td><td>~ 20 h</td></tr></table>

## Multi-level Vision Transformer Feature Fusion Experiment

Using the Sewer-Transformer-ML Tiny version as the base model, multi-level fusion experiments were conducted with qualitative and quantitative comparisons against other classical attention mechanism fusion strategies. The loss curves for each strategy are shown in Figure 5, with

![](images/db5de7df8ef4d09eb24fbf21c216f939fb697e02b245144e30c731e74d75bb9b.jpg)  
Fig. 4. �2<sub>CIW</sub> Accuracy Curve Comparison

![](images/3a750061b09bf10d9b4f4d8c705120c4faf45796f52b722b13907180df468b49.jpg)  
Fig. 5. Loss curve comparison

$F 1 _ { \mathrm { N o r m a l } } , F 2 _ { \mathrm { C I W } }$ , and mAP accuracy evaluation curves shown in Figure 6, 7, and 8 respectively. Among them, Tiny represents the baseline model without multi-level fusion strategy (using Sewer-Transformer-ML Tiny architecture, trained for 300 epochs), CAT represents multi-level feature direct stacking strategy (trained for 300 epochs), STB represents separated attention mechanism fusion strategy (trained for 300 epochs), and CBAM represents channel and spatial combined attention mechanism fusion strategy (trained for 200 epochs). During training, each metric on the validation set was evaluated every 2 epochs.

Experimental results show that in multi-label sewer defect classification, directly merging and stacking multi-level Transformer features (CAT) can improve model performance to some extent, but other attention mechanism-based fusion strategies do not further enhance performance. Analysis suggests this may be because features extracted at each stage of Transformer have already completed information screening and fusion through the self-attention mechanism, and introducing additional attention modules may instead lead to redundant computation. The CBAM strategy performed significantly weaker than other strategies during training, and all accuracy metrics stopped growing or even slightly decreased between epochs 170-200. To avoid overfitting, this strategy was terminated early after 200 epochs.

![](images/82886b9117b3b3da47bf6c22ec3ae1cc79788777593c4e0704c71f586c8f3b62.jpg)  
Fig. 6. $\mathrm { F } _ { 1 }$ Precision Curve Comparison

![](images/4a4d41a689a19e205fbf56e8a7747453506a4dad9d63710489804d01e01d0246.jpg)  
Fig. 7. �2<sub>CIW</sub> Accuracy Curve Comparison

![](images/7f41557d2e4b45b5460f251b90884ba05668d0e2b6d3e1b5ee43ab465372c7e3.jpg)  
Fig. 8. mAP accuracy curve comparison

## Sewer-Mobile-TransNet Experiment

Lightweight design is a key factor for model practicality. Although the multi-level vision Transformer structure demonstrates excellent performance, its massive model weights (hundreds of MB) and long training time limit practical deployment. Therefore, we conducted lightweight experiments to reduce hardware costs, examine the potential for real-time detection, and shorten training cycles. Based on the Comprehensive Multi-label Classification Evaluation, Sewer-MobileNet-ML converges quickly, with all metrics stabilizing after approximately 50 epochs. In this experiment, Sewer-MobileNet-ML was trained for 105 epochs on the Sewer-ML dataset, with validation performed every 5 epochs. Sewer-Mobile-TransNet was trained for 100 epochs, with weights from Sewer-MobileNet-ML at epoch 90 fixed (only updating the subsequent fusion part). To contrast with the Multi-level Vision Transformer Feature Fusion experiment, CBAM and CAT multi-scale feature fusion strategies were also introduced. The model was evaluated on the validation set every epoch, with loss curves shown in Figure 9 and $F 1 _ { \mathrm { N o r m a l } } , F 2 _ { \mathrm { C I W } }$ , and mAP evaluation curves shown in Figures 10, 11, and 12 respectively.

Experimental results show that Sewer-Mobile-TransNet achieved the best performance across all evaluation metrics among lightweight architectures, further validating that small models with appropriate feature fusion can achieve state-of-the-art accuracy. Specifically, it demonstrates that even with approximately 95% parameter reduction, the lightweight model maintains performance competitive with full-scale architectures, validating the efectiveness of our proposed multi-scale feature fusion strategy based on sliding-window multi-head self-attention.

![](images/c7272c7e337f76de01191fcf9a15c674cca3339387493a63a978b228fd79f7b9.jpg)  
Fig. 9. Loss Curve Comparison

![](images/a4d023f0ea7b2186952f7baf80ecf3067fb2d0ef98cb722d1d4750a257f83cd0.jpg)  
Fig. 10. F<sub>1</sub> Score Curve Comparison

![](images/708595f8f489cd09989b72746634b4144bd68438677a8e2128e3aac74adf09e3.jpg)  
Fig. 11. mAP accuracy curve comparison

![](images/ea8d66bfc88f78a6cc66cc9aa6cd0e89a4f27ed9255aa9e9d35f8d3222c29035.jpg)  
Fig. 12. $F 2 _ { \mathrm { C I W } }$ Accuracy Curve Comparison

## SEWER-CAPSULE DATASET EXPERIMENTS

Sewer-Capsule Dataset is a single-label dataset. The proposed methods can also be directly applied to single-label image classification tasks, with the main diference being the loss function during training. To validate the efectiveness of the method on pipeline capsule data and fully utilize the model weights obtained from pre-training on the large-scale Sewer-ML Dataset, we employ comprehensive transfer learning to generalize the model’s classification capability to pipeline capsule data, improving the algorithm’s robustness and adaptability, and reducing the deep learning model’s dependency on training set data volume. Based on this, this section sets up two groups of comparative experiments.

## Sewer-Capsule Image Defect Classification Experiment

The Sewer-Capsule Dataset training set contains 2,353 images, and the validation set contains 1,177 images. Models are trained for 40 epochs, with validation set accuracy evaluated every 2 epochs. Table 6 shows the optimal accuracy of Sewer-Transformer-ML (Tiny version), Sewer-Mobile-TransNet, and ResNet-50 models on the validation set. Experimental results indicate that all models perform well on the pipeline capsule dataset, with Sewer-Mobile-TransNet achieving 96.43% accuracy, significantly outperforming Sewer-Transformer-ML (87.76%) and ResNet-50 (87.51%). This validates that small models achieve superior performance in transfer learning scenarios. Combined with the analysis in literature (Dosovitskiy et al. 2021) of vision Transformer on the large-scale ImageNet dataset, this result further confirms that in sewer defect classification tasks, pure Transformer structures may not necessarily outperform convolutional network structures on small datasets. This is also consistent with the experimental results on Sewer-ML Dataset above, indicating that Transformer architectures are more suitable for large-scale image processing and analysis scenarios.

TABLE 6. Sewer-Capsule Dataset Experimental Results
<table><tr><td>Method</td><td>Acc (%)</td><td> $F 1 _ { \mathrm { N o r m a l } } ( \% )$ </td><td>Precision(%)</td><td>Recall(%)</td></tr><tr><td>Sewer-Mobile-TransNet</td><td>96.43</td><td>94.90</td><td>93.53</td><td>96.72</td></tr><tr><td>Sewer-Transformer-ML</td><td>87.76</td><td>80.90</td><td>83.28</td><td>80.44</td></tr><tr><td>ResNet-50</td><td>87.51</td><td>81.10</td><td>82.35</td><td>81.12</td></tr></table>

## Transfer Learning Experiment

The training and validation set partition used in the Sewer-Capsule Image Defect Classification experiment, although helpful for obtaining good generalization performance, has high costs for acquiring high-quality ground-truth labels in practical applications. We expect to achieve better results with less data. To this end, we swapped the training and validation sets from that experiment, creating a setup with only 1,177 images in the training set and 2,353 images in the validation set. Simultaneously, each model has two groups of comparative experiments: using random initialization and using pre-trained weights based on the Sewer-ML dataset, respectively, to improve model classification performance through transfer learning. Models are trained for 30 epochs, with the validation set evaluated every 2 epochs. The Acc and $F 1 _ { \mathrm { N o r m a l } }$ metric trends are shown in Figures 13 and 14, respectively (Pretrain in the figures indicates model weights obtained from pre-training on the Sewer-ML dataset). Experimental results show that after using pre-trained weights, all accuracy metrics are significantly improved, validating the model’s strong cross-domain generalization capability even with limited data.

## DISCUSSION AND ANALYSIS

Experimental results demonstrate that both Sewer-Transformer-ML and lightweight Sewer-MobileNet-ML models exhibit excellent performance in multi-label classification of sewer defect images. Among them, the Sewer-Transformer-ML Base model achieved the best results on the Sewer-ML dataset and ranked first on the Sewer Defect Classification Challenge leaderboard, significantly outperforming the second-best method by 7.6 percentage points (65.68% vs. 58.08%). The Sewer-Transformer-ML Small and Tiny versions also surpassed the performance of classic convolutional neural network models such as TResNet-XL and ResNet-101. This result confirms that a purely Transformer-based architecture can efectively capture dependencies between labels in multi-label classification tasks, achieving better classification results than CNN structures. From the perspective of trade-ofs between model size and performance, although larger parameter models theoretically have stronger performance, the Small and Tiny versions, with nearly halfthe parameters reduced, only see a 1 to 6 percentage point drop in core evaluation metrics, demonstrating good cost-efectiveness.

![](images/1d488fbf478c0b3a9bab1882178e2565395a11d74170116169d2f2b9d4371956.jpg)  
Fig. 13. Accuracy Curve Comparison (Sewer-Capsule Dataset)

![](images/ecba17778fdd7cdc2b23a0481e4ae88fa9109cad64b818539ce85e5f8a6dfea5.jpg)  
Fig. 14. $\mathrm { F } _ { 1 }$ Accuracy Curve Comparison (Sewer-Capsule Dataset)

Although the Sewer-Transformer-ML Small and Tiny versions have been optimized in model size, they still struggle to meet deployment requirements for mobile or IoT terminals. In contrast, the lightweight model Sewer-MobileNet-ML, while achieving approximately 95% parameter reduction (only about one twentieth of Sewer-Transformer-ML), achieved state-of-the-art accuracy (65.73% $F 2 _ { \mathrm { C I W } } )$ among current end-to-end CNN methods and set a new record for lightweight architectures. Its core accuracy metrics are already very close to the Sewer-Transformer-ML Base version, with some metrics even surpassing the Small and Tiny variants, validating that small models can achieve superior performance.

Through comprehensive and systematic ablation experiments, to further explore the characteristics of Transformer multi-level features, we designed three fusion strategies: direct feature concatenation (CAT), separated attention mechanism (STB), and channel-spatial attention mechanism (CBAM). Comparative results show that directly stacking Transformer multi-level features can efectively improve model performance, while additionally introducing attention modules does not bring further improvement. Analyzing the reasons, features extracted at each stage of Transformer have already completed information screening and fusion through self-attention mechanisms. Applying attention mechanisms again may introduce redundant computation and instead limit performance improvement.

We conducted thorough experiments on the Sewer-Mobile-TransNet structure to further explore the diferences between CNN multi-scale features and Transformer multi-level features. Similar to the Multi-level Vision Transformer Feature Fusion experiment, we performed fusion analysis on multi-scale features extracted by MobileNetV3 and compared two strategies: direct concatenation and channel-spatial attention mechanism. The experiments found that using multi-head selfattention mechanism to fuse CNN multi-scale features can efectively improve model performance.

Comprehensive comparison of the Multi-level Vision Transformer Feature Fusion and Sewer-Mobile-TransNet experiments shows that multi-level features extracted by vision Transformer and multi-scale features extracted by CNN have essential diferences. This also indicates that conventional fusion strategies for CNN may not be efective for multi-level features of vision Transformer. CNN multi-scale features are more suitable for fusion through attention mechanisms to improve model performance, while direct concatenation of Transformer multi-level features yields better results. This discovery provides important reference basis for structural design of deep learning image classification models.

Through comprehensive transfer learning experiments, we validated the defect classification capability of the proposed method on the pipeline capsule dataset. Sewer-Mobile-TransNet achieved 96.43% accuracy under the standard data split with 2,353 training images. When the training set was reduced to 1,177 images, transferring pre-trained weights from the large-scale Sewer-ML dataset consistently improved model performance compared with random initialization. These results demonstrate that useful sewer defect representations can be transferred across inspection platforms, reducing dependence on target-domain annotated data and enhancing the engineering practicality of the method.

## Limitations and Future Work

Despite the demonstrated performance, this study has several limitations. First, the models were evaluated primarily on the Sewer-ML and Sewer-Capsule data sets. The Sewer-Capsule data set is relatively small and was collected using a specific inspection platform; therefore, the reported results may not fully represent generalization across diferent cities, pipe materials, defect standards, imaging conditions, and robotic systems. Second, the ground-truth labels of the Sewer-ML test set are not publicly available, and test performance must be obtained through the oficial evaluation server. This restriction prevents a more detailed independent analysis of class-specific errors and failure cases on the test set. Third, lightweight performance was assessed mainly through parameter count, classification accuracy, and training time. Inference latency, memory consumption, and energy use have not yet been systematically evaluated on mobile or embedded hardware. Consequently, the results demonstrate the potential for edge deployment rather than a completed field deployment. Future research will focus on cross-region and cross-device evaluation, hardware-level deployment tests, few-shot and weakly supervised learning, and extensions toward defect localization, severity assessment, and longitudinal condition analysis.

## CONCLUSION

This study addressed multi-label defect classification in urban sewer inspection images by developing Sewer-Transformer-ML, a hierarchical vision Transformer with multi-level feature fusion. Two lightweight architectures, Sewer-MobileNet-ML and Sewer-Mobile-TransNet, were further investigated to examine the balance among classification performance, model complexity, and engineering application potential.

On the Sewer-ML test set, Sewer-Transformer-ML-Base achieved an $F 2 _ { \mathrm { C I W } }$ of 65.68% and an $F 1 _ { \mathrm { N o r m a l } }$ of 92.68%, ranking first on the public challenge leaderboard and exceeding the secondranked method by 7.6 percentage points in the primary metric. Sewer-MobileNet-ML contained only 17 M parameters, approximately 95% fewer than the 333 M-parameter base model, while achieving an $F 2 _ { \mathrm { C I W } }$ of 65.73%. These results indicate that large model size is not the only route to high-performance sewer defect classification and that lightweight architectures can also achieve competitive performance.

The feature-fusion experiments further demonstrated that the appropriate fusion strategy depends on the underlying network architecture. Direct concatenation was more efective for multi-level Transformer features, which had already undergone self-attention-based information interaction, whereas attention-based fusion provided greater benefits for multiscale CNN features.

This distinction ofers a transferable computational design insight for feature-fusion models used in civil infrastructure image analysis.

Under the standard Sewer-Capsule data split, Sewer-Mobile-TransNet achieved 96.43% classification accuracy. In the reduced-data experiment using 1,177 training images, transferring pretrained knowledge from Sewer-ML consistently improved model performance. Overall, the proposed framework can support automated analysis of large-scale CCTV inspection data and shows potential for adaptation to emerging robotic inspection platforms. Further cross-region evaluation and testing on operational inspection hardware are required to establish field generalizability, computational eficiency, and long-term deployment reliability.

## APPENDIX I. EXPERIMENTAL TRAINING SETTINGS SUMMARY

## Sewer-ML Model Series Training Configurations

The detailed training configurations for the Sewer-Transformer-ML series models are presented below:

• Sewer-Transformer-ML Series Models: Trained on 4 Nvidia A100 GPUs, using the AdamW optimizer. The weight decay coeficient is set to 0.05. The learning rate is scaled with the total batch size according to the linear scaling rule:

$$
\mathrm { { I r } _ { a c t u a l } = 5 \times 1 0 ^ { - 4 } \times \frac { B a t c h \_ S i z e \times G P U \_ N U M } { 5 1 2 } }
$$

where GPU\_NUM (here, 4) denotes the number of GPUs used. The per-GPU batch sizes for diferent model variants are as follows: 320 for Tiny, 196 for Small, and 144 for Base.

• Sewer-MobileNet-ML: Trained on 4 Nvidia A100 GPUs, using the RMSprop optimizer with momentum set to 0.9 and weight decay coeficient set to $1 \times 1 0 ^ { - 5 }$ . The learning rate follows the same scaling rule as above. The per-GPU batch size is set to 512.

## Additional Experiment Configurations

• The Multi-level Vision Transformer Feature Fusion and Sewer-Mobile-TransNet experiments in the main text used the same training configurations as the Sewer-Transformer-ML and Sewer-MobileNet-ML models described in Appendix I, respectively.

## Comparative Study Training Configurations

• Sewer-Mobile-TransNet: Used the same optimizer and hyperparameter setup (RMSprop, etc.) as the Sewer-MobileNet-ML model in Appendix I, but with a batch size of 64 and trained on a single GPU.

• Sewer-Transformer-ML (Tiny): Used the same optimizer and hyperparameter setup (AdamW, etc.) as the Sewer-Transformer-ML model in Appendix I, with a batch size of 64 and trained on a single GPU.

• ResNet-50 (Baseline): Optimizer: SGD with Nesterov momentum (0.9). Initial learning rate: 0.003. Weight decay: 0.001. Batch size: 64. Trained on a single GPU.

## DATA AVAILABILITY STATEMENT

Some or all data, models, or code that support the findings of this study are available from the corresponding author upon reasonable request.

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (Grant No. 62272313).

## REFERENCES

Chen, K., Hu, H., Chen, C., Chen, L., and He, C. (2018). “An intelligent sewer defect detection method based on convolutional neural network.” 2018 IEEE International Conference on Information and Automation (ICIA), Wuyishan, China, 1301–1306 (Aug.).

Dosovitskiy, A. et al. (2021). “An image is worth 16x16 words: Transformers for image recognition at scale.” arXiv:2010.11929 [cs] Accessed: Sep. 16, 2021.

Goharinezhad, S., Hadi, A., and Mirzaei, S. (2025). “Automated defect classification and localization in sewer pipelines using hybrid resnet50–swin transformer and modified yolov8 on cctv inspection images.” Scientific Reports, 15(1), 41755.

Halfawy, M. R. and Hengmeechai, J. (2014). “Eficient algorithm for crack detection in sewer images from closed-circuit television inspections.” Journal of Infrastructure Systems, 20(2), 04013014.

Hassan, S. I. et al. (2019). “Underground sewer pipe condition assessment based on convolutional neural networks.” Automation in Construction, 106, 102849.

Haurum, J. B. and Moeslund, T. B. (2021). “Sewer-ML: A multi-label sewer defect classification dataset and benchmark.” arXiv:2103.10895 [cs] Accessed: Jul. 28, 2021.

He, K., Zhang, X., Ren, S., et al. (2016). “Deep residual learning for image recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 770–778.

Howard, A., Sandler, M., Chu, G., et al. (2019). “Searching for mobilenetv3.” Proceedings of the IEEE/CVF International Conference on Computer Vision, Seoul, 1314–1324.

Kumar, S. S., Abraham, D. M., Jahanshahi, M. R., Iseley, T., and Starr, J. (2018). “Automated defect classification in sewer closed circuit television inspections using deep convolutional neural networks.” Automation in Construction, 91, 273–283.

Li, D., Cong, A., and Guo, S. (2019). “Sewer damage detection from imbalanced CCTV inspection data using deep convolutional neural networks with hierarchical classification.” Automation in Construction, 101, 199–208.

Liu, Z. et al. (2021). “Swin Transformer: Hierarchical vision transformer using shifted windows.” arXiv:2103.14030 [cs] Accessed: Jul. 02, 2021.

Meijer, D., Scholten, L., Clemens, F., and Knobbe, A. (2019). “A defect classification methodology for sewer image sets with convolutional neural networks.” Automation in Construction, 104, 281–298.

Meng, C., Cheng, Q., Zhu, J., Rao, Y., Liu, Z., Liu, F., Wang, K., Du, D., Liu, F., and Chen, Y. (2025). “Enhancing drainage pipe defect detection through multi-model benchmarking and transfer learning: Leveraging public and in-field datasets for practical engineering applications.” Journal of Water Process Engineering, 77, 108506.

Myrans, J., Everson, R., and Kapelan, Z. (2019). “Automated detection of fault types in CCTV sewer surveys.” Journal ofHydroinformatics, 21(1), 153–163.

Nguyen, C. L., Nguyen, A., Brown, J., and Dang, L. M. (2025). “Sewer pipeline condition assessment and defect detection using computer vision.” Automation in Construction, 179, 106479.

Ridnik, T., Lawen, H., Noy, A., et al. (2021). “Tresnet: High performance gpu-dedicated architecture.” Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, Online, 1400–1409.

Shehab, T. and Moselhi, O. (2005). “Automated detection and classification of infiltration in sewer pipes.” Journal of Infrastructure Systems, 11(3), 165–171.

Su, T. C. and Yang, M. D. (2014). “Application of morphological segmentation to leaking defect detection in sewer pipelines.” Sensors, 14(5), 8686–8704.

Wang, Y., He, D., Li, F., et al. (2020). “Multi-label classification with label graph superimposing.” Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 34, New York, 12265–12272.

Woo, S., Park, J., Lee, J. Y., et al. (2018). “Cbam: Convolutional block attention module.” Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 3–19.

Xie, Q., Li, D., Xu, J., Yu, Z., and Wang, J. (2019). “Automatic detection and classification of sewer defects via hierarchical deep learning.” IEEE Trans. Automat. Sci. Eng., 16(4), 1836–1847.

Yang, M.-D. and Su, T.-C. (2008). “Automated diagnosis of sewer pipe defects based on machine learning approaches.” Expert Systems with Applications, 35(3), 1327–1337.

Zhang, B., Yuan, H., Ge, J., Cheng, L., Li, X., and Xiao, C. (2024). “Weak appearance aware pipeline leak detection based on cnn–transformer hybrid architecture.” IEEE Transactions on Instrumentation and Measurement, 74, 1–12.

Zhang, H., Wu, C., Zhang, Z., et al. (2022). “Resnest: Split-attention networks.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, Louisiana, USA, 2736–2746.