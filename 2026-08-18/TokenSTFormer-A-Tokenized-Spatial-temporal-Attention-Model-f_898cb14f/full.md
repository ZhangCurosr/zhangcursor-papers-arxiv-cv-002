# TokenSTFormer: A Tokenized Spatial-temporal Attention Model for Holistic Motion Analysis in Adolescent Idiopathic Scoliosis Screening

Dong Chen<sup>1</sup>, Kenneth M.C. Cheung<sup>1</sup>

1The University of Hong Kong olichen@connect.hku.com; oliver.cd@outlook.com

Abstract. Adolescent Idiopathic Scoliosis (AIS) is a prevalent spinal deformity in adolescents that, if left untreated, can result in severe health outcomes. Traditional screening methods are limited by subjective interpretation, reliance on professional expertise and low scalability. To address these challenges, we present ScoliGait dataset, which comprises 1,516 gait video clips paired with corresponding X-ray records. We also introduce TokenSTFormer, a novel model that tokenizes spatial and temporal semantics to enhance feature representation and convergence. Our model achieves state-of-the-art performance, surpassing vanilla Vision Transformer encoder across key metrics, including accuracy of 0.79. This study highlights the potential of leveraging holistic motion features derived from gait video and attention-based models for scalable, cost-effective AIS screening, paving the way for future clinical applications in scoliosis detection.

Keywords: Adolescent Idiopathic Scoliosis, Gait Analysis, Kinematic Knowledge Map, Spatial-Temporal Tokenization.

## 1 Introduction

Adolescent idiopathic scoliosis (AIS) is a structural, lateral curvature of the spine accompanied by vertebral rotation, typically diagnosed during adolescence. Affecting approximate 5% of children globally [1, 2], AIS can lead to severe health consequences if left untreated, such as chronic back pain and psychosocial distress [3, 4]. The current gold standard for diagnosing AIS requires radiographic imaging and measuring the coronal Cobb Angle (CA) >10°, which indicates scoliosis [3]. However, repeated exposure to X-rays raises concerns about cumulative radiation risk, highlighting the need for non-invasive and scalable screening methods.

Early screening is therefore critical to preventing curve progression and enabling timely intervention, especially during adolescence when the spine is still growing. Traditional AIS screening methods, such as the Adams’ Forward Bending Test and Scoliometer Measurement, are widely used in clinical and school-based settings [3]. However, their effectiveness and efficiency vary across studies [5, 6], often being influenced by subjective interpretation and factors like participants’ obesity [7]. Moreover, these methods face significant challenges in scalability for large-scale screening programs, as they rely on professional expertise, involve high equipment costs, and raise privacy concerns [3].

Advances in technology have introduced alternative approaches for AIS screening, such as using single-camera photographs to analyze back asymmetry [8]. However, methods of using static information fail to incorporate kinematic features providing bio-mechanical insights [9]. Additionally, methods like GaitEdge [10] and SkeletonGait [11] extract spatiotemporal features using synthetic silhouette maps and skeleton maps. Despite their effectiveness, the complex preprocessing pipelines required by these methods limit their practicality in real-world applications.

To address these challenges, we propose ScoliDetect<sup>TM</sup>, a novel system for AIS screening based on gait video analysis, as illustrated in Fig. 1. Specifically, we (1) establish ScoliGait dataset containing 1,516 gait videos paired with corresponding spinal X-ray images and CA measurements. (2) design a de-identified kinematic knowledge map to represent the kinematic features of holistic gait motion, enabling scalable and privacy-conscious screening on mobile devices. (3) introduce TokenSTFormer, a model equipped with Spatial-Temporal Tokenization (STT) to enhance features representation and model convergence.

![](images/cd13b3dc49116ada18c685cc180fd0b5f75197084dae19e943a2dbb0ab1a3af5.jpg)  
Fig. 1. Workflow of ScoliDetectTM system. A gait video recorded using a mobile phone camera is processed to construct a kinematic knowledge map, representing holistic motion features. TokenSTFormer model incorporates Spatial-Temporal Tokenization (STT).

## 2 Dataset

## 2.1 Dataset Description

The ScoliGait dataset was collected using a mobile phone camera at \*\*\* Hospital to support scoliosis screening study. A total of 758 participants who signed informed consent form were enrolled in this study. Basic demographic data is summarized in

Table 1. The dataset was expanded by segmenting non-overlapping video clips to enhance inference robustness across different walking periods. Each clip, recorded at 30Hz frame rate and 1080p resolution, captured 5 seconds of holistic walking motion. In total, the final dataset comprises 1,516 video clips.

To the best of our knowledge, ScoliGait is the first dataset to include both gait videos and corresponding spinal X-rays, which serve as the golden standard medical labels. The annotation quality was validated by senior medical doctors. Notably, the ground truth for scoliosis diagnosis relies on radiographic measurements, unlike traditional screening methods which lack sufficient evidence to serve as reliable labels.

Table 1. Summary of demographic and clinical attributes in the ScoliGait dataset
<table><tr><td>Attributes</td><td>Positive (Cobb angle&gt;10°)</td><td>Negative (Cobb angle≤10°)</td></tr><tr><td>Number of participants</td><td colspan="2">758 subjects having informed consent form</td></tr><tr><td>Non-overlapped video clips</td><td colspan="2">1516 non-overlapping clips (150 frames with 30Hz frame rate)</td></tr><tr><td>Gender(F/M)</td><td>722/320</td><td>275/198</td></tr><tr><td>Age (mean, std)</td><td colspan="2">13.86, 2.44 11.59, 2.86</td></tr></table>

## 2.2 Data collection and preprocessing

The setup and recording process are illustrated in Fig. 1. The camera was positioned at a height of 2.5 meters to capture shoulder-pelvic angles. Participants were instructed to walk at a natural pace, completing one forward-and-backward cycle along a 4-meter-long path, starting 2 meters away from the camera.

## 3 Methodology

This section involves the way of constructing kinematic knowledge maps (Fig. 1) from pose estimation data and designing STT modules to effectively capture gait features.

Given a video clip

$$
\mathsf { V } _ { \mathrm { i } } ^ { ( \mathrm { t } , \mathrm { w } , \mathrm { h } , \mathrm { c } ) } = \{ \mathrm { f } _ { 1 } , \mathrm { f } _ { 2 } , . . . , \mathrm { f } _ { \mathrm { n } } \}\tag{1}
$$

where $f _ { n }$ represents the $n ^ { t h }$ frame of $i ^ { t h }$ subject, (t, w, h, c) represent period, width, height and channel of frames.

$$
\mathsf { M } _ { \mathrm { i } } ^ { \mathrm { ( t , v ) } } = \emptyset ( \tau \ast \mathsf { F } ( \mathsf { V } _ { \mathrm { i } } ^ { \mathrm { ( t , w , h , c ) } } ) )\tag{2}
$$

where $M _ { i } ^ { ( t , v ) }$ represents the kinematic knowledge map with t period and v variates; � is the 2D pose estimation technology; � is an amplifying factor which is 1000 in our setting; ∅ is a prior knowledge function of landmark coordinates transformation.

## 3.1 Kinematic Knowledge Map

To extract kinematic features, we utilized pose estimation technology (YoLoV8) to derive 2D joint coordinates (x, y) from the gait videos [12]. Prior studies have [13, 14] approved that scoliotic gait motion exhibits detectable deviations compared to normal gait. These deviations are primarily induced by musculoskeletal, perceptron or post-adaptive issues. In terms of these prior knowledge, kinematic knowledge map is constructed in three domains: (1) features representing the overall gait pattern in motion space, (2) features capturing the subject's skeletal structure in self-skeleton space, (3) features derived from motion lagging and signal relationships.

The kinematic knowledge map, as shown in Fig. 2, comprises 238 features that represent holistic motion, composing 140 features in motion space, 32 features in selfskeleton space, and 66 features for signal correlation. The numerical values in each section are dependently normalized. Specifically, paired joint-related features are calculated using Euclidean distance, while motion angles between vectors are determined using trigonometric functions. Motion lagging sections are derived through signal cross-correlation, calculated using the SciPy package.

![](images/7a591411505d6558ebfad73339e7c5c4a32eda90b2859f1c1992afecfc57ebf7.jpg)  
Fig. 2. A kinematic knowledge map representing holistic motion features comprises 238 features that represent holistic motion, composing 140 features in motion space, 32 features in self-skeleton space, and 66 features for signal correlation.

## 3.2 Model Architecture

The general architecture of TokenSTFormer is inspired by the Vision Transformer, having residual blocks composed of Multiheaded Self-Attention (MSA) and Multi-

Layer Perceptron (MLP) [15]. The proposed Spatial-Temporal Tokenization is illustrated in Fig. 3.

Building on insights from a prior study [16], a Dense layer is applied after spatial tokens to enhance feature representation across variates. Additionally, other key modules, such as LayerScale [17] and Stochastic Depth [18], are integrated in standard configurations.

Spatial-Temporal Tokenization. The kinematic knowledge map $M _ { i } ^ { ( t , ~ v ) }$ , with t period and v variates, is tokenized into spatial and temporal tokens using 2D convolutional layers with column-size and row-size kernels, respectively. Temporal tokens $z _ { t e m p o r a l } ^ { ( t , ~ d ) }$ and spatial tokens $z _ { s p a t i a l } ^ { ( v , ~ d ) }$ have d output dimension. Both tokens followed LayerNorm (LN) are concatenated as $z _ { i n p u t } ^ { ( t + v , \ d ) }$

$$
z _ { i n p u t } ^ { ( t + v , ~ d ) } = c o n c a t ( L N \Big ( z _ { t e m p o r a l } ^ { ( t , ~ d ) } \Big ) , L N ( \mathrm { D e n s e } ( z _ { s p a t i a l } ^ { ( v , ~ d ) } ) ) \Big )\tag{3}
$$

The main loss function is binary cross entropy (BCM). CLS tokens are respectively applied for temporal and spatial embeddings in our experiments. Auxiliary loss is calculated by Mean Squared Error of these two CLS tokens.

$$
L o s s _ { C L S } = M S E \big ( C L S _ { t e m p } , C L S _ { s p t } \big )\tag{4}
$$

$$
L o s s = L o s s _ { B C M } + L o s s _ { C L S }\tag{5}
$$

![](images/ce6774ec5aeebcdd1be248f55b3712322df28075d36c7a4c3e8ce56a66bff10a.jpg)  
Fig. 3. The figure is Spatial-Temporal Tokenization module

## 3.3 Metrics

Key metrics include accuracy, sensitivity, specificity, Positive Predictive Value $\mathrm { ( P V + ) }$ , and Negative Predictive Value $\mathrm { ( P V - ) }$ . These metrics provide a comprehensive understanding of the model's ability to distinguish between conditions of interest $( \mathrm { e . g . , }$ disease vs. no disease) based on test outcomes. Here, TN represents true negative; $T P$ represents true positive; FN represents false negative; FN represents false negative.

Accuracy: Measures the proportion of correctly classified instances (both positive and negative) among all samples.

$$
\mathrm { A c c u r a c y } ~ = ~ \mathrm { ( T N + T P ) } / \mathrm { ( T N + F N + T P + F P ) }\tag{6}
$$

Positive Predictive Value (PV+): Indicates the proportion of true positive predictions among all positive predictions.

$$
\mathrm { P o s i t i v e ~ P r e d i c t i v e ~ V a l u e ~ = ~ T P / ( T P + F P ) }\tag{7}
$$

Negative Predictive Value (PV−): Represents the proportion of true negative predictions among all negative predictions.

$$
\Delta \mathrm { e g a t i v e ~ P r e d i c t i v e ~ V a l u e } ~ = ~ \mathrm { T N } / ( \mathrm { T N } + \mathrm { F N } )\tag{8}
$$

Sensitivity: Measures the ability of the model to correctly identify true positive cases (i.e., the proportion of actual positives correctly predicted).

$$
\mathrm { S e n s i t i v i t y } = \mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } )\tag{9}
$$

Specificity: Reflects the ability of the model to correctly classify true negative cases (i.e., the proportion of actual negatives correctly predicted).

$$
\mathrm { S p e c i f i c i t y } = \mathrm { T N } / ( \mathrm { T N } + \mathrm { F P } )\tag{10}
$$

## 4 Experiments

## 4.1 Experimental setup

In this section, the proposed TokenSTFormer model is compared with a vanilla Vision Transformer encoder [15], configured with a 6 by 6 patch size and the same hyperparameters to those of the TokenSTFormer model. The training, validation and testing datasets consisted of 1216, 150, 150 samples, respectively. To reduce class imbalance, we stratified the data samples with a 2.2:1 positive-to-negative ratio. Both categories were adequately represented and minimized bias during model evaluation.

Additionally, the contributions of SST were analyzed in ablation study. Table 2 is the details of hyperparameters setting used in this study.

Table 2. Hyperparameter details for model configuration and training
<table><tr><td>Model parameters</td><td>Training parameters</td></tr><tr><td>MLP dimension = 384</td><td>Learning rate = 2e-5</td></tr><tr><td>Number of heads = 6</td><td>Warmup ratio = 0.1</td></tr><tr><td>Number of layers = 5</td><td>Cosine learning rate schedule</td></tr><tr><td>MHA dimension = 6*256</td><td>Optimizer: Adam</td></tr><tr><td>Dropout = 0.1</td><td>Batch size = 64</td></tr></table>

## 4.2 Results

Compared to Vision Transformer encoder [15], the proposed TokenSTFormer consistently outperformed the baseline across all evaluation metrics. As shown in Table 3, TokenSTFormer achieved accuracy of 0.787, demonstrating its superior overall classification capability. Additionally, it achieved a sensitivity of 0.845 and specificity of 0.660, underscoring its effectiveness in correctly identifying both positive and negative cases.

Moreover, TokenSTFormer excelled in predictive values, with PV+ (0.845) and PV− (0.660), highlighting its reliability in making accurate and balanced predictions. These results highlighted the robustness and efficiency of TokenSTFormer for AIS screening.

Table 3. Comparison of evaluation metrics among TokenSTFormer, Vision Transformer encoder, and traditional methods
<table><tr><td></td><td>Accuracy</td><td>Sensitivity</td><td>Specificity</td><td>PV+</td><td>PV-</td></tr><tr><td rowspan="2">Paper Report [5, 6]</td><td></td><td>0.46</td><td>0.84</td><td>0.30</td><td></td></tr><tr><td></td><td>0.51</td><td>0.96</td><td>0.95</td><td>0.53</td></tr><tr><td>Average</td><td></td><td>0.37</td><td>0.90</td><td>0.80</td><td>0.59</td></tr><tr><td></td><td></td><td>0.447</td><td>0.900</td><td>0.683</td><td>0.560</td></tr><tr><td>Transformer encoder</td><td>0.740</td><td>0.796</td><td>0.617</td><td>0.820</td><td>0.580</td></tr><tr><td>TokenSTFormer</td><td>0.787</td><td>0.845</td><td>0.660</td><td>0.845</td><td>0.660</td></tr></table>

## 4.3 Ablation studies

We conducted ablation studies to analyze certain feature effects on performance. As shown in Table 4, LayerNorm and independent positional encoding play critical roles in SST modules to ensure the model’s accuracy and robustness.

Table 4. Metrics comparison among TokenSTFormer, w/o Spatial-Temporal Tokenization.
<table><tr><td></td><td>Accuracy</td><td>Sensitivity</td><td>Specificity</td><td>PV+</td><td>PV-</td></tr><tr><td>SST w/o LayerNorm</td><td>0.720</td><td>0.738</td><td>0.681</td><td>0.835</td><td>0.542</td></tr><tr><td>Single pos encoding</td><td>0.687</td><td>0.728</td><td>0.600</td><td>0.798</td><td>0.500</td></tr><tr><td>TokenSTFormer</td><td>0.787</td><td>0.845</td><td>0.660</td><td>0.845</td><td>0.660</td></tr></table>

Spatial-Temporal Tokenization. The purpose of STT is to separate and normalize temporal and spatial tokens, thereby minimizing the distance between two types of tokens. To further analyze its impact, we analyzed the cosine similarity of spatialtemporal tokens between baseline model and that without LayerNorm in STT, as illustrated in Fig. 4. The majority of points lie above the red dashed line (slope = 1), indicating that the cosine similarity in the baseline model is significantly smaller. These results demonstrate that STT effectively learns temporal-spatial specific transformations to reproject tokens into an enhanced semantic space.

Number of layers. We evaluated the performance metrics of models with varying numbers of attention blocks. As illustrated in Fig. 5, the model achieves the highest accuracy when the number of attention blocks is set to 5. This suggests that an optimal balance between model complexity and performance is achieved at this configuration.

![](images/72d801d976d5c638308b2a819bcee311df9b0c458a15deadf49fbdd3bbf24555.jpg)  
Fig. 4. Scatter plot comparing token cosine similarity distances between the TokenSTFormer and that without STT.

![](images/7f4143f2c8e2e445124759a86b039bcdd191ab4313592013a2ea46576e5db72f.jpg)  
Fig. 5. The plot shows the variation in performance as the number of attention block (layers) increases.

## 5 Conclusions

In this study, we introduced ScoliDetect<sup>TM</sup>, an AI-assisted holistic motion analysis system utilizing a smartphone camera for scalable and cost-effective scoliosis screening. By leveraging Spatial-Temporal Tokenization (SST), the proposed TokenSTFormer effectively learns robust spatial-temporal features. This work represents a promising step toward the deployment of accessible, accurate, and efficient diagnostic tools for scoliosis screening and monitoring.

## References

1. Hengwei, F., Zifang, H., Qifei, W., Weiqing, T., Nali, D., Ping, Y., Junlin, Y.: Prevalence of Idiopathic Scoliosis in Chinese Schoolchildren: A Large, Population-Based Study. Spine, 41(3), 259-64 (2016)

2. Catanzariti, JF., Rimetz, A., Genevieve, F., Renaud, G., Mounet, N.: Idiopathic adolescent scoliosis and obesity: prevalence study. Eur Spine J, 32 (6), 2196-2202 (2023)

3. Luk, K. D., Lee, C. F., Cheung, K. M., Cheng, J. C., Ng, B. K., Lam, T. P., Mak, K. H., Yip, P. S., Fong, D. Y.: Clinical effectiveness of school screening for adolescent idiopathic scoliosis: a large population-based retrospective cohort study. Spine, 35 (17), 1607-14 (2010)

4. Cheng, J. C., Castelein, R. M., Chu, W. C., Danielsson, A. J., Dobbs, M. B., Grivas, T. B., Gurnett, C. A., Luk, K. D., Moreau, A., Newton, P. O., Stokes, I. A., Weinstein, S. L., & Burwell, R. G.: Adolescent idiopathic scoliosis. Nat Rev Dis Primers, 1, 15030 (2015)

5. Amendt, L.E., Ause-Ellias, K.L., Eybers, J.L., Wadsworth, C.T., Nielsen, D.H., Weinstein, S.L.: Validity and reliability testing of the Scoliometer®. Physical therapy, 70(2), 108-117 (1990)

6. Coelho, D.M., Bonagamba, G.H., Oliveira, A.S.: Scoliometer measurements of patients with idiopathic scoliosis. Brazilian journal of physical therapy, 17(2), 179-184 (2013)

7. Margalit, A., McKean, G., Constantine, A., Thompson, C. B., Lee, R. J., Sponseller, P. D.: Body Mass Hides the Curve: Thoracic Scoliometer Readings Vary by Body Mass Index Value. Journal of pediatric orthopedics, 37(4), e255-e260 (2017)

8. Zhang, T., Zhu, C., Zhao, Y., Zhao, M., Wang, Z., Song, R., Meng, N., Sial, A., Diwan, A., Liu, J., Cheung, J. P. Y.: Deep Learning Model to Classify and Monitor Idiopathic Scoliosis in Adolescents Using a Single Smartphone Photograph. JAMA Network Open, 6 (8), e2330617 (2023)

9. Pesenti, S., Prost, S., Pomero, V., Authier, G., Roscigni, L., Viehweger, E., Blondel, B., Jouve, J. L.: Does static trunk motion analysis reflect its true position during daily activities in adolescent with idiopathic scoliosis?. Orthopaedics & traumatology, surgery & research: OTSR, 106(7), 1251–1256 (2020)

10. Liang, J., Fan, C., Hou, S., Shen, C., Huang, Y., Yu, S.: Gaitedge: Beyond plain end-toend gait recognition for better practicality. In: European Conference on Computer Vision, pp. 375-390. Springer Nature Switzerland, (2022)

11. Fan, C., Ma, J., Jin, D., Shen, C., Yu, S.: SkeletonGait: Gait Recognition Using Skeleton Maps. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1662-1669, (2024).

12. Glenn Jocher, Ayush Chaurasia, and J. Qiu. "Ultralytics YOLOv8." https://github.com/ultralytics/ultralytics (accessed 2024)

13. Ji, R., Liu, X., Liu, Y., Yan, B., Yang, J., Lee, W. Y., Wang, L., Tao, C., Kuai, S., Fan, Y.: Kinematic difference and asymmetries during level walking in adolescent patients with different types of mild scoliosis. Biomedical engineering online, 23(1), 22 (2024)

14. Boulcourt, S., Badel, A., Pionnier, R., Neder, Y., Ilharreborde, B., Simon, A. L.: A gait functional classification of adolescent idiopathic scoliosis (AIS) based on spatio-temporal parameters (STP). Gait & Posture, 102, 50–55 (2023)

15. Dosovitskiy, A., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

16. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M.: itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625 (2023)

17. Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H.: Going deeper with image transformers. In: Proceedings of the IEEE/CVF international conference on computer vision, pp. 32-42. IEEE (2021)

18. Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: Computer Vision–ECCV 2016: 14th European Conference, Part IV 14, pp. 646- 661. Springer International Publishing, Amsterdam, The Netherlands (2016)