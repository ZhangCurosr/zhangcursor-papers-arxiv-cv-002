# Multimodal RGB–Infrared Combination for UAV-Based Wildfire Segmentation: A Comparative Study on FLAME3

Matheus F. Kovaleski<sup>∗</sup>, Lu´ıs Garrote<sup>∗</sup>, Cristiano Premebida<sup>∗</sup>, Jer´ ome Mendesˆ <sup>†</sup>, and Joao Ruivo Paulo˜ <sup>∗</sup> <sup>∗</sup> Institute of Systems and Robotics, Department of Electrical and Computer Engineering, University of Coimbra, Portugal Email: {matheus.kovaleski, garrote, cpremebida, jpaulo}@isr.uc.pt

<sup>†</sup> University of Coimbra, CEMMPRE, ARISE, Department of Mechanical Engineering, Polo II, PT-3030-788 Coimbra, Portugal Email: jerome.mendes@uc.pt

Abstract—Unmanned Aerial Vehicles (UAVs) have emerged as a promising platform for firefighting operations due to their flexibility, low operational cost, and ability to acquire highresolution imagery in locations that may be difficult or dangerous to access using conventional methods. Recent advances in deep learning have significantly improved the capabilities of UAVbased wildfire monitoring systems. The present work investigates RGB-infrared fusion for binary wildfire segmentation on the FLAME3 dataset. In this Study, RGB and Infrared baselines are compared with three representative fusion strategies across three segmentation architectures, including U-Net, DeepLabV3+, and SegFormer. The key motivation of this work is to analyze the contribution of each modality, evaluate the impact of fusion timing, and examine how different network architectures exploit multimodal information for UAV wildfire delineation. The findings indicate that thermal information plays a dominant role in UAV segmentation and that feature-level multimodal fusion combined with transformer-based architectures offers the most promising direction for future research.

Keywords—wildfire, UAV, Fusion Deep learning, remote sensing.

## I. INTRODUCTION

Wildfires have become increasingly frequent and severe in many regions of the world, causing substantial environmental, economic, and societal impacts. Climate change, prolonged droughts, and increasing human activity have contributed to the growing intensity and frequency of wildfire events, creating an urgent need for effective monitoring and rapid response systems [1]. Early identification of active fire fronts is critical for supporting firefighting operations, reducing burned areas, and minimizing risks to human life and infrastructure.

Unmanned Aerial Vehicles (UAVs) have emerged as a promising platform for wildfire monitoring due to their flexibility, low operational cost, and ability to acquire highresolution imagery in locations that may be difficult or dangerous to access using conventional methods [2], [3]. Compared with satellite-based systems, UAVs can provide imagery with higher spatial resolution and lower latency, enabling detailed observation of fire behavior and supporting near-real-time decision making during emergency operations.

Recent advances in deep learning have significantly improved the capabilities of UAV-based wildfire monitoring systems. Convolutional Neural Networks (CNNs) and Transformer-based architectures have demonstrated remarkable performance in image understanding tasks, including wildfire classification, detection, and semantic segmentation [1]. Among these approaches, semantic segmentation is particularly attractive because it provides pixel-level delineation of fire regions, offering more detailed spatial information than image-level classification or bounding-box detection. Architectures such as U-Net, DeepLabV3+, and Vision Transformers have achieved state-of-the-art performance in aerial wildfire segmentation tasks [4]–[6].

Most existing wildfire segmentation methods rely primarily on RGB imagery. Although visible-spectrum images provide rich contextual information and detailed spatial structures, their performance may degrade under adverse conditions such as dense smoke, illumination variations, shadows, and lowcontrast fire boundaries [7]. Thermal infrared (IR) imagery offers complementary information by directly capturing heat emissions from active fire regions and remaining relatively robust to changes in lighting conditions. Recent studies have shown that combining visible and thermal modalities can significantly improve wildfire recognition and segmentation performance by exploiting complementary visual and thermal cues [8], [9].

The recent introduction of the FLAME 3 dataset [10] represents an important step toward multimodal wildfire analysis. Unlike previous versions of the FLAME benchmark, FLAME 3 provides synchronized visible-spectrum and radiometric thermal imagery acquired from UAV platforms, enabling the investigation of multimodal learning strategies for wildfire management tasks [11]. The availability of radiometric thermal measurements creates new opportunities for studying how thermal information can complement conventional RGB observations in segmentation applications.

Despite the increasing interest in multimodal wildfire sensing, there is still a limited understanding of how different fusion strategies influence segmentation performance. Existing works often evaluate a single fusion approach or focus on a specific architecture, making it difficult to isolate the effects of modality selection and fusion timing. Furthermore, systematic comparisons between RGB-only, infrared-only, and multimodal approaches under a unified experimental protocol remain scarce.

To address this gap, this paper investigates RGB–infrared fusion for binary wildfire segmentation using the FLAME 3 dataset. We compare RGB-only and infrared-only baselines with three representative fusion strategies—early fusion, middle fusion, and late fusion—across different segmentation architectures, including U-Net, DeepLabV3, and SegFormer. The objective is to analyze the contribution of each modality, evaluate the impact of fusion timing, and examine how different network architectures exploit multimodal information for UAV-based wildfire segmentation.

The main contributions of this comparative study are summarized as follows:

• A systematic comparison between RGB-only and infrared-only modalities for UAV wildfire segmentation using FLAME 3.

• An evaluation of early, middle, and late fusion strategies under a common experimental protocol.

• A comparative analysis of convolutional and Transformer-based segmentation architectures, namely U-Net, DeepLabV3, and SegFormer.

• An investigation of the relationship between fusion strategy, modality selection, and segmentation performance in radiometric UAV wildfire imagery.

The remainder of this paper is organized as follows. Section II reviews related work on UAV wildfire segmentation and multimodal fusion methods. Section III describes the dataset, fusion strategies, network architectures, and experimental setup. Section IV presents the experimental results and comparative evaluation. Section V concludes the paper and outlines future research directions.

## II. RELATED WORK

## A. UAV-Based Wildfire Segmentation

The increasing availability of UAV platforms has significantly improved the acquisition of high-resolution imagery for wildfire monitoring and emergency response. Compared with satellite-based systems, UAVs provide finer spatial resolution and greater operational flexibility, enabling detailed observation of fire fronts, smoke plumes, and burned areas in near real time [3], [12].

The development of publicly available datasets has played a fundamental role in advancing deep learning methods for wildfire analysis. Among these datasets, the FLAME benchmark introduced aerial RGB and thermal imagery with pixel-level annotations, becoming one of the most widely used datasets for wildfire detection and segmentation research [13]. More recently, FLAME 3 extended this benchmark by providing synchronized RGB and radiometric thermal imagery, allowing the investigation of multimodal learning approaches based on calibrated thermal measurements [11]. Additional datasets, such as FireMan-UAV-RGBT and the Boreal Forest Fire dataset, have further contributed to the study of multimodal wildfire perception in UAV environments [14], [15].

Several studies have demonstrated the effectiveness of deep learning for wildfire segmentation using aerial imagery. Comprehensive reviews indicate that segmentation models consistently outperform traditional computer vision approaches by learning robust visual representations directly from image data [1], [16]. However, accurately segmenting wildfire regions remains challenging due to small flame areas, smoke occlusions, illumination changes, and highly heterogeneous backgrounds.

## B. Deep Learning Architectures for Wildfire Segmentation

Encoder–decoder convolutional architectures remain among the most widely adopted approaches for wildfire segmentation. U-Net has become a standard baseline because of its relatively low computational cost and its ability to preserve spatial information through skip connections. Previous studies demonstrated competitive segmentation performance using U-Net across different wildfire datasets and imaging conditions [4].

DeepLabV3+ introduced atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP), enabling the extraction of multi-scale contextual information while preserving spatial resolution. This architecture has consistently achieved superior segmentation accuracy compared with conventional encoder– decoder networks in aerial wildfire imagery [4]. More recent approaches have explored transformer-based models capable of capturing long-range dependencies through self-attention mechanisms. [5], [6] demonstrated that vision transformers such as TransUNet can outperform conventional CNN architectures in wildfire segmentation tasks by jointly modeling local details and global contextual information.

Recent hybrid CNN–Transformer architectures continue to improve segmentation quality in challenging wildfire scenarios. For example, FireFormer combines convolutional feature extraction with transformer-based contextual modeling and achieved competitive performance on the FLAME dataset [17]. These developments suggest that different architectures may exploit multimodal information in distinct ways, motivating comparative evaluations across CNN-based and transformerbased segmentation models.

## C. RGB–Infrared Fusion for Wildfire Analysis

While most wildfire segmentation systems rely on RGB imagery, thermal infrared sensing provides complementary information related to heat emission and active combustion. RGB images offer rich texture, shape, and contextual cues, whereas infrared imagery can highlight active fire regions even under adverse illumination conditions or partial smoke occlusion [7].

Some recent studies have investigated multimodal wildfire analysis using RGB and thermal information. [4] evaluated different image modalities for wildfire segmentation and reported that infrared and fused imagery can improve segmentation performance compared with visible imagery alone. [8] proposed an adaptive RGB-T modality learning framework that jointly exploits modality-specific and shared representations, achieving substantial improvements in segmentation accuracy under day and night conditions. Similarly, [9] introduced a multimodal segmentation framework based on multilevel fusion mechanisms and demonstrated that combining RGB and thermal features improves both IoU and F1-score compared with unimodal baselines.

Despite these promising results, the literature still provides limited understanding of how different fusion strategies influence wildfire segmentation performance. Existing studies typically focus on a single architecture or a specific fusion mechanism, making it difficult to isolate the effects of modality selection and fusion timing. Furthermore, few works systematically compare RGB-only, infrared-only, and multiple fusion strategies under a unified experimental protocol. The present work addresses this gap through a comprehensive evaluation of early, middle, and late fusion approaches across U-Net, DeepLabV3, and SegFormer architectures using the FLAME 3 dataset.

## III. METHODOLOGY

This section describes the dataset, ground-truth generation procedure, multimodal fusion strategies, evaluated segmentation models, and experimental setup adopted in this study.

## A. Dataset

The experiments are conducted using the FLAME 3 dataset [11], a UAV-based wildfire benchmark that provides synchronized RGB imagery, thermal imagery, and radiometric temperature measurements. The dataset contains samples acquired under different wildfire conditions and was specifically designed to support multimodal wildfire analysis using visible and thermal sensing modalities.

For each sample, three data sources are available:

• RGB images captured by the visible-spectrum camera;

• Thermal infrared images stored as colorized thermal JPEG files;

• Radiometric temperature maps provided as Celsius TIFF images.

Only samples containing valid RGB, thermal JPEG, and radiometric TIFF files were included in the experiments. All images were resized to $5 1 2 \times 5 1 2$ pixels before training and evaluation. RGB images were normalized using ImageNet statistics, while thermal images were independently normalized to preserve their temperature-related information.

Dataset partitioning followed a random split strategy with 70% of the samples used for training, 15% for validation, and 15% for testing. The same partition was maintained across all experiments to ensure a fair comparison between architectures and fusion strategies.

## B. Mask Generation

Unlike conventional segmentation datasets that provide manually annotated masks, FLAME 3 includes radiometric temperature measurements that can be used to derive activefire regions.

Binary fire masks were generated directly from the radiometric Celsius TIFF images through temperature thresholding. Let $T ( x , y )$ denote the temperature value at pixel $( x , y )$ . A pixel was labeled as fire when

$$
M ( x , y ) = \left\{ { \begin{array} { l l } { 1 , } & { T ( x , y ) \geq 8 0 ^ { \circ } C } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{1}
$$

where M represents the resulting binary segmentation mask. The threshold of $8 0 ^ { \circ } \mathrm { C }$ was adopted based on the temperature-based fire/no-fire screening criterion reported for the FLAME 3 radiometric thermal data [11].

This procedure produces objective fire-region annotations directly from measured thermal information and avoids the variability associated with manual labeling. The generated masks were used as ground truth for all segmentation experiments.

## C. Fusion Strategies

To investigate the contribution of visible and thermal information, five input configurations are evaluated.

• RGB: only RGB imagery is provided to the network.

• IR: only thermal imagery is used as input.

• Early Fusion: RGB and thermal images are concatenated at the input level, producing a six-channel tensor composed of three RGB channels and three thermal channels.

• Middle Fusion: RGB and thermal modalities are processed through independent feature extractors. Intermediate feature maps are fused before the decoder stage, allowing modality-specific representations to be learned before integration.

• Late Fusion: RGB and thermal data are processed by independent segmentation networks. The final segmentation logits generated by each branch are fused through a learnable fusion layer to obtain the final prediction.

Figure 1 illustrates the evaluated fusion strategies. The RGB and IR configurations serve as unimodal baselines, while the early, middle, and late fusion approaches enable the analysis of how fusion timing influences segmentation performance.

![](images/3de02a28a2c4faf6dd48e41a1388fa9243186d2d9afe52f51c0fdc11f177d49f.jpg)  
Fig. 1: Overview of the proposed RGB–IR wildfire segmentation pipeline. RGB and thermal infrared UAV imagery are combined using different fusion strategies and evaluated using U-Net, DeepLabV3+, and SegFormer architectures. The generated predictions are assessed using standard semantic segmentation metrics.

## D. Segmentation Models

Three semantic segmentation architectures were evaluated in this work: U-Net, DeepLabV3, and SegFormer. These models were selected due to their complementary characteristics regarding spatial-detail preservation, contextual aggregation, and global feature representation.

1) U-Net: U-Net [18] is an encoder–decoder convolutional architecture that uses skip connections to preserve spatial information throughout the reconstruction process. Due to its strong localization capability, U-Net has been widely adopted in segmentation tasks involving small and sparse targets.

2) DeepLabV3: DeepLabV3 [19] incorporates atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale contextual information. In this work, the MobileNet backbone version was adopted due to its favorable balance between computational efficiency and segmentation performance.

3) SegFormer: SegFormer [20] is a transformer-based segmentation architecture that combines hierarchical self-attention representations with a lightweight decoder. The model is capable of capturing long-range dependencies while preserving multi-scale feature representations.

By comparing these architectures, we analyze how convolutional and transformer-based models exploit visible, thermal, and fused multimodal information for wildfire segmentation.

## E. Training and Evaluation Protocol

All experiments were performed using the same training protocol to ensure a fair comparison across architectures and fusion strategies.

The models were trained using binary cross-entropy with logits loss (BCEWithLogitsLoss) and optimized with the AdamW optimizer. The learning rate was fixed at $1 0 ^ { - 4 }$ , the batch size was set to 2, and training was performed for 30 epochs. Model selection was based on the highest validation Intersection over Union (IoU).

Performance was evaluated using four standard segmentation metrics:

• Precision;

• Recall;

• F1-score;

• Intersection over Union (IoU).

During evaluation, segmentation probabilities were converted into binary predictions using a threshold of 0.5. All reported results correspond to the performance obtained on the held-out test set.

## IV. EXPERIMENTAL RESULTS

This section presents the experimental evaluation of RGB– IR fusion strategies for wildfire segmentation on the FLAME 3 dataset. The goal of these experiments is to analyze how visible and infrared modalities contribute to fire segmentation performance, as well as how different fusion strategies affect the behavior of multiple segmentation architectures.

We compare single-modality baselines using RGB-only and IR-only inputs with three multimodal fusion strategies: early fusion, middle fusion, and late fusion. This comparison allows us to assess whether combining RGB and infrared information improves segmentation performance over individual modalities, and whether the fusion stage influences precision, recall, F1-score, and IoU.

## A. Training Setup

All models were trained for binary semantic segmentation using the common training protocol described in Section III-E. The same dataset partitions, preprocessing pipeline, and optimization settings were maintained across all architectures and fusion configurations to ensure a consistent comparison.

TABLE I: Wildfire segmentation performance on the FLAME 3 test set for RGB-only, IR-only, and RGB–IR fusion configurations across the evaluated architectures. Bold values indicate the best result for each metric.
<table><tr><td>Architecture</td><td>Modality</td><td>Prec.</td><td>Rec.</td><td>IoU</td><td>F1</td></tr><tr><td rowspan="5">U-Net</td><td>RGB</td><td>0.70</td><td>0.09</td><td>0.08</td><td>0.17</td></tr><tr><td>IR</td><td>0.74</td><td>0.82</td><td>0.62</td><td>0.75</td></tr><tr><td>RGB-IR Early</td><td>0.92</td><td>0.62</td><td>0.58</td><td>0.73</td></tr><tr><td>RGB–IR Middle</td><td>0.83</td><td>0.72</td><td>0.60</td><td>0.74</td></tr><tr><td>RGB-IR Late</td><td>0.78</td><td>0.78</td><td>0.61</td><td>0.74</td></tr><tr><td rowspan="5">DeepLabV3+</td><td>RGB</td><td>0.61</td><td>0.09</td><td>0.07</td><td>0.19</td></tr><tr><td>IR</td><td>0.83</td><td>0.64</td><td>0.55</td><td>0.69</td></tr><tr><td>RGB-IR Early</td><td>0.79</td><td>0.68</td><td>0.56</td><td>0.71</td></tr><tr><td>RGB-IR Middle</td><td>0.69</td><td>0.51</td><td>0.37</td><td>0.51</td></tr><tr><td>RGB-IR Late</td><td>0.87</td><td>0.64</td><td>0.58</td><td>0.74</td></tr><tr><td rowspan="5">SegFormer</td><td>RGB</td><td>0.59</td><td>0.19</td><td>0.14</td><td>0.24</td></tr><tr><td>IR</td><td>0.89</td><td>0.76</td><td>0.69</td><td>0.81</td></tr><tr><td>RGB-IR Early</td><td>0.85</td><td>0.75</td><td>0.65</td><td>0.76</td></tr><tr><td>RGB-IR Middle</td><td>0.86</td><td>0.84</td><td>0.74</td><td>0.84</td></tr><tr><td>RGB-IR Late</td><td>0.83</td><td>0.87</td><td>0.74</td><td>0.84</td></tr></table>

## B. Evaluation Protocol

All models were evaluated on the FLAME 3 test set using Precision, Recall, F1-score, and Intersection over Union (IoU), as described in Section III-E. The comparison focuses on the contribution of each modality and on whether RGB– IR fusion improves performance over the single-modality baselines across different architectures.

## C. Results

Table I reports the wildfire segmentation results obtained on the FLAME 3 dataset.

1) U-Net Results: U-Net shows a clear performance gap between RGB-only and IR-based configurations. The RGBonly model obtains low recall and IoU, indicating that visible imagery alone is insufficient for accurately detecting fire regions in this dataset. Although RGB achieves moderate precision, the low recall shows that most fire pixels are missed.

The IR-only configuration substantially improves performance, reaching an F1-score of 0.75 and an IoU of 0.62. This indicates that the infrared modality provides more discriminative information for wildfire segmentation than RGB alone. Among the RGB–IR fusion strategies, early, middle, and late fusion achieve similar F1-scores, ranging from 0.73 to 0.74. However, none of the fusion strategies clearly surpass the IRonly baseline.

Early fusion produces the highest precision among the U-Net configurations, but this comes at the cost of lower recall. In contrast, late fusion provides a more balanced precisionrecall trade-off, with recall close to the IR-only setup. Overall, these results suggest that U-Net benefits strongly from infrared information, but RGB does not provide a consistent additional gain when fused with IR.

2) DeepLabV3+ Results: DeepLabV3+ also performs poorly with RGB-only input, achieving low recall and IoU. Similar to U-Net, this confirms that RGB information alone is not sufficient for reliable wildfire segmentation on FLAME 3.

The IR-only configuration provides a strong improvement, with an F1-score of 0.69 and an IoU of 0.55. RGB–IR early fusion slightly improves F1-score and IoU compared with IRonly, suggesting that combining RGB and IR at the input level can provide complementary information for this architecture. However, middle fusion performs substantially worse, with an F1-score of 0.51 and an IoU of 0.37. This indicates that the intermediate fusion design is less effective for DeepLabV3+ in this setting.

The best DeepLabV3+ result is obtained with late fusion, which reaches an F1-score of 0.74 and an IoU of 0.58. This suggests that processing RGB and IR streams more independently before fusion is more suitable for DeepLabV3+ than combining intermediate features too early. Overall, DeepLabV3+ benefits from RGB–IR fusion, but its performance depends strongly on the fusion stage.

3) SegFormer Results: SegFormer achieves the strongest overall results among the evaluated architectures. As observed for U-Net and DeepLabV3+, the RGB-only configuration performs poorly compared with IR-based configurations. However, SegFormer obtains the best RGB-only result among the three models, with an F1-score of 0.24 and an IoU of 0.14.

The IR-only configuration provides a substantial improvement, reaching an F1-score of 0.81 and an IoU of 0.69. This confirms the strong relevance of infrared information for wildfire segmentation. RGB–IR early fusion does not improve over IR-only, suggesting that simple input-level concatenation may introduce redundant or less informative visible features.

The best SegFormer results are obtained with middle and late fusion. Middle fusion achieves the highest IoU and F1- score, while late fusion obtains the highest recall. This indicates that SegFormer is better able to exploit complementary RGB and IR information when modalities are fused after modality-specific feature extraction. The strong recall obtained by late fusion suggests that this strategy is particularly effective at detecting fire pixels, while middle fusion provides a slightly better overall balance between precision and recall.

4) Cross-Model Analysis: The results reveal three main findings. First, RGB-only segmentation performs poorly across all architectures. Although precision remains moderate, recall is consistently low, indicating that models trained only with visible imagery miss most fire pixels. This behavior suggests that RGB information alone is not sufficiently reliable for active wildfire segmentation in the evaluated setting.

Second, infrared information is the dominant modality for wildfire segmentation on FLAME 3. IR-only models substantially outperform RGB-only models across all architectures, with large gains in recall, F1-score, and IoU. This confirms that the infrared modality captures fire-related information that is not easily detected from visible imagery alone.

Third, the effectiveness of RGB–IR fusion depends on the architecture and fusion strategy. For U-Net, fusion does not clearly improve over the IR-only baseline, suggesting that the RGB modality provides limited additional benefit for this encoder-decoder architecture. For DeepLabV3+, late fusion provides the best result, while middle fusion performs poorly. For SegFormer, middle and late fusion achieve the strongest overall performance, indicating that transformerbased architectures may benefit more from modality-specific feature extraction before fusion.

Overall, SegFormer with RGB–IR middle fusion provides the best trade-off between precision, recall, F1-score, and IoU. However, the strong performance of IR-only models across all architectures indicates that infrared imagery is the key modality for wildfire segmentation, while RGB information should be incorporated carefully through an appropriate fusion strategy.

## V. CONCLUSION

This work investigated the use of RGB-infrared fusion strategies for UAV-based wildfire segmentation using the FLAME 3 dataset. A systematic comparison has been conducted between unimodal baselines and multimodal fusion approaches across three segmentation architectures: U-Net, DeepLabV3+, and SegFormer.

The experimental results demonstrated that thermal infrared imagery provides substantially more discriminative information for wildfire segmentation than RGB imagery alone. Across all evaluated architectures, RGB-only configurations produced significantly lower segmentation performance, while infrared-only inputs consistently achieved strong results. These findings highlight the importance of thermal information for active-fire delineation in UAV imagery.

The results also revealed that the effectiveness of multimodal fusion depends on both the selected fusion strategy and the segmentation architecture. Early fusion did not consistently improve performance over infrared-only baselines, while SegFormer with middle fusion achieved the best overall performance. These findings suggest that transformer-based architectures can better exploit RGB–IR information when modality-specific features are learned before fusion.

Overall, this work provides a systematic analysis of RGB– IR fusion for wildfire segmentation under a unified experimental protocol and establishes strong baseline results for FLAME 3. A limitation of this study is that the evaluation is restricted to the FLAME 3 dataset, and the generalization of the observed trends should be further assessed on additional UAV wildfire datasets acquired under different environmental and sensor conditions.

Future work may investigate additional fusion mechanisms, uncertainty-aware segmentation, radiometric temperature integration, and lightweight multimodal architectures suitable for real-time UAV deployment.

## REFERENCES

[1] R. Ghali and M. A. Akhloufi, “Deep learning approaches for wildland fires remote sensing: Classification, detection, and segmentation,” Remote Sensing, vol. 15, no. 7, p. 1821, 2023.

[2] X. Chen, B. Hopkins, H. Wang, L. O’Neill, F. Afghah, A. Razi, P. Fule, J. Coen, E. Rowell, and A. Watts, “Wildland fire detection and´ monitoring using a drone-collected rgb/ir image dataset,” IEEE Access, vol. 10, pp. 121 301–121 317, 2022.

[3] S. P. H. Boroujeni, A. Razi, S. Khoshdel, F. Afghah, J. L. Coen, L. O’Neill, P. Fule, A. Watts, N.-M. T. Kokolakis, and K. G. Vamvoudakis, “A comprehensive survey of research towards ai-enabled unmanned aerial systems in pre-, active-, and post-wildfire management,” Information Fusion, vol. 108, p. 102369, 2024.

[4] H. Harkat, J. M. P. Nascimento, A. Bernardino, and H. F. Thariq Ahmed, “Assessing the impact of the loss function and encoder architecture for fire aerial images segmentation using deeplabv3+,” Remote Sensing, vol. 14, no. 9, 2022.

[5] R. Ghali, M. A. Akhloufi, and W. S. Mseddi, “Deep learning and transformer approaches for uav-based wildfire detection and segmentation,” Sensors, vol. 22, no. 5, 2022.

[6] R. Ghali, M. A. Akhloufi, M. Jmal, W. S. Mseddi, and R. Attia, “Wildfire segmentation using deep vision transformers,” Remote Sensing, vol. 13, no. 17, p. 3527, 2021.

[7] R. Ghali and M. A. Akhloufi, “Deep learning approaches for wildland fires remote sensing: Classification, detection, and segmentation,” Remote Sensing, vol. 15, no. 7, 2023.

[8] X. Rui, Z. Li, X. Zhang, Z. Li, and W. Song, “A rgb-thermal based adaptive modality learning network for day–night wildfire identification,” International Journal of Applied Earth Observation and Geoinformation, vol. 125, p. 103554, 2023.

[9] T. Yue, H. Huang, Q. Wang, B. Song, and Y. Chen, “A multimodal deep learning framework for accurate wildfire segmentation using rgb and thermal imagery,” Applied Sciences, vol. 15, no. 18, 2025.

[10] B. Hopkins, L. O’Neill, M. Marinaccio, F. Afghah, E. Rowell, R. Parsons, and S. Flanary, “Flame 3 - radiometric thermal uav imagery for wildfire management,” 2024.

[11] B. Hopkins, L. ONeill, M. Marinaccio, E. Rowell, R. Parsons, S. Flanary, I. Nazim, C. Seielstad, and F. Afghah, “Flame 3 dataset: Unleashing the power of radiometric thermal uav imagery for wildfire management,” 2024. [Online]. Available: https://arxiv.org/abs/2412.02831

[12] P. Keerthinathan, N. Amarasingam, G. Hamilton, and F. Gonzalez, “Exploring unmanned aerial systems operations in wildfire management: data types, processing algorithms and navigation,” International Journal of Remote Sensing, vol. 44, no. 18, pp. 5628–5685, 2023.

[13] A. Shamsoshoara, F. Afghah, A. Razi, L. Zheng, P. Z. Fule, and´ E. Blasch, “Aerial imagery pile burn detection using deep learning: The flame dataset,” Computer Networks, vol. 193, p. 108001, 2021.

[14] S. Kularatne, C. A. Casado, J. Rajala, T. Hanninen, M. B. L¨ opez, and´ L. Nguyen, “Fireman-uav-rgbt: A novel uav-based rgb-thermal video dataset for the detection of wildfires in the finnish forests,” in 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA), 2024, pp. 1–8.

[15] J. Pesonen, A.-M. Raita-Hakola, J. Joutsalainen, T. Hakala, W. Akhtar, N. Koivumaki, L. Markelin, J. Suomalainen, R. Alves de Oliveira,¨ I. Pol¨ onen, and E. Honkavaara, “Boreal forest fire: Uav-collected wild-¨ fire detection and smoke segmentation dataset,” Scientific Data, vol. 12, no. 1, p. 1419, Aug 2025.

[16] A. Saleh, M. A. Zulkifley, H. H. Harun, F. Gaudreault, I. Davison, and M. Spraggon, “Forest fire surveillance systems: A review of deep learning methods,” Heliyon, vol. 10, no. 1, p. e23127, 2024.

[17] H. Tong, J. Yuan, J. Zhang, H. Wang, and T. Li, “Real-time wildfire monitoring using low-altitude remote sensing imagery,” Remote Sensing, vol. 16, no. 15, 2024.

[18] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.

[19] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoderdecoder with atrous separable convolution for semantic image segmentation,” pp. 833–851, 2018.

[20] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.