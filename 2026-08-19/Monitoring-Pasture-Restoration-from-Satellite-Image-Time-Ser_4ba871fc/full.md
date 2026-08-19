# Monitoring Pasture Restoration from Satellite Image Time Series: Caveats and Opportunities

Linnea Sartorius<sup>1⋆</sup>, Isak Randahl<sup>1⋆</sup>, Delia Fano Yela<sup>2,3</sup>, Georg Andersson<sup>2</sup>, Sadegh Jamali<sup>1</sup>, and Aleksis Pirinen<sup>2,3,4</sup>

<sup>1</sup> Lund University

<sup>2</sup> RISE Research Institutes of Sweden 3 Climate AI Nordics

Swedish Centre for Impacts of Climate Extremes (CLIMES)

Abstract. Monitoring nature restoration at scale is an important but dificult ecological problem. Deep learning methods to analyze satellite image time series (SITS) have been widely used for land surface monitoring. In semi-natural grasslands – the habitat type in focus in this work – restoration outcomes develop gradually, yet satellite observations are influenced by weather, acquisition conditions, and processing artefacts, making it dificult to distinguish genuine restoration signals from unrelated temporal variation. In this work, we examine – to the best of our knowledge, for the first time – whether restoration status can be detected directly from satellite image time series by formulating pasture restoration as a binary deep learning classification problem. We evaluate two common SITS deep learning architectures on diferent Sentinel-2 image combinations, across 1,397 restored Swedish pastures and find that explicitly modeling intra-year variability and per-pasture normalization increases separability, reaching 0.88 accuracy for the best model. We further investigate our results and perform a targeted bias analysis finding that reliable deployment requires temporally balanced labels and evaluation protocols that explicitly test for year-related confounding. We therefore frame our contribution not as a solved restoration-monitoring system, but as a realistic case study of what works, what fails, and what future studies should control for. Code and models are available at https://github.com/aleksispi/ml-nature-resto.

Keywords: Nature restoration · Sustainable agriculture · Satellite image time series · Sentinel-2 · Deep learning

## 1 Introduction

Nature restoration has become a policy priority across Europe, including the recovery of degraded grassland ecosystems that support biodiversity, pollination, and resilient agricultural landscapes [2, 11]. In Sweden, semi-natural pastures are ecologically valuable but have declined substantially as small-scale grazing systems have been abandoned [7, 9]. Monitoring restoration progress in these landscapes is dificult in practice: field inspections are informative but costly, infrequent, and dificult to scale to national programs. Satellite image time series (SITS) ofer an attractive complementary signal as they provide repeated observations over large areas [10, 12]. For example, SITS have recently been used to monitor grazing activity in semi-natural pastures on a season-by-season basis [14]. However, nature restoration poses a distinct problem: rather than detecting management activity within a single growing season, the objective is to characterize ecological recovery trajectories that unfold over several years.

Unlike abrupt land-cover change, restoration is gradual, heterogeneous, and only partially observable from reflectance trajectories. The ecological process unfolds over multiple years, while satellite observations are shaped by seasonality, cloud cover, geography, and sensor-processing changes [1,3]. In other words, the same modeling machinery that can detect restoration-related change can also exploit time-dependent artifacts that happen to correlate with restoration labels. This issue is particularly acute when restoration labels are strongly correlated with calendar year, for example as in this work, when early years predominantly correspond to non-restored states and later years to restored states.

In this paper, we examine these issues in the specific context of pasturelevel restoration monitoring using Sentinel-2 summer time series over Sweden.

We formulate whether a pasture has been restored or not – to the best of our knowledge, for the first time – as a binary classification deep learning task, and evaluate two common architectures identified by [10] on diferent input representations. More precisely, we compare 2D CNNs operating on yearly composites and monthly composite stacks against CNN–LSTM sequence models, and we evaluate a standard global z-score normalization against a more adaptive per-pasture normalization. We further examine the influence of input features, the robustness of key modeling assumptions, and the experimental controls required to support credible claims of restoration detection.

Our results show that detecting nature restoration is feasible, and that both temporal information and normalization strategy influence classification performance. However, our bias analyses show that good restoration accuracy alone is insuficient evidence that the model has learned restoration signals. The dataset contains strong year-related signatures, and models can classify calendar year surprisingly well even when the restoration target is masked or weakened. We therefore also propose a

![](images/6d9452f40a3a08254b9960abb6b46982e0e6d5a68e512892d0b28176a6a386d5.jpg)  
Fig. 1: Spatial distribution of training, validation, and test polygons across Sweden.

reduced-bias data representation designed to suppress year-specific cues.

![](images/076d12c8439ad468f4e86c05f931d5c2528bb4d56af180b3ce57e1062fa50f4e.jpg)

![](images/3ce31258a2df88ca3d3806ac0271520b8a924b0c4e5f58f3ccf09b9d05ecf052.jpg)

![](images/ca36749e3149169e43c73c225de2f75bb61720048dcb33e114b541d5c43c0b46.jpg)  
Fig. 2: Histograms of the spread of start of restoration (left) and end of restoration (middle) years for the 1,397 field polygons, and the restoration duration extents (right).

In summary, we (i) tackle pasture-level nature restoration monitoring using SITS, and compare composite-based CNNs and sequence-based CNN–LSTM models under several input representations; (ii) show that leveraging intra-year variability and per-pasture normalization improves performance, while revealing substantial temporal confounding in the data; (iii) distill practical lessons for computer vision for ecology: strong benchmark performance is possible on real restoration data, but reliable deployment requires temporally balanced data and bias-oriented evaluation.

## 2 Dataset Overview

In this work we leverage pasture polygons and restoration metadata provided by the Swedish Board of Agriculture together with Sentinel-2 L2A imagery acquired between 2018 and 2025. The data contains pastures that received restoration support, including polygon geometry, restoration start year, and the year where restoration was approved. We define the last year as the summer season one year after approval to ensure a complete restored season is captured. After filtering non-polygon artifacts, splitting multi-polygons, and removing very small areas $\mathrm { ( \leq 4 0 0 m ^ { 2 } }$ , i.e. 2 × 2 pixels), the final dataset contains 1,397 pasture polygons. See Fig. 1 for the spatial distribution across Sweden.

For each polygon we use Sentinel-2 summer observations, June to August. Bands at 20 m and 60 m resolution are bilinearly upsampled to 10 m and stored in a common tensor representation padded to 100 × 100 pixels. The upsampling is performed solely to place all spectral bands on a common spatial grid for subsequent CNN processing; it does not increase the efective spatial resolution or introduce new spatial information. Since the interpolated bands retain their original information content, the network can only exploit spatial detail present at the native Sentinel-2 resolutions. Pixels outside the target polygon are masked to suppress context from nearby fields and farms. This is important because multiple pastures can occur within the same satellite tile and, without masking, nearby land-use dynamics can leak into the prediction task.

Cloud filtering is performed using a learned cloud detector [13]. Importantly, the preprocessing accounts for the 2022 Sentinel-2 processing-baseline ofset: filimage availability, so filters clouds without the ofset in order to reduce additional year-related bias. Images with missing values, negative reflectance anomalies, or clear channel-wise outlier statistics are removed. After filtering, approximately 39% of the original images remain, corresponding to roughly between 11 and 18 valid summer images per polygon-year depending on the year.

The classification target is derived from the restoration timeline (see Fig. 2). For the main binary setup, the first available year for a pasture is labeled nonrestored and the last year is labeled restored. This gives two samples per pas-

![](images/565453f3cd4c152053bf1ec6ddae380abd95dcf1669e7fa4dfb58aa316ce09ab.jpg)  
Fig. 3: Restoration statuses across years.

ture and therefore a balanced binary dataset by construction. The middle years are retained for bias analysis and for sequence-model experiments, but their labels are intrinsically more ambiguous because the degree of restoration is not directly observed for each intermediate year (see also Fig. 3). A critical property of the dataset is that class label and calendar year are strongly entangled: early years are dominated by non-restored samples and late years by restored samples, introducing an unavoidable bias. To avoid spatial leakage, train/validation/test splits are made by clustering nearby polygons with a 1.5 km bufer and assigning entire clusters to the same split, while preserving key geographical variability such as growth zones and precipitation regimes. The resulting split is roughly $7 8 . 7 \% / \ 1 2 . 4 \% \ / \ 8 . 9 \%$ in polygons for training, validation, and test.

## 3 Methods

There are four principal families of deep learning architectures for SITS analysis identified by [10]: recurrent neural networks (RNNs) such as Long Short-Term Memory (LSTM) [5], convolutional neural networks (CNNs), hybrid CNN-RNN models, and transformer models. Given our task and data constraints, here we explore CNNs and hybrid CNN-LSTM architectures.

2D CNNs on seasonal composites. The input to the 2D CNN is a yearly seasonal composite formed by taking the pixel-wise median over all valid summer images for one pasture-year. We compare this single-composite representation to two alternatives: (i) a composite augmented with vegetation indices (NDVI, EVI, NDMI, and MSAVI), and (ii) a composite stack in which June, July, and August are each summarized separately and stacked into a 36-channel input. The composite stack is intended to preserve intra-year variation without requiring full recurrent modeling. Two CNN backbones are evaluated: (i) a ResNet50 [4], and (ii) a lighter custom residual CNN tailored to the dataset size. The custom model consists of an initial convolutional layer, two residual blocks at 32 channels, downsampling to 64 channels, two further residual blocks, global average pooling, and a small multilayer perceptron classifier. Models are trained with Adam [8], binary cross-entropy with logits, random flips and 90<sup>◦</sup> rotations, and early stopping using validation performance.

CNN–LSTM sequence models. To model sequential information directly, we extract per-image spatial descriptors with the custom CNN and map the resulting feature vectors through a one-layer LSTM [6]. An FC-layer operating on the final hidden state then yields a decision. Sequence inputs are built in two ways: (i) all valid images from the first year vs. all valid images from the last year, and (ii) multi-year sequences where the early half of the restoration trajectory is treated as negative and the late half as positive. The latter leverages a larger portion of the restoration trajectory, but also introduces additional label ambiguity (cf. Fig. 3).

## 3.1 Normalization Schemes and Evaluation Analyses

We compare two normalization schemes: (i) global normalization uses trainingset means and standard deviations for each band; (ii) per-pasture normalization instead standardizes each pasture using statistics aggregated over its own restoration period. The latter is motivated by the possibility that restoration manifests more consistently as a within-pasture shift than as an absolute spectral state shared across diferent landscapes.

Because restoration labels are strongly entangled with calendar year, standard accuracy metrics alone are insuficient to assess what the models learn. We therefore complement restoration classification with two complementary analyses. First, a restoration trajectory analysis evaluates how often the model predicts “restored” at diferent years of a restoration trajectory, which provides insights on how predictions evolve over time (see Fig. 4 in Sec. 4). Second, temporal bias is assessed in Sec. 4.1 by training models to classify calendar year itself, both on regular masked inputs and on reverse-masked inputs where the pasture is removed and only the surrounding scene remains.

## 4 Experimental Results

Table 1 shows the key quantitative findings. The strongest non-sequential (Custom CNN) representation is the monthly composite stack. With global normalization, the composite stack reaches 0.75 accuracy, improving over the single seasonal composite (0.73). With per-pasture normalization, performance increases substantially to 0.86 accuracy, with balanced specificity (0.88) and recall (0.84). This strongly suggests that intra-year variability is useful and that restoration is easier to detect as a within-pasture change than as a universal end-state shared across sites. We also see (on validation data; bottom two rows of Table 1) the custom CNN architecture outperforms ResNet50.

Table 1: Main results (on the test set unless otherwise specified). Monthly composite stacks outperform a single seasonal composite, and per-pasture normalization yields a substantial performance gain. Sequential CNN–LSTM models provide a further slight improvement, with the multi-year variant achieving the best overall accuracy (0.88) and recall (0.92).
<table><tr><td>Model / input</td><td>Accuracy Specificity Recall</td><td></td><td></td></tr><tr><td>Custom CNN, single composite</td><td>0.73</td><td>0.70</td><td>0.75</td></tr><tr><td>Custom CNN, monthly composites stack</td><td>0.75</td><td>0.67</td><td>0.83</td></tr><tr><td>Custom CNN, monthly composites, added veg indices</td><td>0.72</td><td>0.62</td><td>0.81</td></tr><tr><td>Custom CNN, monthly composites, per-pasture norm</td><td>0.86</td><td>0.88</td><td>0.84</td></tr><tr><td>CNN-LSTM, first/last year, per-pasture norm</td><td>0.85</td><td>0.84</td><td>0.88</td></tr><tr><td>CNN-LSTM, multi-year, per-pasture norm</td><td>0.88</td><td>0.85</td><td>0.92</td></tr><tr><td>Custom CNN, 3-image random stack + per-pasture norm</td><td>0.76</td><td>0.73</td><td>0.79</td></tr><tr><td>Custom CNN, monthly composites, per-pasture norm (val)</td><td>0.88</td><td>0.89</td><td>0.88</td></tr><tr><td>ResNet50, monthly composites, per-pasture norm (val)</td><td>0.80</td><td>0.78</td><td>0.81</td></tr></table>

The sequential (CNN–LSTM) models are competitive and slightly outperform the best 2D CNN. The best sequence model uses multi-year input and per-pasture normalization, with 0.88 accuracy, 0.85 specificity, and 0.92 recall. This indicates that sequence models can exploit temporal information efectively, especially for identifying restored samples. The improvement over the simpler first/last-year variant suggests that modeling longer temporal context provides additional signal beyond a single early–late comparison. However, the modest gain over the composite-stack CNN is small, suggesting that monthly summaries capture a large portion of the useful signal.

Restoration trajectory behavior. The restoration trajectory analysis (Fig. 4), applied to the best-performing 2D CNN (monthly composite stack with perpasture normalization), shows that the model assigns progressively higher restoration probabilities to later years of the trajectory. Predicted restoration probabilities increase monotonically along the restoration trajectory, from low values in the early years to high values in the final years. This indicates that the model captures a temporal ordering signal consistent with restoration progression. However, because calendar year and restoration status are strongly correlated in the dataset (cf. Fig. 3), this monotonic trend may arise either from genuine restoration signals or from the model implicitly predicting the acquisition year. The trajectory alone therefore does not distinguish between these explanations.

## 4.1 Investigating temporal confounding

Table 2 shows that calendar year can be predicted with high accuracy from the data, confirming the presence of strong temporal confounding. Using reversemasked inputs that exclude the pasture entirely, the model achieves an F1 score of 0.72, which indicates that substantial year-specific signal exists in the surrounding context. Even when restricting inputs to the pasture region, year prediction remains well above chance (F1-score 0.49 vs. 0.125), which demonstrates that temporal information is also encoded within the fields themselves. Perpasture normalization reduces year predictability in the reverse-masked setting (0.72→0.64) but increases it in the masked setting (0.49→0.56), which suggests a non-trivial interaction with within-pasture temporal structure. These results show that temporal bias operates at both global and local levels, and cannot be eliminated by masking alone.

![](images/b348dc68906be06b94ded132f25818799a61636e3eb07cf2e5063c4dca5b824e.jpg)

![](images/f622a19eb455d46bfbb4ac7adc7a1400fd376ad2073a467682794077f5116c0d.jpg)  
Fig. 4: Restoration trajectory analysis (left: val, right: test) on the pasture normalized 2D CNN model. The model predicts very few samples as restored in the early years and almost 90% of samples as restored in the final year.

Reducing temporal shortcuts and residual signal. To better understand the extent to which models rely on temporal shortcuts, we construct a reducedbias variant that lowers cloud-filter thresholds and replaces monthly composites with a randomly sampled three-image stack per epoch (one image each from June, July, August). This weakens consistent seasonal structure and reduces the predictability of calendar year $( \mathrm { F 1 } \approx 0 . 2 4 \ – 0 . 3 0 )$ . If restoration performance were entirely driven by year prediction, this reduction should lead to a collapse in accuracy. Instead, restoration classification remains reasonably strong, with a test accuracy of 0.76 (third-to-last row in Table 1), compared to 0.86 for the standard per-pasture normalized composite stack. This drop is meaningful but moderate, which indicates that temporal confounding contributes to performance, while the relatively high remaining accuracy suggests the presence of a non-trivial restoration-related signal that is being detected by the model.

Table 2: Year-prediction results on the validation set. All results are based on using a Custom 2D CNN, and except otherwise specified a single composite per year (the last two rows which use a random image from each of June, July, August), respectively.
<table><tr><td>Model / input F1-macro</td></tr><tr><td>Reverse masking 0.72 Reverse masking + per-pasture norm 0.64</td></tr><tr><td>Regular masking 0.49 Regular masking + per-pasture norm 0.56</td></tr><tr><td>Reverse masking, 3 random image stack (run 1) 0.24</td></tr></table>

## 5 Conclusions

In this work we have investigated whether pasture-level nature restoration can be monitored from Sentinel-2 satellite image time series (SITS) using deep learning. Our findings provide a cautiously positive answer. Both CNN and CNN–LSTM models achieve strong discrimination between restored and non-restored pasture states, which indicates that SITS contain information relevant to restoration monitoring. However, our analyses reveal substantial temporal confounding in the dataset, which prevents the observed performance from being attributed exclusively to restoration-related signals. To address this issue, we introduced a reduced-bias evaluation setting designed to weaken year-specific shortcuts. Although this reduced classification accuracy from 0.86 to 0.76, the remaining performance suggests that meaningful restoration-related information is present and leveraged by the model.

Our analysis shows that within-season variability is informative, that perpasture normalization can be particularly efective in heterogeneous grassland environments, and that targeted temporal bias analyses are essential. Most importantly, we demonstrate that strong benchmark performance alone does not provide suficient evidence of robust ecological monitoring.

Future datasets integrated with this workflow should contain temporally balanced examples where restored and non-restored states coexist within the same years, as well as genuinely negative trajectories that never become restored. Until such data become available, SITS-based deep learning methods for nature restoration monitoring should be viewed as promising but methodologically constrained: capable of capturing ecologically relevant signals, yet highly sensitive to temporal confounding unless study design and evaluation protocols are explicitly constructed to control for it. Code and models are available at https://github.com/aleksispi/ml-nature-resto.

## Acknowledgments

This work was funded by the Swedish National Space Agency (project number 2025-00013). We are grateful for the support and data provided by the Swedish Board of Agriculture, and Niklas Boke Olén in particular.

## References

1. ClearSKY Vision: Sentinel-2 scaling and harmonization. https://clearsky. vision/knowledge/sentinel2-scaling-harmonization (2026), accessed 2026

2. European Union: Regulation (eu) 2024/1991 of the european parliament and of the council (2024), oficial EU regulation

3. Forkel, M., Carvalhais, N., Verbesselt, J., Mahecha, M.D., Neigh, C.S., Reichstein, M.: Trend change detection in ndvi time series: Efects of inter-annual variability and methodology. Remote Sensing 5(5), 2113–2144 (2013)

4. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

5. Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural Computation 9(8), 1735–1780 (1997). https://doi.org/10.1162/neco.1997.9.8.1735

6. Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural computation 9(8), 1735–1780 (1997)

7. Ke, K., et al.: Grazing livestock increases both vegetation and seed bank diversity in remnant and restored grasslands. Journal of Vegetation Science (2020)

8. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

9. Lindborg, R., Eriksson, O.: Efects of restoration on plant species richness and composition in scandinavian semi-natural grasslands. Restoration Ecology (2004)

10. Miller, L., Pelletier, C., Webb, G.I.: Deep learning for satellite image time-series analysis: A review. IEEE Geoscience and Remote Sensing Magazine 12(3), 81–124 (2024)

11. Naturvårdsverket: Förslag till nationell restaureringsplan och författningsändringar till följd av eu-förordning om restaurering av natur (2024), swedish Environmental Protection Agency report

12. Phiri, D., Simwanda, M., Salekin, S., Nyirenda, V.R., Murayama, Y., Ranagalage, M.: Sentinel-2 data for land cover/use mapping: A review. Remote Sensing 12(14), 2291 (2020)

13. Pirinen, A., Abid, N., Paszkowsky, N.A., Timoudas, T.O., Scheirer, R., Ceccobello, C., Kovács, G., Persson, A.: Creating and leveraging a synthetic dataset of cloud optical thickness measures for cloud detection in msi. Remote Sensing 16(4), 694 (2024)

14. Pirinen, A., Yela, D.F., Chakraborty, S., Källman, E.: Grazing detection using deep learning and sentinel-2 time series data. arXiv preprint arXiv:2510.14493 (2025)