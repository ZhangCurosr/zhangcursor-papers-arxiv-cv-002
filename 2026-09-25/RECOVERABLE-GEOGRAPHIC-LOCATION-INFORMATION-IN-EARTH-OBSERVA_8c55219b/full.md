# RECOVERABLE GEOGRAPHIC LOCATION INFORMATION IN EARTH-OBSERVATIONEMBEDDINGS

Peiwen Zhang<sup>1</sup> Kristie Hu<sup>1</sup> Jovana Knezevic<sup>2</sup> Shunde Yin<sup>1</sup> Kyle Gao<sup>3</sup>

<sup>1</sup>University of Waterloo <sup>2</sup>University of Cambridge <sup>3</sup>Aalto University

## ABSTRACT

Earth-observation (EO) foundation models provide reusable embeddings, yet downstream task accuracy does not reveal whether these representations encode geographic information, which may be beneficial for location-aware applications but potentially detrimental when representations invariant to geographic location are desired. We therefore evaluate the geographic coordinate robustness of Tessera v1, Tessera v1.1, and AlphaEarth by testing whether coordinates can be predicted from the embedding representations using 284 quality-verified European solar farms from 2024. We assessed geographic information content information through the association between cosine and geodesic distances and through prediction of projected coordinates in EPSG:3035. Embeddings from all three EO foundation models contain recoverable geographic information. All prediction models significantly outperform training-range uniform random sampling baselines, with AlphaEarth exhibiting the strongest distance association and lowest mean geodesic error. Both Tessera variants also yielded higher geographic distance correlations than the Sentinel-2 controls. These findings motivate geographic information content as an additional criterion for auditing EO foundation models.

Index Terms— Representation learning, embeddings, foundation models, spatial representations, remote sensing

## 1. INTRODUCTION

Earth-observation (EO) foundation models provide reusable pixel-level embeddings for downstream analysis. Tessera summarizes annual Sentinel-1 and Sentinel-2 time series in 128 dimensions [1], whereas AlphaEarth integrates annual satellite images in a 64-dimensional field at 10 m resolution [2]. Their downstream accuracy, however, does not reveal how strongly the embedding representations encode geographic location information, which may be beneficial for location-aware tasks but undesirable when geography-agnostic representations are required. We therefore ask whether geographic locations can be predicted from Tessera v1, Tessera v1.1, and AlphaEarth embeddings within a single semantic class: solar farms.

Geography-aware self-supervised learning, SatCLIP, Tile2Vec, and GeoCLIP all use geographic coordinates or spatial proximity as an explicit training signal—as locationencoder inputs in SatCLIP and GeoCLIP [4], [6], as a geolocation pretext classification target in geography-aware SSL [3], and as the basis for positive-pair construction in Tile2Vec [5]. These studies do not, however, quantify how much geographic location information (lat, lon) remains recoverable from general-purpose EO embeddings that were produced without explicit coordinate inputs. Recent benchmarks emphasize standardized model comparison [7] and broaden the geographic coverage of evaluated datasets [8]. Spatial generalization also depends on the relationship between training and test data: performance can vary with their spatial separation [11] or with geographic distribution shifts [10]. Random crossvalidation can also overstate spatial transfer when nearby observations enter both training and test sets [12], [13], [14], [15]. To ensure a consistent comparison while accounting for spatial dependence, we compare all three embedding representations using the same solar farms sites and spatial folds. Because spatially related farm pairs are not independent [16], we resample at the level of spatial dependency groups rather than treating the 40,186 pairs as independent observations.

Our contribution is a controlled assessment of embedding geometry and coordinate sensitivity:

• Geographic distance association: We test whether cosine distance is associate with geodesic distance using embeddings of a controlled surface, namely photovoltaic surfaces of solar farms.

• Coordinate sensitivity: We evaluate embedding robustness to geographic coordinates using spatially held-out coordinate probes against appearance-based and random baselines.

## 2. DATA AND METHODOLOGY

Dataset and spatial design. We chose solar farms to isolate geographic effects within a single land-use class with relatively consistent materials and radiometric properties worldwide. We selected 284 solar farms across eight countries from the 2024 Q2 Global Renewables Watch release [17], retaining only farms that passed our own visual and coverage checks and had valid 2024 coverage in Tessera v1, Tessera v1.1, and AlphaEarth. The final selected solar farms contains 231 farms in Spain, 15 each in France and the United Kingdom, 8 in Germany, 6 in Ireland, 5 in the Netherlands, and 2 each in Portugal and Romania. Spain accounted for 81.3% of the sample, indicating an uneven geographic distribution within the dataset. For each farm, pixel-level embeddings were aggregated using component-wise medians to reduce sensitivity to outliers. Sentinel-2 center pixel and $3 \times 3$ patch representations served as appearance-based control (Fig. 1), providing a comparison with geographic information recoverable from local multispectral features.

![](images/44f947c0b3b78bb7ac7d5f08443dbcbceef4f0dc9d3fdc5c761606f3bdd64bed.jpg)  
Fig. 1. Sentinel-2 imagery and PCA-coloured embeddings for six solar farms; colours are comparable only within each rows.

Permutation significance and baseline comparison. Coordinate prediction performance was compared with a trainingrange uniform random baseline. For each permutation, we reassigned block-centroid locations while preserving withinblock coordinate offsets. $B = 9 9 9$ draws (minimum possible $p \ = \ 1 / 1 0 0 0 )$ , was used to test two null hypotheses: that the embeddings contain no recoverable geographic-coordinate information, and that the geography-embedding distance association is no stronger than under a random relabeling of spatial blocks. For each, the full nested procedure (Ridge fit with inner-fold parameter selection, or the Spearman correlation against permuted geographic distances) was rerun on every permutation b, giving a statistic $T _ { b }$ where larger values indicate stronger apparent geographic structure $( T = - E $ , the negative held-out geodesic error, for coordinate predicting; $T = | \rho |$ , the absolute Spearman coefficient, for the distance association). Both tests reduce to the same empirical p-value,

$$
p = \frac { 1 + \sum _ { b = 1 } ^ { B } \mathbf { 1 } ( T _ { b } \geq T _ { \mathrm { o b s } } ) } { B + 1 } ,
$$

which saturates at 0.001 once the observed statistic clears all B permutations, establishing that the embeddings contain geographic information beyond the marginal spatial layout of blocks and farms preserved under permutation.

Farm location prediction and predictor comparison. We evaluated location prediction using ordinary linear regression, ridge regression, and multilayer perceptrons (MLPs) with 2-, 4-, 6-, and 8- hidden layers. To reduce short-range spatial dependence between training and test samples, farms within 25 km were grouped into dependency components that could not be split across folds.

We used the same five-fold spatial cross-validation across all the prediction models and embedding representations: each fold served once as the test set, with the remaining folds used for training. For ridge regression and MLPs, we used regularization to penalize large model weights and limit overfitting. The penalty strength is denoted by λ for ridge regression and controlled by the weight-decay parameter for MLPs. For each model run, we selected this parameter using the training data (four-fold spatial validation), keeping farms from the same spatial block together.

In addition, we compared prediction errors against a uniform random baseline that independently guesses latitude and longitude within the ranges observed in the training farms. Prediction accuracy was measured by the geodesic distance between predicted and actual farm coordinates. We estimated 95% confidence intervals by resampling the spatial blocks and compared prediction for the same test farms across different models.

To test how prediction depends on nearby training examples, we varied the training exclusion distance from 25 to 100 km in 5 km increments. For each test fold and distance, we removed any training farm whose centroid lies within that distance of any test farm. Test farms and fold assignments remain fixed throughout the sweep. All models and representations used identical training and test sets at each distance. The 25 km setting reproduces the original training split.

Finally, we conducted leave-one-country-out evaluation using ordinary linear regression, ridge regression, and 8−layer MLPs to predict the farm coordinates when an entire country was excluded from training.

## 3. RESULTS

## 3.1. Embedding distance follows physical distance

All three EO foundation model representations showed positive distance-geography associations (Table 1). Spearman ρ was 0.386 [0.308, 0.492] for Tessera v1, 0.424 [0.344, 0.514] for Tessera v1.1, and 0.751 [0.664, 0.808] for AlphaEarth. The Sentinel-2 center-pixel and $3 \times 3$ patch controls yielded Spear man correlations of 0.294 [0.179, 0.443] and 0.288 [0.178, 0.396], respectively. Permutation tests for Spearman correlation and Ridge regression yielded $p = 0 . 0 0 1$ , providing strong evidence of existing location predictability, and an association between embedding cosine distance and geodesic distance under the specified permutation tests. The Tessera embeddings showed lower correlations than AlphaEarth and higher point estimates than the Sentinel-2 controls.

Table 1. Primary embedding distance-geographic distance correlation and coordinate-decoding results. Brackets give 95% spatial-block-bootstrap confidence intervals.
<table><tr><td>Input</td><td>Dim.</td><td>Spearman ρ [95% CI]</td><td>Mean error, km [95% CI]</td><td>Median error, km [95% CI]</td><td>Equal-country mean, km [95% CI]</td></tr><tr><td>AlphaEarth</td><td>64</td><td>0.751 [0.664, 0.808]</td><td>179.7 [153.9, 231.0]</td><td>144.5 [127.6, 188.2]</td><td>333.3 [201.3, 486.0]</td></tr><tr><td>Tessera v1.1</td><td>128</td><td>0.424 [0.344, 0.514]</td><td>235.9 [192.3, 331.3]</td><td>174.9 [150.3, 235.0]</td><td>613.9 [366.6, 923.7]</td></tr><tr><td>Tessera v1</td><td>128</td><td>0.386 [0.308, 0.492]</td><td>256.7 [207.0, 361.9]</td><td>178.0 [152.5, 250.3]</td><td>676.1 [432.1, 940.8]</td></tr><tr><td>Sentinel-2 center-pixel</td><td>13</td><td>0.294[0.179,0.443]</td><td>391.9[310.2, 518.9]</td><td>304.6 [236.5, 407.5]</td><td>815.2 [490.4, 1169.0]</td></tr><tr><td>Sentinel-2 3 × 3 patch</td><td>117</td><td>0.288 [0.178, 0.396]</td><td>380.8 [303.9, 502.3]</td><td>289.8 [245.1, 390.4]</td><td>807.4 [466.3, 1178.1]</td></tr></table>

![](images/3fa0229196cfc874a2e43da5d61f838155efb550cd66a94fe928e9ece478a196.jpg)

<table><tr><td rowspan=2 colspan=7>TESSERA v1                  TESSERA v1.1                   AlphaEarthPixel        3×3 patch</td></tr><tr><td rowspan=1 colspan=1>Pixel</td><td rowspan=1 colspan=1>3×3 patch</td><td rowspan=1 colspan=1>Pixel</td><td rowspan=1 colspan=1>3×3 patch</td></tr><tr><td rowspan=1 colspan=1>Ordinary linear</td><td rowspan=1 colspan=1>-91*</td><td rowspan=1 colspan=1>-196*</td><td rowspan=1 colspan=1>-86*</td><td rowspan=1 colspan=1>-191*</td><td rowspan=1 colspan=1>-211*</td><td rowspan=1 colspan=1>-316*</td></tr><tr><td rowspan=1 colspan=1>Ridge</td><td rowspan=1 colspan=1>-135*</td><td rowspan=1 colspan=1>-124*</td><td rowspan=1 colspan=1>-156*</td><td rowspan=1 colspan=1>-145*</td><td rowspan=1 colspan=1>-212*</td><td rowspan=1 colspan=1>-201*</td></tr><tr><td rowspan=1 colspan=1>MLP-2</td><td rowspan=1 colspan=1>-79*</td><td rowspan=1 colspan=1>-88*</td><td rowspan=1 colspan=1>-75*</td><td rowspan=1 colspan=1>-84*</td><td rowspan=1 colspan=1>-134*</td><td rowspan=1 colspan=1>-143*</td></tr><tr><td rowspan=1 colspan=1>MLP-4</td><td rowspan=1 colspan=1>-74*</td><td rowspan=1 colspan=1>-74*</td><td rowspan=1 colspan=1>-61*</td><td rowspan=1 colspan=1>-61*</td><td rowspan=1 colspan=1>-137*</td><td rowspan=1 colspan=1>-136*</td></tr><tr><td rowspan=1 colspan=1>MLP-6</td><td rowspan=1 colspan=1>-82*</td><td rowspan=1 colspan=1>-83*</td><td rowspan=1 colspan=1>-56*</td><td rowspan=1 colspan=1>-57*</td><td rowspan=1 colspan=1>-127*</td><td rowspan=1 colspan=1>-128*</td></tr><tr><td rowspan=1 colspan=1>MLP-8</td><td rowspan=1 colspan=1>-90*</td><td rowspan=1 colspan=1>-89*</td><td rowspan=1 colspan=1>-82*</td><td rowspan=1 colspan=1>-82*</td><td rowspan=1 colspan=1>-153*</td><td rowspan=1 colspan=1>–152*</td></tr></table>

∆ = embedding – S2, using the same predictor; negative favours the embedding  
Pixel: center pixel. \* Paired 95% CI excludes zero (unadjusted)

Fig. 2. Farm location prediction under spatial cross-validation. (a–c) Mean geodesic errors for three embedding products and six predictors. (d) Paired error differences relative to Sentinel-2 controls using the same predictor.

## 3.2. Farm location prediction from embeddings

Geographic location prediction across embeddings. AlphaEarth yielded lower mean location-prediction errors than either TESSERA representation for every predictor tested (168–193 km versus 225–308 km) (Fig. 2a-c). TESSERA v1.1 did not consistently improve upon v1: it performed better with ridge regression but had higher mean errors with ordinary linear regression and all MLPs. All three representations substantially outperformed the uniform random baseline (1,350.5 km), indicating that they retained geographic location information useful for predicting held-out farm locations. They also yielded lower mean prediction errors than both Sentinel-2 baselines, center pixel and 3 × 3 patch (Fig. 2d).

Effect of prediction model choice and MLP depth. For each embedding representation, increasing the complexity of prediction models did not consistently reduce error. Ridge yielded lower mean errors than ordinary linear regression for both TESSERA versions and a similar mean error for AlphaEarth(Fig. 2a-c). They also yielded lower mean prediction errors than both Sentinel-2 baselines, center pixel and 3×3 patch (Fig. 2a-c).

## 3.3. Location prediction becomes less accurate as spatial separation increases

We assessed sensitivity to spatial separation by increasing the training exclusion distance from 25 to 100 km in 5 km increments while keeping the test farms and folds fixed. We evaluated ordinary linear regression, ridge regression, and an 8−layer MLP using the same evaluation procedure and grouped inner-fold hyperparameter selection as in the preceding coordinate-prediction experiment.

Prediction errors generally increased with the exclusion distance, although the magnitude depended on the prediction model and representation. Ordinary linear regression showed larger increases for the TESSERA embeddings and Sentinel-2 3 × 3 features, whereas ridge regression and the 8-layer MLP showed more gradual increases. These results suggest that its relative advantage persists when nearby training farms are excluded, while absolute prediction accuracy remains sensitive to the evaluation design.

![](images/4e4ff26963922e8291384b64eba64bd1ef20ec01a6ee11fe1cbdd73a92740513.jpg)  
Fig. 3. Sensitivity of farm location prediction to spatial separation. Mean geodesic error with equal farm weighting as the training exclusion distance increases from 25 to 100 km, using ordinary linear regression, ridge regression, and an 8-layer MLP. Test farms and folds remain fixed across distances.

## 3.4. Location-prediction performance varies across heldout countries.

To test whether geographic information learned from the sampled regions transfers to countries excluded from predictor training, we perform leave-one-country-out evaluation as a geographic generalization test (Fig. 4). For each country, all farms from that country are held out for testing, while the prediction model is trained on the remaining countries using the same preprocessing and grouped inner-fold hyperparameter selection as the previous coordinate prediction experiment.

Across the three prediction models, AlphaEarth yields errors of 157–189 km for the United Kingdom and 217–294 km for Ireland, compared with 466–700 km and 435–643 km for the two TESSERA versions, respectively. For Spain, which contains 231 of the 284 farms, AlphaEarth errors range from 232 to 633 km, compared with 598–1,147 km for TESSERA and 705–1,177 km for Sentinel-2.

The relative performance of the predictors also depends on the held-out country. With AlphaEarth, MLP-8 yields a lower error for Spain than ordinary linear regression or ridge regression (232 km versus 633 and 372 km), but a higher error for Romania (1,515 km versus 884 and 1,058 km). Romania has the largest error within every representation–predictor combination, with values ranging from 884 to 2,224 km. AlphaEarth does not yield the lowest error in every comparison: for Portugal, MLP-8 errors are 220–221 km with TESSERA and 243 km with AlphaEarth. These results indicate variation in geographic transfer across the evaluated countries, with errors remaining on the order of hundreds of kilometres in many settings. Country-level comparisons should be interpreted cau tiously because the held-out samples are uneven, including only two farms each for Portugal and Romania.

![](images/744f317b62d82120ddecb48aff02a5d47ac7d43177a6cc35d9dd8a38cc17b787.jpg)  
Fig. 4. Leave-one-country-out farm location prediction. Mean geodesic errors for three predictors and five input representations, with each country excluded in turn from predictor training.

## 4. DISCUSSION AND CONCLUSION

Using a solar farm dataset restricted to one semantic class, we show that embeddings from all three EO foundation models contain recoverable geographic information. AlphaEarth shows the strongest geographic distance association and lowest location-prediction error across the tested prediction models, while Tessera v1.1 is comparable to v1. Prediction errors generally increase as nearby training farms are excluded, and performance varies across held-out countries.

The generalizability of these findings is limited by uneven geographic coverage, with 81.3% of farms in Spain. Greater spatial separation and country holdouts also reduce training data, complicating the interpretation of prediction errors. The ability to predict locations may partly reflect country-specific or dataset-specific patterns rather than fine-grained geographic information. We therefore do not claim that coordinates are explicitly encoded, and the desirability of geographic dependency ultimately depends on the downstream task, being useful for location-sensitive applications such as climate mapping but potentially undesirable for global similarity search. We recommend assessing geographic information in the embeddings alongside downstream performance and interpreting it in relation to the task’s need for location information.

## 5. REFERENCES

[1] Z. Feng et al., “TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026.

[2] C. F. Brown et al., “AlphaEarth Foundations: An Embedding Field Model for Accurate and Efficient Global Mapping from Sparse Label Data,” arXiv:2507.22291, 2025.

[3] K. Ayush et al., “Geography-Aware Self-Supervised Learning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 10181–10190.

[4] K. Klemmer, E. Rolf, C. Robinson, L. Mackey, and M. Rußwurm, “SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 4, 2025, pp. 4347–4355.

[5] N. Jean, S. Wang, A. Samar, G. Azzari, D. Lobell, and S. Ermon, “Tile2Vec: Unsupervised Representation Learning for Spatially Distributed Data,” in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 1, 2019, pp. 3967–3974.

[6] V. Vivanco Cepeda, G. K. Nayak, and M. Shah, “GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 8690–8701.

[7] N. Dionelis, C. Fibaek, L. Camilleri, A. Luyts, J. Bosmans, and B. Le Saux, “Evaluating and Benchmarking Foundation Models for Earth Observation and Geospatial AI,” arXiv:2406.18295, 2024.

[8] V. Marsocci et al., “PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models,” arXiv:2412.04204, 2024.

[9] Y.-C. Chang et al., “On the Generalizability of Foundation Models for Crop Type Mapping,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), 2025.

[10] P. W. Koh et al., “WILDS: A Benchmark of in-the-Wild Distribution Shifts,” in Proc. 38th Int. Conf. Mach. Learn., vol. 139, 2021, pp. 5637–5664.

[11] E. Rolf et al., “A Generalizable and Accessible Approach to Machine Learning with Global Satellite Imagery,” Nature Communications, vol. 12, art. 4392, 2021, doi:10.1038/s41467-021-24638-z.

[12] D. R. Roberts et al., “Cross-Validation Strategies for Data with Tempo ral, Spatial, Hierarchical, or Phylogenetic Structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881.

[13] P. Ploton et al., “Spatial Validation Reveals Poor Predictive Performance of Large-Scale Ecological Mapping Models,” Nature Communications, vol. 11, art. 4540, 2020, doi:10.1038/s41467-020-18321-y.

[14] N. Karasiak, J.-F. Dejoux, C. Monteil, and D. Sheeren, “Spatial Depen dence between Training and Test Sets: Another Pitfall of Classification Accuracy Assessment in Remote Sensing,” Machine Learning, vol. 111, no. 7, pp. 2715–2740, 2022, doi:10.1007/s10994-021-05972-1.

[15] T. Kattenborn, F. Schiefer, J. Frey, H. Feilhauer, M. D. Mahecha, and C. F. Dormann, “Spatially Autocorrelated Training and Validation Samples Inflate Performance Assessment of Convolutional Neural Net works,” ISPRS Open Journal ofPhotogrammetry and Remote Sensing, vol. 5, art. 100018, 2022, doi:10.1016/j.ophoto.2022.100018.

[16] G. Guillot and F. Rousset, “Dismantling the Mantel Tests,” Methods in Ecology and Evolution, vol. 4, no. 4, pp. 336–344, 2013, doi:10.1111/2041-210x.12018.

[17] C. Robinson et al., “Global Renewables Watch: A Temporal Dataset of Solar and Wind Energy Derived from Satellite Imagery,” arXiv:2503.14860, 2025.