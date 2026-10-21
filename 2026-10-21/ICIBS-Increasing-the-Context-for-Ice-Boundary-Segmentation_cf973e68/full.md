# ICIBS: Increasing the Context for Ice Boundary Segmentation

Marcel Dreier<sup>1,∗</sup>, Dakota Pyles<sup>2</sup>, Nora Gourmelon<sup>1</sup>, Moritz Koch<sup>2</sup>, Fei Wu<sup>1</sup>, Thorsten Seehaus<sup>2</sup>, Matthias Braun<sup>2</sup>, Andreas Maier<sup>1</sup>, Vincent Christlein<sup>1</sup>

Abstract—Mountain glaciers and ice sheets are undergoing rapid loss of ice mass. To assess their current volume and icethickness distribution, and to accurately forecast their future, ground-penetrating radar systems are employed to measure ice thickness. These instruments produce radio-echo sounding data that reveal the internal structure of the ice and its boundaries at the surface and beneath the ice body. However, delineating these interfaces is time-consuming as the surveyed area can span several hundred kilometers. To accelerate the labelling process, automated methods have emerged, but they often struggle to delineate the ice bottom, particularly in temperate glaciers. The high water content inherent in temperate ice increases both scattering and attenuation of the radar signal, resulting in weak and ambiguous reflections at the ice bottom and generally noisier data. As a result, the visual features corresponding to the ice bottom become ambiguous. To address this issue, we propose several extensions to deep learning models that incorporate additional contextual information into the prediction. Specifically, we integrate modifications that incorporate additional spatial and geo-spatial contextual information into the network. With these enhancements, we achieve a new stateof-the-art performance on the IceAnatomy benchmark dataset, with mean errors of 10.9 m for the FAU subset and 43.4 m for the CReSIS subset in a single combined system. We also demonstrate that our method generalises well to previously unseen data, outperforming the baseline by 49.6% on new campaign data from Patagonia. The model code and the new dataset are available at https://github.com/ki7077/ICIBS/tree/main and https://zenodo.org/records/23392580.

Index Terms—remote sensing, GPR, deep learning, radio echosounding.

## I. INTRODUCTION

Estimates indicate that 49–83 % of mountain glaciers will disappear by 2100 [1], which will have a strong impact on local environments, ecosystems, and communities [2, 3]. These projections highlight the need for robust baseline assessments of glacier thicknesses at specific points in time. However, among the more than 200,000 glaciers located outside Greenland and the Antarctic Ice Sheet, less than 5,000 have concrete ice thickness measurements [4, 5]. To address this data scarcity, scientists conduct field campaigns using groundpenetrating radar systems to measure ice thickness in previously unobserved glaciers [6]. The systems operate by transmitting electromagnetic waves into the ice and recording the returning signals, producing radargrams or image-like maps of subsurface reflection amplitudes via inversion techniques [7, 8]. These radargrams enable the interpretation of internal glacier structures along with the delineation of ice boundaries, respectively, the ice thickness, which are essential variables for predicting the future glacier evolution. However, radar surveys accumulate vast amounts of data, making manual analysis increasingly impractical [9].

![](images/b4d4b356ef45390e3772330c6dcecf22be0b7445c1f5f4ad838183ddc1eed751.jpg)  
Fig. 1. Comparing how deep learning models commonly view the problem in contrast to humans: Deep learning models typically approach radargram interpretation using a restricted spatial context to reduce computational complexity, relying primarily on local visual features for boundary prediction. In contrast, human annotators interpret radar data within a substantially broader spatial framework and additionally incorporate information from geospatially proximate radargrams to guide their decisions. The corresponding ice bottom is marked in red . The borders of the radargrams are color-coded according to their corresponding flight in gray and dark yellow

To address this issue, algorithms and deep learning models have been developed to automatically delineate the ice boundaries [10]. These approaches have achieved promising results, particularly for cold glaciers, where the recorded radar reflection amplitudes remain largely undistorted. In contrast, the results for temperate and polythermal glaciers are often worse, as water attenuates and scatters the signal, complicating echo interpretation [11]. Likewise, thick glaciers can attenuate the received signal, further complicating the distinction between internal features. In such cases, accurate delineation of the ice bottom requires frequent manual interpretation. However, that raises the question of why human annotators are better at interpreting radar data.

When comparing the processing of automatic approaches and humans in Fig. 1, we can see that automatic methods commonly process radargram transects in a patchwise manner to stabilize model training and to limit computational costs [12,

13, 14, 15, 16, 11]. While this strategy achieves the intended goals, it also restricts the model’s ability to explore the full spatial context, unlike a human who can view a larger section of the radargram. Furthermore, human annotators are not limited to the spatial features of the radargram when predicting ice boundaries. Instead, they can examine the surveyed area and flightlines, cross-reference them with radargrams from nearby regions, and use this information to infer the glacier’s internal structure. In ambiguous cases, they may also consult other experts to come to a conclusion.

Motivated by these observations, we aim to incorporate such human-inspired contextual reasoning into a deep learning framework for automatic ice-boundary delineation. Specifically, we propose:

• A novel approach to incorporate the local geospatial context;

• A new hooking mechanism for ice bottom detection in radio echo-sounding data;

• A new state-of-the-art ensembling model to refine the predictions.

To provide an overview of the general topic, we first review existing work on ice boundary extraction from radio-echo sounding data in Section II. In Section III, we present our proposed modifications and demonstrate how each can be integrated into a simple U-Net architecture. To empirically validate the efficacy of our approach, we evaluate our model on the IceAnatomy benchmark dataset in Section IV and then discuss potential weaknesses and directions for future work in Section VI. Lastly, we conclude our work in Section VII by summarizing our results and contributions.

## II. RELATED WORK

Radargrams of ice enable the extraction of a wide range of subsurface features, including snow layers, shallow stratigraphy, and ice layers [10]. This study, however, focuses specifically on the automatic delineation of the ice bottom. To solve this task, prior work has proposed a wide variety of methods, including contour-based approaches [17, 18, 19, 20], graphical models [21, 22, 23, 24], classical machine learning approaches [25, 26], and deep learning models [10].

Deep learning networks have become increasingly prevalent due to their ability to capture complex spatial patterns and their flexibility in adapting to the often noisy data distribution. These models are typically designed either to directly predict ice boundaries [12, 13, 14, 15, 16, 11] or to segment the input radargram into distinct semantical regions from which interfaces can subsequently be derived [27, 28, 29, 30, 31, 32, 33]. Among the various architectures, the convolutional U-Net [12, 16, 11, 27, 30, 34] remains the most widely adopted, as it effectively combines multi-scale semantic information with fine-grained spatial detail. Nevertheless, alternative architectures—including ResNet-based models [35], hybrid convolutional neural networks [29, 30], Vision Transformers [31, 32, 33], and convolutional encoders [13, 14, 15, 28], have also been explored.

A fundamental limitation of deep learning models is their limited spatial view of the input. Expanding the spatial resolu tion of the input typically increases the required memory and computational demands quadratically. Moreover, expanding the spatial dimensions often requires deeper and heavier network architectures, which further amplifies model complexity and resource consumption. Thus, larger input data is commonly processed using a sliding-window approach, sacrificing spatial context to remain computationally efficient.

Rijthoven et al. [36] aimed to address this by proposing Hooknet, a dual-branch architecture comprising a target branch and a context branch. The target branch processes the highresolution input patch, while the context branch captures a broader spatial region via a downsampled representation centred on the same location. Information exchange between the branches is achieved by concatenating the context branch feature maps with those of the target branch at the bottleneck layer of the target branch, thereby enabling the integration of broader contextual information without substantially increasing the depth of the target network.

The framework is inherently extensible: multiple context branches can be incorporated to further enlarge the spatial field of view [37], and enhanced coupling strategies have been developed to promote multi-resolution feature exchange [37, 38]. Although hooking-based architectures have achieved promising results in the medical domain [36, 37, 39], document analysis [40], and remote sensing [38, 41, 42], their potential has not been explored for radio-echo sounding data yet.

## III. METHODOLOGY

In this section, we explain the structure of our modifications. We start with the U-Net baseline and successively add our novel hooking mechanism and geography-aware convolutions.

## A. Base Model

As a baseline, we employ a U-Net architecture following the design by Dreier et al. [11], which consists of an encoder, a bottleneck, and a decoder. The encoder comprises five stages, each containing two residual blocks and a downsampling block that halves the spatial resolution of the feature map. In the bottleneck, the network combines residual blocks with atrous spatial pyramid pooling [43] to capture information from multiple effective receptive fields. The decoder mirrors the encoder structure but increases the number of residual blocks to three per stage and replaces the downsample blocks with upsample blocks that progressively restore the original spatial resolution. Additionally, skip connections between the encoder and the decoder merge high-resolution spatial features with deep semantic information. The final predictions for the ice surface and ice bottom are produced by two task-specific heads. Training relies on the combined regression and classification loss proposed in Dreier et al. [11], with a gradual linear increase of the regression loss contribution during the initial 200 training steps to stabilize optimization.

To strengthen the base model’s contextual awareness, we incorporate into our proposed model two additional components: a hooking module and a geographically aware convolutional module (GeoConv). The hooking module facilitates crossresolution information exchange to enlarge the effective spatial context and is appended to the end of the residual block. The GeoConv captures geospatial context by aggregating information from three geospatially adjacent radargrams, providing the network with an extended geospatial perspective. To limit the computational overhead, the hooking module is applied only to residual blocks below the second level, while GeoConv is restricted to the encoder. Figure 2 presents an overview of the model structure.

![](images/6963a4c5b24cca63d948c7111dcc390acf5536059f1888480c08c670b6340eba.jpg)  
Fig. 2. Overview of our proposed architecture, adapted from Dreier et al. [11]. The model receives three geospatially proximate radargram sections, from which three context windows are cropped, as indicated by the outline ( small , medium , and large ). To enable joint processing, the nine radargrams are normalized to a consistent size. For each radargram, the network predicts both the ice surface and the ice bottom interfaces. While both outputs are generated during training, the ice surface prediction task serves exclusively as an auxiliary task and is not considered for the evaluation.

## B. Geographical Sampling Strategy

Identifying geospatially proximate radargram sections for a given target radargram is essential for providing rich and diverse contextual information. Choosing sections from the same flight and close to the target in sequential recording order may introduce redundant information, whereas sampling geospatially distant transits weakens the underlying correlation. To balance these competing factors, we propose a three-step sampling strategy, as illustrated in Fig. 3.

First, the midpoint of the target radargram is defined as an anchor point. Second, around this anchor, a circular region with radius r is constructed, containing all potential reference points. Third, within this region, two reference points are iteratively selected along available flight paths.

The selection process follows two rules. Preference is given to geographically close sections from previously unselected flights, thus maximizing spatial correlation while preserving diversity. If no such candidates are available, the selected point maximizes its distance from already chosen references of the same flight to reduce redundancy. To increase variability among training samples, these two rules of the selection process are relaxed during training. Specifically, preference is only given to samples from distinct flights, and the distance to the anchor point is ignored.

## C. Geospatial Context

To incorporate the geospatial context of the sampled radar grams into the model, it is first necessary to analyze the spatial

![](images/0ea6f8e66e1b741e1817ff4405a2a07702eeba7e58db01a86ea115788ec076ed.jpg)  
Fig. 3. Illustration of the proposed sampling strategy for geospatially adjacent radargrams using flight lines acquired over the Perito Moreno Glacier. The background schematically represents the glacier environment, distinguishing glacier ice rock , and water . Individual flight lines are shown in green , gray , blue , pink , and dark yellow . The sampling procedure first selects an anchor point and then samples two additional reference points within a predefined radius r. Candidates from nearby, distinct flight lines are prioritized to promote diversity among the sampled radargrams. If no suitable candidates are available within this neighborhood, the procedure falls back to sampling more distant points along the same flight line.

structure of the data.

Radio-echo sounding data are acquired using a moving radar system, commonly during airborne surveys. Electromagnetic waves are emitted from a transmitter, and the reflected signals are digitally recorded and stacked to form a trace representing the vertically stacked backscatter at a single point. Each trace is assigned to a unique trace number indicating its position within the sequence of observations along the flight path and georeferenced. Horizontally stacked traces from continuous measurements are then displayed as a radargram (or B-scan), which shows the relationship between neighbouring traces. However, this representation considers only the sequential ordering of measurements and ignores the underlying geospatial structure. Consequently, traces that are geographically proximate but acquired during different flights or at substantially later times within the same flight are not positioned closely in the radargram, preventing vision-based models from effectively inferring their connection. To address this issue, we introduce the GeoConv module, which is specifically designed to combine geospatially proximate features.

![](images/82058ab2d781ca045666ee83047b45a14985727266d7d955c7f281e4a58cb22b.jpg)  
Fig. 4. GeoConv module: The inputs are three feature maps. The module first iteratively samples eight traces that maximize the geographical distance from the previously sampled traces (marked with X). For each sampled trace, the eight nearest traces (outlined in black) are then sampled to approximate the local region. A ResBlock aggregates the local information into the eight initial traces. Finally, two additional residual blocks propagate the aggregated information to every input trace, weighting the eight sampled traces according to their geographical distance from the corresponding input trace.

GeoConv is based on the idea of processing radar data as point clouds rather than conventional visual features. It views radargrams not as two-dimensional images; instead, radargrams are seen as a set of georeferenced traces. Each trace is treated as a point with an x- and y-coordinate corresponding to its geospatial location. This representation enables the use of neural network layers specifically designed for point cloud analysis. In particular, GeoConv builds upon the PointConv framework [44].

GeoConv receives three feature maps derived from geographically close radargram segments, forming a set of geospatially proximate traces C. To capture the global structure of the covered region, GeoConv first samples eight representative traces from C. These traces are selected iteratively such that each newly chosen trace maximizes its geographical distance from the previously selected ones until the eight traces are selected, forming a broad and coarse representation of the geospatial context. To refine this representation with more geospatial information, the k-nearest neighbors algorithm (KNN) identifies the eight closest traces for each sampled trace. The resulting neighboring traces were then horizontally concatenated and fused with the corresponding trace using a one-dimensional residual block. The resulting fused representations are then concatenated with every trace in the radargrams of the geospatial context C, thereby providing the entire context with geospatial information. Since the eight representatives have different proximities to every trace in the context C, their contributions are weighted using inverted and normalized geographical distances. Finally, two residual blocks merge the weighted representations into a single trace, restoring the original input dimensionality. An illustration of the structure is depicted in Fig. 4.

![](images/f66a2c7095c160d1d50a4954233c90728ddd24a1fd7acef81f29356b7e322397.jpg)  
Fig. 5. Preprocessing of the spatial context. An initial context patch of size 2048 × 1024 pixels is sampled from the dataset. Two smaller subregions of size 1024 × 1024 and 512 × 1024 pixels are then extracted from its center. Finally, all three patches are resized to 512×1024 pixels to ensure a consistent input format.

## D. Hooking

Building upon the approach of Rijthoven et al. [36], we propose a specialized hooking component tailored to radargram data. Highly detailed inputs are essential for precise iceboundary delineation, whereas downsampled representations with a bigger field of view provide more context for identifying deeper boundaries. To integrate these complementary scales, we sample three vertically centered windows from one radargram at small (512 × 1024), medium (1024 × 1024), and large (2048 × 1024) resolutions. The small resolution is the target resolution, while the medium and large resolutions serve as additional context. To maintain a consistent and efficient input size for the network, all patches are rescaled to a uniform input size of 512 × 1024, corresponding to fine-, medium-, and coarse-resolution inputs. Figure 5 illustrates the preparation of the data.

In contrast to conventional hooking architectures that rely on separate branches for each resolution, we employ a single shared branch. This design choice is motivated by the natural variation in horizontal resolutions of radargrams, which stems from different acquisition campaigns, processing pipelines, and radar systems. Thus, it is necessary for a generalizable model to learn resolution-invariant features, thereby making resolutionspecialized branches redundant. Additionally, it also reduces the overall parameter count.

Communication between the different context windows is realized through a dedicated hooking module as illustrated in Fig. 6. The hooking module first adjusts the input feature maps to match the finest granularity among the three, then extracts the central region of each map aligned with the spatial extent of the smallest context window. The cropped features are normalized and combined using a zero-initialized one-dimensional convolution across the different resolutions. The processed representations are then summed up with the uncropped feature maps through a residual connection and finally rescaled to the input size.

![](images/e8e274b691b6e0224610759d49055d88ba225266b4b4680a6a82a678c683377b.jpg)  
Fig. 6. Illustration of the proposed hooking module. The module takes as input three vertically centered radargrams with identical dimensions but differen granularity levels. First, the corresponding feature maps are rescaled to match the highest granularity among the three inputs. The central regions of the two larger rescaled feature maps are then cropped to match the spatial extent of the finest-resolution feature map. Next, the aligned feature maps are normalized and processed with a one-dimensional convolution to exchange information across the different context windows. The resulting feature representation i subsequently merged with the uncropped feature maps, and the outputs are reshaped to the original input dimensions for further processing by the network.

To enable consistent multi-resolution feature integration, the hooking module is incorporated at multiple stages of the network hierarchy. It can also be used alongside the GeoConv; however, in this configuration, each spatial location is treated as an individual sample with its own three granularity levels. Similarly, the GeoConv receives three spatially proximate radargrams for each granularity and processes each granularity independently. Since this processing strategy introduces additional computational overhead, Appendix A proposes several approaches to mitigate the associated cost.

## E. Ensembling

Delineating ice boundaries in a given radargram is a complex and error-prone task, as the basal interface is often ambiguous or barely visible. When in doubt, human annotators can ask a colleague or other domain experts for help. Some studies even plan to employ multiple annotators to obtain a more reliable ground truth [45]. We can emulate this strategy in a deep learning setting by constructing an ensemble of our proposed architecture. Specifically, we train five instances of the model from different random initializations. Each instance serves as an independent annotator that learns to predict the ice boundaries. During inference, the predicted probability maps are averaged to generate a combined prediction, thereby increasing the robustness of the final output.

## IV. EXPERIMENTS

## A. Metrics

We assess the performance of our proposed modifications with the metrics employed by Dreier et al. [11]. Specifically, we employ the Mean Absolute Error (MAE) to quantify the absolute pixel-wise distance between the ground truth y and the predicted boundary yˆ [21, 24, 20, 22, 18, 23, 17, 12, 15]. However, since the MAE only measures the distance in pixels, the real-world impact of the application is hardly tangible. Hence, we also compute the Mean Meter Error (MME) [11], which approximates the average distance between the ground truth and the prediction in meters using the mediumdependent wave velocity c<sub>medium</sub> and the radargram-specific vertical resolution of the two-way travel time VR. The precise formulas for both are given in equations 1 and 2.

$$
\mathbf { M A E } ( \hat { y } , y ) = \qquad \quad \frac { 1 } { | \hat { Y } | } \sum _ { \hat { y } \in \hat { Y } } | \hat { y } - y |\tag{1}
$$

$$
\mathbf { M M E } ( \hat { y } , y ) = \qquad \quad \frac { c _ { \mathrm { m e d i u m } } } { 2 | \hat { Y } | } \sum _ { \hat { y } \in \hat { Y } } \mathbf { V R } \cdot | \hat { y } - y |\tag{2}
$$

One downside of the distance-based metrics, like the MAE and MME, is their sensitivity to outliers. A few wrong predictions far from the ground truth can disproportionately increase them, even when most predictions are accurate. To mitigate the effects of such a scenario, we also compute the Average Precision (AP) [23, 17, 15]. This metric measures the relative frequency of correct predictions compared to the total number of predictions. As ice boundaries are usually ambiguous on the pixel level, we relax the strict correctness criterion by employing the Average Precision - 1% (AP-1%) and Average Precision - 5% (AP-5%). These metrics consider predictions within a defined error bound around the ground truth label of the radargram as correct. We define their error bounds as 1% and 5% of the pixel height H of the radargram, respectively. Equations 3 and 4 provide their precise mathematical definitions.

$$
\mathrm { A P } _ { 1 \mathcal { G } _ { 0 } } ( \hat { y } , y ) = \qquad \frac { 1 } { | \hat { Y } | } \sum _ { \hat { y } \in \hat { Y } } \mathbb { 1 } \left[ | \hat { y } - y | < \frac { \mathrm { H } } { 1 0 0 } \right]\tag{3}
$$

$$
\mathbf { A P } _ { 5 \mathcal { Y } _ { 0 } } ( \hat { y } , y ) = \qquad \quad \frac { 1 } { | \hat { Y } | } \sum _ { \hat { y } \in \hat { Y } } \mathbb { 1 } \left[ | \hat { y } - y | < \frac { \mathrm { H } } { 2 0 } \right]\tag{4}
$$

## B. Dataset

We train and evaluate our model on the IceAnatomy benchmark dataset [11] and its standardized training and evaluation splits. IceAnatomy consists of three distinct subsets: CReSIS, FAU, and AWI. Each one is gathered by a different institution using a different radar system, processing scheme, and focusing on a different geographic region.

The CReSIS subset contains data from NASA’s Operation IceBridge campaigns conducted in Antarctica in 2009, covering predominantly polythermal regions that include both ice sheets and ice shelves. It comprises 18 missions, including 12 over West Antarctica and 6 over the Antarctic Peninsula, with 3 of the 18 missions flown at high altitudes. According to the benchmark dataset, we utilize 11 missions for training, 3 for validation, and 4 for testing. All splits include missions from both regions, while high-altitude missions are exclusively used for training.

The FAU subset comprises radargrams acquired during multiple flights over three mountain glaciers. Specifically, three flights were conducted over the Perito Moreno Glacier in March and April, two over the Viedma Glacier during the same period, and two over the Gourdon Glacier in February. The Perito Moreno and Viedma Glaciers are both located in the Southern Patagonian IceField and mainly consist of temperate ice with high water content. This characteristic complicates the interpretation of the corresponding radargrams, as liquid water can scatter the radar signal, leading to ambiguity in the ice bottom position. In contrast, the Gourdon Glacier, located on James Ross Island at the east side of the Antarctic Peninsula, consists primarily of polythermal ice. This makes the interpretation of the corresponding radargrams much clearer, as less noise is introduced into the radargram. For testing, we utilize the third flight of the Perito Moreno Glacier, the second flight over the Viedma Glacier, and the second flight over the Gourdon Glacier. The remaining flights are used for training and validation.

The AWI subset includes radargrams from the Antarctic Peninsula and East Antarctica. Despite its broad physical coverage, the subset is characterized by a coarse horizontal resolution, which results in a limited number of annotated data points. In particular, the test set does not contain a single radargram segment with more than 2048 annotated traces, preventing a meaningful evaluation of our hooking mechanism, which utilizes a context width of 2048 pixels. Furthermore, due to the different processing schemes between the three collected flights, it is also not possible to rearrange the subset to isolate a single flight for testing. Consequently, we decided to exclude the AWI subset from our experiments.

A limitation of the standardized IceAnatomy data splits is their limited ability to assess model generalization to newly acquired campaign data. Although test and train sets consist of mutually exclusive flights, they are not region-, season-, or campaign-exclusive. As a result, models trained and evaluated under this protocol are not required to generalize to unseen campaigns, which constrains their practical applicability. Furthermore, this setup divides the available flights for each region in training and testing, substantially restricting the geospatial context for both. Consequently, the model has reduced capacity to learn meaningful geospatial relationships, and the evaluation does not fully reflect the potential of the GeoConv. To address these issues, we introduce regionally exclusive splits to validate our proposed architectural enhancements in additional experiments. For the CReSIS dataset, all missions conducted in West Antarctica are used for training and validation, while missions from the Antarctic Peninsula are reserved for testing. In the FAU dataset, training and validation are performed using data from the Viedma and Gourdon Glaciers, with the Perito Moreno Glacier reserved for testing. To ensure generalization to new campaign data, we extend the FAU test set with newly acquired data from the Upsala Glacier in the Southern Patagonian Ice Field [46]. Upsala Glacier is also a temperate mountain glacier and faces similar challenges to the Perito Moreno and the Viedma Glacier. Although its data were collected and processed using the same radar system and processing scheme, they were acquired during the spring season (October), in contrast to the Perito Moreno, Viedma, and Gourdon glaciers, where data acquisition primarily occurred between late summer and early autumn (February-April). As such, this dataset is both regionally and seasonally exclusive, providing an out-of-distribution test case representative of realworld deployment.

We note that cross-subset generalization between the FAU and CReSIS datasets is not considered, because differences in radar systems and processing pipelines result in a domain shift that is too large to permit meaningful evaluation.

## C. Experimental Protocol

We conduct three sets of experiments to evaluate our proposed architecture. First, we compare our proposed approach against the baseline presented in Dreier et al. [11] using the standardized IceAnatomy data splits. For this, we train one omni model–defined as a model jointly trained on both the CReSIS and FAU subsets– to enable comparison within the benchmark framework and to validate that our model can generalize to both subsets simultaneously. Second, we assess the effectiveness of our new components using the regional exclusive splits defined in Section IV-B. Here, we train separate models for the FAU and CReSIS subsets to demonstrate the full potential of the proposed modifications under region-exclusive conditions and their capacity to specialize to distinct data sources. Third, we evaluate the ability of our architecture to generalize to newly acquired campaign data. For this experiment, we evaluate the model trained on the regionally exclusive FAU split using unseen data from the Upsala Glacier.

To train our model, we rescale the radargrams to a consistent height of 1024 pixels and crop patches of 512 × 1024 pixels. For our two new context modules, we require additional input data. First, the proposed hooking mechanism requires two additional radargrams of 1024 × 1024 and 2048 × 1024 resolution as spatial context. We center them on the original 512 × 1024 crop and then resize to match the input resolution. Second, the geospatial context mechanism requires two additional radargrams in physical proximity to the initial one. We choose these radargrams according to the rules defined in Section III-C, and set the look-up radius to 250 m for the CReSIS subset and 50 m for the FAU subset to account for the significantly different geospatial distances between traces in the two subsets.

TABLE I  
THIS TABLE SUMMARIZES THE QUANTITATIVE COMPARISON OF OUR PROPOSED MODEL WITH THE BASELINE ON THE STANDARDIZED SPLITS OF THE ICEANATOMY DATASET. BOTH APPROACHES WERE TRAINED ON THE FAU AND CRESIS SUBSETS SIMULTANEOUSLY ACROSS FIVE INDEPENDENT RUNS, AND THE REPORTED VALUES REPRESENT THE MEAN AND STANDARD DEVIATION. THE ENSEMBLING MODEL CONSISTS OF THE COMBINATION OF THE FIVE TRAINED MODELS. A MORE DETAILED VERSION OF THE RESULTS CAN BE FOUND IN THE APPENDIX B IN TAB. IV AND V.
<table><tr><td colspan="8">Ice Boundary Delineation</td></tr><tr><td></td><td colspan="4">FAU</td><td colspan="3"> $C R e S I S$ </td></tr><tr><td>Model</td><td>MAE↓ MME↓</td><td> $A P  – I \% \uparrow$ </td><td>AP-5%↑|</td><td>MAE↓</td><td> $M M E \downarrow$ </td><td> $A P  – I \% \uparrow$ </td><td>AP-5%↑</td></tr><tr><td>Baseline</td><td> $2 5 . 8 \pm 5 . 0$ </td><td> $1 9 . 1 \pm 4 . 0$   $6 7 . 9 \pm 2 . 4$ </td><td> $9 1 . 8 \pm 1 . 6$ </td><td> $1 6 . 4 \pm 4 . 7$ </td><td> $8 4 . 5 \pm 2 2 . 7$ </td><td> $8 6 . 3 \pm 1 . 3$ </td><td> $9 3 . 0 \pm 1 . 1$ </td></tr><tr><td>Ours</td><td> $1 8 . 8 \pm 4 . 2$ </td><td> $1 3 . 7 \pm 3 . 4$   $7 0 . 5 \pm 5 . 3$ </td><td> $9 4 . 0 \pm 1 . 1$ </td><td> $1 1 . 0 \pm 1 . 9$ </td><td> $5 6 . 9 \pm 9 . 2$ </td><td> $8 8 . 8 \pm 1 . 3$ </td><td> $9 5 . 3 \pm 1 . 0$ </td></tr><tr><td>+Ensemble</td><td>15.3</td><td>10.9</td><td>73.4 95.2</td><td>1 8.3</td><td>43.4</td><td>91.0</td><td>96.3</td></tr></table>

As optimizer, we employ AdamW [47] with a batch size of 32 and a learning rate of 0.0005. We halve the learning rate if the ice-bottom AP-1% does not improve for 10 epochs, and we stop training after 35 consecutive epochs without improvement. Every epoch consists of 3000 samples, except for the experiments using only the FAU subset, where 500 samples are used due to the reduced dataset size. To prioritize deeper and more challenging ice bottoms during training, we weight the probability of choosing a sample from a respective flight or section by the sum of the ice surface and ice bottom picks, normalized by the total sum of ice surface and ice bottom picks over the entire dataset. Afterward, we randomly crop a sample from the chosen section. For augmentations, we employ random vertical flips with a probability of $\mathrm { p } = 0 . 5$ , gamma correction $\mathrm { p = 0 . 2 }$ , random brightness and contrast adjustments with $\mathrm { p = 0 . 2 }$ , and random noise with $\mathrm { p = 0 . 2 }$

## V. RESULTS

As shown in Tab. I, our proposed additions to the omni model consistently outperform the baseline of Dreier et al. [11] across all metrics. The MME decreased by 5.4 m on the FAU-subset and by 27.6 m on the CReSIS subset, corresponding to a relative error reduction of 28.3 % and 32.6 %, respectively. Incorporating ensembling further improves performance, yielding final MMEs of 10.9 m on the FAU subset and 43.4 m on the CReSIS subset. Similar improvements are observed for precision, with gains of 5.5 and 4.7 percentage points in AP-1%, and 3.4 and 3.3 percentage points in AP-5% compared to the baseline. Overall, these results suggest that the proposed approach improves both prediction accuracy and robustness across both subsets simultaneously.

These observations are also reflected in the qualitative results presented in Fig. 7. In the depicted radargrams from the Perito Moreno Glacier (FAU subset) and West Antarctica (CReSIS subset), the ensemble model remains more closely aligned with the ground truth in deeper ice layers. In contrast, the baseline model often confuses the ice surface with the bottom of the ice. Moreover, the ensemble model enhances the spatial consistency across consecutive predictions, as the frequency of large jumps in the predicted layer decreases. Nonetheless, both models still struggle in the deepest sections of the radargram, where visual features become unclear due to the weak return signal.

An analysis of the region-specific splits in Tab. II shows that each proposed component yields consistent performance improvements. On the FAU and CReSIS subsets, the introduction of the GeoConv reduces the MME by 6.9 m and 3.8 m, respectively. Incorporating the hooking mechanism yields further improvements of 5.0 m on the FAU and 7.5 m on the CReSIS subset. Finally, the addition of ensembling results in an MME of 21.9 m on the FAU subset, corresponding to a total error reduction of 48.7 %, and an MME of 61.4 on the CReSIS subset, corresponding to a total reduction of 29.6%. Alongside these reductions, both AP-1% and AP-5% consistently increase with the introduction of each module. Compared to the baseline, the FAU subset exhibits gains of 6.7 and 7.6 percentage points in AP-1% and AP-5%, respectively, whereas the CReSIS subset shows improvements of 5.2 and 3.0 percentage points.

Figure 8 visualizes the consistent qualitative improvements with each proposed component. In the FAU subset example, the baseline model struggles to accurately delineate the ice bottom in the deeper regions of the glacier. Incorporating the GeoConv module mitigates this issue, enabling the model to approximate the layer location through the geospatial context; however, the resulting predictions remain fragmented and exhibit notable discontinuities. The introduction of the hooking mechanism substantially improves spatial coherence, yielding more continuous predictions in deeper glacier sections, with only minor outliers near the surface. Finally, the ensemble model further stabilizes the results by reducing outliers via prediction averaging. Similar trends are evident in the CReSIS example in Fig. 8. The baseline model fails to accurately delineate the ice boundary, whereas the inclusion of GeoConv improves predictions in the deeper regions of the ice. With the addition of the hooking mechanism, the predicted boundaries align more closely with the ground truth, particularly in the upper portions of the radargrams. The ensemble then further improves the predictions by reducing outliers and minimizing sensitivity to visual artifacts.

Examining the results of the final experiment on the Upsala Glacier in Tab. III, we observe substantial performance improvements from each proposed component. Specifically, the MAE and MME decreased by 40.0 pixels and 33.6 m, while the AP-1% and the AP-5% showed improvements of 6.3 and 8.5 percentage points. Despite these gains, the out-of-distribution Upsala experiment remains more challenging than simply isolating the Perito Moreno Glacier, as the new campaign introduces scenarios that diverge from prior conditions, such as the season or greater depths. This is also indicated by the higher errors for the MAE and MME when compared to the regional-exclusive evaluation of the Perito Moreno Glacier. A qualitative comparison between the proposed modifications and the baseline in Fig. 9 indicates that these errors primarily arise from predictions of deep ice bottom layers. Nevertheless, the proposed approach helps to substantially reduce these errors.

FAU  
CReSIS  
![](images/9860fd537c15264f939a548dfc0bf08319da2767673c95c8564f0e0a436564bb.jpg)  
Fig. 7. Qualitative results of the baseline and the ensemble model. Left: results from the third flight over the Perito Moreno Glacier portion of the FAU subset. Right: section from the TSK2-07 mission in the West Antarctica portion of the CReSIS subset. The yellow line indicates the ice bottom prediction, while the red dotted line indicates the ground truth.

TABLE II  
RESULTS ON THE ICEANATOMY DATASET USING THE NEWLY DEFINED REGION-EXCLUSIVE SPLITS. TO REDUCE THE INFLUENCE OF STATISTICALOUTLIERS, EACH EXPERIMENT IS REPEATED FIVE TIMES, AND THE MEAN AND STANDARD DEVIATION ARE REPORTED. FOR THE ENSEMBLE, RESULTS AREREPORTED FOR A SINGLE ENSEMBLE INSTANCE, AS REPEATING THE EXPERIMENT FIVE TIMES WOULD REQUIRE TRAINING $5 \times 5$ INDIVIDUAL MODELS ANDINCUR SUBSTANTIAL COMPUTATIONAL COST. A DETAILED BREAKDOWN OF THE RESULTS IS PROVIDED IN APPENDIX B, TABLES VI AND VII.
<table><tr><td></td><td colspan="8">Ice Boundary Delineation</td></tr><tr><td></td><td colspan="4">FAU</td><td colspan="4">CReSIS</td></tr><tr><td>Model</td><td>MAE↓</td><td> $M M E \downarrow$ </td><td> $A P  – I \% \uparrow$ </td><td>AP-5%↑ 一</td><td>MAE↓</td><td>MME↓</td><td> $A P  – I \% \uparrow$ </td><td>AP-5%↑</td></tr><tr><td>Baseline</td><td> $5 0 . 9 \pm 8 . 1$ </td><td> $4 2 . 7 \pm 6 . 8$ </td><td> $4 8 . 5 \pm 0 . 7$ </td><td> $8 1 . 4 \pm 1 . 1$ </td><td> $1 5 . 4 \pm 1 . 4$ </td><td> $8 7 . 2 \pm 8 . 2$ </td><td> $8 2 . 3 \pm 1 . 2$ </td><td> $9 1 . 8 \pm 0 . 7$ </td></tr><tr><td>+GeoConv</td><td> $4 2 . 7 \pm 2 . 3$ </td><td> $3 5 . 8 \pm 1 . 9$ </td><td> $4 9 . 4 \pm 1 . 2$ </td><td> $8 3 . 7 \pm 0 . 5$ </td><td> $1 4 . 7 \pm 1 . 2$ </td><td> $8 3 . 4 \pm 6 . 2$ </td><td> $8 2 . 7 \pm 2 . 0$ </td><td> $9 2 . 5 \pm 0 . 7$ </td></tr><tr><td>+Hooking</td><td> $3 6 . 7 \pm 5 . 0$ </td><td> $3 0 . 8 \pm 4 . 2$ </td><td> $5 0 . 8 \pm 1 . 9$ </td><td> $8 5 . 9 \pm 1 . 4$ </td><td> $1 3 . 5 \pm 1 . 0$ </td><td> $7 5 . 9 \pm 6 . 1$ </td><td> $8 4 . 2 \pm 2 . 1$ </td><td> $9 3 . 4 \pm 1 . 0$ </td></tr><tr><td>+Ensemble</td><td>26.1</td><td>21.9</td><td>55.2</td><td>89.0</td><td>11.1</td><td>61.4</td><td>87.5</td><td>94.8</td></tr></table>

TABLE III

QUANTITATIVE EVALUATION OF OUR SUBSET MODEL ON RADIO-ECHO SOUNDING DATA FROM THE UPSALA GLACIER. RESULTS ARE REPORTED AS THE MEAN AND STANDARD DEVIATION ACROSS FIVE INDEPENDENT RUNS. THE ENSEMBLE MODEL COMBINES THE PREDICTIONS OF THE FIVE INDIVIDUALLY TRAINED MODELS.
<table><tr><td>Region</td><td>MAE↓</td><td>MME↓</td><td>AP-1%↑</td><td>AP-5%↑</td></tr><tr><td>Baseline</td><td> $8 0 . 8 \pm 1 7 . 4$ </td><td> $6 7 . 8 \pm 1 4 . 6$ </td><td> $5 0 . 9 \pm 1 . 0$ </td><td> $8 0 . 9 \pm 0 . 6$ </td></tr><tr><td>+GeoConv</td><td> $5 7 . 8 \pm 8 . 0$ </td><td> $4 8 . 5 \pm 6 . 7$ </td><td> $5 5 . 8 \pm 2 . 2$ </td><td> $8 6 . 2 \pm 1 . 0$ </td></tr><tr><td>+Hooking</td><td> $5 1 . 5 \pm 1 3 . 9$ </td><td> $4 3 . 2 \pm 1 1 . 7$ </td><td> $5 2 . 7 \pm 4 . 3$ </td><td> $8 6 . 8 \pm 2 . 1$ </td></tr><tr><td>+Ensemble</td><td>40.8</td><td>34.2</td><td>57.2</td><td>89.4</td></tr></table>

## VI. DISCUSSION

Our experimental results demonstrate that our additions lead to substantial improvements on both the FAU subset and the CReSIS subset. Notably, the results with the omni model indicate that our modifications generalize to different types of radargrams and acquisition settings simultaneously, while achieving state-of-the-art results. A closer look at the results reveals that the proposed additions especially improve the detection of deeper ice bottom layers. We hypothesize that the extended context helps locate the ice boundaries when the visual features are insufficient or ambiguous. However, our current approach still suffers from occasional outlier predictions where the current context information seems to be insufficient to stabilize the prediction, resulting in discontinuities in the predicted layer. We hypothesize that such cases could be mitigated by incorporating additional structural contextual information from the predicted layer. For example, additional regularizers could favor smoother predictions or directly enforce structural coherence.

In the second set of experiments, we employed the newly proposed regional splits and subset-specific models. Our evaluation indicates that, on average, each additional com ponent consistently improves the model’s performance. In detail, the GeoConv provides broader contextual information via information from other flights that enables coarse-layer localization in deeper and ambiguous regions of the ice, while the hooking mechanism refines the coarse estimate via spatial cues from the surrounding context. The ensemble then further stabilizes the overall prediction by mitigating outliers through averaging over multiple model outputs. These findings suggest that the proposed components are especially beneficial for the localization of deep and ambiguous layers, as they introduce additional context clues that allow cross-validation between multiple patches and radargrams. Thus, the most challenging study sites with deep and ambiguous layers—namely the Perito Moreno Glacier in the FAU subset and West Antarctica in the CReSIS subset—exhibit the strongest performance increase compared to less complex sites such as the Viedma Glacier, Gourdon Glacier, or the Antarctic Peninsula. This trend is evident when analyzing the relative gains in the regional split and the general IceAnatomy split. For the FAU subset, the ensemble model achieves a higher relative gain on the regionspecific split, which evaluates exclusively on the Perito Moreno Glacier, than on the general IceAnatomy split. In contrast, for the CReSIS subset, larger relative gains are observed under the general IceAnatomy split, which includes radargrams from West Antarctica, compared to the region-specific split restricted to the Antarctic Peninsula. However, performance across the CReSIS subset varies strongly among individual missions, reducing the regional influence on the overall evaluation.

![](images/fbcffb95e93e201b148bacbfe2fb08887711847c368503ff69c0623e20659894.jpg)  
Fig. 8. Qualitative comparison of model predictions as the proposed modifications are progressively incorporated. The left panels show radargrams from the first flight over the Perito Moreno Glacier, included in the FAU subset, while the right panels show a radargram from the PEN2-01 mission in the CReSIS subset. The yellow line denotes the predicted ice-bottom, whereas the red dotted line denotes the ground-truth annotation.

In the third and final experiment, we evaluate the FAU-subset specific model on newly acquired data from the Upsala Glacier.

Again, the proposed modifications outperform the baseline, setting state-of-the-art results. When directly comparing the results from the Perito Moreno and the Upsala Glacier, we notice that the MAE and MME for the Upsala Glacier are substantially higher, despite both glaciers containing deep and structurally complex regions. This suggests that newly acquired campaign data may be more challenging than a region-exclusive subset such as Perito Moreno, as it may contain seasonal- or campaign-specific artifacts or a smaller and more restricted geospatial context. Nonetheless, our proposed modifications still improve the AP-1% and MME, indicating that the proposed architectural additions can generalize effectively to newly acquired campaign data. Moreover, the relative performance gains are consistent with those observed for the Perito Moreno Glacier, further supporting the hypothesis that the proposed components are particularly beneficial for deep and structurally complex regions. However, the visual examination indicates that, despite noticeable improvements, the detection of very deep layers, especially those extending over large spatial extents, remains challenging. We hypothesize that these limitations stem from two main factors: first, deeper layers exhibit weaker return signals, which inherently complicates their detection, and second, the subset-specific model was trained exclusively on data from the Viedma and Gourdon Glaciers, which rarely exhibit such extreme extended depths, thereby introducing a bias toward shallower regions.

![](images/21af046daabaed1515927763c33eec45ea3ea5100aa2607f0788daafe54fe1f0.jpg)  
Fig. 9. Qualitative comparison of the baseline and ensemble model predictions on the Upsala Glacier. The yellow line denotes the predicted ice-bottom, whereas the red dotted line denotes the ground-truth annotation.

Baseline

+GeoConv

## VII. CONCLUSION

In this work, we extend a U-Net baseline by introducing two additional modules designed to incorporate contextual information. The first one, GeoConv, provides geospatial context to the model by processing radar traces according to their spatial positions. The second one is a new hooking module designed for radio-echo sounding data, which expands the model’s spatial view and enables greater spatial reasoning.

Our evaluation demonstrates that both extensions substantially improve upon the initial baseline model. When combined with ensembling, our approach achieves a new state-of-theart omni model performance on the IceAnatomy benchmark dataset, with errors of 10.9 m for the FAU subset and 43.4 m for the CReSIS subset. Compared to the baseline, this corresponds to a relative decrease in error of 42.9 % and 48.6 %, respectively. Furthermore, we propose two additional challenging experimental setups. In the first setup, the model is retrained for the FAU and CReSIS subsets individually; however, the data splits are rearranged to be geospatially exclusive. In the second setup, one of the models obtained from the first experiment is further evaluated on previously unseen campaign data from the Upsala glacier to assess its generalization capability to new campaign conditions. For both experiments, the proposed components lead to consistent performance improvements. In the first setup, the ensemble model achieves an MME of 21.9 m for the FAU subset and 61.4 m for the CReSIS subset, corresponding to an error reduction of 48.7% and 29.6% compared to the baseline method. For the second experiment, the proposed approach achieves an overall error of 34.2 m, corresponding to a relative improvement of 49.6 % compared to the baseline. These results highlight the importance of incorporating contextual information to accurately delineate ice boundaries.

## APPENDIX A

## RUNTIME OPTIMIZATIONS

Although the baseline model is relatively lightweight, the proposed modifications introduce computational overhead.

This cost could be reduced in future work through several straightforward inference strategies. First, when sampling geospatially proximate regions, unevaluated sections could be prioritized to avoid redundant computations. Second, unlike during training, larger context windows do not need to be processed during inference once their final interaction with the target granularity has occurred. Furthermore, by restricting the GeoConv and the hooking module to the encoder, the decoder would not need to process additional context images during inference, thereby substantially limiting the computational overhead. Alternatively, the additional context outputs could be leveraged to further refine the final prediction. For example, multiple outputs corresponding to the same location could be weighted and combined to produce a single stable prediction.

## APPENDIX B

## ADDITIONAL RESULTS

To demonstrate the effects of the proposed modifications in various geographical settings, we evaluate each conducted mission and report the results. The performance of the omni model on the general IceAnatomy splits is summarized in Tabs. IV and V, while Tabs. VI and VII present the results for regionexclusive splits using subset-specific models. From the results, we can see that the proposed ensemble model outperforms the baseline across all analyzed settings.

## CODE AND DATA AVAILABILITY

The code will be made publicly available on GitHub in the following repository https://github.com/ki7077/ICIBS/tree/main. The dataset can be accessed and downloaded from https:// zenodo.org/records/23368993 and https://zenodo.org/records/ 23392580.

## ACKNOWLEDGMENTS

This research was funded by the Bayerisches Staatsministerium fur Wissenschaft und Kunst within the Elite Network¨

TABLE IV  
COMPARISON OF THE PROPOSED ENSEMBLE MODEL AND THE BASELINE METHOD ACROSS DIFFERENT GEOGRAPHICAL SETTINGS IN THE CRESIS SUBSET. BOTH MODELS ARE TRAINED ON THE FAU AND CRESIS SUBSETS USING THE OFFICIAL TRAIN-TEST SPLITS OF THE ICE-ANATOMY BENCHMARK DATASET.
<table><tr><td rowspan="2">Mission</td><td colspan="2">Baseline</td><td colspan="2">Ensemble</td></tr><tr><td>MME↓</td><td>AP-1%↑|</td><td>MME↓</td><td>AP-1% ↑</td></tr><tr><td colspan="5">West Antarctica</td></tr><tr><td>TSK2-04</td><td>192.5</td><td>68.7</td><td>103.8</td><td>74.0</td></tr><tr><td>TSK2-07</td><td>373.6</td><td>74.7</td><td>146.3</td><td>85.2</td></tr><tr><td>PIG4-01</td><td>60.1</td><td>83.7</td><td>21.1</td><td>93.6</td></tr><tr><td>PIG4-02</td><td>68.9</td><td>86.6</td><td>57.0</td><td>88.9</td></tr><tr><td colspan="5">Antarctic Peninsula</td></tr><tr><td>PEN4-01</td><td>34.5</td><td>93.3</td><td>14.8</td><td>95.6</td></tr><tr><td>PEN5-01</td><td>43.0</td><td>86.8</td><td>24.2</td><td>92.3</td></tr><tr><td>PEN5-02</td><td>49.5</td><td>82.6</td><td>24.7</td><td>89.9</td></tr></table>

TABLE V

COMPARISON OF THE PROPOSED MODIFICATIONS AND THE BASELINE METHOD ACROSS DIFFERENT GEOGRAPHICAL SETTINGS IN THE FAU SUBSET. THE MODELS ARE TRAINED ON BOTH THE FAU AND CRESIS SUBSETS USING THE OFFICIAL SPLITS OF THE ICE-ANATOMY BENCHMARK DATASET.
<table><tr><td rowspan="2">Mission</td><td colspan="2">Baseline</td><td colspan="2">Ensemble</td></tr><tr><td>MME↓</td><td>AP-1%↑</td><td>MME↓</td><td>AP-1% ↑</td></tr><tr><td>Perito Moreno 3</td><td>46.2</td><td>42.6</td><td>22.2</td><td>51.9</td></tr><tr><td>Viedma 2</td><td>26.3</td><td>58.0</td><td>15.8</td><td>64.4</td></tr><tr><td>Gourdon 2</td><td>5.0</td><td>83.3</td><td>3.8</td><td>86.8</td></tr></table>

Bavaria with the Int. Doct. Program “Measuring and Modelling Mountain Glaciers in a Changing Climate” (IDP M3OCCA) as well as the German Research Foundation (DFG) project “Large-scale Automatic Calving Front Segmentation and Frontal Ablation Analysis of Arctic Glaciers using Synthetic-Aperture Radar Image Sequences (LASSI)” (grant number CH 2080/5-1 and SE 3091/4-1) and the project “PAGE” (grant number SE 3091/5-1) within the DFG Emmy-Noether-Programme. The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universitat Erlangen-N¨ urnberg (FAU) under the NHR projects¨ b110dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the DFG – 440719683. During the preparation of this work, AI technologies were used to assist in the writing process. Specifically, Grammarly (Grammarly, Inc., San Francisco, CA, USA) was used in order to check for grammar and style consistency and DeepL (DeepL SE, Cologne, Germany) and ChatGPT (GPT-4) (OpenAI, San Francisco, CA, USA) were used in order to assist with rephrasing and improving readability. After using these tools, the manuscript was carefully reviewed and the content was edited as needed. No tools or services were used for content generation.

## AUTHOR CONTRIBUTION

Marcel Dreier: Conceptualization, Methodology, Software, Project administration, Writing - Original draft preparation. Dakota Pyles: Writing - review & editing. Nora Gourmelon:

Writing - review & editing. Moritz Koch: Data Collection, Writing - review & editing. Fei Wu: Writing - review & editing. Thorsten Seehaus: Supervision, Writing - review & editing. Matthias Braun: Data Collection, Supervision, Writing – review & editing. Andreas Maier: Supervision, Writing – review & editing. Vincent Christlein: Supervision, Writing - review & editing.

## REFERENCES

[1] D. R. Rounce et al., “Global glacier change in the 21st century: Every increase in temperature matters,” Science, vol. 379, no. 6627, pp. 78–83, 2023. 1

[2] Intergovernmental Panel on Climate Change (IPCC), Climate change 2022 – impacts, adaptation and vulnerability. Cambridge, England: Cambridge University Press, Jun. 2023. 1

[3] K. Calvin et al., “IPCC, 2023: Climate change 2023: Synthesis report. contribution of working groups i, II and III to the sixth assessment report of the intergovernmental panel on climate change [core writing team, h. lee and j. romero (eds.)]. IPCC, geneva, switzerland,” Intergovernmental Panel on Climate Change, Tech. Rep., Jul. 2023. 1

[4] R. Hock, F. Maussion, B. Marzeion, and S. Nowicki, “What is the global glacier ice volume outside the ice sheets?” Journal of Glaciology, vol. 69, no. 273, p. 204–210, 2023. 1

[5] E. Welty et al., “Worldwide version-controlled database of glacier thickness observations,” Earth System Science Data, vol. 12, no. 4, pp. 3039–3055, 2020. [Online]. Available: https://essd.copernicus.org/articles/ 12/3039/2020/ 1

[6] M. Koch et al., “The state and fate of glaciar perito moreno patagonia,” Communications Earth & Environment, vol. 6, no. 1, p. 572, 2025. 1

[7] N. Blindow, C. Salat, V. Gundelach, U. Buschmann, and W. Kahnt, “Performance and calibration of the helicoper gpr system bgr-p30,” in 2011 6th International Workshop on Advanced Ground Penetrating Radar (IWAGPR). IEEE, 2011, pp. 1–5. 1

[8] C. Allen et al., “Antarctic ice depthsounding radar instrumentation for the nasa dc-8,” IEEE Aerospace and Electronic Systems Magazine, vol. 27, no. 3, pp. 4–20, 2012. 1

[9] L. C. Sime, R. C. Hindmarsh, and H. Corr, “Automated processing to derive dip angles of englacial radar reflectors in ice sheets,” Journal of Glaciology, vol. 57, no. 202, pp. 260–266, 2011. 1

[10] H. Moqadam and O. Eisen, “Review article: Feature tracing in radio-echo sounding products of terrestrial ice sheets and planetary bodies,” The Cryosphere, vol. 19, no. 6, pp. 2159–2196, 2025. [Online]. Available: https://tc.copernicus.org/articles/19/2159/2025/ 1, 2

[11] M. Dreier et al., “Iceanatomy: a benchmark dataset and methodology for automatic ice boundary extraction from radio-echo sounding data,” The Cryosphere, vol. 19, no. 11, pp. 5337–5359, 2025. [Online]. Available:

TABLE VI  
EFFECT OF THE PROPOSED MODIFICATIONS ON THE CRESIS SUBSET MODEL EVALUATED USING REGION-EXCLUSIVE SPLITS ACROSS DIFFERENT FLIGHTS.THE BEST-PERFORMING CONFIGURATIONS ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Mission</td><td colspan="2">Baseline</td><td colspan="2">+GeoConv</td><td colspan="2">+Hooking</td><td colspan="2">+Ensemble</td></tr><tr><td>MME↓</td><td>AP-1% ↑</td><td>MME↓</td><td>AP-1% ↑</td><td>MME↓</td><td>AP-1% ↑</td><td>MME↓</td><td>AP-1% ↑</td></tr><tr><td>PEN4-01</td><td>36.7</td><td>92.6</td><td>33.7</td><td>93.5</td><td>27.1</td><td>93.8</td><td>18.0</td><td>94.6</td></tr><tr><td>PEN5-01</td><td>69.8</td><td>82.3</td><td>71.2</td><td>81.3</td><td>62.6</td><td>82.8</td><td>50.4</td><td>86.8</td></tr><tr><td>PEN5-02</td><td>90.6</td><td>76.8</td><td>93.1</td><td>78.0</td><td>81.1</td><td>80.2</td><td>68.0</td><td>84.7</td></tr><tr><td>PEN3-01</td><td>56.7</td><td>87.8</td><td>53.8</td><td>88.5</td><td>45.8</td><td>89.6</td><td>31.2</td><td>94.0</td></tr><tr><td>PEN3-02</td><td>197.7</td><td>65.0</td><td>230.6</td><td>60.0</td><td>208.8</td><td>64.0</td><td>186.9</td><td>69.5</td></tr><tr><td>PEN2-01</td><td>220.5</td><td>57.8</td><td>179.0</td><td>60.7</td><td>152.6</td><td>67.3</td><td>133.1</td><td>70.0</td></tr><tr><td>PEN2-02</td><td>162.9</td><td>70.3</td><td>153.1</td><td>71.7</td><td>163.7</td><td>72.3</td><td>152.2</td><td>75.5</td></tr><tr><td>PEN1-01</td><td>17.4</td><td>97.1</td><td>23.3</td><td>96.0</td><td>18.3</td><td>96.4</td><td>13.0</td><td>97.2</td></tr><tr><td>PEN1-02</td><td>51.2</td><td>91.2</td><td>46.8</td><td>92.6</td><td>40.4</td><td>93.1</td><td>36.4</td><td>93.9</td></tr><tr><td>LVISPEN-01</td><td>206.5</td><td>58.0</td><td>188.2</td><td>56.8</td><td>138.7</td><td>67.0</td><td>86.3</td><td>74.8</td></tr><tr><td>LVISPEN-02</td><td>166.5</td><td>62.5</td><td>181.7</td><td>58.6</td><td>108.5</td><td>73.3</td><td>62.9</td><td>82.0</td></tr></table>

TABLE VII

EFFECT OF THE PROPOSED MODIFICATIONS ON THE FAU SUBSET MODEL EVALUATED USING REGION-EXCLUSIVE SPLITS ACROSS DIFFERENT FLIGHTS.THE BEST-PERFORMING CONFIGURATIONS ARE HIGHLIGHTED IN BOLD.
<table><tr><td>Mission</td><td colspan="2">Baseline</td><td colspan="2">+GeoConv</td><td colspan="2">+Hooking</td><td colspan="2">+Ensemble</td></tr><tr><td></td><td>MME↓</td><td>AP-1%↑</td><td>MME↓</td><td>AP-1% ↑</td><td>MME↓</td><td>AP-1% ↑</td><td>MME↓</td><td>AP-1% ↑</td></tr><tr><td>Perito Moreno 1a</td><td>42.9</td><td>49.6</td><td>41.7</td><td>50.0</td><td>34.9</td><td>50.8</td><td>27.6</td><td>55.7</td></tr><tr><td>Perito Moreno 1b</td><td>37.4</td><td>51.4</td><td>26.1</td><td>51.2</td><td>20.5</td><td>53.3</td><td>11.6</td><td>60.8</td></tr><tr><td>Perito Moreno 2a</td><td>54.2</td><td>47.4</td><td>42.9</td><td>48.5</td><td>36.1</td><td>50.5</td><td>23.5</td><td>55.6</td></tr><tr><td>Perito Moreno 2b</td><td>30.9</td><td>45.1</td><td>26.5</td><td>45.6</td><td>23.3</td><td>48.4</td><td>20.5</td><td>49.9</td></tr><tr><td>Perito Moreno 3</td><td>36.5</td><td>50.2</td><td>24.5</td><td>53.3</td><td>26.2</td><td>52.9</td><td>13.0</td><td>55.6</td></tr></table>

https://tc.copernicus.org/articles/19/5337/2025/ 1, 2, 3, 5, 6, 7

[12] S. Dong, X. Tang, J. Guo, L. Fu, X. Chen, and B. Sun, “Eisnet: Extracting bedrock and internal layers from radiostratigraphy of ice sheets with machine learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–12, 2021. 1, 2, 5

[13] Y. Cai, F. Wan, S. Hu, and S. Lang, “Accurate prediction of ice surface and bottom boundary based on multi-scale feature fusion network,” Applied Intelligence, vol. 52, no. 14, pp. 16 370–16 381, 2022. 2

[14] H. Kamangir, M. Rahnemoonfar, D. Dobbs, J. Paden, and G. Fox, “Deep hybrid wavelet network for ice boundary detection in radra imagery,” in IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2018, pp. 3449–3452. 2

[15] M. Liu-Schiaffini, G. Ng, C. Grima, and D. Young, “Ice thickness from deep learning and conditional random fields: application to ice-penetrating radar data with radiometric validation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022. 2, 5

[16] A. Jebeli et al., “Tssa: two-step semi-supervised annotation for radargrams on the greenland ice sheet,” in IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2023, pp. 56–59. 2

[17] C. M. Gifford, G. Finyom, M. Jefferson, M. Reid, E. L. Akers, and A. Agah, “Automated polar ice thickness estimation from radar imagery,” IEEE Transactions on Image Processing, vol. 19, no. 9, pp. 2456–2469, 2010. 2, 5

[18] J. E. Mitchell, D. J. Crandall, G. C. Fox, M. Rahnemoon-

far, and J. D. Paden, “A semi-automatic approach for estimating bedrock and surface layers from multichannel coherent radar depth sounder imagery,” in Image and Signal Processing for Remote Sensing XIX, L. Bruzzone, Ed., vol. 8892. SPIE, 2013, p. 88921E. 2, 5

[19] M. Rahnemoonfar, A. A. Habashi, J. Paden, and G. C. Fox, “Automatic ice thickness estimation in radar imagery based on charged particles concept,” in 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2017, pp. 3743–3746. 2

[20] M. Rahnemoonfar, G. C. Fox, M. Yari, and J. Paden, “Automatic ice surface and bottom boundaries estimation in radar imagery based on level-set approach,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 9, pp. 5115–5122, 2017. 2, 5

[21] D. J. Crandall, G. C. Fox, and J. D. Paden, “Layer-finding in radar echograms using probabilistic graphical models,” in Proceedings of the 21st International Conference on Pattern Recognition (ICPR2012), 2012, pp. 1530–1533. 2, 5

[22] V. Berger, M. Xu, S. Chu, D. Crandall, J. Paden, and G. C. Fox, “Automated tracking of 2d and 3d ice radar imagery using viterbi and trw-s,” in IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018, pp. 4162–4165. 2, 5

[23] M. Xu, D. J. Crandall, G. C. Fox, and J. D. Paden, “Automatic estimation of ice bottom surfaces from radar imagery,” in 2017 IEEE International Conference on Image Processing (ICIP), 2017, pp. 340–344. 2, 5

[24] S. Lee, J. Mitchell, D. J. Crandall, and G. C. Fox, “Estimating bedrock and surface layer boundaries and

confidence intervals in ice sheet radar imagery using mcmc,” in 2014 IEEE International Conference on Image Processing (ICIP), 2014, pp. 111–115. 2, 5

[25] A.-M. Ilisei and L. Bruzzone, “A model-based technique for the automatic detection of earth continental ice subsurface targets in radar sounder data,” IEEE GEOSCI REMOTE S, vol. 11, no. 11, pp. 1911–1915, 2014. 2

[26] ——, “A system for the automatic classification of ice sheet subsurface targets in radar sounder data,” IEEE T GEOSCI REMOTE, vol. 53, no. 6, pp. 3260–3277, 2015. 2

[27] E. Donini, F. Bovolo, and L. Bruzzone, “A deep learning architecture for semantic segmentation of radar sounder data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2021. 2

[28] M. H. Garc´ıa, E. Donini, and F. Bovolo, “Automatic segmentation of ice shelves with deep learning,” in 2021 IEEE International geoscience and remote sensing symposium IGARSS. IEEE, 2021, pp. 4833–4836. 2

[29] R. Ghosh and F. Bovolo, “A hybrid cnn-transformer architecture for semantic segmentation of radar sounder data,” in IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2022, pp. 1320–1323. 2

[30] ——, “Transsounder: A hybrid transunet-transfuse architectural framework for semantic segmentation of radar sounder data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2022. 2

[31] ——, “An enhanced unsupervised feature learning framework for radar sounder signal segmentation,” in IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2023, pp. 6920–6923. 2

[32] ——, “An unsupervised framework for radar sounder signal segmentation based on enhanced self-supervised transformers,” Authorea Preprints, 2023. 2

[33] ——, “Urs: an unsupervised radargram segmentation network based on self-supervised vit with contrastive feature learning framework,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 15 512–15 524, 2024. 2

[34] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241. 2

[35] Y. Cai, S. Hu, S. Lang, Y. Guo, and J. Liu, “End-to-end classification network for ice sheet subsurface targets in radar imagery,” Applied Sciences, vol. 10, no. 7, p. 2501, 2020. 2

[36] M. van Rijthoven, M. Balkenhol, K. Silin¸a, J. van der Laak, and F. Ciompi, “Hooknet: Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images,” Medical Image Analysis, vol. 68, p. 101890, 2021. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S1361841520302541 2, 4

[37] E. Schnider et al., “Improved distinct bone segmentation in upper-body ct through multi-resolution networks,”

International Journal of Computer Assisted Radiology and Surgery, vol. 18, no. 11, pp. 2091–2099, 2023. 2

[38] F. Wu et al., “Amd-hooknet for glacier front segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–12, 2023. 2

[39] M. van Rijthoven et al., “Multi-resolution deep learning characterizes tertiary lymphoid structures and their prognostic relevance in solid tumors,” Communications Medicine, vol. 4, no. 1, p. 5, 2024. 2

[40] F. Wu et al., “Lightweight cross-attention-based hooknet for historical handwritten document layout analysis: F. wu et al.” International Journal on Document Analysis and Recognition (IJDAR), vol. 28, no. 3, pp. 409–427, 2025. 2

[41] ——, “Contextual hookformer for glacier calving front segmentation,” IEEE Transactions on Geoscience and Remote Sensing, 2024. 2

[42] ——, “Amd-hooknet++: Evolution of amd-hooknet with hybrid cnn–transformer feature enhancement for glacier calving front segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–22, 2026. 2

[43] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 4, pp. 834–848, 2017. 2

[44] W. Wu, Z. Qi, and L. Fuxin, “Pointconv: Deep convolutional networks on 3d point clouds,” in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2019, pp. 9621–9630. 4

[45] E. S. Mannerfelt. The svalbard radar interpretation experiment. [Online]. Available: https://radar.mannerfelt. org/howto 5

[46] M. Koch et al., “Bedrock topography and ice thickness distribution of three major patagonian outlet glaciers unveiled by helicopter-borne ground penetrating radar,” Earth System Science Data Discussions, vol. 2026, pp. 1–29, 2026. [Online]. Available: https: //essd.copernicus.org/preprints/essd-2025-678/ 6

[47] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017. 7