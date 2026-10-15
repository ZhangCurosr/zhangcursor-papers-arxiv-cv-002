# Mapping Arctic Polynyas from Sea Ice Concentration Data Using Hierarchical Partitioning and Differentiated Region Growing: Sensitivity to Threshold and Product Choices

Tianyu Zhang <sup>1,2</sup>, Haiyi Ren <sup>3\*</sup>, Zhilun Zhang <sup>4,5</sup>, Yang Hong <sup>1,2</sup>, Chunlei Mi <sup>6</sup> , Xiao Cheng <sup>4,5</sup>, Fengming Hui <sup>4,5\*</sup>

<sup>1</sup> National Key Laboratory of Space Target Awareness,Beijing 101416,China.

<sup>2</sup> School of Information Science and Engineering, Space Engineering University, Beijing 101416, China.

<sup>3</sup> The State Key Laboratory of Space Information System and Integrated Application,

Beijing Institute of Satellite Information Engineering, Beijing 100086, China.

<sup>4</sup> School of Geospatial Engineering and Science, Sun Yat-sen University, Southern

Marine Science and Engineering Guangdong Laboratory (Zhuhai), Zhuhai 519082,

China.

<sup>5</sup> Key Laboratory of Comprehensive Observation of Polar Environment (Sun Yat-sen

University), Ministry of Education, Zhuhai 519082, China.

<sup>6</sup> Beijing Automotive Technology Center Co., Ltd., Beijing 101300, China

Corresponding author: Fengming Hui (huifm@mail.sysu.edu.cn) ; Haiyi Ren

(haiyiren@mail.bnu.edu.cn )

## Abstract

Accurately delineating Arctic coastal polynyas from sea ice concentration data remains challenging because of complex coastal geometry and spatially heterogeneous ice conditions. We develop a daily sea ice concentration (SIC) mapping framework based on hierarchical partitioning, connected-component analysis, and differentiated region growing for polynya and open-water regions. Applied to September-May ice seasons from 2002/03 to 2024/25, the framework achieves a mean overall accuracy of 96.74% and a mean kappa coefficient of 0.93 in the 63-scene visual assessment. Sensitivity experiments reveal a consistent pattern: the absolute magnitude of SICdefined polynya extent is strongly configuration-dependent, whereas major temporal features are comparatively more stable. Relative to a 75% upper SIC threshold, seasonal extent integrals differ by 13.48%, 32.47%, and 65.04% at thresholds of 80%, 85%, and 90%, respectively, while the broad interannual structure remains recognizable. The response is also strongly regional and seasonal, with the largest threshold sensitivity in the Central Arctic Ocean, very weak sensitivity in the Gulf of St. Lawrence, and greater pan-Arctic curve separation from January onward. At an 80% threshold, productcomparison MAPE is 68.19% between the 6.25 and 25 km products and 11.47% between the 12.5 and 25 km products under their respective comparison periods and reference definitions. Regional product disagreement is likewise heterogeneous. Together, these results show that reproducible long-term polynya monitoring requires a fixed and explicitly documented combination of SIC product, threshold, and spatial classification rules, while broad temporal patterns can remain informative even when absolute extent differs substantially.

Keywords: Arctic polynya; sea ice concentration; connected components; region growing; threshold sensitivity; passive microwave remote sensing.

## 1 Introduction

Arctic polynyas expose open water or thin ice within the surrounding ice cover. During freezing conditions, repeated ice formation in these openings contributes to oceanic heat loss and brine release. Satellite estimates of active frazil and ice production demonstrate their importance across Northern Hemisphere coastal seas[1]. Understanding these processes requires observations of where openings occur, how their boundaries evolve, and how long they persist. Long satellite records now support analyses of Antarctic polynya trends and interdecadal variability[2]. Polynya detection also supports assessments of Southern Ocean and Arctic climate simulations[3], [4]. These applications make a reproducible spatial definition essential: changes in a mapped boundary must have a clear relationship to the detection procedure that generates it.

Passive microwave sea ice concentration (SIC) provides frequent, large-area observations for long-term mapping, including during polar darkness and cloudy conditions. However, coarse sampling mixes ice and water near narrow openings and their boundaries, as recent work north of Greenland demonstrates[5]. Comparisons against Landsat observations also reveal differences among SIC products across the concentration range[6]. A low SIC value alone cannot distinguish a polynya from the marginal ice zone (MIZ) or the open ocean. These features can occupy overlapping concentration ranges. Their separation therefore requires information about the surrounding ice, coastline, and connection to extensive open water[7]. Notably, a mapped SIC boundary identifies a low-concentration feature and does not directly measure ice thickness or production.

Recent improvements in spatial detail broaden the available mapping tools. Synthetic aperture radar (SAR) segmentation of Sentinel-1 imagery maps Arctic ice and water at scales of a few hundred meters[8]. In the western Ross Sea, SAR-based delineation resolves narrow polynyas that coarser passive microwave observations can miss[9]. Super-resolution processing also refines the representation of polynya boundaries and leads north of Greenland[5]. Nevertheless, fine sampling does not remove all classification ambiguity. Deep-learning detection of small landfast-ice polynyas near Canadian communities still struggles when water and surrounding ice have weak backscatter contrasts[10]. Thus, spatial detail and reliable feature discrimination remain related but distinct requirements.

Multisensor methods address some of these ambiguities by combining complementary observations. Fusion of SAR and passive microwave measurements supports pan-Arctic SIC retrieval[11]. Multitask networks can also estimate concentration, ice development stage, and floe size within one framework[12]. Their performance depends on more than network architecture. Controlled input comparisons show that microwave channels and ancillary time, location, and weather information affect sea-ice mapping accuracy[13]. These developments improve the observational basis for detection. They also reinforce the need to distinguish the quality of an input SIC field from the subsequent rules that classify a feature as a polynya.

Automatic polynya methods now explicitly address spatial context and temporal continuity. Region growing already separates ocean-connected water and the MIZ in Northern Hemisphere frazil mapping[1]. Daily object tracking and persistence criteria support the Daily Edge of Each Polynya in Antarctica boundary record[14]. For Arctic SIC imagery, a hybrid U-Net and Gaussian mixture framework identifies candidate polynyas and removes ocean and MIZ false positives[7]. A geomorphon approach instead interprets Antarctic SIC patterns as morphological depressions[15]. These methods demonstrate several viable ways to move beyond a concentration cutoff. Their different treatment of connectivity, shape, and persistence also makes the operational definition part of the resulting polynya record.

A central remaining issue is how much of the derived record is stable under plausible processing choices. Raising the upper SIC threshold necessarily enlarges the concentration range available to the classifier, but in a connectivity-based method it can also change whether neighboring components remain isolated or become connected. Threshold sensitivity is therefore not only an area-expansion problem; it can alter object topology and do so differently among regions and seasons. Recent polynya studies likewise report strong threshold dependence and seasonally varying divergence among detected extents[15], [16]. This motivates testing both the absolute magnitude of mapped extent and the persistence of its temporal structure across threshold settings, and whether this sensitivity varies among regions and seasons.

Input-product sensitivity poses a complementary question. Passive-microwave SIC products differ in retrieval construction, weather filtering, spatial sampling, and their response at intermediate concentrations[6], [17], [18]. Applying the same classifier to several products therefore tests whether a mapping conclusion depends on the observational input as well as on the threshold. Recent Antarctic studies show that threshold, grid, season, and product choice can substantially affect detected polynya statistics[16]. Kern et al.[19] further showed systematic differences between PSSMderived polynya classes and higher-resolution AMSR-E 89-GHz SIC estimates in the Ross Sea and off the Adélie Coast. Examining threshold sensitivity within one product and product sensitivity at a fixed threshold provides two distinct views of the robustness of the resulting Arctic record.

Against this background, the present study links an explicit daily mapping procedure with a systematic sensitivity analysis across Arctic regions and ice seasons. We ask three questions. First, can hierarchical partitioning and differentiated region growing separate enclosed low-SIC openings from ocean-connected MIZ and open water with high classification agreement? Second, how strongly does the mapped extent change when the upper SIC threshold is varied while the input product is fixed? Third, how much does the result change when the SIC product is changed at a fixed threshold? The emphasis is on what remains stable and what changes across configurations, so that the resulting record can be interpreted and reproduced consistently.

The workflow first partitions SIC into high- and low-concentration domains and labels connected components. It then identifies open-water and candidate polynya seeds within the low-concentration domain. Differentiated region growing expands these seeds under separate neighborhood rules, while geographic exclusions address known coastal and inland-water artifacts. This hierarchy uses concentration to identify candidate pixels and spatial context to determine their final class. A fixed 15% seed threshold initializes the regions; a separate upper threshold controls the concentration range available to the extraction procedure.

We apply the framework to September–May seasons from 2002/03 to 2024/25 using the 6.25 km ARTIST Sea Ice (ASI) record. The unavailable 2011/12 season is excluded. Visual assessment covers 63 scenes and the polynya and MIZ classes. The threshold experiment compares upper limits of 75%, 80%, 85%, and 90% within the same SIC product. A second experiment applies an 80% upper threshold to three products: 6.25 km ASI, 12.5 km enhanced NASA Team (NT2), and the 25 km climate data record. Comparisons use the respective periods of data availability. Regional and interannual comparisons then examine whether similar temporal behavior accompanies differences in absolute extent. This analysis provides a practical basis for selecting and reporting configurations in subsequent Arctic polynya studies.

## 2 Data and study area

## 2.1 Sea ice concentration products

Daily SIC fields are taken from the four product entries in Table 1. The 6.25 km ASI series combines AMSR-E and AMSR2 observations[20], [21]. It provides the primary input for method development, threshold comparisons and product comparison. The 12.5 km input is the NT2 SIC field from AU\_SI12 Version 1[22]. The 25 km series uses the NOAA/NSIDC climate data record G02202 Version 5[23]. The near-real-time record G10016 Version 4 extends this series through January–May 2025[24]. These product versions are used throughout the analysis.

Table 1 SIC products used in the analysis.
<table><tr><td>Product and version</td><td>Grid spacing</td><td>Retrieval</td><td>Period used</td><td>Role</td></tr><tr><td>Bremen n6250 ASI v5.4[20], [21]</td><td>6.25 km</td><td>ASI</td><td>Sep. 2002–May 2025; 2011/12 omitted</td><td>Primary mapping, threshold tests and product comparison</td></tr><tr><td>AU_SI12 v1[22]</td><td>12.5 km</td><td>NT2</td><td>Sep. 2012–May 2025</td><td>Product comparison</td></tr><tr><td>G02202 v5[23]</td><td rowspan="2">25 km</td><td>CDR combining NT and Bootstrap</td><td>Sep. 2002–Dec. 2024</td><td rowspan="2">Product comparison</td></tr><tr><td>G10016 v4[24]</td><td>Near-real-time CDR</td><td>Jan.–May 2025</td></tr></table>

Notes: Only September–May observations are analyzed. Dates identify the subsets used, rather than the complete catalog availability of each product. CDR: climate data record.

ASI is based on high-frequency microwave polarization information and includes weather filtering[21]. The NOAA/NSIDC climate data record combines NASA Team and Bootstrap retrievals[23], [24], whereas AU\_SI12 provides enhanced NASA Team, or NT2, SIC[22]. Thus, the input products differ in retrieval construction as well as nominal grid spacing. Here, 6.25, 12.5, and 25 km refer to grid spacing; they should not be interpreted as identical measures of the effective radiometric footprint.

## 2.2 Spatial domain and ice seasons

The analysis uses Northern Hemisphere polar stereographic grids with a latitude of true scale at 70°N[25]. Regional statistics use the NSIDC mask distributed by the Centre for Polar Observation and Modelling[26], [27] and shown in Figure 1. The analysis considers an aggregate pan-Arctic series together with fourteen regional masks: Sea of Okhotsk, Bering Sea, Hudson Bay, Gulf of St. Lawrence, Baffin Bay, Greenland Sea, Barents Sea, Kara Sea, Laptev Sea, East Siberian Sea, Chukchi Sea, Beaufort Sea, Canadian Arctic Archipelago, and Central Arctic Ocean. Notably, as shown in Figure1, the pan-Arctic domain includes the fourteen named regional masks and additional ocean-grid areas in pale blue. Consequently, the pan-Arctic total is not restricted to the sum of the fourteen regional totals.

An ice season runs from September to May of the following year. The study spans 2002/03–2024/25 ice seasons, with the 2011/12 season unavailable in the primary ASI record. Short gaps within otherwise available records are treated as described in Section 3.1. The 2011/12 gap is not reconstructed as a daily ASI observation series. The 12.5 km comparison begins in 2012/13, following the AMSR2 record selected for this analysis.

![](images/1d0347ba02ccfbd1a1c96e4fb268d57252dae1794482f2607c0470dfa331a288.jpg)  
Figure 1. Regional mask used for the 6.25 km analysis, displayed from the existing categorical mask. Numbers identify the fourteen regional labels listed in the legend; gray denotes land or excluded cells and pale blue denotes other ocean-grid areas. The regional-mask source is [26].

## 3 Polynya extent extraction method

The mapping target is an SIC-defined low-concentration region that is separated from extensive open-ocean water by higher-concentration ice or land. The MIZ is the ocean-connected, nonzero-SIC portion of the low-concentration domain. This operational distinction depends on spatial connectivity, rather than SIC alone. Pixels above the upper threshold are referred to as high-concentration ice, since concentration does not determine ice thickness. Figure 2 summarizes the workflow, and Figure 3 illustrates the partition and classification stages.

## 3.1 Data preprocessing

Missing and extensively corrupted SIC scenes are first identified by visual inspection. Short gaps are filled using adjacent valid fields. For a missing date nearest to one end of a gap, the interpolation used in the workflow is expressed as:

$$
C _ { f i l l } = \frac { N } { N + 1 } C _ { n e a r } + \frac { 1 } { N + 1 } C _ { f a r }\tag{1}
$$

where N is the number of consecutive missing or invalid days, $C _ { n e a r }$ is the nearer valid SIC field, and $C _ { f a r }$ is the farther valid field. For gaps spanning three or more days, reconstruction proceeds from the available outer dates toward the gap center. This procedure supplies a continuous input for short interruptions, but does not add independent information about opening or closing events during the missing dates.

Small patches of invalid coastal data were removed using manually delineated masks in the North Pacific and North Atlantic. These masks cover the Gulf of Alaska, southern Bering Sea, southern Sea of Okhotsk, and waters from eastern Newfoundland to the southern United Kingdom. Non-land pixels within these zones were assigned to seawater to prevent spurious polynya detection. The Bohai Sea was also masked and assigned to seawater during preprocessing.

Inland-water masks were constructed using the land surroundings of candidate water bodies. Neighboring land pixels were checked along their boundaries to identify enclosed lakes and rivers. Estuaries and rivers connected to the ocean were delineated manually. The resulting mask excluded these inland waters from polynya extraction.

These geographic exclusions are held fixed throughout the experiments. They remove known coastal and inland-water artifacts before seed identification and ensure that threshold comparisons are made over a consistent operational domain. Their implementation is transferred to each product grid in the product-sensitivity experiment.

## 3.2 Hierarchical partitioning

The first partition divides valid ocean pixels into high-concentration ice, with SIC greater than the upper threshold TH, and a low-concentration domain with $0 \leq \mathrm { S I C } \leq$ TH. The reference configuration uses $\mathrm { T H } = 8 0 \% ;$ the threshold experiment changes only this upper value to 75%, 80%, 85%, or 90%. Four-neighbor connected-component labeling is applied to the low-concentration domain. Up to the two largest connected components form the candidate ocean-connected domain, accommodating the major Atlantic and Pacific open-water regions when they are separate. The remaining components are retained as isolated low-concentration candidates.

Within the candidate ocean-connected domain, a fixed 15% threshold separates very-low-concentration pixels from the intermediate band, $1 5 \% < \mathrm { S I C } \leq \mathrm { T H }$ . Fourneighbor labeling is then applied to the very-low-concentration pixels. The largest components provide open-water seeds, while smaller detached components provide candidate polynya seeds. This lower threshold initializes the seed regions. It does not define the upper SIC limit of the final polynya mask and is held fixed in the sensitivity experiment.

Geographic exceptions prevent large, seasonally enclosed water bodies from being classified as polynyas solely because they are disconnected from the two largest components. The exceptions cover specified parts of the Baltic Sea, Sea of Okhotsk, northern Russian waters, and Hudson Bay. Components in these areas are assigned to open water when they exceed the reported area criterion of $1 5 0 , 0 0 0 \mathrm { k m } ^ { 2 }$ . This criterion excludes large water bodies; it does not impose a minimum polynya area.

![](images/32d015e3bc603d57f61daafca16b7a38687f387f77ea10da3cccbdcdf69c7e8a.jpg)  
Figure 2. Workflow for Arctic polynya mapping: (a) Data preprocessing; (b) hierarchical SIC partitioning and seed identification; (c) sequential region growing and final classification. Connectors A, B, and C denote isolated low-concentration candidates, the intermediate-concentration band and seed regions, respectively. The reference upper SIC threshold is 80%, and the fixed seed threshold is 15%. The final SIC > 0% test applies only to ocean-connected mixture pixels.

## 3.3 Region growing and final classification

Boundary tracing identifies the initial growth pixels of the polynya and open-water seeds. Candidate polynya boundary pixels directly adjacent to open-water seeds are excluded from the initial polynya growth set. Polynya growth then evaluates neighboring pixels iteratively. An admissible candidate is neither land nor highconcentration ice, has not already been assigned to an excluded class, and has SIC no lower than that of its parent growth pixel. Its surrounding 5 × 5 neighborhood must contain no open-water seed pixel. New accepted pixels provide the boundary for the next iteration, and growth stops when no additional pixels are accepted.

Growth evaluates the eight cells surrounding a pixel in a $3 \times 3$ window. A wider separation check evaluates the twenty-four surrounding cells in a $5 \times 5$ window. The wider check limits expansion toward open-water seeds, while the eight-neighbor rule permits diagonal growth. Together, the two neighborhood scales allow local boundary growth while preserving separation from the larger ocean-connected domain.

Open-water growth follows polynya growth. It expands into eligible lowconcentration neighbors while excluding land, high-concentration ice, and pixels already assigned to polynyas. Neighborhoods containing classified polynya pixels are protected from this expansion. The resulting ocean-connected region is provisionally an ice–water mixture. Remaining eligible pixels adjacent to that region are assigned to the mixture; the other residual low-concentration pixels, including isolated candidates from the first partition, are assigned to polynyas. Finally, mixture pixels with $\mathrm { S I C } > 0$ are labeled MIZ, and those with SIC = 0 are labeled open water. Zero-SIC pixels within a polynya retain their polynya label.

(a) Sea Ice Concentration (%)  
![](images/1efc8a23cfd6a190e87ab31bc75672ddedf57476aa352b8435082d2607548aba.jpg)

(b) Sectors  
![](images/c3540ddd1fb0d914c830f1ce9eb689704a7523bca3d8f247aaba679549c71477.jpg)  
(e) Sectors

(c) Segmentation Result  
![](images/2cb6433376846f1c137a3eda15c8f2db88e1cb3debaa2bf89cd3c1442741efbc.jpg)

![](images/6e7523cff6ddffae1b3e664c6e64111097f50069d9bfe2f81e5a3103addbf37b.jpg)

![](images/df09faa605b79a779965f49206d5802ab6a147c94a1d65b0d5cff84284554431.jpg)

![](images/e1c8eef0f885df723914eafe60506ee9e3db76d83d6744baf3845f99454144b2.jpg)

Figure 3. Examples of hierarchical partitioning and final classification on January 25, 2004 (a–c), and February 10, 2024 (d–f). Panels (a, d) show SIC, (b, e) show intermediate partitions, and (c, f) show the final classes. In the partition legends, $S I C _ { H }$ denotes high-concentration ice, $S I C _ { r }$ denotes the detached candidates, $S I C _ { A 2 \_ r }$ denotes the intermediate-concentration band, $S I C _ { A 2 \_ l o w \_ o w }$ and $S I C _ { A 2 \_ l o w \_ r }$ denote the open-water and polynya seed regions, respectively.

## 3.4 Extent and comparison metrics

Daily polynya extent is the sum of the nominal areas of pixels classified as polynya, including both their open-water and ice-covered fractions:

$$
\mathrm { E ( t ) } = \alpha N _ { t }\tag{2}
$$

where $N _ { t }$ is the number of polynya pixels on day t and α is nominal pixel area, calculated as the square of grid spacing in the existing analysis. This is a classified extent, not an open-water area weighted by 1 − SIC. Summing daily extents over an ice season gives a seasonal extent integral:

$$
I _ { s } = \sum _ { t = 1 } ^ { D } E ( t ) \Delta t\tag{3}
$$

Here D counts analyzed days and $\Delta \ t = 1$ day, giving I in km²·day. An integral weights both the size and persistence of mapped openings; it is not the geographical area of a unique footprint. Multi-season daily averages are used to describe how differences evolve through September–May. Month labels on these plots indicate position within the season and do not imply that each curve consists of nine monthly means.

For paired seasonal integrals X and Y, we calculated the mean absolute percentage error (MAPE):

$$
M A P E = \frac { 1 0 0 \% } { K } \sum _ { k = 1 } ^ { K } \bigg | \frac { Y _ { k } - X _ { k } } { X _ { k } } \bigg |\tag{4}
$$

X is the stated reference series, Y is the comparison series, and K is the number of valid paired seasons. MAPE is used as a difference measure between two products, without treating either product as ground truth. The threshold calculations use the lower-threshold series as the reference. For the product comparisons, the 6.25–25 km value uses 25 km as reference, whereas the 12.5–25 km value uses 12.5 km as reference. Different denominators and periods are retained explicitly rather than treated as interchangeable comparisons.

## 3.5 Visual assessment of the classification

The assessment uses stratified random sampling within the mapped polynya and MIZ classes, followed by visual interpretation. Stratification permits different sampling effort among classes, but the reference labels, inclusion probabilities, and estimation procedure must be specified for a population-level accuracy statement[28], [29]. The reported sampling setup assumes expected user’s accuracies of 0.90 for polynyas and 0.85 for the MIZ, with a sampling-control parameter V = 0.01.

The visual assessment concerns the 6.25 km record reported at the 80% threshold. Seven scenes were selected from each September–May calendar month, giving 63 scenes from a stated total of 6,010. Table 2 presents monthly confusion counts and user’s accuracy (UA), producer’s accuracy (PA), overall accuracy (OA), and kappa. The mean OA is 96.74%, and mean kappa is 0.93. Mean UA/PA is 97.32%/95.94% for polynyas and 96.08%/97.49% for the MIZ.

The monthly confusion matrices reveal a consistent asymmetry that is more informative than the mean accuracy alone. Polynya producer’s accuracy decreases to 92.22% in December and 93.15% in April, while the corresponding user’s accuracies remain 99.07% and 97.44%. Under the table’s class convention, these months therefore contain more omission than commission of polynya pixels, indicating that difficult cases tend to be absorbed into the competing MIZ class rather than generated as false polynyas. This pattern is consistent with the known difficulty of passive-microwave SIC retrieval and class separation at intermediate concentrations and mixed pixels[6],[17]. The high overall scores nevertheless show that the connectivity-based workflow maintains strong class agreement across the sampled scenes.

Table 2. Confusion Matrix for Polynya Extent Extraction.
<table><tr><td rowspan="2">Month</td><td rowspan="2">Categories</td><td colspan="2">Pixel Count</td><td rowspan="2">User's</td><td rowspan="2">Producer's</td><td rowspan="2">Overall</td><td rowspan="2">Kappa Coefficient</td></tr><tr><td>Polynya</td><td>MIZ</td></tr><tr><td>Sep.</td><td>Polynya</td><td>279</td><td>6</td><td>Accuracy (%) 98.24</td><td>Accuracy (%) 97.89</td><td>Accuracy (%) 98.57</td><td>0.97</td></tr><tr><td></td><td>MIZ</td><td>5</td><td>477</td><td>98.76</td><td>98.96</td><td></td><td></td></tr><tr><td>Oct.</td><td>Polynya</td><td>272</td><td>8</td><td>98.55</td><td>97.14</td><td>98.45</td><td>0.97</td></tr><tr><td rowspan="2">Nov.</td><td>MIZ</td><td>4</td><td>491</td><td>98.40</td><td>99.19</td><td></td><td></td></tr><tr><td>Polynya MIZ</td><td>350 16</td><td>9 396</td><td>95.63</td><td>97.49 96.12</td><td>96.76</td><td>0.93</td></tr><tr><td rowspan="2">Dec.</td><td>Polynya</td><td>320</td><td>27</td><td>97.78 99.07</td><td>92.22</td><td></td><td></td></tr><tr><td>MIZ</td><td>3</td><td>409</td><td>93.81</td><td>99.27</td><td>96.05</td><td>0.92</td></tr><tr><td rowspan="2">Jan.</td><td>Polynya</td><td>356</td><td>17</td><td>96.22</td><td>95.44</td><td>95.97</td><td>0.92</td></tr><tr><td>MIZ</td><td>14</td><td>382</td><td>95.74</td><td>96.46</td><td></td><td></td></tr><tr><td rowspan="2">Feb.</td><td>Polynya</td><td>377</td><td>12</td><td>94.49</td><td>96.92</td><td>95.57</td><td>0.91</td></tr><tr><td>MIZ</td><td>22</td><td>356</td><td>96.74</td><td>94.18</td><td></td><td></td></tr><tr><td rowspan="2">Mar.</td><td>Polynya</td><td>356</td><td>13</td><td>96.74</td><td>96.48</td><td>96.68</td><td>0.93</td></tr><tr><td>MIZ</td><td>12</td><td>372</td><td>96.62</td><td>96.88</td><td></td><td></td></tr><tr><td rowspan="2">Apr.</td><td>Polynya</td><td>381</td><td>28</td><td>97.44</td><td>93.15</td><td>94.96</td><td>0.90</td></tr><tr><td>MIZ</td><td>10</td><td>335</td><td>92.29</td><td>97.10</td><td></td><td></td></tr><tr><td rowspan="2">May</td><td>Polynya</td><td>439</td><td>15</td><td>99.55</td><td>96.70</td><td>97.64</td><td>0.95</td></tr><tr><td>MIZ</td><td>2</td><td>264</td><td>94.62</td><td>99.25</td><td></td><td></td></tr><tr><td>Avg. MIZ</td><td>Polynya</td><td></td><td></td><td>97.32 96.08</td><td>95.94 97.49</td><td>96.74</td><td>0.93</td></tr></table>

## 4 Sensitivity analysis

## 4.1 Sensitivity to the upper SIC threshold

The threshold experiment applies TH = 75%, 80%, 85%, and 90% to the same 6.25 km ASI input for the 2002/03–2024/25 ice seasons. Figure 4 compares seasonal integrals for the aggregate series and the fourteen regional masks. The experiment therefore isolates the effect of changing the upper SIC threshold within one observational product.

Higher thresholds produce progressively larger seasonal integrals. Relative to 75%, the aggregate differences are 13.48%, 32.47%, and 65.04% at 80%, 85%, and 90%, respectively, and the reported mean absolute differences span approximately 2.49– $1 2 . 0 2 \times 1 0 ^ { 7 } { \mathrm { k m } } ^ { 2 }$ ·day. Across the tested range, the reported mean extent response is about 4.30% per one-percentage-point increase in threshold. More importantly, the widening separation among curves is not captured by one universal multiplicative correction: the effect of admitting higher-SIC pixels accumulates differently among seasons and regions.

Despite the large magnitude differences, the broad interannual structure in Figure 4 remains recognizable across all four thresholds: many relative highs, lows, and multiyear changes occur in similar portions of the record. The key result is therefore a separation between magnitude sensitivity and temporal-pattern stability. Threshold selection strongly affects how much polynya extent is mapped, but it does not erase the principal temporal organization of the record. Similar threshold studies have reported large changes in absolute polynya area while retaining common stages of seasonal development[15], [16]. This relative stability is valuable for long-term comparison, provided that absolute values from different threshold configurations are not treated as interchangeable.

![](images/3c18e7a650c76106198fc5ca39564ccfaad40fca01da273875a22fbd4af74bda.jpg)  
Figure 4. Yearly Arctic Polynya Extent Variation (2002/03-2024/25) based on different thresholds of SIC data (6.25 km). The missing 2011/12 season is not plotted.

Regional threshold sensitivity is highly uneven (Figure 5). The Central Arctic Ocean shows the largest relative response, while the Gulf of St. Lawrence shows very little response. The Kara, East Siberian, Greenland, Beaufort, and Laptev seas also show comparatively large responses, whereas the Okhotsk Sea, Canadian Arctic Archipelago, Bering Sea, Barents Sea, Baffin Bay, Hudson Bay, and Chukchi Sea are less sensitive in the reported comparison. Several regions show their strongest response in the 85%–90% interval. The sensitivity of SIC-defined extent is therefore geographically structured rather than a uniform Arctic-wide percentage uncertainty.

The regional pattern is consistent with the topology of a connectivity-based classifier. Raising TH not only admits additional boundary pixels; it can also join previously separated low-concentration patches or change their connection to the ocean-connected domain. Regions with many pixels near the upper threshold or with fragmented low-SIC geometry can therefore respond more strongly than regions whose mapped openings are sharply bounded. Morphological and geomorphon-based polynya studies similarly show that threshold and spatial context jointly control mapped boundaries[15], [30]. Figure 5 thus reveals a regional mapping sensitivity that arises from both the SIC distribution and the spatial organization of candidate pixels\`\`111.

![](images/30f55548d391acb23c1d06b407c3806622888084bebd284cdd08a83777731afd.jpg)  
Figure 5. Regional relative differences between successive SIC-threshold settings: 75%–80%, 80%–85%, and 85%–90%. Each pair uses its lower threshold as the denominator.

Threshold sensitivity also has a clear seasonal structure (Figure 6). Aggregate separation among the four curves is comparatively small from September to December and becomes much more apparent from January onward. The threshold choice therefore affects late-winter and spring amplitude more strongly than early-season amplitude in the aggregate record. Comparable threshold-based studies have found that the spread among mapped polynya areas varies with the stage of polynya development and season[15], [16]. In the present record, this seasonal dependence reinforces the regional result from Figure 5: threshold uncertainty is not a single constant attached to the entire time series, but varies with both location and time of year.

![](images/4867135d004ac38006a0c513bbc611e7857104feb4578d6cc7c1860576b1801b.jpg)  
Figure 6. Monthly averaged polynya extent over multiple years derived using different sea ice concentration thresholds.

## 4.2 Sensitivity to SIC product selection

The product experiment fixes the upper SIC threshold at 80% and applies the same classification logic to the 6.25 km ASI, 12.5 km NT2, and 25 km NOAA/NSIDC inputs. Figure 7 compares their seasonal integrals over the available periods. Because the products differ in retrieval construction as well as grid spacing, this experiment measures practical product sensitivity rather than a controlled resolution-only effect. That distinction is retained while the results are interpreted in terms of the reproducibility of the mapped record.

Product choice changes the magnitude of the aggregate record substantially. The MAPE is 11.47% for the 12.5–25 km comparison over 2012/13–2024/25 and 68.19% for the 6.25–25 km comparison over the valid 2002/03–2024/25 pairs, excluding 2011/12. The 6.25 km series is generally larger in Figure 7. This direction is consistent with finer gridding retaining more small and fragmented openings, a behavior also reported in polynya sensitivity studies[5], [16], [19], while differences among the underlying SIC retrieval algorithms can independently alter concentration fields[6], [17], [18]. The large 6.25–25 km disagreement should therefore be interpreted as a combined product effect, not as grid spacing alone.

Regional product sensitivity is as heterogeneous as threshold sensitivity. Average product-comparison MAPEs are below 50% for Hudson Bay, the Kara Sea, and the Canadian Arctic Archipelago, but exceed 100% for the Greenland, Okhotsk, and Bering seas. This spread rules out an Arctic-wide rescaling that could convert one product’s polynya extent into another’s. It also agrees with broader SIC evaluations showing that algorithm differences vary across concentration regimes and surface conditions[6], [17], [18]. For coastal polynyas, where narrow or fragmented openings are common, spatial sampling and mixed pixels add another source of divergence[1], [19], [30]. The product experiment therefore identifies geography as a central dimension of uncertainty in mapped extent.

![](images/8e61f15b13fec7f41d1d3df183c649ec7b7b13b2236c892be5cd6606b171da13.jpg)  
Figure 7. Interannual Arctic Polynya Extents (2002/03-2024/25) Derived from SIC Data at Different Resolutions.

Figure 8 adds a temporal dimension to the product comparison. Differences in the Okhotsk and Bering seas, Hudson Bay, and Baffin Bay are especially apparent from January through April. Yet several panels retain similar timing of major seasonal peaks even when their amplitudes differ strongly. Persistent amplitude offsets then accumulate into the seasonal integrals in Figure 7. As in the threshold experiment, the broad timing of activity can therefore be more stable than its absolute magnitude. This distinction helps explain why different SIC products may support a similar qualitative seasonal narrative while producing substantially different quantitative extent estimates.

Taken together, Figures 7 and 8 show that product selection affects both the magnitude and spatial distribution of mapped extent, with the strongest discrepancies concentrated in particular regions and parts of the season. Previous SIC intercomparisons document algorithm-dependent behavior at intermediate concentrations and under changing surface conditions[6], [17], [18], [31], [32], while polynya-specific comparisons demonstrate additional effects of grid spacing and morphology[15], [16], [19]. The present results fit this broader measurement picture: no single factor explains every difference, but the resulting configuration sensitivity is large enough that product identity must be treated as part of the definition of a longterm polynya record.

![](images/f93f417750d5d91919b7f6807217299315e7cafecf90d27cb04feb8d736ae7d1.jpg)  
Figure 8. Multi-year Monthly Averaged Polynya Extents (2012/13-2024/25) Derived from SIC Data at Different Resolutions.

## 5 Discussion

## 5.1 Contribution of hierarchical partitioning and differentiated region growing

The methodological contribution of this study lies in combining hierarchical partitioning with differentiated region growing to resolve the spatial ambiguity between polynyas and the MIZ. Previous studies have used region-growing approaches to separate ocean-connected water from polynya-related features[1], while hybrid deeplearning methods have identified Arctic polynyas by subsequently removing ocean- and

MIZ-related false detections[7]. The present framework separates the classification problem into two linked decisions: concentration defines the pixels eligible for classification, whereas connectivity and neighborhood context determine whether those pixels ultimately belong to a polynya or to the ocean-connected domain. This structure accommodates both detached openings within the consolidated ice pack and smaller candidate openings embedded near the broader ocean-connected low-concentration domain.

A key feature of the hierarchy is that the two SIC thresholds serve different purposes. The upper threshold defines the concentration range available to the classifier, whereas the fixed lower threshold helps identify open-water and candidate-polynya seeds within the ocean-connected low-SIC domain. Connected-component analysis provides the larger-scale spatial context, after which differentiated growth resolves the boundary between competing classes. Polynya growth accepts neighboring pixels whose SIC is no lower than that of the current growth pixel while maintaining separation from open-water seeds; subsequent open-water growth preserves pixels already assigned to polynyas. The resulting classification therefore combines local SIC transitions with larger-scale ocean connectivity. Importantly, zero-SIC pixels can remain classified as part of a polynya when they belong to an enclosed opening, a distinction that cannot be obtained from concentration values alone.

The examples in Figure 3 illustrate how these successive decisions generate spatially coherent polynya, MIZ, and open-water classes. Across the 63 assessed scenes, the mean overall accuracy of 96.74% and mean kappa of 0.93 indicate strong agreement with visual interpretation for the sampled polynya and MIZ classes. Applying the same workflow throughout the available September–May record provides a common operational basis for comparing SIC-defined openings across contrasting Arctic environments. More importantly, the explicit separation of candidate identification, boundary growth, and final classification makes the operational definition traceable, while the sensitivity experiments quantify how the resulting extent changes with threshold and SIC-product choice. Together, these elements provide a reproducible framework for interpreting regional contrasts and temporal variability in SIC-defined Arctic polynya extent.

## 5.2 Interpretation of threshold and product sensitivity

The two sensitivity experiments reveal a common structure. Their comparison conditions are summarized in Table 3. Absolute extent varies much more strongly across configurations than the timing of major temporal features, while the magnitude of this sensitivity differs substantially among regions and through the ice season. For threshold choice, aggregate differences grow from 13.48% to 65.04% across the tested settings while Figures 4 and 6 retain recognizable temporal organization. For product choice, MAPEs range from 11.47% to 68.19% under the stated comparisons, with regional values spanning from below 50% to above 100% in the reported groups. The combined message is therefore not simply that parameters matter, but that they affect the magnitude, geography, and seasonality of the mapped record in different ways.

Table 3. Comparison conditions for interpreting threshold and SIC-product sensitivity.
<table><tr><td>Comparison</td><td>Quantity varied</td><td>Statistic (%)</td><td>Reference series</td><td>Valid seasons</td></tr><tr><td>80% versus 75%</td><td>Upper SIC threshold</td><td>MAPE13.48</td><td>6.25 km ASI, TH = 75%</td><td>2002/03–2024/25, excluding 2011/12</td></tr><tr><td>85% versus 75%</td><td>Upper SIC threshold</td><td>MAPE 32.47</td><td>6.25 km ASI, TH = 75%</td><td>2002/03–2024/25, excluding 2011/12</td></tr><tr><td>90% versus 75%</td><td>Upper SIC threshold</td><td>MAPE 65.04</td><td>6.25 km ASI, TH = 75%</td><td>2002/03–2024/25, excluding 2011/12</td></tr><tr><td>12.5 versus 25 km</td><td>SIC product; TH = 80%</td><td>MAPE 11.47</td><td>12.5 km NT2</td><td>2012/13-2024/25</td></tr><tr><td>6.25 versus 25 km</td><td>SIC product; TH = 80%</td><td>MAPE 68.19</td><td>25 km CDR</td><td>2002/03–2024/25, excluding 2011/12</td></tr></table>

Notes: All statistics compare seasonal extent integrals. MAPE follow Eq. (4), with the listed series as denominator. The threshold comparisons hold the input product fixed. The product comparisons change retrieval and sampling together. Their periods and denominators differ and are therefore interpreted within their stated comparison designs. Values summarize Sections 4.1–4.2.

The threshold and product experiments should therefore be read as complementary rather than ranked against each other. Threshold experiments hold the observational input fixed and expose the sensitivity of the connectivity-based classifier to the admissible SIC range. Product experiments hold the classifier and 80% threshold fixed but change the SIC field itself. Together they identify two separate pathways by which a reported polynya extent can change: the operational boundary chosen within one SIC field and the observational product from which that field is derived.

The literature supports this two-pathway interpretation. SIC inter-comparisons show systematic differences among retrieval algorithms, especially away from compact-ice end members[6], [17], [18], while recent polynya studies demonstrate strong threshold, grid, and morphological sensitivity[15], [16], [19]. Kern et al.[19], for example, compared PSSM-derived polynya maps with higher-resolution AMSR-E 89- GHz ASI concentration maps and showed that differences in spatial resolution and observing frequency can lead to systematic differences in the retrieved open-water and thin-ice signatures, highlighting the sensitivity of polynya estimates to the characteristics of the underlying microwave product. The agreement in direction across independent studies strengthens the practical conclusion here: the processing configuration is part of the measurement definition, not a neutral post-processing detail.

For long-term Arctic monitoring, this means that consistency of configuration is essential for quantitative comparability. A fixed product version, upper SIC threshold, regional mask, geographic exclusions, and missing-data treatment should be reported with the extent record. The 80% threshold used here provides a reproducible operational reference within the tested range, while Figures 4–8 quantify how far results can move when that reference is changed. The more stable temporal features can support comparative analyses, but absolute extent values should remain tied to the configuration that produced them.

## 5.3 Interpretation of accuracy, sensitivity, and the physical meaning of SICdefined polynya extent

The 63-scene assessment shows strong agreement between the mapped polynya and MIZ classes and visual interpretation of SIC imagery. Sampling therefore focused on the polynya and MIZ classes, whose assignment involves a genuine classification decision. By contrast, the high-concentration ice and open-water classes are generated deterministically by the workflow, so their classification carries no assignment error and they were not sampled separately. The reported accuracy therefore estimates the classification performance for the polynya and MIZ classes at the record scale under a stratified random sampling design. Independent higher-resolution observations, such as Landsat or SAR, would provide additional evidence on boundary placement and small features unresolved by the SIC grids[33]. The lower polynya producer's accuracies in December and April identify useful periods for targeted boundary assessment. Broader accuracy statements covering additional classes would require extending the sampling design accordingly[28], [29].

The sensitivity experiments quantify the effects of upper SIC threshold and product selection. Because the products differ in both retrieval algorithm and spatial sampling, their comparison captures a combined product effect. The 80% threshold provides a common reference for these experiments, although it has not been independently calibrated. Alternative neighborhood configurations and geographic exclusion rules were explored during method development, with the final settings selected based on the quality of the resulting polynya delineation and then held fixed throughout the sensitivity experiments. Their individual effects on mapping accuracy and extent have not been systematically quantified. Controlled regridding of a common SIC field, independent threshold calibration, and quantitative comparisons of alternative spatial rules would help distinguish these effects. Data continuity also warrants consideration: the primary record excludes the unavailable 2011/12 season, while interpolation across short gaps does not provide independent information on polynya opening or closure during those dates.

SIC-defined polynya extent represents the total area of pixels assigned to the polynya class, including both their open-water and ice-covered fractions. It therefore provides a consistent spatial basis for examining polynya distribution and temporal variability under a specified mapping configuration. Estimating heat flux or ice production requires additional information on ice thickness, open-water fraction, surface temperature, and atmospheric forcing[34], [35], [36]. Linking the mapped extent with these complementary observations would extend its use to quantitative assessments of polynya thermodynamic processes.

## 6 Conclusions

A pan-Arctic polynya mapping workflow has been developed using hierarchical

SIC partitioning, connected-component labeling, and differentiated growth of polynya and open-water regions. Applied to September–May seasons from 2002/03 to 2024/25, it separates enclosed low-concentration openings from the ocean-connected ice–water mixture and achieves a mean overall accuracy of 96.74% and a mean kappa coefficient of 0.93 in the 63-scene visual assessment.

The sensitivity experiments reveal the central result of the study: absolute SICdefined polynya extent is highly configuration-sensitive, while broad temporal organization is comparatively more stable. Relative to 75%, aggregate threshold differences are 13.48%, 32.47%, and 65.04% at 80%, 85%, and 90%, respectively. The response is strongly regional and seasonal, with the largest threshold sensitivity in the Central Arctic Ocean, very weak sensitivity in the Gulf of St. Lawrence, and increasing aggregate separation from January onward. At a fixed 80% threshold, productcomparison MAPEs are 11.47% for 12.5 versus 25 km and 68.19% for 6.25 versus 25 km under their specified comparison periods and reference series. Regional disagreement ranges from below 50% in some masks to above 100% in others, showing that no universal Arctic-wide correction can reconcile configurations.

These findings shift the interpretation of sensitivity from a single error percentage to a structured property of the mapping system. Threshold and product choice alter magnitude in ways that depend on region and season, whereas many major temporal features remain recognizable. A long-term SIC-based polynya record is therefore most reproducible when product, threshold, masks, and connectivity rules are kept fixed and reported as part of the measurement definition. Within that framework, hierarchical partitioning and region growing provide an interpretable basis for comparing Arctic polynya extent through time.

Funding: This work was supported by the National Natural Science Foundation of China (No. 42376241)

## Acknowledgements

We thank the University of Bremen and the National Snow and Ice Data Center

for providing the SIC products. We also acknowledge the Centre for Polar Observation and Modelling for distributing the regional mask.

## References

[1] K. Nakata and K. I. Ohshima, “Mapping of Active Frazil and Sea Ice Production in the Northern Hemisphere, With Comparison to the Southern Hemisphere,” JGR Oceans, vol. 127, no. 12, p. e2022JC018553, Dec. 2022, doi: 10.1029/2022JC018553.

[2] G. A. Duffy, F. Montiel, A. Purich, and C. I. Fraser, “Emerging long-term trends and interdecadal cycles in Antarctic polynyas,” Proc. Natl. Acad. Sci. U.S.A., vol. 121, no. 11, p. e2321595121, Mar. 2024, doi: 10.1073/pnas.2321595121.

[3] M. Mohrmann, C. Heuzé, and S. Swart, “Southern Ocean polynyas in CMIP6 models,” The , vol. 15, no. 9, pp. 4281–4313, Sep. 2021, doi: 10.5194/tc-15-4281-2021.

[4] C. Heuzé, J. W. Rheinlænder, T. Tian, and C. H. M. Wong, “Winter Arctic polynyas in CMIP6 models,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 20, no. 6, pp. 3643–3682, Jun. 2026, doi: 10.5194/tc-20- 3643-2026.

[5] X. Liu, T. Feng, Y. Yang, and R. Li, “Characterization of north Greenland polynyas with super-resolved passive microwave sea ice concentration,” GIScience & Remote Sensin<sub>g</sub>, vol. 61, no. 1, p. 2300222, Dec. 2024, doi: 10.1080/15481603.2023.2300222.

[6] S. Kern et al., “Satellite passive microwave sea-ice concentration data set intercomparison using Landsat data,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 16, no. 1, pp. 349–378, Jan. 2022, doi: 10.5194/tc-16-349-2022.

[7] C. Heuzé and C. H. M. Wong, “Automatic detection of Arctic polynyas using hybrid supervised-unsupervised deep learning,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 19, no. 11, pp. 6043–6058, Nov. 2025, doi: 10.5194/tc-19-6043-2025.

[8] Y.-R. Wang and X.-M. Li, “Arctic sea ice cover data from spaceborne synthetic aperture radar by deep learning,” Earth S<sub>y</sub>st. Sci. Data, vol. 13, no. 6, pp. 2723–2742, Jun. 2021, doi: 10.5194/essd-13-2723-2021.

[9] G. K. Burada, A. McDonald, J. Renwick, and B. Jolly, “Delineating Polynya Area Using Active and Passive Microwave Sensors for the Western Ross Sea Sector of Antarctica,” Remote , vol. 15, no. 10, p. 2545, May 2023, doi: 10.3390/rs15102545.

[10] N. C. Brubacher, L. Xu, R. Segal, K. A. Scott, and D. A. Clausi, “Detection of Landfast Ice Polynyas from SAR Imagery Using Deep Learning,” Canadian Journal of Remote Sensin<sub>g</sub>, vol. 52, no. 1, p. 2638600, Dec. 2026, doi: 10.1080/07038992.2026.2638600.

[11] T. Wulf, J. Buus-Hinkler, S. Singha, H. Shi, and M. B. Kreiner, “Pan-Arctic sea ice concentration from SAR and passive microwave,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 18, no. 11, pp. 5277–5300, Nov. 2024, doi: 10.5194/tc-18-5277-2024.

[12] X. Chen et al., “MMSeaIce: a collection of techniques for improving sea ice mapping with a multi-task model,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 18, no. 4, pp. 1621–1632, Apr. 2024, doi: 10.5194/tc-18-1621-2024.

[13] X. Chen et al, “A comparative study of data input selection for deep learning-based automated sea ice mapping,” International Journal of A<sub>pp</sub>lied Earth Observation and , vol. 131, p. 103920, Jul. 2024, doi: 10.1016/j.jag.2024.103920.

[14] Y. Lin, Y. Nakayama, K. Liang, Y. Huang, D. Chen, and Q. Yang, “A dataset of the daily edge of each polynya in the Antarctic,” Sci Data, vol. 11, no. 1, p. 1006, Sep. 2024, doi: 10.1038/s41597-024-03848-2.

[15] M. Hurst and L. Boehme, “An automated method for polynya detection using a geomorphon algorithm,” Ocean Sci., vol. 22, no. 4, pp. 2559–2594, Aug. 2026, doi:

10.5194/os-22-2559-2026.

[16] L. L. Landrum, A. K. DuVivier, M. M. Holland, K. Krumhardt, and Z. Sylvester, “Challenges in identifying Antarctic coastal polynyas in satellite observations and climate model output to support ecological climate change research,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 20, no. 3, pp. 1815–1840, Mar. 2026, doi: 10.5194/tc-20-1815-2026.

[17] N. Ivanova et al., “Inter-comparison and evaluation of sea ice algorithms: towards further identification of challenges and optimal approach using passive microwave observations,” , vol. 9, no. 5, pp. 1797–1817, Sep. 2015, doi: 10.5194/tc-9-1797-2015.

[18] S. Kern et al., “Satellite passive microwave sea-ice concentration data set intercomparison: closed ice and ship-based observations,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 13, no. 12, pp. 3261–3307, Dec. 2019, doi: 10.5194/tc-13-3261-2019.

[19] S. Kern, G. Spreen, L. Kaleschke, S. De La Rosa, and G. Heygster, “Polynya Signature Simulation Method polynya area in comparison to AMSR-E 89GHz sea-ice concentrations in the Ross Sea and off the Adélie Coast, Antarctica, for 2002–05: first results,” Ann. Glaciol., vol. 46, pp. 409–418, 2007, doi: 10.3189/172756407782871585.

[20] C. Melsheimer and G. Spreen, “AMSR-E ASI sea ice concentration data, Arctic, version 5.4 (NetCDF) (June 2002 - September 2011).” PANGAEA, p. 50 data points, 2020. doi: 10.1594/PANGAEA.919777.

[21] G. Spreen, L. Kaleschke, and G. Heygster, “Sea ice remote sensing using AMSR‐E 89‐GHz channels,” J. Geo<sub>p</sub>h<sub>y</sub>s. Res., vol. 113, no. C2, p. 2005JC003384, Feb. 2008, doi: 10.1029/2005JC003384.

[22] W. Meier, T. Markus, and J. Comiso, “AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion &amp;amp; Snow Depth Polar Grids, Version 1.” NASA National Snow and Ice Data Center Distributed Active Archive Center, 2018. doi: 10.5067/RA1MIJOYPK3P.

[23] W. Meier, F. Fetterer, A. Windnagel, J. S. Stewart, and T. Stafford, “NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 5.” National Snow and Ice Data Center, 2024. doi: 10.7265/RJZB-PF78.

[24] W. Meier, F. Fetterer, A. Windnagel, J. S. Stewart, and T. Stafford, “Near-Real-Time NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 3.” National Snow and Ice Data Center, 2024. doi: 10.7265/J0Z0-4H87.

[25] “A Guide to NSIDC’s Polar Stereographic Projection | National Snow and Ice Data Center.” Accessed: Sep. 28, 2026. [Online]. Available: https://nsidc.org/data/user-resources/helpcenter/guide-nsidcs-polar-stereographic-projection#anchor-related-articles

[26] Centre for Polar Observation and Modelling, “NSIDC\_Regions\_Masks.nc,” CMIP6\_data repository, GitHub. Accessed: Sep. 28, 2026. [Online]. Available: https://github.com/CPOMUCL/CMIP6\_data/blob/main/NSIDC\_Regions\_Masks.nc

[27] W. Meier and J. Stewart, “Arctic and Antarctic Regional Masks for Sea Ice and Related Data Products, Version 1.” NASA National Snow and Ice Data Center Distributed Active Archive Center, 2023. doi: 10.5067/CYW3O8ZUNIWC.

[28] S. V. Stehman, “Sampling designs for accuracy assessment of land cover,” International , vol. 30, no. 20, pp. 5243–5272, Sep. 2009, doi: 10.1080/01431160903131000.

[29] P. Olofsson, G. M. Foody, M. Herold, S. V. Stehman, C. E. Woodcock, and M. A. Wulder,

“Good practices for estimating area and assessing accuracy of land change,” Remote , vol. 148, pp. 42–57, May 2014, doi: 10.1016/j.rse.2014.02.015.

[30] H. Fu, J. Zhao, and K. E. Frey, “Investigation of polynya dynamics in the northern Bering Sea using greyscale morphology image-processing techniques,” International Journal of , vol. 33, no. 7, pp. 2214–2232, Apr. 2012, doi: 10.1080/01431161.2011.608088.

[31] S. Kern, A. Rösel, L. T. Pedersen, N. Ivanova, R. Saldo, and R. T. Tonboe, “The impact of melt ponds on summertime microwave brightness temperatures and sea-ice concentrations,” The Cr<sub>y</sub>os<sub>p</sub>here, vol. 10, no. 5, pp. 2217–2239, Sep. 2016, doi: 10.5194/tc-10-2217-2016.

[32] S. Kern, T. Lavergne, D. Notz, L. T. Pedersen, and R. Tonboe, “Satellite passive microwave sea-ice concentration data set inter-comparison for Arctic summer conditions,” The , vol. 14, no. 7, pp. 2469–2493, Jul. 2020, doi: 10.5194/tc-14-2469-2020.

[33] H.-S. Jung, S.-M. Lee, J.-H. Kim, and K. Lee, “An Arctic sea ice concentration data record on a 6.25 km polar stereographic grid from 3 years of Landsat-8 imagery,” EarthS<sub>y</sub>st Sci , vol. 17, no. 1, pp. 233–258, Jan. 2025, doi: 10.5194/essd-17-233-2025.

[34] A. Preußer, K. I. Ohshima, K. Iwamoto, S. Willmes, and G. Heinemann, “Retrieval of Wintertime Sea Ice Production in Arctic Polynyas Using Thermal Infrared and Passive Microwave Remote Sensing Data,” JGR Oceans, vol. 124, no. 8, pp. 5503–5528, Aug. 2019, doi: 10.1029/2019JC014976.

[35] K. Iwamoto, K. I. Ohshima, T. Tamura, and S. Nihashi, “Estimation of thin ice thickness from AMSR-E data in the Chukchi Sea,” InternationalJournal ofRemote Sensin<sub>g</sub>, vol. 34, no. 2, pp. 468–489, Jan. 2013, doi: 10.1080/01431161.2012.712229.

[36] H. Kashiwase, K. I. Ohshima, Y. Fukamachi, S. Nihashi, and T. Tamura, “Evaluation of AMSR-E Thin Ice Thickness Algorithm from a Mooring-Based Observation: How Can the Satellite Observe a Sea Ice Field with Nonuniform Thickness Distribution?,” Journal of , vol. 36, no. 8, pp. 1623–1641, Aug. 2019, doi: 10.1175/JTECH-D-18-0218.1.