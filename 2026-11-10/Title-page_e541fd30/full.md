## Title page

## Title

AI-based characterization of lymphoid structures and the tumor microenvironment reveals lower mature tertiary lymphoid structures, lymph-node follicles, and plasma cells after neoadjuvant therapy in lung squamous cell carcinoma

## Authors

Arne Schmidt¹, Justin Nieder²<sup>,</sup>³, Cornelius Boehm¹, Julika Ribbat-Idel¹<sup>,</sup>⁴, Marius Teodorescu¹, Marie-Lisa Eich⁵, Simon Schallenberg⁵, Gabriel Dernbach<sup>1,5</sup>, Katja Lingelbach¹, Miriam Haegele<sup>1</sup>, Timo Milbich¹, Adam Narai¹, Maria Stoelben<sup>1</sup>, Hans Pinckaers<sup>1</sup>, Lukas Ruff¹, Kai Standvoss¹, Christian Watermann², Christiane Kuempers²<sup>,</sup>⁴, Sven Perner⁷, Verena Sailer²<sup>,4</sup>, Jasmin Tiebach<sup>6</sup>, Torsten Goldmann⁶, Sebastian Marwitz⁶, Johannes Braegelmann<sup>8</sup>, Frederick Klauschen<sup>1,9,10,11,12</sup>, Marija Pezer¹, Rosemarie Krupar¹<sup>,</sup>⁴

## Affiliations

¹ Aignostics GmbH, Berlin, Germany

² Institute of Pathology, University Hospital Schleswig-Holstein (UKSH), Campus Luebeck, Germany

³ Schoen Clinic Group Neustadt, Neustadt, Germany

⁴ University of Luebeck, Luebeck, Germany

⁵ Institute of Pathology, Charité – Universitätsmedizin Berlin, Berlin, Germany

⁶ Research Center Borstel, Leibniz Lung Center, Borstel, Germany

⁷ PATHORA Institute of Pathology and Tissue Medicine, Reutlingen, Germany

<sup>8</sup> Medical Clinic III for Oncology, Hematology, Immune-Oncology and Rheumatology, University Hospital Bonn (UKB),

<sup>9</sup> BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany

<sup>10</sup> Institute of Pathology, Ludwig-Maximilians-University, Munich, Germany

<sup>11</sup> German Cancer Consortium (DKTK), German Cancer Research Center (DKFZ), Munich Partner Site,

<sup>12</sup> Bavarian Cancer Research Center (BZKF), Munich Partner Site, Munich, Germany

## Running title

Neoadjuvant therapy-associated differences in TLS, SLS and TME of LUSC

## Additional information

Corresponding author: Rosemarie Krupar; Aignostics GmbH, Alt-Moabit 73/73A, 10555 Berlin; Email: rosemarie.krupar@aignostics.com

Conflict of interest disclosure

Additional information

Manuscript word count: 7409 including title

Number of figures: 4

Number of tables: 1

## Abstract

Mature, germinal centre–bearing tertiary lymphoid structures (TLS) and mature lymph node follicles (secondary lymphoid structures/SLS) sustain the plasma-cell humoral response that shapes outcome and immunotherapy benefit in lung cancer. How this loco-regional axis differs between neoadjuvant-treated and therapy-naïve residua tumors is poorly defined. We analyzed H&E whole-slide images of the resected tumor and paired proximal and distal lymph nodes from matched neoadjuvant-treated (neoLUSC, n=20) and therapy-naïve (naivLUSC, n=20) lung squamous cell carcinoma (LUSC) patients. Aignostics Atlas H&E Tissue Analysis (HETA) models quantified single-cell densities in different tissue compartments, and cell–cell co-localizations. A dedicated deep-learning model, validated against expert pathologist annotations, detected TLS and SLS and graded each as mature or immature. Prognostic value of cell-level readouts was tested in an independent LUSC cohort (n=458, 412 therapy-naïve and 46 treated patients). In neoLUSC patients stromal and carcinoma immune cell densities were lower, most pronounced for the humoral axis: Stromal plasma cells were 72% (p=0.004) and stromal lymphocytes 48% (p=0.015) lower in neoLUSC tumors compared to naivLUSC tumors. This difference was not observed for macrophages and granulocytes. With regard to TLS the difference was selective for maturity: Total and immature TLS did not differ between cohorts, whereas mature TLS in the tumor (p=0.034) and the tumor-distal lymph node (p=0.003) were lower in neoLUSC patients (naïve vs chemotherapy $\mathsf { p { = } } 0 . 0 1 8 ,$ naïve vs chemoradiation p=0.007). Spatially, the peri-TLS plasma cell to tumor lymphocyte association present in naivLUSC tumors (ρ=0.54, p=0.017) was not observed in neoLUSC tumors $_ { ( \rho = 0 . 0 7 ) }$ . In the validation cohort, the cell-level readouts that were lower in treated tumors were prognostic: Stromal plasma cell density (HR 0.69, $_ { \mathsf { p } = 0 . 0 1 1 }$ and lymphocyte–plasma cell co-localization (HR 0.66, p=0.004) predicted longer survival, while neoadjuvant status was adverse (HR 1.50, $\mathsf { p { = } } 0 . 0 2 6 )$ . Neoadjuvant therapy in LUSC thus seems to be associated with fewer mature lymphoid structures and with weaker coupling of the humoral compartment to the tumor rather than with uniformly lower immune readouts, most pronounced after chemoradiation. The readouts that were lower in treated tumors are prognostic in an independent cohort, nominating the stromal plasma cell/co-localization axis as a candidate biomarker.

## Introduction

Lung cancer remains the most commonly diagnosed malignancy and the leading cause of cancer-related death worldwide, accounting for approximately 2.5 million new cases and 1.8 million deaths in 2022 [1]. Non–small cell lung cancer (NSCLC) comprises roughly 80–85% of cases, with lung squamous cell carcinoma (LUSC) representing the second most common subtype after adenocarcinoma [2]. Because most patients present with advanced disease, long-term outcomes are poor, with an overall five-year relative survival of approximately 29% [3]. Immune-checkpoint inhibitors (ICI) directed against the PD-1/PD-L1 axis have become a cornerstone of NSCLC treatment, yet 50–80% of patients derive no durable benefit, and PD-L1 expression alone remains an imperfect predictor of response [4,5]. This has intensified interest in the broader architecture of the tumor immune microenvironment (TIME) as a determinant of antitumor immunity.

Beyond individual immune-cell densities, the spatial organization of the TIME carries independent prognostic and predictive value [6]. Tertiary lymphoid structures (TLS) — ectopic lymphoid aggregates that recapitulate the architecture and function of secondary lymphoid organs — are associated with favorable prognosis and improved ICI response in NSCLC [7]. Critically, it is the maturation state of these structures, rather than their mere presence, that determines their clinical relevance: a germinal center (GC)–bearing, mature TLS sustains the B-cell reactions that drive plasma-cell differentiation and the production of tumor-specific antibodies [8,9]. This plasma-cell output fuels an intratumoral immunity cycle [10], and mature TLS predict ICI efficacy independently of PD-L1 expression and CD8+ T-cell density [11].

This antigen-driven response is not confined to the tumor. Antigen presentation and B-cell maturation classically occur within the follicles and germinal centers of draining lymph nodes, the systemic arm of the humoral response [12], and intratumoral TLS maturation is coupled to the maturation state of these regional nodes in lung cancer [13,14]. Together, the intratumoral (TLS) and regional (lymph node follicle/GC) compartments form a spatially linked axis whose disruption could blunt the entire plasma-cell–driven response.

Neoadjuvant chemotherapy or chemoradiation, increasingly combined with a PD-L1 inhibitor, is now standard o care for resectable stage III NSCLC [15]. On the one hand, cytotoxic therapy can actively promote antitumor immunity: Chemotherapy- and radiation-induced immunogenic cell death releases tumor antigens and damage-associated molecular patterns that activate dendritic cells and prime CD8⁺ T-cell responses [16]. Radiation in particular can engage cytosolic DNA sensing via the cGAS–STING axis and type-I interferon signalling, reshaping the tumor microenvironment (TME) toward a more immunogenic, T-cell–inflamed state [17,18]. Yet these modalities exert pronounced suppressive effects on the very B-cell and germinal-center compartments that underlie mature lymphoid structures, including lymphocyte depletion [19], loss of CD20+ B cells and TLS density in recurrent squamous tumors after chemoradiation [20], and impaired TLS maturation with germinal-center attrition [8].

How these opposing effects ultimately resolve in LUSC is difficult to read from existing data, because the local and regional arms of the response have mostly been examined one compartment at a time, and rarely with the scalable, whole-slide quantification needed to relate the intratumoral and lymph node compartments within the same patients. We therefore set out to resolve, across the full tumor to lymph node axis, if neoadjuvant chemotherapy and radiotherapy are associated with differences in the local TME, the intratumoral TLS, and the regional lymph node follicles. We focused on whether treatment is associated with differences in the maturation of lymphoid structures and in the coupling between the tumor and lymph node compartments, and how these differences relate to prognosis. To do so, we applied AI-based whole-slide segmentation and cell classification, built on a pathology foundation model, to generate quantitative, cell-resolved and spatially explicit readouts, an approach that scales the analysis well beyond manual assessment and renders the local–regional circuit measurable as a whole. We examined these readouts in a matched cohort of neoadjuvant treated (neoLUSC) and treatment-naïve surgical (naivLUSC) LUSC patients, each case paired per patient with a tumor-proximal and a tumor-distal lymph node, so that loco-regional differences between treated and therapy-naïve patients in the mature-lymphoid and plasma cell compartments that drive the antitumor humoral response could be compared along a single tumor-to-lymph node axis.

## Materials and methods

## Study cohorts

Two independent LUSC cohorts were analyzed in this study:

1. Discovery cohort: The first, whole slide image (WSI)-based, discovery cohort was created from the archives of the Institute of Pathology, UKSH Campus Lübeck, and the Pathology of the Research Center Borstel (Leibniz Lung Center) consisting of 40 patients, who were diagnosed with LUSC between 2009 - 2020 and treated by oncologica tumor resection with locoregional lymphadenectomy. Overall survival data, patient characteristics and social histories were provided by the University Hospital Schleswig‐Holstein, Campus Lübeck, the LungenClinic Grosshansdorf and the Institute for Cancer Epidemiology of the University Lübeck. The neoadjuvantly treated neoLUSC subgroup (n=20) comprised patients who had received neoadjuvant chemotherapy and/or radiotherapy prior to resection, including patients who died during neoadjuvant therapy and for whom autopsy tissue was available. The therapy-naïve naivLUSC subgroup (n=20) was restricted to patients who had received a surgica resection alone with no prior neoadjuvant conservative treatment and served as a matched control. Matching was performed manually using pathologically assessed TNM status, UICC stage, sex, and age at diagnosis. Cases were stratified by TNM status and assigned proportionally to the neoLUSC distribution, with ties resolved by the next-ranked criterion. The use of patient data and tumor material was approved by the Ethics Committee of the University of Lübeck (vote of 08 November 2016, AZ 16-277), and all material was fully anonymized. NeoLUSC patients were restricted to Junker regression grade IIa (>10% vital tumor), ensuring both a demonstrable therapy response and sufficient evaluable tumor tissue. Each resection was paired with a tumor-proximal (N1) and a tumor-distal (N2/3) tumor-free lymph node of the same patient, allowing the local TME and its draining nodal compartments to be examined along a single tumor-to-node axis within each patient. Paired tumor-free lymph nodes were evaluable for 37 of 40 patients proximally (N1) and 39 of 40 distally (N2/3). Four lymph node slides were excluded due to out of focus scans. The excluded lymph nodes (three proximal and one distal) were all from the naivLUSC cohort. Neoadjuvant modality comprised chemotherapy (n=11), chemoradiotherapy (n=5), and radiotherapy alone (n=2), with the modality undocumented in two patients. Mean age was 61.2 years (SD 7.1) in the neoLUSC group and 69.1 years (SD 6.9) in the naivLUSC group. The age difference was significant (p = 0.001). TNM staging followed the 8th edition; pT, pN, R-status, and UICC-stage (IA3–IVA). The distributions of all clinica and pathological parameters per group are given in Table 1.

2. The validation cohort: The second cohort was a tumor microarray (TMA)-based TME-focused validation cohort. It was created as a subset from a large, independent, well-characterized and previously described non-small cell lung cancer cohort [21] and included 458 LUSC patients, of which 46 patients had received neoadjuvant chemotherapy and/or radiation prior to resection and 412 patients had received resection without prior therapy.

## Sample selection and WSI processing

For the discovery cohort three representative H&E-stained sections were selected per patient by a board-certified pathologist (R. K.): (1) Tumor tissue with peritumoral parenchyma, (2) a tumor-free lymph node proximal to the primary within the N1 region (stations 10, 11, 12), and (3) a tumor-free lymph node distal to the primary within the N2/3 region (stations 4, 5, 6). Conventional H&E stains were prepared following established protocols and digitized on a Ventana BenchMark Ultra (Roche, Basel, Switzerland). WSIs were visualized in Aignostics’ in house slide viewer

Portal. Annotations for model training, model performance measurements and downstream analyses were also executed in Portal by four board-certified anatomical pathologists (M.-L. E., J. R.-I., M. T., R. K.).

## AI-based analysis of the tumor microenvironment

The TME analysis used Aignostics Atlas H&E Tissue Analysis (HETA) model outputs [22], executing a sequentia per-slide workflow : Tissue QC, tissue segmentation, cell detection and cell classification followed by quantification of spatial readouts. Tissue QC identified valid tissue regions. Valid tissue was segmented into seven tissue types (carcinoma, stroma, epithelium, vessel, blood, necrosis, and other). Cells were detected and classified within valid tissue, excluding necrosis and blood, into nine cell types (carcinoma cell, epithelial cell, endothelial cell, fibroblast, granulocyte, lymphocyte, macrophage, plasma cell, and other), with nuclear morphometric features recorded per cell and each cell assigned to its tissue region. Whole tumor regions (WTR), defined as the outermost boundary enclosing all carcinoma areas, as well as interspaced non-carcinoma TME areas within 500 µm distance of a carcinoma area, were manually annotated by a board-certified anatomical pathologist for each tumor slide. Peri- or intratumoral lymph nodes were excluded from WTR annotations. Additionally, board-certified anatomica pathologists annotated lymphoid areas for each lymph node slide for density normalization and cell level readouts. All model predictions were quantified within the WTR regions. Readouts comprised per-region cell counts, percentages, and densities by cell class, per–tissue-type area and morphometric summaries.

## AI-based analysis of tertiary lymphoid structures and lymphoid follicles

A TLS segmentation model built on Aignostics’ Atlas foundation model backbone [23] was trained to predict three classes (no TLS, immature TLS, mature TLS) using annotations from 29 slides, then evaluated on a held-out set of 10 annotated tumor slides. On raw pixel-level predictions, performance was high for the ‘no TLS’ class (F1 0.99) and moderate for the ‘immature TLS’ and ‘mature TLS’ classes (F1 0.73 and 0.53, respectively). External model validation on an independent urothelial carcinoma hold-out set (10 slides) confirmed reliable ’no TLS’ detection (F1 0.99) with more modest performance for the ‘immature TLS’ and ‘mature TLS’ classes (F1 0.62 and 0.50, respectively) for raw-pixel predictions.

Predictions were then post-processed by excluding those with fewer than 50 lymphocytes, a threshold retained from in house work and consistent with literature convention, and merging spatially close predictions [19]. A maturity threshold, tuned on the training set, classified a TLS as mature when >25% of its area was predicted mature, the best-performing of the tested thresholds (>0%, >25%, >50%, >75%). Post-processing improved agreement on the neoLUSC hold-out set, raising total and ‘immature TLS’ pixel-level F1 to 0.81 and 0.79 and ‘mature TLS’ F1 to 0.61.

In lymph nodes, the model reliably captured lymphoid follicles and classified them into primary follicles and germinal center-containing mature follicles, reported as mature secondary lymphoid structures (SLS), as confirmed by expert-pathologist review.

TLS and mature SLS readouts comprised area, count, density, polygon shape descriptors, and, for TLS, inside-TLS as well as surrounding-ring cell composition. For lymph nodes, it included cell composition within annotated lymphoid areas.

## Statistical analysis

Group differences in each readout, between the therapy-naïve and neoadjuvant cohorts, and for the modality analysis pairwise among the therapy-naïve group and the individual neoadjuvant regimens, were tested with the Mann–Whitney U test, with the rank-biserial correlation r as effect size. Categorical distributions, including the immune-desert/excluded (IDE) phenotype, were compared with Fisher’s exact test. Associations between compartments were quantified by Spearman rank correlation (ρ), computed both pooled across cohorts and separately within each cohort. For survival, continuous readouts were standardised and entered individually into univariable Cox proportional-hazards models (hazard ratio per standard-deviation increase), and the clinical covariates (UICC and nodal stage, age, sex, and neoadjuvant status) were tested as univariable predictors in the same way. Kaplan–Meier curves with median-split stratification and the log-rank test provided complementary group-level comparisons. All tests were two-sided, and p-values are reported without correction for multiple comparisons.

## Results

Neoadjuvant-treated tumors have fewer mature, organised lymphoid structures and lower stromal immune cell densities than therapy-naïve tumors

We first characterised the lymphoid architecture and immune cell composition of treated and therapy-naïve tumors and their draining lymph nodes, comparing TLS and mature SLS, stromal and intratumoral immune-cel densities, as well as the resulting immune phenotype across cohorts (Fig. 1).

Representative H&E sections with corresponding AI-model overlays illustrate TLS in tumors and mature SLS in tumor-distal lymph nodes for a naivLUSC and a neoLUSC patient (Fig. 1A). TLS and SLS are separated into mature (with germinal center) and immature or primary (without germinal center) ones. In tumor and corresponding draining lymph node samples of a naivLUSC patient the model identifies multiple mature TLS (row 1) and mature SLS (row 3) alongside immature ones, whereas the tumor and lymph node of a neoLUSC patient contain predominantly immature TLS (row 2) and SLS (row 4) with no mature counterparts. The total TLS density did not differ between cohorts (Fig. 1B). Neither total TLS density (rank-biserial $\mathsf { r } = - 0 . 2 1 , \mathsf { p } = 0 . 2 6 )$ nor immature TLS density $( \Gamma = - 0 . 2 0 , \ \mathfrak { p } = 0 . 2 9 )$ differed between neoLUSC and naivLUSC resections. The difference was instead confined to the mature, organised compartment: Mature TLS density trended lower in neoLUSC tumors $\mathtt { ( r = - 0 . 2 8 , }$ ${ \mathsf p } = { \mathsf { 0 } } . { \mathsf { 1 1 } } )$ , and mature TLS count was significantly lower $( \mathsf { r } = - 0 . 3 6 , \mathsf { p } = 0 . 0 3 4 )$ . The same directional pattern extended into the draining lymph nodes (Fig. 1C). Mature SLS area per lymphoid tissue tended to be lower in neoLUSC patients in the tumor-proximal node $( \mathsf { r } = - 0 . 3 8 , \mathsf { p } = 0 . 0 5 1 )$ ) and was significantly lower in the tumor-distal node $( \mathsf { r } \ = \ - 0 . 5 5 , \ \mathsf { p } \ = \ 0 . 0 0 3 3 )$ . In conclusion, the lower mature TLS/SLS readouts in neoLUSC were directionally consistent along the tumor-to-node axis, and within the draining nodes was more pronounced distally than proximally.

The tumor microenvironment was resolved using the HETA models, which segment WSI into tissue compartments and classify individual cells. Representative naivLUSC and neoLUSC tumors are shown as the H&E alongside the corresponding model overlays in Fig. 1D. Quantification of the tumor microenvironment revealed directionally consistent lower stromal immune cell densities in neoLUSC compared to naivLUSC tumors, with the largest difference for stromal plasma cells (Fig. 1E). Stromal plasma-cell density was 72% lower $( \mathsf { r } = - 0 . 5 4 , \mathsf { p } = 0 . 0 0 3 6 )$ and stromal lymphocyte density 48% lower $( \mathsf { r } = - 0 . 4 5 , \mathsf { p } = 0 . 0 1 5 )$ . Within the carcinoma compartment, plasma-cell density was 53% lower $( \mathsf { r } = - 0 . 3 9 , \mathsf { p } = 0 . 0 3 8 )$ . Myeloid and granulocytic populations trended in the same direction without reaching significance (macrophage r = −0.22, p = 0.24; granulocyte $\mathsf { r } = - 0 . 2 7 , \mathsf { p } = 0 . 1 5 )$ . Despite these immune cell differences, overall tissue architecture remained broadly comparable between arms. The stroma area was modestly larger (+10%) and vessel area smaller $( - 3 3 \% , \textsf { r } = - 0 . 3 4 , \textsf { p } = 0 . 0 6 8 )$ , while carcinoma and necrosis fractions were essentially stable. The smaller vessel area was one potential structural difference, which did not reach significance. But the slides remained otherwise comparable, suggesting that the lower immune cell densities were not explained by gross differences in tissue composition.

In the next step tumors were assigned an immune phenotype using a density-based inflamed–desert–excluded (IDE) classifier (Fig. 1F). Each tumor was positioned by its lymphocyte density in two compartments, within the carcinoma (x-axis) and within the tumor-associated stroma (y-axis) in the annotated whole tumor region, and classified by thresholds derived from the upper tertiles of these per-slide density distributions (dashed lines):

Tumors with high intratumoral lymphocyte density were assigned an inflamed (hot) phenotype, those with low intratumoral but high stromal lymphocyte density an excluded phenotype, and those low in both a desert (cold) phenotype. Consistent with lower immune cell infiltration, slides of treated tumors more frequently displayed a cold, immune-desert phenotype and less frequently a hot phenotype, although this contingency did not reach significance (Fisher exact test, ${ \mathsf p } = 0 . 1 0 5$ , Fig. 1G). Spatial neighborhood analysis reinforced the same picture (Fig. 1H): The stromal neighborhood contrast matrix at 40 µm was almost uniformly negative across all cell pairs where lymphocyte or plasma cell was the neighbor, indicating fewer immune cell–cell contacts in neoLUSC tumors. The difference was organised by neighbor rather than by centre cell: Plasma cells were the neighbor with the largest difference across nearly all cell types (lymphocyte-, plasma-, macrophage-, and granulocyte-centred neighborhoods all lower at $\mathsf { p } < 0 . 0 1 )$ , whereas macrophages and granulocytes were the neighbors with the smallest differences.

## Plasma cell densities correlate across lymphoid sites in both cohorts, whereas their correlation with tumor lymphocytes is observed only in therapy-naïve tumors

We next examined how these compartments are coordinated with one another, and whether that coordination differs between neoLUSC and naivLUSC patients, by examining the cohort-specific correlation structure of the lymphoid, plasma cell and immune cell density readouts (Fig. 2).

Tracing the plasma cell axis from the draining lymph nodes into the tumor showed numerically higher correlations between lymphoid sites in neoLUSC than in naivLUSC patients, whereas the correlation with the tumor lymphocyte infiltrate seen in naivLUSC was not observed in neoLUSC (Fig. 2A). Plasma cell density in the tumor-distal and tumor-proximal lymph nodes was correlated in neoLUSC tumors $( \rho \ = \ 0 . 7 4 )$ and positively but more weakly correlated in naivLUSC tumors $( \rho = 0 . 3 8 )$ . The same was true for the link between tumor-proximal lymph node and peri-TLS plasma cell density (neoLUSC $\rho = 0 . 7 2 ,$ ${ \mathsf p } < 0 . 0 0 1$ ; naivLUSC $\rho = 0 . 1 7 ,$ not significant). The local (peri-TLS) and regional (lymph node) plasma cell compartments therefore co-varied tightly in neoLUSC patients, consistent with a synchronised humoral compartment across lymphoid sites, which appeared to be more pronounced in the neoLUSC group compared to the naivLUSC group. This coordination did not extend to the tumour lymphocyte infiltrate: Peri-TLS plasma cell density covaried with stromal and intratumoral lymphocyte density in naivLUSC tumors $( \rho = 0 . 4 1$ and $\mathsf { \Omega } \mathsf { \rho } = 0 . 5 4 _ { \cdot }$ the latter ${ \mathsf p } = 0 . 0 1 7 )$ , but not in neoLUSC tumors $( \rho = - 0 . 2 2$ and $\rho = 0 . 0 7 )$ , suggesting that the humoral and cellular arms tracked together in untreated tumours but not in treated tumours.

Extending this analysis to the full set of readouts, cohort-split Spearman correlation matrices showed that the correlation architecture reorganised after neoadjuvant therapy rather than uniformly weakening (Fig. 2B). The plasma cell readouts (tumor-distal lymph node, tumor-proximal lymph node, and peri-TLS ring) were tightly correlated among themselves in both cohorts, even more pronounced in the neoLUSC cohort. This indicates that the plasma cell compartment was internally coordinated across its lymphoid sources in both cohorts. By contrast, the coupling of this plasma cell compartment to the surrounding TME differed between cohorts: Its positive correlations with the stromal immune populations in naivLUSC tumors was not detectable in neoLUSC tumors, mirroring the pairwise pattern in Fig. 2A. Most strikingly, the correlation between tumor TLS density and the local plasma cell readouts was weakly positive in naivLUSC tumors and negative in neoLUSC tumors. The stroma immune cells were correlated with one another in both cohorts, and the secondary lymphoid structures were only weakly linked to the other compartments throughout. Tumor TLS density showed a positive correlation with the stromal immune compartment in both cohorts and showed little relation to intratumoral carcinoma lymphocytes, so the therapy-associated decoupling was concentrated in the plasma cell readouts rather than affecting the organised lymphoid compartment as a whole. Taken together, plasma cell densities were correlated across the draining lymph nodes and the peri-TLS ring in both cohorts, and the correlation between the lymph nodes and the peri-TLS ring was stronger in the neoLUSC cohort. In contrast, the association of plasma cells with tumor TLS density and with stromal lymphocytes, present in naivLUSC tumors, was not observed in neoLUSC tumors. The difference between cohorts therefore lies in the link between plasma cells and the TME, not in their link to the draining lymph nodes.

## Treatment-associated differences are directionally consistent across modalities and show concordant, non-significant trends toward longer survival

Next we assessed whether the treatment-associated differences were consistent across neoadjuvant modalities and whether any of the lymphoid or TME readouts related to overall survival (Fig. 3). We first summarised the treatment-associated differences across all readout families using group-level radar plots of the z-normalised readouts, comparing cohorts (Fig. 3A) and neoadjuvant modalities (Fig. 3B). At the cohort level (Fig. 3A), the TME profile of neoLUSC tumors was systematically smaller than that of naivLUSC tumors: NeoLUSC values lay inside the naivLUSC profile on every stromal and intratumoral (carcinoma) axis, most markedly for stromal and intratumora plasma cells and intratumoral lymphocytes. The tumor TLS and lymph node SLS families, by contrast, differed in both directions rather than being uniformly lower: Lower values were concentrated in the mature and structural axes (mature TLS density, mature SLS area), while other axes were similar or higher, notably peri-TLS ring lymphocyte density and nodal macrophage density, which were higher in neoLUSC. The dominant pattern was therefore broadly lower stromal and intratumoral immune cell densities in neoLUSC alongside mixed-direction differences in the organised lymphoid structures, in which structural and mature readouts were lower while selected cellular axes were similar or higher. Stratifying by modality (Fig. 3B) showed that this pattern was shared across regimens but not identical between them. The chemotherapy and chemoradiotherapy groups showed broadly overlapping, inward-shifted TME profiles, while the radiotherapy group showed the most inward-shifted TME profile of all groups. The lymphoid families again showed mixed-direction differences rather than uniformly lower values, and several axes diverged in a modality specific manner: Within the lymph node SLS family, macrophage and plasma cell densities were higher mainly in the radiotherapy and chemoradiotherapy groups but not in the chemotherapy group. Additionally, the radiotherapy group showed a higher peri-TLS ring lymphocyte density than the other groups. Because the radiotherapy $( \mathsf { n } = 2 )$ and chemoradiotherapy (n = 5) subgroups are small, these group-summarised profiles are descriptive and direction-of-difference only, and individual divergent axes should be read with particular caution.

Modality-stratified per-slide distributions were consistent with this overview (Fig. 3C–F). The proportion of cold tumors was progressively higher from naivLUSC tumors through the chemotherapy and chemoradiotherapy groups, with both radiotherapy cases classified as cold (Fig. 3C). Total TLS density did not differ significantly between any groups (Fig. 3D), and mature TLS density was higher in naivLUSC tumors at trend level only (chemotherapy vs NaivLUSC ${ \rho } = 0 . 0 6 5$ ; Fig. 3E). Mature SLS area showed the clearest separation, being significantly lower than naivLUSC in both, the chemotherapy $( \mathsf { p } = 0 . 0 1 8 )$ and chemoradiotherapy $( \mathsf { p } = 0 . 0 0 7 )$ , groups (Fig. 3F). Where present, the significant contrasts were between a treated modality and the therapy-naïve group; no pairwise comparison between neoadjuvant modalities reached significance.

Survival analysis in the discovery cohort was limited by sample size (n = 32 evaluable, 21 events), and no univariable readout reached statistical significance. The direction of association was nonetheless consistent: The stromal immune compartments, lower in neoLUSC tumors, each showed a protective tendency, with univariable hazard ratios (HR) below 1 for stromal lymphocyte density (HR 0.84), stromal plasma cell density (HR 0.88), stromal macrophage density (HR 0.81), and lymphocyte–plasma cell co-localization (HR 0.84) (Fig. 3G). The clearest tendencies were seen for the organised lymphoid readouts: By Kaplan–Meier median split, higher TLS density (median overall survival 1428 vs 749 days; log-rank ${ \sf p } = 0 . 0 6 9 .$ ; Fig. 3I) and higher peri-TLS ring plasma cell density (924 vs 471 days; p = 0.129; Fig. 3J) were each associated with longer survival. The cohort split itself was null (HR $0 . 8 6 , \mathfrak { p } = 0 . 7 4 ; \mathfrak { F i g } . 3 \mathsf { H } )$

An independent validation cohort confirms the stromal immune signature and supports its prognostic value To validate these findings in an adequately powered, independent setting, we applied the cell level readouts to an independent validation cohort of 458 LUSC patients (46 neoadjuvant-treated, 412 treatment-naïve; Fig. 4). The cohort contrast reproduced the stromal immune signature identified in the discovery cohort (Fig. 4A): Stroma lymphocyte density, stromal plasma cell density, and stromal lymphocyte–plasma cell co-localization were each significantly lower in neoadjuvant-treated tumors (rank-biserial r = −0.18, −0.22, and −0.18; p = 0.044, 0.014, and 0.041, respectively). In contrast, stromal macrophage density did not differ significantly between groups $( \mathsf { p } = 0 . 2 2 )$ recapitulating the absence of a significant macrophage difference seen in the discovery cohort. As expected for a neoadjuvant-treated subgroup, treated patients presented at higher UICC and nodal stage (both ${ \mathsf { p } } < 0 . 0 0 1 )$

In this larger cohort, the survival associations that were directionally suggestive, but underpowered in the discovery set, reached significance (Fig. 4B). In univariable Cox regression (n = 458, 255 events), higher densities of the immune compartments that were lower in treated tumors were each associated with a lower hazard of death: stromal lymphocyte density (HR 0.77 per +1 SD, ${ \sf p } = 0 . 0 0 5 )$ , stromal plasma cell density (HR 0.86, p = 0.038), intratumoral lymphocyte density (HR 0.80, $\mathsf { p } = 0 . 0 2 7 )$ , and both stromal co-localization measures (HR 0.83, p = 0.010 and 0.021) were associated with longer overall survival, while stromal macrophage density was not (HR 0.98, $\mathsf { p } = 0 . 8 0 )$ . Consistent with the higher stage of the treated subgroup, neoadjuvant status itself was associated with shorter survival $( \mathsf { H R 1 . 5 0 } , \mathsf { p } = 0 . 0 2 6 )$ , as were UICC and lymph node stage.

The spatial cell-based prognostic associations of the TME were confirmed by Kaplan–Meier analysis, which additionally revealed differences in the magnitude and structure of the separations (Fig. 4D–F). Higher stroma plasma cell density and higher stromal lymphocyte–plasma cell co-localization were both associated with substantially longer survival, with median overall survival differences of 26 months (67 vs 41 months; log-rank p = 0.0036; Fig. 4E) and 30 months (66 vs 36 months; ${ \mathsf p } = 0 . 0 0 1 0 ;$ Fig. 4F), respectively. In both, the survival curves diverged progressively over the 60-month horizon rather than separating early. In contrast, stromal lymphocyte density alone separated the groups only weakly and non-significantly (58 vs 50 months; ${ \mathsf p } = 0 . 1 5 6 ;$ Fig. 4D). The three markers thus formed a clear rank order: The spatial lymphocyte–plasma cell co-localization and the plasma cell density were the strongest discriminators, whereas bulk stromal lymphocyte density was the weakest. Notably, stromal lymphocytes carried little prognostic information on their own but became the strongest stratifier when their spatial co-localization with plasma cells was considered, suggesting that the prognostic signal is more closely related to the spatially coupled humoral compartment than to lymphocyte abundance per se. TLS prognostic associations were not validated in the larger cohort, as it is a TMA-based cohort.

## Discussion

Chemotherapy, chemoradiation and radiotherapy are widely expected to stimulate antitumor immunity, yet their capacity to deplete lymphocytes and disrupt lymphoid architecture is equally well documented. How these opposing effects resolve in the treated tumor and its draining nodes remains largely unquantified. We therefore examined the local and regional immune response in matched therapy-naïve and neoadjuvant-treated lung squamous cell carcinoma, using whole-slide AI-based tissue segmentation (including tertiary lymphoid structure detection) and cell classification to resolve the tumor immune microenvironment and lymphoid architecture along a tumor-to-lymph node axis. This approach captures not only cellular composition but also the spatial organisation of the response. Against the expectation of net immune activation, treated tumors did not differ from therapy-naïve tumors in a simple manner: Rather than uniformly higher or lower local immune readouts, the differences were selective and spatially organised, in which mature, germinal-centre-bearing TLS and the stromal plasma cell compartment were lower while other cellular compartments differed less. This pattern held across the tumor and its draining lymph nodes and on cell composition level was reproduced in an independent validation cohort. Treated tumors thus differed in the maturation state of the lymphoid response rather than lacking it, with the largest differences in the germinal-centre-associated humoral arm.

Myeloid cells showed a different pattern. Macrophage density was not significantly lower in treated patients in any cohort or compartment. The only difference in the opposite direction was a higher macrophage density in the tumor-draining lymph nodes of irradiated patients, while granulocyte density did not differ significantly. We hypothesise that this pattern, and the higher macrophage density in particular, may reflect a repair response to treatment rather than an intrinsic resistance of the myeloid cells to therapy. Cytotoxic treatment destroys tissue and kills cells on a large scale, and the recruitment of macrophages to clear cellular debris and remodel the injured tumor bed is a well-recognised consequence [25].

The most informative differences emerged not from any single compartment but from the relationships between them. In neoLUSC patients, the plasma-cell compartment was more tightly correlated across lymphoid sites: Densities in the distal and proximal draining nodes, and between the proximal node and the peritumoral TLS ring, co-varied more closely than in untreated tumors. In contrast, the coupling between the lymphoid output and the effector infiltrate of the tumor was not present: The co-variation between peritumoral plasma cells and the lymphocytes populating the tumor stroma and carcinoma, present in therapy-naïve tumors, was not detectable in treated tumors. Because H&E resolves a single lymphocyte class and does not capture antibody output, these associations reflect co-variation of the humoral and cellular infiltrates rather than a direct effector relationship. The same pattern was visible across the wider correlation structure, where within-compartment coherence was retained in both cohorts while the cross-compartment coupling between the lymphoid–plasma cell axis and the stromal and carcinoma compartments was weaker in neoLUSC tumors. This points to a disruption of the link between the local and systemic immune response rather than a general loss of immunity. A plausible interpretation is that the systemic, node-based arm of the humoral response remains organised and internally coordinated, while the local step, the integration of this output into the TME, is selectively disrupted. This would be consistent with the parallel reduction in mature, germinal-centre-bearing TLS: With the peritumoral maturation machinery impaired, the humoral compartment may remain quantitatively present and systemically synchronised yet functionally disconnected from the tumor it surrounds. That mature TLS decline under cytotoxic neoadjuvant therapy is, in itself, expected, germinal-centre content defines the prognostically relevant TLS fraction in this disease and has been shown to fall in chemotherapy- and radiotherapy-treated squamous lung tumors [8,19,20]. Our data add information on coordination rather than abundance alone. Previous work has described therapy-associated changes in the abundance, maturation and localization of lymphoid structures. We additionally find that the coordination between these compartments differs between treated and therapy-naïve patients: In treated patients, the systemic humoral axis was more tightly correlated internally, while its correlation with the local effector compartment was not detectable. Along the tumor-to-lymph node axis, the lesion is therefore best understood not as a loss of lymphoid structures or cells but as a decoupling within an otherwise preserved local–systemic circuit.

The lower number of mature lymphoid structures was not uniform across treatment modalities. Across the three compartments, the profiles of treated tumors differed from naïve ones in shape, not solely in size, in the radar plots, treated profiles lay inside the naïve profiles selectively along the mature-lymphoid and plasma-cell axes, the myeloid axes were comparatively similar, and in the irradiated draining lymph nodes the macrophage axis lay outside. This is consistent with selective rather than uniform differences between cohorts. This difference was most pronounced in the radiation-containing regimens and clearest for the germinal-centre-bearing follicles of the draining lymph nodes, consistent with the radiosensitivity of the germinal-centre reaction and radiotherapy’s lymphodepleting action [8,19,20]. Statistical support was constrained by subgroup size: Only the contrasts against treatment-naïve tumors reached nominal significance, for chemoradiation and, less strongly, chemotherapy, none surviving correction for multiple testing, and no modality ranked above another. The radiotherapy-alone group showed the largest decreases, but with only two such patients these fell short of significance. The modality data are therefore best read as descriptive: The distinct compartment shapes are compatible with larger differences in the radiation-containing regimens, but the subgroups are too small to attribute them to any single treatment. Whether these differences carry prognostic weight could not be resolved in the discovery cohort, which was underpowered for survival analysis: With only 21 deaths among the evaluable patients, no single immune or structural readout reached significance. The direction was nonetheless consistent, every feature that was lower in treated tumors pointed the protective way, higher densities associating with longer survival, with TLS density and the peritumoral plasma-cell ring the strongest tendencies. We therefore consider this result a consequence of limited power rather than evidence of no effect, and tested it in an independent cohort of 458 patients. There the same features were significantly and protectively prognostic: Stromal lymphocyte density, stromal plasma cell density, and the co-localization of lymphocytes with plasma cells each predicted longer overall survival. Lymphocyte density was thus prognostic in its own right, in keeping with the established value of tumor-infiltrating lymphocytes [6]. Its co-localization with plasma cells, however, was the more robust discriminator, separating survival groups where lymphocyte density alone did so only weakly. This may reflect a functional, spatially organised facet of the humoral response, the productive proximity of the two compartments, that abundance alone does not fully capture, and one that is difficult to resolve without automated whole-slide quantification. Furthermore the long-term survival benefit associated with high stromal plasma cell density may reflect sustained output from long-lived, affinity-matured plasma cells generated in germinal-centres [26]. In NSCLC, germinal-centre activity within TLS is linked to local plasma cell generation and favourable long-term survival [9], consistent with this interpretation. We did not assess clonal history or antibody affinity, so the germinal-centre origin remains hypothetical. Macrophage density, which did not differ significantly between treated and therapy-naïve patients, was not prognostic, reinforcing the cross-cohort consistency of the myeloid pattern. Neoadjuvant status itself was associated with shorter survival, which we attribute to the higher stage of the treated group, rather than to the immune differences themselves. These findings gain particular significance from how the treatment of resectable disease has since changed. The patients studied here predate immunotherapy in the neoadjuvant setting. Today, neoadjuvant or perioperative platinum chemotherapy combined with a PD-1/PD-L1 inhibitor is standard of care for resectable NSCLC [18]. Because it is the mature, germinal-centre-bearing TLS and B-cell response that have been linked to checkpoint-inhibitor benefit independently of PD-L1 expression [9–11], our finding that this architecture is selectively less prominent in tumors treated with cytotoxic therapy raises a clinically relevant hypothesis: Prior chemoradiation may deplete the immune substrate on which subsequent immunotherapy depends, with implications for how the two modalities are sequenced and timed. This hypothesis should be interpreted with caution because the literature describes a dual effect of therapy. Radiotherapy can also prime antitumor immunit through immunogenic cell death and type-I interferon signalling, and prior radiation has been linked to improved checkpoint-inhibitor outcomes [27]. The net effect is therefore unlikely to be uniform, depending on modality, dose, and the sequencing and timing of the therapies combined. Our data directly support a more limited conclusion: The stromal plasma-cell compartment, and above all its spatial coupling with tumor lymphocytes, is both lower in tumors treated with cytotoxic therapy and prognostic for survival in the external cohort, a combination that makes it a quantitative, whole-slide-measurable index of the humoral response’s integrity in the residual tumor. Whether this index also reflects the immune substrate available to subsequent immunotherapy remains an open question which would require prospective, immunotherapy-treated cohorts.

Our conclusions are also limited by the timing of sampling. The specimens capture a single, early time point, tissue resected only weeks after therapy, in the window of maximal cytotoxic injury and early repair rather than long-term immune reconstitution. This may explain why this snapshot emphasises disorganisation where longer-term studies have described radiation-induced lymphoid neogenesis. Within this limitation, our data suggest that neoadjuvant cytotoxic therapy does not globally suppress antitumor immunity. Instead, it selectively disrupts an otherwise preserved loco-regional humoral response, at the final step where this response reaches the tumor.

## References

1. Bray F, Laversanne M, Sung H, et al. Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2024;74(3):229–263. doi:10.3322/caac.21834

2. WHO Classification of Tumours Editorial Board. Thoracic Tumours. WHO Classification of Tumours, 5th ed. Lyon: IARC; 2021.

3. National Cancer Institute, Surveillance, Epidemiology, and End Results (SEER) Program. Cancer Stat Facts: Lung and Bronchus Cancer; and Centers for Disease Control and Prevention, U.S. Cancer Statistics — Lung Cancer.

4. Brahmer JR, Tykodi SS, Chow LQM, et al. Safety and activity of anti–PD-L1 antibody in patients with advanced cancer. N Engl J Med. 2012;366(26):2455–2465. doi:10.1056/NEJMoa1200694

5. Yang C, You J, Wang Y, et al. TLS and immune cell profiling: immunomodulatory effects of immunochemotherapy on tumor microenvironment in resectable stage III NSCLC. Front Immunol. 2024;15:1499731. doi:10.3389/fimmu.2024.1499731

6. Fridman WH, Pagès F, Sautès-Fridman C, Galon J. The immune contexture in human tumors: impact on clinical outcome. Nat Rev Cancer. 2012;12(4):298–306. doi:10.1038/nrc3245

7. Sautès-Fridman C, Petitprez F, Calderaro J, Fridman WH. Tertiary lymphoid structures in the era of cancer immunotherapy. Nat Rev Cancer. 2019;19(6):307–325. doi:10.1038/s41568-019-0144-6

8. Siliņa K, Soltermann A, Attar FM, et al. Germinal centers determine the prognostic relevance of tertiary lymphoid structures and are impaired by corticosteroids in lung squamous cell carcinoma. Cancer Res. 2018;78(5):1308–1320. doi:10.1158/0008-5472.CAN-17-1987

9. Germain C, Gnjatic S, Tamzalit F, et al. Presence of B cells in tertiary lymphoid structures is associated with a protective immunity in patients with lung cancer. Am J Respir Crit Care Med. 2014;189(7):832–844. doi:10.1164/rccm.201309-1611OC

10. Fridman WH, Meylan M, Pupier G, Calvez A, Hernandez I, Sautès-Fridman C. Tertiary lymphoid structures and B cells: an intratumoral immunity cycle. Immunity. 2023;56(10):2254–2269. doi:10.1016/j.immuni.2023.08.009

11. Vanhersecke L, Brunet M, Guégan JP, et al. Mature tertiary lymphoid structures predict immune checkpoint inhibitor efficacy in solid tumors independently of PD-L1 expression. Nat Cancer. 2021;2(8):794–802. doi:10.1038/s43018-021-00232-6

12. Fransen MF, van Hall T, Ossendorp F. Immune checkpoint therapy: tumor draining lymph nodes in the spotlights. Int J Mol Sci. 2021;22(17):9401. doi:10.3390/ijms22179401

13. He M, He Q, Cai X, et al. Intratumoral tertiary lymphoid structure (TLS) maturation is influenced by draining lymph nodes of lung cancer. J Immunother Cancer. 2023;11(4):e005539. doi:10.1136/jitc-2022-005539

14. Dieleman S, Kooreman LFS, van Kuijk SMJ, zur Hausen A, Smidt ML, Grabsch HI. Germinal centres within tumor-positive sentinel lymph nodes are positively associated with tumor-infiltrating lymphocytes and tertiary lymphoid structures in breast cancer. Pathol Res Pract. 2024;254:155171. doi:10.1016/j.prp.2024.155171

15. Galluzzi L, Buqué A, Kepp O, Zitvogel L, Kroemer G. Immunogenic cell death in cancer and infectious disease. Nat Rev Immunol. 2017;17(2):97–111. doi:10.1038/nri.2016.107

16. Deng L, Liang H, Xu M, Yang X, Burnette B, Arina A, et al. STING-dependent cytosolic DNA sensing promotes radiation-induced type I interferon-dependent antitumor immunity in immunogenic tumors. Immunity. 2014;41(5):843–852. doi:10.1016/j.immuni.2014.10.019

17. Rodríguez-Ruiz ME, Vitale I, Harrington KJ, Melero I, Galluzzi L. Immunological impact of cell death signalling driven by radiation on the tumour microenvironment. Nat Immunol. 2020;21(2):120–134. doi:10.1038/s41590-019-0561-4

18. Langer T. S3-Leitlinie Prävention, Diagnostik, Therapie und Nachsorge des Lungenkarzinoms. 2024.

19. Belka C, Ottinger H, Kreuzfelder E, et al. Impact of localized radiotherapy on blood immune cells counts and function in humans. Radiother Oncol. 1999;50(2):199–204. doi:10.1016/S0167-8140(98)00130-3

20. Watermann C, Pasternack H, Idel C, et al. Recurrent HNSCC harbor an immunosuppressive tumor immune microenvironment suggesting successful tumor immune evasion. Clin Cancer Res. 2021;27(2):632–644. doi:10.1158/1078-0432.CCR-20-0197

21. Schallenberg S, Dernbach G, Ruane S, et al. AI-powered spatial cell phenomics enhances risk stratification in non-small cell lung cancer. Nat Commun. 2025;16(1):9701. doi:10.1038/s41467-025-65783-z

22. Standvoss K, Hägele M, Krupar R, Ribbat-Idel J, Altschüler J, Erdmann G, et al. Atlas H&E-TME: scalable AI-based tissue profiling at expert pathologist-level accuracy. arXiv:2606.12346 [cs.CV]. 2026. doi:10.48550/arXiv.2606.12346

23. Alber M, Tietz S, Dippel J, Milbich T, Lesort T, Korfiatis P, et al. Atlas: a novel pathology foundation model by Mayo Clinic, Charité, and Aignostics. arXiv:2501.05409 [cs.CV]. 2025. doi:10.48550/arXiv.2501.05409

24. Barmpoutis P, Di Capite M, Kayhanian H, Waddingham W, Alexander DC, Jansen M, Kwong FNK. Tertiary lymphoid structures (TLS) identification and density assessment on H&E-stained digital slides of lung cancer. PLoS One. 2021;16(9):e0256907. doi:10.1371/journal.pone.0256907

25. Becherini C, Lancia A, Detti B, et al. Modulation of tumor-associated macrophage activity with radiation therapy: a systematic review. Strahlenther Onkol. 2023;199(12):1173–1190. doi:10.1007/s00066-023-02097-3

26. Hammarlund E, Thomas A, Amanna IJ, et al. Plasma cell survival in the absence of B cell memory. Nat Commun. 2017;8(1):1781. doi:10.1038/s41467-017-01901-w

27. Shaverdian N, Lisberg AE, Bornazyan K, et al. Previous radiotherapy and the clinical activity and toxicity of pembrolizumab in the treatment of non-small-cell lung cancer: a secondary analysis of the KEYNOTE-001 phase 1 trial. Lancet Oncol. 2017;18(7):895–903. doi:10.1016/S1470-2045(17)30380-7

<table><tr><td>LUSC Cohorts</td><td>neoLUSC (n=20)</td><td>naiveLUSC (n=20)</td></tr><tr><td>Mean age (y)</td><td>61.2 [7.1]</td><td>69.1 [7.1]</td></tr><tr><td>[SD] Gender(%)</td><td></td><td></td></tr><tr><td>m</td><td>15 (75.0)</td><td>15 (75.0)</td></tr><tr><td>T-Stage (%)</td><td></td><td></td></tr><tr><td>1c</td><td>1 (5.0)</td><td>2 (10.0)</td></tr><tr><td>2a</td><td>3 (15.0)</td><td>3 (15.0)</td></tr><tr><td>2b</td><td>6 (30.0)</td><td>6 (30.0)</td></tr><tr><td>3</td><td>6 (30.0</td><td>6 (30.0)</td></tr><tr><td>4</td><td>4 (20.0)</td><td>3 (15.0)</td></tr><tr><td>N-Stage (%)</td><td></td><td></td></tr><tr><td>0</td><td>5 (25.0)</td><td>10 (50.0)</td></tr><tr><td>1</td><td>10 (50.0)</td><td>8 (40.0)</td></tr><tr><td>2</td><td>5 (25.0</td><td>2 (10.0)</td></tr><tr><td>M-Stage (%)</td><td></td><td></td></tr><tr><td>0</td><td>20 (100)</td><td>19 (95.5)</td></tr><tr><td>1a</td><td>0</td><td>1 (5.0)</td></tr><tr><td>R-Status (%)</td><td></td><td></td></tr><tr><td>0</td><td>17 (85.0)</td><td>19 (95.5)</td></tr><tr><td>1</td><td>3 (15.0)</td><td>1 (5.0)</td></tr><tr><td>UICC (%)</td><td></td><td></td></tr><tr><td>I</td><td>0</td><td>3 (15)</td></tr><tr><td>ⅡI</td><td>9 (45.0)</td><td>9 (45.0)</td></tr><tr><td>ⅢII</td><td>11 (55.0)</td><td>7 (35.0)</td></tr><tr><td>IV</td><td>0</td><td>1 (5.0)</td></tr><tr><td>Radiation (%)</td><td>2 (10)</td><td>-</td></tr><tr><td>Chemo(%)</td><td>11 (55.0)</td><td>1</td></tr><tr><td>CRT (%)</td><td>5 (25.0)</td><td>二</td></tr><tr><td>Unclear (%)</td><td>2 (10)</td><td>–</td></tr></table>

A  
![](images/264b555437124679eb670ddc4c4c6bf239758eceef6b7fec57ce1aa79ae6b2fa.jpg)

![](images/1574f4587ecc45b2851ade0bf91b35db25434077f90d171728c3512d61ee049e.jpg)

![](images/773e805452236d47b786474ace6adf7b8dc8ef31c603fe0a155c95fd25e09026.jpg)

![](images/498e711a7acab4d7d38d8c3cd865a5e1299edffcd24137cc0388b6ca330c889c.jpg)

![](images/d12b97d102bfeef749ff746646ac33f8b4cf9668727f9c7f83b5340611a9ee53.jpg)

C  
![](images/02b9e6c0914ea8b551c4660a0051d9181b8c1c13990a941c51e8305e214616e5.jpg)  
D

![](images/f5ebb2814e7f08405a0681996048e8ac38056e3ff536245bf507db190c7a98e5.jpg)

H  
![](images/145900ae5622584297a477303cf8a175f56b7579521e25346480b9167eafa244.jpg)

E  
F  
![](images/d458d2070afad8fe651b5dad7c4cf08b145d06405b9c4b96819d6cc68811962c.jpg)

G  
![](images/ac2871dcf1274868efc27dfe83c5d9eeb4f97f74f1e7c1aa85432e88b770c008.jpg)

![](images/98b2704fb651c3d2b43100ae06e66704fb3d34e77d13e23481f2ed974e68285a.jpg)

Figure 1: Lymphoid structures, immune-cell composition, and microenvironment phenotype in treatment-naive (naivLUSC) and neoadjuvant-treated (neoLUSC) lung squamous cell carcinoma.

(A) Representative H&E regions with tertiary lymphoid structures (TLS) in tumors (rows 1-2, naivLUSC and neoLUSC) and lymphoid follicles (secondary lymphoid structures, SLS, rows 3-4, naivLUSC and neoLUSC) in lymph nodes. Model overlays highlight immature TLS and primary SLS in yellow and mature TLS and SLS with germinal centers in red.

(B) Per-slide tumor TLS quantification by cohort: Total TLS density, immature TLS density, mature TLS density (per mm² tissue), and mature TLS count.

(C) Mature lymphoid area per tissue along the tumor-to-node axis. Mature TLS in tumor (left axis) and mature SLS in tumor-proximal and tumor-distal lymph nodes (right axis) shown as split violins with per-patient connectors. Black circles mark samples shown in A.

(D) Representative tumor H&E regions (row 1 naivLUSC, row 2 neoLUSC) with H&E, tissue segmentation, and cell detection/classification at higher magnification.

(E) Change in tumor micorenvironment (TME) composition (Δ median neoLUSC versus naivLUSC, %) for stromal and intratumoral immune-cell densities (lymphocyte, plasma cell, macrophage, granulocyte) and tissue-area fractions (stroma, carcinoma, vessel, necrosis). Bar direction indicates relative decrease (red) or increase (blue) in neoLUSC tumors compared to naivLUSC tumors.

(F) Density-based immune-phenotype (inflamed, desert, excluded, IDE) assignment from per-slide stromal versus intratumoral (carcinoma) lymphocyte density. Dashed lines indicate classification thresholds.

(G) Proportion of immune phenotypes (hot, excluded, cold) per cohort.

(H) Differential stromal cell–cell co-localization within a 40 µm neighborhood (rank-biserial effect size, neoLUSC versus naivLUSC) for each center–neighbour cell pair.

A  
![](images/3d38887d583a21db3b79907ad5b8703ca32f7f8502ca945e7c5eb7c4f8a60288.jpg)

![](images/098c8249c6b5f4b05ae4d0fbfdf3c7d42767654b89d379eef474c2ea2c4fe634.jpg)

![](images/eed55bd80c0d4c213a3ca2e21a5a1c49b257b0a877e7fa2bf698b0a25dd773e2.jpg)

![](images/457e8132dd520869976a001a55f3de72a5131d7480c4b0278de3d1c23fc0da41.jpg)

Discovery readout correlation structure by cohort (Spearman p)  
B  
![](images/d8cba71590ee2925d19aa6b344d35ec55d1ae9501aae3b40753a13c50477d2a0.jpg)

![](images/5c124eef2d5b02adb4911e2766fef9f9baacebc746b1058ac79fc7812e8b89f4.jpg)  
Figure 2: Cohort-specific correlation structure of plasma cell, tertiary lymphoid structure (TLS), and immune cell density readouts in naivLUSC and neoLUSC patients.

(A) Pairwise Spearman correlations between per tissue normalized readouts, shown as scatter plots with per cohort linear fits (naivLUSC, purple; neoLUSC, pink); each panel is annotated with the pooled and cohort specific correlation coefficient (ρ), p value, and sample size (n) for the following compartment correlations: Plasma cell density in tumor-distal versus tumor-proximal lymph nodes; tumor-proximal lymph node versus peri-TLS plasma cell density, peri-TLS plasma cell density versus stromal lymphocyte density; and peri-TLS plasma cell density versus intratumoral (carcinoma) lymphocyte density. Correlations involving peri-TLS plasma are restricted to TLS-bearing tumours (n = 19 naivLUSC, 18 neoLUSC). The lymph node–ring correlation is further restricted to cases with an evaluable tumor-proximal lymph node (n = 16 naivLUSC, 18 neoLUSC).

(B) Cohort-specific Spearman correlation matrices of the per-tissue–normalized readouts, computed separately within each cohort (naivLUSC, n = 20; neoLUSC, n = 20).

The 18 readouts are arranged into five blocks (dashed separators), in axis order: Mature SLS (SLS area per annotated lymphoid tissue in tumor-near (TN) and tumor-far (TF) lymph node), TLS (mature, immature, and total TLS density), plasma cells in TLS/SLS (plasma cell density tumor-far lymph node, tumor-near lymph node, and peri-TLS ring), stromal TME (endothelial cell, lymphocyte, plasma cell, macrophage, and granulocyte density in stroma) and carcinoma TME (the same five cell densities within carcinoma).

Cell colour encodes the Spearman correlation coefficient ρ from −1 (red) to +1 (blue), as indicated by the colour bar. Correlations are exploratory and not corrected for multiple comparisons.

A  
![](images/2b4fff6eded765a76ace0f06f97745fc4187df3eb35a8cf7b271f37f9def2e76.jpg)

![](images/27d973e28af2ee9adf8d2f4466b6c5079dc7a960405b3305f4b0a43026c5ae79.jpg)

![](images/a8c0aae5537f757ade1c3109aa086a69d04ce19a09e27681af2c3bc39d1f7d8a.jpg)

B  
![](images/e6dadccc08d211400870a32218f5a15881e80f1de569a8a1193ac3400c906727.jpg)

![](images/5770b03d167e250b96da7e54716e731aa313ff6ee9b18d87cfe34050c6bf65c8.jpg)

![](images/6601e8b06f68cce8864a9a04ebb129f26fc5f2301a53057428c33b3e40c156d0.jpg)

C  
![](images/fd2bfefe053a566287f688153591bcdc88220b1d6fd4c2c19ae1cc95f0a7a978.jpg)

D  
![](images/5864dd4b422b429946efbe6b56a81613db848f6b9d7ba8dd5ff7ca117fda03ee.jpg)

E  
![](images/15d8ec5ffb15d4f60fd00c5baeeaa68e35848750778b8a55172ac0db105478f9.jpg)

F  
![](images/fa12e9e53ac03b0357c9e3920a426837a90dfeb87d6edbfa342119c36d38c902.jpg)

G  
![](images/b9db51fe1e32d2b051338332b124d0a5dd8b90cb6471beff8fa2b34636d18493.jpg)  
H

![](images/5ad8d04db725c5dfcb0a814b3f7efc1383d73c10beaf974876cfcdb9b254e793.jpg)

![](images/2fd354739d1e823b7718eaa72c03ca9fad8dde2318986e479c947c0bbee28897.jpg)

![](images/53d89c2a6a3af230f03963d060ec12761d6aded7ecabac3f1030ca54f0478f48.jpg)

## Figure 3: Treatment-associated changes in lymphoid and TME readouts across cohorts and neoadjuvant modalities, and their relationship to overall survival

(A) Cohort comparison of group-summarized, normalized readouts for three readout families, TME (stromal and intratumoral immune cell densities), tumor TLS (total, immature, and mature TLS density and peri-TLS ring macrophage, plasma cell, and lymphocyte density), and secondary lymphoid structures (immature and mature SLS area per lymphoid tissue and lymphoid tissue immune cell densities). NaivLUSC, purple, neoLUSC, pink. (B) The same readout families and axes stratified by neoadjuvant modality (naivLUSC, purple; chemotherapy, orange; chemoradiotherapy [CRT], pink; radiotherapy, green). (C) Immune phenotype (IDE) proportions (hot, excluded, cold) by neoadjuvant modality (naivLUSC, chemotherapy, CRT, radiotherapy).

(D–F) Per-slide distributions by modality (violin plots with individual values and median/quartile lines): (D) Total TLS density per tissue, (E) mature TLS density per tissue, and (F) mature SLS area per lymphoid tissue. (G) Univariable overall survival analysis (Cox proportional-hazards; hazard ratio per +1 SD with 95% confidence interval) for stromal lymphocyte, plasma cell, and macrophage density, intratumoral (carcinoma) lymphocyte density, lymphocyte–plasma cell and plasma cell–lymphocyte colocalization (stroma, 40 µm), age at diagnosis, T stage, post-neoadjuvant N stage, cohort (neoLUSC vs naivLUSC), and sex (male vs female). HR < 1 indicates a protective association.

(H–J) Kaplan–Meier overall survival curves with log-rank P values: (H) by cohort (naivLUSC, n = 14, neoLUSC, n = 18), (I) by tumor TLS density (median split: high, n = 16; low, n = 16), and (J) by peri-TLS ring plasma cell density (median split: high, $\mathsf { n } = 1 5 ; \mathsf { l o w } , \mathsf { n } =$ 15).

A  
B  
![](images/70703b2a9bc622ea776d6b72fc0d79d257f9e2c02b36caf81d6e98e7dd9c6d55.jpg)

C  
![](images/02fe6bc7aec8caab36a9ebf5ae8b1e4939c63ac0c89618dd04620f44bbe2e9b4.jpg)

D  
![](images/3ad24100d0e8d77449530a4691b23d1881d7badab0c59c5fe502eba84d0e258b.jpg)

E  
![](images/a8cb80f9059f9d52795fb0b228dbccfa54c3df8bcafc67bc6c034cbfcd8069f4.jpg)

F  
![](images/96bac07d49fed930a78677905b963d5c94fe1e369ce5a867619e8d1fc0032c03.jpg)  
Figure 4: Independent confirmation of the neoadjuvant-associated stromal immune signature and its prognostic relevance in a large external LUSC cohort.

(A) Cohort contrast (neoadjuvant (n=46) vs non-neoadjuvant (n=412)) for stromal lymphocyte, plasma cell, and macrophage density; intratumoral (carcinoma) lymphocyte density; lymphocyte–plasma cell and plasma cell–lymphocyte colocalization (stroma, 40 µm) in stroma; and clinicopathologic variables (age, sex, UICC stage, pathologic N stage [pTNM], and neoadjuvant status), expressed as rank biserial effect size. Negative values indicate lower values after neoadjuvant therapy. (B) Univariable overall survival analysis for the same variables (Cox proportional hazards; hazard ratio per +1 SD with 95% confidence interval). $\mathsf { H R } < 1$ indicates a protective association. (C) Kaplan–Meier overall survival by cohort (non-neoadjuvant, $\mathsf { n } = 4 1 2 ;$ neoadjuvant, $\mathsf { n } = 4 6 )$

(D–F) Kaplan–Meier overall survival: (D) by median split of stromal lymphocyte density, (E) by stromal plasma-cell density, and (F) by stromal lymphocyte–plasma cell colocalization (40 µm). All Kaplan–Meier comparisons show log-rank p values. Overall survival is shown in months over a 60-month horizon.