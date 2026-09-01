# ImageCAS-X: a dataset and benchmark for coronary artery segmentation and centerline extraction in coronary CT angiography

Kit M. Bransby<sup>1,∗</sup>, Esther Øksnebjerg<sup>1</sup>, Kristofer Kjær<sup>1</sup>, Jacob Kirkeby<sup>1</sup>, Yasmin El Youssef<sup>2</sup>, Aïda Jiménez<sup>1</sup>, Philip R. Pedersson<sup>2</sup>, Martina C. de Knegt<sup>2</sup>, Klaus F. Kofoed<sup>2</sup>, Rasmus R. Paulsen<sup>1,∗</sup>

<sup>1</sup>DTU Compute, Technical University of Denmark, Kongens Lyngby, Denmark <sup>2</sup>Cardiovascular Research Unit, Copenhagen University Hospital – Rigshospitalet, Copenhagen, Denmark

<sup>∗</sup>Corresponding authors: Kit M. Bransby kimbr@dtu.dk & Rasmus R. Paulsen rapa@dtu.dk

Graphical Abstract

![](images/0acf4aceb121515c11305fef7c5622c4929f9b379797548612aca88c7e23ea31.jpg)

## Abstract

Accurate segmentation of the coronary vessel lumen is a prerequisite for quantitative assessment of atherosclerotic plaque and perivascular adipose tissue in coronary computed tomography angiography (CCTA). Cardiologists rely on semi-automated methods for this task because manual vessel tracing and segmentation are labour-intensive. Although many automated methods have been proposed, their validation remains limited by the lack of large, high-quality publicly available datasets. We provide a new dataset of voxel-wise annotations of the vessel lumen and coronary segments, alongside centerlines, and mesh surfaces for 800 scans from the publicly available Image-CAS dataset. Using this dataset, we benchmark established lumen segmentation methods against inter-observer variability, stratifying performance by disease, image quality, coronary dominance, coronary segment, vessel diameter, and lumen attenuation. These labels allow segmentation accuracy to be described in anatomical and clinical context rather than reported as a single aggregate

score. The dataset supports the development and validation of methods for lumen segmentation, plaque and perivascular quantification, and haemodynamic modelling.

## Background & Summary

Coronary computed tomography angiography (CCTA) is the primary non-invasive imaging modality for the assessment of coronary artery disease (CAD). It enables three-dimensional visualisation of the coronary arteries and diferentiation of the lumen, atherosclerotic plaque components, and surrounding perivascular adipose tissue [1, 2]. CCTA is recommended as a first-line diagnostic test for suspected CAD and has an emerging role in quantitative plaque assessment, longitudinal monitoring, and identifying haemodynamically significant lesions [3, 4]. Beyond routine clinical use, CCTA has also driven advances in CAD research through imaging-derived biomarkers such as high-risk plaque features [6, 5, 7], fat attenuation index for perivascular inflammation [8, 9], and computational haemodynamic modelling [10], which are often used in clinical trials and population studies.

These applications rely on anatomically precise delineation of the coronary lumen; however, achieving such contours in CCTA remains technically challenging due to partial volume efects, motion artefacts, blooming artefacts from calcification, and limited spatial resolution [1, 2]. In addition, segmentation algorithms must preserve the continuous, branching topology of the coronary tree, as even small discontinuities can compromise downstream quantitative analysis. Expert annotation is the clinical reference standard, but it is labour-intensive and subject to inter-observer variability, limiting scalability and reproducibility. Despite methodological advances in vessel segmentation algorithms [11, 12, 13, 14, 16], current guidelines continue to recommend semi-automated workflows, reflecting the absence of suficiently validated fully automated solutions [15, 2]. Progress towards this end requires large, high-quality, publicly available datasets that allow for rigorous evaluation, benchmarking, and comparison across methods.

Several CCTA datasets have supported the development of coronary tracing and segmentation methods [17, 18, 19, 20, 21, 14]. Early public benchmarks, such as the MICCAI 2008 coronary artery tracking challenge (CAT08) [17] and the MICCAI 2012 cardiovascular imaging challenge [18], established standardised evaluation protocols for coronary centerline tracing and lumen segmentation. However, with only 32 and 48 cases respectively, these datasets were limited in scale and are no longer publicly available. More recently, the ASOCA dataset [19] released a high-quality label set of lumen segmentations for healthy and diseased patients, but still comprises only 60 cases. Larger cohorts have since been curated. PCCTA120 [20] provided 120 scans with inner and outer wall annotations, yet has not been publicly released. ImageCAS [21], currently the largest dataset with 1,000 scans, released voxel-wise lumen masks but lacks suficient segmentation accuracy for reliable benchmarking and downstream applications (examples provided in Supplementary section 1). To date, no dataset combines large-scale CCTA imaging with high-quality lumen annotations suitable for training and validating vessel segmentation methods.

To address this gap, we provide a large-scale label set for the publicly available ImageCAS dataset, comprising high-quality voxel-wise segmentations of the coronary lumen, alongside mesh surfaces and centerlines. To demonstrate its utility, the label set is used to train, validate, and benchmark state-of-the-art vessel segmentation algorithms, providing a systematic evaluation of their strengths and limitations. Current guidelines recommend that pathological findings be reported per vessel segment [1], reflecting that arteries difer in clinical importance according to the myocardial territory they supply. To support this, we provide additional labels for anatomical segment, disease, and coronary dominance, allowing performance to be assessed in clinically meaningful regions.

This dataset enables several applications previously constrained by the lack of large-scale, highquality lumen annotations. Whereas validation of segmentation pipelines has largely relied on proprietary, industry-held data, the public availability and balanced representation of diseased and non-diseased cases enable independent benchmarking for the first time at this scale. Accurate lumen contours are also a prerequisite for quantifying plaque burden, plaque composition, and perivascular tissue attenuation; because earlier datasets were neither large nor precise enough to validate contouring, our generated data now support the development of plaque and perivascular segmentation algorithms. The dataset also enables validation of automated methods against human performance: by benchmarking algorithms against inter-observer variability, methods can be shown to operate within the range of human agreement which is a prerequisite for deployment in clinical trials and population studies. The accompanying mesh surfaces further enable computational modelling of the coronaries including computed tomography fractional flow reserve (CT-FFR), haemodynamic simulation, and stent planning algorithms that depend on accurate vessel geometry. Beyond coronaryspecific analysis, we anticipate the dataset will be of broader value to the medical imaging and machine learning communities, supporting research in topology-aware segmentation [16, 22] and synthetic data generation [23, 24, 25].

## Methods

## Patient Cohort & Imaging

We retrospectively analysed CCTA data from 1,000 patients collected for the open-source ImageCAS dataset [21] at Guangdong Provincial People’s Hospital between 2012 and 2018. The patients had a documented medical history of ischaemic stroke, transient ischaemic attack, or peripheral artery disease, and included some patients who underwent early revascularisation (within 90 days). The cohort contains both males (n = 586) and females (n = 414), of a minimum age of 18 and average ages of 57.7 and 60.0 years, respectively. All patients underwent CCTA acquired using a Siemens 128-slice dual-source scanner, and either the 30–40% or 60–70% cardiac phase was selected based on optimal coronary image quality. The resulting image volumes contained 512 × 512 × (206–275) voxels, with an in-plane pixel spacing of 0.29–0.43 mm and slice spacing of 0.25–0.45 mm. The original collection of CCTA data received Research Ethics Committee (REC) approval from Guangdong Provincial People’s Hospital, Guangdong Academy of Medical Sciences under Protocol No. 2019324H. It complied with all relevant ethical regulations and the data was anonymised so that sensitive information including name, date of birth, and hospital meta-data were all removed. Our secondary use of this data did not collect any new human-subjects data, and as the data was already anonymised, no ethical approval was required.

## Annotation Protocol

## Software

CCTA analysis was performed using CoronaryExplorer (v0.1), a semi-automatic tool for labelling coronary vessel centerlines and vessel lumen segmentation, developed as a 3D Slicer (v5.10) extension for the purpose of analysing this dataset. The labelling protocol for ImageCAS using CoronaryExplorer is summarised in Fig. 1 and detailed below.

![](images/cceeeae1a06071ae4cbb3ec249277727f70f9544c91801ca35f32ef28c2ca03d.jpg)  
Figure 1: ImageCAS-X data creation pipeline. Top: initial centerlines are generated automatically and manually corrected and named; Middle: the corrected centerlines are used to generate initial lumen contours in cMPR space which are voxelised and manually corrected; Bottom: 3D mesh surface, coronary centerline, lumen segmentations split by coronary segment name, and scan-level descriptors are provided for each patient.

## Image Quality

An expert analyst reviewed all CCTA images and graded each on a five-point Likert scale: nondiagnostic (0), poor (1), adequate (2), good (3), and excellent (4). Images graded non-diagnostic, in which artefacts caused anatomically implausible or ambiguous vessel boundaries, were excluded. In total, 200 scans were excluded owing to motion artefacts (n = 114), step artefacts (n = 76), poor contrast mixing (n = 7), static noise (n = 1), incorrect field of view (n = 1), and file corruption (n = 1). Of the 800 scans retained, 55 were graded poor, 140 adequate, 168 good, and 437 excellent.

## Coronary Dominance

Coronary dominance of each scan was determined where right coronary dominance is defined by the presence of posterior descending artery (PDA) and posterior lateral artery (PLA) from the right coronary artery (RCA), and left dominance is defined by a PDA from the left circumflex artery (LCx). We define co-dominance by the presence of PDAs or PLAs from both RCA and LCx, or the presence of a PDA from the RCA without a PLA from the RCA. The 800 included scans consisted of 729 (91.1%) right dominant, 41 (5.1%) left dominant, and 30 (3.8%) co-dominant cases.

## Coronary Artery Disease

Each scan was assigned a binary label by the analyst to describe if the patient had coronary artery disease. This was achieved by identifying any atherosclerotic lesions composed of calcified, noncalcified or mixed plaque. Of the 800 included scans, 388 (48.5%) were labelled as diseased, and 412 (51.5%) as no disease.

## Centerline Extraction

The selected cases were divided equally among four trained analysts, who used the CoronaryExplorer tool to extract vessel centerlines and annotate the lumen. Generating coronary vessel annotations from scratch is time consuming and impractical at scale, therefore we follow standard workflows [2] by tracing centerlines using a previously validated automated method [26]. The resulting centerlines contained errors and were then manually refined by trimming, removing spurious segments, and drawing any missing vessels, ensuring the final coronary tree conformed to the established 18- segment model [1]. We do not distinguish between proximal, mid and distal segments of the left anterior descending (LAD) and left circumflex (LCx) because they depend on side branch positions which are highly variable, and for consistency we also consider the RCA as a single segment. An additional ‘Other’ category was included for third and forth diagonal or obtuse marginal branches outside of the 18-segment model, which were > 1.8mm in diameter at the most proximal end or as large as the first or second branches. Finally, each branch segment was manually classified according to the 18-segment naming convention. In the Supplementary section 2, we provide further details on the protocol used to discretise variations in coronary anatomy. The total time taken for manual centerline correction was 200 hours.

## Vessel Segmentation

Each extracted centerline was used to generate a curved multiplanar reformatting (cMPR) volume by reparameterising the image volume in a coordinate system aligned with the vessel’s central axis and resampling image intensities onto planes orthogonal to the centerline, producing a straightened representation of the vessel [27]. The in-plane resolution was 0.25 mm with a 16 × 16 mm field of view, and cross-sections were sampled at 0.4 mm intervals along the vessel. The lumen in 100 randomly selected cMPR vessel volumes were manually annotated and used to train and validate a 3D U-Net (further details are provided in Supplementary section 3). The trained network was then applied to all scans to generate initial 3D lumen surface predictions in cMPR volume space. The lumen surface was projected from cMPR space back to the original CCTA volume space and voxelised resulting in a 3D lumen mask. The analysts subsequently reviewed and manually corrected the mask by inspecting each axial slice sequentially. All cases were reviewed by the lead analyst and any errors were corrected. The total time taken for lumen segmentation correction was 270 hours.

## Postprocessing

Segment-labelled centerlines did not always pass through the centre of the corrected lumen contours, so new centerlines were generated from the corrected masks using a skeletonisation method [28], followed by Gaussian smoothing (σ = 0.5 mm) with a sliding window of 5 vertices. The aorta was segmented using TotalSegmentator’s high resolution heart chambers model [29], and start points were assigned to centerline vertices with a degree of 1 lying within 5 mm of the aorta. End points were assigned to all remaining vertices of degree 1, and bifurcation points to vertices with a degree of 3 or more. To recover the segment names, the new centerline tree was partitioned at bifurcation points into segments, which were then matched to the original segment-named centerlines by shortest distance. Segment assignments were manually reviewed by the lead analyst and any errors corrected. Segment names were propagated to every lumen voxel in the segmentation mask by assigning each voxel the name of its nearest centerline point. Mesh surfaces were generated from the corrected masks using the marching cubes algorithm in the Visualization Toolkit (VTK), followed by Taubin smoothing (50 iterations, passband = 0.05). Samples from the dataset are presented in Fig. 2.

![](images/92eb76850d272df7b0939afe8b7b13641ecb358813d142e8b13913fcd0496b70.jpg)  
Figure 2: Sample coronary segments and centerlines for right-, left- and co-dominant trees and diseased versus non-diseased patients.

## Data Subsets

A total of 800 annotated cases were randomly split into training (70%, 560 cases), validation (10%, 80 cases), and test (20%, 160 cases) sets. To assess inter-observer variability, each test case was additionally re-annotated by a diferent analyst, selected at random, following the same protocol without review from the lead analyst and blinded to the first set of labels.

![](images/e9354d44114b71908185935e4abe9531ff5ef42dd852099f8fa929ca608e00a0.jpg)  
Figure 3: Summary of data structures in the ImageCAS-X repository

## Data Records

The ImageCAS-X data generated for this study are publicly accessible via Zenodo (https:// zenodo.org/records/21887809) [30]. The repository is organised by data type into subdirectories for segmentation masks, centerlines, mesh surfaces and file lists. Segmentation masks are stored as compressed NIfTI files in .nii.gz format, and centerlines and mesh surfaces as VTK files in .vtk format. Each centerline carries binary feature attributes that assign start, bifurcation, and end points to every vertex, and a categorical attribute that assigns a coronary segment to each vertex. Coronary segments are also encoded at the voxel level in the segmentation masks, where each voxel is assigned to a discrete class; the mapping from class index to coronary segment name is given in Supplementary section 4. The original ImageCAS volumes [21] are not re-distributed in our repository but are publicly available here: https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas.

Each patient is assigned a unique ID, identical to the original ImageCAS 2023 dataset [21], which prefixes every data file belonging to that patient. Subset membership is indicated by four text files (train, validation, test, exclude) in .txt format. Categorical labels for coronary dominance, disease, and image quality are provided in Descriptors.xlsx. A summary of this repository structure is presented in Fig. 3.

## Technical Validation

## Benchmark

To demonstrate the practical utility and clinical relevance of the data, we trained and benchmarked state-of-the-art lumen segmentation methods. Works from high-impact journals published in the past six years were selected that represent the latest methodological contributions for lumen segmentation in CCTA. These works span three distinct categories: patch-based segmentation (3D-FFR-UNet [12], CAS-Net [11]) where patches are randomly sampled from the full volume, coarse-to-fine segmentation (ImageCAS [21]) where an initial dilated segmentation is used to focus patch segmentation on vessel areas only, and centerline-based segmentation (ADE-HTL [31]) where centerline bifurcation points and connectivity is used to guide lumen segmentation. In addition, we included nnU-Net [32] and Swin-UNETR [33] as general-purpose medical image segmentation methods, nnU-Net with clDice [16] loss as a topology-focused method, and TotalSegmentator [29] as a zero-shot pretrained model. For all methods, we estimated the luminal centerline by skeletonising the voxel based segmentation predictions. Three methods do not provide publicly available code [21, 31, 12]; therefore we reimplemented each based on the written methodology. We excluded CorSegRec [13] and CCA [14] as a reliable implementation was not possible from their written methodology, and DeformCL [34] because it can only handle single branches. Implementation details of all methods are provided in Supplementary section 6.

All experiments were implemented in PyTorch 2.9 and performed on an NVIDIA RTX 5090 GPU with 32 GB of memory. We standardised training and inference by adopting an nnU-Net style pipeline, where common hyperparameters such as preprocessing, resampling resolution, class sampling, augmentation, learning rate, optimiser, postprocessing, and test-time augmentation were fixed. Volumes were isotropically resampled to 0.5 mm and geometric and intensity-based augmentations were applied during training. We defined an epoch as 250 training mini-batches and 50 validation mini-batches, where batch size was maximised to fit the GPU memory, and the total number of epochs was 1,000. An Adam optimiser and polynomial learning rate scheduler were used for all experiments, apart from nnU-Net which used an SGD optimiser. All method-specific hyperparameters such as loss, input size and model configurations were set based on their original implementation. For each method, we tuned the initial learning rate using 0.01, 0.001 and 0.0001, and selected the model with the best validation DSC. The full training and evaluation recipe is provided in Supplementary section 5.

To provide clinical context to the benchmark results and demonstrate the quality of segmentations, we also compared model performance to the inter-observer variability computed from the manually relabelled test set, and to the original ImageCAS labels [21]. Our benchmark and code are available at https://kitbransby.github.io/ImageCAS-X/.

## Statistical Measures

All quantitative results are presented as mean ± standard deviation. To compare segmentations, the Dice similarity coeficient (DSC) was used to measure voxel overlap and 95th percentile Hausdorf distance (HD95) to measure large surface deviations [35], and the Betti number error $( \beta ^ { e r r } )$ computed volumetrically, which compares topology between predicted and ground truth segmentations in two dimensions: the first counts the number of connected components and the second counts the number of loops [36]. Centerline quality was evaluated using clDice [16] to measure topological error of skeletonised segmentations, average symmetric surface distance (ASSD) to describe the overall alignment and HD95 to measure large deviations, and both distances are reported in mm [35].

To measure the efect of lumen attenuation, distal position and vessel diameter on segmentation performance, we sampled local patches centred at diferent positions on the ground truth coronary tree and computed a local DSC for each patch. Specifically, we used every fifth vertex (2 mm) of the ground truth centerline tree to build a $8 \times 8 \times 8$ mm cube centred on that vertex, and the local DSC is computed between the segmentation prediction and ground truth for that patch. The geodesic distance from the ostium, the lumen diameter, the mean lumen Hounsfield unit, and the coronary segment label of each patch was recorded, and patch metrics were binned for each variable within each patient and averaged, to show how the variable afected segmentation performance.

All statistical tests were non-parametric and two-sided, with the patient as the unit of analysis. Method-vs-analyst diferences on the same scans used the Wilcoxon signed-rank test; diferences between independent groups of scans used the Mann–Whitney U test for two groups and the Kruskal– Wallis test for more. Comparison across coronary segments and across ordinal bins were repeated measures since each patient contributed a value to multiple bins, therefore these were assessed with the Skillings–Mack test. Monotonic trends across ordinal groups were described with Spearman’s rank correlation (ρ). Significance was set at $p < 0 . 0 5$

## Quantitative Results

## Inter-observer variability

For the inter-observer variability, the DSC was 92.8, HD95 was 2.46 mm and $\beta ^ { e r r }$ was 0.2 for lumen segmentation, while the clDice was 95.4, ASSD was 0.53 mm and the HD95 was 4.58 mm for centerline extraction (Table 1). When comparing our labels to the human annotators from the original ImageCAS dataset [21] who also segmented the lumen according the 18-segment model, DSC was 41.8, HD95 was 16.15 mm, and $\beta ^ { e r r }$ was 7.0 for lumen segmentation, while the clDice was 78.2, ASSD was 2.23 mm and the HD95 was 18.89 mm for centerline extraction (Table 2). The diferences to our manual annotations are significant $( p < 0 . 0 0 1 )$ and when coupled with qualitative examples (Supplementary Fig. 1) reflect the large improvement in lumen segmentation quality in the new dataset.
<table><tr><td></td><td></td><td></td><td colspan="3">Lumen Segmentation</td><td colspan="3">Lumen Centerline</td></tr><tr><td>Segment</td><td>n</td><td>Agreement↑</td><td>DSC↑</td><td>HD95↓</td><td> $\beta ^ { e r r } \downarrow$ </td><td>clDice↑</td><td>ASSD↓</td><td>HD95↓</td></tr><tr><td>LM</td><td>155</td><td>96.9</td><td> $9 1 . 9 \pm 1 3 . 7$ </td><td> $1 . 4 1 \pm 5 . 5 8$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $9 5 . 6 \pm 1 7 . 6$ </td><td> $0 . 4 0 \pm 1 . 5 2$ </td><td> $1 . 4 5 \pm 5 . 7 9$ </td></tr><tr><td>LAD</td><td>160</td><td>100.0</td><td> $9 2 . 3 \pm 6 . 7$ </td><td> $3 . 3 9 \pm 9 . 0 7$ </td><td> $0 . 1 0 \pm 0 . 3 6$ </td><td> $9 6 . 4 \pm 7 . 9$ </td><td> $0 . 4 8 \pm 1 . 2 9$ </td><td> $3 . 9 0 \pm 9 . 6 7$ </td></tr><tr><td>LCx</td><td>159</td><td>99.4</td><td> $8 4 . 8 \pm 1 9 . 8$ </td><td> $8 . 5 9 \pm 1 5 . 3 0$ </td><td> $0 . 0 8 \pm 0 . 2 9$ </td><td> $8 7 . 3 \pm 2 3 . 0$ </td><td> $1 . 8 5 \pm 3 . 8 6$ </td><td> $9 . 0 2 \pm 1 5 . 8 0$ </td></tr><tr><td>D1</td><td>155</td><td>96.9</td><td> $7 9 . 9 \pm 2 8 . 7$ </td><td> $6 . 6 6 \pm 1 2 . 3 0$ </td><td> $0 . 0 4 \pm 0 . 1 9$ </td><td> $8 5 . 0 \pm 3 0 . 2$ </td><td> $2 . 5 3 \pm 6 . 8 0$ </td><td> $6 . 8 3 \pm 1 2 . 4 0$ </td></tr><tr><td>D2</td><td>91</td><td>89.4</td><td> $8 2 . 9 \pm 2 4 . 3$ </td><td> $3 . 9 6 \pm 6 . 9 2$ </td><td> $0 . 0 2 \pm 0 . 1 5$ </td><td> $8 8 . 1 \pm 2 5 . 3$ </td><td> $1 . 3 6 \pm 3 . 9 7$ </td><td> $4 . 1 0 \pm 7 . 0 1$ </td></tr><tr><td>OM1</td><td>132</td><td>94.4</td><td> $7 4 . 1 \pm 3 2 . 3$ </td><td> $8 . 9 3 \pm 1 5 . 7 0$ </td><td> $0 . 0 5 \pm 0 . 2 1$ </td><td> $7 9 . 8 \pm 3 4 . 2$ </td><td> $3 . 4 7 \pm 8 . 2 7$ </td><td> $9 . 2 1 \pm 1 6 . 0 0$ </td></tr><tr><td>OM2</td><td>46</td><td>87.5</td><td> $7 7 . 7 \pm 2 9 . 1$ </td><td> $5 . 3 8 \pm 9 . 1 6$ </td><td> $0 . 0 2 \pm 0 . 1 5$ </td><td> $8 3 . 7 \pm 3 0 . 6$ </td><td> $2 . 3 3 \pm 6 . 0 4$ </td><td> $5 . 4 0 \pm 9 . 3 8$ </td></tr><tr><td>IM</td><td>43</td><td>89.4</td><td> $8 0 . 6 \pm 2 4 . 5$ </td><td> $5 . 3 2 \pm 1 2 . 7 0$ </td><td> $0 . 2 1 \pm 0 . 5 5$ </td><td> $8 9 . 2 \pm 2 5 . 2$ </td><td> $1 . 2 8 \pm 3 . 9 8$ </td><td> $5 . 4 4 \pm 1 3 . 1 0$ </td></tr><tr><td>RCA</td><td>160</td><td>100.0</td><td> $9 5 . 3 \pm 5 . 0$ </td><td> $2 . 1 8 \pm 6 . 8 1$ </td><td> $0 . 0 8 \pm 0 . 3 3$ </td><td> $9 8 . 2 \pm 6 . 0$ </td><td> $0 . 3 2 \pm 1 . 1 3$ </td><td> $2 . 3 7 \pm 7 . 2 8$ </td></tr><tr><td>R-PDA</td><td>150</td><td>98.8</td><td> $8 2 . 6 \pm 2 1 . 9$ </td><td> $5 . 0 0 \pm 9 . 1 0$ </td><td> $0 . 1 2 \pm 0 . 4 0$ </td><td> $8 8 . 4 \pm 2 2 . 3$ </td><td> $1 . 3 5 \pm 4 . 5 2$ </td><td> $5 . 1 5 \pm 9 . 1 9$ </td></tr><tr><td>R-PLA</td><td>147</td><td>96.2</td><td> $8 3 . 6 \pm 1 8 . 6$ </td><td> $5 . 9 2 \pm 9 . 2 7$ </td><td> $0 . 1 0 \pm 0 . 3 9$ </td><td> $8 9 . 4 \pm 1 8 . 4$ </td><td> $1 . 0 2 \pm 2 . 3 5$ </td><td> $6 . 0 7 \pm 9 . 4 3$ </td></tr><tr><td>L-PDA</td><td>8</td><td>100.0</td><td> $7 5 . 1 \pm 2 9 . 0$ </td><td> $8 . 7 2 \pm 1 8 . 1 0$ </td><td> $0 . 2 5 \pm 0 . 4 3$ </td><td> $8 1 . 6 \pm 3 1 . 8$ </td><td> $2 . 8 2 \pm 6 . 7 6$ </td><td> $8 . 9 9 \pm 1 8 . 3 0$ </td></tr><tr><td>L-PLA</td><td>9</td><td>96.9</td><td> $7 0 . 9 \pm 2 7 . 2$ </td><td> $8 . 3 2 \pm 1 2 . 5 0$ </td><td> $0 . 2 2 \pm 0 . 4 2$ </td><td> $7 7 . 7 \pm 3 0 . 3$ </td><td> $2 . 5 0 \pm 5 . 0 2$ </td><td> $8 . 8 1 \pm 1 3 . 0 0$ </td></tr><tr><td>Other</td><td>14</td><td>88.8</td><td> $8 1 . 3 \pm 1 7 . 0$ </td><td> $1 0 . 1 4 \pm 1 5 . 8 0$ </td><td> $0 . 1 4 \pm 0 . 3 5$ </td><td> $8 6 . 7 \pm 1 6 . 2$ </td><td> $2 . 1 0 \pm 3 . 6 5$ </td><td> $1 1 . 5 5 \pm 1 7 . 0 0$ </td></tr><tr><td>All segments</td><td>160</td><td>95.3</td><td> $9 2 . 8 \pm 3 . 1$ </td><td> $2 . 4 6 \pm 3 . 6 2$ </td><td> $0 . 2 \pm 0 . 4$ </td><td> $9 5 . 4 \pm 3 . 6$ </td><td> $0 . 5 3 \pm 0 . 3 3$ </td><td> $4 . 5 8 \pm 5 . 7 5$ </td></tr></table>

Table 1: Inter-observer segmentation agreement per coronary segment. Agreement is the percentage of scans in which both analysts either annotate that segment or both omit it. Remaining metrics were computed over the n scans in which both analysts annotated the segment. DSC: Dice similarity coeficient (%); $\beta ^ { e r r }$ : Betti number error; HD95: 95th percentile Hausdorf Distance (mm); ASSD: Average symmetric surface distance (mm). Coronary segment names can be found in Supplementary Table 1 and Figure 2.

## Label reliability across scan and vessel characteristics

We analysed the efect of atherosclerotic plaque, scan quality, coronary dominance, coronary segment, lumen attenuation, diameter and distal position on inter-observer variability.

A comparison of DSC performance stratified by the presence of atherosclerotic plaque is presented in Fig. 4(a). Segmentation performance is significantly lower in patients with atherosclerotic plaque than without (91.9 vs. 93.6, $p < 0 . 0 0 1 )$ . The reduction in segmentation performance is expected as plaque tissue components can have similar attenuation to contrast material and large atherosclerotic lesions result in more complex lumen morphology making lumen diferentiation more challenging. To assess the impact of image artefacts on lumen segmentation, we stratify DSC performance by image quality using the previously generated Likert grades (1–4). The results presented in Fig. 4(b) indicate a small positive correlation between DSC and image quality $( \rho = + 0 . 1 3 , p = 0 . 1 1 )$ The correlation is small and not significant reflecting that artefacts typically afect small regions of the scan and therefore only have a relatively small efect on the final segmentation performance. The efect of coronary dominance on segmentation performance is presented in Fig. 4(c). There are no significant diferences between left, right and co-dominance $( p = 0 . 4 0 )$ , which is expected as al coronary dominance types can contain challenging segments.

![](images/9d681f87a877f3e2d5b619b07e6a528c49b53ea93eddae487d9274f2f0bd6501.jpg)

![](images/cb7d74f4066160088beae9a0c4068f915f14242567c451cdb9aa0a873059f717.jpg)

![](images/556f4f27908624ea20db9e501db04f2221c258e6abecceb1f623373c941e05db.jpg)  
Figure 4: Efect of disease (a), image quality (b) and coronary dominance (c) on analyst segmentation performance. Each data point represents a single CCTA scan.

In Table 1, we compare how segmentation performance varies across coronary segment. There is 95.3% agreement between analysts on which segments are present, and high agreement (96.9– 100.0%) for the challenging PDA and PLA segments. There are significant diferences in segmentation performance between the coronary segment for all metrics $( p \textless 0 . 0 0 1 )$ . Specifically, the main branches (LM, LAD, LCx and RCA) were the best segmented $\left( \mathrm { D S C ~ 8 4 . 8 - 9 5 . 3 } \right)$ , while the side branches were consistently worse (70.9–83.6). We reason that this is because they are smaller in diameter and their lumen attenuation is closer to that of the surrounding tissue, making their boundaries less clear. This ambiguity leaves room for interpretation as to how far distally a vesse should be traced and whether a given branch should be included at all, and these judgements can vary between analysts.

In addition, we analyse how local DSC performance varies with lumen attenuation, lumen diameter, and as we move from the aortic ostium distally along the coronary arteries. The results presented in Fig. 5 demonstrate that the local DSC score increases with lumen attenuation $( \rho =$ $+ \ 0 . 9 0 , p < 0 . 0 0 1 )$ , lumen diameter $( \rho = + \ 0 . 8 9 , \ p < 0 . 0 0 1 )$ , and decreases as we move distally along the vessel $( \rho = - \ 0 . 3 6 , p < 0 . 0 0 1 )$ . All three variables can be explained as the concentration of contrast material and the diameter of the vessel decrease distally making diferentiation from fibrous, adipose, or muscular tissue more challenging.

## Comparison between automated methods and human analysts

In Table 2, we present a quantitative comparison of all automated methods alongside inter-observer variability for lumen segmentation and centerline extraction. The best-performing automated method was CAS-Net with a DSC of 91.2, HD95 of 2.99 mm, $\beta ^ { e r r }$ of 1.9 for lumen segmentation, and a clDice of 93.3, ASSD of 0.73 mm and HD95 of 5.75 mm for centerline extraction. A statistically significant diference was found between the inter-observer variability and CAS-Net in DSC $( p < 0 . 0 0 1 )$ , HD95 $( p = 0 . 0 4 )$ and $\beta ^ { e r r } ~ ( p < 0 . 0 0 1 )$ for lumen segmentation, and clDice $( p < 0 . 0 0 1 )$ , ASSD $( p < 0 . 0 0 1 )$ and HD95 $( p = 0 . 0 0 1 )$ for centerline extraction, indicating that automated methods do not yet match human-level agreement on this dataset. We attribute this gap to three factors. First, analysts use surrounding context such as perivascular tissue, plaque and coronary veins to delineate the lumen, while the automated methods can implicitly encode this information they do not receive explicit supervision for these structures. Second, analysts followed a predefined protocol that excluded branches on the basis of the 18-segment model, position within the arterial tree and vessel diameter; these rules are not encoded in the training objective of the automated methods. Third, analysts corrected topological errors arising from image artefacts and ambiguous regions by reasoning about global tree connectivity, a mechanism the automated methods lack. Furthermore, both annotators edited the same automatically generated centerlines and initialised their segmentations using a 3D U-Net which predicted the same boundaries for unedited centerlines, and therefore the DSC inter-observer variability should be considered the upper-bound

![](images/5a638a361049cd4072197d1bbda2398be0c075b884b43bb1bd249eba937e3c4b.jpg)

Figure 5: Efect of lumen attenuation (a), lumen diameter (b) and distal position (c) on analyst segmentation performance. Measured by local DSC on $8 \times 8 \times 8$ mm patch samples.
<table><tr><td rowspan="2"></td><td colspan="4">Lumen Segmentation</td><td colspan="4">Lumen Centerline</td></tr><tr><td>DSC↑</td><td>HD95↓</td><td></td><td> $\beta ^ { e r r } \downarrow$ </td><td>clDice↑</td><td>ASSD↓</td><td></td><td>HD95↓</td></tr><tr><td>TotalSegmentator [29]</td><td> $7 0 . 5 \pm 6 . 2$ </td><td></td><td> $1 9 . 5 9 \pm 6 . 4 5$ </td><td> $4 . 6 \pm 2 . 9$ </td><td> $7 6 . 0 \pm 5 . 3$ </td><td></td><td> $2 . 7 3 \pm 0 . 9 0$ </td><td> $2 3 . 2 6 \pm 7 . 1 7$ </td></tr><tr><td>3D-FFR-UNet [12]</td><td> $8 4 . 9 \pm 5 . 5$ </td><td></td><td> $1 5 . 9 3 \pm 2 0 . 4 2$ </td><td> $4 . 8 \pm 3 . 5$ </td><td> $8 9 . 8 \pm 5 . 0$ </td><td></td><td> $1 . 7 1 \pm 1 . 5 6$ </td><td> $1 7 . 0 0 \pm 1 8 . 2 3$ </td></tr><tr><td>ADE-HTL [31]</td><td> $8 7 . 7 \pm 2 . 8$ </td><td></td><td> ${ \bf 2 . 9 7 \pm 3 . 7 4 }$ </td><td> ${ \bf 1 . 5 \pm 1 . 5 }$ </td><td> ${ \bf 9 3 . 2 \pm 3 . 2 }$ </td><td></td><td> $0 . 7 4 \pm 0 . 3 9$ </td><td> ${ \bf 5 . 6 1 \pm 4 . 8 5 }$ </td></tr><tr><td>Swin-UNETR [33]</td><td> $8 7 . 9 \pm 2 . 7$ </td><td></td><td> $3 . 1 8 \pm 3 . 6 8$ </td><td> $3 . 8 \pm 2 . 3$ </td><td> $9 2 . 5 \pm 3 . 0$ </td><td></td><td> $0 . 7 8 \pm 0 . 3 6$ </td><td> $6 . 1 6 \pm 5 . 1 1$ </td></tr><tr><td>ImageCAS [21]</td><td> $8 7 . 9 \pm 2 . 9$ </td><td></td><td> $4 . 4 5 \pm 4 . 9 0$ </td><td> $4 . 7 \pm 3 . 1$ </td><td> $9 1 . 7 \pm 3 . 5$ </td><td></td><td> $0 . 9 0 \pm 0 . 4 6$ </td><td> $7 . 8 1 \pm 6 . 0 3$ </td></tr><tr><td>nnU-Net [32]</td><td> $8 9 . 8 \pm 3 . 2$ </td><td></td><td> $7 . 0 8 \pm 1 2 . 6 5$ </td><td> $5 . 6 \pm 3 . 5$ </td><td> $9 2 . 3 \pm 3 . 6$ </td><td></td><td> $1 . 0 2 \pm 0 . 7 5$ </td><td> $1 0 . 4 1 \pm 1 3 . 4 1$ </td></tr><tr><td>nnU-Net + clDice [16]</td><td> $9 0 . 0 \pm 3 . 5$ </td><td></td><td> $9 . 7 0 \pm 1 5 . 3 6$ </td><td> $8 . 0 \pm 4 . 4$ </td><td> $9 1 . 7 \pm 3 . 9$ </td><td></td><td> $1 . 2 0 \pm 0 . 9 9$ </td><td> $1 2 . 9 5 \pm 1 4 . 2 0$ </td></tr><tr><td>CAS-Net [11]</td><td> ${ \bf 9 1 . 2 \pm 2 . 8 }$ </td><td></td><td> $2 . 9 9 \pm 3 . 4 7$ </td><td> $1 . 9 \pm 1 . 5$ </td><td> $9 3 . 3 \pm 3 . 2$ </td><td></td><td> ${ \bf 0 . 7 3 \pm 0 . 3 6 }$ </td><td> $5 . 7 5 \pm 4 . 9 8$ </td></tr><tr><td>Inter-observer</td><td> $9 2 . 8 \pm 3 . 1$ </td><td></td><td> $2 . 4 6 \pm 3 . 6 2$ </td><td> $0 . 4 \pm 0 . 4$ </td><td> $9 5 . 4 \pm 3 . 6$ </td><td></td><td> $0 . 5 3 \pm 0 . 3 3$ </td><td> $4 . 5 8 \pm 5 . 7 5$ </td></tr><tr><td>ImageCAS (labels) [21]</td><td> $4 1 . 8 \pm 6 . 7$ </td><td></td><td> $1 6 . 1 5 \pm 8 . 2 5$ </td><td> $7 . 0 \pm 6 . 7$ </td><td> $7 8 . 2 \pm 6 . 9$ </td><td></td><td> $2 . 2 3 \pm 0 . 9 4$ </td><td> $1 8 . 8 9 \pm 8 . 9 1$ </td></tr></table>

Table 2: Quantitative comparison of vessel segmentation algorithms (top) and expert analysts (bottom). Bold indicates the best performing algorithm. DSC: Dice similarity coeficient $( \% ) ; \beta ^ { e r r }$ Betti number error; HD95: 95th percentile Hausdorf Distance (mm); ASSD: Average symmetric surface distance (mm).

![](images/d7555c5eca5c326717b741b8009ffda43a93bd2bd943903007016eccad1ccd32.jpg)  
Figure 6: Qualitative comparison between model predictions using three examples that reflect worst, median and best model performance.

of agreement.

A statistically significant diference was found between CAS-Net and the next best-performing method (ADE-HTL) for the DSC metric $( p \textless 0 . 0 0 1 )$ and $\beta ^ { e r r } ~ ( p ~ < ~ 0 . 0 0 1 )$ , but not for HD95 $( p = 0 . 8 7 )$ in the lumen segmentation task, and clDice metric $( p = 0 . 9 4 )$ , ASSD $( p = 0 . 7 4 )$ and HD95 $( p = 0 . 9 0 )$ in the centerline extraction task. TotalSegmentator was the only pretrained model and performed the worst $( p < 0 . 0 0 1$ , compared to CAS-Net), which could be attributed to the quality of the annotations it was trained on, the annotation protocol used, or a data distribution shift between its training data and ImageCAS.

In Fig. 6, we present three qualitative comparisons of lumen segmentation using examples that reflect the worst (5th percentile), median and best (95th percentile) model performance. These examples demonstrate that topological errors such as vessel breaks are present in all model predictions despite high DSC and clDice.

In Table 3 we compare the computational eficiency of all methods and the analysts at inference time. CAS-Net is also the most computationally eficient. All models predicted the lumen segmentation in < 2 minutes which is significantly faster than the analysts who averaged 35 minutes per scan.

## Data Availability

The dataset is available to download via Zenodo (https://zenodo.org/records/21887809) [30]. The dataset generated in this study is subject to a CC BY 4.0 license. The volumes from original ImageCAS are not re-distributed and are provided by the original authors via Kaggle (https: //www.kaggle.com/datasets/xiaoweixumedicalai/imagecas) under the Apache 2.0 license.

<table><tr><td></td><td>#Params (M)</td><td>Memory (GB)</td><td>#Models</td><td>Inference (s)</td></tr><tr><td>TotalSegmentator [29]</td><td>88.6</td><td>10.9</td><td>1</td><td>12.0</td></tr><tr><td>3D-FFR-UNet [12]</td><td>17.7</td><td>2.7</td><td>2</td><td>38.6</td></tr><tr><td>ADE-HTL [31]</td><td>10.7</td><td>2.7</td><td>1</td><td>41.6</td></tr><tr><td>Swin-UNETR [33]</td><td>70.2</td><td>8.5</td><td>1</td><td>23.7</td></tr><tr><td>ImageCAS [21]</td><td>23.3</td><td>25.9</td><td>3</td><td>14.9</td></tr><tr><td>nnU-Net [32]</td><td>153.9</td><td>2.5</td><td>5</td><td>102.3</td></tr><tr><td>nnU-Net + clDice [16]</td><td>153.9</td><td>2.5</td><td>5</td><td>102.3</td></tr><tr><td>CAS-Net [11]</td><td>6.8</td><td>9.1</td><td>1</td><td>15.5</td></tr><tr><td>Analysts</td><td>一</td><td>-</td><td>-</td><td>2100.0</td></tr></table>

Table 3: Comparison of inference time eficiency between methods. #Params is the number of trainable parameters summed across all models, memory is the peak GPU memory, and inference time is measured from preprocessed volume to final prediction. M: Million, GB: Gigabytes, s: seconds.

## Code Availability

Training and benchmarking code is available at https://github.com/kitbransby/ImageCAS-X.   
Pretrained weights are available via https://zenodo.org/records/21887809.

## References

[1] Leipsic, J., Abbara, S., Achenbach, S., Cury, R., Earls, J.P. et al. SCCT guidelines for the interpretation and reporting of coronary CT angiography: A report of the Society of Cardiovascular Computed Tomography Guidelines Committee. Journal of Cardiovascular Computed Tomography 8, 342–358 (2014).

[2] Nieman, K., García-García, H.M., Hideo-Kajita, A., Collet, C., Dey, D. et al. Standards for quantitative assessments by coronary computed tomography angiography (CCTA). Journal of Cardiovascular Computed Tomography 18, 429–443 (2024).

[3] Knuuti, J., Wijns, W., Saraste, A., Capodanno, D., Barbato, E. et al. 2019 ESC Guidelines for the diagnosis and management of chronic coronary syndromes: The Task Force for the diagnosis and management of chronic coronary syndromes of the European Society of Cardiology (ESC). European heart journal 41, 407–477 (2020).

[4] Writing Committee Members, Gulati, M., Levy, P.D., Mukherjee, D., Amsterdam, E. et al. 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR guideline for the evaluation and diagnosis of chest pain: a report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines. Journal of the American College of Cardiology 78, e187–e285 (2021).

[5] Williams, M.C., Moss, A.J., Dweck, M., Adamson, P.D., Alam, S. et al. Coronary Artery Plaque Characteristics Associated With Adverse Outcomes in the SCOT-HEART Study. Journal of the American College of Cardiology 73, 291–301 (2019).

[6] Williams, M.C., Kwiecinski, J., Doris, M., McElhinney, P., D’Souza, et al. Low-attenuation noncalcified plaque on coronary computed tomography angiography predicts myocardial in-

farction: results from the multicenter SCOT-HEART trial. Circulation 141(18) pp.1452-1462 (2020)

[7] Thomsen, C. & Abdulla, J. Characteristics of high-risk coronary plaques identified by computed tomographic angiography and associated prognosis: a systematic review and meta-analysis. European Heart Journal-Cardiovascular Imaging 17, 120–129 (2016).

[8] Oikonomou, E.K., Marwan, M., Desai, M.Y., Mancio, J., Alashi, A. et al. Non-invasive detection of coronary inflammation using computed tomography and prediction of residual cardiovascular risk (the CRISP CT study): a post-hoc analysis of prospective outcome data. The Lancet 392, 929–939 (2018).

[9] Antonopoulos, A.S., Sanna, F., Sabharwal, N., Thomas, S., Oikonomou, E.K. et al. Detecting human coronary inflammation by imaging perivascular fat. Science Translational Medicine 9, eaal2658 (2017).

[10] Nørgaard, B.L., Leipsic, J., Gaur, S., Seneviratne, S., Ko, B.S. et al. Diagnostic Performance of Noninvasive Fractional Flow Reserve Derived From Coronary Computed Tomography Angiography in Suspected Coronary Artery Disease. Journal of the American College of Cardiology 63, 1145–1155 (2014).

[11] Dong, C., Xu, S., Dai, D., Zhang, Y., Zhang, C. et al. A novel multi-attention, multi-scale 3D deep network for coronary artery segmentation. Medical Image Analysis 85, 102745 (2023).

[12] Song, A., Xu, L., Wang, L., Wang, B., Yang, X. et al. Automatic coronary artery segmentation of CCTA images with an eficient feature-fusion-and-rectification 3D-UNet. IEEE journal of biomedical and health informatics 26, 4044–4055 (2022).

[13] Qiu, Y., Shan, D., Wang, Y., Dong, P., Wu, D. et al. A topology-preserving three-stage framework for fully-connected coronary artery extraction. Medical Image Analysis 103, 103578 (2025).

[14] Yang, X., Xu, L., Yu, S., Xia, Q., Li, H. et al. Segmentation and vascular vectorization for coronary artery by geometry-based cascaded neural network. IEEE Transactions on Medical Imaging 44, 259–269 (2024).

[15] Chandrashekhar, Y., Blankstein, R., Shaw, L.J., Ferencik, M., Leipsic, J., et al. Quantitative Coronary Plaque Analysis Symposium Collaborators, 2025. Quantitative coronary plaque analysis in clinical practice: 2025 ACC scientific statement: a report of the American College of Cardiology. JACC: Cardiovascular Imaging (2025)

[16] Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A. et al. clDice-a novel topologypreserving loss function for tubular structure segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 16560–16569 (2021).

[17] Metz, C., Schaap, M., van Walsum, T., van der Giessen, A., Weustink, A. et al. 3D segmentation in the clinic: A grand challenge II-coronary artery tracking. Insight Journal 1, 6 (2008).

[18] Kirişli, H., Schaap, M., Metz, C., Dharampal, A., Meijboom, W.B. et al. Standardized evaluation framework for evaluating coronary artery stenosis detection, stenosis quantification and lumen segmentation algorithms in computed tomography angiography. Medical image analysis 17, 859–876 (2013).

[19] Gharleghi, R., Adikari, D., Ellenberger, K., Webster, M., Ellis, C. et al. Annotated computed tomography coronary angiogram images and associated data of normal and diseased arteries. Scientific Data 10, 128 (2023).

[20] Tu, R., Tian, C., Wang, L., Deng, Y., Chen, C. et al. Mask SAM 3D for coronary artery and plaque segmentation in CCTA images. International Journal of Computer Assisted Radiology and Surgery (2025).

[21] Zeng, A., Wu, C., Lin, G., Xie, W., Hong, J. et al. ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images. Computerized Medical Imaging and Graphics 109, 102287 (2023).

[22] Clough, J.R., Byrne N., Oksuz I., Zimmer V.A., Schnabel J.A., et al. A topological loss function for deep-learning based image segmentation using persistent homology. IEEE transactions on pattern analysis and machine intelligence. 8766-8778 (2020).

[23] Bransby, K.M., Tufaro, V., Cap, M., Slabaugh, G., Bourantas, C. et al. 3D coronary vessel reconstruction from bi-plane angiography using graph convolutional networks. IEEE 20th International Symposium on Biomedical Imaging (ISBI) (2023)

[24] Wang, Y., Banerjee, A., Choudhury, R.P. and Grau, V., DeepCA: Deep Learning-Based 3D Coronary Artery Tree Reconstruction from Two 2D Non-Simultaneous X-Ray Angiography Projections. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV ) 337-346 (2025)

[25] Feldman, P., Sinnona, M., Delrieux, C., Siless, V. and Iarussi, E., VesselGPT: Autoregressive Modeling of Vascular Geometry. In International Conference on Medical Image Computing and Computer-Assisted Intervention 662-672 (2025)

[26] Baldachowski, M. & Korona, M. Tracing the Coronary Arteries in 3D Computed Tomography Scans. Master’s thesis, Technical University of Denmark (DTU Compute) (2024).

[27] Kanitsar, A., Fleischmann, D., Wegenkittl, R., Felkel, P. and Groller, E., CPR-curved planar reformation. IEEE Conference on Visualization 37-44 (2002)

[28] Lee, T. C., Kashyap, R. L., & Chu, C. N. Building skeleton models via 3-D medial surface axis thinning algorithms. CVGIP: graphical models and image processing, 56(6), 462-478. (1994).

[29] Wasserthal, J., Breit, H.C., Meyer, M.T., Pradella, M., Hinck, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiology: Artificial Intelligence 5(5), 2023.

[30] Bransby, K. M. & Paulsen, R. R. ImageCAS-X. Zenodo https://doi.org/10.5281/zenodo.21887809 (2026)

[31] Zhang, X., Sun, K., Wu, D., Xiong, X., Liu, J. et al. An anatomy-and topology-preserving framework for coronary artery segmentation. IEEE Transactions on Medical imaging 43, 723–733 (2023).

[32] Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J. & Maier-Hein, K.H. nnU-Net: a selfconfiguring method for deep learning-based biomedical image segmentation. Nature methods 18, 203–211 (2021).

[33] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R. et al. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI brain lesion workshop 272-284 (2021).

[34] Zhao, Z., Zhang, Z., Liu, Y., Zhang, Z., Yu, H., et al. DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image. In Proceedings of the Computer Vision and Pattern Recognition Conference 30896-30905 (2025).

[35] Taha, A.A. and Hanbury, A., Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC medical imaging 15(1), 29 (2015)

[36] Wasserman, L. Topological data analysis. Annual review of statistics and its application 5, 501–532 (2018).

## Author Contributions

Conceptualization: K.M.B, R.R.P, K.F.K; Methodology: K.M.B; Software: K.M.B, R.R.P; Validation: K.M.B; Formal analysis: K.M.B; Investigation: K.M.B; Data curation: K.M.B, E.Ø, K.K, J.K, A.J; Supervision: R.R.P, K.F.K, P.R.P, Y.E.Y, M.d.K; Project administration: R.R.P, K.F.K; Funding acquisition: R.R.P, K.F.K; Writing – original draft: K.M.B; Writing – review & editing: all authors

## Competing Interests

K.F.K has received research grants from AP Møller og hustru Chastine McKinney Møllers Fond, Novo Nordisk Foundation, Novo Nordic A/S, Sygeforsikringen Danmark, Research Council of Rigshospitalet, The University of Copenhagen, Canon Medical Systems, GE Healthcare. Remaining authors declare no competing interests.

## Funding

Funding for this research was provided by Novo Nordisk A/S. The funder had no role in study design, data collection, analysis, or preparation of the manuscript.

# ImageCAS-X - Supplementary Material

## A Segmentation accuracy in the ImageCAS dataset

Three visual examples are presented in Supplementary Fig. 7 to explain why the ImageCAS (2023) dataset is not suitable for training and validation of lumen segmentation methods in CCTA.

![](images/2c59fa8f72286586049cd3473f6e41b34d45759f47a641b2c65bdb4f5b149106.jpg)  
Figure 7: Segmentation issues on ImageCAS (2023) dataset (patient 4) such as the inclusion of atherosclerotic plaque (A), false positive pulmonary blood vessels (B), false positive coronary veins (C).

## B Coronary segment labelling

In this section we provide further detail on how the coronary trees were labelled and how ambiguous cases were resolved.

Side branches were not traced where the connection to the parent vessel was unclear, where the branch could not be clearly delineated, or where the diameter at the most distal point was <1 mm. Septal perforators, acute marginals and nodal arteries were not traced. Additional diagona (D) and obtuse marginal (OM) branches, such as D3, D4, OM3 and OM4, were labelled as ‘Other if they were at least as large as the first or second branches or >1.8 mm in diameter at the most proximal end. Where a side branch itself bifurcated, only the larger branch was traced, unless the two branches were of equal size or the smaller was >1.8 mm in diameter, in which case both were traced. The same rule was applied where multiple posterior descending (PDA) or posterior lateral (PLA) arteries were present.

Where a main branch bifurcated, a decision was required as to which branch was the side branch and which was the continuation of the main vessel. In all such cases this was determined by the direction and course of the artery rather than by its size. For example, where the LCx bifurcated, the branch travelling towards the left ventricular wall was labelled the OM and the branch continuing along the left atrioventricular groove was labelled the LCx, even where the OM was the larger of the two. The same argument was applied to the LAD where the branch that travelled along the anterior interventricular groove (AIVG) towards cardiac apex was labelled as the LAD, while the branch that travelled to left ventricular wall was labelled as the diagonal branch. Supplementary Fig. 8 illustrates the labelling decisions applied to anatomical variation in the LCx, the most challenging branch.

![](images/1c2044fd155eabd4161c2530c12feabeb8b00ed91e50d34704bacdd1574f4e4d.jpg)  
Figure 8: Labelling guide for anatomical variation in the left circumflex artery.

## C Initial lumen generation

Due to the significant labour involved in segmenting the coronary lumen from scratch, a 3D U-Net was trained to generate initial contour segmentations. Lumen contours were manually drawn in the cMPR volume from 80 randomly selected vessels from the train set, and a further 20 randomly selected vessels from the validation set were used to select the best model. The model was based on the nnU-Net architecture with five levels with feature dimensions: [32, 64, 128, 256, 512], resulting in 22.2M parameters. Random crops from cMPR volume of size N × H × W were used as input, where N was the number of cMPR frames set at 10 with 0.4mm spacing between frames, and H and W were the height and width of the plane, set at 64 representing a 16 × 16mm cross section using 0.25 mm pixel spacing. The model was trained using the same recipe and hardware described in the benchmark section of the main paper. At inference, a sliding window was used with step size of 1, and the probabilities from multiple windows were averaged to obtain a final prediction for each frame.

## D Coronary segment class indices

Mapping between segment class index and segment name used in this study is presented in Supplementary Table 4.

## E Shared training and inference framework

All methods are trained and evaluated through a single shared framework that fixes everything which must remain identical for a fair comparison (preprocessing, augmentation, optimiser, schedule, training budget, post-processing, test-time augmentation and metrics), while retaining method-

<table><tr><td>Branch</td><td>Segment</td><td>Class ID</td></tr><tr><td>Left main</td><td>LM</td><td>1</td></tr><tr><td>Left anterior descending artery</td><td>LAD</td><td>2</td></tr><tr><td>Left circumflex artery</td><td>LCx</td><td>3</td></tr><tr><td>First diagonal branch</td><td>D1</td><td>4</td></tr><tr><td>Second diagonal branch</td><td>D2</td><td>5</td></tr><tr><td>First obtuse marginal branch</td><td>OM1</td><td>6</td></tr><tr><td>Second obtuse marginal branch</td><td>OM2</td><td>7</td></tr><tr><td>Intermediate marginal (ramus intermedius)</td><td>IM</td><td>8</td></tr><tr><td>Right coronary artery</td><td>RCA</td><td>9</td></tr><tr><td>Right posterior descending artery</td><td>R-PDA</td><td>10</td></tr><tr><td>Right posterior lateral artery</td><td>R-PLA</td><td>11</td></tr><tr><td>Left posterior descending artery</td><td>L-PDA</td><td>12</td></tr><tr><td>Left posterior lateral artery</td><td>L-PLA</td><td>13</td></tr><tr><td>Other</td><td>D3, D4, OM3, or OM4</td><td>14</td></tr></table>

Table 4: Coronary segment names mapped to segmentation indices

specific components: architecture, loss, input sampling strategy, batch size and learning rate. All experiments use a fixed random seed. The shared components are described in the following sections.

## E.1 Inputs & Sampling

Every scan is resampled to isotropic 0.5 mm spacing, and intensities are clipped to a [−200, 1000] HU window and linearly mapped to [0, 1]. The framework supports three input types: whole volume where the full preprocessed volume is passed to the network; random crop where a fixed-size patch is cropped at a random location from the whole volume; and center crop where patches are extracted at precomputed coordinates (e.g. points predicted by an earlier stage) rather than at random locations. For all patch-based methods that take a random patch from each image, foreground (fg) sampling is used to ensure at least 50% of every batch contains patches with lumen class. Without this, small patches $( 6 4 ^ { 3 } – 9 6 ^ { 3 } )$ land on vessel so rarely under uniform cropping that batches are almost entirely background resulting in noisy gradients during backprop.

## E.2 Augmentation

Augmentation is applied during training and is based on nnU-Net’s default pipeline: additive Gaussian noise $( p { = } 0 . 1 , \ \sigma ^ { 2 } \in [ 0 , 0 . 0 1 ] )$ , Gaussian blur $( p { = } 0 . 2 , \sigma \in [ 0 . 5 , 1 . 0 ] )$ , multiplicative brightness $( p \mathrm { { = } } 0 . 1 5 , [ 0 . 7 5 , 1 . 2 5 ] )$ ), contrast $\left( p { = } 0 . 1 5 , [ 0 . 7 5 , 1 . 2 5 ] \right)$ , simulated low resolution $\left( p \mathrm { = } 0 . 2 5 \right.$ , downsample factor [0.5, 1.0]), inverted gamma $( p { = } 0 . 1 , \gamma \in [ 0 . 7 , 1 . 5 ] )$ and gamma $( p { = } 0 . 3 , \gamma \in [ 0 . 7 , 1 . 5 ] )$ . Axis flipping over coronal and sagittal planes are included, but rotations and axial flipping are omitted since coronary anatomy has a genuine superior–inferior orientation that the network should not be trained to treat as ambiguous.

## E.3 Training schedule

An epoch is defined as 250 training and 50 validation batches drawn with replacement. All methods are trained for the same number of epochs with Adam (weight decay $1 0 ^ { - 8 } )$ and a polynomial learning-rate decay (power 0.9), apart from nnU-Net which used SGD. Learning rate and batch size are the only optimisation hyperparameters set per method, so that each can be tuned to its own architecture and memory footprint. Checkpoints are selected on validation performance, by default the mean lumen Dice over validation batches.

## E.4 Inference

For inference, whole-volume methods used a single forward pass, while patch- and crop-based methods are tiled with a sliding window of the method’s own training patch size at 50% overlap where overlapping predictions are combined by a Gaussian-weighted average with per-axis $\sigma = 0 . 1 2 5 \times$ patch size. Inference for centre-driven patch methods occurs only at precomputed centres. Axis mirroring test-time augmentation is applied uniformly to every method: predictions are averaged over all four flip combinations of the two in-plane axes, matching the flip axes used during training. Predicted logits are converted to probabilities and resampled back onto each scan’s original acquisition grid by linear interpolation before discretisation. Post-processing is then applied at full original resolution and is identical for all methods: threshold at 0.5, followed by removal of connected components smaller than 100 voxels. Component removal is size-based rather than keep-largest-N, as the LCx and LAD can connect directly to the aortic annulus resulting in variable number of true components.

## F Method-specific implementation notes

In Supplementary Table 5, we summarise the final hyperparameters chosen for each baseline method. Further implementation details for each method can be found below.
<table><tr><td>Method</td><td>Stage</td><td>Batch Size</td><td>Learning Rate</td><td>Loss</td></tr><tr><td>3D-FFR-UNet</td><td></td><td>4</td><td>0.0001</td><td>Dice</td></tr><tr><td>ADE-HTL</td><td>ADE</td><td>2</td><td>0.0001</td><td>Dice, CE</td></tr><tr><td></td><td>HTL</td><td>4</td><td>0.0001</td><td>Dice, CE, Weighted HD, MSE</td></tr><tr><td>Swin-UNETR</td><td>一</td><td>4</td><td>0.0001</td><td>Dice</td></tr><tr><td>ImageCAS</td><td>Coarse</td><td>8</td><td>0.0001</td><td>Dice</td></tr><tr><td></td><td>Dilated</td><td>8</td><td>0.0001</td><td>Weighted Similarity</td></tr><tr><td></td><td> $1 6 ^ { 3 }$ </td><td>512</td><td>0.0001</td><td>Dice [DS]</td></tr><tr><td></td><td> $3 2 ^ { 3 }$ </td><td>64</td><td>0.0001</td><td>Dice [DS]</td></tr><tr><td></td><td> $6 4 ^ { 3 }$ </td><td>8</td><td>0.0001</td><td>Dice [DS]</td></tr><tr><td>nnU-Net</td><td></td><td>2</td><td>0.01</td><td>Dice, CE [DS]</td></tr><tr><td>nnU-Net + clDice</td><td></td><td>2</td><td>0.01</td><td>Dice, CE, clDice [DS]</td></tr><tr><td>CAS-Net</td><td></td><td>4</td><td>0.0001</td><td>Dice</td></tr></table>

Table 5: Hyperparameter settings for baseline methods. CE: cross-entropy; DS: deep supervision; HD: Hausdorf distance; MSE: mean squared error.

## F.1 3D-FFR-UNet

The authors propose two steps to first automatically identify axial slices containing coronary arteries which is used to crop the volume which is then patched to $6 4 ^ { 3 }$ and fed into a second model. While this first step is reasonable, it is not necessary if foreground over-sampling is applied during training. As this is implemented in our patch data loader already, we skip this step. The authors experiment with many loss functions, and we follow their recommendations to use dice loss.

## F.2 CAS-Net

We find suboptimal results when using their proposed loss recipe on our data and instead we train with a pure dice loss which gives good results.

## F.3 ADE-HTL

Training their method on our hardware is slow and limited to a batch size of 1. Therefore we use key-value average pooling before their BAI module which shrinks the key set by 64×, and train with mixed precision. The authors use a weighted cross-entropy loss, but the weighting parameters are not given. As using the wrong weighting parameters is likely to result in wildly diferent predictions, we simplify by using dice and cross-entropy loss for the connectivity prediction. As dice is robust to class-imbalance, this is a reasonable solution.

## F.4 ImageCAS (2023)

When ensembling final models, we ensemble the probabilities rather than majority voting on the binary segmentation masks as probabilities are richer representations. We find that excluding the stage 1 coarse segmentation model from the ensemble improves performance.