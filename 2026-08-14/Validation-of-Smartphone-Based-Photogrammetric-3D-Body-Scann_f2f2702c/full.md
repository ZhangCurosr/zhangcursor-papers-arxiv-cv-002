Article

# Validation of Smartphone-Based Photogrammetric 3D Body Scanning for Automated Anthropometric Measurements Compared with a Commercial Depth-Sensor-Based Body Scanner

Ruting Cheng <sup>1</sup>\* , Boyuan Feng <sup>1</sup> , Chuhui Qiu <sup>1</sup> , Joaquin A. Calderon <sup>2</sup> , Qing Pan <sup>3</sup> , Yufan Liu <sup>4</sup> and James K. Hahn <sup>1</sup>

<sup>1</sup> Department of Computer Science, The George Washington University, Washington DC, 20052, USA; fby@gwu.edu (B.F.); chqiu@gwu.edu (C.Q.); hahn@gwu.edu (J.H.)

2 Department of Obstetrics and Gynecology, The George Washington University, Washington DC, 20052, USA; joacocal91@gmail.com;

Department of Biostatistics and Bioinformatics, The George Washington University, Washington DC, 20052, USA;   
qpan@gwu.edu;

School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR; yufan.liu@connect.hku.hk

Correspondence: rcheng77@gwu.edu;

## Abstract

3D body scanning has become an important tool in healthcare applications because of its rapid and non-invasive nature. While smartphone-based photogrammetric reconstruction provide a low-cost and accessible alternative to commercial 3D body scanners, their performance for whole-body scanning remains insufficiently validated. Thus, we designed this study to comprehensively validate the photogrammetric 3D scanning application by evaluating automatically extracted whole-body measurements and longitudinal body-shape monitoring. We evaluated a representative application, PolyCam, against the commercial depth-sensor-based Fit3D ProScanner using 144 pregnant participants scanned longitudinally throughout pregnancy. We designed an automatic circumference extraction pipeline to get measurements at four anatomical landmarks from paired 3D scans. A linear mixed-effects model was used to evaluate scanner effects and longitudinal body-shape changes. Measurement consistency was assessed using repeated PolyCam scans and tape measurements on a rigid mannequin. PolyCam demonstrated strong agreement with Fit3D, with average biases below 16 mm, intraclass correlation coefficients above 0.8, and Pearson correlation coefficients above 0.9 across all landmarks. Both systems captured comparable longitudinal body-shape changes. Mannequin experiments showed mean biases below 3.5 mm and no significant differences from tape measurements. These findings support smartphone photogrammetry as a potential accessible alternative to commercial body scanners and applicable for longitudinal 3D body-shape assessment.

Keywords: 3D body scanning, photogrammetric reconstruction, smartphone-based scanning, digital health

## 1. Introduction

With the development of 3D scanning technology, 3D body surface scanning is becoming prevalent not only in the fashion and gaming field, but also in the medical and clinical domains [1–7]. Compared with traditional anthropometric measurements, 3D body shape provides substantially richer geometric information while demonstrating considerable value in body composition estimation, nutritional assessment, and the identification of obesity-related health conditions[8–13]. More recently, 3D body shape analysis has shown promise in obstetrics for fetal growth estimation and cesarean delivery prediction[14–17], in orthopedics for scoliosis assessment[18–20], and in sports science for training optimization[21–23]. In addition, the ability of 3D scanning to preserve both surface geometry and color information has attracted growing interest in medico-legal applications [24,25].

The widespread adoption of 3D body scanning in medical and home-based settings depends largely on its affordability, accessibility, safety, and ease of operation. However, most existing studies rely on commercial depth-sensor-based body scanners, such as Fit3D ProScanner (Fit3D, San Francisco, CA, USA) and Styku (Styku LLC, Los Angeles, CA, USA). Despite their high accuracy, these commercial body scanners remain relatively expensive and are typically limited to specialized clinical or research facilities, restricting their broader adoption for community-based or home-based monitoring [8,10, 14,26,27]. Alternative portable solutions have been explored, which can be broadly categorized into two types. The first category employs specific type of devices or general devices equipped with dedicated depth sensors. For example, Callegari et al. evaluated the accuracy of PolyCam (Polycam Inc., San Francisco, CA, USA) running on a LiDAR-equipped smartphone for medico-legal body documentation[24]. Similarly, Oquendo et al. integrated a Structure Sensor Mark II (Occipital Inc., Boulder, CO, USA) with an iPad Pro (Apple Inc., Cupertino, CA, USA) and demonstrated its feasibility for adolescent idiopathic scoliosis screening by capturing 3D trunk morphology[18]. Without requirement of special hardware, the second category of solutions for portable body-scanning is template-based reconstruction. Poltronieri et al. evaluated a representative application, named MeThreeSixty (Size Stream LLC, Cary, NC, USA), which estimates 3D body shape by fitting a small number of 2D images to a predefined human body template[13]. Idrees et al. evaluated another two applications, Mobile Fit (Size Stream, Cary, NC, USA) and Prism Labs (Prism Labs, Los Angeles, CA, USA), which use two photos and a series of photos captured during the user’s full 360° rotation separately, for parameterized body model fitting [28]. Although this approach is computationally efficient and highly accessible, the reconstructed body shape is constrained by the underlying template model and may fail to capture fine anatomical details and individual body-shape variations. These limitations may become more pronounced for populations with atypical body shapes, such as pregnant individuals, people with severe obesity, and amputees with limb loss [29]. Consequently, there is a need for reconstruction approaches that can accurately preserve geometric fidelity without relying on dedicated hardware or predefined body models.

Unlike these two kinds of methods that use depth sensors for 3D body shape capturing or parameterize 3D body shape with templates, photogrammetric 3D reconstruction offers an alternative approach for affordable 3D body scanning with high accuracy. By identifying corresponding features across multiple images and reconstructing their spatial locations through triangulation, photogrammetry can generate detailed 3D surface models without relying on specialized sensors or predefined body templates. Recent studies have evaluated popular photogrammetric smartphone applications in medical settings. Rudari et al., evaluated 3 professional 3D scanner and 2 3D scanning applications, Scandy Pro (Scandy LLC, New Orleans, LA, USA) and PolyCam, for human hand scanning[30]. Walters et al. compared 3 smartphone applications, PolyCam, Luma (Luma AI Inc, Palo Alto, CA, USA), and Meshroom (AliceVision, Paris, France), to a professional 3D scanner Artec Eva (Artec 3D, Luxembourg City, Luxembourg) on residual limb scanning for health monitoring and socket design[31]. Both studies reported favorable performance of photogrammetry-based reconstruction, with PolyCam demonstrating superior performance among the evaluated smartphone applications. Despite these encouraging findings, existing validation studies have almost exclusively examined localized anatomical regions, such as the hand or residual limb, on relatively small datasets. Whether smartphone-based photogrammetric reconstruction can achieve comparable accuracy for complete human body acquisition remains largely unknown. Whole-body scanning introduces additional challenges, including subject motion, changing viewpoints, and reconstruction over substantially larger anatomical surfaces, all of which may influence measurement accuracy.

Therefore, this study aimed to evaluate the accuracy and consistency of a photogrammetric smartphone approach for whole-body 3D scanning under both realistic scanning conditions using human participants and controlled conditions using a rigid mannequin. Regarding the human dataset, we collected longitudinal 3D body shapes of pregnant participants, which involves both realistic body characteristics and progressive anatomical changes. Circumference measurements automatically extracted from photogrammetric reconstructions were compared with those obtained from a commercial depth-sensor-based scanner. The ability of both scanning approaches to capture longitudinal body-shape changes was evaluated using scans acquired from pregnant participants at different stages of pregnancy. To assess the consistency of the photogrammetric approach, repeated scans of a rigid mannequin were performed and compared with reference tape measurements, thereby eliminating variability introduced by physiological motion. We hypothesized that (1) the photogrammetric approach would demonstrate high agreement with measurements obtained from the commercial scanner, with small and generally acceptable bias, and (2) the photogrammetric approach would exhibit good repeatability and consistency on the mannequin, with lower variability than that observed in human participants. Representative products from each technology category were selected for evaluation, namely PolyCam, as a photogrammetric reconstruction application and Fit3D ProScanner as a commercial depth-sensor-based body scanner. The selection was based on market rating, our preliminary testing and evidence reported in previous studies [8,30–32].

To our knowledge, this is the first study to systematically evaluate a smartphone-based photogrammetric application for whole-body 3D scanning. The findings are expected to inform the development and adoption of smartphone-based photogrammetric whole-body 3D reconstruction technologies for clinical research, remote health monitoring, and future computational body-shape analysis.

## 2. Materials and Methods

## 2.1. 3D Scanning Device

PolyCam (Polycam Inc., San Francisco, CA, USA) was chosen as the representative photogrammetry based application based on its good performance in our preliminary testing and reports from previous validation studies [30,31]. And two smartphone models, iPhone 13 and iPhone 14 (Apple Inc., Cupertino, CA, USA), are used for scanning. In its image mode, PolyCam reconstructs 3D surface geometry from multiple overlapping RGB images using a photogrammetric reconstruction pipeline based on feature matching and multi-view geometry. Depending on scene complexity, reconstruction typically requires approximately 20–2000 RGB images. PolyCam supports both fixed position scanning, in which the target is rotating on a turntable, and dynamic scanning, in which the target keeps standing still and the operator takes photos around the target from different viewpoints. In our experiments, "object" option is selected before scanning. "Photogrammetry", "dense" resolution and "isolate object from environment" options are selected in the interface before processing. In addition, image sets can be uploaded through the PolyCam web platform, enabling centralized processing for applications such as community-based healthcare and facilitating large-scale data collection.

Fit3D ProScanner (Fit3D Inc., San Francisco, CA, USA) was selected as the representative commercial depth-sensor-based scanner due to the justified high precision and repeatability for anthropometric measurements obtained from this device [8,32], and its wide adoption in medical researches [14,33,34]. The Fit3D ProScanner employs three vertically aligned depth cameras, two handles adjustable along vertical direction for maintaining a standardized posture, and a motorized turntable to acquire complete body surface geometry during a 30-second scan.

## 2.2. 3D Optical Scan of Human Participants

## 2.2.1. Recruitment and Data Collection

In this study, we used a dataset collected as part of an obstetrics research project investigating the use of 3D body shape for prenatal care. The dataset includes 3D body scans acquired at two stages of pregnancy using both the Fit3D ProScanner and PolyCam in image mode. In this dataset, A total of 144 pregnant participants were enrolled. Exclusion criteria included: (1) age under 18; (2) multiple gestations; (3) presence of an enlarged fibroid uterus; (4) BMI greater than 60; (5) presence of any unstable medical or emotional condition or chronic disease that could interfere with study participation; (6) history of body altering procedures such as liposuction or plastic surgery.

The first visit was scheduled between 18 and 24 weeks of gestation, and the second visit was scheduled between 31 and 38 weeks of gestation. Each participant underwent two to three scanning sessions per visit using both PolyCam and Fit3D simultaneously. Before starting the scan, participants were instructed to wear form-fitting clothing and tightly secure their hair. During scanning, participants were instructed to hold the scanner handles and maintain an “A-pose” throughout the scanning process. The Fit3D turntable rotated automatically while the vertically aligned 3 depth cameras to fully capture the body shape from 360 degrees. While the Fit3D scanner was acquiring the body shape through turntable rotation, the operator simultaneously captured images using PolyCam in Photo Mode with a smartphone mounted on a tripod positioned 2.5 m from the participant. A sequence of images was captured during the rotation of the participant to reconstruct the 3D body surface. Each scan lasted approximately 30 seconds, during which about 35 images were captured. Three participants were excluded because all scans acquired by either Fit3D or PolyCam had significant distortion. Finally, with the highest-quality scan from each visit selected for analysis, 228 scans from the 141 participants were collected in total, including 116 scans acquired during the first visit and 112 scans acquired during the second visit.

## 2.2.2. Human Scan Data Pre-processing

Unlike 3D scans obtained from the commercial 3D body scanner, raw PolyCam reconstructions may contain surface noise, scaling inconsistencies, and non-human structures such as scanner handles and the turntable. Thus, before extracting measurements from the scans, the 3D body scans obtained using Fit3D ProScanner and PolyCam application were pre-processed to get clean and aligned pairs. The pre-process operations are illustrated in Figure 1.<sup>1</sup>

![](images/fe22af660f7dfc9d503a8b79bdd44d5e842ac45ad4e686e07ce141965b6e5566.jpg)  
Figure 1. Pre-process of the 3D human body shapes scanned by Fit3D ProScanner and PolyCam application.

Coordinate standardization Because Fit3D and PolyCam scans are generated in different coordinate systems, mesh alignment was required prior to analysis. To standardize the coordinates of meshes from different sources, we first centered the mesh, then transformed the mesh using standardization matrix calculated from Principal Component Analysis (PCA) results with the following steps.

Given the coordinates of the extracted point cloud of 3D body mesh, the dominant anatomical axes of each mesh, represented as principal components capturing the largest variation in the data, can be estimated using PCA. The top 3 principal components correspond to the vertical, mediolateral, and anterior directions of the 3D body scan, which can be represented as a 3 × 3 matrix $C ,$ with extracted top 3 principal components as row vectors. Prior to analysis, each scan is transformed to a canonical orientation such that the subject’s superior–inferior, mediolateral, and anteroposterior anatomical axes are aligned with the $\scriptstyle { \mathbf { Z } } ^ { - } , \ x ^ { - } ,$ and y-axes, respectively, with the anterior direction corresponding to the positive y-axis. This target orientation can be represented as matrix $T ,$ where

$$
T = \left[ { \begin{array} { r r r } { 0 } & { 0 } & { 1 } \\ { 1 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \end{array} } \right] .\tag{1}
$$

Suppose the transformation matrix is denoted as $R ,$ the relationship of matrices can be represented as:

$$
R C ^ { T } = T ^ { T } .\tag{2}
$$

Since the principal components are orthonormal, the transformation matrix R can be calculated by:

$$
R = T ^ { T } C .\tag{3}
$$

By applying the calculated R to the whole meshes, the orientations of all meshes are standardized into a common coordinate system. For the few upside down or facing-back outputs, manual flipping is needed. Both Fit3D scans and PolyCam scans need to go through this coordinate standardization process.

Non-human parts removal and scaling of PolyCam scans Since PolyCam app is not designed specifically for human scan, the turntable and scanner handles may be reconstructed together with the body surface. These non-human parts need to be removed before target measurements extraction. The mesh standardization allows critical points to be easily located for segmentation and scaling. As shown in Figure 1, to remove the turntable, the plane between feet is detected and all mesh components below this plane were removed. For the handle removal, we use critical points that have extreme x values of the body scan to locate the handles’ positions. The top points of handles can be located by indexing the minimum and maximum x values of the scan. Given the x values and averaged z value of the two handles tops, the handles can be segmented and removed. Additionally, the PCA-based coordinate standardization is repeated again after the turntable is removed to further adjust the 3D body scan for more accurate handles location.

After non-human parts are removed, the PolyCam mesh was scaled to match the Fit3D counterpart. In case there are vertices of feet removed with turntable, we keep using inter-handle distance for scaling. As the scale ratio is not constant across scan pairs, the PolyCam mesh for each pair was scaled by the ratio of the corresponding Fit3D scan width to the PolyCam scan width to align its scale with the matched Fit3D scan.

Registration of PolyCam captured mesh to Fit3D captured mesh The preceding steps provided an initial alignment between the PolyCam and Fit3D meshes. Fine registration was subsequently performed on the vertices of meshes from two sources using the Iterative Closest Point (ICP) algorithm. Given a source point set $P = p _ { i }$ and a target point set $Q = q _ { i } ,$ , the ICP algorithm iterates over the two steps: (1) establishing correspondences by finding the closest point $q _ { i } \in Q$ for each $p _ { i } ,$ where $P$ is transformed with current S; (2) updating the optimal transformation S by minimizing the following objective

$$
E ( S ) = \sum _ { i } \| S p _ { i } - q _ { i } \| ^ { 2 } .\tag{4}
$$

Before applying ICP to the meshes, the vertices of meshes are downsampled to 10000 points using Farthest Point Sampling (FPS) algorithm. The downsampled vertices of PolyCam meshes are set as source point set and the downsampled vertices of Fit3D meshes are set as target point set. After applying ICP to the corresponding point sets, the optimal transformation $S$ was calculated. The resulting transformation matrix $S$ was subsequently applied to the full-resolution PolyCam mesh to obtain the final registered reconstruction.

Automated measurements extraction With the steps described above, the meshes have nonhuman parts removed and are on a standard coordinate, while the meshes still have noise. We adopt built-in "Merge Close Vertices" and "Laplacian Smooth" functions in MeshLab (Visual Computing Laboratory, Pisa, Italy) to smooth the mesh surfaces. Four geometry-driven landmarks were then defined for automated circumference extraction. Rather than relying on conventional anatomical landmarks that require manual identification and are prone to human error, measurement locations were defined using automatically detectable geometric extrema on the body surface. Circumference measurements were obtained from horizontal cross-sections passing through the corresponding landmark heights. This strategy improves measurement reproducibility and facilitates large-scale automated body-shape analysis.

The four automatically detected landmarks were defined as follows: (1) on the right calf at the height of the maximum lateral protrusion of the right calf in the sagittal view; (2) on the hip at the height of the maximum posterior protrusion of the buttocks in the sagittal view; (3) on the waist at the height of the maximum anterior protrusion of the abdomen in the sagittal view; (4) on the right wrist located 70 mm above the widest part of the hand–handle junction on the right side. All positions were defined and acquired based on Fit3D scans. Since the experiment is only for accuracy evaluation, all values are measured parallel to the ground for simplification.

## 2.3. Repeated Polycam Scans of Rigid Mannequin

In addition to the human study, which evaluated PolyCam under realistic whole-body scanning conditions through simultaneous PolyCam and Fit3D acquisitions, we also assessed the repeatability and consistency of PolyCam using a rigid mannequin. This experiment focused on the performance of the scanning approach itself by minimizing variability arising from subject movement and physiological motion. Before scanning, we marked the four landmarks of right wrist, waist, hip, and right calf with markers, which are slightly different from the definition described in Section 2.2.2 due to the mannequin’s shape and pose. And we fixed the mannequin in an indoor environment with moderate artificial lighting. The mannequin is scanned by an operator holding a smartphone and capturing photographs from multiple viewpoints while moving around the mannequin. The scanning is repeated 15 times. To mimic realistic user behavior and variations in scanning conditions, we varied camera-to-subject distance and number of acquired images in different scanning rounds. Images were captured at distances ranging from 0.7 m to 2.3 m, and the numbers of photos ranging from 20 to 62 with scan duration varying from 50s to 125s. For PolyCam scans on rigid mannequin, only scaling is needed for pre-processing. Circumferences are digitally measured on MeshLab. Tape measurements are also repeated 15 times on the four labeled landmarks for comparison.

## 2.4. Statistical Analysis

## 2.4.1. Accuracy Analysis on Human Data

To evaluate the effects of scanner type while accounting for gestational stages, a linear mixedeffects model (LMM) was employed. Scanner type and visit were included as fixed effects, whereas participant ID was included as a random effect to account for within-subject correlation arising from measurements of each participants in two visits. The model was specified as:

$$
Y = \beta _ { 0 } + \beta _ { 1 } S c a n n e r T y p e + \beta _ { 2 } V i s i t + u + e ,\tag{5}
$$

where Y is the measurements, $\beta _ { 0 }$ is the global intercept, $\beta _ { 1 }$ and $\beta _ { 2 }$ are fixed-effect slopes for scanner type and order of visit, u is the random intercept for patient ID, and e is the residual error.

Circumferences on the 4 landmarks were gathered and analyzed separately. The bias between Fit3D measurements and PolyCam measurements were summarized. Intraclass Correlation Coefficient (ICC) was used to evaluate the reliability and agreement between scanning methods. Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) were calculated on the measurement pairs to quantify how different the measurements from two scanners are. Pearson correlation coefficients were calculated to present the linear association between two scanning methods, and the Wilcoxon signed-rank test was employed to assess whether significant differences existed in the medians of measurements from two scanning methods.

To visualize the scanning results, the Bland-Altman method was used, showing the agreement and systematic bias between the measurements obtained from Fit3D scanner and PolyCam. Scatter plots were provided to show how PolyCam measurements and Fit3D measurements aligned on different landmarks. Heatmaps of vertex-level deviations were generated on 3D body scan for qualitative analysis, which was realized using Open3D library with Python, showing the distance between each vertex of PolyCam scan and its closest vertex from the corresponding Fit3D scan when overlapping.

## 2.4.2. Consistency Analysis on Mannequin Data

As for consistency evaluation, we applied Mann-Whitney U test to compare the repeated tape measurements and PolyCam measurements and determine if there are significant differences between two independent groups. Besides basic statistical measures such as mean and standard deviation, coefficient of variation (CV) was also calculated to quantify relative measurement variability across repeated scans. For results visualization, Box plot was used to illustrate how the measurements obtained by PolyCam and tape distribute.

## 3. Results

## 3.1. Accuracy Evaluation

Summary statistics of circumference measurements extracted from 228 paired PolyCam and Fit3D scans are presented in Table 1. Measurements were additionally stratified by visit to evaluate changes across gestation. The measurements obtained during the second visit were observed larger than those obtained during the first visit. Four waist measurement pairs and three hip measurement pairs were excluded because collapsed or merged body parts in either the Fit3D or PolyCam scan prevented reliable circumference extraction.

Table 1. Summary statistics of circumference measurements obtained from Fit3D and PolyCam at the four landmarks, with measurements from the 2 different visits discussed together and separately.
<table><tr><td rowspan="2"></td><td colspan="4">Fit3D (mm)</td><td colspan="4">PolyCam (mm)</td></tr><tr><td>mean</td><td>std</td><td>min</td><td>max</td><td>mean</td><td>std</td><td>min</td><td>max</td></tr><tr><td>calf (N=228)</td><td>397.81</td><td>41.61</td><td>243.04</td><td>502.34</td><td>382.80</td><td>44.57</td><td>214.05</td><td>490.08</td></tr><tr><td>calf-1 (N=116)</td><td>393.32</td><td>43.60</td><td>243.04</td><td>502.34</td><td>377.64</td><td>47.26</td><td>214.05</td><td>490.08</td></tr><tr><td>calf-2 (N=112)</td><td>402.47</td><td>38.89</td><td>278.25</td><td>494.88</td><td>388.13</td><td>40.93</td><td>253.99</td><td>486.14</td></tr><tr><td>hip (N=225)</td><td>1108.76</td><td>107.66</td><td>877.66</td><td>1494.88</td><td>1095.84</td><td>106.35</td><td>857.57</td><td>1433.95</td></tr><tr><td>hip-1 (N=116)</td><td>1089.84</td><td>98.29</td><td>936.47</td><td>1377.35</td><td>1076.31</td><td>99.25</td><td>911.07</td><td>1318.61</td></tr><tr><td>hip-2 (N=109)</td><td>1128.89</td><td>113.40</td><td>877.66</td><td>1494.88</td><td>1116.62</td><td>109.66</td><td>857.57</td><td>1433.95</td></tr><tr><td>waist (N=224)</td><td>1064.23</td><td>117.68</td><td>835.19</td><td>1363.60</td><td>1056.42</td><td>118.60</td><td>827.16</td><td>1365.36</td></tr><tr><td>waist-1 (N=112)</td><td>1005.85</td><td>99.78</td><td>835.19</td><td>1311.44</td><td>996.88</td><td>101.90</td><td>827.16</td><td>1326.09</td></tr><tr><td>waist-2 (N=112)</td><td>1122.62</td><td>104.53</td><td>861.74</td><td>1363.60</td><td>1115.96</td><td>103.24</td><td>854.46</td><td>1365.36</td></tr><tr><td>wrist (N=228)</td><td>193.50</td><td>23.80</td><td>150.50</td><td>273.73</td><td>183.28</td><td>22.22</td><td>142.78</td><td>263.04</td></tr><tr><td>wrist-1 (N=116)</td><td>191.67</td><td>25.09</td><td>162.43</td><td>273.73</td><td>180.69</td><td>22.23</td><td>148.42</td><td>248.28</td></tr><tr><td>wrist-2 (N=112)</td><td>195.39</td><td>22.23</td><td>150.50</td><td>260.59</td><td>185.96</td><td>21.88</td><td>142.78</td><td>263.04</td></tr></table>

Note: The "-1" or "-2" after location names means this group of measurements are from scans obtained in participants’ 1st visit between 18-24 gestational weeks or the 2nd visit between 31-38 gestational weeks. N in the bracket denotes the number of samples.

Table 1 shows that circumference measurements increased between visits at all four anatomical landmarks, with the largest increase observed for waist circumference. PolyCam consistently yielded slightly smaller measurements than Fit3D while exhibiting comparable variability across landmarks.

With the extracted measurements, we applied LMM to evaluate how the scanning methods related to the measurements. Outputs of LMM on different landmarks are listed in Table 2.

Table 2. LMM output of fixed effects.
<table><tr><td>Parameter</td><td>Estimate</td><td>Std.Error</td><td>df</td><td>t-value</td><td>Pr(&gt;|tl)</td></tr><tr><td>Right calf</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(Intercept)</td><td>392.032</td><td>3.716</td><td>164.516</td><td>105.493</td><td>&lt;2e-16</td></tr><tr><td>scannerpoly</td><td>-15.018</td><td>1.436</td><td>312.228</td><td>-10.459</td><td>&lt;2e-16</td></tr><tr><td>Visit2</td><td>10.731</td><td>1.627</td><td>324.352</td><td>6.594</td><td>1.74E-10</td></tr><tr><td>Hip</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(Intercept)</td><td>1088.635</td><td>8.833</td><td>146.688</td><td>123.246</td><td>&lt;2e-16</td></tr><tr><td>scannerpoly</td><td>-12.919</td><td>1.808</td><td>307.068</td><td>-7.146</td><td>6.53E-12</td></tr><tr><td>Visit2</td><td>40.444</td><td>2.086</td><td>310.561</td><td>19.387</td><td>&lt;2e-16</td></tr><tr><td>Waist</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(Intercept)</td><td>1006.425</td><td>8.785</td><td>150.297</td><td>114.556</td><td>&lt;2e-16</td></tr><tr><td>scannerpoly</td><td>-7.815</td><td>2.288</td><td>305.895</td><td>-3.416</td><td>0.000721</td></tr><tr><td>Visit2</td><td>115.62</td><td>2.629</td><td>311.527</td><td>43.974</td><td>&lt;2e-16</td></tr><tr><td>Right wrist</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(Intercept)</td><td>190.954</td><td>1.979</td><td>203.333</td><td>96.469</td><td>&lt;2e-16</td></tr><tr><td>scannerpoly</td><td>-10.215</td><td>1.17</td><td>310.897</td><td>-8.729</td><td>&lt;2e-16</td></tr><tr><td>Visit2</td><td>5.468</td><td>1.307</td><td>340.425</td><td>4.185</td><td>3.63E-05</td></tr></table>

In Table 2, "Estimate" is the estimation of $\beta$ described in Equation 5; standard error, degrees of freedom (df) and t-value are used to estimate $p$ value. The LMM results provided in Table 2 were consistent with the descriptive statistics presented in Table 1: (1) both scanner type and visit order were significantly associated with measurements at all four landmarks; (2) PolyCam measurements were consistently lower than Fit3D measurements; (3) Measurements obtained during the second visit were significantly larger than those obtained during the first visit on all 4 landmarks particularly on the waist. Focusing on the influences of scanning methods, further analysis was conducted on the bias between PolyCam and Fit3D scan pairs (PolyCam - Fit3D), of which the results are provided in Table 3. The Bland-Altman plots and scatter plots are provided in Figure 2.

Table 3. Analysis on bias between PolyCam measurements and Fit3D measurements.
<table><tr><td></td><td>bias mean</td><td>bias std</td><td>RMSE</td><td>MAE</td><td>ICC</td><td>Pearson</td><td>p-value</td></tr><tr><td>calf</td><td>-15.02</td><td>8.13</td><td>17.08</td><td>15.72</td><td>0.926</td><td>0.984</td><td>&lt;0.001</td></tr><tr><td>calf-1</td><td>-15.67</td><td>7.63</td><td>17.43</td><td>16.01</td><td>0.931</td><td>0.989</td><td>&lt;0.001</td></tr><tr><td>calf-2</td><td>-14.34</td><td>8.56</td><td>16.70</td><td>15.42</td><td>0.918</td><td>0.978</td><td>&lt;0.001</td></tr><tr><td>hip</td><td>-12.92</td><td>13.96</td><td>19.02</td><td>14.31</td><td>0.984</td><td>0.992</td><td>&lt;0.001</td></tr><tr><td>hip-1</td><td>-13.53</td><td>13.34</td><td>19.00</td><td>14.40</td><td>0.982</td><td>0.991</td><td>&lt;0.001</td></tr><tr><td>hip-2</td><td>-12.27</td><td>14.55</td><td>19.04</td><td>14.21</td><td>0.986</td><td>0.992</td><td>&lt;0.001</td></tr><tr><td>waist</td><td>-7.82</td><td>10.16</td><td>12.81</td><td>10.29</td><td>0.994</td><td>0.996</td><td>&lt;0.001</td></tr><tr><td>waist-1</td><td>-8.98</td><td>8.88</td><td>12.63</td><td>10.44</td><td>0.992</td><td>0.996</td><td>&lt;0.001</td></tr><tr><td>waist-2</td><td>-6.65</td><td>11.17</td><td>13.00</td><td>10.14</td><td>0.992</td><td>0.994</td><td>&lt;0.001</td></tr><tr><td>wrist</td><td>-10.21</td><td>9.81</td><td>14.16</td><td>11.45</td><td>0.832</td><td>0.911</td><td>&lt;0.001</td></tr><tr><td>wrist-1</td><td>-10.98</td><td>10.63</td><td>15.28</td><td>12.14</td><td>0.814</td><td>0.906</td><td>&lt;0.001</td></tr><tr><td>wrist-2</td><td>-9.42</td><td>8.81</td><td>12.90</td><td>10.74</td><td>0.844</td><td>0.920</td><td>&lt;0.001</td></tr></table>

Note: The unit is mm; the p-value is calculated using the Wilcoxon signed-rank test; the "-1" or "-2" after location names means this group of measurements are from scans obtained in participants’ 1st visit between 18-24 gestational weeks or the 2nd visit between 31-38 gestational weeks

![](images/b77b879b488084203efdafbe630474ac0a95c07cedc024dc6e958d286928f3b2.jpg)

![](images/437f1b175ed7fdb27836b43a95c1671b1aed050bc3397a71011ee489f5d0c92f.jpg)

![](images/4dd77e81291fb8826bf62608ac0b6d01422e797188a96e3a115ff55f5c3e4ed9.jpg)

![](images/f24d4bae24092498c84228932d1e2ef9c3775704303932469b21ec2c88c3484e.jpg)  
(a)

PolyCam vs Fit3D on Right Calf Circumference Measurement  
![](images/3e5e7fad6659d8f99382eeb2e9fe25072d0cb696d96ba88c285990a7a1230426.jpg)

PolyCam vs Fit3D on Hip Circumference Measurement  
![](images/d6d92b49f71059371a3097feb81531805db8a43f55b2c94a6c276cdb5e52314f.jpg)

PolyCam vs Fit3D on Waist Circumference Measurement PolyCam vs Fit3D on Right Wrist Circumference Measurement  
![](images/38bf5ccdda128be1a94baa410e998069061e0f4caede31476c06be397847bc8b.jpg)

![](images/8b216aed74b7c1fc77f74c8060d15c73d9e0aa87be04903243157abd7d19aee5.jpg)  
(b)  
Figure 2. Comparison of circumference measurements obtained from Fit3D and PolyCam at the four landmarks. (a) Bland-Altman plots, where dashed lines represent limits of agreement and solid line represents mean bias; (b) scatter plots.

The Table 3 shows that PolyCam consistently produced lower circumference measurements than Fit3D across all anatomical landmarks. When measuring larger circumferences such as hip and waist circumferences, PolyCam shows higher relative accuracy than on calf and wrist circumferences. For all landmarks, ICC values exceeded 0.80 at all landmarks and were greater than 0.90 for the calf, hip, and waist. And Pearson coefficients over 0.9 are observed, showing high agreements between two scanning methods. Except measurements on wrist, all other measurements have ICC over 0.9, which is a widely accepted threshold in clinical practice [35]. Wilcoxon signed-rank test on all measurement pairs yielded $p < 0 . 0 0 1$ . The RMSE and MAE scores lower than 20 mm, with RMSE ranging from 12.63 mm to 19.04 mm and MAE ranging from 10.14 mm to 16.01 mm. The Bland-Altman plots and scatter plots in Figure 2 visualize the alignments and deviations between the two type of measurements.

In Table 3, for the measurements extracted from scans of different pregnancy stages, comparable trends were observed across visits for both scanning methods. Along with Table 1, we notice that the bias is proportional to the length of circumferences within landmarks. In late pregnancy, participants have larger measurements across all landmarks, when bias between PolyCam and Fit3D are smaller.

To present a global view of scanning bias between the two methods, five representative participants were randomly selected and the heatmaps were generated to show the vertices-level bias from front and back views. In Figure 3, we observe that most of body regions across all examples exhibited vertex-level deviations below 5 mm. Larger deviations were primarily observed on breasts, underarms, and crotch regions, where bias larger than 25 mm can be found.

![](images/b2759d4fefba2499f78d03fcb29f1f6bba33263855d97fa1f18da34d78b1d04f.jpg)  
Participant-1  
Participant-2  
Participant-3  
Participant-4  
Participant-5

Figure 3. Heatmaps of vertex-level deviations between registered PolyCam and Fit3D scans for five randomly selected participants from front view and back view. Colors represent the Euclidean distance between each PolyCam vertex and its nearest vertex on the corresponding registered Fit3D mesh, with blue indicating small deviations and red indicating large deviations.

## 3.2. Consistency Evaluation

The repeat PolyCam scans and tape measuring were conducted to get two groups of measurements on the 4 landmarks of the rigid mannequin. Since inevitable physiological motions such as breath are eliminated on the mannequin, the results can accurately reflect the capability of PolyCam and ensure the same distribution of samples for consistency evaluation. The landmarks on the mannequin are defined by the reference markers on the mannequin surface, as shown in Figure 4(a). Summary of the measurements from the two sources and analysis results are listed in Table 4. Box plots are presented for comparison in Figure 4(b).

![](images/9faa3a269e9d913047c57ebf02f932668257aa51c4eb49574153815693fe68e3.jpg)  
(a)

![](images/6b7c96f28fc1e9d733927985c3c3dd1ff49e92d83da7f5a767f3a37c161ba197.jpg)

![](images/7cde98d412c83a443eb2a1c62ea1f5eb4839d6aa7aa8bc534c917adde3c762d8.jpg)

![](images/66d7d11e16db67c9b2247619376da6c98daf4eb46ff6858f776171ec1f132427.jpg)

![](images/9c4c8c67c88f22d6f2a0d72698427f50ef6275525e11befec8df19a6ec873da5.jpg)  
(b)  
Figure 4. Landmarks marked on the rigid mannequin and distributions of repeated measurements obtained using PolyCam and tape measurements. (a) PolyCam scan of the rigid mannequin with landmarks highlighted for digital measurement. (b) Box plots of the 15 repeated measurements by tape and PolyCam each on the 4 landmarks.

Table 4. Comparison between tape measurements and PolyCam-based measurements on a rigid mannequin.
<table><tr><td rowspan="2"></td><td colspan="5">PolyCam (mm) (n=15)</td><td colspan="5">Tape (mm) (n=15)</td><td rowspan="2">Mann-Whitney U test p-value</td></tr><tr><td>mean</td><td>std</td><td>min %</td><td>max %</td><td>CV</td><td>mean</td><td>std</td><td>min %</td><td>max %</td><td>CV</td></tr><tr><td>calf</td><td>332.21</td><td>4.59</td><td>-1.42%</td><td>3.88%</td><td>0.014</td><td>329.55</td><td>0.58</td><td>-0.32%</td><td>0.29%</td><td>0.002</td><td>0.135</td></tr><tr><td>hip</td><td>850.92</td><td>9.47</td><td>-2.10%</td><td>1.63%</td><td>0.011</td><td>854.38</td><td>1.95</td><td>-0.34%</td><td>0.60%</td><td>0.002</td><td>0.263</td></tr><tr><td>waist</td><td>644.27</td><td>6.58</td><td>-1.04%</td><td>1.94%</td><td>0.010</td><td>642.07</td><td>0.80</td><td>-0.17%</td><td>0.30%</td><td>0.001</td><td>0.213</td></tr><tr><td>wrist</td><td>154.65</td><td>4.59</td><td>-4.25%</td><td>6.92%</td><td>0.030</td><td>154.11</td><td>2.17</td><td>-1.70%</td><td>2.52%</td><td>0.014</td><td>0.836</td></tr></table>

Note: "min%" is calculated by $\left( x _ { m i n } - \bar { x } _ { t a p e } \right) / \bar { x } _ { t a p e } ,$ where $x _ { m i n }$ is the minimum measurement within the group and $\bar { x } _ { t a p e }$ is the corresponding mean tape measurement; vice versa for "max%".

From the results, we observe that on the rigid mannequin, where confounding factors associated with subject motion were eliminated, the bias of means are below 3.5 mm. Although both Table 4 and Figure 4(b) show that PolyCam measurements have larger standard deviations than tape measurements, the relative extreme values and CVs show a low measurement variability of PolyCam measurements, with CV below 0.030. And by applying Mann-Whitney U test on the measurements from two sources, no significant differences were observed between PolyCam and tape measurements at any landmark.

## 4. Discussion

This study evaluated the accuracy and consistency of a smartphone-based photogrammetric approach for whole-body 3D scanning by comparing PolyCam reconstructions with scans obtained using a commercial depth-sensor-based body scanner. Visual inspection of the reconstructed meshes indicated that PolyCam was capable of preserving the detailed whole-body surface geometry. Across the four automatically extracted circumference measurements, PolyCam showed strong agreement with Fit3D, with ICC values exceeding 0.80 and Pearson correlation coefficients exceeding 0.90 at all landmarks. Furthermore, comparable increases in circumference were also observed between the two pregnancy visits, indicating that both methods captured similar longitudinal body-shape changes. Repeated measurements of the rigid mannequin further showed good accuracy and consistency under controlled conditions.

![](images/a8288edb9108e4cc3c87c9412597ffa8bed72bfc1dafcf9bd1ef13296ca946a7.jpg)

Despite the strong agreement observed between the two methods, PolyCam measurements were consistently smaller than their corresponding Fit3D measurements. The mean differences (PolyCam - Fit3D) were -15.02 mm for the calf, -12.92 mm for the hip, -7.82 mm for the waist, and -10.21 mm for the wrist. However, PolyCam scans on the rigid mannequin did not show the consistent shrink on all measurements. This observation reveals the challenge of accurately scaling 3D body scans obtained by photogrammetric scanning on the non-rigid and rich-detailed human body.

Additionally, although these differences represented a relatively small proportion of the measured circumferences, particularly for the hip and waist, previous studies of localized body regions have reported smaller measurement errors with PolyCam. Walters et al. reported RMSE values ranging from 1.99 mm to 2.31 mm for circumference measurements of reconstructed residual limbs, while Rudari et al. reported a standard deviation of 0.48 mm for finger cross-sectional circumferences reconstructed using PolyCam photo mode [30,31]. However, these studies focused on isolated body parts and acquired images specifically targeting the region of interest. In contrast, the present study evaluated whole-body scanning under realistic conditions, where inevitable postural sway and subtle body movements occur during image acquisition. These methodological differences likely contributed to the larger measurement deviations observed in the present study and should be considered when comparing the results with those reported for localized body-part reconstruction.

The mannequin experiment provides additional support for this interpretation. When subject motion was eliminated, the absolute difference between the mean PolyCam and mean reference tape measurements was below 3.5 mm at all four landmarks, approaching the magnitude of measurement errors reported in previous studies of localized body-part reconstruction [31]. Although PolyCam measurements exhibited greater variability than tape measurements, the Mann–Whitney U test did not detect statistically significant differences between the two measurement distributions at any landmark. These findings suggest that factors associated with whole-body acquisition, including subject motion and fixed view point, may contribute to the larger measurement deviations observed in human participants in addition to errors inherent to the photogrammetric reconstruction process.

![](images/0121b3b4cd58cdb5c2a8eacb352f27c1b53bfe178a72af3a95fc88f443838a7d.jpg)  
collapse

![](images/c42809bb524878bdd709e4985fe42ae7b22a435194f17a95fb6a1f0e29921f6c.jpg)  
merging

![](images/c8382484611a7711924296d02d6d7d59fd6e31dbd364600bcdc4a19c036002dd.jpg)  
jitters

Figure 5. Representative examples of reconstruction artifacts observed in PolyCam and Fit3D scans. Affected regions are highlighted by red rectangles.

To better understand the observed measurement deviations, scan pairs exhibiting relatively large bias were further examined. Three primary factors were identified as potential sources of error. First, local reconstruction errors may arise from insufficient or ambiguous visual texture information. Heatmap analysis revealed that larger deviations frequently occurred in the breast region. During data collection, many participants wore dark-colored sports bras, which reduced visible shading and texture cues in RGB images. Because photogrammetric reconstruction relies on feature correspondence across multiple images, limited surface texture may reduce reconstruction accuracy and make it more difficult to distinguish local concave and convex geometry. Accordingly, the larger vertex-level deviations in these regions, as shown in Figure 5, may reflect local reconstruction artifacts in addition to differences between measurement systems. Second, the fixed image acquisition view point may have contributed the proportional distortion. In the present study, the smartphone remained fixed on a tripod at approximately waist height throughout scanning. Due to the perspective angle, body regions closer to the camera, such as the waist and hips, may have been reconstructed more accurately than regions farther from the optical center. This explanation is also compatible with the more proportional bias observed across landmarks in both the mannequin experiment and the longitudinal human scans, as well as with the spatial distribution of vertex-level deviations shown in the heatmaps, although the effect of camera position needs to be evaluated independently in future studies. (3) The scanning artifacts were also observed in some Fit3D scans. This issue brought about the outliers described in subsection 3.1. Surface collapse in pendulous regions, merging of adjacent body surfaces, and localized surface jitter were observed in several scans, as illustrated in Figure 5. Despite the use of three vertically aligned depth cameras, occlusion in regions such as the lower abdomen and hips may result in incomplete depth information and localized surface collapse.

In addition to the systematic factors discussed above, several limitations of the current study design are listed below. First, while the present work compared two representative technologies, namely a photogrammetry-based smartphone application and a commercial depth-sensor-based body scanner, direct tape measurements were not included in the human participant study. Therefore, the comparison primarily reflects agreement between the two scanning approaches rather than absolute accuracy in human subjects. Second, the consistently negative bias observed across landmarks suggests a potential scaling issue in the photogrammetric reconstruction process. Although scaling was corrected using the width between predefined reference points and fine-tuned with ICP algorithm, residual scaling errors may still have contributed to the systematic underestimation of circumference measurements.

Future studies may further improve whole-body photogrammetric scanning by optimizing the study protocol. Potential strategies include the use of bright, form-fitting clothing with distinctive patterns to enhance feature matching, incorporating an external physical scale reference to improve scaling accuracy, and the acquisition of images obtained from multiple heights and viewing angles while maintaining a stationary subject position. Moreover, direct comparisons between tape measurements and photogrammetric reconstructions in human participants would provide a more comprehensive assessment of measurement accuracy.

With continued validation and optimization, photogrammetry-based whole-body scanning has the potential to provide a practical and accessible solution for longitudinal body-shape monitoring outside clinical environments. As a low-cost and non-invasive technology, it could facilitate telehealth, at-home self-monitoring, and community-based health assessment. Beyond conventional anthropometric measurements, photogrammetric whole-body scans preserve rich 3D surface geometry that may serve as a novel source of information for health assessment. These high-dimensional body shape data could complement traditional clinical variables and support the development of models for longitudinal phenotyping, disease screening, and prediction of conditions in which body morphology is clinically relevant. Beyond healthcare, photogrammetric body scanning also has potential applications in personalized garment design, virtual fitting, ergonomic assessment, and medico-legal documentation.

## 5. Conclusion

This study evaluated a smartphone-based photogrammetric approach for whole-body 3D scanning by comparing PolyCam-derived circumference measurements with those simultaneously obtained using the depth-sensor-based Fit3D ProScanner, and with tape measurements on a rigid mannequin. PolyCam showed strong agreement with Fit3D across the four evaluated landmarks and captured longitudinal changes in body circumference across pregnancy visits. Repeated mannequin measurements further demonstrated good absolute accuracy, repeatability and consistency under controlled conditions. These findings support the potential of smartphone-based photogrammetry as an accessible approach for whole-body anthropometric assessment and longitudinal body-shape monitoring. Further validation using independent scale references and direct anthropometric measurements in human participants is warranted before broader clinical or remote-monitoring applications are considered.

Author Contributions: For research articles with several authors, a short paragraph specifying their individual contributions must be provided. The following statements should be used “Conceptualization, R.C. and J.H.; methodology, R.C. and B.F.; software, R.C.; validation, R.C. and B.F.; formal analysis, R.C. and Q.P.; investigation, R.C. and B,F.; resources, J.H.; data curation, R.C., and J.C.; writing—original draft preparation, R.C.; writing—review and editing, R.C., B.F., J.C., C.Q., Y.L., Q.P. and J.H.; visualization, R.C.; supervision, J.H.; project administration, J.H.; funding acquisition, J.H.

Funding: Research reported in this publication was supported by the National Institute of Diabetes And Digestive and Kidney Diseases under Award Number R01DK129809 and by the National Institute on Aging under Award Number R56AG089080 of the National Institutes of Health. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

Institutional Review Board Statement: The study was conducted in accordance with the Declaration of Helsinki, and approved by the Institutional Review Board of George Washington University (NCR224227, accepted on 08/31/2022).

Informed Consent Statement: Informed consent was obtained from all subjects involved in the study. All data were de-identified prior to analysis.

Data Availability Statement: Data may be made available based on the nature of the request.

Acknowledgments: We would like to express our sincere gratitude to all the volunteers who contributed their data to this study.

Conflicts of Interest: The authors declare no conflicts of interest.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.

## References

1. Kim, Y.; Baytar, F. Accuracy and feasibility of 3D virtual dynamic fit technology. International Journal of Clothing Science and Technology 2024, 36, 499–515.

2. Pei, J.; Park, H.; Ashdown, S.P. Female breast shape categorization based on analysis of CAESAR 3D body scan data. Textile Research Journal 2019, 89, 590–611.

3. Jung, M.; Sim, S.; Kim, J.; Kim, K. Impact of personalized avatars and motion synchrony on embodiment and users’ subjective experience: empirical study. JMIR Serious Games 2022, 10, e40119.

4. Minetto, M.A.; Margara, A.; Quilico, E.; Busso, C.; Graziano, C.; Shepherd, J.A.; Heymsfield, S.B.; Pietrobelli, A. Feasibility and clinical utility of digital anthropometry for precise assessment of outcomes after postbariatric reconstructive plastic surgery. Frontiers in Surgery 2026, 13, 1728844.

5. Feng, B.; Zheng, Y.; Cheng, R.; Feng, S.; Vaziri, K.; Hahn, J.K. Enhanced Body Composition Estimation from 3D Body Scans. In Proceedings of the BIOSTEC (1), 2025, pp. 421–431.

6. Balasubramanian, M.; Sheykhmaleki, P. Emerging roles of 3D body scanning in human-centric applications. Technologies 2025, 13, 126.

7. Brownridge, A.; Twigg, P. Body scanning for avatar production and animation. International Journal of Fashion Design, Technology and Education 2014, 7, 125–132.

8. Ng, B.K.; Hinton, B.J.; Fan, B.; Kanaya, A.M.; Shepherd, J.A. Clinical anthropometrics and body composition from 3D whole-body surface scans. European journal ofclinical nutrition 2016, 70, 1265–1270.

9. Tian, I.Y.; Liu, J.; Wong, M.C.; Kelly, N.N.; Liu, Y.E.; Garber, A.K.; Heymsfield, S.B.; Curless, B.; Shepherd, J.A. 3D convolutional deep learning for nonlinear estimation of body composition from whole body morphology. npj Digital Medicine 2025, 8, 79.

10. Feng, B.; Cheng, R.; Zheng, Y.; Feng, S.; Bai, N.; Vaziri, K.; Hahn, J. Voxel-based Deep Regression for Enhanced Body Composition Estimation from 3D Body Scans. SN Computer Science 2026, 7, 292.

11. Zheng, Y.; Long, Z.; Feng, B.; Cheng, R.; Vaziri, K.; Hahn, J.K. D3bt: Dynamic 3d body transformer for body fat percentage assessment. IEEE Journal of Biomedical and Health Informatics 2024, 29, 848–856.

12. Zheng, Y.; Long, Z.; Cheng, R.; Feng, B.; Vaziri, K.; Zhang, X.; Hahn, J.K. Predicting nonalcoholic fatty liver disease in obese populations with 3D body scans. In Proceedings of the 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2024, pp. 1–4.

13. Poltronieri, T.S.; da Silva, B.R.; Bennett, J.; Heymsfield, S.B.; Shepherd, J.A.; Prado, C.M. Using mobile applications for body composition analysis: a technical review of an artificial intelligence-based tool: Technical review of a body composition assessment mobile app. Clinical Nutrition ESPEN 2026, p. 103105.

14. Cheng, R.; Zheng, Y.; Feng, B.; Qiu, C.; Long, Z.; Calderon, J.A.; Zhang, X.; Phillips, J.M.; Hahn, J.K. Maternal and fetal health status assessment by using machine learning on optical 3D body scans. Medical & biological engineering & computing 2026, 64, 603–617.

15. Dathan-Stumpf, A.; Lia, M.; Meigen, C.; Bornmann, K.; Martin, M.; Aßmann, M.; Kiess, W.; Stepan, H. Novel three-dimensional body scan anthropometry versus MR-pelvimetry for vaginal breech delivery assessment. Journal of Clinical Medicine 2023, 12, 6181.

16. Gleason Jr, R.L.; Yigeremu, M.; Debebe, T.; Teklu, S.; Zewdeneh, D.; Weiler, M.; Frank, N.; Tolentino, L.; Attia, S.; Dixon, J.B.; et al. A safe, low-cost, easy-to-use 3D camera platform to assess risk of obstructed labor due to cephalopelvic disproportion. PloS one 2018, 13, e0203865.

17. Cheng, R.; Feng, B.; Zheng, Y.; Qiu, C.; Aiersilan, A.; Calderon, J.A.; Zhao, W.; Pan, Q.; Hahn, J.K. MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction. arXiv preprint arXiv:2511.03212 2025.

18. Oquendo, Y.; Hollyer, I.; Maschhoff, C.; Calderon, C.; DeBaun, M.; Langner, J.; Javier, N.; Bryson, X.; Richey, A.; Naz, H.; et al. Mobile device-based 3D scanning is superior to scoliometer in assessment of adolescent idiopathic scoliosis. Spine Deformity 2025, 13, 529–537.

19. Kuo, T.J.; Yu, C.Y.; Lin, J.C.; Lin, C.M.; Liou, T.H.; Peng, C.W.; Chen, H.C. Comparison of the performance of a Three-Dimensional Body Scanner and radiography in evaluating adult scoliosis. PeerJ 2026, 14, e20752.

20. Roy, S.; Grünwald, A.T.; Alves-Pinto, A.; Maier, R.; Cremers, D.; Pfeiffer, D.; Lampe, R. A noninvasive 3D body scanner and software tool towards analysis of scoliosis. BioMed research international 2019, 2019, 4715720.

21. Simenko, J.; Ipavec, M.; Vodicar, J.; Rauter, S. Body symmetry/asymmetry in youth judokas in the under 73 kg category. Ido Movementfor Culture. Journal ofMartial Arts Anthropology 2017, 17, 51–55.

22. Schueler, A.; Fichtner, I.; Ueberschär, O. Using the 3D body scanner in elite sports. Proceedings ofthe 3DBODY. TECH 2018.

23. Šimenko, J.; Serti´c, H.; Segedi, I.; Cuk, I. Rapid Assessment of Morphological Asymmetries Using 3D Bodyˇ Scanner and Bioelectrical Impedance Technologies in Sports: A Case of Comparative Analysis Among Age Groups in Judo. Symmetry 2024, 16, 1387.

24. Callegari, E.; Agnolucci, J.; Angiola, F.; Fais, P.; Giorgetti, A.; Giraudo, C.; Viel, G.; Cecchetto, G. The Precision, Inter-Rater Reliability, and Accuracy of a Handheld Scanner Equipped with a Light Detection and Ranging Sensor in Measuring Parts of the Body—A Preliminary Validation Study. Sensors 2024, 24, 500.

25. Kottner, S.; Thali, M.J.; Gascho, D. Using the iPhone’s LiDAR technology to capture 3D forensic data at crime and crash scenes. Forensic Imaging 2023, 32, 200535.

26. Ashby, N.; Jake LaPorte, G.; Richardson, D.; Scioletti, M.; Heymsfield, S.B.; Shepherd, J.A.; McGurk, M.; Bustillos, B.; Gist, N.; Thomas, D.M. Translating digital anthropometry measurements obtained from different 3D body image scanners. European Journal ofClinical Nutrition 2023, 77.

27. Dalamaggas, A.; Gillespie, M.; DellaPolla, A.; Samson, K.; Strudthoff, E.; Eaton, V.; Wallace, M. Assessing Body Fat in Pediatric Osteogenesis Imperfecta: A Preliminary Comparison of Anthropometric Techniques 2022.

28. Idrees, S.; Gill, S.; Vignali, G. Mobile 3D body scanning applications: a review of contact-free AI body measuring solutions for apparel. The Journal of The Textile Institute 2024, 115, 1161–1172.

29. Aiersilan, A.; Cheng, R.; Hahn, J. Investigating Anthropometric Fidelity in SAM 3D Body. arXiv preprint arXiv:2601.06035 2025.

30. Rudari, M.; Breuer, J.; Lauer, H.; Stepien, L.; Lopez, E.; Dragu, A.; Alawi, S.A. Accuracy of three-dimensional scan technology and its possible function in the field of hand surgery. Plastic and Reconstructive Surgery–Global Open 2024, 12, e5745.

31. Walters, S.; Metcalfe, B.; Twiste, M.; Seminati, E.; Bailey, N.Y. Smartphone scanning is a reliable and accurate alternative to contemporary residual limb measurement techniques. PLoS One 2024, 19, e0313542.

32. Sobhiyeh, S.; Kennedy, S.; Dunkel, A.; Dechenaud, M.E.; Weston, J.A.; Shepherd, J.; Wolenski, P.; Heymsfield, S.B. Digital anthropometry for body circumference measurements: Toward the development of universal three-dimensional optical system analysis software. Obesity Science & Practice 2021, 7, 35–44.

33. Minetto, M.A.; Pietrobelli, A.; Busso, C.; Bennett, J.P.; Ferraris, A.; Shepherd, J.A.; Heymsfield, S.B. Digital anthropometry for body circumference measurements: European phenotypic variations throughout the decades. Journal ofpersonalized medicine 2022, 12, 906.

34. Wong, M.C.; Ng, B.K.; Kennedy, S.F.; Hwaung, P.; Liu, E.Y.; Kelly, N.N.; Pagano, I.S.; Garber, A.K.; Chow, D.C.; Heymsfield, S.B.; et al. Children and adolescents’ anthropometrics body composition from 3-D optical surface scans. Obesity 2019, 27, 1738–1749.

35. Portney, L.G.; Watkins, M.P.; et al. Foundations of clinical research: applications to practice; Vol. 892, Pearson/Prentice Hall Upper Saddle River, NJ, 2009.