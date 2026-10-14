# Integrating 3D Posture Neutralization with Statistical Shape Modeling for Head–Neck-Shoulder Ergonomic Design

Jinbo Luo <sup>a,</sup>, Hang Ren <sup>b</sup>, Yadie Yang <sup>a</sup>, Jie Zhang <sup>c</sup>

<sup>a</sup>School of Fashion and Textiles, The Hong Kong Polytechnic University, Hong Kong SAR, China <sup>b</sup> Shenzhen Pingle Orthopedic Hospital, Shenzhen, China <sup>c</sup>Faculty ofApplied Sciences, Macao Polytechnic University, Macau SAR, China

## Abstract

The design of ergonomic products and workspaces requires a deep understanding of human anatomy and posture. In particular, the head-neck-shoulder region is critical for many applications such as cervical orthoses, helmets, and other head-worn or neck-worn devices. This paper presents a framework that integrates posture neutralization with statistical shape modeling (SSM) for the head-neck-shoulder (HNS) region. We acquire a single three-dimensional (3D) scan of the head-neck-shoulder region from each of 200 Chinese adults in a natural posture for each subject. Every scan is then posture-neutralized before SSM. We fit a parametric head-neck model to each scan to separate cervical pose from anatomy, and reset the cervical pose to a common neutral configuration, so that the resulting model captures anatomical shape variation rather than posture-induced deformation. The resulting shape space yields a standardized template and population-wide modes of variation, providing baselines for parametric ergonomic design. We further discuss how the model supports population coverage analysis and the design of products that accommodate anatomical diversity while maintaining neutral cervical posture.

Keywords: Statistical shape modeling, 3D Posture Neutralization, Head-neck-shoulder, Ergonomic design

## 1. Introduction

Accurate acquisition and statistical modeling of 3D head-neck-shoulder morphological data is a prerequisite for the ergonomic design of cervical spine and related wearable products [1, 2, 3]. Without accurate morphological data, design decisions concerning fit, comfort, or safety often rely on empirical estimation, limiting the rigor of evidence-based customization [4, 5].

In recent years, with the widespread adoption of information-based work and lifestyles, prolonged headdown use of electronic devices has led to the increasing prevalence of poor cervical postures such as forward head posture [6, 7]. Studies have shown that when using small electronic devices such as smartphones, the neck flexion angle often reaches between 30<sup>◦</sup> and 45<sup>◦</sup> [7, 8]. Furthermore, simplified biomechanical estimates suggest that for every 15<sup>◦</sup> of forward head tilt, the equivalent load on the cervical spine may increase by approximately 12 kg [9], though this static model has been criticized for oversimplifying physiological loading conditions. Under such sustained highload postures, the cervical musculoskeletal system is at elevated risk of chronic strain injury [10, 11]. The widespread prevalence of these poor postures has directly driven the rapid increase in cervical spine disorders. Research has shown that the incidence rate of cervical spondylosis in urban areas of China reached 405.65 per 100,000 person-years in 2017, with a significantly higher incidence in females (accounting for 60.58%) and the highest incidence observed in the 50- 59 age group [12]. Community-based surveys further estimate the symptomatic prevalence at around 13.76% in Chinese adults [13]. With the widespread adoption of information-based work habits and the aging population, cervical spondylosis exhibits a trend toward earlier onset and has become an increasingly serious public health concern. Against this backdrop, the demand for ergonomic pillows designed for daily cervical support has grown rapidly [14], thereby making the statistical shape modeling (SSM) [15] of 3D head-neck-shoulder shapes more significant and urgent. Beyond motivating the demand for cervical-supportive products, the wide range of cervical postures observed in daily life also creates a methodological challenge for SSM [16]. Scans acquired from subjects in diferent postures mix posture-induced deformation with true anatomical variation between individuals. Separating these two sources of variation is a prerequisite for any shape model intended to inform ergonomic design [17].

![](images/8a185a3e8f904e3bf1963ff1aebbbb0477220e9906bbaeda924238dc01397d1e.jpg)  
Figure 1: Typical applications of 3D head-neck-shoulder morphological data. (This figure was created by using Procreate software.)

Multiple standardized head-scanning databases have been built for diferent populations, such as CAE-SAR [18], NIOSH [19], SizeChina [20], SizeChina-Hunan [21], Size-Korea [22], Headspace [23], LSFM [24], and FaceScape [25]. Established methods include template-based registration [26, 27] and statistical modeling of head shape using principal component analysis (PCA) [28, 29]. These methods support sizing-system development and parametric design of head-support products. Table 1 summarizes regional studies by anatomical coverage, shape representation and posture treatment. Neck and neck-shoulder representations include measurement-based classifications, NURBS surfaces and mean meshes [30, 31, 32, 33].

Statistical shape modeling of articulated body regions faces a fundamental challenge. Scans collected from diferent subjects difer in pose as well as in anatomy, and these two sources of variation become entangled in the resulting shape space [17]. Two streams of work have addressed this entanglement. The first builds poseaware models that factor shape and pose into separate parameter spaces [39, 40, 35]. Representative examples include SCAPE [39] and SMPL [40] for wholebody modeling, and FLAME [35] for the head and face. The second normalizes pose as a preprocessing step, then builds the shape model on the aligned data [17]. Generalized Procrustes analysis (GPA) [41] and related superimposition methods from geometric morphometrics [42] are typical choices. Posture normalization for ergonomic design must preserve individual surface geometry and physical size. In addition, Table 1 shows that existing shape models mainly cover the head or head-neck region, with limited shoulder coverage for integrated anthropometry. The only head-neck dataset, HACK [37], is oriented toward digital human animation, with limited posterior accuracy and insuficient coverage of the neck-shoulder area for ergonomic design. These methodological requirements and data limitations constrain the scientific design of head-neckshoulder supportive products.

Motivated by this need, this paper develops a framework that couples 3D data acquisition, posture neutralization, and statistical shape modeling for the headneck-shoulder region. We collect a scan from each of 200 Chinese adults (gender-balanced, aged 20-70 years) under a self-selected natural standing posture. To prevent inter-subject posture variation from contaminating the shape model, every scan is posture-neutralized before statistical modeling. We fit a parametric headneck model to each scan to separate cervical pose from anatomy, and reset the cervical pose to a neutral configuration that reflects a relaxed, healthy posture. This age range covers the working-age and early-retirement population most afected by prolonged screen use and cervical degeneration [7, 12]. The Chinese cohort additionally supports population-specific sizing, given the reported diferences in head morphology across populations [20]. The main contributions of this work are summarized as follows:

• We develop a statistical shape model (SSM) based on posture-neutralized scans, which captures genuine anatomical diversity rather than postureinduced deformation, providing a standardized shape space and population-level modes of variation to support parametric ergonomic design and population coverage analysis.

Table 1: Comparison of head, neck and shoulder shape studies.
<table><tr><td rowspan="2">Study</td><td rowspan="2">Group</td><td rowspan="2">Age</td><td rowspan="2">Sample size Region</td><td rowspan="2"></td><td rowspan="2">Modeling</td><td colspan="3">Model information Usage</td><td rowspan="2"></td></tr><tr><td>Shape</td><td>Size</td><td>Pose</td></tr><tr><td>Wang et al. (2017) [34]</td><td>Chinese Male</td><td>25-30</td><td>200</td><td>Neck</td><td>×</td><td>√</td><td>√</td><td>√</td><td>Garment design</td></tr><tr><td>FLAME (2017) [35]</td><td>US/European</td><td>18-65</td><td>3,800</td><td>Head-Neck</td><td>√</td><td>√</td><td>X</td><td>√</td><td>Animation</td></tr><tr><td>LYHM (2020) [23]</td><td>Mostly White</td><td>1-89</td><td>1,212</td><td>Head</td><td>√</td><td>√</td><td>×</td><td>×</td><td>Head generation</td></tr><tr><td>Wang and Gu (2021) [32]</td><td>Chinese Female</td><td>18-25</td><td>188</td><td>Neck-Shoulder</td><td>X</td><td>√</td><td>√</td><td>√</td><td>Garment design, anthropometry</td></tr><tr><td>Zhang et al. (2022a) [31]</td><td>Chinese Male</td><td>18-25</td><td>174</td><td>Neck-Shoulder</td><td>X</td><td>√</td><td>V</td><td>√</td><td>Garment design, anthropometry</td></tr><tr><td>Zhang et al. (2022b) [29]</td><td>Chinese</td><td>18-84</td><td>1,496</td><td>Head</td><td>√</td><td>√</td><td>√</td><td>X</td><td>Wearable design, anthropometry</td></tr><tr><td>HIFI3D++ (2022) [36]</td><td>Caucasian/Asian -</td><td></td><td>1,957</td><td>Head</td><td>√</td><td>√</td><td>X</td><td>X</td><td>Reconstruction</td></tr><tr><td>HACK (2023) [37]</td><td>Multiethnic</td><td>15-65</td><td>624+600</td><td>Head-Neck</td><td>√</td><td>√</td><td>X</td><td>√</td><td>Animation</td></tr><tr><td>GNM (2026) [38]</td><td>Multiethnic</td><td>15-84</td><td>5000+</td><td>Head-Neck</td><td>√</td><td>√</td><td>X</td><td>√</td><td>Animation, simulation</td></tr><tr><td>Ours</td><td>Chinese adults</td><td>20-70</td><td>200</td><td>Head-neck-shoulder</td><td>√</td><td>√</td><td>√</td><td>×</td><td>Wearable design, anthropometry</td></tr></table>

![](images/badd48e0c1cec497e103e00734d0b39b1bdceef91d70531f53a1f7739f541f71.jpg)  
Figure 2: Overview of the head-neck-shoulder shape modelling pipeline. (1) Raw three-dimensional scans are collected in natural posture. (2) Anatomical landmarks and non-rigid iterative closest point (NICP) registration establish vertex correspondence. (3) Model-derived per-vertex transformations map each registered head-neck surface to a common neutral posture while retaining individual surface detail. (4) Generalized Procrustes analysis (GPA) aligns the meshes by rotation and translation. (5) Principal component analysis (PCA) represents population shape variation by the mean shape and principal components. The cervical bones [43] and joint chains are shown for illustration only and do not represent experimentally measured skeletal data.

• We construct a novel 3D head-neck-shoulder dataset comprising 200 Chinese adults (balanced by gender, aged 20–70 years), addressing the critical lack of integrated head-neck-shoulder morphological data for ergonomic design.

• We propose a posture-neutralization pipeline for 3D head-neck-shoulder shape modeling. A parametric head-neck model is fitted to each scan to separate cervical pose from craniocervical anatomy, the cervical pose is reset to a common neutral configuration, and the neutralized headneck is recombined with the subject’s shoulder.

## 2. Method

## 2.1. Overview

Figure 2 summarizes the data states used in the analysis. Registration first converts each raw scan to a mesh with shared topology and vertex correspondence while retaining the scan coordinate system. HACK fitting then estimates the subject-specific shape and acquisition posture. This change is applied directly to the registered head-neck vertices. The result is joined to the registered shoulder to form a complete head-neck-shoulder mesh. Generalized Procrustes analysis removes translation and rotation from these meshes, after which principal component analysis is applied to the resulting coordinate matrix.

![](images/6d6ae3724fd5eca0210b98432829bbda6a312a7ca69f4dfbee49f590b05e72f6.jpg)  
(a)

![](images/b1da337f908af14e15d1ab860f02fa14c7a5c496c03ae50cd91ca85f1073b721.jpg)  
(b)

![](images/5597c45de71bfb2b8ca6feaa50222dd65e5c4773d99ab905f54a14e5c86df235.jpg)  
(c)

![](images/e1e3b7cc03078da927eba274959e99b0307f714fce742679b5674b323a909907.jpg)  
(d)  
Figure 3: Distributions of participant characteristics in the study cohort (n = 200). (a) Age. (b) Height. (c) Weight. (d) Body mass index (BMI).

## 2.2. Data Collection

Two hundred Chinese adults (100 male, 100 female) aged 20-70 years were recruited for the study. The distributions of age, height, weight, and body mass index (BMI) are shown in Fig. 3. Subjects were eligible if they had no clinically diagnosed musculoskeletal disorders, no history of cervical injury or surgery, no congenital head or neck deformities, and no scoliosis or severe kyphosis. They were also required to stand independently in a stable upright posture for the duration of the scan. Subjects with metallic implants in the head or neck region were excluded, since these may interfere with optical scanning. This research complied with the American Psychological Association Code of Ethics. The study was approved by the Departmental Research Committee of The Hong Kong Polytechnic University (Reference Number: HSEARS20240912010). Written informed consent was obtained from each participant. Scanning took place in a private room with approximately $3 \mathrm { ~ m ~ } \times 3$ m of clear floor space, uniform overhead lighting, and a flat non-reflective floor. Geometry was acquired with a handheld Creality CR-Scan Otter together with the CreatyScan acquisition software, calibrated against a reference board at the start of each session. One trained operators handled calibration, posture guidance, and scanning throughout the study.

Questionnaire. Each subject completed a structured baseline questionnaire prior to scanning. The questionnaire covered ten categories: demographics, lifestyle and work habits, cervical history and health background, neck pain on a 0-10 numeric rating scale (VAS) [44], the Neck Disability Index (NDI) [45], the Pittsburgh Sleep Quality Index (PSQI) [46], pillow use and preferences, forward-head-posture self-assessment, the PHQ-4 mental health screen, and overall self-rated health. These items are retained as subject-level metadata for later subgroup and correlation analyses.

Scanning. After completing the questionnaire, each subject changed into a standard scanning garment with an open neckline that exposed the clavicles and upper chest, and tucked all hair into an elastic cap so that the cranial vault remained visible. Reflective marker stickers were applied at anatomical landmarks on the neck and shoulders, corresponding to the body landmarks defined in Section 2.3. The subject then adopted a selfselected natural standing posture, with feet shoulderwidth apart, arms relaxed at the sides, gaze directed toward an eye-level target, and the head and neck held comfortably rather than forced to a fixed cervical angle. A reference image illustrating the standing protocol was posted at the scanning station as a visual cue. The scanner was configured for human-body geometry capture at the high-quality setting, and the operator acquired a single full scan by walking once around the subject. Each scan was inspected immediately for coverage, surface noise from clothing folds or stray hair, and motion artefacts. Scans that failed this check were redone on the spot. The procedure produced one textured triangular mesh per subject, for a total of 200 scans that serve as the input to the registration pipeline.

## 2.3. Landmark Annotation

Landmarks act as the common reference between scans and the template throughout the rest of the pipeline. They provide the correspondences that drive the rigid alignment, anchor the non-rigid registration that follows, define the C7 and tragus points used for cervical posture neutralization, and supply the geometric reference for the anthropometric measurements. To support all of these uses with a single annotation pass, we define a fixed set of $N _ { L } = 7 3$ landmarks on the template mesh (see Fig. 4), comprising 22 body landmarks (1-13) and 51 facial landmarks (14-43). Each landmark on the template is paired with a corresponding annotation on every scan. Midline points are labelled with a single integer (8, 9, 11, 12, 25-29, 35, 38, 40, 43), and bilateral pairs share an integer with the sufix a/b denoting the left/right side of the image rather than the anatomical side, so that the same numbering can be read consistently across front, side, and back views. Fig. 4(a) shows the body landmarks in three views, and Fig. 4(b) shows the facial landmarks in three zoomed panels covering the nose (b1), eyebrows and eyes (b2), and mouth (b3).

![](images/847aeafcbc402dac3be6fe5eaff7d3c450f24d6f5a74e3206d06dbd962a50e66.jpg)  
(a1)

![](images/aa7c57f4975cbf51fa7977c4ccb72338369d29c05b0cb7d11127cf0ed02ba7ed.jpg)  
(a2)

![](images/b9c7c0cb0c60c18605a244620dbddaddd9580a12ede83416311abba7e0b4d950.jpg)  
(a3)

![](images/d1a0b8dc4c7c14ef0ce3d24200777efc7d505ab2b6c53a3f5a6bddacc6d9a24a.jpg)  
(b1)

![](images/e8d20d79bd0572cf7ebd10b6ec3cc9897a299192773f78e183a9692aeddd2ab3.jpg)  
(b2)

![](images/a8dc37d3a93ee1d1171a7026eab42440f786db7e01489b2f50d1d5d77b121c3f.jpg)  
(b3)  
Figure 4: Our predefined 73 landmarks on the template mesh, including 22 ear-neck-shoulder landmarks and 51 facial landmarks. (a) Ear-Neck Shoulder landmarks (1-13): shown in front (a1), side (a2), and back (a3) views. Indices 1-4 are bilateral pairs along the anterior helix (1, helix apex, 2, upper helix, 3, lower anterior helix, 4, pre-auricular notch), and 5-7 are bilateral pairs along the posterior helix (5, posterior helix near antitragus, $^ { 6 , }$ mid posterior helix, $^ { 7 , }$ posterior helix apex). Midline points 8 and 9 are the thyroid notch and the suprasternal (jugular) notch. Indices 10a/10b are the supraclavicular fossae. Posterior midline points 11 and 12 are the high cervical point near the C1 level and the C7 vertebra prominens. Indices 13a/13b are the acromion tips. (b) Facial landmarks (14-43) are shown in three zoomed panels. The nose (b1) includes points 25-29 along the midline from nasion to subnasale, and pairs 30a/30b and 31a/31b at the inner and outer nostril edges. The eyebrows and eyes (b2) include brow points 14-18 in five bilateral pairs from lateral to medial, and eye points 19-24 in six pairs running from outer canthus around the eye to lower lid lateral. The mouth (b3) includes outer lip points 32-38 from mouth corner to lower-lip midline, and inner lip points 39-43 from paramedian to midline. Facial indices follow the iBUG 300-W convention [47], reusing points 17-67 of the 68-point scheme. Throughout, sufixes a/b denote the left/right side of the image

![](images/e417ead11043c7355c7202dcccc82f28d3441274149ea6cee6da88c48dc90fe4.jpg)  
(a)

![](images/8ad16d9a4da14500d2d0c30824f4f285de145d23c0773379fd168dd48e4f060a.jpg)  
(b)

![](images/bcb456baf45da212cb544f9e5a214e237635a252152b494cba8fc83dfb12eda9.jpg)  
(c)  
Figure 5: Anthropometric measurements illustrated on a representative subject from three viewpoints; the circled numbers index the eleven measurements. (a) Frontal view: 1 the head circumference (red contour) at the maximum cranial girth, and 2 the neck circumference (green contour) at the level of the thyroid cartilage, each obtained as the perimeter of the polygon formed by intersecting the mesh with the corresponding horizontal plane. (b) Lateral view: 3 the head length, the horizontal distance from the forehead to the back of the head; 4 the vertex-menton distance (yellow dashed line), the straight-line distance from the vertex at the top of the head to the menton at the tip of the chin; 5 the head height, the vertical distance between the vertex and the menton; 6 the neck height, the vertical distance between the C1 and C7 cervical levels; 7 the head-to-wall distance, the shortest distance from the posterior head to the vertical reference (wall) plane; and 8 the neck-to-wall distance, the greatest distance from the posterior neck to the same plane. (c) Posterior view: 9 the head width, 10 the neck width, and 11 the shoulder width (the left-to-right distance between the acromia), each a horizontal distance between bilateral landmarks.

Ear-Neck-Shoulder Landmarks. The 22 body landmarks are organized by anatomical region. Indices 1-4 cover the anterior helix from the helix apex down to the pre-auricular notch on each side, and indices 5-7 cover the posterior helix from the antitragus up to the posterior helix apex. The anterior cervical midline is anchored by the thyroid notch (8) and the suprasternal notch (9), and the posterior cervical midline by a high cervical point near the C1 level (11) and the C7 vertebra prominens (12). Indices 10a/10b mark the supraclavicular fossae, and 13a/13b mark the acromion tips. The cervical midline points (8, 9, 11, 12) and the shoulder line (10, 13) are deliberately densified relative to head-only templates, because the cervical and shoulder geometry is the focus of the present model.

Facial Landmarks. The 51 facial landmarks follow the iBUG 300-W convention [47], reusing indices 17-67 of the 68-point scheme and excluding the 17-point jaw contour. The five facial groups are eyebrow (14-18, five pairs), eye (19-24, six pairs), nose (25-31, five midline points and two pairs along the nasal alae), outer lip (32- 38, five pairs and two midline points), and inner lip (39- 43, three pairs and two midline points). For each scan, these 51 points are detected automatically rather than placed by hand. The textured mesh is rendered from three viewpoints, and a face alignment network [48, 49] predicts 2D landmarks in each rendered image. Each 2D detection is then back-projected onto the mesh surface, and detections of the same landmark from diferent views are reconciled by selecting the candidate with the smallest reprojection residual. Automatic detection is essential at the scale of this study, where manual placement of dense facial points across 200 subjects would be neither consistent nor practical.

## 2.4. Anthropometric Measurements

We compute eleven anthropometric measurements on the head-neck-shoulder mesh of each subject, illustrated from three viewpoints in Fig 5. They fall into four geometric types: two circumferences, six axis-aligned linear dimensions, one Euclidean chord, and two posturerelated wall distances. Each mesh is expressed in a common anatomical frame with unit axes xˆ (medio-lateral), yˆ (vertical, superior-inferior), and zˆ (antero-posterior). All point coordinates and unit axes in this subsection are column vectors in $\mathbb { R } ^ { 3 \times 1 }$ . These quantities provide a persubject anthropometric summary for population statistics and serve to validate the posture normalization.

Circumferences. The two circumferences (Fig. 5a) are obtained by intersecting the mesh with a plane defined by three anatomical points on the skin. For the head circumference $( \bullet \bullet )$ , the three points lie on the maximum horizontal cranial girth: one at the forehead just above the supraorbital ridge, and two on the posterior cranium superior to the ears, following the convention of ISO 7250-1 [50]. For the neck circumference $( \pmb { \theta } )$ , the three points lie at the level of the thyroid cartilage, two on the lateral sides of the neck and one on the anterior midline. Each circumference is then the perimeter of the closed polygon formed by intersecting the mesh S with the corresponding plane Π:

$$
C = \sum _ { i = 1 } ^ { K } \| \mathbf c _ { i + 1 } - \mathbf c _ { i } \| , \quad \mathbf c _ { K + 1 } \equiv \mathbf c _ { 1 } ,\tag{1}
$$

where $\{ \mathbf { c } _ { i } \} _ { i = 1 } ^ { K }$ are the ordered vertices of Π ∩ S, obtained by linear interpolation at each triangle edge that Π crosses, and ∥ · ∥ is the Euclidean norm.

Sagittal lengths and heights. The lateral view (Fig. 5b) provides four linear dimensions. We write each axis-aligned dimension as the absolute projection of the vector between two points onto a unit anatomical axis,

$$
D = | ( { \mathbf v } _ { a } - { \mathbf v } _ { b } ) \cdot \hat { \mathbf e } | , \qquad \hat { \mathbf e } \in \{ \hat { \mathbf x } , \hat { \mathbf y } , \hat { \mathbf z } \} .\tag{2}
$$

The head length $\mathbf { \eta } ( \pmb { \circledast } )$ is the maximum cranial length, the antero-posterior extent of the head from the glabella to the opisthocranion, the most posterior point of the cranium [50]; it is Eq. (2) with $\hat { \textbf { e } } = \hat { \mathbf { z } }$ The vertex– menton distance $( \pmb { \bigcirc } )$ is instead the full Euclidean chord between the vertex, the highest point of the head, and the menton, the lowest point of the chin,

$$
L = \lVert \mathbf { v } _ { \mathrm { v e r t e x } } - \mathbf { v } _ { \mathrm { m e n t o n } } \rVert ,\tag{3}
$$

and the head height $\mathbf { \eta } ( \pmb { \oplus } )$ is its vertical component, given by Eq. (2) for the same two points with $\hat { \mathbf { e } } = \hat { \mathbf { y } }$ . The neck height ( 6 ) applies Eq. (2) with $\hat { \mathbf { e } } = \hat { \mathbf { y } }$ to the high cervical point near the C1 level and the C7 vertebra prominens.

Wall distances. The remaining two lateral measures quantify sagittal posture relative to the upper back. We define the wall plane $\Pi _ { w }$ as the vertical plane passing through the scapulae, with unit normal zˆ; the distance from a surface point v to this plane is,

$$
d ( \mathbf { v } , \Pi _ { w } ) = \left| ( \mathbf { v } - \mathbf { q } ) \cdot \hat { \mathbf { z } } \right| , \qquad \mathbf { q } \in \Pi _ { w } .\tag{4}
$$

The head-to-wall distance $( \pmb { \theta } )$ is the minimum of $d$ over the posterior head, and the neck-to-wall distance $( \pmb { \bigcirc } )$ is the maximum of d over the posterior neck. They capture the depth of the hindbrain and the cervical cur vature.

Widths. The posterior view (Fig. 5c) provides three medio-lateral widths, each obtained from Eq. (2) with $\hat { \mathbf { e } } = \hat { \mathbf { x } } .$ The head width ( 9 ) is the maximum transverse diameter of the head, taken between the two euryons, the most lateral points of the cranium [50]; the neck width ( 10 ) is taken at the narrowest lateral extent of the neck; and the shoulder width ( 11 ) is the biacromial distance between the left and right acromia.

## 2.5. Shape Registration

Shape registration establishes a common mesh representation across subjects. We deform the same headneck-shoulder template onto every scan. The template and every registered output contain $n = 2 1 , 2 1 2$ vertices with identical connectivity and vertex order. A vertex index therefore refers to the same surface location in all subjects. Point distribution models require such point-wise correspondence before population variation can be analysed [15]. Template-based registration with fixed topology has also been used in statistical models of human head and craniofacial shape [24, 23]. In our pipeline, the common correspondence supports the later posture neutralization, model alignment, and principal component analysis.

Initial Alignment. The raw scan and template can difer in position, orientation, and size. We use the 73 anatomical landmarks defined in Section 2.3 to obtain a landmark-guided global alignment before local deformation. The alignment begins with Procrustesbased similarity estimates from corresponding landmarks [41]. A rigid ICP refinement then uses surface geometry to reduce the remaining mismatch in position and orientation [51]. Together, these operations place the scan in the template working coordinate system. The complete scan-to-template similarity transform is recorded. Uniform scale is estimated during this initialization and is not updated during the subsequent NICP step.

Non-rigid Deformation. A single global transform cannot establish dense correspondence when local headneck-shoulder geometry difers between the template and scan. We therefore apply NICP, which assigns a local afine transform to each template vertex and deforms the template toward the aligned scan [27, 26]. The optimization is guided by surface proximity, landmark correspondence, and deformation regularization. The regularization combines local stifness with smoothing between neighboring vertices. Region-dependent surface weights and distance thresholds reduce the influence of unreliable correspondences in diferent anatomical regions. The global scale remains fixed during this step. After optimization, we apply the inverse of the recorded scan-to-template transform. The registered mesh for subject i is denoted by $\mathcal { F } _ { i } = \{ \mathbf { f } _ { i , \nu } \} _ { \nu = 1 } ^ { n }$ . Here, $\mathbf { f } _ { i , \nu } \in \mathbb { R } ^ { 3 }$ represents vertex v of this registered mesh and contains its three coordinates. The mesh has the template topology and vertex order, but its coordinates and physical scale are those of the subject’s scan.

## 2.6. Neck Posture Neutralization

The registered surface contains anatomical variation, acquisition posture, and diferences between the scan and the parametric model. We use HACK [37] to estimate the subject’s shape and acquisition posture. The procedure has three steps. We first fit HACK to the registered head-neck surface, then transport the scan-tomodel residual to a shared reference posture, and finally join the neutralized head-neck surface to the registered shoulder.

## 2.6.1. Reference Head-Neck Model

HACK represents the first $N _ { H } = 2 0 \small { , } 4 8 1$ vertices of each registered mesh [37]. These vertices cover the head and neck and have the same order as the HACK surface. The following equations describe the model for a single subject. The index $\nu \in \{ 1 , \ldots , N _ { H } \}$ denotes a vertex, and $b \in \{ 1 , \ldots , 8 \}$ denotes a joint in the chain $( c _ { 7 } t _ { 1 } \to \cdot \cdot \cdot \to c _ { 1 } c _ { 2 } \to o c _ { 1 } )$ . The fitted parameters are the shape vector $\beta \in \mathbb { R } ^ { 2 0 0 \times 1 }$ and the cervical-pose matrix $\theta \in \mathbb { R } ^ { 8 \times 3 }$ . Each row of θ is a rotation vector in the local joint coordinate system. Its direction gives the rotation axis and its magnitude gives the angle in radians.

The shape-dependent point at vertex v is denoted by $\gamma _ { \nu } ( \beta )$ . The pose-corrective ofset at the same vertex is denoted by $\delta _ { \nu } ( \theta )$ . These are two independent functions of the shape and pose parameters, respectively.

The fixed cervical skeleton contains eight joints. For joint b, $\mathbf { G } _ { b } ( \theta ) \in \mathbb { R } ^ { 4 \times 4 }$ is the transformation from the binding configuration to posture θ. Each vertex has a scalar skinning weight $w _ { \nu , b } \in [ 0 , 1 ]$ , with $\begin{array} { r } { \sum _ { b = 1 } ^ { 8 } w _ { \nu , b } = 1 } \end{array}$ The weighted joint transforms $\mathbf { M } _ { \nu } ( \theta ) \in \mathbb { R } ^ { 4 \times 4 }$ give

$$
{ \bf M } _ { \nu } ( \theta ) = \sum _ { b = 1 } ^ { 8 } w _ { \nu , b } { \bf G } _ { b } ( \theta )\tag{5}
$$

The complete HACK surface for one subject is then represented by

$$
\mathbf { H } ( \beta , \theta ) = [ \mathbf { M } _ { \nu } ( \theta ) \left( \pmb { \gamma } _ { \nu } ( \beta ) + \pmb { \delta } _ { \nu } ( \theta ) \right) ] _ { \nu = 1 } ^ { N _ { H } }\tag{6}
$$

The bracket collects the $N _ { H }$ vertex columns. Thus, $\mathbf { H } ( \beta , \theta ) \in \mathbb { R } ^ { 4 \times N _ { H } }$ directly denotes the complete posed HACK surface.

![](images/b26334e149f7033cf07b946d2f2fdb29e1ff8da778b89dd9573727126d596a65.jpg)

Figure 6: Neck posture neutralization. (a) The HACK reference model is fitted to the registered head-neck surface to estimate individual shape, acquired posture and the global similarity transform between model and registered coordinates. (b) The fitted shape is held fixed while cervical posture is reset to the common neutral configuration. Joint transformations and fixed skinning weights define the per-vertex neutralization. R: Rotation $\mathbf { M a t r i x } \in \ R ^ { 3 \times 3 }$ . (c) The model-derived neutralization is transferred through vertex correspondence to update the original registered surface, with coordinate changes determined by the fitted similarity transform. The cervical bones and joint chains are shown for illustration only and do not represent experimentally measured skeletal data.  
![](images/3edc4a602ba8fbebbcb8c9c9eeae51d47d1edc0cc58dc8cbac0bbad5b734d2f7.jpg)  
Figure 7: Cervical skinning weights on a representative neutralized registered head-neck surface. Weight fields for the C7–T1, C6–C7, C5–C6, C4–C5, C3–C4, C2–C3, C1–C2 and occiput–C1 (Oc–C1) joints, respectively.

## 2.6.2. Model Parameter Estimation

Before the neck posture can be neutralized, we need to determine the subject’s shape and the posture adopted during scanning. We therefore fit HACK to the registered head-neck surface. The fitted parameters are $\beta , \theta ,$ and a similarity transform $\mathbf { T } \in \mathbb { R } ^ { 4 \times \bar { 4 } }$ . The transform T maps HACK model coordinates to registered-scan coordinates.

The registered head-neck surface is written as $\mathbf { F } ^ { h n }$ ∈ $\mathbb { R } ^ { 4 \times N _ { H } }$ . The parameters are estimated by minimizing the scalar objective

$$
E = d \big ( \mathbf { T } \cdot \mathbf { H } ( \beta , \theta ) , \mathbf { F } ^ { h n } \big ) + \lambda _ { \beta } | | \beta | | _ { 2 } ^ { 2 } + \lambda _ { \theta } | | \theta | | _ { F } ^ { 2 } .\tag{7}
$$

Here, $d ( \cdot , \cdot )$ returns a scalar distance between two surfaces. The non-negative scalars $\lambda _ { \beta }$ and $\lambda _ { \theta }$ control the shape and pose constraints. The optimization returns $\beta ^ { * } , \theta ^ { * }$ , and T. The fitted shape $\beta ^ { * }$ is kept fixed during posture neutralization.

## 2.6.3. Vertex Transformation Computation

After model fitting, we calculate the change that moves each head-neck vertex from the acquired posture to the common reference posture. We define the reference posture as $\boldsymbol { \theta } _ { n } = \mathbf { 0 } _ { 8 \times 3 }$ , which sets all eight cervical rotation vectors to zero. For each head-neck vertex, the change from the fitted posture $\theta ^ { * }$ to the reference posture $\theta _ { n }$ is

$$
\mathbf { D } _ { \nu } = \mathbf { M } _ { \nu } ( \theta _ { n } ) \mathbf { M } _ { \nu } ( \theta ^ { * } ) ^ { - 1 }\tag{8}
$$

The matrix $\mathbf { D } _ { \nu } \in \mathbb { R } ^ { 4 \times 4 }$ is the fitted-to-reference posture transform for vertex v.

The per-vertex matrices $\mathbf { D } _ { \nu }$ define a displacement field over the head-neck surface. We apply this field to the registered surface to obtain the neutralized vertex positions. Let $\mathbf { f } _ { \nu } \in \mathbb { R } ^ { 4 \times 1 }$ be the registered head-neck vertex and $\mathbf { y } _ { \nu } \in \mathbb { R } ^ { 4 \times 1 }$ its neutralized position. The updated vertex is computed as

$$
\mathbf { y } _ { \nu } = \mathbf { T } \mathbf { D } _ { \nu } \mathbf { T } ^ { - 1 } \mathbf { f } _ { \nu } .\tag{9}
$$

The inverse transform $\mathbf { T } ^ { - 1 }$ maps the registered vertex to HACK coordinates. The matrix D<sub>v</sub> applies the fitted-toreference posture change. The transform T then returns the vertex to registered-scan coordinates.

## 2.7. Mesh Alignment

Vertex correspondence alone does not remove diferences in mesh position and orientation. We therefore align the complete head-neck-shoulder meshes before PCA using generalized Procrustes analysis (GPA) [41]. GPA uses the first three Cartesian coordinates of each homogeneous vertex. The alignment is estimated from 1,993 neck vertices and then applied to the complete mesh. It removes rotation and translation but does not change scale or reflect the surface.

After alignment, the coordinates of subject i are flattened into the column vector $x _ { i } \in \mathbb { R } ^ { 3 n \times 1 }$ . For m subjects, the PCA input matrix is

$$
\begin{array} { r } { X = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] ^ { \top } \in \mathbb { R } ^ { m \times 3 n } . } \end{array}\tag{10}
$$

## 2.8. Statistical Model Construction

The final modelling step represents population-level shape variation in a lower-dimensional form. We apply principal component analysis to X to construct the statistical shape model [52]. The model represents the population using a mean shape and principal components, with each eigenvalue giving the variance explained by its component.

Let $\bar { x } \in \mathbb { R } ^ { 3 n \times 1 }$ denote the mean shape vector. Let $P = [ \pmb { \mathrm { p } } _ { 1 } , \dots , \pmb { \mathrm { p } } _ { k } ] \in \mathbb { R } ^ { 3 n \times k }$ contain the first k principal directions, ordered by decreasing eigenvalue. Each column $\mathbf { p } _ { j } \in \mathbb { R } ^ { 3 n \times 1 }$ is one principal direction. The corresponding eigenvalues form the diagonal matrix Λ = $\mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \bar { \lambda _ { k } } ) \in \mathbb { R } ^ { k \times k }$ , where $\lambda _ { j }$ is the non-negative eigenvalue of component j.

A shape vector $\boldsymbol { x } \in \mathbb { R } ^ { 3 n \times 1 }$ is approximated by

$$
x = \bar { x } + P \alpha ,\tag{11}
$$

where $\alpha = [ \alpha _ { 1 } , \ldots , \alpha _ { k } ] ^ { \top } \in \mathbb { R } ^ { k \times 1 }$ contains the PC scores. Each score gives the deviation from the mean along its corresponding principal direction.

## 3. Results

## 3.1. Posture-Neutralized Statistical Shape Model

## 3.1.1. Average Head-Neck-Shoulder Shape

Figure 8 shows the average posture-neutralized headneck-shoulder shape in frontal and lateral views. This mean shape provides the reference for the principal component variations reported below.

![](images/2df84a520335ca97ed7efb56ee2375c690a3dfc87508cd3a4098c5c4175bb022.jpg)  
(a)

![](images/25f45deaf95afe84275bfd6edc6993c959bcf7bf44ed057a7a1262a905f21f81.jpg)  
(b)  
Figure 8: Average posture-neutralized head-neck-shoulder shape. The same mean mesh is shown in (a) frontal and (b) lateral views.

## 3.1.2. Principal Components and Model Evaluation

Model Visualization. Figure 9 shows the first five principal components of the posture-neutralized statistical shape model. Together, these components explained 89.52% of the total shape variance. PC1 explained 68.19%, followed by PC2 at 11.26%, PC3 at 5.81%, PC4 at 2.45%, and PC5 at 1.81%. Each component is shown at −3 SD and +3 SD from the mean shape.

Model Evaluation. Figure 10 summarizes the variance structure and training-set reconstruction performance of the model fitted to 200 aligned meshes. Three principal components were required to explain 80% of the variance, 6 were required to explain 90%, and 11 were required to explain 95%. The mean per-vertex reconstruction error decreased from $1 1 . 3 3 1 \pm 5 . 1 5 6$ mm for the mean-shape baseline to $2 . 4 6 2 \pm 0 . 4 2 6$ mm when 11 principal components were retained.

Model Analysis. We interpreted the first five PCs using the size measurements defined in Section 2.4 and shown in Fig. 11. All changes below refer to the modelgenerated endpoints from −3 SD to +3 SD. PC1 represented coordinated increases in head, neck, and shoulder dimensions. Head circumference, neck circumference, neck height, and shoulder width increased by 53.2, 170.8, 56.5, and 186.9 mm, respectively. PC2 mainly combined reductions in neck circumference and neck width with reduced shoulder width. These measurements decreased by 86.9, 19.6, and 50.9 mm, respectively, while neck height increased by 6.9 mm.

PC3 mainly changed neck height, which decreased by 10.4 mm. Neck circumference and neck width increased by 16.8 and 5.6 mm, respectively. PC4 mainly changed head height and vertex-to-menton distance, which decreased by 14.7 and 14.8 mm, respectively. PC5 combined increased neck height with reduced head and neck dimensions. Neck height increased by

+3 SD

PC1 68.19%

![](images/0f0e5063337360a0cc4793c4f590627b4319b149334ed281d764d23c3d1d6e5e.jpg)

PC2 11.26%

![](images/f8d8ea604e42e36ac5434db020bac8ba0cfcb2386ef20b04bb84f2e0b5945674.jpg)

PC3 5.81%

![](images/53b311a9a372d4f27581cc57f8669f3e815c060e4102fbee5f738ef5613e0b0b.jpg)

![](images/1664526167e4306f61cf1fdd87ebc582fa18dbbc976d5a7e1f2ae715bc9a5e34.jpg)

PC5 1.81%

PC4 2.45%

![](images/3c154cfcc25a6e08a03fdb163584418187f90e81a8c143d7eabaedf63e0a4e09.jpg)

![](images/4f68f020ead3d7bcd9f350d8269dcae049816290f07b06f56afe0cbc6d43583a.jpg)

![](images/f6914fd295583fc7ed1c1da6eacb08f42be204c06a8cb074f0cf838e487d0a08.jpg)  
-3 SD

![](images/27637713c23d55dcff4adf7dec5eb5afb2f870140b34da84cbca817f31accc1c.jpg)

![](images/134236d988b1e1e1f0c7557b6f32b2fe5dfbef9bbf2810204b3bebfc617d2ae1.jpg)

![](images/7bb25c59b4afe62bff9e8573b696687926e752d12c51065f09087be48a44ad71.jpg)

Overlay

![](images/df6abaf1b514557ee0107a41aebcab5133ea407a7523415c1d77667d20ccf566.jpg)

![](images/0d1a62e1c639f9b185316202258dbb46b2afa65f70f2ed9a094dc12d9c1e4247.jpg)

![](images/2a17baac7257ae590d2e761c0db621a9201293c31b7ad2c87bf74cd81bfd5a0e.jpg)

![](images/2e360ded8aa992444738d85583cfe1af155322c74b2f141cebfd209f392a887c.jpg)

![](images/99c68559c2ad38eb023ea36131ed722a7a4216997fba17c54e1defcc83a45748.jpg)

Figure 9: First five principal components of the posture-neutralized statistical shape model. The first and second rows show the +3 SD and −3 SD endpoints in the left lateral view. The third row overlays both endpoints. Slate blue denotes +3 SD, and orange denotes −3 SD.  
![](images/02b84e29cffb5d9168930f354bd2641ac0fe1cfdfaab181ee3c710759573f3f5.jpg)  
(a)

![](images/48481a4827559c5351e1951cde7d21360af278c8940a7c397efae17915be8e88.jpg)  
(b)

![](images/c756bad3a8caa9490d7115b53bab4b6b8536254afe7d70d8ee58e9e1ba85bcb3.jpg)  
(c)  
Figure 10: Evaluation and analysis of posture-neutralized head-neck shoulder model SSM. (a) Compactness: cumulative explained variance of the first 30 principal components. (b) Compactness: mean reconstruction error with standard deviation for diferent numbers of principal components. (c) BMI distribution of the standardized first and second PC scores in the dataset.

17.0 mm, while neck circumference and neck width decreased by 46.6 and 18.9 mm, respectively. Head circumference, head length, and head width decreased by 41.7, 13.7, and 13.4 mm, respectively.

We used Pearson correlations to assess BMI associations with the first 11 PC scores separately in females and males. Gender-specific BMI regressions in the shared posture-neutralized PCA model predicted mean surfaces at BMI 18.5, 23, 25 and 30 kg/m<sup>2</sup>. Figure 12 shows predicted mean shapes from the gender-specific BMI regressions. BMI was positively correlated with PC1 in females $( r ~ = ~ 0 . 2 4 3 )$ and males $( r ~ = ~ 0 . 5 1 6 )$ The corresponding correlations with PC2 were negative $( r = - 0 . 4 1 5$ and $r = - 0 . 5 8 2$ , respectively). Higher BMI was associated with outward changes around the lower face, neck and shoulder region in these predicted surfaces.

![](images/d37bf67d284069a6a525b7f742c62f56a44c04d948a432a82f909c90c4e9c831.jpg)  
(a)

![](images/039e0703e778daa97e9bed4cf2eee7d317da836c766e7a55bb01ba2305600f9b.jpg)

![](images/54e5a8e7ba81d9753e5d3c47cfb27406cb931f257fb0f4c9953513186807649e.jpg)

![](images/1d2fbcaaa36e844d3d6b445148db12cb5842fcdb2afb59edf759eda17e18a53c.jpg)

![](images/1a05119dc19bfab6ab233220ad5f92143179c9f33ecdd96ce94fa8d99b00b13a.jpg)  
(b)

Figure 11: Anthropometric changes along the first five principal components of the posture-neutralized model. (a) Endpoint changes in nine dimensions. (b) Absolute endpoint measurements for head circumference, neck circumference, neck height, and shoulder width.  
![](images/ce250628375c545e9deaca869d4385bf7416840219100fb61ab551d30cc60b7f.jpg)  
(a)  
(b)  
Figure 12: Predicted head-neck shape across gender and BMI. (a) Mean surfaces from gender-specific BMI regressions at BMI 18.5, 23, 25 and 30 kg/m<sup>2</sup>. (b) Surface displacement between successive BMI levels. Red and blue indicate outward and inward normal displacement, respectively.

## 3.2. Model Comparisons

## 3.2.1. Neutralization vs. Non-neutralization

Without posture neutralization, the first five PCs mixed size variation with changes in cervical posture. The model was fitted to registered surfaces from the same 200 participants, and these components explained

88.68% of the total variance. PC1 mainly represented overall size. PC2, PC3 and PC5 combined changes in head and neck shape with cervical bending.

The enlarged cervical chains quantified this posture variation at the generated endpoints (Fig. 13). Without neutralization, the absolute HACK sagittal boneaxis angle diferences were 13 50<sup>◦</sup>, 15 38<sup>◦</sup> and 6 14<sup>◦</sup> for PC2, PC3 and PC5, respectively. The neutralized chains shared the imposed angle of 27 59<sup>◦</sup>. The changes within the non-neutralized PCs show that shape variation cannot be interpreted independently of posture in this model.

SSM Comparison. Posture neutralization concentrated more variance in the first components (Fig. 14a). PC1 explained 68.19% of coordinate variance after neutralization and 63.22% without neutralization. The first five components explained 89.52% and 88.68%, respectively.

Generalization error decreased with the number of retained components (Fig. 14b). At five components, RMSE was 4 404 ± 1 013 mm with neutralization and 4 645 ± 1 071 mm without neutralization. At ten components, the errors were $3 . 3 5 3 \pm 0 . 7 6 4$ and 3 308 ± 0 767 mm, respectively.

Specificity error was lower after neutralization across the first ten components (Fig. 14c). At ten components, it was 10 107 ± 2 643 mm with neutralization and $1 1 . 1 1 8 \pm 2 . 8 8 8$ mm without neutralization.

## 3.2.2. Model Construction Methods

We construct a PCA-based SSM that represents shape variation along the leading eigenvectors of the coordinate covariance matrix [15, 52]. We compare this representation with three alternatives. ICA seeks statistically independent components rather than directions of maximum variance [53]. Polynomial-kernel KPCA uses polynomial feature interactions to represent nonlinear variation, whereas RBF-kernel KPCA uses Gaussian similarities between shape vectors [54]. These baselines compare linear variance-based, linear independencebased, and nonlinear kernel-based representations of shape variation.

Compactness difered among the four representations (Fig. 15a). Five components retained 89.52% of coordinate variance for PCA and 13.90% of total independence measure for ICA. Polynomial and RBF KPCA retained 80.67% and 52.89% of their respective kernelfeature variance.

Generalization performance was best for PCA among all tested methods at five, ten, and twenty components (Fig. 15b). At 20 components, mean RMSE was 2 451± 0 538 mm for PCA, $3 . 3 8 1 \pm 0 . 8 2 0$ mm for ICA, 3 932 ± 1 538 mm for polynomial KPCA, and $3 . 1 7 2 { \pm } 1 . 0 3 5$ mm for RBF KPCA.

Specificity yielded a diferent ranking from generalization (Fig. 15c). At 20 components, the errors were $1 0 . 8 5 1 \pm 2 . 5 0 8$ mm for PCA, $6 . 4 7 4 \pm 1 . 7 2 9$ mm for

ICA, 107 672 ± 44 021 mm for polynomial KPCA, and $8 . 9 4 3 \pm 2 . 0 0 9$ mm for RBF KPCA. ICA achieved the lowest specificity error, whereas PCA achieved the lowest generalization error. The large polynomial-KPCA specificity error indicates unstable preimage extrapolation under the chosen sampling range.

## 4. Discussions

To our knowledge, this is the first posture-neutralized parametric model focused on head-neck-shoulder anthropometry that preserves physical size. Existing models mainly describe craniofacial variation in the head or head–neck regions and do not include the shoulders [19, 23]. Our model covers the head, neck, and shoulders in a common estimated cervical pose, providing a foundation for the design of neck-related products.

## 4.1. Anthropometry

Our posture-neutralized model showed greater compactness and lower specificity error than the matched non-neutralized model. Earlier whole-body normalization also increased compactness and reduced bodyshape prediction error from anthropometric measurements [17]. Posture correction similarly improved the accuracy of a hand SSM relative to a model without correction [55]. These studies and our results support controlling joint posture before statistical shape analysis. In our model, held-out reconstruction error improved at five components but not at ten. The present framework applies this approach to head-neck-shoulder anthropometry while retaining physical size.

Our PC1 represented coordinated size variation across the head, neck and shoulders, whereas PC2 mainly involved neck circumference and shoulder width. Earlier head models also associated PC1 with overall size [19, 56], but linked PC2 to facial proportions or width [19, 29]. Overall size is therefore a shared pattern, while the dominant regional proportions difer. Including the neck and shoulders may therefore shift the leading variation towards these regions.

Our gender-specific regressions linked higher BMI to outward changes around the lower face, neck and shoulders in the predicted mean shapes (Fig. 12). Previous head models showed earlier maturation of head size in girls than boys [57] and wider, shorter heads in Chinese than Caucasian populations [58]. A shared finding is that population characteristics are associated with differences in average geometry. Our results concern BMIrelated variation across the adult head-neck-shoulder region, whereas these studies describe head growth and population diferences.

![](images/82bb236425187a0691836f966a1b6eda36f5db8cc2c87ea62daa92d0abbe3c09.jpg)  
Figure 13: Enlarged cervical-chain Cobb angle comparison for the first five PC endpoints. Rows show (a) the neutralized model imposed and (b) the non-neutralized model with natural poses. Blue solid chains denote +3 SD, and orange dashed chains denote −3 SD.

![](images/6562579954a45c36d25eacb6208924294386e1a4a7679103b115387c05b79c01.jpg)  
(a)

![](images/a26b2aa1199fb08761b5f251bdbffc974c55a0c16b562266de18ae3d0710f3aa.jpg)  
(b)

![](images/2260ac48b6adf517b9fe8d3fe09251c05290e7c18359c9b3c82ce763fd9381a4.jpg)  
(c)  
Figure 14: SSM evaluation with and without posture neutralization for the same participants. (a) Compactness: individual PC variance fractions and cumulative coordinate variance for the first ten components. (b) Generalization: mean corresponding-vertex RMSE on held-out surfaces using a shared five-fold partition. (c) Specificity: mean nearest-training-shape RMSE for generated shapes per dimension.

## 4.2. Applications

The posture-neutralized HNS shape space is a design input, not a treatment target for an individual user. It can support population coverage analysis, size grouping, representative virtual forms, and parameterized prototypes. These uses extend established applications of 3D anthropometry in sizing systems and product design [2, 3]. Designers can select shapes that span the retained variation, inspect the associated surface regions and measurements, and test candidate product geometries before physical prototyping [59]. This approach may reduce reliance on a single mean form or on independent one-dimensional measurements when a product contacts several head, neck, and shoulder regions.

For pillows and cervical supports, the model can provide candidate information on neck height, neck circumference, cervical contour, and the head-neck relationship [60]. For helmets and other head-worn devices, it can also inform the head dimensions, shoulder clearance, and contact regions that should be considered together. A low-dimensional PCA representation is practical for such iterations because it achieved the lowest held-out reconstruction error among the tested representations. This result supports its use in the present design workflow. It does not establish that PCA covers every valid HNS shape or that a lower specificity error alone would define a better model. The model also does not validate comfort, pressure distribution, wearing stability, or clinical benefit. Those outcomes require productspecific tests [4, 14, 61].

![](images/88ea7e1c01b3546bf9026dc669cb257e3adefa142f134f7bd7d2cf79dfaca00c.jpg)  
(a)

![](images/1f4a53597e76ee714df1371c559a2fdff0bfb8b4c777104f66de1671de53b0cc.jpg)  
(b)

![](images/18e380a38c26880a6f36f181f7b485f9716019827c97a89d257eb53a1203b220.jpg)  
(c)  
Figure 15: Evaluation of four SSM representations. (a) Compactness for the first 50 components of the full dataset. (b) Generalization for the first 20 components. (c) Specificity for the first 20 components on a logarithmic vertical axis.

## 5. Conclusion

This study developed a posture-neutralized statistical shape model for the head-neck-shoulder region from three-dimensional scans of 200 Chinese adults. Mapping each scan to a common estimated cervical pose before PCA reduced the mixing of cervical bending with the leading shape modes seen in the matched model without neutralization. Eleven anthropometric measurements made these modes easier to interpret, and PCA had the lowest generalization error among the tested representations. Within this cohort, the neutral-pose shape space may support population coverage analysis, size grouping, representative virtual forms, and parameterized ergonomic prototypes. External cohorts, repeated and multi-posture scans, and product tests are needed to assess transferability, reproducibility, fit, pressure, and comfort.

## 6. Acknowledgments

This work was supported by the National Natural Science Foundation of China (NSFC) Youth Grant (Project No. 52605353), the Innovation and Technology Fund (ITF) through The Hong Kong Research Institute of Textiles and Apparel (HKRITA) (Project No. ITP/016/26TI), the Science and Technology Development Fund of Macau SAR (Project No. 0057/2025/ITP1), Macao Polytechnic University (Project No. RP/FCA-15/2025), and Scientific Research Project of the Health System in Pingshan District, Shenzhen (Project No.2025097).

## References

[1] J. Zhang, Y. Luximon, P. Shah, P. Li, 3D statistical head modeling for face/head-related product design: A state-of-the-art review, Computer-Aided Design 159 (2023) 103483.

[2] D. Lacko, T. Huysmans, J. Vleugels, G. De Bruyne, M. M. Van Hulle, J. Sijbers, S. Verwulgen, Product sizing with 3D anthropometry and k-medoids clustering, Computer-Aided Design 91 (2017) 60–74.

[3] C.-C. Kuo, M.-J. Wang, J.-M. Lu, Developing sizing systems using 3D scanning head anthropometric data, Measurement 152 (2020) 107264:1–107264:11.

[4] J. Langley, S. Pancani, K. Kilner, H. Reed, A. Stanton, N. Heron, S. Judge, A. McCarthy, S. Baxter, C. Mazza, C. J. McDermott,\` A comfort assessment of existing cervical orthoses, Ergonomics 61 (2) (2018) 329–338.

[5] A. Luximon, Y. Zhang, Y. Luximon, M. Xiao, Sizing and grading for wearable products, Computer-Aided Design 44 (1) (2012) 77–84.

[6] L. F. Fatharani, R. Purnamasari, M. R. D. Ramadhan, G. P. Adhyatma, Prevalence and characteristics of forward head posture among university students, Journal of Public Health and Pharmacy 5 (1) (2025) 96–104.

[7] R. Intaruk, P. Sengsoon, Diferences in posture, neck angle, and body discomfort during various electronic device usage with virtual classroom, International Journal of Environmental Research and Public Health 22 (9) (2025) 1418:1–1418:13.

[8] S. Lee, Y.-H. Choi, J. Kim, Efects of the cervical flexion angle during smartphone use on muscle fatigue and pain in the cervical erector spinae and upper trapezius in normal adults in their 20s, Journal of Physical Therapy Science 29 (5) (2017) 921–923.

[9] K. K. Hansraj, Assessment of stresses in the cervical spine caused by posture and position of the head, Surgical Technology International 25 (25) (2014) 277–286.

[10] N. F. Mahmoud, K. A. Hassan, S. F. Abdelmajeed, I. M. Moustafa, A. G. Silva, The relationship between forward head posture and neck pain: a systematic review and meta-analysis, Current Reviews in Musculoskeletal Medicine 12 (4) (2019) 562–577.

[11] S. Yang, M. Boudier-Reveret, Y. G. Yi, K. Y. Hong, M. C.´ Chang, Treatment of chronic neck pain in patients with forward

head posture: a systematic narrative review, Healthcare 11 (19) (2023) 2604:1–2604:11.

[12] L. Zhuo, S. Li, F. Zhou, S. Wang, Incidence and costs of cervical spondylosis in urban China from 2013 to 2017: a nationwide population-based study, The Lancet Regional Health–Western Pacific 68 (2026) 101820.

[13] Y. Lv, W. Tian, D. Chen, Y. Liu, L. Wang, F. Duan, The prevalence and associated factors of symptomatic cervical spondylosis in Chinese adults: a community-based cross-sectional study, BMC Musculoskeletal Disorders 19 (1) (2018) 325:1–325:12.

[14] J.-X. Lei, P.-F. Yang, A.-L. Yang, Y.-F. Gong, P. Shang, X. Yuan, Ergonomic consideration in pillow height determinants and evaluation, Healthcare 9 (10) (2021) 1333:1–1333:25.

[15] T. F. Cootes, C. J. Taylor, D. H. Cooper, J. Graham, Active shape models—their training and application, Computer Vision and Image Understanding 61 (1) (1995) 38–59.

[16] Z. Salahzadeh, N. Maroufi, A. Ahmadi, H. Behtash, A. Razmjoo, M. Gohari, M. Parnianpour, Assessment of forward head posture in females: observational and photogrammetry methods, Journal of Back and Musculoskeletal Rehabilitation 27 (2) (2014) 131–139.

[17] F. Danckaers, T. Huysmans, A. Hallemans, G. De Bruyne, S. Truijen, J. Sijbers, Posture normalisation of 3D body scans, Ergonomics 62 (6) (2019) 834–848.

[18] C. R. Harrison, K. M. Robinette, CAESAR: Summary statistics for the adult population (ages 18-65) of the United States of America, Interim Report AFRL-HE-WP-TR-2002-0170, Air Force Research Laboratory, Human Efectiveness Directorate, Crew System Interface Division, Wright-Patterson Air Force Base, Ohio (2002).

[19] Z. Zhuang, C. Shu, P. Xi, M. Bergman, M. Joseph, Headand-face shape variations of US civilian workers, Applied Ergonomics 44 (5) (2013) 775–784.

[20] Y. Luximon, R. Ball, L. Justice, The 3D Chinese head and face modeling, Computer-Aided Design 44 (1) (2012) 40–47.

[21] H. Wang, W. Yang, Y. Yu, W. Chen, R. Ball, 3D digital anthropometric study on Chinese head and face, in: International Conference and Exhibition on 3D Body Scanning and Processing Technologies, 2018, pp. 287–295.

[22] H. Seo, J. I. Kim, H. Kim, Development of Korean head forms for respirator performance testing, Safety and Health at Work 11 (1) (2020) 71–79.

[23] H. Dai, N. Pears, W. Smith, C. Duncan, Statistical modeling of craniofacial shape and texture, International Journal of Computer Vision 128 (2) (2020) 547–571.

[24] J. Booth, A. Roussos, A. Ponniah, D. Dunaway, S. Zafeiriou, Large scale 3D morphable models, International Journal of Computer Vision 126 (2) (2018) 233–254.

[25] H. Zhu, H. Yang, L. Guo, Y. Zhang, Y. Wang, M. Huang, M. Wu, Q. Shen, R. Yang, X. Cao, FaceScape: 3D facial dataset and benchmark for single-view 3D face reconstruction, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (12) (2023) 14528–14545.

[26] B. Allen, B. Curless, Z. Popovic, The space of human body´ shapes: reconstruction and parameterization from range scans, in: ACM SIGGRAPH, 2003, pp. 587–594.

[27] B. Amberg, S. Romdhani, T. Vetter, Optimal step nonrigid ICP algorithms for surface registration, in: IEEE Conference on Computer Vision and Pattern Recognition, 2007, pp. 1–8.

[28] B. Egger, W. A. P. Smith, A. Tewari, S. Wuhrer, M. Zollhoefer, T. Beeler, F. Bernard, T. Bolkart, A. Kortylewski, S. Romdhani, C. Theobalt, V. Blanz, T. Vetter, 3D morphable face models— past, present, and future, ACM Transactions on Graphics 39 (5) (2020) 157:1–157:38.

[29] J. Zhang, H. Iftikhar, P. Shah, Y. Luximon, Age and sex fac-

tors integrated 3D statistical models of adults’ heads, International Journal of Industrial Ergonomics 90 (2022) 103321:1– 103321:13.

[30] S. Wang, Y. Xu, H. Wang, Characterisation and classification of Chinese male ofice workers’ necks using 3-D body measurements, International Journal of Fashion Design, Technology and Education 10 (1) (2017) 101–109.

[31] J. Zhang, K. Xu, S. Zhao, R. Wang, B. Gu, Automatic recognition of the neck–shoulder shape based on 2D photos, Textile Research Journal 92 (23-24) (2022) 5095–5105.

[32] T. Wang, B. Gu, Automatic recognition and 3D modeling of the neck-shoulder human shape based on 2D images, International Journal of Clothing Science and Technology 33 (5) (2021) 796– 810.

[33] G. Baiamonte, A. Acri, G. Laudani, M. Cal\`ı, R. Ambu, Development of a Methodology to Model an Average Neck Shape from 3D Optical Scanning, in: Design Tools and Methods in Industrial Engineering III, Springer Nature Switzerland, 2024, pp. 57–64.

[34] S. Wang, Y. Xu, H. Wang, Finite element modelling of Chinese male ofice workers’ necks using 3D body measurements, The Journal of the Textile Institute 108 (5) (2017) 766–775.

[35] T. Li, T. Bolkart, M. J. Black, H. Li, J. Romero, Learning a model of facial shape and expression from 4D scans, ACM Transactions on Graphics 36 (6) (2017) 194:1–194:17.

[36] Z. Chai, H. Zhang, J. Ren, D. Kang, Z. Xu, X. Zhe, C. Yuan, L. Bao, REALY: Rethinking the evaluation of 3D face reconstruction, in: European Conference on Computer Vision, 2022, pp. 74–92.

[37] L. Zhang, Z. Zhao, X. Cong, Q. Zhang, S. Gu, Y. Gao, R. Zheng, W. Yang, L. Xu, J. Yu, HACK: Learning a parametric head and neck model for high-fidelity animation, ACM Transactions on Graphics 42 (4) (2023) 41:1–41:20.

[38] S. Ploumpis, J. Bednarik, G. Zoss, R. Guseinov, L. Prasso, P. Chandran, O. Boyne, V. Choutas, T. Bolkart, D. Wang, M. Chai, D. Qiu, S. Winberg, G. Rainer, L. Bridgeman, L. Helminger, E. Collins, D. Vicini, J. Riviere, Y. Boetzel, A. Koumis, S. Moschoglou, J. Busch, C. Herrera, J. Still, S. Ysebert, P. Lincoln, S. Orts Escolano, C. Rhemann, E. Wood, T. Beeler, S. Zafeiriou, GNM Head: A Generative aNthropometric Model of the human head, arXiv preprint arXiv:2607.23687 (2026) 1–35.

[39] D. Anguelov, P. Srinivasan, D. Koller, S. Thrun, J. Rodgers, J. Davis, SCAPE: Shape completion and animation of people, in: ACM SIGGRAPH, 2005, pp. 408–416.

[40] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, M. J. Black, SMPL: A skinned multi-person linear model, ACM Transactions on Graphics 34 (6) (2015) 248:1–248:16.

[41] J. C. Gower, Generalized Procrustes analysis, Psychometrika 40 (1) (1975) 33–51.

[42] P. Mitteroecker, P. Gunz, Advances in geometric morphometrics, Evolutionary Biology 36 (2) (2009) 235–247.

[43] D. Meng, M. Keller, E. Boyer, M. Black, S. Pujades, Learning a statistical full spine model from partial observations, in: Shape in Medical Imaging, 2020, pp. 122–133.

[44] M. Koo, S.-W. Yang, Visual analogue scale, Encyclopedia 5 (4) (2025) 190:1–190:11.

[45] H. Vernon, S. Mior, The Neck Disability Index: a study of reliability and validity, Journal of Manipulative and Physiological Therapeutics 14 (7) (1991) 409–415.

[46] D. J. Buysse, C. F. Reynolds, III, T. H. Monk, S. R. Berman, D. J. Kupfer, The Pittsburgh sleep quality index: a new instrument for psychiatric practice and research, Psychiatry Research 28 (2) (1989) 193–213.

[47] C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou,

M. Pantic, 300 faces in-the-wild challenge: database and results, Image and Vision Computing 47 (2016) 3–18.

[48] A. Bulat, G. Tzimiropoulos, How far are we from solving the 2D & 3D face alignment problem? (and a dataset of 230,000 3D facial landmarks), in: IEEE International Conference on Computer Vision, 2017, pp. 1021–1030.

[49] V. Kazemi, J. Sullivan, One millisecond face alignment with an ensemble of regression trees, in: IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 1867–1874.

[50] International Organization for Standardization, ISO 7250- 1:2017 Basic Human Body Measurements for Technological Design – Part 1: Body Measurement Definitions and Landmarks, Standard (2017).

[51] P. Besl, N. D. McKay, A method for registration of 3-d shapes, IEEE Transactions on Pattern Analysis and Machine Intelligence 14 (2) (1992) 239–256.

[52] I. T. Jollife, J. Cadima, Principal component analysis: A review and recent developments, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374 (2065) (2016) 20150202.

[53] A. Hyvarinen, Fast and robust fixed-point algorithms for inde-¨ pendent component analysis, IEEE Transactions on Neural Networks 10 (3) (1999) 626–634.

[54] B. Scholkopf, A. Smola, K.-R. M¨ uller, Nonlinear component¨ analysis as a kernel eigenvalue problem, Neural Computation 10 (5) (1998) 1299–1319.

[55] Y. Yang, T. Yuan, T. Huysmans, W. S. Elkhuizen, F. Tajdari, Y. Song, Posture-invariant three dimensional human hand statistical shape model, Journal of Computing and Information Science in Engineering 21 (3) (2021) 031006:1–031006:12.

[56] Y. Liu, P. Xi, M. Joseph, Z. Zhuang, C. Shu, L. Jiang, M. Bergman, W. Chen, Variations in head-and-face shape of Chinese civilian workers, Annals of Occupational Hygiene 59 (7) (2015) 932–944.

[57] J. Zhang, F. Fu, X. Shi, Y. Luximon, Modeling 3D geometric growth patterns and variations of children’s heads, Applied Ergonomics 108 (2023) 103933:1–103933:11.

[58] J. Zhang, H. Ren, Y. Luximon, X. Wang, 3D head growth and aging trajectories in Chinese and Caucasian for headwear customization: From modeling to comparison, Applied Ergonomics 131 (2026) 104658.

[59] Y. Luximon, R. M. Ball, E. H. Chow, A design and evaluation tool using 3D head templates, Computer-Aided Design and Applications 13 (2) (2016) 153–161.

[60] H.-L. Chen, D. Cai, Body dimension measurements for pillow design for Taiwanese, Work 41 (Suppl. 1) (2012) 1288–1295.

[61] T. Y. Pang, T. S. T. Lo, T. Ellena, H. Mustafa, J. Babalija, A. Subic, Fit, stability and comfort assessment of custom-fitted bicycle helmet inner liner designs, based on 3D anthropometric data, Applied Ergonomics 68 (2018) 240–248.