# Interpretable Face Contrast Modeling for Smartphone Portrait Rendering

Yujin Cho<sup>1,2</sup> Benoˆıt Pochon<sup>2</sup> Jean-Michel Morel<sup>3</sup> Gabriele Facciolo<sup>1</sup>

<sup>1</sup>Universite Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France ´

<sup>2</sup>DXOMARK, France <sup>3</sup>Lingnan University, Hong Kong SAR, China

{yujin.cho,gabriele.facciolo}@ens-paris-saclay.fr bpochon@dxomark.com jeanmichelmorel@LN.edu.hk

https://github.com/nuniniyujin/FaceContrast

## Abstract

The quality of smartphone portrait photography has become increasingly important to both manufacturers and users. Recent devices use complex computational pipelines to improve facial rendering under varied lighting. These pipelines vary widely across manufacturers, so the same portrait can look noticeably different across smartphones. These differences affect multiple aspects of facial appearance, making perceptual preference difficult to characterize with a single measure. To address this, we propose an interpretableframework that decomposesface contrast into three components: overall facial tonal separation, eye-region contrast, and micro-contrast of fine skin details. We formulate dedicated metrics for each component and validate them via pairwise human judgments, demonstrating that our approach correlates more closely with human contrast perception than traditional contrast measures. To support this analysis, we introduce a realistic mannequin-based dataset designed for systematic facial contrast evaluation, comprising 508 full-resolution captures from 29 smartphones under diverse lighting, with region-specific masks and extensive pairwise perceptual annotations. As a downstream analysis, we show that combining the three component-wise metrics provides a meaningful basis for modeling balanced facial contrast preference in a controlled setting, and we present a preliminary evaluation on a smaller real-portrait dataset indicating that insights derived from the mannequin study can extend to real images. Our work provides a controlled first step toward defining and evaluating face contrast in portrait imaging.

## 1. Introduction

With the recent growth of smartphone photography, portrait imaging has become an increasingly important application. Modern devices rely on complex Image Signal Processing (ISP) pipelines to improve facial rendering under a wide range of lighting conditions [7, 14]. These pipelines combine operations such as multi-frame fusion, tone mapping, denoising, sharpening, and face-specific enhancement, and they vary widely across manufacturers [14, 40, 42]. As a result, the same portrait can appear noticeably different across devices, while the underlying processing remains largely a black box. These differences are further accentuated in portrait photography, where humans are particularly apt at distinguishing subtle differences in faces [24, 36, 63]. Together, these processing and perceptual factors make portrait quality difficult to explain with a single perceptual factor. Instead, multiple attributes such as exposure, detail preservation, noise, and local enhancement are all crucial [7, 12, 56]. One particularly important attribute is face contrast. It influences dynamic range, mid-tone rendering, local contrast, and fine-detail visibility across the face. In practice, variations in face contrast can change the apparent depth of the face, the clarity of the eyes, and the visibility of the skin texture, even when the overall content of the scene remains the same, as illustrated in Figure 1 and 2. Yet face contrast remains underexplored as a standalone perceptual attribute. Prior work has examined specific portrait quality factors, such as noise and texture [7, 12], but has not isolated face contrast itself as a distinct target of analysis. This partially stems from the complex evaluation of face contrast in photography. Whereas general contrast measures are well studied [37], face contrast does not have a single widely accepted definition. Existing work on facial contrast [54, 58, 59] has mostly focused on feature-to-skin luminance or color differences, which do not fully capture how contrast is rendered in photographic portraits. Moreover, the appearance of face contrast is strongly affected by imaging conditions and processing, including illumination, tone reproduction, presence of flare and veil, local enhancement, and texture rendering [18, 48]. As a result, face contrast has mostly been studied under idealized conditions, limiting its generalization across diverse lighting and smartphone rendering conditions.

Existing approaches do not fully address this gap. General natural image quality assessment methods mainly target global perceptual quality [25, 31, 71], whereas works related to face image quality [3, 26] are often primarily designed for recognition utility rather than to understand how facial appearance is rendered. Our goal is different: rather than predicting global portrait quality or selecting the more preferred rendering, we aim to define and quantify the perceived amount of face contrast in a way that is interpretable and directly tied to portrait rendering. In this work, we propose to decompose face contrast into three perceptual components: overall contrast, eye contrast, and micro-contrast. Overall contrast captures coarse tonal separation across the face. Eye contrast is similar, but treated separately due to the importance of eyes in face perception [50, 57]. Finally, micro-contrast captures the visibility of fine facial details in skin regions, focusing on how texture is rendered in the final portrait image rather than on intrinsic skin texture alone. The three components are intended to characterize the perceived amount of contrast in different facial regions and spatial scales. Since more contrast is not necessarily better, we further consider balanced contrast, a holistic preference judgment over the relative balance of these components. This downstream analysis examines how the three component-wise measures relate to contrast preference in the controlled mannequin setting. To study these component-wise contrast measures under controlled variations, we build on prior mannequin-based studies of portrait texture quality [45, 65] and use realistic mannequins to compare facial contrast across devices and lighting conditions. The main contributions of this paper are as follows:

![](images/ae82c330ae1fef419c537d1488a87edc37571fd09eb883509f1e427e127de4f9.jpg)  
(a) 5 lux

![](images/2ca7811f55b1e36da6a31c3f4d11b278d459d1ec57780734a470ad15a5c7ca8c.jpg)  
(b) 100 lux

![](images/ec95418f42694a9f06a4983c07af8de7e9eb9d85528b17be3129eabb89424a5c.jpg)  
(c) 1000 lux

![](images/511ef70e88d731e86114965f5fbf16d58ec500af96c63677a4d30a35e0dbd3ce.jpg)  
(d) 10000 lux  
Figure 1. Examples from the proposed dataset for the two mannequin subjects, Eugene and Diana, shown under four illumination levels and multiple backlight conditions (∆EV = 0, 4, 7, and 9). Each sub-image shows the full controlled capture setup together with an inset of the face crop, highlighting how facial contrast rendering varies across devices and lighting conditions.

• We introduce a controlled mannequin-based dataset with full-resolution captures, region-specific masks, and pairwise human perceptual annotations, enabling controlled evaluation of face contrast across devices and lighting conditions.

• We decompose face contrast in smartphone portrait imaging into three perceptual components: overall contrast, eye contrast, and micro-contrast, and propose dedicated metrics to estimate their perceived amounts using pairwise psychophysical judgments.

• As a downstream analysis on the controlled mannequin dataset, we show that the proposed three-component representation provides a meaningful basis for modeling balanced contrast preference.

• We further evaluate the proposed component-wise metrics on real portrait images, providing preliminary evidence of applicability beyond the controlled setting.

## 2. Related Work

Previous work related to our study includes general image quality assessment, portrait quality assessment, face image quality for recognition, and classical contrast modeling. General image quality assessment (IQA) methods aim to predict overall perceptual image quality, either with reference images [33, 34, 68, 78] or in a no-reference setting. In the no-reference setting, early hand-crafted models rely on natural-scene statistics [38, 39], while more recent approaches [2, 5, 10, 25, 31, 62, 67, 71, 72, 79] are learning-based. These learning-based approaches are typically trained on human-rated image-quality datasets [52, 53, 60]. More recently, multi-modal large language models (MLLMs) based IQA methods use quality-aware prompting and scoring with vision-language representations [30, 70, 73, 74]. These methods model global perceptual quality from image content and distortion patterns, but they are not designed to isolate face contrast as a specific perceptual attribute. Portrait quality assessment methods [7–9, 56] study portrait-specific attributes such as preservation of face detail, exposure, naturalness, noise, and global portrait quality. A related line of work studies face image quality for recognition [3, 4, 6, 22, 26], where the goal is to estimate whether an image is suitable for matching or verification. While both directions are relevant, neither is primarily intended to evaluate face contrast itself.

![](images/56dd9c0d1abb1fa9ed1c2c3b1ee3453a8d5635cc2cd9bc22cfad07e85856e85e.jpg)  
(a) 5 lux, ∆EV = 9  
(b) 5 lux, ∆EV = 9  
(c) 10000 lux, ∆EV = 0  
Figure 2. Examples of face without background captured at two illumination levels (5 and 10000 lux) with different smartphones. Facial rendering varies with lighting and device-specific processing, causing noticeable differences in face contrast.

Contrast has long been studied in imaging, and classical measures such as Michelson contrast, Weber contrast, and RMS contrast remain widely used [37, 43, 55]. These formulations are well established for controlled stimuli, simple image patterns, and global image statistics. Previous work has also shown that contrast becomes harder to define for complex images, where perception depends on spatial structure and local image content [48]. This limitation is especially relevant for portraits, where perceived facial contrast depends not only on overall luminance range but also on the spatial distribution of tonal variation and the visibility of fine details. There have also been attempts to define facial contrast more explicitly in color science, cosmetics, and face perception [54, 58, 59]. In this literature, facial contrast is defined as the luminance or color difference between facial features, such as the eyes, lips, or eyebrows, and the surrounding skin, often through adapted Michelson formulations. Although these definitions are a starting point for our study, they present limitations when exploring portrait photos taken with various smartphones under different lighting conditions. Unfortunately, this ideal lighting scenario is not consistent with the way consumers take portraits, prompting the definition of more robust metrics.

## 3. Contrast Definition

We define face contrast as a multi-scale perceptual property that cannot be fully captured by a single global statistic. We thus decompose it into three complementary components: overall contrast, eye contrast, and micro-contrast. This decomposition was developed through consultations with professional photographers to provide meaningful compositions that can vary independently. It also provides a practical and interpretable framework for face contrast analysis. Figure 2 illustrates the diversity of facial contrast. Although images (a) and (b) were captured under the same lighting condition, they differ markedly because of device-specific face rendering: (a) shows high contrast from cast shadows, whereas (b) exhibits overly harsh skin texture. By contrast, image (c) shows high eye contrast with smooth skin. This diversity reinforces that contrast is an inherently complex attribute and further motivates the decomposition into three components.

• Overall contrast describes the global tonal separation across the face, reflecting how clearly bright and dark are separated [11, 37].

• Eye contrast captures the local luminance separation within the eye region, which is semantically and visually critical in portrait perception [50, 57].

• Micro-contrast quantifies the visibility of fine facial details, such as pores in the skin and small local structures [7, 12].

• Balanced contrast describes the perceived harmony among different aspects of facial contrast. It reflects whether global tonal separation, eye definition, and finedetail visibility are rendered in a natural and visually pleasing balance.

Overall, these definitions capture properties of contrast but are not directly quantifiable. This prompts us to focus on human judgment as the reference for the level of contrast, which we then try to model with specific metrics.

## 4. Dataset construction

We construct a portrait dataset using industry-grade realistic mannequins in a laboratory setup with controlled illumination, scene geometry, and camera motion. The complete dataset contains 508 full-resolution images acquired with 29 smartphones from multiple manufacturers under four illumination conditions (Figure 1). For perceptual annotation, we supplement the original captures with DxO Photo-Lab9 edits that increase or decrease tonal or micro-contrast. This yields 800 images with 2,400 region-specific masks (Fig. 3) [75]. Region-specific masks allow each contrast metric to be computed in its corresponding perceptually relevant region, while reducing interference from other components. The dataset uses two high-fidelity mannequins with different skin tones, Eugene and Diana, which reliably trigger the portrait rendering pipelines of the tested smartphones. Additional details on the mannequins, including materials and reflectance measurements, are provided in the supplementary material (Sec. A.3). The images are organized into five groups based on lighting conditions, as shown in Table 1. Groups 1–4 correspond to four distinct lighting conditions, while Group 5 combines non-overlapping samples from these conditions. Each of Groups 1–4 contains 75 images per mannequin, while Group 5 contains 100 images per mannequin.

Table 1. Lighting conditions used to construct the five comparison groups. ∆EV denotes the calibrated exposure difference between the illumination on the mannequin and the light produced by the LED panel.
<table><tr><td>Group</td><td>Light Uniformity</td><td>∆EV / Illuminance (lux)</td></tr><tr><td>Group1</td><td>Flat</td><td> $\overline { { \Delta \mathrm { E V } = 0 , 5 { \cdot } 1 0 0 \mathrm { l u x } } }$ </td></tr><tr><td>Group2</td><td>Flat</td><td> $\Delta \mathrm { E V } = 0 , 1 0 0 0 { - } 1 0 0 0 0 \ \mathrm { l u x }$ </td></tr><tr><td>Group3</td><td>Slightly non-uniform</td><td> $\Delta \mathrm { E V } = 4 , 1 0 0 { - } 1 0 0 0 \mathrm { l u x }$ </td></tr><tr><td>Group4</td><td>Strongly non-uniform</td><td> $\Delta \mathrm { E V } = 7 , 9 ; 5 { \mathrm { - } } 1 0 0 0 \mathrm { l u x }$ </td></tr><tr><td>Group5</td><td>Mixed</td><td>Groups 1–4</td></tr></table>

![](images/e17b76257f4196752691c61374cda67573c13dd523f8ceede33d8a07ebbd0e4c.jpg)  
Figure 3. Illustration of the face mask $( \Omega _ { f } )$ , eye mask $( \Omega _ { e } ) .$ , and skin mask $( \Omega _ { s } )$ used throughout our work.

## 4.1. Annotation Protocol and JOD Scaling

To align the proposed metrics with perception, we collect pairwise human judgments for each contrast component separately. Each pair is formed from images of the same mannequin identity, keeping facial structure and surface texture fixed across the two images so that observers can focus on differences in photographic rendering rather than differences between subjects. This setup also supports device-wise analysis of contrast rendering, reported in the supplementary material (Sec. F.4). Pairwise judgments allow for more precise annotations than ambiguous absolute scoring [80]. During each annotation, explicit guidelines are provided so that contrast is assessed separately from preference, sharpness, or general image quality. (see the supplementary material Sec. C). The annotation questions are phrased to match each target contrast attribute:

1. Overall contrast: “Which image has the larger range from darks to highlights?”

2. Eye contrast: “Choose the image with the darker pupil and brighter sclera (white of the eye).”

3. Micro-contrast: “Choose the image where the tiny details on the skin surface are more visible.”

4. Balanced contrast: “Which image has more balanced face contrast?”

These task-specific prompts help to disentangle the three contrast components and encourage consistent perceptual judgments among observers. Balanced contrast is collected as a separate annotation task and is used only for the downstream preference analysis, not to define the three contrast components. Across all four tasks, trained image-quality experts completed 20,000 pairwise comparisons over 160 hours in total. To improve annotation efficiency, image pairs are sampled using an active pairwise comparison strategy [20], so that the collected comparisons focus on informative pairs rather than exhaustively enumerating all possible combinations. For each contrast task, pairwise choices are converted into interval-scale values in Just Objectionable Difference (JOD) units using a Thurstone Case V paired-comparison model. A difference of 1 JOD between two images means that 75% of observers would vote for the image with the higher JOD value [49]. The JOD scale is estimated independently for each mannequin and comparison set. For Groups 1–4, a separate scale is estimated for each lighting condition, whereas Group 5 uses a single mixed-lighting scale per mannequin.

## 5. Proposed Face Contrast Measures

We propose metrics for the three face-contrast components defined above: overall contrast, eye contrast, and micro-contrast. These interpretable, perceptually grounded measures are evaluated against human annotations in the following section. Following prior work on facial contrast [54, 58, 59], we analyze facial lightness in the CIELAB color space. Specifically, we use the $L ^ { * }$ channel and normalize it to [0, 1], which provides an approximately perceptual lightness representation while separating tonal variation from chromatic variation. We denote by L the normalized lightness image, and $\Omega _ { f } , \Omega _ { e } ,$ , and $\Omega _ { s }$ denote the face, eye, and skin masks, respectively.

## 5.1. Overall Contrast $\left( C _ { \mathrm { o v e r a l l } } \right)$

Overall contrast captures large-scale luminance variation throughout the face, including shading, illumination gradients, and tonal transitions between bright and dark regions. In portraits, this component contributes strongly to the perceived depth and dimensionality of the face. Traditional global contrast measures such as Michelson contrast [37] do not fully capture this effect as they reduce the image to a single global luminance statistic and ignore the spatial distribution of contrast. We therefore measure overall contrast using a multi-scale formulation based on locally normalized luminance variation. For a pixel $x \in \Omega _ { f }$ in the face mask, we define the local contrast response at scale r as

$$
c _ { r } ( x ) = \frac { \sigma _ { r } ( x ) } { \mu _ { r } ( x ) + \epsilon } ,\tag{1}
$$

where $\mu _ { r } ( x )$ and $\sigma _ { r } ( x )$ denote the local mean and standard deviation of the lightness image $L ,$ respectively, computed over an $r \times r$ square patch centered at $x ,$ and $\epsilon = 1 0 ^ { - 6 }$ is used for numerical stability. This formulation is related to classical local contrast normalization in complex images [48]. By normalizing the local standard deviation by the local mean, it reduces sensitivity to absolute luminance and emphasizes relative tonal variation across the face. As a result, the metric is less dominated by raw brightness level and better reflects how tonal contrast is rendered locally across the face. To capture contrast across multiple spatial scales, we pool the local response over the face mask and average it across multiple scales ${ \mathcal { R } } ,$ where $\left| \Omega _ { f } \right|$ denotes the number of pixels in the face mask:

$$
C _ { \mathrm { o v e r a l l } } = \frac { 1 } { | \mathcal { R } | } \sum _ { r \in \mathcal { R } } \left( \frac { 1 } { | \Omega _ { f } | } \sum _ { x \in \Omega _ { f } } c _ { r } ( x ) \right) .\tag{2}
$$

This multi-scale design is consistent with prior work [18] showing that contrast and illumination effects are perceived across multiple spatial scales. Smaller windows respond to local tonal variation and fine shading transitions, while larger windows capture broader tonal structure across regions such as the cheeks, forehead, and jaw. In our experiments, we set $\mathcal { R } = 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6$ , with $| \mathcal { R } | = 5 ,$ , to capture both local and global facial contrast structures [23]. (see Sec D.2 in supplementary material.)

## 5.2. Eye Contrast $( C _ { \mathrm { e y e } } )$

The eyes are structurally distinct from the rest of the face because they contain both very dark regions (pupil) and very bright ones (sclera). Their contrast is therefore not always well captured by global facial contrast alone. Previous work on face perception [50, 57] also supports treating the eye region as a separate component of face contrast due to its singular role in face recognition. In addition, directional lighting can also strongly affect the separation between the pupil, iris, and sclera, making the eyes appear flatter or grayer. We thus treat eye contrast as a separate component and measure it from the dominant dark–bright separation within the eye region. We divide the luminance values within the eye mask $\Omega _ { e }$ into two classes using the Otsu thresholding method [47]. The left-and right-eye masks are merged before applying a single Otsu threshold. Let D and B denote the resulting dark and bright pixel sets, respectively. We define eye contrast as

$$
C _ { \mathrm { e y e } } = \mu _ { B } - \mu _ { D } ,\tag{3}
$$

where $\mu _ { B }$ and $\mu _ { D }$ are the mean lightness values of the bright and dark classes.

## 5.3. Micro-Contrast $\left( C _ { \mathrm { m i c r o } } \right)$

Micro-contrast is intended to capture the visibility of fine-scale facial texture in skin regions while remaining relatively insensitive to broader shading and illumination. In photography, micro-contrast is a commonly used but imprecise term, sometimes used interchangeably with texture visibility [21, 69]. Here, we use it more specifically to describe the extent to which fine facial details are rendered in the final portrait image, distinguishing it from a general preference for stronger sharpening or texture enhancement. As micro-contrast depends on both surface texture and the camera pipeline, our controlled mannequin setup fixes the underlying facial texture within each identity. This allows us to isolate how different devices render the same fine-scale details. We now describe each stage of the proposed $C _ { \mathrm { m i c r o } }$ computation.

Log-DoG Decomposition. To reduce the effect of illumination and isolate texture-scale variation, we first operate in the logarithmic domain. Let L denote the lightness image, and define the logarithmic transformed image as $L _ { \mathrm { l o g } } = \log ( L + \epsilon )$ , where ϵ is a small constant for numerical stability. This transform converts multiplicative illumination changes into additive ones, making the representation more stable to brightness differences and allowing the metric to emphasize relative texturescale luminance variations while reducing its sensitivity to global illumination gradients [55]. We then apply a Difference of Gaussians (DoG) operator to extract a band of spatial frequencies associated with micro texture, $D _ { \mathrm { l o g } } = G _ { \sigma _ { \mathrm { f i n e } } } * L _ { \mathrm { l o g } } - G _ { \sigma _ { \mathrm { c o a r s e } } * } L _ { \mathrm { l o g } } $ , where $G _ { \sigma }$ denotes Gaussian smoothing with standard deviation σ. Inspired by Gaussian scale-space sampling [41], we use $\sigma _ { \mathrm { f i n e } } ~ = ~ { \frac { 1 } { \sqrt { 2 } } }$ and $\sigma _ { \mathrm { c o a r s e } } ~ = ~ { \frac { 3 } { \sqrt { 2 } } }$ , which provide a band-pass response that suppresses broad shading while preserving fine facial detail [51]. The ratio $\sigma _ { \mathrm { c o a r s e } } / \sigma _ { \mathrm { f i n e } } = 3$ follows a previously established center–surround DoG configuration [16]. The use of DoG is motivated by early vision models of center–surround filtering and edge detection [35], and can be interpreted as an approximation of Peli-style local contrast normalization [18, 48].

Local Energy Pooling. The DoG response $D _ { \mathrm { l o g } }$ is a signed signal, and its local mean may be close to zero even when strong texture is present. To obtain a positive measure of local textural strength, we compute the local RMS energy over a finite neighborhood. For each pixel $x \in \Omega _ { s } ^ { } ,$ , let $N ( x )$ denote an 11 × 11 window centered at x. The local energy is defined as

$$
E _ { \mathrm { l o c a l } } ( x ) = \sqrt { \frac { 1 } { | N ( x ) | } \sum _ { y \in N ( x ) } D _ { \log } ( y ) ^ { 2 } } .\tag{4}
$$

This local energy representation is consistent with micro-contrast as a measure of fine-grained local variation, as well as with the classical Filter–Rectify–Filter (FRF) framework for texture perception and early visua processing. The latter has been used to explain how the visual system estimates texture energy and texture boundaries [1, 27, 28].

Perceptual Response Compression. Following Stevens power law [61], perceived magnitude is often modeled as a power function of stimulus intensity. In the specific case of contrast perception, Legge and Foley [29] showed that the visual system applies a non-linear transducer that progressively compresses higher contrast levels. In other words, the physical strength of the local texture does not translate linearly into the perceived texture contrast. To model perceptual saturation, we apply a power-law nonlinearity to the local energy:

$$
S _ { \mathrm { l o c a l } } ( x ) = \left( E _ { \mathrm { l o c a l } } ( x ) \right) ^ { 0 . 6 } .\tag{5}
$$

The exponent 0.6 is chosen to reflect supra-threshold compression of perceived contrast, reducing the dominance of strong local responses. This nonlinearity is applied to local energy rather than directly to luminance values, consistent with response-compression models operating on the outputs of spatially filtered visual channels [19].

Robust Spatial Pooling. The final micro-contrast measure is obtained by robustly pooling the perceptual local responses over the skin mask $\Omega _ { s }$ to reduce the influence of specular highlights, sensor noise, and mask boundary artifacts. Let $q _ { 1 0 }$ and q<sub>90</sub> denote the 10th and 90th percentiles of the set $\{ S _ { \mathrm { l o c a l } } ( x ) \mid x \in \Omega _ { s } \}$ . We define

$$
\begin{array} { l } { \displaystyle { C _ { \mathrm { m i c r o } } = \frac { 1 } { | \mathcal { K } | } \sum _ { x \in { \mathcal { K } } } S _ { \mathrm { l o c a l } } ( x ) , } } \\ { \displaystyle { \mathcal { K } = \{ x \in \Omega _ { s } \mid q _ { 1 0 } \leq S _ { \mathrm { l o c a l } } ( x ) \leq q _ { 9 0 } \} . } } \end{array}\tag{6}
$$

This formulation improves robustness to outlier responses while preserving a perceptually meaningful measure of micro-contrast in skin regions. Parameter sensitivity analysis in Sec. D.4 and Fig. S13 show that the metric remain stable under reasonable variations.

## 6. Experiments and Results

We evaluate the proposed methods under two complementary settings. (i) Within-lighting evaluation is conducted separately for Groups 1–4 using a device-disjoint train/validation split comprising 19 training devices and 10 validation devices. It measures generalization to unseen devices within a fixed lighting condition. The complete device allocation is reported in the supplementary material (Sec. F.2). (ii) Mixed-lighting evaluation is conducted on Group 5, which contains images from multiple lighting conditions that do not overlap with Groups 1–4 and are assessed on a common JOD scale. For fitted balanced contrast preference predictors, Groups 1–4 are used for parameter estimation and Group 5 is used exclusively for evaluation. We evaluate the Overall, Eye, and Micro contrast measures by computing Pearson’s linear correlation coefficient (PLCC) and Spearman’s rank correlation coefficient (SRCC) against their corresponding component-specific JOD scores, as reported in Table 2. For balanced-contrast preference, the three contrast components are instead jointly fitted to predict balanced-contrast JOD, with results reported in Table 3.

Table 2. Comparison of proposed and reference measures for overall, eye, and micro-contrast under within-lighting devicedisjoint evaluation on Groups 1–4 and mixed-lighting evaluation on Group 5. Results are based on 500 bootstrap resamples.
<table><tr><td>Contrast</td><td>Metric</td><td>Within-lighting PLCC SRCC</td><td></td><td>Mixed-lighting PLCC</td><td>SRCC</td></tr><tr><td rowspan="8">Overall</td><td>Proposed overall contrast</td><td>0.792</td><td>0.780</td><td>0.688</td><td>0.693</td></tr><tr><td>Global log-luminance std.</td><td>0.723</td><td>0.708</td><td>0.604</td><td>0.566</td></tr><tr><td>Global luminance std</td><td>0.675</td><td>0.646</td><td>0.551</td><td>0.562</td></tr><tr><td>Michelson contrast (p2–p98) [37]</td><td>0.731</td><td>0.710</td><td>0.583</td><td>0.542</td></tr><tr><td>Facial feature contrast [58]</td><td>-0.026</td><td>-0.005</td><td>-0.302</td><td>-0.263</td></tr><tr><td>Normalized facial feature contrast [54]</td><td>-0.019</td><td>-0.016</td><td>-0.254</td><td>-0.259</td></tr><tr><td>Facial contrast vector magnitude [59]</td><td>0.137</td><td>0.226</td><td>-0.281</td><td>-0.103</td></tr><tr><td>Otsu-based eye contrast [47]</td><td>0.879</td><td>0.855</td><td>0.892</td><td>0.891</td></tr><tr><td rowspan="4"> $_ \mathrm { E y e }$ </td><td>Dark-bright separation  $( h _ { i } ( 1 - l _ { o } ) )$ </td><td>0.852</td><td>0.832</td><td>0.901</td><td>0.901</td></tr><tr><td>Percentile range  $( h _ { i } - l _ { o } )$ </td><td>0.847</td><td>0.855</td><td>0.882</td><td></td></tr><tr><td>Michelson normalized range [37]</td><td>0.680</td><td></td><td></td><td>0.892</td></tr><tr><td></td><td></td><td>0.704</td><td>0.703</td><td>0.677</td></tr><tr><td rowspan="7">Micro</td><td>Proposed micro-contrast</td><td>0.849</td><td>0.842</td><td>0.797</td><td>0.787</td></tr><tr><td>GLCM contrast [17]</td><td>0.682</td><td>0.753</td><td>0.594</td><td>0.706</td></tr><tr><td>GLCM dissimilarity [17]</td><td>0.812</td><td>0.829</td><td>0.763</td><td>0.783</td></tr><tr><td>LBP entropy [46]</td><td>0.257</td><td>0.232</td><td>0.371</td><td>0.305</td></tr><tr><td>Wavelet energy (L1) [32]</td><td>0.804</td><td>0.819</td><td>0.759</td><td>0.775</td></tr><tr><td>CPBD [44]</td><td>0.336</td><td>0.305</td><td>0.358</td><td>0.283</td></tr><tr><td>S3 sharpness [66] Michelson contrast  $\left( { \mathsf { p } } 2 { \mathsf { - p } } 9 8 \right) [ 3 7 ]$ </td><td>0.490 0.223</td><td>0.478 0.241</td><td>0.604 0.282</td><td>0.564 0.256</td></tr></table>

## 6.1. Amount of Contrast Analysis

Overall Contrast. We compare the proposed metric with classical luminance statistics and prior facial contrast measures [54, 58, 59]. It achieves the highest nominal correlations, with PLCC/SRCC of 0.792/0.780 for withinlighting and 0.688/0.693 for mixed-lighting evaluation. Global luminance measures are competitive within lighting but degrade under mixed lighting. Feature-to-skin measures show weaker correlations, as they quantify luminance or chromatic differences between facial features and the surrounding skin rather than face-wide shading and tonal variation. These results suggest that overall facial contrast depends on the spatial distribution of relative tonal variation across the face.

Eye Contrast. We compare the Otsu-based formulation with three reference measures derived from the luminance distribution within the eye mask. Let $h _ { i }$ and $l _ { o }$ denote the 98th and 2nd percentiles of eye-region lightness, respectively. The Otsu-based formulation achieves PLCC/SRCC values of 0.879/0.855 for within-lighting and 0.892/0.891 for mixed-lighting evaluation. The percentile-based measures perform comparably; dark–bright separation is slightly higher under mixed-lighting, while Otsu performs slightly better overall within-lighting. On average however, Otsu achieves higher performance without needing any parameter tuning to remove outliers. Overall, the results show that dark–bright luminance separation is a strong indicator of perceptual eye contrast.

Micro-Contrast. We compare the proposed formulation with conventional texture descriptors, including GLCM [17], LBP entropy [46], and wavelet energy [32], as well as the perceptual sharpness measures CPBD [44] and S3 [66]. Although these methods were not specifically designed for facial micro-contrast, they provide relevant baselines for assessing alignment with human judgments of fine facial texture. The proposed metric achieves the highest nominal correlations in both settings, reaching PLCC/SRCC values of 0.849/0.842 for within-lighting and 0.797/0.787 for mixed-lighting evaluation. GLCM dissimilarity and Wavelet energy also provides good performance, suggesting that local gray-level co-occurrence pattern and spatial-frequency energy capture perceptually relevant skin micro-contrast. However, CPBD and S3 show lower correlations, indicating that perceived micro-contrast is not fully characterized by image sharpness alone. LBP entropy and Michelson contrast [37] perform substantially worse, indicating that structural diversity or luminance range are insufficient to capture perceived micro-contrast.

## 6.2. Balanced Contrast Preference Analysis

As a downstream analysis, we examine whether the three proposed contrast components provide a useful basis for modeling balanced contrast preference. In this task, annotators selected the image with the most balanced contrast, reflecting an overall judgment rather than any single component in isolation. Our best variant is a secondorder polynomial regressor over the three proposed contrast components (see supplementary Sec D.6). As shown in Table 3, this model reaches PLCC/SRCC values of 0.728/0.739 under within-lighting and 0.773/0.778 under mixed-lighting. Each component alone yields non-trivial but suboptimal performance, indicating that balanced contrast preference is not explained by a single attribute. We further compare against five related families: hand-crafted IQA [38, 39], general IQA [2, 10, 25, 62, 67, 71], MLLMbased IQA [30, 70, 73], portrait IQA [9], and face-quality assessment [3, 4, 6, 22, 26]. The IQA methods primarily target global perceptual image quality, whereas portrait IQA focuses on portrait quality. Face-quality models assess facial image quality, often in the context of biometric utility or recognition-related tasks. We evaluate both offthe-shelf and after lightweight task adaptation to balanced contrast preference. Before adaptation, most learning-based methods perform markedly worse, indicating a gap between their original objectives and balanced contrast assessment. MLLM-based IQA models show relatively strong zero-shot performance. Given the limited training data, we evaluate these large models zero-shot to avoid unreliable adaptation.

For a fairer comparison, we fine-tune remaining learned baselines using lightweight trainable heads while keeping the pretrained representation largely frozen. After adaptation, performance improves substantially for several learned baselines, especially face-quality models. Among all methods, fine-tuned IFQA [22] achieves the highest performance. IFQA was pretrained using human preference judgments of restored faces, which may make its representation particularly suitable for this task. Nevertheless, our proposed component-based model remains the strongest among learning-free methods and the second-best method overall. Combining our metrics surpasses most neuralnetwork-based baselines, showing that the proposed decomposition captures a substantial part of balanced contrast preference in the controlled setting. The remaining gap to fine-tuned IFQA suggests that balanced contrast judgments may also depend on additional cues beyond the three explicit contrast components. This latter aspect is where neural-network approaches shine: using large amounts of pretraining data, they are able to efficiently model human preference. While our method performs best in low-data regimes or when interpretability is desired, learning-based approaches can be used to capture the remaining subtlety in human contrast preferences. We expect the interaction between the two paradigms to yield improved models of contrast preference.

## 6.3. Generalization to Real Portrait Data

To examine whether the mannequin-based findings extend to real portraits, we collected a smaller, less controlled dataset of 444 images from 10 subjects/scenes. It contains 29 unique capture devices, 18 of which are not present in the mannequin dataset. We generate contrast-adjusted variants using DxO PhotoLab 9 to obtain at least 44 images per scene for reliable pairwise comparison. It includes more varied facial appearances and poses, including three-quarter views (Supplementary Fig. S14). We collected pairwise annotations for overall contrast, eye contrast, and microcontrast using the same questions as in the controlled study and converted them to JOD scores, requiring 80 annotation hours. We compare the proposed zero-shot metrics with IFQA fine-tuned on the mannequin annotations (Table 4). Our metrics outperform fine-tuned IFQA for eye and microcontrast, while IFQA performs better for overall contrast. Without fitting on mannequin or real-portrait annotations, the proposed metrics retain meaningful correlations on real portraits. We additionally report scene-wise performance with uncertainty in Supplementary Table S9. These results suggest that the mannequin dataset can bridge controlled contrast analysis and real-portrait evaluation. Broader evaluation across identities, skin tones, poses, and capture conditions remains important future work.

Table 3. Balanced contrast preference prediction before and after task adaptation. We compare the proposed component-based method with hand-crafted, general IQA, portrait IQA, and face-quality baselines. “3 (raw)” denotes the fitted coefficients of $a x ^ { 2 } + b x + c$ for a scalar score. Our combined model uses a second-order polynomial over three contrast components with 9 trainable parameters. Entries are omitted when a before- or after-adaptation counterpart is not applicable. All results are computed from 500 bootstrap resamples.
<table><tr><td rowspan="3">Type</td><td rowspan="3">Method</td><td rowspan="3">Expl.</td><td rowspan="3">Total</td><td rowspan="3">Train.</td><td colspan="4">Before Adaptation</td><td colspan="4">After Adaptation</td></tr><tr><td colspan="2">Within-lighting</td><td colspan="2">Mixed-lighting</td><td colspan="2">Within-lighting</td><td colspan="2">Mixed lighting</td></tr><tr><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td></tr><tr><td rowspan="6">Hand-crafted</td><td>BRISQUE [38]</td><td></td><td>3 (raw)</td><td>3 (raw)</td><td>0.115</td><td>0.063</td><td>0.223</td><td>0.190</td><td>0.169</td><td>0.134</td><td>0.235</td><td>0.316</td></tr><tr><td>NIQE[39]</td><td>&gt;&gt;</td><td>3 (raw)</td><td>3 (raw)</td><td>0.411</td><td>0.430</td><td>0.533</td><td>0.603</td><td>0.397</td><td>0.405</td><td>0.461</td><td>0.577</td></tr><tr><td>Ours (Overall)</td><td>√</td><td>3 (raw)</td><td>3 (raw)</td><td>0.567</td><td>0.602</td><td>0.579</td><td>0.586</td><td>0.689</td><td>0.637</td><td>0.647</td><td>0.589</td></tr><tr><td>Ours (Eye)</td><td></td><td>3 (raw)</td><td>3 (raw)</td><td>0.456</td><td>0.447</td><td>0.623</td><td>0.596</td><td>0.647</td><td>0.570</td><td>0.645</td><td>0.634</td></tr><tr><td>Ours (Micro)</td><td>&gt;&gt;</td><td>3 (raw)</td><td>3 (raw)</td><td>0.465</td><td>0.523</td><td>0.574</td><td>0.586</td><td>0.546</td><td>0.542</td><td>0.669</td><td>0.628</td></tr><tr><td>Ours (combined)</td><td>√</td><td>9 (raw)</td><td>9 (raw)</td><td></td><td></td><td></td><td></td><td>0.728</td><td>0.739</td><td>0.773</td><td>0.778</td></tr><tr><td rowspan="6">General IQA</td><td>ARNIQA [2]</td><td>X</td><td>23.56M</td><td>4K</td><td>0.252</td><td>0.251</td><td>0.474</td><td>0.509</td><td>0.355</td><td>0.319</td><td>0.308</td><td>0.296</td></tr><tr><td>CLIPIQA [67]</td><td>X</td><td>102.06M</td><td>1K</td><td>0.474</td><td>0.495</td><td>0.540</td><td>0.581</td><td>0.635</td><td>0.612</td><td>0.759</td><td>0.739</td></tr><tr><td>HyperIQA [62]</td><td>×</td><td>27.43M</td><td>2K</td><td>0.285</td><td>0.268</td><td>0.462</td><td>0.478</td><td>0.338</td><td>0.320</td><td>0.377</td><td>0.385</td></tr><tr><td>MANIQA [71]</td><td>×</td><td>135.8M</td><td>296K</td><td>0.412</td><td>0.404</td><td>0.608</td><td>0.619</td><td>0.547</td><td>0.598</td><td>0.709</td><td>0.772</td></tr><tr><td>MUSIQ [25]</td><td>×</td><td>27.13M</td><td>4K</td><td>0.382</td><td>0.402</td><td>0.597</td><td>0.641</td><td>0.353</td><td>0.315</td><td>0.523</td><td>0.488</td></tr><tr><td>TOPIQ[10]</td><td>X</td><td>45.20M</td><td>133K</td><td>0.373</td><td>0.355</td><td>0.587</td><td>0.592</td><td>0.661</td><td>0.642</td><td>0.753</td><td>0.751</td></tr><tr><td rowspan="4">MLLM IQA</td><td>Q-Align [70]</td><td>×</td><td>8.2B</td><td>0</td><td>0.507</td><td>0.511</td><td>0.615</td><td>0.632</td><td></td><td>1</td><td></td><td></td></tr><tr><td>DeQA-Score [73]</td><td>×</td><td>8.2B</td><td>0</td><td>0.609</td><td>0.572</td><td>0.744</td><td>0.753</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Q-Insight [30]</td><td>×</td><td>8.29B</td><td>0</td><td>0.571</td><td>0.590</td><td>0.639</td><td>0.698</td><td></td><td>一</td><td>1</td><td></td></tr><tr><td>Full-HyperIQA [9]</td><td>×</td><td>140.22M</td><td>24K</td><td>0.364</td><td>0.367</td><td>0.584</td><td>0.568</td><td>0.395</td><td>0.394</td><td>0.571</td><td>0.555</td></tr><tr><td rowspan="4">Face Quality</td><td>CR_FIQA_L [6]</td><td></td><td>65.29M</td><td>66K</td><td>0.150</td><td>0.146</td><td>0.319</td><td>0.220</td><td>0.533</td><td>0.545</td><td>0.708</td><td>0.720</td></tr><tr><td>DifFIQA_R [3]</td><td>× ×</td><td>65.29M</td><td>66K</td><td>0.394</td><td>0.390</td><td>0.565</td><td>0.499</td><td>0.554</td><td>0.538</td><td>0.763</td><td>0.781</td></tr><tr><td>GraFIQs_R50 [26]</td><td>×</td><td>43.7M</td><td>66K</td><td>-0.141</td><td>-0.132</td><td>-0.297</td><td>-0.199</td><td>0.539</td><td>0.500</td><td>0.658</td><td>0.654</td></tr><tr><td>eDifFIQA_S [4]</td><td>×</td><td>24.11M</td><td>66K</td><td>0.496</td><td>0.490</td><td>0.588</td><td>0.505</td><td>0.609</td><td>0.598</td><td>0.791</td><td>0.805</td></tr><tr><td></td><td>IFQA (VIT) [22]</td><td>×</td><td>657.04M</td><td>164K</td><td>0.330</td><td>0.283</td><td>0.410</td><td>0.461</td><td>0.858</td><td>0.856</td><td>0.852</td><td>0.868</td></tr></table>

Table 4. Real-portrait generalization results on 444 images across 10 scenes. PLCC and SRCC are computed separately within each scene and then macro-averaged across scenes. Ours is applied in a zero-shot manner, while IFQA is fine-tuned on the mannequinbased training annotations for each corresponding task.
<table><tr><td rowspan="2">Target</td><td>Ours</td><td>IFQA (fine-tuned)</td></tr><tr><td>PLCC SRCC</td><td>PLCC SRCC</td></tr><tr><td>Overall contrast</td><td>0.792 0.825</td><td>0.858 0.862</td></tr><tr><td>Eye contrast</td><td>0.753 0.768</td><td>0.633 0.629</td></tr><tr><td>Micro-contrast</td><td>0.717 0.731</td><td>0.693 0.678</td></tr></table>

![](images/67b52f78902f52f9e63482a9b6e521a0c9d05e18d66cef1193f59c28f70fba6a.jpg)  
Figure 4. Examples of real portraits used in our preliminary realworld analysis. Images span multiple subjects, devices, and edited contrast variants.

## 7. Conclusion

We constructed a realistic mannequin-based dataset captured using multiple smartphones under diverse lighting conditions. For controlled face contrast evaluation, we paired the dataset with region-specific masks and psychophysical annotations. Rather than treating face contrast as a single global quantity, we decomposed it into three perceptual components: overall contrast, eye contrast, and micro-contrast, and proposed an interpretable metric for each attribute. The key improvement over previous metrics lies in leveraging the multi-scale and multifrequency aspects of contrast perception. Our experimental results demonstrate that the three proposed metrics align closely with human contrast perception, and perform strongly against conventional contrast, texture and sharpness measures. These results suggest that face contrast in portrait imaging is better understood as a structured, multi-scale perceptual property rather than as a simple global statistic. As a downstream analysis, we examined the preference for balanced facial contrast in the controlled mannequin setting. The measured overall, eye, and micro-contrast components provide a meaningful basis for explaining holistic contrast preference: each component alone is insufficient, while their combination explains substantially more variation in human judgments. A finetuned face-quality model achieved slightly higher correlations, suggesting that balanced contrast judgments may be influenced by additional cues beyond the three explicit contrast components. We also evaluated the proposed metrics on real-world portraits, showing that our zero-shot metrics generalize beyond the controlled mannequin setting. We hope that the proposed dataset and interpretable formulation will support future work on interpretable portrait quality modeling and face-contrast evaluation.

## 8. Acknowledgements

This work was financed by a grant from ANRT (Association Nationale de la Recherche et de la Technologie) N<sup>◦</sup> 2023/0335. This work used GENCI-IDRIS HPC and storage resources under allocations AD011017622, AD011014895R2, AD011014895R1, and AD011014895R on Jean Zay.

## References

[1] Edward H Adelson and James R Bergen. Spatiotemporal energy models for the perception of motion. Journal of the optical society ofamerica A, 2(2):284–299, 1985. 6

[2] Lorenzo Agnolucci, Leonardo Galteri, Marco Bertini, and Alberto Del Bimbo. Arniqa: Learning distortion manifold for image quality assessment. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 189–198, 2024. 2, 7, 8

[3] Ziga Babnik, Peter Peer, and Vitomir<sup>ˇ</sup> Struc. Diffiqa: Face<sup>ˇ</sup> image quality assessment using denoising diffusion probabilistic models. In 2023 IEEE international joint conference on biometrics (IJCB), pages 1–10. IEEE, 2023. 2, 3, 7, 8

[4] Ziga Babnik, Peter Peer, and Vitomir <sup>ˇ</sup> Struc. ediffiqa: To- <sup>ˇ</sup> wards efficient face image quality assessment based on denoising diffusion probabilistic models. IEEE Transactions on Biometrics, Behavior, and Identity Science, 6(4):458–474, 2024. 3, 7, 8

[5] Sebastian Bosse, Dominique Maniry, Klaus-Robert Muller,¨ Thomas Wiegand, and Wojciech Samek. Deep neural networks for no-reference and full-reference image quality assessment. IEEE Transactions on image processing, 27(1): 206–219, 2017. 2

[6] Fadi Boutros, Meiling Fang, Marcel Klemt, Biying Fu, and Naser Damer. Cr-fiqa: face image quality assessment by learning sample relative classifiability. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5836–5845, 2023. 3, 7, 8

[7] Nicolas Chahine, Stefania Calarasanu, Davide Garcia-Civiero, Theo Cayla, Sira Ferradans, and Jean Ponce. An image quality assessment dataset for portraits. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9968–9978, 2023. 1, 2, 3, 12

[8] Nicolas Chahine, Marcos V Conde, Daniela Carfora, Gabriel Pacianotto, Benoit Pochon, Sira Ferradans, Radu Timofte, Zhichao Duan, Xinrui Xu, Yipo Huang, et al. Deep portrait quality assessment. a ntire 2024 challenge survey. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6732–6744, 2024.

[9] Nicolas Chahine, Sira Ferradans, Javier Vazquez-Corral, and Jean Ponce. Generalized portrait quality assessment. Pattern Recognition Letters, 189:122–128, 2025. 2, 7, 8

[10] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2024. 2, 7, 8

[11] Qiulin Chen and Jan P Allebach. Face image lighting enhancement using a 3d model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2775–2784, 2023. 3

[12] Yujin Cho, Minh Khang Tran, Benoit Pochon, Jean-Michel Morel, Gabriele Facciolo, and Sira Ferradans. Adapting deqa-score for attribute-specific portrait quality assessment. Electronic Imaging, 38:1–8, 2026. 1, 3

[13] Roger R Davidson and Peter H Farquhar. A bibliography on the method of paired comparisons. Biometrics, pages 241– 252, 1976. 12

[14] Mauricio Delbracio, Damien Kelly, Michael S Brown, and Peyman Milanfar. Mobile computational photography: A tour. Annual review ofvision science, 7(1):571–604, 2021. 1

[15] Tim Esler. facenet-pytorch: Pretrained pytorch face detection (mtcnn) and facial recognition (inceptionresnet) models. https://github.com/timesler/facenetpytorch, 2019. GitHub repository, accessed 2026-03-27. 25

[16] Thorsten Hansen and Heiko Neumann. A recurrent model of contour integration in primary visual cortex. Journal of Vision, 8(8):8–8, 2008. 5

[17] Robert M Haralick, Karthikeyan Shanmugam, and Its’ Hak Dinstein. Textural features for image classification. IEEE Transactions on systems, man, and cybernetics, (6):610–621, 2007. 6, 7

[18] Andrew M Haun and Eli Peli. Perceived contrast in complex images. Journal ofvision, 13(13):3–3, 2013. 1, 5

[19] David J Heeger. Normalization of cell responses in cat striate cortex. Visual neuroscience, 9(2):181–197, 1992. 6

[20] Kevin G Jamieson and Robert Nowak. Active ranking using pairwise comparisons. Advances in neural information processing systems, 24, 2011. 4

[21] Pye Jirsa. The ultimate guide to contrast in photography, 2022. 5

[22] Byungho Jo, Donghyeon Cho, In Kyu Park, and Sungeun Hong. Ifqa: Interpretable face quality assessment. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision, pages 3444–3453, 2023. 3, 7, 8

[23] Daniel J Jobson, Zia-ur Rahman, and Glenn A Woodell. A multiscale retinex for bridging the gap between color images and the human observation of scenes. IEEE Transactions on Image processing, 6(7):965–976, 1997. 5

[24] Nancy Kanwisher, Josh McDermott, and Marvin M Chun. The fusiform face area: a module in human extrastriate cortex specialized for face perception. Journal ofneuroscience, 17(11):4302–4311, 1997. 1

[25] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 1, 2, 7, 8

[26] Jan Niklas Kolf, Naser Damer, and Fadi Boutros. Grafiqs: Face image quality assessment using gradient magnitudes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1490–1499, 2024. 2, 3, 7, 8

[27] Michael S Landy and James R Bergen. Texture segregation and orientation gradient. Vision research, 31(4):679–691, 1991. 6

[28] Michael S Landy and Norma Graham. 73 visual perception of texture. The visual neurosciences, pages 1106–1118, 2004. 6

[29] Gordon E Legge and John M Foley. Contrast masking in human vision. Journal ofthe optical Society ofAmerica, 70 (12):1458–1471, 1980. 6

[30] Weiqi Li, Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Jian Zhang, et al. Q-insight: Understanding image quality via visual reinforcement learning. Advances in Neural Information Processing Systems, 38:36802–36827, 2026. 2, 7, 8

[31] Pavan C Madhusudana, Neil Birkbeck, Yilin Wang, Balu Adsumilli, and Alan C Bovik. Image quality assessment using contrastive learning. arXiv:2110.13266, 2021. 1, 2

[32] Stephane G Mallat. A theory for multiresolution signal decomposition: the wavelet representation. IEEE transactions on pattern analysis and machine intelligence, 11(7):674– 693, 2002. 6, 7

[33] Rafal Mantiuk, Scott J Daly, Karol Myszkowski, and Hans-Peter Seidel. Predicting visible differences in high dynamic range images: model and its calibration. In Human Vision and Electronic Imaging X, pages 204–214. SPIE, 2005. 2

[34] Rafal Mantiuk, Kil Joong Kim, Allan G. Rempel, and Wolfgang Heidrich. HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions. ACM Trans. Graph., 30(4):40, 2011. 2

[35] David Marr and Ellen Hildreth. Theory of edge detection. Proceedings of the Royal Society of London. Series B. Biological Sciences, 207(1167):187–217, 1980. 5

[36] Daphne Maurer, Richard Le Grand, and Catherine J Mondloch. The many faces of configural processing. Trends in cognitive sciences, 6(6):255–260, 2002. 1

[37] Albert Abraham Michelson. Studies in optics. Courier Corporation, 1995. 1, 3, 4, 6, 7, 19, 20

[38] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing, 21(12): 4695–4708, 2012. 2, 7, 8

[39] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012. 2, 7, 8

[40] Antoine Monod, Julie Delon, and Thomas Veit. An analysis and implementation of the hdr+ burst denoising method. Image Processing On Line, 11:142–169, 2021. 1

[41] Jean-Michel Morel and Guoshen Yu. Is sift scale invariant? Inverse Problems and Imaging, 5(1):115–136, 2011. 5

[42] Chamin Morikawa, Michihiro Kobayashi, Masaki Satoh, Yasuhiro Kuroda, Teppei Inomata, Hitoshi Matsuo, Takeshi Miura, and Masaki Hilaga. Image and video processing on mobile devices: a survey. The Visual Computer, 37(12): 2931–2949, 2021. 1

[43] Bernard Moulden, Fred Kingdom, and Linda F Gatley. The standard deviation of luminance as a metric for contrast in random-dot images. Perception, 19(1):79–101, 1990. 3

[44] Niranjan D Narvekar and Lina J Karam. A no-reference image blur metric based on the cumulative probability of blur detection (cpbd). IEEE Transactions on Image Processing, 20(9):2678–2683, 2011. 6, 7

[45] Chahine Nicolas and Belkarfa Salim. Portrait quality assessment using multi-scale cnn. In London Imaging Meeting, pages 5–10. Society for Imaging Science and Technology, 2021. 2

[46] Timo Ojala, Matti Pietikainen, and Topi Maenpaa. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on pattern analysis and machine intelligence, 24(7):971–987, 2002. 6, 7

[47] Nobuyuki Otsu. A threshold selection method from graylevel histograms. Automatica, 11:285–296, 1975. 5, 6, 19, 20

[48] Eli Peli. Contrast in complex images. Journal of the optical society of America A, 7(10):2032–2040, 1990. 1, 3, 5

[49] Maria Perez-Ortiz, Aliaksei Mikhailiuk, Emin Zerman, Vedad Hulusic, Giuseppe Valenzise, and Rafał K Mantiuk. From pairwise comparisons and rating to a unified quality scale. IEEE Transactions on Image Processing, 29:1139– 1151, 2019. 4, 12

[50] Matthew F Peterson and Miguel P Eckstein. Looking just below the eyes is optimal across face recognition tasks. Proceedings of the National Academy of Sciences, 109(48): E3314–E3323, 2012. 2, 3, 5

[51] Ana Belen Petro, Catalina Sbert, and Jean-Michel Morel.´ Multiscale retinex. Image processing on line, pages 71–88, 2014. 5

[52] Nikolay Ponomarenko, Vladimir Lukin, Alexander Zelen sky, Karen Egiazarian, Marco Carli, and Federica Battisti. Tid2008-a database for evaluation of full-reference visual quality assessment metrics. Advances of modern radioelectronics, 10(4):30–45, 2009. 2

[53] Nikolay Ponomarenko, Oleg Ieremeiev, Vladimir Lukin, Karen Egiazarian, Lina Jin, Jaakko Astola, Benoit Vozel, Kacem Chehdi, Marco Carli, Federica Battisti, et al. Color image database tid2013: Peculiarities and preliminary results. In European workshop on visual information processing (EUVIP), pages 106–111. IEEE, 2013. 2

[54] Aurelie Porcheron, Emmanuelle Mauger, and Richard Rus-´ sell. Aspects of facial contrast decrease with age and are cues for age perception. PloS one, 8(3):e57985, 2013. 1, 3, 4, 6, 17

[55] Nicolaas Prins et al. Psychophysics: a practical introduction. Academic Press, 2016. 3, 5

[56] Guanyi Qin, Jie Liang, Bingbing Zhang, Lishen Qu, Ya-nan Guan, Hui Zeng, Lei Zhang, Radu Timofte, et al. NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: Professional Image Quality Assessment (Track 1) . In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2026. 1, 2

[57] Jessica Royer, Caroline Blais, Isabelle Charbonneau, Karine Dery, Jessica Tardif, Brad Duchaine, Fr´ ed´ eric Gosselin, and´ Daniel Fiset. Greater reliance on the eye region predicts better face recognition ability. Cognition, 181:12–20, 2018. 2, 3, 5

[58] Richard Russell. A sex difference in facial contrast and its exaggeration by cosmetics. Perception, 38(8):1211–1219, 2009. 1, 3, 4, 6, 17

[59] Richard Russell, Aurelie Porcheron, Jennifer R Sweda,´ Alex L Jones, Emmanuelle Mauger, and Frederique Morizot. Facial contrast is a cue for perceiving health from the face. Journal of Experimental Psychology: Human Perception and Performance, 42(9):1354, 2016. 1, 3, 4, 6, 18

[60] Hamid R Sheikh, Muhammad F Sabir, Alan C Bovik, et al. A statistical evaluation of recent full reference image quality assessment algorithms. IEEE Trans. Image Process., 15(11): 3440–3451, 2006. 2

[61] Stanley S Stevens. On the psychophysical law. Psychologi cal review, 64(3):153, 1957. 6

[62] Shaolin Su, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge, Jinqiu Sun, and Yanning Zhang. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2, 7, 8

[63] James W Tanaka and Martha J Farah. Parts and wholes in face recognition. The Quarterlyjournal ofexperimental psychology, 46(2):225–245, 1993. 1

[64] Louis L Thurstone. A law of comparative judgment. Psychological review, 101(2):266, 1994. 12

[65] Daniela Carfora Ventura, Gabriel Pacianotto Gouveia, Ana Calarasanu, Valentine Tosel, Nicolas Chahine, and Sira Ferradans. From video conferences to dslrs: An in-depth texture evaluation with realistic mannequins. Electronic Imaging, 36:1–6, 2024. 2

[66] Cuong T. Vu, Thien D. Phan, and Damon M. Chandler. S3: A spectral and spatial measure of local perceived sharpness in natural images. IEEE Transactions on Image Processing, 21(3):934–945, 2011. 6, 7

[67] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023. 2, 7, 8

[68] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions in Image Processing, 13(4):600–612, 2004. 2

[69] Mark Wieczorek. Sharpness vs microcontrast, 2018. Published in Ice Cream Geometry. 5

[70] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023. 2, 7, 8

[71] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1191–1200, 2022. 1, 2, 7, 8

[72] Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, and Alan Bovik. From patches to pictures (paq-2-piq): Mapping the perceptual space of picture quality. In Proceedings ofthe IEEE/CVF conference on

computer vision and pattern recognition, pages 3575–3585, 2020. 2

[73] Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14483–14494. IEEE, 2025. 2, 7, 8

[74] Zhiyuan You, Jinjin Gu, Xin Cai, Zheyuan Li, Kaiwen Zhu, Chao Dong, and Tianfan Xue. Enhancing descriptive image quality assessment with a large-scale multi-modal dataset. IEEE Transactions on Image Processing, 2025. 2

[75] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In Proceed ings of the European conference on computer vision (ECCV), pages 325–341, 2018. 3, 25

[76] Changqian Yu, Changxin Gao, Jingbo Wang, Gang Yu, Chunhua Shen, and Nong Sang. Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation. International journal of computer vision, 129 (11):3051–3068, 2021. 25

[77] Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE signal processing letters, 23 (10):1499–1503, 2016. 25

[78] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 2

[79] Weixia Zhang, Kede Ma, Jia Yan, Dexiang Deng, and Zhou Wang. Blind image quality assessment using a deep bilinear convolutional neural network. IEEE Transactions on Circuits and Systems for Video Technology, 30(1):36–47, 2020. 2

[80] Weixia Zhang, Kede Ma, Guangtao Zhai, and Xiaokang Yang. Uncertainty-aware blind image quality assessment in the laboratory and wild. IEEE Transactions on Image Pro cessing, 30:3474–3486, 2021. 4

[81] Peter Zolliker and Zofia Baranczuk. Error estimation of´ paired comparison tests for thurstone’s case v. In Conference on Colour in Graphics, Imaging, and Vision, pages 39–44. Society of Imaging Science and Technology, 2010. 12

# Interpretable Face Contrast Modeling for Smartphone Portrait Rendering

Supplementary Material

## A. Annotation Protocol

## A.1. Annotation Viewing Conditions

All annotations were performed under controlled viewing conditions to limit unintended variation across observers. Images were presented on a calibrated 32-inch 4K BenQ monitor at a fixed viewing distance of 65 cm. The monitor was configured to standard sRGB conditions with a D65 white point, while ambient illumination was regulated to minimize glare and visual fatigue, following [7]. Face crops were displayed at a standardized scale according to the preprocessing procedure described in the main paper. This setup was intended to keep facial detail within a perceptually appropriate range for normal viewing and fine-level comparison.

## A.2. JOD Scaling

For each contrast task, the collected pairwise choices were converted into interval-scale Just Objectionable Difference (JOD) values using a probabilistic paired-comparison scaling framework [49]. Let $q _ { i }$ denote the latent perceptual score associated with image i. For a pair of images (i, j), the observer is assumed to choose image i over image j with probability determined by the difference between their latent scores. Following the Thurstone Case V observer model [13, 64], the internal perceptual response for each image is modeled as a noisy version of its latent score:

$$
r _ { i } = q _ { i } + \epsilon _ { i } ,\tag{1}
$$

where $\epsilon _ { i }$ is a random noise term. Under a Gaussian observer model, the probability that image i is preferred over image j is

$$
P ( i \succ j ) = \Phi \left( \frac { q _ { i } - q _ { j } } { \sigma } \right) ,\tag{2}
$$

where $\Phi ( \cdot )$ denotes the cumulative distribution function of the standard normal distribution and σ controls the observer uncertainty. The latent scores $\left\{ q _ { i } \right\}$ are estimated from the measured pairwise responses by maximizing the likelihood of the observed comparisons. The resulting fitted values define an interval perceptual scale, which we report in JOD units. Under this scaling, a difference of 1 JOD corresponds approximately to 75% agreement in pairwise comparisons.

Intuitively, this procedure maps binary pairwise decisions onto a continuous perceptual axis. Images with similar JOD values are difficult to distinguish perceptually for the target attribute, whereas larger JOD differences correspond to stronger perceived differences. These JOD values are then used as perceptually grounded references for evaluating the proposed contrast metrics. To quantify uncertainty in the recovered perceptual scale, we estimate confidence intervals using the analytical error-estimation framework of Zolliker and Baranczuk [ ´ 81]. Because our annotation protocol uses active pairwise sampling, these intervals should be interpreted as approximate scale-level uncertainty estimates rather than exact finite-sample guarantees. We use them as a practical indicator of the stability of the recovered JOD values under the incomplete comparison design.

## A.3. Controlled Portrait Test Bench

Figure S1 shows examples of images captured using a controlled portrait-imaging test bench. The setup is designed to ensure repeatable acquisition while reproducing realistic portrait scenes. It includes a realistic mannequin, a composite chart, a flare light source, a multispectral lighting system, a ColorChecker with grayscale patches, and an automated moving object. Together, these components allow us to vary facial appearance, illumination, flare, tonal and color references, and temporal scene conditions. The use of a mannequin allows us to capture the same face repeatedly while avoiding variations in expression, pose, and subject motion. Its skin-like reflectance and fine surface details also make it suitable for evaluating how different smartphones render facial appearance under controlled conditions. The mannequins are realistic replicas of human subjects and are specifically designed for image-quality evaluation. They were tested to reliably trigger smartphone portrait modes. The mannequins are constructed from polyurethane foam with proprietary skin-like pigmentation, silicone components, real hair, and glass eyes. Figure S2 shows that the realistic mannequins closely follow the reflectance trends of the corresponding real human skin across the visible wavelength range. The composite chart and the ColorChecker with grayscale patches provide reference structures for checking exposure and color consistency throughout the acquisition process. Illumination is controlled using a multispectral lighting system so that both the spectrum and intensity of the scene can be varied systematically. In this work, lux denotes the measured illuminance on the mannequin, while ∆EV denotes the calibrated exposure difference between the light on the mannequin and the light produced by the LED panel. Thus, lux controls the physical amount of ambient light reaching the setup, whereas ∆EV controls the relative strength of the LED backlight with respect to the mannequin illumination. By varying both factors, we obtain portrait images spanning flat and non-uniform lighting conditions, as well as a broad range of facial contrast appearances. A dedicated flare light source is used to introduce controlled stray-light and veiling-glare conditions, which can reduce tonal separation and affect perceived face contrast. Finally, the automated moving object introduces temporal variation into the scene, allowing the test bench to account for motion effects in multi-frame capture and ISP processing.

![](images/2e08b2aa5af66a5a03f8c9dd9a8e15aee54716a82897dcf3d0adeea59d1edd16.jpg)  
(a) 5 lux

![](images/0a59545c846950bb34e4e0d3a381654208ad84eda0f6686f8fbe1e9bc73ee034.jpg)  
(b) 100 lux

![](images/c1d3bdfdf629f164d62ffd51132828be9149d41177a065171c89a804e30dd56d.jpg)  
(c) 1000 lux

![](images/57233ee5e0a49947b58bcf176c3429962fbede74f83c0d9b36263391b056fe2f.jpg)  
(d) 10000 lux  
Figure S1. Examples from the full-resolution dataset. The two mannequins are Eugene and Diana, and the images were captured using a Samsung Galaxy Z Fold6. The dataset includes multiple ∆EV conditions (0, 4, 7, and 9), but here we show only ∆EV = 0 and ∆EV = 4 under backlit conditions.

## B. Annotation Guidelines Overview

The goal of this annotation task is to break down face contrast into a set of visually meaningful aspects that can be reliably evaluated across different smartphone images and lighting conditions. The appearance of facial contrast is affected by many factors, such as shading, skin reflectance, tone adjustment, noise, and fine details, so we collect annotations that separately describe each of these elements.

To make the evaluation simple and consistent, all images come from a fixed portrait scene using two realistic mannequins: (i) Eugene, with a bright/white skin tone, and (ii) Diana, with a dark skin tone. Because the mannequins have different skin tones, we do not compare them to each other. All comparisons are made only within the same mannequin (Eugene vs. Eugene, and Diana vs. Diana). This ensures that the evaluation focuses on differences created by the smartphone camera rather than differences in skin tone.

During the annotation, you will see masks that highlight specific parts of the face: (i) the face-only area, (ii) the skinonly region, and (iii) the eye region, as shown in Figure S3. These masks are provided to guide your attention toward the areas we want you to evaluate and to reduce distractions from the background or other elements in the scene.

1. Which image has more contrast? For this task, we make three separate annotations. Please note that we do not evaluate color contrast or texture contrast in the photographic sense. Here, color contrast refers to contrast created by differences in color, while texture contrast refers to contrast created by differences in surface texture. Instead, we focus only on tonal contrast for the overall contrast and eye-region contrast questions, and on micro-contrast for the skin-detail evaluation. Also, we are not choosing which image looks better. The goal is simply to decide which image has more contrast. In some cases, the segmented face mask may include small black pixels along the mask boundary. These black pixels are segmentation artifacts and should be completely ignored during the evaluation.

2. Which image has more balanced contrast? For this task, you will see the full face crop region (red box in Figure S3). Here, annotators select the image they prefer between the two images.

## B.1. Overall Contrast

In this task, annotators are shown whole-face masked images without any background. As shown in Figure S4, they compare several face images taken under different illumination and exposure settings. When we talk about overall contrast, we mainly mean tonal contrast: the difference between the dark and bright parts of the face. Bright regions such as the sclera and dark regions such as the eye sockets naturally contribute to the overall tonal range of the face. This is especially important for dark-skinned mannequins, where the eyes may contain the strongest bright–dark separation. However, this task is distinct from the Eye-Region

![](images/70919d736e1e13d13a0beb2c31217adcf7542efe7db03905b7e1c6a69d9ac08e.jpg)  
(a)

![](images/1d6581cbdeb645dbda8563b548c30edb212faa74564888bf39eb20eda10a0796.jpg)  
(b)  
Figure S2. Spectral reflectance comparison between real human skin and the corresponding realistic mannequin surface for (a) Eugene and (b) Diana. The mannequins follow similar wavelengthdependent reflectance trends to the real subjects.

Contrast task: Overall Contrast measures global tonal separation across the whole face, whereas Eye-Region Contrast evaluates only the local tonal structure inside the eye region (e.g., sclera vs. iris vs. pupil).

Lighting can strongly influence overall contrast. For example, in Figures S4a and S4b, strong directional lighting creates a clear separation between bright and dark areas, so these images have higher overall contrast. In contrast, the image in Figure S4c does not have strong lighting differences, so the contrast is more moderate and even. It is important to remember that we are not choosing the image that looks the most pleasing or natural. For example, Figure S4c may look the most pleasing overall, but our goal here is simply to decide which image has more contrast. Finally, in some cases, such as Figure S4d, the face may look flat because of flare or veiling light. This reduces the separation between bright and dark areas, so we consider it to have low overall contrast.

Figure S5 shows an example of a tonal contrast difference. The right image has higher tonal contrast than the left image. In the right image, the pupil is darker and the sclera is brighter. In the beard area, the white hairs appear brighter and the dark hairs appear darker. The separation between individual hairs is also clearer.

## B.2. Eye-Region Contrast

In this task, annotators are shown only the eye masks, without any background. When judging contrast in the eyes, consider the following: contrast is higher when the pupil appears darker and the white of the eye appears brighter. Sometimes, there may be small white reflections on the pupil caused by light. In such cases, a brighter reflection also contributes to a higher-contrast appearance. Please note that natural eye color, such as blue or brown, should not influence the decision. Eye color is not related to the contrast we want to measure.

## B.3. Micro-Contrast

In this task, annotators are shown only the skin masks, without any background. The goal is to judge how clearly the small skin textures appear. Please focus on details such as pores, fine lines, and eyebrow or forehead texture. When these tiny details look sharper and more visible, the image has higher micro-contrast. When they look soft or blurred, the micro-contrast is lower. For the Eugene mannequin, the beard area may also be visible. If the beard texture appears clear, this also counts as higher micro-contrast. Please note that micro-contrast is not about color or lighting. For example, an image taken in a dark environment may look very noisy, making it hard to see the skin texture. In that case, even if the color or brightness is different, the microcontrast should still be judged as low because the fine details are not visible.

Figure S7 shows an example of micro-contrast. The right image has higher micro-contrast. When assessing microcontrast, however, we focus on local details such as the eyebrows, forehead, pores, and beard. In the right image, the pores are more textured and stand out more. The separation between dark and bright hairs is more visible, and the wrinkles on the forehead stand out more clearly. The righ image therefore shows richer local detail.

## B.4. Balanced Contrast Preference

In this task, annotators are shown the face crop region with the background. They select the image they prefer from the two images shown in the comparison tool.

## C. Detailed Annotation Guidelines

The annotations should be performed according to the specific guidelines. To assist the annotators during this process, we associate each guideline with a question that helps clarify what needs to be evaluated. Moreover, we provide brief practical guidance for annotators when choosing one image

(a) Full-resolution image

![](images/beaed1525b0dcf684fc9c5183664e193acf25d20ab87a7e4d745bb148d871edb.jpg)

![](images/4d64a5e872ad6a7826c8bce93f10a4f131c39ac03937f63ca92634eae9d07bbc.jpg)

(a) Bright skin tone mannequin (Eugene), taken under 100 lux – LED4000K light.  
![](images/67ed3f20d5229653c060f5b7a1d85005418f135c04ce73dd50caef70bb937dc7.jpg)  
(b) Dark skin tone mannequin (Diana), taken under 100 lux – LED4000K light with ∆EV = 4.  
Figure S3. Example scenes used to illustrate the annotation protocol for the two mannequin subjects. (a) Eugene under 100 lux, LED 4000K lighting. (b) Diana under 100 lux, LED 4000K lighting with ∆EV = 4. The examples highlight the facial regions used to guide the overallcontrast, eye-contrast, and micro-contrast annotation tasks.

from the two displayed. Contrast rendition can be assessed according to the following guidelines and criteria:

## C.1. Overall Contrast Guideline

Overall contrast means tonal contrast: the intrinsic tonal depth of the image. The main question is: “Does the face show a wide range of tones—from deep dark areas to clean highlights, with distinct midtones in between?” This refers to the global tonal rendering of the image, independent of lighting direction.

In the pairwise comparison tool, the short version is displayed: “Which image has the larger range from darks to highlights?”

We now describe how to evaluate each aspect in detail. Tonal contrast should be judged by looking at how different the bright, medium, and dark parts of the face appear.

• Bright vs. Dark Areas on the Face: Look for clearly bright areas and clearly dark areas on the face, as shown in Figure S6. Faces with higher contrast show obvious differences between light and dark regions.

– For the Eugene mannequin, compare the white hair to the dark hair, and check the eyes.

– For the Diana mannequin, look at bright skin areas (chin, cheeks) versus darker areas, and the eyes. Look at the shadow around the lips and the highlight on the

![](images/7f7f515967ffcd938edf7bc212f218d0d641795dc6b35e808dc30797853cbe1b.jpg)  
(a) 5 lux, LED 2700K, ∆EV = 9.0

![](images/d0da72d68753cffd463a2722606b9aea952380b872341bd492e22a589e862338.jpg)  
(b) 100 lux, LED 4000K, ∆EV = 7.0

![](images/3bb1a40851d255efa54fd5dd2a5cf470d92aa7ffc9bb18927d6b4bbc1a2bcc1a.jpg)  
(c) 100 lux,LED 4000K, ∆EV = 0.0

![](images/45fbd06c5c4fcc91f6526011de25f203d82f31af8e60574dc0f9d1fd9b3da977.jpg)  
(d) 1000 lux, SME D65, ∆EV = 0.0

Figure S4. Examples of the same scene captured under different illumination levels and exposure settings.  
![](images/fac78f47eedd589a5f999e3c7376600b6e88dbacf61e8eb0a3b2b13fd7cfb0e4.jpg)  
Figure S5. Tonal contrast (Overall contrast) example

lips too.

If the bright parts look truly bright and the dark parts look clearly dark, the contrast is higher. If everything looks similar in tone, the contrast is low.

• Does the Face Look Flat or Washed Out? Check whether the face looks clear and solid, or flat and foggy. A low-contrast face often looks grayish, dull, or veiled, as if a thin haze covers it. You should still be able to see normal facial details between the bright and dark areas. If the middle tones disappear and everything blends together, the contrast is low.

## C.2. Eye-Region Contrast Guideline

Main question: “Do the black and white parts of the eyes look truly black and truly white, or do they look gray and washed out?” High eye contrast means that the black pupil, the iris, and the white of the eye are easy to tell apart.

![](images/34ee4d000c74971411c2118a9bfc39c0a32ab88dd2f41d5cfb732211be7bc1d0.jpg)

![](images/fe014b6e1c800c260ba9659d8005b32d0b03f1b07c392b42ee1eb2125173c93a.jpg)  
Diana

![](images/1d403de00ef14925af746d36d4b467c7174f24a11bb394ee3f59e5feb6027bd9.jpg)  
Figure S6. Dark and bright regions to inspect for Eugene and Diana, with a washed-out example for overall contrast.

![](images/5f37765625cf7c44a849590731880852e7ae24cc746ea527be936cdeb640cf6a.jpg)  
Figure S7. Example of micro-contrast: the image on the right exhibits higher micro-contrast than the image on the left.

In the pairwise comparison tool, the short version is displayed: “Select the image with the darker pupil and brighter sclera (white of the eye).”

Check these two things (for both mannequins), as shown in Figure S8:

• White of the Eye: Look at the white part of the eye (sclera). Does it look white and bright, or does it look gray, yellow, or dull? Whiter whites mean higher eye contrast.

• Black of the Eye: Look at the black center of the eye (pupil). Does it look deep black, or does it look gray or milky? Darker blacks mean higher eye contrast.

## Mannequin-specific reminders:

• Eugene (blue eyes): Ignore how strong the blue color looks. Only judge whether the pupil looks clearly black compared to the rest of the eye.

• Diana (brown eyes): Check whether the black pupil is darker than the brown iris. If the pupil and iris blend together into the same dark tone, eye contrast is low.

![](images/029a7651201fba72665862717dd0f535698c89a4875e46ce543058b456c35e95.jpg)  
Figure S8. Regions to inspect for eye contrast: pupil darkness, catch-light brightness, and sclera brightness.

## C.3. Micro-Contrast Guideline

Main question: “Can you clearly see the tiny details on the skin surface?” When evaluating micro-contrast, we are looking for local regions where we can see details and texture. High micro-contrast means that the texture looks very crisp. Low micro-contrast causes the skin to appear overly smooth, plastic, or softened. Note that higher microcontrast does not mean that the image looks more pleasant.

In the pairwise comparison tool, the short version is displayed: “Choose the image where the tiny details on the skin surface are more visible.”

For Eugene, we focus on these questions by looking at the regions of interest shown in Figure S9:

• Beard Separation: Look closely at the beard. Can you see a clear, sharp separation between the white hairs and the dark hairs, or do they blend into a gray mix?

• Mustache Detail: Are the individual hairs of the mustache well preserved and crisp, or does the mustache look like a blurry, soft patch?

• Cheek / Forehead Texture: Can you see the tiny holes (pores) on the cheeks, or is the skin smoothed out?

For Diana, we focus on these questions:

• Forehead Detail: Look at the forehead area. Can you see actual skin texture and fine lines, or does the skin look too smooth (like plastic)?

• Eyebrow Definition: Since there is no beard, look at the eyebrows. Can you count individual eyebrow hairs, or do they look like a painted smudge?

• Over-smoothing Check: Does the skin look naturally textured, or has the camera softened it too much, making it look artificial?

## C.4. Balanced Contrast Preference Guideline

Main question: “Which image has more balanced contrast?” Balanced contrast refers to the image whose facial contrast appears most natural and visually well balanced overall. Annotators should consider whether the face has sufficient tonal separation without appearing too flat, too harsh, over-sharpened, or unnatural. Unlike the previous three tasks, this judgment is based on overall preference rather than the amount of a single contrast component.

![](images/178dd9c90f13967796eff7771a3d602870c259ae0d746b1e443955e5bf526ed9.jpg)

![](images/58308ea9dc3d3063bf9028d3fc0039115388f26fa3ccb96f3a9c546d61665b2d.jpg)  
Figure S9. Regions to inspect for micro-contrast in Eugene and Diana.

## D. Ablation Study

## D.1. Visualization of Overall Contrast

Figure S10 visualizes the multi-scale formulation used for overall contrast. For each example image, we show the local normalized contrast map computed at different window sizes, together with the resulting merged contrast map. Smaller windows respond more strongly to fine local variations, whereas larger windows emphasize broader facial shading and illumination structure. By combining these responses across scales, the proposed formulation captures overall facial contrast in a way that is more spatially informative than a single global luminance statistic.

## D.1.1. Baseline Formulations

Traditional measures typically define contrast as the luminance difference between discrete facial features (eyes, eyebrows, lips) and the surrounding skin.

1. Russell (2009) [58]: Utilizes a simple luminance range (difference) to define contrast $C _ { R u s s e l l } { : }$

$$
C _ { \mathrm { R u s s e l l } } = L _ { \mathrm { s k i n } } - L _ { \mathrm { f e a t u r e s } } ,\tag{3}
$$

where $L _ { \mathrm { s k i n } }$ and $L _ { \mathrm { f e a t u r e s } }$ are the mean luminance values of the skin and facial feature pixels, respectively. This metric measures the absolute luminance gap without normalization.

2. Pocheron et al. (2013) [´ 54]: Adopted the Michelson contrast formulation to ensure the metric is scaled relative to the overall brightness:

$$
C _ { \mathrm { P o c h \acute { e } r o n } } = \frac { L _ { \mathrm { s k i n } } - L _ { \mathrm { f e a t u r e s } } } { L _ { \mathrm { s k i n } } + L _ { \mathrm { f e a t u r e s } } } .\tag{4}
$$

This normalized version accounts for varying global illumination levels but still relies on region-wide averages.

![](images/783935773d14eb68815dfe909746745cc45603d431666ce89f9d03f7af0b9950.jpg)  
Figure S10. Visualization of the multi-scale overall-contrast formulation for three example images. For each image, we show the origina face mask, the local normalized contrast map computed at each scale $r \in \mathcal { R }$ , where R = {16, 32, 64, 128, 256}, and the resulting merged contrast map. Smaller scales emphasize fine local variations, while larger scales capture broader facial shading and illumination structure.

3. Russel et al. (2016) [59]: Expanded the Michelson approach by calculating contrast for multiple individual features (eyes, lips, eyebrows) and combining them into a single Facial Contrast Vector Magnitude $| \check { \vec { C } } _ { J } |$

$$
| \vec { C } _ { \mathrm { R u s s e l } } | = \sqrt { C _ { \mathrm { e y e s } } ^ { 2 } + C _ { \mathrm { l i p s } } ^ { 2 } + C _ { \mathrm { e y e b r o w s } } ^ { 2 } } ,\tag{5}
$$

where each $C _ { i }$ is calculated using the Michelson formula as in Pocheron et al. (2013).´

## D.2. Overall Contrast Ablation Study

Table S1 examines the sensitivity of the proposed overallcontrast measure to the choice of spatial scales. We consider both leave-one-scale-out variants of the reference set, $\mathcal { R } =$ {16, 32, 64, 128, 256}, and alternative scale sets obtained by shifting or densifying the sampled scales. Removing either of the two finest scales (16 or 32 pixels) consistently decreases both PLCC and SRCC under the Within-lighting and Mixed-lighting evaluations. Removing the intermediate 64- pixel scale results in only a small change, whereas removing the two coarsest scales yields higher nominal correlations. In particular, excluding the 256-pixel scale gives the highest nominal correlations in the leave-one-scale-out analysis, reaching PLCC/SRCC values of 0.815/0.806 within lighting and 0.729/0.729 under mixed lighting, compared with

0.792/0.780 and 0.688/0.693, respectively, for the reference configuration. The bootstrap distributions nevertheless exhibit substantial overlap, indicating that the observed differences should be interpreted as a tendency rather than a definitive advantage of a particular scale set.

A similar trend is observed when perturbing the complete scale set. Shifting all scales toward smaller spatial support yields moderately higher nominal correlations, while shifting them toward larger support reduces performance. Increasing the sampling density using half-octave spacing produces results close to the reference configuration, suggesting that the metric does not benefit substantially from a denser sampling of the same spatial range. Overall, these results indicate that the proposed measure is reasonably robust to moderate changes in the scale set, while also suggesting that the coarsest spatial support may be less informative for the face crops considered in our dataset. We retain the originally specified $R \ =$ 16, 32, 64, 128, 256 configuration in the main experiments to avoid retrospectively selecting the scale set based on the evaluation annotations, and report the present analysis as a sensitivity study rather than using it for post-hoc parameter tuning.

Table S1. Scale-set ablation of the proposed overall-contrast measure. Scale values denote square-patch side lengths in pixels. Results are reported as the mean ± standard deviation over 500 bootstrap resamples. The symbol † indicates the reference configuration used in the main experiments, while bold values indicate the highest nominal correlation within each analysis block.
<table><tr><td rowspan="2">Scale setting</td><td colspan="2">Within-lighting condition</td><td colspan="2">Mixed-lighting condition</td></tr><tr><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td></tr><tr><td colspan="5">Leave-one-scale-out analysis</td></tr><tr><td>{16, 32, 64, 128, 256}†</td><td> $0 . 7 9 2 \pm 0 . 0 7 0$ </td><td> $0 . 7 8 0 \pm 0 . 0 8 1$ </td><td> $0 . 6 8 8 \pm 0 . 1 3 3$ </td><td> $0 . 6 9 3 \pm 0 . 0 9 7$ </td></tr><tr><td>Without 16</td><td> $0 . 7 7 8 \pm 0 . 0 7 7$ </td><td> $0 . 7 5 5 \pm 0 . 0 9 2$ </td><td> $0 . 6 6 2 \pm 0 . 1 4 7$ </td><td> $0 . 6 7 1 \pm 0 . 1 0 7$ </td></tr><tr><td>Without 32</td><td> $0 . 7 8 1 \pm 0 . 0 7 6$ </td><td> $0 . 7 6 5 \pm 0 . 0 9 0$ </td><td> $0 . 6 6 9 \pm 0 . 1 4 5$ </td><td> $0 . 6 7 8 \pm 0 . 1 0 4$ </td></tr><tr><td>Without 64</td><td> $0 . 7 8 8 \pm 0 . 0 7 4$ </td><td> $0 . 7 7 3 \pm 0 . 0 8 8$ </td><td> $0 . 6 8 1 \pm 0 . 1 4 0$ </td><td> $0 . 6 8 7 \pm 0 . 1 0 2$ </td></tr><tr><td>Without 128</td><td> $0 . 7 9 8 \pm 0 . 0 6 7$ </td><td> $0 . 7 9 1 \pm 0 . 0 7 7$ </td><td> $0 . 7 0 1 \pm 0 . 1 2 7$ </td><td> $0 . 7 0 5 \pm 0 . 0 9 1$ </td></tr><tr><td>Without 256</td><td> $\mathbf { 0 . 8 1 5 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 0 . 8 0 6 \pm 0 . 0 6 5 }$ </td><td> $\mathbf { 0 . 7 2 9 \pm 0 . 1 0 7 }$ </td><td> $\mathbf { 0 . 7 2 9 \pm 0 . 0 7 4 }$ </td></tr><tr><td colspan="5">Scale-set perturbation</td></tr><tr><td>Smaller: {12, 24, 48, 96, 192}</td><td> $\mathbf { 0 . 8 0 8 \pm 0 . 0 5 9 }$ </td><td> $\mathbf { 0 . 7 9 6 \pm 0 . 0 7 0 }$ </td><td> $\mathbf { 0 . 7 1 3 \pm 0 . 1 1 9 }$ </td><td> $\mathbf { 0 . 7 1 4 \pm 0 . 0 8 5 }$ </td></tr><tr><td>Reference:  $\{ 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \} ^ { \dagger }$ </td><td> $0 . 7 9 2 \pm 0 . 0 7 0$ </td><td> $0 . 7 8 0 \pm 0 . 0 8 1$ </td><td> $0 . 6 8 8 \pm 0 . 1 3 3$ </td><td> $0 . 6 9 3 \pm 0 . 0 9 7$ </td></tr><tr><td>Larger: {20, 40, 80, 160, 320}</td><td> $0 . 7 8 7 \pm 0 . 0 7 5$ </td><td> $0 . 7 7 0 \pm 0 . 0 8 5$ </td><td> $0 . 6 6 5 \pm 0 . 1 4 7$ </td><td> $0 . 6 7 7 \pm 0 . 1 0 4$ </td></tr><tr><td>Half-octave dense: {16, 23, 32, 45, 64, 91, 128, 181, 256}</td><td> $0 . 7 9 7 \pm 0 . 0 6 8$ </td><td> $0 . 7 8 8 \pm 0 . 0 7 4$ </td><td> $0 . 6 9 2 \pm 0 . 1 3 0$ </td><td> $0 . 6 9 6 \pm 0 . 0 9 4$ </td></tr></table>

## D.3. Eye Contrast Ablation Study

As a simple baseline, eye-region contrast can be defined using a Michelson-style normalized range computed from percentile extrema within the eye mask:

$$
L _ { \mathrm { m i n } } = P _ { 1 } ( L _ { \mathrm { e y e } } ) , \qquad L _ { \mathrm { m a x } } = P _ { 9 9 } ( L _ { \mathrm { e y e } } ) ,\tag{6}
$$

and

$$
C _ { \mathrm { e y e } } = \frac { L _ { \mathrm { m a x } } - L _ { \mathrm { m i n } } } { L _ { \mathrm { m a x } } + L _ { \mathrm { m i n } } } .\tag{7}
$$

This formulation measures the separation between the darkest and brightest structures in the eye region and can provide a reasonable approximation when the pupil/iris and sclera are clearly separated.

However, this normalized range becomes less reliable in low-contrast cases. When both the bright and dark structures are compressed into a narrow luminance range, the resulting score may no longer reflect the perceived separation between the pupil, iris, and sclera. As a result, the Michelson-style ordering can become perceptually unstable. Figure S11 illustrates such a failure case: although Image 212030 appears to have lower eye contrast than Image 212022, the Michelson [37] measure does not preserve this perceptual ordering.

For this reason, we adopt the Otsu-based formulation in the main method. By partitioning the luminance values within the eye mask into dominant dark and bright classes, the proposed measure better reflects the perceptually meaningful separation between the darker pupil/iris region and the brighter sclera and highlights. Figure S12 further shows that even in a challenging low-contrast regime, the Otsubased [47] formulation still yields a perceptually plausible ranking.

## D.4. Micro-Contrast Ablation Study

We first evaluate the contribution of the three main stages in the proposed micro-contrast formulation: local energy pooling, perceptual response compression, and robust spatial pooling. Tables S2 and S3 report the component ablation results under the within-lighting and mixed-lighting evaluations, respectively. In these experiments, we use the reference configuration $\sigma _ { f } = 1 / \sqrt { 2 } \approx 0 . 7 , \sigma _ { c } / \sigma _ { f } = 3 ,$ $W \ = \ 1 1 , \ \gamma \ = \ 0 . 6$ , and [q<sub>10</sub>, q<sub>90</sub>] pooling. Each component is removed individually while all other applicable parameters are held fixed. Specifically, removing robust spatial pooling replaces the [q<sub>10</sub>, q<sub>90</sub>] trimmed mean with an untrimmed spatial mean. Removing perceptual response compression sets $\gamma = 1 .$ , while retaining the reference scale parameters, local RMS integration with an $1 1 \times 1 1$ window, and robust pooling. Removing local energy pooling replaces the spatially localized RMS response with a single global DoG statistic computed using the same fine and coarse scales.

Under Within-lighting condition, the complete formulation achieves a PLCC/SRCC of $0 . 8 4 8 \pm 0 . 0 2 4 / 0 . 8 4 1 \pm$ 0.028. Removing robust spatial pooling reduces the correlations to $0 . 8 2 5 \pm 0 . 0 2 6 / 0 . 8 1 4 \pm 0 . 0 2 9$ , demonstrating that percentile trimming improves robustness to isolated responses caused by specular highlights, shadow noise, and mask-boundary artifacts. Removing perceptual response compression reduces the result to $0 . 8 0 2 \pm 0 . 0 2 9 / 0 . 8 1 5 \pm$ 0.029. The larger reduction in PLCC than in SRCC suggests that the nonlinear transducer contributes particularly to the linear calibration between the objective response and subjective JOD, while having a smaller effect on rank ordering. The largest degradation occurs without local energy pooling, for which performance decreases to $0 . 6 6 4 \pm 0 . 0 6 9 / 0 . 6 6 8 \pm 0 . 0 7 7 .$ The same trend is observed under Mixed-lighting condition. The complete formulation achieves $0 . 7 9 7 \pm 0 . 0 1 1 / 0 . 7 8 7 \pm 0 . 0 1 9$ , compared with $0 . 7 7 5 \pm 0 . 0 1 1 / 0 . 7 6 3 \pm 0 . 0 0 0$ without robust spatial pooling and $0 . 7 3 6 \pm 0 . 0 1 9 / 0 . 7 6 3 \pm 0 . 0 0 1$ without perceptual response compression. Replacing local energy pooling with a global DoG statistic further reduces performance to $0 . 6 3 1 \pm 0 . 1 0 2 / 0 . 6 2 9 \pm 0 . 0 7 7$ . These results consistently indicate that micro-contrast is better represented as a spatially localized texture response than as a single global highfrequency statistic.

![](images/5818e539a2130a1c66715964382f256ecff0b96c97a96846be2b7a06374ac6ee.jpg)  
Figure S11. Face images and corresponding eye-region crops ordered by the Michelson eye-contrast measure [37]. While this formu lation provides a reasonable ranking in some cases, it can fail in low-contrast regimes. In particular, the perceptual ordering between Images 212022 and 212030 is not preserved. For visualization, sRGB images are shown, but the metric is computed on the ${ \bar { L ^ { * } } }$ channel only.

![](images/d76580a558ffdcb47cbe7f2d7b91d5cef77f74baa8c482d45a624c90778ee0a7.jpg)  
Figure S12. Low-contrast eye-region examples ordered by the proposed Otsu-based eye-contrast measure [47]. Although these images exhibit weaker dark–bright separation than high-contrast cases, the proposed formulation still yields a perceptually meaningful ranking.

In addition to the component ablation, we evaluate the sensitivity of the formulation to its numerical parameters. Figure S13 reports a one-factor-at-a-time sensitivity analysis over the fine scale $\sigma _ { f } ,$ coarse-to-fine scale ratio $\rho =$ $\sigma _ { c } / \sigma _ { f }$ , local integration window W, perceptual exponent $\gamma ,$ and robust pooling interval. These parameters were fixed before the evaluation according to the adopted scalespace formulation, perceptual model, and implementation design, rather than selected by maximizing performance on the evaluation sets.

The settings $\sigma _ { f } = 1 / \sqrt { 2 } \approx 0 . 7$ and $\sigma _ { f } = 1 . 0$ produce comparable correlations, whereas performance decreases at $\sigma _ { f } = \sqrt { 2 }$ and coarser scales. This trend indicates that relatively fine spatial scales are more suitable for representing the facial details associated with subjective micro-contrast. Performance is also stable when the coarse-to-fine scale ratio is varied from $\rho = 3 { \mathrm { ~ t o ~ } } \rho = 5 ; $ relative to $\rho = 3 ,$ , the largest absolute change across the four reported correlations is only 0.006. We therefore retain the reference ratio $\rho = 3 .$

Local window size. Changing the local integration window from $5 \times 5 ~ 1 0 ~ 1 9 \times 1 9$ produces limited variation. Under the Within-lighting evaluation, PLCC ranges from 0.843 to 0.853 and SRCC from 0.835 to 0.849. Under the mixed-lighting evaluation, the corresponding ranges are 0.794–0.799 and 0.783–0.791. By contrast, replacing local RMS integration with a single global DoG statistic reduces PLCC and SRCC under both settings, as reported in Tables S2 and S3. These results demonstrate the importance of spatially localized energy integration while showing that performance is only weakly affected by its exact window size. We retain the intermediate $1 1 \times 1 1$ window.

Perceptual response compression. Response compression improves correlation relative to the uncompressed setting $( \gamma = 1 )$ . The highest nominal correlations are obtained with $\gamma = 0 . 4$ , while the reference value $\gamma = 0 . 6$ gives slightly lower results. The differences between these settings are 0.018 in PLCC and 0.013 in SRCC under the Within-lighting evaluation, and 0.020 and 0.012, respectively, under the Mixed-lighting evaluation. We retain $\gamma =$ 0.6 because it was specified from the adopted perceptual response model before the sensitivity analysis rather than selected retrospectively.

Percentile trimming. Percentile trimming improves performance relative to untrimmed spatial averaging. Replacing the untrimmed mean with $[ q _ { 1 0 } , q _ { 9 0 } ]$ pooling increases PLCC/SRCC from 0.825/0.814 to 0.848/0.841 under the Within-lighting evaluation and from 0.775/0.763 to 0.797/0.787 under the Mixed-lighting evaluation. Narrower intervals provide further nominal improvements; however, the reference interval $[ q _ { 1 0 } , q _ { 9 0 } ]$ retains the central 80% of spatial responses and provides a less aggressive form of outlier rejection.

Table S2. Ablation study of the proposed micro-contrast formulation under Within-lighting condition. Results are reported as the mean ± standard deviation over 500 bootstrap resamples.
<table><tr><td>Variant</td><td>PLCC</td><td>SRCC</td></tr><tr><td>Proposed micro-contrast</td><td> $0 . 8 4 9 \pm 0 . 0 2 4$ </td><td> $0 . 8 4 2 \pm 0 . 0 2 8$ </td></tr><tr><td>w/o robust spatial pooling</td><td> $0 . 8 2 5 \pm 0 . 0 2 6$ </td><td> $0 . 8 1 4 \pm 0 . 0 2 9$ </td></tr><tr><td>w/o perceptual response compression</td><td> $0 . 8 0 2 \pm 0 . 0 2 9$ </td><td> $0 . 8 1 5 \pm 0 . 0 2 9$ </td></tr><tr><td>w/o local energy pooling</td><td> $0 . 6 6 4 \pm 0 . 0 6 9$ </td><td> $0 . 6 6 8 \pm 0 . 0 7 7$ </td></tr></table>

Table S3. Ablation study of the proposed micro-contrast formulation under Mixed-lighting condition. Results are reported as the mean ± standard deviation over 500 bootstrap resamples.
<table><tr><td>Variant</td><td>PLCC</td><td>SRCC</td></tr><tr><td>Proposed micro-contrast</td><td> $0 . 7 9 7 \pm 0 . 0 1 1$ </td><td> $0 . 7 8 7 \pm 0 . 0 1 9$ </td></tr><tr><td>w/o robust spatial pooling</td><td> $0 . 7 7 5 \pm 0 . 0 1 1$ </td><td> $0 . 7 6 3 \pm 0 . 0 0 0$ </td></tr><tr><td>w/o perceptual response compression</td><td> $0 . 7 3 6 \pm 0 . 0 1 9$ </td><td> $0 . 7 6 3 \pm 0 . 0 0 1$ </td></tr><tr><td>w/o local energy pooling</td><td> $0 . 6 3 1 \pm 0 . 1 0 2$ </td><td> $0 . 6 2 9 \pm 0 . 0 7 7$ </td></tr></table>

Overall, the component ablations show that local energy integration, perceptual response compression, and robust spatial pooling each contribute to agreement with subjective judgments. The parameter sweeps further demonstrate limited sensitivity to the local window size and scale ratio, together with gradual and interpretable trends for the fine scale, perceptual exponent, and pooling interval. Although some neighboring configurations achieve higher nominal correlations, the reference configuration provides consistently strong performance under both evaluation settings without post-hoc parameter selection.

## D.5. Per-Mannequin Analysis

Table S4 reports the within-lighting results separately for Diana and Eugene. All three proposed metrics remain positively correlated with their corresponding perceptual judgments for both mannequins. Eye and micro-contrast show similar performance across the two identities, while overall contrast exhibits greater variation.

## D.6. Balanced Contrast Preference Ablation Study

We compare three balanced contrast preference-model variants using the proposed contrast measures as predictors: a non-negative linear regressor, a second-order polynomial regressor, and a pairwise-trained MLP. The second-order polynomial augments the three contrast components with their squared and cross-component interaction terms. The pairwise-trained MLP uses the same augmented representation and independently maps each image to a scalar balanced contrast score. It is trained using pairwise ranking supervision together with a balanced contrast JOD regression regularizer, following the training protocol described in the next subsection. Table S5 compares the three variants under the within-lighting and mixed-lighting evaluations.

Table S4. Per-mannequin performance of the proposed contrast metrics under the within-lighting evaluation. Correlations are macro-averaged over Groups 1–4 for each mannequin. Uncertainty denotes the standard deviation over 500 bootstrap resamples of the four group-level correlations.
<table><tr><td rowspan="2">Contrast</td><td colspan="2">Diana</td><td colspan="2">Eugene</td></tr><tr><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td></tr><tr><td>Overall contrast</td><td> $0 . 8 8 0 \pm 0 . 0 1 7$ </td><td> $0 . 8 7 5 \pm 0 . 0 2 6$ </td><td> $0 . 7 0 5 \pm 0 . 1 2 9$ </td><td> $0 . 6 9 1 \pm 0 . 1 4 5$ </td></tr><tr><td>Eye contrast</td><td> $0 . 9 3 4 \pm 0 . 0 1 5$ </td><td> $0 . 9 1 9 \pm 0 . 0 0 7$ </td><td> $0 . 8 2 4 \pm 0 . 0 6 4$ </td><td> $0 . 7 8 4 \pm 0 . 1 0 1$ </td></tr><tr><td>Micro-contrast</td><td> $0 . 8 6 3 \pm 0 . 0 4 0$ </td><td> $0 . 8 6 2 \pm 0 . 0 4 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 2 6$ </td><td> $0 . 8 2 2 \pm 0 . 0 3 3$ </td></tr></table>

The two evaluations differ in how the balanced contrast preference predictors are fitted and evaluated. In the within-lighting evaluation, a separate model is fitted for each mannequin and each of Groups 1–4 using the corresponding device-disjoint training split, and evaluated on the held-out devices from the same group. Because the balanced contrast JOD scale is estimated independently for each group and mannequin, each model is fitted and evaluated within its corresponding JOD scale. In the mixedlighting evaluation, a single model per mannequin is fitted using Groups 1–4 and evaluated on the held-out Group 5 without refitting or using its balanced contrast labels. For each mannequin, the training balanced contrast JOD values therefore originate from four independently estimated group-specific scales, whereas Group 5 provides a single mixed-lighting JOD scale. Within-lighting results are computed separately for the eight Group–mannequin evaluation subsets and then macro-averaged. Mixed-lighting results are computed separately for Diana and Eugene on Group 5 and then macro-averaged over the two mannequins.

As shown in Table S5, the second-order polynomial regressor achieves the highest overall performance, reaching PLCC/SRCC values of 0.728/0.739 within lighting and $0 . 7 7 3 / 0 . 7 7 8$ under mixed lighting. The pairwise-trained MLP is competitive but remains slightly below the polynomial model overall, while the non-negative linear regressor performs worse in both settings. We therefore use the second-order polynomial regressor in the main balanced contrast preference analysis, as it remains compact and explicitly parameterized while capturing quadratic effects and cross-component interactions.

Sensitivity to the training JOD scales. Given the independently estimated Group–mannequin balanced contrast

![](images/9d2ef8095f9abcc577f3a025bc5427d5e0871f7efc0c00dc642b620a0b67be8d.jpg)  
Figure S13. Parameter sensitivity of the proposed micro-contrast measure under the Within-lighting condition (top) and mixed-lighting (bottom) evaluations. We vary the fine scale $\sigma _ { f } ,$ scale ratio $\rho = \sigma _ { c } / \sigma _ { f }$ , local integration window W, perceptual exponent γ, and robust pooling interval. Solid and dashed lines denote PLCC and SRCC, respectively, and error bars represent the standard deviation over 500 bootstrap resamples. The gray bands and † symbols indicate the reference configuration used in the main experiments: $\sigma _ { f } = 1 / \sqrt { 2 } \approx 0 . 7$ $\rho = 3 , W = 1 1 , \gamma = 0 . 6 ,$ and [q<sub>10</sub>, q<sub>90</sub>] pooling.

Table S5. Comparison of balanced contrast preference predictors under the within-lighting and mixed-lighting evaluations.  
Table S6. Sensitivity of the second-order polynomial balanced contrast preference predictor to normalization of the Groups 1– 4 training targets.
<table><tr><td>Method</td><td>PLCC</td><td>SRCC</td><td>RMSE</td></tr><tr><td>Within-lighting evaluation</td><td></td><td></td><td></td></tr><tr><td>Non-negative linear regressor</td><td> $0 . 6 3 5 \pm 0 . 0 8 5$ </td><td> $0 . 6 5 4 \pm 0 . 0 7 3$ </td><td> $1 . 1 1 1 \pm 0 . 0 9 0$ </td></tr><tr><td>Second-order polynomial regressor</td><td> $\mathbf { 0 . 7 2 8 \pm 0 . 0 5 5 }$ </td><td> $\mathbf { 0 . 7 3 9 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 1 . 0 6 3 \pm 0 . 1 1 5 }$ </td></tr><tr><td>Pairwise-trained MLP</td><td> $0 . 7 0 8 \pm 0 . 0 5 5$ </td><td> $0 . 7 3 4 \pm 0 . 0 4 4$ </td><td> $1 . 1 1 8 \pm 0 . 1 1 4$ </td></tr><tr><td>Mixed-lighting evaluation</td><td></td><td></td><td></td></tr><tr><td>Non-negative linear regressor</td><td> $0 . 7 1 5 \pm 0 . 0 4 4$ </td><td> $0 . 6 8 2 \pm 0 . 0 1 6$ </td><td> $1 . 0 9 4 \pm 0 . 0 2 9$ </td></tr><tr><td>Second-order polynomial regressor</td><td> $\mathbf { 0 . 7 7 3 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 0 . 7 7 8 \pm 0 . 0 4 6 }$ </td><td> $\mathbf { 0 . 9 7 5 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Pairwise-trained MLP</td><td> $0 . 7 5 9 \pm 0 . 0 6 2$ </td><td> $0 . 7 5 6 \pm 0 . 0 4 9$ </td><td> $0 . 9 8 3 \pm 0 . 0 2 4$ </td></tr></table>

<table><tr><td>Training target</td><td>PLCC</td><td>SRCC</td></tr><tr><td>Raw balanced contrast JOD</td><td> $0 . 7 7 3 \pm 0 . 0 5 4$ </td><td> $0 . 7 7 8 \pm 0 . 0 4 6$ </td></tr><tr><td>Group-mannequin z-normalized JOD</td><td> $0 . 7 7 4 \pm 0 . 0 5 0$ </td><td> $0 . 7 7 5 \pm 0 . 0 3 8$ </td></tr></table>

in the main experiments.

JOD scales described above, we test whether differences in their locations and dispersions affect mixed-lighting performance. We refit the second-order polynomial model after z-normalizing the balanced contrast JOD training targets separately within each Group–mannequin comparison set. The normalization is applied only to the Groups 1–4 training targets; Group 5 remains unchanged. As shown in Table S6, the mixed-lighting correlations are essentially unchanged: PLCC and SRCC change from 0.773/0.778 with the original balanced contrast JOD targets to 0.774/0.775 with Group–mannequin z-normalized targets. This indicates that the mixed-lighting correlations are not materially sensitive to differences in the locations and dispersions of the independently estimated source JOD scales. We therefore retain the original balanced contrast JOD targets used

## D.6.1. Fitted Coefficients

Table S7 reports the coefficients of the non-negative linear and second-order polynomial models in standardized feature space. The “Fitting data” column identifies the data used to estimate each set of coefficients. The linear coefficients are constrained to be non-negative, whereas the second-order polynomial coefficients are estimated without sign constraints. The models are fitted under independently estimated balanced contrast JOD scales. In addition, feature standardization is performed separately for each training set. We report the coefficients to make the fitted models and their parameter estimation explicit rather than to infer relative perceptual importance across lighting groups or mannequins.

Table S7. Coefficients of the linear and second-order polynomial preference regressors in standardized feature space. For within-lighting evaluation, models are fitted separately within Groups 1–4. For mixed-lighting evaluation, models are fitted on Groups 1–4 and applied without refitting to the held-out Group 5 set.
<table><tr><td>Evaluation</td><td>Fitting data</td><td>Subject</td><td>Model</td><td> $\alpha _ { O }$ </td><td> $\alpha _ { E }$ </td><td>αM</td><td> $\alpha _ { O ^ { 2 } }$ </td><td> $\alpha _ { E ^ { 2 } }$ </td><td> $\alpha _ { M ^ { 2 } }$ </td><td> $\alpha _ { O E }$ </td><td>αOM</td><td>αEM</td></tr><tr><td rowspan="10">Within-lighting</td><td>G1</td><td>Diana</td><td>Linear Polynomial</td><td>0.3737 0.6424</td><td>0.3475 0.3075</td><td>0.1114 0.2604</td><td>一 -0.4039</td><td>一 0.0006</td><td>一 -0.3136</td><td>一 -0.0219</td><td>0.1000</td><td>0.2704</td></tr><tr><td></td><td>Eugene</td><td>Linear Polynomial</td><td>0.1839 0.4860</td><td>0.0000 -0.2704</td><td>0.2491 0.4312</td><td>-0.4301</td><td>-0.0327</td><td>-0.4807</td><td>-0.4350</td><td>0.7321</td><td>0.5574</td></tr><tr><td></td><td></td><td>Linear</td><td>0.0000</td><td>1.3586</td><td>0.3842</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>G2</td><td>Diana</td><td>Polynomial Linear</td><td>-0.3201</td><td>1.3867</td><td>0.8643</td><td>-2.0840</td><td>-4.3643</td><td>-1.9313</td><td>5.7795</td><td>1.2028</td><td>1.0930</td></tr><tr><td></td><td>Eugene</td><td>Polynomial</td><td>0.8423 1.0003</td><td>0.3132 0.0538</td><td>0.2065 0.2690</td><td>-0.4833</td><td>-0.3169</td><td>-0.1009</td><td>0.4215</td><td>-0.0670</td><td>0.0423</td></tr><tr><td>G3</td><td>Diana</td><td>Linear</td><td>0.6623</td><td>0.4001</td><td>0.0000</td><td></td><td>一</td><td>一</td><td>1</td><td>一</td><td></td></tr><tr><td></td><td>Eugene</td><td>Polynomial Linear</td><td>0.8821 0.2555</td><td>0.5117 0.0448</td><td>-0.0024 0.4794</td><td>-1.0349</td><td>-0.8659</td><td>-0.2808</td><td>1.4400</td><td>0.2754</td><td>-0.0865</td></tr><tr><td></td><td></td><td>Polynomial</td><td>-0.0577</td><td>0.0732</td><td>1.0654</td><td>-0.8306</td><td>-0.2800</td><td>-0.8362</td><td>-0.7581</td><td>1.4209</td><td>0.3522</td></tr><tr><td></td><td></td><td></td><td>0.7120</td><td>0.7386</td><td>0.0406</td><td></td><td>1</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Diana</td><td>Linear Polynomial</td><td>0.8128</td><td>0.6468</td><td>0.1072</td><td>-0.1729</td><td>-0.0588</td><td>0.0460</td><td>1 0.0686</td><td>一 -0.0714</td><td>0.3098</td></tr><tr><td>G4</td><td></td><td>Linear</td><td>0.0000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Eugene</td><td></td><td></td><td>0.7800</td><td>0.3131</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Polynomial</td><td>0.0294</td><td>1.0096</td><td>0.3282</td><td>-0.2183</td><td>-0.2613</td><td>0.1306</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-0.6179</td><td>0.0959</td><td>0.4796</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Diana</td><td>Linear</td><td>0.1750</td><td>0.6530</td><td>0.3610</td><td></td><td></td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>Mixed-lighting</td><td></td><td>Polynomial</td><td>0.6543</td><td>0.0039</td><td>0.5306</td><td>-0.5738</td><td>-0.4526</td><td>-0.3662</td><td>0.7366</td><td>0.2660</td><td>0.1194</td></tr><tr><td>G1-4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Eugene</td><td>Linear</td><td>0.1768</td><td>0.3320</td><td>0.3802</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Polynomial</td><td>0.4816</td><td>0.0155</td><td>0.3340</td><td>-0.3749</td><td>-0.3212</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-0.1104</td><td>0.4249</td><td>0.1800</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-0.2911</td><td></td><td></td><td></td></tr></table>

## D.6.2. Training Details for Learned Balanced contrast Models

To adapt the learned baselines to the balanced contrast preference task, we fine-tune lightweight trainable heads while keeping the pretrained representation largely frozen. For general IQA models, we optimize only the prediction head. For face-quality models, we keep the pretrained feature extractor or baseline score frozen and train only a small probe head on top. This design keeps the adaptation lightweight while allowing each model to align with the target preference supervision. Training is performed using withinsubset pair sampling. Specifically, image pairs are formed only within the same (Group, Person) subset, so that the model learns preference ordering under comparable lighting and subject conditions. Each sampled pair is ordered canonically as (hi, lo), where hi has a higher Preference JOD than lo. The main objective is a pairwise ranking loss,

$$
\mathcal { L } _ { \mathrm { r a n k } } = \mathrm { s o f t p l u s } ( - ( s _ { \mathrm { h i } } - s _ { \mathrm { l o } } ) ) ,\tag{8}
$$

where $s _ { \mathrm { h i } }$ and $s _ { \mathrm { l o } }$ are the predicted preference scores for the higher- and lower-JOD images, respectively. This loss encourages the preferred image to receive a larger predicted score. To stabilize training across subsets, we additionally use a group-wise z-regularization term:

$$
\mathcal { L } _ { z } = \frac { 1 } { 2 } \left[ \left( s _ { \mathrm { h i } } - z _ { \mathrm { h i } } \right) ^ { 2 } + \left( s _ { \mathrm { l o } } - z _ { \mathrm { l o } } \right) ^ { 2 } \right] ,\tag{9}
$$

where $z _ { \mathrm { h i } }$ and $z _ { \mathrm { l o } }$ denote normalized target values computed within each (Group, Person) subset. The final optimization objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { z } \mathcal { L } _ { z } , } \end{array}\tag{10}
$$

This adaptation protocol is intended to match the pairwise supervision format of the preference task rather than to redefine the proposed contrast components. We set $\lambda _ { z } =$ 0.2 in all adapted-baseline experiments. For feature-probe models, each frozen feature dimension is standardized using statistics estimated from the training split only. When affine calibration is used for evaluation, its scale and offset are also estimated exclusively from the training split and then applied unchanged to the validation or test split.

## E. Real Portrait Generalization Study

Figure S14 shows representative examples from the realportrait dataset, which contains 444 images across 10 scenes and 29 unique capture devices. Each scene corresponds to a different portrait subject and includes images captured with multiple devices. Table S8 reports the device coverage for each scene. To support sufficient pairwise comparisons for subjective evaluation, each scene was expanded to approximately 44 images by including contrastadjusted variants of the original captures. The resulting dataset spans diverse subjects, devices, and contrast renderings under less controlled real-world conditions. This experiment complements the controlled mannequin study by evaluating whether the proposed component-wise contrast metrics generalize to real portrait images.

Table S9 reports scene-wise PLCC/SRCC for overall contrast, eye contrast, and micro-contrast. For each attribute, we compare the proposed zero-shot metric with IFQA fine-tuned on the corresponding mannequin-based annotations. Correlations are computed separately for each scene and macro-averaged across the 10 scenes. The proposed metrics achieve average PLCC/SRCC values of 0.792/0.825 for overall contrast, 0.753/0.768 for eye contrast, and 0.717/0.731 for micro-contrast. For overall contrast, fine-tuned IFQA performs better on average, whereas the proposed metric remains competitive without training on either mannequin or real-portrait annotations. For eye contrast, the proposed metric achieves higher average PLCC and SRCC than fine-tuned IFQA. Performance varies across scenes for both approaches, reflecting the greater diversity of subjects and capture conditions in the real-portrait dataset.

![](images/b2323e5449f26f4f860432c6981b0286b0ab45887b6ef76d6aa31aa24081e1bb.jpg)  
Figure S14. Real-portrait ablation study on overall contrast. A total of 444 images from 29 unique capture devices/sources are used. Each scene corresponds to a different subject and contains approximately 44 images captured with an average of five devices, together with contrast-adjusted variants generated using DxO PhotoLab 9. Human judgments were collected through pairwise comparisons for overal contrast, eye contrast, and micro-contrast. The proposed method shows strong agreement with the subjective annotations across diverse real-portrait scenes. From top to bottom, the rows correspond to: scene 8005 103 and scene 1606 031; scene 0105 089 and scene 0109 2; scene 1119 105 and scene 1113 510; scene 3115 305 and Riverside; and scene 0109 3 and scene 5203 106.

Table S10 further examines the effect of mannequinbased fine-tuning on IFQA. Without task-specific adaptation, IFQA yields negative average correlations with all three contrast attributes, indicating that its off-the-shelf quality scores are not directly aligned with the attributespecific contrast judgments considered in this work. Finetuning on the corresponding mannequin-based annotations substantially improves performance across all three attributes. This distinction is important when interpreting the comparison with the proposed metrics: our metrics are applied directly to real portraits without fitting to either mannequin or real-portrait annotations, whereas IFQA benefits from task-specific supervision on the mannequin dataset. In particular, the improvement after fine-tuning, including the strong performance of IFQA for overall contrast, provides additional evidence that the controlled mannequin annotations capture contrast-related information that transfers to real portraits. Overall, these results support the proposed mannequin dataset as a controlled bridge for studying photographic face contrast, while also showing that the proposed zero-shot metrics generalize meaningfully to real portrait images.

Table S8. Device distribution across real-portrait scenes (29 unique capture devices in total).
<table><tr><td>Scene</td><td>Device Count</td><td>Devices</td></tr><tr><td>0105_089</td><td>8</td><td>Apple iPhone 15 Pro Max, Sony Alpha 9 II, Honor Magic6 Pro, Huawei Pura 70 Ultra, Oppo Find X7 Ultra, Samsung Galaxy S24 Ultra, Vivo X100 Pro, Xiaomi 14 Ultra</td></tr><tr><td>0109_2</td><td>7</td><td>Apple iPhone 14 Pro, Google Pixel 7 Pro, Honor Magic5 Pro, Huawei P60 Pro, Oppo Find X6 Pro, Samsung Galaxy S23 Ultra,</td></tr><tr><td>0109_3</td><td>7</td><td>Sony A7R IV Apple iPhone 14 Pro, Google Pixel 7 Pro, Honor Magic5 Pro, Huawei P60 Pro, Oppo Find X6 Pro, Samsung Galaxy S23 Ultra,</td></tr><tr><td>1113.510</td><td>5</td><td>Sony A7R IV Apple iPhone 15 Pro Max, Sony Alpha 9 II, Google Pixel 8 Pro, OnePlus 12, Samsung Galaxy S24 Ultra</td></tr><tr><td>1119_105</td><td>5</td><td>Apple iPhone 15 Pro Max, Sony Alpha 9 II, Google Pixel 8 Pro, OnePlus 12, Samsung Galaxy S24 Ultra</td></tr><tr><td>1606_031</td><td>6</td><td>Apple iPhone 16 Pro Max, Honor Magic7 Pro, Huawei Pura 70 Ultra, Oppo Find X8 Pro, Vivo X200 Pro, Xiaomi 15 Pro</td></tr><tr><td>3115_305</td><td>5</td><td>Apple iPhone 15 Pro Max, Sony Alpha 9 II, Google Pixel 8 Pro, OnePlus 12, Samsung Galaxy S24 Ultra</td></tr><tr><td>5203_106</td><td>3</td><td>Apple iPhone 16 Pro, Google Pixel 9 Pro XL, Honor Magic7 Pro</td></tr><tr><td>8005_103</td><td>5</td><td>Apple iPhone 16 Pro, Google Pixel 9 Pro XL, Honor Magic7 Pro, Samsung Galaxy S25 Ultra, Xiaomi 15 Ultra</td></tr><tr><td>Riverside</td><td>3</td><td>Apple iPhone 17 Pro, Apple iPhone Air, Samsung Galaxy S25 Edge</td></tr></table>

## F. Controlled Mannequin Dataset

The original full-resolution dataset contains 508 images in total across the two mannequins, with 250 images for Diana and 258 images for Eugene. To construct the comparison groups used for perceptual annotation, we supplemented the original captures with edited images so that each group contained a sufficient number of samples for pairwise comparison. The resulting composition of original and edited images for each group is summarized in Table S11.

## F.1. Face Cropping and Region Mask Extraction

We first detect the face region in each image to ensure that contrast analysis focuses solely on the face rather than the background. For the face detection, we used the MTCNN implementation [77] provided by the facenet-pytorch library [15]. Each crop is resized to a fixed resolution of 585 × 820 pixels. This step reduces variation in face scale and alignment and standardizes the image presentation for pairwise comparison experiments. We then extract semantic facial regions using a BiSeNet face parsing model [75, 76], from which we derive three masks used throughout this work: a face mask for overall contrast, an eye mask for eye contrast, and a skin mask for micro-contrast as shown in Figure S3.

## F.2. Mannequin Dataset Composition Evaluation.

The complete dataset contains images captured with 29 smartphones from multiple manufacturers, covering a broad range of device categories and ISP characteristics. Table S12 summarizes the device brands and models included in the dataset. The dataset contains two mannequin subjects, Eugene and Diana, and the processed comparison subsets are organized into five groups corresponding to different lighting regimes, as described in the main paper.

We evaluate the data under two complementary conditions. Within-lighting condition corresponds to a withingroup device-disjoint split on Groups 1–4, while Mixedlighting condition is a cross-group evaluation in which Groups 1–4 are used as the source set and Group 5 is used as the target set. The purpose of Within-lighting condition is to measure generalization across unseen devices within the same lighting regime, whereas Mixed-lighting condition evaluates transfer to a broader mixed group with more diverse contrast variation.

## F.2.1. Within-lighting condition.

This protocol is applied only to Groups 1–4. For each mannequin and each group, the split is performed by device rather than by image, so that the same smartphone never appears in both training and validation. This provides a strict device-disjoint evaluation setting and avoids optimistic correlations caused by device-specific ISP signatures appearing in both subsets. Among the remaining samples, the validation devices are selected to satisfy three criteria: (i) strict device separation, (ii) a validation size close to the target ratio, and (iii) a reasonable balance between the training and validation subsets in JOD space. The balancing criterion uses the mean and standard deviation of the four annotation dimensions (Preference, Overall, Eye, and Micro) as a secondary constraint when constructing the split. This procedure yields a device-disjoint partition while keeping the train and validation subsets reasonably matched in perceptual-score distribution.

Table S14 reports the number of effective training and validation devices for each group and mannequin under Within-lighting condition. As expected, the exact number of available devices varies slightly across groups because some captures are removed during duplicate filtering, but the split remains strictly device-disjoint in all cases. Table S15 further summarizes the effective number of samples per mannequin after duplicate removal. The resulting split remains close to a two-thirds / one-third partition, with no device overlap between the training and validation sets.

## F.2.2. Mixed-lighting condition.

Mixed-lighting condition uses Groups 1–4 as the source set and Group 5 as the target set. As Group 5 is constructed by mixing samples from the other lighting groups, any duplicated samples from Groups 1–4 are excluded from the training set and removed from the released evaluation subset to avoid leakage across protocols. Group 5 provides a broader and more heterogeneous evaluation setting than Within-lighting condition. In this protocol, no additional train/validation split is performed within Group 5; instead, the goal is to measure how well a method transfers from the structured lighting conditions represented by Groups 1– 4 to the more diverse contrast distribution represented by Group 5.

Table S9. Scene-wise PLCC and SRCC on the real-portrait dataset for overall contrast, eye contrast, and micro-contrast. The proposed metrics are applied zero-shot, while IFQA is evaluated after fine-tuning on the corresponding mannequin-based annotations. Values are reported as the point estimate ± standard deviation over 500 image-level bootstrap resamples.
<table><tr><td rowspan="3">Scene</td><td colspan="4">Overall</td><td colspan="4"> $_ \mathrm { E y e }$ </td><td colspan="4">Micro</td></tr><tr><td colspan="2">Ours</td><td colspan="2">IFQA (fine-tuned)</td><td colspan="2">Ours</td><td colspan="2">IFQA (fine-tuned)</td><td colspan="2">Ours</td><td colspan="2">IFQA (fine-tuned)</td></tr><tr><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td></tr><tr><td></td><td></td><td>8005_103 0.914 ± 0.031 0.898 ± 0.046 0.889 ± 0.030 0.894 ± 0.049</td><td></td><td></td><td> $0 . 4 8 7 \pm 0 . 1 0 0$ </td><td>0.549 ± 0.125 -0.228 ± 0.164</td><td></td><td> $- 0 . 1 3 9 \pm 0 . 1 7 0$ </td><td>0.512 ± 0.105</td><td> $0 . 5 3 2 \pm 0 . 1 4 5$ </td><td>0.472 ± 0.098</td><td> $0 . 5 3 0 \pm 0 . 1 3 3$ </td></tr><tr><td></td><td>1606.031 0.632 ± 0.079</td><td> $0 . 6 7 1 \pm 0 . 0 8 4$ </td><td> $0 . 8 7 6 \pm 0 . 0 3 3$ </td><td>0.868 ± 0.046</td><td> $0 . 7 9 3 \pm 0 . 0 3 9$ </td><td> $0 . 8 5 1 \pm 0 . 0 4 5$ </td><td> $0 . 7 9 1 \pm 0 . 0 4 5$ </td><td> $0 . 8 3 7 \pm 0 . 0 4 8$ </td><td> $0 . 6 3 1 \pm 0 . 0 9 5$ </td><td> $0 . 7 3 6 \pm 0 . 0 7 2$ </td><td>0.844 ± 0.043</td><td></td></tr><tr><td>0105.089 0.835 ± 0.038</td><td></td><td> $0 . 8 5 1 \pm 0 . 0 4 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 3 8$ </td><td> $0 . 8 5 3 \pm 0 . 0 4 3$ </td><td> $0 . 8 3 2 \pm 0 . 0 3 2$ </td><td> $0 . 8 3 8 \pm 0 . 0 3 8$ </td><td> $0 . 7 4 8 \pm 0 . 0 6 9$ </td><td> $0 . 7 2 5 \pm 0 . 0 7 6$ </td><td> $0 . 8 6 8 \pm 0 . 0 3 0$ </td><td></td><td></td><td> $0 . 7 4 9 \pm 0 . 0 8 2$ </td></tr><tr><td></td><td>0.689 ± 0.049</td><td> $0 . 7 1 7 \pm 0 . 0 6 6$ </td><td> $0 . 7 3 3 \pm 0 . 0 7 1$ </td><td>0.713 ± 0.103</td><td> $0 . 8 9 3 \pm 0 . 0 2 0$ </td><td> $0 . 9 2 3 \pm 0 . 0 2 6$ </td><td> $0 . 9 1 9 \pm 0 . 0 1 7$ </td><td> $0 . 9 4 3 \pm 0 . 0 2 3$ </td><td> $0 . 7 0 4 \pm 0 . 0 7 0$ </td><td> $0 . 8 7 0 \pm 0 . 0 3 3$ </td><td>0.861 ± 0.031</td><td> $0 . 8 5 7 \pm 0 . 0 4 3$ </td></tr><tr><td>0109_2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 6 7 0 \pm 0 . 0 8 9$ </td><td>0.807 ± 0.053</td><td> $0 . 8 4 4 \pm 0 . 0 5 9$ </td></tr><tr><td>1119.105 0.800 ± 0.035</td><td></td><td> $0 . 8 7 9 \pm 0 . 0 4 5$ </td><td>0.880 ± 0.040 0.857 ± 0.050</td><td></td><td> $0 . 7 6 4 \pm 0 . 0 7 8$ </td><td> $0 . 6 6 5 \pm 0 . 1 0 0$ </td><td> $0 . 9 5 6 \pm 0 . 0 1 0$ </td><td> $0 . 9 3 8 \pm 0 . 0 3 0$ </td><td> $0 . 3 1 6 \pm 0 . 1 1 6$ </td><td> $0 . 2 4 4 \pm 0 . 1 3 6$ </td><td>0.816 ± 0.066</td><td> $0 . 6 0 0 \pm 0 . 1 0 3$ </td></tr><tr><td>1113.510 0.624 ± 0.090</td><td></td><td> $0 . 6 7 1 \pm 0 . 0 9 2$ </td><td>0.846 ± 0.032 0.863 ± 0.045</td><td></td><td> $0 . 8 9 2 \pm 0 . 0 4 8$ </td><td> $0 . 8 7 8 \pm 0 . 0 5 4$ </td><td> $0 . 8 9 2 \pm 0 . 0 4 4$ </td><td> $0 . 8 8 7 \pm 0 . 0 5 5$ </td><td> $0 . 9 0 7 \pm 0 . 0 1 9$ </td><td> $0 . 9 5 9 \pm 0 . 0 1 7$ </td><td>0.797 ± 0.034</td><td> $0 . 8 4 3 \pm 0 . 0 5 7$ </td></tr><tr><td>3115_305 0.880 ± 0.036</td><td></td><td> $0 . 8 9 7 \pm 0 . 0 3 8$ </td><td>0.949 ± 0.012 0.948 ± 0.020</td><td></td><td> $0 . 8 1 7 \pm 0 . 0 2 6$ </td><td> $0 . 8 8 0 \pm 0 . 0 3 1$ </td><td> $0 . 6 9 3 \pm 0 . 0 7 3$ </td><td> $0 . 6 2 3 \pm 0 . 1 0 3$ </td><td> $0 . 9 3 0 \pm 0 . 0 1 7$ </td><td> $0 . 9 4 3 \pm 0 . 0 2 8$ </td><td>0.772 ± 0.057</td><td> $0 . 8 2 3 \pm 0 . 0 6 1$ </td></tr><tr><td>Riverside 0.929 ± 0.030</td><td></td><td>0.984 ± 0.035 0.957 ± 0.019 0.956 ± 0.063  $\begin{array} { l } { { \sf U . 9 0 4 \pm 0 . 0 . 0 3 . } } \\ { { \displaystyle 0 . 8 9 8 + 0 . 0 3 8 } } \end{array}$ </td><td>0.917 ± 0.029 0.898 ± 0.040</td><td></td><td> $0 . 3 7 9 \pm 0 . 1 9 7$   $0 . 8 0 8 \pm 0 . 0 3 9$ </td><td>0.363 ± 0.274  $0 . 8 8 5 \pm 0 . 0 3 6$ </td><td> $0 . 6 8 8 \pm 0 . 1 6 4$   $0 . 8 1 0 \pm 0 . 0 4 4$ </td><td> $0 . 7 0 3 \pm 0 . 2 3 5$   $0 . 7 4 5 \pm 0 . 0 7 9$ </td><td> $0 . 5 7 9 \pm 0 . 1 1 5$ </td><td></td><td>0.963 ± 0.024 0.940 ± 0.069 0.541 ± 0.186</td><td> $0 . 5 4 4 \pm 0 . 1 9 4$ </td></tr><tr><td>0109.3 5203_1060.744 ± 0.052</td><td>0.877 ± 0.026</td><td> $0 . 7 8 2 \pm 0 . 0 6 4$ </td><td>0.695 ± 0.065</td><td> $0 . 7 7 3 \pm 0 . 0 6 5$ </td><td> $0 . 8 6 3 \pm 0 . 0 3 5$ </td><td> $0 . 8 5 3 \pm 0 . 0 5 1$ </td><td> $0 . 0 5 9 \pm 0 . 1 7 0$ </td><td> $0 . 0 3 1 \pm 0 . 1 5 7$ </td><td> $0 . 7 6 2 \pm 0 . 0 5 9$ </td><td> $0 . 5 6 6 \pm 0 . 1 3 2$ </td><td> $0 . 8 6 7 \pm 0 . 0 3 2$ </td><td> $0 . 8 3 7 \pm 0 . 0 5 3$ </td></tr><tr><td colspan="9">Average  ${ \bf 0 . 7 9 2 \pm 0 . 0 1 6 }$   $\mathbf { 0 . 8 2 5 \pm 0 . 0 1 7 }$  0.858 ± 0.013  ${ \bf 0 . 8 6 2 \pm 0 . 0 1 7 }$   $\mathbf { 0 . 7 5 3 \pm 0 . 0 2 5 }$  0.768 ± 0.034</td><td></td><td> $0 . 8 5 1 \pm 0 . 0 6 3$   $0 . 1 5 6 \pm 0 . 1 1 4$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 2 7 }$ </td><td> $0 . 1 4 7 \pm 0 . 1 4 2$   ${ \bf 0 . 6 7 8 \pm 0 . 0 3 2 }$ </td></tr></table>

Table S10. Macro-average PLCC and SRCC across real-portrait scenes for IFQA before and after fine-tuning on the mannequinbased annotations.
<table><tr><td rowspan="2">Target</td><td colspan="2">IFQA (fine-tuned)</td><td colspan="2">IFQA (non-fine-tuned)</td></tr><tr><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td></tr><tr><td>Overall contrast</td><td>0.858</td><td>0.862</td><td>-0.075</td><td>-0.086</td></tr><tr><td>Eye contrast</td><td>0.633</td><td>0.629</td><td>-0.117</td><td>-0.106</td></tr><tr><td>Micro-contrast</td><td>0.693</td><td>0.678</td><td>-0.113</td><td>-0.105</td></tr></table>

Table S11. Image composition by group and mannequin (original vs. edited).

![](images/034857d4a37aebcf3a87b892942d3c4b4cc311f400ba59237803a5c50ebcc966.jpg)

<table><tr><td rowspan="2">Group</td><td colspan="3">Diana</td><td colspan="3">Eugene</td></tr><tr><td>Original</td><td>Edited</td><td>Total</td><td>Original</td><td>Edited</td><td>Total</td></tr><tr><td>Group 1</td><td>38</td><td>37</td><td>75</td><td>42</td><td>33</td><td>75</td></tr><tr><td>Group 2</td><td>47</td><td>28</td><td>75</td><td>49</td><td>26</td><td>75</td></tr><tr><td>Group 3</td><td>45</td><td>30</td><td>75</td><td>41</td><td>34</td><td>75</td></tr><tr><td>Group 4</td><td>59</td><td>16</td><td>75</td><td>62</td><td>13</td><td>75</td></tr><tr><td>Group 5</td><td>61</td><td>39</td><td>100</td><td>64</td><td>36</td><td>100</td></tr><tr><td>Total</td><td>250</td><td>150</td><td>400</td><td>258</td><td>142</td><td>400</td></tr></table>

Figure S15. Annotation stability of the JOD scales under withinlighting and mixed-lighting evaluation. Each observer is excluded in turn, and the JOD scale reconstructed from the remaining observers is compared with the reference scale estimated from all annotations. Bars show the mean PLCC and SRCC across the four observer exclusions, and error bars indicate the corresponding standard deviation.

## F.3. Annotation Consistency Analysis

We conducted two complementary analyses to assess the consistency of the collected annotations. First, we computed binary pairwise agreement as a direct measure on the raw pairwise decisions. The component-level tasks showed moderate agreement: eye contrast achieved 0.666± 0.101, micro-contrast $0 . 7 2 6 \pm 0 . 0 9 2$ , and overall contrast

$0 . 6 7 8 \pm 0 . 1 0 6$ . Micro-contrast produced the highest agreement, consistent with the relatively direct judgment of finedetail visibility. Balanced contrast showed lower agreement $0 . 5 8 8 \pm 0 . 1 2 4$ , reflecting the more subjective nature of this holistic task. Second, we also evaluated the stability of the recovered perceptual scales using a leave-one-observer-out analysis. For each task, we repeatedly removed one observer, recomputed the JOD scale from the remaining annotations, and measured its correlation with the JOD scale obtained from the full annotation set. The results are shown in Figure S15. The resulting PLCC and SRCC values show that the inferred scales remain correlated after observer removal, especially for micro-contrast, which achieves the highest correlations under both evaluation settings. This indicates that the recovered JOD scales retain meaningful stability when individual observers are excluded and can serve as perceptual references for metric validation.

Table S12. Device brands and models included in the mannequin dataset.
<table><tr><td>Brand</td><td># Devices</td><td>Models</td></tr><tr><td>Samsung</td><td>8</td><td>A55 5G, A56 5G, S24 Ultra, S25 Ultra, Z Flip5, Z Flip6, Z Fold5, Z Fold6</td></tr><tr><td>Google</td><td>7</td><td>Pixel 8, 8 Pro, 8a, 9, 9 Pro Fold, 9 Pro XL, 9a</td></tr><tr><td>Xiaomi</td><td>6</td><td>13T Pro, 14, 14 Ultra, 15 Ultra, Redmi Note 13 Pro+, Redmi Note 14 Pro+</td></tr><tr><td>Huawei</td><td>2</td><td>Mate 60 Pro+, Pura 70 Ultra</td></tr><tr><td>Vivo</td><td>2</td><td>X100 Pro, X100 Ultra</td></tr><tr><td>Honor</td><td>2</td><td>Magic 6 Pro, Magic 7 Pro</td></tr><tr><td>Oppo</td><td>1</td><td>Find X7 Ultra</td></tr><tr><td>Asus</td><td>1</td><td>Zenfone 11 Ultra</td></tr></table>

Table S13. Device sets used for Within-lighting condition on Groups 1–4. The split is strictly device-disjoint.
<table><tr><td>Training devices (19)</td><td>AsusZenfone11Ultra, GooglePixel8, GooglePixel9, GooglePixel9ProFold, GooglePixel9ProXL, GooglePixel9a, HuaweiPura70Ultra, SamsungGalaxyA555G, SamsungGalaxyA565G, SamsungGalaxyS25Ultra, SamsungGalaxyZFlip6, Sam- sungGalaxyZFold5, SamsungGalaxyZFold6, VivoX100Pro, VivoX100Ultra, Xiaomi14Ultra, Xiaomi15Ultra, XiaomiRedmiNote13ProPlus5G, XiaomiRed- miNote14ProPlus5G</td></tr><tr><td>Validation devices (10)</td><td>GooglePixel8Pro, GooglePixel8a, HonorMagic6Pro, HonorMagic7Pro, Huawei- Mate60ProPlus, OppoFindX7Ultra, SamsungGalaxyS24Ultra, SamsungGalaxyZFlip5, Xiaomi13TPro, Xiaomi14</td></tr></table>

## F.4. Device-Level Contrast Analysis

This analysis illustrates how the proposed annotations can be used to compare smartphone rendering behavior at the device level. Figure S16 shows the device-level contrast analysis. To remove cross-subgroup scale differences, each JOD target was normalized within each (group, mannequin) subgroup. For sample i in subgroup $c = ( g , p )$ , we computed

$$
z _ { i } ^ { ( m ) } = \frac { x _ { i } ^ { ( m ) } - \mu _ { c } ^ { ( m ) } } { \sigma _ { c } ^ { ( m ) } } ,\tag{11}
$$

where $x _ { i } ^ { ( m ) }$ is the raw JOD value for metric m, and $\mu _ { c } ^ { ( m ) }$ $\sigma _ { c } ^ { ( m ) }$ are the subgroup-wise mean and standard deviation.

To avoid bias from non-uniform sample counts across groups and mannequins, we used macro aggregation. For each device d, metric m, and subgroup c, we first computed a subgroup-level device mean

$$
\mu _ { d , c } ^ { ( m ) } = \frac { 1 } { N _ { d , c } } \sum _ { i \in \mathcal { T } _ { d , c } } z _ { i } ^ { ( m ) } ,\tag{12}
$$

where $\mathcal { T } _ { d , c }$ is the set of samples captured by device d in subgroup $c ,$ and $N _ { d , c } ~ = ~ \vert \mathcal { T } _ { d , c } \vert$ . Then we averaged these subgroup means with equal weight:

$$
\bar { z } _ { d } ^ { ( m ) } = \frac { 1 } { \left| C _ { d } \right| } \sum _ { c \in C _ { d } } \mu _ { d , c } ^ { ( m ) } ,\tag{13}
$$

where $C _ { d }$ is the set of available (group, mannequin) subgroups for device d. Therefore, each bar in the ranking plots represents the macro-aggregated device mean of normalized JOD scores. Error bars indicate the 95% confidence interval estimated across the available subgroup-level means for each device.

Device score and Contrast Amount Ordering Using the macro-aggregated score $\bar { z } _ { d } ^ { ( m ) }$ , we then applied metric-wise min–max scaling across devices for visualization:

$$
\tilde { s } _ { d } ^ { ( m ) } = \frac { \bar { z } _ { d } ^ { ( m ) } - \operatorname* { m i n } _ { d ^ { \prime } } \bar { z } _ { d ^ { \prime } } ^ { ( m ) } } { \operatorname* { m a x } _ { d ^ { \prime } } \bar { z } _ { d ^ { \prime } } ^ { ( m ) } - \operatorname* { m i n } _ { d ^ { \prime } } \bar { z } _ { d ^ { \prime } } ^ { ( m ) } } \in [ 0 , 1 ] .\tag{14}
$$

Figure S17 (a) shows this normalized profile for Overall, Eye, Micro, and Balanced contrast. We define the contrast-amount index as the equal-weight average of the three amount dimensions:

$$
C _ { d } = \frac { 1 } { 3 } \Big ( \tilde { s } _ { d } ^ { ( \mathrm { O v e r a l l } ) } + \tilde { s } _ { d } ^ { ( \mathrm { E y e } ) } + \tilde { s } _ { d } ^ { ( \mathrm { M i c r o } ) } \Big ) .\tag{15}
$$

Figure S17 (b) ranks devices by $C _ { d }$ . This index captures amount of contrast rather than perceptual quality; therefore, a higher $C _ { d }$ does not necessarily imply a higher Preference score. We treat balanced contrast Preference as the primary utility endpoint, and $C _ { d }$ as an explanatory descriptor of physical contrast strength.

## G. Pairwise Comparison Tool

During the experiments, the annotators performed pairwise comparisons using a custom annotation tool designed to present two images side by side and record their judgments for different perceptual attributes. For each comparison, the annotators selected the image that best matched the target criterion, depending on the annotation task. Figures S18 and S19 show example interfaces used for overall tonal contrast annotation. Figures S20 and S21 are for eye contrast annotation. Figures S22 and S23 illustrate example interfaces for micro-contrast annotation. Lastly, Figures S24 and S25 show example interfaces for balanced contrast annotation. Before the main experiment, all annotators were provided with detailed annotation guidelines and received two hours of prior training to ensure a consistent understanding of the evaluation criteria. Each annotation session took approximately one hour per attribute. The total annotation time amounted to 160 hours across all annotations.

![](images/64ad6d1b410d22c77af5e1e4b761d3a8248f716a95389e6fefcabc9baefc0e3b.jpg)

(a) Overall Contrast JOD ranking  
![](images/79b44a9176df2a805686b2896dd758e9f7edfbe4e70c59d7ba28b270c1d237f5.jpg)

(b) Eye Contrast JOD ranking  
![](images/cdefd313a4bfb810748a4748cea26e7b4e85779ffb0b2066c39b7651db606401.jpg)

(c) Micro Contrast JOD ranking  
![](images/1921b4674105d10c597f868991b378984facd2fc2dc4a5810cbe58cc8d8ec034.jpg)  
(d) Balanced Contrast JOD ranking  
Figure S16. Device-level pooled rankings with 95% confidence intervals for four JOD targets. Overall, Eye, and Micro represent amount of contrast dimensions, while Balanced contrast reflects subjective utility. Scores are macro-aggregated subgroup-normalized JOD values expressed in standard-deviation units.

Table S14. Device split summary by group and mannequin under Within-lighting condition.
<table><tr><td>Group</td><td>Person</td><td>Train Dev</td><td>Val Dev</td><td>Overlap Dev</td></tr><tr><td>Group1</td><td>Diana</td><td>18</td><td>10</td><td>0</td></tr><tr><td>Group1</td><td>Eugene</td><td>18</td><td>10</td><td>0</td></tr><tr><td>Group2</td><td>Diana</td><td>16</td><td>10</td><td>0</td></tr><tr><td>Group2</td><td>Eugene</td><td>17</td><td>10</td><td>0</td></tr><tr><td>Group3</td><td>Diana</td><td>17</td><td>10</td><td>0</td></tr><tr><td>Group3</td><td>Eugene</td><td>17</td><td>10</td><td>0</td></tr><tr><td>Group4</td><td>Diana</td><td>18</td><td>10</td><td>0</td></tr><tr><td>Group4</td><td>Eugene</td><td>18</td><td>10</td><td>0</td></tr></table>

Table S15. Split summary aggregated by mannequin under Within-lighting condition.
<table><tr><td>Person</td><td>Total</td><td>Train Total</td><td>Val Total</td><td>Train Ratio</td><td>Val Ratio</td></tr><tr><td>Diana</td><td>300</td><td>195</td><td>105</td><td>0.6500</td><td>0.3500</td></tr><tr><td>Eugene</td><td>300</td><td>200</td><td>100</td><td>0.6667</td><td>0.3333</td></tr></table>

## G.1. JOD-Ordered Examples for Each Attribute

To illustrate the perceptual scales recovered from the pairwise annotations, we show example images sorted by their JOD values for each attribute. In all figures, the examples are ordered from lower to higher JOD. For overall contrast, eye contrast, and micro-contrast, higher JOD corresponds to more of the corresponding contrast attribute. For balanced contrast preference, higher JOD indicates a stronger human preference for the corresponding image. This ordering allows a visual inspection of how each recovered perceptual scale varies across the dataset. Figures S26 and S27 show overall contrast examples for Eugene and Diana. Figures S28 and S29 show eye-contrast examples. Figures S30 and S31 show micro-contrast examples. Finally, Figures S32 and S33 show examples sorted by balanced contrast preference JOD.

Device × Metric Normalized Profile (higher = stronger amount / higher preference)
<table><tr><td rowspan=6 colspan=1>VivoX100UltraHuaweiPura70UltraGooglePixel9aHuaweiMate60ProPlusOppoFindX7Ultra</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1>Xiaomi15Ultra</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>SamsungGalaxyZFlip5</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.62</td></tr><tr><td rowspan=6 colspan=1>AsusZenfone11UltraSamsungGalaxyS25UltraSamsungGalaxyA565GSamsungGalaxyA555GHonorMagic6ProSamsungGalaxyZFold6</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.55</td></tr><tr><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.54</td></tr><tr><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.53</td></tr><tr><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.52</td></tr><tr><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.48</td></tr><tr><td rowspan=2 colspan=1>device       SamsungGalaxyZFlip6GooglePixel9ProXL</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.47</td></tr><tr><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.47</td></tr><tr><td rowspan=1 colspan=1>GooglePixel8a</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.46</td></tr><tr><td rowspan=1 colspan=1>XiaomiRedmiNote13ProPlus5G</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1>GooglePixel9ProFold</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1>SamsungGalaxyS24UItra</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.43</td></tr><tr><td rowspan=1 colspan=1>XiaomiRedmiNote14ProPlus5G</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.41</td></tr><tr><td rowspan=1 colspan=1>SamsungGalaxyZFold5</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.40</td></tr><tr><td rowspan=1 colspan=1>GooglePixel8</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.39</td></tr><tr><td rowspan=1 colspan=1>GooglePixel8Pro</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>0.32</td></tr><tr><td rowspan=1 colspan=1>HonorMagic7Pro</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.32</td></tr><tr><td rowspan=1 colspan=1>Xiaomi14</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.29</td></tr><tr><td rowspan=1 colspan=1>GooglePixel9</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.26</td></tr><tr><td rowspan=1 colspan=1>Xiaomi13TPro</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.11</td></tr><tr><td rowspan=1 colspan=1>Xiaomi14Ultra</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.02</td></tr></table>

(a) Device scorecard across Overall/Eye/Micro/Preference (balanced contrast). Device Ordering by Contrast Amount (not quality ranking)  
![](images/ed03bad541b93a37adb3ce6c3abcee0260c02c497237241898a1404b243a0e60.jpg)  
(b) Device ordering by contrast-amount index $C _ { d }$ (equal-weight average of Overall/Eye/Micro).  
Figure S17. Device-level summary under z-score-based analysis. Panel (a) provides a compact cross-metric profile per device, and panel (b) highlights relative ordering by contrast amount.

![](images/b5003e08be7fae7e50ed2e1df02197d4c8fb3537a3410182ea1172b0ec23eff2.jpg)  
Figure S18. Example of the annotation tool used to acquire pairwise comparisons (Overall contrast, Eugene)

![](images/ae73eb1cfbb198a51e1658e83bfffd4fd04cfa7f37ee4de1aa7fbf25a9912f85.jpg)  
Figure S19. Example of the annotation tool used to acquire pairwise comparisons (Overall contrast, Diana)

![](images/1dfeb0b95caf553f1f9f1a2ddc6b2499c27d911b14b037832a5d984bcd9887cd.jpg)  
Figure S20. Example of the annotation tool used to acquire pairwise comparisons (Eye contrast, Eugene)

![](images/d7d00e52fdac0192a4cc4d99fe1e47974efac10e38a71623443e26bebeeaf57b.jpg)  
Figure S21. Example of the annotation tool used to acquire pairwise comparisons (Eye contrast, Diana)

![](images/388eed233165524dc89f24ed8423f3a1bff54496bd91e863d21aba52e4a5fbe6.jpg)  
Figure S22. Example of the annotation tool used to acquire pairwise comparisons (Micro-contrast, Eugene)

![](images/645a6c4bb534572d81f9a28b0dabfe4b33215431666f095165f2ea80ac0c1014.jpg)  
Figure S23. Example of the annotation tool used to acquire pairwise comparisons (Micro-contrast, Diana)

Figure S24. Example of the annotation tool used to acquire pairwise comparisons (Balanced contrast, Eugene)  
![](images/40312042b61d6eed1a80ad941924405ccc4112d4b0ac9434f85d8d3125efcf26.jpg)  
Figure S25. Example of the annotation tool used to acquire pairwise comparisons (Balanced contrast, Diana)

![](images/7a1e6c954ced7ede176379e62c9f1d53e40b724e1ba22ffba2b0152307a3569f.jpg)

![](images/535db71b0e715994b54769d1ce03c4c4b84a67eb53dcf004073168ed46d65c06.jpg)

![](images/03a4602b645a4c1762bae57a98dcbacd9e1f60aba03b616993c8ec3140d79e12.jpg)

![](images/d4030f2ad9c09c5d86e862f01c71a8c3053464f71bf60b4bd0adae1c2f7a4129.jpg)  
Figure S26. Examples of Eugene images sorted by JOD for overall contrast, from lower to higher values.

![](images/dea949d23743766ba37af2b6630a8f9b84fb5a8b0ada6c0dea38fa37844d472c.jpg)

![](images/76ca59f075520664fc3f43410d872922a2f70a5d9b9c10dca206a91d0602fb03.jpg)

![](images/68022c159465c1c748d9d300310f3d7b8dd54bbf5a14b54d362c2db59387e786.jpg)

![](images/e5ad9074f4b4ad728633d06a9e67d7209eace95a43b0ab43181041c8f0f052be.jpg)  
Figure S27. Examples of Diana images sorted by JOD for overall contrast, from lower to higher values.

![](images/3e18d389c417606f73c18c669db73c96e738f3a7879269e69e41fdbcd157cf47.jpg)  
Figure S28. Examples of Eugene images sorted by JOD for eye contrast, from lower to higher values.

![](images/bd5096a35386137034732dc03b9b1c827838c7b8c55ab09616e84a1b788198b6.jpg)  
Figure S29. Examples of Diana images sorted by JOD for eye contrast, from lower to higher values.

![](images/e268185fa82218193f75ccc540f3bbff25e0be62fee9afccc35528654a2f8012.jpg)

![](images/b8741cf73c8afa9961a8529fe8f536a99b597acb1488c051960d55290b476e54.jpg)

![](images/85d509cf99e4dd43fb33f3115cef5478d7aefbd340f569c75fe6aeb75c3070f2.jpg)

![](images/c2a96d878c23ef722215bd814efa4237e554aedfc37e59fb2104c1a09e5d09b1.jpg)  
Figure S30. Examples of Eugene images sorted by JOD for micro-contrast, from lower to higher values.

![](images/440cecb87bff6855a8a3fe27f3e518de4d5a6d15aa522f62aae8336dddb9d715.jpg)

![](images/517684d310261f90701c56a398a93f1d0a5904a93c33016a6a5c7a9b2db071f0.jpg)  
-1.69

![](images/dc3e1ea154c39483d31a168e7652e793bcf6a4ef04230b057997ec09381b1c82.jpg)  
1.22

![](images/2ef71859448a1d43e93ea2d84243daeee4b773f92b2f54d48089ba67529cdf3d.jpg)  
Figure S31. Examples of Diana images sorted by JOD for micro-contrast, from lower to higher values.

![](images/61d552cb4becd01e4bc827cf61fda1eef9f6ca9abfc94f4013ecb07be6b87203.jpg)  
-4.8

![](images/30abfa42e5462f7801d4ee6de35c3678e91f4bd2edab5918acd1f015ea8879b0.jpg)  
-2.1

![](images/5dc8f9a673e4a4ae460d45ae9298eeb0a4b11b1ebb91795d3ad4828379dda468.jpg)  
0.07

![](images/16843f9b9f04121da018255defff7946a1b6348f849d8bb01fb7004c1b7fbc13.jpg)  
2.42  
Figure S32. Examples of Eugene images sorted by JOD for balanced contrast preference, from lower to higher values.

![](images/29126bd83d910ba3c6db2b07040f58910d4cd6f003ca012108ee89076f1ff8be.jpg)  
-5.12

![](images/49545dc942e9328f113852a2c8f1712cb32d23db8eccb839e840e0f6e46f2762.jpg)  
-2.12

![](images/ca02b156725b636985ea4423633080b05d5735e98c5a200c194c183cab78154d.jpg)  
0.09

![](images/47d6d01a134a0556ae3137ad5b336049e11de0f42fadfe49f56d850ebb085d1d.jpg)  
2.12  
Figure S33. Examples of Diana images sorted by JOD for balanced contrast preference, from lower to higher values.