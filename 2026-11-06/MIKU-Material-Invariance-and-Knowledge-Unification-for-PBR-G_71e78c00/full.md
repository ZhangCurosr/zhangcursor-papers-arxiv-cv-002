![](images/39287cb057a617c7074f6b9840168d2358bfbeb9f38e0361989080c84bbbf2f0.jpg)

# MIKU: Material Invariance and Knowledge Unification for PBR Generation

YAJIE BAO<sup>∗</sup>, Nanjing University, China and DreamTech, China

JIAYE FU, Peking University, China and DreamTech, China

YOUTIAN LIN, Nanjing University, China

SHUANG WU, Nanjing University, China and DreamTech, China

JIACHEN QIAN, The University of Hong Kong, Hong Kong and DreamTech, China

FEIHU ZHANG, DreamTech, China

XUN CAO, Nanjing University, China

YAO YAO<sup>†</sup>, Nanjing University, China

![](images/15f91299796f390b1658afa90f47fa8f2cec89b29488d0fbe75e2c166dd81b7c.jpg)  
Fig. 1. The entire scene is assembled from AIGC-generated meshes, where both the surface textures and PBR material maps are consistently synthesized by our method.

<sup>∗</sup>This work was conducted during an internship at DreamTech.   
<sup>†</sup>Corresponding author.

Authors’ Contact Information: Yajie Bao, Nanjing University, China and DreamTech, China, 652025710001@smail.nju.edu.cn; Jiaye Fu, Peking University, China and DreamTech, China, jyfu@stu.pku.edu.cn; Youtian Lin, Nanjing University, China, linyoutian.loyot@gmail.com; Shuang Wu, Nanjing University, China and DreamTech, China, wushuang@smail.nju.edu.cn; Jiachen Qian, The University ofHong Kong, Hong Kong and DreamTech, China, qianjiachen@dreamtech.ai; Feihu Zhang, DreamTech, China, zhangfeihu@dreamtech.ai; Xun Cao, Nanjing University, China, caoxun@nju. edu.cn; Yao Yao, Nanjing University, China, yaoyao@nju.edu.cn.

In physically based rendering (PBR) content creation, high-quality material generation remains a fundamental challenge, primarily hindered by the absence of standardized representations across existing datasets and synthesis pipelines. The heterogeneity in material definitions, rendering conventions, and parameterization schemes introduces significant inconsistencies, ultimately leading to unstable and physically implausible PBR synthesis results. To systematically address these limitations, we propose MIKU, a unified framework that establishes standardized PBR asset representation and enables physically consistent material generation across diverse 3D content domains. Our core contribution is a comprehensive unified PBR material processing pipeline that standardizes the entire workflow from data preprocessing to final texture synthesis. This pipeline enforces consistent representation across geometry and texture domains throughout material generation. Critically, it addresses the long-standing challenge of inconsistent material parameterizations by introducing a normalization

scheme that harmonizes diverse input sources into a common representation space, enabling robust cross-dataset training and inference.Building upon this principled formulation, we train a novel physically aware difusion model specifically designed for PBR texture generation. Our model operates within a two-stage multi-view synthesis framework.The first stage focuses on geometry-aware albedo generation, where multi-view albedo maps are synthesized under explicit geometric guidance to ensure cross-view ap pearance consistency. The second stage performs metallic–roughness (MR) generation using our trained difusion model, which predicts physically plausible roughness and metallic maps conditioned on the generated albedo and the unified representation. The model is trained end-to-end on standard ized data, learning to generate materials that respect category-level material priors and semantic part boundaries.By explicitly decoupling representation standardization from appearance synthesis, MIKU substantially improves both the stability and physical consistency of generated materials. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in both texture quality and semantic consistency across diferent object parts, significantly advancing automated material generation for 3D content creation workflows.

CCS Concepts: • Computing methodologies → Artificial intelligence.

Additional Key Words and Phrases: PBR material generation, texture generation, difusion models

ACM Reference Format:

Yajie Bao, Jiaye Fu, Youtian Lin, Shuang Wu, Jiachen Qian, Feihu Zhang, Xun Cao, and Yao Yao. 2026. MIKU: Material Invariance and Knowledge Unification for PBR Generation . ACM Trans. Graph. 45, 6, Article 181 (De cember 2026), 17 pages. htps://doi.org/10.1145/3842521

## 1 Introduction

The generation of high-fidelity textures for 3D models is a fundamental component ofrealistic digital asset creation and has attracted sustained research interest in the graphics and vision communities [He et al. 2025b; Jin et al. 2024; Liu et al. 2023, 2024; Poole et al. 2022; Shi et al. 2023; Tang et al. 2024]. Among texture generation methods, physically based rendering (PBR) has become a central paradigm for producing realistic materials by modeling surface reflectance with properties such as albedo, roughness, and metallic, enabling consistent rendering under varying illumination. This has motivated increasing eforts in PBR-aware texture generation.

A line of work adopts single-view image-conditioned formulations, where generation is guided by a single reference view to transfer appearance cues such as structure and material properties to the output. Methods such as IntrinsicAnything [Chen et al. 2024], IntrinsiX [Kocsis et al. 2025], DualMat [Huang et al. 2025a], and RGB↔X [Zeng et al. 2024b] leverage difusion priors, cross-intrinsic attention, and bidirectional or dual-path difusion frameworks to enhance intrinsic decomposition and improve cross-channel consistency. However, these approaches are fundamentally constrained by single-view ambiguity, which often leads to incomplete geometric reasoning and limited cross-view consistency.

Another line of work adopts multi-view image-conditioned formulations, where generation is guided by a set of calibrated or uncalibrated observations to better constrain 3D appearance reasoning. Methods such as SuperMat [Hong et al. 2025], MVPainter [Shao et al. 2025], IDArb [Li et al. 2024], MaterialMVP [He et al. 2025b], MuMA [Zhu et al. 2025], PacTure [Fei et al. 2025], PBR3DGen [Wei et al. 2026], MCMat [Zhu et al. 2024], and LumiTex [Bao et al.

2025] build upon difusion-based architectures with cross-view and cross-domain attention, geometry or reference-aware conditioning, and multi-view feature aggregation to jointly synthesize physically based material properties with improved spatial coherence and appearance consistency across views. Despite these advantages, we observe that the training process of models for PBR attribute prediction is often unstable, particularly under varying data distributions and view configurations. In addition, the learned representations often exhibit limited consistency in PBR material estimation, where visually similar or semantically identical objects may be assigned diferent predictions of physically based properties across instances or views. Furthermore, their generalization capability is largely restricted to specific training datasets and predefined data distributions, resulting in degraded robustness when encountering unseen categories, diverse material patterns, or in-the-wild 3D assets.

This raises a central question: Does the observed inconsistency primarily stem from limitations of the generative model, or from the inherent ambiguity in PBR material definitions? Addressing this question is non-trivial, as learning PBR material attributes from image observations is intrinsically ill-posed. A fundamental challenge in learning PBR material attributes is the ambiguity of artist-defined reflectance annotations. For the same object, dif ferent artists may assign substantially diferent metallic–roughness (MR) configurations, all of which can be physically plausible yet correspond to distinct material interpretations and visually reasonable renderings. Moreover, even highly similar rendered appearances may originate from markedly diferent MR definitions, indicating that small image-space variations do not necessarily imply small discrepancies in the underlying material-parameter space. As shown in Fig. 2, even when geometry, albedo, camera view, and illumination are fixed, large discrepancies in MR space can be compressed into only subtle diferences in the rendered shading. This weakens image-space supervision and makes material recovery an underconstrained inverse rendering problem, where multiple PBR configurations can explain similar appearances. Training on such ambiguous annotations encourages networks to regress toward averaged or over-smoothed material estimates, leading to reduced material contrast, loss of high-frequency reflectance details, and inconsistent predictions across views or instances.

Motivated by the above observations, we introduce a unified data construction and learning framework for physically consistent PBR material generation. Specifically, for each artist-authored asset, we establish a compact PBR material taxonomy and render the asset from six calibrated viewpoints. We filter low-quality samples using a joint complexity metric over albedo and material (MR) variance. The remaining assets are processed by a vision-language model for material recognition, followed by SAM3 [Carion et al. 2025]-based segmentation guided by predicted material prompts. We further apply afine projection with residual injection and manual refinement to preserve fine-grained details and ensure cross-view consistency. On the resulting dataset, we train a two-stage packed PBR generator under a multi-resolution and multi-view formulation, enabling physically consistent material synthesis. The main contributions of this work are summarized as follows.:

![](images/c8d930359bc40a9e7291fe2fa66b9c1f413c14343d3953cc3bb15d1297895aa9.jpg)  
Fig. 2. Illustration of the discrepancy atenuation in image-supervised PBR learning. Even with fixed geometry, albedo, camera view, and illumination, substantially diferent MR conditions may result in only minor shading diferences after rendering.

• We propose a unified data processing pipeline for constructing high-quality PBR datasets, which enforces consistent material definitions, controlled illumination, and semantic alignment across assets. This pipeline produces large-scale, standardized training data with improved diversity and reliability.

• We introduce an end-to-end multi-view PBR prediction model that operates at 1024 × 1024 resolution and achieves inference within one minute. The model jointly predicts geometry consistent material properties across views, enabling eficient and high-fidelity PBR generation.

• We identify and demonstrate that inconsistency in material definitions and illumination introduces significant ambiguity into the supervision signal, which negatively impacts model stability. By enforcing data-level consistency, our approach reduces this ambiguity and leads to more stable and coherent material predictions across spatial regions and viewpoints.

## 2 Related Work

Recent 3D generation methods have shifted from geometric modeling alone to jointly synthesizing geometry and physically-based rendering (PBR) materials, requiring disentangled intrinsic channels—albedo, roughness, and metallic—that remain consistent across views; however, prior methods sufer from entanglement with lighting and inconsistent artist-defined annotations, which degrade relighting, destabilize training, and increase variance across object categories, motivating our approach, which addresses both issues.

Texture Generation. Existing texture generation methods can be broadly grouped into several categories according to their model architectures, input representations, and supervision settings. The first category builds upon large-scale pretrained image difusion models [Labs 2024, 2025; Peebles and Xie 2023; Podell et al. 2023; Ramesh et al. 2021; Rombach et al. 2022]. An important line of work leverages Score Distillation Sampling (SDS) [Poole et al. 2022], which optimizes 3D representations by distilling gradients from pretrained 2D difusion models, providing implicit multi-view supervision. Subsequent works improve geometry fidelity, view consistency, and material quality through enhanced regularization, guidance strategies, and hybrid representations [Chen et al. 2023; Lin et al. 2023;

Tang et al. 2023; Wang et al. 2023]. Building upon these advances, another branch explicitly synthesizes view-consistent multi-view observations by incorporating geometric guidance (e.g., depth, normals, or camera poses) to enforce cross-view alignment [Bao et al. 2025; He et al. 2025b; Hong et al. 2025; Huang et al. 2025b; Hunyuan3D et al. 2025; Lai et al. 2025a; Li et al. 2025; Liang et al. 2025; Liu et al. 2025, 2024; Long et al. 2024; Zeng et al. 2024a]. The second category formulates multi-view generation within a video difusion framework, where diferent viewpoints are modeled as temporally correlated frames. Temporal attention and motion-aware priors are leveraged to improve cross-view consistency and reduce inter-view discrepancies [Kim et al. 2025; Voleti et al. 2024; Yuan et al. 2025; Zuo et al. 2024]. The third category explores native 3D generation methods [Chen et al. 2025; Lai et al. 2025b; Xiang et al. 2025a; Yu et al. 2024; Zeng et al. 2025], which explicitly model 3D representa tions and generate texture through rendering-based or 3D-aware generative pipelines, thereby enforcing geometric consistency in a unified spatial domain. Complementary to texture generation, mesh generation methods [Jia et al. 2025; Lai et al. 2025a; Wu et al. 2024, 2025; Xiang et al. 2025b] synthesize textured 3D assets by first generating geometry and then producing textures via UV mapping or image-based refinement within an AIGC pipeline.

PBR Generation. Physically-based rendering (PBR) generation aims to infer intrinsic material properties, such as albedo, metallic, and roughness, that are consistent with underlying geometry and illumination, enabling physically faithful rendering of 3D assets. The first category focuses on single-view PBR generation, which estimates material properties from a single image under limited geometric and lighting cues, often sufering from inherent ambiguity due to under-constrained observations [Chen et al. 2024; Huang et al. 2025a; Kocsis et al. 2025; Luo et al. 2025; Raghavan et al. 2025; Vecchio et al. 2024; Xue et al. 2024, 2026; Ying et al. 2025; Zeng et al. 2024b]. The second category explores multi-view image-based PBR generation, which leverages multiple observations to learn viewconsistent material representations and improves texture-level synthesis through cross-view supervision [Bao et al. 2025; Hasselgren et al. 2026; He et al. 2025b; Li et al. 2024; Munkberg et al. 2025; Wei et al. 2026; Zhu et al. 2025, 2024].

![](images/07e799a674b690f9f46745c5149fc45ffbb3f6e8b37e11ccbbbb0f56f2c49568.jpg)  
Fig. 3. Visualization of our color-preserving illumination strategy. Compared with chromatic illumination, which introduces color casts and appearance ambiguity, achromatic illumination preserves intrinsic albedo colors while providing shading cues.

Table 1. Metallic–roughness (MR) definitions for the 20 predefined semantic material categories and the four optional fine-grained metal subclasses under our unified PBR specification. Each category is associated with a representative MR range that serves as a consistent material prior during data standardization.
<table><tr><td>Material Class</td><td>Metallic</td><td>Roughness</td></tr><tr><td>Skin</td><td>0.0</td><td>0.40-0.50</td></tr><tr><td>Eye</td><td>0.0</td><td>0.02-0.12</td></tr><tr><td>Hair</td><td>0.0</td><td>0.35-0.45</td></tr><tr><td>Metal</td><td>1.0</td><td>0.00-0.10</td></tr><tr><td>Teeth</td><td>0.0</td><td>0.28-0.34</td></tr><tr><td>Lips</td><td>0.0</td><td>0.30-0.36</td></tr><tr><td>Plastic</td><td>0.0</td><td>0.32-0.38</td></tr><tr><td>Rubber</td><td>0.0</td><td>0.48-0.61</td></tr><tr><td>Glass</td><td>0.0</td><td>0.02-0.05</td></tr><tr><td>Transparent Liquid</td><td>0.0</td><td>0.01-0.04</td></tr><tr><td>Ceramic</td><td>0.0</td><td>0.15-0.26</td></tr><tr><td>Stone</td><td>0.0</td><td>0.66-0.73</td></tr><tr><td>Wood</td><td>0.0</td><td>0.40-0.60</td></tr><tr><td>Leather</td><td>0.0</td><td>0.49-0.56</td></tr><tr><td>Fabric</td><td>0.0</td><td>0.77-0.84</td></tr><tr><td>Fur</td><td>0.0</td><td>0.73-0.80</td></tr><tr><td>Paper</td><td>0.0</td><td>0.82-0.89</td></tr><tr><td>Bone</td><td>0.0</td><td>0.47-0.54</td></tr><tr><td>Gemstone</td><td>0.0</td><td>0.03-0.07</td></tr><tr><td>Gold</td><td>1.0</td><td>0.09-0.15</td></tr><tr><td>Mirror Metal</td><td>1.0</td><td>0.01-0.03</td></tr><tr><td>Polished Metal</td><td>1.0</td><td>0.19-0.25</td></tr><tr><td>Semi-Rough Metal</td><td>1.0</td><td>0.35-0.42</td></tr><tr><td>Matte Metal</td><td>1.0</td><td>0.68-0.76</td></tr></table>

## 3 Dataset Curation

## 3.1 Relighted Data for Albedo Learning

Environmental illumination plays a critical role in rendering PBR training data. As illustrated in Fig. 3, chromatic illumination can introduce strong color casts and spatially varying shading efects, leading to substantial discrepancies between the observed appearance and the underlying albedo. In particular, under chromatic illumination, the reflected appearance is jointly determined by intrinsic material properties and the spectral composition of the environment, which increases the ambiguity of albedo estimation since multiple albedo–illumination combinations may explain similar observations.

To reduce this entanglement, we primarily adopt achromatic illumination, as shown in the third column of Fig. 3, where the lighting is constrained to be color-neutral and mainly contributes highlights and shadows without introducing additional chromatic cues. This setting encourages the model to focus on geometry-dependent shading and material-dependent reflectance variations, enabling more stable recovery of intrinsic albedo from shaded observations. It also reduces the risk that the network incorrectly attributes illuminationinduced color shifts to the underlying surface reflectance. Nevertheless, real-world inference scenarios may still involve chromatic light sources, such as neon city environments, sunset scenes, or outdoor scenes with strong environmental color casts. Therefore, rather than completely excluding colored lighting, we use achromatic illumination with a probability of 90% and chromatic illumination with a probability of 10% during training. This mixed strategy improves robustness to realistic colored lighting while maintaining a predominantly color-preserving supervision signal for PBR material learning.

## 3.2 Semantic-Aligned PBR Curation

As illustrated in Fig. 4, our dataset construction begins by consolidating assets from two complementary sources: the publicly available TexVerse dataset [Zhang et al. 2025], which contains approximately 160K assets, and a curated in-house collection of 100K high-quality proprietary assets. Before data cleaning, filtering, and material refinement, this initial asset pool comprises approximately 260K PBR assets in total, providing a large-scale starting corpus for subsequent dataset construction.

## Material Taxonomy and Multi-view Rendering.

For each artist-authored asset, we convert its material annotations into a unified PBR attribute space rather than relying on datasetspecific material definitions. We first organize the materials into 20 predefined semantic categories and map each category to a canonical metallic–roughness (MR) representation, as summarized in Table 1. Rather than manually assigning arbitrary parameters or enforcing a single fixed value for each category, we estimate the empirical MR distribution of each material class from the collected artist-authored assets. For each category, we identify the dominant high-density region and adopt the corresponding MR interval as its representative reference range. For example, dielectric materials such as wood are assigned a metallic value of 0 and exhibit a dominant roughness range of approximately 0.4–0.6, whereas highly reflective metallic materials typically have metallic values close to 1.0 and substantially lower roughness. These category-level priors serve as reference ranges rather than strict physical constraints, thereby retaining instance-specific variations while suppressing inconsistent or atypical artist-defined annotations. This standardized parameterization reduces annotation ambiguity and provides more consistent supervision across heterogeneous data sources.

For the generic Metal category, we further provide an optional fine-grained specification to capture the substantial variation in reflective behavior among metallic surfaces. Specifically, metallic materials can be subdivided into four roughness-dependent subclasses: Mirror Metal, Polished Metal, Semi-Rough Metal, and Matte Metal. These subclasses correspond to progressively increasing roughness levels, enabling the unified representation to distinguish metallic surfaces with markedly diferent reflection characteristics while retaining Metal as the general semantic category.

![](images/c718ef2cd71ac1d7a2a1ce46bc1bef4efa8cbbb2f44214191cc03ad2b28d513e.jpg)  
Fig. 4. Overview of Our Multi-Stage Pipeline for High-Quality PBR Material Dataset Construction and Refinement. This figure illustrates our end-to-end pipeline for constructing a high-quality PBR material dataset. We first collect multi-source data from the public TexVerse dataset and private asset collections. We then define physically based material properties and render multi-view observations under controlled setings. Next, we apply albedo extraction and PBR-aware filtering to remove low-quality samples. Vision-Language Models (VLMs) with SAM3 are used for material recognition and semantic segmentation to enable fine-grained decomposition. Finally, we enhance texture details via afine transformations and high-frequency boosting, followed by human validation to ensure data quality.

To ensure suficient appearance diversity and geometric coverage, each asset is further rendered from six calibrated viewpoints using the illumination strategy described above.

Data Filtering via Complexity Analysis. To ensure the quality and diversity of the dataset, we introduce a complexity metric defined over both albedo and MR representations. The proposed metric jointly captures global photometric variation and local structural details, which are critical for identifying informative training samples. Specifically, given a set of � rendered views $\{ I ^ { ( i ) } \} _ { i = 1 } ^ { N } ,$ , we define the overall complexity score as the average across views:

$$
C = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \lambda _ { v } \operatorname { V a r } ( I ^ { ( i ) } ) + \lambda _ { g } \mathbb { E } \left[ \Vert \nabla I ^ { ( i ) } \Vert _ { 2 } ^ { 2 } \right] \right) .\tag{1}
$$

where Var(·) measures the spatial variance of pixel intensities, reflecting global appearance diversity, and ∇� denotes the image gradient, whose squared ℓ<sub>2</sub> norm captures high-frequency structural information. The expectation operator E[·] is computed over all pixels. In practice, we set $\lambda _ { v } ~ = ~ 0 . 9$ and $\lambda _ { g } \ : = \ : 0 . 1$ to balance the contributions of photometric variation and structural complexity. Samples with low complexity scores are filtered out, as they typically correspond to trivial textures or overly simple geometry with limited learning signal. This filtering strategy efectively removes uninformative samples and enhances the overall diversity and rep resentational richness of the dataset, which we find to be crucial for learning richer high-frequency texture details and fine-grained material variations.

Iterative Material Recognition and Segmentation. The filtered samples are further processed through an iterative visionlanguage-guided segmentation framework. For assets with reliable artist-defined component decomposition or clear mesh connectivity, we additionally construct part-level segmentation renderings to provide explicit structural cues for material recognition. Specifically, each suficiently large and visible mesh component is rendered with a unique region ID, while tiny or barely visible components are discarded to reduce recognition noise. For assets with unreliable or overly fragmented component decomposition, we only use the corresponding multi-view renderings as VLM inputs.

Given the available inputs, i.e., multi-view renderings optionally accompanied by ID-based segmentation maps, we employ a vision-language model (VLM) to perform coarse-grained material recognition. When explicit region IDs are available, they help the VLM associate material predictions with specific object parts rather than relying solely on holistic image-level appearance. Conditioned on the predicted material prompts and multi-view observations, we apply SAM3 [Carion et al. 2025] to generate semantic-aware segmentation masks, yielding an initial material-consistent decomposition.

For assets with ID-based segmentation inputs, we further validate the SAM3-predicted masks by matching them with the VLMassigned component regions using an intersection-over-union (IoU) criterion. Only masks with an IoU greater than 0.7 are retained, ensuring suficient spatial agreement between part-level material recognition and pixel-level segmentation. For assets without reliable ID-based component maps, where segmentation relies primarily on SAM3 predictions, we instead retain only masks whose SAM3 confidence score exceeds 0.7. These threshold-based filtering steps remove unstable or poorly localized masks and improve the reliability of the resulting material-aware decomposition.

To further improve semantic accuracy, we adopt an iterative refinement strategy. The retained segmented regions are fed back into the VLM for region-level material reassessment, and the updated material predictions are used to guide subsequent segmentation. Finally, the VLM performs a quality assessment on each refined material region, evaluating whether the predicted material label is semantically consistent with the corresponding visual appearance. Only regions with a VLM confidence score above 0.9 are preserved in the final decomposition. This closed-loop process progressively improves material recognition, suppresses unreliable segmentations, and enhances the consistency between semantic material labels and pixel-level masks.

Afine Material Projection with Detail Preservation. Instead of directly replacing segmented regions, which may overwrite artistdesigned details and introduce visible discontinuities, we propose an afine material projection strategy. This operation maps each segmented region into a canonical material domain while preserving its original spatial structure and local material layout. To further retain fine-scale appearance details, we explicitly reintroduce highfrequency residuals extracted from the original renderings using Gaussian filtering:

$$
\mathbf { X } _ { \mathrm { r e f i n e d } } = \mathbf { X } _ { \mathrm { a f f n e } } + \lambda _ { \mathrm { h f } } \left( \mathbf { X } - G _ { \sigma } * \mathbf { X } \right) ,\tag{2}
$$

where X denotes the original rendering or material map, $\mathbf { X } _ { \mathrm { a f f i n e } }$ denotes the afine-projected result, $G _ { \sigma }$ is a Gaussian kernel with standard deviation $\sigma ,$ ∗ denotes convolution, and $\lambda _ { \mathrm { h f } }$ controls the strength of high-frequency residual injection. In our implementation, we set $\lambda _ { \mathrm { h f } } = 2 . 0$ . This residual injection preserves fine geometric details and realistic surface variations while avoiding the loss of artist-authored high-frequency texture cues.

Manual Refinement and Consistency Enforcement. Due to the inherent ambiguity in automated parsing—especially under com plex structures and cross-view discrepancies—we further conduct a manual inspection and refinement stage. This step ensures semantic correctness and enforces multi-view consistency across all samples in the dataset.

Overall, through the above filtering and cleaning procedures, the pipeline produces a high-quality PBR dataset with consistent material definitions and rich geometric variations. Although the initial data pool is substantially larger, only approximately 20K samples remain suitable for MR training, providing reliable supervision for stable and physically consistent material learning.

## 4 Multi-View PBR Material Generation

After data cleaning, normalization, and refinement, the initial 260K assets are reduced to approximately 20K high-quality PBR-consistent samples, highlighting the scarcity of reliable assets with complete and reliable PBR annotations under the unified specification. To better exploit this limited supervision, we formulate PBR material generation as a two-stage process: high-resolution multi-view intrinsic decomposition for albedo estimation, followed by albedoconditioned prediction of full PBR properties.

This design also accounts for the diferent spatial requirements of PBR components. Since albedo contains rich texture and color details, it is predicted at $1 0 2 4 ^ { 2 }$ resolution, whereas metallic and roughness maps encode relatively smoother reflectance variations and are predicted at $5 1 2 ^ { 2 }$ resolution. This enables eficient training while preserving high-resolution appearance details and suficient material fidelity.

## 4.1 Illumination-Invariant Albedo Estimation

Dynamic Resolution Permutation-Optimized Packing. Motivated by PacTure[Fei et al. 2025], multi-view aggregation is formulated as a packing problem to improve spatial eficiency and scalability. Existing fixed-resolution heuristics (e.g., MaxRects) are suboptimal for anisotropic view distributions, often leading to low occupancy and ineficient texture utilization.

To address this, a permutation-based optimization is introduced to jointly determine packing order and implicit per-view resolution. Given rectangles $\mathcal R ~ = ~ \{ ( w _ { i } , h _ { i } ) \} _ { i = 1 } ^ { N }$ , permutations $\pi ~ \in ~ S _ { N }$ define sequential insertion under MaxRects with Best Short Side Fit (BSSF), yielding a deterministic layout within a bounding box of area $A _ { \mathrm { b b o x } } ( \pi ) = W ( \pi ) H ( \pi )$

The objective is formulated as maximizing packing eficiency:

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi \in S _ { N } } U ( \pi ) = \arg \operatorname* { m a x } _ { \pi \in S _ { N } } \frac { \sum _ { i = 1 } ^ { N } w _ { i } h _ { i } } { W ( \pi ) H ( \pi ) } .\tag{3}
$$

Since exact optimization over the permutation space $S _ { N }$ is intractable due to its factorial complexity, we approximate $\pi ^ { * }$ using a Genetic Algorithm (GA). In this framework, each individual encodes a permutation �, and its fitness is defined by the packing eficiency $U ( \pi )$ . The population is evolved via rank-based selection, Partially-Mapped Crossover (PMX), and swap mutation operators to explore the combinatorial search space efectively.

To ensure a feasible initialization of the packing process, we estimate the initial bin dimensions $( W _ { 0 } , H _ { 0 } )$ based on the total area of all rectangles:

$$
W _ { 0 } = \left[ \sqrt { \sum _ { i = 1 } ^ { N } w _ { i } h _ { i } } \right] , \quad H _ { 0 } = \left\lceil \frac { \sum _ { i = 1 } ^ { N } w _ { i } h _ { i } } { W _ { 0 } } \right\rceil .\tag{4}
$$

If a given permutation results in an infeasible layout under the current bin constraints, we progressively expand the bounding region to ensure successful placement.

Since the input images are first spatially downsampled by the VAE encoder and subsequently tokenized through patchification, we perform multi-view packing directly in the reduced latent-patch space. Specifically, the packing resolution is defined according to the per-view size after a 16× spatial downsampling. This design aligns the packing layout with the efective token resolution used by the network, thereby avoiding unnecessary interference or misalignment among diferent views after VAE compression and patch partitioning.

![](images/1bf2295458497dc2fb23f249a4ddb0a5137799d3832d3f9172d68702519ef992.jpg)  
Fig. 5. Overview of our conditional difusion framework for multi-view albedo generation. Given a noisy input, a reference image, and geometric guidance, the model progressively denoises the input to synthesize consistent multi-view albedo maps.

Moreover, optimizing the layout in this compact space substantially reduces the computational overhead of the packing procedure. In practice, the complete packing process takes less than 3 seconds per asset, making it eficient enough for large-scale data preprocessing and training.

As illustrated in Fig. 5, given packed multi-view input $\mathbf { x } _ { 0 } ,$ we construct noisy samples via

$$
\begin{array} { r } { \mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { 0 } + \left( 1 - \alpha _ { t } \right) \boldsymbol { \epsilon } , \quad \boldsymbol { \epsilon } \sim N ( 0 , \mathbf { I } ) . } \end{array}\tag{5}
$$

We define the conditioning signal as $\mathbf { c } = \{ \mathbf { c } _ { g e o } , \mathbf { c } _ { r e f } \}$ , where $\mathbf { c } _ { g e o } =$ 0.5 nrm + 0.5 pos. The reference image ${ \bf c } _ { r e f }$ is randomly sampled from a multi-view set $\boldsymbol { \mathcal { V } } _ { : }$ where each view is rendered either as a shaded PBR image or an albedo image with equal probability.

The velocity predictor $\mathbf { f } _ { \theta }$ is trained using a flow matching objective:

$$
\mathcal { L } = \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , \epsilon } \left[ \left\| \mathbf { f } _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } _ { g e o } , \mathbf { c } _ { r e f } ) - ( \epsilon - \mathbf { x } _ { t } ) \right\| _ { 2 } ^ { 2 } \right] .\tag{6}
$$

Classifier-free guidance (CFG) conditioned on geometric cues. During inference, we apply classifier-free guidance (CFG) [Ho and Salimans 2022]to control the influence of geometric conditioning. Given the noisy input $x _ { t } ,$ the reference condition $\mathbf { c } _ { r e f } ,$ and the geometric condition $\mathbf { c } _ { g e o }$ , the guided noise prediction is defined as:

$$
\begin{array} { r l } & { \hat { \epsilon } _ { \mathrm { c f g } } ( x _ { t } , \mathbf { c } _ { g e o } , \mathbf { c } _ { r e f } ) = f _ { \theta } ( x _ { t } , \mathcal { O } , \mathbf { c } _ { r e f } ) } \\ & { \qquad + ~ w \cdot \Big ( f _ { \theta } ( x _ { t } , \mathbf { c } _ { g e o } , \mathbf { c } _ { r e f } ) - f _ { \theta } ( x _ { t } , \mathcal { O } , \mathbf { c } _ { r e f } ) \Big ) , } \end{array}\tag{7}
$$

where � denotes the guidance scale and ∅ represents the dropped geometric condition.

By interpolating between the geometry-conditioned and geometry unconditioned predictions, the guidance term selectively strength ens structures that are consistent with the input geometry. This encourages the generated material maps to better follow surface boundaries and local geometric variations, while preserving the appearance information provided by the reference condition.

Accelerated Inference within Difusion Distillation Framework. To further enhance the practicality of our model, we apply Distribution Matching Distillation (DMD) [Yin et al. 2024] to accelerate inference. Specifically, we distill our multi-step teacher model $f _ { \eta }$ into a few-step student $f _ { \theta }$ by minimizing the reverse KL divergence between their output distributions $\hbar \theta$ and $\begin{array} { r } { p _ { \eta } \colon } \end{array}$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D M D } } = D _ { \mathrm { K L } } \left( p _ { \theta } \parallel \hat { p } _ { \eta } \right) } \\ & { \qquad = \mathbb { E } _ { z , t , \epsilon } \left[ - \left( \log \hat { p } _ { \eta } ( \hat { x } ) - \log \hat { p } _ { \theta } ( \hat { x } ) \right) \right] . } \end{array}\tag{8}
$$

Since the densities are intractable, the gradient is estimated via the diference between the teacher and student score functions:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } _ { z , t , \epsilon } \left[ - \left( s _ { \eta } ( \hat { x } ) - s _ { \theta } ( \hat { x } ) \right) \nabla _ { \boldsymbol { \theta } } \hat { x } \right] .\tag{9}
$$

where $s _ { \eta }$ and �<sub>�</sub> denote the scores ofthe teacher and student distributions, respectively, both obtained from their corresponding difusion models on perturbed samples. With this distillation objective, we transfer the original 30-step CFG-guided sampling process into a compact 6-step inference model, substantially improving generation eficiency while preserving output quality. As a result, the entire model requires only about 40 seconds on average for inference.

## 4.2 Intrinsic Material Estimation

Deterministic Inference within Difusion Framework. As il lustrated in Fig. 6, motivated by Lotus2.0 [He et al. 2025a], we formulate albedo-to-material prediction as a deterministic inference problem. Under our normalized PBR data definition, each albedo is associated with a unique and physically consistent set of material attributes, making the target PBR representation efectively oneto-one conditioned on the given albedo. Therefore, the mapping is primarily constrained by the normalized material definition and photometric consistency, rather than stochastic variation. Empirically, injecting additional noise during training or inference introduces unnecessary ambiguity and destabilizes optimization. We thus adopt a deterministic prediction formulation for stable and consistent PBR reconstruction.

![](images/aae5f1f8f9838506c27fce78f8802f40bce5eed9e3719a786e304ade5de54b80.jpg)  
Fig. 6. Overview of our deterministic difusion framework for multi-view PBR material estimation. Given a packed albedo input $\mathbf { x } _ { r g b }$ and a shaded reference image $\mathbf { x } _ { r e f } ,$ we formulate a conditional difusion process to predict the target material representation $\mathbf { x } _ { m r }$

Accordingly, the problem is formulated as a deterministic conditional mapping within a difusion framework, where stochasticity is confined to the training-time forward process and inference is noisefree. Under the packed multi-view representation, given albedo $\mathbf { x } _ { r g b } { \mathrm { ; } }$ shaded reference ${ \bf x } _ { r e f }$ , and target material $\mathbf { x } _ { m r } .$ , the difusion pair is defined as:

$$
\mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { m r } + \sigma _ { t } \mathbf { x } _ { r g b } .\tag{10}
$$

The reference image ${ \bf x } _ { r e f }$ is encoded using a hybrid feature representation that jointly captures semantic structure and low-level appearance information. Specifically, we employ a pretrained DINO encoder [Oquab et al. 2023] to extract high-level semantic features, which provide robust global cues that are relatively invariant to viewpoint and local appearance variations. In parallel, a VAE encoder [Kingma and Welling 2013] is used to obtain latent appearance features that preserve fine-grained photometric and material-related details:

$$
\mathbf { f } _ { d i n o } = \mathcal { E } _ { d i n o } ( \mathbf { x } _ { r e f } ) , \quad \mathbf { f } _ { v a e } = \mathcal { E } _ { v a e } ( \mathbf { x } _ { r e f } ) .\tag{11}
$$

The resulting semantic and appearance embeddings are concatenated and fused through an MLP to produce the reference conditioning representation:

$$
{ \bf z } _ { r e f } = \mathrm { M L P } \left( \left[ { \bf f } _ { d i n o } , { \bf f } _ { v a e } \right] \right) .\tag{12}
$$

Given the reference condition ${ \bf z } _ { r e f } ,$ , the model is trained to predict the residual between the RGB observation and the target metallic– roughness representation. The training objective is formulated as

an $L _ { 2 }$ reconstruction loss:

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { t , { \bf x } _ { m r } } \left[ \left\| f _ { \theta } ( { \bf x } _ { t } , { \bf z } _ { r e f } ) - ( { \bf x } _ { r g b } - { \bf x } _ { m r } ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{13}
$$

## 5 Implementation Details

Training. We adopt a parameter-eficient fine-tuning strategy by freezing the pretrained backbone and optimizing only the LoRA parameters [Hu et al. 2022]. For both the albedo estimation and PBR material generation stages, the LoRA rank and scaling factor are set to 256. Our model is built upon the FLUX.2 backbone [Labs 2025] and trained on 32 NVIDIA A800 GPUs with a total batch size of 64. We use the AdamW optimizer with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ , a learning rate of $5 \times 1 0 ^ { - 5 }$ , and a linear warm-up schedule over the first 2,000 optimization steps.

For timestep sampling, we adopt a logit-normal sampling strategy to balance training across diferent noise levels. In all experiments, the mean and standard deviation of the logit-normal distribution are set to $\mu = - 1$ and $\sigma = 1$ , respectively. This setting is used consistently for packed multi-view inputs during training. To improve training stability and support high-resolution albedo estimation, we employ a hierarchical resolution training strategy for the albedo stage. Specifically, the albedo model is first trained at 512 × 512 resolution for 5,000 steps to learn stable global appearance, coarse spatial layout, and cross-view consistency. Starting from this checkpoint, we further fine-tune it at 1024 × 1024 resolution for an additional 10,000 steps to enhance high-frequency texture details, boundary fidelity, and fine-scale appearance variations.

In contrast, the PBR material generation stage focuses on metallic– roughness (MR) prediction, whose spatial variations are generally smoother and do not require the same resolution as albedo. Therefore, the MR model is trained only at 512 × 512 resolution for 15,000 steps, reducing computational cost while preserving suficient material fidelity.

## 6 Experiments

To systematically evaluate the performance of our framework, we conduct a comprehensive set of experiments focusing on both the quality and eficiency of PBR generation. We perform rigorous quantitative and qualitative comparisons against state-of-the-art methods in PBR material estimation and generation.

## 6.1 Qualitative Results

Visual Comparisons against Existing Methods. We evaluate single-view texture generation fidelity under arbitrary novel viewpoints. To this end, we construct a challenging benchmark from Objaverse [Deitke et al. 2023], consisting of 200 high-quality 3D assets with diverse geometries and complex surface structures. For each asset, we render six random viewpoints for both quantitative and qualitative evaluation of the generated texture and material maps.

As shown in Fig. 7, we compare our method with representative state-of-the-art approaches. MvAdapter [Huang et al. 2025b] often fails to disentangle intrinsic albedo from illumination, resulting in residual shading artifacts. Unitex [Liang et al. 2025] suppresses

MV-Adapter

UniTex

Hunyuan3d-2.1

LumiTex

Ours

![](images/c179ed77f08afced299bebb891f62f9643b2a61395cab6a3d486bc17695fe97c.jpg)  
Fig. 7. Qualitative comparison of texture and PBR material generation across diferent methods. Each example is arranged in two rows: the first row shows the rendered results under a fixed illumination seting, while the second row presents the predicted material maps, including the MR map, i.e., metallic and roughness channels, and the generated albedo texture. Compared with existing methods, our approach produces sharper and more detailed albedo textures, while also yielding cleaner, more spatially coherent, and semantically consistent MR predictions across diverse samples.

lighting efects to some extent but does not explicitly recover full PBR material properties. Hunyuan3d-2.1 [Hunyuan3D et al. 2025] predicts both albedo and material maps, yet its material predictions are often inconsistent across geometrically or semantically similar regions. Although LumiTex [Bao et al. 2025] produces visually plausible textures, its material estimates remain unstable and may lack cross-view physical consistency. Moreover, these methods tend to lose fine-grained albedo details compared with the input image, leading to reduced local texture fidelity.

![](images/e29a7a210144ec1c6ee7799226d9d664a3927350ff5d82cf11914dd9e3f08332.jpg)  
Fig. 8. Visualization results of PBR materials and texture maps generated for two given meshes under image guidance. The rendered results under diverse lighting environments demonstrate that the model efectively adapts to varying illumination conditions. In particular, metallic regions exhibit physically plausible reflections, accurately capturing environmental lighting and producing realistic appearance.

In contrast, our method recovers albedo maps with richer details and stronger alignment with the input appearance, while efectively reducing illumination contamination. It also generates more stable and spatially consistent PBR material maps whose properties are better correlated with the recovered albedo and remain coherent across viewpoints, demonstrating improved texture fidelity, crossview consistency, and physical plausibility.

Visual Results under Varying Illumination Conditions. As illustrated in Fig. 8, we present several representative cases rendered under a diverse set of environmental lighting conditions. The results show that the estimated PBR material properties efectively capture intrinsic surface characteristics and exhibit appropriate responses to illumination changes. In particular, the rendered appearances vary consistently with the incident lighting, demonstrating coherent shading, specular behavior, and reflectance patterns across diferent environments. This indicates that our method produces physically plausible and lighting-aware material representations, enabling stable and realistic appearance synthesis under novel illumination conditions.

## 6.2 Quantitative Analysis

To quantitatively evaluate the texture fidelity ofthe generated assets, we adopt a comprehensive set of metrics covering distribution-level realism, perceptual quality, and semantic alignment. As shown in Tab. 2, our method consistently outperforms existing approaches across all evaluated criteria. Specifically, CMMD [Jayasumana et al.

2024] and FID [Heusel et al. 2017] evaluate the distributional fidelity of generated textures, while LPIPS [Zhang et al. 2018] measures perceptual similarity to reference appearances. CLIP Score [Hessel et al. 2021] and CLIP-FID [Parmar et al. 2022] further assess semantic alignment and feature-space consistency. Overall, these metrics requires characterize the visual and semantic quality of the generated texture appearance. These results demonstrate that our method produces visually faithful textures that are well aligned with the input guidance.

Furthermore, we conduct a user study to compare our method with existing approaches, as shown in Fig. 10. The results further confirm the efectiveness of our framework, where our method achieves state-of-the-art performance in terms of visual quality, material realism, and consistency with the input reference.

## 6.3 Ablation Study

Efect ofclassifier-free guidance (CFG) conditioned on geometric cues. To evaluate the efectiveness of classifier-free guid ance (CFG) conditioned on geometric cues, we train and compare several CFG variants under diferent conditioning strategies, as il lustrated in Fig. 9. We first observe that conventional text-based CFG provides negligible improvement when strong image conditioning is present, with minimal changes in the generated results. This suggests that the model tends to prioritize image tokens over textual guidance during cross-attention, leading to an implicit sup pression of text-conditioned signals. In contrast, CFG based on reference images does not lead to consistent quality improvements and may introduce visual artifacts, indicating potential conflicts between image-level guidance and the primary conditioning signal.

![](images/265c834a7675059afbf8164ba0ef1004c7f1cfaf8be0c771ab0197046b368207.jpg)  
Fig. 9. Qualitative comparison of texture generation using geometry-based classifier-free guidance (CFG) versus alternative CFG strategies. The results show that geometry-guided CFG produces sharper textures that more closely align with the conditioning image, leading to improved structural fidelity and overal visual consistency.

Table 2. Quantitative Comparison with State-of-the-Art Methods on PBR Texture Generation Quality and Eficiency.
<table><tr><td>Method</td><td>FID↓</td><td>CMMD ↓</td><td>LPIPS ↓</td><td>CLIP-I↑</td><td>CLIP-FID ↓</td><td>Inference Time (s) ↓</td><td>Albedo Resolution ↑</td><td>MR Resolution ↑</td></tr><tr><td>MvAdapter</td><td>165.49</td><td>2.2235</td><td>0.2383</td><td>0.9097</td><td>22.911</td><td>36</td><td>768</td><td>None</td></tr><tr><td>UniTex</td><td>161.27</td><td>2.1898</td><td>0.2174</td><td>0.9052</td><td>22.623</td><td>220</td><td>512</td><td>None</td></tr><tr><td>Hunyuan3d-2.1</td><td>154.88</td><td>2.2625</td><td>0.2167</td><td>0.9132</td><td>20.132</td><td>48</td><td>768</td><td>768</td></tr><tr><td>LumiTex</td><td>144.61</td><td>1.8677</td><td>0.1992</td><td>0.9215</td><td>16.182</td><td>110+110</td><td>512</td><td>512</td></tr><tr><td>Ours</td><td>121.25</td><td>1.4652</td><td>0.1612</td><td>0.9411</td><td>13.358</td><td>40+12</td><td>1024</td><td>512</td></tr></table>

By comparison, CFG conditioned on geometric cues consistently improves structural fidelity and fine-grained detail consistency. This indicates that geometric information provides a more stable and complementary conditioning signal, efectively constraining the generation process and improving alignment between the synthesized appearance and the underlying geometry. Furthermore, as shown in Tab. 3, we quantitatively evaluate the efect of the pro posed geometry-conditioned CFG by comparing model performance with and without this guidance strategy. The results demonstrate that enabling geometry-conditioned CFG substantially improves performance across evaluation metrics, further confirming its effectiveness in enhancing generation quality and geometry-aware consistency.

Table 3. Ablation study on the efect of classifier-free guidance (CFG) in our framework. Enabling CFG significantly improves generation quality and semantic consistency across all evaluation metrics.
<table><tr><td>Method</td><td>Setting</td><td>FID ↓</td><td>CMMD↓</td><td>LPIPS ↓</td><td>CLIP-I↑</td><td>CLIP-FID ↓</td></tr><tr><td rowspan="2">Ours</td><td>w/o CFG</td><td>128.55</td><td>1.5439</td><td>0.1688</td><td>0.9342</td><td>15.172</td></tr><tr><td>w/ CFG</td><td>121.25</td><td>1.4652</td><td>0.1612</td><td>0.9411</td><td>13.358</td></tr></table>

Efect ofPBR Material Consistency on Inference Stability. To evaluate the impact of unified PBR material specifications, we conduct a controlled ablation study using the same curated dataset under two settings: training with the original artist-defined material parameters and training with our standardized material specification. As shown in Fig. 11, the model trained with unified materials produces more stable and consistent MR predictions across regions sharing the same semantic material, whereas the model trained on artist-defined parameters exhibits visible inconsistencies caused by non-uniform material definitions.

![](images/4a95ff5e46e5bc347fc01b70b12ff614a53502dc65e8d363f924e5aded3aa102.jpg)  
Fig. 10. User-study results comparing perceptual quality and referenceimage consistency across diferent methods for multi-view image generation. The bar chart reports the average scores for each method, with higher values indicating beter perceived quality and consistency with the reference image.

![](images/ec9634a27f5f4ef42a08b3789f783917b960db45dae81683546858003911dac4.jpg)  
Ref  
Original-trained  
Curated-trained  
Fig. 11. Visual comparison between models trained on original artist-defined PBR data and our unified normalized PBR data. The first column shows the reference image, the second column presents results from the model trained on the original artist-defined PBR data, and the third column shows results from the model trained on our unified normalized PBR data.

Table 4. Ablation study on the efect of PBR material standardization. The standardized material specification improves prediction consistency under semantic-preserving appearance variations.
<table><tr><td>Training Data</td><td>PBR Consistency MSE ↓</td></tr><tr><td>Artist-defined Materials</td><td>0.0513</td></tr><tr><td>Standardized Materials</td><td>0.0258</td></tr></table>

Because our standardization modifies the original PBR annotations, evaluation against either the original or standardized labels would introduce bias. We therefore conduct two complementary user studies to assess perceptual material quality. First, we perform a controlled ablation on our model, varying only whether PBR standardization is applied. Participants compare paired mesh renderings of the predicted MR attributes in terms of material consistency and physical plausibility. As shown in Fig. 13 (left), our standardized variant achieves a 73.2% win rate, with 14.1% ties and 12.7% losses. Second, we fine-tune Hunyuan3D-V2.1 [Hunyuan3D et al. 2025] using either the raw collected data or our curated data and evaluate the results on a five-point scale. As shown in Fig. 13 (right), fine-tuning on our curated data achieves the highest mean rating of 3.88, compared with 3.04 for the pretrained model and 2.11 for fine-tuning on the raw data. These results demonstrate that our unified PBR specification reduces supervision ambiguity and improves material consistency and physical plausibility across diferent model architectures.

Evaluation ofPBR Stability underSemantic-Preserving Style Variations. Beyond improving consistency across regions with identical semantic materials, we further evaluate whether our standardized material representation improves the physical stability of predictions under appearance variations. Ideally, inputs depicting the same object with diferent color distributions or visual styles should produce similar PBR properties, since their underlying physical materials remain unchanged.

To evaluate this property, we generate multiple semantic-preserving style variants of the same input image using GPT-based image editing. We then predict PBR maps for each variant and measure the consistency between predictions using the mean squared error (MSE).

As show in Tab. 4, the curated dataset achieves significantly lower prediction variance, indicating that our normalization strategy improves PBR stability and reduces ambiguity introduced by inconsistent material annotations. Importantly, our method does not enforce identical PBR values for all instances within the same semantic category. Instead, it regularizes inconsistent material definitions while preserving spatial variations and high-frequency appearance details.

## 6.4 Additional Applications and Analysis

Texture Stylization from Arbitrary Images. Our model supports stylized texture generation from arbitrary input images without requiring domain-specific constraints or curated references. Given an input image with diverse appearance characteristics, the model extracts high-level visual cues and translates them into coherent texture representations suitable for 3D assets. This flexibility enables a wide range of applications, including rapid texture prototyping, image-guided mesh stylization, and content creation from in-thewild images. In particular, the generated textures preserve the overall stylistic attributes of the input while maintaining clean appearance, coherent surface details, and stable visual quality under varying illumination conditions. As shown in Fig. 12, our method produces high-quality stylized textures that can be directly applied to 3D meshes, resulting in visually realistic renderings with strong detail fidelity, harmonious style transfer, and consistent appearance across diferent scenarios.

![](images/9f59167e1cd2325a729fd01877c88ae46288dfb5cd3fecae680fe1ff849f6114.jpg)  
Fig. 12. Visualization results of arbitrary image-guided texture generation and mesh stylization. The leftmost images show the reference guidance images, while the rendered results on the right are obtained by generating textures conditioned on each reference image and applying them back onto the same underlying mesh. Despite the substantial appearance diferences among the guidance images, our method produces coherent and harmonious stylized textures that are well integrated with the shared geometry, demonstrating strong image-guided stylization capability.

![](images/ff1394e8fa944a16397be54f1d6ec60ddd4348a0171305a02878841c6c3c10e7.jpg)

![](images/cc90fc70745c110f27a0daaf909b5243fa09e3841c74a30452d4d234f987987a.jpg)  
Fig. 13. User study on perceptual PBR material quality. Left: Pairwise preferences between our model trained on the curated, standardized annotations and that trained on the original annotations, where win, tie, and loss are defined from the perspective of the former. Right: Mean five-point ratings of Hunyuan3D-V2.1 under the pretrained seting and after fine-tuning on either the raw collected data or our curated data.

Text and Logo Fidelity under Reference Guidance. In addition to overall appearance and material consistency, preserving distinctive local patterns such as text, logos, and graphic symbols is also important for reference-guided texture generation. These elements often contain high-frequency structures and precise semantic layouts, making them challenging to reproduce faithfully on complex 3D surfaces. As shown in Fig. 14, our method efectively transfers such reference-specific details onto the target mesh while maintaining clear boundaries, recognizable character shapes, and coherent spatial alignment. The results demonstrate that our model can preserve fine-grained textual and logo-like patterns under reference guidance, further improving the fidelity and controllability of the generated textures.

Extended Qualitative Results. As shown in Fig. 15, we present additional results on a diverse set of inputs, where the generated texture and PBR material maps are applied back onto the corresponding meshes for rendering under novel illumination conditions. Our method consistently produces high-quality textures and physically plausible material properties. The generated albedo maps are clean and largely free of shading artifacts, while the recovered reflectance parameters exhibit strong spatial coherence.

## 7 Limitations and Future Work

Although our data construction pipeline improves the consistency and reliability of PBR material annotations, it still involves limited manual intervention. Specifically, the final validation stage relies on human inspection to identify ambiguous, low-quality, or visually implausible samples. This process helps maintain dataset quality but may introduce subjective biases, as diferent annotators may have varying judgments regarding material plausibility, texture fidelity, and visual consistency. Moreover, manual screening increases the cost of dataset construction and may afect scalability to larger asset collections.

![](images/32bb8e78454c644d4b485e55933188c831fed2ebb64ebecbdc2ccd6e0d037ce8.jpg)  
Fig. 14. Visualization of text and logo fidelity under reference guidance. For each example, the leftmost image denotes the reference input containing distinctive textual or logo-like paterns. The two rendered results on the right show the textured mesh after back-projecting the generated maps, where the left rendering presents the frontal view and the right rendering presents the back view. The results demonstrate that our method can preserve fine-grained text and log structures from the reference image while maintaining coherent appearance across diferent viewpoints.

Furthermore, our current approach focuses on achieving relative physical consistency rather than strict physical correctness. Since the material definitions are derived from artist-created assets, the resulting PBR distributions inevitably contain certain artistic approximations and may deviate from the true physical properties of real-world materials. While our standardization strategy reduces annotation ambiguity and improves consistency under a unified rendering protocol, it does not fully recover physically measured material properties. In future work, we plan to incorporate physically acquired material measurements, such as captured SVBRDFs from dedicated acquisition devices, to develop more physically accurate and grounded material representations.

## 8 Conclusion

In this work, we investigate the fundamental challenges of learning physically based material properties from image observations, with particular emphasis on the ambiguities arising from inconsistent illumination conditions and non-uniform PBR material definitions. We show that variations in environment lighting and artist-defined material parameterizations can substantially entangle intrinsic material attributes with view- and illumination-dependent shading efects, resulting in unstable supervision signals and inconsistent material predictions. To mitigate these issues, we establish a controlled data generation pipeline with consistent illumination and introduce a unified specification for PBR material properties. Built upon this normalized PBR representation informed by empirical material priors, we further train a state-of-the-art generative model capable of synthesizing high-quality texture maps together with coherent PBR material properties. Through extensive experiments, including ablation studies on material consistency, we demonstrate that our data-centric design efectively reduces ambiguity in the learning process and improves the stability, coherence, and physical plausibility of model inference, particularly across spatially separated regions that share the same underlying material.

Overall, our findings underscore the critical role of data consistency in disentangling intrinsic material properties from image observations. Beyond achieving strong generation performance, this study provides practical insights for constructing robust, scal able, and more consistent and empirically grounded PBR material generation systems.

## 9 Acknowledgments

This work was supported by National Natural Science Foundation of China (62472213), Gusu Innovation & Entrepreneurship Leading Talents Program (ZXL2024361), and computing resources from DreamTech.

![](images/a303ad5a5c1813f45ea29367b38ff0b34d73fcd3db805aac600a07d7a5e4651a.jpg)  
Fig. 15. Additional qualitative results on diverse inputs. Our method demonstrates strong generalization across a wide range of mesh geometries, styles, and input variations, consistently producing high-quality textures along with semantically accurate and physically plausible PBR materials.

## References

Jingzhi Bao, Hongze Chen, Lingting Zhu, Chenyu Liu, Runze Zhang, Keyang Luo, Zeyu Hu, Weikai Chen, Yingda Yin, Xin Wang, et al. 2025. LumiTex: Towards High-Fidelity PBR Texture Generation with Illumination Context. arXiv preprint arXiv:2511.19437 (2025).

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Di dac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025).

Chia-Hao Chen, Zi-Xin Zou, Yan-Pei Cao, Ze Yuan, Guan Luo, Xiaojuan Qi, Ding Liang, Song-Hai Zhang, and Yuan-Chen Guo. 2025. LaFiTe: A Generative Latent Field for 3D Native Texturing. arXiv preprint arXiv:2512.04786 (2025).

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023. Fantasia3D: Disentan gling Geometry and Appearance for High-quality Text-to-3D Content Creation. arXiv:2303.13873 [cs.CV] htps://arxiv.org/abs/2303.13873

Xi Chen, Sida Peng, Dongchen Yang, Yuan Liu, Bowen Pan, Chengfei Lv, and Xiaowei Zhou. 2024. Intrinsicanything: Learning difusion priors for inverse rendering under unknown illumination. In European Conference on Computer Vision. Springer, 450–467.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Obja verse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 13142–13153.

Fan Fei, Jiajun Tang, Fei-Peng Tian, Boxin Shi, and Ping Tan. 2025. Pacture: Eficient pbr texture generation on packed views with visual autoregressive models. arXiv preprint arXiv:2505.22394 (2025).

Jon Hasselgren, Milos Hasan, Zheng Zeng, and Jacob Munkberg. 2026. VideoMatGen: PBR materials through joint generative modeling. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2440–2450.

Jing He, Haodong Li, Mingzhi Sheng, and Ying-Cong Chen. 2025a. Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model. arXiv preprint arXiv:2512.01030 (2025).

Zebin He, Mingxin Yang, Shuhui Yang, Yixuan Tang, Tao Wang, Kaihao Zhang, Guanying Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, et al. 2025b. Materialmvp: Illumination-invariant material generation via multi-view pbr difusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 26294–26305.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing. 7514–7528.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017).

Jonathan Ho and Tim Salimans. 2022. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022).

Yijia Hong, Yuan-Chen Guo, Ran Yi, Yulong Chen, Yan-Pei Cao, and Lizhuang Ma. 2025. Supermat: Physically consistent pbr material estimation at interactive rates. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 25083– 25093.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. Iclr 1, 2 (2022), 3.

Yifeng Huang, Zhang Chen, Yi Xu, Minh Hoai, and Zhong Li. 2025a. DualMat: PBR Material Estimation via Coherent Dual-Path Difusion. arXiv:2508.05060 [cs.CV] htps://arxiv.org/abs/2508.05060

Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. 2025b. Mv-adapter: Multi-view consistent image generation made easy. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 16377–16387.

Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. 2025. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442 (2025).

Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. 2024. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9307–9315.

Tanghui Jia, Dongyu Yan, Dehao Hao, Yang Li, Kaiyi Zhang, Xianyi He, Lanjiong Li, Jinnan Chen, Lutao Jiang, Qishen Yin, Long Quan, Ying-Cong Chen, and Li Yuan. 2025. UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement. arxiv preprint arXiv:2512.21185 (2025).

Xiaojie Jin, Bowen Zhang, Weibo Gong, Kai Xu, Xueqing Deng, Peng Wang, Zhao Zhang, Xiaohui Shen, and Jiashi Feng. 2024. Mv-adapter: Multimodal video transfer learning for video text retrieval. In Proceedings ofthe ieee/cvfconference on computer vision and pattern recognition. 27144–27153.

Seungwook Kim, Yichun Shi, Kejie Li, Minsu Cho, and Peng Wang. 2025. RapidMV: Leveraging Spatio-Angular Representations for Eficient and Consistent Text-to-Multi-View Synthesis. arXiv preprint arXiv:2509.24410 (2025)

Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013).

Peter Kocsis, Lukas Höllein, and Matthias Nießner. 2025. Intrinsix: High-quality pbr generation using image priors. arXiv preprint arXiv:2504.01008 (2025).

Black Forest Labs. 2024. FLUX. htps://github.com/black-forest-labs/flux.

Black Forest Labs. 2025. FLUX.2: Frontier Visual Intelligence. htps://bfl.ai/blog/flux-2.

Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. 2025a. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504 (2025).

Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Xin Yang, Xin Huang, Jingwei Huang, Xiangyu Yue, and Chunchao Guo. 2025b. NaTex: Seamless Texture Generation as Latent Color Difusion. arXiv preprint arXiv:2511.16317 (2025).

Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, et al. 2025. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747 (2025).

Zhibing Li, Tong Wu, Jing Tan, Mengchen Zhang, Jiaqi Wang, and Dahua Lin. 2024. Idarb: Intrinsic decomposition for arbitrary number ofinput views and illuminations. arXiv preprint arXiv:2412.12083 (2024).

Yixun Liang, Kunming Luo, Xiao Chen, Rui Chen, Hongyu Yan, Weiyu Li, Jiarui Liu, and Ping Tan. 2025. Unitex: Universal high fidelity generative texturing for 3d shapes. arXiv preprint arXiv:2505.23253 (2025)

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3D: High-Resolution Text-to-3D Content Creation. arXiv:2211.10440 [cs.CV] htps://arxiv. org/abs/2211.10440

Chenyu Liu, Hongze Chen, Jingzhi Bao, Lingting Zhu, Runze Zhang, Weikai Chen, Zeyu Hu, Yingda Yin, Keyang Luo, and Xin Wang. 2025. CaliTex: Geometry-Calibrated Attention for View-Coherent 3D Texture Generation. arXiv preprint arXiv:2511.21309 (2025).

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023).

Yuxin Liu, Minshan Xie, Hanyuan Liu, and Tien-Tsin Wong. 2024. Text-guided texturing by synchronized multi-view difusion. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain difusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9970–9980.

Di Luo, Shuhui Yang, Mingxin Yang, Jiawei Lu, Yixuan Tang, Xintong Han, Zhuo Chen, Beibei Wang, and Chunchao Guo. 2025. MatPedia: A Universal Generative Foundation for High-Fidelity Material Synthesis. arXiv preprint arXiv:2511.16957 (2025).

Jacob Munkberg, Zian Wang, Ruofan Liang, Tianchang Shen, and Jon Hasselgren. 2025. Videomat: Extracting pbr materials from video difusion models. In Computer Graphics Forum, Vol. 44. Wiley Online Library, e70180

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. 2022. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 11410–11420.

William Peebles and Saining Xie. 2023. Scalable difusion models with transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4195–4205.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent difusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023).

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022).

Nithin Raghavan, Krishna Mullia, Alexander Trevithick, Fujun Luan, Miloš Hašan, and Ravi Ramamoorthi. 2025. Generative neural materials. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Rad ford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International conference on machine learning. Pmlr, 8821–8831.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Mingqi Shao, Feng Xiong, Zhaoxu Sun, and Mu Xu. 2025. Mvpainter: Accurate and detailed 3d texture generation via multi-view difusion with geometric control. arXiv

preprint arXiv:2505.12635 (2025).

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023. Mvdream: Multi-view difusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023).

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023. Dreamgaussian: Generative gaussian splatting for eficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023).

Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Furukawa, and Rakesh Ranjan. 2024. Mvdifusion++: A dense high-resolution multi-view difusion model for single or sparse-view 3d object reconstruction. In European Conference on Computer Vision. Springer, 175–191

Giuseppe Vecchio, Renato Sortino, Simone Palazzo, and Concetto Spampinato. 2024. Matfuse: controllable material generation with difusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 4429–4438.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. 2024. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video difusion. In European Conference on Computer Vision. Springer, 439–457.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. arXiv:2305.16213 [cs.LG] htps://arxiv.org/abs/2305. 16213

Xiaokang Wei, Bowen Zhang, Xianghui Yang, Yuxuan Wang, Chunchao Guo, Xi Zhao, and Yan Luximon. 2026. Pbr3dgen: A vlm-guided mesh generation with high-quality pbr texture. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 10575–10583.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. 2024. Direct3d: Scalable image-to-3d generation via 3d latent difusion transformer. Advances in Neural Information Processing Systems 37 (2024), 121859– 121881.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. 2025. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412 (2025).

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. 2025a. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692 (2025).

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025b. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21469–21480.

Bowen Xue, Claudio Guarnera, Shuang Zhao, and Zahra Montazeri. 2024. Reflectance Fusion: Difusion-based text to SVBRDF Generation. In Eurographics Symposium on Rendering. Eurographics Association.

Bowen Xue, Saeed Hadadan, Zheng Zeng, Fabrice Rousselle, Zahra Montazeri, and Milos Hasan. 2026. VideoNeuMat: Neural Material Extraction from Generative Video Models. arXiv:2602.07272 [cs.CV] htps://arxiv.org/abs/2602.07272

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. 2024. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6613–6623.

Zhi Ying, Boxiang Rong, Jingyu Wang, and Maoyuan Xu. 2025. Chord: Chain of rendering decomposition for pbr material estimation from generated texture images. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–11.

Xin Yu, Ze Yuan, Yuan-Chen Guo, Ying-Tian Liu, Jianhui Liu, Yangguang Li, Yan-Pei Cao, Ding Liang, and Xiaojuan Qi. 2024. Texgen: a generative difusion model for mesh textures. ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–14.

Ze Yuan, Xin Yu, Yangtian Sun, Yuan-Chen Guo, Yan-Pei Cao, Ding Liang, and Xiaojuan Qi. 2025. SeqTex: Generate Mesh Textures in Video Sequence. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–12.

Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. 2024a. Paint3d: Paint anything 3d with lighting-less texture difusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 4252–4262.

Yifei Zeng, Yajie Bao, Jiachen Qian, Shuang Wu, Youtian Lin, Hao Zhu, Buyu Li, Feihu Zhang, Xun Cao, and Yao Yao. 2025. TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond. arXiv preprint arXiv:2512.02993 (2025).

Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick Hold-Geofroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and Miloš Hašan. 2024b. RGB↔X: Image decomposition and synthesis using material- and lighting-aware difusion models. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH ’24). ACM, 1–11. doi:10.1145/3641519.3657445

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Yibo Zhang, Li Zhang, Rui Ma, and Nan Cao. 2025. Texverse: A universe of 3d objects with high-resolution textures. arXiv preprint arXiv:2508.10868 (2025).

Lingting Zhu, Jingrui Ye, Runze Zhang, Zeyu Hu, Yingda Yin, Lanjiong Li, Jinnan Chen, Shengju Qian, Xin Wang, Qingmin Liao, and Lequan Yu. 2025. MuMA: 3D PBR Texturing via Multi-Channel Multi-View Generation and Agentic Post-Processing. arXiv:2503.18461 [cs.CV] htps://arxiv.org/abs/2503.18461

Shenhao Zhu, Lingteng Qiu, Xiaodong Gu, Zhengyi Zhao, Chao Xu, Yuxiao He, Zhe Li, Xiaoguang Han, Yao Yao, Xun Cao, et al. 2024. MCMat: Multiview-consistent and physically accurate pbr material generation. arXiv preprint arXiv:2412.14148 (2024).

Qi Zuo, Xiaodong Gu, Lingteng Qiu, Yuan Dong, Weihao Yuan, Rui Peng, Siyu Zhu, Liefeng Bo, Zilong Dong, Qixing Huang, et al. 2024. Videomv: Consistent multi-view generation based on large video generative model. (2024).