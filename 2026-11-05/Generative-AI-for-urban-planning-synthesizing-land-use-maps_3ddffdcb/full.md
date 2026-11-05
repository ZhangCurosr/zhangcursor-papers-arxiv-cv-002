# Generative AI for urban planning: synthesizing land-use maps with satellite imagery via DualControlNet

Qingqi Song<sup>1</sup>, Yuzhou Chen<sup>1</sup>, Lingqian Hu<sup>2</sup>, Shenhao Wang<sup>1,\*</sup>

November 5, 2026

<sup>1</sup> Department of Urban and Regional Planning, University of Florida <sup>2</sup> College of Architecture, Texas A&M University

## Abstract

Various land-use models have been developed to characterize land-use patterns and facilitate land-use scenario planning. However, existing land-use models are largely predictive, forecasting land-use transitions based on numeric data, lacking the capability to synthesize non-existing land-use maps from simple text prompts. Meanwhile, generative AI is increasingly applied to urban systems, but existing generative approaches remain unimodal, weakly controllable, and evaluated mostly by generic image quality rather than domain-specific land-use metrics. To address these gaps, we introduce DualControlNet, a generative land-use model that jointly synthesizes high-resolution land-use maps and satellite imagery, simultaneously addressing the challenges in multimodality, controllability, and verification. Empirically, we train DualControl-Net on more than 40,000 paired tiles from Orlando and Chicago and evaluate it using land-use, spatial consistency, and image quality metrics. Relative to unimodal ControlNet baselines, DualControlNet improves the consistency in land-use diversity by 62.8% and in building coverage ratio by 16.3%, more than doubles cross-modal spatial consistency, and improves visual quality by around three times. Through scenario-based generative experiments, we demonstrate that DualControlNet can support sustainable land-use development through multidimensional controls. It creates land-use maps that successfully increase land-use density and diversity, and respond to textual land-use prompts, such as “high-density development”, thus assisting land-use scenario exploration through intuitive textual and imagery formats. Overall, this work advances land-use modeling by introducing a generative framework that is multimodal, controllable, and verifiable using land-use metrics. It further demonstrates the potential of generative AI for lowering the technical barrier to land-use scenario planning and enabling more intuitive and participatory exploration of urban futures.

\*Corresponding author: Shenhao Wang; shenhaowang@ufl.edu

Acknowledgments: The corresponding author acknowledges the support from Research Opportunity Seed Grant (ROSF-2023) and Artificial Intelligence and Informatics Research Institute Seed Grant (AIIRI-2026) at the University of Florida.

## 1 Introduction

Land-use patterns crucially shape urban sustainability, health, equity, and safety (Cervero and Kockelman, 1997; Ewing and Cervero, 2010; Lyu et al., 2025; Weng et al., 2019; Gu et al., 2022; Kim and Hipp, 2021). To characterize land-use patterns and inform planning practices, researchers have developed a rich set of land-use metrics, including land-use diversity, density, and building coverage, together with land-use models that forecast how urban land use evolves over time (Boeing, 2017; He et al., 2023; Lopane et al., 2023). These land-use models, ranging from cellular automata to statistical methods, enable researchers to predict land-use transitions using numeric and historical land-use data (Cruz-Bello et al., 2024; Guan et al., 2024; Cai et al., 2026). However, they share a common limitation. As they are largely predictive models, they forecast incremental transitions from existing land-use categories, but cannot synthesize land-use patterns that do not yet exist. This limitation matters for scenario planning, a core planning practice in which planners envision and compare alternative urban futures before committing to a specific plan. Such practice requires the capacity to generate new land-use maps directly from planning objectives, rather than only extrapolating from the past.

Recently, generative AI introduces an opportunity of directly synthesizing high-fidelity spatial representations from abstract inputs such as textual prompts (Ho et al., 2020; Rombach et al., 2022). However, the planning research community still lacks a verifiable, controllable, and multimodal generative framework for land-use scenario planning. First, most generative studies remain unimodal, synthesizing a single urban layer such as satellite imagery or a building footprint map in isolation (Wu and Biljecki, 2022; Dedring and Rienow, 2024; Wang et al., 2025a; Sun et al., 2026; He et al., 2026). This is a fundamental limitation because urban systems are inherently multimodal<sup>1</sup>: land use, urban form, road networks, and other layers align spatially and should be generated jointly to preserve their cross-modal consistency. Second, existing models ofer weak domain-specific con trollability, providing limited mechanisms to steer generation using interpretable land-use metrics such as land-use diversity or building density (Lowry and Lowry, 2014; Abou Jaoude et al., 2022). Yet such control is essential for scenario planning, where planners specify explicit development objectives and expect matching outputs. Third, generated outputs are typically evaluated only by generic image realism metrics, which reveal whether an image looks plausible but not whether it is geographically valid against targeted land-use metrics (Castro Lopes et al., 2025; Ye et al., 2022a; Abdollahi et al., 2020). Therefore, it is crucial to envision a generative land-use model that synthesizes land use jointly with other urban modalities, responds to interpretable planning objectives, and is evaluated against domain-specific land-use metrics.

To address these gaps, we introduce DualControlNet, a generative land-use model that jointly synthesizes land-use maps and satellite imagery, controls generation with interpretable land-use metrics, and verifies the generated outputs against the same metrics. Technically, we assemble more than 40,000 observations from Orlando and Chicago, each aligning a satellite image, a landuse map, and a control map of major roads and water bodies. We compute land-use metrics such as building coverage ratio, land-use diversity, and average land-use area from every observation, and translate them into structured language prompts. We then extend the ControlNet architecture to a dual-modality setting and add a geospatial consistency loss that regularizes the boundary geometry shared by the two generated modalities. Such a research design allows us to address three research questions: first, whether the generated land-use maps satisfy the land-use metrics specified in the prompts; second, whether the generated land-use map and satellite imagery remain spatially consistent with each other; lastly, whether the generated satellite imagery is visually realistic for

credible planning communication.

After the technical evaluation, we further use DualControlNet for generative scenario planning of sustainable land development. We start from uni-dimensional control, increasing the target building coverage ratio and examining whether the generated land-use map achieves the intended development intensity. We then move to multidimensional control, jointly raising building coverage and the number of land-use categories at underdeveloped sites. Such generative scenario planning is highly relevant to the cycle in planning practice: a planner states development objectives in plain language, receives a spatially coherent and visually inspectable land-use map within seconds, and then verifies that scenario against the very initial objectives.

This study is organized as follows. Section 2 reviews land-use metrics and modeling, generative AI for urban systems, and scenario planning, and identifies the research gaps. Section 3 describes the multimodal data and introduces DualControlNet with its geospatial consistency loss. Section 4 presents the experimental design and the three families of evaluation metrics. Section 5 reports the model benchmarks and the generative scenario planning experiments. Section 6 discusses the implications for planning practice and the limitations. Section 7 concludes our study.

## 2 Literature Review

## 2.1 Land-use metrics and modeling

Researchers have designed a wide range of land-use metrics to characterize land-use patterns and their implications for urban sustainability, health, equity, and safety. The land-use metrics include land-use diversity, which is typically measured by entropy-based indices and captures the functional heterogeneity of an area. They also include density, building coverage ratio, or average building size, which capture the intensity and compactness of development. The classic density–diversity–design (3D) framework established these as core dimensions of land use (Cervero and Kockelman, 1997), and subsequent work extended them to more general urban form metrics, including street design, accessibility, transit proximity, and spatial-network properties (Ewing and Cervero, 2010; Boeing, 2017). These land-use metrics matter because they can explain broad urban outcomes, such as sustainability, health, equity, and safety. For sustainability, the land-use metrics explain walking, transit use, and urban vitality (Aditjandra et al., 2016; Farahani et al., 2013; Lyu et al., 2025). For equity, they can measure spatial and social equity of people’s accessibility especially around the elderly communities (Gu et al., 2022). For safety, the land-use metrics can measure the crime risk across street segments (Kim and Hipp, 2021). Theoretical frameworks such as Lynch’s legibility (Lynch, 2008) and comprehensive urban design (Carmona, 2021) further situate land-use metrics as diagnostic tools for inclusive, sustainable, and resilient urban development. Such land-use metrics have been widely incorporated into land-use modeling through regression approaches (Boarnet, 2011), multi-source big data approaches (Zhang et al., 2019; Lyu et al., 2025), or recently machine learning methods (Li et al., 2024).

Diverse land-use models have been developed to understand and forecast urban land-use changes based on historical land-use patterns or sociodemographic drivers. The first land-use model family is cellular automata (CA) and its variants, which simulate land-use change through local transition rules, including partitioned and vector-based formulations that account for spatial heterogeneity and parcel structure (He et al., 2023; Cruz-Bello et al., 2024; Yang et al., 2023; Yao et al., 2024). Recent studies also expanded the classical CA approach by integrating deep learning architectures such as convolutional neural networks (CNNs) and graph neural networks (GNNs) to capture spatial dependencies and nonlinear transition dynamics (Guan et al., 2024; Xu et al., 2024). Besides CA models, a second family relies on statistical or machine learning methods, ranging from dynamic spatial regression (Tepe, 2023) to deep learning architectures that learn nonlinear transition patterns in an end-to-end manner (Wang et al., 2022; R¨ath et al., 2026; Wang et al., 2025b). Along this line, researchers further developed models that integrate transportation systems (Lopane et al., 2023), cooperative models (Zeng et al., 2025), and vision-language models (Cai et al., 2026). Despite their methodological diversity, these models share a common logic: they are predictive models forecasting the transition of land-use categories based on historical data, mostly relying on incremental changes in land-use categories rather than directly synthesizing non-existing land-use patterns.

## 2.2 Generative AI for urban systems

Generative AI is increasingly applied to geospatial analytics and urban planning tasks, enabling the synthesis of street layouts, building footprints, and urban imagery. Early work often leveraged GAN-based pipelines to support design automation and rapid visualization: MasterplanGAN and Urban-GAN generate masterplan-like urban layouts to streamline large-scale design workflows (Ye et al., 2022b; Quan, 2022), and other approaches reconstruct missing or sparse geospatial datasets to alleviate data scarcity in mapping and planning support (Wu et al., 2022). As summarized by Jiang et al. (2024), existing work highlights the promise of generative urban design in enhancing creativity, participation, and communication in the planning process.

However, the generation of land-use maps imposes stronger requirements than generic image synthesis. Specifically, the research community still lacks a generative framework that can jointly generate multiple urban layers while remaining controllable and verifiable. First of all, urban systems are inherently multimodal: land use, road networks, building footprints, and other spatial representations are spatially aligned layers and must remain consistent when one spatial map is generated. Most of the existing generative literature, however, remains centered on producing a single layer and only loosely connecting to other modalities in downstream steps (Ye et al., 2022b; Quan, 2022; Wu et al., 2022). Emerging multimodal generative directions suggest feasibility but also underline the gap. For instance, multimodal approaches that combine visual and auditory perceptions extend what urban representations can include, ofering richer depictions of human experience in cities (Zhuang et al., 2024). Meanwhile, unified or cooperative multimodal difusion models for text–image and audio–visual generation improve cross-modal reasoning and alignment (Li et al., 2025b; Chen et al., 2026; Rojas et al., 2025; Hayakawa et al., 2025). Yet, for landuse scenario planning, the core unmet need is a multimodal generator that can jointly produce spatial layers with explicit cross-modal geometric consistency, rather than synthesizing modalities independently.

Second, current generative models lack suficient controllability over planning objectives. In practice, urban planning is inherently goal-driven and therefore requires controlled generation: planners articulate explicit development objectives and constraints, and expect model outputs to respond predictably to these specified inputs. Difusion models provide a strong foundation for controlled generation because they avoid common GAN limitations such as mode collapse and training instability (Karras et al., 2020; Saxena and Cao, 2021), and can integrate objectives as controls (Dhariwal and Nichol, 2021; Croitoru et al., 2023). Technical advances in latent-space modeling (Rombach et al., 2022), optimized sampling strategies (Nichol and Dhariwal, 2021), and initialization refinements (Chung et al., 2022) have further improved their imagery generation eficiency. ControlNet marks an important milestone by injecting spatial conditions such as segmentation and depth maps into pretrained difusion models without modifying their weights (Zhang et al., 2023), while language–vision models such as CLIP and Imagen strengthen prompt grounding and prompt– output alignment (Radford et al., 2021; Saharia et al., 2022). Multi-view and geometry-aware dif fusion frameworks such as MVDream, 3D Adapter, and CMD further improve three-dimensional consistency and controllable editing across views (Shi et al., 2024; Chen et al., 2025; Li et al., 2025a). Recent planning-oriented studies have also begun to operationalize difusion-based generation with geospatial controls and iterative human guidance (He et al., 2026; Sun et al., 2026). Despite this progress, the controllability in generative urban models remains limited: current control interfaces are predominantly generic rather than grounded in the context of land-use planning.

Third, most of the generative outputs are evaluated by their general visual realism, but much less by domain-specific criteria such as land-use metrics. In existing generative studies, evaluation relies primarily on generic visual realism metrics such as Fr´echet Inception Distance (FID), which assess perceptual fidelity but do not verify whether outputs satisfy domain-specific requirements. Visual realism alone cannot assess whether a generated land-use scenario is geographically valid, structurally consistent across modalities, or aligned with stated planning objectives. Although difusion models have substantially improved output fidelity (Dhariwal and Nichol, 2021; Croitoru et al., 2023), the evaluation of spatial generation remains challenging without explicit domain-specific benchmarks. This limitation makes verifiable evaluation a core requirement of our work: planning-oriented generative frameworks need explicit and interpretable verification, for cross-modal structural coherence and for objective compliance that go beyond generic image realism metrics.

## 2.3 Scenario planning and evaluation

Scenario planning is a common approach for exploring multiple plausible urban futures before a municipality commits to a specific development path. Rather than producing a single forecast, it constructs a set of scenarios that bound key uncertainties and make tradeofs explicit for decisionmaking (Schoemaker, 1991; Van Notten et al., 2003). In urban planning practice, scenario planning is particularly common in land-use, transportation and growth management contexts, where alternative development patterns are compared to support public communication and deliberation (Bartholomew, 2007). A typical workflow includes: (1) defining general principles; (2) constructing narratives and translating them into specific objectives such as zoning requirements, growth expectations, or land density targets; (3) generating spatial scenarios using sketch planning, or land-use simulation models; and (4) evaluating impacts through indicators or multi-criteria methods, often supported by participatory planning tools (Couclelis, 2005; Labiosa et al., 2013). In fact, such an iteration over principle definition, goal specification, scenario generation, and lastly evaluation is mimicked in our design of DualControlNet.

Predictive land-use models can support scenario planning by using land-use change dynamics to forecast future land-use transitions. However, the literature has identified several structural limitations of predictive modeling in exploratory and design-oriented planning contexts. First, there is a high technical barrier and adoption friction in practice. Predictive models often require extensive data preparation, calibration, and parameter tuning, and the lack of institutional capacity and user familiarity can hinder their practical uptake (Vonk et al., 2005; Vonk and Geertman, 2008). This technical complexity may limit their accessibility in iterative and participatory planning processes. Second, many predictive models are explicitly calibrated to reproduce historical transition patterns. As noted in scenario planning literature, such calibration can bias model outputs toward path dependence and incremental change, making it dificult to represent disruptive futures or normative visions that intentionally depart from business-as-usual trajectories (Couclelis, 2005; Molinero-Parejo et al., 2023). Because the modeling logic is grounded in transition probabilities derived from past data, alternative spatial configurations that lack historical precedent may be underexplored. Third, validation in predictive land-use modeling is commonly based on reproducing observed short-term spatial patterns. Reviews of land-use modeling highlight challenges related to non-stationarity, equifinality, and structural uncertainty when extending projections over longer horizons (Van Vliet et al., 2016). These characteristics complicate the interpretation of long-term scenario outputs, particularly when scenarios aim to explore transformative change. Fourth, even within established modeling families such as urban cellular automata, considerable diversity exists in model structures and parameterizations. This heterogeneity complicates transparent model selection and explanation to stakeholders, potentially reducing interpretability in collaborative planning settings (Sant´e et al., 2010). Finally, because no single predictive framework is universally applicable across contexts and policy questions, scenario planning often benefits from multi-model comparison approaches; however, such practices increase analytical workload and coordination complexity (Hewitt et al., 2022). Together, these limitations motivate complementary approaches that enable faster iteration, clearer control of scenario assumptions, and more communicable outputs, while treating predictive land-use models as one component within a broader planning support workflow rather than the sole scenario generator (Couclelis, 2005).

## 2.4 Summary of research landscape and gaps

Traditional land-use models have provided valuable tools for quantifying and forecasting land development using interpretable land-use metrics. However, despite their methodological diversity, these approaches remain predictive: they rely on historical data to predict gradual transition in existing land-use categories, rather than directly creating new land-use maps. Generative AI, especially difusion models, has made it feasible to synthesize high-fidelity urban representations, such as satellite imagery or building footprints (Ho et al., 2020; Rombach et al., 2022; Wu and Biljecki, 2022; Dedring and Rienow, 2024; Sun et al., 2026; He et al., 2026). However, urban development practice still calls for a unified generative framework that supports multimodal joint generation, controllability based on targeted land-use metrics, and domain-specific evaluation beyond general visual realism (Lowry and Lowry, 2014; Abou Jaoude et al., 2022; Castro Lopes et al., 2025).

## 3 Data and Methods

Figure 1 illustrates the overall workflow of the DualControlNet framework for land-use scenario generation and evaluation. The pipeline consists of four stages: (a) data collection, (b) data processing, (c) model training, and (d) model evaluation. In the data collection stage, we assemble multimodal urban data, including satellite imagery, land-use imagery, and control maps derived from OpenStreetMap and Mapbox. During data processing, we compute a set of land use metrics, such as building coverage ratio, road density, land-use diversity, and average land-use area, which are further translated into structured language prompts to guide generation. In the model training stage, DualControlNet takes both control maps and metric-informed language prompts as inputs to jointly generate land use and satellite imagery. Finally, DualControlNet is evaluated from three sets of metrics: domain-specific land-use metrics, spatial consistency between land-use maps and satellite imagery, and visual realism.

![](images/cf129e1681e0c73ca3deff6ee90a6d208f156d653a3a8103af87fd188d2db78f.jpg)  
Figure 1: Workflow for joint land-use map and satellite imagery generation

## 3.1 Data Collection and Processing

We first developed a multi-source data collection framework combining high-resolution satellite imagery and land use maps from OpenStreetMap. The dataset was collected using a custom-built GeoDataCollector that systematically samples urban areas through a grid-based approach. We collected satellite image data from two U.S. cities: Chicago and Orlando. The data was collected centered around each city’s downtown area. For Chicago, the collection area was divided into a 180 × 120 grid, resulting in 21,600 image tiles. For Orlando, a 150 × 150 grid was used, yielding 22,500 image tiles. Each tile covers a fixed ground area of about 0.45 km × 0.45 km. The method calculates the exact latitude and longitude boundaries of each tile. Figure 2 illustrates the study areas and provides examples of the collected satellite and land use images.

![](images/559ab54d4e8c7f6bbee83b6579d2253592b9e4c9dd93671534cde3d9cb565b19.jpg)  
Figure 2: Study areas and examples of spatially aligned satellite and land use images

For satellite imagery, we used Mapbox Satellite v9, which provides high-resolution imagery primarily sourced from Maxar/DigitalGlobe between 2011-2023. Images were acquired through the Mapbox Static Images API at a resolution of 512x512 pixels, with consideration of geographical coordinates and zoom level 17 to maintain consistent spatial coverage across diferent latitude positions. Using OpenStreetMap, we gathered land use maps with diverse functions. Land use categories include residential, commercial, and industrial zones; urban service facilities such as educational institutions, healthcare centers, and recreational spaces; natural landscape elements such as forests, water bodies, and open green areas. These land-use categories together constitute the land use map. Lastly, we created a set of control maps to guide the generation of land use and satellite imagery. The control maps include water areas and transportation infrastructure. We extracted water areas from OpenStreetMap, including rivers, lakes, and other significant water bodies. As summarized in Figure 3, we collected three types of spatially aligned images: (1) high-resolution satellite imagery showing the physical characteristics of the urban landscape, (2) detailed land use maps displaying the spatial organization of land-use elements, and (3) control maps featuring water areas and major road networks.

![](images/13e381c1da2142c8eee14aeab015b3e8edbfa1d9081ff2a29fb66ebe9f21569a.jpg)  
Figure 3: Example of multimodal data representations: text prompts, satellite images, land-use maps, and control images

Imagery alone cannot make generation controllable or verifiable, because a planner specifies a scenario in numbers rather than in pixels. We therefore compute nine quantitative metrics for every tile, grouped into two families that play distinct roles in our framework (Table 1). The first family, land-use metrics, is the target of both control and verification. Building coverage ratio measures how intensively land is developed; land-use diversity, computed as a Shannon entropy over land-use proportions, measures how functionally mixed it is; the number of land-use categories records how many distinct functions are present; and average land-use area measures the spatial granularity at which those functions are organized. These four quantities can be specified as a planning objective on the input side and recovered from a generated map on the output side, which allows a generated scenario to be verified.

The second family, general built environment metrics, describes the morphological and infrastructural context in which land use is embedded. Road density and intersection density together characterize the extent and granularity of the street network, road network complexity summarizes its structural intricacy as the ratio of segments to intersections, average building size records the typical footprint of individual structures, and average distance to amenities captures local accessibility to urban services. Unlike the land-use metrics, these quantities are not recoverable from a generated land-use map and hence do not serve as verification targets. Instead, they condition generation by situating each scenario within a plausible urban form, ensuring that a prompted level of development is realized with a street network and building fabric consistent with real cities.

Table 1: Summary of land-use and built environment metrics
<table><tr><td>Metric</td><td>Description</td><td>Formula</td></tr><tr><td colspan="3">Land Use Metrics</td></tr><tr><td>Building Coverage Ratio</td><td>Proportion of land area cov- ered by buildings.</td><td> $\begin{array} { r } { B C R = \frac { A r e a _ { \mathrm { b u i l d i n g s } } } { A r e a _ { \mathrm { t o t a l } } } } \end{array}$ </td></tr><tr><td>Land Use Diversity</td><td>Shannon diversity index, where  $p _ { i }$  is the proportion of land area for land-use type i.</td><td> $\begin{array} { r } { L U D = - \sum _ { i = 1 } ^ { n } p _ { i } \ln ( p _ { i } ) } \end{array}$ </td></tr><tr><td>Land Use Categories</td><td>Number of distinct land-use Ncategories types in the area.</td><td></td></tr><tr><td>Average Land Use Area</td><td>Average area per land-use category, where  $A r e a _ { i }$  is the area of type i.</td><td> $\begin{array} { r } { A L U A = \frac { \sum _ { i = 1 } ^ { n } A r e a _ { i } } { n } } \end{array}$ </td></tr><tr><td colspan="3">Other Built Environment Metrics</td></tr><tr><td>Road Density</td><td>Total length of roads per square kilometer.</td><td> $\begin{array} { r } { R D = \frac { L e n g t h _ { \mathrm { r o a d s } } } { A r e a _ { \mathrm { k m ^ { 2 } } } } } \end{array}$ </td></tr><tr><td>Intersection Density</td><td>Number of road intersections per square kilometer.</td><td> $\begin{array} { r } { I D = \frac { N _ { \mathrm { { i n t e r s e c t i o n s } } } } { A r e a _ { \mathrm { { k m ^ { 2 } } } } } } \end{array}$ </td></tr><tr><td>ity</td><td>Road Network Complex- Ratio of road segments (edges) to intersections (nodes).</td><td> $\begin{array} { r } { R N C = \frac { N _ { \mathrm { e d g e s } } } { N _ { \mathrm { n o d e s } } } } \end{array}$ </td></tr><tr><td>Average Building Size</td><td>Average area per building, where  $A r e a _ { j }$  is the area of building j.</td><td> $A B S = \frac { \sum _ { j = 1 } ^ { m } A r e a _ { j } } { m }$ </td></tr><tr><td>Average Distance to Mean distance from the cen- Amenities</td><td>ter point to all amenities.</td><td> $\begin{array} { r } { A D A = \frac { \sum _ { i = 1 } ^ { n } d _ { i } } { n } } \end{array}$ </td></tr></table>

The nine metrics in Table 1 are numeric, whereas a difusion model is conditioned through language. Our first processing step therefore translates each tile’s numeric metrics into a structured text prompt, illustrated in the Text Prompt column of Figure 3. The translation is deliberately hybrid: every metric enters the prompt twice, once as a qualitative descriptor and once as its exact numeric value in parentheses. A prompt reads, for example, “A realistic image of an urban area in Chicago...This area has medium building coverage ratio (0.52), sparse and minimal urban development with very few buildings. It features low road density (11.5 km/km<sup>2</sup>) where sparse road network, with low intersection density (78.2 intersections/km<sup>2</sup>)...” Each prompt thus opens by naming the city, and then proceeds metric by metric through building coverage ratio, landuse diversity, road density, intersection density, and the remaining quantities, rendering each as a categorical level, followed by the measured value and a short verbal description of what that value means for urban form. The multimodal data representations in Figure 3 complete our data preparation for DualControlNet, as the control images and text prompts serving as inputs and the paired satellite imagery and land-use maps as outputs.

## 3.2 Difusion Models and ControlNet Baselines

We adopt Stable Difusion as the baseline model. Difusion models generate samples by learning a reverse denoising process that inverts a forward process, in which Gaussian noise is gradually added to a clean latent sample z<sub>0</sub> to obtain a noisy latent $z _ { t }$ at timestep t. Given a conditioning input x, which is the text prompt in our setting, a neural network $\epsilon \theta$ is trained to predict the injected noise ϵ at each timestep. The training objective is:

$$
L _ { ( \theta ) } = \mathbb { E } _ { z _ { 0 } , t , \epsilon \sim \mathcal { N } ( 0 , 1 ) } \left[ \| \epsilon - \epsilon _ { \theta } ( z _ { t } , t , \tau _ { \theta } ( x ) ) \| ^ { 2 } \right] .\tag{1}
$$

Although text prompts enable diverse image synthesis, they provide limited spatial controllability for land-use map generation. We therefore build on ControlNet (Zhang et al., 2023), which augments a pretrained difusion U-Net with additional conditioning while leaving the original pretrained weights intact. ControlNet duplicates the base U-Net into a frozen copy and a trainable copy, and introduces two zero-initialized $1 \times 1$ convolutions, denoted $\mathcal { Z } ( \cdot ; \cdot )$ , to inject the control signal and merge the conditional residual. Let $\mathcal { F } ( \cdot ; \Theta )$ denote the frozen base model and $\mathcal { F } ( \cdot ; \Theta _ { c } )$ the trainable copy, initialized with $\Theta _ { c } = \Theta$ . Given a control condition $^ { c , }$ the conditioned output is:

$$
y _ { c } = \mathcal { F } ( x ; \Theta ) + \mathcal { Z } ( \mathcal { F } ( x + \mathcal { Z } ( c ; \Theta _ { z 1 } ) ; \Theta _ { c } ) ; \Theta _ { z 2 } ) ,\tag{2}
$$

and the denoising objective for training the ControlNet branch is:

$$
L _ { ( \Theta _ { c } , \Theta _ { z 1 } , \Theta _ { z 2 } ) } = \mathbb { E } _ { z _ { 0 } , t , c , \epsilon \sim \mathcal { N } ( 0 , 1 ) } \left[ \| \epsilon - \epsilon _ { \Theta _ { c } , \Theta _ { z 1 } , \Theta _ { z 2 } } ( z _ { t } , t , c ) \| ^ { 2 } \right] .\tag{3}
$$

This objective is consistent with standard difusion training practice (Dhariwal and Nichol, 2021; Croitoru et al., 2023). Because the control signal enters through the trainable copy alone, the formulation lets generation follow the explicit spatial constraints carried by the control images, which is essential for our tasks that demand spatial structures.

## 3.3 Designing DualControlNet

Multimodal land-use scenario planning needs not only realistic imagery, but also cross-modal consistency: roads, blocks, and parcel boundaries should align between the satellite imagery and the land-use maps. In practice, training two separate ControlNets for satellite imagery and land-use maps can produce individually plausible outputs, yet their geometric structures may drift because the two outputs are optimized independently and do not share an explicit mechanism to enforce spatial alignment.

Therefore, we propose DualControlNet to support multimodal generation with cross-modal spatial consistency. The core idea is to extend the ControlNet formulation to a dual-modality setting and introduce a geospatial consistency loss that regularizes the geometric alignment between the generated land-use map and satellite imagery, as illustrated in Figure 4. DualControlNet is not only a higher-fidelity generator, but also a multimodal generator whose outputs are spatially aligned and more amenable to evaluation against land-use metrics. This modification is implemented within the training objective computed at each difusion timestep t.

Given a clean latent $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { B \times C \times H \times W }$ , the forward process samples noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ and produces $\mathbf { x } _ { t } \sim q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ . The network predicts a target y<sub>target</sub> under a chosen parameterization (noise, $\mathbf { x } _ { \mathrm { 0 } }$ , or velocity). Under velocity prediction,

$$
\mathbf { v } = \alpha _ { t } \epsilon + \beta _ { t } \mathbf { x } _ { 0 } ,\tag{4}
$$

![](images/86d921b34ef45da5163163bad01527e33e949aaa85239242cc4b98734ad520fb.jpg)  
Figure 4: Demonstration of DualControlNet

where $\alpha _ { t }$ and $\beta _ { t }$ are determined by the noise schedule. We optimize the denoising objective

$$
\mathcal { L } _ { \mathrm { s i m p l e } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon } \left[ \left\| \hat { \mathbf { y } } - \mathbf { y } _ { \mathrm { t a r g e t } } \right\| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

Following standard practice, we use log-variance reweighting and an additional VLB term, which are combined into the overall training objective $\mathcal { L } _ { \mathrm { t o t a l } }$ (Dhariwal and Nichol, 2021; Croitoru et al., 2023).

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { s i m p l e } } \cdot { \mathcal { L } } _ { \mathrm { r e w e i g h t e d } } + \lambda _ { \mathrm { v l b } } \cdot { \mathcal { L } } _ { \mathrm { v l b } } .\tag{6}
$$

These formulations follow standard difusion training practice (Dhariwal and Nichol, 2021; Croitoru et al., 2023), and the conditioning mechanism is adapted from ControlNet (Zhang et al., 2023).

Our key modification is to add a geospatial consistency loss computed on the reverse-predicted clean latent $\mathbf { x } _ { 0 } ^ { \mathrm { p r e d } }$ , reconstructed from $\mathbf { x } _ { t }$ and the model output $\hat { \mathbf { y } }$ :

$$
\mathbf { x } _ { 0 } ^ { \mathrm { p r e d } } = \frac { \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \cdot \hat { \mathbf { y } } } { \sqrt { \bar { \alpha } _ { t } } } ,\tag{7}
$$

where $\bar { \alpha } _ { t }$ is the cumulative product of the noise schedule up to timestep t. We split $\mathbf { x } _ { 0 } ^ { \mathrm { p r e d } }$ along the width dimension at the midpoint $W _ { \mathrm { m i d } } = \lfloor W / 2 \rfloor$ :

$$
\begin{array} { r } { { \mathbf { x } } _ { 0 , \mathrm { l e f t } } ^ { \mathrm { p r e d } } = { \mathbf { x } } _ { 0 } ^ { \mathrm { p r e d } } [ : , : , : , : W _ { \mathrm { m i d } } ] , \quad { \mathbf { x } } _ { 0 , \mathrm { r i g h t } } ^ { \mathrm { p r e d } } = { \mathbf { x } } _ { 0 } ^ { \mathrm { p r e d } } [ : , : , : , W _ { \mathrm { m i d } } : ] . } \end{array}\tag{8}
$$

To capture geometric boundaries, we compute finite-diference gradients along height and width:

$$
\begin{array} { r } { \nabla _ { h } ^ { \mathrm { l e f t } } = \mathbf { x } _ { 0 , \mathrm { l e f t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , 1 : ; \cdot ] - \mathbf { x } _ { 0 , \mathrm { l e f t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , \cdot - 1 , \cdot ] , \quad \nabla _ { h } ^ { \mathrm { r i g h t } } = \mathbf { x } _ { 0 , \mathrm { r i g h t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , 1 : ; \cdot ] - \mathbf { x } _ { 0 , \mathrm { r i g h t } } ^ { \mathrm { p r e d } } [ \cdot ; \cdot , \cdot - 1 , \cdot ] . } \end{array}\tag{9}
$$

$$
\begin{array} { r } { \nabla _ { w } ^ { \mathrm { l e f t } } = \mathbf { x } _ { 0 , \mathrm { l e f t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , \cdot , 1 : ] - \mathbf { x } _ { 0 , \mathrm { l e f t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , \cdot , \cdot - 1 ] , \quad \nabla _ { w } ^ { \mathrm { r i g h t } } = \mathbf { x } _ { 0 , \mathrm { r i g h t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , \cdot , 1 : ] - \mathbf { x } _ { 0 , \mathrm { r i g h t } } ^ { \mathrm { p r e d } } [ \cdot , \cdot , \cdot , \cdot - 1 ] . } \end{array}\tag{10}
$$

To address possible size mismatch, we crop both sides to the minimum overlapping size:

$$
H _ { \mathrm { m i n } } = \operatorname* { m i n } ( \mathrm { h e i g h t } ( \nabla _ { h } ^ { \mathrm { l e f t } } ) , \mathrm { h e i g h t } ( \nabla _ { h } ^ { \mathrm { r i g h t } } ) ) , \quad W _ { \mathrm { m i n } } ^ { h } = \operatorname* { m i n } ( \mathrm { w i d t h } ( \nabla _ { h } ^ { \mathrm { l e f t } } ) , \mathrm { w i d t h } ( \nabla _ { h } ^ { \mathrm { r i g h t } } ) ) ,\tag{11}
$$

$$
H _ { \mathrm { m i n } } ^ { w } = \operatorname* { m i n } ( \mathrm { h e i g h t } ( \nabla _ { w } ^ { \mathrm { l e f t } } ) , \mathrm { h e i g h t } ( \nabla _ { w } ^ { \mathrm { r i g h t } } ) ) , \quad W _ { \mathrm { m i n } } = \operatorname* { m i n } ( \mathrm { w i d t h } ( \nabla _ { w } ^ { \mathrm { l e f t } } ) , \mathrm { w i d t h } ( \nabla _ { w } ^ { \mathrm { r i g h t } } ) ) .\tag{12}
$$

We define the contour consistency loss as the sum of L1 distances between the cropped gradient maps:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e o } } ^ { ( h ) } = 1 1 ( \nabla _ { h } ^ { \mathrm { l e f t } } [ : , : , : H _ { \mathrm { m i n } } , : W _ { \mathrm { m i n } } ^ { h } ] , \nabla _ { h } ^ { \mathrm { r i g h t } } [ : , : , : H _ { \mathrm { m i n } } , : W _ { \mathrm { m i n } } ^ { h } ] ) , } \end{array}\tag{13}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e o } } ^ { ( w ) } = 1 1 ( \nabla _ { w } ^ { \mathrm { l e f t } } [ : , : , : H _ { \mathrm { m i n } } ^ { w } , : W _ { \mathrm { m i n } } ] , \nabla _ { w } ^ { \mathrm { r i g h t } } [ : , : , : H _ { \mathrm { m i n } } ^ { w } , : W _ { \mathrm { m i n } } ] ) , } \end{array}\tag{14}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e o } } = \mathcal { L } _ { \mathrm { g e o } } ^ { ( h ) } + \mathcal { L } _ { \mathrm { g e o } } ^ { ( w ) } . } \end{array}\tag{15}
$$

This auxiliary term encourages the two modalities to share consistent boundary geometry, improving cross-modal structural alignment. We apply a small weight $\lambda _ { \mathrm { g e o } }$ to avoid overpowering the denoising objective. The final objective is:

$$
{ \mathcal { L } } _ { \mathrm { f i n a l } } = { \mathcal { L } } _ { \mathrm { s i m p l e } } + \lambda _ { \mathrm { v l b } } \cdot { \mathcal { L } } _ { \mathrm { v l b } } + \lambda _ { \mathrm { g e o } } \cdot { \mathcal { L } } _ { \mathrm { g e o } } .\tag{16}
$$

In our experiments, we set $\lambda _ { \mathrm { g e o } } = 0 . 0 0 1$ to balance geometric regularization and generative performance. By integrating this geospatial prior into the latent difusion framework, DualControlNet improves cross-modal structural coherence while retaining the high-fidelity synthesis of difusion models.

## 4 Experiment Design

## 4.1 Model evaluation

A generated land-use map is useful to planning only if it is verifiable, and verification demands more than visual plausibility. We therefore evaluate the models along three groups of metrics, summarized in Table 2. The first group, land-use metrics, asks whether a generated map satisfies the planning objectives that the prompt specified, comparing the building coverage ratio, land-use diversity, average land-use area, and land-use composition of the generated map against the real maps. The second group, spatial consistency metrics, asks whether the geometric structures in the satellite image and the land-use map align. The third group, visual realism, asks whether the generated satellite imagery is perceptually credible. The three groups are complementary: a model can produce a realistic satellite image that contradicts its own land-use map, or a pair of wellaligned images that ignores the requested density. Because DualControlNet generates the paired modalities jointly, we can quantify all three groups without relying on subjective inspection.

The land-use metrics are the core of our verification, and computing them requires the postprocessing of generated land-use maps. The model returns a land-use map as an image rather than as a vector layer, so the same metrics we defined in Table 1 must be recovered from the pixels. We recover them as follows: for land-use diversity and the number of land-use categories, the generated map is downsampled and converted into an RGB pixel array, and color quantization groups similar colors into clusters; we retain only clusters occupying more than 1% of the image and discard very bright colors, which correspond to unclassified areas. The retained clusters provide the set of land-use categories, and their area shares provide the land-use composition. For building coverage ratio, we identify building pixels through predefined RGB ranges, construct a binary mask, and take the ratio of building pixels to total pixels. For average land-use area, we build a binary mask per retained category, apply connected-component labeling to delineate contiguous parcels, discard components below a minimum-size threshold, and average the areas of the remainder. Each of the four land-use metrics is then converted into a consistency score, defined in Table 2, such that a higher value indicates that the generated map reproduces the reference map and a lower value indicates larger diferences.

Table 2: Eight evaluation metrics used in model assessment.
<table><tr><td>Metric</td><td>Description</td><td>Formula</td></tr><tr><td colspan="3">Land use metrics</td></tr><tr><td>versity Consistency</td><td>Avg. Land Use Di- Measures similarity in land-use di- versity between generated and ref- erence maps.</td><td> $\begin{array} { r } { S _ { \mathrm { L U D } } = 1 - \frac { \left| L U D _ { g } - L U D _ { r } \right| } { \operatorname* { m a x } ( L U D _ { g } , L U D _ { r } ) } } \end{array}$ </td></tr><tr><td>tency</td><td>Avg. Building Cov- Measures similarity in building erage Ratio Consis- coverage ratio between generated and reference land-use maps.</td><td> $S _ { \mathrm { B C R } } = 1 - | B C R _ { g } - B C R _ { r } |$ </td></tr><tr><td>tency</td><td>Avg. Average Land Measures similarity in average Use Area Consis- land-use area between generated and reference land-use maps.</td><td> $\begin{array} { r } { S _ { \mathrm { A L U A } } = 1 - \frac { \left| A L U A _ { g } - A L U A _ { r } \right| } { \operatorname* { m a x } \left( A L U A _ { g } , A L U A _ { r } \right) } } \end{array}$ </td></tr><tr><td>Similarity</td><td>Avg. Distribution Evaluates statistical similarity of land-use proportion distributions between generated and reference maps.</td><td> $\begin{array} { r } { S _ { \mathrm { d i s t } } = 1 - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| p _ { i } - q _ { i } \right| } \end{array}$ </td></tr><tr><td colspan="3">Spatial consistency</td></tr><tr><td>Cosine Similarity</td><td>Measures directional alignment Cosine Similarity between generated remote sens- ing imagery and the corresponding land-use map, reflecting global ori-</td><td> $\begin{array} { r } { \sum _ { i = 1 } ^ { N } G _ { 1 } ( i ) \bar { G _ { 2 } } ( i ) } \end{array}$   $\begin{array} { r } { \overline { { \sqrt { \sum _ { i = 1 } ^ { N } G _ { 1 } ( i ) ^ { 2 } } } } \sqrt { \sum _ { i = 1 } ^ { N } G _ { 2 } ( i ) ^ { 2 } } } \end{array}$ </td></tr><tr><td>bution Similarity</td><td>entation consistency. Magnitude Distri- Compares the distribution of vec- tor magnitudes between the gener- ated satellite image and land-use image.</td><td> $\begin{array} { r } { S _ { \mathrm { m a g } } = \frac { 1 } { 1 + W ( P _ { 1 } ^ { m } , P _ { 2 } ^ { m } ) } } \end{array}$ </td></tr><tr><td>tion Similarity</td><td>Direction Distribu- Evaluates whether directional fea- tures are preserved between the generated satellite image and land- use image.</td><td> $S _ { \mathrm { d i r } } = 1 - d _ { \mathrm { c o s } } ( P _ { 1 } ^ { d } , P _ { 2 } ^ { d } )$ </td></tr><tr><td colspan="3">Visual realism Fréchet Inception Compares generated remote sens- FID</td></tr><tr><td>Distance (FID)</td><td>ing imagery with real images to quantify overall realism and gen- eration quality.</td><td> $\begin{array} { r l r l r l } { = } & { { } } & { \| \mu _ { r } } & { { } - } & { \mu _ { g } \| ^ { 2 } } & { { } + } & { } \end{array}$   $\mathrm { T r } \big ( \Sigma _ { r } + \Sigma _ { g } - 2 ( \Sigma _ { r } \Sigma _ { g } ) ^ { 1 / 2 } \big )$ </td></tr></table>

The spatial consistency metrics evaluate whether the satellite image and the land-use map depict the same urban form. We assess it through image gradients, since parcel boundaries, road edges, and block faces can be approximated by gradients in both land-use and satellite images. Cosine similarity compares the direction of the two gradient fields, magnitude distribution similarity compares their distributions of gradient strength, and direction distribution similarity compares their distributions of gradient orientation. Visual realism is assessed with the Fr´echet Inception Distance (FID), which compares latent distributions of generated and real satellite imagery, with lower values indicating closer agreement. FID measures only perceptual credibility and not geographic validity.

## 4.2 Experiment process

All models were trained and evaluated on high-performance computing platform, using both NVIDIA A100 and L4 GPUs. The dataset comprises over 40,000 paired satellite and land use images from Chicago and Orlando for training, with an additional 4,000 pairs reserved for testing. Four diferent models were developed and compared, each trained for 5 epochs, with training for each model taking approximately 20 GPU hours. Four distinct model configurations are compared. The first two models are built upon our DualControlNet architecture. The first model, DualControlNet without (w/o) Geo Loss, serves as a baseline. While capable of dual-task generation, it deliberately omits our proposed geospatial loss function. In contrast, the second model, DualControlNet shares the same architecture but crucially integrates the geospatial loss to enhance the spatial consistency between the generated satellite and land use outputs. The third model, ControlNet (Land use), is specialized to generate only the land use map from a single input control map. The fourth model, ControlNet (Satellite), is dedicated solely to generate the satellite image from its respective control map.

## 5 Results

## 5.1 Evaluating model performance

Table 3 and Table 4 compare four models across land-use, spatial consistency, and visual realism metrics in Chicago and Orlando. Two of these models are unimodal ControlNet baselines, conditioned on land use or satellite imagery alone, and two are DualControlNet variants that jointly generate both modalities, with and without the geospatial loss. Our comparison therefore proceeds in two steps: we first contrast multimodal DualControlNet against the unimodal baselines to isolate the value of controllable and verifiable generation, and then contrast DualControlNet with and without the geospatial loss to isolate the loss’s contribution to joint generation. We begin with land-use metrics, as these indicators directly reflect whether the generated outputs align with the text prompts and can be verified against domain-specific measures. We then turn to spatial consistency and visual realism metrics, with the former tackling multimodal coherence and the latter serving as a general evaluation of imagery quality.

Both DualControlNet variants substantially outperform the unimodal ControlNet in terms of aligning land-use metrics. Relative to the unimodal baselines, the best DualControlNet improves Land Use Diversity Consistency from 0.3428 to 0.5580, Average Land Use Area Consistency from 0.1755 to 0.2624, Building Coverage Ratio Consistency from 0.6362 to 0.7400, and Distribution Similarity from 0.8249 to 0.8734. Such increases correspond to percentage gains of 62.8%, 49.5%, 16.3%, and 5.9%. The improvement shows that jointly generating land use together with satellite imagery, rather than generating land use or satellite imagery in isolation, yields outputs that are both more controllable and more faithful to the target land-use metrics. DualControlNet with geospatial loss attains the highest Average Building Coverage Ratio Consistency of 0.7400 and the highest Average Land Use Area Consistency of 0.2624, improving on the version without it by 5.8% and 8.4%, respectively, particularly in reconstructing density and building blocks. The version without geospatial loss retains a modest edge in Land Use Diversity Consistency (0.5580 vs. 0.4992, 11.8%) and Distribution Similarity (0.8734 vs. 0.8570, 1.9%), suggesting that relaxing the geospatial constraint allows slightly greater flexibility in learning diverse land-use compositions. Importantly, the satellite-only ControlNet produces satellite imagery but no land-use map, so it cannot be verified on any land-use metric and is reported as N/A throughout Table 3. This result again highlights verifiable generation as an important contribution of DualControlNet.

Table 3: Models evaluated by land-use metrics (I)
<table><tr><td>Metric (City)</td><td>Dual Control Net w/o Geo Loss</td><td>Dual Control Net w/ Geo Loss</td><td>Control Net Lan- dUse</td><td>Control Net Satellite</td></tr><tr><td>Avg. Land Use Diversity Con. (All)</td><td>0.5580</td><td>0.4992</td><td>0.3428</td><td>N/A</td></tr><tr><td>Avg. Land Use Diversity Con. (Orlando)</td><td>0.5075</td><td>0.4391</td><td>0.3296</td><td>N/A</td></tr><tr><td>Avg. Land Use Diversity Con. (Chicago)</td><td>0.6084</td><td>0.5592</td><td>0.3560</td><td>N/A</td></tr><tr><td>Avg. Building Coverage Ratio Con. (All)</td><td>0.6993</td><td>0.7400</td><td>0.6362</td><td>N/A</td></tr><tr><td>Avg. Building Coverage Ratio Con. (Orlando)</td><td>0.7805</td><td>0.7971</td><td>0.7323</td><td>N/A</td></tr><tr><td>Avg. Building Coverage Ratio Con. (Chicago)</td><td>0.6180</td><td>0.6829</td><td>0.5400</td><td>N/A</td></tr><tr><td>Avg. Average Land Use Area Con. (All)</td><td>0.2421</td><td>0.2624</td><td>0.1755</td><td>N/A</td></tr><tr><td>Avg. Average Land Use Area Con. (Orlando)</td><td>0.2201</td><td>0.2337</td><td>0.1734</td><td>N/A</td></tr><tr><td>Avg. Average Land Use Area Con. (Chicago)</td><td>0.2640</td><td>0.2911</td><td>0.1775</td><td>N/A</td></tr><tr><td>Avg. Distribution Sim. (All)</td><td>0.8734</td><td>0.8570</td><td>0.8249</td><td>N/A</td></tr><tr><td>Avg. Distribution Sim. (Orlando)</td><td>0.8711</td><td>0.8474</td><td>0.8208</td><td>N/A</td></tr><tr><td>Avg. Distribution Sim. (Chicago)</td><td>0.8756</td><td>0.8665</td><td>0.8290</td><td>N/A</td></tr></table>

Table 4: Models evaluated by spatial consistency and visual realism (II)
<table><tr><td>Metric (City)</td><td>Net w/o Geo Loss</td><td>DualControl DualControl ControlNet Net w/ Geo Loss</td><td>LandUse</td><td>ControlNet Satellite</td></tr><tr><td>FID (All)</td><td>61</td><td>58</td><td>213</td><td>61</td></tr><tr><td>FID (Orlando)</td><td>76</td><td>78</td><td>215</td><td>81</td></tr><tr><td>FID (Chicago)</td><td>93</td><td>82</td><td>248</td><td>89</td></tr><tr><td>Cosine Sim. (All)</td><td>0.2180</td><td>0.2364</td><td>0.1118</td><td>0.1083</td></tr><tr><td>Cosine Sim. (Orlando)</td><td>0.2473</td><td>0.2754</td><td>0.1122</td><td>0.1125</td></tr><tr><td>Cosine Sim. (Chicago)</td><td>0.1887</td><td>0.1973</td><td>0.1113</td><td>0.1041</td></tr><tr><td>Magnitude Dist. Sim. (All)</td><td>0.9782</td><td>0.9804</td><td>0.9713</td><td>0.9733</td></tr><tr><td>Magnitude Dist. Sim. (Orlando)</td><td>0.9801</td><td>0.9817</td><td>0.9716</td><td>0.9738</td></tr><tr><td>Magnitude Dist. Sim. (Chicago)</td><td>0.9763</td><td>0.9791</td><td>0.9710</td><td>0.9728</td></tr><tr><td>Direction Dist. Sim. (All)</td><td>0.5415</td><td>0.5911</td><td>0.2894</td><td>0.3181</td></tr><tr><td>Direction Dist. Sim. (Orlando)</td><td>0.4757</td><td>0.5177</td><td>0.2989</td><td>0.3281</td></tr><tr><td>Direction Dist. Sim. (Chicago)</td><td>0.6072</td><td>0.6645</td><td>0.2799</td><td>0.3081</td></tr></table>

For spatial consistency metrics, both DualControlNet variants again dominate the unimodal baselines. The best DualControlNet more than doubles Cosine Similarity, reaching 0.2364 versus 0.1118 for ControlNet LandUse and 0.1083 for ControlNet Satellite, and roughly doubles Direction Distribution Similarity, reaching 0.5911 versus 0.2894 and 0.3181 respectively. Within the Dual-ControlNet family, the geospatial loss then provides a consistent additional gain. Cosine Similarity increases from 0.2180 to 0.2364, an 8.4% improvement; Magnitude Distribution Similarity rises from 0.9782 to 0.9804; and Direction Distribution Similarity increases substantially with a 9.2% improvement. These gains confirm that the geospatial consistency loss enhances global and directional structural coherence across modalities. Lastly, we assess visual realism using FID scores. DualControlNet is far sharper than the land-use-only baseline, lowering FID from 213 to 58, and it performs on par with or better than the satellite-only baseline, which reaches 61. Adding the geospatial loss lowers the overall FID from 61 to 58, and in Chicago from 93 to 82, while slightly worsening FID in Orlando. These results indicate that multimodal joint generation, reinforced by geospatial constraints, not only improves cross-modal consistency but also enhances perceptual image quality.

Across Orlando and Chicago, performance is generally consistent for land-use metrics and spatial consistency. DualControlNet with geospatial loss achieves higher Building Coverage Ratio Consistency in Orlando (0.7971) than in Chicago (0.6829), while spatial similarity metrics are also slightly stronger in Orlando. However, Direction Distribution Similarity is higher in Chicago (0.6645), likely due to its denser and more regular street network, which strengthens directional alignment under control-map conditioning.

Overall, the model evaluation highlights that land-use metrics successfully provide domainspecific verification of generated land-use maps, while spatial consistency metrics indicate the importance of multimodal generation across urban layers. DualControlNet with geospatial loss achieves the best balance across these three dimensions, supporting multimodal generation, metricbased control, and domain-specific verification.

![](images/666a3dfa5e0dd23cf03d7214f2c12922f2f064bc2c730e0a172841e31ca8c860.jpg)  
Figure 5: Examples of synthetic satellite images and land use images in Orlando

## 5.2 Generative baselines: reconstructing land-use patterns with original prompts

Given a control map encoding only major roads and water bodies, DualControlNet can reconstruct a high-resolution satellite image and a land-use map corresponding to the original land-use <sup>(a)</sup> <sup>River</sup> <sup>Corridor</sup> <sup>(b)</sup> <sup>Industrial</sup> <sup>District</sup> <sup>(c)</sup> <sup>Commercial</sup> <sup>Corridor</sup> <sup>(d)</sup> <sup>Urban</sup> <sup>Grid</sup> <sup>(e)</sup> <sup>Mixed</sup> <sup>Development</sup>prompts. Figure 5 illustrates synthetic land-use maps in Orlando across five diverse urban environments, including Forest Edge, Water Corridor, Suburban Woodland, Waterfront District, and <sup>a</sup>Lake Community.

olAcross these urban environment types, the generated land-use maps and satellite imagery preo<sup>I</sup>serve key structural elements from the control input. For example, in the Water Corridor and Lake Community cases, the model maintains the geometry of water bodies while generating realistic surrounding residential and green land uses. In the Forest Edge and Suburban Woodland examples, natural landscape patterns are reconstructed with gradual transitions between built and <sup>g</sup> vegetated areas. In the Waterfront District scenario, the model captures dense development along Ithe shoreline, aligning building footprints in the satellite image with corresponding urban land-use llip<sup>u</sup>categories. The model also learns semantic correspondences between roof morphology in satellite <sup>a</sup>Oimagery and land-use categories. For instance, single-family houses in Suburban Woodland and e<sup>d</sup>Lake Community scenarios exhibit smaller, detached roof structures associated with residential land <sup>r</sup>use, whereas larger and more regular building footprints in Waterfront District areas correspond g<sup>e</sup>eto commercial or mixed-use classifications.

![](images/afb15f3c31c143a32980c9cd03e61be2b4781ed3a2364731c3783fa329718ae5.jpg)  
Figure 6: Examples of synthetic satellite images and land use images in Chicago

Figure 6 demonstrates that DualControlNet can generate land-use maps for diverse urban environments similarly in Chicago, including River Corridor, Industrial District, Commercial Corridor, Urban Grid, and Mixed Development areas. Similar to Orlando, the model preserves the primary water and road structures specified in the control maps, demonstrating consistent generation aligning with imagery inputs. In the River Corridor scenario, DualControlNet clearly maintains the alignment between river geometry and adjacent green or industrial land use. In Industrial District and Commercial Corridor areas, the model generates larger building blocks and structured road networks consistent with high-intensity development. The Urban Grid example highlights the model’s ability to reconstruct regular block patterns and orthogonal street layouts characteristic of Chicago’s dense street network. In the Mixed Development case, diverse land-use categories are arranged at the block level, reflecting fine-grained spatial organization.

Compared to Orlando, Chicago examples exhibit higher building density and more regular grid structures, which are well captured in both satellite imagery and land-use maps. However, the visual realism of satellite imagery in Chicago is slightly lower (FID = 82) than in Orlando (FID = 78), with some brightness shifts and minor distortions in areas containing tall buildings and cast shadows. This diference may be attributed to the higher structural complexity and denser urban fabric in Chicago.

## 5.3 Generative scenario planning with single land-use dimensions

To demonstrate the potential of generative scenario planning, we conduct a controlled experiment using a single land-use dimension, namely building coverage ratio. This dimension is chosen because building coverage ratio indicates the intensity of land development, which approximates sustainability, compactness, and even urban innovations. Here we use both language prompts that explicitly specify the target density level as “low-density”, “medium-density”, or “high-density development”, together with numeric targets across three scenarios in which the building coverage ratio is set to be 0, 0.35, and 0.6. As shown in Figure 7, the top row shows the generated satellite imagery, and the bottom row presents the corresponding land-use maps together with the building coverage ratio computed from the generated outputs.

The results illustrate the success of such uni-dimensional generative scenario planning since the land-use maps do respond to various density specifications. In the low-density scenario, where the building coverage ratio is set to 0, no residential buildings are generated. The land-use map is dominated by vegetated and open areas, and the satellite imagery reflects continuous green surfaces with minimal built structures. Roads remain sparse and primarily define the outer boundary of the site rather than organizing internal blocks. In the medium-density scenario, where the building coverage ratio is set to 0.35, residential buildings begin to cluster along internal road segments. Green areas are reduced and redistributed between building parcels, forming a balanced configuration between developed and open spaces. The road network becomes more structured and subdivides the site into smaller plots, supporting a recognizable neighborhood layout. The measured building coverage ratio from the generated land-use map is 0.36, closely matching the target value specified in the prompt. In the high-density scenario, where the building coverage ratio is set to 0.6, most land is occupied by residential buildings. Green areas are compressed into narrow bufers and residual patches. Roads assume a stronger organizing function, dividing the site into compact blocks that accommodate dense building footprints. The evaluated building coverage ratio reaches 0.61, again closely aligned with the target density. Across all three density levels, the diference between the specified and measured building coverage ratio remains minimal.

Prompt “A realistic image of an urban area in Orlando. The urban form shows building coverage ratio {x}, where {density description}…"  
![](images/f13fe7d0833dacf2202f4e0e833abd96c15997a0520d71197210e326c2ba8586.jpg)  
Building coverage ratio from generated land-use maps  
Figure 7: Generating land use and satellite imagery for various building coverage ratios

With such uni-dimensional control, we also observe high randomness and diversity in generated outcomes. Using the same prompt, DualControlNet can produce multiple land-use maps and satellite images that satisfy the same density condition, and yet exhibit substantially diferent spatial and functional configurations. As shown in Figure 8, DualControlNet produces three visually and functionally diferentiated scenarios while maintaining the same building coverage ratio. In Figure 8a, the generated satellite image exhibits smaller buildings with detached layouts and roof colors typical of single-family housing. The corresponding land-use map reflects this morphology through spatially distributed residential parcels. In Figure 8b, the built structures become moderately sized with more varied shapes and roof tones, forming a denser and more heterogeneous configuration consistent with commercial development. In Figure 8c, the built environment is dominated by large and simple building footprints with uniform roofs, visually resembling industrial facilities. In each case, the satellite imagery and land use map maintain cross-modal coherence, while the generated outputs reflect distinct functional identities.

(a) Residential  
(b) Commercial  
(c) Industrial  
![](images/fba6e5258198e1f0aecfd2a6b65393ded7c12f5c9605fe1c2fde2596e1e0b4d1.jpg)  
Figure 8: Diverse generated land-use maps and satellite images under the same prompts.

This capability contrasts with traditional predictive land use models, which operate primarily on numeric transitions of land use categories and cannot generate highly diverse visual representations. Predictive models can estimate how much land may convert from one category to another, but they do not synthesize the corresponding building forms, roof structures, or spatial textures visible in remote sensing imagery, let alone the highly diverse visual typologies that may emerge from simple textual prompts. In contrast, our generative land-use modeling framework simultaneously produces satellite imagery and land use maps, conditioning on both quantitative density specifications and qualitative descriptions.

## 5.4 Generative scenario planning with multiple land-use dimensions

Planning practice rarely pursues a single land-use objective in isolation. For example, a density target is typically pursued together with a diversity target, since compact development without a mix of functions produces neither walkable neighborhoods nor local access to jobs and services. Generative scenario planning is therefore useful to planners only if it can respond to several landuse dimensions at once. To test this capability, we jointly control building coverage ratio and the number of land use categories to simulate a sustainable development scenario. We deliberately select relatively underdeveloped sites in both Chicago and Orlando: sites with near-zero built coverage and only two or three land use categories in the ground truth. These sites provide the largest headroom for testing whether the model can generate a sustainable development scenario under multidimensional land-use controls, namely one with higher land-use density and higher land-use diversity than the current condition.

Prompt “A realistic image of an urban area in Chicago with jointly adjusted land use metrics including building coverage ratio {x}, … land use categories $\lbrace \rbrace \rbrace \dots \ "$

x̂ =0.00  
(a) Ground Truth  
![](images/7b2385dd7c4a2dc4821d8e5f1addab2317f04f92f70323b5691c23e7aa739529.jpg)

(b) Original prompt  
![](images/65737b56ce5b19c8e70fa29d8da4458ba3eebf3361f2a334c2515826f6c8d013.jpg)

(c) Edited prompt  
![](images/eec8736ccc59be4ddc31570a34b40a260883e05453dc7f3108993e3e0e8d7db8.jpg)

![](images/6a14fd8e1163e4e5c9943262ea4f49c0376adc3cc7fd817a5c3dc3b32c0d45bd.jpg)

![](images/82ad4fceda5f2b9e3cf62e7cb71bc730790b8cf98b2972851225d7ffa98e650b.jpg)  
ŷ=3

![](images/9b4f558580d40ecaeac7dd17fca909e1c82c2ea189e2342072c374e5a96105c0.jpg)  
ŷ=7  
Figure 9: Generative scenario planning in Chicago under joint adjustment of multiple land-use metrics. (a) Ground truth. (b) Original prompt. (c) Edited Prompt. BCR = building coverage ratio; LUC = land use categories. Both are calculated from generated land use maps.

Figure 9 presents the Chicago site. Figure 9a shows the ground truth, in which the satellite image contains a river corridor with adjacent open land and few built structures, and the corresponding land-use map records three land use categories with large contiguous areas. Figure 9b shows the scenario generated under the original prompt, which specifies a building coverage ratio of 0.00 and three land use categories. The generated outputs return a measured building coverage ratio of 0.00 and three land use categories, reproducing the specified inputs.

Figure 9c presents the sustainable development scenario under multidimensional land-use controls. The measured building coverage ratio rises to 0.30 and the number of land use categories increases from three to seven, confirming quantitative alignment between the planning objectives and the generated outputs. Spatially, a commercial zone emerges along the riverside and forms a dominant built mass that reorganizes the waterfront. Building footprints expand and convert previously open terrain into compact development parcels. Surrounding this commercial core, apartment-style residential neighborhoods appear in organized blocks supported by a more structured road layout that subdivides the site into accessible parcels. The gain in diversity is expressed through functional zones beyond the commercial core, while green spaces and water bodies remain preserved along the river edge, maintaining environmental continuity under denser development. This combination of intensified development and preserved environmental bufers is precisely the trade-of that planners negotiate in waterfront redevelopment, and the model renders it as a visible and measurable land-use map rather than an abstract target.

Prompt “A realistic image of an urban area in Orlando with jointly adjusted land use metrics including building coverage ratio {x}, … land use categories $\{ y \} \ldots "$

Generated Satellite Image

(a) Ground Truth  
![](images/eff42eddd91fd573ddb8d0a331442208bbbc22f7f9d51f1e62d4724626370873.jpg)

(b) Original prompt  
![](images/ca7dde3418ffa68d3eca600683813ad32c66c37934dcc1dfa47192afcb0e62c9.jpg)

(c) Edited prompt  
![](images/ffe0f02ec30420cb41ce08462d0662b108c05f992bd3b4c49eba9c37e6522ac4.jpg)

![](images/bef4c4408b2169f002709f3e0f39dec4483d445dd7712b8d0038605768f88753.jpg)

![](images/cb5054d3ca9369025700199fb01fded02f0adaef3a28edbe87d7c988660ea1c1.jpg)  
ŷ=2

![](images/621cd7a665d80c2dcc003eb7e91f44614e4f22c3d275e9f8cc53ad02a4da7ff9.jpg)  
ŷ=6  
Figure 10: Generative scenario planning in Orlando under joint adjustment of multiple land-use metrics. (a) Ground truth. (b) Original prompt. (c) Edited Prompt. BCR = building coverage ratio; LUC = land use categories. Both are calculate from generated land use maps.

Figure 10 presents the Orlando site. Figure 10a shows the ground truth, in which the satellite image contains grassland or meadow areas adjacent to water bodies with few built structures, and the land-use map records two land use categories. Figure 10b shows the scenario generated under the original prompt, which specifies a building coverage ratio of 0.00 and two land use categories, and the generated outputs return the same values. The generated satellite image renders the vegetated surface with a woodland texture rather than the lighter grassland appearance of the ground truth, and the category boundaries in the generated land-use map do not align exactly with the subtle texture transitions in the satellite image. This slight inconsistency suggests that the model may generalize vegetated areas toward dominant patterns in the training data, indicating a limitation in fine-grained generation.

Figure 10c presents the sustainable development scenario under multidimensional land-use controls, in which the building coverage ratio prompt increases from 0.00 to 0.45 and the number of land use categories from two to five. The measured building coverage ratio in the generated landuse map reaches 0.46, closely matching the specified density target, while the measured number of land use categories reaches six, slightly exceeding the specified diversity target. The generated outputs organize the new development into two distinct zones. In the upper portion of the site, the satellite image shows a compact cluster of mid-sized flat-roofed buildings served by paved lots and access drives, and the land-use map represents this zone as a dominant commercial polygon that occupies most of the upper and central area. In the lower portion, the satellite image shows largeblock warehouse structures with wide rectangular footprints set behind a road corridor, indicating a logistics or light-industrial typology. This land-use outcome is highly relevant to planning because it concentrates employment uses on previously undeveloped land while retaining an environmental bufer around the water, which is the spatial trade-of planners negotiate when siting commercial and industrial capacity.

But we also observe some cross-modal misalignment in such multidimensional scenario generation. As shown in Figure 10c, the land-use map assigns the lower portion of the site to a green category, whereas the corresponding satellite image renders that same area as large-block warehouse facilities. The two modalities therefore disagree on the functional identity of the lower blocks. This misalignment suggests that the model might fail to propagate the built structures into the semantic categories of the land-use map, and it reinforces the need for the cross-modal validation that our land-use and spatial consistency metrics are designed to support.

Overall, our sustainable land-use development generation under multidimensional land-use controls demonstrates that DualControlNet can shape density, functional composition, and spatial configuration simultaneously. The generated outputs align with the prespecified numeric targets in both Chicago and Orlando, and they remain verifiable through the same land-use metrics in the prompts. This closes an important loop in the practice of scenario planning: a planner states multidimensional development goals, the model returns a spatially coherent and visually inspectable land-use map, and the map is then checked against the initial planning goals rather than generalpurpose visual realism alone.

## 6 Discussion

Our study demonstrates DualControlNet as a generative land-use model, as opposed to the predominant predictive land-use models. Empirically, it synthesized land-use maps that do not exist in the training data and yet match the prescribed planning objectives for both Chicago and Orlando. Such a generative land-use model ofers several capabilities that predictive land-use models cannot ofer. First, it is controlled by textual prompts, so a planner can specify “high-density development” rather than a numeric transition matrix. Second, it directly returns visual outcomes, namely a land-use map paired with satellite-like imagery, which renders an abstract density target as an intuitive urban map. Third, it can generate diverse outcomes: under an identical density prompt, the model produced single-family, commercial, and industrial morphologies (Figure 8), presenting planners with a portfolio of alternatives rather than a single predicted trajectory. Predictive landuse models, whether cellular automata, vector-based, or learning-based (He et al., 2023; Cruz-Bello et al., 2024; Guan et al., 2024; Yao et al., 2024; Cai et al., 2026), forecast numeric transitions among existing land-use categories and cannot synthesize the final land-use maps and satellite imagery. Therefore, we complement the predictive models with a generative one, shifting the forecasting question to envisioning, rendering, and comparing multiple land-use scenarios.

Our study also demonstrates a control-verification cycle in the generation process. In DualControlNet, land-use metrics serve a dual role. First, they are control factors: embedded in textual prompts, they specify the planning objective that guides generation. Second, they are verification criteria: because the model returns a land-use map rather than satellite imagery alone, the same metrics can be recomputed from the generated output and compared against the initial objectives. The generated land-use maps recovered the prescribed density and diversity targets in both Chicago and Orlando, closing a cycle among planning objectives, generation, and post-generation verification. A satellite-only model cannot close this cycle, because imagery without semantic land-use categories yields no land-use metric for verification, however realistic it appears. We suggest that this cycle of controlled and verified generation should extend beyond land use. It applies to any domain whose objectives can be expressed as interpretable metrics that are both specifiable in a prompt and recomputable from the generated artifact. Examples can include connectivity and accessibility in transportation network design, floor area ratio in building design, or even habitat connectivity in ecological planning. In each case, the generated outcomes are verified against the domain’s own criteria rather than generic metrics (Castro Lopes et al., 2025; Ye et al., 2022a; Abdollahi et al., 2020). In short, we seek to make domain-specific metrics serve as both the control interface and the verification standard. A generative approach without domain-specific verification wouldn’t be trustworthy for planning practice.

The multimodal design of DualControlNet carries further implications for planning practice. The two generated outcomes are complementary: land-use maps describe functional categories and provide semantic structure, while satellite imagery conveys texture, morphology, building patterns, and other visual characteristics. This multimodal generation matters because the same land-use category can contain substantial morphological heterogeneity. As shown in Figure 8, under an identical building coverage prompt, the model can generate three typologies, all satisfying the same numeric target. A predictive model reporting the same coverage ratio would collapse these three futures into one number, let alone the subtle diferences conveyed through satellite imagery. By jointly generating both layers, DualControlNet represents visual richness while retaining a semantic anchor for verification, extending beyond prior generative urban research (Ye et al., 2022b; Quan, 2022; Jiang et al., 2024). The geospatial consistency loss holds the two layers together. In fact, other consistency losses are also possible, and as urban systems always consist of multiple interdependent layers, future work may explore additional cross-modal constraints across transportation, building, land use, and environmental layers.

Beyond technical aspects, the generative land-use models could lower barriers to public participation and planning communication. Predictive land-use models have long been used to support public engagement, but they are often dificult for non-experts to interpret. Such models require extensive data preparation, calibration, and parameter tuning that can exceed the institutional capacity of planning agencies (Vonk et al., 2005; Vonk and Geertman, 2008). On the contrary, the prompt-based generation of land-use maps ofers a lower-friction entry point: a stakeholder can state “high-density development” and receive a visual land-use map within seconds. This implication is consistent with prior work that argues for AI’s potential to broaden participation by helping non-experts generate and compare design alternatives (Quan, 2022; Jiang et al., 2024). Residents, policymakers, and community groups may not easily interpret land-use entropy, intersection density, or building coverage ratios, but they can readily discuss a rendered satellite image and its corresponding land-use map. Therefore, generative AI can thus support deliberation by translating abstract planning indicators into visual representations.

Several limitations are worth acknowledging. First, the generated outcome pairs sometimes show cross-modal mismatches. In the multidimensional generation, the land-use map assigns the lower blocks to a green category while the satellite image renders the same area as warehouse facilities (Figure 10c). We also observe semantic confusion in transition areas, where wetland or water-adjacent spaces are represented as built-up or industrial land uses with irregular building blocks. These limitations actually echo land-use simulation research, which shows that parce structure, within-parcel variation, and neighborhood relationships afect spatial validity (Guan et al., 2024; Yao et al., 2024). Second, our land-use labels are derived from OpenStreetMap, which is not suficiently fine-grained in undeveloped areas. As a result, the input labels can be inconsistent across regions. Finer sources such as oficial plans, parcel maps, and points of interest would improve label fidelity. Third, the dual generation pipeline is not fully stable, occasionally producing map flips and orientation errors, and generated tiles are not continuous across tile boundaries. These limitations suggest that future work should explore additional cross-modal constraints, boundaryaware generation, and more robust training strategies to improve the quality of land-use map generations.

## 7 Conclusions

Land-use patterns shape urban sustainability, health, and safety. However, existing land-use models are largely predictive, forecasting incremental transitions among established land-use categories from historical data, while lacking the capacity to envision land-use patterns that do not yet exist. Meanwhile, existing generative models remain unimodal, weakly controllable, and evaluated mainly by generic image metrics, being insuficient for land-use scenario planning. To address these gaps, we developed DualControlNet, a generative land-use model that jointly synthesizes land-use maps and satellite imagery, controls generation with interpretable land-use metrics such as building coverage ratio and land-use diversity, and verifies the generated outputs against the same metrics. We trained the model on more than 40,000 paired tiles from Orlando and Chicago and evaluated it along land-use, spatial consistency, and visual realism dimensions. This study yields the following findings.

First, joint multimodal generation substantially improves the consistency between the generated land-use maps and the targeted land-use metrics. Relative to the unimodal ControlNet baselines, DualControlNet raises the consistency in land-use diversity by 62.8%, in land-use area by 49.5%, in building coverage ratio by 16.3%, and in distribution similarity by 5.9%. Notably, the satellite-only baseline cannot be verified on any land-use metric, because imagery alone carries no semantic landuse categories. Second, the geospatial consistency loss strengthens cross-modal alignment. Both DualControlNet variants roughly double the cross-modal alignment of the unimodal baselines, and adding the loss further improves gradient-based cosine similarity and direction distribution similarity, while also lowering the Fr´echet Inception Distance. Multimodal joint generation thus enhances not only cross-modal coherence but also perceptual image quality. With the technical achievements, we conducted scenario-based generative experiments demonstrating that DualControlNet responds to both single and multiple land-use controls. Under single-dimensional control, the BCR measured from the generated land-use maps tracks the prompted targets closely, while still producing residential, commercial, and industrial morphologies under an identical density prompt. Under multidimensional control, the model raises density and diversity together at underdeveloped sites in both Chicago and Orlando, generating riverside commercial cores and employment clusters that meet the specified targets while preserving environmental bufers. Overall, these findings show that land-use metrics can serve simultaneously as the control interface and the verification standard for generative land-use modeling.

This study suggests that generative AI can complement predictive land-use modeling: it shifts the planning question from forecasting a single trajectory to envisioning, rendering, and comparing multiple land-use maps. It potentially lowers the technical barrier by allowing stakeholders to state an objective in plain language and inspect a visually explicit scenario within seconds. Several limitations remain, as discussed in Section 6: residual cross-modal misalignment, coarse land-use labels in undeveloped areas, and discontinuity of the generated tiles at their boundaries. Future work should fold these failure modes into the control–verification loop by developing boundary-aware, topology-aware, and semantics-aware verification mechanisms, and by extending the framework to finer land-use sources and additional cities. Overall, this study presents a generative land-use modeling framework that is multimodal, controllable, and verifiable, and it demonstrates the potential of generative AI to support urban planning practice.

## Data and Codes Availability Statement

The anonymized data and Python scripts that support the findings of this study are available in a GitHub repository:https://github.com/qingqisqq/DualControlNet.

## References

Abdollahi, A., Pradhan, B., Gite, S., Alamri, A., 2020. Building footprint extraction from high resolution aerial images using generative adversarial network (gan) architecture. IEEE Access 8, 209517–209527.

Abou Jaoude, G., Mumm, O., Carlow, V.M., 2022. An overview of scenario approaches: a guide for urban design and planning. Journal of planning literature 37, 467–487.

Aditjandra, P.T., Cao, X.J., Mulley, C., 2016. Exploring changes in public transport use and walking following residential relocation. Journal of Transport and Land Use 9, 77–95. URL: http://www.jstor. org/stable/26203231. publisher: Journal of Transport and Land Use.

Bartholomew, K., 2007. Land use-transportation scenario planning: promise and reality. Transportation 34, 397–412. URL: http://link.springer.com/10.1007/s11116-006-9108-2, doi:10.1007/ s11116-006-9108-2.

Boarnet, M.G., 2011. A Broader Context for Land Use and Travel Behavior, and a Research Agenda. Journal of the American Planning Association 77, 197–213. URL: http://www.tandfonline.com/doi/abs/10. 1080/01944363.2011.593483, doi:10.1080/01944363.2011.593483.

Boeing, G., 2017. OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems 65, 126–139. URL: https://linkinghub. elsevier.com/retrieve/pii/S0198971516303970, doi:10.1016/j.compenvurbsys.2017.05.004.

Cai, Z., Karvonen, A., Cong, C., Huang, W., 2026. Fine-grained urban land use simulation: Integrating spatial dynamic modeling with a pre-trained vision-language model. Computers, Environment and Urban Systems 126.

Carmona, M., 2021. Public Places Urban Spaces: The Dimensions of Urban Design. 3 ed., Routledge. URL: https://www.taylorfrancis.com/books/9781351656627, doi:10.4324/9781315158457.

Castro Lopes, J., Oliveira, J.L., Lopes, R.P., 2025. A comprehensive review of synthetic image generation methods in remote sensing. International Journal of Remote Sensing , 1–29.

Cervero, R., Kockelman, K., 1997. Travel demand and the 3Ds: Density, diversity, and design. Transportation Research Part D: Transport and Environment 2, 199–219. URL: https://linkinghub.elsevier. com/retrieve/pii/S1361920997000096, doi:10.1016/S1361-9209(97)00009-6.

Chen, H., Shen, B., Liu, Y., Shi, R., Zhou, L., Lin, C.Z., Gu, J., Su, H., Wetzstein, G., Guibas, L., 2025. 3D-Adapter: Geometry-Consistent Multi-View Difusion for High-Quality 3D Generation. URL: http://arxiv.org/abs/2410.18974, doi:10.48550/arXiv.2410.18974. arXiv:2410.18974.

Chen, J., Hansen-Estruch, P., Han, X., Hu, Y., Dinan, E., Kamath, A., Drozdzal, M., Askari-Hemmat, R., Zettlemoyer, L., Ghazvininejad, M., 2026. Unified Text-Image Generation with Weakness-Targeted Post-Training. URL: http://arxiv.org/abs/2601.04339, doi:10.48550/arXiv.2601.04339. arXiv:2601.04339 version: 2.

Chung, H., Sim, B., Ye, J.C., 2022. Come-closer-difuse-faster: Accelerating conditional difusion models

for inverse problems through stochastic contraction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12413–12422.

Couclelis, H., 2005. “Where has the Future Gone?” Rethinking the Role of Integrated Land-Use Models in Spatial Planning. Environment and Planning A: Economy and Space 37, 1353–1371. URL: https: //journals.sagepub.com/doi/10.1068/a3785, doi:10.1068/a3785.

Croitoru, F.A., Hondru, V., Ionescu, R.T., Shah, M., 2023. Difusion Models in Vision: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 10850–10869. URL: https://ieeexplore.ieee.org/document/10081412/;jsessionid= sHOhp2ganK6PipOGpDX20kb-JN45Tys5Garxt7DJBXOlwQd1TbYt!20310694, doi:10.1109/TPAMI.2023. 3261988.

Cruz-Bello, G.M., Romero-Sanchez, M.E., Galeana-Pizana, J.M., 2024. Can regionalization enhance the performance of land-use change models in rapidly urbanizing areas? Environment and Planning B: Urban Analytics and City Science .

Dedring, T., Rienow, A., 2024. Synthesis and evaluation of seamless, large-scale, multispectral satellite images using generative adversarial networks on land use and land cover and sentinel-2 data. GIScience & Remote Sensing 61, 2364460.

Dhariwal, P., Nichol, A., 2021. Difusion Models Beat GANs on Image Synthesis, in: Advances in Neura Information Processing Systems, Curran Associates, Inc.. pp. 8780–8794. URL: https://proceedings. neurips.cc/paper\_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.

Ewing, R., Cervero, R., 2010. Travel and the Built Environment: A Meta-Analysis. Journal of the American Planning Association 76, 265–294. URL: http://www.tandfonline.com/doi/abs/10.1080/ 01944361003766766, doi:10.1080/01944361003766766.

Farahani, R.Z., Miandoabchi, E., Szeto, W.Y., Rashidi, H., 2013. A review of urban transportation network design problems. European journal of operational research 229, 281–302. Publisher: Elsevier.

Gu, Z., Luo, X., Chen, Y., Liu, X., Xiao, C., Liang, Y., 2022. Density, Diversity, and Design: Evaluating the Equity of the Elderly Communities in Three Measures of the Built Environment. Land 11, 1976. URL: https://www.mdpi.com/2073-445X/11/11/1976, doi:10.3390/land11111976.

Guan, Q., Li, J., Zhai, Y., Liang, X., Yao, Y., 2024. Hashgat-vca: A vector cellular automata model with hash function and graph attention network for urban land-use change simulation. Landscape and Urban Planning 250, 105145.

Hayakawa, A., Ishii, M., Shibuya, T., Mitsufuji, Y., 2025. MMDisCo: Multi-Modal Discriminator-Guided Cooperative Difusion for Joint Audio and Video Generation. URL: http://arxiv.org/abs/2405. 17842, doi:10.48550/arXiv.2405.17842. arXiv:2405.17842.

He, F., Yang, J., Zhang, Y., Yu, W., Xiao, X., Xia, J., 2023. Does partition matter? a new approach to modeling land use change. Computers, Environment and Urban Systems 106, 102041.

He, M., Liang, Y., Wang, S., Zheng, Y., Wang, Q., Zhuang, D., Tian, L., Zhao, J., 2026. Humanguided urban form generation using multimodal difusion models. Building and Environment 287, 113892. URL: https://linkinghub.elsevier.com/retrieve/pii/S0360132325013629, doi:10.1016/ j.buildenv.2025.113892.

Hewitt, R.J., Shadman Roodposhti, M., Bryan, B.A., 2022. There’s no best model! Addressing limitations of land-use scenario modelling through multi-model ensembles. International Journal of Geographical Information Science 36, 2352–2385. URL: https://www.tandfonline.com/doi/full/10.1080/13658816. 2022.2098299, doi:10.1080/13658816.2022.2098299.

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851.

Jiang, F., Ma, J., Webster, C.J., Chiaradia, A.J., Zhou, Y., Zhao, Z., Zhang, X., 2024. Generative urban design: A systematic review on problem formulation, design generation, and decision-making. Progress in Planning 180, 100795. URL: https://linkinghub.elsevier.com/retrieve/pii/S0305900623000569, doi:10.1016/j.progress.2023.100795.

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T., 2020. Analyzing and Improving the Image Quality of StyleGAN, pp. 8110–8119. URL: https://openaccess.thecvf.com/content\_ CVPR\_2020/html/Karras\_Analyzing\_and\_Improving\_the\_Image\_Quality\_of\_StyleGAN\_CVPR\_2020\_ paper.html.

Kim, Y.A., Hipp, J.R., 2021. Density, diversity, and design: Three measures of the built envi-

ronment and the spatial patterns of crime in street segments. Journal of Criminal Justice 77, 101864. URL: https://linkinghub.elsevier.com/retrieve/pii/S0047235221000842, doi:10.1016/ j.jcrimjus.2021.101864.

Labiosa, W., Forney, W., Esnard, A.M., Mitsova-Boneva, D., Bernknopf, R., Hearn, P., Hogan, D., Pearlstine, L., Strong, D., Gladwin, H., Swain, E., 2013. An integrated multi-criteria scenario evaluation web tool for participatory land-use planning in urbanized areas: The Ecosystem Portfolio Model. Environmental Modelling & Software 41, 210–222. URL: https://linkinghub.elsevier.com/retrieve/pii/ S1364815212003118, doi:10.1016/j.envsoft.2012.10.012.

Li, P., Ma, S., Chen, J., Liu, Y., Zhang, C., Xue, W., Luo, W., Shefer, A., Wang, W., Guo, Y., 2025a. CMD: Controllable Multiview Difusion for 3D Editing and Progressive Generation, in: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, ACM, Vancouver BC Canada. pp. 1–10. URL: https://dl.acm.org/doi/10.1145/3721238.3730722, doi:10.1145/3721238.3730722.

Li, X., Shi, L., Tang, J., Li, J., Zhao, P., Liu, Q., Chen, J., Ma, C., 2024. Efects of the built environment on travel distance in bus-oriented, medium-sized cities in China. Journal of Transport and Land Use 17, 423– 455. URL: https://jtlu.org/index.php/jtlu/article/view/2427, doi:10.5198/jtlu.2024.2427.

Li, Z., Li, H., Shi, Y., Farimani, A.B., Kluger, Y., Yang, L., Wang, P., 2025b. Dual Difusion for Unified Image Generation and Understanding. URL: http://arxiv.org/abs/2501.00289, doi:10.48550/arXiv. 2501.00289. arXiv:2501.00289.

Lopane, F.D., Kalantzi, E., Milton, R., Batty, M., 2023. A land-use transport-interaction framework for large scale strategic urban modeling. Computers, Environment and Urban Systems 104, 102007. doi:10. 1016/j.compenvurbsys.2023.102007.

Lowry, J.H., Lowry, M.B., 2014. Comparing spatial metrics that quantify urban form. Computers, Environment and Urban Systems 44, 59–67.

Lynch, K., 2008. The image of the city. Publication of the Joint Center for Urban studies. 33. print ed., M.I.T. Press, Cambridge, Mass.

Lyu, G., Angkawisittpan, N., Fu, X., Sonasang, S., 2025. Investigating the relationship between built environment and urban vitality using big data. Scientific Reports 15, 579. URL: https://www.nature. com/articles/s41598-024-84279-2, doi:10.1038/s41598-024-84279-2.

Molinero-Parejo, R., Aguilera-Benavente, F., G´omez-Delgado, M., Shurupov, N., 2023. Combining a land parcel cellular automata (LP-CA) model with participatory approaches in the simulation of disruptive future scenarios of urban land use change. Computers, Environment and Urban Systems 99, 101895. URL: https://linkinghub.elsevier.com/retrieve/pii/S0198971522001399, doi:10.1016/ j.compenvurbsys.2022.101895.

Nichol, A.Q., Dhariwal, P., 2021. Improved Denoising Difusion Probabilistic Models, in: Proceedings of the 38th International Conference on Machine Learning, PMLR. pp. 8162–8171. URL: https:// proceedings.mlr.press/v139/nichol21a.html.

Quan, S.J., 2022. Urban-GAN: An artificial intelligence-aided computation system for plural urban design. Environment and Planning B: Urban Analytics and City Science 49, 2500–2515. ISBN: 2399-8083 Publisher: SAGE Publications Sage UK: London, England.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I., 2021. Learning Transferable Visual Models From Natural Language Supervision, in: Proceedings of the 38th International Conference on Machine Learning, PMLR. pp. 8748–8763. URL: https://proceedings.mlr.press/v139/radford21a.html.

R¨ath, Y.M., Grˆet-Regamey, A., van Strien, M.J., 2026. High-resolution urban land use change modeling via sequential classifiers. Computers, Environment and Urban Systems 117, 102400. doi:10.1016/j. compenvurbsys.2026.102400.

Rojas, K., Zhu, Y., Zhu, S., Ye, F.X.F., Tao, M., 2025. Difuse Everything: Multimodal Difusion Models on Arbitrary State Spaces. URL: http://arxiv.org/abs/2506.07903, doi:10.48550/arXiv.2506.07903. arXiv:2506.07903 version: 2.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes,

R., Karagol Ayan, B., Salimans, T., Ho, J., Fleet, D.J., Norouzi, M., 2022. Photorealistic Text-to-Image Difusion Models with Deep Language Understanding. Advances in Neural Information Processing Systems 35, 36479–36494. URL: https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html.

Sant´e, I., Garc´ıa, A.M., Miranda, D., Crecente, R., 2010. Cellular automata models for the simulation of real-world urban processes: A review and analysis. Landscape and Urban Planning 96, 108– 122. URL: https://linkinghub.elsevier.com/retrieve/pii/S0169204610000472, doi:10.1016/j. landurbplan.2010.03.001.

Saxena, D., Cao, J., 2021. Generative Adversarial Networks (GANs): Challenges, Solutions, and Future Directions. ACM Computing Surveys 54, 63:1–63:42. URL: https://doi.org/10.1145/3446374, doi:10.1145/3446374.

Schoemaker, P.J.H., 1991. When and how to use scenario planning: A heuristic approach with illustration. Journal of Forecasting 10, 549–564. URL: https://onlinelibrary.wiley.com/doi/10.1002/ for.3980100602, doi:10.1002/for.3980100602.

Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X., 2024. MVDream: Multi-view Difusion for 3D Generation. URL: http://arxiv.org/abs/2308.16512, doi:10.48550/arXiv.2308.16512. arXiv:2308.16512.

Sun, K., Liang, Y., He, M., Zheng, Y., Prakash, A., Wang, S., Zhao, J., Pentland, A.S., 2026. Envisioning global urban development with satellite imagery and generative AI. URL: http://arxiv.org/abs/ 2603.26831, doi:10.48550/arXiv.2603.26831. arXiv:2603.26831.

Tepe, E., 2023. History, neighborhood, and proximity as factors of land-use change: A dynamic spatial regression model. Environment and Planning B: Urban Analytics and City Science 51, 7–22. doi:10. 1177/23998083231164397.

Van Notten, P.W., Rotmans, J., Van Asselt, M.B., Rothman, D.S., 2003. An updated scenario typology. Futures 35, 423–443. URL: https://linkinghub.elsevier.com/retrieve/pii/S0016328702000903, doi:10.1016/S0016-3287(02)00090-3.

Van Vliet, J., Bregt, A.K., Brown, D.G., Van Delden, H., Heckbert, S., Verburg, P.H., 2016. A review of current calibration and validation practices in land-change modeling. Environmental Modelling & Software 82, 174–182. URL: https://linkinghub.elsevier.com/retrieve/pii/S1364815216301128, doi:10.1016/j.envsoft.2016.04.017.

Vonk, G., Geertman, S., 2008. Improving the Adoption and Use of Planning Support Systems in Practice. Applied Spatial Analysis and Policy 1, 153–173. URL: http://link.springer.com/10.1007/ s12061-008-9011-7, doi:10.1007/s12061-008-9011-7.

Vonk, G., Geertman, S., Schot, P., 2005. Bottlenecks Blocking Widespread Usage of Planning Support Systems. Environment and Planning A: Economy and Space 37, 909–924. URL: https://journals. sagepub.com/doi/10.1068/a3712, doi:10.1068/a3712.

Wang, J., Hadjikakou, M., Hewitt, R.J., Bryan, B.A., 2022. Simulating large-scale urban land-use patterns and dynamics using the U-Net deep learning architecture. Computers, Environment and Urban Systems 97, 101855. doi:10.1016/j.compenvurbsys.2022.101855.

Wang, Q., Liang, Y., Zheng, Y., Xu, K., Zhao, J., Wang, S., 2025a. Generative ai for urban planning: Synthesizing satellite imagery via difusion models. arXiv preprint arXiv:2505.08833 .

Wang, Z., Feng, T., Safikhani, A., Tepe, E., 2025b. Enhancing transparency in land use change modeling: Leveraging explainable AI techniques for urban growth prediction with spatially distributed insights. Computers, Environment and Urban Systems 121, 102322. doi:10.1016/j.compenvurbsys. 2025.102322.

Weng, M., Ding, N., Li, J., Jin, X., Xiao, H., He, Z., Su, S., 2019. The 15-minute walkable neighborhoods: Measurement, social inequalities and implications for building healthy communities in urban China. Journal of Transport and Health 13, 259–273. URL: https://www.sciencedirect.com/science/article/ pii/S2214140518305103, doi:10.1016/j.jth.2019.05.005.

Wu, A.N., Biljecki, F., 2022. Ganmapper: geographical data translation. International Journal of Geographical Information Science 36, 1394–1422.

Wu, A.N., Stoufs, R., Biljecki, F., 2022. Generative Adversarial Networks in the built environment: A comprehensive review of the application of GANs across data types and scales. Building and Environment 223, 109477. URL: https://linkinghub.elsevier.com/retrieve/pii/S0360132322007089, doi:10. 1016/j.buildenv.2022.109477.

Xu, X., Ding, D., Liu, X., 2024. A three-dimensional future land use simulation (FLUS-3D) model for simulating the 3D urban dynamics under the shared socio-economic pathways. Landscape and Urban Planning 250, 105135. doi:10.1016/j.landurbplan.2024.105135.

Yang, J., Zhu, X., Chen, W., Sun, Y., Zhu, J., 2023. Modeling land-use change using partitioned vector cellular automata while considering urban spatial structure. Environment and Planning B: Urban Analytics and City Science 50, 2273–2293. doi:10.1177/23998083231152887.

Yao, Y., Jiang, Y., Sun, Z., et al., 2024. Applicability and sensitivity analysis of vector cellular automata model for land cover change. Computers, Environment and Urban Systems 109, 102090.

Ye, X., Du, J., Ye, Y., 2022a. Masterplangan: Facilitating the smart rendering of urban master plans via generative adversarial networks. Environment and Planning B: Urban Analytics and City Science 49, 794–814.

Ye, X., Du, J., Ye, Y., 2022b. MasterplanGAN: Facilitating the smart rendering of urban master plans via generative adversarial networks. Environment and Planning B: Urban Analytics and City Science 49, 794–814. URL: http://journals.sagepub.com/doi/10.1177/23998083211023516, doi:10.1177/ 23998083211023516.

Zeng, C., Yao, Y., Liu, J., Sun, Z., Zhou, K., Chen, D., Guan, Q., 2025. CoCA: Spatial cooperative simulation and future prediction of “land-population-economy” in urban agglomerations. Landscape and Urban Planning 263, 105442. doi:10.1016/j.landurbplan.2025.105442.

Zhang, L., Rao, A., Agrawala, M., 2023. Adding Conditional Control to Text-to-Image Difusion Models, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3836–3847. URL: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang\_Adding\_ Conditional\_Control\_to\_Text-to-Image\_Diffusion\_Models\_ICCV\_2023\_paper.html.

Zhang, L., Ye, Y., Zeng, W., Chiaradia, A., 2019. A Systematic Measurement of Street Quality through Multi-Sourced Urban Data: A Human-Oriented Analysis. International Journal of Environmental Research and Public Health 16, 1782. URL: https://www.mdpi.com/1660-4601/16/10/1782, doi:10.3390/ijerph16101782.

Zhuang, Y., Kang, Y., Fei, T., Bian, M., Du, Y., 2024. From hearing to seeing: Linking auditory and visua place perceptions with soundscape-to-image generative artificial intelligence. Computers, Environment and Urban Systems 110, 102122. Publisher: Elsevier.

## A Appendix A

## Residential and Urban

residential - #FFCA65 - RGB: (255, 202, 101)

commercial - #FF8080- RGB:(255,128,128)

industrial - #BFBFBF - RGB: (191, 191, 191)

retail - #FF5555 - RGB: (255, 85, 85)

parking - #EFEFEF - RGB: (239, 239, 239)

## Educational and Healthcare

school - #FFF0AA - RGB: (255, 240, 170)

university - #FFF0AA - RGB: (255, 240, 170)

hospital - #FFE6E6 - RGB: (255,230,230)

## Green Spaces and Recreation

park - #B2D8B2 - RGB: (178, 216, 178)

garden - #B2D8B2 - RGB: (178, 216, 178)

recreation\_ground - #B8E6B8 - RGB: (184, 230, 1

playground - #CCE6CC - RGB: (204, 230, 204)

sports\_centre - #CCE6CC - RGB: (204, 230, 204)

stadium - #CCE6CC - RGB: (204, 230, 204)

pitch - #CCE6CC - RGB: (204, 230, 204)

golf\_course - #B2D8B2 - RGB: (178, 216, 178)

## Natural Areas

forest - #8CBF8C - RGB: (140, 191, 140)

wood - #8CBF8C - RGB: (140, 191, 140)

grass - #CCFFCC - RGB: (204, 255, 204)

grassland - #CCFFCC - RGB: (204, 255, 204)

meadow - #CCFFCC - RGB: (204, 255, 204)

heath - #CCE6CC - RGB: (204, 230, 204)

scrub - #B8E6B8 - RGB: (184,230, 184)

wetland - #A6E6E6 - RGB: (166, 230, 230)

water - #B3D9FF - RGB: (179, 217, 255)

beach - #FFF5CC - RGB: (255,245,204)

sand - #FFF5CC - RGB: (255, 245, 204)

## Agricultural

farmland - #FFFFCC - RGB: (255, 255, 204)

orchard - #E6FFB3 - RGB: (230, 255, 179)

vineyard - #E6FFB3 - RGB: (230, 255, 179)

## Other

cemetery - #D1CFCD - RGB: (209, 207, 205)

Figure 11: Land Use Legend Demonstration