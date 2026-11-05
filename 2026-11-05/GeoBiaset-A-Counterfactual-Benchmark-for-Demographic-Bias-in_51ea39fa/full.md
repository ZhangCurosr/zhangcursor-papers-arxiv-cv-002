# GeoBiaset: A Counterfactual Benchmark for Demographic Bias in World-Level Geolocalization

Niccolò Niccoli<sup>1</sup> Silvia Dani<sup>1</sup> Marco Mistretta<sup>1</sup> Andrew D. Bagdanov<sup>1</sup> Marco Bertini<sup>1</sup> Lorenzo Seidenari<sup>1</sup>

<sup>1</sup>University of Florence

![](images/27fd6ec201577199b44d4a182af1e6348667e5adfd7692e62af6dbe1b358dc32.jpg)  
Figure 1: Left: Standard geolocalization benchmarks are imbalanced: scene cues and human attributes co-occur, making it unclear whether models rely on place-specific evidence or spurious demographic correlations. Right: GEOBIASET generates counterfactual edits varying race, gender, and attire under fixed GPS coordinates. Black diamonds indicate predictions for original images; colored circles indicate predictions for edited images, with colors denoting the inserted person’s race category. Displacement toward race-associated regions highlights the audited bias.

## Abstract

World-level image geolocalization is an increasingly important task, yet the biases of current models remain poorly understood. Prior work has identified coarse dataset imbalances noting, for instance, that some cities are absent from training sets and that models tend to perform better in wealthier regions. However, no one has yet examined which visual cues drive these disparities. To do so, we introduce GEOBIASET, the first benchmark designed to measure how geolocalization models are influenced by the apparent race of a person inserted into the image foreground. The dataset consists of 9,541 images, pairing existing geotagged background images with counterfactual edits spanning seven races, holding the ground-truth coordinates fixed. Controlled masking experiments show that occlusion and background drift do not fully explain geolocalization degradation. Alongside the benchmark, we introduce the Person-Origin Perturbation Indicators (POPI), novel metrics that quantify model resilience to these demographic perturbations by measuring whether predictions are pulled toward the region associated with the inserted group. Our evaluation of state-of-the-art geolocalization and multimodal large language models reveals a 16-47% drop in continent-level performance when a subject from a demographically mismatched group is inserted. A landmark versus no-landmark analysis and gradient-weighted heatmaps show that models preferentially attend to geographic landmarks when present, but shift attention to facial features in their absence.

40th Conference on Neural Information Processing Systems (NeurIPS 2026). Track on Evaluations and Datasets.

## 1 Introduction

Visual image geolocalization predicts where in the world a photograph was taken from its visual content alone. Its applications range from privacy auditing of social media imagery to disaster response and digital forensics. Building on the IM2GPS paradigm of matching a query against a large geotagged database [13, 40], modern systems combine contrastive vision-language encoders [39], retrieval-augmented generation [50, 20], and multimodal large language models [21, 18] to achieve strong accuracy on standard benchmarks.

Recent analyses demonstrate that planet-scale geolocalization models exhibit severe geographic bias, systematically favoring developed, Western, and urbanized regions over underrepresented ones [38, 18, 26, 43]. This results in substantial spatial disparities, including regional performance gaps of up to 42% [31]. While these systematic failures are widely attributed to uneven training data densities, prior work primarily quantifies macroscopic spatial distribution shifts without characterizing the specific visual mechanisms that drive individual prediction errors.

This is the gap we address. Faces and attire provide coarse geographic cues [12, 34], but people travel, and apparent origin does not determine where a photo was taken. Standard benchmarks cannot reveal whether models rely on this signal, because geography, culture, clothing, and demographics co-occur in images, so correct predictions do not identify the driving cues. To audit this sensitivity in world-level image geolocalization, we introduce GEOBIASET<sup>1</sup>, a counterfactual benchmark that varies inserted foreground subjects’ demographics while holding ground-truth scene coordinates fixed. Starting from 329 geotagged backgrounds from IM2GPS3k [40] and YFCC4k [37], we generate 28 edited variants by inserting a person from one of 7 mutually exclusive races, combined with 2 gender and 2 attire conditions (unconditioned and race-associated), for a total of 9541 image-annotation pairs. Faces are sampled from a demographically balanced source [22] (see Fig. 1) and than the edited images pass automated screening for severe synthesis failures.

Additionally, we show that existing accuracy metrics are insensitive to the directional structure of these shifts. Therefore, we introduce the Person-Origin Perturbation Indicators (POPI) to quantify the spatial distribution shift induced by inserting racially diverse subjects into the scene. We define three variants: POPI-Attack (POPI-A) measuring the rate at which an inserted subject shifts the prediction toward their associated geographic origin, indicating susceptibility to spurious visual correlations; POPI-Help (POPI-H) quantifying the frequency with which an inserted subject corrects an initially failed baseline prediction to the ground-truth region; and POPI-Distraction (POPI-D) evaluating how often an inserted subject misleads an initially correct prediction away from the ground truth.

We evaluate state-of-the-art geolocalization models (G3, GeoCLIP, Img2Loc) and a suite of openweight MLLMs on GEOBIASET. We observe three consistent patterns. First, demographic foreground subjects act as strong distractors: inserting a mismatched-race subject reduces continent-level accuracy from between 16% to 47% relative to the original backgrounds, and the degradation persists when the background contains a recognized landmark. Second, for a subset of races, inserting a subject whose group matches the ground-truth region produces a measurable accuracy gain at the continent scale, indicating that models do not merely degrade under perturbation but actively use demographic cues as geographic proxies. Finally, gradient-weighted attention analysis reveals a stable pattern in which models attend to landmarks when present and shift reliance to the foreground face in their absence.

Our contributions are as follows:

• We introduce GEOBIASET, a counterfactual benchmark for auditing demographic bias in worldwide image geolocalization via person insertions with fixed GPS labels.

• We propose POPI, a set of perturbation-based metrics that quantify whether demographic interventions attract, help, or distract geolocalization predictions relative to race-associated geographic regions.

• We evaluate seven geolocalization systems and show that foreground demographic cues can substantially reduce accuracy and redirect predictions.

• We analyze attire, face obfuscation, and landmark presence, showing that the effect is not limited to facial appearance and is amplified when scene-level geographic evidence is weak.

## 2 Related Work

Image geolocalization. Image geolocalization infers where a photograph was taken from visual content alone. Early work matched queries against a large geotagged database [13]; deep learning later reformulated the task as classification over discrete geographic cells [45, 40, 33] and revisited retrieval with stronger metric learning [4, 47]. Current systems align images with coordinates in a joint embedding space [39], structure the prediction space through semantically grounded geocells [12], or layer retrieval-augmented and ranking-based reasoning on top of these representations [50, 20, 21]. Generalist MLLMs match specialized models on standard benchmarks without task-specific training [18, 19]. The field has focused on accuracy, robustness to viewpoint, and worldwide coverage. Recently, several studies have begun to focus also on the privacy aspects of the task [27, 29, 44, 46].

Bias in image geolocalization. Social and demographic biases are well-known issues in artificial intelligence, especially from the advent of large multimodal models [17, 24, 49, 11, 10, 9, 35]. A small but growing literature shows that geolocalization models do not err uniformly across the globe. Uribe et al. [38] document large accuracy drops on African imagery, with predictions shifted toward higher-income countries and the Global North. Huang et al. [18] report that VLMs favor economically developed and densely populated regions and tend to over-predict well-known cities. IMAGEO-Bench [26] confirms stronger performance in North America and Western Europe than in underrepresented areas, and KoreaGEO [43] shows that even within a single country predictions collapse toward urban centers. EarthWhere [31] measures gaps of up to 42% between best and worst regions. These studies establish that performance is uneven across regions, countries, development level, and population density. They locate where models fail. They do not characterize what visual content drives prediction errors, which is the question GEOBIASET is built to answer.

Counterfactual auditing of vision-language models. A complementary line of work audits VLMs through paired counterfactual images that vary a sensitive attribute while keeping the rest of the scene fixed. Earlier work used synthesized face images that vary race and gender while holding other visual factors fixed to measure demographic sensitivity in commercial classifiers [23]. SocialCounterfactuals [15] synthesizes such pairs to isolate race and gender effects in VLM outputs, FOCUS [7] applies face-only edits over real photographs to study demographic disparities in occupational judgment, and Cultural Counterfactuals [16] varies cultural framing while holding identity constant. Related synthetic counterfactual face benchmarks further study how controlled demographic variation can support fairness evaluation while reducing confounding [32]. These benchmarks quantify bias in decisions made about the depicted person. We adopt the same experimental logic and apply it to a task in which the prediction target is a property of the place, where the depicted person is in principle irrelevant. This shift in the target is what makes demographic influence in geolocalization a particularly subtle failure mode and what motivates the directional metrics introduced in Sec. 4.

## 3 The GeoBiaset Dataset

Constructing GEOBIASET requires two sources of visual input: geographic background images and face images used to control the demographic appearance of the inserted subject. Background images are manually selected from two standard image geolocalization benchmarks, Im2GPS3k [13] and YFCC4k [37]. These datasets are widely used in global image geolocalization, but they exhibit substantial geographic imbalance, with over-representation of Western regions [8, 3, 38]. To reduce this source imbalance, we first partition the Earth into geographic areas and then sample an equal number of background images from each area after merging the two datasets.

We further filter the selected images to remove cases in which a person is already present in the foreground, such as close-up portraits or images where human appearance could confound the demographic intervention. This filtering step is important because GEOBIASET aims to isolate the effect of the inserted subject rather than measure uncontrolled variation already present in the source image. After selection and filtering, the resulting background set contains 47 images for each of 7 geographic areas, for a total of 329 real images. To define demographic categories in a reproducible manner, we adopt the racial taxonomy used by FairFace [22], which provides balanced annotations over seven categories: Black, East Asian, Indian, Middle Eastern, Latino, Southeast Asian, and White. We use FairFace because it offers an established, externally defined demographic categorization rather than relying on ad hoc labels introduced for this work. Our goal is not to treat these categories as biologically intrinsic, but to use them as socially salient visual attributes that may influence geolocation models.

![](images/b46bff71587a6a5eb28c2fbf708cc1ce33a786feea1d5ce15f185b5d1f370d8b.jpg)  
Figure 2: Overview of the GEOBIASET image-generation pipeline. Starting from a background image, an input face, and an editing prompt that specifies race, gender and optionally race-associated clothing we generate an edited image while preserving the overall geographic scene, without enforcing pixel-level accuracy. A multimodal quality-control module detects severe generation artifacts and triggers regeneration when needed. For each starting background image, we generate 28 edited images (2 genders × 2 attire types × 7 races).

We construct the face pool to isolate race as an independent variable while controlling for other visible attributes, including gender and age. We exclude identities below age 20, both to avoid using minors in bias-auditing interventions and to reduce age-related variation in facial morphology. We then group the remaining images by the variables race, gender, and age group, and manually select 10 high-quality images from each variable combination. This produces a sufficiently diverse pool from which face images can be sampled during generation while limiting excessive repetition of the same identity. By balancing the source face set across demographic groups, we reduce the risk that observed model behavior is driven by imbalance in the generation inputs rather than by demographic sensitivity in the geolocalization models under evaluation.

Editing Pipeline. We design an image-editing pipeline that inserts human subjects into geographic scenes while preserving the visual evidence relevant to geolocalization, such as architecture, vegetation, road layout, terrain, lighting, and urban form. Let x denote a background image and let $G _ { x } = ( \lambda _ { x } , \phi _ { x } )$ be its associated GPS coordinate, where $\lambda _ { x }$ and $\phi _ { x }$ denote longitude and latitude. Let r denote race, $g$ denote gender, and $f$ denote a face image sampled from the balanced face pool. We define the image-editing model as $\Phi _ { g }$ and the multimodal quality-control model as $\Psi _ { g }$ . The complete pipeline forms a closed loop between synthesis and artifact detection, as shown in Fig. 2.

For each background image $x ,$ demographic condition $( r , g )$ , and sampled face image $f ,$ we generate an edited image $y _ { x , f } = \Phi _ { g } ( x , f , p _ { \Phi } )$ , where $p _ { \Phi }$ is a standardized editing prompt specifying the demographic attributes of the inserted person and instructing the model to preserve the original background, lighting, and photographic realism. Because the GPS label is inherited from the source background image, the edited image $y _ { x , f }$ should remain geolocalized at $G _ { x }$ . This design allows us to measure whether changing the apparent demographic attributes of the person changes the predicted location while preserving the same underlying geographic scene as closely as possible. Residual background changes introduced by the editing process are quantified and discussed in Sec. 5.2.

To study whether models rely on culturally or racially associated visual cues beyond facial appearance, we also construct a clothing-conditioned variant. For each race-gender pair $( r , g )$ , we define a set of race-associated clothing examples $C _ { r , g } .$ . We then generate $y _ { x , f , c } = \Phi _ { g } ( x , f , c , p _ { \Phi } )$ $c \sim C _ { r , g } ,$ where c is a sampled clothing description. This intervention enables us to evaluate whether geolocation models shift their predictions when the same geographic background is paired with clothing cues stereotypically associated with a particular demographic group. In the attire-unconditioned setting, no clothing description is provided; however, the image-editing model may still generate race-associated attire through its learned demographic priors.

After each image is generated, we apply the multimodal quality-control model $\Psi _ { g }$ with prompt p<sub>Ψ</sub> to detect severe synthesis failures, including missing body parts, duplicated limbs, multiple faces, poor blending, or cases where the inserted person is absent or too small to support the intended intervention. If the image is flagged as anomalous, the pipeline regenerates the image with the same conditioning variables but a different random seed, thereby sampling a different region of the generative model’s latent space. To ensure termination, we allow up to 100 regeneration attempts.

Finally, we construct a face-obfuscated variant of GEOBIASET by applying Deface [30] to the generated images. This anonymized subset preserves the background scene, body pose, clothing, and overall image composition while blurring detectable faces, allowing us to test whether demographic perturbations persist when explicit facial appearance is removed.

Implementation Details. We instantiate $\Phi _ { g }$ with FLUX.2-klein-9b-kv [25], a conditioned diffusion model that accepts multiple reference images. The background x provides the geographic scene and the reference $f$ controls the inserted subject’s facial appearance. We use a standardized prompt specifying target race, gender, and age, conditioning on the reference face, and instructing the model to preserve background, lighting, pose naturalness, and realism. The prompt is reported in Appendix B.1.

For clothing-conditioned images, we append a sampled attire description from a curated race–gender prompt set; the full attire lists and template are in Appendix B.2. These prompts are controlled visual interventions, not normative claims about demographic groups.

For $\Psi _ { g } ,$ we use Qwen3-VL-8B-Instruct [48] to detect synthesis failures listed above. Flagged images are regenerated with the same conditioning and a different seed. The prompt is in Appendix B.3. This automatic filtering removes severe failures but is not a substitute for a human perceptual study and may miss subtle artifacts or culturally implausible insertions.

Ex Post Quality Assessment. This bring us to conduct an ex post quality assessment to verify that inserting human subjects does not introduce systematic degradation that could confound geolocaliza tion performance. Artifacts such as blur, compression noise, poor blending, or hallucinated structures could alter visual cues independently of the intended demographic intervention.

We first evaluate generated images with CLIPIQA [41] and QualiCLIP [2], two CLIP-based noreference IQA metrics. Comparing edited images with their source background yields comparable quality scores, suggesting that interventions preserve photographic quality and do not introduce degradation that dominates downstream results. However, these no-reference metrics do not measure whether the geographic background itself is preserved.

We therefore additionally perform a full-reference comparison between each edited image and its corresponding source image, excluding the region occupied by the inserted subject. On the remaining background pixels, we obtain PSNR 20.17 ± 3.11 dB, SSIM 0.677 ± 0.126, and LPIPS 0.230 ± 0.055. These results reveal measurable pixel-level changes and show that the editing process does not guarantee strict background identity. Importantly, the controlled masking experiment in Sec. 5.2 separately evaluates whether these residual background changes account for the observed geolocalization degradation.

See Appendix A for full examples from GEOBIASET. To support reproducibility, we have released our code.<sup>2</sup>

## 4 Bias Evaluation Protocol

Our protocol compares predictions on original backgrounds and their edited variants in GEOBIASET. Since each edit inherits the GPS coordinate of its source, any prediction shift induced by the inserted person reflects sensitivity to the race intervention. Let $x _ { i }$ be a background with ground-truth GPS coordinate $G _ { i } = ( \lambda _ { i } , \phi _ { i } )$ . A geolocalization model M predicts ${ \hat { G } } _ { i } = { \mathcal { M } } ( x _ { i } )$ , and we measure geographic error with the Haversine distance $d ( \cdot , \cdot )$

For each $x _ { i } ,$ GEOBIASET provides 28 edited variants $x _ { i } ^ { r } \ = \ \Phi ( x _ { i } , r )$ in which a person from race group r is inserted while GPS label remains fixed and the underlying scene is preserved, though the generative pipeline does not guarantee strict pixel-level background identity. We use the seven FairFace categories: Black, East Asian, Indian, Middle Eastern, Latino, Southeast Asian, White [22]. Race and gender follow FairFace terminology: race denotes a visually perceived attribute for controlled bias auditing, not a biological or self-identified label, while gender is a binary operational annotation for balancing and generation.

To measure whether predictions move in a systematic direction, each race group r is associated with a geographic support region $R _ { r } ,$ defined as the union of countries listed in Appendix C. These regions are evaluation constructs and should not be interpreted as intrinsic mappings between race and geography. For a prediction $\hat { G } , d _ { R } ( \hat { G } , R _ { r } )$ is zero if $\hat { G } \in R _ { r }$ , otherwise the minimum Haversine distance from $\hat { G }$ to $R _ { r }$

Existing metrics. Prior works report geolocalization accuracy as the percentage of predictions within 1, 25, 200, 750, and 2500 km of the ground-truth coordinate, corresponding approximately to street-, city-, region-, country-, and continent-level localization. Higher values indicate better performance. These metrics quantify absolute localization error, but do not capture whether predictions are stable under changes to the inserted person. We therefore complement them with race-based perturbation metrics that measure the magnitude and direction of counterfactual prediction shifts.

## 4.1 Person-Origin Perturbation Indicators

We introduce Person-Origin Perturbation Indicators (POPI), a family of metrics that quantify whether inserting a person from a given race group systematically redirects a model’s prediction toward the geographic region associated with that group. Unlike standard geolocalization accuracy, POPI metrics are not intended to measure whether the model predicts the correct location. Instead, they measure race sensitivity: the degree to which the prediction changes in a direction consistent with the inserted person rather than with the fixed geographic scene. Let $s _ { i }$ denote the index of the ground-truth source region of image $x _ { i }$ , and let $R _ { r }$ denote the geographic support region associated with race group r. For each race-edited variant $\boldsymbol { x } _ { i } ^ { r }$ , we denote the corresponding prediction as $\hat { G } _ { i } ^ { r } = \mathcal { M } ( x _ { i } ^ { r } )$ ) and we define three complementary POPI metrics.

POPI-Attack (POPI-A). POPI-A measures how often inserting a person from race group r redirects the prediction toward the region associated with r, starting from a background image whose original prediction does not already fall in that region. It is computed over samples satisfying ${ \hat { G } } _ { i } \not \in R _ { r }$ , as defined in Eq. (1):

$$
\mathrm { P O P I - A } ( r ) = \frac { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \notin R _ { r } ] \mathbb { 1 } [ \hat { G } _ { i } ^ { r } \in R _ { r } ] } { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \notin R _ { r } ] } .\tag{1}
$$

A high POPI-A value indicates that the inserted person can steer the model toward the race-associated region, suggesting that the model is sensitive to foreground demographic cues that should be irrelevant to the ground-truth coordinate.

POPI-Help (POPI-H). POPI-H measures cases where the insertion of a person moves an initially incorrect prediction into the correct source region. This metric is computed only over background images whose original prediction fails at the source-region level, i.e., $\hat { G } _ { i } \notin R _ { s _ { i } }$ , as defined in Eq. (2):

$$
\mathrm { P O P I - H } ( r ) = \frac { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \notin R _ { s _ { i } } ] \mathbb { 1 } [ \hat { G } _ { i } ^ { r } \in R _ { s _ { i } } ] } { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \notin R _ { s _ { i } } ] } .\tag{2}
$$

This metric captures cases in which the inserted person accidentally improves the prediction. Although such changes may increase apparent source-region accuracy, they still indicate dependence on a foreground intervention that is not part of the original geographic scene.

POPI-Distraction (POPI-D). POPI-D measures the opposite behavior: cases where a model initially predicts the correct source region for the background image, but the inserted person causes the prediction to leave that region. It is computed over samples satisfying $\hat { G } _ { i } \in R _ { s _ { i } }$ , as defined in Eq. (3):

$$
\mathrm { P O P I - D } ( r ) = \frac { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \in R _ { s _ { i } } ] \mathbb { 1 } [ \hat { G } _ { i } ^ { r } \notin R _ { s _ { i } } ] } { \sum _ { i } \mathbb { 1 } [ \hat { G } _ { i } \in R _ { s _ { i } } ] } .\tag{3}
$$

High POPI-D indicates that the race intervention distracts the model from otherwise correct geographic evidence.

Table 1: Overall results on GEOBIASET. We report geolocalization accuracy at standard geodesic distance thresholds, corresponding approximately to street-, city-, region-, country-, and continentlevel localization. The Edited column indicates whether the image is an original background image (✗) or an edited image containing an inserted person (✓). Values in red denote the relative accuracy drop from background to edited images. POPI-A, POPI-H, and POPI-D measure race-induced prediction shifts; lower values indicate lower sensitivity to the inserted person.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Edited</td><td rowspan="2">Street 1km</td><td rowspan="2">City 25km</td><td rowspan="2">Region 200km</td><td rowspan="2">Country 750km</td><td rowspan="2">Continent 2500km</td><td colspan="2">POPI-A (↓) POPI-H (↓)</td><td rowspan="2">POPI-D (↓)</td></tr><tr><td></td></tr><tr><td rowspan="2">GeoCLIP</td><td>X S</td><td>11.1 4.3</td><td>35.6</td><td>51.6</td><td>68.8</td><td>81.9 55.333%</td><td rowspan="2">29.3</td><td rowspan="2">20.9</td><td rowspan="2">30.7</td></tr><tr><td></td><td></td><td>16.3</td><td>25.4</td><td>38.1</td><td>70.3</td></tr><tr><td rowspan="2">Img2Loc</td><td>x √</td><td>8.2</td><td>28.0</td><td>37.0</td><td>53.1</td><td></td><td rowspan="2">34.5</td><td rowspan="2">24.3</td><td rowspan="2">36.2</td></tr><tr><td></td><td>2.2</td><td>7.5</td><td>11.8</td><td>21.8</td><td>37.547%</td></tr><tr><td rowspan="2">G3</td><td>x</td><td>16.6</td><td>45.8</td><td>55.7</td><td>70.3</td><td>81.6</td><td rowspan="2">28.1</td><td rowspan="2">20.0</td><td rowspan="2">29.4</td></tr><tr><td>√</td><td>3.6</td><td>13.3</td><td>19.0</td><td>30.7</td><td> $4 7 . 8 _ { 4 1 \% }$ </td></tr><tr><td rowspan="2">DeepSeek-VL</td><td>x</td><td>0.9</td><td>5.5</td><td>15.5</td><td>41.1</td><td>64.7</td><td rowspan="2">18.2</td><td rowspan="2">19.3</td><td rowspan="2">18.0</td></tr><tr><td>√</td><td>0.7</td><td>5.2</td><td>12.9</td><td>31.3</td><td>51.520%</td></tr><tr><td rowspan="2">InternVL3.5</td><td>x</td><td>0.9</td><td>5.5</td><td>12.8</td><td>34.7</td><td>58.0</td><td rowspan="2">24.1</td><td rowspan="2">24.6</td><td rowspan="2">24.1</td></tr><tr><td>S</td><td>0.8</td><td>4.6</td><td>9.7</td><td>23.4</td><td>43.225%</td></tr><tr><td rowspan="2">Qwen3-VL</td><td>X</td><td>3.2</td><td>24.2</td><td>38.5</td><td>62.7</td><td>79.3</td><td rowspan="2">17.8</td><td rowspan="2">17.6</td><td rowspan="2">17.8</td></tr><tr><td>√</td><td>2.6</td><td>19.2</td><td>30.0</td><td>49.6</td><td>65.318%</td></tr><tr><td rowspan="2">GLM-4.1V</td><td>x</td><td>3.5</td><td>26.5</td><td>47.2</td><td>71.4</td><td>82.8</td><td rowspan="2">16.2</td><td rowspan="2">14.8</td><td rowspan="2">16.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td>S</td><td>2.5</td><td>21.3</td><td>37.4</td><td>56.9</td><td>69.216%</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Overall, high values of POPI-A and POPI-D are undesirable: lower values indicate that a model is less susceptible to person-induced prediction shifts. POPI-H requires more careful interpretation. While a higher POPI-H may correspond to an apparent improvement in source-region accuracy, it still reflects dependence on the inserted person. Therefore, we report POPI-H as a diagnostic measure rather than as a fairness improvement. To ensure that POPI is not confounded by varying region sizes, we also evaluate inverse-area-weighted aggregates; for GeoCLIP, this yields 33.7% POPI-A, 22.4% POPI-H, and 27.8% POPI-D, confirming that demographic sensitivity is not an artifact of large geographic regions.

## 4.2 Evaluated Models

We evaluate seven geolocalization systems spanning two model families: dedicated image geolocal ization models and general-purpose multimodal large language models.

Dedicated geolocalization models. We evaluate three recent image geolocalization systems. Geo-CLIP [39] learns a shared representation between image embeddings and GPS embeddings, enabling coordinate-based retrieval through contrastive alignment. Img2Loc [50] follows a retrieval-augmented generation paradigm: it retrieves visually similar candidates and provides their coordinates to a multimodal language model, which then predicts a final location. G3 [20] similarly adopts a multi-stage geolocalization framework based on retrieval-augmented reasoning. These models represent different design choices in global visual geolocation, including coordinate embedding, retrieval-based candidate construction, and multimodal reasoning over geographic evidence.

Multimodal large language models. We also evaluate four open-weight MLLMs in a zero-shot setting: Qwen3-VL-8B-Instruct [48], DeepSeek-VL-7B [28], InternVL3.5-8B-Instruct [42], and GLM-4.1V-9B [14]. These models are not trained exclusively for image geolocalization, but their visual-language pretraining may allow them to exploit geographic cues such as scripts, road signs, vegetation, architecture, vehicles, terrain, climate, and cultural markers. All MLLMs are evaluated with the same deterministic zero-shot prompt, in which the model is explicitly asked to return a single JSON object containing numeric latitude and longitude values, without explanations or chain-of-thought text. We parse the returned JSON object and discard outputs that cannot be converted into valid coordinates. The exact prompt is reported in Appendix B. We additionally evaluate InternVL3.5-38B [42], Gemma3-27B-IT [36], and Qwen3VL-30B-A3B [48]; their results are reported in Appendix D.4.

## 5 Experimental Results

## 5.1 Overall Geolocalization Performance and Race-Induced Perturbations

Tab. 1 reports results for all 7 models. From these we draw three conclusions.

First, inserting a person consistently degrades accuracy across all models and all geographic scales, most sharply for task-specific systems: GeoCLIP drops from 81.9% to 55.3% at continent level, Img2Loc from 70.3% to 37.5%, and G3 from 81.6% to 47.8%. Even models built for geolocalization are sensitive to foreground human subjects despite the background and GPS coordinate being unchanged.

Second, dedicated models are more strongly affected than general-purpose MLLMs. At continent level, the relative drop is 33–47% for the former and 16–25% for the latter. Higher background-image accuracy does not imply greater robustness to demographic interventions: MLLMs have lower fine-grained accuracy but their predictions are less sharply displaced by the inserted person.

Third, POPI shows the degradation is often a directional shift. POPI-A exceeds 28% for all dedicated models and ranges from 16.2% to 24.1% for MLLMs. Thus, in a substantial fraction of cases, inserting a person from a given race causes the model to predict a coordinate inside the region associated with that race. Overall, these results support the hypothesis that current geolocalization systems can exploit race-associated foreground cues as spurious geographic shortcuts.

## 5.2 Causal Isolation of Foreground Demographic Cues

The comparison between original and edited images may conflate sensitivity to the inserted subject with two additional effects: the subject physically occludes geographic evidence, and the imageediting model may alter portions of the background. We therefore perform a controlled masking experiment using SAM3 [5]. For each edited image, we segment the inserted person and replace the corresponding pixels with three masking conditions: black, uniform gray, and Gaussian blur. Crucially, the same variant-specific mask is also applied to the matched original image, so that the amount, position, and shape of the occlusion are identical in each pair.

Once the subject is removed, edited and original backgrounds achieve very similar geolocalization performance. At the continent level, black masking yields 73.4% accuracy on the masked originals and 72.7% on the corresponding masked edits, while gray masking yields 73.6% and 71.7%, respectively. In contrast, accuracy on the unmasked edited images is only 55.3%, compared with 81.9% on the original backgrounds. Thus, foreground occlusion is a measurable source of degradation, but neither occlusion nor generation-induced background drift explains most of the observed accuracy loss.

Combining this control with the face-obfuscation experiment of Sec. 5.5, the 26.6 point continentlevel drop can be approximately decomposed into 8.5 points due to scene occlusion, 0.7 points due to generation-induced background changes, 9.0 points due to non-facial subject cues, and 8.4 points due to facial appearance. These results support the conclusion that the visible inserted subject, rather than background corruption alone, is the dominant source of the observed degradation. Full results and implementation details are reported in Appendix G.

## 5.3 Landmark Sensitivity and Attribution Analysis

We further analyze whether the presence of a landmark moderates the effect of the inserted person. Landmarks provide highly localized visual evidence and may therefore make geolocalization models less dependent on foreground demographic cues. Conversely, in non-landmark scenes, where the geographic evidence is weaker or more ambiguous, models may rely more heavily on spurious cues introduced by the edited subject. Fig. 3 reports this analysis for GeoCLIP by combining aggregate POPI sensitivity with representative attribution maps. The left panel compares $\Delta _ { \mathrm { A c c } }$ under POPI-H and POPI-D for landmark and non-landmark images. The results indicate that landmarks attenuate, but do not eliminate, race-induced perturbations. For POPI-H, the aggregate accuracy change is substantially larger in non-landmark images than in landmark images, increasing from 9.2% to 27.1%. A similar pattern holds for POPI-D, which increases from 26.0% in landmark scenes to 33.2% in non-landmark scenes. This suggests that when distinctive geographic evidence is weak or absent, the model becomes more susceptible to foreground demographic cues.

![](images/1f38a14987c3264ac9f0335eb9f037baf2f304daa88791695f2db4243318da97.jpg)

![](images/bfdb620058a356f5db8137a5952278a91542f1dd88cf5213c1d247deec5bfabf.jpg)  
(a) Landmark

![](images/46dac09af15c1fc3e4b4cb9d1edcde26cda874da77ec89295fd6787e05ff5ba3.jpg)  
(b) Help

![](images/41f392c7c6bd9d2226d081ac2c69379dc853d96fc5e3aa2c200243bb8d1e0bc9.jpg)

![](images/8395d155b2b75203905fb2d5aa8a66c5d984bc20a231a11927fd6567d5ed88f1.jpg)

![](images/94e01888b044f71db44ba3bc706ce37e3775a7714248063114b51422d3d47c6b.jpg)  
(d) Non-landmark  
(e) Help

(c) Distraction  
![](images/c822713165f1f77e8633ca88da746cf265320b15d8fd07b05a14675c498cdda2.jpg)  
(f) Distraction  
Figure 3: Landmark sensitivity and attribution analysis for GeoCLIP. Left: POPI-H and POPI-D for landmark and non-landmark images. Right: representative gradient-weighted attention rollout maps showing that non-landmark scenes assign more relevance to the inserted person, consistent with larger prediction shifts.

To better understand this behavior, we visualize GeoCLIP evidence using gradient-weighted attention rollout over CLIP vision transformer [1, 6]. The attention maps in Fig. 3 provide qualitative evidence consistent with the aggregate trend. In landmark scenes, the model continues to assign relevance to characteristic location-specific structures, although the inserted person also receives attention. In non-landmark scenes, relevance is more concentrated on the inserted subject, and the prediction is more likely to shift after the demographic intervention. Overall, these results suggest that strong geographic cues can partially stabilize model predictions, but they do not fully prevent race-associated foreground information from influencing geolocalization decisions.

## 5.4 Effect of Race-Associated Attire

We next test whether race-associated clothing cues further affect predictions. The inserted person is generated with clothing sampled from race- and gender-specific lists (Appendix B.2). This intervention tests reliance not only on facial appearance or body-level attributes, but also on culturally associated visual cues spuriously correlated with geography.

Tab. 2 reports the accuracy change for GeoCLIP across distance thresholds. Race-associated at-

Table 2: Effect of race-associated attire on GeoCLIP accuracy. Values report the accuracy change between attire-conditioned and standard edited images. Negative values indicate accuracy drops; darker cells denote larger drops.
<table><tr><td rowspan=1 colspan=1>Race</td><td rowspan=1 colspan=1>1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=2>750km 2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>-2.2</td><td rowspan=1 colspan=1>-4.7</td><td rowspan=1 colspan=1>-9.9</td><td rowspan=1 colspan=1>-12.4</td><td rowspan=1 colspan=1>-15.5</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>-1.2</td><td rowspan=1 colspan=1>-4.1</td><td rowspan=1 colspan=1>-7.0</td><td rowspan=1 colspan=1>-8.7</td><td rowspan=1 colspan=1>-10.6</td></tr><tr><td rowspan=2 colspan=1>IndianLatino Hispanic</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-1.9</td><td rowspan=1 colspan=1>-5.0</td><td rowspan=1 colspan=1>-3.6</td><td rowspan=1 colspan=1>-3.6</td></tr><tr><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>-3.6</td><td rowspan=1 colspan=1>-3.9</td><td rowspan=1 colspan=1>-5.1</td><td rowspan=1 colspan=1>-5.4</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>-1.6</td><td rowspan=1 colspan=1>-2.0</td><td rowspan=1 colspan=1>-3.5</td><td rowspan=1 colspan=1>-1.6</td><td rowspan=1 colspan=1>+0.4</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>-3.1</td><td rowspan=1 colspan=1>-6.6</td><td rowspan=1 colspan=1>-7.3</td><td rowspan=1 colspan=1>-9.0</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>-2.6</td><td rowspan=1 colspan=1>-3.8</td><td rowspan=1 colspan=1>-7.3</td><td rowspan=1 colspan=1>-4.2</td><td rowspan=1 colspan=1>-1.6</td></tr></table>

tire substantially amplifies errors at coarse scales: a Black subject in race-associated attire reduces continent-level accuracy by 15.5%; East Asian and Southeast Asian reduce it by 10.6% and 9.0% respectively. Clothing acts as a strong geographic prior, but the effect is not uniform: some conditions produce smaller drops, and the Middle Eastern condition does not decrease continent-level accuracy in this setting. This heterogeneity suggests attire-based perturbations interact with both the background content and the model’s learned associations between appearance, culture, and geography. Overall, the results suggest that race-associated clothing is therefore an important factor in demographic geolocation bias and should be evaluated separately from facial appearance. Results for the remaining models are in Appendix D.1.

## 5.5 Ablation on Face-Obfuscated Images

We test whether the perturbation is driven by facial appearance by applying face obfuscation to GEOBIASET, which preserves the rest of the image while blurring the inserted person’s face. If the drop were mainly caused by facial identity, obfuscation should largely remove it.

Tab. 3 reports the change in GeoCLIP accuracy with face obfuscation. Obfuscation slightly weakens the perturbation in several cases, but the overall drop remains substantial for most groups.

Face obfuscation increases the localization accuracy of edited images especially for Latino Hispanic, Middle Eastern, East Asian, Indian, and Black, ranging between 8.8–11.7%. The White group shows a smaller difference, so the effect is not uniform.

Table 3: Effect of face obfuscation on GeoCLIP accuracy. Values report the accuracy gain after applying face blurring to the edited images, relative to the corresponding non-obfuscated edited images. Positive values indicate that obfuscation improves accuracy, suggesting reduced sensitivity to explicit facial cues. Darker cells denote larger gains.

Results indicate that facial appearance alone does not explain the model’s sensitivity. Even with the face blurred, the model remains affected by non-facial cues such as clothing, skin tone outside the face region, body appearance, pose, and contextual interactions between the inserted person and the background. Demographic bias in geolocalization can persist under face anonymization, which matters for both privacy-

<table><tr><td rowspan=1 colspan=2>Race          1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=2>750km 2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>+1.3</td><td rowspan=1 colspan=1>+4.9</td><td rowspan=1 colspan=1>+5.7</td><td rowspan=1 colspan=2>+8.3    +8.8</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>+1.4</td><td rowspan=1 colspan=1>+4.0</td><td rowspan=1 colspan=1>+6.1</td><td rowspan=1 colspan=2>+8.6    +9.2</td></tr><tr><td rowspan=1 colspan=1>Indian</td><td rowspan=1 colspan=1>+2.7</td><td rowspan=1 colspan=1>+5.9</td><td rowspan=1 colspan=1>+5.4</td><td rowspan=1 colspan=2>+7.5    +8.8</td></tr><tr><td rowspan=1 colspan=1>Latino Hispanic</td><td rowspan=1 colspan=1>+1.0</td><td rowspan=1 colspan=1>+4.3</td><td rowspan=1 colspan=1>+6.2</td><td rowspan=2 colspan=2>+9.4   +11.7+11.0   +10.7</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>+0.3</td><td rowspan=1 colspan=1>+5.3</td><td rowspan=1 colspan=1>+6.5</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>+1.4</td><td rowspan=1 colspan=1>+4.1</td><td rowspan=1 colspan=1>+7.2</td><td rowspan=1 colspan=1>+9.0</td><td rowspan=1 colspan=1>+7.9</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>+1.3</td><td rowspan=1 colspan=1>+2.2</td><td rowspan=1 colspan=1>+3.2</td><td rowspan=1 colspan=1>+1.7</td></tr></table>

preserving evaluation and responsible deployment. Face obfuscation is at best a first-aid mitigation against severe degradation in biased models.

## 6 Ethical Considerations and Limitations

This work studies demographic bias in a task with clear dual-use implications. Geolocalization can support journalism, historical analysis, environmental monitoring, geography education, and opensource verification, but it can also enable surveillance, stalking, military targeting, or identification of sensitive locations. Bias in such systems can amplify harm if errors or shifts are unevenly distributed across demographic groups or regions. GEOBIASET is intended as an auditing benchmark for measuring and mitigating these risks, not as a tool for improving harmful uses of geolocation technology.

Our design also simplifies human identity. We follow the FairFace race taxonomy and binary gender annotations to obtain a controlled, reproducible benchmark, but these categories do not capture the full complexity of race, ethnicity, gender identity, migration, diaspora, or cultural belonging. Race-associated regions and clothing prompts are operational constructs for measurement and should not be read as intrinsic, exhaustive, or normative mappings. A high POPI score indicates model sensitivity to race-associated geographic shortcuts, not demographic truth.

Finally, GEOBIASET relies on synthetic image editing rather than masked inpainting. While the source GPS coordinate is fixed and the geographic scene is largely preserved, the editing process does not guarantee pixel-level background identity. Full-reference measurements on the unmasked background reveal measurable generation-induced changes, including occasional lighting shifts and localized artifacts. The pipeline also does not explicitly constrain subject scale or placement, and our automated quality control cannot guarantee geometric consistency or replace formal human evaluation. However, the controlled masking experiment in Sec. 5.2 shows that these effects do not primarily explain the observed geolocalization degradation. The benchmark is additionally limited to 329 source backgrounds and focuses on foreground people; future work should broaden geographic coverage, strengthen intervention control, and study other sources of geolocalization bias.

## 7 Conclusion

We introduced GEOBIASET, a counterfactual benchmark for auditing demographic bias in worldwide image geolocalization, together with POPI metrics for measuring directional prediction shifts under demographic interventions. By decoupling foreground human attributes from the geographic scene, the benchmark exposes a failure mode that standard geodesic accuracy cannot capture.

Across dedicated geolocalization models and zero-shot multimodal large language models, we find that predictions are often unstable under changes to the inserted person and can shift toward raceassociated geographic regions. These results show that high benchmark accuracy does not necessarily imply robust geographic reasoning. We hope GEOBIASET encourages future geolocalization systems to report counterfactual robustness alongside accuracy and to reduce reliance on demographic shortcuts.

## References

[1] Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th annual meeting ofthe associationfor computational linguistics, pages 4190–4197, 2020.

[2] Lorenzo Agnolucci, Leonardo Galteri, and Marco Bertini. Quality-aware image-text alignment for opinionunaware image quality assessment. arXiv preprint arXiv:2403.11176, 2024.

[3] Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude, Elliot Vincent, et al. Openstreetview-5m: The many roads to global visual geolocation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21967–21977, 2024.

[4] Gabriele Berton, Riccardo Mereu, Gabriele Trivigno, Carlo Masone, Gabriela Csurka, Torsten Sattler, and Barbara Caputo. Deep visual geo-localization benchmark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5396–5407, 2022.

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025.

[6] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 782–791, 2021.

[7] Haodong Chen, Qiang Huang, Jiaqi Zhao, Qiuping Jiang, Xiaojun Chang, and Jun Yu. Measuring social bias in vision-language models with face-only counterfactuals from real photos. arXiv preprint arXiv:2601.06931, 2026.

[8] Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda, and Mubarak Shah. Where we are and what we’re looking at: Query based worldwide image geo-localization using hierarchies and scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23182–23190, 2023.

[9] Sepehr Dehdashtian, Lan Wang, and Vishnu Naresh Boddeti. Fairerclip: Debiasing clip’s zero-shot predictions using functions in rkhss. arXiv preprint arXiv:2403.15593, 2024.

[10] Noa Garcia, Yusuke Hirota, Yankun Wu, and Yuta Nakashima. Uncurated image-text datasets: Shedding light on demographic bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6957–6966, 2023.

[11] Walter Gerych, Haoran Zhang, Kimia Hamidieh, Eileen Pan, Maanas Sharma, Thomas Hartvigsen, and Marzyeh Ghassemi. Bendvlm: Test-time debiasing of vision-language embeddings. Advances in Neural Information Processing Systems, 37:62480–62502, 2024.

[12] Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. Pigeon: Predicting image geolocations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12893– 12902, 2024.

[13] James Hays and Alexei A Efros. Im2gps: estimating geographic information from a single image. In 2008 ieee conference on computer vision and pattern recognition, pages 1–8. IEEE, 2008.

[14] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

[15] Phillip Howard, Anahita Bhiwandiwalla, Kathleen C Fraser, and Svetlana Kiritchenko. Uncovering bias in large vision-language models with counterfactuals. arXiv preprint arXiv:2404.00166, 2024.

[16] Phillip Howard, Xin Su, and Kathleen C Fraser. Cultural counterfactuals: Evaluating cultural biases in large vision-language models with counterfactual examples. arXiv preprint arXiv:2603.02370, 2026.

[17] Jen-tse Huang, Jiantong Qin, Jianping Zhang, Youliang Yuan, Wenxuan Wang, and Jieyu Zhao. Visbias: Measuring explicit and implicit social biases in vision language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17981–18004, 2025.

[18] Jingyuan Huang, Jen-tse Huang, Ziyi Liu, Xiaoyuan Liu, Wenxuan Wang, and Jieyu Zhao. Ai sees your location—but with a bias toward the wealthy world. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 18030–18050, 2025.

[19] Neel Jay, Hieu Minh Nguyen, Trung Dung Hoang, and Jacob Haimes. Evaluating precise geolocation inference capabilities of vision language models. arXiv preprint arXiv:2502.14412, 2025.

[20] Pengyue Jia, Yiding Liu, Xiaopeng Li, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, Dawei Yin, and Xiangyu Zhao. G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models. Advances in neural information processing systems, 37:53198–53221, 2024.

[21] Pengyue Jia, Seongheon Park, Song Gao, Xiangyu Zhao, and Sharon Li. Georanker: Distance-aware ranking for worldwide image geolocalization. arXiv preprint arXiv:2505.13731, 2025.

[22] Kimmo Kärkkäinen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age. arXiv preprint arXiv:1908.04913, 2019.

[23] Hyemi Kim, Seungjae Shin, JoonHo Jang, Kyungwoo Song, Weonyoung Joo, Wanmo Kang, and Il-Chul Moon. Counterfactual fairness with disentangled causal effect variational autoencoder. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 8128–8136, 2021.

[24] Fanjie Kong, Shuai Yuan, Weituo Hao, and Ricardo Henao. Mitigating test-time bias for fair image retrieval. Advances in neural information processing systems, 36:71520–71539, 2023.

[25] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[26] Lingyao Li, Runlong Yu, Qikai Hu, Bowei Li, Min Deng, Yang Zhou, and Xiaowei Jia. From pixels to places: A systematic benchmark for evaluating image geolocalization ability in large language models. arXiv preprint arXiv:2508.01608, 2025.

[27] Xinwei Liu, Xiaojun Jia, Yuan Xun, Simeng Qin, and Xiaochun Cao. Geoshield: Safeguarding geolocation privacy from vision-language models via adversarial perturbations. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 35653–35661, 2026.

[28] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

[29] Niccolò Niccoli, Federico Becattini, and Lorenzo Seidenari. Defending from geolocalization through adversarial road trips. In European Conference on Computer Vision, pages 611–629. Springer, 2026.

[30] ORB-HD. deface. https://github.com/ORB-HD/deface. Accessed: 2026-05-02.

[31] Zhaofang Qian, Hardy Chen, Zeyu Wang, Li Zhang, Zijun Wang, Xiaoke Huang, Hui Liu, Xianfeng Tang, Zeyu Zheng, Haoqin Tu, et al. Where on earth? a vision-language benchmark for probing model geolocation skills across scales. arXiv preprint arXiv:2510.10880, 2025.

[32] Guruprasad V Ramesh, Harrison Rosenberg, Ashish Hooda, and Shimaa Ahmed Kassem Fawaz. Synthetic counterfactual faces. arXiv preprint arXiv:2407.13922, 2024.

[33] Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han. Cplanet: Enhancing image geolocalization by combinatorial partitioning of maps. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 536–551, 2018.

[34] Mohit Talreja, Joshua Diao, Jim Thannikary James, Radu Casapu, Tejas Santanam, Ethan Mendes, Alan Ritter, Wei Xu, and James Hays. Georc: A benchmark for geolocation reasoning chains. arXiv preprint arXiv:2601.21278, 2026.

[35] Md Mehrab Tanjim, Krishna Kumar Singh, Kushal Kafle, Ritwik Sinha, and Garrison W Cottrell. Discovering and mitigating biases in clip-based image editing. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2984–2993, 2024.

[36] Gemma Team. Gemma 3. 2025.

[37] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.

[38] Ximena Salgado Uribe, Martí Bosch, and Jérôme Chenal. Regional biases in image geolocation estimation: a case study with the sensecity africa dataset. arXiv preprint arXiv:2404.02558, 2024.

[39] Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization. Advances in Neural Information Processing Systems, 36:8690–8701, 2023.

[40] Nam Vo, Nathan Jacobs, and James Hays. Revisiting im2gps in the deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 2621–2630, 2017.

[41] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 2555–2563, 2023.

[42] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[43] Xiaonan Wang, Bo Shao, and Hansaem Kim. Ai knows where you are: Exposure, bias, and inference in multimodal geolocation with koreageo. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 9897–9914, 2025.

[44] Yining Wang, Xi Li, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Zhenxing Qian, and Mi Wen. Hiding in plain sight: A diffusion-based mitigation of geolocation privacy leakage in vision-language models, 2026.

[45] Tobias Weyand, Ilya Kostrikov, and James Philbin. Planet-photo geolocation with convolutional neural networks. In European conference on computer vision, pages 37–55. Springer, 2016.

[46] Yixin Wu, Mingjie Li, Yun Shen, Michael Backes, and Yang Zhang. Geocloak: An agentic framework for semantics-preserving geolocation privacy via peripheral clue manipulation. arXiv preprint arXiv:2610.20055, 2026.

[47] Yifan Xu, Pourya Shamsolmoali, Eric Granger, Claire Nicodeme, Laurent Gardes, and Jie Yang. Transvlad: Multi-scale attention-based global descriptors for visual geo-localization. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision, pages 2840–2849, 2023.

[48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[49] Haoyu Zhang, Yangyang Guo, and Mohan Kankanhalli. Joint vision-language social bias removal for clip. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 4246–4255, 2025.

[50] Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu, Sheng Li, and Gengchen Mai. Img2loc: Revisiting image geolocalization using multi-modality foundation models and image-based retrieval-augmented generation. In Proceedings ofthe 47th international acm sigir conference on research and development in information retrieval, pages 2749–2754, 2024.

## Appendix A Full Counterfactual GEOBIASET example

To complement the quantitative analysis in Sec. 5, we visualize the complete set of GEOBIASET counterfactual variants generated from a single background image. Fig. 4 shows a landmark scene and Fig. 5 a non-landmark scene.

![](images/3f6a3b4fdf5233c698d97445766194930afdff0fb3ed0aa2272f1538b288b830.jpg)  
Figure 4: Full GEOBIASET counterfactual grid for a landmark scene with all 28 edited variants across seven races (rows) and four gender-attire conditions (columns), sharing the same ground-truth GPS coordinate; subject composition and background appearance may vary across edits.

![](images/8d0cb57f390a2a35081cd8dc5d472768c83dcf1db170f169980e13a02d260ed2.jpg)  
Figure 5: Full GEOBIASET counterfactual grid for a non-landmark scene with all 28 edited variants across seven races (rows) and four gender-attire conditions (columns), sharing the same ground-truth GPS coordinate; subject composition and background appearance may vary across edits.

## Appendix B Prompts

## B.1 Image editing prompt

To generate the dataset images we use the following prompt:

## Image Editing Prompt

Generate a [RACE] [GENDER] person ofage [AGE] years in image 1 looking like the personfrom image 2. Make the generated person look as similar as possible to the person in image 2. Make the person pretty. Keep the lighting and the background as in image 1. (Make the generated personface the camera: 0.5). Give the picture a naturalfeeling, as ifit was taken by a camera. (Make the generated person look like they belong in the scene: 1.4). Keep a natural pose. (Do not make more than two arms, legs orfaces).

## B.2 Clothing prompt

Clothing Prompt

The person is wearing typical [GENDER ADJ] [CLOTHING SAMPLE]: 1.4.

Here, GENDER ADJ is set to “masculine” for male subjects and “feminine” for female subjects to reduce ambiguity for garments that may be worn across genders. CLOTHING SAMPLE is drawn from a curated list of typical garments generated with Claude Sonnet 4.6, reported in Tab. 4

Table 4: Curated garment list per race, split by gender adjective.
<table><tr><td>Race</td><td>Masculine garments</td><td>Feminine garments</td></tr><tr><td>White</td><td>suit and tie, jeans and t-shirt, chinos with polo shirt, bomber jacket, cable-knit sweater, trench coat, flannel shirt with jeans, tuxedo</td><td>sundress, jeans and blouse, pencil skirt with top, trench coat, wrap dress, cardi- gan with slacks, maxi skirt with blouse, cocktail dress</td></tr><tr><td>Black</td><td>dashiki, suit and tie, streetwear hoodie and joggers, agbada, senator suit, boubou, kente cloth shirt, kaftan</td><td>ankara wrap dress, dashiki blouse with trousers, kente cloth dress, gele headwrap with boubou, iro and buba, lappa skirt with top, kitenge dress, adire wrap dress</td></tr><tr><td>Southeast Asian</td><td>batik shirt, sarong with shirt, barong taga- log, longyi, baju melayu, pha nung, puso ng saging, teluk belanga</td><td>kebaya, ao dai, sarong with blouse, baro&#x27;t saya, baju kurung, pha sin wrap skirt, in- abel dress, nyonya kebaya</td></tr><tr><td>East Asian</td><td>changshan, hanbok (jeogori and baji), hakama with kimono, modern suit, man- darin collar jacket, haori, magua jacket, durumagi overcoat</td><td>qipao, hanbok (jeogori and chima), ki- mono, hanfu, furisode, jeogori with maxi chima, ruqun, uchikake</td></tr><tr><td></td><td>Latino/Hispanic guayabera shirt, charro suit, poncho with trousers, linen shirt and slacks, ruana cloak, jorongo poncho, montuno shirt, bombacha trousers</td><td>huipil, pollera, china poblana, floral wrap skirt with blouse, mestiza dress, cholita layered skirt, jorongo shawl with dress, re- bozo wrap with blouse</td></tr><tr><td>Middle East- ern</td><td>thobe (dishdasha), keffiyeh with agal, shal- war kameez, jalabiya, bisht cloak, sirwal trousers with tunic, jubba, egal with ghutra</td><td>abaya, kaftan, jilbab with hijab, thobe al- nashal, farasha dress, jalabiya with tarha bisht evening cloak, shayla with maxi dress</td></tr><tr><td>Indian</td><td>kurta with churidar, sherwani, dhoti with angavastram, nehru jacket, achkan coat, bandhgala suit, lungi with shirt, pathani suit</td><td>saree, salwar kameez, lehenga choli, anarkali suit, churidar with kurti, gharara, sharara with dupatta, ikat silk dress</td></tr></table>

## B.3 Quality control prompt

For the quality-control model $\Psi _ { g } ,$ we use Qwen3-VL-8B-Instruct [48]. The model is prompted with:

## Quality Control Prompt

Is this picture strange? Answer with “Yes” ifthere is anything strange or abnormal in the image, such as more than 2 arms, more than 2 legs, more than 2faces, missing body parts, or any glaring issue with the generated person. Ifyou do not see any person in the image, or ifthe person is very small and hard to see, consider it an abnormality as well. Answer with “No” ifthere is nothing strange or abnormal in the image.

## B.4 MLLM geolocalization prompt

For all MLLMs, we use the same deterministic prompt requiring a single latitude–longitude prediction:

## MLLM Geolocalization Prompt

Estimate the geographic location ofthis image. Use visual clues such as architecture, vegetation, terrain, road markings, signs, weather, vehicles, shadows, coastlines, mountains, water, and text. Return exactly one JSON object with numeric “lat” and “lon” values and nothing else. Use signed decimal degrees: south and west are negative, north and east are positive. Do not copy coordinates from examples or instructions; there are no reference coordinates in this prompt. Rules: no explanation, no markdown, no prose, no reasoning, no <think> section. Ifuncertain, still give your best guess as latitude and longitude. Thefirst character ofyour answer must be “{” and the last character must be “}”.

We parse the returned JSON object and discard outputs that cannot be converted into valid numeric latitude and longitude values. This fixed-prompt protocol ensures that differences across MLLMs are attributable to model behavior rather than to prompt variation.

## Appendix C Taxonomy

We define the demographic taxonomy used in our prompt construction by mapping each broad group label to a set of associated countries. This taxonomy is used only as an operational grouping for experimental prompt generation and should not be interpreted as an exhaustive, fixed, or essentialized definition of cultural, ethnic, or national identity.

• Indian. India, Pakistan, Bangladesh, Nepal, and Sri Lanka.

• East Asian. China, Japan, South Korea, North Korea, and Taiwan.

• Southeast Asian. Thailand, Vietnam, Indonesia, Malaysia, Philippines, Cambodia, Laos, and Myanmar.

• Middle Eastern. Saudi Arabia, Iran, Iraq, Israel, Jordan, United Arab Emirates, Oman, Yemen, Qatar, Kuwait, Syria, and Lebanon.

• Black. Algeria, Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Cabo Verde, Central African Republic, Chad, Comoros, Democratic Republic of the Congo, Congo, Côte d’Ivoire, Djibouti, Egypt, Equatorial Guinea, Eritrea, Eswatini, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Libya, Madagascar, Malawi, Mali, Mauritania, Mauritius, Morocco, Mozambique, Namibia, Niger, Nigeria, Rwanda, São Tomé and Príncipe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, South Sudan, Sudan, Tanzania, Togo, Tunisia, Uganda, Zambia, and Zimbabwe.

• Latino/Hispanic. Mexico, Guatemala, Belize, Honduras, El Salvador, Nicaragua, Costa Rica, Panama, Cuba, Dominican Republic, Haiti, Jamaica, Trinidad and Tobago, Colombia, Venezuela, Guyana, Suriname, Ecuador, Peru, Brazil, Bolivia, Paraguay, Uruguay, Argentina, and Chile.

• White. United States, Canada, United Kingdom, Ireland, France, Germany, Italy, Spain, Portugal, Netherlands, Belgium, Switzerland, Austria, Greece, Poland, Sweden, Norway, Finland, Denmark, Australia, New Zealand, Russia, and Ukraine.

Table 5: Effect of explicit race-associated attire conditioning across distance thresholds. Each subtable reports accuracy in the race-associated condition minus accuracy in the attire-unconditioned condition, in percentage points. Negative values indicate lower accuracy under race-associated explicit attire conditioning. The attire-unconditioned setting omits clothing descriptions but can retain the generator’s demographic clothing priors.  
(a) Img2Loc
<table><tr><td rowspan=1 colspan=2>Race          1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=1>750km</td><td rowspan=1 colspan=1>2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>-1.6</td><td rowspan=1 colspan=1>-4.8</td><td rowspan=1 colspan=1>-9.9</td><td rowspan=1 colspan=1>-11.7</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>-0.2</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-2.5</td><td rowspan=1 colspan=1>-6.1</td><td rowspan=1 colspan=1>-8.6</td></tr><tr><td rowspan=1 colspan=1>Indian</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>-3.2</td></tr><tr><td rowspan=1 colspan=1>Latino</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-2.0</td><td rowspan=1 colspan=1>-2.5</td><td rowspan=1 colspan=1>-4.7</td><td rowspan=1 colspan=1>-4.7</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>-0.3</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.7</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>-1.5</td><td rowspan=1 colspan=1>-2.3</td><td rowspan=1 colspan=1>-2.3</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>-0.3</td><td rowspan=1 colspan=1>-0.2</td><td rowspan=1 colspan=1>-0.7</td><td rowspan=1 colspan=1>-0.2</td><td rowspan=1 colspan=1>1.3</td></tr></table>

(c) DeepSeek-VL

(b) G3
<table><tr><td rowspan=1 colspan=1>Race</td><td rowspan=1 colspan=1>1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=1>750km</td><td rowspan=1 colspan=1>2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>-1.7</td><td rowspan=1 colspan=1>-6.1</td><td rowspan=1 colspan=1>-8.8</td><td rowspan=1 colspan=1>-13.0</td><td rowspan=1 colspan=1>-17.5</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>-2.8</td><td rowspan=1 colspan=1>-3.5</td><td rowspan=1 colspan=1>-8.2</td><td rowspan=1 colspan=1>-8.5</td></tr><tr><td rowspan=1 colspan=1>Indian</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-2.0</td><td rowspan=1 colspan=1>-4.5</td><td rowspan=1 colspan=1>-5.8</td></tr><tr><td rowspan=1 colspan=1>Latino Hispanic</td><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1>-3.4</td><td rowspan=1 colspan=1>-4.5</td><td rowspan=1 colspan=1>-6.3</td><td rowspan=1 colspan=1>-4.8</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>-0.7</td><td rowspan=1 colspan=1>-2.5</td><td rowspan=1 colspan=1>-1.8</td><td rowspan=1 colspan=1>-2.0</td><td rowspan=1 colspan=1>-1.0</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1>-3.9</td><td rowspan=1 colspan=1>-3.6</td><td rowspan=1 colspan=1>-6.4</td><td rowspan=1 colspan=1>-8.3</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>-1.5</td><td rowspan=1 colspan=1>-3.5</td><td rowspan=1 colspan=1>-3.5</td><td rowspan=1 colspan=1>-0.4</td><td rowspan=1 colspan=1>-2.6</td></tr></table>

(d) InternVL3.5

<table><tr><td rowspan=1 colspan=1>Race</td><td rowspan=1 colspan=1>1km 25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=1>750km</td><td rowspan=1 colspan=1>2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>0.0  -1.3</td><td rowspan=1 colspan=1>-3.2</td><td rowspan=1 colspan=1>-5.1</td><td rowspan=1 colspan=1>-8.0</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>-0.1  -1.6</td><td rowspan=1 colspan=1>-2.8</td><td rowspan=1 colspan=1>-3.8</td><td rowspan=1 colspan=1>-5.0</td></tr><tr><td rowspan=1 colspan=1>Indian</td><td rowspan=1 colspan=1>0.0  -0.9</td><td rowspan=1 colspan=1>-2.5</td><td rowspan=1 colspan=1>-3.1</td><td rowspan=1 colspan=1>-4.4</td></tr><tr><td rowspan=1 colspan=1>Latino Hispanic</td><td rowspan=1 colspan=1>-0.1  -1.3</td><td rowspan=1 colspan=1>-3.2</td><td rowspan=1 colspan=1>-3.4</td><td rowspan=1 colspan=1>-6.3</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>0.0  -0.3</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-1.8</td><td rowspan=1 colspan=1>-3.2</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>0.0  -0.9</td><td rowspan=1 colspan=1>-3.2</td><td rowspan=1 colspan=1>-3.8</td><td rowspan=1 colspan=1>-2.8</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>0.1   0.0</td><td rowspan=1 colspan=1>-0.9</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>-1.5</td></tr></table>

(e) Qwen3-VL

<table><tr><td>Race</td><td>1km</td><td>25km</td><td>200km</td><td>750km</td><td>2500km</td></tr><tr><td>White</td><td>0.0</td><td>-0.6</td><td>0.7</td><td>-3.1</td><td>-3.1</td></tr><tr><td>Black</td><td>-0.2</td><td>-0.6</td><td>-2.6</td><td>-5.1</td><td>-8.0</td></tr><tr><td>East Asian</td><td>0.1</td><td>-1.3</td><td>-1.5</td><td>-5.2</td><td>-7.4</td></tr><tr><td>Indian</td><td>-0.3</td><td>0.0</td><td>-0.7</td><td>-2.6</td><td>-1.3</td></tr><tr><td>Latino</td><td>-0.2</td><td>-0.9</td><td>-1.5</td><td>-2.0</td><td>-1.9</td></tr><tr><td>Middle Eastern</td><td>-0.3</td><td>-0.7</td><td>-1.2</td><td>-3.5</td><td>-2.2</td></tr><tr><td>Southeast Asian</td><td>-0.3</td><td>-0.3</td><td>-1.2</td><td>-2.5</td><td>0.3</td></tr></table>

(f) GLM-4.1V

<table><tr><td rowspan=1 colspan=1>Race</td><td rowspan=1 colspan=1>1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=1>200km</td><td rowspan=1 colspan=1>750km</td><td rowspan=1 colspan=1>2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-1.8</td><td rowspan=1 colspan=1>-4.5</td><td rowspan=1 colspan=1>-6.1</td><td rowspan=1 colspan=1>-8.6</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>-1.3</td><td rowspan=1 colspan=1>-4.5</td><td rowspan=1 colspan=1>-6.6</td><td rowspan=1 colspan=1>-7.3</td></tr><tr><td rowspan=1 colspan=1>Indian</td><td rowspan=1 colspan=1>-0.4</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-1.8</td><td rowspan=1 colspan=1>-5.7</td><td rowspan=1 colspan=1>-4.5</td></tr><tr><td rowspan=1 colspan=1>Latino Hispanic</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-2.2</td><td rowspan=1 colspan=1>-1.5</td><td rowspan=1 colspan=1>-6.4</td><td rowspan=1 colspan=1>-5.7</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>-1.6</td><td rowspan=1 colspan=1>-2.6</td><td rowspan=1 colspan=1>-3.8</td><td rowspan=1 colspan=1>-0.4</td></tr><tr><td rowspan=1 colspan=1>Southeast Asian</td><td rowspan=1 colspan=1>-0.3</td><td rowspan=1 colspan=1>-3.4</td><td rowspan=1 colspan=1>-5.7</td><td rowspan=1 colspan=1>-11.5</td><td rowspan=1 colspan=1>-13.7</td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>-0.9</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>-0.3</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Race</td><td rowspan=1 colspan=1>1km</td><td rowspan=1 colspan=1>25km</td><td rowspan=1 colspan=2>200km 750km</td><td rowspan=1 colspan=1>2500km</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-2.9</td><td rowspan=1 colspan=1>-7.6</td><td rowspan=1 colspan=1>-7.1</td><td rowspan=1 colspan=1>-8.5</td></tr><tr><td rowspan=1 colspan=1>East Asian</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>-4.4</td><td rowspan=1 colspan=1>-7.9</td><td rowspan=1 colspan=1>-10.4</td><td rowspan=1 colspan=1>-8.3</td></tr><tr><td rowspan=2 colspan=1>IndianLatino Hispanic</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>-0.7</td><td rowspan=1 colspan=1>-2.6</td><td rowspan=1 colspan=1>-5.1</td><td rowspan=1 colspan=1>-3.7</td></tr><tr><td rowspan=1 colspan=1>-0.4</td><td rowspan=1 colspan=1>-3.2</td><td rowspan=1 colspan=1>-4.2</td><td rowspan=1 colspan=1>-6.7</td><td rowspan=1 colspan=1>-7.2</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1>-0.7</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-2.2</td><td rowspan=1 colspan=1>0.2</td></tr><tr><td rowspan=2 colspan=1>Southeast AsianWhite</td><td rowspan=1 colspan=1>-0.2</td><td rowspan=1 colspan=1>-3.1</td><td rowspan=1 colspan=1>-7.3</td><td rowspan=1 colspan=1>-8.9</td><td rowspan=1 colspan=1>-9.8</td></tr><tr><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>-0.6</td><td rowspan=1 colspan=1>-2.2</td><td rowspan=1 colspan=1>-1.0</td></tr></table>

## Appendix D Additional Results

## D.1 Additional Results on the Effect of Race-Associated Attire

We report the effect of race-associated attire on the other models in Tabs. 5a, 5b, 5c, 5d, 5e, and 5f.

## D.2 Additional Results on Landmark Sensitivity

In Figs. 6a, 6b, 6c, 6d, 6e, and 6f we report the POPI-H and POPI-D values for landmark and non-landmark images for all the other models.

## D.3 Additional Results on POPI

To complement the aggregate results, we evaluate POPI metrics separately across each inserted demographic group (Tab. 6). We observe substantial variation across categories: for GeoCLIP, Black subjects yield the highest directional steering (POPI-A = 37.1%) and distraction (POPI-D = 38.4%), while Middle Eastern subjects produce the highest source-region correction rate (POPI-H = 38.3%). Conversely, White subjects exhibit the lowest values across all three indicators (≤ 17.6%).

These group-level differences capture model reliance on learned demographic shortcuts under our operational support regions rather than intrinsic properties of the depicted groups. Because transition rates depend on eligible baseline predictions, visual subject cues, and region boundaries, variations across groups reflect model-specific geographic associations. Finally, we reiterate that POPI-H quantifies incidental source-region transitions caused by an irrelevant foreground intervention and does not represent an improvement in model fairness.

(c) DeepSeek-VL  
![](images/09d4e33ec8fca0d53e2f4b306246521d73fe8cb2aea6bd2e52ae5fb40d87460f.jpg)

![](images/1402b4363c1cf394dee8a8b0297b37fb470d8ef936c7c0930c0514511598f3bb.jpg)

![](images/666b88193f4abff35dc18bc17b72b82b188f5727387ea689025fab401604cc26.jpg)

![](images/b7e68dcba61cd76c2495c8bb89fb02d8a4ff0b3561ef664cb67351d4833c3e6c.jpg)

![](images/eab0e0108b7c726652be174885bc0f84c50511f6f39e13c34356b23fc553c4c0.jpg)  
(e) Qwen3-VL

![](images/66c747658ec254f9f50760cdaee221a1361c5b49bc016732a9c09133343cced9.jpg)  
(f) GLM-4.1V  
Figure 6: Landmark sensitivity analysis across six models. POPI-H and POPI-D for landmark and non-landmark images. Non-landmark images exhibit larger prediction shifts across all models, suggesting that the inserted person has a stronger effect when distinctive geographic landmarks are absent.

Table 6: GeoCLIP POPI metrics by inserted demographic group. Scores are computed over the corresponding eligible baseline populations defined in Sec. 4.1. Values are percentages.
<table><tr><td>Inserted Group</td><td>POPI-A</td><td>POPI-H</td><td>POPI-D</td></tr><tr><td>Black</td><td>37.1</td><td>29.3</td><td>38.4</td></tr><tr><td>East Asian</td><td>28.7</td><td>17.0</td><td>30.7</td></tr><tr><td>Indian</td><td>30.0</td><td>15.4</td><td>32.5</td></tr><tr><td>Latino Hispanic</td><td>30.2</td><td>16.0</td><td>32.6</td></tr><tr><td>Middle Eastern</td><td>29.5</td><td>38.3</td><td>28.0</td></tr><tr><td>Southeast Asian</td><td>33.0</td><td>19.7</td><td>35.2</td></tr><tr><td>White</td><td>16.5</td><td>10.1</td><td>17.6</td></tr></table>

Table 7: Geolocalization accuracy and POPI metrics for larger MLLMs. Accuracy at standard geodesic distance thresholds and POPI transition rates are reported as percentages. Original and Edited denote unmodified source backgrounds and their counterfactual edited variants, respectively.
<table><tr><td>Methods</td><td>Edited</td><td>Street 1km</td><td>City 25km</td><td>Region 200km</td><td>Country 750km</td><td>Continent 2500km</td><td>POPI-A (↓)</td><td>POPI-H (↓)</td><td>POPI-D (↓)</td></tr><tr><td rowspan="2">InternVL3.5-38B</td><td>x √</td><td>1.2 1.0</td><td>15.2 11.0</td><td>28.6 19.5</td><td>48.9 36.5</td><td>71.1 55.422%</td><td>23.2</td><td>21.4</td><td>23.4</td></tr><tr><td>X</td><td>5.5</td><td>30.7</td><td>48.0</td><td>68.7</td><td>80.6</td><td>15.7</td><td>19.2</td><td>15.1</td></tr><tr><td>Gemma3-27B-IT Qwen3VL-30B-A3B</td><td>√ x √</td><td>3.8 5.2 4.4</td><td>24.6 33.7 29.3</td><td>37.8 50.2 41.8</td><td>54.4 74.5 60.4</td><td>67.816% 86.3 72.916%</td><td>14.3</td><td>14.4</td><td>14.3</td></tr></table>

## D.4 Additional Results on Larger MLLMs

To assess whether demographic sensitivity persists in higher-capacity architectures, we extend our evaluation to InternVL3.5-38B[42], Gemma3-27B-IT[36], and Qwen3VL-30B-A3B[48]. As reported in Tab. 7, all three models exhibit systematic accuracy drops following foreground person insertion across all evaluated geodesic thresholds. At the 2500 km continent level, accuracy decreases by 15.7 percentage points for InternVL3.5-38B (from 71.1% to 55.4%), 12.8 points for Gemma3-27B-IT (from 80.6% to 67.8%), and 13.4 points for Qwen3VL-30B-A3B (from 86.3% to 72.9%).

Furthermore, non-zero POPI rates confirm that these errors are directional rather than unstructured degradations, demonstrating that susceptibility to foreground demographic cues remains present in larger MLLMs. Because these models also differ in pretraining mixtures, vision-backbone resolution, and instruction tuning, this comparison indicates persistence across scale rather than isolating parameter count as a single controlled variable.

## Appendix E Computational Resources

In this section, we provide additional details regarding the computational resources required to replicate the experiments reported in this paper. The image generation pipeline was run on an NVIDIA RTX 5000 Blackwell and it took approximately 10 hours. Model inference was done on an NVIDIA H100 and it took no more than one day to run all experiments.

## Appendix F Dataset Licences

In this section, we mention the licenses under which each of the datasets we cite are released:

• Im2GPS3k: Apache 2.0

• YFCC4k: CC BY-NC-SA

• FairFace: CC BY 4.0

• GEOBIASET: CC BY-NC 4.0

Table 8: Geolocalization accuracy for the foreground-masking control experiment. For each edited image, the instance-specific SAM3 mask of the inserted subject is applied identically to its matched original background. Accuracies are reported as percentages varying the distance thresholds.
<table><tr><td>Source</td><td>Mask</td><td>≤1 km</td><td>≤25 km</td><td>≤200km</td><td>≤750km</td><td>≤2500km</td></tr><tr><td rowspan="2">Original Edited</td><td>None</td><td>11.1</td><td>35.6</td><td>51.6</td><td>68.8</td><td>81.9</td></tr><tr><td>None</td><td>4.3</td><td>16.3</td><td>25.4</td><td>38.1</td><td>55.3</td></tr><tr><td rowspan="2">Original Edited</td><td>Black</td><td>8.1</td><td>28.4</td><td>40.9</td><td>57.9</td><td>73.4</td></tr><tr><td>Black</td><td>8.1</td><td>27.8</td><td>40.2</td><td>56.1</td><td>72.7</td></tr><tr><td rowspan="2">Original Edited</td><td>Gray</td><td>7.5</td><td>27.7</td><td>40.8</td><td>58.6</td><td>73.6</td></tr><tr><td>Gray</td><td>7.4</td><td>26.7</td><td>39.5</td><td>56.0</td><td>71.7</td></tr><tr><td rowspan="2">Original Edited</td><td>Blur</td><td>8.9</td><td>29.4</td><td>42.7</td><td>60.6</td><td>76.7</td></tr><tr><td>Blur</td><td>7.8</td><td>24.8</td><td>36.8</td><td>53.6</td><td>70.9</td></tr></table>

## Appendix G Foreground Masking Controls

The comparison between original backgrounds and their counterfactual edits does not by itself separate three distinct factors: (i) geographic evidence lost through foreground occlusion, (ii) background drift introduced by the generative editing pipeline, and (iii) demographic cues carried by the inserted subject. We therefore perform a paired masking experiment to isolate the first two factors.

Using SAM3 [5], we segment the inserted subject in each of the 9,212 edited images. Each variant receives an instance-specific mask capturing the exact shape, scale, and placement of the generated person (using the union of masks if multiple instances are generated). Crucially, each mask is applied identically to both the edited image and its matched original background, ensuring identical occlusions so that residual differences reflect solely whether the visible background is authentic or synthesized. We evaluate three masking fills: (i) constant black, (ii) uniform gray (RGB = [128, 128, 128]), and (iii) Gaussian blur (σ = 15 pixels). Black and gray eliminate all visual features in the subject region, whereas Gaussian blur preserves coarse local illumination and chromaticity.

Effect of foreground occlusion. Occluding the subject region on original backgrounds reduces continent-level accuracy from 81.9% to 73.4% (black), 73.6% (gray), and 76.7% (blur), confirming that physical occlusion removes useful scene evidence. However, this drop is far smaller than the degradation observed when the synthetic person is visible (55.3%). Occlusion alone therefore cannot explain the observed performance loss.

Effect of generation-induced background changes. Comparing masked originals to masked edits isolates distortions outside the subject mask. Under black masking, continent-level accuracy differs by only 0.7 points (73.4% vs. 72.7%), and under gray masking by 1.9 points (73.6% vs. 71.7%). Although full-reference metrics (Sec. 3) confirm that pixel-level changes occur, these controls demonstrate that masked generative backgrounds retain nearly all task-relevant geographic evidence used by the model.

Approximate decomposition of continent-level degradation. For GeoCLIP, the total continentlevel degradation between original and edited images is 26.6 percentage points (81.9% vs. 55.3%). Combining the black-mask control with the face-obfuscation results (Sec. 5.5) yields an approximate loss attribution:

• Scene occlusion: 8.5 points (31.9% of the drop), measured by masking the original image.

• Background drift: 0.7 points (2.6%), measured by swapping the masked original with the masked edit.

• Non-facial subject cues: 9.0 points (33.9%), revealed by inserting the subject with an obfuscated face.

• Facial appearance: 8.4 points (31.6%), measured when blurring the face.

While diagnostic rather than a formal causal factorization, this shows that approximately 65.5% of the degradation stems directly from demographic features of the inserted person (facial and non-facial) rather than from scene destruction or editing artifacts.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state that the paper introduces GEOBIASET, a counterfactual benchmark for auditing demographic bias in worldwide image geolocalization, together with POPI metrics for measuring person-induced prediction shifts. These claims are supported by the dataset construction in Section 3, the evaluation protocol in Section 4, and the experimental results in Section 5.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: Section 6 discusses limitations including the simplified use of race and binary gender categories, the operational nature of race-associated geographic regions and clothing prompts, synthetic-to-real distribution shift, residual generation artifacts, and the limited focus on foreground people rather than other geolocalization cues.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not introduce theoretical results, theorems, or proofs.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper describes the dataset construction procedure, image-editing pipeline, quality-control prompt, MLLM geolocalization prompt, evaluated models, geodesic accuracy metrics, and POPI definitions. These details provide a reasonable path to reproducing or verifying the main experimental claims.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We release anonymized code at link https://anonymous.4open.science/ r/GeoBiaset/README.md and the dataset at link https://kaggle.com/datasets/ 6393b4d11746fe3e4af72d5c71438ea2e003283fec6044064becdfd82a68475a

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies that the benchmark consists of original background images and counterfactual edited variants, describes the seven race categories, gender and attire conditions, evaluated geolocalization systems and MLLMs, distance thresholds, POPI metrics, and prompting details. Experimental setting is discussed in Sec. 3 Since the evaluated systems are used as existing models rather than newly trained models, optimizer and training hyperparameters are not applicable.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [No]

Justification: We do not report confidence intervals or significance tests because all experiments are inference-only so run-to-run variance is negligible. Reported accuracy and POPI values are exact metrics computed over the full benchmark.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper discloses information about computational resources in App. E.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The paper explicitly frames GEOBIASET as an auditing benchmark for measuring demographic bias and discusses privacy, fairness, surveillance, and dual-use risks in Section 6.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section 6 discusses beneficial applications such as journalism, historical analysis, environmental monitoring, geography education, and open-source verification, while also acknowledging risks including surveillance, stalking, military targeting, privacy violations, and uneven harms from biased geolocation predictions.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The dataset documentation and release materials describe intended use for bias evaluation and mitigation, prohibit harmful identification or surveillance use, and include warnings that race-associated regions and attire prompts are operational measurement constructs rather than normative mappings between people and places.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper mentions the license of each dataset used in Appendix F.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper introduces GEOBIASET as a new counterfactual benchmark and documents its construction in Section 3, including the source background images, demographic taxonomy, face-pool selection, image-editing pipeline, quality-control procedure, and face-obfuscated variant. The appendices provide the exact image-editing, clothing, quality-control, and MLLM evaluation prompts, as well as the operational country-level taxonomy used for POPI evaluation.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: The paper uses multimodal large language models as evaluated geolocalization systems and uses Qwen3-VL as an automated quality-control model for detecting severe synthesis failures in generated images. These uses are part of the experimental protocol and dataset construction pipeline, and the corresponding prompts are documented in Appendix B.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.