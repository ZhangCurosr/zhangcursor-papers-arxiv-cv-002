# Front-to-Back: Benchmarking Vision-Language Models for Asymmetric Cross-View Vehicle Re-Identification

Moseli Mots’oehli<sup>1,2</sup> and Thulani Babeli<sup>1</sup>

<sup>1</sup> MindForge AI, Johannesburg, South Africa 2 University of Hawai’i at M¯anoa, Honolulu, HI, USA

Abstract. Matching the same vehicle across front and rear cameras is dificult because the cameras do not share a view and the vehicle’s appearance changes substantially. We introduce Front2Back-ReID, a benchmark of 500 manually verified vehicle handovers from 20 recording sequences in South Africa. Each example asks a model to match a vehicle highlighted in a front-camera image to the same vehicle among at least three candidates in a later rear-camera image. We evaluate seven zeroshot vision-language models, four image-retrieval baselines, and 25 human participants. Models are tested using full front RGB images, cropped target vehicles, and binary silhouettes. The strongest VLM achieved 76.6% Rank-1 accuracy on target crops, compared with 74.0% for the frozen SigLIP2 baseline; this diference was not statistically clear. Human participants achieved 94.0% accuracy with full images and 92.2% with target crops. Under our evaluation setup, enabling reasoning improved accuracy across all three input conditions for every model evaluated in both modes. We also found that VLMs generally performed worse on full scenes than on target crops. These results show that generalpurpose VLMs do not yet consistently outperform strong visual retrieval for front-to-rear vehicle matching, while humans remain substantially more reliable.

![](images/8892db90e483af0f9558e26fd624a641d7d4132ea639c0012748b22deb705dc3.jpg)  
Fig. 1: A highlighted front-view vehicle disappears through a blind interval and reappears in a closed rear gallery. The task is to select its unique match under full-RGB, crop, or silhouette front evidence.

## 1 Introduction

Although the Waymo Open Dataset’s End-to-End Driving collection and similar datasets provide images from eight cameras covering the full 360<sup>◦</sup> around the vehicle [29], resource-constrained settings motivate methods for simpler rigs with fewer views, making identity association across gaps in coverage particularly important. Recent work studies vehicle matching across non-overlapping cameras [11] and uses temporal features and motion prediction to maintain tracks through occlusions [18]. We focus on front-to-rear matching: identifying a vehicle seen ahead when it later appears behind the ego vehicle, without intervening observations, a fixed inter-view time interval, or a reliable front-to-rear transform.

Such handovers arise during overtakes, pass-bys, turns, merges, and intersection crossings. Large viewpoint changes [4] and new distractors in the rear scene complicate identity matching. We evaluate zero-shot vision-language models (VLMs) on a focused task: given one highlighted front target and a closed rear gallery, which candidate is the same vehicle? Each gallery contains at least three candidates, including exactly one match. This evaluates identity association across an observation gap.

Motivated by diagnostic evaluations of VLM visual grounding [6], we examine whether zero-shot VLMs ofer an advantage over frozen visual retrieval and how their performance changes with the supplied evidence. We keep the rear gallery unchanged and show the front target in three forms: the full RGB scene, a target crop, and a binary silhouette. These comparisons reveal how models respond to diferent presentations of the same target. Because context, target scale, and representation also change, they do not isolate the efects of context or reasoning alone.

Front2Back-ReID contains 500 manually verified handovers from 20 South African recording sequences. We evaluate seven VLMs across the three input conditions, four crop-based retrieval controls, and 25 human participants on full-RGB and target-crop trials. The strongest completed VLM achieves 76.6% Rank-1 accuracy on crops, versus 74.0% for frozen SigLIP2, but this diference is not statistically significant. Humans achieve 94.0% on full scenes and 92.2% on crops. Under the shared prompt protocol, crops outperform full scenes in every informative, VLM comparison.

Our contributions are:

1. A front-to-rear association benchmark: 500 asymmetric, closed-gallery handovers from 20 South African recording sequences, with manually verified identities and dificulty annotations.

2. A diagnostic study of visual evidence: seven zero-shot VLMs evaluated with full scenes, target crops, and binary silhouettes, alongside four crop-based retrieval controls and a 25-participant human reference. The best completed VLM shows no statistically significant advantage over SigLIP2 on crops, and crops outperform full scenes in every informative, completed VLM comparison.

## 2 Related Works

## 2.1 Cross-view Vehicle Re-identification

Vehicle Re-ID has largely been studied as retrieval across fixed surveillance cameras. VehicleID and VERI-Wild match vehicle appearance across large city camera networks [12,14], and VeRi-776 and CityFlow add spatial and temporal cues to narrow the search [13, 20]. In these settings, cameras are static and often elevated, galleries are large, and the recordings come mostly from Asia, North America, and Europe, with little to no coverage of Africa.

Large viewpoint change, especially between front and rear views, is a wellknown dificulty. Methods address it with viewpoint-dependent metrics [4], local features and reranking [27], generated-view adaptation [25], alignment of visible regions [15], view-invariant pretraining [26], and fusion of complementary views [30]. We do not propose another such method. Instead, we ask how well existing systems use identity cues across views when evaluated under shared conditions.

Front2Back-ReID changes both the viewpoint setting and the geography. The cameras ride on the ego vehicle, so a target leaves the front camera’s view and reappears in the rear camera from a very diferent viewpoint after a blind interval, among $K _ { i } \geq 3$ candidates supplied by surrounding trafic. Fixed galleries and controlled front evidence let us compare zero-shot VLMs, frozen retrieval models, and humans on the same decisions. The recordings come from South African roads, where the mix of vehicles, the layout of the road, and the driving conditions difer from those of existing benchmarks. To our knowledge, Front2Back-ReID is the first vehicle Re-ID benchmark recorded on Southern African roads.

## 2.2 Foundation Models for Visual Retrieval

Foundation models are increasingly used for vehicle Re-ID, but usually after task-specific training; CLIP-ReID, for example, fine-tunes CLIP on standard person and vehicle benchmarks [10]. Used frozen, encoders such as DINOv2 [17] and SigLIP2 [22] are strong general-purpose matchers, but they are trained to group similar content, not to separate individual instances: two white sedans may look alike to them. It has been shown that this blind spot carries over to multimodal models, whose errors trace to visually distinct images with similar CLIP embeddings [21]. Since modern VLMs commonly rely on pretrained visual encoders, we use frozen DINOv2 and SigLIP2 as representation baselines. ranking rear candidates by crop-level embedding similarity therefore lets us ask whether language supervision improves visual identity matching, and whether generative multimodal models add gains beyond embedding similarity on the cross-view vehicle Re-ID task.

## 2.3 Visual and Spatial Reasoning in VLMs

Recent VLMs have gained the abilities our task appears to need. LLaVA-OneVision is trained across single-image, multi-image, and video tasks [8]; Qwen2.5-VL processes images at dynamic resolution and can localize objects [1]; SpatialVLM improves spatial reasoning through dedicated supervision [3]; and $\mathrm { V } ^ { \ast }$ uses languageguided search to find small details in crowded, high-resolution images [28]. Applying such models zero-shot to Re-ID in driving is only beginning: the closest study has a VLM describe each object crop and matches the descriptions in language space [2]. We instead ask VLMs to compare images directly, with no video or cross-rig geometry to fall back on. Full scenes versus target crops test whether context helps, and running each model with and without reasoning(for applicable models) tests whether extended reasoning improves performance.

## 2.4 Diagnostic Benchmarks for Visual Grounding

Several benchmarks test whether VLMs ground their answers in the image. HallusionBench uses controlled question pairs to expose hallucination and visual illusion [6], MuirBench pairs multi-image questions with unanswerable variants [24], and MIHBench tests whether models keep object identity consistent across images, finding that errors depend on how many images are shown and where distractors appear [9]. These benchmarks probe grounding by varying the question or the image set. Front2Back-ReID instead holds the rear gallery fixed and varies only the form of the front evidence (RGB, crop, or silhouette), on real handovers where viewpoint reverses and trafic supplies the distractors. Frozen retrieval baselines and human evaluators on the same galleries show whether an instance is solvable from appearance alone.

## 3 Benchmark and Task

## 3.1 Task Definition

An overview of Front2Back-ReID is shown in Fig. 2, including benchmark construction, task setup, evidence conditions, and evaluation. Each example pairs a front-left image containing one highlighted vehicle with a later rear-left image containing $K _ { i } \geq 3$ labelled candidates. The task is to select the unique candidate matching the front target; all others are distractors (Fig. 1). No intervening video, front-to-rear transform, or range estimates are supplied to matchers.

## 3.2 Data Collection and Annotation

We recorded 20 sequences on diferent days and at diferent locations. We mounted two low-cost MMLove stereo cameras, shown in Fig. 3, on the vehicle: one on the windscreen facing forward and one on the rear window facing backward. Each costs under US\$100, has a 60 mm stereo baseline, and records at 1920 × 1080 pixels and 30 fps. Camera streams were software-synchronized during recording and downsampled to 10 fps for processing.

![](images/31791fcac40931ec1fe1720faa02b6bcfa9f47c96ffe55c371e7e2c14f446efa.jpg)  
Fig. 2: Overview of Front2Back-ReID. Curators correct and verify YOLOE-26 proposals from front and rear recordings, yielding 500 handovers $( \mathrm { S } ^ { 2 } \mathrm { M } ^ { \dot { 2 } }$ depth aids dificulty analysis only). A matcher must find the unique rear-gallery match of a highlighted front-left target after a blind interval, given full RGB, crop, or silhouette. We compare zero-shot VLMs, frozen retrieval baselines, and humans on the same decisions.

Curators corrected initial YOLOE-26 [23] generated bboxes and Silhouette mask polylines, selected front targets, and verified their rear-view matches. We estimated the range of the front-target from the rectified front stereo using the variant S of $\mathrm { S ^ { 2 } M ^ { 2 } }$ [16] and the valid median disparity within each target bbox. These distance estimates are used to support dificulty analysis only.

![](images/1ec8d2ebddaf99590b0c7460a9fac27c5b079897189663970687a42ecf2a20fa.jpg)  
Fig. 3: Standard stereo cameras used for data collection.

Table 1: Front2Back-ReID summary. Each record is one closed-gallery handover.
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Hand-labeled associations</td><td>500</td></tr><tr><td>Recordings</td><td>20</td></tr><tr><td>Task</td><td>Closed gallery, 1-of-  $K _ { i } , K _ { i } \geq 3 ,$ </td></tr><tr><td>Canonical views</td><td>Front-left, rear-left</td></tr><tr><td>Model conditions</td><td>3: RGB, crop, mask</td></tr><tr><td>Human reference conditions</td><td>2: RGB, crop</td></tr><tr><td>Human reference judgments</td><td> $2 5 \times 4 0 = 1 0 0 0$ </td></tr></table>

## 3.3 Construction and statistics

Beyond the setup in Table 1, dificulty comes from both sides of a handover: how many rear vehicles compete with the target, and how far, small, or occluded the target is (Fig. 4). We measure rear-side competition with $V _ { i } ,$ the number of vehicles among the rear candidates. This difers from the gallery size $K _ { i } ,$ the number of candidates a matcher chooses from, on 164 pairs, so the $3 , 4 \mathrm { - } 5 ,$ and $\geq 6$ groups hold 45/163/292 pairs under $V _ { i }$ but 38/127/335 under $K _ { i } .$ . All countstratified results use $V _ { i }$ . We further detail this distinction, scene composition, and recording durations in Supplementary Secs. S1 and S8.

![](images/d7ee0e2170083247688eb2ade71f4450d6a3aca71329828a9f0b1f96aab1604f.jpg)

![](images/5ba07f02e5ebad93c92745f6505e5d3d316b2bc2c81c90a858f848bdc3121204.jpg)

![](images/3bd58c8b5598e6f43e5e792abb8b8c2c09fbcd241693b93da635de44326d662f.jpg)

![](images/720be41805a8662cf18b812251ce0cc90ce99fd0c4449fc3e9832f9e899f1673.jpg)  
Fig. 4: Benchmark dificulty distributions: vehicle-only candidate count, approximate front-target range, normalized target scale, and occlusion. Range uses the 341 pairs with valid $\mathrm { S } ^ { 2 } \mathrm { M } ^ { 2 }$ stereo depth; the other distributions use all 500 pairs.

## 3.4 Dificulty Stratification

We group pairs by what makes matching hard: the number of rear vehicles $V _ { i } .$ and the target’s size, occlusion, and truncation in each view. Size split into three equal beans, and front-target depth into five equal bins over the 341 pairs with valid $\mathrm { S ^ { 2 } M ^ { 2 } }$ estimates. These dificulty attributes are used only for analysis and are not given as input to models or evaluators. We provide the thresholds in Supplementary Sec. S1 and analyze depth and size sensitivity in Sec. S7.

## 4 Evaluation Protocol

Each model sees the same rear gallery with three versions of the front evidence, shown in Fig. 5: the full RGB scene with the target marked, an RGB crop of the target, and a binary silhouette. The crop and silhouette each remove something, but not only that. The crop drops the surrounding scene but also enlarges the target, and the silhouette keeps only shape, dropping color and texture. We therefore treat diferences between conditions as efects of how the input is presented, not of a single cue. Rear candidates are labeled only by their aliases (C1, C2, . . . ); box coordinates, detector classes and confidences, and other diagnostic metadata are neither drawn on the images nor included in the prompts. Figure 6 gives the full prompt. All conditions share the same task rules and output format; only the description of the front evidence changes. We set temperature to 0 where the model allows it.

![](images/2119cddcb8546d72bbcc9ac74a02efad83ea880cb561292fdd6daea260092182.jpg)  
Fig. 5: Representative benchmark inputs. Columns show the marked full-RGB query, the RGB target crop, the binary silhouette used in the mask condition, and the rear candidate gallery. The silhouette was supplied as black foreground on white background and contained no color, texture, lights, windows, or markings. Green match boxes and red distractor boxes are reader annotations in this figure only.

![](images/85d5967ab49b5d97cf7b5796f39b9fad9514049b983a10df8511ff04627a71e8.jpg)  
Fig. 6: Prompt specification used for all VLM evaluations. The upper and lower boxes give the decision rules and output schema common to all conditions; the middle boxes state the visual evidence available in each condition.

## 5 Models, Baselines, and Human Reference

Controls. Four non-generative baselines isolate progressively richer visual cues. Uniform sampling over the $K _ { i }$ candidates fixes chance. An HSV histogram matched by Bhattacharyya distance [7, 19] measures what color alone recovers. DINOv2 ViT-B/14 [17] and SigLIP2 Base Patch16-224 [22] rank candidate crops by cosine similarity of frozen embeddings, measuring how much identity is recoverable without language.

VLMs. Table 2 lists the evaluated VLMs, chosen to span the options most accessible to resource-constrained researchers and practitioners: a small open model run locally, larger open models served through a hosted API, and closed frontier models current as of June 2026 as an upper reference.

Human reference. We recruited twenty-five adults, each of whom completed 40 trials (20 RGB, 20 crop), selecting the matching rear bbox within a 25-minute limit and without AI assistance. We assign pairs to participants with a seeded random procedure, constrained so that each participant’s trials are balanced across dificulty levels, scene conditions, and source videos. Every pair receives one judgment per condition from diferent participants; no participant sees both versions of a pair. All sessions were completed, yielding 1000 judgments.

Table 2: Evaluated VLMs. Experiments were run between 24 June and 5 July 2026 using the listed models. Parameter counts unavailable from providers are marked $\mathrm { n / d } .$
<table><tr><td>Model</td><td></td><td></td><td>Family Access Provider</td><td>Params</td></tr><tr><td>Open-family models</td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-OneVision 0.5B</td><td>Open</td><td>Local</td><td>Local</td><td>0.5B</td></tr><tr><td>Llama 4 Scout</td><td>Open</td><td>API</td><td>Groq</td><td>17B/109B</td></tr><tr><td>Qwen 3.6 27B</td><td>Open</td><td>API</td><td>Groq</td><td>27B</td></tr><tr><td>Closed hosted</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5</td><td>Closed</td><td>API</td><td>OpenAI</td><td>n/d</td></tr><tr><td>GPT-5.4 mini</td><td>Closed</td><td>API</td><td>OpenAI</td><td>n/d</td></tr><tr><td>L Gemini 2.5 Pro</td><td>Closed</td><td>API</td><td>Google AI Studio</td><td>n/d</td></tr><tr><td>1 Gemini 2.5 Flash</td><td>Closed</td><td>API</td><td>Google AI Studio</td><td>n/d</td></tr></table>

## 6 Evaluation Metrics

For batch i, the target appears exactly once among $K _ { i }$ candidates. The primary metric is Rank-1 accuracy:

$$
{ \mathrm { R a n k - 1 } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { / k } [ y _ { i } = \arg \operatorname* { m a x } _ { k } p _ { i , k } ] .\tag{1}
$$

Chance is $1 / K _ { i }$ . Because galleries are event-local, full-dataset retrieval rank is not meaningful.

We report two-sided 95% bias-corrected and accelerated (BCa) confidence intervals [5] from 100,000 bootstrap resamples over pairs (seed 32025). Model Outputs that could not be parsed or named an unlisted candidate count as incorrect. Human accuracy and decision time are reported per condition, each bootstrapped over its 500 judgments; human–model comparisons are paired over shared pairs. Since pairs from the same recording or participant may be correlated, so these intervals may be somewhat narrow.

## 7 Results and Analysis

Tables 3 and 4 report reasoning-disabled and reasoning-enabled VLM results, respectively. Bracketed values denote 95% BCa confidence intervals. All reported model results are based on complete N = 500 runs. Some Qwen reasoningenabled runs remained incomplete after multiple retries because of repeated inference failures, including request errors, timeouts, and invalid or missing outputs. We therefore exclude these runs from the results and document them separately as Supplementary material, Sec. S9. The human-reference row is based on the completed 25-participant study and 40 question pairs per person.

We organize our analysis around three questions: how retrieval and reasoning afect VLM performance, how humans compare in accuracy and response time, and which visual factors drive dificulty for humans and models.

Table 3: Rank-1 accuracy (%) without reasoning, with 95% BCa intervals. Retrieval controls use crops only. Gemini 2.5 Pro is excluded here because its reasoning cannot be disabled. Bold and underline mark the best and second-best non-human result per column. $\varDelta _ { \mathrm { c t x } }$ is full RGB minus target crop: every VLM except the C1-only LLaVA-OneVision loses accuracy when given the full scene, while humans gain slightly. On crops, the best VLMs perform on par with the frozen SigLIP2 encoder, and all remain far below humans.

<table><tr><td>Method</td><td>Full RGB</td><td>Target crop</td><td>Silhouette</td><td> $\pmb { \varDelta } _ { \mathbf { c t x } }$ </td></tr><tr><td>Retrieval controls</td><td></td><td></td><td></td><td></td></tr><tr><td>Random gallery</td><td></td><td>17.8 [14.4–21.2]</td><td></td><td></td></tr><tr><td>HSV histogram</td><td></td><td>47.6 [43.0–51.8]</td><td></td><td></td></tr><tr><td>DINOv2 ViT-B/14</td><td></td><td>49.4 [44.8–53.6]</td><td></td><td></td></tr><tr><td>SigLIP2 Base</td><td></td><td>74.0 [69.8–77.6]</td><td></td><td></td></tr><tr><td>Open-family VLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-OneVision 0.5B</td><td>27.4 [23.4–31.2]</td><td>27.4 [23.4–31.2]</td><td>27.4 [23.4–31.2]</td><td>0.0</td></tr><tr><td>Llama 4 Scout</td><td>54.8 [50.2–59.0]</td><td> $6 1 . 4 \ [ 5 6 . 8 - 6 5 . 4 ]$ </td><td>34.2 [30.0–38.2]</td><td>-6.6</td></tr><tr><td>Qwen 3.6 27B</td><td>63.6 [59.2–67.6]</td><td>74.6 [70.4–78.0]</td><td>32.2 [28.0–36.2]</td><td>-11.0</td></tr><tr><td>Closed hosted VLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5</td><td> $5 8 . 4 \ [ 5 3 . 8 - 6 2 . 4 ]$ </td><td> $7 4 . 8 \ [ 7 0 . 6 - 7 8 . 2 ]$ </td><td>36.8[32.4–40.8]</td><td>-16.4</td></tr><tr><td>GPT-5.4 mini</td><td> $4 4 . 6 \ : [ 4 0 . 0 - 4 8 . 8 ]$ </td><td> $5 8 . 4 \ [ 5 3 . 8 - 6 2 . 4 ]$ </td><td> $2 8 . 6 \ [ 2 4 . 6 - 3 2 . 6 ]$ </td><td>-13.8</td></tr><tr><td>Gemini 2.5 Flash</td><td> $5 0 . 6 \ [ 4 6 . 0 - 5 4 . 8 ]$ </td><td> $6 4 . 0 \ [ 5 9 . 6 - 6 8 . 0 ]$ </td><td> $2 5 . 2 \ [ 2 1 . 4 - 2 9 . 0 ]$ </td><td>-13.4</td></tr><tr><td>Human reference</td><td>94.0 [91.6–95.8]</td><td> $9 2 . 2 \ [ 8 9 . 6 - 9 4 . 4 ]$ </td><td></td><td>+1.8</td></tr></table>

Table 4: Rank-1 accuracy (%) with medium-efort reasoning. $\varDelta _ { r }$ is the change from the same model without reasoning (Table 3). Every model evaluated in both modes improves, but full RGB still trails the crop by 11–14 points. Bold and underline mark the best and second-best model.
<table><tr><td></td><td colspan="2">Full RGB</td><td colspan="2">Target crop</td><td colspan="2">Silhouette</td><td></td></tr><tr><td>Method</td><td>Rank-1</td><td>∆r</td><td>Rank-1</td><td> $\varDelta _ { r }$ </td><td>Rank-1</td><td> $\varDelta _ { r }$ </td><td> $\pmb { \varDelta } _ { \mathbf { c t x } }$ </td></tr><tr><td>GPT-5.5</td><td>62.8 [58.2–66.8]</td><td>+4.4</td><td>76.6[72.6–80.0]</td><td>+1.8</td><td>43.8[39.4–48.0]</td><td>+7.0</td><td>-13.8</td></tr><tr><td>GPT-5.4 mini</td><td>58.8 [54.2–62.8]</td><td>+14.2</td><td>71.0 [66.6–74.6]</td><td>+12.6</td><td> $\underline { { 4 0 . 0 } } \ [ 3 5 . 6 - 4 4 . 2 ]$ </td><td></td><td> $+ 1 1 . 4 \quad - 1 2 . 2 \quad$ </td></tr><tr><td>Gemini 2.5 Pro</td><td>60.6 [56.0–64.6]</td><td></td><td> $\underline { { 7 1 . 8 } } \ : [ 6 7 . 4 - 7 5 . 4 ]$ </td><td></td><td> $3 4 . 6 \ [ 3 0 . 4 - 3 8 . 6 ]$ </td><td></td><td>-11.2</td></tr><tr><td>L Gemini 2.5 Flash</td><td>54.4 [49.8–58.6]</td><td>+3.8</td><td> $6 7 . 2 \ [ 6 2 . 8 - 7 1 . 0 ]$ </td><td>+3.2</td><td> $3 2 . 0 \ [ 2 7 . 8 - 3 6 . 0 ]$ </td><td>+6.8</td><td>-12.8</td></tr><tr><td>Human reference</td><td>94.0 [91.6–95.8]</td><td></td><td>92.2 [89.6–94.4]</td><td></td><td></td><td></td><td>+1.8</td></tr></table>

Reasoning helps selectively and does not beat frozen retrieval. With reasoning enabled, GPT-5.5 is the best crop model, 2.6 points above SigLIP2. The paired

![](images/1599343124759ae206770d676a4754270ca61e3ecbebb3eca61e61930ea1dcf6.jpg)  
Fig. 7: Target-crop Rank-1 accuracy without reasoning, by vehicle-only candidate count, front-target range, scale, and occlusion. The panel labeled “Rear-gallery size” uses vehicle-only count $V _ { i } ,$ not gallery size $K _ { i } .$ . The range panel uses the 341 pairs with valid $\mathrm { S } ^ { 2 } \mathrm { M } ^ { \mathrm { \bar { 2 } } }$ depth, split into five equal-frequency bins; the others use all 500. Smaller targets and heavier occlusion generally lower accuracy; range and candidatecount trends are weaker.

![](images/7716de8887e9c2e3277d79c94c1f8f1633e699ff20ad654b8356917c54c75a3f.jpg)  
Fig. 8: Target-crop accuracy without reasoning by weather, road context, and correct rear-target scale and occlusion. GPT-5.5 and Llama 4 Scout fall sharply on small or heavily occluded targets, while SigLIP2 stays flatter. Construction is omitted from the road-context panel (two pairs).

95% BCa interval, [−2.2, 7.4], includes zero: GPT-5.5 alone is correct on 81 pairs and SigLIP2 alone on 68, so the advantage is not significant. We find that reasoning gains are also uneven (Fig. 9): GPT-5.4 mini gains 11–14 points in every condition, whereas GPT-5.5 and Gemini 2.5 Flash gain most on silhouettes and show minimal gain on crops. We include the list of all paired contrasts in Supplementary Sec. S4.

The same pattern holds for every VLM except LLaVA-OneVision: without reasoning, crops beat full scenes by 6.6–16.4 points, and silhouettes fall 27.2–42.4 points below crops. Because each change also alters target scale or representation, these gaps reflect how the input is presented, not context alone. LLaVA-OneVision selects C1 on every trial, so its identical 27.4% scores simply equal the share of pairs whose match is C1, even though its outputs are valid JSON. We detail model explanations and a pair missed by all ten complete crop runs in Supplementary Sec. S6, and document the incomplete Qwen runs in Sec. S9.

Humans remain far ahead. Humans reach 94.0% on full RGB and 92.2% on crops, 31.2 and 15.6 points above reasoning-enabled GPT-5.5, and 18.2 points above SigLIP2 on crops. Incorrect answers also take longer (Fig. 10): median decision time is 20.0 versus 7.8 s for incorrect and correct answers on full RGB, and 20.5 versus 7.0 s on crops. This may reflect harder pairs or hesitation. Fig. 11 shows five of the ten pairs missed in both conditions; with only two judgments per pair, this is a coarse dificulty signal. Supplementary Sec. S5 repeats the timing analysis without responses over 60 s that may be due to our tool’s interface issues.

Paired effect of provider reasoning mode  
![](images/a7621abecac6de0460aef122a5acc5db72dd867bbb6b3bdf84251266f5fcf600.jpg)  
Fig. 9: Change in Rank-1 accuracy with medium-efort reasoning for complete matched runs. Diferences and uncertainty intervals are paired by annotation; intervals crossing zero do not establish a directional gain.

![](images/8849da8973794b60da2e4e7fbaae816f1e878fbf06f2c084238b1ee71c71a43e.jpg)  
Fig. 10: Human decision time by outcome. Points show medians and bars the interquartile range (IQR). All responses contribute, including those above 60 s.

Small rear targets separate humans from models. Target scale separates humans from models more than vehicle count does (Figs. 7, 8, and 12). On the smallest third of rear targets, human accuracy (pooled over both conditions) stays at 93.2%, while crop accuracy without reasoning drops for every model and falls to 38.7% for Llama 4 Scout. Human accuracy is nearly flat across these strata, suggesting that models miss identity cues people can still use, although humans also miss some pairs in both conditions (Fig. 11). Front-target depth shows no consistent trend across the five depth bins; we analyze depth sensitivity further in Supplementary Sec. S7.

![](images/367e9724075937de516dcdaa61d65a0795417b74f987aa9844ff54a910b6b030.jpg)  
Fig. 11: Five of the ten pairs missed in both human conditions. Columns show the full query, target crop, and rear gallery; green marks the match and red the distractors. Each pair received one judgment per condition.

Human reference across difficulty strata (both conditions pooled)  
![](images/91228e6ad4447210080b0ba7972e241ba850b0863a0840c4ecd776d5bbaeb99d.jpg)  
Fig. 12: Human Rank-1 accuracy by dificulty stratum, pooled over full RGB and target crops (1,000 judgments from 25 participants). Uncertainty intervals resample responses. Accuracy remains high even for small or heavily occluded targets, where models degrade (Figs. 7 and 8).

## 8 Limitations

Front2Back-ReID covers 500 handovers from 20 recordings in five South African cities, about 1.9 hours of driving. While limited, it is designed for controlled comparison of methods on real handovers, not for broad claims about other countries, fleets, or sensors. Daytime scenes dominate (97.6%), so night and adverse weather are underrepresented, and depth is available for 341 pairs, which limits range analysis to that subset. Our evidence conditions change more than one factor at a time; a crop, for example, removes context but also enlarges the target, so we interpret them as efects of input presentation. Finally, our intervals treat pairs as independent, and each pair has one human judgment per condition. Resampling by recording and collecting more judgments per pair would give more conservative intervals and a sharper human reference.

## 9 Conclusion

We introduced Front2Back-ReID, a benchmark of 500 verified front-to-rear vehicle handovers recorded with low-cost cameras on South African roads, where no calibrated transform links the front and rear views. Holding each rear gallery fixed, we varied the front evidence between full scene, target crop, and silhouette, and compared zero-shot VLMs, frozen retrieval models, and 25 human participants on the same decisions. Three findings stand out.

First, the best VLM, reasoning-enabled GPT-5.5, reaches 76.6% on crops, not significantly above the frozen SigLIP2 encoder at 74.0%, while humans reach 92.2–94.0%. Secondly, full scenes yield lower accuracy than target crops for every informative VLM run by 6.6–16.4 points without reasoning, whereas humans score 1.8 points higher with full RGB scenes. Cropping also changes target scale, so this diference cannot be attributed to context alone; reasoning does not remove the observed VLM gap.

Third, small rear targets separate models from humans, whose accuracy stays nearly flat across dificulty strata. Front2Back-ReID thus ofers a focused test of cross-view identity association for settings where calibrated multi-sensor rigs are out of reach. We plan to extend this work to night driving, diverse weather conditions and more regions, adding the intervening video so methods can use motion between views, and developing methods that make better use of scene context for identity matching, rather than being distracted by it.

## Acknowledgements

We thank the 25 participants in the human evaluation for their time, and Dalitso Chomey for valuable discussions on the paper.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923 (2025), https://arxiv.org/abs/2502.13923 4

2. Borges, E., Abreu, M., Garrote, L., Nunes, U.J.: Zero-shot semantic reidentification for autonomous driving: A VLM baseline study (2026) 4

3. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024), https://openaccess.thecvf.com/content/CVPR2024/ papers/Chen\_SpatialVLM\_Endowing\_Vision- Language\_Models\_with\_Spatial\_ Reasoning\_Capabilities\_CVPR\_2024\_paper.pdf 4

4. Chu, R., Sun, Y., Li, Y., Liu, Z., Zhang, C., Wei, Y.: Vehicle re-identification with viewpoint-aware metric learning. In: IEEE/CVF International Conference on Computer Vision (2019), https://openaccess.thecvf.com/content\_ICCV\_ 2019/html/Chu\_Vehicle\_Re-Identification\_With\_Viewpoint-Aware\_Metric\_ Learning\_ICCV\_2019\_paper.html 2, 3

5. Efron, B.: Better bootstrap confidence intervals. Journal of the American Statistical Association 82(397), 171–185 (1987). https://doi.org/10.1080/01621459. 1987.10478410 9

6. Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., Manocha, D., Zhou, T.: HallusionBench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14375–14385 (2024), https://openaccess.thecvf.com/content/CVPR2024/ html/Guan\_HallusionBench\_An\_Advanced\_Diagnostic\_Suite\_for\_Entangled\_ Language\_Hallucination\_and\_CVPR\_2024\_paper.html 2, 4

7. Hafner, J., Sawhney, H.S., Equitz, W., Flickner, M., Niblack, W.: Eficient color histogram indexing for quadratic form distance functions. IEEE Transactions on Pattern Analysis and Machine Intelligence 17(7), 729–736 (1995). https://doi. org/10.1109/34.391417 8

8. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., Li, C.: LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024), https://arxiv.org/abs/2408.03326 4

9. Li, J., Wu, M., Jin, Z., Chen, H., Ji, J., Sun, X., Cao, L., Ji, R.: MIHBench: Benchmarking and mitigating multi-image hallucinations in multimodal large language models. arXiv preprint arXiv:2508.00726 (2025), https://arxiv.org/abs/2508. 00726 4

10. Li, S., Sun, L., Li, Q.: CLIP-ReID: Exploiting vision-language model for image reidentification without concrete text labels. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 1405–1413 (2023) 3

11. Lin, Y., Lockyer, S., Sui, M., Gan, L., Stanek, F., Zarbock, M., Li, W., Evans, A., Zhang, N.: RoundaboutHD: High-resolution real-world urban environment benchmark for multi-camera vehicle tracking. arXiv preprint arXiv:2507.08729 (2025). https://doi.org/10.48550/arXiv.2507.08729, https://arxiv.org/abs/2507. 08729 2

12. Liu, H., Tian, Y., Wang, Y., Pang, L., Huang, T.: Deep relative distance learning: Tell the diference between similar vehicles. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 2167–2175 (2016). https://doi.org/10. 1109/CVPR.2016.238, https://openaccess.thecvf.com/content\_cvpr\_2016/ html/Liu\_Deep\_Relative\_Distance\_CVPR\_2016\_paper.html 3

13. Liu, X., Liu, W., Mei, T., Ma, H.: A deep learning-based approach to progressive vehicle re-identification for urban surveillance. In: European Conference on Computer Vision. pp. 869–884 (2016). https://doi.org/10.1007/978-3-319-46475-6\_53, https://xinchenliu.com/papers/2016\_ECCV\_PVID.pdf 3

14. Lou, Y., Bai, Y., Liu, J., Wang, S., Duan, L.Y.: VERI-Wild: A large dataset and a new method for vehicle re-identification in the wild. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019), https://openaccess.thecvf. com/content\_CVPR\_2019/html/Lou\_VERI- Wild\_A\_Large\_Dataset\_and\_a\_New\_ Method\_for\_Vehicle\_CVPR\_2019\_paper.html 3

15. Meng, D., Li, L., Liu, X., Li, Y., Yang, S., Zha, Z.J., Gao, X., Wang, S., Huang, Q.: Parsing-based view-aware embedding network for vehicle re-identification. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020), https://openaccess.thecvf.com/content\_CVPR\_2020/papers/Meng\_Parsing-Based \_ View - Aware \_ Embedding \_ Network \_ for \_ Vehicle \_ Re - Identification \_ CVPR\_2020\_paper.pdf 3

16. Min, J., Jeon, Y., Kim, J., Choi, M.: S<sup>2</sup>M<sup>2</sup>: Scalable stereo matching model for reliable depth estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025) 5

17. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., J’egou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024), https://arxiv.org/abs/2304.07193 3, 8

18. Pang, Z., Li, J., Tokmakov, P., Chen, D., Zagoruyko, S., Wang, Y.X.: Standing between past and future: Spatio-temporal modeling for multi-camera 3D multiobject tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), https://arxiv.org/abs/2302.03802 2

19. Swain, M.J., Ballard, D.H.: Color indexing. International Journal of Computer Vision 7(1), 11–32 (1991). https://doi.org/10.1007/BF00130487 8

20. Tang, Z., Naphade, M., Liu, M.Y., Yang, X., Birchfield, S., Wang, S., Kumar, R., Anastasiu, D., Hwang, J.N.: CityFlow: A city-scale benchmark for multitarget multi-camera vehicle tracking and re-identification. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8797–8806 (2019), https: //openaccess.thecvf.com/content\_CVPR\_2019/html/Tang\_CityFlow\_A\_City-Scale\_Benchmark\_for\_Multi-Target\_Multi-Camera\_Vehicle\_Tracking\_and\_ CVPR\_2019\_paper.html 3

21. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9568–9578 (2024), https://openaccess. thecvf.com/content/CVPR2024/html/Tong\_Eyes\_Wide\_Shut\_Exploring\_the\_ Visual\_Shortcomings\_of\_Multimodal\_LLMs\_CVPR\_2024\_paper.html 3

22. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., H’enaf, O., Harmsen, J., Steiner, A., Zhai, X.: Siglip 2: Multilingual vision-language encoders with

improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025), https://arxiv.org/abs/2502.14786 3, 8

23. Wang, A., Liu, L., Chen, H., Lin, Z., Han, J., Ding, G.: YOLOE: Real-time seeing anything. In: IEEE/CVF International Conference on Computer Vision (2025), https://arxiv.org/abs/2503.07465 5

24. Wang, F., Fu, X., Huang, J.Y., Li, Z., Liu, Q., Liu, X., Ma, M.D., Xu, N., Zhou, W., Zhang, K., Yan, T., Mo, W., Liu, H.H., Lu, P., Li, C., Xiao, C., Chang, K.W., Roth, D., Zhang, S., Poon, H., Chen, M.: MuirBench: A comprehensive benchmark for robust multi-image understanding. In: International Conference on Learning Representations (2025), https://proceedings.iclr.cc/paper\_files/ paper/2025/hash/9cf6139382f98623d08cc595622f3fb1-Abstract-Conference. html 4

25. Wang, Q., Min, W., Han, Q., Yang, Z., Xiong, X., Zhu, M., Zhao, H.: Viewpoint adaptation learning with cross-view distance metric for robust vehicle reidentification. Information Sciences 564, 71–84 (2021). https://doi.org/10. 1016/j.ins.2021.02.013, https://www.sciencedirect.com/science/article/ pii/S0020025521001559 3

26. Wang, Q., Zhang, Z., Wang, D., Gai, D., Xiong, X., Xu, J., Zhou, R.: VehicleMAE: View-asymmetry mutual learning for vehicle re-identification pre-training via masked autoencoders. In: IEEE/CVF International Conference on Computer Vision. pp. 4701–4711 (2025), https://openaccess.thecvf.com/content/ ICCV2025 / html / Wang \_ VehicleMAE \_ View - asymmetry \_ Mutual \_ Learning \_ for \_ Vehicle\_Re- identification\_Pre- training\_via\_Masked\_ICCV\_2025\_paper. html 3

27. Wang, Y., Li, H., Wei, Y., Wang, C., Wang, L.: Vehicle re-identification based on unsupervised local area detection and view discrimination. Image and Vision Computing 104, 104008 (2020). https://doi.org/10.1016/j.imavis. 2020.104008, https://www.sciencedirect.com/science/article/abs/pii/ S0262885620301402 3

28. Wu, P., Xie, S.: V\*: Guided visual search as a core mechanism in multimodal LLMs. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13084–13094 (2024), https://openaccess.thecvf.com/content/CVPR2024/ html/Wu\_V\_Guided\_Visual\_Search\_as\_a\_Core\_Mechanism\_in\_Multimodal\_ CVPR\_2024\_paper.html 4

29. Xu, R., Lin, H., Jeon, W., Feng, H., Zou, Y., Sun, L., Gorman, J., Tolstaya, E., Tang, S., White, B., Sapp, B., Tan, M., Hwang, J.J., Anguelov, D.: WOD-E2E: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026), https://openaccess.thecvf.com/content/ CVPR2026/papers/Xu\_WOD-E2E\_Waymo\_Open\_Dataset\_for\_End-to-End\_Driving\_ in\_Challenging\_Long-tail\_CVPR\_2026\_paper.pdf 2

30. Zheng, A., Zhang, C., Li, C., Tang, J., Tan, C.: Multi-query vehicle reidentification: Viewpoint-conditioned network, unified dataset and new metric. IEEE Transactions on Image Processing 32, 5948–5960 (2023). https: //doi.org/10.1109/TIP.2023.3326691, https://aihuazheng.github.io/ publications / pdf / 2023 / 2023 - Multi - Query \_ Vehicle \_ Re - Identification \_ Viewpoint-Conditioned\_Network\_Unified\_Dataset\_and\_New\_Metric.pdf 3

## Supplementary Material

We provide additional dataset and protocol details, numerical comparisons, and examples of model and human errors. Unless stated otherwise, accuracy intervals use 100,000 BCa resamples (seed 32025). Model intervals resample pairs; pooled human-stratum intervals resample individual responses. Latency bars show interquartile ranges (IQRs).

## S1 Dataset composition and dificulty definitions

The 500 handovers come from 20 South African recordings. Gallery size $K _ { i }$ counts every candidate shown to evaluators, including non-vehicle road users. The dificulty analyses instead use $V _ { i } ,$ , the number of vehicles among those candidates after class and ego-vehicle filtering. The counts difer on 164 pairs. For $K _ { i } ,$ the 3, 4–5, and $\geq 6$ groups contain 38, 127, and 335 pairs; for $V _ { i } ,$ they contain 45, 163, and 292. Candidate aliases and ground-truth matches are unchanged, and detector classes are hidden from evaluators. Count-stratified plots use $V _ { i }$ and cannot, by themselves, establish how accuracy varies with gallery size $K _ { i }$

Table S1: Vehicle-only candidate counts and visibility (500 pairs). V<sub>i</sub> difers from actual gallery cardinality.
<table><tr><td>Attribute</td><td>Count</td><td>Share (%)</td></tr><tr><td>Vehicle-only candidate count</td><td></td><td></td></tr><tr><td> $V _ { i } = 3$ </td><td>45</td><td>9.0</td></tr><tr><td> $V _ { i } = 4 { - } 5$ </td><td>163</td><td>32.6</td></tr><tr><td> $V _ { i } \geq 6$ </td><td>292</td><td>58.4</td></tr><tr><td>Front-target visibility</td><td></td><td></td></tr><tr><td>No occlusion</td><td>377</td><td>75.4</td></tr><tr><td>Partial occlusion</td><td>95</td><td>19.0</td></tr><tr><td>Heavy occlusion</td><td>28</td><td>5.6</td></tr><tr><td>Boundary truncation</td><td>66</td><td>13.2</td></tr><tr><td>Rear-target visibility</td><td></td><td></td></tr><tr><td>No occlusion</td><td>385</td><td>77.0</td></tr><tr><td>Partial occlusion</td><td>94</td><td>18.8</td></tr><tr><td>Heavy occlusion</td><td>21</td><td>4.2</td></tr><tr><td>Boundary truncation</td><td>48</td><td>9.6</td></tr></table>

Scale and range bins. Normalized target area is divided into tertiles at 0.93% and 2.26% in the front view (bin counts 166/167/167) and 0.66% and 1.57% in the rear view (166/166/168). Occlusion is labeled none, partial, or heavy; truncation denotes contact with an image boundary. The 341 pairs with valid approximate front-stereo depth form five near-equal bins (68/68/68/68/69), separated at 6.91, 9.91, 14.54, and 22.28 m. Depth plots exclude pairs without valid stereo estimates; they do not substitute inverse image area for depth. These annotations support trial balancing and dificulty analysis but are not supplied to matchers.

Table S2: Scene-context composition across the 500 handovers.
<table><tr><td>Factor</td><td>Category</td><td>Count</td><td>Share (%)</td></tr><tr><td>Time of day</td><td>Day</td><td>488</td><td>97.6</td></tr><tr><td></td><td>Dawn or dusk</td><td>10</td><td>2.0</td></tr><tr><td></td><td>Night</td><td>2</td><td>0.4</td></tr><tr><td>Weather</td><td>Overcast</td><td>237</td><td>47.4</td></tr><tr><td></td><td>Clear</td><td>180</td><td>36.0</td></tr><tr><td></td><td>Sun glare</td><td>83</td><td>16.6</td></tr><tr><td>Road context</td><td>Highway</td><td>191</td><td>38.2</td></tr><tr><td></td><td>Urban</td><td>171</td><td>34.2</td></tr><tr><td></td><td>Suburban</td><td>136</td><td>27.2</td></tr><tr><td></td><td>Construction</td><td>2</td><td>0.4</td></tr></table>

## S2 Additional protocol details and dificult cases

Each front-evidence condition uses the same rear gallery. Candidate aliases remain visible, while coordinates, detector classes, and detector confidence scores are withheld. The prompt instructs models not to use candidate order, position, box size, or road plausibility as identity evidence, although visual ordering remains apparent. A crop removes scene context and changes the target’s efective scale. Figure S1 illustrates four challenging cases under this protocol.

## S3 Evidence-condition comparisons

From the reasoning-disabled results in Table 3 of the main paper, crops improve Rank-1 accuracy over full RGB by 6.6 points for Llama 4 Scout, 11.0 for Qwen 3.6 27B, 16.4 for GPT-5.5, 13.8 for GPT-5.4 mini, and 13.4 for Gemini 2.5 Flash. Their respective losses from crop to silhouette are 27.2, 42.4, 38.0, 29.8, and 38.8 points. These are observed diferences, without separate significance tests. They reflect changes in input presentation, including target scale and available cues. LLaVA-OneVision returned C1 throughout, so its unchanged accuracy does not show insensitivity to the missing evidence.

## S4 Paired reasoning comparisons

Figure 9 in the main paper compares medium-efort reasoning with reasoning disabled on the same pairs. Its 95% BCa intervals use 100,000 paired resamples. GPT-5.4 mini gained 14.2 points on full RGB [9.4, 19.0], 12.6 on crops [8.4, 16.8], and 11.4 on silhouettes [6.8, 16.0]. GPT-5.5 gained 4.4 points on full RGB [0.4, 8.4] and 7.0 on silhouettes [3.2, 11.0]; Gemini 2.5 Flash gained 6.8 on silhouettes [2.2, 11.4].

![](images/928bd4808179a42496e24fb682d5ef4db83300136ad0a08207c8b9d8ee7d2238.jpg)  
Fig. S1: Dificult handovers from Front2Back-ReID. Columns show the marked full-RGB query, target crop, and fixed rear gallery. Human participants saw one of the two front inputs and did not see the answer annotations. Green marks the match and red the distractors for the reader; detector classes are omitted and people are blurred. Rows show (a) many vehicle candidates (V =14), (b) a distant front target (≈ 113 m approximate stereo depth), (c) front-target occlusion (≈ 61%), and (d) boundary truncation.

The remaining paired intervals cross zero: GPT-5.5 crops, +1.8 points [−1.2, 4.8]; Gemini 2.5 Flash full RGB, +3.8 [−0.4, 8.0]; and Gemini 2.5 Flash crops, +3.2 [−0.8, 7.2]. Thus, reasoning helped GPT-5.4 mini across all three inputs, but the evidence for gains in the other matched cells is mixed. In particular, the best crop result (GPT-5.5) has no clear paired reasoning gain.

## S5 Human accuracy, latency, and errors

Each of the 25 participants answered 20 full-RGB and 20 crop trials, without seeing the same pair twice. Each pair received one judgment per condition from diferent participants. The main-paper timing plot uses all 1,000 judgments: 470 correct and 30 incorrect on full RGB, and 461 correct and 39 incorrect on crops.

These give accuracies of 94.0% and 92.2%, respectively. Pooled human-stratum intervals resample individual responses, without clustering by pair or participant; they describe uncertainty under that sampling choice.

## S5.1 Decision latency

Across all responses, median decision time was 8.09 s (IQR 5.75–13.08) for full RGB and 7.27 s (5.29–12.60) for crops. These times include interface use and individual diferences; they do not measure reasoning alone.

The supplementary latency plots exclude 10 responses longer than 60 s, leaving 492 full-RGB and 498 crop responses. This difers from the main-paper timing plot, which includes all 1,000 responses. After filtering, median times for correct and incorrect responses are 7.74 and 17.57 s on full RGB (n = 465 and 27), and 6.98 and 19.21 s on crops (n = 460 and 38). Without filtering, the corresponding medians are 7.78 and 20.01 s on full RGB, and 7.00 and 20.53 s on crops. The 60 s cutof removes five correct and three incorrect full-RGB responses, and one correct and one incorrect crop response. Figures S3 and S4 split the filtered sample by dificulty and scene context. Their candidate-count labels refer to $V _ { i } ,$ not gallery size $K _ { i }$ . These plots are descriptive; they do not adjust for participant or question order.

![](images/fdaad29daf1eb9e3501dfae3b996ef84f6dde1fd31ba5be220b1be6153d5d109.jpg)  
Fig. S2: Human decision-time distribution by input condition, excluding 10 responses above 60 s. Violin width shows density; points are responses, horizontal marks are medians, and thick vertical segments are IQRs. Time is shown on a log scale.

## S5.2 Pairs missed in both human conditions

The main-paper plate shows five of the ten pairs missed in both human conditions. Each pair received one full-RGB and one crop judgment, so these examples show two observed errors per pair. They do not establish majority failure or intrinsic ambiguity.

![](images/e72f8d1ef0bd6815a2392260c5859987e4fd04197a63fcd6bb208f6203389e42.jpg)  
Fig. S3: Human decision time by dificulty stratum after the 60 s cutof, shown separately for full RGB and crops. Points are medians and bars are IQRs; counts pool both input conditions.

## S6 Qualitative model reasoning examples

The response schema requested one short sentence citing visual evidence. Table S3 shows four saved front-crop responses: two correct matches and two errors based on plausible but misleading cues. These sentences are model outputs, not verified explanations of the models’ decisions.

Figure S5 and Table S4 show one pair missed by all ten complete front-crop VLM runs. Every run chose C1, a prominent large truck, while the labeled match was the smaller, more distant C2. LLaVA-OneVision also chose C1, but its run selected C1 on every trial and echoed a prompt template rather than giving visual evidence.

![](images/3a296fd8260a28191f6bb7fd8594cf367123e770c67eb31bd0f499537c79b515.jpg)  
Fig. S4: Human decision time by scene context, correctness, and visibility after the 60 s cutof. Incorrect responses take longer; weather, road context, and occlusion show smaller median diferences.

![](images/09a222bb0c4d3a9e346ca1718dbfa7c6e40666b9d46d16d243038bf93ccdd01c.jpg)  
Fig. S5: Front-crop pair missed by all 10 complete front-crop VLM cells. The correct rear candidate is C2 (green); C1 is the common selected distractor (red). Other candidates are drawn with thin outlines to preserve the small ground-truth box.

Table S3: Saved model evidence sentences from front-crop trials. “GT” is the correct rear-candidate alias; R denotes medium-efort reasoning.
<table><tr><td>Model cell</td><td>Outcome</td><td>Pick GT</td><td>Evidence sentence</td></tr><tr><td>GPT-5.5 (R)</td><td>Correct</td><td>C1 C1</td><td>C1 matches the target&#x27;s white low sedan silhouette with a dark side-window band.</td></tr><tr><td>Gemini 2.5 Pro Correct (R)</td><td></td><td>C2  / C2</td><td>The target and candidate C2 are both large white tanker trucks.</td></tr><tr><td>GPT-5.5 (R)</td><td>Wrong</td><td>C3 /C1</td><td>C3 matches the target&#x27;s dark gray SUV body, chrome-accented grille, and headlight shape.</td></tr><tr><td>Gemini 2.5 Flash Wrong (R)</td><td></td><td>C1  / C2</td><td>The target vehicle and C1 are both white vans with identical yellow and gold stripe livery on their sides.</td></tr></table>

Table S4: Saved predictions for Fig. S5. All runs chose C1; the correct answer is C2. R denotes medium-efort reasoning; confidence is the model’s reported value.
<table><tr><td>Model cell</td><td></td><td></td><td>Pick Conf. Evidence sentence</td></tr><tr><td>GPT-5.5 (R)</td><td>C1</td><td>0.86</td><td>C1 is the only candidate showing a large white-cab tractor with a long reddish-brown box trailer matching the target.</td></tr><tr><td>GPT-5.5 (no R)</td><td>C1</td><td>0.72</td><td>C1 shows the same large white tractor with a long reddish-brown box trailer silhouette as the front target.</td></tr><tr><td>GPT-5.4 mini (R)</td><td>C1</td><td>0.44</td><td>C1 is the only candidate with a long semi-trailer silhouette and white cab/trailer combination matching the target truck.</td></tr><tr><td>GPT-5.4 mini (no R) C1</td><td></td><td>0.72</td><td>C1 best matches the front target&#x27;s white boxy vehicle shape with a light cab and long rectangular cargo body.</td></tr><tr><td>Gemini 2.5 Pro (R)</td><td>C1</td><td>0.92</td><td>The target is a white tractor pulling a white trailer, and C1 is the only candidate that is a large white truck.</td></tr><tr><td>Gemini 2.5 Flash (R) C1</td><td></td><td>0.92</td><td>The target vehicle is a white semi-truck with a red shipping container, which precisely matches candidate C1.</td></tr><tr><td>R)</td><td></td><td></td><td>Gemini 2.5 Flash (no C1 1.00 The target vehicle is a white semi-truck with a red container, which perfectly matches candidate C1.</td></tr><tr><td></td><td></td><td></td><td>Qwen 3.6 27B (no R) C1 0.90 The target vehicle is a white semi-truck cab, which matches the white truck cab visible in candidate C1, whereas all other</td></tr><tr><td>R)</td><td></td><td></td><td>candidates are small dark vehicles. Llama 4 Scout (no C1 1.00 The truck&#x27;s white cab and brown cargo container match.</td></tr><tr><td>LLaVA-OneVision 0.5B (no R)</td><td></td><td></td><td>C1 0.11 one short sentence citing visible identity evidence.</td></tr></table>

## S7 Depth sensitivity

Approximate front-stereo depth is available for 341 pairs. Across five equalfrequency depth bins, crop accuracy is non-monotonic for all three plotted methods (Fig. S6, left). Normalized front-target area shows a clearer association with accuracy (right): SigLIP2 rises from 50.0% in the smallest-area quintile to 91.0% in the largest, while GPT-5.5 changes less. These binned comparisons support the main paper’s scale analysis; they do not isolate scale from other properties of the examples.

![](images/c5db3f5055f14941a3ef325f8039a4afde5a6a71bde3f81e25b65460ac2b31a8.jpg)  
Fig. S6: Reasoning-disabled crop accuracy in five bins of valid approximate front-stereo depth (n = 341) and normalized front-target area (n = 500). Points are placed at each bin’s median depth or area. Depth shows no monotonic trend; SigLIP2 improves as target area increases.

## S8 Recording-sequence durations

Twenty-one sequences were recorded, of which 20 contribute accepted pairs to the benchmark. The 21 recordings total approximately 1.9 hours; durations range from 46 seconds to 17.2 minutes, with a median of 4.7 minutes (Fig. S7).

![](images/df6661bceabe5de4c282d19c2f6528f363b4c5ca6288a83fc403db1c1bd1f72b.jpg)  
Fig. S7: Durations of all 21 recorded sequences, including the one with no accepted pair (46 seconds to 17.2 minutes; median 4.7 minutes; approximately 1.9 hours in total).

## S9 Excluded incomplete runs

The main-paper interval analyses include only complete N = 500 runs. The reasoning-enabled Qwen 3.6 27B runs reached 470/500 full-RGB, 480/500 crop, and 416/500 silhouette trials before repeated Groq request failures and truncated completions prevented completion. We exclude these incomplete runs. Provider failures and empty or truncated outputs were not scored; non-empty outputs that could not be parsed or selected an unlisted candidate counted as incorrect. This records the limits of these runs under the evaluated settings, not a reliability ranking of model families or providers.