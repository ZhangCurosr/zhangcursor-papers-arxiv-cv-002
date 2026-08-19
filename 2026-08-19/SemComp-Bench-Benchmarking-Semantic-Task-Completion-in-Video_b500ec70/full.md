# SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation

Keyu Tu<sup>1</sup> Zhuowei Chen<sup>1</sup> Mengqi Huang<sup>1,2,†</sup> Yuxin Wang<sup>2</sup> Jiahao Zhu<sup>3</sup> Zhendong Mao<sup>1</sup> Yongdong Zhang<sup>1</sup>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>FrameX.AI <sup>3</sup>Sun Yat-sen University <sup>†</sup>Corresponding author

## Abstract

We introduce Semantic Task Completion Video Generation, an outcome-oriented video generation task. Under this formulation, success requires both achievement of the intended outcome and semantic grounding. Semantic grounding characterizes the correspondence between the reference image and the generated outcome in terms of high-level semantics relevant to the task. Evaluation focuses on the generated outcome and requires neither the presentation of a complete sequence of intermediate task steps nor conventional appearance consistency with the reference image. To support systematic evaluation, we construct SemComp-Data, an evaluation dataset covering six domains. Each instance comprises a reference image, a detailed instruction, a brief instruction, and an outcome-centric video clip. A scalable four-stage curation pipeline converts raw videos into standardized SemComp-Data instances. We further introduce SemComp-Bench, an evaluation protocol that uses a vision-language model (VLM) to answer structured binary questions. SemComp-Bench reports the OA Score and the GR Score for Outcome Achievement and Generation Reliability, respectively. Experiments on representative video generation models show that achieving intended outcomes while maintaining task-relevant semantic grounding in reference images remains challenging.

Date: August 19, 2026

Project Page: https://SemComp-Bench.github.io

Correspondence: Mengqi Huang at huangmq@ustc.edu.cn

## 1 Introduction

Recent video generation models [4, 27, 28] achieve impressive visual fidelity and temporal coherence. However, their ability to achieve an instructed outcome while maintaining high-level semantic grounding remains underexplored. We formulate this problem as Semantic Task Completion Video Generation. Here, grounding preserves task-relevant semantic relationships and reference attributes while allowing unrelated attributes to change. For example, given a banknote image and the instruction “Fold the banknote into a turtle,” a successful video must visibly present the referenced banknote as turtle-shaped origami, as illustrated in Fig. 1. Intermediate folding steps are unnecessary, but the result must remain grounded in the input banknote rather than replacing it with an unrelated turtle.

Existing datasets and benchmarks emphasize visual fidelity, temporal coherence, and subject consistency, but rarely assess outcome achievement jointly with high-level semantic grounding. We therefore construct SemComp-Data, an evaluation dataset of image-text-video triplets curated from full-context real-world videos.

![](images/17f72533c4d1f334151f83c9b2c6073947968829e1111f0e4077239609171f4b.jpg)  
Figure 1 Demonstration of Semantic Task Completion Video Generation, which focuses not on the transformation process but solely on outcome achievement and semantic grounding.

![](images/882bb5b25c21ee81ca95c58c6aef8e177e27450c57365accda52d71379ab0492.jpg)  
Figure 2 Representative SemComp-Data instances. Each example shows a reference frame, its outcome-centric video clip, and the brief instruction for compact presentation; every instance also includes a detailed instruction.

Such full-context curation naturally ensures task feasibility while enabling fine-grained visual alignment. Each instance pairs the same reference and outcome with brief and detailed instructions describing the intended result at diferent levels of specificity. The curation pipeline comprises Candidate Filtering, State Mining, Video Extension, and Instruction Structuring. Candidate Filtering screens raw videos, State Mining localizes and verifies reference–outcome frames, and Video Extension extracts an outcome-centric clip around the localized outcome. Instruction Structuring then identifies alignment constraints and produces instructions. Figure 2 presents representative SemComp-Data instances.

Building on SemComp-Data, SemComp-Bench uses evidence-grounded binary VLM judgments to assess two complementary dimensions: Outcome Achievement and Generation Reliability, reported as the OA Score and GR Score. The evaluator answers predefined questions and supports each answer with visual evidence, yielding focused and interpretable judgments. OA measures outcome realization and semantic grounding, with safeguards for task-relevant entity consistency and global visual continuity. GR complements it by assessing physical violations, blur, rendering artifacts, local instability, and corrupted text or interface elements. Together, the scores support systematic evaluation and criterion-level failure diagnosis. The main contributions of this work are summarized as follows:

• We formulate Semantic Task Completion Video Generation, a new task that requires generated videos to achieve instructed outcomes while preserving task-relevant semantic relationships with reference images.

• We construct SemComp-Data through a scalable, low-cost pipeline that converts raw videos into imagetext-video triplets with paired brief and detailed instructions.

• We introduce SemComp-Bench, a VLM-based evaluation protocol that measures Outcome Achievement and Generation Reliability using interpretable, evidence-grounded binary judgments.

## 2 Related Work

## 2.1 Video Generation Benchmarks

Video generation benchmarks evaluate visual fidelity, temporal coherence, and prompt adherence [10, 13, 14, 22, 34]. Personalized and controllable benchmarks further assess subject identity and intended motion [19, 32], while recent work examines physical plausibility, causal consistency, and commonsense constraints [2, 3, 18, 21, 31, 33]. Together, these benchmarks cover generation quality, identity, motion, and physical plausibility. However, they do not directly test whether a video achieves an outcome jointly defined by an instruction and reference context while preserving task-relevant semantic grounding, motivating SemComp-Bench.

## 2.2 Video Generation Models

Current models increasingly support high-quality, instruction-following generation. Open-source systems such as Wan2.2 [28], CogVideoX [30], HunyuanVideo [16], and Pyramid Flow [15] support text- and imageconditioned generation, while LTX [11, 12] emphasizes eficiency. Closed-source systems, including Sora, Veo, MovieGen, Runway, Kling, Seedance, Hailuo, and Pika, further support multimodal conditioning, multi-shot narratives, camera control, and editing [1, 4, 9, 17, 23–26]. Together, these models show strong synthesis and control, but their outcome achievement under task-relevant reference grounding remains insuficiently understood.

## 3 SemComp-Data Construction

Starting from full-context videos in Koala-36M [29], we construct SemComp-Data as a collection of structured image-text-video triplets $x _ { i } = \left( r _ { i } , \mathcal { T } _ { i } , o _ { i } \right)$ , where $\bar { \mathcal { T } } _ { i } = ( i _ { i } ^ { \mathrm { b r i e f } } , i _ { i } ^ { \mathrm { d e t a i l e d } } )$ is an aligned instruction pair describing the same intended outcome at two levels of specificity. The reference frame $r _ { i }$ defines the initial context, the paired instructions specify the desired outcome, and the outcome-centric video clip $o _ { i }$ depicts successful completion. Both $r _ { i }$ and $o _ { i }$ are drawn from the same task instance, ensuring that the visual elements are genuinely associated rather than independently collected. This full-context construction provides evidence that the intended outcome is realizable and visually verifiable for the given reference. As illustrated in Fig. 3, we develop a four-stage curation pipeline to construct standardized evaluation instances from full-context videos. The pipeline supports scalable data construction without requiring the manual design of individual tasks or the production of corresponding outcome-centric videos. The following subsections describe each stage in detail.

## 3.1 Stage 1: Candidate Filtering

We use the Koala-36M [29] dataset as the source pool, owing to its broad thematic coverage and strong representation of real-world scenarios. Because SemComp-Data relies on visually observable outcomes, we first apply Title-Based Keyword Filtering to exclude narration-dependent videos whose titles contain lexical patterns such as talk show, interview, and news, thereby retaining candidates that are more likely to be visually self-contained, as shown in Fig. 3(a). We uniformly sample frames from each retained video and arrange them into a mosaic representation, termed the video abstract, which provides a compact visual summary for eficient categorization. Using these abstracts, an advanced VLM assigns each video to one of six domains and an associated task category; videos lacking suficient visual evidence for reliable categorization are labeled as Uncertain and discarded. The retained domains are Arts and Precision, Beauty and Fashion, Crafts and DIY, Food and Cooking, Gardening and Pets, and Sports and Fitness. For each task category, we manually define the reference and outcome states according to its characteristic completion pattern. These definitions specify what constitutes the initial context and the completed outcome for downstream state mining. The complete state definitions and the keyword list used for title-based filtering are provided in the supplementary material.

## 3.2 Stage 2: State Mining

As shown in Fig. 3(b), this stage uses the manually specified category-specific state definitions from Stage 1 to mine reliable reference–outcome frame pairs through two steps: State Grounding and Quality Checking. We formulate State Grounding as frame-level timestamp localization rather than direct outcome-segment localization, allowing the VLM to focus on representative visual evidence of the reference and outcome states without modeling the intermediate process or resolving ambiguous segment boundaries. Given a full-context video and its state definitions, the VLM first identifies the demonstrated task and expected outcome and then locates the timestamps that best match the two state definitions. The reference and outcome frames at these timestamps are extracted for subsequent processing. The extracted frame pair is then verified through QA-based Quality Checking, which serves as a conservative consistency screen. The VLM is provided with the two unlabeled frames and the task description obtained during State Grounding and is asked to identify which frame represents the outcome. A valid pair should allow the outcome frame to be identified unambiguously. If the predicted outcome does not match the grounded outcome frame, the pair is conservatively discarded, as the disagreement may reflect an ambiguous reference–outcome relationship, insuficient visual evidence, or VLM error. The remaining reference and outcome frames, together with their timestamps, constitute the outputs of this stage.

![](images/4b8322ca8794fd326fa402621e348491a62d6328745492e53b758ebb40756f18.jpg)  
Figure 3 Overview of the four-stage SemComp-Data curation pipeline. Panels (a)–(d) show Candidate Filtering, State Mining, Video Extension, and Instruction Structuring, respectively. The right column shows the corresponding stage outputs, and the bottom row presents the final data triplet.

## 3.3 Stage 3: Video Extension

As shown in Fig. 3(c), this stage takes the full-context video and the verified outcome timestamp from Stage 2 as inputs. Following the shot detection and same-scene merging procedure of Panda-70M [8], we partition the full-context video into segments with consistent camera viewpoints. Using the verified timestamp as a temporal anchor, we extract an outcome-centric clip (o ) by selecting the core segment containing the outcome frame and merging visually consistent neighboring segments until a minimum duration of 3 s is reached. Compared with directly prompting a VLM to localize the outcome interval, this timestamp-anchored strategy more efectively centers the extracted clip on the completed outcome while ofering greater eficiency and lower cost. The clips have an average duration of approximately 4.03 s in SemComp-Data, providing suficient temporal evidence of the completed outcome while minimizing intermediate processes and unrelated content. The final outcome-centric clip constitutes the output of this stage.

## 3.4 Stage 4: Instruction Structuring

As shown in Fig. 3(d), this stage takes the reference frame and outcome-centric video clip as inputs and produces two aligned instruction variants—a brief instruction i<sup>brief</sup><sub>i</sub> and a detailed instruction i<sup>detailed</sup><sub>i</sub> —through Normalized Relation Description, Attribute Selection, and Instruction Composition. In preliminary experiments, directly prompting the VLM with the two inputs often produces overly verbose instructions containing incidental details, such as brand names and on-screen text. To capture the essential reference–outcome relation, Normalized Relation Description constrains the VLM to generate a brief instruction i<sup>brief</sup> of no more than 30 words following the template [Verb] + [Main Subject in r ] + [Preposition] + [Main State in o ].

In parallel, Attribute Selection determines which reference attributes should be preserved and which can be discarded during outcome-centric video generation. Because the task-relevant attributes vary across categories, the VLM first selects an applicable alignment type from four options: Object Element, Person Identity, Object Appearance, and Scene. For example, person identity and task-relevant appearance cues should be preserved when the reference and outcome depict the same person with diferent makeup or hairstyles, whereas the spatial layout and design structure should be retained when transforming a blueprint into its corresponding physical result. Based on the selected alignment type, the VLM identifies the attributes to preserve and discard. The definitions and available options for the alignment types are provided in the supplementary material.

Instruction Composition expands the brief instruction i<sup>brief</sup><sub>i</sub> into the detailed instruction i<sup>detailed</sup><sub>i</sub> by combining (i) the selected alignment type and preserve/discard attributes, which specify the reference-grounding constraints, and (ii) fine-grained visual characteristics that describe the completed outcome, including its background, lighting, colors, shapes, and component-level details. These outcome characteristics do not constitute reference appearance constraints unless they are explicitly selected for preservation. The resulting detailed instruction jointly specifies the desired outcome and its reference-grounding requirements, making it suitable for evaluation and potentially useful for future task-specific training. The two variants describe the same task instance at diferent levels of specificity, enabling controlled evaluation of instruction specificity without changing the underlying reference or target outcome. The brief instruction better reflects typical user prompting habits and therefore provides a more challenging evaluation setting.

Figure 4 summarizes the alignment-type and domain distributions of SemComp-Data, together with highfrequency words in the brief instructions.

![](images/34dd1ca36b91cb5d54d3043e4387fda12ec48a62495be43df91db634e2beabc9.jpg)  
(a) Domain-wise Distribution of Alignment Types

![](images/13087308a9a79db6b10aaea84735567b924e16a448b09a290403033d739078d1.jpg)  
(b) High-Frequency Words in Brief Instructions

![](images/ed19d2c7e6effbd3fde8bb3e14142e579bd9d38f4de3bb89c7233de5ef9ad9c6.jpg)  
(c) Domain Distribution

Figure 4 Statistics of SemComp-Data: (a) distribution of alignment types across domains, with distinct textures denoting diferent types; (b) high-frequency words in the instructions; and (c) distribution of evaluation instances across the six domains.
<table><tr><td>Models</td><td> $A _ { \mathrm { o r } }$ </td><td> $A _ { \mathrm { s g } }$ </td><td> $A _ { \mathrm { g e c } }$ </td><td> $A _ { \mathrm { g v c } }$ </td><td>OA Score</td></tr><tr><td>Seedance 2.0 [5]</td><td>0.839</td><td>0.744</td><td>0.444</td><td>0.594</td><td>20.0%</td></tr><tr><td>Wan2.2-TI2V-5B [28]</td><td>0.589</td><td>0.400</td><td>0.689</td><td>0.922</td><td>23.3%</td></tr><tr><td>Wan2.2-I2V-A14B [28]</td><td>0.800</td><td>0.528</td><td>0.628</td><td>0.789</td><td>28.3%</td></tr><tr><td>CogVideoX1.5-5B-I2V [30]</td><td>0.550</td><td>0.389</td><td>0.506</td><td>0.744</td><td>14.4%</td></tr><tr><td>SkyReels-V2-I2V-14B-720P [7]</td><td>0.733</td><td>0.489</td><td>0.522</td><td>0.772</td><td>22.8%</td></tr><tr><td>HY†-1.5-720P-I2V</td><td>0.878</td><td>0.706</td><td>0.583</td><td>0.794</td><td>37.8%</td></tr><tr><td>Phantom-1.3B [20]</td><td>0.539</td><td>0.356</td><td>0.322</td><td>0.511</td><td>3.9%</td></tr></table>

Table 1 SemComp-Core Outcome Achievement with detailed instructions. $A _ { \mathrm { o r } } , A _ { \mathrm { s g } } , A _ { \mathrm { g e c } } ,$ , and $A _ { \mathrm { g v c } }$ denote pass rates for outcome realization, semantic grounding, grounded entity consistency, and global visual continuity, respectively. The OA Score is the joint pass rate across four criteria. <sup>†</sup>HY denotes HunyuanVideo [27]. Bold and underlined values indicate the best and second-best values, respectively.

## 4 SemComp-Bench Evaluation

Building on SemComp-Data, SemComp-Bench evaluates generated videos through structured binary questions, each targeting a specific criterion for interpretable assessment and failure diagnosis. The criteria are organized into two complementary dimensions: Outcome Achievement (OA), which measures outcome realization, reference grounding, and task-relevant continuity; and Generation Reliability (GR), which assesses visual, physical, and temporal reliability. Their aggregate results are reported as the OA Score and GR Score, respectively. For a single evaluation run on a set of N samples, let $a _ { c } ^ { i } , g _ { c } ^ { i } \in \{ 0 , 1 \}$ denote the binary criterion scores of sample i for criterion c in the Outcome Achievement and Generation Reliability dimensions, respectively, where 1 indicates a pass and 0 otherwise. The full set of binary questions is provided in the supplementary material.

## 4.1 Outcome Achievement

For each sample $i ,$ outcome achievement is evaluated using four criteria: outcome realization, semantic grounding, grounded entity consistency, and global visual continuity. Their binary criterion scores are $a _ { \mathrm { o r } } ^ { i } .$ $a _ { \mathrm { s g } } ^ { i } , \ : a _ { \mathrm { g e c } } ^ { i } ,$ and ${ a } _ { \mathrm { g v c } } ^ { i } ,$ respectively. All four criteria are evaluated on the same temporally ordered sequence of 27 frames uniformly sampled from each generated video. Outcome realization and semantic grounding additionally use the reference image and instruction, whereas grounded entity consistency and global visual continuity use only the sampled sequence. The corresponding dataset-level pass rates are $A _ { \mathrm { o r } } , A _ { \mathrm { s g } } , A _ { \mathrm { g e c } }$ , and $A _ { \mathrm { g v c } } ,$ where $\begin{array} { r } { A _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } a _ { c } ^ { i } } \end{array}$

The outcome realization criterion assesses whether the generated video visibly reaches the completed state specified by the instruction at a coarse semantic level, independent of fine-grained correspondence to the reference image. The semantic grounding criterion assesses whether the realized outcome preserves or modifies task-relevant entities and attributes according to the reference-instruction pair. The grounded entity consistency criterion assesses whether task-relevant key entities remain semantically identifiable throughout the sampled sequence, without unexplained disappearance, replacement, or unintended drift in identity, material, appearance, or task-relevant structure. The global visual continuity criterion examines the temporally ordered sampled sequence for abrupt global switches in scene, viewpoint, layout, background, or composition. A common, but not exclusive, failure case occurs when a reference-matching initial frame is followed by a visually disconnected continuation. These two criteria serve as validity safeguards for the generated outcome and do not assess the completeness or procedural correctness of intermediate task steps.

<table><tr><td>Models</td><td> $G _ { \mathrm { p p } }$ </td><td> $G _ { \mathrm { v c } }$ </td><td> $G _ { \mathrm { a f r } }$ </td><td> $G _ { \mathrm { w s c } }$ </td><td> $G _ { \mathrm { t i } }$ </td><td>GR Score</td></tr><tr><td>Seedance 2.0 [5]</td><td>0.883</td><td>0.994</td><td>0.994</td><td>0.739</td><td>0.978</td><td>91.8%</td></tr><tr><td>Wan2.2-TI2V-5B [28]</td><td>0.794</td><td>0.978</td><td>0.889</td><td>0.722</td><td>0.889</td><td>85.4%</td></tr><tr><td>Wan2.2-I2V-A14B [28]</td><td>0.911</td><td>0.972</td><td>0.967</td><td>0.672</td><td>0.928</td><td>89.0%</td></tr><tr><td>CogVideoX1.5-5B-I2V [30]</td><td>0.944</td><td>0.811</td><td>0.772</td><td>0.583</td><td>0.828</td><td>78.8%</td></tr><tr><td> $\mathrm { S k y R e e l s – V 2 – I 2 V - 1 4 B – 7 2 0 P \ [ 7 ] }$ </td><td>0.778</td><td>0.922</td><td>0.739</td><td>0.328</td><td>0.789</td><td>71.1%</td></tr><tr><td>HY†-1.5-720P-I2V</td><td>0.800</td><td>0.939</td><td>0.883</td><td>0.472</td><td>0.861</td><td>79.1%</td></tr><tr><td>Phantom-1.3B [20]</td><td>0.778</td><td>0.972</td><td>0.856</td><td>0.506</td><td>0.728</td><td>76.8%</td></tr></table>

Table 2 SemComp-Core Generation Reliability with detailed instructions. Each $G _ { c }$ denotes the pass rate for a reliability criterion, and the GR Score is the mean across five criteria. <sup>†</sup>HY denotes HunyuanVideo [27]. Bold and underlined values indicate the best and second-best values.
<table><tr><td>Model</td><td>Modality</td><td>Instruction Type</td><td> $A _ { \mathrm { o r } }$ </td><td> $A _ { \mathrm { s g } }$ </td><td> $A _ { \mathrm { g e c } }$ </td><td> $A _ { \mathrm { g v c } }$ </td><td>OA Score</td></tr><tr><td rowspan="3">Wan2.2-A14B [28]</td><td>I2V</td><td>Detailed</td><td>0.800</td><td>0.528</td><td>0.628</td><td>0.789</td><td>28.3%</td></tr><tr><td>T2V</td><td>Detailed</td><td>0.889</td><td>0.522</td><td>0.317</td><td>0.117</td><td>4.4%</td></tr><tr><td>T2V</td><td>Brief</td><td>0.389</td><td>0.111</td><td>0.806</td><td>0.250</td><td>0.6%</td></tr><tr><td rowspan="3">CogVideoX1.5-5B [30]</td><td>I2V</td><td>Detailed</td><td>0.550</td><td>0.389</td><td>0.506</td><td>0.744</td><td>14.4%</td></tr><tr><td>T2V</td><td>Detailed</td><td>0.567</td><td>0.272</td><td>0.361</td><td>0.161</td><td>5.0%</td></tr><tr><td>T2V</td><td>Brief</td><td>0.567</td><td>0.150</td><td>0.678</td><td>0.161</td><td>0.6%</td></tr><tr><td rowspan="3">HY†-1.5-720P</td><td>I2V</td><td>Detailed</td><td>0.878</td><td>0.706</td><td>0.583</td><td>0.794</td><td>37.8%</td></tr><tr><td>T2V</td><td>Detailed</td><td>0.833</td><td>0.606</td><td>0.389</td><td>0.133</td><td>4.4%</td></tr><tr><td>T2V</td><td>Brief</td><td>0.550</td><td>0.100</td><td>0.761</td><td>0.178</td><td>1.7%</td></tr></table>

Table 3 Outcome Achievement on SemComp-Core across modalities and instruction settings. Within each model family, the I2V and T2V variants use checkpoints of the same parameter scale. Each $A _ { c }$ denotes a criterion-level pass rate, and the OA Score is the joint pass rate. <sup>†</sup>HY denotes HunyuanVideo [27].

All four indicators are pass-coded, with 1 denoting that the corresponding criterion is satisfied. For the two failure-oriented questions on entity inconsistency and global visual discontinuity, a “No” response is mapped to 1. Overall success is conjunctive: the sample-level Outcome Achievement indicator $A ^ { i }$ equals 1 only when sample i passes all four criteria. The dataset-level OA Score is the mean of $A ^ { i }$ across all samples:

$$
\begin{array} { l } { { \displaystyle { \cal A } ^ { i } = a _ { \mathrm { o r } } ^ { i } a _ { \mathrm { s g } } ^ { i } a _ { \mathrm { g e c } } ^ { i } a _ { \mathrm { g v c } } ^ { i } \in \{ 0 , 1 \} , } } \\ { { \displaystyle ~ \mathrm { O A } ~ \mathrm { S c o r e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } A ^ { i } . } } \end{array}\tag{1}
$$

## 4.2 Generation Reliability

For each sample i, the VLM evaluates generation reliability independently of outcome achievement and reference grounding using five binary criteria: physical plausibility, visual clarity, artifact-free rendering, within-scene spatiotemporal coherence, and text and interface integrity. Their binary criterion scores are $g _ { \mathrm { p p } } ^ { i } .$ $g _ { \mathrm { v c } } ^ { i } , g _ { \mathrm { a f r } } ^ { i } , g _ { \mathrm { w s c } } ^ { i }$ , and $g _ { \mathrm { t i } } ^ { i } ,$ , respectively. The corresponding dataset-level pass rates are $G _ { \mathrm { p p } } , G _ { \mathrm { v c } } , G _ { \mathrm { a f r } } , G _ { \mathrm { w s c } }$ , and $G _ { \mathrm { t i } } .$ where $\begin{array} { r } { G _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } g _ { c } ^ { i } } \end{array}$

The physical plausibility criterion assesses visible motions and interactions for clear violations of basic physical principles; it neither requires the instructed transformation process to be shown nor treats an omitted intermediate process as a failure. The visual clarity criterion identifies content that is dificult to recognize because of blur, improper exposure, insuficient resolution, or unclear scene details. The artifact-free rendering criterion detects scene-inconsistent synthetic artifacts, such as abnormal noise, corrupted regions, blank patches, or color streaks. The within-scene spatiotemporal coherence criterion detects low-level rendering instability within continuous scenes, such as flicker, transient duplication, and local geometric deformation, irrespective of task semantics. Unlike grounded entity consistency, this criterion concerns local rendering stability rather than the semantic persistence of task-relevant entities. Abrupt global switches across the sampled sequence are evaluated separately by global visual continuity in the Outcome Achievement dimension. The text and interface integrity criterion detects corrupted or unrecognizable text, numbers, icons, and interface elements.

All five questions are failure-oriented, and the reported indicators are pass-coded: a “No” response is mapped to 1. The sample-level Generation Reliability score $G ^ { i }$ is the arithmetic mean of the five binary criterion scores, and the dataset-level GR Score is the mean of $G ^ { i }$ across all samples:

$$
\begin{array} { c l l } { { G ^ { i } = \displaystyle \frac { 1 } { 5 } \left( g _ { \mathrm { p p } } ^ { i } + g _ { \mathrm { v c } } ^ { i } + g _ { \mathrm { a f r } } ^ { i } + g _ { \mathrm { w s c } } ^ { i } + g _ { \mathrm { t i } } ^ { i } \right) \in [ 0 , 1 ] , } } \\ { { { } } } & { { { } } } \\ { { \mathrm { G R } \mathrm { S c o r e } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } G ^ { i } . } } \end{array}\tag{2}
$$

Together, the criterion-level pass rates identify specific sources of generation unreliability.

## 5 Experiments

## 5.1 Experimental Setup

We randomly sample approximately 20K videos from Koala-36M [29] and process them using the curation pipeline, yielding 1,273 structured evaluation instances. To construct a domain-balanced evaluation subset, SemComp-Core comprises 60 instances, with 10 selected from each domain via stratified sampling to approximately preserve the within-domain alignment-type distributions. Each SemComp-Core instance includes both instruction variants. In analyses of instruction specificity, the paired variants share the same reference image and outcome target, so only the level of instruction specificity varies. Representative open- and closed-source video generation models [5, 7, 20, 27, 28, 30] are evaluated on this subset. All models in Tables 1 and 2 use I2V with the reference image and detailed instruction; Wan2.2-TI2V-5B runs in image-conditioned mode.

Open-source models use their oficial default inference configurations, whereas closed-source models use their exposed default settings; a fixed seed is applied whenever explicit seed control is available. Under every setting, one 720p video is generated per instance, from which 27 frames are uniformly sampled over the full temporal extent to form a fixed-length, temporally ordered sequence for evaluation. Every video is scored through three independent VLM calls made at diferent times, with all criterion-level pass rates and both aggregate scores computed per call using the definitions above; the reported results are arithmetic means of the three corresponding run-level scores. We instantiate the VLM used in the curation and evaluation experiments as Doubao-Seed-1.8 through the Volcano Engine API [6], as preliminary experiments indicate that it ofers a favorable trade-of between evaluation eficiency and cost.

## 5.2 Outcome Achievement Results

Table 1 reports the OA Score under detailed instructions. HunyuanVideo-1.5-720P-I2V achieves the highest OA Score of 37.8%, followed by Wan2.2-I2V-A14B at 28.3%. Their results are consistent with relatively balanced pass rates across the four OA criteria, rather than dominance in any single criterion. Seedance 2.0 performs strongly in outcome realization and semantic grounding, but its lower pass rates on grounded entity consistency and global visual continuity reduce its joint OA Score. In contrast, Wan2.2-TI2V-5B achieves the highest pass rates on grounded entity consistency and global visual continuity while performing less strongly in outcome realization and semantic grounding. These complementary performance profiles, together with the fact that the best OA Score remains below 40%, indicate that robust outcome achievement requires both task fidelity and outcome-video validity.

## 5.3 Generation Reliability Results

Table 2 reports generation reliability under detailed instructions. Seedance 2.0 achieves the highest GR Score of 91.8%. Among the evaluated open-source models, Wan2.2-I2V-A14B performs best at 89.0%. Both models perform strongly in visual clarity and artifact-free rendering. However, within-scene spatiotemporal coherence remains the primary bottleneck across all models, with pass rates ranging from only 0.328 to 0.739. This gap suggests that current models can often generate visually convincing individual frames but struggle to maintain stable visual evolution within scenes. Comparing the GR and OA rankings further indicates that generation reliability does not necessarily correspond to outcome achievement: Seedance 2.0 leads in GR but performs substantially worse in OA, whereas HunyuanVideo-1.5-720P-I2V leads in OA despite a lower GR ranking. Wan2.2-I2V-A14B ranks second on both metrics, indicating comparatively balanced performance across the two evaluation dimensions.

## 5.4 Conditioning Effects

As shown in Table 3, I2V variants consistently outperform their T2V counterparts under detailed instructions across all three model families. The gains mainly arise from improved semantic grounding, grounded entity consistency, and global visual continuity, while outcome-realization pass rates remain broadly comparable and are higher for T2V in two of the three model families. These findings underscore the indispensable role of reference-image conditioning in semantic task completion, particularly in preserving entity identity, appearance, and structural consistency during complex object interactions and state changes.

Within the T2V setting, detailed instructions consistently achieve higher OA Scores than brief instructions, primarily through better outcome realization and semantic grounding. In contrast, brief instructions often yield higher grounded entity consistency and global visual continuity, likely because their simpler and less constrained content is easier to generate coherently. These results reveal a trade-of between instruction specificity and generation dificulty: detailed instructions better define the intended compositional outcome but place greater demands on semantic understanding and coordinated generation, whereas brief instructions are associated with higher outcome-video consistency but lower task-fulfillment rates.

## 5.5 Visualization of Generated Videos

As shown in Fig. 5, the evaluated models tend to preserve the appearance of the reference image and depict the process of task execution, yet often fail to achieve the intended outcome. Moreover, generated videos often sufer from reliability issues, including physically implausible transformations and abrupt visual transitions, as illustrated by the failure cases of Wan2.2-I2V-A14B and Phantom-1.3B shown in the figure. Additional visualizations of generated videos are provided in the supplementary material.

Phantom-1.3B  
[The Reference Image]  
![](images/bbde0f0d1cc4029fb8ff495482a7793eb64b771f45258d6e58802800312e3f64.jpg)

## [Detailed Instruction]

Generate a video that reaches the following state: a pair of hands uses two ropes—one blue with white and red speckles and the other solid orange—to tie a double sheet bend knot. The hands thread the orange rope through a loop formed by the blue rope and wrap it around the loop to form the knot. They then pull the ends of both ropes to tighten the knot, adjusting the loops and strands to ensure a secure connection. The background is a plain, neutral gray surface that keeps the focus on the ropes and the knot-tying process. The category, number, color palette, and material of the ropes should remain consistent with those in the reference image, while their initially coiled state and spatial arrangement may change as needed to form the knot.

![](images/4f3089817511943dba4511bec2f7cc23de3ecb4d8ef035bf605879209f45155d.jpg)  
The Outcome-Centric Video Clip

![](images/b72602b41f9381c6474968b2ef4d7abe42f03f7f5f1735e6a30f0c003d50bfc6.jpg)  
Seedance 2.0

![](images/8d887b1f0d9c1c7f055de8630a894786c97a622a4db5def841462096cb6ba983.jpg)  
Wan2.2-TI2V-5B

![](images/68bb37c716a4c7997d6f90e990e1957af2a7b353fa26b5c229066358e2fcb637.jpg)  
Wan2.2-I2V-A14B

![](images/834a7b76b3b1050f38514b4ff14a13e3e2caebc2dc46b816a50e9f24b5493159.jpg)  
CogVideoX1.5-5B-I2V

![](images/160f8b22c0487b25a38fd83fd69c7faaa3322654ba637492b8038667603d4c89.jpg)

![](images/c2c802bbcc94cd5edfa68481f525a86d58e8013088cd6be55eb87c703a8b7deb.jpg)  
HunyuanVideo-1.5-720P-I2V

SkyReels-V2-I2V-14B-720P  
![](images/1466699d8384a4b1dec9a247de20eac2f23622b05e267a364eb4e38e14c6f0af.jpg)  
Figure 5 Generated-video examples. The outcome-centric video clip and the reference image are extracted from the same full-context video.

## 6 Conclusion

In this paper, we introduce Semantic Task Completion Video Generation, which requires generated videos to realize an instructed outcome while maintaining semantic grounding in the reference context. We develop SemComp-Data through a scalable four-stage curation pipeline, with each instance comprising a reference image, paired brief and detailed instructions, and an outcome-centric video clip. Crucially, the paired image and clip are extracted from the same full-context video, ensuring task authenticity and achievability as well as fine-grained attribute alignment between the reference and target, while the dataset preserves diverse real-world scenarios from the source collection. Building on this dataset, SemComp-Bench is the first benchmark to jointly evaluate Outcome Achievement and Generation Reliability. Experiments with representative video generation models reveal substantial limitations in completing instructed tasks and preserving reference-grounded semantics, indicating that this capability remains largely underexplored. The potential of SemComp-Data to improve model performance through task-specific training remains to be empirically validated.

## References

[1] Kuaishou AI. Kling ai: Ai-powered video generation model. https://klingai.com, 2025.

[2] Zechen Bai, Hai Ci, and Mike Zheng Shou. Impossible videos. arXiv preprint arXiv:2503.14378, 2025.

[3] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

[4] ByteDance. Seedance: Multimodal video generation model. https://seed.bytedance.com, 2025.

[5] ByteDance Seed. Seedance 2.0 oficial launch. https://seed.bytedance.com/en/blog/ seedance-2-0-official-launch, 2026.

[6] Bytedance Seed. Seed1.8 Model Card: Towards Generalized Real-World Agency. arXiv preprint arXiv:2603.20633, 2026.

[7] Guanghui Chen et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

[8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.

[9] Google DeepMind. Veo: A state-of-the-art video generation model. https://deepmind.google/technologies/ veo/, 2025.

[10] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656, 2024.

[11] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103, 2024.

[12] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

[13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

[14] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[15] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for eficient video generative modeling. In International Conference on Learning Representations, volume 2025, pages 23378–23402, 2025.

[16] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[17] Pika Labs. Pika: Ai video generation platform. https://pika.art, 2025.

[18] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. Advances in Neural Information Processing Systems, 38, 2026.

[19] Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. Vmbench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13087–13098, 2025.

[20] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.

[21] Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, et al. Rise-video: Can video generators decode implicit world rules? arXiv preprint arXiv:2602.05986, 2026.

[22] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024.

[23] MiniMax. Hailuo ai video generation model. https://hailuoai.video, 2025.

[24] OpenAI. Video generation models as world simulators. https://openai.com/research/ video-generation-models-as-world-simulators, 2024.

[25] Adam Polyak, Amit Zohar, Andrew Brown, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

[26] Runway. Introducing runway gen-4. https://runwayml.com, 2025.

[27] Tencent Hunyuan Foundation Model Team. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/ abs/2511.18870.

[28] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[29] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025.

[30] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.

[31] Jianhao Yuan, Fabio Pizzati, Francesco Pinto, Lars Kunze, Ivan Laptev, Paul Newman, Philip Torr, and Daniele De Martini. Likephys: Evaluating intuitive physics understanding in video difusion models via likelihood preference. arXiv preprint arXiv:2510.11512, 2025.

[32] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.

[33] Yuke Zhao, Wangbo Zhao, Weijie Wang, Zeyu Zhang, Dakai An, Akide Liu, Yinghao Yu, Jiasheng Tang, Fan Wang, Wei Wang, et al. Worldolympiad: Can your world model survive a triathlon? arXiv preprint arXiv:2606.11129, 2026.

[34] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

## Supplementary Material

The supplementary material provides further details on SemComp-Data and SemComp-Bench. Specifically, it describes the title-filtering procedure (Sec. A), the domain and category taxonomy (Sec. B), category-specific definitions of reference and outcome states (Sec. C), alignment types and preservation attributes (Sec. D), and the complete evaluation prompt (Sec. E). It also presents dataset statistics and instances (Sec. F), additional generated-video comparisons (Sec. G), and the standard deviations of the reported results (Sec. H).

The complete curation pipeline and its stage-specific prompts are provided in the Code folder of the Code and Data Supplementary materials, while the Data folder provides the metadata, source-video URLs, and timestamps required to reconstruct SemComp-Data.

<table><tr><td>Configuration Group</td><td>Keyword Strings</td></tr><tr><td>news_keywords (13)</td><td>News; Briefing; Coverage; Current Affairs; Deep Dive; Documentary; Exclusive; Headline; Interview; Live; Live Update; Press Conference; Report.</td></tr><tr><td>movie_keywords (13)</td><td>Behind the scenes; Blooper; Casting; Movie Clip; Playback; Preview; Review; Spoiler; Teaser; TV Series; Talk; Talks; TalkShow.</td></tr><tr><td>entertainment_keywords (19)</td><td>ASMR; Celebrity; Challenge; Concert; Gaming; Highlights; Music Video; Podcast; Prank; Reaction; Reacts; Travel; Unboxing; Variety Show; Vlog; HBO; SportsCenter; NFL; NBA.</td></tr></table>

Table S1 Complete title-filtering keyword configuration.

## A Title-Based Keyword Filtering

Title-based filtering is used to remove narration-dependent or primarily informational videos whose outcomes cannot be reliably verified from visual evidence alone. Such videos often require spoken explanations, contextual knowledge, or other nonvisual information to determine whether the depicted task has been completed successfully, making them unsuitable for visually grounded evaluation. To ensure that this filtering step is transparent and reproducible, the implementation loads 45 keyword strings from the filtering configuration. Although these keywords are organized into three groups for readability and maintenance, the groups are flattened into a single list before matching. Each candidate title is checked against every keyword using direct, case-sensitive substring matching, as implemented in the released code. A video is excluded whenever its title contains at least one listed string with matching capitalization; an exact match to the entire title is not required. This simple deterministic procedure ensures that the same filtering criterion is applied consistently across all candidate videos. The complete filtering configuration is reported in Table S1.

## B Domain and Category Taxonomy

SemComp-Data organizes its instances into six broad domains and 21 fine-grained categories. The domains characterize general application scenarios, whereas the categories distinguish task-specific reference-to-outcome patterns. Table S2 lists the complete taxonomy.

<table><tr><td>Domain</td><td>Category</td><td>Definition</td></tr><tr><td rowspan="2">Food and Cooking</td><td>Dish Making Food Transformation</td><td>A dish or food product is completed from ingredients. The same food item undergoes a visible cooking-related state</td></tr><tr><td>Food Plating</td><td>change. Food elements are arranged into a complete presentation.</td></tr><tr><td rowspan="3">Beauty and Fashion</td><td>Styling</td><td>The appearance of the same person is visibly restyled.</td></tr><tr><td>Tool Cleaning</td><td>A beauty or fashion tool is cleaned to a reusable condition.</td></tr><tr><td>Try-on Product Reveal</td><td>An external appearance reference is applied to a subject. A previously concealed product becomes visible.</td></tr><tr><td rowspan="2">Sports and Fitness</td><td>Gear Setup Body Preparation</td><td>Sports gear or equipment is configured into a ready-to-use state.</td></tr><tr><td>Sports Edits</td><td>The subject's body enters a sports-related prepared state. Source event footage or identity cues are transformed into a completed sports-media edit</td></tr><tr><td rowspan="4">Crafts and DIY</td><td>Assembly</td><td>Separate parts are combined into a complete structure or object.</td></tr><tr><td>Restoration Renovation</td><td>The condition of the same object is restored or refurbished. A built space is reorganized or remodeled into a new spatial state.</td></tr><tr><td>Blueprint Construction</td><td>A physical structure is built according to an abstract design</td></tr><tr><td>Material Shaping</td><td>specification. A raw material is physically reshaped into a new form.</td></tr><tr><td rowspan="3">Gardening and Pets</td><td>Plant Treatment</td><td>A plant is improved through care-related operations.</td></tr><tr><td>Floral Design</td><td>Floral or plant elements are trimmed and arranged into an</td></tr><tr><td>Pet Grooming</td><td>aesthetic design. A pet's appearance is improved through grooming.</td></tr><tr><td rowspan="2">Arts and Precision</td><td>Artwork Creation</td><td>An artwork is completed from an unfinished visual basis.</td></tr><tr><td>Digital Creation</td><td>A digital creative product is completed from a visual or abstract reference.</td></tr><tr><td>Category</td><td>Reference State</td><td>Outcome State</td></tr><tr><td>Dish Making</td><td>Ingredients or an incomplete food preparation are visible.</td><td>A recognizable completed dish or food product is visible.</td></tr><tr><td>Food Transformation</td><td>A food item is visible before a</td><td>The same food item exhibits the intended visible state change.</td></tr><tr><td>Food Plating</td><td>cooking-related state change. Food components are present before final</td><td>The food components form a complete</td></tr><tr><td>Styling</td><td>arrangement. A person is visible before the target styling</td><td>plated presentation. The same person visibly exhibits the</td></tr><tr><td>Tool Cleaning</td><td>change.</td><td>completed styling result. A beauty or fashion tool is visibly soiled or The same tool is visibly clean and reusable.</td></tr><tr><td>Try-on</td><td>not ready for reuse. The subject and an external appearance</td><td>The subject visibly exhibits the referenced</td></tr><tr><td>Product Reveal</td><td>reference are identifiable. The product is concealed, covered, or not</td><td>garment, accessory, or appearance. The product is exposed and visually</td></tr><tr><td>Gear Setup</td><td>yet identifiable. Sports gear is unconfigured, disassembled,</td><td>identifiable. The gear is visibly configured in a</td></tr><tr><td>Body Preparation</td><td>or not ready for use. A subject is visible before completing a</td><td>ready-to-use state. The subject visibly reaches the intended</td></tr><tr><td>Sports Edits</td><td>sports-related preparation. Source event footage or identity cues for a</td><td>prepared state. A completed sports-media edit is visible.</td></tr><tr><td>Assembly</td><td>sports edit are visible. Separate or partially combined components The components form a complete structure</td><td></td></tr><tr><td>Restoration</td><td>are visible. A worn, damaged, or degraded object is</td><td>or object. The same object is visibly restored or</td></tr><tr><td>Renovation</td><td>visible. A built space is visible before</td><td>refurbished. The space exhibits the completed renovated</td></tr><tr><td>Blueprint</td><td>reorganization or remodeling. An abstract design, plan, or blueprint is</td><td>layout or appearance. A corresponding completed physical</td></tr><tr><td>Construction</td><td>visible.</td><td>structure is visible.</td></tr><tr><td>Material Shaping</td><td>Raw or incompletely shaped material is</td><td>The material has the intended completed</td></tr><tr><td>Plant Treatment</td><td>visible. A plant is visible before care or treatment.</td><td>form. The plant exhibits a visibly improved state</td></tr><tr><td>Floral Design</td><td>Unarranged floral or plant elements are</td><td>after treatment. The elements form a completed aesthetic</td></tr><tr><td>Pet Grooming</td><td>visible. A pet is visible before grooming.</td><td>arrangement. The same pet exhibits the completed</td></tr><tr><td>Artwork Creation</td><td>An unfinished artwork or visual basis is</td><td>grooming result. A completed artwork is visible</td></tr><tr><td>Digital Creation</td><td>visible. A visual reference or unfinished digital</td><td>A completed digital creative product is</td></tr><tr><td>Sculpting</td><td>artifact is visible. Raw or partially shaped sculptable material</td><td>visible. A completed three-dimensional sculpture is</td></tr></table>

Table S2 Domain–category taxonomy and category definitions used in SemComp-Data.

Table S3 Category-specific reference and outcome state definitions used for State Mining.

## C Reference and Outcome State Definitions

The construction pipeline uses category-specific state definitions to localize representative reference and outcome frames. A reference state describes the initial visual evidence required to ground the task, while an outcome state describes the visually verifiable completed result. Table S3 summarizes the definitions used for State Mining.

## D Alignment and Attributes

Instruction Structuring first assigns one of four alignment types and then selects instance-specific attributes to preserve from the reference image. The alignment type identifies the principal form of semantic grounding in the reference; it does not require all attributes associated with that type to remain unchanged. Table S4 gives the operational definitions. The complete preservation-attribute vocabulary contains 17 options spanning object semantics, visual appearance, human identity, and scene structure.

<table><tr><td>Alignment Type</td><td>Operational Definition</td><td>Typical Relevant Attributes</td></tr><tr><td>Object Element</td><td>Grounds the outcome in task-relevant object categories, components, ingredients, materials, quantities, or colors appearing in the reference, without requiring the exact appearance of a specific object instance.</td><td>Object category, composition, count, material, and color palette.</td></tr><tr><td>Identity</td><td>Requires the depicted person to remain identifiable across the reference and generated outcome.</td><td>Person identity, face appearance, body appearance, and pose.</td></tr><tr><td>Object Appearance</td><td>Grounds the outcome in the instance-level visual appearance of a specific reference object.</td><td>Object category, color palette, material, shape structure, surface appearance, and graphic details.</td></tr><tr><td>Scene</td><td>Grounds the outcome in the organization or context of the reference scene.</td><td>Scene type, scene layout, spatial relation, background context, and viewpoint.</td></tr><tr><td colspan="3">Complete preservation-attribute vocabulary (17): object_category, object_composition, object_count, material, color_palette, shape_structure, surface_appearance, graphic_details, person_identity, face_appearance, body_appearance, pose, scene_type, scene_layout, spatial_relation, background_context, and viewpoint.</td></tr></table>

Table S4 Operational definitions of the four alignment types and the complete preservation-attribute vocabulary. Attribute selection remains instance-specific.

## E SemComp-Bench Evaluation Prompts

For each generated video, 27 frames are uniformly sampled and presented in temporal order. The evaluator answers each question with yes or no and provides visual evidence for the decision. The two positive OA questions pass on yes; the remaining OA questions and all GR questions are failure-oriented and pass on no. SemComp-Bench combines a common system prompt with criterion-specific task prompts. The system prompt in Fig. S1 defines the evaluator role, restricts each binary decision to yes or no, and requires all requested question identifiers to appear in the output.

Outcome Achievement is evaluated using two prompt groups. The first prompt, shown in Fig. S2, evaluates Outcome Realization and Semantic Grounding through $Q _ { \mathrm { O A } } ^ { 1 }$ and $Q _ { \mathrm { O A } } ^ { 2 }$ . A brief video-frame description generated by the VLM during State Mining is supplied as auxiliary context; the binary decisions remain grounded in the visible sampled frames. The second prompt, shown in Fig. S3, evaluates Grounded Entity Consistency and Global Visual Continuity through $Q _ { \mathrm { O A } } ^ { 3 }$ and $Q _ { \mathrm { O A } } ^ { 4 }$ . Grounded Entity Consistency checks for unexplained disappearance, replacement, or task-relevant semantic drift of key entities, whereas Global Visual Continuity checks for abrupt whole-frame changes. These questions are evaluated from temporally ordered sampled frames, with the reference image and instruction additionally provided when required by the criterion. Together, the four OA criteria separate task fulfillment from outcome-video validity. Outcome Realization first establishes whether the instructed completed state is visibly present at a coarse semantic level, while Semantic Grounding checks the task-relevant correspondence between that outcome and the reference–instruction pair. Grounded Entity Consistency does not require pixel-level appearance constancy; it only flags unexplained changes that prevent key entities from remaining semantically identifiable. Global Visual Continuity instead operates at the whole-frame level and detects abrupt switches in the scene, viewpoint, layout, background, or composition. Neither safeguard requires the video to reproduce the intermediate procedure. Generation Reliability uses a dedicated standalone prompt for Physical Plausibility and a joint prompt for the remaining diagnostic criteria. Figure S4 shows the standalone prompt for $Q _ { \mathrm { G R } } ^ { 1 } { \mathrm { : } }$ and Fig. S5 shows the joint prompt for $Q _ { \mathrm { G R } } ^ { 2 } { - } Q _ { \mathrm { G R } } ^ { 5 }$ , covering Visual Clarity, Artifact-Free Rendering, Within-Scene Spatiotemporal Coherence, and Text and Interface Integrity. All five GR questions are failure-oriented, so a no response denotes a pass. GR is assessed independently of task completion and semantic grounding. Physical Plausibility examines visible motion, contact, interaction, and material behavior across the full temporally ordered sampled sequence. The remaining questions distinguish recognizability problems, scene-inconsistent rendering artifacts, local instability within otherwise continuous scenes, and malformed text or interface elements. Because these failure types are not mutually exclusive, the evaluator answers each question independently and may cite the same visual evidence for more than one criterion. Criterion-specific answers and concise evidence keep the reported pass rates interpretable and consistent with the main-paper score definitions.

![](images/b8af8f590d84b357b3de02eb44b340cb400b828c829b795aec8360186b8d4324.jpg)  
Figure S1 Common system prompt used for SemComp-Bench evaluation.

![](images/5c109ab52e3dfea871046ed89dd56f5e4e68ba3b5820ff1fe7b8115c9755246a.jpg)  
Figure S2 Outcome Achievement prompt for Outcome Realization $( Q _ { \mathrm { O A } } ^ { 1 } )$ and Semantic Grounding $( Q _ { \mathrm { O A } } ^ { 2 } )$

![](images/e9a7b381e884171272e6f9df57e34005fe9c7bc662f6ea9165cd19d3abc5350f.jpg)  
Figure S3 Outcome Achievement prompt for Grounded Entity Consistency $( Q _ { \mathrm { O A } } ^ { 3 } )$ and Global Visual Continuity (Q<sup>4</sup><sub>OA</sub>).

![](images/91533cb793f07ba1841a397c32cacf44201da68d0a1179caa4cd4f944ec0a202.jpg)  
Figure S4 Generation Reliability prompt for Physical Plausibility $( Q _ { \mathrm { G R } } ^ { 1 } )$

![](images/ce01f3a836415a72ef3302c9286eaf3193b3c01d5eca17ebcd9cd90ec3a1f418.jpg)  
Figure S5 Generation Reliability prompt for Visual Clarity, Artifact-Free Rendering, Within-Scene Spatiotemporal Coherence, and Text and Interface Integrity $( Q _ { \mathrm { G R } } ^ { 2 } { - } Q _ { \mathrm { G R } } ^ { 5 } )$

## F Dataset Statistics and Instances

The curation pipeline yields 1,273 SemComp-Data instances. Figure S6 presents 22 additional instances spanning diverse task categories. Each example shows the reference image, three temporally ordered frames from the paired outcome-centric clip, and the brief instruction.

![](images/338e69f80be97fbb5dd5a25005fa09558298996bf7975723a1fbace63ed5a055.jpg)  
Dress small dog in new dress

![](images/c7fcbb6e2defdbe3b104aec6110502bb24468232d7115d4b762445df5187e6a3.jpg)  
Cook into potato tortil a

![](images/9096acfdb2bcf271b3318d9a776ebe4f6f2ab6ec583e40241b910a62b8b3f1d3.jpg)  
Make topped cookie dough

![](images/116d3834a3892c1f1b8024263ef0540a902d30cd83aaf337aa2f0e034f539310.jpg)  
Make ingredients into mini ring cakes

![](images/73e716989e14a9d78822a8152e946437f0e23a249d496de175e337096dcbb8bc.jpg)  
Dress doll in July-themed outfit

![](images/1bf40a791e04234d87ecbb7126a5e38099010aaaf3c93b9c4359cbaca0348199.jpg)  
Assemble into layered pin

![](images/21bfe5a9d6a87f7f8db402f0fe86ace205c275b2beb6642636b77ad5e92a7cd2.jpg)  
Transform dark hair to blonde ombre

![](images/48855107aece361511366a20b7a57e3e9e20a2b1b4ab06960157b15656c9631f.jpg)  
Form yarn into magic circle

![](images/ab1dc1720c9d78209c3b805936a31282b0066c6708e635e09081a0676c2dfd9a.jpg)  
Print peacock on tote bag

![](images/b435e92d78424941d577bdb58b5f7902d7e1d81495d5fa9d47785d83d06c4106.jpg)  
Assemble block figures

![](images/353627c2aea1aa0aadf7979c5d7066fb65783614e166843ce2c572a662be03e7.jpg)  
Cook bacon to crisp strips

![](images/bc7015d5f38ff46b7dcb5eb604b4bcf33363633a31a59cafdbfa65e509226b54.jpg)

![](images/1e092b2f4c69c6c17da3ac5aaefd88de7b8bc0d7424747c1aaa33235c4c02bea.jpg)  
Wrap heart frame into yarn wreath  
Assemble bands into bracelet

![](images/2783b4f13dfc3211a04b6c08b0d1d574d4b7bf039831e82661a2472fbc6c5928.jpg)  
Make into fried sticks & sauce

![](images/f3ea666a8c4edeae9714624461b95f4b0fb03bad25766d3c86c288779744ccd3.jpg)  
Transform to shimmer gold makeup

![](images/db992f246bdea9cee236f436a090f60f9fcad9af4653048fc87ca989e1d9744e.jpg)  
Print multi-color 3D object

![](images/1dabdce6ea8f571a71d3a98dd99cc129b348fb957d4656f34b3572de8eca56ac.jpg)  
Fry raw eggs into fried eggs

![](images/285887b07f3ae2a4f48655cf10463b46cfa92c99b0047d8488ea3e4ba85d2dc0.jpg)  
Bake apple crisp dog treats

![](images/c75b4029c7d794c806a2f73568fa7a3c7d3e4ac3d4b042c7f500c24009e15404.jpg)  
Give man dramatic full makeup

![](images/b6939ce7bb6e315c28aa80f8c56fe39ebd32501477d7fdb35e74351035724cad.jpg)

![](images/bece0337deb91126c92284d2650c87b29521cd47e26e75c6555765cc4158c743.jpg)  
Build robot circuit from schematic  
Turn flat paper into decorated gift bag

![](images/dd71fbf1bd2b143dd8febc0a58c597e4411e6a84431c9039e57b3d3154ffe299.jpg)  
Cover graffiti wall with stucco

Figure S6 Additional SemComp-Data instances. Each example shows a reference image, three temporally ordered frames sampled from the paired outcome-centric video clip, and the corresponding brief instruction.

The arrows denote the intended semantic transitions to the completed outcomes. This notation does not imply that a generated video must reproduce the intermediate procedures. The splitting module first partitions each full-context video into shots and merges temporally adjacent, visually consistent shots into events. The event containing the verified outcome timestamp is then selected, after which duration-constrained extraction produces the outcome-centric clip. Events within the nominal target range of 3–4 s are retained in full; otherwise, an outcome-centered window is extracted from the full-context video with a target duration of 4 s for an overlong event or 3 s for an underlength event, with boundary-aware temporal shifting when necessary. This procedure results in a mean clip duration of approximately 4.03 s.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of instances</td><td>1,273</td></tr><tr><td>Minimum clip duration</td><td>3.020 s</td></tr><tr><td>Median clip duration</td><td>4.067 s</td></tr><tr><td>Mean clip duration</td><td>4.033 s</td></tr><tr><td>Maximum clip duration</td><td>4.125 s</td></tr><tr><td>SemComp-Core instances</td><td>60</td></tr><tr><td>Sampled evaluation frames</td><td>27</td></tr></table>

Table S5 Additional dataset and evaluation statistics.

## G Qualitative Results

Figure S7 provides four additional SemComp-Core comparisons spanning Sculpting, Artwork Creation, Restoration, and Food Transformation. Within each case, all seven models receive the same reference condition and detailed instruction; the paired brief instruction is displayed in the figure only for compact presentation. Their outputs are shown as temporally ordered frame sequences. These matched comparisons expose complementary failure modes that are not fully captured by a terminal frame alone. Some sequences preserve the reference scene but make limited progress toward the specified outcome, whereas others depict a partial state change while introducing entity deformation, material drift, or abrupt visual transitions. The examples therefore illustrate why Outcome Achievement and Generation Reliability provide complementary views of semantic task completion.

![](images/04fb475db68330efda4291533fb0f0d3a7b0897b0ebb9d5498206b31acf0bb8f.jpg)

![](images/24ca2a39a835b6954d766e61df038625cc4285dab01e7271f771819d91de3006.jpg)

![](images/bb2999acd6208fe2e5fdb5efd5857c9addb82d0d3463ec50c323a376aa8e4261.jpg)

![](images/e2cb306440f92d67a608190f64e3e5876cfa4c175441c27d6a806b89cc0b4ecb.jpg)  
Figure S7 Additional generated-video comparisons on four SemComp-Core instances. Outputs from all seven I2V models in each block were generated using the shared reference condition and detailed instruction; the paired brief instruction is displayed only for compact presentation. Temporally ordered output frames illustrate diferences in both outcome achievement and generation reliability.

## H Result Variability

Each generated video is evaluated by three independent VLM calls made at diferent times. Tables S6–S8 report the sample standard deviation of the three corresponding run-level scores. Specifically, for run-level score $x _ { r } ,$ we compute $\begin{array} { r } { \sigma ( x ) = \sqrt { \frac { \sum _ { r = 1 } ^ { 3 } ( x _ { r } - \bar { x } ) ^ { 2 } } { 3 - 1 } } } \end{array}$ . All entries are measured in percentage points (pp). The OA Score standard deviations use the conjunctive definition in which a sample must pass all four OA criteria, whereas the GR Score standard deviations are computed from the run-level mean of the five GR criterion pass rates. Thus, the aggregate-score columns follow exactly the definitions used for the results reported in the main paper.

<table><tr><td>Model</td><td> $\sigma ( A _ { \mathrm { o r } } )$ </td><td> $\sigma ( A _ { \mathrm { s g } } )$ </td><td> $\sigma ( A _ { \mathrm { g e c } } )$ </td><td> $\sigma ( A _ { \mathrm { g v c } } )$ </td><td>σ(OA Score)</td></tr><tr><td>Seedance 2.0</td><td>0.96</td><td>0.96</td><td>2.55</td><td>7.52</td><td>4.41</td></tr><tr><td>Wan2.2-TI2V-5B</td><td>4.19</td><td>4.41</td><td>3.85</td><td>0.96</td><td>3.33</td></tr><tr><td>Wan2.2-I2V-A14B</td><td>1.67</td><td>2.55</td><td>5.09</td><td>3.47</td><td>2.89</td></tr><tr><td>CogVideoX1.5-5B-I2V</td><td>3.33</td><td>4.19</td><td>2.55</td><td>0.96</td><td>0.96</td></tr><tr><td> $\mathrm { S k y R e e l s – V 2 – I 2 V - 1 4 B  – 7 2 0 P }$ </td><td>1.67</td><td>3.47</td><td>3.47</td><td>2.55</td><td>0.96</td></tr><tr><td> $\mathrm { H Y ^ { \dagger } - 1 . 5 - 7 2 0 P \mathrm { - } I 2 V }$ </td><td>2.55</td><td>2.55</td><td>2.89</td><td>1.92</td><td>3.47</td></tr><tr><td>Phantom-1.3B</td><td>0.96</td><td>1.92</td><td>1.92</td><td>0.96</td><td>0.96</td></tr></table>

Table S6 Sample standard deviations for SemComp-Core Outcome Achievement under detailed instructions (pp; three runs). <sup>†</sup>HY denotes HunyuanVideo.

<table><tr><td>Model</td><td> $\sigma ( G _ { \mathrm { p p } } )$ </td><td> $\sigma ( G _ { \mathrm { v c } } )$ </td><td> $\sigma ( G _ { \mathrm { a f r } } )$ </td><td> $\sigma ( G _ { \mathrm { w s c } } )$ </td><td> $\sigma ( G _ { \mathrm { t i } } )$ </td><td> $\sigma ( \mathrm { G R } \ \mathrm { S c o r e } )$ </td></tr><tr><td>Seedance 2.0</td><td>3.33</td><td>0.96</td><td>0.96</td><td>1.92</td><td>0.96</td><td>1.58</td></tr><tr><td>Wan2.2-TI2V-5B</td><td>0.96</td><td>2.55</td><td>6.74</td><td>12.95</td><td>0.96</td><td>3.95</td></tr><tr><td>Wan2.2-I2V-A14B</td><td>5.85</td><td>0.96</td><td>1.67</td><td>1.92</td><td>4.19</td><td>1.53</td></tr><tr><td>CogVideoX1.5-5B-I2V</td><td>2.55</td><td>13.47</td><td>6.31</td><td>8.82</td><td>8.55</td><td>7.12</td></tr><tr><td>SkyReels-V2-I2V-14B-720P</td><td>0.96</td><td>3.47</td><td>6.74</td><td>11.10</td><td>3.85</td><td>4.53</td></tr><tr><td> $\mathrm { H Y ^ { \dagger } - 1 . 5 - 7 2 0 P \mathrm { - } I 2 V }$ </td><td>8.82</td><td>1.92</td><td>0.00</td><td>6.94</td><td>0.96</td><td>1.35</td></tr><tr><td>Phantom-1.3B</td><td>5.09</td><td>0.96</td><td>8.22</td><td>15.84</td><td>6.94</td><td>7.20</td></tr></table>

Table S7 Sample standard deviations across three evaluation runs for the SemComp-Core Generation Reliability results under detailed instructions. All values are in percentage points. <sup>†</sup>HY denotes HunyuanVideo.

<table><tr><td>Model</td><td>Modality</td><td>Instruction</td><td> $\sigma ( A _ { \mathrm { o r } } )$ </td><td> $\sigma ( A _ { \mathrm { s g } } )$ </td><td> $\sigma ( A _ { \mathrm { g e c } } )$ </td><td> $\sigma ( A _ { \mathrm { g v c } } )$ </td><td> $\sigma ( \mathrm { O A }$  Score)</td></tr><tr><td rowspan="3">Wan2.2-A14B</td><td>I2V</td><td>Detailed</td><td>1.67</td><td>2.55</td><td>5.09</td><td>3.47</td><td>2.89</td></tr><tr><td>T2V</td><td>Detailed</td><td>2.55</td><td>4.19</td><td>1.67</td><td>4.41</td><td>2.55</td></tr><tr><td>T2V</td><td>Brief</td><td>0.96</td><td>0.96</td><td>2.55</td><td>7.26</td><td>0.96</td></tr><tr><td rowspan="3">CogVideoX1.5-5B</td><td>I2V</td><td>Detailed</td><td>3.33</td><td>4.19</td><td>2.55</td><td>0.96</td><td>0.96</td></tr><tr><td>T2V</td><td>Detailed</td><td>3.33</td><td>0.96</td><td>2.55</td><td>0.96</td><td>1.67</td></tr><tr><td>T2V</td><td>Brief</td><td>2.89</td><td>1.67</td><td>0.96</td><td>3.47</td><td>0.96</td></tr><tr><td rowspan="3"> $\mathrm { H Y ^ { \dagger } { - } 1 . 5 { - } 7 2 0 P }$ </td><td>I2V</td><td>Detailed</td><td>2.55</td><td>2.55</td><td>2.89</td><td>1.92</td><td>3.47</td></tr><tr><td>T2V</td><td>Detailed</td><td>0.00</td><td>2.55</td><td>6.74</td><td>5.00</td><td>1.92</td></tr><tr><td>T2V</td><td>Brief</td><td>1.67</td><td>3.33</td><td>4.19</td><td>5.36</td><td>0.00</td></tr></table>

Table S8 Sample standard deviations across three evaluation runs for the Outcome Achievement conditioning study. All values are in percentage points. <sup>†</sup>HY denotes HunyuanVideo.