# P<sup>3</sup>-VLM: A Point-based Alternative for Grounded 3D Vision-Language Models

Anna-Maria Halacheva, Jan-Nico Zaech, Sombit Dey, Luc Van Gool, Danda Pani Paudel INSAIT, Sofia University “St. Kliment Ohridski”

<sup></sup> https://p3vlm.insait.ai/

![](images/00036c7263fa338b28bcb1ce8a81c32cce86ff345fb4f1728b69efab7fb8bb1b.jpg)  
Figure 1: We address bounding box induced semantic leakage and thus the visual blindness in 3D VLMs using a point-based grounding alternative. This is challenging because both 3D VLMs and blind LLMs struggle with point queries. To overcome this, we develop a novel tri-pathway 3D VLM.

## Abstract

Examining existing benchmarks for grounding in 3D VLMs reveals a pervasive “semantic leakage" in the evaluation and prompting protocols: a heavy reliance on bounding boxes as spatial prompts or representation primitives. Because bounding boxes are highly informative of object categories, they introduce a shortcut for the models to neglect visual information during 3D VLM training. We demonstrate this with a “blind" LLM that, prompted only with the box coordinates, can perform surprisingly well. To further analyze this shortcut, we propose point-based grounding prompting – querying models with a single 3D point instead of a box. Under this protocol, the accuracy of the blind LLM drops from 36.4% to 18.0% on ScanNet’s Nr3D, and similarly degrades for 3D VLMs (e.g. LL3DA).

Building on this insight, we introduce P<sup>3</sup>-VLM, a novel tri-pathway 3D VLM that supports point-based prompts and achieves 52.1% state-of-the-art accuracy on location-grounded captioning under this setting. To achieve this, we redesign the architecture while keeping it detector-free, scene-centric, and suitable for 3D GS inputs. P<sup>3</sup>-VLM utilizes three complementary pathways (location-, task-, and context-aware), enabling strong reasoning about scenes without any bounding box dependencies. The key benefit of point-based learning is a consistent improvement across settings: P<sup>3</sup>-VLM achieves state-of-the-art performance under point-based grounding while also improving robustness, OOD generalization, and performance on ungrounded scene-centric 3D VQA benchmarks, reflecting its reduced reliance on bounding-box shortcuts and resulting stronger visual-spatial reasoning.

## 1 Introduction

3D Vision-Language Models (VLMs) hold the promise of generalist scene understanding: given a 3D representation of an environment, such a model should be able to answer both grounded queries – those anchored to a specific location or object in the scene – and ungrounded queries about the scene at large [8, 24, 8, 20, 23]. Significant progress has been made in this direction, yet a fundamental question remains open: to what extent do current 3D VLMs actually observe the scene, rather than exploit contextual shortcuts?

We investigate this question through grounded object understanding: identifying the correct object given a 3D spatial prompt. This requires jointly reasoning about geometry and semantics, including localizing the queried region, interpreting its visual content, and producing a semantically accurate response [11, 6, 19, 18]. We find that existing object- and scene-centric paradigms share a common issue of “semantic leakage” in evaluation protocols, arising from the widespread use of bounding boxes as either prompts or representation primitives. Because bounding boxes carry strong categorical cues through their size, shape, and aspect ratio, they enable shortcut solutions (e.g., a tall narrow box often corresponds to a door; beds have standardized measurements) that reduce the need for true visual-spatial reasoning. They also implicitly assume accurate object localization, which is usually unrealistic in real-world 3D applications.

The box dependency appears across both object-centric VLMs [24, 23, 58], which inherit box-based object representations from detectors, and scene-centric models [19, 8], which still rely on bounding boxes as grounding prompts at inference time. As a result, models can exploit categorical priors encoded in box geometry rather than visual evidence. We demonstrate this effect by showing that a blind LLM trained only on bounding box coordinates, without any visual input, achieves 36.4% accuracy on Nr3D as shown in Fig. 1. We further show that the overreliance on bounding boxes makes the existing 3D VLMs fragile, leading to significant degradation in scenarios with atypical object sizes, noisy reconstructions, or poorly aligned bounding boxes in Fig. 2 (robustness analysis).

These findings motivate both a new evaluation protocol and a new model design. We propose pointbased grounding prompting, where queries are specified using only a single 3D point on the target object. Unlike bounding boxes, point prompts do not reveal object shape or extent, substantially reducing category-specific geometric cues and incentivizing models to rely on genuine visual-spatial understanding. Under this stricter protocol, the shortcut enabled by bounding boxes largely disappears: the accuracy of the blind LLM drops sharply from 36.4% to 18.0% on Nr3D. Existing 3D VLMs, including scene-centric models such as [19, 8], also degrade substantially, further exposing their overreliance on bounding-box cues (e.g. LL3DA drops from 47.7% to 33.1%, see Fig. 1).

To address this new challenging setting, where existing scene-centric 3D VLMs struggle, we introduce P<sup>3</sup>-VLM, a detector-free, scene-centric 3D VLM built on 3D Gaussian scene representations (3D GS) and designed specifically for point-based grounding. P<sup>3</sup>-VLM utilizes a tri-pathway architecture consisting of: (1) a location-aware pathway that encodes compact positional cues from the query point, (2) a task-aware pathway that sparsifies and refines query-relevant volumetric features, and (3) a context-aware pathway that summarizes the global scene for grounded reasoning. By anchoring visual features directly to scene voxels rather than detector-derived geometry, P<sup>3</sup>-VLM enables spatially grounded reasoning under the proposed point-based alternative.

Under point-based grounding, P<sup>3</sup>-VLM achieves 52.4% class accuracy on Nr3D, substantially outperforming both the blind LLM baseline (18.0%) and previous 3D VLMs adapted to this setting (33.1% and 29.3% for LL3DA and GaussianVLM, respectively; see Tab. 2). This large margin demonstrates that its visual encoder is genuinely load-bearing. Moreover, by avoiding bounding-box shortcuts, P<sup>3</sup>-VLM is not only more robust but also exhibits stronger scene-level OOD generalization (see Fig. 2), while maintaining consistent gains on ungrounded scene understanding benchmarks such as ScanQA [3] and SQA3D [33]. Overall, this work makes the following contributions:

• We identify bounding box leakage as a pervasive shortcut in current 3D VLMs, hurting visual-spatial understanding, and introduce point-based grounding as a stricter protocol.

• We present P<sup>3</sup>-VLM, a detector-free, scene-centric 3D VLM operating on 3D Gaussian representations via three complementary pathways (location, task, and context aware).

• We show that P<sup>3</sup>-VLM achieves SOTA performance under point-based grounding while improving robustness, OOD generalization, and performance on VQA benchmarks.

![](images/b03db2f0f90d5c2df7737a5ae1a6e4f8d66949d608e4ade832fac9e45ab6c8e1.jpg)

![](images/b16bf76e0463b00192b6a02879627ac9a54eb298db3c842aa5c8bc273201b955.jpg)

![](images/903b67ac82673d060799e498010e7912974f806809fa0ab7053d053c9058d9fc.jpg)  
Figure 2: Performance, robustness, and generalization. Our P<sup>3</sup>-VLM consistently outperforms VLM baselines LL3DA [8] and GaussianVLM [19] (GSVLM). Notably, P<sup>3</sup>-VLM on points (p) offers better robustness and generalization than box-based methods (b), despite less spatial information.

## 2 Related Work

## 2.1 3D-Grounded Scene Understanding

3D-grounded scene understanding refers to tasks involving visual grounding in 3D space, which can manifest on either the input or the output side of the model [6, 1, 19, 34, 30]. On the output side, the model receives a reference to an object or category and must predict a spatial specification for the matching object(s) – a 3D bounding box or segmentation mask [34, 28, 30, 32, 2, 52, 27, 42]. Specialist models for this setting [28, 30], including those built on language backbones [34, 52, 2], achieve strong performance by combining task-specific pretraining, structured output modeling, and dedicated detection heads. However, because of this specialization, the state-of-the-art methods sacrifice their free-form answer and thus generalist capabilities. Generalist models with multi-task open-language capabilities [8, 23, 24] have been unable to match specialist performance on outputside grounding, lacking, for instance, the structured output modeling and full-scale task-specific supervision. Furthermore, existing free-form benchmarks for diverse scene understanding are largely restricted to ScanNet-scale datasets [3, 33, 20, 24, 25, 6, 1, 19], hindering the training of generalist models for output-side grounding at the scale necessary to achieve state-of-the-art performance [34].

Input-side grounded understanding entails providing the grounding on the input side through a 3D location query, with model output – a caption of the referenced object [19, 8, 18, 58, 24, 56]. This setting is more compatible with open-form answer models. Some free-form VLMs also report bounding box outputs in this setting, yet those are obtained via independent, frozen detectors [9, 41] entirely separate from their core architecture [8, 23, 24], making the box prediction not a product of the model’s learned visual understanding. Exploring free-form 3D generalist VLMs, we operate in the input-side captioning setting [19, 18, 56]. The captioning quality directly measures the VLM’s visual understanding, not focusing on box predictions, obtainable through an external module.

## 2.2 3D VLMs for Multi-Task Scene Reasoning

VLMs for 3D scenes aim to support a broad range of tasks, including object captioning [6, 1], question answering [3, 20], spatial reasoning [33, 54] and beyond [19, 20], through a single unified model [19, 8, 55, 39]. The dominant paradigm in this setting has been object-centric: detect a set of object proposals, encode their point cloud features to separate tokens, and align them with language [54, 24, 23, 29, 57, 47, 53, 10, 46, 31, 10]. Prior research suggests that this approach is fundamentally limited as scene-level understanding is bounded by detector recall [19], and is emergent rather than by design [19, 56, 36, 16]. Thus, a parallel line of work moves away from detector dependencies, spanning video VLMs for 3D understanding [17, 55, 39, 22, 7, 49, 45], and native 3D models. The latter processes 3D scenes either (1) in a region-based manner requiring incorporation of advanced methods for instilling cross-regional awareness (e.g., GCNs [18]) or (2) via scene-centric strategies that encode the environment at once as a whole [19, 8, 56]. GaussianVLM [19] pioneered the use of 3D GS for scene-centric understanding and acts as a baseline for this work. However, it suffers from reduced robustness and generalization due to reliance on semantic shortcuts (Sec. 3.2,5.2, D.3).

## 3 From Bounding Box Semantic Leakage to Point Grounding

A foundational assumption underlying 3D VLM evaluation is that grounded query performance reflects genuine visual-spatial reasoning – an assumption which we investigate in this section. When models are queried about a specific object via a bounding box prompt – the universal convention across 3D VLMs – the box itself encodes strong categorical information: its dimensions, aspect ratio, and volume are highly predictive of object class. A tall, narrow box standing up from floor level implies a door; a box with measurements of a queen-sized bed – a bed, etc. We term this phenomenon bounding box semantic leakage and demonstrate that it acts as a dominant signal that can bypass visual perception in 3D VLMs when it comes to grounding tasks.

3D-Grounded Object Captioning. To investigate this phenomenon, we focus on the task of 3D-Grounded Object Captioning, which requires a model to generate a natural language description for a specific object given a spatial query. We evaluate various 3D VLMs on two ScanNet-based benchmarks: Nr3D [1] and ScanRefer [6]. Firstly, we evaluate models under the standard prompting protocol, bounding-box prompting, specifying the query location as a box center and extent.

## 3.1 Bounding Box Semantic Leakage

We aim to isolate the contribution of geometric priors from other key aspects of 3D grounding such as semantics and context awareness. For this, we conduct two controlled experiments on Nr3D. Specifically, we examine the effect of the strong reliance on bounding boxes within the textual and/or the visual component of the prompt, highlighting that the leakage can occur via both information channels and cause over-reliance on this geometric prior rather than utilization of rich visual features.

Table 1: Object-centric VLMs do not utilize their 3D encoder. A blind LLM operating on bounding box coordinates alone, and an LLM with only box coordinate embeddings, perform close to the full OD-VLM on Nr3D.
<table><tr><td>Model</td><td>Class Acc. (%)</td></tr><tr><td>Blind LLM (box coords only)</td><td>36.4</td></tr><tr><td>Box encoder + LLM</td><td>44.2</td></tr><tr><td>Full OD-VLM (LEO)</td><td>45.1</td></tr></table>

Leakage via Textual Prompting. First, we examine the effect of bounding boxes as grounding prompts within the textual LLM task prompt. We fine-tune a "blind" LLM (Qwen3-VL 4B Instruct LLM [4]) via LoRA to predict object classes from box center, yaw and extent alone, with no visual encoder of any kind. The fine-tuned blind LLM achieves 36% class accuracy – a significant result that establishes box geometry as a strong categorical signal independent of any visual observation.

Leakage via Architectural Priors. Leakage can also occur through the visual channel, particularly in Object-Detector VLMs (OD-VLMs). These models typically use a decoupled architecture: a detector segments the scene, and a 3D encoder transforms each instance’s point cloud into tokens for the LLM. We investigate whether the 3D encoder contributes meaningful semantic understanding beyond what the detector’s box metadata already provides.

We ask whether the visual encoder in a full object-centric VLM is strongly utilized on top of this prior. For this, we evaluate an OD-VLM (LEO [24]) against a modified version where the visual features are simply encodings of the bounding boxes (center and extent) of the objects within the scene, derived with a lightweight 2-layer MLP. We observe very close performance of the two models (Tab. 1). These results suggest that, under bounding-box-based visual prompting, object class predictions are once again driven primarily by geometric priors rather than visual features.

## 3.2 Point-Based Grounding

Our findings indicate that the prevalent use of bounding boxes as spatial prompts may provide geometric priors that simplify the grounding task. To isolate visual perception from these structural cues, we propose point-based grounding prompting: a protocol where location queries are specified as a single 3D coordinate (x, y, z). By removing information about the target’s extent or aspect ratio, it serves as a more faithful benchmark for a model’s ability to extract semantics from visual features.

The effect of removing box geometry is immediate. We fine-tune two variants of the blind LLM: one trained on bounding box prompts and evaluated accordingly, and one trained and evaluated on point-based prompts (Supp. Sec. A.1). Class accuracy drops from 36.4% to 18.0% (Tab. 2) – a 50% relative reduction achieved by changing only the prompt format, with no modification to the model architecture. This confirms the prior inherent in the bounding-box grounding prompts, and its reduced effect in point prompts.

![](images/f4780879d5cc79e5e053165d22d758cec78b59dfc689c0360f6e3ff7af6a5e5b.jpg)  
Figure 3: P<sup>3</sup>-VLM architecture overview. Three complementary pathways process the voxelized 3D scene: the location-aware pathway extracts a single token at the queried point; the task-aware pathway produces prompt-guided scene tokens by first refining the task prompt tokens and subsequently utilizing them for visual refinement of the sparsified features; the context-aware pathway distills 8 global scene summary tokens via unconditional cross-attention. Notably, the context and task pathways operate on the sparsified features for efficiency gains.

We then ask whether existing scene-centric VLMs can close this gap when adapted to the new prompt. For each model, we fine-tune on point-based grounding and re-evaluate under the stricter protocol (training details in the Supplementary A.1). The results in Tab. 2 show that fine-tuning does recover some performance, yet a substantial gap relative to the box-based baseline persists across all models: on Nr3D LL3DA drops from 47.7% to 33.1%, and GaussianVLM from 46.2% to 29.3%. The fact that explicit training on point prompts does not close this gap suggests that the limitation is not one of prompt formatting, but of visual capacity – these models have not developed the dense visual representations that point-based grounding demands.

## 4 P<sup>3</sup>-VLM

P<sup>3</sup>-VLM is a scene-centric 3D VLM, that takes as input 3D Gaussian Splats (GS) and a natural language query and produces a free-form language response relying on point-based spatial prompting. An overview of the architecture is shown in Fig. 3. The scene is first voxelized: a sparse 3D GS encoder processes the raw Gaussian splats into a set of voxel features $F ~ \in ~ \mathbb { R } ^ { N \times D }$ with associated 3D coordinates $C \in \mathbb { R } ^ { N \times 3 }$ , where each feature is anchored to a specific spatial location. These voxel features then feed three complementary pathways: location-aware, task-aware, and context-aware, which together produce the visual token sequence as input to the LLM. To the visual tokens, we append the user language task tokens L.

Table 2: Object captioning performance on Nr3D and ScanRefer with different prompting strategies. The blind LLM’s sharp drop confirms that bounding box geometry was the dominant signal. Also the 3D VLM baselines (LL3DA, GaussianVLM) trained on boxes drop considerably even after fine-tuning on point prompts.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Box</td><td colspan="2">Point</td></tr><tr><td>Acc</td><td>Sim</td><td>Acc</td><td>Sim</td></tr><tr><td>Nr3D</td><td>Blind LLM LL3DA GaussianVLM</td><td>36.4 47.7 46.2</td><td>48.1 48.2</td><td>18.0 33.1 29.3</td><td>39.5 35.5</td></tr><tr><td>Scan Refer</td><td>Blind LLM LL3DA GaussianVLM</td><td>38.1 48.02 50.4</td><td>55.9 59.1</td><td>18.4 29.2 33.2</td><td>34.9 41.4</td></tr></table>

## 4.1 Tri-Pathway Visual Processing

The 3D visual encoder produces rich, yet dense, visual features. Even for our downsampled scans, this can produce more than 10k tokens, which are computationally heavy to be processed as a sequence in the attention mechanism. To alleviate this, we sparsify the dense visual features by aggregation and refinement, which additionally increases their spatial awareness, to make them consumable by the context and task pathways. Notably, the location pathway operates on raw encoder features as it focuses exclusively on locality rather than global awareness. Detailed implementation specifications and hyperparameter choices are provided in Supp. Sec. D.

Sparsification. The raw voxel features are spatially downsampled to at most 2048 tokens per scene via average pooling on a uniform 3D grid, making subsequent self-attention tractable. The downsampled features are then spatially enriched and lifted to LLM space by a stack of spatiallyaware self-attention layers where both queries and keys are rotated by 3D Rotary Position Embeddings (3D RoPE) [43]. Applying 3D RoPE to both Q and K encodes pairwise relative 3D distances between voxels directly in the attention scores, giving the projector spatial awareness without positional embedding lookups. The output is $\mathbf { S } \in \widetilde { \mathbb { R } ^ { N \times D _ { \mathrm { l n } } } }$ with LLM hidden dimension $D _ { \mathrm { l m } }$

Location-Aware Pathway. The location-aware pathway produces a single token that encodes the visual content at the queried 3D point. Given the query location $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ , we identify the closest voxel center (with voxel id i). The voxel feature F is concatenated with the normalized query location and passed through a four-layer MLP projecting to LLM hidden dimension:

$$
\mathbf { T } _ { \mathrm { l o c } } = \mathrm { M L P } ( [ \mathbf { F } _ { i } \left. \mathbf { p } \right] ) \mathbf { \Lambda } \in \mathbb { R } ^ { D _ { \mathrm { l m } } } .\tag{1}
$$

The concatenation of raw spatial coordinates with the voxel feature allows the MLP to jointly reason about what is at a location and where that location is, without a separate conditioning mechanism. When no query location is provided the pathway is suppressed and no location token is injected.

Context-Aware Pathway. The context-aware pathway produces a compact, prompt-independent scene summary. A set of $K { = } 8$ learnable query tokens $( Q _ { \mathrm { l e a r n } } )$ cross-attends over the sparsified and spatially enriched features S through two cross-attention layers

$$
\begin{array} { r } { \mathbf { T } _ { \mathrm { c t x } } = \mathrm { X A t t n } ( \mathbf { Q } _ { \mathrm { l e a r n } } , \mathbf { S } ) \ \in \mathbb { R } ^ { K \times D _ { \mathrm { l m } } } . } \end{array}\tag{2}
$$

Because the queries are unconditional, they learn to summarize scene-level properties (object layout, room type, spatial extent) useful regardless of the query. We observe that this global context allows the task-aware pathway to specialize more sharply: without context tokens, the compressor attends broadly across all voxels; with them, attention concentrates on query-relevant spatial regions.

Task-Aware Pathway. The sparsified visual representation is consumed by the task-aware pathway for further sparsification of the tokens with respect to the user task query. The language query L is processed by two self-attention layers to $\mathbf { L } _ { \mathrm { r e f i n e d } } .$ . Subsequently, we add a 3D RoPE embedding [43] to each voxel token, which then cross-attends to the text tokens. This asymmetry encodes the relevance of each prompt token for each spatial location in the attention score, allowing selective refinement of spatially relevant voxels. The update is applied as a residual with zero-initialized output projection for stable training:

$$
\mathbf { S } _ { \mathrm { t a s k } } = \mathrm { L a y e r N o r m } ( \mathbf { S } + \mathbf { W } _ { \mathrm { o u t } } \cdot \mathrm { X A t t n } _ { 3 \mathrm { D R o P E } } ( \mathbf { S } , \mathbf { L } _ { \mathrm { r e f i n e d } } ) ) .\tag{3}
$$

The task-refined scene $\mathbf { S } _ { \mathrm { t a s k } }$ is compressed via 128 learnable tokens to $\mathbf { T } _ { \mathrm { t a s k } }$ LLM-ready tokens.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We follow standard training protocols within the scene-centric 3D VLM field [8, 19] and train our models within the ScanNet [15] settings. Notably, we use 3D GS representation [30] as visual input. We perform joint training on multiple of tasks without task-specific fine-tuning: general question-answering (ScanQA [3]), situated question-answering (SQA3D [33]) and location-grounded captioning (on Nr3D [1], ScanRefer [6]).

Evaluation. Following baseline protocols [19], we evaluate question-answering on a multitude of metrics covering linguistic similarity (METEOR (M) [5], ROUGE (R) [12], BLEU-4 (B4) [35], CIDEr (C) [44]), and exact-match accuracy. For the latter, we adopt the refined EM protocol (EM-R) [24]. For captioning, we evaluate target object identification via accuracy (Acc) and caption similarity via Sentence-BERT Similarity (Sim) [40], with metrics discussion provided in Supp. Sec. A.2. Results are reported on the validation sets of ScanRefer, Nr3D, ScanQA and test set of SQA3D.

Training. Training is carried out in two stages. In the first stage, only the location-aware pathway is active, with the model being trained on point-based grounded captioning (Nr3D and ScanRefer), using the single location token as the sole visual input. This allows the location extractor to learn a stable voxel-to-language mapping before the full scene representation is introduced. In the second stage, all three pathways are active and the model is trained jointly on the full task mixture. Both stages are trained under the standard autoregressive language modeling objective (Supp. Sec. A.5).

Implementation Details. P<sup>3</sup>-VLM uses Qwen3-VL-4B [4] as LLM backbone, fine-tuned with LoRA [21] with rank 16 [19, 24]. The 3D encoder is initialized from SceneSplat [30] and kept unfrozen. The three visual pathway components are trained from scratch. Training takes 10 hours on 2 NVIDIA-H200 GPUs with 5 epochs of pre-training the location pathway, followed by 12 epochs of full model training, using batch size 16 across both stages. We use the AdamW optimizer with a cosine annealing learning rate schedule, learning rate 3e-5 and weight decay of 0.05. We perform evaluation every epoch. For further training and hyperparameter details, please refer to Supp. Sec. A.5. We train and evaluate two variants of P<sup>3</sup>-VLM differing solely in the adopted location prompting strategy - bounding box $( P ^ { 3 } - V L M ( b )$ with box format [center, yaw, extent]) or point $( P ^ { 3 } { \bar { \cdot } } V L M ( p ) { \bar { ) } }$ For the latter we use the centers of the objects, derived from the ground truth segmentation [15].

To isolate language model performance from architectural contributions, we also retrain one of the baselines, GaussianVLM, using its original training protocol while replacing its language decoder with ours: Qwen3-VL-4B (denoted GaussianVLM<sup>Q</sup>). This ensures that both $\mathrm { P ^ { 3 }  – V L M }$ and GaussianVLM<sup>Q</sup> share identical 3D (SceneSplat) and language backbones.

## 5.2 Grounded Object Understanding

We evaluate P<sup>3</sup>-VLM on the ability to perform grounded visual understanding via location-prompted object captioning on Nr3D and ScanRefer. As other methods do not report class accuracy on the captioned object, we obtain the predictions from all baseline models and extract the predicted object class with an LLM-based evaluator (Gemini 2.5 Flash [13]). More details on our evaluation protocol are provided in Sec. A.3.

Results. As shown in Table 3, both $\mathrm { P ^ { 3 } - V L M }$ (b) and P<sup>3</sup>-VLM (p) obtain significantly better classaccuracy on both Nr3D and ScanRefer captioning. As expected, $\mathrm { P ^ { 3 } - V L M }$ (b), our box-based method, slightly outperforms our point-based method on indomain grounded tasks, courtesy of having access to the object’s geometric properties. The stable sentence similarity results (Sim) demonstrate that $\mathrm { P ^ { 3 } - V L M }$ not only identifies the correct object but also maintains high linguistic similarity to the entire caption.

Priors analysis in points and boxes. Even point prompts are not entirely free of priors: absolute coordinates can still correlate with object category through scene-layout regularities. We examine and quantify these residual priors with a geometry-only kNN classifier, trained on the Nr3D train split and evaluated on the validation split (38 classes with at least 8 instances). Using the object center as the point prompt, the horizontal position (xy) alone yields 8.1% accuracy, the height (z) alone 17.8%, and the full 3D center (xyz) 16.5%, compared to 33.0% for the full bounding box. Height thus accounts for nearly all of the residual signal, while absolute room position is a weak cue. Per class, the center-based kNN reaches $\geq 3 0 \%$ recall on only 3/38 classes (chair, picture, monitor), compared to 15/38 for the box-based one: the residual prior is confined to objects with a consistent mounting or placement height, whereas shape-distinctive objects (e.g., bed, keyboard, door) leak almost entirely through box geometry.

Table 3: Object captioning performance on Nr3D and ScanRefer. We report classification accuracy (Acc) and caption similarity (Sim). $\mathrm { P ^ { 3 } \mathrm { - } \check { V } L M }$ consistently outperforms baselines in Acc, showing superior visual grounding across point (p) and box (b) prompts. GaussianVLM<sup>Q</sup>: [19] with Qwen3- VL-4B LLM, retrained.
<table><tr><td></td><td colspan="2">Nr3D</td><td colspan="2">ScanRefer</td></tr><tr><td>Model</td><td>Acc</td><td>Sim</td><td>Acc</td><td>Sim</td></tr><tr><td>LL3DA (b)</td><td>47.7</td><td>48.1</td><td>48.0</td><td>55.9</td></tr><tr><td>GaussianVLM (b)</td><td>46.2</td><td>48.2</td><td>50.4</td><td>59.1</td></tr><tr><td>GaussianVLMQ (b)</td><td>47.1</td><td>47.8</td><td>51.7</td><td>59.3</td></tr><tr><td>P3-VLM (p)</td><td>52.1</td><td>46.0</td><td>56.1</td><td>54.2</td></tr><tr><td>P3-VLM (b)</td><td>52.4</td><td>46.3</td><td>58.4</td><td>52.5</td></tr></table>

Table 4: Robustness of 3D VLMs using box prompting to geometric shortcut disruption. Class accuracy (%) on Nr3D under two failure regimes: non-tight bounding boxes, and noisy reconstructions, provided together with In-distribution performance for reference. Models relying on bounding box semantic leakage degrade sharply across both conditions.
<table><tr><td></td><td>In-Dist.</td><td>Non-Tight Boxes</td><td>Noisy Recon.</td></tr><tr><td>LL3DA</td><td>47.7</td><td>28.6</td><td>33.4</td></tr><tr><td>GaussianVLM</td><td>46.2</td><td>25.3</td><td>32.2</td></tr><tr><td>P3-VLM (b)</td><td>52.4</td><td>36.4</td><td>39.9</td></tr></table>

Table 5: P<sup>3</sup>-VLM (p) robustness to query point perturbation. We train on object centers (query offset x=0) and evaluate with query points drawn uniformly at random from a ball of radius x× the box extent around the center, or uniformly from the object surface. We observe graceful degradation even at extreme offsets.
<table><tr><td>Query|</td><td colspan="2">Nr3D</td><td colspan="2">ScanRefer</td></tr><tr><td>Offset |</td><td>Acc</td><td>Sim</td><td>Acc</td><td>Sim</td></tr><tr><td>0</td><td>52.1</td><td>46.0</td><td>56.1</td><td>54.2</td></tr><tr><td>0.1 0.3</td><td>50.6 (-1.5) 49.9 (-2.2)</td><td>45.6 (-0.4) 45.7 (-0.3)</td><td>56.5 (+0.4)</td><td>54.0 (-0.2)</td></tr><tr><td>0.6</td><td></td><td></td><td>56.1 (0.0)</td><td>54.0 (-0.2)</td></tr><tr><td>0.9</td><td>47.4 (-4.7)</td><td>45.5 (-0.5)</td><td>54.1 (-2.0)</td><td>53.6 (-0.6)</td></tr><tr><td></td><td>46.6 (-5.5)</td><td>45.1 (-0.9)</td><td>52.8 (-3.3)</td><td>53.1 (-1.1)</td></tr><tr><td>Surf.</td><td>50.7 (-1.4)</td><td>45.4 (-0.6)</td><td>56.9 (+0.8)</td><td>54.1 (-0.1)</td></tr></table>

## 5.3 Out-of-Distribution Generalization

Robustness to Bounding Box Perturbation. The bounding boxes provide a categorical prior that often does not hold in noisy or out-of-distribution data, e.g., spaces with larger cabinets or larger windows. We evaluate two failure regimes. (1) Non-tight bounding boxes, produced by imperfect detectors or coarse annotations, dilute the geometric signal that the model has come to depend on. (2) Noisy or partial reconstructions introduce geometric artifacts that corrupt box estimates further.

To simulate regime (1), we shrink each GT box by a per-class scale factor $s _ { c } = \bar { \iota } _ { c } ^ { 1 / 3 }$ , where $\bar { \iota } _ { c }$ is the mean IoU of Vote2Cap-DETR [9] detections on class c, so that the resulting box volume matches the average detection quality of a state-of-the-art detector. Vote2Cap-DETR is chosen as it is adopted within the LL3DA baseline, keeping the simulation as faithful as possible to the conditions these models were designed for. Regime (2) is simulated symmetrically: boxes are expanded by the same per-class factor $1 / s _ { c } ,$ reflecting the tendency of detectors to produce loose predictions under reconstruction noise. Full per-class IoU values and implementation details are provided in Supp. Sec. A.4. As shown in Tab. 4, each of these conditions induces a severe performance drop – not because the visual scene has changed dramatically, but because the leakage channel the model relied on has been disrupted. Notably, P<sup>3</sup>-VLM (b) remains the most stable.

Robustness to Query Point Perturbation. To test the spatial stability of $\mathrm { P ^ { 3 } - V L M \left( p \right) }$ , trained on exact center coordinates, we evaluate it on random query points: for offset ratio $x \in \{ 0 . 1 , 0 . 3 , 0 . 6 , 0 . 9 \}$ , the query is drawn uniformly at random from a ball of radius x times the box extent, centered at the object center. $\mathrm { A t \it { \ x } = } 0 . 9$ the sampling region covers nearly the full object volume. As shown in Tab. 5, performance degrades gracefully with perturbation distance, dropping by only 5.5 accuracy points on Nr3D and 3.3 on ScanRefer at the most extreme setting (0.9× extent). We further query points sampled uniformly from the object surface, which changes performance only modestly, confirming that P<sup>3</sup>-VLM (p) does not depend on the privileged center point.

Table 6: ScanQA Results. GaussianVLM<sup>Q</sup>: [19] with Qwen3-VL-4B LLM, retrained.
<table><tr><td colspan="4">ScanQA</td></tr><tr><td>Model</td><td>EM-R</td><td>M</td><td>R</td></tr><tr><td>ScanQA [3]</td><td>1</td><td>13.14</td><td>33.33</td></tr><tr><td>3D-VLP[26]</td><td>1</td><td>13.53</td><td>34.51</td></tr><tr><td>VoteNet+MCAN [51]</td><td>1</td><td>11.4</td><td>29.8</td></tr><tr><td>LSceneLLM [56]</td><td>1</td><td>17.95</td><td>1</td></tr><tr><td>LL3DA [8]</td><td>30.3</td><td>22.8</td><td>34.7</td></tr><tr><td>GaussianVLM [19]</td><td>30.7</td><td>22.9</td><td>34.8</td></tr><tr><td>GaussianVLMQ</td><td>30.9</td><td>21.1</td><td>35.3</td></tr><tr><td>P3-VLM (p)</td><td>31.9</td><td>22.6</td><td>35.3</td></tr><tr><td>P3-VLM (b)</td><td>30.5</td><td>21.2</td><td>33.9</td></tr></table>

## 5.4 Ungrounded Scene Understanding

We further examine the generalist capabilities of P<sup>3</sup>-VLM on three ungrounded scene understanding benchmarks: ScanQA, SQA3D, and GaussianVLM out-of-distribution benchmark (GaussianVLM-OOD) [19] (Tables 6, 7, 8). The latter probes VLM generalization on an object counting task on out-of-distribution ScanNet++ scenes [50], measured via exact match accuracy (EM). P<sup>3</sup>-VLM (p) achieves state-of-the-art performance across all three benchmarks, surpassing the state-of-the-art GaussianVLM on most metrics despite being trained with the strictly harder point-based prompting protocol. On the ungrounded scene-level OOD benchmark, P<sup>3</sup>-VLM (p) reaches 17.31 EM versus 9.3 for GaussianVLM, with $\mathrm { P ^ { 3 } - V L M }$ (b) scoring 15.42. The +1.89 EM advantage of (p) over (b) on this benchmark indicates that removing bounding box supervision improves out-of-distribution robustness, consistent with our hypothesis that point-based training discourages reliance on box-derived shortcuts and encourages stronger visual representations.

![](images/3f785eb3956d200cf2060fc044ef5e58960cd8920fcffe20b52f3024020bf0fc.jpg)  
(a) ScanQA  
(b) Nr3D  
Figure 4: Qualitative results on ScanQA and Nr3D. (a) P<sup>3</sup>-VLM can correctly answer scan questions, sometimes even with answers not captured by the benchmark ground truth. (b) Through the combination of task, context, and location pathway, on Nr3D, P<sup>3</sup>-VLM identifies the correct object class, surrounding context, and spatial relationships. The performance remains robust even under perturbations (offset factor x = 0.9).

Table 7: SQA3D Results.  
GaussianVLM<sup>Q</sup>: [19] with Qwen3-VL-4B LLM, retrained.
<table><tr><td colspan="6">SQA3D</td></tr><tr><td>Model</td><td>EM-R</td><td>B4</td><td>C</td><td>M</td><td>R</td></tr><tr><td>LL3DA [8]</td><td>43.3</td><td>1.0</td><td>107.2</td><td>21.4</td><td>42.14</td></tr><tr><td>LEO [24]</td><td>50.9</td><td>9.4</td><td>124.7</td><td>25.5</td><td>48.4</td></tr><tr><td>GaussianVLM [19]</td><td>51.4</td><td>17.1</td><td>129.6</td><td>26.4</td><td>50.2</td></tr><tr><td>GaussianVLMQ</td><td>51.9</td><td>18.9</td><td>125.4</td><td>26.7</td><td>50.0</td></tr><tr><td>P3-VLM (p)</td><td>51.8</td><td>30.6</td><td>130.4</td><td>26.6</td><td>50.3</td></tr><tr><td>P3-VLM (b)</td><td>51.9</td><td>19.5</td><td>130.3</td><td>26.4</td><td>50.5</td></tr></table>

Table 8: Out-of-distribution results. Zero-shot evaluation on scenes from unseen ScanNet++ [50].
<table><tr><td colspan="2">GaussianVLM-OOD</td></tr><tr><td>Model</td><td>EM</td></tr><tr><td>LL3DA [8] LEO [24]</td><td>1.5 7.9</td></tr><tr><td>GaussianVLM [19]</td><td>9.3</td></tr><tr><td>P3-VLM (p)</td><td>17.31</td></tr><tr><td>P3-VLM (b)</td><td>15.42</td></tr></table>

## 5.5 Ablation Studies

We evaluate the contribution of different components within $\mathrm { P ^ { 3 } - V L M }$ through a series of ablation studies considering our tri-pathway design (Tab. 9) with ablations on the design of separate components provided in Supp. Sec. C. Our ablations cover location-grounded and ungrounded tasks with both object-level and scene-level understanding focus.

We show the necessity of understanding 3D scenes at three levels of granularity by ablating the three pathways of $\mathrm { P ^ { 3 } - V L M }$ . Removing any pathway causes consistent drops in task performance. Notably, the location pathway proves an important component enabling scene-level tasks (SQA3D), although it is used only with location-based queries. As we observe, it allows the other two pathways to focus on scene-level understanding rather than generalist capabilities.

Matched baseline. To fully isolate the contribution of our visual processing, we construct a matched baseline that shares the 3D encoder and LLM with P<sup>3</sup>-VLM, but uses more simplistic visual features: the query point is represented by the raw feature of its nearest voxel, and scene information is provided only via the context tokens, i.e., without our location MLP and without the task pathway. The matched

Table 9: Design ablation studies on P<sup>3</sup>-VLM (p). Top: Each of the pathways is removed independently. Bottom: A matched baseline that isolates visual processing innovations by employing a multi-pathway variant with minimal structural complexity – utilizing only nearest-voxel features at the query point and global context tokens.
<table><tr><td rowspan="2"></td><td colspan="2">Nr3D</td><td colspan="3">ScanQA</td><td colspan="5">SQA3D</td></tr><tr><td>Acc</td><td>Sim</td><td>EM-R</td><td>M</td><td>R</td><td>EM-R</td><td>B4</td><td>C</td><td>M</td><td>R</td></tr><tr><td>w/o location pathway</td><td>49.3</td><td>43.9</td><td>30.4</td><td>24.2</td><td>34.1</td><td>45.6</td><td>0</td><td>114.6</td><td>23.4</td><td>44.3</td></tr><tr><td>w/o task pathway</td><td>50.2</td><td>46.0</td><td>30.1</td><td>24.2</td><td>19.2</td><td>47.9</td><td>22.5</td><td>119.0</td><td>24.1</td><td>46.2</td></tr><tr><td>w/o context pathway</td><td>50.2</td><td>45.2</td><td>22.7</td><td>14.7</td><td>23.3</td><td>49.4</td><td>34.8</td><td>123.2</td><td>25.2</td><td>47.8</td></tr><tr><td>Matched baseline</td><td>39.8</td><td>41.9</td><td>31.3</td><td>21.7</td><td>33.9</td><td>49.6</td><td>20.2</td><td>124.2</td><td>25.3</td><td>48.1</td></tr><tr><td>Full model</td><td>52.1</td><td>46.0</td><td>31.9</td><td>22.9</td><td>34.8</td><td>51.8</td><td>30.58</td><td>130.4</td><td>26.59</td><td>50.32</td></tr></table>

baseline falls far behind P<sup>3</sup>-VLM on point-based grounding (Nr3D: 39.8% vs. 52.1%, ScanRefer: 44.2% vs. 56.1% Acc), showing that the pathway choices and their design are well-motivated.

## 6 Conclusions

In this work, we highlight the problem of semantic leakage in current 3D VLMs, causing these models to neglect visual information. We propose a new point-based grounding prompting protocol and show that other methods struggle under this new setting. To address this, we present P<sup>3</sup>- VLM, a novel 3D VLM based on a tri-pathway design. We demonstrate the proposed point-based grounding approach with our architecture leads to improved visual understanding, demonstrated through improved performance, generalization, and robustness on multiple 3D tasks and benchmarks.

Limitations First, P<sup>3</sup>-VLM requires 3D GS representations as input, introducing an extra reconstruction step that adds complexity; however, this step has been performed at scale in prior works [30, 32] and industrial applications [38, 14], and is shown to yield richer visual features [19]. Second, our evaluation is primarily focused on perception-oriented benchmarks, with limited coverage of language-intensive or dialogue-driven tasks. Extending P<sup>3</sup>-VLM to more complex multimodal reasoning settings remains an important direction for future work. Third, our approach relies on voxelized visual features, currently obtained via the SceneSplat encoder. Evaluating how well the method generalizes to alternative 3D visual encoders is an interesting avenue for further investigation.

Impact We envision this work as a step toward enabling real-world applications such as high-level planning for navigation and visual assistance systems. Additionally, we hope it draws attention to the issue of information leakage in 3D grounding, encouraging the development of more robust and general-purpose 3D vision-language models.

## Acknowledgments

This research was partially funded by the Ministry of Education and Science of Bulgaria (support for INSAIT, part of the Bulgarian National Roadmap for Research Infrastructure).

## References

[1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas J. Guibas. ReferIt3D: Neural listeners for fine-grained 3d object identification in real-world scenes. European Conference on Computer Vision (ECCV), 2020.

[2] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, Jakob Engel, Edward Miller, Richard Newcombe, and Vasileios Balntas. Scenescript: Reconstructing scenes with an autoregressive structured language model. In European Conference on Computer Vision (ECCV), 2024.

[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005.

[6] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. European Conference on Computer Vision (ECCV), 2020.

[7] Haijier Chen, Bo Xu, Shoujian Zhang, Haoze Liu, Jiaxuan Lin, and Jingrong Wang. Vid-llm: A compact video-based 3d multimodal llm with reconstruction-reasoning synergy. arXiv preprint arXiv:2509.24385, 2025.

[8] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, et al. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[9] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, et al. End-to-end 3d dense captioning with vote2cap-detr. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[10] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, et al. Grounded 3d-llm with referent tokens. arXiv:2405.10370, 2024.

[11] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang. Scan2cap: Context-aware dense captioning in rgb-d scans. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[12] Lin Chin-Yew. Rouge: A package for automatic evaluation of summaries. Proceedings ofthe Workshop on Text Summarization Branches Out, 2004, 2004.

[13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[14] Eric Cornwell, Dario Macangano, Stanford Lee, and Daniel Zilberman. Guidance for open source 3d reconstruction toolbox for gaussian splats on aws. See https://aws-solutions-librarysamples.github.io/compute/open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws.html (accessed 2026-05-05), 2024.

[15] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[16] Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys, et al. SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[17] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Shijie Zhou, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction, 2025.

[18] Mei Guofeng, Lin Wei, Riz Luigi, Wu Yujiao, Poiesi Fabio, and Wang Yiming. Perla: Perceptive 3d language assistant. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[19] Anna-Maria Halacheva, Jan-Nico Zaech, Xi Wang, Danda Pani Paudel, and Luc Van Gool. Gaussianvlm: Scene-centric 3d vision-language models using language-aligned gaussian splats for embodied reasoning and beyond. IEEE Robotics Autom. Lett., 10(12):12588–12595, 2025.

[20] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, et al. 3d-llm: Injecting the 3d world into large language models. NeurIPS, 2023.

[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, et al. Lora: Low-rank adaptation of large language models. ICLR, 2022.

[22] Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, and Jiangmiao Pang. G<sup>2</sup>vlm: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688, 2025.

[23] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. NeurIPS, 2024.

[24] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, et al. An embodied generalist agent in 3d world. ICML, 2024.

[25] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. Sceneverse: Scaling 3d vision-language learning for grounded scene understanding. In European Conference on Computer Vision (ECCV), 2024.

[26] Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, and Yinjie Lei. Context-aware alignment and mutual masking for 3d-language pre-training. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10984–10994, 2023.

[27] Maksim Kolodiazhnyi, Anna Vorontsova, Matvey Skripkin, Danila Rukhovich, and Anton Konushin. Unidet3d: Multi-dataset indoor 3d object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 4365–4373, 2025.

[28] Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22225–22233, 2025.

[29] Mingsheng Li, Xin Chen, Chi Zhang, Sijin Chen, Hongyuan Zhu, et al. M3dbench: Towards omni 3d assistant with interleaved multi-modal instructions. European Conference on Computer Vision (ECCV), 2025.

[30] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al. Scenesplat: Gaussian splatting-based scene understanding with visionlanguage pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[31] Xiongkun Linghu, Jiangyong Huang, Xuesong Niu, Xiaojian Ma, et al. Multi-modal situated reasoning in 3d scenes. NeurIPS, 2024.

[32] Mengjiao Ma, Qi Ma, Yue Li, Jiahuan Cheng, Runyi Yang, Bin Ren, Nikola Popovic, Mingqiang Wei, Nicu Sebe, Luc Van Gool, Theo Gevers, Martin R. Oswald, and Danda Pani Paudel. Scenesplat++: A large dataset and comprehensive benchmark for language gaussian splatting. In Proceedings ofthe 39th Conference on Neural Information Processing Systems (NeurIPS 2025), 2025.

[33] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, et al. Sqa3d: Situated question answering in 3d scenes. ICLR, 2023.

[34] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. In Advances in Neural Information Processing Systems, 2025.

[35] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. ACL, 2002.

[36] Songyou Peng, Kyle Genova, Chiyu "Max" Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser. Openscene: 3d scene understanding with open vocabularies. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[37] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

[38] Pix4D. Transform your work with precise 3D scans and AR with PIX4Dcatch. https://www.pix4d.com/product/pix4dcatch/ (accessed 2026-05-05), 2025.

[39] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. GPT4scene: Understand 3d scenes from videos with vision-language models. In The Fourteenth International Conference on Learning Representations, 2026.

[40] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP, 2019.

[41] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8216–8223. IEEE, 2023.

[42] Yichao Shen, Zigang Geng, Yuhui Yuan, Yutong Lin, Ze Liu, Chunyu Wang, Han Hu, Nanning Zheng, and Baining Guo. V-DETR: DETR with vertex relative position encoding for 3d object detection. In The Twelfth International Conference on Learning Representations, 2024.

[43] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[44] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[45] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9275–9286, October 2025.

[46] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, Xihui Liu, Cewu Lu, Dahua Lin, and Jiangmiao Pang. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19757–19767, June 2024.

[47] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023.

[48] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Self-supervised learning of reliable point representations. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22193–22204. IEEE, 2025.

[49] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10632–10643, June 2025.

[50] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[51] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[52] Yuanwen Yue, Theodora Kontogianni, Konrad Schindler, and Francis Engelmann. Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[53] Tatiana Zemskova and Dmitry Yudin. 3dgraphllm: Combining semantic graphs and large language models for 3d scene understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8885–8895, October 2025.

[54] Yue Zhang, Zhiyang Xu, Ying Shen, Parisa Kordjamshidi, and Lifu Huang. SPARTUN3d: Situated spatial understanding of 3d world in large language model. In The Thirteenth International Conference on Learning Representations, 2025.

[55] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8995–9006, June 2025.

[56] Hongyan Zhi, Peihao Chen, Junyan Li, Shuailei Ma, Xinyu Sun, Tianhang Xiang, Yinjie Lei, Mingkui Tan, and Chuang Gan. Lscenellm: Enhancing large 3d scene understanding using adaptive visual preferences. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3761–3771, 2025.

[57] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, et al. 3d-vista: Pre-trained transformer for 3d vision and text alignment. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[58] Ziyu Zhu, Zhuofan Zhang, Xiaojian Ma, Xuesong Niu, et al. Unifying 3d vision-language understanding via promptable queries. European Conference on Computer Vision (ECCV), 2025.

## A Experimental Details

## A.1 Experimental Details: Point-Based Grounding Adaptation

This section provides full experimental details for the results reported in Sec. 3.2.

Blind LLM Experiments. We fine-tune Qwen3-VL-4B-Instruct [4] on object classification within the Nr3D validation set under two prompt conditions: bounding box and point location. In both cases, the location query is provided as a text string with coordinates rounded to two decimal places. Bounding boxes are encoded via center coordinates, yaw angle and scale, as $\langle c _ { x } , c _ { y } , c _ { z } , y a w , w , h , l \rangle$ without explicitly naming the format in the prompt. Point prompts contain only the 3D coordinate $\langle x , y , z \rangle$

Fine-tuning is performed with LoRA using rank $r { = } 1 6 .$ , scaling factor $\alpha { = } 3 2$ , dropout 0.05, and a learning rate of $3 \times 1 0 ^ { - 5 }$ . The LoRA adapters are applied to the query, key, value, and output projection layers (q\_proj, k\_proj, v\_proj, o\_proj). As shown in Fig. 5, validation accuracy saturates between epochs 6–8 and overfitting emerges thereafter.

Scene-Centric VLMs. Based on the convergence behavior observed for the blind LLM, we finetune all scene-centric VLMs for 6 epochs. To preserve learned visual representations, only the LLM component of each model is fine-tuned, using the same LoRA configuration described above. Fine-tuning is performed exclusively on Nr3D. We evaluate after every epoch and report results from the checkpoint with the highest validation accuracy. Point prompt format is consistent with the blind LLM and P<sup>3</sup>-VLM (p) experiments $( \langle x , y , z \rangle )$ ). For bounding box prompting evaluation, we adhere to the bounding box formats used by the specific baselines.

## A.2 N-gram Metrics Conceal Failures in Spatial Understanding

Standard 3D scene understanding benchmarks such as ScanRefer and Nr3D evaluate dense captioning models using n-gram overlap metrics – CIDEr, METEOR, ROUGE – gated by an IoU threshold of 0.25 or 0.50. While these metrics are practical for large-scale evaluation, they measure linguistic surface similarity rather than semantic correctness, and in particular are insensitive to whether the model has correctly identified the class of the referred object. A response that fluently describes the wrong object category can achieve a high n-gram score if it shares common words with the reference caption [19].

To quantify this effect, we compare the correlation of standard n-gram evaluation with class accuracy: the fraction of generated captions from which the correct object category can be extracted. This metric directly measures whether the model has understood what it is looking at, independently of linguistic fluency. As shown in Tab. 10, models that achieve competitive n-gram scores exhibit strikingly low class accuracy – in several cases falling below a blind LLM that receives no visual input whatsoever. These results indicate that n-gram metrics systematically overestimate the spatial understanding of current 3D VLMs, and motivate the use of class accuracy as a more diagnostic evaluation criterion.

![](images/72f80e499223d2bbe9be99b0ea5f4f60344d131da84e831e688aeb6e1f256a21.jpg)  
Figure 5: Validation accuracy of the blind LLM during fine-tuning under bounding-box and point-based prompting. Performance saturates between epochs 6–8, with overfitting emerging afterwards. This curve motivates the 6-epoch fine-tuning budget used for all VLM experiments.

Table 10: Different evaluation metrics on captioning on the Nr3D dataset. It is evident that high n-gram based metrics do not signal correct object identification.
<table><tr><td>Model</td><td>Sim</td><td>M</td><td>R</td><td>Class Acc</td></tr><tr><td>Blind LLM</td><td>34.8</td><td>17.9</td><td>21.3</td><td>36.4</td></tr><tr><td>LL3DA</td><td>48.1</td><td>5.8</td><td>9.9</td><td>47.7</td></tr><tr><td>GaussianVLM</td><td>48.2</td><td>20.8</td><td>48.2</td><td>46.2</td></tr><tr><td>P3-VLM</td><td>46.0</td><td>18.6</td><td>16.4</td><td>52.1</td></tr></table>

## A.3 LLM-Based Class Accuracy Evaluator

Class accuracy is computed by extracting the primary object class from each generated caption with Gemini 2.5 Flash [13] and comparing it to the ground-truth class. Captions are processed in numbered batches with the following prompt:

You are an object classification assistant. You will receive a numbered list of short scene descriptions. For each description, identify the single primary object being described. Reply ONLY with a JSON array of strings – one object class per entry, in the same order. Use simple, lowercase class names (e.g. "chair", "trash can", "table", "sofa"). If the caption is empty or doesn’t describe a recognizable object, reply with "none" for that entry. No explanations, no extra keys, no markdownfences, just the raw JSON array.

Tab. 11 shows representative extractions. We further carried out a human quality control on 5% of the validation predictions and found no extraction errors, owing to the simple nature of the task: the generated captions name the described object explicitly.  
Table 11: Examples of LLM-based class extraction on ScanRefer and Nr3D captions generated by LL3DA 3D VLM.
<table><tr><td></td><td>Generated caption</td><td>Extracted class</td><td>GT class</td></tr><tr><td>ScanRefer</td><td>sos this is a white kitchen cabinet . it is to the left of the oven . eos</td><td>kitchen cabinet</td><td>kitchen cabinet</td></tr><tr><td>ScanRefer</td><td>sos there is a rectangular soap dispenser . it is on the wall above the sink . eos</td><td>soap dispenser</td><td>soap dispenser</td></tr><tr><td>Nr3D</td><td>sos the stool is in the corner of the room . it is to the right of the whiteboard . eos</td><td>stool</td><td>stool</td></tr><tr><td>Nr3D</td><td>sos the plant is on the floor . it is to the right of the door . eos</td><td>plant</td><td>plant</td></tr></table>

## A.4 Out-of-Distribution Generalization

Robustness to Bounding Box Perturbation. To obtain our bounding box perturbation scales, we firstly obtain bounding boxes for every ScanRefer and Nr3D instance via the Vote2Cap-DETR object detector. Here we utilize only the validation split of the datasets as the object detector itself has been trained on the ScanNet dataset train split. The scales are derived based on mean IoU between the ground truth boxes and the predicted ones following our approach introduced in Sec. 5.2. This results to the following scales given in Fig. 6.

## A.5 Implementation Details

We train P<sup>3</sup>-VLM using AdamW with a learning rate of $3 \times 1 0 ^ { - 5 } , \beta = ( 0 . 9 , 0 . 9 9 9 )$ , and weight decay 0.05, with a linear warmup 200 steps followed by cosine decay. Training uses a batch size of

![](images/4b56ce165e152e3d1ebfe5927134386f0a5a062663d7d475afd9d2125c35afaa.jpg)  
Figure 6: Mean IoU per object class from ScanRefer and Nr3D annotations, validation split.

16 with 4 gradient accumulation steps (effective batch size 64) and gradient clipping at norm 5.0. The language backbone is Qwen3-VL-4B-Instruct, fine-tuned via LoRA (rank $1 6 , \alpha = 3 2$ , dropout 0.05) applied to all attention projection matrices. The maximum input and output context lengths are set to 2048 and 1024 tokens, respectively. Hyperparameter choices are detailed in Tab. 13 and are selected in close adherence to the GaussianVLM baseline [19]. The training objective is the standard autoregressive objective:

$$
\mathcal { L } = - \sum _ { t } \log p _ { \theta } \left( y _ { t } \mid y _ { < t } , T _ { l o c } , T _ { t a s k } , T _ { c t x } , L \right)\tag{4}
$$

, where $y _ { t }$ denotes the t-th token of the target response $y = ( y _ { 1 } , \dots , y _ { T } ) , y _ { < t }$ the tokens generated earlier, and θ the set of trainable parameters. The loss minimizes the negative log-likelihood of the target sequence conditioned on the language prompt L and all three visual pathway outputs: the location token $\mathbf { T } _ { \mathrm { l o c } } ,$ the compressed task tokens $\mathbf { T } _ { \mathrm { t a s k } } ^ { - }$ , and the context summary tokens $\mathbf { \bar { T } } _ { \mathrm { c t x } }$

## A.6 Query point robustness

We carry out robustness experiments of P<sup>3</sup>-VLM to the query point offset from the center of the object. We notice that $\mathrm { \dot { P } ^ { 3 } - V L M }$ is very stable to even the extreme condition where the query-point lies near the edge of the object. We plot the mean accuracy across the different samples with error bar indicating 1σ deviation, shown in Fig. 7.

![](images/f478699ac5ca704105c7478aeee12fc01c5d0237bcc642bcf427a6b4147350e2.jpg)  
Figure 7: $\mathrm { P ^ { 3 } - V L M }$ on Nr3D benchmark

## A.7 Training with Surface-Point Grounding

We acknowledge the center point being a canonical point: its height is a stable, category-specific statistic. To escape this prior, we also analyze prompting with random surface points instead. A point sampled uniformly on the object surface may lie anywhere within the object’s vertical extent, so its height no longer encodes a consistent per-category value. We retrain P<sup>3</sup>-VLM with random surface points as the grounding signal during training. As shown in Tab. 12, this yields a further, substantial improvement in grounding accuracy (Nr3D: 58.8% vs. 52.1% for center training; ScanRefer: 64.3% vs. 56.1%), while the caption similarity remains comparable. A surface-trained model evaluated on center points performs worse (50.1% on Nr3D), as the center lies outside of its training distribution.

Table 12: Center- vs. surface-point grounding on Nr3D and ScanRefer. c: object center, sp: random surface point. Training on surface points substantially improves class accuracy.
<table><tr><td colspan="3"></td><td colspan="2">Nr3D</td><td colspan="2">ScanRefer</td></tr><tr><td>Model</td><td>Train</td><td>Eval</td><td>Acc</td><td>Sim</td><td>Acc</td><td>Sim</td></tr><tr><td>LL3DA</td><td>box</td><td>box</td><td>47.7</td><td>48.1</td><td>48.0</td><td>55.9</td></tr><tr><td>GaussianVLM</td><td>box</td><td>box</td><td>46.2</td><td>48.2</td><td>50.4</td><td>59.1</td></tr><tr><td>P3-VLM (b)</td><td>box</td><td>box</td><td>52.4</td><td>46.3</td><td>58.4</td><td>52.5</td></tr><tr><td>P3-VLM (p)</td><td>C</td><td>C</td><td>52.1</td><td>46.0</td><td>56.1</td><td>54.2</td></tr><tr><td>P3-VLM (p)</td><td>C</td><td>sp</td><td>50.7</td><td>45.4</td><td>56.9</td><td>54.1</td></tr><tr><td>P3-VLM (p)</td><td>sp</td><td>C</td><td>50.1</td><td>42.7</td><td>50.4</td><td>50.2</td></tr><tr><td>P3-VLM (p)</td><td>sp</td><td>sp</td><td>58.8</td><td>47.2</td><td>64.3</td><td>54.7</td></tr></table>

Per-class analysis. To understand where the gain originates, we relate the per-class change in accuracy (center- to surface-point training) to the physical object size, computed from the ground-truth 3D bounding boxes. Medium-to-large furniture (tables, desks, cabinets, shelves, bookshelves, seating) improves consistently, with an average gain of 21.3 accuracy points. Small or thin objects (pillows, boxes, monitors, keyboards, cups) improve much less, by 6.2 points on average; this difference is statistically significant (Mann-Whitney U test, $p = 0 . 0 1 4 )$ . We attribute this behavior to the feature pooling at the query point. Prompting a large object by its center forces the visual features to be pooled from deep inside the object’s interior, or even from empty space, e.g., below a table top or inside a cabinet, yielding uninformative features. Surface points, by contrast, anchor the feature pooling directly to regions rich in visible geometry and texture, which substantially improves visual grounding for larger structures.

![](images/d3607ea38047f7d54d8ba3cdfe414d2979da0ee301bb847e15f98679ab62a431.jpg)  
Figure 8: Qualitative results on situated question answering (SQA3D benchmark) and out-ofdistribution generalization (GaussianVLM-OOD benchmark). Our model achieves better performance on both benchmarks, showing visual understanding capabilities both on in-domain ScanNet scenes, and on out-of-domain ScanNet++ scenes.

## B Qualitative Examples

We provide qualitative results showcasing the capabilities of P<sup>3</sup>-VLM in Fig. 4, 8 across the tasks of 3D grounded captioning (Nr3D benchmark, validation split), general question-anwering (ScanQA benchmark, validation split), situated question answering (SQA3D benchmark, test split) and out-of-distribution generalization (GaussianVLM-OOD benchmark, test split). We observe visual understanding on in-domain settings within ScanNet input scenes, and out-of-domain on ScanNet++.

Table 13: Training hyperparameters for P<sup>3</sup>-VLM.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9, 0.999)</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>LR schedule</td><td>Linear warmup + cosine decay</td></tr><tr><td>Warmup steps</td><td>200</td></tr><tr><td>Gradient clip (norm)</td><td>5.0</td></tr><tr><td colspan="2">Batching 16</td></tr><tr><td colspan="2">Batch size (per GPU)</td></tr><tr><td colspan="2">Gradient accumulation steps</td></tr><tr><td colspan="2">Effective batch size</td></tr><tr><td colspan="2">Language model</td></tr><tr><td colspan="2">Backbone Qwen3-VL-4B-Instruct</td></tr><tr><td>Max input length</td><td>2048</td></tr><tr><td>Max output length</td><td>512</td></tr><tr><td>LoRA</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td>Rank (r)</td><td>16</td></tr><tr><td>Alpha (α)</td><td>32</td></tr><tr><td>Dropout</td><td>0.05</td></tr><tr><td>Target modules</td><td>q, k, v, o_proj</td></tr></table>

Table 14: Location pathway and Sparsifier design ablation studies. We ablate the point-based variant, P<sup>3</sup>-VLM (p). Loc. Design: raw vs. processed features and FiLM vs. MLP extraction. Pos. Enc.: positional encoding strategy in visual sparsification step.
<table><tr><td rowspan="2"></td><td colspan="2">Nr3D</td><td colspan="3">ScanQA</td><td colspan="5">SQA3D</td></tr><tr><td>Acc</td><td>Sim</td><td>EM-R</td><td>M</td><td>R</td><td>EM-R</td><td>B4</td><td>C</td><td>M</td><td>R</td></tr><tr><td colspan="10">Location Sparsifier Design</td></tr><tr><td>Processed + MLP Raw + FiLM</td><td>42.3</td><td>47.1</td><td>一</td><td>-</td><td></td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>48.5</td><td>46.4 Sparsifier Design: Positional Encoding</td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>None Learned absolute</td><td>47.9</td><td>43.7 46.0</td><td>29.3</td><td>18.8</td><td>31.1</td><td>46.6</td><td>28.7 21.6</td><td>114.5</td><td>23.4</td><td>44.4</td></tr><tr><td>Full model</td><td>49.4 52.1</td><td>46.0</td><td>26.9 31.9</td><td>20.9 22.9</td><td>32.1 34.8</td><td>45.3 51.8</td><td>30.58</td><td>113.8 130.4</td><td>23.3 26.59</td><td>43.8 50.32</td></tr></table>

## C Further Ablations

Complementary to the tri-pathway design ablation provided in Sec. 5.5, we also evaluate specific design decisions within the location pathway and sparsifier components of $\mathrm { P ^ { 3 } - V L M }$

Location Pathway Design. For the location pathway we consider only the location-based captioning task, as the other tasks do not utilize the pathway. First, we ablate whether scene-context information improves local information extraction. The results demonstrate better location understanding on raw voxel features (F) compared to utilization of already sparsified and refined features (S).We attribute this to the sparsifier smoothing voxel features through spatial self-attention, which reduces their locality – precisely the property the location pathway relies on. We also ablate the extraction strategy comparing our MLP-based approach to FiLM location conditioning, with exact details on the FiLM alternative provided in the Supplementary. Results show that processing raw voxel features (F) with an MLP strategy delivers best results.

Sparsification of Visual Tokens. We also ablate the role of positional encodings (PE) for increasing scene-level awareness within the sparsification process. We evaluate the sparsifier module with 3D RoPE PE, learned absolute PE added to voxel features, and no PE. 3D RoPE yields the strongest performance, confirming that explicit spatial structure in the global features is critical.

## D P<sup>3</sup>-VLM Architecture Details

## D.1 Location Pathway Designs.

We investigated two different location pathway designs – FiLM- and MLP-based, Both first identify the nearest voxel to the target location, retrieving its feature vector as a local scene descriptor. The first strategy, FiLM-based extraction, conditions this voxel feature on the queried location via Feature-wise Linear Modulation [37]: we use a shallow location encoder MLP (2 linear layers) to map the 3D point coordinates p to affine parameters $( \gamma , \beta ) \in \mathbb { R } ^ { D } \times \mathbb { R } ^ { D }$ . Those are afterwards used to modulate the retrieved voxel feature $\mathbf { F } _ { i } \in \mathbb { R } ^ { D }$ as $\mathbf { T } _ { \mathrm { l o c } } = \sigma ( \gamma ) \odot \mathbf { F } _ { i } + \boldsymbol { \beta }$ , where $\sigma ( \cdot )$ denotes the sigmoid function, applied to γ to act as a soft selection mask over feature dimensions. We note that all notation is introduced in Sec. 4.

We note that the MLP-based variant is introduced in Sec. 4. The MLP structure we adopt is four-layer MLP with GELU activations.

## D.2 Sparsifier RoPE Details

The spatial refinement is handled by 4 self-attention layers where both queries and keys are rotated by 3D Rotary Position Embeddings. For head dimension $d _ { h }$ (divisible by 6), per-axis frequencies are:

$$
\theta _ { i , { \mathrm { a x i s } } } = { \frac { \mathrm { p o s } _ { \mathrm { a x i s } } } { 1 0 0 0 } } \cdot { \mathrm { b a s e } } ^ { - 2 i / d _ { h } } , \quad i = 0 , \dots , { \frac { d _ { h } } { 6 } } - 1 ,\tag{5}
$$

with base = 10,000.

## D.3 Comparison to the GaussianVLM Architecture

P<sup>3</sup>-VLM and GaussianVLM [19] are both scene-centric 3D GS VLMs, but are built around different priorities, from which their architectural differences follow. GaussianVLM is semantics-oriented: it fuses language into the 3D backbone itself (per-Gaussian SigLIP2 features), so every downstream token is semantically grounded before sparsification. This favors scene-level tasks and semantic VQA, where broad semantic understanding matters more than precise spatial extent. P<sup>3</sup>-VLM is geometry-oriented, designed to strengthen grounding, which leads to the following design decisions:

• Context pathway. A dedicated, unconditional pathway explicitly recovers global scene semantics; GaussianVLM has no counterpart.

• Task pathway. In P<sup>3</sup>-VLM, voxel tokens query the text, and the result is applied as a residual correction to the visual features before compression, i.e., language reshapes geometry. In GaussianVLM, text tokens query and directly select visual tokens, i.e., language selects semantics.

• Location pathway. P<sup>3</sup>-VLM derives the local token with an MLP over the concatenated geometric and semantic features at the queried point, explicitly learning location understanding jointly with content. GaussianVLM retrieves the per-Gaussian semantic feature nearest to the query point, without a comparable geometric grounding mechanism.

• Positional information. P<sup>3</sup>-VLM fuses positional information early via 3D RoPE, reinforcing geometry throughout the visual processing; GaussianVLM adds position as a final Fourier embedding, secondary to semantics.

## D.4 Point Cloud Encoder Variant

To assess whether P<sup>3</sup>-VLM generalizes beyond 3D GS inputs, we replace the SceneSplat encoder with the self-supervised point cloud encoder Sonata [48] (P<sup>3</sup>-VLM-PC), keeping the tri-pathway design and training protocol unchanged. Results are shown in Tab. 15. Owing to Sonata’s large-scale pre-training, P<sup>3</sup>-VLM-PC achieves strong location-grounded captioning results under point prompts (Nr3D: 58.1%, ScanRefer: 63.3% Acc), outperforming the 3D GS variant. However, Sonata features are not language-aligned, unlike SceneSplat features, which results in lower performance on the VQA benchmarks. This shows that the tri-pathway design transfers to other 3D representations, with the choice of encoder trading off grounding against language-aligned scene understanding.

Table 15: P<sup>3</sup>-VLM (p) with a 3D GS (SceneSplat) vs. point cloud (Sonata) encoder on grounded captioning (Nr3D, ScanRefer) and VQA (ScanQA, SQA3D).
<table><tr><td rowspan="3">Encoder</td><td colspan="2">Nr3D</td><td colspan="2">ScanRefer</td><td colspan="3">ScanQA</td><td colspan="5">SQA3D</td></tr><tr><td>Acc</td><td>Sim</td><td>Acc</td><td>Sim</td><td>EM-R</td><td>M</td><td>R</td><td>EM-R</td><td>B4</td><td>C</td><td>M</td><td>R</td></tr><tr><td>SceneSplat (3D GS)</td><td>52.1</td><td>46.0</td><td>56.1</td><td>54.2</td><td>31.9</td><td>22.6</td><td>35.3</td><td>51.8</td><td>30.6</td><td>130.4</td><td>26.6</td><td>50.3</td></tr><tr><td>Sonata (PC)</td><td>58.1</td><td>47.3</td><td>63.3</td><td>54.8</td><td>30.5</td><td>22.8</td><td>33.9</td><td>48.3</td><td>16.8</td><td>126.9</td><td>23.8</td><td>48.8</td></tr></table>

## E Datasets Information

Point Cloud and 3D GS Representations. Our method uses 3D GS scene input, differing from the point cloud input consumed by other methods. To ensure maximal similarity in setup to the point cloud-based methods, we use 3D GS representations with Gaussian centers initialized at the point locations from the original point clouds ("Fixed XYZ" strategy). This guarantees the exact same geometrical properties under the two setups. Our 3D GS data is consistent to the one utilized within the GaussianVLM experiments and is obtained from the GaussianWorld dataset [30].

Dataset Licenses and Terms of Use. We report the licenses and terms of use for all datasets in Table 16 and adhere strictly to each. All datasets are publicly released. We draw attention to ScanNet [15] and ScanNet++ [50] being subject to custom terms of use, requiring separate inspection.

All other datasets are released under standard open-source licenses. The 3D GS representations of ScanNet and ScanNet++ scans are obtained from the respective subsets under GaussianWorld (SceneSplat-7K) and inherit their respective terms.

Table 16: Licenses and terms of use for the used datasets.
<table><tr><td>Dataset</td><td>License / Terms of Use</td></tr><tr><td>ScanNet [15]</td><td>ScanNet Terms of Use1</td></tr><tr><td>ScanRefer [6]</td><td>Creative Commons BY-NC-SA 3.0</td></tr><tr><td>ScanQA [3]</td><td>Apache 2.0</td></tr><tr><td>SQA3D [33]</td><td>Apache 2.0</td></tr><tr><td>Nr3D [1]</td><td>MIT</td></tr><tr><td>ScanNet++ [50]</td><td>ScanNet++ Terms of Use²</td></tr><tr><td>GaussianWorld (SceneSplat-7K) [30]</td><td>Same as ScanNet, ScanNet++</td></tr></table>

## F Broader Impact

Our work aims to make a step towards VLMs with strong visual understanding capabilities, showing strong robustness and out-of-distribution generalization. With this we invision real-world utilization of 3D VLMs for applications that require free-form, open-world, spatially grounded reasoning, such as robotics, assistive technologies, and AR/VR systems.

We also note the potential environmental implications of training VLM models such as ours. While P<sup>3</sup>-VLM is designed with sparsification modules and scene downsampling, its training still requires significant compute resources. With this in mind, we underscore the necessity of research into environmentally-friendly training approaches.