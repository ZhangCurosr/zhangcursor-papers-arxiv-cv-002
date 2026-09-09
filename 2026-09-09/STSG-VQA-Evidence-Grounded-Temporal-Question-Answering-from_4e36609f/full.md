# STSG-VQA: Evidence-Grounded Temporal Question Answering from Surgical Spatio-Temporal Scene Graphs

Jing Li and Duygu Sarikaya

Abstract— Despite recent advances in surgical visionlanguage models (VLMs), temporal reasoning remains limited because existing supervision is largely frame-centric. Frame-level scene graphs (SGs) have proven effective in providing structured representations of surgical environments but do not explicitly model the dynamics of surgical workflows. To explicitly model how surgical states evolve across time, we introduce a multi-level structured temporal supervision methodology that augments framelevel surgical SGs with object-level continuity, event-level interaction continuity, and procedure-level connectivity. We then execute temporal queries over the resulting spatiotemporal scene graphs (STSGs) to generate evidencegrounded question–answer pairs, which together form the STSG-VQA benchmark. Each question is linked to the temporal interval and STSG evidence used to derive its reference answer, enabling traceable verification. The benchmark contains 18,458 question–answer pairs across seven temporal categories. Fine-tuning Qwen3-VL-4B and Hulu-Med-4B with STSG-derived supervision improves questionlevel micro accuracy by 24.39 and 19.56 percentage points over their zero-shot baselines and by 16.50 and 14.25 points over static scene-graph supervision, respectively. These gains span all temporal categories, indicating that STSGderived supervision helps surgical VLMs reason over temporally grounded interactions rather than isolated frames. The code and dataset will be made publicly available upon acceptance.

Index Terms— Surgical Video Question Answering, Spatio-temporal Scene Graphs, Temporal Reasoning.

## I. INTRODUCTION

D <sup>RIVEN</sup> <sup>by</sup> <sup>recent</sup> <sup>breakthroughs</sup> <sup>in</sup> <sup>vision-language</sup> models (VLMs), surgical visual question answering (VQA) has achieved remarkable progress, demonstrating great promise for decision support, postoperative analysis, and surgical education. Surgical videos encompass rich and highly structured procedural information, characterized by transitions between surgical phases, the entry and exit of surgical instruments within the field of view, and the deformation of anatomies under traction. Fundamentally, clinically meaningful surgical actions are typically defined by the interactions among instruments, actions, and targets, formulated as an action triplet ⟨instrument, verb, target⟩ [1]. Therefore, a practical surgical VLM should not be confined to single-frame content recognition. Rather, it requires robust temporal understanding to answer dynamic questions, including tracking object movements, capturing ongoing interactions, and detecting contextual changes before and after phase transitions. However, developing VLMs tailored for surgical applications poses unique challenges. This is primarily attributed to the scarcity of domain-specific datasets, particularly those assessing temporal understanding and reasoning capabilities.

![](images/f71598efbe2c50140335e0febdda446ab6ecbe5c94ca8d26e3324d2ae436eeb4.jpg)  
Fig. 1. Two forms of limited temporal supervision in existing surgical VQA datasets. (a) Static questions focus on visible instrument identity and anatomical counting [5]. (b) A multi-frame example asks for a procedure-associated instrument and does not explicitly require reasoning over temporal relations among the frames [6]. The question wording and answers are reproduced verbatim from the original datasets.

Improving temporal reasoning requires more than exposing a model to additional video frames; the supervision must explicitly capture how surgical states interact and change over time. This motivates a structured temporal representation that enables models to learn how surgical scenes evolve rather than only what is visible in individual frames. However, existing surgical VQA datasets provide limited support for fine-grained temporal supervision. As illustrated in Fig. 1, some datasets provide only frame-level supervision (a), whereas others include video clips but pose coarse-grained recognition questions that do not explicitly probe temporal relations (b). Moreover, question wording and procedural priors may enable models to answer without sufficiently relying on visual evidence [7].

Recent studies have increasingly incorporated temporal encoders and video-language training objectives [8], [9], and large surgical VLMs have expanded the range of surgical visual tasks that can be addressed with instruction tuning [10]. Nevertheless, the supervision and evaluation signals for finegrained temporal reasoning remain limited. In particular, correctly answering a coarse-grained recognition question from one or a few salient frames does not establish that a model can track the same surgical entities across time or reason about how their interactions persist and change throughout a procedure.

Structured representations offer a complementary path. Scene graphs encode objects and their relations, and have been used to improve compositional visual reasoning in both general [11] and surgical [5], [12] settings. In surgery, scene-graphbased VQA is particularly well suited because instrumenttarget interactions are naturally expressed as structured triplets. However, frame-level surgical scene graphs represent objects and relations within individual images [5] and do not explicitly associate corresponding object instances or interaction triplets across frames. Without such cross-frame connectivity, repeated observations of the same triplet remain independent framelevel facts rather than being consolidated into a temporally bounded event. Consequently, questions generated directly from these graphs are limited to frame-local properties.

To address these limitations, we introduce a multi-level structured temporal supervision methodology for surgical video reasoning. The methodology transforms isolated framelevel scene graphs (SGs) [5] into an event-centered representation that jointly models object persistence and interaction continuity. We instantiate this methodology through spatio-temporal scene graphs (STSGs) and an evidencegrounded temporal query framework, resulting in the STSG-VQA benchmark. It covers seven question categories: count, duration, ordering, extreme, boundary, phase transition, and concurrency. In total, it comprises 18,458 QA pairs split into training, validation, and test sets of 13,932, 1,775, and 2,751, respectively. The evaluation protocol supports deterministic answer formats and LLM-judged (Qwen3.5-4B [33]) openended interaction descriptions. Initial baselines with Qwen2.5- VL [15], Qwen3-VL [16], and LLaVA-NeXT-Video [17] show that current VLMs remain weak on such surgical temporal understanding and reasoning tasks. Our contributions and findings are below:

• We introduce an event-centered structured temporal reasoning methodology that extends frame-level surgical scene graphs, transforming isolated surgical scenes into a representation of their temporal evolution.

• We introduce an evidence-grounded temporal query framework that operationalizes the proposed representation into seven reasoning capabilities: event counting, duration aggregation, interaction ordering, event concurrence, dominant-activity identification, immediate crossboundary reasoning, and broader pre/post proceduralchange reasoning.

• We proposed the STSG-VQA, a benchmark containing 18,458 question–answer pairs from 45 laparoscopic videos [5], [13], [14], together with a unified evaluation protocol covering deterministic and open-ended answers.

• We show that structured temporal supervision benefits both temporal reasoning and static surgical VQA, increasing the contribution of visual evidence, and yielding the strongest frame-level performance when combined with static supervision.

## II. RELATED WORK

## A. Surgical Vision-Language Models

The rise of VLMs has demonstrated remarkable capabilities in medical image understanding. CLIP-style models [18]–[21] learn joint visual-textual representations by aligning medical images with their corresponding clinical reports, thereby supporting downstream applications such as automated report generation and image-to-text retrieval.

Building upon these vision–language alignment frameworks, multimodal large language models such as LLaVA-Med [22], Med-Flamingo [23], and Med-Gemini [24] have extended medical image understanding toward instruction following, VQA, and open-ended multimodal dialogue. In surgery, domain-specific models such as SurgVLM [10] and SurgVidLM [26] have further extended this paradigm toward surgical perception, temporal analysis, and video-level understanding. These models operate as multimodal reasoning systems that integrate imaging data, patient records, and clinical guidelines within unified medical workflows, enabling more context-aware clinical understanding.

Although surgical VLMs have achieved promising results on image-based tasks, extending them to video understanding remains challenging because models must reason over complex temporal dependencies across evolving surgical scenes [25]. This requires models to identify transient visual evidence, integrate instrument-tissue interactions across frames, and retain procedural context over extended temporal intervals [8], [9], [26].

## B. Surgical Visual Question Answering

Surgical VQA has emerged as a specialized research paradigm that extends conventional VQA frameworks to the surgical domain. By jointly modeling visual information and natural language queries in complex operative scenes, it offers a promising direction for clinical training and decision support [9]. Early work introduced transformer-based surgical VQA on endoscopic images [2], while Surgical-VQLA extended the task by jointly predicting an answer and a bounding box around the question-relevant visual region in robotic surgery [3]. However, this grounding remains spatial and frame-level, without identifying how the corresponding entities or interactions persist across time. PitVQA further explored LLMbased VQA for pituitary surgery with image-grounded text embeddings [4]. More recently, SSG-VQA incorporated surgical scene graph knowledge to reduce question-conditioned bias and support geometry-aware reasoning over surgical triplets [5].

SurgViVQA advanced video-level surgical VQA by introducing a Masked Video–Text Encoder that fuses questions with short video clips to learn temporally aware latent representations for open-ended surgical question answering [8]. While this advances video-level surgical VQA beyond isolated-frame analysis, temporal evidence remains encoded in latent clip-level representations rather than explicitly represented through object tracks or temporally bounded interaction events. Our work instead introduces an event-centered STSG representation and derives evidence-grounded supervision for temporal reasoning.

TABLE I  
COMPARISON WITH REPRESENTATIVE SURGICAL VQA BENCHMARKS.
<table><tr><td>Dataset</td><td>Visual scope</td><td>Answer format</td><td></td><td>Temporal reasoning Structured evidence</td></tr><tr><td>EndoVis18-VQA</td><td>Frame image</td><td>Closed-set</td><td>No</td><td>No</td></tr><tr><td>PitVQA</td><td>Frame image</td><td>Open-ended</td><td>No</td><td>No</td></tr><tr><td>SSG-VQA</td><td>Frame-level SG</td><td>Closed-set</td><td>No</td><td>SG</td></tr><tr><td>REAL-Colon-VQA</td><td>8-frame clip</td><td>Open-ended</td><td>Yes</td><td>No</td></tr><tr><td>STSG-VQA (ours)</td><td>Video segment</td><td>Both</td><td>Yes</td><td>STSG</td></tr></table>

## C. Structured Knowledge for Enhanced Reasoning

Structured visual knowledge has been used to support reasoning beyond visual features. Visual Genome introduced dense object, attribute, and relationship annotations for imagelevel SG reasoning [27]. Action Genome extended this idea to videos by representing actions as spatio-temporal scene graphs, i.e., temporally ordered object-relation graphs that capture changes occurring throughout an action [11]. Several studies [28]–[30] have shown that explicitly representing objects across consecutive frames and their relationships as nodes and edges can be effectively applied to video understanding tasks. In surgical video analysis, structured labels such as triplets have been shown to capture fine-grained tool-tissue interactions [13], and recent surgical scene graph work models objects, actions, and anatomical relations for downstream understanding [5], [31]. However, such representations remain largely underexplored in surgical video analysis, where frequent instrument-tissue occlusions, rapid camera motion, and abrupt appearance changes (e.g., smoke, blood, trocar views) make cross-frame identity association unstable [10].

## III. METHODOLOGY

## A. Overview

We formulate surgical temporal reasoning through three complementary forms of structured connectivity: object-level continuity, which preserves entity identity across frames; event-level continuity, which consolidates repeated interactions into temporally extended surgical events; and procedure-level connectivity, which relates these events to surgical phases. Together, these components transform isolated frame-level scene descriptions into an event-centered representation of surgical state evolution. We use this representation to derive evidencegrounded temporal supervision, with each question and answer traceable to the graph relations and temporal intervals from which it is generated, rather than relying solely on free-form natural-language annotation. As summarized in Table I, STSG-VQA differs from existing surgical VQA benchmarks by jointly supporting video-level temporal reasoning and explicit evidence grounding through the STSG representation.

## B. Multi-Level Spatio-Temporal Representation

We model each surgical video as an ordered sequence of frame-level SGs and construct the corresponding video-level STSG by introducing temporal, procedural, and event-level

connectivity, as illustrated in Fig. 2 (a). For video i, the SG at frame t is represented as

$$
\mathrm { S G } _ { i , t } = \left( \mathcal { O } _ { i , t } , \mathcal { R } _ { i , t } \right) ,\tag{1}
$$

where $\mathcal { O } _ { i , t }$ denotes the set of detected entities and $\mathcal { R } _ { i , t }$ denotes the set of intra-frame semantic relations. Each object in $\mathcal { O } _ { i , t }$ stores its basic visual and semantic attributes $( \mathrm { e . g . }$ name, entity type, bounding box). The relation set contains spatial predicates between detected entities and surgical action triplets, represented as ⟨instrument, verb, target⟩, which encode the acting instrument, the performed surgical action, and the corresponding target, respectively.

The video-level STSG is obtained by composing the phaseaware frame-level SGs and subsequently augmenting them with cross-frame temporal links and event-level abstractions:

$$
\mathrm { S T S G } _ { i } = \mathrm { C o m p o s e } \left( \{ ( \mathrm { S G } _ { i , t } , p _ { i } ( t ) ) \} _ { t \in \mathcal { T } _ { i } } \right) \oplus \Delta _ { i } ^ { \mathrm { t e m p } } \oplus \Delta _ { i } ^ { \mathrm { e v t } } ,\tag{2}
$$

where $\mathcal { T } _ { i }$ denotes the chronologically ordered frame set of video $i , p _ { i } ( t )$ is the surgical phase associated with frame t, and ⊕ denotes graph augmentation with additional nodes or edges. The composition operator preserves the entity nodes and intraframe relations of each SG while associating the corresponding frame with its procedural phase. The temporal linking $\Delta _ { i } ^ { \mathrm { t e m } \mathbf { \overline { { p } } } }$ establishes object-level continuity across frames, whereas the event aggregation $\Delta _ { i } ^ { \mathrm { e v t } }$ abstracts persistent frame-level triplets into temporally grounded event nodes. The resulting STSG therefore provides a compact structured representation of the surgical video that emphasizes task-relevant entities, interactions, procedural context, and their temporal evolution.

The temporal augmentation of the frame-level SGs proceeds in three stages: cross-frame object association, frame-level action grounding, and event-level temporal aggregation. These stages progressively transform isolated frame observations into persistent entities and temporally coherent surgical interactions.

1) Object-Level Persistence: We first establish object continuity across successive frames. Given two consecutive frames t and $t + 1$ , candidate correspondences are restricted to object observations sharing the same component label (e.g., grasper) and semantic type (e.g., instrument). Their spatial compatibility is measured by the intersection-over-union (IoU) of their bounding boxes:

$$
\mathrm { I o U } ( b _ { a } , b _ { b } ) = \frac { | b _ { a } \cap b _ { b } | } { | b _ { a } \cup b _ { b } | } .\tag{3}
$$

Among admissible candidates, a one-to-one assignment is obtained by maximizing the overall bounding-box overlap. Accepted correspondences introduce temporal coreference edges between matched observations, thereby linking otherwise isolated frame-level detections into persistent object tracks. These tracks provide the entity-level temporal continuity required for associating surgical interactions across frames.

2) Frame-level Action Grounding: Given the established object observations, we next resolve each semantic action triplet to the specific entities participating in the interaction. When a frame-level SG contains a verb-specific relation, its local object indices directly identify the corresponding instrument and target instances. This relation-guided grounding is particularly important when multiple instances share the same component label, such as two graspers appearing in the same frame, for which semantic labels alone are insufficient to determine which instance performs the action.

![](images/fe8e0555fbf4b3997f1069561c44f81603e72dc4c8c111c39b05142a7631bc70.jpg)

![](images/7f70b0c13cda7eb578374aa5129afdbe8af1a460bf7885fd360d27623a498921.jpg)

Question: Within the interval from 00:01:05 to 00:11:04 in the Calot triangle dissection phase, while the hook dissecting the omentum is shown, which other listed instrument is also being used? A. clipper B. bipolar C. grasper D. irrigator Answer: C  
![](images/c86b04b54658e24e88513f24a26494088c3cf95374873aaa016666a3833a42d5.jpg)  
Question: Between 00:15:59 and 00:16:58 during the Calot triangle dissection phase, is the grasper retracting the gallbladder seen before the hook dissecting the cystic plate? Answer: True (b)  
Fig. 2. (a) Overview of STSG construction from frame-level surgical scene graphs. Within-frame spatial and instrument–target interaction edges preserve local semantic relations, temporal tracks link observations of the same object across frames, and repeated triplets are aggregated into continuous event nodes. Line styles denote spatial, interaction, and temporal associations, as indicated in the legend. (b) Representative temporal QA instances. The top sample illustrates a concurrency question, which assesses the co-occurrence of surgical interactions by identifying another instrument being used while the hook dissects the omentum. The bottom example illustrates an ordering question, which assesses the temporal precedence between the grasper retracting the gallbladder and the hook dissecting the cystic plate. Bounding boxes and highlighted object instances indicate the visual evidence associated with the corresponding objects and interactions.

If an endpoint specified by the semantic annotation is not supported by a visual detection, the missing endpoint is retained only as implicit graph evidence rather than instantiated as a reliable visual entity. This preserves the available semantic information while maintaining a distinction between visually observed and weakly inferred evidence. Each successfully grounded interaction is represented as a frame-level action edge connecting the participating entities.

3) Event-level temporal aggregation: Frame-level action edges are subsequently consolidated across time because a surgical maneuver typically persists over multiple consecutive frames. Rather than treating repeated occurrences of the same interaction as independent observations, compatible action edges are accumulated into an event state. For example, continuous gallbladder retraction across several frames is represented as a single temporally grounded event rather than as a sequence of redundant frame-level triplets.

To accommodate transient occlusions and missed detections, an active event is allowed to remain open across short observation gaps. The event is terminated when no compatible interaction is observed beyond the predefined temporal tolerance or when the video ends. Each completed event records its participating entities, temporal boundaries, procedural phase context, and supporting frame-level evidence.

Through these successive augmentations, the STSG preserves fine-grained frame-level observations while introducing two complementary forms of temporal structure: object tracks encode how surgical entities persist over time, whereas event nodes encode how their interactions unfold over temporal intervals. This representation therefore provides the structural evidence required to reason about when an interaction occurs, how long it persists, how different events are ordered or overlap, and how they relate to the surrounding object trajectories and procedural phases. Algorithm 1 summarizes the complete video-level STSG construction procedure.

<table><tr><td colspan="2">Algorithm 1 Video-Level STSG Construction</td></tr><tr><td>Require: Frame-level SGs  $\{ p _ { i } ( t ) \} _ { t \in \mathcal { T } _ { i } }$  Ensure: Video-level STSG STSGi</td><td> $\{ \mathrm { S G } _ { i , t } \} _ { t \in \mathcal { T } _ { i } }$  phase labels</td></tr><tr><td>1: Initialize  $\mathrm { S T S G } _ { i } ,$  states.</td><td>active object tracks, and open event</td></tr><tr><td>2: for  $t \in \mathcal T _ { i }$  3:</td><td>in temporal order do Add the frame objects, intra-frame relations, and phase</td></tr><tr><td>4:</td><td>context. Associate object observations with active tracks and add</td></tr><tr><td>5:</td><td>temporal coreference edges. Ground action triplets and insert frame-level action</td></tr><tr><td></td><td>edges.</td></tr><tr><td>6:</td><td>Extend compatible event states or initialize new events.</td></tr><tr><td>7:</td><td>Close event states whose temporal tolerance has been exceeded.</td></tr><tr><td>8: end for</td><td>9: Close the remaining events and attach their participants,</td></tr><tr><td>10: return</td><td>temporal boundaries, and phase context.  $\mathrm { S T S G } _ { i }$ </td></tr></table>

## C. Evidence-Grounded Temporal Supervision

Given the constructed STSG<sub>i</sub>, temporal QA pairs are generated by executing deterministic, phase-aware queries over graph events and object tracks. Each query specifies a semantic interaction, an answer scope, and the temporal property to be inferred. The interaction can be represented either as a complete triplet or as a partially specified interaction in which the instrument or target is left unspecified. This supports questions about both specific surgical interactions and broader patterns involving an action, instrument, or target. For comparisonbased questions, we retain only fully grounded interactions with explicit instrument and target instances, excluding implicit endpoints that cannot be reliably verified from the video.

Surgical phases serve as natural units for temporal sampling. For every individual phase, we implement early, middle, and late sampling windows across various temporal scales of up to 600 seconds. Boundary and Phase-transition questions instead use dedicated windows around adjacent phases. Importantly, the interval used to determine an answer is distinguished from the video segment presented to the model. The latter includes additional temporal context around the answer scope, whereas only evidence within the time interval explicitly stated in the question contributes to the answer. This design requires the model to localize the relevant interval while preventing surrounding context from contaminating the reference label.

For each semantic query q, we collect all matching STSG events that overlap the answer interval W. The event intervals are clipped to W, and overlapping intervals or intervals separated by no more than δ are merged into a single temporal episode. We set δ = 2 seconds to reduce fragmentation caused by brief missed detections. The resulting episode set is denoted by

$$
\mathcal { E } _ { i , q } ( W ) = \{ [ u _ { m } , v _ { m } ] \} _ { m = 1 } ^ { M _ { i , q } } ,\tag{4}
$$

where $M _ { i , q }$ is the number of merged episodes. The corresponding episode count and cumulative duration are computed as

$$
\begin{array} { l } { { \displaystyle N _ { i , q } ( W ) = M _ { i , q } , } } \\ { { \displaystyle D _ { i , q } ( W ) = \frac { 1 } { f _ { i } } \sum _ { m = 1 } ^ { M _ { i , q } } \left( v _ { m } - u _ { m } + 1 \right) , } } \end{array}\tag{5}
$$

where $f _ { i }$ denotes the temporal sampling rate, set to one frame per second (FPS) in our STSG construction. Consequently, Count questions refer to distinct temporal episodes rather than individual event nodes, whereas Duration questions measure the total length of the merged episode intervals.

The same episode representation supports compositional temporal reasoning. Ordering is determined from the first occurrence of each interaction, with temporally ambiguous comparisons excluded. Concurrency is established from interval intersections between activities performed by different instruments, with a minimum sustained overlap used to suppress incidental frame-level coincidences. Extreme questions identify the dominant instrument, target, or complete interaction according to cumulative duration or episode frequency; tied or insufficiently separated comparisons are discarded. Together, these categories require the model to reason about when an event occurs, whether two events coexist, and which activity predominates over an extended interval.

Procedural annotations introduce two complementary forms of phase-aware reasoning. Boundary questions focus on directly adjacent phase transitions using a symmetric 15 seconds window on each side of the boundary, clipped to the corresponding phase limits. They test whether the same interaction continues across the exact boundary, identify the first interaction, action, or target appearing afterward, and summarize the change from the last pre-boundary event to the first new post-boundary event. Phase-transition questions operate at a broader scale by comparing up to the final 180 seconds of the preceding phase with the first 180 seconds of the subsequent phase. They characterize dominant early-phase actions and interactions, changes in dominant activity, the largest increase in episode frequency, and interactions newly visible after the transition.

TABLE II  
DISTRIBUTION OF TEMPORAL QA CATEGORIES IN STSG-VQA.
<table><tr><td>Category</td><td>Temporal focus</td><td>Number</td><td>Prop.</td></tr><tr><td>Concurrency</td><td>Concurrent surgical activities</td><td>5,862</td><td>31.76%</td></tr><tr><td>Ordering</td><td>Temporal order of interactions</td><td>4,904</td><td>26.57%</td></tr><tr><td>Extreme</td><td>Longest or most frequent activity</td><td>2,752</td><td>14.91%</td></tr><tr><td>Count</td><td>Number of activity episodes</td><td>1,800</td><td>9.75%</td></tr><tr><td>Duration</td><td>Cumulative visible activity time</td><td>1,350</td><td>7.31%</td></tr><tr><td></td><td>Phase transition Pre/post procedural change</td><td>1,197</td><td>6.48%</td></tr><tr><td>Boundary</td><td>Immediate cross-boundary behavior</td><td>593</td><td>3.21%</td></tr><tr><td colspan="2">Total</td><td>18,458</td><td>100.0%</td></tr></table>

Having defined the temporal relations underlying each question category, we construct negative instances from observed STSG facts rather than arbitrary semantic combinations. Zerocount queries use interactions absent from the selected scope but observed elsewhere in the video; Ordering and Extreme negatives reverse verified temporal or ranked relations; Concurrency negatives pair co-visible but non-overlapping activities; and phase-aware negatives violate the corresponding boundary or transition conditions. This preserves semantic plausibility while requiring answers to be grounded in temporal evidence.

After positive and negative candidates are instantiated, we apply a diversity-aware sampler that limits repeated use of the same semantic query and phase section; ensures coverage of all feasible question subtypes; and prioritizes candidates with unambiguous category-specific temporal evidence. As summarized in Table II, this procedure yields 18,458 QA pairs across 45 videos, with Concurrency and Ordering constituting the largest components.

Finally, to mitigate template-induced shortcuts, we randomly sample from 108 distinct natural-language templates covering seven temporal-reasoning categories and their associated question subtypes. The resulting benchmark includes numeric, directional, binary, four-option multiple-choice, and concise open-ended answers. Each QA pair retains the displayed video segment, its exact answer scope, the supporting temporal segments, and the structured graph evidence used to derive the reference answer. Fig. 2 (b) illustrates representative generated questions from the Concurrency and Ordering categories.

TABLE III  
VIDEO-LEVEL PARTITION AND STSG-VQA STATISTICS.
<table><tr><td>Split</td><td>Videos</td><td>Frames</td><td>Event nodes</td><td>QA pairs</td></tr><tr><td>Training</td><td>35</td><td>65,535</td><td>20,180</td><td>13,932</td></tr><tr><td>Validation</td><td>5</td><td>11,300</td><td>3,684</td><td>1,775</td></tr><tr><td>Test</td><td>5</td><td>10,564</td><td>3,279</td><td>2,751</td></tr><tr><td>Total</td><td>45</td><td>87,399</td><td>27,143</td><td>18,458</td></tr></table>

## IV. EXPERIMENTS

Our experiments are organized around a central question: does structured temporal supervision improve a VLM’s ability to reason over the evolution of surgical events? We first compare zero-shot performance, static scene-graph supervision, and the proposed structured temporal supervision across seven temporal reasoning categories. We then test whether the resulting gains reflect increased use of visual evidence and whether structured temporal supervision transfers to and complements frame-level surgical VQA. Finally, we use STSG-VQA to characterize the remaining zero-shot limitations of current VLMs and the effect of increasing the frame budget.

## A. Dataset and Experimental Setting

The experiments use two complementary surgical VQA datasets: the proposed STSG-VQA benchmark and SSGVQA [5]. STSG-VQA is the primary benchmark for evaluating reasoning over temporally extended surgical events, whereas SSGVQA evaluates compositional understanding of individual surgical frames. This pairing allows us to separate the ability to recognize a surgical state from the ability to reason about how that state persists, changes, or interacts with other states over time.

STSG-VQA is constructed from 45 laparoscopic cholecystectomy videos with aligned frame-level surgical scene graphs [5], surgical phase annotations [13], and a manually verified set of valid frames. The videos span 90,089 seconds, corresponding to approximately 25.0 hours of operative video. After validity filtering and sampling at 1 FPS, the resulting STSGs contain 87,399 frame nodes, 634,460 object observations, and 27,143 temporally aggregated event nodes. To prevent visual and procedural leakage, the data are partitioned at the video level rather than at the QA level (Table III).

For the frame-level experiments, we follow the original SSGVQA formulation and evaluation protocol. SSGVQA contains question–answer pairs grounded in individual surgical frames and is therefore used both as a source of static scenegraph supervision and as a target for testing whether the benefits of structured temporal supervision extend to framelevel surgical VQA.

## B. Evaluation Protocol

We evaluate three general-purpose open-source VLMs— Qwen2.5-VL-7B [15], Qwen3-VL-4B [16], and LLaVA-NeXT-Video-7B [17]—together with the surgical-domain

Hulu-Med-4B model [32]. For each QA item, frames are uniformly sampled in chronological order from the interval specified by segment start time and segment end time. Unless otherwise stated, a model receives 16 frames and a prompt containing the segment time span, naturallanguage question, required answer format, and output prefix FINAL ANSWER:. The answer scope is not used to crop the visual input. At inference, the model receives neither the reference answer nor the STSG-derived evidence used to generate and verify it, including the supporting event intervals, object tracks, and graph relations. Therefore, the observed fine-tuning gains cannot be attributed to direct access to the structured evidence from which the labels were derived.

Predictions are aligned with references by qa id. Missing predictions are scored as incorrect, while duplicate and unmatched predictions are retained as diagnostics. Integer counts require an exact match after parsing the first integer. For duration questions, we use a continuous and non-decreasing tolerance that combines an absolute allowance for short intervals with a relative allowance for longer intervals:

$$
\tau ( r ) = \left\{ \begin{array} { l l } { 3 , } & { r \leq 1 0 , } \\ { \operatorname* { m a x } ( 3 , 0 . 2 0 r ) , } & { 1 0 < r \leq 6 0 , } \\ { \operatorname* { m i n } ( 3 0 , \operatorname* { m a x } ( 1 2 , 0 . 1 5 r ) ) , } & { r > 6 0 , } \end{array} \right.\tag{6}
$$

where r is the reference duration in seconds. The values from adjacent regimes agree at $r = 1 0$ and $r = 6 0$ , avoiding an artificial change in scoring strictness at either boundary. A prediction rˆ is correct when $| \hat { r } - r | \le \tau ( r )$ . Binary answers are normalized across true/false and yes/no variants. Multiplechoice responses are parsed as A–D labels, with a fallback that maps predicted option text to the alternatives stated in the question. An unparseable deterministic answer is counted as incorrect.

Open-ended responses are scored by a deterministic local Qwen3.5-4B judge [33]. The judge receives only the question, category and sub-category, segment windows, reference answer, and candidate answer, and returns a binary semanticequivalence decision.

Let $c _ { j } \in \{ 0 , 1 \}$ denote correctness for QA item j. We report question-level micro accuracy

$$
\mathrm { A c c u r a c y } _ { \mathrm { m i c r o } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } c _ { j } ,\tag{7}
$$

and define video-balanced accuracy as

$$
\mathrm { A c c u r a c y } _ { \mathrm { v i d e o } } = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \frac { 1 } { N _ { i } } \sum _ { j \in \mathcal { Q } _ { i } } c _ { j } ,\tag{8}
$$

where $\mathcal { Q } _ { i }$ contains the $N _ { i }$ questions from video i. This second measure prevents videos with more generated questions from disproportionately determining the aggregate result.

## C. Implementation Details

We adapt Qwen3-VL-4B and Hulu-Med-4B to STSG-VQA using multimodal QLoRA. Both models are loaded with 4-bit NormalFloat quantization, double quantization, and bfloat16 computation. LoRA modules are applied to language-attention, visual-to-language connector, and late vision-attention components, using component-specific ranks and learning rates fixed throughout training. The models are trained for 1.5 epochs with a per-device batch size of 1, 16 gradient-accumulation steps, and loss restricted to assistantanswer tokens. Training and the main test evaluation use 16 frames sampled uniformly and chronologically from each visual-context interval. Text-only evaluation is additionally used to quantify reliance on linguistic and procedural priors.

TABLE IV  
QUESTION-LEVEL ACCURACY (%) ACROSS TEMPORAL CATEGORIES IN THE 16-FRAME SETTING. N DENOTES THE NUMBER OF TEST QUESTIONS. “+SSGVQA” AND “+OURS” DENOTE FINE-TUNING ON SSGVQA AND STSG-VQA, RESPECTIVELY, FOLLOWED BY EVALUATION ON THE STSG-VQA TEST SET. PARENTHESES SHOW PERCENTAGE-POINT CHANGES FROM THE CORRESPONDING ZERO-SHOT BACKBONE.
<table><tr><td>Model N</td><td>Concurrency 1005</td><td>Ordering 758</td><td>Count 200</td><td>Extreme 421</td><td>Duration 150</td><td>Boundary 69</td><td>Phase trans. 148</td></tr><tr><td>Qwen2.5-VL</td><td>57.61</td><td>47.89</td><td>14.50</td><td>38.95</td><td>12.67</td><td>42.03</td><td>45.95</td></tr><tr><td>LLaVA-NeXT-Video</td><td>34.73</td><td>33.91</td><td>24.00</td><td>26.60</td><td>6.67</td><td>37.68</td><td>44.59</td></tr><tr><td>Qwen3-VL</td><td>53.13</td><td>37.99</td><td>19.50</td><td>45.37</td><td>10.67</td><td>43.48</td><td>47.97</td></tr><tr><td>Hulu-Med</td><td>51.44</td><td>49.08</td><td>20.50</td><td>48.22</td><td>14.67</td><td>39.13</td><td>47.30</td></tr><tr><td>Hulu-Med+SSGVQA</td><td>63.08 (+11.64)</td><td>50.40 (+1.32)</td><td>30.00 (+9.50)</td><td>47.51 (-0.71)</td><td>8.00 (-6.67)</td><td>55.07 (+15.94)</td><td>48.65 (+1.35)</td></tr><tr><td>Hulu-Med+STSG-VQA (Ours)</td><td>80.40 (+28.96)</td><td>61.74 (+12.66)</td><td>40.00 (+19.50)</td><td>63.66 (+15.44)</td><td>16.67 (+2.00)</td><td>71.01 (+31.88)</td><td>62.16 (+14.86)</td></tr><tr><td>Qwen3-VL+SSGVQA</td><td>57.91 (+4.78)</td><td>53.83 (+15.84)</td><td>32.00 (+12.50)</td><td>52.02 (+6.65)</td><td>7.33 (-3.34)</td><td>59.42 (+15.94)</td><td>41.22 (-6.75)</td></tr><tr><td>Qwen3-VL+STSG-VQA (Ours)</td><td>82.29 (+29.16)</td><td>63.19 (+25.20)</td><td>34.00 (+14.50)</td><td>68.17 (+22.80)</td><td>18.67 (+8.00)</td><td>72.46 (+28.98)</td><td>68.24 (+20.27)</td></tr></table>

TABLE V  
MICRO ACCURACY (%) UNDER VISUAL-INPUT CONTROLS ON STSG-VQA.
<table><tr><td>Model</td><td>Text</td><td>16 frames</td><td> $\Delta _ { \mathrm { 1 6 - t e x t } }$ </td></tr><tr><td>Qwen3-VL zero-shot</td><td>41.26</td><td>42.49</td><td>+1.23</td></tr><tr><td>Qwen3-VL + STSG-VQA</td><td>49.98</td><td>66.88</td><td>+16.90</td></tr><tr><td>Hulu-Med zero-shot</td><td>39.88</td><td>45.51</td><td>+5.63</td></tr><tr><td>Hulu-Med + STSG-VQA</td><td>51.58</td><td>65.07</td><td>+13.49</td></tr></table>

We conduct three complementary evaluations involving SSGVQA. First, models fine-tuned only on SSGVQA are evaluated on the STSG-VQA test set; these rows are denoted by “+SSGVQA” in Table IV. They measure how far static scene-graph supervision transfers to temporal reasoning. Second, models fine-tuned only on STSG-VQA are evaluated directly on the official SSGVQA test split; these results test whether structured temporal supervision transfers to framelevel surgical VQA without direct SSGVQA fine-tuning. Third, SSGVQA-only and hybrid models are compared on the SSGVQA test split, where hybrid training combines SSGVQA and STSG-VQA examples. All experiments are executed on a single NVIDIA L40S GPU with 48 GB memory.

## D. Experimental Results

1) Structured Temporal Supervision Improves Temporal Reasoning: Table IV presents the central result of this study. Structured temporal supervision increases question-level micro accuracy from 42.49% to 66.88% for Qwen3-VL and from 45.51% to 65.07% for Hulu-Med, corresponding to improvements of 24.39 and 19.56 percentage points over their zero-shot backbones. It also outperforms static scene-graph supervision by 16.50 and 14.25 points, respectively. Crucially, these improvements extend across all seven temporal reasoning categories for both model families, whereas static scene-graph supervision produces uneven transfer and reduces duration accuracy for both models. This distinction shows that stronger recognition of instruments, targets, and within-frame relations is not sufficient for temporal reasoning. Learning from temporally connected surgical events additionally improves the model’s ability to reason about event persistence, concurrence, ordering, and procedural transitions.

The largest gains over the zero-shot backbones occur in concurrency and boundary reasoning: Qwen3-VL improves by 29.16 and 28.98 points, while Hulu-Med improves by 28.96 and 31.88 points, respectively. These tasks require the model to relate simultaneous interactions or states on opposite sides of a procedural boundary, making their gains particularly consistent with learning event-level temporal structure rather than only a richer surgical vocabulary.

The remaining weakness is also structured. Duration is the lowest-scoring category after STSG-VQA fine-tuning, reaching only 18.67% for Qwen3-VL and 16.67% for Hulu-Med; count reaches 34.00% and 40.00%, respectively. STSG supervision therefore improves the identification and comparison of event states more readily than exact temporal accumulation. This distinction suggests that learning which events coexist or change is easier than maintaining a calibrated memory of how often, or for how long, they occur under sparse video sampling.

2) Structured Temporal Supervision Increases the Contribution of Visual Evidence: Having established that structured temporal supervision improves temporal question answering, we next examine whether these gains reflect greater use of visual evidence rather than stronger reliance on linguistic or procedural priors. Table V tests whether the temporal gains can be explained by textual regularities alone. The zeroshot Qwen3-VL model already achieves 41.26% without any image, only 1.23 points below its 16-frame result. Hulu-Med exhibits a larger but still limited 5.63-point visual advantage. These high text-only scores expose substantial answerability from question wording and procedural priors in both model families, and show why a high video-QA score alone is not sufficient evidence of visually grounded reasoning.

STSG-VQA fine-tuning substantially changes the relationship between language and vision. The 16-frame advantage

Question: Within the interval from 00:07:47 to 00:17:46 in the Calot triangle dissection phase, when the grasper is retracting, which listed target is seen first? A. omentum B. abdominal wall cavity C. cystic artery D. gallbladder Answer: D

![](images/a6e514f5037d0fbf4940473b1cf40df6ef19cc4826524d6c26c5a6342c35f8aa.jpg)  
Qwen3VL finetuned with both SSGVQA and STSG-VQA

Fig. 3. Decoder-attention comparison for the same temporal question. The top and bottom rows show Qwen3-VL-4B after SSGVQA-only and hybrid fine-tuning, respectively. Attention from answer-token positions to visual tokens is averaged over all heads, the final four decoder layers, and all answer-generation steps. Warmer colors indicate higher normalized attention.  
TABLE VI  
PERFORMANCE (%) ON THE SSGVQA TEST SET. “STSG-VQA” DENOTES DIRECT EVALUATION AFTER STSG-VQA-ONLY FINE-TUNING; “HYBRID” COMBINES SSGVQA AND STSG-VQA SUPERVISION.
<table><tr><td>Model</td><td>Setting</td><td>Acc.</td><td>mAP</td><td>mAR</td><td>mAF1</td><td>wF1</td></tr><tr><td>Qwen3-VL-4B</td><td>zero-shot</td><td>31.69</td><td>17.14</td><td>20.93</td><td>16.70</td><td>27.52</td></tr><tr><td>Qwen3-VL-4B</td><td>SSG-VQA</td><td>67.18</td><td>60.88</td><td>54.56</td><td>55.58</td><td>66.61</td></tr><tr><td>Qwen3-VL-4B</td><td>STSG-VQA</td><td>33.90</td><td>22.49</td><td>20.35</td><td>17.77</td><td>29.43</td></tr><tr><td>Qwen3-VL-4B</td><td>Hybrid</td><td>71.01</td><td>65.07</td><td>57.86</td><td>59.09</td><td>70.60</td></tr><tr><td>Hulu-Med-4B</td><td>zero-shot</td><td>26.10</td><td>20.70</td><td>23.72</td><td>17.32</td><td>21.94</td></tr><tr><td>Hulu-Med-4B</td><td>SSG-VQA</td><td>64.06</td><td>54.72</td><td>48.35</td><td>49.00</td><td>63.41</td></tr><tr><td>Hulu-Med-4B</td><td>STSG-VQA</td><td>30.85</td><td>21.93</td><td>22.55</td><td>17.57</td><td>26.70</td></tr><tr><td>Hulu-Med-4B</td><td>Hybrid</td><td>67.10</td><td>56.11</td><td>49.86</td><td>50.47</td><td>66.22</td></tr></table>

Note: Acc. denotes overall accuracy. mAP, mAR, and mAF1 are the unweighted averages of per-class precision, recall, and F1 across the 51 answer classes, respectively. wF1 denotes the class-support-weighted average of per-class F1.

over text-only input expands to 16.90 points for Qwen3- VL and 13.49 points for Hulu-Med. Text-only accuracy also increases after fine-tuning, indicating that the supervision teaches useful surgical and task regularities; however, the much larger improvement when frames are present makes a purely linguistic-shortcut account insufficient. Because the graph evidence is not supplied at inference, the expanded visual margin is consistent with the model learning to map observed frame sequences to the event relations encoded by the STSG-derived targets.

3) Temporal Reasoning Complements Static Surgical VQA: We next test whether the benefits of structured temporal supervision extend beyond temporal question answering to frame-level surgical VQA. The two benchmarks examine complementary directions of transfer. On STSG-VQA, static scene-graph supervision improves aggregate accuracy for both backbones, but the resulting gains remain uneven across temporal reasoning categories (Table IV). Conversely, Table VI shows that fine-tuning with structured temporal supervision alone improves SSGVQA accuracy from 31.69% to 33.90% for Qwen3-VL and from 26.10% to 30.85% for Hulu-Med.

Weighted F1 also increases from 27.52% to 29.43% and from 21.94% to 26.70%, respectively. The transfer is not uniform across all metrics—mAR decreases slightly for both models— so temporal QA supervision does not replace direct frame-level training. Nevertheless, the improvements in accuracy, mAP, mAF1, and wF1 for both backbones indicate that learning event histories can reinforce reusable representations of static surgical interactions.

Hybrid training provides the clearest evidence that the two supervisory signals are complementary. Relative to SSGVQAonly training, adding STSG-VQA examples improves every reported SSGVQA metric for both backbones. Qwen3-VL increases from 67.18% to 71.01% accuracy, with macro F1 rising from 55.58% to 59.09%; Hulu-Med increases from 64.06% to 67.10% accuracy, with macro F1 rising from 49.00% to 50.47%. Static scene graphs teach the model to resolve the constituents of a surgical state, whereas STSG-derived QA pairs additionally supervise persistence, co-occurrence, ordering, and transition. Their combination is therefore better understood as joint learning of surgical states and their dynamics, rather than as competition between two VQA datasets.

4) Qualitative Evidence of Temporally Selective Attention: We visualize decoder attention maps to show which visual regions receive attention while the fine-tuned Qwen3-VL-4B predicts its answer. Specifically, we extract causal selfattention from answer-token prediction positions to the visual tokens of eight uniformly sampled frames. We then average these weights across all attention heads, the final four decoder layers, and all answer-generation steps. The resulting perframe visual-token scores are mapped back to their spatial patch grids, normalized, and interpolated to the original frame resolution for visualization. As shown in Fig. 3, the SSGVQAonly model identifies salient anatomy around the Calot triangle and instrument–tissue interaction regions in individual frames, but does not consistently prioritize the evidence most relevant to the temporal-ordering question. Its attention therefore ap pears dominated by frame-level visual saliency. In contrast, the hybrid model exhibits attention that is qualitatively more concentrated on frames and regions associated with grasper retraction, suggesting improved alignment between the queried interaction and its occurrence across the sequence.

TABLE VII  
ZERO-SHOT PERFORMANCE ON THE STSG-VQA TEST SPLIT. ALL MODELS USE THE SAME PROMPTING AND EVALUATION PROTOCOL. FRAMES DENOTES THE NUMBER OF UNIFORMLY SAMPLED VISUAL OBSERVATIONS.
<table><tr><td rowspan="2">Model</td><td colspan="2">Setting</td><td colspan="5">Answer-Format Accuracy (%)</td><td colspan="2">Overall Accuracy (%)</td></tr><tr><td>Params Frames</td><td></td><td>Binary</td><td>Count</td><td></td><td>Duration Multi-choice</td><td>Open-ended</td><td>Video Avg.</td><td>Micro</td></tr><tr><td colspan="2">Open-source general-purpose VLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL</td><td>4B</td><td>8</td><td>40.85</td><td>19.00</td><td>10.67</td><td>49.87</td><td>28.30</td><td>39.37</td><td>40.79</td></tr><tr><td>Qwen3-VL</td><td>4B</td><td>16</td><td>44.17</td><td>19.50</td><td>10.67</td><td>50.88</td><td>27.67</td><td>40.85</td><td>42.49</td></tr><tr><td>Qwen3-VL</td><td>4B</td><td>32</td><td>49.38</td><td>18.50</td><td>10.67</td><td>51.73</td><td>25.79</td><td>43.23</td><td>44.67</td></tr><tr><td>Qwen2.5-VL</td><td>7B</td><td>8</td><td>54.31</td><td>16.50</td><td>10.67</td><td>50.80</td><td>8.81</td><td>43.47</td><td>45.04</td></tr><tr><td>Qwen2.5-VL</td><td>7B</td><td>16</td><td>55.55</td><td>14.50</td><td>12.67</td><td>51.14</td><td>6.29</td><td>43.66</td><td>45.47</td></tr><tr><td>Qwen2.5-VL</td><td>7B</td><td>32</td><td>56.59</td><td>15.50</td><td>8.67</td><td>50.72</td><td>7.55</td><td>43.79</td><td>45.62</td></tr><tr><td>LLaVA-NeXT-Video</td><td>7B</td><td>8</td><td>48.06</td><td>28.50</td><td>7.33</td><td>24.85</td><td>5.03</td><td>31.59</td><td>31.92</td></tr><tr><td>LLaVA-NeXT-Video</td><td>7B</td><td>16</td><td>48.06</td><td>24.00</td><td>6.67</td><td>24.77</td><td>5.66</td><td>31.23</td><td>31.55</td></tr><tr><td>LLaVA-NeXT-Video</td><td>7B</td><td>32</td><td>48.06</td><td>24.00</td><td>9.33</td><td>24.77</td><td>6.29</td><td>31.32</td><td>31.73</td></tr><tr><td colspan="2">Surgical VLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hulu-Med-4B</td><td>4B</td><td>8</td><td>50.43</td><td>24.00</td><td>16.00</td><td>53.33</td><td>14.47</td><td>44.91</td><td>45.80</td></tr><tr><td>Hulu-Med-4B</td><td>4B</td><td>16</td><td>50.14</td><td>20.50</td><td>14.67</td><td>53.58</td><td>15.09</td><td>44.39</td><td>45.51</td></tr><tr><td>Hulu-Med-4B</td><td>4B</td><td>32</td><td>49.19</td><td>24.50</td><td>16.00</td><td>53.58</td><td>11.95</td><td>44.25</td><td>45.33</td></tr></table>

Note: Video Avg. denotes video-balanced accuracy. Micro denotes question-level accuracy over all test QA pairs. Open-ended answers are scored by the semantic-equivalence judge.

5) Current VLMs Remain Limited in Zero-Shot Temporal Reasoning: After establishing the effect of the proposed structured temporal supervision, we use STSG-VQA to characterize the limitations of current general-purpose and surgicaldomain VLMs under zero-shot evaluation. Table VII establishes the zero-shot performance landscape on STSG-VQA. The strongest overall result is obtained by Hulu-Med-4B with 8 frames, reaching 45.80% micro accuracy and 44.91% videobalanced accuracy. Under the common 16-frame setting, Hulu-Med and Qwen2.5-VL are effectively tied at 45.51% and 45.47% micro accuracy, respectively, followed by Qwen3-VL at 42.49% and LLaVA-NeXT-Video at 31.55%. Their videobalanced results follow the same general pattern. The negligible 0.04-point difference between Hulu-Med and Qwen2.5- VL indicates that surgical-domain specialization does not, by itself, confer a decisive advantage in fine-grained temporal reasoning.

The frame-budget sweep further shows that access to more observations does not automatically produce stronger temporal reasoning. Qwen3-VL is the only model to improve consistently, increasing from 40.79% with 8 frames to 44.67% with 32 frames. In contrast, Qwen2.5-VL remains on a narrow plateau (45.04–45.62%), LLaVA-NeXT-Video remains nearly unchanged (31.55–31.92%), and Hulu-Med decreases slightly from 45.80% to 45.33%. Thus, temporal reasoning depends not only on the amount of visual evidence but also on whether the model can select, integrate, and retain information across observations.

The answer-format results clarify what underlies these aggregate trends. Qwen2.5-VL is strongest on binary questions,

Hulu-Med performs best on multiple-choice and duration questions, LLaVA-NeXT-Video obtains the highest count accuracy, and Qwen3-VL is strongest on open-ended answers. Notably, Qwen3-VL’s gain from additional frames is driven primarily by binary accuracy, which rises from 40.85% to 49.38%, whereas its open-ended accuracy decreases from 28.30% to 25.79%. More frames can therefore improve coarse decisions without producing a corresponding improvement in semantically precise event reconstruction. Across all models, count, duration, and open-ended questions remain substantially more difficult than binary and multiple-choice decisions, reinforcing the need for supervision that explicitly models event evolution rather than relying on additional visual observations alone.

## V. DISCUSSION

The experiments distinguish surgical-state recognition from event-centered temporal reasoning. Zero-shot VLMs can often exploit linguistic and procedural priors, while static SSGVQA supervision improves instantaneous entity–relation recognition but transfers unevenly to temporal tasks. In contrast, STSG-VQA supervision improves all seven temporal categories, increases dependence on visual input, and complements static supervision in hybrid training. Because the STSG is used to construct supervision but is withheld at inference, these findings are consistent with VLMs internalizing structured knowledge about event persistence, concurrence, and transition rather than directly retrieving graph evidence.

Several limitations constrain this interpretation. STSG-VQA contains 45 videos from a single procedure, and video-level splitting cannot establish generalization across institutions, acquisition systems, patient populations, or other procedures.

Moreover, generated templates and phase-conditioned questions retain textual and workflow priors despite grounded negative construction and template diversification. External validation, held-out paraphrase families, phase-name masking, and temporally shuffled or reversed inputs would provide stricter tests of generalization and temporal dependence.

Finally, 1-Hz STSG construction and uniform 16-frame sampling can miss brief interactions and repeated-event boundaries, contributing to the remaining difficulty in count and duration reasoning. Future work should investigate denser event-local sampling, uncertainty-aware tracking, hierarchical temporal memory, and explicit event-state updates. The present benchmark is intended for model development and evaluation and is not clinically validated for decision support.

## VI. CONCLUSION

We introduced a structured temporal reasoning framework that augments frame-level surgical scene graphs with object-level continuity, event-level interaction continuity, and procedure-level connectivity. We then propose the STSG-VQA benchmark to supervise and evaluate seven forms of temporal reasoning. Zero-shot VLMs exhibit substantial reliance on textual and procedural priors, whereas structured temporal supervision improves every temporal category and substantially increases the contribution of visual evidence. Its transfer to frame-level SSGVQA, together with the consistent gains obtained through hybrid training, further shows that surgical-state recognition and temporal state-transition reasoning provide complementary supervisory signals.

These findings position structured temporal supervision not merely as a means to populate a benchmark, but as a reasoning scaffold that connects recognition of surgical states with modeling of their evolution. Duration estimation and event counting remain challenging, yet the resulting framework provides a concrete step toward surgical multimodal models that reason over persistent entities, interactions, and procedural transitions rather than isolated frames.

## REFERENCES

[1] C. I. Nwoye et al., “Recognition of instrument–tissue interactions in endoscopic videos via action triplets,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2020, pp. 364–374.

[2] L. Seenivasan, M. Islam, A. K. Krishna, and H. Ren, “Surgical-VQA: Visual question answering in surgical scenes using transformer,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2022, pp. 33–43.

[3] L. Bai, M. Islam, L. Seenivasan, and H. Ren, “Surgical-VQLA: Transformer with gated vision–language embedding for visual question localized-answering in robotic surgery,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 6859–6865, doi: 10.1109/ICRA48891.2023.10160403.

[4] R. He et al., “PitVQA: Image-grounded text embedding LLM for visual question answering in pituitary surgery,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2024, pp. 488–498.

[5] K. Yuan, M. Kattel, J. L. Lavanchy, N. Navab, V. Srivastav, and N. Padoy, “Advancing surgical VQA with scene graph knowledge,” Int. J. Comput. Assist. Radiol. Surg., vol. 19, no. 7, pp. 1409–1417, 2024.

[6] Y. Li et al., “SurgPub-Video: A comprehensive surgical video framework for enhanced surgical intelligence in vision–language model,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 8, pp. 6628–6635, 2026, doi: 10.1609/aaai.v40i8.37593.

[7] J. Shin, K. Y. Kim, E. Cho, S. T. Kim, and N. Oh, “SurgCheck: Do vision–language models really look at images in surgical VQA?” arXiv:2605.01911, 2026.

[8] M. O. Drago et al., “SurgViVQA: Temporally grounded video question answering for surgical scene understanding,” Int. J. Comput. Assist. Radiol. Surg., 2026, doi: 10.1007/s11548-026-03695-z.

[9] S. Li et al., “SurgTEMP: Temporal-aware surgical video question answering with text-guided visual memory for laparoscopic cholecystectomy,” arXiv:2603.29962, 2026.

[10] Z. Zeng et al., “SurgVLM: A large vision–language model and systematic evaluation benchmark for surgical intelligence,” arXiv:2506.02555, 2025.

[11] J. Ji, R. Krishna, L. Fei-Fei, and J. C. Niebles, “Action Genome: Actions as compositions of spatio-temporal scene graphs,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 10236–10247.

[12] E. Ozsoy, E. P. Ornek, U. Eck, T. Czempiel, F. Tombari, and N. Navab, “4D-OR: Semantic scene graphs for OR domain modeling,” arXiv:2203.11937, 2022.

[13] C. I. Nwoye et al., “Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos,” Med. Image Anal., vol. 78, 2022, Art. no. 102433.

[14] C. I. Nwoye and N. Padoy, “Data splits and metrics for method benchmarking on surgical action triplet datasets,” arXiv:2204.05235, 2022.

[15] S. Bai et al., “Qwen2.5-VL technical report,” arXiv:2502.13923, 2025. [15] S. Bai et al., “Qwen2.5-VL technical report," arXiv:2502.13923, 2025

[16] S. Bai et al., “Qwen3-VL technical report,” arXiv:2511.21631, 2025.

[17] Y. Zhang, B. Li, H. Liu, Y. J. Lee, L. Gui, D. Fu, J. Feng, Z. Liu, and C. Li, “LLaVA-NeXT: A strong zeroshot video understanding model,” LLaVA Blog, Apr. 30, 2024. [Online]. Available: https://llava-vl.github.io/blog/ 2024-04-30-llava-next-video/. Accessed on: Jul. 28, 2026.

[18] Z. Wang, Z. Wu, D. Agarwal, and J. Sun, “MedCLIP: Contrastive learning from unpaired medical images and text,” in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), Abu Dhabi, United Arab Emirates, 2022, pp. 3876–3887.

[19] S. Eslami, C. Meinel, and G. de Melo, “PubMedCLIP: How much does CLIP benefit visual question answering in the medical domain?” in Findings Assoc. Comput. Linguistics: EACL, Dubrovnik, Croatia, 2023, pp. 1181–1193.

[20] S. Zhang et al., “A multimodal biomedical foundation model trained from fifteen million image–text pairs,” NEJM AI, vol. 2, no. 1, 2025, Art. no. AIoa2400640.

[21] K. Yuan et al., “Learning multimodal representations by watching hundreds of surgical video lectures,” Med. Image Anal., vol. 105, 2025, Art. no. 103644.

[22] C. Li et al., “LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 28541–28564.

[23] M. Moor et al., “Med-Flamingo: A multimodal medical few-shot learner,” in Proc. 3rd Mach. Learn. Health Symp., vol. 225, 2023, pp. 353–367.

[24] K. Saab et al., “Capabilities of Gemini models in medicine,” arXiv:2404.18416, 2024.

[25] U. Khan et al., “Surgical scene understanding in the era of foundation AI models: A comprehensive review,” arXiv:2502.14886, 2025.

[26] G. Wang et al., “SurgVidLM: Towards multi-grained surgical video understanding with large language models,” arXiv:2506.17873, 2025.

[27] R. Krishna et al., “Visual Genome: Connecting language and vision using crowdsourced dense image annotations,” Int. J. Comput. Vis., vol. 123, no. 1, pp. 32–73, 2017.

[28] Y. Zhao et al., “Constructing holistic spatio-temporal scene graphs for video semantic role labeling,” in Proc. ACM Int. Conf. Multimedia, 2023, pp. 5281–5291.

[29] H. Fei et al., “Video-of-thought: Step-by-step video reasoning from perception to cognition,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 235, 2024, pp. 13109–13125.

[30] H. Qiu et al., “STEP: Enhancing Video-LLMs’ compositional reasoning by spatio-temporal graph-guided self-training,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 3284–3294.

[31] J. Shin, E. Cho, K. Y. Kim, J. Y. Kim, S. T. Kim, and N. Oh, “Towards holistic surgical scene graphs,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2025, pp. 617–626.

[32] S. Jiang et al., “Hulu-Med: A transparent generalist model toward holistic medical vision–language understanding,” arXiv:2510.08668, 2025.

[33] Qwen Team, “Qwen3.5: Towards native multimodal agents,” Qwen Blog, Feb. 2026. [Online]. Available: https://qwen.ai/blog? id=qwen3.5. Accessed on: Jul. 28, 2026.