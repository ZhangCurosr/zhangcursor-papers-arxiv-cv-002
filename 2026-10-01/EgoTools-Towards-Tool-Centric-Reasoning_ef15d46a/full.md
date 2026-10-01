# EgoTools: Towards Tool-Centric Reasoning

# in Real-World Egocentric Videos

Shulin Tian<sup>1,2,\*</sup> Junsu Kim<sup>1,3,\*</sup> Shuai Liu<sup>1,\*</sup> Hao Li<sup>1,\*,★</sup> Yujiao Shen<sup>1</sup> Sihan Li<sup>1</sup> Zhe Yang<sup>1</sup> Yeongon Kim<sup>1</sup> Feiyu Li<sup>4</sup> Jialin Wu<sup>5</sup> Yichi Zhang<sup>1</sup> Wenhui Wang<sup>6</sup> Runmao Yao<sup>1</sup> Yuhao Dong<sup>1</sup> Zhaoxi Chen<sup>1,★</sup> Fangzhou Hong<sup>1,★</sup> Antonino Furnari<sup>7</sup> Jingkang Yang<sup>1</sup> Hongyuan Zhu<sup>2</sup> Ziwei Liu<sup>1,†,★</sup>

<sup>1</sup>S-Lab, Nanyang Technological University <sup>2</sup>A\*STAR <sup>3</sup>KAIST <sup>4</sup>PKU <sup>5</sup>FDU <sup>6</sup>School of Biological Sciences, Nanyang Technological University <sup>7</sup>University of Catania

<sup></sup> Project Page: ropedia.github.io/egotools

<sup>§</sup> Code: github.com/Ropedia/EgoTools

Dataset: huggingface.co/ropedia-ai/egotools-data

Model: huggingface.co/ropedia-ai/egotools-8b

![](images/32484c2d12300dfe7a0479e86ff7e7262a83d1b3c6699e3f1c091077630e7932.jpg)  
Figure 1: Overview of EgoTools. EgoTools is an egocentric tool-use dataset that spans multiple environments, with dense hierarchical captions and long-sequence 4D object tracking. We also propose EgoTools-Bench, a 1,000-question benchmark spanning diverse real-world environments and reasoning tasks.

Real-world embodied tasks, from everyday activities to professional procedures, require agents to act under physical constraints while tracking evolving object and task states. Tool use sits at the heart of such tasks, as many everyday and professional activities are tool-mediated. Understanding them requires reasoning about affordances, hand–tool–object geometry, procedural progress, and causal effects on target objects. Yet despite strong performance on perception-oriented video tasks such as captioning and general video QA, current multimodal video models remain limited in this form of tool-centric embodied reasoning. Progress in this direction has been limited by the lack of real-world egocentric data and diagnostic benchmarks. To address this gap, we introduce EgoTools, the first comprehensive suite for egocentric tool-use understanding. It consists of two complementary components: EgoTools-Data, a large-scale corpus of 100 hours of tool-centric egocentric recordings with synchronized audio, dense captions, reasoning-heavy narrations, and supplementary 3D information; and EgoTools-Bench, a diagnostic benchmark of 1,000 QA pairs across four tracks that cover tool-use understanding from perception and geometry to procedure and causal reasoning. Experimental results show that current models still struggle to ground tool use in visual evidence: Gemini-3.1-Pro achieves 66.9% overall accuracy but only 51.7% on Perception & Grounding. Beyond evaluation, we validate EgoTools-Data as a training resource. On the full 1,000-question benchmark, full supervised fine-tuning improves Qwen3-VL-8B-Instruct from 50.0% to 60.9%, under strict source-video separation. Together, these results establish EgoTools as a unified resource for both training and diagnostic evaluation of real-world egocentric tool-use understanding.

## 1. Introduction

Human activities in the physical world, from daily chores to professional procedures, involve not only bare-hand manipulation but also the skilled use of tools. For an embodied agent, understanding such activities demands more than simply recognizing visible actions and objects. Such understanding also requires models to reason about tool–target contact, spatial coordination, procedural progress, and the physical outcomes of actions. This tight coupling of low-level perception with high-level temporal and causal awareness makes tool use a distinct challenge for embodied reasoning. The egocentric viewpoint naturally foregrounds this challenge by centering the observations of hands, tools, and their immediate consequences in a continuously evolving task state. From this perspective, the interplay of physical form, functional intent, and procedural dynamics is directly and persistently visible, making egocentric tool use a uniquely demanding and revealing testbed for embodied reasoning.

Despite strong performance on perception-oriented video tasks, current multimodal video models [2, 3, 19, 25, 61] remain limited in egocentric tool-use reasoning. Progress in this important direction has been held back by the lack of real-world egocentric data with dense tool-use annotations and by the absence of diagnostic benchmarks that isolate this reasoning. Most existing egocentric datasets [10, 20, 21, 30, 40, 44] focus on activities, objects, or hand–object interactions. Although tools are often present in these videos, they are rarely foregrounded as the central unit of explanation. In previous benchmarks [8, 9, 11, 23, 32, 37], actions may be labeled or queried at the level of “cook eggs”, while the spatula and pan that mediate the activity go unmentioned, and the tools through which actions unfold are effectively invisible in their annotations. This dual absence creates both an evaluation gap and a supervision gap: current benchmarks do not isolate tool-mediated reasoning, and existing egocentric corpora rarely provide dense training signals that connect tool choice, target-object state, manipulation context, and causal outcomes.

To address this gap, we introduce EgoTools, the first comprehensive suite that brings together dedicated resources for both training and evaluation of egocentric tool-use understanding. At its core, EgoTools comprises two complementary components: EgoTools-Data, a large-scale realworld egocentric corpus designed to provide the rich training signals that current models lack, and EgoTools-Bench, a carefully constructed diagnostic benchmark that isolates the specific reasoning demands of tool-mediated activities for rigorous evaluation. EgoTools-Data comprises 100 hours of tool-centric egocentric recordings spanning diverse everyday and professional tasks, with synchronized audio, dense textual annotations ranging from surface-level captions to narrations that explicitly foreground affordances, procedural structure, and causal effects, as well as supplementary 3D information. Its annotations expose learnable signals for tool choice and substitution, object-state changes, procedural progress, and grounded hand–tool–object interactions. Together, these annotations transform passive video into structured supervision for understanding how humans conduct embodied tasks with tools in the physical world. From this corpus, we construct EgoTools-Bench, a set of 1,000 question–answer pairs that systematically probe a model’s capacity for embodied tool use. Our benchmark consists of four tracks: Affordance & Causality evaluates goal-directed reasoning about tool use, including why a tool is chosen or switched, what it affords, and how actions establish preconditions and produce consequences; Perception & Grounding focuses on directly observable facts such as tool and object identities, attributes, states, and quantities; Procedural Dynamics probes the observable organization of tool-use processes over time, including tool–action ordering, step transitions, workspace staging, and fine-grained manipulation; and Spatial Reasoning evaluates egocentric geometric understanding of hand–tool–object relations, including position, alignment, containment, support, contact, and depth. These tracks provide a comprehensive evaluation of tool-use understanding from perception and geometry to procedure and causal reasoning.

We evaluate representative multimodal video models on EgoTools-Bench. On the qualitycontrolled full 1,000-question benchmark, Qwen3-VL-8B-Instruct reaches 50.0%, indicating that substantial room remains for egocentric tool-use reasoning. We then test whether the annotations in EgoTools-Data constitute actionable supervision rather than serving only as an intermediate source for benchmark construction. Under strict source-video separation, fine-tuning Qwen3- VL-8B-Instruct on instruction data derived from the non-benchmark portion of EgoTools-Data improves accuracy from 50.0% to 60.9%. The model improves on three of the four reasoning tracks, with Spatial Reasoning the exception. Thus, EgoTools not only exposes a substantial gap in current video-language models, but also provides supervision that can partially close this tool-centric reasoning gap.

## 2. Related Work

Egocentric video datasets and benchmarks. Egocentric video provides the actor’s visual evidence, including hands, manipulated objects, surrounding context, and temporally ordered actions [4, 13, 29, 36]. Large-scale datasets such as EPIC-KITCHENS, Ego4D, Ego-Exo4D, Assembly101, MECCANO, and HOI4D have enabled first-person action, object, hand-object, procedural, industrial, and skilled-activity understanding [10, 20, 21, 30, 40, 44]. Recent video-language benchmarks further evaluate long-context QA, temporal grounding, planning, assistance, firstperson reasoning, and long-form egocentric understanding [8, 9, 11, 23, 32, 37, 55, 60]. However, these resources are typically organized around activities, objects, events, or plans, leaving explicit tool-use reasoning under-specified. Closely related instructional-video tasks study detour retrieval and step differences involving ingredients, tools, or techniques [1, 33], but they do not systematically test whether models can justify tool choice, compare feasible substitutes, or adapt manipulation to changing object states. EgoTools-Data pairs egocentric recordings with participant-provided tool-centric narrations and focal-tool grounding. EgoTools-Bench evaluates complementary aspects of tool-use understanding from held-out egocentric videos.

Table 1 | Comparison with representative egocentric datasets and benchmarks. A checkmark indicates that the feature is a central design component; indicates partial support. Signals denote sensor/video inputs and exclude QA text or narrations used for benchmark construction: = RGB/video, = 3D/depth/pose, = inertial signals, and = gaze; non-RGB signals may be available only for subsets of large datasets. Hours are reported total video hours or computed from source-reported clip counts and durations; “–” denotes not applicable or not reported. Embodied interaction benchmarks are included as scope contrast rather than directly comparable egocentric video corpora.
<table><tr><td>Dataset / Benchmark</td><td>Scene</td><td>Video Hours</td><td>#QA</td><td>Signals</td><td>Env.</td><td>Multi- Tool-Centric Narration</td><td>Narr.-Linked Tool Grounding</td><td>QA Benchmark</td><td>Substitution</td><td>Tool Choice / Physical Tool- Use Reasoning</td></tr><tr><td colspan="9">Egocentric Observation Datasets</td><td></td><td></td></tr><tr><td>EPIC-KITCHENS [10]</td><td>Kitchen</td><td>100</td><td></td><td></td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Ego4D [20]</td><td>Real-world</td><td>3,670</td><td>14.5K</td><td>00</td><td>√</td><td>x</td><td>x</td><td>X</td><td>x</td><td>x</td></tr><tr><td>Ego-Exo4D [21]</td><td>Skilled</td><td>1,286</td><td></td><td></td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Assembly101 [44]</td><td>Assembly</td><td>513</td><td></td><td></td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>MECCANO [40]</td><td>Industrial</td><td>6.9</td><td></td><td>o0</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>HOI4D [30]</td><td>Indoor HOI</td><td>44.4</td><td></td><td></td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td colspan="9">- Egocentric Video Reasoning Benchmarks-</td><td></td><td></td></tr><tr><td>EgoSchema [32]</td><td>Real-world</td><td>250+</td><td>5,031</td><td></td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>EgoTaskQA [23]</td><td>Task videos</td><td>14</td><td>40K</td><td></td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>EgoPlan-Bench [8]</td><td>Daily tasks</td><td>一</td><td>4,939</td><td></td><td>x</td><td>x</td><td>x</td><td></td><td>x</td><td>x</td></tr><tr><td>EgoIntent [35]</td><td>Daily tasks</td><td>2.9</td><td>3,014</td><td></td><td>x</td><td>x</td><td>x</td><td>√</td><td>X</td><td>x</td></tr><tr><td>GroundVQA [11]</td><td>Long videos</td><td>736</td><td>303K</td><td></td><td>x</td><td>x</td><td>x</td><td>V</td><td>x</td><td>x</td></tr><tr><td>EgoTempo [37]</td><td>Temporal</td><td>6.3</td><td>500</td><td></td><td>x</td><td>x</td><td>x</td><td>√</td><td>X</td><td>x</td></tr><tr><td colspan="9">- Embodied Interaction Benchmarks</td><td></td><td></td></tr><tr><td>OpenEQA [31]</td><td>Indoor EQA</td><td>一</td><td>1.6K</td><td></td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>RoboVQA [45]</td><td>Robotics</td><td>238</td><td>829 K</td><td></td><td>X</td><td>x</td><td>x</td><td>√</td><td>X</td><td>x</td></tr><tr><td>RoboCasa365 [34]</td><td>Sim kitchens</td><td>2,200+</td><td></td><td></td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>X</td></tr><tr><td>EgoTools (Ours)</td><td>Real-world</td><td>100</td><td>1,000</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Physical tool-use reasoning. Physical tool use requires reasoning beyond object categories: a model must connect a tool’s function and affordances to the current goal, target material, contact and motion constraints, and evolving object state. Robotics has studied related problems through tool-use surveys, task-oriented grasping, tool-flow prediction, affordance-centric manipulation, and egocentric affordance learning [12, 27, 39, 43, 54]. These works are valuable for execution, but they often focus on structured tasks, constrained manipulation, or short interaction episodes. A parallel line evaluates MLLMs and VLMs on affordance grounding and physical tool understanding [22, 38, 57, 59]; many such settings rely on static images, 3D scenes, synthetic data, isolated interactions, or robotics setups. EgoTools-Bench complements both lines with human-centered first-person evidence, where real tools are selected, substituted, and adapted within continuous tasks. Accordingly, EgoTools evaluates whether models can connect visual evidence to functional choices, feasible alternatives, temporal manipulation steps, and changes in object state.

![](images/5f5cf23c98f8271143ff5fc22f9e26bfc6582917248d65d57b395cf5c72f2ac7.jpg)

![](images/a6541a211a8e9d15cd63c26bdc6bda52d6ec206fadb1b2381e2b40145b3fc551.jpg)

![](images/038c0e8eeb4769cd9b2b62d136d89de3d82824ac1b79bdaee23fd4532d423324.jpg)  
Figure 2 | Benchmark distribution and annotation examples. (a) Distribution of 1,000 QA pairs across four tool-use reasoning tracks and their fine-grained subtracks. (b) Example annotations, including captions, tool-centric narrations, and finalized QA pairs produced through human annotation with MLLM-assisted checking. (c) Word-cloud comparison with EPIC-KITCHENS-100 and Ego4D, showing that EgoTools-Bench is more tool-dense and centered on tools, actions, objects, and state changes.

## 3. EgoTools Data Suite

This section introduces EgoTools, a real-world egocentric video suite for physically grounded tool-use understanding. We first describe the collection and preprocessing of EgoTools-Data, a 100-hour real-world egocentric video corpus captured with synchronized video, audio, and geometry-related signals. We then introduce its multimodal annotation pipeline, including hierarchical dense captions, tool-centric narrations with 2D grounding, and 3D annotations derived from reconstruction and long-horizon object tracking. Finally, we describe two complementary resources derived from EgoTools-Data and separated at the source-video level: (i) an EgoTools-derived video instruction-tuning corpus constructed from the training pool, and (ii) EgoTools-Bench, a 1,000-question 8-way multiple-choice diagnostic benchmark constructed from a curated 40.34-hour benchmark-reserved pool. EgoTools-Bench comprises 900 human-crafted questions and 100 human-verified spatial questions across four tracks: Affordance & Causality, Perception & Grounding, Procedural Dynamics, and Spatial Reasoning.

## 3.1. Data Collection and Preprocessing

We collect raw egocentric videos with HOMIE<sup>1</sup>, a lightweight head-mounted multimodal recording device designed with four synchronized fisheye camera views, together with audio and auxiliary motion/synchronization signals that support downstream spatial processing. These signals support 3D processing, including reconstruction, pose/depth estimation, and object/hand tracking. For annotation and training, we use the rectified front-left RGB stream as the canonical view, applying zoom and pitch adjustments to obtain a natural first-person perspective while preserving hand–tool–object interactions. Rectification details and examples are provided in Appendix D.1. The final videos are 1024 × 1024, 20 FPS, single-view egocentric streams synchronized with audio.

Our data collection is organized around two broad contexts: daily activities and expertiseintensive procedures. Within these contexts, we collect videos across seven tool-use domains: kitchen, classroom, research lab, repair workshop, craft, office, and household, shown in Figure 1. These domains cover diverse tool-mediated activities, from everyday manipulation to craft, scientific, and fabrication procedures. Household recordings capture everyday tool use, craft and repair-workshop recordings emphasize material transformation and manual operations, and laboratory recordings include both educational and professional experiments requiring specialized knowledge and expert tool handling. Data collection was conducted across multiple kitchens, workshops, laboratories, and daily-living spaces at universities and research sites in Asia. Our data collectors include graduate and undergraduate students from diverse disciplinary backgrounds, with domain expertise matched to the task whenever specialized knowledge is required. In particular, expertise-intensive recordings are performed or reviewed by collectors familiar with the corresponding procedures, ensuring that the captured tool use is both natural and technically valid. Details of the collection domains and anonymized participant information are provided in Appendix C.

The collected and curated EgoTools-Data corpus contains approximately 100 hours of egocentric video across the seven tool-use domains described above. To balance controlled task coverage with natural tool-use behavior, we adopt two complementary collection settings: structured task-guided and open-ended participant-driven. In the structured task-guided setting, participants follow predefined task sequences prepared by the data collection team. Each sequence specifies a task goal and key procedural steps, yielding clear task boundaries, observable task-state changes, and controlled coverage of hand–tool–object interaction. In the open-ended participant-driven setting, participants are given only a broad topic or high-level goal and complete the task in their own manner. This setting allows spontaneous tool choices, procedural adaptation, repeated attempts, error recovery, and opportunistic substitutions. Together, the structured setting provides consistent procedural data for reliable annotation and benchmark construction, while the participant-driven setting captures the variability and adaptivity of in-the-wild tool use.

Source-video partition. Before constructing the instruction-tuning corpus and benchmark, we partition the recordings at the source-video level into a training pool and a benchmark-reserved pool. The 40.34-hour benchmark pool is held out from all stages of instruction-data construction. Consequently, no clip, caption, narration, synthetic QA, or other annotation derived from a benchmark source video is included in model training.

## 3.2. Multimodal Annotation and Curation

Textual Annotations. We provide two complementary textual annotations: dense captions and tool-centric narrations. Dense captions describe visible actions and task progress across temporal scales, while narrations capture participant intent, tool grounding, and tool-use reasoning. For dense captioning, we use Gemini-3-Flash [17] in a streaming hierarchical pipeline: each video is split into 5-minute clips, with 32 uniformly sampled frames used to generate a global context caption. Each clip is then captioned sequentially in 5-second chunks, where the first chunk uses the global caption and later chunks use the previous caption as memory for temporal consistency. Chunk captions are further aggregated into 1-minute windows and 5-minute summaries. In parallel, object recognition on key frames provides visual evidence for post-hoc hallucination checks. Following common practice in egocentric video benchmarks, where narrations serve as weak supervision for first-person activities [10, 20], we also collect narrations for EgoTools. Unlike prior datasets that encourage broad descriptions of visible actions, our narrations are explicitly tool-centric: collectors who perform the tasks narrate meaningful tool-use events using a reference narration template, emphasizing tool selection, usage intent, and effects on target objects or task states. For selected narrations, collectors annotate 2D grounding points on corresponding keyframes for focal tools or objects chosen for tracking, linking textual mentions to visual instances and providing initialization cues for long-horizon 3D tracking. The dense captions and corrected English narrations are subsequently converted into temporally grounded video instruction examples, providing supervision for action description, procedural summarization, tool selection and substitution, object-state tracking, and next-step prediction. The textual annotation interfaces are shown in Figure 3. Additional details on the narration template and examples are provided in Appendix D.2.

![](images/c29fb8bee6e96deea4206f388e3eedbfc91da74bab344571bf0a6a01f8d63bd7.jpg)  
Figure 3 | Textual annotation pipeline. (a) Tool-centric narrations are annotated from grounded egocentric video segments. (b) Benchmark QA pairs are constructed and refined through an annotation interface with MLLM-assisted checking.

3D Annotations. As shown in Figure 4, we collect raw geometric data with the HOMIE device from Ropedia, Inc., including egocentric videos, camera poses, and depth maps. Following Holi-Spatial [16], we train a 3D Gaussian Splatting model [28] to build a multi-view consistent scene geometry, reduce artifacts such as floaters, and render high-fidelity, temporally coherent depth maps for each frame.

For robust long-term object tracking, we introduce a recursive VLM-guided segmentation framework. Tool-centric narrations provide initial 2D grounding points and semantic captions, from which SAM2 [42] propagates object masks through the video. When later frames contain additional grounding points, a VLM [19] verifies tracking consistency. If drift or semantic mismatch is detected, the framework re-initializes SAM2 from the failure point and refines

![](images/418c4852747bb1e1347b0a3d192c6f3fa9622f2574f920485b8045060ea9d64f.jpg)  
Figure 4 | 3D annotation pipeline. Raw geometry from the HOMIE device is reconstructed with 3DGS to obtain consistent depth, then combined with tool-centric narrations for recursive VLM-guided object tracking. The resulting masks are lifted to 3D for object boxes and spatial QA generation.

the segmentation. This feedback loop yields spatio-temporal masks with both geometric precision and long-range semantic coherence. Finally, we lift the tracked masks into 3D using the reconstructed depth and camera poses, enabling object boxes and long-horizon 4D object trajectories. Based on these annotations, we design spatial QA templates that query object motion and spatial relations. Together, the original 100-hour egocentric videos, dense textual annotations, and 3D annotations constitute EgoTools-Data, from which we derive a video instruction-tuning corpus and a source-video-disjoint EgoTools-Bench evaluation set.

## 3.3. Video Instruction-Tuning Data Construction

To evaluate whether EgoTools-Data provides actionable training supervision, we construct a video instruction-tuning corpus exclusively from the EgoTools training pool. Every training instance is derived from EgoTools videos and their associated dense captions, corrected English narrations, and temporal metadata. The instruction-tuning corpus contains 184,679 examples, and every example is labeled with the capability it supervises. Grounding-oriented supervision asks what is visible and where: which tool is in use, which object it contacts, what attribute or local state has changed, and how hands, tools, and objects are arranged in space. Reasoningoriented supervision asks why and in what order: why a tool is chosen for a material or goal, what the causal effect of an action is, how a task progresses through its steps, and what happens next. A third group provides descriptive supervision through dense captioning and narration completion, which teaches the model to verbalize first-person visual evidence without a question format.

Concretely, 69,760 examples (37.8%) supervise the four EgoTools-Bench abilities directly, split into affordance and causality (14,175), perception and grounding (14,755), procedural dynamics (22,943), and spatial reasoning (17,887). A further 69,554 examples (37.7%) are episode-level multiple-choice questions about action ordering, overall activity, and task goals, including 5,000 next-action questions that pair a video with the current observation frame. General video QA contributes 31,281 examples (16.9%) of mixed episode-level questions. The remainder is descriptive or open-ended: 9,084 examples (4.9%) of dense captioning and narration completion, and 5,000 (2.7%) of single-image open-ended QA. Synthetic QA candidates are filtered for visual answerability, temporal grounding, single-best-answer validity, and resistance to textual shortcuts, and answer letters are balanced within each option-count group to reduce positional bias. Figure 5 summarizes the resulting distribution.

## 3.4. Benchmark Construction and Quality Control

Benchmark QA Construction. Based on EgoTools-Data, we select a high-quality 40.34-hour benchmark-reserved subset for EgoTools-Bench construction, prioritizing annotation quality, tool-use density, task diversity, and environmental coverage. The benchmark and instructiontuning corpus are disjoint at the source-video level. We manually create QA pairs with both original data collectors and external annotators: collectors contribute task-aware procedural and causal questions, while external annotators identify visually inferable but model-challenging reasoning points. Domain-specific videos, such as wet-lab procedures, are assigned to annotators with relevant expertise. Annotators identify segments requiring physically grounded tool-use reasoning, including tool selection, object-state changes, and temporal ordering of tool-mediated actions, and formulate 8-way multiple-choice questions with one correct answer and seven distractors. Each benchmark clip is extracted as a localized temporal window around an anchor moment, with most clips lasting at least three minutes. We limit temporal overlap and control QA density across source videos to maintain diversity over videos, tools, and task contexts. This yields 900 human-crafted QA pairs. We further generate 100 spatially grounded QA pairs from 3D annotations using the Holi-Spatial [16] pipeline, augmented with focal-tool trajectories and object-state signals. These questions target spatial motion, state changes, and tool–object interaction dynamics, and are included only after human verification. Overall, the average temporal span per QA is 4.19 minutes.

Quality Control. All benchmark items undergo a fix-first quality-control pipeline that repairs valid annotator intent before removal. We audit structural validity, video grounding, and answer-choice quality to detect malformed questions, duplicates, empty submissions, missing visual evidence, temporal misalignment, inconsistent tool identity, and weak distractors. Empty or verbatim-duplicate items are removed; repairable issues are fixed with deterministic rules or minimal model-assisted edits, including reference normalization, first-person leakage re moval, and unambiguous answer completion. Low-confidence cases are sent to human review. After repair, questions are normalized to an 8-way multiple-choice format and screened for shortcuts such as near-duplicate choices, revealing wording, lexical imbalance, answer-length or position bias, and text-only solvability. Text-only and multi-model sanity checks are used to flag potentially trivial or shortcut-solvable items for review. Flagged items are hardened by replacing distractors with visually plausible, length-balanced alternatives while preserving the original question and ground-truth answer when possible. Final revision and retention decisions are based on visual grounding, unique answerability, and annotation quality rather than the correctness of any particular model. Details are in Appendix F.

Independent Reliability Study. To quantitatively assess annotation reliability, we conducted an independent study on a stratified sample of 250 benchmark questions, including 100 Affordance & Causality questions. Five annotators who had not written the original questions each evaluated 100 assigned items, such that every question received two independent judgments. Annotators viewed only the video, question, and answer choices without access to the original answer key. Across 500 judgments, agreement with the original key was 89.8%, including 86.5% on Affordance & Causality questions. Exact agreement between the two assigned annotators was 88.4% at the item level, and 94.4% of items were judged to have a uniquely best answer supported by visible evidence. Fourteen items (5.6%) were flagged as ambiguous or having multiple plausible answers; all were manually reviewed, and items failing the unique-answer criterion were revised or replaced. All results reported in this paper were recomputed on the resulting quality-controlled benchmark.

## 3.5. Distribution

Figure 2 summarizes the composition of EgoTools-Bench, which contains 1,000 QA pairs across four tracks: 1) Affordance & Causality (363), 2) Perception & Grounding (236), 3) Procedural Dynamics (222), and 4) Spatial Reasoning (179). These tracks cover core aspects of egocentric tool-use understanding, including tool recognition, object-state grounding, hand– tool–object relations, task progress, and the causal effects of tool use. This distribution reflects the goal of EgoTools-Bench: evaluating tool use as a reasoning problem involving tool selection, application, and effects, while retaining coverage of perception, procedure, and spatial reasoning. Each track is further divided into fine-grained subtracks for detailed analysis of model strengths and failure modes. The seven collection domains characterize the coverage of the full EgoTools-Data corpus rather than a domain-balanced benchmark. EgoTools-Bench exhibits a strongly long-tailed distribution across domains, with the majority of questions drawn from Kitchen videos; complete domain-level statistics are provided in Appendix E.1. We therefore focus our primary analysis on reasoning capabilities rather than domain-level performance. To avoid over-representing individual recordings, we limit temporal overlap among benchmark clips and cap QA density per source video. Figure 2 also compares the vocabulary distribution of EgoTools-Bench with EPIC-KITCHENS-100 and Ego4D. The word clouds qualitatively illustrate that EgoTools-Bench places greater lexical emphasis on tools, manipulated objects, actions, and state changes than on general egocentric activity recognition.

## 4. Experiments

## 4.1. Experimental Setup

We evaluate EgoTools-Bench with human experts, proprietary models, and open-source models, and compare model performance with existing egocentric video benchmarks including EgoSchema [32], EgoPlan [8], and EgoThink [9]. The model set covers standard video-language models, audio-capable omni models, and thinking-style variants from Gemini [19], Qwen [2, 3, 53, 50], InternVL [51, 61], LLaVA [25, 58], MiMo-VL [48], and GLM [49] families. For EgoTools-Bench, we report overall accuracy and track-wise accuracy following the four tracks introduced in Section 3.5; existing benchmark results are reported when available under comparable settings. For each model, we report the number of input frames and whether audio is used. Most opensource instruct models use 64 uniformly sampled frames; thinking models use 64 or 512 frames depending on their supported setting; Gemini [19] models use 1 FPS video input. Audio-enabled models receive only synchronized non-narration audio. Corrected narrations, narration audio, dense captions, and annotation metadata are never provided as input.

Human evaluation. Human performance was measured using five evaluators who had not authored the questions they evaluated. Each evaluator answered all 1,000 benchmark questions over multiple sessions and could freely seek and replay the videos. Evaluators did not have access to the answer key, and there were no skipped or invalid responses. Overall and track-wise accuracies are micro-averaged over 5,000 individual judgments. The human results therefore reflect a free-viewing condition rather than the fixed-frame input budget used for model evaluation.

EgoTools training. To evaluate the training utility of EgoTools-Data, we adapt Qwen3-VL-8B-Instruct with a single stage of full-parameter supervised fine-tuning of its language-model component, while keeping the visual encoder and multimodal aligner frozen. We train on the 184,679-example EgoTools instruction corpus described in Section 3.3. No videos, images, or QA annotations from EgoSchema, EgoPlan-Bench, or EgoThink are used, and all training videos are disjoint from EgoTools-Bench at the source-video level. Optimization and input settings are provided in Appendix H.

## 4.2. Main Results

EgoTools-Bench is challenging for current video-language models. Table 2 summarizes performance across human experts, proprietary models, open-source instruct models, and open-source thinking models. EgoTools-Bench uses an 8-way multiple-choice format, so chance performance is 12.5%. Across open-source instruct models, performance remains low, with overall accuracy ranging from 38.9% to 50.0%, far below the human expert score of 83.2%. The strongest open-source instruct result is Qwen3-VL-8B-Instruct [2] at 50.0%, followed by InternVL3.5-8B-Instruct [51] at 48.7% and Qwen2.5-Omni-7B-Instruct [53] at 47.1%. Among open-source thinking models, GLM4.1V-Thinking [49] achieves the strongest overall result at 50.7%. These results indicate that EgoTools-Bench exposes a substantial gap in current models’ ability to reason about tool-mediated actions, rather than an isolated weakness of one architecture. The final row reports EgoTools-8B, our model fine-tuned on EgoTools-Data; it is listed separately from the zero-shot model baselines and analyzed in §4.3.

Performance across egocentric benchmarks. Table 2 also reports results on existing egocentric benchmarks to contextualize the evaluated models. For example, Qwen3-VL-8B-Instruct reaches

Table 2 | Performance comparison across egocentric video understanding benchmarks. We compare human performance, proprietary models, and open-source video-language models on three existing egocentric benchmarks and our EgoTools-Bench benchmark. The last five columns report results on EgoTools-Bench across four research-facing tracks and the overall average. Track abbreviations are: AC = Affordance & Causality, PG = Perception & Grounding, PD = Procedural Dynamics, and SR = Spatial Reasoning.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Frames</td><td rowspan="2">Audio</td><td rowspan="2"></td><td rowspan="2">EgoSchema [32] EgoPlan [8] EgoThink [9]</td><td rowspan="2"></td><td colspan="5">EgoTools</td></tr><tr><td>AC</td><td>PG</td><td>PD</td><td>SR</td><td>Overall</td></tr><tr><td colspan="10">Human Baseline</td></tr><tr><td>Human Expert</td><td>一</td><td>√</td><td>~76</td><td>一</td><td></td><td>82.1</td><td>85.2</td><td>83.3</td><td>82.7</td><td>83.2</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Gemini-3.1-Pro [19]</td><td>1fps</td><td>√</td><td></td><td></td><td></td><td>69.7</td><td>51.7</td><td>72.1</td><td>74.9</td><td>66.9</td></tr><tr><td>Gemini-3-Flash [17]</td><td>1fps</td><td>√</td><td>一 一</td><td>一 一</td><td>一 一</td><td>64.5</td><td>50.0</td><td>73.0</td><td>72.6</td><td>64.4</td></tr><tr><td>Gemini-3.1-Flash-Lite [18]</td><td>1fps</td><td>√</td><td>1</td><td></td><td>一</td><td>58.1</td><td>44.5</td><td>59.9</td><td>58.7</td><td>55.4</td></tr><tr><td colspan="10">Open-Source Models (Instruct)</td></tr><tr><td>Qwen3-VL-8B-Instruct [2]</td><td>64</td><td>x</td><td>69.0</td><td>42.3</td><td>61.0</td><td>46.1</td><td>45.9</td><td>57.1</td><td>55.4</td><td>50.0</td></tr><tr><td>Qwen3-VL-4B-Instruct [2]</td><td>64</td><td>x</td><td>69.6</td><td>42.0</td><td>61.9</td><td>41.7</td><td>45.4</td><td>50.2</td><td>50.0</td><td>45.7</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [3]</td><td>64</td><td>x</td><td>61.0</td><td>32.0</td><td>57.7</td><td>40.1</td><td>43.3</td><td>50.2</td><td>47.8</td><td>44.3</td></tr><tr><td>Qwen2.5-Omni-7B-Instruct [53]</td><td>64</td><td>√</td><td>65.2</td><td>一</td><td>58.9</td><td>44.7</td><td>45.9</td><td>51.8</td><td>46.7</td><td>47.1</td></tr><tr><td>Qwen2-VL-7B-Instruct [50]</td><td>64</td><td>x</td><td>63.6</td><td>39.9</td><td>60.1</td><td>46.6</td><td>35.6</td><td>46.2</td><td>47.8</td><td>44.2</td></tr><tr><td>InternVL3.5-8B-Instruct [51]</td><td>64</td><td>x</td><td>63.8</td><td>一</td><td>56.0</td><td>45.8</td><td>45.9</td><td>53.4</td><td>53.3</td><td>48.7</td></tr><tr><td>InternVL3-8B-Instruct [61]</td><td>64</td><td>x</td><td>69.6</td><td>一</td><td>59.4</td><td>44.7</td><td>44.9</td><td>51.8</td><td>48.9</td><td>47.1</td></tr><tr><td>LLaVA-OneVision-7B [25]</td><td>64</td><td>x</td><td>63.4</td><td>一</td><td>54.9</td><td>37.6</td><td>37.1</td><td>42.9</td><td>37.0</td><td>38.9</td></tr><tr><td>LLaVA-Video-7B-Qwen2 [58]</td><td>64</td><td>x</td><td>52.4</td><td>1</td><td>58.0</td><td>38.2</td><td>39.7</td><td>43.3</td><td>37.0</td><td>39.8</td></tr><tr><td>MiMo-VL-7B-SFT [48]</td><td>64</td><td>x</td><td>54.4</td><td>35.9</td><td>60.7</td><td>41.4</td><td>44.9</td><td>49.4</td><td>40.2</td><td>44.2</td></tr><tr><td colspan="10">Open-Source Models (Thinking)</td></tr><tr><td>Qwen3-VL-8B-Thinking [2]</td><td>512</td><td>x</td><td>70.3</td><td>43.6</td><td>62.0</td><td>43.6</td><td>41.8</td><td>48.2</td><td>55.4</td><td>45.7</td></tr><tr><td>Qwen3-VL-4B-Thinking [2]</td><td>512</td><td>x</td><td>59.8</td><td>43.1</td><td>62.3</td><td>37.6</td><td>36.6</td><td>36.4</td><td>41.3</td><td>37.4 45.3</td></tr><tr><td>MiMo-VL-7B-RL [48]</td><td>64</td><td>x</td><td>57.0</td><td>36.6 25.3</td><td>59.4</td><td>43.1</td><td>45.9</td><td>50.6</td><td>39.1</td><td></td></tr><tr><td>GLM4.1V-Thinking [49]</td><td>64</td><td>x</td><td>66.1</td><td></td><td>62.7</td><td>46.1</td><td>50.0</td><td>57.1</td><td>53.3</td><td>50.7</td></tr><tr><td colspan="10">Ours</td></tr><tr><td>EgoTools-8B (Ours)</td><td>64</td><td>x</td><td>67.6</td><td>35.2</td><td>64.0</td><td>55.4</td><td>59.7</td><td>68.1</td><td>44.4</td><td>60.9</td></tr></table>

69.0% on EgoSchema [32] and 61.0% on EgoThink [9], compared with 50.0% on EgoTools-Bench. Qwen3-VL-4B-Instruct shows a similar pattern, with 69.6% on EgoSchema, 61.9% on EgoThink, and 45.7% on EgoTools-Bench. Thus, broad first-person activity understanding does not necessarily imply robust tool-use reasoning about affordances, grounding, procedures, spatial relations, and state changes.

Additional model and input capacity do not close the gap. Overall accuracy varies across models with different parameter scales, audio support, and thinking-style inference. We examine these differences by track and model class in §4.4, then by tool-use cues and error patterns in §4.5. These results motivate a complementary question: whether the missing tool-centric capabilities can be learned from the supervision provided by EgoTools-Data.

## 4.3. Training Utility of EgoTools-Data

EgoTools supervision improves tool-centric reasoning. Under the standard 64-frame MP4- based evaluation protocol, single-stage full supervised fine-tuning on 184,679 EgoTools-derived instruction examples improves Qwen3-VL-8B-Instruct from 50.0% to 60.9% on EgoTools-Bench, a gain of 10.9 percentage points. The improvement covers three of the four tracks: +9.3 points on Affordance & Causality, +13.8 on Perception & Grounding, and +11.0 on Procedural Dynamics, while Spatial Reasoning decreases by 11.0 points. Because the training and evaluation resources are separated at the source-video level, these gains reflect generalization to held-out tool-use episodes rather than memorization of benchmark clips. The result demonstrates that EgoTools-Data is not merely an intermediate source for benchmark construction, but provides actionable supervision for learning egocentric tool-use reasoning. Despite explicit spatial supervision, SR performance decreases after fine-tuning, indicating that spatial reasoning remains a limitation of the current training recipe.

![](images/8498065aaad6be1daf9934bb8de2bf592a3e586a5fc0143fc8b843bda3a1f3d9.jpg)  
(a)

![](images/e204aa4a72712e97b3aab0322f70962beceea51912bb7870b75c1feb2c9653c2.jpg)  
Figure 5 | Composition of the EgoTools-Data instruction-tuning corpus. (a) The inner ring groups the 184,679 examples by what they supervise, and the outer ring gives the corresponding categories; segment angles are schematic rather than strictly proportional, so that the smallest categories remain readable. (b) Shares drawn to scale; the four tool-centric categories are nested under their group total. Exact numbers are listed in Table 10.

Performance on EgoSchema. The single-stage model reaches 67.6% on EgoSchema, 1.4 points below the 69.0% backbone.

The EgoPlan and EgoThink results are consistent with answer-format coverage. Beyond multiple-choice video QA, the training mixture covers two further answer formats built from EgoTools training videos: 5,000 next-action multiple-choice examples that pair a video with the current observation frame, and 5,000 single-image open-ended examples. These formats match how EgoPlan-Bench and EgoThink pose their questions. On EgoThink, the model reaches 64.0% macro accuracy, exceeding the 61.0% backbone by 3.0 points, whereas on EgoPlan-Bench it reaches 35.2%, still 7.1 points below the 42.3% backbone. Because these training subsets mirror the answer formats used by EgoPlan-Bench and EgoThink, these results do not isolate the effects of format coverage from other effects of fine-tuning. We therefore limit our primary claim to the training utility of EgoTools-Data for tool-centric reasoning rather than claiming uniform improvement across all egocentric benchmarks.

## 4.4. Analysis

Perception & Grounding is the diagnostic anchor. Perception & Grounding (PG) is particularly dependent on concrete visual evidence because it asks about the focal tool, target object, contact point, object attribute, quantity, or local state change. Activity context alone rarely determines which visually similar tool is being used, which object it contacts, or what has changed locally. Open-source instruct models score between 35.6% and 45.9% on PG, and even

Table 3 | Training utility of EgoTools-Data. EgoTools-8B is Qwen3-VL-8B-Instruct after a single stage of full supervised fine-tuning on 184,679 EgoTools-derived instruction examples. On EgoTools-Bench, both models use the same standard 64-frame MP4-based evaluation protocol. The instruction-tuning corpus and EgoTools-Bench are disjoint at the source-video level. EgoTools-Bench results are reported on the full quality-controlled 1,000-question benchmark; EgoSchema is reported on the full 5,031-question set, EgoPlan-Bench on its 3,343-question validation set, and EgoThink as the macro-averaged score over its twelve single-image subtasks.
<table><tr><td rowspan="2">Model</td><td colspan="5">EgoTools-Bench</td><td colspan="3">Existing Egocentric Benchmarks</td></tr><tr><td>AC</td><td>PG</td><td>PD</td><td>SR</td><td>Overall</td><td>EgoSchema</td><td>EgoPlan</td><td>EgoThink</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>46.1</td><td>45.9</td><td>57.1</td><td>55.4</td><td>50.0</td><td>69.0</td><td>42.3</td><td>61.0</td></tr><tr><td>EgoTools-8B</td><td>55.4</td><td>59.7</td><td>68.1</td><td>44.4</td><td>60.9</td><td>67.6</td><td>35.2</td><td>64.0</td></tr></table>

Table 4 | Input-evidence ablation on the full 1,000-question EgoTools-Bench. All conditions use the same Qwen3-VL-8B-Instruct backbone, deterministic decoding, and scoring configuration. The visual conditions use the same pre-extracted frame-list pipeline, with the ordered condition serving as the matched within-protocol control.
<table><tr><td>Input condition</td><td>Overall</td><td>AC</td><td>PG</td><td>PD</td><td>SR</td></tr><tr><td>Ordered 64 frames</td><td>51.7</td><td>47.4</td><td>50.0</td><td>57.1</td><td>57.6</td></tr><tr><td>Shuffled 64 frames</td><td>49.3</td><td>45.5</td><td>47.4</td><td>53.9</td><td>56.5</td></tr><tr><td>Middle frame</td><td>46.9</td><td>44.7</td><td>45.9</td><td>51.0</td><td>45.7</td></tr><tr><td>Text-only</td><td>42.7</td><td>39.0</td><td>36.1</td><td>51.4</td><td>47.8</td></tr></table>

Gemini-3.1-Pro reaches 51.7%, still 18–23 points below its AC, PD, and SR scores. Together with the input ablation in Table 4, where PG shows the largest improvement from visual evidence over text-only input, these results identify precise visual grounding as a persistent bottleneck.

Model variants exhibit different track-wise strengths. The higher-level tracks admit different sources of contextual support: tool-function priors for AC, task-order context for PD, and workspace geometry and physical commonsense for SR. Qwen3-VL-8B-Instruct improves over its 4B counterpart most strongly on PD and SR, by 6.9 and 5.4 points, respectively, while improving PG by only 0.5 points. For this model pair, the performance difference is larger on procedural and spatial reasoning than on precise grounding. The omni and thinking variants also show mixed patterns: Qwen2.5-Omni-7B-Instruct improves over Qwen2.5-VL-7B-Instruct on AC, PG, and PD but scores slightly lower on SR, while Qwen3-VL-8B-Thinking underperforms its instruct counterpart on AC, PG, and PD and only matches it on SR. In contrast, GLM4.1V-Thinking achieves the strongest zero-shot open-source result in Table 2, at 50.7%. These heterogeneous patterns characterize the evaluated configurations; because the variants differ in training and sometimes input budgets, they do not isolate the effects of scale, audio, or deliberation.

## 4.5. Ablations and Error Analysis

EgoTools-Bench requires visual evidence beyond language priors. Table 4 compares temporally ordered 64-frame input with the same frames in shuffled order, a single middle frame, and text-only input on the full 1,000-question benchmark. All conditions yield zero parsing or inference failures. Ordered video achieves 51.7% overall accuracy, compared with 42.7% for text-only input, corresponding to a 9.0-point improvement. The substantial gain from visual input confirms that the benchmark cannot be solved solely through answer-choice patterns, toolfunction commonsense, or language priors. Text-only performance nevertheless remains above the 12.5% chance level. This indicates that some questions retain exploitable commonsense or answer-option cues, despite the benchmark’s anti-shortcut filtering. We therefore interpret text-only performance as evidence of residual language priors rather than as evidence that video input is unnecessary.

Multi-frame coverage is important. Using only the middle frame reduces overall accuracy from 51.7% to 46.9%, showing that a single static observation is insufficient for many questions. The 4.8-point gap indicates that models benefit from observing multiple moments of the tool-use process, including the selection of a tool, its interaction with a target object, intermediate state changes, and the resulting task outcome. The effect is particularly strong for Spatial Reasoning, where ordered multi-frame input outperforms the middle-frame condition by 11.9 points. This result suggests that spatial questions frequently require integrating evidence distributed across the clip rather than reading a single hand–tool–object configuration.

Temporal order provides additional information. Shuffling the same 64 frames reduces overall accuracy from 51.7% to 49.3%. Because the visual content and frame count remain unchanged, the 2.4-point difference isolates the contribution of temporal ordering. Procedural Dynamics is the most order-sensitive track, with ordered input outperforming shuffled input by 3.2 points. This is consistent with the track’s emphasis on step transitions, action ordering, preparation, and changes in tool use over time. By contrast, Spatial Reasoning shows only a 1.1-point difference between ordered and shuffled frames. Together with its large orderedversus-middle-frame gap, this pattern indicates that SR benefits primarily from multi-frame coverage, while depending less strongly on the precise temporal sequence of those frames.

The benchmark tracks require complementary evidence. The ablation effects differ systematically across the four tracks. Perception & Grounding shows the largest dependence on visual evidence, improving by 13.9 points over text-only input. Procedural Dynamics is the most sensitive to temporal order, while Spatial Reasoning benefits most from multi-frame coverage. These complementary patterns support the intended diagnostic roles of the benchmark tracks: PG tests grounding in concrete visual evidence, PD tests temporally ordered process understanding, and SR tests the integration of spatial evidence across multiple observations. All ablation conditions use the same deterministic frame-list pipeline. The ordered condition therefore serves only as the matched control for comparisons within this ablation. Its absolute score is not directly compared with the separate main benchmark result obtained using the standard MP4-based evaluation protocol.

Implicit tool-use questions are generally harder. Table 5 separates questions that explicitly mention the relevant tool-use cue from questions where the model must infer it from the video context. Proprietary Gemini models show a consistent advantage on explicit questions, with explicit-minus-implicit gaps of 4.9– 7.7 points. Qwen3-VL-8B-Instruct shows the same direction with a

Table 5 | Performance on implicit and explicit tool-use questions. Δ denotes Explicit−Implicit.
<table><tr><td>Model</td><td>Full Bench.|</td><td>Implicit</td><td>Explicit</td><td>Δ</td></tr><tr><td>Gemini-3.1-Pro</td><td>66.9</td><td>65.2</td><td>72.9</td><td>+7.7</td></tr><tr><td>Gemini-3-Flash</td><td>64.4</td><td>63.1</td><td>68.0</td><td>+4.9</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>55.4</td><td>54.3</td><td>59.5</td><td>+5.2</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>50.0</td><td>49.1</td><td>53.1</td><td>+4.0</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>45.7</td><td>45.8</td><td>45.2</td><td>-0.6</td></tr></table>

smaller 4.0-point gap. Because the explicit and implicit subsets contain different questions, these gaps may also reflect differences in their content and difficulty. Qwen3-VL-8B-Thinking is the exception, with a small negative gap of 0.6 points. Implicit questions provide a useful subset for evaluating tool-use understanding when the relevant cue is not explicitly stated in the question.

Error overlap reveals shared failures. The error map in Figure 6 compares the Jaccard overlap between models’ incorrect predictions. The strongest off-diagonal overlap is between Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Thinking (0.63), showing substantial shared errors between these two variants. Gemini models also share a moderate error profile with one another, especially Pro and Flash (0.53), but their overlap with Qwen3-VL-8B is lower (0.35–0.38 for Pro/Flash). Gemini-3.1-Flash-Lite has higher overlap with Qwen3-VL-8B-Instruct (0.50) and Qwen3-VL-8B-Thinking (0.46). Overall, the overlap pattern supports that failures are systematic and partially model-family-specific: thinkingstyle variants retain many failures of their instruct counterparts, while proprietary and opensource models exhibit more complementary error sets. Taken together, the track-wise results, input ablations, implicit-explicit comparison, and error-overlap analysis provide complementary views of model performance on EgoTools-Bench. These analyses help diagnose weaknesses in tool-use understanding across reasoning tracks, input conditions, and question subsets.

![](images/284165c660d5b16b7d118a34885a16db5c3b59eb7aa4f5b26525c96b13c384d9.jpg)  
Figure 6 | Jaccard overlap of model errors on EgoTools-Bench. This highlights shared failure patterns within and across model families.

## 5. Conclusion

We introduce EgoTools, a tool-centric egocentric video suite that reframes embodied video understanding around the physical instruments through which humans act, adapt, and transform the world. By combining real-world tool-use recordings, dense hierarchical annotations, long-horizon spatial grounding, and a diagnostic QA benchmark, EgoTools exposes a key limitation of current multimodal video models: recognizing activities is not sufficient to understand how tools mediate action, causality, procedure, and state change. At the same time, supervised fine-tuning on EgoTools-derived instruction data improves Qwen3-VL-8B-Instruct from 50.0% to 60.9% on EgoTools-Bench, demonstrating that the proposed data provides actionable supervision for tool-centric reasoning. The gains span three of the four tracks, while Spatial Reasoning remains a limitation of the current training recipe. Nevertheless, a substantial gap to human performance remains, particularly in grounding tool-mediated interactions in precise visual evidence. Ultimately, EgoTools pushes egocentric video understanding beyond recognizing what people do toward understanding how intelligence operates through tools, opening a path toward embodied AI systems that can observe, reason, and assist in the physical world.

## References

[1] Kumar Ashutosh, Zihui Xue, Tushar Nagarajan, and Kristen Grauman. 2024. Detours for navigating instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18804–18815.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

[4] Sven Bambach, Stefan Lee, David J. Crandall, and Chen Yu. 2015. Lending a hand: Detecting hands and recognizing activities in complex egocentric interactions. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1949–1957.

[5] Leonard Barmann and Alex Waibel. 2022. Where did i leave my keys? episodic-memory-based question answering on egocentric videos. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1559–1567.

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, and 1 others. 2023. RT-2: Vision-language-action models transfer web knowledge to robotic control. Preprint, arXiv:2307.15818.

[7] Eun Chang, Zhuangqun Huang, Yiwei Liao, Sagar Ravi Bhavsar, Amogh Param, Tammy Stark, Adel Ahmadyan, Xiao Yang, Jiaqi Wang, Ahsan Abdullah, Giang Nguyen, Akil Iyer, David Hall, Elissa Li, Shane Moon, Nicolas Scheffer, Kirmani Ahmed, Babak Damavandi, Rakesh Wanga, and 3 others. 2025. WearVQA: A visual question answering benchmark for wearables in egocentric authentic real-world scenarios. Preprint, arXiv:2511.22154.

[8] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. 2024. EgoPlan-Bench: Benchmarking multimodal large language models for human-level planning. Preprint, arXiv:2312.06722.

[9] Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. 2024. EgoThink: Evaluating first-person perspective thinking capability of vision-language models. Preprint, arXiv:2311.15596.

[10] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and 1 others. 2022. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, 130(1):33–55.

[11] Shangzhe Di and Weidi Xie. 2024. Grounded question-answering in long egocentric videos. Preprint, arXiv:2312.06505.

[12] Kuan Fang, Yuke Zhu, Animesh Garg, Andrey Kurenkov, Viraj Mehta, Li Fei-Fei, and Silvio Savarese. 2018. Learning task-oriented grasping for tool manipulation from simulated self-supervision. Preprint, arXiv:1806.09266.

[13] Alireza Fathi, Yin Li, and James M. Rehg. 2012. Learning to recognize daily actions using gaze. In Computer Vision – ECCV 2012, pages 314–327. Springer Berlin Heidelberg.

[14] Hengjian Gao, Kaiwei Zhang, Shibo Wang, Mingjie Chen, Qihang Cao, Xianfeng Wang, Yucheng Zhu, Xiongkuo Min, Wei Sun, Dandan Zhu, and Guangtao Zhai. 2026. LifeEval: A multimodal benchmark for assistive AI in egocentric daily life tasks. Preprint, arXiv:2603.00490.

[15] Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, and Dorsa Sadigh. 2024. Physically grounded vision-language models for robotic manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12462–12469.

[16] Yuanyuan Gao, Hao Li, Yifei Liu, Xinhao Ji, Yuning Gong, Yuanjun Liao, Fangfu Liu, Manyuan Zhang, Yuchen Yang, Dan Xu, and 1 others. 2026. Holi-spatial: Evolving video streams into holistic 3d spatial intelligence. arXiv preprint arXiv:2603.07660.

[17] Google DeepMind. 2025. Gemini 3 flash model card. https://deepmind.google/models/model-cards/ gemini-3-flash. Published December 2025.

[18] Google DeepMind. 2026. Gemini 3.1 flash-lite model card. https://deepmind.google/models/ model-cards/gemini-3-1-flash-lite/. Published March 2026.

[19] Google DeepMind. 2026. Gemini 3.1 pro model card. https://deepmind.google/models/model-cards/ gemini-3-1-pro/. Published February 2026.

[20] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, and 66 others. 2022. Ego4D: Around the world in 3,000 hours of egocentric video. Preprint, arXiv:2110.07058.

[21] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, and 1 others. 2024. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400.

[22] Siyuan Huang, Iaroslav Ponomarenko, Zhengkai Jiang, Xiaoqi Li, Xiaobin Hu, Peng Gao, Hongsheng Li, and Hao Dong. 2024. ManipVQA: Injecting robotic affordance and physically grounded information into multi-modal large language models. Preprint, arXiv:2403.11289.

[23] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. 2022. EgoTaskQA: Understanding human tasks in egocentric videos. Preprint, arXiv:2210.03929.

[24] Kangsan Kim, Yanlai Yang, Suji Kim, Woongyeong Yeo, Youngwan Lee, Mengye Ren, and Sung Ju Hwang. 2026. MA-EgoQA: Question answering over egocentric videos from multiple embodied agents. Preprint, arXiv:2603.09827.

[25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.

[26] Chuhan Li, Ruilin Han, Joy Hsu, Yongyuan Liang, Rajiv Dhawan, Jiajun Wu, Ming-Hsuan Yang, and Xin Eric Wang. 2026. Learning situated awareness in the real world. Preprint, arXiv:2602.16682.

[27] Gen Li, Nikolaos Tsagkas, Jifei Song, Ruaridh Mon-Williams, Sethu Vijayakumar, Kun Shao, and Laura Sevilla-Lara. 2025. Learning precise affordances from egocentric videos for robotic manipulation. Preprint, arXiv:2408.10123.

[28] Hao Li, Minghan Qin, Zhengyu Zou, Diqi He, Xinhao Ji, Bohan Li, Bingquan Dai, Dingewn Zhang, and Junwei Han. 2024. Langsurf: Language-embedded surface gaussians for 3d scene understanding. arXiv preprint arXiv:2412.17635.

[29] Yin Li, Miao Liu, and James M. Rehg. 2018. In the eye of beholder: Joint learning of gaze and actions in first person video. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 619–635.

[30] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. 2024. HOI4D: A 4d egocentric dataset for category-level human-object interaction. Preprint, arXiv:2203.01577.

[31] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, and 5 others. 2024. OpenEQA: Embodied question answering in the era of foundation models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16488–16498.

[32] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. 2023. EgoSchema: A diagnostic benchmark for very long-form video language understanding. Preprint, arXiv:2308.09126.

[33] Tushar Nagarajan and Lorenzo Torresani. 2024. Step differences in instructional video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18740–18750.

[34] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. 2026. RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations (ICLR).

[35] Ye Pan, Chi Kit Wong, Yuanhuiyi Lyu, Hanqian Li, Jiahao Huo, Jiacheng Chen, Lutao Jiang, Xu Zheng, and Xuming Hu. 2026. EgoIntent: An egocentric step-level benchmark for understanding what, why, and next. Preprint, arXiv:2603.12147.

[36] Hamed Pirsiavash and Deva Ramanan. 2012. Detecting activities of daily living in first-person camera views. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2847–2854. IEEE.

[37] Chiara Plizzari, Alessio Tonioni, Yongqin Xian, Achin Kulshrestha, and Federico Tombari. 2025. Omnia de EgoTempo: Benchmarking temporal understanding of multi-modal LLMs in egocentric videos. Preprint, arXiv:2503.13646.

[38] Shengyi Qian, Weifeng Chen, Min Bai, Xiong Zhou, Zhuowen Tu, and Li Erran Li. 2024. AffordanceLLM: Grounding affordance from vision language models. Preprint, arXiv:2401.06341.

[39] Meiying Qin, Jake Brawer, and Brian Scassellati. 2023. Robot tool use: A survey. Frontiers in Robotics and AI, 9.

[40] Francesco Ragusa, Antonino Furnari, and Giovanni Maria Farinella. 2023. Meccano: A multimodal egocentric dataset for humans behavior understanding in the industrial-like domain. Computer vision and image understanding, 235:103764.

[41] Francesco Ragusa, Michele Mazzamuto, Rosario Forte, Irene D’Ambra, James Fort, Jakob Engel, Antonino Furnari, and Giovanni Maria Farinella. 2025. Ego-EXTRA: Video-language egocentric dataset for EXpert-TRAinee assistance. Preprint, arXiv:2512.13238.

[42] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, and 1 others. 2024. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714.

[43] Daniel Seita, Yufei Wang, Sarthak J. Shetty, Edward Yao Li, Zackory Erickson, and David Held. 2022. ToolFlowNet: Robotic manipulation with tools via predicting tool flow from point clouds. Preprint, arXiv:2211.09006.

[44] Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao. 2022. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. Preprint, arXiv:2203.14712.

[45] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J. Joshi, Pete Florence, Wei Han, Robert Baruch, Yao Lu, Suvir Mirchandani, Peng Xu, Pannag Sanketi, Karol Hausman, Izhak Shafran, and 2 others. 2023. RoboVQA: Multimodal long-horizon reasoning for robotics. Preprint, arXiv:2311.00899.

[46] Pengzhan Sun, Junbin Xiao, Tze Ho Elden Tse, Yicong Li, Arjun Akula, and Angela Yao. 2025. Visual intention grounding for egocentric assistants. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2512–2522.

[47] Qiance Tang, Ziqi Wang, Jieyu Lin, Ziyun Li, Barbara De Salvo, and Sai Qian Zhang. 2026. EgoEverything: A benchmark for human behavior inspired long context egocentric video understanding in AR environment. Preprint, arXiv:2604.08342.

[48] Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, and 55 others. 2025. Mimo-vl technical report. Preprint, arXiv:2506.03569.

[49] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, and 74 others. 2026. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. Preprint, arXiv:2507.01006.

[50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. Preprint, arXiv:2409.12191.

[51] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, and 56 others. 2025. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. Preprint, arXiv:2508.18265.

[52] Junbin Xiao, Shenglang Zhang, Pengxiang Zhu, and Angela Yao. 2026. Ego-grounding for personalized questionanswering in egocentric videos. Preprint, arXiv:2604.01966.

[53] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-omni technical report. Preprint, arXiv:2503.20215.

[54] Natsuki Yamanobe, Weiwei Wan, Ixchel G. Ramirez-Alpizar, Damien Petit, Tokuo Tsuji, Shuichi Akizuki, Manabu Hashimoto, Kazuyuki Nagata, and Kensuke Harada. 2017. A brief review of affordance in robotic manipulation research. Advanced Robotics, 31(19–20):1086–1101.

[55] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco Cominelli, Zhongang Cai, Bo Li, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg Widmer, and 3 others. 2025. EgoLife: Towards egocentric life assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28885–28900.

[56] Lei Yao, Yong Chen, Yuejiao Su, Yi Wang, Moyun Liu, and Lap-Pui Chau. 2026. HAMMER: Harnessing MLLM via cross-modal integration for intention-driven 3d affordance grounding. Preprint, arXiv:2603.02329.

[57] Chunlin Yu, Hanqing Wang, Ye Shi, Haoyang Luo, Sibei Yang, Jingyi Yu, and Jingya Wang. 2025. SeqAfford: Sequential 3d affordance reasoning via multimodal large language model. Preprint, arXiv:2412.01550.

[58] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2025. Llava-video: Video instruction tuning with synthetic data. Preprint, arXiv:2410.02713.

[59] Zixin Zhang, Kanghao Chen, Xingwang Lin, Lutao Jiang, Xu Zheng, Yuanhuiyi Lyu, Litao Guo, Yinchuan Li, and Ying-Cong Chen. 2025. PhysToolBench: Benchmarking physical tool understanding for MLLMs. Preprint, arXiv:2510.09507.

[60] Wenqi Zhou, Kai Cao, Hao Zheng, Yunze Liu, Xinyi Zheng, Miao Liu, Per Ola Kristensson, Walterio Mayol-Cuevas, Fan Zhang, Weizhe Lin, and Junxiao Shen. 2025. X-LeBench: A benchmark for extremely long egocentric video understanding. Preprint, arXiv:2501.06835.

[61] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, and 32 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. Preprint, arXiv:2504.10479.

## Appendix Contents

A Contributors and Acknowledgments 21   
B Ethics 21   
C Dataset Collection and Contributor Information 22   
C.1 Data Collection Settings 22   
C.2 Anonymized Contributor Statistics 23   
D Preprocessing and Annotation Protocols 24   
D.1 View Rectification Settings . 24   
D.2 Narration Annotation Guidelines 25   
E Dataset Statistics and Release 26   
E.1 Dataset and Benchmark Statistics 26   
E.2 Release and Privacy Considerations 27   
F Quality Control and Normalization 29   
F.1 Fix-First Quality Control . 29   
F.2 Eight-Choice Normalization and Anti-Shortcut Checks 29   
G Additional Related Work 30   
H Training and Evaluation Details 31   
H.1 Single-Stage Training Procedure 31   
H.2 Training Data Composition 32   
H.3 Evaluation Protocols 32

## A. Contributors and Acknowledgments

Author contributions. Shulin Tian, Junsu Kim, Shuai Liu, and Hao Li contributed equally, mainly to data collection, benchmark construction, training, evaluation, and writing. Yujiao Shen, Sihan Li, Zhe Yang, Yeongon Kim, Feiyu Li, Jialin Wu, Yichi Zhang, Wenhui Wang, Runmao Yao, and Yuhao Dong contributed to data annotation and logistics. Zhaoxi Chen, Fangzhou Hong, Ziwei Liu, Antonino Furnari, Jingkang Yang, and Hongyuan Zhu supervised the project and guided its overall research direction. Ziwei Liu served as the corresponding author.

Acknowledgments. This work is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 grant (MOE-T2EP20223-0002), and by cash and in-kind funding from NTU S-Lab and its industry partners. We thank Ropedia for providing hardware, API access, and annotation support. We are grateful to Jewoo Park, Jeongwon Yoo, Yulgyeom Kim, Yuan Fei, Haiheng Liu, Gordon Chen, Jiarui Zheng, Junhan Zhu, and Hongmei Xu for their assistance with egocentric video data collection and annotation. We also thank all participants, annotators, reviewers, and infrastructure contributors for their efforts in data collection, annotation, quality control, and release preparation for EgoTools.

## B. Ethics

EgoTools was constructed with attention to informed participation, permitted data use, privacy protection, and responsible public release. We provide an anonymized summary of the consent, data-use, and release safeguards applied during data collection, annotation, benchmark construction, model training, and dataset distribution.

Voluntary participation and informed consent. Participation in egocentric video recording, narration, and annotation was voluntary. Participants received no monetary compensation for contributing recordings, narrations, or annotations. Before contributing data, participants were informed about the purpose of the project, the types of data to be collected, the intended research uses, and the possibility that cleared materials would be publicly released for research purposes. Participants provided consent covering the applicable data collection, annotation, research-use, and release activities.

Institutional ethics review. No formal IRB or institutional ethics approval was sought for the EgoTools data collection. We explicitly report the absence of formal ethics review and describe below the informed-consent, privacy, anonymization, and release safeguards applied throughout the project.

Consent and permitted use. Participants contributed egocentric videos, narrations, and associated annotations for research on physical tool-use reasoning, egocentric video understanding, benchmark construction, grounded video-language understanding, multimodal model training, and dataset release. Within the applicable consent and data-use scope, derived materials may include corrected English narrations, dense hierarchical captions, benchmark question–answer pairs, timestamps, grounding annotations, object trajectories, and other research annotations generated from the contributed recordings.

Collected data. Depending on the participant role and recording setup, internally collected materials may include raw fisheye videos, rectified egocentric videos, synchronized audio, spoken narrations, corrected English text narrations, dense hierarchical captions, benchmark QA annotations, clip metadata, timestamp annotations, grounding-oriented focal-tool annotations, depth or pose information, and selected trajectory or object-state annotations.

Public release scope. The public release is restricted to materials covered by the applicable participant consent, release authorization, anonymization procedures, and privacy review. The release focuses on rectified egocentric videos, dense hierarchical captions, corrected English narrations, benchmark QA pairs, clip metadata, timestamps, split information, and selected finalized grounding annotations. Original narration audio and selected 3D trajectory assets are released only when covered by the corresponding release authorization and privacy review. Raw fisheye videos are not publicly released.

Privacy and release safeguards. Released metadata excludes names, contact information, institution-specific identifiers, public profiles, and exact per-video contribution counts. Sensitive, private, or accidentally captured content is reviewed before release, and materials deemed unsuitable for public distribution are excluded. Additional grounding-oriented annotations and 3D trajectory assets are released incrementally only after annotation finalization and the corresponding authorization and privacy checks.

Participant rights and data removal. Before public release, participants may request the exclusion of data associated with their participation in accordance with the project release policy. Requests received after release are handled according to the applicable consent terms, dataset license, and release procedures.

Research-use restrictions. EgoTools is distributed under a research-use license specifying permitted uses, redistribution conditions, and usage restrictions. The dataset is not intended for participant identification, biometric profiling, surveillance, or other uses that conflict with the original consent and release scope.

## C. Dataset Collection and Contributor Information

## C.1. Data Collection Settings

EgoTools uses two complementary collection settings to balance procedural control with natural variation in real-world tool use. We provide representative anonymized examples below.

Structured task-guided recording. In a structured task-guided session, the data collection team prepares a task goal and a sequence of key procedural steps, while allowing the participant to perform the manipulation naturally. For example, in a kitchen recording, a participant may be asked to prepare a simple pan-cooked dish by washing ingredients, cutting them on a board, heating a pan, spreading oil, manipulating food with a spatula or chopsticks, and transferring the result to a plate. The participant is not scripted at the frame level, but the task sequence ensures that the recording contains clear procedural boundaries, repeated hand–tool–object interactions, and observable state changes such as raw ingredients being cut, food being cooked, or tools being cleaned for later use. Such sessions are useful for benchmark construction because they provide temporally coherent clips with well-defined goals and visually grounded state transitions.

Open-ended participant-driven recording. In an open-ended participant-driven session, the participant is given only a broad activity goal and completes it in their own manner. For example, a participant may be asked to perform an unscripted household, craft, laboratory, or workshop activity using the tools they consider appropriate. This setting captures natural variation that is difficult to script, such as choosing a narrower tool because the target opening is small, replacing one tool with another after an unsuccessful attempt, or changing the manipulation strategy as the target object bends, tears, loosens, or becomes unstable. These recordings complement the structured sessions by exposing models to spontaneous tool choice, substitution, recovery, and state-dependent manipulation.

## C.2. Anonymized Contributor Statistics

To preserve anonymity, we report aggregate role and background statistics for contributors from whom complete information was collected. These statistics cover a documented subset of the broader author and contributor pool and are not intended to enumerate every manuscript author or every individual acknowledged in Appendix A. We do not report names, contact information, institutional identifiers, public profiles, or exact per-video contribution counts.

Contributor roles. Contributors may participate in one or more roles, including video participant, narrator, QA annotator, QA reviewer, data curator, grounding annotator, or trajectory annotator. Since several contributors played multiple roles, the role counts in Table 6 are not mutually exclusive.

Table 6 | Contributor roles in the documented subset. Counts are non-exclusive.
<table><tr><td>Role</td><td># Contributors</td><td>Main Responsibility</td></tr><tr><td>Participant / narrator</td><td>16</td><td>Record egocentric videos and provide tool-centric narrations</td></tr><tr><td>QA annotator</td><td>10</td><td>Write human-crafted benchmark QA pairs</td></tr><tr><td>QA reviewer</td><td>2</td><td>Review answer validity, visual evidence, and distractor quality</td></tr><tr><td>Data curator</td><td>3</td><td>Curate videos, manage annotation progress, and organize benchmark data</td></tr><tr><td>Grounding annotator</td><td>1</td><td>Annotate focal-tool grounding for selected instances</td></tr><tr><td>Trajectory annotator</td><td>2</td><td>Construct or verify trajectory annotations for selected clips</td></tr></table>

Anonymization policy. These aggregate statistics are intended to document the range of contributor backgrounds and project roles without enabling linkage to real identities or specific released recordings. We do not release names, institutions, contact information, public profiles, or exact per-video contribution counts.

Table 7 | Contributor backgrounds. Aggregate (a) degree and field and (b) age-group distributions for the documented subset.  
(a) Degree and field
<table><tr><td colspan="2">Degree</td><td colspan="2">Field</td></tr><tr><td>Category</td><td>#</td><td>Category</td><td>#</td></tr><tr><td>Undergrad.</td><td>12</td><td>Computer Science</td><td>12</td></tr><tr><td>Master&#x27;s</td><td>5</td><td>Electrical Eng.</td><td>7</td></tr><tr><td>PhD</td><td>5</td><td>Biology</td><td>2</td></tr><tr><td>Postdoc</td><td>1</td><td>Chemistry</td><td>1</td></tr><tr><td></td><td></td><td>Film / Media</td><td>1</td></tr></table>

(b) Age group  
![](images/47f20ead621e8697872d89620ab50a039dbe23d85d5e0a6f62b455b2ed4139e3.jpg)

## D. Preprocessing and Annotation Protocols

## D.1. View Rectification Settings

Recording device and modalities. EgoTools videos are recorded with HOMIE, a headmounted multimodal recording platform that captures four synchronized fisheye camera streams together with audio and auxiliary motion and synchronization signals. These raw modalities provide the source data for downstream view rectification, temporal alignment, and selected geometry-related processing.

Canonical view construction. For annotation, instruction-data construction, benchmark construction, model evaluation, and public release, we use the rectified front-left RGB stream as the canonical egocentric view. The rectification process uses synchronized views from the same recording to produce a spatially normalized first-person representation while preserving the hand–tool–object interactions and surrounding workspace required for tool-use understanding.

Figure 7 shows a representative rectification example. The target and reference views are processed using fixed view parameters to produce the canonical output. For the displayed example, the output resolution is 1024 × 1024, the zoom factor is 0.78125, the field of view is 90.0<sup>◦</sup>, and the pitch correction is −5.0<sup>◦</sup>, with yaw and roll set to 0.0<sup>◦</sup>. The values shown in the figure correspond to the displayed example.

The resulting videos are single-view egocentric streams with a resolution of 1024 × 1024 at 20 FPS and synchronized audio. The same canonical video representation is used for annotation, instruction-data construction, benchmark construction, model evaluation, and the publicly released video assets.

![](images/244f3d4ee12f0eb521cac86f6b1a5bacb6dde0091e14e545f31bf186a6fd7faa.jpg)  
Figure 7 | Canonical-view rectification. Example input views and the rectified first-person frame.

Release scope. The public release includes the rectified canonical videos rather than the original raw fisheye camera streams. We report a representative rectification configuration and the output specifications used in our data processing, while raw fisheye streams and internal preprocessing utilities are not included in the release.

## D.2. Narration Annotation Guidelines

Participants provide narrations for meaningful tool-use events rather than densely describing every visible action. The goal is to capture which tool is being used, what target is being acted on, why the selected tool is appropriate in the current context, and what alternative tool may be relevant when applicable.

Annotation interface and procedure. The annotation interface allows participants or data collectors to watch rectified egocentric videos, identify relevant timestamps, and provide toolcentric narrations. Narrations are associated with temporally localized video segments and are written or corrected with reference to the visible hand–tool–object interaction.

The narration guidelines encourage annotators to specify the selected tool, a relevant alternative when applicable, the target object, the performed action, and the physical or procedural rationale for the tool choice. Narrations are free-form and need not follow a rigid sentence structure, but the following template is provided as a reference:

I am using [the] [property] [tool] [location/identity cue] rather than [the] [property] [alternative tool] [location/identity cue] to [action] on/for [target] because [reason].

Narration targets. Narrations focus on tool-use moments that involve meaningful decisions or state-dependent behavior, including selecting a tool, switching between tools, adapting manipulation to a changing object state, preparing a tool for a later step, recovering from an unsuccessful attempt, or using a tool in a physically appropriate manner. Participants are not required to narrate every visible action.

Narration language and normalization. Source narrations may be provided in English, Korean, or Chinese. Korean and Chinese narrations are transcribed and translated into English, and all narrations are subsequently normalized and corrected by bilingual or English-proficient project members when necessary. The corrected English text is paired with the corresponding video segment and used as the primary narration annotation signal for instruction-data construction.

Language tags retain information about the original narration language. Original narration audio is preserved as an internal dataset asset and may be included in the public release only when covered by the applicable release authorization and privacy review.

Benchmark language and evidence separation. All final EgoTools-Bench questions and answer choices are written in English. When benchmark construction relies on auxiliary narrations or rough temporal descriptions originally provided in Korean or Chinese, these materials are translated and normalized into English before QA construction or verification. Final benchmark items are reviewed for linguistic clarity, consistency with timestamped visual evidence, single-best-answer validity, and distractor quality.

Tool-centric narrations, corrected narration text, and narration audio are not provided to models during EgoTools-Bench evaluation. They are also not supplied to benchmark annotators as privileged answer evidence; benchmark questions must remain answerable from the corresponding video and question context.

Grounding-oriented annotations. Narrations may be associated with focal tools or target objects. For selected instances, annotators provide 2D grounding points on corresponding keyframes, linking textual mentions to visible object instances. These grounding annotations provide initialization cues for long-horizon tracking and support focal-tool localization, trajectory construction, and grounded video-language training.

Grounding annotations are available only for selected instances and are progressively released as annotation finalization, release authorization, and privacy review are completed.

## Examples.

• I am using the narrow chopsticks rather than the spoon to reach the jam inside the jar because the spoon is too wide to fit through the opening.

• I am using the flat spatula rather than the chopsticks to lift the egg because the broad surface can support the soft egg without tearing it.

• I am using the pipette rather than the larger container to transfer a small amount of liquid because the pipette provides finer volume control.

## E. Dataset Statistics and Release

## E.1. Dataset and Benchmark Statistics

EgoTools contains 646 egocentric videos totaling 100.37 hours across seven tool-use domains: kitchen, classroom, research laboratory, repair workshop, craft, office, and household. Throughout the main paper, we refer to this corpus as approximately 100 hours. The corpus additionally contains 361,332 hierarchical captions and 6,519 tool-centric narrations. From these recordings, we reserve 40.34 hours of high-quality, tool-dense source videos exclusively for benchmark construction and evaluation.

Table 8 | EgoTools dataset and benchmark statistics.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total EgoTools videos</td><td>646</td></tr><tr><td>Total video duration</td><td>100.37 hours</td></tr><tr><td>Hierarchical captions</td><td>361,332</td></tr><tr><td>Tool-centric narrations</td><td>6,519</td></tr><tr><td>Benchmark-reserved source duration</td><td>40.34 hours</td></tr><tr><td>Benchmark QA pairs</td><td>1,000</td></tr><tr><td>Human-crafted QA pairs</td><td>900</td></tr><tr><td>Human-verified spatial QA pairs</td><td>100</td></tr></table>

Domain-level composition. Table 9 reports the distribution of the complete EgoTools-Data corpus and the corresponding domain allocation of EgoTools-Bench. Corpus statistics describe the full data collection, whereas benchmark QA counts describe the source domain assigned to each of the 1,000 benchmark questions.

Table 9 | Domain composition of EgoTools-Data and EgoTools-Bench. The benchmark is not domainbalanced.
<table><tr><td colspan="5"></td><td rowspan="2">EgoTools-Bench</td></tr><tr><td>Domain</td><td># Videos</td><td>Hours</td><td># Captions</td><td>#Narrations</td></tr><tr><td>Kitchen</td><td>276</td><td>32.37</td><td>116,532</td><td>2,113</td><td># QAs 837</td></tr><tr><td>Classroom</td><td>74</td><td>6.86</td><td>24,696</td><td>467</td><td>15</td></tr><tr><td>Research Lab</td><td>61</td><td>30.92</td><td>111,312</td><td>1,887</td><td>99</td></tr><tr><td>Repair Workshop</td><td>84</td><td>12.34</td><td>44,424</td><td>763</td><td>16</td></tr><tr><td>Craft</td><td>67</td><td>10.17</td><td>36,612</td><td>657</td><td>23</td></tr><tr><td>Office</td><td>30</td><td>2.83</td><td>10,188</td><td>213</td><td>6</td></tr><tr><td>Household</td><td>54</td><td>4.88</td><td>17,568</td><td>419</td><td>4</td></tr><tr><td>Total</td><td>646</td><td>100.37</td><td>361,332</td><td>6,519</td><td>1,000</td></tr></table>

The domain distribution highlights the distinction between corpus coverage and benchmark composition. EgoTools-Data contains substantial recordings from all seven domains, including approximately 31 hours of research-laboratory activity and more than 12 hours of repairworkshop activity. EgoTools-Bench, however, is dominated by Kitchen questions and should not be interpreted as a domain-balanced evaluation set. The benchmark was constructed primarily to cover complementary tool-use reasoning capabilities, annotation quality, and challenging tool-mediated interactions rather than to support statistically balanced comparisons across collection domains. Domain-level results for the smallest benchmark slices would therefore be unstable and are not used for primary performance claims.

Reasoning-track distribution. EgoTools-Bench contains 1,000 8-way multiple-choice question– answer pairs, including 900 human-crafted questions and 100 human-verified spatial questions generated from the 3D annotation pipeline. Across the four research-facing tracks, the benchmark includes 363 Affordance & Causality questions, 236 Perception & Grounding questions, 222 Procedural Dynamics questions, and 179 Spatial Reasoning questions. This distribution reflects the benchmark’s emphasis on tool selection, application, and physical effects while maintaining coverage of perceptual grounding, procedural understanding, and spatial hand–tool–object relations.

Evaluation-only benchmark partition. All benchmark source videos, clips, questions, answers, and derived annotations are reserved exclusively for evaluation. No clip, caption, narration, synthetic QA, or other annotation derived from a benchmark-reserved source video is included in the instruction-tuning corpus.

## E.2. Release and Privacy Considerations

The EgoTools release is scoped to research use and includes only materials covered by the applicable participant consent, release authorization, anonymization procedures, and privacy review. The release provides processed rectified egocentric videos rather than the original raw fisheye recordings, together with the annotations and metadata required for benchmark evaluation and dataset documentation. Additional consent and privacy safeguards are described in Appendix B.

Initial release components. The initial release package will include the following cleared components:

• Rectified egocentric video clips;

• Dense hierarchical captions;

• Corrected English tool-centric narrations;

• All 1,000 EgoTools-Bench question–answer pairs and their evaluation annotations;

• Clip metadata and timestamps;

• Source-video-disjoint training and benchmark split information; and

• Dataset documentation and benchmark evaluation metadata.

Incrementally released components. The following components are released incrementally as their annotation, authorization, and privacy-review procedures are completed:

• Original narration audio for consent-cleared examples;

• Selected finalized focal-tool grounding annotations;

• Selected object-state and trajectory annotations; and

• 3D trajectory visualizations for selected clips.

The availability of these components may differ across examples because grounding, audio, and trajectory assets require additional annotation-finalization and release-review procedures.

Excluded components. Raw fisheye camera streams are not publicly released. The release also excludes personally identifying metadata, uncleared recordings or audio, sensitive or accidentally captured content, internal identity mappings, exact per-video contributor counts, and internal preprocessing utilities.

Privacy safeguards. Names, contact information, institution-specific identifiers, public-profile information, and other personally identifying information are removed or withheld from released metadata. Sensitive, private, or accidental content is reviewed before release, and materials deemed unsuitable for public distribution are excluded. Contributors may request exclusion of data associated with their participation in accordance with the release policy and applicable consent terms.

Access conditions and license. EgoTools is distributed under a research-use license specifying permitted uses, redistribution conditions, and usage restrictions. Privacy-sensitive or conditionally released components are handled according to their applicable authorization and privacy-review outcomes. The dataset is not intended for participant identification, biometric profiling, surveillance, or other uses inconsistent with the original collection and release scope.

Limitations. Although EgoTools-Data spans seven real-world tool-use domains, it does not exhaustively cover all tools, environments, participant populations, cultural practices, or professional procedures. The benchmark distribution is strongly concentrated in Kitchen videos and should not be treated as representative of the domain distribution of the full corpus. Grounding, narration-audio, and 3D trajectory components are available only for examples that have passed the corresponding annotation-finalization, release-authorization, and privacy-review procedures.

## F. Quality Control and Normalization

## F.1. Fix-First Quality Control

Raw human annotations are processed under a fix-first quality-control policy that prioritizes repairing valid annotations before discarding them. Items are retained whenever deterministic or minimal model-assisted edits can preserve the original annotator intent without introducing ambiguity.

Structural audit. Each item is evaluated using a 22-flag binary audit covering defects such as placeholders, missing fields, malformed options, duplicate entries, first-person leakage, personally identifiable information, answer–option mismatch, inappropriate full-video clip usage, and distractor anomalies.

Filtering policy. Content-empty submissions and verbatim duplicates are removed. Repairable defects are corrected through deterministic rules or minimal model-assisted edits. Cases for which the intended meaning or unique answer cannot be established with high confidence are routed to human review rather than being automatically accepted.

Model-assisted repair. We use Gemini-2.5-Flash-Lite with temperature 0 for model-assisted repair. The repair prompt instructs the model to perform the smallest meaning-preserving edit possible. Typical edits include replacing personally identifying names with role-based references, rewriting first-person expressions into third-person form, correcting surface-level grammar, and completing truncated answer text only when the intended content is unambiguous. Edits that could alter the semantic content or answer validity are routed to human review.

## F.2. Eight-Choice Normalization and Anti-Shortcut Checks

Eight-choice normalization. All benchmark items are normalized to exactly eight answer options. We dispatch each item according to its current distractor count: items with fewer than seven distractors are augmented, items with exactly seven distractors are validated, and items with more than seven distractors are reduced to seven plausible alternatives. This process produces a consistent 8-way multiple-choice format while preserving the original correct answer and question intent whenever possible.

Anti-shortcut checks. Each normalized item and the resulting benchmark corpus are subjected to deterministic anti-shortcut checks before finalization. Item-level checks detect meta-options such as “all of the above” or “none of the above,” duplicate or near-duplicate options, tokenset permutations of the correct answer, substring or superstring containment, and excessive question–answer lexical overlap. Corpus-level checks additionally examine answer-position imbalance and systematic answer-length bias. Failed items are routed to bounded regeneration or human review, with detected violations provided as feedback for revising the answer options.

Length-bias check. For each item, we compare the character length of the correct answer with the distribution of distractor lengths. Let $L _ { \mathrm { a n s } }$ denote the character length of the correct answer, and let $\mu _ { \mathrm { d i s t } }$ and $\sigma _ { \mathrm { d i s t } }$ denote the mean and standard deviation of the seven distractor lengths. We compute

$$
z = \frac { L _ { \mathrm { a n s } } - \mu _ { \mathrm { d i s t } } } { \sigma _ { \mathrm { d i s t } } } .
$$

Items satisfying |�| > 1.5 are flagged for repair. Previously cached eight-option snapshots are subjected to the same checks rather than being accepted without revalidation, ensuring that the length constraint is consistently enforced across all benchmark items. An earlier cached snapshot exhibited a corpus-level mean standardized deviation of �¯ ≈ +2.21. After applying the finalized check and repair procedure, the retained benchmark has �¯ ≈ −0.15, with every item satisfying |�| ≤ 1.5.

Option shuffling. The final answer-option order is shuffled using a deterministic seed derived from each QA identifier. This reduces positional priors while keeping evaluation reproducible. Because every released benchmark item contains eight answer choices, the main benchmark tables report standard multiple-choice accuracy with a consistent chance baseline of 12.5%.

## G. Additional Related Work

Broader egocentric video reasoning. Egocentric video benchmarks have expanded from action and object recognition toward episodic memory, planning, temporal localization, situated reasoning, and assistance. QAEgo4D and GroundVQA study episodic-memory QA and temporally grounded QA in long egocentric videos [5, 11]. EgoTaskQA evaluates task-oriented questions about dependencies, effects, goals, and beliefs [23]; EgoSchema and EgoTempo emphasize long-context and temporal reasoning [32, 37]; and EgoPlan-Bench evaluates planning from first-person videos [8]. More recent benchmarks examine first-person thinking, life-log and real-time assistance, expert–trainee support, situated awareness, multi-agent egocentric QA, intent understanding, personalized grounding, and AR or life-logging environments [9, 14, 24, 26, 35, 41, 47, 52, 55, 60]. These resources substantially broaden egocentric evaluation around memory, temporal context, planning, and user assistance. However, they generally do not isolate the physical reasoning required to determine why a particular tool is appropriate, which substitute is feasible, or how manipulation should adapt as the target object changes state.

Instructional, embodied, and robotic interaction benchmarks. Instructional-video datasets provide useful comparisons because they frequently contain tools, ingredients, techniques, and alternative procedures. VidDetours retrieves procedural detours from how-to videos, while StepDiff identifies differences between instructional clips [1, 33]. Their primary focus, however, is procedural retrieval or comparison rather than physical tool-use reasoning grounded in a continuous first-person task. Embodied interaction benchmarks provide another related but distinct setting. OpenEQA evaluates open-vocabulary embodied question answering in realworld environments [31], RoboVQA provides large-scale video-language supervision from robotic demonstrations [45], and RoboCasa365 studies simulated kitchen manipulation across diverse tasks and environments [34]. These resources offer important scope contrasts, but they do not provide the same combination of natural human egocentric video, tool-centric narration, narration-linked focal-tool grounding, and explicit QA over tool choice, substitution, ordered manipulation, and state-dependent adaptation.

Affordance grounding and physical tool understanding. Robotics and embodied AI have long treated tool use as a problem of affordance, grasping, motion, and task execution. Taskoriented grasping studies how tools should be grasped for activities such as sweeping and hammering [12], while ToolFlowNet predicts dense tool motion for manipulation tasks such as scooping and pouring [43]. Broader surveys and manipulation methods study robot tool use, affordance-based manipulation, precise affordance grounding, and physically grounded policies [6, 15, 27, 39, 54]. Recent multimodal benchmarks and methods further evaluate whether MLLMs and VLMs can recognize affordances, infer physical constraints, and ground tool-related interactions [22, 38, 57, 59, 56]. These studies reveal persistent weaknesses in reasoning about tool functions, physical constraints, and multi-step interactions. Nevertheless, many evaluations are based on static images, synthetic or reconstructed 3D scenes, short interactions, or constrained robotic settings. EgoTools-Bench is complementary: it evaluates physical tool-use reasoning from natural first-person videos in which models must interpret real human tool selection, substitution, manipulation, and adaptation over time.

Wearable assistants. Wearable and egocentric assistant benchmarks are especially relevant because they require models to answer situated questions from the user’s viewpoint. WearVQA studies visual question answering for wearable devices [7], while visual-intention grounding evaluates whether models can infer user needs and intentions from egocentric observations [46]. These settings motivate first-person assistance but do not specifically center tool-mediated physical reasoning. EgoTools-Bench therefore complements wearable-assistant research by focusing on questions that require understanding the functional role of tools, feasible substitutes, temporally ordered manipulation, and adaptation to changing object states.

## H. Training and Evaluation Details

## H.1. Single-Stage Training Procedure

Training mixture. We use the 184,679-example corpus described in Section 3.3. By answer format, the mixture contains 170,595 multiple-choice examples, 5,000 short open-ended examples, and 9,084 dense-captioning and narration-completion examples; answer letters in the next-action subset are balanced across the four positions at 1,250 examples each. Table 10 gives the capability-level breakdown.

The next-action and single-image open-ended examples are generated from EgoTools training videos and their caption timelines; no video, image, or QA annotation from EgoSchema, EgoPlan-Bench, or EgoThink is used. A subset of their question strings coincides with generic benchmark question templates, while media overlap with the benchmark reference set is zero.

Optimization. We perform full-parameter supervised fine-tuning of the language-model component of Qwen3-VL-8B-Instruct, while keeping the vision encoder and multimodal projector frozen. Thus, “Full SFT” refers to full-parameter updating of the language-model component rather than updating every module in the multimodal architecture.

We optimize using AdamW with a learning rate of $2 . 3 \times 1 0 ^ { - 6 }$ , a constant learning-rate schedule, and no warmup. Training is performed for one epoch with a per-device batch size of 1, gradient accumulation over 16 steps, and 8 GPUs, yielding an effective batch size of 128 and 1,443 optimizer steps. We use bfloat16 precision, DeepSpeed ZeRO Stage 3, FlashAttention, and gradient checkpointing.

Each training video is represented by 64 uniformly sampled frames at a target rate of 2 FPS, with the per-video frame count fixed to 64. We cap each image at 1,024 visual tokens and use a maximum multimodal sequence length of 8,192 tokens.

Source-video separation. The EgoTools training pool and EgoTools-Bench are partitioned before instruction-data or benchmark construction. No video, clip, caption, narration, synthetic QA pair, grounding annotation, or other derivative of a benchmark-reserved source video is

used during training.

## H.2. Training Data Composition

The training procedure uses 184,679 EgoTools-derived examples. Table 10 summarizes their composition by the capability each example supervises.

Table 10 | Training data by supervised capability. Shares of the 184,679-example mixture are rounded to one decimal place.
<table><tr><td>Supervised capability</td><td># Examples</td><td>Share</td></tr><tr><td>Affordance &amp; Causality (AC): tool-use purpose, intent, and causal rationale</td><td>14,175</td><td>7.7%</td></tr><tr><td>Perception &amp; Grounding (PG): tool and state identification, attributes, and frame-level visual evidence</td><td>14,755</td><td>8.0%</td></tr><tr><td>Procedural Dynamics (PD): temporal order, workflow, and state change within a clip</td><td>22,943</td><td>12.4%</td></tr><tr><td>Spatial Reasoning (SR): spatial relations, placement, and hand-tool-object geometry</td><td>17,887</td><td>9.7%</td></tr><tr><td>Episode-level multiple choice: action ordering, activity summarization, and goal inference over a whole episode (64,554), plus next-action questions paired with the current observation</td><td>69,554</td><td>37.7%</td></tr><tr><td>frame (5,000) General video QA: mixed episode-level questions not targeted at a single ability</td><td>31,281</td><td>16.9%</td></tr><tr><td>Dense captioning and narration completion</td><td>9,084</td><td>4.9%</td></tr><tr><td>Single-image open-ended QA</td><td>5,000</td><td>2.7%</td></tr><tr><td>Total</td><td>184,679</td><td>100.0%</td></tr></table>

## H.3. Evaluation Protocols

The results in Table 3 use the standard 64-frame MP4-based protocol for EgoTools-Bench. EgoSchema is evaluated on the full 5,031-question set, EgoPlan-Bench on its 3,343-question validation set, and EgoThink using macro-averaged accuracy over its twelve single-image subtasks, judged by gpt-4o-2024-11-20.