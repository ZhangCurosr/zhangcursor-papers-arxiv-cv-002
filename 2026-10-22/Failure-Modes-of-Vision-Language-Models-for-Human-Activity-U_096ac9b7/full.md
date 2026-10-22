# Failure Modes of Vision-Language Models for Human Activity Understanding in Thermal HRI

Han Hu and He Chen

Abstract— Thermal cameras enable robots to perceive human activities in darkness while revealing less personally identifiable appearance information and fewer details of the surrounding environment than RGB cameras. As vision-language models (VLMs), originally developed for RGB imagery, are increasingly integrated into embodied systems for scene understanding and high-level planning, thermal sensing offers a promising way to extend activity perception to low-light and privacy-sensitive settings. However, the failure modes of VLMs in thermal human activity understanding for human-robot interaction (HRI) remain underexplored. We fill this gap by evaluating seven locally deployable VLMs, including models developed for physical AI and robotics, on two person-centric thermal datasets. Our diagnostic tasks examine fine-grained action recognition, activity timeline reconstruction, and the ability to withhold unsupported answers. Paired RGB recordings serve as a performance reference, while controlled variations in frame sampling, thermal input presentation, and reasoning are used to assess whether the observed errors can be mitigated. Our results show that models can recognize broad activity categories but struggle with fine-grained actions such as coughing, sneezing, and yawning. They also produce inaccurate event timelines and hallucinate actions involving objects. Simple thermal input preprocessing and additional reasoning offer only limited, model-dependent gains. Abstention behavior also varies across models even under explicit instructions, limiting their reliability for practical robotic use.

## I. INTRODUCTION

Vision language models (VLMs) increasingly form part of the perception-and-reasoning stack of robotic systems, supporting visual understanding, task decomposition, and high-level planning [1]. For robots that share spaces with people, human activity is a central part of this understanding: knowing what a person is doing, what they are interacting with, and which events have just occurred provides important context for deciding whether, when, and how to act. Activity understanding therefore connects visual observation with downstream reasoning and planning in embodied systems.

Robot perception of human activity in the home, however, is constrained by both lighting conditions and privacy concerns. Thermal video captures body shape and motion cues in darkness while revealing less facial and household appearance detail than RGB imagery [2], [3]. This makes it well suited for human activity sensing in bedrooms and other private spaces, where a robot may need to detect a fall or recognize signs of discomfort even when the lights are off. When combined with the language interface of VLMs, thermal observations could support activity understanding at multiple levels of abstraction.

Prior work [3] found that VLMs recognize some thermal activities but struggle with subtle gestures and movements, while more recent thermal VLM studies target general thermal image understanding, such as scene semantics and temperature cues [4], [5], [6], [7]. Yet overall accuracy scores and isolated failure cases cannot show where VLMs fail in understanding human activities from thermal video, nor how much additional observation and reasoning can help. RGB video benchmarks also expose difficulties in fine-grained motion recognition [8], [9], although the different visual cues in thermal video limit direct transfer of these findings. As a result, thermal video-based human activity understanding for HRI still lacks a concrete basis for choosing activityunderstanding tasks, configuring perception pipelines, and prioritizing improvements.

We address this gap through a diagnostic evaluation of seven locally deployable VLMs on two person-centric thermal video datasets, testing three aspects of activity understanding: distinctions between specific actions, recovery of action–object interactions and their temporal extent, and the decision between an action answer and backoff. Paired RGB– thermal recordings and controlled changes in observation and inference examine how failures respond to different conditions, and models developed for physical AI or robotics are compared with their general-purpose base-model reference.

Our systematic evaluation yields four key observations:

• Recurring action confusions. Across seven models, five incorrect answer labels account for 42.9% of pooled exact-action errors. We also observe that recurring substitutions involve similar body movements or hand– object configurations, and these confusions from the test clips can cross functional categories, describing physiological events or potentially urgent behavior as routine activities, such as coughing labeled as touching the face and staggering as walking.

• Activity understanding errors. In long clips, models recover only 30.2%–51.8% of annotated events with the correct action and object labels, even without requiring accurate temporal boundaries. For harder joint tasks, their predictions also contain inaccurate temporal boundaries and hallucinated actions involving objects.

• Limited intervention gains. Across the two tested models, pseudocoloring, contrast enhancement, person cropping, denser frame sampling, and additional reasoning change exact-action accuracy by a limited 5.9% to +5.9% point relative to their respective paired baselines.

• Unreliable abstention. Models exhibit different answering tendencies but struggle to balance rejecting unsupported answers with retaining valid ones, even under explicit instructions. For example, Qwen3.6-27B correctly abstains on only 6.1% of questions without a valid candidate, whereas RoboBrain2.5 answers only 3.2% of questions containing one. We conclude this behavior as unacceptable for practical robotic application.

## II. RELATED WORK

a) Human Activity Understanding for Robots: Visual human activity understanding has been studied for robotic assistance through action–object relationships and object affordances [10], [11]. Recent work also investigates finegrained action recognition and temporal segmentation in assembly, using pose cues, visual context, and vision–language models [12], [13], [14]. In caregiving, OpenRoboCare [15] shows that video-captioning models still omit fine-grained details and misidentify actions. These studies examine activity understanding in concrete robotic settings, but their results with RGB or RGB-D observations leaves open how reliably the same action distinctions and temporal structure can be recovered from thermal observations, where the available visual cues differ.

b) Thermal Human Sensing and Activity Understanding: Thermal human sensing spans pedestrian detection, motion analysis, and activity understanding. Benchmarks such as LLVIP [16] support pedestrian detection under poor illumination, establishing a foundation for locating people without evaluating the activities they perform. Beyond detection, InfAR [17] investigates infrared action classification, while OrganicHAR [18] discovers activity categories suited to privacy-preserving sensors, including thermal arrays. Thermal-IM [19] exploits heat traces left by human–object contact to infer past human poses, motivated by household robotic assistance. Building on Thermal-IM, CREMA [20] constructs ThermalQA to evaluate multimodal fusion for inferring past human actions. Recent work also evaluates VLMs on thermal activities. A zero-shot thermal recognition study [3] reports stronger recognition of large body movements than fine gestures and safety-related actions, while CUHK-X [21] broadens multimodal evaluation to activity descriptions, contextual understanding, and action ordering. These findings establish both the potential of thermal activity perception and limitations that remain relevant to its use in robots.

Together, these studies highlight the importance of human activity understanding for robot perception and HRI, and explore thermal sensing as a promising source of information for these applications. Building on these studies, we take the evaluation of thermal activity perception for robots a step further. Beyond assessing whether models can recognize human activities under various environments, we design controlled experiments specialized for characterizing where errors concentrate, which actions are repeatedly confused, and which failure patterns are shared across models or specific to them.

## III. EVALUATION DESIGN

We structure the evaluation around three aspects of thermal activity understanding: distinguishing specific actions, recovering action–object interactions and their temporal extent, and choosing between an action answer and backoff. OctoNet [22] provides short single-action clips for action recognition and candidate-set interventions, while Thermal-IM [19] provides recordings with annotated interactions for recognition and temporal grounding. Figure 1 provides examples of the thermal observations, questions, and reference answers.

## A. Fine-grained Action Recognition

We manually select intervals from the OctoNet recordings in which each participant repeatedly performs the labeled action. After excluding one all-black clip, the evaluation set contains 865 thermal clips at 320 240 resolution, approximately 15 seconds each, covering 62 actions and 20 participants.

Each question asks the model to identify the observed action from 20 candidates and return its option letter (Fig. 1a). To probe distinctions between similar actions, we select the 19 distractors using rules that prioritize predefined groups of visually or semantically related actions, then fill remaining slots from other labels using a fixed seed. Candidate identities and order are shared across models, and random-choice accuracy is 5%. Exact-action accuracy measures selection of the labeled action, while per-action accuracy identifies which actions are repeatedly missed. To distinguish finelabel confusions from errors crossing broader activity types, we map the same predictions and ground-truth labels to 12 functional activity categories. We defined the mapping from the meanings of the action labels, grouping activities by everyday purpose or health and safety relevance, as motivated by the Index of ADL [23] and OrganicHAR [18]. The same mapping was applied to ground-truth and predicted labels across all models. For example, walking and turning are grouped as locomotion, squats and jumping rope as exercise, and hugs and handshakes as affiliative contact. Categorywise accuracy is the fraction of predictions whose functional categories match the ground truth.

## B. Interaction Recognition and Temporal Grounding

The official Thermal-IM test split contains 118 thermal recordings at 384 288 resolution with 199 annotated action– object events. The recordings span 3.0–143.2 seconds (median 46.5), and 55 contain multiple annotated events. Each event specifies an action, an object, and a start and end time in seconds, kept at their original annotation precision. We use synchronized RGB recordings for modality comparisons.

a) Joint task: The model identifies every observed interaction from the dataset-wide action and object vocabularies and returns a chronological list of (action, object, start time, end time) entries (Fig. 1b). This task evaluates both which interactions are recovered and what video segment they are located in the recording.

![](images/6097e37f49038fdba86412736027192d098fedc549f52f9f923644ff44fc4cc3.jpg)

## (b) Thermal-IM: joint interaction grounding

![](images/8db05da0cf058d7bc32da5fc8698c07ad04a1468a32ca0f0cb71b8158d0cdedb.jpg)  
Q: Which action-object interactions occur, and when?

![](images/075abc04aee0074ddab06b1d3356dc4c53ee57e781a1495ebf9280ab063c28bb.jpg)  
Temporal grounding with given interactions: leg stretching + yoga mat → []  
Fig. 1. Task examples: (a) fine-grained action recognition and (b) action–object interaction temporal grounding.

b) Temporal grounding with given interactions: To evaluate temporal grounding with known action and object identities, the model is given all distinct annotated action– object pairs in each recording, together with one absent pair. It is asked to return the start and end times of every occurrence of each given pair, or an empty list if the pair is absent (Fig. 1b). Comparison with the joint task shows how temporal grounding changes when action and object identities are known.

c) Scoring: For predicted and annotated intervals $I _ { p } =$ $[ s _ { p } , e _ { p } ]$ and $I _ { g } = [ s _ { g } , e _ { g } ]$ , temporal intersection over union is

$$
\operatorname { t I o U } ( I _ { p } , I _ { g } ) = { \frac { | I _ { p } \cap I _ { g } | } { | I _ { p } \cup I _ { g } | } } ,
$$

using continuous interval lengths, $\left| I \right| = e - s .$ . Within each recording, we match predictions and annotations with the same action–object pair one-to-one, maximizing total tIoU.

Pooling counts across recordings, let $N _ { p }$ and $N _ { g }$ denote the numbers of predicted and annotated events. Let $M _ { \mathrm { p a i r } }$ be the number of matched pairs without a temporal-overlap threshold, and $M _ { 0 . 5 }$ the number of those matches with tIoU 0.5. We report

$$
\mathrm { F 1 _ { p a i r } } = \frac { 2 M _ { \mathrm { p a i r } } } { N _ { p } + N _ { g } } , \qquad \mathrm { F 1 _ { e v e n t } } = \frac { 2 M _ { 0 . 5 } } { N _ { p } + N _ { g } } .
$$

Pair F1 measures interaction recovery under the requested output format; event F1 additionally requires sufficient temporal overlap. Mean tIoU summarizes temporal agreement over matched events. All three measures range from 0 to 1, with higher values indicating better interaction recovery and temporal agreement. Invalid temporal outputs count as empty predictions, contributing missed events to both F1 scores.

d) Presence test for temporal interventions: A separate 15-choice question asks which interaction is supported by the video, offering 14 action–object pairs and a None option. This test examines interaction selection without requiring temporal boundaries. We measure the fraction of responses selecting the original target interaction across completeevent clips, controls without an annotated target event, and temporal-order transformations, keeping the question and candidates unchanged. The intervention conditions are described in Section IV.

## C. Candidate-set Backoff

We test how the availability of the labeled action changes a model’s decision to answer or back off. This evaluation uses the same underlying principle as absent-answer detection in MM-UPD [24]: testing whether models can withhold an answer when the correct candidate is unavailable. We use OctoNet rather than Thermal-IM to keep the backoff analysis directly comparable with the single-action recognition baseline, using identical clips and matched distractors across questions with and without the labeled action. For each OctoNet question, the ground truth (GT)-present version retains the original 20 actions, while the GT-absent version replaces the labeled action with another distractor, preserving the other 19. Both versions add the same backoff option, None of the listed actions / Cannot determine from the video, for 21 choices in total. We evaluate these paired questions under the instruction settings described in Section IV.

Correct-backoff rate is the fraction of GT-absent questions for which the model selects backoff; false-backoff rate is the corresponding fraction on GT-present questions. For GTpresent questions, coverage measures the fraction receiving an action answer, and answered accuracy measures exactaction accuracy among those answers. Together, these measures characterize both the model’s tendency to answer and the correctness of the action labels it retains.

## IV. EXPERIMENTAL SETUP

## A. Models and Inference

We evaluate seven locally deployable VLMs without task-specific adaptation to the evaluation data. The model set includes Qwen3.5-4B [25], Qwen3.6-27B [26], Qwen3.8-27B [27], Gemma-4-31B-IT [28], Qwen3-VL-8B-Instruct [29], Cosmos3-Nano Reasoner [30], and RoboBrain2.5-8B-NV [31]. Qwen3-VL serves as the general-purpose base-model reference for Cosmos3 and RoboBrain2.5, allowing us to examine whether additional training for physical AI or robotics improves thermal activity understanding.

Since robotic use requires responses fast enough to inform ongoing behavior, we adopt direct answers without extended reasoning as the primary setting and evaluate additional reasoning separately. All models are served locally with vLLM on a single NVIDIA RTX PRO 6000 Blackwell GPU; models in the 30B range run from NVFP4-quantized checkpoints, while the smaller models run at native precision. Models receive native video inputs sampled by their video processors at a target rate of 5 FPS unless otherwise stated. For both the joint task and temporal grounding with given interactions, frames display their elapsed time in seconds, and both predicted intervals and annotations use continuous seconds (Fig. 1b).

## B. Controlled Comparisons

The frame-sampling, input-transformation, reasoning, and temporal-evidence comparisons modify the thermal input or its processing, so we run them with two representative models, Qwen3.5-4B and Qwen3.6-27B. The RGB–thermal and explicit-backoff comparisons leave the thermal input unchanged and cover all seven models.

a) Frame sampling: We examine whether denser observations help distinguish actions and whether the benefit continues as more frames are provided. We compare target rates of 2, 5, and 10 FPS on all 865 single-action clips, starting from the Qwen video processor’s default rate of 2 FPS [25].

b) Thermal input transformations: Hand and face movements often occupy only a small part of a thermal frame. We test whether changing the presentation of these cues helps distinguish similar actions. JET pseudocolor and percentile-based contrast enhancement emphasize intensity differences; a tracked person crop enlarges the actor; and a two-view input combines a tracked full-person crop with its upper-body crop to emphasize hand and face movements while retaining context. We also estimate body keypoints from the thermal clips using ViTPose++ [32] and provide synchronized skeleton videos alongside the thermal input. After visually checking crop and skeleton quality and excluding unreliable outputs, we evaluate 293 clips from five participants, covering all 62 actions with three to five clips per action. Each transformation is compared with its own matched baseline run.

c) Reasoning: Reasoning modes allocate additional computation before producing an answer, which often improves performance on general tasks [28]. We test whether this benefit extends to interpreting thermal actions by comparing direct answers with reasoning-enabled inference on 289 clips from the five-participant subset used for input transformations, excluding four tap clips whose labels do not precisely describe the recorded motion. We inspect the reasoning chains of 12 changed predictions to examine the visual interpretations accompanying corrected and newly introduced errors.

d) RGB–thermal comparison: We compare synchronized RGB and thermal recordings in both the joint task and temporal grounding with given interactions, using the same inference settings. We report action–object recognition and temporal grounding separately to characterize how the modality gap differs between identifying interactions and locating them.

e) Temporal-evidence intervention: We examine how event content and temporal order affect interaction-presence answers. For each of the 118 Thermal-IM recordings, we extract a complete annotated occurrence of the target interaction, choosing the longest occurrence when it repeats. Each event clip is paired with an equal-duration clip of the same person performing other activities, drawn from a recording with no annotated target action, while the question and candidates stay unchanged. To perturb temporal order, we reverse the target event or cut it into consecutive blocks of at most three seconds each (at least four blocks) and shuffle them. Shuffling preserves all frames and within-block order with their original playback speed and same 5 FPS sample rate.

f) Backoff settings: All seven models are evaluated on the 865 paired GT-present and GT-absent questions with an explicit instruction to select backoff when the model identi fies that no listed action is supported or the visual evidence is insufficient. For Qwen3.5-4B, Qwen3.6-27B, Qwen3.8- 27B, and Gemma-4-31B-IT, we also evaluate an optiononly setting without this instruction. Both settings retain the same backoff option and candidate wording, allowing us to examine how explicit guidance changes answering and backoff behavior beyond providing the backoff option alone.

## V. RESULTS AND ANALYSIS

## A. Fine-grained Action Recognition

a) Confusions between specific actions: The results reveal a gap between broad activity recognition and finegrained action identification (Fig. 2a). Large body movements are often recognized, while short hand-to-face actions, physiological events, and gestures with similar trajectories remain difficult.

Across the seven models, the five most frequent incorrect answer labels (waving, lifting a hand, touching the face, drinking, and turning) account for 42.9% of all exactaction errors. Recurring substitutions include brushing teeth predicted as drinking, yawning and coughing as touching the face, a stop gesture as lifting a hand, and dodging as turning. These errors show repeated confusion between actions sharing similar body movements or hand–object configurations.

b) Consequences for activity meaning: When mapped to functional categories, 50.8%–61.1% of each model’s exact-action errors retain the correct broad category. But these similar movement-level confusions can also cross category boundaries (Fig. 2b–c). In the tested clips, staggering is predicted as walking or turning, while pushing or punching someone is predicted as conversation. These substitutions present potentially urgent behavior as ordinary movement or communication, losing the distinction between safety-related and non-urgent activities in the resulting descriptions.

(a) Overall accuracy  
![](images/011a59c904934c5518b940600a00b3a0c13fb3f5eba4dd7f8e6612f5b1b212d2.jpg)

![](images/2d8b2c37b4252687be3b07fa5d46fadd7dccee05165b0ecd090d8c406c8b24b6.jpg)

(c) Pooled confusion  
![](images/fc3c2533d0de29d3cb986f7eadb0126b4060d2f875f2e95e5220b071a17e3184.jpg)  
IDs 1-12 as in (b); row-normalized (%). Cells below 5% are unlabelled.  
Fig. 2. Action recognition on 865 thermal clips. (a) Exact and category-wise accuracy. (b) Per-category recall (clip counts in parentheses). (c) Pooled category confusion, row-normalized.

TABLE I  
FRAME SAMPLING AND INPUT TRANSFORMATIONS.
<table><tr><td>Setting</td><td>Qwen3.5-4B Qwen3.6-27B</td></tr><tr><td>Exact-action accuracy (%), 865 clips</td><td></td></tr><tr><td>2 FPS</td><td>37.8 46.9</td></tr><tr><td>5FPS</td><td>42.9 52.3</td></tr><tr><td>10 FPS</td><td>43.7 52.7</td></tr><tr><td>Accuracy change (pp), 293 clips</td><td></td></tr><tr><td>JET -2.0</td><td>+0.3</td></tr><tr><td>Contrast +1.0</td><td>+0.7</td></tr><tr><td>Person crop</td><td>+2.7 +2.7</td></tr><tr><td>Person crop + upper body*</td><td>0.0 +3.4</td></tr><tr><td>Thermal + skeleton</td><td>+1.0 -3.1</td></tr></table>

∗ Compared with its matched full-person-crop baseline.

c) Sampling and thermal presentation: Denser sampling and changes in thermal presentation give partial answers to these confusions (Table I). Both models improve in exact-action accuracy when the target sampling rate rises from 2 to 5 FPS, with little further gain at 10 FPS. Person cropping provides small gains for both models, whereas JET and contrast enhancement do not help. Skeleton and upperbody views have model-dependent effects: adding skeletons helps Qwen3.5-4B but hurts Qwen3.6-27B, while the full-plus-upper-body input benefits Qwen3.6-27B but leaves Qwen3.5-4B unchanged. These observation-side changes alleviate some errors without consistently resolving finegrained action confusions.

d) Reasoning: Additional reasoning does not resolve these confusions either. For Qwen3.5-4B, 25 corrections are offset by 24 new errors, leaving accuracy nearly unchanged. For Qwen3.6-27B, 30 new errors outweigh 13 corrections and reduce accuracy by 5.9 percentage points. The inspected reasoning chains include misidentified objects, confused body poses or motion patterns, and assumptions about unclear visual details. Some repeatedly reconsider candidates or acknowledge ambiguity before choosing a specific interpretation, including cases that overturn a correct direct answer.

## B. Interaction Recognition and Temporal Grounding

a) Joint task: In longer recordings, errors concern not only when events occur, but which interactions are recovered. The joint predictions recover only part of the annotated interactions (Fig. 3a). Missed or misidentified events and invalid responses reduce pair F1, and requiring temporal overlap further reduces event F1 because a correctly named interaction can still receive inaccurate boundaries.

Differences in interaction recovery are not captured by isolated-action accuracy alone. Cosmos3 and Qwen3-VL have similar exact-action accuracy (43.7% and 44.5%), but their thermal pair F1 scores are 0.257 and 0.371, respectively; RoboBrain2.5 reaches 0.248. Thus, the additional physical-AI or robotics specialization does not provide a consistent advantage in recovering these interactions under direct answering.

Figure 4 gives an example result of the four-event recording of Fig. 1b. For sit–stool at 5.0–28.3 s, Qwen3.6-27B returns 4.0–28.0 s (tIoU = 0.95), Cosmos3 returns a 1.6- second fragment at the start, while RoboBrain2.5 reports one invented interaction leg stretching–stool at 32–88 s. This example shows that same event can be either localized almost exactly or reduced to a fragment by another, and replaced by an unrelated record from different models.

b) Temporal grounding with given interactions: With action–object pairs given, the task evaluates their temporal localization. Event F1 rises in all seven model comparisons once the interaction identities are given, with scores ranging from 0.256 to 0.699 (Fig. 3b). We also observe that mean tIoU is higher in the joint task (0.53–0.87 versus 0.48– 0.77) when compared on a pair match. This gap is largely compositional rather than behavioral: on events recovered in both tasks, given identities yield equal or better boundaries for six of the seven models; the joint task simply skips the harder occurrences that the task with given interactions must attempt.

(a) Joint interaction recovery  
![](images/1586f227d010d055373fb7e55f48235d95685ff66fc4ab67a3997a3d43cd7bb9.jpg)

(b) Temporal grounding with given interactions  
![](images/b741da4520a0fcc92c18827a4a0e01ba66b89d2ea5f1d5082703bac57618e8f7.jpg)

Fig. 3. Interaction recognition and temporal grounding on 118 thermal recordings. (a) Joint pair and event F1. (b) Event F1 for the joint task and temporal grounding with given interactions.  
![](images/a9bf3994dc592ff0ce1f1738217f660465939bd084f485021b10bd09002650b7.jpg)  
Fig. 4. The sit–stool event in Fig. 1b: Qwen3.6 recovers it, Cosmos3 predicts a fragment, and RoboBrain2.5 omits it.

c) Paired RGB–thermal comparison: The paired RGB comparison changes the visual input instead of the task information. RGB yields higher action–object pair F1 than thermal input across all seven models (Fig. 5a). The mean relative change in correctly matched event counts from thermal to RGB is 59.0%. Qwen3-VL also outperforms Cosmos3 and RoboBrain2.5 in pair F1 under RGB input, indicating that their lower scores relative to the base-model reference are not confined to thermal video. Figure 5b shows that RGB also recovers more annotated events for every model, while mean tIoU over matched events changes by at most 0.07 for all but Cosmos3 and RoboBrain2.5, for which it improves.

d) Temporal-evidence intervention: A complementary interaction-presence test examines whether recognition errors remain when complete target events are shown and no boundaries are requested. Selection of the target label drops by more than 50 percentage points for both models when the target event is replaced with an unrelated activity (Fig. 6a). This shows that their answers respond to the activity shown, but we still observe missed correct label for complete annotated target events, which means that relevant event content cannot fully resolve the interaction confusions.

Reversing events or shuffling blocks of at most three seconds changes 16.1–19.5% of answers without a reliable decline in target selection. This pattern is understandable given the prevalence of sustained postures and object-contact interactions in the evaluated events. Posture and contact cues remain visible after reordering. Together with the sharp decline under content replacement, these results suggest that the models can exploit spatial evidence in thermal frames, while temporal order is less decisive for these interactionpresence judgments.

TABLE II  
EXPLICIT-BACKOFF RESULTS (%).
<table><tr><td>Model</td><td></td><td>Coverage Answered acc. Correct backoff</td><td></td></tr><tr><td>Qwen3.5-4B</td><td>50.1</td><td>55.0</td><td>67.7</td></tr><tr><td>Qwen3.6-27B</td><td>97.2</td><td>52.1</td><td>6.1</td></tr><tr><td>Qwen3.8-27B</td><td>85.9</td><td>49.1</td><td>25.0</td></tr><tr><td>Gemma-4-31B-IT</td><td>92.7</td><td>45.0</td><td>16.1</td></tr><tr><td>Qwen3-VL-8B</td><td>93.5</td><td>46.2</td><td>15.3</td></tr><tr><td>Cosmos3-Nano</td><td>82.2</td><td>44.7</td><td>31.2</td></tr><tr><td>RoboBrain2.5-8B</td><td>3.2</td><td>67.9</td><td>98.5</td></tr></table>

## C. Candidate-set Backoff

A further question is whether allowing models to withhold an answer improves their performance. Backoff exposes two opposite failures beyond action recognition (Table II). Under explicit instructions, Qwen3.6-27B, Qwen3- VL, and Cosmos3 continue answering incorrectly on most GT-absent questions. At the other extreme, RoboBrain2.5 rejects 98.5% of GT-absent questions but answers only 28 of 865 GT-present questions; its 67.9% answered accuracy therefore accompanies only 3.2% coverage. The same model reaches 36.8% exact-action accuracy without a backoff option (Fig. 2a): it is not simply unable to classify the clips, yet it almost stops answering once a way out exists.

a) Instruction comparison: The paired instruction comparison examines whether explicit guidance prompt can adjust this tendency. When no explicit guidance is given, all four models rarely back off on GT-absent questions, with correct backoff between 6.8% and 13.8%. With the instruction added, Qwen3.5-4B rejects more GT-absent questions but also backs off on roughly half of the GT-present questions; its answered accuracy rises from 44.1% to 55.0% as coverage falls. Qwen3.8-27B and Gemma-4-31B-IT exhibit milder versions of this trade-off. Qwen3.6-27B instead becomes more willing to answer when the correct action is absent. The same instruction therefore shifts models toward different answering behaviors rather than uniformly improving backoff. These responses make backoff difficult to rely on without evaluating the chosen model: explicit guidance can suppress valid answers or fail to prevent unsupported ones.

(a) Action–object recognition  
![](images/cf6237614ff93931215e55c0e89fd89e35979a1f6696d740b1990b48982aaf56.jpg)

![](images/d76677f6242c413cf6fbf83d0168a933dcc3c01f7f3f050b55ddbac920a31320.jpg)  
Fig. 5. RGB–thermal comparison in the joint task. (a) Pair F1. (b) Thermal (open) to RGB (filled) operating points: pair-matched event counts and mean tIoU over matched events. Scores use elapsed seconds.

![](images/82a2c83b3f155ad1f6606bb58b7a01e8d086a33afe2ca114f355b56bbddd0bfe.jpg)

![](images/91e192e3d1dabf2ce95cc50c42100b3de39bd7ad4e8fd5faa44a74711592afd7.jpg)  
Fig. 6. Temporal-evidence interventions for tested models. (a) Target selection with target events vs. unrelated activities. (b) Target selection and answer changes after reversing or block-shuffling events.

## VI. CONCLUSION

This paper examines where VLMs fail in understanding human activities from thermal video by evaluating seven locally deployable models on two person-centric datasets along action recognition, interaction grounding, and candidate-set backoff. The failures concentrate in three places. Models often recover the broad activity but not the specific action: Qwen3.6-27B reaches 52.3% overall accuracy yet only 16.7% on coughing, sneezing, and yawning, and five labels account for 42.9% of all action errors. Our evaluation result shows that these confusions can cross functional categories, presenting health-related or potentially urgent behavior as routine activities: coughing is labeled as touching the face, and staggering as walking or turning. Meanwhile, models recover only part of what happened and when: joint pair F1 stays between 0.248 and 0.487, and a correctly named event can still receive an unusable interval, leaving records that are plausible but wrong. And models cannot be relied on to decide whether to answer: with backoff available, Qwen3.6- 27B answers 97.2% of GT-present questions but correctly backs off on only 6.1% of GT-absent questions, while

RoboBrain2.5 answers just 3.2% of GT-present questions.

Even with complete event views, models continue to confuse interactions, while the tested changes to frame sampling, thermal input presentation, and reasoning provide only limited or inconsistent gains. RGB improves action– object pair F1 across all seven models, and supplying interaction identities improves event F1. These comparisons point to action and object recognition as a major bottleneck. Reliable use in robots also requires event labels and intervals supported by the video, together with backoff that rejects unsupported interpretations while retaining valid answers.

## VII. LIMITATIONS AND FUTURE WORK

Three limitations bound these findings. First, the activity taxonomies and recording protocols of OctoNet [22] and Thermal-IM [19] reflect their origins in multimodal sensing and human-motion analysis. Their fixed camera setups and activity distributions differ from what a robot would encounter in a given household, and our 5 FPS clip-based evaluation includes neither active observation nor the coupling between activity interpretation and robot behavior. Second, our analysis is behavioral: we characterize how model outputs change across observation and inference conditions, but not where inside a model the failures arise, for example whether thermal cues are lost in visual encoding, or whether temporal evidence is integrated across frames at all. Layer-wise analyses [33] offer a complementary route to such mechanistic questions. Third, our tasks measure recognition, localization, and answering behavior, but they remain proxies for robotic applications. We note that a thermal-video benchmark that evaluates whether a robot responds appropriately to human activity is still missing.

Action understanding remains a hard engineering problem, and our results show that thermal video makes it harder. Since no single change we tested resolves these errors, we propose combining thermal-specific adaptation, complementary sensing, and active observation. Thermalspecific adaptation should target the recurring action and object confusions identified here, from specialized motion representations and recognizers [34] to thermal-adapted encoders. Synchronized complementary sensors can supply cues that thermal video resolves poorly, with fusion evaluated by whether it corrects specific confusions rather than by aggregate scores. A robot can also act: revisiting a suspected onset, moving closer to an ambiguous object, or choosing a new viewpoint turns observation into an active process. Agentic video-understanding workflows such as PyVision-RL [35] extend this idea, selecting relevant clips and checking uncertain interpretations against stored observations; their value should be weighed against the additional observation and computation they require.

## REFERENCES

[1] Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, et al., “π<sub>0.7</sub>: A steerable generalist robotic foundation model with emergent capabilities,” arXiv preprint arXiv:2604.15483, 2026.

[2] E. Griffiths, S. Assana, and K. Whitehouse, “Privacy-preserving image processing with binocular thermal cameras,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 1, no. 4, pp. 1–25, 2018.

[3] H. Hu and H. Chen, “Can vision language models zero-shot thermal images for private and robust human robot interaction?” in Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction, 2026, pp. 238–242.

[4] Z. Cao, J. Zhang, and R. Zhang, “IRGPT: Understanding real-world infrared image with bi-cross-modal curriculum on large-scale benchmark,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 166–176.

[5] A. Shrivastava, K. Gangani, L. Jain, M. Goel, and N. Batra, “ThermEval: A structured benchmark for evaluation of vision-language models on thermal imagery,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2026, pp. 9789–9800.

[6] M. Moshtaghi, S. H. Khajavi, and J. Pajarinen, “RGB-Th-Bench: A dense benchmark for visual-thermal understanding of vision language models,” arXiv preprint arXiv:2503.19654, 2025.

[7] R. Thushara, Y. Ranasinghe, J. Paranjape, and V. M. Patel, “Thermo-VL: Extending vision-language models to thermal imaging,” arXiv preprint arXiv:2605.21882, 2026.

[8] W. Hong, Y. Cheng, Z. Yang, W. Wang, L. Wang, X. Gu, S. Huang, Y. Dong, and J. Tang, “Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 8450–8460.

[9] C. Tu, L. Zhang, P. Ye, X. Zeng, W. Cheng, G. Yu, T. Chen, et al., “Favor-bench: A comprehensive benchmark for fine-grained video motion understanding,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[10] C. L. Teo, Y. Yang, H. Daume, C. Ferm ´ uller, and Y. Aloimonos, ¨ “Towards a watson that sees: Language-guided action recognition for robots,” in 2012 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2012, pp. 374–381.

[11] H. S. Koppula, R. Gupta, and A. Saxena, “Learning human activities and object affordances from RGB-D videos,” The International journal of robotics research, vol. 32, no. 8, pp. 951–970, 2013.

[12] S. W. Hyder, M. Usama, A. Zafar, M. Naufil, F. J. Fateh, A. Konin, M. Z. Zia, and Q.-H. Tran, “Action segmentation using 2D skeleton heatmaps and multi-modality fusion,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 1048–1055.

[13] H. Zheng, R. Lee, H. Liang, Y. Lu, and X. Xu, “DuCAS: A knowledgeenhanced dual-hand compositional action segmentation method for human-robot collaborative assembly,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7175–7180.

[14] H. Zheng, J. Huang, T. Zheng, X. Xu, and T. Alhanai, “Compositional context fine-tuning vision-language model for complex assembly action understanding from videos,” in 2026 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026. [Online]. Available: https://arxiv.org/abs/2607.10797

[15] X. Liang, Z. Liu, K. Lin, E. Gu, R. Ye, T. Nguyen, C. Hsu, Z. Wu, X. Yang, C. S. Y. Cheung, et al., “OpenRoboCare: A multimodal multi-task expert demonstration dataset for robot caregiving,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 2661–2668.

[16] X. Jia, C. Zhu, M. Li, W. Tang, and W. Zhou, “LLVIP: A visibleinfrared paired dataset for low-light vision,” in 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). IEEE, 2021, pp. 3489–3497.

[17] C. Gao, Y. Du, J. Liu, J. Lv, L. Yang, D. Meng, and A. G. Hauptmann, “InfAR dataset: Infrared action recognition at different times,” Neurocomputing, vol. 212, pp. 36–47, 2016.

[18] P. Patidar, R. Arakawa, R. Grac¸a, R. Moutinho, A. Soares, A. Vasconcelos, F. Talami, J. C. da Silva, I. Silva, C. Mendes Santos, et al., “OrganicHAR: Towards activity discovery in organic settings for privacy preserving sensors using efficient video analysis,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 9, no. 4, pp. 1–32, 2025.

[19] Z. Tang, W. Ye, W.-C. Ma, and H. Zhao, “What happened 3 seconds ago? inferring the past with thermal imaging,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 17 111–17 120.

[20] S. Yu, J. Yoon, and M. Bansal, “CREMA: Generalizable and efficient video-language reasoning via multimodal modular fusion,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 74 382–74 406.

[21] S. Jiang, M. Yuan, X. Ji, B. Yang, Z. Liu, L. Xu, Y. Li, Y. He, L. Dong, W. Lu, et al., “A large-scale multimodal dataset and benchmarks for human activity scene understanding and reasoning,” in Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services, 2026, pp. 352–370.

[22] D. Yuan, X. Zhang, W. Hou, S. Lyu, Y. Yu, L. J.-T. Yu, C. Li, and C. Wu, “OctoNet: A large-scale multi-modal dataset for human activity understanding grounded in motion-captured 3D pose labels,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[23] S. Katz, A. B. Ford, R. W. Moskowitz, B. A. Jackson, and M. W. Jaffe, “Studies of illness in the aged: the index of ADL: a standardized measure of biological and psychosocial function,” JAMA, vol. 185, no. 12, pp. 914–919, 1963.

[24] A. Miyai, J. Yang, J. Zhang, Y. Ming, Q. Yu, G. Irie, Y. Li, H. H. Li, Z. Liu, and K. Aizawa, “Unsolvable problem detection: Robust understanding evaluation for large multimodal models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, July 2025, pp. 6497–6540. [Online]. Available: https://aclanthology.org/2025.acl-long.326/

[25] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5

[26] ——, “Qwen3.6-27B: Flagship-level coding in a 27B dense model,” April 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.6-27b

[27] ——, “Qwen3.8-Max: A new bar for coding and cowork,” August 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.8

[28] Gemma Team, “Gemma 4 technical report,” 2026. [Online]. Available: https://arxiv.org/abs/2607.02770

[29] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[30] N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A. Azzolini, J. Bai, M. Bala, Y. Balaji, J. Bapst, et al., “Cosmos 3: Omnimodal world models for physical AI,” arXiv preprint arXiv:2606.02800, 2026.

[31] H. Tan, E. Zhou, Z. Li, Y. Xu, Y. Ji, X. Chen, C. Chi, P. Wang, H. Jia, Y. Ao, et al., “RoboBrain 2.5: Depth in sight, time in mind,” arXiv preprint arXiv:2601.14352, 2026.

[32] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “ViTPose++: Vision transformer for generic body pose estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 2, pp. 1212– 1230, 2023.

[33] X. Liu, Y. Li, X. Zhuang, X. Li, Z. Lu, H. Yang, I. Razzak, and Y. Xie, “Grounding isn’t knowing: Do VLMs need object localization for spatial reasoning?” arXiv preprint arXiv:2608.23074, 2026.

[34] K. Yokoyama and N. Ukita, “A plug-and-play 2D motion interface for real-world motion language models,” arXiv preprint arXiv:2608.15984, 2026.

[35] S. Zhao, S. Lin, M. Li, H. Zhang, W. Peng, K. Zhang, and C. Wei, “PyVision-RL: Forging open agentic vision models via RL,” arXiv preprint arXiv:2602.20739, 2026.