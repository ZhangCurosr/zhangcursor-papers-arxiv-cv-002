# Teach a Molmo2Fish: Towards interactive fish tracking with natural language guidance

Kai van Brunt, Justin Kay , and Sara Beery

Massachusetts Institute of Technology, Cambridge MA 02139, USA Correspondence to kav@mit.edu

Abstract. Computer vision is increasingly used to automate recognition tasks in large ecological datasets, but more complex tasks such as multi-object tracking continue to pose challenges. As researchers seek to incorporate vision models in ecology workflows, various lines of research have explored how to make imperfect predictions useful through humanin-the-loop processes. We propose a new approach to working with imperfect tracking predictions through an interactive prediction correction workflow taking place as a conversation with a multimodal large language model, which we tailor to a sonar fish tracking dataset as an initial proof of concept. We investigate the performance of the tool, Molmo2Fish, across guided and unguided tasks, correcting its own predicted tracks and external tracks. We find that Molmo2Fish achieves high performance on fish tracking and track correction tasks, but there is still much room to improve on incorporating natural language guidance. The code and data are publicly available at github.com/tidalove/molmo2fish.

Keywords: Multi-object tracking · Multimodal large language model · Human-AI interaction

## 1 Introduction

Computer vision is increasingly used in ecology and conservation to automate recognition tasks in large datasets, including species classification [15,19], object detection [3], segmentation [34], and multi-object tracking [12, 23]. While detection and classification models have proven to be robust across many deployment scenarios in ecology [3,15], more complex tasks such as multi-object tracking continue to pose challenges and require specialization to individual deployments [23]. Improving the performance of tracking models could have significant impacts for ecology, enabling conservation use cases such as behavior analysis [31], population monitoring [1, 30, 34], and real-time response to ecological incidents [7, 12].

In particular, we consider the use case of counting migrating salmon in sonar video [1, 23]. Sonar video cameras are deployed underwater in rivers to count the number of salmon migrating each year to spawn, informing the setting of fishing quotas and enabling monitoring of population response to conservation actions. Stakeholders have begun incorporating computer vision-based processing to reduce the time needed to process sonar video into salmon counts. This processing involves a specialized detection model and tracker that tracks fish as they enter and exit the field of view; counts are then determined according to the number of tracks. While these specialized models perform well at river locations that appear in training data, their performance is brittle under environmental shift, causing prediction errors at new deployments and in changing environments [21, 23, 24].

![](images/e989ef09b77ad77a5edb6077bd862c7a8eec8984127d72c20a0615fe7e4b20b2.jpg)  
Fig. 1: Molmo2Fish is an interactive tool for creating and editing multi-object tracking predictions. Users interact with a multimodal LLM that has been fine-tuned for fish tracking (top) and correcting tracks with natural language (bottom).

Correcting erroneous tracking predictions involves collecting new ground truth annotations and re-training tracking models, presenting two barriers to practical use by fisheries stakeholders: (1) multi-object tracking annotations are notoriously time-consuming to collect, requiring users to iteratively draw and adjust bounding boxes in potentially crowded scenes with complex movement [28, 29]; and (2) successfully re-training and validating models requires machine learning and data science expertise, which many fisheries stakeholders do not yet have.

We propose a new approach to working with imperfect tracking predictions that addresses both of these problems simultaneously via an interactive prediction correction workflow taking place as a conversation with a multimodal large language model (MLLM). MLLMs are trained to process visual and text inputs jointly through a large language model (LLM) architecture. Recent work has demonstrated these capabilities for multi-object tracking, in which an MLLM is prompted in natural language to track an object of interest in a video, and outputs text tokens specifying the locations of that object in each frame [11]. We propose to extend these capabilities with a multi-turn interaction in which the MLLM’s initial predictions can be referenced and modified with additional natural language prompts (see Figure 1). This workflow addresses the problems above: instead of requiring manual labeling from scratch, users guide the model to produce better tracks; instead of requiring fine-tuning, the model updates its predictions in real-time conditioned on the user’s correction prompts.

We evaluate the potential of our proposed workflow by training Molmo2Fish, a fine-tuned version of the open MLLM Molmo2 [11], for tracking and iterative track correction in fisheries sonar video. Our experiments demonstrate first of all that MLLMs can indeed be specialized for fisheries sonar data through parameter-eficient fine-tuning [20], despite significant diferences between sonar video and the typical MLLM training set. We also demonstrate the potential to imbue MLLMs with multi-turn track correction capabilities that both improve overall tracking performance and target specific user-specified modifications. However, there is still much room for improvement, and we discuss the potential for future work to improve upon our approach by leveraging more data and compute intensive training. To the best of our knowledge, our experiments are the first to empirically evaluate the capabilities of MLLMs in fisheries sonar, as well as the first to investigate and validate the proposed multi-turn track correction interaction. In summary, our contributions are:

1. We propose a new approach to working with imperfect multi-object tracking predictions through a multi-turn interaction with a MLLM.

2. We introduce Molmo2Fish, a custom MLLM that is trained to enable this workflow for the ecological use case of tracking salmon in sonar video.

3. We empirically evaluate the performance of Molmo2Fish across both tracking and track correction tasks, providing insight into the current and future capabilities of using MLLMs for tracking and iterative prediction correction.

## 2 Related work

Multi-object tracking for ecology. Multi-object tracking (MOT) refers to the accurate localization and assignment to individual tracks of all objects of interest in a video. Traditionally, the most specialized and lightweight option has been the two-stage tracking-by-detection pipeline. An object detection model such as YOLO [9], Faster-RCNN [32], or SSD [26] first predicts per-frame detections and classifications of all objects of interest. Next, an of-the-shelf tracker such as SORT [5] or ByteTrack [35] links together detections across frames to form tracks. Such systems are often first-choice options for deployment because of their high speed and low demand for compute, but they are limited in their out-of-sample prediction ability, requiring additional supervised fine-tuning or domain adaptive training to address [21].

Ecological studies rely on detecting and tracking animals in videos to inform population monitoring [23, 34], behavioral analysis in the wild [14, 30] and in situ [18, 31], following animal populations in real time [12], preventing poaching [7], and preventing accidents [33]. Domain-relevant video object tracking datasets are collected and published to encourage development of deep learning methods that can infer such tracks from potentially huge amounts of video data eficiently and cost-efectively. We focus on the ecological use case of tracking fish in sonar video, building upon the Caltech Fish Counting (CFC) dataset [23]. CFC is the largest open computer vision dataset for fisheries sonar, containing MOT annotations for 1,500 videos sourced from four diferent sonar deployments. We extend CFC with correction trajectories including natural language annotations to enable training Molmo2Fish for our proposed workflow.

Multimodal LLMs. LLMs trained to interpret and answer questions about non-text modalities—image, audio, and video—have been growing in capability over the past few years. More recently, such multimodal LLMs (MLLMs) have been endowed with explicit video grounding capabilities: in addition to general video question answering, action recognition, and complex video understanding, increasingly MLLMs are trained to point at specific locations in a video as an auxiliary step in video reasoning, or as the final output to a pointing, tracking, or counting query. Gemini 3 Pro [17] leads performance among proprietary models, while PerceptionLM [6, 10] and Qwen3-VL [2] are examples of open models with such abilities. We elect to build on Molmo2 [11], a fully open MLLM that achieves state-of-the-art performance on MOT tasks. However, Molmo2—like most MLLMs—is trained predominantly on natural RGB footage. We show that it fails to generalize out of the box to fisheries sonar video, requiring fine-tuning.

Working with imperfect predictions. As researchers seek to incorporate imperfect machine learning models in ecology workflows and data processing pipelines, an increasingly popular line of research in ML asks how to make imperfect predictions useful through human-in-the-loop processes. Methods for active inference propose human-in-the-loop interactions that analyze or modify predictions directly. Existing approaches include statistical de-biasing [36], transductive prediction [25], and model selection [22].

In computer vision, interactive video object segmentation methods allow users to interact directly with predicted segmentations through point-and-click interactions that provide positive or negative feedback. Examples include SAM3 [8], which propagates the correction in a learned fashion to other tracks and other frames, and XMEM++ [4] and EVA-VOS [13] that additionally suggest the next most useful frame for the user to annotate. Molmo2Fish difers by operating on point tracks rather than segmentations and by interacting through natural language on an entire video rather than direct user action on individual frames.

## 3 Molmo2Fish

Molmo2Fish is a tool to interactively predict and correct tracks in video based on natural language prompts. Users participate in a multi-turn conversation with Molmo2Fish, prescribing specific corrections to tracks or nudging model predictions in the right direction by iteratively prompting the model with additional information (Figure 1). Initial predictions might be user-provided, generated by an external tracker, or generated by Molmo2Fish itself; Molmo2Fish is agnostic to the source of the tracks.

## 3.1 Model details

We build on Molmo2 [11], a fully open MLLM trained at scale on a mixture of tasks which require spatial reasoning, video understanding, and precise video localization. In particular, Molmo2 can perform multi-object tracking of objects including people, animals, and automobiles, on a visually diverse set of videos.

Molmo2’s architecture consists of three basic components, also illustrated in Figure 2. The vision encoder (SigLIP 2) ingests video frames (potentially downsampled) spanning the length of the clip, encoding them into a latent visual feature space. The connector projects vision features to LLM embedding space so that they can be treated the same as language tokens by an LLM. The LLM (Qwen3) stacks vision tokens and user’s text tokens in a single sequence and processes it.

Molmo2 outputs all responses as text. To specify the tracks it predicts for a video, it outputs a sequence of {t track\_id x\_coord y\_coord} strings, each denoting one instance’s (track\_id) position (x\_coord, y\_coord) on one frame (at time t in seconds). If an object of a particular track\_id is not present at time t then Molmo2 neglects to output it in that step of the sequence. The sequence order is fixed by sorting all ground truth points by (t, x\_coord, y\_coord), i.e. the earliest points in the sequence are those which come earliest temporally, followed by those with a lower x coordinate, then those with a lower y coordinate.

## 3.2 Instruction tuning for track correction

Molmo2’s video understanding and grounding capabilities form a strong starting point for our tool, but it was trained only for one-step, non-interactive tasks—it has no notion of referring back to a previous step to inform its next output. Hence, to build Molmo2Fish we fine-tune Molmo2 concurrently on two objectives: (1) a pure tracking task of the same format as tracking tasks in the Molmo2 training corpus, in order to improve domain-specific tracking performance, and (2) an interactive track correction task, which Molmo2 has not seen before, that enables interactive track correction as illustrated in Figure 1. In Section 4 we describe the composition of data used to train Molmo2Fish.

The text content of each training example for the pure tracking task takes the same form as tracking tasks in Molmo2’s training set. In our training, these examples look like:

User: ⟨video⟩ Track all fish.   
Assistant:

The text content of each training example for the track correction task takes the form of an artificial multi-turn conversation between the user and LLM which fits entirely in the LLM’s context window. A track correction example looks like:

![](images/e859e3a3bd042f174919f7c2649750af9505082f14debe777fca2d9ba9ef3f97.jpg)  
Fig. 2: Molmo2Fish architecture. Molmo2Fish consists of the same three components as the Molmo2 MLLM we build upon: a video encoder, a LLM, and a “connector” that projects video representations into the LLM’s representation space. Video frames and prompts are encoded and processed together, and outputs take the form of text token predictions. We extend the training protocol of the original Molmo2 to enable multi-turn corrections by constructing prompts as illustrated: first, the original video frames and a general prompt such as “Track all fish”; then, a simulated text response from Molmo2 containing erroneous predictions; and finally a second textual correction prompt.

User: ⟨video⟩ Track all fish. Assistant: ⟨corrupted\_point\_sequence⟩ User: ⟨correction\_prompt⟩ Assistant:

We refer to the sequence (i: {prompt, tracks}) for i in num\_steps constituting a track correction example as a trajectory. In practice, due to context window length limitations, in our experiments we train and evaluate on only the single-step case (one correction step).

The model is trained with cross entropy loss between its output logits and the tokens corresponding to the ground truth point sequence string. In a pure tracking task, this corresponds to the model’s immediate prediction (following “Track all fish”) of all tracks. In a track correction task, this corresponds only to the model’s response following the user’s correction prompt.

We limit our work to finetuning on a single dataset (described in Section 4.1) for faster training, easier correction prompt generation, and as a proof of concept that can motivate future work to develop a more general model.

## 4 Dataset

We design Molmo2Fish with a specific ecological use case: tracking fish in sonar video. To do so, we extend the Caltech Fish Counting (CFC) dataset [23], the largest open dataset of fisheries sonar video, to enable the interactive workflow described in Section 3. Here, we describe CFC in more detail and how we adapted it for Molmo2Fish, with more detailed dataset statistics in Tab. 6.

Table 1: Dataset tasks. Overview of all tasks used in the dataset, for both training and evaluation.
<table><tr><td>Name</td><td>Description</td><td>Splits</td></tr><tr><td>Pure tracking</td><td>Generic &quot;track all fish&quot; task, target is all points on all fish on all frames</td><td>train, val</td></tr><tr><td>Guided tracking</td><td>&quot;Track X fish&quot;, where X is some descriptor: &quot;track the last fish&quot;, &quot;track the top three fish&quot;, etc; target is points only on those identified fish</td><td>train, val</td></tr><tr><td>Synthetic track corrections</td><td>Tracking corrections generated synthetically</td><td>train, val</td></tr><tr><td>Targeted track corrections</td><td>Tracking corrections fixing only target mistakes</td><td>train, val</td></tr><tr><td>Real track corrections</td><td>Tracking corrections generated from real model predictions:</td><td></td></tr><tr><td>YOLO+SORT</td><td>Tracking corrections with pre-correction predic- tion from YOLO+SORT predictions</td><td>val only</td></tr><tr><td>Molmo-low</td><td>Tracking corrections with pre-correction predic- tion from step110 MoLMo2FisH checkpoint</td><td>train, val</td></tr><tr><td>Molmo-high</td><td>Tracking corrections with pre-correction predic- tion from step300 MoLMo2FisH checkpoint</td><td>train, val</td></tr></table>

## 4.1 Caltech Fish Counting

CFC consists of sonar videos of salmon annotated with bounding boxes and tracks. Tracks can be used to derive upstream and downstream counts of traveling salmon, which are used for estimation of salmon escapement statistics.

The original dataset contains data from four diferent sonar deployments: the Kenai left bank, Kenai right bank, Elwha, and Nushagak rivers. These locations are visually distinct and together span a range of tracking conditions in fisheries sonar data, from the Elwha river’s small, sparse fish to the Nushagak’s dense clusters of large fish.

We construct training and validation sets from the union of these river locations. We randomly select 25% of video clips for validation and train on the remainder for Elwha, Nushagak, and Rightbank. We use the original paper’s split of Kenai data. This training/validation split, which incorporates data evenly from across these locations, is diferent from the original CFC training/validation split, which was designed to evaluate OOD generalization by training on a single location and holding out every other location.

![](images/9524871964fafd3c9e1be54c1750987bf55c7568835e027bda412f7798a68d7f.jpg)  
Fig. 3: From a video and its corresponding ground truth annotations, we construct three sets of predictions to use as the “pre-correction” step during training and validation: “synthetic” which applies a corruption function to ground truth, and “Molmo-low” and “Molmo-high” which come from a model’s tracking predictions on the clip.

## 4.2 Trajectory generation pipeline

We build a set of synthetic correction trajectories with CFC videos and annotations as source material. We generate multiple correction trajectories per video, each consisting of: (1) an initial set of imperfect tracks, (2) a natural language correction prompt that addresses one or more tracking errors from (1), and (3) the intended, post-correction set of tracks.

Classifying tracking mistakes. We construct artificially corrupted correction trajectories, inspired by common failure modes in multi-object tracking. Previous work on evaluating multi-object trackers classifies tracking errors into five types: false negatives, false positives, fragmentations, mergers, and deviations [27]. We build upon these corruption primitives to construct a set of possible corruptions enumerated in Table 2, including more complex patterns that are constructed from the sequential application of multiple primitives.

Synthetic corrections. From each clip’s ground truth tracks we apply corruptions programmatically to arrive at the initial, step-1 tracks. The model tries to correct all corruptions and recover the ground truth tracks in step 2.

Targeted corrections. We also include a set of targeted corrections in the set of synthetic corrections. For this set we write prompts asking to correct only the n most recent corruptions of total N corruptions applied to the ground truth tracks. The post-correction tracks the model learns to generate are thus the ground truth tracks with only the first N − n corruptions applied. We also compute HOTA against that flawed ground truth. To perfectly execute a targeted correction, the model must pay attention to the user prompt: it cannot just rely on learning to correct as well as possible in response to every video.

Prompt variety. A human user desiring a specific correction to an incorrect track prediction may issue a variety of prompts to Molmo2Fish, difering in style and information content. We thus include prompts with a range of information content.

Table 2: All possible corruptions to apply to ground truth tracks (possibly in combination) to generate the trajectories in synthetic correction dataset.
<table><tr><td>Corruption</td><td>Function</td></tr><tr><td>add</td><td>Adds a spurious track</td></tr><tr><td>delete</td><td>Removes one track</td></tr><tr><td>fragment</td><td>Splits a track into two, assigning one track ID before a chosen frame and another after</td></tr><tr><td>merge</td><td>Picks two tracks to assign to the same track ID</td></tr><tr><td>deviate</td><td>Applies a spatial displacement</td></tr><tr><td>partial_delete</td><td>Removes a portion of a track</td></tr><tr><td>duplicate</td><td>Adds a spurious track close to a track</td></tr><tr><td>drift</td><td>Adds a directional shift to a track</td></tr><tr><td>jitter</td><td>Adds random noise to a track</td></tr><tr><td>stall</td><td>After some frame, track stays still</td></tr><tr><td>id_switch</td><td>Two tracks are covered by one (one track moves from one fish to another)</td></tr><tr><td>split_gap</td><td>One track is split into two, with some intermediate missing coverage</td></tr><tr><td>swap</td><td>Two tracks have their IDs swapped at a certain frame</td></tr><tr><td>light_subsample delete_most</td><td>Reduces up to 50% of annotated points per track Deletes all but 1-2 ground truth tracks per video, leaving them</td></tr><tr><td></td><td>untouched</td></tr><tr><td>subsample</td><td>Reduces annotated points per track, up to retaining only 2 points per track</td></tr></table>

No-info prompts read as “Fix all mistakes in these tracks, if any exist.” The correct output may be a duplicate of the pre-correction tracks. Wrongonly prompts read as “Fix all mistakes in these tracks.” Vague prompts (≤ 20 words) specify what corrections to make but don’t include timestamps or exact counts of dense fish clusters, and sometimes don’t refer precisely to track IDs. Full prompts (≤ 40 words) fully specify what corrections to make, including timestamps, exact counts of dense fish clusters, and track IDs. In practice, we find a diference of < 1% across the board in final HOTA between no-info and wrong-only evaluations, and between vague and full evaluations, so for concision we omit all results on wrong-only and vague prompts throughout the paper.

![](images/cbc82c70bf8e3af1aa64103c5c53c5e67899a4da67380fe8b79c379df9abea2f.jpg)  
Fig. 4: An illustration of the add, delete, fragment, merge, and deviate corruptions, with line color corresponding to predicted track ID.

Writing correction prompts. Multiple sets of corruptions are applied to a video’s ground truth tracks to construct trajectories of varying correction difficulty and complexity. To generate the correction prompt for a trajectory, an LLM (Opus 4.5) receives a record of which corruptions were applied in what order and the sorted track IDs computed before and after each corruption.

Correcting Molmo’s predictions. To bolster Molmo2Fish’s self-correction ability, we additionally include a set of correction trajectories whose pre-correction steps are Molmo2’s predictions on the video. We fine-tune a version of Molmo2 only on the tracking task and not on the correction task and save its predictions on the entire CFC dataset to use as the pre-correction state. The step110 checkpoint, with lower performance, produces the “Molmo-low” set; the step300 checkpoint, with higher performance, produces the “Molmo-high” set.

We provision an Opus 4.5 agent to generate natural language correction texts for each case. We manually reviewed the subset of the resulting prompts corresponding to the most structurally complex videos to edit prompts for accuracy. See Appendix D for more detail on all data generation steps.

## 5 Experimental setup

Training configuration. For the LLM to finish generating a correction, its context window must fit the video, initial prompt, initial prediction, user correction prompt, and final prediction. For CFC clips which are too long to fit, we slice them into parts so they fit fully in the context (see the Sliced clips row of Table 6), in addition to downsampling frames by a factor of 3 when fed into the model so the video can span a greater temporal extent.

We perform rank 64 LoRA [20] finetuning on all components with a learning rate of 0.001. We LoRA finetune as opposed to fully finetuning because of computational limitations. We train all models for 300 steps with a batch size of 64 and a maximum sequence length of 16,384 tokens. The model is trained with cross-entropy loss on its final response only (i.e. the output step it predicts, either step 1 for the pure tracking task or step 2 for the track correction task). Evaluation metrics. We evaluate multi-object tracking performance using Higher Order Tracking Accuracy (HOTA) [27] which measures both detection accuracy and long-range association accuracy over all tracks in a video (see Appendix Sec. E for more details). To accommodate the diferences between Molmo2’s tracking outputs (points) and CFC’s annotation format (bounding boxes), we modify HOTA such that a true positive always refers to a point landing in its ground truth box; we thus do not average over any localization thresholds as in the original implementation.

Evaluation protocol. To understand Molmo2Fish’s correction ability we employ the following protocol. From multiple sets of initial predictions produced by various models run on the entire CFC validation set of videos, we generate correction prompts (of all four full, vague, wrong-only, and no-info varieties), as described in Section 4, and run a single correction step on each video. We then compare the step-1 (pre-correction) HOTA averaged over the whole validation set and the step-2 (post-correction) average HOTA to measure the change in tracking performance resulting from the correction step.

We evaluate Molmo2Fish on three sets of initial (step-1) predictions, also described in Tab. 1: (1) YOLO+SORT predictions. The CFC authors built a lightweight two-stage object detection and tracking pipeline using the object detector YOLO [9] and of-the-shelf tracker SORT [5]. We follow their procedure but train on our training set as described in Sec. 4.1. We also evaluate corrections to step-1 tracks generated by Molmo2Fish in response to the prompt “Track all fish”, (2) Molmo-low predictions from an early-checkpoint Molmo2Fish model trained only on tracking and (3) Molmo-high predictions from the final tracking-only checkpoint.

![](images/033194affc8f3e246ef34d2a022c7f407d6815f1ba5b56cf4cb8c4292217358e.jpg)  
Fig. 5: HOTA of pre-correction (dashed lines) and post-correction (bars) predictions on each river. Pre-correction predictions are generated by Molmo and YOLO+SORT respectively; post-correction predictions are generated by Molmo2Fish from either the Molmo-high or YOLO+SORT baseline.

## 6 Results

Correcting model-generated predictions. When evaluated with the above single-step correction protocol, Molmo2Fish’s post-correction prediction improves from the YOLO+SORT prediction baseline on three out of four test locations (see Fig. 5, left three panels). This is true regardless of whether prompts provide specific corruption-correction information or not—both no-info and full prompts lead to tracking improvements. On the other hand, when starting from Molmo-high (i.e., the strongest Molmo-generated prediction set), improvements are less consistent and generally more modest. This indicates that while Molmo2Fish succeeds at correcting predictions that difer from its own, it struggles to further improve its own best predictions beyond their initial accuracy even with targeted prompting. We also observe that Molmo2Fish struggles both pre- and post-correction on videos that contain many fish, e.g. at the Nushagak location where tracks are dense. See Appendix A for examples.

Ablating fine-tuning strategies. We ablate parameters fine-tuned in Tab. 3 and data included in training in Tab. 4. In our parameter fine-tuning ablation, all versions of the model are trained on the same data mixture of pure tracking and synthetic, targeted, and Molmo-low corrections. Fine-tuning more components of the model improves performance significantly across all tasks except on Molmohigh corrections, which all versions of the model perform similarly at. Some fine-tuning is necessary to make Molmo2 useful for the CFC data and on the track correction task: its out-of-the-box performance (the leftmost column in Tab. 3) is low on both evaluations. In our training data ablation, all models have LLM, connector, and ViT LoRA fine-tuned, for the same number of steps. Again, some fine-tuning on the track correction task is necessary to make the model useful for that task: the model trained on pure tracking only (the leftmost column in Tab. 4) performs poorly on the synthetic correction task even though it leads at pure tracking. Only the models trained on Molmo-low (rightmost two columns in Tab. 4) meaningfully correct both the synthetic and Molmo-low precorrection baselines. Molmo-high, which starts from a baseline closer to ground truth, is more dificult for all models to correct, even the model in the rightmost column of Tab. 4 which has seen Molmo-high trajectories during training.

Table 3: Fine-tuning ablation. HOTA on each validation set for Molmo2 out-ofthe-box (no fine-tuning) and Molmo2 trained by fine-tuning the LLM only, LLM and connector only, and all components. For all correction tasks, the pre-correction HOTA of the step-1, corrupted tracks is listed in gray next to the task name, and the reported score is HOTA on the entire validation set after a single correction step on each video.
<table><tr><td></td><td>Molmo2</td><td>LLM</td><td>LLM+Connector</td><td>LLM+Connector+ViT</td></tr><tr><td>Pure Tracking</td><td>0.049</td><td>0.608</td><td>0.656</td><td>0.737</td></tr><tr><td>Synthetic Corrections</td><td>0.484</td><td>0.484</td><td>0.484</td><td>0.484</td></tr><tr><td>No-info</td><td></td><td>0.757</td><td>0.769</td><td>0.799</td></tr><tr><td>Full</td><td>0.164</td><td>0.767</td><td>0.780</td><td>0.817</td></tr><tr><td>Targeted Corrections</td><td></td><td>0.843</td><td>0.843</td><td>0.843</td></tr><tr><td>No-info</td><td></td><td>0.623</td><td>0.632</td><td>0.640</td></tr><tr><td>Full</td><td></td><td>0.895</td><td>0.899</td><td>0.900</td></tr><tr><td>Molmo-low Corrections</td><td></td><td>0.482</td><td>0.482</td><td>0.482</td></tr><tr><td>No-info</td><td></td><td>0.514</td><td>0.528</td><td>0.615</td></tr><tr><td>Full</td><td></td><td>0.548</td><td>0.580</td><td>0.669</td></tr><tr><td>Molmo-high Corrections</td><td></td><td>0.737</td><td>0.737</td><td>0.737</td></tr><tr><td>No-info</td><td></td><td>0.739</td><td>0.741</td><td>0.737</td></tr><tr><td>Full</td><td></td><td>0.740</td><td>0.749</td><td>0.747</td></tr></table>

Molmo2Fish attends to the user’s correction prompt. “Targeted” corrections are evaluated against the partially corrected corrupted tracks and not against the fully correct ground truth, as described in Sec. 4.2. The correction prompt must specify which corrections to make, and Molmo2Fish must listen, to make such corrections successfully. As expected, and visible in Fig. 6, Molmo2Fish consistently improves over the pre-correction baseline when provided with a fully detailed prompt, and consistently degrades the tracks relative to the target when provided with a generic prompt of the form “Fix all tracks”, which leads it to fix mistakes that are left in the targeted ground truth (“Targeted Corrections” row of Tab. 3).

More information doesn’t consistently help. The rightmost column of Tab. 3 shows Molmo2Fish improves significantly over the pre-correction baseline for synthetic corrections, but incorporating prompts with full detail only modestly improves its predictions. Fully detailed prompts significantly improve Molmo2Fish’s performance on Molmo-low, but in that case, the corrected tracks still lag behind the model’s one-shot tracking performance. On Molmohigh, Molmo2Fish barely improves over the pre-correction baseline and is helped only slightly by detailed prompts. Similarly, in Fig. 5, when correcting from the YOLO+SORT baseline (rightmost group of bars in each chart), Molmo2Fish performs only marginally better given detailed prompts (“full” bar) vs. without any information (“no-info” bar).

Table 4: Training dataset ablation. HOTA after LoRA fine-tuning all model components for 300 steps on each dataset mixture, for a model trained only on the tracking task; only on tracking and synthetic corrections; on tracking, synthetic corrections, and the Molmo-low set; and on all training sets. (We include an extra rightmost column reflecting a training mixture containing all sets showing that performance can be further improved by training for more steps, but opted to compare all other models at 300 steps.) For all correction tasks, the pre-correction HOTA of the step-1, corrupted tracks is listed in gray next to the task name, and the reported score is HOTA on the entire validation set after a single correction step on each video. Only the models that saw Molmo-low data during training make appreciable corrections to Molmo-low trajectories. Despite training on Molmo-high, the rightmost columns’ models still fail to make significant corrections to Molmo-high trajectories.
<table><tr><td></td><td></td><td></td><td></td><td>Pure tracking + synthetic + Molmo-low + Molmo-high | +120 steps</td><td></td></tr><tr><td>Pure Tracking</td><td>0.780</td><td>0.763</td><td>0.737</td><td>0.729</td><td>0.793</td></tr><tr><td>Synthetic Corrections</td><td>0.484</td><td>0.484</td><td>0.484</td><td>0.484</td><td>0.484</td></tr><tr><td>No-info</td><td></td><td>0.820</td><td>0.799</td><td>0.766</td><td>0.803</td></tr><tr><td>Full</td><td>0.415</td><td>0.833</td><td>0.817</td><td>0.811</td><td>0.848</td></tr><tr><td>Molmo-low Corrections</td><td></td><td>0.482</td><td>0.482</td><td>0.482</td><td>0.482</td></tr><tr><td>No-info</td><td></td><td>0.473</td><td>0.615</td><td>0.567</td><td>0.576</td></tr><tr><td>Full</td><td></td><td>0.500</td><td>0.669</td><td>0.634</td><td>0.694</td></tr><tr><td>Molmo-high Corrections</td><td></td><td>0.737</td><td>0.737</td><td>0.737</td><td>0.737</td></tr><tr><td>No-info</td><td></td><td>0.740</td><td>0.737</td><td>0.744</td><td>0.743</td></tr><tr><td>Full</td><td></td><td>0.755</td><td>0.747</td><td>0.754</td><td>0.767</td></tr></table>

## 7 Discussion and conclusions

We successfully adapted Molmo2 to tracking in a new visual domain and to a new mode of interaction: generating better tracks based on flawed initial predictions. We found that LoRA finetuning is suficient to introduce these capabilities to Molmo2, which is unable to complete these tasks zero-shot, and that training on our data increases tracking accuracy on the CFC dataset from just 5% to 74%. Further, building on a model that initially has no notion of referring to its previous output, we showed it is possible to add track correction capabilities using our data generation and fine-tuning framework. Molmo2Fish is able to perform targeted corrections, and is also a strong general-purpose track correction model, often improving predictions even without specific guidance.

However, there are limitations and room for improvement. Molmo2Fish is sensitive to distribution shift between the set of corruptions and correction prompts seen during training and at test-time. Further, while it is promising that Molmo2Fish’s initial tracking performance is so high, we observed that additional natural language feedback often does not help the model to substantially improve upon its own initial performance. Simultaneously, it’s clear that Molmo2Fish pays attention to the user’s correction prompt: it successfully makes targeted corrections when provided with a prompt describing the specific mistakes to fix and fails without a prompt, as it should.

To interpret Molmo2Fish’s behavior on non-“targeted” track corrections, we can consider the correction trajectories as existing across a dificulty spectrum. For the easiest trajectories, whether Molmo2Fish is provided a detailed prompt or not, it is able to make the desired correction. For the hardest trajectories, whether Molmo2Fish is provided a detailed prompt or not, it is unable to make the desired correction. Only for trajectories occupying an intermediate range on this dificulty spectrum does Molmo2Fish’s behavior change depending on the presence of a detailed prompt. Empirically, it seems that the prevalence of examples occupying this intermediate range in the dataset is quite low, so that in our evaluation setup, the improvement to aggregate HOTA over the whole dataset is just a few percent at most. The project of training a more capable track correction model would endeavor to understand the dynamics within and/or increase the ceiling on this intermediate range: a model which could more intelligently use language and video in tandem may be able to leverage language priors to improve on those more dificult predictions.

![](images/bb5807df86742d7fec46c2c5c2454e29e029335f911c1385f13ff043046c4b0c.jpg)  
Fig. 6: HOTA before and after correction, with no-info prompts (top row) and full prompts (bottom row), for each subset of the track correction evaluation data. Each dot represents a single trajectory. For the targeted track correction task (where HOTA of 1.0 corresponds to the partially-corrupted ground truth as described in Sec. 4.2), a generic “Fix all tracks” prompt results in a large aggregate drop in HOTA (top row), while Molmo2Fish makes the targeted correction successfully when the prompt specifies what to fix (bottom row). In other tasks, the change between no-info and full corrections is subtler.

On the other hand, for the targeted track correction task, regardless of the dificulty of correcting the initial prediction, Molmo2Fish must attend to the user prompt to understand which corrections to apply or not, so in aggregate, a much larger improvement is seen between the no-info and full evaluations.

In conclusion, Molmo2Fish is a model that allows for interactive track correction in the form of a conversation with an MLLM. We investigated what works, what doesn’t, and how future development might bring us closer to a model that can help users interact with and modify tracking predictions on video in a flexible manner, reducing the efort needed to develop and test ecological workflows on tracks in video.

## Acknowledgements

Thanks to Michael Hobley, Suzanne Stathatos, Madison Van Horn, Sevan Brodjian, and Erik Young for their insights and feedback throughout.

## References

1. Atlas, W.I., Ma, S., Chou, Y.C., Connors, K., Scurfield, D., Nam, B., Ma, X., Cleveland, M., Doire, J., Moore, J.W., et al.: Wild salmon enumeration and monitoring using deep learning empowered detection and tracking. Frontiers in Marine Science 10, 1200408 (2023)

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report (2025), https://arxiv.org/abs/2511.21631

3. Beery, S., Morris, D., Yang, S.: Eficient pipeline for camera trap image review (2019)

4. Bekuzarov, M., Bermudez, A., Lee, J.Y., Li, H.: XMem++: Production-level video segmentation from few annotated frames (Jul 2023)

5. Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B.: Simple online and realtime tracking. In: 2016 IEEE International Conference on Image Processing (ICIP). p. 3464–3468. IEEE (2016). https://doi.org/10.1109/icip.2016.7533003, http: //dx.doi.org/10.1109/ICIP.2016.7533003

6. Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., Wang, J., Monteiro, M., Xu, H., Dong, S., Ravi, N., Li, D., Dollár, P., Feichtenhofer, C.: Perception encoder: The best visual embeddings are not at the output of the network. arXiv:2504.13181 (2025)

7. Bondi, E., Jain, R., Aggrawal, P., Anand, S., Hannaford, R., Kapoor, A., Piavis, J., Shah, S., Joppa, L., Dilkina, B., Tambe, M.: BIRDSAI: A dataset for detection and tracking in aerial thermal infrared videos. In: 2020 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE (Mar 2020)

8. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: SAM 3: Segment anything with concepts (Nov 2025)

9. Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., Shan, Y.: Yolo-world: Real-time open-vocabulary object detection (2024), https://arxiv.org/abs/2401.17270

10. Cho, J.H., Madotto, A., Mavroudi, E., Afouras, T., Nagarajan, T., Maaz, M., Song, Y., Ma, T., Hu, S., Rasheed, H., Sun, P., Huang, P.Y., Bolya, D., Jain, S., Martin, M., Wang, H., Ravi, N., Jain, S., Stark, T., Moon, S., Damavandi, B., Lee, V., Westbury, A., Khan, S., Krähenbühl, P., Dollár, P., Torresani, L., Grauman, K., Feichtenhofer, C.: Perceptionlm: Open-access data and models for detailed visual understanding. arXiv:2504.13180 (2025)

11. Clark, C., Zhang, J., Ma, Z., Park, J.S., Salehi, M., Tripathi, R., Lee, S., Ren, Z., Kim, C.D., Yang, Y., Shao, V., Yang, Y., Huang, W., Gao, Z., Anderson, T., Zhang, J., Jain, J., Stoica, G., Han, W., Farhadi, A., Krishna, R.: Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611 (2026)

12. Dat, N.N., Richardson, T., Watson, M., Meier, K., Kline, J., Reid, S., Maalouf, G., Hine, D., Mirmehdi, M., Burghardt, T.: WildLive: Near real-time visual wildlife tracking onboard UAVs (Apr 2025)

13. Delatolas, T., Kalogeiton, V., Papadopoulos, D.P.: Learning the what and how of annotation in video object segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2024)

14. Gabef, V., Qi, H., Flaherty, B., Sumbül, G., Mathis, A., Tuia, D.: MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the swiss alps (Mar 2025)

15. Gadot, T., Istrate, S., Kim, H., Morris, D., Beery, S., Birch, T., Ahumada, J.: To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images. IET Computer Vision 18(8), 1193– 1208 (2024). https://doi.org/https://doi.org/10.1049/cvi2.12318, https: //ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cvi2.12318

16. Girit, U., Hariharan, K., Shor, E.: RAMURE. https : / / github . com / fulcrumresearch/ramure (2026)

17. Google DeepMind: Gemini 3 pro model card (Nov 2025), https://storage. googleapis.com/deepmind- media/Model- Cards/Gemini- 3- Pro- Model- Card. pdf, version 2

18. Haalck, L., Mangan, M., Wystrach, A., Clement, L., Webb, B., Risse, B.: CATER: Combined animal tracking & environment reconstruction. Sci. Adv. 9(16), eadg2094 (Apr 2023)

19. Hernandez, A., Miao, Z., Vargas, L., Beery, S., Dodhia, R., Lavista, J.: Pytorchwildlife: A collaborative deep learning framework for conservation (2024)

20. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models (2021), https://arxiv. org/abs/2106.09685

21. Kay, J., Haucke, T., Stathatos, S., Deng, S., Young, E., Perona, P., Beery, S., Van Horn, G.: Align and distill: Unifying and improving domain adaptive object detection. Transactions on Machine Learning Research (2025), https://openreview. net/forum?id=ssXSrZ94sR, featured Certification

22. Kay, J., Horn, G.V., Maji, S., Sheldon, D., Beery, S.: Consensus-driven active model selection (2025), https://arxiv.org/abs/2507.23771

23. Kay, J., Kulits, P., Stathatos, S., Deng, S., Young, E., Beery, S., Van Horn, G., Perona, P.: The caltech fish counting dataset: A benchmark for multiple-object tracking and counting (Jul 2022)

24. Kay, J., Stathatos, S., Deng, S., Young, E., Perona, P., Beery, S., Van Horn, G.: Unsupervised domain adaptation in the real world: A case study in sonar video. In: NeurIPS 2023 Computational Sustainability: Promises and Pitfalls from Theory to Deployment (2023)

25. Kurinchi-Vendhan, R., Beery, S.: Finding needles in the haystack: Transductive active labeling in ecology (2026), https://arxiv.org/abs/2606.03821

26. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., Berg, A.C.: SSD: Single Shot MultiBox Detector, p. 21–37. Springer International Publishing (2016). https://doi.org/10.1007/978-3-319-46448-0\_2, http://dx.doi.org/ 10.1007/978-3-319-46448-0\_2

27. Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., Leibe, B.: Hota: A higher order metric for evaluating multi-object tracking. International Journal of Computer Vision pp. 1–31 (2020)

28. May, G., Dalsasso, E., Delplanque, A., Kellenberger, B., Tuia, D.: How to minimize the annotation efort in aerial wildlife surveys. Ecological Informatics 91, 103387 (2025). https://doi.org/https://doi.org/10.1016/j.ecoinf.2025.103387, https://www.sciencedirect.com/science/article/pii/S1574954125003966

29. Mededovic, E., Laurentius, V., Wu, Y., Kopaczka, M., Chen, Z., Schulz, M., Tolba, R., Stegmaier, J.: No free lunch in annotation either: An objective evaluation of foundation models for streamlining annotation in animal tracking (Feb 2025)

30. Naik, H., Yang, J., Das, D., Crofoot, M.C., Rathore, A., Sridhar, V.H.: Buck-Tales : A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes (Nov 2024)

31. Pedersen, M., Haurum, J.B., Bengtson, S.H., Moeslund, T.B.: 3D-ZeF: A 3D zebrafish tracking benchmark dataset (Jun 2020)

32. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks (2016), https://arxiv.org/abs/1506.01497

33. Sun, Z.W., Hua, Z.X., Li, H.C., Qi, Z.P., Li, X., Li, Y., Zhang, J.C.: FBD-SV-2024: Flying bird object detection dataset in surveillance video (Aug 2024)

34. Wasmuht, D.F., Brookes, O., Schall, M., Palencia, P., Beirne, C., Burghardt, T., Mirmehdi, M., Kühl, H., Arandjelovic, M., Pottie, S., Bermant, P., Asheim, B., Toh, Y.J., Elzinga, A., Holmberg, J., Whitworth, A., Flatt, E., Gustafson, L., Ryali, C., Hu, Y.T., Guo, B., Westbury, A., Saenko, K., Suris, D.: The SA-FARI dataset: Segment anything in footage of animals for recognition and identification (Nov 2025)

35. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., Wang, X.: Bytetrack: Multi-object tracking by associating every detection box (2022), https://arxiv.org/abs/2110.06864

36. Zrnic, T., Candès, E.J.: Active statistical inference. arXiv preprint arXiv:2403.03208 (2024)

## A Molmo2Fish prediction examples

Here we provide examples of model behavior on both the tracking task (given just the frames of a video, output tracks for all fish) and the correction task (fill in the last step of a correction trajectory). We describe Molmo2Fish’s strengths and weaknesses, including direct comparisons to the YOLO+SORT traditional two-stage object detection and tracking method that we compare against throughout this work.

## A.1 Tracking examples

We show examples of Molmo2Fish’s tracking predictions next to YOLO+SORT’s predictions in Fig. 7 and share general observations on qualitative trends in its predictions compared to those output by YOLO+SORT.

Molmo2Fish’s baseline performance is much higher on the Elwha river than YOLO+SORT. The fish in the Elwha river appear small and are often occluded by noise and sonar reflections. On the Elwha, an understanding of the motion of the fish over the entire clip is helpful for localizing it on any given frame. While YOLO tends to miss detections on individual frames where the fish is less visible, leading SORT to fragment a single fish into multiple tracks, Molmo2Fish learns what a coherent fish track looks like over the course of a video and maintains the same track when appropriate. In addition, the Elwha is relatively sparse—not many fish swim through the sonar camera’s field of view at the same time— avoiding a weakness of Molmo2Fish, which struggles with keeping track of many fish.

On the other hand, Molmo2Fish performs much worse than YOLO+SORT on the Nushagak, where fish appear large and easy to localize, but many fish swim through at the same time. In fact Molmo2 was not trained on scenes with above 40 tracks, and the Nushagak clips routinely bump up against that upper limit, with 30 to 40 fish swimming through in a single clip. While the YOLO+SORT algorithm is equally robust regardless of clip length or total number of tracks, Molmo2Fish gets confused once the track count becomes large, forgetting to drop tracks once they exit the frame or extrapolating tracks horizontally.

## A.2 Correction examples

In Fig. 8 and Fig. 9 we show correction examples where Molmo2Fish makes a better correction from step zero (top row of each group) when given a fully detailed prompt (right column) than when given a generic prompt (left column). In all cases, the target tracks are the ground truth tracks, shown in the bottom row of each group. When the prompt helps, it’s often by helping Molmo2Fish make judgment calls more aligned with the human annotator’s judgment.

We also show examples in Fig. 10 and Fig. 11 of the fully detailed prompt hurting Molmo2Fish’s correction performance. These cases are usually explained by the prompt neglecting to mention some flaw in the initial tracks that Molmo2Fish fixes when told to “fix any mistakes” but does not fix when specific corrections are prompted that don’t include that initial flaw. For example, when correcting from the YOLO+SORT baseline, it often happens that the “full” prompt does not capture the intermittency of the YOLO+SORT tracks.

![](images/ab60f2d8735b7b4b7fe9d52be7fb00636da4fbda9f3257a704812f6c85918b8a.jpg)

![](images/22ef1ee4ff7ae0e0bc55aa415e606dd9a3f3f9724e4d873c921da0514e0462ec.jpg)  
Fig. 7: Examples of Molmo2Fish’s tracking predictions (middle row) compared against YOLO+SORT’s tracking predictions (top row) on two clips, from the Rightbank (left group) and Nushagak (right group) locations. Left: Molmo2Fish successfully carries some tracks through the whole video that YOLO+SORT drops on some frames, but Molmo2Fish’s predicted track on the topmost fish stalls and stops following it, while YOLO+SORT’s prediction doesn’t. Right: Molmo2Fish successfully localizes fish on more frames for more of the video but there are so many fish that it “loses track” and begins to extrapolate tracks horizontally rather than ending them.

![](images/0e19e35c18946fd05c38d7ba7cd86ec3e1393f18297f8697fb83f6238e1a15eb.jpg)

![](images/79ad3b208403902c7c870528ca97f38abe5f8919d7a4da23cf2c3918d56c3dec.jpg)  
Fig. 8: Example of Molmo2Fish making a better correction (second row) from the same pre-correction step (top row) when provided with a detailed prompt (right) vs when provided with a generic prompt (left). The ground truth tracks are displayed on the same frame sequence in the bottom row for comparison. Here, Molmo2Fish recognizes the upper left blob as a fish unless explicitly told by the user that it is a false detection.  
Fig. 9: Similar to Fig. 8, an example where Molmo2Fish makes a better correction, as measured by HOTA, when provided with a detailed prompt vs without. Molmo2Fish thinks the mistake is that the top fish (blue) is a false detection: actually the mistake pointed out by the user is that the first track (red) must be present longer.

![](images/43445654abc4e2c62724b4f1cc6bb8e0a7992df0edfd7467ffd8aa528f577a8c.jpg)  
Fig. 10: Example of Molmo2Fish making a worse correction when provided with a detailed prompt vs without. Since Molmo2Fish can’t recognize the presence of the dificult-to-localize fish that appears early in the video, it merely refrains from making a correction at all when prompted to “Adjust the tracks if anything seems incorrect”, whereas it puts a point down randomly (and incorrectly) when told that an additional fish does exist in that region.

![](images/037374b15c59fa2262befff6df7b2338c564dd6311a6fd1d4d3491b58d2005bd.jpg)

![](images/eaa26f7e76f19976d90800765913cf4ed98176ad7ddea0b9dbb35f69a7ced862.jpg)  
Fig. 11: Example of Molmo2Fish making a worse correction when provided with a detailed prompt vs without. Here Molmo2Fish is diverted from its initially correct prediction of the first fish’s track by the user prompt.

## B Additional experiments

## B.1 Generalization challenges

We investigate the ability of Molmo2Fish to generalize to corruption types not seen during training in Tab. 5. We hold out a set of corruption types (split\_gap, swap, and duplicate) during training on only synthetic trajectories and measure Molmo2Fish’s performance on a set of test clips including those held-out corruptions. Holding out swap means that during training the model never sees correction trajectories that undo swap, regardless of whether swap is explicitly named in the prompt. We call trajectories containing these corruptions the “OOD set” and trajectories not containing these corruptions the “ID set”.

We see that distribution shift in corruptions/corrections causes poor performance. The two models perform similarly when evaluated on the ID set. But evaluated on the OOD set (top row group of Tab. 5), with or without language guidance, the model trained only on the ID set (left column) lags behind the model that saw all corruptions during training (right column).

Table 5: HOTA on evaluation of models trained on pure tracking and either a) ID set, only synthetic trajectories not including held-out corruptions or b) all synthetic trajectories. Both models perform similarly evaluated on the ID set, which contains corruptions both saw during training. A large performance gap exists between the two models when evaluated on the OOD set, which is not recovered by natural language guidance.
<table><tr><td colspan="2"></td><td>Trained on ID only</td><td>Trained on all</td></tr><tr><td rowspan="3">OOD set</td><td>Pre-correction baseline</td><td>0.628</td><td>0.628</td></tr><tr><td>No-info</td><td>0.810</td><td>0.856</td></tr><tr><td>Full</td><td>0.824</td><td>0.882</td></tr><tr><td rowspan="3">ID set</td><td>Pre-correction baseline</td><td>0.432</td><td>0.432</td></tr><tr><td>No-info</td><td>0.805</td><td>0.806</td></tr><tr><td>Full</td><td>0.823</td><td>0.815</td></tr></table>

## C Dataset details

Table 6: Full dataset statistics. Split by location, for the train and validation sets. “Clips” refers to number of clips in original CFC dataset: we randomly allocate clips to the train or validation set before slicing to fit in Molmo2Fish’s context window.
<table><tr><td colspan="2"></td><td>Kenai</td><td>Rightbank</td><td>Elwha</td><td>Nushagak</td></tr><tr><td colspan="6">Train</td></tr><tr><td colspan="2">Clips</td><td>482</td><td>512</td><td>175</td><td>55</td></tr><tr><td colspan="2">Sliced clips</td><td>858</td><td>844</td><td>273</td><td>154</td></tr><tr><td>Examples</td><td>Pure tracking</td><td>858</td><td>844</td><td>273</td><td>154</td></tr><tr><td></td><td>Guided tracking</td><td>2048</td><td>2130</td><td>492</td><td>449</td></tr><tr><td></td><td>Synthetic corrections</td><td>2104</td><td>2116</td><td>404</td><td>602</td></tr><tr><td></td><td>Molmo-low corrections</td><td>856</td><td>844</td><td>273</td><td>142</td></tr><tr><td></td><td>Molmo-high corrections</td><td>858</td><td>843</td><td>273</td><td>154</td></tr><tr><td></td><td>Targeted corrections</td><td>788</td><td>646</td><td>144</td><td>72</td></tr><tr><td colspan="6">Validation</td></tr><tr><td colspan="2">Clips</td><td>64</td><td>144</td><td>48</td><td>17</td></tr><tr><td colspan="2">Sliced clips</td><td>159</td><td>144</td><td>69</td><td>70</td></tr><tr><td>Examples</td><td>Pure tracking</td><td>159</td><td>144</td><td>69</td><td>70</td></tr><tr><td></td><td>Guided tracking</td><td>539</td><td>567</td><td>153</td><td>240</td></tr><tr><td></td><td>Synthetic corrections</td><td>435</td><td>372</td><td>93</td><td>280</td></tr><tr><td></td><td>Molmo-low corrections</td><td>159</td><td>144</td><td>69</td><td></td></tr><tr><td></td><td></td><td>159</td><td>144</td><td>69</td><td>67</td></tr><tr><td></td><td>Molmo-high corrections Targeted corrections</td><td>158</td><td>144</td><td>68</td><td>69 70</td></tr></table>

## D Data generation process

Below we describe the process of generating correction prompts for both the real and synthetic correction datasets. We use the open-source software ramure [16] to provision LLMs, to expose tools to them, and to collect their responses in a reproducible, programmatic fashion.

## D.1 Real corrections

To generate the pre-correction step for the trajectories making up the “real” correction dataset, we collect predictions on the training set and validation set of a version of Molmo2 fine-tuned only on fish tracking at each of step 110 and step 300 of its training. We refer to these sets as ‘Molmo-low’ and ‘Molmo-high’ respectively throughout the paper. Corrections are written for each trajectory by an LLM which has access to:

– the incorrect tracks

– an initial attempt (deterministic Hungarian matching) at a track assignment between each predicted track and a ground truth track

– the ground truth tracks

– a set of tools (function calls) that can help it understand high-level track features

The LLM’s system prompt is shown below.

You review fish tracking predictions and write brief, casual correction feedback. Your job has two parts:

Part 1: Review Track Assignments

The deterministic matching algorithm has made initial assignments (predicted tracks → ground truth tracks). Review these — they may be wrong. Common issues to look for:

– Two predicted tracks may be FRAGMENTS of one fish (temporally sequential, spatially continuous → should be merged)

– A track marked "spurious/false positive" may actually be a fragment of a nearby matched track

– A track marked as matching one GT may actually correspond to a diferent GT

Look at timing, spatial position, and motion direction to judge whether assignments make sense.

Part 2: Write Correction Text

Write a single correction message (1-3 sentences max, casual tone).

– Be brief and casual (1–3 sentences max)

– Reference prediction track numbers (Track 1, Track 2, etc.)

– Use vague time references: “around 35s”, “near the end”, “early on”, “at the start”

– Use spatial descriptions for missing fish: “between tracks 3 and 4”, “in the center”, “near the top”

– For videos with many tracks or many errors: be MORE vague and holistic (e.g., “your tracks aren’t exiting at the right time”)

– For videos with few tracks and specific errors: be MORE specific (e.g., “Track 2 stays on the frame much longer than it should”)

– Acknowledge correct tracks briefly when some are wrong (e.g., “Track 3 is good”)

– Never use pixel coordinates or technical jargon

– Sound natural, like a human giving quick feedback

– Don’t start with “I” or “The model” — address the tracker directly with “you/your” or just describe what’s wrong

## Error patterns you may see:

– fragmented: model splits one fish into multiple sequential tracks (should be merged)

– stall: track freezes in place while fish keeps moving

– drift: track gradually moves away from the actual fish position

– ofset: track follows the fish’s motion but is consistently shifted away from it

– early\_exit: track ends too early (fish is still there)

– late\_exit: track stays too long (fish has already left)

– late\_entry: track starts too late (fish was already there)

– diverge: track suddenly jumps away from the fish

– spurious: a predicted track with no real fish there

– missing: a real fish that wasn’t tracked at all

## Examples of good correction text:

– “Track 2 stays on the frame much longer than it should. The fish exits around 35s”

– “Track 1 follows the fish correctly but it should exit around 40s. Track 2 follows the fish correctly at first but loses it around 37s, that fish keeps drifting upwards to the left and then downwards to the left.”

“All the tracks stall at some point. Tracks 1, 2, and 3 even start of stalled and don’t follow their respective fish as they’re moving.”

– “you missed a fish at the beginning in the bottom part of the video that exits out the left. tracks 1 and 2 should be merged. Tracks 3 and 4 should be merged, and are missing some coverage in the middle of the track”

– “you’re missing a fish swimming between tracks 3 and 4”

– “your tracks aren’t exiting at the right time - the fish are continuously exiting the frame out the left”

“you’re missing a fish in the center at the start of the clip. Both tracks 1 and 2 start drifting upwards of their actual fish and are present for too long. Track 3 is good.”

– “tracks 1 and 2 are drifting above the actual locations of the fish”

– “there are only 2 fish, not 3, and they start swimming downwards at around 37s, diverging from the predicted tracks”

## Output format

<confidence>high or low</confidence>

<reasoning>1-2 sentences on why you’re confident or not

--- what’s ambiguous?</reasoning>

<correction>your correction text here</correction>

Mark confidence as LOW when ANY of these apply:

– There are tracks marked “spurious” that overlap temporally/spatially with other tracks (could be fragments)

– Multiple short-lived tracks (< 10 frames) exist — they might be fragments of longer fish

The number of predicted tracks doesn’t match the number of GT tracks AND the discrepancy isn’t easily explained by a single missing/spurious track

– A track’s error could reasonably be described in multiple diferent ways

– You had to choose between two plausible interpretations of the track structure

Mark confidence as HIGH only when ALL of these apply:

– Track assignments are clearly one-to-one with no ambiguity

– Errors are simple and unambiguous (clear stall, clear late\_exit, obvious single missing fish)

– No short-lived tracks that could plausibly be fragments – Your correction would be the same regardless of how you interpret the ambiguous parts

We manually review all trajectories marked low confidence for accuracy. We observe empirically that most errors in captioning originate from model assignments between predicted and ground truth tracks that don’t align with how a human would make the assignments. As a simple example, consider a clip with just one fish and two predicted tracks. Depending on the tracks’ proximity to the ground truth fish track, the LLM may claim that track 2 is spurious while a human may claim that track 2 and track 1 are part of the same track, or vice versa. These are subjective judgments which we address by writing prompts ourselves, but the ambiguity

## D.2 Synthetic corrections

Each video gets four synthetic trajectories, with each trajectory resulting from a number of stacked corruptions that ranges depending on the number of fish in the video clip. For clips with two or fewer fish, all trajectories contain up to four corruptions at most. For clips with three or more fish, one trajectory contains 1-2 corruptions; one contains 3-4 corruptions; another 4-6 corruptions; another 7+ corruptions. “One corruption” is one application of one of the functions listed in Tab. 2.

A log is constructed which contains the order of corruptions applied, all precorruption (ground truth) track IDs, and all post-corruption (seen by the model)

track IDs. This is important so that the LLM correctly refers to the track IDs which are seen by the model and assigned deterministically.

Given this log and the same set of tools as in the real data generation pipeline in case the LLM wants to look at the actual tracks (e.g. to inform some phrasing), the LLM is asked to rephrase the log into a natural language correction prompt, following some guidelines. The full system prompt is reproduced below.

You turn a structured fish-tracking correction LOG into ONE natural-language correction a human reviewer would give. The log is an ORDERED list of every issue with the model’s tracks, joined by commas (and ‘then’ before the last). SUMMARISE the WHOLE list into a single coherent correction that covers EVERY item — do not drop any, do not invent any. Merge naturally where several items concern the same fish, and keep a sensible order.

Notation: most items name tracks as [step0\_id, step1\_id] — step0 is the track id the model currently shows (what the reviewer points at), step1 is the id it should become; None means the track is absent from that step. When two items share the same step1 id, they are the same fish.

Issue vocabulary:

– add track [X, None]: a spurious track on no real fish — should be removed.

– delete track [None, X]: a real fish the model missed — should be added.

– stall (mid): a track frozen in place mid-way while the fish keeps moving the frozen stretch should follow the fish (endpoints are unchanged; this is NOT an early/late issue).

’frozen in place past frame N ... trim its end back to frame N’ / ’frozen in place before frame N ... trim its start forward to frame N’: an EDGE stall — the track is stuck on a frozen box and OVERSHOOTS where the fish really is, so it must be SHORTENED (trimmed) to frame N. Do NOT say it stops early or should be extended — it runs too long/starts too early.

’starts late ... extend its start back to frame N’ / ’stops early ... extend its end to frame N’: frames were dropped from one end so the track is TOO SHORT and must be LENGTHENED (extended) to frame N. This is the OPPOSITE of an edge stall.

– jitter / drift: a track drifting of the fish.

– split / near-duplicate / split-with-gap: TWO tracks that are really one fish (should be merged into one track).

– split-into-N: N tracks (3 or 4) that are all really ONE fish, broken into consecutive fragments (should all be merged into one track).

‘missing some of its points’ / ‘make it continuous’: a track that exists but has gaps — some frames were dropped; it should be filled in to be continuous (NOT added from scratch, NOT merged).

– ‘trying to cover both tracks i and j’: one track that drifts from one fish onto another (an id switch) and should be two separate tracks.

– merge: two fish collapsed into one track (should be split apart).

– swap: two tracks’ ids are swapped.

‘sparse’ / ‘fill in the rest of the points’: a track that exists but is missing many of its points throughout — it should be densified, NOT added from scratch.

‘pick up the rest of the fish tracks’: most fish are untracked; track the remaining fish (sometimes naming the frames at which they pass by, and an exact count for a dense cluster).

STYLE — write like a clear human note:

– HARD CAP: 40 words maximum.

– Frame numbers from the log are FRAME NUMBERS — you MAY keep them (a later stage converts frames to seconds). Do NOT convert them yourself or invent times.

– If the log states an exact count for a dense cluster of fish, you MAY cite it.

– Never mention pixels or raw coordinates; say ‘upward’, ‘to the left’, etc.

– You have geometry tools (track positions, distances, frame snapshots). They are OPTIONAL — use them only if you want to refer to a fish by its position relative to other tracks (e.g. ‘the leftmost track’, ‘the fish below track 2’).

Call submit\_correction exactly once with the correction text. Do not say anything else.

We also inject prompts randomly (50-50) with one or the other of the following style note since we observe anecdotally that otherwise the LLM is biased heavily towards using descriptive phrasing.

– STYLE FOR THIS ONE: PRESCRIPTIVE — phrase it as an instruction/action (e.g. ‘add the track on the fish at the top’, ‘merge tracks 2 and 3’, ‘extend track 4 to the end’).

STYLE FOR THIS ONE: DESCRIPTIVE — phrase it as an observation of what’s wrong (e.g. ‘you missed a fish at the top’, ‘tracks 2 and 3 are the same fish’, ‘track 4 stops too early’).

## D.3 Targeted corrections

Targeted corrections are generated through the same pipeline as synthetic corrections except that the step-1 target is not the video ground truth (GT) but a corrupted version of the video GT. The LLM captioner additionally just receives the last n corruptions to write a prompt for.

## E HOTA

To calculate HOTA, first we arrive at a bijective matching of predicted detections to ground truth detections. True positives (TPs), false positives (FPs), and false negatives (FNs) are determined as usual from this matching.

HOTA’s novelty lies in formalizing true positive, false positive, and false negative associations (links between any two detections to form tracks) as well, and using these to calculate association accuracies that don’t depend on matching predicted and ground truth tracks to each other.

For each true positive pair of detections $c ,$ the set of TPAs (true positive associations) consists of those TPs with both the same predicted track ID (prID) and ground truth track ID (gtID) as $c ,$ or:

\tex {TPA}(c)=\{k },\ k in \{ tex {TP}|\tex {prID}(k)=\tex {prID}(c)\land \tex {gtID}(k)=\tex {gtID}(c)\

$$
k \in \{ \mathrm { T P } | \mathrm { p r I D } ( k ) = \mathrm { p r I D } ( c ) \land \mathrm { g t I D } ( k ) = \mathrm { g t I D } ( c ) \}\tag{1}
$$

Similarly,

$$
\begin{array} { r l r } & { \mathrm { F N A } ( c ) = \{ k \} , } & \\ & { \quad } & { \quad k \in \{ \mathrm { T P } | \mathrm { p r I D } ( k ) \neq \mathrm { p r I D } ( c ) \land \mathrm { g t I D } ( k ) = \mathrm { g t I D } ( c ) \} } \\ & { } & { \quad \cup \left\{ \mathrm { F N } | \mathrm { g t I D } ( k ) = \mathrm { g t I D } ( c ) \right\} } \end{array}\tag{2}
$$

\tex {FPA}(c)=\{k }, \in { tex TP}|\tex {prID}(k)=\tex {prID}(c)\land tex {g ID}(k)\neq t x {g ID}(c)\ up \{ tex FP}|\tex {prID}(k)=\tex {prID}(c)\

$$
\begin{array} { r } { k \in \{ \mathrm { T P } | \mathrm { p r I D } ( k ) = \mathrm { p r I D } ( c ) \wedge \mathrm { g t I D } ( k ) \neq \mathrm { g t I D } ( c ) \} \qquad } \\ { \cup \{ \mathrm { F P } | \mathrm { p r I D } ( k ) = \mathrm { p r I D } ( c ) \} } \end{array}\tag{3}
$$

Then HOTA is computed as:

$$
\mathrm { H O T A } = \sqrt { \frac { \sum _ { c \in \mathrm { T P } } \mathcal { A } ( c ) } { | \mathrm { T P } | + | \mathrm { F P } | + | \mathrm { F N } | } }\tag{4}
$$

$$
A ( c ) = \frac { \lvert \mathrm { T P A } ( c ) \rvert } { \lvert \mathrm { T P A } ( c ) \rvert + \lvert \mathrm { F P A } ( c ) \rvert + \lvert \mathrm { F N A } ( c ) \rvert }\tag{5}
$$

HOTA determines associations with respect to each true positive pair of detections (a correct match between a predicted and ground truth detection). In its original formulation, then, the final HOTA number comes from an average over localization thresholds of HOTA calculated at each threshold of the similarity score between a predicted and ground truth detection to count as a true positive.

For bounding box based methods, this similarity score might be IoU. But since Molmo2Fish predicts point tracks and CFC provides bounding box ground truth tracks, our similarity score is always either 1 or 0: a predicted point is matched to a ground truth bounding box if the point is inside the box and it’s assigned to that box via a Hungarian matching which optimizes HOTA. We do not average over any thresholds. For the YOLO+SORT results in this paper, we take the centroid of the YOLO-predicted bounding box as the “predicted point”. This means our HOTA metric is not directly comparable to HOTA calculated conventionally (based on bounding box IoU) on the same dataset.